"""Fine-tuning of the networks. One unit is one network trained on the training part of one fold: the validation part
selects the checkpoint, and the test part is scored once, after training. Every epoch prints the training loss, the
training accuracy (on the augmented training batches of that epoch), the validation accuracy and the validation QWK;
accuracies are means of the eight item accuracies. A finished unit is skipped when the notebook runs again, and its
stored epoch log is printed instead."""
import os
import copy
import json
import time
import zlib
import hashlib
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import data as bd
from .data import VALID, N_ITEMS, class_counts
from .metrics import mean_item_accuracy, mean_item_kappa
from .models import NETWORKS, MINUTES_PER_FOLD, build_network
from .scorers import ENSEMBLE, label
from .utils import atomic_save, save_json, load_json, save_csv

VERSION = '2.0.0'          # part of every unit signature
SEED = 2026
MAX_RESTARTS = 2
INFER_BATCH = 128
AUGMENT = {'shift': dict(rot=0.0, shift=0.04, blur=0.0),                  # translation by up to 4 % of the canvas
           'rotate_shift_blur': dict(rot=10.0, shift=0.10, blur=0.3)}     # rotation 10 degrees, translation 10 %, blur


def _torch():
    import torch, torch.nn.functional as F
    return torch, F


def _amp(device):
    torch, _ = _torch()
    cuda = device == 'cuda'
    bf16 = cuda and torch.cuda.is_bf16_supported()
    return dict(device_type='cuda', dtype=torch.bfloat16 if bf16 else torch.float16, enabled=cuda)


def _device(device=None):
    torch, _ = _torch()
    return device or ('cuda' if torch.cuda.is_available() else 'cpu')


def _gpu(D, device):
    """Inputs and labels kept on the training device for the whole run."""
    torch, _ = _torch()
    if getattr(D, '_dev', None) != device:
        D._t = SimpleNamespace(imgs=torch.from_numpy(D.imgs).to(device), item=torch.from_numpy(D.item).long().to(device),
                               score=torch.from_numpy(D.score).long().to(device), valid=torch.from_numpy(VALID).to(device))
        D._dev = device
    return D._t


def augment(ink, mode):
    """Training-time augmentation on the GPU. Applied to training batches only, after the split."""
    torch, F = _torch()
    a = AUGMENT[mode]
    B, dev = len(ink), ink.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * np.deg2rad(a['rot'])
    t = (torch.rand(B, 2, device=dev) * 2 - 1) * 2 * a['shift']            # normalised coordinates span 2
    theta = torch.stack([torch.stack([ang.cos(), -ang.sin(), t[:, 0]], 1),
                         torch.stack([ang.sin(), ang.cos(), t[:, 1]], 1)], 1)
    out = F.grid_sample(ink, F.affine_grid(theta, list(ink.shape), align_corners=False), mode='bilinear',
                        padding_mode='zeros', align_corners=False)
    out = out * (torch.rand(B, 1, 1, 1, device=dev) * 0.5 + 0.7)             # stroke darkness 0.7 to 1.2
    out = out + torch.randn_like(out) * 0.02 * (torch.rand(B, 1, 1, 1, device=dev) < 0.5).float()
    if a['blur']:
        blur = F.avg_pool2d(F.pad(out, (1, 1, 1, 1), mode='replicate'), 3, 1)
        out = torch.where(torch.rand(B, 1, 1, 1, device=dev) < a['blur'], blur, out)
    return out.clamp(0, 1)


def predict(net, D, idx, device, embed=False, batch=INFER_BATCH):
    """Score probabilities (and normalised embeddings) of the drawings idx, without augmentation."""
    torch, _ = _torch()
    G = _gpu(D, device)
    amp = _amp(device)
    net.eval()
    idx_t = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(device)
    while True:
        try:
            Q, E = [], []
            with torch.no_grad():
                for i in range(0, len(idx_t), batch):
                    b = idx_t[i:i + batch]
                    ink = 1 - G.imgs[b].float().div_(255).unsqueeze(1)
                    with torch.autocast(**amp):
                        logits, e = net(ink, G.item[b])
                    Q.append(torch.softmax(logits.float(), 1).cpu())
                    if embed:
                        E.append(e.float().cpu())
            q = torch.cat(Q).numpy().astype(np.float32)
            return q, (torch.cat(E).numpy().astype(np.float16) if embed else None)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 4:
                raise
            batch //= 2


# ---------------------------------------------------------------------------------------------------
# Files of one unit (network x fold)
# ---------------------------------------------------------------------------------------------------
def unit_paths(P, key, k):
    d = os.path.join(P.runs, key)
    return SimpleNamespace(npz=os.path.join(d, f'fold{k}.npz'), weights=os.path.join(d, f'fold{k}.safetensors'),
                           status=os.path.join(d, f'fold{k}.json'), history=os.path.join(d, f'fold{k}_history.csv'))


def signature(key, cfg, D, k):
    tr, va, te = bd.split(D, k)
    split_id = hashlib.sha1('|'.join(D.df.file.values[np.r_[tr, -1, va, -1, te]]).encode()).hexdigest()
    return json.dumps({'version': VERSION, 'seed': SEED, 'network': key, 'fold': k, 'input_px': int(D.imgs.shape[1]),
                       'split_sha1': split_id, 'config': cfg}, sort_keys=True, default=str)


def unit_done(paths, sig):
    if not os.path.exists(paths.npz):
        return False
    try:
        return str(np.load(paths.npz)['signature']) == sig
    except Exception:
        return False


def _matmul_params(net):
    """Weights and biases of linear and convolution layers, which run in bfloat16 under autocast."""
    import torch.nn as nn
    names = set()
    for name, m in net.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            for p in ('weight', 'bias'):
                if getattr(m, p, None) is not None:
                    names.add(f'{name}.{p}' if name else p)
    return names


def save_weights(net, path, metadata):
    """Stores the evaluated weights. Linear and convolution tensors are stored in bfloat16, the precision in which they
    were used, and all other tensors in float32, so reloading reproduces the stored predictions."""
    import torch
    from safetensors.torch import save_file
    low = _matmul_params(net)
    state = {n: (t.detach().to('cpu', torch.bfloat16) if n in low else t.detach().to('cpu')).clone().contiguous()
             for n, t in net.state_dict().items()}
    atomic_save(path, lambda p: save_file(state, p, metadata={k: str(v) for k, v in metadata.items()}))


def load_weights(net, path):
    from safetensors.torch import load_file
    ref = net.state_dict()
    net.load_state_dict({n: t.to(ref[n].dtype) for n, t in load_file(path).items()})
    return net


# ---------------------------------------------------------------------------------------------------
# Log lines (the same for live training and for the stored log of a finished unit)
# ---------------------------------------------------------------------------------------------------
def _start_line(header, n_tr, n_va, n_te, lr):
    return f'{header}: {n_tr} training, {n_va} validation and {n_te} test drawings, backbone learning rate {lr:.0e}'


def _epoch_line(h):
    return (f'  epoch {int(h["epoch"]):2d}  loss {h["train_loss"]:.3f}  train acc {100 * h["train_accuracy"]:5.1f}%  '
            f'val acc {100 * h["val_accuracy"]:5.1f}%  val QWK {h["val_qwk"]:6.3f}  lr {h["lr"]:.1e}')


def _restart_line(attempt, lr, epoch):
    return (f'  epochs {epoch - 2} to {epoch}: every validation prediction was the most frequent training score of its '
            f'item; restart {attempt} of {MAX_RESTARTS} at backbone learning rate {lr:.1e}')


def _result_line(z, n_te):
    return (f'  best epoch {int(z["best_epoch"])} of {int(z["epochs_run"])}: train acc {100 * float(z["train_accuracy"]):.1f}%, '
            f'val acc {100 * float(z["val_accuracy"]):.1f}%, test acc {100 * float(z["test_accuracy"]):.1f}% on {n_te} '
            f'test drawings ({float(z["seconds"]) / 60:.1f} min)')


# ---------------------------------------------------------------------------------------------------
# Training of one unit
# ---------------------------------------------------------------------------------------------------
class Collapse(Exception):
    def __init__(self, epoch):
        super().__init__(epoch)
        self.epoch = epoch


def collapsed(pred, items, mode, qwk):
    """True when the validation QWK is at most 0.02 and at least 98% of the validation predictions of every item are
    that item's most frequent training score."""
    return qwk <= 0.02 and all(np.mean(pred[items == i] == mode[i]) >= 0.98 for i in np.unique(items))


def _fit(key, k, D, cfg, attempt, pretrained, max_steps, device, history, log):
    torch, F = _torch()
    cuda = device == 'cuda'
    seed = zlib.crc32(f'{SEED}|{key}|{k}|{attempt}'.encode()) % (2 ** 31)
    torch.manual_seed(seed)
    np.random.seed(seed)
    scale = 0.5 ** attempt
    tr, va, te = bd.split(D, k)
    G = _gpu(D, device)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    model = build_network(cfg, pretrained).to(device).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc = list(model.encoder.parameters())
    in_enc = {id(p) for p in enc}
    opt = torch.optim.AdamW([
        {'params': [p for p in enc if p.ndim > 1], 'lr': cfg['lr'] * scale, 'weight_decay': cfg['weight_decay']},
        {'params': [p for p in enc if p.ndim <= 1], 'lr': cfg['lr'] * scale, 'weight_decay': 0.0},
        {'params': [p for p in model.parameters() if id(p) not in in_enc], 'lr': cfg['head_lr'] * scale,
         'weight_decay': 1e-4}])
    spe = max(len(tr) // cfg['batch'], 1)
    total, warm = cfg['epochs'] * spe, cfg['warmup'] * spe

    def lr_factor(step):                         # linear warm-up from 4 % of the learning rate, then cosine decay
        if step < warm:
            return 0.04 + 0.96 * step / warm
        return 0.5 * (1 + np.cos(np.pi * min((step - warm) / max(total - warm, 1), 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    amp = _amp(device)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda and amp['dtype'] == torch.float16)
    eps = cfg['smoothing']
    tr_t = torch.from_numpy(tr).to(device)
    mode = class_counts(D.item[tr], D.score[tr]).argmax(1)
    yv, iv = D.score[va], D.item[va]

    def loss_fn(logits, item, y):                 # cross-entropy with label smoothing over the item's valid scores
        logp = F.log_softmax(logits, 1)
        valid = G.valid[item].float()
        nll = -logp.gather(1, y[:, None]).squeeze(1)
        smooth = -(logp * valid).sum(1) / valid.sum(1)
        return ((1 - eps) * nll + eps * smooth).mean()

    def ema_update(step):
        d = min(cfg['ema'], (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm.detach(), 1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)

    best, best_state, best_row, stall, flat, step = (-1.0, -1.0), None, None, 0, 0, 0
    for ep in range(1, cfg['epochs'] + 1):
        t_ep = time.time()
        model.train()
        order = tr_t[torch.randperm(len(tr), device=device)]
        run_loss, nb = torch.zeros((), device=device), 0
        hits, seen = torch.zeros(N_ITEMS, device=device), torch.zeros(N_ITEMS, device=device)
        for i in range(0, spe * cfg['batch'], cfg['batch']):
            b = order[i:i + cfg['batch']]
            item_b, y_b = G.item[b], G.score[b]
            ink = augment(1 - G.imgs[b].float().div_(255).unsqueeze(1), cfg['augment'])
            with torch.autocast(**amp):
                logits, _ = model(ink, item_b)
            loss = loss_fn(logits, item_b, y_b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema_update(step)
            step += 1
            nb += 1
            run_loss += loss.detach()
            right = (logits.detach().argmax(1) == y_b).float()
            hits.index_add_(0, item_b, right)
            seen.index_add_(0, item_b, torch.ones_like(right))
            if max_steps is not None and step >= max_steps:
                break
        q_va, _ = predict(ema, D, va, device)
        pv = q_va.argmax(1)
        acc = mean_item_accuracy(yv, pv, iv)
        qk = float(np.nan_to_num(mean_item_kappa(yv, pv, iv, 'quadratic'), nan=-1.0))
        filled = seen > 0
        row = dict(attempt=attempt, epoch=ep, train_loss=float(run_loss) / max(nb, 1),
                   train_accuracy=float((hits[filled] / seen[filled]).mean()), val_accuracy=acc, val_qwk=qk,
                   lr=opt.param_groups[0]['lr'], seconds=time.time() - t_ep)
        history.append(row)
        log(_epoch_line(row))
        if (acc, qk) > best:
            best, best_row, stall = (acc, qk), row, 0
            best_state = {n: t.detach().to('cpu', copy=True) for n, t in ema.state_dict().items()}
        else:
            stall += 1
        flat = flat + 1 if collapsed(pv, iv, mode, qk) else 0
        if attempt < MAX_RESTARTS and ep >= cfg['warmup'] + 3 and flat >= 3:
            del model, ema, opt
            if cuda:
                torch.cuda.empty_cache()
            raise Collapse(ep)
        if stall >= cfg['patience'] or (max_steps is not None and step >= max_steps):
            break

    ema.load_state_dict(best_state)
    q_va, _ = predict(ema, D, va, device)
    q_te, emb_te = predict(ema, D, te, device, embed=True)
    out = SimpleNamespace(ema=ema, q_va=q_va, q_te=q_te, emb_te=emb_te, best_row=best_row,
                          epochs_run=sum(h['attempt'] == attempt for h in history), lr=cfg['lr'] * scale,
                          test_accuracy=mean_item_accuracy(D.score[te], q_te.argmax(1), D.item[te]),
                          n_params=sum(p.numel() for p in ema.parameters()),
                          peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0)
    del model, opt
    return out


def train_unit(key, k, D, P, header, cfg=None, pretrained=True, save=False, max_steps=None, device=None, log=print):
    """Trains one network on one fold, restarting at half the learning rate after a collapse, and stores the
    validation and test probabilities, the test embeddings, the epoch log and (if save) the weights."""
    torch, _ = _torch()
    cfg = dict(cfg or NETWORKS[key])
    device = _device(device)
    paths, sig = unit_paths(P, key, k), signature(key, cfg, D, k)
    tr, va, te = bd.split(D, k)
    log(_start_line(header, len(tr), len(va), len(te), cfg['lr']))
    t0, history = time.time(), []
    for attempt in range(MAX_RESTARTS + 1):
        try:
            res = _fit(key, k, D, cfg, attempt, pretrained, max_steps, device, history, log)
            break
        except Collapse as c:
            log(_restart_line(attempt + 1, cfg['lr'] * 0.5 ** (attempt + 1), c.epoch))
    secs = time.time() - t0
    if save:
        save_weights(res.ema, paths.weights, {'network': key, 'fold': k, 'signature': sig})
    save_csv(pd.DataFrame(history), paths.history, index=False)
    b = res.best_row
    z = dict(va_idx=va, te_idx=te, q_va=res.q_va, q_te=res.q_te, emb_te=res.emb_te,
             train_counts=class_counts(D.item[tr], D.score[tr]), best_epoch=b['epoch'], epochs_run=res.epochs_run,
             train_accuracy=b['train_accuracy'], val_accuracy=b['val_accuracy'], val_qwk=b['val_qwk'],
             test_accuracy=res.test_accuracy, restarts=b['attempt'], lr=res.lr, n_params=res.n_params,
             seconds=secs, peak_gb=res.peak_gb, signature=sig)
    atomic_save(paths.npz, lambda p: np.savez(p, **z))
    save_json(paths.status, {'state': 'done', 'network': key, 'fold': k, 'updated': time.time()})
    log(_result_line(z, len(te)))
    del res
    if device == 'cuda':
        torch.cuda.empty_cache()


def _replay(paths, header, D, k, log):
    """Prints the stored epoch log of a unit finished in an earlier session."""
    tr, va, te = bd.split(D, k)
    z = np.load(paths.npz)
    h = pd.read_csv(paths.history)
    log(_start_line(header, len(tr), len(va), len(te), float(z['lr']) / 0.5 ** int(z['restarts']))
        + '  [trained in an earlier session; stored log]')
    for attempt, g in h.groupby('attempt'):
        for r in g.to_dict('records'):
            log(_epoch_line(r))
        if attempt < int(z['restarts']):
            log(_restart_line(attempt + 1, float(z['lr']) / 0.5 ** int(z['restarts']) * 0.5 ** (attempt + 1),
                              int(g.epoch.max())))
    log(_result_line(z, len(te)))


# ---------------------------------------------------------------------------------------------------
# Embedding of the pretrained network (before fine-tuning), for the representation figure
# ---------------------------------------------------------------------------------------------------
def pretrained_embedding(D, P, key='convnext_512', cfg=None, pretrained=True, device=None, log=print):
    path = os.path.join(P.embeddings, f'{key}_pretrained{D.tag}.npy')
    if os.path.exists(path):
        return path
    device = _device(device)
    net = build_network(cfg or NETWORKS[key], pretrained).to(device)
    _, e = predict(net, D, np.arange(len(D.score)), device, embed=True)
    atomic_save(path, lambda p: np.save(p, e))
    log(f'Embedded all drawings with the pretrained {label(key)} encoder, before fine-tuning')
    del net
    return path


# ---------------------------------------------------------------------------------------------------
# All units
# ---------------------------------------------------------------------------------------------------
def run(D, P, keys=None, save='ensemble', cfgs=None, pretrained=True, max_steps=None, device=None, log=print):
    """Trains every network on every fold, network by network. A unit that fails is logged and tried once more at the
    end. save: 'ensemble' keeps the weights of the three members of the proposed ensemble, 'all' or 'none'."""
    keys, cfgs = list(keys or NETWORKS), cfgs or {}
    cfg_of = lambda key: dict(cfgs.get(key, NETWORKS[key]))
    units = [(key, k) for key in keys for k in range(D.n_folds)]
    done = lambda u: unit_done(unit_paths(P, *u), signature(u[0], cfg_of(u[0]), D, u[1]))
    left = [u for u in units if not done(u)]
    hours = sum(MINUTES_PER_FOLD.get(key, 5) for key, _ in left) / 60
    log(f'{len(units)} network-fold units: {len(units) - len(left)} finished earlier, {len(left)} to train '
        f'(about {hours:.1f} h on an A100)')
    if 'convnext_512' in keys:
        pretrained_embedding(D, P, 'convnext_512', cfg=cfgs.get('convnext_512'), pretrained=pretrained, device=device,
                             log=log)

    def attempt(n, u):
        key, k = u
        header = f'[{n}/{len(units)}] {label(key)}, fold {k}'
        keep = save == 'all' or (save == 'ensemble' and key in ENSEMBLE)
        try:
            train_unit(key, k, D, P, header, cfg=cfg_of(key), pretrained=pretrained, save=keep, max_steps=max_steps,
                       device=device, log=log)
            return True
        except Exception as e:
            log(f'{header}: FAILED ({type(e).__name__}: {str(e)[:300]})')
            save_json(unit_paths(P, key, k).status, {'state': 'failed', 'network': key, 'fold': k,
                                                     'detail': str(e)[:2000], 'traceback': traceback.format_exc(),
                                                     'updated': time.time()})
            torch, _ = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

    failed = []
    for n, u in enumerate(units, 1):
        if done(u):
            _replay(unit_paths(P, *u), f'[{n}/{len(units)}] {label(u[0])}, fold {u[1]}', D, u[1], log)
        elif not attempt(n, u):
            failed.append((n, u))
    for n, u in failed:
        log(f'Second attempt: {label(u[0])}, fold {u[1]}')
        attempt(n, u)
    return summary(D, P, keys, cfgs)


def summary(D, P, keys=None, cfgs=None):
    """Mean over the folds of the training, validation and test accuracy of every network (mean of the item
    accuracies; training accuracy on the augmented batches of the selected epoch), and the number of restarts."""
    keys, cfgs = list(keys or NETWORKS), cfgs or {}
    rows = []
    for key in keys:
        for k in range(D.n_folds):
            paths = unit_paths(P, key, k)
            if unit_done(paths, signature(key, dict(cfgs.get(key, NETWORKS[key])), D, k)):
                z = np.load(paths.npz)
                rows.append({'Network': label(key), 'Fold': k, 'State': 'done', 'Restarts': int(z['restarts']),
                             'Best epoch': int(z['best_epoch']), 'Epochs run': int(z['epochs_run']),
                             'Train accuracy (%)': 100 * float(z['train_accuracy']),
                             'Validation accuracy (%)': 100 * float(z['val_accuracy']),
                             'Test accuracy (%)': 100 * float(z['test_accuracy'])})
            else:
                state = load_json(paths.status).get('state', 'missing') if os.path.exists(paths.status) else 'missing'
                rows.append({'Network': label(key), 'Fold': k, 'State': state})
    t = pd.DataFrame(rows)
    save_csv(t, os.path.join(P.metrics, 'training_summary.csv'), index=False)
    ok = t[t.State == 'done']
    s = ok.groupby('Network', sort=False).agg(**{
        'Folds finished': ('Fold', 'size'), 'Restarts': ('Restarts', 'sum'),
        'Train accuracy (%)': ('Train accuracy (%)', 'mean'), 'Validation accuracy (%)': ('Validation accuracy (%)', 'mean'),
        'Test accuracy (%)': ('Test accuracy (%)', 'mean')}) if len(ok) else pd.DataFrame()
    return s.reindex([label(k) for k in keys]).round(2)
