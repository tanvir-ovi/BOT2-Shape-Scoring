"""Fine-tuning of one network on one fold, inference, embeddings, weight files and the resumable run schedule."""
import os
import copy
import json
import time
import zlib
import socket
import hashlib
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import data as bd
from .data import VALID, class_counts
from .models import NETWORKS, ENSEMBLE, MINUTES_PER_FOLD, build_network
from .utils import atomic_save, save_json, load_json, save_csv

VERSION = '1.0.0'          # part of every unit signature: a finished unit is retrained only if the code version changes
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
    c = {n: v for n, v in cfg.items() if n not in ('label', 'group')}
    return json.dumps({'version': VERSION, 'seed': SEED, 'network': key, 'fold': k, 'input_px': int(D.imgs.shape[1]),
                       'split_sha1': split_id, 'config': c}, sort_keys=True, default=str)


def unit_done(paths, sig):
    if not os.path.exists(paths.npz):
        return False
    try:
        return str(np.load(paths.npz)['signature']) == sig
    except Exception:
        return False


def _status(paths, **fields):
    try:
        save_json(paths.status, {**fields, 'updated': time.time(), 'host': socket.gethostname()})
    except OSError:
        pass


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
# Training of one unit
# ---------------------------------------------------------------------------------------------------
class Collapse(Exception):
    pass


def collapsed(pred, items, mode, qwk):
    """True when the validation predictions are the training mode of every item (no agreement beyond chance)."""
    return qwk <= 0.02 and all(np.mean(pred[items == i] == mode[i]) >= 0.98 for i in np.unique(items))


def _accuracy_qwk(y, p, items):
    from .evaluation import mean_item_accuracy, mean_item_qwk
    return mean_item_accuracy(y, p, items), float(np.nan_to_num(mean_item_qwk(y, p, items), nan=-1.0))


def train_unit(key, k, D, P, cfg=None, pretrained=True, save=False, max_steps=None, device=None, worker=0, log=print):
    """Trains one network on one fold. Returns False if the unit was already finished with the same signature."""
    cfg = dict(cfg or NETWORKS[key])
    paths = unit_paths(P, key, k)
    sig = signature(key, cfg, D, k)
    if unit_done(paths, sig):
        return False
    for attempt in range(MAX_RESTARTS + 1):
        try:
            _fit(key, k, D, P, cfg, sig, paths, attempt, pretrained, save, max_steps, _device(device), worker, log)
            return True
        except Collapse:
            log(f'{key} fold {k}: validation predictions collapsed to the most frequent score; restarting at '
                f'{0.5 ** (attempt + 1):g} x the learning rate')
    return True


def _fit(key, k, D, P, cfg, sig, paths, attempt, pretrained, save, max_steps, device, worker, log):
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
    t0 = time.time()
    _status(paths, key=key, fold=k, state='running', worker=worker, epoch=0, attempt=attempt)

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

    history, best, best_state, best_epoch, stall, flat, step = [], (-1.0, -1.0), None, 0, 0, 0, 0
    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        order = tr_t[torch.randperm(len(tr), device=device)]
        run_loss, nb = torch.zeros((), device=device), 0
        for i in range(0, spe * cfg['batch'], cfg['batch']):
            b = order[i:i + cfg['batch']]
            ink = augment(1 - G.imgs[b].float().div_(255).unsqueeze(1), cfg['augment'])
            with torch.autocast(**amp):
                logits, _ = model(ink, G.item[b])
            loss = loss_fn(logits, G.item[b], G.score[b])
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
            if max_steps is not None and step >= max_steps:
                break
        q_va, _ = predict(ema, D, va, device)
        pv = q_va.argmax(1)
        acc, qk = _accuracy_qwk(yv, pv, iv)
        history.append(dict(epoch=ep, loss=float(run_loss) / max(nb, 1), val_accuracy=acc, val_qwk=qk,
                            lr=opt.param_groups[0]['lr']))
        log(f'{key} fold {k} epoch {ep:2d}  loss {history[-1]["loss"]:.3f}  val accuracy {acc:.4f}  val QWK {qk:.3f}',
            quiet=True)
        _status(paths, key=key, fold=k, state='running', worker=worker, epoch=ep, attempt=attempt)
        _heartbeat(P, worker)
        if (acc, qk) > best:
            best, best_epoch, stall = (acc, qk), ep, 0
            best_state = {n: t.detach().to('cpu', copy=True) for n, t in ema.state_dict().items()}
        else:
            stall += 1
        flat = flat + 1 if collapsed(pv, iv, mode, qk) else 0
        if attempt < MAX_RESTARTS and ep >= cfg['warmup'] + 3 and flat >= 3:
            del model, ema, opt
            if cuda:
                torch.cuda.empty_cache()
            raise Collapse()
        if stall >= cfg['patience'] or (max_steps is not None and step >= max_steps):
            break

    ema.load_state_dict(best_state)
    q_va, _ = predict(ema, D, va, device)
    q_te, emb_te = predict(ema, D, te, device, embed=True)
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0
    n_params = sum(p.numel() for p in ema.parameters())
    if save:
        try:
            save_weights(ema, paths.weights, {'network': key, 'fold': k, 'signature': sig})
        except Exception as e:
            log(f'{key} fold {k}: weights not saved ({type(e).__name__}: {e})')
    save_csv(pd.DataFrame(history), paths.history, index=False)
    atomic_save(paths.npz, lambda p: np.savez(
        p, va_idx=va, te_idx=te, q_va=q_va, q_te=q_te, emb_te=emb_te, train_counts=class_counts(D.item[tr], D.score[tr]),
        best_epoch=best_epoch, epochs_run=len(history), val_accuracy=best[0], val_qwk=best[1], restarts=attempt,
        lr=cfg['lr'] * scale, n_params=n_params, seconds=secs, peak_gb=peak, signature=sig))
    _status(paths, key=key, fold=k, state='done', worker=worker, epoch=best_epoch, attempt=attempt)
    log(f'{key:16s} fold {k}  best epoch {best_epoch:2d} of {len(history):2d}  val accuracy {100 * best[0]:.2f}%  '
        f'restarts {attempt}  {secs / 60:.1f} min  peak {peak:.1f} GB')
    del model, ema, opt
    if cuda:
        torch.cuda.empty_cache()


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
    log(f'Embedded all drawings with the pretrained {key} encoder')
    del net
    return path


# ---------------------------------------------------------------------------------------------------
# Run schedule: one Colab tab, or several tabs that share the units
# ---------------------------------------------------------------------------------------------------
def plan(keys, n_folds, n_workers):
    """Assigns every (network, fold) unit to a worker, balancing the estimated A100 minutes."""
    units = [(key, k) for k in range(n_folds) for key in keys]
    load, owner = [0.0] * n_workers, {}
    for u in sorted(units, key=lambda u: (-MINUTES_PER_FOLD.get(u[0], 5), u[1])):
        w = int(np.argmin(load))
        owner[u] = w
        load[w] += MINUTES_PER_FOLD.get(u[0], 5)
    return units, owner


def _heartbeat(P, worker):
    try:
        save_json(os.path.join(P.logs, f'worker{worker}.json'), {'worker': worker, 'updated': time.time(),
                                                                 'host': socket.gethostname()})
    except OSError:
        pass


def _heartbeat_age(P, worker, since):
    path = os.path.join(P.logs, f'worker{worker}.json')
    try:
        return time.time() - load_json(path)['updated']
    except Exception:
        return time.time() - since


def run(D, P, keys=None, worker=0, n_workers=1, save='ensemble', stale_minutes=20, cfgs=None, pretrained=True,
        max_steps=None, device=None, log=print):
    """Trains every unit assigned to this worker that is not finished. Worker 0 then waits for the other workers and
    takes over units whose worker stopped sending heartbeats."""
    keys = list(keys or NETWORKS)
    cfgs = cfgs or {}
    units, owner = plan(keys, D.n_folds, n_workers)

    def finished(u):
        cfg = dict(cfgs.get(u[0], NETWORKS[u[0]]))
        return unit_done(unit_paths(P, *u), signature(u[0], cfg, D, u[1]))

    def attempt(u):
        key, k = u
        _heartbeat(P, worker)
        keep = save == 'all' or (save == 'ensemble' and key in ENSEMBLE)
        try:
            train_unit(key, k, D, P, cfg=cfgs.get(key), pretrained=pretrained, save=keep, max_steps=max_steps,
                       device=device, worker=worker, log=log)
        except Exception as e:
            log(f'{key} fold {k}: FAILED ({type(e).__name__}: {str(e)[:300]})')
            log(traceback.format_exc(), quiet=True)
            _status(unit_paths(P, key, k), key=key, fold=k, state='failed', worker=worker, detail=str(e)[:2000])
            torch, _ = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    mine = [u for u in units if owner[u] == worker]
    todo = [u for u in mine if not finished(u)]
    minutes = sum(MINUTES_PER_FOLD.get(key, 5) for key, _ in todo)
    log(f'Worker {worker} of {n_workers}: {len(mine)} units assigned, {len(mine) - len(todo)} finished, '
        f'about {minutes / 60:.1f} h of A100 time left')
    if worker == 0 and 'convnext_512' in keys:
        pretrained_embedding(D, P, 'convnext_512', cfg=cfgs.get('convnext_512'), pretrained=pretrained, device=device,
                             log=log)
    for u in todo:
        attempt(u)
    _heartbeat(P, worker)

    def state(u):
        try:
            return load_json(unit_paths(P, *u).status).get('state')
        except Exception:
            return None

    if worker == 0:                               # one retry of failed units; takeover of units of stopped workers
        since, retried = time.time(), set()
        while True:
            pending = [u for u in units if not finished(u)]
            ages = {w: _heartbeat_age(P, w, since) for w in range(1, n_workers)}
            takeover = [u for u in pending if u not in retried and
                        (owner[u] == 0 or state(u) == 'failed' or ages[owner[u]] > stale_minutes * 60)]
            if takeover:
                for u in takeover:
                    if not finished(u):
                        log(f'Retrying {u[0]} fold {u[1]}' if owner[u] == 0 else f'Worker 0 takes over {u[0]} fold {u[1]}')
                        retried.add(u)
                        attempt(u)
                continue
            if not pending or all(u in retried for u in pending):
                break
            time.sleep(60)
    return progress(P, D, keys, cfgs)


def progress(P, D, keys=None, cfgs=None):
    """Validation accuracy (%) of every finished unit; other states are named."""
    keys, cfgs = list(keys or NETWORKS), cfgs or {}
    rows = {}
    for key in keys:
        row = {}
        for k in range(D.n_folds):
            paths = unit_paths(P, key, k)
            cfg = dict(cfgs.get(key, NETWORKS[key]))
            if unit_done(paths, signature(key, cfg, D, k)):
                z = np.load(paths.npz)
                row[f'Fold {k}'] = f'{100 * float(z["val_accuracy"]):.1f}' + (f' (restart {int(z["restarts"])})'
                                                                             if int(z['restarts']) else '')
            elif os.path.exists(paths.status):
                row[f'Fold {k}'] = load_json(paths.status).get('state', '')
            else:
                row[f'Fold {k}'] = 'missing'
        rows[NETWORKS[key]['label']] = row
    t = pd.DataFrame(rows).T
    t.index.name = 'Validation accuracy (%)'
    return t
