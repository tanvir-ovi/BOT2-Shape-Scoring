"""Networks: the three members of the proposed ensemble, the baselines and the controls, their shared training recipe
and their builders."""
import pandas as pd

from .data import N_ITEMS, N_CLS, VALID
from .scorers import SCORERS, ENSEMBLE, PROPOSED, NETWORK_KEYS, label, paradigm, role

BASE = dict(kind='timm', size=384, lr=1e-4, head_lr=1e-3, batch=32, epochs=30, warmup=2, patience=8, ema=0.998,
            weight_decay=0.05, drop_path=0.1, smoothing=0.05, augment='shift', grad_ckpt=False, timm_kw={})

_CNX = 'convnextv2_tiny.fcmae_ft_in22k_in1k_384'
_VIT = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
_XCP = 'legacy_xception.tf_in1k'
_BEIT = dict(kind='beit', size=224, lr=3e-5)
_CNX_DATA = 'FCMAE self-supervised, then ImageNet-22k and ImageNet-1k'


def _net(**kw):
    return dict(BASE, **kw)


NETWORKS = {
    'convnext_512':     _net(model=_CNX, size=512, data=_CNX_DATA),
    'dinov2_336':       _net(model='vit_large_patch14_reg4_dinov2.lvd142m', size=336, lr=2e-5, grad_ckpt=True,
                             timm_kw=dict(img_size=336), data='Self-supervised, LVD-142M natural images'),
    'dit_224':          _net(model='microsoft/dit-base', data='Self-supervised, 42 million scanned document pages', **_BEIT),
    'resnet50':         _net(model='resnet50.a1_in1k', lr=2e-4, data='ImageNet-1k'),
    'mobilenetv3':      _net(model='mobilenetv3_large_100.ra_in1k', lr=2e-4, data='ImageNet-1k'),
    'efficientnetv2_s': _net(model='tf_efficientnetv2_s.in21k_ft_in1k', lr=2e-4, data='ImageNet-21k, then ImageNet-1k'),
    'vit_b16':          _net(model=_VIT, size=224, lr=3e-5, augment='rotate_shift_blur',
                             data='ImageNet-21k, then ImageNet-1k'),
    'cvit':             _net(kind='cvit', model=f'{_XCP} + {_VIT}', cnn=_XCP, cnn_size=299, vit=_VIT, vit_kw={}, size=224,
                             lr=3e-5, augment='rotate_shift_blur',
                             data='ImageNet-1k (Xception); ImageNet-21k, then ImageNet-1k (ViT-B/16)'),
    'beit_natural':     _net(model='microsoft/beit-base-patch16-224-pt22k-ft22k',
                             data='Self-supervised, then supervised, ImageNet-22k', **_BEIT),
    'beit_sketch':      _net(model='kmewhort/beit-sketch-classifier',
                             data='BEiT-B ImageNet-22k, then QuickDraw sketches', **_BEIT),
    'convnext_224':     _net(model=_CNX, size=224, data=_CNX_DATA),
}
assert list(NETWORKS) == NETWORK_KEYS
AUGMENT_NAME = {'shift': 'translation up to 4%', 'rotate_shift_blur': 'rotation up to 10°, translation up to 10%, blur'}
MINUTES_PER_FOLD = {'convnext_512': 13, 'dinov2_336': 18, 'dit_224': 3, 'resnet50': 4, 'mobilenetv3': 3,
                    'efficientnetv2_s': 6, 'vit_b16': 2.5, 'cvit': 5, 'beit_natural': 3.5, 'beit_sketch': 3.5,
                    'convnext_224': 3}                                                   # A100 estimates


def scorer_table():
    """Every scorer with its paradigm, role and input."""
    from .classical import SETTINGS
    rows = []
    for key in SCORERS:
        if key == PROPOSED:
            inp, how = 'Image', 'mean of the score probabilities of ' + ', '.join(label(m) for m in ENSEMBLE)
        elif key in NETWORKS:
            c = NETWORKS[key]
            inp = f"Image, {c['cnn_size']} + {c['size']} px" if c['kind'] == 'cvit' else f"Image, {c['size']} px"
            how = f"fine-tuned from {c['model']} ({c['data']})"
        elif key == 'rules':
            inp, how = 'OpenCV measurements', 'one rule per rubric criterion; score = criteria passed, 0 if the basic shape fails'
        else:
            inp, how = '64 OpenCV features' if key != 'majority' else 'Item only', SETTINGS[key]
        rows.append({'Key': key, 'Scorer': label(key), 'Paradigm': paradigm(key), 'Role': role(key), 'Input': inp,
                     'Method': how})
    return pd.DataFrame(rows).set_index('Key')


def network_table():
    """Pretrained weights, input size, learning rates and augmentation of every network."""
    rows = []
    for key, c in NETWORKS.items():
        rows.append({'Key': key, 'Network': label(key), 'Role': role(key), 'Pretrained weights': c['model'],
                     'Pretraining data': c['data'],
                     'Input (px)': f"{c['cnn_size']} + {c['size']}" if c['kind'] == 'cvit' else str(c['size']),
                     'Backbone LR': f"{c['lr']:.0e}", 'Head LR': f"{c['head_lr']:.0e}",
                     'Augmentation': AUGMENT_NAME[c['augment']]})
    return pd.DataFrame(rows).set_index('Key')


def recipe():
    """Training settings shared by every network."""
    b = BASE
    return pd.DataFrame([
        ('Loss', f"cross-entropy over the item's valid scores, label smoothing {b['smoothing']}"),
        ('Sampling', 'natural frequencies, no class weights'),
        ('Optimiser', f"AdamW, weight decay {b['weight_decay']} on backbone weights and 1e-4 on the head, "
                      'gradient norm clipped at 2'),
        ('Schedule', f"linear warm-up from 4% of the learning rate over {b['warmup']} epochs, then cosine decay"),
        ('Epochs', f"at most {b['epochs']}, early stopping after {b['patience']} epochs without a better validation accuracy"),
        ('Batch', f"{b['batch']} drawings"),
        ('Weights evaluated', f"exponential moving average (decay {b['ema']}) at the epoch with the best validation accuracy"),
        ('Regularisation', f"stochastic depth {b['drop_path']}, dropout 0.3 before the score head"),
        ('Augmentation', 'training batches only: translation (and rotation and blur where listed), stroke darkness, noise'),
        ('Precision', 'bfloat16 autocast on the GPU'),
        ('Collapse rule', 'when, for 3 consecutive epochs from epoch 5, the validation QWK is at most 0.02 and at least '
                          '98% of the validation predictions of every item are its most frequent training score, the '
                          'run restarts at half the learning rate, at most twice'),
        ('Decision', 'most probable valid score of the item'),
    ], columns=['Setting', 'Value']).set_index('Setting')


# ---------------------------------------------------------------------------------------------------
# Network builders
# ---------------------------------------------------------------------------------------------------
def _torch():
    import torch, torch.nn as nn, torch.nn.functional as F
    return torch, nn, F


def _prep(ink, size, mean, std):
    """Ink map (B, 1, H, W; 1 = stroke) to a normalised RGB image of the encoder's input size."""
    torch, _, F = _torch()
    x = 1 - ink
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False, antialias=x.shape[-1] > size)
    return (x.expand(-1, 3, -1, -1) - mean) / std


def build_encoder(cfg, pretrained=True):
    """nn.Module mapping the ink map to one feature vector per drawing; the width is stored in .n_out."""
    torch, nn, F = _torch()
    import timm

    def stats(module, mean, std):
        module.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        module.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    if cfg['kind'] == 'timm':
        class TimmEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                kw = dict(cfg['timm_kw'], drop_path_rate=cfg['drop_path'])
                self.net = timm.create_model(cfg['model'], pretrained=pretrained, num_classes=0, **kw)
                if cfg['grad_ckpt']:
                    self.net.set_grad_checkpointing(True)
                pc = self.net.pretrained_cfg
                stats(self, pc.get('mean', (0.485, 0.456, 0.406)), pc.get('std', (0.229, 0.224, 0.225)))

            def forward(self, ink):
                x = _prep(ink, cfg['size'], self.mean, self.std)
                return self.net(x.contiguous(memory_format=torch.channels_last))
        enc = TimmEncoder()

    elif cfg['kind'] == 'cvit':
        class CViTEncoder(nn.Module):
            """Pooled Xception features concatenated with the ViT-B/16 class token, then FC 512, ReLU and dropout."""
            def __init__(self):
                super().__init__()
                self.cnn = timm.create_model(cfg['cnn'], pretrained=pretrained, num_classes=0)
                self.vit = timm.create_model(cfg['vit'], pretrained=pretrained, num_classes=0, **cfg['vit_kw'])
                stats(self, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                self.fc = nn.Sequential(nn.Linear(self.cnn.num_features + self.vit.num_features, 512), nn.ReLU(),
                                        nn.Dropout(0.5))

            def forward(self, ink):
                a = self.cnn(_prep(ink, cfg['cnn_size'], self.mean, self.std))
                b = self.vit(_prep(ink, cfg['size'], self.mean, self.std))
                return self.fc(torch.cat([a, b], 1))
        enc = CViTEncoder()

    elif cfg['kind'] == 'beit':
        class BeitEncoder(nn.Module):
            """BEiT-B backbone (also DiT); the drawing feature is the mean of the patch tokens."""
            def __init__(self):
                super().__init__()
                from transformers import BeitConfig, BeitModel
                if cfg.get('hf_config'):
                    self.net = BeitModel(BeitConfig(**cfg['hf_config']), add_pooling_layer=False)
                elif pretrained:
                    self.net, info = BeitModel.from_pretrained(cfg['model'], add_pooling_layer=False, output_loading_info=True)
                    missing = [k for k in info['missing_keys'] if 'pooler' not in k]
                    if missing:
                        raise RuntimeError(f'{cfg["model"]}: {len(missing)} encoder tensors missing, e.g. {sorted(missing)[:3]}')
                else:
                    self.net = BeitModel(BeitConfig.from_pretrained(cfg['model']), add_pooling_layer=False)
                stats(self, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

            def forward(self, ink):
                h = self.net(pixel_values=_prep(ink, cfg['size'], self.mean, self.std)).last_hidden_state
                return h[:, 1:].mean(1)
        enc = BeitEncoder()
    else:
        raise ValueError(cfg['kind'])

    with torch.no_grad():
        enc.eval()
        enc.n_out = enc(torch.zeros(1, 1, 64, 64)).shape[1]
    enc.train()
    return enc


def build_network(cfg, pretrained=True):
    """Encoder, layer normalisation and one score head per item. forward() returns the logits of the drawing's item,
    with scores above that item's maximum masked, and the normalised feature vector used as the embedding."""
    torch, nn, F = _torch()

    class ScoreNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = build_encoder(cfg, pretrained)
            d = self.encoder.n_out
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(0.3)
            self.head = nn.Linear(d, N_ITEMS * N_CLS)
            self.register_buffer('valid', torch.from_numpy(VALID), persistent=False)

        def forward(self, ink, item):
            e = self.norm(self.encoder(ink).float())
            rows = torch.arange(len(item), device=e.device)
            logits = self.head(self.drop(e)).view(-1, N_ITEMS, N_CLS)[rows, item].float()
            return logits.masked_fill(~self.valid[item], -1e4), e

    return ScoreNet()
