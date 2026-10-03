"""Networks of the study: the three members of the proposed ensemble, the baselines and the controls."""
import pandas as pd

from .data import N_ITEMS, N_CLS, VALID

BASE = dict(kind='timm', size=384, lr=1e-4, head_lr=1e-3, batch=32, epochs=30, warmup=2, patience=8, ema=0.998,
            weight_decay=0.05, drop_path=0.1, smoothing=0.05, augment='shift', grad_ckpt=False, timm_kw={})

_CNX = 'convnextv2_tiny.fcmae_ft_in22k_in1k_384'
_VIT = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
_XCP = 'legacy_xception.tf_in1k'
_BEIT = dict(kind='beit', size=224, lr=3e-5)


def _net(label, group, **kw):
    return dict(BASE, label=label, group=group, **kw)


NETWORKS = {
    'convnext_512':     _net('ConvNeXt V2-T, 512 px', 'member', model=_CNX, size=512),
    'dinov2_336':       _net('DINOv2 ViT-L/14', 'member', model='vit_large_patch14_reg4_dinov2.lvd142m', size=336, lr=2e-5,
                             grad_ckpt=True, timm_kw=dict(img_size=336)),
    'dit_224':          _net('DiT-B, documents', 'member', model='microsoft/dit-base', **_BEIT),
    'resnet50':         _net('ResNet-50', 'baseline', model='resnet50.a1_in1k', lr=2e-4),
    'efficientnetv2_s': _net('EfficientNetV2-S', 'baseline', model='tf_efficientnetv2_s.in21k_ft_in1k', lr=2e-4),
    'vit_b16':          _net('ViT-B/16', 'baseline', model=_VIT, size=224, lr=3e-5, augment='rotate_shift_blur'),
    'cvit':             _net('CViT', 'baseline', kind='cvit', model=f'{_XCP} + {_VIT}', cnn=_XCP, cnn_size=299, vit=_VIT,
                             vit_kw={}, size=224, lr=3e-5, augment='rotate_shift_blur'),
    'beit_natural':     _net('BEiT-B, natural images', 'control', model='microsoft/beit-base-patch16-224-pt22k-ft22k', **_BEIT),
    'beit_sketch':      _net('BEiT-B, sketches', 'control', model='kmewhort/beit-sketch-classifier', **_BEIT),
    'convnext_224':     _net('ConvNeXt V2-T, 224 px', 'control', model=_CNX, size=224),
}
ENSEMBLE = ('convnext_512', 'dinov2_336', 'dit_224')
PROPOSED, PROPOSED_LABEL = 'dpe', 'DPE'
CLASSICAL = {'majority': 'Most frequent score', 'geometric_gb': 'Geometric + boosting'}
MINUTES_PER_FOLD = {'convnext_512': 13, 'dinov2_336': 18, 'dit_224': 3, 'resnet50': 4, 'efficientnetv2_s': 6, 'vit_b16': 2.5,
                    'cvit': 5, 'beit_natural': 3.5, 'beit_sketch': 3.5, 'convnext_224': 3}        # A100 estimates


def systems():
    """Every scorer in display order: proposed ensemble, its members, baselines, controls, feature-based scorers."""
    return [PROPOSED, *ENSEMBLE, *[k for k, c in NETWORKS.items() if c['group'] == 'baseline'],
            *[k for k, c in NETWORKS.items() if c['group'] == 'control'], *CLASSICAL]


def label(key):
    if key == PROPOSED:
        return PROPOSED_LABEL
    return NETWORKS[key]['label'] if key in NETWORKS else CLASSICAL[key]


def group(key):
    if key == PROPOSED:
        return 'proposed'
    return NETWORKS[key]['group'] if key in NETWORKS else 'classical'


def table():
    rows = [{'Key': k, 'Network': c['label'], 'Role': c['group'], 'Pretrained weights': c['model'], 'Input (px)': c['size'],
             'Backbone LR': c['lr'], 'Augmentation': c['augment']} for k, c in NETWORKS.items()]
    rows.append({'Key': PROPOSED, 'Network': PROPOSED_LABEL, 'Role': 'proposed',
                 'Pretrained weights': 'mean of the probabilities of ' + ', '.join(ENSEMBLE), 'Input (px)': '',
                 'Backbone LR': '', 'Augmentation': ''})
    return pd.DataFrame(rows).set_index('Key')


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
