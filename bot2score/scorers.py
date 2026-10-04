"""Every scorer of the study, with its display name, paradigm and role."""

PROPOSED, PROPOSED_LABEL = 'dpe', 'DPE'
ENSEMBLE = ('convnext_512', 'dinov2_336', 'dit_224')

DL, ML, RB, REF = 'Deep learning', 'Machine learning', 'Rule-based', 'Reference'

SCORERS = {                                    # key: (label, paradigm, role), in display order
    'dpe':                 ('DPE', DL, 'proposed'),
    'convnext_512':        ('ConvNeXt V2-T, 512 px', DL, 'member'),
    'dinov2_336':          ('DINOv2 ViT-L/14', DL, 'member'),
    'dit_224':             ('DiT-B, documents', DL, 'member'),
    'resnet50':            ('ResNet-50', DL, 'baseline'),
    'mobilenetv3':         ('MobileNetV3-Large', DL, 'baseline'),
    'efficientnetv2_s':    ('EfficientNetV2-S', DL, 'baseline'),
    'vit_b16':             ('ViT-B/16', DL, 'baseline'),
    'cvit':                ('CViT', DL, 'baseline'),
    'beit_natural':        ('BEiT-B, natural images', DL, 'control'),
    'beit_sketch':         ('BEiT-B, sketches', DL, 'control'),
    'convnext_224':        ('ConvNeXt V2-T, 224 px', DL, 'control'),
    'gradient_boosting':   ('Gradient boosting', ML, 'classifier'),
    'random_forest':       ('Random forest', ML, 'classifier'),
    'svm':                 ('SVM, RBF kernel', ML, 'classifier'),
    'logistic_regression': ('Logistic regression', ML, 'classifier'),
    'rules':               ('OpenCV rubric rules', RB, 'rules'),
    'majority':            ('Most frequent score', REF, 'reference'),
}
PAIRS = {                                      # the proposed ensemble with one member left out
    'dpe_wo_dit':      ('DPE without DiT-B', ('convnext_512', 'dinov2_336')),
    'dpe_wo_dinov2':   ('DPE without DINOv2', ('convnext_512', 'dit_224')),
    'dpe_wo_convnext': ('DPE without ConvNeXt V2', ('dinov2_336', 'dit_224')),
}
SELECTED = {                                   # in each fold, the scorer of a paradigm with the best validation accuracy
    'best_network':    ('Best single network', DL),
    'best_classifier': ('Best classifier', ML),
}
NETWORK_KEYS = [k for k, (_, p, r) in SCORERS.items() if p == DL and r != 'proposed']
CLASSIFIER_KEYS = [k for k, (_, _, r) in SCORERS.items() if r == 'classifier']


def label(key):
    if key in SCORERS:
        return SCORERS[key][0]
    if key in PAIRS:
        return PAIRS[key][0]
    return SELECTED[key][0]


def paradigm(key):
    if key in SCORERS:
        return SCORERS[key][1]
    return DL if key in PAIRS else SELECTED[key][1]


def role(key):
    if key in SCORERS:
        return SCORERS[key][2]
    return 'pair' if key in PAIRS else 'selected'


def individual():
    """The proposed ensemble and every scorer trained or fitted on its own, in display order."""
    return list(SCORERS)
