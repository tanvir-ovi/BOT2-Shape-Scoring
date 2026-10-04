"""Independent check of a finished run: rebuilds each saved network from its weight file, predicts its test fold
again and compares with the stored predictions; lists SHA-256 checksums of every result file."""
import os
import glob

import numpy as np
import pandas as pd

from .models import NETWORKS, build_network
from .scorers import label
from .training import unit_paths, load_weights, predict, _device
from .utils import sha256, save_csv


def check_weights(D, P, keys=None, cfgs=None, device=None, log=print):
    """For every stored weight file: the largest absolute difference between the re-predicted and the stored test
    probabilities, and the share of test drawings whose predicted score is unchanged."""
    device, cfgs, rows = _device(device), cfgs or {}, []
    for key in keys or NETWORKS:
        for k in range(D.n_folds):
            paths = unit_paths(P, key, k)
            if not (os.path.exists(paths.weights) and os.path.exists(paths.npz)):
                continue
            z = np.load(paths.npz)
            net = load_weights(build_network(cfgs.get(key, NETWORKS[key]), pretrained=False), paths.weights).to(device)
            q, _ = predict(net, D, z['te_idx'], device)
            rows.append({'Network': label(key), 'Fold': k, 'Test drawings': len(q),
                         'Largest probability difference': float(np.abs(q - z['q_te']).max()),
                         'Same predicted score (%)': 100 * float(np.mean(q.argmax(1) == z['q_te'].argmax(1)))})
            del net
    t = pd.DataFrame(rows)
    if len(t):
        save_csv(t, os.path.join(P.metrics, 'weight_verification.csv'), index=False)
        same = int((t['Same predicted score (%)'] == 100).sum())
        log(f'Re-predicted {len(t)} test folds from the saved weights: the predicted score of every test drawing is '
            f'unchanged in {same} of {len(t)} folds; largest probability difference '
            f'{t["Largest probability difference"].max():.1e}')
    else:
        log('No saved weights found')
    return t


def manifest(P, log=print):
    """SHA-256 checksum, size and path of every result file except the input cache and the running logs."""
    rows = []
    for path in sorted(glob.glob(os.path.join(P.root, '**', '*'), recursive=True)):
        rel = os.path.relpath(path, P.root).replace(os.sep, '/')
        name = os.path.basename(path)
        if os.path.isfile(path) and not name.startswith(('inputs_', 'manifest')) and not (rel.startswith('logs/') and
                                                                                          name.endswith('.txt')):
            rows.append({'File': rel, 'Bytes': os.path.getsize(path), 'SHA-256': sha256(path)})
    t = pd.DataFrame(rows)
    save_csv(t, os.path.join(P.root, 'manifest_sha256.csv'), index=False)
    log(f'Checksums of {len(t)} files written to manifest_sha256.csv')
    return t
