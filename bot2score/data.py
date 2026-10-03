"""BOT-2 shape items, archive indexing, image preprocessing and child-grouped cross-validation folds."""
import os
import re
import shutil
import zipfile
import hashlib
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
import cv2

from .utils import atomic_save, parallel_map, save_csv

FACETS = {
    'Circle':             ['basic', 'closure', 'edges', 'size'],
    'Square':             ['basic', 'closure', 'edges', 'orientation', 'size'],
    'Overlapped circle':  ['basic', 'closure', 'edges', 'orientation', 'overlap', 'size'],
    'Wave':               ['basic', 'edges', 'orientation', 'size'],
    'Triangle':           ['basic', 'closure', 'edges', 'orientation', 'size'],
    'Diagonal':           ['basic', 'closure', 'edges', 'orientation', 'size'],
    'Star':               ['basic', 'closure', 'edges', 'orientation', 'size'],
    'Overlapped pencils': ['basic', 'closure', 'edges', 'orientation', 'overlap', 'size'],
}
ITEMS = list(FACETS)                                                  # BOT-2 administration order
ITEM_NAME = {s: 'Diamond' if s == 'Diagonal' else s for s in ITEMS}   # the folder "Diagonal" holds the diamond item
ITEM_LABEL = {'Circle': 'Circle', 'Square': 'Square', 'Overlapped circle': 'Overlapping circles', 'Wave': 'Wavy line',
              'Triangle': 'Triangle', 'Diagonal': 'Diamond', 'Star': 'Star', 'Overlapped pencils': 'Overlapping pencils'}
MAX_SCORE = {s: len(f) for s, f in FACETS.items()}
ITEM_ID = {s: i for i, s in enumerate(ITEMS)}
N_ITEMS, N_CLS = len(ITEMS), max(MAX_SCORE.values()) + 1
K_OF_ITEM = np.array([MAX_SCORE[s] for s in ITEMS])
VALID = np.arange(N_CLS)[None, :] <= K_OF_ITEM[:, None]
PATTERN = re.compile(r'^img(\d+)-([a-z]+)-(\d+)(?:\((\d+)\))?\.png$', re.I)

N_FOLDS, N_INNER, FOLD_SEED = 5, 8, 2026
INPUT_SIZE = 512


def parse_facets(score, code, k):
    if code:
        fv = [int(c) for c in code]
    elif score in (0, k):
        fv = [int(score == k)] * k
    else:
        return ''
    ok = len(fv) == k and max(fv) <= 1 and sum(fv) == score
    return ''.join(map(str, fv)) if ok else ''


def build_index(root):
    """One row per drawing. Byte-identical files with conflicting scores are removed. Children are inferred from the
    running image number, which restarts the BOT-2 item order for every child."""
    rows, dropped = [], []
    for item in ITEMS:
        k = MAX_SCORE[item]
        for f in sorted(os.listdir(os.path.join(root, item))):
            m = PATTERN.match(f)
            if not m:
                dropped.append((item, f, 'file name')); continue
            num, score = int(m.group(1)), int(m.group(3))
            if score > k:
                dropped.append((item, f, 'score above maximum')); continue
            p = os.path.join(root, item, f)
            rows.append(dict(file=f, path=p, item=item, num=num, score=score, facets=parse_facets(score, m.group(4), k),
                             md5=hashlib.md5(open(p, 'rb').read()).hexdigest()))
    df = pd.DataFrame(rows)
    dup = df.md5.duplicated(keep=False)
    dropped += [(i, f, 'duplicate, conflicting score') for i, f in zip(df['item'][dup], df.file[dup])]
    df = df[~dup].drop(columns='md5').reset_index(drop=True)
    order = df['item'].map(ITEMS.index)
    seq = pd.DataFrame({'num': df.num, 'order': order}).sort_values(['num', 'order'])
    start = (seq.order.diff() <= 0) | (seq.num.diff() > len(ITEMS))
    df['child'] = start.cumsum().reindex(df.index)
    return df, pd.DataFrame(dropped, columns=['item', 'file', 'reason'])


def ink_map(path, max_side=1024):
    """Pencil strokes as a map in [0, 1] (1 = ink), after background flattening and removal of redaction patches."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    s = max_side / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    white = (img.min(axis=2) >= 250).astype(np.uint8)
    if 0 < white.mean() < 0.5:
        img = cv2.inpaint(img, cv2.dilate(white, np.ones((7, 7), np.uint8)), 5, cv2.INPAINT_TELEA)
    bg = cv2.medianBlur(cv2.dilate(img, np.ones((15, 15), np.uint8)), 21).astype(np.float32) + 1
    ink = np.clip(1 - (img.astype(np.float32) / bg).min(axis=2), 0, 1)
    ink = cv2.GaussianBlur(ink, (3, 3), 0)
    h, w = ink.shape
    core = ink[int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)]
    return np.clip(ink / max(np.percentile(core, 99.5), 0.08), 0, 1)


def to_input(path, size=INPUT_SIZE):
    """Square network input: the ink map centred on a white canvas, dark strokes on white, uint8."""
    ink = ink_map(path)
    h, w = ink.shape
    n = max(h, w)
    canvas = np.zeros((n, n), np.float32)
    canvas[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = ink
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return (255 - canvas * 255).astype(np.uint8)


def extract_archive(zip_path, local_dir):
    root = os.path.join(local_dir, 'Shapes')
    if not os.path.isdir(root):
        os.makedirs(local_dir, exist_ok=True)
        src = zip_path
        if zip_path.startswith('/content/drive'):                 # reading a local copy is much faster than Drive
            src = os.path.join(local_dir, 'Shapes.zip')
            shutil.copy(zip_path, src)
        with zipfile.ZipFile(src) as z:
            z.extractall(local_dir)
        if src != zip_path:
            os.remove(src)
    return root


def make_folds(df, n_folds=N_FOLDS, seed=FOLD_SEED):
    """Child-grouped folds stratified by item and score: every drawing of an inferred child is in one fold."""
    from sklearn.model_selection import StratifiedGroupKFold
    strat = (df['item'] + '_' + df.score.astype(str)).values
    fold = np.full(len(df), -1)
    for k, (_, te) in enumerate(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(df, strat, df.child.values)):
        fold[te] = k
    return fold


def split(D, k, n_inner=N_INNER, seed=FOLD_SEED):
    """Training, validation and test indices of fold k. The test part is fold k; the validation part is one eighth of
    the remaining children, used only for checkpoint selection."""
    from sklearn.model_selection import StratifiedGroupKFold
    rest, te = np.where(D.fold != k)[0], np.where(D.fold == k)[0]
    a, b = next(StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed + k).split(rest, D.strata[rest], D.child[rest]))
    tr, va = rest[a], rest[b]
    assert not (set(D.child[tr]) & set(D.child[va])) and not (set(D.child[np.r_[tr, va]]) & set(D.child[te]))
    return tr, va, te


def prepare(zip_path, P, local_dir='/content/data', size=INPUT_SIZE, workers=None, max_children=None, log=print):
    """Extracts the archive, builds the index and folds, and loads (or builds once) the network input cache."""
    root = extract_archive(zip_path, local_dir)
    df, dropped = build_index(root)
    if max_children:
        df = df[df.child < max_children].reset_index(drop=True)
    df['fold'] = make_folds(df)
    assert df.groupby('child').fold.nunique().max() == 1
    tag = f'_sub{max_children}' if max_children else ''
    index_file = os.path.join(P.data, f'index{tag}.csv')
    if os.path.exists(index_file):
        saved = pd.read_csv(index_file, dtype={'facets': str}, keep_default_na=False)
        assert saved.file.tolist() == df.file.tolist() and (saved.fold.values == df.fold.values).all(), \
            f'{index_file} was written by a different build of the index'
    else:
        save_csv(df.drop(columns='path'), index_file, index=False)
        save_csv(dropped, os.path.join(P.data, f'excluded_files{tag}.csv'), index=False)

    cache = os.path.join(P.data, f'inputs_{size}{tag}.npz')
    imgs = None
    if os.path.exists(cache):
        z = np.load(cache)
        if z['files'].tolist() == df.file.tolist():
            imgs = z['x']
    if imgs is None:
        log('Building the input cache (first run only) ...')
        imgs = np.stack(parallel_map(partial(to_input, size=size), df.path.tolist(), workers))
        atomic_save(cache, lambda p: np.savez_compressed(p, x=imgs, files=np.array(df.file.tolist(), dtype=str)))

    D = SimpleNamespace(df=df, dropped=dropped, imgs=imgs, item=df['item'].map(ITEM_ID).values,
                        score=df.score.values.astype(int), child=df.child.values, fold=df.fold.values,
                        strata=(df['item'] + '_' + df.score.astype(str)).values, n_folds=N_FOLDS, size=size, tag=tag)
    D.kmax = K_OF_ITEM[D.item]
    log(f'{len(df)} drawings from {df.child.nunique()} inferred children, {len(dropped)} files excluded, '
        f'{N_FOLDS} child-grouped folds, inputs {imgs.shape[1]} x {imgs.shape[2]} px')
    return D


def class_counts(items, scores):
    cnt = np.zeros((N_ITEMS, N_CLS))
    np.add.at(cnt, (items, scores), 1)
    return cnt


def score_table(D):
    """Drawings per item and score, with the share at full marks."""
    t = pd.crosstab(D.df['item'].map(ITEM_NAME), D.df.score).reindex([ITEM_NAME[s] for s in ITEMS]).fillna(0).astype(int)
    t['Total'] = t.sum(1)
    t['Full marks (%)'] = [100 * t.loc[ITEM_NAME[s], MAX_SCORE[s]] / t.loc[ITEM_NAME[s], 'Total'] for s in ITEMS]
    t.index.name, t.columns.name = 'Item', 'Score'
    return t.round(1)


def fold_table(D):
    rows = []
    for k in range(D.n_folds):
        tr, va, te = split(D, k)
        rows.append({'Fold': k, 'Training drawings': len(tr), 'Validation drawings': len(va), 'Test drawings': len(te),
                     'Test children': len(np.unique(D.child[te])),
                     'Test below full marks (%)': 100 * np.mean(D.score[te] < D.kmax[te])})
    return pd.DataFrame(rows).set_index('Fold').round(1)
