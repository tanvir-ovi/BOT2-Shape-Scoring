"""BOT-2 shape items and rubric criteria, indexing of the archive, image preprocessing and the child-grouped folds."""
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

FACETS = {                                     # rubric criteria of every item, in rubric order
    'Circle':             ['basic', 'closure', 'edges', 'size'],
    'Square':             ['basic', 'closure', 'edges', 'orientation', 'size'],
    'Overlapped circle':  ['basic', 'closure', 'edges', 'orientation', 'overlap', 'size'],
    'Wave':               ['basic', 'edges', 'orientation', 'size'],
    'Triangle':           ['basic', 'closure', 'edges', 'orientation', 'size'],
    'Diagonal':           ['basic', 'closure', 'edges', 'orientation', 'size'],
    'Star':               ['basic', 'closure', 'edges', 'orientation', 'size'],
    'Overlapped pencils': ['basic', 'closure', 'edges', 'orientation', 'overlap', 'size'],
}
ITEMS = list(FACETS)                           # archive folder names in BOT-2 administration order
ITEM_LABEL = {'Circle': 'Circle', 'Square': 'Square', 'Overlapped circle': 'Overlapped circles', 'Wave': 'Wave',
              'Triangle': 'Triangle', 'Diagonal': 'Diamond', 'Star': 'Star', 'Overlapped pencils': 'Overlapped pencils'}
CRITERIA = ['basic', 'closure', 'edges', 'orientation', 'overlap', 'size']
CRITERION_LABEL = {'basic': 'Basic shape', 'closure': 'Closure', 'edges': 'Edges', 'orientation': 'Orientation',
                   'overlap': 'Overlap', 'size': 'Overall size'}
MAX_SCORE = {s: len(f) for s, f in FACETS.items()}
ITEM_ID = {s: i for i, s in enumerate(ITEMS)}
N_ITEMS, N_CLS = len(ITEMS), max(MAX_SCORE.values()) + 1
K_OF_ITEM = np.array([MAX_SCORE[s] for s in ITEMS])
VALID = np.arange(N_CLS)[None, :] <= K_OF_ITEM[:, None]
PATTERN = re.compile(r'^img(\d+)-([a-z]+)-(\d+)(?:\((\d+)\))?\.png$', re.I)

N_FOLDS, N_INNER, FOLD_SEED = 5, 8, 2026
INPUT_SIZE = 512


def parse_facets(score, code, k):
    """Criterion marks as a string of 0 and 1 in rubric order. A score of 0 or full marks fixes every mark; other
    scores need the code in the file name, which must have one digit per criterion and sum to the score."""
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
                dropped.append((ITEM_LABEL[item], f, 'file name')); continue
            num, score = int(m.group(1)), int(m.group(3))
            if score > k:
                dropped.append((ITEM_LABEL[item], f, 'score above the maximum')); continue
            p = os.path.join(root, item, f)
            rows.append(dict(file=f, path=p, item=item, num=num, score=score, facets=parse_facets(score, m.group(4), k),
                             md5=hashlib.md5(open(p, 'rb').read()).hexdigest()))
    df = pd.DataFrame(rows)
    dup = df.md5.duplicated(keep=False)
    dropped += [(ITEM_LABEL[i], f, 'byte-identical copy with a different score') for i, f in zip(df['item'][dup], df.file[dup])]
    df = df[~dup].drop(columns='md5').reset_index(drop=True)
    order = df['item'].map(ITEMS.index)
    seq = pd.DataFrame({'num': df.num, 'order': order}).sort_values(['num', 'order'])
    start = (seq.order.diff() <= 0) | (seq.num.diff() > len(ITEMS))
    df['child'] = start.cumsum().reindex(df.index)
    return df, pd.DataFrame(dropped, columns=['Item', 'File', 'Reason'])


def marks_array(df):
    """Examiner's criterion marks with one column per criterion in CRITERIA: 1 passed, 0 failed, NaN when the mark is
    unknown or the criterion is not part of the item."""
    M = np.full((len(df), len(CRITERIA)), np.nan, np.float32)
    for r, (item, code) in enumerate(zip(df['item'], df.facets)):
        for c, b in zip(FACETS[item], code):
            M[r, CRITERIA.index(c)] = int(b)
    return M


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
    the remaining children, used only to select the checkpoint of a network and the scorer of a paradigm."""
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
        log('Building the network input cache (first run only) ...')
        imgs = np.stack(parallel_map(partial(to_input, size=size), df.path.tolist(), workers))
        atomic_save(cache, lambda p: np.savez_compressed(p, x=imgs, files=np.array(df.file.tolist(), dtype=str)))

    D = SimpleNamespace(df=df, dropped=dropped, imgs=imgs, item=df['item'].map(ITEM_ID).values,
                        score=df.score.values.astype(int), child=df.child.values, fold=df.fold.values,
                        marks=marks_array(df), strata=(df['item'] + '_' + df.score.astype(str)).values,
                        n_folds=N_FOLDS, size=size, tag=tag)
    D.kmax = K_OF_ITEM[D.item]
    log(f'{len(df)} drawings of {df.child.nunique()} inferred children, {len(dropped)} files excluded, '
        f'{N_FOLDS} child-grouped folds, network inputs {imgs.shape[1]} x {imgs.shape[2]} px')
    return D


def class_counts(items, scores):
    cnt = np.zeros((N_ITEMS, N_CLS))
    np.add.at(cnt, (items, scores), 1)
    return cnt


def score_table(D, P=None):
    """Drawings per item and expert score, with the share at full marks."""
    labels = [ITEM_LABEL[s] for s in ITEMS]
    t = pd.crosstab(D.df['item'].map(ITEM_LABEL), D.df.score).reindex(index=labels, columns=range(N_CLS), fill_value=0)
    t.loc['All items'] = t.sum(0)
    t.columns = [f'Score {c}' for c in t.columns]
    t['Drawings'] = t.sum(1)
    full = [t.loc[ITEM_LABEL[s], f'Score {MAX_SCORE[s]}'] / t.loc[ITEM_LABEL[s], 'Drawings'] for s in ITEMS]
    t['Full marks (%)'] = [100 * f for f in full] + [100 * float(np.mean(D.score == D.kmax))]
    t.insert(0, 'Maximum score', [str(MAX_SCORE[s]) for s in ITEMS] + [''])
    t.index.name = 'Item'
    if P is not None:
        save_csv(t, os.path.join(P.data, 'score_distribution.csv'))
    return t.round(1)


def criterion_table(D, P=None):
    """Share (%) of drawings that fail each rubric criterion. A score of 0 means that the basic shape failed and sets
    every other criterion to 0, so the other criteria are counted among drawings scored above 0. Drawings whose
    criterion marks are unknown are left out."""
    rows, counts = [], []
    for i, s in enumerate(ITEMS):
        row = {'Item': ITEM_LABEL[s]}
        for c in CRITERIA:
            if c not in FACETS[s]:
                row[CRITERION_LABEL[c]] = np.nan
                continue
            m = (D.item == i) & (D.score >= 0 if c == 'basic' else D.score > 0)
            v = D.marks[m, CRITERIA.index(c)]
            v = v[~np.isnan(v)]
            row[CRITERION_LABEL[c]] = 100 * float(np.mean(v == 0)) if len(v) else np.nan
            counts.append({'Item': ITEM_LABEL[s], 'Criterion': CRITERION_LABEL[c], 'Drawings with a known mark': len(v),
                           'Failed': int((v == 0).sum())})
        rows.append(row)
    t = pd.DataFrame(rows).set_index('Item')
    if P is not None:
        save_csv(t, os.path.join(P.data, 'criterion_failures.csv'))
        save_csv(pd.DataFrame(counts), os.path.join(P.data, 'criterion_failure_counts.csv'), index=False)
    return t.round(1)


def fold_table(D, P=None):
    rows = []
    for k in range(D.n_folds):
        tr, va, te = split(D, k)
        rows.append({'Fold': k, 'Training drawings': len(tr), 'Validation drawings': len(va), 'Test drawings': len(te),
                     'Training children': len(np.unique(D.child[tr])), 'Validation children': len(np.unique(D.child[va])),
                     'Test children': len(np.unique(D.child[te])),
                     'Test drawings below full marks (%)': 100 * np.mean(D.score[te] < D.kmax[te])})
    t = pd.DataFrame(rows).set_index('Fold')
    if P is not None:
        save_csv(t, os.path.join(P.data, 'folds.csv'))
    return t.round(1)
