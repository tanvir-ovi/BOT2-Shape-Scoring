"""Out-of-fold predictions, metrics, paired child-level statistics and the embedding analysis."""
import os

import numpy as np
import pandas as pd

from .data import ITEMS, ITEM_NAME, MAX_SCORE, N_ITEMS, N_CLS
from .models import NETWORKS, ENSEMBLE, CLASSICAL, PROPOSED, systems, label, group
from .training import unit_paths, unit_done, signature
from .utils import atomic_save, save_csv

N_BOOT = 10000          # child-level bootstrap and permutation resamples
MARGIN = 1.0            # equivalence margin, accuracy points
SEED = 2026
FAMILIES = {            # fixed before the runs; Holm correction within each family
    'RQ1': [(PROPOSED, b) for b in ('majority', 'geometric_gb', 'resnet50', 'efficientnetv2_s', 'vit_b16', 'cvit',
                                     'convnext_512', 'dinov2_336', 'dit_224')],
    'RQ2': [('dit_224', 'beit_natural'), ('beit_sketch', 'beit_natural'), ('dit_224', 'beit_sketch'),
            ('convnext_512', 'convnext_224')],
}


# ---------------------------------------------------------------------------------------------------
# Out-of-fold predictions
# ---------------------------------------------------------------------------------------------------
def collect(D, P, log=print):
    """Test-fold probabilities of every scorer for all drawings, plus the proposed ensemble (mean of its members)."""
    n, R = len(D.score), {}
    for key in list(NETWORKS) + list(CLASSICAL):
        q, found = np.full((n, N_CLS), np.nan, np.float32), 0
        for k in range(D.n_folds):
            paths = unit_paths(P, key, k)
            ok = unit_done(paths, signature(key, NETWORKS[key], D, k)) if key in NETWORKS else os.path.exists(paths.npz)
            if ok:
                z = np.load(paths.npz)
                q[z['te_idx']] = z['q_te']
                found += 1
        if found == D.n_folds:
            R[key] = q
        else:
            log(f'{label(key)}: {found} of {D.n_folds} folds finished, left out of the analysis')
    if all(m in R for m in ENSEMBLE):
        R[PROPOSED] = np.mean([R[m] for m in ENSEMBLE], 0)
    R = {k: R[k] for k in systems() if k in R}

    df = D.df[['file', 'item', 'child', 'fold', 'score']].copy()
    df['item'] = df['item'].map(ITEM_NAME)
    for key, q in R.items():
        df[f'pred_{key}'] = q.argmax(1)
        df[f'p_below_full_{key}'] = 1 - q[np.arange(n), D.kmax]
    save_csv(df, os.path.join(P.predictions, 'oof_predictions.csv'), index=False)
    atomic_save(os.path.join(P.predictions, 'oof_probabilities.npz'),
                lambda p: np.savez_compressed(p, files=np.array(D.df.file.tolist(), dtype=str), **R))
    log(f'Out-of-fold predictions of {len(R)} scorers for {n} drawings')
    return R


# ---------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------
def qwk(y, p, n=N_CLS):
    """Quadratic weighted kappa on the full integer scale (scores absent from y and p still count)."""
    y, p = np.asarray(y, int), np.asarray(p, int)
    O = np.bincount(y * n + p, minlength=n * n).reshape(n, n).astype(float)
    tot = O.sum()
    if tot == 0:
        return np.nan
    E = np.outer(O.sum(1), O.sum(0)) / tot
    i = np.arange(n)
    W = (i[:, None] - i[None, :]) ** 2
    den = (W * E).sum()
    return 1 - (W * O).sum() / den if den > 0 else np.nan


def mean_item_accuracy(y, p, items):
    return float(np.mean([np.mean(y[items == i] == p[items == i]) for i in range(N_ITEMS) if (items == i).any()]))


def mean_item_qwk(y, p, items):
    v = [qwk(y[items == i], p[items == i]) for i in range(N_ITEMS) if (items == i).any()]
    return np.nan if np.all(np.isnan(v)) else float(np.nanmean(v))


def score_metrics(y, p, kmax):
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    below = y < kmax
    return {'Accuracy': np.mean(y == p), 'Balanced accuracy': balanced_accuracy_score(y, p), 'Macro precision': mp,
            'Macro recall': mr, 'Macro F1': mf, 'Weighted precision': wp, 'Weighted recall': wr, 'Weighted F1': wf,
            'QWK': qwk(y, p), 'MAE': np.mean(np.abs(y - p)), 'Within 1 point': np.mean(np.abs(y - p) <= 1),
            'Recall below full marks': np.mean(p[below] < kmax) if below.any() else np.nan}


def _below_full(D, q):
    return D.score < D.kmax, 1 - q[np.arange(len(q)), D.kmax]


def _children(D):
    _, inv = np.unique(D.child, return_inverse=True)
    return inv, inv.max() + 1


def _cells(D, v):
    """Sums of v and counts per (child, item) cell."""
    c, C = _children(D)
    S, N = np.zeros((C, N_ITEMS)), np.zeros((C, N_ITEMS))
    np.add.at(S, (c, D.item), v)
    np.add.at(N, (c, D.item), 1)
    return S, N


def _bootstrap(S, N, B=N_BOOT, seed=SEED):
    """Mean over items of sum(v) / count under resampling of children with replacement."""
    rng, C, out = np.random.default_rng(seed), len(S), []
    for start in range(0, B, 1000):
        W = rng.multinomial(C, np.full(C, 1 / C), size=min(1000, B - start)).astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            out.append(np.nanmean((W @ S) / (W @ N), 1))
    return np.concatenate(out)


def _permutation_p(S, N, B=N_BOOT, seed=SEED + 1):
    """Two-sided p of a paired sign-flip test: the two scorers' predictions are swapped for all drawings of a child."""
    rng, C = np.random.default_rng(seed), len(S)
    tot = N.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        obs = abs(np.nanmean(S.sum(0) / tot))
        hits = 0
        for start in range(0, B, 1000):
            signs = rng.choice([-1.0, 1.0], size=(min(1000, B - start), C))
            hits += int((np.abs(np.nanmean((signs @ S) / tot, 1)) >= obs - 1e-12).sum())
    return (hits + 1) / (B + 1)


def mcnemar_exact(y, pa, pb):
    from scipy.stats import binomtest
    a, b = int(((pa == y) & (pb != y)).sum()), int(((pa != y) & (pb == y)).sum())
    return a, b, (binomtest(a, a + b, 0.5).pvalue if a + b else 1.0)


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def summary(D, R, P=None):
    """One row per scorer: mean per-item accuracy with its child-level 95% interval and fold SD, agreement measures,
    and detection of drawings below full marks."""
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support
    rows = []
    for key, q in R.items():
        p = q.argmax(1)
        correct = (p == D.score).astype(float)
        S, N = _cells(D, correct)
        lo, hi = np.percentile(_bootstrap(S, N), [2.5, 97.5])
        per_item = pd.DataFrame({ITEM_NAME[s]: score_metrics(D.score[D.item == i], p[D.item == i], MAX_SCORE[s])
                                 for i, s in enumerate(ITEMS)}).T
        folds = [mean_item_accuracy(D.score[D.fold == k], p[D.fold == k], D.item[D.fold == k]) for k in range(D.n_folds)]
        yb, sb = _below_full(D, q)
        bp, br, bf, _ = precision_recall_fscore_support(yb, p < D.kmax, average='binary', zero_division=0)
        rows.append({'Key': key, 'Scorer': label(key), 'Role': group(key),
                     'Accuracy (%)': 100 * per_item.Accuracy.mean(), 'CI low': 100 * lo, 'CI high': 100 * hi,
                     'Fold SD': 100 * np.std(folds, ddof=1), 'Pooled accuracy (%)': 100 * correct.mean(),
                     'Balanced accuracy': per_item['Balanced accuracy'].mean(), 'Macro F1': per_item['Macro F1'].mean(),
                     'Weighted F1': per_item['Weighted F1'].mean(), 'QWK': per_item.QWK.mean(),
                     'MAE': np.mean(np.abs(D.score - p)), 'Precision below full': bp, 'Recall below full': br,
                     'F1 below full': bf, 'AP below full': average_precision_score(yb, sb)})
    t = pd.DataFrame(rows).set_index('Key')
    if P is not None:
        save_csv(t, os.path.join(P.metrics, 'summary.csv'))
    return t


def item_accuracy(D, R, P=None):
    t = pd.DataFrame({label(key): {ITEM_NAME[s]: 100 * np.mean(q.argmax(1)[D.item == i] == D.score[D.item == i])
                                   for i, s in enumerate(ITEMS)} for key, q in R.items()}).T
    t['Mean'] = t.mean(1)
    if P is not None:
        save_csv(t, os.path.join(P.metrics, 'accuracy_per_item.csv'))
    return t


def fold_accuracy(D, R, P=None):
    t = pd.DataFrame({label(key): {f'Fold {k}': 100 * mean_item_accuracy(D.score[D.fold == k], q.argmax(1)[D.fold == k],
                                                                         D.item[D.fold == k]) for k in range(D.n_folds)}
                      for key, q in R.items()}).T
    t['Mean'], t['SD'] = t.mean(1), t.iloc[:, :D.n_folds].std(1, ddof=1)
    if P is not None:
        save_csv(t, os.path.join(P.metrics, 'accuracy_per_fold.csv'))
    return t


def item_metrics(D, R, P=None):
    """Precision, recall and F1 (macro and weighted), accuracy, QWK and MAE for every scorer and item."""
    rows = []
    for key, q in R.items():
        p = q.argmax(1)
        for i, s in enumerate(ITEMS):
            m = D.item == i
            rows.append({'Scorer': label(key), 'Item': ITEM_NAME[s], 'Drawings': int(m.sum()),
                         **score_metrics(D.score[m], p[m], MAX_SCORE[s])})
    t = pd.DataFrame(rows)
    if P is not None:
        save_csv(t, os.path.join(P.metrics, 'metrics_per_item.csv'), index=False)
    return t


def confusion(D, R, P=None):
    rows = []
    for key, q in R.items():
        p = q.argmax(1)
        for i, s in enumerate(ITEMS):
            m = D.item == i
            cm = np.zeros((MAX_SCORE[s] + 1,) * 2, int)
            np.add.at(cm, (D.score[m], p[m]), 1)
            rows += [{'Scorer': label(key), 'Item': ITEM_NAME[s], 'Expert score': a, 'Predicted score': b,
                      'Drawings': int(cm[a, b])} for a in range(len(cm)) for b in range(len(cm))]
    t = pd.DataFrame(rows)
    if P is not None:
        save_csv(t, os.path.join(P.metrics, 'confusion_matrices.csv'), index=False)
    return t


def below_full_curves(D, R, P=None):
    """Precision-recall curves for detecting drawings below full marks, pooled over items."""
    from sklearn.metrics import precision_recall_curve
    curves = {}
    for key, q in R.items():
        if key == 'majority':
            continue
        yb, sb = _below_full(D, q)
        prec, rec, _ = precision_recall_curve(yb, sb)
        curves[key] = (rec, prec)
    if P is not None:
        rows = [{'Scorer': label(k), 'Recall': r, 'Precision': p} for k, (rr, pp) in curves.items() for r, p in zip(rr, pp)]
        save_csv(pd.DataFrame(rows), os.path.join(P.metrics, 'precision_recall_below_full.csv'), index=False)
    return curves


# ---------------------------------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------------------------------
def compare(D, R, P=None, families=None):
    """Paired differences in mean per-item accuracy (A minus B) on the same out-of-fold drawings. Intervals and p
    values resample whole inferred children. A pair is 'different' when the Holm-adjusted permutation p is below 0.05
    and the 95% interval excludes zero, 'equivalent' when the 90% interval lies within the margin."""
    out = []
    for fam, pairs in (families or FAMILIES).items():
        rows = []
        for a, b in pairs:
            if a not in R or b not in R:
                continue
            pa, pb = R[a].argmax(1), R[b].argmax(1)
            S, N = _cells(D, (pa == D.score).astype(float) - (pb == D.score).astype(float))
            with np.errstate(invalid='ignore', divide='ignore'):
                delta = 100 * np.nanmean(S.sum(0) / N.sum(0))
            boot = 100 * _bootstrap(S, N)
            only_a, only_b, p_mc = mcnemar_exact(D.score, pa, pb)
            rows.append({'Family': fam, 'A': label(a), 'B': label(b), 'Delta (points)': delta,
                         'CI95 low': np.percentile(boot, 2.5), 'CI95 high': np.percentile(boot, 97.5),
                         'CI90 low': np.percentile(boot, 5), 'CI90 high': np.percentile(boot, 95),
                         'p (permutation)': _permutation_p(S, N), 'Only A correct': only_a, 'Only B correct': only_b,
                         'p (McNemar)': p_mc})
        if not rows:
            continue
        t = pd.DataFrame(rows)
        t['p (Holm)'] = holm(t['p (permutation)'])
        excl = (t['CI95 low'] > 0) | (t['CI95 high'] < 0)
        equiv = (t['CI90 low'] > -MARGIN) & (t['CI90 high'] < MARGIN)
        t['Verdict'] = np.where((t['p (Holm)'] < 0.05) & excl, 'different', np.where(equiv, 'equivalent', 'inconclusive'))
        out.append(t)
    t = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if P is not None:
        save_csv(t, os.path.join(P.statistics, 'paired_comparisons.csv'), index=False)
    return t


# ---------------------------------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------------------------------
def oof_embedding(D, P, key):
    E = None
    for k in range(D.n_folds):
        z = np.load(unit_paths(P, key, k).npz)
        if E is None:
            E = np.zeros((len(D.score), z['emb_te'].shape[1]), np.float16)
        E[z['te_idx']] = z['emb_te']
    return E


def tsne(X, seed=SEED):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = PCA(min(50, X.shape[1], len(X) - 1), random_state=seed).fit_transform(X)
    return TSNE(2, perplexity=min(30.0, (len(X) - 1) / 3), init='pca', learning_rate='auto',
                random_state=seed).fit_transform(X)


def separation(D, X, seed=SEED):
    """Separation of drawings at full marks from those below, measured in the embedding space before projection:
    leave-one-out 10-nearest-neighbour balanced accuracy within each item (mean over items), and the cosine
    silhouette of the two groups."""
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    X = X.astype(np.float32)
    below = D.score < D.kmax
    bal = []
    for i in range(N_ITEMS):
        m = np.where(D.item == i)[0]
        if below[m].all() or (~below[m]).all() or len(m) < 12:
            continue
        nn = NearestNeighbors(n_neighbors=11, metric='cosine').fit(X[m])
        nb = nn.kneighbors(X[m], return_distance=False)[:, 1:]
        vote = below[m][nb].mean(1) > 0.5
        y = below[m]
        bal.append(0.5 * (np.mean(vote[y]) + np.mean(~vote[~y])))
    rng = np.random.default_rng(seed)
    s = rng.choice(len(X), min(2000, len(X)), replace=False)
    return {'kNN balanced accuracy': float(np.mean(bal)) if bal else np.nan,
            'Silhouette (cosine)': float(silhouette_score(X[s], below[s], metric='cosine'))}


def embeddings(D, R, P, log=print):
    """Stores the out-of-fold embedding of every network and projects three of them with t-SNE: the pretrained
    ConvNeXt V2 before fine-tuning, the fine-tuned ConvNeXt V2 and the concatenated members of the ensemble."""
    E = {key: oof_embedding(D, P, key) for key in NETWORKS if key in R}
    for key, e in E.items():
        atomic_save(os.path.join(P.embeddings, f'{key}_oof{D.tag}.npy'), lambda p: np.save(p, e))
    panels = []
    pre = os.path.join(P.embeddings, f'convnext_512_pretrained{D.tag}.npy')
    if os.path.exists(pre):
        panels.append(('pretrained', 'ConvNeXt V2-T before fine-tuning', np.load(pre)))
    if 'convnext_512' in E:
        panels.append(('convnext_512', 'ConvNeXt V2-T fine-tuned', E['convnext_512']))
    if all(m in E for m in ENSEMBLE):
        panels.append((PROPOSED, 'DPE members concatenated', np.hstack([E[m] for m in ENSEMBLE])))
    coords, sep = [], []
    for key, title, X in panels:
        xy = tsne(X.astype(np.float32))
        coords.append(pd.DataFrame({'Panel': key, 'Title': title, 'file': D.df.file.values, 'Item': D.df['item'].map(ITEM_NAME),
                                    'Below full marks': D.score < D.kmax, 'x': xy[:, 0], 'y': xy[:, 1]}))
        sep.append({'Panel': key, 'Embedding': title, 'Dimensions': X.shape[1], **separation(D, X)})
        log(f't-SNE done: {title}')
    coords = pd.concat(coords, ignore_index=True) if coords else pd.DataFrame()
    sep = pd.DataFrame(sep).set_index('Panel') if sep else pd.DataFrame()
    save_csv(coords, os.path.join(P.embeddings, 'tsne_coordinates.csv'), index=False)
    save_csv(sep, os.path.join(P.metrics, 'embedding_separation.csv'))
    return {'coords': coords, 'separation': sep}
