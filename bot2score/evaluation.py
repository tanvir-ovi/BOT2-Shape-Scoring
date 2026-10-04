"""Out-of-fold test predictions of every scorer, accuracy and agreement with the expert scores, paired child-level
statistics, detection of drawings below full marks and the embedding analysis. Every drawing is scored once, by the
scorer trained without the fold that holds it. Unless a name says otherwise, a metric is the mean of its values on the
eight items."""
import os

import numpy as np
import pandas as pd

from .data import ITEMS, ITEM_LABEL, MAX_SCORE, N_ITEMS, N_CLS
from .metrics import kappa, kappa_from_counts, binary_kappa, cramers_v, mean_item_accuracy, band
from .models import NETWORKS
from .scorers import (PAIRS, SELECTED, ENSEMBLE, PROPOSED, NETWORK_KEYS, CLASSIFIER_KEYS, individual, label, paradigm,
                      role)
from .training import unit_paths, unit_done, signature
from .utils import atomic_save, save_csv

N_BOOT = 10000          # child-level bootstrap and permutation resamples
MARGIN = 1.0            # equivalence margin, accuracy points
SEED = 2026
FAMILIES = {            # fixed before the run; Holm correction within each family
    'Paradigms':   [('best_network', 'best_classifier'), ('best_network', 'rules'), ('best_classifier', 'rules')],
    'Proposed':    [(PROPOSED, b) for b in individual() if b != PROPOSED],
    'Ensemble':    [(PROPOSED, p) for p in PAIRS],
    'Pretraining': [('dit_224', 'beit_natural'), ('beit_sketch', 'beit_natural'), ('dit_224', 'beit_sketch'),
                    ('convnext_512', 'convnext_224')],
}
_W = {}


# ---------------------------------------------------------------------------------------------------
# Out-of-fold predictions
# ---------------------------------------------------------------------------------------------------
def _fold_file(P, key, D, k):
    if key in NETWORKS:
        paths = unit_paths(P, key, k)
        return paths.npz if unit_done(paths, signature(key, dict(NETWORKS[key]), D, k)) else None
    f = os.path.join(P.runs, key, f'fold{k}.npz')
    return f if os.path.exists(f) else None


def collect(D, P, log=print):
    """Test-fold probabilities of every scorer for all drawings; the proposed ensemble and the ensembles without one
    member (means of member probabilities); and, in each fold, the network and the classifier with the highest
    validation accuracy."""
    n, R, val = len(D.score), {}, {}
    for key in individual():
        if key == PROPOSED:
            continue
        q, acc = np.full((n, N_CLS), np.nan, np.float32), {}
        for k in range(D.n_folds):
            f = _fold_file(P, key, D, k)
            if f is None:
                continue
            z = np.load(f)
            q[z['te_idx']] = z['q_te']
            acc[k] = mean_item_accuracy(D.score[z['va_idx']], z['q_va'].argmax(1), D.item[z['va_idx']])
        if len(acc) == D.n_folds:
            R[key], val[key] = q, acc
        else:
            log(f'{label(key)}: {len(acc)} of {D.n_folds} folds finished; left out of the analysis')
    if all(m in R for m in ENSEMBLE):
        R[PROPOSED] = np.mean([R[m] for m in ENSEMBLE], 0)
        for key, (_, members) in PAIRS.items():
            R[key] = np.mean([R[m] for m in members], 0)
    rows = []
    for key, pool in (('best_network', NETWORK_KEYS), ('best_classifier', CLASSIFIER_KEYS)):
        pool = [m for m in pool if m in val]
        if not pool:
            continue
        q = np.full((n, N_CLS), np.nan, np.float32)
        for k in range(D.n_folds):
            pick = max(pool, key=lambda m: val[m][k])
            te = np.where(D.fold == k)[0]
            q[te] = R[pick][te]
            rows.append({'Paradigm': paradigm(key), 'Fold': k, 'Chosen scorer': label(pick),
                         'Validation accuracy (%)': 100 * val[pick][k]})
        R[key] = q
    save_csv(pd.DataFrame(rows), os.path.join(P.metrics, 'paradigm_selection.csv'), index=False)
    R = {k: R[k] for k in individual() + list(PAIRS) + list(SELECTED) if k in R}

    df = D.df[['file', 'item', 'child', 'fold', 'score']].copy()
    df['item'] = df['item'].map(ITEM_LABEL)
    for key, q in R.items():
        df[f'pred_{key}'] = q.argmax(1)
    save_csv(df, os.path.join(P.predictions, 'oof_predictions.csv'), index=False)
    atomic_save(os.path.join(P.predictions, 'oof_probabilities.npz'),
                lambda p: np.savez_compressed(p, files=np.array(D.df.file.tolist(), dtype=str), **R))
    log(f'Out-of-fold test predictions for {n} drawings: {sum(k in R for k in individual())} scorers, '
        f'{sum(k in R for k in PAIRS)} ensembles without one member, {sum(k in R for k in SELECTED)} validation-selected')
    return R


def selection(P):
    """The network and the classifier chosen on the validation part of each fold."""
    t = pd.read_csv(os.path.join(P.metrics, 'paradigm_selection.csv'))
    t['Chosen on validation'] = t['Chosen scorer'] + ' (' + t['Validation accuracy (%)'].map('{:.2f}%'.format) + ')'
    return t.pivot(index='Fold', columns='Paradigm', values='Chosen on validation')


# ---------------------------------------------------------------------------------------------------
# Child-level resampling
# ---------------------------------------------------------------------------------------------------
def _children(D):
    _, inv = np.unique(D.child, return_inverse=True)
    return inv, inv.max() + 1


def _weights(C):
    """Multinomial counts of 10,000 resamples of the C children (the same resamples for every scorer)."""
    if C not in _W:
        rng = np.random.default_rng(SEED)
        _W[C] = np.concatenate([rng.multinomial(C, np.full(C, 1 / C), size=min(1000, N_BOOT - s)).astype(np.float32)
                                for s in range(0, N_BOOT, 1000)])
    return _W[C]


def _cells(D, v):
    """Sums of v and counts per (child, item) cell."""
    c, C = _children(D)
    S, N = np.zeros((C, N_ITEMS)), np.zeros((C, N_ITEMS))
    np.add.at(S, (c, D.item), v)
    np.add.at(N, (c, D.item), 1)
    return S, N


def _bootstrap(S, N):
    """Mean over items of sum(v) / count for every resample of children."""
    W = _weights(len(S))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.nanmean((W @ S) / (W @ N), 1)


def _kappa_boot(D, p, weights=None):
    """Kappa (or QWK) of every item for every resample of children: array (resamples, items)."""
    c, C = _children(D)
    W = _weights(C)
    out = np.full((N_BOOT, N_ITEMS), np.nan)
    for i in range(N_ITEMS):
        m = D.item == i
        M = np.zeros((C, N_CLS * N_CLS), np.float32)
        np.add.at(M, (c[m], D.score[m] * N_CLS + p[m]), 1)
        out[:, i] = kappa_from_counts((W @ M).reshape(-1, N_CLS, N_CLS), weights)
    return out


def _permutation_p(S, N, seed=SEED + 1):
    """Two-sided p of a paired sign-flip test: the two scorers' predictions are swapped for all drawings of a child."""
    rng, C = np.random.default_rng(seed), len(S)
    tot = N.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        obs = abs(np.nanmean(S.sum(0) / tot))
        hits = 0
        for start in range(0, N_BOOT, 1000):
            signs = rng.choice([-1.0, 1.0], size=(min(1000, N_BOOT - start), C))
            hits += int((np.abs(np.nanmean((signs @ S) / tot, 1)) >= obs - 1e-12).sum())
    return (hits + 1) / (N_BOOT + 1)


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


# ---------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------
def item_scores(y, p, kmax):
    """Metrics of one item: y expert scores, p predicted scores, kmax the full marks of the item."""
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    below = y < kmax
    k = kappa(y, p)
    return {'Drawings': len(y), 'Accuracy': np.mean(y == p), 'Balanced accuracy': balanced_accuracy_score(y, p),
            'Macro precision': mp, 'Macro recall': mr, 'Macro F1': mf, 'Weighted F1': wf, 'Kappa': k,
            'QWK': kappa(y, p, 'quadratic'), "Cramér's V": cramers_v(y, p),
            'Full-marks kappa': binary_kappa(y == kmax, p == kmax), 'Within one point': np.mean(np.abs(y - p) <= 1),
            'MAE': np.mean(np.abs(y - p)), 'Recall below full marks': np.mean(p[below] < kmax) if below.any() else np.nan,
            'Agreement (kappa)': band(k)}


def _per_item(D, p):
    return pd.DataFrame({ITEM_LABEL[s]: item_scores(D.score[D.item == i], p[D.item == i], MAX_SCORE[s])
                         for i, s in enumerate(ITEMS)}).T.infer_objects()


def _below_full(D, q):
    return D.score < D.kmax, 1 - q[np.arange(len(q)), D.kmax]


def _fold_accuracies(D, p):
    return [mean_item_accuracy(D.score[D.fold == k], p[D.fold == k], D.item[D.fold == k]) for k in range(D.n_folds)]


def summary(D, R, P=None):
    """One row per scorer: accuracy with its child-level 95% interval and fold SD, accuracy over all drawings,
    imbalance-aware metrics, agreement, and the detection of drawings below full marks (pooled over items)."""
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support
    rows = []
    for key in [k for k in individual() if k in R]:
        q = R[key]
        p = q.argmax(1)
        correct = (p == D.score).astype(float)
        lo, hi = np.percentile(_bootstrap(*_cells(D, correct)), [2.5, 97.5])
        per = _per_item(D, p)
        num = lambda c: per[c].astype(float).mean()
        yb, sb = _below_full(D, q)
        bp, br, bf, _ = precision_recall_fscore_support(yb, p < D.kmax, average='binary', zero_division=0)
        rows.append({'Key': key, 'Scorer': label(key), 'Paradigm': paradigm(key), 'Role': role(key),
                     'Accuracy (%)': 100 * num('Accuracy'), 'CI low': 100 * lo, 'CI high': 100 * hi,
                     'Fold SD': 100 * np.std(_fold_accuracies(D, p), ddof=1),
                     'Accuracy, all drawings (%)': 100 * correct.mean(), 'Balanced accuracy': num('Balanced accuracy'),
                     'Macro F1': num('Macro F1'), 'Weighted F1': num('Weighted F1'), 'MAE': num('MAE'),
                     'Within one point (%)': 100 * num('Within one point'), 'Kappa': num('Kappa'), 'QWK': num('QWK'),
                     'Full-marks kappa': num('Full-marks kappa'), 'Precision below full': bp, 'Recall below full': br,
                     'F1 below full': bf, 'AP below full': average_precision_score(yb, sb)})
    t = pd.DataFrame(rows).set_index('Key')
    if P is not None:
        save_csv(t, os.path.join(P.metrics, 'summary.csv'))
    return t


def agreement(D, R, P=None):
    """Agreement with the expert: Cohen's kappa and QWK with child-level 95% intervals, the kappa of the decision
    'full marks or not', the share within one point, and the number of items whose kappa is above 0.60
    (substantial or better on the Landis and Koch scale)."""
    rows = []
    for key in [k for k in individual() if k in R]:
        p = R[key].argmax(1)
        per = _per_item(D, p)
        kb, qb = np.nanmean(_kappa_boot(D, p), 1), np.nanmean(_kappa_boot(D, p, 'quadratic'), 1)
        k_items = per['Kappa'].astype(float)
        rows.append({'Key': key, 'Scorer': label(key), 'Paradigm': paradigm(key),
                     'Kappa': k_items.mean(), 'Kappa CI low': np.nanpercentile(kb, 2.5),
                     'Kappa CI high': np.nanpercentile(kb, 97.5), 'QWK': per['QWK'].astype(float).mean(),
                     'QWK CI low': np.nanpercentile(qb, 2.5), 'QWK CI high': np.nanpercentile(qb, 97.5),
                     'Full-marks kappa': per['Full-marks kappa'].astype(float).mean(),
                     'Within one point (%)': 100 * per['Within one point'].astype(float).mean(),
                     'Items with kappa above 0.60': int((k_items > 0.60).sum()), 'Agreement (kappa)': band(k_items.mean())})
    t = pd.DataFrame(rows).set_index('Key')
    if P is not None:
        save_csv(t, os.path.join(P.metrics, 'agreement.csv'))
    return t


def item_agreement(D, R, P=None, keys=(PROPOSED, 'best_network', 'best_classifier', 'rules')):
    """Kappa and QWK of every item with child-level 95% intervals, for the proposed ensemble and the scorer chosen
    to represent each paradigm."""
    rows = []
    for key in [k for k in keys if k in R]:
        p = R[key].argmax(1)
        kb, qb = _kappa_boot(D, p), _kappa_boot(D, p, 'quadratic')
        for i, s in enumerate(ITEMS + ['Mean']):
            if s == 'Mean':
                kv, qv = np.nanmean([r['Kappa'] for r in rows[-N_ITEMS:]]), np.nanmean([r['QWK'] for r in rows[-N_ITEMS:]])
                kd, qd = np.nanmean(kb, 1), np.nanmean(qb, 1)
                name = 'Mean'
            else:
                m = D.item == i
                kv, qv = kappa(D.score[m], p[m]), kappa(D.score[m], p[m], 'quadratic')
                kd, qd = kb[:, i], qb[:, i]
                name = ITEM_LABEL[s]
            rows.append({'Key': key, 'Scorer': label(key), 'Item': name, 'Kappa': kv,
                         'Kappa CI low': np.nanpercentile(kd, 2.5), 'Kappa CI high': np.nanpercentile(kd, 97.5),
                         'QWK': qv, 'QWK CI low': np.nanpercentile(qd, 2.5), 'QWK CI high': np.nanpercentile(qd, 97.5),
                         'Agreement (kappa)': band(kv)})
    t = pd.DataFrame(rows)
    if P is not None:
        save_csv(t, os.path.join(P.metrics, 'agreement_per_item.csv'), index=False)
    return t


def item_metrics(D, R, P=None):
    """Every metric of every scorer on every item."""
    rows = []
    for key in [k for k in individual() if k in R]:
        per = _per_item(D, R[key].argmax(1))
        per.insert(0, 'Scorer', label(key))
        rows.append(per.rename_axis('Item').reset_index())
    t = pd.concat(rows, ignore_index=True)
    if P is not None:
        save_csv(t, os.path.join(P.metrics, 'metrics_per_item.csv'), index=False)
    return t


def item_accuracy(D, R, P=None):
    t = pd.DataFrame({label(key): {ITEM_LABEL[s]: 100 * np.mean(R[key].argmax(1)[D.item == i] == D.score[D.item == i])
                                   for i, s in enumerate(ITEMS)} for key in individual() if key in R}).T
    t['Mean'] = t.mean(1)
    if P is not None:
        save_csv(t, os.path.join(P.metrics, 'accuracy_per_item.csv'))
    return t


def fold_accuracy(D, R, P=None):
    t = pd.DataFrame({label(key): dict(zip([f'Fold {k}' for k in range(D.n_folds)],
                                           100 * np.array(_fold_accuracies(D, R[key].argmax(1)))))
                      for key in individual() if key in R}).T
    t['Mean'], t['SD'] = t.mean(1), t.iloc[:, :D.n_folds].std(1, ddof=1)
    if P is not None:
        save_csv(t, os.path.join(P.metrics, 'accuracy_per_fold.csv'))
    return t


def overview(D, R, keys, name, P=None):
    """Accuracy with its 95% interval, kappa and QWK of the scorers keys."""
    rows = []
    for key in [k for k in keys if k in R]:
        p = R[key].argmax(1)
        lo, hi = np.percentile(_bootstrap(*_cells(D, (p == D.score).astype(float))), [2.5, 97.5])
        per = _per_item(D, p)
        rows.append({'Scorer': label(key), 'Paradigm': paradigm(key),
                     'Accuracy (%)': 100 * per['Accuracy'].astype(float).mean(), 'CI low': 100 * lo, 'CI high': 100 * hi,
                     'Kappa': per['Kappa'].astype(float).mean(), 'QWK': per['QWK'].astype(float).mean()})
    t = pd.DataFrame(rows).set_index('Scorer')
    if P is not None:
        save_csv(t, os.path.join(P.metrics, f'{name}.csv'))
    return t


def paradigms(D, R, P=None):
    """The scorer chosen on validation for each paradigm, the rules, the reference and the proposed ensemble."""
    return overview(D, R, ['best_network', 'best_classifier', 'rules', 'majority', PROPOSED], 'paradigms', P)


def ensemble(D, R, P=None):
    """The proposed ensemble, the ensembles without one member and the members alone."""
    return overview(D, R, [PROPOSED, *PAIRS, *ENSEMBLE], 'ensemble_members', P)


def confusion(D, R, P=None):
    rows = []
    for key in [k for k in individual() if k in R]:
        p = R[key].argmax(1)
        for i, s in enumerate(ITEMS):
            m = D.item == i
            cm = np.zeros((MAX_SCORE[s] + 1,) * 2, int)
            np.add.at(cm, (D.score[m], p[m]), 1)
            rows += [{'Scorer': label(key), 'Item': ITEM_LABEL[s], 'Expert score': a, 'Predicted score': b,
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
    """Paired differences in accuracy (A minus B, points) on the same out-of-fold drawings. Intervals and p values
    resample whole inferred children. A pair is 'different' when the Holm-adjusted permutation p is below 0.05 and the
    95% interval excludes zero, 'equivalent' when the 90% interval lies within the margin, else 'inconclusive'."""
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
            rows.append({'Family': fam, 'A': label(a), 'B': label(b), 'Key A': a, 'Key B': b, 'Delta (points)': delta,
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


THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')


def _tsne(X, seed=SEED):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = PCA(min(50, X.shape[1], len(X) - 1), random_state=seed).fit_transform(X)
    return TSNE(2, perplexity=min(30.0, (len(X) - 1) / 3), init='pca', learning_rate='auto',
                random_state=seed).fit_transform(X)


def _separation(X, items, below, seed=SEED):
    """Separation of drawings at full marks from those below, measured in the embedding space before projection:
    leave-one-out 10-nearest-neighbour balanced accuracy within each item (mean over items), and the cosine
    silhouette of the two groups."""
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    bal = []
    for i in range(N_ITEMS):
        m = np.where(items == i)[0]
        if below[m].all() or (~below[m]).all() or len(m) < 12:
            continue
        nb = NearestNeighbors(n_neighbors=11, metric='cosine').fit(X[m]).kneighbors(X[m], return_distance=False)[:, 1:]
        vote, y = below[m][nb].mean(1) > 0.5, below[m]
        bal.append(0.5 * (np.mean(vote[y]) + np.mean(~vote[~y])))
    s = np.random.default_rng(seed).choice(len(X), min(2000, len(X)), replace=False)
    return {'kNN balanced accuracy': float(np.mean(bal)) if bal else np.nan,
            'Silhouette (cosine)': float(silhouette_score(X[s], below[s], metric='cosine'))}


def _project_job(src, dst, seed):
    """Runs in a fresh process: t-SNE coordinates and separation indices of one embedding."""
    with np.load(src) as z:
        X, items, below = z['X'].astype(np.float32), z['items'], z['below']
    sep = _separation(X, items, below, seed)
    np.savez(dst, xy=_tsne(X, seed), knn=sep['kNN balanced accuracy'], silhouette=sep['Silhouette (cosine)'])


def project(X, items, below, seed=SEED, timeout_minutes=15, threads=4):
    """t-SNE and separation indices in a fresh process with a few threads, so that thread pools left over from GPU
    training cannot block them. After timeout_minutes the first two principal components are used instead."""
    import multiprocessing as mp
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, 'in.npz'), os.path.join(tmp, 'out.npz')
        np.savez(src, X=X, items=items, below=below)
        saved = {v: os.environ.get(v) for v in THREAD_VARS}
        os.environ.update({v: str(threads) for v in THREAD_VARS})
        try:
            proc = mp.get_context('spawn').Process(target=_project_job, args=(src, dst, seed), daemon=True)
            proc.start()
        finally:
            for v, old in saved.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old
        proc.join(timeout_minutes * 60)
        if proc.is_alive():
            proc.terminate()
            proc.join()
        if proc.exitcode == 0 and os.path.exists(dst):
            with np.load(dst) as z:
                return z['xy'], {'kNN balanced accuracy': float(z['knn']),
                                 'Silhouette (cosine)': float(z['silhouette'])}, 't-SNE'
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-6)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T, {'kNN balanced accuracy': np.nan, 'Silhouette (cosine)': np.nan}, 'PCA'


def embeddings(D, R, P, log=print):
    """Stores the out-of-fold embedding of every network and projects three of them with t-SNE (in a separate
    process): the pretrained ConvNeXt V2 before fine-tuning, the fine-tuned ConvNeXt V2 and the concatenated members
    of the ensemble."""
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
        log(f'Projecting: {title}')
        xy, idx, method = project(X.astype(np.float32), D.item, D.score < D.kmax)
        if method != 't-SNE':
            log(f'{title}: t-SNE did not finish, the panel shows the first two principal components')
            title = f'{title} (PCA)'
        coords.append(pd.DataFrame({'Panel': key, 'Title': title, 'file': D.df.file.values,
                                    'Item': D.df['item'].map(ITEM_LABEL), 'Below full marks': D.score < D.kmax,
                                    'x': xy[:, 0], 'y': xy[:, 1]}))
        sep.append({'Panel': key, 'Embedding': title, 'Projection': method, 'Dimensions': X.shape[1], **idx})
    coords = pd.concat(coords, ignore_index=True) if coords else pd.DataFrame()
    sep = pd.DataFrame(sep).set_index('Panel') if sep else pd.DataFrame()
    save_csv(coords, os.path.join(P.embeddings, 'tsne_coordinates.csv'), index=False)
    save_csv(sep, os.path.join(P.metrics, 'embedding_separation.csv'))
    return {'coords': coords, 'separation': sep}
