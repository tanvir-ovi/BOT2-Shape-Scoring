"""Accuracy and agreement between expert and predicted scores. Kappa values use the full score scale of the item, so a
score that neither rater gave still counts as a category."""
import numpy as np

from .data import N_ITEMS, N_CLS

BANDS = ((0.20, 'slight'), (0.40, 'fair'), (0.60, 'moderate'), (0.80, 'substantial'), (np.inf, 'almost perfect'))


def confusion(y, p, n=N_CLS):
    """Counts of (expert, predicted) score pairs; rows are expert scores."""
    return np.bincount(np.asarray(y, int) * n + np.asarray(p, int), minlength=n * n).reshape(n, n).astype(float)


def kappa_from_counts(O, weights=None):
    """Cohen's kappa (weights None) or quadratic weighted kappa ('quadratic') of one confusion matrix (n, n) or a
    stack of them (..., n, n). Undefined values (one category only) are NaN."""
    O = np.asarray(O, float)
    n = O.shape[-1]
    i = np.arange(n)
    W = ((i[:, None] - i[None, :]) ** 2).astype(float) if weights == 'quadratic' else (i[:, None] != i[None, :]).astype(float)
    tot = O.sum((-2, -1))
    with np.errstate(invalid='ignore', divide='ignore'):
        E = O.sum(-1)[..., :, None] * O.sum(-2)[..., None, :] / tot[..., None, None]
        num, den = (W * O).sum((-2, -1)), (W * E).sum((-2, -1))
        k = np.where(den > 0, 1 - num / np.where(den > 0, den, 1), np.nan)
    return k


def kappa(y, p, weights=None, n=N_CLS):
    return float(kappa_from_counts(confusion(y, p, n), weights))


def binary_kappa(y, p):
    """Cohen's kappa of two binary ratings; 0 when either rating has a single value."""
    y, p = np.asarray(y, int), np.asarray(p, int)
    if len(y) == 0:
        return np.nan
    po = np.mean(y == p)
    pe = np.mean(y) * np.mean(p) + (1 - np.mean(y)) * (1 - np.mean(p))
    return 0.0 if pe >= 1 - 1e-12 else float((po - pe) / (1 - pe))


def cramers_v(y, p):
    """Association between expert and predicted scores (Cramér's V of their contingency table)."""
    ys, ps = np.unique(y), np.unique(p)
    if len(ys) < 2 or len(ps) < 2:
        return np.nan
    t = np.array([[np.sum((y == a) & (p == b)) for b in ps] for a in ys], float)
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    chi2 = ((t - e) ** 2 / e).sum()
    return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))


def _per_item(fn, y, p, items):
    return [fn(y[items == i], p[items == i]) for i in range(N_ITEMS) if (items == i).any()]


def mean_item_accuracy(y, p, items):
    """Mean over items of the share of drawings whose predicted score equals the expert score."""
    return float(np.mean(_per_item(lambda a, b: np.mean(a == b), y, p, items)))


def mean_item_kappa(y, p, items, weights=None):
    v = _per_item(lambda a, b: kappa(a, b, weights), y, p, items)
    return np.nan if np.all(np.isnan(v)) else float(np.nanmean(v))


def band(k):
    """Landis and Koch label of a kappa value; each band includes its upper bound (0.61 to 0.80 is substantial)."""
    if not np.isfinite(k):
        return ''
    if k < 0:
        return 'poor'
    return next(name for upper, name in BANDS if k <= upper)
