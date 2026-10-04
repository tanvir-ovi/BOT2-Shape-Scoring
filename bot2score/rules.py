"""OpenCV rubric rules. Every criterion of every item is judged by one geometric measurement compared with a cut-off,
and the item score is the number of criteria passed, or 0 when the basic shape fails, as the BOT-2 rubric prescribes.
The cut-off of a criterion is set on the training drawings of each fold, where the rule agrees most often with the
examiner's mark for that criterion; when no cut-off agrees more often than passing every drawing, the criterion always
passes."""
import os
import itertools

import numpy as np
import pandas as pd

from . import data as bd
from .data import ITEMS, FACETS, CRITERIA, CRITERION_LABEL, ITEM_LABEL, N_CLS
from .metrics import binary_kappa, mean_item_accuracy
from .utils import atomic_save, save_csv

CLOSURE = 'closure'
RULES = {   # criterion: (measurement, direction) or CLOSURE; 'min' passes at or above the cut-off, 'max' at or below,
            # 'typical' within the cut-off of the median of the passing training drawings
    'Circle':             {'basic': ('circularity', 'min'), 'closure': CLOSURE, 'edges': ('ellipse_ratio', 'max'),
                           'size': ('size_rel', 'min')},
    'Square':             {'basic': ('corner_dev4', 'max'), 'closure': CLOSURE, 'edges': ('side_ratio', 'max'),
                           'orientation': ('inner_rect_angle', 'max'), 'size': ('size_rel', 'min')},
    'Overlapped circle':  {'basic': ('n_holes', 'min'), 'closure': CLOSURE, 'edges': ('half_ratio_tb', 'max'),
                           'orientation': ('holes_dx', 'max'), 'overlap': ('lens_ratio', 'typical'),
                           'size': ('size_rel', 'min')},
    'Wave':               {'basic': ('n_holes', 'max'), 'edges': ('half_ratio_lr', 'max'),
                           'orientation': ('aspect', 'typical'), 'size': ('size_rel', 'min')},
    'Triangle':           {'basic': ('corner_dev3', 'max'), 'closure': CLOSURE, 'edges': ('side_ratio', 'max'),
                           'orientation': ('top_peak_dev', 'max'), 'size': ('size_rel', 'min')},
    'Diagonal':           {'basic': ('corner_dev4', 'max'), 'closure': CLOSURE, 'edges': ('side_ratio', 'max'),
                           'orientation': ('diamond_tilt', 'max'), 'size': ('size_rel', 'min')},
    'Star':               {'basic': ('points_dev5', 'max'), 'closure': CLOSURE, 'edges': ('peak_ratio', 'max'),
                           'orientation': ('top_peak_dev', 'max'), 'size': ('size_rel', 'min')},
    'Overlapped pencils': {'basic': ('n_holes', 'min'), 'closure': CLOSURE, 'edges': ('half_ratio_lr', 'max'),
                           'orientation': ('top_heavy', 'typical'), 'overlap': ('holes_dx', 'typical'),
                           'size': ('size_rel', 'min')},
}
MEASUREMENT = {
    'circularity':      'roundness of the outer contour (no corners)',
    'corner_dev4':      'difference between the number of corners and 4',
    'corner_dev3':      'difference between the number of corners and 3',
    'points_dev5':      'difference between the number of points and 5',
    'n_holes':          'number of enclosed regions',
    'ellipse_ratio':    'longest over shortest diameter',
    'side_ratio':       'longest over shortest side',
    'half_ratio_tb':    'width ratio of the upper and lower halves',
    'half_ratio_lr':    'height ratio of the left and right halves',
    'peak_ratio':       'longest over shortest point',
    'inner_rect_angle': 'tilt of the sides from horizontal (degrees)',
    'diamond_tilt':     'tilt of the sides from 45 degrees',
    'top_peak_dev':     'angle between the top corner and straight up (degrees)',
    'holes_dx':         'horizontal offset of the two largest enclosed regions',
    'aspect':           'width over height',
    'top_heavy':        'share of the filled area in the upper half',
    'lens_ratio':       'area share of the overlap region',
    'size_rel':         'drawing size relative to the image',
    CLOSURE:            'longest overshoot tail, gap between free stroke ends, retraced share of the stroke',
}
DIRECTION = {'min': 'passes at or above the cut-off', 'max': 'passes at or below the cut-off',
             'typical': 'passes within the cut-off of the typical value', 'closure': 'fails when any listed limit is reached'}
# closure fails when an overshoot tail, a gap between the free ends of an open stroke or the retraced share of the
# stroke reaches its cut-off; the grid runs from lenient to strict and np.inf switches a test off
CLOSURE_GRID = list(itertools.product((np.inf, 0.50, 0.30, 0.20, 0.15, 0.10), (np.inf, 0.25, 0.15, 0.10, 0.05),
                                      (np.inf, 0.40, 0.20, 0.10, 0.05)))


def measurements(G):
    """The geometric features plus the four derived measurements used by the rules."""
    M = G.copy()
    M['corner_dev4'] = (G['vertices_4'] - 4).abs()
    M['corner_dev3'] = (G['vertices_4'] - 3).abs()
    M['points_dev5'] = (G['radial_peaks'] - 5).abs()
    M['diamond_tilt'] = (45 - G['inner_rect_angle']).abs()
    return M.fillna(0.0).reset_index(drop=True)


def _closure_pass(M, rows, cut):
    tail, gap, thick = cut
    t, g = M['max_tail'].values[rows], M['end_gap'].values[rows]
    o, r = M['open_curve'].values[rows], M['thick_frac'].values[rows]
    return ~((t >= tail) | ((o > 0) & (g >= gap)) | (r >= thick))


def criterion_pass(M, rows, rule, cut):
    """True where the drawings rows pass the criterion."""
    if rule == CLOSURE:
        return _closure_pass(M, rows, cut)
    (col, direction), (c, ref) = rule, cut
    v = M[col].values[rows]
    if direction == 'typical':
        return np.abs(v - ref) <= c
    return v >= c if direction == 'min' else v <= c


def _agreement(y, passed):
    return float(np.mean(np.asarray(passed, int) == y)) if len(y) else 0.0


def fit_cut(M, rows, y, rule):
    """Cut-off with which the rule agrees most often with the examiner's marks y on the training rows. Candidates run
    from lenient to strict, so a tie keeps the more lenient cut-off, and the cut-off that never fails is kept unless a
    candidate agrees more often. Returns the cut-off, its kappa and its agreement on the training rows."""
    everyone = np.ones(len(y), bool)
    if rule == CLOSURE:
        best, best_a = (np.inf, np.inf, np.inf), _agreement(y, everyone)
        for cut in CLOSURE_GRID:
            a = _agreement(y, _closure_pass(M, rows, cut))
            if a > best_a + 1e-12:
                best, best_a = cut, a
        return best, binary_kappa(y, _closure_pass(M, rows, best)), best_a
    col, direction = rule
    v = M[col].values[rows]
    ref = float(np.median(v[y == 1])) if direction == 'typical' and (y == 1).any() else 0.0
    if direction == 'typical':
        v = np.abs(v - ref)
    cands = np.unique(np.quantile(v, np.linspace(0, 1, 101))) if len(v) else np.array([])
    if direction == 'min':
        best = (-np.inf, ref)
    else:
        best, cands = (np.inf, ref), cands[::-1]
    best_a = _agreement(y, everyone)
    for c in cands:
        a = _agreement(y, v >= c if direction == 'min' else v <= c)
        if a > best_a + 1e-12:
            best, best_a = (float(c), ref), a
    passed = v >= best[0] if direction == 'min' else v <= best[0]
    return best, binary_kappa(y, passed), best_a


def score(M, D, idx, cuts):
    """Item scores (one-hot over the score scale) and criterion decisions of the drawings idx."""
    crit = np.full((len(idx), len(CRITERIA)), np.nan, np.float32)
    pred = np.zeros(len(idx), int)
    for i, s in enumerate(ITEMS):
        sel = np.where(D.item[idx] == i)[0]
        if not len(sel):
            continue
        rows = idx[sel]
        passed = np.stack([criterion_pass(M, rows, RULES[s][c], cuts[(s, c)]) for c in FACETS[s]], 1).astype(int)
        for j, c in enumerate(FACETS[s]):
            crit[sel, CRITERIA.index(c)] = passed[:, j]
        pred[sel] = np.where(passed[:, 0] == 1, passed.sum(1), 0)
    return np.eye(N_CLS, dtype=np.float32)[pred], crit


def _training_rows(D, tr, i, c):
    """Training drawings of item i with a known mark for criterion c; drawings scored 0 are used for the basic shape
    only, because a score of 0 sets the other marks to 0."""
    j = CRITERIA.index(c)
    keep = (D.item[tr] == i) & ~np.isnan(D.marks[tr, j]) & ((D.score[tr] > 0) | (c == 'basic'))
    rows = tr[keep]
    return rows, D.marks[rows, j].astype(int)


def _cut_text(rule, cut):
    if rule == CLOSURE:
        names = ('tail', 'gap', 'retraced')
        parts = [f'{n} ≥ {v:.2f}' for n, v in zip(names, cut) if np.isfinite(v)]
        return ', '.join(parts) if parts else 'always passes'
    (col, direction), (c, ref) = rule, cut
    if not np.isfinite(c):
        return 'always passes'
    if direction == 'typical':
        return f'{ref:.3g} ± {c:.3g}'
    return f'{"≥" if direction == "min" else "≤"} {c:.3g}'


def run(D, G, P, log=print):
    """Sets the cut-offs on the training part of every fold and scores the validation and test parts."""
    M = measurements(G)
    rows_out, acc = [], []
    for k in range(D.n_folds):
        tr, va, te = bd.split(D, k)
        cuts = {}
        for i, s in enumerate(ITEMS):
            for c in FACETS[s]:
                rows, y = _training_rows(D, tr, i, c)
                cut, kap, agree = fit_cut(M, rows, y, RULES[s][c])
                cuts[(s, c)] = cut
                rule = RULES[s][c]
                rows_out.append({'Item': ITEM_LABEL[s], 'Criterion': CRITERION_LABEL[c], 'Fold': k,
                                 'Measurement': 'closure' if rule == CLOSURE else rule[0],
                                 'Direction': 'closure' if rule == CLOSURE else rule[1],
                                 'Cut-off': _cut_text(rule, cut), 'Training drawings': len(y),
                                 'Training failures': int((y == 0).sum()), 'Training agreement (%)': 100 * agree,
                                 'Training kappa': kap})
        q_va, _ = score(M, D, va, cuts)
        q_te, crit_te = score(M, D, te, cuts)
        atomic_save(os.path.join(P.runs, 'rules', f'fold{k}.npz'), lambda p: np.savez(
            p, va_idx=va, te_idx=te, q_va=q_va, q_te=q_te, crit_te=crit_te))
        acc.append((mean_item_accuracy(D.score[va], q_va.argmax(1), D.item[va]),
                    mean_item_accuracy(D.score[te], q_te.argmax(1), D.item[te])))
    t = pd.DataFrame(rows_out)
    save_csv(t, os.path.join(P.metrics, 'rule_cutoffs.csv'), index=False)
    va_acc, te_acc = np.mean(acc, 0) * 100
    log(f'Rubric rules set on the training part of {D.n_folds} folds: validation accuracy {va_acc:.2f}%, '
        f'test accuracy {te_acc:.2f}% (mean of the folds)')
    return rule_table(t)


def rule_table(t):
    """One row per item and criterion: the measurement, its rule, the cut-off set in every fold, and the mean
    agreement and kappa of the rule with the examiner on the training drawings."""
    rows = []
    for (item, crit), g in t.groupby(['Item', 'Criterion'], sort=False):
        first = g.iloc[0]
        row = {'Item': item, 'Criterion': crit, 'Measurement': MEASUREMENT[first['Measurement']],
               'Rule': DIRECTION[first['Direction']]}
        row.update({f'Fold {k}': c for k, c in zip(g['Fold'], g['Cut-off'])})
        row['Training agreement (%)'] = g['Training agreement (%)'].mean()
        row['Training kappa'] = g['Training kappa'].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index(['Item', 'Criterion']).round(3)


def criterion_agreement(D, P):
    """Agreement of every rule with the examiner's criterion marks on the test folds: kappa and the share of drawings
    failing the criterion by each rater. Criteria other than the basic shape are judged on drawings scored above 0."""
    n = len(D.score)
    crit = np.full((n, len(CRITERIA)), np.nan, np.float32)
    for k in range(D.n_folds):
        z = np.load(os.path.join(P.runs, 'rules', f'fold{k}.npz'))
        crit[z['te_idx']] = z['crit_te']
    rows = []
    for i, s in enumerate(ITEMS):
        for c in FACETS[s]:
            j = CRITERIA.index(c)
            m = (D.item == i) & ~np.isnan(D.marks[:, j]) & ((D.score > 0) | (c == 'basic'))
            y, p = D.marks[m, j].astype(int), crit[m, j].astype(int)
            rows.append({'Item': ITEM_LABEL[s], 'Criterion': CRITERION_LABEL[c], 'Drawings': int(m.sum()),
                         'Failed by the examiner (%)': 100 * float(np.mean(y == 0)),
                         'Failed by the rule (%)': 100 * float(np.mean(p == 0)),
                         'Agreement (%)': 100 * float(np.mean(y == p)),
                         'Kappa': binary_kappa(y, p) if (y == 0).any() or (p == 0).any() else np.nan})
    t = pd.DataFrame(rows)
    save_csv(t, os.path.join(P.metrics, 'rule_criterion_agreement.csv'), index=False)
    return t.set_index(['Item', 'Criterion']).round(3)
