"""Paper figures as vector PDF at IEEE print width (column 3.49 in, text 7.14 in) with 8 pt serif text. Every figure is
checked for overlapping text, text outside the page and empty margins before it is saved."""
import os
import shutil
import itertools
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .data import ITEMS, ITEM_LABEL, MAX_SCORE, N_ITEMS, CRITERIA, CRITERION_LABEL, FACETS
from .evaluation import MARGIN
from .scorers import PROPOSED, PROPOSED_LABEL, NETWORK_KEYS, label, role

COL_W, TXT_W = 252 / 72.27, 516 / 72.27          # IEEE conference column and text width (inches)
FS = 8
GREEN, ORANGE, BLUE, GREY, PURPLE, LIGHT, RED = '#3E9E7E', '#E07B39', '#3B6FB6', '#7F7F7F', '#7B5EA7', '#B3B3B3', '#C0392B'
ROLE_COLOR = {'proposed': GREEN, 'member': ORANGE, 'baseline': BLUE, 'control': BLUE, 'classifier': GREY,
              'rules': PURPLE, 'reference': LIGHT, 'pair': ORANGE, 'selected': BLUE}
ROLE_NAME = {'proposed': 'Proposed ensemble', 'member': 'Ensemble member', 'baseline': 'Other network',
             'classifier': 'Machine learning', 'rules': 'OpenCV rubric rules', 'reference': 'Most frequent score'}
ROLE_ORDER = ('proposed', 'member', 'baseline', 'classifier', 'rules', 'reference')
VERDICT_COLOR = {'different': GREEN, 'equivalent': BLUE, 'inconclusive': GREY}
VERDICT_NAME = {'different': 'Different', 'equivalent': 'Equivalent within margin', 'inconclusive': 'Inconclusive'}
SHORT = {'Overlapped circles': 'Overlapped\ncircles', 'Overlapped pencils': 'Overlapped\npencils'}
PAIR_LABEL = {('dit_224', 'beit_natural'): 'Documents vs natural images',
              ('beit_sketch', 'beit_natural'): 'Sketches vs natural images',
              ('dit_224', 'beit_sketch'): 'Documents vs sketches', ('convnext_512', 'convnext_224'): '512 px vs 224 px',
              ('best_network', 'best_classifier'): 'Network vs classifier',
              ('best_network', 'rules'): 'Network vs rubric rules', ('best_classifier', 'rules'): 'Classifier vs rubric rules'}
SELECTED_STYLE = {PROPOSED: (GREEN, '*', 7.5), 'best_network': (BLUE, 'o', 4.2), 'best_classifier': (GREY, '^', 4.4),
                  'rules': (PURPLE, 's', 3.8)}


def setup():
    """Times-compatible serif font (Liberation Serif) and the rcParams of the paper figures."""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    if 'Liberation Serif' not in names and shutil.which('apt-get'):
        subprocess.run(['apt-get', '-qq', 'install', '-y', 'fonts-liberation'], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for d in ('/usr/share/fonts/truetype/liberation', '/usr/share/fonts/truetype/liberation2'):
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.endswith('.ttf'):
                        font_manager.fontManager.addfont(os.path.join(d, fn))
        names = {f.name for f in font_manager.fontManager.ttflist}
    font = next(f for f in ('Liberation Serif', 'Times New Roman', 'STIXGeneral', 'DejaVu Serif') if f in names)
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'serif', 'font.serif': [font], 'mathtext.fontset': 'stix',
        'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS,
        'legend.fontsize': FS, 'figure.titlesize': FS, 'figure.labelsize': FS, 'axes.titleweight': 'normal',
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
        'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
        'axes.unicode_minus': True, 'figure.constrained_layout.use': True, 'figure.constrained_layout.h_pad': 0.02,
        'figure.constrained_layout.w_pad': 0.02, 'savefig.dpi': 300})
    return font


# ---------------------------------------------------------------------------------------------------
# Layout check and export
# ---------------------------------------------------------------------------------------------------
def _texts(fig):
    def ticks(axis):
        lo, hi = sorted(axis.get_view_interval())
        out = []
        for t in axis.get_major_ticks():
            if lo - 1e-9 <= t.get_loc() <= hi + 1e-9:
                out += [lab for lab in (t.label1, t.label2) if lab.get_visible() and lab.get_text().strip()]
        return out
    arts = list(fig.texts)                      # includes the figure-level x and y labels
    for ax in fig.axes:
        arts += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + ticks(ax.xaxis) + ticks(ax.yaxis)
        arts += list(ax.texts)
        if ax.get_legend() is not None:
            arts += ax.get_legend().get_texts()
    for leg in fig.legends:
        arts += leg.get_texts()
    return [a for a in dict.fromkeys(arts) if a.get_visible() and a.get_text().strip()]


def _overlap(a, b):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w > 0.5 and h > 0.5


def check_layout(fig, max_margin=0.06):
    """Lists overlapping labels, legends over plots, text outside the page and empty margins wider than max_margin."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    boxes = [(a, a.get_window_extent(r)) for a in _texts(fig)]
    problems = []
    for (a, ba), (b, bb) in itertools.combinations(boxes, 2):
        if _overlap(ba, bb):
            problems.append(f'text overlap: "{a.get_text()}" and "{b.get_text()}"')
    for a, ba in boxes:
        if ba.x0 < -0.5 or ba.y0 < -0.5 or ba.x1 > W + 0.5 or ba.y1 > H + 0.5:
            problems.append(f'text outside the page: "{a.get_text()}"')
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    for leg in legends:
        lb = leg.get_window_extent(r)
        for ax in fig.axes:
            if ax.get_label() != '<colorbar>' and ax.axison and _overlap(lb, ax.get_window_extent(r)):
                problems.append('legend overlaps a plotting area')
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    ink = np.where((img < 245).any(axis=2))
    margins = {'left': ink[1].min() / fig.dpi, 'right': (img.shape[1] - 1 - ink[1].max()) / fig.dpi,
               'top': ink[0].min() / fig.dpi, 'bottom': (img.shape[0] - 1 - ink[0].max()) / fig.dpi}
    problems += [f'empty {side} margin of {m:.3f} in' for side, m in margins.items() if m > max_margin]
    return problems


def save(fig, name, P=None, log=print):
    problems = check_layout(fig)
    for p in problems:
        log(f'{name}: {p}')
    if P is not None:
        path = os.path.join(P.figures, name)
        fig.savefig(path)                          # no tight bounding box: the page keeps the exact print width
        log(f'{name}: {fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in, '
            + ('layout check passed' if not problems else f'{len(problems)} layout problems'))
    plt.show()
    plt.close(fig)
    return problems


def _fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def _signed(v):
    return f'{v:+.2f}'.replace('-', '−')


def _bold(labels, flags):
    for lab, flag in zip(labels, flags):
        if flag:
            lab.set_fontweight('bold')


def _roles_present(roles):
    roles = {'baseline' if r == 'control' else r for r in roles}
    return [r for r in ROLE_ORDER if r in roles]


# ---------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------
def scores(D, P=None, log=print):
    """Share of the drawings of each item at full marks and at each distance below it."""
    names = ('Full marks', '1 point below', '2 points below', '3 or more points below', 'Score 0')
    colors = (GREEN, '#A6D28F', '#F2D16B', '#E8955B', RED)
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ys = np.arange(N_ITEMS)[::-1]
    for y, (i, s) in zip(ys, enumerate(ITEMS)):
        sc, k = D.score[D.item == i], MAX_SCORE[s]
        shares = 100 * np.array([np.mean(sc == k), np.mean(sc == k - 1), np.mean(sc == k - 2),
                                 np.mean((sc > 0) & (sc <= k - 3)), np.mean(sc == 0)])
        left = 0.0
        for share, col in zip(shares, colors):
            if share > 0:
                ax.barh(y, share, left=left, color=col, height=0.72, lw=0)
            left += share
        ax.text(shares[0] / 2, y, f'{shares[0]:.0f}%', ha='center', va='center', color='white')
    ax.set_yticks(ys, [ITEM_LABEL[s] for s in ITEMS])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.45, N_ITEMS - 0.55)
    ax.set_xlabel('Drawings of the item (%)')
    fig.legend(handles=[Patch(color=c, label=n) for n, c in zip(names, colors)], loc='outside lower center', ncol=2,
               handlelength=1.2, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.1)
    return save(fig, 'Fig_Scores.pdf', P, log)


def criteria(CT, RA, P=None, log=print):
    """(a) Share of drawings failing each rubric criterion (examiner), (b) kappa between each OpenCV rule and the
    examiner on the test folds. Blank cells are criteria that the item does not have; a dash marks a kappa that is
    undefined because neither rater failed the criterion."""
    items = [ITEM_LABEL[s] for s in ITEMS]
    cols = [CRITERION_LABEL[c] for c in CRITERIA]
    applies = np.array([[c in FACETS[s] for c in CRITERIA] for s in ITEMS])
    A = CT.reindex(index=items, columns=cols).values.astype(float)
    K = RA['Kappa'].unstack('Criterion').reindex(index=items, columns=cols).values.astype(float)
    fig, axes = plt.subplots(2, 1, figsize=(TXT_W, 3.85))
    specs = ((A, 'Reds', 0.0, max(60.0, float(np.nanmax(A))), '{:.1f}', '(a) Drawings that fail the criterion (%)'),
             (K, 'Purples', 0.0, 1.0, '{:.2f}', '(b) Agreement of the rule with the examiner (kappa)'))
    for ax, (M, cmap_name, vmin, vmax, fmt, title) in zip(axes, specs):
        cmap = mpl.colormaps[cmap_name].copy()
        cmap.set_bad('white')
        ax.imshow(np.ma.masked_invalid(np.clip(M, vmin, vmax)), cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not applies[i, j]:
                    continue
                v = M[i, j]
                if np.isfinite(v):
                    dark = (min(max(v, vmin), vmax) - vmin) / (vmax - vmin) > 0.55
                    ax.text(j, i, fmt.format(v).replace('-', '−'), ha='center', va='center',
                            color='white' if dark else 'black')
                else:
                    ax.text(j, i, '–', ha='center', va='center', color='0.45')
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(items)), items)
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(items)), minor=True)
        ax.grid(which='minor', color='white', lw=1.0)
        ax.tick_params(which='both', length=0)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.4)
            sp.set_color('0.6')
        ax.set_title(title, loc='left', pad=3)
    return save(fig, 'Fig_Criteria.pdf', P, log)


# ---------------------------------------------------------------------------------------------------
# Accuracy and agreement
# ---------------------------------------------------------------------------------------------------
def accuracy(S, F, P=None, log=print):
    """Accuracy of every scorer: single folds, all drawings with the child-level 95% interval, and the value."""
    order = S.sort_values('Accuracy (%)').index.tolist()
    n = len(order)
    fold_cols = [c for c in F.columns if c.startswith('Fold')]
    fig, ax = plt.subplots(figsize=(COL_W, 0.17 * n + 1.1))
    for y, key in enumerate(order):
        c = ROLE_COLOR[S.loc[key, 'Role']]
        ax.plot([S.loc[key, 'CI low'], S.loc[key, 'CI high']], [y, y], color=c, lw=1.3, solid_capstyle='butt', zorder=2)
        folds = F.loc[S.loc[key, 'Scorer'], fold_cols].values.astype(float)
        ax.scatter(folds, np.full(len(folds), y), s=6, color=c, alpha=0.55, lw=0, zorder=3)
        ax.plot(S.loc[key, 'Accuracy (%)'], y, 'D', mfc='white', mec=c, mew=1.0, ms=4.0, zorder=4)
    ax.set_yticks(range(n), [S.loc[k, 'Scorer'] for k in order])
    _bold(ax.get_yticklabels(), [k == PROPOSED for k in order])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel('Accuracy (%), mean of the eight items')
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='0.9', lw=0.5)
    ax.set_axisbelow(True)
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(range(n), [f'{S.loc[k, "Accuracy (%)"]:.2f}' for k in order])
    for lab, key in zip(right.get_yticklabels(), order):
        lab.set_color(ROLE_COLOR[S.loc[key, 'Role']])
    _bold(right.get_yticklabels(), [k == PROPOSED for k in order])
    right.tick_params(axis='y', length=0, pad=2)
    right.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    right.text(1.0, 1.0, ' Mean', transform=right.transAxes, ha='left', va='bottom')
    handles = [Line2D([], [], marker='o', ls='none', ms=3, color='0.4', alpha=0.6, label='Single fold'),
               Line2D([], [], marker='D', mfc='white', mec='0.3', ls='-', color='0.3', lw=1.3, ms=4.0,
                      label='All drawings, 95% CI')] + \
              [Patch(color=ROLE_COLOR[r], label=ROLE_NAME[r]) for r in _roles_present(S.Role)]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, handlelength=1.6, columnspacing=1.2,
               handletextpad=0.5, borderaxespad=0.1)
    return save(fig, 'Fig_Accuracy.pdf', P, log)


def agreement(IA, P=None, log=print):
    """Cohen's kappa and QWK of every item with child-level 95% intervals for the proposed ensemble and the scorer
    chosen to represent each paradigm, over the Landis and Koch bands."""
    keys = [k for k in SELECTED_STYLE if k in set(IA.Key)]
    cols = [ITEM_LABEL[s] for s in ITEMS] + ['Mean']
    offs = np.linspace(-0.27, 0.27, len(keys)) if len(keys) > 1 else np.zeros(1)
    bands = [(-1.0, 0.0, 'poor'), (0.0, 0.2, 'slight'), (0.2, 0.4, 'fair'), (0.4, 0.6, 'moderate'),
             (0.6, 0.8, 'substantial'), (0.8, 1.0, 'almost perfect')]
    fig, axes = plt.subplots(2, 1, figsize=(TXT_W, 4.5), sharex=True)
    for ax, (metric, title) in zip(axes, (('Kappa', "(a) Cohen's kappa"), ('QWK', '(b) Quadratic weighted kappa'))):
        low = float(np.nanmin(IA[f'{metric} CI low'].values))
        ymin = min(-0.05, np.floor((low - 0.03) * 10) / 10)
        centers, names = [], []
        for n_, (a, b, name) in enumerate(bands):
            if b <= ymin:
                continue
            ax.axhspan(max(a, ymin), b, color='0.93' if n_ % 2 else 'white', lw=0, zorder=0)
            centers.append((max(a, ymin) + b) / 2)
            names.append(name)
        for j, key in enumerate(keys):
            t = IA[IA.Key == key].set_index('Item').reindex(cols)
            c, mk, ms = SELECTED_STYLE[key]
            x = np.arange(len(cols)) + offs[j]
            v = t[metric].values.astype(float)
            err = np.clip([v - t[f'{metric} CI low'].values, t[f'{metric} CI high'].values - v], 0, None)
            ax.errorbar(x, v, yerr=err, fmt='none', ecolor=c, elinewidth=0.9, capsize=0, zorder=2)
            ax.plot(x, v, mk, color=c, mec='white', mew=0.4, ms=ms, ls='none', zorder=3)
        ax.axvline(len(cols) - 1.5, color='0.55', lw=0.6, zorder=1)
        ax.set_ylim(ymin, 1.0)
        ax.set_ylabel('Kappa' if metric == 'Kappa' else 'QWK')
        ax.set_title(title, loc='left', pad=3)
        right = ax.twinx()
        right.set_ylim(ax.get_ylim())
        right.set_yticks(centers, names)
        right.tick_params(axis='y', length=0, pad=2, colors='0.35')
        right.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    axes[1].set_xticks(range(len(cols)), [SHORT.get(c, c) for c in cols])
    axes[1].set_xlim(-0.6, len(cols) - 0.4)
    axes[1].tick_params(axis='x', length=0)
    handles = [Line2D([], [], marker=SELECTED_STYLE[k][1], color=SELECTED_STYLE[k][0], mec='white', mew=0.4,
                      ms=SELECTED_STYLE[k][2], ls='none', label=label(k)) for k in keys]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(keys), handletextpad=0.3, columnspacing=1.6,
               borderaxespad=0.1)
    return save(fig, 'Fig_Agreement.pdf', P, log)


def item_accuracy(T, S, P=None, log=print):
    """Accuracy (%) of every scorer on every item; the most-frequent-score row is the share of the modal score."""
    order = list(T.sort_values('Mean', ascending=False).index)
    M = T.loc[order].values
    fig, ax = plt.subplots(figsize=(TXT_W, 0.2 * len(order) + 0.85))
    im = ax.imshow(M, cmap='RdYlGn', vmin=40, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', color='white' if M[i, j] < 52 or M[i, j] > 93 else 'black')
    ax.set_xticks(range(M.shape[1]), [SHORT.get(ITEM_LABEL[s], ITEM_LABEL[s]) for s in ITEMS] + ['Mean'])
    ax.set_yticks(range(len(order)), order)
    _bold(ax.get_yticklabels(), [o == PROPOSED_LABEL for o in order])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(M.shape[1] - 1.5, color='white', lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, aspect=15)
    cb.set_label('Accuracy (%)')
    cb.outline.set_linewidth(0.4)
    return save(fig, 'Fig_Item_Accuracy.pdf', P, log)


# ---------------------------------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------------------------------
def _forest(panels, name, P, log):
    """Forest plot with one panel per (title, rows of compare(), row labels); each panel has its own x range."""
    panels = [p for p in panels if len(p[1])]
    if not panels:
        return []
    sizes = [len(t) for _, t, _ in panels]
    fig = plt.figure(figsize=(TXT_W, 0.19 * sum(sizes) + 0.46 * len(panels) + 0.62))
    gs = fig.add_gridspec(len(panels), 2, height_ratios=[s + 1.3 for s in sizes], width_ratios=[2.5, 1.0])
    seen = set()
    for r, (title, t, names) in enumerate(panels):
        t = t.reset_index(drop=True)
        n = len(t)
        ax = fig.add_subplot(gs[r, 0])
        tab = fig.add_subplot(gs[r, 1], sharey=ax)
        ys = np.arange(n)[::-1]
        lo, hi = min(t['CI95 low'].min(), -MARGIN), max(t['CI95 high'].max(), MARGIN)
        pad = 0.06 * (hi - lo)
        ax.axvspan(-MARGIN, MARGIN, color=GREEN, alpha=0.12, lw=0, zorder=0)
        ax.axvline(0, color=RED, ls='--', lw=0.7, zorder=1)
        for y, (_, row) in zip(ys, t.iterrows()):
            c = VERDICT_COLOR[row.Verdict]
            seen.add(row.Verdict)
            ax.plot([row['CI95 low'], row['CI95 high']], [y, y], color=c, lw=1.6, solid_capstyle='butt', zorder=2)
            ax.plot(row['Delta (points)'], y, 'o', color=c, mec='white', mew=0.5, ms=4.6, zorder=3)
        ax.set_yticks(ys, names)
        ax.tick_params(axis='y', length=0)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(f'({"abcdefgh"[r]}) {title}', loc='left', pad=3)
        ax.grid(axis='x', color='0.9', lw=0.5)
        ax.set_axisbelow(True)
        if r == len(panels) - 1:
            ax.set_xlabel('Difference in accuracy (points)')
        tab.axis('off')
        cols = [(0.02, 'Δ (points)', lambda row: _signed(row['Delta (points)'])),
                (0.40, 'p', lambda row: _fmt_p(row['p (permutation)'])),
                (0.72, 'p (Holm)', lambda row: _fmt_p(row['p (Holm)']))]
        for x, head, fn in cols:
            tab.text(x, n - 0.4, head, ha='left', va='bottom', transform=tab.get_yaxis_transform(), fontweight='bold')
            for y, (_, row) in zip(ys, t.iterrows()):
                tab.text(x, y, fn(row), ha='left', va='center', transform=tab.get_yaxis_transform())
    handles = [Line2D([], [], color=VERDICT_COLOR[v], lw=1.6, marker='o', ms=4.6, mec='white', mew=0.5,
                      label=VERDICT_NAME[v]) for v in ('different', 'equivalent', 'inconclusive') if v in seen] + \
              [Patch(color=GREEN, alpha=0.12, label=f'Equivalence margin (±{MARGIN:g} point)'),
               Line2D([], [], color=RED, ls='--', lw=0.7, label='No difference')]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), handlelength=1.6, columnspacing=1.2,
               handletextpad=0.5, borderaxespad=0.1)
    return save(fig, name, P, log)


def comparisons(C, P=None, log=print):
    """The proposed ensemble minus every other scorer and minus each ensemble without one member (Holm within the
    family of all scorers and within the family of ensembles)."""
    prop = C[C.Family == 'Proposed']
    nets, feats = prop[prop['Key B'].isin(NETWORK_KEYS)], prop[~prop['Key B'].isin(NETWORK_KEYS)]
    ens = C[C.Family == 'Ensemble']
    return _forest([(f'{PROPOSED_LABEL} minus each network', nets, list(nets.B)),
                    (f'{PROPOSED_LABEL} minus each feature-based scorer', feats, list(feats.B)),
                    (f'{PROPOSED_LABEL} minus the ensemble without one member', ens, list(ens.B))],
                   'Fig_Paired_Comparisons.pdf', P, log)


def paradigms(C, P=None, log=print):
    """Paradigms (network and classifier chosen on validation in each fold, rubric rules) and the pretraining source
    and input size of single networks."""
    par, pre = C[C.Family == 'Paradigms'], C[C.Family == 'Pretraining']
    names = lambda t: [PAIR_LABEL.get((a, b), f'{la} vs {lb}') for a, b, la, lb in zip(t['Key A'], t['Key B'], t.A, t.B)]
    return _forest([('Paradigms: network and classifier chosen on validation in each fold', par, names(par)),
                    ('Pretraining source of BEiT-B and input size of ConvNeXt V2-T', pre, names(pre))],
                   'Fig_Paradigms_Pretraining.pdf', P, log)


# ---------------------------------------------------------------------------------------------------
# Drawings below full marks, confusion, embeddings, training
# ---------------------------------------------------------------------------------------------------
def precision_recall(D, R, S, curves, P=None, log=print):
    """Detection of drawings below full marks, pooled over items: (a) precision-recall curves with average precision
    and the operating point of each scorer's decision, (b) operating points of every scorer over constant-F1
    contours."""
    from sklearn.metrics import average_precision_score
    yb = D.score < D.kmax
    shown = [k for k in (PROPOSED, 'best_network', 'best_classifier') if k in curves]
    colour = {PROPOSED: GREEN, 'best_network': BLUE, 'best_classifier': GREY}
    prevalence = float(np.mean(yb))
    fig, (a, b) = plt.subplots(1, 2, figsize=(TXT_W, 3.15))
    for k in shown:
        rec, prec = curves[k]
        sb = 1 - R[k][np.arange(len(yb)), D.kmax]
        pred = R[k].argmax(1) < D.kmax
        tp = float(np.sum(pred & yb))
        a.plot(rec, prec, color=colour[k], lw=1.4 if k == PROPOSED else 1.0, zorder=3 if k == PROPOSED else 2,
               label=f'{label(k)} (AP {average_precision_score(yb, sb):.3f})')
        a.plot(tp / max(yb.sum(), 1), tp / max(pred.sum(), 1), 'o', color=colour[k], mec='white', mew=0.6, ms=5, zorder=4)
    if 'rules' in S.index:
        a.plot(S.loc['rules', 'Recall below full'], S.loc['rules', 'Precision below full'], 's', color=PURPLE, mec='white',
               mew=0.6, ms=4.5, zorder=4, label=f'{label("rules")} (decision only)')
    a.axhline(prevalence, color='0.6', ls=':', lw=0.8, label=f'No skill ({prevalence:.3f})')
    a.set_title('(a) Precision-recall curves', loc='left', pad=3)
    for c in (0.5, 0.6, 0.7, 0.8):
        rr = np.linspace(c / 2 + 1e-3, 1, 400)
        pp = c * rr / (2 * rr - c)
        keep = pp <= 1
        b.plot(rr[keep], pp[keep], color='0.6', lw=0.6, ls=':', zorder=1)
        b.text(0.995, c / (2 - c) - 0.012, f'F$_1$ = {c:.1f}', ha='right', va='top', color='0.35')
    for k in S.index:
        if k == 'majority' or not np.isfinite(S.loc[k, 'Precision below full']) or S.loc[k, 'Recall below full'] == 0:
            continue
        r_ = S.loc[k, 'Role']
        b.plot(S.loc[k, 'Recall below full'], S.loc[k, 'Precision below full'], '*' if k == PROPOSED else 'o',
               color=ROLE_COLOR[r_], mec='white', mew=0.5, ms=10 if k == PROPOSED else 5, zorder=4 if k == PROPOSED else 3)
    b.set_title('(b) Operating points and constant-F$_1$ contours', loc='left', pad=3)
    for ax in (a, b):
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('Recall for drawings below full marks')
        ax.grid(color='0.92', lw=0.5)
        ax.set_axisbelow(True)
    a.set_ylabel('Precision for drawings below full marks')
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, handlelength=1.6, borderaxespad=0.1, borderpad=0.15)
    roles = _roles_present(S.drop(index='majority', errors='ignore').Role)
    handles = [Line2D([], [], marker='*' if r == 'proposed' else 'o', ls='none', color=ROLE_COLOR[r], mec='white',
                      ms=9 if r == 'proposed' else 5, label=ROLE_NAME[r]) for r in roles] + \
              [Line2D([], [], color='0.6', ls=':', lw=0.6, label='Constant F$_1$')]
    b.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.6, borderpad=0.15,
             borderaxespad=0.1)
    return save(fig, 'Fig_Precision_Recall.pdf', P, log)


def confusion(D, R, key=PROPOSED, P=None, log=print):
    """Expert against predicted score for every item (out-of-fold); shading is per panel, blank cells are zero."""
    p = R[key].argmax(1)
    fig, axes = plt.subplots(2, 4, figsize=(TXT_W, 3.55))
    for n, (ax, s) in enumerate(zip(axes.ravel(), ITEMS)):
        m = D.item == ITEMS.index(s)
        k = MAX_SCORE[s]
        cm = np.zeros((k + 1, k + 1), int)
        np.add.at(cm, (D.score[m], p[m]), 1)
        ax.imshow(cm, cmap='Blues', norm=PowerNorm(0.45, vmin=0, vmax=max(cm.max(), 1)), aspect='auto')
        for i in range(k + 1):
            for j in range(k + 1):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] > 0.35 * cm.max() else 'black')
        ax.set_xticks(range(k + 1), range(k + 1))
        ax.set_yticks(range(k + 1), range(k + 1))
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.4)
            sp.set_color('0.6')
        ax.set_title(ITEM_LABEL[s], pad=2)
        if n % 4 == 0:
            ax.set_ylabel('Expert score')
        if n >= 4:
            ax.set_xlabel('Predicted score')
    return save(fig, 'Fig_Confusion.pdf', P, log)


def embeddings(E, P=None, log=print):
    """t-SNE of out-of-fold embeddings; colour marks drawings at full marks and below full marks."""
    coords = E['coords']
    panels = list(dict.fromkeys(coords.Panel))
    fig, axes = plt.subplots(1, len(panels), figsize=(TXT_W, 2.55), squeeze=False)
    for n, (ax, key) in enumerate(zip(axes[0], panels)):
        c = coords[coords.Panel == key]
        full = ~c['Below full marks'].values.astype(bool)
        ax.scatter(c.x[full], c.y[full], s=1.6, color=BLUE, alpha=0.45, lw=0, rasterized=True)
        ax.scatter(c.x[~full], c.y[~full], s=1.6, color=ORANGE, alpha=0.75, lw=0, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
        ax.set_title(f'({"abc"[n]}) {c.Title.iloc[0]}', pad=3)
    handles = [Line2D([], [], marker='o', ls='none', color=BLUE, ms=4, label='Full marks'),
               Line2D([], [], marker='o', ls='none', color=ORANGE, ms=4, label='Below full marks')]
    fig.legend(handles=handles, loc='outside upper center', ncol=2, handletextpad=0.3, columnspacing=1.5,
               borderaxespad=0.1)
    return save(fig, 'Fig_Embeddings.pdf', P, log)


def training(D, P, keys=None, log=print):
    """Training and validation accuracy per epoch of every network: mean over the five folds (lines) and range
    (shading), for the run that produced the evaluated weights."""
    from .training import unit_paths
    data = {}
    for key in keys or NETWORK_KEYS:
        hs = []
        for k in range(D.n_folds):
            f = unit_paths(P, key, k).history
            if os.path.exists(f):
                h = pd.read_csv(f)
                hs.append(h[h.attempt == h.attempt.max()])
        if hs:
            data[key] = hs
    if not data:
        return []
    ncols = 4
    nrows = int(np.ceil(len(data) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(TXT_W, 1.42 * nrows + 0.65), sharex=True, sharey=True,
                             squeeze=False)
    flat = axes.ravel()
    low = 100.0
    last = int(max(h.epoch.max() for hs in data.values() for h in hs))
    for ax, (key, hs) in zip(flat, data.items()):
        E = int(max(h.epoch.max() for h in hs))
        ep = np.arange(1, E + 1)
        for col, color, ls in (('train_accuracy', '0.35', '--'), ('val_accuracy', ROLE_COLOR[role(key)], '-')):
            M = np.full((len(hs), E), np.nan)
            for r, h in enumerate(hs):
                M[r, h.epoch.values.astype(int) - 1] = 100 * h[col].values
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                lo_, hi_, mean = np.nanmin(M, 0), np.nanmax(M, 0), np.nanmean(M, 0)
            ax.fill_between(ep, lo_, hi_, color=color, alpha=0.15, lw=0)
            ax.plot(ep, mean, color=color, ls=ls, lw=1.0)
            low = min(low, float(np.nanmin(lo_)))
        ax.set_title(label(key), pad=2)
        ax.grid(color='0.92', lw=0.5)
        ax.set_axisbelow(True)
    for i, ax in enumerate(flat):
        if i >= len(data):
            ax.axis('off')
        elif i + ncols >= len(data):
            ax.tick_params(labelbottom=True)
    flat[0].set_ylim(max(0.0, np.floor(low / 10) * 10), 100)
    flat[0].set_xlim(0.5, last + 0.5)
    flat[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, integer=True, min_n_ticks=1))
    handles = [Line2D([], [], color='0.35', ls='--', lw=1.0, label='Training (augmented batches)')] + \
              [Line2D([], [], color=ROLE_COLOR[r], ls='-', lw=1.0, label=f'Validation, {ROLE_NAME[r].lower()}')
               for r in _roles_present([role(k) for k in data])] + \
              [Patch(color='0.35', alpha=0.15, label='Range over the folds')]
    fig.legend(handles=handles, loc='outside upper center', ncol=len(handles), handlelength=1.8, columnspacing=1.4,
               handletextpad=0.5, borderaxespad=0.1)
    fig.supxlabel('Epoch')
    fig.supylabel('Accuracy (%), mean of the items')
    return save(fig, 'Fig_Training.pdf', P, log)
