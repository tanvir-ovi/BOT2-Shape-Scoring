"""Paper figures as vector PDF at IEEE print width, 8 pt serif text, with an automatic check for overlapping text."""
import os
import shutil
import subprocess
import itertools

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .data import ITEMS, ITEM_NAME, ITEM_LABEL, MAX_SCORE
from .models import PROPOSED, PROPOSED_LABEL, ENSEMBLE, NETWORKS, label, group
from .evaluation import MARGIN

COL_W, TXT_W = 252 / 72.27, 516 / 72.27          # IEEE conference column and text width (inches)
FS = 8
GREEN, ORANGE, BLUE, GREY, RED = '#3E9E7E', '#E07B39', '#3B6FB6', '#7F7F7F', '#C0392B'
ROLE_COLOR = {'proposed': GREEN, 'member': ORANGE, 'baseline': BLUE, 'control': BLUE, 'classical': GREY}
ROLE_NAME = {'proposed': 'Proposed ensemble', 'member': 'Ensemble member', 'baseline': 'Other network',
             'classical': 'Feature-based'}
VERDICT_COLOR = {'different': GREEN, 'equivalent': BLUE, 'inconclusive': GREY}
SHORT = {'Overlapped circle': 'Overlapping\ncircles', 'Overlapped pencils': 'Overlapping\npencils'}
PAIR_LABEL = {('dit_224', 'beit_natural'): 'Documents vs natural images', ('beit_sketch', 'beit_natural'): 'Sketches vs natural images',
              ('dit_224', 'beit_sketch'): 'Documents vs sketches', ('convnext_512', 'convnext_224'): '512 px vs 224 px'}


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
        'legend.fontsize': FS, 'figure.titlesize': FS, 'axes.titleweight': 'normal', 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'axes.unicode_minus': True,
        'figure.constrained_layout.use': True, 'figure.constrained_layout.h_pad': 0.02,
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
    arts = list(fig.texts)
    for ax in fig.axes:
        arts += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + ticks(ax.xaxis) + ticks(ax.yaxis)
        arts += list(ax.texts)
        if ax.get_legend() is not None:
            arts += ax.get_legend().get_texts()
    for leg in fig.legends:
        arts += leg.get_texts()
    return [a for a in arts if a.get_visible() and a.get_text().strip()]


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


def _bold_proposed(labels, keys):
    for lab, key in zip(labels, keys):
        if key == PROPOSED:
            lab.set_fontweight('bold')


# ---------------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------------
def accuracy(S, F, P=None, log=print):
    """Mean per-item accuracy of every scorer: folds, all drawings with the child-level 95% interval, mean."""
    order = S.sort_values('Accuracy (%)').index.tolist()
    n = len(order)
    fold_cols = [c for c in F.columns if c.startswith('Fold')]
    fig, ax = plt.subplots(figsize=(COL_W, 0.17 * n + 0.95))
    for y, key in enumerate(order):
        c = ROLE_COLOR[S.loc[key, 'Role']]
        ax.plot([S.loc[key, 'CI low'], S.loc[key, 'CI high']], [y, y], color=c, lw=1.3, solid_capstyle='butt', zorder=2)
        folds = F.loc[S.loc[key, 'Scorer'], fold_cols].values.astype(float)
        ax.scatter(folds, np.full(len(folds), y), s=6, color=c, alpha=0.55, lw=0, zorder=3)
        ax.plot(S.loc[key, 'Accuracy (%)'], y, 'D', mfc='white', mec=c, mew=1.0, ms=4.0, zorder=4)
    ax.set_yticks(range(n), [S.loc[k, 'Scorer'] for k in order])
    _bold_proposed(ax.get_yticklabels(), order)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel('Mean per-item accuracy (%)')
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='0.9', lw=0.5)
    ax.set_axisbelow(True)
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(range(n), [f'{S.loc[k, "Accuracy (%)"]:.2f}' for k in order])
    for lab, key in zip(right.get_yticklabels(), order):
        lab.set_color(ROLE_COLOR[S.loc[key, 'Role']])
    _bold_proposed(right.get_yticklabels(), order)
    right.tick_params(axis='y', length=0, pad=2)
    right.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    right.text(1.0, 1.0, ' Mean', transform=right.transAxes, ha='left', va='bottom')
    roles = [r for r in ('proposed', 'member', 'baseline', 'classical') if r in set(S.Role.replace('control', 'baseline'))]
    handles = [Line2D([], [], marker='o', ls='none', ms=3, color='0.4', alpha=0.6, label='Single fold'),
               Line2D([], [], marker='D', mfc='white', mec='0.3', ls='-', color='0.3', lw=1.3, ms=4.0,
                      label='All drawings, 95% CI')] + \
              [Patch(color=ROLE_COLOR[r], label=ROLE_NAME[r]) for r in roles]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, handlelength=1.6, columnspacing=1.2,
               handletextpad=0.5, borderaxespad=0.1)
    problems = save(fig, 'Fig_Accuracy.pdf', P, log)
    return problems


def comparisons(C, P=None, log=print):
    """Paired differences in mean per-item accuracy with child-level 95% intervals, the equivalence margin and
    Holm-adjusted p values (a: proposed ensemble against every comparator, b: pretraining source and input size)."""
    fams = [f for f in ('RQ1', 'RQ2') if f in set(C.Family)]
    sizes = [int((C.Family == f).sum()) for f in fams]
    fig = plt.figure(figsize=(TXT_W, 0.19 * sum(sizes) + 0.42 * len(fams) + 0.62))
    gs = fig.add_gridspec(len(fams), 2, height_ratios=[s + 1.2 for s in sizes], width_ratios=[2.5, 1.0])
    keys_of = {label(k): k for k in list(NETWORKS) + [PROPOSED, 'majority', 'geometric_gb']}
    lo = min(C['CI95 low'].min(), -MARGIN) - 0.4
    hi = max(C['CI95 high'].max(), MARGIN) + 0.4
    for r, fam in enumerate(fams):
        t = C[C.Family == fam].reset_index(drop=True)
        n = len(t)
        ax = fig.add_subplot(gs[r, 0])
        tab = fig.add_subplot(gs[r, 1], sharey=ax)
        ys = np.arange(n)[::-1]
        ax.axvspan(-MARGIN, MARGIN, color=GREEN, alpha=0.12, lw=0, zorder=0)
        ax.axvline(0, color=RED, ls='--', lw=0.7, zorder=1)
        for y, (_, row) in zip(ys, t.iterrows()):
            c = VERDICT_COLOR[row.Verdict]
            ax.plot([row['CI95 low'], row['CI95 high']], [y, y], color=c, lw=1.6, solid_capstyle='butt', zorder=2)
            ax.plot(row['Delta (points)'], y, 'o', color=c, mec='white', mew=0.5, ms=4.6, zorder=3)
        if fam == 'RQ1':
            names = list(t.B)
            title = f'(a) {PROPOSED_LABEL} minus each comparator'
        else:
            names = [PAIR_LABEL.get((keys_of.get(a), keys_of.get(b)), f'{a} vs {b}') for a, b in zip(t.A, t.B)]
            title = '(b) Pretraining source of BEiT-B and input size of ConvNeXt V2-T'
        ax.set_yticks(ys, names)
        ax.tick_params(axis='y', length=0)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(lo, hi)
        ax.set_title(title, loc='left', pad=3)
        ax.grid(axis='x', color='0.9', lw=0.5)
        ax.set_axisbelow(True)
        if r == len(fams) - 1:
            ax.set_xlabel('Difference in mean per-item accuracy (points)')
        tab.axis('off')
        cols = [(0.02, 'Δ (points)', lambda row: _signed(row['Delta (points)'])),
                (0.40, 'p', lambda row: _fmt_p(row['p (permutation)'])),
                (0.72, 'p (Holm)', lambda row: _fmt_p(row['p (Holm)']))]
        for x, head, fn in cols:
            tab.text(x, n - 0.4, head, ha='left', va='bottom', transform=tab.get_yaxis_transform(), fontweight='bold')
            for y, (_, row) in zip(ys, t.iterrows()):
                tab.text(x, y, fn(row), ha='left', va='center', transform=tab.get_yaxis_transform())
    handles = [Line2D([], [], color=VERDICT_COLOR[v], lw=1.6, marker='o', ms=4.6, mec='white', mew=0.5, label=lab)
               for v, lab in (('different', 'Different'), ('equivalent', 'Equivalent within margin'),
                              ('inconclusive', 'Inconclusive'))] + \
              [Patch(color=GREEN, alpha=0.12, label=f'Equivalence margin (±{MARGIN:g} point)'),
               Line2D([], [], color=RED, ls='--', lw=0.7, label='No difference')]
    fig.legend(handles=handles, loc='outside lower center', ncol=5, handlelength=1.6, columnspacing=1.2,
               handletextpad=0.5, borderaxespad=0.1)
    problems = save(fig, 'Fig_Paired_Comparisons.pdf', P, log)
    return problems


def precision_recall(D, R, S, curves, P=None, log=print):
    """Detection of drawings below full marks, pooled over items: (a) precision-recall curves with average precision,
    (b) operating points of the argmax decision of every scorer over constant-F1 contours."""
    nets = [k for k in S.index if k in R and k != PROPOSED and S.loc[k, 'Role'] != 'classical']
    best_member = max([k for k in nets if S.loc[k, 'Role'] == 'member'], key=lambda k: S.loc[k, 'Accuracy (%)'], default=None)
    best_other = max([k for k in nets if S.loc[k, 'Role'] in ('baseline', 'control')], key=lambda k: S.loc[k, 'Accuracy (%)'],
                     default=None)
    shown = [k for k in (PROPOSED, best_member, best_other, 'geometric_gb') if k in curves]
    colour = {PROPOSED: GREEN, best_member: ORANGE, best_other: BLUE, 'geometric_gb': GREY}
    prevalence = float(np.mean(D.score < D.kmax))
    fig, (a, b) = plt.subplots(1, 2, figsize=(TXT_W, 3.05))
    for k in shown:
        rec, prec = curves[k]
        a.plot(rec, prec, color=colour[k], lw=1.4 if k == PROPOSED else 1.0, zorder=3 if k == PROPOSED else 2,
               label=f'{label(k)} (AP {S.loc[k, "AP below full"]:.3f})')
        a.plot(S.loc[k, 'Recall below full'], S.loc[k, 'Precision below full'], 'o', color=colour[k], mec='white',
               mew=0.6, ms=5, zorder=4)
    a.axhline(prevalence, color='0.6', ls=':', lw=0.8, label=f'No skill ({prevalence:.3f})')
    a.set_title('(a) Precision-recall curves', loc='left', pad=3)
    f_levels = (0.5, 0.6, 0.7, 0.8)
    for c in f_levels:
        rr = np.linspace(c / 2 + 1e-3, 1, 400)
        pp = c * rr / (2 * rr - c)
        keep = pp <= 1
        b.plot(rr[keep], pp[keep], color='0.6', lw=0.6, ls=':', zorder=1)
        b.text(0.995, c / (2 - c) - 0.012, f'F$_1$ = {c:.1f}', ha='right', va='top', color='0.35')
    for k in S.index:
        if k == 'majority' or not np.isfinite(S.loc[k, 'Precision below full']) or S.loc[k, 'Recall below full'] == 0:
            continue
        role = S.loc[k, 'Role']
        b.plot(S.loc[k, 'Recall below full'], S.loc[k, 'Precision below full'], '*' if k == PROPOSED else 'o',
               color=ROLE_COLOR[role], mec='white', mew=0.5, ms=10 if k == PROPOSED else 5, zorder=4 if k == PROPOSED else 3)
    b.set_title('(b) Operating points and constant-F$_1$ contours', loc='left', pad=3)
    for ax in (a, b):
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('Recall for drawings below full marks')
        ax.grid(color='0.92', lw=0.5)
        ax.set_axisbelow(True)
    a.set_ylabel('Precision for drawings below full marks')
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, handlelength=1.6, borderaxespad=0.1)
    roles = [r for r in ('proposed', 'member', 'baseline', 'classical')
             if r in set(S.drop(index='majority', errors='ignore').Role.replace('control', 'baseline'))]
    handles = [Line2D([], [], marker='*' if r == 'proposed' else 'o', ls='none', color=ROLE_COLOR[r], mec='white',
                      ms=9 if r == 'proposed' else 5, label=ROLE_NAME[r]) for r in roles] + \
              [Line2D([], [], color='0.6', ls=':', lw=0.6, label='Constant F$_1$')]
    b.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.6,
             borderaxespad=0.1)
    problems = save(fig, 'Fig_Precision_Recall.pdf', P, log)
    return problems


def item_accuracy(T, S, P=None, log=print):
    """Accuracy (%) of every scorer on every item; the most-frequent-score row is the share of the modal score."""
    by_label = {S.loc[k, 'Scorer']: k for k in S.index}
    order = list(T.sort_values('Mean', ascending=False).index)
    M = T.loc[order].values
    fig, ax = plt.subplots(figsize=(TXT_W, 0.2 * len(order) + 0.85))
    im = ax.imshow(M, cmap='RdYlGn', vmin=40, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', color='white' if M[i, j] < 52 or M[i, j] > 93 else 'black')
    ax.set_xticks(range(M.shape[1]), [SHORT.get(s, ITEM_LABEL[s]) for s in ITEMS] + ['Mean'])
    ax.set_yticks(range(len(order)), order)
    _bold_proposed(ax.get_yticklabels(), [by_label.get(o) for o in order])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(M.shape[1] - 1.5, color='white', lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, aspect=15)
    cb.set_label('Accuracy (%)')
    cb.outline.set_linewidth(0.4)
    problems = save(fig, 'Fig_Item_Accuracy.pdf', P, log)
    return problems


def confusion(D, R, key=PROPOSED, P=None, log=print):
    """Expert against predicted score for every item (out-of-fold); shading is per panel."""
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
    problems = save(fig, 'Fig_Confusion.pdf', P, log)
    return problems


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
    problems = save(fig, 'Fig_Embeddings.pdf', P, log)
    return problems
