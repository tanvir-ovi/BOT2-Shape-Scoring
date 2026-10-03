"""Feature-based scorers: the most frequent training score per item, and gradient boosting on 64 geometric features."""
import os
import json

import numpy as np
import pandas as pd
import cv2

from . import data as bd
from .data import N_ITEMS, N_CLS, class_counts, ink_map
from .utils import atomic_save, parallel_map, save_csv

SEED = 2026
K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


# ---------------------------------------------------------------------------------------------------
# Geometric features of the extracted stroke mask
# ---------------------------------------------------------------------------------------------------
def remove_frame(m, band=0.08):
    h, w = m.shape
    lines = cv2.HoughLinesP(m * 255, 1, np.pi / 180, 80, minLineLength=int(0.3 * min(h, w)), maxLineGap=10)
    if lines is None:
        return m
    out = m.copy()
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if abs(y2 - y1) < abs(x2 - x1):
            edge = max(y1, y2) < band * h or min(y1, y2) > (1 - band) * h
        else:
            edge = max(x1, x2) < band * w or min(x1, x2) > (1 - band) * w
        if edge:
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 0, 7)
    return out


def ink_mask(ink, band=0.06, min_area=30):
    from skimage.filters import apply_hysteresis_threshold
    m = apply_hysteresis_threshold(ink, 0.22, 0.5).astype(np.uint8)
    m = remove_frame(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))
    h, w = m.shape
    inner = np.zeros_like(m)
    inner[int(band * h):int((1 - band) * h), int(band * w):int((1 - band) * w)] = 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        comp = lab == i
        if st[i, 4] >= min_area and (comp & (inner == 1)).sum() / st[i, 4] >= 0.5:
            out[comp] = 1
    return out


def angle_diff(a, b, period):
    d = abs(a - b) % period
    return min(d, period - d)


def radial_profile(cnt, bins=72):
    c = cnt[:, 0, :].astype(np.float32)
    d = c - c.mean(0)
    ang = (np.degrees(np.arctan2(-d[:, 1], d[:, 0])) + 360) % 360
    prof = np.zeros(bins)
    np.maximum.at(prof, (ang / (360 / bins)).astype(int) % bins, np.hypot(d[:, 0], d[:, 1]))
    for i in np.where(prof == 0)[0]:
        prof[i] = prof[i - 1]
    return prof / (prof.mean() + 1e-6)


def radial_peaks(prof, prom=0.08, win=4):
    n = len(prof)
    return [i for i in range(n) if prof[i] - prof.min() > prom
            and all(prof[i] > prof[(i - k) % n] for k in range(1, win + 1))
            and all(prof[i] >= prof[(i + k) % n] for k in range(1, win + 1))]


def extent(a, axis):
    idx = np.where(a.any(axis=axis))[0]
    return idx.max() - idx.min() + 1 if len(idx) else 1


def geo_features(m, ink):
    from scipy.ndimage import convolve
    from skimage.morphology import skeletonize
    h, w = m.shape
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return {}
    f = {}
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    D = float(max(bw, bh))

    f['width_rel'], f['height_rel'] = bw / w, bh / h
    f['size_rel'] = np.sqrt(bw * bh / (w * h))
    f['aspect'] = bw / bh
    f['ink_density'] = m.sum() / (bw * bh)
    f['cx_rel'], f['cy_rel'] = (xs.mean() - x0) / bw, (ys.mean() - y0) / bh
    _, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    f['n_comp'] = int((st[1:, 4] > 0.05 * m.sum()).sum())
    xm, ym = (x0 + x1) // 2, (y0 + y1) // 2
    hl, hr = extent(m[y0:y1 + 1, x0:xm], 1), extent(m[y0:y1 + 1, xm:x1 + 1], 1)
    wt, wb = extent(m[y0:ym, x0:x1 + 1], 0), extent(m[ym:y1 + 1, x0:x1 + 1], 0)
    f['half_ratio_lr'] = max(hl, hr) / max(min(hl, hr), 1)
    f['half_ratio_tb'] = max(wt, wb) / max(min(wt, wb), 1)

    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    holes = sorted([c for c, hh in zip(cnts, hier[0]) if hh[3] >= 0], key=cv2.contourArea, reverse=True)
    ha = [cv2.contourArea(c) / (bw * bh) for c in holes]
    sig = [c for c, a in zip(holes, ha) if a > 0.02]
    f['n_holes'] = len(sig)
    for i in range(3):
        f[f'hole{i + 1}_rel'] = ha[i] if i < len(ha) else 0.0
    if len(sig) >= 2:
        (ax_, ay_), (bx_, by_) = [c[:, 0, :].mean(0) for c in sig[:2]]
        f['holes_dx'], f['holes_dy'] = abs(ax_ - bx_) / D, abs(ay_ - by_) / D
    else:
        f['holes_dx'] = f['holes_dy'] = 0.0
    if len(sig) >= 3:
        top3 = sig[:3]
        mid = top3[int(np.argsort([c[:, 0, 1].mean() for c in top3])[1])]
        f['lens_ratio'] = cv2.contourArea(mid) / sum(cv2.contourArea(c) for c in top3)
    else:
        f['lens_ratio'] = 0.0

    ext, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = max(ext, key=cv2.contourArea)
    A, Pm = cv2.contourArea(big), cv2.arcLength(big, True)
    f['circularity'] = 4 * np.pi * A / (Pm ** 2 + 1e-6)
    f['solidity'] = A / (cv2.contourArea(cv2.convexHull(big)) + 1e-6)
    if len(big) >= 5:
        _, (e1, e2), _ = cv2.fitEllipse(big)
        f['ellipse_ratio'] = max(e1, e2) / (min(e1, e2) + 1e-6)
    else:
        f['ellipse_ratio'] = 1.0
    _, (r1, r2), ra = cv2.minAreaRect(big)
    f['rect_ratio'] = max(r1, r2) / (min(r1, r2) + 1e-6)
    f['rect_angle'] = angle_diff(ra, 0, 90)
    mu = cv2.moments(m, binaryImage=True)
    f['axis_angle'] = angle_diff(0.5 * np.degrees(np.arctan2(2 * mu['mu11'], mu['mu20'] - mu['mu02'])), 0, 180)
    lam = np.linalg.eigvalsh(np.array([[mu['mu20'], mu['mu11']], [mu['mu11'], mu['mu02']]]))
    f['elongation'] = np.sqrt(lam[1] / (lam[0] + 1e-6))

    target = sig[0] if sig else big
    Pt = cv2.arcLength(target, True)
    for e in (0.02, 0.04, 0.06):
        f[f'vertices_{int(e * 100)}'] = len(cv2.approxPolyDP(target, e * Pt, True))
    poly = cv2.approxPolyDP(target, 0.04 * Pt, True)[:, 0, :].astype(np.float32)
    sides = np.hypot(*(np.roll(poly, -1, 0) - poly).T)
    f['side_ratio'] = sides.max() / (sides.min() + 1e-6) if len(sides) >= 3 else 0.0
    f['inner_rect_angle'] = angle_diff(cv2.minAreaRect(target)[2], 0, 90)
    prof = radial_profile(target)
    peaks = radial_peaks(prof)
    f['radial_cv'] = prof.std()
    f['radial_range'] = prof.max() / (prof.min() + 1e-6)
    f['radial_peaks'] = len(peaks)
    top = sorted(prof[peaks], reverse=True)[:5]
    f['peak_ratio'] = top[0] / top[-1] if len(top) >= 2 else 1.0
    f['top_peak_dev'] = min([angle_diff(i * 5 + 2.5, 90, 360) for i in peaks], default=90.0)
    spec = np.abs(np.fft.rfft(prof - prof.mean())) / len(prof)
    for k in range(1, 9):
        f[f'fft_{k}'] = spec[k]

    filled = np.zeros_like(m)
    cv2.drawContours(filled, ext, -1, 1, -1)
    fl = filled[y0:y1 + 1, x0:x1 + 1]
    f['sym_lr'] = (fl & fl[:, ::-1]).sum() / ((fl | fl[:, ::-1]).sum() + 1e-6)
    f['sym_ud'] = (fl & fl[::-1, :]).sum() / ((fl | fl[::-1, :]).sum() + 1e-6)
    f['top_heavy'] = fl[:bh // 2].sum() / (fl.sum() + 1e-6)
    for name, lines in (('row', [int(y0 + bh * q) for q in (0.3, 0.5, 0.7)]),
                        ('col', [int(x0 + bw * q) for q in (0.3, 0.5, 0.7)])):
        cr = []
        for L in lines:
            v = (m[L, x0:x1 + 1] if name == 'row' else m[y0:y1 + 1, L]).astype(int)
            cr.append((np.diff(v) == 1).sum() + v[0])
        f[f'cross_{name}'] = float(np.mean(cr))

    sk = skeletonize(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0)
    nb = convolve(sk.astype(np.uint8), K8, mode='constant') * sk
    junc = cv2.dilate((nb >= 3).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    _, bl, bst, _ = cv2.connectedComponentsWithStats((sk & ~junc).astype(np.uint8), connectivity=8)
    blen = bst[:, 4] / D
    ends = np.argwhere(nb == 1)
    tails = [blen[i] for i in {bl[y, x] for y, x in ends if bl[y, x] > 0} if blen[i] >= 0.04]
    f['n_branch'] = int((blen[1:] >= 0.04).sum())
    f['n_tails'] = len(tails)
    f['n_junc'] = cv2.connectedComponents(junc.astype(np.uint8))[0] - 1
    f['max_tail'] = max(tails, default=0.0)
    f['sum_tail'] = float(sum(tails))
    f['skel_len'] = sk.sum() / D
    tips = np.array([(y, x) for y, x in ends if blen[bl[y, x]] >= 0.04])
    if len(tips) >= 2:
        dd = np.hypot(*(tips[:, None, :] - tips[None, :, :]).transpose(2, 0, 1))
        dd[np.eye(len(tips), dtype=bool)] = np.inf
        f['end_gap'] = dd.min() / D
    else:
        f['end_gap'] = 0.0
    f['open_curve'] = float(f['n_junc'] == 0 and len(tips) >= 2)

    width = 2 * cv2.distanceTransform(m, cv2.DIST_L2, 5)[sk]
    wm = np.median(width) + 1e-6
    f['stroke_width'] = wm / D
    f['width_p95'] = np.percentile(width, 95) / wm
    f['thick_frac'] = (width > 1.7 * wm).mean()
    v = cv2.GaussianBlur(ink, (5, 5), 0)[sk]
    vm = np.median(v) + 1e-6
    f['ink_p95'] = np.percentile(v, 95) / vm
    f['dark_frac'] = (v > 1.3 * vm).mean()

    hu = cv2.HuMoments(cv2.moments(filled, binaryImage=True)).ravel()
    for i in range(4):
        f[f'hu_{i + 1}'] = -np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-12)
    return f


def extract(path):
    ink = ink_map(path, max_side=512)
    return geo_features(ink_mask(ink), ink)


def geometric_features(D, P, workers=None, log=print):
    """64 geometric descriptors per drawing, computed once and cached in results/data."""
    path = os.path.join(P.data, f'geometric_features{D.tag}.csv')
    if os.path.exists(path):
        g = pd.read_csv(path)
        if g.file.tolist() == D.df.file.tolist():
            return g.drop(columns='file').fillna(0).values.astype(np.float32), list(g.columns[1:])
    log('Extracting geometric features (first run only) ...')
    g = pd.DataFrame(parallel_map(extract, D.df.path.tolist(), workers))
    g.insert(0, 'file', D.df.file.values)
    save_csv(g, path, index=False)
    return g.drop(columns='file').fillna(0).values.astype(np.float32), list(g.columns[1:])


# ---------------------------------------------------------------------------------------------------
# Fitting on each fold
# ---------------------------------------------------------------------------------------------------
def _boosting_probs(X, tr, parts, D):
    """One gradient-boosting classifier per item, fitted on the training part; probabilities for each part."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    out = [np.zeros((len(ev), N_CLS), np.float32) for ev in parts]
    for i in range(N_ITEMS):
        a = tr[D.item[tr] == i]
        y = D.score[a]
        clf = None
        if len(np.unique(y)) > 1:
            clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
                                                 early_stopping=False, random_state=SEED).fit(X[a], y)
        for q, ev in zip(out, parts):
            b = np.where(D.item[ev] == i)[0]
            if not len(b):
                continue
            if clf is None:
                q[b, y[0]] = 1.0
            else:
                q[np.ix_(b, clf.classes_)] = clf.predict_proba(X[ev[b]])
    return out


def run(D, P, workers=None, log=print):
    """Fits both scorers on the training part of every fold and stores their validation and test probabilities."""
    X, _ = geometric_features(D, P, workers, log)
    for k in range(D.n_folds):
        tr, va, te = bd.split(D, k)
        mode = class_counts(D.item[tr], D.score[tr]).argmax(1)
        onehot = lambda idx: np.eye(N_CLS, dtype=np.float32)[mode[D.item[idx]]]
        out = {'majority': (onehot(va), onehot(te)), 'geometric_gb': tuple(_boosting_probs(X, tr, [va, te], D))}
        for key, (q_va, q_te) in out.items():
            path = os.path.join(P.runs, key, f'fold{k}.npz')
            atomic_save(path, lambda p: np.savez(p, va_idx=va, te_idx=te, q_va=q_va, q_te=q_te,
                                                 signature=json.dumps({'scorer': key, 'fold': k, 'seed': SEED})))
    log(f'Feature-based scorers fitted on {D.n_folds} folds ({X.shape[1]} geometric features)')
