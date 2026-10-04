"""Machine-learning scorers on the 64 OpenCV features: logistic regression, a support vector machine with an RBF kernel,
a random forest and gradient boosting, with one model per item fitted on the training part of each fold, and the most
frequent training score of each item as the reference. The hyperparameters are fixed in advance, and every model
predicts the most probable score."""
import os

import numpy as np
import pandas as pd

from . import data as bd
from .data import N_ITEMS, N_CLS, class_counts
from .metrics import mean_item_accuracy
from .scorers import CLASSIFIER_KEYS, label
from .utils import atomic_save, save_csv

SEED = 2026
SETTINGS = {
    'logistic_regression': 'features mapped to normal scores by quantiles, multinomial, C = 1',
    'svm':                 'features mapped to normal scores by quantiles, RBF kernel, C = 1, gamma = scale, Platt probabilities',
    'random_forest':       '500 trees, square-root feature sampling',
    'gradient_boosting':   'histogram gradient boosting, 300 trees of 15 leaves, learning rate 0.05, L2 = 1',
    'majority':            'most frequent training score of the item',
}


def make(key, n_samples):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    normal = lambda: QuantileTransformer(n_quantiles=min(200, n_samples), output_distribution='normal', random_state=SEED)
    if key == 'logistic_regression':
        return make_pipeline(normal(), LogisticRegression(C=1.0, max_iter=5000))
    if key == 'svm':
        return make_pipeline(normal(), SVC(C=1.0, gamma='scale', probability=True, random_state=SEED))
    if key == 'random_forest':
        return RandomForestClassifier(n_estimators=500, max_features='sqrt', n_jobs=-1, random_state=SEED)
    if key == 'gradient_boosting':
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
                                              early_stopping=False, random_state=SEED)
    raise ValueError(key)


def _probabilities(key, X, tr, parts, D):
    """One classifier per item fitted on the training rows; probabilities over the score scale for each part."""
    out = [np.zeros((len(ev), N_CLS), np.float32) for ev in parts]
    for i in range(N_ITEMS):
        a = tr[D.item[tr] == i]
        y = D.score[a]
        clf = make(key, len(a)).fit(X[a], y) if len(np.unique(y)) > 1 else None
        for q, ev in zip(out, parts):
            b = np.where(D.item[ev] == i)[0]
            if not len(b):
                continue
            if clf is None:
                q[b, y[0]] = 1.0
            else:
                q[np.ix_(b, clf.classes_)] = clf.predict_proba(X[ev[b]])
    return out


def run(D, G, P, log=print):
    """Fits every scorer on the training part of each fold; stores validation and test probabilities."""
    X = G.values.astype(np.float32)
    rows = []
    for k in range(D.n_folds):
        tr, va, te = bd.split(D, k)
        mode = class_counts(D.item[tr], D.score[tr]).argmax(1)
        onehot = lambda idx: np.eye(N_CLS, dtype=np.float32)[mode[D.item[idx]]]
        out = {'majority': (onehot(va), onehot(te))}
        for key in CLASSIFIER_KEYS:
            out[key] = tuple(_probabilities(key, X, tr, [va, te], D))
        for key, (q_va, q_te) in out.items():
            atomic_save(os.path.join(P.runs, key, f'fold{k}.npz'),
                        lambda p: np.savez(p, va_idx=va, te_idx=te, q_va=q_va, q_te=q_te))
            rows.append({'Scorer': label(key), 'Fold': k,
                         'Validation accuracy (%)': 100 * mean_item_accuracy(D.score[va], q_va.argmax(1), D.item[va]),
                         'Test accuracy (%)': 100 * mean_item_accuracy(D.score[te], q_te.argmax(1), D.item[te])})
        log(f'Fold {k}: {len(out)} feature-based scorers fitted on {len(tr)} training drawings')
    t = pd.DataFrame(rows)
    save_csv(t, os.path.join(P.metrics, 'classifier_folds.csv'), index=False)
    s = t.groupby('Scorer', sort=False)[['Validation accuracy (%)', 'Test accuracy (%)']].mean()
    s = s.join(t.pivot(index='Scorer', columns='Fold', values='Test accuracy (%)').add_prefix('Test, fold '))
    s['Settings'] = [SETTINGS[k] for k in ['majority', *CLASSIFIER_KEYS]]
    return s.round(2)
