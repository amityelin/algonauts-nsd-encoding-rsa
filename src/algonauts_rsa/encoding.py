"""Vertex-wise ridge encoding with the original nested-CV behavior."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold


def fast_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    y_mean = np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - (ss_res / ss_tot)
    r2[~np.isfinite(r2)] = np.nan
    return r2


def nested_ridge_encoding(
    features,
    responses,
    outer_folds=5,
    inner_folds=3,
    alphas=None,
    seed=42,
):
    """Preserve the notebook's exact outer KFold and RidgeCV behavior."""
    if alphas is None:
        alphas = np.logspace(-2, 4, 12)
    splitter = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    all_fold_r2, chosen_alphas = [], []
    for train_index, test_index in splitter.split(features):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = responses[train_index], responses[test_index]
        ridge_cv = RidgeCV(alphas=alphas, cv=inner_folds, scoring="r2")
        ridge_cv.fit(x_train, y_train)
        alpha = float(ridge_cv.alpha_)
        chosen_alphas.append(alpha)
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        all_fold_r2.append(fast_r2(y_test, model.predict(x_test)))
    fold_r2 = np.stack(all_fold_r2, axis=0)
    return np.nanmean(fold_r2, axis=0), chosen_alphas, fold_r2
