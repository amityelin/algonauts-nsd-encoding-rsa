"""Group-level helpers copied from the comparison notebook."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, shapiro


def shapiro_normality(values, min_n=3):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < min_n:
        return False, np.nan
    try:
        _, p_value = shapiro(values)
        return p_value > 0.05, p_value
    except Exception:
        return False, np.nan


def cohens_dz(differences):
    differences = np.asarray(differences, float)
    differences = differences[np.isfinite(differences)]
    if differences.size < 2 or np.std(differences, ddof=1) == 0:
        return np.nan
    return np.mean(differences) / np.std(differences, ddof=1)


def rank_biserial_effect(differences):
    differences = np.asarray(differences, float)
    differences = differences[np.isfinite(differences)]
    differences = differences[differences != 0]
    if differences.size == 0:
        return np.nan
    ranks = rankdata(np.abs(differences))
    denominator = differences.size * (differences.size + 1) / 2.0
    return (np.sum(ranks[differences > 0]) - np.sum(ranks[differences < 0])) / denominator


def fdr_bh(p_values, alpha=0.05):
    p_values = np.asarray(p_values, dtype=float)
    mask = np.isfinite(p_values)
    finite = p_values[mask]
    if finite.size == 0:
        return np.full_like(p_values, np.nan), np.zeros_like(p_values, bool)
    order = np.argsort(finite)
    ranked = finite[order]
    adjusted = np.empty(finite.size, float)
    previous = 1.0
    for index in range(finite.size - 1, -1, -1):
        previous = min(previous, ranked[index] * finite.size / (index + 1))
        adjusted[order[index]] = previous
    result = np.full_like(p_values, np.nan)
    result[mask] = adjusted
    return result, result < alpha


def bootstrap_ci(values, statistic=np.median, n_boot=5000, ci=95, seed=42):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    estimates = [statistic(rng.choice(values, size=values.size, replace=True)) for _ in range(n_boot)]
    low = np.percentile(estimates, (100 - ci) / 2)
    high = np.percentile(estimates, 100 - (100 - ci) / 2)
    return statistic(values), (low, high)
