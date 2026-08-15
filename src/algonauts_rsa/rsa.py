"""Representational similarity analysis preserving notebook behavior."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def choose_random_indices(n: int, k: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(k, n), replace=False))


def compute_rdm(values, metric="correlation", zscore_columns=True):
    array = np.asarray(values).copy()
    if zscore_columns:
        array = (array - array.mean(axis=0, keepdims=True)) / (
            array.std(axis=0, keepdims=True) + 1e-8
        )
    return squareform(pdist(array, metric=metric))


def rsa_spearman_perm(rdm1, rdm2, n_perm=1000, seed=42):
    upper = np.triu_indices_from(rdm1, k=1)
    values1, values2 = rdm1[upper], rdm2[upper]
    rho, _ = spearmanr(values1, values2)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        permutation = rng.permutation(rdm2.shape[0])
        permuted = rdm2[permutation][:, permutation][upper]
        permuted_rho, _ = spearmanr(values1, permuted)
        if abs(permuted_rho) >= abs(rho):
            count += 1
    return float(rho), float((count + 1) / (n_perm + 1))
