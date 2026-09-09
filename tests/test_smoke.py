import numpy as np

from algonauts_rsa.config import get_config
from algonauts_rsa.encoding import nested_ridge_encoding
from algonauts_rsa.rsa import compute_rdm, rsa_spearman_perm


def test_named_configurations_are_distinct():
    smoke, fast, full = (get_config(name) for name in ("smoke", "fast", "full"))
    assert (fast.outer_folds, fast.inner_folds, fast.n_permutations, fast.rsa_subset_size) == (3, 2, 200, 600)
    assert (full.outer_folds, full.inner_folds, full.n_permutations, full.rsa_subset_size) == (5, 3, 1000, None)
    assert smoke.debug_n_images == 64


def test_synthetic_encoding_and_rsa_are_deterministic():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(48, 6))
    weights = rng.normal(size=(6, 4))
    responses = features @ weights + rng.normal(scale=0.05, size=(48, 4))
    config = get_config("smoke")

    result1 = nested_ridge_encoding(features, responses, config.outer_folds, config.inner_folds,
                                    config.alphas, config.seed)
    result2 = nested_ridge_encoding(features, responses, config.outer_folds, config.inner_folds,
                                    config.alphas, config.seed)
    np.testing.assert_allclose(result1[0], result2[0])
    np.testing.assert_allclose(result1[2], result2[2])
    assert np.nanmedian(result1[0]) > 0.9

    rdm_features = compute_rdm(features, metric="cosine", zscore_columns=True)
    rdm_responses = compute_rdm(responses, metric="correlation", zscore_columns=True)
    rsa1 = rsa_spearman_perm(rdm_features, rdm_responses, config.n_permutations, config.seed)
    rsa2 = rsa_spearman_perm(rdm_features, rdm_responses, config.n_permutations, config.seed)
    assert rsa1 == rsa2
    assert -1 <= rsa1[0] <= 1
    assert 0 < rsa1[1] <= 1
