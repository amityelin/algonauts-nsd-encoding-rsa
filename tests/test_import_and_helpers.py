import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from algonauts_rsa.rsa import choose_random_indices


def test_package_import_has_no_filesystem_or_auth_side_effects(tmp_path):
    environment = os.environ.copy()
    environment["HOME"] = str(tmp_path / "home")
    environment["USERPROFILE"] = str(tmp_path / "profile")
    before = set(tmp_path.rglob("*"))
    subprocess.run(
        [sys.executable, "-I", "-c", "import sys; sys.path.insert(0, r'src'); import algonauts_rsa"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
    )
    assert set(tmp_path.rglob("*")) == before


def test_extracted_random_indices_match_original_notebook_algorithm():
    n, k, seed = 100, 17, 42
    rng = np.random.default_rng(seed)
    expected = np.sort(rng.choice(n, size=min(k, n), replace=False))
    np.testing.assert_array_equal(choose_random_indices(n, k, seed), expected)
    np.testing.assert_array_equal(choose_random_indices(5, 20, seed), np.arange(5))
