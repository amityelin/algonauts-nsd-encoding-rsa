import json

import pandas as pd
import pytest

from algonauts_rsa.results import FINDINGS_COLUMNS, load_validated_findings


def _write_artifacts(tmp_path):
    rows = []
    for roi in ("EBA", "FFA", "PPA"):
        row = dict.fromkeys(FINDINGS_COLUMNS, 0.1)
        row.update(result_status="preliminary_FAST_MODE", n_subjects=8, roi=roi)
        rows.append(row)
    csv_path = tmp_path / "findings.csv"
    manifest_path = tmp_path / "manifest.json"
    pd.DataFrame(rows, columns=FINDINGS_COLUMNS).to_csv(csv_path, index=False)
    manifest_path.write_text(
        json.dumps({"status": "preliminary FAST_MODE results", "n_subjects": 8,
                    "rois": ["EBA", "FFA", "PPA"]}), encoding="utf-8"
    )
    return csv_path, manifest_path


def test_valid_findings_load(tmp_path):
    csv_path, manifest_path = _write_artifacts(tmp_path)
    frame, manifest = load_validated_findings(csv_path, manifest_path)
    assert tuple(frame.roi) == ("EBA", "FFA", "PPA")
    assert manifest["n_subjects"] == 8


def test_findings_schema_is_exact(tmp_path):
    csv_path, manifest_path = _write_artifacts(tmp_path)
    frame = pd.read_csv(csv_path).drop(columns=["median_paired_delta_rsa_rho"])
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_validated_findings(csv_path, manifest_path)


def test_findings_rois_are_exact(tmp_path):
    csv_path, manifest_path = _write_artifacts(tmp_path)
    frame = pd.read_csv(csv_path)
    frame.loc[2, "roi"] = "V1"
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="exactly these ROIs"):
        load_validated_findings(csv_path, manifest_path)
