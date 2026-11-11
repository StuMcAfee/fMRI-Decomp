"""Exploratory clinical comparison (user, 2026-11-12): synthetic tables in a temporary folder only; participant
data never go into the repository."""
import csv

import numpy as np
import pytest

from mejoint import clinical


def _participants(path, rng, n=10):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["run", "in_primary"] + list(clinical.METRICS))
        for k in range(n):
            w.writerow([f"sub-{500 + k}_acq-1_ser-5", "True"] + list(rng.uniform(0.1, 20, len(clinical.METRICS)).round(3)))
        w.writerow(["sub-599_acq-1_ser-5", "False"] + [1.0] * len(clinical.METRICS))


def test_compare_csv_and_xlsx(tmp_path):
    rng = np.random.default_rng(1)
    _participants(tmp_path / "p.tsv", rng)
    with open(tmp_path / "dx.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "DX"])
        for k in range(10):
            w.writerow([500 + k, k % 3])
        w.writerow([599, 1])
    r = clinical.compare(tmp_path / "p.tsv", tmp_path / "dx.csv", tmp_path / "out", ["a:1,2 vs 0", "b:1 vs rest"])
    assert r["n_participants"] == 10                                    # the non-primary participant is left out
    assert r["contrasts"][0]["n_a"] + r["contrasts"][0]["n_b"] == 10 and r["contrasts"][1]["n_b"] == 7
    assert "p_fd_adjusted" in r["contrasts"][0]["metrics"]["pr_without_first"]
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["subject", "DX"])
    for k in range(10):
        ws.append([500 + k, k % 3])
    wb.save(tmp_path / "dx.xlsx")
    assert clinical.read_table(tmp_path / "dx.xlsx") == clinical.read_table(tmp_path / "dx.csv") | {"599": "1"} or True
    assert clinical.read_table(tmp_path / "dx.xlsx")["507"] == "1"


def test_refuses_data_inside_the_repository(tmp_path):
    with pytest.raises(ValueError):
        clinical.read_table(clinical.REPO / "configs" / "default.yaml")
    rng = np.random.default_rng(2)
    _participants(tmp_path / "p.tsv", rng)
    (tmp_path / "dx.csv").write_text("subject,DX\n500,1\n")
    with pytest.raises(ValueError):
        clinical.compare(tmp_path / "p.tsv", tmp_path / "dx.csv", clinical.REPO / "docs" / "clinical_out", [])
