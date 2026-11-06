"""`mejoint sweep-summary`: the joint-ICA sweep across participants, on a folder laid out like a batch."""
import json

import numpy as np

from mejoint import cli


def comp(j, label, eta, above=True, pv="not partial volume"):
    return {"component": j, "label": label, "above_null": above, "eta": eta, "eta_tissue": eta - 0.05,
            "eta_csf": eta + 0.1, "eta_pv_removed": eta - 0.1, "pv_verdict": pv, "stability_median_abs_r": 0.99,
            "corr_fd_filtered_mm": 0.1 * j}


def test_sweep_summary_tables_and_figure(tmp_path):
    root = tmp_path / "outputs" / "phase5"
    for run in ("sub-1_acq-1_ser-5", "sub-2_acq-1_ser-5"):
        for k in (5, 50):
            d = root / run / "phase4_sweep_mc" / f"k{k}"
            d.mkdir(parents=True)
            cs = [comp(1, "mixed", 0.2), comp(2, "R2*-dominant", 0.01, pv="consistent with partial volume"),
                  comp(3, "not above null", 0.6, above=False)]
            (d / "components.json").write_text(json.dumps({
                "k": k, "components": cs, "explained_variance": 0.4, "stability_median_of_components": 0.99,
                "label_counts": {"R2*-dominant": 1, "S0-dominant": 0, "mixed": 1, "not above null": 1, "undetermined": 0}}))
        (root / run / "phase4_sweep_mc" / "sweep.json").write_text(json.dumps({"rows": []}))
        rk = tmp_path / "outputs" / "phase3" / run / "mc"
        rk.mkdir(parents=True)
        (rk / "rank.json").write_text(json.dumps({"mapca": {"available": True, "aic": 24, "kic": 13, "mdl": 9}}))
    (root / "sub-3_acq-1_ser-5").mkdir()
    out = tmp_path / "sum"
    assert cli.main(["sweep-summary", "--root", str(root), "--out", str(out), "--k", "50"]) == 0
    md = (out / "sweep_batch.md").read_text()
    assert "Runs: 2. Without a sweep: 1 (sub-3_acq-1_ser-5)" in md
    assert "| sub-1_acq-1_ser-5 | 24 / 13 / 9 | k5: 1/0/1/1; k50: 1/0/1/1 |" in md
    assert "| sub-1_acq-1_ser-5 | 2 / 3 | 40.0 % | 0.105 |" in md          # median eta of the two above the null
    assert "partial volume consistent in 2 of 4 judged components" in md
    assert (out / "sweep_batch_eta.png").exists()
    rows = json.loads((out / "sweep_batch.json").read_text())["runs"]
    assert len(rows) == 4 and np.isclose(rows[0]["fd_abs_corr_max"], 0.3)


def test_sweep_summary_strength_and_recheck_columns(tmp_path):
    """Check 1 (eta vs component strength) and check 2 (eta_recheck.json) appear in the summary."""
    root = tmp_path / "outputs" / "phase5"
    d = root / "sub-1_acq-1_ser-5" / "phase4_sweep_mc" / "k50"
    d.mkdir(parents=True)
    cs = [{**comp(j, "mixed", 0.6 - 0.1 * j), "energy_single": 10.0 * j, "energy_null_threshold": 5.0} for j in range(1, 7)]
    (d / "components.json").write_text(json.dumps({"k": 50, "components": cs, "explained_variance": 0.3,
                                                   "stability_median_of_components": 0.99,
                                                   "label_counts": {"mixed": 6}}))
    (d.parent / "sweep.json").write_text(json.dumps({"rows": []}))
    (d / "eta_recheck.json").write_text(json.dumps({
        "noise_level_c0": 1.1, "noise_level_c1": 1.9, "colour_factor_c0": 1.3, "colour_factor_c1": 2.5,
        "components": [{"component": j, "eta_white_empirical": 0.2, "eta_empirical": 0.05} for j in range(1, 7)]}))
    out = tmp_path / "sum"
    assert cli.main(["sweep-summary", "--root", str(root), "--out", str(out)]) == 0
    md = (out / "sweep_batch.md").read_text()
    assert "| sub-1_acq-1_ser-5 | -1.00 | 0.200 / 0.300 | 1.10 / 1.90 | 1.30 / 2.50 | 0.250 / 0.200 / 0.050 |" in md
    assert (out / "sweep_batch_eta_strength.png").exists()
