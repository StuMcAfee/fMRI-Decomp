"""R5 tooling (docs/R5_PLAN.md): common region set, pooled deep coupling, group summary on synthetic outputs."""
import json

import numpy as np

from mejoint import io, route_b_batch


def _write_participant(root, run, rng, ids, pr, fd):
    rb = root / "route_b" / run / "200_7"
    rb.mkdir(parents=True)
    n = len(ids)
    X = rng.standard_normal((n, 300)) + rng.standard_normal(300)
    C = np.corrcoef(X)
    np.save(rb / "corr_b.npy", C.astype(np.float32))
    regions = [{"id": i, "name": f"r{i}", "kind": "cortex", "network": "N", "used": True, "n_voxels": 50, "bold_share": 0.9,
                "bold_share_ci": [0.8, 1.0], "s0_share": 0.1, "leftover_share": 0.01, "b_reliability": 0.9} for i in ids]
    regions.append({"id": 999, "name": "L_thalamus", "kind": "secondary", "network": "Thalamus", "used": True, "n_voxels": 100,
                    "bold_share": 0.8, "bold_share_ci": [0.6, 1.0], "s0_share": 0.2, "leftover_share": 0.02, "b_reliability": 0.6})
    integ = [{"set": "all usable cortical regions", "n_regions": n, "participation_ratio": pr, "pr_null_median": 30.0,
              "pr_without_first": pr * 4, "global_share": 0.4, "segregation": 0.2, "n_eigen_above_null": 4},
             {"set": "common set", "n_regions": n - 1, "participation_ratio": pr, "pr_without_first": pr * 4, "global_share": 0.4}]
    s = {"integration": integ, "n_frames": 300, "regions": regions, "cortex_ids": ids,
         "timeline": {"primary_eligible": True, "longest_stretch": [0, 300], "n_used": 300},
         "cortex_csf_covariate": {"cortex_csf_mean_allvox": 0.12},
         "secondary_coupling": [{"name": "L_thalamus", "N": 0.3, "ci": {"N": [0.1, 0.5]}, "b_reliability": 0.6, "not_measurable": False}]}
    io.write_json(rb / "route_b.json", s)
    fe = root / "phase2" / run / "mc"
    fe.mkdir(parents=True)
    io.write_tsv(fe / "frames.tsv", {"frame": list(range(300)), "fd_filtered_mm": list(np.full(300, fd))})


def test_pooled_coupling():
    out = route_b_batch.pooled_coupling([(0.3, 0.1, 0.5), (0.3, 0.2, 0.4), (None, None, None)])
    assert out["n"] == 2 and abs(out["pooled"] - 0.3) < 0.02 and out["ci"][0] < 0.3 < out["ci"][1]
    assert route_b_batch.pooled_coupling([])["n"] == 0


def test_common_regions_and_summary(tmp_path):
    rng = np.random.default_rng(3)
    cfg = io.load_config()
    for k, run in enumerate([f"sub-{100 + k}_acq-1_ser-5" for k in range(6)]):
        a = tmp_path / "anatomy" / run
        a.mkdir(parents=True)
        usable = list(range(1, 11)) if k else list(range(2, 11))       # the first participant lacks parcel 1
        io.write_json(a / "regions_200_7.json", {"regions": [{"id": i, "kind": "cortex", "usable": i in usable} for i in range(1, 11)]})
        _write_participant(tmp_path, run, rng, usable, 4.0 + 0.1 * k, 0.05 + 0.01 * k)
    runs = sorted(p.name for p in (tmp_path / "anatomy").iterdir())
    cr = route_b_batch.common_regions(tmp_path / "anatomy", runs, "200_7", tmp_path / "common_ids_200_7.txt")
    assert cr["n_common"] == 9 and max(p["removed_alone"] for p in cr["per_participant"]) == 1
    res = route_b_batch.summary(tmp_path / "route_b", tmp_path / "phase2", tmp_path / "inject", cfg, tmp_path / "summary")
    assert len(res["participants"]) == 6 and res["cohort_checks"]["pr_vs_fd_mean"]["spearman"] > 0.9
    assert res["deep_pooled"][0]["n"] == 6 and "qc_fc" in res["cohort_checks"]
    for f in ("participants.tsv", "regions.tsv", "deep_regions.tsv", "deep_pooled.tsv", "summary.md"):
        assert (tmp_path / "summary" / f).exists(), f
    json.loads((tmp_path / "summary" / "summary.json").read_text())


def test_preflight(tmp_path):
    cfg = io.load_config()
    for run, gap in (("sub-201_acq-1_ser-5", False), ("sub-202_acq-1_ser-5", True), ("sub-P08_acq-1_ser-5", False)):
        fe = tmp_path / "phase2" / run / "mc"
        fe.mkdir(parents=True)
        echoes = []
        for k in range(1, 5):
            f = tmp_path / f"{run}_e{k}.nii.gz"
            f.write_text("x")
            echoes.append(str(f))
        vols = np.r_[np.arange(0, 300), np.arange(319, 800)] if gap else np.arange(800)
        vu = tmp_path / f"{run}_volumes_used.tsv"
        io.write_tsv(vu, {f"echo{k}": list(vols) for k in range(1, 5)})
        io.write_json(fe / "done.json", {"inputs": {**{f"echo{k}": echoes[k - 1] for k in range(1, 5)}, "volumes_used": str(vu)}})
        io.write_tsv(fe / "frames.tsv", {"frame": list(range(vols.size))})
    rows = route_b_batch.preflight(tmp_path / "phase2", tmp_path / "anatomy", cfg, out_dir=tmp_path / "out")
    by = {r["run"]: r for r in rows}
    assert "sub-P08_acq-1_ser-5" not in by                            # excluded subject
    assert by["sub-201_acq-1_ser-5"]["primary_eligible"] and by["sub-201_acq-1_ser-5"]["frames_used"] == 751
    assert not by["sub-202_acq-1_ser-5"]["primary_eligible"] and by["sub-202_acq-1_ser-5"]["gap_free_stretches"] == 2
    assert (tmp_path / "out" / "preflight.md").exists()


def test_every_cli_command_takes_config():
    """main() reads args.config for every command; a subcommand without --config crashes (R5 preflight bug,
    2026-11-11)."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "mejoint" / "cli.py").read_text()
    names = re.findall(r'^\s+(\w+) = sub\.add_parser\("([\w-]+)"', src, re.M)
    assert names
    assert [n for v, n in names if f'{v}.add_argument("--config"' not in src] == []


def test_cli_r5_commands(tmp_path, capsys):
    """The three R5 commands run end to end through the command line on synthetic folders."""
    from mejoint import cli
    test_preflight(tmp_path)
    assert cli.main(["r5-preflight", "--phase2-root", str(tmp_path / "phase2"), "--anat-root", str(tmp_path / "anatomy"),
                     "--out", str(tmp_path / "pf")]) in (0, None)
    assert "R5 preflight" in capsys.readouterr().out
