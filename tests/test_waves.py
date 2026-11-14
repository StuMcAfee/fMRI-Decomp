"""W2 (WAVES_BRIEF 8.2): the individual wave stage on simulated data at the real TEs, TR and multiband slice
timing. Each test states its simulation; the slow set (coverage, pure-noise false positives) runs separately."""
import os

import numpy as np
import pytest
from scipy import signal

from mejoint import frontend, io, simulate, waves

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))
BASE = {"grid": [32, 32, 24], "n_volumes": 600, "sources": [], "respiration": {"enabled": False}, "spikes": {"n": 0},
        "skull_base": {"enabled": False}, "slice_timing": {"enabled": True}}
NEURAL = {"name": "n", "kind": "neural", "geometry": "planar", "speed_mm_s": 6.0, "direction": [1, 0, 0], "q": 0.5,
          "band_hz": [0.03, 0.05]}
STANDING = {"name": "s", "kind": "standing", "q": 0.5, "band_hz": [0.03, 0.05]}
# BOLD-positive hyperaemia: inflow raises S0 (s > 0) while dR2* falls (q < 0), so s and -TE q share a sign and
# z_e = s - TE_e q never passes through zero. With q > 0 the two parts cancel near TE = s/q and the TE lag
# picks up an arbitrary half-cycle wrap (recorded in docs/decisions.md, W2).
VASC = {"name": "v", "kind": "vascular", "speed_mm_s": 60.0, "s": 0.02, "q": -0.5, "tissue_delay_s": 3.0,
        "band_hz": [0.03, 0.05]}


def cfg_(**w):
    c = io.load_config()
    return io.merge(c, {"simulate": {"echo_sigma_scale": c["frontend"]["echo_noise_ratios"]},
                        "waves": {"n_modes": 4, "bootstrap_modes": 1, **w}})


def run(wspec, seed_off=0, bootstrap=0, surrogates=0, over=None, **w):
    cfg = cfg_(**w)
    sim = simulate.simulate(cfg, seed=SEED * 100 + 30 + seed_off, overrides={**BASE, "waves": [wspec], **(over or {})})
    fe = frontend.run(sim.data[sim.mask], sim.te_ms, cfg, sim.mask)
    out = waves.run(fe["c"][:, 0], fe["c"][:, 1], fe["R"], sim.mask, sim.voxel_mm, sim.te_ms, sim.tr_s, cfg,
                    slice_offsets_s=sim.slice_offsets_s, seed=SEED, bootstrap=bootstrap, surrogates=surrogates)
    return sim, fe, out


@pytest.fixture(scope="module")
def neural():
    return run(NEURAL, bootstrap=40)


@pytest.fixture(scope="module")
def vascular():
    return run(VASC, seed_off=1, bootstrap=40)


def test_neural_wave_speed_direction_and_coherence(neural, record_property):
    sim, fe, out = neural
    m = out["measures"][0]
    record_property("simulation", sim.describe())
    record_property("mode1", {k: m[k] for k in ("speed_mm_s", "direction", "R", "fit_quality", "f_c_hz")})
    assert m["speed_mm_s"] == pytest.approx(6.0, rel=0.10)
    cosang = float(np.dot(m["direction"], [1, 0, 0]))
    assert np.degrees(np.arccos(np.clip(cosang, -1, 1))) < 15
    assert m["R"] > 0.8
    assert abs(m["lag_vs_slice_time_slope"]) < 0.15                   # slice timing corrected


def test_lag_sign_convention(neural):
    """Later voxels (larger planted delay) get larger lag: lag = -arg(q) / (2 pi f_c)."""
    sim, fe, out = neural
    lag = out["measures"][0]["_lag"]
    true = sim.waves[0]["delay_s"][sim.mask]
    period = 1 / out["f_c"][0]
    d = np.angle(np.exp(2j * np.pi * (lag - true) / period))            # wrapped difference
    assert np.std(d) * period / (2 * np.pi) < 1.0                       # lag tracks the planted delay (up to a constant)


@pytest.mark.parametrize("stc", [True, False])
def test_standing_wave_and_the_slice_timing_artifact(stc, record_property):
    sim, fe, out = run(STANDING, seed_off=2, slice_timing_correction=stc)
    m = out["measures"][0]
    record_property("mode1", {k: m[k] for k in ("speed_mm_s", "fit_quality", "lag_vs_slice_time_slope")})
    if stc:   # no spurious propagation
        assert m["speed_mm_s"] > 100 and abs(m["lag_vs_slice_time_slope"]) < 0.15
    else:     # the artifact is there, with the sign the correction undoes (later slices lead)
        assert -1.2 < m["lag_vs_slice_time_slope"] < -0.8


def test_eta_neural_in_the_r2_band_and_vascular_above_it(neural, vascular, record_property):
    margin = io.load_config()["metrics"]["eta_label_margin"]
    mn, mv = neural[2]["measures"][0], vascular[2]["measures"][0]
    record_property("eta", {"neural": (mn["eta"], mn["eta_band_R2"]), "vascular": (mv["eta"], mv["eta_band_R2"])})
    assert mn["eta"] <= mn["eta_band_R2"][1] + margin
    assert mv["eta"] > mv["eta_band_R2"][1] + margin


def test_te_dependent_lag(neural, vascular, record_property):
    """Neural: the TE4 - TE1 lag interval includes 0. Vascular (dS0 inflow, dR2* 3 s later): excludes 0,
    positive (late echoes follow the delayed dR2*)."""
    cn = neural[2]["bootstrap"]["ci"]["te_lag_median_beyond_null_s"][0]
    cv = vascular[2]["bootstrap"]["ci"]["te_lag_median_beyond_null_s"][0]
    record_property("ci", {"neural": cn, "vascular": cv})
    assert cn[0] <= 0 <= cn[1]
    assert cv[0] > 0
    sim, fe, out = vascular
    tube = sim.waves[0]["tube"][sim.mask]
    dl = out["measures"][0]["_dlag"]
    assert np.nanmedian(dl[tube]) > 0.3                                # in the vessel itself


def test_c0_only_matches_an_independent_cpca_reference(record_property):
    """Brief 4.6: the main code on c0 alone (zscore_c0, equal weights, linear form, no slice timing) equals a
    minimal reference: SVD of the band-passed analytic signal of the optimally combined data, z-scored per voxel,
    written here independently of waves.py."""
    c = io.merge(cfg_(voxel_scaling="zscore_c0", slice_timing_correction=False),
                 {"frontend": {"weighting": "equal", "signal_form": "linear"},
                  "simulate": {"echo_sigma_scale": [1.0, 1.0, 1.0, 1.0], "snr0": 400.0}})
    sim = simulate.simulate(c, seed=SEED * 100 + 40, overrides={**BASE, "slice_timing": {"enabled": False},
                                                                 "waves": [NEURAL, {**STANDING, "q": 0.3}]})
    S = sim.data[sim.mask].astype(np.float64)
    fe = frontend.run(sim.data[sim.mask], sim.te_ms, c, sim.mask)
    ok = fe["use"].all(1)
    # --- independent reference
    te = sim.te_ms
    w = te[None, :] * np.exp(-te[None, :] / fe["t2s_ms"][ok, None])
    oc = np.einsum("ve,vet->vt", w / w.sum(1, keepdims=True), S[ok])
    sos = signal.butter(4, [0.01, 0.10], btype="bandpass", fs=1 / sim.tr_s, output="sos")
    a = signal.hilbert(signal.sosfiltfilt(sos, oc - oc.mean(1, keepdims=True), axis=1), axis=1)
    a = a / np.real(a).std(1, keepdims=True)
    U, sv, Vh = np.linalg.svd(a, full_matrices=False)
    ref_tc = Vh[:3].T                                                   # time courses (up to phase)
    # --- main code, c0 only
    m3 = np.zeros(sim.mask.shape, bool)
    idx = np.argwhere(sim.mask)[ok]
    m3[tuple(idx.T)] = True
    out = waves.run(fe["c"][ok, 0], fe["c"][ok, 1], fe["R"][ok], m3, sim.voxel_mm, te, sim.tr_s, c, seed=0,
                    bootstrap=0, surrogates=0, channels="c0")
    j, r, _ = waves.match_modes(ref_tc, out["tc"][:, :3])       # waves tc = conj(eigvec of a^H a) = the Vh rows
    record_property("matched_abs_r", np.round(r, 5).tolist())
    assert (r > 0.99).all()


def test_contiguous_block_and_fractional_delay():
    blk = waves.contiguous_block(np.r_[np.arange(100), np.arange(102, 300), np.arange(330, 400)], 368, 2)
    assert blk["grid"].size == 300 and blk["n_interpolated"] == 2       # the 2-volume gap is filled, not the 30
    t = np.arange(400) * 0.75
    x = np.sin(2 * np.pi * 0.04 * t)[None]
    y = waves.fractional_delay(x, np.array([1.3]), 0.75)
    assert np.allclose(y[0, 50:350], np.sin(2 * np.pi * 0.04 * (t[50:350] - 1.3)), atol=1e-3)


def test_complex_backsubstitution_is_exact():
    rng = np.random.default_rng(SEED)
    V = 30
    R = np.zeros((V, 2, 2))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 1] = rng.uniform(5, 50, V), -rng.uniform(1, 30, V), rng.uniform(2, 20, V)
    s = (rng.normal(size=V) + 1j * rng.normal(size=V)) * 0.01
    q = (rng.normal(size=V) + 1j * rng.normal(size=V))
    m1 = R[:, 1, 1] * s
    m0 = R[:, 0, 0] * q / 1000 + R[:, 0, 1] * s
    s2, q2 = waves.backsub_complex(m0[:, None], m1[:, None], R)
    assert np.allclose(s2[:, 0], s) and np.allclose(q2[:, 0], q)


@pytest.mark.slow
def test_pure_noise_false_positive_rate(record_property):
    """Brief 8.2: no mode above the energy null in more than 5 % of pure-noise runs."""
    hits = 0
    n = 20
    for i in range(n):
        cfg = cfg_()
        sim = simulate.simulate(cfg, seed=SEED * 1000 + 500 + i, overrides={**BASE, "grid": [20, 20, 16], "n_volumes": 450,
                                                                              "waves": []})
        fe = frontend.run(sim.data[sim.mask], sim.te_ms, cfg, sim.mask)
        out = waves.run(fe["c"][:, 0], fe["c"][:, 1], fe["R"], sim.mask, sim.voxel_mm, sim.te_ms, sim.tr_s, cfg,
                        slice_offsets_s=sim.slice_offsets_s, seed=i, bootstrap=0, surrogates=40)
        hits += any(m["above_null"] for m in out["measures"])
    record_property("runs_with_a_mode_above_null", f"{hits} / {n}")
    assert hits <= 1


@pytest.mark.slow
def test_bootstrap_coverage_of_the_true_speed(record_property):
    """Brief 8.2: the 95 % bootstrap interval of the speed covers the true speed in 90-98 % of 100 runs."""
    cover = 0
    n = 100
    for i in range(n):
        cfg = cfg_(n_modes=2)
        sim = simulate.simulate(cfg, seed=SEED * 1000 + i, overrides={**BASE, "grid": [20, 20, 16], "n_volumes": 450,
                                                                        "snr0": 60.0, "waves": [{**NEURAL, "q": 0.25}]})
        fe = frontend.run(sim.data[sim.mask], sim.te_ms, cfg, sim.mask)
        out = waves.run(fe["c"][:, 0], fe["c"][:, 1], fe["R"], sim.mask, sim.voxel_mm, sim.te_ms, sim.tr_s, cfg,
                        slice_offsets_s=sim.slice_offsets_s, seed=i, bootstrap=100, surrogates=0)
        lo, hi = out["bootstrap"]["ci"]["speed_mm_s"][0]
        cover += lo <= 6.0 <= hi
    record_property("coverage", cover / n)
    assert 0.90 <= cover / n <= 0.98


def test_waves_command_writes_outputs_and_figures(tmp_path):
    """`mejoint waves` on a real-layout front-end folder: SliceTiming read from the echo-1 sidecar, modes,
    maps, both figures, summary and done.json written; `mejoint decompose --waves` reaches the same code path."""
    import json
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_frontend import make_run_folder

    from mejoint import cli
    cfg = io.merge(cfg_(), {"frontend": {"mppca": {"enabled": False}}})
    sim = simulate.simulate(cfg, seed=SEED * 100 + 50, overrides={**BASE, "grid": [24, 24, 16], "n_volumes": 420,
                                                                   "waves": [NEURAL]})
    rd = make_run_folder(sim, tmp_path)
    frontend.run_subject(rd, cfg, tmp_path / "fe", source="raw", nuisance=False)
    cpath = tmp_path / "cfg.yaml"
    import yaml
    cpath.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}))
    assert cli.main(["waves", "--frontend-dir", str(tmp_path / "fe"), "--out", str(tmp_path / "w"), "--config", str(cpath),
                     "--n-modes", "3", "--plot-modes", "1", "--bootstrap", "5", "--surrogates", "3"]) == 0
    w = tmp_path / "w"
    for f in ("modes.npz", "modes.tsv", "waves.json", "waves_summary.md", "done.json", "config_used.yaml",
              "figures/wave_mode_1.png", "figures/wave_diagnostics_1.png"):
        assert (w / f).exists(), f
    for f in ("q_amp", "q_phase", "s_amp", "s_phase", "eta", "lag", "phase_lag_deg", "hilbert_phase_lag_deg", "plv",
              "explained_fraction", "s0_vs_r2_phase_deg"):
        assert (w / "maps" / f"{f}.nii.gz").exists(), f
    assert (w / "modes_by_region.tsv").exists()
    summ = json.loads((w / "waves.json").read_text())
    assert summ["slice_timing_corrected"] and summ["slice_timing_source"].endswith("sim_e1.json")
    assert summ["modes"][0]["speed_mm_s"] == pytest.approx(6.0, rel=0.15)
    assert summ["motion_source"].endswith("mcflirt.par") and summ["motion"]["null95"] > 0
    assert summ["modes"][0]["plv_median_ref"] > 0.8                    # the direct Hilbert phase locks to the mode
    # phase 0 = the explained-fraction-weighted reference mean; direction in words (+x of a RAS grid = right)
    reg = {r["region"]: r for r in summ["by_region"]["1"]}
    assert abs(reg["reference"]["phase_lag_deg"]) < 0.5 and {"edge", "interior"} <= set(reg)
    assert summ["modes"][0]["direction_label"] == "right" and summ["axcodes"] == "RAS"
    assert "motion_r_trans_y" in summ["modes"][0]
    # motion-orthogonal sensitivity: the mode's time course no longer follows the motion parameters
    assert cli.main(["waves", "--frontend-dir", str(tmp_path / "fe"), "--out", str(tmp_path / "w_mo"), "--config",
                     str(cpath), "--n-modes", "2", "--bootstrap", "0", "--surrogates", "2", "--no-figures",
                     "--motion-orthogonal"]) == 0
    s3 = json.loads((tmp_path / "w_mo" / "waves.json").read_text())
    assert s3["motion_orthogonal"] and max(s3["motion"]["r_max"]) < 1e-6
    assert s3["modes"][0]["speed_mm_s"] == pytest.approx(6.0, rel=0.15)
    # without the correction the slice times are still used for the check (slope ~ -1 would flag the artifact)
    assert cli.main(["waves", "--frontend-dir", str(tmp_path / "fe"), "--out", str(tmp_path / "w_nostc"), "--config",
                     str(cpath), "--n-modes", "2", "--bootstrap", "0", "--surrogates", "0", "--no-figures",
                     "--no-slice-timing"]) == 0
    s2 = json.loads((tmp_path / "w_nostc" / "waves.json").read_text())
    assert not s2["slice_timing_corrected"] and np.isfinite(s2["modes"][0]["lag_vs_slice_time_slope"])
    # the decompose flag runs the same function into <out>/waves
    assert cli.main(["decompose", "--frontend-dir", str(tmp_path / "fe"), "--out", str(tmp_path / "p4"), "--config",
                     str(cpath), "--k", "3", "--waves", "--wave-n-modes", "2", "--wave-bootstrap", "0",
                     "--wave-surrogates", "0"]) == 0
    assert (tmp_path / "p4" / "waves" / "modes.tsv").exists()
    assert not (tmp_path / "p4" / "waves" / "figures").exists()          # figures only with --wave-figures


def test_s0_r2_timing_separates_sign_from_timing():
    """dS0 relative to the BOLD-positive dR2* part: same sign with a delay gives the delay; opposite sign at the
    same time is flagged as sign-opposed with zero folded timing (the case that broke the TE4 - TE1 lag)."""
    V, fc = 400, 0.04
    R = np.zeros((V, 2, 2))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 1] = 40.0, -10.0, 4.0
    vb = np.full(V, 1e-6)
    q = -0.5 * np.ones(V, complex)                                       # BOLD-positive: -q = +0.5, phase 0
    tau = 2.0                                                            # dS0 part 2 s later
    s_late = 0.01 * np.exp(-1j * 2 * np.pi * fc * tau) * np.ones(V)
    r = waves.s0_r2_timing(s_late, q, R, vb, fc, 3.0)
    assert r["n"] == V and r["fraction_opposed"] == 0 and r["timing_median_s"] == pytest.approx(tau)
    r = waves.s0_r2_timing(-0.01 * np.ones(V, complex), q, R, vb, fc, 3.0, f_long=np.linspace(0, 0.9, V))
    assert r["fraction_opposed"] == 1 and r["timing_median_s"] == pytest.approx(0, abs=1e-9)
    fl = np.linspace(0, 0.9, V)
    assert [x["n_present"] for x in r["by_f_long"]] == [int(((fl >= a) & (fl < b)).sum()) for a, b in
                                                        ((0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01))]
    assert all(x["fraction_opposed"] == 1 and x["fraction_same_sign"] == 0 for x in r["by_f_long"])


def test_motion_correlation_finds_a_planted_motion_mode():
    rng = np.random.default_rng(SEED)
    T, tr = 600, 0.75
    W = io.load_config()["waves"]
    blk = waves.contiguous_block(None, T, 2)
    M = np.cumsum(rng.standard_normal((T, 6)), axis=0) * 0.01
    tc = waves.bandpass_analytic(M[:, 3][None], tr, W["band_hz"])[0]
    tc = np.column_stack([tc / np.sqrt(np.mean(np.abs(tc) ** 2)),
                          waves.bandpass_analytic(rng.standard_normal((1, T)), tr, W["band_hz"])[0]])
    mc = waves.motion_correlation(tc, M, blk, tr, W, rng, n_null=50)
    assert mc["r"][0][3] == pytest.approx(1.0, abs=1e-6) and mc["r_max"][0] > mc["null95"]
    assert mc["r_max"][1] < 0.6
    # lag: a mode that follows the motion parameter 2 s later (and with opposite sign: the sign is arbitrary)
    late = waves.bandpass_analytic(-waves.fractional_delay(M[:, 3][None], np.array([2.0]), tr), tr, W["band_hz"])[0]
    fc = waves.mode_frequency(late, tr)
    ml = waves.motion_correlation(late[:, None], M, blk, tr, W, rng, n_null=10, fc=[fc])
    assert ml["r"][0][3] > 0.95 and ml["lag_s"][0][3] == pytest.approx(2.0, abs=0.3)


def test_s0_r2_sign_null_matches_pure_r2_with_noise():
    """With weak dR2* and correlated s / q estimation noise, opposite signs appear without any dS0; the null
    reproduces both the number of voxels passing and the opposed fraction."""
    rng = np.random.default_rng(SEED + 7)
    V, fc = 20000, 0.04
    R = np.zeros((V, 2, 2))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 1] = 40.0, -30.0, 6.0
    vb = np.full(V, 1e-4)
    q_true = -0.5 * np.exp(1j * rng.uniform(0, 2 * np.pi, V)) * rng.uniform(0, 1, V)
    sd = np.sqrt(vb / 2)
    m0 = R[:, 0, 0] * q_true / 1000 + sd * (rng.standard_normal(V) + 1j * rng.standard_normal(V))
    m1 = sd * (rng.standard_normal(V) + 1j * rng.standard_normal(V))
    s, q = waves.backsub_complex(m0[:, None], m1[:, None], R)
    r = waves.s0_r2_timing(s[:, 0], q[:, 0], R, vb, fc, 1.5, rng=rng, n_draws=20)
    assert r["n"] > 200
    assert r["n"] == pytest.approx(r["null_n_expected"], rel=0.15)
    assert r["fraction_opposed"] == pytest.approx(r["null_fraction_opposed"], abs=0.05)


def test_s0_r2_sign_check_on_vascular_and_partial_volume_waves(vascular, record_property):
    """Vascular wave (dS0 inflow leads the BOLD-positive dR2* part by 3 s): voxels with both parts beyond the
    null, mostly same sign, dS0 timing about -3 s. Pure dR2* wave with CSF partial volume: apparent dS0 of
    opposite sign, more often where the CSF fraction is larger (Phase 5 derivation)."""
    mv = vascular[2]["measures"][0]
    record_property("vascular", {k: mv[k] for k in ("s0r2_n_voxels", "s0r2_null_n_expected",
                                                    "s0r2_fraction_sign_opposed", "s0r2_timing_median_s")})
    assert mv["s0r2_n_voxels"] > 5 * max(mv["s0r2_null_n_expected"], 1)
    assert mv["s0r2_fraction_sign_opposed"] < 0.2
    assert -3.6 < mv["s0r2_timing_median_s"] < -2.0
    sim, fe, out = run({**NEURAL, "q": -0.5}, seed_off=3,
                       over={"csf_partial_volume": {"enabled": True, "fraction_max": 0.7}})
    W = io.load_config()["waves"]
    m = out["measures"][0]
    s_c, q_c = waves.backsub_complex(out["m0_cf"][:, [0]], out["m1_cf"][:, [0]], fe["R"])
    r = waves.s0_r2_timing(s_c[:, 0], q_c[:, 0], fe["R"], out["vb_emp0"][:, 0], out["f_c"][0], W["amp_z"],
                           f_long=sim.csf_fraction[sim.mask], vb1_k=out["vb_emp1"][:, 0])
    record_property("partial_volume", {"n": r["n"], "fraction_opposed": r["fraction_opposed"],
                                       "timing": r["timing_median_s"], "by_f_long": r["by_f_long"], "eta": m["eta"]})
    # partial volume: apparent dS0 in phase with the BOLD-positive part (same sign, no timing difference), more
    # often where the CSF fraction is larger; no opposite signs
    assert r["n"] >= 50 and r["fraction_opposed"] < 0.05 and abs(r["timing_median_s"]) < 0.5
    fr = [x["fraction_same_sign"] for x in r["by_f_long"] if np.isfinite(x["fraction_same_sign"])]
    assert len(fr) >= 3 and fr[-1] > fr[0]


def test_project_out_removes_the_motion_subspace():
    rng = np.random.default_rng(SEED)
    T, tr = 500, 0.75
    W = io.load_config()["waves"]
    blk = waves.contiguous_block(None, T, 2)
    M = np.cumsum(rng.standard_normal((T, 6)), axis=0)
    Qm = waves.motion_basis(M, blk, tr, W)
    A = waves.bandpass_analytic(rng.standard_normal((50, T)) + M[:, :1].T, tr, W["band_hz"])
    waves.project_out(A, Qm)
    assert np.abs(A @ np.conj(Qm)).max() < 1e-8 * np.abs(A).max() * T


def test_batch_summary_collects_runs_and_flags_missing(tmp_path):
    """`mejoint waves-summary`: one row per run, variant and mode; runs without results for the commit listed."""
    import json

    from mejoint import cli
    mode = {"mode": 1, "above_null": True, "eigenvalue_over_null": 2.5, "f_c_hz": 0.033, "var_frac_stacked": 0.26,
            "motion_r_max": 0.6, "motion_best_param": "trans_y", "motion_r_null95": 0.4,
            **{f"motion_r_{n}": 0.1 for n in ("rot_x", "rot_y", "rot_z", "trans_x", "trans_z")}, "motion_r_trans_y": 0.6,
            "eta": 0.23, "s0r2_n_voxels": 2198, "s0r2_null_n_expected": 2.0, "s0r2_fraction_sign_opposed": 0.91,
            "direction_label": "posterior-superior", "motion_best_lag_s": 1.5}
    for run in ("sub-1_acq-1_ser-5", "sub-2_acq-1_ser-5"):
        for var in ("noreg", "noreg_motion_orth"):
            d = tmp_path / "waves" / run / "abc1234" / var
            d.mkdir(parents=True)
            (d / "waves.json").write_text(json.dumps({"modes": [mode, {**mode, "mode": 2, "above_null": False}],
                                                      "by_region": {"1": [{"region": "edge", "opposed_per_present": 0.3}]}}))
    (tmp_path / "waves" / "sub-3_acq-1_ser-5").mkdir()
    assert cli.main(["waves-summary", "--root", str(tmp_path / "waves"), "--commit", "abc1234",
                     "--out", str(tmp_path / "sum")]) == 0
    md = (tmp_path / "sum" / "waves_batch.md").read_text()
    assert "motion-coupled (largest motion r above its null): 2 of 2" in md and "sub-3_acq-1_ser-5" in md
    assert "lag of the signal after its strongest motion parameter, coupled runs: median +1.5 s" in md
    assert (tmp_path / "sum" / "waves_batch_motion.png").exists()
    rows = json.loads((tmp_path / "sum" / "waves_batch.json").read_text())["rows"]
    assert len(rows) == 8                                               # 2 runs x 2 variants x 2 modes
