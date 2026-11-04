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
    for f in ("q_amp", "q_phase", "s_amp", "s_phase", "eta", "lag"):
        assert (w / "maps" / f"{f}.nii.gz").exists(), f
    summ = json.loads((w / "waves.json").read_text())
    assert summ["slice_timing_corrected"] and summ["slice_timing_source"].endswith("sim_e1.json")
    assert summ["modes"][0]["speed_mm_s"] == pytest.approx(6.0, rel=0.15)
    # the decompose flag runs the same function into <out>/waves
    assert cli.main(["decompose", "--frontend-dir", str(tmp_path / "fe"), "--out", str(tmp_path / "p4"), "--config",
                     str(cpath), "--k", "3", "--waves", "--wave-n-modes", "2", "--wave-bootstrap", "0",
                     "--wave-surrogates", "0"]) == 0
    assert (tmp_path / "p4" / "waves" / "modes.tsv").exists()
    assert not (tmp_path / "p4" / "waves" / "figures").exists()          # figures only with --wave-figures
