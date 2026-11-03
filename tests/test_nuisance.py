"""Phase 2b: nuisance projection (CLAUDE.md section 8).

Required test: projecting the echoes and then forming channels equals projecting the channels.
Also: segments across gaps, FD, the breathing-peak band-stop, the projector's df factors, sigma and
whitening after projection on simulated data, and the end-to-end run with and without regression.
"""
import os

import numpy as np
import pytest

from mejoint import frontend, io, nuisance, simulate

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))
TR = 0.75


@pytest.fixture
def cfg():
    return io.load_config()


def fake_par(T, rng, resp_hz=0.35, resp_mm=0.15):
    """Slow random-walk head motion plus respiratory pseudo-motion along x and z translation."""
    walk = np.cumsum(rng.normal(0, 1, (T, 6)), axis=0)
    walk = np.apply_along_axis(lambda x: np.convolve(x, np.ones(25) / 25, mode="same"), 0, walk)
    walk *= np.array([1e-3] * 3 + [0.05] * 3)
    t = np.arange(T) * TR
    resp = resp_mm * np.sin(2 * np.pi * resp_hz * t + rng.uniform(0, 2 * np.pi))
    par = walk.copy()
    par[:, 3] += resp
    par[:, 5] += 0.7 * resp
    return par + rng.normal(0, 0.003, (T, 6)) * np.array([1e-2] * 3 + [1] * 3), walk


def test_segments_and_fd_respect_gaps():
    vols = np.r_[np.arange(0, 300), np.arange(319, 800)]           # a 19-frame gap, as in run P10
    segs = nuisance.segments(vols)
    assert [(s.start, s.stop) for s in segs] == [(0, 300), (300, 781)]
    par = np.zeros((781, 6))
    par[300:, 3] = 5.0                                             # a jump across the gap is not motion
    fd = nuisance.framewise_displacement(par, segs)
    assert fd[300] == 0 and fd.max() == 0
    rng = np.random.default_rng(SEED)
    p1 = rng.normal(0, 0.01, (50, 6))
    ref = np.abs(np.diff(p1, axis=0))
    expect = np.r_[0, ref[:, :3].sum(1) * 50 + ref[:, 3:].sum(1)]   # mbme_tedana's Power FD
    assert np.allclose(nuisance.framewise_displacement(p1, [slice(0, 50)]), expect)


def test_breathing_peak_is_found_and_band_stopped(cfg):
    rng = np.random.default_rng(SEED + 1)
    T = 800
    par, walk = fake_par(T, rng)
    segs = nuisance.segments(n=T)
    pf, info = nuisance.filter_motion(par, TR, cfg["nuisance"], segs)
    pk = info["respiratory_peak"]
    assert info["applied"] and abs(pk["freq_hz"] - 0.35) <= pk["frequency_resolution_hz"]
    from scipy import signal
    f, p0 = signal.welch(par[:, 3], fs=1 / TR, nperseg=256)
    _, p1 = signal.welch(pf[:, 3], fs=1 / TR, nperseg=256)
    i = np.argmin(np.abs(f - 0.35))
    assert p1[i] < 0.02 * p0[i]                                     # breathing peak removed (> 98 % of its power)
    slow = (f < 0.1)
    assert np.allclose(p1[slow], p0[slow], rtol=0.05)               # slow motion kept
    fd0 = nuisance.framewise_displacement(par, segs)
    fd1 = nuisance.framewise_displacement(pf, segs)
    assert fd1.mean() < 0.5 * fd0.mean()                            # pseudo-motion no longer counts as FD


def test_no_band_stop_without_a_breathing_peak(cfg):
    rng = np.random.default_rng(SEED + 2)
    par = np.cumsum(rng.normal(0, 0.01, (800, 6)), axis=0)          # smooth random walk only, no peak
    pf, info = nuisance.filter_motion(par, TR, cfg["nuisance"], nuisance.segments(n=800))
    assert not info["applied"] and "no prominent" in info["reason"]
    assert np.array_equal(pf, par)


def test_projector_properties_and_df_factors(cfg):
    rng = np.random.default_rng(SEED + 3)
    T = 400
    par, _ = fake_par(T, rng)
    vols = np.r_[np.arange(200), np.arange(210, 410)]
    P, rec = nuisance.build(par, TR, cfg, volumes=vols)
    assert rec["info"]["n_segments"] == 2 and P.rank == 24 + 2      # 24 regressors + one constant per segment
    X = np.column_stack([np.ones(T), rec["X"]])
    assert np.abs(P.apply(X.T)).max() < 1e-9                        # regressors are removed exactly
    a = rng.normal(size=(5, T))
    assert np.allclose(P.apply(P.apply(a)), P.apply(a))             # idempotent
    assert P.var_factor() == pytest.approx((T - 26) / T)
    n = rng.normal(size=(20000, T))                                 # white noise: factors match Monte Carlo
    pn = P.apply(n)
    assert np.mean(pn ** 2) == pytest.approx(P.var_factor(), rel=0.01)
    assert np.mean(np.diff(pn, axis=1) ** 2) / 2 == pytest.approx(P.diff_factor(), rel=0.01)
    h = 1 - P.kept_per_frame()
    assert h.sum() == pytest.approx(P.rank)
    assert np.abs(np.mean(pn ** 2, axis=0) - (1 - h)).max() < 0.05      # per frame: 20000 samples, ~5 SD
    nan_row = np.full((1, T), np.nan)
    assert np.isnan(P.apply(nan_row)).all() and np.isfinite(P.apply(a)).all()


def test_projecting_echoes_equals_projecting_channels(cfg):
    """CLAUDE.md Phase 2b: identical regressors commute with the per-voxel echo mixing."""
    rng = np.random.default_rng(SEED + 4)
    T, V, E = 300, 200, 4
    te = np.asarray(cfg["acquisition"]["te_ms"])
    P, _ = nuisance.build(fake_par(T, rng)[0], TR, cfg)
    y = rng.normal(0, 0.01, (V, E, T))
    m = 1000 * np.exp(-te / rng.uniform(20, 70, (V, 1))) / np.asarray(cfg["frontend"]["echo_noise_ratios"]) / 30.0
    use = np.ones((V, E), bool)
    use[::5, 3] = False                                             # mixed 3- and 4-echo voxels
    a = frontend.project(P.apply(y), m, use, te)                    # project echoes, then form channels
    b = frontend.project(y, m, use, te)                             # form channels, then project
    for k in ("c", "r"):
        ok = np.isfinite(a[k])
        assert np.array_equal(ok, np.isfinite(b[k]))
        ref = P.apply(b[k].astype(np.float64))[ok]
        assert np.abs(a[k][ok] - ref).max() < 1e-5 * np.abs(ref).max()   # equal to float32 precision


@pytest.mark.slow
def test_projection_removes_a_motion_paced_violation(cfg, record_property):
    """A slow violation that follows a regressor inflates the non-differenced sigma (2.06 x true in the
    Phase 2 test); with the projector it is removed, both sigma estimators recover the true sigma, the
    whitened noise is white with the expected kept variance, and pure dR2* signal still gives c1 = 0."""
    over = {"grid": [30, 32, 18], "n_volumes": 600, "slow_violation": {"enabled": True, "amplitude": 0.03},
            "spikes": {"n": 0}, "skull_base": {"enabled": False}, "respiration": {"enabled": False},
            "sources": [{"name": "r2", "kind": "r2", "extent": "diffuse", "q": 0.5}]}
    c = io.merge(cfg, {"simulate": {"echo_sigma_scale": cfg["frontend"]["echo_noise_ratios"]}})
    sim = simulate.simulate(c, seed=SEED * 100 + 40, overrides=over)
    record_property("simulation", sim.describe())
    T = sim.n_volumes
    rng = np.random.default_rng(SEED + 5)
    d = np.asarray(sim.meta["slow_violation_timecourse"])
    X = np.column_stack([d, rng.normal(size=(T, 23)).cumsum(0)])    # the violation's driver + 23 unrelated regressors
    P = nuisance.Projector(X, nuisance.segments(n=T), T)
    S = sim.data[sim.mask]
    f0 = frontend.run(S, sim.te_ms, c, sim.mask)
    f1 = frontend.run(S, sim.te_ms, c, sim.mask, nuisance=P)
    true = sim.sigma[sim.mask]
    ok = ~f1["sigma_low_snr"] & np.isfinite(f1["sigma_plain"]) & ~f0["sigma_low_snr"]
    med = lambda a: float(np.median(a[ok] / true[ok]))
    record_property("plain_noreg_reg", [med(f0["sigma_plain"]), med(f1["sigma_plain"])])
    record_property("diff_noreg_reg", [med(f0["sigma_diff"]), med(f1["sigma_diff"])])
    assert med(f0["sigma_plain"]) > 1.10                            # violation inflates it without regression
    assert med(f1["sigma_plain"]) == pytest.approx(1.0, abs=0.03)
    assert med(f1["sigma_diff"]) == pytest.approx(1.0, abs=0.03)
    # whitened channels + residual: covariance = kept variance x I in high-SNR 4-echo voxels
    snr_min = (sim.mu_base[sim.mask] / sim.sigma_echo[sim.mask]).min(1)
    sel = (f1["n_use"] == 4) & (snr_min >= 10)
    # signal-free part: the residual (c carries the dR2* source)
    z = f1["r"][sel].astype(np.float64)
    Cr = np.einsum("vit,vjt->ij", z, z) / (z.shape[0] * T)
    record_property("residual_cov", Cr.round(4).tolist())
    assert np.abs(Cr - P.var_factor() * np.eye(2)).max() < 0.05
    c1 = f1["c"][sel, 1].astype(np.float64)                         # pure dR2*: c1 is noise only
    assert np.mean(c1 ** 2) == pytest.approx(P.var_factor(), rel=0.05)
    # the violation lives in the residual: more residual energy is removed than thermal noise alone loses
    record_property("energy_kept", f1["nuisance_energy_kept"])
    assert f1["nuisance_energy_kept"]["r0"] < P.var_factor() - 0.05


def test_run_subject_with_and_without_regression(cfg, tmp_path, record_property):
    """Real-run layout: the projector is built from the run's mcflirt.par (here with breathing
    pseudo-motion), recorded in the summary, its regressors saved; --no-nuisance keeps the Phase 2 path."""
    import json
    from test_frontend import make_run_folder
    c = io.merge(cfg, {"simulate": {"echo_sigma_scale": cfg["frontend"]["echo_noise_ratios"]},
                       "frontend": {"mppca": {"enabled": False}}})
    sim = simulate.simulate(c, seed=SEED * 100 + 41, overrides={"grid": [20, 20, 12], "n_volumes": 240})
    rd = make_run_folder(sim, tmp_path)
    par, _ = fake_par(sim.n_volumes, np.random.default_rng(SEED + 6))
    np.savetxt(rd / "motion" / "mcflirt.par", par)
    s1 = frontend.run_subject(rd, c, tmp_path / "reg", source="raw")
    s0 = frontend.run_subject(rd, c, tmp_path / "noreg", source="raw", nuisance=False)
    n = s1["nuisance"]
    record_property("nuisance", {k: n[k] for k in ("filter", "rank", "var_factor", "diff_factor", "energy_kept")})
    assert n["enabled"] and n["rank"] == 25 and n["filter"]["applied"]
    assert abs(n["filter"]["respiratory_peak"]["freq_hz"] - 0.35) < 0.02
    assert n["fd_mean_filtered"] < n["fd_mean_unfiltered"]
    assert (tmp_path / "reg" / "nuisance_regressors.tsv").exists()
    assert "Nuisance projection" in (tmp_path / "reg" / "summary.md").read_text()
    assert not s0["nuisance"]["enabled"] and not (tmp_path / "noreg" / "nuisance_regressors.tsv").exists()
    # sigma is corrected for the removed df: both variants agree (no motion-paced violation simulated)
    assert s1["sigma_median"] == pytest.approx(s0["sigma_median"], rel=0.03)
    fr = np.genfromtxt(tmp_path / "reg" / "frames.tsv", delimiter="\t", names=True)
    assert np.isfinite(fr["fd_filtered_mm"]).all() and np.isfinite(fr["residual_energy_no_regression"]).all()
    done = json.loads((tmp_path / "reg" / "done.json").read_text())
    assert "config_sha256" in done
