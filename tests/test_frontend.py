"""Phase 2: front end tests (a)-(e) of CLAUDE.md section 8, plus the section 5.3 properties.

Every test states the SNR of its simulated data. MEJOINT_TEST_SEED reruns everything under another seed.
"""
import os

import numpy as np
import pytest

from mejoint import frontend, io, simulate

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))
G = {"grid": [24, 24, 14], "n_volumes": 400}
NO_VIOL = {"spikes": {"n": 0}, "skull_base": {"enabled": False}}
CLEAN = {"sources": [], "respiration": {"enabled": False}, **NO_VIOL}
R2_ONLY = [{"name": "r2a", "kind": "r2", "extent": "focal", "q": 0.5, "radius_vox": 1.5},
           {"name": "r2b", "kind": "r2", "extent": "diffuse", "q": 0.25}]


@pytest.fixture
def cfg():
    return io.load_config()


def sim_with(cfg, offset=0, **over):
    return simulate.simulate(cfg, seed=SEED * 100 + offset, overrides=io.merge(G, over))


def note(sim, record_property, extra=""):
    record_property("simulation", sim.describe() + extra)
    print(sim.describe() + extra)


def run_fe(sim, cfg, **kw):
    return frontend.run(sim.data[sim.mask], sim.te_ms, cfg, sim.mask, **kw)


# ----------------------------------------------------------------------------- section 5.3 construction
def test_sign_convention_patterns_and_undefined_channels(cfg):
    rng = np.random.default_rng(SEED)
    te = np.array(cfg["acquisition"]["te_ms"])
    V, T = 40, 30
    use = np.ones((V, 4), bool)
    use[10:20, 3] = False                     # 3 echoes
    use[20:30, 2:] = False                    # 2 echoes
    use[30:, 1:] = False                      # 1 echo
    mu = 500 * np.exp(-te / 40)[None] * rng.uniform(0.8, 1.2, (V, 1))
    y = rng.standard_normal((V, 4, T))
    pr = frontend.project(y, mu, use, te)
    R = pr["R"]
    assert np.all(R[:, 0, 0] > 0)
    assert np.all(R[:30, 1, 1] > 0) and np.all(R[:30, 0, 1] < 0)        # R01 < 0 after the convention
    assert np.isnan(pr["c"][30:, 1]).all() and np.isfinite(pr["c"][:30]).all()   # c1 undefined, not zero
    assert np.isfinite(pr["r"][:10]).all()                               # 4 echoes: 2 residual channels
    assert np.isfinite(pr["r"][10:20, 0]).all() and np.isnan(pr["r"][10:20, 1]).all()
    assert np.isnan(pr["r"][20:]).all()                                  # <= 2 echoes: no residual
    # 1 usable echo: c0 is that echo whitened, with negative sign (the design column is -TE, R00 > 0)
    assert np.allclose(pr["c"][30:, 0], -mu[30:, :1] * y[30:, 0], rtol=1e-5)


def test_backsubstitution_recovers_a_and_b_exactly(cfg, record_property):
    """Property 4: s = m1 / R11, q = (m0 - R01 s) / R00 for pure, mixed and opposite-sign sources."""
    sim = sim_with(cfg, snr0=None, **NO_VIOL)
    note(sim, record_property, " (noise-free)")
    m = sim.mask
    Sd = sim.data[m].astype(np.float64)
    mu = Sd.mean(2)
    pr = frontend.project(frontend.signal_form(Sd, mu), mu, np.ones_like(mu, bool), sim.te_ms)
    R, c = pr["R"], pr["c"].astype(np.float64)
    s = c[:, 1] / R[:, 1, 1][:, None]
    q = (c[:, 0] - R[:, 0, 1][:, None] * s) / R[:, 0, 0][:, None]        # in ms^-1 (TE in ms)
    a, b = sim.truth_ab()
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    assert np.abs(s - a).max() < 1e-5
    assert np.abs(1000 * q - b).max() < 1e-3                             # s^-1; float32 storage limit


def test_monoexponential_fit_is_exact_without_noise(cfg):
    sim = sim_with(cfg, snr0=None, **CLEAN)
    m = sim.mask
    s0, t2, _ = frontend.fit_monoexp(sim.mu_base[m], sim.te_ms, np.ones((m.sum(), 4), bool))
    assert np.allclose(t2, sim.t2s_ms[m], rtol=1e-6)
    assert np.allclose(s0, sim.s0[m], rtol=1e-6)


def test_sigma_statistics_constants():
    assert frontend.chi2_median(2) == pytest.approx(2 * np.log(2))
    assert frontend.chi2_median(1) == pytest.approx(0.4549, abs=1e-4)
    assert frontend.sigma_rel_se(2, 800) == pytest.approx(0.0255, abs=5e-4)    # ~2.5 % per voxel
    assert frontend.sigma_rel_se(1, 800) == pytest.approx(0.0411, abs=5e-4)


# ----------------------------------------------------------------------------- tests (a)-(e)
def test_a_whitened_noise_covariance_is_identity(cfg, record_property):
    """(a) channel and residual noise covariance within +/-5 % of I where mu_min / sigma >= 5."""
    sim = sim_with(cfg, 1, grid=[28, 28, 16], snr0=60.0, **CLEAN)
    note(sim, record_property)
    fe = run_fe(sim, cfg)
    snr_min = sim.mu_base[sim.mask].min(1) / sim.sigma[sim.mask]
    Z = np.concatenate([fe["c"], fe["r"]], axis=1).astype(np.float64)
    full = fe["n_use"] == 4

    def cov(sel):
        z = Z[sel] - Z[sel].mean(2, keepdims=True)
        return np.einsum("vit,vjt->ij", z, z) / (z.shape[0] * (z.shape[2] - 1))

    C = cov(full & (snr_min >= 5))
    record_property("cov_mu_min_ge_5", np.round(C, 3).tolist())
    assert np.abs(C - np.eye(4)).max() <= 0.05
    for lo, hi in [(5, 10), (10, np.inf)]:                                # reported per band
        sel = full & (snr_min >= lo) & (snr_min < hi)
        if sel.sum() >= 50:
            Cb = cov(sel)
            record_property(f"cov_band_{lo}_{hi}", np.round(np.diag(Cb), 3).tolist())
            assert np.abs(Cb - np.eye(4)).max() <= 0.05


def test_b_c1_is_zero_for_pure_r2star(cfg, record_property):
    """(b) a pure dR2* signal leaves c1 (and r) at zero; with noise c1 carries no trace of it."""
    clean = sim_with(cfg, 2, snr0=None, sources=R2_ONLY, **NO_VIOL)
    Sd = clean.data[clean.mask]
    mu = Sd.mean(2)
    pr = frontend.project(frontend.signal_form(Sd, mu), mu, np.ones_like(mu, bool), clean.te_ms)
    scale = np.abs(pr["c"][:, 0]).max()
    assert np.abs(pr["c"][:, 1]).max() / scale < 1e-4                    # float32 storage limit
    assert np.nanmax(np.abs(pr["r"])) / scale < 1e-4

    sim = sim_with(cfg, 2, sources=R2_ONLY, **NO_VIOL)
    note(sim, record_property)
    fe = run_fe(sim, cfg)
    _, b = sim.truth_ab()
    strong = b.std(1) > 0.2
    r0 = np.array([np.corrcoef(fe["c"][v, 0], b[v])[0, 1] for v in np.flatnonzero(strong)])
    r1 = np.array([np.corrcoef(fe["c"][v, 1], b[v])[0, 1] for v in np.flatnonzero(strong)])
    null_sd = 1 / np.sqrt(sim.n_volumes)
    assert np.median(np.abs(r0)) > 0.5                                   # the signal is in c0 ...
    assert abs(r1.mean()) < 3 * null_sd                                  # ... and not in c1
    assert np.percentile(np.abs(r1), 95) < 2.5 * null_sd


def test_c_c0_is_negative_optimal_combination(cfg, record_property):
    """(c) c0 = -k_v x OC deviation (tedana weights TE exp(-TE/T2*)), linear form, equal sigma."""
    sim = sim_with(cfg, 3, **NO_VIOL)
    note(sim, record_property)
    Sd = sim.data[sim.mask].astype(np.float64)
    fe = run_fe(sim, cfg, form="linear")
    full = fe["n_use"] == 4
    w = sim.te_ms[None] * np.exp(-sim.te_ms[None] / fe["t2s_ms"][:, None])
    oc = (w[:, :, None] * Sd).sum(1) / w.sum(1, keepdims=True)
    ocd = oc - oc.mean(1, keepdims=True)
    # exact when mu is mono-exponential with the same T2*
    mu_fit = fe["muhat"]
    pr = frontend.project(Sd / mu_fit[:, :, None] - 1, mu_fit / fe["sigma"][:, None], np.ones_like(mu_fit, bool), sim.te_ms)
    cc = np.array([np.corrcoef(pr["c"][v, 0], ocd[v])[0, 1] for v in np.flatnonzero(full)])
    assert cc.max() < -1 + 1e-9
    # and to within noise with the front end's own (empirical) means
    cc2 = np.array([np.corrcoef(fe["c"][v, 0], ocd[v])[0, 1] for v in np.flatnonzero(full)])
    assert cc2.max() < -0.9999


def test_d_sigma_recovery(cfg, record_property):
    """(d) sigma recovered with brain-median error < 2 %, and 95 % of voxels within 3 SE.
    Spatially varying (g-factor-like) noise, all sources and spikes on; skull-base voxels (where
    curvature was injected on purpose) are excluded from the check."""
    sim = sim_with(cfg, 4, gfactor={"enabled": True})
    note(sim, record_property)
    fe = run_fe(sim, cfg)
    ok = ~sim.regions["skull_base"][sim.mask]
    true = sim.sigma[sim.mask]
    err = fe["sigma"][ok] / true[ok] - 1
    record_property("sigma_median_rel_err", float(np.median(err)))
    assert abs(np.median(err)) < 0.02
    vox = ok & ~fe["sigma_low_snr"]
    se = frontend.sigma_rel_se(fe["df"][vox], sim.n_volumes) * true[vox]
    within = np.mean(np.abs(fe["sigma"][vox] - true[vox]) < 3 * se)
    record_property("fraction_within_3SE", float(within))
    assert within >= 0.95
    assert abs(np.median(fe["sigma_smooth"][ok] / true[ok] - 1)) < 0.02


def test_e_residual_energy_rises_where_violations_were_injected(cfg, record_property):
    """(e) residual energy rises with injected curvature, spikes and noise-floor violations."""
    sim = sim_with(cfg, 5, spikes={"n": 3, "echo_sd": 0.05})
    note(sim, record_property)
    fe = run_fe(sim, cfg)
    m = sim.mask
    sb = sim.regions["skull_base"][m] & (sim.dephasing_hz[m] > 3) & (fe["df"] > 0)
    inner = sim.regions["interior"][m] & ~sim.regions["skull_base"][m] & ~sim.regions["spike"][m]
    # dynamic curvature (respiration-modulated dephasing) raises residual energy at the skull base
    assert np.nanmedian(fe["resid_excess"][sb]) > np.nanmedian(fe["resid_excess"][inner]) + 0.05
    assert np.nanmedian(fe["resid_excess"][inner]) == pytest.approx(1.0, abs=0.03)
    # static curvature shows as a misfit of the mono-exponential decay
    assert np.median(fe["static_misfit"][sb]) > 10 * np.median(fe["static_misfit"][inner])
    # spikes: frames with injected per-echo jumps carry extra residual energy
    others = np.delete(fe["frame_energy"], sim.spike_frames)
    assert np.median(others) == pytest.approx(1.0, abs=0.03)
    assert fe["frame_energy"][sim.spike_frames].max() > np.percentile(others, 99.5)
    # noise floor: including echoes near the floor raises residual energy; snr_min removes most of it
    # T2* 20-35 ms at SNR0 50 puts echo 4 near the floor (mu/sigma ~2-9) brain-wide, for any seed
    low = sim_with(cfg, 6, snr0=50.0, t2s_ms_range=[20.0, 35.0],
                   sources=[{"name": "r2d", "kind": "r2", "extent": "diffuse", "q": 1.0}],
                   respiration={"enabled": True, "q": 1.0}, **NO_VIOL)
    note(low, record_property, " (floor case)")
    snr4 = low.mu_base[low.mask][:, 3] / low.sigma[low.mask]
    keep = frontend.run(low.data[low.mask], low.te_ms, cfg, low.mask)
    incl = frontend.run(low.data[low.mask], low.te_ms, io.merge(cfg, {"frontend": {"snr_min": 0.0}}), low.mask)
    sel = (snr4 < 4) & (keep["df"] > 0) & (incl["df"] > 0)
    assert sel.sum() >= 50
    assert np.nanmedian(incl["resid_excess"][sel]) > 1.08
    assert np.nanmedian(incl["resid_excess"][sel]) > np.nanmedian(keep["resid_excess"][sel]) + 0.05


def test_echo_usability_and_nonpositive_frames(cfg, record_property):
    sim = sim_with(cfg, 7, snr0=40.0, **CLEAN)
    note(sim, record_property)
    S = sim.data[sim.mask].copy()
    S[:5, 3, 10] = 0.0                                                   # spline-like undershoot
    fe = frontend.run(S, sim.te_ms, cfg, sim.mask)
    assert not fe["use"][:5, 3].any() and fe["nonpositive"][:5, 3].all()
    snr4 = sim.mu_base[sim.mask][:, 3] / sim.sigma[sim.mask]
    excluded = ~fe["use"][5:, 3]
    assert np.all(snr4[5:][excluded] < 7) and np.all(snr4[5:][~excluded] > 3.5)   # cut near snr_min = 5
    assert np.array_equal(fe["df"], np.clip(fe["use"].sum(1) - 2, 0, None))
    assert np.isfinite(fe["c"]).all()                                    # >= 2 echoes everywhere here


def test_sigma_bootstrap_when_echo4_is_weak_everywhere(cfg, record_property):
    """No voxel has echo-4 mu/sigma > 10: sigma_init falls back to echoes 1-3 and still recovers sigma."""
    sim = sim_with(cfg, 8, snr0=50.0, t2s_ms_range=[20.0, 32.0], **CLEAN)     # echo-4 mu/sigma < ~8.5
    note(sim, record_property)
    assert (sim.mu_base[sim.mask][:, 3] / sim.sigma[sim.mask]).max() < 10
    fe = run_fe(sim, cfg)
    assert fe["trusted"].sum() >= 50
    assert abs(np.median(fe["sigma"] / sim.sigma[sim.mask] - 1)) < 0.02
    with pytest.raises(ValueError, match="sigma_init_snr"):
        too_low = sim_with(cfg, 9, snr0=8.0, **CLEAN)
        run_fe(too_low, cfg)


# ----------------------------------------------------------------------------- real-data runner (synthetic run folder)
def make_run_folder(sim, root, run_id="sub-90001_acq-REDACTED_ser-5"):
    """A fake mbme_tedana run folder around simulated echoes, laid out like the real ones."""
    import json

    import nibabel as nib

    paths = sim.to_nifti(root / "nifti")
    rd = root / "OUT" / run_id
    (rd / "motion").mkdir(parents=True)
    T = sim.n_volumes
    (rd / "volumes_used.tsv").write_text("echo1\techo2\techo3\techo4\n" + "".join(f"{i}\t{i}\t{i}\t{i}\n" for i in range(T)))
    np.savetxt(rd / "motion" / "framewise_displacement.txt", np.abs(np.random.default_rng(0).normal(0, 0.1, T)))
    par = np.cumsum(np.random.default_rng(1).normal(0, 0.01, (T, 6)), axis=0) * np.array([0.01] * 3 + [1] * 3)
    np.savetxt(rd / "motion" / "mcflirt.par", par)
    nib.save(nib.Nifti1Image(sim.mask.astype(np.uint8), np.diag([*sim.voxel_mm, 1.0])), str(rd / "brain_mask.nii.gz"))
    (rd / "result.json").write_text(json.dumps({"status": "ok", "run_id": run_id, "input_echo1": str(paths["echoes"][0]),
                                                "echo_times_ms": sim.te_ms.tolist()}))
    (rd / "echo_info.json").write_text(json.dumps({"echo_files": [str(p) for p in paths["echoes"]]}))
    return rd


def test_run_subject_end_to_end(cfg, tmp_path, record_property):
    sim = sim_with(cfg, 10, grid=[20, 20, 12], n_volumes=60, gfactor={"enabled": True})
    note(sim, record_property)
    rd = make_run_folder(sim, tmp_path)
    c2 = io.merge(cfg, {"frontend": {"mppca": {"n_volumes": 60, "slice_step": 3}}})
    summ = frontend.run_subject(rd, c2, tmp_path / "fe", source="raw")
    out = tmp_path / "fe"
    for f in ("mu", "snr", "t2s_ms", "sigma", "sigma_smooth", "df", "resid_excess", "static_misfit", "mppca_sigma", "R"):
        assert (out / f"{f}.nii.gz").exists(), f
    for f in ("c0.npy", "c1.npy", "r0.npy", "r1.npy", "R.npy", "frames.tsv", "summary.md", "summary.json", "done.json"):
        assert (out / f).exists(), f
    import nibabel as nib
    sig = np.asarray(nib.load(str(out / "sigma.nii.gz")).dataobj)[sim.mask]
    assert abs(np.median(sig / sim.sigma[sim.mask]) - 1) < 0.03
    c0 = np.load(out / "c0.npy")
    assert c0.shape == (sim.mask.sum(), sim.n_volumes)
    assert summ["mppca_check"]["verdict"] == "OK"
    assert "Sigma cross-check" in (out / "summary.md").read_text()
    assert "frame_energy_vs_fd_spearman" in summ
    record_property("mppca_ratios", [r["ratio_residual_over_mppca"] for r in summ["mppca_check"]["per_echo"]])


def test_excluded_subject_is_refused(cfg, tmp_path):
    sim = sim_with(cfg, 11, grid=[16, 16, 10], n_volumes=20, **CLEAN)
    rd = make_run_folder(sim, tmp_path, run_id="sub-P08_acq-TIMESTAMP_ser-4")
    with pytest.raises(ValueError, match="excluded"):
        frontend.run_subject(rd, cfg, tmp_path / "fe")



def test_static_shape_separates_curvature_and_alternation(cfg):
    te = np.array(cfg["acquisition"]["te_ms"])
    base = np.log(1000) - te / 50.0
    use = np.ones((3, 4), bool)
    alt = (-1.0) ** np.arange(4)
    biexp = np.log(0.7 * np.exp(-te / 30) + 0.3 * np.exp(-te / 150)) + np.log(1000)
    mu = np.exp(np.stack([base, base + 0.02 * alt, biexp]))
    curv, alt_c = frontend.static_shape(mu, te, use)
    assert abs(curv[0]) < 1e-9 and abs(alt_c[0]) < 1e-9              # mono-exponential: nothing
    assert alt_c[1] > 0.005 and abs(alt_c[1]) > 5 * abs(curv[1])      # odd/even pattern lands in 'alternating'
    assert curv[2] > 0.005 and abs(curv[2]) > 3 * abs(alt_c[2])        # two-compartment decay is convex


def test_drift_quiet_window_and_mask_holes():
    import nibabel as nib
    T = 300
    par = np.zeros((T, 6))
    par[:100, 3] = np.linspace(2.0, 0.0, 100)                          # drifting start, still afterwards
    d = frontend.drift_mm(par)
    assert d[0] > d[150] and d[150] == 0
    assert frontend.quiet_window(par, 100) >= 100
    m = np.zeros((12, 12, 8), bool)
    m[2:10, 2:10, 1:7] = True
    m[5:7, 5:7, 3:5] = False                                           # an enclosed hole of 8 voxels
    h = frontend.mask_holes(m, np.eye(4))
    assert h["holes_3d"].sum() == 8 and h["holes_axial"].sum() == 8
    assert nib.aff2axcodes(np.eye(4)) == ("R", "A", "S")


@pytest.mark.parametrize("scale", [None, [1.0, 0.8, 0.65, 0.55]])
def test_background_noise_detects_per_echo_scaling(cfg, tmp_path, record_property, scale):
    """Air outside the head is pure noise: its level per echo must stay equal when the echoes share one
    thermal sigma, and must recover deliberately applied per-echo scale factors."""
    sim = sim_with(cfg, 12, grid=[30, 30, 14], n_volumes=60)
    note(sim, record_property, f" (per-echo scaling {scale})")
    if scale is not None:
        sim.data *= np.asarray(scale, np.float32)[None, None, None, :, None]
    rd = make_run_folder(sim, tmp_path)
    c2 = io.merge(cfg, {"frontend": {"mppca": {"enabled": False}}})
    summ = frontend.run_subject(rd, c2, tmp_path / "fe", source="raw")
    ratios = [b["sd_ratio_to_echo1"] for b in summ["background_noise"]]
    record_property("background_sd_ratios", ratios)
    expect = np.ones(4) if scale is None else np.asarray(scale)
    assert np.allclose(ratios, expect, atol=0.05)
    assert summ["background_noise"][0]["n_voxels"] >= 50
    assert "drift_mm_p50_p90_max" in summ and "static_curvature_p10_50_90" in summ
    md = (tmp_path / "fe" / "summary.md").read_text()
    assert "Noise in air outside the head" in md and "Shape of the static decay" in md
