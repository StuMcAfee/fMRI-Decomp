"""Phase 1: the simulator produces exactly the signal and noise it claims to.

Every test states the SNR of the data it uses (recorded with pytest's record_property and printed),
as CLAUDE.md section 8 requires.
"""
import json
import os

import nibabel as nib
import numpy as np
import pytest

from mejoint import inventory, io, simulate

SMALL = {"grid": [20, 20, 12], "n_volumes": 120}
SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))          # rerun the suite under other seeds with this
CLEAN = {"sources": [], "respiration": {"enabled": False}, "spikes": {"n": 0}, "skull_base": {"enabled": False}}


@pytest.fixture
def cfg():
    return io.load_config()


def sim_with(cfg, seed=None, **over):
    seed = SEED if seed is None else seed
    return simulate.simulate(cfg, seed=seed, overrides=io.merge(SMALL, over))


def note(sim, record_property):
    """State the SNR of the simulated data in the test report."""
    record_property("simulation", sim.describe())
    print(sim.describe())


def residual_energy(logratio, te_ms):
    """Energy of the per-voxel log-ratio outside span{1, TE} (unweighted): (V, T)."""
    D = np.column_stack([np.ones_like(te_ms), te_ms])
    P = np.eye(len(te_ms)) - D @ np.linalg.pinv(D)
    return np.einsum("ef,vft->vet", P, logratio).__pow__(2).sum(1)


# ----------------------------------------------------------------------------- reproducibility
def test_seeded_and_reproducible(cfg, record_property):
    a = sim_with(cfg, seed=7)
    b = sim_with(cfg, seed=7)
    c = sim_with(cfg, seed=8)
    note(a, record_property)
    assert np.array_equal(a.data, b.data)
    assert not np.array_equal(a.data, c.data)
    assert a.meta["seed"]["entropy"] == 7
    d = simulate.simulate(cfg, overrides=SMALL)                   # default: SeedSequence(project_seed, (1,))
    assert d.meta["seed"] == {"entropy": cfg["project_seed"], "spawn_key": [1]}


# ----------------------------------------------------------------------------- signal model
def test_noise_free_log_ratio_is_exactly_a_minus_te_b(cfg, record_property):
    sim = sim_with(cfg, snr0=None, spikes={"n": 0}, skull_base={"enabled": False})
    record_property("simulation", sim.describe() + " (noise-free)")
    m = sim.mask
    y = np.log(sim.data[m].astype(np.float64) / sim.mu_base[m][:, :, None])     # (V, E, T)
    a, b = sim.truth_ab()
    model = a[:, None, :] - (sim.te_ms / 1000.0)[None, :, None] * b[:, None, :]
    assert np.abs(y - model).max() < 1e-5                                       # float32 storage only
    # a pure-dR2* voxel: per-echo log changes scale exactly with TE
    src = next(s for s in sim.sources if s["name"] == "r2_focal")
    v = tuple(src["center"])
    others = [s for s in sim.sources if s["name"] != "r2_focal"]
    if all(abs(s["q_map"][v]) + abs(s["s_map"][v]) < 1e-9 for s in others):
        yv = np.log(sim.data[v].astype(np.float64) / sim.mu_base[v][:, None])
        ratio = yv[3] / yv[0]
        assert np.allclose(ratio[np.abs(yv[0]) > 1e-4], sim.te_ms[3] / sim.te_ms[0], rtol=1e-3)


def test_anatomy_t2star_and_skull_base(cfg, record_property):
    sim = sim_with(cfg)
    note(sim, record_property)
    t2 = sim.t2s_ms[sim.mask]
    assert t2.min() >= 20.0 - 1e-9 and t2.max() <= 70.0 + 1e-9
    assert np.percentile(t2, 99) > 60                                   # long T2* (children ~58 ms) present
    sb = sim.regions["skull_base"]
    assert sb.any() and np.median(sim.t2s_ms[sb]) < np.median(sim.t2s_ms[sim.regions["interior"]])
    assert sim.dephasing_hz[sb].max() == pytest.approx(cfg["simulate"]["skull_base"]["dephasing_hz_max"])
    assert sim.dephasing_hz[~sb].max() == 0


def test_sources_maps_and_timecourses(cfg, record_property):
    sim = sim_with(cfg, n_volumes=400)
    note(sim, record_property)
    by = {s["name"]: s for s in sim.sources}
    for s in sim.sources:
        assert s["timecourse"].mean() == pytest.approx(0, abs=1e-9)
        assert s["timecourse"].std() == pytest.approx(1, rel=1e-9)
    spec = {s["name"]: s for s in cfg["simulate"]["sources"]}
    assert np.abs(by["r2_focal"]["q_map"]).max() == pytest.approx(spec["r2_focal"]["q"])
    assert np.abs(by["r2_focal"]["s_map"]).max() == 0
    assert np.abs(by["s0_focal"]["s_map"]).max() == pytest.approx(spec["s0_focal"]["s"])
    assert np.abs(by["s0_focal"]["q_map"]).max() == 0
    v = tuple(by["mixed_opposite"]["center"])
    assert by["mixed_opposite"]["q_map"][v] < 0 < by["mixed_opposite"]["s_map"][v]
    v = tuple(by["mixed_same"]["center"])
    assert by["mixed_same"]["q_map"][v] > 0 and by["mixed_same"]["s_map"][v] > 0
    centers = [np.array(c) for c in sim.meta["focal_centers"].values()]
    assert min(np.linalg.norm(p - q) for i, p in enumerate(centers) for q in centers[i + 1:]) >= 4 * 1.5
    # neural-like time courses live in their band; respiration peaks in 0.25-0.5 Hz
    f = np.fft.rfftfreq(sim.n_volumes, sim.tr_s)
    lo, hi = cfg["simulate"]["band_hz"]
    pw = np.abs(np.fft.rfft(by["r2_focal"]["timecourse"])) ** 2
    assert pw[(f >= lo) & (f <= hi)].sum() / pw.sum() > 0.99
    pr = np.abs(np.fft.rfft(by["global_resp"]["timecourse"])) ** 2
    assert 0.25 <= f[np.argmax(pr)] <= 0.5


# ----------------------------------------------------------------------------- noise
@pytest.mark.parametrize("n_coils", [1, 4])
def test_noise_sd_and_background_floor(cfg, n_coils, record_property):
    sim = sim_with(cfg, n_volumes=200, n_coils=n_coils, **CLEAN)
    note(sim, record_property)
    sig = cfg["simulate"]["s0_mean"] / cfg["simulate"]["snr0"]
    # high SNR: temporal SD of the magnitude equals the channel sigma
    hi = sim.mask & (sim.mu_base[..., 0] / sim.sigma > 50)
    sd = sim.data[hi][:, 0].std(axis=1, ddof=1)
    assert np.median(sd) / sig == pytest.approx(1.0, abs=0.03)
    # background (no signal): mean magnitude = central chi mean, which grows with the number of coils
    bg = ~sim.mask
    expected = simulate.magnitude_mean(0.0, sig, n_coils)
    assert sim.data[bg].mean() / expected == pytest.approx(1.0, abs=0.01)
    if n_coils == 1:
        assert expected / sig == pytest.approx(np.sqrt(np.pi / 2))


def test_rician_floor_biases_late_echo_means(cfg, record_property):
    """At low SNR the magnitude mean exceeds the true signal exactly as the Rician formula says."""
    # short T2* (20-25 ms) and SNR0 25 put echo 4 at mu/sigma ~1.2-2.2 in nearly every voxel, for any seed
    sim = sim_with(cfg, n_volumes=600, snr0=25.0, t2s_ms_range=[20.0, 25.0], **CLEAN)
    note(sim, record_property)
    mu4, sg = sim.mu_base[..., 3][sim.mask], sim.sigma[sim.mask]
    meas = sim.data[sim.mask][:, 3].astype(np.float64).mean(1)
    low = mu4 / sg < 2
    assert low.sum() > 100
    pred = simulate.magnitude_mean(mu4[low], sg[low])
    assert np.median(meas[low] / pred) == pytest.approx(1.0, abs=0.02)        # formula matches
    assert np.median(meas[low] / mu4[low]) > 1.15                               # and the bias is real
    assert simulate.magnitude_mean(1.0, 1.0) == pytest.approx(1.549, abs=1e-3)  # CLAUDE.md 5.4: up to ~1.6x


def test_gfactor_noise_follows_smooth_map(cfg, record_property):
    sim = sim_with(cfg, n_volumes=300, gfactor={"enabled": True}, **CLEAN)
    note(sim, record_property)
    hi = sim.mask & (sim.mu_base[..., 0] / sim.sigma > 30)
    sd = sim.data[hi][:, 0].std(axis=1, ddof=1)
    assert np.median(sd / sim.sigma[hi]) == pytest.approx(1.0, abs=0.03)
    # noisier parts of the map really are noisier, by the predicted ratio
    q1, q3 = np.percentile(sim.sigma[hi], [25, 75])
    lo_, hi_ = sim.sigma[hi] <= q1, sim.sigma[hi] >= q3
    measured = np.median(sd[hi_]) / np.median(sd[lo_])
    predicted = np.median(sim.sigma[hi][hi_]) / np.median(sim.sigma[hi][lo_])
    assert predicted > 1.1 and measured == pytest.approx(predicted, rel=0.03)
    g = sim.sigma / (cfg["simulate"]["s0_mean"] / cfg["simulate"]["snr0"])
    assert g.min() == pytest.approx(1.0) and g.max() == pytest.approx(1.5)


# ----------------------------------------------------------------------------- model violations
def test_spikes_violate_the_model_only_where_injected(cfg, record_property):
    sim = sim_with(cfg, snr0=None, skull_base={"enabled": False}, spikes={"n": 3})
    record_property("simulation", sim.describe() + " (noise-free)")
    m = sim.mask
    y = np.log(sim.data[m].astype(np.float64) / sim.mu_base[m][:, :, None])
    res = residual_energy(y, sim.te_ms)                                        # (V, T)
    expected = np.zeros_like(res, bool)
    expected[np.ix_(sim.regions["spike"][m], sim.spike_frames)] = True
    assert res[~expected].max() < 1e-9
    assert res[expected].min() > 1e-6


def test_skull_base_curvature_static_and_dynamic(cfg, record_property):
    sim = sim_with(cfg, snr0=None, spikes={"n": 0})
    record_property("simulation", sim.describe() + " (noise-free)")
    m, sb = sim.mask, sim.regions["skull_base"]
    # static: a mono-exponential (log-linear) fit to the baseline decay misfits only at the skull base
    D = np.column_stack([np.ones(4), sim.te_ms])
    P = np.eye(4) - D @ np.linalg.pinv(D)
    stat = np.sqrt(((np.log(sim.mu_base[m]) @ P) ** 2).sum(1))
    sbm = sb[m]
    deph = sim.dephasing_hz[m] > 1.0
    assert stat[~sbm].max() < 1e-10
    assert stat[sbm & deph].min() > 1e-4
    # dynamic: respiration modulates the dephasing, so curvature also fluctuates in time there
    y = np.log(sim.data[m].astype(np.float64) / sim.mu_base[m][:, :, None])
    dyn = residual_energy(y, sim.te_ms).sum(1)
    assert dyn[~sbm].max() < 1e-9
    assert dyn[sbm & deph].min() > 1e-9


# ----------------------------------------------------------------------------- reporting and files
def test_snr_reported_and_ordered(cfg, record_property):
    sim = sim_with(cfg)
    note(sim, record_property)
    th = sim.meta["thermal_snr"]
    assert len(th) == 4 and all(a > b for a, b in zip(th, th[1:]))           # later echoes: lower SNR
    assert len(sim.meta["tsnr"]) == 4 and all(t > 0 for t in sim.meta["tsnr"])


def test_nifti_roundtrip_and_phase0_audit(cfg, tmp_path, record_property):
    sim = sim_with(cfg)
    note(sim, record_property)
    paths = sim.to_nifti(tmp_path / "sim")
    img = nib.load(str(paths["echoes"][0]))
    assert img.shape == (*SMALL["grid"], SMALL["n_volumes"])
    assert float(img.header.get_zooms()[3]) == pytest.approx(cfg["acquisition"]["tr_s"])
    assert np.allclose(np.asarray(img.dataobj), sim.echo(1))
    assert json.loads(paths["meta"].read_text())["seed"]["entropy"] == SEED
    inv = inventory.inventory_subject(tmp_path / "sim", cfg)                  # the Phase 0 audit reads it
    assert len(inv["echo_files"]) == 4 and inv["te_match"] is True


def test_invalid_settings_are_refused(cfg):
    with pytest.raises(ValueError, match="kind"):
        sim_with(cfg, sources=[{"name": "x", "kind": "t1", "extent": "focal"}])
    with pytest.raises(ValueError, match="sinc"):
        sim_with(cfg, skull_base={"dephasing_hz_max": 20.0})
