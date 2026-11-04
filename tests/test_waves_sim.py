"""W1 (WAVES_BRIEF 8.1): the simulator's traveling waves, standing oscillation, vascular transit, slice
timing and slice leakage, checked against their ground truth on noise-free data. Gate: < 2 min on CPU."""
import os

import numpy as np
import pytest

from mejoint import io, simulate

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))
QUIET = {"sources": [], "respiration": {"enabled": False}, "spikes": {"n": 0}, "skull_base": {"enabled": False},
         "snr0": None, "n_volumes": 400}


@pytest.fixture
def cfg():
    return io.load_config()


def sim_with(cfg, waves, grid=(24, 24, 24), **over):
    return simulate.simulate(cfg, seed=SEED * 100 + 1, overrides={**QUIET, "grid": list(grid), "waves": waves, **over})


def logsig(sim, echo=0):
    """Demeaned log signal of one echo, (X, Y, Z, T)."""
    y = np.log(np.maximum(sim.data[..., echo, :].astype(np.float64), 1e-12))
    return y - y.mean(-1, keepdims=True)


def delay(y1, y2, tr, band=(0.01, 0.10)):
    """Delay of y1 relative to y2 (s, positive = y1 later) from the cross-spectrum phase slope in the band."""
    f = np.fft.rfftfreq(y1.size, tr)
    X = np.fft.rfft(y1) * np.conj(np.fft.rfft(y2))
    b = (f >= band[0]) & (f <= band[1]) & (np.abs(X) > 1e-12 * np.abs(X).max())
    w = np.abs(X[b])
    ph = np.unwrap(np.angle(X[b]))
    slope = np.sum(w * f[b] * ph) / np.sum(w * f[b] ** 2)              # phase = -2 pi f tau, through the origin
    return -slope / (2 * np.pi)


def test_planar_neural_wave_delays_follow_distance_over_speed(cfg):
    v = 6.0
    sim = sim_with(cfg, [{"name": "n", "kind": "neural", "geometry": "planar", "speed_mm_s": v, "direction": [1, 0, 0],
                          "q": 0.5}])
    y = logsig(sim)
    m = sim.mask
    z, yc = 12, 12
    xs = [x for x in range(24) if m[x, yc, z]]
    x0, x1 = xs[2], xs[-3]
    got = delay(y[x1, yc, z], y[x0, yc, z], sim.tr_s)
    want = (x1 - x0) * sim.voxel_mm[0] / v
    assert got == pytest.approx(want, abs=0.02)
    w = sim.waves[0]
    assert w["delay_s"][x1, yc, z] - w["delay_s"][x0, yc, z] == pytest.approx(want)
    # echo independence: a pure dR2* wave has the same timing at every echo
    assert delay(logsig(sim, 3)[x1, yc, z], logsig(sim, 0)[x1, yc, z], sim.tr_s) == pytest.approx(0, abs=1e-6)


def test_radial_wave_delays_grow_with_distance_from_the_centre(cfg):
    v = 10.0
    sim = sim_with(cfg, [{"name": "r", "kind": "neural", "geometry": "radial", "speed_mm_s": v,
                          "center_vox": [11.5, 11.5, 11.5], "q": 0.5}])
    y = logsig(sim)
    a, b = (12, 12, 12), (19, 12, 12)
    w = sim.waves[0]
    want = w["delay_s"][b] - w["delay_s"][a]
    dist = np.linalg.norm((np.array(b) - 11.5) * sim.voxel_mm) - np.linalg.norm((np.array(a) - 11.5) * sim.voxel_mm)
    assert want == pytest.approx(dist / v)                              # delay map = distance from the centre / speed
    assert delay(y[b], y[a], sim.tr_s) == pytest.approx(want, abs=0.02)


def test_standing_wave_has_no_lag_without_slice_timing(cfg):
    sim = sim_with(cfg, [{"name": "s", "kind": "standing", "q": 0.5}])
    y = logsig(sim)
    assert abs(delay(y[12, 12, 18], y[12, 12, 6], sim.tr_s)) < 1e-6
    assert abs(delay(y[18, 12, 12], y[6, 12, 12], sim.tr_s)) < 1e-6


def test_slice_timing_makes_later_slices_lead(cfg):
    """A slice acquired delta seconds later samples the oscillation delta later, so it appears to LEAD by
    delta: the artifact the individual stage must correct (and the sign it must use)."""
    sim = sim_with(cfg, [{"name": "s", "kind": "standing", "q": 0.5}], slice_timing={"enabled": True, "mb_factor": 6})
    off = sim.slice_offsets_s
    assert off.size == 24 and np.unique(off).size == 4                 # 24 slices / MB 6 = 4 groups
    assert np.allclose(off[:4], off[4:8])                               # z and z + 4 acquired together
    y = logsig(sim)
    za, zb = int(np.argmin(off[4:20])) + 4, int(np.argmax(off[4:20])) + 4
    got = delay(y[12, 12, zb], y[12, 12, za], sim.tr_s)
    assert got == pytest.approx(-(off[zb] - off[za]), abs=1e-3)
    assert off[zb] - off[za] > 0


def test_vascular_transit_has_te_dependent_lag(cfg):
    """In the vessel, fast dS0 inflow and dR2* delayed by the hemodynamic delay share voxels: early echoes
    follow the S0 part, late echoes the delayed R2* part (WAVES_BRIEF 4.7 item 7)."""
    sim = sim_with(cfg, [{"name": "v", "kind": "vascular", "speed_mm_s": 60.0, "s": 0.02, "q": 0.5,
                          "tissue_delay_s": 2.0, "radius_mm": 4.0}])
    w = sim.waves[0]
    tube = np.argwhere(w["tube"])
    vx = tuple(tube[len(tube) // 2])
    # the echoes disagree in timing in the vessel (not just in sign or scale); in tissue they agree exactly
    r_tube = np.corrcoef(logsig(sim, 3)[vx], logsig(sim, 0)[vx])[0, 1]
    tv0 = tuple(np.argwhere(w["tissue"])[0])
    r_tissue = np.corrcoef(logsig(sim, 3)[tv0], logsig(sim, 0)[tv0])[0, 1]
    assert abs(r_tube) < 0.9 and r_tissue == pytest.approx(1.0, abs=1e-9)
    # separate the two parts exactly (noise-free log signal is linear in a and b)
    te = sim.te_ms
    Y = np.stack([logsig(sim, e)[vx] for e in range(4)])
    A = np.column_stack([np.ones(4), -te])
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    assert delay(b, a, sim.tr_s) == pytest.approx(2.0, abs=0.05)        # dR2* part 2 s after the dS0 part
    tissue = np.argwhere(w["tissue"])
    tv = tuple(tissue[0])
    Yt = np.stack([logsig(sim, e)[tv] for e in range(4)])
    at, bt = np.linalg.lstsq(A, Yt, rcond=None)[0]
    # tissue: dR2* only (float32 data: rounding ~1e-7 against a log-signal change of ~1e-2)
    assert np.abs(at).max() < 1e-4 * np.abs(bt * te.max()).max()


def test_slice_leakage_couples_slices_acquired_together(cfg):
    wv = [{"name": "n", "kind": "neural", "geometry": "planar", "speed_mm_s": 3.0, "direction": [0, 0, 1], "q": 0.5}]
    s0 = sim_with(cfg, wv, slice_timing={"enabled": True})
    s1 = sim_with(cfg, wv, slice_timing={"enabled": True}, slice_leakage={"enabled": True, "fraction": 0.2})
    c = lambda s: np.corrcoef(logsig(s)[12, 12, 6], logsig(s)[12, 12, 10])[0, 1]   # 4 slices apart = one group
    assert c(s1) > c(s0) + 0.02


def test_invalid_wave_kind_and_empty_band_are_rejected(cfg):
    with pytest.raises(ValueError, match="kind"):
        sim_with(cfg, [{"name": "x", "kind": "spiral"}])
    with pytest.raises(ValueError, match="band"):
        sim_with(cfg, [{"name": "x", "kind": "standing", "band_hz": [0.3, 0.301]}], n_volumes=100)


def test_waves_with_noise_and_sources_run(cfg):
    """A realistic mixture (noise, sources, respiration, slice timing) runs and keeps its ground truth."""
    sim = simulate.simulate(cfg, seed=SEED, overrides={"grid": [24, 24, 24], "n_volumes": 300,
                                                         "waves": [{"name": "n", "kind": "neural", "speed_mm_s": 8.0}],
                                                         "slice_timing": {"enabled": True}})
    assert np.isfinite(sim.data).all() and len(sim.waves) == 1 and sim.waves[0]["delay_s"].max() > 0
