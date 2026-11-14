"""`mejoint te-check`: is a component's dS0 a TE-independent change, or a volume change of a compartment
with a different T2* (curvature)? Synthetic voxels through the real front end."""
import os

import numpy as np
from scipy import signal

from mejoint import frontend, io, te_check

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))
TE = np.array([16.20, 31.22, 46.24, 61.26])


def make(kind, rng, V=3000, T=400):
    """Voxels with tissue (T2* 40-50 ms) and a blood-like compartment (T2* 20 ms, 2.5x the tissue S0: fresh
    inflowing spins at TR 0.75 s). kind 'exchange': the compartment's volume follows the time course (5 % +- 2 %);
    'r2_pv': the tissue R2* follows it (+- 1 s^-1), compartment static at 5 %; 'r2': the same without the
    compartment (single compartment)."""
    b, a = signal.butter(3, 0.08)
    tc = signal.filtfilt(b, a, rng.standard_normal(T))
    tc = (tc - tc.mean()) / tc.std()
    f = (0.0 if kind == "r2" else 0.05) + (0.02 * tc if kind == "exchange" else 0.0 * tc)
    dr2 = (1.0 * tc if kind in ("r2", "r2_pv") else 0.0 * tc) / 1000.0  # per ms
    t2t = rng.uniform(40, 50, V)
    S0 = 1000.0
    tis = np.exp(-TE[None, :, None] * (1.0 / t2t[:, None, None] + dr2[None, None, :]))
    blood = 2.5 * np.exp(-TE[None, :, None] / 20.0)
    f = np.broadcast_to(np.asarray(f, float), (T,))
    S = S0 * ((1 - f)[None, None, :] * tis + f[None, None, :] * blood)
    ratios = np.array([1, 0.721, 0.631, 0.560])
    sigma = S0 * np.exp(-16.2 / 45.0) / 150.0
    S = S + sigma * ratios[None, :, None] * rng.standard_normal(S.shape)
    return S.astype(np.float32), tc


def run_check(kind):
    rng = np.random.default_rng(SEED + {"exchange": 0, "r2": 1, "r2_pv": 2}[kind])
    S, tc = make(kind, rng)
    cfg = io.merge(io.load_config(), {"frontend": {"mppca": {"enabled": False}}})
    side = int(np.ceil(S.shape[0] ** (1 / 3))) + 1
    mask3d = np.zeros((side, side, side), bool)
    mask3d.flat[: S.shape[0]] = True
    fe = frontend.run(S, TE, cfg, mask3d)
    M = tc[:, None]
    ch = {"c0": fe["c"][:, 0], "c1": fe["c"][:, 1], "r0": fe["r"][:, 0], "r1": fe["r"][:, 1]}
    m0 = te_check.ols_betas(ch["c0"], M)
    m1 = te_check.ols_betas(ch["c1"], M)
    return te_check.check(ch, fe["R"], fe["mu"], fe["sigma"], fe["echo_noise_ratios"], fe["use"], fe["t2s_ms"], TE,
                          M, m0, m1)


def test_volume_exchange_shows_curvature_opposed_sign_a_zero_crossing_and_its_t2():
    out = run_check("exchange")
    c = out["components"][0]
    assert out["R_rebuild_max_rel_error"] < 1e-4                       # the basis matches the front end's
    assert c["curvature_z"] > 100 and c["curvature_fraction"] > 0.01
    assert c["fraction_opposed"] > 0.9 and c["n_sig"] > 10 * max(c["null_n_sig"], 1)   # S0 and R2* rise together
    assert c["fraction_te1_te4_opposite"] > 0.9 and 16 < c["profile_zero_crossing_ms"] < 61
    assert 15 <= c["exchange_t2x_ms"] <= 25                             # the planted 20 ms compartment


def test_single_compartment_r2_has_no_curvature_and_one_sign_across_echoes():
    out = run_check("r2")
    c = out["components"][0]
    # 8 simulated datasets: z -1.2 to 6.8 (heavier-tailed than normal with thousands of voxels), fraction 0.0000;
    # the effect size is the measure, z only supports it
    assert abs(c["curvature_fraction"]) < 0.001 and abs(c["curvature_z"]) < 10
    assert c["fraction_te1_te4_opposite"] < 0.1 and c["profile_zero_crossing_ms"] is None
    assert c["n_sig"] <= 5 * max(c["null_n_sig"], 1) + 20               # no dS0 beyond noise
    prof = np.array(c["profile_pct"])
    assert np.all(prof < 0) and np.corrcoef(prof, -TE)[0, 1] > 0.999    # every echo falls, in proportion to TE


def test_r2_with_a_static_blood_compartment_curves_but_does_not_cross_zero():
    """Partial volume with a short-T2* compartment also bends the TE dependence (curvature is not specific to a
    volume change), but its apparent dS0 is small: no sign change between echoes."""
    out = run_check("r2_pv")
    c = out["components"][0]
    assert c["curvature_z"] > 8 and c["curvature_fraction"] < 0.002     # detectable, but ~500x below the exchange
    assert c["fraction_te1_te4_opposite"] < 0.1 and c["profile_zero_crossing_ms"] is None


def test_curvature_null_with_nuisance_projection(tmp_path):
    """Pure noise through the front end WITH the motion nuisance projection: the curvature z of every component
    stays near 0. Before the fix (2026-11-06) the calibration's simulated noise was not projected, the noise was
    under-estimated by ~p / (T - k - 1), and the median curvature z here was ~4."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_frontend import make_run_folder

    from mejoint import decompose, frontend, io, simulate
    cfg = io.load_config()
    c = io.merge(cfg, {"simulate": {"echo_sigma_scale": cfg["frontend"]["echo_noise_ratios"]},
                       "frontend": {"mppca": {"enabled": False}}, "nuisance": {"n_surrogates": 1},
                       "decompose": {"n_restarts": 4}, "metrics": {"n_null_pipeline": 5, "n_boot": 4, "band_mc": 10, "n_null": 10}})
    G = {"grid": [32, 32, 18], "n_volumes": 200, "spikes": {"n": 0}, "skull_base": {"enabled": False},
         "respiration": {"enabled": False}, "sources": []}
    sim = simulate.simulate(c, seed=12, overrides=G)
    rd = make_run_folder(sim, tmp_path)
    fe = tmp_path / "fe"
    frontend.run_subject(rd, c, fe, source="raw", nuisance=True)
    assert (fe / "nuisance_Q.npy").exists()
    decompose.run_dir(fe, c, fe / "k8", 8)
    out = te_check.check_dir(fe, fe / "k8")
    cz = np.array([x["curvature_z"] for x in out["components"]], float)
    assert abs(np.median(cz)) < 1.5, cz
