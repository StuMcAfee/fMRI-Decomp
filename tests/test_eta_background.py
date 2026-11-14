"""Background-corrected eta (eta_background.py): a known pure-dR2* and a known pure-dS0 source on top of a strong,
broad, slow background with mixed physics. The thermal-only eta of the true time course is pulled toward the
background; with the circular-shift (or phase) surrogates' energies removed it returns to ~0 and ~1."""
import os

import numpy as np

from mejoint import eta_background as eb
from mejoint import inject, metrics

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))


def make(rng, V=20000, T=800, n_bg=40):
    R = np.zeros((V, 2, 2))
    R[:, 0, 0] = rng.uniform(2000, 5000, V)                           # front-end scale: ~(mu / sigma) x |TE| (ms)
    R[:, 0, 1] = -rng.uniform(50, 100, V)
    R[:, 1, 1] = rng.uniform(20, 50, V)
    # sources and background drawn independently (one orthogonalized set would make the sources pick up no
    # background at all, unlike real data)
    U = np.column_stack([inject.time_courses(T, 0.75, [0.01, 0.1], 2, rng), inject.time_courses(T, 0.75, [0.01, 0.1], n_bg, rng)])
    c0 = rng.standard_normal((V, T))
    c1 = rng.standard_normal((V, T))
    # broad slow background: mixed dS0 and dR2* patterns over the whole brain
    for i in range(n_bg):
        s = rng.standard_normal(V) * 0.003
        q = rng.standard_normal(V) * 0.05 / 1000.0
        m0, m1 = R[:, 0, 0] * q + R[:, 0, 1] * s, R[:, 1, 1] * s
        c0 += m0[:, None] * U[:, 2 + i]
        c1 += m1[:, None] * U[:, 2 + i]
    fp_r2, fp_s0 = np.zeros(V, bool), np.zeros(V, bool)
    fp_r2[:800], fp_s0[800:1600] = True, True
    q = np.where(fp_r2, 0.4 / 1000.0, 0.0)
    s = np.where(fp_s0, 0.012, 0.0)
    c0 += (R[:, 0, 0] * q)[:, None] * U[:, 0] + (R[:, 0, 1] * s)[:, None] * U[:, 1]
    c1 += (R[:, 1, 1] * s)[:, None] * U[:, 1]
    return c0, c1, R, U[:, :2]


def test_whole_brain_interval_covers_the_truth():
    rng = np.random.default_rng(SEED + 3)
    c0, c1, R, Ut = make(rng)
    bg = eb.background(c0, c1, R, Ut, n=40, rng=rng, min_shift=40)
    thermal = bg["own"]["eta"]
    for m in ("shift", "phase"):
        res = eb.corrected(bg, m)
        # 10 simulated datasets: corrected 0.00 +- 0.04 and 1.04 +- 0.04; thermal-only 0.20 and 0.85
        assert abs(res["eta"][0]) < 0.15, (m, res["eta"], thermal)
        assert abs(res["eta"][1] - 1) < 0.15, (m, res["eta"], thermal)
        assert res["eta_lo"][0] < res["eta"][0] < res["eta_hi"][0]
        assert np.all(res["p"] <= 0.05), res["p"]
    assert thermal[0] > 0.1 and thermal[1] < 0.92, thermal             # the background biased the thermal-only eta


def test_surrogates_shape_and_properties():
    rng = np.random.default_rng(SEED)
    t = inject.time_courses(800, 0.75, [0.01, 0.1], 1, rng)[:, 0]
    S = eb.surrogates(t, 30, rng, "shift", 40)
    assert S.shape == (800, 30)
    assert np.allclose(np.sort(S, axis=0), np.sort(t)[:, None])         # same values, rotated
    assert np.all(np.abs(S.T @ t / 800) < 0.6)
    P = eb.surrogates(t, 30, rng, "phase", 40)
    assert np.allclose(np.abs(np.fft.rfft(P, axis=0)), np.abs(np.fft.rfft(t))[:, None], atol=1e-8)


def test_pure_noise_background_is_unbiased():
    """No structured background, time courses not fitted to the data: corrected and thermal eta agree in expectation
    and the time courses are not above their surrogates (a data-derived component would be: p is descriptive)."""
    rng = np.random.default_rng(SEED + 4)
    V, T = 5000, 800
    c0, c1 = rng.standard_normal((V, T)), rng.standard_normal((V, T))
    R = np.zeros((V, 2, 2)); R[:, 0, 0], R[:, 0, 1], R[:, 1, 1] = 30.0, -10.0, 10.0
    Ut = inject.time_courses(T, 0.75, [0.01, 0.1], 3, rng)
    bg = eb.background(c0, c1, R, Ut, n=40, rng=rng, min_shift=40)
    res = eb.corrected(bg, "shift")
    assert np.all(res["p"] > 0.01), res["p"]
    tot = bg["shift"]["E_S0"] + bg["shift"]["E_R2"]
    assert abs(tot.mean()) < 3 * tot.std() / np.sqrt(tot.size) + 0.5   # surrogate energy ~0 after thermal debiasing


def test_support_version_recovers_pure_sources():
    """eta where the component lives (voxels chosen in one half of the run, measured in the other): close to the
    truth even when the whole-brain correction is thrown off by a low-rank background."""
    rng = np.random.default_rng(SEED + 3)
    c0, c1, R, Ut = make(rng)
    bg = eb.background(c0, c1, R, Ut, n=40, rng=rng, min_shift=40)
    for m in ("shift", "phase"):
        res = eb.corrected(bg, m, "support")
        assert abs(res["eta"][0]) < 0.15, (m, res["eta"])
        assert abs(res["eta"][1] - 1) < 0.15, (m, res["eta"])
        assert np.all(res["p"] <= 0.05), res["p"]
        assert np.all(res["n_support"] > 300), res["n_support"]
