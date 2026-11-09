"""Route B, plan step 2 (docs/ROUTE_B_PLAN.md): region averaging, the split, the region noise estimate, shares,
coupling and the circular-shift null, on synthetic voxels at the real echo times and noise ratios.

- averaging keeps the model linear: noise-free regions return their a(t) and b(t) exactly;
- the region noise measured from the leftover channels matches the truth, including when neighbouring voxels'
  noise is correlated (motion-correction interpolation); the independent-voxel value would be too small;
- BOLD-like share ~1 for a pure R2* region, ~0 for a pure S0 region, the energy-weighted truth for a mixed one;
- dimensionality: regions driven by independent time courses have a higher participation ratio than regions
  sharing one; under independence the circular-shift null holds the observed values;
- segregation is positive when correlations are higher within networks than between them.
"""
import os

import numpy as np
import pytest

from mejoint import inject, regions

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))
TE = np.array([16.20, 31.22, 46.24, 61.26])
RATIOS = np.array([1.0, 0.721, 0.631, 0.560])
TR = 0.75
BAND = (0.01, 0.10)


def voxels(rng, P, n_per, T, a, b, sigma=8.0, rho_noise=0.0, s0=1000.0):
    """P regions of n_per voxels. a, b: (P, T) region S0-like (fraction) and R2*-like (s^-1) changes (every voxel
    of a region carries them). Gaussian thermal noise with per-echo ratios; rho_noise: correlation of the noise
    between voxels of the same region (interpolation-like)."""
    V = P * n_per
    labels = np.repeat(np.arange(1, P + 1), n_per)
    t2s = rng.uniform(35, 60, V)
    mu_true = s0 * np.exp(-TE[None, :] / t2s[:, None])                     # (V, E)
    y = a[labels - 1][:, None, :] - (TE[None, :, None] / 1000.0) * b[labels - 1][:, None, :]
    S = mu_true[:, :, None] * np.exp(y)
    sd = sigma * RATIOS[None, :, None]
    if rho_noise > 0:
        shared = np.repeat(rng.standard_normal((P, 4, T)), n_per, axis=0)
        own = rng.standard_normal((V, 4, T))
        noise = np.sqrt(rho_noise) * shared + np.sqrt(1 - rho_noise) * own
    else:
        noise = rng.standard_normal((V, 4, T))
    S = S + sd * noise
    mu = S.mean(axis=2)
    return S, labels, mu, np.full(V, sigma), np.ones((V, 4), bool)


def slow(rng, n, T):
    return inject.time_courses(T, TR, BAND, n, rng).T                    # (n, T), unit SD, orthogonal


def test_noise_free_regions_recover_a_and_b():
    rng = np.random.default_rng(SEED)
    P, T = 3, 400
    u = slow(rng, 3, T)
    a = np.stack([0 * u[0], 0.01 * u[1], 0.016 * u[2]])
    b = np.stack([0.5 * u[0], 0 * u[1], 0.5 * u[2]])
    S, labels, mu, sigma, use = voxels(rng, P, 30, T, a, b, sigma=1e-6)
    rs = regions.region_signals(S, labels, mu, sigma, RATIOS, use)
    sp = regions.split(rs["ybar"], rs["wsum"], TE)
    for p in range(P):
        assert np.allclose(sp["a"][p], a[p] - a[p].mean(), atol=2e-4), p
        assert np.allclose(sp["b"][p], b[p] - b[p].mean(), atol=0.01), p


@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_region_noise_from_leftover_channels(rho):
    rng = np.random.default_rng(SEED + 1)
    P, n_per, T = 6, 60, 600
    zero = np.zeros((P, T))
    S, labels, mu, sigma, use = voxels(rng, P, n_per, T, zero, zero, rho_noise=rho)
    rs = regions.region_signals(S, labels, mu, sigma, RATIOS, use)
    sp = regions.split(rs["ybar"], rs["wsum"], TE)
    nv = regions.leftover_noise(sp["r"])
    # truth: independent voxels give 1; a shared fraction rho multiplies the region-average variance by
    # 1 + rho (n_per - 1)
    truth = 1.0 + rho * (n_per - 1)
    assert np.all(np.abs(nv / truth - 1) < 0.15), (nv, truth)
    # the channels' own variance agrees (no signal): the leftover estimate is the right noise for c0, c1
    assert np.all(np.abs(sp["c"].var(axis=2).mean(1) / nv - 1) < 0.2)


def test_shares_pure_and_mixed():
    rng = np.random.default_rng(SEED + 2)
    P, T = 3, 800
    u = slow(rng, 3, T)
    a = np.stack([0 * u[0], 0.006 * u[1], 0.008 * u[2]])
    b = np.stack([0.3 * u[0], 0 * u[1], 0.25 * u[2]])
    S, labels, mu, sigma, use = voxels(rng, P, 80, T, a, b, rho_noise=0.1)
    rs = regions.region_signals(S, labels, mu, sigma, RATIOS, use)
    sp = regions.split(rs["ybar"], rs["wsum"], TE)
    nv = regions.leftover_noise(sp["r"])
    sh = regions.shares(sp, nv, TR, BAND)
    # truth for the mixed region: the same energies computed on noise-free data
    S0, *_ = voxels(np.random.default_rng(9), P, 80, T, a, b, sigma=1e-6)
    sp0 = regions.split(regions.region_signals(S0, labels, S0.mean(2), np.full(S0.shape[0], 8.0), RATIOS, use)["ybar"],
                        rs["wsum"], TE)
    sh0 = regions.shares(sp0, np.zeros(P), TR, BAND)
    assert sh["bold_share"][0] > 0.95, sh["bold_share"]
    assert sh["bold_share"][1] < 0.05, sh["bold_share"]
    assert abs(sh["bold_share"][2] - sh0["bold_share"][2]) < 0.05, (sh["bold_share"], sh0["bold_share"])
    assert np.all(np.abs(sh["leftover_share"]) < 0.05), sh["leftover_share"]


def test_curvature_shows_as_leftover():
    """A compartment-volume change (not linear in TE) puts energy in the leftover share."""
    rng = np.random.default_rng(SEED + 3)
    P, n_per, T = 2, 80, 800
    u = slow(rng, 2, T)
    zero = np.zeros((P, T))
    S, labels, mu, sigma, use = voxels(rng, P, n_per, T, zero, np.stack([0.3 * u[0], 0 * u[1]]))
    # region 2: an extra signal with a curved (U-shaped) echo profile, outside span{1, TE}. (A profile strong at
    # echo 1 only is mostly absorbed as an S0 + R2* mixture: only its non-linear part reaches the leftover.)
    prof = np.array([1.0, -1.0, -1.0, 1.0]) * 0.005
    S[labels == 2] *= np.exp(prof[None, :, None] * u[1][None, None, :])
    mu = S.mean(2)
    rs = regions.region_signals(S, labels, mu, sigma, RATIOS, use)
    sp = regions.split(rs["ybar"], rs["wsum"], TE)
    sh = regions.shares(sp, regions.leftover_noise(sp["r"]), TR, BAND)
    assert sh["leftover_share"][0] < 0.05 and sh["leftover_share"][1] > 0.1, sh["leftover_share"]


def test_dimensionality_and_circular_shift_null():
    rng = np.random.default_rng(SEED + 4)
    P, T = 16, 800
    nets = np.repeat(np.arange(4), 4)
    nv = np.full(P, 0.01)
    # A: four networks, each driven by its own time course (plus a little own signal)
    drivers = slow(rng, 4, T)
    own = np.stack([slow(rng, 1, T)[0] for _ in range(P)])
    xa = drivers[nets] + 0.4 * own + rng.standard_normal((P, T)) * 0.1
    # B: one time course drives everything
    xb = drivers[[0] * P] + 0.4 * own + rng.standard_normal((P, T)) * 0.1
    ca, cb = regions.coupling(xa, nv, TR, BAND), regions.coupling(xb, nv, TR, BAND)
    pra, prb = regions.participation_ratio(ca["corr"]), regions.participation_ratio(cb["corr"])
    assert pra > 2.5 and prb < 1.5 and pra > prb, (pra, prb)
    seg = regions.segregation(ca["corr"], nets)
    assert seg > 0.5, seg
    null = regions.shift_null(xa, nv, TR, BAND, 200, 40, rng, networks=nets)
    assert pra < np.quantile(null["pr"], 0.05)                       # integration beyond chance: lower dimensionality
    k = regions.count_above_null(regions.eigenvalues(ca["corr"]), null["eig"])
    assert k == 4, k
    # independent regions: the observed participation ratio lies inside its null, about 5 % of edges beyond it
    # drawn independently (one orthogonalized set would make the regions exactly uncorrelated, unlike real data)
    xi = np.stack([slow(rng, 1, T)[0] for _ in range(P)]) + rng.standard_normal((P, T)) * 0.1
    ci = regions.coupling(xi, nv, TR, BAND)
    ni = regions.shift_null(xi, nv, TR, BAND, 200, 40, rng)
    lo, hi = np.quantile(ni["pr"], [0.01, 0.99])
    assert lo <= regions.participation_ratio(ci["corr"]) <= hi
    iu = np.triu_indices(P, 1)
    thr = np.quantile(np.abs(ni["edges"]), 0.95, axis=0)
    frac = np.mean(np.abs(ci["corr"][iu]) > thr)
    assert frac < 0.15, frac


def test_global_share():
    rng = np.random.default_rng(SEED + 5)
    P, T = 12, 800
    g = slow(rng, 1, T)[0]
    x = 0.8 * g[None, :] + 0.6 * slow(rng, P, T)
    c = regions.coupling(x, np.zeros(P), TR, BAND)
    gs = regions.global_share(c["x_band"], c["signal_var"])
    assert 0.55 < gs < 0.75, gs                                       # truth 0.64 / (0.64 + 0.36) = 0.64
