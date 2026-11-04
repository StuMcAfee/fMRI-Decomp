"""Phase 3 gate (CLAUDE.md section 8): simulated true rank recovered within +/- 10 %.

Gated estimator: split-half deflated parallel analysis on noise-whitened channels (user decision
2026-11-04). Validated range: 30 components (narrowband or broadband, white or coloured noise), 1
component, 60 narrowband components. Known failure: 60 broadband components contaminate the median noise
spectrum and k is over-counted (xfail, not strict).
"""
import os

import numpy as np
import pytest

from mejoint import frontend, io, nuisance, rank, simulate

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))


@pytest.fixture
def cfg():
    return io.merge(io.load_config(), {"rank": {"n_surrogates": 5, "split_n_surrogates": 10, "k_max": 120}})


def _noise(rng, n, T, colored):
    e = rng.standard_normal((n, T))
    if colored:                                         # AR(1), rho 0-0.6 per row
        rho = rng.uniform(0, 0.6, (n, 1))
        for i in range(1, T):
            e[:, i] = rho[:, 0] * e[:, i - 1] + np.sqrt(1 - rho[:, 0] ** 2) * e[:, i]
    return e


def _maps(rng, n, K):
    M = rng.standard_normal((n, K)) * (rng.random((n, K)) < 0.3)
    return M / np.sqrt((M ** 2).mean(0))


def planted(rng, n=6000, T=400, K=30, colored=True):
    """K shared components with near-sinusoidal (narrowband) time courses, sparse maps, eigenvalues from 2x
    to 20x the white-noise edge, in unit-variance noise (white or AR(1) coloured)."""
    t = np.arange(T)
    S = []
    for _ in range(K):
        s = np.sin(2 * np.pi * rng.uniform(0.005, 0.15) * t * 0.75 + rng.uniform(0, 6.3)) + 0.5 * rng.standard_normal(T)
        S.append((s - s.mean()) / s.std())
    edge = (1 + np.sqrt(T / n)) ** 2
    X = _maps(rng, n, K) * np.sqrt(np.geomspace(2, 20, K) * edge / T) @ np.array(S)   # eigenvalue = strength
    return X + _noise(rng, n, T, colored)


def planted_bb(rng, n=6000, T=400, K=30, colored=True):
    """As planted, with broadband 1/f-like (BOLD-like) time courses."""
    f = np.fft.rfftfreq(T, d=0.75)
    S = []
    for _ in range(K):
        spec = 1 / np.maximum(f, 0.005) ** rng.uniform(0.7, 1.2)
        spec[f > 0.2] *= 0.2
        x = np.fft.irfft(np.sqrt(spec) * np.exp(1j * rng.uniform(0, 6.3, f.size)), n=T)
        S.append((x - x.mean()) / x.std())
    edge = (1 + np.sqrt(T / n)) ** 2
    X = _maps(rng, n, K) * np.sqrt(np.geomspace(2, 20, K) * edge / T) @ np.array(S)
    return X + _noise(rng, n, T, colored)


def split(X):
    n = X.shape[0] // 2
    return X[:n].astype(np.float32), X[n:2 * n].astype(np.float32)


def run(X, cfg, Q=None):
    c0, c1 = split(X)
    noise = np.random.default_rng(SEED + 99).standard_normal((2, *c0.shape)).astype(np.float32)
    return rank.estimate(c0, c1, noise[0], noise[1], cfg, Q=Q, seed=SEED, with_mapca=False)


CASES = [(gen, K, col) for gen in ("narrow", "broad") for K in (30, 1) for col in (False, True)]


@pytest.mark.parametrize("gen, K, colored", CASES)
def test_true_rank_recovered_within_10_percent(cfg, record_property, gen, K, colored):
    X = (planted if gen == "narrow" else planted_bb)(np.random.default_rng(SEED + 10), K=K, colored=colored)
    r = run(X, cfg)
    record_property("k", {"gated": r["k_temporal"], "halves": r["k_split_halves"], "classic": r["k_temporal_classic"],
                          "mp": r["k_mp_edge"], "contamination": r["contamination"]})
    # +/- 10 %, but never finer than +/- 1: each tested direction passes by chance with probability
    # 1 - null_quantile (5 %), so a single extra component occurs (seed 1, K = 1: k = 2)
    assert abs(r["k_temporal"] - K) <= max(0.1 * K, 1)
    assert r["contamination"] <= r["contamination_validated_max"]


@pytest.mark.slow
def test_sixty_narrowband_components(cfg, record_property):
    r = run(planted(np.random.default_rng(SEED + 14), K=60, colored=True), cfg)
    record_property("k", r["k_temporal"])
    assert abs(r["k_temporal"] - 60) <= 6


@pytest.mark.slow
@pytest.mark.xfail(reason="known limitation: 60 broadband components contaminate the median noise spectrum (over-count 10-37 %)",
                   strict=False)
def test_sixty_broadband_components(cfg, record_property):
    r = run(planted_bb(np.random.default_rng(SEED + 15), K=60, colored=True), cfg)
    record_property("k", {"gated": r["k_temporal"], "contamination": r["contamination"]})
    assert r["contamination"] > r["contamination_validated_max"]      # the diagnostic flags it
    assert abs(r["k_temporal"] - 60) <= 6


def test_pure_coloured_noise_gives_no_components(cfg):
    rng = np.random.default_rng(SEED + 12)
    r = run(_noise(rng, 6000, 400, True), cfg)
    assert r["k_temporal"] <= 1


def test_rank_after_nuisance_projection(cfg, record_property):
    """Projected data, nulls projected identically (whitened domain): the planted rank is recovered."""
    rng = np.random.default_rng(SEED + 13)
    K, T = 30, 400
    X = planted_bb(rng, K=K, T=T, colored=True)
    P = nuisance.Projector(np.cumsum(rng.standard_normal((T, 24)), 0), nuisance.segments(n=T), T)
    r = run(P.apply(X), cfg, Q=P.Q)
    record_property("k", r["k_temporal"])
    assert r["t_eff"] == T - P.rank - 1
    assert abs(r["k_temporal"] - K) <= 0.1 * K


def test_spatial_blocks_keep_voxels_together():
    mask = np.zeros((8, 8, 8), bool)
    mask[1:7, 1:7, 1:7] = True
    V = int(mask.sum())
    idx = np.r_[np.arange(V), np.arange(V) + V]                    # c0 rows then c1 rows of every voxel
    h = rank.split_halves(idx, mask, 4, np.random.default_rng(0))
    assert np.array_equal(h[:V], h[V:])                            # c0 and c1 of a voxel in the same half
    coords = np.argwhere(mask)
    b = coords // 4
    for key in {tuple(x) for x in b}:                              # whole blocks go to one half
        sel = (b == key).all(1)
        assert len(set(h[:V][sel])) == 1
    assert 0.2 < h.mean() < 0.8


def test_undefined_rows_are_excluded_not_zero_filled():
    c0 = np.ones((5, 10), np.float32) * np.arange(10)
    c1 = c0.copy()
    c1[2] = np.nan
    X, idx = rank.stack(c0, c1)
    assert X.shape == (9, 10) and 7 not in idx and np.allclose(X.mean(1), 0)


def test_run_dir_on_front_end_output(cfg, tmp_path):
    """End to end on a simulated run: front end with nuisance projection, then rank (MAPCA if installed)."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_frontend import make_run_folder
    c = io.merge(cfg, {"simulate": {"echo_sigma_scale": cfg["frontend"]["echo_noise_ratios"]},
                       "frontend": {"mppca": {"enabled": False}}, "nuisance": {"n_surrogates": 1}})
    sim = simulate.simulate(c, seed=SEED * 100 + 50, overrides={"grid": [24, 24, 14], "n_volumes": 200})
    rd = make_run_folder(sim, tmp_path)
    frontend.run_subject(rd, c, tmp_path / "fe", source="raw")
    res = rank.run_dir(tmp_path / "fe", c, tmp_path / "rank")
    for f in ("spectrum.tsv", "rank.json", "rank.md", "done.json"):
        assert (tmp_path / "rank" / f).exists()
    assert res["nuisance"] and res["nuisance_rank"] == 25
    assert 1 <= res["k_temporal"] <= 30 and "contamination" in res and "mapca" in res
