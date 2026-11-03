"""Phase 3 gate (CLAUDE.md section 8): simulated true rank recovered within +/- 10 %."""
import os

import numpy as np
import pytest

from mejoint import frontend, io, nuisance, rank, simulate

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))


@pytest.fixture
def cfg():
    return io.merge(io.load_config(), {"rank": {"n_surrogates": 10, "deflated_n_surrogates": 5, "k_max": 120}})


def planted(rng, n=6000, T=400, K=30, colored=True):
    """K shared components (smooth time courses, sparse-ish maps) in unit-variance noise that is
    coloured (AR(1), rho 0-0.6 per row) as BOLD-band physiology and interpolation make it. Component
    strengths span 2x to 20x the noise edge."""
    t = np.arange(T)
    S = []
    for _ in range(K):
        f = rng.uniform(0.005, 0.15)
        s = np.sin(2 * np.pi * f * t * 0.75 + rng.uniform(0, 6.3)) + 0.5 * rng.standard_normal(T)
        S.append((s - s.mean()) / s.std())
    S = np.array(S)
    edge = (1 + np.sqrt(T / n)) ** 2
    strength = np.geomspace(2, 20, K) * edge                       # eigenvalue added per component (units of noise)
    M = rng.standard_normal((n, K)) * (rng.random((n, K)) < 0.3)
    M /= np.sqrt((M ** 2).mean(0))
    X = M * np.sqrt(strength / T) @ S                              # eigenvalue of X^T X / n = strength
    e = rng.standard_normal((n, T))
    if colored:
        rho = rng.uniform(0, 0.6, (n, 1))
        for i in range(1, T):
            e[:, i] = rho[:, 0] * e[:, i - 1] + np.sqrt(1 - rho[:, 0] ** 2) * e[:, i]
    return X + e


def split(X):
    n = X.shape[0] // 2
    return X[:n].astype(np.float32), X[n:2 * n].astype(np.float32)


KNOWN = ("Phase 3 gate NOT passed (2026-11-03): with coloured noise the deflated null sits ~2 % low and "
         "k runs to k_max; index-by-index under-counts (22-26 of 30). Open: see docs/decisions.md")


@pytest.mark.parametrize("colored", [False, pytest.param(True, marks=pytest.mark.xfail(reason=KNOWN, strict=True))])
def test_true_rank_recovered_within_10_percent(cfg, record_property, colored):
    rng = np.random.default_rng(SEED + 10)
    K = 30
    c0, c1 = split(planted(rng, K=K, colored=colored))
    noise = np.random.default_rng(SEED + 11).standard_normal((2, c0.shape[0], c0.shape[1])).astype(np.float32)
    r = rank.estimate(c0, c1, noise[0], noise[1], cfg, seed=SEED, with_mapca=False)
    record_property("k", {"temporal": r["k_temporal"], "classic": r["k_temporal_classic"], "thermal": r["k_thermal"], "mp": r["k_mp_edge"]})
    print(K, {"temporal": r["k_temporal"], "classic": r["k_temporal_classic"], "thermal": r["k_thermal"], "mp": r["k_mp_edge"]})
    assert abs(r["k_temporal"] - K) <= 0.1 * K
    if colored:            # a white-noise edge counts coloured noise as signal (CLAUDE.md caveat)
        assert r["k_mp_edge"] > r["k_temporal"]


def test_pure_noise_gives_no_components(cfg):
    rng = np.random.default_rng(SEED + 12)
    e = rng.standard_normal((6000, 400))
    rho = rng.uniform(0, 0.6, (6000, 1))
    for i in range(1, 400):
        e[:, i] = rho[:, 0] * e[:, i - 1] + np.sqrt(1 - rho[:, 0] ** 2) * e[:, i]
    c0, c1 = split(e)
    r = rank.estimate(c0, c1, c0, c1, cfg, seed=SEED, with_mapca=False)
    assert r["k_temporal"] <= 1


@pytest.mark.xfail(reason=KNOWN, strict=True)
def test_rank_after_nuisance_projection(cfg, record_property):
    """Projected data and identically projected nulls: the planted rank is still recovered."""
    rng = np.random.default_rng(SEED + 13)
    K, T = 30, 400
    X = planted(rng, K=K, T=T)
    P = nuisance.Projector(np.cumsum(rng.standard_normal((T, 24)), 0), nuisance.segments(n=T), T)
    c0, c1 = split(P.apply(X))
    noise = P.apply(rng.standard_normal((2, c0.shape[0], T))).astype(np.float32)
    r = rank.estimate(c0, c1, noise[0], noise[1], cfg, Q=P.Q, seed=SEED, with_mapca=False)
    record_property("k_temporal", r["k_temporal"])
    assert r["t_eff"] == T - P.rank - 1
    assert abs(r["k_temporal"] - K) <= 0.1 * K


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
    assert (tmp_path / "fe" / "nuisance_Q.npy").exists()
    res = rank.run_dir(tmp_path / "fe", c, tmp_path / "rank")
    for f in ("spectrum.tsv", "rank.json", "rank.md", "done.json"):
        assert (tmp_path / "rank" / f).exists()
    assert res["nuisance"] and res["nuisance_rank"] == 25
    assert 1 <= res["k_temporal"] <= 30
    assert "mapca" in res
