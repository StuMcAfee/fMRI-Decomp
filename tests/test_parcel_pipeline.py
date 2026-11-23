"""Parcel pipeline measures (docs/PARCEL_ICA_PLAN.md section 8): node count, participation coefficient, segregation."""
import numpy as np

from mejoint import parcel_pipeline as pp


def test_node_count_gaps_and_single_parcels():
    # a chain of parcels 1-2-3-4-5-6 (adjacent in sequence); winners A A gap A B B
    ids = np.arange(1, 7)
    pairs = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    win = np.array([0, 0, -1, 0, 1, 1])
    nc = pp.node_count(ids, win, pairs)
    assert nc["node_count"] == 3 and nc["patch_sizes"] == [2, 2, 1]           # the gap splits component 0
    assert pp.node_count(ids, -np.ones(6, int), pairs)["node_count"] == 0


def test_participation_coefficient_and_segregation():
    # two modules, strong within, none between -> PC 0, segregation 1; uniform weights -> PC 0.5, segregation 0
    lab = np.array(["a", "a", "b", "b"], dtype=object)
    W = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], float)
    assert np.allclose(pp.participation_coefficient(W, lab), 0.0)
    assert abs(pp.segregation_positive(W, lab) - 1.0) < 1e-12
    U = np.ones((4, 4)) - np.eye(4)
    pc = pp.participation_coefficient(U, lab)
    assert np.allclose(pc, 1 - ((1 / 3) ** 2 + (2 / 3) ** 2))
    assert abs(pp.segregation_positive(U, lab)) < 1e-12
    # nodes without a module are left out of the graph
    lab2 = np.array(["a", "a", "b", ""], dtype=object)
    assert np.isnan(pp.participation_coefficient(U, lab2)[3])


def test_positive_weights_and_reliability():
    C = np.array([[1.0, 1.3, -0.4], [1.3, 1.0, np.nan], [-0.4, np.nan, 1.0]])
    W = pp.positive_weights(C)
    assert W[0, 1] == 1.0 and W[0, 2] == 0.0 and W[1, 2] == 0.0 and np.all(np.diag(W) == 0)
    g = np.random.default_rng(0)
    rel = pp.reliability(g.standard_normal((50, 800)), 0.75, (0.01, 0.1))      # pure unit noise -> about 0
    assert abs(np.median(rel)) < 0.1


def test_map_fidelity_matches_simulation():
    """The expected map fidelity from an eigenvalue agrees with the observed overlap of a planted spike."""
    from mejoint import parcel_ica as pi
    g = np.random.default_rng(0)
    P, T = 600, 300
    for ell in (4.0, 8.0):
        est, emp = [], []
        for _ in range(6):
            v = g.standard_normal(P)
            v /= np.linalg.norm(v)
            X = g.standard_normal((P, T)) + np.sqrt(ell - 1) * np.outer(v, g.standard_normal(T))
            lam, U = np.linalg.eigh(X @ X.T / T)
            est.append(pi.map_fidelity(lam[-1], P, T)[0])
            emp.append(abs(U[:, -1] @ v))
        assert abs(np.mean(est) - np.mean(emp)) < 0.05
    assert pi.map_fidelity(pi.mp_edge(P, T), P, T)[0] == 0.0
    lam7 = pi.fidelity_threshold(0.7, P, T)
    assert abs(pi.map_fidelity(lam7, P, T)[0] - 0.7) < 1e-6


def test_significant_edges_keep_real_connections():
    """Two coupled groups of parcels plus independent ones: edges within the coupled groups pass the circular-shift
    null at FDR 0.05; edges among independent parcels mostly do not."""
    from mejoint.parcel_sim import band_noise
    g = np.random.default_rng(1)
    T, tr = 600, 0.75
    s1, s2 = band_noise(T, tr, (0.01, 0.1), g, 2)
    z = np.vstack([np.outer(np.ones(10), s1) * 2.0, np.outer(np.ones(10), s2) * 2.0, np.zeros((20, T))])
    z = z + band_noise(T, tr, (0.01, 0.1), g, 40) * 1.0 + g.standard_normal((40, T))
    from mejoint import regions
    corr = regions.coupling(z, np.ones(40), tr, (0.01, 0.1))["corr"]
    null = pp.pr_null(z, np.ones(40), tr, (0.01, 0.1), 40, 40, g)
    S = pp.significant_edges(corr, null, 0.05)
    within = S[:10, :10][np.triu_indices(10, 1)].mean()
    indep = S[20:, 20:][np.triu_indices(20, 1)].mean()
    assert within > 0.9 and indep < 0.1


def test_remove_first_pattern():
    """A shared pattern plus two groups: after removal the shared part is gone, the groups remain."""
    g = np.random.default_rng(2)
    P, T = 30, 2000
    shared = g.standard_normal(T)
    grp = g.standard_normal((2, T))
    X = np.outer(np.ones(P), shared) * 1.5 + np.vstack([np.outer(np.ones(15), grp[0]), np.outer(np.ones(15), grp[1])]) + g.standard_normal((P, T))
    R = np.corrcoef(X)
    Rb = pp.remove_first_pattern(R)
    assert np.allclose(np.diag(Rb), 1.0)
    within = Rb[:15, :15][np.triu_indices(15, 1)].mean()
    between = Rb[:15, 15:].mean()
    assert R[:15, 15:].mean() > 0.5 and within > 0.2 and between < 0.0


def test_dominance_confidence():
    """Non-overlapping simulated networks: most strong parcels are confidently dominated; a parcel loaded equally on
    two components is not."""
    from mejoint import parcel_ica as pi
    from mejoint import parcel_sim as ps
    d = ps.simulate(600, T=600, reliability=0.6, seed=7, include=("network",))
    pr = pi.prepare(d["c0"], d["c1"], d["R"], d["noise_var"])
    ica = pi.spatial_ica(pr["zb"], 7, range(4))
    dom = pp.dominance(pr["zb"], ica["maps"], ica["tc"])
    strong = dom["z1"] >= 1
    assert (dom["confidence"][strong] >= 0.95).mean() > 0.6
    # two equal loadings -> confidence about 0.5
    g = np.random.default_rng(1)
    tc = g.standard_normal((2, 600))
    tc /= tc.std(1, keepdims=True)
    m0 = g.normal(0, 1, 200)
    m0[0] = 3.0
    m1 = g.permutation(m0[1:])
    maps = np.column_stack([m0, np.r_[3.0, m1]])                       # identical value distributions, equal loading
    assert np.isclose(maps[:, 0].std(), maps[:, 1].std())
    z = maps @ tc + g.standard_normal((200, 600))
    d2 = pp.dominance(z, maps, tc)
    assert 0.2 < d2["confidence"][0] < 0.8


def test_reliability_floor_from_correction_precision():
    """About 0.31 for 797 frames at TR 0.75 s in 0.01-0.1 Hz; higher with fewer frames; the stated precision holds."""
    r = pp.reliability_floor(797, 0.75, (0.01, 0.1), 0.15)
    assert 0.28 < r < 0.34
    assert pp.reliability_floor(531, 0.75, (0.01, 0.1), 0.15) > r
