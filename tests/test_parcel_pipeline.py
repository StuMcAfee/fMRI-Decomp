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
