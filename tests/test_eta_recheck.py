"""`mejoint eta-recheck`: eta debiased with each voxel's own noise per channel."""
import os

import numpy as np
from scipy import signal

from mejoint import eta_recheck, io

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))


def test_excess_c1_noise_inflates_reported_eta_and_the_recheck_removes_it(tmp_path):
    """Pure dR2* components; c0 noise unit and white (thermal), c1 noise twice the variance and correlated in
    time (AR(1) 0.7). The thermal debiasing leaves the excess c1 noise in E_S0 (eta > 0); the empirical debiasing
    brings eta back to ~0 and reports the excess (noise level c1 ~ 2, colour factor > 1)."""
    rng = np.random.default_rng(SEED)
    T, k, V = 500, 3, 4000
    b, a = signal.butter(3, 0.08)
    M = signal.filtfilt(b, a, rng.standard_normal((T, k)), axis=0)
    M = (M - M.mean(0)) / M.std(0)
    R = np.zeros((V, 2, 2))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 1] = 40.0, -30.0, 6.0
    m0_true = rng.standard_normal((V, k))         # dR2* signal above the per-voxel beta noise
    c0 = m0_true @ M.T + rng.standard_normal((V, T))
    e1 = signal.lfilter([1.0], [1.0, -0.7], rng.standard_normal((V, T)), axis=1)
    c1 = e1 / e1.std() * np.sqrt(2.0)
    Mc = M - M.mean(0)
    P = Mc @ np.linalg.inv(Mc.T @ Mc)
    m0 = (c0 - c0.mean(1, keepdims=True)) @ P
    m1 = (c1 - c1.mean(1, keepdims=True)) @ P
    vb = np.broadcast_to(np.diag(np.linalg.inv(Mc.T @ Mc)), (V, k))
    fe, kd = tmp_path / "fe", tmp_path / "k3"
    fe.mkdir(); kd.mkdir()
    np.save(fe / "c0.npy", c0.astype(np.float32)); np.save(fe / "c1.npy", c1.astype(np.float32)); np.save(fe / "R.npy", R)
    io.write_tsv(kd / "time_courses.tsv", {f"c{j + 1}": list(M[:, j]) for j in range(k)})
    for name, arr in (("m0_cf", m0), ("m1_cf", m1), ("vb_cf", vb)):
        np.save(kd / f"{name}.npy", arr.astype(np.float32))
    r = eta_recheck.recheck_k(fe, kd, max_lag=40)
    assert r["noise_level_c0"] == np.float64(r["noise_level_c0"]) and abs(r["noise_level_c0"] - 1) < 0.1
    assert abs(r["noise_level_c1"] - 2) < 0.2 and r["colour_factor_c1"] > 2 and abs(r["colour_factor_c0"] - 1) < 0.2
    assert r["eta_reported_median"] > 0.1                      # thermal debiasing: spurious dS0
    assert abs(r["eta_empirical_median"]) < 0.05               # empirical debiasing: pure dR2* again
    assert r["eta_white_empirical_median"] > r["eta_empirical_median"] + 0.02   # the level alone is not enough
    assert (kd / "eta_recheck.json").exists()
