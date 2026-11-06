"""eta recomputed with empirical noise (`mejoint eta-recheck`; user decision 2026-11-06), a sensitivity
analysis beside the reported eta.

The reported eta (CLAUDE.md 7.1) is debiased with the beta noise variance for UNIT WHITE noise in both
channels (the thermal expectation after whitening). Here the same cross-fitted maps (m0_cf, m1_cf) are
debiased with each voxel's own noise, per channel: the residual of the channel after removing all k component
time courses (and the mean), its autocovariance up to `max_lag` frames, and the exact variance of each beta
under that autocovariance, var(beta_j) = sum_l gamma(l) c_j(l), with c_j(l) the lag-l autocorrelation sum of
the j-th column of P = M (M^T M)^-1. Reported per component: eta as reported (recomputed, a check), eta with
the empirical noise LEVEL only (white), and eta with level and temporal correlation; per run: the noise level
relative to thermal in c0 and c1 (1 = thermal) and the colour factor.

The residual has no power in the regressors' directions, so its autocovariance under-states the betas' noise
(about 10 % for 3 slow regressors and AR(1) 0.7 noise); a per-channel factor from noise simulated with the
channel's median autocovariance corrects it (`calibrate`). Approximations: one factor per channel and
component (spectral shape varies across voxels); the residual and P use the full-data time courses, while the
cross-fitted betas came from time courses estimated on the other voxel half (matched to these, Phase 4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def energies2(m0, m1, R, vb0, vb1) -> dict:
    """metrics.energies with separate beta noise variances for the c0 and c1 betas (V, k)."""
    R00, R01, R11 = R[:, 0, 0][:, None], R[:, 0, 1][:, None], R[:, 1, 1][:, None]
    ok = np.isfinite(m0) & np.isfinite(m1) & (R11 > 0) & np.isfinite(vb0) & np.isfinite(vb1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ws = (R01 ** 2 + R11 ** 2) / R11 ** 2
        r2 = m0 - R01 * m1 / R11
        e_s0 = np.where(ok, ws * m1 ** 2 - ws * vb1, 0.0).sum(0)
        e_r2 = np.where(ok, r2 ** 2 - (vb0 + R01 ** 2 / R11 ** 2 * vb1), 0.0).sum(0)
        tot = e_s0 + e_r2
        eta = np.where(tot != 0, e_s0 / tot, np.nan)
    return {"E_S0": e_s0, "E_R2": e_r2, "eta": eta}


def beta_noise(x, M, max_lag: int, n_nuisance: int = 0, chunk: int = 10000, calib: bool = True, seed: int = 0):
    """Per row of x (n, T): beta noise variance per component (n, k) from the residual autocovariance, the same
    with the residual variance only (white), and the residual variance (lag 0, dof-corrected)."""
    T, k = M.shape
    Mc = M - M.mean(0)
    P = Mc @ np.linalg.inv(Mc.T @ Mc)                                  # (T, k): beta = x_demeaned @ P
    L = int(min(max_lag, T - 1))
    c = np.stack([np.sum(P[: T - l] * P[l:], axis=0) for l in range(L + 1)])   # (L+1, k)
    C = c * np.r_[1.0, np.full(L, 2.0)][:, None]
    scale = T / max(T - k - 1 - n_nuisance, 1)                          # residual dof
    n = x.shape[0]
    vb, vbw, g0 = (np.full((n, k), np.nan), np.full((n, k), np.nan), np.full(n, np.nan))
    acs = []
    for i in range(0, n, chunk):
        xi = np.asarray(x[i:i + chunk], np.float64)
        ok = np.isfinite(xi).all(1)
        if not ok.any():
            continue
        xd = xi[ok] - xi[ok].mean(1, keepdims=True)
        res = xd - (xd @ P) @ Mc.T
        F = np.fft.rfft(res, n=2 * T, axis=1)
        ac = np.fft.irfft(np.abs(F) ** 2, n=2 * T, axis=1)[:, : L + 1] / T * scale
        idx = np.flatnonzero(ok) + i
        vb[idx] = ac @ C
        vbw[idx] = ac[:, :1] * C[0][None, :]
        g0[idx] = ac[:, 0]
        if calib:
            acs.append(ac)
    if calib and acs:
        ac_all = np.concatenate(acs)
        ac_med = np.median(ac_all / ac_all[:, :1], axis=0) * np.median(ac_all[:, 0])
        fac = calibrate(M, ac_med, max_lag, n_nuisance, np.random.default_rng(seed))
        vb = vb * fac[None, :]
        beta_noise.last_factor = fac
        beta_noise.last_factor_rel_se = calibrate.last_rel_se
    return vb, vbw, g0


def calibrate(M, ac_med, max_lag: int, n_nuisance: int, rng, n_sim: int = 5000) -> np.ndarray:
    """Correction factor per component (k,) for the bias of the residual-based estimate: the residual has no
    power in the directions of the regressors, so its autocovariance under-states the noise the betas carry.
    Noise with the channel's median autocovariance is simulated, its betas' actual variance compared with the
    estimate, and the ratio returned (1 = no bias)."""
    T, k = M.shape
    L = ac_med.size - 1
    f = np.fft.rfftfreq(T)
    w = 0.5 * (1 + np.cos(np.pi * np.arange(L + 1) / (L + 1)))           # Tukey taper: a valid spectrum
    S = ac_med[0] + 2 * np.sum((ac_med[1:] * w[1:])[:, None] * np.cos(2 * np.pi * np.arange(1, L + 1)[:, None] * f[None, :]), 0)
    S = np.maximum(S, 1e-6 * ac_med[0])
    Z = np.fft.irfft(np.fft.rfft(rng.standard_normal((n_sim, T)), axis=1) * np.sqrt(S)[None, :], n=T, axis=1)
    Mc = M - M.mean(0)
    beta = (Z - Z.mean(1, keepdims=True)) @ (Mc @ np.linalg.inv(Mc.T @ Mc))
    vb, _, _ = beta_noise(Z, M, max_lag, n_nuisance, calib=False)
    fac = np.mean(beta ** 2, 0) / np.mean(vb, 0)
    # relative standard error of the factor (Monte Carlo): shared by every voxel, so tests that sum over voxels
    # must add it to their noise (te_check)
    calibrate.last_rel_se = np.std(beta ** 2, 0) / np.sqrt(n_sim) / np.mean(beta ** 2, 0)
    return fac


def recheck_k(frontend_dir, k_dir, max_lag: int = 40) -> dict:
    fe, kd = Path(frontend_dir), Path(k_dir)
    c0, c1, R = np.load(fe / "c0.npy"), np.load(fe / "c1.npy"), np.load(fe / "R.npy")
    p = int(np.load(fe / "nuisance_Q.npy").shape[1]) if (fe / "nuisance_Q.npy").exists() else 0
    tcs = np.genfromtxt(kd / "time_courses.tsv", delimiter="\t", names=True)
    M = np.column_stack([tcs[n] for n in tcs.dtype.names]).astype(np.float64)
    m0, m1, vb_cf = (np.load(kd / f"{n}.npy").astype(np.float64) for n in ("m0_cf", "m1_cf", "vb_cf"))
    vb0, vb0w, g00 = beta_noise(c0, M, max_lag, p)
    f0 = getattr(beta_noise, "last_factor", np.ones(M.shape[1]))
    vb1, vb1w, g01 = beta_noise(c1, M, max_lag, p)
    f1 = getattr(beta_noise, "last_factor", np.ones(M.shape[1]))
    rep = energies2(m0, m1, R, vb_cf, vb_cf)
    white = energies2(m0, m1, R, vb0w, vb1w)
    emp = energies2(m0, m1, R, vb0, vb1)
    with np.errstate(divide="ignore", invalid="ignore"):
        colour0 = np.nanmedian(vb0 / vb0w)
        colour1 = np.nanmedian(vb1 / vb1w)
    comps = [{"component": j + 1, "eta_reported": round(float(rep["eta"][j]), 4),
              "eta_white_empirical": round(float(white["eta"][j]), 4), "eta_empirical": round(float(emp["eta"][j]), 4)}
             for j in range(M.shape[1])]
    out = {"k": int(M.shape[1]), "max_lag": int(max_lag), "n_nuisance": p,
           "noise_level_c0": round(float(np.nanmedian(g00)), 4), "noise_level_c1": round(float(np.nanmedian(g01)), 4),
           "colour_factor_c0": round(float(colour0), 4), "colour_factor_c1": round(float(colour1), 4),
           "calibration_factor_c0_median": round(float(np.median(f0)), 4),
           "calibration_factor_c1_median": round(float(np.median(f1)), 4),
           "eta_reported_median": round(float(np.nanmedian(rep["eta"])), 4),
           "eta_white_empirical_median": round(float(np.nanmedian(white["eta"])), 4),
           "eta_empirical_median": round(float(np.nanmedian(emp["eta"])), 4), "components": comps}
    from . import io
    io.write_json(kd / "eta_recheck.json", out)
    return out


def run(phase5_root, phase2_root, ks=None, max_lag: int = 40, sweep_name: str = "phase4_sweep_mc",
        frontend_name: str = "mc") -> list[dict]:
    """Every run (and every k folder, or only `ks`) under phase5_root; front ends from phase2_root/<run>/mc."""
    rows = []
    for run_dir in sorted(p for p in Path(phase5_root).iterdir() if p.is_dir()):
        fe = Path(phase2_root) / run_dir.name / frontend_name
        if not (fe / "c0.npy").exists():
            print(f"{run_dir.name}: no front end at {fe}; skipped")
            continue
        for kd in sorted((run_dir / sweep_name).glob("k*"), key=lambda p: int(p.name[1:])):
            if ks and int(kd.name[1:]) not in ks:
                continue
            if not (kd / "m0_cf.npy").exists():
                print(f"{run_dir.name} {kd.name}: no cross-fitted maps; skipped")
                continue
            r = recheck_k(fe, kd, max_lag)
            rows.append({"run": run_dir.name, **{k: v for k, v in r.items() if k != "components"}})
            print(f"{run_dir.name} k{r['k']}: eta reported {r['eta_reported_median']:.3f}, white {r['eta_white_empirical_median']:.3f}, "
                  f"empirical {r['eta_empirical_median']:.3f}; noise level c0 {r['noise_level_c0']:.2f}, c1 {r['noise_level_c1']:.2f}; "
                  f"colour c0 {r['colour_factor_c0']:.2f}, c1 {r['colour_factor_c1']:.2f}", flush=True)
    return rows
