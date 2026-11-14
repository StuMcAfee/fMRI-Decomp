"""What is the dS0 in the components? Three post hoc checks of the joint-ICA sweep (`mejoint te-check`; user
decision 2026-11-06), motivated by eta ~0.4 in every component of every run (decisions.md):

A. Sign and ratio. In voxels where both the cross-fitted dS0 (s) and dR2* (q) exceed their empirical noise:
   the fraction with s in phase with q (S0 and R2* change together: "sign-opposed" in the signal, which then
   changes sign between echoes at TE* = s/q) against a pure-dR2* null, and the median ratio s/q in ms.
B. Curvature. The component's energy in the residual channels (r0: curvature, the part of TE^2 outside
   span{1, TE}; r1: odd/even alternation), debiased with empirical noise, as a fraction of its channel energy,
   with a z-score against noise; and the voxelwise correlation of the curvature beta with the S0-channel beta.
   A TE-independent dS0 (inflow) has no curvature; a change in the volume of a compartment with a different
   T2* (e.g. blood) has curvature in a fixed relation to its apparent dS0.
C. Echo profile. The component's signal change at each echo (model-free: the four channel betas mapped back
   to the echoes) in its 200 strongest voxels, sign-aligned so that dR2* > 0; the fraction of those voxels
   where TE1 and TE4 change in opposite directions; the TE where the median profile crosses zero; and,
   exploratory (four echoes, one residual degree of freedom), the T2* of a compartment whose volume change
   best explains the profile: y_e = a - TE_e b + c (exp(-TE_e (1/T2X - 1/T2t)) - 1).

Conventions: r0 is sign-fixed so that its basis vector has a positive TE^2 coefficient (R22 > 0; the front end
leaves that sign to the QR routine); positive curvature = convex decay change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

T2X_GRID = (8, 10, 12, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200, 300, 500)


def design(te_ms) -> np.ndarray:
    te = np.asarray(te_ms, float)
    return np.column_stack([-te, np.ones(te.size), te ** 2, (-1.0) ** np.arange(te.size)])


def basis(mu, sigma, ratios, te_ms):
    """Per-voxel whitening weights m (V, E), Q (V, E, E) and R (V, E, E) as in frontend.project (4-echo voxels),
    with the sign of the first two columns fixed (diag R > 0) as the front end does, and the curvature column
    fixed to R22 > 0."""
    m = (np.asarray(mu, float) / np.asarray(ratios, float)[None, :]) / np.asarray(sigma, float)[:, None]
    Q, R = np.linalg.qr(m[:, :, None] * design(te_ms)[None])
    sgn = np.sign(np.diagonal(R, axis1=1, axis2=2)[:, :3])
    sgn[sgn == 0] = 1
    Q[:, :, :3] *= sgn[:, None, :]
    R[:, :3, :] *= sgn[:, :, None]
    return m, Q, R


def ols_betas(x, M):
    Mc = M - M.mean(0)
    P = Mc @ np.linalg.inv(Mc.T @ Mc)
    xx = np.asarray(x, np.float64)
    ok = np.isfinite(xx).all(1)
    b = np.full((xx.shape[0], M.shape[1]), np.nan)
    b[ok] = (xx[ok] - xx[ok].mean(1, keepdims=True)) @ P
    return b


def exchange_fit(prof, te_ms, t2t):
    """Best compartment T2* for a 4-echo profile (exploratory). Returns (best T2X, residual of the linear model,
    residual at the best T2X, median residual over the grid)."""
    te = np.asarray(te_ms, float)
    A = np.column_stack([np.ones(te.size), -te])
    lin = np.linalg.lstsq(A, prof, rcond=None)
    r_lin = float(np.sum((prof - A @ lin[0]) ** 2))
    res = []
    for t2x in T2X_GRID:
        d = 1.0 / t2x - 1.0 / t2t
        if abs(d) < 1e-3:                                               # collinear with the linear model
            res.append(np.nan)
            continue
        g = np.exp(-te * d) - 1.0
        X = np.column_stack([A, g])
        cf = np.linalg.lstsq(X, prof, rcond=None)[0]
        res.append(float(np.sum((prof - X @ cf) ** 2)))
    res = np.array(res)
    if not np.isfinite(res).any():
        return np.nan, r_lin, np.nan, np.nan
    i = int(np.nanargmin(res))
    return float(T2X_GRID[i]), r_lin, float(res[i]), float(np.nanmedian(res))


def check(ch: dict, R2: np.ndarray, mu, sigma, ratios, use, t2s, te_ms, M, m0_cf, m1_cf, max_lag=40,
          n_nuisance=0, z=3.0, n_top=200, n_draws=20, rng=None) -> dict:
    """Arrays version. ch: {"c0", "c1", "r0", "r1"} (V, T); R2: the front end's (V, 2, 2) factors."""
    from .eta_recheck import beta_noise
    rng = np.random.default_rng(0) if rng is None else rng
    te = np.asarray(te_ms, float)
    V, k = m0_cf.shape
    four = np.asarray(use, bool).all(1) & np.isfinite(ch["r1"]).all(1)
    # betas and empirical noise of every channel (full-data time courses)
    B, VB, SE = {}, {}, {}
    for name in ("c0", "c1", "r0", "r1"):
        B[name] = ols_betas(ch[name], M)
        VB[name], _, _ = beta_noise(ch[name], M, max_lag, n_nuisance)
        SE[name] = getattr(beta_noise, "last_factor_rel_se", np.zeros(M.shape[1]))
    # recomputed basis (4-echo voxels), checked against the saved factors, curvature sign fixed
    idx = np.flatnonzero(four)
    m, Q, Rf = basis(np.asarray(mu)[idx], np.asarray(sigma)[idx], ratios, te)
    rmatch = float(np.nanmax(np.abs(Rf[:, :2, :2] - R2[idx]) / np.maximum(np.abs(R2[idx]), 1e-9))) if idx.size else np.nan
    # the front end fixed only the first two column signs; its r0 basis vector is the QR routine's, ours has
    # R22 > 0: the saved r0 times sign(raw R22) is r0 in our convention
    Rraw = np.linalg.qr(m[:, :, None] * design(te)[None])[1]
    r0_sign = np.ones(V)
    r0_sign[idx] = np.sign(Rraw[:, 2, 2])                               # multiply the saved r0 by this to fix R22 > 0
    r0_sign[r0_sign == 0] = 1
    b_r0 = B["r0"] * r0_sign[:, None]
    # --- A: sign and ratio (cross-fitted maps, empirical noise)
    R00, R01, R11 = R2[:, 0, 0][:, None], R2[:, 0, 1][:, None], R2[:, 1, 1][:, None]
    vb0, vb1 = VB["c0"], VB["c1"]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = m1_cf / R11
        q = (m0_cf - R01 * s) / R00 * 1000.0                            # s^-1
        sd_s = np.sqrt(vb1) / R11
        sd_q = np.sqrt(vb0 + vb1 * R01 ** 2 / R11 ** 2) / R00 * 1000.0
    sig = np.isfinite(s) & np.isfinite(q) & (np.abs(s) > z * sd_s) & (np.abs(q) > z * sd_q)
    opp = s * q > 0                                                     # S0 and R2* change together
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_ms = s / q * 1000.0                                       # dS0/S0 per s^-1, in ms; = TE* if > 0
    # null: pure dR2* (each voxel's own q) plus noise
    n_ok, n_op = np.zeros(k), np.zeros(k)
    for _ in range(int(n_draws)):
        n0 = rng.standard_normal((V, k)) * np.sqrt(np.nan_to_num(vb0))
        n1 = rng.standard_normal((V, k)) * np.sqrt(np.nan_to_num(vb1))
        mm0, mm1 = R00 * np.nan_to_num(q) / 1000.0 + n0, n1
        with np.errstate(divide="ignore", invalid="ignore"):
            s_ = mm1 / R11
            q_ = (mm0 - R01 * s_) / R00 * 1000.0
        ok_ = np.isfinite(s_) & np.isfinite(q_) & (np.abs(s_) > z * sd_s) & (np.abs(q_) > z * sd_q)
        n_ok += ok_.sum(0)
        n_op += (ok_ & (s_ * q_ > 0)).sum(0)
    # --- B: curvature energy
    def en(name, b=None):
        bb = B[name] if b is None else b
        v = VB[name]
        okk = np.isfinite(bb) & np.isfinite(v)
        e = np.where(okk, bb ** 2 - v, 0.0).sum(0)
        # noise of the sum: each voxel's chi-square spread plus the calibration's shared relative error
        sd = np.sqrt(np.where(okk, 2 * v ** 2, 0.0).sum(0) + (SE[name] * np.where(okk, v, 0.0).sum(0)) ** 2)
        return e, sd
    e_c0, _ = en("c0")
    e_c1, _ = en("c1")
    e_r0, sd_r0 = en("r0", b_r0)
    e_r1, sd_r1 = en("r1")
    tot = e_c0 + e_c1 + e_r0 + e_r1
    # --- C: echo profiles in the strongest voxels (4 echoes)
    bq_full = np.stack([B["c0"][idx], B["c1"][idx], b_r0[idx], B["r1"][idx]], axis=1)      # (v, 4, k)
    # back to the echoes: y = (Q c) / m with the curvature column in our sign convention
    yb = np.einsum("veq,vqk->vek", Q, np.nan_to_num(bq_full)) / m[:, :, None]                    # (v, E, k)
    t2s_i = np.asarray(t2s)[idx]
    comps = []
    for j in range(k):
        sj, qj = s[:, j], q[:, j]
        sel = sig[:, j]
        n_sig = int(sel.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            zq = np.abs(q[idx, j]) / sd_q[idx, j]
        top = np.argsort(-np.nan_to_num(zq))[: int(n_top)]
        prof = yb[top, :, j] * np.sign(np.nan_to_num(q[idx[top], j]))[:, None]   # aligned: dR2* > 0
        med = np.nanmedian(prof, axis=0)
        cross = np.nan
        for e in range(te.size - 1):                                    # first sign change of the median profile
            if med[e] == 0 or med[e] * med[e + 1] < 0:
                cross = float(te[e] + (te[e + 1] - te[e]) * med[e] / (med[e] - med[e + 1])) if med[e] != med[e + 1] else float(te[e])
                break
        t2x, r_lin, r_best, r_med = exchange_fit(med, te, float(np.nanmedian(t2s_i[top])) if top.size else 45.0)
        okc = sel & np.isfinite(b_r0[:, j])
        cc = float(np.corrcoef(b_r0[okc, j], B["c1"][okc, j])[0, 1]) if okc.sum() >= 20 else np.nan
        comps.append({
            "component": j + 1,
            "n_sig": n_sig, "null_n_sig": round(float(n_ok[j] / n_draws), 1),
            "fraction_opposed": round(float(opp[sel, j].mean()), 4) if n_sig >= 20 else None,
            "null_fraction_opposed": round(float(n_op[j] / n_ok[j]), 4) if n_ok[j] >= 20 else None,
            "ratio_ms_median": round(float(np.nanmedian(alpha_ms[sel, j])), 2) if n_sig >= 20 else None,
            "curvature_fraction": round(float(e_r0[j] / tot[j]), 6) if tot[j] > 0 else None,
            "curvature_z": round(float(e_r0[j] / sd_r0[j]), 2) if sd_r0[j] > 0 else None,
            "alternating_fraction": round(float(e_r1[j] / tot[j]), 6) if tot[j] > 0 else None,
            "alternating_z": round(float(e_r1[j] / sd_r1[j]), 2) if sd_r1[j] > 0 else None,
            "curvature_vs_s0_corr": round(cc, 3) if np.isfinite(cc) else None,
            "profile_pct": [round(float(x) * 100, 4) for x in med],
            "fraction_te1_te4_opposite": round(float(np.mean(prof[:, 0] * prof[:, -1] < 0)), 3) if top.size else None,
            "profile_zero_crossing_ms": round(cross, 1) if np.isfinite(cross) else None,
            "exchange_t2x_ms": t2x, "exchange_resid_linear": r_lin, "exchange_resid_best": r_best,
            "exchange_resid_grid_median": r_med})
    return {"k": int(k), "n_voxels_4echo": int(four.sum()), "R_rebuild_max_rel_error": rmatch, "components": comps}


def check_dir(frontend_dir, k_dir, **kw) -> dict:
    import nibabel as nib
    fe, kd = Path(frontend_dir), Path(k_dir)
    mask3d = np.asarray(nib.load(str(fe / "analysis_mask.nii.gz")).dataobj) > 0
    mapv = lambda n: np.asarray(nib.load(str(fe / f"{n}.nii.gz")).dataobj, np.float64)[mask3d]
    fs = json.loads((fe / "summary.json").read_text())
    ch = {n: np.load(fe / f"{n}.npy") for n in ("c0", "c1", "r0", "r1")}
    p = int(np.load(fe / "nuisance_Q.npy").shape[1]) if (fe / "nuisance_Q.npy").exists() else 0
    tcs = np.genfromtxt(kd / "time_courses.tsv", delimiter="\t", names=True)
    M = np.column_stack([tcs[n] for n in tcs.dtype.names]).astype(np.float64)
    out = check(ch, np.load(fe / "R.npy"), mapv("mu"), mapv("sigma"), fs["echo_noise_ratios_used"], mapv("usable") > 0.5,
                mapv("t2s_ms"), fs["te_ms"], M, np.load(kd / "m0_cf.npy").astype(np.float64),
                np.load(kd / "m1_cf.npy").astype(np.float64), n_nuisance=p, **kw)
    from . import io
    io.write_json(kd / "te_check.json", out)
    figure(out, fs["te_ms"], kd / "te_profiles.png", kd)
    return out


def figure(out, te_ms, path, k_dir=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    comps = out["components"]
    order = list(range(len(comps)))
    if k_dir is not None and (Path(k_dir) / "components.json").exists():
        cj = json.loads((Path(k_dir) / "components.json").read_text())["components"]
        en = [float(c.get("energy_single") or 0) for c in cj]
        order = list(np.argsort(en)[::-1])
    te = np.asarray(te_ms, float)
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    for ax, j in zip(axs.ravel(), order[:8]):
        c = comps[j]
        prof = np.array(c["profile_pct"])
        ax.plot(te, prof, "o-", color="#2a78d6", label="component")
        scale = prof[-1] / te[-1] if te[-1] else 0
        ax.plot(te, scale * te, "--", color="#52514e", lw=1, label="pure dR2* (through TE4)")
        ax.axhline(0, color="#e4e3df")
        ax.set_title(f"comp {c['component']}: opposed {c['fraction_opposed']}, curv z {c['curvature_z']}", fontsize=8.5, loc="left")
    axs[0, 0].legend(fontsize=7, frameon=False)
    for ax in axs[1]:
        ax.set_xlabel("TE (ms)")
    for ax in axs[:, 0]:
        ax.set_ylabel("signal change, % (dR2* > 0)")
    where = ""
    if k_dir is not None:
        kd = Path(k_dir)
        where = f"{kd.parent.parent.name}, {kd.name}: "
    fig.suptitle(f"{where}echo profiles of the strongest components (median of the 200 strongest voxels each)", x=0.01,
                 ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(phase5_root, phase2_root, ks=(50,), sweep_name="phase4_sweep_mc", frontend_name="mc") -> list[dict]:
    rows = []
    for run_dir in sorted(p for p in Path(phase5_root).iterdir() if p.is_dir()):
        fe = Path(phase2_root) / run_dir.name / frontend_name
        if not (fe / "r0.npy").exists():
            print(f"{run_dir.name}: no front end with residual channels at {fe}; skipped")
            continue
        for k in ks:
            kd = run_dir / sweep_name / f"k{k}"
            if not (kd / "m0_cf.npy").exists():
                print(f"{run_dir.name} k{k}: no sweep folder; skipped")
                continue
            r = check_dir(fe, kd)
            cs = r["components"]
            med = lambda key: np.nanmedian([np.nan if c[key] is None else c[key] for c in cs])
            print(f"{run_dir.name} k{k}: opposed {med('fraction_opposed'):.2f} (null {med('null_fraction_opposed'):.2f}), "
                  f"ratio {med('ratio_ms_median'):.1f} ms, curvature z {med('curvature_z'):.1f}, "
                  f"TE1/TE4 opposite {med('fraction_te1_te4_opposite'):.2f}, exchange T2* {med('exchange_t2x_ms'):.0f} ms; "
                  f"R rebuild error {r['R_rebuild_max_rel_error']:.1e}", flush=True)
            rows.append({"run": run_dir.name, "k": k})
    return rows
