"""Phase 4 per-component measurements (CLAUDE.md 7.1): classification becomes measurement.

eta, the noise-debiased S0 energy fraction, from the channel maps (m0, m1) and the per-voxel QR factors:
  s = m1 / R11,  R00 q = m0 - R01 m1 / R11
  E_S0 = sum_v (R01^2 + R11^2) s_v^2 - vb sum_v (R01^2 + R11^2) / R11^2
  E_R2 = sum_v (R00 q_v)^2           - vb sum_v (1 + R01^2 / R11^2)
  eta  = E_S0 / (E_S0 + E_R2)
vb is the beta noise variance of the component for unit-noise channels: diag((M^T M)^-1) of the joint
GLM (= 1 / ||t||^2 for a single regressor). Voxels with c1 undefined are excluded.

Significance: eta is interpreted only when the total debiased energy exceeds its null. Components are the
top-k variance directions, so random time courses are not a fair null (on pure noise every component
exceeded it ~10x). The null goes through the same selection: rows phase-randomized independently
(shared structure destroyed, each row's spectrum kept; mirror-extended), projected and demeaned like the
data, and their top-k temporal PCA directions measured. Significance is family-wise: a component must
exceed the chosen quantile of the LARGEST surrogate energy among k directions (pure noise: no component
above it in 95 % of datasets). Single random Gaussian time courses are still reported for reference.

Selection bias: CLAUDE.md's debiasing subtracts the noise expected for a FIXED time course, but component
time courses are chosen from the data's own top-variance directions (simulation: a pure dR2* source showed
E_S0 = 8.9 of 98 with no dS0 in the data; subtracting the surrogates' mean bias over-corrected strong
components, eta 4.0). eta therefore uses CROSS-FITTED maps (decompose.py): each voxel's betas come from a
time course estimated from the other half of the voxels, so its noise did not choose the regressor and
the fixed-regressor debiasing is exact.
Reference bands at matched component SNR: for each component, pure-dR2*, pure-dS0 and noise-only maps with
the component's footprint, voxel R factors and debiased energy, plus beta noise of variance vb.
"""
from __future__ import annotations

import numpy as np


def _R(R):
    return R[:, 0, 0][:, None], R[:, 0, 1][:, None], R[:, 1, 1][:, None]


def energies(m0: np.ndarray, m1: np.ndarray, R: np.ndarray, vb) -> dict:
    """E_S0, E_R2, eta per component (columns of m0, m1). vb: scalar or (k,) beta noise variance."""
    R00, R01, R11 = _R(R)
    ok = np.isfinite(m0) & np.isfinite(m1) & (R11 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ws = (R01 ** 2 + R11 ** 2) / R11 ** 2                         # E_S0 weight on m1^2
        wr = 1 + R01 ** 2 / R11 ** 2                                   # noise weight of R00 q
        r2 = m0 - R01 * m1 / R11                                       # = R00 q
    z = lambda a: np.where(ok, a, 0.0)
    vb = np.asarray(vb, float)
    vb = vb if vb.ndim == 2 else np.broadcast_to(vb.reshape(1, -1) if vb.ndim == 1 else vb, m0.shape)   # (V, k)
    e_s0 = z(ws * m1 ** 2).sum(0) - z(vb * ws).sum(0)
    e_r2 = z(r2 ** 2).sum(0) - z(vb * wr).sum(0)
    tot = e_s0 + e_r2
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(tot != 0, e_s0 / tot, np.nan)
    return {"E_S0": e_s0, "E_R2": e_r2, "total": tot, "eta": eta, "n_voxels": ok.sum(0)}


def _single_regression_maps(c0, c1, Tc):
    """Maps of c0, c1 on each column of Tc (T, n) separately (unit-SD, demeaned columns: beta = X t / T)."""
    T = Tc.shape[0]
    f = lambda c: np.where(np.isfinite(c).all(1, keepdims=True), np.nan_to_num(c.astype(np.float64)) @ Tc / T, np.nan)
    return f(c0), f(c1), 1.0 / T


def _unit(Tc, Q=None):
    Tc = Tc - Tc.mean(0)
    if Q is not None:
        Tc = Tc - Q @ (Q.T @ Tc)
    return Tc / Tc.std(0)


def energy_null(c0, c1, R, M, n_null: int, rng, Q=None) -> dict:
    """Debiased total energy for random Gaussian and phase-randomized component time courses."""
    T, k = M.shape
    G = _unit(rng.standard_normal((T, int(n_null))), Q)
    F = np.fft.rfft(M, axis=0)
    reps = max(1, int(np.ceil(n_null / k)))
    P = []
    for _ in range(reps):
        ph = rng.uniform(0, 2 * np.pi, F.shape)
        ph[0] = 0
        P.append(np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=T, axis=0))
    P = _unit(np.concatenate(P, axis=1)[:, :int(n_null)], Q)
    out = {}
    for name, Tc in (("gaussian", G), ("phase_randomized", P)):
        a0, a1, vb = _single_regression_maps(c0, c1, Tc)
        out[name] = energies(a0, a1, R, vb)["total"]
    return out


def energy_null_pipeline(c0, c1, R, k: int, n_sur: int, rng, Q=None, M=None) -> dict:
    """Debiased energies (E_S0, E_R2, total; arrays n_sur x k) of the top-k temporal PCA directions of
    phase-randomized surrogates. With M (the component time courses), the surrogates are built from the
    data with the components removed (noise only: phase randomization of rows that still hold strong
    components inherits their power and inflates the null; simulation: threshold 154 vs a 148-energy
    dS0 source); the component directions are projected out of each surrogate and its variance matched
    to the residual."""
    from .rank import phase_randomize_rows, spectrum

    V = c0.shape[0]
    A = np.concatenate([c0, c1], axis=0)
    ok = np.isfinite(A).all(1)
    X = A[ok].astype(np.float64)
    X -= X.mean(1, keepdims=True)
    Um = None
    if M is not None:
        Um = np.linalg.qr(M - M.mean(0))[0]
        X = X - (X @ Um) @ Um.T                                         # components removed: noise only
    tot = float((X ** 2).sum())
    out = []
    for _ in range(int(n_sur)):
        Z = phase_randomize_rows(X, rng)
        if Q is not None:
            Z = Z - (Z @ Q) @ Q.T
        if Um is not None:
            Z = Z - (Z @ Um) @ Um.T
        Z -= Z.mean(1, keepdims=True)
        Z *= np.sqrt(tot / (Z ** 2).sum())
        _, U = spectrum(Z, vectors=True)
        Tc = _unit(U[:, :int(k)], Q)
        Zf = np.full(A.shape, np.nan)
        Zf[ok] = Z
        a0, a1, vb = _single_regression_maps(Zf[:V], Zf[V:], Tc)
        out.append(energies(a0, a1, R, vb))
    return {key: np.array([o[key] for o in out]) for key in ("E_S0", "E_R2", "total")}


def component_single_energy(c0, c1, R, M) -> np.ndarray:
    a0, a1, vb = _single_regression_maps(c0, c1, (M - M.mean(0)) / M.std(0))
    return energies(a0, a1, R, vb)["total"]


def reference_bands(m0, m1, R, vb, E, rng, n_mc: int = 100, q=(0.025, 0.975)) -> list[dict]:
    """Per component: eta intervals for pure-dR2*, pure-dS0 and noise-only maps at the component's
    footprint, R factors and debiased energy E, with beta noise of variance vb (matched SNR)."""
    R00, R01, R11 = _R(R)
    ok = np.isfinite(m0) & np.isfinite(m1) & (R11 > 0)
    out = []
    vb = np.asarray(vb, float)
    vb = np.nanmedian(np.where(ok, vb, np.nan), axis=0) if vb.ndim == 2 else np.broadcast_to(vb, (m0.shape[1],))
    for j in range(m0.shape[1]):
        v = ok[:, j]
        r00, r01, r11 = R00[v, 0], R01[v, 0], R11[v, 0]
        if E[j] <= 0 or v.sum() < 10:
            out.append({"R2": [np.nan, np.nan], "S0": [np.nan, np.nan], "noise": [np.nan, np.nan]})
            continue
        r2 = (m0[v, j] - r01 * m1[v, j] / r11)
        sm = m1[v, j] / r11
        # footprints scaled to energy E: pure R2*: R00 q = a * r2 (energy a^2 |r2|^2); pure S0: s = b * sm
        a = np.sqrt(E[j] / max((r2 ** 2).sum(), 1e-300))
        b = np.sqrt(E[j] / max(((r01 ** 2 + r11 ** 2) * sm ** 2).sum(), 1e-300))
        sd = np.sqrt(vb[j])
        etas = {"R2": [], "S0": [], "noise": []}
        for _ in range(int(n_mc)):
            n0 = sd * rng.standard_normal(v.sum())
            n1 = sd * rng.standard_normal(v.sum())
            for name, (x0, x1) in (("R2", (a * r2, 0 * r2)), ("S0", (r01 * b * sm, r11 * b * sm)), ("noise", (0 * r2, 0 * r2))):
                e = energies((x0 + n0)[:, None], (x1 + n1)[:, None], np.stack([np.stack([r00, r01], 1),
                             np.stack([0 * r00, r11], 1)], 1), vb[j])
                etas[name].append(float(e["eta"][0]))
        out.append({k2: [float(np.nanquantile(v2, q[0])), float(np.nanquantile(v2, q[1]))] for k2, v2 in etas.items()})
    return out


def label(eta, significant, band, margin: float = 0.0) -> str:
    """Measurement, not a noise label (CLAUDE.md 7.1: do not hard-label dS0 as noise). The matched-SNR
    bands hold thermal noise only; `margin` adds the decomposition cross-talk measured in simulation."""
    if not significant:
        return "not above null"
    if not np.isfinite(eta) or np.isnan(band["R2"][1]):
        return "undetermined"
    if eta <= band["R2"][1] + margin:
        return "R2*-dominant"
    if eta >= band["S0"][0] - margin:
        return "S0-dominant"
    return "mixed"


def debiased_norms(m0, m1, R, vb) -> tuple[np.ndarray, np.ndarray]:
    """||s|| and ||q|| (s^-1) with the expected noise removed: sqrt(max(sum x^2 - sum var(x), 0)).
    var(s_v) = vb / R11^2; var(q_v) = vb (1 + R01^2 / R11^2) / R00^2 (x 1e6 for s^-1).
    A plain norm of a noisy map is biased upward, and resampling frames adds noise, so undebiased
    bootstrap intervals sat above their own estimate on real data (2026-11-04)."""
    from .decompose import backsub

    R00, R01, R11 = _R(R)
    s, q = backsub(m0, m1, R)
    vb = np.asarray(vb, float)
    vb = vb if vb.ndim == 2 else np.broadcast_to(vb.reshape(1, -1), m0.shape)
    ok = np.isfinite(s) & np.isfinite(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_s = vb / R11 ** 2
        var_q = vb * (1 + R01 ** 2 / R11 ** 2) / R00 ** 2 * 1e6
    z = lambda a: np.where(ok, a, 0.0)
    ns = np.sqrt(np.maximum(z(s ** 2 - var_s).sum(0), 0))
    nq = np.sqrt(np.maximum(z(q ** 2 - var_q).sum(0), 0))
    return ns, nq


def bootstrap_norms(c0, c1, R, dec: dict, n_boot: int, block: int, rng) -> dict:
    """Debiased ||s|| and ||q|| (s^-1) of the cross-fitted maps, with block-bootstrap (frames)
    intervals. Each half's voxels are refitted on the other half's time course with frame weights w
    (how often each frame was drawn); beta noise variance diag(G^-1 M^T W^2 M G^-1), G = M^T W M."""
    cf = dec["crossfit"]
    V = c0.shape[0]
    X0 = np.nan_to_num(c0.astype(np.float64))
    X1 = np.nan_to_num(c1.astype(np.float64))
    bad0 = ~np.isfinite(c0).all(1)
    bad1 = ~np.isfinite(c1).all(1)
    halves = [(~cf["half_vox"], cf["M_from_A"]), (cf["half_vox"], cf["M_from_B"])]
    T, k = cf["M_from_A"].shape
    nb = int(np.ceil(T / block))
    NS, NQ = [], []
    for _ in range(int(n_boot)):
        w = np.zeros(T)
        for s0 in rng.integers(0, T - block + 1, nb):
            w[s0:s0 + block] += 1
        m0 = np.full((V, k), np.nan)
        m1 = np.full((V, k), np.nan)
        vb = np.full((V, k), np.nan)
        for use, Mh in halves:
            Gi = np.linalg.inv(Mh.T @ (Mh * w[:, None]))
            H = Gi @ (Mh.T @ (Mh * (w ** 2)[:, None])) @ Gi
            P = (Mh * w[:, None]) @ Gi
            m0[use], m1[use] = X0[use] @ P, X1[use] @ P
            vb[use] = np.diag(H)[None, :]
        m0[bad0] = np.nan
        m1[bad1] = np.nan
        ns, nq = debiased_norms(m0, m1, R, vb)
        NS.append(ns)
        NQ.append(nq)
    NS, NQ = np.array(NS), np.array(NQ)
    return {"s_ci": np.percentile(NS, [2.5, 97.5], axis=0), "q_ci": np.percentile(NQ, [2.5, 97.5], axis=0)}


def tissue_summary(m0, m1, vb, R, classes: np.ndarray, kappa: np.ndarray, q_full=None, q_full_sd=None,
                   z_min: float = 2.5, min_s0_share: float = 0.05, consistent_margin: float = 0.10,
                   f_long=None, f_long_max: float = 0.9) -> list[dict]:
    """Per component (cross-fitted maps): eta and energy share within each data-only class, and the
    partial-volume diagnostic (option c). Across tissue, mixed and CSF-dominated voxels with a long-T2*
    fraction <= f_long_max (kappa diverges only near pure CSF; at fraction 0.7 the prediction matched the
    observed dS0 within 2 % in simulation, so excluding CSF-dominated voxels lost the evidence), and where
    the component is present (|q| of the
    full-data map > z_min noise SDs; over the whole brain the noise of ~10^4 empty voxels dilutes beta
    to ~0.1 even when partial volume explains the component), the dS0 map s is regressed on the map that
    partial volume predicts, x = kappa q, with S0-energy weights ws = R01^2 + R11^2:
        two explanations of the dS0 energy are compared in those voxels: the partial-volume prediction
    s = kappa q (no free parameter) and the best constant ratio s = alpha q. pv_explained close to
    ratio_explained: the dS0 is consistent with partial volume; pv_explained much lower (or negative): it
    is not (e.g. a physiological dS0 with its own sign and size). A single free regression on kappa q could
    not tell them apart (mixed source: beta -1.9), and a two-regressor fit was unstable because kappa varies
    little within one footprint. eta_pv_removed = eta after subtracting exactly kappa q in those voxels. s = m1 / R11 and q = (m0 - R01 s) / R00 share channel noise:
    var(s) = vb / R11^2, var(q) = vb (1 + R01^2 / R11^2) / R00^2, cov(s, q) = -R01 vb / (R11^2 R00)."""
    from .tissue import CLASS_NAMES

    R00, R01, R11 = R[:, 0, 0][:, None], R[:, 0, 1][:, None], R[:, 1, 1][:, None]
    vb = np.asarray(vb, float)
    vb = vb if vb.ndim == 2 else np.broadcast_to(vb.reshape(1, -1), m0.shape)
    k_ms = (kappa * 1000.0)[:, None]                                    # per unit dR2* in 1/ms
    base = np.isfinite(m0) & np.isfinite(m1) & (R11 > 0)
    sel = base & np.isfinite(k_ms) & np.isin(classes, [1, 2, 3])[:, None]
    if f_long is not None:                                   # kappa diverges only near pure CSF
        sel &= (np.nan_to_num(f_long, nan=1.0) <= f_long_max)[:, None]
    if q_full is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            sel &= np.abs(q_full) > z_min * q_full_sd
    with np.errstate(divide="ignore", invalid="ignore"):
        s = m1 / R11
        q = (m0 - R01 * s) / R00                                        # 1/ms
        var_s = vb / R11 ** 2
        var_q = vb * (1 + R01 ** 2 / R11 ** 2) / R00 ** 2
        cov = -R01 * vb / (R11 ** 2 * R00)
        ws = R01 ** 2 + R11 ** 2
        x = k_ms * q
    z = lambda a, m: np.where(m, a, 0.0)
    e_r2 = z((R00 * q) ** 2 - vb * (1 + R01 ** 2 / R11 ** 2), base).sum(0)
    e_s0 = z(ws * (s ** 2 - var_s), base).sum(0)
    # "eta if the predicted partial-volume dS0 were removed": s - kappa q in the selected voxels
    with np.errstate(invalid="ignore"):
        s_res = np.where(sel, s - x, s)
        var_res = np.where(sel, var_s + k_ms ** 2 * var_q - 2 * k_ms * cov, var_s)
    e_s0_res = z(ws * (s_res ** 2 - var_res), base).sum(0)
    # two one-line explanations of the dS0 in the selected voxels (noise-debiased energies):
    #   partial volume, s = kappa q (no free parameter), vs a free constant ratio, s = alpha q
    e_none = z(ws * (s ** 2 - var_s), sel).sum(0)
    e_r2_sel = z((R00 * q) ** 2 - vb * (1 + R01 ** 2 / R11 ** 2), sel).sum(0)
    e_pv = z(ws * (s_res ** 2 - var_res), sel).sum(0)
    S11 = z(ws * (q ** 2 - var_q), sel).sum(0)
    b1 = z(ws * (q * s - cov), sel).sum(0)
    n_sel = sel.sum(0)
    out = []
    for j in range(m0.shape[1]):
        tot = e_s0[j] + e_r2[j]
        row = {"pv_n_voxels": int(n_sel[j])}
        share = e_none[j] / (e_none[j] + e_r2_sel[j]) if e_none[j] + e_r2_sel[j] > 0 else 0.0
        row["s0_share_where_present"] = round(float(share), 4)
        if n_sel[j] < 20 or S11[j] <= 0:
            row["pv_verdict"] = "too few voxels with dR2*"
        elif share < min_s0_share:
            row["pv_verdict"] = "no dS0 to explain"
        else:
            alpha = b1[j] / S11[j]
            pv = 1 - e_pv[j] / e_none[j]
            ratio = alpha ** 2 * S11[j] / e_none[j]
            row["pv_explained"] = round(float(pv), 4)                       # partial-volume prediction, no free parameter
            row["ratio_explained"] = round(float(ratio), 4)                 # best constant s/q ratio
            row["ratio_dS0_per_s"] = round(float(alpha) / 1000.0, 6)        # that ratio (dS0/S0 per s^-1)
            row["pv_verdict"] = ("consistent with partial volume" if pv > 0 and pv >= ratio - consistent_margin
                                 else "not partial volume")
        if e_s0_res[j] + e_r2[j] > 0:
            row["eta_pv_removed"] = round(float(e_s0_res[j] / (e_s0_res[j] + e_r2[j])), 4)
        for c, name in CLASS_NAMES.items():
            msk = (classes == c)
            if msk.sum() < 20:
                continue
            en = energies(m0[msk][:, [j]], m1[msk][:, [j]], R[msk], vb[msk][:, [j]])
            row[f"share_{name}"] = round(float(en["total"][0] / tot), 4) if tot > 0 else None
            row[f"eta_{name}"] = round(float(en["eta"][0]), 4) if np.isfinite(en["eta"][0]) else None
        out.append(row)
    return out


def measure(dec: dict, c0, c1, R, cfg: dict, rng, Q=None, frames: dict | None = None, tissue: dict | None = None) -> dict:
    """All per-component measurements for one decomposition."""
    Mc = cfg["metrics"]
    M = dec["time_courses"]
    m0, m1, vb = dec["m0_cf"], dec["m1_cf"], dec["vb_cf"]           # cross-fitted maps for eta and bands
    en = energies(m0, m1, R, vb)
    single = component_single_energy(c0, c1, R, M)
    nul = energy_null(c0, c1, R, M, int(Mc["n_null"]), rng, Q)
    qn = float(Mc["null_quantile"])
    # two passes: (1) conservative null from the data as they are (strong components inflate it);
    # (2) the components above it (certainly signal) are removed before building the final null. On pure
    # noise nothing passes (1), so nothing is removed: removing selected noise directions would deplete
    # the null (simulation: a false positive).
    pipe1 = energy_null_pipeline(c0, c1, R, M.shape[1], int(Mc["n_null_pipeline"]), rng, Q)
    thr1 = float(np.quantile(pipe1["total"].max(axis=1), qn))
    strong = single > thr1
    if strong.any():
        pipe = energy_null_pipeline(c0, c1, R, M.shape[1], int(Mc["n_null_pipeline"]), rng, Q, M=M[:, strong])
    else:
        pipe = pipe1
    thr = float(np.quantile(pipe["total"].max(axis=1), qn))          # family-wise: largest of k per surrogate
    sig = single > thr
    bands = reference_bands(m0, m1, R, vb, en["total"], rng, int(Mc["band_mc"]))
    boot = bootstrap_norms(c0, c1, R, dec, int(Mc["n_boot"]), int(Mc["boot_block_frames"]), rng)
    ns_hat, nq_hat = debiased_norms(m0, m1, R, vb)
    s, qq = dec["s"], dec["q_per_s"]
    rows = []
    for j in range(M.shape[1]):
        row = {"component": j + 1, "iq": round(float(dec["iq"][j]), 4),
               "stability_median_abs_r": round(float(dec["stability_median"][j]), 4),
               "eta": round(float(en["eta"][j]), 4), "E_S0": float(en["E_S0"][j]), "E_R2": float(en["E_R2"][j]),
               "energy_single": float(single[j]), "energy_null_threshold": float(thr), "above_null": bool(sig[j]),
               "band_R2": [round(x, 4) for x in bands[j]["R2"]], "band_S0": [round(x, 4) for x in bands[j]["S0"]],
               "label": label(en["eta"][j], sig[j], bands[j], float(Mc.get("eta_label_margin", 0.0))),
               "norm_s": float(ns_hat[j]), "norm_s_ci": [float(x) for x in boot["s_ci"][:, j]],
               "norm_q_per_s": float(nq_hat[j]), "norm_q_ci": [float(x) for x in boot["q_ci"][:, j]],
               "norm_s_raw": float(np.sqrt(np.nansum(s[:, j] ** 2))), "norm_q_raw_per_s": float(np.sqrt(np.nansum(qq[:, j] ** 2)))}
        if frames:
            t = M[:, j]
            for key, f in frames.items():
                f = np.asarray(f, float)
                if f.size == t.size and np.isfinite(f).all() and f.std() > 0:
                    row[f"corr_{key}"] = round(float(np.corrcoef(t, f)[0, 1]), 4)
                    row[f"corr_power_{key}"] = round(float(np.corrcoef(t ** 2, f)[0, 1]), 4)
        rows.append(row)
    if tissue is not None:
        R00, R01, R11 = R[:, 0, 0][:, None], R[:, 0, 1][:, None], R[:, 1, 1][:, None]
        vbf = np.broadcast_to(np.asarray(dec["beta_noise_var"], float).reshape(1, -1), m0.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            q_sd = np.sqrt(vbf * (1 + R01 ** 2 / R11 ** 2)) / R00 * 1000.0      # SD of the full-data q (s^-1)
        ts = tissue_summary(m0, m1, vb, R, tissue["classes"], tissue["kappa"], q_full=dec["q_per_s"], q_full_sd=q_sd,
                            z_min=float(cfg["metrics"].get("pv_z_min", 2.5)),
                            min_s0_share=float(cfg["metrics"].get("pv_min_s0_share", 0.05)),
                            consistent_margin=float(cfg["metrics"].get("pv_consistent_margin", 0.10)),
                            f_long=tissue.get("f_long"), f_long_max=float(cfg["metrics"].get("pv_f_long_max", 0.9)))
        for row, t in zip(rows, ts):
            row.update(t)
    return {"components": rows, "null": {"threshold": float(thr), "quantile": qn, "family_wise": True,
                                         "threshold_first_pass": thr1, "n_removed_for_null": int(strong.sum()),
                                         "n_surrogates": int(pipe["total"].shape[0]),
                                         "gaussian_p95": float(np.quantile(nul["gaussian"], qn)),
                                         "phase_randomized_p95": float(np.quantile(nul["phase_randomized"], qn))}}
