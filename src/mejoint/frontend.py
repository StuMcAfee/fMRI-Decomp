"""Front end (Phase 2; CLAUDE.md section 5): from 4 magnitude echoes to two noise-whitened physical
channels (c0: OC-like, c1: dS0-only) and a residual noise reference (r).

Steps, in the non-circular order of CLAUDE.md 5.4:
  1. temporal mean mu_e per voxel; voxel-echoes with any frame <= 0 are unusable (log undefined);
  2. sigma_init from the sigma-free residual Q_perp^T (mu o y) with all usable echoes, trusted only
     where mu_min / sigma_init > sigma_init_snr; a smooth sigma map from the trusted voxels;
  3. echo usability: mu_e / sigma_smooth >= snr_min; df = (usable echoes) - 2;
  4. per-voxel sigma from the sigma-free residual with the usable echoes (median over time);
     the smooth map replaces it in low-SNR and df = 0 voxels;
  5. whitening with mu / sigma and a per-voxel QR of the whitened design [-TE, 1, TE^2, alternating],
     with diag(R) > 0 in every voxel (R01 < 0);
  6. the static mono-exponential fit (S0, T2*) on the temporal mean, used for maps and the
     static-misfit (curvature) map, not for the channels.

Sigma is never estimated from the temporal variance of the data (CLAUDE.md 5.4.7).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, stats

def _quiet():
    """Silence 'mean of empty slice' warnings: NaN rows (undefined channels) are expected."""
    import warnings

    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter("ignore", RuntimeWarning)

    class _C:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            ctx.__exit__(*a)
            return False
    return _C()


# ======================================================================= design and projection
def design(te_ms: np.ndarray) -> np.ndarray:
    """Square design for n usable echoes: [-TE, 1, TE^2, alternating][:, :n] (CLAUDE.md 5.3)."""
    te = np.asarray(te_ms, float)
    n = te.size
    cols = [-te, np.ones(n), te ** 2, (-1.0) ** np.arange(n)]
    return np.column_stack(cols[:n])


def project(y: np.ndarray, m: np.ndarray, use: np.ndarray, te_ms) -> dict:
    """Per-voxel QR projection of scaled data, grouped by usable-echo pattern.

    y: (V, E, T) data (log-ratio or linear form); m: (V, E) per-echo scale (mu / sigma to whiten,
    mu alone for the sigma-free residual); use: (V, E) usable echoes.
    Returns c (V, 2, T), r (V, 2, T) [NaN where undefined], R (V, 2, 2), n_use (V,).
    With 1 usable echo c1 is NaN (never zero-filled); with <= 2 there is no residual.
    """
    te_ms = np.asarray(te_ms, float)
    V, E, T = y.shape
    c = np.full((V, 2, T), np.nan, np.float32)
    r = np.full((V, 2, T), np.nan, np.float32)
    R = np.full((V, 2, 2), np.nan)
    n_use = use.sum(1)
    code = (use.astype(np.int64) * (1 << np.arange(E))).sum(1)
    for k in np.unique(code):
        idx = np.flatnonzero(code == k)
        cols = np.flatnonzero((k >> np.arange(E)) & 1)
        n = cols.size
        if n == 0:
            continue
        D = design(te_ms[cols])                                        # (n, n)
        mm = m[np.ix_(idx, cols)].astype(np.float64)                   # (v, n)
        Q, Rm = np.linalg.qr(mm[:, :, None] * D[None])                 # square: complete = reduced
        p = min(n, 2)
        sgn = np.sign(np.diagonal(Rm, axis1=1, axis2=2)[:, :p])
        sgn[sgn == 0] = 1
        Q[:, :, :p] *= sgn[:, None, :]
        Rm[:, :p, :] *= sgn[:, :, None]
        yw = mm[:, :, None] * y[np.ix_(idx, cols)].astype(np.float64)  # (v, n, T)
        proj = np.einsum("vnk,vnt->vkt", Q, yw)
        c[idx, :p] = proj[:, :p]
        if n > 2:
            r[idx, : n - 2] = proj[:, 2:n]
        R[idx, :p, :p] = Rm[:, :p, :p]
    return {"c": c, "r": r, "R": R, "n_use": n_use}


def signal_form(S: np.ndarray, mu: np.ndarray, form: str = "log") -> np.ndarray:
    """y_e(t), zero mean over time in every echo.

    log (default): ln S - mean_t ln S, i.e. ln(S / geometric mean). This equals CLAUDE.md's
    ln(S / mu) up to a constant per echo; that constant (about TE^2 var(b) / 2) is not in
    span{1, TE}, so removing it keeps the channels exactly zero for pure dR2* signals.
    linear: S / mu - 1 with mu the arithmetic mean (already zero mean; tests only).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if form == "log":
            ls = np.log(S)
            return ls - ls.mean(axis=-1, keepdims=True)
        if form == "linear":
            return S / mu[..., None] - 1.0
    raise ValueError("form must be 'log' or 'linear'")


# ======================================================================= sigma
def chi2_median(df) -> np.ndarray:
    """Median of chi-square(df): 2 ln 2 for df = 2, 0.4549 for df = 1 (CLAUDE.md 5.4.4)."""
    return stats.chi2.median(np.asarray(df, float))


def sigma_rel_se(df, n_frames: int) -> np.ndarray:
    """Relative standard error of the median-based sigma: 1 / (4 sqrt(n) f(m) m), m the chi2 median."""
    df = np.asarray(df, float)
    m = stats.chi2.median(df)
    return 1.0 / (4.0 * np.sqrt(n_frames) * stats.chi2.pdf(m, df) * m)


def sigma_from_residual(u: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Per-voxel sigma from the sigma-free residual u = Q_perp^T (mu o y) = sigma * r.

    sigma^2 = median_t(sum_k u_k^2) / median(chi2_df), robust to spikes. u is demeaned over time first
    (ln(S / mean S) has a small constant offset). NaN where df = 0.
    """
    u = u.astype(np.float64)
    with np.errstate(invalid="ignore"), _quiet():
        u = u - np.nanmean(u, axis=2, keepdims=True)
    e = np.nansum(u ** 2, axis=1)                                      # (V, T)
    out = np.full(u.shape[0], np.nan)
    ok = df >= 1
    out[ok] = np.sqrt(np.median(e[ok], axis=1) / chi2_median(df[ok]))
    return out


def smooth_map(values: np.ndarray, weights: np.ndarray, mask3d: np.ndarray, fwhm_vox: float) -> np.ndarray:
    """Normalised-convolution smoothing of a masked map: sum G*(w v) / sum G*(w), evaluated in the mask.
    Fills gaps (df = 0, low SNR) from trusted neighbours; g-factor varies smoothly."""
    s = fwhm_vox / np.sqrt(8 * np.log(2))
    v3 = np.zeros(mask3d.shape)
    w3 = np.zeros(mask3d.shape)
    ok = np.isfinite(values) & (weights > 0)
    v3[mask3d] = np.where(ok, values, 0.0)
    w3[mask3d] = np.where(ok, weights, 0.0)
    num = ndimage.gaussian_filter(v3 * w3, s, mode="constant")
    den = ndimage.gaussian_filter(w3, s, mode="constant")
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-12), np.nan)[mask3d]
    if np.isnan(out).any():                                           # isolated gaps: brain median
        out[np.isnan(out)] = np.nanmedian(values[ok]) if ok.any() else np.nan
    return out


# ======================================================================= static fit
def fit_monoexp(mu: np.ndarray, te_ms, use: np.ndarray, n_iter: int = 15):
    """Non-linear least squares of mu_e = S0 exp(-TE_e / T2*) over the usable echoes, per voxel.

    Equal sigma across echoes makes unweighted signal-domain LS the ML fit. Initialised by a
    log-linear fit weighted by mu^2; Gauss-Newton on (ln S0, R2*). T2* in ms, capped to 1-500 ms.
    Only voxels with >= 2 usable echoes are fitted (with one echo the problem is undetermined and the
    normal equations are singular); the others get NaN.
    """
    te = np.asarray(te_ms, float)
    V, E = mu.shape
    s0 = np.full(V, np.nan)
    t2 = np.full(V, np.nan)
    ok = use.sum(1) >= 2
    if ok.any():
        m, w = mu[ok], np.where(use[ok], 1.0, 0.0)
        lm = np.log(np.clip(m, 1e-12, None))
        # weighted log-linear start: minimise sum w mu^2 (ln mu - (a - TE r))^2
        ww = w * m ** 2
        sw, st, stt = ww.sum(1), (ww * te).sum(1), (ww * te ** 2).sum(1)
        sy, sty = (ww * lm).sum(1), (ww * te * lm).sum(1)
        det = sw * stt - st ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = -(sw * sty - st * sy) / det
            a = (sy + r2 * st) / sw
        r2 = np.clip(np.nan_to_num(r2, nan=1 / 40), 1 / 500, 1.0)
        a = np.nan_to_num(a, nan=0.0)
        a_lo, a_hi = lm.min(1) - 1.0, lm.max(1) + te.max() * 1.0 + 1.0   # plausible ln S0 range
        for _ in range(n_iter):
            f = np.exp(a[:, None] - te[None] * r2[:, None])
            res = (m - f) * w
            j1, j2 = f * w, -te[None] * f * w
            A11, A12, A22 = (j1 * j1).sum(1), (j1 * j2).sum(1), (j2 * j2).sum(1)
            b1, b2 = (j1 * res).sum(1), (j2 * res).sum(1)
            dt = A11 * A22 - A12 ** 2
            good = dt > 1e-12 * A11 * A22                                 # well-conditioned voxels only
            with np.errstate(divide="ignore", invalid="ignore"):
                da = np.where(good, (A22 * b1 - A12 * b2) / dt, 0.0)
                dr = np.where(good, (A11 * b2 - A12 * b1) / dt, 0.0)
            a = np.clip(a + np.clip(da, -1.0, 1.0), a_lo, a_hi)           # bounded steps
            r2 = np.clip(r2 + np.clip(dr, -0.05, 0.05), 1 / 500, 1.0)
        s0[ok] = np.exp(a)
        t2[ok] = 1.0 / r2
    muhat = s0[:, None] * np.exp(-te[None] / t2[:, None])
    return s0, t2, muhat


# ======================================================================= the whole front end
def run(S: np.ndarray, te_ms, cfg: dict, mask3d: np.ndarray | None = None, form: str | None = None,
        chunk: int | None = None, quiet_frames: np.ndarray | None = None) -> dict:
    """Front end on masked magnitude data S (V, E, T). mask3d (X, Y, Z) is needed for the smooth sigma
    map (V must be the voxels of mask3d in C order). Returns maps (V, ...) and channels (V, 2, T)."""
    F = cfg["frontend"]
    form = form or F.get("signal_form", "log")
    chunk = chunk or int(F.get("chunk_voxels", 5000))
    te_ms = np.asarray(te_ms, float)
    V, E, T = S.shape
    if mask3d is None:
        raise ValueError("mask3d is required (smooth sigma map)")

    mu = S.mean(axis=2, dtype=np.float64)                              # (V, E)
    nonpos = (S <= 0).any(axis=2)
    use0 = ~nonpos & (mu > 0)
    if F.get("nonpositive_policy", "exclude_echo") != "exclude_echo":
        raise ValueError("only nonpositive_policy = exclude_echo is implemented")

    def residual_sigma(use):
        sig = np.full(V, np.nan)
        for i0 in range(0, V, chunk):
            sl = slice(i0, min(i0 + chunk, V))
            y = signal_form(S[sl], mu[sl], form)
            pr = project(np.where(use[sl][:, :, None], y, 0.0), mu[sl], use[sl], te_ms)
            sig[sl] = sigma_from_residual(pr["r"], np.clip(use[sl].sum(1) - 2, 0, None))
        return sig

    # --- step 2: sigma_init in high-SNR voxels, smooth map.
    # A voxel is trusted if its weakest included echo has mu / sigma_ref > sigma_init_snr, using all
    # usable echoes or, failing that, echoes 1-3 (echo 4 is often near the floor brain-wide).
    # sigma_ref is a heavily smoothed map of the echoes-1-3 estimates: deciding trust from the voxel's
    # own (noisy) estimate would preferentially keep voxels whose sigma came out low.
    thr = F["sigma_init_snr"]
    fwhm = float(F.get("sigma_smooth_fwhm_vox", 3.0))
    first3 = use0 & (np.arange(E) < 3)[None]
    sg4, sg3 = residual_sigma(use0), residual_sigma(first3)
    sigma_ref = smooth_map(sg3, np.isfinite(sg3).astype(float), mask3d, 2 * fwhm)
    with np.errstate(divide="ignore", invalid="ignore"):
        t4 = (np.clip(use0.sum(1) - 2, 0, None) >= 1) & np.isfinite(sg4) & \
            (np.where(use0, mu, np.inf).min(1) / sigma_ref > thr)
        t3 = ~t4 & (first3.sum(1) == 3) & np.isfinite(sg3) & \
            (np.where(first3, mu, np.inf).min(1) / sigma_ref > thr)
    sig0 = np.where(t4, sg4, np.where(t3, sg3, np.nan))
    trusted = t4 | t3
    n_min = int(F.get("sigma_init_min_voxels", 50))
    if trusted.sum() < n_min:
        raise ValueError(f"only {trusted.sum()} voxels have mu/sigma > sigma_init_snr = {thr} on 3 or 4 echoes "
                         f"(need {n_min}); SNR too low for the residual-based sigma (lower sigma_init_snr?)")
    sig_smooth = smooth_map(sig0, trusted.astype(float), mask3d, fwhm)

    # --- step 3: echo usability from the smooth map
    snr = mu / sig_smooth[:, None]
    use = use0 & (snr >= F["snr_min"])
    n_use = use.sum(1)
    df = np.clip(n_use - 2, 0, None)

    # --- step 4: per-voxel sigma with the usable echoes; smooth map where low SNR or df = 0
    sig_vox = residual_sigma(use)
    mu_min = np.where(use, mu, np.inf).min(1)
    low = (df == 0) | ~np.isfinite(sig_vox) | (mu_min / sig_smooth < F["sigma_init_snr"])
    sigma = np.where(low, sig_smooth, sig_vox)
    if F.get("whiten_with", "voxel") == "smooth":
        sigma = sig_smooth

    # --- step 5: whitening and channels
    c = np.full((V, 2, T), np.nan, np.float32)
    r = np.full((V, 2, T), np.nan, np.float32)
    R = np.full((V, 2, 2), np.nan)
    for i0 in range(0, V, chunk):
        sl = slice(i0, min(i0 + chunk, V))
        y = signal_form(S[sl], mu[sl], form)
        pr = project(np.where(use[sl][:, :, None], y, 0.0), mu[sl] / sigma[sl, None], use[sl], te_ms)
        c[sl], r[sl], R[sl] = pr["c"], pr["r"], pr["R"]

    # --- step 6: static fit and diagnostics
    s0, t2, muhat = fit_monoexp(mu, te_ms, use)
    with np.errstate(divide="ignore", invalid="ignore"), _quiet():
        lr = np.where(use, np.log(mu / muhat), np.nan)
        static_misfit = np.sqrt(np.nanmean(lr ** 2, axis=1))
        rc = r.astype(np.float64) - np.nanmean(r, axis=2, keepdims=True)
        # residual energy per frame and voxel, in units of the per-voxel sigma (about 1 under the model)
        e_frame = np.where(df[:, None] > 0, np.nansum(rc ** 2, axis=1) / df[:, None], np.nan)   # (V, T)
        resid_energy = np.nanmean(e_frame, axis=1)
        # the same against the smooth sigma: > 1 where a localised violation inflates the per-voxel sigma
        resid_excess = resid_energy * (sigma / sig_smooth) ** 2
    with _quiet():
        frame_energy = np.nanmean(e_frame, axis=0)                     # (T,) brain mean per frame
        # residual split: r0 ~ curvature-like, r1 ~ odd/even-echo (alternating) pattern; vs smooth sigma
        scale = (sigma / sig_smooth) ** 2
        resid_excess_r0 = np.where(df >= 1, np.nanmean(rc[:, 0] ** 2, axis=1) * scale, np.nan)
        resid_excess_r1 = np.where(df >= 2, np.nanmean(rc[:, 1] ** 2, axis=1) * scale, np.nan)
    static_curv, static_alt = static_shape(mu, te_ms, use)
    pe_sel = use.all(1) & (snr[:, -1] >= float(F.get("per_echo_noise_min_snr", 10.0)))
    per_echo = None
    if pe_sel.sum() >= 500:
        idx = np.flatnonzero(pe_sel)
        cap = int(F.get("per_echo_noise_max_voxels", 20000))
        if idx.size > cap:
            idx = idx[np.linspace(0, idx.size - 1, cap).astype(int)]
        per_echo = per_echo_noise(S[idx], mu[idx], te_ms, form)
        if quiet_frames is not None and quiet_frames.sum() >= 100:   # frames nearest the median head position
            per_echo["low_drift"] = per_echo_noise(S[idx][:, :, quiet_frames], mu[idx], te_ms, form)

    return {"mu": mu, "use": use, "n_use": n_use, "df": df, "nonpositive": nonpos, "snr": snr,
            "sigma": sigma, "sigma_voxel": sig_vox, "sigma_smooth": sig_smooth, "sigma_init": sig0,
            "sigma_low_snr": low, "trusted": trusted, "s0": s0, "t2s_ms": t2, "muhat": muhat,
            "static_misfit": static_misfit, "static_logres": lr, "static_curvature": static_curv,
            "static_alternating": static_alt, "resid_excess_r0": resid_excess_r0, "resid_excess_r1": resid_excess_r1,
            "per_echo_noise": per_echo,
            "resid_energy": resid_energy, "resid_excess": resid_excess,
            "frame_energy": frame_energy, "c": c, "r": r, "R": R, "form": form}


def static_shape(mu: np.ndarray, te_ms, use: np.ndarray):
    """Shape of the static log-decay outside span{1, TE} (4-echo voxels; NaN otherwise).

    curvature: component along the TE^2 direction (orthogonalised to 1 and TE), oriented so that a
    convex log-decay (e.g. partial-volume mixing of two T2* values) is POSITIVE.
    alternating: component along the odd/even-echo pattern after removing 1, TE and TE^2 (an
    echo-specific readout artifact). Both in log units (unit-norm basis vectors)."""
    te = np.asarray(te_ms, float)
    Q, _ = np.linalg.qr(design(te))                                   # [1-TE.., 1, TE^2, alt] spans
    q_curv, q_alt = Q[:, 2], Q[:, 3]
    if q_curv[0] + q_curv[-1] < 0:                                    # convex = positive at both ends
        q_curv = -q_curv
    if q_alt[0] < 0:                                                  # alternating: + on echo 1
        q_alt = -q_alt
    full = use.all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = np.log(mu)
    curv = np.where(full, lm @ q_curv, np.nan)
    alt = np.where(full, lm @ q_alt, np.nan)
    return curv, alt


def per_echo_noise(S: np.ndarray, mu: np.ndarray, te_ms, form: str = "log", n_iter: int = 100) -> dict:
    """Per-echo thermal noise RATIOS (echo 1 = 1) from the residual, inside the brain.

    The sigma-free residual u = Q_perp^T (mu o y) (Q_perp orthogonal to mu o [1, TE]) contains no BOLD
    or S0 signal of any amplitude, whatever the per-echo noise. Its 2 x 2 noise covariance in voxel v
    is g_v * sum_e tau_e q_ve q_ve^T, a known mixture of the per-echo variances tau_e (g_v = spatial
    noise scale, e.g. g-factor). Pooling voxels with different T2* (different q) identifies tau up to
    scale. Robust covariances (median-based) resist spikes. Use voxels with all 4 echoes at high SNR.
    Simulation: recovers ratios within ~3 % with and without a g-factor map and with all sources on."""
    te = np.asarray(te_ms, float)
    D = design(te)

    def solve(Sx, mx):
        Q, _ = np.linalg.qr(mx[:, :, None] * D[None])
        qp = Q[:, :, 2:4]
        y = signal_form(Sx, mx, form)
        u = np.einsum("vek,vet->vkt", qp, mx[:, :, None] * y)
        u -= u.mean(2, keepdims=True)
        rv = lambda x: np.median(x ** 2, axis=-1) / 0.45494
        C = np.stack([rv(u[:, 0]), (rv(u[:, 0] + u[:, 1]) - rv(u[:, 0] - u[:, 1])) / 4, rv(u[:, 1])], 1)
        F = np.stack([qp[:, :, 0] ** 2, qp[:, :, 0] * qp[:, :, 1], qp[:, :, 1] ** 2], axis=2)
        tau = np.ones(te.size)
        for _ in range(n_iter):
            M = np.einsum("e,vek->vk", tau, F)
            g = (C * M).sum(1) / (M * M).sum(1)
            A = (g[:, None, None] * F).transpose(0, 2, 1).reshape(-1, te.size)
            tau, *_ = np.linalg.lstsq(A, C.reshape(-1), rcond=None)
            tau = tau / tau[0]
        return np.sqrt(np.clip(tau, 0, None)), float(np.linalg.cond(A))

    S = S.astype(np.float64)
    ratio, cond = solve(S, mu)
    half = np.arange(S.shape[0]) % 2 == 0
    r_a, _ = solve(S[half], mu[half])
    r_b, _ = solve(S[~half], mu[~half])
    return {"ratio": ratio, "ratio_half_a": r_a, "ratio_half_b": r_b, "condition": cond, "n_voxels": int(S.shape[0])}


def drift_mm(par: np.ndarray) -> np.ndarray:
    """Per-frame distance (mm) of head position from the run's median position, from mcflirt .par
    (rotations x, y, z in rad scaled by a 50 mm radius; translations in mm). Unlike FD (frame-to-frame
    change), this sees slow drift."""
    d = np.column_stack([par[:, :3] * 50.0, par[:, 3:6]])
    return np.abs(d - np.median(d, axis=0)).sum(axis=1)


def quiet_window(par: np.ndarray, n: int) -> int:
    """Start of the n consecutive frames whose positions stay closest to their own median."""
    d = np.column_stack([par[:, :3] * 50.0, par[:, 3:6]])
    T = d.shape[0]
    if T <= n:
        return 0
    costs = [np.abs(d[s:s + n] - np.median(d[s:s + n], axis=0)).sum(1).mean() for s in range(0, T - n + 1, 10)]
    return int(10 * np.argmin(costs))


def mask_holes(mask3d: np.ndarray, affine) -> dict:
    """Voxels enclosed by the brain mask but not in it: 3D hole filling and slice-wise (axial) filling."""
    import nibabel as nib

    filled3 = ndimage.binary_fill_holes(mask3d)
    ax = next(i for i, c in enumerate(nib.aff2axcodes(affine)) if c in ("S", "I"))
    filled2 = np.zeros_like(mask3d)
    for z in range(mask3d.shape[ax]):
        sl = [slice(None)] * 3
        sl[ax] = z
        filled2[tuple(sl)] = ndimage.binary_fill_holes(mask3d[tuple(sl)])
    return {"holes_3d": filled3 & ~mask3d, "holes_axial": filled2 & ~mask3d}


# ======================================================================= MP-PCA cross-check
def mppca_sigma(data4d: np.ndarray, mask3d: np.ndarray, cfg: dict, start: int = 0) -> np.ndarray:
    """dipy MP-PCA noise SD (3D, NaN outside sampled voxels) for one echo of RAW (not motion-
    corrected) data. Patch centres on a sparse grid (every `inplane_step`-th voxel in-plane, every
    `slice_step`-th slice) for speed; each patch still uses all its neighbours. dipy API checked in 1.12.1: mppca(arr, *, mask, patch_radius, return_sigma)."""
    import warnings

    from dipy.denoise.localpca import mppca

    M = cfg["frontend"]["mppca"]
    d = data4d[..., start: start + int(M["n_volumes"])].astype(np.float64)
    centres = np.zeros_like(mask3d)
    st, ip = int(M["slice_step"]), int(M.get("inplane_step", 1))
    centres[::ip, ::ip, ::st] = mask3d[::ip, ::ip, ::st]            # sparse centres: sigma varies smoothly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, sg = mppca(d, mask=centres, patch_radius=int(M["patch_radius"]), return_sigma=True)
    return np.where(centres, sg, np.nan)


# ======================================================================= one real subject
def resolve_run(run_dir, cfg: dict, source: str = "raw") -> dict:
    """Inputs of one mbme_tedana run: echo files, paired volumes, TEs, brain mask, FD."""
    import json
    import re
    from pathlib import Path

    run_dir = Path(run_dir)
    res = json.loads((run_dir / "result.json").read_text())
    run_id = res.get("run_id") or run_dir.name
    m = re.match(r"sub-([^_]+)", run_id)
    subject = str(res.get("subject") or (m.group(1) if m else run_id))
    if subject in {str(k) for k in (cfg.get("subjects", {}).get("exclude") or {})}:
        raise ValueError(f"subject {subject} is excluded in configs/default.yaml: {cfg['subjects']['exclude'][subject]}")
    te = res.get("echo_times_ms") or cfg["acquisition"]["te_ms"]
    if source == "raw":
        ei = run_dir / "echo_info.json"
        files = json.loads(ei.read_text()).get("echo_files") if ei.exists() else None
        if not files:
            e1 = res["input_echo1"]
            files = [re.sub(r"_e1(\.nii(\.gz)?)$", rf"_e{k}\1", e1) for k in range(1, len(te) + 1)]
        vu = np.loadtxt(run_dir / "volumes_used.tsv", skiprows=1, ndmin=2).astype(int)
        volumes = [vu[:, k] for k in range(vu.shape[1])]
    elif source == "mc":
        files = [str(run_dir / "work" / f"echo-{k}_desc-mc_bold.nii.gz") for k in range(1, len(te) + 1)]
        volumes = None
    else:
        raise ValueError("source must be 'raw' or 'mc'")
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"missing echo files: {missing}")
    fdp = run_dir / "motion" / "framewise_displacement.txt"
    parp = run_dir / "motion" / "mcflirt.par"
    return {"par": np.loadtxt(parp, ndmin=2) if parp.exists() else None,"run_dir": str(run_dir), "run_id": run_id, "subject": subject, "source": source,
            "echo_files": [str(f) for f in files], "volumes": volumes, "te_ms": [float(t) for t in te],
            "mask": str(run_dir / "brain_mask.nii.gz"), "fd": np.loadtxt(fdp) if fdp.exists() else None}


def resolve_echo1(echo1, cfg: dict) -> dict:
    """Inputs from the raw echo files alone (no mbme_tedana or tedana outputs): echoes found next to
    echo 1 (_e1 -> _ek, dcm2niix names), TEs from the JSON sidecars. Equal volume counts are required;
    unequal counts need mbme_tedana's volume pairing (use --run-dir)."""
    import json
    import re
    from pathlib import Path

    import nibabel as nib

    e1 = Path(echo1)
    m = re.match(r"^(?P<base>.+)_e1(?P<ext>\.nii(?:\.gz)?)$", e1.name)
    if not m:
        raise ValueError(f"echo-1 file name must end in _e1.nii(.gz): {e1.name}")
    files = []
    k = 1
    while (e1.with_name(f"{m['base']}_e{k}{m['ext']}")).exists():
        files.append(e1.with_name(f"{m['base']}_e{k}{m['ext']}"))
        k += 1
    n_exp = len(cfg["acquisition"]["te_ms"])
    if len(files) != n_exp:
        raise ValueError(f"expected {n_exp} echoes next to {e1.name}, found {len(files)}")
    te = []
    for f in files:
        js = f.with_name(re.sub(r"\.nii(\.gz)?$", ".json", f.name))
        t = json.loads(js.read_text()).get("EchoTime") if js.exists() else None
        te.append(None if t is None else (1000.0 * float(t) if float(t) < 1 else float(t)))
    if None in te:
        raise ValueError("EchoTime missing from a JSON sidecar; cannot confirm TEs")
    counts = [nib.load(str(f)).shape[3] for f in files]
    if len(set(counts)) > 1:
        raise ValueError(f"echo volume counts differ {counts}: volumes must be paired first (use --run-dir with "
                         "mbme_tedana's volumes_used.tsv)")
    subj = next((p for p in e1.parts[:-1] if p.isdigit()), None)
    ms = re.search(r"_(\d{14})_(\d+)_e1\.nii", e1.name)
    run_id = f"sub-{subj}_acq-{ms.group(1)}_ser-{ms.group(2)}" if (subj and ms) else m["base"]
    if subj and subj in {str(k) for k in (cfg.get("subjects", {}).get("exclude") or {})}:
        raise ValueError(f"subject {subj} is excluded in configs/default.yaml: {cfg['subjects']['exclude'][subj]}")
    return {"run_dir": None, "run_id": run_id, "subject": subj, "source": "raw", "echo_files": [str(f) for f in files],
            "volumes": None, "te_ms": te, "mask": None, "fd": None, "par": None}


def inferior_third(mask3d: np.ndarray, affine) -> np.ndarray:
    """Brain voxels in the inferior third of the mask's superior-inferior extent: an APPROXIMATE
    posterior-fossa / skull-base level (no anatomical masks exist for this data)."""
    import nibabel as nib

    codes = nib.aff2axcodes(affine)
    ax = next(i for i, c in enumerate(codes) if c in ("S", "I"))
    idx = np.indices(mask3d.shape)[ax]
    zs = idx[mask3d]
    lo, hi = zs.min(), zs.max()
    cut = lo + (hi - lo) / 3 if codes[ax] == "S" else hi - (hi - lo) / 3
    return mask3d & ((idx <= cut) if codes[ax] == "S" else (idx >= cut))


def run_subject(run_dir, cfg: dict, out_dir, source: str = "raw", mppca: bool | None = None,
                mask_source: str | None = None, echo1=None) -> dict:
    """Front end on one real run: maps (NIfTI), channel cache (.npy), summary, provenance."""
    import json
    from pathlib import Path

    import nibabel as nib

    from . import io

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from . import mask as maskmod

    if (run_dir is None) == (echo1 is None):
        raise ValueError("give exactly one of run_dir (mbme_tedana run folder) or echo1 (raw echo-1 file)")
    info = resolve_run(run_dir, cfg, source) if run_dir is not None else resolve_echo1(echo1, cfg)
    ref = nib.load(info["echo_files"][0])
    pipe_mask = None
    if info.get("mask") and Path(info["mask"]).exists():
        pipe_mask = np.asarray(nib.load(info["mask"]).dataobj) > 0
    holes = mask_holes(pipe_mask, ref.affine) if pipe_mask is not None else None
    # analysis mask (user decision 2026-11-02): built from the input echoes ("auto", default); the
    # pipeline's mask (compute_epi_mask: never fills holes, drops the deep brain) only for comparisons
    mask_source = mask_source or cfg["frontend"].get("mask_source", "auto")
    mask_info = {"source": mask_source}
    if mask_source == "auto":
        def temporal_mean(k):
            d = np.asarray(nib.load(info["echo_files"][k]).dataobj, dtype=np.float32)
            if info["volumes"] is not None:
                d = d[..., np.asarray(info["volumes"][k], int)]
            return d.mean(axis=3)
        si = next(i for i, c in enumerate(nib.aff2axcodes(ref.affine)) if c in ("S", "I"))
        mk = maskmod.build_mask(temporal_mean(0), ref.header.get_zooms()[:3], cfg, si_axis=si,
                                mean_echo_last=temporal_mean(len(info["echo_files"]) - 1), te_ms=info["te_ms"])
        mask3d = mk["mask"]
        mask_info.update(volume_ml=mk["volume_ml"], head_volume_ml=mk["head_volume_ml"], air_level=round(mk["air_level"], 3))
    elif mask_source in ("pipeline", "pipeline_filled"):
        if pipe_mask is None:
            raise ValueError(f"mask_source {mask_source} needs the pipeline's brain_mask.nii.gz (use --run-dir)")
        mask3d = pipe_mask if mask_source == "pipeline" else (pipe_mask | holes["holes_3d"] | holes["holes_axial"])
    else:
        raise ValueError("mask_source must be 'auto', 'pipeline' or 'pipeline_filled'")
    if pipe_mask is not None:
        mask_info.update(dice_vs_pipeline=round(maskmod.dice(mask3d, pipe_mask), 4),
                         n_only_analysis_mask=int((mask3d & ~pipe_mask).sum()),
                         n_only_pipeline_mask=int((pipe_mask & ~mask3d).sum()))
    do_mp = cfg["frontend"]["mppca"]["enabled"] if mppca is None else mppca
    if do_mp and source != "raw":
        do_mp = False                                   # CLAUDE.md 5.4: MP-PCA check on raw echoes only
    mp, mpq, bg, extra = {}, {}, {}, {}
    B = cfg["frontend"].get("background", {"dilate_vox": 4, "rel_threshold": 0.02})
    nmp = int(cfg["frontend"]["mppca"]["n_volumes"])
    q_start = quiet_window(info["par"], nmp) if info["par"] is not None else None

    def hook(k, d4):
        if k == 0:                                     # unmasked echo-1 mean and the air (background) mask
            m1 = d4.mean(axis=3)
            extra["mean_echo1"] = m1
            far = ~ndimage.binary_dilation(mask3d, iterations=int(B["dilate_vox"]))
            extra["background"] = far & (m1 < float(B["rel_threshold"]) * np.median(m1[mask3d]))
        b = extra["background"]
        if b.sum() >= 50:                              # air: pure noise; its level per echo tracks thermal sigma
            bd = d4[b]
            bg[k] = {"mean": float(np.median(bd.mean(1))), "sd": float(np.median(bd.std(1, ddof=1))),
                     "n_voxels": int(b.sum())}
        if do_mp:
            mp[k] = mppca_sigma(d4, mask3d, cfg)
            if q_start is not None and cfg["frontend"]["mppca"].get("quiet_window", True):
                mpq[k] = mppca_sigma(d4, mask3d, cfg, start=q_start)

    S = io.load_masked_echoes(info["echo_files"], mask3d, info["volumes"], hook)
    te = np.asarray(info["te_ms"])
    quiet = None
    if info["par"] is not None and len(info["par"]) == S.shape[2]:
        dr = drift_mm(info["par"])
        quiet = dr <= np.median(dr)
    fe = run(S, te, cfg, mask3d, quiet_frames=quiet)
    V, E, T = S.shape

    # ---- maps
    save = lambda name, vals: io.save_map(vals, mask3d, ref, out_dir / f"{name}.nii.gz")
    save("mu", fe["mu"])
    save("snr", fe["snr"])
    save("usable", fe["use"].astype(float))
    save("nonpositive", fe["nonpositive"].astype(float))
    for k in ("t2s_ms", "s0", "sigma", "sigma_voxel", "sigma_smooth", "resid_energy", "resid_excess", "static_misfit"):
        save(k, fe[k])
    save("n_usable_echoes", fe["n_use"].astype(float))
    save("df", fe["df"].astype(float))
    save("sigma_ratio_voxel_smooth", fe["sigma_voxel"] / fe["sigma_smooth"])
    save("R", fe["R"].reshape(V, 4)[:, [0, 1, 3]])                    # R00, R01, R11
    for k in ("static_curvature", "static_alternating", "resid_excess_r0", "resid_excess_r1", "static_logres"):
        save(k, fe[k])
    io.save_volume(extra["mean_echo1"], ref, out_dir / "mean_echo1_unmasked.nii.gz")
    io.save_volume(extra["background"].astype(np.float32), ref, out_dir / "background_mask.nii.gz")
    io.save_volume(mask3d.astype(np.float32), ref, out_dir / "analysis_mask.nii.gz")
    if pipe_mask is not None:
        io.save_volume(pipe_mask.astype(np.float32), ref, out_dir / "pipeline_mask.nii.gz")
    if holes is not None:
        io.save_volume(holes["holes_axial"].astype(np.float32), ref, out_dir / "mask_holes_axial.nii.gz")
    if mp:
        mpv = np.stack([mp[k][mask3d] for k in range(E)], axis=1)
        save("mppca_sigma", mpv)

    # ---- channel cache for later phases
    for name, arr in (("c0", fe["c"][:, 0]), ("c1", fe["c"][:, 1]), ("r0", fe["r"][:, 0]), ("r1", fe["r"][:, 1])):
        np.save(out_dir / f"{name}.npy", arr)
    np.save(out_dir / "R.npy", fe["R"])
    drift = drift_mm(info["par"]) if info["par"] is not None and len(info["par"]) == T else None
    io.write_tsv(out_dir / "frames.tsv", {"frame": list(range(T)), "residual_energy": list(fe["frame_energy"]),
                                         "fd_mm": list(info["fd"]) if info["fd"] is not None and len(info["fd"]) == T else ["n/a"] * T,
                                         "drift_mm": list(drift) if drift is not None else ["n/a"] * T})

    # ---- summary
    inf = inferior_third(mask3d, ref.affine)[mask3d]
    pct = lambda x, q=(10, 50, 90): [round(float(v), 4) for v in np.nanpercentile(x, q)]
    summ = {
        "run_id": info["run_id"], "source": source, "mask_source": mask_source, "mask": mask_info, "n_voxels": int(V),
        "n_voxels_pipeline_mask": int(pipe_mask.sum()) if pipe_mask is not None else None,
        "n_volumes": int(T), "te_ms": info["te_ms"],
        "inferior_third_note": "approximate posterior-fossa / skull-base level (inferior third of the brain mask)",
        "n_voxels_inferior_third": int(inf.sum()),
        "usable_fraction_per_echo": [round(float(v), 4) for v in fe["use"].mean(0)],
        "usable_fraction_per_echo_inferior": [round(float(v), 4) for v in fe["use"][inf].mean(0)],
        "n_usable_echoes_fraction": {str(n): round(float(np.mean(fe["n_use"] == n)), 4) for n in range(E, -1, -1)},
        "echo4_snr_p10_50_90": pct(fe["snr"][:, -1]),
        "echo4_snr_p10_50_90_inferior": pct(fe["snr"][inf, -1]),
        "t2s_ms_p10_50_90": pct(fe["t2s_ms"]), "t2s_ms_p10_50_90_inferior": pct(fe["t2s_ms"][inf]),
        "sigma_median": float(np.nanmedian(fe["sigma"])), "sigma_smooth_median": float(np.nanmedian(fe["sigma_smooth"])),
        "fraction_sigma_from_smooth_map": round(float(np.mean(fe["sigma_low_snr"])), 4),
        "n_trusted_voxels": int(fe["trusted"].sum()),
        "sigma_voxel_over_smooth_p10_50_90": pct(fe["sigma_voxel"] / fe["sigma_smooth"]),
        "resid_energy_median": float(np.nanmedian(fe["resid_energy"])),
        "resid_excess_median": float(np.nanmedian(fe["resid_excess"])),
        "resid_excess_median_inferior": float(np.nanmedian(fe["resid_excess"][inf])),
        "static_misfit_median": float(np.nanmedian(fe["static_misfit"])),
        "static_misfit_median_inferior": float(np.nanmedian(fe["static_misfit"][inf])),
        "nonpositive_voxels_per_echo": [int(v) for v in fe["nonpositive"].sum(0)],
    }
    from scipy.stats import spearmanr
    if info["fd"] is not None and len(info["fd"]) == T:
        summ["frame_energy_vs_fd_spearman"] = round(float(spearmanr(fe["frame_energy"], info["fd"])[0]), 4)
    if drift is not None:
        summ["frame_energy_vs_drift_spearman"] = round(float(spearmanr(fe["frame_energy"], drift)[0]), 4)
        summ["drift_mm_p50_p90_max"] = [round(float(v), 3) for v in np.percentile(drift, [50, 90, 100])]
    # static decay shape and residual split
    lres = fe["static_logres"]
    summ["static_logres_median_per_echo"] = [round(float(v), 4) for v in np.nanmedian(lres, axis=0)]
    summ["static_logres_median_per_echo_inferior"] = [round(float(v), 4) for v in np.nanmedian(lres[inf], axis=0)]
    summ["static_curvature_p10_50_90"] = pct(fe["static_curvature"])
    summ["static_alternating_p10_50_90"] = pct(fe["static_alternating"])
    if fe["per_echo_noise"] is not None:
        pe = fe["per_echo_noise"]
        clean = lambda d: {k: ([round(float(x), 4) for x in v] if isinstance(v, np.ndarray) else v)
                           for k, v in d.items() if k != "low_drift"}
        summ["per_echo_noise"] = clean(pe)
        if "low_drift" in pe:
            summ["per_echo_noise"]["low_drift"] = clean(pe["low_drift"])
    summ["resid_excess_r0_median"] = float(np.nanmedian(fe["resid_excess_r0"]))
    summ["resid_excess_r1_median"] = float(np.nanmedian(fe["resid_excess_r1"]))
    # mask holes and background noise
    if holes is not None:
        summ["mask_holes"] = {"axial_fill_voxels": int(holes["holes_axial"].sum()), "fill3d_voxels": int(holes["holes_3d"].sum()),
                              "axial_fill_fraction_of_mask": round(float(holes["holes_axial"].sum() / pipe_mask.sum()), 4)}
    if bg:
        b1 = bg[0]
        summ["background_noise"] = [{"echo": k + 1, "median_mean": round(bg[k]["mean"], 3), "median_temporal_sd": round(bg[k]["sd"], 3),
                                     "mean_ratio_to_echo1": round(bg[k]["mean"] / b1["mean"], 4),
                                     "sd_ratio_to_echo1": round(bg[k]["sd"] / b1["sd"], 4), "n_voxels": bg[k]["n_voxels"]}
                                    for k in sorted(bg)]
    if mp:
        exp = float(cfg["frontend"]["mppca_expected_ratio"])
        rows = []
        for k in range(E):
            sel = np.isfinite(mp[k][mask3d])
            ratio = float(np.nanmedian(fe["sigma"][sel]) / np.nanmedian(mp[k][mask3d][sel]))
            row_q = {}
            if k in mpq:
                selq = np.isfinite(mpq[k][mask3d])
                row_q = {"mppca_sigma_median_quiet": float(np.nanmedian(mpq[k][mask3d][selq])),
                         "ratio_residual_over_mppca_quiet": round(float(np.nanmedian(fe["sigma"][selq]) /
                                                                         np.nanmedian(mpq[k][mask3d][selq])), 4)}
            rows.append({"echo": k + 1, **row_q, "mu_median_same_voxels": float(np.nanmedian(fe["mu"][sel, k])),
                         "mppca_sigma_median": float(np.nanmedian(mp[k][mask3d][sel])),
                         "residual_sigma_median_same_voxels": float(np.nanmedian(fe["sigma"][sel])),
                         "ratio_residual_over_mppca": round(ratio, 4),
                         "ratio_corrected_for_simulated_bias": round(ratio / exp, 4)})
        worst = max(abs(r["ratio_corrected_for_simulated_bias"] - 1) for r in rows)
        tol = float(cfg["frontend"]["mppca_sigma_tolerance"])
        summ["mppca_check"] = {"per_echo": rows, "max_abs_deviation": round(worst, 4), "tolerance": tol,
                               "quiet_window_start": q_start, "n_volumes": nmp,
                               "verdict": "OK" if worst <= tol else "STOP: residual and MP-PCA sigma differ by more than the tolerance"}
    io.write_json(out_dir / "summary.json", summ)
    io.write_text(out_dir / "summary.md", render_summary(summ))
    io.write_config_copy(cfg, out_dir)
    inputs = {f"echo{k + 1}": f for k, f in enumerate(info["echo_files"])}
    if run_dir is not None:
        inputs.update({"pipeline_mask": info["mask"], "volumes_used": str(Path(run_dir) / "volumes_used.tsv")})
    io.write_json(out_dir / "done.json", {"stage": f"frontend_{source}", **io.provenance(cfg, inputs)})
    return summ


def render_summary(s: dict) -> str:
    L = [f"# Front end: {s['run_id']} ({s['source']} echoes, mask: {s.get('mask_source', 'pipeline')})", "",
         f"{s['n_voxels']} brain voxels in the analysis mask"
         + (f" (pipeline mask {s['n_voxels_pipeline_mask']})" if s.get('n_voxels_pipeline_mask') else "") + ", "
         f"{s['n_volumes']} volumes, TE {s['te_ms']} ms. "
         f"'Inferior third' = {s['inferior_third_note']} ({s['n_voxels_inferior_third']} voxels).", "",
         "| Quantity | Whole brain | Inferior third |", "|---|---|---|",
         f"| Usable fraction per echo | {s['usable_fraction_per_echo']} | {s['usable_fraction_per_echo_inferior']} |",
         f"| Echo-4 SNR (mu/sigma) p10/50/90 | {s['echo4_snr_p10_50_90']} | {s['echo4_snr_p10_50_90_inferior']} |",
         f"| T2* (ms) p10/50/90 | {s['t2s_ms_p10_50_90']} | {s['t2s_ms_p10_50_90_inferior']} |",
         f"| Residual energy vs smooth sigma (median; 1 = model holds) | {s['resid_excess_median']:.3f} | {s['resid_excess_median_inferior']:.3f} |",
         f"| Static mono-exponential misfit (median, log units) | {s['static_misfit_median']:.4f} | {s['static_misfit_median_inferior']:.4f} |",
         "", f"- Voxels by number of usable echoes (4/3/2/1/0): {s['n_usable_echoes_fraction']}",
         f"- Sigma: median {s['sigma_median']:.3f} (smooth map {s['sigma_smooth_median']:.3f}); "
         f"{100 * s['fraction_sigma_from_smooth_map']:.1f} % of voxels use the smooth map; {s['n_trusted_voxels']} trusted voxels",
         f"- Per-voxel / smooth sigma p10/50/90: {s['sigma_voxel_over_smooth_p10_50_90']}",
         f"- Voxels with any frame <= 0, per echo: {s['nonpositive_voxels_per_echo']}"]
    if "frame_energy_vs_fd_spearman" in s:
        L.append(f"- Frame-wise residual energy vs FD (Spearman): {s['frame_energy_vs_fd_spearman']}")
    if "frame_energy_vs_drift_spearman" in s:
        L.append(f"- Frame-wise residual energy vs head drift from median position (Spearman): "
                 f"{s['frame_energy_vs_drift_spearman']}; drift p50/p90/max {s['drift_mm_p50_p90_max']} mm")
    mk = s.get("mask") or {}
    if mk.get("source") == "auto":
        L.append(f"- Analysis mask built from the echoes: {mk.get('volume_ml')} ml (head {mk.get('head_volume_ml')} ml)"
                 + (f"; Dice with the pipeline mask {mk['dice_vs_pipeline']}, {mk['n_only_analysis_mask']} voxels only in "
                    f"the analysis mask, {mk['n_only_pipeline_mask']} only in the pipeline mask" if "dice_vs_pipeline" in mk else ""))
    if "mask_holes" in s:
        h = s["mask_holes"]
        L.append(f"- Pipeline-mask holes: {h['axial_fill_voxels']} voxels enclosed slice-wise but outside the mask "
                 f"({100 * h['axial_fill_fraction_of_mask']:.1f} % of the mask); {h['fill3d_voxels']} enclosed in 3D")
    if "static_logres_median_per_echo" in s:
        L += ["", "## Shape of the static decay (log units; mean signal vs mono-exponential fit)", "",
              f"- Median ln(mu / fit) per echo, whole brain: {s['static_logres_median_per_echo']}; inferior third: "
              f"{s['static_logres_median_per_echo_inferior']}",
              f"- Curvature (positive = convex, e.g. partial-volume mixing) p10/50/90: {s['static_curvature_p10_50_90']}",
              f"- Odd/even-echo alternation p10/50/90: {s['static_alternating_p10_50_90']}",
              f"- Dynamic residual vs smooth sigma, median: curvature-like r0 {s['resid_excess_r0_median']:.3f}, "
              f"alternating-like r1 {s['resid_excess_r1_median']:.3f}"]
    if "per_echo_noise" in s:
        pe = s["per_echo_noise"]
        L += ["", "## Per-echo thermal noise from the residual, inside the brain (ratios to echo 1)", "",
              f"Equal noise across echoes (CLAUDE.md 5.3) gives [1, 1, 1, 1]. Estimated: **{pe['ratio']}** "
              f"(two independent voxel halves: {pe['ratio_half_a']} / {pe['ratio_half_b']}; {pe['n_voxels']} voxels "
              f"with 4 usable echoes and echo-4 SNR >= 10; condition {pe['condition']:.0f}). The residual holds no BOLD or "
              "S0 signal, so physiology cannot inflate it; head motion and model violations can."]
        if "low_drift" in pe:
            ld = pe["low_drift"]
            L.append(f"On the half of frames nearest the median head position: **{ld['ratio']}** "
                     f"(halves {ld['ratio_half_a']} / {ld['ratio_half_b']}).")
    if "background_noise" in s:
        L += ["", "## Noise in air outside the head, per echo (pure noise: ratios track thermal sigma)", "",
              "| Echo | Median mean | Median temporal SD | Mean / echo 1 | SD / echo 1 | Voxels |", "|---|---|---|---|---|---|"]
        L += [f"| {b['echo']} | {b['median_mean']:.2f} | {b['median_temporal_sd']:.2f} | {b['mean_ratio_to_echo1']:.3f} | "
              f"{b['sd_ratio_to_echo1']:.3f} | {b['n_voxels']} |" for b in s["background_noise"]]
    if "mppca_check" in s:
        m = s["mppca_check"]
        L += ["", "## Sigma cross-check: residual vs MP-PCA (raw echoes)", "",
              f"MP-PCA on frames 0-{m.get('n_volumes', 100) - 1} and on the quietest window (start frame {m.get('quiet_window_start')}).", "",
              "| Echo | Mean signal | MP-PCA sigma | MP-PCA sigma (quiet) | Residual sigma | Ratio | Ratio (quiet) | Ratio / simulated bias |",
              "|---|---|---|---|---|---|---|---|"]
        f = lambda v: "n/a" if v is None else f"{v:.3f}"
        L += [f"| {r['echo']} | {f(r.get('mu_median_same_voxels'))} | {r['mppca_sigma_median']:.3f} | {f(r.get('mppca_sigma_median_quiet'))} | "
              f"{r['residual_sigma_median_same_voxels']:.3f} | {r['ratio_residual_over_mppca']:.3f} | "
              f"{f(r.get('ratio_residual_over_mppca_quiet'))} | {r['ratio_corrected_for_simulated_bias']:.3f} |" for r in m["per_echo"]]
        L += ["", f"Verdict (tolerance {100 * m['tolerance']:.0f} %): **{m['verdict']}**"]
    return "\n".join(L) + "\n"


# ======================================================================= all runs (noise survey)
def run_all(pipeline_out, cfg: dict, out_root, source: str = "raw", mppca: bool = False, log=print) -> list[dict]:
    """Front end on every mbme_tedana run (excluded subjects skipped), then one table of the per-echo
    noise ratios and key numbers across runs (<out>/noise_survey.tsv / .md). Runs already done with the
    same config are skipped; a failing run is recorded and the loop continues."""
    import json
    import traceback
    from pathlib import Path

    from . import io

    out_root = Path(out_root)
    chash = io.config_hash(cfg)
    excluded = {str(k) for k in (cfg.get("subjects", {}).get("exclude") or {})}
    rows = []
    for rd in sorted(p.parent for p in Path(pipeline_out).glob("*/result.json")):
        subj = rd.name.split("_")[0].replace("sub-", "")
        if subj in excluded:
            log(f"{rd.name}: excluded")
            continue
        od = out_root / rd.name / source
        done = od / "done.json"
        try:
            if done.exists() and json.loads(done.read_text()).get("config_sha256") == chash and (od / "summary.json").exists():
                log(f"{rd.name}: skipped (done)")
            else:
                run_subject(rd, cfg, od, source=source, mppca=mppca)
                log(f"{rd.name}: ok")
            s = json.loads((od / "summary.json").read_text())
            pe = s.get("per_echo_noise") or {}
            ld = pe.get("low_drift") or {}
            rows.append({"run": rd.name, "n_voxels": s["n_voxels"], "mask_ml": (s.get("mask") or {}).get("volume_ml"),
                         "ratio": pe.get("ratio"), "half_a": pe.get("ratio_half_a"), "half_b": pe.get("ratio_half_b"),
                         "ratio_low_drift": ld.get("ratio"), "n_voxels_ratio": pe.get("n_voxels"),
                         "sigma_median": round(s["sigma_median"], 2), "echo4_usable": s["usable_fraction_per_echo"][-1],
                         "resid_vs_drift": s.get("frame_energy_vs_drift_spearman"),
                         "t2s_median": s["t2s_ms_p10_50_90"][1]})
        except Exception as exc:
            io.write_text(od / "error.txt", traceback.format_exc())
            rows.append({"run": rd.name, "error": f"{type(exc).__name__}: {exc}"})
            log(f"{rd.name}: FAILED ({exc})")
    write_noise_survey(rows, out_root, cfg)
    return rows


def write_noise_survey(rows: list[dict], out_root, cfg: dict) -> None:
    from pathlib import Path

    from . import io

    out_root = Path(out_root)
    good = [r for r in rows if "error" not in r and r.get("ratio")]
    cols = ["run", "ratio", "ratio_low_drift", "half_a", "half_b", "n_voxels_ratio", "sigma_median", "echo4_usable",
            "t2s_median", "resid_vs_drift", "mask_ml", "n_voxels", "error"]
    fmt = lambda v: "[" + ", ".join(f"{x:.3f}" for x in v) + "]" if isinstance(v, list) else ("" if v is None else v)
    io.write_tsv(out_root / "noise_survey.tsv", {c: [fmt(r.get(c)) for r in rows] for c in cols})
    L = ["# Per-echo noise survey (front end on every run)", "",
         "Ratios of thermal noise to echo 1, from the residual inside the brain. Equal noise (CLAUDE.md 5.3) = [1, 1, 1, 1].", "",
         "| Run | All frames | Low-drift frames | Voxel halves | Echo-4 usable | Residual vs drift |", "|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            L.append(f"| {r['run']} | FAILED: {r['error']} | | | | |")
        else:
            L.append(f"| {r['run'].split('_acq')[0]} | {fmt(r['ratio'])} | {fmt(r['ratio_low_drift'])} | "
                     f"{fmt(r['half_a'])} / {fmt(r['half_b'])} | {r['echo4_usable']:.2f} | {r['resid_vs_drift']} |")
    if good:
        A = np.array([r["ratio"] for r in good], float)
        L += ["", f"Across {len(good)} runs, all frames: median {fmt(list(np.median(A, 0)))}, "
              f"interquartile range {fmt(list(np.percentile(A, 25, 0)))} to {fmt(list(np.percentile(A, 75, 0)))}."]
        lows = [r["ratio_low_drift"] for r in good if r.get("ratio_low_drift")]
        if lows:
            B = np.array(lows, float)
            L.append(f"Low-drift frames: median {fmt(list(np.median(B, 0)))}.")
    io.write_text(out_root / "noise_survey.md", "\n".join(L) + "\n")
    io.write_json(out_root / "noise_survey_status.json", {"rows": rows, **io.provenance(cfg, {})})
