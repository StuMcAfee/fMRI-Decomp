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
    log-linear fit weighted by mu^2; Gauss-Newton on (ln S0, R2*). T2* in ms; NaN with < 2 echoes.
    """
    te = np.asarray(te_ms, float)
    V, E = mu.shape
    w = np.where(use, 1.0, 0.0)
    ok = use.sum(1) >= 2
    lm = np.log(np.clip(mu, 1e-12, None))
    # weighted log-linear start: minimise sum w mu^2 (ln mu - (a - TE r))^2
    ww = w * mu ** 2
    sw, st, stt = ww.sum(1), (ww * te).sum(1), (ww * te ** 2).sum(1)
    sy, sty = (ww * lm).sum(1), (ww * te * lm).sum(1)
    det = sw * stt - st ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = -(sw * sty - st * sy) / det
        a = (sy + r2 * st) / sw
    r2 = np.clip(np.nan_to_num(r2, nan=1 / 40), 1 / 500, 1.0)
    a = np.nan_to_num(a, nan=0.0)
    for _ in range(n_iter):
        f = np.exp(a[:, None] - te[None] * r2[:, None])
        res = (mu - f) * w
        j1, j2 = f * w, -te[None] * f * w
        A11, A12, A22 = (j1 * j1).sum(1), (j1 * j2).sum(1), (j2 * j2).sum(1)
        b1, b2 = (j1 * res).sum(1), (j2 * res).sum(1)
        dt = A11 * A22 - A12 ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            da = np.where(dt > 0, (A22 * b1 - A12 * b2) / dt, 0.0)
            dr = np.where(dt > 0, (A11 * b2 - A12 * b1) / dt, 0.0)
        a = a + da
        r2 = np.clip(r2 + dr, 1 / 500, 1.0)
    s0 = np.where(ok, np.exp(a), np.nan)
    t2 = np.where(ok, 1.0 / r2, np.nan)
    muhat = s0[:, None] * np.exp(-te[None] / t2[:, None])
    return s0, t2, muhat


# ======================================================================= the whole front end
def run(S: np.ndarray, te_ms, cfg: dict, mask3d: np.ndarray | None = None, form: str | None = None,
        chunk: int | None = None) -> dict:
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

    return {"mu": mu, "use": use, "n_use": n_use, "df": df, "nonpositive": nonpos, "snr": snr,
            "sigma": sigma, "sigma_voxel": sig_vox, "sigma_smooth": sig_smooth, "sigma_init": sig0,
            "sigma_low_snr": low, "trusted": trusted, "s0": s0, "t2s_ms": t2, "muhat": muhat,
            "static_misfit": static_misfit, "resid_energy": resid_energy, "resid_excess": resid_excess,
            "frame_energy": frame_energy, "c": c, "r": r, "R": R, "form": form}


# ======================================================================= MP-PCA cross-check
def mppca_sigma(data4d: np.ndarray, mask3d: np.ndarray, cfg: dict) -> np.ndarray:
    """dipy MP-PCA noise SD (3D, NaN outside sampled voxels) for one echo of RAW (not motion-
    corrected) data. Patch centres on a sparse grid (every `inplane_step`-th voxel in-plane, every
    `slice_step`-th slice) for speed; each patch still uses all its neighbours. dipy API checked in 1.12.1: mppca(arr, *, mask, patch_radius, return_sigma)."""
    import warnings

    from dipy.denoise.localpca import mppca

    M = cfg["frontend"]["mppca"]
    d = data4d[..., : int(M["n_volumes"])].astype(np.float64)
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
    return {"run_dir": str(run_dir), "run_id": run_id, "subject": subject, "source": source,
            "echo_files": [str(f) for f in files], "volumes": volumes, "te_ms": [float(t) for t in te],
            "mask": str(run_dir / "brain_mask.nii.gz"), "fd": np.loadtxt(fdp) if fdp.exists() else None}


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


def run_subject(run_dir, cfg: dict, out_dir, source: str = "raw", mppca: bool | None = None) -> dict:
    """Front end on one real run: maps (NIfTI), channel cache (.npy), summary, provenance."""
    import json
    from pathlib import Path

    import nibabel as nib

    from . import io

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = resolve_run(run_dir, cfg, source)
    mimg = nib.load(info["mask"])
    mask3d = np.asarray(mimg.dataobj) > 0
    ref = nib.load(info["echo_files"][0])
    do_mp = cfg["frontend"]["mppca"]["enabled"] if mppca is None else mppca
    if do_mp and source != "raw":
        do_mp = False                                   # CLAUDE.md 5.4: MP-PCA check on raw echoes only
    mp = {}

    def hook(k, d4):
        if do_mp:
            mp[k] = mppca_sigma(d4, mask3d, cfg)

    S = io.load_masked_echoes(info["echo_files"], mask3d, info["volumes"], hook)
    te = np.asarray(info["te_ms"])
    fe = run(S, te, cfg, mask3d)
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
    if mp:
        mpv = np.stack([mp[k][mask3d] for k in range(E)], axis=1)
        save("mppca_sigma", mpv)

    # ---- channel cache for later phases
    for name, arr in (("c0", fe["c"][:, 0]), ("c1", fe["c"][:, 1]), ("r0", fe["r"][:, 0]), ("r1", fe["r"][:, 1])):
        np.save(out_dir / f"{name}.npy", arr)
    np.save(out_dir / "R.npy", fe["R"])
    io.write_tsv(out_dir / "frames.tsv", {"frame": list(range(T)), "residual_energy": list(fe["frame_energy"]),
                                         "fd_mm": list(info["fd"]) if info["fd"] is not None and len(info["fd"]) == T else ["n/a"] * T})

    # ---- summary
    inf = inferior_third(mask3d, ref.affine)[mask3d]
    pct = lambda x, q=(10, 50, 90): [round(float(v), 4) for v in np.nanpercentile(x, q)]
    summ = {
        "run_id": info["run_id"], "source": source, "n_voxels": int(V), "n_volumes": int(T), "te_ms": info["te_ms"],
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
    if info["fd"] is not None and len(info["fd"]) == T:
        from scipy.stats import spearmanr
        summ["frame_energy_vs_fd_spearman"] = round(float(spearmanr(fe["frame_energy"], info["fd"])[0]), 4)
    if mp:
        exp = float(cfg["frontend"]["mppca_expected_ratio"])
        rows = []
        for k in range(E):
            sel = np.isfinite(mp[k][mask3d])
            ratio = float(np.nanmedian(fe["sigma"][sel]) / np.nanmedian(mp[k][mask3d][sel]))
            rows.append({"echo": k + 1, "mppca_sigma_median": float(np.nanmedian(mp[k][mask3d][sel])),
                         "residual_sigma_median_same_voxels": float(np.nanmedian(fe["sigma"][sel])),
                         "ratio_residual_over_mppca": round(ratio, 4),
                         "ratio_corrected_for_simulated_bias": round(ratio / exp, 4)})
        worst = max(abs(r["ratio_corrected_for_simulated_bias"] - 1) for r in rows)
        tol = float(cfg["frontend"]["mppca_sigma_tolerance"])
        summ["mppca_check"] = {"per_echo": rows, "max_abs_deviation": round(worst, 4), "tolerance": tol,
                               "verdict": "OK" if worst <= tol else "STOP: residual and MP-PCA sigma differ by more than the tolerance"}
    io.write_json(out_dir / "summary.json", summ)
    io.write_text(out_dir / "summary.md", render_summary(summ))
    io.write_config_copy(cfg, out_dir)
    inputs = {f"echo{k + 1}": f for k, f in enumerate(info["echo_files"])}
    inputs.update({"mask": info["mask"], "volumes_used": str(Path(run_dir) / "volumes_used.tsv")})
    io.write_json(out_dir / "done.json", {"stage": f"frontend_{source}", **io.provenance(cfg, inputs)})
    return summ


def render_summary(s: dict) -> str:
    L = [f"# Front end: {s['run_id']} ({s['source']} echoes)", "",
         f"{s['n_voxels']} brain voxels, {s['n_volumes']} volumes, TE {s['te_ms']} ms. "
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
    if "mppca_check" in s:
        m = s["mppca_check"]
        L += ["", "## Sigma cross-check: residual vs MP-PCA (raw echoes)", "",
              "| Echo | MP-PCA sigma | Residual sigma (same voxels) | Ratio | Ratio / simulated bias |",
              "|---|---|---|---|---|"]
        L += [f"| {r['echo']} | {r['mppca_sigma_median']:.3f} | {r['residual_sigma_median_same_voxels']:.3f} | "
              f"{r['ratio_residual_over_mppca']:.3f} | {r['ratio_corrected_for_simulated_bias']:.3f} |" for r in m["per_echo"]]
        L += ["", f"Verdict (tolerance {100 * m['tolerance']:.0f} %): **{m['verdict']}**"]
    return "\n".join(L) + "\n"
