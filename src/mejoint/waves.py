"""Traveling-wave extension, individual stage (docs/WAVES_BRIEF.md sections 4-6). A fresh implementation
(user decision 2026-11-03): no earlier CPCA code is used or consulted.

Pipeline, per participant, on the cached front-end channels (c0, c1 and the per-voxel QR factors R; the
per-echo noise weighting is inherited from the front end, never assumed equal):
  1. contiguous time: the longest block of consecutive kept volumes; gaps of <= max_interp_gap volumes are
     filled by linear interpolation and recorded (brief 4.2);
  2. slice-timing correction, an exact fractional delay in the Fourier domain: a voxel in a slice acquired
     delta seconds into the TR is shifted by exp(-i 2 pi f delta), with mirror padding against wrap-around
     (brief 4.3; the sign is verified on simulated data);
  3. zero-phase band-pass (Butterworth, both directions) and the analytic signal (scipy.signal.hilbert);
  4. scaling: noise-whitened units (default) or zscore_c0 (both channels of a voxel divided by the SD of
     that voxel's band-passed c0; keeps the c1 / c0 ratio) (brief 4.5);
  5. complex PCA of the stacked complex channels [c0 | c1] from the T x T Hermitian Gram matrix (complex128,
     numpy.linalg.eigh); no rotation (brief 4.6). Mode k has the time course z_k (unit RMS, positive
     frequencies) and the channel maps m = least-squares betas of each channel on z_k;
  6. complex back-substitution s = m1 / R11, q = (m0 - R01 s) / R00 (q in s^-1);
  7. global phase: the amplitude-weighted mean of q over the reference voxels is set to phase 0.

Conventions (checked in simulation, tests/test_waves.py):
  - lag tau (s) = -arg(q) / (2 pi f_c): with time courses ~ exp(+i w t), a voxel that is later by tau has map
    phase -w tau. The brief writes "lag = phase / (2 pi f_c)" but its gradient formula carries the minus
    sign; this implementation follows the gradient formula (decisions.md, W1).
  - lag gradient (no unwrapping): grad tau = -Im(conj(z) grad z) / (|z|^2 2 pi f_c), in s / mm, on the complex
    map smoothed with `gradient_smooth_mm`; the local propagation direction is grad tau / |grad tau|.
  - wave vector: k maximizing |sum_v w_v z_v exp(+i k . x_v)| (x in mm); speed = 2 pi f_c / |k|, direction
    = k / |k| (towards later voxels); a standing mode has k ~ 0.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

# ============================================================================ preprocessing


def contiguous_block(volumes, T: int, max_gap: int) -> dict:
    """Frames to use: the longest stretch of kept volumes whose gaps are <= max_gap (filled by interpolation).
    volumes: original frame numbers of the kept volumes (len T) or None (all consecutive)."""
    v = np.arange(T) if volumes is None else np.asarray(volumes, int)
    if v.size != T:
        raise ValueError(f"{v.size} volume indices for {T} frames")
    breaks = np.flatnonzero(np.diff(v) > max_gap + 1) + 1
    edges = np.r_[0, breaks, T]
    seg = max(zip(edges[:-1], edges[1:]), key=lambda ab: v[ab[1] - 1] - v[ab[0]])
    a, b = int(seg[0]), int(seg[1])
    grid = np.arange(v[a], v[b - 1] + 1)
    return {"cache_index": np.arange(a, b), "original": v[a:b], "grid": grid,
            "n_interpolated": int(grid.size - (b - a)), "n_segments_total": int(len(edges) - 1)}


def apply_block(X: np.ndarray, blk: dict) -> np.ndarray:
    """X (n, T_cache) -> (n, T_grid) on the regular grid, interpolating the filled gaps linearly."""
    sub = X[:, blk["cache_index"]]
    if blk["n_interpolated"] == 0:
        return sub
    out = np.empty((X.shape[0], blk["grid"].size), X.dtype)
    for i in range(X.shape[0]):
        out[i] = np.interp(blk["grid"], blk["original"], sub[i])
    return out


def fractional_delay(X: np.ndarray, delay_s: np.ndarray, tr: float) -> np.ndarray:
    """Shift each row by its delay (s): x(t) -> x(t - delay), exactly in the Fourier domain, with mirror
    padding (length 2T) against wrap-around."""
    T = X.shape[1]
    Xe = np.concatenate([X, X[:, ::-1]], axis=1)
    f = np.fft.rfftfreq(2 * T, tr)
    F = np.fft.rfft(Xe, axis=1) * np.exp(-2j * np.pi * f[None, :] * np.asarray(delay_s, float)[:, None])
    return np.fft.irfft(F, n=2 * T, axis=1)[:, :T]


def bandpass_analytic(X: np.ndarray, tr: float, band, order: int = 4) -> np.ndarray:
    sos = signal.butter(int(order), [float(band[0]), float(band[1])], btype="bandpass", fs=1.0 / tr, output="sos")
    Y = signal.sosfiltfilt(sos, X - X.mean(1, keepdims=True), axis=1)
    return signal.hilbert(Y, axis=1)


def phase_randomize(X: np.ndarray, rng) -> np.ndarray:
    """Real rows with independent random phases (each row's power spectrum kept)."""
    T = X.shape[1]
    F = np.fft.rfft(X, axis=1)
    ph = rng.uniform(0, 2 * np.pi, F.shape)
    ph[:, 0] = 0
    if T % 2 == 0:
        ph[:, -1] = 0
    return np.fft.irfft(F * np.exp(1j * ph), n=T, axis=1)


# ============================================================================ complex PCA


def cpca(Z: np.ndarray, n_modes: int, chunk: int = 20000) -> dict:
    """Complex PCA of rows Z (n, T) complex. Returns time courses (T, K) with unit RMS (positive frequency
    for analytic input), maps (n, K) = least-squares betas of each row on each time course, eigenvalues of
    Z^H Z / n and the variance fraction of each mode."""
    n, T = Z.shape
    G = np.zeros((T, T), np.complex128)
    for i in range(0, n, chunk):
        z = Z[i:i + chunk].astype(np.complex128)
        G += z.conj().T @ z
    lam, V = np.linalg.eigh(G)
    lam, V = lam[::-1].real, V[:, ::-1]
    K = int(min(n_modes, T))
    tc = np.conj(V[:, :K]) * np.sqrt(T)                                 # unit RMS
    maps = (np.asarray(Z, np.complex128) @ V[:, :K]) / np.sqrt(T)        # beta = sum_t x conj(z) / T
    return {"tc": tc, "maps": maps, "eig": lam[:K] / n, "var_frac": lam[:K] / lam.sum()}


def backsub_complex(m0, m1, R):
    R00, R01, R11 = R[:, 0, 0][:, None], R[:, 0, 1][:, None], R[:, 1, 1][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = m1 / R11
        q = (m0 - R01 * s) / R00 * 1000.0
    return s, q


def mode_frequency(z: np.ndarray, tr: float) -> float:
    """Power-weighted mean |frequency| of a complex time course."""
    F = np.fft.fft(z)
    f = np.abs(np.fft.fftfreq(z.size, tr))
    P = np.abs(F) ** 2
    return float((f * P).sum() / P.sum())


def complex_corr(a: np.ndarray, b: np.ndarray) -> complex:
    a = a - a.mean()
    b = b - b.mean()
    return complex(np.vdot(b, a) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def match_modes(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian matching of columns of B to columns of A on |complex correlation|. Returns (index into B for
    each A column, |corr|, the phase that rotates each matched B onto A)."""
    from scipy.optimize import linear_sum_assignment

    a = A - A.mean(0)
    b = B - B.mean(0)
    a = a / (np.linalg.norm(a, axis=0) + 1e-300)
    b = b / (np.linalg.norm(b, axis=0) + 1e-300)
    C = a.conj().T @ b                                                   # <a_i, b_j>
    r, c = linear_sum_assignment(-np.abs(C))
    idx = np.full(A.shape[1], -1)
    idx[r] = c
    cc = C[r, c][np.argsort(r)]
    return idx, np.abs(cc), np.angle(cc)


# ============================================================================ measures


def _smooth_complex(vol: np.ndarray, mask: np.ndarray, fwhm_mm: float, voxel_mm) -> np.ndarray:
    """Normalized-convolution Gaussian smoothing of a complex volume inside the mask."""
    if fwhm_mm <= 0:
        return np.where(mask, vol, 0)
    sig = [fwhm_mm / 2.3548 / float(v) for v in voxel_mm]
    m = mask.astype(float)
    num_r = ndimage.gaussian_filter(np.where(mask, vol.real, 0.0), sig)
    num_i = ndimage.gaussian_filter(np.where(mask, vol.imag, 0.0), sig)
    den = ndimage.gaussian_filter(m, sig)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mask & (den > 1e-6), (num_r + 1j * num_i) / den, 0)


def lag_gradient(qvol: np.ndarray, mask: np.ndarray, f_c: float, voxel_mm, fwhm_mm: float) -> np.ndarray:
    """grad tau (s / mm), (X, Y, Z, 3), from the complex map without unwrapping."""
    z = _smooth_complex(qvol, mask, fwhm_mm, voxel_mm)
    g = np.gradient(z, *[float(v) for v in voxel_mm])
    with np.errstate(invalid="ignore", divide="ignore"):
        gt = np.stack([-np.imag(np.conj(z) * gi) / (np.abs(z) ** 2 * 2 * np.pi * f_c) for gi in g], axis=-1)
    gt[~mask] = np.nan
    return gt


def _fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def wave_vector_fit(z: np.ndarray, x_mm: np.ndarray, f_c: float, speeds=(1.0, 300.0), n_dir: int = 400,
                    n_speed: int = 40) -> dict:
    """Maximize F(k) = |sum_v z_v exp(+i k . x_v)| / sum_v |z_v| over wave vectors (grid, then Nelder-Mead).
    Speed = 2 pi f_c / |k| (inf for k = 0), direction = k / |k|, fit quality = F."""
    from scipy.optimize import minimize

    w = 2 * np.pi * f_c
    x = x_mm - x_mm.mean(0)
    norm = np.abs(z).sum()
    if norm == 0:
        return {"speed_mm_s": np.nan, "direction": [np.nan] * 3, "fit_quality": np.nan, "k": [0, 0, 0]}
    D = _fib_sphere(n_dir)
    mags = w / np.geomspace(speeds[0], speeds[1], n_speed)
    best = (np.abs(z.sum()) / norm, np.zeros(3))
    for kmag in mags:
        K = D * kmag                                                    # (n_dir, 3)
        F = np.abs(np.exp(1j * (x @ K.T)).T @ z) / norm                 # (n_dir,)
        j = int(np.argmax(F))
        if F[j] > best[0]:
            best = (float(F[j]), K[j].copy())
    obj = lambda k: -np.abs(np.exp(1j * (x @ k)) @ z) / norm
    res = minimize(obj, best[1], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 400})
    k = res.x if -res.fun >= best[0] else best[1]
    q = float(-obj(k))
    kn = float(np.linalg.norm(k))
    return {"speed_mm_s": float(w / kn) if kn > 1e-12 else float("inf"),
            "direction": (k / kn).tolist() if kn > 1e-12 else [0.0, 0.0, 0.0], "fit_quality": q, "k": k.tolist()}


def directional_coherence(grad: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """R = |weighted mean of the unit gradient vectors| (3D); also the unit vectors used."""
    g = grad.reshape(-1, 3)
    w = weights.reshape(-1)
    n = np.linalg.norm(g, axis=1)
    ok = np.isfinite(n) & (n > 0) & (w > 0)
    u = g[ok] / n[ok, None]
    return float(np.linalg.norm((w[ok, None] * u).sum(0)) / w[ok].sum()), u


def te_lag(s: np.ndarray, q_per_s: np.ndarray, te_ms, f_c: float) -> np.ndarray:
    """Per voxel lag (s) per echo from the echo-domain amplitude z_e = s - TE_e q (log-ratio units),
    lag_e = -arg(z_e) / (2 pi f_c), unwrapped across echoes. Returns (V, E)."""
    te = np.asarray(te_ms, float) / 1000.0
    ze = s[:, None] - te[None, :] * q_per_s[:, None]
    ph = np.unwrap(np.angle(ze), axis=1)
    return -ph / (2 * np.pi * f_c)


def complex_energies(m0, m1, R, vb):
    """eta with |.|^2 (CLAUDE.md 7.1) for complex maps; vb = beta noise variance E|beta_noise|^2 per voxel
    and mode (V, K) or broadcastable."""
    R00, R01, R11 = R[:, 0, 0][:, None], R[:, 0, 1][:, None], R[:, 1, 1][:, None]
    ok = np.isfinite(m0) & np.isfinite(m1) & (R11 > 0)
    vb = np.broadcast_to(np.asarray(vb, float), m0.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        ws = (R01 ** 2 + R11 ** 2) / R11 ** 2
        wr = 1 + R01 ** 2 / R11 ** 2
        r2 = m0 - R01 * m1 / R11
    z = lambda a: np.where(ok, a, 0.0)
    e_s0 = z(ws * np.abs(m1) ** 2 - ws * vb)
    e_r2 = z(np.abs(r2) ** 2 - wr * vb)
    return e_s0, e_r2


# ============================================================================ driver


def prepare(c0, c1, vox_slice_offset_s, tr, W: dict, blk: dict, Q=None):
    """Contiguous time, slice-timing correction, band-pass, analytic signal and scaling of both channels.
    Returns complex (V, T) arrays and the per-voxel scale (1 for whitened units)."""
    out = []
    offs = np.broadcast_to(np.asarray(vox_slice_offset_s, float), (c0.shape[0],))
    CH = 20000                                                          # rows per chunk (memory)
    for c in (c0, c1):
        A = np.empty((c.shape[0], blk["grid"].size), np.complex128)
        for i in range(0, c.shape[0], CH):
            X = apply_block(np.nan_to_num(np.asarray(c[i:i + CH], np.float64)), blk)
            if W.get("slice_timing_correction", True):
                # a slice acquired delta into the TR samples u(t + delta): delay it by +delta (exp(-i 2 pi f delta));
                # verified in simulation (lag vs slice-time slope -1 uncorrected, 0 corrected; -2 with the wrong sign)
                X = fractional_delay(X, offs[i:i + CH], tr)
            A[i:i + CH] = bandpass_analytic(X, tr, W["band_hz"], int(W.get("filter_order", 4)))
        out.append(A)
    a0, a1 = out
    a1[~np.isfinite(c1).all(1)] = np.nan
    scale = np.ones(c0.shape[0])
    if W.get("voxel_scaling", "whitened") == "zscore_c0":
        sd = np.real(a0).std(1)
        scale = np.where(sd > 0, sd, 1.0)
        a0, a1 = a0 / scale[:, None], a1 / scale[:, None]
    return a0, a1, scale


def thermal_beta_variance(tcs: np.ndarray, blk: dict, T_cache: int, offsets_s, tr, W, n_rows: int, rng,
                          Q=None) -> np.ndarray:
    """E|beta|^2 for unit white thermal noise passed through exactly the same preprocessing (Monte Carlo,
    including the nuisance projection Q if the channels were projected): the noise-debiasing constant of
    complex eta (brief 4.7 item 6: calibrated, not the real-valued constant)."""
    noise = rng.standard_normal((int(n_rows), int(T_cache)))
    if Q is not None:
        noise = noise - (noise @ Q) @ Q.T
    off = np.resize(np.asarray(offsets_s), int(n_rows))
    a, _, _ = prepare(noise, noise, off, tr, {**W, "voxel_scaling": "whitened"}, blk)
    Tg = a.shape[1]
    beta = a @ np.conj(tcs) / Tg
    return (np.abs(beta) ** 2).mean(0)                                 # (K,)


def run(c0, c1, R, mask3d, voxel_mm, te_ms, tr, cfg: dict, slice_offsets_s=None, volumes=None, ref=None,
        seed=None, Q=None, channels: str = "both", bootstrap: int | None = None, surrogates: int | None = None) -> dict:
    """Individual stage on arrays (V aligned with mask3d in C order). ref: (V,) bool reference voxels for the
    global phase and the speed statistics (data-only tissue class or the mask). Returns modes and measures."""
    W = cfg["waves"]
    rng = np.random.default_rng(seed)
    V, T0 = c0.shape
    coords = np.argwhere(mask3d)
    x_mm = coords * np.asarray(voxel_mm, float)
    offs = np.zeros(V) if slice_offsets_s is None else np.asarray(slice_offsets_s, float)[coords[:, 2]]
    blk = contiguous_block(volumes, T0, int(W.get("max_interp_gap", 2)))
    Tg = blk["grid"].size
    if Tg < int(W.get("min_contiguous_volumes", 400)):
        return {"skipped": f"longest contiguous block {Tg} < min_contiguous_volumes {W.get('min_contiguous_volumes', 400)}",
                "block": blk}
    a0, a1, scale = prepare(c0, c1, offs, tr, W, blk, Q)
    ok1 = np.isfinite(a1).all(1)
    rows = [a0] if channels == "c0" else [a0, a1[ok1]]
    Z = np.concatenate(rows, axis=0)
    K = int(W.get("n_modes", 10))
    pc = cpca(Z, K)
    tc = pc["tc"]
    m0 = pc["maps"][:V]
    m1 = np.full((V, K), np.nan, complex)
    if channels != "c0":
        m1[ok1] = pc["maps"][V:]
    # back to whitened units for back-substitution when the zscore_c0 scaling was used
    m0w, m1w = m0 * scale[:, None], m1 * scale[:, None]
    s, q = backsub_complex(m0w, m1w, R) if channels != "c0" else (np.full_like(m0, np.nan), m0w)
    refv = np.ones(V, bool) if ref is None else np.asarray(ref, bool)
    # global phase convention: amplitude-weighted mean of q over the reference voxels -> phase 0
    alpha = np.zeros(K)
    for k in range(K):
        qq = q[refv, k]
        qq = qq[np.isfinite(qq)]
        alpha[k] = np.angle(np.sum(np.abs(qq) * qq)) if qq.size else 0.0
    rot = np.exp(-1j * alpha)
    m0, m1, m0w, m1w, s, q = (a * rot[None, :] for a in (m0, m1, m0w, m1w, s, q))
    tc = tc * np.conj(rot)[None, :]
    fc = np.array([mode_frequency(tc[:, k], tr) for k in range(K)])
    # cross-fitted maps (eta, TE lag): time courses from the other voxel half (Phase 4 lesson: modes are
    # chosen from the voxels' own noise)
    from .rank import acquisition_axes, split_halves
    half = split_halves(np.arange(V), mask3d, int(cfg["rank"].get("split_block_vox", 4)), rng,
                        slab_axis=acquisition_axes(cfg)["readout"])
    m0c = np.full((V, K), np.nan, complex)
    m1c = np.full((V, K), np.nan, complex)
    for est, use in ((half, ~half), (~half, half)):
        rows_h = [a0[est]] + ([a1[est & ok1]] if channels != "c0" else [])
        ph = cpca(np.concatenate(rows_h), K)
        idx, _, phs = match_modes(tc, ph["tc"])
        tch = ph["tc"][:, idx] * np.exp(1j * phs)[None, :]
        Gi = np.linalg.inv(np.conj(tch.conj().T @ tch) / Tg)
        P = np.conj(tch) @ Gi / Tg                    # complex GLM: beta = x conj(Z) (Z^T conj(Z))^-1
        m0c[use] = (a0[use] @ P) * scale[use, None]
        if channels != "c0":
            u1 = use & ok1
            m1c[u1] = (a1[u1] @ P) * scale[u1, None]
    vb_unit = thermal_beta_variance(tc, blk, T0, offs, tr, W, int(W.get("noise_mc_rows", 2000)), rng, Q)   # (K,)
    vb = vb_unit[None, :] * np.ones((V, 1))                              # whitened units: same for every voxel
    out = {"tc": tc, "m0": m0w, "m1": m1w, "s": s, "q": q, "m0_cf": m0c, "m1_cf": m1c, "f_c": fc,
           "var_frac_stacked": pc["var_frac"], "eig": pc["eig"], "phase_rotation": alpha, "block": blk,
           "n_volumes_used": int(Tg), "vb_thermal": vb_unit, "channels": channels,
           "slice_timing_corrected": bool(W.get("slice_timing_correction", True)) and slice_offsets_s is not None,
           "voxel_scaling": W.get("voxel_scaling", "whitened")}
    # variance fraction in c0 alone
    c0_energy = np.sum(np.abs(a0) ** 2)
    out["var_frac_c0"] = np.array([np.sum(np.abs(m0[:, k]) ** 2) * Tg / c0_energy for k in range(K)])
    # per-mode measures
    out["_offsets"] = offs if out["slice_timing_corrected"] or slice_offsets_s is not None else None
    out["measures"] = [mode_measures(out, k, mask3d, x_mm, voxel_mm, te_ms, R, refv, vb, cfg, rng) for k in range(K)]
    # energy null: phase-randomized surrogates through the same pipeline, family-wise (largest eigenvalue)
    n_sur = int(W.get("surrogates", 200) if surrogates is None else surrogates)
    if n_sur > 0:
        out["null"] = energy_null(c0, c1, offs, tr, W, blk, K, n_sur, rng, channels, pc["eig"])
        for k, msr in enumerate(out["measures"]):
            msr["above_null"] = bool(pc["eig"][k] > out["null"]["threshold"])
    n_boot = int(W.get("bootstrap", 200) if bootstrap is None else bootstrap)
    if n_boot > 0:
        out["bootstrap"] = bootstrap_modes(a0, a1, ok1, scale, R, out, mask3d, x_mm, voxel_mm, te_ms, refv, cfg, n_boot,
                                           rng, channels)
    return out


def mode_measures(out, k, mask3d, x_mm, voxel_mm, te_ms, R, refv, vb, cfg, rng) -> dict:
    W = cfg["waves"]
    q, s, fc = out["q"][:, k], out["s"][:, k], float(out["f_c"][k])
    V = q.size
    amp = np.abs(q)
    okq = np.isfinite(q)
    # voxels where the mode is present: |q| above `amp_z` noise SDs of q (from the calibrated beta noise)
    R00, R01, R11 = R[:, 0, 0], R[:, 0, 1], R[:, 1, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sd_q = np.sqrt(vb[:, k] * (1 + R01 ** 2 / R11 ** 2)) / R00 * 1000.0
    hi = refv & okq & (amp > float(W.get("amp_z", 3.0)) * sd_q)
    if hi.sum() < 20:                                                    # too few: the strongest reference voxels
        hi = refv & okq & (amp >= np.nanpercentile(amp[refv & okq], 90))
    # lag
    lag = -np.angle(q) / (2 * np.pi * fc)
    # gradient
    qv = np.zeros(mask3d.shape, complex)
    qv[mask3d] = np.nan_to_num(q)
    grad = lag_gradient(qv, mask3d, fc, voxel_mm, float(W.get("gradient_smooth_mm", 4.0)))[mask3d]    # (V, 3)
    gn = np.linalg.norm(grad, axis=1)
    local_speed = np.where(gn > 0, 1.0 / gn, np.nan)
    R_coh, _ = directional_coherence(grad[hi], amp[hi] ** 2)
    # wave vector (high-amplitude reference voxels)
    wv = wave_vector_fit(np.nan_to_num(q[hi]), x_mm[hi], fc)
    # eta (cross-fitted, debiased)
    e_s0, e_r2 = complex_energies(out["m0_cf"][:, [k]], out["m1_cf"][:, [k]], R, vb[:, [k]])
    E_S0, E_R2 = float(e_s0.sum()), float(e_r2.sum())
    eta = E_S0 / (E_S0 + E_R2) if E_S0 + E_R2 > 0 else np.nan
    # TE-dependent lag on cross-fitted maps, with a per-voxel noise null for a pure dR2* voxel
    s_c, q_c = backsub_complex(out["m0_cf"][:, [k]], out["m1_cf"][:, [k]], R)
    tl = te_lag(s_c[:, 0], q_c[:, 0], te_ms, fc)
    dlag = tl[:, -1] - tl[:, 0]
    null_d, null_draws = te_lag_null(q_c[:, 0], R, vb[:, k], te_ms, fc, rng, int(W.get("te_lag_null_draws", 20)))
    beyond = np.abs(dlag) > null_d
    sel = hi & np.isfinite(dlag)
    # matched-SNR eta bands: pure-dR2* and pure-dS0 maps with this mode's footprint and noise
    bands = eta_bands(out["m0_cf"][:, k], out["m1_cf"][:, k], R, vb[:, k], E_S0 + E_R2, rng, int(W.get("band_mc", 50)))
    # slice-timing check: lag regressed on the slice acquisition offset (slope -1 = uncorrected, 0 = corrected)
    off = out.get("_offsets")
    slope = np.nan
    if off is not None and np.ptp(off[hi]) > 0:
        lg = lag[hi]
        slope = float(np.polyfit(off[hi], lg - np.median(lg), 1)[0])
    # path profile along the fitted propagation axis (figures; bootstrap bands use the same bins)
    prof = path_profile(q, s, out["m0_cf"][:, k], out["m1_cf"][:, k], R, vb[:, k], x_mm, hi, wv["direction"], fc,
                        te_ms, float(W.get("path_bin_mm", 6.0)))
    nd = null_draws[:, sel].ravel()
    nd = nd[np.isfinite(nd)]
    if nd.size > 20000:
        nd = rng.choice(nd, 20000, replace=False)
    return {"mode": k + 1, "f_c_hz": fc, "var_frac_stacked": float(out["var_frac_stacked"][k]),
            "var_frac_c0": float(out["var_frac_c0"][k]),
            "speed_mm_s": wv["speed_mm_s"], "direction": wv["direction"], "fit_quality": wv["fit_quality"],
            "local_speed_median": float(np.nanmedian(local_speed[hi])) if hi.any() else np.nan,
            "local_speed_iqr": [float(x) for x in np.nanpercentile(local_speed[hi], [25, 75])] if hi.any() else [np.nan, np.nan],
            "R": R_coh, "eta": eta, "E_S0": E_S0, "E_R2": E_R2, "eta_band_R2": bands["R2"], "eta_band_S0": bands["S0"],
            "lag_vs_slice_time_slope": slope, "n_voxels_present": int(hi.sum()),
            "te_lag_median_s": float(np.nanmedian(dlag[sel])) if sel.any() else np.nan,
            "te_lag_fraction_beyond_null": float(np.mean(beyond[sel])) if sel.any() else np.nan,
            "te_lag_median_beyond_null_s": float(np.nanmedian(dlag[sel & beyond])) if (sel & beyond).sum() >= 5 else np.nan,
            "_null_d": null_d, "_null_dlag_draws": nd, "_profile": prof,
            "_lag": lag, "_grad": grad, "_dlag": dlag, "_te_lags": tl, "_hi": hi, "_beyond": beyond}


def profile_lags(q, s, bin_index, n_bins, te_ms, fc):
    """Per-bin, per-echo lag (s): amplitude-weighted circular mean of z_e = s - TE_e q over the bin's voxels,
    unwrapped along the path at TE1, the other echoes placed within half a cycle of TE1, referenced to the
    first bin at TE1. (B, E), NaN for empty bins."""
    te = np.asarray(te_ms, float) / 1000.0
    s0 = np.where(np.isfinite(s), s, 0.0)
    ze = s0[:, None] - te[None, :] * np.nan_to_num(q)[:, None]
    use = bin_index >= 0
    Zb = np.zeros((n_bins, te.size), complex)
    np.add.at(Zb, bin_index[use], ze[use])
    ok = np.abs(Zb[:, 0]) > 0
    ph = np.full(Zb.shape, np.nan)
    if ok.sum() == 0:
        return ph
    p1 = np.unwrap(np.angle(Zb[ok, 0]))
    dp = np.angle(Zb[ok] * np.conj(Zb[ok, :1]))                        # each echo relative to TE1, in (-pi, pi]
    ph[ok] = p1[:, None] + dp
    lags = -ph / (2 * np.pi * fc)
    return lags - lags[np.flatnonzero(ok)[0], 0]


def path_profile(q, s, m0c, m1c, R, vb_k, x_mm, hi, direction, fc, te_ms, bin_mm) -> dict:
    """Voxels present in the mode (hi), projected on the fitted propagation axis and binned by distance from
    the earliest projection: per-bin lag per echo, eta (cross-fitted, debiased) and the voxel count."""
    d = np.asarray(direction, float)
    V = q.size
    bin_index = np.full(V, -1)
    if not np.isfinite(d).all() or np.linalg.norm(d) == 0 or hi.sum() < 5:
        return {"bin_index": bin_index, "centres_mm": np.zeros(0), "lags": np.zeros((0, len(te_ms))),
                "eta": np.zeros(0), "n": np.zeros(0, int)}
    proj = x_mm @ (d / np.linalg.norm(d))
    p0 = proj[hi].min()
    b = np.floor((proj - p0) / bin_mm).astype(int)
    bin_index[hi] = b[hi]
    nb = int(b[hi].max()) + 1
    n = np.bincount(bin_index[hi], minlength=nb)
    bin_index[hi & (n[np.clip(b, 0, nb - 1)] < 5)] = -1                  # bins with fewer than 5 voxels are dropped
    n = np.bincount(bin_index[bin_index >= 0], minlength=nb)
    lags = profile_lags(q, s, bin_index, nb, te_ms, fc)
    lags[n < 5] = np.nan
    e_s0, e_r2 = complex_energies(m0c[:, None], m1c[:, None], R, np.asarray(vb_k, float)[:, None])
    es = np.bincount(bin_index[bin_index >= 0], e_s0[bin_index >= 0, 0], minlength=nb)
    er = np.bincount(bin_index[bin_index >= 0], e_r2[bin_index >= 0, 0], minlength=nb)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where((n >= 5) & (es + er > 0), es / (es + er), np.nan)
    return {"bin_index": bin_index, "centres_mm": (np.arange(nb) + 0.5) * bin_mm, "lags": lags, "eta": eta, "n": n}


def voxel_eta(m0c, m1c, R, vb_k, mask3d, voxel_mm, fwhm_mm: float) -> np.ndarray:
    """Voxelwise eta for display: the debiased S0 and R2* energies (cross-fitted) smoothed with a Gaussian of
    `fwhm_mm` inside the mask, then eta = E_S0 / (E_S0 + E_R2), clipped to [0, 1]; NaN where the smoothed total
    is not positive. Smoothing is for viewing only; every reported eta uses unsmoothed energies."""
    e_s0, e_r2 = complex_energies(m0c[:, None], m1c[:, None], R, np.asarray(vb_k, float)[:, None])
    sd = np.asarray(fwhm_mm, float) / 2.3548 / np.asarray(voxel_mm, float)
    def sm(x):
        v = np.zeros(mask3d.shape)
        v[mask3d] = x
        w = ndimage.gaussian_filter(mask3d.astype(float), sd)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (ndimage.gaussian_filter(v, sd) / w)[mask3d]
    es, er = sm(e_s0[:, 0]), sm(e_r2[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(es + er > 0, np.clip(es / (es + er), 0, 1), np.nan)


def eta_bands(m0, m1, R, vb_k, E, rng, n_mc: int = 50) -> dict:
    """eta intervals (2.5-97.5 %) for pure-dR2* and pure-dS0 complex maps with the mode's footprint, R and
    debiased energy E, plus complex beta noise of variance vb (matched SNR; thermal noise only)."""
    R00, R01, R11 = R[:, 0, 0], R[:, 0, 1], R[:, 1, 1]
    ok = np.isfinite(m0) & np.isfinite(m1) & (R11 > 0)
    if E <= 0 or ok.sum() < 10:
        return {"R2": [np.nan, np.nan], "S0": [np.nan, np.nan]}
    r2 = (m0 - R01 * m1 / R11)[ok]
    sm = (m1 / R11)[ok]
    a = np.sqrt(E / max(np.sum(np.abs(r2) ** 2), 1e-300))
    b = np.sqrt(E / max(np.sum((R01[ok] ** 2 + R11[ok] ** 2) * np.abs(sm) ** 2), 1e-300))
    sd = np.sqrt(np.asarray(vb_k, float)[ok] / 2)
    Rk = R[ok]
    out = {"R2": [], "S0": []}
    for _ in range(int(n_mc)):
        n0 = sd * (rng.standard_normal(ok.sum()) + 1j * rng.standard_normal(ok.sum()))
        n1 = sd * (rng.standard_normal(ok.sum()) + 1j * rng.standard_normal(ok.sum()))
        for name, (x0, x1) in (("R2", (a * r2, 0 * r2)), ("S0", (R01[ok] * b * sm, R11[ok] * b * sm))):
            es, er = complex_energies((x0 + n0)[:, None], (x1 + n1)[:, None], Rk, np.asarray(vb_k, float)[ok][:, None])
            tot = es.sum() + er.sum()
            out[name].append(es.sum() / tot if tot > 0 else np.nan)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in out.items()}


def te_lag_null(q_per_s, R, vb_k, te_ms, fc, rng, n_draws: int) -> np.ndarray:
    """Per voxel: the 95th percentile of |TE4 - TE1 lag| that noise alone produces around a pure dR2* map with
    the voxel's own q (beta noise variance vb in whitened channel units), and the signed draws (n_draws, V)."""
    R00, R01, R11 = R[:, 0, 0], R[:, 0, 1], R[:, 1, 1]
    sd = np.sqrt(np.asarray(vb_k, float) / 2)                           # complex: half the variance per part
    m0 = R00 * q_per_s / 1000.0
    out = []
    for _ in range(int(n_draws)):
        n0 = sd * (rng.standard_normal(q_per_s.size) + 1j * rng.standard_normal(q_per_s.size))
        n1 = sd * (rng.standard_normal(q_per_s.size) + 1j * rng.standard_normal(q_per_s.size))
        s_, q_ = backsub_complex((m0 + n0)[:, None], n1[:, None], R)
        tl = te_lag(s_[:, 0], q_[:, 0], te_ms, fc)
        out.append(tl[:, -1] - tl[:, 0])
    out = np.array(out)
    return np.nanpercentile(np.abs(out), 95, axis=0), out


def energy_null(c0, c1, offs, tr, W, blk, K, n_sur, rng, channels, eig) -> dict:
    """Family-wise mode-energy null: the largest eigenvalue of phase-randomized surrogates (rows randomized
    independently: each row's spectrum kept, shared timing destroyed) through the same preprocessing."""
    tops = []
    V = c0.shape[0]
    for _ in range(int(n_sur)):
        s0 = phase_randomize(np.nan_to_num(c0.astype(np.float64)), rng)
        s1 = phase_randomize(np.nan_to_num(c1.astype(np.float64)), rng)
        s1[~np.isfinite(c1).all(1)] = np.nan
        a0, a1, _ = prepare(s0, s1, offs, tr, W, blk)
        ok1 = np.isfinite(a1).all(1)
        Z = a0 if channels == "c0" else np.concatenate([a0, a1[ok1]])
        tops.append(cpca(Z, 1)["eig"][0])
    q = float(W.get("null_quantile", 0.95))
    return {"threshold": float(np.quantile(tops, q)), "quantile": q, "n_surrogates": int(n_sur)}


def bootstrap_modes(a0, a1, ok1, scale, R, out, mask3d, x_mm, voxel_mm, te_ms, refv, cfg, n_boot, rng, channels) -> dict:
    """Circular block bootstrap over time (blocks of at least one cycle of the slowest reported mode): the
    complex PCA is rerun, modes matched by |complex correlation| of their maps, phases aligned, and the
    scalars recomputed. Returns 95 % intervals per mode."""
    W = cfg["waves"]
    V, T = a0.shape
    K = out["tc"].shape[1]
    tr_cycles = 1.0 / max(float(np.min(out["f_c"])), 1e-3)
    tr = float(cfg["acquisition"]["tr_s"])
    L = int(np.ceil(tr_cycles / tr))
    L = max(1, min(L, T // 2))
    ref_maps = np.concatenate([out["m0"], np.where(np.isfinite(out["m1"]), out["m1"], 0)], axis=0)
    keys = ("speed_mm_s", "fit_quality", "R", "var_frac_stacked", "te_lag_median_s", "te_lag_median_beyond_null_s")
    vals = {kk: [] for kk in keys}
    prof_vals = []
    KB = int(min(K, W.get("bootstrap_modes", W.get("plot_modes", 3))))   # modes with intervals (cost)
    for _ in range(int(n_boot)):
        starts = rng.integers(0, T, int(np.ceil(T / L)))
        idx = np.concatenate([(np.arange(L) + s0) % T for s0 in starts])[:T]
        rows = [a0[:, idx]] + ([a1[ok1][:, idx]] if channels != "c0" else [])
        pc = cpca(np.concatenate(rows), K)
        m0b = pc["maps"][:V] * scale[:, None]
        m1b = np.full((V, K), np.nan, complex)
        if channels != "c0":
            m1b[ok1] = pc["maps"][V:] * scale[ok1, None]
        bm = np.concatenate([m0b, np.where(np.isfinite(m1b), m1b, 0)], axis=0)
        j, _, ph = match_modes(ref_maps, bm)
        rot = np.exp(1j * ph)
        m0b, m1b = m0b[:, j] * rot[None, :], m1b[:, j] * rot[None, :]
        s_b, q_b = backsub_complex(m0b, m1b, R) if channels != "c0" else (np.full_like(m0b, np.nan), m0b)
        row = {kk: [] for kk in keys}
        prow = []
        for k in range(KB):
            fc = float(out["f_c"][k])
            hi = out["measures"][k]["_hi"]
            qk = q_b[:, k]
            wv = wave_vector_fit(np.nan_to_num(qk[hi]), x_mm[hi], fc, n_dir=150, n_speed=25)
            qv = np.zeros(mask3d.shape, complex)
            qv[mask3d] = np.nan_to_num(qk)
            grad = lag_gradient(qv, mask3d, fc, voxel_mm, float(W.get("gradient_smooth_mm", 4.0)))[mask3d]
            Rb, _ = directional_coherence(grad[hi], np.abs(qk[hi]) ** 2)
            tl = te_lag(s_b[:, k], qk, te_ms, fc)
            d = (tl[:, -1] - tl[:, 0])[hi]
            row["speed_mm_s"].append(wv["speed_mm_s"])
            row["fit_quality"].append(wv["fit_quality"])
            row["R"].append(Rb)
            row["var_frac_stacked"].append(float(pc["var_frac"][j[k]]) if j[k] >= 0 else np.nan)
            row["te_lag_median_s"].append(float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan)
            dd = (tl[:, -1] - tl[:, 0])
            bey = hi & (np.abs(dd) > out["measures"][k]["_null_d"])
            row["te_lag_median_beyond_null_s"].append(float(np.nanmedian(dd[bey])) if bey.sum() >= 5 else np.nan)
            pr = out["measures"][k]["_profile"]
            nb = pr["lags"].shape[0]
            if nb:
                lb = profile_lags(qk, s_b[:, k], pr["bin_index"], nb, te_ms, fc)
                # place the resample within half a cycle of the full-data profile, bin by bin
                per = 1.0 / fc
                lb = lb + per * np.round((pr["lags"] - lb) / per)
                prow.append(lb)
            else:
                prow.append(pr["lags"])
        for kk in keys:
            vals[kk].append(row[kk])
        prof_vals.append(prow)
    ci = {kk: np.nanpercentile(np.array(v, float), [2.5, 97.5], axis=0).T.tolist() for kk, v in vals.items()}
    prof_ci = []
    for k in range(KB):
        arr = np.array([pv[k] for pv in prof_vals], float)               # (n_boot, B, E)
        with np.errstate(all="ignore"):
            prof_ci.append(np.nanpercentile(arr, [2.5, 97.5], axis=0) if arr.size else np.zeros((2, 0, len(te_ms))))
    return {"n": int(n_boot), "block_frames": L, "ci": ci, "profile_ci": prof_ci}


# ============================================================================ folder interface (brief section 6)


def find_slice_timing(frontend_dir) -> tuple[np.ndarray | None, str | None]:
    """SliceTiming (s) of the echo-1 acquisition behind a front-end output: the echo-1 sidecar recorded in the
    front end's done.json, else the raw echo 1 named in the pipeline's result.json. (None, None) if absent."""
    import json
    import re
    from pathlib import Path

    fd = Path(frontend_dir)
    cands = []
    try:
        inp = json.loads((fd / "done.json").read_text()).get("inputs", {})
    except (OSError, ValueError):
        inp = {}
    e1 = (inp.get("echo1") or {}).get("path")
    if e1:
        cands.append(Path(e1))
    vu = (inp.get("volumes_used") or {}).get("path")
    if vu and (Path(vu).parent / "result.json").exists():
        res = json.loads((Path(vu).parent / "result.json").read_text())
        if res.get("input_echo1"):
            cands.append(Path(res["input_echo1"]))
    for c in cands:
        js = Path(re.sub(r"\.nii(\.gz)?$", ".json", str(c)))
        if js.exists():
            st = json.loads(js.read_text()).get("SliceTiming")
            if st:
                return np.asarray(st, float), str(js)
    return None, None


def find_volumes(frontend_dir, T: int):
    """Original indices of the T frames (echo-1 column of the pipeline's volumes_used.tsv), so gaps inside a run
    are known; None (taken as consecutive) when it is missing or does not match."""
    import json
    from pathlib import Path

    try:
        inp = json.loads((Path(frontend_dir) / "done.json").read_text()).get("inputs", {})
        vu = np.loadtxt(inp["volumes_used"]["path"], skiprows=1, ndmin=2).astype(int)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return vu[:, 0] if vu.shape[0] == T else None


def run_dir(frontend_dir, cfg: dict, out_dir, tissue_dir=None, figures: bool = True, slice_timing: bool | None = None,
            n_jobs: int | None = None, data_type: str | None = None) -> dict:
    """Individual wave stage on a cached front-end folder (c0/c1.npy, R.npy, analysis mask, nuisance_Q.npy when
    projected). Writes <out_dir>/: modes.npz, modes.tsv, maps/*.nii.gz, figures/, waves_summary.md, done.json.
    This is the single code path behind `mejoint waves` and `mejoint decompose --waves`."""
    import json
    import re
    from pathlib import Path

    import nibabel as nib

    from . import io
    from .frontend import inferior_third

    fd, out_dir = Path(frontend_dir), Path(out_dir)
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)
    W = cfg["waves"]
    if slice_timing is not None:
        cfg = io.merge(cfg, {"waves": {"slice_timing_correction": bool(slice_timing)}})
        W = cfg["waves"]
    c0, c1 = np.load(fd / "c0.npy"), np.load(fd / "c1.npy")
    R = np.load(fd / "R.npy")
    mimg = nib.load(str(fd / "analysis_mask.nii.gz"))
    mask3d = np.asarray(mimg.dataobj) > 0
    voxel_mm = tuple(float(x) for x in mimg.header.get_zooms()[:3])
    axcodes = nib.aff2axcodes(mimg.affine)
    Q = np.load(fd / "nuisance_Q.npy") if (fd / "nuisance_Q.npy").exists() else None
    fs = json.loads((fd / "summary.json").read_text())
    te_ms = np.asarray(fs.get("te_ms") or cfg["acquisition"]["te_ms"], float)
    tr = float(cfg["acquisition"]["tr_s"])
    m = re.match(r"sub-(\d+)", str(fs.get("run_id") or ""))
    subj = int(m.group(1)) if m else 0
    offsets, st_src = find_slice_timing(fd)
    if W.get("slice_timing_correction", True):
        if offsets is None:
            raise FileNotFoundError("slice-timing correction is on but no SliceTiming was found for this run "
                                    "(echo-1 JSON sidecar); use --no-slice-timing to run without it")
        if offsets.size != mask3d.shape[2]:
            raise ValueError(f"SliceTiming has {offsets.size} entries but the volume has {mask3d.shape[2]} slices")
    else:
        offsets = None
    volumes = find_volumes(fd, c0.shape[1])
    ref, ref_name = None, "analysis mask"
    if tissue_dir is not None and (Path(tissue_dir) / "classes.npy").exists():
        cls = np.load(Path(tissue_dir) / "classes.npy")
        if cls.shape[0] == c0.shape[0] and (cls == 1).sum() >= 100:
            ref, ref_name = cls == 1, "data-only tissue class 'tissue'"
    seed = io.seed_sequence(cfg, subj, 0, 6, 0)                             # stage 6 = waves
    channels = W.get("channels", "both")
    out = run(c0, c1, R, mask3d, voxel_mm, te_ms, tr, cfg, slice_offsets_s=offsets, volumes=volumes, ref=ref,
              seed=seed, Q=Q, channels=channels)
    if "skipped" in out:
        summ = {"run_id": fs.get("run_id"), "skipped": out["skipped"]}
        io.write_json(out_dir / "waves.json", summ)
        io.write_text(out_dir / "waves_summary.md", f"# Waves: {fs.get('run_id')}\n\nSkipped: {out['skipped']}\n")
        return summ
    K = out["q"].shape[1]
    refv = np.ones(c0.shape[0], bool) if ref is None else ref
    fwhm = float(W.get("eta_display_fwhm_mm", 6.0))
    eta_vox = np.stack([voxel_eta(out["m0_cf"][:, k], out["m1_cf"][:, k], R, out["vb_thermal"][k] * np.ones(c0.shape[0]),
                                  mask3d, voxel_mm, fwhm) for k in range(K)], axis=1)
    lag = np.stack([msr["_lag"] for msr in out["measures"]], axis=1)
    # maps
    for name, arr in (("q_amp", np.abs(out["q"])), ("q_phase", np.angle(out["q"])), ("s_amp", np.abs(out["s"])),
                      ("s_phase", np.angle(out["s"])), ("eta", eta_vox), ("lag", lag)):
        io.save_map(arr.astype(np.float32), mask3d, mimg, out_dir / "maps" / f"{name}.nii.gz")
    np.savez_compressed(out_dir / "modes.npz", m0=out["m0"], m1=out["m1"], s=out["s"], q=out["q"], tc=out["tc"],
                        f_c=out["f_c"], phase_rotation=out["phase_rotation"], m0_cf=out["m0_cf"], m1_cf=out["m1_cf"],
                        eig=out["eig"], var_frac_stacked=out["var_frac_stacked"], block_grid=out["block"]["grid"],
                        phase_convention=np.array("amplitude-weighted mean of q over the reference voxels has phase 0; "
                                                  "time courses ~ exp(+i 2 pi f t); lag = -arg(q) / (2 pi f_c)"))
    rows = mode_rows(out)
    io.write_tsv(out_dir / "modes.tsv", {c: [r.get(c) for r in rows] for c in rows[0]})
    meta = {"participant": fs.get("run_id"), "data_type": data_type or ("simulated data" if "sim" in str(fs.get("source", "")) else
                                                                        f"real data ({fs.get('source')} echoes)"),
            "tr_s": tr, "te_ms": te_ms.tolist(), "band_hz": W["band_hz"],
            "nuisance": "motion regression" if Q is not None else "no motion regression"}
    figs = []
    if figures:
        from . import wave_figures as wf
        (out_dir / "figures").mkdir(exist_ok=True)
        inf = inferior_third(mask3d, mimg.affine)[mask3d]
        n_plot = int(min(K, W.get("plot_modes", 3)))
        extra = {"eta_voxel": eta_vox, "eta_fwhm_mm": fwhm, "ref": refv,
                 "rep_voxels": [wf.representative_voxels(out, k, refv, eta_vox[:, k], inf) for k in range(n_plot)]}
        for k in range(n_plot):
            p1, p2 = out_dir / "figures" / f"wave_mode_{k + 1}.png", out_dir / "figures" / f"wave_diagnostics_{k + 1}.png"
            wf.figure_mode(out, k, mask3d, voxel_mm, axcodes, meta, p1, extra)
            wf.figure_diagnostics(out, k, mask3d, voxel_mm, axcodes, meta, p2, extra)
            figs += [str(p1), str(p2)]
    summ = {"run_id": fs.get("run_id"), "source": fs.get("source"), "channels": out["channels"],
            "nuisance_projected": Q is not None, "slice_timing_corrected": out["slice_timing_corrected"],
            "slice_timing_source": st_src, "voxel_scaling": out["voxel_scaling"], "reference_voxels": ref_name,
            "n_volumes_used": out["n_volumes_used"], "n_interpolated": int(out["block"].get("n_interpolated", 0)),
            "band_hz": W["band_hz"], "null": out.get("null"), "bootstrap_n": (out.get("bootstrap") or {}).get("n", 0),
            "modes": rows, "figures": figs}
    io.write_json(out_dir / "waves.json", summ)
    io.write_text(out_dir / "waves_summary.md", render(summ))
    io.write_config_copy(cfg, out_dir)
    io.write_json(out_dir / "done.json", {"stage": "waves", **io.provenance(
        cfg, {"frontend_summary": str(fd / "summary.json"), "c0": str(fd / "c0.npy")}, hash_inputs=False,
        seeds={"waves": {"entropy": seed.entropy, "spawn_key": list(seed.spawn_key)}})})
    return summ


SCALARS = ("f_c_hz", "var_frac_stacked", "var_frac_c0", "speed_mm_s", "fit_quality", "local_speed_median", "R", "eta",
           "eta_band_R2", "eta_band_S0", "lag_vs_slice_time_slope", "n_voxels_present", "te_lag_median_s",
           "te_lag_fraction_beyond_null", "te_lag_median_beyond_null_s", "above_null")


def mode_rows(out: dict) -> list[dict]:
    """One row per mode: the section 4.7 scalars, the direction, and bootstrap intervals where computed."""
    ci = (out.get("bootstrap") or {}).get("ci", {})
    rows = []
    for k, msr in enumerate(out["measures"]):
        r = {"mode": k + 1}
        for key in SCALARS:
            v = msr.get(key)
            if isinstance(v, (list, tuple, np.ndarray)):
                r[key] = "[" + ", ".join(f"{float(x):.4g}" for x in v) + "]"
            elif isinstance(v, (bool, np.bool_)):
                r[key] = bool(v)
            elif v is None:
                r[key] = ""
            else:
                r[key] = round(float(v), 5)
        r["direction"] = "[" + ", ".join(f"{float(x):.3f}" for x in msr["direction"]) + "]"
        r["eigenvalue"] = round(float(out["eig"][k]), 4)
        if out.get("null"):
            r["eigenvalue_over_null"] = round(float(out["eig"][k] / out["null"]["threshold"]), 4)
        for key, v in ci.items():
            r[f"{key}_ci"] = (f"[{v[k][0]:.4g}, {v[k][1]:.4g}]" if k < len(v) else "")
        rows.append(r)
    return rows


def render(s: dict) -> str:
    L = [f"# Traveling-wave modes: {s['run_id']} ({s['source']} echoes)", "",
         f"Channels {s['channels']}, {'with' if s['nuisance_projected'] else 'without'} motion regression; "
         f"{s['n_volumes_used']} contiguous volumes ({s['n_interpolated']} interpolated); band {s['band_hz']} Hz; "
         f"slice-timing correction {'applied' if s['slice_timing_corrected'] else 'not applied'}"
         f"{' (' + s['slice_timing_source'] + ')' if s.get('slice_timing_source') else ''}; normalization "
         f"{s['voxel_scaling']}; reference voxels: {s['reference_voxels']}.", ""]
    if s.get("null"):
        L.append(f"Energy null: largest eigenvalue of {s['null']['n_surrogates']} phase-randomized surrogates, "
                 f"{s['null']['quantile']:.2f} quantile = {s['null']['threshold']:.4g} (family-wise).")
    L.append(f"Bootstrap: {s['bootstrap_n']} circular block resamples (intervals for the plotted modes).")
    L += ["", "| Mode | f_c Hz | var % | Eig / null | Speed mm/s [95 % CI] | Fit | R | eta [R2* band] | TE4-TE1 lag s "
              "(beyond null %) | Lag-slice slope |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["modes"]:
        L.append(f"| {r['mode']} | {r['f_c_hz']:.3f} | {100 * r['var_frac_stacked']:.1f} | {r.get('eigenvalue_over_null', '')} | "
                 f"{r['speed_mm_s']:.3g} {r.get('speed_mm_s_ci', '')} | {r['fit_quality']:.2f} | {r['R']:.2f} | "
                 f"{r['eta']:.3f} {r['eta_band_R2']} | {r['te_lag_median_s']:.2f} "
                 f"({100 * r['te_lag_fraction_beyond_null']:.0f}) | {r['lag_vs_slice_time_slope']:.2f} |")
    L += ["", "Reading the table: speed and direction come from the plane-wave fit (a standing mode has a very high "
              "'speed' and low fit quality); R is the coherence of the local propagation directions; eta is the "
              "noise-debiased S0 energy fraction (interpret only for modes above the null); a lag-slice slope near "
              "-1 means uncorrected slice timing, near 0 corrected. Modes are measurements, not labels "
              "(no neural / vascular classification is made)."]
    return "\n".join(L) + "\n"
