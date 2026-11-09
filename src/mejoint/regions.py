"""Route B (docs/ROUTE_B_PLAN.md; approved 2026-11-09): BOLD-like variance and connectivity at region level.

The signal of each region is averaged echo by echo before anything is separated, so no decomposition (and no
detection limit) is involved:

1. Region signals. For voxel v, echo e, frame t, y_ve(t) = ln(S_ve(t) / mu_ve) (log-ratio, exactly linear in the
   S0-like change a and the R2*-like change b: y = a - TE b). The region average weights each voxel and echo by its
   inverse thermal noise variance w_ve = (mu_ve / (sigma_v r_e))^2 (0 where the echo is unusable):
       ybar_e(t) = sum_v w_ve y_ve(t) / sum_v w_ve,   noise variance 1 / sum_v w_ve (independent voxels).
   Averaging keeps the model exactly linear: the region's a and b are weighted averages of the voxels'.
2. Split. The front end's per-voxel construction (QR of the noise-scaled design [-TE, 1, TE^2, alternating]) is
   applied to the region signal with scale m_e = sqrt(sum_v w_ve): channels c0 (BOLD + S0), c1 (S0 only) and two
   leftover channels r0, r1; then a(t) = c1 / R11 and b(t) = (c0 - R01 a) / R00 (s^-1).
3. Region noise from the region's own leftover channels. Motion-correction interpolation correlates neighbouring
   voxels' noise, so the region's real noise is larger than the independent-voxel value; the leftover channels
   hold thermal noise (plus slow model violations), so the noise variance is measured from their frame-to-frame
   differences (slow violations cancel), as the front end does for sigma.
4. Shares in the analysis band: debiased energies E_S0, E_R2 (as eta, CLAUDE.md 7.1, with time in place of voxels)
   and the leftover energy above noise; BOLD-like share = E_R2 / (E_S0 + E_R2).
5. Coupling and integration: correlations of the regions' b(t) with the noise removed from the variances;
   participation ratio of the eigenvalues (effective dimensionality; primary outcome, D4); count of eigenvalues
   above the circular-shift null; segregation; global share. Null: each region's time course circularly shifted by
   its own random lag (user, 2026-11-08).
"""
from __future__ import annotations

import numpy as np


# ============================================================================ 1-2: region signals and split
def region_signals(S: np.ndarray, labels: np.ndarray, mu: np.ndarray, sigma: np.ndarray, ratios, use: np.ndarray,
                   n_regions: int | None = None) -> dict:
    """S (V, E, T) magnitudes; labels (V,) region index 1..P (0 = none); mu (V, E), sigma (V,), ratios (E,),
    use (V, E). Returns ybar (P, E, T), wsum (P, E), n_voxels (P,)."""
    S = np.asarray(S, np.float64)
    V, E, T = S.shape
    labels = np.asarray(labels, int)
    P = int(n_regions or labels.max())
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(S / mu[:, :, None])
        w = np.where(use, (mu / (sigma[:, None] * np.asarray(ratios, float)[None, :])) ** 2, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    y = np.where(np.isfinite(y), y, 0.0)
    ybar = np.zeros((P, E, T))
    wsum = np.zeros((P, E))
    nvox = np.zeros(P, int)
    for p in range(1, P + 1):
        idx = np.flatnonzero(labels == p)
        nvox[p - 1] = idx.size
        if not idx.size:
            continue
        ws = w[idx].sum(0)
        wsum[p - 1] = ws
        with np.errstate(divide="ignore", invalid="ignore"):
            ybar[p - 1] = np.einsum("ve,vet->et", w[idx], y[idx]) / ws[:, None]
    ybar = np.where(np.isfinite(ybar), ybar, 0.0)
    return {"ybar": ybar, "wsum": wsum, "n_voxels": nvox}


def split(ybar: np.ndarray, wsum: np.ndarray, te_ms, Q=None) -> dict:
    """Region channels and physical time courses. Q (T, p): nuisance basis projected out of the region signals
    first (it commutes with averaging and with the split). Returns c (P, 2, T), r (P, 2, T), R (P, 2, 2),
    a (P, T) S0-like (fraction), b (P, T) R2*-like (s^-1)."""
    from .frontend import project
    y = np.asarray(ybar, np.float64)
    y = y - y.mean(axis=2, keepdims=True)
    if Q is not None:
        y = y - np.einsum("pet,tk,sk->pes", y, Q, Q)
    use = wsum > 0
    pr = project(y, np.sqrt(wsum), use, te_ms)
    c, r, R = pr["c"].astype(np.float64), pr["r"].astype(np.float64), pr["R"]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = c[:, 1] / R[:, 1, 1][:, None]
        b = (c[:, 0] - R[:, 0, 1][:, None] * a) / R[:, 0, 0][:, None] * 1000.0
    return {"c": c, "r": r, "R": R, "a": a, "b": b}


def leftover_noise(r: np.ndarray, removed_df: int = 0) -> np.ndarray:
    """Noise variance per region (unit = thermal noise of an independent-voxel region channel) from the two
    leftover channels, by frame-to-frame differences (white noise: var(diff) = 2 var; slow model violations
    cancel). removed_df: regressors projected out (scales the estimate by T / (T - removed_df))."""
    r = np.asarray(r, np.float64)
    d = np.diff(r, axis=2)
    v = np.nanmean(d ** 2, axis=(1, 2)) / 2.0
    T = r.shape[2]
    return v * T / max(T - int(removed_df), 1)


# ============================================================================ 3-4: band and shares
def band_pass(x: np.ndarray, tr: float, band) -> np.ndarray:
    """Ideal (FFT) band-pass along the last axis; mean removed."""
    T = x.shape[-1]
    F = np.fft.rfft(x - x.mean(-1, keepdims=True), axis=-1)
    f = np.fft.rfftfreq(T, tr)
    F[..., (f < band[0]) | (f > band[1])] = 0.0
    return np.fft.irfft(F, n=T, axis=-1)


def band_fraction(T: int, tr: float, band) -> float:
    """Share of white-noise variance that falls in the band (FFT band-pass)."""
    f = np.fft.rfftfreq(T, tr)
    keep = (f >= band[0]) & (f <= band[1])
    wgt = np.where((f == 0) | ((T % 2 == 0) & (f == f[-1])), 1.0, 2.0)
    return float((wgt * keep).sum() / T)


def shares(sp: dict, nv: np.ndarray, tr: float, band) -> dict:
    """Per region: debiased energies in the band (summed over frames, unit-noise channels scaled by nv) and the
    shares. BOLD-like share = E_R2 / (E_S0 + E_R2); leftover share = leftover energy above noise / all three."""
    c, r, R = sp["c"], sp["r"], sp["R"]
    T = c.shape[2]
    bf = band_fraction(T, tr, band)
    cb = band_pass(c, tr, band)
    rb = band_pass(np.nan_to_num(r), tr, band)
    R00, R01, R11 = R[:, 0, 0], R[:, 0, 1], R[:, 1, 1]
    ws = (R01 ** 2 + R11 ** 2) / R11 ** 2
    wr = 1.0 + R01 ** 2 / R11 ** 2
    noise = nv * bf * T                                                # expected noise energy per channel in the band
    e_s0 = ws * (cb[:, 1] ** 2).sum(1) - ws * noise
    e_r2 = ((cb[:, 0] - (R01 / R11)[:, None] * cb[:, 1]) ** 2).sum(1) - wr * noise
    n_left = np.isfinite(r).all(2).sum(1)                              # leftover channels present (0-2)
    e_left = (rb ** 2).sum((1, 2)) - n_left * noise
    with np.errstate(divide="ignore", invalid="ignore"):
        bold = e_r2 / (e_s0 + e_r2)
        left = e_left / (e_s0 + e_r2 + e_left)
    return {"E_S0": e_s0, "E_R2": e_r2, "E_left": e_left, "bold_share": bold, "s0_share": 1.0 - bold,
            "leftover_share": left, "noise_energy": noise}


def b_noise_variance(R: np.ndarray, nv: np.ndarray) -> np.ndarray:
    """Noise variance of b(t) (s^-2) per region for channel noise variance nv."""
    R00, R01, R11 = R[:, 0, 0], R[:, 0, 1], R[:, 1, 1]
    return nv * (1.0 + R01 ** 2 / R11 ** 2) / R00 ** 2 * 1e6


# ============================================================================ 5: coupling and integration
def coupling(x: np.ndarray, noise_var: np.ndarray, tr: float, band) -> dict:
    """x (P, T) region time courses (e.g. b), noise_var (P,) their white-noise variance. Band-passed covariance
    with the expected noise removed from the diagonal, and the matching correlation (noise-corrected variances).
    Regions whose noise-corrected variance is not positive get NaN rows."""
    T = x.shape[1]
    xb = band_pass(np.asarray(x, np.float64), tr, band)
    C = xb @ xb.T / T
    sig = np.diag(C) - np.asarray(noise_var, float) * band_fraction(T, tr, band)
    Cn = C.copy()
    np.fill_diagonal(Cn, sig)
    ok = sig > 0
    d = np.sqrt(np.where(ok, sig, np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        Rr = Cn / np.outer(d, d)
    np.fill_diagonal(Rr, np.where(ok, 1.0, np.nan))
    return {"cov": Cn, "corr": Rr, "signal_var": sig, "ok": ok, "x_band": xb}


def participation_ratio(C: np.ndarray) -> float:
    """Effective dimensionality (sum lambda)^2 / sum lambda^2 of a symmetric matrix (negative eigenvalues, which
    noise removal can produce, set to 0); NaN rows / columns dropped."""
    ok = np.isfinite(C).all(0) & np.isfinite(C).all(1)
    lam = np.clip(np.linalg.eigvalsh(C[np.ix_(ok, ok)]), 0.0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.sum() > 0 else float("nan")


def eigenvalues(C: np.ndarray) -> np.ndarray:
    ok = np.isfinite(C).all(0) & np.isfinite(C).all(1)
    return np.sort(np.clip(np.linalg.eigvalsh(C[np.ix_(ok, ok)]), 0.0, None))[::-1]


def segregation(corr: np.ndarray, networks: np.ndarray) -> float:
    """(mean within-network correlation - mean between-network correlation) / mean within-network correlation,
    over region pairs with finite values (after Chan et al. 2014; defined here on the noise-corrected BOLD-like
    correlations, negative values kept)."""
    networks = np.asarray(networks)
    iu = np.triu_indices(corr.shape[0], 1)
    v = corr[iu]
    same = networks[iu[0]] == networks[iu[1]]
    ok = np.isfinite(v)
    w, bt = v[ok & same].mean(), v[ok & ~same].mean()
    return float((w - bt) / w) if w != 0 else float("nan")


def global_share(xb: np.ndarray, signal_var: np.ndarray) -> float:
    """Share of the regions' (noise-corrected) band variance carried by their common mean time course."""
    ok = signal_var > 0
    g = xb[ok].mean(0)
    gn = g / np.linalg.norm(g)
    expl = ((xb[ok] @ gn) ** 2).sum() / xb.shape[1]
    return float(expl / signal_var[ok].sum())


def shift_null(x: np.ndarray, noise_var: np.ndarray, tr: float, band, n: int, min_shift: int, rng,
               networks=None) -> dict:
    """Circular-shift null (user, 2026-11-08): every region's band-passed time course shifted by its own random
    lag (min_shift..T - min_shift frames, wrapping around); the regions keep their own dynamics, their alignment
    is destroyed. Returns null distributions of the participation ratio, the eigenvalues, every correlation,
    and the segregation."""
    T = x.shape[1]
    xb = band_pass(np.asarray(x, np.float64), tr, band)
    P = xb.shape[0]
    pr, eig, seg = np.zeros(n), [], np.zeros(n)
    edges = np.zeros((n, P * (P - 1) // 2))
    iu = np.triu_indices(P, 1)
    nb = np.asarray(noise_var, float) * band_fraction(T, tr, band)
    for i in range(n):
        lags = rng.integers(min_shift, T - min_shift + 1, P)
        xs = np.stack([np.roll(xb[p], int(lags[p])) for p in range(P)])
        C = xs @ xs.T / T
        sig = np.diag(C) - nb
        np.fill_diagonal(C, sig)
        d = np.sqrt(np.where(sig > 0, sig, np.nan))
        with np.errstate(divide="ignore", invalid="ignore"):
            Rr = C / np.outer(d, d)
        np.fill_diagonal(Rr, np.where(sig > 0, 1.0, np.nan))
        pr[i] = participation_ratio(Rr)
        eig.append(eigenvalues(Rr))
        edges[i] = Rr[iu]
        if networks is not None:
            seg[i] = segregation(Rr, networks)
    m = min(len(e) for e in eig)
    return {"pr": pr, "eig": np.array([e[:m] for e in eig]), "edges": edges, "segregation": seg if networks is not None else None}


def count_above_null(eig_obs: np.ndarray, eig_null: np.ndarray, q: float = 0.95) -> int:
    """Leading eigenvalues above the q-quantile of the same-rank null eigenvalue, counted until the first that is not."""
    thr = np.quantile(eig_null, q, axis=0)
    k = 0
    for i in range(min(len(eig_obs), thr.size)):
        if eig_obs[i] > thr[i]:
            k += 1
        else:
            break
    return k
