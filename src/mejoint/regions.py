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


def leftover_noise_spectral(r: np.ndarray, tr: float, f_min: float = 0.1, removed_df: int = 0) -> np.ndarray:
    """Robust alternative (diagnostic, 2026-11-10): the median periodogram of the leftover channels above f_min
    (white noise: each periodogram value is the noise variance times an exponential variable, median ln 2).
    Narrow-band structure (cardiac aliases, respiration) moves a mean-based estimate such as the frame-difference
    one, but barely moves a median; broadband structure moves both; slow violations below f_min are ignored."""
    r = np.nan_to_num(np.asarray(r, np.float64))
    T = r.shape[2]
    F = np.fft.rfft(r - r.mean(2, keepdims=True), axis=2)
    f = np.fft.rfftfreq(T, tr)
    keep = (f >= f_min) & (f > 0) & ~((T % 2 == 0) & (np.arange(f.size) == f.size - 1))
    P = np.abs(F[:, :, keep]) ** 2 / T
    present = (np.abs(r).sum(2) > 0)                                   # leftover channels that exist
    out = np.full(r.shape[0], np.nan)
    for p in range(r.shape[0]):
        if present[p].any():
            out[p] = np.median(P[p][present[p]]) / np.log(2.0)
    return out * T / max(T - int(removed_df), 1)


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
    return {"cov": Cn, "corr": Rr, "signal_var": sig, "ok": ok, "x_band": xb,
            "noise_band_var": np.asarray(noise_var, float) * band_fraction(T, tr, band)}


def participation_ratio(C: np.ndarray) -> float:
    """Effective dimensionality (sum lambda)^2 / sum lambda^2 of a symmetric matrix (negative eigenvalues, which
    noise removal can produce, set to 0). Regions without signal above their noise (NaN diagonal) are dropped
    row and column; the first version dropped every row that held any NaN, which emptied the whole matrix as soon
    as one region had no signal (fixed 2026-11-10)."""
    ok = _signal_rows(C)
    if ok.sum() < 2:
        return float("nan")
    lam = np.clip(np.linalg.eigvalsh(C[np.ix_(ok, ok)]), 0.0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.sum() > 0 else float("nan")


def participation_ratio_residual(C: np.ndarray) -> float:
    """Participation ratio with the first (brain-wide) pattern removed: (sum lambda - lambda_1)^2 /
    (sum lambda^2 - lambda_1^2), from the eigenvalues of the full matrix (user decision 2026-11-10, secondary
    measure). R3 showed that the ordinary ratio rises when any signal is added beside one dominant pattern, even a
    shared one; this version reads the dimensionality of the structure beyond that pattern and is reported next to
    the global share, which reads the pattern's dominance."""
    ok = _signal_rows(C)
    if ok.sum() < 3:
        return float("nan")
    lam = np.sort(np.clip(np.linalg.eigvalsh(C[np.ix_(ok, ok)]), 0.0, None))[::-1]
    rest = lam[1:]
    return float(rest.sum() ** 2 / (rest ** 2).sum()) if rest.sum() > 0 else float("nan")


def _signal_rows(C: np.ndarray) -> np.ndarray:
    """Regions whose noise-corrected variance is positive (finite diagonal); their mutual entries are finite."""
    ok = np.isfinite(np.diag(C))
    sub = C[np.ix_(ok, ok)]
    if not np.isfinite(sub).all():                                    # defensive: drop rows that still hold NaN
        ok[np.flatnonzero(ok)[~np.isfinite(sub).all(1)]] = False
    return ok


def eigenvalues(C: np.ndarray) -> np.ndarray:
    ok = _signal_rows(C)
    if ok.sum() < 1:
        return np.zeros(0)
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
    if not (ok & same).any() or not (ok & ~same).any():               # no within- or no between-network pairs
        return float("nan")
    w, bt = v[ok & same].mean(), v[ok & ~same].mean()
    return float((w - bt) / w) if w != 0 else float("nan")


def global_share(xb: np.ndarray, signal_var: np.ndarray, noise_band=None) -> float:
    """How much of the regions' signal is common to all of them: the variance of their mean time course (its own
    noise removed: mean noise / n) over their mean noise-corrected variance. ~1 when one signal drives every region,
    ~0 when the regions are independent."""
    ok = signal_var > 0
    if ok.sum() < 2:                                                   # no region with signal above its noise
        return float("nan")
    msv = float(signal_var[ok].mean())
    nb = 0.0 if noise_band is None else float(np.asarray(noise_band)[ok].mean())
    g = xb[ok].mean(0)
    gv = float(g.var()) - nb / ok.sum()
    return float(gv / msv) if msv > 0 else float("nan")


def shift_null(x: np.ndarray, noise_var: np.ndarray, tr: float, band, n: int, min_shift: int, rng,
               networks=None) -> dict:
    """Circular-shift null (user, 2026-11-08): every region's band-passed time course shifted by its own random
    lag (min_shift..T - min_shift frames, wrapping around); the regions keep their own dynamics, their alignment
    is destroyed. Returns null distributions of the participation ratio (also with the first pattern removed),
    the eigenvalues, every correlation, and the segregation."""
    T = x.shape[1]
    xb = band_pass(np.asarray(x, np.float64), tr, band)
    P = xb.shape[0]
    pr, pr1, eig, seg = np.zeros(n), np.zeros(n), [], np.zeros(n)
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
        pr1[i] = participation_ratio_residual(Rr)
        eig.append(eigenvalues(Rr))
        edges[i] = Rr[iu]
        if networks is not None:
            seg[i] = segregation(Rr, networks)
    m = min(len(e) for e in eig)
    return {"pr": pr, "pr_resid": pr1, "eig": np.array([e[:m] for e in eig]), "edges": edges, "segregation": seg if networks is not None else None}


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


def count_above_null_deflated(eig_obs: np.ndarray, eig_null: np.ndarray, q: float = 0.95, max_k: int = 30) -> int:
    """Deflated count (candidate, 2026-11-10; adopted only if it passes the R3 injections): the i-th eigenvalue's
    share of what is left after the larger ones are removed, lambda_i / sum_{j>=i} lambda_j, against the same
    quantity of the null's i-th eigenvalue; counted until the first that is not above the q-quantile. The plain
    count compares raw eigenvalues, so a dominant first eigenvalue (half the trace here) pushes every later one
    below the null."""
    m = min(len(eig_obs), eig_null.shape[1] - 1, max_k)
    k = 0
    for i in range(m):
        tail_o = eig_obs[i:].sum()
        tail_n = eig_null[:, i:].sum(1)
        if tail_o <= 0:
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            thr = np.nanquantile(eig_null[:, i] / tail_n, q)
        if eig_obs[i] / tail_o > thr:
            k += 1
        else:
            break
    return k


# ============================================================================ intervals and the real-data runner
def share_ci(sp: dict, nv: np.ndarray, tr: float, band, n_boot: int = 200, block: int = 27, rng=None, q=(0.025, 0.975)):
    """Block-bootstrap interval of each region's BOLD-like share: frames of the band-passed channels resampled in
    circular blocks (keeps their autocorrelation), energies recomputed, the fixed noise energy removed."""
    rng = np.random.default_rng(0) if rng is None else rng
    c, R = sp["c"], sp["R"]
    T = c.shape[2]
    bf = band_fraction(T, tr, band)
    cb = band_pass(c, tr, band)
    R01, R11 = R[:, 0, 1], R[:, 1, 1]
    ws = (R01 ** 2 + R11 ** 2) / R11 ** 2
    wr = 1.0 + R01 ** 2 / R11 ** 2
    es_t = ws[:, None] * cb[:, 1] ** 2                                 # per-frame energy contributions
    er_t = (cb[:, 0] - (R01 / R11)[:, None] * cb[:, 1]) ** 2
    noise = nv * bf * T
    nb = int(np.ceil(T / block))
    out = np.full((n_boot, c.shape[0]), np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, T, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        e_s0 = es_t[:, idx].sum(1) - ws * noise
        e_r2 = er_t[:, idx].sum(1) - wr * noise
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = e_r2 / (e_s0 + e_r2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(out, q[0], axis=0), np.nanquantile(out, q[1], axis=0)


def analyze(S, labels, mu, sigma, ratios, use, te_ms, table: list, cfg: dict, Q=None, rng=None, common_ids=None,
            frames=None) -> dict:
    """Route B on one run, arrays version. S (V, E, T) masked magnitudes; labels (V,) region ids; table: region
    rows from anatomy (id, name, kind, network, n_voxels, usable). Returns per-region rows, the coupling
    matrices, integration measures (full usable set and, if given, the common set) and their circular-shift nulls.
    Shares use every frame; coupling and integration use `frames` (the common number of volumes, R5-1, user
    2026-11-11), or the first regions.n_volumes_common frames, or all."""
    G = cfg.get("regions", {})
    tr = float(cfg["acquisition"]["tr_s"])
    band = tuple(G.get("band_hz", [0.01, 0.10]))
    rng = np.random.default_rng(0) if rng is None else rng
    nT = G.get("n_volumes_common")
    P = int(max(r["id"] for r in table))
    rs = region_signals(S, labels, mu, sigma, ratios, use, n_regions=P)
    sp = split(rs["ybar"], rs["wsum"], te_ms, Q)
    removed = 0 if Q is None else Q.shape[1]
    nv = leftover_noise(sp["r"], removed)
    nv = np.where(np.isfinite(nv) & (nv > 0), nv, np.nan)
    nv_spec = leftover_noise_spectral(sp["r"], tr, float(band[1]), removed)
    sh = shares(sp, np.nan_to_num(nv, nan=1.0), tr, band)
    lo, hi = share_ci(sp, np.nan_to_num(nv, nan=1.0), tr, band, int(G.get("n_boot", 200)),
                      max(2, int(round(float(G.get("boot_block_s", 20.0)) / tr))), rng)
    T_all = S.shape[2]
    if frames is None and nT:
        frames = np.arange(min(int(nT), T_all))
    sp_full = sp
    if frames is not None:                                             # coupling and integration on these frames
        frames = np.asarray(frames, int)
        sp = {**sp, "c": sp["c"][..., frames], "r": sp["r"][..., frames], "a": sp["a"][:, frames], "b": sp["b"][:, frames]}
    T = sp["b"].shape[1]
    rows = []
    for r in table:
        p = r["id"] - 1
        ok = bool(r.get("usable")) and bool(np.isfinite(nv[p])) and bool(np.isfinite(sp["c"][p]).all())
        below = [n for n, k in (("S0", "E_S0"), ("R2", "E_R2")) if ok and np.isfinite(sh[k][p]) and sh[k][p] < 0]
        bs = float(np.clip(sh["bold_share"][p], 0.0, 1.0)) if ok and np.isfinite(sh["bold_share"][p]) else np.nan
        rows.append({**{k: r[k] for k in ("id", "name", "kind", "hemisphere", "network", "n_voxels")}, "used": ok,
                     "noise_var": _r(nv[p]), "noise_var_spectral_ratio": _r(nv_spec[p] / nv[p]) if ok else None,
                     "bold_share": _r(bs) if ok else None,
                     "bold_share_raw": _r(sh["bold_share"][p]) if ok else None,
                     "energy_below_noise": ";".join(below),
                     "bold_share_ci": [_r(np.clip(lo[p], 0, 1)), _r(np.clip(hi[p], 0, 1))] if ok else None,
                     "s0_share": _r(1.0 - bs) if ok else None, "leftover_share": _r(sh["leftover_share"][p]) if ok else None,
                     "E_S0": _r(sh["E_S0"][p]), "E_R2": _r(sh["E_R2"][p]), "E_left": _r(sh["E_left"][p])})
    # reliability of each region's BOLD-like time course: the share of its slow-band variance that is signal
    # (1 - noise / total). Noise correction divides by its square root, so it amplifies errors where it is low.
    xb_all = band_pass(sp["b"], tr, band)
    bf = band_fraction(T, tr, band)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_all = 1.0 - b_noise_variance(sp["R"], nv) * bf / xb_all.var(1)
    for x in rows:
        x["b_reliability"] = _r(rel_all[x["id"] - 1]) if x["used"] else None
    used = np.array([x["used"] for x in rows])
    ids = np.array([x["id"] for x in rows])
    kinds = np.array([x["kind"] for x in rows])
    nets = np.array([x["network"] for x in rows])
    cx = used & (kinds == "cortex")
    sel = ids[cx] - 1
    bnv = b_noise_variance(sp["R"][sel], nv[sel])
    cb = coupling(sp["b"][sel], bnv, tr, band)
    ca = coupling(sp["a"][sel], nv[sel] / sp["R"][sel, 1, 1] ** 2, tr, band)
    co = coupling(sp["c"][sel, 0], nv[sel], tr, band)
    min_shift = int(round(float(G.get("min_shift_s", 30.0)) / tr))
    n_null = int(G.get("n_null", 1000))

    def integration(mask_sub, label):
        import warnings
        with warnings.catch_warnings():                                # regions without signal above noise give NaN rows
            warnings.simplefilter("ignore", RuntimeWarning)
            return _integration(mask_sub, label)

    def _integration(mask_sub, label):
        b = sp["b"][sel][mask_sub]
        nb_ = bnv[mask_sub]
        c_ = coupling(b, nb_, tr, band)
        null = shift_null(b, nb_, tr, band, n_null, min_shift, rng, networks=nets[cx][mask_sub])
        eig = eigenvalues(c_["corr"])
        pr = participation_ratio(c_["corr"])
        pr1 = participation_ratio_residual(c_["corr"])
        return {"set": label, "n_regions": int(mask_sub.sum()), "n_regions_with_signal": int(_signal_rows(c_["corr"]).sum()),
                "participation_ratio": _r(pr),
                "pr_null_median": _r(np.nanmedian(null["pr"])), "pr_null_q05_q95": [_r(np.nanquantile(null["pr"], 0.05)), _r(np.nanquantile(null["pr"], 0.95))],
                "pr_percentile_in_null": _r(float(np.mean(null["pr"] <= pr))),
                **_edges_above_1(c_["corr"], rel_all[sel][mask_sub]),
                "pr_without_first": _r(pr1),
                "pr_without_first_null_median": _r(np.nanmedian(null["pr_resid"])),
                "pr_without_first_null_q05_q95": [_r(np.nanquantile(null["pr_resid"], 0.05)), _r(np.nanquantile(null["pr_resid"], 0.95))],
                "pr_without_first_percentile_in_null": _r(float(np.mean(null["pr_resid"] <= pr1))),
                "n_eigen_above_null": int(count_above_null(eig, null["eig"])),
                "n_eigen_above_null_deflated": int(count_above_null_deflated(eig, null["eig"])),
                "segregation": _r(segregation(c_["corr"], nets[cx][mask_sub])),
                **_segregation_perm(c_["corr"], nets[cx][mask_sub], n_null, rng),
                "global_share": _r(global_share(c_["x_band"], c_["signal_var"], c_["noise_band_var"])),
                "eigenvalues": [round(float(x), 4) for x in eig[:30]],
                "null_eig_q95": [round(float(x), 4) for x in np.nanquantile(null["eig"], 0.95, axis=0)[:30]] if null["eig"].size else [],
                "edges_beyond_null": _r(_edge_fraction(c_["corr"], null["edges"])),
                "_edge_thr": np.nanquantile(np.abs(null["edges"]), 0.95, axis=0) if null["edges"].size else None}
    full = integration(np.ones(sel.size, bool), "all usable cortical regions")
    edge_thr_full = full.pop("_edge_thr")
    out_int = [full]
    if common_ids is not None:
        m = np.isin(ids[cx], np.asarray(list(common_ids), int))
        if m.sum() >= 3:
            out_int.append(integration(m, "common set"))
    rmin = G.get("reliability_min")
    if rmin is not None:
        m = np.nan_to_num(rel_all[sel], nan=-1.0) >= float(rmin)
        if 3 <= m.sum() < m.size:                                      # sensitivity set (user, 2026-11-10)
            out_int.append(integration(m, f"cortex, reliability >= {rmin}"))
    for i in out_int:
        i.pop("_edge_thr", None)
    # secondary regions: coupling of their b with each cortical network's mean b
    # noise-corrected like the cortical matrix (the first version used plain correlations, which deep regions'
    # larger noise attenuates): covariance over the square root of both noise-corrected variances; the network
    # mean's noise is the mean of its regions' noise over their number (independent noise across regions)
    sec, sec_raw = [], []
    n_boot = int(G.get("n_boot", 200))
    block = max(2, int(round(float(G.get("boot_block_s", 20.0)) / tr)))
    nblk = int(np.ceil(T / block))
    boot_idx = [((rng.integers(0, T, nblk)[:, None] + np.arange(block)[None, :]).ravel()[:T] % T) for _ in range(n_boot)]

    def _ccorr(x, y, nx, ny):
        x, y = x - x.mean(), y - y.mean()
        vx, vy = float(np.mean(x * x)) - nx, float(np.mean(y * y)) - ny
        return float(np.mean(x * y)) / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan
    for r in [x for x in rows if x["kind"] == "secondary" and x["used"]]:
        p = r["id"] - 1
        row, raw = {"name": r["name"]}, {"name": r["name"]}
        nb_p = float(b_noise_variance(sp["R"][[p]], nv[[p]])[0]) * bf
        for net in sorted(set(nets[cx])):
            k = nets[cx] == net
            m = xb_all[sel[k]].mean(0)
            nb_m = float(bnv[k].mean()) * bf / k.sum()
            cv = float(np.mean((xb_all[p] - xb_all[p].mean()) * (m - m.mean())))
            vp, vm = float(xb_all[p].var()) - nb_p, float(m.var()) - nb_m
            row[net] = _r(cv / np.sqrt(vp * vm)) if vp > 0 and vm > 0 else None
            raw[net] = _r(np.corrcoef(xb_all[p], m)[0, 1])
            bs = np.array([_ccorr(xb_all[p][ix], m[ix], nb_p, nb_m) for ix in boot_idx])
            bs = bs[np.isfinite(bs)]
            row.setdefault("ci", {})[net] = [_r(np.quantile(bs, 0.025)), _r(np.quantile(bs, 0.975))] if bs.size > 10 else None
        row["b_reliability"] = r["b_reliability"]
        rmin = G.get("reliability_min")
        row["not_measurable"] = rmin is not None and (r["b_reliability"] is None or r["b_reliability"] < float(rmin))
        vals = [v for k, v in row.items() if k not in ("name", "ci", "b_reliability", "not_measurable") and v is not None]
        row["noise_overcorrected"] = bool(vals) and max(abs(v) for v in vals) > 1.0
        sec.append(row)
        sec_raw.append(raw)
    return {"regions": rows, "integration": out_int, "secondary_coupling": sec, "secondary_coupling_raw": sec_raw, "cortex_ids": ids[cx].tolist(),
            "cortex_networks": nets[cx].tolist(), "corr_b": cb["corr"], "corr_a": ca["corr"], "corr_oc": co["corr"],
            "n_frames": int(T), "n_frames_total": int(T_all), "nuisance_rank": int(removed), "band_hz": list(band),
            "a": sp_full["a"], "b": sp_full["b"], "c0": sp_full["c"][:, 0], "R": sp["R"], "noise_var": nv,
            "b_int": sp["b"], "frames": frames,
            "r_cortex": sp["r"][sel], "cov_b": cb["cov"], "noise_band_b": cb["noise_band_var"], "edge_thr_b": edge_thr_full}


def _segregation_perm(corr: np.ndarray, networks: np.ndarray, n: int, rng, q: float = 0.95) -> dict:
    """Segregation null by permuting the network labels over the regions (the matrix kept): is the network structure
    stronger than a random grouping of the same regions? Replaces the circular-shift null for segregation, which was
    undefined (its within-network mean is ~0, so the ratio explodes: null 95 % of 3.5-4.6 in R4). Label permutation
    ignores spatial autocorrelation (neighbouring parcels share networks), so it is liberal."""
    networks = np.asarray(networks)
    iu = np.triu_indices(corr.shape[0], 1)
    v = corr[iu]
    ok = np.isfinite(v)

    def parts(lab):
        same = lab[iu[0]] == lab[iu[1]]
        return v[ok & same].mean(), v[ok & ~same].mean()
    w, bt = parts(networks)
    null = np.array([np.subtract(*parts(rng.permutation(networks))) for _ in range(n)])
    return {"within_mean": _r(w), "between_mean": _r(bt), "within_minus_between": _r(w - bt),
            "segregation_null_q95": _r(float(np.quantile(null / np.maximum(np.abs(w), 1e-12), q))),
            "within_minus_between_null_q95": _r(float(np.quantile(null, q))),
            "segregation_percentile_in_null": _r(float(np.mean(null < (w - bt))))}


def _edges_above_1(corr: np.ndarray, rel: np.ndarray) -> dict:
    """Correlations above 1 after noise correction, and the reliability of the regions they involve (low
    reliability: the correction divides by a small, uncertain signal variance)."""
    iu = np.triu_indices(corr.shape[0], 1)
    hit = np.abs(corr[iu]) > 1.0
    reg = np.unique(np.r_[iu[0][hit], iu[1][hit]])
    return {"n_edges_above_1": int(hit.sum()), "n_edges": int(np.isfinite(corr[iu]).sum()),
            "reliability_median": _r(np.nanmedian(rel)),
            "reliability_q05": _r(np.nanquantile(rel, 0.05)) if np.isfinite(rel).any() else None,
            "edges_above_1_reliability_median": _r(np.nanmedian(rel[reg])) if reg.size else None}


def _edge_fraction(corr, null_edges, q=0.95):
    iu = np.triu_indices(corr.shape[0], 1)
    thr = np.nanquantile(np.abs(null_edges), q, axis=0)
    v = np.abs(corr[iu])
    ok = np.isfinite(v) & np.isfinite(thr)
    return float(np.mean(v[ok] > thr[ok])) if ok.any() else float("nan")


def _r(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def select_frames(volumes, n_total: int, n_common) -> dict:
    """Frames for coupling and integration (R5-1, user 2026-11-11): the first n_common frames of the longest
    gap-free stretch (the whole run when it has no gap). If that stretch is shorter than n_common, its frames are
    used, and the run is not eligible for the primary (reported separately): band-pass filtering and circular
    shifts assume an unbroken series. volumes: original frame numbers of the kept frames (None = consecutive)."""
    from .nuisance import segments
    segs = segments(None if volumes is None else np.asarray(volumes, int), n_total)
    longest = max(segs, key=lambda x: x.stop - x.start)
    L = longest.stop - longest.start
    out = {"n_total": int(n_total), "n_segments": len(segs), "longest_stretch": [int(longest.start), int(longest.stop)]}
    if n_common is None:
        fr = np.arange(longest.start, longest.stop)
        out.update(status="all frames of the longest stretch", primary_eligible=True)
    elif L >= int(n_common):
        fr = np.arange(longest.start, longest.start + int(n_common))
        out.update(status="ok", primary_eligible=True)
    else:
        fr = np.arange(longest.start, longest.stop)
        out.update(status=f"longest gap-free stretch {L} < {int(n_common)} frames: not in the primary", primary_eligible=False)
    out["n_used"] = int(fr.size)
    out["frames"] = fr
    return out


def adjacency(labels3d: np.ndarray, ids) -> np.ndarray:
    """Regions that share a voxel face (6-neighbourhood), in the order of ids."""
    ids = list(ids)
    pos = {v: i for i, v in enumerate(ids)}
    A = np.zeros((len(ids), len(ids)), bool)
    for ax in range(3):
        a = np.moveaxis(labels3d, ax, 0)
        u, v = a[:-1].ravel(), a[1:].ravel()
        m = (u != v) & (u > 0) & (v > 0)
        for x, y in set(zip(u[m].tolist(), v[m].tolist())):
            if x in pos and y in pos:
                A[pos[x], pos[y]] = A[pos[y], pos[x]] = True
    return A


def shared_noise_check(r: np.ndarray, cov_b: np.ndarray, noise_band: np.ndarray, adj: np.ndarray) -> dict:
    """Diagnostic (2026-11-11): noise shared between regions. The noise correction removes each region's own noise
    from the diagonal only, which assumes independent noise across regions; motion-correction resampling and the
    EPI point-spread make neighbouring voxels' noise correlated across parcel borders, which would add noise
    covariance to adjacent pairs. The leftover channels hold noise without the BOLD- or S0-like signal: their
    frame-difference correlations between regions estimate the noise correlation (approximation: assumed equal for
    the b time courses). Returns its summary by adjacency and the integration measures with the estimated noise
    covariance also removed from the off-diagonal."""
    d = np.diff(np.nan_to_num(np.asarray(r, np.float64)), axis=2)
    d = d - d.mean(2, keepdims=True)
    sd = d.std(2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = d / sd[:, :, None]
    z = np.nan_to_num(z)
    P = z.shape[0]
    rho = np.zeros((P, P))
    n = 0
    for k in range(z.shape[1]):
        if np.any(sd[:, k] > 0):
            rho += z[:, k] @ z[:, k].T / z.shape[2]
            n += 1
    rho /= max(n, 1)
    iu = np.triu_indices(P, 1)
    a = adj[iu]
    nb = np.asarray(noise_band, float)
    C = np.array(cov_b, float)
    C2 = C - rho * np.sqrt(np.outer(nb, nb))
    np.fill_diagonal(C2, np.diag(C))
    dg = np.sqrt(np.where(np.diag(C2) > 0, np.diag(C2), np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        R2 = C2 / np.outer(dg, dg)
    np.fill_diagonal(R2, np.where(np.isfinite(dg), 1.0, np.nan))
    se = 1.0 / np.sqrt(n * z.shape[2])
    return {"noise_corr_adjacent_median": _r(np.median(rho[iu][a])) if a.any() else None,
            "noise_corr_adjacent_q90": _r(np.quantile(rho[iu][a], 0.9)) if a.any() else None,
            "noise_corr_nonadjacent_median": _r(np.median(rho[iu][~a])) if (~a).any() else None,
            "noise_corr_nonadjacent_q90": _r(np.quantile(rho[iu][~a], 0.9)) if (~a).any() else None,
            "n_adjacent_pairs": int(a.sum()), "n_pairs": int(a.size), "sampling_se": _r(se),
            "participation_ratio_shared_noise_removed": _r(participation_ratio(R2)),
            "pr_without_first_shared_noise_removed": _r(participation_ratio_residual(R2)),
            "n_edges_above_1_shared_noise_removed": int(np.sum(np.abs(R2[iu]) > 1.0))}


def deep_csf_check(S, labels, mu, sigma, ratios, use, te_ms, table: list, cfg: dict, Q, res: dict, csf: np.ndarray,
                   thresholds=(0.5, 0.3)) -> list:
    """Diagnostic for the thalamus and striatum (2026-11-10): how much of each region's averaging weight sits in
    CSF-dominated voxels (CSF probability > 0.5), per echo (bright, long-T2* CSF voxels weigh more than tissue,
    most at the late echoes), and the region's shares, reliability and BOLD-like coupling with the cortical network
    means recomputed without voxels above each CSF threshold. Labels used by the outcomes are not changed."""
    G = cfg.get("regions", {})
    tr = float(cfg["acquisition"]["tr_s"])
    band = tuple(G.get("band_hz", [0.01, 0.10]))
    fr = res.get("frames")
    T = S.shape[2] if fr is None else len(fr)
    bf = band_fraction(T, tr, band)
    sec = [r for r in table if r["kind"] == "secondary"]
    if not sec:
        return []
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(use, (mu / (sigma[:, None] * np.asarray(ratios, float)[None, :])) ** 2, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    # cortical network means (the outcome's own time courses)
    cid = np.asarray(res["cortex_ids"]) - 1
    cnet = np.asarray(res["cortex_networks"])
    xb_c = band_pass(res.get("b_int", res["b"])[cid], tr, band)
    nb_c = b_noise_variance(res["R"][cid], res["noise_var"][cid]) * bf
    means = {n: (xb_c[cnet == n].mean(0), float(nb_c[cnet == n].mean()) / (cnet == n).sum()) for n in sorted(set(cnet))}

    def one(idx):
        if idx.size < int(G.get("min_voxels", 20)):
            return None
        lab = np.ones(idx.size, int)
        rs = region_signals(S[idx], lab, mu[idx], sigma[idx], ratios, use[idx], n_regions=1)
        sp = split(rs["ybar"], rs["wsum"], te_ms, Q)
        nv = leftover_noise(sp["r"], 0 if Q is None else Q.shape[1])
        sh = shares(sp, nv, tr, band)
        xb = band_pass(sp["b"] if fr is None else sp["b"][:, fr], tr, band)[0]
        nb = float(b_noise_variance(sp["R"], nv)[0]) * bf
        out = {"n_voxels": int(idx.size), "bold_share": _r(np.clip(sh["bold_share"][0], 0, 1)),
               "leftover_share": _r(sh["leftover_share"][0]), "reliability": _r(1.0 - nb / xb.var())}
        for n, (m, nm) in means.items():
            x, y = xb - xb.mean(), m - m.mean()
            vx, vy = float(np.mean(x * x)) - nb, float(np.mean(y * y)) - nm
            out[n] = _r(float(np.mean(x * y)) / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else None
        return out
    rows = []
    for r in sec:
        idx = np.flatnonzero(labels == r["id"])
        dom = csf[idx] > 0.5
        tot = w[idx].sum(0)
        row = {"name": r["name"], "n_voxels": int(idx.size), "n_csf_dominated": int(dom.sum()),
               "csf_weight_share_by_echo": [_r(float(w[idx][dom, e].sum() / tot[e])) if tot[e] > 0 else None for e in range(w.shape[1])],
               "all_voxels": one(idx)}
        for t in thresholds:
            row[f"csf_below_{t}"] = one(idx[csf[idx] <= t])
        rows.append(row)
    return rows


def run_dir(frontend_dir, anat_dir, cfg: dict, out_dir, atlas: str | None = None, common_ids_file=None) -> dict:
    """Route B on one run from its front-end folder (maps, nuisance basis, echo paths in done.json) and its anatomy
    folder (labels_<atlas>.nii.gz, regions_<atlas>.json). Writes route_b.json, regions_route_b.tsv, the coupling
    matrices (.npy) and route_b.png."""
    import json
    from pathlib import Path

    import nibabel as nib

    from . import io
    fe, A, out = Path(frontend_dir), Path(anat_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    atlas = atlas or cfg.get("regions", {}).get("atlas", "200_7")
    mimg = nib.load(str(fe / "analysis_mask.nii.gz"))
    mask3d = np.asarray(mimg.dataobj) > 0
    lab3 = np.rint(np.asarray(nib.load(str(A / f"labels_{atlas}.nii.gz")).dataobj)).astype(int)
    if lab3.shape != mask3d.shape:
        raise ValueError(f"labels grid {lab3.shape} differs from the front end {mask3d.shape}")
    table = json.loads((A / f"regions_{atlas}.json").read_text())["regions"]
    done = json.loads((fe / "done.json").read_text())
    files = [done["inputs"][f"echo{k}"]["path"] for k in range(1, 5) if f"echo{k}" in done.get("inputs", {})]
    fs = json.loads((fe / "summary.json").read_text())
    S = io.load_masked_echoes(files, mask3d)
    vol = lambda n: np.asarray(nib.load(str(fe / f"{n}.nii.gz")).dataobj, np.float64)[mask3d]
    mu, sigma, use = vol("mu"), vol("sigma"), vol("usable") > 0.5
    Q = np.load(fe / "nuisance_Q.npy") if (fe / "nuisance_Q.npy").exists() else None
    common = None
    if common_ids_file is not None and Path(common_ids_file).exists():
        common = [int(x) for x in Path(common_ids_file).read_text().split() if x.strip().isdigit()]
    vols = None
    vpath = done.get("inputs", {}).get("volumes_used")
    vpath = vpath["path"] if isinstance(vpath, dict) else vpath
    if vpath and Path(vpath).exists():
        vu = np.loadtxt(vpath, skiprows=1, ndmin=2).astype(int)
        if vu.shape[0] == S.shape[2]:
            vols = vu[:, 0]
    tl = select_frames(vols, S.shape[2], cfg.get("regions", {}).get("n_volumes_common"))
    res = analyze(S, lab3[mask3d], mu, sigma, fs["echo_noise_ratios_used"], use, fs["te_ms"], table, cfg, Q,
                  np.random.default_rng(int(cfg.get("project_seed", 0)) + 11), common, frames=tl["frames"])
    # mean tissue probabilities per region (FSL FAST in EPI space; csf_epi from anatomy_run.lsf since 2026-11-10)
    try:
        cids = res["cortex_ids"]
        shared = shared_noise_check(res["r_cortex"], res["cov_b"], res["noise_band_b"], adjacency(lab3, cids))
    except Exception as exc:                                           # a diagnostic; never block the main result
        print(f"shared noise check failed: {exc}")
        shared = None
    deep = []
    if (A / "csf_epi.nii.gz").exists():
        csf3 = np.asarray(nib.load(str(A / "csf_epi.nii.gz")).dataobj, np.float64)
        try:
            deep = deep_csf_check(S, lab3[mask3d], mu, sigma, fs["echo_noise_ratios_used"], use, fs["te_ms"], table, cfg, Q,
                                  res, np.nan_to_num(csf3[mask3d]))
        except Exception as exc:                                       # a diagnostic; never block the main result
            print(f"deep CSF check failed: {exc}")
    # sensitivity (user, 2026-11-11): the primary drops CSF-rich cortical voxels; the same measures on cortex with
    # all voxels (anatomy's labels_<atlas>_allvox), and the participant's cortical CSF fraction (covariate)
    sens = []
    G_ = cfg.get("regions", {})
    anat_json = json.loads((A / f"regions_{atlas}.json").read_text())
    csf_cov = anat_json.get("cortex_csf_covariate")
    fa, fj = A / f"labels_{atlas}_allvox.nii.gz", A / f"regions_{atlas}_allvox.json"
    if G_.get("allvoxel_sensitivity", True) and fa.exists() and fj.exists():
        lab_a = np.rint(np.asarray(nib.load(str(fa)).dataobj)).astype(int)
        tab_a = json.loads(fj.read_text())["regions"]
        cfg_c = io.merge(cfg, {"regions": {"reliability_min": None}})
        r2 = analyze(S, lab_a[mask3d], mu, sigma, fs["echo_noise_ratios_used"], use, fs["te_ms"], tab_a, cfg_c, Q,
                     np.random.default_rng(int(cfg.get("project_seed", 0)) + 11), common, frames=tl["frames"])
        row = dict(r2["integration"][0])
        row["set"] = "cortex, all voxels (CSF-rich voxels kept; sensitivity)"
        sens.append(row)
    for name, key in (("gm_epi", "gm_mean"), ("csf_epi", "csf_mean")):
        f = A / f"{name}.nii.gz"
        if f.exists():
            t = np.asarray(nib.load(str(f)).dataobj, np.float64)
            for r in res["regions"]:
                m = lab3 == r["id"]
                r[key] = _r(float(np.nanmean(t[m]))) if m.any() else None
    for k in ("corr_b", "corr_a", "corr_oc"):
        np.save(out / f"{k}.npy", res[k].astype(np.float32))
    # each region's time courses (row = region id - 1) for recovery against injected truth and later analyses
    np.savez_compressed(out / "region_timecourses.npz", a=res["a"].astype(np.float32), b=res["b"].astype(np.float32),
                        c0=res["c0"].astype(np.float32), R=res["R"], noise_var=res["noise_var"],
                        nuisance_Q=(np.load(fe / "nuisance_Q.npy") if (fe / "nuisance_Q.npy").exists() else np.zeros((res["n_frames"], 0))))
    rows = res["regions"]
    io.write_tsv(out / "regions_route_b.tsv", {k: [(";".join(map(str, r[k])) if isinstance(r[k], list) else r[k]) for r in rows] for k in rows[0]})
    summ = {"run": fs.get("run_id"), "atlas": atlas, "frontend_dir": str(fe), "anat_dir": str(A), "n_frames": res["n_frames"],
            "nuisance_rank": res["nuisance_rank"], "band_hz": res["band_hz"], "integration": res["integration"] + sens,
            "secondary_coupling": res["secondary_coupling"], "secondary_coupling_raw": res["secondary_coupling_raw"],
            "deep_csf_check": deep, "shared_noise_check": shared,
            "timeline": {k: v for k, v in tl.items() if k != "frames"},
            "cortex_csf_max": anat_json.get("cortex_csf_max"), "cortex_csf_covariate": csf_cov, "cortex_ids": res["cortex_ids"], "regions": rows,
            "network_medians": _network_medians(rows), **io.provenance(cfg, {})}
    io.write_json(out / "route_b.json", summ)
    io.write_text(out / "route_b.md", render(summ))
    try:
        figure(summ, res, out / "route_b.png")
    except Exception as exc:
        print(f"route B figure failed: {exc}")
    try:
        matrix_figure(summ, res, out / "route_b_matrices.png")
    except Exception as exc:
        print(f"route B matrix figure failed: {exc}")
    try:
        surface_figure(rows, atlas, out / "route_b_surface.png", title=f"{summ['run']}: region measures on the cortex (fsaverage5, display only)")
    except Exception as exc:                                           # nilearn missing or a plotting problem
        print(f"route B surface figure failed: {exc}")
    return summ


def matrix_figure(summ, res, path) -> None:
    """Paired connectivity matrices ordered by network: BOLD-like (R2*-like) and ordinary (c0, OC-like), BOLD-like
    minus ordinary, and BOLD-like with only the edges beyond the circular-shift null (95 %). S0-like connectivity is
    not an outcome (user, 2026-11-11: region S0-like time courses are mostly noise); corr_a.npy stays on disk only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nets = np.array(res["cortex_networks"])
    order = np.argsort(nets, kind="stable")
    o = lambda M: np.asarray(M)[np.ix_(order, order)]
    Cb, Ca, Co = o(res["corr_b"]), o(res["corr_a"]), o(res["corr_oc"])
    P = Cb.shape[0]
    beyond = np.full((P, P), np.nan)
    thr = res.get("edge_thr_b")
    if thr is not None and len(thr) == P * (P - 1) // 2:
        T = np.zeros((P, P))
        iu = np.triu_indices(P, 1)
        T[iu] = thr
        T = T + T.T
        T = o(T)
        Cb_ = Cb.copy()
        beyond = np.where(np.abs(Cb_) > T, Cb_, np.nan)
        np.fill_diagonal(beyond, np.nan)
    D = Cb - Co
    np.fill_diagonal(D, np.nan)
    panels = [("BOLD-like (R2*-like)", Cb, -1, 1), ("ordinary (OC-like c0)", Co, -1, 1),
              ("BOLD-like minus ordinary", D, -0.5, 0.5), ("BOLD-like, edges beyond null", beyond, -1, 1)]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2))
    bounds = np.flatnonzero(nets[order][1:] != nets[order][:-1]) + 0.5
    ticks = [(np.flatnonzero(nets[order] == n).mean(), n) for n in dict.fromkeys(nets[order])]
    for ax, (t, M, lo, hi) in zip(axs, panels):
        im = ax.imshow(M, cmap="RdBu_r", vmin=lo, vmax=hi, interpolation="nearest")
        for b in bounds:
            ax.axhline(b, color="k", lw=0.4); ax.axvline(b, color="k", lw=0.4)
        ax.set_xticks([x for x, _ in ticks], [n for _, n in ticks], rotation=90, fontsize=6)
        ax.set_yticks([x for x, _ in ticks], [n for _, n in ticks], fontsize=6)
        ax.set_title(t, fontsize=9, loc="left")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{summ['run']}: cortical connectivity (noise-corrected, 0.01-0.1 Hz), ordered by network", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def surface_figure(rows: list, atlas: str, path, title: str = "") -> None:
    """Cortical maps of the BOLD-like, S0-like and leftover shares and the reliability of each parcel, painted on
    the Schaefer parcels in MNI space and shown on nilearn's bundled fsaverage5 inflated surface (display only:
    each vertex takes the parcel it falls in; parcels not used are grey)."""
    import matplotlib
    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt
    import nibabel as nib
    from nilearn import datasets, plotting, surface
    repo = Path(__file__).resolve().parents[2]
    n_par, n_net = atlas.split("_")
    img = nib.load(str(repo / "atlases" / "schaefer2018" / f"Schaefer2018_{n_par}Parcels_{n_net}Networks_order_FSLMNI152_2mm.nii.gz"))
    fs = datasets.fetch_surf_fsaverage("fsaverage5")

    def labels(h):
        try:
            v = surface.vol_to_surf(img, fs[f"pial_{h}"], interpolation="nearest_most_frequent", radius=3.0)
        except (TypeError, ValueError):                                # older nilearn
            v = surface.vol_to_surf(img, fs[f"pial_{h}"], interpolation="nearest", radius=3.0)
        return np.rint(np.nan_to_num(v)).astype(int)
    lab = {"left": labels("left"), "right": labels("right")}
    cx = [r for r in rows if r["kind"] == "cortex"]
    nmax = max(r["id"] for r in cx)
    measures = [("bold_share", "BOLD-like share", "viridis", 0, 1), ("s0_share", "S0-like share", "magma", 0, 0.5),
                ("leftover_share", "leftover share", "magma", 0, 0.1), ("b_reliability", "reliability", "cividis", 0, 1)]
    views = [("left", "lateral"), ("left", "medial"), ("right", "lateral"), ("right", "medial")]
    fig = plt.figure(figsize=(3.2 * len(views), 2.6 * len(measures)))
    for i, (key, name, cmap, lo, hi) in enumerate(measures):
        for j, (h, view) in enumerate(views):
            vals = np.full(nmax + 1, np.nan)
            want = "LH" if h == "left" else "RH"
            for r in cx:
                if r["used"] and r.get(key) is not None and r.get("hemisphere") == want:
                    vals[r["id"]] = r[key]
            ax = fig.add_subplot(len(measures), len(views), i * len(views) + j + 1, projection="3d")
            plotting.plot_surf_stat_map(fs[f"infl_{h}"], vals[np.clip(lab[h], 0, nmax)], bg_map=fs[f"sulc_{h}"], hemi=h,
                                        view=view, cmap=cmap, vmin=lo, vmax=hi, axes=ax, figure=fig,
                                        colorbar=(j == len(views) - 1), bg_on_data=True)
            ax.set_title((name + ": " if j == 0 else "") + f"{h} {view}", fontsize=9, loc="left")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _network_medians(rows):
    out = {}
    for net in sorted({r["network"] for r in rows if r["used"]}):
        rr = [r for r in rows if r["used"] and r["network"] == net]
        med = lambda k: _r(np.median([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None
        out[net] = {"n": len(rr), "bold_share": med("bold_share"), "s0_share": med("s0_share"), "leftover_share": med("leftover_share")}
    return out


def render(s: dict) -> str:
    L = [f"# Route B: {s['run']} ({s['atlas']})", "",
         f"{s['n_frames']} frames for coupling and integration"
         + (f" ({s['timeline']['status']}; {s['timeline']['n_total']} frames in the run, {s['timeline']['n_segments']} gap-free stretch(es); shares use all)"
            if s.get("timeline") else "") + f", nuisance rank {s['nuisance_rank']}, band {s['band_hz'][0]}-{s['band_hz'][1]} Hz. "
         "Shares: BOLD-like = R2*-like energy / (R2*-like + S0-like), thermal noise removed; leftover = energy outside the "
         "S0 + R2* model. Integration on the noise-corrected correlations of the regions' R2*-like time courses, against "
         "circular shifts (each region its own lag).", "", "## Integration", "",
         "Segregation null: network labels permuted over the regions (liberal: ignores that neighbouring parcels share "
         "networks). Thalamus and striatum coupling noise-corrected like the cortical matrix (plain correlations in the "
         "JSON). Shares are clipped to 0-1; regions whose S0-like or R2*-like energy fell below its noise estimate are listed.", "",
         (f"Primary cortex: voxels with CSF probability > {s.get('cortex_csf_max')} removed (user, 2026-11-11). "
          + (f"Cortical CSF fraction (covariate): mean CSF probability {_r(s['cortex_csf_covariate'].get('cortex_csf_mean_allvox'))} "
             f"over all cortical voxels; {s['cortex_csf_covariate'].get('cortex_voxels_removed')} of "
             f"{s['cortex_csf_covariate'].get('cortex_voxels_allvox')} voxels removed." if s.get("cortex_csf_covariate") else "")
          if s.get("cortex_csf_max") is not None else "Cortex: all voxels (no CSF exclusion)."), "",
         "| set | regions (with signal) | participation ratio | null median [5-95 %] | percentile in null | PR without first pattern | null median [5-95 %] | percentile in null | eigenvalues above null | segregation (null 95 %) | global share | edges beyond null |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i in s["integration"]:
        L.append(f"| {i['set']} | {i['n_regions']} ({i.get('n_regions_with_signal')}) | {i['participation_ratio']} | {i['pr_null_median']} {i['pr_null_q05_q95']} | "
                 f"{i['pr_percentile_in_null']} | {i.get('pr_without_first')} | {i.get('pr_without_first_null_median')} {i.get('pr_without_first_null_q05_q95')} | "
                 f"{i.get('pr_without_first_percentile_in_null')} | {i['n_eigen_above_null']} | {i['segregation']} ({i['segregation_null_q95']}; within {i.get('within_mean')}, between {i.get('between_mean')}) | "
                 f"{i['global_share']} | {i['edges_beyond_null']} |")
    L += ["", "## Shares by network (median over usable regions)", "", "| network | regions | BOLD-like | S0-like | leftover |", "|---|---|---|---|---|"]
    for net, m in s["network_medians"].items():
        L.append(f"| {net} | {m['n']} | {m['bold_share']} | {m['s0_share']} | {m['leftover_share']} |")
    sn = s.get("shared_noise_check")
    if sn:
        full = s["integration"][0]
        L += ["", "## Noise shared between neighbouring regions (diagnostic, all usable cortical regions)", "",
              "Frame-difference correlation of the leftover channels (noise only) between regions; the noise correction "
              "assumes it is 0. Then the integration measures with that shared noise also removed.", "",
              f"- adjacent pairs ({sn['n_adjacent_pairs']} of {sn['n_pairs']}): median {sn['noise_corr_adjacent_median']}, "
              f"90th percentile {sn['noise_corr_adjacent_q90']}",
              f"- non-adjacent pairs: median {sn['noise_corr_nonadjacent_median']}, 90th percentile {sn['noise_corr_nonadjacent_q90']} "
              f"(sampling SE {sn['sampling_se']})",
              f"- participation ratio {full['participation_ratio']} -> {sn['participation_ratio_shared_noise_removed']}; "
              f"without first pattern {full.get('pr_without_first')} -> {sn['pr_without_first_shared_noise_removed']}; "
              f"correlations above 1: {full.get('n_edges_above_1')} -> {sn['n_edges_above_1_shared_noise_removed']}"]
    flag = [r for r in s["regions"] if r.get("energy_below_noise")]
    sub = [r for r in s["regions"] if r["kind"] == "secondary"]
    if sub:
        L += ["", "## Thalamus and striatum, each region", "",
              "Noise ratio: region noise / the thermal noise of independent voxels (frame differences; cortex median below). "
              "Spectral / difference: the robust median-spectrum estimate over the frame-difference one (~1 for white "
              "noise; below 1 where fast structure inflates the difference estimate). Tissue: mean FAST probabilities.", "",
              "| region | voxels | used | BOLD-like [95 % CI] | S0-like | leftover | energy below noise | noise ratio | spectral / difference | reliability | GM | CSF |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            L.append(f"| {r['name']} | {r['n_voxels']} | {r['used']} | {r['bold_share']} {r['bold_share_ci']} | {r['s0_share']} | {r['leftover_share']} | "
                     f"{r.get('energy_below_noise') or '-'} | {r['noise_var']} | {r.get('noise_var_spectral_ratio')} | {r.get('b_reliability')} | {r.get('gm_mean')} | {r.get('csf_mean')} |")
    cx = [r for r in s["regions"] if r["kind"] == "cortex" and r["used"]]
    med = lambda k: _r(np.median([r[k] for r in cx if r.get(k) is not None])) if any(r.get(k) is not None for r in cx) else None
    L += ["", "Reliability: share of a region's slow BOLD-like variance that is signal (1 - noise / total); noise "
          "correction divides by its square root, so it is unstable where reliability is low.", "",
          f"Cortex (median over used regions): noise ratio {med('noise_var')}, spectral / difference {med('noise_var_spectral_ratio')}, "
          f"reliability {med('b_reliability')}, GM {med('gm_mean')}, CSF {med('csf_mean')}."]
    for i in s["integration"]:
        L.append(f"- {i['set']}: {i.get('n_edges_above_1')} of {i.get('n_edges')} correlations above 1 after noise correction; "
                 f"reliability of the regions involved (median) {i.get('edges_above_1_reliability_median')} against "
                 f"{i.get('reliability_median')} for all (5th percentile {i.get('reliability_q05')})")
    L += ["", f"Regions with an energy below its noise estimate (share clipped): {len(flag)}"
          + (": " + ", ".join(f"{r['name']} ({r['energy_below_noise']})" for r in flag[:20]) + (" ..." if len(flag) > 20 else "") if flag else "")]
    if s.get("deep_csf_check"):
        nets_ = [k for k in s["secondary_coupling"][0] if k not in ("name", "noise_overcorrected", "ci", "b_reliability", "not_measurable")] if s["secondary_coupling"] else []
        L += ["", "## Thalamus and striatum: CSF check (diagnostic; the outcome labels are unchanged)", "",
              "Weight share: the part of the region's averaging weight in voxels with CSF probability > 0.5, per echo "
              "(CSF is bright at late echoes, so it weighs more than its voxel count). Then the region recomputed "
              "without voxels above each CSF threshold.", "",
              "| region | voxels (CSF > 0.5) | weight share in CSF voxels, echoes 1-4 | version | voxels | BOLD-like | leftover | reliability | " + " | ".join(nets_) + " |",
              "|---|---|---|---|---|---|---|---|" + "---|" * len(nets_)]
        for d in s["deep_csf_check"]:
            for j, key in enumerate([k for k in d if k == "all_voxels" or k.startswith("csf_below_")]):
                v = d[key]
                head = f"| {d['name']} | {d['n_voxels']} ({d['n_csf_dominated']}) | {d['csf_weight_share_by_echo']} " if j == 0 else "| | | "
                if v is None:
                    L.append(head + f"| {key} | too few voxels | | | |" + " |" * len(nets_))
                else:
                    L.append(head + f"| {key} | {v['n_voxels']} | {v['bold_share']} | {v['leftover_share']} | {v['reliability']} | "
                             + " | ".join(str(v.get(n)) for n in nets_) + " |")
    if s["secondary_coupling"]:
        nets = [k for k in s["secondary_coupling"][0] if k not in ("name", "noise_overcorrected", "ci", "b_reliability", "not_measurable")]
        L += ["", "## Thalamus and striatum: BOLD-like coupling with each cortical network's mean", "",
              "Noise-corrected, with block-bootstrap 95 % intervals; n.m. = not measurable (reliability below the floor).", "",
              "| region | " + " | ".join(nets) + " |", "|---|" + "---|" * len(nets)]
        for r in s["secondary_coupling"]:
            ci = r.get("ci") or {}
            if r.get("not_measurable"):
                L.append(f"| {r['name']} (reliability {r.get('b_reliability')}: not measurable) | "
                         + " | ".join(f"n.m. {ci.get(n)}" for n in nets) + " |")
            else:
                L.append(f"| {r['name']} (reliability {r.get('b_reliability')}){' (a value above 1)' if r.get('noise_overcorrected') else ''} | "
                         + " | ".join(f"{r[n]} {ci.get(n)}" for n in nets) + " |")
    return "\n".join(L) + "\n"


def figure(summ, res, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    nm = summ["network_medians"]
    nets = list(nm)
    x = np.arange(len(nets))
    b = np.array([nm[n]["bold_share"] or 0 for n in nets]); s0 = np.array([nm[n]["s0_share"] or 0 for n in nets])
    axs[0].bar(x, b, color="#2a78d6", label="BOLD-like")
    axs[0].bar(x, s0, bottom=b, color="#eb6834", label="S0-like")
    axs[0].set_xticks(x, nets, rotation=45, ha="right", fontsize=8)
    axs[0].set_ylabel("share of slow variance (median region)")
    axs[0].legend(fontsize=7, frameon=False)
    nets_c = np.array(res["cortex_networks"])
    order = np.argsort(nets_c, kind="stable")
    C = res["corr_b"][np.ix_(order, order)]
    im = axs[1].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    axs[1].set_title("BOLD-like correlations (ordered by network)", fontsize=9, loc="left")
    plt.colorbar(im, ax=axs[1], fraction=0.046)
    i0 = summ["integration"][0]
    k = len(i0["eigenvalues"])
    axs[2].plot(np.arange(1, k + 1), i0["eigenvalues"], "o-", color="#2a78d6", ms=3, label="observed")
    axs[2].plot(np.arange(1, len(i0["null_eig_q95"]) + 1), i0["null_eig_q95"], "--", color="#52514e", label="circular shift, 95 %")
    axs[2].set_yscale("log")
    axs[2].set_title(f"eigenvalues; participation ratio {i0['participation_ratio']} (null median {i0['pr_null_median']})", fontsize=9, loc="left")
    axs[2].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{summ['run']}: route B", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
