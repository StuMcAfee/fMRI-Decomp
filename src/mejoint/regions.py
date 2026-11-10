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


def analyze(S, labels, mu, sigma, ratios, use, te_ms, table: list, cfg: dict, Q=None, rng=None, common_ids=None) -> dict:
    """Route B on one run, arrays version. S (V, E, T) masked magnitudes; labels (V,) region ids; table: region
    rows from anatomy (id, name, kind, network, n_voxels, usable). Returns per-region rows, the coupling
    matrices, integration measures (full usable set and, if given, the common set) and their circular-shift nulls."""
    G = cfg.get("regions", {})
    tr = float(cfg["acquisition"]["tr_s"])
    band = tuple(G.get("band_hz", [0.01, 0.10]))
    rng = np.random.default_rng(0) if rng is None else rng
    nT = G.get("n_volumes_common")
    if nT:
        S = S[..., :int(nT)]
        Q = None if Q is None else np.linalg.qr(Q[:int(nT)])[0]
    P = int(max(r["id"] for r in table))
    rs = region_signals(S, labels, mu, sigma, ratios, use, n_regions=P)
    sp = split(rs["ybar"], rs["wsum"], te_ms, Q)
    removed = 0 if Q is None else Q.shape[1]
    nv = leftover_noise(sp["r"], removed)
    nv = np.where(np.isfinite(nv) & (nv > 0), nv, np.nan)
    sh = shares(sp, np.nan_to_num(nv, nan=1.0), tr, band)
    lo, hi = share_ci(sp, np.nan_to_num(nv, nan=1.0), tr, band, int(G.get("n_boot", 200)),
                      max(2, int(round(float(G.get("boot_block_s", 20.0)) / tr))), rng)
    T = S.shape[2]
    rows = []
    for r in table:
        p = r["id"] - 1
        ok = bool(r.get("usable")) and bool(np.isfinite(nv[p])) and bool(np.isfinite(sp["c"][p]).all())
        rows.append({**{k: r[k] for k in ("id", "name", "kind", "hemisphere", "network", "n_voxels")}, "used": ok,
                     "noise_var": _r(nv[p]), "bold_share": _r(sh["bold_share"][p]) if ok else None,
                     "bold_share_ci": [_r(lo[p]), _r(hi[p])] if ok else None,
                     "s0_share": _r(sh["s0_share"][p]) if ok else None, "leftover_share": _r(sh["leftover_share"][p]) if ok else None,
                     "E_S0": _r(sh["E_S0"][p]), "E_R2": _r(sh["E_R2"][p]), "E_left": _r(sh["E_left"][p])})
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
                "pr_without_first": _r(pr1),
                "pr_without_first_null_median": _r(np.nanmedian(null["pr_resid"])),
                "pr_without_first_null_q05_q95": [_r(np.nanquantile(null["pr_resid"], 0.05)), _r(np.nanquantile(null["pr_resid"], 0.95))],
                "pr_without_first_percentile_in_null": _r(float(np.mean(null["pr_resid"] <= pr1))),
                "n_eigen_above_null": int(count_above_null(eig, null["eig"])),
                "segregation": _r(segregation(c_["corr"], nets[cx][mask_sub])),
                "segregation_null_q95": _r(np.nanquantile(null["segregation"], 0.95)) if np.isfinite(null["segregation"]).any() else None,
                "global_share": _r(global_share(c_["x_band"], c_["signal_var"], c_["noise_band_var"])),
                "eigenvalues": [round(float(x), 4) for x in eig[:30]],
                "null_eig_q95": [round(float(x), 4) for x in np.nanquantile(null["eig"], 0.95, axis=0)[:30]] if null["eig"].size else [],
                "edges_beyond_null": _r(_edge_fraction(c_["corr"], null["edges"]))}
    full = integration(np.ones(sel.size, bool), "all usable cortical regions")
    out_int = [full]
    if common_ids is not None:
        m = np.isin(ids[cx], np.asarray(list(common_ids), int))
        if m.sum() >= 3:
            out_int.append(integration(m, "common set"))
    # secondary regions: coupling of their b with each cortical network's mean b
    sec = []
    xb_all = band_pass(sp["b"], tr, band)
    for r in [x for x in rows if x["kind"] == "secondary" and x["used"]]:
        p = r["id"] - 1
        row = {"name": r["name"]}
        for net in sorted(set(nets[cx])):
            m = xb_all[sel[nets[cx] == net]].mean(0)
            row[net] = _r(np.corrcoef(xb_all[p], m)[0, 1])
        sec.append(row)
    return {"regions": rows, "integration": out_int, "secondary_coupling": sec, "cortex_ids": ids[cx].tolist(),
            "cortex_networks": nets[cx].tolist(), "corr_b": cb["corr"], "corr_a": ca["corr"], "corr_oc": co["corr"],
            "n_frames": int(T), "nuisance_rank": int(removed), "band_hz": list(band),
            "a": sp["a"], "b": sp["b"], "c0": sp["c"][:, 0], "R": sp["R"], "noise_var": nv}


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
    res = analyze(S, lab3[mask3d], mu, sigma, fs["echo_noise_ratios_used"], use, fs["te_ms"], table, cfg, Q,
                  np.random.default_rng(int(cfg.get("project_seed", 0)) + 11), common)
    for k in ("corr_b", "corr_a", "corr_oc"):
        np.save(out / f"{k}.npy", res[k].astype(np.float32))
    # each region's time courses (row = region id - 1) for recovery against injected truth and later analyses
    np.savez_compressed(out / "region_timecourses.npz", a=res["a"].astype(np.float32), b=res["b"].astype(np.float32),
                        c0=res["c0"].astype(np.float32), R=res["R"], noise_var=res["noise_var"],
                        nuisance_Q=(np.load(fe / "nuisance_Q.npy") if (fe / "nuisance_Q.npy").exists() else np.zeros((res["n_frames"], 0))))
    rows = res["regions"]
    io.write_tsv(out / "regions_route_b.tsv", {k: [(";".join(map(str, r[k])) if isinstance(r[k], list) else r[k]) for r in rows] for k in rows[0]})
    summ = {"run": fs.get("run_id"), "atlas": atlas, "frontend_dir": str(fe), "anat_dir": str(A), "n_frames": res["n_frames"],
            "nuisance_rank": res["nuisance_rank"], "band_hz": res["band_hz"], "integration": res["integration"],
            "secondary_coupling": res["secondary_coupling"], "cortex_ids": res["cortex_ids"], "regions": rows,
            "network_medians": _network_medians(rows), **io.provenance(cfg, {})}
    io.write_json(out / "route_b.json", summ)
    io.write_text(out / "route_b.md", render(summ))
    try:
        figure(summ, res, out / "route_b.png")
    except Exception as exc:
        print(f"route B figure failed: {exc}")
    return summ


def _network_medians(rows):
    out = {}
    for net in sorted({r["network"] for r in rows if r["used"]}):
        rr = [r for r in rows if r["used"] and r["network"] == net]
        med = lambda k: _r(np.median([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None
        out[net] = {"n": len(rr), "bold_share": med("bold_share"), "s0_share": med("s0_share"), "leftover_share": med("leftover_share")}
    return out


def render(s: dict) -> str:
    L = [f"# Route B: {s['run']} ({s['atlas']})", "",
         f"{s['n_frames']} frames, nuisance rank {s['nuisance_rank']}, band {s['band_hz'][0]}-{s['band_hz'][1]} Hz. "
         "Shares: BOLD-like = R2*-like energy / (R2*-like + S0-like), thermal noise removed; leftover = energy outside the "
         "S0 + R2* model. Integration on the noise-corrected correlations of the regions' R2*-like time courses, against "
         "circular shifts (each region its own lag).", "", "## Integration", "",
         "| set | regions | participation ratio | null median [5-95 %] | percentile in null | PR without first pattern | null median [5-95 %] | percentile in null | eigenvalues above null | segregation (null 95 %) | global share | edges beyond null |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i in s["integration"]:
        L.append(f"| {i['set']} | {i['n_regions']} | {i['participation_ratio']} | {i['pr_null_median']} {i['pr_null_q05_q95']} | "
                 f"{i['pr_percentile_in_null']} | {i.get('pr_without_first')} | {i.get('pr_without_first_null_median')} {i.get('pr_without_first_null_q05_q95')} | "
                 f"{i.get('pr_without_first_percentile_in_null')} | {i['n_eigen_above_null']} | {i['segregation']} ({i['segregation_null_q95']}) | "
                 f"{i['global_share']} | {i['edges_beyond_null']} |")
    L += ["", "## Shares by network (median over usable regions)", "", "| network | regions | BOLD-like | S0-like | leftover |", "|---|---|---|---|---|"]
    for net, m in s["network_medians"].items():
        L.append(f"| {net} | {m['n']} | {m['bold_share']} | {m['s0_share']} | {m['leftover_share']} |")
    if s["secondary_coupling"]:
        nets = [k for k in s["secondary_coupling"][0] if k != "name"]
        L += ["", "## Thalamus and striatum: BOLD-like coupling with each cortical network's mean", "",
              "| region | " + " | ".join(nets) + " |", "|---|" + "---|" * len(nets)]
        for r in s["secondary_coupling"]:
            L.append(f"| {r['name']} | " + " | ".join(str(r[n]) for n in nets) + " |")
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
