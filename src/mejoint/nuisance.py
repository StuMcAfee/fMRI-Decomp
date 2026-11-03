"""Phase 2b (CLAUDE.md section 8): one nuisance projector for the whole front end.

Regressors come from the run's motion parameters (mcflirt .par: rotations x, y, z in rad, then
translations x, y, z in mm; one row per kept volume). In fast-TR data, breathing changes the B0 field and
makes the head appear to move (pseudo-motion); it shows in the motion parameters as a peak at the
breathing rate. The parameters are band-stopped there (or low-passed) before they become regressors or FD.
No physiological recordings or phase exist for this study (Phase 0), so RETROICOR / RVT / dB0(t)
regressors are not available.

The projector P = I - Q Q^T (Q: orthonormal basis of [one constant per contiguous segment, regressors])
acts on time. The front end mixes echoes per voxel with a fixed matrix, so projecting the log-ratio echoes
y and then forming the channels equals forming the channels and then projecting them (identical
regressors commute; tested). The front end therefore projects the channels c and the residual r.

After projection the thermal noise variance falls by tr(P) / T on average and that of the frame-to-frame
differenced residual by tr(D P D^T) / (2 (T - 1)) (D = first differences). Sigma estimates are corrected
by these exact factors: this is the "reduce the temporal df by p + 1" of CLAUDE.md Phase 2b.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def segments(volumes=None, n: int | None = None) -> list[slice]:
    """Contiguous stretches of kept frames. volumes: original frame numbers of the kept volumes (echo 1,
    from volumes_used.tsv); a jump > 1 starts a new segment. None: one segment of n frames."""
    if volumes is None:
        return [slice(0, int(n))]
    v = np.asarray(volumes, int)
    edges = np.r_[0, np.flatnonzero(np.diff(v) != 1) + 1, v.size]
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def framewise_displacement(par: np.ndarray, segs: list[slice], radius_mm: float = 50.0) -> np.ndarray:
    """Power et al. (2012) FD: sum of absolute frame-to-frame changes, rotations as arc length on a
    50 mm sphere. 0 at the first frame of every segment (no change is defined across a gap)."""
    d = np.zeros(len(par))
    for s in segs:
        x = par[s]
        if len(x) > 1:
            dd = np.abs(np.diff(x, axis=0))
            d[s.start + 1:s.stop] = dd[:, :3].sum(1) * radius_mm + dd[:, 3:6].sum(1)
    return d


def respiratory_peak(par: np.ndarray, tr: float, search_hz, segs: list[slice], nperseg: int = 128) -> dict | None:
    """Breathing peak in the motion parameters: Welch spectra of the detrended, standardised parameters,
    summed over the 6 parameters and averaged over segments long enough for one window.
    Head motion has a sloping (roughly power-law) spectrum, so the peak is measured against a robust
    straight-line fit of log power vs log frequency over the search band:
    prominence = power / fitted background at the frequency where that ratio is largest."""
    lo, hi = float(search_hz[0]), float(search_hz[1])
    acc, n_used, f = None, 0, None
    for s in segs:
        x = par[s]
        if len(x) < nperseg:
            continue
        x = signal.detrend(x, axis=0)
        sd = x.std(0)
        x = x[:, sd > 0] / sd[sd > 0]
        f, p = signal.welch(x, fs=1.0 / tr, nperseg=nperseg, axis=0)
        p = p.sum(1) * len(x)
        acc = p if acc is None else acc + p
        n_used += len(x)
    if acc is None:
        return None
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return None
    from scipy.stats import theilslopes
    lf, lp = np.log(f[band]), np.log(np.maximum(acc[band], 1e-300))
    slope, icpt, _, _ = theilslopes(lp, lf)
    excess = lp - (icpt + slope * lf)
    j = int(np.argmax(excess))
    i = np.flatnonzero(band)[j]
    return {"freq_hz": round(float(f[i]), 4), "prominence": round(float(np.exp(excess[j])), 3),
            "background_slope": round(float(slope), 3),
            "frequency_resolution_hz": round(float(f[1] - f[0]), 4), "frames_used": int(n_used)}


def filter_motion(par: np.ndarray, tr: float, N: dict, segs: list[slice]) -> tuple[np.ndarray, dict]:
    """Remove respiratory pseudo-motion from the motion parameters, zero-phase, segment by segment.
    notch: Butterworth band-stop at the detected breathing peak +/- notch_halfwidth_hz (skipped, and
    reported, when the peak is not prominent); lowpass: Butterworth low-pass; none: unchanged."""
    kind = N.get("motion_filter", "notch")
    info = {"kind": kind, "applied": False}
    if kind == "none":
        return par.copy(), info
    fs, order = 1.0 / tr, int(N.get("filter_order", 4))
    nyq = 0.5 * fs
    if kind == "notch":
        pk = respiratory_peak(par, tr, N["resp_search_hz"], segs)
        info["respiratory_peak"] = pk
        if pk is None or pk["prominence"] < float(N["min_peak_prominence"]):
            info["reason"] = "no prominent breathing peak in the motion parameters"
            return par.copy(), info
        w = float(N["notch_halfwidth_hz"])
        band = [max(pk["freq_hz"] - w, 0.005), min(pk["freq_hz"] + w, 0.98 * nyq)]
        sos = signal.butter(order, band, btype="bandstop", fs=fs, output="sos")
        info["band_hz"] = [round(b, 4) for b in band]
    elif kind == "lowpass":
        sos = signal.butter(order, float(N["lowpass_hz"]), btype="lowpass", fs=fs, output="sos")
        info["cutoff_hz"] = float(N["lowpass_hz"])
    else:
        raise ValueError("nuisance.motion_filter must be notch, lowpass or none")
    out = par.astype(np.float64).copy()
    padlen = 3 * (2 * sos.shape[0] + 1)
    skipped = 0
    for s in segs:
        if s.stop - s.start > padlen:
            out[s] = signal.sosfiltfilt(sos, out[s], axis=0)
        else:
            skipped += 1                                     # too short to filter: left unfiltered
    info.update(applied=True, segments_unfiltered=skipped)
    return out, info


def motion_regressors(par: np.ndarray, segs: list[slice], expansion: int = 24) -> tuple[np.ndarray, list[str]]:
    """6: parameters; 12: + frame-to-frame changes (0 at segment starts); 24: + squares of both
    (of the demeaned columns). Columns are demeaned; the projector adds the segment constants."""
    names6 = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    p = par[:, :6] - par[:, :6].mean(0)
    cols, names = [p], list(names6)
    if expansion not in (6, 12, 24):
        raise ValueError("nuisance.expansion must be 6, 12 or 24")
    if expansion >= 12:
        d = np.zeros_like(p)
        for s in segs:
            d[s.start + 1:s.stop] = np.diff(p[s], axis=0)
        cols.append(d)
        names += [f"d_{n}" for n in names6]
    if expansion == 24:
        sq = np.concatenate(cols, axis=1) ** 2
        cols.append(sq - sq.mean(0))
        names += [f"{n}_sq" for n in names]
    return np.concatenate(cols, axis=1), names


def polynomial_regressors(segs: list[slice], T: int, order: int) -> tuple[np.ndarray, list[str]]:
    """Legendre polynomials of order 1..order within each segment (0 elsewhere)."""
    cols, names = [], []
    for j, s in enumerate(segs):
        n = s.stop - s.start
        x = np.linspace(-1, 1, n) if n > 1 else np.zeros(n)
        for k in range(1, order + 1):
            c = np.zeros(T)
            c[s] = np.polynomial.legendre.Legendre.basis(k)(x)
            cols.append(c)
            names.append(f"poly{k}_seg{j}")
    return (np.column_stack(cols) if cols else np.zeros((T, 0))), names


class Projector:
    """P = I - Q Q^T over time. Q: orthonormal basis (SVD, rank-revealing) of one constant per segment
    plus the regressors X (T, p)."""

    def __init__(self, X: np.ndarray, segs: list[slice], T: int, tol: float = 1e-8):
        const = np.zeros((T, len(segs)))
        for j, s in enumerate(segs):
            const[s, j] = 1.0
        C = np.concatenate([const, np.asarray(X, float).reshape(T, -1)], axis=1)
        nrm = np.linalg.norm(C, axis=0)
        C = C[:, nrm > 0] / nrm[nrm > 0]
        U, sv, _ = np.linalg.svd(C, full_matrices=False)
        keep = sv > tol * sv[0]
        self.Q = U[:, keep]
        self.T = int(T)
        self.rank = int(keep.sum())                           # p + 1 (with one segment and full-rank X)
        self.n_columns = int(C.shape[1])

    def apply(self, a: np.ndarray) -> np.ndarray:
        """Project the last (time) axis. Rows containing NaN stay NaN (undefined channels)."""
        a64 = np.asarray(a, np.float64)
        out = a64 - (a64 @ self.Q) @ self.Q.T
        return out.astype(a.dtype, copy=False) if np.issubdtype(np.asarray(a).dtype, np.floating) else out

    def kept_per_frame(self) -> np.ndarray:
        """Noise variance kept at each frame: 1 - h_t (h_t = leverage, the diagonal of Q Q^T). Frames with
        unusual regressor values (large motion) keep less noise than the average var_factor."""
        return 1.0 - (self.Q ** 2).sum(1)

    def var_factor(self) -> float:
        """Mean noise variance kept by P: tr(P) / T = (T - rank) / T."""
        return (self.T - self.rank) / self.T

    def diff_factor(self) -> float:
        """Noise variance of first differences kept by P: tr(D P D^T) / (2 (T - 1))."""
        DQ = np.diff(self.Q, axis=0)
        return (2.0 * (self.T - 1) - float((DQ ** 2).sum())) / (2.0 * (self.T - 1))

    def factor(self, difference: bool) -> float:
        return self.diff_factor() if difference else self.var_factor()


def build(par: np.ndarray, tr: float, cfg: dict, volumes=None) -> tuple[Projector, dict]:
    """Projector and a record of what went into it (filter, regressors, rank, noise factors, FD)."""
    N = cfg["nuisance"]
    par = np.asarray(par, float)
    T = len(par)
    segs = segments(volumes, T)
    if sum(s.stop - s.start for s in segs) != T:
        raise ValueError("motion parameters and kept volumes disagree in length")
    pf, finfo = filter_motion(par, tr, N, segs)
    X, names = motion_regressors(pf, segs, int(N.get("expansion", 24)))
    Xp, pnames = polynomial_regressors(segs, T, int(N.get("poly_order", 0)))
    X, names = np.concatenate([X, Xp], axis=1), names + pnames
    P = Projector(X, segs, T)
    fd_raw = framewise_displacement(par, segs)
    fd_f = framewise_displacement(pf, segs)
    info = {"enabled": True, "filter": finfo, "expansion": int(N.get("expansion", 24)),
            "poly_order": int(N.get("poly_order", 0)), "n_regressors": len(names), "n_segments": len(segs),
            "rank": P.rank, "var_factor": round(P.var_factor(), 5), "diff_factor": round(P.diff_factor(), 5),
            "fd_mean_unfiltered": round(float(fd_raw.mean()), 4), "fd_mean_filtered": round(float(fd_f.mean()), 4),
            "fd_p95_unfiltered": round(float(np.percentile(fd_raw, 95)), 4),
            "fd_p95_filtered": round(float(np.percentile(fd_f, 95)), 4)}
    return P, {"info": info, "X": X, "names": names, "fd_filtered": fd_f, "fd_unfiltered": fd_raw,
               "par_filtered": pf, "segments": segs}
