"""Phase 3 (CLAUDE.md section 8): how many components the stacked whitened channels [c0 | c1] hold.

Spectrum: eigenvalues of the T x T Gram matrix of the stacked, time-demeaned channels (float64), divided
by the number of rows, so unit white noise sits in the Marchenko-Pastur band (1 +/- sqrt(T_eff / n))^2.

Nulls (the same nuisance projection and demeaning as the data):
- temporal (gated): every row phase-randomized independently. Each row keeps its own power spectrum
  (signal and coloured noise included), but structure shared across rows is destroyed. Because rows keep
  the power of the shared components, an index-by-index comparison (classic parallel analysis; reported)
  hides weak components when the rows share a spectrum (narrowband, or 1/f like BOLD). The gated estimator
  is therefore deflated (sequential) parallel analysis (Buja & Eyuboglu 1992, from memory): component k
  passes if the data's k-th eigenvalue exceeds the chosen quantile of the top eigenvalue of
  phase-randomized data with the top k-1 components removed (the removed components and the nuisance
  basis are projected out of the null too). The search steps one component at a time from the index-by-index estimate.
- thermal: the residual channels r0, r1, which hold thermal noise and violations but no signal in
  span{1, TE}. The data rows are subsampled to the residual's row count (same aspect ratio).
  Physiological noise lies inside span{1, TE}, so this edge sits low and pushes k up (CLAUDE.md caveat);
  it is reported, not gated.
- analytic: the Marchenko-Pastur edge for unit noise. The thermal sigma level is uncertain by up to
  ~20 % (Phase 2 MP-PCA check), i.e. ~44 % in variance; reported only.
Fixed sweep for later phases: MAPCA AIC / KIC / MDL (the `mapca` package tedana uses, normalize=True as in
tedana) on the stacked channel maps, plus fixed k from the config.
"""
from __future__ import annotations

import numpy as np


def stack(c0: np.ndarray, c1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows of c0 and c1 with all frames finite (undefined c1 rows excluded, never zero-filled),
    demeaned over time. Returns (X (n, T) float64, row index into [c0 rows | c1 rows])."""
    A = np.concatenate([c0, c1], axis=0)
    ok = np.isfinite(A).all(axis=1)
    X = A[ok].astype(np.float64)
    X -= X.mean(axis=1, keepdims=True)
    return X, np.flatnonzero(ok)


def gram(X: np.ndarray, chunk: int = 20000) -> np.ndarray:
    T = X.shape[1]
    G = np.zeros((T, T))
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk]
        G += x.T @ x
    return G / X.shape[0]


def spectrum(X: np.ndarray, vectors: bool = False):
    """Eigenvalues (descending) of X^T X / n (float64); with vectors=True also the temporal eigenvectors."""
    if vectors:
        lam, V = np.linalg.eigh(gram(X))
        return np.clip(lam[::-1], 0, None), V[:, ::-1]
    return np.clip(np.linalg.eigvalsh(gram(X))[::-1], 0, None)


def phase_randomize_rows(X: np.ndarray, rng, chunk: int = 20000, mirror: bool = True) -> np.ndarray:
    """Independent random phases per row: each row's spectrum kept, shared timing destroyed.
    mirror=True: each row is mirror-extended to length 2T before randomizing and the first T samples are
    kept. Plain phase randomization treats the series as periodic and has no half-cycle frequencies, so it
    under-represents the slowest mode of non-periodic data with a sloping spectrum (AR noise, 1/f BOLD);
    the null then sits too low (test: coloured noise top eigenvalue 3.6 % low without mirroring)."""
    T = X.shape[1]
    out = np.empty_like(X)
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk]
        if mirror:
            x = np.concatenate([x, x[:, ::-1]], axis=1)
        L = x.shape[1]
        F = np.fft.rfft(x, axis=1)
        ph = rng.uniform(0, 2 * np.pi, F.shape)
        ph[:, 0] = 0.0
        if L % 2 == 0:
            ph[:, -1] = 0.0
        out[i:i + chunk] = np.fft.irfft(F * np.exp(1j * ph), n=L, axis=1)[:, :T]
    return out


def project(X: np.ndarray, Q: np.ndarray | None) -> np.ndarray:
    return X if Q is None else X - (X @ Q) @ Q.T


def k_above(lam: np.ndarray, thr: np.ndarray) -> int:
    """Number of leading eigenvalues above their threshold (stops at the first one below)."""
    below = np.flatnonzero(lam <= thr)
    return int(below[0]) if below.size else int(lam.size)


def temporal_null(X: np.ndarray, Q, n_sur: int, rng) -> np.ndarray:
    """(n_sur, T) eigenvalues of phase-randomized surrogates, projected and demeaned like the data."""
    out = []
    for _ in range(int(n_sur)):
        Y = project(phase_randomize_rows(X, rng), Q)
        Y -= Y.mean(axis=1, keepdims=True)
        out.append(spectrum(Y))
    return np.array(out)


def deflated_pass(X, lam, V, k: int, Q, n_sur: int, q: float, rng) -> tuple[bool, float]:
    """Does component k (1-based) exceed the null built from the data minus its top k-1 components?"""
    B = V[:, :k - 1]
    Y = X - (X @ B) @ B.T if k > 1 else X
    tops = []
    for _ in range(int(n_sur)):
        Z = project(phase_randomize_rows(Y, rng), Q)
        if k > 1:
            Z = Z - (Z @ B) @ B.T
        Z -= Z.mean(axis=1, keepdims=True)
        Z *= np.sqrt((Y ** 2).sum() / (Z ** 2).sum())          # same total variance as the deflated data
        tops.append(spectrum(Z)[0])
    thr = float(np.quantile(tops, q))
    return bool(lam[k - 1] > thr), thr


def deflated_k(X, lam, V, Q, n_sur: int, q: float, rng, k_max: int, k_start: int = 1) -> tuple[int, dict]:
    """Sequential deflated test: k = the last component before the first failure. Starts at k_start (the
    index-by-index estimate, which errs low when rows share a spectrum); if k_start itself fails, steps
    down. Testing far above the true k is unsafe (removing many noise directions makes the test pass by
    mistake with coloured noise), so the search only moves one component at a time from the start."""
    tested = {}

    def ok(k):
        if k not in tested:
            tested[k] = deflated_pass(X, lam, V, k, Q, n_sur, q, rng)
        return tested[k][0]

    k = max(1, min(int(k_start), int(k_max)))
    if not ok(k):
        while k > 1 and not ok(k - 1):
            k -= 1
        return k - 1, tested                                    # k - 1 passed (or k = 1 failed: 0)
    while k < int(k_max) and ok(k + 1):
        k += 1
    return k, tested


def mp_edge(n: int, t_eff: int) -> float:
    return float((1 + np.sqrt(t_eff / n)) ** 2)


def mapca_estimates(c0, c1, mask3d) -> dict:
    """MAPCA AIC / KIC / MDL (tedana's estimator) on the stacked channels: c0 and c1 maps placed side by
    side along x with a 2-voxel gap, so MAPCA's spatial subsampling sees two brains, not one."""
    try:
        import mapca
        import nibabel as nib
        from mapca import MovingAveragePCA
    except Exception as exc:                                   # not installed: reported, not fatal
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    X, Y, Z = mask3d.shape
    T = c0.shape[1]
    gap = 2
    vol = np.zeros((2 * X + gap, Y, Z, T), np.float32)
    m = np.zeros((2 * X + gap, Y, Z), np.uint8)
    for j, c in enumerate((c0, c1)):
        ok = np.isfinite(c).all(1)
        block = np.zeros((X, Y, Z, T), np.float32)
        bm = np.zeros((X, Y, Z), bool)
        idx = np.argwhere(mask3d)
        block[tuple(idx[ok].T)] = c[ok]
        bm[tuple(idx[ok].T)] = True
        x0 = j * (X + gap)
        vol[x0:x0 + X] = block
        m[x0:x0 + X] = bm
    aff = np.eye(4)
    est = MovingAveragePCA(criterion="mdl", normalize=True)
    est.fit(nib.Nifti1Image(vol, aff), nib.Nifti1Image(m, aff))
    return {"available": True, "mapca_version": getattr(mapca, "__version__", "?"),
            "aic": int(est.aic_["n_components"]), "kic": int(est.kic_["n_components"]),
            "mdl": int(est.mdl_["n_components"]),
            "subsampling": {k: (float(v) if np.ndim(v) == 0 else np.asarray(v).tolist())
                            for k, v in (est.subsampling_ or {}).items()}}


def estimate(c0, c1, r0, r1, cfg: dict, Q=None, seed=None, mask3d=None, with_mapca: bool = True) -> dict:
    """All Phase 3 numbers for one set of channels. Q: nuisance basis (T, rank) or None."""
    Rk = cfg["rank"]
    rng = np.random.default_rng(seed)
    X, _ = stack(c0, c1)
    n, T = X.shape
    t_eff = T - (0 if Q is None else Q.shape[1]) - 1           # projector rank includes the constant; demeaning
    lam, V = spectrum(X, vectors=True)
    q = float(Rk["null_quantile"])
    null = temporal_null(X, Q, int(Rk["n_surrogates"]), rng)
    thr = np.quantile(null, q, axis=0)
    k_classic = k_above(lam, thr)
    k_max = min(int(Rk.get("k_max", 300)), t_eff - 1)
    k_temporal, tested = deflated_k(X, lam, V, Q, int(Rk["deflated_n_surrogates"]), q, rng, k_max, k_start=max(k_classic, 1))
    # thermal null: residual rows; data rows subsampled to the same aspect ratio
    Rr, _ = stack(r0, r1)
    n_r = Rr.shape[0]
    if n_r >= 2 * T:
        sub = np.sort(rng.choice(n, size=min(n, n_r), replace=False))
        lam_sub = spectrum(X[sub])
        lam_res = spectrum(Rr if n_r <= n else Rr[np.sort(rng.choice(n_r, size=n, replace=False))])
        k_thermal = k_above(lam_sub, np.full(T, lam_res[0]))
    else:
        lam_sub, lam_res, k_thermal = None, None, None
    edge = mp_edge(n, t_eff)
    out = {"n_rows": int(n), "n_frames": int(T), "t_eff": int(t_eff), "eigenvalues": lam,
           "null_quantile": q, "null_threshold": thr, "null_mean": null.mean(0), "n_surrogates": int(null.shape[0]),
           "k_temporal": k_temporal, "k_temporal_classic": k_classic, "k_max": k_max,
           "deflated_tests": {int(k): {"pass": v[0], "threshold": round(v[1], 5), "eigenvalue": round(float(lam[k - 1]), 5)}
                              for k, v in sorted(tested.items())},
           "k_thermal": k_thermal, "mp_edge": edge, "k_mp_edge": int((lam > edge).sum()),
           "eigenvalues_subsampled": lam_sub, "residual_eigenvalues": lam_res, "n_residual_rows": int(n_r),
           "sweep_fixed": [int(k) for k in Rk["sweep_fixed"]]}
    if with_mapca and mask3d is not None and Rk.get("mapca", True):
        out["mapca"] = mapca_estimates(c0, c1, mask3d)
    return out


def run_dir(frontend_dir, cfg: dict, out_dir) -> dict:
    """Phase 3 on one front-end output folder (channel cache c0/c1/r0/r1.npy, analysis mask,
    nuisance_Q.npy if the projection was applied). Writes spectrum.tsv, rank.json, rank.md, done.json."""
    import json
    from pathlib import Path

    import nibabel as nib

    from . import io

    fd, out_dir = Path(frontend_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ch = {k: np.load(fd / f"{k}.npy") for k in ("c0", "c1", "r0", "r1")}
    mask3d = np.asarray(nib.load(str(fd / "analysis_mask.nii.gz")).dataobj) > 0
    Qp = fd / "nuisance_Q.npy"
    Q = np.load(Qp) if Qp.exists() else None
    fs = json.loads((fd / "summary.json").read_text())
    import re
    m = re.match(r"sub-(\d+)", str(fs.get("run_id") or ""))
    subj = int(m.group(1)) if m else 0
    ss = io.seed_sequence(cfg, subj, 0, 3)                     # stage 3 = rank nulls
    res = estimate(ch["c0"], ch["c1"], ch["r0"], ch["r1"], cfg, Q=Q, seed=ss, mask3d=mask3d)
    T = res["n_frames"]
    col = lambda a: [float(x) for x in a] if a is not None else ["n/a"] * T
    io.write_tsv(out_dir / "spectrum.tsv", {"index": list(range(1, T + 1)), "eigenvalue": col(res["eigenvalues"]),
                                            "null_mean": col(res["null_mean"]), "null_threshold": col(res["null_threshold"]),
                                            "eigenvalue_subsampled": col(res["eigenvalues_subsampled"]),
                                            "residual_eigenvalue": col(res["residual_eigenvalues"])})
    keep = {k: v for k, v in res.items() if not isinstance(v, np.ndarray) and k not in ("eigenvalues_subsampled", "residual_eigenvalues")}
    keep.update(run_id=fs.get("run_id"), source=fs.get("source"), nuisance=Q is not None,
                nuisance_rank=None if Q is None else int(Q.shape[1]), frontend_dir=str(fd))
    io.write_json(out_dir / "rank.json", keep)
    io.write_text(out_dir / "rank.md", render(keep))
    io.write_config_copy(cfg, out_dir)
    io.write_json(out_dir / "done.json", {"stage": "rank", **io.provenance(cfg, {"frontend_summary": str(fd / "summary.json")})})
    return keep


def render(r: dict) -> str:
    L = ["**EXPLORATORY: the Phase 3 rank gate has not passed (no estimator yet recovers the simulated rank within 10 % "
         "with coloured noise; docs/decisions.md). Do not use these k as final.**", "",
         f"# Rank: {r.get('run_id')} ({r.get('source')} echoes, nuisance projection: "
         f"{'yes, rank ' + str(r['nuisance_rank']) if r.get('nuisance') else 'no'})", "",
         f"Stacked channels [c0 | c1]: {r['n_rows']} rows x {r['n_frames']} frames (effective {r['t_eff']} after projection "
         "and demeaning).", "",
         "| Estimate | k | Note |", "|---|---|---|",
         f"| Temporal null, deflated (intended gate; over-counts with coloured noise) | {r['k_temporal']} | each component above the {r['null_quantile']:.2f} "
         "quantile of the top eigenvalue of phase-randomized data with the stronger components removed |",
         f"| Temporal null, index by index | {r['k_temporal_classic']} | {r['n_surrogates']} surrogates; conservative "
         "when rows share a spectrum (under-counts) |",
         f"| Thermal null (residual) | {r['k_thermal']} | above the top residual eigenvalue, {r['n_residual_rows']} rows "
         "(aspect matched); sits low (physiology is inside span{1, TE}) |",
         f"| Marchenko-Pastur edge | {r['k_mp_edge']} | edge {r['mp_edge']:.3f} for unit noise; sigma level uncertain by ~20 % |"]
    mp = r.get("mapca") or {}
    if mp.get("available"):
        L += [f"| MAPCA AIC / KIC / MDL | {mp['aic']} / {mp['kic']} / {mp['mdl']} | tedana's estimator (mapca "
              f"{mp['mapca_version']}, normalize=True) on the stacked channel maps |"]
    elif mp:
        L += [f"| MAPCA | n/a | {mp.get('reason')} |"]
    L += ["", f"Fixed sweep for Phase 4: MAPCA estimates and k = {r['sweep_fixed']}."]
    return "\n".join(L) + "\n"
