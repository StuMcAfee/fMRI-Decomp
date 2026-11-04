"""Phase 3 (CLAUDE.md section 8): how many components the stacked whitened channels [c0 | c1] hold.

Spectrum: eigenvalues of the T x T Gram matrix of the stacked, time-demeaned channels (float64), divided
by the number of rows, so unit white noise sits in the Marchenko-Pastur band (1 +/- sqrt(T_eff / n))^2.

Nulls (the same nuisance projection and demeaning as the data):
- temporal, GATED (user decision 2026-11-04): split-half deflated parallel analysis on noise-whitened
  channels. (1) One common temporal whitening filter C^(-1/2) for all rows, C = Toeplitz of the median
  (across rows) autocovariance, tapered beyond max_lag: shared components keep their rank, the average
  noise spectrum becomes flat. (2) Voxels are split into two halves by spatial blocks (neighbouring voxels,
  whose noise can be correlated, stay together; c0 and c1 rows of a voxel stay together). (3) Temporal
  directions u_1, u_2, ... come from half A; the statistic for u_k is half B's variance along u_k (no
  selection bias: B's noise did not choose u_k). (4) Its null: half B with u_1..u_(k-1) removed,
  phase-randomized per row (mirror-extended), the same projections applied, variance rescaled, then
  measured along u_k. k = the last direction before the first failure; the two halves swap roles and
  k_temporal is their mean.
  Why: index-by-index parallel analysis inherits the shared components' spectral power and under-counts;
  plain deflation is biased by selection and, with coloured noise, by spectral depletion (it over-counts).
  Simulation (gate tests): 30 components, narrowband or broadband, white or coloured noise: within +/- 1;
  1 component: exact; 60 narrowband: 60-66; 60 broadband: over-counts 10-37 % because the many shared
  components contaminate the median noise spectrum. The contamination diagnostic reports this: validated
  up to ~0.45 (the 30-component cases score ~0.44; the failing 60-component cases ~0.63).
- temporal, reported: index-by-index parallel analysis (under-counts); the old unsplit deflated search
  (over-counts with coloured noise; off by default, slow).
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


def orthonormal(M: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    U, sv, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, sv > tol * max(sv.max(), 1e-300)]


def noise_whitener(X: np.ndarray, max_lag: int = 40, chunk: int = 20000) -> dict:
    """One temporal filter for all rows: W = C^(-1/2), C = Toeplitz of the median (across rows) linear
    autocovariance (zero-padded FFT, so not circular), Tukey-tapered to 0 at max_lag. Also returns
    C^(1/2), which maps the removed subspace (nuisance, mean) into the whitened domain."""
    from scipy.linalg import toeplitz

    n, T = X.shape
    acs = []
    for i in range(0, n, chunk):
        F = np.fft.rfft(X[i:i + chunk], n=2 * T, axis=1)
        acs.append(np.fft.irfft(np.abs(F) ** 2, axis=1)[:, :T] / T)
    med = np.median(np.concatenate(acs, axis=0), axis=0)
    lag = np.arange(T)
    taper = np.where(lag <= max_lag, 0.5 * (1 + np.cos(np.pi * lag / max_lag)), 0.0)
    w, V = np.linalg.eigh(toeplitz(med * taper))
    w = np.maximum(w, 1e-3 * w.max())
    return {"W": (V / np.sqrt(w)) @ V.T, "C_half": (V * np.sqrt(w)) @ V.T, "median_lag0": float(med[0]),
            "median_autocov": med}


def split_halves(idx: np.ndarray, mask3d, block: int, rng) -> np.ndarray:
    """Half membership per stacked row. Rows map to voxels (idx modulo V); voxels are grouped in
    block^3 spatial blocks and whole blocks go to one half at random. Without a mask: random voxels."""
    if mask3d is None:
        V = int(idx.max()) + 1
        lab = rng.permutation(V) < V // 2
        return lab[idx % V] if idx.max() >= V else lab[idx]
    coords = np.argwhere(mask3d)
    V = coords.shape[0]
    b = coords // int(block)
    bid = np.unique(b, axis=0, return_inverse=True)[1].ravel()
    lab_block = rng.random(bid.max() + 1) < 0.5
    return lab_block[bid][idx % V]


def split_deflated(Xw: np.ndarray, Qw: np.ndarray, half: np.ndarray, n_sur: int, q: float, k_max: int,
                   step: int, rng) -> dict:
    """Split-half deflated parallel analysis on whitened rows (see module docstring)."""
    res = {}
    for name, a, b in (("A_to_B", half, ~half), ("B_to_A", ~half, half)):
        _, U = spectrum(Xw[a], vectors=True)
        U = U[:, :k_max + step]
        B = Xw[b]
        stat = ((B @ U) ** 2).mean(axis=0)
        tot_b = float((B ** 2).sum())
        k, trace = 0, []
        while k < k_max:
            Uk = U[:, :k]
            Y = B - (B @ Uk) @ Uk.T if k else B
            tot = float((Y ** 2).sum())
            idx = list(range(k, min(k + step, k_max)))
            nulls = []
            for _ in range(int(n_sur)):
                Z = phase_randomize_rows(Y, rng)
                Z = Z - (Z @ Qw) @ Qw.T
                if k:
                    Z = Z - (Z @ Uk) @ Uk.T
                Z *= np.sqrt(tot / (Z ** 2).sum())
                nulls.append(((Z @ U[:, idx]) ** 2).mean(axis=0))
            thr = np.quantile(nulls, q, axis=0)
            passed = stat[idx] > thr
            trace += [{"k": i + 1, "stat": round(float(stat[i]), 5), "threshold": round(float(t), 5), "pass": bool(p)}
                      for i, t, p in zip(idx, thr, passed)]
            if not passed.all():
                k += int(np.argmin(passed))
                break
            k += len(idx)
        res[name] = {"k": int(k), "trace": trace, "row_variance_total": tot_b}
    return res


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
    tested, k_old = {}, None
    if Rk.get("old_deflated", False):
        k_old, tested = deflated_k(X, lam, V, Q, int(Rk["deflated_n_surrogates"]), q, rng, k_max, k_start=max(k_classic, 1))
    # gated: split-half deflated parallel analysis on noise-whitened rows
    wh = noise_whitener(X, int(Rk.get("whiten_max_lag", 40)))
    Xw = X @ wh["W"]
    Q0 = np.ones((T, 1)) / np.sqrt(T) if Q is None else np.column_stack([Q, np.ones(T) / np.sqrt(T)])
    Qw = orthonormal(wh["C_half"] @ Q0)
    Xw = Xw - (Xw @ Qw) @ Qw.T                                 # exact: rows already lie outside this subspace
    Xw /= np.sqrt((Xw ** 2).mean())
    _, idx_rows = stack(c0, c1)
    half = split_halves(idx_rows, mask3d, int(Rk.get("split_block_vox", 4)), rng)
    sp = split_deflated(Xw, Qw, half, int(Rk.get("split_n_surrogates", 10)), q, k_max, int(Rk.get("split_step", 3)), rng)
    kA, kB = sp["A_to_B"]["k"], sp["B_to_A"]["k"]
    k_temporal = int(round(0.5 * (kA + kB)))
    bulk = lam[k_temporal:t_eff]
    contamination = float(1 - bulk.mean() / wh["median_lag0"]) if bulk.size and wh["median_lag0"] > 0 else None
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
           "k_temporal": k_temporal, "k_split_halves": [kA, kB], "k_temporal_classic": k_classic, "k_max": k_max,
           "k_old_deflated": k_old,
           "split_trace": {k: v["trace"] for k, v in sp.items()},
           "contamination": None if contamination is None else round(contamination, 4),
           "contamination_validated_max": float(Rk.get("contamination_validated_max", 0.45)),
           "eigenvalues_whitened": spectrum(Xw),
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
                                            "residual_eigenvalue": col(res["residual_eigenvalues"]),
                                            "eigenvalue_whitened": col(res["eigenvalues_whitened"])})
    keep = {k: v for k, v in res.items() if not isinstance(v, np.ndarray) and k not in ("eigenvalues_subsampled", "residual_eigenvalues")}
    keep.update(run_id=fs.get("run_id"), source=fs.get("source"), nuisance=Q is not None,
                nuisance_rank=None if Q is None else int(Q.shape[1]), frontend_dir=str(fd))
    io.write_json(out_dir / "rank.json", keep)
    io.write_text(out_dir / "rank.md", render(keep))
    io.write_config_copy(cfg, out_dir)
    io.write_json(out_dir / "done.json", {"stage": "rank", **io.provenance(cfg, {"frontend_summary": str(fd / "summary.json")})})
    return keep


def render(r: dict) -> str:
    c, cmax = r.get("contamination"), r.get("contamination_validated_max", 0.45)
    inside = c is not None and c <= cmax
    L = [f"# Rank: {r.get('run_id')} ({r.get('source')} echoes, nuisance projection: "
         f"{'yes, rank ' + str(r['nuisance_rank']) if r.get('nuisance') else 'no'})", "",
         f"Stacked channels [c0 | c1]: {r['n_rows']} rows x {r['n_frames']} frames (effective {r['t_eff']} after projection "
         "and demeaning).", "",
         f"**Gated k = {r['k_temporal']}** (split-half deflated parallel analysis on noise-whitened channels; halves "
         f"{r['k_split_halves'][0]} and {r['k_split_halves'][1]}).",
         f"Contamination of the noise spectrum by shared signal: {c} (validated up to {cmax}: "
         + ("**inside** the validated range)." if inside else "**OUTSIDE** the validated range: k may be over-counted; see docs/decisions.md)."),
         "", "| Estimate | k | Note |", "|---|---|---|",
         f"| **Split-half deflated, whitened (gated)** | **{r['k_temporal']}** | simulation: within +/- 1 of 30 (white or coloured, "
         "narrowband or broadband); over-counts 10-37 % with 60 broadband components |",
         f"| Index by index | {r['k_temporal_classic']} | {r['n_surrogates']} surrogates; under-counts when rows share a spectrum |",
         f"| Thermal null (residual) | {r['k_thermal']} | above the top residual eigenvalue; the residual holds a few structured "
         "components, so this is not a thermal null (reported only) |",
         f"| Marchenko-Pastur edge | {r['k_mp_edge']} | edge {r['mp_edge']:.3f} for unit white noise; counts coloured noise |"]
    if r.get("k_old_deflated") is not None:
        L.append(f"| Unsplit deflated (old) | {r['k_old_deflated']} | over-counts with coloured noise |")
    mp = r.get("mapca") or {}
    if mp.get("available"):
        L += [f"| MAPCA AIC / KIC / MDL | {mp['aic']} / {mp['kic']} / {mp['mdl']} | tedana's estimator (mapca "
              f"{mp['mapca_version']}, normalize=True) on the stacked channel maps |"]
    elif mp:
        L += [f"| MAPCA | n/a | {mp.get('reason')} |"]
    L += ["", f"Fixed sweep for Phase 4: MAPCA estimates and k = {r['sweep_fixed']}."]
    return "\n".join(L) + "\n"
