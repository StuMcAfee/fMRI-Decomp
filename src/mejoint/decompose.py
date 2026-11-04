"""Phase 4, option 1 (CLAUDE.md section 7): joint spatial ICA on the stacked whitened channels [c0 | c1].

Steps:
  1. Stack the c0 and c1 rows of every voxel (c1 rows undefined in 1-echo voxels are excluded, never
     zero-filled); demean over time.
  2. PCA from the T x T Gram matrix in float64; keep k components (k from the fixed sweep, Phase 3).
  3. Whiten once: Y = U_k sqrt(n) (rows = stacked voxel rows, k unit-variance uncorrelated columns).
  4. FastICA (scikit-learn, whiten=False: verified in 1.9.1 that n_components is then ignored, so the
     data are reduced to k dimensions here) with >= 30 restarts from different seeds, in parallel.
  5. Robust components (robustica / icasso-style): the spatial maps of all restarts are clustered on
     |correlation|; because Y has identity covariance, the correlation between two maps is the dot
     product of their unmixing vectors. Each cluster gives its centrotype (the member most similar to
     the others), its stability index Iq and the number of restarts it contains.
  6. Time courses (unit SD) from the reduced data; channel maps (m0, m1) are joint OLS betas of c0 and
     c1 on all k time courses (GLM); back-substitution gives s = m1 / R11 (dS0/S0) and
     q = (m0 - R01 s) / R00 (dR2*; TE in ms, reported in s^-1).
  7. Sign convention: the stacked map [m0 | m1] has positive skewness.
Stability across seeds: every restart is matched to the final components (Hungarian on |corr|); the
median matched |corr| per component is reported.
"""
from __future__ import annotations

import os
import warnings

import numpy as np


def stack_rows(c0: np.ndarray, c1: np.ndarray) -> dict:
    """Stacked rows with all frames finite, demeaned (float64), plus each row's voxel and channel."""
    V = c0.shape[0]
    A = np.concatenate([c0, c1], axis=0)
    ok = np.isfinite(A).all(axis=1)
    X = A[ok].astype(np.float64)
    X -= X.mean(axis=1, keepdims=True)
    vox = np.r_[np.arange(V), np.arange(V)][ok]
    ch = np.r_[np.zeros(V, int), np.ones(V, int)][ok]
    return {"X": X, "voxel": vox, "channel": ch, "n_voxels": V}


def pca_reduce(X: np.ndarray, k: int, chunk: int = 20000) -> dict:
    """Top-k temporal PCA from the Gram matrix (float64). Returns V (T, k), eigenvalues lam (k,) of
    X^T X, Y = X V / sqrt(lam) * sqrt(n) (whitened rows), explained variance fraction."""
    n, T = X.shape
    G = np.zeros((T, T))
    for i in range(0, n, chunk):
        x = X[i:i + chunk]
        G += x.T @ x
    lam, Vt = np.linalg.eigh(G)
    lam, Vt = lam[::-1], Vt[:, ::-1]
    if k > int((lam > 1e-12 * lam[0]).sum()):
        raise ValueError(f"k = {k} exceeds the data rank")
    Vk, lk = Vt[:, :k], lam[:k]
    Y = (X @ Vk) / np.sqrt(lk) * np.sqrt(n)
    return {"V": Vk, "lam": lk, "Y": Y, "explained": float(lk.sum() / lam.sum())}


def _one_run(Y, seed, max_iter, tol, fun):
    from sklearn.decomposition import FastICA
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(whiten=False, fun=fun, max_iter=max_iter, tol=tol, random_state=int(seed))
        ica.fit(Y)
    W = ica.components_
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    return W, int(ica.n_iter_), bool(ica.n_iter_ < max_iter)


def ica_restarts(Y: np.ndarray, seeds, n_jobs: int | None = None, max_iter: int = 1000, tol: float = 1e-6,
                 fun: str = "logcosh") -> list[dict]:
    """FastICA from every seed on the whitened data (unmixing matrices, rows unit norm)."""
    from joblib import Parallel, delayed

    if n_jobs is None:
        n_jobs = int(os.environ.get("LSB_DJOB_NUMPROC", "1"))
    out = Parallel(n_jobs=n_jobs)(delayed(_one_run)(Y, s, max_iter, tol, fun) for s in seeds)
    return [{"W": W, "n_iter": it, "converged": cv, "seed": int(s)} for (W, it, cv), s in zip(out, seeds)]


def cluster_restarts(runs: list[dict], k: int) -> dict:
    """Cluster the maps of all restarts (|corr| = |w_a . w_b|) into k clusters (average linkage).
    Per cluster: centrotype, stability index Iq (mean intra - mean extra similarity), restarts present."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    Wall = np.concatenate([r["W"] for r in runs], axis=0)               # (R k, k)
    run_id = np.repeat(np.arange(len(runs)), k)
    S = np.abs(Wall @ Wall.T)
    np.fill_diagonal(S, 1.0)
    D = np.clip(1.0 - S, 0, None)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, t=k, criterion="maxclust") - 1
    cents, iq, n_runs, size = [], [], [], []
    for c in range(lab.max() + 1):
        mem = np.flatnonzero(lab == c)
        sub = S[np.ix_(mem, mem)]
        j = mem[np.argmax(sub.mean(1))]
        cents.append(Wall[j])
        other = np.flatnonzero(lab != c)
        extra = S[np.ix_(mem, other)].mean() if other.size else 0.0
        iq.append(float(sub.mean() - extra))
        n_runs.append(int(np.unique(run_id[mem]).size))
        size.append(int(mem.size))
    order = np.argsort(iq)[::-1]
    return {"W": np.array(cents)[order], "iq": np.array(iq)[order], "n_runs": np.array(n_runs)[order],
            "size": np.array(size)[order], "n_clusters": int(lab.max() + 1)}


def hungarian_abs_corr(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match columns of B to columns of A by |corr| (Hungarian). Returns (index into B for each A
    column, matched |corr|)."""
    from scipy.optimize import linear_sum_assignment

    a = (A - A.mean(0)) / np.maximum(A.std(0), 1e-12)
    b = (B - B.mean(0)) / np.maximum(B.std(0), 1e-12)
    C = np.abs(a.T @ b) / A.shape[0]
    r, c = linear_sum_assignment(-C)
    idx = np.full(A.shape[1], -1)
    idx[r] = c
    return idx, C[r, c][np.argsort(r)]


def glm_maps(X: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint OLS betas of rows X (n, T) on time courses M (T, k), and the beta noise variance for unit
    white noise, diag((M^T M)^-1) (the generalisation of 1 / ||t||^2)."""
    G = M.T @ M
    Gi = np.linalg.inv(G)
    return X @ M @ Gi, np.diag(Gi).copy()


def crossfit_maps(X, S, M, voxel, half_vox) -> tuple[np.ndarray, np.ndarray]:
    """Betas of each row on time courses estimated from the OTHER half of the voxels.
    S: (n, k) spatial maps of the stacked rows; M: (T, k) full-data time courses (for sign alignment).
    Returns betas (n, k), their noise variance (n, k) for unit white noise, and the two half time
    courses: Ms[0] is estimated from half A (used for half-B voxels), Ms[1] from half B."""
    n, k = S.shape
    B = np.full((n, k), np.nan)
    VB = np.full((n, k), np.nan)
    in_a = half_vox[voxel]
    Ms = []
    for est, use in ((in_a, ~in_a), (~in_a, in_a)):
        Se, Xe = S[est], X[est]
        Mh = np.linalg.solve(Se.T @ Se, Se.T @ Xe).T                    # (T, k) time courses from this half
        Mh = (Mh - Mh.mean(0)) / Mh.std(0)
        sg = np.sign(np.sum(Mh * M, axis=0))
        Mh = Mh * np.where(sg == 0, 1, sg)
        b, vb = glm_maps(X[use], Mh)
        B[use], VB[use] = b, vb[None, :]
        Ms.append(Mh)
    return B, VB, Ms


def decompose(c0, c1, R, cfg: dict, k: int, seed_root, n_jobs: int | None = None, mask3d=None) -> dict:
    """Option 1 on one set of channels. R: (V, 2, 2) per-voxel QR factors. seed_root: SeedSequence.
    mask3d (optional): voxel halves for the cross-fitted maps follow the readout-slab split of Phase 3
    (alias partners and most neighbours together); without it, random voxel halves."""
    D = cfg["decompose"]
    st = stack_rows(c0, c1)
    X = st["X"]
    n, T = X.shape
    pc = pca_reduce(X, int(k))
    seeds = [int(s.generate_state(1)[0]) for s in seed_root.spawn(int(D["n_restarts"]))]
    runs = ica_restarts(pc["Y"], seeds, n_jobs=n_jobs, max_iter=int(D["max_iter"]), tol=float(D["tol"]),
                        fun=str(D.get("fun", "logcosh")))
    cl = cluster_restarts(runs, int(k))
    Wf = cl["W"]
    # time courses (T, k): X ~ S M^T with S = Y W^T, M = V lam^(1/2) W^T / sqrt(n); unit SD
    M = pc["V"] * np.sqrt(pc["lam"]) @ Wf.T / np.sqrt(n)
    M = (M - M.mean(0)) / M.std(0)
    # channel maps per voxel (NaN where the channel is undefined)
    V = st["n_voxels"]
    beta, vb = glm_maps(X, M)
    m0 = np.full((V, M.shape[1]), np.nan)
    m1 = np.full((V, M.shape[1]), np.nan)
    m0[st["voxel"][st["channel"] == 0]] = beta[st["channel"] == 0]
    m1[st["voxel"][st["channel"] == 1]] = beta[st["channel"] == 1]
    # sign convention: positive skewness of the stacked map
    from scipy.stats import skew
    sk = skew(np.concatenate([np.nan_to_num(m0), np.nan_to_num(m1)], axis=0), axis=0)
    sgn = np.where(sk < 0, -1.0, 1.0)
    M, m0, m1, Wf = M * sgn, m0 * sgn, m1 * sgn, Wf * sgn[:, None]
    s, q = backsub(m0, m1, R)
    # cross-fitted maps (eta, bands): time course from one voxel half, betas for the other
    from .rank import acquisition_axes, split_halves
    rng_h = np.random.default_rng(seed_root.spawn(1)[0])
    if mask3d is not None:
        half_vox = split_halves(np.arange(V), mask3d, int(cfg["rank"].get("split_block_vox", 4)), rng_h,
                                slab_axis=acquisition_axes(cfg)["readout"])
    else:
        half_vox = rng_h.permutation(V) < V // 2
    S_rows = pc["Y"] @ Wf.T                                            # (n, k) spatial maps of the stacked rows
    Bcf, VBcf, Ms_cf = crossfit_maps(X, S_rows, M, st["voxel"], half_vox)
    m0_cf = np.full((V, M.shape[1]), np.nan)
    m1_cf = np.full((V, M.shape[1]), np.nan)
    vb_cf = np.full((V, M.shape[1]), np.nan)
    r0, r1 = st["channel"] == 0, st["channel"] == 1
    m0_cf[st["voxel"][r0]], m1_cf[st["voxel"][r1]] = Bcf[r0], Bcf[r1]
    vb_cf[st["voxel"][r0]] = VBcf[r0]                                   # same time course for both channels of a voxel
    # stability across restarts: each restart's maps matched to the final maps (|corr| = |w . w'|)
    from scipy.optimize import linear_sum_assignment
    stab = []
    for r in runs:
        C = np.abs(Wf @ r["W"].T)
        a, b = linear_sum_assignment(-C)
        stab.append(C[a, b][np.argsort(a)])
    stab = np.array(stab)                                              # (R, k)
    return {"k": int(k), "time_courses": M, "m0": m0, "m1": m1, "s": s, "q_per_s": q,
            "m0_cf": m0_cf, "m1_cf": m1_cf, "vb_cf": vb_cf,
            "crossfit": {"half_vox": half_vox, "M_from_A": Ms_cf[0], "M_from_B": Ms_cf[1]},
            "beta_noise_var": vb, "iq": cl["iq"], "n_runs_in_cluster": cl["n_runs"], "cluster_size": cl["size"],
            "stability_median": np.median(stab, axis=0), "stability_p10": np.percentile(stab, 10, axis=0),
            "n_restarts": len(runs), "n_converged": int(sum(r["converged"] for r in runs)),
            "seeds": seeds, "explained_variance": pc["explained"], "n_rows": int(n), "n_frames": int(T),
            "unmixing": Wf}


def backsub(m0: np.ndarray, m1: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CLAUDE.md 5.3 property 4: s = m1 / R11 (dS0/S0), q = (m0 - R01 s) / R00 (dR2*), q returned in
    s^-1 (the design uses TE in ms). NaN where c1 is undefined."""
    R00, R01, R11 = R[:, 0, 0][:, None], R[:, 0, 1][:, None], R[:, 1, 1][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = m1 / R11
        q = (m0 - R01 * s) / R00 * 1000.0
    return s, q


def tedana_style_oc(S: np.ndarray, te_ms, t2s_ms: np.ndarray) -> np.ndarray:
    """Optimally combined data with tedana's weights w_e = TE_e exp(-TE_e / T2*) (normalised), linear
    signal; returned demeaned (OC deviation, V x T). Used only for the Phase 4 equivalence test."""
    te = np.asarray(te_ms, float)
    w = te[None, :] * np.exp(-te[None, :] / t2s_ms[:, None])
    w = w / w.sum(1, keepdims=True)
    oc = np.einsum("ve,vet->vt", w, S.astype(np.float64))
    return oc - oc.mean(1, keepdims=True)


def zscore_rows(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(1, keepdims=True)
    return X / np.maximum(X.std(1, keepdims=True), 1e-12)


def single_channel_ica(X: np.ndarray, k: int, seeds, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    """Spatial ICA of one channel (rows = voxels) with fixed seeds, first restart only (deterministic):
    returns time courses (T, k). For the tedana-equivalence test."""
    pc = pca_reduce(X, k)
    W, _, _ = _one_run(pc["Y"], seeds[0], max_iter, tol, "logcosh")
    M = pc["V"] * np.sqrt(pc["lam"]) @ W.T / np.sqrt(X.shape[0])
    return (M - M.mean(0)) / M.std(0)


# ============================================================================ real-data runner
def run_dir(frontend_dir, cfg: dict, out_dir, k: int, n_jobs: int | None = None) -> dict:
    """Option 1 at one k on a front-end output folder (channel cache c0/c1.npy, R.npy, analysis mask,
    nuisance_Q.npy if projected, frames.tsv). Writes maps (s, q in s^-1, m0, m1) as 4D NIfTI,
    time_courses.tsv, components.tsv / .json, decompose.md and done.json."""
    import json
    import re
    from pathlib import Path

    import nibabel as nib

    from . import io, metrics

    fd, out_dir = Path(frontend_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c0, c1 = np.load(fd / "c0.npy"), np.load(fd / "c1.npy")
    R = np.load(fd / "R.npy")
    mimg = nib.load(str(fd / "analysis_mask.nii.gz"))
    mask3d = np.asarray(mimg.dataobj) > 0
    Q = np.load(fd / "nuisance_Q.npy") if (fd / "nuisance_Q.npy").exists() else None
    fs = json.loads((fd / "summary.json").read_text())
    m = re.match(r"sub-(\d+)", str(fs.get("run_id") or ""))
    subj = int(m.group(1)) if m else 0
    ss = io.seed_sequence(cfg, subj, 0, 4, int(k))                  # stage 4 = decomposition
    dec = decompose(c0, c1, R, cfg, int(k), ss, n_jobs=n_jobs, mask3d=mask3d)
    frames = {}
    fr_path = fd / "frames.tsv"
    if fr_path.exists():
        fr = np.genfromtxt(fr_path, delimiter="\t", names=True)
        for key in ("fd_mm", "fd_filtered_mm", "drift_mm"):
            if key in (fr.dtype.names or ()):
                frames[key] = fr[key]
    rng = np.random.default_rng(io.seed_sequence(cfg, subj, 0, 5, int(k)))   # stage 5 = metrics
    me = metrics.measure(dec, c0, c1, R, cfg, rng, Q=Q, frames=frames)
    for name, arr in (("s", dec["s"]), ("q_per_s", dec["q_per_s"]), ("m0", dec["m0"]), ("m1", dec["m1"])):
        io.save_map(arr, mask3d, mimg, out_dir / f"{name}.nii.gz")
    T = dec["time_courses"].shape[0]
    io.write_tsv(out_dir / "time_courses.tsv", {f"c{j + 1}": list(dec["time_courses"][:, j]) for j in range(dec["k"])})
    rows = me["components"]
    cols = [c for c in rows[0].keys()]
    fmt = lambda v: "[" + ", ".join(f"{x:.4g}" for x in v) + "]" if isinstance(v, list) else v
    io.write_tsv(out_dir / "components.tsv", {c: [fmt(r.get(c)) for r in rows] for c in cols})
    summ = {"run_id": fs.get("run_id"), "source": fs.get("source"), "k": dec["k"], "n_rows": dec["n_rows"],
            "n_frames": dec["n_frames"], "explained_variance": round(dec["explained_variance"], 4),
            "n_restarts": dec["n_restarts"], "n_converged": dec["n_converged"], "null": me["null"],
            "nuisance_projected": Q is not None, "components": rows,
            "label_counts": {lab: sum(r["label"] == lab for r in rows)
                             for lab in ("R2*-dominant", "S0-dominant", "mixed", "not above null", "undetermined")},
            "stability_median_of_components": round(float(np.median(dec["stability_median"])), 4)}
    io.write_json(out_dir / "components.json", summ)
    io.write_text(out_dir / "decompose.md", render(summ))
    io.write_config_copy(cfg, out_dir)
    io.write_json(out_dir / "done.json", {"stage": f"decompose_k{dec['k']}", "seeds_restarts": dec["seeds"],
                                          **io.provenance(cfg, {"frontend_summary": str(fd / "summary.json")})})
    return summ


def render(s: dict) -> str:
    L = [f"# Joint ICA (option 1): {s.get('run_id')} ({s.get('source')} echoes), k = {s['k']}", "",
         f"{s['n_rows']} stacked rows x {s['n_frames']} frames; {s['explained_variance'] * 100:.1f} % of the variance in the "
         f"{s['k']} PCA dimensions; {s['n_restarts']} FastICA restarts ({s['n_converged']} converged); nuisance projection: "
         f"{'yes' if s['nuisance_projected'] else 'no'}.", "",
         f"Energy null (top-{s['k']} directions of phase-randomized surrogates, {s['null']['quantile']:.2f} quantile): "
         f"{s['null']['threshold']:.1f}. Labels are measurements (CLAUDE.md 7.1), not noise labels: "
         + ", ".join(f"{k}: {v}" for k, v in s["label_counts"].items() if v) + ".",
         f"Median component stability across restarts (matched |r|): {s['stability_median_of_components']}.", "",
         "| # | Label | eta | R2* band | S0 band | Energy / null | Iq | Stability | ||s|| [95 % CI] | ||q|| s^-1 [95 % CI] |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    thr = s["null"]["threshold"]
    for r in s["components"]:
        L.append(f"| {r['component']} | {r['label']} | {r['eta']} | {r['band_R2']} | {r['band_S0']} | "
                 f"{r['energy_single'] / thr:.2f} | {r['iq']} | {r['stability_median_abs_r']} | "
                 f"{r['norm_s']:.3g} [{r['norm_s_ci'][0]:.3g}, {r['norm_s_ci'][1]:.3g}] | "
                 f"{r['norm_q_per_s']:.3g} [{r['norm_q_ci'][0]:.3g}, {r['norm_q_ci'][1]:.3g}] |")
    return "\n".join(L) + "\n"


def run_sweep(frontend_dir, cfg: dict, out_root, rank_dir=None, ks=None, n_jobs: int | None = None) -> list[dict]:
    """The fixed k sweep (Phase 3 outcome): MAPCA AIC / KIC / MDL from rank.json (if given) plus
    decompose.k_sweep_fixed. One folder per k and a sweep summary (sweep.md / sweep.json)."""
    import json
    from pathlib import Path

    from . import io

    out_root = Path(out_root)
    if ks is None:
        ks = [int(k) for k in cfg["decompose"]["k_sweep_fixed"]]
        if rank_dir is not None and (Path(rank_dir) / "rank.json").exists():
            mp = (json.loads((Path(rank_dir) / "rank.json").read_text()).get("mapca") or {})
            if mp.get("available"):
                ks = [int(mp["mdl"]), int(mp["kic"]), int(mp["aic"])] + ks
    ks = sorted(set(ks))
    rows = []
    for k in ks:
        s = run_dir(frontend_dir, cfg, out_root / f"k{k}", k, n_jobs=n_jobs)
        rows.append({"k": k, **s["label_counts"], "stability": s["stability_median_of_components"],
                     "explained": s["explained_variance"]})
    io.write_json(out_root / "sweep.json", {"rows": rows, **io.provenance(cfg, {})})
    L = ["# Joint ICA across the fixed k sweep (no gated k; Phase 3 outcome)", "",
         "| k | R2*-dominant | S0-dominant | mixed | not above null | Median stability | Variance in k dims |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['k']} | {r['R2*-dominant']} | {r['S0-dominant']} | {r['mixed']} | {r['not above null']} | {r['stability']} | "
          f"{r['explained'] * 100:.1f} % |" for r in rows]
    L += ["", "Counts depend on k and on the estimator (CLAUDE.md section 11); they are not the primary outcome."]
    io.write_text(out_root / "sweep.md", "\n".join(L) + "\n")
    return rows
