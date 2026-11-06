"""Two-stage design, step 1 (user request 2026-11-06): diagnostic only, nothing new in the package.

On one run, the joint ICA in three variants, each at k = MAPCA AIC and k = 25 (same robust path: 30 FastICA
restarts, clustered centrotypes; only the input rows differ):
  (a) current: stacked [c0 | c1], thermal whitening (decompose.decompose, the Phase 4 seeds);
  (b) c0 only, thermal whitening;
  (c) c0 only, each voxel z-scored over time (tedana-like weighting).
Each is matched to tedana's components (the run's mbme_tedana output) by the Hungarian assignment on
|spatial correlation| (c0-channel maps vs tedana's component maps, over the voxels both masks share) and on
|temporal correlation| (time courses vs tedana's mixing matrix). Writes diag.md, per-variant matches, maps
(NIfTI) and montages.

Usage (repository root):
  PYTHONPATH=src python scripts/diag_c0_variants.py <front-end folder> <tedana folder> <out folder> \
      [--rank-dir <rank folder>] [--k 25 ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def find(folder: Path, patterns) -> Path | None:
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def load_tedana(td: Path, mask3d):
    """tedana 26 outputs (file names searched, not assumed): component maps, mixing matrix, metrics."""
    import nibabel as nib
    mp = find(td, ["*desc-ICA_stat-z_components.nii.gz", "*desc-ICA_components.nii.gz", "*ICA*components*.nii.gz",
                   "betas_OC.nii.gz"])
    mx = find(td, ["*desc-ICA_mixing.tsv", "*ICA_mixing.tsv", "ica_mixing.tsv"])
    mt = find(td, ["*desc-tedana_metrics.tsv", "*tedana_metrics.tsv", "ica_metrics.tsv"])
    if mp is None or mx is None:
        raise FileNotFoundError(f"tedana maps / mixing not found in {td} (looked for *ICA*components*.nii.gz, *ICA_mixing.tsv)")
    maps4d = np.asarray(nib.load(str(mp)).dataobj, np.float64)
    if maps4d.shape[:3] != mask3d.shape:
        raise ValueError(f"tedana maps grid {maps4d.shape[:3]} differs from the analysis grid {mask3d.shape}")
    maps = maps4d[mask3d]                                               # (V, n) on our grid and mask
    maps[~np.any(maps != 0, axis=1)] = np.nan                           # outside tedana's mask: missing, not zero
    import csv
    with open(mx) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    mixing = np.array(rows[1:], float)                                  # (T, n)
    meta = []
    if mt is not None:
        with open(mt) as f:
            rd = list(csv.DictReader(f, delimiter="\t"))
        meta = [{"classification": r.get("classification", ""), "tags": r.get("classification_tags", "")} for r in rd]
    return {"maps": maps, "mixing": mixing, "meta": meta, "files": {"maps": str(mp), "mixing": str(mx),
                                                                     "metrics": str(mt) if mt else None}}


def corr_abs(A, B):
    """|Pearson r| between columns of A (n, p) and B (n, q), rows with any NaN dropped."""
    ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    a, b = A[ok] - A[ok].mean(0), B[ok] - B[ok].mean(0)
    a /= np.maximum(np.linalg.norm(a, axis=0), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=0), 1e-12)
    return np.abs(a.T @ b), int(ok.sum())


def hungarian(C):
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-C)
    return r, c, C[r, c]


def ica_rows(X, k, seed_root, cfg, n_jobs):
    """The decompose.decompose ICA path on arbitrary rows: PCA to k, restarts, clustering, time courses."""
    from mejoint import decompose as dc
    D = cfg["decompose"]
    pc = dc.pca_reduce(X, int(k))
    seeds = [int(s.generate_state(1)[0]) for s in seed_root.spawn(int(D["n_restarts"]))]
    runs = dc.ica_restarts(pc["Y"], seeds, n_jobs=n_jobs, max_iter=int(D["max_iter"]), tol=float(D["tol"]),
                           fun=str(D.get("fun", "logcosh")))
    cl = dc.cluster_restarts(runs, int(k))
    n = X.shape[0]
    M = pc["V"] * np.sqrt(pc["lam"]) @ cl["W"].T / np.sqrt(n)
    M = (M - M.mean(0)) / M.std(0)
    expl = float(np.sum(pc["lam"]) / np.einsum("ij,ij->", X, X))
    stab = float(np.median(cl["iq"]))
    return M, expl, stab


def montage(maps, mask3d, path, title, n_show=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zs = np.flatnonzero(mask3d.sum((0, 1)) > 0.05 * mask3d.sum((0, 1)).max())
    sl = [int(z) for z in np.linspace(zs[0] + 2, zs[-1] - 2, 5)]
    K = min(n_show, maps.shape[1])
    fig, axs = plt.subplots(K, 5, figsize=(7.5, 1.5 * K))
    for j in range(K):
        v = np.full(mask3d.shape, np.nan)
        m = maps[:, j]
        v[mask3d] = (m - np.nanmean(m)) / max(np.nanstd(m), 1e-12)
        for i, z in enumerate(sl):
            ax = axs[j, i]
            ax.imshow(np.where(mask3d[:, :, z], 0, np.nan).T, origin="lower", cmap="Greys", vmin=-1, vmax=1)
            ax.imshow(np.where(np.abs(v[:, :, z]) > 2, v[:, :, z], np.nan).T, origin="lower", cmap="RdBu_r",
                      vmin=-6, vmax=6, interpolation="nearest")
            ax.axis("off")
        axs[j, 0].text(-0.1, 0.5, str(j + 1), transform=axs[j, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle(f"{title}: components by variance, |z| > 2 (array orientation, inferior to superior)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frontend_dir")
    ap.add_argument("tedana_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--rank-dir", help="rank folder with rank.json (MAPCA AIC)")
    ap.add_argument("--k", type=int, action="append", help="extra k (default 25 plus MAPCA AIC)")
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    import nibabel as nib

    from mejoint import decompose as dc
    from mejoint import io
    cfg = io.load_config(a.config)
    fe, out = Path(a.frontend_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mimg = nib.load(str(fe / "analysis_mask.nii.gz"))
    mask3d = np.asarray(mimg.dataobj) > 0
    c0, c1, R = np.load(fe / "c0.npy"), np.load(fe / "c1.npy"), np.load(fe / "R.npy")
    fs = json.loads((fe / "summary.json").read_text())
    m = re.match(r"sub-(\d+)", str(fs.get("run_id") or ""))
    subj = int(m.group(1)) if m else 0
    ks = list(a.k or [25])
    aic = None
    if a.rank_dir and (Path(a.rank_dir) / "rank.json").exists():
        aic = (json.loads((Path(a.rank_dir) / "rank.json").read_text()).get("mapca") or {}).get("aic")
        if aic:
            ks = [int(aic)] + [k for k in ks if k != int(aic)]
    td = load_tedana(Path(a.tedana_dir), mask3d)
    n_t = td["maps"].shape[1]
    acc = [i for i, mm in enumerate(td["meta"]) if mm["classification"].lower() == "accepted"]
    lb = [i for i, mm in enumerate(td["meta"]) if "likely bold" in mm["tags"].lower()]
    n_jobs = int(__import__("os").environ.get("LSB_DJOB_NUMPROC", "1"))
    ok0 = np.isfinite(c0).all(1)
    X0 = c0[ok0].astype(np.float64)
    X0 -= X0.mean(1, keepdims=True)
    Xz = dc.zscore_rows(X0)
    rows, detail = [], {}
    for k in ks:
        for var in ("a", "b", "c"):
            name = {"a": "(a) [c0 | c1], thermal", "b": "(b) c0, thermal", "c": "(c) c0, z-scored"}[var]
            print(f"== k {k} variant {name}", flush=True)
            if var == "a":
                ss = io.seed_sequence(cfg, subj, 0, 4, int(k))           # the Phase 4 seeds
                dec = dc.decompose(c0, c1, R, cfg, int(k), ss, n_jobs=n_jobs, mask3d=mask3d)
                M, expl = dec["time_courses"], float(dec["explained_variance"])
                stab = float(np.median(dec["iq"]))                     # cluster stability index Iq (as for b, c)
                rowsX = X0
            else:
                ss = io.seed_sequence(cfg, subj, 0, 40 if var == "b" else 41, int(k))
                rowsX = X0 if var == "b" else Xz
                M, expl, stab = ica_rows(rowsX, k, ss, cfg, n_jobs)
                stab = float("nan") if not np.isfinite(stab) else stab
            beta, _ = dc.glm_maps(rowsX, M)                              # c0-channel maps (z-scored rows for c)
            maps = np.full((c0.shape[0], M.shape[1]), np.nan)
            maps[ok0] = beta
            order = np.argsort(-np.sum(beta ** 2, 0))                     # by map energy
            maps, M = maps[:, order], M[:, order]
            Cs, n_vox = corr_abs(maps, td["maps"])
            T = min(M.shape[0], td["mixing"].shape[0])
            Ct, _ = corr_abs(M[:T], td["mixing"][:T]) if M.shape[0] == td["mixing"].shape[0] else (np.full((k, n_t), np.nan), 0)
            r_s, c_s, v_s = hungarian(Cs)
            r_t, c_t, v_t = hungarian(np.nan_to_num(Ct))
            best_s = Cs.max(0)                                          # each tedana component's best match
            tag = f"k{k}_{var}"
            io.save_map(maps.astype(np.float32), mask3d, mimg, out / f"maps_{tag}.nii.gz")
            io.write_tsv(out / f"time_courses_{tag}.tsv", {f"c{j + 1}": list(M[:, j]) for j in range(M.shape[1])})
            montage(maps, mask3d, out / f"montage_{tag}.png", f"k = {k}, {name}")
            match = [{"component": int(i) + 1, "tedana": int(j), "spatial_r": round(float(v), 3),
                      "tedana_class": td["meta"][j]["classification"] if td["meta"] else "",
                      "tedana_tags": td["meta"][j]["tags"] if td["meta"] else ""}
                     for i, j, v in sorted(zip(r_s, c_s, v_s), key=lambda x: -x[2])]
            io.write_tsv(out / f"match_{tag}.tsv", {c: [r[c] for r in match] for c in match[0]})
            rec = lambda idx: (sum(best_s[i] >= 0.7 for i in idx), len(idx))
            rows.append({"k": k, "variant": name, "explained": expl, "stability": stab, "n_vox": n_vox,
                         "s_med": float(np.median(v_s)), "s_05": int((v_s >= 0.5).sum()), "s_07": int((v_s >= 0.7).sum()),
                         "t_med": float(np.median(v_t)) if np.isfinite(Ct).all() else np.nan,
                         "t_05": int((v_t >= 0.5).sum()), "t_07": int((v_t >= 0.7).sum()), "pairs": int(len(v_s)),
                         "acc": rec(acc), "lb": rec(lb)})
            detail[tag] = match
            print(f"   spatial matched |r| median {rows[-1]['s_med']:.2f}; temporal {rows[-1]['t_med']:.2f}", flush=True)
    L = [f"# Step 1 diagnostic: {fs.get('run_id')} ({fs.get('source')} echoes, nuisance projection: "
         f"{'yes' if (fe / 'nuisance_Q.npy').exists() else 'no'})", "",
         f"tedana: {n_t} components ({len(acc)} accepted, {len(lb)} tagged Likely BOLD); files: {td['files']}.",
         f"Spatial matching over {rows[0]['n_vox'] if rows else 0} voxels in both masks (tedana maps vs c0-channel maps); "
         "temporal: time courses vs tedana's mixing matrix. Hungarian assignment on |r| (min(k, n) pairs).", "",
         "| k | Variant | Variance in k dims | Stability Iq (median) | Spatial: median matched r, pairs >= 0.5 / >= 0.7 | Temporal: median, >= 0.5 / >= 0.7 | "
         "tedana accepted reproduced (best spatial r >= 0.7) | Likely BOLD reproduced |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['k']}{' (AIC)' if aic and r['k'] == int(aic) else ''} | {r['variant']} | {100 * r['explained']:.1f} % | "
                 f"{r['stability']:.3f} | "
                 f"{r['s_med']:.2f}, {r['s_05']} / {r['s_07']} of {r['pairs']} | {r['t_med']:.2f}, {r['t_05']} / {r['t_07']} | "
                 f"{r['acc'][0]} / {r['acc'][1]} | {r['lb'][0]} / {r['lb'][1]} |")
    L += ["", "Montages: montage_k<k>_<variant>.png (components ordered by map energy). Maps: maps_*.nii.gz (c0-channel "
              "betas; z-scored rows for (c)). Per-pair matches: match_*.tsv."]
    io.write_text(out / "diag.md", "\n".join(L) + "\n")
    io.write_json(out / "diag.json", {"rows": rows, "tedana_files": td["files"]})
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.exit(main())
