"""Phase 5 injection calibration (user decision 2026-11-06).

Known sources are added to the RAW echoes of a real run, before any processing; the run then goes through
the unchanged pipeline (mbme_tedana volume pairing, motion correction and tedana; the mejoint front end;
joint ICA at fixed k), and the sources are looked up in the output. This answers two questions:

1. Calibration: where do known dR2*-only, dS0-only and mixed sources land (eta, label, echo profile,
   curvature) when they pass through the real noise and the real pipeline? Those positions are the
   real-data bands for calling components likely neuronal (dR2*-like) or likely not.
2. The echo-1 effect: the real components show eta ~ 0.4, dS0 and dR2* of the same sign with a ratio near
   32 ms, and curved echo profiles (te-check). If an injected pure dR2* source comes out the same way, the
   pipeline or the noise creates the effect; if it comes out clean, the effect belongs to the signal.

Signal model (CLAUDE.md 5.1, exact in the log): source i has a unit-SD time course u_i(t), a spatial weight
w_i(v) (cos^2 taper, 1 at the centre) and amplitudes a_i (dS0/S0) and b_i (dR2*, s^-1). Echo e of voxel v
becomes S_e + mu_e [exp(sum_i w_i (a_i - TE_e b_i) u_i) - 1], with mu_e the temporal mean of the raw echo:
the change is added to the measured signal, so the measured noise is untouched.

Placement is data-only (no anatomy; user decision 2026-11-06): from the run's own analysis mask and
data-only tissue classes (an earlier front end and `mejoint tissue` on the same run). Regions:
  cortical: centre radius to radius + cortical_band_mm from the mask edge, upper two thirds of the brain;
  deep:     centre at least radius + deep_min_mm from the edge, upper two thirds;
  inferior: inferior third and posterior half (an approximate posterior-fossa level), at least radius from
            the edge.
Each sphere must lie mostly in the tissue or mixed classes (not CSF, short-T2* or dropout voxels).

What injection cannot test: multiband slice leakage (the injected signal is added after reconstruction,
so it never leaks), coil combination and anything else upstream of the magnitude images. Sources are fixed
in scanner space; after motion correction they move by minus the head motion (small in these runs).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

TYPES = ("r2", "s0", "mixed")
REGIONS = ("cortical", "deep", "inferior")


# ============================================================================ sources
def amplitudes(kind: str, level: dict, mixed_ratio_ms: float) -> tuple[float, float]:
    """(a, b): temporal SD at the centre of dS0/S0 (fraction) and dR2* (s^-1)."""
    q, s = float(level["q_sd_per_s"]), float(level["s_sd"])
    if kind == "r2":
        return 0.0, q
    if kind == "s0":
        return s, 0.0
    if kind == "mixed":
        return float(mixed_ratio_ms) / 1000.0 * q, q
    raise ValueError(f"unknown source type {kind}")


def time_courses(T: int, tr_s: float, band_hz, n: int, rng) -> np.ndarray:
    """n band-limited Gaussian time courses (T, n): zero mean, unit SD, mutually orthogonal."""
    lo, hi = (float(x) for x in band_hz)
    X = rng.standard_normal((T, n))
    F = np.fft.rfft(X, axis=0)
    f = np.fft.rfftfreq(T, float(tr_s))
    F[(f < lo) | (f > hi)] = 0.0
    X = np.fft.irfft(F, n=T, axis=0)
    X -= X.mean(0)
    Qx, _ = np.linalg.qr(X)                                            # orthonormal; still in band, zero mean
    return Qx / Qx.std(0)


def ball(radius_mm: float, voxel_mm) -> np.ndarray:
    r = [int(np.ceil(radius_mm / float(v))) for v in voxel_mm]
    g = np.mgrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1].astype(float)
    d = np.sqrt(sum((g[i] * float(voxel_mm[i])) ** 2 for i in range(3)))
    return (d <= radius_mm).astype(float)


def weight_maps(shape, centers, radius_mm: float, voxel_mm, mask3d=None) -> np.ndarray:
    """(X, Y, Z, n) weights: cos^2(pi d / (2 R)) inside radius R of each centre, 0 outside (and outside the mask)."""
    idx = np.indices(shape).astype(float)
    out = np.zeros(tuple(shape) + (len(centers),), np.float32)
    for i, c in enumerate(centers):
        d = np.sqrt(sum(((idx[a] - c[a]) * float(voxel_mm[a])) ** 2 for a in range(3)))
        w = np.where(d < radius_mm, np.cos(np.pi * d / (2.0 * radius_mm)) ** 2, 0.0)
        if mask3d is not None:
            w = w * mask3d
        out[..., i] = w
    return out


def region_masks(mask3d, affine, voxel_mm, cfg: dict) -> dict:
    """Candidate centres per region (boolean 3D arrays), from the mask alone."""
    import nibabel as nib
    from scipy import ndimage

    from .frontend import inferior_third
    I = cfg["inject"]
    R = float(I["radius_mm"])
    dist = ndimage.distance_transform_edt(mask3d, sampling=[float(v) for v in voxel_mm])   # mm to the nearest non-brain voxel
    inf = inferior_third(mask3d, affine)
    codes = nib.aff2axcodes(affine)
    ap = next(i for i, c in enumerate(codes) if c in ("A", "P"))
    idx = np.indices(mask3d.shape)[ap]
    lo, hi = idx[mask3d].min(), idx[mask3d].max()
    mid = (lo + hi) / 2.0
    posterior = (idx <= mid) if codes[ap] == "A" else (idx >= mid)
    upper = mask3d & ~inf
    return {"cortical": upper & (dist >= R) & (dist <= R + float(I["cortical_band_mm"])),
            "deep": upper & (dist >= R + float(I["deep_min_mm"])),
            "inferior": inf & posterior & (dist >= R)}


def place_sources(mask3d, affine, voxel_mm, cfg: dict, rng, classes3d=None) -> list[dict]:
    """One source of every type in every region; centres random among the candidates, at least
    min_separation_mm apart, with the sphere mostly in tissue or mixed classes (relaxed in 0.1 steps)."""
    from scipy import ndimage
    I = cfg["inject"]
    types = list(I.get("types", TYPES))
    regions = list(I.get("regions", REGIONS))
    R = float(I["radius_mm"])
    vox = np.asarray(voxel_mm, float)
    k = ball(R, vox)
    nb = ndimage.convolve(mask3d.astype(float), k, mode="constant")
    good = mask3d if classes3d is None else (mask3d & np.isin(classes3d, (1, 2)))
    tf = ndimage.convolve(good.astype(float), k, mode="constant") / np.maximum(nb, 1.0)
    cand = region_masks(mask3d, affine, vox, cfg)
    sep = float(I["min_separation_mm"])
    placed: list[dict] = []
    for region in regions:
        order = list(rng.permutation(types))
        for kind in order:
            need = float(I["min_tissue_fraction"])
            c = None
            while c is None and need >= 0.4 - 1e-9:
                pts = np.argwhere(cand[region] & (tf >= need))
                pts = pts[rng.permutation(len(pts))]
                for p in pts:
                    if all(np.sqrt((((p - np.asarray(o["center"])) * vox) ** 2).sum()) >= sep for o in placed):
                        c = p
                        break
                if c is None:
                    need -= 0.1
            if c is None:
                raise RuntimeError(f"no place for a {kind} source in region {region} (mask too small for "
                                   f"radius {R} mm and separation {sep} mm)")
            placed.append({"id": f"{region}_{kind}", "region": region, "type": kind,
                           "center": [int(x) for x in c], "tissue_fraction": round(float(tf[tuple(c)]), 3),
                           "tissue_fraction_required": round(need, 2)})
    return placed


# ============================================================================ injection
def log_change(w: np.ndarray, U: np.ndarray, a, b, te_ms) -> np.ndarray:
    """Injected log-signal change (Nv, E, T): sum_i w_i (a_i - TE_e b_i) u_i(t)."""
    A = (w * np.asarray(a, float)[None, :]) @ U.T                    # (Nv, T)
    B = (w * np.asarray(b, float)[None, :]) @ U.T
    te = np.asarray(te_ms, float) / 1000.0
    return A[:, None, :] - te[None, :, None] * B[:, None, :]


def inject_masked(S: np.ndarray, w: np.ndarray, U: np.ndarray, a, b, te_ms) -> np.ndarray:
    """Masked magnitudes S (Nv, E, T) with the sources added (returns a new float64 array)."""
    S = np.asarray(S, np.float64)
    mu = S.mean(axis=2, keepdims=True)
    return S + mu * np.expm1(log_change(w, U, a, b, te_ms))


def inject_file(src, dst, e: int, frames_raw: np.ndarray, W3: np.ndarray, U: np.ndarray, a, b, te_ms) -> dict:
    """Echo e of one run: add the sources to the raw volumes that make up the analysis frames
    (frames_raw[f] = raw volume of frame f for this echo; from volumes_used.tsv). Integer files without
    scaling keep their type (rounded; voxels outside the spheres stay bit-identical); others become float32."""
    import os

    import nibabel as nib
    img = nib.load(str(src))
    hdr = img.header.copy()
    dt = np.dtype(hdr.get_data_dtype())
    slope, inter = hdr.get_slope_inter()
    keep_int = np.issubdtype(dt, np.integer) and (slope is None or slope == 1) and (inter is None or inter == 0)
    data = np.asarray(img.dataobj, dtype=np.float32)
    frames_raw = np.asarray(frames_raw, int)
    if len(np.unique(frames_raw)) != frames_raw.size:
        raise ValueError(f"{src}: a raw volume is used by more than one frame; injection would be ambiguous")
    sv = np.nonzero(W3.max(axis=3) > 0)
    w = W3[sv].astype(np.float64)                                      # (Nv, n)
    D = data[sv].astype(np.float64)                                    # (Nv, T_raw)
    mu = D[:, frames_raw].mean(axis=1)
    y = log_change(w, U, a, b, [te_ms[e]])[:, 0, :]                    # (Nv, T)
    D[:, frames_raw] += mu[:, None] * np.expm1(y)
    if keep_int:
        info = np.iinfo(dt)
        D = np.clip(np.rint(D), info.min, info.max)
    data[sv] = D.astype(np.float32)
    if keep_int:
        out = nib.Nifti1Image(data.astype(dt), img.affine, hdr)
    else:
        hdr.set_data_dtype(np.float32)
        hdr.set_slope_inter(None, None)
        out = nib.Nifti1Image(data, img.affine, hdr)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp" + ("".join(dst.suffixes[-2:]) if dst.name.endswith(".gz") else dst.suffix))
    nib.save(out, str(tmp))
    os.replace(tmp, dst)
    return {"echo": e + 1, "file": str(dst), "dtype": str(dt if keep_int else np.dtype(np.float32)), "n_voxels_changed": int(len(sv[0])),
            "max_abs_change_pct": round(float(np.abs(np.expm1(y)).max() * 100), 3)}


def mirror_path(e1: Path, root: Path) -> Path:
    """Where the injected echo 1 goes: root / <path from the first all-digit folder (the subject) down>,
    so a subject number found in the path stays the same (mbme_tedana and mejoint read it from there)."""
    parts = e1.parts
    i = next((j for j, p in enumerate(parts[:-1]) if re.fullmatch(r"\d+", p)), None)
    rel = Path(*parts[i:]) if i is not None else Path(e1.parent.name) / e1.name
    return root / rel


def make_dir(run_dir, cfg: dict, level: str, seed: int, out_dir, baseline_frontend, tissue_dir=None, log=print) -> dict:
    """Write injected raw echoes for one run, plus the truth (sources, weights, time courses) and an
    mbme_tedana list file. Skips when truth.json already holds the same settings."""
    import hashlib
    import os

    import nibabel as nib

    from . import frontend, io
    run_dir, out_dir, fe = Path(run_dir), Path(out_dir), Path(baseline_frontend)
    I = cfg["inject"]
    if level not in I["levels"]:
        raise ValueError(f"level must be one of {list(I['levels'])}")
    params = {"run": run_dir.name, "level": level, "seed": int(seed), "inject": I,
              "project_seed": int(cfg["project_seed"])}
    phash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    tj = out_dir / "truth.json"
    if tj.exists():
        old = json.loads(tj.read_text())
        if old.get("params_hash") == phash and all(Path(f["file"]).exists() for f in old.get("echo_files", [])):
            log(f"injected echoes already made with these settings: {out_dir}")
            return old
    info = frontend.resolve_run(run_dir, cfg, "raw")
    te = info["te_ms"]
    vols = np.column_stack([np.asarray(v, int) for v in info["volumes"]])   # (T, E) raw volume per frame and echo
    T = vols.shape[0]
    mimg = nib.load(str(fe / "analysis_mask.nii.gz"))
    mask3d = np.asarray(mimg.dataobj) > 0
    voxel_mm = [float(x) for x in mimg.header.get_zooms()[:3]]
    classes3d = None
    if tissue_dir is not None and (Path(tissue_dir) / "classes.npy").exists():
        cl = np.load(Path(tissue_dir) / "classes.npy")
        if cl.size == mask3d.sum():
            classes3d = np.zeros(mask3d.shape, np.int8)
            classes3d[mask3d] = cl
    subj = int(info["subject"]) if str(info["subject"]).isdigit() else 0
    rng = np.random.default_rng(io.seed_sequence(cfg, subj, 7, list(I["levels"]).index(level), int(seed)))
    src = place_sources(mask3d, mimg.affine, voxel_mm, cfg, rng, classes3d)
    for s in src:
        a, b = amplitudes(s["type"], I["levels"][level], float(I["mixed_ratio_ms"]))
        s.update({"a_sd": a, "b_sd_per_s": b})
    W3 = weight_maps(mask3d.shape, [s["center"] for s in src], float(I["radius_mm"]), voxel_mm, mask3d)
    U = time_courses(T, float(cfg["acquisition"]["tr_s"]), I["band_hz"], len(src), rng)
    for i, s in enumerate(src):
        s["n_voxels"] = int((W3[..., i] > 0).sum())
        s["center_mm"] = [round(float(x), 1) for x in nib.affines.apply_affine(mimg.affine, s["center"])]
    a = [s["a_sd"] for s in src]
    b = [s["b_sd_per_s"] for s in src]
    files = [Path(f) for f in info["echo_files"]]
    e1_new = mirror_path(files[0], out_dir / "echoes")
    out_files = []
    for e, f in enumerate(files):
        dst = e1_new.with_name(f.name)
        log(f"  echo {e + 1}: {f} -> {dst}")
        out_files.append(inject_file(f, dst, e, vols[:, e], W3, U, a, b, te))
    # every other entry next to echo 1 (JSON sidecars, DICOM folders) as a symlink, so the folder looks the same
    names = {f.name for f in files}
    for p in files[0].parent.iterdir():
        q = e1_new.parent / p.name
        if p.name in names or q.exists() or q.is_symlink():
            continue
        os.symlink(p.resolve(), q)
    io.save_volume(W3, mimg, out_dir / "truth_weights.nii.gz")
    io.write_tsv(out_dir / "truth_timecourses.tsv", {s["id"]: list(U[:, i]) for i, s in enumerate(src)})
    np.savetxt(out_dir / "volumes_used_injected.tsv", vols, fmt="%d", delimiter="\t",
               header="\t".join(f"echo{k + 1}" for k in range(vols.shape[1])), comments="")
    (out_dir / "list.txt").write_text(str(e1_new) + "\n")
    truth = {"run": run_dir.name, "run_dir": str(run_dir), "level": level, "seed": int(seed), "params_hash": phash,
             "te_ms": te, "n_frames": int(T), "radius_mm": float(I["radius_mm"]), "voxel_mm": voxel_mm,
             "mixed_ratio_ms": float(I["mixed_ratio_ms"]), "band_hz": list(I["band_hz"]),
             "baseline_frontend": str(fe), "tissue_dir": None if tissue_dir is None else str(tissue_dir),
             "echo1_injected": str(e1_new), "echo_files": out_files, "sources": src,
             **io.provenance(cfg, {"echo1": str(files[0])})}
    io.write_json(tj, truth)
    return truth


# ============================================================================ recovery
def _corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _wls(est, w):
    """Weighted amplitude at the sphere: sum w est / sum w^2 (est = amplitude x w in the model)."""
    ok = np.isfinite(est) & (w > 0)
    d = float((w[ok] ** 2).sum())
    return float((w[ok] * est[ok]).sum() / d) if d > 0 else np.nan


def footprint_profile(ch: dict, mu, sigma, ratios, use, te_ms, M, w, core: float = 0.5) -> dict:
    """Echo profile and curvature of time courses M (T, k) in one source's own voxels: the four channel betas
    mapped back to the echoes (as te-check does), averaged with the source weight w over the 4-echo voxels
    where w >= core (amplitude at the source centre, fraction; same sign as the time course). Unlike
    te-check's profile (median over a component's 200 strongest voxels), the voxels are fixed by the truth,
    so noise voxels are not selected. Curvature fraction: debiased r0 energy over the energy of all four channels."""
    from . import te_check as tc
    te = np.asarray(te_ms, float)
    four = np.asarray(use, bool).all(1) & np.isfinite(ch["r1"]).all(1) & (np.asarray(w) >= core)
    idx = np.flatnonzero(four)
    k = M.shape[1]
    if idx.size < 3:
        return {"profile": np.full((te.size, k), np.nan), "curvature_fraction": np.full(k, np.nan), "n_voxels": int(idx.size)}
    m, Qb, _ = tc.basis(np.asarray(mu)[idx], np.asarray(sigma)[idx], ratios, te)
    Rraw = np.linalg.qr(m[:, :, None] * tc.design(te)[None])[1]
    sgn = np.sign(Rraw[:, 2, 2])
    sgn[sgn == 0] = 1
    B = {n: tc.ols_betas(np.asarray(ch[n])[idx], M) for n in ("c0", "c1", "r0", "r1")}
    B["r0"] = B["r0"] * sgn[:, None]
    bq = np.stack([B["c0"], B["c1"], B["r0"], B["r1"]], axis=1)                  # (v, 4, k)
    y = np.einsum("veq,vqk->vek", Qb, bq) / m[:, :, None]                         # (v, E, k) log-signal change per unit time course
    ww = np.asarray(w)[idx]
    prof = np.einsum("v,vek->ek", ww, y) / float((ww ** 2).sum())
    Mc = M - M.mean(0)
    vb = np.diag(np.linalg.inv(Mc.T @ Mc))                                        # unit-noise beta variance
    n = idx.size
    e_r0 = (B["r0"] ** 2).sum(0) - n * vb
    e_all = sum((B[x] ** 2).sum(0) for x in B) - 4 * n * vb
    with np.errstate(divide="ignore", invalid="ignore"):
        cf = np.where(e_all > 0, e_r0 / e_all, np.nan)
    return {"profile": prof, "curvature_fraction": cf, "n_voxels": int(n)}


def recover(c0, c1, R, Q, M, W, U, a, b, te_ms, maps: dict, footprint=None, comp_rows=None, te_rows=None,
            tedana=None, fp_inputs: dict | None = None, pca: dict | None = None) -> list[dict]:
    """Per injected source: the best component and what it carries.

    c0, c1 (V, T) channels; R (V, 2, 2); Q (T, p) nuisance basis or None; M (T, k) component time courses;
    W (V, n) truth weights; U (T, n) truth time courses; a, b (n,) amplitudes; maps: m0, m1 (full data)
    and m0_cf, m1_cf (cross-fitted), each (V, k); footprint (V, n) bool: sphere plus a 2-voxel margin
    (default W > 0); comp_rows / te_rows: components.json and te_check.json rows; tedana: {"mixing": (T, K),
    "metrics": list of rows} or None; fp_inputs: {"ch": c0/c1/r0/r1, "mu", "sigma", "ratios", "use"} for the
    echo profile and curvature in the source's own voxels (footprint_profile); pca: {"V": (T, K) temporal
    eigenvectors of the stacked channels, largest first, "n_rows": stacked rows} for detectability."""
    from scipy.optimize import linear_sum_assignment

    from . import decompose, metrics
    T, k = M.shape
    n = U.shape[1]
    Up = U - Q @ (Q.T @ U) if Q is not None else U.copy()
    rho = Up.std(0)                                                    # share of each source kept by the nuisance projection
    Ut = (Up - Up.mean(0)) / rho
    Mz = (M - M.mean(0)) / M.std(0)
    C = Ut.T @ Mz / T                                                  # (n, k) correlations
    ri, ci = linear_sum_assignment(-np.abs(C))
    hung = dict(zip(ri, ci))
    s_full, q_full = decompose.backsub(maps["m0"], maps["m1"], R)
    s_cf, q_cf = decompose.backsub(maps["m0_cf"], maps["m1_cf"], R)
    vb = np.diag(np.linalg.inv(Mz.T @ Mz))                            # beta noise variance per component (unit noise)
    fp = (W > 0) if footprint is None else footprint
    te = np.asarray(te_ms, float)
    tmix = None
    if tedana is not None and tedana.get("mixing") is not None and tedana["mixing"].shape[0] == T:
        X = tedana["mixing"]
        tmix = (X - X.mean(0)) / X.std(0)
    rows = []
    for i in range(n):
        j = int(np.argmax(np.abs(C[i])))
        sg = 1.0 if C[i, j] >= 0 else -1.0
        srt = np.sort(np.abs(C[i]))[::-1]
        w = W[:, i]
        f = fp[:, i]
        row = {"source": i, "best_component": j + 1, "r": round(float(abs(C[i, j])), 4),
               "second_r": round(float(srt[1]), 4) if k > 1 else None,
               "hungarian_component": int(hung[i]) + 1 if i in hung else None,
               "kept_by_nuisance_projection": round(float(rho[i]), 4)}
        # amplitude at the sphere, full-data and cross-fitted maps, vs the truth after the projection
        for tag, ss, qq in (("", s_full, q_full), ("_cf", s_cf, q_cf)):
            ah, bh = _wls(sg * ss[:, j], w), _wls(sg * qq[:, j], w)
            row[f"a_hat{tag}"] = round(ah, 6)
            row[f"b_hat_per_s{tag}"] = round(bh, 5)
            row[f"a_ratio{tag}"] = round(float(ah / (rho[i] * a[i])), 3) if a[i] != 0 else None
            row[f"b_ratio{tag}"] = round(float(bh / (rho[i] * b[i])), 3) if b[i] != 0 else None
            row[f"ratio_ms{tag}"] = round(ah / bh * 1000.0, 2) if bh != 0 and np.isfinite(bh) else None
        row["a_true_kept"] = round(float(rho[i] * a[i]), 6)
        row["b_true_kept_per_s"] = round(float(rho[i] * b[i]), 5)
        row["ratio_ms_true"] = round(a[i] / b[i] * 1000.0, 2) if b[i] != 0 else None
        # map agreement over the footprint (sphere + margin; whole-brain correlation is dominated by noise)
        row["map_r_q"] = round(_corr(sg * q_full[f, j], b[i] * w[f]), 4) if b[i] != 0 else None
        row["map_r_s"] = round(_corr(sg * s_full[f, j], a[i] * w[f]), 4) if a[i] != 0 else None
        # where the component's channel energy sits: share inside the footprint (noise-debiased)
        e = np.nan_to_num(maps["m0"][:, j] ** 2 - vb[j]) + np.where(np.isfinite(maps["m1"][:, j]), maps["m1"][:, j] ** 2 - vb[j], 0.0)
        tot = float(e.sum())
        row["energy_share_in_footprint"] = round(float(e[f].sum()) / tot, 4) if tot > 0 else None
        # oracle: the true (projected) time course as a fixed regressor; no ICA. Debiasing is exact here.
        m0o, m1o, vbo = metrics._single_regression_maps(c0, c1, Ut[:, [i]])
        en_all = metrics.energies(m0o, m1o, R, vbo)
        m0f, m1f = np.where(f[:, None], m0o, np.nan), np.where(f[:, None], m1o, np.nan)
        en_fp = metrics.energies(m0f, m1f, R, vbo)
        so, qo = decompose.backsub(m0o, m1o, R)
        aho, bho = _wls(so[:, 0], w), _wls(qo[:, 0], w)
        row["eta_oracle"] = round(float(en_all["eta"][0]), 4)
        row["eta_oracle_footprint"] = round(float(en_fp["eta"][0]), 4)
        row["a_hat_oracle"] = round(aho, 6)
        row["b_hat_oracle_per_s"] = round(bho, 5)
        row["ratio_ms_oracle"] = round(aho / bho * 1000.0, 2) if bho != 0 and np.isfinite(bho) else None
        # detectability: the source's channel variance summed over its footprint (noise-debiased; unit-noise
        # channels), against the white-noise PCA detection limit sqrt(rows / T); and how much of its time course
        # the top-k PCA dimensions of the data contain (spatial ICA only sees those dimensions)
        S = float(np.where(f, m0o[:, 0] ** 2 - vbo, 0.0).sum() + np.where(f & np.isfinite(m1o[:, 0]), m1o[:, 0] ** 2 - vbo, 0.0).sum())
        row["source_energy"] = round(S, 2)
        if pca is not None:
            row["detection_limit_white"] = round(float(np.sqrt(pca["n_rows"] / T)), 2)
            uu = Ut[:, i] - Ut[:, i].mean()
            cap = np.cumsum((pca["V"].T @ (uu / np.linalg.norm(uu))) ** 2)
            for kk in (50, 90, 150, 300):
                if kk <= cap.size:
                    row[f"captured_top{kk}"] = round(float(cap[kk - 1]), 4)
            hit = np.flatnonzero(cap >= 0.8)
            row["k_for_80pct"] = int(hit[0]) + 1 if hit.size else None
        # ICA: eta of the component and of its footprint only (cross-fitted maps, as in the sweep)
        vcf = maps.get("vb_cf")
        vbj = vcf[:, [j]] if vcf is not None else vb[j]
        m0c, m1c = maps["m0_cf"][:, [j]], maps["m1_cf"][:, [j]]
        row["eta_ica_footprint"] = round(float(metrics.energies(np.where(f[:, None], m0c, np.nan),
                                                                np.where(f[:, None], m1c, np.nan), R, vbj)["eta"][0]), 4)
        if comp_rows is not None:
            cr = comp_rows[j]
            for key in ("eta", "above_null", "label", "band_R2", "band_S0", "energy_single", "stability_median_abs_r", "iq"):
                row[key if key != "eta" else "eta_ica"] = cr.get(key)
        if te_rows is not None:
            tr_ = te_rows[j]
            for key in ("profile_pct", "curvature_fraction", "curvature_z", "profile_zero_crossing_ms", "ratio_ms_median",
                        "fraction_opposed", "null_fraction_opposed", "fraction_te1_te4_opposite", "n_sig"):
                row[f"te_{key}"] = tr_.get(key)
        # expected echo profile (%, dR2* > 0 alignment as in te-check; dS0-only: sign arbitrary)
        sb = np.sign(b[i]) if b[i] != 0 else 1.0
        row["profile_expected_pct"] = [round(float(100 * rho[i] * (sb * a[i] - t / 1000.0 * abs(b[i]))), 4) for t in te]
        if fp_inputs is not None:
            fpp = footprint_profile(fp_inputs["ch"], fp_inputs["mu"], fp_inputs["sigma"], fp_inputs["ratios"], fp_inputs["use"],
                                    te, np.column_stack([sg * Mz[:, j], Ut[:, i]]), w)
            row["profile_footprint_pct"] = [round(float(100 * x), 4) for x in fpp["profile"][:, 0]]
            row["profile_footprint_oracle_pct"] = [round(float(100 * x), 4) for x in fpp["profile"][:, 1]]
            row["curvature_fraction_footprint"] = round(float(fpp["curvature_fraction"][0]), 5) if np.isfinite(fpp["curvature_fraction"][0]) else None
            row["curvature_fraction_footprint_oracle"] = (round(float(fpp["curvature_fraction"][1]), 5)
                                                          if np.isfinite(fpp["curvature_fraction"][1]) else None)
            row["footprint_profile_voxels"] = fpp["n_voxels"]
            row["profile_expected_signed_pct"] = [round(float(100 * rho[i] * (a[i] - t / 1000.0 * b[i])), 4) for t in te]
        if tmix is not None:
            ct = Ut[:, i] @ tmix / T
            jt = int(np.argmax(np.abs(ct)))
            row["tedana_component"] = jt
            row["tedana_r"] = round(float(abs(ct[jt])), 4)
            met = tedana.get("metrics")
            if met is not None and jt < len(met):
                for key in ("kappa", "rho", "classification", "classification_tags"):
                    if key in met[jt]:
                        row[f"tedana_{key}"] = met[jt][key]
        rows.append(row)
    return rows


def natives(M_inj, M_base, inj_comps: set, rows_inj=None, rows_base=None) -> dict:
    """Do the other components survive the injection? Hungarian match (|corr| of time courses) of the
    non-injected components to the baseline decomposition of the same run."""
    from scipy.optimize import linear_sum_assignment
    if M_base is None or M_base.shape[0] != M_inj.shape[0]:
        return {"available": False}
    keep = [j for j in range(M_inj.shape[1]) if j not in inj_comps]
    A = (M_inj[:, keep] - M_inj[:, keep].mean(0)) / M_inj[:, keep].std(0)
    B = (M_base - M_base.mean(0)) / M_base.std(0)
    C = np.abs(A.T @ B / A.shape[0])
    r, c = linear_sum_assignment(-C)
    rr = C[r, c]
    out = {"available": True, "n_compared": len(keep), "n_matched_r09": int((rr >= 0.9).sum()),
           "median_r": round(float(np.median(rr)), 4)}
    if rows_inj is not None and rows_base is not None:
        d = [abs(float(rows_inj[keep[a]]["eta"]) - float(rows_base[b]["eta"])) for a, b, x in zip(r, c, rr) if x >= 0.9
             and rows_inj[keep[a]].get("eta") is not None and rows_base[b].get("eta") is not None]
        out["median_abs_eta_change_matched"] = round(float(np.median(d)), 4) if d else None
        nat = [float(rows_inj[j]["eta"]) for j in keep if rows_inj[j].get("above_null") and rows_inj[j].get("eta") is not None]
        out["native_eta_above_null"] = nat
    return out


def stacked_pca(c0, c1, chunk: int = 20000) -> dict:
    """Temporal eigenvectors (T, T), largest first, of the stacked channel rows [c0 | c1] (rows with all frames
    finite, demeaned over time), as the decomposition's PCA step sees them (T x T Gram, float64)."""
    T = c0.shape[1]
    G = np.zeros((T, T))
    n = 0
    for X in (c0, c1):
        for i in range(0, X.shape[0], chunk):
            x = np.asarray(X[i:i + chunk], np.float64)
            x = x[np.isfinite(x).all(1)]
            x -= x.mean(1, keepdims=True)
            G += x.T @ x
            n += x.shape[0]
    lam, V = np.linalg.eigh(G)
    o = np.argsort(lam)[::-1]
    return {"V": V[:, o], "lam_per_row": lam[o] / max(n, 1), "n_rows": n}


def _tsv_matrix(path) -> np.ndarray:
    t = np.genfromtxt(path, delimiter="\t", names=True)
    return np.column_stack([t[n] for n in t.dtype.names]).astype(np.float64)


def _tedana(run_dir) -> dict | None:
    """tedana mixing matrix and metrics from an mbme_tedana run folder (<run>/tedana/), if present."""
    import csv
    if run_dir is None:
        return None
    td = Path(run_dir) / "tedana"
    mix = sorted(td.glob("*ICA_mixing.tsv"))
    met = sorted(td.glob("*tedana_metrics.tsv"))
    if not mix:
        return None
    out = {"mixing": _tsv_matrix(mix[0]), "metrics": None}
    if met:
        with open(met[0]) as fh:
            out["metrics"] = [dict(r) for r in csv.DictReader(fh, delimiter="\t")]
        for r in out["metrics"]:
            for key in ("kappa", "rho"):
                try:
                    r[key] = round(float(r[key]), 2)
                except (KeyError, TypeError, ValueError):
                    pass
    return out


def recover_dir(inject_dir, frontend_dir, k_dir, mbme_run=None, baseline_k_dir=None, te_check: bool = True) -> dict:
    """recover() on folders; writes recovery.json, recovery.md and recovery.png in inject_dir."""
    import nibabel as nib
    from scipy import ndimage

    from . import io
    from . import te_check as tc
    idir, fe, kd = Path(inject_dir), Path(frontend_dir), Path(k_dir)
    truth = json.loads((idir / "truth.json").read_text())
    src = truth["sources"]
    mimg = nib.load(str(fe / "analysis_mask.nii.gz"))
    mask3d = np.asarray(mimg.dataobj) > 0
    W3 = np.asarray(nib.load(str(idir / "truth_weights.nii.gz")).dataobj, np.float64)
    if W3.ndim == 3:
        W3 = W3[..., None]
    W = W3[mask3d]
    fp3 = np.stack([ndimage.binary_dilation(W3[..., i] > 0, iterations=2) for i in range(W3.shape[3])], axis=3)
    FP = fp3[mask3d]
    U = _tsv_matrix(idir / "truth_timecourses.tsv")
    c0, c1 = np.load(fe / "c0.npy"), np.load(fe / "c1.npy")
    R = np.load(fe / "R.npy")
    Q = np.load(fe / "nuisance_Q.npy") if (fe / "nuisance_Q.npy").exists() else None
    M = _tsv_matrix(kd / "time_courses.tsv")
    if U.shape[0] != M.shape[0]:
        raise ValueError(f"truth has {U.shape[0]} frames, the decomposition {M.shape[0]}")
    notes = []
    if mbme_run is not None and (Path(mbme_run) / "volumes_used.tsv").exists():
        vu = np.loadtxt(Path(mbme_run) / "volumes_used.tsv", skiprows=1, ndmin=2).astype(int)
        vi = np.loadtxt(idir / "volumes_used_injected.tsv", skiprows=1, ndmin=2).astype(int)
        if vu.shape != vi.shape or not np.array_equal(vu, vi):
            notes.append("WARNING: mbme_tedana paired the injected echoes differently from the original run; "
                         "the truth time courses are misaligned in the changed frames")
    vox = lambda name: np.asarray(nib.load(str(kd / f"{name}.nii.gz")).dataobj, np.float64)[mask3d]
    maps = {"m0": vox("m0"), "m1": vox("m1"), "m0_cf": np.load(kd / "m0_cf.npy").astype(np.float64),
            "m1_cf": np.load(kd / "m1_cf.npy").astype(np.float64), "vb_cf": np.load(kd / "vb_cf.npy").astype(np.float64)}
    cj = json.loads((kd / "components.json").read_text())
    te_rows = None
    if te_check and (fe / "r0.npy").exists():
        tj = kd / "te_check.json"
        te_rows = (json.loads(tj.read_text()) if tj.exists() else tc.check_dir(fe, kd))["components"]
    ted = _tedana(mbme_run)
    fpi = None
    if (fe / "r0.npy").exists() and (fe / "r1.npy").exists():
        fs = json.loads((fe / "summary.json").read_text())
        mapv = lambda name: np.asarray(nib.load(str(fe / f"{name}.nii.gz")).dataobj, np.float64)[mask3d]
        fpi = {"ch": {"c0": c0, "c1": c1, "r0": np.load(fe / "r0.npy"), "r1": np.load(fe / "r1.npy")}, "mu": mapv("mu"),
               "sigma": mapv("sigma"), "ratios": fs["echo_noise_ratios_used"], "use": mapv("usable") > 0.5}
    a = [float(s["a_sd"]) for s in src]
    b = [float(s["b_sd_per_s"]) for s in src]
    pca = stacked_pca(c0, c1)
    rows = recover(c0, c1, R, Q, M, W, U, a, b, truth["te_ms"], maps, FP, cj["components"], te_rows, ted, fpi, pca)
    for r, s in zip(rows, src):
        r.update({"id": s["id"], "type": s["type"], "region": s["region"], "center": s["center"], "a_sd": s["a_sd"],
                  "b_sd_per_s": s["b_sd_per_s"]})
    inj = {r["best_component"] - 1 for r in rows if r["r"] >= 0.5}
    nat = {"available": False}
    if baseline_k_dir is not None and (Path(baseline_k_dir) / "time_courses.tsv").exists():
        bj = json.loads((Path(baseline_k_dir) / "components.json").read_text())
        nat = natives(M, _tsv_matrix(Path(baseline_k_dir) / "time_courses.tsv"), inj, cj["components"], bj["components"])
    else:
        nat["native_eta_above_null"] = [float(c["eta"]) for j, c in enumerate(cj["components"])
                                        if j not in inj and c.get("above_null") and c.get("eta") is not None]
    if te_rows is not None:
        nat["native_curvature_fraction"] = [te_rows[j]["curvature_fraction"] for j in range(len(te_rows))
                                            if j not in inj and cj["components"][j].get("above_null")
                                            and te_rows[j].get("curvature_fraction") is not None]
    out = {"run": truth["run"], "level": truth["level"], "seed": truth["seed"], "k": int(M.shape[1]),
           "inject_dir": str(idir), "frontend_dir": str(fe), "k_dir": str(kd), "mbme_run": None if mbme_run is None else str(mbme_run),
           "baseline_k_dir": None if baseline_k_dir is None else str(baseline_k_dir), "notes": notes,
           "tedana_found": ted is not None, "null_threshold": cj.get("null", {}).get("threshold"),
           "sources": rows, "natives": nat, **io.provenance({}, {})}
    io.write_json(idir / "recovery.json", out)
    io.write_text(idir / "recovery.md", render(out, truth))
    try:
        figure(out, truth, kd, mask3d, W3, U, M, idir / "recovery.png")
    except Exception as exc:                                           # the numbers matter more than the figure
        print(f"recovery figure failed: {exc}")
    return out


# ============================================================================ reports
def _f(v, fmt=".3f"):
    if v is None:
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_f(x, ".2f") for x in v) + "]"
    if isinstance(v, str):
        return v
    try:
        return "n/a" if not np.isfinite(float(v)) else format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def render(out: dict, truth: dict) -> str:
    L = [f"# Injection recovery: {out['run']}, level {out['level']}, seed {out['seed']}, k = {out['k']}", ""]
    L += [f"{len(out['sources'])} sources injected into the raw echoes (radius {truth['radius_mm']} mm, band "
          f"{truth['band_hz'][0]}-{truth['band_hz'][1]} Hz), then the unchanged pipeline. Amplitudes are temporal SDs at "
          "the source centre; 'kept' = the share that survives the nuisance projection.", ""]
    L += [f"- {n}" for n in out["notes"]]
    L += ["## Found?", "",
          "| source | dS0 SD | dR2* SD (s^-1) | best comp | r | 2nd r | above null | label | energy in footprint |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in out["sources"]:
        L.append(f"| {r['id']} | {_f(r['a_sd'], '.4f')} | {_f(r['b_sd_per_s'], '.2f')} | {r['best_component']} | {_f(r['r'])} | "
                 f"{_f(r['second_r'])} | {_f(r.get('above_null'))} | {r.get('label', 'n/a')} | {_f(r['energy_share_in_footprint'])} |")
    if any("captured_top50" in r for r in out["sources"]):
        L += ["", "## Could the decomposition see it?", "",
              "Source energy: the source's channel variance summed over its footprint (noise-debiased; unit thermal noise). "
              "The white-noise PCA limit is sqrt(rows / T); real data also hold large structured signals above it. "
              "Captured: share of the injected time course inside the top-k PCA dimensions of the data (ICA at k sees only those; a time course unrelated to the data scores k / T, e.g. 0.06 at k = 50, T = 800).", "",
              "| source | source energy | white-noise limit | captured top 50 | top 90 | top 150 | top 300 | k for 80 % |",
              "|---|---|---|---|---|---|---|---|"]
        for r in out["sources"]:
            L.append(f"| {r['id']} | {_f(r.get('source_energy'), '.1f')} | {_f(r.get('detection_limit_white'), '.1f')} | "
                     f"{_f(r.get('captured_top50'))} | {_f(r.get('captured_top90'))} | {_f(r.get('captured_top150'))} | "
                     f"{_f(r.get('captured_top300'))} | {r.get('k_for_80pct', 'n/a')} |")
    L += ["", "## Physics: eta and the dS0/dR2* ratio", "",
          "eta oracle: the true time course as a fixed regressor (no ICA); eta ICA: the matched component "
          "(cross-fitted maps, as in the sweep). Ratio = recovered dS0/S0 over dR2* at the sphere, in ms "
          "(true: 0 for r2, undefined for s0, the mixed ratio for mixed). The real components show eta ~ 0.4 "
          "and a ratio near 32 ms.", "",
          "| source | eta oracle | eta oracle (footprint) | eta ICA | eta ICA (footprint) | ratio true | ratio oracle | ratio ICA | ratio ICA cf | dR2* recovered / true | dS0 recovered / true |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    rat = lambda r, key: "-" if r["type"] == "s0" else _f(r[key], ".1f")      # no dR2* in a dS0-only source: ratio undefined
    for r in out["sources"]:
        L.append(f"| {r['id']} | {_f(r['eta_oracle'])} | {_f(r['eta_oracle_footprint'])} | {_f(r.get('eta_ica'))} | "
                 f"{_f(r['eta_ica_footprint'])} | {_f(r['ratio_ms_true'], '.1f')} | {rat(r, 'ratio_ms_oracle')} | "
                 f"{rat(r, 'ratio_ms')} | {rat(r, 'ratio_ms_cf')} | {_f(r['b_ratio'], '.2f')} | {_f(r['a_ratio'], '.2f')} |")
    if any("te_profile_pct" in r for r in out["sources"]):
        L += ["", "## Echo profile and curvature (te-check instrument)", "",
              "| source | profile expected (%) | profile of the component (%) | zero crossing (ms) | curvature fraction | curvature z |",
              "|---|---|---|---|---|---|"]
        for r in out["sources"]:
            L.append(f"| {r['id']} | {_f(r['profile_expected_pct'])} | {_f(r.get('te_profile_pct'))} | "
                     f"{_f(r.get('te_profile_zero_crossing_ms'), '.1f')} | {_f(r.get('te_curvature_fraction'), '.4f')} | "
                     f"{_f(r.get('te_curvature_z'), '.1f')} |")
    if any("profile_footprint_pct" in r for r in out["sources"]):
        L += ["", "## Echo profile and curvature in the source's own voxels", "",
              "Weighted over the 4-echo voxels of the sphere core (weight >= 0.5); % at the centre, same sign as the injected "
              "time course. Component: the matched ICA time course; oracle: the true time course. These voxels are fixed "
              "by the truth, so no noise voxels are selected (te-check's profile above uses the component's 200 strongest "
              "voxels, which for a focal source are mostly noise).", "",
              "| source | expected (%) | component (%) | oracle (%) | curvature component | curvature oracle | voxels |",
              "|---|---|---|---|---|---|---|"]
        for r in out["sources"]:
            L.append(f"| {r['id']} | {_f(r.get('profile_expected_signed_pct'))} | {_f(r.get('profile_footprint_pct'))} | "
                     f"{_f(r.get('profile_footprint_oracle_pct'))} | {_f(r.get('curvature_fraction_footprint'), '.4f')} | "
                     f"{_f(r.get('curvature_fraction_footprint_oracle'), '.4f')} | {r.get('footprint_profile_voxels', 'n/a')} |")
    if out.get("tedana_found"):
        L += ["", "## tedana on the same injected echoes", "", "| source | tedana comp | r | kappa | rho | classification |",
              "|---|---|---|---|---|---|"]
        for r in out["sources"]:
            L.append(f"| {r['id']} | {r.get('tedana_component', 'n/a')} | {_f(r.get('tedana_r'))} | {_f(r.get('tedana_kappa'), '.1f')} | "
                     f"{_f(r.get('tedana_rho'), '.1f')} | {r.get('tedana_classification', 'n/a')} |")
    nat = out["natives"]
    L += ["", "## The other components", ""]
    if nat.get("available"):
        L.append(f"Matched to the decomposition without injection: {nat['n_matched_r09']} of {nat['n_compared']} with "
                 f"|r| >= 0.9 (median {nat['median_r']}); median |eta change| of those {_f(nat.get('median_abs_eta_change_matched'))}.")
    else:
        L.append("No baseline decomposition of this run was given.")
    ne = nat.get("native_eta_above_null") or []
    if ne:
        L.append(f"Native components above the null: eta median {np.median(ne):.3f} (IQR {np.percentile(ne, 25):.3f}-"
                 f"{np.percentile(ne, 75):.3f}, n = {len(ne)}).")
    nc = nat.get("native_curvature_fraction") or []
    if nc:
        L.append(f"Their curvature fraction: median {np.median(nc):.4f}.")
    return "\n".join(L) + "\n"


def figure(out, truth, kd, mask3d, W3, U, M, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    rows = out["sources"]
    n = len(rows)
    s4 = np.asarray(nib.load(str(Path(kd) / "s.nii.gz")).dataobj, np.float64)
    q4 = np.asarray(nib.load(str(Path(kd) / "q_per_s.nii.gz")).dataobj, np.float64)
    te = np.asarray(truth["te_ms"], float)
    tr = 0.75
    fig, axs = plt.subplots(n, 4, figsize=(15, 2.5 * n), squeeze=False)
    for i, r in enumerate(rows):
        j = r["best_component"] - 1
        sg = 1.0 if np.dot(U[:, i] - U[:, i].mean(), M[:, j] - M[:, j].mean()) >= 0 else -1.0
        nt = min(400, U.shape[0])
        t = np.arange(nt) * tr
        ax = axs[i, 0]
        ax.plot(t, U[:nt, i], color="#52514e", lw=1, label="injected")
        ax.plot(t, sg * (M[:nt, j] - M[:nt, j].mean()) / M[:nt, j].std(), color="#2a78d6", lw=1, label=f"comp {j + 1}")
        ax.set_title(f"{r['id']}: r = {r['r']:.2f}, eta ICA {_f(r.get('eta_ica'), '.2f')}, oracle {_f(r['eta_oracle'], '.2f')}",
                     fontsize=8.5, loc="left")
        if i == 0:
            ax.legend(fontsize=7, frameon=False)
        z = int(r["center"][2])
        cont = W3[:, :, z, i].T > 0
        for col, vol, name, unit in ((1, q4, "dR2*", "s^-1"), (2, s4, "dS0/S0", "")):
            ax = axs[i, col]
            sl = sg * vol[:, :, z, j].T
            lim = np.nanpercentile(np.abs(sl[np.isfinite(sl)]), 99.5) if np.isfinite(sl).any() else 1.0
            ax.imshow(np.where(mask3d[:, :, z].T, sl, np.nan), origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.contour(cont, levels=[0.5], colors="k", linewidths=0.6)
            ax.set_title(f"{name} map, slice {z} (+-{lim:.3g} {unit})", fontsize=8, loc="left")
            ax.set_xticks([]); ax.set_yticks([])
        ax = axs[i, 3]
        if r.get("profile_footprint_pct") is not None:
            ax.plot(te, r["profile_expected_signed_pct"], "s--", color="#52514e", label="expected")
            ax.plot(te, r["profile_footprint_pct"], "o-", color="#2a78d6", label="component, source voxels")
            ax.plot(te, r["profile_footprint_oracle_pct"], "^:", color="#1baf7a", label="oracle, source voxels")
        else:
            ax.plot(te, r["profile_expected_pct"], "s--", color="#52514e", label="expected")
        if r.get("te_profile_pct") is not None:
            ax.plot(te, r["te_profile_pct"], "o-", color="#eb6834", lw=0.8, ms=3, label="te-check (200 strongest voxels)")
        ax.axhline(0, color="#e4e3df")
        ax.set_title(f"echo profile (%), curvature at source {_f(r.get('curvature_fraction_footprint'), '.3f')}, "
                     f"te-check {_f(r.get('te_curvature_fraction'), '.3f')}", fontsize=8, loc="left")
        if i == 0:
            ax.legend(fontsize=7, frameon=False)
    axs[-1, 0].set_xlabel("time (s)")
    axs[-1, 3].set_xlabel("TE (ms)")
    fig.suptitle(f"{out['run']}, {out['level']} (seed {out['seed']}), k {out['k']}: injected sources and their best components",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ============================================================================ across runs
def summary(root, out_dir=None) -> dict:
    """Pool every recovery.json under root: real-data positions of known sources, by type and level."""
    from . import io
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root
    recs = [json.loads(p.read_text()) for p in sorted(root.glob("**/recovery.json"))]
    rows, nat_eta, nat_curv = [], [], []
    for rc in recs:
        for r in rc["sources"]:
            rows.append({"run": rc["run"], "level": rc["level"], "seed": rc["seed"], **r})
        nat_eta += rc["natives"].get("native_eta_above_null") or []
        nat_curv += rc["natives"].get("native_curvature_fraction") or []
    groups = {}
    for r in rows:
        groups.setdefault((r["type"], r["level"]), []).append(r)
    rec_r = None
    try:
        rec_r = float(io.load_config()["inject"]["recovered_r"])
    except Exception:
        rec_r = 0.9
    table = []
    for (kind, level), rs in sorted(groups.items()):
        found = [r for r in rs if r["r"] >= rec_r]
        med = lambda key, src=found: (float(np.nanmedian([np.nan if r.get(key) is None else float(r[key]) for r in src]))
                                      if src else None)
        rng_ = lambda key: ([round(float(np.nanmin(v)), 3), round(float(np.nanmax(v)), 3)]
                            if (v := [float(r[key]) for r in found if r.get(key) is not None]) else None)
        table.append({"type": kind, "level": level, "n": len(rs), "n_recovered": len(found),
                      "n_above_null": sum(bool(r.get("above_null")) for r in found),
                      "eta_ica_median": med("eta_ica"), "eta_ica_range": rng_("eta_ica"),
                      "eta_oracle_median": med("eta_oracle", rs), "eta_oracle_range": ([round(float(np.nanmin(v)), 3), round(float(np.nanmax(v)), 3)]
                                                                                     if (v := [float(r["eta_oracle"]) for r in rs]) else None),
                      "ratio_ms_ica_median": med("ratio_ms"), "ratio_ms_oracle_median": med("ratio_ms_oracle", rs),
                      "b_ratio_median": med("b_ratio"), "a_ratio_median": med("a_ratio"),
                      "curvature_fraction_median": med("te_curvature_fraction"),
                      "curvature_footprint_median": med("curvature_fraction_footprint"),
                      "curvature_footprint_oracle_median": med("curvature_fraction_footprint_oracle", rs),
                      "zero_crossings": sum(r.get("te_profile_zero_crossing_ms") is not None for r in found),
                      "best_r_median": med("r", rs), "best_r_max": max(float(r["r"]) for r in rs),
                      "eta_oracle_footprint_median": med("eta_oracle_footprint", rs),
                      "source_energy_median": med("source_energy", rs),
                      "detection_limit_white": med("detection_limit_white", rs),
                      "captured_top50_median": med("captured_top50", rs),
                      "k_for_80pct_median": med("k_for_80pct", rs),
                      "labels": {lab: sum(r.get("label") == lab for r in found) for lab in
                                 sorted({str(r.get("label")) for r in found})}})
    res = {"n_recoveries": len(recs), "recovered_r": rec_r, "table": table, "rows": rows,
           "native_eta_above_null": {"n": len(nat_eta), "median": float(np.median(nat_eta)) if nat_eta else None,
                                     "iqr": [float(np.percentile(nat_eta, 25)), float(np.percentile(nat_eta, 75))] if nat_eta else None},
           "native_curvature_fraction_median": float(np.median(nat_curv)) if nat_curv else None}
    out_dir.mkdir(parents=True, exist_ok=True)
    io.write_json(out_dir / "inject_summary.json", res)
    io.write_text(out_dir / "inject_summary.md", render_summary(res))
    try:
        summary_figure(res, nat_eta, out_dir / "inject_summary.png")
    except Exception as exc:
        print(f"summary figure failed: {exc}")
    return res


def render_summary(res: dict) -> str:
    L = ["# Injection calibration: known sources through the real pipeline", "",
         f"{res['n_recoveries']} injection runs. Recovered = best component |r| >= {res['recovered_r']}. "
         "eta ICA, ratio and curvature are over recovered sources; eta oracle (true time course, no ICA) over all.", "",
         "| type | level | n | recovered | above null | eta ICA median [range] | eta oracle median [range] | ratio ICA (ms) | ratio oracle (ms) | dR2* rec/true | dS0 rec/true | curvature te-check | curvature at source (ICA / oracle) | te-check zero crossings | labels |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in res["table"]:
        L.append(f"| {t['type']} | {t['level']} | {t['n']} | {t['n_recovered']} | {t['n_above_null']} | "
                 f"{_f(t['eta_ica_median'])} {_f(t['eta_ica_range'])} | {_f(t['eta_oracle_median'])} {_f(t['eta_oracle_range'])} | "
                 f"{_f(t['ratio_ms_ica_median'], '.1f')} | {_f(t['ratio_ms_oracle_median'], '.1f')} | {_f(t['b_ratio_median'], '.2f')} | "
                 f"{_f(t['a_ratio_median'], '.2f')} | {_f(t['curvature_fraction_median'], '.4f')} | "
                 f"{_f(t['curvature_footprint_median'], '.4f')} / {_f(t['curvature_footprint_oracle_median'], '.4f')} | {t['zero_crossings']} | "
                 f"{', '.join(f'{k} {v}' for k, v in t['labels'].items())} |")
    L += ["", "All sources (found or not): best component |r|, the oracle eta at the source's own voxels, and detectability "
              "(source energy vs the white-noise PCA limit; share of the time course in the top 50 PCA dimensions, chance 50 / T; k needed for 80 %).", "",
          "| type | level | best r median (max) | eta oracle at source | source energy | white-noise limit | captured top 50 | k for 80 % |",
          "|---|---|---|---|---|---|---|---|"]
    for t in res["table"]:
        L.append(f"| {t['type']} | {t['level']} | {_f(t.get('best_r_median'), '.2f')} ({_f(t.get('best_r_max'), '.2f')}) | "
                 f"{_f(t.get('eta_oracle_footprint_median'))} | {_f(t.get('source_energy_median'), '.1f')} | "
                 f"{_f(t.get('detection_limit_white'), '.1f')} | {_f(t.get('captured_top50_median'))} | {_f(t.get('k_for_80pct_median'), '.0f')} |")
    ne = res["native_eta_above_null"]
    if ne["n"]:
        L += ["", f"Native components above the null in the same decompositions: eta median {ne['median']:.3f} "
                  f"(IQR {ne['iqr'][0]:.3f}-{ne['iqr'][1]:.3f}, n = {ne['n']}); curvature fraction median "
                  f"{_f(res['native_curvature_fraction_median'], '.4f')}."]
    L += ["", "How to read it: if recovered r2 sources sit at eta ~ 0 with a ratio near 0 ms and little curvature, the "
              "pipeline does not create the native components' eta ~ 0.4, 32 ms ratio and curvature; those belong to "
              "the signal. If r2 sources come out like the natives, the effect is made by the noise or the processing. "
              "Oracle vs ICA separates the front end (oracle) from the decomposition (ICA)."]
    return "\n".join(L) + "\n"


def summary_figure(res, nat_eta, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["rows"]
    types = [t for t in TYPES if any(r["type"] == t for r in rows)]
    cols = {"strong": "#2a78d6", "weak": "#eb6834"}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, key, title in ((axs[0], "eta_ica", "eta of the matched component (ICA)"), (axs[1], "eta_oracle", "eta oracle (true time course)")):
        if nat_eta:
            ax.axhspan(np.percentile(nat_eta, 25), np.percentile(nat_eta, 75), color="#e4e3df", label="native components, IQR")
        for x, t in enumerate(types):
            for r in rows:
                if r["type"] == t and r.get(key) is not None:
                    jit = (hash(r["id"] + r["run"]) % 100) / 400 - 0.12
                    mk = "o" if r["r"] >= res["recovered_r"] else "x"
                    ax.plot(x + jit, r[key], mk, color=cols.get(r["level"], "#52514e"), ms=5)
        ax.set_xticks(range(len(types)), types)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(title, fontsize=9, loc="left")
    axs[0].set_ylabel("eta")
    for lev, c in cols.items():
        axs[0].plot([], [], "o", color=c, label=lev)
    axs[0].plot([], [], "x", color="#52514e", label="not recovered")
    axs[0].legend(fontsize=7, frameon=False)
    ax = axs[2]
    for x, t in enumerate(types):
        for r in rows:
            v = r.get("ratio_ms")
            if r["type"] == t and v is not None and r["r"] >= res["recovered_r"] and t != "s0":
                ax.plot(x + (hash(r["id"] + r["run"]) % 100) / 400 - 0.12, v, "o", color=cols.get(r["level"], "#52514e"), ms=5)
    ax.axhline(32, color="#52514e", ls="--", lw=1)
    ax.text(0.02, 33, "native components ~32 ms", fontsize=7)
    ax.set_xticks(range(len(types)), types)
    ax.set_title("dS0/dR2* ratio at the source (ms; r2 truth 0)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
