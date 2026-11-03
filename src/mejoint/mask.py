"""Robust brain mask built from the input echoes alone (no tedana or pipeline outputs).

Why not a global intensity threshold (nilearn compute_epi_mask, used by mbme_tedana): a 20-channel
sum-of-squares image without intensity normalisation is darker in the centre, and at TR 0.75 s CSF is
T1-saturated, so the deep brain falls below a whole-image threshold and the mask gets a central hole
(seen across participants). This mask avoids global intensity thresholds:

  1. head vs air: echo-1 mean above k_head x the air level, where the air level is the median of the
     image border (pure noise); the threshold follows the noise, not the intensity histogram;
  2. shading correction: divide by a heavily smoothed copy of itself (normalised convolution inside
     the head), so brain intensity is ~1 everywhere, centre included;
  3. brain vs scalp: keep normalised intensity above k_brain (drops the dark skull gap) AND a brain-like
     decay: mu_last / mu_1 = exp(-(TE_last - TE_1) / T2*) is independent of coil shading, and neck and
     face muscle (T2* ~20-30 ms) decays faster than brain (40-70 ms) or CSF (longer); erode once to cut
     thin bridges (scalp, orbits), keep the largest connected piece, regrow;
  4. fill every enclosed hole in 3D and slice by slice (ventricles, dark deep structures), close once,
     then regrow one voxel into bright head tissue (edge brain whose late echoes are lost to dephasing).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _largest_component(m: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(m)
    if n <= 1:
        return m.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def fill_holes(m: np.ndarray, axis: int = 2) -> np.ndarray:
    """Fill enclosed holes in 3D and in every slice along `axis` (the superior-inferior axis)."""
    out = ndimage.binary_fill_holes(m)
    for z in range(m.shape[axis]):
        sl = [slice(None)] * 3
        sl[axis] = z
        out[tuple(sl)] = ndimage.binary_fill_holes(out[tuple(sl)])
    return out


def _smooth_inside(img: np.ndarray, region: np.ndarray, sigma_vox) -> np.ndarray:
    num = ndimage.gaussian_filter(np.where(region, img, 0.0), sigma_vox, mode="constant")
    den = ndimage.gaussian_filter(region.astype(float), sigma_vox, mode="constant")
    return np.where(den > 1e-3, num / np.maximum(den, 1e-12), np.nan)


def build_mask(mean_echo1: np.ndarray, voxel_mm, cfg: dict, si_axis: int = 2,
               mean_echo_last: np.ndarray | None = None, te_ms=None) -> dict:
    """Brain mask from the temporal mean of echo 1 (3D). Returns the mask, the head mask and the
    numbers used, so every step can be checked."""
    M = cfg["mask"]
    m1 = np.asarray(mean_echo1, float)
    vox = np.asarray(voxel_mm, float)
    b = int(M["border_vox"])
    border = np.zeros(m1.shape, bool)
    border[:b], border[-b:], border[:, :b], border[:, -b:] = True, True, True, True
    air = float(np.median(m1[border]))
    if air <= 0:                                                       # zero-filled background: use a positive floor
        pos = m1[border & (m1 > 0)]
        air = float(np.median(pos)) if pos.size else float(np.percentile(m1[m1 > 0], 1))

    head = _largest_component(m1 > float(M["k_head"]) * air)
    head = fill_holes(head, si_axis)

    sig = float(M["shading_fwhm_mm"]) / np.sqrt(8 * np.log(2)) / vox
    shade = _smooth_inside(m1, head, sig)
    norm = np.where(head, m1 / shade, 0.0)
    core = head & (norm > float(M["k_brain"]))
    decay_ok = np.ones(m1.shape, bool)
    r_min = None
    if mean_echo_last is not None and te_ms is not None:
        dte = float(te_ms[-1]) - float(te_ms[0])
        r_min = float(np.exp(-dte / float(M["t2s_min_ms"])))
        with np.errstate(divide="ignore", invalid="ignore"):
            decay_ok = np.asarray(mean_echo_last, float) / m1 > r_min
        core &= decay_ok
    core = ndimage.binary_erosion(core, iterations=int(M["erode_vox"]))
    core = _largest_component(core)
    core = ndimage.binary_dilation(core, iterations=int(M["erode_vox"])) & head
    brain = fill_holes(core, si_axis)
    brain = ndimage.binary_closing(brain, iterations=int(M["close_vox"])) & head
    brain = fill_holes(brain, si_axis)
    for _ in range(int(M.get("regrow_vox", 0))):
        brain |= ndimage.binary_dilation(brain) & head & (norm > float(M["k_brain"]))

    voxel_ml = float(np.prod(vox)) / 1000.0
    return {"mask": brain, "head": head, "normalised": norm, "air_level": air, "decay_ratio_min": r_min,
            "volume_ml": round(float(brain.sum()) * voxel_ml, 1), "head_volume_ml": round(float(head.sum()) * voxel_ml, 1),
            "n_voxels": int(brain.sum())}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


# ======================================================================= QC across runs
def qc_all(pipeline_out, cfg: dict, out_dir, log=print) -> list[dict]:
    """Build the analysis mask for every mbme_tedana run (raw echoes 1 and last only) and record its
    volume and agreement with the pipeline mask. Writes <out>/<run>/{mean_echo1,mask}.nii.gz and
    <out>/mask_qc.tsv; the contact sheet is drawn by scripts/plot_mask_qc.py."""
    import traceback
    from pathlib import Path

    import nibabel as nib

    from . import frontend, io

    out_dir = Path(out_dir)
    rows = []
    for rd in sorted(p.parent for p in Path(pipeline_out).glob("*/result.json")):
        try:
            info = frontend.resolve_run(rd, cfg, "raw")
        except ValueError as exc:                                      # e.g. excluded subject
            log(f"{rd.name}: skipped ({exc})")
            continue
        try:
            ref = nib.load(info["echo_files"][0])

            def tmean(k):
                d = np.asarray(nib.load(info["echo_files"][k]).dataobj, dtype=np.float32)
                return d[..., np.asarray(info["volumes"][k], int)].mean(axis=3)
            si = next(i for i, c in enumerate(nib.aff2axcodes(ref.affine)) if c in ("S", "I"))
            m1, ml = tmean(0), tmean(len(info["echo_files"]) - 1)
            mk = build_mask(m1, ref.header.get_zooms()[:3], cfg, si_axis=si, mean_echo_last=ml, te_ms=info["te_ms"])
            od = out_dir / rd.name
            io.save_volume(m1, ref, od / "mean_echo1.nii.gz")
            io.save_volume(mk["mask"].astype(np.float32), ref, od / "mask.nii.gz")
            row = {"run": rd.name, "volume_ml": mk["volume_ml"], "head_volume_ml": mk["head_volume_ml"],
                   "air_level": round(mk["air_level"], 2), "pipeline_mask": info["mask"]}
            pm = Path(info["mask"])
            if pm.exists():
                p = np.asarray(nib.load(str(pm)).dataobj) > 0
                vml = float(np.prod(ref.header.get_zooms()[:3])) / 1000.0
                row.update(pipeline_volume_ml=round(float(p.sum()) * vml, 1), dice_vs_pipeline=round(dice(mk["mask"], p), 4),
                           ml_only_analysis=round(float((mk["mask"] & ~p).sum()) * vml, 1),
                           ml_only_pipeline=round(float((p & ~mk["mask"]).sum()) * vml, 1))
            rows.append(row)
            log(f"{rd.name}: {mk['volume_ml']} ml")
        except Exception as exc:
            io.write_text(out_dir / rd.name / "error.txt", traceback.format_exc())
            rows.append({"run": rd.name, "error": f"{type(exc).__name__}: {exc}"})
            log(f"{rd.name}: FAILED ({exc})")
    cols = ["run", "volume_ml", "head_volume_ml", "pipeline_volume_ml", "dice_vs_pipeline", "ml_only_analysis",
            "ml_only_pipeline", "air_level", "pipeline_mask", "error"]
    io.write_tsv(out_dir / "mask_qc.tsv", {c: [r.get(c, "") for r in rows] for c in cols})
    io.write_json(out_dir / "mask_qc_status.json", {"rows": rows, **io.provenance(cfg, {})})
    return rows
