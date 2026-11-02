"""Robust brain mask built from the input echoes alone (no tedana or pipeline outputs).

Why not a global intensity threshold (nilearn compute_epi_mask, used by mbme_tedana): a 20-channel
sum-of-squares image without intensity normalisation is darker in the centre, and at TR 0.75 s CSF is
T1-saturated, so the deep brain falls below a whole-image threshold and the mask gets a central hole
(seen across participants). This mask avoids global intensity thresholds:

  1. head vs air: echo-1 mean above k_head x the air level, where the air level is the median of the
     image border (pure noise); the threshold follows the noise, not the intensity histogram;
  2. shading correction: divide by a heavily smoothed copy of itself (normalised convolution inside
     the head), so brain intensity is ~1 everywhere, centre included;
  3. brain vs scalp: keep normalised intensity above k_brain (drops the dark skull gap), erode once to
     cut thin bridges (scalp, orbits), keep the largest connected piece, regrow;
  4. fill every enclosed hole in 3D and slice by slice (ventricles, dark deep structures), close once.
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


def build_mask(mean_echo1: np.ndarray, voxel_mm, cfg: dict, si_axis: int = 2) -> dict:
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
    core = ndimage.binary_erosion(core, iterations=int(M["erode_vox"]))
    core = _largest_component(core)
    core = ndimage.binary_dilation(core, iterations=int(M["erode_vox"])) & head
    brain = fill_holes(core, si_axis)
    brain = ndimage.binary_closing(brain, iterations=int(M["close_vox"])) & head
    brain = fill_holes(brain, si_axis)

    voxel_ml = float(np.prod(vox)) / 1000.0
    return {"mask": brain, "head": head, "normalised": norm, "air_level": air,
            "volume_ml": round(float(brain.sum()) * voxel_ml, 1), "head_volume_ml": round(float(head.sum()) * voxel_ml, 1),
            "n_voxels": int(brain.sum())}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0
