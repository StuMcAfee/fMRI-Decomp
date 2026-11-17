"""Design A, the lesion-network analysis (docs/SURFACE_PLAN.md section 3; user 2026-11-17).

For each patient, the cortical territories connected to the damaged cerebellum, from a healthy cortico-cerebellar
connectivity model (Nettekoven et al. 2024, `Nettekoven_2024_Fusion_L2.pdconn.nii`: weights W from 1876 cortical
parcels to 6918 SUIT 3 mm cerebellar voxels). No cerebellar BOLD is used.

  D_p = sum_v L_v W+_vp / sum_v W+_vp

L_v: damaged fraction of model voxel v (0-1), from a lesion map in SUIT space (or dose / prescribed dose, clipped to
0-1, for the dose-weighted score); W+: the positive part of the weights (user decision; |W| as sensitivity). D_p is
the share of cortical parcel p's positive cerebellar connectivity that lands in damaged tissue. Schaefer 200/7
parcels take the vertex-weighted mean over the icosahedral parcels they overlap on fs_LR 32k; networks the mean over
their parcels. Participant maps are data: everything is written under outputs/, never into the repository.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_model(pdconn) -> dict:
    """Weights (cerebellar voxels x cortical parcels), the voxels' grid indices and affine, the parcel axis."""
    import nibabel as nib
    img = nib.load(str(pdconn))
    vox_ax, par_ax = img.header.get_axis(0), img.header.get_axis(1)
    if type(vox_ax).__name__ != "BrainModelAxis" or type(par_ax).__name__ != "ParcelsAxis":
        raise ValueError(f"{pdconn}: expected cerebellar voxels x cortical parcels (BrainModelAxis x ParcelsAxis)")
    return {"W": np.asarray(img.dataobj, np.float64), "ijk": np.asarray(vox_ax.voxel), "affine": np.asarray(vox_ax.affine),
            "shape": tuple(vox_ax.volume_shape), "parcels": par_ax, "names": [str(n) for n in par_ax.name]}


def damaged_fraction(lesion_img, ijk: np.ndarray, affine: np.ndarray, voxel_mm: float = 3.0, sub: int = 3,
                     scale: float | None = None) -> dict:
    """Damaged fraction of each model voxel: the lesion (or dose / scale) map sampled at sub^3 points inside each
    3 mm voxel (nearest neighbour), averaged, clipped to 0-1. Also how many model voxels fall outside the map."""
    from scipy import ndimage
    data = np.asarray(lesion_img.dataobj, np.float64)
    if data.ndim > 3:
        data = data[..., 0]
    if scale:
        data = data / float(scale)
    data = np.clip(np.nan_to_num(data), 0.0, 1.0)
    off = (np.arange(sub) + 0.5) / sub - 0.5                          # sub-voxel offsets in units of the model voxel
    grid = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    centres = (affine[:3, :3] @ ijk.T).T + affine[:3, 3]             # model voxel centres, mm
    step = affine[:3, :3] / 1.0
    pts = centres[:, None, :] + grid[None] @ step.T                    # (V, sub^3, 3) mm
    inv = np.linalg.inv(lesion_img.affine)
    vox = (pts.reshape(-1, 3) @ inv[:3, :3].T) + inv[:3, 3]
    inside = np.all((vox > -0.5) & (vox < np.array(data.shape) - 0.5), axis=1)
    vals = ndimage.map_coordinates(data, vox.T, order=0, mode="constant", cval=0.0)
    frac = vals.reshape(len(ijk), -1).mean(1)
    out_of_map = (~inside.reshape(len(ijk), -1)).all(1)
    return {"fraction": frac, "n_outside_map": int(out_of_map.sum()), "n_damaged_voxels": int((frac > 0).sum()),
            "damaged_volume_ml": float(frac.sum() * voxel_mm ** 3 / 1000.0)}


def disconnection(W: np.ndarray, frac: np.ndarray, weights: str = "positive") -> np.ndarray:
    """D_p for every cortical parcel (W: voxels x parcels; frac: damaged fraction per voxel)."""
    Wp = np.clip(W, 0, None) if weights == "positive" else np.abs(W)
    tot = Wp.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, frac @ Wp / tot, np.nan)


def to_schaefer(d_ico: np.ndarray, ico_l: np.ndarray, ico_r: np.ndarray, sch_l: np.ndarray, sch_r: np.ndarray,
                n_left: int = 938) -> dict:
    """Vertex-weighted mean of the icosahedral parcels' D within each Schaefer parcel (fs_LR 32k vertices; vertices
    without an icosahedral parcel, e.g. medial wall, are ignored). d_ico in model order (id - 1)."""
    out = {}
    for ico, sch, offset in ((np.asarray(ico_l, int), np.asarray(sch_l, int), 0), (np.asarray(ico_r, int), np.asarray(sch_r, int), n_left)):
        ok = (ico > 0) & (sch > 0)
        vals = d_ico[ico[ok] + offset - 1]
        for s in np.unique(sch[ok]):
            v = vals[sch[ok] == s]
            v = v[np.isfinite(v)]
            out[int(s)] = float(v.mean()) if v.size else float("nan")
    return dict(sorted(out.items()))


def run(lesion_path, pdconn, resources, out_dir, scale: float | None = None, label: str = "lesion") -> dict:
    """One patient's disconnection maps: icosahedral parcels (TSV and CIFTI .pscalar.nii for Workbench), Schaefer
    200/7 parcels and networks (TSV / JSON), with the positive-weight (primary) and |W| (sensitivity) scores."""
    import nibabel as nib

    from . import io, surface
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = load_model(pdconn)
    les = nib.load(str(lesion_path))
    df = damaged_fraction(les, m["ijk"], m["affine"], scale=scale)
    if df["n_outside_map"] > 0.05 * len(m["ijk"]):
        raise ValueError(f"{df['n_outside_map']} of {len(m['ijk'])} model voxels fall outside the {label} map: is it in SUIT space?")
    d_pos = disconnection(m["W"], df["fraction"], "positive")
    d_abs = disconnection(m["W"], df["fraction"], "absolute")
    io.write_tsv(out / f"{label}_disconnection_ico1002.tsv",
                 {"parcel": m["names"], "D_positive": [round(float(x), 6) for x in d_pos], "D_absolute": [round(float(x), 6) for x in d_abs]})
    scal = nib.cifti2.ScalarAxis([f"{label}_D_positive", f"{label}_D_absolute"])
    hdr = nib.cifti2.Cifti2Header.from_axes((scal, m["parcels"]))
    nib.save(nib.cifti2.Cifti2Image(np.vstack([d_pos, d_abs]).astype(np.float32), hdr), str(out / f"{label}_disconnection.pscalar.nii"))
    R = Path(resources)
    sl, sr, names = surface.schaefer_from_dlabel(R / "Schaefer2018_200Parcels_7Networks_order.dlabel.nii")
    il = np.asarray(nib.load(str(R / "Icosahedron1002.L.label.gii")).darrays[0].data, int)
    ir = np.asarray(nib.load(str(R / "Icosahedron1002.R.label.gii")).darrays[0].data, int)
    s_pos, s_abs = to_schaefer(d_pos, il, ir, sl, sr), to_schaefer(d_abs, il, ir, sl, sr)
    ids = sorted(s_pos)
    io.write_tsv(out / f"{label}_disconnection_schaefer200_7.tsv",
                 {"id": ids, "name": [names[i] for i in ids], "network": [surface.network_of(names[i]) for i in ids],
                  "D_positive": [round(s_pos[i], 6) for i in ids], "D_absolute": [round(s_abs[i], 6) for i in ids]})
    net = {}
    for n in surface.YEO7:
        for hemi, tag in (("LH", "_LH_"), ("RH", "_RH_"), ("both", "")):
            v = [s_pos[i] for i in ids if surface.network_of(names[i]) == n and tag in names[i] and np.isfinite(s_pos[i])]
            net.setdefault(n, {})[hemi] = round(float(np.mean(v)), 6) if v else None
    summ = {"label": label, "lesion_file": str(lesion_path), "scale": scale, "model": str(pdconn),
            "damaged_model_voxels": df["n_damaged_voxels"], "damaged_volume_ml_on_model_grid": round(df["damaged_volume_ml"], 2),
            "model_voxels_outside_map": df["n_outside_map"], "D_positive_cortex_mean": round(float(np.nanmean(d_pos)), 6),
            "D_positive_by_network": net, **io.provenance({}, {})}
    io.write_json(out / f"{label}_disconnection.json", summ)
    return summ
