"""Surface-based cortical regions (docs/SURFACE_PLAN.md, user 2026-11-17).

FreeSurfer surfaces resampled to fs_LR 32k carry two cortical parcellations defined on that mesh: Schaefer 2018
200 / 7 (CBIG) and the 1876-parcel icosahedral tessellation (Icosahedron1002) used by the Diedrichsen-lab
cortico-cerebellar connectivity models. Connectome Workbench maps each parcellation onto the cortical ribbon of the
participant's EPI grid (scripts/surface_labels.lsf); this module prepares the label files, combines the two
hemispheres, and exports parcel time courses in the models' own parcel order (CIFTI .ptseries.nii, parcels taken
from the model file), so that route B's BOLD-like signals can be used directly with the models. The functional data
are never resampled: parcels are pooled in native EPI space, as everywhere in route B.

Icosahedron numbering (checked against Nettekoven_2024_Fusion_L2.pdconn.nii, 2026-11-17): label k of the left
hemisphere is model parcel cortex_left_k (index k - 1); label k of the right hemisphere is model parcel 938 + k.
Region ids here follow the model order: left k -> id k, right k -> id 938 + k.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

YEO7 = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")


# ---------------------------------------------------------------- label files
def schaefer_from_dlabel(path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Schaefer dlabel on fs_LR 32k -> per-vertex labels for the left and right hemisphere (global ids, 0 = none,
    full 32492-vertex meshes) and {id: name}."""
    import nibabel as nib
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)[0].astype(int)
    bm = img.header.get_axis(1)
    out = {}
    for name, sl, model in bm.iter_structures():
        full = np.zeros(model.nvertices[name], int)
        full[model.vertex] = data[sl]
        out["L" if "LEFT" in name else "R"] = full
    names = {int(k): v[0] for k, v in img.header.get_axis(0).label[0].items() if int(k) > 0}
    return out["L"], out["R"], names


def network_of(name: str) -> str:
    """'7Networks_LH_Vis_1' -> 'Vis'."""
    parts = name.split("_")
    return parts[2] if len(parts) > 2 else ""


def ico_table(ico_l: np.ndarray, ico_r: np.ndarray, sch_l: np.ndarray, sch_r: np.ndarray, sch_names: dict,
              n_left: int = 938) -> tuple[dict, np.ndarray, np.ndarray]:
    """Region table for the icosahedral parcels in the model order, each with the Yeo-7 network that most of its
    vertices carry in the Schaefer parcellation on the same mesh (vertices outside Schaefer, e.g. medial wall,
    are ignored; a parcel with none gets network ''). Returns (table, left labels as ids, right labels as ids)."""
    table = {}
    gl = np.asarray(ico_l, int).copy()
    gr = np.where(np.asarray(ico_r, int) > 0, np.asarray(ico_r, int) + n_left, 0)
    for hemi, g, sch in (("LH", gl, sch_l), ("RH", gr, sch_r)):
        for pid in np.unique(g[g > 0]):
            s = sch[g == pid]
            nets = [network_of(sch_names[v]) for v in s if v > 0 and v in sch_names]
            net = max(set(nets), key=nets.count) if nets else ""
            side = "left" if hemi == "LH" else "right"
            table[int(pid)] = {"name": f"cortex_{side}_{int(pid):02d}", "hemisphere": hemi, "network": net}
    return dict(sorted(table.items())), gl, gr


def write_label_gii(values: np.ndarray, hemi: str, names: dict, path) -> None:
    """GIFTI label file (one map) with structure metadata, for wb_command -label-to-volume-mapping."""
    import nibabel as nib
    from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable, GiftiMetaData
    lt = GiftiLabelTable()
    rng = np.random.default_rng(0)
    for key in [0] + sorted(int(v) for v in np.unique(values) if v > 0):
        lab = GiftiLabel(key=int(key), red=0.0, green=0.0, blue=0.0, alpha=0.0) if key == 0 else \
            GiftiLabel(key=int(key), red=float(rng.random()), green=float(rng.random()), blue=float(rng.random()), alpha=1.0)
        lab.label = "???" if key == 0 else names.get(int(key), f"region_{key}")
        lt.labels.append(lab)
    meta = GiftiMetaData({"AnatomicalStructurePrimary": "CortexLeft" if hemi == "L" else "CortexRight"})
    da = GiftiDataArray(np.asarray(values, np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    nib.save(GiftiImage(darrays=[da], labeltable=lt, meta=meta), str(path))


def prepare(resources, out_dir) -> dict:
    """Write schaefer200.{L,R}.label.gii, ico1002.{L,R}.label.gii (ids as in the region tables) and
    atlas_table_ico1002_7.json into out_dir, from the files fetched by scripts/fetch_fs32k_resources.sh."""
    import nibabel as nib
    R, O = Path(resources), Path(out_dir)
    O.mkdir(parents=True, exist_ok=True)
    sl, sr, names = schaefer_from_dlabel(R / "Schaefer2018_200Parcels_7Networks_order.dlabel.nii")
    il = np.asarray(nib.load(str(R / "Icosahedron1002.L.label.gii")).darrays[0].data, int)
    ir = np.asarray(nib.load(str(R / "Icosahedron1002.R.label.gii")).darrays[0].data, int)
    for name, arr in (("Schaefer L", sl), ("Schaefer R", sr), ("Icosahedron L", il), ("Icosahedron R", ir)):
        if arr.size != 32492:
            raise ValueError(f"{name} has {arr.size} vertices, expected 32492 (fs_LR 32k)")
    table, gl, gr = ico_table(il, ir, sl, sr, names)
    write_label_gii(sl, "L", names, O / "schaefer200.L.label.gii")
    write_label_gii(sr, "R", names, O / "schaefer200.R.label.gii")
    inames = {k: v["name"] for k, v in table.items()}
    write_label_gii(gl, "L", inames, O / "ico1002.L.label.gii")
    write_label_gii(gr, "R", inames, O / "ico1002.R.label.gii")
    (O / "atlas_table_ico1002_7.json").write_text(json.dumps({str(k): v for k, v in table.items()}, indent=1))
    nets = [v["network"] for v in table.values()]
    return {"n_schaefer": len(names), "n_ico": len(table), "ico_without_network": int(sum(n == "" for n in nets)),
            "ico_by_network": {n: int(sum(x == n for x in nets)) for n in YEO7}}


def combine_hemis(left_path, right_path, out_path, ref_path=None) -> dict:
    """Combine the two hemispheres' ribbon-mapped label volumes (EPI grid). A voxel claimed by both hemispheres
    (midline) is left unlabelled. Returns counts."""
    import nibabel as nib
    a = nib.load(str(left_path))
    L = np.rint(np.asarray(a.dataobj)).astype(int)
    R = np.rint(np.asarray(nib.load(str(right_path)).dataobj)).astype(int)
    if L.shape != R.shape:
        raise ValueError(f"hemisphere volumes differ in grid: {L.shape} vs {R.shape}")
    both = (L > 0) & (R > 0)
    out = np.where(both, 0, np.maximum(L, R)).astype(np.int32)
    ref = nib.load(str(ref_path)) if ref_path else a
    img = nib.Nifti1Image(out, ref.affine, ref.header)
    img.set_data_dtype(np.int32)
    nib.save(img, str(out_path))
    return {"voxels_left": int((L > 0).sum()), "voxels_right": int((R > 0).sum()), "voxels_both_dropped": int(both.sum()),
            "n_labels": int(np.unique(out[out > 0]).size)}


def label_overlap(a: np.ndarray, b: np.ndarray, ids) -> dict:
    """Per-region Dice between two label volumes on the same grid (volumetric vs surface-based parcels)."""
    d = []
    for i in ids:
        x, y = a == i, b == i
        s = x.sum() + y.sum()
        if s:
            d.append(2 * (x & y).sum() / s)
    d = np.array(d)
    return {"n": int(d.size), "dice_median": round(float(np.median(d)), 3) if d.size else None,
            "dice_q10_q90": [round(float(np.quantile(d, 0.1)), 3), round(float(np.quantile(d, 0.9)), 3)] if d.size else None}


# ---------------------------------------------------------------- parcel time courses for the connectivity models
def parcel_timecourses(frontend_dir, anat_dir, atlas: str, cfg: dict) -> dict:
    """Route B's region signals, split and leftover noise for every labelled parcel (no usability threshold, no
    integration measures): BOLD-like b (1/s), S0-like a (fraction), noise variance, voxel count, frames used."""
    import nibabel as nib

    from . import io, regions
    fe, A = Path(frontend_dir), Path(anat_dir)
    mask3d = np.asarray(nib.load(str(fe / "analysis_mask.nii.gz")).dataobj) > 0
    lab3 = np.rint(np.asarray(nib.load(str(A / f"labels_{atlas}.nii.gz")).dataobj)).astype(int)
    if lab3.shape != mask3d.shape:
        raise ValueError(f"labels grid {lab3.shape} differs from the front end {mask3d.shape}")
    done = json.loads((fe / "done.json").read_text())
    fs = json.loads((fe / "summary.json").read_text())
    files = [done["inputs"][f"echo{k}"]["path"] for k in range(1, 5) if f"echo{k}" in done.get("inputs", {})]
    S = io.load_masked_echoes(files, mask3d)
    vol = lambda n: np.asarray(nib.load(str(fe / f"{n}.nii.gz")).dataobj, np.float64)[mask3d]
    mu, sigma, use = vol("mu"), vol("sigma"), vol("usable") > 0.5
    if mu.ndim == 1:
        mu = mu[:, None]
    if use.ndim == 1:
        use = use[:, None]
    Q = np.load(fe / "nuisance_Q.npy") if (fe / "nuisance_Q.npy").exists() else None
    labels = lab3[mask3d]
    P = int(labels.max())
    rs = regions.region_signals(S, labels, mu, sigma, fs["echo_noise_ratios_used"], use, P)
    sp = regions.split(rs["ybar"], rs["wsum"], fs["te_ms"], Q)
    nv = regions.leftover_noise(sp["r"], 0 if Q is None else Q.shape[1])   # as regions.analyze
    return {"b": sp["b"], "a": sp["a"], "noise_var": nv, "n_voxels": rs["n_voxels"], "tr": float(cfg["acquisition"]["tr_s"]),
            "te_ms": fs["te_ms"]}


def write_ptseries(values: np.ndarray, parcels_axis, tr: float, path) -> None:
    """(parcels x frames) values -> CIFTI .ptseries.nii with the given ParcelsAxis (frames x parcels on disk)."""
    import nibabel as nib
    series = nib.cifti2.SeriesAxis(start=0.0, step=float(tr), size=values.shape[1], unit="SECOND")
    hdr = nib.cifti2.Cifti2Header.from_axes((series, parcels_axis))
    nib.save(nib.cifti2.Cifti2Image(np.asarray(values.T, np.float32), hdr), str(path))


def export_for_models(frontend_dir, anat_dir, cfg: dict, pdconn, out_dir, atlas: str = "ico1002_7") -> dict:
    """Parcel time courses in the connectivity models' parcel order and parcel definitions (taken from the model
    file): bold.ptseries.nii (BOLD-like dR2*, 1/s), s0.ptseries.nii (S0-like, fraction), parcels.tsv (voxels,
    noise variance; parcels without voxels are NaN)."""
    import nibabel as nib

    from . import io
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pax = nib.load(str(pdconn)).header.get_axis(1)
    n = len(pax)
    tc = parcel_timecourses(frontend_dir, anat_dir, atlas, cfg)
    T = tc["b"].shape[1]
    B = np.full((n, T), np.nan)
    A = np.full((n, T), np.nan)
    P = min(n, tc["b"].shape[0])
    has = tc["n_voxels"][:P] > 0
    B[:P][has] = tc["b"][:P][has]
    A[:P][has] = tc["a"][:P][has]
    write_ptseries(B, pax, tc["tr"], out / "bold.ptseries.nii")
    write_ptseries(A, pax, tc["tr"], out / "s0.ptseries.nii")
    nvox = np.zeros(n, int)
    nvox[:P] = tc["n_voxels"][:P]
    nvar = np.full(n, np.nan)
    nvar[:P][has] = tc["noise_var"][:P][has]
    io.write_tsv(out / "parcels.tsv", {"parcel": list(pax.name), "n_voxels": list(nvox), "noise_var": [round(float(v), 6) for v in nvar]})
    summ = {"atlas": atlas, "n_parcels": n, "n_with_voxels": int((nvox > 0).sum()), "frames": int(T), "tr_s": tc["tr"],
            "voxels_per_parcel_median": float(np.median(nvox[nvox > 0])) if (nvox > 0).any() else None,
            "pdconn": str(pdconn), "units": {"bold": "dR2* (1/s)", "s0": "dlnS0 (fraction)"}, **io.provenance(cfg, {})}
    io.write_json(out / "export.json", summ)
    return summ
