"""Route B step 3 (docs/ROUTE_B_PLAN.md): regions in each participant's own EPI space.

The registrations are done by FSL in `scripts/anatomy_run.lsf` (bet, fast, epi_reg, flirt, fnirt, invwarp,
applywarp); this module finds the T1, reads the atlas label tables, and turns the atlas images resampled into EPI
space into the region labels route B uses:

- cortex: Schaefer 2018 parcels (200 / 7 networks primary, decision D2), voxels inside the analysis mask, with
  grey-matter probability >= regions.gm_min (FSL FAST on the T1, brought into EPI space), and not CSF-dominated or
  dropout in the data-only tissue classes;
- secondary (decision D1): left and right thalamus and striatum (caudate + putamen + accumbens) from the
  Harvard-Oxford subcortical atlas shipped with FSL, inside the analysis mask, not CSF-dominated or dropout (FAST
  grey-matter probability is unreliable in deep nuclei, so it is not applied there);
- cerebellum and brainstem are never regions (user decision 2026-11-08).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

SUBCORTICAL = {"thalamus": ("Thalamus",), "striatum": ("Caudate", "Putamen", "Accumbens")}


def find_t1(run_name: str, list_file=None, t1_root=None) -> str | None:
    """T1 for a run named sub-<subject>_acq-<YYYYMMDDhhmmss>_ser-<n>: first a line of list_file (quotes allowed)
    with /<subject>/ in its path and the acquisition time in its name, then the same subject alone; then a search
    under t1_root/<subject>/ for *T1*MPRAGE*.nii* (same preference)."""
    m = re.match(r"sub-(\d+)(?:_acq-(\d+))?", run_name)
    if not m:
        return None
    subj, acq = m.group(1), m.group(2)
    cands = []
    if list_file is not None and Path(list_file).exists():
        for line in Path(list_file).read_text().splitlines():
            s = line.strip().strip('"').strip("'").strip()
            if s and not s.startswith("#") and f"/{subj}/" in s:
                cands.append(s)
    if not cands and t1_root is not None and (Path(t1_root) / subj).is_dir():
        cands = [str(p) for p in sorted((Path(t1_root) / subj).rglob("*T1*MPRAGE*.nii*"))]
    if not cands:
        return None
    same = [c for c in cands if acq and acq in Path(c).name]
    return (same or cands)[0]


def schaefer_table(lut_path) -> dict:
    """index -> {"name", "hemisphere", "network"} from a Schaefer freeview label table."""
    out = {}
    for line in Path(lut_path).read_text().splitlines():
        f = line.split()
        if len(f) < 2 or not f[0].isdigit():
            continue
        parts = f[1].split("_")                                       # <N>Networks_<LH|RH>_<network>_..._<n>
        out[int(f[0])] = {"name": f[1], "hemisphere": parts[1] if len(parts) > 1 else "",
                          "network": parts[2] if len(parts) > 2 else ""}
    return out


def ho_subcortical_values(xml_path) -> dict:
    """Image value -> label name for the Harvard-Oxford subcortical atlas (FSL atlas XML: <label index="i">, image
    value = i + 1)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(str(xml_path)).getroot()
    return {int(lab.get("index")) + 1: (lab.text or "").strip() for lab in root.iter("label")}


def subcortical_groups(values: dict) -> dict:
    """(hemisphere, group) -> list of image values, e.g. ("L", "striatum") -> [5, 6, 11]."""
    out = {}
    for v, name in values.items():
        hemi = "L" if name.lower().startswith("left") else ("R" if name.lower().startswith("right") else None)
        if hemi is None:
            continue
        for group, keys in SUBCORTICAL.items():
            if any(k.lower() in name.lower() for k in keys):
                out.setdefault((hemi, group), []).append(v)
    return out


def build_labels(cortex_epi: np.ndarray, sub_epi: np.ndarray | None, gm_epi: np.ndarray, mask3d: np.ndarray,
                 classes3d: np.ndarray | None, table: dict, groups: dict | None, gm_min: float = 0.3,
                 min_voxels: int = 20, secondary: bool = True) -> dict:
    """Region labels (3D, 0 = none) and the region table. Cortex ids 1..N follow the Schaefer index; secondary
    regions follow, in the order L/R thalamus, L/R striatum."""
    ok = mask3d.copy()
    if classes3d is not None:
        ok &= ~np.isin(classes3d, (3, 5))                              # data-only classes: CSF-dominated, few echoes
    lab = np.zeros(mask3d.shape, np.int32)
    rows = []
    cx = np.rint(cortex_epi).astype(int)
    keep_c = ok & (cx > 0) & (np.nan_to_num(gm_epi) >= gm_min)
    lab[keep_c] = cx[keep_c]
    for i, info in sorted(table.items()):
        n = int((lab == i).sum())
        rows.append({"id": i, "name": info["name"], "kind": "cortex", "hemisphere": info["hemisphere"],
                     "network": info["network"], "n_voxels": n, "usable": n >= min_voxels})
    nxt = max(table) + 1 if table else 1
    if secondary and sub_epi is not None and groups:
        sx = np.rint(sub_epi).astype(int)
        for group in ("thalamus", "striatum"):
            for hemi in ("L", "R"):
                vals = groups.get((hemi, group), [])
                sel = ok & np.isin(sx, vals) & (lab == 0)
                lab[sel] = nxt
                n = int(sel.sum())
                rows.append({"id": nxt, "name": f"{hemi}_{group}", "kind": "secondary", "hemisphere": "LH" if hemi == "L" else "RH",
                             "network": group.capitalize(), "n_voxels": n, "usable": n >= min_voxels})
                nxt += 1
    return {"labels": lab, "regions": rows}


def run_dir(anat_dir, frontend_dir, cfg: dict, tissue_dir=None, atlas: str = "200_7") -> dict:
    """Read the FSL outputs of scripts/anatomy_run.lsf in anat_dir and write labels_<atlas>.nii.gz,
    regions_<atlas>.tsv / .json and a QC figure."""
    import nibabel as nib

    from . import io
    A, fe = Path(anat_dir), Path(frontend_dir)
    G = cfg.get("regions", {})
    mimg = nib.load(str(fe / "analysis_mask.nii.gz"))
    mask3d = np.asarray(mimg.dataobj) > 0
    vol = lambda name: np.asarray(nib.load(str(A / name)).dataobj, np.float64)
    cortex = vol(f"schaefer_{atlas}_epi.nii.gz")
    sub = vol("ho_sub_epi.nii.gz") if (A / "ho_sub_epi.nii.gz").exists() else None
    gm = vol("gm_epi.nii.gz")
    for name, arr in (("cortex atlas", cortex), ("grey matter", gm)):
        if arr.shape != mask3d.shape:
            raise ValueError(f"{name} in {A} has grid {arr.shape}, the front end {mask3d.shape}")
    classes3d = None
    if tissue_dir is not None and (Path(tissue_dir) / "classes.npy").exists():
        cl = np.load(Path(tissue_dir) / "classes.npy")
        if cl.size == mask3d.sum():
            classes3d = np.zeros(mask3d.shape, np.int8)
            classes3d[mask3d] = cl
    repo = Path(__file__).resolve().parents[2]
    n_par, n_net = atlas.split("_")
    table = schaefer_table(repo / "atlases" / "schaefer2018" / f"Schaefer2018_{n_par}Parcels_{n_net}Networks_order.txt")
    groups = None
    xml = A / "HarvardOxford-Subcortical.xml"
    if xml.exists():
        groups = subcortical_groups(ho_subcortical_values(xml))
    res = build_labels(cortex, sub, gm, mask3d, classes3d, table, groups, gm_min=float(G.get("gm_min", 0.3)),
                       min_voxels=int(G.get("min_voxels", 20)), secondary=bool(G.get("secondary", True)))
    io.save_volume(res["labels"].astype(np.float32), mimg, A / f"labels_{atlas}.nii.gz")
    rows = res["regions"]
    io.write_tsv(A / f"regions_{atlas}.tsv", {k: [r[k] for r in rows] for k in rows[0]})
    summ = {"atlas": atlas, "n_regions": len(rows), "n_usable": sum(r["usable"] for r in rows),
            "cortex_voxels": int(sum(r["n_voxels"] for r in rows if r["kind"] == "cortex")),
            "median_cortex_region_voxels": float(np.median([r["n_voxels"] for r in rows if r["kind"] == "cortex"])),
            "secondary": [r for r in rows if r["kind"] == "secondary"], "gm_min": float(G.get("gm_min", 0.3)),
            "classes_used": classes3d is not None, "regions": rows, **io.provenance(cfg, {})}
    io.write_json(A / f"regions_{atlas}.json", summ)
    try:
        qc_figure(A, mimg, res["labels"], A / f"qc_{atlas}.png")
        if (A / "t1_mni.nii.gz").exists() and ((A / "mni_template.nii.gz").exists() or (A / "mni_template_brain.nii.gz").exists()):
            qc_mni(A, A / "qc_t1_to_mni.png")
        if all((A / f).exists() for f in ("epi2t1.nii.gz", "t1_wmseg.nii.gz", "t1.nii.gz")):
            qc_epi_in_t1(A, A / "qc_epi_to_t1.png")
    except Exception as exc:                                           # the labels matter more than the figures
        print(f"QC figure failed: {exc}")
    return summ


def qc_mni(A: Path, path) -> None:
    """The participant's T1 warped to MNI space with the template's brain outline (T1-to-template check)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from scipy import ndimage
    t1 = np.asarray(nib.load(str(A / "t1_mni.nii.gz")).dataobj, np.float64)
    if (A / "mni_template_brain.nii.gz").exists():                      # the template's brain: outline at the brain surface
        brain = np.asarray(nib.load(str(A / "mni_template_brain.nii.gz")).dataobj, np.float64) > 0
    else:
        tpl = np.asarray(nib.load(str(A / "mni_template.nii.gz")).dataobj, np.float64)
        brain = tpl > np.percentile(tpl[tpl > 0], 30) if (tpl > 0).any() else tpl > 0
    fig, axs = plt.subplots(1, 6, figsize=(15.6, 3))
    cuts = [("z", int(t1.shape[2] * f)) for f in (0.35, 0.5, 0.65)] + [("x", t1.shape[0] // 3), ("x", t1.shape[0] // 2), ("y", t1.shape[1] // 2)]
    lim = np.percentile(t1[t1 > 0], 99) if (t1 > 0).any() else 1
    for ax, (axis, i) in zip(axs, cuts):
        sl = {"x": (t1[i], brain[i]), "y": (t1[:, i], brain[:, i]), "z": (t1[:, :, i], brain[:, :, i])}[axis]
        ax.imshow(sl[0].T, origin="lower", cmap="gray", vmin=0, vmax=lim)
        b = ndimage.binary_fill_holes(sl[1]).astype(float).T
        if (b > 0.5).any():
            ax.contour(b, levels=[0.5], colors="orange", linewidths=0.7)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{axis} = {i}", fontsize=8)
    fig.suptitle(f"{A.name}: T1 in MNI space, template brain surface in orange (check by eye)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def qc_figure(A: Path, mimg, labels: np.ndarray, path) -> None:
    """The region labels on the EPI mean (native EPI grid), six axial slices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    epi = np.asarray(nib.load(str(A / "epi_mean.nii.gz")).dataobj, np.float64)
    zs = np.linspace(int(labels.shape[2] * 0.25), int(labels.shape[2] * 0.85), 6).astype(int)
    fig, axs = plt.subplots(1, len(zs), figsize=(2.6 * len(zs), 3.2))
    lim = np.percentile(epi[epi > 0], 99) if (epi > 0).any() else 1
    cmap = plt.get_cmap("tab20")
    for ax, z in zip(axs, zs):
        ax.imshow(epi[:, :, z].T, origin="lower", cmap="gray", vmin=0, vmax=lim)
        L = labels[:, :, z].T.astype(float)
        ax.imshow(np.ma.masked_where(L == 0, (L % 20) / 20.0), origin="lower", cmap=cmap, vmin=0, vmax=1, alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slice {z}", fontsize=8)
    fig.suptitle(f"{A.name}: regions on the EPI (cortex parcels, thalamus, striatum)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _cuts(shape):
    return [("z", int(shape[2] * f)) for f in (0.35, 0.5, 0.65)] + [("x", int(shape[0] * 0.35)), ("x", shape[0] // 2), ("y", shape[1] // 2)]


def _slice(vol, axis, i):
    return {"x": vol[i], "y": vol[:, i], "z": vol[:, :, i]}[axis]


def qc_epi_in_t1(A: Path, path) -> None:
    """Registration check at T1 resolution: the EPI mean resampled into T1 space by epi_reg (top) and the T1
    (bottom), both with the T1 white-matter boundary (red). If the EPI is well aligned, its white matter fills
    the red outline and the cortical ribbon sits just outside it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    epi = np.asarray(nib.load(str(A / "epi2t1.nii.gz")).dataobj, np.float64)
    wm = np.asarray(nib.load(str(A / "t1_wmseg.nii.gz")).dataobj, np.float64)
    t1 = np.asarray(nib.load(str(A / "t1.nii.gz")).dataobj, np.float64)
    cuts = _cuts(epi.shape)
    fig, axs = plt.subplots(2, len(cuts), figsize=(2.7 * len(cuts), 5.8))
    for row, img, name in ((0, epi, "EPI in T1 space"), (1, t1, "T1")):
        pos = img[img > 0]
        lim = np.percentile(pos, 99) if pos.size else 1
        for ax, (axis, i) in zip(axs[row], cuts):
            ax.imshow(_slice(img, axis, i).T, origin="lower", cmap="gray", vmin=0, vmax=lim)
            w = _slice(wm, axis, i).T
            if (w > 0.5).any() and (w <= 0.5).any():
                ax.contour(w, levels=[0.5], colors="red", linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name}, {axis} = {i}", fontsize=7)
    fig.suptitle(f"{A.name}: EPI-to-T1 registration, T1 white-matter boundary in red (check by eye)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
