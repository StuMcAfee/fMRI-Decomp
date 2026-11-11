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
                 min_voxels: int = 20, secondary: bool = True, csf_epi: np.ndarray | None = None,
                 secondary_csf_max: float | None = None, cortex_csf_max: float | None = None) -> dict:
    """Region labels (3D, 0 = none) and the region table. Cortex ids 1..N follow the Schaefer index; secondary
    regions follow, in the order L/R thalamus, L/R striatum. With csf_epi and secondary_csf_max, thalamus and
    striatum voxels with CSF probability above it are dropped (user, 2026-11-10)."""
    ok = mask3d.copy()
    if classes3d is not None:
        ok &= ~np.isin(classes3d, (3, 5))                              # data-only classes: CSF-dominated, few echoes
    lab = np.zeros(mask3d.shape, np.int32)
    rows = []
    cx = np.rint(cortex_epi).astype(int)
    keep_c = ok & (cx > 0) & (np.nan_to_num(gm_epi) >= gm_min)
    before = np.where(keep_c, cx, 0)
    if csf_epi is not None and cortex_csf_max is not None:            # primary since 2026-11-11 (user)
        keep_c &= ~(np.nan_to_num(csf_epi) > float(cortex_csf_max))
    lab[keep_c] = cx[keep_c]
    for i, info in sorted(table.items()):
        n = int((lab == i).sum())
        rows.append({"id": i, "name": info["name"], "kind": "cortex", "hemisphere": info["hemisphere"],
                     "network": info["network"], "n_voxels": n, "usable": n >= min_voxels,
                     "n_voxels_before_csf_exclusion": int((before == i).sum())})
    nxt = max(table) + 1 if table else 1
    if secondary and sub_epi is not None and groups:
        sx = np.rint(sub_epi).astype(int)
        for group in ("thalamus", "striatum"):
            for hemi in ("L", "R"):
                vals = groups.get((hemi, group), [])
                sel = ok & np.isin(sx, vals) & (lab == 0)
                n_before = int(sel.sum())
                if csf_epi is not None and secondary_csf_max is not None:
                    sel &= ~(np.nan_to_num(csf_epi) > float(secondary_csf_max))
                lab[sel] = nxt
                n = int(sel.sum())
                rows.append({"id": nxt, "name": f"{hemi}_{group}", "kind": "secondary", "hemisphere": "LH" if hemi == "L" else "RH",
                             "network": group.capitalize(), "n_voxels": n, "usable": n >= min_voxels,
                             "n_voxels_before_csf_exclusion": n_before})
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
    csf_max = G.get("secondary_csf_max")
    ccsf_max = G.get("cortex_csf_max")
    csf = vol("csf_epi.nii.gz") if (A / "csf_epi.nii.gz").exists() else None
    if (csf_max is not None or ccsf_max is not None) and csf is None:
        raise FileNotFoundError(f"a CSF exclusion is set (regions.secondary_csf_max / cortex_csf_max) but {A / 'csf_epi.nii.gz'} "
                                "is missing (rerun scripts/anatomy_run.lsf)")
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
    kw = dict(gm_min=float(G.get("gm_min", 0.3)), min_voxels=int(G.get("min_voxels", 20)),
              secondary=bool(G.get("secondary", True)), csf_epi=csf,
              secondary_csf_max=None if csf_max is None else float(csf_max))
    res = build_labels(cortex, sub, gm, mask3d, classes3d, table, groups,
                       cortex_csf_max=None if ccsf_max is None else float(ccsf_max), **kw)
    io.save_volume(res["labels"].astype(np.float32), mimg, A / f"labels_{atlas}.nii.gz")
    rows = res["regions"]
    csf_cov = None
    if ccsf_max is not None:
        # sensitivity labels with every cortical voxel (deep regions unchanged) and the participant's cortical CSF
        # fraction (covariate, user 2026-11-11)
        res_all = build_labels(cortex, sub, gm, mask3d, classes3d, table, groups, cortex_csf_max=None, **kw)
        io.save_volume(res_all["labels"].astype(np.float32), mimg, A / f"labels_{atlas}_allvox.nii.gz")
        io.write_json(A / f"regions_{atlas}_allvox.json", {"atlas": atlas, "regions": res_all["regions"]})
        cmask = np.isin(res_all["labels"], list(table))
        csf_cov = {"cortex_csf_mean_allvox": float(np.nanmean(csf[cmask])) if cmask.any() else None,
                   "cortex_voxels_allvox": int(cmask.sum()),
                   "cortex_voxels_removed": int(cmask.sum() - np.isin(res["labels"], list(table)).sum())}
    io.write_tsv(A / f"regions_{atlas}.tsv", {k: [r[k] for r in rows] for k in rows[0]})
    summ = {"atlas": atlas, "n_regions": len(rows), "n_usable": sum(r["usable"] for r in rows),
            "cortex_voxels": int(sum(r["n_voxels"] for r in rows if r["kind"] == "cortex")),
            "median_cortex_region_voxels": float(np.median([r["n_voxels"] for r in rows if r["kind"] == "cortex"])),
            "secondary": [r for r in rows if r["kind"] == "secondary"], "gm_min": float(G.get("gm_min", 0.3)),
            "secondary_csf_max": csf_max, "cortex_csf_max": ccsf_max, "cortex_csf_covariate": csf_cov,
            "classes_used": classes3d is not None, "regions": rows, **io.provenance(cfg, {})}
    io.write_json(A / f"regions_{atlas}.json", summ)
    try:
        qc_figure(A, mimg, res["labels"], A / f"qc_{atlas}.png")
        qc_subcortical(A, mimg, res["labels"], rows, A / "qc_subcortical.png")
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


def qc_subcortical(A: Path, mimg, labels: np.ndarray, table: list, path) -> None:
    """Thalamus and striatum labels (outlines; left red, right blue) on the EPI reference and on the CSF probability,
    axial and coronal slices through the regions (native EPI grid): checks left-right placement and CSF overlap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    sec = [r for r in table if r["kind"] == "secondary"]
    if not sec:
        return
    epi = np.asarray(nib.load(str(A / "epi_mean.nii.gz")).dataobj, np.float64)
    csf = np.asarray(nib.load(str(A / "csf_epi.nii.gz")).dataobj, np.float64) if (A / "csf_epi.nii.gz").exists() else None
    m = np.isin(labels, [r["id"] for r in sec])
    if not m.any():
        return
    zs = np.flatnonzero(m.any((0, 1))); ys = np.flatnonzero(m.any((0, 2)))
    cuts = [("z", int(np.percentile(zs, q))) for q in (25, 50, 75)] + [("y", int(np.percentile(ys, q))) for q in (25, 50, 75)]
    rows = [("EPI reference", epi, "gray")] + ([("CSF probability (white = CSF)", csf, "bone")] if csf is not None else [])
    fig, axs = plt.subplots(len(rows), len(cuts), figsize=(2.7 * len(cuts), 2.9 * len(rows)), squeeze=False)
    col = {"LH": "#e3342f", "RH": "#2a78d6"}
    for i, (title, vol, cm) in enumerate(rows):
        lim = np.percentile(vol[vol > 0], 99) if (vol > 0).any() else 1
        for j, (ax_, k) in enumerate(cuts):
            ax = axs[i, j]
            ax.imshow(_slice(vol, ax_, k).T, origin="lower", cmap=cm, vmin=0, vmax=lim if cm == "gray" else 1)
            for r in sec:
                L = (_slice(labels, ax_, k) == r["id"]).T.astype(float)
                if L.any():
                    ax.contour(L, levels=[0.5], colors=col[r["hemisphere"]], linewidths=1.0 if "thal" in r["name"] else 1.6,
                               linestyles="-" if "thal" in r["name"] else "--")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{title}: {ax_} = {k}", fontsize=7)
    side = {"R": "right", "L": "left"}.get(nib.aff2axcodes(mimg.affine)[0], "?")
    fig.suptitle(f"{A.name}: thalamus (solid) and striatum (dashed) as labelled; red = labelled left, blue = labelled right; "
                 f"screen right = subject {side} (from the image header)", x=0.01, ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _cuts(shape, cover=None):
    """Three axial slices in the upper brain (above the dropout zone) and three coronal slices (front, middle,
    back): a left-right shift shows in both. With cover (the EPI's footprint in T1 space) the slices are placed
    within the covered extent; otherwise at fixed fractions of the grid."""
    if cover is not None and cover.any():
        zs = np.flatnonzero(cover.any(axis=(0, 1)))
        ys = np.flatnonzero(cover.any(axis=(0, 2)))
        z0, z1, y0, y1 = zs.min(), zs.max(), ys.min(), ys.max()
        return ([("z", int(z0 + (z1 - z0) * f)) for f in (0.5, 0.65, 0.8)] +
                [("y", int(y0 + (y1 - y0) * f)) for f in (0.3, 0.5, 0.7)])
    return [("z", int(shape[2] * f)) for f in (0.55, 0.62, 0.7)] + [("y", int(shape[1] * f)) for f in (0.4, 0.5, 0.6)]


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
    cuts = _cuts(epi.shape, epi > 0)
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


# ============================================================================ registration check (R2, 2026-11-10)
EPI_REFS = ("echo1", "echo2", "echo3", "t2sw", "t2s")


def reference_images(frontend_dir, out_dir, names=EPI_REFS) -> list:
    """Candidate EPI reference images for epi_reg, on the EPI grid, zero outside the analysis mask: the temporal
    mean of echo 1, 2 or 3 (front end mu maps), and t2sw, the echoes' means combined with the T2*-weighted
    weights TE exp(-TE / T2*) (more grey / white contrast than echo 1)."""
    import nibabel as nib

    from . import io
    fe, out = Path(frontend_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mimg = nib.load(str(fe / "analysis_mask.nii.gz"))
    mu = np.nan_to_num(np.asarray(nib.load(str(fe / "mu.nii.gz")).dataobj, np.float64))
    te = np.asarray(json.loads((fe / "summary.json").read_text())["te_ms"], float)
    written = []
    for name in names:
        if name.startswith("echo"):
            k = int(name[4:]) - 1
            vol = mu[..., k]
        elif name == "t2s":
            # the static T2* map (fit to the run's mean decay) as the registration image (user, 2026-11-10): tissue
            # contrast (white matter shorter than grey matter, CSF longer) without coil shading or intensity
            # dependence; clipped so failed fits and CSF do not dominate
            t2 = np.nan_to_num(np.asarray(nib.load(str(fe / "t2s_ms.nii.gz")).dataobj, np.float64), nan=0.0)
            vol = np.clip(t2, 0.0, 150.0)
            vol = np.where(vol >= 5.0, vol, 0.0)
        elif name == "t2sw":
            t2 = np.nan_to_num(np.asarray(nib.load(str(fe / "t2s_ms.nii.gz")).dataobj, np.float64), nan=40.0)
            t2 = np.clip(t2, 5.0, 300.0)
            w = te[None, None, None, :] * np.exp(-te[None, None, None, :] / t2[..., None])
            vol = (w * mu).sum(-1) / np.maximum(w.sum(-1), 1e-12)
        else:
            raise ValueError(f"unknown EPI reference {name}")
        vol = np.where(np.asarray(mimg.dataobj) > 0, vol, 0.0)
        io.save_volume(vol, mimg, out / f"ref_{name}.nii.gz")
        written.append(name)
    return written


def shift_scan(epi_t1: np.ndarray, wm: np.ndarray, affine, max_mm: float = 8.0, ring_vox: int = 2) -> dict:
    """How far the T1 white-matter boundary sits from the EPI's white-matter contrast, per axis.

    EPI (already in T1 space) is darker in white matter than in the grey matter just outside it. For a shift s of
    the white-matter mask along one axis, contrast(s) = (mean EPI in a ring just outside the boundary - mean EPI in
    a ring just inside) / their average, over voxels the EPI covers. The shift with the highest contrast is where
    the boundary fits the EPI best; 0 means aligned. Returns per axis: the shifts (mm), the contrast curve, the best
    shift, and the side the EPI is displaced toward (axis codes from the affine)."""
    import nibabel as nib
    from scipy import ndimage
    E = np.asarray(epi_t1, np.float64)
    W = np.asarray(wm) > 0.5
    cov = E > 0
    out_ring = ndimage.binary_dilation(W, iterations=ring_vox) & ~W
    in_ring = W & ~ndimage.binary_erosion(W, iterations=ring_vox)
    zooms = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(0))
    codes = nib.aff2axcodes(affine)
    opposite = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}
    res = {}
    for a in range(3):
        n = int(round(max_mm / zooms[a]))
        shifts, curve = [], []
        for s in range(-n, n + 1):
            o = np.roll(out_ring, s, axis=a) & cov
            i = np.roll(in_ring, s, axis=a) & cov
            if o.sum() < 100 or i.sum() < 100:
                curve.append(np.nan)
            else:
                mo, mi = E[o].mean(), E[i].mean()
                curve.append((mo - mi) / (0.5 * (mo + mi)))
            shifts.append(s * zooms[a])
        curve = np.array(curve)
        b = int(np.nanargmax(curve)) if np.isfinite(curve).any() else n
        best = shifts[b]
        name = {frozenset("LR"): "left-right", frozenset("AP"): "anterior-posterior", frozenset("SI"): "superior-inferior"}[frozenset(codes[a] + opposite[codes[a]])]
        # a mask shift of +s (toward codes[a]) fits best: the EPI's anatomy sits toward codes[a] from the T1 boundary
        side = {"L": "left", "R": "right", "A": "anterior", "P": "posterior", "S": "superior", "I": "inferior"}[codes[a] if best > 0 else opposite[codes[a]]]
        inside = 0 < b < len(curve) - 1                                 # a peak at the edge of the range is no peak
        res[name] = {"shifts_mm": [round(float(x), 2) for x in shifts], "contrast": [None if not np.isfinite(c) else round(float(c), 5) for c in curve],
                     "best_shift_mm": round(float(best), 2), "peak_inside_range": bool(inside),
                     "contrast_at_0": None if not np.isfinite(curve[n]) else round(float(curve[n]), 5),
                     "contrast_best": None if not np.isfinite(curve[b]) else round(float(curve[b]), 5),
                     "epi_displaced_toward": (side if best != 0 else "none") if inside else "no peak within the range"}
    return res


def regcheck(anat_dir, refs=EPI_REFS) -> dict:
    """shift_scan for the current registration (epi2t1.nii.gz) and every candidate in reg_<ref>/; writes
    regcheck.json / .md and regcheck.png in anat_dir."""
    import nibabel as nib

    from . import io
    A = Path(anat_dir)
    wimg = nib.load(str(A / "t1_wmseg.nii.gz"))
    wm = np.asarray(wimg.dataobj)
    cands = [("current", A / "epi2t1.nii.gz")] + [(r, A / f"reg_{r}" / "epi2t1.nii.gz") for r in refs]
    rows = {}
    for name, path in cands:
        if not path.exists():
            continue
        e = np.asarray(nib.load(str(path)).dataobj, np.float64)
        rows[name] = shift_scan(e, wm, wimg.affine)
    out = {"anat_dir": str(A), "candidates": rows}
    io.write_json(A / "regcheck.json", out)
    L = [f"# Registration check: {A.name}", "",
         "Best shift of the T1 white-matter boundary against the EPI's white-matter contrast (0 = aligned; the EPI is "
         "displaced toward the side named). Contrast: (just outside - just inside the boundary) / average, at 0 shift "
         "and at the best shift; higher means a sharper fit.", "",
         "| EPI reference | left-right | anterior-posterior | superior-inferior | contrast at 0 / best (left-right) |", "|---|---|---|---|---|"]
    for name, r in rows.items():
        cell = lambda k: (f"{r[k]['best_shift_mm']:+.1f} mm ({r[k]['epi_displaced_toward']})" if r[k].get("peak_inside_range", True)
                          else "no peak (curve rises to the edge of +-8 mm: not measurable)")
        L.append(f"| {name} | {cell('left-right')} | {cell('anterior-posterior')} | {cell('superior-inferior')} | "
                 f"{r['left-right']['contrast_at_0']} / {r['left-right']['contrast_best']} |")
    io.write_text(A / "regcheck.md", "\n".join(L) + "\n")
    try:
        _regcheck_figure(A, rows, cands, wm, A / "regcheck.png")
    except Exception as exc:
        print(f"regcheck figure failed: {exc}")
    return out


def _regcheck_figure(A, rows, cands, wm, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    names = [n for n, _ in cands if n in rows]
    fig, axs = plt.subplots(len(names), 4, figsize=(14, 3.0 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        ax = axs[r, 0]
        for k, col in (("left-right", "#2a78d6"), ("anterior-posterior", "#a3a19b"), ("superior-inferior", "#1baf7a")):
            ax.plot(rows[name][k]["shifts_mm"], rows[name][k]["contrast"], color=col, label=k)
        ax.axvline(0, color="#52514e", lw=0.6)
        ax.set_title(f"{name}: left-right best {rows[name]['left-right']['best_shift_mm']:+.1f} mm", fontsize=8, loc="left")
        ax.set_xlabel("shift of the white-matter boundary (mm)", fontsize=7)
        if r == 0:
            ax.legend(fontsize=6, frameon=False)
        e = np.asarray(nib.load(str(dict(cands)[name])).dataobj, np.float64)
        cuts = [c for i, c in enumerate(_cuts(e.shape, e > 0)) if i in (1, 2, 4)]     # two upper axial, one mid coronal
        lim = np.percentile(e[e > 0], 99) if (e > 0).any() else 1
        for c, (axis, i) in enumerate(cuts, start=1):
            ax = axs[r, c]
            ax.imshow(_slice(e, axis, i).T, origin="lower", cmap="gray", vmin=0, vmax=lim)
            w = _slice(wm.astype(float), axis, i).T
            if (w > 0.5).any() and (w <= 0.5).any():
                ax.contour(w, levels=[0.5], colors="red", linewidths=0.4)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name}, {axis} = {i}", fontsize=7)
    fig.suptitle(f"{A.name}: EPI reference candidates, T1 white-matter boundary in red", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
