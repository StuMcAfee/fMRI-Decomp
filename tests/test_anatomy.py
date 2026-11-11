"""Route B step 3, the Python side (anatomy.py); the FSL registrations run only on the cluster.

- the T1 is found by subject and scan time (list file with quotes, then a folder search);
- the bundled Schaefer label tables parse into 200 / 400 regions with hemisphere and network;
- Harvard-Oxford subcortical labels group into left / right thalamus and striatum;
- region labels: cortex limited to the mask, grey matter >= gm_min, no CSF / dropout voxels; thalamus and striatum
  added after the cortex; regions below min_voxels flagged; run_dir writes labels, table and QC figure.
"""
from pathlib import Path

import nibabel as nib
import numpy as np

from mejoint import anatomy, io

REPO = Path(__file__).resolve().parents[1]


def test_find_t1(tmp_path):
    lst = tmp_path / "t1.txt"
    lst.write_text('"/data/Rad/P04/SCANDATE/x/2_-_SAG/2_-_SAG_3D_T1_MPRAGE_TIMESTAMP_2.nii.gz"\n'
                   '"/data/Rad/P04/20250101/x/SAG_3D_T1_MPRAGE_20250101000000_2.nii.gz"\n'
                   '"/data/Rad/P15/SCANDATE/x/SAG_3D_T1_MPRAGE_TIMESTAMP_2.nii.gz"\n')
    assert anatomy.find_t1("sub-P04_acq-TIMESTAMP_ser-5", lst).endswith("TIMESTAMP_2.nii.gz")
    assert "P15" in anatomy.find_t1("sub-P15_acq-TIMESTAMP_ser-5", lst)
    root = tmp_path / "Rad"
    f = root / "P12" / "SCANDATE" / "MR" / "SAG_3D_T1_MPRAGE_TIMESTAMP_2.nii.gz"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"")
    assert anatomy.find_t1("sub-P12_acq-TIMESTAMP_ser-5", lst, root) == str(f)
    assert anatomy.find_t1("sub-99999_acq-1_ser-1", lst, root) is None


def test_schaefer_tables():
    for n, k in ((200, 7), (400, 17)):
        t = anatomy.schaefer_table(REPO / "atlases" / "schaefer2018" / f"Schaefer2018_{n}Parcels_{k}Networks_order.txt")
        assert sorted(t) == list(range(1, n + 1))
        assert {v["hemisphere"] for v in t.values()} == {"LH", "RH"}
        nets = {v["network"] for v in t.values()}
        assert len(nets) == k, nets
    t7 = anatomy.schaefer_table(REPO / "atlases" / "schaefer2018" / "Schaefer2018_200Parcels_7Networks_order.txt")
    assert {v["network"] for v in t7.values()} == {"Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"}
    img = nib.load(str(REPO / "atlases" / "schaefer2018" / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"))
    assert img.shape == (91, 109, 91) and int(np.asarray(img.dataobj).max()) == 200


def test_subcortical_groups(tmp_path):
    names = ["Left Cerebral White Matter", "Left Cerebral Cortex", "Left Lateral Ventrical", "Left Thalamus", "Left Caudate",
             "Left Putamen", "Left Pallidum", "Brain-Stem", "Left Hippocampus", "Left Amygdala", "Left Accumbens",
             "Right Cerebral White Matter", "Right Cerebral Cortex", "Right Lateral Ventricle", "Right Thalamus",
             "Right Caudate", "Right Putamen", "Right Pallidum", "Right Hippocampus", "Right Amygdala", "Right Accumbens"]
    xml = tmp_path / "ho.xml"
    xml.write_text("<atlas><data>" + "".join(f'<label index="{i}" x="0" y="0" z="0">{n}</label>' for i, n in enumerate(names))
                   + "</data></atlas>")
    vals = anatomy.ho_subcortical_values(xml)
    assert vals[4] == "Left Thalamus" and vals[21] == "Right Accumbens"
    g = anatomy.subcortical_groups(vals)
    assert g[("L", "thalamus")] == [4] and sorted(g[("L", "striatum")]) == [5, 6, 11]
    assert g[("R", "thalamus")] == [15] and sorted(g[("R", "striatum")]) == [16, 17, 21]
    assert all(8 not in v for v in g.values())                          # brainstem never a region


def _toy(shape=(12, 12, 8)):
    mask = np.zeros(shape, bool); mask[1:-1, 1:-1, 1:-1] = True
    cortex = np.zeros(shape); cortex[1:6, 1:-1, 1:-1] = 1; cortex[6:-1, 1:-1, 1:-1] = 2
    gm = np.full(shape, 0.8); gm[1:3] = 0.1                             # a low grey-matter strip in region 1
    sub = np.zeros(shape); sub[3:5, 3:5, 3:5] = 4; sub[8:10, 3:5, 3:5] = 16    # L thalamus inside region 1, R caudate in 2
    classes = np.ones(shape, np.int8); classes[:, :, 1] = 3                # a CSF-dominated slice
    table = {1: {"name": "7Networks_LH_Vis_1", "hemisphere": "LH", "network": "Vis"},
             2: {"name": "7Networks_RH_Default_1", "hemisphere": "RH", "network": "Default"}}
    groups = {("L", "thalamus"): [4], ("R", "thalamus"): [15], ("L", "striatum"): [5, 6, 11], ("R", "striatum"): [16, 17, 21]}
    return mask, cortex, gm, sub, classes, table, groups


def test_build_labels_rules():
    mask, cortex, gm, sub, classes, table, groups = _toy()
    r = anatomy.build_labels(cortex, sub, gm, mask, classes, table, groups, gm_min=0.3, min_voxels=20)
    lab = r["labels"]
    assert lab[~mask].max() == 0
    assert np.all(lab[:, :, 1] == 0)                                   # CSF-dominated slice left out
    assert np.all(lab[1:3] == 0)                                       # low grey matter left out (cortex)
    ids = {row["name"]: row["id"] for row in r["regions"]}
    assert set(ids) == {"7Networks_LH_Vis_1", "7Networks_RH_Default_1", "L_thalamus", "R_thalamus", "L_striatum", "R_striatum"}
    # subcortical voxels inside a cortical parcel stay cortex (cortex first); R caudate in parcel 2 likewise
    assert (lab == ids["L_thalamus"]).sum() == 0
    rows = {row["name"]: row for row in r["regions"]}
    assert rows["R_thalamus"]["n_voxels"] == 0 and not rows["R_thalamus"]["usable"]
    assert rows["7Networks_LH_Vis_1"]["usable"] and rows["7Networks_LH_Vis_1"]["kind"] == "cortex"
    r2 = anatomy.build_labels(cortex, sub, gm, mask, classes, table, groups, secondary=False)
    assert len(r2["regions"]) == 2
    # CSF exclusion for the deep regions only (user, 2026-11-10)
    csf = np.zeros(mask.shape)
    csf[...] = 0.9
    r3 = anatomy.build_labels(cortex, sub, gm, mask, classes, table, groups, gm_min=0.3, min_voxels=20,
                              csf_epi=csf, secondary_csf_max=0.3)
    rows3 = {row["name"]: row for row in r3["regions"]}
    assert all(rows3[n]["n_voxels"] == 0 for n in ("L_thalamus", "R_thalamus", "L_striatum", "R_striatum"))
    assert rows3["L_striatum"]["n_voxels_before_csf_exclusion"] == rows["L_striatum"]["n_voxels"]
    assert rows3["7Networks_LH_Vis_1"]["n_voxels"] == rows["7Networks_LH_Vis_1"]["n_voxels"]   # cortex untouched
    r4 = anatomy.build_labels(cortex, sub, gm, mask, classes, table, groups, gm_min=0.3, min_voxels=20,
                              csf_epi=csf, cortex_csf_max=0.3)                    # primary cortex rule (2026-11-11)
    rows4 = {row["name"]: row for row in r4["regions"]}
    assert rows4["7Networks_LH_Vis_1"]["n_voxels"] == 0
    assert rows4["7Networks_LH_Vis_1"]["n_voxels_before_csf_exclusion"] == rows["7Networks_LH_Vis_1"]["n_voxels"]


def test_run_dir_writes_outputs(tmp_path):
    mask, cortex, gm, sub, classes, table, groups = _toy((20, 22, 16))
    aff = np.diag([3.28, 3.28, 3.3, 1.0])
    fe, A = tmp_path / "fe", tmp_path / "anat"
    fe.mkdir(); A.mkdir()
    ref = nib.Nifti1Image(mask.astype(np.uint8), aff)
    nib.save(ref, str(fe / "analysis_mask.nii.gz"))
    atl = np.zeros(mask.shape); atl[mask] = np.where(np.indices(mask.shape)[0][mask] < 10, 1, 101)   # LH Vis_1, RH region 101
    for name, arr in (("schaefer_200_7_epi", atl), ("ho_sub_epi", sub), ("gm_epi", gm), ("epi_mean", mask * 100.0),
                      ("wm_epi", (np.indices(mask.shape)[2] == 8) & mask), ("csf_epi", np.zeros(mask.shape))):
        nib.save(nib.Nifti1Image(np.asarray(arr, np.float32), aff), str(A / f"{name}.nii.gz"))
    names = ["Left Cerebral White Matter", "Left Cerebral Cortex", "Left Lateral Ventrical", "Left Thalamus", "Left Caudate",
             "Left Putamen", "Left Pallidum", "Brain-Stem", "Left Hippocampus", "Left Amygdala", "Left Accumbens",
             "Right Cerebral White Matter", "Right Cerebral Cortex", "Right Lateral Ventricle", "Right Thalamus",
             "Right Caudate", "Right Putamen", "Right Pallidum", "Right Hippocampus", "Right Amygdala", "Right Accumbens"]
    (A / "HarvardOxford-Subcortical.xml").write_text(
        "<atlas><data>" + "".join(f'<label index="{i}">{n}</label>' for i, n in enumerate(names)) + "</data></atlas>")
    cfg = io.load_config()
    out = anatomy.run_dir(A, fe, cfg, atlas="200_7")
    assert out["n_regions"] == 200 + 4
    used = [r for r in out["regions"] if r["n_voxels"] > 0]
    assert {r["id"] for r in used} == {1, 101}
    for f in ("labels_200_7.nii.gz", "regions_200_7.tsv", "regions_200_7.json", "qc_200_7.png"):
        assert (A / f).exists(), f
    assert (A / "labels_200_7_allvox.nii.gz").exists() and out["cortex_csf_covariate"]["cortex_voxels_removed"] == 0


def _blob(shape, cx, radii=(35, 42, 32)):
    g = np.indices(shape).astype(float)
    c = np.array([cx, shape[1] / 2, shape[2] / 2])[:, None, None, None]
    return np.sqrt((((g - c) / np.array(radii)[:, None, None, None]) ** 2).sum(0))


def test_shift_scan_finds_a_left_right_shift():
    """EPI anatomy displaced 3 voxels along the left-right axis from the T1 white matter: the scan reports 3 mm on
    that axis toward the right side, 0 on the others; with no displacement, 0 everywhere."""
    shape = (90, 100, 80)
    wm = _blob(shape, 45) < 0.6
    aff = np.diag([-1.0, 1.0, 1.0, 1.0])                                # axis 0 increases toward subject left
    for d, side in ((3, "left"), (-3, "right"), (0, "none")):
        r = _blob(shape, 45 + d)
        epi = np.where(r < 1, np.where(r < 0.6, 100.0, 160.0), 0.0)
        res = anatomy.shift_scan(epi, wm, aff)
        assert res["left-right"]["best_shift_mm"] == abs(d) * np.sign(d), res["left-right"]
        assert res["left-right"]["epi_displaced_toward"] == side
        assert res["anterior-posterior"]["best_shift_mm"] == 0 and res["superior-inferior"]["best_shift_mm"] == 0
        assert res["left-right"]["peak_inside_range"]
    # a displacement beyond the scanned range has no peak inside it: flagged, not reported as a shift
    r = _blob(shape, 45 + 12)
    epi = np.where(r < 1, np.where(r < 0.6, 100.0, 160.0), 0.0)
    far = anatomy.shift_scan(epi, wm, aff, max_mm=8.0)["left-right"]
    assert not far["peak_inside_range"] and far["epi_displaced_toward"] == "no peak within the range"


def test_reference_images(tmp_path):
    shape = (12, 12, 8)
    mask = np.zeros(shape, bool); mask[2:-2, 2:-2, 2:-2] = True
    aff = np.diag([3.28, 3.28, 3.3, 1.0])
    te = [16.20, 31.22, 46.24, 61.26]
    mu = np.zeros(shape + (4,)); mu[mask] = 1000 * np.exp(-np.array(te) / 40.0)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), str(tmp_path / "analysis_mask.nii.gz"))
    nib.save(nib.Nifti1Image(np.where(mask[..., None], mu, np.nan).astype(np.float32), aff), str(tmp_path / "mu.nii.gz"))
    nib.save(nib.Nifti1Image(np.where(mask, 40.0, np.nan).astype(np.float32), aff), str(tmp_path / "t2s_ms.nii.gz"))
    (tmp_path / "summary.json").write_text('{"te_ms": [16.20, 31.22, 46.24, 61.26]}')
    names = anatomy.reference_images(tmp_path, tmp_path / "anat")
    assert names == list(anatomy.EPI_REFS)
    t2 = np.asarray(nib.load(str(tmp_path / "anat" / "ref_t2s.nii.gz")).dataobj)
    assert np.allclose(t2[mask], 40.0) and np.all(t2[~mask] == 0)
    e2 = np.asarray(nib.load(str(tmp_path / "anat" / "ref_echo2.nii.gz")).dataobj)
    assert np.allclose(e2[mask], 1000 * np.exp(-te[1] / 40.0), rtol=1e-5) and np.all(e2[~mask] == 0)
    sw = np.asarray(nib.load(str(tmp_path / "anat" / "ref_t2sw.nii.gz")).dataobj)
    assert np.all(np.isfinite(sw)) and mu[mask][:, 3].max() < sw[mask].min() < mu[mask][:, 0].min()


def test_run_dir_out_dir_leaves_primary_labels(tmp_path):
    """A sensitivity variant (e.g. grey-matter threshold 0.5) writes its labels to another folder, copies the
    tissue maps route B reads, and leaves the primary labels alone."""
    test_run_dir_writes_outputs(tmp_path)
    import nibabel as nib

    from mejoint import io
    A, fe = tmp_path / "anat", tmp_path / "fe"
    before = np.asarray(nib.load(str(A / "labels_200_7.nii.gz")).dataobj).copy()
    cfg = io.merge(io.load_config(), {"regions": {"gm_min": 0.99}})
    out = anatomy.run_dir(A, fe, cfg, atlas="200_7", out_dir=tmp_path / "gm")
    assert (tmp_path / "gm" / "labels_200_7.nii.gz").exists() and (tmp_path / "gm" / "csf_epi.nii.gz").exists()
    assert np.array_equal(np.asarray(nib.load(str(A / "labels_200_7.nii.gz")).dataobj), before)
    assert out["cortex_voxels"] <= int((before > 0).sum())
