"""Surface-based cortical regions (mejoint.surface): icosahedral table and numbering, hemisphere combination,
CIFTI parcel time series, and anatomy labels built from a non-Schaefer region table."""
import json

import nibabel as nib
import numpy as np

from mejoint import anatomy, io, surface


def test_ico_table_numbering_and_networks():
    # 6 vertices per hemisphere; icosahedral labels 1-2 per hemisphere; Schaefer ids 1 (Vis, left), 101 (Default, right)
    ico_l = np.array([0, 1, 1, 2, 2, 2]); ico_r = np.array([1, 1, 2, 2, 0, 0])
    sch_l = np.array([0, 1, 1, 1, 0, 0]); sch_r = np.array([101, 101, 101, 0, 0, 0])
    names = {1: "7Networks_LH_Vis_1", 101: "7Networks_RH_Default_PFC_1"}
    table, gl, gr = surface.ico_table(ico_l, ico_r, sch_l, sch_r, names, n_left=2)
    assert list(table) == [1, 2, 3, 4]                                  # right label k -> id n_left + k
    assert table[1] == {"name": "cortex_left_01", "hemisphere": "LH", "network": "Vis"}
    assert table[2]["network"] == "Vis" and table[3]["network"] == "Default" and table[4]["network"] == "Default"
    assert gr.tolist() == [3, 3, 4, 4, 0, 0] and gl.tolist() == ico_l.tolist()


def test_combine_hemis_drops_midline_conflicts(tmp_path):
    L = np.zeros((4, 4, 2), np.int32); R = np.zeros_like(L)
    L[:2] = 5; R[1:] = 9                                                # row 1 claimed by both hemispheres
    for n, a in (("l", L), ("r", R)):
        nib.save(nib.Nifti1Image(a, np.eye(4)), str(tmp_path / f"{n}.nii.gz"))
    r = surface.combine_hemis(tmp_path / "l.nii.gz", tmp_path / "r.nii.gz", tmp_path / "c.nii.gz")
    C = np.asarray(nib.load(str(tmp_path / "c.nii.gz")).dataobj)
    assert r["voxels_both_dropped"] == 8 and (C[1] == 0).all() and (C[0] == 5).all() and (C[2:] == 9).all()


def test_ptseries_round_trip(tmp_path):
    verts = [{"CIFTI_STRUCTURE_CORTEX_LEFT": np.array([0, 1])}, {"CIFTI_STRUCTURE_CORTEX_RIGHT": np.array([2])}]
    pax = nib.cifti2.ParcelsAxis(np.array(["cortex_left_01", "cortex_right_02"]), [np.zeros((0, 3), int)] * 2, verts,
                                 nvertices={"CIFTI_STRUCTURE_CORTEX_LEFT": 10, "CIFTI_STRUCTURE_CORTEX_RIGHT": 10})
    vals = np.vstack([np.arange(5.0), np.full(5, np.nan)])
    surface.write_ptseries(vals, pax, 1.5, tmp_path / "x.ptseries.nii")
    img = nib.load(str(tmp_path / "x.ptseries.nii"))
    assert img.shape == (5, 2) and img.header.get_axis(0).step == 1.5
    assert list(img.header.get_axis(1).name) == ["cortex_left_01", "cortex_right_02"]
    X = np.asarray(img.dataobj)
    assert np.allclose(X[:, 0], np.arange(5.0)) and np.isnan(X[:, 1]).all()


def test_label_overlap():
    a = np.zeros((4, 4, 1), int); b = np.zeros_like(a)
    a[:2] = 1; b[:2] = 1; a[2:] = 2; b[3:] = 2
    r = surface.label_overlap(a, b, [1, 2])
    assert r["n"] == 2 and r["dice_q10_q90"][1] <= 1.0 and r["dice_median"] > 0.6


def test_anatomy_reads_a_region_table_next_to_the_labels(tmp_path):
    """A surface-based atlas brings its own region table (atlas_table_<atlas>.json); thalamus / striatum ids
    follow the largest table id, as for Schaefer."""
    shape = (6, 6, 4)
    mask = np.zeros(shape, bool); mask[1:5, 1:5, :] = True
    fe = tmp_path / "fe"; A = tmp_path / "anat"; fe.mkdir(); A.mkdir()
    aff = np.eye(4)
    nib.save(nib.Nifti1Image(mask.astype(np.float32), aff), str(fe / "analysis_mask.nii.gz"))
    lab = np.zeros(shape, np.int32); lab[1:3, 1:5, :] = 7; lab[3:5, 1:5, :] = 1000
    nib.save(nib.Nifti1Image(lab, aff), str(A / "schaefer_ico_7_epi.nii.gz"))
    for n, v in (("gm_epi", 0.9), ("csf_epi", 0.0)):
        nib.save(nib.Nifti1Image(np.where(mask, v, 0).astype(np.float32), aff), str(A / f"{n}.nii.gz"))
    table = {"7": {"name": "cortex_left_07", "hemisphere": "LH", "network": "Vis"},
             "1000": {"name": "cortex_right_1000", "hemisphere": "RH", "network": "Default"}}
    (A / "atlas_table_ico_7.json").write_text(json.dumps(table))
    cfg = io.merge(io.load_config(), {"regions": {"min_voxels": 5}})
    r = anatomy.run_dir(A, fe, cfg, atlas="ico_7")
    by = {x["id"]: x for x in r["regions"]}
    assert by[7]["usable"] and by[7]["network"] == "Vis" and by[1000]["usable"] and by[1000]["network"] == "Default"
    assert (A / "labels_ico_7.nii.gz").exists()
