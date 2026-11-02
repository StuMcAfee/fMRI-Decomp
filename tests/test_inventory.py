"""Phase 0 audit on fake folders laid out like mbme_tedana.py inputs and outputs, with known answers.

Layout mimicked (from mbme_tedana.py):
  <nifti_dir>/rest_<stamp>_<series>_e{1..4}.nii.gz + .json   (dcm2niix default names)
  <nifti_dir>/<series>_-_REST/  DICOMs, one level deeper
  <run_dir>/result.json, echo_info.json, volumes_used.tsv, motion/mcflirt.par, brain_mask.nii.gz,
            work/echo-k_desc-mc_bold.nii.gz (with --keep-intermediates), tedana/
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mejoint import io, inventory

TE_MS = [16.20, 31.22, 46.24, 61.26]
STAMP = "20260421124817"
SHAPE = (10, 10, 8)


def _write(path: Path, data, meta=None):
    img = nib.Nifti1Image(np.asarray(data, np.float32), np.diag([3.5, 3.5, 3.5, 1.0]))
    img.header.set_xyzt_units("mm", "sec")
    if img.ndim == 4:
        img.header["pixdim"][4] = 0.75
    nib.save(img, str(path))
    if meta is not None:
        path.with_name(path.name.replace(".nii.gz", ".json")).write_text(json.dumps(meta))


def _brain():
    b = np.zeros(SHAPE, bool)
    b[2:8, 2:8, 1:7] = True
    return b


def _echo_data(rng, t_ms, T, n_noise, n_nss):
    mu = np.where(_brain(), 1000 * np.exp(-t_ms / 45.0), 5.0)
    d = mu[..., None] * (1 + 0.003 * rng.standard_normal(SHAPE + (T,))) + rng.standard_normal(SHAPE + (T,))
    if n_nss:
        d[..., :n_nss] *= np.linspace(1.4, 1.1, n_nss)
    if n_noise:
        d[..., T - n_noise:] = np.abs(rng.standard_normal(SHAPE + (n_noise,)))   # no RF: noise only
    return d


def _dicom(path: Path, series, desc, itype, echo, te):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SeriesNumber, ds.SeriesDescription, ds.ImageType = series, desc, itype
    ds.EchoNumbers, ds.EchoTime, ds.SliceThickness = echo, te, 3.5
    try:
        ds.save_as(str(path), enforce_file_format=True)        # pydicom >= 3
    except TypeError:
        ds.save_as(str(path), write_like_original=False)       # pydicom 2.x


def make_subject(root: Path, n_vol=40, n_noise=2, n_nss=3, phase=True, physio=False, te=TE_MS,
                 unequal=False, decoy=True, series=5, dicoms=True, seed=0):
    rng = np.random.default_rng(seed)
    nd = root / "SE000001"
    nd.mkdir(parents=True)
    slicetiming = [0.0, 0.1875, 0.375, 0.5625] * 2       # 8 slices, MB 2
    base = {"RepetitionTime": 0.75, "MultibandAccelerationFactor": 2, "PartialFourier": 0.875,
            "PhaseEncodingDirection": "j-", "ParallelReductionFactorInPlane": 2, "SliceTiming": slicetiming,
            "MagneticFieldStrength": 3, "CoilString": "HEA;HEP", "SeriesDescription": "rest_mbme"}
    for e, t in enumerate(te, start=1):
        T = n_vol - (e - 1) if unequal else n_vol
        meta = {**base, "EchoTime": t / 1000, "EchoNumber": e, "SeriesNumber": series,
                "ImageType": ["ORIGINAL", "PRIMARY", "M", "MB", "ND"]}
        _write(nd / f"rest_{STAMP}_{series}_e{e}.nii.gz", _echo_data(rng, t, T, n_noise, n_nss), meta)
        if phase:   # Siemens: phase is its own series number
            _write(nd / f"rest_{STAMP}_{series + 1}_e{e}_ph.nii.gz", rng.uniform(-4096, 4096, SHAPE + (T,)),
                   {**meta, "SeriesNumber": series + 1, "ImageType": ["ORIGINAL", "PRIMARY", "P", "MB", "ND"]})
        if decoy:   # another 4-echo series with MORE volumes: the heuristic alone would pick it
            _write(nd / f"task_{STAMP}_{series + 4}_e{e}.nii.gz", _echo_data(rng, t, n_vol + 10, 0, 0),
                   {**base, "EchoTime": t / 1000, "EchoNumber": e, "SeriesNumber": series + 4})
    # field map: 3D magnitude + phase difference; must NOT count as rest phase
    _write(nd / f"gre_field_mapping_{STAMP}_3_e1.nii.gz", np.ones(SHAPE), {"EchoTime": 0.00492})
    _write(nd / f"gre_field_mapping_{STAMP}_3_e2_ph.nii.gz", np.ones(SHAPE),
           {"EchoTime": 0.00738, "ImageType": ["ORIGINAL", "PRIMARY", "P", "ND"]})
    _write(nd / f"rest_{STAMP}_{series}_e1_sbref.nii.gz", np.ones(SHAPE + (1,)), {"EchoTime": te[0] / 1000})
    if physio:
        (nd / "Physio_RESP.log").write_text("fake")
    if dicoms:
        dd = nd / f"{series}_-_REST_MBME"
        dd.mkdir()
        for e, t in enumerate(te, start=1):
            for k in range(3):
                _dicom(dd / f"IM{e}{k:03d}", series, "rest_mbme", ["ORIGINAL", "PRIMARY", "M", "MB", "ND"], e, t)
        other = nd / "3_-_GRE_FIELD"
        other.mkdir()
        _dicom(other / "IM0001", 3, "gre_field_mapping", ["ORIGINAL", "PRIMARY", "M", "ND"], 1, 4.92)

    rd = root / "OUT" / f"sub-41781_acq-{STAMP}_ser-{series}"
    (rd / "motion").mkdir(parents=True)
    (rd / "work").mkdir()
    (rd / "tedana").mkdir()
    n_used = n_vol - (3 if unequal else 0)
    (rd / "volumes_used.tsv").write_text("echo1\techo2\techo3\techo4\n" +
                                         "".join(f"{i}\t{i}\t{i}\t{i}\n" for i in range(n_used)))
    np.savetxt(rd / "motion" / "mcflirt.par", np.zeros((n_used, 6)))
    _write(rd / "brain_mask.nii.gz", _brain().astype(np.float32))
    for e, t in enumerate(te, start=1):
        mc = _echo_data(rng, t, n_used, n_noise, 0)
        if e == 4:
            mc[2, 2, 1, 5] = -3.0     # spline undershoot in one brain voxel
        _write(rd / "work" / f"echo-{e}_desc-mc_bold.nii.gz", mc)
    e1 = nd / f"rest_{STAMP}_{series}_e1.nii.gz"
    params = {"dummy_scans": 0, "max_volumes": None, "tree": "tedana_orig", "tedpca": "aic",
              "ica_method": "robustica", "n_robust_runs": 30, "seed": 42, "interp": "spline"}
    (rd / "result.json").write_text(json.dumps({
        "status": "ok", "input_echo1": str(e1), "run_id": rd.name, "tedana_version": "26.0.3",
        "parameters": params, "echo_times_ms": te, "te_source": "dicom", "n_volumes": n_used,
        "n_vols_per_echo_raw": [n_vol] * 4, "volume_pairing": "equal counts", "first_volume_signal_ratio": 1.4,
        "mean_fd_mm": 0.1, "max_fd_mm": 0.5, "n_vols_fd_gt_0p5mm": 0, "mask_voxels": int(_brain().sum()),
        "n_likely_bold": 2, "n_components_total": 3}))
    (rd / "echo_info.json").write_text(json.dumps({"te_info": {
        "source": "dicom", "tes_ms": te, "json": {"tes_ms": te}, "dicom": {"tes_ms": te, "folder": "x"},
        "dicom_error": None}}))
    (rd / "tedana" / "desc-tedana_metrics.tsv").write_text(
        "Component\tclassification\tclassification_tags\n"
        "ICA_00\taccepted\tLikely BOLD\nICA_01\trejected\tUnlikely BOLD\nICA_02\taccepted\tLikely BOLD,Low variance\n")
    (rd / "tedana" / "desc-ICA_mixing.tsv").write_text("ICA_00\n0\n")
    return nd, rd


@pytest.fixture
def cfg():
    return io.load_config()


def test_full_subject(tmp_path, cfg):
    nd, rd = make_subject(tmp_path)
    inv = inventory.inventory_subject(None, cfg, run_dir=rd, subject="41781")   # NIfTI folder from result.json

    assert inv["nifti_dir"] == str(nd)
    assert inv["rest_series_how"] == "result.json input_echo1"
    assert inv["series_number"] == 5
    assert [Path(f["path"]).name for f in inv["echo_files"]] == [f"rest_{STAMP}_5_e{e}.nii.gz" for e in range(1, 5)]
    assert inv["te_match"] is True
    assert len(inv["phase_files"]) == 4                                  # series 6 phase, matched by grid/volumes
    assert [Path(p).name for p in inv["other_phase_files"]] == [f"gre_field_mapping_{STAMP}_3_e2_ph.nii.gz"]
    assert inv["sidecar"]["mb_from_slicetiming"] == 2
    pe = inv["volumes"]["per_echo"]
    assert [e["n_trailing_noise_frames"] for e in pe] == [2, 2, 2, 2]
    assert [e["n_nonsteady_leading"] for e in pe] == [3, 3, 3, 3]
    assert inv["physio_files"] == []

    d = inv["dicom"]
    assert Path(d["folders_scanned"][-1]["folder"]).name == "5_-_REST_MBME"   # series folder searched first
    assert sorted({s["echo_numbers"] for s in d["series"] if s["series_number"] == 5}) == ["1", "2", "3", "4"]
    assert all(s["series_number"] != 3 for s in d["series"])                  # stopped before the field map

    r = inv["run"]
    assert r["result"]["parameters"]["ica_method"] == "robustica"
    assert r["result_json_interp"] == {"parameters.interp": "spline"}
    assert r["echo_info"]["source"] == "dicom"
    assert r["tedana"]["n_likely_bold_tag"] == 2 and r["tedana"]["desc-ICA_mixing.tsv"] is True
    assert r["volumes_used"]["n_rows"] == 40 and r["volumes_used"]["last_row"] == [39] * 4
    mcc = inv["mc_checks"]
    assert [m["n_trailing_noise_frames"] for m in mcc] == [2, 2, 2, 2]
    assert [m["n_mask_voxels_any_nonpositive"] for m in mcc] == [0, 0, 0, 1]

    out = tmp_path / "out"
    inventory.write_outputs(inv, cfg, out)
    md = (out / "inventory.md").read_text()
    assert "| TE (ms), JSON | [16.2, 31.22, 46.24, 61.26] | JSON EchoTime | confirmed |" in md
    assert "Phase saved (rest series) | 4 matching" in md and "YES: ask user" in md
    assert "MB stack separation | 4 slices" in md
    assert "YES: pipeline included them" in md
    for f in ("inventory.json", "volume_means.tsv", "done.json", "config_used.yaml"):
        assert (out / f).exists()
    done = json.loads((out / "done.json").read_text())
    assert done["config_sha256"] == io.config_hash(cfg)
    assert "result_json" in done["inputs"] and all(v["sha256"] for v in done["inputs"].values())


def test_heuristic_without_run_dir_picks_decoy(tmp_path, cfg):
    """Without result.json the heuristic picks the longer decoy series, and says it used a heuristic."""
    nd, _ = make_subject(tmp_path, dicoms=False)
    inv = inventory.inventory_subject(nd, cfg)
    assert inv["rest_series_key"] == f"task_{STAMP}_9"
    assert "heuristic" in inv["rest_series_how"]


def test_clean_subject_and_te_mismatch(tmp_path, cfg):
    nd, rd = make_subject(tmp_path, n_noise=0, n_nss=0, phase=False, te=[15.0, 31.22, 46.24, 61.26],
                          decoy=False, dicoms=False)
    inv = inventory.inventory_subject(nd, cfg, run_dir=rd)
    assert inv["te_match"] is False
    assert inv["phase_files"] == []
    pe = inv["volumes"]["per_echo"]
    assert all(e["n_trailing_noise_frames"] == 0 for e in pe)
    assert all(e["n_nonsteady_leading"] == 0 for e in pe)
    md = inventory.render_markdown(inv, cfg)
    assert "MISMATCH" in md
    assert "| Phase saved (rest series) | 0 matching 4D phase NIfTI(s) | filenames/ImageType; DICOM sample | no |" in md


def test_physio_found_is_flagged(tmp_path, cfg):
    nd, rd = make_subject(tmp_path, physio=True, decoy=False, dicoms=False)
    md = inventory.render_markdown(inventory.inventory_subject(nd, cfg, run_dir=rd), cfg)
    assert "CONFLICTS with user report" in md


def test_unequal_volume_counts(tmp_path, cfg):
    nd, rd = make_subject(tmp_path, unequal=True, phase=False, decoy=False, dicoms=False)
    inv = inventory.inventory_subject(nd, cfg, run_dir=rd)
    assert [f["n_volumes"] for f in inv["echo_files"]] == [40, 39, 38, 37]
    assert "UNEQUAL" in inventory.render_markdown(inv, cfg)


def test_pipeline_script_and_spaces_in_paths(tmp_path, cfg):
    root = tmp_path / "Monitoring Visit"
    nd, rd = make_subject(root, decoy=False, dicoms=False)
    script = root / "mbme_tedana.py"
    script.write_text("import argparse\n# comment about flirt is skipped\n"
                      "p.add_argument('--interp', default='spline')\n"
                      "cmd = ['tedana', '--tedpca', 'aic', '--seed', '42']\nx = 1\n")
    inv = inventory.inventory_subject(None, cfg, run_dir=rd, pipeline_script=script)
    ps = inv["pipeline_script"]
    assert ps["exists"] and ps["n_matches"] == 3
    assert [n for n, _ in ps["matches"]] == [1, 3, 4]
    md = inventory.render_markdown(inv, cfg)
    assert "--tedpca" in md and "comment about flirt" not in md
    missing = inventory.pipeline_script_check(root / "nope.py", cfg)
    assert missing == {"path": str(root / "nope.py"), "exists": False}


def test_cli_runs(tmp_path):
    from mejoint.cli import main
    nd, rd = make_subject(tmp_path, decoy=False, dicoms=False)
    assert main(["inventory", "--run-dir", str(rd), "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "inventory.md").exists()
    with pytest.raises(SystemExit):
        main(["inventory", "--out", str(tmp_path / "o2")])
