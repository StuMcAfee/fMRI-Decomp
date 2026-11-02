"""Phase 0 audit on fake dcm2niix-style folders with known answers."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mejoint import io, inventory

TE_MS = [16.20, 31.22, 46.24, 61.26]


def _write_echo(path: Path, data, meta):
    img = nib.Nifti1Image(data.astype(np.float32), np.diag([3.5, 3.5, 3.5, 1.0]))
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = 0.75
    nib.save(img, str(path))
    path.with_name(path.name.replace(".nii.gz", ".json")).write_text(json.dumps(meta))


def make_subject(root: Path, n_vol=40, n_noise=2, n_nss=3, phase=True, physio=True, te=TE_MS,
                 unequal=False, seed=0):
    rng = np.random.default_rng(seed)
    nd = root / "nifti"
    nd.mkdir(parents=True)
    shape = (10, 10, 8)
    brain = np.zeros(shape, bool)
    brain[2:8, 2:8, 1:7] = True
    slicetiming = [0.0, 0.1875, 0.375, 0.5625] * 2       # 8 slices, MB 2
    for e, t in enumerate(te, start=1):
        T = n_vol - (e - 1) if unequal else n_vol
        mu = np.where(brain, 1000 * np.exp(-t / 45.0), 5.0)
        d = mu[..., None] * (1 + 0.003 * rng.standard_normal(shape + (T,))) + rng.standard_normal(shape + (T,))
        d[..., :n_nss] *= np.linspace(1.4, 1.1, n_nss) if n_nss else 1
        if n_noise:
            d[..., T - n_noise:] = np.abs(rng.standard_normal(shape + (n_noise,)))   # no RF: noise only
        meta = {"EchoTime": t / 1000, "EchoNumber": e, "RepetitionTime": 0.75,
                "ImageType": ["ORIGINAL", "PRIMARY", "M", "MB", "ND"], "MultibandAccelerationFactor": 2,
                "PartialFourier": 0.875, "PhaseEncodingDirection": "j-", "ParallelReductionFactorInPlane": 2,
                "SliceTiming": slicetiming, "MagneticFieldStrength": 3, "CoilString": "HEA;HEP"}
        _write_echo(nd / f"rest_e{e}.nii.gz", d, meta)
        if phase:
            _write_echo(nd / f"rest_e{e}_ph.nii.gz", rng.uniform(-4096, 4096, shape + (T,)),
                        {**meta, "ImageType": ["ORIGINAL", "PRIMARY", "P", "MB", "ND"]})
    sbref = nd / "rest_sbref_e1.nii.gz"
    _write_echo(sbref, np.ones(shape + (1,)), {"EchoTime": te[0] / 1000})
    if physio:
        (nd / "physio").mkdir()
        (nd / "physio" / "Physio_RESP.log").write_text("fake")
    # a fake DICOM (magic bytes only) one level deeper
    (nd / "dicom").mkdir()
    (nd / "dicom" / "IM0001").write_bytes(b"\0" * 128 + b"DICM" + b"\0" * 16)

    rd = root / "run"
    (rd / "motion").mkdir(parents=True)
    (rd / "work").mkdir()
    (rd / "tedana").mkdir()
    (rd / "volumes_used.tsv").write_text("echo1\techo2\n0\t0\n1\t1\n")
    np.savetxt(rd / "motion" / "mcflirt.par", np.zeros((n_vol, 6)))
    for e in range(1, 5):
        nib.save(nib.Nifti1Image(np.zeros(shape + (3,), np.float32), np.eye(4)),
                 str(rd / "work" / f"echo-{e}_desc-mc_bold.nii.gz"))
    (rd / "result.json").write_text(json.dumps({"parameters": {"interp": "spline", "keep_intermediates": True}}))
    (rd / "tedana" / "desc-tedana_metrics.tsv").write_text(
        "Component\tclassification\tclassification_tags\n"
        "ICA_00\taccepted\tLikely BOLD\nICA_01\trejected\tUnlikely BOLD\nICA_02\taccepted\tLikely BOLD\n")
    return nd, rd


@pytest.fixture
def cfg():
    return io.load_config()


def test_full_subject(tmp_path, cfg):
    nd, rd = make_subject(tmp_path)
    inv = inventory.inventory_subject(nd, cfg, run_dir=rd, subject="fake01")

    assert len(inv["echo_files"]) == 4
    assert [Path(f["path"]).name for f in inv["echo_files"]] == [f"rest_e{e}.nii.gz" for e in range(1, 5)]
    assert inv["te_match"] is True
    assert len(inv["phase_files"]) == 4
    assert inv["sidecar"]["mb_from_slicetiming"] == 2
    pe = inv["volumes"]["per_echo"]
    assert [e["n_trailing_noise_frames"] for e in pe] == [2, 2, 2, 2]
    assert [e["n_nonsteady_leading"] for e in pe] == [3, 3, 3, 3]
    assert any("Physio_RESP" in p["path"] for p in inv["physio_files"])
    assert inv["dicom"]["n_dicom_files"] == 1
    r = inv["run"]
    assert r["result_json_interp"] == {"parameters.interp": "spline"}
    assert len(r["mc_echoes"]) == 4
    assert r["tedana"]["n_likely_bold_tag"] == 2
    assert r["mcflirt_par"]["n_rows"] == 40

    out = tmp_path / "out"
    inventory.write_outputs(inv, cfg, out)
    md = (out / "inventory.md").read_text()
    assert "| TE (ms), JSON | [16.2, 31.22, 46.24, 61.26] | JSON EchoTime | confirmed |" in md
    assert "Phase saved" in md and "YES: ask user" in md
    assert "MB stack separation | 4 slices" in md
    for f in ("inventory.json", "volume_means.tsv", "done.json", "config_used.yaml"):
        assert (out / f).exists()
    done = json.loads((out / "done.json").read_text())
    assert done["config_sha256"] == io.config_hash(cfg)
    assert all(v["sha256"] for v in done["inputs"].values())


def test_clean_subject_and_te_mismatch(tmp_path, cfg):
    nd, _ = make_subject(tmp_path, n_noise=0, n_nss=0, phase=False, physio=False,
                         te=[15.0, 31.22, 46.24, 61.26])
    inv = inventory.inventory_subject(nd, cfg)
    assert inv["te_match"] is False
    assert inv["phase_files"] == []
    pe = inv["volumes"]["per_echo"]
    assert all(e["n_trailing_noise_frames"] == 0 for e in pe)
    assert all(e["n_nonsteady_leading"] == 0 for e in pe)
    assert inv["physio_files"] == []
    md = inventory.render_markdown(inv, cfg)
    assert "MISMATCH" in md
    assert "| Phase saved | 0 phase file(s) | filenames/ImageType | no |" in md


def test_unequal_volume_counts(tmp_path, cfg):
    nd, _ = make_subject(tmp_path, unequal=True, phase=False)
    inv = inventory.inventory_subject(nd, cfg)
    assert [f["n_volumes"] for f in inv["echo_files"]] == [40, 39, 38, 37]
    assert "UNEQUAL" in inventory.render_markdown(inv, cfg)


def test_cli_runs(tmp_path):
    from mejoint.cli import main
    nd, rd = make_subject(tmp_path)
    assert main(["inventory", "--nifti-dir", str(nd), "--run-dir", str(rd), "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "inventory.md").exists()
