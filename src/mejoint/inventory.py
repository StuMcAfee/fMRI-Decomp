"""Phase 0 audit: what was acquired, and what the existing pipeline left on disk.

For one subject this module
  1. finds the echo NIfTIs and their JSON sidecars (in the NIfTI folder or one level deeper),
  2. confirms TEs, volume counts, matrix, TR and the sequence fields named in CLAUDE.md,
  3. looks for saved phase, physiological logs, noise frames and non-steady-state volumes,
  4. reads DICOM headers (if pydicom is installed) for Siemens protocol lines (MB, CAIPI, coils),
  5. checks the mbme_tedana run folder (motion-corrected echoes, motion parameters, tedana outputs).

Nothing here changes the data. Every reported value carries a source label
("JSON", "header", "derived", "heuristic", ...) so you can see what was measured
and what was inferred.
"""
from __future__ import annotations

import csv
import fnmatch
import json
import re
from pathlib import Path

import numpy as np

from . import io

NIFTI_SUFFIXES = (".nii.gz", ".nii")
PHASE_TOKEN = re.compile(r"(_ph|_phase|_part-phase)(?=[._]|$)", re.I)
ECHO_TOKEN = re.compile(r"(_e|_echo-?)(\d+)(?=[._]|$)", re.I)


# ======================================================================= file discovery
def _stem(p: Path) -> str:
    for s in NIFTI_SUFFIXES:
        if p.name.endswith(s):
            return p.name[: -len(s)]
    return p.stem


def _walk(root: Path, depth: int):
    """All files in root and up to `depth` folder levels below it."""
    root = Path(root)
    if not root.is_dir():
        return
    level = [root]
    for d in range(depth + 1):
        nxt = []
        for folder in level:
            try:
                entries = sorted(folder.iterdir())
            except PermissionError:
                continue
            for e in entries:
                if e.is_file():
                    yield e
                elif e.is_dir() and d < depth:
                    nxt.append(e)
        level = nxt


def _read_json(p: Path) -> dict:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as exc:  # unreadable sidecar is itself a finding
        return {"_error": f"could not read: {exc}"}


def find_nifti_files(nifti_dir: Path, cfg: dict) -> list[dict]:
    """Every NIfTI with its sidecar, echo number, phase flag and header facts (no voxel data loaded)."""
    import nibabel as nib

    icfg = cfg["inventory"]
    out = []
    for p in _walk(nifti_dir, icfg["search_depth"]):
        if not any(fnmatch.fnmatch(p.name, pat) for pat in icfg["nifti_patterns"]):
            continue
        stem = _stem(p)
        sidecar = p.with_name(stem + ".json")
        meta = _read_json(sidecar) if sidecar.exists() else {}
        image_type = meta.get("ImageType", [])
        if isinstance(image_type, str):
            image_type = image_type.split("\\")
        is_phase = bool(PHASE_TOKEN.search(stem)) or any(t.upper() in ("P", "PHASE") for t in image_type)

        echo = meta.get("EchoNumber")
        if echo is None:
            m = ECHO_TOKEN.search(stem)
            echo = int(m.group(2)) if m else None
        series_key = PHASE_TOKEN.sub("", ECHO_TOKEN.sub("", stem))

        try:
            img = nib.load(str(p))
            hdr = img.header
            shape = tuple(int(s) for s in img.shape)
            zooms = tuple(float(z) for z in hdr.get_zooms())
            info = {
                "shape": shape,
                "n_volumes": shape[3] if len(shape) == 4 else 1,
                "voxel_mm": zooms[:3],
                "tr_header_s": _tr_seconds(hdr, zooms),
                "datatype": str(hdr.get_data_dtype()),
                "qform_code": int(hdr["qform_code"]),
                "sform_code": int(hdr["sform_code"]),
                "affine": np.round(img.affine, 4).tolist(),
            }
        except Exception as exc:
            info = {"error": f"could not read header: {exc}", "n_volumes": 0}

        out.append({"path": str(p), "stem": stem, "sidecar": str(sidecar) if sidecar.exists() else None,
                    "json": meta, "echo": echo, "is_phase": is_phase, "series_key": series_key,
                    "image_type": image_type, **info})
    return out


def _tr_seconds(hdr, zooms):
    if len(zooms) < 4:
        return None
    unit = hdr.get_xyzt_units()[1]
    tr = float(zooms[3])
    return tr / 1000.0 if unit == "msec" else tr / 1e6 if unit == "usec" else tr


def pick_rest_series(files: list[dict], cfg: dict) -> tuple[str | None, list[dict], list[str]]:
    """Choose the multi-echo BOLD series: most magnitude echoes, then most volumes."""
    minvol = cfg["inventory"]["min_volumes_bold"]
    groups: dict[str, list[dict]] = {}
    for f in files:
        if not f["is_phase"] and f.get("n_volumes", 0) >= minvol:
            groups.setdefault(f["series_key"], []).append(f)
    if not groups:
        return None, [], []
    best = max(groups, key=lambda k: (len(groups[k]), max(f["n_volumes"] for f in groups[k])))
    echoes = sorted(groups[best], key=lambda f: (f["echo"] is None, f["echo"] or 0, f["path"]))
    others = [k for k in groups if k != best]
    return best, echoes, others


# ======================================================================= volume-level checks
def _leading_run(flags: np.ndarray) -> int:
    n = 0
    for f in flags:
        if not f:
            break
        n += 1
    return n


def volume_checks(echo_files: list[dict], cfg: dict) -> dict:
    """Noise frames (trailing, near-zero mean) and non-steady-state volumes (leading, bright).

    Loads one echo at a time as float32 to keep memory near one echo (~0.5 GB for 800 volumes).
    """
    icfg = cfg["inventory"]
    res = {"per_echo": [], "volume_means": {}}
    mask = None
    for f in echo_files:
        data, _ = io.load_4d_float32(f["path"])
        if data.ndim != 4:
            res["per_echo"].append({"echo": f["echo"], "error": "not 4D"})
            continue
        T = data.shape[3]
        flat = data.reshape(-1, T)
        whole = flat.mean(axis=0)

        # noise frames: trailing volumes with whole-volume mean < threshold x run median
        low = whole < icfg["noise_frame_rel_threshold"] * np.median(whole)
        n_trailing_noise = _leading_run(low[::-1])
        n_low_anywhere = int(low.sum())
        keep = np.ones(T, bool)
        if n_trailing_noise:
            keep[T - n_trailing_noise:] = False

        # crude in-brain mask from the first echo's temporal mean (signal frames only)
        if mask is None or mask.shape[0] != flat.shape[0]:
            tmean = flat[:, keep].mean(axis=1)
            mask = tmean > icfg["mask_fraction_of_p98"] * np.percentile(tmean, 98)
        inmask = flat[mask].mean(axis=0) if mask.any() else whole

        # non-steady-state: leading volumes far above the median of the signal frames
        sig = inmask[keep]
        med = np.median(sig)
        rsd = 1.4826 * np.median(np.abs(sig - med))
        rsd = rsd if rsd > 0 else 1e-6 * abs(med) + 1e-12
        z = (inmask - med) / rsd
        n_nss = min(_leading_run(z > icfg["nss_robust_z"]), icfg["nss_max_volumes"])

        res["per_echo"].append({
            "echo": f["echo"], "n_volumes": T,
            "n_trailing_noise_frames": n_trailing_noise,
            "n_low_mean_frames_anywhere": n_low_anywhere,
            "noise_frame_mean_ratio": float(np.median(whole[~keep]) / np.median(whole[keep])) if n_trailing_noise else None,
            "n_nonsteady_leading": n_nss,
            "first5_excess_pct": [float(100 * (v / med - 1)) for v in inmask[:5]],
            "first5_robust_z": [float(v) for v in z[:5]],
            "mask_voxels": int(mask.sum()),
        })
        res["volume_means"][f"echo{f['echo']}_whole"] = whole
        res["volume_means"][f"echo{f['echo']}_inmask"] = inmask
        del data, flat
    return res


# ======================================================================= sidecar summary
def sidecar_summary(echo_files: list[dict], cfg: dict) -> dict:
    icfg = cfg["inventory"]
    fields = {}
    for name in icfg["json_fields"]:
        vals = [f["json"].get(name, "absent") for f in echo_files]
        fields[name] = vals
    extra = {}
    for f in echo_files:
        for k, v in f["json"].items():
            if k not in icfg["json_fields"] and any(s.lower() in k.lower() for s in icfg["json_key_substrings"]):
                extra.setdefault(k, []).append(v)

    # MB factor inferred from SliceTiming: number of slices / number of distinct slice times
    mb_inferred = None
    st = echo_files[0]["json"].get("SliceTiming") if echo_files else None
    if st:
        n_unique = len(np.unique(np.round(np.asarray(st, float), 4)))
        if n_unique and len(st) % n_unique == 0:
            mb_inferred = len(st) // n_unique
    return {"fields": fields, "extra_keys": extra, "mb_from_slicetiming": mb_inferred,
            "n_slicetiming": len(st) if st else None}


# ======================================================================= DICOM headers
def dicom_scan(nifti_dir: Path, cfg: dict) -> dict:
    """Count DICOMs and read one header per series for protocol facts. Optional: needs pydicom."""
    icfg = cfg["inventory"]
    files = []
    for p in _walk(nifti_dir, icfg.get("dicom_search_depth", icfg["search_depth"])):
        if p.name.endswith((".nii", ".nii.gz", ".json", ".tsv", ".txt", ".bval", ".bvec")):
            continue
        try:
            with open(p, "rb") as fh:
                head = fh.read(132)
            if len(head) == 132 and head[128:132] == b"DICM":
                files.append(p)
        except OSError:
            continue
        if len(files) >= icfg["dicom_max_files"]:
            break
    out = {"n_dicom_files": len(files), "series": [], "note": None}
    if not files:
        return out
    try:
        import pydicom
    except ImportError:
        out["note"] = "pydicom not installed: DICOMs counted but headers not read"
        return out

    groups: dict[tuple, dict] = {}
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            continue
        itype = tuple(str(x) for x in ds.get("ImageType", []))
        key = (ds.get("SeriesNumber"), str(ds.get("SeriesDescription", "")), itype, ds.get("EchoNumbers"))
        g = groups.setdefault(key, {"count": 0, "path": p, "ds": ds})
        g["count"] += 1

    for (snum, sdesc, itype, echo), g in sorted(groups.items(), key=lambda kv: (kv[0][0] or 0, str(kv[0][3]))):
        ds = g["ds"]
        rec = {
            "series_number": snum, "series_description": sdesc, "image_type": list(itype),
            "echo_numbers": echo, "n_files": g["count"], "example": str(g["path"]),
            "EchoTime_ms": _f(ds.get("EchoTime")), "SliceThickness": _f(ds.get("SliceThickness")),
            "SpacingBetweenSlices": _f(ds.get("SpacingBetweenSlices")),
            "ImageComments": str(ds.get("ImageComments", "")) or None,
            "PATModeText_0051_1011": _private(ds, 0x0051, 0x1011),
            "CoilString_0051_100F": _private(ds, 0x0051, 0x100F),
            "is_physio": "PHYSIO" in [t.upper() for t in itype] or "physio" in sdesc.lower(),
            "ascconv": _ascconv_lines(ds, icfg["ascconv_keywords"]),
        }
        out["series"].append(rec)
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _private(ds, group, elem):
    try:
        el = ds.get((group, elem))
        return None if el is None else str(el.value)
    except Exception:
        return None


def _ascconv_lines(ds, keywords) -> list[str] | str:
    """Siemens protocol lines (from the CSA series header) that mention any keyword."""
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from nibabel.nicom import csareader
        csa = csareader.get_csa_header(ds, "series")
        if not csa:
            return "no Siemens CSA series header (e.g. enhanced/XA DICOM)"
        items = csa["tags"].get("MrPhoenixProtocol", {}).get("items", [])
        text = items[0] if items else ""
        if isinstance(text, bytes):
            text = text.decode("latin-1")
        m = re.search(r"### ASCCONV BEGIN.*?###(.*)### ASCCONV END", text, re.S)
        body = m.group(1) if m else text
        return [ln.strip() for ln in body.splitlines() if any(k.lower() in ln.lower() for k in keywords)]
    except Exception as exc:
        return f"could not parse CSA header: {exc}"


# ======================================================================= physio search
def physio_search(dirs: list[Path], cfg: dict) -> list[dict]:
    pats = [p.lower() for p in cfg["inventory"]["physio_patterns"]]
    found, seen = [], set()
    for d in dirs:
        if d is None:
            continue
        for p in _walk(Path(d), cfg["inventory"]["search_depth"] + 1):
            if p in seen:
                continue
            if any(fnmatch.fnmatch(p.name.lower(), pat) for pat in pats):
                seen.add(p)
                found.append({"path": str(p), "bytes": p.stat().st_size})
    return found


# ======================================================================= mbme_tedana run folder
def run_dir_check(run_dir: Path | None) -> dict | None:
    if run_dir is None:
        return None
    run_dir = Path(run_dir)
    out = {"run_dir": str(run_dir), "exists": run_dir.is_dir()}
    if not run_dir.is_dir():
        return out

    vu = run_dir / "volumes_used.tsv"
    if vu.exists():
        with open(vu) as f:
            rows = list(csv.reader(f, delimiter="\t"))
        out["volumes_used"] = {"columns": rows[0] if rows else [], "n_rows": max(len(rows) - 1, 0)}
    else:
        out["volumes_used"] = None

    par = _first(run_dir, "motion/mcflirt.par", "mcflirt.par")
    if par is not None:
        m = np.loadtxt(par, ndmin=2)
        out["mcflirt_par"] = {"n_rows": int(m.shape[0]), "n_cols": int(m.shape[1])}
    else:
        out["mcflirt_par"] = None

    import nibabel as nib
    mc = []
    for p in sorted((run_dir / "work").glob("echo-*_desc-mc_bold.nii*")):
        try:
            mc.append({"path": str(p), "shape": tuple(int(s) for s in nib.load(str(p)).shape)})
        except Exception as exc:
            mc.append({"path": str(p), "error": str(exc)})
    out["mc_echoes"] = mc

    out["masks"] = [str(p) for p in sorted(run_dir.rglob("*mask*.nii*"))][:20]

    rj = _first(run_dir, "result.json", "result.json")
    out["result_json_path"] = str(rj) if rj else None
    if rj is not None:
        res = _read_json(rj)
        out["result_json_keys"] = list(res.keys()) if isinstance(res, dict) else None
        out["result_json_interp"] = _find_keys(res, "interp")
        out["result_json_versions"] = _find_keys(res, "version")
    else:
        out["result_json_keys"] = None

    ted = run_dir / "tedana"
    if ted.is_dir():
        files = sorted(p for p in ted.rglob("*") if p.is_file())
        tinfo = {"n_files": len(files)}
        metrics = [p for p in files if p.name.endswith("metrics.tsv")]
        if metrics:
            with open(metrics[0]) as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            tinfo["metrics_file"] = str(metrics[0])
            tinfo["n_components"] = len(rows)
            if rows and "classification" in rows[0]:
                cls = {}
                for r in rows:
                    cls[r["classification"]] = cls.get(r["classification"], 0) + 1
                tinfo["classification_counts"] = cls
            if rows and "classification_tags" in rows[0]:
                tinfo["n_likely_bold_tag"] = sum("Likely BOLD" in (r["classification_tags"] or "") for r in rows)
        out["tedana"] = tinfo
    else:
        out["tedana"] = None
    return out


def _first(root: Path, expected: str, name: str):
    """The file at its expected place, else the first file with that name anywhere under root."""
    p = root / expected
    if p.is_file():
        return p
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def pipeline_script_check(path, cfg: dict) -> dict | None:
    """Hash of mbme_tedana.py and its lines that set interpolation, tedana flags, masks, seeds."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    kws = [k.lower() for k in cfg["inventory"]["pipeline_keywords"]]
    lines = path.read_text(errors="replace").splitlines()
    hits = [(i + 1, ln.rstrip()) for i, ln in enumerate(lines)
            if any(k in ln.lower() for k in kws) and not ln.strip().startswith("#")]
    return {"path": str(path), "exists": True, "sha256": io.file_sha256(path), "n_lines": len(lines),
            "matches": hits[: cfg["inventory"]["pipeline_max_lines"]], "n_matches": len(hits)}


def _find_keys(obj, substring, prefix=""):
    """All (dotted key, value) pairs whose key contains `substring`, at any nesting depth."""
    hits = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if substring.lower() in str(k).lower() and not isinstance(v, (dict, list)):
                hits[path] = v
            hits.update(_find_keys(v, substring, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.update(_find_keys(v, substring, f"{prefix}[{i}]"))
    return hits


# ======================================================================= one subject
def inventory_subject(nifti_dir, cfg: dict, run_dir=None, physio_dirs=(), subject=None,
                      load_data: bool = True, pipeline_script=None) -> dict:
    nifti_dir = Path(nifti_dir)
    files = find_nifti_files(nifti_dir, cfg)
    key, echoes, others = pick_rest_series(files, cfg)
    phase_files = [f for f in files if f["is_phase"]]

    inv = {
        "subject": subject, "nifti_dir": str(nifti_dir),
        "n_nifti_files": len(files),
        "rest_series_key": key,
        "other_bold_series": others,
        "echo_files": [{k: f[k] for k in ("path", "sidecar", "echo", "shape", "n_volumes", "voxel_mm",
                                          "tr_header_s", "datatype", "qform_code", "sform_code") if k in f}
                       for f in echoes],
        "phase_files": [f["path"] for f in phase_files],
        "phase_image_types": sorted({"\\".join(f["image_type"]) for f in phase_files}),
        "magnitude_image_types": sorted({"\\".join(f["image_type"]) for f in echoes}),
        "all_nifti": [{"path": f["path"], "n_volumes": f.get("n_volumes"), "is_phase": f["is_phase"],
                       "echo": f["echo"]} for f in files],
    }
    if echoes:
        aff = [np.asarray(f["affine"]) for f in echoes if "affine" in f]
        inv["affines_identical"] = bool(all(np.allclose(a, aff[0], atol=1e-3) for a in aff))
        shp = [tuple(f["shape"][:3]) for f in echoes if "shape" in f]
        inv["spatial_shapes_identical"] = len(set(shp)) == 1
        te_json = [f["json"].get("EchoTime") for f in echoes]
        inv["te_json_ms"] = [None if t is None else round(1000 * float(t), 3) for t in te_json]
        exp = cfg["acquisition"]["te_ms"]
        tol = cfg["acquisition"]["te_tolerance_ms"]
        inv["te_expected_ms"] = exp
        inv["te_match"] = (len(inv["te_json_ms"]) == len(exp) and None not in inv["te_json_ms"]
                           and all(abs(a - b) <= tol for a, b in zip(inv["te_json_ms"], exp)))
        inv["sidecar"] = sidecar_summary(echoes, cfg)
        inv["volumes"] = volume_checks(echoes, cfg) if load_data else None
    inv["dicom"] = dicom_scan(nifti_dir, cfg)
    inv["physio_files"] = physio_search([nifti_dir, *(Path(p) for p in physio_dirs),
                                         Path(run_dir) if run_dir else None], cfg)
    inv["run"] = run_dir_check(Path(run_dir) if run_dir else None)
    inv["pipeline_script"] = pipeline_script_check(pipeline_script, cfg)
    return inv


# ======================================================================= the table
def _uniq(vals):
    u = []
    for v in vals:
        if v not in u:
            u.append(v)
    return u[0] if len(u) == 1 else u


def build_table(inv: dict, cfg: dict) -> list[tuple[str, str, str, str]]:
    """Rows of (item, value, source, status) for the Phase 0 gate."""
    rows = []
    add = lambda item, val, src, status: rows.append((item, str(val), src, status))
    ef = inv["echo_files"]
    sc = inv.get("sidecar", {"fields": {}, "extra_keys": {}})
    F = sc["fields"]

    add("Rest series", inv["rest_series_key"] or "NOT FOUND", "filenames", "found" if ef else "NOT FOUND")
    add("Magnitude echoes", f"{len(ef)}: " + ", ".join(Path(f["path"]).name for f in ef), "filenames/JSON",
        "confirmed" if len(ef) == 4 else "CHECK")
    if inv["other_bold_series"]:
        add("Other 4D series", ", ".join(inv["other_bold_series"]), "filenames", "info")
    if ef:
        add("TE (ms), JSON", inv["te_json_ms"], "JSON EchoTime",
            "confirmed" if inv["te_match"] else f"MISMATCH vs {inv['te_expected_ms']}")
        add("Volumes per echo", [f.get("n_volumes") for f in ef], "header",
            "equal" if len({f.get('n_volumes') for f in ef}) == 1 else "UNEQUAL (pairing needed)")
        add("Matrix", _uniq([f.get("shape", ())[:3] for f in ef]), "header",
            "confirmed" if _uniq([tuple(f.get("shape", ())[:3]) for f in ef]) == tuple(cfg["acquisition"]["expected_matrix"]) else "CHECK")
        add("Voxel size (mm)", _uniq([tuple(round(v, 3) for v in f.get("voxel_mm", ())) for f in ef]), "header", "found")
        add("Echo affines identical", inv["affines_identical"], "header", "ok" if inv["affines_identical"] else "CHECK")
        add("TR (s)", f"JSON {_uniq(F.get('RepetitionTime', []))}; header {_uniq([f.get('tr_header_s') for f in ef])}",
            "JSON/header", "found")
        add("Field strength (T)", _uniq(F.get("MagneticFieldStrength", [])), "JSON", _st(F, "MagneticFieldStrength"))
        add("Scanner / software", f"{_uniq(F.get('Manufacturer', []))} {_uniq(F.get('ManufacturersModelName', []))}; "
            f"{_uniq(F.get('SoftwareVersions', []))}", "JSON", _st(F, "Manufacturer"))
        add("Sequence", f"{_uniq(F.get('SequenceName', []))}; {_uniq(F.get('PulseSequenceDetails', []))}", "JSON",
            _st(F, "PulseSequenceDetails"))
        add("ImageType (magnitude)", "; ".join(inv["magnitude_image_types"]), "JSON", "found")
        add("MB factor", f"JSON {_uniq(F.get('MultibandAccelerationFactor', []))}; from SliceTiming "
            f"{sc.get('mb_from_slicetiming')} ({sc.get('n_slicetiming')} slice times)", "JSON/derived",
            _st(F, "MultibandAccelerationFactor"))
        add("In-plane acceleration", f"{_uniq(F.get('ParallelReductionFactorInPlane', []))} "
            f"({_uniq(F.get('ParallelAcquisitionTechnique', []))})", "JSON", _st(F, "ParallelReductionFactorInPlane"))
        add("Partial Fourier", _uniq(F.get("PartialFourier", [])), "JSON", _st(F, "PartialFourier"))
        add("Phase-encode direction", f"{_uniq(F.get('PhaseEncodingDirection', []))} (axis "
            f"{_uniq(F.get('PhaseEncodingAxis', []))}; DICOM {_uniq(F.get('InPlanePhaseEncodingDirectionDICOM', []))})",
            "JSON", _st(F, "PhaseEncodingDirection"))
        add("Readout", f"EES {_uniq(F.get('EffectiveEchoSpacing', []))} s; TRT {_uniq(F.get('TotalReadoutTime', []))} s",
            "JSON", _st(F, "EffectiveEchoSpacing"))
        coil = {k: _uniq(F.get(k, [])) for k in ("CoilCombinationMethod", "ReceiveCoilName", "ReceiveCoilActiveElements", "CoilString")}
        coil.update({k: _uniq(v) for k, v in sc["extra_keys"].items() if "coil" in k.lower()})
        add("Coil / combination", coil, "JSON", "found" if any(v != "absent" for v in coil.values()) else "not recorded")
        sl = _uniq(F.get("SpacingBetweenSlices", []))
        th = _uniq(F.get("SliceThickness", []))
        add("Slice thickness / spacing (mm)", f"{th} / {sl}", "JSON", _st(F, "SliceThickness"))
        mb = _uniq(F.get("MultibandAccelerationFactor", []))
        mb = mb if isinstance(mb, (int, float)) else sc.get("mb_from_slicetiming")
        nz = ef[0].get("shape", (0, 0, 0))[2]
        if mb and nz and nz % int(mb) == 0:
            step = nz // int(mb)
            dz = sl if isinstance(sl, (int, float)) else ef[0]["voxel_mm"][2]
            add("MB stack separation", f"{step} slices = {step * dz:.2f} mm (alias partners of slice z: z +/- k*{step})",
                "derived (nz / MB)", "derived; verify")
        else:
            add("MB stack separation", "cannot derive", "-", "ask user")
        disc = {k: _uniq(F.get(k, [])) for k in ("NumberOfVolumesDiscardedByScanner", "NumberOfVolumesDiscardedByUser")}
        add("Volumes discarded (JSON)", disc, "JSON", "found" if any(v != "absent" for v in disc.values()) else "not recorded")

    add("Phase saved", f"{len(inv['phase_files'])} phase file(s)" + (f"; ImageType {inv['phase_image_types']}" if inv["phase_files"] else ""),
        "filenames/ImageType", "YES: ask user how to proceed" if inv["phase_files"] else "no")

    vol = inv.get("volumes")
    if vol:
        pe = vol["per_echo"]
        nf = [e.get("n_trailing_noise_frames") for e in pe]
        add("Noise frames (trailing)", f"{nf} per echo; low-mean frames anywhere {[e.get('n_low_mean_frames_anywhere') for e in pe]}",
            f"heuristic: mean < {cfg['inventory']['noise_frame_rel_threshold']:.0%} of median",
            "YES: ask user how to proceed" if any(nf) else "none")
        add("Non-steady-state (leading)", f"{[e.get('n_nonsteady_leading') for e in pe]} per echo; echo-1 first 5 excess % "
            f"{[round(v, 2) for v in pe[0]['first5_excess_pct']]}",
            f"heuristic: robust z > {cfg['inventory']['nss_robust_z']}", "found" if any(e.get("n_nonsteady_leading") for e in pe) else "none detected")
    else:
        add("Noise frames / non-steady-state", "voxel data not loaded", "-", "NOT CHECKED")

    ph = inv["physio_files"]
    dphys = [s for s in inv["dicom"]["series"] if s.get("is_physio")]
    add("Physio recordings", (", ".join(Path(p["path"]).name for p in ph[:6]) + (" ..." if len(ph) > 6 else "")) if ph or dphys
        else "none found", "filename patterns + DICOM ImageType",
        "YES: ask user how to proceed" if ph or dphys else "none found (ask user where they would be)")

    d = inv["dicom"]
    add("DICOMs", f"{d['n_dicom_files']} files, {len(d['series'])} series read" + (f"; {d['note']}" if d["note"] else ""),
        "file magic", "found" if d["n_dicom_files"] else "none in NIfTI folder")
    asc = [ln for s in d["series"] if isinstance(s.get("ascconv"), list) for ln in s["ascconv"]]
    caipi = [ln for ln in asc if any(k in ln.lower() for k in ("fovshift", "caipi", "shift"))]
    add("CAIPI shift", "; ".join(sorted(set(caipi))[:4]) or "not found", "DICOM ASCCONV (raw)",
        "raw: user to interpret" if caipi else "ask user / protocol PDF")

    r = inv.get("run")
    if r:
        add("mbme_tedana run folder", r["run_dir"], "-", "found" if r["exists"] else "NOT FOUND")
        if r["exists"]:
            add("Motion-corrected echoes", [m.get("shape") for m in r["mc_echoes"]] or "none (needs --keep-intermediates)",
                "work/echo-k_desc-mc_bold", "found" if r["mc_echoes"] else "MISSING")
            add("volumes_used.tsv", r["volumes_used"], "run folder", "found" if r["volumes_used"] else "MISSING")
            add("mcflirt.par", r["mcflirt_par"], "run folder", "found" if r["mcflirt_par"] else "MISSING")
            add("Interpolation (result.json)", r.get("result_json_interp") or "not found", "result.json",
                "found" if r.get("result_json_interp") else "CHECK")
            add("tedana outputs", r["tedana"], "run folder", "found" if r["tedana"] else "MISSING")
    add("Tissue / cavity masks", "source not yet specified", "-", "ask user")
    return rows


def _st(F, key):
    vals = F.get(key, [])
    if not vals or all(v == "absent" for v in vals):
        return "not recorded"
    return "found" if len({json.dumps(v, sort_keys=True) for v in vals}) == 1 else "DIFFERS across echoes"


def render_markdown(inv: dict, cfg: dict) -> str:
    rows = build_table(inv, cfg)
    lines = [f"# Phase 0 audit: {inv.get('subject') or inv['nifti_dir']}", "",
             f"NIfTI folder: `{inv['nifti_dir']}`", "",
             "| Item | Value | Source | Status |", "|---|---|---|---|"]
    for item, val, src, st in rows:
        lines.append(f"| {item} | {val.replace('|', '/')} | {src} | {st} |")
    vol = inv.get("volumes")
    if vol:
        lines += ["", "## Per-echo volume checks", "", "| Echo | Volumes | Trailing noise frames | Leading NSS | First-5 robust z |",
                  "|---|---|---|---|---|"]
        for e in vol["per_echo"]:
            lines.append(f"| {e.get('echo')} | {e.get('n_volumes')} | {e.get('n_trailing_noise_frames')} | "
                         f"{e.get('n_nonsteady_leading')} | {[round(z, 1) for z in e.get('first5_robust_z', [])]} |")
    ps = inv.get("pipeline_script")
    if ps and ps.get("exists"):
        lines += ["", f"## mbme_tedana.py settings (raw lines; {ps['n_matches']} matches, sha256 {ps['sha256'][:12]})", "", "```"]
        lines += [f"{n:5d}: {ln}" for n, ln in ps["matches"]] + ["```"]
    elif ps:
        lines += ["", f"mbme_tedana.py NOT FOUND at `{ps['path']}`"]
    asc = [(s["series_description"], s["ascconv"]) for s in inv["dicom"]["series"] if isinstance(s.get("ascconv"), list) and s["ascconv"]]
    if asc:
        lines += ["", "## Siemens protocol lines (raw, unverified interpretation)", ""]
        desc, lns = asc[0]
        lines += [f"Series: {desc}", "", "```"] + lns[:60] + ["```"]
    return "\n".join(lines) + "\n"


def write_outputs(inv: dict, cfg: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    vm = (inv.get("volumes") or {}).pop("volume_means", None)
    if vm:
        T = max(len(v) for v in vm.values())
        cols = {"volume": list(range(T))}
        for k, v in vm.items():
            cols[k] = list(v) + [float("nan")] * (T - len(v))
        io.write_tsv(out_dir / "volume_means.tsv", cols)
    io.write_json(out_dir / "inventory.json", inv)
    io.write_text(out_dir / "inventory.md", render_markdown(inv, cfg))
    io.write_config_copy(cfg, out_dir)
    inputs = {f"echo{f['echo']}_json": f["sidecar"] for f in inv["echo_files"] if f.get("sidecar")}
    inputs.update({f"echo{f['echo']}_nifti": f["path"] for f in inv["echo_files"]})
    if inv.get("pipeline_script") and inv["pipeline_script"].get("exists"):
        inputs["mbme_tedana_py"] = inv["pipeline_script"]["path"]
    io.write_json(out_dir / "done.json", {"stage": "inventory", **io.provenance(cfg, inputs)})
