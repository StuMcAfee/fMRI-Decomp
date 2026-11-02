"""Phase 0 audit: what was acquired, and what the existing pipeline (mbme_tedana.py) left on disk.

For one subject this module
  1. finds the rest echoes (exactly the series mbme_tedana used, via result.json `input_echo1`),
  2. confirms TEs, volume counts, matrix, TR and the sequence fields named in CLAUDE.md,
  3. looks for saved phase of the same series, physiological logs, noise frames and
     non-steady-state volumes,
  4. reads a sample of DICOM headers (series-numbered subfolder first) for Siemens protocol lines,
  5. checks the run folder: parameters actually used, TE cross-check, motion-corrected echoes
     (volume count, trailing noise frames, non-positive values inside the brain mask).

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
NON_DICOM_EXT = (".nii", ".nii.gz", ".json", ".bval", ".bvec", ".txt", ".csv", ".tsv", ".log",
                 ".py", ".sh", ".zip", ".gz", ".mat", ".png", ".jpg", ".html", ".par")
PHASE_TOKEN = re.compile(r"(_ph|_phase|_part-phase)(?=[._]|$)", re.I)
ECHO_TOKEN = re.compile(r"(_e|_echo-?)(\d+)(?=[._]|$)", re.I)
DCM2NIIX_SERIES = re.compile(r"_(\d{14})_(\d+)_e\d+")     # dcm2niix default: ..._<stamp>_<series>_e1

# result.json fields from mbme_tedana.py worth showing in the audit
RESULT_FIELDS = ("status", "input_echo1", "run_id", "tedana_version", "parameters", "echo_times_ms",
                 "te_source", "n_volumes", "n_vols_per_echo_raw", "volume_pairing",
                 "n_vols_dropped_pairing", "tr_s", "voxel_size_mm", "first_volume_signal_ratio",
                 "mean_fd_mm", "max_fd_mm", "n_vols_fd_gt_0p5mm", "mask_voxels",
                 "n_components_total", "n_likely_bold", "n_accepted_total", "n_rejected",
                 "varex_likely_bold_pct", "qc_no_provisional_accept_fallback", "focal_status")


# ======================================================================= file discovery
def _stem(p: Path) -> str:
    for s in NIFTI_SUFFIXES:
        if p.name.endswith(s):
            return p.name[: -len(s)]
    return p.stem


def series_key_of(stem: str) -> str:
    """File stem with echo and phase tokens removed: all echoes of one series share it."""
    return PHASE_TOKEN.sub("", ECHO_TOKEN.sub("", stem))


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
            except (PermissionError, OSError):
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
            info = {"error": f"could not read header: {exc}", "n_volumes": 0, "shape": ()}

        out.append({"path": str(p), "stem": stem, "sidecar": str(sidecar) if sidecar.exists() else None,
                    "json": meta, "echo": echo, "is_phase": is_phase, "series_key": series_key_of(stem),
                    "image_type": image_type, **info})
    return out


def _tr_seconds(hdr, zooms):
    if len(zooms) < 4:
        return None
    unit = hdr.get_xyzt_units()[1]
    tr = float(zooms[3])
    return tr / 1000.0 if unit == "msec" else tr / 1e6 if unit == "usec" else tr


def pick_rest_series(files: list[dict], cfg: dict, echo1_path=None):
    """Choose the multi-echo BOLD series.

    If mbme_tedana's result.json names the echo-1 file, use exactly that series.
    Otherwise: most magnitude echoes, then most volumes. Returns (key, echoes, others, how).
    """
    minvol = cfg["inventory"]["min_volumes_bold"]
    groups: dict[str, list[dict]] = {}
    for f in files:
        if not f["is_phase"] and f.get("n_volumes", 0) >= minvol:
            groups.setdefault(f["series_key"], []).append(f)
    if not groups:
        return None, [], [], "none found"
    how = "most echoes, then most volumes (heuristic)"
    best = max(groups, key=lambda k: (len(groups[k]), max(f["n_volumes"] for f in groups[k])))
    if echo1_path is not None:
        want = series_key_of(_stem(Path(echo1_path)))
        if want in groups:
            best, how = want, "result.json input_echo1"
        else:
            how = f"WARNING: result.json echo-1 series '{want}' not found here; fell back to heuristic"
    echoes = sorted(groups[best], key=lambda f: (f["echo"] is None, f["echo"] or 0, f["path"]))
    others = [k for k in groups if k != best]
    return best, echoes, others, how


def series_number(echo_files: list[dict]):
    """Series number from the JSON sidecar, else from the dcm2niix file name."""
    if not echo_files:
        return None
    sn = echo_files[0]["json"].get("SeriesNumber")
    if sn is not None:
        return int(sn)
    m = DCM2NIIX_SERIES.search(echo_files[0]["stem"])
    return int(m.group(2)) if m else None


def split_phase(files: list[dict], echoes: list[dict], cfg: dict):
    """Phase images that can belong to the rest series (4D, same grid and volume count) vs others
    (e.g. field-map phase). On Siemens the phase is usually saved as its own series number."""
    if not echoes:
        return [], [f for f in files if f["is_phase"]]
    grid = tuple(echoes[0].get("shape", ())[:3])
    nvol = {f.get("n_volumes") for f in echoes}
    minvol = cfg["inventory"]["min_volumes_bold"]
    rest_ph, other_ph = [], []
    for f in files:
        if not f["is_phase"]:
            continue
        ok = tuple(f.get("shape", ())[:3]) == grid and f.get("n_volumes", 0) >= minvol and f.get("n_volumes") in nvol
        (rest_ph if ok else other_ph).append(f)
    return rest_ph, other_ph


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


def mc_echo_checks(run: dict, cfg: dict) -> list[dict] | None:
    """Motion-corrected echoes: volume count, trailing noise frames, and voxels inside the
    pipeline's brain mask that ever reach <= 0 (spline resampling can undershoot; the
    log-ratio front end cannot use such voxel-echoes)."""
    import nibabel as nib

    if not run or not run.get("mc_echoes"):
        return None
    mpath = run.get("brain_mask")
    mask = np.asarray(nib.load(mpath).dataobj).astype(bool).ravel() if mpath else None
    out = []
    for m in run["mc_echoes"]:
        if "error" in m:
            out.append(m)
            continue
        data, _ = io.load_4d_float32(m["path"])
        T = data.shape[3]
        flat = data.reshape(-1, T)
        whole = flat.mean(axis=0)
        low = whole < cfg["inventory"]["noise_frame_rel_threshold"] * np.median(whole)
        rec = {"path": m["path"], "n_volumes": T, "n_trailing_noise_frames": _leading_run(low[::-1])}
        if mask is not None and mask.size == flat.shape[0]:
            vals = flat[mask]
            nonpos = (vals <= 0).any(axis=1)
            rec.update(mask_voxels=int(mask.sum()), n_mask_voxels_any_nonpositive=int(nonpos.sum()),
                       frac_mask_voxels_any_nonpositive=float(nonpos.mean()),
                       min_value_in_mask=float(vals.min()))
        out.append(rec)
        del data, flat
    return out


# ======================================================================= sidecar summary
def sidecar_summary(echo_files: list[dict], cfg: dict) -> dict:
    icfg = cfg["inventory"]
    fields = {}
    for name in icfg["json_fields"]:
        fields[name] = [f["json"].get(name, "absent") for f in echo_files]
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
def _is_dicom(p: Path) -> bool:
    if p.name.lower().endswith(NON_DICOM_EXT) or not p.is_file():
        return False
    try:
        with open(p, "rb") as fh:
            head = fh.read(132)
        return len(head) == 132 and head[128:132] == b"DICM"
    except OSError:
        return False


def candidate_dicom_folders(nifti_dir: Path, series: int | None) -> list[Path]:
    """Same search order as mbme_tedana.py: the NIfTI folder, then subfolders whose name starts
    with the series number, then subfolders mentioning REST, then the rest."""
    try:
        subdirs = sorted(p for p in Path(nifti_dir).iterdir() if p.is_dir())
    except OSError:
        subdirs = []

    def priority(p: Path) -> int:
        n = p.name.upper()
        if series is not None and re.match(rf"^0*{series}(\D|$)", n):
            return 0
        return 1 if "REST" in n else 2

    return [Path(nifti_dir)] + sorted(subdirs, key=priority)


def dicom_scan(nifti_dir: Path, cfg: dict, series: int | None = None) -> dict:
    """Read a random sample of DICOM headers, series-numbered folder first, and stop once the
    rest series (magnitude) has been seen. Needs pydicom; otherwise DICOMs are only counted."""
    icfg = cfg["inventory"]
    rng = np.random.default_rng(cfg.get("project_seed", 0))
    out = {"series_hint": series, "folders_scanned": [], "n_files_listed": 0, "series": [], "note": None}
    try:
        import pydicom
    except ImportError:
        pydicom = None
        out["note"] = "pydicom not installed: DICOM headers not read"

    groups: dict[tuple, dict] = {}
    for folder in candidate_dicom_folders(nifti_dir, series)[: icfg["dicom_max_folders"]]:
        try:
            files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.lower().endswith(NON_DICOM_EXT))
        except OSError:
            continue
        if not files:
            continue
        out["n_files_listed"] += len(files)
        n_dcm = 0
        for j in rng.permutation(len(files))[: icfg["dicom_max_headers_per_folder"]]:
            p = files[j]
            if not _is_dicom(p):
                continue
            n_dcm += 1
            if pydicom is None:
                continue
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                itype = tuple(str(x) for x in ds.get("ImageType", []))
                key = (int(ds.get("SeriesNumber", -1) or -1), str(ds.get("SeriesDescription", "")), itype,
                       str(ds.get("EchoNumbers", "")))
            except Exception:
                continue
            g = groups.setdefault(key, {"count": 0, "path": p, "ds": ds, "folder": str(folder)})
            g["count"] += 1
        out["folders_scanned"].append({"folder": str(folder), "n_files": len(files), "n_dicom_in_sample": n_dcm})
        if series is not None and any(k[0] == series for k in groups):
            break

    for (snum, sdesc, itype, echo), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][3])):
        ds = g["ds"]
        out["series"].append({
            "series_number": snum, "series_description": sdesc, "image_type": list(itype),
            "echo_numbers": echo, "n_headers_read": g["count"], "folder": g["folder"],
            "example": str(g["path"]), "sop_class": str(ds.get("SOPClassUID", "")),
            "EchoTime_ms": _f(ds.get("EchoTime")), "SliceThickness": _f(ds.get("SliceThickness")),
            "SpacingBetweenSlices": _f(ds.get("SpacingBetweenSlices")),
            "ImageComments": str(ds.get("ImageComments", "")) or None,
            "PATModeText_0051_1011": _private(ds, 0x0051, 0x1011),
            "CoilString_0051_100F": _private(ds, 0x0051, 0x100F),
            "is_phase": any(t.upper() in ("P", "PHASE") for t in itype),
            "is_physio": "PHYSIO" in [t.upper() for t in itype] or "physio" in sdesc.lower(),
            "ascconv": _ascconv_lines(ds, icfg["ascconv_keywords"]),
        })
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


def _ascconv_text(ds) -> str | None:
    """Siemens protocol text: CSA series header (classic DICOM), else private tag (0021,1019)
    (enhanced/XA DICOM; location unverified on your data)."""
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from nibabel.nicom import csareader
        csa = csareader.get_csa_header(ds, "series")
        if csa:
            items = csa["tags"].get("MrPhoenixProtocol", {}).get("items", [])
            if items:
                t = items[0]
                return t.decode("latin-1") if isinstance(t, bytes) else str(t)
    except Exception:
        pass
    try:
        el = ds.get((0x0021, 0x1019))
        if el is not None:
            v = el.value
            t = v.decode("latin-1", errors="replace") if isinstance(v, bytes) else str(v)
            if "ASCCONV" in t:
                return t
    except Exception:
        pass
    return None


def _ascconv_lines(ds, keywords) -> list[str] | str:
    """Protocol lines that mention any keyword, reported raw for the user to interpret."""
    text = _ascconv_text(ds)
    if not text:
        return "no Siemens protocol (ASCCONV) text found"
    m = re.search(r"### ASCCONV BEGIN.*?###(.*)### ASCCONV END", text, re.S)
    body = m.group(1) if m else text
    return [ln.strip() for ln in body.splitlines() if any(k.lower() in ln.lower() for k in keywords)]


# ======================================================================= physio search
def physio_search(dirs: list[Path], cfg: dict) -> list[dict]:
    pats = [p.lower() for p in cfg["inventory"]["physio_patterns"]]
    found, seen = [], set()
    for d in dirs:
        if d is None:
            continue
        for p in _walk(Path(d), cfg["inventory"]["search_depth"]):
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

    rj = _first(run_dir, "result.json", "result.json")
    out["result_json_path"] = str(rj) if rj else None
    res = _read_json(rj) if rj else {}
    out["result"] = {k: res.get(k) for k in RESULT_FIELDS if k in res}
    out["result_json_interp"] = _find_keys(res, "interp")

    ej = run_dir / "echo_info.json"
    if ej.exists():
        e = _read_json(ej).get("te_info", {})
        out["echo_info"] = {"source": e.get("source"), "tes_ms": e.get("tes_ms"),
                            "json_tes_ms": (e.get("json") or {}).get("tes_ms"),
                            "dicom_tes_ms": (e.get("dicom") or {}).get("tes_ms"),
                            "dicom_folder": (e.get("dicom") or {}).get("folder"),
                            "dicom_error": e.get("dicom_error")}
    else:
        out["echo_info"] = None

    vu = run_dir / "volumes_used.tsv"
    if vu.exists():
        with open(vu) as f:
            rows = list(csv.reader(f, delimiter="\t"))
        out["volumes_used"] = {"columns": rows[0] if rows else [], "n_rows": max(len(rows) - 1, 0)}
        if len(rows) > 1:
            first = [int(x) for x in rows[1]]
            last = [int(x) for x in rows[-1]]
            out["volumes_used"].update(first_row=first, last_row=last)
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
    work = run_dir / "work"
    out["work_files"] = sorted(p.name for p in work.iterdir())[:30] if work.is_dir() else None

    bm = run_dir / "brain_mask.nii.gz"
    out["brain_mask"] = str(bm) if bm.exists() else None

    ted = run_dir / "tedana"
    if ted.is_dir():
        files = sorted(p for p in ted.rglob("*") if p.is_file())
        tinfo = {"n_files": len(files)}
        metrics = [p for p in files if p.name.endswith("desc-tedana_metrics.tsv")] or \
                  [p for p in files if p.name.endswith("metrics.tsv")]
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
                tinfo["n_likely_bold_tag"] = sum(
                    "Likely BOLD" in {t.strip() for t in (r["classification_tags"] or "").split(",")} for r in rows)
        for want in ("desc-ICA_mixing.tsv", "desc-optcom_bold.nii.gz", "desc-adaptiveGoodSignal_mask.nii.gz",
                     "T2starmap.nii.gz"):
            tinfo[want] = any(p.name.endswith(want) for p in files)
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
    snap = io.REPO_ROOT / "external" / "mbme_tedana.py"
    same = None
    if snap.exists():
        same = snap.read_text(errors="replace").splitlines() == lines
    return {"path": str(path), "exists": True, "sha256": io.file_sha256(path), "n_lines": len(lines),
            "matches_repo_snapshot": same,
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
    run = run_dir_check(Path(run_dir) if run_dir else None)
    echo1 = ((run or {}).get("result") or {}).get("input_echo1")
    if nifti_dir is None:
        if not echo1:
            raise ValueError("--nifti-dir not given and result.json has no input_echo1")
        nifti_dir = Path(echo1).parent
    nifti_dir = Path(nifti_dir)

    files = find_nifti_files(nifti_dir, cfg)
    key, echoes, others, how = pick_rest_series(files, cfg, echo1)
    rest_ph, other_ph = split_phase(files, echoes, cfg)
    snum = series_number(echoes)

    inv = {
        "subject": subject, "nifti_dir": str(nifti_dir),
        "n_nifti_files": len(files),
        "rest_series_key": key, "rest_series_how": how, "series_number": snum,
        "other_bold_series": others,
        "echo_files": [{k: f[k] for k in ("path", "sidecar", "echo", "shape", "n_volumes", "voxel_mm",
                                          "tr_header_s", "datatype", "qform_code", "sform_code") if k in f}
                       for f in echoes],
        "phase_files": [f["path"] for f in rest_ph],
        "phase_image_types": sorted({"\\".join(f["image_type"]) for f in rest_ph}),
        "other_phase_files": [f["path"] for f in other_ph],
        "magnitude_image_types": sorted({"\\".join(f["image_type"]) for f in echoes}),
        "all_nifti": [{"path": f["path"], "n_volumes": f.get("n_volumes"), "is_phase": f["is_phase"],
                       "echo": f["echo"], "SeriesDescription": f["json"].get("SeriesDescription")} for f in files],
    }
    if echoes:
        aff = [np.asarray(f["affine"]) for f in echoes if "affine" in f]
        inv["affines_identical"] = bool(all(np.allclose(a, aff[0], atol=1e-3) for a in aff))
        te_json = [f["json"].get("EchoTime") for f in echoes]
        inv["te_json_ms"] = [None if t is None else round(1000 * float(t), 3) for t in te_json]
        exp = cfg["acquisition"]["te_ms"]
        tol = cfg["acquisition"]["te_tolerance_ms"]
        inv["te_expected_ms"] = exp
        inv["te_match"] = (len(inv["te_json_ms"]) == len(exp) and None not in inv["te_json_ms"]
                           and all(abs(a - b) <= tol for a, b in zip(inv["te_json_ms"], exp)))
        inv["sidecar"] = sidecar_summary(echoes, cfg)
        inv["volumes"] = volume_checks(echoes, cfg) if load_data else None
    inv["dicom"] = dicom_scan(nifti_dir, cfg, snum)
    inv["physio_files"] = physio_search([nifti_dir, *(Path(p) for p in physio_dirs),
                                         Path(run_dir) if run_dir else None], cfg)
    inv["run"] = run
    inv["mc_checks"] = mc_echo_checks(run, cfg) if load_data else None
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

    add("Rest series", f"{inv['rest_series_key'] or 'NOT FOUND'} (series {inv.get('series_number')})",
        f"chosen by: {inv.get('rest_series_how')}", "found" if ef and not inv.get("rest_series_how", "").startswith("WARN") else "CHECK")
    add("Magnitude echoes", f"{len(ef)}: " + ", ".join(Path(f["path"]).name for f in ef), "filenames/JSON",
        "confirmed" if len(ef) == 4 else "CHECK")
    if inv["other_bold_series"]:
        add("Other 4D series in folder", ", ".join(inv["other_bold_series"]), "filenames", "info")
    if ef:
        add("TE (ms), JSON", inv["te_json_ms"], "JSON EchoTime",
            "confirmed" if inv["te_match"] else f"MISMATCH vs {inv['te_expected_ms']}")
        add("Volumes per echo (raw)", [f.get("n_volumes") for f in ef], "header",
            "equal" if len({f.get('n_volumes') for f in ef}) == 1 else "UNEQUAL (pairing needed)")
        add("Matrix", _uniq([tuple(f.get("shape", ())[:3]) for f in ef]), "header",
            "confirmed" if _uniq([tuple(f.get("shape", ())[:3]) for f in ef]) == tuple(cfg["acquisition"]["expected_matrix"]) else "CHECK")
        add("Voxel size (mm)", _uniq([tuple(round(v, 3) for v in f.get("voxel_mm", ())) for f in ef]), "header", "found")
        add("Echo affines identical", inv["affines_identical"], "header", "ok" if inv["affines_identical"] else "CHECK")
        add("TR (s)", f"JSON {_uniq(F.get('RepetitionTime', []))}; header {_uniq([f.get('tr_header_s') for f in ef])}",
            "JSON/header", "found")
        add("Series description / protocol", f"{_uniq(F.get('SeriesDescription', []))} / {_uniq(F.get('ProtocolName', []))}",
            "JSON", _st(F, "SeriesDescription"))
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

    dph = [s for s in inv["dicom"]["series"] if s.get("is_phase")]
    add("Phase saved (rest series)",
        f"{len(inv['phase_files'])} matching 4D phase NIfTI(s)" + (f"; ImageType {inv['phase_image_types']}" if inv["phase_files"] else "")
        + (f"; phase DICOM series {sorted({s['series_number'] for s in dph})}" if dph else ""),
        "filenames/ImageType; DICOM sample",
        "YES: ask user how to proceed" if inv["phase_files"] else ("CHECK: phase DICOMs seen" if dph else "no"))
    if inv["other_phase_files"]:
        add("Other phase images (e.g. field map)", ", ".join(Path(p).name for p in inv["other_phase_files"][:4]),
            "filenames", "info")

    vol = inv.get("volumes")
    if vol:
        pe = vol["per_echo"]
        nf = [e.get("n_trailing_noise_frames") for e in pe]
        add("Noise frames (trailing, raw)", f"{nf} per echo; low-mean frames anywhere {[e.get('n_low_mean_frames_anywhere') for e in pe]}",
            f"heuristic: mean < {cfg['inventory']['noise_frame_rel_threshold']:.0%} of median",
            "YES: ask user how to proceed" if any(nf) else "none")
        add("Non-steady-state (leading, raw)", f"{[e.get('n_nonsteady_leading') for e in pe]} per echo; echo-1 first 5 excess % "
            f"{[round(v, 2) for v in pe[0]['first5_excess_pct']]}",
            f"heuristic: robust z > {cfg['inventory']['nss_robust_z']}", "found" if any(e.get("n_nonsteady_leading") for e in pe) else "none detected")
    else:
        add("Noise frames / non-steady-state", "voxel data not loaded", "-", "NOT CHECKED")

    ph = inv["physio_files"]
    dphys = [s for s in inv["dicom"]["series"] if s.get("is_physio")]
    add("Physio recordings", (", ".join(Path(p["path"]).name for p in ph[:6]) + (" ..." if len(ph) > 6 else "")) if ph or dphys
        else "none found", "filename patterns + DICOM ImageType",
        "found: CONFLICTS with user report" if ph or dphys else "none (consistent with user report)")

    d = inv["dicom"]
    add("DICOMs", f"{d['n_files_listed']} files listed in {len(d['folders_scanned'])} folder(s); "
        f"{len(d['series'])} header group(s) read" + (f"; {d['note']}" if d["note"] else ""),
        "random sample, series folder first", "found" if d["series"] else "none read")
    asc = [ln for s in d["series"] if isinstance(s.get("ascconv"), list) for ln in s["ascconv"]]
    caipi = [ln for ln in asc if any(k in ln.lower() for k in ("fovshift", "caipi", "shift"))]
    add("CAIPI shift", "; ".join(sorted(set(caipi))[:4]) or "not found", "DICOM ASCCONV (raw)",
        "raw: user to interpret" if caipi else "unknown (no protocol PDF)")

    r = inv.get("run")
    if r:
        add("mbme_tedana run folder", r["run_dir"], "-", "found" if r["exists"] else "NOT FOUND")
        if r["exists"]:
            res = r.get("result") or {}
            add("Run status / tedana version", f"{res.get('status')} / {res.get('tedana_version')}", "result.json",
                "ok" if res.get("status") == "ok" else "CHECK")
            add("Parameters used", res.get("parameters"), "result.json", "found" if res.get("parameters") else "MISSING")
            ei = r.get("echo_info") or {}
            add("TE cross-check (pipeline)", f"used {ei.get('tes_ms')} from {ei.get('source')}; JSON {ei.get('json_tes_ms')}; "
                f"DICOM {ei.get('dicom_tes_ms')}" + (f"; DICOM error: {ei.get('dicom_error')}" if ei.get("dicom_error") else ""),
                "echo_info.json", "found" if ei else "MISSING")
            add("Volumes (pipeline)", f"raw {res.get('n_vols_per_echo_raw')} -> used {res.get('n_volumes')} "
                f"({res.get('volume_pairing')}); volumes_used.tsv {r['volumes_used']}", "result.json / volumes_used.tsv",
                "found" if r["volumes_used"] else "MISSING")
            add("First-volume signal ratio", res.get("first_volume_signal_ratio"), "result.json (pipeline check)",
                "CHECK: > 1.10" if (res.get("first_volume_signal_ratio") or 0) > 1.10 else "ok")
            add("Motion", f"mean FD {res.get('mean_fd_mm')} mm, max {res.get('max_fd_mm')} mm, "
                f"{res.get('n_vols_fd_gt_0p5mm')} vols > 0.5 mm; mcflirt.par {r['mcflirt_par']}",
                "result.json (unfiltered Power FD)", "found" if r["mcflirt_par"] else "MISSING")
            mcc = inv.get("mc_checks") or []
            add("Motion-corrected echoes", [m.get("shape") for m in r["mc_echoes"]] or "none (needs --keep-intermediates)",
                "work/echo-k_desc-mc_bold", "found" if r["mc_echoes"] else "MISSING")
            if mcc:
                add("MC echoes: trailing noise frames", [m.get("n_trailing_noise_frames") for m in mcc], "heuristic",
                    "YES: pipeline included them" if any(m.get("n_trailing_noise_frames") for m in mcc) else "none")
                add("MC echoes: brain-mask voxels ever <= 0", [m.get("n_mask_voxels_any_nonpositive") for m in mcc],
                    f"brain_mask.nii.gz ({mcc[0].get('mask_voxels')} voxels)", "info (log-ratio must exclude these)")
            add("Interpolation", r.get("result_json_interp") or "not found", "result.json", "found" if r.get("result_json_interp") else "CHECK")
            add("Pipeline brain mask", r.get("brain_mask") or "not found", "run folder", "found" if r.get("brain_mask") else "MISSING")
            t = r.get("tedana") or {}
            add("tedana outputs", {k: t.get(k) for k in ("n_components", "classification_counts", "n_likely_bold_tag",
                                                          "desc-ICA_mixing.tsv", "T2starmap.nii.gz")},
                "tedana/ folder", "found" if t else "MISSING")
            add("Likely BOLD (pipeline count)", res.get("n_likely_bold"), "result.json", "info")
    add("Tissue / cavity masks", "no source exists (user)", "-", "to derive before Phase 5")

    ps = inv.get("pipeline_script")
    if ps:
        add("mbme_tedana.py", f"sha256 {ps.get('sha256', '')[:12]}; matches repo snapshot: {ps.get('matches_repo_snapshot')}",
            "file", "found" if ps.get("exists") else "NOT FOUND")
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
        lines += ["", "## Per-echo volume checks (raw echoes)", "",
                  "| Echo | Volumes | Trailing noise frames | Leading NSS | First-5 robust z |", "|---|---|---|---|---|"]
        for e in vol["per_echo"]:
            lines.append(f"| {e.get('echo')} | {e.get('n_volumes')} | {e.get('n_trailing_noise_frames')} | "
                         f"{e.get('n_nonsteady_leading')} | {[round(z, 1) for z in e.get('first5_robust_z', [])]} |")
    d = inv["dicom"]
    if d["series"]:
        lines += ["", "## DICOM header groups (sample)", "", "| Series | Description | ImageType | Echo | TE (ms) | Headers read |",
                  "|---|---|---|---|---|---|"]
        for s in d["series"][:30]:
            lines.append(f"| {s['series_number']} | {s['series_description']} | {'/'.join(s['image_type'])} | "
                         f"{s['echo_numbers']} | {s['EchoTime_ms']} | {s['n_headers_read']} |")
    asc = [(s["series_description"], s["ascconv"]) for s in d["series"]
           if isinstance(s.get("ascconv"), list) and s["ascconv"] and s["series_number"] == inv.get("series_number")]
    asc = asc or [(s["series_description"], s["ascconv"]) for s in d["series"] if isinstance(s.get("ascconv"), list) and s["ascconv"]]
    if asc:
        desc, lns = asc[0]
        lines += ["", "## Siemens protocol lines (raw, interpretation unverified)", "", f"Series: {desc}", "", "```"] + lns[:60] + ["```"]
    ps = inv.get("pipeline_script")
    if ps and ps.get("exists"):
        lines += ["", f"## mbme_tedana.py settings (raw lines; {ps['n_matches']} matches)", "", "```"]
        lines += [f"{n:5d}: {ln}" for n, ln in ps["matches"]] + ["```"]
    elif ps:
        lines += ["", f"mbme_tedana.py NOT FOUND at `{ps['path']}`"]
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
    r = inv.get("run") or {}
    if r.get("result_json_path"):
        inputs["result_json"] = r["result_json_path"]
    if inv.get("pipeline_script") and inv["pipeline_script"].get("exists"):
        inputs["mbme_tedana_py"] = inv["pipeline_script"]["path"]
    io.write_json(out_dir / "done.json", {"stage": "inventory", **io.provenance(cfg, inputs)})
