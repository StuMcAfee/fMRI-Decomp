"""Reading data, writing outputs safely, and recording provenance.

Every stage of the pipeline uses these helpers so that each output folder records
exactly which inputs, config, code version and package versions produced it.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


# ---------------------------------------------------------------- config
def load_config(path: str | os.PathLike | None = None) -> dict:
    """Read the YAML config (default: configs/default.yaml)."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("extends", None)                     # short override files (sensitivity analyses): extends: default
    if base:
        bpath = DEFAULT_CONFIG if base == "default" else (path.parent / base)
        cfg = merge(load_config(bpath), cfg)
    cfg["_config_path"] = str(path)
    return cfg


def config_hash(cfg: dict) -> str:
    """Hash of the config content (ignores where it was loaded from)."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()


def seed_sequence(cfg: dict, *spawn_key: int) -> np.random.SeedSequence:
    """Reproducible seeds: SeedSequence(project_seed, spawn_key=(subject, run, stage, restart, ...))."""
    return np.random.SeedSequence(int(cfg["project_seed"]), spawn_key=tuple(int(k) for k in spawn_key))


def merge(base: dict, overrides: dict | None) -> dict:
    """Copy of `base` with `overrides` applied (nested dicts merged, other values replaced)."""
    import copy

    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------- hashing / versions
def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_sha(repo: Path = REPO_ROOT) -> str:
    """Commit of this code, with '-dirty' if there are uncommitted changes."""
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def package_versions(names=("numpy", "scipy", "nibabel", "pyyaml", "pydicom", "scikit-learn", "joblib", "threadpoolctl",
                                  "dipy", "mapca", "tedana", "mejoint")) -> dict:
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "not installed"
    return out


def provenance(cfg: dict, inputs: dict | None = None, seeds: dict | None = None,
               hash_inputs: bool = True) -> dict:
    """The record written as done.json next to every stage's outputs."""
    inputs = inputs or {}
    hashed = {}
    for label, p in inputs.items():
        p = Path(p)
        if p.is_file():
            hashed[label] = {"path": str(p), "sha256": file_sha256(p) if hash_inputs else None,
                             "bytes": p.stat().st_size}
        else:
            hashed[label] = {"path": str(p), "sha256": None}
    return {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "git_sha": git_sha(),
        "config_path": cfg.get("_config_path"),
        "config_sha256": config_hash(cfg),
        "packages": package_versions(),
        "inputs": hashed,
        "seeds": seeds or {},
    }


# ---------------------------------------------------------------- atomic writers
def _atomic_write(path: Path, data: bytes) -> None:
    """Write to a temp file in the same folder, then rename: a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path, obj) -> None:
    _atomic_write(Path(path), json.dumps(obj, indent=2, default=_to_jsonable).encode())


def write_text(path, text: str) -> None:
    _atomic_write(Path(path), text.encode())


def write_tsv(path, columns: dict) -> None:
    """Write equal-length columns as a tab-separated table."""
    names = list(columns)
    n = len(next(iter(columns.values()))) if columns else 0
    lines = ["\t".join(names)]
    for i in range(n):
        lines.append("\t".join(_fmt(columns[c][i]) for c in names))
    write_text(path, "\n".join(lines) + "\n")


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "n/a" if not np.isfinite(v) else f"{v:.6g}"
    return str(v)


def write_config_copy(cfg: dict, out_dir) -> None:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    write_text(Path(out_dir) / "config_used.yaml", yaml.safe_dump(clean, sort_keys=False))


# ---------------------------------------------------------------- image loading
def load_4d_float32(path) -> tuple[np.ndarray, "object"]:
    """Load a NIfTI as float32 (scaling applied) without caching a float64 copy."""
    import nibabel as nib
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32), img


def load_masked_echoes(paths, mask3d: np.ndarray, volumes=None, per_echo_hook=None) -> np.ndarray:
    """Masked magnitude data (V, E, T) float32 from one 4D NIfTI per echo.

    volumes: optional list (one index array per echo) selecting and pairing volumes across echoes
    (mbme_tedana's volumes_used.tsv). per_echo_hook(k, data4d_selected) is called with each echo's
    full selected 4D array before it is masked and freed (used for the MP-PCA cross-check)."""
    import nibabel as nib

    out = None
    for k, p in enumerate(paths):
        img = nib.load(str(p))
        if img.shape[:3] != mask3d.shape:
            raise ValueError(f"{p}: grid {img.shape[:3]} differs from the mask {mask3d.shape}")
        d = np.asarray(img.dataobj, dtype=np.float32)
        if volumes is not None:
            d = d[..., np.asarray(volumes[k], int)]
        if per_echo_hook is not None:
            per_echo_hook(k, d)
        if out is None:
            out = np.empty((int(mask3d.sum()), len(paths), d.shape[3]), np.float32)
        out[:, k, :] = d[mask3d]
        del d
    return out


def save_map(values: np.ndarray, mask3d: np.ndarray, ref_img, path, fill=np.nan) -> None:
    """Write masked values (V,) or (V, K) as a 3D or 4D float32 NIfTI on the reference grid (atomic)."""
    import nibabel as nib

    values = np.asarray(values, dtype=np.float32)
    vol = np.full(mask3d.shape + values.shape[1:], fill, np.float32)
    vol[mask3d] = values
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_slope_inter(None, None)
    img = nib.Nifti1Image(vol, ref_img.affine, hdr)
    if vol.ndim == 3:
        img.header.set_data_shape(vol.shape)
    tmp = path.with_name(f".{path.name}.tmp.nii.gz")
    nib.save(img, str(tmp))
    os.replace(tmp, path)


def save_volume(vol: np.ndarray, ref_img, path) -> None:
    """Write a full-grid 3D/4D array as float32 NIfTI on the reference grid (atomic)."""
    import nibabel as nib

    vol = np.asarray(vol, dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_slope_inter(None, None)
    img = nib.Nifti1Image(vol, ref_img.affine, hdr)
    img.header.set_data_shape(vol.shape)
    tmp = path.with_name(f".{path.name}.tmp.nii.gz")
    nib.save(img, str(tmp))
    os.replace(tmp, path)
