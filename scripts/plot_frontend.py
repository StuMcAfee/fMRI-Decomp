"""QC sheet of the Phase 2 front end on one real run (Phase 2 gate evidence).

Rows: mean signal (echo 1, echo 4), echo-4 SNR, usable echoes, T2*, sigma, per-voxel / smooth sigma,
residual energy against the smooth sigma (1 = model holds), static mono-exponential misfit, and the
MP-PCA sigma (echo 1) if computed. Columns: axial slices from the bottom of the brain upwards (the
first two are at the posterior-fossa / skull-base level) and a mid-sagittal slice. Axial images are
in neurological orientation (subject's right on the image's right, marked R; anterior up); the
sagittal image has anterior on the right. Bottom: frame-wise residual energy and FD (two panels,
one axis each).

Usage:  PYTHONPATH=src python scripts/plot_frontend.py <frontend output folder> [OUT.png]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8322f"])
GRAY = LinearSegmentedColormap.from_list("gray_seq", ["#0b0b0b", "#f0efec"])
STEPS5 = matplotlib.colors.ListedColormap(["#d03b3b", "#eda100", "#cde2fb", "#6da7ec", "#184f95"])   # 0..4 usable echoes


def load(folder: Path, name: str):
    p = folder / f"{name}.nii.gz"
    if not p.exists():
        return None
    img = nib.as_closest_canonical(nib.load(str(p)))
    return np.asarray(img.dataobj, dtype=np.float32)


def main(argv=None):
    argv = argv or sys.argv[1:]
    folder = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else folder / "frontend_qc.png"
    summ = json.loads((folder / "summary.json").read_text())
    mu = load(folder, "mu")
    brain = np.isfinite(mu[..., 0]) & (mu[..., 0] > 0)
    zs = np.where(brain.any(axis=(0, 1)))[0]
    zlo, zhi = zs.min(), zs.max()
    fracs = [0.06, 0.16, 0.30, 0.50, 0.72]
    zcuts = [int(round(zlo + f * (zhi - zlo))) for f in fracs]
    xmid = int(round(np.mean(np.where(brain.any(axis=(1, 2)))[0])))

    rows = []
    snr = load(folder, "snr")
    rows.append(("Mean signal, echo 1", mu[..., 0], GRAY, {"vmin": 0, "vmax": np.nanpercentile(mu[..., 0][brain], 99)}, "a.u."))
    rows.append(("Mean signal, echo 4", mu[..., 3], GRAY, {"vmin": 0, "vmax": np.nanpercentile(mu[..., 3][brain], 99)}, "a.u."))
    rows.append(("Echo-4 SNR (mu / sigma)", snr[..., 3], SEQ, {"vmin": 0, "vmax": np.nanpercentile(snr[..., 3][brain], 98)}, "mu/sigma"))
    rows.append(("Usable echoes", load(folder, "n_usable_echoes"), STEPS5,
                 {"norm": BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 5)}, "count"))
    rows.append(("T2* (mono-exponential fit)", load(folder, "t2s_ms"), SEQ, {"vmin": 10, "vmax": 90}, "ms"))
    sig = load(folder, "sigma")
    rows.append(("Thermal sigma (residual-based)", sig, SEQ, {"vmin": 0, "vmax": np.nanpercentile(sig[brain], 99)}, "a.u."))
    rows.append(("Per-voxel / smooth sigma", load(folder, "sigma_ratio_voxel_smooth"), DIV, {"norm": TwoSlopeNorm(1.0, 0.6, 1.4)}, "ratio"))
    rows.append(("Residual energy vs smooth sigma (1 = model holds)", load(folder, "resid_excess"), DIV,
                 {"norm": TwoSlopeNorm(1.0, 0.4, 3.0)}, "ratio"))
    rows.append(("Static mono-exponential misfit", load(folder, "static_misfit"), SEQ, {"vmin": 0, "vmax": 0.05}, "log units"))
    mp = load(folder, "mppca_sigma")
    if mp is not None:
        rows.append(("MP-PCA sigma, echo 1 (raw; sampled slices)", mp[..., 0], SEQ,
                     {"vmin": 0, "vmax": np.nanpercentile(sig[brain], 99)}, "a.u."))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
    ncol = len(zcuts) + 1
    fig = plt.figure(figsize=(1.75 * ncol + 1.2, 1.6 * len(rows) + 3.2))
    gs = fig.add_gridspec(len(rows) + 2, ncol + 1, width_ratios=[1] * ncol + [0.08],
                          height_ratios=[1] * len(rows) + [0.9, 0.9], hspace=0.35, wspace=0.05)
    for i, (title, vol, cmap, kw, unit) in enumerate(rows):
        vol = np.where(brain, vol, np.nan) if vol is not None else None
        im = None
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if vol is None:
                continue
            img = vol[:, :, zcuts[j]].T if j < len(zcuts) else vol[xmid, :, :].T
            im = ax.imshow(img, origin="lower", cmap=cmap, interpolation="nearest", **kw)
            if i == 0:
                lab = f"axial z={zcuts[j]}" + (" (PF level)" if j < 2 else "") if j < len(zcuts) else f"sagittal x={xmid}"
                ax.set_title(lab, fontsize=7, color=INK2)
            if j == 0:
                ax.set_ylabel(title, fontsize=7, color=INK, rotation=0, ha="right", va="center", labelpad=6)
                ax.text(0.97, 0.97, "R", color=MUTED, fontsize=6, va="top", ha="right", transform=ax.transAxes)
        if im is not None:
            cax = fig.add_subplot(gs[i, ncol])
            cb = fig.colorbar(im, cax=cax)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=6, length=0, colors=INK2)
            cb.set_label(unit, fontsize=6, color=INK2)

    fr = np.genfromtxt(folder / "frames.tsv", delimiter="\t", names=True, dtype=None, encoding=None)
    t = np.arange(len(fr))
    ax = fig.add_subplot(gs[len(rows), :ncol])
    ax.plot(t, fr["residual_energy"].astype(float), color="#2a78d6", lw=0.9)
    ax.set_ylabel("residual energy\n(brain mean)", fontsize=7)
    ax.set_xticklabels([])
    ax.grid(axis="y", color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax2 = fig.add_subplot(gs[len(rows) + 1, :ncol], sharex=ax)
    try:
        fd = fr["fd_mm"].astype(float)
        ax2.plot(t, fd, color="#52514e", lw=0.9)
    except (ValueError, TypeError):
        ax2.text(0.5, 0.5, "FD not available", transform=ax2.transAxes, ha="center", color=MUTED)
    ax2.set_ylabel("FD (mm,\nunfiltered)", fontsize=7)
    ax2.set_xlabel("frame", fontsize=7)
    ax2.grid(axis="y", color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    rho = summ.get("frame_energy_vs_fd_spearman")
    ax.set_title(f"Frame-wise residual energy and FD" + (f" (Spearman {rho:.2f})" if rho is not None else ""),
                 fontsize=7, color=INK, loc="left")

    fig.suptitle(f"Phase 2 front end: {summ['run_id']} ({summ['source']} echoes); {summ['n_voxels']} voxels, "
                 f"{summ['n_volumes']} volumes", fontsize=8, color=INK2, x=0.01, ha="left", y=0.995)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
