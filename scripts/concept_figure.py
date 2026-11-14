#!/usr/bin/env python
"""Concept figure (Nature-style schematic) for the multi-echo separation and route B.

  PYTHONPATH=src python scripts/concept_figure.py --out <folder>

Writes concept_figure.pdf (vector, fonts embedded), .svg and .png (300 dpi). Everything is drawn from synthetic
signals, closed-form noise calculations and the bundled Schaefer atlas; the only study numbers are aggregate
results already in docs/decisions.md (injection recovery counts).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: E402

MM = 1 / 25.4
TE = np.array([16.20, 31.22, 46.24, 61.26])
RATIOS = np.array([1.0, 0.721, 0.631, 0.560])
BOLD, S0C, NOISE, INK, INK2, RULE = "#2a78d6", "#eb6834", "#9a9893", "#111111", "#55534f", "#d9d7d2"
NETS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETCOL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32b2b"])

plt.rcParams.update({
    "font.family": "Inter", "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6.5, "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5, "legend.fontsize": 5.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.titlepad": 3})


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=8, fontweight="bold", color=INK, ha="left", va="top")


def bandlimited(rng, n, T, tr=0.75, band=(0.01, 0.1)):
    f = np.fft.rfftfreq(T, tr)
    F = np.fft.rfft(rng.standard_normal((n, T)), axis=1)
    F[:, (f < band[0]) | (f > band[1])] = 0
    x = np.fft.irfft(F, n=T, axis=1)
    return x / x.std(1, keepdims=True)


def surface_image():
    """Left lateral inflated surface with the Schaefer 200 / 7 parcels coloured by network; one parcel in black."""
    import nibabel as nib
    from nilearn import datasets, plotting, surface
    img = nib.load(str(REPO / "atlases" / "schaefer2018" / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"))
    names = [ln.split("\t")[1] for ln in open(REPO / "atlases" / "schaefer2018" / "Schaefer2018_200Parcels_7Networks_order.txt")]
    net_of = np.array([0] + [NETS.index(n.split("_")[2]) + 1 for n in names])
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    try:
        lab = surface.vol_to_surf(img, fs["pial_left"], interpolation="nearest_most_frequent", radius=3.0)
    except (TypeError, ValueError):
        lab = surface.vol_to_surf(img, fs["pial_left"], interpolation="nearest", radius=3.0)
    lab = np.rint(np.nan_to_num(lab)).astype(int)
    vals = net_of[np.clip(lab, 0, len(net_of) - 1)].astype(float)
    pick = next(i + 1 for i, n in enumerate(names) if n.startswith("7Networks_LH_Default_Par"))   # an inferior parietal default parcel
    vals[lab == pick] = 8
    vals[lab == 0] = 9
    cmap = ListedColormap(NETCOL + ["#111111", "#e4e2dd"])
    f = plt.figure(figsize=(3, 2.2), dpi=300)
    ax = f.add_subplot(111, projection="3d")
    plotting.plot_surf_roi(fs["infl_left"], vals, bg_map=fs["sulc_left"], hemi="left", view="lateral", cmap=cmap,
                           vmin=0.5, vmax=9.5, axes=ax, figure=f, colorbar=False, bg_on_data=False)
    f.canvas.draw()
    im = np.asarray(f.canvas.buffer_rgba())[..., :3].copy()
    plt.close(f)
    nonwhite = np.where(im.min(2) < 245)
    y0, y1, x0, x1 = nonwhite[0].min(), nonwhite[0].max(), nonwhite[1].min(), nonwhite[1].max()
    im = im[max(y0 - 4, 0):y1 + 4, max(x0 - 4, 0):x1 + 4]
    blk = np.where((im.max(2) < 40))
    centre = (blk[1].mean() / im.shape[1], blk[0].mean() / im.shape[0]) if blk[0].size else (0.6, 0.4)
    return im, centre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    W, H = 183 * MM, 128 * MM
    fig = plt.figure(figsize=(W, H))

    # section headers
    fig.text(0.012, 0.985, "Rationale: two physical sources, a costly split, and the limit of whole-brain decomposition", fontsize=7, fontweight="semibold", color=INK2, va="top")
    fig.text(0.012, 0.50, "Route B: separate BOLD-like from S0-like signal region by region", fontsize=7, fontweight="semibold", color=INK2, va="top")
    fig.add_artist(plt.Line2D([0.012, 0.988], [0.515, 0.515], color=RULE, lw=0.6, transform=fig.transFigure))

    # ---------------------------------------------------------------- a: two physical sources
    letter(fig, 0.012, 0.95, "a")
    ax = fig.add_axes([0.06, 0.62, 0.17, 0.27])
    te = np.linspace(0, 75, 200)
    T2 = 45.0
    base = np.exp(-te / T2)
    ax.plot(te, base, color=INK, lw=1.0, label="baseline")
    ax.plot(te, 1.10 * base, color=S0C, lw=1.0, ls=(0, (3, 1.5)), label="ΔS0: net signal")
    ax.plot(te, np.exp(-te * (1 / T2 - 0.0035)), color=BOLD, lw=1.0, ls=(0, (3, 1.5)), label="ΔR2*: BOLD")
    ax.plot(TE, np.exp(-TE / T2), "o", ms=3, color=INK, mec="white", mew=0.5, zorder=4)
    for t in TE:
        ax.axvline(t, color=RULE, lw=0.5, zorder=0)
    ax.set_xlabel("echo time (ms)")
    ax.set_ylabel("signal (a.u.)")
    ax.set_xlim(0, 75); ax.set_ylim(0, 1.15)
    ax.set_title("Four echoes per volume")
    ax.legend(loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0)
    ax.text(38, 0.07, "changes exaggerated for display", fontsize=4.8, color=INK2, ha="center")
    ax2 = fig.add_axes([0.285, 0.62, 0.12, 0.27])
    ax2.plot(te, np.full_like(te, 1.0), color=S0C, lw=1.2)
    ax2.plot(te, te * 1e-3 * 100 * 0.35, color=BOLD, lw=1.2)
    ax2.plot(TE, np.full(4, 1.0), "o", ms=3, color=S0C, mec="white", mew=0.5)
    ax2.plot(TE, TE * 1e-3 * 100 * 0.35, "o", ms=3, color=BOLD, mec="white", mew=0.5)
    ax2.axhline(0, color=INK2, lw=0.5)
    ax2.set_xlim(0, 75); ax2.set_ylim(-0.3, 2.9)
    ax2.set_xlabel("echo time (ms)")
    ax2.set_ylabel("log signal change (%)")
    ax2.set_title("Echo dependence")
    ax2.text(70, 1.15, "ΔS0: flat", color=S0C, ha="right", va="bottom", fontsize=5.5)
    ax2.text(3, 2.75, "ΔR2*: grows\nwith TE", color=BOLD, ha="left", va="top", fontsize=5.5)

    # ---------------------------------------------------------------- b: voxel vs region
    letter(fig, 0.43, 0.95, "b")
    ax = fig.add_axes([0.48, 0.62, 0.17, 0.27])
    mu = np.exp(-TE / T2) / np.exp(-TE[0] / T2)
    w = (mu * 50 / RATIOS) ** 2
    X = np.column_stack([np.ones(4), -TE * 1e-3])                   # a (fraction), b (1/s)
    C = np.linalg.inv(X.T @ (w[:, None] * X))
    Cp = np.array([[C[0, 0] * 1e4, C[0, 1] * 1e2], [C[0, 1] * 1e2, C[1, 1]]])    # (%, 1/s)
    r = Cp[0, 1] / np.sqrt(Cp[0, 0] * Cp[1, 1])
    v0 = 1.0 / np.sum(w * (TE * 1e-3) ** 2)
    infl = np.sqrt(Cp[1, 1] / v0)
    for n_vox, col, alpha, lab in ((1, NOISE, 0.35, "one voxel"), (60, BOLD, 0.55, "region of 60 voxels")):
        P = rng.multivariate_normal([0, 0], Cp / n_vox, 500)
        ax.scatter(P[:, 0], P[:, 1], s=1.2, color=col, alpha=alpha, linewidths=0, label=lab, rasterized=True)
        ev, evec = np.linalg.eigh(Cp / n_vox)
        ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
        ax.add_patch(Ellipse((0, 0), 2 * 2 * np.sqrt(ev[1]), 2 * 2 * np.sqrt(ev[0]), angle=ang, fill=False,
                             ec=INK if n_vox > 1 else INK2, lw=0.6, ls="-" if n_vox > 1 else (0, (2, 1.5))))
    lim_x, lim_y = 3.2 * np.sqrt(Cp[0, 0]), 3.2 * np.sqrt(Cp[1, 1])
    ax.set_xlim(-lim_x, lim_x); ax.set_ylim(-lim_y, lim_y)
    ax.axhline(0, color=RULE, lw=0.5, zorder=0); ax.axvline(0, color=RULE, lw=0.5, zorder=0)
    ax.set_xlabel("ΔS0 estimate (%)")
    ax.set_ylabel("ΔR2* estimate (1/s)")
    ax.set_title("Estimation noise in one frame")
    leg = ax.legend(loc="lower right", frameon=False, markerscale=4, handletextpad=0.2, borderaxespad=0)
    for h in leg.legend_handles:
        h.set_alpha(1)
    ax.text(0.03, 0.97, f"one voxel: estimates correlated\n(r = {abs(r):.2f}); ΔR2* noise ×{infl:.1f}\nvs ignoring ΔS0",
            transform=ax.transAxes, fontsize=5, color=INK2, va="top")

    # ---------------------------------------------------------------- c: detection by whole-brain ICA vs route B
    letter(fig, 0.70, 0.95, "c")
    ax = fig.add_axes([0.815, 0.62, 0.12, 0.25])
    rows = [("Whole-brain ICA,\nfocal sources", 0, 108, NOISE), ("Whole-brain ICA,\nnetwork sources", 2, 33, NOISE),
            ("Route B, injected\nregional shares", 22, 22, BOLD)]
    for i, (lab, k, n, col) in enumerate(rows):
        ax.barh(i, 1.0, color="#efeeeb", height=0.55)
        ax.barh(i, k / n, color=col, height=0.55)
        ax.text(1.03, i, f"{k}/{n}", va="center", fontsize=5.5, color=INK)
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=5.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50", "100 %"])
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.text(0.725, 0.905, "Known signals added to real scans", fontsize=6.5, fontweight="semibold", color=INK, va="bottom")
    fig.text(0.725, 0.585, "ICA: isolated as a component (|r| ≥ 0.9)\nroute B: share recovered within 0.1",
             fontsize=4.6, color=INK2, va="top")

    # ---------------------------------------------------------------- d: parcels in native space
    letter(fig, 0.012, 0.465, "d")
    im, centre = surface_image()
    axd = fig.add_axes([0.015, 0.11, 0.185, 0.31])
    axd.imshow(im)
    axd.axis("off")
    axd.set_title("1  Cortical regions in each\n    participant's own EPI space", loc="left", fontsize=6.5)
    cx, cy = centre[0] * im.shape[1], centre[1] * im.shape[0]
    axd.annotate("one region", (cx, cy), xytext=(im.shape[1] * 0.80, im.shape[0] * 1.05), fontsize=5.5, color=INK,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=INK), ha="center", annotation_clip=False)
    for i, (n, c) in enumerate(zip(["Visual", "Somatomotor", "Dorsal att.", "Salience", "Limbic", "Control", "Default"], NETCOL)):
        fig.text(0.02 + (i % 3) * 0.074, 0.08 - (i // 3) * 0.022, "■", color=c, fontsize=6, va="center")
        fig.text(0.029 + (i % 3) * 0.074, 0.08 - (i // 3) * 0.022, n, fontsize=4.6, color=INK2, va="center")

    # ---------------------------------------------------------------- e: four echo signals of the region
    letter(fig, 0.235, 0.465, "e")
    T = 320
    t = np.arange(T) * 0.75
    z = bandlimited(rng, 2, T)
    a_t = 0.0015 * z[1]                                             # ΔlnS0
    b_t = 0.35e-3 * z[0] + 0.15e-3 * z[1]                           # ΔR2* (1/ms)
    ax = fig.add_axes([0.30, 0.10, 0.14, 0.31])
    greys = ["#b9b7b2", "#8f8d88", "#5e5c58", "#232220"]
    for e in range(4):
        y = (a_t - TE[e] * b_t + 0.0012 * RATIOS[e] * rng.standard_normal(T)) * 100
        ax.plot(t, y - e * 3.2, color=greys[e], lw=0.7)
        ax.text(-6, -e * 3.2, f"{TE[e]:.0f} ms", fontsize=5, color=INK2, va="center", ha="right")
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, t[-1])
    ax.set_title("2  Average each echo\n    over the region", loc="left", x=-0.3)
    ax.plot([t[-1] + 6] * 2, [-11.2, -9.2], color=INK, lw=0.8, clip_on=False)
    ax.text(t[-1] + 9, -10.2, "2 %", fontsize=5, rotation=90, va="center", ha="left")

    # ---------------------------------------------------------------- f: split into S0-like, BOLD-like, leftover
    letter(fig, 0.49, 0.465, "f")
    ax = fig.add_axes([0.54, 0.27, 0.14, 0.14])
    ax.plot(t, b_t * 1000, color=BOLD, lw=0.8)
    ax.fill_between(t, -0.08, 0.08, color=NOISE, alpha=0.25, lw=0)
    ax.set_ylabel("ΔR2* (1/s)")
    ax.set_xticks([]); ax.set_xlim(0, t[-1])
    ax.set_title("3  Split by echo dependence;\n    noise from leftover channels", loc="left", x=-0.2)
    ax.text(0.99, 0.98, "BOLD-like", color=BOLD, ha="right", va="top", fontsize=5.5, transform=ax.transAxes, bbox=dict(fc="white", ec="none", pad=0.6))
    ax = fig.add_axes([0.54, 0.10, 0.14, 0.14])
    ax.plot(t, a_t * 100, color=S0C, lw=0.8)
    ax.fill_between(t, -0.05, 0.05, color=NOISE, alpha=0.25, lw=0)
    ax.set_ylabel("ΔS0/S0 (%)")
    ax.set_xlabel("time (s)"); ax.set_xlim(0, t[-1])
    ax.text(0.99, 0.98, "S0-like", color=S0C, ha="right", va="top", fontsize=5.5, transform=ax.transAxes, bbox=dict(fc="white", ec="none", pad=0.6))
    axs = fig.add_axes([0.695, 0.10, 0.014, 0.31])
    shares = [("BOLD-like", 0.92, BOLD), ("S0-like", 0.07, S0C), ("leftover", 0.01, NOISE)]
    bottom = 0
    for lab, v, c in shares:
        axs.bar(0, v, bottom=bottom, color=c, width=1, edgecolor="white", linewidth=0.6)
        bottom += v
    axs.set_xlim(-0.5, 0.5); axs.set_ylim(0, 1); axs.axis("off")
    fig.text(0.702, 0.09, "share", fontsize=5, color=INK2, va="top", ha="center")
    fig.text(0.713, 0.395, "S0-like +\nleftover", fontsize=4.8, color=S0C, va="center")
    fig.text(0.713, 0.22, "BOLD-\nlike", fontsize=4.8, color=BOLD, va="center")

    # ---------------------------------------------------------------- g: connectivity and integration
    letter(fig, 0.79, 0.465, "g")
    from mejoint.regions import participation_ratio, participation_ratio_residual
    per = 12
    n = 7 * per
    L = np.zeros((n, 8))
    L[:, 0] = 0.75 * (1 + 0.3 * rng.standard_normal(n))
    for k in range(7):
        L[k * per:(k + 1) * per, k + 1] = 0.8
    Cm = L @ L.T
    d = np.sqrt(np.diag(Cm) + 1.0)
    Cm = (Cm + np.eye(n)) / np.outer(d, d)
    ax = fig.add_axes([0.83, 0.27, 0.14, 0.14])
    Cshow = Cm.copy(); np.fill_diagonal(Cshow, np.nan)
    ax.imshow(Cshow, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4)
    for k in range(7):
        ax.add_patch(plt.Rectangle((-3.5, k * per - 0.5), 2.2, per, color=NETCOL[k], clip_on=False, lw=0))
    ax.set_title("4  BOLD-like connectivity,\n    noise removed", loc="left", x=-0.12)
    ev = np.sort(np.linalg.eigvalsh(Cm))[::-1][:12]
    ax = fig.add_axes([0.83, 0.10, 0.14, 0.12])
    ax.bar(0, ev[0], color=NOISE, width=0.7)
    ax.bar(np.arange(1, ev.size), ev[1:], color=BOLD, width=0.7)
    ax.set_yscale("log")
    ax.set_xticks([]); ax.set_xlabel("patterns (eigenvalues)")
    ax.set_ylabel("variance", labelpad=1)
    ax.text(0.6, ev[0], "brain-wide\npattern", fontsize=4.8, color=INK2, va="center")
    ax.text(ev.size - 0.5, ev[1] * 1.6, "counted by\nthe primary", fontsize=4.8, color=BOLD, ha="right", va="bottom")
    fig.text(0.80, 0.05, "primary: PR without first pattern", fontsize=5.3, color=INK)
    fig.text(0.80, 0.026, "= (Σλ − λ₁)² / (Σλ² − λ₁²)", fontsize=5.3, color=INK)

    # arrows along the route B row
    for x0, x1 in ((0.207, 0.232), (0.462, 0.488), (0.758, 0.785)):
        fig.add_artist(FancyArrowPatch((x0, 0.26), (x1, 0.26), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=7, lw=0.8, color=INK2))
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"concept_figure.{ext}")
    fig.savefig(out / "concept_figure.png", dpi=300)
    print("written", out)


if __name__ == "__main__":
    main()
