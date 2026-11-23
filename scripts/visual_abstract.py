#!/usr/bin/env python
"""Visual abstract of the parcel pipeline (docs/PARCEL_ICA_PLAN.md section 8), left to right, for a neuroscience
audience.

  PYTHONPATH=src python scripts/visual_abstract.py --resources outputs/atlases_fs32k --out <folder>

Writes visual_abstract.pdf (vector text, fonts embedded), .svg and .png (300 dpi). Brain maps are drawn from simulated
signals on the real icosahedral parcel layout (fs_LR 32k); no participant data are used.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle  # noqa: E402

MM = 1 / 25.4
TE = np.array([16.20, 31.22, 46.24, 61.26])
BOLD, S0C, PHYS, NOISE = "#2a78d6", "#eb6834", "#d1366b", "#8d8b85"
INK, INK2, PANEL, RULE = "#151514", "#55534f", "#f5f4f0", "#dcdad4"

plt.rcParams.update({
    "font.family": "Inter", "font.size": 6, "axes.labelsize": 5.6, "xtick.labelsize": 5, "ytick.labelsize": 5,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 1.8,
    "ytick.major.size": 1.8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white"})


def simulated_maps(resources):
    """Simulated participant on the real icosahedral layout: winner-take-all of a 7-component spatial ICA, a
    breathing-related component map, and a network-sorted correlation matrix."""
    from mejoint import parcel_ica as pi
    from mejoint import parcel_pipeline as pp
    from mejoint import parcel_sim as ps
    from mejoint.surface_plot import zscore
    an = pp.ico_anatomy(resources)
    nets = np.array(an["yeo7"], dtype=object)
    ps.networks = lambda xyz, rng, n_seeds_per_net=3: nets
    d = ps.simulate(1876, reliability=0.6, seed=11, amp={"respiratory": 0.5, "resp_s0": 0.002})
    pr = pi.prepare(d["c0"], d["c1"], d["R"], d["noise_var"])
    ica = pi.spatial_ica(pr["zb"], 8, range(6))
    names = d["truth"]["names"]
    tc_r = d["truth"]["tc"][names.index("respiratory")]
    j_r = int(np.argmax(np.abs(np.corrcoef(np.vstack([tc_r, ica["tc"]]))[0, 1:])))
    keep = [j for j in range(ica["maps"].shape[1]) if j != j_r]
    Z = zscore(ica["maps"][:, keep])
    win = np.argmax(Z, 1)
    # colour each kept component by the Yeo-7 network it overlaps most
    best = []
    for j in range(Z.shape[1]):
        r = {n: np.corrcoef(Z[:, j], (nets[pr["ok"]] == n).astype(float))[0, 1] for n in set(nets) - {""}}
        best.append(max(r, key=r.get))
    resp_map = zscore(ica["maps"][:, [j_r]])[:, 0]
    from mejoint.regions import band_pass
    neural = [i for i, n in enumerate(names) if n.startswith("network") or n == "global"]
    sig = d["truth"]["maps_b"][:, neural] @ d["truth"]["tc"][neural]
    x = band_pass(sig + 2.0 * np.random.default_rng(3).standard_normal(sig.shape) * sig.std(1, keepdims=True), 0.75, (0.01, 0.1))
    sub = np.random.default_rng(4).choice(len(x), 420, replace=False)
    order = sub[np.lexsort((np.arange(len(sub)), nets[sub]))]
    C = np.corrcoef(x[order])
    return {"ids": np.flatnonzero(pr["ok"]) + 1, "win": win, "best": best, "resp": resp_map, "C": C,
            "C_nets": nets[order], "nets": nets}


def brain(ax, res, colour_by_id, h="L", view="lateral", missing=None):
    from mejoint import surface_plot as sp
    vc = sp.label_colours(colour_by_id, res["labels"][h], missing if missing is not None else sp.MEDIAL)
    sp.draw(ax, *res["surf"][h], vc, h, view)


def panel(fig, x, w, y, hgt, num, title, sub):
    fig.add_artist(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                                  facecolor=PANEL, edgecolor=RULE, lw=0.6, zorder=0))
    fig.add_artist(Circle((x + 0.014, y + hgt - 0.033), 0.0105, transform=fig.transFigure, facecolor=INK, edgecolor="none"))
    fig.text(x + 0.014, y + hgt - 0.034, str(num), color="white", fontsize=6.2, fontweight="bold", ha="center", va="center")
    fig.text(x + 0.031, y + hgt - 0.022, title, fontsize=7.0, fontweight="semibold", color=INK, va="top")
    fig.text(x + 0.012, y + hgt - 0.062, sub, fontsize=5.0, color=INK2, va="top", linespacing=1.3)


def chip(fig, x, y, w, text, colour, filled=True):
    fig.add_artist(FancyBboxPatch((x, y), w, 0.052, boxstyle="round,pad=0,rounding_size=0.01", transform=fig.transFigure,
                                  facecolor=colour if filled else "white", edgecolor=colour, lw=0.8))
    fig.text(x + w / 2, y + 0.026, text, fontsize=5.0, color="white" if filled else colour, ha="center", va="center",
             fontweight="semibold", linespacing=1.15)


def arrow(fig, x0, x1, y, label=None):
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>,head_length=4,head_width=2.6",
                                   color=INK2, lw=0.9, shrinkA=0, shrinkB=0))
    if label:
        fig.text((x0 + x1) / 2, y + 0.016, label, fontsize=4.6, color=INK2, ha="center", va="bottom", linespacing=1.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resources", default=str(REPO / "outputs" / "atlases_fs32k"))
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    from mejoint import surface_plot as sp
    res = sp.load_resources(a.resources, "ico1002_7")
    col7 = sp.network_colours(a.resources, "Schaefer2018_200Parcels_7Networks_order.dlabel.nii")
    sim = simulated_maps(a.resources)

    fig = plt.figure(figsize=(230 * MM, 120 * MM))
    fig.text(0.012, 0.975, "From multi-echo physics to individual cortical networks", fontsize=9, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.925, "Each step removes or measures one class of signal; what remains is passed to the right.  "
             "Cortex in 1876 icosahedral parcels (fs_LR 32k), one participant at a time.", fontsize=5.6, color=INK2, va="top")
    xs = [0.012 + i * 0.1965 for i in range(5)]
    W, Y, H = 0.182, 0.17, 0.70
    titles = [("Acquire and parcellate", "4-echo EPI (TE 16-61 ms), motion-corrected;\nechoes averaged within each parcel"),
              ("Separate by echo physics", "split each parcel's signal by how it\nchanges with echo time"),
              ("Remove non-neural signal", "small ICA (k = leftover-channel count),\nrepeated until nothing qualifies"),
              ("Map individual networks", "ICA on the cleaned signal\n(k = components above thermal noise)"),
              ("Integrate and report", "graph measures on noise-corrected,\npositive connectivity")]
    for i, (t, s) in enumerate(titles):
        panel(fig, xs[i], W, Y, H, i + 1, t, s)

    # 1: echo decay + parcels
    ax = fig.add_axes([xs[0] + 0.03, 0.55, W - 0.05, 0.17])
    te = np.linspace(0, 70, 200)
    ax.plot(te, np.exp(-te / 45), color=INK2, lw=0.9)
    ax.plot(TE, np.exp(-TE / 45), "o", color=BOLD, ms=3.2, mec="white", mew=0.5)
    for k, t in enumerate(TE):
        ax.text(t, np.exp(-t / 45) + 0.09, f"E{k + 1}", fontsize=4.6, color=INK2, ha="center")
    ax.set_xlabel("echo time (ms)", labelpad=1)
    ax.set_ylabel("signal", labelpad=1)
    ax.set_yticks([])
    ax.set_ylim(0, 1.15)
    rng = np.random.default_rng(2)
    pal = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    cols = {int(p): np.r_[0.55 * pal[rng.integers(20)][:3] + 0.45, 1] for p in range(1, 1877)}
    ax = fig.add_axes([xs[0] + 0.01, 0.235, W - 0.02, 0.27])
    brain(ax, res, cols)

    # 2: echo-time dependence
    ax = fig.add_axes([xs[1] + 0.03, 0.50, W - 0.05, 0.22])
    t2 = np.linspace(0, 70, 50)
    ax.axhline(0, color=RULE, lw=0.6)
    ax.plot(t2, 0.03 * t2 / 61, color=BOLD, lw=1.4)
    ax.plot(t2, np.full_like(t2, 0.018), color=S0C, lw=1.4)
    ax.plot(TE, rng.normal(0, 0.0035, 4), "o", color=NOISE, ms=2.6)
    ax.text(68, 0.0365, "ΔR2* (BOLD-like)", color=BOLD, fontsize=5, ha="right", va="bottom", fontweight="semibold")
    ax.text(2, 0.0205, "ΔS0", color=S0C, fontsize=5, ha="left", va="bottom", fontweight="semibold")
    ax.text(68, -0.008, "leftover: noise", color=NOISE, fontsize=5, ha="right", va="top")
    ax.set_xlabel("echo time (ms)", labelpad=1)
    ax.set_ylabel("signal change", labelpad=1)
    ax.set_yticks([])
    ax.set_ylim(-0.016, 0.046)
    fig.text(xs[1] + 0.012, 0.425, "ΔR2* grows with TE: BOLD-like", fontsize=5.0, color=BOLD, va="top", fontweight="semibold")
    fig.text(xs[1] + 0.012, 0.402, "ΔS0 is equal at every TE: inflow,\npulsation, motion-related change", fontsize=5.0, color=S0C, va="top", linespacing=1.3, fontweight="semibold")
    fig.text(xs[1] + 0.012, 0.34, "leftover echo patterns hold only\nthermal noise: the reference for\nevery later threshold", fontsize=5.0, color=INK2, va="top", linespacing=1.3)
    fig.text(xs[1] + 0.012, 0.264, "exact linear split, parcel by parcel;\nno decomposition needed", fontsize=5.0, color=INK, va="top", linespacing=1.3)

    # 3: spectrum + breathing map + loop
    ax = fig.add_axes([xs[2] + 0.03, 0.55, W - 0.05, 0.16])
    f = np.linspace(0.003, 0.667, 400)
    neural = 1.1 / (1 + (f / 0.06) ** 4)
    breath = 0.95 * np.exp(-0.5 * ((f - 0.28) / 0.022) ** 2) + 0.3 * np.exp(-0.5 * ((f - 0.56) / 0.025) ** 2)
    ax.axvspan(0.01, 0.1, color=BOLD, alpha=0.12, lw=0)
    ax.fill_between(f, 0, neural, color=BOLD, alpha=0.55, lw=0)
    ax.fill_between(f, 0, breath, color=PHYS, alpha=0.8, lw=0)
    ax.text(0.055, 1.18, "neural band", color=BOLD, fontsize=4.8, ha="center")
    ax.text(0.3, 1.0, "breathing", color=PHYS, fontsize=4.8, ha="left", fontweight="semibold")
    ax.set_xlim(0, 0.667)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([])
    ax.set_xlabel("frequency (Hz)", labelpad=1)
    ax.set_ylabel("power", labelpad=1)
    ax = fig.add_axes([xs[2] + 0.008, 0.33, W - 0.05, 0.155])
    norm = matplotlib.colors.Normalize(-3, 3)
    cmap = sp.diverging_cmap()
    brain(ax, res, {int(p): cmap(norm(v)) for p, v in zip(sim["ids"], sim["resp"])})
    fig.add_artist(FancyArrowPatch((xs[2] + W - 0.03, 0.47), (xs[2] + W - 0.03, 0.35), transform=fig.transFigure,
                                   connectionstyle="arc3,rad=-0.9", arrowstyle="-|>,head_length=3,head_width=2", color=INK2, lw=0.8))
    fig.text(xs[2] + W - 0.012, 0.41, "repeat", fontsize=4.6, color=INK2, rotation=90, va="center", ha="center")
    fig.text(xs[2] + 0.012, 0.312, "flagged by power above 0.12 Hz\n(breathing, aliased cardiac) or S0\ncoupling; their maps are speckled,\nnot network-like", fontsize=5.0, color=INK2, va="top", linespacing=1.3)

    # 4: winner-take-all
    comp_col = {}
    seen = {}
    for j, n in enumerate(sim["best"]):
        base = col7.get(n, np.array([0.5, 0.5, 0.5, 1]))[:3]
        s = (0.0, 0.4, -0.3)[seen.get(n, 0) % 3]
        seen[n] = seen.get(n, 0) + 1
        comp_col[j] = np.r_[base + (1 - base) * s if s >= 0 else base * (1 + s), 1]
    wcols = {int(p): comp_col[w] for p, w in zip(sim["ids"], sim["win"])}
    ax = fig.add_axes([xs[3] + 0.02, 0.49, W - 0.04, 0.2])
    brain(ax, res, wcols, "L", "lateral")
    ax = fig.add_axes([xs[3] + 0.02, 0.30, W - 0.04, 0.2])
    brain(ax, res, wcols, "L", "medial")
    fig.text(xs[3] + 0.012, 0.285, "winner-take-all partition: each parcel\ncoloured by its strongest component\n(hue: nearest Yeo-7 network)", fontsize=5.0, color=INK2, va="top", linespacing=1.3)

    # 5: connectivity matrix + measures
    ax = fig.add_axes([xs[4] + 0.04, 0.505, W - 0.07, 0.2])
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("s", ["#ffffff", "#9ec5f4", "#2a78d6", "#0d366b"])
    ax.imshow(np.clip(sim["C"], 0, None), cmap=seq, vmin=0, vmax=0.6, interpolation="nearest")
    cn = sim["C_nets"]
    cuts = np.flatnonzero(cn[1:] != cn[:-1]) + 1
    for c in cuts:
        ax.axhline(c - 0.5, color="white", lw=0.5)
        ax.axvline(c - 0.5, color="white", lw=0.5)
    st, en = np.r_[0, cuts], np.r_[cuts, len(cn)]
    for s_, e_ in zip(st, en):
        ax.add_patch(matplotlib.patches.Rectangle((-14, s_ - 0.5), 10, e_ - s_, color=col7.get(cn[s_], (0.6, 0.6, 0.6, 1)), clip_on=False, lw=0))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title("connectivity sorted by network", fontsize=5.2, color=INK2, pad=2)
    items = [("participation ratio", "dimensionality beyond the\nbrain-wide pattern"),
             ("participation coefficient,\nsegregation", "Yeo-7, Yeo-17, individual"),
             ("node count, undominated\ncortex", "organisation of the individual map"),
             ("interactive HTML report", "networks first, removed noise below")]
    yy = 0.465
    for t, s in items:
        nt, nd = t.count("\n") + 1, s.count("\n") + 1
        fig.text(xs[4] + 0.012, yy, t, fontsize=5.1, color=INK, fontweight="semibold", va="top", linespacing=1.2)
        fig.text(xs[4] + 0.012, yy - 0.022 * nt - 0.002, s, fontsize=4.7, color=INK2, va="top", linespacing=1.2)
        yy -= 0.022 * nt + 0.02 * nd + 0.016

    # arrows: what is passed on
    passes = ["passes on: 4 echoes per parcel", "passes on: BOLD-like signal", "passes on: cleaned BOLD-like signal",
              "passes on: network components", ""]
    for i in range(5):
        if i < 4:
            arrow(fig, xs[i] + W + 0.001, xs[i + 1] - 0.001, 0.52)
        if passes[i]:
            fig.text(xs[i] + 0.012, Y + 0.018, passes[i], fontsize=4.9, color=BOLD, fontstyle="italic", va="bottom")

    # bottom: what is separated at each step
    fig.text(0.012, 0.137, "separated at each step", fontsize=5.4, color=INK2, fontweight="semibold", va="center")
    chip(fig, xs[0] + 0.008, 0.07, W - 0.016, "head motion\n(regressed beforehand)", NOISE)
    chip(fig, xs[1] + 0.008, 0.07, (W - 0.022) / 2, "ΔS0 signal", S0C)
    chip(fig, xs[1] + 0.014 + (W - 0.022) / 2, 0.07, (W - 0.022) / 2, "thermal noise\n(measured)", NOISE, filled=False)
    chip(fig, xs[2] + 0.008, 0.07, W - 0.016, "cardiorespiratory\nBOLD-like signal", PHYS)
    chip(fig, xs[3] + 0.008, 0.07, W - 0.016, "kept: neural\nBOLD-like networks", BOLD)
    chip(fig, xs[4] + 0.008, 0.07, W - 0.016, "measures per\nparticipant", INK2, filled=False)
    fig.text(0.012, 0.028, "Brain maps: simulated data on the real parcel layout. No physiological recordings are needed: non-neural sources are inferred "
             "from their spectra, echo-time behaviour and spatial pattern.\nSlow physiological signal inside the neural band (e.g. CO2, heart-rate variability) cannot be separated this way and remains a limitation.",
             fontsize=4.6, color=INK2, va="center", linespacing=1.35)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(out / f"visual_abstract.{ext}", **kw)
    print(out / "visual_abstract.png")


if __name__ == "__main__":
    main()
