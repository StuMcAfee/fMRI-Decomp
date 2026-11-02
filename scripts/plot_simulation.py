"""QC figure of one simulated run (Phase 1 gate evidence).

Row 1: the static anatomy the simulator builds (S0, T2*, echo-4 thermal SNR, skull-base dephasing).
Row 2: where the planted sources are (dR2* and dS0/S0 maps, projected through all slices) and the
       time courses of three of them.
Row 3: the physics the decomposition must exploit: the per-echo signal change of a pure-dR2*, a
       pure-dS0 and a mixed voxel (pure dR2* grows with TE, pure dS0 is flat across echoes).

Usage:  PYTHONPATH=src python scripts/plot_simulation.py OUT.png [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mejoint import io, simulate  # noqa: E402

# reference palette (dataviz skill): surface, ink, categorical slots 1-3, blue ramp, red arm, neutral mid
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ECHO_STEPS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # ordinal: echo 1 -> 4
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUES)
DIV = LinearSegmentedColormap.from_list("blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8322f"])
GRAY = LinearSegmentedColormap.from_list("gray_seq", ["#0b0b0b", "#f0efec"])


def signed_projection(vol):
    """Value with the largest magnitude along z (keeps the sign): shows every source at once."""
    idx = np.abs(vol).argmax(axis=2)
    return np.take_along_axis(vol, idx[..., None], axis=2)[..., 0]


def style(ax, title):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)


def image(fig, ax, img, cmap, title, unit, norm=None, vmin=None, vmax=None):
    im = ax.imshow(img.T, origin="lower", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    style(ax, title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK2, length=0)
    cb.set_label(unit, fontsize=7, color=INK2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = io.load_config()
    sim = simulate.simulate(cfg, seed=args.seed, overrides={"n_volumes": 400})
    # same seed without noise: identical maps and time courses (noise is drawn last)
    clean = simulate.simulate(cfg, seed=args.seed, overrides={"n_volumes": 400, "snr0": None})
    by = {s["name"]: s for s in sim.sources}
    m = sim.mask
    z_mid = by["r2_focal"]["center"][2]
    y_mid = sim.mask.shape[1] // 2

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.facecolor": SURF,
                         "figure.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
    fig = plt.figure(figsize=(13, 9.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05], hspace=0.55, wspace=0.35)

    # ---- row 1: anatomy
    cor = lambda vol: np.where(m[:, y_mid, :], vol[:, y_mid, :], np.nan)     # coronal slice, skull base at the bottom
    image(fig, fig.add_subplot(gs[0, 0]), cor(sim.s0), GRAY, f"S0 (coronal y={y_mid})", "a.u.")
    image(fig, fig.add_subplot(gs[0, 1]), cor(sim.t2s_ms), SEQ, "T2* (short at the skull base)", "ms", vmin=20, vmax=70)
    snr4 = cor(sim.mu_base[..., 3] / sim.sigma)
    image(fig, fig.add_subplot(gs[0, 2]), snr4, SEQ, "Echo-4 thermal SNR", "mu / sigma", vmin=0, vmax=np.nanmax(snr4))
    image(fig, fig.add_subplot(gs[0, 3]), cor(sim.dephasing_hz), SEQ, "Dephasing (skull base)", "Hz", vmin=0)

    # ---- row 2: planted sources
    q = sum(s["q_map"] for s in sim.sources if s["name"] != "global_resp")
    s0m = 100 * sum(s["s_map"] for s in sim.sources)
    qp = np.where(m.any(2), signed_projection(q), np.nan)
    sp = np.where(m.any(2), signed_projection(s0m), np.nan)
    lim_q, lim_s = np.nanmax(np.abs(qp)), np.nanmax(np.abs(sp))
    image(fig, fig.add_subplot(gs[1, 0]), qp, DIV, "Planted dR2* maps (all slices)", "s$^{-1}$",
          norm=TwoSlopeNorm(0, -lim_q, lim_q))
    image(fig, fig.add_subplot(gs[1, 1]), sp, DIV, "Planted dS0/S0 maps (all slices)", "%",
          norm=TwoSlopeNorm(0, -lim_s, lim_s))
    ax = fig.add_subplot(gs[1, 2:])
    t = np.arange(sim.n_volumes) * sim.tr_s
    show = [("r2_focal", "dR2* source (0.01-0.1 Hz)"), ("s0_focal", "dS0 source (0.01-0.1 Hz)"),
            ("global_resp", "global respiration (~0.3 Hz)")]
    for i, (name, lab) in enumerate(show):
        y = by[name]["timecourse"][:200] - 4.5 * i
        ax.plot(t[:200], y, color=CAT[i], lw=1.2)
        ax.text(t[199] + 2, y[-20:].mean(), lab, color=INK2, fontsize=8, va="center")
    for f in sim.spike_frames[sim.spike_frames < 200]:
        ax.axvline(f * sim.tr_s, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks([])
    ax.set_xlim(0, t[199] + 55)
    ax.set_xlabel("time (s); dotted lines: model-violating spike frames", fontsize=8)
    style(ax, "Unit-SD time courses (first 150 s)")

    # ---- row 3: the physics
    ax = fig.add_subplot(gs[2, 0:2])
    v = tuple(by["mixed_same"]["center"])
    pct = 100 * (clean.data[v].astype(float) / clean.mu_base[v][:, None] - 1)
    for e in range(4):
        ax.plot(t[:200], pct[e, :200], color=ECHO_STEPS[e], lw=1.4, label=f"TE {sim.te_ms[e]:.2f} ms")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=4, handlelength=1.5, labelcolor=INK2)
    ax.set_xlim(0, t[199])
    ax.set_ylabel("% signal change")
    ax.set_xlabel("time (s)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    style(ax, f"Mixed-source voxel {v}, noise-free: same time course, echo-dependent size")

    ax = fig.add_subplot(gs[2, 2:])
    for i, (name, lab) in enumerate([("r2_focal", "pure dR2*"), ("s0_focal", "pure dS0"), ("mixed_same", "mixed")]):
        vv = tuple(by[name]["center"])
        x = by[name]["timecourse"]
        y = np.log(sim.data[vv].astype(float) / sim.mu_base[vv][:, None])
        beta = 100 * (y - y.mean(1, keepdims=True)) @ x / (x @ x)            # % change per unit source, per echo
        ax.plot(sim.te_ms, beta, color=CAT[i], lw=2, marker="o", ms=6, mec=SURF, mew=1.5)
        ax.text(sim.te_ms[-1] + 1.5, beta[-1], lab, color=INK2, fontsize=8, va="center")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_xticks(sim.te_ms)
    ax.set_xlim(10, 75)
    ax.set_xlabel("echo time (ms)")
    ax.set_ylabel("signal change per unit source (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    style(ax, "Echo dependence at each source centre: dR2* scales with TE, dS0 is flat")

    fig.suptitle(f"Phase 1 simulator, seed {args.seed}: {sim.describe()}", fontsize=8, color=INK2, x=0.01, ha="left", y=0.995)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
