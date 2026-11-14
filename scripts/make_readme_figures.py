"""Explanatory figures for README.md. Every number comes from the package's own code (simulator, front-end
projection, decomposition), so the figures stay true to the implementation.
Usage: PYTHONPATH=src python scripts/make_readme_figures.py [out_dir]   (default docs/figures)"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mejoint import decompose, frontend, io, metrics, simulate

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
ECHO = ["#86b6ef", "#5598e7", "#256abf", "#104281"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
cfg = io.load_config()
TE = np.asarray(cfg["acquisition"]["te_ms"])
RAT = np.asarray(cfg["frontend"]["echo_noise_ratios"])


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name)


# ------------------------------------------------------------------ 1. signal model
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
te = np.linspace(0, 70, 200)
t2 = 45.0
base = np.exp(-te / t2)
dR, dS = -2.0e-3, 0.02                         # exaggerated for visibility: dR2* = -2 s^-1, dS0/S0 = +2 %
ax[0].plot(te, base, color=INK2, lw=1.5, label="baseline (T2* 45 ms)")
ax[0].plot(te, base * np.exp(-te * dR), color=BLUE, lw=1.5, label="neural (BOLD): R2* falls")
ax[0].plot(te, base * (1 + dS), color=ORANGE, lw=1.5, ls="--", label="S0 change: inflow, motion")
for e, c in zip(TE, ECHO):
    ax[0].axvline(e, color=c, lw=0.8, alpha=0.7)
ax[0].set(xlabel="echo time TE (ms)", ylabel="signal (relative)", title="A  One voxel, four echoes")
ax[0].legend(fontsize=7.5, frameon=False)
pct_b = (np.exp(-te * dR) - 1) * 100
ax[1].plot(te, pct_b, color=BLUE, lw=1.5, label="BOLD: grows with TE")
ax[1].plot(te, np.full_like(te, dS * 100), color=ORANGE, lw=1.5, ls="--", label="S0: same at every TE")
for e, c in zip(TE, ECHO):
    ax[1].scatter([e], [(np.exp(-e * dR) - 1) * 100], color=c, zorder=3, s=22)
    ax[1].scatter([e], [dS * 100], color=c, zorder=3, s=22, marker="s")
ax[1].set(xlabel="echo time TE (ms)", ylabel="signal change (%)", title="B  The two kinds of change")
ax[1].legend(fontsize=7.5, frameon=False)
y_b, y_s = -TE * dR, np.full(4, np.log(1 + dS))
ax[2].plot([0, 70], [0, -70 * dR], color=BLUE, lw=1, alpha=0.5)
ax[2].plot([0, 70], [np.log(1 + dS)] * 2, color=ORANGE, lw=1, ls="--", alpha=0.5)
ax[2].plot([0, 70], [np.log(1 + dS), np.log(1 + dS) - 70 * dR], color=AQUA, lw=1, alpha=0.6)
for e, c, yb, ys in zip(TE, ECHO, y_b, y_s):
    ax[2].scatter([e], [yb], color=c, s=22, zorder=3)
    ax[2].scatter([e], [ys], color=c, s=22, marker="s", zorder=3)
    ax[2].scatter([e], [yb + ys], color=c, s=22, marker="^", zorder=3)
ax[2].text(66, -66 * dR, "b: slope = -dR2*", color=BLUE, fontsize=7.5, ha="right", va="bottom")
ax[2].text(2, np.log(1 + dS) + 0.002, "a: intercept = d ln S0", color=ORANGE, fontsize=7.5)
ax[2].text(66, np.log(1 + dS) - 66 * dR, "mixed", color=AQUA, fontsize=7.5, ha="right", va="top")
ax[2].set(xlabel="echo time TE (ms)", ylabel="y = ln(S / mean S)", title="C  Log-ratio: a straight line in TE")
fig.suptitle("Figure 1. The multi-echo signal model: y_e = a - TE_e b + noise", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "readme_1_signal_model.png")

# ------------------------------------------------------------------ 2. separation penalty
rng = np.random.default_rng(0)
mu = 1e4 * np.exp(-TE / 45.0)
sig = mu[0] / 50 * RAT                                         # per-echo thermal noise, echo-1 SNR 50
W = np.diag((mu / sig) ** 2)
D = np.column_stack([np.ones(4), -TE])
cov = np.linalg.inv(D.T @ W @ D)
draws = rng.multivariate_normal([0, 0], cov, 3000)
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.7))
r = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
ax[0].scatter(draws[:, 0] * 100, draws[:, 1] * 1000, s=3, color=BLUE, alpha=0.3)
ax[0].set(xlabel="estimated dS0/S0 (%)  [a]", ylabel="estimated dR2* (s^-1)  [b]",
          title=f"A  Noise in one voxel and one frame: r = {r:.2f}")
ax[0].text(0.02, 0.97, "Pure noise. Fitting both a and b per voxel\nmakes their errors strongly coupled.",
           transform=ax[0].transAxes, va="top", fontsize=7.5, color=INK2)
t2s = np.linspace(20, 70, 26)
infl = []
for t2_ in t2s:
    m_ = 1e4 * np.exp(-TE / t2_)
    W_ = np.diag((m_ / (m_[0] / 50 * RAT)) ** 2)
    sd_joint = np.sqrt(np.linalg.inv(D.T @ W_ @ D)[1, 1])
    sd_b_only = np.sqrt(1 / (TE @ W_ @ TE))
    infl.append(sd_joint / sd_b_only)
ax[1].plot(t2s, infl, color=BLUE, lw=1.8)
ax[1].set(xlabel="tissue T2* (ms)", ylabel="dR2* noise SD, ratio",
          title="B  Cost of also fitting dS0, voxel by voxel")
ax[1].text(0.03, 0.97, "dR2* noise when dS0 is fitted too, relative to\nassuming dS0 = 0. This is why separation is done\n"
           "at the component level, where thousands of\nvoxels share one time course.", transform=ax[1].transAxes,
           fontsize=7.5, color=INK2, va="top")
fig.suptitle("Figure 2. Why the two maps are separated per component, not per voxel", x=0.01, ha="left", fontsize=11,
             fontweight="bold")
fig.tight_layout()
save(fig, "readme_2_separation_penalty.png")

# ------------------------------------------------------------------ 3. what each channel sees
m = (mu / RAT) / (mu[0] / 50)
kinds = {"neural dR2*": -TE * 1e-3, "dS0": np.full(4, 0.01), "curvature\n(partial volume,\ndephasing)":
         1e-6 * (TE - TE.mean()) ** 2, "odd/even echo\n(ghosting)": 0.002 * (-1.0) ** np.arange(4)}
E = []
for y in kinds.values():
    pr = frontend.project(y[None, :, None], m[None, :], np.ones((1, 4), bool), TE)
    e = np.r_[pr["c"][0, :, 0] ** 2, pr["r"][0, :, 0] ** 2]
    E.append(e / e.sum())
E = np.array(E)
fig, ax = plt.subplots(figsize=(8.6, 3.4))
names = ["c0 (OC-like: BOLD + S0)", "c1 (dS0 only)", "r0 (residual: curvature)", "r1 (residual: alternation)"]
cols = [BLUE, ORANGE, GREY, "#c9c8c3"]
x = np.arange(len(kinds))
bottom = np.zeros(len(kinds))
for j in range(4):
    ax.bar(x, E[:, j], bottom=bottom, color=cols[j], label=names[j], width=0.6)
    bottom += E[:, j]
ax.set_xticks(x, list(kinds.keys()), fontsize=8)
ax.set(ylabel="share of the change's energy", title="What each whitened channel responds to (one voxel, 4 echoes)")
ax.legend(fontsize=7.5, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.suptitle("Figure 3. The front end turns 4 echoes into 2 physical channels plus a residual", x=0.01, ha="left",
             fontsize=11, fontweight="bold", y=1.04)
save(fig, "readme_3_channels.png")

# ------------------------------------------------------------------ 4. pipeline
fig, ax = plt.subplots(figsize=(13, 3.0))
ax.set_axis_off()
ax.set_xlim(0, 12.2)
ax.set_ylim(0, 3)
boxes = [("4 echoes\n(raw or motion-\ncorrected)", 0.1), ("brain mask\nfrom the echoes", 1.75),
         ("front end\nmu, T2*, sigma,\nwhitening\n-> c0, c1, r", 3.4), ("nuisance\nprojection\n(motion)", 5.25),
         ("rank\n(spectra,\nk sweep)", 6.9), ("joint ICA\non [c0 | c1]", 8.55), ("per component:\nq (dR2*), s (dS0),\neta, nulls", 10.2)]
for i, (t, x0) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((x0, 0.9), 1.45, 1.25, boxstyle="round,pad=0.04", fc="#eef4fc" if i % 2 == 0 else "#fdf0ea",
                                ec=GRID))
    ax.text(x0 + 0.725, 1.52, t, ha="center", va="center", fontsize=7.8, color=INK)
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x0 + 1.5, 1.52), (boxes[i + 1][1] - 0.05, 1.52), arrowstyle="-|>", mutation_scale=10,
                                     color=INK2))
ax.text(0.1, 0.45, "Every stage writes its outputs with a done.json (inputs, config hash, git SHA, versions, seeds). "
        "Tissue classes and traveling waves (branch claude/waves) build on the same cached front end.", fontsize=7.5, color=INK2)
fig.suptitle("Figure 4. The pipeline", x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, "readme_4_pipeline.png")

# ------------------------------------------------------------------ 5. a simulated decomposition
c5 = io.merge(cfg, {"simulate": {"echo_sigma_scale": list(RAT)}, "decompose": {"n_restarts": 10},
                    "metrics": {"n_boot": 10, "band_mc": 30, "n_null_pipeline": 5}})
src = [{"name": "dR2* only", "kind": "r2", "extent": "focal", "q": 0.5, "radius_vox": 2.5},
       {"name": "dS0 only", "kind": "s0", "extent": "focal", "s": 0.015, "radius_vox": 2.5},
       {"name": "mixed", "kind": "mixed", "extent": "focal", "q": 0.5, "s": 0.015, "radius_vox": 2.5}]
sim = simulate.simulate(c5, seed=3, overrides={"grid": [32, 32, 18], "n_volumes": 400, "spikes": {"n": 0},
                                              "skull_base": {"enabled": False}, "respiration": {"enabled": False},
                                              "sources": src})
fe = frontend.run(sim.data[sim.mask], sim.te_ms, c5, sim.mask)
dec = decompose.decompose(fe["c"][:, 0], fe["c"][:, 1], fe["R"], c5, 3, np.random.SeedSequence(1), n_jobs=2, mask3d=sim.mask)
me = metrics.measure(dec, fe["c"][:, 0], fe["c"][:, 1], fe["R"], c5, np.random.default_rng(0))
fig, ax = plt.subplots(3, 4, figsize=(11, 7.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
for i, s_ in enumerate(sim.sources):
    ct = [abs(np.corrcoef(s_["timecourse"], dec["time_courses"][:, j])[0, 1]) for j in range(3)]
    j = int(np.argmax(ct))
    ref = s_["q_map"] if np.abs(s_["q_map"]).max() > 0 else s_["s_map"]
    z = int(np.unravel_index(np.argmax(np.abs(ref)), ref.shape)[2])
    qv = np.full(sim.mask.shape, np.nan); qv[sim.mask] = dec["q_per_s"][:, j]
    sv = np.full(sim.mask.shape, np.nan); sv[sim.mask] = dec["s"][:, j] * 100
    sg = np.sign(np.corrcoef(s_["timecourse"], dec["time_courses"][:, j])[0, 1])
    tv = np.full(sim.mask.shape, np.nan); tv[sim.mask] = (ref[sim.mask] * (100 if ref is s_["s_map"] else 1))
    for a, (img, lab, lim) in zip(ax[i, :3], ((tv, "planted map", None), (sg * qv, "recovered q (dR2*, s^-1)", 0.6),
                                              (sg * sv, "recovered s (dS0, %)", 1.8))):
        L = lim or np.nanmax(np.abs(img[:, :, z])) or 1
        a.imshow(np.rot90(img[:, :, z]), cmap="RdBu_r", vmin=-L, vmax=L)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_title(lab if i == 0 else "", fontsize=8.5)
    ax[i, 0].set_ylabel(s_["name"], fontsize=9, color=INK, rotation=0, ha="right", va="center", labelpad=10)
    row = me["components"][j]
    ax[i, 3].plot(sg * dec["time_courses"][:120, j], color=BLUE, lw=1, label="recovered")
    ax[i, 3].plot(s_["timecourse"][:120], color=INK2, lw=0.8, ls="--", label="planted")
    ax[i, 3].set_title(f"eta = {row['eta']:.2f}  ({row['label']})", fontsize=9)
    ax[i, 3].tick_params(labelsize=7)
ax[0, 3].legend(fontsize=7, frameon=False)
ax[2, 3].set_xlabel("frame")
fig.suptitle("Figure 5. Joint ICA on simulated data: each component carries a dR2* map and a dS0 map\n"
             "eta = share of the component's physical energy that is dS0 (0 = pure dR2*, 1 = pure dS0). "
             f"Simulated: {sim.describe().split(';')[0]}", x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout()
save(fig, "readme_5_components.png")

# ------------------------------------------------------------------ 6. partial volume
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ft = np.linspace(1.0, 0.3, 36)
for ratio, col in ((0.5, ECHO[1]), (1.0, ECHO[2]), (2.0, ECHO[3])):
    etas = []
    for f in ft:
        St, Sc = f * np.exp(-TE / 45.0), (1 - f) * ratio * np.exp(-TE / 200.0)
        mu_ = St + Sc
        y = np.log((St * np.exp(-TE * 0.5e-3) + Sc) / mu_)[None, :, None]
        pr = frontend.project(y, ((mu_ / RAT) / (mu_[0] / 100))[None, :], np.ones((1, 4), bool), TE)
        c, R = pr["c"][0, :, 0], pr["R"][0]
        a = c[1] / R[1, 1]
        q = (c[0] - R[0, 1] * a) / R[0, 0]
        es0, er2 = (R[0, 1] ** 2 + R[1, 1] ** 2) * a ** 2, (R[0, 0] * q) ** 2
        etas.append(es0 / (es0 + er2))
    ax.plot(1 - ft, etas, color=col, lw=1.8, label=f"CSF / tissue signal at TE 0 = {ratio}")
ax.set(xlabel="CSF fraction of the voxel", ylabel="eta of a pure tissue dR2* change",
       title="A pure BOLD change looks partly dS0 when CSF shares the voxel")
ax.legend(fontsize=7.5, frameon=False)
fig.suptitle("Figure 6. Partial volume and eta (two-compartment model, tissue T2* 45 ms, CSF 200 ms)", x=0.01, ha="left",
             fontsize=10, fontweight="bold", y=1.03)
save(fig, "readme_6_partial_volume.png")
