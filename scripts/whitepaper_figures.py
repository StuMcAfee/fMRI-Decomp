#!/usr/bin/env python
"""White paper figures from the real data, pseudonymized.

Reads only results that already exist under outputs/ (route B, front end, injections, sensitivity analyses,
summaries) and writes figures and aggregate numbers to outputs/whitepaper/:
  fig_*.png       one figure per file; participants are labelled by pseudonym (outputs/pseudonyms.tsv), never by
                  subject number, run name, scan date or time
  numbers.json    aggregate numbers quoted in the text
  manifest.md     what was made, and anything that could not be made (with the reason)
Nothing is recomputed from the raw echoes. Each figure is made independently: one failure does not stop the rest.

Run on the cluster (login node is fine; about 2-5 minutes, 4 GB):
  PYTHONPATH=src ~/miniconda3_new/envs/mejoint/bin/python scripts/whitepaper_figures.py
or as a job: bash scripts/whitepaper_figures.lsf
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# ---------------------------------------------------------------- style (validated reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NETS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NET_LABEL = {"Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "Dorsal attention", "SalVentAttn": "Salience / ventral attention",
             "Limbic": "Limbic", "Cont": "Control", "Default": "Default"}
NET_COLOR = {n: SERIES[i] for i, n in enumerate(NETS)}
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32b2b"])
SEQ = LinearSegmentedColormap.from_list("blue", ["#f3f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2})


def save(fig, out: Path, name: str, made: list):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    made.append(name)


# ---------------------------------------------------------------- reading
def subject(run: str) -> str:
    return run.split("_")[0].replace("sub-", "")


def read_pseudonyms(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"STOP: {path} not found. The figures label participants by pseudonym only; create the key first.")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[str(row["subject"]).strip()] = row["pseudonym"].strip()
    return out


def load_runs(outputs: Path, key: dict, atlas: str) -> dict:
    """Every participant with a route B result: pseudonym -> dict of what was read."""
    runs = {}
    for f in sorted((outputs / "route_b").glob(f"sub-*/{atlas}/route_b.json")):
        run = f.parent.parent.name
        pid = key.get(subject(run))
        if pid is None:
            print("WARNING: a participant has no pseudonym in the key and is left out of every figure", file=sys.stderr)
            continue
        s = json.loads(f.read_text())
        d = {"run": run, "dir": f.parent, "summary": s, "pid": pid,
             "in_primary": (s.get("timeline") or {}).get("primary_eligible", True)}
        for k in ("corr_b", "corr_oc"):
            p = f.parent / f"{k}.npy"
            d[k] = np.load(p) if p.exists() else None
        tc = f.parent / "region_timecourses.npz"
        d["tc"] = dict(np.load(tc)) if tc.exists() else None
        fr = outputs / "phase2" / run / "mc" / "frames.tsv"
        if fr.exists():
            with open(fr) as fh:
                rr = list(csv.DictReader(fh, delimiter="\t"))
            try:
                d["fd"] = np.array([float(r["fd_filtered_mm"]) for r in rr])
            except (KeyError, ValueError):
                d["fd"] = None
        else:
            d["fd"] = None
        runs[pid] = d
    return dict(sorted(runs.items()))


def primary(d):
    return d["summary"]["integration"][0]


def cortex_rows(d):
    return [r for r in d["summary"]["regions"] if r["kind"] == "cortex" and r["used"]]


def network_order(ids, rows_by_id):
    """Region ids sorted by network (fixed order), then id; returns ids, network per id."""
    nets = [rows_by_id.get(i, {}).get("network", "") for i in ids]
    order = sorted(range(len(ids)), key=lambda j: (NETS.index(nets[j]) if nets[j] in NETS else 99, ids[j]))
    return [ids[j] for j in order], [nets[j] for j in order]


def fz(r):
    return np.arctanh(np.clip(np.asarray(r, float), -0.999, 0.999))


def without_first(C):
    from mejoint.route_b_batch import _without_first
    return _without_first(C)


def bandpass(x, tr, band):
    from scipy.signal import butter, filtfilt
    b, a = butter(2, [band[0] * 2 * tr, band[1] * 2 * tr], btype="band")
    return filtfilt(b, a, x, axis=-1)


def label_points(ax, xs, ys, labels):
    """Pseudonym labels beside points; a label that would sit on an earlier one moves down a line."""
    placed = []
    span = np.ptp(ys) or 1.0
    for x, y, lab in sorted(zip(xs, ys, labels), key=lambda v: -v[1]):
        dy = 3
        while any(abs(x - px) < 0.06 * (np.ptp(xs) or 1) and abs((y - py) / span) < 0.05 and pdy == dy for px, py, pdy in placed):
            dy -= 9
        placed.append((x, y, dy))
        ax.annotate(lab, (x, y), xytext=(4, dy), textcoords="offset points", fontsize=7, color=INK2)


# ---------------------------------------------------------------- figures
def fig_frontend_maps(runs, outputs, ex, out, made, nums):
    """The front end in one participant: mean signal, T2*, thermal noise, usable echoes, S0 sensitivity, residual."""
    import nibabel as nib
    d = runs[ex]
    fe = outputs / "phase2" / d["run"] / "mc"
    ld = lambda n: np.asarray(nib.load(str(fe / f"{n}.nii.gz")).dataobj, float)
    mask = ld("analysis_mask") > 0
    maps = [("mean_echo1_unmasked", "Mean signal, echo 1 (a.u.)", "gray", None),
            ("t2s_ms", "T2* (ms)", SEQ, (0, 100)),
            ("sigma", "Thermal noise SD, echo 1 (a.u.)", SEQ, None),
            ("n_usable_echoes", "Usable echoes", SEQ, (0, 4)),
            ("R", "S0 sensitivity R11 (a.u.)", SEQ, None),
            ("resid_excess", "Residual energy / thermal", SEQ, (0, 3))]
    zs = np.flatnonzero(mask.any((0, 1)))
    slices = np.linspace(zs[0] + 0.12 * (zs[-1] - zs[0]), zs[0] + 0.85 * (zs[-1] - zs[0]), 6).round().astype(int)
    fig, axes = plt.subplots(len(maps), len(slices), figsize=(1.45 * len(slices) + 1.2, 1.55 * len(maps)))
    for i, (name, label, cmap, lim) in enumerate(maps):
        v = ld(name)
        if v.ndim == 4:
            v = v[..., 2]                                            # R holds R00, R01, R11
        if name != "mean_echo1_unmasked":
            v = np.where(mask, v, np.nan)
        lo, hi = lim if lim else (np.nanpercentile(v[mask], 2), np.nanpercentile(v[mask], 98))
        for j, z in enumerate(slices):
            ax = axes[i, j]
            im = ax.imshow(np.rot90(v[:, :, z]), cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f"slice {z}", fontsize=8, fontweight="normal", color=INK2)
        cb = fig.colorbar(im, ax=axes[i, :], fraction=0.015, pad=0.01)
        cb.ax.tick_params(labelsize=7)
        axes[i, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
    fig.suptitle(f"Front end in one participant ({ex}): inferior (left) to superior (right)", x=0.02, ha="left",
                 fontsize=10, fontweight="bold")
    save(fig, out, "fig_frontend_maps", made)
    t2 = ld("t2s_ms")[mask]
    nu = ld("n_usable_echoes")[mask]
    nums["frontend_example"] = {"pid": ex, "t2s_median_ms": round(float(np.nanmedian(t2)), 1),
                                "t2s_iqr_ms": [round(float(np.nanpercentile(t2, 25)), 1), round(float(np.nanpercentile(t2, 75)), 1)],
                                "fraction_voxels_4_echoes": round(float(np.mean(nu >= 4)), 3),
                                "fraction_voxels_le2_echoes": round(float(np.mean(nu <= 2)), 3), "n_voxels": int(mask.sum())}


def fig_timecourses(runs, cfg, ex, out, made, nums):
    """Region time courses split into the R2*-like (BOLD-like) and S0-like parts, with the leftover noise level."""
    d = runs[ex]
    tc = d["tc"]
    tr = float(cfg["acquisition"]["tr_s"])
    band = cfg["regions"]["band_hz"]
    rows = {r["id"]: r for r in cortex_rows(d)}
    pick = []
    for net in ("Vis", "SomMot", "Default"):
        cand = [r for r in rows.values() if r["network"] == net and r.get("b_reliability") is not None]
        if cand:
            pick.append(max(cand, key=lambda r: r["b_reliability"]))
    T = tc["b"].shape[1]
    n = min(T, int(240 / tr))
    t = np.arange(n) * tr
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
    for k, r in enumerate(pick):
        b = bandpass(tc["b"][r["id"] - 1], tr, band)[:n]                 # already 1/s (regions.split)
        a = bandpass(tc["a"][r["id"] - 1], tr, band)[:n] * 100.0         # ln S0 -> %
        lab = f"{NET_LABEL[r['network']]} region (reliability {r['b_reliability']:.2f})"
        axes[0].plot(t, b + 0.0 * k, color=NET_COLOR[r["network"]], lw=1.4, label=lab)
        axes[1].plot(t, a, color=NET_COLOR[r["network"]], lw=1.4, label=lab)
    axes[0].set_ylabel("ΔR2* (1/s)")
    axes[1].set_ylabel("ΔS0/S0 (%)")
    axes[1].set_xlabel("time (s)")
    axes[0].set_title("BOLD-like (R2*-like) part, band-passed")
    axes[1].set_title("S0-like part, band-passed (same regions)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=7, ncol=3)
    fig.suptitle(f"Region signals in one participant ({ex}): first 4 minutes", x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_timecourses", made)
    sd = lambda key, scale: [round(float(np.std(bandpass(tc[key][r["id"] - 1], tr, band)) * scale), 3) for r in pick]
    nums["timecourse_example"] = {"pid": ex, "regions": [r["network"] for r in pick], "sd_dR2s_per_s": sd("b", 1.0),
                                  "sd_dS0_percent": sd("a", 100.0)}


def fig_shares(runs, out, made, nums):
    """Region shares (BOLD-like, S0-like, leftover) and reliability by network, all participants."""
    keys = [("bold_share", "BOLD-like share", (0, 1)), ("s0_share", "S0-like share", (0, 0.6)),
            ("leftover_share", "Leftover share", None), ("b_reliability", "Reliability (signal share)", (0, 1))]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
    rng = np.random.default_rng(0)
    nums["shares_by_network"] = {}
    for ax, (k, lab, lim) in zip(axes, keys):
        for i, net in enumerate(NETS):
            v = np.array([r[k] for d in runs.values() for r in cortex_rows(d) if r["network"] == net and r.get(k) is not None], float)
            if not v.size:
                continue
            y = np.full(v.size, i) + rng.uniform(-0.28, 0.28, v.size)
            ax.scatter(v, y, s=3, color=NET_COLOR[net], alpha=0.35, linewidths=0)
            q = np.percentile(v, [25, 50, 75])
            ax.plot([q[0], q[2]], [i, i], color=INK, lw=2.2, solid_capstyle="round")
            ax.plot(q[1], i, "o", ms=6, color=INK, mec=SURF, mew=1.5)
            nums["shares_by_network"].setdefault(net, {})[k] = round(float(q[1]), 3)
        ax.set_title(lab)
        if lim:
            ax.set_xlim(*lim)
        else:
            allv = np.array([r[k] for d in runs.values() for r in cortex_rows(d) if r.get(k) is not None], float)
            lo_, hi_ = np.percentile(allv, [1, 99])
            ax.set_xlim(lo_ - 0.02, hi_ + 0.02)
            n_out = int(np.sum((allv < lo_ - 0.02) | (allv > hi_ + 0.02)))
            ax.set_title(f"{lab}\n({n_out} values off axis)" if n_out else lab)
        if k == "b_reliability":
            ax.axvline(0.5, color=INK2, lw=1, ls="--")
    axes[0].set_yticks(range(len(NETS)))
    axes[0].set_yticklabels([NET_LABEL[n] for n in NETS])
    axes[0].invert_yaxis()
    fig.suptitle(f"Cortical regions of {len(runs)} participants (dots: regions; black: median and interquartile range)",
                 x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_shares_by_network", made)
    allb = [r["bold_share"] for d in runs.values() for r in cortex_rows(d) if r.get("bold_share") is not None]
    rel = [r["b_reliability"] for d in runs.values() for r in cortex_rows(d) if r.get("b_reliability") is not None]
    nums["shares_overall"] = {"bold_share_median": round(float(np.median(allb)), 3),
                              "bold_share_iqr": [round(float(np.percentile(allb, 25)), 3), round(float(np.percentile(allb, 75)), 3)],
                              "reliability_median": round(float(np.median(rel)), 3),
                              "fraction_below_reliability_floor": round(float(np.mean(np.array(rel) < 0.5)), 3),
                              "n_region_values": len(allb)}


def fig_surface(runs, atlas, out, made):
    """Group-median shares and reliability painted on the cortical surface (display only)."""
    from mejoint.regions import surface_figure
    by = {}
    for d in runs.values():
        for r in d["summary"]["regions"]:
            if r["kind"] == "cortex":
                by.setdefault(r["id"], []).append(r)
    rows = []
    for i, rr in sorted(by.items()):
        used = [r for r in rr if r["used"]]
        med = lambda k: float(np.median([r[k] for r in used if r.get(k) is not None])) if any(r.get(k) is not None for r in used) else None
        rows.append({"id": i, "kind": "cortex", "hemisphere": rr[0].get("hemisphere"), "used": len(used) >= len(rr) / 2,
                     "bold_share": med("bold_share"), "s0_share": med("s0_share"), "leftover_share": med("leftover_share"),
                     "b_reliability": med("b_reliability")})
    surface_figure(rows, atlas, out / "fig_surface_group_median.png",
                   title=f"Group median over {len(runs)} participants (parcels used in fewer than half are grey)")
    made.append("fig_surface_group_median")


def common_ids(runs):
    sets = [set(d["summary"]["cortex_ids"]) for d in runs.values() if d["corr_b"] is not None]
    return sorted(set.intersection(*sets)) if sets else []


def sub_matrix(d, key, ids):
    C = d[key]
    pos = {v: i for i, v in enumerate(d["summary"]["cortex_ids"])}
    ix = [pos[i] for i in ids]
    return np.asarray(C, float)[np.ix_(ix, ix)]


def fig_matrices(runs, out, made, nums):
    """Group-mean connectivity on the common parcels: BOLD-like, ordinary (OC-like), difference, without the first pattern."""
    P = {k: d for k, d in runs.items() if d["in_primary"] and d["corr_b"] is not None and d["corr_oc"] is not None}
    ids = common_ids(P)
    rows_by_id = {r["id"]: r for d in P.values() for r in d["summary"]["regions"]}
    ids, nets = network_order(ids, rows_by_id)
    mean = lambda mats: np.tanh(np.nanmean([fz(M) for M in mats], axis=0))
    B = mean([sub_matrix(d, "corr_b", ids) for d in P.values()])
    O = mean([sub_matrix(d, "corr_oc", ids) for d in P.values()])
    W = mean([without_first(sub_matrix(d, "corr_b", ids)) for d in P.values()])
    for M in (B, O, W):
        np.fill_diagonal(M, np.nan)
    panels = [(B, "BOLD-like (R2*-like)", 1), (O, "Ordinary (OC-like channel)", 1), (B - O, "BOLD-like minus ordinary", 0.3),
              (W, "BOLD-like, first pattern removed", 0.5)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    bounds = [0] + [j for j in range(1, len(nets)) if nets[j] != nets[j - 1]] + [len(nets)]
    for ax, (M, title, lim) in zip(axes, panels):
        im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.grid(False)
        for bnd in bounds[1:-1]:
            ax.axhline(bnd - 0.5, color=INK, lw=0.5); ax.axvline(bnd - 0.5, color=INK, lw=0.5)
        mids = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)]
        ax.set_xticks(mids); ax.set_xticklabels([nets[bounds[i]] for i in range(len(bounds) - 1)], rotation=90, fontsize=7)
        ax.set_yticks(mids); ax.set_yticklabels([nets[bounds[i]] for i in range(len(bounds) - 1)], fontsize=7)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    fig.suptitle(f"Mean noise-corrected correlations over {len(P)} participants, {len(ids)} common parcels (Fisher-z average)",
                 x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_matrices", made)
    iu = np.triu_indices(len(ids), 1)
    same = np.array([nets[a] == nets[b] for a, b in zip(*iu)])
    nums["matrices"] = {"n_participants": len(P), "n_common": len(ids),
                        "mean_r_bold": round(float(np.nanmean(B[iu])), 3), "mean_r_ordinary": round(float(np.nanmean(O[iu])), 3),
                        "mean_r_bold_within_network": round(float(np.nanmean(B[iu][same])), 3),
                        "mean_r_bold_between_network": round(float(np.nanmean(B[iu][~same])), 3),
                        "mean_r_without_first_within": round(float(np.nanmean(W[iu][same])), 3),
                        "mean_r_without_first_between": round(float(np.nanmean(W[iu][~same])), 3)}


def fig_spectra(runs, out, made):
    """Eigenvalue spectra of each participant's BOLD-like correlations against the circular-shift null."""
    P = [d for d in runs.values() if primary(d).get("eigenvalues")]
    n = len(P)
    cols = min(6, n)
    rowsn = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rowsn, cols, figsize=(2.1 * cols, 1.9 * rowsn), sharex=True, sharey=True, squeeze=False)
    for ax, d in zip(axes.ravel(), P):
        i = primary(d)
        e = np.array(i["eigenvalues"], float)
        q = np.array(i.get("null_eig_q95") or [], float)
        k = np.arange(1, e.size + 1)
        if q.size:
            ax.plot(np.arange(1, q.size + 1), q, color=INK2, lw=1.2, ls="--", label="null 95 %")
        ax.plot(k, e, "o-", color=SERIES[0], ms=2.5, lw=1.4, label="data")
        ax.set_yscale("log")
        ax.set_title(f"{d['pid']}" + ("" if d["in_primary"] else " (not in primary)"), fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("eigenvalue index")
    for ax in axes[:, 0]:
        ax.set_ylabel("eigenvalue")
    fig.suptitle("Eigenvalue spectra of BOLD-like correlations (first 30) and the circular-shift null", x=0.02, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_spectra", made)


def participant_table(runs):
    T = []
    for pid, d in runs.items():
        i = primary(d)
        T.append({"pid": pid, "in_primary": d["in_primary"], "pr1": i.get("pr_without_first"), "pr1_ci": i.get("pr_without_first_ci"),
                  "pr1_null": i.get("pr_without_first_null_q05_q95"), "pr1_null_med": i.get("pr_without_first_null_median"),
                  "pr": i.get("participation_ratio"), "pr_null": i.get("pr_null_q05_q95"), "pr_null_med": i.get("pr_null_median"),
                  "gs": i.get("global_share"), "seg": i.get("segregation"), "pr_ci": i.get("participation_ratio_ci"),
                  "gs_ci": i.get("global_share_ci"), "bias": i.get("pr_without_first_boot_bias"),
                  "fd": float(np.mean(d["fd"])) if d["fd"] is not None else None})
    return T


def fig_primary(runs, out, made, nums):
    """Per participant: the primary with its interval and null, the ordinary PR with its null, the global share."""
    T = participant_table(runs)
    y = np.arange(len(T))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 0.36 * len(T) + 1.4), sharey=True)
    specs = [("pr1", "pr1_ci", "pr1_null", "pr1_null_med", "PR without first pattern (primary)"),
             ("pr", "pr_ci", "pr_null", "pr_null_med", "Ordinary participation ratio"),
             ("gs", "gs_ci", None, None, "Global BOLD-like share")]
    for ax, (k, kci, knull, kmed, title) in zip(axes, specs):
        for j, t in enumerate(T):
            if knull and t.get(knull):
                ax.plot(t[knull], [j, j], color=GRID, lw=7, solid_capstyle="round")
                ax.plot(t[kmed], j, "|", color=INK2, ms=8)
            if t.get(kci):
                ax.plot(t[kci], [j, j], color=SERIES[0], lw=1.6)
            if t.get(k) is not None:
                ax.plot(t[k], j, "o", ms=6, color=SERIES[0] if t["in_primary"] else SURF, mec=SERIES[0], mew=1.5)
        ax.set_title(title, fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([t["pid"] + ("" if t["in_primary"] else "*") for t in T])
    axes[0].invert_yaxis()
    fig.suptitle("Integration measures per participant (dot: estimate; blue line: 95 % bootstrap interval; grey band: "
                 "circular-shift null 5-95 %, tick: null median; * hollow: not in primary)", x=0.02, ha="left",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_primary", made)
    P = [t for t in T if t["in_primary"] and t["pr1"] is not None]
    hw = [(t["pr1_ci"][1] - t["pr1_ci"][0]) / 2 for t in P if t.get("pr1_ci")]
    above = [t["pr1"] > t["pr1_null"][1] for t in P if t.get("pr1_null")]
    below = [t["pr1"] < t["pr1_null"][0] for t in P if t.get("pr1_null")]
    below_pr = [t["pr"] < t["pr_null"][0] for t in P if t.get("pr_null")]
    nmed = [t["pr1_null_med"] for t in P if t.get("pr1_null_med") is not None]
    nums["primary"] = {"n": len(P), "median": round(float(np.median([t["pr1"] for t in P])), 2),
                       "range": [round(min(t["pr1"] for t in P), 2), round(max(t["pr1"] for t in P), 2)],
                       "ci_halfwidth_median": round(float(np.median(hw)), 2) if hw else None,
                       "ci_halfwidth_range": [round(min(hw), 2), round(max(hw), 2)] if hw else None,
                       "n_above_null_q95": int(sum(above)), "n_below_null_q05": int(sum(below)), "n_with_null": len(above),
                       "n_pr_below_null_q05": int(sum(below_pr)),
                       "null_median_range": [round(min(nmed), 1), round(max(nmed), 1)] if nmed else None,
                       "boot_bias_median": round(float(np.median([t["bias"] for t in P if t.get("bias") is not None])), 2) if any(t.get("bias") is not None for t in P) else None,
                       "global_share_range": [round(min(t["gs"] for t in P), 3), round(max(t["gs"] for t in P), 3)],
                       "pr_range": [round(min(t["pr"] for t in P), 2), round(max(t["pr"] for t in P), 2)]}


def fig_global(runs, out, made, nums):
    """Why the primary leaves out the first pattern: ordinary PR tracks the global share, the primary does not."""
    from scipy.stats import spearmanr
    T = [t for t in participant_table(runs) if t["in_primary"] and t["gs"] is not None]
    gs = np.array([t["gs"] for t in T])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, k, lab, col in ((axes[0], "pr", "Ordinary participation ratio", SERIES[1]),
                            (axes[1], "pr1", "PR without first pattern (primary)", SERIES[0])):
        v = np.array([t[k] for t in T])
        rho = spearmanr(gs, v)[0]
        ax.scatter(gs, v, s=40, color=col, edgecolors=SURF, linewidths=1.5, zorder=3)
        label_points(ax, gs, v, [t["pid"] for t in T])
        ax.set_xlabel("global BOLD-like share")
        ax.set_ylabel(lab)
        ax.set_title(f"{lab}\nSpearman ρ = {rho:.2f}, n = {len(T)}", fontsize=9)
        nums.setdefault("global", {})[f"{k}_vs_global_share_spearman"] = round(float(rho), 3)
    fd = np.array([t["fd"] if t["fd"] is not None else np.nan for t in T])
    ok = np.isfinite(fd)
    rho = spearmanr(fd[ok], gs[ok])[0]
    axes[2].scatter(fd[ok], gs[ok], s=40, color=SERIES[2], edgecolors=SURF, linewidths=1.5, zorder=3)
    axes[2].set_xlabel("mean framewise displacement, filtered (mm)")
    axes[2].set_ylabel("global BOLD-like share")
    axes[2].set_title(f"Global share and motion\nSpearman ρ = {rho:.2f}", fontsize=9)
    nums["global"]["global_share_vs_fd_spearman"] = round(float(rho), 3)
    fig.tight_layout()
    save(fig, out, "fig_global_share", made)


def fig_qcfc(runs, atlas, out, made, nums):
    """QC-FC: edges correlated with motion across participants, before and after removing the first pattern."""
    from scipy.stats import pearsonr, spearmanr

    from mejoint.route_b_batch import _centroids
    P = {k: d for k, d in runs.items() if d["in_primary"] and d["corr_b"] is not None and d["fd"] is not None}
    ids = common_ids(P)
    cen = _centroids(atlas)
    iu = np.triu_indices(len(ids), 1)
    dist = np.array([np.linalg.norm(cen[ids[a]] - cen[ids[b]]) for a, b in zip(*iu)])
    fd = np.array([float(np.mean(d["fd"])) for d in P.values()])
    res = {}
    for name, f in (("ordinary", lambda M: M), ("without_first", without_first)):
        Z = np.array([fz(f(sub_matrix(d, "corr_b", ids))[iu]) for d in P.values()])
        ok = np.all(np.isfinite(Z), axis=0)
        r = np.full(Z.shape[1], np.nan)
        p = np.full(Z.shape[1], np.nan)
        for j in np.flatnonzero(ok):
            r[j], p[j] = pearsonr(Z[:, j], fd)
        res[name] = (r, p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.linspace(-1, 1, 41)
    for (name, (r, p)), col, lab in zip(res.items(), (SERIES[1], SERIES[0]), ("BOLD-like correlations", "first pattern removed")):
        m = np.isfinite(r)
        axes[0].hist(r[m], bins=bins, color=col, alpha=0.55, label=f"{lab}: {np.mean(p[m] < 0.05) * 100:.1f} % of edges p < 0.05")
        axes[1].scatter(dist[m], r[m], s=1.5, color=col, alpha=0.25, linewidths=0, label=lab)
        nums.setdefault("qcfc", {})[name] = {"median_r": round(float(np.median(r[m])), 3),
                                             "frac_p_below_0p05": round(float(np.mean(p[m] < 0.05)), 3),
                                             "distance_spearman": round(float(spearmanr(dist[m], r[m])[0]), 3)}
    axes[0].axvline(0, color=INK2, lw=1)
    axes[0].set_xlabel("correlation of edge strength with mean FD across participants")
    axes[0].set_ylabel("edges")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].set_title(f"QC-FC, {len(P)} participants, {len(ids)} common parcels", fontsize=9)
    axes[1].axhline(0, color=INK2, lw=1)
    axes[1].set_xlabel("distance between parcel centroids (mm, MNI)")
    axes[1].set_ylabel("QC-FC correlation")
    axes[1].set_title("Distance dependence", fontsize=9)
    leg = axes[1].legend(fontsize=7, markerscale=6)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout()
    save(fig, out, "fig_qcfc", made)
    nums["qcfc"]["n_participants"] = len(P)


def fig_motion(runs, ex, cfg, out, made):
    """Filtered framewise displacement in one participant and the distribution of mean FD across participants."""
    d = runs[ex]
    tr = float(cfg["acquisition"]["tr_s"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), gridspec_kw={"width_ratios": [3, 1]})
    if d["fd"] is not None:
        t = np.arange(d["fd"].size) * tr
        axes[0].plot(t, d["fd"], color=SERIES[0], lw=1)
        axes[0].axhline(0.2, color=INK2, lw=1, ls="--")
        axes[0].text(t[-1], 0.205, "0.2 mm", ha="right", va="bottom", fontsize=7, color=INK2)
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("FD, filtered (mm)")
        axes[0].set_title(f"Framewise displacement after respiratory filtering ({ex})", fontsize=9)
    v = [(pid, float(np.mean(x["fd"]))) for pid, x in runs.items() if x["fd"] is not None]
    axes[1].barh([p for p, _ in v], [m for _, m in v], color=SERIES[0], height=0.6)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean FD (mm)")
    axes[1].set_title("All participants", fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig_motion", made)


def fig_sensitivity(outputs, key, out, made, nums):
    """Second-pass sensitivity analyses: does each participant's primary, and the ranking, hold?"""
    s = json.loads((outputs / "sensitivity" / "sensitivity.json").read_text())
    names = {"atlas_400_17": "400 parcels / 17 networks", "band_008": "band 0.01-0.08 Hz", "censor": "censoring FD > 0.2 mm",
             "equal_weighting": "equal echo weights", "gm05": "grey-matter threshold 0.5", "gsr": "global signal regression",
             "motion_lowpass6": "6 low-passed motion parameters"}
    V = [r["variant"] for r in s["variants"]]
    tab = [t for t in s["participants"] if t["in_primary"] and key.get(subject(t["run"]))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(V) + 1)
    for t in tab:
        vals = [t["main"]] + [t.get(v) for v in V]
        axes[0].plot(x, vals, "-o", color=SERIES[0], alpha=0.45, ms=3, lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["main"] + [names.get(v, v) for v in V], rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("PR without first pattern")
    axes[0].set_title(f"Primary per participant across analysis choices (n = {len(tab)})", fontsize=9)
    w = 0.26
    for j, (k, lab) in enumerate((("primary_spearman_with_main", "primary"), ("global_share_spearman_with_main", "global share"),
                                  ("pr_spearman_with_main", "ordinary PR"))):
        axes[1].barh(np.arange(len(V)) + (j - 1) * w, [r.get(k) or 0 for r in s["variants"]], height=w, color=SERIES[j], label=lab)
    axes[1].set_yticks(np.arange(len(V)))
    axes[1].set_yticklabels([names.get(v, v) for v in V], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Spearman with the main analysis")
    axes[1].set_title("Does the ranking of participants hold?", fontsize=9)
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    save(fig, out, "fig_sensitivity", made)
    nums["sensitivity"] = {r["variant"]: {k: r.get(k) for k in ("n", "primary_spearman_with_main", "primary_median_change",
                                                                 "global_share_spearman_with_main", "pr_spearman_with_main",
                                                                 "pr_vs_main_primary_spearman", "censored_fraction_median_max")}
                           for r in s["variants"]}


def fig_injection_regions(outputs, key, out, made, nums):
    """Route B injection validation (R3/R5): known region-wide signals added to the raw echoes at resting strength."""
    recs = []
    for f in sorted((outputs / "inject").glob("sub-*/regions_*_s*/recovery_regions.json")):
        o = json.loads(f.read_text())
        pid = key.get(subject(f.parent.parent.name))
        if pid is None or o.get("level") != "weak":
            continue
        recs.append((pid, o))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    types = sorted({t for _, o in recs for t in o["by_type"]})
    for j, t in enumerate(types):
        tr_ = [o["by_type"][t]["share_true"] for _, o in recs if t in o["by_type"] and o["by_type"][t].get("share_true") is not None]
        rc = [o["by_type"][t]["share_injected_median"] for _, o in recs if t in o["by_type"] and o["by_type"][t].get("share_true") is not None]
        axes[0].scatter(tr_, rc, s=26, color=SERIES[j % 8], edgecolors=SURF, linewidths=1, label=t, zorder=3)
    axes[0].plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    axes[0].fill_between([0, 1], [-0.1, 0.9], [0.1, 1.1], color=GRID, alpha=0.6, lw=0, label="tolerance ±0.1")
    axes[0].set_xlim(-0.02, 1.02); axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_xlabel("true BOLD-like share of the injected signal")
    axes[0].set_ylabel("recovered (median over regions)")
    axes[0].set_title("Share recovery by injected type", fontsize=9)
    axes[0].legend(fontsize=7)
    conds = ["independent", "shared"]
    for j, c in enumerate(conds):
        v = [o["dimensionality"]["pr_without_first"] for _, o in recs if o.get("condition") == c]
        dv = [b - a for a, b in v if a is not None and b is not None]
        axes[1].scatter(np.full(len(dv), j) + np.random.default_rng(j).uniform(-0.12, 0.12, len(dv)), dv, s=26,
                        color=SERIES[j], edgecolors=SURF, linewidths=1, zorder=3)
        if dv:
            axes[1].plot([j - 0.25, j + 0.25], [np.median(dv)] * 2, color=INK, lw=2)
        nums.setdefault("injection_regions", {})[f"d_pr_without_first_{c}"] = [round(float(x), 2) for x in sorted(dv)]
    axes[1].axhline(0, color=INK2, lw=1)
    axes[1].set_xticks(range(len(conds)))
    axes[1].set_xticklabels(["independent networks", "one shared network"])
    axes[1].set_ylabel("change in PR without first pattern")
    axes[1].set_title("Effect of injected structure on the primary", fontsize=9)
    keys = [("within_injected_networks", "within injected"), ("between_injected_networks", "between injected"),
            ("injected_to_other", "injected to other"), ("other_pairs", "other pairs")]
    for j, c in enumerate(conds):
        vals = [[o["coupling"].get(k) for _, o in recs if o.get("condition") == c and o["coupling"].get(k) is not None] for k, _ in keys]
        pos = np.arange(len(keys)) + (j - 0.5) * 0.35
        axes[2].bar(pos, [np.median(v) if v else 0 for v in vals], width=0.33, color=SERIES[j], label=conds[j])
    axes[2].axhline(0, color=INK2, lw=1)
    axes[2].set_xticks(range(len(keys)))
    axes[2].set_xticklabels([lab for _, lab in keys], rotation=20, ha="right", fontsize=7)
    axes[2].set_ylabel("median change in BOLD-like correlation")
    axes[2].set_title("Where coupling changed", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.suptitle(f"Injection into the raw echoes at resting strength ({len({p for p, _ in recs})} participants)", x=0.02,
                 ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_injection_regions", made)
    errs = [abs(v["share_injected_median"] - v["share_true"]) for _, o in recs for v in o["by_type"].values()
            if v.get("share_injected_median") is not None and v.get("share_true") is not None]
    nums["injection_regions"].update({"n_recoveries": len(recs), "share_abs_error_median": round(float(np.median(errs)), 3) if errs else None,
                                      "share_abs_error_max": round(float(np.max(errs)), 3) if errs else None,
                                      "fraction_within_0p1": round(float(np.mean(np.array(errs) <= 0.1)), 3) if errs else None})
    sb = [b for _, o in recs for b in o.get("size_bins", [])]
    if sb:
        bins = sorted({b["voxels"] for b in sb}, key=lambda s: int(s.split("-")[0]))
        nums["injection_regions"]["share_error_by_region_size"] = {
            b: round(float(np.median([x["median_abs_share_error"] for x in sb if x["voxels"] == b and x["median_abs_share_error"] is not None])), 3)
            for b in bins}


def fig_injection_ica(outputs, out, made, nums):
    """Whole-brain spatial ICA (Phase 5): which injected sources were recovered, by layout, type, strength and k."""
    s = json.loads((outputs / "inject" / "inject_summary.json").read_text())
    tab = s["table"]
    ks = sorted({r["k"] for r in tab})
    rowsk = sorted({(r["layout"], r["type"], r["level"]) for r in tab}, key=lambda x: (x[0], x[1], x[2] != "strong"))
    M = np.full((len(rowsk), len(ks)), np.nan)
    R = np.full_like(M, np.nan)
    for r in tab:
        i, j = rowsk.index((r["layout"], r["type"], r["level"])), ks.index(r["k"])
        M[i, j] = r["n_recovered"] / r["n"] if r["n"] else np.nan
        R[i, j] = r.get("best_r_median") if r.get("best_r_median") is not None else np.nan
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.32 * len(rowsk) + 1.6), sharey=True)
    for ax, A, title, lim in ((axes[0], M, "Fraction recovered (best |r| ≥ 0.9)", (0, 1)), (axes[1], R, "Best match |r| (median)", (0, 1))):
        im = ax.imshow(A, cmap=SEQ, vmin=lim[0], vmax=lim[1], aspect="auto")
        ax.grid(False)
        for (i, j), v in np.ndenumerate(A):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=INK if v < 0.6 else SURF)
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("ICA dimension k")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7)
    axes[0].set_yticks(range(len(rowsk)))
    axes[0].set_yticklabels([f"{a}, {b}, {c}" for a, b, c in rowsk], fontsize=7)
    fig.suptitle("Whole-brain spatial ICA on injected sources (layout, type, strength)", x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_injection_ica", made)
    nums["injection_ica"] = {"n_recoveries": s.get("n_recoveries"),
                             "recovered_by_layout": {lay: [int(sum(r["n_recovered"] for r in tab if r["layout"] == lay)),
                                                           int(sum(r["n"] for r in tab if r["layout"] == lay))]
                                                     for lay in sorted({r["layout"] for r in tab})}}


def fig_deep(outputs, out, made, nums):
    """Thalamus and striatum coupling with the cortical networks, pooled over participants where measurable."""
    s = json.loads((outputs / "route_b" / "summary" / "summary.json").read_text())
    pooled = s["deep_pooled"]
    regs = sorted({p["region"] for p in pooled})
    fig, axes = plt.subplots(1, len(regs), figsize=(2.7 * len(regs), 3.2), sharey=True, sharex=True)
    for ax, rg in zip(np.atleast_1d(axes), regs):
        for i, net in enumerate(NETS):
            p = next((x for x in pooled if x["region"] == rg and x["network"] == net), None)
            if not p or p.get("pooled") is None:
                continue
            ax.plot(p["ci"], [i, i], color=NET_COLOR[net], lw=2)
            ax.plot(p["pooled"], i, "o", color=NET_COLOR[net], ms=6, mec=SURF, mew=1.2)
        n = next((x for x in pooled if x["region"] == rg), {})
        ax.set_title(f"{rg.replace('_', ' ')}\nmeasurable in {n.get('n')} of {n.get('n_total')}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("pooled BOLD-like coupling (r)")
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(NETS)))
    ax0.set_yticklabels([NET_LABEL[n] for n in NETS], fontsize=7)
    ax0.invert_yaxis()
    fig.tight_layout()
    save(fig, out, "fig_deep_coupling", made)
    nums["deep"] = {rg: next((x["n"] for x in pooled if x["region"] == rg), None) for rg in regs}


def fig_bootstrap_bias(runs, out, made, nums):
    """How strongly the primary depends on the effective number of frames: the bootstrap bias per participant."""
    T = [t for t in participant_table(runs) if t["in_primary"] and t.get("bias") is not None]
    if not T:
        raise RuntimeError("no bootstrap bias in route_b.json (rerun route B with the current code)")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter([t["pr1"] for t in T], [t["bias"] for t in T], s=40, color=SERIES[0], edgecolors=SURF, linewidths=1.5)
    label_points(ax, np.array([t["pr1"] for t in T]), np.array([t["bias"] for t in T]), [t["pid"] for t in T])
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("PR without first pattern")
    ax.set_ylabel("bootstrap bias (resamples - estimate)")
    ax.set_title("Fewer independent frames lower the measure", fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig_bootstrap_bias", made)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default=str(REPO / "outputs"))
    ap.add_argument("--out", default=None, help="default: <outputs>/whitepaper")
    ap.add_argument("--example", default="P04", help="pseudonym of the participant shown in the single-participant figures")
    ap.add_argument("--atlas", default="200_7")
    a = ap.parse_args()
    from mejoint import io
    cfg = io.load_config()
    outputs = Path(a.outputs)
    out = Path(a.out) if a.out else outputs / "whitepaper"
    out.mkdir(parents=True, exist_ok=True)
    key = read_pseudonyms(outputs / "pseudonyms.tsv")
    runs = load_runs(outputs, key, a.atlas)
    if not runs:
        sys.exit("STOP: no route B results found under outputs/route_b")
    ex = a.example if a.example in runs else next(iter(runs))
    made, failed, nums = [], [], {"n_participants": len(runs), "n_in_primary": sum(d["in_primary"] for d in runs.values()),
                                  "pseudonyms": list(runs), "not_in_primary": [k for k, d in runs.items() if not d["in_primary"]],
                                  "example": ex}
    steps = [("front-end maps", lambda: fig_frontend_maps(runs, outputs, ex, out, made, nums)),
             ("region time courses", lambda: fig_timecourses(runs, cfg, ex, out, made, nums)),
             ("shares by network", lambda: fig_shares(runs, out, made, nums)),
             ("surface maps", lambda: fig_surface(runs, a.atlas, out, made)),
             ("connectivity matrices", lambda: fig_matrices(runs, out, made, nums)),
             ("eigenvalue spectra", lambda: fig_spectra(runs, out, made)),
             ("primary per participant", lambda: fig_primary(runs, out, made, nums)),
             ("global share", lambda: fig_global(runs, out, made, nums)),
             ("QC-FC", lambda: fig_qcfc(runs, a.atlas, out, made, nums)),
             ("motion", lambda: fig_motion(runs, ex, cfg, out, made)),
             ("sensitivity analyses", lambda: fig_sensitivity(outputs, key, out, made, nums)),
             ("route B injections", lambda: fig_injection_regions(outputs, key, out, made, nums)),
             ("whole-brain ICA injections", lambda: fig_injection_ica(outputs, out, made, nums)),
             ("thalamus and striatum", lambda: fig_deep(outputs, out, made, nums)),
             ("bootstrap bias", lambda: fig_bootstrap_bias(runs, out, made, nums))]
    for name, fn in steps:
        try:
            fn()
            print(f"  made: {name}")
        except Exception as exc:                                       # report and continue
            plt.close("all")
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED: {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
    io.write_json(out / "numbers.json", nums)
    L = ["# White paper figures (pseudonymized)", "", f"{len(runs)} participants ({nums['n_in_primary']} in the primary); "
         f"single-participant figures show {ex}.", "", "## Made", ""] + [f"- {m}.png" for m in made]
    if failed:
        L += ["", "## Not made", ""] + [f"- {n}: {e}" for n, e in failed]
    L += ["", "numbers.json holds the aggregate numbers quoted in the text."]
    io.write_text(out / "manifest.md", "\n".join(L) + "\n")
    print(f"\n{len(made)} figures in {out}; {len(failed)} not made (see manifest.md)")


if __name__ == "__main__":
    main()
