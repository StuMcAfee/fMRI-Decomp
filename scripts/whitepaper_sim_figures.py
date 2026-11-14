#!/usr/bin/env python
"""White paper figures from simulations and closed-form calculations (no participant data).

  PYTHONPATH=src python scripts/whitepaper_sim_figures.py --out <folder>

Writes fig_sim_*.png and sim_numbers.json. Every number comes from the formulas in CLAUDE.md section 5 or from the
package's own functions (mejoint.frontend.project, mejoint.regions) applied to synthetic data with known truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from whitepaper_figures import DIVERGING, GRID, INK, INK2, SERIES, SURF, save  # noqa: E402  (shared style)

TE = np.array([16.20, 31.22, 46.24, 61.26])
RATIOS = np.array([1.0, 0.721, 0.631, 0.560])
TR = 0.75
BAND = (0.01, 0.10)


def qr_channels(m, r_assumed, te):
    """Front-end QR (CLAUDE.md 5.3) for one voxel: Q (E x E) and R, with the sign convention diag(R[:2,:2]) > 0."""
    E = te.size
    D = np.column_stack([-te, np.ones(E), te ** 2, (-1.0) ** np.arange(E)])[:, :E]
    Xw = (m / r_assumed)[:, None] * D
    Q, R = np.linalg.qr(Xw, mode="complete")
    sgn = np.sign(np.diag(R[:2, :2]))
    Q[:, :2] *= sgn
    R[:2, :] *= sgn[:, None]
    return Q, R


# ---------------------------------------------------------------- 1. signal model
def fig_echo_dependence(out, made, nums):
    te = np.linspace(0, 70, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    for lab, a, b, col in (("ΔR2* = +1 s⁻¹ (BOLD-like)", 0.0, 1e-3, SERIES[0]), ("ΔS0/S0 = +1 % (S0-like)", 0.01, 0.0, SERIES[1]),
                           ("both together", 0.01, 1e-3, SERIES[2])):
        ax.plot(te, (a - te * b) * 100, color=col, label=lab)
        ax.plot(TE, (a - TE * b) * 100, "o", color=col, ms=6, mec=SURF, mew=1.2)
    ax.axhline(0, color=INK2, lw=1)
    for t in TE:
        ax.axvline(t, color=GRID, lw=1, zorder=0)
    ax.set_xlabel("echo time (ms); dots: the four acquired echoes")
    ax.set_ylabel("log-ratio signal change (%)")
    ax.set_title("A. The two physical sources differ only in their echo dependence", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    err_lin = (np.exp(-TE * 1e-3) - 1 - (-TE * 1e-3)) / (TE * 1e-3) * 100       # linear form vs exact, % of the change
    ax.bar(np.arange(4) - 0.18, np.abs(err_lin), width=0.34, color=SERIES[0], label="ΔR2* = 1 s⁻¹")
    ax.bar(np.arange(4) + 0.18, [abs(0.2 - np.log(1.2)) / np.log(1.2) * 100] * 4, width=0.34, color=SERIES[1], label="20 % S0 spike")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"TE {t:.1f}" for t in TE])
    ax.set_ylabel("error of the linear ΔS/S form (%)")
    ax.set_title("B. The log-ratio form is exactly linear in ΔS0 and ΔR2*", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig_sim_echo_dependence", made)
    nums["linear_form_error_te4_percent"] = round(float(abs(err_lin[-1])), 2)
    nums["linear_form_error_s0_spike_percent"] = round(abs(0.2 - np.log(1.2)) / np.log(1.2) * 100, 1)


# ---------------------------------------------------------------- 2. cost of voxelwise separation
def separation_cost(t2s, te, sig):
    """SD inflation of the voxelwise ΔR2* estimate when ΔS0 is also estimated (vs ΔS0 = 0, all 4 echoes) and the
    correlation of the two estimates; WLS weights mu^2 / sigma_e^2."""
    out = []
    te4 = TE
    sig4 = sig if sig.size == 4 else np.ones(4)
    for T2 in t2s:
        mu4 = np.exp(-te4 / T2)
        w4 = (mu4 / sig4) ** 2
        base = 1.0 / np.sum(w4 * te4 ** 2)                       # var(b) with a = 0, 4 echoes
        keep = np.isin(te4, te)
        w = w4[keep]
        X = np.column_stack([np.ones(keep.sum()), -te4[keep]])
        C = np.linalg.inv(X.T @ (w[:, None] * X))
        out.append((np.sqrt(C[1, 1] / base), -C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])))
    return np.array(out)


def fig_separation_cost(out, made, nums):
    t2s = np.linspace(20, 70, 51)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    cases = [("4 echoes, equal noise", TE, np.ones(4), SERIES[0]), ("4 echoes, measured noise ratios", TE, RATIOS, SERIES[2]),
             ("3 echoes (no 61 ms), equal noise", TE[:3], np.ones(4), SERIES[1])]
    for lab, te, sig, col in cases:
        c = separation_cost(t2s, te, sig)
        axes[0].plot(t2s, c[:, 0], color=col, label=lab)
        axes[1].plot(t2s, np.abs(c[:, 1]), color=col, label=lab)
        key = lab.split(",")[0].replace(" ", "_") + ("_ratios" if "ratios" in lab else "")
        nums[f"inflation_{key}"] = [round(float(c[:, 0].min()), 2), round(float(c[:, 0].max()), 2)]
        nums[f"abs_corr_{key}"] = [round(float(np.abs(c[:, 1]).min()), 3), round(float(np.abs(c[:, 1]).max()), 3)]
    axes[0].set_xlabel("T2* (ms)")
    axes[0].set_ylabel("ΔR2* noise SD, relative")
    axes[0].set_title("A. Estimating ΔS0 too inflates ΔR2* noise\n(vs ΔS0 assumed 0, 4 echoes)", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("T2* (ms)")
    axes[1].set_ylabel("|correlation| of the two estimates")
    axes[1].set_title("B. Voxelwise ΔS0 and ΔR2* estimates are\nstrongly correlated", fontsize=9)
    # C. what the separation costs in voxels: reliability of a region's BOLD-like time course (thermal noise only)
    T2, sd_sig = 45.0, 0.3                                         # resting dR2* SD in band (1/s), illustrative
    bf = 2 * (BAND[1] - BAND[0]) * TR                              # share of white noise inside the band
    n = np.logspace(0, 3, 60)
    X = np.column_stack([np.ones(4), -TE])
    mu = np.exp(-TE / T2) / np.exp(-TE[0] / T2)
    rel_at = {}
    for j, snr1 in enumerate((20.0, 50.0)):
        w = (mu * snr1 / RATIOS) ** 2
        v_sep = np.linalg.inv(X.T @ (w[:, None] * X))[1, 1] * 1e6 * bf          # (1/s)^2, one voxel, in band
        v_fix = 1.0 / np.sum(w * TE ** 2) * 1e6 * bf
        for v, ls, lab in ((v_sep, "-", "ΔS0 estimated (route B)"), (v_fix, "--", "ΔS0 assumed 0")):
            rel = sd_sig ** 2 / (sd_sig ** 2 + v / n)
            axes[2].semilogx(n, rel, ls, color=SERIES[j], label=f"SNR {snr1:.0f}, {lab}")
        rel_at[int(snr1)] = {"voxels_for_rel_0p9_separated": int(np.ceil(9 * v_sep / sd_sig ** 2)),
                             "voxels_for_rel_0p9_fixed": int(np.ceil(9 * v_fix / sd_sig ** 2))}
    axes[2].axhline(0.5, color=INK2, lw=1, ls=":")
    axes[2].set_xlabel("voxels averaged in the region")
    axes[2].set_ylabel("reliability of the BOLD-like time course")
    axes[2].set_title(f"C. Averaging over a region pays the cost\n(thermal noise, T2* {T2:.0f} ms, signal SD {sd_sig} s⁻¹)", fontsize=9)
    axes[2].legend(fontsize=6, loc="lower right")
    nums["region_reliability_example"] = rel_at
    fig.tight_layout()
    save(fig, out, "fig_sim_separation_cost", made)


# ---------------------------------------------------------------- 3. front-end channels
def fig_channels(out, made, nums):
    T2 = 45.0
    m = np.exp(-TE / T2)
    Q, R = qr_channels(m, RATIOS, TE)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    names = ["c0 (OC-like)", "c1 (ΔS0 only)", "r0 (curvature)", "r1 (alternating)"]
    for j in range(4):
        axes[0].plot(TE, Q[:, j], "o-", color=SERIES[j], label=names[j], ms=5)
    axes[0].axhline(0, color=INK2, lw=1)
    axes[0].set_xticks(TE)
    axes[0].set_xlabel("echo time (ms)")
    axes[0].set_ylabel("weight on the whitened echo")
    axes[0].set_title(f"A. Channel weights across echoes (T2* {T2:.0f} ms)", fontsize=9)
    axes[0].legend(fontsize=7)
    # B. response of each channel to unit sources
    W = m / RATIOS
    resp = {}
    for lab, a, b in (("ΔR2* = 1 s⁻¹", 0.0, 1e-3), ("ΔS0/S0 = 1 %", 0.01, 0.0)):
        y = a - TE * b
        resp[lab] = Q.T @ (W * y)
    x = np.arange(4)
    for j, (lab, v) in enumerate(resp.items()):
        axes[1].bar(x + (j - 0.5) * 0.36, v / np.max(np.abs(list(resp.values()))), width=0.34, color=SERIES[j], label=lab)
    axes[1].axhline(0, color=INK2, lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["c0", "c1", "r0", "r1"])
    axes[1].set_ylabel("channel response (relative)")
    axes[1].set_title("B. A pure ΔR2* change never reaches c1, r0 or r1", fontsize=9)
    axes[1].legend(fontsize=7)
    # C. back-substitution recovers both maps exactly
    rng = np.random.default_rng(4)
    s_true = rng.normal(0, 0.01, 300)
    q_true = rng.normal(0, 1e-3, 300)
    rec = []
    for s, q in zip(s_true, q_true):
        cc = Q[:, :2].T @ (W * (s - TE * q))
        s_hat = cc[1] / R[1, 1]
        q_hat = (cc[0] - R[0, 1] * s_hat) / R[0, 0]
        rec.append((s_hat, q_hat))
    rec = np.array(rec)
    axes[2].scatter(q_true * 1000, rec[:, 1] * 1000, s=8, color=SERIES[0], alpha=0.6, label="ΔR2* (1/s)")
    axes[2].scatter(s_true * 100, rec[:, 0] * 100, s=8, color=SERIES[1], alpha=0.6, label="ΔS0/S0 (%)")
    lim = 3.5
    axes[2].plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls="--")
    axes[2].set_xlabel("true (mixed sources, either sign)")
    axes[2].set_ylabel("recovered by back-substitution")
    axes[2].set_title("C. Both physical maps recovered from (c0, c1)", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig_sim_channels", made)
    nums["c1_response_to_pure_r2s"] = float(abs(resp["ΔR2* = 1 s⁻¹"][1]))
    nums["backsub_max_abs_error"] = float(np.max(np.abs(rec - np.column_stack([s_true, q_true]))))


def fig_whitening(out, made, nums):
    """Noise covariance of (c0, c1, r0, r1) when the true per-echo noise ratios are used or ignored (exact)."""
    T2 = 45.0
    m = np.exp(-TE / T2)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3))
    for ax, (lab, assumed) in zip(axes, (("per-echo weights (used)", RATIOS), ("equal weights (sensitivity analysis)", np.ones(4)))):
        Q, _ = qr_channels(m, assumed, TE)
        D = np.diag((RATIOS / assumed) ** 2)                       # noise variance of each whitened echo
        C = Q.T @ D @ Q
        im = ax.imshow(C - np.eye(4), cmap=DIVERGING, vmin=-0.6, vmax=0.6)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(C):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(["c0", "c1", "r0", "r1"]); ax.set_yticklabels(["c0", "c1", "r0", "r1"])
        ax.set_title(lab, fontsize=9)
        nums[f"whitening_max_dev_{'per_echo' if assumed is RATIOS else 'equal'}"] = round(float(np.max(np.abs(C - np.eye(4)))), 3)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    fig.suptitle("Channel noise covariance (numbers) and its deviation from the identity (colour);\ntrue echo noise ratios [1, 0.72, 0.63, 0.56]",
                 x=0.02, ha="left", fontsize=9, fontweight="bold")
    save(fig, out, "fig_sim_whitening", made)


def fig_sigma_estimator(out, made, nums):
    """Thermal noise from the leftover channel: plain vs frame-differenced estimator under model violations."""
    rng = np.random.default_rng(7)
    T, nvox = 800, 4000
    t = np.arange(T) * TR
    amps = np.linspace(0, 1.0, 11)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for ax, (lab, gen) in zip(axes, (("slow drift (< 0.02 Hz)", lambda a: a * np.cumsum(rng.standard_normal((nvox, T)), 1) / np.sqrt(T) * 2.0),
                                     ("respiration-paced (0.3 Hz)", lambda a: a * np.sin(2 * np.pi * 0.3 * t[None, :] + rng.uniform(0, 6.3, (nvox, 1)))))):
        plain, diff = [], []
        for a in amps:
            r = rng.standard_normal((nvox, T)) + gen(a)
            r = r - r.mean(1, keepdims=True)
            plain.append(np.sqrt(np.median(r ** 2, axis=1) / 0.4549).mean())
            d = np.diff(r, axis=1)
            diff.append(np.sqrt(np.median(d ** 2, axis=1) / (2 * 0.4549)).mean())
        ax.plot(amps, plain, "o-", color=SERIES[1], ms=4, label="plain residual")
        ax.plot(amps, diff, "o-", color=SERIES[0], ms=4, label="frame-differenced (used)")
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.set_xlabel("violation amplitude (in units of the thermal SD)")
        ax.set_title(lab, fontsize=9)
        nums[f"sigma_bias_at_1_{lab.split()[0]}"] = {"plain": round(float(plain[-1]), 3), "differenced": round(float(diff[-1]), 3)}
    axes[0].set_ylabel("estimated σ / true σ")
    axes[0].legend(fontsize=7)
    fig.suptitle("Thermal σ from one leftover channel (df = 1), 800 frames", x=0.02, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig_sim_sigma_estimator", made)


# ---------------------------------------------------------------- 4. route B on synthetic regions
def bandlimited(rng, n, T):
    from mejoint.regions import band_pass
    x = band_pass(rng.standard_normal((n, T)), TR, BAND)
    return x / x.std(1, keepdims=True)


def fig_route_b_recovery(out, made, nums):
    """Region averaging, split, leftover noise and shares on synthetic magnitude data with known truth."""
    from mejoint.regions import leftover_noise, region_signals, shares, split
    rng = np.random.default_rng(11)
    T, P = 600, 60
    sizes = rng.choice([10, 20, 40, 80, 160], P)
    frac = rng.uniform(0, 1, P)                                    # mix of R2* vs S0 amplitude
    z = bandlimited(rng, 2 * P, T)
    b = (0.6e-3 * np.sqrt(frac))[:, None] * z[:P]                 # ΔR2* (1/ms), up to ~0.6 1/s SD
    a = (0.006 * np.sqrt(1 - frac))[:, None] * z[P:]              # ΔlnS0, up to ~0.6 % SD
    labels = np.repeat(np.arange(1, P + 1), sizes)
    V = labels.size
    T2 = rng.uniform(35, 65, V)
    S0 = 1000.0
    mu = S0 * np.exp(-TE[None, :] / T2[:, None])
    sigma = np.full(V, S0 / 50.0)                                  # echo-1 SNR ~ 50 / exp(-16/T2*)
    clean = mu[:, :, None] * np.exp(a[labels - 1][:, None, :] - TE[None, :, None] * b[labels - 1][:, None, :])
    noise = sigma[:, None, None] * RATIOS[None, :, None] * rng.standard_normal((V, 4, T))
    use = np.ones((V, 4), bool)
    res = {}
    for name, S in (("truth", clean), ("noisy", clean + noise)):
        rs = region_signals(S, labels, mu, sigma, RATIOS, use, P)
        sp = split(rs["ybar"], rs["wsum"], TE)
        nv = leftover_noise(sp["r"]) if name == "noisy" else np.zeros(P)
        res[name] = (sp, shares(sp, nv, TR, BAND), nv)
    true_share = res["truth"][1]["bold_share"]
    est_share = np.clip(res["noisy"][1]["bold_share"], 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for j, sz in enumerate(sorted(set(sizes))):
        m = sizes == sz
        axes[0].scatter(true_share[m], est_share[m], s=24, color=SERIES[j], edgecolors=SURF, linewidths=0.8, label=f"{sz} voxels", zorder=3)
    axes[0].plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    axes[0].fill_between([0, 1], [-0.1, 0.9], [0.1, 1.1], color=GRID, alpha=0.6, lw=0)
    axes[0].set_xlim(-0.02, 1.02); axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_xlabel("true BOLD-like share")
    axes[0].set_ylabel("estimated (noise removed)")
    axes[0].set_title("A. Shares recovered at resting amplitudes", fontsize=9)
    axes[0].legend(fontsize=7)
    sp_n = res["noisy"][0]
    rb = [np.corrcoef(sp_n["b"][p], b[p] * 1000)[0, 1] for p in range(P) if frac[p] > 0.2]
    ra = [np.corrcoef(sp_n["a"][p], a[p])[0, 1] for p in range(P) if frac[p] < 0.8]
    axes[1].hist(rb, bins=np.linspace(0, 1, 21), color=SERIES[0], alpha=0.7, label="ΔR2* time course")
    axes[1].hist(ra, bins=np.linspace(0, 1, 21), color=SERIES[1], alpha=0.7, label="ΔS0 time course")
    axes[1].set_xlabel("correlation with the true time course (unfiltered)")
    axes[1].set_ylabel("regions")
    axes[1].set_title("B. Region time courses of both parts", fontsize=9)
    axes[1].legend(fontsize=7)
    # C. leftover noise against the noise the region channels actually carry
    nv = res["noisy"][2]
    true_nv = np.array([np.var(np.diff(np.asarray(sp_n["c"][p] - res["truth"][0]["c"][p]), axis=1)) / 2 for p in range(P)])
    axes[2].scatter(true_nv, nv, s=24, color=SERIES[2], edgecolors=SURF, linewidths=0.8, zorder=3)
    lo, hi = min(true_nv.min(), nv.min()) * 0.9, max(true_nv.max(), nv.max()) * 1.1
    axes[2].plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("noise variance in the region channels (truth)")
    axes[2].set_ylabel("estimated from the leftover channels")
    axes[2].set_title("C. Region noise from the leftover channels", fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig_sim_route_b_recovery", made)
    err = np.abs(est_share - true_share)
    nums["route_b_sim"] = {"n_regions": P, "share_abs_error_median": round(float(np.median(err)), 3),
                           "share_abs_error_max": round(float(np.max(err)), 3), "fraction_within_0p1": round(float(np.mean(err <= 0.1)), 3),
                           "share_error_by_size": {int(s): round(float(np.median(err[sizes == s])), 3) for s in sorted(set(sizes))},
                           "r_b_median": round(float(np.median(rb)), 3), "r_a_median": round(float(np.median(ra)), 3),
                           "noise_ratio_median": round(float(np.median(nv / true_nv)), 3)}


# ---------------------------------------------------------------- 5. integration measures
def network_corr(n_net, per, w_net, g):
    """Population correlation of regions with one global, network and private parts."""
    n = n_net * per
    L = np.zeros((n, n_net + 1))
    L[:, 0] = g * (1.0 + 0.35 * np.random.default_rng(5).standard_normal(n))   # regions differ in global loading
    for k in range(n_net):
        L[k * per:(k + 1) * per, k + 1] = w_net
    C = L @ L.T
    d = np.sqrt(np.diag(C) + 1.0)
    C = (C + np.eye(n)) / np.outer(d, d)
    return C, L


def fig_pr_global(out, made, nums):
    from mejoint.regions import participation_ratio, participation_ratio_residual
    gs = np.linspace(0, 1.4, 29)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for j, w in enumerate((0.6, 0.9)):
        pr, pr1, share = [], [], []
        for g in gs:
            C, _ = network_corr(7, 24, w, g)
            ev = np.linalg.eigvalsh(C)
            pr.append(participation_ratio(C)); pr1.append(participation_ratio_residual(C)); share.append(ev[-1] / ev.sum())
        axes[0].plot(share, pr, color=SERIES[j], label=f"network strength {w}")
        axes[1].plot(share, pr1, color=SERIES[j], label=f"network strength {w}")
        nums[f"pr_global_w{w}"] = {"pr_range": [round(min(pr), 2), round(max(pr), 2)], "pr1_range": [round(min(pr1), 2), round(max(pr1), 2)]}
    axes[0].set_xlabel("share of variance in the first (brain-wide) pattern")
    axes[0].set_ylabel("ordinary participation ratio")
    axes[0].set_title("A. Ordinary PR mostly reads the global pattern", fontsize=9)
    axes[1].set_xlabel("share of variance in the first (brain-wide) pattern")
    axes[1].set_ylabel("PR without the first pattern")
    axes[1].set_title("B. Leaving out the first pattern removes that dependence", fontsize=9)
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    # C. structure beyond the global pattern: more independent networks raise the primary
    for j, n_net in enumerate((3, 5, 7, 10)):
        C, _ = network_corr(n_net, 168 // n_net, 0.8, 0.6)
        axes[2].bar(j, participation_ratio_residual(C), color=SERIES[0], width=0.6)
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(["3", "5", "7", "10"])
    axes[2].set_xlabel("independent networks (same 168 regions, same global pattern)")
    axes[2].set_ylabel("PR without the first pattern")
    axes[2].set_title("C. The primary counts patterns beyond the global one", fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig_sim_pr_global", made)


def fig_frames(out, made, nums):
    """Finite-sample bias: the measure falls as the number of independent frames falls."""
    from mejoint.regions import participation_ratio_residual
    rng = np.random.default_rng(3)
    C, L = network_corr(7, 24, 0.8, 0.6)
    n = C.shape[0]
    chol = np.linalg.cholesky(C)
    neff = np.array([40, 60, 80, 110, 160, 240, 400, 800])
    true = participation_ratio_residual(C)
    vals = []
    for k in neff:
        v = [participation_ratio_residual(np.corrcoef(chol @ rng.standard_normal((n, k)))) for _ in range(30)]
        vals.append((np.mean(v), np.std(v)))
    vals = np.array(vals)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.axhline(true, color=INK2, lw=1, ls="--", label="population value")
    ax.fill_between(neff, vals[:, 0] - vals[:, 1], vals[:, 0] + vals[:, 1], color=SERIES[0], alpha=0.2, lw=0)
    ax.plot(neff, vals[:, 0], "o-", color=SERIES[0], ms=4, label="estimate (mean ± SD of 30)")
    ax.axvline(110, color=SERIES[1], lw=1.2, ls=":")
    ax.text(118, true - 1.0, "≈ independent values per region\nin 797 frames, 0.01-0.1 Hz", fontsize=7, color=INK2, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("independent frames")
    ax.set_ylabel("PR without the first pattern")
    ax.set_title(f"Finite-sample bias ({n} regions)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig_sim_frames", made)
    nums["frames_bias"] = {"population": round(float(true), 2), "at_110": round(float(vals[list(neff).index(110), 0]), 2),
                           "at_800": round(float(vals[-1, 0]), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made, nums = [], {}
    for fn in (fig_echo_dependence, fig_separation_cost, fig_channels, fig_whitening, fig_sigma_estimator,
               fig_route_b_recovery, fig_pr_global, fig_frames):
        fn(out, made, nums)
        print("made", fn.__name__)
    (out / "sim_numbers.json").write_text(json.dumps(nums, indent=1, default=float))
    print(json.dumps(nums, indent=1, default=float))


if __name__ == "__main__":
    main()
