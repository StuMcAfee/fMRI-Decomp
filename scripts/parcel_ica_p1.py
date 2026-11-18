"""Parcel ICA gate P1 (docs/PARCEL_ICA_PLAN.md): simulation only, no participant data.

Grid: two parcel scales (1876 icosahedral-like parcels at median in-band reliability 0.3; 200 Schaefer-like parcels at
0.78) x physiological maps distinct or overlapping (shared vascular weighting) x physiological strength (0, 0.5, 1, 2,
4, 8 times the reference amplitudes in parcel_sim) x seeds. For each synthetic participant: dimensionality (MP edge,
leftover-channel null), robust spatial ICA at k = MP count, component measurements, a candidate removal rule, and
its effect: share of the true physiological energy removed and of the true network + global energy kept.

Usage: PYTHONPATH=src python scripts/parcel_ica_p1.py <out_dir> [n_seeds]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from mejoint import parcel_ica as pi
from mejoint import parcel_sim as ps

# candidate removal rule (to be fixed at P1 and then held for P2-P4): high-frequency signal share (noise-debiased,
# above 0.12 Hz) above HIGH, or S0 coupling (eta) above ETA
HIGH, ETA = 0.5, 0.5
SCALES = ((1876, 0.30), (200, 0.78))
MULTS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
REF = {"respiratory": 0.12, "resp_s0": 0.0006, "cardiac": 0.04, "card_s0": 0.002, "slow_physio": 0.20}


def energy_left(X, tcs):
    """Energy of X (parcels x frames) after projecting out the time courses (rows of tcs), over the energy before."""
    e0 = float((X ** 2).sum())
    return float((pi.remove_components(X, tcs) ** 2).sum()) / e0 if e0 > 0 else float("nan")


def one(P, rel, overlap, mult, seed):
    d = ps.simulate(P, reliability=rel, seed=seed, overlap=overlap, amp={k: v * mult for k, v in REF.items()})
    pr = pi.prepare(d["c0"], d["c1"], d["R"], d["noise_var"], d["r"])
    zb, T = pr["zb"], pr["zb"].shape[1]
    dim = pi.dimensionality(zb, T - 1, pr["null_channels"])
    tr = d["truth"]
    names = tr["names"]
    sdb = np.sqrt(1 + pr["rho"] ** 2) / d["R"][:, 0, 0] * 1000.0          # b noise SD per parcel (whitening scale)
    srcw = {n: np.outer(tr["maps_b"][:, i] / sdb, tr["tc"][i]) for i, n in enumerate(names)}
    rec = {"P": P, "reliability": rel, "overlap": overlap, "mult": mult, "seed": seed, "k_mp": dim["k_mp"],
           "k_null": dim.get("k_null"), "reliability_reached": d["reliability_median"]}
    k = dim["k_mp"]
    if k < 2:
        rec.update({"k": k, "components": [], "recovery": {}, "flagged": 0})
        return rec
    ica = pi.spatial_ica(zb, k, range(30))
    sp = pi.spectral_measures(ica["tc"], d["tr"], resp_hz=d["resp_hz"], noise_var=ica["tc_noise_var"])
    folds = np.random.default_rng(seed).permutation(np.arange(zb.shape[0]) % 2)
    s0 = pi.s0_coupling(ica["maps"], zb, pr["cw"], pr["rho"], folds)
    C = np.corrcoef(np.vstack([tr["tc"], ica["tc"]]))[: len(names), len(names):]
    comps = []
    for j in range(k):
        i = int(np.argmax(np.abs(C[:, j])))
        kind = names[i] if abs(C[i, j]) >= 0.5 else "unmatched"
        flag = bool(sp["high_share"][j] > HIGH or (np.isfinite(s0["eta"][j]) and s0["eta"][j] > ETA))
        comps.append({"best_source": kind, "r": round(float(abs(C[i, j])), 3), "high_share": round(float(sp["high_share"][j]), 3),
                      "band_share": round(float(sp["band_share"][j]), 3), "resp_share": round(float(sp["resp_share"][j]), 3),
                      "peak_hz": round(float(sp["peak_hz"][j]), 3), "peak_ratio": round(float(sp["peak_ratio"][j]), 2),
                      "eta": None if not np.isfinite(s0["eta"][j]) else round(float(s0["eta"][j]), 3),
                      "iq": round(float(ica["iq"][j]), 3), "flag": flag})
    fl = np.array([c["flag"] for c in comps])
    rm = ica["tc"][fl]
    neural = sum(v for n, v in srcw.items() if n.startswith("network") or n == "global")
    fast = sum(srcw[n] for n in ("respiratory", "cardiac") if n in srcw)
    rec.update({"k": k, "components": comps, "flagged": int(fl.sum()),
                "recovery": {n: round(float(np.abs(C[i]).max()), 3) for i, n in enumerate(names)},
                "neural_kept": round(energy_left(neural, rm), 4),
                "fast_physio_removed": round(1 - energy_left(fast, rm), 4) if mult > 0 else None,
                "slow_physio_removed": round(1 - energy_left(srcw["slow_physio"], rm), 4) if mult > 0 else None,
                "network_match": {n: round(abs(v["r"]), 3) for n, v in pi.match_networks(ica["maps"][:, ~fl] if (~fl).sum() >= 7 else ica["maps"], d["networks"][pr["ok"]]).items()}})
    return rec


def summarise(rows):
    out = []
    for P, rel in SCALES:
        for ov in (False, True):
            for m in MULTS:
                g = [r for r in rows if r["P"] == P and r["overlap"] == ov and r["mult"] == m]
                if not g:
                    continue
                mean = lambda key: float(np.nanmean([r[key] if r.get(key) is not None else np.nan for r in g]))
                rec = lambda n: float(np.nanmean([r["recovery"].get(n, np.nan) for r in g]))
                out.append({"P": P, "overlap": ov, "mult": m, "n": len(g), "k_mp": mean("k_mp"), "k_null": mean("k_null"),
                            "flagged": mean("flagged"), "neural_kept": mean("neural_kept"),
                            "fast_removed": mean("fast_physio_removed") if m > 0 else None,
                            "slow_removed": mean("slow_physio_removed") if m > 0 else None,
                            "rec_network": float(np.nanmean([np.mean([v for n, v in r["recovery"].items() if n.startswith("network")]) for r in g if r["recovery"]])),
                            "rec_respiratory": rec("respiratory") if m > 0 else None, "rec_cardiac": rec("cardiac") if m > 0 else None,
                            "rec_slow": rec("slow_physio") if m > 0 else None,
                            "net_map_match": float(np.nanmean([np.mean(list(r["network_match"].values())) for r in g if r.get("network_match")]))})
    return out


def render(summ):
    f = lambda v, n=2: "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"
    L = ["# Parcel ICA, gate P1 (simulation)", "",
         f"Candidate removal rule: noise-debiased signal share above 0.12 Hz > {HIGH}, or S0 coupling eta > {ETA}.",
         "Physiological strength: multiple of the reference amplitudes (respiratory 0.12 1/s, cardiac 0.04 1/s with dS0 0.2 %, "
         "slow physiological 0.20 1/s; networks 0.25 1/s, global 0.30 1/s). Recovery: |r| of the best component time course.", "",
         "| parcels | physio maps | strength | k (MP) | k (null) | flagged | network rec. | network map match | resp. rec. | cardiac rec. | slow rec. | fast physio removed | slow physio removed | neural kept |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summ:
        L.append(f"| {s['P']} | {'overlapping' if s['overlap'] else 'distinct'} | {s['mult']:g} | {f(s['k_mp'],1)} | {f(s['k_null'],1)} | {f(s['flagged'],1)} | "
                 f"{f(s['rec_network'])} | {f(s['net_map_match'])} | {f(s['rec_respiratory'])} | {f(s['rec_cardiac'])} | {f(s['rec_slow'])} | "
                 f"{f(s['fast_removed'])} | {f(s['slow_removed'])} | {f(s['neural_kept'])} |")
    return "\n".join(L) + "\n"


def figure(rows, summ, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, aqua, yellow, magenta = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
    fig, ax = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for col, (P, rel) in enumerate(SCALES):
        for ov, ls in ((False, "-"), (True, "--")):
            g = [s for s in summ if s["P"] == P and s["overlap"] == ov and s["mult"] > 0]
            x = [s["mult"] for s in g]
            a = ax[0, col]
            for key, c, lab in (("rec_network", blue, "networks"), ("rec_respiratory", orange, "respiratory"),
                                ("rec_cardiac", magenta, "cardiac"), ("rec_slow", aqua, "slow physiological")):
                a.plot(x, [s[key] for s in g], ls, color=c, lw=2, marker="o", ms=5, label=None if ov else lab)
            b = ax[1, col]
            for key, c, lab in (("neural_kept", blue, "network + global energy kept"), ("fast_removed", orange, "respiratory + cardiac energy removed"),
                                ("slow_removed", aqua, "slow physiological energy removed")):
                b.plot(x, [s[key] for s in g], ls, color=c, lw=2, marker="o", ms=5, label=None if ov else lab)
        for a, t in ((ax[0, col], "source recovery (|r|, time course)"), (ax[1, col], "effect of the removal rule")):
            a.set_xscale("log", base=2)
            a.set_ylim(0, 1.02)
            a.set_xlabel("physiological strength (x reference)")
            a.set_title(f"{P} parcels, reliability {rel}\n{t}", fontsize=10)
            a.grid(alpha=0.25)
    for a in (ax[0, 0], ax[1, 0]):
        a.plot([], [], "-", color="#555", lw=1.5, label="distinct maps")
        a.plot([], [], "--", color="#555", lw=1.5, label="overlapping maps")
    ax[0, 0].legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.9, ncol=2)
    ax[1, 0].legend(fontsize=7, loc="center left", frameon=True, framealpha=0.9)
    a = ax[0, 2]
    cols = {"network": blue, "global": "#6250d6", "respiratory": orange, "cardiac": magenta, "slow_physio": aqua, "unmatched": "#9a9a92"}
    for r in rows:
        for c in r["components"]:
            key = "network" if c["best_source"].startswith("network") else c["best_source"]
            a.scatter(c["high_share"], c["eta"] if c["eta"] is not None else np.nan, s=10, color=cols[key], alpha=0.5, lw=0)
    for key, c in cols.items():
        a.scatter([], [], color=c, s=20, label=key.replace("_", " "))
    a.axvline(HIGH, color="#555", lw=1, ls=":")
    a.axhline(ETA, color="#555", lw=1, ls=":")
    a.set_xlabel("signal share above 0.12 Hz (noise-debiased)")
    a.set_ylabel("S0 coupling (eta)")
    a.set_title("all components\ncoloured by best-matching source", fontsize=10)
    a.legend(fontsize=7, frameon=True, framealpha=0.9, loc="upper left")
    a.set_xlim(-0.02, 1.02)
    a.set_ylim(-0.02, 1.02)
    a = ax[1, 2]
    for (P, rel), c in zip(SCALES, (blue, orange)):
        for ov, ls in ((False, "-"), (True, "--")):
            g = [s for s in summ if s["P"] == P and s["overlap"] == ov]
            a.plot([max(s["mult"], 0.25) for s in g], [s["k_mp"] for s in g], ls, color=c, lw=2, marker="o", ms=5,
                   label=f"{P} parcels, {'overlapping' if ov else 'distinct'} maps")
    a.axhline(11, color="#555", lw=1, ls=":")
    a.text(0.27, 10.85, "simulated sources (11)", fontsize=7, color="#555", va="top")
    a.set_xscale("log", base=2)
    a.set_xlabel("physiological strength (x reference; 0 plotted at 0.25)")
    a.set_ylabel("dimensions")
    a.set_title("dimensionality\n(eigenvalues above the MP edge)", fontsize=10)
    a.legend(fontsize=7, frameon=True, framealpha=0.9, loc="lower right")
    a.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    out = Path(sys.argv[1])
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for P, rel in SCALES:
        for ov in (False, True):
            for m in MULTS:
                for s in range(n_seeds):
                    rows.append(one(P, rel, ov, m, 1000 + s))
                print(P, ov, m, "done", flush=True)
    # pure noise: no sources at all (sources excluded), the MP count should be 0
    noise = []
    for P, rel in SCALES:
        for s in range(n_seeds):
            d = ps.simulate(P, reliability=rel, seed=2000 + s)
            g = np.random.default_rng(3000 + s)                       # same R and noise level, sources removed
            d["c0"], d["c1"] = g.standard_normal(d["c0"].shape), g.standard_normal(d["c1"].shape)
            pr = pi.prepare(d["c0"], d["c1"], d["R"], d["noise_var"], d["r"])
            dim = pi.dimensionality(pr["zb"], pr["zb"].shape[1] - 1, pr["null_channels"])
            noise.append({"P": P, "seed": 2000 + s, "k_mp": dim["k_mp"], "k_null": dim["k_null"]})
    summ = summarise(rows)
    (out / "p1_rows.json").write_text(json.dumps(rows, indent=1))
    (out / "p1_summary.json").write_text(json.dumps({"rule": {"high_share": HIGH, "eta": ETA}, "summary": summ, "pure_noise": noise}, indent=1))
    md = render(summ) + "\nPure noise (no sources): " + ", ".join(f"{r['P']} parcels k_MP {r['k_mp']} / k_null {r['k_null']}" for r in noise) + "\n"
    (out / "p1_summary.md").write_text(md)
    figure(rows, summ, out / "p1_summary.png")
    print(md)


if __name__ == "__main__":
    main()
