"""Cross-participant summary of the joint-ICA sweep (`mejoint sweep-summary`; user request 2026-11-06): for
every run under outputs/phase5/<run>/phase4_sweep_mc/, the label counts at every k, and at one common k the
components' eta (all, tissue class, CSF class), the partial-volume verdicts, stability, explained variance and
the largest correlation of a component time course with framewise displacement. A description across
participants, not a group test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LABELS = ("R2*-dominant", "S0-dominant", "mixed", "not above null", "undetermined")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _med(x):
    x = np.asarray([_num(v) for v in x], float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def _strength(c) -> float:
    """Component energy relative to the family-wise energy null (1 = at the null)."""
    e, t = _num(c.get("energy_single")), _num(c.get("energy_null_threshold"))
    return e / t if np.isfinite(e) and np.isfinite(t) and t > 0 else np.nan


def _spearman(a, b) -> float:
    a, b = np.array([_num(x) for x in a], float), np.array([_num(x) for x in b], float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    from scipy.stats import spearmanr
    return float(spearmanr(a[ok], b[ok]).statistic)


def collect(root, sweep_name: str = "phase4_sweep_mc") -> tuple[list[dict], list[dict], list[str]]:
    """(per run x k rows, per component rows, runs without a sweep)."""
    root = Path(root)
    runs, comps, missing = [], [], []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        sw = run / sweep_name / "sweep.json"
        if not sw.exists():
            missing.append(run.name)
            continue
        rk = root.parent / "phase3" / run.name / "mc" / "rank.json"
        mp = (json.loads(rk.read_text()).get("mapca") or {}) if rk.exists() else {}
        for kd in sorted((run / sweep_name).glob("k*/components.json"), key=lambda p: int(p.parent.name[1:])):
            s = json.loads(kd.read_text())
            erf = kd.parent / "eta_recheck.json"
            er = json.loads(erf.read_text()) if erf.exists() else {}
            erc = {c["component"]: c for c in er.get("components", [])}
            k = int(s["k"])
            cs = s["components"]
            above = [c for c in cs if c.get("above_null")]
            fd_key = next((key for key in ("corr_fd_filtered_mm", "corr_fd_mm") if any(key in c for c in cs)), None)
            fdr = [abs(_num(c.get(fd_key))) for c in cs] if fd_key else []
            judged = [c for c in above if c.get("pv_verdict") in ("consistent with partial volume", "not partial volume")]
            runs.append({"run": run.name, "k": k, **{lab: s["label_counts"].get(lab, 0) for lab in LABELS},
                         "mapca_aic": mp.get("aic"), "mapca_kic": mp.get("kic"), "mapca_mdl": mp.get("mdl"),
                         "n_components": len(cs), "n_above_null": len(above),
                         "explained_variance": s.get("explained_variance"),
                         "stability_median": s.get("stability_median_of_components"),
                         "eta_median": _med(c.get("eta") for c in above),
                         "eta_tissue_median": _med(c.get("eta_tissue") for c in above),
                         "eta_csf_median": _med(c.get("eta_csf") for c in above),
                         "eta_pv_removed_median": _med(c.get("eta_pv_removed") for c in above),
                         "n_pv_judged": len(judged),
                         "n_pv_consistent": sum(c["pv_verdict"] == "consistent with partial volume" for c in judged),
                         "fd_corr_key": fd_key, "fd_abs_corr_max": float(np.nanmax(fdr)) if fdr else np.nan,
                         "fd_abs_corr_median": float(np.nanmedian(fdr)) if fdr else np.nan,
                         "strength_eta_spearman": _spearman([c.get("eta") for c in above],
                                                            [_strength(c) for c in above]),
                         "eta_strongest5_median": _med(c.get("eta") for c in sorted(above, key=_strength)[-5:]),
                         "eta_weakest5_median": _med(c.get("eta") for c in sorted(above, key=_strength)[:5]),
                         "noise_level_c0": er.get("noise_level_c0"), "noise_level_c1": er.get("noise_level_c1"),
                         "colour_factor_c0": er.get("colour_factor_c0"), "colour_factor_c1": er.get("colour_factor_c1"),
                         "eta_white_empirical_median": _med(erc.get(c.get("component"), {}).get("eta_white_empirical") for c in above),
                         "eta_empirical_median": _med(erc.get(c.get("component"), {}).get("eta_empirical") for c in above),
                         "strength_eta_empirical_spearman": _spearman(
                             [erc.get(c.get("component"), {}).get("eta_empirical") for c in above], [_strength(c) for c in above])})
            for c in cs:
                comps.append({"run": run.name, "k": k, "component": c.get("component"), "label": c.get("label"),
                              "above_null": c.get("above_null"), "eta": c.get("eta"), "eta_tissue": c.get("eta_tissue"),
                              "eta_csf": c.get("eta_csf"), "eta_pv_removed": c.get("eta_pv_removed"),
                              "pv_verdict": c.get("pv_verdict"), "stability": c.get("stability_median_abs_r"),
                              "fd_corr": c.get(fd_key) if fd_key else None, "energy_over_null": _strength(c),
                              "eta_white_empirical": erc.get(c.get("component"), {}).get("eta_white_empirical"),
                              "eta_empirical": erc.get(c.get("component"), {}).get("eta_empirical")})
    return runs, comps, missing


def render(runs: list[dict], missing: list[str], k_main: int) -> str:
    f = lambda v, n=2: "" if not np.isfinite(_num(v)) else f"{_num(v):.{n}f}"
    names = sorted({r["run"] for r in runs})
    L = ["# Joint-ICA sweep across participants", "",
         "Per run: component labels at every k of the fixed sweep (MAPCA AIC / KIC / MDL plus the fixed k), then at "
         f"k = {k_main} (common to all runs) eta, partial volume and motion. Labels are measurements (CLAUDE.md 7.1); "
         "on real data the thermal-only eta bands are too narrow and nearly every component is above the energy null, "
         "so 'mixed' is expected (decisions.md, Phase 4). A description, not a group test.", "",
         f"Runs: {len(names)}. Without a sweep: {len(missing)}" + (f" ({', '.join(missing)})" if missing else "") + ".", "",
         "## Labels at every k (R2*-dominant / S0-dominant / mixed / not above null)", "",
         "| Run | MAPCA AIC / KIC / MDL | Labels by k | Stability (median, by k) |", "|---|---|---|---|"]
    for name in names:
        rr = sorted([r for r in runs if r["run"] == name], key=lambda r: r["k"])
        mp = rr[0]
        L.append(f"| {name} | {mp['mapca_aic']} / {mp['mapca_kic']} / {mp['mapca_mdl']} | "
                 + "; ".join(f"k{r['k']}: {r['R2*-dominant']}/{r['S0-dominant']}/{r['mixed']}/{r['not above null']}" for r in rr)
                 + " | " + ", ".join(f(r["stability_median"], 3) for r in rr) + " |")
    main = sorted([r for r in runs if r["k"] == k_main], key=lambda r: r["run"])
    L += ["", f"## At k = {k_main}", "",
          "| Run | Above null | Variance in k dims | eta median | eta tissue | eta CSF | eta, partial volume removed | "
          "Partial volume: consistent / judged | Stability | FD correlation: max, median |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in main:
        L.append(f"| {r['run']} | {r['n_above_null']} / {r['n_components']} | {f(100 * _num(r['explained_variance']), 1)} % | "
                 f"{f(r['eta_median'], 3)} | {f(r['eta_tissue_median'], 3)} | {f(r['eta_csf_median'], 3)} | "
                 f"{f(r['eta_pv_removed_median'], 3)} | {r['n_pv_consistent']} / {r['n_pv_judged']} | "
                 f"{f(r['stability_median'], 3)} | {f(r['fd_abs_corr_max'])}, {f(r['fd_abs_corr_median'])} |")
    L += ["", f"## Is eta about component strength or noise? (k = {k_main})", "",
          "Check 1: if eta reflects noise left after the thermal debiasing, it rises as components weaken towards the "
          "energy null (strength = energy / null; Spearman rho of eta with strength negative, weakest > strongest). "
          "Check 2 (`mejoint eta-recheck`): eta debiased with each voxel's own noise per channel (level only, then "
          "level and autocorrelation); noise level 1 = thermal.", "",
          "| Run | rho(eta, strength) | eta: 5 strongest / 5 weakest | Noise level c0 / c1 | Colour factor c0 / c1 | "
          "eta reported / white empirical / empirical | rho(empirical eta, strength) |", "|---|---|---|---|---|---|---|"]
    for r in main:
        L.append(f"| {r['run']} | {f(r['strength_eta_spearman'])} | {f(r['eta_strongest5_median'], 3)} / "
                 f"{f(r['eta_weakest5_median'], 3)} | {f(r['noise_level_c0'])} / {f(r['noise_level_c1'])} | "
                 f"{f(r['colour_factor_c0'])} / {f(r['colour_factor_c1'])} | {f(r['eta_median'], 3)} / "
                 f"{f(r['eta_white_empirical_median'], 3)} / {f(r['eta_empirical_median'], 3)} | "
                 f"{f(r['strength_eta_empirical_spearman'])} |")
    if main and not any(np.isfinite(_num(r["eta_empirical_median"])) for r in main):
        L += ["", "Check 2 not run yet: `mejoint eta-recheck`, then this summary again."]
    if main:
        L += ["", f"Across runs at k = {k_main}: eta median {f(_med(r['eta_median'] for r in main), 3)} "
                  f"(tissue {f(_med(r['eta_tissue_median'] for r in main), 3)}, CSF {f(_med(r['eta_csf_median'] for r in main), 3)}); "
                  f"partial volume consistent in {sum(r['n_pv_consistent'] for r in main)} of "
                  f"{sum(r['n_pv_judged'] for r in main)} judged components."]
    return "\n".join(L) + "\n"


def figure(comps: list[dict], path, k_main: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = [c for c in comps if c["k"] == k_main and c.get("above_null")]
    if not cs:
        return
    names = sorted({c["run"] for c in cs})
    cols = {"R2*-dominant": "#2a78d6", "S0-dominant": "#eb6834", "mixed": "#1baf7a"}
    fig, axs = plt.subplots(1, 2, figsize=(14, 0.32 * len(names) + 2.5), sharey=True)
    _strength_figure(cs, Path(path).with_name("sweep_batch_eta_strength.png"), k_main)
    rng = np.random.default_rng(0)
    for i, n in enumerate(names):
        for c in (c for c in cs if c["run"] == n):
            y = i + rng.uniform(-0.25, 0.25)
            axs[0].plot(_num(c["eta"]), y, "o", ms=3, alpha=0.6, color=cols.get(c["label"], "#888888"))
        et = [_num(c["eta_tissue"]) for c in cs if c["run"] == n]
        ec = [_num(c["eta_csf"]) for c in cs if c["run"] == n]
        axs[1].plot(np.nanmedian(et), i, "o", color="#2a78d6", label="tissue class" if i == 0 else None)
        axs[1].plot(np.nanmedian(ec), i, "s", color="#eb6834", label="CSF class" if i == 0 else None)
    axs[0].set_yticks(range(len(names)), [n.split("_")[0] for n in names])
    axs[0].invert_yaxis()
    axs[0].set_xlabel("eta (S0 energy fraction)")
    axs[0].set_title(f"Components above the null at k = {k_main} (colour = label)", loc="left", fontsize=10)
    for lab, col in cols.items():
        axs[0].plot([], [], "o", color=col, label=lab)
    axs[0].legend(loc="lower right", fontsize=7, frameon=False)
    axs[1].set_xlabel("median eta within the class")
    axs[1].set_title("eta by data-only tissue class", loc="left", fontsize=10)
    axs[1].legend(loc="lower right", fontsize=8, frameon=False)
    for ax in axs:
        ax.set_xlim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(root, out_dir, k_main: int = 50, sweep_name: str = "phase4_sweep_mc") -> dict:
    from . import io
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, comps, missing = collect(root, sweep_name)
    if runs:
        io.write_tsv(out_dir / "sweep_batch.tsv", {c: [r.get(c, "") for r in runs] for c in runs[0]})
    if comps:
        io.write_tsv(out_dir / "sweep_batch_components.tsv", {c: [r.get(c, "") for r in comps] for c in comps[0]})
    io.write_text(out_dir / "sweep_batch.md", render(runs, missing, k_main))
    figure(comps, out_dir / "sweep_batch_eta.png", k_main)
    io.write_json(out_dir / "sweep_batch.json", {"k_main": k_main, "runs": runs, "missing": missing})
    return {"n_runs": len({r["run"] for r in runs}), "missing": missing}


def _strength_figure(cs, path, k_main) -> None:
    import matplotlib.pyplot as plt
    has_emp = any(np.isfinite(_num(c.get("eta_empirical"))) for c in cs)
    fig, axs = plt.subplots(1, 2 if has_emp else 1, figsize=(12 if has_emp else 6.5, 4.5), squeeze=False)
    x = np.array([_num(c["energy_over_null"]) for c in cs])
    for ax, key, title in zip(axs[0], ("eta", "eta_empirical"),
                              ("eta as reported (thermal debiasing)", "eta with empirical noise (check 2)")):
        y = np.array([_num(c.get(key)) for c in cs])
        ax.scatter(x, y, s=6, alpha=0.4, color="#2a78d6")
        ax.set_xscale("log")
        ax.axvline(1, color="#eb6834", lw=1, ls="--", label="energy null")
        ax.set_ylim(-0.2, 1.05)
        ax.set_xlabel("component energy / null")
        ax.set_ylabel("eta")
        ax.set_title(f"{title}, k = {k_main}, all runs", loc="left", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
