"""Cross-participant summary of a wave batch (`mejoint waves-summary`): for every run folder under
outputs/waves/<run>/<commit>/, the leading modes of each variant (no regression, motion-regressed,
motion-orthogonal) with their motion correlations, energy against the null, eta, the dS0 / dR2* sign check by
region, direction, and the run's mean framewise displacement. Question it answers first (user, 2026-11-05):
is the coupling of the leading mode with the y translation seen in 47159 unique to that participant?

This is a description across participants, not the group stage (brief section 7; W4): no labels, no tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VARIANTS = ("noreg", "reg", "noreg_motion_orth")
NAMES = ("rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z")


def _fd(frontend_dir) -> tuple[float, float]:
    """Mean framewise displacement (raw and filtered, mm) from the front end's frames.tsv."""
    try:
        fr = np.genfromtxt(Path(frontend_dir) / "frames.tsv", delimiter="\t", names=True)
    except (OSError, ValueError, TypeError):
        return np.nan, np.nan
    get = lambda k: (float(np.nanmean(fr[k])) if k in (fr.dtype.names or ()) and np.isfinite(fr[k]).any()
                     else np.nan)
    return get("fd_mm"), get("fd_filtered_mm")


def collect(root, commit: str, max_modes: int = 3) -> tuple[list[dict], list[str]]:
    """One row per run, variant and mode (the first `max_modes` modes, plus any further mode above the null)."""
    rows, missing = [], []
    for run in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        base = run / commit
        if not (base / "noreg" / "waves.json").exists():
            missing.append(run.name)
            continue
        for var in VARIANTS:
            f = base / var / "waves.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())
            if s.get("skipped"):
                rows.append({"run": run.name, "variant": var, "mode": 0, "skipped": s["skipped"]})
                continue
            fd, fdf = _fd(s.get("frontend_dir", ""))
            reg = s.get("by_region", {})
            for r in s["modes"]:
                k = int(r["mode"])
                if k > max_modes and r.get("above_null") is not True:
                    continue
                g = {x["region"]: x for x in reg.get(str(k), reg.get(k, []))}
                row = {"run": run.name, "variant": var, "mode": k, "above_null": r.get("above_null"),
                       "eig_over_null": r.get("eigenvalue_over_null"), "f_c_hz": r.get("f_c_hz"),
                       "var_pct": round(100 * float(r.get("var_frac_stacked", np.nan)), 2),
                       "motion_r_max": r.get("motion_r_max"), "motion_best_param": r.get("motion_best_param"),
                       "motion_r_null95": r.get("motion_r_null95"),
                       **{f"r_{n}": r.get(f"motion_r_{n}") for n in NAMES},
                       "eta": r.get("eta"), "s0r2_n": r.get("s0r2_n_voxels"), "s0r2_null_n": r.get("s0r2_null_n_expected"),
                       "s0r2_opposed": r.get("s0r2_fraction_sign_opposed"),
                       "opposed_per_present_edge": (g.get("edge") or {}).get("opposed_per_present"),
                       "opposed_per_present_interior": (g.get("interior") or {}).get("opposed_per_present"),
                       "phase_lag_deg_inferior": (g.get("inferior_third") or {}).get("phase_lag_deg"),
                       "direction": r.get("direction_label"), "speed_mm_s": r.get("speed_mm_s"),
                       "fit_quality": r.get("fit_quality"), "R": r.get("R"),
                       "explained_fraction": r.get("explained_fraction_median_ref"),
                       "fd_mean_mm": round(fd, 4), "fd_filtered_mean_mm": round(fdf, 4)}
                rows.append(row)
    return rows, missing


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    return x


def render(rows: list[dict], missing: list[str], commit: str) -> str:
    L = [f"# Wave batch summary (code {commit})", "",
         "Leading wave modes across participants, from `outputs/waves/<run>/" + commit + "/`. A description, not a "
         "test: no group labels, no multiple-comparison control. Motion r = |complex correlation| of the mode time "
         "course with each band-passed motion parameter; null95 = 95th percentile of the largest of 6 with "
         "phase-randomized motion parameters.", ""]
    runs = sorted({r["run"] for r in rows})
    L.append(f"Runs summarized: {len(runs)}. Without results for this commit: {len(missing)}"
             + (f" ({', '.join(missing)})" if missing else "") + ".")
    for var in VARIANTS:
        m1 = [r for r in rows if r["variant"] == var and r["mode"] == 1]
        if not m1:
            continue
        above = [r for r in m1 if r.get("above_null") is True]
        coupled = [r for r in m1 if _num(r["motion_r_max"]) > _num(r["motion_r_null95"])]
        best = {}
        for r in coupled:
            best[r["motion_best_param"]] = best.get(r["motion_best_param"], 0) + 1
        ty = np.array([_num(r["r_trans_y"]) for r in m1])
        L += ["", f"## {var}: mode 1", "",
              f"- above the energy null: {len(above)} of {len(m1)} runs;",
              f"- motion-coupled (largest motion r above its null): {len(coupled)} of {len(m1)}; strongest parameter: "
              + (", ".join(f"{k} {v}" for k, v in sorted(best.items(), key=lambda kv: -kv[1])) or "none") + ";",
              f"- r with trans_y: median {np.nanmedian(ty):.2f}, range {np.nanmin(ty):.2f}-{np.nanmax(ty):.2f}." if np.isfinite(ty).any()
              else "- r with trans_y: not available.", "",
              "| Run | Eig / null | Var % | f_c Hz | Motion r max (parameter) | null95 | r trans_y | Mean FD mm | eta | dS0 "
              "voxels (null) | opposed / present: edge, interior | Direction |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda v, n=2: "" if not np.isfinite(_num(v)) else f"{_num(v):.{n}f}"
        for r in sorted(m1, key=lambda r: r["run"]):
            L.append(f"| {r['run']} | {f(r['eig_over_null'])} | {f(r['var_pct'], 1)} | {f(r['f_c_hz'], 3)} | "
                     f"{f(r['motion_r_max'])} ({r['motion_best_param']}) | {f(r['motion_r_null95'])} | {f(r['r_trans_y'])} | "
                     f"{f(r['fd_mean_mm'], 3)} | {f(r['eta'], 3)} | {f(r['s0r2_n'], 0)} ({f(r['s0r2_null_n'], 1)}) | "
                     f"{f(r['opposed_per_present_edge'])}, {f(r['opposed_per_present_interior'])} | {r['direction']} |")
    L += ["", "Reading: a motion-coupled leading mode in many runs means the coupling is not unique to one "
              "participant; the motion-orthogonal variant shows what remains once the band-passed motion is removed. "
              "Interpret only modes above the energy null."]
    return "\n".join(L) + "\n"


def figure(rows: list[dict], path, variant: str = "noreg") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m1 = sorted([r for r in rows if r["variant"] == variant and r["mode"] == 1], key=lambda r: r["run"])
    if not m1:
        return
    R = np.array([[_num(r[f"r_{n}"]) for n in NAMES] for r in m1])
    null = np.array([_num(r["motion_r_null95"]) for r in m1])
    fd = np.array([_num(r["fd_mean_mm"]) for r in m1])
    fig, axs = plt.subplots(1, 2, figsize=(13, 0.35 * len(m1) + 2.5), gridspec_kw={"width_ratios": [1.4, 1]})
    im = axs[0].imshow(R, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(len(m1)):
        for j in range(len(NAMES)):
            if np.isfinite(R[i, j]):
                axs[0].text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if R[i, j] < 0.6 else "black", fontweight="bold" if R[i, j] > null[i] else None)
    axs[0].set_xticks(range(len(NAMES)), NAMES)
    axs[0].set_yticks(range(len(m1)), [r["run"].split("_")[0] for r in m1])
    axs[0].set_title(f"Mode 1 vs motion parameters ({variant}); bold = above its null", loc="left", fontsize=10)
    fig.colorbar(im, ax=axs[0], fraction=0.04).set_label("|correlation|")
    rmax = np.nanmax(R, axis=1)
    axs[1].scatter(fd, rmax, c=["#eb6834" if a > b else "#2a78d6" for a, b in zip(rmax, null)])
    for x, y, r in zip(fd, rmax, m1):
        axs[1].annotate(r["run"].split("_")[0].replace("sub-", ""), (x, y), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
    axs[1].set_xlabel("mean framewise displacement (mm)")
    axs[1].set_ylabel("largest motion |r| of mode 1")
    axs[1].set_title("Coupling vs head motion (orange = above null)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(root, commit: str, out_dir) -> dict:
    from . import io
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = collect(root, commit)
    if rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        io.write_tsv(out_dir / "waves_batch.tsv", {c: [r.get(c, "") for r in rows] for c in cols})
    io.write_text(out_dir / "waves_batch.md", render(rows, missing, commit))
    figure(rows, out_dir / "waves_batch_motion.png")
    io.write_json(out_dir / "waves_batch.json", {"commit": commit, "rows": rows, "missing": missing})
    return {"n_rows": len(rows), "missing": missing}
