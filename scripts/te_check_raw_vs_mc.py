"""Is the echo-1 effect created by motion correction? (diagnostic, 2026-11-06)

The components of the motion-corrected joint-ICA sweep (their time courses, k folder) are measured in two
front ends of the same run with the identical method (te_check.check, plain least-squares betas of that front
end's own channels on the same time courses): the motion-corrected echoes and the raw echoes (no motion
correction; the same paired frames). mbme_tedana estimates motion on echo 1 alone and applies it to every echo;
if that creates the TE1-dominated, curved changes, they shrink or vanish in the raw echoes. The raw echoes carry
uncorrected motion (more edge effects), so a persisting effect is informative, a vanishing one decisive.

Usage: PYTHONPATH=src python scripts/te_check_raw_vs_mc.py <run> [--k 50] [--phase2-root outputs/phase2]
                                                              [--phase5-root outputs/phase5] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def measure(fe: Path, M: np.ndarray):
    import nibabel as nib

    from mejoint import te_check
    mask3d = np.asarray(nib.load(str(fe / "analysis_mask.nii.gz")).dataobj) > 0
    mapv = lambda n: np.asarray(nib.load(str(fe / f"{n}.nii.gz")).dataobj, np.float64)[mask3d]
    fs = json.loads((fe / "summary.json").read_text())
    ch = {n: np.load(fe / f"{n}.npy") for n in ("c0", "c1", "r0", "r1")}
    if ch["c0"].shape[1] != M.shape[0]:
        raise ValueError(f"{fe}: {ch['c0'].shape[1]} frames, the time courses have {M.shape[0]}")
    p = int(np.load(fe / "nuisance_Q.npy").shape[1]) if (fe / "nuisance_Q.npy").exists() else 0
    m0, m1 = te_check.ols_betas(ch["c0"], M), te_check.ols_betas(ch["c1"], M)
    out = te_check.check(ch, np.load(fe / "R.npy"), mapv("mu"), mapv("sigma"), fs["echo_noise_ratios_used"],
                         mapv("usable") > 0.5, mapv("t2s_ms"), fs["te_ms"], M, m0, m1, n_nuisance=p)
    out["nuisance_rank"] = p
    return out, fs["te_ms"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--phase2-root", default="outputs/phase2")
    ap.add_argument("--phase5-root", default="outputs/phase5")
    ap.add_argument("--out", default="outputs/te_check_raw_vs_mc")
    a = ap.parse_args(argv)
    from mejoint import io
    kd = Path(a.phase5_root) / a.run / "phase4_sweep_mc" / f"k{a.k}"
    fe_mc = Path(a.phase2_root) / a.run / "mc"
    fe_raw = next((p for p in (Path(a.phase2_root) / a.run / "raw", Path(a.phase2_root) / "all" / a.run / "raw")
                   if (p / "r0.npy").exists()), None)
    if fe_raw is None:
        print(f"{a.run}: no raw-echo front end with channels found; skipped")
        return 1
    tcs = np.genfromtxt(kd / "time_courses.tsv", delimiter="\t", names=True)
    M = np.column_stack([tcs[n] for n in tcs.dtype.names]).astype(np.float64)
    res = {}
    for name, fe in (("mc", fe_mc), ("raw", fe_raw)):
        res[name], te = measure(fe, M)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cm, cr = res["mc"]["components"], res["raw"]["components"]
    fmt = lambda v, f=".3f": "n/a" if v is None or not np.isfinite(v) else format(v, f)
    g = lambda cs, key: np.array([np.nan if c[key] is None else c[key] for c in cs], float)
    keys = ("curvature_fraction", "fraction_te1_te4_opposite", "ratio_ms_median", "n_sig")
    med = {name: {key: float(np.nanmedian(g(res[name]["components"], key))) for key in keys} for name in res}
    cross = {name: int(sum(c["profile_zero_crossing_ms"] is not None for c in res[name]["components"])) for name in res}
    paired = float(np.nanmedian(g(cr, "curvature_fraction") / g(cm, "curvature_fraction")))
    row = {"run": a.run, "k": a.k, "raw_frontend": str(fe_raw), "nuisance_rank_mc": res["mc"]["nuisance_rank"],
           "nuisance_rank_raw": res["raw"]["nuisance_rank"], "median": med, "n_zero_crossing": cross,
           "median_raw_over_mc_curvature": paired, "components": {"mc": cm, "raw": cr}}
    io.write_json(out / f"{a.run}_k{a.k}.json", row)
    # overlay figure: the 8 strongest components (by mc curvature fraction) in both front ends
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(-np.nan_to_num(g(cm, "curvature_fraction")))[:8]
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    tt = np.asarray(te, float)
    for ax, j in zip(axs.ravel(), order):
        ax.plot(tt, cm[j]["profile_pct"], "o-", color="#2a78d6", label="motion-corrected")
        ax.plot(tt, cr[j]["profile_pct"], "s--", color="#eb6834", label="raw echoes")
        ax.axhline(0, color="#e4e3df")
        ax.set_title(f"comp {j + 1}: curvature {fmt(cm[j]['curvature_fraction'])} mc, {fmt(cr[j]['curvature_fraction'])} raw",
                     fontsize=8.5, loc="left")
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{a.run}, k{a.k}: the same components' echo profiles (% change, dR2* > 0) in motion-corrected and raw echoes",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{a.run}_k{a.k}.png", dpi=110)
    plt.close(fig)
    print(f"{a.run} k{a.k}: curvature fraction median mc {fmt(med['mc']['curvature_fraction'])}, raw {fmt(med['raw']['curvature_fraction'])} "
          f"(paired raw/mc {fmt(paired, '.2f')}); TE1/TE4 opposite mc {med['mc']['fraction_te1_te4_opposite']:.2f}, raw "
          f"{med['raw']['fraction_te1_te4_opposite']:.2f}; zero crossings mc {cross['mc']}, raw {cross['raw']}; "
          f"ratio ms mc {fmt(med['mc']['ratio_ms_median'], '.1f')}, raw {fmt(med['raw']['ratio_ms_median'], '.1f')}; "
          f"nuisance rank mc {res['mc']['nuisance_rank']}, raw {res['raw']['nuisance_rank']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.exit(main())
