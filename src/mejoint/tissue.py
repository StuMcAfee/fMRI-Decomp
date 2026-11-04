"""Phase 5: data-only tissue classes and the partial-volume coupling factor (user decision 2026-11-04:
options (a) eta stratified by CSF partial volume, (c) a partial-volume coupling diagnostic).

Two-compartment fit of each voxel's temporal-mean decay (4 echoes):
    mu_e = A_t exp(-TE_e / T2t) + A_c exp(-TE_e / T2c)
T2c (long, CSF-like) is fixed (`tissue.csf_t2s_ms`); A_t, A_c >= 0 and T2t on a grid are fitted by
weighted least squares (weights 1 / (sigma r_e)^2). f_long = A_c / (A_t + A_c) is the long-T2* signal
fraction at TE = 0. A convex decay gives f_long > 0; a mono-exponential or a concave one (through-slice
dephasing) gives f_long ~ 0.

Partial-volume coupling: a pure dR2* change Delta b in the tissue compartment changes the log signal by
-w_t(TE) TE Delta b (w_t = tissue share of the fitted signal at each TE). Passed through the voxel's real
front-end projection, it gives an apparent dS0 per unit apparent dR2*: kappa_v = a / q (q in s^-1).
For a component, s_pv = kappa_v q_v is the dS0 map that partial volume alone predicts.

Data-only classes (grey vs white matter is not separable from these EPI echoes; the T1 option will be):
    1 tissue (f_long < tissue_max), 2 mixed, 3 CSF-dominated (f_long >= csf_min),
    4 short T2* (T2t < short_t2s_ms: veins / sinuses or susceptibility near air and bone; ambiguous),
    5 few usable echoes (< 3: dropout; f_long not estimable).
"""
from __future__ import annotations

import numpy as np

CLASS_NAMES = {1: "tissue", 2: "mixed", 3: "csf", 4: "short_t2s", 5: "few_echoes"}


def two_compartment_fit(mu: np.ndarray, te_ms, use: np.ndarray, w: np.ndarray, t2c_ms: float,
                        t2t_grid=None) -> dict:
    """Grid over T2t, closed-form non-negative 2-parameter WLS per voxel. mu, use, w: (V, E)."""
    te = np.asarray(te_ms, float)
    grid = np.arange(8.0, 150.0, 1.0) if t2t_grid is None else np.asarray(t2t_grid, float)
    V = mu.shape[0]
    W = np.where(use, w, 0.0)
    ec = np.exp(-te / t2c_ms)
    best = {"cost": np.full(V, np.inf), "A_t": np.zeros(V), "A_c": np.zeros(V), "t2t": np.full(V, np.nan)}
    for t2 in grid:
        et = np.exp(-te / t2)
        # normal equations [[Stt, Stc], [Stc, Scc]] [A_t, A_c] = [Sty, Scy]
        Stt = (W * et ** 2).sum(1); Scc = (W * ec ** 2).sum(1); Stc = (W * et * ec).sum(1)
        Sty = (W * et * mu).sum(1); Scy = (W * ec * mu).sum(1)
        det = Stt * Scc - Stc ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            At = np.where(det > 0, (Sty * Scc - Scy * Stc) / det, 0.0)
            Ac = np.where(det > 0, (Scy * Stt - Sty * Stc) / det, 0.0)
            # non-negativity: fall back to the best single-compartment fit
            t_only = Sty / Stt
            bad = (At < 0) | (Ac < 0) | ~np.isfinite(At) | ~np.isfinite(Ac)
            At = np.where(bad, np.maximum(t_only, 0), At)
            Ac = np.where(bad, 0.0, Ac)
        fit = At[:, None] * et[None, :] + Ac[:, None] * ec[None, :]
        cost = (W * (mu - fit) ** 2).sum(1)
        better = cost < best["cost"]
        for k, v in (("cost", cost), ("A_t", At), ("A_c", Ac)):
            best[k] = np.where(better, v, best[k])
        best["t2t"] = np.where(better, t2, best["t2t"])
    tot = best["A_t"] + best["A_c"]
    with np.errstate(divide="ignore", invalid="ignore"):
        best["f_long"] = np.where(tot > 0, best["A_c"] / tot, np.nan)
    ok = use.sum(1) >= 3
    for k in ("f_long", "t2t"):
        best[k] = np.where(ok, best[k], np.nan)
    best["t2c_ms"] = float(t2c_ms)
    return best


def pv_kappa(fit: dict, mu: np.ndarray, te_ms, use: np.ndarray, weights_m: np.ndarray) -> np.ndarray:
    """kappa_v = apparent dS0/S0 per unit apparent dR2* (s^-1) for a pure tissue dR2* change, through the
    voxel's own projection (weights_m = the front end's noise-scaled mean / sigma, (V, E))."""
    from .frontend import project

    te = np.asarray(te_ms, float)
    et = fit["A_t"][:, None] * np.exp(-te[None, :] / np.where(np.isfinite(fit["t2t"]), fit["t2t"], 1.0)[:, None])
    ec = fit["A_c"][:, None] * np.exp(-te[None, :] / fit["t2c_ms"])
    with np.errstate(divide="ignore", invalid="ignore"):
        wt = np.where(et + ec > 0, et / (et + ec), 1.0)
    y = (-wt * te[None, :])[:, :, None]                                   # log change per unit Delta b (1/ms)
    pr = project(np.where(use[:, :, None], y, 0.0), weights_m, use, te)
    c = pr["c"][:, :, 0].astype(np.float64)
    R = pr["R"]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = c[:, 1] / R[:, 1, 1]
        q = (c[:, 0] - R[:, 0, 1] * a) / R[:, 0, 0] * 1000.0             # s^-1 per unit Delta b in 1/ms
        kappa = np.where(np.isfinite(a) & (q != 0), a / q, np.nan)
    kappa[~np.isfinite(fit["f_long"])] = np.nan
    return kappa


def classes(fit: dict, n_use: np.ndarray, cfg: dict) -> np.ndarray:
    T = cfg["tissue"]
    lab = np.full(n_use.shape, 2, np.int8)
    f, t2 = fit["f_long"], fit["t2t"]
    lab[f < float(T["tissue_max"])] = 1
    lab[f >= float(T["csf_min"])] = 3
    lab[(t2 < float(T["short_t2s_ms"])) & (lab != 3)] = 4
    lab[(n_use < 3) | ~np.isfinite(f)] = 5
    return lab


def run(mu, te_ms, use, sigma, ratios, cfg) -> dict:
    """Fit, kappa and classes from front-end quantities (V, E) / (V,)."""
    T = cfg["tissue"]
    w = 1.0 / (sigma[:, None] * np.asarray(ratios)[None, :]) ** 2
    fit = two_compartment_fit(mu, te_ms, use, w, float(T["csf_t2s_ms"]))
    m = (mu / np.asarray(ratios)[None, :]) / sigma[:, None]
    kappa = pv_kappa(fit, mu, te_ms, use, m)
    lab = classes(fit, use.sum(1), cfg)
    return {**fit, "kappa": kappa, "classes": lab}


def run_dir(frontend_dir, cfg: dict, out_dir) -> dict:
    """Data-only tissue classes for one front-end output folder."""
    from pathlib import Path

    import nibabel as nib

    from . import io

    fd, out_dir = Path(frontend_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mimg = nib.load(str(fd / "analysis_mask.nii.gz"))
    mask3d = np.asarray(mimg.dataobj) > 0
    get = lambda name: np.asarray(nib.load(str(fd / f"{name}.nii.gz")).dataobj)[mask3d]
    mu, use, sigma = get("mu").astype(np.float64), get("usable") > 0.5, get("sigma").astype(np.float64)
    import json
    fs = json.loads((fd / "summary.json").read_text())
    ratios = np.asarray(fs.get("echo_noise_ratios_used") or cfg["frontend"]["echo_noise_ratios"], float)
    res = run(mu, fs["te_ms"], use, sigma, ratios, cfg)
    for k in ("f_long", "t2t", "kappa"):
        io.save_map(res[k], mask3d, mimg, out_dir / f"{k}.nii.gz")
    io.save_map(res["classes"].astype(np.float32), mask3d, mimg, out_dir / "classes.nii.gz", fill=0)
    np.save(out_dir / "classes.npy", res["classes"])
    np.save(out_dir / "kappa.npy", res["kappa"])
    np.save(out_dir / "f_long.npy", res["f_long"])
    vox_ml = float(np.prod(mimg.header.get_zooms()[:3])) / 1000.0
    summ = {"run_id": fs.get("run_id"), "source": "data-only (multi-echo decay)", "csf_t2s_ms": res["t2c_ms"],
            "class_names": CLASS_NAMES,
            "class_ml": {CLASS_NAMES[c]: round(float((res["classes"] == c).sum() * vox_ml), 1) for c in CLASS_NAMES},
            "f_long_p10_50_90": [round(float(x), 4) for x in np.nanpercentile(res["f_long"], [10, 50, 90])],
            "kappa_p10_50_90": [round(float(x), 6) for x in np.nanpercentile(res["kappa"], [10, 50, 90])]}
    io.write_json(out_dir / "tissue.json", summ)
    L = [f"# Data-only tissue classes: {summ['run_id']}", "",
         f"Two-compartment fit of the mean decay with a fixed long T2* of {summ['csf_t2s_ms']:.0f} ms. Grey vs white matter is "
         "not separable from these echoes (T1 option later). Short-T2* voxels may be veins / sinuses or susceptibility "
         "near air and bone (ambiguous).", "",
         "| Class | Volume (ml) |", "|---|---|"] + [f"| {k} | {v} |" for k, v in summ["class_ml"].items()] + [
         "", f"Long-T2* signal fraction p10/50/90: {summ['f_long_p10_50_90']}",
         f"Partial-volume coupling kappa (apparent dS0/S0 per apparent s^-1) p10/50/90: {summ['kappa_p10_50_90']}"]
    io.write_text(out_dir / "tissue.md", "\n".join(L) + "\n")
    io.write_json(out_dir / "done.json", {"stage": "tissue_data_only", **io.provenance(cfg, {"frontend_summary": str(fd / "summary.json")})})
    return summ
