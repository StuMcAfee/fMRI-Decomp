"""Route B, phase R3 (CLAUDE.md section 8; docs/ROUTE_B_PLAN.md step 4): injection validation at region level.

Synthetic "networks" made of real cortical parcels of the participant (from the route B region labels) receive known
region-wide signals in the raw echoes, before any processing: every voxel of every parcel of a network carries the
network's time course with the network's physics. Default: two dR2*-only networks, one dS0-only, one mixed (dS0/S0
= 0.032 s x dR2*), 8 parcels each, at resting strength. Conditions: independent (each network its own time course)
and shared (all networks one time course). The run then goes through the unchanged pipeline (mbme_tedana, front end)
and route B with the run's own region labels; the result is compared with the truth and with route B on the same
run without injection:

- amplitude and share: each injected region's S0-like and R2*-like time courses regressed on the true (projected)
  time course: recovered / injected amplitude, and the injected part's BOLD-like share against its truth;
- coupling: change of the BOLD-like correlations within injected networks, between them, and elsewhere;
- dimensionality: change of the participation ratio and of the number of eigenvalues above the circular-shift
  null (independent: up to +4 patterns; shared: +1);
- region size: amplitude error by region size (sets the smallest usable region).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def place(table: list, cfg: dict, rng) -> list:
    """Disjoint networks of usable cortical parcels (random, both hemispheres). Returns
    [{"id", "type", "parcels": [ids]}]."""
    L = cfg["inject"].get("region_layout", {})
    types = list(L.get("types", ["r2", "r2", "s0", "mixed"]))
    k = int(L.get("parcels_per_network", 8))
    vmin = int(L.get("min_voxels", cfg.get("regions", {}).get("min_voxels", 20)))
    pool = [r["id"] for r in table if r["kind"] == "cortex" and r.get("usable") and r["n_voxels"] >= vmin]
    if len(pool) < k * len(types):
        raise RuntimeError(f"only {len(pool)} usable cortical regions for {len(types)} networks of {k}")
    pick = list(rng.permutation(pool)[: k * len(types)])
    return [{"id": f"net{i + 1}_{t}", "type": t, "parcels": sorted(int(x) for x in pick[i * k:(i + 1) * k])}
            for i, t in enumerate(types)]


def weights(labels3d: np.ndarray, networks: list) -> np.ndarray:
    """(X, Y, Z, n) weights: 1 on every voxel of the network's parcels."""
    W = np.zeros(labels3d.shape + (len(networks),), np.float32)
    for i, n in enumerate(networks):
        W[..., i] = np.isin(labels3d, n["parcels"])
    return W


# ============================================================================ recovery
def _unit(x, Q=None):
    x = x - x.mean(0)
    if Q is not None and Q.size:
        x = x - Q @ (Q.T @ x)
    return x


def recover_dir(inject_dir, route_b_dir, baseline_dir, te_check: bool = False) -> dict:
    """Compare route B on the injected run (route_b_dir) with the truth and with route B on the same run without
    injection (baseline_dir). Writes recovery_regions.json / .md / .png in inject_dir."""
    from . import io
    idir, rb, base = Path(inject_dir), Path(route_b_dir), Path(baseline_dir)
    truth = json.loads((idir / "truth.json").read_text())
    U = np.genfromtxt(idir / "truth_timecourses.tsv", delimiter="\t", names=True)
    U = np.column_stack([U[n] for n in U.dtype.names]).astype(np.float64)
    tc = np.load(rb / "region_timecourses.npz")
    a, b, R, Q = tc["a"].astype(np.float64), tc["b"].astype(np.float64), tc["R"], tc["nuisance_Q"]
    T = b.shape[1]
    U = U[:T]
    Q = Q[:T] if Q.size else None
    if Q is not None and Q.size:
        Q = np.linalg.qr(Q)[0]
    Up = _unit(U, Q)
    rho = Up.std(0) / U.std(0)
    inj = json.loads((rb / "route_b.json").read_text())
    bas = json.loads((base / "route_b.json").read_text())
    used = {r["id"]: r for r in inj["regions"] if r["used"]}
    rows = []
    for i, net in enumerate(truth["sources"]):
        a_true, b_true = float(net["a_sd"]), float(net["b_sd_per_s"])
        for pid in net["parcels"]:
            p = pid - 1
            if pid not in used or not np.isfinite(b[p]).all():
                rows.append({"network": net["id"], "type": net["type"], "region": pid, "used": False})
                continue
            xb, xa = b[p] - b[p].mean(), a[p] - a[p].mean()
            # the region carries b_true x (projected truth): regressing on the projected truth returns b_true itself
            bh = float(xb @ Up[:, i] / (Up[:, i] @ Up[:, i]))
            ah = float(xa @ Up[:, i] / (Up[:, i] @ Up[:, i]))
            R00, R01, R11 = R[p, 0, 0], R[p, 0, 1], R[p, 1, 1]
            er = lambda bb: (R00 * bb / 1000.0) ** 2                    # channel energy of the R2*-like part
            es = lambda aa: (R01 ** 2 + R11 ** 2) * aa ** 2
            share_hat = er(bh) / (er(bh) + es(ah)) if er(bh) + es(ah) > 0 else np.nan
            share_true = er(b_true) / (er(b_true) + es(a_true)) if er(b_true) + es(a_true) > 0 else np.nan
            rows.append({"network": net["id"], "type": net["type"], "region": pid, "used": True,
                         "n_voxels": used[pid]["n_voxels"], "b_true": b_true, "a_true": a_true,
                         "b_hat": round(bh, 5), "a_hat": round(ah, 6),
                         "b_ratio": round(bh / b_true, 3) if b_true else None, "a_ratio": round(ah / a_true, 3) if a_true else None,
                         "share_injected": round(float(share_hat), 4), "share_true": round(float(share_true), 4)})
    # coupling and dimensionality against the run without injection
    ids_i, ids_b = inj["cortex_ids"], bas["cortex_ids"]
    common = sorted(set(ids_i) & set(ids_b))
    Ci, Cb = np.load(rb / "corr_b.npy"), np.load(base / "corr_b.npy")
    pi = {x: k for k, x in enumerate(ids_i)}
    pb = {x: k for k, x in enumerate(ids_b)}
    net_of = {pid: net["id"] for net in truth["sources"] for pid in net["parcels"]}
    d_within, d_between, d_mixed, d_other = [], [], [], []
    for x in range(len(common)):
        for y in range(x + 1, len(common)):
            r1, r2 = common[x], common[y]
            ci, cb = Ci[pi[r1], pi[r2]], Cb[pb[r1], pb[r2]]
            if not np.isfinite(ci):
                continue
            d = ci - (cb if np.isfinite(cb) else 0.0)                  # no detectable signal without injection: no coupling
            n1, n2 = net_of.get(r1), net_of.get(r2)
            if n1 is not None and n1 == n2:
                d_within.append(d)
            elif n1 is not None and n2 is not None:
                d_between.append(d)
            elif n1 is not None or n2 is not None:
                d_mixed.append(d)                                      # one injected region, one not
            else:
                d_other.append(d)                                      # neither injected: should not change
    ii, bb = inj["integration"][0], bas["integration"][0]
    med = lambda v: round(float(np.median(v)), 4) if len(v) else None
    used_rows = [r for r in rows if r["used"]]
    sizes = np.array([r["n_voxels"] for r in used_rows]) if used_rows else np.array([])
    size_bins = []
    for lo, hi in ((20, 40), (40, 80), (80, 160), (160, 10 ** 6)):
        sel = [r for r in used_rows if lo <= r["n_voxels"] < hi and r["b_ratio"] is not None]
        if sel:
            sel_s = [r for r in used_rows if lo <= r["n_voxels"] < hi and np.isfinite(r["share_injected"]) and np.isfinite(r["share_true"])]
            size_bins.append({"voxels": f"{lo}-{hi if hi < 10 ** 6 else ''}", "n": len(sel),
                              "median_abs_b_error": med([abs(r["b_ratio"] - 1) for r in sel]),
                              "median_abs_share_error": med([abs(r["share_injected"] - r["share_true"]) for r in sel_s])})
    by_type = {}
    for t in sorted({r["type"] for r in used_rows}):
        rr = [r for r in used_rows if r["type"] == t]
        by_type[t] = {"n_regions": len(rr),
                      "b_ratio_median": med([r["b_ratio"] for r in rr if r["b_ratio"] is not None]),
                      "a_ratio_median": med([r["a_ratio"] for r in rr if r["a_ratio"] is not None]),
                      "share_injected_median": med([r["share_injected"] for r in rr if np.isfinite(r["share_injected"])]),
                      "share_true": rr[0]["share_true"],
                      "share_abs_error_median": med([abs(r["share_injected"] - r["share_true"]) for r in rr
                                                     if np.isfinite(r["share_injected"]) and np.isfinite(r["share_true"])])}
    out = {"run": truth["run"], "level": truth["level"], "condition": truth.get("condition"), "seed": truth["seed"],
           "kept_by_nuisance_projection": [round(float(x), 4) for x in rho],
           "by_type": by_type, "regions": rows, "size_bins": size_bins,
           "coupling": {"within_injected_networks": med(d_within), "between_injected_networks": med(d_between),
                        "injected_to_other": med(d_mixed), "other_pairs": med(d_other),
                        "other_pairs_abs_change_median": med(np.abs(d_other)) if d_other else None,
                        "n_within": len(d_within), "n_between": len(d_between), "n_other": len(d_other)},
           "dimensionality": {"participation_ratio": [bb["participation_ratio"], ii["participation_ratio"]],
                              "pr_without_first": [bb.get("pr_without_first"), ii.get("pr_without_first")],
                              "eigen_above_null": [bb["n_eigen_above_null"], ii["n_eigen_above_null"]],
                              "eigen_above_null_deflated": [bb.get("n_eigen_above_null_deflated"), ii.get("n_eigen_above_null_deflated")],
                              "segregation": [bb["segregation"], ii["segregation"]],
                              "global_share": [bb["global_share"], ii["global_share"]]},
           **io.provenance({}, {})}
    io.write_json(idir / "recovery_regions.json", out)
    io.write_text(idir / "recovery_regions.md", render(out))
    return out


def render(o: dict) -> str:
    L = [f"# R3 injection recovery: {o['run']}, {o['condition']} networks, level {o['level']}, seed {o['seed']}", "",
         "Injected networks of real cortical parcels; route B on the injected run against the truth and against route B "
         "on the same run without injection. Amplitudes: region time courses regressed on the true time course "
         "(after the nuisance projection); share: the injected part's BOLD-like share.", "",
         "| network type | regions | dR2* recovered / injected | dS0 recovered / injected | injected share (truth) | share error |",
         "|---|---|---|---|---|---|"]
    for t, v in o["by_type"].items():
        L.append(f"| {t} | {v['n_regions']} | {v['b_ratio_median']} | {v['a_ratio_median']} | {v['share_injected_median']} ({v['share_true']}) | {v['share_abs_error_median']} |")
    c, d = o["coupling"], o["dimensionality"]
    L += ["", "## Coupling: change in BOLD-like correlation (injected minus without injection, median)", "",
          f"- within injected networks: {c['within_injected_networks']} ({c['n_within']} pairs)",
          f"- between injected networks: {c['between_injected_networks']} ({c['n_between']} pairs; independent: negative, added independent signal dilutes correlations; shared: positive)",
          f"- injected region with a region that was not injected: {c['injected_to_other']} (negative: dilution)",
          f"- pairs of regions that were not injected: {c['other_pairs']} (median absolute change {c['other_pairs_abs_change_median']}; should be ~0)", "",
          "## Integration (without injection -> injected)", "",
          f"- participation ratio: {d['participation_ratio'][0]} -> {d['participation_ratio'][1]}",
          f"- participation ratio without the first (brain-wide) pattern: {d.get('pr_without_first', [None, None])[0]} -> {d.get('pr_without_first', [None, None])[1]}",
          f"- eigenvalues above the circular-shift null: {d['eigen_above_null'][0]} -> {d['eigen_above_null'][1]} (independent: up to +4; shared: +1)",
          f"- deflated count (candidate): {d.get('eigen_above_null_deflated', [None, None])[0]} -> {d.get('eigen_above_null_deflated', [None, None])[1]}",
          f"- segregation: {d['segregation'][0]} -> {d['segregation'][1]}; global share: {d['global_share'][0]} -> {d['global_share'][1]}", "",
          "## Errors by region size", "",
          "Amplitude error: the injected signal sits only in the labelled voxels; motion and resampling spread part of it into "
          "unlabelled neighbours, more in small regions. The share is unaffected when both parts shrink alike.", "",
          "| voxels | regions | median abs dR2* amplitude error | median abs share error |", "|---|---|---|---|"]
    L += [f"| {s['voxels']} | {s['n']} | {s['median_abs_b_error']} | {s.get('median_abs_share_error')} |" for s in o["size_bins"]]
    return "\n".join(L) + "\n"


def summary(root, out_dir=None) -> dict:
    """Pool every recovery_regions.json under root by condition and level."""
    from . import io
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root
    recs = [json.loads(p.read_text()) for p in sorted(root.glob("**/recovery_regions.json"))]
    L = ["# R3: route B injection validation (pooled)", "",
         "| run | condition | level | type | dR2* rec/inj | dS0 rec/inj | injected share (truth) | within-network change | between-network change | uninjected pairs: median abs change | PR: before -> after | PR without first pattern: before -> after | global share: before -> after | eigenvalues above null: before -> after | deflated count: before -> after |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in recs:
        c, d = o["coupling"], o["dimensionality"]
        for t, v in o["by_type"].items():
            L.append(f"| {o['run']} | {o['condition']} | {o['level']} | {t} | {v['b_ratio_median']} | {v['a_ratio_median']} | "
                     f"{v['share_injected_median']} ({v['share_true']}) | {c['within_injected_networks']} | {c['between_injected_networks']} | "
                     f"{c.get('other_pairs_abs_change_median')} | {d['participation_ratio'][0]} -> {d['participation_ratio'][1]} | "
                     f"{d.get('pr_without_first', [None, None])[0]} -> {d.get('pr_without_first', [None, None])[1]} | "
                     f"{d['global_share'][0]} -> {d['global_share'][1]} | "
                     f"{d['eigen_above_null'][0]} -> {d['eigen_above_null'][1]} | "
                     f"{d.get('eigen_above_null_deflated', [None, None])[0]} -> {d.get('eigen_above_null_deflated', [None, None])[1]} |")
    allb, alls = {}, {}
    for o in recs:
        for s in o["size_bins"]:
            allb.setdefault(s["voxels"], []).append(s["median_abs_b_error"])
            alls.setdefault(s["voxels"], []).append(s.get("median_abs_share_error"))
    fmt = lambda d: "; ".join(f"{k} voxels {np.median([x for x in v if x is not None]):.3f}" for k, v in d.items() if any(x is not None for x in v))
    L += ["", "Amplitude error by region size (median over runs): " + fmt(allb),
          "", "Share error by region size (median over runs): " + fmt(alls)]
    res = {"n_recoveries": len(recs), "recoveries": recs}
    out_dir.mkdir(parents=True, exist_ok=True)
    io.write_json(out_dir / "inject_regions_summary.json", res)
    io.write_text(out_dir / "inject_regions_summary.md", "\n".join(L) + "\n")
    return res
