"""R5: route B over all participants (docs/R5_PLAN.md, approved 2026-11-11).

- preflight: one table of every run's inputs (front end, echo files, kept volumes and gaps, T1, anatomy, coil);
- common_regions: the cortical parcels usable in every participant (R5-2), and how many each participant removes;
- summary: participant, region and deep-region tables, the pooled thalamus / striatum coupling (R5-4), the
  per-participant injection sensitivity (R5-3) and the cohort checks (QC-FC and its distance dependence).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _subject(run: str) -> str:
    return run.split("_")[0].replace("sub-", "")


def list_runs(phase2_root, cfg: dict) -> list[str]:
    """Run folders with a motion-corrected front end, excluded subjects dropped."""
    excl = {str(k) for k in (cfg.get("subjects", {}).get("exclude") or {})}
    out = []
    for d in sorted(Path(phase2_root).glob("sub-*")):
        if (d / "mc" / "done.json").exists() and _subject(d.name) not in excl:
            out.append(d.name)
    return out


def _volumes(fe: Path):
    """Original frame numbers of the kept frames (echo 1) and the number of frames, from the front end."""
    done = json.loads((fe / "done.json").read_text())
    vp = done.get("inputs", {}).get("volumes_used")
    vp = vp["path"] if isinstance(vp, dict) else vp
    n = None
    if (fe / "frames.tsv").exists():
        n = sum(1 for _ in open(fe / "frames.tsv")) - 1
    vols = None
    if vp and Path(vp).exists():
        vu = np.loadtxt(vp, skiprows=1, ndmin=2).astype(int)
        vols = vu[:, 0]
        n = n if n is not None else vols.size
    return vols, n, done


def preflight(phase2_root, anat_root, cfg: dict, t1_list=None, out_dir=None, atlas: str = "200_7") -> list[dict]:
    from . import anatomy, io, regions
    G = cfg.get("regions", {})
    head_only = {str(s) for s in cfg.get("acquisition", {}).get("coil_head_only", [])}
    rows = []
    for run in list_runs(phase2_root, cfg):
        fe = Path(phase2_root) / run / "mc"
        vols, n, done = _volumes(fe)
        ins = done.get("inputs", {})
        echoes = [ins.get(f"echo{k}") for k in range(1, 5)]
        echoes = [e["path"] if isinstance(e, dict) else e for e in echoes]
        ok_e = all(e and Path(e).exists() for e in echoes)
        tl = regions.select_frames(vols if vols is not None and n == vols.size else None, n or 0, G.get("n_volumes_common")) if n else None
        t1 = anatomy.find_t1(run, t1_list or G.get("t1_list"), G.get("t1_root"))
        rows.append({"run": run, "echo_files": "ok" if ok_e else "MISSING",
                     "frames": n, "gap_free_stretches": tl["n_segments"] if tl else None,
                     "frames_used": tl["n_used"] if tl else None,
                     "primary_eligible": tl["primary_eligible"] if tl else False,
                     "t1": t1 or "NOT FOUND",
                     "anatomy": (Path(anat_root) / run / f"labels_{atlas}.nii.gz").exists(),
                     "coil": "head only" if _subject(run) in head_only else "head-neck"})
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        io.write_tsv(out / "preflight.tsv", {k: [r[k] for r in rows] for k in rows[0]} if rows else {"run": []})
        L = ["# R5 preflight", "", f"{len(rows)} runs (excluded subjects dropped). Frames used: the first "
             f"{G.get('n_volumes_common')} of the longest gap-free stretch (R5-1).", "",
             "| run | echo files | frames | gap-free stretches | frames used | in primary | T1 | anatomy | coil |",
             "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['run']} | {r['echo_files']} | {r['frames']} | {r['gap_free_stretches']} | {r['frames_used']} | "
              f"{r['primary_eligible']} | {r['t1']} | {r['anatomy']} | {r['coil']} |" for r in rows]
        io.write_text(out / "preflight.md", "\n".join(L) + "\n")
    return rows


def common_regions(anat_root, runs: list[str], atlas: str, out_file) -> dict:
    """Cortical parcels usable in every participant (R5-2) and, per participant, how many parcels it alone removes
    (usable in all the others but not in it): the number that shows whether one participant shrinks the set."""
    from . import io
    usable = {}
    for run in runs:
        f = Path(anat_root) / run / f"regions_{atlas}.json"
        if f.exists():
            rr = json.loads(f.read_text())["regions"]
            usable[run] = {r["id"] for r in rr if r["kind"] == "cortex" and r["usable"]}
    if not usable:
        raise FileNotFoundError(f"no regions_{atlas}.json under {anat_root}")
    common = set.intersection(*usable.values())
    per = []
    for run, u in usable.items():
        others = [v for k, v in usable.items() if k != run]
        alone = (set.intersection(*others) - u) if others else set()
        per.append({"run": run, "usable": len(u), "removed_alone": len(alone)})
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(i) for i in sorted(common)) + "\n")
    L = [f"# Common region set ({atlas})", "", f"{len(common)} cortical parcels usable in all {len(usable)} participants "
         f"(written to {out.name}).", "", "| run | usable parcels | parcels only this participant removes |", "|---|---|---|"]
    L += [f"| {p['run']} | {p['usable']} | {p['removed_alone']} |" for p in sorted(per, key=lambda x: -x["removed_alone"])]
    io.write_text(out.with_suffix(".md"), "\n".join(L) + "\n")
    return {"n_common": len(common), "ids": sorted(common), "per_participant": per}


def _fz(r):
    return np.arctanh(np.clip(np.asarray(r, float), -0.999, 0.999))


def pooled_coupling(values: list[tuple[float, float, float]]) -> dict:
    """R5-4: inverse-variance weighted mean of Fisher-z values over the measurable participants; each value's
    standard error from its 95 % bootstrap interval ((z_hi - z_lo) / (2 * 1.96))."""
    v = [(x, lo, hi) for x, lo, hi in values if x is not None and lo is not None and hi is not None and hi > lo]
    if not v:
        return {"n": 0, "pooled": None, "ci": None}
    z = _fz([x for x, _, _ in v])
    se = (_fz([h for _, _, h in v]) - _fz([lo for _, lo, _ in v])) / (2 * 1.96)
    w = 1.0 / np.maximum(se, 1e-6) ** 2
    m = float(np.sum(w * z) / np.sum(w))
    s = float(1.0 / np.sqrt(np.sum(w)))
    return {"n": len(v), "pooled": round(float(np.tanh(m)), 4), "ci": [round(float(np.tanh(m - 1.96 * s)), 4), round(float(np.tanh(m + 1.96 * s)), 4)]}


def _injection(inject_root, run: str) -> dict:
    """Per-participant sensitivity (R5-3) from the region-level injections at resting strength."""
    out = {}
    for f in sorted((Path(inject_root) / run).glob("regions_*_s*/recovery_regions.json")):
        o = json.loads(f.read_text())
        if o.get("level") != "weak":
            continue
        d = o["dimensionality"]
        pr, pr1 = d.get("participation_ratio", [None, None]), d.get("pr_without_first", [None, None])
        dd = lambda a: round(a[1] - a[0], 4) if a and None not in a else None
        errs = [abs(v["share_injected_median"] - v["share_true"]) for v in o["by_type"].values()
                if v.get("share_injected_median") is not None and v.get("share_true") is not None]
        out[o.get("condition")] = {"d_pr": dd(pr), "d_pr_without_first": dd(pr1), "max_share_error": round(max(errs), 4) if errs else None}
    return out


def _motion(phase2_root, run: str, tl: dict | None) -> dict:
    f = Path(phase2_root) / run / "mc" / "frames.tsv"
    if not f.exists():
        return {}
    import csv
    with open(f) as fh:
        rr = list(csv.DictReader(fh, delimiter="\t"))
    try:
        fd = np.array([float(r["fd_filtered_mm"]) for r in rr])
    except (KeyError, ValueError):
        return {}
    if tl and tl.get("longest_stretch") and tl.get("n_used"):
        a = tl["longest_stretch"][0]
        fd = fd[a:a + tl["n_used"]]
    return {"fd_mean": round(float(fd.mean()), 4), "fd_max": round(float(fd.max()), 4), "fd_frac_above_0p2": round(float(np.mean(fd > 0.2)), 4)}


def _centroids(atlas: str) -> dict:
    import nibabel as nib
    repo = Path(__file__).resolve().parents[2]
    n_par, n_net = atlas.split("_")
    img = nib.load(str(repo / "atlases" / "schaefer2018" / f"Schaefer2018_{n_par}Parcels_{n_net}Networks_order_FSLMNI152_2mm.nii.gz"))
    lab = np.rint(np.asarray(img.dataobj)).astype(int)
    out = {}
    for i in np.unique(lab[lab > 0]):
        ijk = np.argwhere(lab == i).mean(0)
        out[int(i)] = (img.affine[:3, :3] @ ijk + img.affine[:3, 3])
    return out


def _without_first(C: np.ndarray) -> np.ndarray:
    """Correlation matrix with its first (brain-wide) pattern removed: C - lambda_1 v1 v1', renormalized to unit
    diagonal (the matrix the participation ratio without the first pattern describes). Rows with NaN stay NaN."""
    from .regions import _signal_rows
    C = np.asarray(C, float)
    ok = _signal_rows(C)                                   # regions with signal (a NaN row must not blank the rest)
    out = np.full_like(C, np.nan)
    if ok.sum() < 3:
        return out
    S = C[np.ix_(ok, ok)]
    w, V = np.linalg.eigh(S)
    R = S - w[-1] * np.outer(V[:, -1], V[:, -1])
    d = np.sqrt(np.clip(np.diag(R), 1e-12, None))
    out[np.ix_(ok, ok)] = R / np.outer(d, d)
    return out


def _qcfc(edges: list, fd: np.ndarray, dist: np.ndarray) -> dict:
    """QC-FC: each edge (Fisher z) correlated across participants with mean FD; median |r|, fraction p < 0.05
    (about 0.05 expected without a motion effect) and the distance dependence of r."""
    from scipy.stats import pearsonr, spearmanr
    Z = np.array([_fz(e) for e in edges])
    ok = np.all(np.isfinite(Z), axis=0)
    r = np.full(Z.shape[1], np.nan)
    pv = np.full(Z.shape[1], np.nan)
    for j in np.flatnonzero(ok):
        r[j], pv[j] = pearsonr(Z[:, j], fd)
    m = np.isfinite(r)
    rho_d = spearmanr(r[m], dist[m])[0] if m.sum() > 10 else np.nan
    return {"n_participants": len(edges), "n_edges": int(m.sum()), "median_abs_r": round(float(np.nanmedian(np.abs(r))), 3),
            "median_r": round(float(np.nanmedian(r)), 3), "frac_p_below_0p05": round(float(np.nanmean(pv[m] < 0.05)), 3),
            "distance_dependence_spearman": round(float(rho_d), 3) if np.isfinite(rho_d) else None}


def summary(route_b_root, phase2_root, inject_root, cfg: dict, out_dir, atlas: str = "200_7") -> dict:
    from scipy.stats import pearsonr, spearmanr

    from . import io
    head_only = {str(s) for s in cfg.get("acquisition", {}).get("coil_head_only", [])}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    part, reg, deep, mats = [], [], [], {}
    for f in sorted(Path(route_b_root).glob(f"sub-*/{atlas}/route_b.json")):
        s = json.loads(f.read_text())
        run = f.parent.parent.name
        integ = s["integration"]
        prim = integ[0]
        by = {i["set"]: i for i in integ}
        common = by.get("common set")
        relset = next((i for k, i in by.items() if k.startswith("cortex, reliability")), None)
        allv = next((i for k, i in by.items() if k.startswith("cortex, all voxels")), None)
        tl = s.get("timeline") or {}
        cx = [r for r in s["regions"] if r["kind"] == "cortex" and r["used"]]
        rel = [r["b_reliability"] for r in cx if r.get("b_reliability") is not None]
        cov = s.get("cortex_csf_covariate") or {}
        inj = _injection(inject_root, run)
        g = lambda i, k: (i or {}).get(k)
        part.append({"run": run, "in_primary": tl.get("primary_eligible", True), "frames": s["n_frames"],
                     "regions": prim["n_regions"], "regions_common": g(common, "n_regions"),
                     "pr": prim["participation_ratio"], "pr_null_median": prim["pr_null_median"],
                     "pr_without_first": prim.get("pr_without_first"), "global_share": prim["global_share"],
                     "segregation": prim["segregation"], "eigen_above_null": prim["n_eigen_above_null"],
                     "pr_common": g(common, "participation_ratio"), "pr_without_first_common": g(common, "pr_without_first"),
                     "global_share_common": g(common, "global_share"),
                     "pr_allvox": g(allv, "participation_ratio"), "pr_without_first_allvox": g(allv, "pr_without_first"),
                     "pr_reliable": g(relset, "participation_ratio"),
                     "cortex_csf_fraction": cov.get("cortex_csf_mean_allvox"),
                     "reliability_median": round(float(np.median(rel)), 4) if rel else None,
                     "reliability_q05": round(float(np.quantile(rel, 0.05)), 4) if rel else None,
                     "coil": "head only" if _subject(run) in head_only else "head-neck",
                     **_motion(phase2_root, run, tl),
                     "inj_independent_d_pr": g(inj.get("independent"), "d_pr"),
                     "inj_independent_d_pr_without_first": g(inj.get("independent"), "d_pr_without_first"),
                     "inj_shared_d_pr": g(inj.get("shared"), "d_pr"),
                     "inj_shared_d_pr_without_first": g(inj.get("shared"), "d_pr_without_first"),
                     "inj_max_share_error": max([v["max_share_error"] for v in inj.values() if v.get("max_share_error") is not None], default=None)})
        for r in s["regions"]:
            reg.append({"run": run, "id": r["id"], "name": r["name"], "kind": r["kind"], "network": r["network"], "used": r["used"],
                        "n_voxels": r["n_voxels"], "bold_share": r["bold_share"],
                        "bold_share_ci_lo": (r.get("bold_share_ci") or [None, None])[0],
                        "bold_share_ci_hi": (r.get("bold_share_ci") or [None, None])[1],
                        "s0_share": r["s0_share"], "leftover_share": r["leftover_share"],
                        "reliability": r.get("b_reliability"), "energy_below_noise": r.get("energy_below_noise", "")})
        for d in s.get("secondary_coupling", []):
            nets = [k for k in d if k not in ("name", "noise_overcorrected", "ci", "b_reliability", "not_measurable")]
            for n in nets:
                ci = (d.get("ci") or {}).get(n) or [None, None]
                deep.append({"run": run, "region": d["name"], "network": n, "measurable": not d.get("not_measurable", False),
                             "coupling": d[n], "ci_lo": ci[0], "ci_hi": ci[1], "reliability": d.get("b_reliability")})
        cb = f.parent / "corr_b.npy"
        if cb.exists() and tl.get("primary_eligible", True):
            mats[run] = (np.load(cb), s["cortex_ids"])
    # pooled deep coupling over measurable participants (R5-4)
    pooled = []
    for key in sorted({(d["region"], d["network"]) for d in deep}):
        vals = [(d["coupling"], d["ci_lo"], d["ci_hi"]) for d in deep if (d["region"], d["network"]) == key and d["measurable"]]
        pooled.append({"region": key[0], "network": key[1], "n_total": sum(1 for d in deep if (d["region"], d["network"]) == key),
                       **pooled_coupling(vals)})
    # cohort checks on the participants in the primary
    P = [p for p in part if p["in_primary"]]
    checks = {"n_participants": len(P)}
    import warnings
    warnings.filterwarnings("ignore", message="An input array is constant")
    if len(P) >= 5:
        # each integration measure against data-quality and anatomy variables (and the primary against the global
        # share: R5 review, 2026-11-12, found the primary tracking the brain-wide pattern across participants)
        for meas in ("pr", "pr_without_first", "global_share"):
            for k in ("fd_mean", "cortex_csf_fraction", "regions", "reliability_median", "global_share"):
                if k == meas:
                    continue
                xy = [(p.get(meas), p.get(k)) for p in P if p.get(k) is not None and p.get(meas) is not None]
                if len(xy) >= 5:
                    rho, pv = spearmanr([a for a, _ in xy], [b for _, b in xy])
                    checks[f"{meas}_vs_{k}"] = {"spearman": round(float(rho), 3), "p": round(float(pv), 4), "n": len(xy)}
        for c in ("head only", "head-neck"):
            v = [p["pr"] for p in P if p["coil"] == c and p["pr"] is not None]
            checks[f"pr_mean_{c}"] = {"mean": round(float(np.mean(v)), 3) if v else None, "n": len(v)}
        # QC-FC on the common parcels: edge (Fisher z) across participants against mean FD, and its distance dependence
        ids_sets = [set(ids) for _, ids in mats.values()]
        cids = sorted(set.intersection(*ids_sets)) if ids_sets else []
        runs_fd = [p["run"] for p in P if p.get("fd_mean") is not None and p["run"] in mats]
        if len(cids) >= 3 and len(runs_fd) >= 5:
            fd = np.array([next(p["fd_mean"] for p in P if p["run"] == run) for run in runs_fd])
            cen = _centroids(atlas)
            iu = np.triu_indices(len(cids), 1)
            dist = np.array([np.linalg.norm(cen[cids[a]] - cen[cids[b]]) for a, b in zip(*iu)])
            subs = []
            for run in runs_fd:
                C, ids = mats[run]
                pos = {v: i for i, v in enumerate(ids)}
                ix = [pos[i] for i in cids]
                subs.append(np.asarray(C, float)[np.ix_(ix, ix)])
            checks["qc_fc"] = _qcfc([M[iu] for M in subs], fd, dist)
            # the same after removing each participant's first (brain-wide) pattern from its matrix
            checks["qc_fc_without_first"] = _qcfc([_without_first(M)[iu] for M in subs], fd, dist)
    # write
    tsv = lambda name, rows: io.write_tsv(out / name, {k: [r.get(k) for r in rows] for k in rows[0]}) if rows else None
    tsv("participants.tsv", part)
    tsv("regions.tsv", reg)
    tsv("deep_regions.tsv", deep)
    tsv("deep_pooled.tsv", pooled)
    res = {"atlas": atlas, "n_volumes_common": cfg.get("regions", {}).get("n_volumes_common"), "participants": part, "deep_pooled": pooled, "cohort_checks": checks, **io.provenance(cfg, {})}
    io.write_json(out / "summary.json", res)
    io.write_text(out / "summary.md", render(res))
    return res


def render(s: dict) -> str:
    L = [f"# R5: route B over all participants ({s['atlas']})", "",
         f"Primary: cortex without CSF-rich voxels, first {s.get('n_volumes_common')} frames of the longest gap-free stretch. 'in primary' = "
         "False: the stretch was too short (reported, not in the cohort checks). Injection: change at resting strength "
         "(independent networks / one shared network).", "",
         "Primary (user, 2026-11-12): PR without the first (brain-wide) pattern, read with the global share; the ordinary "
         "PR is secondary (across participants it mostly tracks the global share). dPR-1 / dPR: change with injection.", "",
         "| run | in primary | regions (common) | PR without first (primary) | global share | PR | segregation | PR all voxels | "
         "CSF fraction | reliability median | mean FD | coil | inj. indep. dPR-1 / dPR | inj. shared dPR-1 / dPR |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in s["participants"]:
        r4 = lambda v: round(v, 4) if isinstance(v, float) else v
        L.append(f"| {p['run']} | {p['in_primary']} | {p['regions']} ({p.get('regions_common')}) | {p.get('pr_without_first')} | "
                 f"{p['global_share']} | {p['pr']} | {p['segregation']} | {p.get('pr_allvox')} | {r4(p.get('cortex_csf_fraction'))} | "
                 f"{p.get('reliability_median')} | {p.get('fd_mean')} | {p['coil']} | "
                 f"{p.get('inj_independent_d_pr_without_first')} / {p.get('inj_independent_d_pr')} | "
                 f"{p.get('inj_shared_d_pr_without_first')} / {p.get('inj_shared_d_pr')} |")
    L += ["", "## Thalamus and striatum: pooled coupling over measurable participants (Fisher z, inverse-variance weights)", "",
          "| region | network | measurable / total | pooled [95 % CI] |", "|---|---|---|---|"]
    L += [f"| {d['region']} | {d['network']} | {d['n']} / {d['n_total']} | {d['pooled']} {d['ci']} |" for d in s["deep_pooled"]]
    L += ["", "## Cohort checks (participants in the primary)", "", "```", json.dumps(s["cohort_checks"], indent=1), "```"]
    return "\n".join(L) + "\n"
