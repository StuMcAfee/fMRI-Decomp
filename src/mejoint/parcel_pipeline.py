"""Two-stage parcel denoising and network pipeline on the icosahedral parcels (docs/PARCEL_ICA_PLAN.md section 8;
user decisions 2026-11-23). Exploratory: route B is not changed.

Stage 1, physics: echo-averaged parcel signals split into BOLD-like and S0-like parts, thermal noise from the
         leftover channels (route B, icosahedral export); each parcel's BOLD-like signal whitened to unit noise and
         its reliability (signal share of its slow-band variance) measured.
Stage 2, non-neural removal, repeated: spatial ICA at the map-fidelity count of the remaining data (user, 2026-11-23;
         originally the leftover-channel count, too small to give breathing its own component); components
         meeting the non-neural criteria (signal share above 0.12 Hz, or S0 coupling) are projected out of the BOLD-like
         signal, both channels and the leftover (null) channels; repeat on what remains until no component qualifies
         (at most max_rounds; reaching the cap is reported). The degrees of freedom shrink by one per removed
         component.
Stage 3, networks: spatial ICA on the cleaned signal with k from the thermal-noise (Marchenko-Pastur) threshold
         (capped); per component its best positive Yeo-7 / Yeo-17 overlap and hemispheric asymmetry; winner-take-all
         partition (winner where its z-scored loading >= z_min).
Measures: reliable coverage (floor from the precision of the noise correction), node count (contiguous same-winner
         patches over all parcels with voxels; undominated and weak parcels are gaps; single parcels count), patches per winning component, undominated share; participation
         ratio (all patterns and without the first) with a circular-shift null; participation coefficient and system
         segregation (weighted, positive, noise-corrected correlations clipped at 1) under Yeo-7, Yeo-17 and the
         data-driven partition; three parcel sets: reliable parcels (main), every parcel with voxels (A), reliable
         parcels with undominated parcels assigned to their strongest component (B).
Outputs (research storage only): pipeline.json / .md, measures.tsv, parcels.tsv, denoised.npz, figures, report.html.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import parcel_ica as pi

DEFAULTS = {"k_max": 100, "max_rounds": 5, "z_min": 1.0, "n_seeds": 30, "n_null": 50, "high_share": 0.5, "eta": 0.5,
            "high_hz": 0.12, "stage3_k": "fidelity", "fidelity_min": 0.7, "edge_fdr_q": 0.05, "dominance_q": 0.95,
            "correction_error_max": 0.15, "stage2_k": "fidelity"}


# ------------------------------------------------------------------ anatomy of the icosahedral parcels
def ico_anatomy(resources, n_left: int = 938) -> dict:
    """Yeo-7 and Yeo-17 network of every icosahedral parcel (vertex majority in Schaefer 200/7 and 200/17 on fs_LR 32k)
    and the parcel adjacency (parcels sharing a mesh edge), in model order (id 1..1876)."""
    import nibabel as nib

    from . import surface
    R = Path(resources)
    il = np.asarray(nib.load(str(R / "Icosahedron1002.L.label.gii")).darrays[0].data, int)
    ir = np.asarray(nib.load(str(R / "Icosahedron1002.R.label.gii")).darrays[0].data, int)
    n = 2 * n_left
    out = {"n": n}
    for tag, f in (("yeo7", "Schaefer2018_200Parcels_7Networks_order.dlabel.nii"), ("yeo17", "Schaefer2018_200Parcels_17Networks_order.dlabel.nii")):
        if not (R / f).exists():
            out[tag] = None
            continue
        sl, sr, names = surface.schaefer_from_dlabel(R / f)
        table, _, _ = surface.ico_table(il, ir, sl, sr, names, n_left)
        out[tag] = np.array([table.get(i, {}).get("network", "") for i in range(1, n + 1)], dtype=object)
    pairs = set()
    for h, lab, off in (("L", il, 0), ("R", ir, n_left)):
        faces = np.asarray(nib.load(str(R / f"tpl-fs32k_hemi-{h}_inflated.surf.gii")).darrays[1].data, int)
        fl = lab[faces]
        for a, b in ((0, 1), (1, 2), (0, 2)):
            x, y = fl[:, a], fl[:, b]
            m = (x != y) & (x > 0) & (y > 0)
            for p, q in zip(x[m] + off, y[m] + off):
                pairs.add((min(p, q), max(p, q)))
    out["adjacent_pairs"] = np.array(sorted(pairs), int) if pairs else np.zeros((0, 2), int)
    return out


# ------------------------------------------------------------------ measures
def reliability_floor(T: int, tr: float, band, max_error: float = 0.15) -> float:
    """Lowest reliability r at which the noise-correction factor 1/sqrt(r) is known within max_error (relative): the
    in-band variance is estimated with about 2 x (frequency bins in the band) degrees of freedom, so its relative SE
    is s = sqrt(2 / df); the factor's relative error is about (1 - r) / r x s / 2, giving r = 1 / (1 + 2 max_error / s)."""
    f = np.fft.rfftfreq(int(T), tr)
    nb = int(((f >= band[0]) & (f <= band[1])).sum())
    s = np.sqrt(2.0 / max(2 * nb, 1))
    return float(1.0 / (1.0 + 2.0 * max_error / s))


def reliability(z: np.ndarray, tr: float, band) -> np.ndarray:
    """Signal share of each whitened parcel signal's slow-band variance: 1 - expected noise / observed (unit noise)."""
    from .regions import band_fraction, band_pass
    vb = band_pass(z, tr, band).var(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - band_fraction(z.shape[1], tr, band) / vb


def node_count(ids: np.ndarray, winner: np.ndarray, adjacent_pairs: np.ndarray) -> dict:
    """Contiguous same-winner patches among parcels that are nodes (winner >= 0); everything else is a gap."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ids = np.asarray(ids, int)
    node = np.asarray(winner) >= 0
    pos = {int(p): i for i, p in enumerate(ids)}
    rows, cols = [], []
    for p, q in adjacent_pairs:
        i, j = pos.get(int(p)), pos.get(int(q))
        if i is None or j is None or not (node[i] and node[j]) or winner[i] != winner[j]:
            continue
        rows += [i, j]
        cols += [j, i]
    nn = int(node.sum())
    if nn == 0:
        return {"node_count": 0, "patch_sizes": []}
    sub = np.flatnonzero(node)
    remap = -np.ones(len(ids), int)
    remap[sub] = np.arange(nn)
    if rows:
        r_, c_ = remap[np.array(rows, int)], remap[np.array(cols, int)]
    else:
        r_, c_ = np.zeros(0, int), np.zeros(0, int)
    A = coo_matrix((np.ones(len(r_)), (r_, c_)), shape=(nn, nn)).tocsr()
    ncomp, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab)
    return {"node_count": int(ncomp), "patch_sizes": sorted(sizes.tolist(), reverse=True), "patch_label": lab, "node_index": sub}


def positive_weights(corr: np.ndarray) -> np.ndarray:
    """Weighted positive graph: noise-corrected correlations clipped to [0, 1], no self-connections, NaN -> 0."""
    W = np.clip(np.nan_to_num(np.asarray(corr, float), nan=0.0), 0.0, 1.0)
    np.fill_diagonal(W, 0.0)
    return W


def participation_coefficient(W: np.ndarray, labels) -> np.ndarray:
    """P_i = 1 - sum_m (s_im / s_i)^2 over modules (weighted, positive); nodes without a module or without strength:
    NaN. Only nodes with a module enter the graph."""
    labels = np.asarray(labels, dtype=object)
    has = np.array([(l is not None) and (l != "") and (l != -1) for l in labels])
    pc = np.full(len(labels), np.nan)
    if has.sum() < 2:
        return pc
    Ws = W[np.ix_(has, has)]
    lab = labels[has]
    s = Ws.sum(1)
    acc = np.zeros(len(lab))
    for m in set(lab.tolist()):
        acc += (Ws[:, lab == m].sum(1) / np.where(s > 0, s, np.nan)) ** 2
    pc[has] = np.where(s > 0, 1.0 - acc, np.nan)
    return pc


def segregation_positive(W: np.ndarray, labels) -> float:
    """(mean within-module weight - mean between-module weight) / mean within-module weight (weighted, positive)."""
    labels = np.asarray(labels, dtype=object)
    has = np.array([(l is not None) and (l != "") and (l != -1) for l in labels])
    if has.sum() < 3:
        return float("nan")
    Ws, lab = W[np.ix_(has, has)], labels[has]
    iu = np.triu_indices(len(lab), 1)
    same = lab[iu[0]] == lab[iu[1]]
    v = Ws[iu]
    if not same.any() or same.all():
        return float("nan")
    w, b = v[same].mean(), v[~same].mean()
    return float((w - b) / w) if w > 0 else float("nan")


def remove_first_pattern(R: np.ndarray) -> np.ndarray:
    """Correlations beyond the brain-wide pattern (user, 2026-11-23): the first eigencomponent (lambda_1 v1 v1') is
    removed from the noise-corrected correlation matrix of the parcels with signal and the remainder is rescaled to
    unit diagonal, so each entry is the correlation of two parcels after the pattern all of them share. Rows without
    signal (NaN diagonal) stay NaN."""
    R = np.asarray(R, float)
    ok = np.isfinite(np.diag(R))
    out = np.full_like(R, np.nan)
    if ok.sum() < 3:
        return out
    S = R[np.ix_(ok, ok)].copy()
    S = np.where(np.isfinite(S), S, 0.0)
    lam, V = np.linalg.eigh(S)
    v = V[:, -1]
    Rr = S - lam[-1] * np.outer(v, v)
    d = np.diag(Rr).copy()
    good = d > 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        Rn = Rr / np.sqrt(np.outer(np.where(good, d, np.nan), np.where(good, d, np.nan)))
    np.fill_diagonal(Rn, np.where(good, 1.0, np.nan))
    out[np.ix_(ok, ok)] = Rn
    return out


def pr_null(z: np.ndarray, noise_var: np.ndarray, tr: float, band, n: int, min_shift: int, rng) -> dict:
    """Circular-shift null (each parcel's band-passed signal shifted by its own random lag): the participation ratio
    (all patterns and without the first) per surrogate, and for every edge the mean and SD of its noise-corrected
    correlation over the surrogates (accumulated, not stored), for the edge test."""
    from .regions import band_fraction, band_pass
    T = z.shape[1]
    xb = band_pass(np.asarray(z, float), tr, band)
    P = xb.shape[0]
    nb = np.asarray(noise_var, float) * band_fraction(T, tr, band)
    pr, pr1 = np.zeros(n), np.zeros(n)
    s1, s2, cnt = np.zeros((P, P)), np.zeros((P, P)), np.zeros((P, P))
    for i in range(n):
        lags = rng.integers(min_shift, T - min_shift + 1, P)
        xs = np.stack([np.roll(xb[p], int(lags[p])) for p in range(P)])
        C = xs @ xs.T / T
        sig = np.diag(C) - nb
        ok = sig > 0
        d = np.sqrt(np.where(ok, sig, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            Rf = C / np.outer(d, d)
        Rb = remove_first_pattern(Rf)                                  # edges are tested beyond the brain-wide pattern
        fin = np.isfinite(Rb)
        s1 += np.where(fin, Rb, 0.0)
        s2 += np.where(fin, Rb ** 2, 0.0)
        cnt += fin
        Rr = Rf[np.ix_(ok, ok)].copy()
        np.fill_diagonal(Rr, 1.0)
        lam = np.sort(np.clip(np.linalg.eigvalsh(Rr), 0, None))[::-1]
        pr[i] = lam.sum() ** 2 / (lam ** 2).sum()
        rest = lam[1:]
        pr1[i] = rest.sum() ** 2 / (rest ** 2).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        sd = np.sqrt(np.clip(s2 / cnt - mean ** 2, 0, None) * cnt / np.maximum(cnt - 1, 1))
    return {"pr": pr, "pr_resid": pr1, "edge_mean": mean, "edge_sd": sd, "edge_n": cnt}


def significant_edges(corr: np.ndarray, null: dict, q: float = 0.05) -> np.ndarray:
    """Edges reliably above chance (user, 2026-11-23; tested on the correlations beyond the brain-wide pattern, with the
    same removal applied to every surrogate): each edge's correlation against its own
    circular-shift null (z = (r - null mean) / null SD, one-sided, normal approximation), Benjamini-Hochberg false
    discovery rate q over all edges. Returns a symmetric boolean matrix (diagonal False)."""
    from scipy.stats import norm
    P = corr.shape[0]
    iu = np.triu_indices(P, 1)
    r, m, sd = corr[iu], null["edge_mean"][iu], null["edge_sd"][iu]
    ok = np.isfinite(r) & np.isfinite(m) & np.isfinite(sd) & (sd > 0)
    p = np.ones(r.size)
    p[ok] = norm.sf((r[ok] - m[ok]) / sd[ok])
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    passed = np.zeros(p.size, bool)
    below = np.flatnonzero(ranked <= q)
    if below.size:
        passed[order[: below.max() + 1]] = True
    S = np.zeros((P, P), bool)
    S[iu] = passed & ok
    return S | S.T


def integration(z: np.ndarray, noise_var: np.ndarray, tr: float, band, partitions: dict, null=None, fdr_q: float = 0.05) -> dict:
    """Integration measures on one parcel set: noise-corrected in-band correlations (route B's coupling), the
    participation ratio (all patterns / without the first) from them, and per partition the participation
    coefficient (mean over nodes and per parcel) and segregation on the positive weights (clipped at 1)."""
    from . import regions
    cp = regions.coupling(z, noise_var, tr, band)
    corr = cp["corr"]
    out = {"n_parcels": int(z.shape[0]), "n_with_signal": int(np.isfinite(np.diag(corr)).sum()),
           "pr": regions.participation_ratio(corr), "pr_resid": regions.participation_ratio_residual(corr),
           "global_share": regions.global_share(cp["x_band"], cp["signal_var"], cp["noise_band_var"])}
    if null is not None:
        out["pr_null_median"] = float(np.median(null["pr"]))
        out["pr_resid_null_median"] = float(np.median(null["pr_resid"]))
        out["pr_resid_null_q05"] = float(np.quantile(null["pr_resid"], 0.05))
    # graph: correlations beyond the brain-wide pattern, positive, edges reliably above chance (user, 2026-11-23)
    corr_b = remove_first_pattern(corr)
    out["corr_beyond_first"] = True
    W = positive_weights(corr_b)
    if null is not None and "edge_sd" in null:                         # keep only edges reliably above chance
        S = significant_edges(corr_b, null, fdr_q)
        W = np.where(S, W, 0.0)
        nsig = np.isfinite(np.diag(corr))
        npairs = nsig.sum() * (nsig.sum() - 1) / 2
        out["edges_kept"] = int(np.triu(W > 0, 1).sum())
        out["edge_density"] = float(out["edges_kept"] / npairs) if npairs else float("nan")
        out["nodes_without_edges"] = int(((W > 0).sum(1) == 0)[nsig].sum())
        out["edge_rule"] = f"beyond the brain-wide pattern, above the circular-shift null, FDR q = {fdr_q}"
    out["W"] = W
    out["pc"], out["pc_map"], out["seg"] = {}, {}, {}
    for name, lab in partitions.items():
        if lab is None:
            continue
        pc = participation_coefficient(W, lab)
        out["pc"][name] = float(np.nanmean(pc)) if np.isfinite(pc).any() else float("nan")
        out["pc_map"][name] = pc
        out["seg"][name] = segregation_positive(W, lab)
    return out


def dominance(z: np.ndarray, maps: np.ndarray, tc: np.ndarray) -> dict:
    """Per parcel: strength z1 (strongest z-scored loading), margin z1 - z2, and the confidence that the strongest
    component truly exceeds the second strongest given the loadings' estimation uncertainty (user, 2026-11-23,
    replacing a shuffled-map null that called clear dominance unclear for sparse maps). Each parcel's loadings are the
    regression of its whitened signal on the component time courses; their standard errors come from the parcel's
    residual variance and (tc tc')^-1, expressed in the same z units as the maps (divided by each map's SD over
    parcels); confidence = Phi(margin / sqrt(SE1^2 + SE2^2))."""
    from scipy.stats import norm
    from .surface_plot import zscore
    maps, tc, z = np.asarray(maps, float), np.asarray(tc, float), np.asarray(z, float)
    Zm = zscore(maps)
    n = Zm.shape[0]
    if Zm.shape[1] < 2:
        return {"z1": Zm[:, 0] if Zm.shape[1] else np.zeros(n), "margin": np.full(n, np.nan), "confidence": np.full(n, np.nan)}
    k, T = tc.shape
    resid = z - maps @ tc
    rv = resid.var(1) * T / max(T - k, 1)
    cinv = np.diag(np.linalg.inv(tc @ tc.T))
    se = np.sqrt(rv[:, None] * cinv[None, :]) / np.maximum(maps.std(0), 1e-12)[None, :]
    o = np.argsort(Zm, axis=1)
    i = np.arange(n)
    j1, j2 = o[:, -1], o[:, -2]
    z1, margin = Zm[i, j1], Zm[i, j1] - Zm[i, j2]
    conf = norm.cdf(margin / np.sqrt(se[i, j1] ** 2 + se[i, j2] ** 2))
    return {"z1": z1, "margin": margin, "confidence": conf, "se_median": float(np.median(se))}


def component_labels(maps: np.ndarray, ids: np.ndarray, nets7, nets17, n_left: int = 938) -> list:
    """Per component: best positive Yeo-7 and Yeo-17 overlap (correlation of the z-scored map with the network
    indicator) and hemispheric asymmetry of its positive loadings ((left - right) / (left + right))."""
    from .surface_plot import zscore
    Z = zscore(np.asarray(maps, float))
    left = np.asarray(ids) <= n_left
    out = []
    for j in range(Z.shape[1]):
        rec = {}
        for tag, nets in (("yeo7", nets7), ("yeo17", nets17)):
            if nets is None:
                rec[tag], rec[tag + "_r"] = "", None
                continue
            nets = np.asarray(nets, dtype=object)
            best, br = "", -np.inf
            for n in sorted(set(nets.tolist()) - {""}):
                ind = (nets == n).astype(float)
                if ind.std() == 0:
                    continue
                r = float(np.corrcoef(Z[:, j], ind)[0, 1])
                if r > br:
                    best, br = n, r
            rec[tag], rec[tag + "_r"] = best, round(br, 3)
        pos = np.clip(Z[:, j], 0, None)
        sl, sr = pos[left].sum(), pos[~left].sum()
        rec["asymmetry"] = round(float((sl - sr) / (sl + sr)), 3) if sl + sr > 0 else None
        out.append(rec)
    return out


# ------------------------------------------------------------------ the pipeline
def run(input_dir, cfg: dict, out_dir, resources=None, n_jobs: int | None = None) -> dict:
    """The whole pipeline on one participant's icosahedral export (outputs/route_b_surf/<run>/ico1002_7)."""
    from . import io
    from .regions import band_fraction
    P_ = {**DEFAULTS, **cfg.get("parcel_pipeline", {})}
    G = cfg.get("regions", {})
    band = tuple(G.get("band_hz", (0.01, 0.10)))
    rmin_route_b = float(G.get("reliability_min", 0.5))
    tr = float(cfg["acquisition"]["tr_s"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resources = Path(resources or cfg.get("surface", {}).get("resources") or Path(__file__).resolve().parents[2] / "outputs" / "atlases_fs32k")
    inp = pi.load_inputs(input_dir)
    pr = pi.prepare(inp["c0"], inp["c1"], inp["R"], inp["noise_var"], inp["r"], inp["frames"])
    ok = pr["ok"]
    ids = np.asarray(inp["ids"])[ok]
    T = pr["zb"].shape[1]
    nuis = int(inp["nuisance_rank"])
    anat = ico_anatomy(resources)
    n_total = anat["n"]
    nets7 = anat["yeo7"][ids - 1] if anat["yeo7"] is not None else None
    nets17 = anat["yeo17"][ids - 1] if anat["yeo17"] is not None else None
    hemi = np.where(ids <= n_total // 2, "L", "R")
    folds = (hemi == "R").astype(int)
    resp = None
    try:
        fs = json.loads((Path(inp["frontend_dir"]) / "summary.json").read_text())
        resp = (fs.get("nuisance", {}).get("filter", {}).get("respiratory_peak") or {}).get("freq_hz")
    except Exception:
        pass
    seed_root = int(cfg.get("project_seed", 0))
    seeds = lambda stage: [int(x) for x in np.random.SeedSequence(seed_root, spawn_key=(43, stage)).generate_state(int(P_["n_seeds"]))]
    rule = lambda sp, s0: (sp["high_share"] > P_["high_share"]) | (np.nan_to_num(s0["eta"], nan=0.0) > P_["eta"])

    # ---- stage 2: repeated removal of non-neural components
    Z, CW = pr["zb"].copy(), pr["cw"].copy()
    nulls = [n.copy() for n in pr.get("null_channels") or []]
    removed, rounds, stop = [], [], "maximum number of rounds reached"
    for rnd in range(int(P_["max_rounds"])):
        df = T - nuis - 1 - len(removed)
        dim = pi.dimensionality(Z, df, nulls or None)
        kn = dim.get("k_null")
        # detection k (user, 2026-11-23): the map-fidelity count of the remaining data, as in stage 3; at the
        # leftover-channel count (3-6 in this protocol) breathing could not get its own component and was missed
        fid2 = pi.map_fidelity(dim["eigenvalues"], Z.shape[0], df)
        kf = int(np.sum(fid2 >= float(P_["fidelity_min"])))
        k = int(min(kf if P_["stage2_k"] == "fidelity" else (kn or 0), P_["k_max"]))
        rec = {"round": rnd + 1, "df": int(df), "k_null": None if kn is None else int(kn), "k_fidelity": kf, "k": k,
               "mp_edge": dim["mp_edge"], "null_threshold": dim.get("null_threshold"),
               "fidelity_eigenvalue": pi.fidelity_threshold(float(P_["fidelity_min"]), Z.shape[0], df),
               "eigenvalues_top": [round(float(x), 3) for x in dim["eigenvalues"][:60]]}
        if k < 2:
            rounds.append({**rec, "components": []})
            stop = "fewer than 2 components with reliable maps"
            break
        ica = pi.spatial_ica(Z, k, seeds(rnd), n_jobs=n_jobs)
        sp = pi.spectral_measures(ica["tc"], tr, band, resp_hz=resp or 0.3, high_hz=P_["high_hz"], noise_var=ica["tc_noise_var"])
        s0 = pi.s0_coupling(ica["maps"], Z, CW, pr["rho"], folds)
        fl = rule(sp, s0)
        comps = [{"round": rnd + 1, "component": j, "removed": bool(fl[j]), "iq": round(float(ica["iq"][j]), 3),
                  "band_share": round(float(sp["band_share"][j]), 3), "high_share": round(float(sp["high_share"][j]), 3),
                  "resp_share": round(float(sp["resp_share"][j]), 3), "peak_hz": round(float(sp["peak_hz"][j]), 3),
                  "peak_ratio": round(float(sp["peak_ratio"][j]), 2),
                  "eta": None if not np.isfinite(s0["eta"][j]) else round(float(s0["eta"][j]), 3)} for j in range(k)]
        rec["components"] = comps
        rec["maps_removed"] = ica["maps"][:, fl]
        rec["tc_removed"] = ica["tc"][fl]
        rounds.append(rec)
        if not fl.any():
            stop = "no component met the non-neural criteria"
            break
        rm = ica["tc"][fl]
        removed += list(rm)
        Z = pi.remove_components(Z, rm)
        CW = np.stack([pi.remove_components(CW[:, c], rm) for c in range(CW.shape[1])], 1)
        nulls = [pi.remove_components(n, rm) for n in nulls]
    q = len(removed)
    rm_all = np.array(removed) if q else np.zeros((0, T))
    den = pi.denoise_output(out, Path(input_dir), inp, pr, rm_all, np.ones(q, bool), tr, band, nets7 if nets7 is not None else [""] * len(ids), nuis)

    # ---- stage 3: networks on the cleaned signal (thermal-noise threshold)
    df3 = T - nuis - 1 - q
    dim3 = pi.dimensionality(Z, df3)
    # k: components whose maps are expected to correlate >= fidelity_min with the true maps (user, 2026-11-23); the
    # thermal-noise (Marchenko-Pastur) count is kept as the upper bound and reported
    fid = pi.map_fidelity(dim3["eigenvalues"], Z.shape[0], df3)
    k_fid = int(np.sum(fid >= float(P_["fidelity_min"])))
    k_rule = int(dim3["k_mp"]) if P_["stage3_k"] == "mp" else k_fid
    k3 = int(min(k_rule, P_["k_max"]))
    st3 = {"df": int(df3), "k_mp": int(dim3["k_mp"]), "k_fidelity": k_fid, "fidelity_min": float(P_["fidelity_min"]),
           "fidelity_eigenvalue": pi.fidelity_threshold(float(P_["fidelity_min"]), Z.shape[0], df3), "k_rule": P_["stage3_k"],
           "fidelity_top": [round(float(x), 3) for x in fid[:60]],
           "k": k3, "k_capped": bool(k_rule > P_["k_max"]), "mp_edge": dim3["mp_edge"],
           "eigenvalues_top": [round(float(x), 3) for x in dim3["eigenvalues"][:60]]}
    noise_scale = (T - nuis - q) / max(T - nuis, 1)                   # unit noise reduced by the removed dimensions
    rel = reliability(Z, tr, band)
    # signal amplitude and thermal noise per parcel in physical units (BOLD-like dR2*, 1/s), both over the analysis
    # band: noise SD = whitening scale x sqrt(band share of white noise); signal SD = noise-corrected in-band SD
    from .regions import band_pass as _bp
    Rk = np.asarray(inp["R"], float)[ok]
    sd_b = np.sqrt(np.asarray(inp["noise_var"], float)[ok] * (1 + pr["rho"] ** 2)) / Rk[:, 0, 0] * 1000.0
    bf = band_fraction(T, tr, band)
    noise_sd_band = sd_b * np.sqrt(bf * noise_scale)
    signal_sd_band = sd_b * np.sqrt(np.clip(_bp(Z, tr, band).var(1) - bf * noise_scale, 0, None))
    # floor for the noise-corrected integration measures (user, 2026-11-23): the lowest reliability at which the
    # correction factor 1/sqrt(reliability) is known within correction_error_max, given the in-band degrees of freedom
    # (two per frequency bin in the band); map-based measures (winner-take-all, dominance, node count) use every parcel
    rmin = reliability_floor(T, tr, band, float(P_["correction_error_max"]))
    reliable = rel >= rmin
    allp = np.ones(len(ids), bool)
    res = {"input": str(input_dir), "atlas": inp["atlas"], "n_parcels_total": n_total, "n_parcels_with_voxels": int(len(ids)),
           "n_frames": int(T), "tr_s": tr, "band_hz": list(band), "nuisance_rank": nuis, "respiratory_peak_hz": resp,
           # fewer frames than the common number: k, node count and other count-based measures are not comparable
           # with the other participants (user, 2026-11-23)
           "n_frames_common": G.get("n_volumes_common"),
           "comparable": bool(G.get("n_volumes_common") is None or T >= int(G.get("n_volumes_common"))),
           "settings": P_, "reliability_min": round(rmin, 3), "reliability_min_rule": f"correction factor within {P_['correction_error_max']}",
           "stage2": {"rounds": [{k_: v for k_, v in r.items() if k_ not in ("maps_removed", "tc_removed")} for r in rounds],
                      "n_removed": q, "stop": stop, "cap_reached": stop == "maximum number of rounds reached"},
           "denoising": den, "stage3": st3}
    comp3, wta = [], None
    if k3 >= 2:
        ica3 = pi.spatial_ica(Z, k3, seeds(100), n_jobs=n_jobs)
        sp3 = pi.spectral_measures(ica3["tc"], tr, band, resp_hz=resp or 0.3, high_hz=P_["high_hz"], noise_var=ica3["tc_noise_var"])
        s03 = pi.s0_coupling(ica3["maps"], Z, CW, pr["rho"], folds)
        labs = component_labels(ica3["maps"], ids, nets7, nets17)
        from .surface_plot import zscore
        Zm = zscore(ica3["maps"])
        win_full = np.argmax(Zm, axis=1)
        zwin = Zm[np.arange(len(win_full)), win_full]
        # dominance (user, 2026-11-23): strong = z1 >= z_min; clear = confidence >= dominance_q that the strongest
        # component truly exceeds the second; dominated = strong and clear (the winner); shared = strong, not clear;
        # weak = not strong
        dom = dominance(Z, ica3["maps"], ica3["tc"])
        strong = dom["z1"] >= P_["z_min"]
        clear = dom["confidence"] >= P_["dominance_q"]
        category = np.where(~strong, "weak", np.where(clear, "dominated", "shared"))
        win = np.where(strong & clear, win_full, -1)
        for j in range(k3):
            comp3.append({"component": j, "iq": round(float(ica3["iq"][j]), 3), **labs[j],
                          "band_share": round(float(sp3["band_share"][j]), 3), "high_share": round(float(sp3["high_share"][j]), 3),
                          "peak_hz": round(float(sp3["peak_hz"][j]), 3), "peak_ratio": round(float(sp3["peak_ratio"][j]), 2),
                          "eta": None if not np.isfinite(s03["eta"][j]) else round(float(s03["eta"][j]), 3),
                          "meets_nonneural_criteria": bool(rule(sp3, s03)[j]),
                          "n_parcels_won": int((win == j).sum())})
        wta = {"win": win, "win_full": win_full, "zwin": zwin, "maps": ica3["maps"], "tc": ica3["tc"], "tc_noise_var": ica3["tc_noise_var"],
               "dom_pct": dom["confidence"], "margin": dom["margin"], "category": category}
    res["stage3"]["components"] = comp3

    # ---- measures
    win_r = wta["win"] if wta else -np.ones(len(ids), int)            # map-based: all parcels with voxels
    nc = node_count(ids, win_r, anat["adjacent_pairs"])
    n_win = int(len(set(win_r[win_r >= 0].tolist())))
    metrics = {"reliable_parcels": int(reliable.sum()), "reliable_coverage_pct": round(100.0 * reliable.sum() / n_total, 2),
               "k_stage3": k3, "node_count": nc["node_count"], "components_winning": n_win,
               "patches_per_component": round(nc["node_count"] / n_win, 3) if n_win else None,
               "single_parcel_patches": int(sum(1 for s in nc["patch_sizes"] if s == 1)),
               "undominated_pct": round(100.0 * float((wta["category"] == "shared").mean()), 2) if wta else None,
               "weak_pct": round(100.0 * float((wta["category"] == "weak").mean()), 2) if wta else None,
               "dominated_pct": round(100.0 * float((wta["category"] == "dominated").mean()), 2) if wta else None,
               "dominance_confidence_median": round(float(np.nanmedian(wta["dom_pct"])), 3) if wta else None,
               "reliability_floor": round(rmin, 3),
               "coverage_at_0_5_pct": round(100.0 * float((rel >= 0.5).sum()) / n_total, 2),
               "n_removed_stage2": q,
               "signal_sd_band_median": round(float(np.median(signal_sd_band)), 5),
               "noise_sd_band_median": round(float(np.median(noise_sd_band)), 5)}
    res["metrics"] = metrics
    rng = np.random.default_rng(seed_root + 31)
    min_shift = max(1, int(round(float(G.get("min_shift_s", 30.0)) / tr)))
    nv = np.full(len(ids), noise_scale)
    sets = {}
    if wta:
        dd = np.where(wta["win"] >= 0, wta["win"], -1)
        for name, mask, ddlab in (("main: reliable parcels", reliable, dd), ("A: all parcels with voxels", np.ones(len(ids), bool), dd),
                                  ("B: reliable, undominated assigned", reliable, wta["win_full"])):
            idx = np.flatnonzero(mask)
            parts = {"Yeo-7": None if nets7 is None else nets7[idx], "Yeo-17": None if nets17 is None else nets17[idx],
                     "data-driven": np.asarray(ddlab)[idx].astype(object)}
            null = None
            if name.startswith("main") or name.startswith("A"):
                null = pr_null(Z[idx], nv[idx], tr, band, int(P_["n_null"]), min_shift, rng)
                if name.startswith("main"):
                    main_null = null
            else:
                null = main_null                                       # same parcels as the main set
            sets[name] = {"idx": idx, **integration(Z[idx], nv[idx], tr, band, parts, null, float(P_["edge_fdr_q"]))}
    res["integration"] = {name: {k_: v for k_, v in s.items() if k_ not in ("W", "pc_map", "idx")} for name, s in sets.items()}
    for name in list(sets):                                            # the null's edge arrays are large: not kept
        sets[name].pop("edge_null", None)

    # ---- outputs
    def num(v):
        return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4)
    rows = []
    for name, s in res["integration"].items():
        base = {"set": name, "n_parcels": s["n_parcels"], "n_with_signal": s["n_with_signal"], "pr": num(s["pr"]),
                "pr_resid": num(s["pr_resid"]), "pr_resid_null_median": num(s.get("pr_resid_null_median")), "global_share": num(s["global_share"]),
                "edge_density": num(s.get("edge_density")), "nodes_without_edges": s.get("nodes_without_edges")}
        for part in ("Yeo-7", "Yeo-17", "data-driven"):
            base[f"pc_{part}"] = num(s["pc"].get(part))
            base[f"seg_{part}"] = num(s["seg"].get(part))
        rows.append(base)
    if rows:
        io.write_tsv(out / "measures.tsv", {k_: [r[k_] for r in rows] for k_ in rows[0]})
    main = sets.get("main: reliable parcels")
    pcmap = {}
    if main:
        for part, v in main["pc_map"].items():
            full = np.full(len(ids), np.nan)
            full[main["idx"]] = v
            pcmap[part] = full
    io.write_tsv(out / "parcels.tsv", {"parcel": ids.tolist(), "yeo7": list(nets7) if nets7 is not None else [""] * len(ids),
                                       "yeo17": list(nets17) if nets17 is not None else [""] * len(ids),
                                       "reliability": [num(v) for v in rel], "reliable": reliable.astype(int).tolist(),
                                       "signal_sd_band": [round(float(v), 5) for v in signal_sd_band],
                                       "noise_sd_band": [round(float(v), 5) for v in noise_sd_band],
                                       "winner": (wta["win"].tolist() if wta else [-1] * len(ids)),
                                       "dominance": (wta["category"].tolist() if wta else [""] * len(ids)),
                                       "dominance_confidence": ([num(v) for v in wta["dom_pct"]] if wta else [None] * len(ids)),
                                       "strongest": (wta["win_full"].tolist() if wta else [-1] * len(ids)),
                                       **{f"pc_{p}": [num(v) for v in pcmap[p]] for p in pcmap}})
    np.savez_compressed(out / "stage3.npz", ids=ids, maps=wta["maps"] if wta else np.zeros((len(ids), 0)),
                        tc=wta["tc"] if wta else np.zeros((0, T)), winner=win_r, reliability=rel)
    io.write_json(out / "pipeline.json", {**res, **io.provenance(cfg, {})})
    io.write_text(out / "pipeline.md", render(res))
    try:
        from . import parcel_report
        parcel_report.build(out, res, ids=ids, rel=rel, reliable=reliable, rounds=rounds, wta=wta, sets=sets, pcmap=pcmap, map_mask=allp,
                            amp={"signal": signal_sd_band, "noise": noise_sd_band},
                            nets7=nets7, nets17=nets17, resources=resources, tr=tr, resp=resp, dim3=dim3)
    except Exception as exc:                                           # the report never blocks the results
        import traceback
        traceback.print_exc()
        print(f"report failed: {exc}")
    return res


def render(res: dict) -> str:
    m, s2, s3 = res["metrics"], res["stage2"], res["stage3"]
    f = lambda v: "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    L = ["# Parcel pipeline (icosahedral parcels; exploratory, route B unchanged)", ""]
    if not res.get("comparable", True):
        L += [f"**Not comparable: {res['n_frames']} frames, fewer than the common {res.get('n_frames_common')}; k, node count and other "
              "count-based measures cannot be compared with other participants.**", ""]
    L += [
         f"- {res['n_parcels_with_voxels']} of {res['n_parcels_total']} parcels with voxels; {res['n_frames']} frames; breathing peak "
         + (f"{res['respiratory_peak_hz']} Hz" if res['respiratory_peak_hz'] else "not found in the motion parameters (0.3 Hz assumed)"),
         f"- Stage 2 (non-neural removal): {s2['n_removed']} components removed over {len(s2['rounds'])} round(s); stopped: {s2['stop']}"
         + ("  **(cap reached: check the criteria)**" if s2["cap_reached"] else ""),
         f"- Stage 3 (networks): k = {s3['k']} (expected map fidelity >= {s3.get('fidelity_min')}: {s3.get('k_fidelity')}; thermal-noise threshold "
         f"{s3['k_mp']}{'; capped' if s3['k_capped'] else ''})",
         "", "## Metrics", "",
         f"- Reliable parcels (floor {res['reliability_min']}, {res['reliability_min_rule']}): {m['reliable_parcels']} ({m['reliable_coverage_pct']} % of "
         f"{res['n_parcels_total']}; {m.get('coverage_at_0_5_pct')} % at 0.5); the floor applies to the integration measures only",
         f"- Node count (contiguous same-winner patches): {m['node_count']}; components winning parcels: {m['components_winning']} of k = {m['k_stage3']}; "
         f"patches per component: {f(m['patches_per_component'])}; single-parcel patches: {m['single_parcel_patches']}",
         f"- Dominance, all parcels with voxels (strong: z >= {res['settings']['z_min']}; clear: confidence >= {res['settings']['dominance_q']} that "
         f"the strongest component exceeds the second): dominated {f(m.get('dominated_pct'))} %, undominated (shared) {f(m['undominated_pct'])} %, "
         f"weak {f(m.get('weak_pct'))} %; median confidence {f(m.get('dominance_confidence_median'))}",
         f"- In-band BOLD-like amplitude, median over parcels (1/s): signal {m.get('signal_sd_band_median')}, thermal noise {m.get('noise_sd_band_median')}", "",
         "## Integration (noise-corrected in-band correlations; graph measures on positive weights clipped at 1)", "",
         "Graph measures use correlations beyond the brain-wide pattern (first pattern removed) and only edges reliably above chance "
         "(circular-shift null per edge, false discovery rate).", "",
         "| set | parcels (with signal) | PR | PR without first (null median) | global share | edge density | PC Yeo-7 | PC Yeo-17 | PC data-driven | seg Yeo-7 | seg Yeo-17 | seg data-driven |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in res.get("integration", {}).items():
        L.append(f"| {name} | {s['n_parcels']} ({s['n_with_signal']}) | {f(s['pr'])} | {f(s['pr_resid'])} ({f(s.get('pr_resid_null_median'))}) | "
                 f"{f(s['global_share'])} | {f(s.get('edge_density'))} | {f(s['pc'].get('Yeo-7'))} | {f(s['pc'].get('Yeo-17'))} | {f(s['pc'].get('data-driven'))} | "
                 f"{f(s['seg'].get('Yeo-7'))} | {f(s['seg'].get('Yeo-17'))} | {f(s['seg'].get('data-driven'))} |")
    L += ["", "## Stage 2 rounds", "", "| round | df | k (map fidelity; leftover null) | removed | components (above 0.12 Hz / S0 coupling) |", "|---|---|---|---|---|"]
    for r in s2["rounds"]:
        cs = "; ".join(f"c{c['component']}{'*' if c['removed'] else ''} {c['high_share']:.2f}/{c['eta'] if c['eta'] is not None else '-'}" for c in r.get("components", []))
        L.append(f"| {r['round']} | {r['df']} | {r['k']} ({r.get('k_null')}) | {sum(c['removed'] for c in r.get('components', []))} | {cs} |")
    dn = res.get("denoising") or {}
    if dn:
        L += ["", f"Denoising: share of BOLD-like variance removed per parcel (10th / 50th / 90th percentile): in band {dn.get('inband_removed_p10_50_90')}, "
                  f"whole band {dn.get('total_removed_p10_50_90')}"]
    return "\n".join(L) + "\n"
