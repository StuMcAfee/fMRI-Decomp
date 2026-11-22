"""Spatial ICA on parcel BOLD-like signals (docs/PARCEL_ICA_PLAN.md; branch claude/parcel-ica, user 2026-11-18).

Inputs per participant: BOLD-like time courses b (parcels x frames, full band), S0-like time courses a, and each
parcel's thermal noise SD for both (from the leftover echo channels, route B). Steps:

1. whiten: each parcel's signal divided by its own noise SD, so thermal noise has unit variance everywhere;
2. dimensionality: eigenvalues of the whitened data against the Marchenko-Pastur edge for unit noise and against an
   empirical null (pure-noise parcel signals through the same steps);
3. spatial ICA (maps over parcels independent), robust restarts clustered into stable components (decompose.py);
4. measurements per component (classification becomes measurement, CLAUDE.md 7.1): spectrum (band, high-frequency
   and respiratory shares, narrow peaks), cross-fitted S0 coupling (a parcel-level eta), global correlation, map
   statistics;
5. removal of chosen components by one projector, and matching of the others to the Yeo-7 networks.
Exploratory: no route B measure changes unless the user adopts this after the validation gates.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- 1-2: whitening and dimensionality
def whiten(x: np.ndarray, noise_sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Demean each parcel and divide by its noise SD. Returns (z for the usable parcels, usable mask)."""
    x = np.asarray(x, np.float64)
    sd = np.asarray(noise_sd, np.float64)
    ok = np.isfinite(sd) & (sd > 0) & np.all(np.isfinite(x), axis=1)
    z = (x[ok] - x[ok].mean(1, keepdims=True)) / sd[ok, None]
    return z, ok


def eigenvalues(z: np.ndarray, df: int | None = None) -> np.ndarray:
    """Eigenvalues of z z' / df (descending, nonzero part from the frames x frames Gram matrix, float64)."""
    df = z.shape[1] if df is None else int(df)
    lam = np.linalg.eigvalsh(z.T @ z)[::-1] / df
    return lam[: min(z.shape)]


def mp_edge(n_parcels: int, df: int) -> float:
    """Upper Marchenko-Pastur edge for unit-variance white noise, n_parcels variables, df samples."""
    return float((1.0 + np.sqrt(n_parcels / df)) ** 2)


def dimensionality(z: np.ndarray, df: int, null_z: list | None = None, q: float = 0.95) -> dict:
    """Number of eigenvalues above the MP edge and, with null data (list of whitened pure-noise matrices of the
    same shape), above the null's index-wise q-quantile."""
    lam = eigenvalues(z, df)
    edge = mp_edge(z.shape[0], df)
    out = {"eigenvalues": lam, "mp_edge": edge, "k_mp": int(np.sum(lam > edge))}
    if null_z:
        # the null's largest eigenvalue (pure noise of the same shape through the same steps): index-wise comparison
        # over-counts, because each signal dimension pushes the noise eigenvalues below it up by one index
        top = np.array([eigenvalues(n, df)[0] for n in null_z])
        thr = float(np.quantile(top, q)) if len(top) > 1 else float(top[0])
        out["null_threshold"] = thr
        out["k_null"] = int(np.sum(lam > thr))
    return out


# ---------------------------------------------------------------- 3: spatial ICA
def spatial_ica(z: np.ndarray, k: int, seeds, n_jobs: int | None = 1) -> dict:
    """Spatial ICA with robust restarts (parcels are the samples).

    FastICA without its own whitening does not centre the data, and parcel maps have large means (the brain-wide
    pattern, positive network loadings), so each frame's mean over parcels is removed first; the centred data are
    reduced to k dimensions and whitened (decompose.pca_reduce), the restarts are clustered (cluster_restarts), the
    time courses are the least-squares fit of the centred data on the centred maps, and the full maps (means
    included) are the regression of the uncentred data on those time courses. Sign: positive skew of the map.
    Returns maps (parcels x k), time courses (k x frames), stability index iq, restarts per cluster, explained
    variance of the centred data."""
    from . import decompose
    zc = z - z.mean(0, keepdims=True)
    pca = decompose.pca_reduce(zc, int(k))
    runs = decompose.ica_restarts(pca["Y"], list(seeds), n_jobs=n_jobs)
    cl = decompose.cluster_restarts(runs, int(k))
    mc = pca["Y"] @ cl["W"].T
    skew = (mc ** 3).mean(0)
    mc = mc * np.where(skew < 0, -1.0, 1.0)
    pinv = np.linalg.pinv(mc)
    tc = pinv @ zc
    sd = np.maximum(tc.std(1), 1e-12)
    tc = tc / sd[:, None]
    noise_var = (pinv ** 2).sum(1) / sd ** 2                           # thermal noise variance in each unit-SD time course
    maps = np.linalg.lstsq(tc.T, z.T, rcond=None)[0].T
    return {"maps": maps, "maps_centred": mc, "tc": tc, "tc_noise_var": noise_var, "iq": cl["iq"], "n_runs": cl["n_runs"],
            "explained": pca["explained"]}


# ---------------------------------------------------------------- 4: measurements
def spectral_measures(tc: np.ndarray, tr: float, band=(0.01, 0.10), resp_hz: float | None = None,
                      resp_halfwidth: float = 0.03, high_hz: float = 0.12, noise_var=None) -> dict:
    """Per component: shares of the time course's signal power in the analysis band, above high_hz, and at the
    breathing frequency and its first harmonic (folded at Nyquist); the strongest narrow peak above high_hz (smoothed
    peak / median power there) and its frequency. With noise_var (thermal noise variance of each time course, from
    spatial_ica), the flat thermal-noise power is subtracted first, so a weak component's noise does not read as
    high-frequency signal; shares are then of the noise-debiased power (negative bins clipped at zero)."""
    tc = np.asarray(tc, np.float64)
    T = tc.shape[1]
    f = np.fft.rfftfreq(T, tr)
    P = np.abs(np.fft.rfft(tc - tc.mean(1, keepdims=True), axis=1)) ** 2
    if noise_var is not None:
        P = np.clip(P - T * np.asarray(noise_var, float)[:, None], 0, None)
    pos = f > 0
    tot = np.maximum(P[:, pos].sum(1), 1e-30)
    sh = lambda m: P[:, m & pos].sum(1) / tot
    out = {"band_share": sh((f >= band[0]) & (f <= band[1])), "high_share": sh(f > high_hz)}
    nyq = 0.5 / tr
    if resp_hz:
        h2 = 2 * resp_hz if 2 * resp_hz <= nyq else 2 * nyq - 2 * resp_hz       # first harmonic, folded
        m = (np.abs(f - resp_hz) <= resp_halfwidth) | (np.abs(f - h2) <= resp_halfwidth)
        out["resp_share"] = sh(m)
    hi = f > high_hz
    # narrow peaks against the local spectrum (running median over about 0.1 Hz of the raw periodogram), so the
    # falling 1/f-like slope does not make the lowest searched frequency look like a peak
    from scipy.ndimage import median_filter, uniform_filter1d
    raw = np.abs(np.fft.rfft(tc - tc.mean(1, keepdims=True), axis=1)) ** 2
    w = max(5, int(round(0.1 / (f[1] - f[0]))) | 1)
    base = median_filter(raw, size=(1, w), mode="nearest")
    Ps = uniform_filter1d(raw, 3, axis=1)
    ratio = (Ps / np.maximum(base, 1e-30))[:, hi]
    j = np.argmax(ratio, axis=1)
    out["peak_ratio"] = ratio[np.arange(len(j)), j]
    out["peak_hz"] = f[hi][j]
    return out


def prepare(c0, c1, R, noise_var, r=None, frames=None) -> dict:
    """Noise-whitened inputs from route B's parcel channels (region_timecourses.npz or parcel_channels.npz).

    Channels c0, c1 (parcels x frames) carry thermal noise of variance noise_var per parcel, independent between the
    two channels; cw = c / sqrt(noise_var) has unit noise. The BOLD-like signal b = (c0 - R01 a) / R00 (a = c1 / R11)
    whitened to unit noise is zb = (cw0 - rho cw1) / sqrt(1 + rho^2), rho = R01 / R11 (zb is proportional to b in
    every parcel). The same combination of the whitened leftover channels r (pure noise plus model violations) is the
    empirical null. Parcels without finite channels or noise are dropped (mask `ok`)."""
    c0, c1 = np.asarray(c0, np.float64), np.asarray(c1, np.float64)
    R = np.asarray(R, np.float64)
    nv = np.asarray(noise_var, np.float64)
    if frames is not None:
        c0, c1 = c0[:, frames], c1[:, frames]
    ok = (np.isfinite(nv) & (nv > 0) & np.all(np.isfinite(c0), 1) & np.all(np.isfinite(c1), 1)
          & np.all(np.isfinite(R[:, :2, :2].reshape(len(R), -1)), 1) & (np.abs(c0).sum(1) > 0))
    sd = np.sqrt(nv[ok])[:, None]
    cw = np.stack([c0[ok] / sd, c1[ok] / sd], 1)
    cw = cw - cw.mean(2, keepdims=True)
    rho = R[ok, 0, 1] / R[ok, 1, 1]
    zb = (cw[:, 0] - rho[:, None] * cw[:, 1]) / np.sqrt(1 + rho ** 2)[:, None]
    out = {"zb": zb, "cw": cw, "rho": rho, "ok": ok}
    if r is not None:
        r = np.asarray(r, np.float64)
        if frames is not None:
            r = r[..., frames]
        rw = np.nan_to_num(r[ok]) / sd[:, :, None]
        rw = rw - rw.mean(2, keepdims=True)
        present = np.abs(rw).sum(2) > 0                                   # leftover channels that exist (4 echoes: 2)
        out["null_channels"] = [rw[:, j] for j in range(rw.shape[1]) if present[:, j].mean() > 0.9]
    return out


def s0_coupling(maps: np.ndarray, zb: np.ndarray, cw: np.ndarray, rho: np.ndarray, folds: np.ndarray) -> dict:
    # zb: the data the maps were estimated from (whitened b, or the whitened c0 channel for the c0 input); the
    # held-out time courses must come from the same data as the maps
    """Cross-fitted S0 coupling per component, the parcel-level analogue of eta (CLAUDE.md 7.1).

    For each fold of parcels, the component time courses are estimated from the other parcels (least squares of zb
    on their maps), so they contain none of the held-out parcels' noise; on the held-out parcels the whitened
    channel betas on the unit-norm time course, m0 and m1, have unit noise variance, and the physical energies are
    debiased exactly:  E_S0 = sum (1 + rho^2)(m1^2 - 1),  E_R2 = sum [(m0 - rho m1)^2 - (1 + rho^2)],
    eta = E_S0 / (E_S0 + E_R2): about 0 for a pure dR2* component, 1 for pure dS0."""
    folds = np.asarray(folds)
    k = maps.shape[1]
    Eb, Ea = np.zeros(k), np.zeros(k)
    for g in np.unique(folds):
        fit, te = folds != g, folds == g
        t = np.linalg.lstsq(maps[fit], zb[fit], rcond=None)[0]
        t = t - t.mean(1, keepdims=True)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        m0, m1 = cw[te, 0] @ t.T, cw[te, 1] @ t.T                      # (parcels in fold, k)
        w = (1 + rho[te] ** 2)[:, None]
        Ea += (w * (m1 ** 2 - 1)).sum(0)
        Eb += ((m0 - rho[te, None] * m1) ** 2 - w).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tot = np.clip(Ea, 0, None) + np.clip(Eb, 0, None)
        eta = np.where(tot > 0, np.clip(Ea, 0, None) / tot, np.nan)
    return {"E_R2": Eb, "E_S0": Ea, "eta": eta}


def global_correlation(tc: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Correlation of each component time course with the mean whitened BOLD-like signal over parcels."""
    g = z.mean(0)
    g = (g - g.mean()) / max(g.std(), 1e-12)
    t = (tc - tc.mean(1, keepdims=True)) / np.maximum(tc.std(1, keepdims=True), 1e-12)
    return t @ g / len(g)


def map_measures(maps: np.ndarray, covariates: dict | None = None) -> dict:
    """Map kurtosis (sparse, focal maps are heavy-tailed) and correlations with parcel covariates (e.g. CSF
    fraction)."""
    m = (maps - maps.mean(0)) / np.maximum(maps.std(0), 1e-12)
    out = {"kurtosis": (m ** 4).mean(0) - 3.0}
    for name, v in (covariates or {}).items():
        v = np.asarray(v, float)
        vz = (v - v.mean()) / max(v.std(), 1e-12)
        out[f"corr_{name}"] = vz @ m / len(vz)
    return out


# ---------------------------------------------------------------- 5: removal and network matching
def remove_components(x: np.ndarray, tc_remove: np.ndarray) -> np.ndarray:
    """Project the given component time courses (rows) out of every parcel's signal (one projector, like the
    motion regressors); the caller reduces the temporal degrees of freedom by their number."""
    x = np.asarray(x, np.float64)
    if tc_remove is None or len(tc_remove) == 0:
        return x.copy()
    Tm = np.asarray(tc_remove, np.float64).T
    Tm = Tm - Tm.mean(0)
    Qm, _ = np.linalg.qr(Tm)
    return x - (x @ Qm) @ Qm.T


def match_networks(maps: np.ndarray, networks: list, names=("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")) -> dict:
    """Match components to network templates (indicator maps over parcels) by |correlation| (Hungarian). Returns
    for each network the matched component and its correlation."""
    from scipy.optimize import linear_sum_assignment
    nets = np.asarray(networks)
    Tm = np.column_stack([(nets == n).astype(float) for n in names])
    a = (Tm - Tm.mean(0)) / np.maximum(Tm.std(0), 1e-12)
    b = (maps - maps.mean(0)) / np.maximum(maps.std(0), 1e-12)
    C = a.T @ b / len(nets)
    r, c = linear_sum_assignment(-np.abs(C))
    return {names[i]: {"component": int(j), "r": float(C[i, j])} for i, j in zip(r, c)}


# ---------------------------------------------------------------- real data (gate P2)
def load_inputs(folder) -> dict:
    """Parcel channels for the ICA from either source, as (cortical parcels x frames) arrays with global parcel ids:
    - route B output (region_timecourses.npz + route_b.json; volumetric or surface-based Schaefer parcels): the
      cortical regions;
    - icosahedral export (parcel_channels.npz + export.json, outputs/route_b_surf/<run>/ico1002_7): all 1876 parcels
      in the connectivity model's order (parcels without voxels are dropped later), networks from the icosahedral
      table (Yeo-7 by vertex majority)."""
    import json
    from pathlib import Path
    F = Path(folder)
    if (F / "parcel_channels.npz").exists():
        npz = np.load(F / "parcel_channels.npz")
        ex = json.loads((F / "export.json").read_text())
        n = npz["c0"].shape[0]
        ids = np.arange(1, n + 1)
        table = {}
        tf = Path(ex.get("anat_dir", "")) / "atlas_table_ico1002_7.json"
        if tf.exists():
            table = {int(k): v for k, v in json.loads(tf.read_text()).items()}
        nets = np.array([str(table.get(i, {}).get("network", "")) for i in ids])
        hemi = np.array([("L" if i <= n // 2 else "R") for i in ids])
        return {"source": "icosahedral export", "atlas": ex.get("atlas", "ico1002_7"), "ids": ids, "networks": nets, "hemisphere": hemi,
                "c0": npz["c0"], "c1": npz["c1"], "R": npz["R"], "noise_var": npz["noise_var"], "r": npz["r"],
                "frames": npz["frames"] if "frames" in npz.files else None, "nuisance_rank": int(npz["nuisance_rank"]),
                "frontend_dir": ex.get("frontend_dir"), "band_hz": (0.01, 0.1), "table_found": bool(table)}
    summ_rb = json.loads((F / "route_b.json").read_text())
    npz = np.load(F / "region_timecourses.npz")
    ids = np.asarray(summ_rb["cortex_ids"], int)
    rows = {r["id"]: r for r in summ_rb["regions"]}
    nets = np.array([str(rows[i].get("network")) for i in ids])
    hemi = np.array(["L" if "_LH_" in str(rows[i].get("name")) else "R" if "_RH_" in str(rows[i].get("name")) else "" for i in ids])
    R = np.asarray(npz["R"], float)
    c1 = npz["c1"] if "c1" in npz.files else npz["a"] * R[:, 1, 1, None]
    frames = npz["frames"] if "frames" in npz.files else None
    tl = summ_rb.get("timeline") or {}
    if frames is None and tl.get("n_used"):                            # older outputs: rebuild route B's frames
        start = int((tl.get("longest_stretch") or [0])[0])
        frames = np.arange(start, start + int(tl["n_used"]))
    sel = ids - 1
    return {"source": "route B", "atlas": summ_rb.get("atlas", "200_7"), "ids": ids, "networks": nets, "hemisphere": hemi,
            "c0": npz["c0"][sel], "c1": c1[sel], "R": R[sel], "noise_var": npz["noise_var"][sel],
            "r": npz["r"][sel] if "r" in npz.files else None, "frames": frames,
            "nuisance_rank": int(summ_rb.get("nuisance_rank") or 0), "frontend_dir": summ_rb.get("frontend_dir"),
            "band_hz": tuple(summ_rb.get("band_hz") or (0.01, 0.1))}


def run_dir(route_b_dir, cfg: dict, out_dir, k: int | None = None, n_seeds: int = 30, input_channel: str = "b",
            n_jobs: int | None = None) -> dict:
    """Parcel ICA on one participant's route B output (region_timecourses.npz, route_b.json): cortical parcels, the
    frames route B's coupling uses, spatial ICA at k (default: the MP count), measurements per component, the
    candidate removal rule (flags only; nothing is removed from route B), Yeo-7 matching. input_channel "b" (the
    agreed design) or "c0" (OC-like channel, descriptive comparison: it carries dS0 as well). Writes components.tsv,
    parcel_ica.json / .md, maps.npy, timecourses.npy and parcel_ica.png under out_dir (research storage only)."""
    import json
    from pathlib import Path

    from . import io
    rb, out = Path(route_b_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inp = load_inputs(rb)
    pr = prepare(inp["c0"], inp["c1"], inp["R"], inp["noise_var"], inp["r"], inp["frames"])
    nets = inp["networks"][pr["ok"]]
    ids = inp["ids"]
    hemi = inp["hemisphere"][pr["ok"]]
    summ_rb = {"frontend_dir": inp["frontend_dir"], "band_hz": inp["band_hz"]}
    nuis = int(inp["nuisance_rank"])
    z = pr["zb"] if input_channel == "b" else pr["cw"][:, 0]
    T = z.shape[1]
    df = T - nuis - 1
    nulls = None
    if pr.get("null_channels"):
        nulls = pr["null_channels"] if input_channel == "c0" else [n for n in pr["null_channels"]]
    dim = dimensionality(z, df, nulls)
    # without a given k, the MP count, capped (with 1876 parcels the count can reach hundreds, where the restart
    # clustering becomes slow and the components mostly split networks; P2 first look)
    k_max = int(cfg.get("parcel_ica", {}).get("k_max", 100))
    kk = int(k or min(dim["k_mp"], k_max))
    tr = float(cfg["acquisition"]["tr_s"])
    resp = None
    try:
        fs = json.loads((Path(summ_rb["frontend_dir"]) / "summary.json").read_text())
        resp = (fs.get("nuisance", {}).get("filter", {}).get("respiratory_peak") or {}).get("freq_hz")
    except Exception:
        pass
    res = {"route_b_dir": str(rb), "atlas": inp["atlas"], "source": inp["source"], "input": input_channel, "n_parcels": int(z.shape[0]), "n_frames": int(T), "df": int(df),
           "k_mp": dim["k_mp"], "k_null": dim.get("k_null"), "mp_edge": dim["mp_edge"], "k_used": kk,
           "k_capped": bool(not k and dim["k_mp"] > k_max),
           "eigenvalues_top": [round(float(x), 3) for x in dim["eigenvalues"][:40]], "respiratory_peak_hz": resp,
           "rule": {"high_share": 0.5, "eta": 0.5}}
    if kk < 2:
        res["note"] = "fewer than 2 dimensions above the MP edge: no ICA"
        io.write_json(out / "parcel_ica.json", {**res, **io.provenance(cfg, {})})
        return res
    # restart seeds from the project seed (CLAUDE.md section 12; stage key 41 = parcel ICA), logged in the output
    seeds = [int(x) for x in np.random.SeedSequence(int(cfg.get("project_seed", 0)), spawn_key=(41,)).generate_state(n_seeds)]
    res["seeds"] = seeds
    ica = spatial_ica(z, kk, seeds, n_jobs=n_jobs)
    sp = spectral_measures(ica["tc"], tr, band=tuple(summ_rb.get("band_hz") or (0.01, 0.1)), resp_hz=resp or 0.3,
                           noise_var=ica["tc_noise_var"])
    # cross-fitting folds: the two hemispheres (neighbouring parcels can share noise through motion-correction
    # interpolation; the hemispheres do not), random halves only if hemispheres are unknown
    if set(np.unique(hemi)) == {"L", "R"}:
        folds = (hemi == "R").astype(int)
    else:
        folds = np.random.default_rng(int(cfg.get("project_seed", 0)) + 23).permutation(np.arange(z.shape[0]) % 2)
    res["folds"] = "hemispheres" if set(np.unique(hemi)) == {"L", "R"} else "random halves"
    s0 = s0_coupling(ica["maps"], z, pr["cw"], pr["rho"], folds)          # time courses from the ICA input itself
    gc = global_correlation(ica["tc"], pr["zb"])
    mm = map_measures(ica["maps"])
    nm = match_networks(ica["maps"], list(nets))
    best_net = {v["component"]: (n, v["r"]) for n, v in nm.items()}
    comps = []
    for j in range(kk):
        flag = bool(sp["high_share"][j] > 0.5 or (np.isfinite(s0["eta"][j]) and s0["eta"][j] > 0.5))
        comps.append({"component": j, "iq": round(float(ica["iq"][j]), 3), "n_runs": int(ica["n_runs"][j]),
                      "band_share": round(float(sp["band_share"][j]), 3), "high_share": round(float(sp["high_share"][j]), 3),
                      "resp_share": round(float(sp["resp_share"][j]), 3), "peak_hz": round(float(sp["peak_hz"][j]), 3),
                      "peak_ratio": round(float(sp["peak_ratio"][j]), 2),
                      "eta": None if not np.isfinite(s0["eta"][j]) else round(float(s0["eta"][j]), 3),
                      "global_r": round(float(gc[j]), 3), "map_kurtosis": round(float(mm["kurtosis"][j]), 2),
                      "network": best_net.get(j, ("", None))[0], "network_r": None if j not in best_net else round(best_net[j][1], 3),
                      "flag": flag})
    res["components"] = comps
    res["n_flagged"] = int(sum(c["flag"] for c in comps))
    res["network_match"] = nm
    res["explained"] = round(float(ica["explained"]), 4)
    np.save(out / "maps.npy", ica["maps"].astype(np.float32))
    np.save(out / "timecourses.npy", ica["tc"].astype(np.float32))
    io.write_tsv(out / "components.tsv", {k_: [c[k_] for c in comps] for k_ in comps[0]})
    io.write_json(out / "parcel_ica.json", {**res, "cortex_ids_used": ids[pr["ok"]].tolist(), "parcel_networks_used": nets.tolist(),
                                            **io.provenance(cfg, {})})
    io.write_text(out / "parcel_ica.md", render(res))
    try:
        figure(res, ica, dim, tr, out / "parcel_ica.png", resp)
    except Exception as exc:                                           # a figure never blocks the result
        print(f"figure failed: {exc}")
    resources = Path(cfg.get("surface", {}).get("resources") or Path(__file__).resolve().parents[2] / "outputs" / "atlases_fs32k")
    if (resources / "tpl-fs32k_hemi-L_inflated.surf.gii").exists():
        try:
            surface_figures(out, resources)
        except Exception as exc:
            print(f"surface figures failed: {exc}")
    else:
        print(f"surface figures skipped: no inflated surfaces in {resources} (run scripts/fetch_fs32k_resources.sh)")
    return res


def surface_figures(out_dir, resources, z_min: float = 1.0) -> dict:
    """Weight maps and the winner-take-all map on the fs_LR 32k inflated template for one parcel ICA output folder
    (maps.npy, parcel_ica.json; parcel networks and the atlas from the route B folder it names). Writes
    surface_weights_p<i>.png, surface_wta.png and surface_wta.json."""
    import json
    from pathlib import Path

    from . import io, surface_plot
    out = Path(out_dir)
    pj = json.loads((out / "parcel_ica.json").read_text())
    maps = np.load(out / "maps.npy")
    ids = [int(i) for i in pj["cortex_ids_used"]]
    rows = {}
    if pj.get("parcel_networks_used") is not None:
        atlas, nets = pj.get("atlas", "200_7"), list(pj["parcel_networks_used"])
    else:                                                              # outputs written before 2026-11-18 (route B only)
        rb = json.loads((Path(pj["route_b_dir"]) / "route_b.json").read_text())
        atlas = rb.get("atlas", "200_7")
        rows = {r["id"]: r for r in rb["regions"]}
        nets = [str(rows[i].get("network")) for i in ids]
    if atlas == "200_7" and not rows and (Path(pj["route_b_dir"]) / "route_b.json").exists():
        rows = {r["id"]: r for r in json.loads((Path(pj["route_b_dir"]) / "route_b.json").read_text())["regions"]}
    res = surface_plot.load_resources(resources, atlas)
    if atlas == "200_7":                                               # route B ids must name the same parcels as the dlabel
        bad = [i for i in ids if rows.get(i, {}).get("name") and res["schaefer_names"].get(i) != rows[i]["name"]]
        if bad:
            raise ValueError(f"route B parcel names differ from the surface atlas for ids {bad[:5]}")
    comps = pj["components"]
    titles = [f"c{c['component']}  {c['network'] or ''}{' (r ' + str(c['network_r']) + ')' if c['network_r'] is not None else ''}\n"
              f">0.12 Hz {c['high_share']:.2f}  S0 {c['eta'] if c['eta'] is not None else '-'}{chr(10) + 'FLAGGED' if c['flag'] else ''}" for c in comps]
    files = surface_plot.weight_maps(maps, ids, res, out / "surface_weights", titles)
    tag = f"{pj['input']} input, k = {pj['k_used']}"
    wta = surface_plot.winner_take_all(maps, ids, nets, res, out / "surface_wta.png", [c["flag"] for c in comps], z_min,
                                       f"Winner-take-all over {len(comps)} components ({tag}; parcels shown on the fs_LR 32k template by identity)")
    io.write_json(out / "surface_wta.json", {"atlas": atlas, "z_min": z_min, "parcel_ids": ids, **wta})
    return {"weights": files, "wta": str(out / "surface_wta.png")}


def render(res: dict) -> str:
    L = [f"# Parcel ICA ({res['input']} input; exploratory, nothing removed from route B)", "",
         f"- {res['n_parcels']} cortical parcels, {res['n_frames']} frames (df {res['df']}); breathing peak "
         f"{res['respiratory_peak_hz'] or 'not found (0.3 Hz assumed)'} Hz",
         f"- Dimensions above the MP edge ({res['mp_edge']:.2f}): {res['k_mp']}; above the leftover-channel null: {res['k_null']}; k used {res['k_used']}",
         f"- Candidate rule (P1): signal share above 0.12 Hz > 0.5 or S0 coupling > 0.5: {res.get('n_flagged', 0)} flagged", ""]
    if res.get("components"):
        L += ["| comp | stability Iq | in band | above 0.12 Hz | at breathing | peak (Hz, ratio) | S0 coupling | global r | network (r) | flag |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in res["components"]:
            L.append(f"| {c['component']} | {c['iq']:.2f} | {c['band_share']:.2f} | {c['high_share']:.2f} | {c['resp_share']:.2f} | "
                     f"{c['peak_hz']:.3f}, {c['peak_ratio']:.1f} | {c['eta'] if c['eta'] is not None else '-'} | {c['global_r']:.2f} | "
                     f"{c['network'] or '-'} ({c['network_r'] if c['network_r'] is not None else '-'}) | {'yes' if c['flag'] else ''} |")
    return "\n".join(L) + "\n"


def figure(res, ica, dim, tr, path, resp=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from scipy.ndimage import uniform_filter1d
    k = ica["tc"].shape[0]
    fig = plt.figure(figsize=(12, 3 + 0.9 * ((k + 2) // 3)), constrained_layout=True)
    gs = fig.add_gridspec(1 + (k + 2) // 3, 3)
    a = fig.add_subplot(gs[0, :])
    lam = dim["eigenvalues"][:60]
    a.plot(np.arange(1, lam.size + 1), lam, "o-", color="#2a78d6", ms=4, lw=1.5, label="whitened data")
    a.axhline(dim["mp_edge"], color="#555", ls=":", lw=1, label="MP edge (unit noise)")
    if dim.get("null_threshold") is not None:
        a.axhline(dim["null_threshold"], color="#eb6834", ls="--", lw=1, label="leftover-channel null")
    a.set_yscale("log")
    a.set_xlabel("index")
    a.set_ylabel("eigenvalue")
    a.set_title(f"eigenvalues ({res['input']} input): {res['k_mp']} above the MP edge", fontsize=10)
    a.legend(fontsize=8, frameon=False)
    T = ica["tc"].shape[1]
    f = np.fft.rfftfreq(T, tr)
    P = np.abs(np.fft.rfft(ica["tc"] - ica["tc"].mean(1, keepdims=True), axis=1)) ** 2 / T
    for j in range(k):
        ax = fig.add_subplot(gs[1 + j // 3, j % 3])
        ax.plot(f[1:], uniform_filter1d(P[j, 1:], 5), color="#2a78d6", lw=1)
        ax.axhline(ica["tc_noise_var"][j], color="#9a9a92", lw=0.8, ls=":")
        ax.axvspan(0.01, 0.1, color="#1baf7a", alpha=0.12, lw=0)
        if resp:
            ax.axvline(resp, color="#eb6834", lw=0.8, ls="--")
        c = res["components"][j]
        ax.set_title(f"c{j}: {c['network'] or '-'} | >0.12 Hz {c['high_share']:.2f} | S0 {c['eta'] if c['eta'] is not None else '-'}"
                     f"{' | FLAG' if c['flag'] else ''}", fontsize=8)
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(labelsize=7)
        if j // 3 == (k - 1) // 3:
            ax.set_xlabel("Hz", fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)
