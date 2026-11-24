"""Injection test of the parcel pipeline (docs/PARCEL_ICA_PLAN.md section 9; approved by the user 2026-11-24, with the
twice-network strength as the primary level).

Known signals are added to the RAW echoes in every voxel of chosen icosahedral parcels (EPI labels from
outputs/anatomy_surf/<run>), then the whole chain is rerun (mbme_tedana, front end, icosahedral export, parcel pipeline
with whole and high-frequency-only removal). Sources:
  N1-N3  neural-like: about 40 parcels each inside one Yeo-7 network (Default, Cont, SomMot), grown as two contiguous
         patches (one per hemisphere); dR2* only; band-limited 0.01-0.1 Hz; dR2* SD 0.2 1/s (resting strength);
         N2 (Cont) shares about half its parcels with B.
  B      breathing-like: the participant's breathing frequency (from the motion parameters) with slow amplitude and
         rate modulation, plus its first harmonic, plus about 30 % of its power in 0.01-0.1 Hz; dR2* with a dS0 part
         set for an expected S0 coupling of about 0.4; speckled map (random 30 % of parcels, random weights and signs).
  C      cardiac-alias-like: 1.2 Hz with heart-rate jitter sampled at the TR (folds to about 0.133 Hz at 0.75 s);
         mostly dS0 (expected S0 coupling about 0.7); sparse map (10 % of parcels).
Levels: x2 (primary: artifacts at twice the networks' dR2* SD) and x1 (equal).

Recovery uses that the region averaging and the echo split are linear: source s changes parcel p's BOLD-like signal
by b_s w_sp u_s(t) (1/s). The share of that truth removed by a run is its energy in the span of the run's removed time
courses (denoised.npz), over the whole band and over 0.01-0.1 Hz.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NETS = (("N1", "Default"), ("N2", "Cont"), ("N3", "SomMot"))
LEVELS = {"x2": 2.0, "x1": 1.0}
NET_SD = 0.2                     # dR2* SD of the neural-like networks (1/s), the `weak` (resting) level


def a_for_eta(b_sd: float, eta: float, te_ms, ratios=None, t2s_ms: float = 45.0) -> float:
    """dS0/S0 SD giving an expected S0 coupling eta for a dR2* SD b_sd, from the whitened channel geometry
    (E_S0 = (R01^2 + R11^2) a^2, E_R2 = (R00 b / 1000)^2, CLAUDE.md 7.1) at a typical T2*."""
    from .frontend import design
    te = np.asarray(te_ms, float)
    r = np.ones_like(te) if ratios is None else np.asarray(ratios, float)
    m = np.exp(-te / t2s_ms) / r
    _, Rm = np.linalg.qr(m[:, None] * design(te))
    R00, R01, R11 = abs(Rm[0, 0]), Rm[0, 1], abs(Rm[1, 1])
    return float(R00 * b_sd / 1000.0 * np.sqrt(eta / (1 - eta)) / np.sqrt(R01 ** 2 + R11 ** 2))


def _grow(seed: int, pool: set, adj: dict, n: int, rng) -> list:
    """A contiguous patch of up to n parcels from the pool, grown from seed through adjacency."""
    patch, front = [seed], [seed]
    while front and len(patch) < n:
        cur = front.pop(int(rng.integers(len(front))))
        for nb in rng.permutation(sorted(adj.get(cur, ()))):
            nb = int(nb)
            if nb in pool and nb not in patch:
                patch.append(nb)
                front.append(nb)
                if len(patch) >= n:
                    break
    return patch


def build(labels3d: np.ndarray, table: dict, T: int, tr: float, te_ms, ratios, level: str, resp_hz: float, rng,
          n_net: int = 40, min_voxels: int = 3) -> dict:
    """Sources, voxel weights W3 (X, Y, Z, n), unit-SD time courses U (T, n), amplitudes a (dS0/S0) and b (1/s)."""
    from . import regions
    from .parcel_sim import band_noise, oscillation
    ids = [int(i) for i in np.unique(labels3d) if i > 0]
    nvox = {i: int((labels3d == i).sum()) for i in ids}
    ids = [i for i in ids if nvox[i] >= min_voxels]
    A = regions.adjacency(labels3d, ids)
    adj = {ids[i]: {ids[j] for j in np.flatnonzero(A[i])} for i in range(len(ids))}
    net_of = {i: str(table.get(i, {}).get("network", "")) for i in ids}
    left = lambda i: i <= 938
    src, used = [], set()
    for name, net in NETS:
        parcels = []
        for hemi in (True, False):
            pool = {i for i in ids if net_of[i] == net and left(i) == hemi and i not in used}
            if not pool:
                continue
            seed = int(rng.choice(sorted(pool)))
            parcels += _grow(seed, pool, adj, n_net // 2, rng)
        used |= set(parcels)
        src.append({"id": name, "kind": "neural", "network": net, "parcels": sorted(parcels),
                    "weights": [1.0] * len(parcels)})
    # breathing: 30 % of parcels, about half of N2's parcels included
    n2 = next(s for s in src if s["id"] == "N2")["parcels"]
    half = list(rng.choice(n2, len(n2) // 2, replace=False))
    rest = [i for i in ids if i not in half]
    nb = int(0.3 * len(ids))
    bpar = sorted(set(half) | set(int(x) for x in rng.choice(rest, max(nb - len(half), 0), replace=False)))
    src.append({"id": "B", "kind": "breathing", "parcels": bpar,
                "weights": list(np.round(rng.uniform(0.3, 1.0, len(bpar)) * rng.choice([-1, 1], len(bpar)), 4))})
    cpar = sorted(int(x) for x in rng.choice(ids, int(0.1 * len(ids)), replace=False))
    src.append({"id": "C", "kind": "cardiac", "parcels": cpar, "weights": list(np.round(rng.uniform(0.5, 1.0, len(cpar)), 4))})
    # time courses (unit SD)
    U = np.zeros((T, len(src)))
    U[:, :3] = band_noise(T, tr, (0.01, 0.1), rng, 3).T
    osc = oscillation(T, tr, resp_hz, rng, 0.02, 0.5)
    t = np.arange(T) * tr
    harm = np.sin(2 * np.pi * 2 * resp_hz * t + rng.uniform(0, 2 * np.pi))     # sampled at the TR: folds as in the scanner
    slow = band_noise(T, tr, (0.01, 0.1), rng)[0]
    hf = osc + 0.5 * harm
    hf = (hf - hf.mean()) / hf.std()
    U[:, 3] = np.sqrt(0.7) * hf + np.sqrt(0.3) * slow
    U[:, 4] = oscillation(T, tr, 1.2, rng, 0.015, 0.2)
    U = (U - U.mean(0)) / U.std(0)
    f = LEVELS[level]
    b = [NET_SD] * 3 + [f * NET_SD, f * NET_SD]
    a = [0.0] * 3 + [a_for_eta(b[3], 0.4, te_ms, ratios), a_for_eta(b[4], 0.7, te_ms, ratios)]
    W3 = np.zeros(labels3d.shape + (len(src),), np.float32)
    for k, s in enumerate(src):
        for p, w in zip(s["parcels"], s["weights"]):
            W3[..., k][labels3d == p] = w
        s.update({"b_sd_per_s": b[k], "a_sd": a[k], "n_parcels": len(s["parcels"]), "n_voxels": int((W3[..., k] != 0).sum())})
    band_share = []
    fq = np.fft.rfftfreq(T, tr)
    for k in range(len(src)):
        P = np.abs(np.fft.rfft(U[:, k])) ** 2
        band_share.append(round(float(P[(fq >= 0.01) & (fq <= 0.1)].sum() / P[1:].sum()), 3))
    for s, bs in zip(src, band_share):
        s["band_share"] = bs
    return {"sources": src, "W3": W3, "U": U, "a": a, "b": b, "resp_hz": resp_hz}


# ============================================================================ recovery
def _proj_share(X: np.ndarray, Rm: np.ndarray) -> float:
    """Share of X's energy (parcels x frames) in the span of the rows of Rm (frames)."""
    e0 = float((X ** 2).sum())
    if e0 <= 0 or Rm is None or len(Rm) == 0:
        return 0.0
    M = Rm - Rm.mean(1, keepdims=True)
    Q, _ = np.linalg.qr(M.T)
    return float(((X @ Q) ** 2).sum() / e0)


def recover(inject_dir, runs: dict, out_name: str = "recovery_parcels") -> dict:
    """runs: {mode: parcel pipeline output folder of the injected run}. Writes <out_name>.json / .md in inject_dir."""
    from . import io
    from .regions import band_pass
    idir = Path(inject_dir)
    truth = json.loads((idir / "truth.json").read_text())
    Ut = np.genfromtxt(idir / "truth_timecourses.tsv", delimiter="\t", names=True)
    U = np.column_stack([Ut[n] for n in Ut.dtype.names]).astype(float)
    tr = float(truth.get("tr_s", 0.75))
    out = {"run": truth["run"], "level": truth["level"], "resp_hz": truth.get("resp_hz"), "modes": {}}
    for mode, d in runs.items():
        d = Path(d)
        den = np.load(d / "denoised.npz")
        fr = den["frames"]
        Rm = den["removed_tc"]
        ids = list(den["ids"])
        pos = {int(p): i for i, p in enumerate(ids)}
        st3 = np.load(d / "stage3.npz")
        maps, sids = st3["maps"], list(st3["ids"])
        spos = {int(p): i for i, p in enumerate(sids)}
        rows = []
        for k, s in enumerate(truth["sources"]):
            keep = [(pos[p], w) for p, w in zip(s["parcels"], s["weights"]) if p in pos]
            if not keep:
                rows.append({"source": s["id"], "kind": s["kind"], "n_parcels_used": 0})
                continue
            w = np.array([x[1] for x in keep]) * float(s["b_sd_per_s"])
            X = np.outer(w, U[fr, k])
            Xb = band_pass(X, tr, (0.01, 0.1))
            rec = {"source": s["id"], "kind": s["kind"], "n_parcels_used": len(keep),
                   "removed_whole_band": round(_proj_share(X, Rm), 4), "removed_in_band": round(_proj_share(Xb, Rm), 4)}
            rec["kept_in_band"] = round(1 - rec["removed_in_band"], 4)
            if maps.size:
                ind = np.zeros(len(sids))
                for p, wv in zip(s["parcels"], s["weights"]):
                    if p in spos:
                        ind[spos[p]] = wv
                if ind.std() > 0:
                    r = [abs(np.corrcoef(ind, maps[:, j])[0, 1]) for j in range(maps.shape[1])]
                    rec["best_stage3_map_r"] = round(float(max(r)), 3)
            rows.append(rec)
        pj = json.loads((d / "pipeline.json").read_text())
        out["modes"][mode] = {"sources": rows, "n_removed": pj["stage2"]["n_removed"], "k_stage3": pj["stage3"]["k"]}
    # pre-set criteria (docs/PARCEL_ICA_PLAN.md section 9)
    crit = {}
    for mode, m in out["modes"].items():
        S = {r["source"]: r for r in m["sources"]}
        crit[mode] = {"artifact_removed_ok": all(S[x].get("removed_whole_band", 0) >= 0.5 for x in ("B", "C") if x in S),
                      "neural_kept_ok": all(S[x].get("kept_in_band", 0) >= 0.9 for x in ("N1", "N2", "N3") if x in S),
                      "N2_kept_in_band": S.get("N2", {}).get("kept_in_band"), "B_removed_whole_band": S.get("B", {}).get("removed_whole_band")}
    if "whole" in crit and "high_only" in crit:
        w, h = crit["whole"], crit["high_only"]
        adopt = (w["N2_kept_in_band"] is not None and w["N2_kept_in_band"] < 0.9 and h["N2_kept_in_band"] >= 0.9
                 and h["B_removed_whole_band"] >= 0.5)
        out["decision"] = "high_only" if adopt else "whole"
    out["criteria"] = crit
    io.write_json(idir / f"{out_name}.json", out)
    io.write_text(idir / f"{out_name}.md", render(out))
    return out


def render(out: dict) -> str:
    f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    L = [f"# Parcel pipeline injection ({out['level']}; breathing {out.get('resp_hz')} Hz)", ""]
    for mode, m in out["modes"].items():
        c = out["criteria"][mode]
        L += [f"## Removal: {mode} ({m['n_removed']} removed, stage 3 k = {m['k_stage3']})", "",
              f"- artifact removed >= 0.5 (B, C, whole band): {'yes' if c['artifact_removed_ok'] else 'NO'}; "
              f"networks kept >= 0.9 in band: {'yes' if c['neural_kept_ok'] else 'NO'}", "",
              "| source | kind | parcels | removed, whole band | removed, in band | kept, in band | best stage-3 map r |", "|---|---|---|---|---|---|---|"]
        for r in m["sources"]:
            L.append(f"| {r['source']} | {r['kind']} | {r.get('n_parcels_used')} | {f(r.get('removed_whole_band'))} | {f(r.get('removed_in_band'))} | "
                     f"{f(r.get('kept_in_band'))} | {f(r.get('best_stage3_map_r'))} |")
        L.append("")
    if "decision" in out:
        L.append(f"**Decision rule (pre-set): {out['decision']}**")
    return "\n".join(L) + "\n"


def summary(paths, out_dir) -> dict:
    """Pool recovery_parcels.json files (participants x levels): per level, mode and source the median and range of
    the shares, and how many runs met each criterion."""
    from . import io
    recs = [json.loads(Path(p).read_text()) for p in paths]
    res = {}
    for r in recs:
        for mode, m in r["modes"].items():
            for s in m["sources"]:
                key = (r["level"], mode, s["source"])
                res.setdefault(key, []).append(s)
    rows = []
    for (lev, mode, src), ss in sorted(res.items()):
        g = lambda k: [x[k] for x in ss if x.get(k) is not None]
        rw, rb, kb = g("removed_whole_band"), g("removed_in_band"), g("kept_in_band")
        rows.append({"level": lev, "mode": mode, "source": src, "n": len(ss),
                     "removed_whole_band_median": round(float(np.median(rw)), 3) if rw else None,
                     "removed_whole_band_range": [round(min(rw), 3), round(max(rw), 3)] if rw else None,
                     "kept_in_band_median": round(float(np.median(kb)), 3) if kb else None,
                     "kept_in_band_range": [round(min(kb), 3), round(max(kb), 3)] if kb else None})
    dec = [r.get("decision") for r in recs if r["level"] == "x2"]
    out = {"n_runs": len(recs), "rows": rows, "decisions_x2": dec}
    o = Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    io.write_json(o / "injection_parcels_summary.json", out)
    L = ["# Parcel pipeline injection: summary", "", f"Runs: {len(recs)}; pre-set decision at x2 per run: {dec}", "",
         "| level | removal | source | runs | removed, whole band (median, range) | kept, in band (median, range) |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['level']} | {r['mode']} | {r['source']} | {r['n']} | {r['removed_whole_band_median']} {r['removed_whole_band_range']} | "
                 f"{r['kept_in_band_median']} {r['kept_in_band_range']} |")
    io.write_text(o / "injection_parcels_summary.md", "\n".join(L) + "\n")
    return out
