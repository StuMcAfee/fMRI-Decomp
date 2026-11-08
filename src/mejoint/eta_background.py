"""Background-corrected eta per component (user decision 2026-11-08).

Why: the injection runs showed that the reported eta of a component is set by a background, not by the
component. Any slow time course regressed on the channels picks up energy from the data's own slow fluctuations
by chance, summed over ~40,000 voxels; the thermal debiasing does not remove it, and that background has
eta ~0.25-0.45. A pure-dR2* network found at k = 150 (right amplitude, right ratio) was reported with eta 0.22;
every native component sits at ~0.42.

Method: for each component time course t (unit SD, in the nuisance-projected space), build surrogates that keep
its temporal structure but break its alignment with the data, regress the channels on each one (single
regression, as for t itself), and subtract the mean surrogate E_S0 and E_R2 from t's own:
    eta_bg = (E_S0(t) - mean E_S0(surr)) / [(E_S0(t) - mean E_S0(surr)) + (E_R2(t) - mean E_R2(surr))]
Two kinds of surrogate (user, 2026-11-08):
  shift: t circularly shifted by a random lag of at least min_shift_s (the end wraps to the beginning); keeps
         the exact spectrum and every within-series feature, and is the stronger control (user's experience
         with local field potentials);
  phase: t with its Fourier phases randomized (same amplitude spectrum).
Both are projected off the nuisance basis and restandardized like t. Also reported: the component's energy over
the surrogates' (and a z), and the share of surrogates with at least its energy (p). That p is descriptive only:
a component's time course is fitted to the same data, so it beats shifted copies of itself even on pure noise
(dry run, 2026-11-08: all six noise components p <= 0.02). Which components count stays with the pipeline null
(top-k directions of surrogate data, through the same selection; `above_null` in components.json).

The component's own maps here are single-regression maps of t (not the joint, cross-fitted maps of the sweep), so
the component and its surrogates are measured the same way; this is what gave the right answer for the true
time courses of the injected sources.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import metrics


def surrogates(t: np.ndarray, n: int, rng, method: str, min_shift: int) -> np.ndarray:
    """(T, n) surrogates of one time course."""
    T = t.size
    if method == "shift":
        lo, hi = int(min_shift), T - int(min_shift)
        if hi < lo:
            raise ValueError(f"min_shift {min_shift} too large for {T} frames")
        lags = rng.integers(lo, hi + 1, int(n))
        return np.stack([np.roll(t, int(L)) for L in lags], axis=1)
    if method == "phase":
        F = np.fft.rfft(t)
        ph = rng.uniform(0, 2 * np.pi, (F.size, int(n)))
        ph[0] = 0.0
        if T % 2 == 0:
            ph[-1] = 0.0
        return np.fft.irfft(np.abs(F)[:, None] * np.exp(1j * ph), n=T, axis=0)
    raise ValueError("method must be shift or phase")


def _maps(X, ok, Tc):
    """Single-regression betas of rows X (finite rows ok) on unit-SD columns Tc: X t / T."""
    out = np.full((X.shape[0], Tc.shape[1]), np.nan)
    out[ok] = X[ok] @ Tc / Tc.shape[0]
    return out


def background(c0, c1, R, M, n: int = 50, rng=None, Q=None, methods=("shift", "phase"), min_shift: int = 40,
               chunk: int = 400) -> dict:
    """Per component (columns of M): own energies and the surrogate energies for each method."""
    rng = np.random.default_rng(0) if rng is None else rng
    T, k = M.shape
    X0 = np.nan_to_num(np.asarray(c0, np.float64))
    X1 = np.nan_to_num(np.asarray(c1, np.float64))
    ok0 = np.isfinite(c0).all(1)
    ok1 = np.isfinite(c1).all(1)
    Mu = metrics._unit(np.asarray(M, np.float64), Q)
    own = metrics.energies(_maps(X0, ok0, Mu), _maps(X1, ok1, Mu), R, 1.0 / T)
    out = {"own": {"E_S0": own["E_S0"], "E_R2": own["E_R2"], "eta": own["eta"]}, "n": int(n), "min_shift": int(min_shift)}
    for method in methods:
        S0 = np.zeros((k, n))
        R2 = np.zeros((k, n))
        per = max(1, int(chunk) // int(n))                               # components per chunk
        for g in range(0, k, per):
            cols = list(range(g, min(k, g + per)))
            S = np.concatenate([surrogates(Mu[:, j], n, rng, method, min_shift) for j in cols], axis=1)
            S = metrics._unit(S, Q)
            en = metrics.energies(_maps(X0, ok0, S), _maps(X1, ok1, S), R, 1.0 / T)
            S0[cols] = en["E_S0"].reshape(len(cols), n)
            R2[cols] = en["E_R2"].reshape(len(cols), n)
        out[method] = {"E_S0": S0, "E_R2": R2}
    out["support"] = support_background(X0, ok0, X1, ok1, R, Mu, n, rng, methods, min_shift)
    return out


def support_background(X0, ok0, X1, ok1, R, Mu, n, rng, methods, min_shift, z_sel: float = 3.0, min_vox: int = 20) -> dict:
    """eta where the component lives, cross-fitted in time: its voxels are chosen in one half of the run (debiased
    channel energy of its single-regression betas above z_sel noise SDs), its energies and its surrogates'
    (circular shifts / phase randomization of the half's time course) are measured on the other half, and the
    two directions are summed. Over a few hundred voxels the chance background is small and its spread is far
    smaller than over the whole brain."""
    T, k = Mu.shape
    h = T // 2
    halves = ((np.arange(h), np.arange(h, T)), (np.arange(h, T), np.arange(h)))
    unit = lambda x: (x - x.mean(0)) / x.std(0)
    own = {"E_S0": np.zeros(k), "E_R2": np.zeros(k)}
    sur = {m: {"E_S0": np.zeros((k, n)), "E_R2": np.zeros((k, n))} for m in methods}
    nsup = np.zeros((k, 2), int)
    for hi_, (sel, meas) in enumerate(halves):
        Ts = unit(Mu[sel])
        vb_s = 1.0 / sel.size
        A0 = _maps(X0[:, sel], ok0, Ts)
        A1 = _maps(X1[:, sel], ok1, Ts)
        e = np.nan_to_num(A0 ** 2 - vb_s) + np.where(np.isfinite(A1), A1 ** 2 - vb_s, 0.0)
        z = e / (2.0 * vb_s)
        Tm = unit(Mu[meas])
        vb_m = 1.0 / meas.size
        for j in range(k):
            idx = np.flatnonzero(z[:, j] > z_sel)
            nsup[j, hi_] = idx.size
            if idx.size < min_vox:
                continue
            Y0, Y1, Rj = X0[idx][:, meas], X1[idx][:, meas], R[idx]
            o0, o1 = ok0[idx], ok1[idx]
            en = metrics.energies(_maps(Y0, o0, Tm[:, [j]]), _maps(Y1, o1, Tm[:, [j]]), Rj, vb_m)
            own["E_S0"][j] += en["E_S0"][0]
            own["E_R2"][j] += en["E_R2"][0]
            for m in methods:
                S = unit(surrogates(Tm[:, j], n, rng, m, min(min_shift, meas.size // 4)))
                es = metrics.energies(_maps(Y0, o0, S), _maps(Y1, o1, S), Rj, vb_m)
                sur[m]["E_S0"][j] += es["E_S0"]
                sur[m]["E_R2"][j] += es["E_R2"]
    bad = nsup.min(1) < min_vox
    for d in (own["E_S0"], own["E_R2"]):
        d[bad] = np.nan
    for m in methods:
        sur[m]["E_S0"][bad] = np.nan
        sur[m]["E_R2"][bad] = np.nan
    return {"own": own, "n_support": nsup.mean(1), **sur}


def corrected(bg: dict, method: str, where: str = "brain") -> dict:
    """eta with the mean surrogate energies removed, and the component's energy against the surrogates.
    where: "brain" (all voxels) or "support" (the component's own voxels, cross-fitted in time). eta_lo / eta_hi:
    5th / 95th percentile of eta with each single surrogate's energies removed (how much the background a given
    time course happens to pick up can vary)."""
    src = bg if where == "brain" else bg["support"]
    own = src["own"]
    s0, r2 = src[method]["E_S0"], src[method]["E_R2"]
    es = own["E_S0"] - s0.mean(1)
    er = own["E_R2"] - r2.mean(1)
    tot_own = own["E_S0"] + own["E_R2"]
    tot_sur = s0 + r2
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(es + er > 0, es / (es + er), np.nan)
        bg_eta = s0.mean(1) / tot_sur.mean(1)
        ratio = tot_own / tot_sur.mean(1)
        z = (tot_own - tot_sur.mean(1)) / tot_sur.std(1)
        esi, eri = own["E_S0"][:, None] - s0, own["E_R2"][:, None] - r2
        etai = np.where(esi + eri > 0, esi / (esi + eri), np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(etai, 5, axis=1), np.nanpercentile(etai, 95, axis=1)
    p = ((tot_sur >= tot_own[:, None]).sum(1) + 1) / (tot_sur.shape[1] + 1)
    p = np.where(np.isfinite(tot_own), p, np.nan)
    out = {"eta": eta, "eta_lo": lo, "eta_hi": hi, "background_eta": bg_eta, "energy_over_background": ratio, "z": z, "p": p}
    if where == "support":
        out["n_support"] = bg["support"]["n_support"]
    return out


def run_dir(frontend_dir, k_dir, cfg: dict, n: int | None = None) -> dict:
    """Background-corrected eta for every component of one decomposition folder; writes eta_background.json/.md."""
    from . import io
    fe, kd = Path(frontend_dir), Path(k_dir)
    B = cfg.get("metrics", {}).get("background", {})
    n = int(n or B.get("n", 50))
    methods = tuple(B.get("methods", ["shift", "phase"]))
    tr = float(cfg["acquisition"]["tr_s"])
    min_shift = int(round(float(B.get("min_shift_s", 30.0)) / tr))
    c0, c1 = np.load(fe / "c0.npy"), np.load(fe / "c1.npy")
    R = np.load(fe / "R.npy")
    Q = np.load(fe / "nuisance_Q.npy") if (fe / "nuisance_Q.npy").exists() else None
    t = np.genfromtxt(kd / "time_courses.tsv", delimiter="\t", names=True)
    M = np.column_stack([t[c] for c in t.dtype.names]).astype(np.float64)
    if M.shape[0] != c0.shape[1]:
        raise ValueError(f"{kd}: {M.shape[0]} frames, the front end has {c0.shape[1]}")
    bg = background(c0, c1, R, M, n, np.random.default_rng(int(cfg.get("project_seed", 0)) + 8), Q, methods, min_shift)
    cj = json.loads((kd / "components.json").read_text()) if (kd / "components.json").exists() else {"components": []}
    old = {c["component"]: c for c in cj.get("components", [])}
    res = {m: corrected(bg, m) for m in methods}
    res_s = {m: corrected(bg, m, "support") for m in methods}
    rows = []
    for j in range(M.shape[1]):
        r = {"component": j + 1, "eta_reported": old.get(j + 1, {}).get("eta"), "above_null": old.get(j + 1, {}).get("above_null"),
             "eta_single_thermal": _r(bg["own"]["eta"][j])}
        for m in methods:
            for key in ("eta", "eta_lo", "eta_hi", "background_eta", "energy_over_background", "z", "p"):
                r[f"{key}_{m}"] = _r(res[m][key][j])
            for key in ("eta", "eta_lo", "eta_hi", "background_eta", "energy_over_background", "p"):
                r[f"{key}_support_{m}"] = _r(res_s[m][key][j])
        r["n_support_voxels"] = _r(res_s[methods[0]]["n_support"][j], 0)
        rows.append(r)
    out = {"k_dir": str(kd), "frontend_dir": str(fe), "k": int(M.shape[1]), "n_surrogates": n, "methods": list(methods),
           "min_shift_frames": min_shift, "components": rows, **io.provenance(cfg, {})}
    io.write_json(kd / "eta_background.json", out)
    io.write_text(kd / "eta_background.md", render(out))
    return out


def _r(x, nd=4):
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _summ(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "median": round(float(np.median(v)), 3), "q25": round(float(np.percentile(v, 25)), 3),
            "q75": round(float(np.percentile(v, 75)), 3), "below_0.2": int((v < 0.2).sum()), "above_0.8": int((v > 0.8).sum())}


def render(out: dict) -> str:
    rows = out["components"]
    L = [f"# Background-corrected eta: {out['k_dir']} (k = {out['k']})", "",
         f"{out['n_surrogates']} surrogates per component and method; circular shifts of at least {out['min_shift_frames']} frames. "
         "eta_shift / eta_phase: eta with the surrogates' mean energies removed. p: share of the component's surrogates "
         "with at least its energy (descriptive: a component is fitted to the data and beats its own shifted copies even "
         "on noise; which components count is the pipeline null, above_null).", ""]
    for m in out["methods"]:
        sig_s = [r for r in rows if r.get("above_null") is not False and r.get(f"eta_support_{m}") is not None]
        ss = _summ(r.get(f"eta_support_{m}") for r in sig_s)
        L.append(f"- {m}, support (the component's own voxels, cross-fitted in time): {len(sig_s)} of {len(rows)} above the "
                 f"pipeline null with a support; corrected eta median {ss.get('median')} (IQR {ss.get('q25')}-{ss.get('q75')}), < 0.2: "
                 f"{ss.get('below_0.2')}, > 0.8: {ss.get('above_0.8')}.")
        sig = [r for r in rows if r.get("above_null") is not False]
        s_all, s_sig = _summ(r.get(f"eta_{m}") for r in rows), _summ(r.get(f"eta_{m}") for r in sig)
        L.append(f"- {m}, whole brain: {len(sig)} of {len(rows)} components above the pipeline null; corrected eta of those: "
                 f"median {s_sig.get('median')} (IQR {s_sig.get('q25')}-{s_sig.get('q75')}), < 0.2: {s_sig.get('below_0.2')}, "
                 f"> 0.8: {s_sig.get('above_0.8')}; background eta median {_summ(r.get(f'background_eta_{m}') for r in rows).get('median')}.")
    L += ["", "| comp | eta reported | eta (single, thermal only) | " + " | ".join(f"eta {m} | background eta {m} | energy / background {m} | p {m}" for m in out["methods"]) + " |",
          "|---|---|---|" + "---|---|---|---|" * len(out["methods"])]
    f = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"
    for r in rows:
        L.append(f"| {r['component']} | {f(r['eta_reported'])} | {f(r['eta_single_thermal'])} | " +
                 " | ".join(f"{f(r.get(f'eta_{m}'))} | {f(r.get(f'background_eta_{m}'))} | {f(r.get(f'energy_over_background_{m}'), 2)} | {f(r.get(f'p_{m}'))}"
                            for m in out["methods"]) + " |")
    return "\n".join(L) + "\n"


def summary(phase5_root, out_dir, ks=(50, 90, 150), sweep_name="phase4_sweep_mc", runs=None) -> dict:
    """Native components across runs: corrected eta distributions per k and method (above their circular-shift
    surrogates, p <= 0.05), and a histogram figure."""
    from . import io
    root = Path(phase5_root)
    rows = []
    for rd in sorted(p for p in (root.iterdir() if root.is_dir() else []) if p.is_dir()):
        if runs and not any(str(r) in rd.name for r in runs):
            continue
        for k in ks:
            f = rd / sweep_name / f"k{k}" / "eta_background.json"
            if f.exists():
                for c in json.loads(f.read_text())["components"]:
                    rows.append({"run": rd.name, "k": k, **c})
    table = []
    for k in ks:
        for m in ("shift", "phase"):
            rk = [r for r in rows if r["k"] == k and r.get(f"eta_{m}", "x") != "x"]
            if not rk:
                continue
            sig = [r for r in rk if r.get("above_null") is not False]
            table.append({"k": k, "method": m, "n_runs": len({r["run"] for r in rk}), "n_components": len(rk), "n_above": len(sig),
                          "eta_reported": _summ(r.get("eta_reported") for r in sig), "eta_corrected": _summ(r.get(f"eta_{m}") for r in sig),
                          "eta_support": _summ(r.get(f"eta_support_{m}") for r in sig),
                          "background_eta": _summ(r.get(f"background_eta_{m}") for r in rk)})
    out = {"table": table, "rows": rows}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    io.write_json(out_dir / "eta_background_summary.json", out)
    L = ["# Background-corrected eta of the native components", "",
         "Components above the pipeline null. eta reported: the sweep's eta; corrected: with the surrogates' mean energies "
         "removed, over the whole brain and over the component's own voxels (support; chosen in one half of the run, "
         "measured in the other). < 0.2: dR2*-dominated (BOLD-like); > 0.8: dS0-dominated.", "",
         "| k | surrogates | runs | components | above null | eta reported median | corrected, whole brain: median (IQR), < 0.2, > 0.8 | corrected, support: median (IQR), < 0.2, > 0.8 | background eta |",
         "|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        e, s = t["eta_corrected"], t["eta_support"]
        L.append(f"| {t['k']} | {t['method']} | {t['n_runs']} | {t['n_components']} | {t['n_above']} | {t['eta_reported'].get('median')} | "
                 f"{e.get('median')} ({e.get('q25')}-{e.get('q75')}), {e.get('below_0.2')}, {e.get('above_0.8')} | "
                 f"{s.get('median')} ({s.get('q25')}-{s.get('q75')}), {s.get('below_0.2')}, {s.get('above_0.8')} | {t['background_eta'].get('median')} |")
    io.write_text(out_dir / "eta_background_summary.md", "\n".join(L) + "\n")
    try:
        _figure(rows, ks, out_dir / "eta_background_summary.png")
    except Exception as exc:
        print(f"figure failed: {exc}")
    return out


def _figure(rows, ks, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [k for k in ks if any(r["k"] == k for r in rows)]
    fig, axs = plt.subplots(1, max(1, len(ks)), figsize=(4.6 * max(1, len(ks)), 3.8), squeeze=False)
    bins = np.linspace(-0.4, 1.4, 37)
    for ax, k in zip(axs[0], ks):
        rk = [r for r in rows if r["k"] == k]
        for key, col, lab in (("eta_reported", "#a3a19b", "reported"), ("eta_shift", "#2a78d6", "corrected, whole brain (circular shift)"),
                              ("eta_support_shift", "#1baf7a", "corrected, support (circular shift)"),
                              ("eta_phase", "#eb6834", "corrected, whole brain (phase)")):
            v = [r[key] for r in rk if r.get(key) is not None and r.get("above_null") is not False]
            if v:
                ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step", color=col, lw=1.6, label=f"{lab} (n {len(v)})")
        ax.set_title(f"k = {k}: components above the pipeline null", fontsize=9, loc="left")
        ax.set_xlabel("eta (0 = dR2*, 1 = dS0)")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
