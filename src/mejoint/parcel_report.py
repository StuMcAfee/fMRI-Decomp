"""Per-participant HTML report of the parcel pipeline (docs/PARCEL_ICA_PLAN.md section 8): one self-contained file
with pre-rendered images (fast loading). Order: summary metrics; network estimates (interactive: choose a component
to see its surface map, time course and spectrum); partitions; integration measures and connectivity matrices;
removed non-neural components shown explicitly with their effect; the physics stage. Participant data: the report
is written under outputs/ (research storage) only.
"""
from __future__ import annotations

import base64
import csv
import html
import io as _io
import json
from pathlib import Path

import numpy as np

DROPPED = np.array([0.55, 0.55, 0.53, 1.0])        # below the reliability floor


def _b64(path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _png(fig, dpi=110) -> str:
    import matplotlib.pyplot as plt
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _spectrum(tc, tr, n_points=220):
    from scipy.ndimage import uniform_filter1d
    T = len(tc)
    f = np.fft.rfftfreq(T, tr)[1:]
    P = (np.abs(np.fft.rfft(tc - tc.mean())) ** 2 / T)[1:]
    P = uniform_filter1d(P, 5)
    step = max(1, len(f) // n_points)
    return f[::step], P[::step]


def component_colours(labels7: list, net_colours: dict, flagged=None) -> np.ndarray:
    """Hue = best Yeo-7 network (atlas colour), lightness steps for further components of the same network."""
    steps = (0.0, 0.45, -0.35, 0.7, -0.55, 0.25, -0.2, 0.55, -0.45)
    seen, col = {}, np.zeros((len(labels7), 4))
    for j, n in enumerate(labels7):
        if flagged is not None and flagged[j]:
            col[j] = [0.33, 0.33, 0.31, 1]
            continue
        i = seen.get(n, 0)
        seen[n] = i + 1
        base = net_colours.get(n, np.array([0.5, 0.5, 0.5, 1]))[:3]
        s = steps[i % len(steps)]
        rgb = base + (1 - base) * s if s >= 0 else base * (1 + s)
        col[j] = np.r_[rgb, 1.0]
    return col


def build(out, res, ids, rel, reliable, rounds, wta, sets, pcmap, nets7, nets17, resources, tr, resp, dim3, amp=None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Patch

    from . import surface_plot as sp
    out = Path(out)
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    R = sp.load_resources(resources, "ico1002_7")
    col7 = sp.network_colours(resources, "Schaefer2018_200Parcels_7Networks_order.dlabel.nii")
    f17 = Path(resources) / "Schaefer2018_200Parcels_17Networks_order.dlabel.nii"
    col17 = sp.network_colours(resources, f17.name) if f17.exists() else {}
    ids = np.asarray(ids)
    seq = LinearSegmentedColormap.from_list("seq_blue", ["#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])
    div = sp.diverging_cmap()
    imgs = {}

    # ---- stage 3 components: one surface row each (interactive section)
    comps = res["stage3"].get("components", [])
    comp_js = []
    if wta:
        Z = sp.zscore(wta["maps"])
        cols = component_colours([c["yeo7"] for c in comps], col7)
        norm = Normalize(-3, 3)
        for j, c in enumerate(comps):
            vals = dict(zip(ids.tolist(), Z[:, j]))
            vc = {h: sp.vertex_colours(vals, R["labels"][h], div, norm) for h in ("L", "R")}
            p = fd / f"stage3_c{j:02d}.png"
            sp.rows_figure(R, [("", vc)], p, row_height=1.45, width=8.0, dpi=80, titles_left=False)
            f, P = _spectrum(wta["tc"][j], tr)
            comp_js.append({"label": f"c{j}", "yeo7": c["yeo7"], "yeo7_r": c["yeo7_r"], "yeo17": c["yeo17"], "yeo17_r": c["yeo17_r"],
                            "asym": c["asymmetry"], "won": c["n_parcels_won"], "high": c["high_share"], "band": c["band_share"],
                            "eta": c["eta"], "flag": c["meets_nonneural_criteria"],
                            "colour": "#%02x%02x%02x" % tuple(int(255 * v) for v in cols[j][:3]),
                            "img": _b64(p), "tc": [round(float(v), 3) for v in wta["tc"][j]],
                            "f": [round(float(v), 4) for v in f], "P": [float(f"{v:.4g}") for v in P],
                            "noise": float(wta["tc_noise_var"][j])})

        # ---- partitions: data-driven winner-take-all (with gaps), Yeo-7, Yeo-17
        win = wta["win"]
        cby = {}
        for p_, w_, r_ in zip(ids.tolist(), win, reliable):
            cby[p_] = DROPPED if not r_ else (sp.NOWIN if w_ < 0 else cols[w_])
        vc_w = {}
        for h in ("L", "R"):
            v = sp.label_colours(cby, R["labels"][h])
            lab = R["labels"][h]
            compv = np.full(lab.size, -9)
            for p_, w_, r_ in zip(ids.tolist(), win, reliable):
                if r_ and w_ >= 0:
                    compv[lab == p_] = w_
            fv = compv[R["surf"][h][1]]
            border = (fv.min(1) != fv.max(1)) & (fv.max(1) >= 0)
            v[np.unique(R["surf"][h][1][border])] *= np.array([0.6, 0.6, 0.6, 1])
            vc_w[h] = v
        rows = [("data-driven\n(winner-take-all)", vc_w)]
        if nets7 is not None:
            rows.append(("Yeo-7", {h: sp.label_colours({p_: col7.get(n, sp.MEDIAL) for p_, n in zip(ids.tolist(), nets7)}, R["labels"][h]) for h in ("L", "R")}))
        if nets17 is not None and col17:
            rows.append(("Yeo-17", {h: sp.label_colours({p_: col17.get(n, sp.MEDIAL) for p_, n in zip(ids.tolist(), nets17)}, R["labels"][h]) for h in ("L", "R")}))
        won = sorted(set(win[(win >= 0) & reliable].tolist()))
        leg = [Patch(color=cols[j], label=f"c{j}: {comps[j]['yeo7']}") for j in won][:40]
        leg += [Patch(color=sp.NOWIN, label="undominated or weak"), Patch(color=DROPPED, label="below reliability floor"),
                Patch(color=sp.MEDIAL, label="no parcel / no voxels")]
        sp.rows_figure(R, rows, fd / "partitions.png", legend=leg, row_height=1.55, width=10.0, dpi=110)
        imgs["partitions"] = _b64(fd / "partitions.png")

    # ---- reliability, participation coefficient, in-band variance removed
    vals = dict(zip(ids.tolist(), np.clip(np.nan_to_num(rel, nan=0.0), 0, 1)))
    sp.rows_figure(R, [("reliability", {h: sp.vertex_colours(vals, R["labels"][h], seq, Normalize(0, 1)) for h in ("L", "R")})],
                   fd / "reliability.png", cbar=(seq, Normalize(0, 1), f"reliability (floor {res['reliability_min']})"), row_height=1.5)
    imgs["reliability"] = _b64(fd / "reliability.png")
    if wta and "dom_pct" in wta:                                       # continuous dominance map (weak parcels grey)
        cby = {}
        for p_, r_, c_, v_ in zip(ids.tolist(), reliable, wta["category"], wta["dom_pct"]):
            cby[p_] = DROPPED if not r_ else (np.array([0.80, 0.80, 0.78, 1.0]) if c_ == "weak" else seq(float(v_)))
        sp.rows_figure(R, [("dominance\n(confidence)", {h: sp.label_colours(cby, R["labels"][h]) for h in ("L", "R")})],
                       fd / "dominance.png", cbar=(seq, Normalize(0, 1), "confidence strongest > second"), row_height=1.5)
        imgs["dominance"] = _b64(fd / "dominance.png")
    if amp is not None:                                                # signal amplitude and thermal noise, same scale
        vmax = float(np.nanpercentile(np.r_[amp["signal"], amp["noise"]], 98)) or 1.0
        nrm = Normalize(0, vmax)
        rows = [(f"in-band signal SD\n(median {np.median(amp['signal']):.3f} 1/s)", {h: sp.vertex_colours(dict(zip(ids.tolist(), amp["signal"])), R["labels"][h], seq, nrm) for h in ("L", "R")}),
                (f"in-band thermal noise SD\n(median {np.median(amp['noise']):.3f} 1/s)", {h: sp.vertex_colours(dict(zip(ids.tolist(), amp["noise"])), R["labels"][h], seq, nrm) for h in ("L", "R")})]
        sp.rows_figure(R, rows, fd / "amplitude_noise.png", cbar=(seq, nrm, "BOLD-like dR2* SD (1/s)"), row_height=1.5)
        imgs["amplitude"] = _b64(fd / "amplitude_noise.png")
    if pcmap:
        rows = [(f"PC, {k}", {h: sp.vertex_colours({p_: v for p_, v in zip(ids.tolist(), pcmap[k]) if np.isfinite(v)}, R["labels"][h], seq, Normalize(0, 1))
                              for h in ("L", "R")}) for k in pcmap]
        sp.rows_figure(R, rows, fd / "participation.png", cbar=(seq, Normalize(0, 1), "participation coefficient"), row_height=1.5)
        imgs["participation"] = _b64(fd / "participation.png")
    dp = out / "denoise_parcels.tsv"
    if dp.exists() and res["stage2"]["n_removed"]:
        with open(dp) as fh:
            rr = list(csv.DictReader(fh, delimiter="\t"))
        v = {int(r["parcel"]): float(r["inband_removed"]) for r in rr if r["inband_removed"] not in ("", "nan", "None")}
        vm = max(0.05, float(np.nanpercentile(np.abs(list(v.values())), 98))) if v else 0.1
        sp.rows_figure(R, [("in-band variance\nremoved", {h: sp.vertex_colours(v, R["labels"][h], div, Normalize(-vm, vm)) for h in ("L", "R")})],
                       fd / "removed_inband.png", cbar=(div, Normalize(-vm, vm), "in-band share removed"), row_height=1.5)
        imgs["removed_inband"] = _b64(fd / "removed_inband.png")

    # ---- connectivity matrices sorted by partition (main set; undominated parcels shown with their strongest component)
    main = sets.get("main: reliable parcels")
    if main is not None and wta:
        idx = main["idx"]
        W = main["W"]
        parts = []
        if nets7 is not None:
            parts.append(("Yeo-7", np.asarray(nets7)[idx]))
        if nets17 is not None:
            parts.append(("Yeo-17", np.asarray(nets17)[idx]))
        parts.append(("data-driven (strongest component)", np.array([f"c{w}" for w in wta["win_full"][idx]], dtype=object)))
        fig, axs = plt.subplots(1, len(parts), figsize=(5.2 * len(parts), 5.4), constrained_layout=True)
        axs = np.atleast_1d(axs)
        vmax = float(np.nanpercentile(W[W > 0], 99)) if (W > 0).any() else 1.0
        for ax, (name, lab) in zip(axs, parts):
            lab = np.array([l if l else "none" for l in lab], dtype=object)
            hemi = np.where(ids[idx] <= res["n_parcels_total"] // 2, 0, 1)
            order = np.lexsort((hemi, lab))
            ax.imshow(W[np.ix_(order, order)], cmap=seq, vmin=0, vmax=vmax, interpolation="nearest")
            ls = lab[order]
            cuts = np.flatnonzero(ls[1:] != ls[:-1]) + 1
            for cpos in cuts:
                ax.axhline(cpos - 0.5, color="#555", lw=0.4)
                ax.axvline(cpos - 0.5, color="#555", lw=0.4)
            starts = np.r_[0, cuts]
            ends = np.r_[cuts, len(ls)]
            if len(starts) <= 25:
                ax.set_xticks((starts + ends) / 2 - 0.5)
                ax.set_xticklabels([ls[s] for s in starts], rotation=90, fontsize=6)
                ax.set_yticks((starts + ends) / 2 - 0.5)
                ax.set_yticklabels([ls[s] for s in starts], fontsize=6)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title(f"sorted by {name}", fontsize=9)
        imgs["matrices"] = _png(fig, 100)

    # ---- removed (stage 2) components: surface map, time course, spectrum
    removed_html = []
    norm = Normalize(-3, 3)
    for r in rounds:
        mr, tcr = r.get("maps_removed"), r.get("tc_removed")
        if mr is None or mr.shape[1] == 0:
            continue
        Zr = sp.zscore(mr)
        rem = [c for c in r["components"] if c["removed"]]
        for j, c in enumerate(rem):
            vals = dict(zip(ids.tolist(), Zr[:, j]))
            p = fd / f"stage2_r{r['round']}_c{c['component']:02d}.png"
            sp.rows_figure(R, [("", {h: sp.vertex_colours(vals, R["labels"][h], div, norm) for h in ("L", "R")})], p,
                           row_height=1.45, width=8.0, dpi=80, titles_left=False)
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 1.9), constrained_layout=True)
            t = np.arange(tcr.shape[1]) * tr
            a1.plot(t, tcr[j], color="#2a78d6", lw=0.6)
            a1.set_xlabel("s", fontsize=7)
            a1.tick_params(labelsize=6)
            f, P = _spectrum(tcr[j], tr)
            a2.semilogy(f, P, color="#2a78d6", lw=0.9)
            a2.axvspan(*res["band_hz"], color="#1baf7a", alpha=0.12, lw=0)
            if resp:
                a2.axvline(resp, color="#eb6834", lw=0.8, ls="--")
            a2.set_xlabel("Hz", fontsize=7)
            a2.tick_params(labelsize=6)
            removed_html.append(
                f"<div class='card'><h4>round {r['round']}, component c{c['component']}</h4>"
                f"<p class='m'>above 0.12 Hz {c['high_share']:.2f} &middot; at breathing frequency {c['resp_share']:.2f} &middot; "
                f"peak {c['peak_hz']:.3f} Hz &middot; S0 coupling {c['eta'] if c['eta'] is not None else '-'} &middot; stability {c['iq']:.2f}</p>"
                f"<img src='data:image/png;base64,{_b64(p)}'><img src='data:image/png;base64,{_png(fig, 90)}'></div>")

    # ---- eigenvalue spectra
    fig, axs = plt.subplots(1, 2, figsize=(10, 2.8), constrained_layout=True)
    for r in res["stage2"]["rounds"]:
        lam = np.array(r["eigenvalues_top"])
        axs[0].semilogy(np.arange(1, lam.size + 1), lam, "o-", ms=3, lw=1, label=f"round {r['round']} (k {r['k']})")
        if r.get("null_threshold"):
            axs[0].axhline(r["null_threshold"], color="#eb6834", ls="--", lw=0.8)
    axs[0].set_title("stage 2: eigenvalues and leftover-channel threshold (dashed)", fontsize=8)
    axs[0].legend(fontsize=6, frameon=False)
    lam3 = np.array(res["stage3"]["eigenvalues_top"])
    axs[1].semilogy(np.arange(1, lam3.size + 1), lam3, "o-", ms=3, lw=1, color="#2a78d6")
    axs[1].axhline(res["stage3"]["mp_edge"], color="#555", ls=":", lw=0.8)
    if res["stage3"].get("fidelity_eigenvalue"):
        axs[1].axhline(res["stage3"]["fidelity_eigenvalue"], color="#2a78d6", ls="--", lw=0.8)
    axs[1].set_title(f"stage 3: thermal-noise edge (dotted, {res['stage3']['k_mp']}), map fidelity >= {res['stage3'].get('fidelity_min')} "
                     f"(dashed): k = {res['stage3']['k']}", fontsize=8)
    for a in axs:
        a.tick_params(labelsize=6)
    imgs["eigen"] = _png(fig, 100)

    path = out / "report.html"
    path.write_text(_html(res, comp_js, imgs, removed_html))
    return path


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else html.escape(str(v))


def _html(res, comp_js, imgs, removed_html) -> str:
    m, s2, s3 = res["metrics"], res["stage2"], res["stage3"]
    img = lambda k, alt: f"<img alt='{alt}' src='data:image/png;base64,{imgs[k]}'>" if k in imgs else "<p class='m'>not available</p>"
    integ = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{s['n_parcels']} ({s['n_with_signal']})</td><td>{_fmt(s['pr'], 2)}</td>"
        f"<td>{_fmt(s['pr_resid'], 2)} <span class='m'>({_fmt(s.get('pr_resid_null_median'), 2)})</span></td><td>{_fmt(s['global_share'])}</td>"
        f"<td>{_fmt(s.get('edge_density'))}</td>"
        + "".join(f"<td>{_fmt(s['pc'].get(p))}</td>" for p in ("Yeo-7", "Yeo-17", "data-driven"))
        + "".join(f"<td>{_fmt(s['seg'].get(p))}</td>" for p in ("Yeo-7", "Yeo-17", "data-driven")) + "</tr>"
        for n, s in res.get("integration", {}).items())
    rounds = "".join(f"<tr><td>{r['round']}</td><td>{r['df']}</td><td>{r['k']}</td>"
                     f"<td>{sum(c['removed'] for c in r.get('components', []))}</td></tr>" for r in s2["rounds"])
    dn = res.get("denoising") or {}
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parcel pipeline report</title>
<style>
:root{{--bg:#fcfcfb;--fg:#1f1f1d;--mut:#6b6a64;--line:#dddcd6;--acc:#2a78d6;--card:#ffffff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--fg:#f0efec;--mut:#a9a89f;--line:#3a3a37;--acc:#3987e5;--card:#232321}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:16px}} h1{{font-size:20px;margin:8px 0 4px}} h2{{font-size:16px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}}
h4{{margin:4px 0;font-size:13px}} .m{{color:var(--mut);font-size:12px;margin:2px 0}} img{{max-width:100%;display:block;background:#fff;border-radius:4px}}
table{{border-collapse:collapse;font-size:12px;margin:6px 0;overflow-x:auto;display:block}} td,th{{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;white-space:nowrap}}
.kpi{{display:flex;flex-wrap:wrap;gap:10px}} .kpi div{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 12px;min-width:120px}}
.kpi b{{display:block;font-size:18px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px;margin:8px 0}}
#comps button{{margin:2px;padding:3px 7px;border:1px solid var(--line);border-radius:4px;background:var(--card);color:var(--fg);cursor:pointer;font-size:12px}}
#comps button.on{{outline:2px solid var(--acc)}} .sw{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:middle}}
canvas{{width:100%;height:150px;background:var(--card);border:1px solid var(--line);border-radius:4px}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
@media (max-width:700px){{.two{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Parcel pipeline: networks and denoising</h1>
{"" if res.get("comparable", True) else f"<p style='border:1px solid #d1366b;border-radius:6px;padding:6px 10px;color:#d1366b'><b>Not comparable:</b> {res['n_frames']} frames, fewer than the common {res.get('n_frames_common')}. k, node count and other count-based measures cannot be compared with other participants.</p>"}
<p class="m">Icosahedral parcels (fs_LR 32k, 1876); {res['n_frames']} frames, TR {res['tr_s']} s; breathing peak {str(res['respiratory_peak_hz']) + ' Hz' if res['respiratory_peak_hz'] else 'not found in the motion parameters'}.
Exploratory: route B is unchanged.</p>
<div class="kpi">
<div><span class="m">reliable coverage</span><b>{m['reliable_coverage_pct']} %</b><span class="m">{m['reliable_parcels']} parcels</span></div>
<div><span class="m">node count</span><b>{m['node_count']}</b><span class="m">{_fmt(m['patches_per_component'], 2)} patches / component</span></div>
<div><span class="m">network components (k)</span><b>{m['k_stage3']}</b><span class="m">map fidelity &ge; {s3.get('fidelity_min')} &middot; {m['components_winning']} win parcels</span></div>
<div><span class="m">undominated (shared)</span><b>{_fmt(m['undominated_pct'], 1)} %</b><span class="m">weak {_fmt(m.get('weak_pct'), 1)} % of reliable parcels</span></div>
<div><span class="m">non-neural removed</span><b>{s2['n_removed']}</b><span class="m">{len(s2['rounds'])} round(s){' &middot; CAP REACHED' if s2['cap_reached'] else ''}</span></div>
</div>

<h2>Network estimates (stage 3)</h2>
<p class="m">k = components whose maps are expected to correlate at least {s3.get('fidelity_min')} with the true maps ({s3.get('k_fidelity')};
thermal-noise threshold: {s3['k_mp']}). Choose a component: its map (loading, z over parcels), time course and spectrum (band 0.01-0.1 Hz shaded; dotted line: thermal noise level).
Labels give the best positive overlap with Yeo-7 / Yeo-17; asymmetry = (left - right) / (left + right) of positive loadings.</p>
<div id="comps"></div><div class="card"><h4 id="ct"></h4><p class="m" id="cm"></p><img id="ci" alt="component map">
<div class="two"><canvas id="tcv"></canvas><canvas id="spv"></canvas></div></div>

<h2>Partitions</h2>
<p class="m">Data-driven: each reliable parcel coloured by its winning component (z &ge; {res['settings']['z_min']} and a clear margin over the second strongest); hue = the component's best Yeo-7 network,
lightness steps separate components of one network; patch borders darkened. Node count = contiguous same-winner patches ({m['node_count']},
single-parcel patches {m['single_parcel_patches']}).</p>
{img('partitions', 'partitions')}
<h4>Dominance</h4>
<p class="m">For each reliable parcel: the confidence that its strongest component truly exceeds the second strongest, given the loadings'
estimation uncertainty. Dark = one network clearly owns the parcel; light = the top two cannot be told apart (shared, undominated below
{res['settings'].get('dominance_q')}); grey = no strong component (weak); dark grey = below the reliability floor. Confidence also rises with more
frames and stronger signal.
Dominated {_fmt(m.get('dominated_pct'), 1)} %, undominated (shared) {_fmt(m['undominated_pct'], 1)} %, weak {_fmt(m.get('weak_pct'), 1)} %.</p>
{img('dominance', 'dominance')}

<h2>Integration</h2>
<p class="m">Noise-corrected in-band correlations; participation ratio (PR) with the circular-shift null median; participation coefficient (PC) and
segregation (seg) on the correlations beyond the brain-wide pattern (first pattern removed, as in the primary outcome), positive weights clipped at
1, using only edges reliably above chance (each edge against its own circular-shift null, false discovery rate 0.05); edge density = share of
possible edges kept. Main: reliable parcels (data-driven partition without undominated parcels); A: every parcel with
voxels; B: reliable parcels with undominated parcels assigned to their strongest component.</p>
<table><tr><th>set</th><th>parcels (with signal)</th><th>PR</th><th>PR without first (null)</th><th>global share</th><th>edge density</th>
<th>PC Yeo-7</th><th>PC Yeo-17</th><th>PC data-driven</th><th>seg Yeo-7</th><th>seg Yeo-17</th><th>seg data-driven</th></tr>{integ}</table>
<h4>Connectivity beyond the brain-wide pattern (main set; edges reliably above chance), sorted by partition</h4>{img('matrices', 'connectivity matrices')}
<h4>Participation coefficient by parcel (main set)</h4>{img('participation', 'participation coefficient')}

<h2>Removed non-neural components (stage 2)</h2>
<p class="m">Repeated spatial ICA at the leftover-channel count; removed when the signal share above 0.12 Hz &gt; {res['settings']['high_share']} or the S0 coupling &gt;
{res['settings']['eta']}. Stopped: {html.escape(s2['stop'])}. In-band BOLD-like variance removed per parcel (10th / 50th / 90th percentile):
{dn.get('inband_removed_p10_50_90')}; whole band {dn.get('total_removed_p10_50_90')}.</p>
<table><tr><th>round</th><th>df</th><th>k (leftover null)</th><th>removed</th></tr>{rounds}</table>
{''.join(removed_html) or "<p class='m'>No component met the criteria.</p>"}
{img('removed_inband', 'in-band variance removed')}

<h2>Physics stage</h2>
<p class="m">Reliability = signal share of each parcel's slow BOLD-like variance. Below: the two parts it compares, on one scale: in-band signal
amplitude (noise-corrected) and in-band thermal noise. Low coverage from high noise points to acquisition (coil, head position, motion); low
coverage from low signal points to physiology or state (arousal, breathing, vascular reactivity).</p>
{img('reliability', 'reliability')}
{img('amplitude', 'signal amplitude and thermal noise')}
{img('eigen', 'eigenvalues')}
<p class="m">Generated by mejoint (parcel_pipeline). Input: {html.escape(res['input'])}.</p>
</main>
<script>
const C={json.dumps(comp_js)};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function plot(cv,xs,ys,opt){{const d=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=w*d;cv.height=h*d;const g=cv.getContext('2d');g.scale(d,d);
g.clearRect(0,0,w,h);const L=36,B=18,T=6,Rm=6;let ys2=opt.log?ys.map(v=>Math.log10(Math.max(v,1e-12))):ys;const x0=Math.min(...xs),x1=Math.max(...xs);
let y0=Math.min(...ys2),y1=Math.max(...ys2);if(opt.floor!==undefined){{const fl=opt.log?Math.log10(opt.floor):opt.floor;y0=Math.min(y0,fl);y1=Math.max(y1,fl);}}
const X=v=>L+(v-x0)/(x1-x0||1)*(w-L-Rm),Y=v=>T+(1-(v-y0)/(y1-y0||1))*(h-T-B);g.font='10px system-ui';g.fillStyle=css('--mut');
if(opt.band){{g.fillStyle='rgba(27,175,122,0.13)';g.fillRect(X(opt.band[0]),T,X(opt.band[1])-X(opt.band[0]),h-T-B);}}
g.strokeStyle=css('--line');g.beginPath();g.moveTo(L,T);g.lineTo(L,h-B);g.lineTo(w-Rm,h-B);g.stroke();g.fillStyle=css('--mut');
g.fillText(opt.xl,w-Rm-30,h-4);g.fillText(opt.yl,2,T+8);if(opt.floor!==undefined){{const fy=Y(opt.log?Math.log10(opt.floor):opt.floor);g.setLineDash([2,3]);g.beginPath();g.moveTo(L,fy);g.lineTo(w-Rm,fy);g.stroke();g.setLineDash([]);}}
g.strokeStyle=css('--acc');g.lineWidth=1;g.beginPath();xs.forEach((x,i)=>{{const px=X(x),py=Y(ys2[i]);i?g.lineTo(px,py):g.moveTo(px,py);}});g.stroke();}}
function show(i){{const c=C[i];document.querySelectorAll('#comps button').forEach((b,j)=>b.classList.toggle('on',j===i));
document.getElementById('ct').textContent=c.label+': '+(c.yeo7||'-')+' (r '+c.yeo7_r+') / '+(c.yeo17||'-')+' (r '+c.yeo17_r+')';
document.getElementById('cm').textContent='parcels won '+c.won+' · asymmetry '+c.asym+' · in band '+c.band+' · above 0.12 Hz '+c.high+' · S0 coupling '+c.eta+(c.flag?' · meets non-neural criteria (not removed)':'');
document.getElementById('ci').src='data:image/png;base64,'+c.img;
plot(document.getElementById('tcv'),c.tc.map((v,k)=>k*{res['tr_s']}),c.tc,{{xl:'s',yl:'time course'}});
plot(document.getElementById('spv'),c.f,c.P,{{log:true,xl:'Hz',yl:'power',band:{json.dumps(res['band_hz'])},floor:c.noise}});}}
const box=document.getElementById('comps');C.forEach((c,i)=>{{const b=document.createElement('button');b.innerHTML='<span class="sw" style="background:'+c.colour+'"></span>'+c.label+' '+(c.yeo7||'');b.onclick=()=>show(i);box.appendChild(b);}});
if(C.length)show(0);window.addEventListener('resize',()=>{{const on=[...document.querySelectorAll('#comps button')].findIndex(b=>b.classList.contains('on'));if(on>=0)show(on);}});
</script></body></html>"""
