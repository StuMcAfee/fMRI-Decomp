"""The two per-participant wave figures (docs/WAVES_BRIEF.md section 5), styled after the mock-ups in
docs/mockups/waves/ and adapted from a flat sheet to volumes.

  wave_mode_k.png         A q (lag on a cyclic map, faded by amplitude, gradient arrows), axial montage with the
                          posterior fossa; B s (faded against a 1 % S0 change); C voxelwise eta; D cycle
                          snapshots (axial and sagittal); E space-time plot along the fitted axis.
  wave_diagnostics_k.png  A lag per echo along the axis with bootstrap bands; B TE4 - TE1 lag against its noise
                          null; C eta along the axis; D direction histogram (polar) with R; E echo phasors in three
                          representative voxels; F power spectrum of the mode time course.

Volumes are shown in RAS display orientation (right of the participant on the right of the image, anterior
up), using the orientation codes of the front end's reference image.
"""
from __future__ import annotations

import numpy as np

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BRAIN = "#e9e8e4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ECHO_RAMP = ["#86b6ef", "#5598e7", "#256abf", "#104281"]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
        "legend.frameon": False})
    return plt


# ------------------------------------------------------------------ orientation
def ras_transform(axcodes):
    """(perm, flip): array axis perm[i] holds RAS axis i; flip[i] when it runs L, P or I."""
    axcodes = tuple(axcodes or ("R", "A", "S"))
    pairs = (("R", "L"), ("A", "P"), ("S", "I"))
    perm, flip = [], []
    for pos, neg in pairs:
        j = next(i for i, c in enumerate(axcodes) if c in (pos, neg))
        perm.append(j)
        flip.append(axcodes[j] == neg)
    return perm, flip


def to_ras(vol, axcodes):
    perm, flip = ras_transform(axcodes)
    v = np.transpose(vol, perm + list(range(3, vol.ndim)))
    for i, f in enumerate(flip):
        if f:
            v = np.flip(v, i)
    return v


def vec_to_ras(vec, axcodes):
    """Vectors (..., 3) in array-axis mm components to RAS components."""
    perm, flip = ras_transform(axcodes)
    v = vec[..., perm]
    return v * np.where(flip, -1.0, 1.0)


def _vol(x, mask3d, fill=np.nan, dtype=float):
    v = np.full(mask3d.shape + x.shape[1:], fill, dtype=dtype)
    v[mask3d] = x
    return v


# ------------------------------------------------------------------ colour encoding
def lag_rgb(lag, amp, period, inside):
    """Cyclic `twilight` colour of the lag within the cycle, faded towards the brain background by amplitude
    (amp in [0, 1]); outside the brain the figure background."""
    import matplotlib
    import matplotlib.colors as mc
    cyc = matplotlib.colormaps["twilight"]
    rgb = cyc(np.mod(np.nan_to_num(lag), period) / period)[..., :3]
    a = np.clip(np.nan_to_num(amp), 0, 1)[..., None]
    out = rgb * a + np.array(mc.to_rgb(BRAIN)) * (1 - a)
    out[~inside] = mc.to_rgb(SURF)
    return out


def montage_slices(mask_ras, n):
    zs = np.flatnonzero(mask_ras.sum((0, 1)) > 0.05 * mask_ras.sum((0, 1)).max())
    if zs.size == 0:
        return []
    lo, hi = zs[0] + 1, zs[-1] - 1
    return [int(round(z)) for z in np.linspace(lo, max(lo, hi), n)]


def _imshow(ax, img2d):
    ax.imshow(np.transpose(img2d, (1, 0, 2)) if img2d.ndim == 3 else img2d.T, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _cycle_colorbar(fig, ax, period, label="lag within cycle (s)"):
    import matplotlib
    import matplotlib.pyplot as plt
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colormaps["twilight"], norm=plt.Normalize(0, period))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label(label, color=INK2)
    cb.outline.set_visible(False)


# ------------------------------------------------------------------ figure 1
def figure_mode(out: dict, k: int, mask3d, voxel_mm, axcodes, meta: dict, path, extra: dict) -> None:
    plt = _style()
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    m = out["measures"][k]
    fc = float(out["f_c"][k])
    period = 1.0 / fc
    q, s = out["q"][:, k], out["s"][:, k]
    hi = m["_hi"]
    amp_q = np.abs(q)
    qref = float(np.nanpercentile(amp_q[hi], 95)) if hi.any() else float(np.nanmax(amp_q))
    lag = m["_lag"]
    lagv = to_ras(_vol(lag, mask3d), axcodes)
    aqv = to_ras(_vol(amp_q / max(qref, 1e-12), mask3d, 0.0), axcodes)
    asv = to_ras(_vol(np.abs(np.nan_to_num(s)) / 0.01, mask3d, 0.0), axcodes)
    lagsv = to_ras(_vol(-np.angle(s) / (2 * np.pi * fc), mask3d), axcodes)
    inside = to_ras(mask3d, axcodes)
    grad = vec_to_ras(to_ras(_vol(m["_grad"], mask3d), axcodes), axcodes)
    etav = to_ras(_vol(extra["eta_voxel"][:, k], mask3d), axcodes)
    vox = np.asarray(voxel_mm, float)[ras_transform(axcodes)[0]]

    fig = plt.figure(figsize=(17, 12.5))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.35, 1], height_ratios=[1, 0.95], hspace=0.30, wspace=0.08,
                  left=0.03, right=0.95, top=0.90, bottom=0.05)
    # A: q montage, 2 x 4
    zs = montage_slices(inside, 8)
    gA = GridSpecFromSubplotSpec(2, 4, gs[0, 0], wspace=0.02, hspace=0.12)
    axA = []
    for i, z in enumerate(zs):
        ax = fig.add_subplot(gA[i // 4, i % 4])
        _imshow(ax, lag_rgb(lagv[:, :, z], aqv[:, :, z], period, inside[:, :, z]))
        st = 3
        X, Y = np.meshgrid(np.arange(0, inside.shape[0], st), np.arange(0, inside.shape[1], st), indexing="ij")
        gx, gy = grad[X, Y, z, 0], grad[X, Y, z, 1]
        n = np.hypot(gx, gy)
        sel = inside[X, Y, z] & (aqv[X, Y, z] > 0.3) & (n > 0) & np.isfinite(n)
        if sel.any():
            ax.quiver(X[sel], Y[sel], (gx / n)[sel], (gy / n)[sel], color=INK, width=0.006, scale=22, alpha=0.8)
        ax.set_title(f"z = {z * vox[2]:.0f} mm", fontsize=7.5, fontweight="normal", color=INK2)
        axA.append(ax)
    if axA:
        axA[0].text(0, 1.2, "A  ΔR2* map q", transform=axA[0].transAxes, fontsize=10, fontweight="bold")
        _cycle_colorbar(fig, axA, period)
    capA = ("Colour = lag within the cycle (cyclic scale); faded = low amplitude (full colour at the 95th percentile\n"
            f"of |q| where the mode is present, {qref:.2g} s⁻¹); arrows = local propagation direction in the slice (phase\n"
            "gradient). Axial slices from inferior (posterior fossa) to superior; RAS display (participant's right on the right).")
    if len(axA) > 4:
        axA[4].text(0, -0.1, capA, transform=axA[4].transAxes, fontsize=8, color=INK2, va="top")
    # B and C: 4 slices each
    gBC = GridSpecFromSubplotSpec(2, 1, gs[0, 1], hspace=0.28)
    z4 = zs[::2] if len(zs) >= 4 else zs
    gB = GridSpecFromSubplotSpec(1, len(z4), gBC[0], wspace=0.02)
    axB = []
    for i, z in enumerate(z4):
        ax = fig.add_subplot(gB[i])
        if out["channels"] == "c0":
            _imshow(ax, lag_rgb(lagv[:, :, z], 0 * aqv[:, :, z], period, inside[:, :, z]))
        else:
            _imshow(ax, lag_rgb(lagsv[:, :, z], asv[:, :, z], period, inside[:, :, z]))
        axB.append(ax)
    if axB:
        axB[0].text(0, 1.08, "B  ΔS0 map s", transform=axB[0].transAxes, fontsize=10, fontweight="bold")
        axB[0].text(0, -0.12, "Same encoding; full colour at a 1 % S0 change: faded means little S0 content."
                    + (" (c0 only: no s map.)" if out["channels"] == "c0" else ""),
                    transform=axB[0].transAxes, fontsize=8, color=INK2, va="top")
    gC = GridSpecFromSubplotSpec(1, len(z4), gBC[1], wspace=0.02)
    axC = []
    im = None
    for i, z in enumerate(z4):
        ax = fig.add_subplot(gC[i])
        e = np.where(inside[:, :, z], etav[:, :, z], np.nan)
        ax.imshow(np.where(inside[:, :, z], 0.0, np.nan).T, origin="lower", cmap="Greys", vmin=0, vmax=10)
        im = ax.imshow(e.T, origin="lower", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        axC.append(ax)
    if axC:
        axC[0].text(0, 1.08, "C  Voxelwise η", transform=axC[0].transAxes, fontsize=10, fontweight="bold")
        cb = fig.colorbar(im, ax=axC, fraction=0.03, pad=0.01)
        cb.set_label("S0 energy fraction", color=INK2)
        cb.outline.set_visible(False)
        axC[0].text(0, -0.12, f"0 = pure ΔR2*, 1 = pure ΔS0. Debiased energies smoothed {extra['eta_fwhm_mm']:.0f} mm "
                    "for display;\nmode η (all voxels, unsmoothed) = "
                    f"{m['eta']:.3f}, ΔR2*-only band [{m['eta_band_R2'][0]:.3f}, {m['eta_band_R2'][1]:.3f}].",
                    transform=axC[0].transAxes, fontsize=8, color=INK2, va="top")
    # D: cycle snapshots (BOLD-positive real part: -Re(q e^{i phi}))
    gD = GridSpecFromSubplotSpec(2, 4, gs[1, 0], wspace=0.04, hspace=0.18)
    refv = extra["ref"]
    cand = np.flatnonzero(hi) if hi.any() else np.flatnonzero(refv)
    vmax_i = cand[np.nanargmax(amp_q[cand])] if cand.size else 0
    cx = to_ras(_vol(np.arange(q.size), mask3d, -1, int), axcodes)
    xyz = np.argwhere(cx == vmax_i)[0]
    qv = to_ras(_vol(np.nan_to_num(q), mask3d, 0, complex), axcodes)
    lim = np.nanpercentile(np.abs(qv[inside]), 99)
    for j, deg in enumerate((0, 90, 180, 270)):
        ph = np.exp(1j * np.radians(deg))
        for r, (img, msk, lab) in enumerate(((qv[:, :, xyz[2]], inside[:, :, xyz[2]], "axial"),
                                             (qv[xyz[0], :, :], inside[xyz[0], :, :], "sagittal"))):
            ax = fig.add_subplot(gD[r, j])
            if r == 0 and j == 0:
                axD0 = ax
            frame = np.where(msk, -np.real(img * ph), np.nan)
            ax.imshow(frame.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(f"{deg}°, t = {deg / 360 * period:.1f} s", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
    axD0.text(0, 1.22, "D  Cycle snapshots (BOLD-positive, one period; slices through the strongest voxel)",
              transform=axD0.transAxes, fontsize=10, fontweight="bold")
    # E: space-time
    axE = fig.add_subplot(gs[1, 1])
    pr = m["_profile"]
    tr = float(meta["tr_s"])
    tc = out["tc"][:, k]
    if pr["lags"].shape[0]:
        nb = pr["lags"].shape[0]
        Qb = np.zeros(nb, complex)
        np.add.at(Qb, pr["bin_index"][pr["bin_index"] >= 0], np.nan_to_num(q)[pr["bin_index"] >= 0])
        Qb = Qb / np.maximum(pr["n"], 1)
        nt = int(min(tc.size, max(4 * period, 200.0) / tr))
        stp = -np.real(Qb[:, None] * tc[None, :nt])
        stp = stp / max(np.nanmax(np.abs(stp)), 1e-12)
        stp[pr["n"] < 5] = np.nan
        t = np.arange(nt) * tr
        axE.imshow(stp, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
                   extent=[0, nt * tr, 0, nb * (pr["centres_mm"][1] - pr["centres_mm"][0] if nb > 1 else 6.0)])
        v = m["speed_mm_s"]
        dmax = pr["centres_mm"][-1] if nb else 0
        if np.isfinite(v) and v < 1000:
            first = np.flatnonzero(pr["n"] >= 5)
            b0 = first[0] if first.size else 0
            kk = int(round(period / tr))
            t0 = t[int(np.nanargmax(stp[b0, :max(1, min(kk, nt))]))] if nt else 0.0
            t0 -= pr["centres_mm"][b0] / v
            for c in np.arange(-int(dmax / v / period) - 1, nt * tr / period + 1):
                tt = t0 + c * period
                axE.plot([tt, tt + dmax / v], [0, dmax], color=INK, lw=1, ls="--", alpha=0.6)
            axE.set_xlim(0, nt * tr)
            axE.set_ylim(0, nb * (pr["centres_mm"][1] - pr["centres_mm"][0] if nb > 1 else 6.0))
        ci = (out.get("bootstrap") or {}).get("ci", {}).get("speed_mm_s")
        cit = f" [{ci[k][0]:.1f}, {ci[k][1]:.1f}]" if ci and k < len(ci) else ""
        axE.text(0.97, 0.04, f"fitted speed {v:.1f}{cit} mm/s\nfit quality {m['fit_quality']:.2f}, R = {m['R']:.2f}",
                 transform=axE.transAxes, ha="right", va="bottom", fontsize=8.5,
                 bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.85))
    else:
        axE.text(0.5, 0.5, "too few voxels for a path", transform=axE.transAxes, ha="center")
    axE.set_title("E  Space–time plot along the fitted propagation axis")
    axE.set_xlabel("time (s)")
    axE.set_ylabel("distance along the axis (mm)")
    axE.grid(False)
    _header(fig, f"Wave mode {k + 1}", meta, out, k)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _header(fig, title, meta, out, k):
    fc = float(out["f_c"][k])
    m = out["measures"][k]
    an = m.get("above_null")
    fig.text(0.01, 0.975, f"{title}: {meta.get('participant', '?')}", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.952,
             f"{meta.get('data_type', 'real data')}. f_c = {fc:.3f} Hz (period {1 / fc:.1f} s); {out['n_volumes_used']} "
             f"volumes; slice-timing correction {'applied' if out['slice_timing_corrected'] else 'NOT applied'}; "
             f"normalization {out['voxel_scaling']}; channels {out['channels']}"
             f"{', ' + meta['nuisance'] if meta.get('nuisance') else ''}; variance {100 * m['var_frac_stacked']:.1f} % "
             f"of the band-passed channels; "
             + ("energy above the family-wise null" if an else "energy NOT above the family-wise null" if an is not None
                else "no energy null computed") + ".",
             fontsize=8.5, color=INK2)


# ------------------------------------------------------------------ figure 2
def figure_diagnostics(out: dict, k: int, mask3d, voxel_mm, axcodes, meta: dict, path, extra: dict) -> None:
    plt = _style()
    m = out["measures"][k]
    fc = float(out["f_c"][k])
    te = np.asarray(meta["te_ms"], float)
    pr = m["_profile"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.12, hspace=0.42, wspace=0.28)
    # A: lag per echo along the axis
    ax = axs[0, 0]
    bci = (out.get("bootstrap") or {}).get("profile_ci") or []
    if pr["lags"].shape[0]:
        x = pr["centres_mm"]
        for e in range(te.size):
            ax.plot(x, pr["lags"][:, e], "o-", ms=3.5, lw=1.2, color=ECHO_RAMP[e % 4], label=f"TE {te[e]:.1f} ms")
            if k < len(bci) and np.size(bci[k]):
                ax.fill_between(x, bci[k][0][:, e], bci[k][1][:, e], color=ECHO_RAMP[e % 4], alpha=0.15, lw=0)
        ax.legend(loc="lower right", fontsize=8)
        sp = m["speed_mm_s"]
        ax.text(0.03, 0.97, f"binned ({pr['centres_mm'][1] - pr['centres_mm'][0] if x.size > 1 else 0:.0f} mm), "
                "bands = 95 % bootstrap. Fitted speed "
                f"{sp:.1f} mm/s.\nSame lag at every echo: one ΔR2* source. Echoes ordered\nin lag: S0 and ΔR2* parts "
                "with different timing.", transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.9))
    ax.set_xlabel("distance along the axis (mm)")
    ax.set_ylabel("lag (s)")
    ax.set_title("A  Lag per echo along the axis")
    # B: TE4 - TE1 lag against its null
    ax = axs[0, 1]
    sel = m["_hi"] & np.isfinite(m["_dlag"])
    d = m["_dlag"][sel]
    nd = m["_null_dlag_draws"]
    half = 0.5 / fc
    lim = max(float(np.nanpercentile(np.abs(d), 99.5)) if d.size else 0.0,
              float(np.nanpercentile(np.abs(nd), 99.5)) if nd.size else 0.0, 0.1)
    lim = min(max(lim, 0.2), half)
    bins = np.linspace(-lim, lim, 41)
    if nd.size:
        ax.hist(np.clip(nd, -lim, lim), bins=bins, density=True, histtype="stepfilled", color=GRID, edgecolor=INK2,
                lw=0.8, label="noise null (pure ΔR2*, same q and noise)")
    if d.size:
        ax.hist(np.clip(d, -lim, lim), bins=bins, density=True, color=AQUA if m["te_lag_fraction_beyond_null"] > 0.1 else BLUE,
                alpha=0.7, edgecolor=SURF, label="voxels where the mode is present")
    ax.axvline(0, color=INK2, lw=0.8)
    ci = (out.get("bootstrap") or {}).get("ci", {}).get("te_lag_median_s")
    cit = f" [{ci[k][0]:.2f}, {ci[k][1]:.2f}]" if ci and k < len(ci) else ""
    ax.text(0.98, 0.84, f"median {m['te_lag_median_s']:.2f}{cit} s\n{100 * m['te_lag_fraction_beyond_null']:.0f} % of "
            "voxels beyond their null\n≈ 0: one ΔR2* source; ≠ 0: S0 and ΔR2*\nparts with different timing",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.set_xlabel("lag at TE4 minus lag at TE1 (s)")
    ax.set_ylabel("density")
    ax.set_title("B  TE dependence of lag")
    # C: eta along the axis
    ax = axs[0, 2]
    if pr["eta"].size:
        ax.plot(pr["centres_mm"], pr["eta"], color=AQUA if m["eta"] > m["eta_band_R2"][1] else BLUE)
    b = m["eta_band_R2"]
    if np.isfinite(b).all():
        ax.axhspan(b[0], b[1], color=BLUE, alpha=0.12, lw=0, label="ΔR2*-only band (matched SNR)")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("distance along the axis (mm)")
    ax.set_ylabel("S0 energy fraction η")
    ax.set_title(f"C  η along the axis (mode η = {m['eta']:.3f})")
    # D: direction histogram, polar, in the plane of the fitted direction
    axs[1, 0].remove()
    ax = fig.add_subplot(2, 3, 4, projection="polar")
    dvec = np.asarray(m["direction"], float)
    g = m["_grad"][m["_hi"]]
    w = np.abs(out["q"][m["_hi"], k]) ** 2
    ok = np.isfinite(g).all(1) & (np.linalg.norm(g, axis=1) > 0)
    if ok.sum() and np.isfinite(dvec).all() and np.linalg.norm(dvec) > 0:
        e1 = dvec / np.linalg.norm(dvec)
        ax_i = int(np.argmin(np.abs(e1)))
        e2 = np.eye(3)[ax_i] - e1 * e1[ax_i]
        e2 /= np.linalg.norm(e2)
        u = g[ok] / np.linalg.norm(g[ok], axis=1, keepdims=True)
        ang = np.arctan2(u @ e2, u @ e1)
        h, edges = np.histogram(ang, bins=24, range=(-np.pi, np.pi), weights=w[ok])
        ax.bar(edges[:-1], h / max(h.max(), 1e-300), width=np.diff(edges), align="edge", color=BLUE, alpha=0.75,
               edgecolor=SURF)
        names = "xyz"
        ax.set_title(f"D  Propagation direction (0° = fitted, {names[ax_i]}-axis component at 90°)", pad=18)
    else:
        ax.set_title("D  Propagation direction", pad=18)
    ax.set_yticklabels([])
    ax.text(0.5, -0.2, f"R = {m['R']:.2f} (directional coherence, 1 = one direction everywhere)\nfitted direction "
            f"(array axes) [{', '.join(f'{x:.2f}' for x in dvec)}]", transform=ax.transAxes, ha="center", fontsize=8,
            color=INK2, va="top")
    # E: echo phasors in three representative voxels
    ax = axs[1, 1]
    tes = te / 1000.0
    marks = ("o", "s", "D")
    for i, ((lab, vi), mk, col) in enumerate(zip(extra["rep_voxels"][k], marks, (BLUE, AQUA, ORANGE))):
        if vi is None:
            continue
        sv = out["s"][vi, k]
        sv = 0.0 if not np.isfinite(sv) else sv
        z = sv - tes * out["q"][vi, k]
        z = z / max(np.abs(z).max(), 1e-30)
        ax.plot(z.real, z.imag, "-", color=col, lw=1.2)
        for e in range(te.size):
            ax.plot(z.real[e], z.imag[e], mk, ms=8, color=ECHO_RAMP[e % 4], mec=col, mew=1.5)
        ax.annotate(lab, (z.real[-1], z.imag[-1]), textcoords="offset points", xytext=(8, 4 + 12 * (i - 1)),
                    fontsize=8, color=col)
    ax.axhline(0, color=GRID, lw=2)
    ax.axvline(0, color=GRID, lw=2)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("real (normalized)")
    ax.set_ylabel("imaginary")
    ax.set_title("E  Echo phasors z_e = s − TE_e q")
    ax.text(0, -0.15, "TE1 (light) to TE4 (dark). Points on a line through the origin share one phase;\n"
            "a line that misses the origin means a mixture with different timing.", transform=ax.transAxes,
            fontsize=8, color=INK2, va="top")
    # F: power spectrum of the time course
    ax = axs[1, 2]
    tc = np.real(out["tc"][:, k])
    tr = float(meta["tr_s"])
    f = np.fft.rfftfreq(tc.size, tr)
    p = np.abs(np.fft.rfft(tc - tc.mean())) ** 2
    ax.semilogy(f, p / p.max(), color=BLUE, lw=1.2)
    band = meta.get("band_hz", (0.01, 0.1))
    ax.axvspan(band[0], band[1], color=GRID, alpha=0.6, lw=0, label="analysis band")
    ax.axvline(fc, color=ORANGE, lw=1.5, ls="--", label=f"f_c = {fc:.3f} Hz")
    ax.set_xlim(0, min(f.max(), 3 * band[1]))
    ax.set_ylim(1e-4, 1.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("relative power")
    ax.set_title("F  Power spectrum of the mode time course")
    _header(fig, f"Multi-echo diagnostics, wave mode {k + 1}", meta, out, k)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def representative_voxels(out: dict, k: int, refv, eta_vox, inferior) -> list:
    """(label, voxel index): the highest-|q| reference voxel, the highest-eta voxel where the mode is present,
    and the highest-|q| posterior-fossa (inferior third) voxel where the mode is present."""
    m = out["measures"][k]
    a = np.abs(np.nan_to_num(out["q"][:, k]))
    hi = m["_hi"]
    pick = lambda sel, score: (int(np.flatnonzero(sel)[np.nanargmax(score[sel])]) if sel.any()
                               and np.isfinite(score[sel]).any() else None)
    v1 = pick(refv & hi if (refv & hi).any() else refv, a)
    e = np.where(np.isfinite(eta_vox), eta_vox, -1.0)
    v2 = pick(hi, e)
    pf = inferior & hi if (inferior & hi).any() else inferior
    v3 = pick(pf, a)
    return [("strongest", v1), ("highest η", v2), ("posterior fossa", v3)]
