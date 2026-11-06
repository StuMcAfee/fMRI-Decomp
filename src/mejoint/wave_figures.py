"""The two per-participant wave figures (docs/WAVES_BRIEF.md section 5), styled after the mock-ups in
docs/mockups/waves/, adapted from a flat sheet to volumes, with phase maps (user decision 2026-11-05).

  wave_mode_k.png         A relative phase of the dR2* map q (cyclic hsv, 0 = reference voxels, + = later), faded
                          by each voxel's explained fraction, gradient arrows, axial montage with the posterior
                          fossa; B the direct Hilbert relative phase (check map), faded by phase locking; C the
                          phase of the dS0 map s, faded against a 1 % S0 change; D voxelwise eta; E cycle snapshots;
                          F space-time plot of the band-passed DATA along the fitted axis with the mode's crest
                          lines; G the mode's own reconstruction (rank one: propagation is built in).
  wave_diagnostics_k.png  A phase lag per echo along the axis (bootstrap bands); B TE4 - TE1 lag against its noise
                          null; C dS0 vs dR2* relative phase; D sign-opposed fraction by CSF fraction; E eta along
                          the axis; F direction histogram (polar) with R; G echo phasors; H motion correlation;
                          I power spectrum of the mode time course.

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
def phase_cmap():
    """hsv, shifted so that phase 0 (the reference voxels) is red; for a colour bar from -180 to 180 degrees."""
    import matplotlib
    from matplotlib.colors import ListedColormap
    deg = np.linspace(-180, 180, 361)
    return ListedColormap(matplotlib.colormaps["hsv"](np.mod(deg, 360) / 360))


def phase_rgb(phase_deg, alpha, inside):
    """Cyclic hsv colour of a phase (degrees), faded towards the brain background by alpha in [0, 1] (hsv never
    reaches the background colour, so faded is never confused with a phase); outside the brain the background."""
    import matplotlib
    import matplotlib.colors as mc
    rgb = matplotlib.colormaps["hsv"](np.mod(np.nan_to_num(phase_deg), 360) / 360)[..., :3]
    a = np.clip(np.nan_to_num(alpha), 0, 1)[..., None]
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
    _bare(ax)


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _phase_colorbar(fig, axes, period, label="phase lag (degrees; + = later)"):
    import matplotlib.pyplot as plt
    sm = plt.cm.ScalarMappable(cmap=phase_cmap(), norm=plt.Normalize(-180, 180))
    cb = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.01, ticks=[-180, -90, 0, 90, 180])
    cb.set_label(label, color=INK2)
    cb.outline.set_visible(False)


def _ref_scale(x, refv):
    v = np.asarray(x, float)[refv]
    v = v[np.isfinite(v)]
    return max(float(np.percentile(v, 95)), 1e-12) if v.size else 1.0


def crest_lines(tc_seg, phi_b, valid):
    """Crest lines of the BOLD-positive field Re(Q_b e^{i pi} z(t)) per distance bin b: the times where the
    instantaneous phase arg z(t) + phi_b crosses 2 pi n upwards (phi_b = arg Q_b + pi, unwrapped along the bins so
    that one n is one crest). Returns a list of (bin indices, times in frames)."""
    th = np.unwrap(np.angle(tc_seg))
    bins = np.flatnonzero(valid)
    if bins.size == 0:
        return []
    ph = np.unwrap(phi_b[bins])
    lo = int(np.floor((th.min() + ph.min()) / (2 * np.pi)))
    hi = int(np.ceil((th.max() + ph.max()) / (2 * np.pi)))
    lines = []
    for n in range(lo, hi + 1):
        tt = np.full(bins.size, np.nan)
        for i, p in enumerate(ph):
            f = th + p - 2 * np.pi * n
            c = np.flatnonzero((f[:-1] < 0) & (f[1:] >= 0))
            if c.size:
                j = c[0]
                tt[i] = j + (-f[j]) / (f[j + 1] - f[j])
        if np.isfinite(tt).sum() >= 2:
            lines.append((bins, tt))
    return lines


# ------------------------------------------------------------------ figure 1
def figure_mode(out: dict, k: int, mask3d, voxel_mm, axcodes, meta: dict, path, extra: dict) -> None:
    plt = _style()
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    m = out["measures"][k]
    fc = float(out["f_c"][k])
    period = 1.0 / fc
    q, s = out["q"][:, k], out["s"][:, k]
    refv = extra["ref"]
    hi = m["_hi"]
    ef, plv = out["explained_fraction"][:, k], out["plv"][:, k]
    ef_ref, plv_ref = _ref_scale(ef, refv), _ref_scale(plv, refv)
    inside = to_ras(mask3d, axcodes)
    R = lambda x, fill=np.nan, dt=float: to_ras(_vol(x, mask3d, fill, dt), axcodes)
    ph_q = R(-np.degrees(np.angle(q)))                                    # phase lag, + = later
    ph_h = R(-np.degrees(out["hilbert_phase"][:, k]))
    ph_s = R(-np.degrees(np.angle(s)))
    a_q, a_h = R(np.nan_to_num(ef) / ef_ref, 0.0), R(np.nan_to_num(plv) / plv_ref, 0.0)
    a_s = R(np.abs(np.nan_to_num(s)) / 0.01, 0.0)
    grad = vec_to_ras(to_ras(_vol(m["_grad"], mask3d), axcodes), axcodes)
    etav = R(extra["eta_voxel"][:, k])
    vox = np.asarray(voxel_mm, float)[ras_transform(axcodes)[0]]

    fig = plt.figure(figsize=(17, 15))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.3, 1], height_ratios=[1.05, 1], hspace=0.22, wspace=0.10,
                  left=0.03, right=0.95, top=0.92, bottom=0.04)
    # A: phase of q, 2 x 4 montage
    zs = montage_slices(inside, 8)
    gA = GridSpecFromSubplotSpec(2, 4, gs[0, 0], wspace=0.02, hspace=0.12)
    axA = []
    for i, z in enumerate(zs):
        ax = fig.add_subplot(gA[i // 4, i % 4])
        _imshow(ax, phase_rgb(ph_q[:, :, z], a_q[:, :, z], inside[:, :, z]))
        st = 3
        X, Y = np.meshgrid(np.arange(0, inside.shape[0], st), np.arange(0, inside.shape[1], st), indexing="ij")
        gx, gy = grad[X, Y, z, 0], grad[X, Y, z, 1]
        n = np.hypot(gx, gy)
        sel = inside[X, Y, z] & (a_q[X, Y, z] > 0.3) & (n > 0) & np.isfinite(n)
        if sel.any():
            ax.quiver(X[sel], Y[sel], (gx / n)[sel], (gy / n)[sel], color=INK, width=0.006, scale=22, alpha=0.8)
        ax.set_title(f"z = {z * vox[2]:.0f} mm", fontsize=7.5, fontweight="normal", color=INK2)
        axA.append(ax)
    if axA:
        axA[0].text(0, 1.2, "A  Relative phase of the ΔR2* map q (from the mode)", transform=axA[0].transAxes,
                    fontsize=10, fontweight="bold")
        _phase_colorbar(fig, axA, period)
    if len(axA) > 4:
        axA[4].text(0, -0.08, "Colour = phase lag relative to the reference voxels (0 = their weighted mean; cyclic hsv); "
                    "faded = the mode explains\nlittle of the voxel's band-passed signal (full colour at the 95th "
                    f"percentile of the reference voxels, {ef_ref:.2g}); arrows = local\npropagation direction in the "
                    f"slice (phase gradient). ±180° = ±{period / 2:.1f} s at f_c. Axial slices inferior (posterior fossa) "
                    "to superior; RAS display (right on the right).", transform=axA[4].transAxes, fontsize=8, color=INK2, va="top")
    # B, C, D: 4 slices each
    gR = GridSpecFromSubplotSpec(3, 1, gs[0, 1], hspace=0.42)
    z4 = zs[::2] if len(zs) >= 4 else zs

    def row(g, imgs, title, note):
        gg = GridSpecFromSubplotSpec(1, max(len(z4), 1), g, wspace=0.02)
        axs = []
        for i, z in enumerate(z4):
            ax = fig.add_subplot(gg[i])
            imgs(ax, z)
            axs.append(ax)
        if axs:
            axs[0].text(0, 1.1, title, transform=axs[0].transAxes, fontsize=10, fontweight="bold")
            axs[0].text(0, -0.06, note, transform=axs[0].transAxes, fontsize=7.5, color=INK2, va="top")
        return axs

    row(gR[0], lambda ax, z: _imshow(ax, phase_rgb(ph_h[:, :, z], a_h[:, :, z], inside[:, :, z])),
        "B  Direct Hilbert relative phase (check)",
        "Each voxel's c0 phase minus the mode's phase, averaged over time; faded = weak phase locking\n"
        f"(full colour at PLV {plv_ref:.2f}). Should match A where the mode dominates.")
    if out["channels"] == "c0":
        row(gR[1], lambda ax, z: _imshow(ax, phase_rgb(ph_s[:, :, z], 0 * a_s[:, :, z], inside[:, :, z])),
            "C  ΔS0 map s", "c0 only: no s map.")
    else:
        row(gR[1], lambda ax, z: _imshow(ax, phase_rgb(ph_s[:, :, z], a_s[:, :, z], inside[:, :, z])),
            "C  Phase of the ΔS0 map s", "Same colours; full colour at a 1 % S0 change.")
    ims = []

    def eta_img(ax, z):
        ax.imshow(np.where(inside[:, :, z], 0.0, np.nan).T, origin="lower", cmap="Greys", vmin=0, vmax=10)
        ims.append(ax.imshow(np.where(inside[:, :, z], etav[:, :, z], np.nan).T, origin="lower", cmap="Blues",
                             vmin=0, vmax=1, interpolation="nearest"))
        _bare(ax)
    axD = row(gR[2], eta_img, "D  Voxelwise η",
              f"0 = pure ΔR2*, 1 = pure ΔS0; energies smoothed {extra['eta_fwhm_mm']:.0f} mm for display. Mode η = "
              f"{m['eta']:.3f};\nthermal-only ΔR2* band [{m['eta_band_R2'][0]:.3f}, {m['eta_band_R2'][1]:.3f}] "
              "(too narrow on real data).")
    if ims:
        cb = fig.colorbar(ims[-1], ax=axD, fraction=0.03, pad=0.01)
        cb.set_label("S0 energy fraction", color=INK2)
        cb.outline.set_visible(False)
    # E: cycle snapshots (BOLD-positive real part: -Re(q e^{i phi}))
    gE = GridSpecFromSubplotSpec(2, 4, gs[1, 0], wspace=0.04, hspace=0.18)
    amp_q = np.abs(np.nan_to_num(q))
    cand = np.flatnonzero(hi & refv) if (hi & refv).any() else (np.flatnonzero(hi) if hi.any() else np.flatnonzero(refv))
    vmax_i = cand[np.nanargmax(amp_q[cand])] if cand.size else 0
    cx = R(np.arange(q.size), -1, int)
    xyz = np.argwhere(cx == vmax_i)[0]
    qv = R(np.nan_to_num(q), 0, complex)
    lim = np.nanpercentile(np.abs(qv[inside]), 99)
    axE0 = None
    for j, deg in enumerate((0, 90, 180, 270)):
        ph = np.exp(1j * np.radians(deg))
        for r, (img, msk, lab) in enumerate(((qv[:, :, xyz[2]], inside[:, :, xyz[2]], "axial"),
                                             (qv[xyz[0], :, :], inside[xyz[0], :, :], "sagittal"))):
            ax = fig.add_subplot(gE[r, j])
            axE0 = ax if axE0 is None else axE0
            ax.imshow(np.where(msk, -np.real(img * ph), np.nan).T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                      interpolation="nearest")
            _bare(ax)
            if r == 0:
                ax.set_title(f"{deg}°, t = {deg / 360 * period:.1f} s", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
    axE0.text(0, 1.22, "E  Cycle snapshots (BOLD-positive; slices through the strongest reference voxel)",
              transform=axE0.transAxes, fontsize=10, fontweight="bold")
    # F (data) and G (reconstruction): space-time along the fitted axis, with the mode's crest lines
    gFG = GridSpecFromSubplotSpec(2, 1, gs[1, 1], height_ratios=[2.2, 1], hspace=0.35)
    axF, axG = fig.add_subplot(gFG[0]), fig.add_subplot(gFG[1])
    pr = m["_profile"]
    tr = float(meta["tr_s"])
    tc = out["tc"][:, k]
    nb = pr["lags"].shape[0]
    if nb:
        binw = pr["centres_mm"][1] - pr["centres_mm"][0] if nb > 1 else float(pr["centres_mm"][0] * 2)
        Qb = np.zeros(nb, complex)
        sel = pr["bin_index"] >= 0
        np.add.at(Qb, pr["bin_index"][sel], np.nan_to_num(q)[sel])
        valid = pr["n"] >= 5
        nt = int(min(tc.size, max(5 * period, 240.0) / tr))
        ext = [0, nt * tr, 0, nb * binw]
        data = m.get("_st_data")
        if data is not None:
            axF.imshow(data[:, :nt], origin="lower", aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                       interpolation="nearest", extent=ext)
        rec = -np.real(Qb[:, None] * tc[None, :nt])
        rec = rec / np.maximum(np.abs(rec).max(1, keepdims=True), 1e-30)
        rec[~valid] = np.nan
        axG.imshow(rec, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
                   extent=ext)
        for bins, tt in crest_lines(tc[:nt], np.angle(Qb) + np.pi, valid):
            for ax in (axF, axG):
                ax.plot(tt * tr, pr["centres_mm"][bins], color=INK, lw=1, ls="--", alpha=0.7)
        for ax in (axF, axG):
            ax.set_xlim(0, nt * tr)
            ax.set_ylim(0, nb * binw)
        ci = (out.get("bootstrap") or {}).get("ci", {}).get("speed_mm_s")
        cit = f" [{ci[k][0]:.1f}, {ci[k][1]:.1f}]" if ci and k < len(ci) else ""
        axF.text(0.99, 0.02, f"plane-wave fit: {m['speed_mm_s']:.1f}{cit} mm/s, fit quality {m['fit_quality']:.2f}, "
                 f"R = {m['R']:.2f}", transform=axF.transAxes, ha="right", va="bottom", fontsize=8,
                 bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.85))
    else:
        axF.text(0.5, 0.5, "too few voxels for a path", transform=axF.transAxes, ha="center")
    axF.set_title("F  Space–time: band-passed data along the fitted axis (BOLD-positive, rows z-scored)")
    axF.set_ylabel("distance along the axis (mm)")
    axG.set_title("G  The mode's reconstruction (one map × one time course: propagation built in)", fontsize=9)
    axG.set_xlabel("time (s)")
    axG.set_ylabel("mm")
    axF.text(0, -0.1, "Dashed = the mode's crest lines from its instantaneous phase. Rows average the voxels of each "
             "6 mm bin\nwhose mode phase is within 90° of the bin's mean (opposite signs would cancel).",
             transform=axF.transAxes, fontsize=7.5, color=INK2, va="top")
    for ax in (axF, axG):
        ax.grid(False)
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
             f"{'; band-passed motion projected out' if out.get('motion_orthogonal') else '; motion NOT projected out'}"
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
    period = 1.0 / fc
    te = np.asarray(meta["te_ms"], float)
    pr = m["_profile"]
    fig, axs = plt.subplots(3, 3, figsize=(16, 14.5))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.91, bottom=0.07, hspace=0.48, wspace=0.30)
    to_deg = lambda lag_s: np.asarray(lag_s) * fc * 360.0
    # A: phase lag per echo along the axis (degrees, seconds on the right)
    ax = axs[0, 0]
    bci = (out.get("bootstrap") or {}).get("profile_ci") or []
    if pr["lags"].shape[0]:
        x = pr["centres_mm"]
        for e in range(te.size):
            ax.plot(x, to_deg(pr["lags"][:, e]), "o-", ms=3.5, lw=1.2, color=ECHO_RAMP[e % 4], label=f"TE {te[e]:.1f} ms")
            if k < len(bci) and np.size(bci[k]):
                ax.fill_between(x, to_deg(bci[k][0][:, e]), to_deg(bci[k][1][:, e]), color=ECHO_RAMP[e % 4], alpha=0.15,
                                lw=0)
        ax.legend(loc="lower right", fontsize=8)
        sec = ax.secondary_yaxis("right", functions=(lambda d: d / 360.0 / fc, lambda t: t * fc * 360.0))
        sec.set_ylabel("s at f_c", color=INK2)
        ax.text(0.03, 0.97, "6 mm bins, bands = 95 % bootstrap. Same phase at every\necho: one ΔR2* source; echoes "
                "ordered: parts with\ndifferent timing.", transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.9))
    ax.set_xlabel("distance along the axis (mm)")
    ax.set_ylabel("phase lag (degrees)")
    ax.set_title("A  Phase lag per echo along the axis")
    # B: TE4 - TE1 lag against its null (not clipped)
    ax = axs[0, 1]
    sel = m["_hi"] & np.isfinite(m["_dlag"])
    d, nd = m["_dlag"][sel], m["_null_dlag_draws"]
    lim = max(float(np.nanmax(np.abs(d))) if d.size else 0.0, float(np.nanmax(np.abs(nd))) if nd.size else 0.0, 0.2)
    bins = np.linspace(-lim, lim, 61)
    if nd.size:
        ax.hist(nd, bins=bins, density=True, histtype="stepfilled", color=GRID, edgecolor=INK2, lw=0.8,
                label="noise null (pure ΔR2*)")
    if d.size:
        ax.hist(d, bins=bins, density=True, color=BLUE, alpha=0.7, edgecolor=SURF, label="voxels with the mode")
    for v in (-period / 2, period / 2):
        if abs(v) <= lim:
            ax.axvline(v, color=ORANGE, lw=1, ls=":")
    ax.axvline(0, color=INK2, lw=0.8)
    ci = (out.get("bootstrap") or {}).get("ci", {}).get("te_lag_median_s")
    cit = f" [{ci[k][0]:.2f}, {ci[k][1]:.2f}]" if ci and k < len(ci) else ""
    ax.text(0.98, 0.80, f"median {m['te_lag_median_s']:.2f}{cit} s\n{100 * m['te_lag_fraction_beyond_null']:.0f} % beyond "
            "their null\ndotted = ± half a period (sign flips)", transform=ax.transAxes, ha="right", va="top",
            fontsize=8)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    ax.set_xlabel("lag at TE4 minus lag at TE1 (s)")
    ax.set_ylabel("density")
    ax.set_title("B  TE dependence of lag (brief 4.7)")
    # C: dS0 vs dR2* relative phase
    ax = axs[0, 2]
    s0 = m["_s0r2"]
    dd = np.degrees(s0["_delta"][s0["_ok"]])
    if dd.size:
        b = np.linspace(-180, 180, 37)
        ax.hist(dd, bins=b, color=AQUA, edgecolor=SURF)
        ax.axvspan(-180, -90, color=ORANGE, alpha=0.08, lw=0)
        ax.axvspan(90, 180, color=ORANGE, alpha=0.08, lw=0, label="sign-opposed")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.text(0.98, 0.97, f"{s0['n']} voxels with both parts above noise\nsign-opposed: "
            f"{100 * s0['fraction_opposed']:.0f} %" if s0["n"] >= 20 else
            f"{s0['n']} voxels with both parts above noise\n(fewer than 20: not summarized)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    if np.isfinite(s0.get("null_n_expected", np.nan)):
        ax.text(0.98, 0.62, f"pure-ΔR2* null: {s0['null_n_expected']:.0f} voxels expected"
                + (f",\n{100 * s0['null_fraction_opposed']:.0f} % of them opposed" if np.isfinite(s0["null_fraction_opposed"])
                   else ""), transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    if s0["n"] >= 20:
        ax.text(0.98, 0.80, f"timing of ΔS0 (folded): median {s0['timing_median_s']:.2f} s\n(+ = ΔS0 later; ±90° = "
                f"±{period / 4:.1f} s)", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.set_xlabel("phase of s relative to −q (degrees)")
    ax.set_ylabel("voxels")
    ax.set_title("C  ΔS0 relative to BOLD-positive ΔR2*")
    # D: significant dS0 by CSF fraction, same sign vs opposite sign
    ax = axs[1, 0]
    fl = s0.get("by_f_long")
    if fl:
        xs = np.arange(len(fl))
        same = np.nan_to_num([r["fraction_same_sign"] for r in fl])
        opp = np.nan_to_num([r["fraction_opposed"] for r in fl])
        ax.bar(xs - 0.2, same, width=0.4, color=AQUA, label="same sign as BOLD-positive ΔR2*")
        ax.bar(xs + 0.2, opp, width=0.4, color=ORANGE, label="opposite sign")
        top = max(float(np.max(np.r_[same, opp])), 0.05)
        for xi, r in zip(xs, fl):
            ax.text(xi, top * 1.08, f"n {r['n_present']}", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(xs, [r["f_long"] for r in fl])
        ax.set_ylim(0, top * 1.35)
        ax.legend(loc="upper left", fontsize=7.5)
        ax.text(0.98, 0.80, "partial volume predicts a rising\nsame-sign fraction, not opposite signs",
                transform=ax.transAxes, fontsize=7.5, va="top", ha="right", color=INK2)
    else:
        ax.text(0.5, 0.5, "no tissue classes given (--tissue-dir)", transform=ax.transAxes, ha="center", color=INK2)
    ax.set_xlabel("CSF fraction f_long")
    ax.set_ylabel("fraction of voxels with the mode")
    ax.set_title("D  Significant ΔS0 by CSF fraction")
    # E: eta along the axis
    ax = axs[1, 1]
    if pr["eta"].size:
        ax.plot(pr["centres_mm"], pr["eta"], color=AQUA if m["eta"] > m["eta_band_R2"][1] else BLUE)
    bnd = m["eta_band_R2"]
    if np.isfinite(bnd).all():
        ax.axhspan(bnd[0], bnd[1], color=BLUE, alpha=0.12, lw=0, label="ΔR2*-only band (thermal, matched SNR)")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("distance along the axis (mm)")
    ax.set_ylabel("S0 energy fraction η")
    ax.set_title(f"E  η along the axis (mode η = {m['eta']:.3f})")
    # F: direction histogram, polar, in the plane of the fitted direction
    axs[1, 2].remove()
    ax = fig.add_subplot(3, 3, 6, projection="polar")
    dvec = np.asarray(m["direction"], float)
    g = m["_grad"][m["_hi"]]
    w = np.nan_to_num(out["explained_fraction"][m["_hi"], k])
    okg = np.isfinite(g).all(1) & (np.linalg.norm(g, axis=1) > 0)
    if okg.sum() and np.isfinite(dvec).all() and np.linalg.norm(dvec) > 0:
        e1 = dvec / np.linalg.norm(dvec)
        ax_i = int(np.argmin(np.abs(e1)))
        e2 = np.eye(3)[ax_i] - e1 * e1[ax_i]
        e2 /= np.linalg.norm(e2)
        u = g[okg] / np.linalg.norm(g[okg], axis=1, keepdims=True)
        ang = np.arctan2(u @ e2, u @ e1)
        h, edges = np.histogram(ang, bins=24, range=(-np.pi, np.pi), weights=w[okg])
        ax.bar(edges[:-1], h / max(h.max(), 1e-300), width=np.diff(edges), align="edge", color=BLUE, alpha=0.75,
               edgecolor=SURF)
        ax.set_title(f"F  Local directions (0° = fitted, {'xyz'[ax_i]}-axis part at 90°)", pad=18)
    else:
        ax.set_title("F  Local directions", pad=18)
    ax.set_yticklabels([])
    ax.text(0.5, -0.2, f"R = {m['R']:.2f} (1 = one direction everywhere)\nfitted direction towards later voxels: "
            f"{m.get('direction_label') or '?'} (RAS {m.get('direction_ras', '?')})", transform=ax.transAxes,
            ha="center", fontsize=8, color=INK2, va="top")
    # G: echo phasors
    ax = axs[2, 0]
    tes = te / 1000.0
    for i, ((lab, vi), mk, col) in enumerate(zip(extra["rep_voxels"][k], ("o", "s", "D"), (BLUE, AQUA, ORANGE))):
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
    ax.set_title("G  Echo phasors z_e = s − TE_e q")
    ax.text(0, -0.2, "TE1 (light) to TE4 (dark). A line through the origin: one phase.\nCrossing the origin: "
            "opposite signs. Missing it: different timing.", transform=ax.transAxes, fontsize=8, color=INK2, va="top")
    # H: motion correlation
    ax = axs[2, 1]
    mot = out.get("motion")
    if mot:
        r = np.asarray(mot["r"])[k]
        ax.bar(np.arange(r.size), r, color=BLUE, alpha=0.8)
        ax.axhline(mot["null95"], color=ORANGE, ls="--", lw=1.2, label="null 95 % (largest of 6)")
        lags = np.asarray(mot.get("lag_s", [[np.nan] * r.size] * (k + 1)))[k]
        for xi, (rv, lg) in enumerate(zip(r, lags)):
            if np.isfinite(lg):
                ax.text(xi, rv + 0.02, f"{lg:+.1f} s", ha="center", fontsize=7, color=INK2)
        ax.text(0.02, 0.97, "labels: lag of the signal after the parameter\n(folded to ± a quarter period)",
                transform=ax.transAxes, fontsize=7.5, va="top", color=INK2)
        ax.set_xticks(np.arange(r.size), [n.replace("_", "\n") for n in mot["names"]])
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "no motion parameters found", transform=ax.transAxes, ha="center", color=INK2)
    ax.set_ylabel("|correlation| with the mode")
    ax.set_title("H  Mode time course vs motion parameters")
    # I: power spectrum
    ax = axs[2, 2]
    tcr = np.real(out["tc"][:, k])
    tr = float(meta["tr_s"])
    f = np.fft.rfftfreq(tcr.size, tr)
    p = np.abs(np.fft.rfft(tcr - tcr.mean())) ** 2
    ax.semilogy(f, p / p.max(), color=BLUE, lw=1.2)
    band = meta.get("band_hz", (0.01, 0.1))
    ax.axvspan(band[0], band[1], color=GRID, alpha=0.6, lw=0, label="analysis band")
    ax.axvline(fc, color=ORANGE, lw=1.5, ls="--", label=f"f_c = {fc:.3f} Hz")
    ax.set_xlim(0, min(f.max(), 3 * band[1]))
    ax.set_ylim(1e-4, 1.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("relative power")
    ax.set_title("I  Power spectrum of the mode time course")
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
