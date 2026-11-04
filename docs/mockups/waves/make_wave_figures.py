"""Mock-up figures: individual and case-control measures for multi-echo traveling-wave modes.

All data are SIMULATED to illustrate how each measure behaves; no panel shows real results.
Model: a complex wave mode has a dS0 map s and a dR2* map q (complex: amplitude + phase),
shared complex time course exp(i w t). Echo-e signal of the mode: z_e = s - TE_e q.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from pathlib import Path

OUT = Path(__file__).parent
rng = np.random.default_rng(7)

# ---------------- style (reference palette, light mode) ----------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ECHO_RAMP = ["#86b6ef", "#5598e7", "#256abf", "#104281"]   # ordinal, TE1 -> TE4
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "legend.frameon": False,
})
TE = np.array([16.20, 31.22, 46.24, 61.26]) / 1000.0     # s
T_PERIOD = 25.0                                            # s (0.04 Hz)
W = 2 * np.pi / T_PERIOD

# ---------------- a flattened cortical sheet ----------------
NX, NY, PX = 90, 60, 2.0                                    # 2 mm pixels -> 180 x 120 mm
x = np.arange(NX) * PX; y = np.arange(NY) * PX
X, Y = np.meshgrid(x, y)
mask = ((X - 90) / 88) ** 2 + ((Y - 60) / 58) ** 2 < 1

# Neural dR2* wave: spreads from a posterior source at 6 mm/s
src = (4.0, 64.0); v_neural = 6.0
dist = np.hypot(X - src[0], Y - src[1])
tau_R = dist / v_neural                                     # lag (s)
amp_R = 0.4 + 0.6 * np.exp(-((X - 100) ** 2 + (Y - 60) ** 2) / (2 * 55 ** 2))
# BOLD-positive convention: dR2* is negative when signal rises -> q = -A exp(-i w tau)
q_neural = -amp_R * 1.0 * np.exp(-1j * W * tau_R)          # s^-1 scale
s_neural = 0.0 * q_neural

# Vascular mode: fast S0 (inflow) transit along a vessel + delayed tissue dR2*
vy = 60 + 22 * np.sin(2 * np.pi * x / 180)                  # vessel path y(x)
dv = np.abs(Y - vy[None, :])
along = X                                                    # distance along the vessel ~ x
amp_S = 0.012 * np.exp(-dv ** 2 / (2 * 5 ** 2))             # dS0/S0 (1.2 % peak)
tau_S = along / 40.0                                         # 40 mm/s blood transit
amp_Rv = 0.6 * np.exp(-dv ** 2 / (2 * 14 ** 2))
tau_Rv = tau_S + 2.5                                         # tissue response lags inflow
s_vasc = amp_S * np.exp(-1j * W * tau_S)
q_vasc = -amp_Rv * np.exp(-1j * W * tau_Rv)

def noisy(z, sd):
    return z + sd * (rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape)) / np.sqrt(2)

q_n, s_n = noisy(q_neural, 0.05), noisy(s_neural, 0.0006)
q_v, s_v = noisy(q_vasc, 0.05), noisy(s_vasc, 0.0006)

# Whitening constants (T2* 45 ms) to express both maps in comparable energy units (Eq. 8)
mu = np.exp(-TE / 0.045)
Xw = mu[:, None] * np.column_stack([-TE * 1000, np.ones(4)])  # TE in ms to match R units
_, R = np.linalg.qr(Xw); R = R * np.sign(np.diag(R))[:, None]
R00, R01, R11 = R[0, 0], R[0, 1], R[1, 1]

def eta_map(s, q):
    es = (R01 ** 2 + R11 ** 2) * np.abs(s) ** 2
    er = R00 ** 2 * np.abs(q / 1000.0) ** 2                  # q in s^-1 -> ms^-1
    return es / (es + er + 1e-30)

def lag_map(z):
    return np.mod(-np.angle(z) / W, T_PERIOD)                  # 0..T s

CYC = plt.get_cmap("twilight")
def phase_rgb(z, amp_ref):
    """Cyclic, perceptually uniform colour = lag; opacity toward the surface = amplitude."""
    a = np.clip(np.abs(z) / amp_ref, 0, 1)[..., None]
    rgb = CYC(lag_map(z) / T_PERIOD)[..., :3]
    surf = np.array(matplotlib.colors.to_rgb(SURF))
    rgb = rgb * a + surf * (1 - a)
    rgb[~mask] = surf
    return rgb

def lag_gradient(z):
    """Gradient of lag from the complex map (no unwrapping): grad(phase) = Im(conj(z) grad z)/|z|^2."""
    from scipy.ndimage import gaussian_filter
    zs = gaussian_filter(z.real, 2) + 1j * gaussian_filter(z.imag, 2)   # 4 mm smoothing for display
    z = zs
    dzy, dzx = np.gradient(z)
    den = np.abs(z) ** 2 + 1e-30
    gx = -np.imag(np.conj(z) * dzx) / den / W
    gy = -np.imag(np.conj(z) * dzy) / den / W
    return gx, gy

def style_map(ax, title):
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

def add_arrows(ax, z):
    gx, gy = lag_gradient(z)
    n = np.hypot(gx, gy) + 1e-12
    sl = (slice(4, None, 7), slice(4, None, 7))
    m = mask[sl] & (np.abs(z)[sl] > 0.3 * np.abs(z).max())
    ax.quiver(np.arange(NX)[sl[1]][None, :].repeat(m.shape[0], 0)[m],
              np.arange(NY)[sl[0]][:, None].repeat(m.shape[1], 1)[m],
              (gx / n)[sl][m], (gy / n)[sl][m], color=INK, width=0.004, scale=28,
              headwidth=4, alpha=0.8)

def lag_colorbar(fig, ax):
    sm = plt.cm.ScalarMappable(cmap=CYC, norm=plt.Normalize(0, T_PERIOD))
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("lag within cycle (s)", color=INK2); cb.outline.set_visible(False)

TAG = "Simulated data. Illustrates the measures; not results."

# =====================================================================
# Figure 1: one participant, one neural wave mode
# =====================================================================
fig = plt.figure(figsize=(12, 7.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.35, wspace=0.25)

ax = fig.add_subplot(gs[0, 0:2])
ax.imshow(phase_rgb(q_n, np.abs(q_n).max()), origin="lower", interpolation="nearest")
add_arrows(ax, q_n)
style_map(ax, "A  ΔR2* map q: lag (hue), amplitude (brightness), propagation (arrows)")
lag_colorbar(fig, ax)
ax.text(1, -4, "Flattened cortical sheet, 180 × 120 mm", color=INK2, fontsize=8)

ax = fig.add_subplot(gs[0, 2])
ax.imshow(phase_rgb(s_n, np.abs(q_n).max() * 0.012), origin="lower", interpolation="nearest")
style_map(ax, "B  ΔS0 map s (same scale)")
ax.text(1, -4, "Near zero everywhere: no S0 content", color=INK2, fontsize=8)

ax = fig.add_subplot(gs[0, 3])
im = ax.imshow(np.where(mask, eta_map(s_n, q_n), np.nan), origin="lower", cmap="Blues", vmin=0, vmax=1)
style_map(ax, "C  Voxelwise η")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.outline.set_visible(False)
cb.set_label("S0 energy fraction (0 = pure ΔR2*)", color=INK2)

# D: cycle snapshots
for i, th in enumerate([0, 90, 180, 270]):
    ax = fig.add_subplot(gs[1, i] if i < 2 else gs[1, i])
    pass
plt.close(fig)

# Rebuild bottom row with snapshots (2 cols) + space-time (2 cols)
fig = plt.figure(figsize=(12, 8.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.95], hspace=0.38, wspace=0.28)
ax = fig.add_subplot(gs[0, 0:2])
ax.imshow(phase_rgb(q_n, np.abs(q_n).max()), origin="lower", interpolation="nearest")
add_arrows(ax, q_n)
style_map(ax, "A  ΔR2* map q")
lag_colorbar(fig, ax)
ax.text(0, -6, "Colour = lag within the cycle (cyclic scale), faded = low amplitude, arrows = local\npropagation "
        "direction (phase gradient). Flattened cortical sheet, 180 × 120 mm.", color=INK2, fontsize=8, va="top")

ax = fig.add_subplot(gs[0, 2])
ax.imshow(phase_rgb(s_n, 0.012), origin="lower", interpolation="nearest")
style_map(ax, "B  ΔS0 map s")
ax.text(0, -6, "Same encoding, scaled to a 1.2% S0\nchange: faded means no S0 content.", color=INK2, fontsize=8, va="top")

ax = fig.add_subplot(gs[0, 3])
im = ax.imshow(np.where(mask, eta_map(s_n, q_n), np.nan), origin="lower", cmap="Blues", vmin=0, vmax=1)
style_map(ax, "C  Voxelwise η")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.outline.set_visible(False)
cb.set_label("S0 energy fraction", color=INK2)
ax.text(0, -6, "0 = pure ΔR2*, 1 = pure ΔS0.", color=INK2, fontsize=8, va="top")

# D: four snapshots in a 2x2 inside gs[1, 0:2]
sub = gs[1, 0:2].subgridspec(2, 2, hspace=0.25, wspace=0.05)
vmax = np.abs(q_n).max()
for i, th in enumerate([0, 90, 180, 270]):
    ax = fig.add_subplot(sub[i // 2, i % 2])
    frame = np.real(-q_n * np.exp(1j * np.deg2rad(th)))       # BOLD-positive
    ax.imshow(np.where(mask, frame, np.nan), origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    style_map(ax, f"{th}°, t = {th / 360 * T_PERIOD:.1f} s")
    ax.title.set_fontsize(8.5); ax.title.set_fontweight("normal"); ax.title.set_color(INK2)
    if i == 0:
        ax.text(0, 1.3, "D  Cycle snapshots (BOLD-positive, one period)", transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=INK)

# E: space-time plot along the propagation axis
ax = fig.add_subplot(gs[1, 2:4])
row = int(src[1] / PX)
d_line = x                                                    # mm along row
t = np.arange(0, 100, 0.75)
z_line = -q_n[row]
st = np.real(z_line[None, :] * np.exp(1j * W * t)[:, None])
st += 0.25 * rng.standard_normal(st.shape)
ax.imshow(st.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2,
          extent=[t[0], t[-1], d_line[0], d_line[-1]])
ax.grid(False)
for k in range(4):                                            # wavefront guide lines
    t0 = k * T_PERIOD
    ax.plot([t0, t0 + d_line[-1] / v_neural], [0, d_line[-1]], color=INK, lw=1, ls="--", alpha=0.6)
ax.set_xlim(t[0], t[-1]); ax.set_ylim(d_line[0], d_line[-1])
ax.set_xlabel("time (s)"); ax.set_ylabel("distance from source (mm)")
ax.set_title("E  Space–time plot along the propagation axis")
ax.text(62, 10, f"slope = speed\n({v_neural:.0f} mm/s here)", color=INK, fontsize=8.5,
        bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.85))

fig.suptitle("Figure 1. Visualizing one participant's traveling-wave mode", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=INK)
fig.text(0.01, 0.945, TAG, color=INK2, fontsize=8.5)
fig.savefig(OUT / "fig1_individual_wave_mode.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# Figure 2: diagnostics that separate a neural wave from vascular transit
# =====================================================================
fig, axs = plt.subplots(2, 3, figsize=(12, 7.4), gridspec_kw=dict(hspace=0.45, wspace=0.3))

def path_samples(xs, ys):
    return np.clip((ys / PX).astype(int), 0, NY - 1), np.clip((xs / PX).astype(int), 0, NX - 1)

xs_path = np.linspace(6, 170, 28)
iy_n, ix_n = path_samples(xs_path, np.full_like(xs_path, src[1]))
iy_v, ix_v = path_samples(xs_path, 60 + 22 * np.sin(2 * np.pi * xs_path / 180) + 6)   # tissue next to vessel
dist_path_n = np.hypot(xs_path - src[0], 0)
dist_path_v = xs_path

def echo_lags(s, q, iy, ix):
    out = []
    for e in range(4):
        z = s[iy, ix] - TE[e] * q[iy, ix]
        out.append(np.unwrap(-np.angle(z)) / W)
    out = np.array(out)
    return out - out[0, 0] + 0.0

for col, (name, s, q, iy, ix, dpath) in enumerate([
        ("Neural wave", s_n, q_n, iy_n, ix_n, dist_path_n),
        ("Vascular transit", s_v, q_v, iy_v, ix_v, dist_path_v)]):
    ax = axs[0, col]
    lags = echo_lags(s, q, iy, ix)
    for e in range(4):
        ax.plot(dpath, lags[e], "o-", ms=3.5, lw=1.2, color=ECHO_RAMP[e],
                label=f"TE {TE[e] * 1000:.1f} ms")
    slope = np.polyfit(dpath, lags[-1], 1)[0]
    ax.set_title(f"{'AB'[col]}  {name}: lag per echo")
    ax.set_xlabel("distance along path (mm)"); ax.set_ylabel("lag (s)")
    note = ("Lags identical across echoes:\none ΔR2* source. Speed = 1/slope\n"
            f"= {1 / slope:.1f} mm/s") if col == 0 else \
           ("Lag depends on TE: early echoes\nfollow fast S0 inflow, late echoes\nthe delayed tissue ΔR2*")
    ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", fontsize=8, color=INK,
            bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.9))
    if col == 0:
        ax.legend(loc="lower right", fontsize=7.5)

# C: per-echo phase spread (TE dependence of lag) at sampled points, both modes
ax = axs[0, 2]
for name, s, q, iy, ix, c in [("Neural", s_n, q_n, iy_n, ix_n, BLUE), ("Vascular", s_v, q_v, iy_v, ix_v, AQUA)]:
    lags = echo_lags(s, q, iy, ix)
    spread = lags[-1] - lags[0]                               # TE4 minus TE1 lag
    ax.hist(spread, bins=np.linspace(-0.5, 3, 25), color=c, alpha=0.75, label=name, edgecolor=SURF)
ax.set_title("C  TE dependence of lag")
ax.set_xlabel("lag at TE4 minus lag at TE1 (s)"); ax.set_ylabel("voxels on path")
ax.text(0.12, ax.get_ylim()[1] * 0.80, "Neural", color=INK, fontsize=8.5)
ax.text(1.15, ax.get_ylim()[1] * 0.45, "Vascular", color=INK, fontsize=8.5)
ax.text(0.97, 0.97, "≈ 0: one ΔR2* source\n> 0: S0 and ΔR2* parts\nwith different timing",
        transform=ax.transAxes, fontsize=8, color=INK2, va="top", ha="right")

# D: eta along the path
ax = axs[1, 0]
for name, s, q, iy, ix, dpath, c in [("Neural wave", s_n, q_n, iy_n, ix_n, dist_path_n, BLUE),
                                     ("Vascular transit (on vessel)", s_v, q_v,
                                      *path_samples(xs_path, 60 + 22 * np.sin(2 * np.pi * xs_path / 180)),
                                      dist_path_v, AQUA)]:
    e = eta_map(s, q)[iy, ix]
    ax.plot(dpath, e, "-", color=c)
    ax.text(dpath[-1] + 2, e[-1], name.split(" (")[0], color=INK, fontsize=8, va="center")
ax.set_ylim(-0.03, 1.03); ax.set_xlim(0, 215)
ax.set_title("D  η along the path")
ax.set_xlabel("distance along path (mm)"); ax.set_ylabel("S0 energy fraction η")

# E: propagation directions, polar histogram, both modes
ax = fig.add_subplot(2, 3, 5, projection="polar"); axs[1, 1].remove()
rtxt = []
for name, q, c in [("Neural", q_n, BLUE), ("Vascular", q_v, AQUA)]:
    gx, gy = lag_gradient(q)
    ang = np.arctan2(gy, gx)[mask & (np.abs(q) > 0.3 * np.abs(q).max())]
    Rlen = np.abs(np.mean(np.exp(1j * ang)))
    h, b = np.histogram(ang, bins=24, range=(-np.pi, np.pi))
    ax.bar(b[:-1], h / h.max(), width=2 * np.pi / 24, color=c, alpha=0.6, edgecolor=SURF, align="edge")
    rtxt.append(f"{name}: R = {Rlen:.2f}")
ax.set_yticklabels([]); ax.grid(color=GRID)
ax.set_title("E  Propagation direction", pad=24)
ax.text(0.5, -0.18, "   ".join(rtxt) + "\nR = directional coherence (1 = one direction everywhere).",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color=INK2)

# F: per-echo complex amplitude in one voxel (phasor view)
ax = axs[1, 2]
for name, s, q, (iy, ix), c, mk in [("Neural", s_n, q_n, (int(src[1] / PX), 40), BLUE, "o"),
                                    ("Vascular", s_v, q_v, (int((60 + 22 * np.sin(2 * np.pi * 80 / 180)) / PX), 40), AQUA, "s")]:
    z = np.array([s[iy, ix] - TE[e] * q[iy, ix] for e in range(4)])
    z = z / np.abs(z).max()
    ax.plot(np.real(z), np.imag(z), "-", color=c, lw=1)
    for e in range(4):
        ax.plot(np.real(z[e]), np.imag(z[e]), mk, ms=7, color=ECHO_RAMP[e], mec=c, mew=1.5)
    ax.text(np.real(z[-1]) + (0.1 if name == "Vascular" else -0.05), np.imag(z[-1]) + (0 if name == "Vascular" else 0.12), name, fontsize=8.5, color=INK)
ax.axhline(0, color=GRID); ax.axvline(0, color=GRID)
ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
ax.set_title("F  Echo phasors in one voxel")
ax.set_xlabel("real (normalized)"); ax.set_ylabel("imaginary")
ax.text(0.0, -0.22, "Points: TE1 (light) to TE4 (dark). Points on a line through the origin\nshare one phase; a line that misses the origin means a mixture.",
        transform=ax.transAxes, fontsize=7.8, color=INK2, va="top")

fig.suptitle("Figure 2. Multi-echo diagnostics: neural wave versus vascular transit", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=INK, y=0.99)
fig.text(0.01, 0.945, TAG, color=INK2, fontsize=8.5)
fig.savefig(OUT / "fig2_multiecho_wave_diagnostics.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# Figure 3: case-control comparison of subject-level wave measures (hypothetical)
# =====================================================================
n = 20
groups = {"Controls": dict(c=BLUE), "Patients": dict(c=ORANGE)}
meas = [
    ("A  Wave-mode variance", "% of c0 variance", (8.0, 2.0), (6.2, 2.4)),
    ("B  Propagation speed", "mm/s", (6.0, 1.2), (4.9, 1.5)),
    ("C  Directional coherence R", "mean resultant length", (0.74, 0.07), (0.63, 0.11)),
    ("D  Mode η", "S0 energy fraction", (0.10, 0.04), (0.14, 0.05)),
]
fig = plt.figure(figsize=(12, 7.2))
gs = fig.add_gridspec(2, 4, hspace=0.65, wspace=0.45)
for i, (title, unit, mc, mp) in enumerate(meas):
    ax = fig.add_subplot(gs[0, i])
    vals = {}
    for j, (g, (m, sd)) in enumerate(zip(groups, [mc, mp])):
        v = rng.normal(m, sd, n); vals[g] = v
        xj = j + rng.uniform(-0.13, 0.13, n)
        ax.plot(xj, v, "o", ms=5, color=groups[g]["c"], alpha=0.75, mec=SURF, mew=0.8)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(n)
        ax.errorbar(j + 0.3, v.mean(), yerr=ci, fmt="_", ms=14, color=INK, elinewidth=1.5, capsize=0)
    # permutation test on the mean difference
    a, b = vals["Controls"], vals["Patients"]
    obs = b.mean() - a.mean(); pool = np.concatenate([a, b])
    perm = np.array([np.subtract(*[rng.permutation(pool)[k * n:(k + 1) * n].mean() for k in (1, 0)])
                     for _ in range(5000)])
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / 5001
    ax.set_xticks([0, 1]); ax.set_xticklabels(list(groups)); ax.set_xlim(-0.5, 1.6)
    ax.set_title(title); ax.set_ylabel(unit)
    ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.5, -0.22, f"Δ = {obs:+.2g}, permutation {ptxt}", transform=ax.transAxes,
            ha="center", fontsize=8, color=INK2)

# E: cerebellar-cortical phase offset, circular
ax = fig.add_subplot(gs[1, 0:2], projection="polar")
offs = {"Controls": rng.vonmises(np.deg2rad(-45), 6, n), "Patients": rng.vonmises(np.deg2rad(-10), 2.5, n)}
for g, ph in offs.items():
    r = 0.85 + rng.uniform(-0.06, 0.06, n)
    ax.plot(ph, r, "o", ms=5, color=groups[g]["c"], alpha=0.8, mec=SURF, mew=0.8, label=g)
    mvec = np.mean(np.exp(1j * ph))
    ax.annotate("", xy=(np.angle(mvec), np.abs(mvec) * 0.8), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=groups[g]["c"], lw=2))
    groups[g]["circ"] = f"{g}: mean {np.rad2deg(np.angle(mvec)):.0f}°, R = {np.abs(mvec):.2f}"
ax.set_ylim(0, 1.05); ax.set_yticklabels([]); ax.grid(color=GRID)
ax.set_thetagrids([0, 90, 180, 270], ["0°", "+90° (cerebellum lags)", "", "−90° (cerebellum leads)"])
ax.set_title("E  Cerebellar–cortical phase offset, per participant", pad=18)
ax.legend(loc="upper left", bbox_to_anchor=(-0.75, 1.02), fontsize=8)
ax.text(-0.75, 0.72, groups["Controls"]["circ"] + "\n" + groups["Patients"]["circ"] +
        "\n\nArrow: circular mean; its length is\nacross-participant consistency.\nTest: circular permutation (or\n"
        "Watson–Williams) on angle;\nseparately on R.", transform=ax.transAxes, fontsize=7.8, color=INK2, va="top")

# F: lag profile along a cortical hierarchy, group means with CI
ax = fig.add_subplot(gs[1, 2:4])
regions = ["V1", "MT+", "SPL", "PMd", "dlPFC", "FPN"]
base_c = np.array([0, 0.6, 1.3, 2.0, 2.8, 3.4]); base_p = base_c * 0.75 + 0.2
for g, base in [("Controls", base_c), ("Patients", base_p)]:
    sub = base[None, :] + rng.normal(0, 0.45, (n, len(regions)))
    m = sub.mean(0); ci = 1.96 * sub.std(0, ddof=1) / np.sqrt(n)
    xx = np.arange(len(regions)) + (-0.08 if g == "Controls" else 0.08)
    ax.errorbar(xx, m, yerr=ci, fmt="o-", ms=5, color=groups[g]["c"], lw=1.8, elinewidth=1.2, capsize=0)
    ax.text(xx[-1] + 0.15, m[-1], g, color=INK, fontsize=8.5, va="center")
ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions)
ax.set_xlim(-0.4, len(regions) + 0.4)
ax.set_ylabel("lag relative to V1 (s)")
ax.set_title("F  Lag profile along a cortical hierarchy (mean, 95% CI)")
ax.text(0.0, -0.25, "Slope of this profile = an apparent speed per participant; compare slopes between groups\n"
        "(mixed model with region × group). Regions are illustrative.", transform=ax.transAxes,
        fontsize=8, color=INK2, va="top")

fig.suptitle("Figure 3. Case–control comparison of subject-level wave measures", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=INK, y=1.0)
fig.text(0.01, 0.955, "Hypothetical group differences (n = 20 per group). Illustrates the measures and tests; not results.",
         color=INK2, fontsize=8.5)
fig.savefig(OUT / "fig3_case_control_measures.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("done")
