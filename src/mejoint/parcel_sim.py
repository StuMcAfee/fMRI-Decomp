"""Parcel-level simulation for the parcel ICA (gate P1, docs/PARCEL_ICA_PLAN.md).

Generates what route B produces for each parcel: the two whitened-scale channels c0, c1, the leftover channels r
(pure thermal noise) and the 2 x 2 matrix R, from known BOLD-like (dR2*, 1/s) and S0-like (dS0/S0) sources.
Channels follow the region model exactly: c0 = R00 b / 1000 + R01 a, c1 = R11 a, plus independent noise of variance
noise_var (CLAUDE.md 5.3, regions.split). R comes from the real TEs, a per-parcel T2* and a per-parcel precision
factor (voxel count and signal-to-noise); the overall noise level is set so that the median in-band reliability of
the network + global part of b (route B's signal share of slow BOLD-like variance) matches a target.

Sources (each with a known parcel map and time course):
  network_1..7   band-limited 0.01-0.1 Hz dR2*, maps = the 7 networks with heterogeneous loading;
  global         band-limited dR2* with a positive, regionally varying map (the brain-wide pattern);
  slow_physio    slow (0.01-0.05 Hz) dR2* with a vascular-weighted map (breathing depth / CO2 / heart-rate
                 variation: BOLD-like by the echo-time test, not separable by spectrum);
  respiratory    dR2* at the breathing frequency (amplitude- and rate-modulated) with an inferior / vein-weighted
                 map and a small dS0 part;
  cardiac        heart rate ~1.2 Hz aliased by the TR (0.133 Hz at TR 0.75 s), sparse vascular map, mostly dS0.
Geometry: parcels as points on a sphere (two hemispheres), networks as contiguous patches, "inferior" = low z.
"""
from __future__ import annotations

import numpy as np

NETS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")


def sphere_points(n: int, rng) -> np.ndarray:
    """Quasi-uniform points on the unit sphere (Fibonacci lattice, slightly jittered)."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    xyz = np.column_stack([np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z])
    xyz += rng.normal(0, 0.3 / np.sqrt(n), xyz.shape)
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


def networks(xyz: np.ndarray, rng, n_seeds_per_net: int = 3) -> np.ndarray:
    """Network label per parcel: each network owns a few patches (nearest of 7 x n_seeds seed points), mirrored
    between hemispheres (x > 0 and x < 0), so networks are distributed like the real ones."""
    seeds = sphere_points(len(NETS) * n_seeds_per_net, rng)
    seeds[:, 0] = np.abs(seeds[:, 0])
    owner = np.repeat(np.arange(len(NETS)), n_seeds_per_net)
    p = xyz.copy()
    p[:, 0] = np.abs(p[:, 0])                                          # mirror symmetric
    d = p @ seeds.T
    return np.array(NETS)[owner[np.argmax(d, 1)]]


def band_noise(T: int, tr: float, band, rng, n: int = 1) -> np.ndarray:
    """Unit-variance Gaussian noise restricted to a frequency band (FFT)."""
    F = np.fft.rfft(rng.standard_normal((n, T)), axis=1)
    f = np.fft.rfftfreq(T, tr)
    F[:, (f < band[0]) | (f > band[1])] = 0
    x = np.fft.irfft(F, T, axis=1)
    return x / x.std(1, keepdims=True)


def oscillation(T: int, tr: float, f0: float, rng, f_sd: float, am_depth: float = 0.5) -> np.ndarray:
    """Quasi-periodic signal at f0 (Hz): instantaneous frequency wandering slowly with SD f_sd, amplitude modulated
    by a slow positive envelope; sampled at the TR, so frequencies above Nyquist alias exactly as in the scanner."""
    t = np.arange(T) * tr
    fi = f0 + f_sd * band_noise(T, tr, (0.002, 0.03), rng)[0]
    ph = 2 * np.pi * np.cumsum(fi) * tr + rng.uniform(0, 2 * np.pi)
    env = 1 + am_depth * band_noise(T, tr, (0.005, 0.05), rng)[0]
    x = np.clip(env, 0.1, None) * np.sin(ph)
    return (x - x.mean()) / x.std()


def r_matrices(te_ms, t2s_ms: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """R (P, 2, 2) of the whitened [-TE, 1] design for monoexponential means mu_e = exp(-TE/T2*) scaled by each
    parcel's precision factor, with the QR sign convention (diag R > 0)."""
    from .frontend import design
    te = np.asarray(te_ms, float)
    D = design(te)
    out = np.zeros((len(t2s_ms), 2, 2))
    for p, (t2, g) in enumerate(zip(t2s_ms, precision)):
        m = g * np.exp(-te / t2)
        _, Rm = np.linalg.qr(m[:, None] * D)
        Rm = Rm[:2, :2] * np.sign(np.diag(Rm)[:2])[:, None]
        out[p] = Rm
    return out


def simulate(n_parcels: int = 1876, T: int = 797, tr: float = 0.75, te_ms=(16.20, 31.22, 46.24, 61.26),
             reliability: float = 0.3, seed: int = 0, amp: dict | None = None, resp_hz: float = 0.28,
             heart_hz: float = 1.2, include: tuple | None = None, overlap: bool = True) -> dict:
    """One synthetic participant. Returns the route B parcel arrays (c0, c1, r, R, noise_var, a, b), the truth
    (maps_b, maps_a, time courses, names), geometry (xyz, networks) and the calibration actually reached."""
    rng = np.random.default_rng(seed)
    E = len(te_ms)
    A = {"network": 0.25, "global": 0.30, "slow_physio": 0.20, "respiratory": 0.12, "resp_s0": 0.0006,
         "cardiac": 0.04, "card_s0": 0.002}
    A.update(amp or {})
    xyz = sphere_points(n_parcels, rng)
    net = networks(xyz, rng)
    inferior = np.clip(-xyz[:, 2], 0, None)                            # 0 at the equator and above, 1 at the pole
    vascular = np.zeros(n_parcels)
    vascular[rng.choice(n_parcels, max(3, n_parcels // 10), replace=False)] = 1.0   # vein / sinus-adjacent parcels
    smooth = np.tanh(2 * (xyz @ rng.normal(0, 1, 3)) + rng.normal(0, 0.3, n_parcels)) * 0.5 + 0.5   # smooth gradient
    names, maps_b, maps_a, tcs = [], [], [], []

    def add(name, mb, ma, tc):
        names.append(name)
        maps_b.append(mb)
        maps_a.append(ma)
        tcs.append(tc)

    for n in NETS:
        load = (net == n) * rng.uniform(0.5, 1.5, n_parcels)
        add(f"network_{n}", A["network"] * load, np.zeros(n_parcels), band_noise(T, tr, (0.01, 0.10), rng)[0])
    add("global", A["global"] * rng.uniform(0.6, 1.4, n_parcels), np.zeros(n_parcels), band_noise(T, tr, (0.01, 0.10), rng)[0])
    # overlap=True: the three physiological maps share the vascular weighting (realistic, hard); False: distinct maps
    smap = (0.3 + 0.7 * vascular + 0.4 * inferior) if overlap else (0.2 + smooth)
    add("slow_physio", A["slow_physio"] * smap * rng.uniform(0.7, 1.3, n_parcels), np.zeros(n_parcels),
        band_noise(T, tr, (0.01, 0.05), rng)[0])
    rmap = ((0.2 + inferior + 0.8 * vascular) if overlap else (0.1 + 1.5 * inferior)) * rng.uniform(0.7, 1.3, n_parcels)
    add("respiratory", A["respiratory"] * rmap, A["resp_s0"] * rmap, oscillation(T, tr, resp_hz, rng, 0.02))
    cmap = vascular * rng.uniform(0.5, 1.5, n_parcels) + 0.05 * rng.uniform(0, 1, n_parcels)
    add("cardiac", A["cardiac"] * cmap, A["card_s0"] * cmap, oscillation(T, tr, heart_hz, rng, 0.015, 0.2))
    if include is not None:
        keep = [i for i, nm in enumerate(names) if nm in include or nm.split("_")[0] in include]
        names = [names[i] for i in keep]
        maps_b, maps_a, tcs = [maps_b[i] for i in keep], [maps_a[i] for i in keep], [tcs[i] for i in keep]
    Mb, Ma, Tc = np.array(maps_b).T, np.array(maps_a).T, np.array(tcs)   # (P, n_src), (P, n_src), (n_src, T)
    b_true, a_true = Mb @ Tc, Ma @ Tc                                  # 1/s and fraction
    # per-parcel R: T2* 40-70 ms (children), precision factor lognormal (voxel count, signal to noise)
    t2s = rng.uniform(40, 70, n_parcels)
    g = np.exp(rng.normal(0, 0.4, n_parcels))
    R0 = r_matrices(te_ms, t2s, g)
    # in-band reliability of b for a given overall scale s (R = s R0, unit channel noise)
    f = np.fft.rfftfreq(T, tr)
    band = (f >= 0.01) & (f <= 0.10)
    # calibrate on the network and global sources only, so the noise level does not change with the physiological
    # sources' strength (their effect is then measured at fixed network reliability)
    neural = [i for i, nm in enumerate(names) if nm.startswith("network") or nm == "global"]
    Fb = np.fft.rfft(Mb[:, neural] @ Tc[neural], axis=1)
    sig_band = (np.abs(Fb[:, band]) ** 2).sum(1) * 2 / T ** 2          # in-band signal variance
    frac = band.sum() / (len(f) - 1)                                   # white noise share in band
    rho = R0[:, 0, 1] / R0[:, 1, 1]
    nb0 = (1 + rho ** 2) / R0[:, 0, 0] ** 2 * 1e6                      # b noise variance at scale 1, (1/s)^2

    def med_rel(s):
        nvar = nb0 / s ** 2 * frac
        return np.median(sig_band / (sig_band + nvar))
    lo, hi = 1e-3, 1e6
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if med_rel(mid) < reliability else (lo, mid)
    s = np.sqrt(lo * hi)
    R = R0 * s
    nv = np.ones(n_parcels)
    c0 = R[:, 0, 0, None] * b_true / 1000.0 + R[:, 0, 1, None] * a_true + rng.standard_normal((n_parcels, T))
    c1 = R[:, 1, 1, None] * a_true + rng.standard_normal((n_parcels, T))
    r = rng.standard_normal((n_parcels, max(E - 2, 0), T))
    if E - 2 < 2:                                                      # keep the route B shape (P, 2, T)
        r = np.concatenate([r, np.zeros((n_parcels, 2 - (E - 2), T))], 1)
    a = c1 / R[:, 1, 1, None]
    b = (c0 - R[:, 0, 1, None] * a) / R[:, 0, 0, None] * 1000.0
    return {"c0": c0, "c1": c1, "r": r, "R": R, "noise_var": nv, "a": a, "b": b, "tr": tr, "te_ms": list(te_ms),
            "truth": {"names": names, "maps_b": Mb, "maps_a": Ma, "tc": Tc, "b": b_true, "a": a_true},
            "xyz": xyz, "networks": net, "vascular": vascular, "inferior": inferior, "t2s": t2s,
            "reliability_median": float(med_rel(s)), "resp_hz": resp_hz, "heart_hz": heart_hz,
            "cardiac_alias_hz": float(abs(heart_hz - round(heart_hz * tr) / tr))}
