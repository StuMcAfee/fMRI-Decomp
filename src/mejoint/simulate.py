"""Ground-truth multi-echo simulator (Phase 1).

Builds magnitude echoes at the real TEs from known maps and time courses, so every later step
(front end, rank, decomposition, metrics) can be checked against the truth.

Signal model per voxel v, echo e, volume t (TE in seconds, R2* and q in s^-1):

    ln S_e(t) = ln S0_v + a_v(t) - TE_e * (R2*_v + b_v(t)) + ln D_ve(t) + spike_ve(t)
    a_v(t) = sum_k s_kv x_k(t)      (delta ln S0, about dS0/S0)
    b_v(t) = sum_k q_kv x_k(t)      (dR2*)

    D_ve(t) = |sinc(df_v(t) * TE_e)|  through-slice dephasing near the synthetic skull base
                                      (curvature in the decay; df_v(t) can follow respiration)
    spike_ve(t)                       model-violating frames: an S0-like jump plus an independent
                                      jump per echo (not in span{1, TE})

Without dephasing and spikes, ln(S / baseline) is exactly a - TE * b (CLAUDE.md section 5.1).
Noise: complex Gaussian with SD sigma per channel, then magnitude. With n_coils = 1 this is
Rician; with n_coils = L it is sum-of-squares (noncentral chi with 2L degrees of freedom,
equal coil sensitivities). At high SNR the magnitude noise SD equals sigma in both cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage, special

from . import io

KINDS = ("r2", "s0", "mixed")
EXTENTS = ("focal", "diffuse")


# ======================================================================= helpers
def magnitude_mean(nu, sigma, n_coils: int = 1):
    """Expected magnitude for true signal nu, channel noise SD sigma, L coils (sum of squares).

    E[M] = sigma * sqrt(2) * Gamma(L + 1/2) / Gamma(L) * 1F1(-1/2; L; -nu^2 / (2 sigma^2)).
    L = 1 gives the Rician mean sigma * sqrt(pi/2) * L_{1/2}(-nu^2 / 2 sigma^2).
    """
    nu = np.asarray(nu, float)
    sigma = np.asarray(sigma, float)
    L = int(n_coils)
    pref = np.sqrt(2.0) * np.exp(special.gammaln(L + 0.5) - special.gammaln(L))
    return sigma * pref * special.hyp1f1(-0.5, L, -(nu ** 2) / (2 * sigma ** 2))


def _smooth_field(rng, shape, smooth, mask=None):
    """Smooth Gaussian random field, zero mean and unit SD (inside `mask` if given)."""
    f = ndimage.gaussian_filter(rng.standard_normal(shape), smooth, mode="reflect")
    ref = f[mask] if mask is not None and mask.any() else f
    return (f - ref.mean()) / (ref.std() + 1e-12)


def _bandlimited(rng, T, tr, lo, hi):
    """Gaussian noise restricted to [lo, hi] Hz, zero mean, unit SD."""
    spec = np.fft.rfft(rng.standard_normal(T))
    f = np.fft.rfftfreq(T, tr)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, T)
    if x.std() == 0:            # band empty for very short series: fall back to the lowest frequencies
        spec = np.fft.rfft(rng.standard_normal(T))
        spec[3:] = 0
        spec[0] = 0
        x = np.fft.irfft(spec, T)
    return (x - x.mean()) / x.std()


def _respiration(rng, T, tr, f0, jitter):
    """Quasi-periodic breathing trace: frequency and amplitude drift slowly. Unit SD."""
    slow = lambda: _bandlimited(rng, T, tr, 0.0, 0.02) if T > 30 else np.zeros(T)
    freq = f0 + jitter * slow()
    amp = 1.0 + 0.2 * slow()
    x = amp * np.sin(2 * np.pi * np.cumsum(freq) * tr + rng.uniform(0, 2 * np.pi))
    return (x - x.mean()) / x.std()


# ======================================================================= container
@dataclass
class Simulation:
    data: np.ndarray            # (X, Y, Z, E, T) float32 magnitude
    te_ms: np.ndarray           # (E,)
    tr_s: float
    voxel_mm: tuple
    mask: np.ndarray            # (X, Y, Z) brain
    s0: np.ndarray              # (X, Y, Z)
    t2s_ms: np.ndarray          # (X, Y, Z)
    sigma: np.ndarray           # (X, Y, Z) channel noise SD, equal across echoes
    mu_base: np.ndarray         # (X, Y, Z, E) noise-free baseline (static dephasing included, no sources)
    dephasing_hz: np.ndarray    # (X, Y, Z) static field spread across the voxel
    sources: list               # dicts: name, kind, extent, timecourse (T,), s_map, q_map (X, Y, Z)
    regions: dict               # name -> (X, Y, Z) bool: brain, interior, skull_base, spike
    spike_frames: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_echoes(self):
        return self.data.shape[3]

    @property
    def n_volumes(self):
        return self.data.shape[4]

    def echo(self, k: int) -> np.ndarray:
        """4D series (X, Y, Z, T) of echo k (1-based)."""
        return self.data[..., k - 1, :]

    def masked(self, region: str = "brain") -> np.ndarray:
        """(V, E, T) data inside a region."""
        return self.data[self.regions[region]]

    def truth_ab(self, region: str = "brain"):
        """True a(t) (delta ln S0) and b(t) (dR2*, s^-1) per voxel: two (V, T) arrays."""
        m = self.regions[region]
        X = np.stack([s["timecourse"] for s in self.sources]) if self.sources else np.zeros((0, self.n_volumes))
        S = np.stack([s["s_map"][m] for s in self.sources], axis=1) if self.sources else np.zeros((m.sum(), 0))
        Q = np.stack([s["q_map"][m] for s in self.sources], axis=1) if self.sources else np.zeros((m.sum(), 0))
        return S @ X, Q @ X

    def tsnr(self, region: str = "brain") -> list[float]:
        """Median temporal SNR (mean / SD over time) per echo, including all simulated fluctuations."""
        d = self.masked(region).astype(np.float64)
        return [float(np.median(d[:, e].mean(1) / d[:, e].std(1))) for e in range(self.n_echoes)]

    @property
    def echo_sigma_scale(self) -> np.ndarray:
        return np.asarray(self.meta.get("echo_sigma_scale") or [1.0] * self.n_echoes, float)

    @property
    def sigma_echo(self) -> np.ndarray:
        """(X, Y, Z, E) thermal noise SD of every echo (sigma x per-echo scale)."""
        return self.sigma[..., None] * self.echo_sigma_scale

    def thermal_snr(self, region: str = "brain") -> list[float]:
        """Median baseline signal / thermal sigma per echo."""
        m = self.regions[region]
        return [float(np.median(self.mu_base[m][:, e] / self.sigma_echo[m][:, e])) for e in range(self.n_echoes)]

    def describe(self) -> str:
        base = (f"grid {self.data.shape[:3]}, {self.n_volumes} vols, TR {self.tr_s} s, "
                f"{len(self.sources)} sources, {len(self.spike_frames)} spikes, n_coils {self.meta.get('n_coils')}; ")
        if self.meta.get("tsnr") is None:
            return base + "noise-free (no thermal noise)"
        return (base + f"tSNR per echo {np.round(self.meta['tsnr'], 1).tolist()}; "
                f"thermal SNR per echo {np.round(self.meta['thermal_snr'], 1).tolist()}")

    def to_nifti(self, out_dir) -> dict:
        """Write one 4D NIfTI + JSON sidecar per echo (dcm2niix-like names) and the truth."""
        import nibabel as nib

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        aff = np.diag([*self.voxel_mm, 1.0])
        paths = []
        for k in range(1, self.n_echoes + 1):
            img = nib.Nifti1Image(self.echo(k).astype(np.float32), aff)
            img.header.set_xyzt_units("mm", "sec")
            img.header["pixdim"][4] = self.tr_s
            p = out_dir / f"sim_e{k}.nii.gz"
            nib.save(img, str(p))
            io.write_json(out_dir / f"sim_e{k}.json", {"EchoTime": float(self.te_ms[k - 1]) / 1000.0, "EchoNumber": k,
                                                       "RepetitionTime": self.tr_s, "Simulated": True})
            paths.append(p)
        np.savez_compressed(
            out_dir / "truth.npz", mask=self.mask, s0=self.s0, t2s_ms=self.t2s_ms, sigma=self.sigma,
            mu_base=self.mu_base, dephasing_hz=self.dephasing_hz, spike_frames=self.spike_frames,
            source_names=np.array([s["name"] for s in self.sources]),
            source_kinds=np.array([s["kind"] for s in self.sources]),
            timecourses=np.stack([s["timecourse"] for s in self.sources]) if self.sources else np.zeros((0, self.n_volumes)),
            s_maps=np.stack([s["s_map"] for s in self.sources]) if self.sources else np.zeros((0, *self.mask.shape)),
            q_maps=np.stack([s["q_map"] for s in self.sources]) if self.sources else np.zeros((0, *self.mask.shape)),
            **{f"region_{k}": v for k, v in self.regions.items()})
        io.write_json(out_dir / "truth.json", self.meta)
        return {"echoes": paths, "truth": out_dir / "truth.npz", "meta": out_dir / "truth.json"}


# ======================================================================= the simulator
def simulate(cfg: dict, seed=None, overrides: dict | None = None) -> Simulation:
    """Simulate one multi-echo run.

    cfg: the project config (TEs and TR come from cfg["acquisition"], everything else from
    cfg["simulate"] with `overrides` merged in). seed: int, SeedSequence or None (then
    SeedSequence(project_seed, spawn_key=(1,)), stage 1 = simulation).
    """
    P = io.merge(cfg["simulate"], overrides)
    te_ms = np.asarray(cfg["acquisition"]["te_ms"], float)
    te_s = te_ms / 1000.0
    tr = float(cfg["acquisition"]["tr_s"])
    voxel_mm = tuple(cfg["acquisition"].get("voxel_mm", [3.281, 3.281, 3.3]))
    ss = seed if isinstance(seed, np.random.SeedSequence) else (
        np.random.SeedSequence(seed) if seed is not None else io.seed_sequence(cfg, 1))
    rng = np.random.default_rng(ss)
    shape = tuple(int(g) for g in P["grid"])
    T = int(P["n_volumes"])
    E = te_ms.size
    sm = float(P["smoothness_vox"])

    # ---------------------------------------------------------------- anatomy
    c = [(n - 1) / 2 for n in shape]
    r = [0.42 * n for n in shape]
    xx, yy, zz = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    brain = ((xx - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((zz - c[2]) / r[2]) ** 2 <= 1.0
    sb = P["skull_base"]
    skull = np.zeros(shape, bool)
    if sb["enabled"]:
        z_lo = int(np.where(brain.any(axis=(0, 1)))[0].min())
        skull = brain & (zz < z_lo + int(sb["slices"]))
    interior = ndimage.binary_erosion(brain, iterations=1) & ~skull

    s0 = np.where(brain, P["s0_mean"] * (1 + P["s0_variation"] * _smooth_field(rng, shape, sm, brain)), 0.0)
    s0 = np.clip(s0, 0.0, None)

    lo, hi = P["t2s_ms_range"]
    f = _smooth_field(rng, shape, sm, brain)
    fb = f[brain]
    t2 = np.full(shape, float(hi))
    t2[brain] = lo + (hi - lo) * (fb - fb.min()) / (fb.max() - fb.min())
    if skull.any():
        fs = (f[skull] - f[skull].mean()) / (f[skull].std() + 1e-12)       # centred on the configured value
        t2[skull] = np.clip(sb["t2s_ms"] * (1 + 0.1 * fs), lo, hi)
    r2s = 1000.0 / t2                                                     # s^-1

    snr0 = P["snr0"]
    sigma0 = 0.0 if snr0 is None else P["s0_mean"] / float(snr0)
    g = np.ones(shape)
    if P["gfactor"]["enabled"]:
        gf = _smooth_field(rng, shape, float(P["gfactor"]["smoothness_vox"]), brain)
        lo_g, hi_g = gf[brain].min(), gf[brain].max()                     # the range spans the brain itself
        g = 1.0 + P["gfactor"]["amplitude"] * np.clip((gf - lo_g) / (hi_g - lo_g), 0, 1)
    sigma = sigma0 * g

    df0 = np.zeros(shape)
    if skull.any():
        z_lo = int(np.where(skull.any(axis=(0, 1)))[0].min())
        depth = 1.0 - (zz - z_lo) / max(int(sb["slices"]), 1)            # 1 at the bottom slice
        df0 = np.where(skull, sb["dephasing_hz_max"] * np.clip(depth, 0, 1), 0.0)
    kappa = float(sb["resp_modulation"]) if sb["enabled"] else 0.0
    if np.max(df0) * (1 + 3 * abs(kappa)) * te_s.max() >= 1.0:
        raise ValueError("skull_base dephasing reaches the first zero of sinc(df * TE); lower dephasing_hz_max")

    # ---------------------------------------------------------------- time courses and maps
    lo_hz, hi_hz = P["band_hz"]
    resp = _respiration(rng, T, tr, P["respiration"]["freq_hz"], P["respiration"]["freq_jitter_hz"])
    sources, centers = [], []
    for spec in P["sources"]:
        kind, extent = spec["kind"], spec.get("extent", "focal")
        if kind not in KINDS or extent not in EXTENTS:
            raise ValueError(f"source {spec.get('name')}: kind must be one of {KINDS}, extent one of {EXTENTS}")
        tc = _bandlimited(rng, T, tr, lo_hz, hi_hz)
        if extent == "focal":
            rad = float(spec.get("radius_vox", 1.5))
            cand = np.argwhere(interior)
            for _ in range(1000):
                ctr = cand[rng.integers(len(cand))]
                if all(np.linalg.norm(ctr - o) >= 4 * rad for o in centers):
                    break
            else:
                raise ValueError("cannot place focal sources without overlap; enlarge the grid or reduce radius_vox")
            centers.append(ctr)
            d2 = (xx - ctr[0]) ** 2 + (yy - ctr[1]) ** 2 + (zz - ctr[2]) ** 2
            pattern = np.exp(-d2 / (2 * rad ** 2)) * brain
        else:   # diffuse: rectified smooth field (spatially extended, positive, non-Gaussian)
            pattern = np.clip(_smooth_field(rng, shape, sm, brain), 0, None) * brain
            ctr = None
        pattern = pattern / pattern.max()
        q = float(spec.get("q", 0.0)) if kind in ("r2", "mixed") else 0.0
        s = float(spec.get("s", 0.0)) if kind in ("s0", "mixed") else 0.0
        sources.append({"name": spec["name"], "kind": kind, "extent": extent, "timecourse": tc,
                        "s_map": s * pattern, "q_map": q * pattern,
                        "center": None if ctr is None else [int(v) for v in ctr]})
    if P["respiration"]["enabled"]:
        pat = 0.5 + 0.5 * (lambda h: (h - h[brain].min()) / (h[brain].max() - h[brain].min()))(_smooth_field(rng, shape, 2 * sm, brain))
        sources.append({"name": "global_resp", "kind": "r2", "extent": "global", "timecourse": resp,
                        "s_map": np.zeros(shape), "q_map": float(P["respiration"]["q"]) * pat * brain, "center": None})

    # ---------------------------------------------------------------- spikes
    sp = P["spikes"]
    n_sp = int(sp["n"])
    spike_frames = np.sort(rng.choice(np.arange(2, max(T - 2, 3)), size=n_sp, replace=False)) if n_sp else np.zeros(0, int)
    spike_region = (brain & ~ndimage.binary_erosion(brain, iterations=2)) if sp["region"] == "edge" else brain.copy()
    spike_jumps = {int(t): float(sp["s0"]) + float(sp["echo_sd"]) * rng.standard_normal(E) for t in spike_frames}

    # slow model violation: a drift-paced change in decay curvature (outside span{1, TE}), like head-position
    # dependent dephasing; ln S += A * m_v * d(t) * p_e, d slow (unit SD, below cutoff_hz), p_e = centred TE^2
    sv = P.get("slow_violation") or {"enabled": False}
    slow = None
    if sv.get("enabled"):
        d = _bandlimited(rng, T, tr, 0.0, float(sv.get("cutoff_hz", 0.01)))
        pe = (te_s - te_s.mean()) ** 2
        pe = pe - pe.mean()
        pe = pe / np.abs(pe).max()
        mv = 0.5 + 0.5 * np.clip(_smooth_field(rng, shape, sm, brain), -1, 1)
        slow = {"timecourse": d, "pattern": pe, "map": mv, "amplitude": float(sv.get("amplitude", 0.01))}

    # optional CSF partial volume: a static long-T2* compartment that carries no BOLD; the tissue
    # compartment (with all sources) is scaled by (1 - f). Random numbers are drawn only when enabled,
    # so every other simulation is unchanged.
    pv = P.get("csf_partial_volume") or {"enabled": False}
    fcsf = np.zeros(shape)
    if pv.get("enabled"):
        fld = _smooth_field(rng, shape, sm, brain)
        fcsf = np.where(brain, float(pv.get("fraction_max", 0.6)) * np.clip(0.5 + 0.5 * fld, 0, 1), 0.0)
    csf_sig = (float(pv.get("s0_ratio", 1.0)) * s0[..., None] * np.exp(-te_s * 1000.0 / float(pv.get("t2s_ms", 200.0)))
               if pv.get("enabled") else None)                                  # (X, Y, Z, E)

    # ---------------------------------------------------------------- signal + noise, in voxel chunks
    flat = lambda a: a.reshape(-1, *a.shape[3:]) if a.ndim > 3 else a.reshape(-1)
    nvox = int(np.prod(shape))
    K = len(sources)
    Xtc = np.stack([s["timecourse"] for s in sources]) if K else np.zeros((0, T))
    Smap = np.stack([flat(s["s_map"]) for s in sources], axis=1) if K else np.zeros((nvox, 0))
    Qmap = np.stack([flat(s["q_map"]) for s in sources], axis=1) if K else np.zeros((nvox, 0))
    b_flat, s0_f, r2_f, df_f, sig_f = flat(brain), flat(s0), flat(r2s), flat(df0), flat(sigma)
    spk_f = flat(spike_region)
    L = int(P["n_coils"])
    esc = np.asarray(P.get("echo_sigma_scale") or [1.0] * E, float)
    if esc.size != E:
        raise ValueError("echo_sigma_scale needs one value per echo")
    data = np.empty((nvox, E, T), np.float32)
    step = int(P["chunk_voxels"])
    for i0 in range(0, nvox, step):
        sl = slice(i0, min(i0 + step, nvox))
        n = sl.stop - sl.start
        S = np.zeros((n, E, T))
        inb = b_flat[sl]
        if inb.any():
            a = Smap[sl][inb] @ Xtc                                             # (v, T)
            b = Qmap[sl][inb] @ Xtc
            logS = (np.log(s0_f[sl][inb])[:, None, None] + a[:, None, :]
                    - te_s[None, :, None] * (r2_f[sl][inb][:, None, None] + b[:, None, :]))
            dfv = df_f[sl][inb]
            if np.any(dfv):
                dft = dfv[:, None, None] * (1 + kappa * resp[None, None, :])
                logS = logS + np.log(np.abs(np.sinc(dft * te_s[None, :, None])))
            spk = spk_f[sl][inb]
            for t, jump in spike_jumps.items():
                logS[spk, :, t] += jump[None, :]
            if slow is not None:
                mv = flat(slow["map"])[sl][inb]
                logS += slow["amplitude"] * mv[:, None, None] * slow["pattern"][None, :, None] * slow["timecourse"][None, None, :]
            S[inb] = np.exp(logS)
            if csf_sig is not None:
                fv = flat(fcsf)[sl][inb]
                S[inb] = S[inb] * (1 - fv)[:, None, None] + (fv[:, None] * flat(csf_sig)[sl][inb])[:, :, None]
        if sigma0 > 0:
            sg = sig_f[sl][:, None, None] * esc[None, :, None]           # per-echo noise scale (1 = equal)
            M2 = (S + sg * rng.standard_normal((n, E, T))) ** 2 + (sg * rng.standard_normal((n, E, T))) ** 2
            if L > 1:
                M2 += sg ** 2 * rng.chisquare(2 * (L - 1), size=(n, E, T))
            S = np.sqrt(M2)
        data[sl] = S
    data = data.reshape(*shape, E, T)

    mu_base = s0[..., None] * np.exp(-te_s * r2s[..., None]) * np.abs(np.sinc(df0[..., None] * te_s))
    if csf_sig is not None:
        mu_base = mu_base * (1 - fcsf[..., None]) + fcsf[..., None] * csf_sig
    sim = Simulation(data=data, te_ms=te_ms, tr_s=tr, voxel_mm=voxel_mm, mask=brain, s0=s0, t2s_ms=t2,
                     sigma=sigma, mu_base=mu_base, dephasing_hz=df0, sources=sources,
                     regions={"brain": brain, "interior": interior, "skull_base": skull, "spike": spike_region & brain},
                     spike_frames=spike_frames)
    sim.csf_fraction = fcsf
    sim.meta = {"params": P, "te_ms": te_ms.tolist(), "tr_s": tr, "n_coils": L, "echo_sigma_scale": esc.tolist(),
                "seed": {"entropy": int(ss.entropy), "spawn_key": list(ss.spawn_key)},
                "spike_frames": spike_frames.tolist(), "focal_centers": {s["name"]: s["center"] for s in sources if s["center"]},
                "slow_violation_timecourse": slow["timecourse"].tolist() if slow is not None else None,
                "tsnr": sim.tsnr() if sigma0 > 0 else None, "thermal_snr": sim.thermal_snr() if sigma0 > 0 else None}
    return sim
