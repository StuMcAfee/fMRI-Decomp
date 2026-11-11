"""Route B, plan step 2 (docs/ROUTE_B_PLAN.md): region averaging, the split, the region noise estimate, shares,
coupling and the circular-shift null, on synthetic voxels at the real echo times and noise ratios.

- averaging keeps the model linear: noise-free regions return their a(t) and b(t) exactly;
- the region noise measured from the leftover channels matches the truth, including when neighbouring voxels'
  noise is correlated (motion-correction interpolation); the independent-voxel value would be too small;
- BOLD-like share ~1 for a pure R2* region, ~0 for a pure S0 region, the energy-weighted truth for a mixed one;
- dimensionality: regions driven by independent time courses have a higher participation ratio than regions
  sharing one; under independence the circular-shift null holds the observed values;
- segregation is positive when correlations are higher within networks than between them.
"""
import os

import numpy as np
import pytest

from mejoint import inject, io, regions

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))
TE = np.array([16.20, 31.22, 46.24, 61.26])
RATIOS = np.array([1.0, 0.721, 0.631, 0.560])
TR = 0.75
BAND = (0.01, 0.10)


def voxels(rng, P, n_per, T, a, b, sigma=8.0, rho_noise=0.0, s0=1000.0):
    """P regions of n_per voxels. a, b: (P, T) region S0-like (fraction) and R2*-like (s^-1) changes (every voxel
    of a region carries them). Gaussian thermal noise with per-echo ratios; rho_noise: correlation of the noise
    between voxels of the same region (interpolation-like)."""
    V = P * n_per
    labels = np.repeat(np.arange(1, P + 1), n_per)
    t2s = rng.uniform(35, 60, V)
    mu_true = s0 * np.exp(-TE[None, :] / t2s[:, None])                     # (V, E)
    y = a[labels - 1][:, None, :] - (TE[None, :, None] / 1000.0) * b[labels - 1][:, None, :]
    S = mu_true[:, :, None] * np.exp(y)
    sd = sigma * RATIOS[None, :, None]
    if rho_noise > 0:
        shared = np.repeat(rng.standard_normal((P, 4, T)), n_per, axis=0)
        own = rng.standard_normal((V, 4, T))
        noise = np.sqrt(rho_noise) * shared + np.sqrt(1 - rho_noise) * own
    else:
        noise = rng.standard_normal((V, 4, T))
    S = S + sd * noise
    mu = S.mean(axis=2)
    return S, labels, mu, np.full(V, sigma), np.ones((V, 4), bool)


def slow(rng, n, T):
    return inject.time_courses(T, TR, BAND, n, rng).T                    # (n, T), unit SD, orthogonal


def test_noise_free_regions_recover_a_and_b():
    rng = np.random.default_rng(SEED)
    P, T = 3, 400
    u = slow(rng, 3, T)
    a = np.stack([0 * u[0], 0.01 * u[1], 0.016 * u[2]])
    b = np.stack([0.5 * u[0], 0 * u[1], 0.5 * u[2]])
    S, labels, mu, sigma, use = voxels(rng, P, 30, T, a, b, sigma=1e-6)
    rs = regions.region_signals(S, labels, mu, sigma, RATIOS, use)
    sp = regions.split(rs["ybar"], rs["wsum"], TE)
    for p in range(P):
        assert np.allclose(sp["a"][p], a[p] - a[p].mean(), atol=2e-4), p
        assert np.allclose(sp["b"][p], b[p] - b[p].mean(), atol=0.01), p


@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_region_noise_from_leftover_channels(rho):
    rng = np.random.default_rng(SEED + 1)
    P, n_per, T = 6, 60, 600
    zero = np.zeros((P, T))
    S, labels, mu, sigma, use = voxels(rng, P, n_per, T, zero, zero, rho_noise=rho)
    rs = regions.region_signals(S, labels, mu, sigma, RATIOS, use)
    sp = regions.split(rs["ybar"], rs["wsum"], TE)
    nv = regions.leftover_noise(sp["r"])
    # truth: independent voxels give 1; a shared fraction rho multiplies the region-average variance by
    # 1 + rho (n_per - 1)
    truth = 1.0 + rho * (n_per - 1)
    assert np.all(np.abs(nv / truth - 1) < 0.15), (nv, truth)
    # the channels' own variance agrees (no signal): the leftover estimate is the right noise for c0, c1
    assert np.all(np.abs(sp["c"].var(axis=2).mean(1) / nv - 1) < 0.2)


def test_shares_pure_and_mixed():
    rng = np.random.default_rng(SEED + 2)
    P, T = 3, 800
    u = slow(rng, 3, T)
    a = np.stack([0 * u[0], 0.006 * u[1], 0.008 * u[2]])
    b = np.stack([0.3 * u[0], 0 * u[1], 0.25 * u[2]])
    S, labels, mu, sigma, use = voxels(rng, P, 80, T, a, b, rho_noise=0.1)
    rs = regions.region_signals(S, labels, mu, sigma, RATIOS, use)
    sp = regions.split(rs["ybar"], rs["wsum"], TE)
    nv = regions.leftover_noise(sp["r"])
    sh = regions.shares(sp, nv, TR, BAND)
    # truth for the mixed region: the same energies computed on noise-free data
    S0, *_ = voxels(np.random.default_rng(9), P, 80, T, a, b, sigma=1e-6)
    sp0 = regions.split(regions.region_signals(S0, labels, S0.mean(2), np.full(S0.shape[0], 8.0), RATIOS, use)["ybar"],
                        rs["wsum"], TE)
    sh0 = regions.shares(sp0, np.zeros(P), TR, BAND)
    assert sh["bold_share"][0] > 0.95, sh["bold_share"]
    assert sh["bold_share"][1] < 0.05, sh["bold_share"]
    assert abs(sh["bold_share"][2] - sh0["bold_share"][2]) < 0.05, (sh["bold_share"], sh0["bold_share"])
    assert np.all(np.abs(sh["leftover_share"]) < 0.05), sh["leftover_share"]


def test_curvature_shows_as_leftover():
    """A compartment-volume change (not linear in TE) puts energy in the leftover share."""
    rng = np.random.default_rng(SEED + 3)
    P, n_per, T = 2, 80, 800
    u = slow(rng, 2, T)
    zero = np.zeros((P, T))
    S, labels, mu, sigma, use = voxels(rng, P, n_per, T, zero, np.stack([0.3 * u[0], 0 * u[1]]))
    # region 2: an extra signal with a curved (U-shaped) echo profile, outside span{1, TE}. (A profile strong at
    # echo 1 only is mostly absorbed as an S0 + R2* mixture: only its non-linear part reaches the leftover.)
    prof = np.array([1.0, -1.0, -1.0, 1.0]) * 0.005
    S[labels == 2] *= np.exp(prof[None, :, None] * u[1][None, None, :])
    mu = S.mean(2)
    rs = regions.region_signals(S, labels, mu, sigma, RATIOS, use)
    sp = regions.split(rs["ybar"], rs["wsum"], TE)
    sh = regions.shares(sp, regions.leftover_noise(sp["r"]), TR, BAND)
    assert sh["leftover_share"][0] < 0.05 and sh["leftover_share"][1] > 0.1, sh["leftover_share"]


def test_dimensionality_and_circular_shift_null():
    rng = np.random.default_rng(SEED + 4)
    P, T = 16, 800
    nets = np.repeat(np.arange(4), 4)
    nv = np.full(P, 0.01)
    # A: four networks, each driven by its own time course (plus a little own signal)
    drivers = slow(rng, 4, T)
    own = np.stack([slow(rng, 1, T)[0] for _ in range(P)])
    xa = drivers[nets] + 0.4 * own + rng.standard_normal((P, T)) * 0.1
    # B: one time course drives everything
    xb = drivers[[0] * P] + 0.4 * own + rng.standard_normal((P, T)) * 0.1
    ca, cb = regions.coupling(xa, nv, TR, BAND), regions.coupling(xb, nv, TR, BAND)
    pra, prb = regions.participation_ratio(ca["corr"]), regions.participation_ratio(cb["corr"])
    assert pra > 2.5 and prb < 1.5 and pra > prb, (pra, prb)
    seg = regions.segregation(ca["corr"], nets)
    assert seg > 0.5, seg
    null = regions.shift_null(xa, nv, TR, BAND, 200, 40, rng, networks=nets)
    assert pra < np.quantile(null["pr"], 0.05)                       # integration beyond chance: lower dimensionality
    k = regions.count_above_null(regions.eigenvalues(ca["corr"]), null["eig"])
    assert k == 4, k
    # independent regions: the observed participation ratio lies inside its null, about 5 % of edges beyond it
    # drawn independently (one orthogonalized set would make the regions exactly uncorrelated, unlike real data)
    xi = np.stack([slow(rng, 1, T)[0] for _ in range(P)]) + rng.standard_normal((P, T)) * 0.1
    ci = regions.coupling(xi, nv, TR, BAND)
    ni = regions.shift_null(xi, nv, TR, BAND, 200, 40, rng)
    lo, hi = np.quantile(ni["pr"], [0.01, 0.99])
    assert lo <= regions.participation_ratio(ci["corr"]) <= hi
    iu = np.triu_indices(P, 1)
    thr = np.quantile(np.abs(ni["edges"]), 0.95, axis=0)
    frac = np.mean(np.abs(ci["corr"][iu]) > thr)
    assert frac < 0.15, frac


def test_global_share():
    rng = np.random.default_rng(SEED + 5)
    P, T = 12, 800
    g = slow(rng, 1, T)[0]
    x = 0.8 * g[None, :] + 0.6 * slow(rng, P, T)
    c = regions.coupling(x, np.zeros(P), TR, BAND)
    gs = regions.global_share(c["x_band"], c["signal_var"], c["noise_band_var"])
    assert 0.55 < gs < 0.8, gs                                        # truth: var(mean) / mean var = (0.64 + 0.36 / 12) / 1 = 0.67
    xi = np.stack([slow(rng, 1, T)[0] for _ in range(P)])
    ci = regions.coupling(xi, np.zeros(P), TR, BAND)
    assert abs(regions.global_share(ci["x_band"], ci["signal_var"], ci["noise_band_var"])) < 0.2   # independent regions


def _cfg_small():
    from mejoint import io
    c = io.load_config()
    return io.merge(c, {"regions": {"n_null": 100, "n_boot": 50}})


def test_analyze_networks_end_to_end():
    """16 regions in 4 networks, each network driven by its own R2*-like time course (plus each region's own weak
    signal): shares ~1 with intervals that contain them, 4 eigenvalues above the circular-shift null, positive
    segregation; the common-set path gives the same answer on a subset of whole networks."""
    rng = np.random.default_rng(SEED + 6)
    P, n_per, T = 16, 40, 800
    nets = np.repeat(np.arange(4), 4)
    drivers = slow(rng, 4, T)
    own = np.stack([slow(rng, 1, T)[0] for _ in range(P)])
    b = 0.25 * drivers[nets] + 0.08 * own
    a = np.zeros((P, T))
    S, labels, mu, sigma, use = voxels(rng, P, n_per, T, a, b)
    table = [{"id": i + 1, "name": f"r{i + 1}", "kind": "cortex", "hemisphere": "LH", "network": f"N{nets[i]}",
              "n_voxels": n_per, "usable": True} for i in range(P)]
    res = regions.analyze(S, labels, S.mean(2), sigma, RATIOS, use, TE, table, _cfg_small(), common_ids=list(range(1, 9)))
    sh = np.array([r["bold_share"] for r in res["regions"]])
    assert np.all(sh > 0.9), sh
    ci = np.array([r["bold_share_ci"] for r in res["regions"]])
    assert np.all(ci[:, 0] <= sh) and np.all(sh <= ci[:, 1])
    full, common = res["integration"]
    assert full["n_regions"] == 16 and full["n_eigen_above_null"] == 4, full
    assert full["segregation"] > 0.5 and full["pr_percentile_in_null"] < 0.05, full
    assert common["n_regions"] == 8 and common["n_eigen_above_null"] == 2, common
    assert res["corr_b"].shape == (16, 16)


def test_run_dir_files(tmp_path):
    """Route B on a simulated run through the real file layout (front end folder, anatomy folder with labels and
    the region table): outputs written, every usable region analysed."""
    import json
    import sys

    import nibabel as nib

    sys.path.insert(0, os.path.dirname(__file__))
    from test_frontend import make_run_folder

    from mejoint import frontend, io, simulate
    cfg = io.merge(_cfg_small(), {"frontend": {"mppca": {"enabled": False}}, "nuisance": {"n_surrogates": 1}})
    cfg = io.merge(cfg, {"simulate": {"echo_sigma_scale": cfg["frontend"]["echo_noise_ratios"]}})
    G = {"grid": [24, 24, 14], "n_volumes": 200, "spikes": {"n": 0}, "skull_base": {"enabled": False},
         "respiration": {"enabled": False}, "sources": []}
    sim = simulate.simulate(cfg, seed=SEED + 21, overrides=G)
    rd = make_run_folder(sim, tmp_path)
    fe = tmp_path / "fe"
    frontend.run_subject(rd, cfg, fe, source="raw")
    mimg = nib.load(str(fe / "analysis_mask.nii.gz"))
    mask = np.asarray(mimg.dataobj) > 0
    lab = np.zeros(mask.shape, np.int32)
    xs = np.indices(mask.shape)
    blocks = (xs[0] // 6) * 4 + (xs[1] // 6) + 1                      # 16 blocks
    lab[mask] = blocks[mask]
    A = tmp_path / "anat"
    io.save_volume(lab.astype(np.float32), mimg, A / "labels_200_7.nii.gz")
    rows = [{"id": i, "name": f"r{i}", "kind": "cortex", "hemisphere": "LH", "network": f"N{(i - 1) // 4}",
             "n_voxels": int((lab == i).sum()), "usable": bool((lab == i).sum() >= 20)} for i in range(1, 17)]
    io.write_json(A / "regions_200_7.json", {"regions": rows})
    csf = np.zeros(mask.shape, np.float32)
    csf[xs[0] < 3] = 0.9
    io.save_volume(csf, mimg, A / "csf_epi.nii.gz")
    io.save_volume(lab.astype(np.float32), mimg, A / "labels_200_7_allvox.nii.gz")   # all-voxel sensitivity labels
    io.write_json(A / "regions_200_7_allvox.json", {"regions": rows})
    out = regions.run_dir(fe, A, cfg, tmp_path / "rb")
    sets = [i["set"] for i in out["integration"]]
    assert any(x.startswith("cortex, all voxels") for x in sets), sets
    for f in ("route_b.json", "route_b.md", "regions_route_b.tsv", "corr_b.npy", "route_b.png"):
        assert (tmp_path / "rb" / f).exists(), f
    used = [r for r in out["regions"] if r["used"]]
    assert len(used) == sum(r["usable"] for r in rows) and out["integration"][0]["n_regions"] == len(used)
    json.loads((tmp_path / "rb" / "route_b.json").read_text())


def test_inject_regions_place_and_weights():
    from mejoint import inject_regions, io
    cfg = io.load_config()
    table = [{"id": i, "name": f"r{i}", "kind": "cortex", "hemisphere": "LH" if i < 60 else "RH", "network": "N",
              "n_voxels": 15 if i % 10 == 0 else 50, "usable": i % 7 != 0} for i in range(1, 121)]
    table += [{"id": 121, "name": "L_thalamus", "kind": "secondary", "hemisphere": "LH", "network": "Thalamus", "n_voxels": 200, "usable": True}]
    nets = inject_regions.place(table, cfg, np.random.default_rng(SEED))
    assert [n["type"] for n in nets] == ["r2", "r2", "s0", "mixed"]
    ids = [p for n in nets for p in n["parcels"]]
    assert len(ids) == 32 and len(set(ids)) == 32
    ok = {r["id"] for r in table if r["kind"] == "cortex" and r["usable"] and r["n_voxels"] >= 20}
    assert set(ids) <= ok
    lab = np.zeros((10, 10, 3), int); lab.flat[:121] = np.arange(1, 122)
    W = inject_regions.weights(lab, nets)
    assert W.shape == (10, 10, 3, 4) and np.all(W.sum(-1) <= 1) and W[..., 0].sum() == 8


def test_participation_ratio_with_a_region_without_signal():
    """One region without signal above its noise (NaN row and column) must not empty the matrix (bug fixed
    2026-11-10): the participation ratio and eigenvalues come from the remaining regions."""
    C = np.eye(5)
    C[0, 1] = C[1, 0] = 0.5
    full = regions.participation_ratio(C)
    Cn = np.full((6, 6), np.nan)
    Cn[1:, 1:] = C
    assert np.isclose(regions.participation_ratio(Cn), full)
    assert regions.eigenvalues(Cn).size == 5


def test_participation_ratio_without_first_pattern():
    """Secondary measure (user decision 2026-11-10): the participation ratio with the first pattern removed. A
    stronger brain-wide signal lowers the ordinary ratio but leaves this one about unchanged; adding networks on
    top of a dominant brain-wide signal raises the ordinary ratio (the R3 caveat) but lowers this one, because the
    structure beyond the brain-wide pattern becomes more organized (more integrated)."""
    rng = np.random.default_rng(SEED + 7)
    P, T = 16, 800
    nets = np.repeat(np.arange(4), 4)
    nv = np.full(P, 0.01)
    dr, g, own = slow(rng, 4, T), slow(rng, 1, T)[0], slow(rng, P, T)
    noise = rng.standard_normal((P, T)) * 0.1
    pr, pr1 = [], []
    for w in (0.0, 1.0, 2.0):
        c = regions.coupling(dr[nets] + w * g + 0.6 * own + noise, nv, TR, BAND)
        pr.append(regions.participation_ratio(c["corr"]))
        pr1.append(regions.participation_ratio_residual(c["corr"]))
    assert pr[0] > 2 * pr[2], pr
    assert max(pr1) / min(pr1) < 1.15, pr1
    base = g[None].repeat(P, 0) * 2.0 + 0.6 * own + noise
    c0 = regions.coupling(base, nv, TR, BAND)
    c4 = regions.coupling(base + dr[nets], nv, TR, BAND)
    assert regions.participation_ratio(c4["corr"]) > regions.participation_ratio(c0["corr"])
    assert regions.participation_ratio_residual(c4["corr"]) < regions.participation_ratio_residual(c0["corr"])
    null = regions.shift_null(dr[nets] + 0.6 * own + noise, nv, TR, BAND, 50, 40, rng)
    assert null["pr_resid"].shape == (50,) and np.isfinite(null["pr_resid"]).all()


def test_segregation_permutation_null_and_subcortical_noise_correction():
    """R4 fixes: the segregation null permutes network labels (finite, below the observed for real networks,
    around it for random labels); the thalamus-like region's coupling is noise-corrected (close to the true
    correlation with its network although its own noise is large)."""
    rng = np.random.default_rng(SEED + 8)
    P = 16
    nets = np.repeat(np.arange(4), 4)
    T = 800
    x = slow(rng, 4, T)[nets] + 0.5 * slow(rng, P, T)
    c = regions.coupling(x, np.full(P, 1e-4), TR, BAND)
    out = regions._segregation_perm(c["corr"], nets, 200, rng)
    assert out["within_minus_between"] > out["within_minus_between_null_q95"] and out["segregation_percentile_in_null"] > 0.95
    out_r = regions._segregation_perm(c["corr"], rng.permutation(nets), 200, rng)
    assert out_r["segregation_percentile_in_null"] < 0.99
    # end to end: one secondary region driven by network N0 with a lot of noise
    P2, n_per = 9, 40
    nets2 = np.r_[np.repeat(np.arange(2), 4), 0]
    drivers = slow(rng, 2, T)
    own = np.stack([slow(rng, 1, T)[0] for _ in range(P2)])
    b = 0.25 * drivers[nets2] + 0.08 * own
    S, labels, mu, sigma, use = voxels(rng, P2, n_per, T, np.zeros((P2, T)), b)
    sigma = sigma.copy()
    sigma[labels == P2] *= 8.0                                          # the deep region is much noisier
    S = S.copy()
    noisy = labels == P2
    S[noisy] = S[noisy] + rng.standard_normal(S[noisy].shape) * (sigma[noisy] * 0.99)[:, None, None] * np.array(RATIOS)[None, :, None]
    table = [{"id": i + 1, "name": f"r{i + 1}", "kind": "cortex", "hemisphere": "LH", "network": f"N{nets2[i]}",
              "n_voxels": n_per, "usable": True} for i in range(P2 - 1)]
    table.append({"id": P2, "name": "thal", "kind": "secondary", "hemisphere": "LH", "network": "Thalamus", "n_voxels": n_per, "usable": True})
    res = regions.analyze(S, labels, S.mean(2), sigma, RATIOS, use, TE, table, _cfg_small())
    corr, raw = res["secondary_coupling"][0]["N0"], res["secondary_coupling_raw"][0]["N0"]
    assert corr is not None and raw < corr, (raw, corr)
    xb = regions.band_pass(b, TR, BAND)
    truth = np.corrcoef(xb[-1], xb[:4].mean(0))[0, 1]
    assert abs(corr - truth) < 0.15, (corr, truth, raw)
    row = res["secondary_coupling"][0]
    lo, hi = row["ci"]["N0"]
    assert lo <= corr <= hi and lo <= truth + 0.05, (lo, corr, hi, truth)
    rel = {r["name"]: r["b_reliability"] for r in res["regions"]}
    assert rel["thal"] < np.median([rel[f"r{i}"] for i in range(1, 9)]), rel


def test_spectral_noise_estimate_resists_narrow_band_structure():
    """The median-spectrum noise estimate agrees with the frame-difference one on white noise and stays near the
    thermal level when a fast narrow-band signal (a cardiac alias) is added; the frame-difference one rises."""
    rng = np.random.default_rng(SEED + 9)
    P, T = 20, 800
    r = rng.standard_normal((P, 2, T)) * np.sqrt(2.0)
    d, s_ = regions.leftover_noise(r), regions.leftover_noise_spectral(r, TR, 0.1)
    assert np.allclose(np.median(d), 2.0, rtol=0.05) and np.allclose(np.median(s_), 2.0, rtol=0.08), (np.median(d), np.median(s_))
    t = np.arange(T) * TR
    r2 = r + 3.0 * np.sin(2 * np.pi * 0.45 * t + rng.uniform(0, 6.3, (P, 2, 1)))
    d2, s2 = regions.leftover_noise(r2), regions.leftover_noise_spectral(r2, TR, 0.1)
    assert np.median(d2) > 3.0 and np.allclose(np.median(s2), 2.0, rtol=0.1), (np.median(d2), np.median(s2))


def test_deep_csf_check():
    """The CSF diagnostic for deep regions: weight shares per echo in [0, 1], the all-voxel version reproduces
    the main analysis, and removing the CSF-dominated voxels leaves a usable region."""
    rng = np.random.default_rng(SEED + 10)
    P, n_per, T = 9, 40, 600
    nets = np.r_[np.repeat(np.arange(2), 4), 0]
    drivers = slow(rng, 2, T)
    b = 0.25 * drivers[nets] + 0.08 * slow(rng, P, T)
    S, labels, mu, sigma, use = voxels(rng, P, n_per, T, np.zeros((P, T)), b)
    table = [{"id": i + 1, "name": f"r{i + 1}", "kind": "cortex", "hemisphere": "LH", "network": f"N{nets[i]}",
              "n_voxels": n_per, "usable": True} for i in range(P - 1)]
    table.append({"id": P, "name": "thal", "kind": "secondary", "hemisphere": "LH", "network": "Thalamus", "n_voxels": n_per, "usable": True})
    cfg = _cfg_small()
    res = regions.analyze(S, labels, S.mean(2), sigma, RATIOS, use, TE, table, cfg)
    csf = np.zeros(labels.size)
    csf[np.flatnonzero(labels == P)[:8]] = 0.9
    out = regions.deep_csf_check(S, labels, S.mean(2), sigma, RATIOS, use, TE, table, cfg, None, res, csf)
    d = out[0]
    assert d["n_csf_dominated"] == 8 and all(0 <= x <= 1 for x in d["csf_weight_share_by_echo"])
    main = [r for r in res["regions"] if r["name"] == "thal"][0]
    assert abs(d["all_voxels"]["bold_share"] - main["bold_share"]) < 1e-3
    assert d["csf_below_0.5"]["n_voxels"] == n_per - 8 and d["csf_below_0.5"]["N0"] is not None


def test_deflated_eigen_count():
    """A dominant brain-wide pattern pushes later eigenvalues below the plain null; the deflated count still finds
    the structure (4 networks on top of a strong common signal: plain 1, deflated 4) and finds none in
    independent regions."""
    rng = np.random.default_rng(SEED + 11)
    P, T = 60, 800
    nets = np.repeat(np.arange(4), 15)
    nv = np.full(P, 0.01)
    x = 2.0 * slow(rng, 1, T) + 0.5 * slow(rng, 4, T)[nets] + slow(rng, P, T) + 0.1 * rng.standard_normal((P, T))
    c = regions.coupling(x, nv, TR, BAND)
    null = regions.shift_null(x, nv, TR, BAND, 200, 40, rng)
    e = regions.eigenvalues(c["corr"])
    assert regions.count_above_null(e, null["eig"]) <= 2
    assert regions.count_above_null_deflated(e, null["eig"]) == 4
    xi = slow(rng, P, T) + 0.1 * rng.standard_normal((P, T))
    ci = regions.coupling(xi, nv, TR, BAND)
    ni = regions.shift_null(xi, nv, TR, BAND, 200, 40, rng)
    assert regions.count_above_null_deflated(regions.eigenvalues(ci["corr"]), ni["eig"]) <= 1


def test_adjacency_and_shared_noise_check():
    """Neighbouring regions sharing noise: the leftover-channel correlation finds it for adjacent pairs only, and
    removing it brings the noise-corrected correlations of independent regions back towards 0."""
    lab = np.zeros((6, 2, 2), int)
    for i in range(6):
        lab[i] = i + 1
    A = regions.adjacency(lab, range(1, 7))
    assert A[0, 1] and A[1, 2] and not A[0, 2] and A.sum() == 10
    rng = np.random.default_rng(SEED + 12)
    P, T = 6, 4000
    sh = rng.standard_normal((P + 1, 2, T))
    noise = rng.standard_normal((P, 2, T)) + 0.7 * (sh[:-1] + sh[1:])  # neighbours share one source each side
    xb_sig = slow(rng, P, T) * 0.3                                     # independent signals
    b = xb_sig + noise[:, 0]
    nv = noise[:, 0].var(1)
    c = regions.coupling(b, nv, TR, BAND)
    out = regions.shared_noise_check(noise, c["cov"], c["noise_band_var"], A)
    assert out["noise_corr_adjacent_median"] > 0.2 and abs(out["noise_corr_nonadjacent_median"]) < 0.05, out
    iu = np.triu_indices(P, 1)
    before = np.nanmedian(np.abs(c["corr"][iu][A[iu]]))
    assert out["participation_ratio_shared_noise_removed"] > regions.participation_ratio(c["corr"]), (out, before)


def test_select_frames():
    """R5-1: first n frames of an unbroken run; the longest gap-free stretch when there is a gap; not eligible for
    the primary when that stretch is too short."""
    out = regions.select_frames(None, 800, 751)
    assert out["primary_eligible"] and out["frames"][0] == 0 and out["n_used"] == 751
    vols = np.r_[np.arange(0, 300), np.arange(319, 800)]               # a 19-volume gap after frame 300
    out = regions.select_frames(vols, vols.size, 751)
    assert not out["primary_eligible"] and out["n_used"] == 481 and out["longest_stretch"] == [300, 781]
    out = regions.select_frames(vols, vols.size, 400)
    assert out["primary_eligible"] and out["frames"][0] == 300 and out["n_used"] == 400


def test_global_regression_and_censoring():
    """Sensitivity switches (user, 2026-11-12): region-level global signal regression removes a shared signal
    (global share falls, dimensionality rises); censoring leaves corrupted frames out of the covariance."""
    rng = np.random.default_rng(SEED + 13)
    P, n_per, T = 12, 40, 600
    nets = np.repeat(np.arange(3), 4)
    g = slow(rng, 1, T)[0]
    b = 0.25 * slow(rng, 3, T)[nets] + 0.3 * g[None, :] + 0.05 * slow(rng, P, T)
    S, labels, mu, sigma, use = voxels(rng, P, n_per, T, np.zeros((P, T)), b)
    table = [{"id": i + 1, "name": f"r{i + 1}", "kind": "cortex", "hemisphere": "LH", "network": f"N{nets[i]}",
              "n_voxels": n_per, "usable": True} for i in range(P)]
    cfg = _cfg_small()
    base = regions.analyze(S, labels, S.mean(2), sigma, RATIOS, use, TE, table, cfg)["integration"][0]
    cfg_g = io.merge(cfg, {"regions": {"global_regression": True}})
    gsr = regions.analyze(S, labels, S.mean(2), sigma, RATIOS, use, TE, table, cfg_g)["integration"][0]
    assert gsr["global_share"] < 0.5 * base["global_share"] and gsr["participation_ratio"] > base["participation_ratio"], (base, gsr)
    # censoring: corrupt 10 % of frames with large region-specific spikes
    S2 = S.copy()
    bad = rng.choice(T, T // 10, replace=False)
    spikes = rng.standard_normal((P, bad.size)) * 3.0                 # region-specific dR2* spikes (s^-1)
    te = np.asarray(TE, float)[None, :, None] / 1000.0
    for p in range(P):
        v = labels == p + 1
        S2[np.ix_(v, np.arange(len(TE)), bad)] *= np.exp(-te * spikes[p][None, None, :])
    censor = np.zeros(T, bool)
    censor[bad] = True
    r_bad = regions.analyze(S2, labels, S2.mean(2), sigma, RATIOS, use, TE, table, cfg)
    r_cen = regions.analyze(S2, labels, S2.mean(2), sigma, RATIOS, use, TE, table, cfg, censor=censor)
    assert r_cen["censored_fraction"] == 0.1
    err = lambda r: abs(r["integration"][0]["participation_ratio"] - base["participation_ratio"])
    assert err(r_cen) < err(r_bad), (err(r_cen), err(r_bad))
