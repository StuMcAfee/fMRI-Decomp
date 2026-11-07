"""Phase 5 injection calibration (inject.py).

- time courses: zero mean, unit SD, mutually orthogonal, inside the band;
- injection is exact in the log: ln(S'/mu) = w (a - TE b) u for noise-free data;
- file injection: paired volumes respected, voxels outside the spheres bit-identical, integer type kept;
- placement: every region gets every type, centres separated, inferior sources in the inferior third;
- end to end on simulated echoes (front end, joint ICA, recovery): sources found with |r| > 0.9, the
  oracle eta is ~0 for dR2*, ~1 for dS0, and the recovered dS0/dR2* ratio is ~0 ms for dR2* and the
  injected 32 ms for the mixed source.
"""
import json
import os

import nibabel as nib
import numpy as np
import pytest

from mejoint import decompose, frontend, inject, io, metrics, simulate

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))
TE = [16.20, 31.22, 46.24, 61.26]


def test_time_courses_orthogonal_in_band():
    rng = np.random.default_rng(SEED)
    U = inject.time_courses(600, 0.75, [0.01, 0.1], 9, rng)
    assert np.allclose(U.mean(0), 0, atol=1e-10)
    assert np.allclose(U.std(0), 1, atol=1e-10)
    C = U.T @ U / 600
    assert np.abs(C - np.eye(9)).max() < 1e-8
    P = np.abs(np.fft.rfft(U, axis=0)) ** 2
    f = np.fft.rfftfreq(600, 0.75)
    out = (f < 0.01) | (f > 0.1)
    assert P[out].sum() / P.sum() < 1e-10


def test_injection_exact_in_log():
    rng = np.random.default_rng(SEED)
    V, T = 30, 200
    mu = rng.uniform(200, 2000, (V, 4))
    S = np.repeat(mu[:, :, None], T, axis=2)
    w = rng.uniform(0, 1, (V, 3))
    U = inject.time_courses(T, 0.75, [0.01, 0.1], 3, rng)
    a, b = [0.0, 0.01, 0.016], [0.5, 0.0, 0.5]
    S2 = inject.inject_masked(S, w, U, a, b, TE)
    y = np.log(S2 / S2.mean(axis=2, keepdims=True))
    exp = inject.log_change(w, U, a, b, TE)
    # the temporal mean of exp(y) is not exactly 1, so compare after removing each row's mean
    assert np.allclose(y - y.mean(2, keepdims=True), exp - exp.mean(2, keepdims=True), atol=1e-12)


def test_inject_file_pairs_and_preserves(tmp_path):
    rng = np.random.default_rng(SEED)
    shape, Traw = (10, 12, 8), 50
    base = rng.integers(9000, 11000, shape + (Traw,)).astype(np.int16)
    src = tmp_path / "in_e1.nii.gz"
    nib.save(nib.Nifti1Image(base, np.eye(4)), str(src))
    frames = np.delete(np.arange(Traw), 7)                             # raw volume 7 not used (paired away)
    T = frames.size
    W3 = inject.weight_maps(shape, [[5, 6, 4]], 3.0, [1, 1, 1])
    U = inject.time_courses(T, 0.75, [0.01, 0.2], 1, rng)
    info = inject.inject_file(src, tmp_path / "out" / "in_e1.nii.gz", 1, frames, W3, U, [0.0], [0.5], TE)
    img = nib.load(info["file"])
    assert img.get_data_dtype() == np.int16
    out = np.asarray(img.dataobj)
    inside = W3[..., 0] > 0
    assert np.array_equal(out[~inside], base[~inside])                 # untouched outside the sphere
    assert np.array_equal(out[..., 7], base[..., 7])                   # unused raw volume untouched
    c = (5, 6, 4)
    got = out[c][frames].astype(float) - base[c][frames].astype(float)
    mu = base[c][frames].mean()
    want = mu * np.expm1(-TE[1] / 1000 * 0.5 * U[:, 0])
    assert np.abs(got - want).max() <= 0.51                           # integer rounding only


def test_place_sources_regions_and_separation():
    shape = (52, 64, 48)
    idx = np.indices(shape).astype(float)
    c = np.array([25.5, 31.5, 23.5])[:, None, None, None]
    mask = (((idx - c) / np.array([22, 28, 20])[:, None, None, None]) ** 2).sum(0) <= 1
    aff = np.diag([3.281, 3.281, 3.3, 1.0])                            # RAS: axis 1 anterior, axis 2 superior
    cfg = io.load_config()
    rng = np.random.default_rng(SEED)
    src = inject.place_sources(mask, aff, [3.281, 3.281, 3.3], cfg, rng)
    assert len(src) == 9
    assert {(s["region"], s["type"]) for s in src} == {(r, t) for r in inject.REGIONS for t in inject.TYPES}
    vox = np.array([3.281, 3.281, 3.3])
    P = np.array([s["center"] for s in src])
    D = np.sqrt((((P[:, None] - P[None]) * vox) ** 2).sum(-1)) + np.eye(9) * 1e9
    assert D.min() >= cfg["inject"]["min_separation_mm"]
    inf = frontend.inferior_third(mask, aff)
    for s in src:
        assert mask[tuple(s["center"])]
        assert inf[tuple(s["center"])] == (s["region"] == "inferior")
        if s["region"] == "inferior":
            assert s["center"][1] <= 31.5                              # posterior half


@pytest.fixture(scope="module")
def recovered():
    cfg = io.load_config()
    cfg = io.merge(cfg, {"simulate": {"echo_sigma_scale": cfg["frontend"]["echo_noise_ratios"]},
                         "metrics": {"n_boot": 5, "band_mc": 20, "n_null": 20, "n_null_pipeline": 5},
                         "decompose": {"n_restarts": 6}})
    G = {"grid": [32, 32, 18], "n_volumes": 400, "spikes": {"n": 0}, "skull_base": {"enabled": False},
         "respiration": {"enabled": False}, "sources": [
             {"name": "native_r2", "kind": "r2", "extent": "focal", "q": 0.5, "radius_vox": 1.5},
             {"name": "native_s0", "kind": "s0", "extent": "focal", "s": 0.015, "radius_vox": 1.5}]}
    sim = simulate.simulate(cfg, seed=SEED * 100 + 71, overrides=G)
    m = sim.mask
    rng = np.random.default_rng(SEED + 5)
    inside = np.argwhere(m)
    centers, vox = [], np.asarray(sim.voxel_mm, float)
    for p in inside[rng.permutation(len(inside))]:                    # three sources well inside, apart
        if all(np.sqrt((((p - q) * vox) ** 2).sum()) > 25 for q in centers) and \
                all(m[tuple(np.clip(p + d, 0, np.array(m.shape) - 1))] for d in np.array([[3, 0, 0], [-3, 0, 0], [0, 3, 0], [0, -3, 0], [0, 0, 2], [0, 0, -2]])):
            centers.append(p)
        if len(centers) == 3:
            break
    W3 = inject.weight_maps(m.shape, centers, 8.0, sim.voxel_mm, m)
    T = sim.data.shape[-1]
    U = inject.time_courses(T, sim.tr_s, [0.01, 0.1], 3, rng)
    a, b = [0.0, 0.015, 0.0256], [0.5, 0.0, 0.8]                      # r2, s0, mixed (32 ms; zero crossing near TE2, so less signal per unit dR2*)
    S = inject.inject_masked(sim.data[m], W3[m], U, a, b, sim.te_ms).astype(np.float32)
    fe = frontend.run(S, sim.te_ms, cfg, m)
    c0, c1 = fe["c"][:, 0], fe["c"][:, 1]
    dec = decompose.decompose(c0, c1, fe["R"], cfg, 8, np.random.SeedSequence(SEED), n_jobs=1, mask3d=m)
    me = metrics.measure(dec, c0, c1, fe["R"], cfg, np.random.default_rng(SEED))
    maps = {k: dec[k] for k in ("m0", "m1", "m0_cf", "m1_cf", "vb_cf")}
    rows = inject.recover(c0, c1, fe["R"], None, dec["time_courses"], W3[m], U, a, b, sim.te_ms, maps,
                          comp_rows=me["components"])
    return rows


def test_recovery_finds_sources(recovered):
    for r in recovered:
        assert r["r"] > 0.9, r
        assert r["energy_share_in_footprint"] > 0.5, r


def test_recovery_physics(recovered):
    r2, s0, mx = recovered
    assert r2["eta_oracle"] < 0.1 and r2["eta_oracle_footprint"] < 0.1, r2
    assert s0["eta_oracle_footprint"] > 0.9, s0                        # whole-brain oracle eta is noise-dominated for weak sources
    assert 0.1 < mx["eta_oracle_footprint"] < 0.9, mx
    assert abs(r2["ratio_ms_oracle"]) < 4, r2
    assert abs(mx["ratio_ms_oracle"] - 32) < 6, mx
    assert abs(r2["b_ratio"] - 1) < 0.2 and abs(mx["b_ratio"] - 1) < 0.2, (r2, mx)
    assert abs(s0["a_ratio"] - 1) < 0.2, s0
    assert r2["eta_ica"] < 0.15 and s0["eta_ica_footprint"] > 0.85, (r2["eta_ica"], s0["eta_ica_footprint"])
    json.dumps(recovered)                                              # rows are JSON-ready


def test_files_end_to_end(tmp_path):
    """Run-folder layout: make_dir writes injected echoes and the truth; a front end and a decomposition of
    the injected echoes; recover_dir writes recovery.json / .md / .png; summary pools them."""
    import shutil
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_frontend import make_run_folder
    cfg = io.load_config()
    cfg = io.merge(cfg, {"simulate": {"echo_sigma_scale": cfg["frontend"]["echo_noise_ratios"]},
                         "frontend": {"mppca": {"enabled": False}}, "nuisance": {"n_surrogates": 1},
                         "decompose": {"n_restarts": 4},
                         "metrics": {"n_null_pipeline": 4, "n_boot": 4, "band_mc": 10, "n_null": 10},
                         "inject": {"radius_mm": 7.0, "min_separation_mm": 15.0, "deep_min_mm": 3.0,
                                    "types": ["r2", "s0"], "regions": ["cortical", "deep"]}})
    G = {"grid": [32, 32, 18], "n_volumes": 200, "spikes": {"n": 0}, "skull_base": {"enabled": False},
         "respiration": {"enabled": False}, "sources": []}
    sim = simulate.simulate(cfg, seed=SEED * 100 + 73, overrides=G)
    rd = make_run_folder(sim, tmp_path)
    frontend.run_subject(rd, cfg, tmp_path / "fe0", source="raw")
    idir = tmp_path / "inject" / "strong_s1"
    truth = inject.make_dir(rd, cfg, "strong", 1, idir, tmp_path / "fe0")
    assert len(truth["sources"]) == 4
    e1 = truth["echo1_injected"]
    assert (idir / "list.txt").read_text().strip() == e1
    assert os.path.islink(os.path.join(os.path.dirname(e1), "sim_e1.json"))      # sidecars linked
    again = inject.make_dir(rd, cfg, "strong", 1, idir, tmp_path / "fe0")         # same settings: skipped
    assert again["params_hash"] == truth["params_hash"]
    # the "mbme_tedana run" of the injected echoes: same pairing and motion, injected files
    ird = tmp_path / "OUT_inject" / rd.name
    shutil.copytree(rd, ird)
    files = [f["file"] for f in truth["echo_files"]]
    (ird / "echo_info.json").write_text(json.dumps({"echo_files": files}))
    res = json.loads((ird / "result.json").read_text())
    (ird / "result.json").write_text(json.dumps({**res, "input_echo1": files[0]}))
    frontend.run_subject(ird, cfg, tmp_path / "fe1", source="raw")
    decompose.run_dir(tmp_path / "fe1", cfg, tmp_path / "dec" / "k6", 6)
    out = inject.recover_dir(idir, tmp_path / "fe1", tmp_path / "dec" / "k6", mbme_run=ird)
    for f in ("recovery.json", "recovery.md", "recovery.png"):
        assert (idir / f).exists(), f
    assert out["notes"] == []
    # plumbing and physics through the files (detection itself is tested above; these spheres are tiny and
    # 200 frames short): the true time course recovers the injected amplitudes and channel assignment
    for r in out["sources"]:
        assert "te_curvature_fraction" in r and "eta_ica" in r
        if r["type"] == "r2":
            assert abs(r["b_hat_oracle_per_s"] / r["b_true_kept_per_s"] - 1) < 0.15, r
            assert r["eta_oracle_footprint"] < 0.1, r
        else:
            assert abs(r["a_hat_oracle"] / r["a_true_kept"] - 1) < 0.15, r
            assert r["eta_oracle_footprint"] > 0.85, r
    summ = inject.summary(tmp_path / "inject")
    assert summ["n_recoveries"] == 1 and (tmp_path / "inject" / "inject_summary.md").exists()


def test_place_networks_mirrored_and_separated():
    """Network layout: one network per type, each 3 mirrored cortical pairs + 1 inferior sphere, every sphere of
    every network at least network.min_separation_mm from every other; pairs on opposite sides of the midline."""
    shape = (52, 64, 48)
    idx = np.indices(shape).astype(float)
    c = np.array([25.5, 31.5, 23.5])[:, None, None, None]
    mask = (((idx - c) / np.array([22, 28, 20])[:, None, None, None]) ** 2).sum(0) <= 1
    aff = np.diag([3.281, 3.281, 3.3, 1.0])                            # RAS: axis 0 left-right
    cfg = io.load_config()
    vox = np.array([3.281, 3.281, 3.3])
    src = inject.place_networks(mask, aff, vox, cfg, np.random.default_rng(SEED))
    N = cfg["inject"]["network"]
    assert sorted(s["type"] for s in src) == sorted(inject.TYPES)
    allc = np.array([p for s in src for p in s["centers"]])
    assert len(allc) == 3 * (2 * N["cortical_pairs"] + N["inferior"])
    D = np.sqrt((((allc[:, None] - allc[None]) * vox) ** 2).sum(-1)) + np.eye(len(allc)) * 1e9
    assert D.min() >= N["min_separation_mm"] - 1e-9
    mid = np.argwhere(mask)[:, 0].mean()
    for s in src:
        cs = np.array(s["centers"])
        for a, b in zip(cs[0:6:2], cs[1:6:2]):                         # pairs: opposite hemispheres, near-mirror
            assert (a[0] - mid) * (b[0] - mid) < 0
            mirror = a.astype(float).copy(); mirror[0] = 2 * mid - mirror[0]
            assert np.sqrt((((b - mirror) * vox) ** 2).sum()) <= N["mirror_tolerance_mm"] + 1e-9
        assert s["sphere_regions"][-1] == "inferior"
    W = inject.weight_maps(shape, [s["centers"] for s in src], cfg["inject"]["radius_mm"], vox, mask)
    n = (W > 0).sum(axis=(0, 1, 2))
    assert np.all(n > 6 * 80), n                                         # ~7 spheres of ~118 voxels each
    assert np.all((W > 0).sum(-1) <= 1)                                  # networks do not overlap
