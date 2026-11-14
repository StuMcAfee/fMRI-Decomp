"""Phase 4 gate (CLAUDE.md section 8): joint ICA (option 1) and per-component metrics.

- planted sources recovered with |r| > 0.9: q for dR2*-only, s for dS0-only, both for mixed (maps
  compared over each source's footprint plus a 2-voxel margin: whole-brain correlation of a 14-voxel
  focal source is dominated by noise in the ~10^4 empty voxels);
- eta in the correct band (matched-SNR bands + the cross-talk margin);
- pure noise: no component above the energy null;
- dR2*-only data: no spurious mixed or S0 components;
- tedana equivalence: z-scored c0 (linear form, equal weights) vs z-scored OC, same k and seed, |r| > 0.99;
- stability across 30 restarts (Hungarian on |corr|) reported.
"""
import os

import numpy as np
import pytest
from scipy import ndimage

from mejoint import decompose, frontend, io, metrics, simulate

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))
G = {"grid": [32, 32, 18], "n_volumes": 400, "spikes": {"n": 0}, "skull_base": {"enabled": False},
     "respiration": {"enabled": False}}
SOURCES = [{"name": "r2a", "kind": "r2", "extent": "focal", "q": 0.5, "radius_vox": 1.5},
           {"name": "r2b", "kind": "r2", "extent": "focal", "q": 0.5, "radius_vox": 1.5},
           {"name": "s0a", "kind": "s0", "extent": "focal", "s": 0.015, "radius_vox": 1.5},
           {"name": "s0b", "kind": "s0", "extent": "focal", "s": 0.015, "radius_vox": 1.5},
           {"name": "mix_same", "kind": "mixed", "extent": "focal", "q": 0.5, "s": 0.015, "radius_vox": 1.5},
           {"name": "mix_opp", "kind": "mixed", "extent": "focal", "q": -0.5, "s": 0.015, "radius_vox": 1.5}]


def base_cfg():
    c = io.load_config()
    return io.merge(c, {"simulate": {"echo_sigma_scale": c["frontend"]["echo_noise_ratios"]},
                        "metrics": {"n_boot": 20, "band_mc": 50}})


def run_all(cfg, sources, seed, k, n_restarts=30):
    sim = simulate.simulate(cfg, seed=seed, overrides={**G, "sources": sources})
    fe = frontend.run(sim.data[sim.mask], sim.te_ms, cfg, sim.mask)
    c = io.merge(cfg, {"decompose": {"n_restarts": n_restarts}})
    dec = decompose.decompose(fe["c"][:, 0], fe["c"][:, 1], fe["R"], c, k, np.random.SeedSequence(seed), n_jobs=2)
    me = metrics.measure(dec, fe["c"][:, 0], fe["c"][:, 1], fe["R"], c, np.random.default_rng(seed))
    return sim, fe, dec, me


@pytest.fixture(scope="module")
def planted():
    cfg = base_cfg()
    return {k: run_all(cfg, SOURCES, SEED * 100 + 60, k) for k in (6, 10)}


def match(sim, dec, src):
    """Component whose time course best matches the source; map |r| over the footprint + 2 voxels."""
    m = sim.mask
    ct = [abs(np.corrcoef(src["timecourse"], dec["time_courses"][:, j])[0, 1]) for j in range(dec["k"])]
    j = int(np.argmax(ct))
    ref = src["q_map"] if np.abs(src["q_map"]).max() > 0 else src["s_map"]
    sup = ndimage.binary_dilation(np.abs(ref) > 0.05 * np.abs(ref).max(), iterations=2)[m]

    def r(est, true):
        ok = np.isfinite(est) & sup
        return abs(np.corrcoef(true[ok], est[ok])[0, 1])

    rq = r(dec["q_per_s"][:, j], src["q_map"][m]) if np.abs(src["q_map"]).max() > 0 else None
    rs = r(dec["s"][:, j], src["s_map"][m]) if np.abs(src["s_map"]).max() > 0 else None
    return j, ct[j], rq, rs


@pytest.mark.parametrize("k", [6, 10])
def test_planted_sources_recovered(planted, record_property, k):
    sim, fe, dec, me = planted[k]
    record_property("simulation", sim.describe())
    out = {}
    for src in sim.sources:
        j, rt, rq, rs = match(sim, dec, src)
        out[src["name"]] = {"tc": round(rt, 3), "q": rq and round(rq, 3), "s": rs and round(rs, 3)}
        assert rt > 0.9, (src["name"], rt)
        if src["kind"] in ("r2", "mixed"):
            assert rq > 0.9, (src["name"], rq)
        if src["kind"] in ("s0", "mixed"):
            assert rs > 0.9, (src["name"], rs)
    record_property("recovery", out)
    assert len({match(sim, dec, s)[0] for s in sim.sources}) == len(sim.sources)   # one component each


@pytest.mark.parametrize("k", [6, 10])
def test_eta_falls_in_the_correct_band(planted, record_property, k):
    sim, fe, dec, me = planted[k]
    want = {"r2": "R2*-dominant", "s0": "S0-dominant", "mixed": "mixed"}
    got = {}
    for src in sim.sources:
        j = match(sim, dec, src)[0]
        row = me["components"][j]
        got[src["name"]] = (row["label"], row["eta"])
        assert row["above_null"], src["name"]
        assert row["label"] == want[src["kind"]], (src["name"], row)
    record_property("labels", got)


def test_extra_components_at_k10_stay_near_the_null(planted):
    sim, fe, dec, me = planted[10]
    used = {match(sim, dec, s)[0] for s in sim.sources}
    extra = [r for j, r in enumerate(me["components"]) if j not in used]
    assert sum(r["above_null"] for r in extra) <= 1


@pytest.mark.parametrize("seed", [1, 2])
def test_pure_noise_gives_no_components_above_the_null(record_property, seed):
    sim, fe, dec, me = run_all(base_cfg(), [], SEED * 100 + 70 + seed, 10, n_restarts=10)
    record_property("simulation", sim.describe())
    ratios = [round(r["energy_single"] / me["null"]["threshold"], 3) for r in me["components"]]
    record_property("energy_over_null", ratios)
    assert sum(r["above_null"] for r in me["components"]) == 0


def test_r2_only_data_gives_no_spurious_mixed_components(record_property):
    src = [{"name": f"r2{i}", "kind": "r2", "extent": "focal", "q": 0.5, "radius_vox": 1.5} for i in range(5)] + \
          [{"name": "r2d", "kind": "r2", "extent": "diffuse", "q": 0.25}]
    sim, fe, dec, me = run_all(base_cfg(), src, SEED * 100 + 80, 10, n_restarts=10)
    record_property("simulation", sim.describe())
    sig = [(r["label"], r["eta"]) for r in me["components"] if r["above_null"]]
    record_property("significant", sig)
    assert len(sig) >= 5
    assert all(lab == "R2*-dominant" for lab, _ in sig)


def test_stability_across_30_restarts_is_reported(planted, record_property):
    sim, fe, dec, me = planted[6]
    assert dec["n_restarts"] == 30 and dec["n_converged"] >= 28
    record_property("stability_median", np.round(dec["stability_median"], 4).tolist())
    record_property("iq", np.round(dec["iq"], 4).tolist())
    assert (dec["stability_median"] > 0.95).all()


def test_tedana_equivalence(record_property):
    """CLAUDE.md Phase 4: OC with the same T2* map, per-voxel z-scoring, fixed k and seed, vs c0 (linear
    form, equal weights and equal noise). k = the number of planted signals: components beyond the
    signal subspace are Gaussian noise directions, which ICA cannot identify (k = 8 with 6 signals
    matched the 6 signals at >= 0.997 and the noise directions arbitrarily)."""
    cfg = io.merge(io.load_config(), {"frontend": {"weighting": "equal", "signal_form": "linear"},
                                       "simulate": {"echo_sigma_scale": [1.0, 1.0, 1.0, 1.0], "snr0": 400.0}})
    sim = simulate.simulate(cfg, seed=SEED * 100 + 90, overrides=G)
    record_property("simulation", sim.describe())
    S = sim.data[sim.mask]
    fe = frontend.run(S, sim.te_ms, cfg, sim.mask)
    oc = decompose.tedana_style_oc(S, sim.te_ms, fe["t2s_ms"])
    c0 = fe["c"][:, 0].astype(np.float64)
    ok = np.isfinite(c0).all(1) & fe["use"].all(1)
    a, b = decompose.zscore_rows(oc[ok]), decompose.zscore_rows(c0[ok])
    k = len(sim.sources)
    for seed in (123, 7):
        A = decompose.single_channel_ica(a, k, [seed])
        B = decompose.single_channel_ica(b, k, [seed])
        _, r = decompose.hungarian_abs_corr(A, B)
        record_property(f"matched_abs_r_seed{seed}", np.round(r, 5).tolist())
        assert (r > 0.99).all()


def test_backsubstitution_is_exact():
    rng = np.random.default_rng(SEED)
    V, k = 50, 3
    R = np.zeros((V, 2, 2))
    R[:, 0, 0] = rng.uniform(5, 50, V)
    R[:, 0, 1] = -rng.uniform(1, 30, V)
    R[:, 1, 1] = rng.uniform(2, 20, V)
    s, q = rng.normal(0, 0.01, (V, k)), rng.normal(0, 1.0, (V, k))          # q in s^-1
    m1 = R[:, 1, 1][:, None] * s
    m0 = R[:, 0, 0][:, None] * q / 1000 + R[:, 0, 1][:, None] * s
    s2, q2 = decompose.backsub(m0, m1, R)
    assert np.allclose(s2, s) and np.allclose(q2, q)


def test_eta_debiasing_on_pure_noise_maps():
    """Noise-only maps with known beta variance: debiased energies average ~0."""
    rng = np.random.default_rng(SEED + 1)
    V = 5000
    R = np.zeros((V, 2, 2))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 1] = rng.uniform(5, 50, V), -rng.uniform(1, 30, V), rng.uniform(2, 20, V)
    vb = 0.01
    e = [metrics.energies(rng.normal(0, np.sqrt(vb), (V, 1)), rng.normal(0, np.sqrt(vb), (V, 1)), R, vb) for _ in range(50)]
    tot = np.array([x["total"][0] for x in e])
    raw_scale = vb * V * 2
    assert abs(tot.mean()) < 0.05 * raw_scale


def test_run_sweep_end_to_end(tmp_path):
    """Real-run layout: front end (nuisance projection), then the k sweep writes maps, tables and a figure."""
    import subprocess
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_frontend import make_run_folder
    cfg = io.merge(base_cfg(), {"frontend": {"mppca": {"enabled": False}}, "nuisance": {"n_surrogates": 1},
                                "decompose": {"n_restarts": 5}, "metrics": {"n_null_pipeline": 5, "n_boot": 5, "band_mc": 20}})
    sim = simulate.simulate(cfg, seed=SEED * 100 + 95, overrides={**G, "grid": [24, 24, 14], "n_volumes": 200,
                                                                   "sources": SOURCES[:3]})
    rd = make_run_folder(sim, tmp_path)
    frontend.run_subject(rd, cfg, tmp_path / "fe", source="raw")
    rows = decompose.run_sweep(tmp_path / "fe", cfg, tmp_path / "p4", ks=[3, 5])
    assert [r["k"] for r in rows] == [3, 5]
    for k in (3, 5):
        d = tmp_path / "p4" / f"k{k}"
        for f in ("s.nii.gz", "q_per_s.nii.gz", "m0.nii.gz", "m1.nii.gz", "time_courses.tsv", "components.tsv",
                  "components.json", "decompose.md", "done.json"):
            assert (d / f).exists(), f
    assert (tmp_path / "p4" / "sweep.md").exists()
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "plot_components.py")
    res = subprocess.run([sys.executable, script, str(tmp_path / "p4" / "k3"), str(tmp_path / "fe")], capture_output=True,
                         text=True, env={**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "src")})
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "p4" / "k3" / "components.png").exists()


def test_debiased_norms_recover_the_planted_norms_and_sit_inside_their_intervals(planted, record_property):
    """||q|| for dR2* sources and ||s|| for dS0 sources (debiased, cross-fitted) match the planted map
    norms, and each estimate lies inside its bootstrap interval (the undebiased version did not on
    real data)."""
    sim, fe, dec, me = planted[6]
    m = sim.mask
    out = {}
    for src in sim.sources:
        j = match(sim, dec, src)[0]
        row = me["components"][j]
        if src["kind"] == "r2":
            true, est, ci = np.linalg.norm(src["q_map"][m]), row["norm_q_per_s"], row["norm_q_ci"]
        elif src["kind"] == "s0":
            true, est, ci = np.linalg.norm(src["s_map"][m]), row["norm_s"], row["norm_s_ci"]
        else:
            continue
        out[src["name"]] = (round(float(true), 4), round(est, 4), [round(x, 4) for x in ci])
        assert est == pytest.approx(true, rel=0.15), src["name"]
        assert ci[0] <= est <= ci[1], src["name"]
    record_property("norms_true_est_ci", out)
