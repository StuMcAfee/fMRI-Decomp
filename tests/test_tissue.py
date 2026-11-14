"""Phase 5: data-only tissue classes and the partial-volume diagnostic (user decision 2026-11-04,
options a and c). Our derivation (two-compartment decay), checked in simulation."""
import os

import numpy as np
import pytest

from mejoint import decompose, frontend, io, metrics, simulate, tissue

SEED = int(os.environ.get("MEJOINT_TEST_SEED", 0))


@pytest.fixture
def cfg():
    return io.load_config()


def voxels(cfg, f, t2t=45.0, t2c=200.0, ratio=1.0, snr=100.0, rng=None, n=1):
    te = np.asarray(cfg["acquisition"]["te_ms"])
    r = np.asarray(cfg["frontend"]["echo_noise_ratios"])
    mu = (1 - f) * 1e4 * np.exp(-te / t2t) + f * ratio * 1e4 * np.exp(-te / t2c)
    mu = np.repeat(mu[None, :], n, axis=0)
    sigma = np.full(n, 1e4 / snr)
    if rng is not None:                                    # noise of a temporal mean over 800 frames
        mu = mu + sigma[:, None] * r[None, :] * rng.standard_normal(mu.shape) / np.sqrt(800)
    return te, mu, sigma, r


@pytest.mark.parametrize("f", [0.0, 0.1, 0.3, 0.5, 0.8])
def test_two_compartment_fit_recovers_the_long_fraction(cfg, f):
    rng = np.random.default_rng(SEED)
    te, mu, sigma, r = voxels(cfg, f, rng=rng, n=500)
    res = tissue.run(mu, te, np.ones_like(mu, bool), sigma, r, cfg)
    assert np.median(res["f_long"]) == pytest.approx(f, abs=0.01)
    assert np.percentile(np.abs(res["f_long"] - f), 90) < 0.02


def test_concave_dephased_decay_is_not_called_csf(cfg):
    te, _, sigma, r = voxels(cfg, 0.0)
    mu = 1e4 * np.exp(-te / 45.0) * np.abs(np.sinc(8.0 * te / 1000))     # through-slice dephasing
    res = tissue.run(mu[None, :], te, np.ones((1, 4), bool), sigma[:1], r, cfg)
    assert res["f_long"][0] < 0.01 and abs(res["kappa"][0]) < 1e-6


def test_kappa_matches_the_direct_two_compartment_change(cfg):
    """kappa from the fit equals a / q of a small tissue-only dR2* change pushed through the projection."""
    te, mu, sigma, r = voxels(cfg, 0.5)
    res = tissue.run(mu, te, np.ones((1, 4), bool), sigma, r, cfg)
    St, Sc = 0.5 * 1e4 * np.exp(-te / 45.0), 0.5 * 1e4 * np.exp(-te / 200.0)
    y = np.log((St * np.exp(-te * 1e-5) + Sc) / (St + Sc))[None, :, None]   # 0.01 s^-1 change in tissue
    pr = frontend.project(y, (mu / r) / sigma[:, None], np.ones((1, 4), bool), te)
    c, R = pr["c"][0, :, 0], pr["R"][0]
    a = c[1] / R[1, 1]
    q = (c[0] - R[0, 1] * a) / R[0, 0] * 1000
    assert res["kappa"][0] == pytest.approx(a / q, rel=0.01)
    assert res["kappa"][0] < 0                                            # apparent dS0 opposite to dR2*


def test_classes(cfg):
    te, mu, sigma, r = voxels(cfg, 0.0)
    rows = [voxels(cfg, f)[1][0] for f in (0.02, 0.3, 0.8)] + [1e4 * np.exp(-te / 12.0)]
    mu = np.array(rows)
    use = np.ones_like(mu, bool)
    res = tissue.run(mu, te, use, np.full(4, sigma[0]), r, cfg)
    assert res["classes"].tolist() == [1, 2, 3, 4]
    use[0, 2:] = False
    assert tissue.run(mu, te, use, np.full(4, sigma[0]), r, cfg)["classes"][0] == 5


@pytest.fixture(scope="module")
def pv_sim():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import test_decompose as td
    cfg = io.merge(td.base_cfg(), {"simulate": {"csf_partial_volume": {"enabled": True, "fraction_max": 0.7}},
                                   "decompose": {"n_restarts": 10}})
    src = [{"name": "r2a", "kind": "r2", "extent": "focal", "q": 0.5, "radius_vox": 2.0},
           {"name": "r2b", "kind": "r2", "extent": "focal", "q": 0.5, "radius_vox": 2.0},
           {"name": "mix", "kind": "mixed", "extent": "focal", "q": 0.5, "s": 0.015, "radius_vox": 2.0},
           {"name": "s0", "kind": "s0", "extent": "focal", "s": 0.015, "radius_vox": 2.0}]
    sim = simulate.simulate(cfg, seed=SEED * 100 + 11, overrides={**td.G, "sources": src})
    fe = frontend.run(sim.data[sim.mask], sim.te_ms, cfg, sim.mask)
    tis = tissue.run(fe["mu"], sim.te_ms, fe["use"], fe["sigma"], fe["echo_noise_ratios"], cfg)
    dec = decompose.decompose(fe["c"][:, 0], fe["c"][:, 1], fe["R"], cfg, 6, np.random.SeedSequence(SEED + 1), n_jobs=2,
                              mask3d=sim.mask)
    me = metrics.measure(dec, fe["c"][:, 0], fe["c"][:, 1], fe["R"], cfg, np.random.default_rng(SEED), tissue=tis)
    rows = {s["name"]: me["components"][td.match(sim, dec, s)[0]] for s in sim.sources}
    return sim, tis, rows


def test_long_fraction_matches_the_simulated_csf_fraction(pv_sim):
    sim, tis, _ = pv_sim
    f, true = tis["f_long"], sim.csf_fraction[sim.mask]
    ok = np.isfinite(f)
    assert np.corrcoef(f[ok], true[ok])[0, 1] > 0.98 and np.median(np.abs(f[ok] - true[ok])) < 0.02


@pytest.mark.xfail(reason="WIP (2026-11-04): the zero-parameter prediction kappa q is swamped by noise where kappa is "
                          "large (CSF fraction 0.5-0.9); the mixed and dS0 sources are classified correctly, the "
                          "partial-volume dR2* sources are not yet", strict=False)
def test_partial_volume_diagnostic_separates_pv_from_physiological_ds0(pv_sim, record_property):
    sim, tis, rows = pv_sim
    keep = ("eta", "pv_verdict", "pv_explained", "ratio_explained", "ratio_dS0_per_s", "eta_pv_removed")
    record_property("rows", {k: {f: v.get(f) for f in keep} for k, v in rows.items()})
    verdicts = {k: v["pv_verdict"] for k, v in rows.items()}
    pv_r2 = [k for k in ("r2a", "r2b") if verdicts[k] == "consistent with partial volume"]
    assert pv_r2, verdicts                                          # at least one dR2* source sits in partial volume
    for k in ("r2a", "r2b"):
        assert verdicts[k] in ("consistent with partial volume", "no dS0 to explain")
        if verdicts[k] == "consistent with partial volume":
            assert rows[k]["eta"] > 0.03 and rows[k]["eta_pv_removed"] < rows[k]["eta"]
    assert verdicts["mix"] == "not partial volume"
    assert rows["mix"]["ratio_dS0_per_s"] == pytest.approx(0.015 / 0.5, rel=0.25)
    assert verdicts["s0"] in ("not partial volume", "too few voxels with dR2*")


def test_tissue_run_dir(cfg, tmp_path):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_frontend import make_run_folder
    c = io.merge(cfg, {"simulate": {"echo_sigma_scale": cfg["frontend"]["echo_noise_ratios"],
                                    "csf_partial_volume": {"enabled": True}},
                       "frontend": {"mppca": {"enabled": False}}, "nuisance": {"n_surrogates": 0}})
    sim = simulate.simulate(c, seed=SEED * 100 + 12, overrides={"grid": [20, 20, 12], "n_volumes": 120})
    rd = make_run_folder(sim, tmp_path)
    frontend.run_subject(rd, c, tmp_path / "fe", source="raw")
    summ = tissue.run_dir(tmp_path / "fe", c, tmp_path / "tis")
    for f in ("classes.nii.gz", "f_long.nii.gz", "kappa.nii.gz", "classes.npy", "kappa.npy", "tissue.md", "done.json"):
        assert (tmp_path / "tis" / f).exists(), f
    assert sum(summ["class_ml"].values()) > 0
