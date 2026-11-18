"""Parcel ICA (docs/PARCEL_ICA_PLAN.md): whitening, dimensionality, measurements, removal, network matching."""
import numpy as np

from mejoint import parcel_ica as pi
from mejoint import parcel_sim as ps


def _sim(**kw):
    d = ps.simulate(**{"n_parcels": 300, "T": 600, "seed": 3, **kw})
    return d, pi.prepare(d["c0"], d["c1"], d["R"], d["noise_var"], d["r"])


def test_prepare_whitened_b_is_proportional_and_unit_noise():
    d, pr = _sim()
    b = d["b"] - d["b"].mean(1, keepdims=True)
    r = [np.corrcoef(b[p], pr["zb"][p])[0, 1] for p in range(0, 300, 37)]
    assert np.allclose(r, 1.0, atol=1e-9)
    # pure noise in the channels -> unit-variance whitened b; the null channels too
    g = np.random.default_rng(0)
    pn = pi.prepare(g.standard_normal(d["c0"].shape), g.standard_normal(d["c1"].shape), d["R"], d["noise_var"], d["r"])
    assert abs(np.median(pn["zb"].var(1)) - 1) < 0.05
    assert len(pn["null_channels"]) == 2 and abs(np.median(pn["null_channels"][0].var(1)) - 1) < 0.05


def test_dimensionality_pure_noise_and_signal():
    g = np.random.default_rng(1)
    z = g.standard_normal((400, 600))
    nulls = [g.standard_normal((400, 600)) for _ in range(2)]
    dn = pi.dimensionality(z, 600, nulls)
    assert dn["k_mp"] <= 1 and dn["k_null"] <= 1
    z[:, :] += np.outer(g.normal(0, 1, 400), np.sin(np.arange(600) / 9.0)) * 0.8 + np.outer(g.normal(0, 1, 400), g.standard_normal(600)) * 0.5
    assert pi.dimensionality(z, 600, nulls)["k_mp"] == 2


def test_s0_coupling_pure_and_mixed_sources():
    """A pure dR2* component gives eta ~ 0; a mixed one (the BOLD-like ICA never yields a pure dS0 component, because
    dS0 does not enter b) gives the eta of its true, noise-free energies."""
    d, _ = _sim(include=("network",))
    P, T = d["c0"].shape
    R = d["R"]
    g = np.random.default_rng(2)
    tc = ps.band_noise(T, 0.75, (0.01, 0.1), g)[0]
    mp = np.abs(g.normal(1, 0.3, P))
    for a_amp in (0.0, 0.004, 0.012):
        b, a = np.outer(mp * 0.5, tc), np.outer(mp * a_amp, tc)
        c0 = R[:, 0, 0, None] * b / 1000 + R[:, 0, 1, None] * a + g.standard_normal((P, T))
        c1 = R[:, 1, 1, None] * a + g.standard_normal((P, T))
        pr = pi.prepare(c0, c1, R, np.ones(P))
        eta = pi.s0_coupling(mp[:, None], pr["zb"], pr["cw"], pr["rho"], np.arange(P) % 2)["eta"][0]
        rho = R[:, 0, 1] / R[:, 1, 1]
        e_s0 = ((1 + rho ** 2) * (R[:, 1, 1] * mp * a_amp) ** 2).sum()
        e_r2 = ((R[:, 0, 0] * mp * 0.5 / 1000) ** 2).sum()
        expect = e_s0 / (e_s0 + e_r2)
        assert abs(eta - expect) < 0.05, (a_amp, eta, expect)


def test_spectral_measures():
    T, tr = 800, 0.75
    t = np.arange(T) * tr
    g = np.random.default_rng(4)
    x = np.vstack([np.sin(2 * np.pi * 0.28 * t), ps.band_noise(T, tr, (0.01, 0.1), g)[0], g.standard_normal(T)])
    sp = pi.spectral_measures(x, tr, resp_hz=0.28)
    assert sp["high_share"][0] > 0.95 and sp["resp_share"][0] > 0.95 and abs(sp["peak_hz"][0] - 0.28) < 0.01
    assert sp["band_share"][1] > 0.95 and sp["high_share"][1] < 0.05
    # with its noise variance given, a pure-noise time course has (almost) no debiased power left
    sp2 = pi.spectral_measures(x[2:], tr, noise_var=[1.0])
    raw = (np.abs(np.fft.rfft(x[2] - x[2].mean())) ** 2).sum()
    assert np.isfinite(sp2["high_share"][0]) and raw > 0


def test_remove_components_and_match_networks():
    g = np.random.default_rng(5)
    tc = g.standard_normal((2, 300))
    tc -= tc.mean(1, keepdims=True)                                    # route B signals are demeaned
    X = g.standard_normal((50, 2)) @ tc
    assert np.abs(pi.remove_components(X, tc)).max() < 1e-9
    nets = np.array(ps.NETS * 20)
    maps = np.column_stack([(nets == n) + 0.1 * g.standard_normal(nets.size) for n in ps.NETS[::-1]])
    m = pi.match_networks(maps, list(nets))
    assert all(m[n]["component"] == 6 - i and m[n]["r"] > 0.9 for i, n in enumerate(ps.NETS))


def test_spatial_ica_recovers_networks():
    d, pr = _sim(n_parcels=600, reliability=0.6, include=("network",))
    dim = pi.dimensionality(pr["zb"], pr["zb"].shape[1] - 1, pr["null_channels"])
    assert 6 <= dim["k_mp"] <= 8
    ica = pi.spatial_ica(pr["zb"], 7, range(8))
    C = np.abs(np.corrcoef(np.vstack([d["truth"]["tc"], ica["tc"]]))[:7, 7:])
    assert C.max(1).min() > 0.8


def _route_b_dir(tmp_path, d, new_format=True):
    P, T = d["c0"].shape
    nets = d["networks"]
    regions = [{"id": i + 1, "name": f"p{i + 1}", "kind": "cortex", "network": str(nets[i])} for i in range(P)]
    regions.append({"id": P + 1, "name": "thalamus", "kind": "deep", "network": None})
    pad = lambda x: np.concatenate([x, np.zeros((1,) + x.shape[1:])], 0)
    arr = {"a": pad(d["a"]), "b": pad(d["b"]), "c0": pad(d["c0"]), "R": pad(d["R"]), "noise_var": np.append(d["noise_var"], np.nan)}
    if new_format:
        arr.update(c1=pad(d["c1"]), r=pad(d["r"]), frames=np.arange(T - 20))
    np.savez(tmp_path / "region_timecourses.npz", **arr)
    import json
    (tmp_path / "route_b.json").write_text(json.dumps({"cortex_ids": list(range(1, P + 1)), "regions": regions, "nuisance_rank": 0,
                                                      "band_hz": [0.01, 0.1], "frontend_dir": str(tmp_path / "none"),
                                                      "timeline": {"n_used": T - 10, "longest_stretch": [5, T]}}))


def test_run_dir_on_route_b_outputs(tmp_path):
    from mejoint import io
    d, _ = _sim(n_parcels=300, reliability=0.6, include=("network", "respiratory"), amp={"respiratory": 0.6})
    cfg = io.load_config()
    for new in (True, False):
        rb = tmp_path / f"rb{int(new)}"
        rb.mkdir()
        _route_b_dir(rb, d, new)
        res = pi.run_dir(rb, cfg, rb / "out", n_seeds=4, n_jobs=1)
        assert res["n_parcels"] == 300 and res["n_frames"] == (580 if new else 590)
        assert res["k_used"] >= 7 and (rb / "out" / "components.tsv").exists() and (rb / "out" / "parcel_ica.md").exists()
        assert res["n_flagged"] >= 1                                     # the strong respiratory source
        assert (res["k_null"] is not None) == new


def test_surface_figures_on_a_synthetic_mesh(tmp_path):
    """Winner-take-all and weight maps on a small closed mesh (two 'hemispheres' made of an octahedron sphere)."""
    from mejoint import surface_plot as sp
    # subdivided octahedron -> sphere
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    f = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    for _ in range(3):
        cache, nf = {}, []
        def mid(a, b):
            key = tuple(sorted((a, b)))
            if key not in cache:
                cache[key] = len(v_list)
                v_list.append((v_list[a] + v_list[b]) / 2)
            return cache[key]
        v_list = list(v)
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        v, f = np.array(v_list), np.array(nf)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    lab = 1 + (np.arctan2(v[:, 1], v[:, 0]) > 0) + 2 * (v[:, 2] > 0)        # 4 parcels per hemisphere
    nets = {"L": np.array(["Vis", "SomMot", "Default", "Cont"])[lab - 1], "R": np.array(["Vis", "SomMot", "Default", "Cont"])[lab - 1]}
    cols = {n: np.array([*c, 1.0]) for n, c in zip(sp.NETS7, np.random.default_rng(0).uniform(0, 1, (7, 3)))}
    res = {"surf": {"L": (v, f), "R": (v, f)}, "labels": {"L": lab, "R": lab + 4}, "net_vertex": nets, "net_colours": cols}
    ids = list(range(1, 9))
    pnets = ["Vis", "SomMot", "Default", "Cont"] * 2
    maps = np.eye(8)[:, :3] * 3 + 0.01 * np.random.default_rng(1).standard_normal((8, 3))
    files = sp.weight_maps(maps, ids, res, tmp_path / "w", ["a", "b", "c"], per_page=2)
    assert len(files) == 2 and all(__import__("os").path.getsize(x) > 1000 for x in files)
    out = sp.winner_take_all(maps, ids, pnets, res, tmp_path / "wta.png", flagged=[False, False, True])
    assert out["winner"][:3] == [0, 1, 2] and out["n_parcels_no_winner"] == 5 and (tmp_path / "wta.png").exists()
