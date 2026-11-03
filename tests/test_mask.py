"""Robust mask: recovers the whole brain, ventricles and dark centre included, where a global
intensity threshold (the compute_epi_mask approach) leaves a central hole."""
import numpy as np
import pytest

from mejoint import io, mask


@pytest.fixture
def cfg():
    return io.load_config()


TE = [16.20, 31.22, 46.24, 61.26]


def phantom(shading=0.35, csf=0.3, seed=0, neck=False):
    """Echo-1 mean of a synthetic head: brain with dark ventricles, dark skull gap, dim scalp, air with
    Rician noise, and centre-dark coil shading (centre at `shading` x the edge)."""
    rng = np.random.default_rng(seed)
    shape = (52, 64, 48)
    zz, yy, xx = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    c = [(n - 1) / 2 for n in shape]
    r = lambda a: np.sqrt(((zz - c[0]) / a[0]) ** 2 + ((yy - c[1]) / a[1]) ** 2 + ((xx - c[2]) / a[2]) ** 2)
    brain = r([20, 26, 19]) <= 1
    head = r([24, 30, 23]) <= 1
    scalp = head & (r([22.5, 28.5, 21.5]) > 1)
    vent = r([4, 8, 4]) <= 1
    img = np.zeros(shape)
    img[brain] = 1000.0
    img[vent] = 1000.0 * csf                                           # T1-saturated CSF at short TR
    img[scalp] = 450.0
    rr = np.clip(r([24, 30, 23]), 0, 1)
    t2 = np.full(shape, 50.0)                                          # brain T2* (ms)
    t2[vent] = 200.0
    t2[scalp] = 30.0
    muscle = np.zeros(shape, bool)
    if neck:   # neck below a dark skull plate; joined to the brain only by the cord (foramen magnum)
        bottom = int(np.ceil(c[0] - 20))                               # lowest brain slice (axis 0 = inferior-superior)
        cord = (np.sqrt((yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= 3) & (zz < bottom + 3)
        plate = (zz >= bottom - 2) & (zz < bottom) & ~cord             # skull plate: dark
        muscle = (zz < bottom - 2) & (np.sqrt(((yy - c[1]) / 22) ** 2 + ((xx - c[2]) / 16) ** 2) <= 1) & ~cord
        img[plate] = 0.0
        img[muscle] = 950.0                                            # as bright as brain at echo 1
        t2[muscle] = 20.0
        img[cord & ~brain] = 900.0
    rr = np.clip(r([24, 30, 23]), 0, 1)
    shade = shading + (1 - shading) * rr ** 2                           # coil shading: dark centre
    sigma = 12.0
    rician = lambda x: np.sqrt((x + sigma * rng.standard_normal(shape)) ** 2 + (sigma * rng.standard_normal(shape)) ** 2)
    rim = np.zeros(shape, bool)
    if neck:   # dephased brain edge (lateral, inferior: like cerebellum next to the mastoid): normal at echo 1, fast decay
        rim = brain & (r([20, 26, 19]) > 0.85) & (zz < c[0] - 6) & (xx > c[2] + 6)
        t2[rim] = 15.0
    m1 = rician(img * shade)
    m4 = rician(img * shade * np.exp(-(TE[-1] - TE[0]) / t2))
    if neck:
        return m1, m4, brain, scalp, vent, muscle, rim
    return m1, brain, scalp, vent


def global_threshold_mask(img):
    """Stand-in for a histogram-valley global threshold: Otsu on the whole image, largest component."""
    v = np.sort(img.ravel())
    best, thr = -1, v[len(v) // 2]
    for t in np.percentile(v, np.linspace(5, 95, 91)):
        a, b = v[v <= t], v[v > t]
        between = a.size * b.size * (a.mean() - b.mean()) ** 2
        if between > best:
            best, thr = between, t
    return mask._largest_component(img > thr)


def test_mask_recovers_brain_despite_shading_and_dark_csf(cfg):
    img, brain, scalp, vent = phantom()
    out = mask.build_mask(img, (3.281, 3.281, 3.3), cfg)
    m = out["mask"]
    assert mask.dice(m, brain) > 0.95
    assert m[vent].mean() == 1.0                                       # ventricles and dark centre included
    assert (m & scalp).sum() / scalp.sum() < 0.05                      # scalp excluded
    naive = global_threshold_mask(img)
    assert naive[vent].mean() < 0.5                                    # the failure mode this mask avoids


@pytest.mark.parametrize("shading,csf", [(0.6, 0.6), (0.35, 0.3), (0.25, 0.15)])
def test_mask_is_stable_across_shading_strength(cfg, shading, csf):
    img, brain, scalp, vent = phantom(shading=shading, csf=csf, seed=1)
    m = mask.build_mask(img, (3.281, 3.281, 3.3), cfg)["mask"]
    assert mask.dice(m, brain) > 0.93 and m[vent].all()


def test_fill_holes_slicewise():
    m = np.ones((10, 10, 6), bool)
    m[4:6, 4:6, :] = False                                             # a tube open at both ends: 3D fill misses it
    from scipy import ndimage
    assert not ndimage.binary_fill_holes(m)[4:6, 4:6, :].any()        # 3D filling alone leaves it open
    assert mask.fill_holes(m, axis=2).all()                            # slice-wise filling closes it



def test_mask_excludes_neck_muscle_by_decay(cfg):
    """Neck muscle touching the brain is as bright as brain; only its fast decay (T2* ~20 ms) separates it."""
    m1, m4, brain, scalp, vent, muscle, rim = phantom(neck=True, seed=2)
    vox = (3.281, 3.281, 3.3)
    with_decay = mask.build_mask(m1, vox, cfg, mean_echo_last=m4, te_ms=TE)["mask"]
    intensity_only = mask.build_mask(m1, vox, cfg)["mask"]
    assert (intensity_only & muscle).sum() / muscle.sum() > 0.5          # the failure seen on real data
    assert (with_decay & muscle).sum() / muscle.sum() < 0.12               # thin band around the cord only
    assert with_decay[vent].all() and mask.dice(with_decay & ~muscle, brain) > 0.95
    assert rim.sum() > 200 and with_decay[rim].mean() > 0.9            # dephased brain edge kept (cerebellum, OFC)
    # limit: regrowth adds a band of up to regrow_vox voxels where short-T2* tissue touches the brain


def test_mask_qc_all_runs(cfg, tmp_path):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_frontend import make_run_folder
    from mejoint import simulate
    out_pipe = tmp_path / "OUT"
    for i, run in enumerate(["sub-90001_acq-20260101000000_ser-5", "sub-51497_acq-20260603154805_ser-4"]):
        sim = simulate.simulate(cfg, seed=i, overrides={"grid": [30, 30, 18], "n_volumes": 20})
        rd = make_run_folder(sim, tmp_path / f"r{i}", run_id=run)
        rd.rename(out_pipe / run) if out_pipe.exists() else (out_pipe.mkdir(), rd.rename(out_pipe / run))
    rows = mask.qc_all(out_pipe, cfg, tmp_path / "qc", log=lambda *_: None)
    assert [r["run"] for r in rows] == ["sub-90001_acq-20260101000000_ser-5"]          # excluded subject skipped
    assert rows[0]["dice_vs_pipeline"] > 0.95 and (tmp_path / "qc" / "mask_qc.tsv").exists()
