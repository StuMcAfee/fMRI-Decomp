"""Design A (mejoint.disconnect): damaged fraction on the model grid, disconnection scores, Schaefer aggregation."""
import nibabel as nib
import numpy as np

from mejoint import disconnect


def test_damaged_fraction_from_a_finer_map():
    """Model voxel (1,1,1) spans 1.5-4.5 mm on each axis; its 3 x 3 x 3 sample points sit at 2, 3, 4 mm, i.e. at
    1 mm grid indices 3, 4, 5 (index i is centred at i - 1 mm). A lesion over x indices 3-4 (two of the three
    sample layers) gives 2/3; the other model voxel is untouched."""
    aff3 = np.diag([3.0, 3.0, 3.0, 1.0]); ijk = np.array([[1, 1, 1], [3, 1, 1]])
    aff1 = np.eye(4); aff1[:3, 3] = -1.0
    L = np.zeros((15, 9, 9), np.float32)
    L[3:5, 3:6, 3:6] = 1.0
    r = disconnect.damaged_fraction(nib.Nifti1Image(L, aff1), ijk, aff3)
    assert r["n_outside_map"] == 0
    assert np.isclose(r["fraction"][0], 2 / 3) and r["fraction"][1] == 0.0


def test_dose_scale_and_clipping():
    aff = np.diag([3.0, 3.0, 3.0, 1.0]); ijk = np.array([[0, 0, 0], [1, 0, 0]])
    dose = nib.Nifti1Image(np.array([[[54.0]], [[27.0]]], np.float32).repeat(1, 1), aff)
    r = disconnect.damaged_fraction(dose, ijk, aff, sub=1, scale=54.0)
    assert np.allclose(r["fraction"], [1.0, 0.5])


def test_disconnection_positive_weights():
    W = np.array([[1.0, -2.0], [3.0, 1.0], [-1.0, 1.0]])             # 3 cerebellar voxels x 2 cortical parcels
    frac = np.array([1.0, 0.0, 0.5])
    d = disconnect.disconnection(W, frac, "positive")
    assert np.isclose(d[0], 1.0 / 4.0) and np.isclose(d[1], 0.5 / 2.0)
    da = disconnect.disconnection(W, frac, "absolute")
    assert np.isclose(da[0], (1 + 0.5) / 5.0)


def test_to_schaefer_vertex_weighted():
    d_ico = np.array([0.2, 0.6, 1.0])                                 # ids 1, 2 (left), 3 (right, n_left = 2)
    ico_l = np.array([1, 1, 1, 2, 0]); sch_l = np.array([5, 5, 5, 5, 5])
    ico_r = np.array([1, 1]); sch_r = np.array([9, 0])
    s = disconnect.to_schaefer(d_ico, ico_l, ico_r, sch_l, sch_r, n_left=2)
    assert np.isclose(s[5], (3 * 0.2 + 0.6) / 4) and np.isclose(s[9], 1.0)
