"""Torquato's descriptors (morphogen.analysis.correlation): the lineal-path, two-point
cluster and pore-size functions on images with exact counts and against the closed forms
of the Boolean model of spheres; the radial distribution function and the structure
factor of point patterns (Poisson points, hard spheres, a lattice)."""

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen import analysis as an
from morphogen.image import VoxelImage

R, P = 0.05, 0.3


@pytest.fixture(scope="module")
def boolean():
    """A periodic Boolean model of spheres on 128^3 voxels, and its intensity."""
    model = mg.models.BooleanSpheres(radius=R, volume_fraction=P)
    img = model.generate(mg.Domain.cube(1.0, periodic=True), seed=0).voxelize((128, 128, 128))
    return img, -math.log(1 - P) / (4 / 3 * math.pi * R**3)


def test_lineal_path():
    """Exact counts of segments on runs (periodic and not), ``L(0)`` the fraction, and
    the matrix of the Boolean model: ``(1 - p) exp(-θ π R^2 z)``."""
    data = np.zeros((8, 2, 2), dtype=np.uint8)
    data[1:4, 0, 0] = 1  # a run of three voxels
    data[:, 1, 1] = 1  # a line full of the phase
    lp = an.lineal_path(VoxelImage(data, periodic=True), 1, max_lag=4)
    # four lines of eight positions: the run gives 3, 2, 1 segments, the full line 8
    assert np.allclose(lp.values["x"] * 32, [11, 10, 9, 8, 8])
    lp = an.lineal_path(VoxelImage(data, periodic=False), 1, max_lag=4)
    assert np.allclose(lp.values["x"] * 4 * (8 - np.arange(5)), [11, 9, 7, 5, 4])
    assert lp.values["x"][0] == lp.volume_fraction == data.mean()
    assert np.allclose(lp.values["y"], [data.mean(), 0])  # no run along y longer than one


def test_lineal_path_and_pore_sizes_of_the_boolean_model(boolean):
    img, theta = boolean
    z, lv = an.lineal_path(img, 0, max_lag=30).mean()
    np.testing.assert_allclose(lv, (1 - P) * np.exp(-theta * math.pi * R**2 * z), atol=0.01)
    # the pores: exp(-θ 4/3 π ((R + δ)^3 - R^3)), to the first order in the voxel size
    # (spheres of 6.4 voxels: about 4 % low on the mean)
    ps = an.pore_size_function(img, 0, bins=30)
    expected = np.exp(-theta * 4 / 3 * math.pi * ((R + ps.delta) ** 3 - R**3))
    far = ps.delta >= 2 / 128
    np.testing.assert_allclose(ps.exceedance[far], expected[far], atol=0.065)
    assert ps.exceedance[0] == 1.0 and np.all(np.diff(ps.exceedance) <= 0)
    mean = np.sum(0.5 * (expected[1:] + expected[:-1]) * np.diff(ps.delta))
    assert ps.mean == pytest.approx(mean, rel=0.06)
    assert np.sum(ps.density * np.diff(ps.delta)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no voxel"):
        an.pore_size_function(VoxelImage(np.ones((4, 4, 4), dtype=np.uint8)), 0)


def test_two_point_cluster(boolean):
    """``C_2 <= S_2``, ``C_2(0)`` the fraction; two separate cubes: no pair of the same
    cluster across them, where ``S_2`` counts the pairs of both."""
    img, _ = boolean
    s2 = an.covariance(img, 1, max_lag=20).mean()[1]
    c2 = an.two_point_cluster(img, 1, max_lag=20).mean()[1]
    assert np.all(c2 <= s2 + 1e-12) and c2[0] == pytest.approx(s2[0])
    data = np.zeros((32, 8, 8), dtype=np.uint8)
    data[2:6, 2:6, 2:6] = 1
    data[10:14, 2:6, 2:6] = 1
    image = VoxelImage(data, periodic=True)
    s2 = an.covariance(image, 1, max_lag=10).values["x"]
    c2 = an.two_point_cluster(image, 1, max_lag=10).values["x"]
    assert s2[8] > 0 and c2[8] == 0.0 and np.allclose(c2[:4], s2[:4])
    assert np.allclose(c2[:4] * data.size, 2 * 16 * (4 - np.arange(4)))


def test_point_patterns():
    """``g = 1`` and ``S = 1`` for Poisson points; ``g = 0`` inside the contact distance of
    hard spheres, with a peak at contact; ``S = 0`` below the Bragg peaks of a lattice."""
    rng = np.random.default_rng(1)
    pts = rng.random((3000, 3)) * 10.0
    r, g = an.radial_distribution(pts, (10, 10, 10), bins=25)
    assert np.abs(g[4:] - 1.0).max() < 0.15
    _, s = an.structure_factor(pts, (10, 10, 10))
    assert s.mean() == pytest.approx(1.0, abs=0.05)
    spheres = mg.models.HardcoreSpheres(radius=0.3, volume_fraction=0.3).generate(
        mg.Domain.cube(10.0, periodic=True), seed=0).spheres  # fmt: skip
    r, g = an.radial_distribution(spheres.centers, (10, 10, 10), r_max=2.0, bins=40)
    assert np.all(g[r < 0.58] == 0) and g[(r > 0.6) & (r < 0.7)].max() > 1.3
    lattice = np.indices((10, 10, 10)).reshape(3, -1).T + 0.5
    _, s = an.structure_factor(lattice, (10, 10, 10), k_max=1.9 * math.pi)
    assert s.max() < 1e-20
    with pytest.raises(ValueError, match="half the box"):
        an.radial_distribution(pts, (10, 10, 10), r_max=6.0)
