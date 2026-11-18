"""Anisotropy: the MIL fabric tensor (isotropic and affinely stretched Boolean spheres, a
foam) and the first minima of the covariance (an egg crate, lamellae, a foam)."""

import numpy as np
import pytest

import morphogen as mg
from morphogen import analysis as an
from morphogen.image import VoxelImage


@pytest.fixture(scope="module")
def spheres():
    structure = mg.models.BooleanSpheres(radius=0.05, volume_fraction=0.3)
    return structure.generate(mg.Domain.cube(1.0, periodic=True), seed=1).voxelize((96, 96, 96))


def test_fabric_of_isotropic_and_stretched_structures(spheres):
    iso = an.fabric_tensor(spheres)
    assert iso.degree < 0.05
    # MIL along the axes: the mean chord lengths of the phase
    for axis, d in ((0, (1, 0, 0)), (2, (0, 0, 1))):
        k = next(i for i, u in enumerate(iso.directions) if np.allclose(u, d))
        assert iso.mil[k] == pytest.approx(an.chord_lengths(spheres, 1, axis).mean(), rel=0.05)
    # the same voxels on a grid of spacings 1 : 1.5 : 2, an affine stretch of the structure
    stretched = an.fabric_tensor(VoxelImage(spheres.data, spacing=(1.0, 1.5, 2.0), periodic=True))
    np.testing.assert_allclose(stretched.ratios, [2.0, 1.5, 1.0], rtol=0.04)
    np.testing.assert_allclose(np.abs(stretched.axes[:, ::-1]), np.eye(3), atol=0.05)
    assert stretched.degree == pytest.approx(0.5, abs=0.03)
    out = an.measure(spheres, ["anisotropy"])
    assert set(out) == {"mil_1", "mil_2", "mil_3", "degree_of_anisotropy"}


def test_fabric_of_an_elongated_foam():
    foam = mg.models.LevelSetFoam(
        seed_radius=0.1, aspect=(1.2, 2.0), thickness=0.3, packing="rsa", packing_fraction=0.3
    )
    img = foam.generate(mg.Domain.cube(2.0, periodic=True), seed=2).voxelize((64, 64, 64))
    f = an.fabric_tensor(img, 0)
    assert (
        1.0 < f.ratios[1] < f.ratios[0] and np.argmax(np.abs(f.axes[:, 0])) == 2
    )  # longest along z
    np.testing.assert_allclose(
        an.fabric_tensor(img, 1).ratios, f.ratios, rtol=1e-9
    )  # P_L common to both


def test_covariance_minima_of_exact_and_cellular_structures():
    n = 96
    x, y, z = np.meshgrid(*(np.arange(n) + 0.5,) * 3, indexing="ij")
    crate = np.cos(2 * np.pi * x / 16) * np.cos(2 * np.pi * y / 24) * np.cos(2 * np.pi * z / 32) > 0
    hole = an.covariance_minima(
        VoxelImage(crate.astype(np.uint8), spacing=(0.5, 0.5, 0.5), periodic=True)
    )
    np.testing.assert_allclose(hole.lags, [4.0, 6.0, 8.0], atol=1e-12)  # half the periods
    np.testing.assert_allclose(hole.values, 0.0, atol=1e-12)
    np.testing.assert_allclose(hole.ratios, [2.0, 1.5, 1.0])
    lamellae = np.zeros((96, 8, 8), np.uint8)
    lamellae[np.arange(96) % 12 < 6] = 1
    lags = an.covariance_minima(VoxelImage(lamellae, periodic=True)).lags
    assert lags[0] == pytest.approx(6.0) and np.isnan(lags[1:]).all()  # none along the lamellae
    # a dense foam of seeds 1 : 1.2 : 2: the lags follow the cells, the longest along z
    foam = mg.models.LevelSetFoam(
        seed_radius=0.1,
        aspect=(1.2, 2.0),
        packing="dense",
        packing_fraction=0.6,
        volume_fraction=0.25,
    )
    img = foam.generate(mg.Domain.cube(2.0, periodic=True), seed=2).voxelize((64, 64, 64))
    hole = an.covariance_minima(img, 1)
    assert np.argmax(hole.lags) == 2 and hole.ratios[0] == pytest.approx(2.0, rel=0.1)
    assert hole.values.max() < hole.plateau
    out = an.measure(img, ["covariance_minima"])
    np.testing.assert_allclose([out[f"covariance_minimum_{a}"] for a in "xyz"], hole.lags)
