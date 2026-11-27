"""The reconstruction from descriptors by gradient descent (morphology.reconstruction): the
two-point probability and the lineal paths of a relaxed field against morphoGen's own on
binary fields, the gradient of the energy against finite differences, the targets in closed
form (the covariance of a Boolean model of spheres, the lineal paths of its pores from its
Choquet capacity), reconstructions of a Debye medium, of the pores of a Boolean model —
the covariance alone leaves the lineal paths astray, which their own term mends — and of
a 3D structure from a planar section, and the method of the model reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

import morphogen as mg
from morphogen.analysis.correlation import lineal_path
from morphogen.analysis.covariance import covariance_map
from morphogen.domain import Domain, Grid
from morphogen.morphology import reconstruction as rc

UNIT = (1.0, 1.0, 1.0)


def test_descriptors_of_binary_fields():
    """On a field of 0 and 1, the two-point probability of every lag is morphoGen's
    covariance map, and the lineal paths its lineal-path function, exactly."""
    rng = np.random.default_rng(0)
    b = (rng.random((12, 10, 8)) < 0.4).astype(np.uint8)
    image = mg.VoxelImage(b, spacing=UNIT, periodic=True)
    np.testing.assert_allclose(np.fft.fftshift(rc.two_point(b)), covariance_map(image, 1),
                               atol=1e-15)  # fmt: skip
    measured = lineal_path(image, 1, max_lag=4)
    lines = rc.lineal_paths(b, 4)
    for i, name in enumerate("xyz"):
        np.testing.assert_allclose(lines[i], measured.values[name], atol=1e-15)
    lags = rc.lag_vectors((4, 3, 1), (0.5, 2.0, 1.0))
    assert lags[3, 2, 0].tolist() == [-0.5, -2.0, 0.0] and lags[2, 1, 0].tolist() == [1.0, 2.0, 0.0]


def test_gradient_of_the_energy():
    """Every term of the energy (two-point probability, fraction, binarization, lineal
    paths, cells at the bounds included) against centred finite differences."""
    rng = np.random.default_rng(1)
    shape = (8, 7, 6)
    target = rc.debye_descriptors(shape, UNIT, fraction=0.35, length=1.5, max_lag=3.0)
    target.lineal = rc.lineal_paths((rng.random(shape) < 0.35).astype(float), 3, (0, 1, 2))
    x = rng.random(shape)
    x[0, 0, :3] = 0.0  # at the lower bound: the products of the other cells
    e, g = rc.energy(x, target, 0.2, 0.7, 3.0)
    g = g.reshape(shape)
    for _ in range(16):
        i = tuple(int(rng.integers(0, n)) for n in shape)
        h = 1e-7
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h if x[i] > h else 0.0
        step = up[i] - down[i]
        fd = rc.energy(up, target, 0.2, 0.7, 3.0)[0] - rc.energy(down, target, 0.2, 0.7, 3.0)[0]
        assert fd / step == pytest.approx(g[i], rel=1e-5, abs=1e-9)
    assert e > 0


def test_targets_in_closed_form():
    """The covariance of a Boolean model of spheres at every lag of the grid (the pores:
    ``1 - 2p + C``), the lineal paths of its pores from its Choquet capacity; a model without
    a covariance of the distance alone measured on a realization."""
    spheres = mg.models.BooleanSpheres(radius=2.0, volume_fraction=0.3)
    shape = (16, 16, 16)
    grains = rc.descriptors_of_model(spheres, shape, UNIT, phase=1, max_lag=5.0)
    distance = np.linalg.norm(rc.lag_vectors(shape, UNIT), axis=-1)
    np.testing.assert_allclose(grains.two_point, spheres.covariance(distance))
    assert grains.fraction == pytest.approx(0.3)
    assert grains.mask.sum() == int((distance <= 5.0).sum())
    pores = rc.descriptors_of_model(spheres, shape, UNIT, phase=0, max_lag=5.0, lineal=True)
    np.testing.assert_allclose(pores.two_point, 1.0 - 0.6 + grains.two_point)
    assert "Choquet capacity" in pores.source and pores.lineal.shape == (3, 6)
    expected = 1.0 - spheres.choquet_capacity("segment", 3.0)
    np.testing.assert_allclose(pores.lineal[:, 3], expected)
    hardcore = mg.models.HardcoreSpheres(radius=0.3, volume_fraction=0.3)
    measured = rc.descriptors_of_model(hardcore, (12, 12, 12), (0.25, 0.25, 0.25), phase=1,
                                       max_lag=1.0, seed=2)  # fmt: skip
    assert "realization" in measured.source and 0.0 < measured.fraction < 1.0


def test_reconstruction_of_a_debye_medium():
    """The two-point probability of a Debye medium at every lag within the largest, in 2D
    and in 3D, within 0.5 % of the variance of the indicator, the fraction exact."""
    for shape, lag in (((48, 48, 1), 12.0), ((24, 24, 24), 6.0)):
        target = rc.debye_descriptors(shape, UNIT, fraction=0.3, length=2.0, max_lag=lag)
        done = rc.reconstruct(target, seed=1, iterations=2000)
        assert set(np.unique(done.structure)) == {0, 1}
        assert done.errors["two_point"] < 0.005
        assert abs(done.errors["fraction"]) < 1e-3
        assert done.iterations > 0 and len(done.energies) == len(rc.STAGES)


def test_the_covariance_does_not_determine_the_morphology():
    """The pores of plane sections of a Boolean model of spheres, from their exact targets:
    the covariance alone reproduced at every lag, the lineal path not (7 % of the variance);
    with the lineal paths among the targets, both within 1 %."""
    spheres = mg.models.BooleanSpheres(radius=4.0, volume_fraction=0.3)
    target = rc.descriptors_of_model(spheres, (96, 96, 1), UNIT, phase=0, max_lag=16.0,
                                     lineal=True)  # fmt: skip
    alone = rc.reconstruct(target, seed=2, iterations=1500, lineal_weight=0.0)
    assert alone.errors["two_point"] < 0.005 and alone.errors["lineal_path"] > 0.05
    both = rc.reconstruct(target, seed=2, iterations=1500, lineal_weight=1.0)
    assert both.errors["two_point"] < 0.01 and both.errors["lineal_path"] < 0.01
    end = rc.lineal_paths(both.structure.astype(float), 16, (0, 1)).mean(axis=0)[-1]
    assert end == pytest.approx(target.lineal.mean(axis=0)[-1], abs=0.005)


def test_a_volume_from_a_section():
    """A 3D structure from the two-point probability of a planar section of a Boolean
    model of spheres on the three coordinate planes: within 1 % there, and within 5 % of the
    exact 3D covariance at the lags off the planes."""
    spheres = mg.models.BooleanSpheres(radius=3.0, volume_fraction=0.3)
    domain = Domain(size=(128.0, 128.0, 7.0), periodic=True)
    voxels = spheres.generate(domain, seed=11).voxelize(grid=Grid(domain, (128, 128, 7)))
    section = mg.VoxelImage((np.asarray(voxels.data)[:, :, 3:4] == 1).astype(np.uint8),
                            spacing=UNIT, periodic=True)  # fmt: skip
    target = rc.descriptors_of_image(section, (24, 24, 24), UNIT, phase=1, max_lag=6.0)
    assert target.source == "the sections of an image"
    done = rc.reconstruct(target, seed=3, iterations=900)
    assert done.errors["two_point"] < 0.01
    exact = rc.descriptors_of_model(spheres, (24, 24, 24), UNIT, phase=1, max_lag=6.0)
    off = exact.mask & ~target.mask
    s2 = rc.two_point(done.structure.astype(float))
    variance = exact.fraction * (1.0 - exact.fraction)
    assert np.sqrt(np.mean((s2 - exact.two_point)[off] ** 2)) / variance < 0.05
    with pytest.raises(ValueError, match="planar"):
        rc.descriptors_of_image(voxels, (8, 8, 8), UNIT, max_lag=2.0, sections=True)


def test_the_method_of_the_model(tmp_path):
    """Reconstruction(method="gradient"): from the exact covariance of a reference, of a
    Debye medium and of an image (a planar one for a volume); its configuration; its
    errors."""
    reference = mg.models.BooleanSpheres(radius=0.3, volume_fraction=0.3)
    model = mg.models.Reconstruction(reference=reference, cell_size=0.1, method="gradient",
                                     iterations=600, max_lag=6)  # fmt: skip
    micro = model.generate(mg.Domain.cube(2.4, periodic=True), seed=1)
    a = micro.attributes
    assert a["method"] == "gradient" and "closed form" in a["targets"]
    assert a["error_two_point"] < 0.02 and abs(a["error_fraction"]) < 1e-3
    assert mg.models.model_from_config(model.to_config()) == model
    debye = mg.models.Reconstruction(correlation_length=0.2, volume_fraction=0.4,
                                     cell_size=0.1, method="gradient", iterations=600,
                                     dimension=2)  # fmt: skip
    planar = debye.generate(mg.Domain(size=(4.8, 4.8, 0.1), periodic=True), seed=2)
    assert planar.attributes["error_two_point"] < 0.02
    section = mg.VoxelImage((np.random.default_rng(3).random((64, 64, 1)) < 0.35).astype(np.uint8),
                            spacing=(0.1, 0.1, 0.1), periodic=True)  # fmt: skip
    mg.io.save_image(section, tmp_path / "section.vti")
    from_image = mg.models.Reconstruction(image=str(tmp_path / "section.vti"), cell_size=0.1,
                                          method="gradient", iterations=300, max_lag=4)  # fmt: skip
    volume = from_image.generate(mg.Domain.cube(1.6, periodic=True), seed=4)
    assert volume.attributes["targets"] == "the sections of an image"
    target = from_image.descriptors(mg.Domain.cube(1.6, periodic=True), seed=4)
    assert target.shape == (16, 16, 16) and target.fraction == pytest.approx(section.data.mean())
    with pytest.raises(ValueError, match="method"):
        mg.models.Reconstruction(correlation_length=0.2, volume_fraction=0.4, method="swaps")
    with pytest.raises(ValueError, match="iterations"):
        mg.models.Reconstruction(correlation_length=0.2, volume_fraction=0.4, iterations=0)
