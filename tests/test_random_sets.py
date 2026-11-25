"""Choquet capacities, morphological covariance, Minkowski functionals and Betti numbers
against the closed forms of Boolean and Gaussian models."""

from __future__ import annotations

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen import analysis as an
from morphogen.image import VoxelImage


@pytest.fixture(scope="module")
def spheres():
    structure = mg.models.BooleanSpheres(radius=0.06, volume_fraction=0.3).generate(
        mg.Domain.cube(1.0, periodic=True), seed=3
    )
    # the closed forms at the intensity of the realization: its own fluctuations aside
    model = mg.models.BooleanSpheres(radius=0.06, intensity=float(len(structure.spheres)))
    return model, structure.voxelize((96, 96, 96))


def test_capacity_of_boolean_spheres(spheres):
    model, img = spheres
    h = img.spacing[0]
    r = model.radius.moment(1)
    t = an.choquet_capacity(img, "ball", np.array([0.0, 1, 2, 4]) * h)
    assert t.values[0] == pytest.approx(t.volume_fraction)
    # -log Q(B_r) is proportional to (R + r)^3: the digital dilation of the digitized
    # spheres is that of spheres a third of a voxel smaller
    lq = np.log(t.complement)
    radius = r * (lq / lq[0]) ** (1.0 / 3.0) - t.sizes
    assert np.all(np.abs(radius - r) < 0.5 * h)
    np.testing.assert_allclose(
        t.values, model.choquet_capacity("ball", t.sizes + radius - r), rtol=0.01
    )
    # segments: -log Q(l) / -log q = 1 + 3 l / 4 R (the spheres swept along them)
    for direction in ((1, 0, 0), (1, 1, 1)):
        seg = an.choquet_capacity(img, "segment", np.array([4, 12, 24]) * h, direction=direction)
        ratio = np.log(seg.complement) / lq[0]
        np.testing.assert_allclose(ratio, 1.0 + 0.75 * seg.sizes / r, rtol=0.04)
    pair = an.choquet_capacity(img, "pair", np.array([2, 6, 20]) * h, direction=(0, 1, 0))
    np.testing.assert_allclose(pair.values, model.choquet_capacity("pair", pair.sizes), rtol=0.03)
    # H(r): the distribution of the distance to the spheres from outside
    hs = t.contact_distribution()
    assert hs[0] == pytest.approx(0.0, abs=1e-12) and np.all(np.diff(hs) > 0)


def test_capacity_minus_sampling_and_elements(spheres):
    _, img = spheres
    h = img.spacing[0]
    per = an.choquet_capacity(img, "ball", [3 * h])
    box = an.choquet_capacity(img, "ball", [3 * h], periodic=False)
    assert box.values[0] == pytest.approx(per.values[0], abs=0.02)
    # the capacity is the volume fraction of the dilation, whatever the estimator
    for element, size in (("ball", 3 * h), ("segment", 5 * h), ("cube", 3 * h), ("pair", 6 * h)):
        t = an.choquet_capacity(img, element, [size])
        dilated = an.dilation(img, element, size)
        assert dilated.mean() == pytest.approx(t.values[0], abs=1e-12)
    # openings are inside the set and closings contain it
    x = an.indicator(img, 1)
    o, c = an.opening(img, "ball", 2 * h), an.closing(img, "ball", 2 * h)
    assert not np.any(o & ~x) and not np.any(x & ~c)


def test_capacity_of_poisson_fibers():
    model = mg.models.PoissonFibers(radius=0.03, volume_fraction=0.2)
    img = model.generate(mg.Domain.cube(1.0, periodic=False), seed=5).voxelize((96, 96, 96))
    h, r = img.spacing[0], model.radius.mean()  # a constant law of radii
    # minus sampling; -log Q(B_r) is proportional to (R + r)^2, -log Q(l) to 1 + l / 2R
    t = an.choquet_capacity(img, "ball", np.array([0.0, 2, 5]) * h)
    lq = np.log(t.complement)
    radius = r * np.sqrt(lq / lq[0]) - t.sizes
    assert np.all(np.abs(radius - r) < 0.5 * h)
    seg = an.choquet_capacity(img, "segment", np.array([6, 20]) * h, direction=(1, 1, 0))
    np.testing.assert_allclose(np.log(seg.complement) / lq[0], 1.0 + 0.5 * seg.sizes / r, rtol=0.05)
    assert model.choquet_capacity("ball", 0.0) == pytest.approx(model.theory()["volume_fraction"])


def test_morphological_covariance_of_gaussian_field():
    ell = 3.0 / 96
    model = mg.models.GaussianExcursion(correlation_length=ell, volume_fraction=0.5)
    field = model.generate(mg.Domain.cube(1.0, periodic=True), seed=11).field((96, 96, 96))
    k = an.morphological_covariance(field, max_lag=12)
    lags, values = k.mean()
    z = np.asarray(field.data)
    rho = np.exp(-0.5 * (lags / ell) ** 2)
    # E[min(X, Y)] = m - s sqrt((1 - rho) / pi) for a Gaussian pair of correlation rho
    expected = z.mean() - z.std() * np.sqrt((1.0 - rho) / math.pi)
    np.testing.assert_allclose(values, expected, atol=0.02)
    hi = an.morphological_covariance(field, max_lag=12, operation="dilation").mean()[1]
    np.testing.assert_allclose(
        hi + values, 2.0 * z.astype(float).mean(), atol=1e-6
    )  # min + max = X + Y
    binary = an.morphological_covariance(
        VoxelImage((z > 0).astype(float), periodic=True), max_lag=6
    )
    cov = an.covariance(VoxelImage((z > 0).astype(np.uint8), periodic=True), max_lag=6)
    np.testing.assert_allclose(binary.values["x"], cov.values["x"], atol=1e-12)


def test_minkowski_functionals_of_boolean_spheres(spheres):
    model, img = spheres
    m = an.minkowski(img)
    th = model.theory()
    assert m.volume_fraction == pytest.approx(th["volume_fraction"], rel=0.01)
    assert m.specific_surface == pytest.approx(th["specific_surface"], rel=0.06)
    # spheres closer than a voxel touch on the image: M_V is a little low (and the
    # Euler characteristic, a difference of nearly equal terms, too noisy to test here)
    assert m.mean_curvature == pytest.approx(th["mean_curvature_density"], rel=0.15)
    n, radius = 64, 20.0  # one ball: M = 4 pi R, chi = 1
    ball = VoxelImage(_ball(n, (31.5, 31.5, 31.5), radius).astype(np.uint8))
    one = an.minkowski(ball)
    assert one.mean_curvature * n**3 == pytest.approx(4.0 * math.pi * radius, rel=0.03)
    assert one.euler_density * n**3 == pytest.approx(1.0)


def test_minkowski_curves_of_gaussian_excursions():
    ell = 4.0 / 96
    levels = [-1.0, 0.0, 1.0]
    model = mg.models.GaussianExcursion(correlation_length=ell, threshold=0.0)
    field = model.generate(mg.Domain.cube(1.0, periodic=True), seed=2).field((96, 96, 96))
    z = np.asarray(field.data)
    z = (z - z.mean()) / z.std()
    curves = an.minkowski_curves(VoxelImage(z, spacing=field.spacing, periodic=True), levels)
    for i, u in enumerate(levels):
        th = mg.models.GaussianExcursion(correlation_length=ell, threshold=u).theory()
        assert curves["volume_fraction"][i] == pytest.approx(th["volume_fraction"], abs=0.01)
        assert curves["specific_surface"][i] == pytest.approx(th["specific_surface"], rel=0.05)
        # the scales of M_V and chi_V (both vanish at some levels)
        m_scale, chi_scale = ell**-2 / math.sqrt(2.0 * math.pi), ell**-3 / (4.0 * math.pi**2)
        assert curves["mean_curvature"][i] == pytest.approx(
            th["mean_curvature_density"], abs=0.08 * m_scale
        )
        assert curves["euler_density"][i] == pytest.approx(
            th["euler_density"], abs=0.05 * chi_scale
        )


def _ball(n, centre, radius):
    g = np.indices((n, n, n)).transpose(1, 2, 3, 0) - np.asarray(centre)
    return np.linalg.norm(g, axis=-1) <= radius


def test_betti_numbers():
    n = 40
    g = np.indices((n, n, n)).transpose(1, 2, 3, 0) - (n - 1) / 2.0
    rho = np.hypot(g[..., 0], g[..., 1])
    torus = (rho - 12.0) ** 2 + g[..., 2] ** 2 <= 4.0**2
    shell = _ball(n, (19.5, 19.5, 19.5), 15) & ~_ball(n, (19.5, 19.5, 19.5), 10)
    two = _ball(n, (10, 10, 10), 6) | _ball(n, (29, 29, 29), 6)
    eight = ((np.hypot(g[..., 0] - 9, g[..., 1]) - 7.0) ** 2 + g[..., 2] ** 2 <= 2.5**2) | (
        (np.hypot(g[..., 0] + 9, g[..., 1]) - 7.0) ** 2 + g[..., 2] ** 2 <= 2.5**2
    )
    for mask, expected in (
        (torus, (1, 1, 0)),
        (shell, (1, 0, 1)),
        (two, (2, 0, 0)),
        (eight, (1, 2, 0)),
    ):
        b = an.betti_numbers(VoxelImage(mask.astype(np.uint8)))
        assert (b.b0, b.b1, b.b2) == expected
        assert b.euler == b.b0 - b.b1 + b.b2
    assert an.connectivity_density(VoxelImage(torus.astype(np.uint8))) == pytest.approx(1 / n**3)


def test_morphological_measures_by_name(spheres):
    _, img = spheres
    out = an.measure(img, ["minkowski", "betti"])
    assert {"V_V", "S_V", "M_V", "chi_V", "betti_0", "betti_1", "betti_2"} <= set(out)
    assert out["V_V"] == pytest.approx(an.volume_fraction(img))
