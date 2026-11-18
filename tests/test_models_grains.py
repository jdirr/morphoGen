"""Grain and fibre models (Boolean models, Poisson fibres, hard spheres) against theory."""

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen import analysis
from morphogen.kernels.rsa import random_sequential_addition
from morphogen.models import BooleanCylinders, BooleanSpheres, HardcoreSpheres, PoissonFibers
from morphogen.models.fibers import _clip_lines
from morphogen.models.hardcore import largest_first
from morphogen.seeding import derive_seed

# -- Boolean spheres -------------------------------------------------------------------------


def test_boolean_spheres_volume_fraction_and_covariance():
    model = BooleanSpheres(radius=1.0, volume_fraction=0.3)
    th = model.theory()
    assert th["volume_fraction"] == pytest.approx(0.3)
    domain = mg.Domain.cube(24.0)
    vf, cov = [], []
    for seed in range(3):
        img = model.generate(domain, seed=seed).voxelize(shape=144)
        vf.append(analysis.volume_fraction(img))
        lags, c = analysis.covariance(img, max_lag=24).mean()
        cov.append(c)
    # standard deviation of the volume fraction of one sample: sqrt(p q A3 / V)
    sigma = math.sqrt(0.3 * 0.7 * th["integral_range"] / domain.volume) / math.sqrt(3)
    assert abs(np.mean(vf) - 0.3) < 4 * sigma
    np.testing.assert_allclose(np.mean(cov, axis=0), model.covariance(lags), atol=4 * sigma)


def test_boolean_spheres_specific_surface_converges():
    model = BooleanSpheres(radius=1.0, volume_fraction=0.3)
    micro = model.generate(mg.Domain.cube(16.0), seed=4)
    errors = []
    for n in (64, 128, 256):
        img = micro.voxelize(shape=n)
        p = analysis.volume_fraction(img)
        # theory with the intensity matching this realization's volume fraction
        theta = -math.log(1 - p) / model.mean_grain_volume
        expected = theta * (1 - p) * 4 * math.pi
        errors.append(analysis.specific_surface(img) / expected - 1)
    # Crofton estimate biased low by missed short chords, by about h / mean chord
    assert errors[0] < errors[1] < errors[2] < 0
    assert abs(errors[2]) < 0.02


def test_boolean_spheres_integral_range_limits():
    # small volume fraction: A3 tends to the grain volume
    dilute = BooleanSpheres(radius=1.0, volume_fraction=1e-4)
    assert dilute.integral_range() == pytest.approx(4 / 3 * math.pi, rel=1e-3)
    m = BooleanSpheres(radius=1.0, volume_fraction=0.5)
    # C(0) = p and C(h) = p^2 beyond the grain diameter
    np.testing.assert_allclose(m.covariance([0.0, 2.0, 5.0]), [0.5, 0.25, 0.25], atol=1e-12)


def test_boolean_spheres_random_radii_unbiased_in_small_windows():
    model = BooleanSpheres(radius={"type": "uniform", "low": 0.5, "high": 1.5}, volume_fraction=0.3)
    vf = [
        analysis.volume_fraction(
            model.generate(mg.Domain.cube(6.0, periodic=False), seed=s).voxelize(shape=30)
        )
        for s in range(60)
    ]
    sigma = math.sqrt(0.21 * model.integral_range() / 216.0) / math.sqrt(60)
    # germs implanted in the dilated window: no edge effect
    assert abs(np.mean(vf) - 0.3) < 4 * sigma


def test_boolean_spheres_parameters():
    m = BooleanSpheres(radius=2.0, intensity=0.01)
    assert m.theory()["volume_fraction"] == pytest.approx(1 - math.exp(-0.01 * 32 / 3 * math.pi))
    with pytest.raises(ValueError, match="exactly one"):
        BooleanSpheres(radius=1.0, volume_fraction=0.2, intensity=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        BooleanSpheres(radius=1.0)
    with pytest.raises(ValueError):
        BooleanSpheres(radius=1.0, volume_fraction=1.2)
    with pytest.raises(ValueError, match="half the periodic box"):
        BooleanSpheres(radius=3.0, volume_fraction=0.2).generate(mg.Domain.cube(5.0))


# -- Boolean cylinders -------------------------------------------------------------------


@pytest.mark.parametrize(
    "caps, orientation, periodic",
    [
        ("flat", "isotropic", True),
        ("round", "isotropic", False),
        ("flat", {"type": "aligned", "direction": [0, 0, 1]}, False),
        ("round", {"type": "planar", "normal": [0, 0, 1]}, True),
    ],
)
def test_boolean_cylinders_volume_fraction(caps, orientation, periodic):
    model = BooleanCylinders(
        radius=0.4, length=4.0, orientation=orientation, caps=caps, volume_fraction=0.2
    )
    vf = [
        analysis.volume_fraction(
            model.generate(mg.Domain.cube(12.0, periodic=periodic), seed=s).voxelize(
                voxel_size=0.15
            )
        )
        for s in range(6)
    ]
    assert abs(np.mean(vf) - 0.2) < 4 * np.std(vf, ddof=1) / math.sqrt(len(vf)) + 0.004


def test_crofton_13_directions_handle_anisotropy():
    model = BooleanCylinders(
        radius=0.5, length=8.0, orientation={"type": "aligned"}, volume_fraction=0.15
    )
    micro = model.generate(mg.Domain.cube(16.0), seed=2)
    img = micro.voxelize(voxel_size=0.08)
    p = analysis.volume_fraction(img)
    theta = -math.log(1 - p) / model.mean_grain_volume
    expected = theta * (1 - p) * model.mean_grain_surface
    err13 = analysis.specific_surface(img) / expected - 1
    err3 = analysis.specific_surface(img, directions=3) / expected - 1
    # 13 directions: ~2 % quadrature error for this anisotropy plus the digitization of
    # fibres aligned with the grid; 3 axes: about -15 % for fibres along an axis
    assert abs(err13) < 0.07
    assert err3 < -0.12 and abs(err3) > 2 * abs(err13)


# -- Poisson fibres --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def poisson_fibre_samples():
    model = PoissonFibers(radius=1.0, volume_fraction=0.16)
    domain = mg.Domain.cube(30.0, periodic=False)
    micros = [model.generate(domain, seed=100 + s) for s in range(40)]
    images = [m.voxelize(voxel_size=0.25) for m in micros]
    return model, domain, micros, images


def test_poisson_fibers_volume_fraction(poisson_fibre_samples):
    model, domain, _, images = poisson_fibre_samples
    vf = np.array([analysis.volume_fraction(img) for img in images])
    # variance scaling of Dirrenberger et al. (2014): D^2(V) ~ 0.44 V^(-2/3) for R = 1
    sigma = math.sqrt(0.44 * domain.volume ** (-2 / 3) / len(vf))
    assert abs(vf.mean() - 0.16) < 4 * sigma
    assert model.theory()["variance_exponent"] == pytest.approx(2 / 3)


def test_poisson_fibers_covariance(poisson_fibre_samples):
    model, _, _, images = poisson_fibre_samples
    cov = np.mean(
        [analysis.covariance(img, max_lag=40, periodic=False).mean()[1] for img in images], axis=0
    )
    lags = np.arange(41) * 0.25
    np.testing.assert_allclose(cov, model.covariance(lags), atol=0.008)
    # slow decay towards p^2: infinite integral range
    assert model.covariance(20.0) - 0.16**2 > 1e-4


def test_poisson_fibers_count_and_isotropy(poisson_fibre_samples):
    model, domain, micros, _ = poisson_fibre_samples
    counts, dz = [], []
    for m in micros:
        seg = m.segments
        d = (seg.end - seg.start) / seg.lengths[:, None]
        t0, t1 = _clip_lines(seg.start, d, domain.lower, domain.upper)
        counts.append(np.count_nonzero(t0 <= t1))  # axes crossing the box
        dz.extend(np.abs(d[:, 2]))
    expected = model.expected_fibers(domain)
    assert abs(np.mean(counts) - expected) < 4 * math.sqrt(expected / len(counts))
    assert np.mean(dz) == pytest.approx(0.5, abs=0.02)


def test_poisson_fibers_theory_and_errors():
    m = PoissonFibers(radius=2.0, length_density=0.01)
    th = m.theory()
    assert th["volume_fraction"] == pytest.approx(1 - math.exp(-0.01 * math.pi * 4))
    assert th["theta_jeulin"] == pytest.approx(0.01 / math.pi)
    assert math.isinf(th["integral_range"])
    with pytest.raises(ValueError, match="periodic"):
        m.generate(mg.Domain.cube(10.0, periodic=True))


# -- hard-core spheres -----------------------------------------------------------------------


def _min_gap(spheres, box=None):
    diff = spheres.centers[:, None] - spheres.centers[None]
    if box is not None:
        diff -= box * np.round(diff / box)
    dist = np.linalg.norm(diff, axis=2) - spheres.radii[:, None] - spheres.radii[None]
    np.fill_diagonal(dist, np.inf)
    return dist.min()


def test_hardcore_spheres_periodic_target():
    model = HardcoreSpheres(radius=1.0, volume_fraction=0.3)
    micro = model.generate(mg.Domain.cube(12.0), seed=1)
    s = micro.spheres
    assert s.volumes().sum() / 12.0**3 == pytest.approx(0.3, abs=4 / 3 * math.pi / 12.0**3)
    assert _min_gap(s, 12.0) >= -1e-12
    img = micro.voxelize(shape=96, supersampling=2)
    assert analysis.volume_fraction(img) == pytest.approx(0.3, abs=0.005)


def test_hardcore_spheres_polydisperse_non_periodic():
    model = HardcoreSpheres(
        radius={"type": "uniform", "low": 0.5, "high": 1.0}, volume_fraction=0.25, min_gap=0.1
    )
    micro = model.generate(mg.Domain.cube(8.0, periodic=False), seed=2)
    s = micro.spheres
    assert _min_gap(s) >= 0.1 - 1e-12
    assert np.all(micro.domain.distance_to_box(s.centers) <= s.radii)
    with pytest.raises(RuntimeError, match="jammed"):
        HardcoreSpheres(radius=1.0, volume_fraction=0.5, max_attempts=50_000).generate(
            mg.Domain.cube(10.0), seed=0
        )


def test_hardcore_spheres_place_the_largest_first():
    # drawn late, a large sphere found no room left: these radii jammed from a fraction of
    # 0.31 in the order of their draw, and an RVE study of the interface on the template
    # (cubes of edge 10) on its 17th realization
    radius = {"type": "uniform", "low": 0.8, "high": 1.2}
    model = HardcoreSpheres(radius=radius, volume_fraction=0.3, min_gap=0.05)
    s = model.generate(mg.Domain.cube(10.0), seed=derive_seed(1, 0, 16)).spheres
    assert _min_gap(s, 10.0) >= 0.05 - 1e-12
    dense = HardcoreSpheres(radius=radius, volume_fraction=0.38, min_gap=0.05)
    s = dense.generate(mg.Domain.cube(20.0), seed=1).spheres
    assert s.volumes().sum() / 20.0**3 == pytest.approx(0.38, abs=0.002)
    assert _min_gap(s, 20.0) >= 0.05 - 1e-12
    # equal spheres are placed as before
    box, radii = np.full(3, 10.0), np.full(60, 1.0)
    before = random_sequential_addition(radii, box, 0.0, np.random.default_rng(5), 10**6)
    after = largest_first(radii, box, 0.0, np.random.default_rng(5), 10**6)
    assert np.array_equal(after, before)
