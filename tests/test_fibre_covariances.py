"""Closed-form covariances of fibre systems (Peyrega, Jeulin et al.): the quadratures of the
orientation laws, the covariograms of cylinders and spherocylinders, the covariances of
Poisson fibres and Boolean fibres of any law against realizations along the axes and the
mean fibre axis."""

import math

import numpy as np
import pytest
from scipy import stats

import morphogen as mg
from morphogen import analysis, theory
from morphogen.models import BooleanCylinders, PoissonFibers
from morphogen.orientations import (
    Aligned,
    Beta,
    Girdle,
    Isotropic,
    Mixture,
    Planar,
    VonMisesFisher,
    angle_weights,
    orientation_from_config,
)

LAWS = [
    Isotropic(),
    Beta(0.2),
    Beta(4.0, (1.0, 1.0, 0.0)),
    Girdle(0.4),
    Girdle(0.1, (1.0, 0.0, 0.0)),
    VonMisesFisher(5.0, (0.0, 1.0, 1.0)),
    Planar((1.0, 2.0, 3.0)),
    Mixture((Beta(0.114), Beta(10.1, (1.0, 0.0, 0.0))), (0.393, 0.607)),
]


@pytest.mark.parametrize("law", LAWS, ids=lambda law: type(law).__name__)
def test_quadratures_of_the_orientation_laws(law):
    u, w = law.quadrature()
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0) and w.sum() == pytest.approx(1.0)
    s = law.sample(np.random.default_rng(3), 400_000)
    for f in (lambda v: v[:, 0] ** 2, lambda v: v[:, 2] ** 2, lambda v: np.abs(v[:, 0] * v[:, 1])):
        assert w @ f(u) == pytest.approx(np.mean(f(s)), abs=2.5e-3)  # 4 standard errors
    assert orientation_from_config(law.to_config()) == law


def test_von_mises_fisher_and_mixture_laws():
    law = VonMisesFisher(3.0, (0.0, 0.0, 2.0))
    w = law.sample(np.random.default_rng(1), 20_000)[:, 2]

    def cdf(x):
        return (np.exp(3.0 * x) - math.exp(-3.0)) / (2.0 * math.sinh(3.0))

    assert stats.kstest(w, cdf).pvalue > 0.01
    flat = VonMisesFisher(0.0).sample(np.random.default_rng(2), 20_000)[:, 2]
    assert stats.kstest(flat, "uniform", args=(-1.0, 2.0)).pvalue > 0.01
    laws = [{"type": "aligned"}, {"type": "aligned", "direction": [1, 0, 0]}]
    mix = orientation_from_config({"type": "mixture", "weights": [1, 3], "laws": laws})
    u = mix.sample(np.random.default_rng(4), 40_000)
    assert mix.weights == (0.25, 0.75) and np.mean(u[:, 2] == 1.0) == pytest.approx(0.25, abs=0.01)
    with pytest.raises(ValueError, match="weights"):
        Mixture((Isotropic(),), (1.0, 2.0))
    with pytest.raises(ValueError, match="laws\\[1\\]"):
        orientation_from_config({"type": "mixture", "laws": ["isotropic", {"type": "bad"}]})
    with pytest.raises(ValueError, match="kappa"):
        VonMisesFisher(-1.0)


def test_angle_weights():
    b, w = angle_weights(Aligned((0.0, 0.0, 1.0)), (1.0, 0.0, 1.0))
    assert b == pytest.approx([math.pi / 4]) and w == pytest.approx([1.0])
    b, w = angle_weights(Beta(0.5), (0.0, 0.0, 1.0))  # gathered on the grid
    s = Beta(0.5).sample(np.random.default_rng(5), 200_000)
    assert w.sum() == pytest.approx(1.0) and w @ np.cos(b) ** 2 == pytest.approx(
        np.mean(s[:, 2] ** 2), abs=2e-3
    )
    b, w = angle_weights(Beta(0.5))  # without a direction: cos b uniform
    assert w @ np.cos(b) == pytest.approx(0.5)


def test_covariograms_of_cylinders_and_spherocylinders():
    r, ell = 1.0, 6.0
    volume = math.pi * r * r * ell + 4.0 / 3.0 * math.pi * r**3
    k = theory.spherocylinder_covariogram
    assert k(0.0, 0.0, r, ell) == pytest.approx(volume, rel=1e-12)
    d = np.array([0.3, 1.0, 1.7, 2.5])  # across the axis: the cylinder and a sphere
    exact = ell * theory.disk_intersection_area(d, r) + theory.sphere_covariogram(d, r)
    np.testing.assert_allclose(k(0.0, d, r, ell), exact, rtol=3e-5)  # the tangent sections
    a = np.array([0.5, 3.0, 6.0, 6.5, 7.9])  # along it: a shorter capsule, then two balls
    exact = np.where(a <= ell, math.pi * r * r * (ell - a) + 4.0 / 3.0 * math.pi * r**3,
                     theory.sphere_covariogram(a - ell, r))  # fmt: skip
    np.testing.assert_allclose(k(a, 0.0, r, ell), exact, rtol=1e-10)
    # the integral of the covariogram over the space is the squared volume
    c, wc = np.polynomial.legendre.leggauss(96)
    c, wc = 0.5 * (c + 1.0), 0.5 * wc
    x, wx = np.polynomial.legendre.leggauss(300)
    h, wx = 0.5 * (ell + 2 * r) * (x + 1.0), 0.5 * (ell + 2 * r) * wx
    weight = 4.0 * math.pi * (wx * h * h)[:, None] * wc[None]
    along, across = h[:, None] * c[None], h[:, None] * np.sqrt(1.0 - c * c)[None]
    assert np.sum(weight * k(along, across, r, ell)) == pytest.approx(volume**2, rel=2e-5)
    flat = theory.cylinder_covariogram(along, across, r, ell)
    assert np.sum(weight * flat) == pytest.approx((math.pi * r * r * ell) ** 2, rel=2e-5)
    # an oblique shift against points drawn in the capsule
    rng = np.random.default_rng(1)
    lo, hi = np.array([-r, -r, -r]), np.array([r, r, ell + r])
    p = lo + rng.random((2_000_000, 3)) * (hi - lo)

    def inside(q):
        z = np.clip(q[:, 2], 0.0, ell)
        return q[:, 0] ** 2 + q[:, 1] ** 2 + (q[:, 2] - z) ** 2 <= r * r

    shift = np.array([0.8, 0.0, 2.0])
    estimate = np.prod(hi - lo) * np.mean(inside(p) & inside(p - shift))
    assert estimate == pytest.approx(k(2.0, 0.8, r, ell), rel=0.01)


def test_lens_area():
    assert theory.lens_area(0.5, 2.0, 1.0) == pytest.approx(math.pi)  # the small one inside
    assert theory.lens_area(3.0, 2.0, 1.0) == 0.0
    d = np.linspace(0.0, 2.0, 9)
    np.testing.assert_allclose(theory.lens_area(d, 1.0, 1.0), theory.disk_intersection_area(d, 1.0))


@pytest.mark.parametrize("caps", ["flat", "round"])
def test_dilute_integral_range_and_origin(caps):
    for law in (Isotropic(), Beta(0.3)):
        model = BooleanCylinders(radius=1.0, length=8.0, caps=caps, volume_fraction=1e-5,
                                 orientation=law)  # fmt: skip
        v = model.mean_grain_volume
        assert model.mean_covariogram(0.0, (0.0, 1.0, 0.0)) == pytest.approx(v, rel=1e-10)
        # A3 -> E[V^2] / E[V], the volume of a fibre (directions on a grid: 1e-3)
        assert model.integral_range() == pytest.approx(v, rel=1e-3)
        cov = model.covariance([0.0, 30.0], (1.0, 0.0, 0.0))
        assert cov[0] == pytest.approx(1e-5, rel=1e-6) and abs(cov[1] - 1e-10) < 1e-14
    random = BooleanCylinders(radius={"type": "uniform", "low": 0.5, "high": 1.0},
                              length={"type": "uniform", "low": 4.0, "high": 8.0},
                              caps=caps, volume_fraction=1e-5)  # fmt: skip
    if caps == "flat":
        r4 = (1.0 - 0.5**5) / (5 * 0.5)
        l2 = (8.0**3 - 4.0**3) / (3 * 4.0)
        expected = math.pi**2 * r4 * l2 / random.mean_grain_volume
        assert random.integral_range() == pytest.approx(expected, rel=1e-3)


def _along(data, step, lags: int, periodic: bool) -> np.ndarray:
    """The covariance of a binary image along a lattice direction (unit steps ``step``)."""
    out = []
    for k in range(lags + 1):
        s = np.asarray(step) * k
        if periodic:
            out.append(np.mean(data & np.roll(data, tuple(s), axis=(0, 1, 2))))
        else:
            n = data.shape
            a = data[tuple(slice(0, n[i] - s[i]) for i in range(3))]
            b = data[tuple(slice(s[i], n[i]) for i in range(3))]
            out.append(np.mean(a & b))
    return np.array(out)


def _residuals(model, realizations, step, periodic, fitted):
    """Mean differences between the measured covariances (along x, y, z and the lattice
    direction ``step``) and those of the model of the measured fraction of each
    realization (which removes the fluctuation of the number of fibres)."""
    h = realizations[0].spacing[0]
    out = {k: [] for k in ("x", "y", "z", "mean axis")}
    direction = np.asarray(step, dtype=float)
    for img in realizations:
        data = img.data == 1
        fit = fitted(float(data.mean()))
        cov = analysis.covariance(img, max_lag=40, periodic=periodic)
        lags = np.arange(41) * h
        for k, e in zip("xyz", np.eye(3), strict=True):
            out[k].append(cov.values[k] - fit.covariance(lags, e))
        mine = _along(data, step, 23, periodic)
        dist = np.arange(24) * h * np.linalg.norm(direction)
        out["mean axis"].append(mine - fit.covariance(dist, direction))
    return {k: np.mean(v, axis=0) for k, v in out.items()}


@pytest.mark.parametrize("caps", ["round", "flat"])
def test_boolean_fibres_covariance_along_directions(caps):
    """Fibres of a beta law about (1, 1, 1): the covariances measured along the three axes
    and along the mean fibre axis follow the closed forms (Boolean round fibres: within
    0.0026 over four realizations of 120^3 voxels; 0.0028 flat)."""
    axis = (1.0, 1.0, 1.0)

    def fitted(p):
        return BooleanCylinders(radius=1.0, length=8.0, caps=caps, volume_fraction=p,
                                orientation=Beta(0.3, axis))  # fmt: skip

    model = fitted(0.2)
    images = [model.generate(mg.Domain.cube(24.0), seed=s).voxelize(voxel_size=0.2)
              for s in range(4)]  # fmt: skip
    res = _residuals(model, images, (1, 1, 1), True, fitted)
    for k, r in res.items():
        assert np.abs(r).max() < 0.005, k
    # the mean axis differs from the axes: C(3.46) along it and along x
    dist = 10 * 0.2 * math.sqrt(3.0)
    assert model.covariance(dist, axis) - model.covariance(dist, (1, 0, 0)) > 0.007


def test_poisson_fibres_covariance_along_directions():
    """Poisson lines of a girdle about z and of a von Mises-Fisher law about (1, 1, 1):
    the measured covariances along the axes and the mean axis within 0.004 of the closed
    forms over twelve windows (0.0016 and 0.0021 measured)."""
    for law, step in ((Girdle(0.4), (0, 0, 1)), (VonMisesFisher(4.0, (1.0, 1.0, 1.0)), (1, 1, 1))):

        def fitted(p, law=law):
            return PoissonFibers(radius=1.0, volume_fraction=p, orientation=law)

        model = fitted(0.16)
        domain = mg.Domain.cube(30.0, periodic=False)
        images = [model.generate(domain, seed=100 + s).voxelize(voxel_size=0.25)
                  for s in range(12)]  # fmt: skip
        res = _residuals(model, images, step, False, fitted)
        for k, r in res.items():
            assert np.abs(r).max() < 0.004, (type(law).__name__, k)
    girdle = PoissonFibers(radius=1.0, volume_fraction=0.16, orientation=Girdle(0.4))
    assert girdle.covariance(1.25, (1, 0, 0)) - girdle.covariance(1.25, (0, 0, 1)) > 0.015


def test_poisson_fibres_of_any_law():
    iso = PoissonFibers(radius=1.0, volume_fraction=0.16)
    h = np.linspace(0.0, 10.0, 21)
    x, w = np.polynomial.legendre.leggauss(128)
    sin_b = np.sqrt(1.0 - (0.5 * (x + 1.0)) ** 2)
    mean_lens = 0.5 * np.sum(w * theory.disk_intersection_area(h[:, None] * sin_b, 1.0), axis=1)
    q = 0.84
    np.testing.assert_allclose(iso.covariance(h), 1 - 2 * q + q * q * np.exp(iso.lv * mean_lens))
    assert iso.covariance(h, (1.0, 2.0, 3.0)) == pytest.approx(iso.covariance(h))
    # lines along z: the fibres are a Boolean model of disks in the planes normal to them
    lines = PoissonFibers(radius=1.0, volume_fraction=0.16, orientation=Aligned())
    domain = mg.Domain(size=(10.0, 20.0, 30.0), periodic=False)
    assert lines.expected_fibers(domain) == pytest.approx(lines.lv * 200.0)
    assert lines.covariance(7.0, (0.0, 0.0, 1.0)) == pytest.approx(0.16)
    assert lines.choquet_capacity("segment", 5.0, (0.0, 0.0, 1.0)) == pytest.approx(0.16)
    assert lines.choquet_capacity("segment", 5.0, (1.0, 0.0, 0.0)) == pytest.approx(
        1.0 - math.exp(-lines.lv * (math.pi + 10.0))
    )
    beta = PoissonFibers(radius=1.0, volume_fraction=0.16, orientation=Beta(0.5))
    m = beta.generate(mg.Domain.cube(20.0, periodic=False), seed=3)
    d = (m.segments.end - m.segments.start) / m.segments.lengths[:, None]
    assert np.mean(d[:, 2] ** 2) > 0.42  # towards z (isotropic: 1/3)
    assert beta.theory()["volume_fraction"] == pytest.approx(0.16)


def test_curves_tool_draws_the_theory_along_each_axis():
    from morphogen.analysis.covariance import AxisCovariance
    from morphogen.gui.tools.statistics import _axis_covariances, _capacities

    lags = np.arange(5) * 0.5
    curve = AxisCovariance({a: lags for a in "xyz"}, {a: lags for a in "xyz"}, 0.2)
    model = BooleanCylinders(radius=1.0, length=8.0, volume_fraction=0.2, orientation=Girdle(0.4))
    out = _axis_covariances(model, curve)
    assert set(out) == {"x", "y", "z"} and out["x"][1][2] != pytest.approx(out["z"][1][2])
    single = _axis_covariances(BooleanCylinders(radius=1.0, length=8.0, volume_fraction=0.2), curve)
    assert isinstance(single, tuple) and single[1][0] == pytest.approx(0.2)
    # the capacity of the segments measured, along x, not its mean over the directions
    lines = PoissonFibers(radius=1.0, volume_fraction=0.1, orientation=Aligned((0.0, 0.0, 1.0)))
    sizes = np.array([0.0, 4.0, 12.0])
    _, along = _capacities(lines, "segment", sizes)
    assert along == pytest.approx(lines.choquet_capacity("segment", sizes, (1.0, 0.0, 0.0)))
    assert along[-1] > lines.choquet_capacity("segment", sizes)[-1] + 0.05
