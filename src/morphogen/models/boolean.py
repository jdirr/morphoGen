"""Boolean models: unions of random grains implanted on Poisson points."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen import theory
from morphogen._utils import check_fraction, check_positive, exactly_one
from morphogen.distributions import Constant, Distribution
from morphogen.domain import Domain
from morphogen.models.affine import ANISOTROPY, Anisotropy, anisotropic
from morphogen.models.base import Model, param, register
from morphogen.models.union import PrimitiveUnion
from morphogen.orientations import Isotropic, OrientationDistribution, angle_weights
from morphogen.primitives import Segments, Spheres
from morphogen.seeding import SeedLike, make_rng


def _germ_window(domain: Domain, extent: float, name: str) -> Domain:
    """Window in which germs must be drawn for an unbiased sample of ``domain``."""
    if domain.periodic:
        if 2.0 * extent >= min(domain.size):
            raise ValueError(
                f"{name}: grains (extent {extent:g}) must be smaller than half the periodic "
                f"box (edge {min(domain.size):g})"
            )
        return domain
    return domain.dilated(extent)


@register
@anisotropic
@dataclass(frozen=True)
class BooleanSpheres(Model):
    """Boolean model of spheres: union of balls centred on a Poisson point process.

    Give either the volume fraction of the union or the intensity of the germs; the
    other follows from ``p = 1 - exp(-theta * E[4/3 pi R^3])``.
    """

    name = "boolean_spheres"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "spheres"}
    count_label = "spheres"

    radius: Distribution = param(
        Constant(1.0),
        doc="Sphere radius (number or distribution)",
        kind="distribution",
        unit="length",
    )
    volume_fraction: float | None = param(
        None, doc="Volume fraction of the union of spheres", kind="fraction"
    )
    intensity: float | None = param(
        None, doc="Number of sphere centres per unit volume", unit="1/length^3"
    )
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        exactly_one("volume_fraction", self.volume_fraction, "intensity", self.intensity)
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        else:
            check_positive(self.intensity, "intensity")

    @property
    def mean_grain_volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius.moment(3)

    @property
    def theta(self) -> float:
        """Intensity of the Poisson germs."""
        if self.intensity is not None:
            return self.intensity
        return theory.boolean_intensity(self.volume_fraction, self.mean_grain_volume)

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        rng, seed_info = make_rng(seed)
        rmax = self.radius.upper()
        window = _germ_window(domain, rmax, self.name)
        n = rng.poisson(self.theta * window.volume)
        centers = window.lower + rng.random((n, 3)) * window.lengths
        radii = self.radius.sample(rng, n)
        spheres = Spheres(centers, radii)
        if not domain.periodic:
            spheres = spheres.subset(domain.distance_to_box(centers) <= radii)
        return PrimitiveUnion(domain, spheres=spheres, model=self, seed=seed_info)

    # -- theory --------------------------------------------------------------------------

    def expected_count(self, domain: Domain) -> float:
        """Mean number of germs (grain centres) in the domain."""
        return self.theta * domain.volume

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        theta = self.theta
        q = math.exp(-theta * self.mean_grain_volume)
        s = 4.0 * math.pi * self.radius.moment(2)  # mean surface of the spheres
        m = 4.0 * math.pi * self.radius.moment(1)  # their integral of mean curvature
        return {
            "intensity": theta,
            "volume_fraction": 1.0 - q,
            "specific_surface": theta * q * s,
            # Miles' formulae of isotropic Boolean models of convex grains
            "mean_curvature_density": q * (theta * m - math.pi**2 / 32.0 * (theta * s) ** 2),
            "euler_density": q
            * (theta - theta**2 * m * s / (4.0 * math.pi) + math.pi / 384.0 * (theta * s) ** 3),
            "integral_range": self.integral_range(),
            "variance_exponent": 1.0,
        }

    def choquet_capacity(self, element: str, size) -> np.ndarray:
        """``T(K) = 1 - exp(-theta E[|A' (+) K'|])``: balls of radius ``size`` (dilated
        spheres of radius R + r), segments of length ``size`` (the sphere swept along the
        segment: (4/3) pi R^3 + pi R^2 l), pairs of points at distance ``size``
        (``2 p - C(h)``)."""
        size = np.asarray(size, dtype=float)
        theta = self.theta
        m1, m2, m3 = (self.radius.moment(k) for k in (1, 2, 3))
        if element == "ball":
            volume = 4.0 / 3.0 * math.pi * (m3 + 3.0 * size * m2 + 3.0 * size**2 * m1 + size**3)
        elif element == "segment":
            volume = 4.0 / 3.0 * math.pi * m3 + math.pi * m2 * size
        elif element == "pair":
            p = 1.0 - math.exp(-theta * self.mean_grain_volume)
            return 2.0 * p - self.covariance(size)
        else:
            raise ValueError('element: "ball", "segment" or "pair"')
        return -np.expm1(-theta * volume)

    def _mean_covariogram(self, h):
        nodes, weights = self.radius.quadrature(96)
        h = np.asarray(h, dtype=float)
        return np.sum(weights * theory.sphere_covariogram(h[..., None], nodes), axis=-1)

    def covariance(self, h) -> np.ndarray:
        """C(h) = P(x in A, x + h in A) for the union A of the spheres."""
        theta = self.theta
        q = math.exp(-theta * self.mean_grain_volume)
        big_q = q * q * np.exp(theta * self._mean_covariogram(h))
        return 1.0 - 2.0 * q + big_q

    def integral_range(self) -> float:
        """A3 = int (C(h) - p^2) dh / (p (1 - p)) of the indicator of the union."""
        theta = self.theta
        q = math.exp(-theta * self.mean_grain_volume)
        p = 1.0 - q

        def f(h):
            return q * q * math.expm1(theta * float(self._mean_covariogram(h)))

        return theory.radial_integral(f, 2.0 * self.radius.upper()) / (p * q)


@register
@dataclass(frozen=True)
class BooleanCylinders(Model):
    """Boolean model of finite fibres: cylinders (flat caps) or spherocylinders.

    Fibre centres form a Poisson point process; radius, length (of the cylindrical
    part) and axis orientation are independent random variables. The covariance
    (:meth:`covariance`) is in closed form along any direction, for any law of the axes
    (Peyrega et al. 2009).
    """

    name = "boolean_cylinders"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "fibres"}
    count_label = "fibres"

    radius: Distribution = param(
        Constant(1.0), doc="Fibre radius", kind="distribution", unit="length"
    )
    length: Distribution = param(
        Constant(10.0),
        doc="Length of the cylindrical part of the fibres",
        kind="distribution",
        unit="length",
    )
    orientation: OrientationDistribution = param(
        Isotropic(),
        doc="Distribution of the fibre axes (isotropic, aligned, planar, beta, girdle, "
        "von_mises_fisher, mixture)",
        kind="orientation",
    )
    caps: str = param(
        "flat", doc="End caps: 'flat' (cylinders) or 'round' (spherocylinders)", kind="str"
    )
    volume_fraction: float | None = param(None, doc="Volume fraction of fibres", kind="fraction")
    intensity: float | None = param(None, doc="Fibre centres per unit volume", unit="1/length^3")

    def _validate(self) -> None:
        exactly_one("volume_fraction", self.volume_fraction, "intensity", self.intensity)
        if self.caps not in ("flat", "round"):
            raise ValueError(f"caps: expected 'flat' or 'round', got {self.caps!r}")
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        else:
            check_positive(self.intensity, "intensity")

    @property
    def mean_grain_volume(self) -> float:
        v = math.pi * self.radius.moment(2) * self.length.moment(1)
        if self.caps == "round":
            v += 4.0 / 3.0 * math.pi * self.radius.moment(3)
        return v

    @property
    def mean_grain_surface(self) -> float:
        s = 2.0 * math.pi * self.radius.moment(1) * self.length.moment(1)
        if self.caps == "round":
            return s + 4.0 * math.pi * self.radius.moment(2)
        return s + 2.0 * math.pi * self.radius.moment(2)

    @property
    def theta(self) -> float:
        if self.intensity is not None:
            return self.intensity
        return theory.boolean_intensity(self.volume_fraction, self.mean_grain_volume)

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        rng, seed_info = make_rng(seed)
        rmax, lmax = self.radius.upper(), self.length.upper()
        extent = 0.5 * lmax + rmax
        window = _germ_window(domain, extent, self.name)
        n = rng.poisson(self.theta * window.volume)
        centers = window.lower + rng.random((n, 3)) * window.lengths
        axes = self.orientation.sample(rng, n)
        radii = self.radius.sample(rng, n)
        half = 0.5 * self.length.sample(rng, n)[:, None]
        segments = Segments(
            centers - half * axes, centers + half * axes, radii, round_caps=self.caps == "round"
        )
        if not domain.periodic:
            lo, hi = segments.bounds()
            keep = np.all((hi >= domain.lower) & (lo <= domain.upper), axis=1)
            segments = segments.subset(keep)
        return PrimitiveUnion(domain, segments=segments, model=self, seed=seed_info)

    def expected_count(self, domain: Domain) -> float:
        """Mean number of germs (grain centres) in the domain."""
        return self.theta * domain.volume

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        theta = self.theta
        q = math.exp(-theta * self.mean_grain_volume)
        return {
            "intensity": theta,
            "volume_fraction": 1.0 - q,
            "specific_surface": theta * q * self.mean_grain_surface,
            "integral_range": self.integral_range(),
        }

    def _covariogram(self, h, angle) -> np.ndarray:
        """``E_{r,l}[K]`` at the distances ``h`` and angles ``angle`` to the axis
        (broadcasting): ``(l - h cos b)+ lens(h sin b)`` for cylinders, the expectations over
        the lengths and the radii apart (32 quantiles each when random); the integral of
        the lens of the sections for spherocylinders (6 quantiles of each when random)."""
        along, across = h * np.cos(angle), h * np.sin(angle)
        if self.caps == "flat":
            ls, wl = self.length.quadrature(32)
            rs, wr = self.radius.quadrature(32)
            reach = sum(w * np.maximum(x - along, 0.0) for x, w in zip(ls, wl, strict=True))
            lens = sum(
                w * theory.disk_intersection_area(across, x) for x, w in zip(rs, wr, strict=True)
            )
            return reach * lens
        out = np.zeros(np.broadcast_shapes(np.shape(along), np.shape(across)))
        for r, wr in zip(*self.radius.quadrature(6), strict=True):
            for ell, wl in zip(*self.length.quadrature(6), strict=True):
                out += wr * wl * theory.spherocylinder_covariogram(along, across, r, ell)
        return out

    def mean_covariogram(self, h, direction=None) -> np.ndarray:
        """``E[K(h)] = E|K ∩ (K + h)|``, the mean geometric covariogram of the fibres over
        their radii, lengths and axes, for ``h`` of the lengths ``h`` along ``direction``
        (averaged over the directions without one): ``K = (l - |h·u|)+ lens(|h × u|)`` for
        cylinders (Peyrega et al. 2009, eq. 15), the integral along the axis of the lens of
        the sections for spherocylinders (:func:`morphogen.theory.spherocylinder_covariogram`);
        ``E[K(0)] = E[V]`` and ``int E[K(h)] dh = E[V^2]``."""
        h = np.abs(np.asarray(h, dtype=float))
        b, w = angle_weights(self.orientation, direction)
        return (self._covariogram(h.reshape(-1, 1), b[None, :]) @ w).reshape(h.shape)

    def covariance(self, h, direction=None) -> np.ndarray:
        """``C(h) = P(x in A, x + h in A) = 1 - 2q + q^2 exp(theta E[K(h)])`` along
        ``direction`` (any for isotropic axes), or averaged over the directions of ``h``
        without one (:func:`morphogen.theory.oriented_covariance`)."""
        theta = self.theta
        q = math.exp(-theta * self.mean_grain_volume)
        return theory.oriented_covariance(
            q, theta, self._covariogram, h, self.orientation, direction
        )

    def integral_range(self) -> float:
        """``A3 = int (C(h) - p^2) dh / (p q)`` over the space: ``4 pi int h^2 (C̄(h) - p^2)
        dh / (p q)``, ``C̄`` the covariance averaged over the directions (96 Gauss-Legendre
        distances up to the largest extent of a fibre); the mean volume of the fibres
        weighted by their volumes, ``E[V^2] / E[V]``, as ``p -> 0``."""
        theta = self.theta
        q = math.exp(-theta * self.mean_grain_volume)
        p = 1.0 - q
        reach = self.length.upper() + 2.0 * self.radius.upper()
        x, w = np.polynomial.legendre.leggauss(96)
        h, w = 0.5 * reach * (x + 1.0), 0.5 * reach * w
        centred = self.covariance(h) - p * p
        return float(np.sum(w * 4.0 * math.pi * h * h * centred) / (p * q))
