"""Closed-form results of stochastic geometry used by the models (see docs/theory.md)."""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special, stats

#: Poisson-Voronoi tessellation in R^3 (intensity theta of germs):
#: squared coefficient of variation of the cell volume (Gilbert 1962),
VORONOI_VOLUME_CV2 = 0.1790
#: mean number of faces of the typical cell (Meijering 1953), 2 + 48 pi^2 / 35,
VORONOI_MEAN_FACES = 2.0 + 48.0 * math.pi**2 / 35.0
#: and cell-boundary area per unit volume S_V = c * theta^(1/3), with
#: c = (256 pi / 3)^(1/3) Gamma(5/3) / 2 (Meijering 1953).
VORONOI_SURFACE_COEFF = (256.0 * math.pi / 3.0) ** (1.0 / 3.0) * math.gamma(5.0 / 3.0) / 2.0

#: Jamming (saturation) volume fraction of random sequential addition of equal spheres.
RSA_JAMMING_FRACTION = 0.3841
#: Fraction of the maximally random jammed packings of equal spheres (Torquato, Truskett
#: & Debenedetti 2000), which fast Lubachevsky–Stillinger compressions approach.
MRJ_FRACTION = 0.64


def sphere_covariogram(h, r):
    """Volume of ``B(0, r) ∩ B(h, r)`` for ``|h| = h`` (broadcasts)."""
    h = np.asarray(h, dtype=float)
    r = np.asarray(r, dtype=float)
    v = np.pi / 12.0 * (4.0 * r + h) * (2.0 * r - h) ** 2
    return np.where(h < 2.0 * r, v, 0.0)


def disk_intersection_area(d, r):
    """Area of the intersection of two disks of radius ``r`` whose centres are ``d`` apart."""
    d = np.asarray(d, dtype=float)
    x = np.clip(d / (2.0 * r), 0.0, 1.0)
    a = 2.0 * r * r * np.arccos(x) - 0.5 * d * np.sqrt(np.maximum(4.0 * r * r - d * d, 0.0))
    return np.where(d < 2.0 * r, a, 0.0)


def lens_area(d, r1, r2):
    """Area of the intersection of two disks of radii ``r1`` and ``r2`` whose centres are
    ``d`` apart (broadcasts): the smaller disk when it lies in the other, else the two
    circular segments cut by their common chord."""
    d, r1, r2 = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (d, r1, r2)))
    inside = d <= np.abs(r1 - r2)
    out = np.where(inside, np.pi * np.minimum(r1, r2) ** 2, 0.0)
    part = ~inside & (d < r1 + r2)
    if np.any(part):
        dd, a, b = d[part], r1[part], r2[part]
        ca = np.clip((dd * dd + a * a - b * b) / (2.0 * dd * a), -1.0, 1.0)
        cb = np.clip((dd * dd + b * b - a * a) / (2.0 * dd * b), -1.0, 1.0)
        k = (-dd + a + b) * (dd + a - b) * (dd - a + b) * (dd + a + b)
        out[part] = (
            a * a * np.arccos(ca) + b * b * np.arccos(cb) - 0.5 * np.sqrt(np.maximum(k, 0.0))
        )
    return out


def cylinder_covariogram(along, across, radius, length):
    """Volume of ``K ∩ (K + h)`` for the cylinder ``K`` of ``radius`` and ``length`` (flat
    caps), ``h`` of components ``along = |h·u|`` and ``across = |h × u|`` relative to its
    axis ``u`` (broadcasts): ``(l - along)+ lens(across)`` (Peyrega et al. 2009, eq. 15)."""
    along = np.abs(np.asarray(along, dtype=float))
    return np.maximum(length - along, 0.0) * disk_intersection_area(across, radius)


#: Gauss-Legendre nodes per piece of the integrals along the axis of spherocylinders
_CAPSULE_NODES = 24


def spherocylinder_covariogram(along, across, radius, length):
    """Volume of ``K ∩ (K + h)`` for the spherocylinder ``K`` (round caps) of ``radius``
    around an axis of ``length``, ``h`` of components ``along`` and ``across`` its axis
    (broadcasts): the integral along the axis of the lens of the two sections, disks of
    the radii ``rho(z)`` and ``rho(z - along)`` of the profile (``r`` on the axis, ``sqrt(r^2
    - z^2)`` over the caps), ``across`` apart, by Gauss-Legendre over the pieces between
    the ends of the axes and of the caps and the height where the facing caps have equal
    sections (exact on the cylinders: ``(l - along)+ lens``)."""
    a = np.abs(np.asarray(along, dtype=float))
    a, d, r, ell = np.broadcast_arrays(
        a, *(np.asarray(x, dtype=float) for x in (across, radius, length))
    )
    lo, hi = a - r, ell + r  # the axis of K from 0 to l, that of K + h from a to a + l
    cuts = np.stack([lo, np.zeros_like(a), ell, a, a + ell, 0.5 * (ell + a), hi], axis=-1)
    cuts = np.sort(np.clip(cuts, lo[..., None], np.maximum(lo, hi)[..., None]), axis=-1)
    x, w = np.polynomial.legendre.leggauss(_CAPSULE_NODES)
    half = 0.5 * np.diff(cuts, axis=-1)  # (..., 6)
    z = 0.5 * (cuts[..., 1:] + cuts[..., :-1])[..., None] + half[..., None] * x  # (..., 6, m)

    def profile(t):  # the radius of the section of K at the height t
        out = np.sqrt(
            np.maximum(r[..., None, None] ** 2 - np.maximum(-t, t - ell[..., None, None]) ** 2, 0.0)
        )
        return np.where((t >= 0.0) & (t <= ell[..., None, None]), r[..., None, None], out)

    area = lens_area(d[..., None, None], profile(z), profile(z - a[..., None, None]))
    return np.sum(half * np.sum(w * area, axis=-1), axis=-1)


def oriented_covariance(q: float, rate: float, covariogram, h, orientation, direction=None):
    """``C(h) = 1 - 2 q + Q(h)`` of a Boolean model of grains of axes drawn from
    ``orientation`` (fibres, or the cylinders of Poisson lines): ``Q(h) = q^2 exp(rate
    E_u[covariogram(|h|, b)])``, ``covariogram(h, b)`` (broadcasting, ``b`` the angle between
    ``h`` and the axis of a grain) its mean geometric covariogram over the sizes of the
    grains, averaged over the angles of :func:`morphogen.orientations.angle_weights`. Along
    ``direction`` (a vector), or, for isotropic axes, along any; without a direction, the
    covariance of anisotropic axes averaged over the directions of ``h`` (72 directions of
    a half sphere: 6 Gauss-Legendre cosines and 12 azimuths), the mean taken of ``Q``."""
    from morphogen.orientations import ANGLE_GRID, Isotropic, angle_weights, half_sphere

    h = np.asarray(h, dtype=float)
    flat = np.abs(h.reshape(-1))
    if direction is not None or isinstance(orientation, Isotropic):
        b, w = angle_weights(orientation, direction)
        big_q = q * q * np.exp(rate * (covariogram(flat[:, None], b[None, :]) @ w))
    else:  # the mean over the directions of h: the angles on the grid
        grid = np.linspace(0.0, 0.5 * np.pi, ANGLE_GRID)
        table = covariogram(flat[:, None], grid[None, :])
        big_q = np.zeros(len(flat))
        for u, wu in zip(*half_sphere(6), strict=True):
            big_q += (
                wu * q * q * np.exp(rate * (table @ angle_weights(orientation, u, grid=True)[1]))
            )
    return (1.0 - 2.0 * q + big_q).reshape(h.shape)


def boolean_intensity(volume_fraction: float, mean_grain_volume: float) -> float:
    """Intensity theta of a Boolean model: p = 1 - exp(-theta * E[V])."""
    return -math.log1p(-volume_fraction) / mean_grain_volume


def boolean_volume_fraction(intensity: float, mean_grain_volume: float) -> float:
    """The volume fraction ``1 - exp(-θ V̄)`` of a Boolean model of intensity θ and mean grain
    volume V̄."""
    return -math.expm1(-intensity * mean_grain_volume)


def bivariate_normal_upper(level, rho):
    """P(X > level, Y > level) for standard normals X, Y of correlation ``rho``.

    Uses P = (1 - Phi(l))^2 + 1/(2 pi) int_0^asin(rho) exp(-l^2 / (1 + sin t)) dt.
    """
    rho = np.asarray(rho, dtype=float)
    q = stats.norm.sf(level)
    x, w = np.polynomial.legendre.leggauss(64)
    a = np.arcsin(np.clip(rho, -1.0, 1.0))[..., None]
    t = 0.5 * a * (x + 1.0)
    f = np.exp(-(level**2) / (1.0 + np.sin(t)))
    return q * q + 0.5 * a[..., 0] * np.sum(w * f, axis=-1) / (2.0 * math.pi)


def radial_integral(f, upper: float) -> float:
    """``int_{|h| < upper} f(|h|) dh`` in R^3 for an isotropic function ``f``."""
    value, _ = integrate.quad(lambda h: 4.0 * math.pi * h * h * f(h), 0.0, upper, limit=200)
    return float(value)


def normal_quantile(p: float) -> float:
    """The quantile of order ``p`` of the standard normal distribution."""
    return float(special.ndtri(p))
