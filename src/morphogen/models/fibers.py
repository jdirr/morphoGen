"""Fibre models: Poisson fibres, a Boolean model on isotropic Poisson lines (Jeulin
1991); hard-core (non-overlapping) fibres by random sequential addition; curved fibres,
worm-like chains, arcs or helices, overlapping or not; bending fibres, random walks packed
without overlap to high fractions by forces (Altendorf & Jeulin 2011), drawn into
contact (Keilmann, Redenbach & Willot 2026); entangled cross-linked fibres, straight
fibres bonded at a fraction of their contacts (Chatti et al. 2018, 2020)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen import theory
from morphogen._utils import check_fraction, check_positive, exactly_one
from morphogen.distributions import Constant, Distribution
from morphogen.domain import Domain
from morphogen.kernels.rsa import (
    random_sequential_addition_fibres,
    random_sequential_addition_polylines,
)
from morphogen.models import _sizes as sizes
from morphogen.models.base import Model, param, register
from morphogen.models.boolean import _germ_window
from morphogen.models.combination import Combined
from morphogen.models.union import PrimitiveUnion
from morphogen.orientations import (
    Aligned,
    Isotropic,
    OrientationDistribution,
    Planar,
    angle_weights,
    half_sphere,
    orthonormal_basis,
)
from morphogen.primitives import Segments, Spheres
from morphogen.seeding import SeedLike, make_rng


def _clip_lines(points, directions, lower, upper):
    """Parameters [t0, t1] of the lines ``p + t d`` inside the box (slab method)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / directions
        ta = (lower - points) * inv
        tb = (upper - points) * inv
    t_near = np.where(np.isnan(ta), -np.inf, np.minimum(ta, tb))
    t_far = np.where(np.isnan(tb), np.inf, np.maximum(ta, tb))
    # lines parallel to a slab and outside it never enter the box
    parallel = directions == 0.0
    outside = parallel & ((points < lower) | (points > upper))
    t0 = np.max(np.where(parallel, -np.inf, t_near), axis=1)
    t1 = np.min(np.where(parallel, np.inf, t_far), axis=1)
    t1 = np.where(np.any(outside, axis=1), -np.inf, t1)
    return t0, t1


@register
@dataclass(frozen=True)
class PoissonFibers(Model):
    """Infinite cylinders of radius R around Poisson lines, isotropic or of directions
    drawn from ``orientation``.

    The model is parameterized by the volume fraction ``p`` of fibres or by the mean
    fibre length per unit volume ``L_V`` (``p = 1 - exp(-L_V pi E[R^2])`` for any law of
    directions). In the notation of Jeulin and of Dirrenberger et al. (2014) the intensity
    of the lines is ``theta = L_V / pi`` and ``p = 1 - exp(-theta pi^2 R^2)``. The radius is
    a number (monodisperse fibres) or a distribution (polydisperse: one radius per line,
    independent of the line), the closed forms then taking the moments of its law.

    Infinite lines cannot be made periodic: domains must be non-periodic. The sample is
    the exact restriction of the infinite random medium to the box. The covariance
    (:meth:`covariance`) is in closed form along any direction, for any law.
    """

    name = "poisson_fibers"
    phases: ClassVar[dict[int, str]] = {0: "pores", 1: "fibres"}
    supports_periodic = False
    count_label = "fibres"

    radius: Distribution = param(
        Constant(1.0),
        doc="Fibre radius: a number (monodisperse) or a distribution (polydisperse, one "
        "radius per fibre)",
        kind="distribution",
        unit="length",
    )
    volume_fraction: float | None = param(None, doc="Volume fraction of fibres", kind="fraction")
    length_density: float | None = param(
        None, doc="Mean fibre length per unit volume L_V", unit="1/length^2"
    )
    orientation: OrientationDistribution = param(
        Isotropic(),
        doc="Distribution of the directions of the lines (isotropic, aligned, planar, beta, "
        "girdle, von_mises_fisher, mixture)",
        kind="orientation",
    )

    def _validate(self) -> None:
        exactly_one("volume_fraction", self.volume_fraction, "length_density", self.length_density)
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        else:
            check_positive(self.length_density, "length_density")

    @property
    def lv(self) -> float:
        """Mean fibre length per unit volume."""
        if self.length_density is not None:
            return self.length_density
        return -math.log1p(-self.volume_fraction) / (math.pi * self.radius.moment(2))

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        self._check_domain(domain)
        rng, seed_info = make_rng(seed)
        r = self.radius.upper()
        # lines hitting the ball containing the box dilated by r: the number is Poisson
        # with mean L_V * pi * rho^2 (mean projected area of the ball)
        rho = 0.5 * domain.diagonal + r
        n = rng.poisson(self.lv * math.pi * rho * rho)
        directions = self.orientation.sample(rng, n).reshape(n, 3)  # isotropic: uniform
        e1, e2 = orthonormal_basis(directions)
        u = rng.random(n)
        phi = rng.uniform(0.0, 2.0 * math.pi, n)
        radial = rho * np.sqrt(u)
        points = (
            domain.center
            + (radial * np.cos(phi))[:, None] * e1
            + (radial * np.sin(phi))[:, None] * e2
        )
        radii = sizes.draw(self.radius, rng, n)  # one per line, independent of it
        lower, upper = domain.lower - r, domain.upper + r
        t0, t1 = _clip_lines(points, directions, lower, upper)
        keep = t0 <= t1
        a = points[keep] + t0[keep, None] * directions[keep]
        b = points[keep] + t1[keep, None] * directions[keep]
        segments = Segments(a, b, radii[keep], round_caps=True)
        micro = PrimitiveUnion(domain, segments=segments, model=self, seed=seed_info)
        spread = sizes.size_summary("radius", radii[keep], self.radius)
        if spread:
            micro.metadata["summary"] = {"fibres": int(np.sum(keep)), **spread}
        return micro

    # -- theory --------------------------------------------------------------------------

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        lv = self.lv
        q = math.exp(-lv * math.pi * self.radius.moment(2))
        return {
            "length_density": lv,
            "theta_jeulin": lv / math.pi,
            "volume_fraction": 1.0 - q,
            "specific_surface": 2.0 * math.pi * self.radius.moment(1) * lv * q,
            "integral_range": math.inf,
            "variance_exponent": 2.0 / 3.0,
        }

    def choquet_capacity(self, element: str, size, direction=None) -> np.ndarray:
        """``T(K) = P(K hits A)``: the lines of direction ``u`` hitting a convex body are
        those crossing its projection on the plane normal to ``u`` (on average ``L_V S / 4``
        isotropic lines, Cauchy), and those hitting the fibres hit ``A' (+) K'``: for balls
        of radius ``size`` a ball of radius R + r (any law of directions; ``E[(R + r)^2]``
        over that of the radii), for segments of length ``size`` along ``direction`` a
        spherocylinder of projection ``pi R^2 + 2 R l |sin g|`` (``g`` its angle to the
        line, ``E|sin g| = pi/4`` for isotropic lines; without a direction, the mean of the
        capacities over the directions; ``E[R^2]`` and ``E[R]``); pairs of points at
        distance ``size``: ``2 p - C(h)``."""
        size = np.asarray(size, dtype=float)
        lv, law = self.lv, self.radius
        r1, r2 = law.moment(1), law.moment(2)
        if element == "ball":
            nodes, weights = law.quadrature()
            dilated = sum(w * (x + size) ** 2 for x, w in zip(nodes, weights, strict=True))
            return -np.expm1(-lv * math.pi * dilated)
        if element == "segment":
            if isinstance(self.orientation, Isotropic):
                return -np.expm1(-lv * (math.pi * r2 + 0.5 * math.pi * r1 * size))
            if direction is None:
                rule = zip(*half_sphere(6), strict=True)
                return sum(w * self.choquet_capacity(element, size, u) for u, w in rule)
            b, w = angle_weights(self.orientation, direction)
            sine = float(np.sum(w * np.sin(b)))
            return -np.expm1(-lv * (math.pi * r2 + 2.0 * r1 * size * sine))
        if element == "pair":
            p = -math.expm1(-lv * math.pi * r2)
            return 2.0 * p - self.covariance(size, direction)
        raise ValueError('element: "ball", "segment" or "pair"')

    def expected_fibers(self, domain: Domain) -> float:
        """Mean number of fibre axes hitting the box, ``L_V`` times the mean area of its
        projection along the lines, ``sum_k A_k E|u_k|`` (``A_k`` the area of the faces
        normal to the axis ``k``): ``L_V S / 4`` for isotropic lines (Dirrenberger 2014,
        eq. 9)."""
        if isinstance(self.orientation, Isotropic):
            return self.lv * domain.surface_area / 4.0
        u, w = self.orientation.quadrature()
        lx, ly, lz = domain.size
        mean = w @ np.abs(u.reshape(-1, 3))  # E|u_k|
        return self.lv * float(ly * lz * mean[0] + lx * lz * mean[1] + lx * ly * mean[2])

    def expected_count(self, domain: Domain) -> float:
        return self.expected_fibers(domain)

    def covariance(self, h, direction=None) -> np.ndarray:
        """C(h) = P(x in A, x + h in A), for ``h`` along ``direction`` (any for isotropic
        lines; averaged over the directions of ``h`` without one).

        The lines of direction ``u`` hitting {x, x + h} dilated by the fibre radius are
        those crossing the union of two disks of radius R, ``|h| sin(b)`` apart, in the
        plane normal to ``u`` (``b`` the angle between ``h`` and ``u``): ``P(x, x + h not in
        A) = q^2 exp(L_V E_u[lens(|h| sin b)])``, the mean over the law of the directions
        (cos b uniform for isotropic lines; :func:`morphogen.theory.oriented_covariance`)
        and over that of the radii (``q = exp(-L_V pi E[R^2])``).
        """
        lv, law = self.lv, self.radius
        q = math.exp(-lv * math.pi * law.moment(2))
        nodes, weights = law.quadrature(64)

        def lens(dist, b):
            return sum(w * theory.disk_intersection_area(dist * np.sin(b), x)
                       for x, w in zip(nodes, weights, strict=True))  # fmt: skip

        return theory.oriented_covariance(q, lv, lens, h, self.orientation, direction)


def _check_lumen(lumen: float) -> None:
    if not 0.0 <= lumen < 1.0:
        raise ValueError("lumen: the inner radius over the radius, in [0, 1)")


def _fibres(domain: Domain, segments: Segments, lumen: float, model, seed_info, summary=None):
    """The union of the fibres, or, for hollow ones, of their walls: the union of the
    fibres minus that of their lumens (exact without overlaps; a lumen crossing another
    fibre opens its wall there); ``summary``, that of the model, in the summary of both."""
    outer = PrimitiveUnion(domain, segments=_crop(segments, domain), model=model, seed=seed_info)
    if lumen <= 0.0:
        micro = outer
    else:
        inner = Segments(segments.start, segments.end, segments.radii * lumen, segments.round_caps)
        lumens = PrimitiveUnion(domain, segments=_crop(inner, domain), model=model, seed=seed_info)
        combined = Combined if summary is None else _Bonded
        micro = combined(domain, "difference", [outer, lumens], model=model, seed=seed_info)
    if summary is not None:
        micro.metadata["summary"] = summary
    return micro


def _crop(segments: Segments, domain: Domain) -> Segments:
    """The segments that reach a non-periodic domain."""
    if domain.periodic:
        return segments
    lo, hi = segments.bounds()
    return segments.subset(np.all((hi >= domain.lower) & (lo <= domain.upper), axis=1))


@register
@dataclass(frozen=True)
class HardcoreFibers(Model):
    """Non-overlapping straight fibres (spherocylinders) placed by random sequential
    addition (RSA).

    Fibres are added one at a time, with uniformly random centres and axes drawn from
    ``orientation``; a trial is rejected when the fibre would come closer than ``min_gap``
    to one already placed. The process jams at a volume fraction that falls as the fibres
    lengthen (isotropic fibres) and rises when they are aligned.

    Generation always takes place in a periodic box; for non-periodic domains the box is
    enlarged by a fibre length and cropped, which avoids wall effects.
    """

    name = "hardcore_fibers"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "fibres"}
    count_label = "fibres"

    radius: Distribution = param(
        Constant(0.5), doc="Fibre radius", kind="distribution", unit="length"
    )
    length: Distribution = param(
        Constant(5.0),
        doc="Length of the cylindrical part of the fibres (round caps)",
        kind="distribution",
        unit="length",
    )
    orientation: OrientationDistribution = param(
        Isotropic(),
        doc="Distribution of the fibre axes (isotropic, aligned, planar)",
        kind="orientation",
    )
    volume_fraction: float | None = param(None, doc="Target volume fraction", kind="fraction")
    number: int | None = param(None, doc="Number of fibres in the generation box", kind="int")
    min_gap: float = param(0.0, doc="Minimal distance between fibre surfaces", unit="length")
    max_attempts: int | None = param(
        None, doc="Maximal number of trials (default: 10000 per fibre)", kind="int"
    )
    lumen: float = param(
        0.0,
        doc="Hollow fibres: inner radius of their lumens over their radius (0: solid fibres; "
        "the lumens belong to phase 0)",
        kind="fraction",
    )

    def _validate(self) -> None:
        exactly_one("volume_fraction", self.volume_fraction, "number", self.number)
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        elif self.number < 0:
            raise ValueError("number: must be non-negative")
        check_positive(self.min_gap, "min_gap", allow_zero=True)
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("max_attempts: must be positive")
        _check_lumen(self.lumen)

    @property
    def mean_volume(self) -> float:
        return math.pi * self.radius.moment(2) * self.length.moment(1) + (
            4.0 / 3.0 * math.pi * self.radius.moment(3)
        )

    @property
    def mean_surface(self) -> float:
        return 2.0 * math.pi * self.radius.moment(1) * self.length.moment(1) + (
            4.0 * math.pi * self.radius.moment(2)
        )

    def _draw(self, rng, box_volume: float) -> tuple[np.ndarray, np.ndarray]:
        """Radii and lengths of the fibres to place: the given number, or as many as make
        the total volume closest to the target."""
        if self.number is not None:
            return self.radius.sample(rng, self.number), self.length.sample(rng, self.number)
        target = self.volume_fraction * box_volume
        radii, lengths = np.empty(0), np.empty(0)
        while True:
            m = int(1.1 * target / self.mean_volume) + 16
            radii = np.concatenate([radii, self.radius.sample(rng, m)])
            lengths = np.concatenate([lengths, self.length.sample(rng, m)])
            cum = np.cumsum(math.pi * radii**2 * lengths + 4.0 / 3.0 * math.pi * radii**3)
            if cum[-1] >= target:
                break
        k = int(np.searchsorted(cum, target))
        n = k + 1 if k == 0 or cum[k] - target <= target - cum[k - 1] else k
        return radii[:n], lengths[:n]

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        rng, seed_info = make_rng(seed)
        reach = self.length.upper() + 2.0 * self.radius.upper() + self.min_gap
        if domain.periodic:
            if reach >= min(domain.size) / 2.0:
                raise ValueError(
                    f"hardcore_fibers: fibres (reach {reach:g}) must be shorter than half the "
                    f"periodic box (edge {min(domain.size):g})"
                )
            box = domain
        else:
            box = domain.dilated(reach)
        radii, lengths = self._draw(rng, box.volume)
        max_attempts = self.max_attempts or 10_000 * max(len(radii), 1)
        local, axes = random_sequential_addition_fibres(
            radii, lengths, box.lengths, self.min_gap, rng, self.orientation, max_attempts
        )
        centres = box.lower + local
        half = 0.5 * lengths[:, None] * axes
        segments = Segments(centres - half, centres + half, radii, round_caps=True)
        return _fibres(domain, segments, self.lumen, self, seed_info)

    def expected_count(self, domain: Domain) -> float | None:
        """Mean number of fibre centres in the domain (given a volume fraction)."""
        if self.volume_fraction is None:
            return None
        return self.volume_fraction * domain.volume / self.mean_volume

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        if self.volume_fraction is not None:
            p = self.volume_fraction
        elif domain is not None and domain.periodic:
            p = self.number * self.mean_volume / domain.volume
        else:
            return {}
        # no overlaps: the fraction and the surface are those of the fibres
        out = {
            "volume_fraction": p,
            "specific_surface": p * self.mean_surface / self.mean_volume,
            "intensity": p / self.mean_volume,
        }
        if self.lumen > 0.0:  # the walls of the hollow fibres (volume_fraction: with lumens)
            lam = self.lumen
            r2, r3, ell = self.radius.moment(2), self.radius.moment(3), self.length.moment(1)
            wall = math.pi * (1.0 - lam**2) * r2 * ell + 4.0 / 3.0 * math.pi * (1.0 - lam**3) * r3
            out["solid_fraction"] = p * wall / self.mean_volume
        return out


def _inverse_langevin(c: float) -> float:
    """The concentration ``kappa`` of the von Mises-Fisher distribution on the sphere
    whose mean cosine, ``coth(kappa) - 1/kappa``, is ``c`` (0 < c < 1)."""
    k = c * (3.0 - c * c) / (1.0 - c * c)  # Cohen's approximation, then Newton
    for _ in range(50):
        if k > 30.0:
            f, df = 1.0 - 1.0 / k - c, 1.0 / (k * k)
        else:
            f = 1.0 / math.tanh(k) - 1.0 / k - c
            df = 1.0 / (k * k) - 1.0 / math.sinh(k) ** 2
        step = f / df
        k = max(k - step, 0.5 * k)
        if abs(step) <= 1e-14 * k:
            break
    return k


SHAPES = ("worm_like", "arc", "helix")


@register
@dataclass(frozen=True)
class CurvedFibers(Model):
    """Curved fibres: chains of ``segments`` straight spherocylinders along curved axes,
    a Boolean model or non-overlapping fibres.

    ``shape`` sets the axes. ``worm_like``: the direction performs a persistent random
    walk, each segment of length ``ds = length / segments`` turning from the previous one
    by an angle whose cosine follows the von Mises-Fisher distribution of mean
    ``exp(-ds / persistence)``, about a uniformly random perpendicular axis, so that the
    tangents decorrelate as ``exp(-s / persistence)`` along the contour (the discrete
    worm-like chain); the first direction follows ``orientation``. ``arc``: circular arcs
    of radius ``curvature_radius`` (a distribution) bent in a uniformly random plane, the
    tangent at the middle of the contour following ``orientation`` (the curved fibres of
    PuMA). ``helix``: crimped fibres, helices of radius ``helix_radius`` and pitch
    ``helix_pitch`` about straight axes following ``orientation``, of random phase. The
    segments have round caps, so that the joints are smooth.

    ``overlap: true``: a Boolean model, the centres of the fibres (the middles of their
    contours) a Poisson point process. ``overlap: false``: fibres placed one at a time
    by random sequential addition, a trial rejected when it comes closer than
    ``min_gap`` to a fibre already placed (generated in a periodic box, enlarged and
    cropped for non-periodic domains). The radius is a number (monodisperse fibres) or a
    distribution (polydisperse: one radius per fibre, the moments of its law in the
    volumes and fractions of the theory).
    """

    name = "curved_fibers"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "fibres"}
    count_label = "fibres"

    radius: Distribution = param(
        Constant(0.5),
        doc="Fibre radius: a number (monodisperse) or a distribution (polydisperse, one "
        "radius per fibre)",
        kind="distribution",
        unit="length",
    )
    length: float = param(20.0, doc="Contour length of the fibres", unit="length")
    persistence: float = param(
        10.0,
        doc="Persistence length of worm-like chains: the tangents decorrelate as "
        "exp(-s / persistence)",
        unit="length",
    )
    segments: int = param(10, doc="Straight segments per fibre", kind="int")
    orientation: OrientationDistribution = param(
        Isotropic(),
        doc="Distribution of the first direction (worm-like chains), the middle tangent "
        "(arcs) or the axis (helices) of the fibres",
        kind="orientation",
    )
    volume_fraction: float | None = param(None, doc="Volume fraction of fibres", kind="fraction")
    intensity: float | None = param(None, doc="Fibre centres per unit volume", unit="1/length^3")
    shape: str = param("worm_like", doc="Axes of the fibres: worm_like, arc or helix", kind="str")
    curvature_radius: Distribution = param(
        Constant(20.0),
        doc="Radius of curvature of the arcs",
        kind="distribution",
        unit="length",
    )
    helix_radius: float = param(1.0, doc="Radius of the helices (crimp amplitude)", unit="length")
    helix_pitch: float = param(
        10.0, doc="Pitch of the helices: their advance along the axis per turn", unit="length"
    )
    overlap: bool = param(
        True,
        doc="Boolean model; false: non-overlapping fibres by random sequential addition",
        kind="bool",
    )
    min_gap: float = param(
        0.0, doc="Minimal distance between the surfaces of non-overlapping fibres", unit="length"
    )
    max_attempts: int | None = param(
        None,
        doc="Maximal number of trials of non-overlapping fibres (default 10000 each)",
        kind="int",
    )
    lumen: float = param(
        0.0,
        doc="Hollow fibres: inner radius of their lumens over their radius (0: solid fibres; "
        "the lumens belong to phase 0)",
        kind="fraction",
    )

    def _validate(self) -> None:
        check_positive(self.length, "length")
        check_positive(self.persistence, "persistence")
        if self.segments < 1:
            raise ValueError("segments: at least one per fibre")
        exactly_one("volume_fraction", self.volume_fraction, "intensity", self.intensity)
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        else:
            check_positive(self.intensity, "intensity")
        if self.shape not in SHAPES:
            raise ValueError(f"shape: one of {', '.join(SHAPES)}")
        check_positive(self.helix_radius, "helix_radius", allow_zero=True)
        check_positive(self.helix_pitch, "helix_pitch")
        check_positive(self.min_gap, "min_gap", allow_zero=True)
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("max_attempts: must be positive")
        _check_lumen(self.lumen)

    @property
    def mean_grain_volume(self) -> float:
        """Mean volume of a straight fibre, ``pi E[R^2] L + 4/3 pi E[R^3]``: the union of
        the segments of a curved one is a little smaller (their caps overlap more at the
        bends)."""
        r2, r3 = self.radius.moment(2), self.radius.moment(3)
        return math.pi * r2 * self.length + 4.0 / 3.0 * math.pi * r3

    @property
    def theta(self) -> float:
        if self.intensity is not None:
            return self.intensity
        if not self.overlap:
            return self.volume_fraction / self.mean_grain_volume
        return theory.boolean_intensity(self.volume_fraction, self.mean_grain_volume)

    @property
    def correlation(self) -> float:
        """Mean cosine of the angle between consecutive segments (worm-like chains)."""
        return math.exp(-self.length / self.segments / self.persistence)

    def tangents(self, rng, n: int) -> np.ndarray:
        """Directions (n x segments x 3) of the segments of ``n`` worm-like chains."""
        m = self.segments
        out = np.empty((n, m, 3))
        if n == 0:
            return out
        out[:, 0] = self.orientation.sample(rng, n)
        c = self.correlation
        kappa = _inverse_langevin(c) if c > 1e-12 else 0.0
        for k in range(1, m):
            t = out[:, k - 1]
            u = rng.random(n)
            if kappa < 1e-8:
                w = 2.0 * u - 1.0
            else:
                w = 1.0 + np.log(u + (1.0 - u) * math.exp(-2.0 * kappa)) / kappa
            phi = rng.uniform(0.0, 2.0 * math.pi, n)
            e1, e2 = orthonormal_basis(t)
            sin = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
            turn = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
            nxt = w[:, None] * t + sin[:, None] * turn
            out[:, k] = nxt / np.linalg.norm(nxt, axis=1)[:, None]
        return out

    def axes(self, rng, n: int) -> np.ndarray:
        """Points (n x (segments + 1) x 3) of the axes of ``n`` fibres, relative to the
        middles of their contours."""
        m = self.segments
        if self.shape == "worm_like":
            points = np.zeros((n, m + 1, 3))
            points[:, 1:] = np.cumsum(self.tangents(rng, n) * (self.length / m), axis=1)
        else:
            s = (np.arange(m + 1) / m - 0.5) * self.length  # arc length from the middle
            t = self.orientation.sample(rng, n).reshape(n, 3)
            e1, e2 = orthonormal_basis(t)
            phi = rng.uniform(0.0, 2.0 * math.pi, n)[:, None]
            u = np.cos(phi) * e1 + np.sin(phi) * e2
            if self.shape == "arc":
                rho = self.curvature_radius.sample(rng, n)
                if np.any(rho <= 0.0):
                    raise ValueError("curvature_radius: positive radii")
                a = s[None, :] / rho[:, None]
                points = rho[:, None, None] * (
                    np.sin(a)[..., None] * t[:, None, :]
                    + (1.0 - np.cos(a))[..., None] * u[:, None, :]
                )
            else:
                w = np.cross(t, u)
                c = self.helix_pitch / (2.0 * math.pi)
                a = s / math.hypot(self.helix_radius, c)  # turning angle from the middle
                points = (
                    self.helix_radius
                    * (
                        np.cos(a)[None, :, None] * u[:, None, :]
                        + np.sin(a)[None, :, None] * w[:, None, :]
                    )
                    + c * a[None, :, None] * t[:, None, :]
                )
        if m % 2 == 0:
            middle = points[:, m // 2]
        else:
            middle = 0.5 * (points[:, m // 2] + points[:, m // 2 + 1])
        return points - middle[:, None, :]

    def _packed(self, domain: Domain, rng) -> tuple[np.ndarray, np.ndarray]:
        """Axes and radii of non-overlapping fibres placed by random sequential
        addition."""
        reach = self.length + 2.0 * self.radius.upper() + self.min_gap
        if domain.periodic:
            if reach >= min(domain.size) / 2.0:
                raise ValueError(
                    f"curved_fibers: fibres (reach {reach:g}) must be shorter than half the "
                    f"periodic box (edge {min(domain.size):g})"
                )
            box = domain
        else:
            box = domain.dilated(reach)
        n = round(self.theta * box.volume)
        offsets = self.axes(rng, n)
        radii = sizes.draw(self.radius, rng, n)
        max_attempts = self.max_attempts or 10_000 * max(n, 1)
        centres = random_sequential_addition_polylines(
            offsets, radii, box.lengths, self.min_gap, rng, max_attempts
        )
        return box.lower + centres[:, None, :] + offsets, radii

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        rng, seed_info = make_rng(seed)
        if self.overlap:
            reach = 0.5 * self.length + self.radius.upper()
            window = _germ_window(domain, reach, self.name)
            n = rng.poisson(self.theta * window.volume)
            centres = window.lower + rng.random((n, 3)) * window.lengths
            points = self.axes(rng, n) + centres[:, None, :]
            radii = sizes.draw(self.radius, rng, n)
        else:
            points, radii = self._packed(domain, rng)
        m = self.segments
        segments = Segments(points[:, :-1].reshape(-1, 3), points[:, 1:].reshape(-1, 3),
                            np.repeat(radii, m))  # fmt: skip
        summary = None
        if sizes.polydisperse(self.radius):  # the radii of the fibres that reach the domain
            near = np.ones(len(radii), dtype=bool)
            if not domain.periodic:
                lo, hi = segments.bounds()
                inside = np.all((hi >= domain.lower) & (lo <= domain.upper), axis=1)
                near = inside.reshape(-1, m).any(axis=1)
            spread = sizes.size_summary("radius", radii[near], self.radius)
            summary = {"fibres": int(near.sum()), **spread}
        return _fibres(domain, segments, self.lumen, self, seed_info, summary)

    def expected_count(self, domain: Domain) -> float:
        """Mean number of fibre centres in the domain."""
        return self.theta * domain.volume

    def mean_square_end_to_end(self) -> float:
        """Mean square distance between the ends of the axis of a fibre: for worm-like
        chains ds^2 (m + 2 sum_d (m - d) c^d); the chords of the arcs (exact for their
        polylines); the axial advance and the radial chord of the helices."""
        m, length = self.segments, self.length
        ds = length / m
        if self.shape == "worm_like":
            c = self.correlation
            d = np.arange(1, m)
            return ds * ds * (m + 2.0 * float(np.sum((m - d) * c**d)))
        if self.shape == "arc":
            nodes, weights = self.curvature_radius.quadrature(96)
            chord = 2.0 * nodes * np.sin(0.5 * length / nodes)
            return float(np.sum(weights * chord**2))
        c = self.helix_pitch / (2.0 * math.pi)
        a = length / math.hypot(self.helix_radius, c)
        return (c * a) ** 2 + (2.0 * self.helix_radius * math.sin(0.5 * a)) ** 2

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        theta = self.theta
        if self.overlap:
            # exact for straight fibres; a little above that of curved ones (overlaps)
            fraction = 1.0 - math.exp(-theta * self.mean_grain_volume)
        else:
            fraction = theta * self.mean_grain_volume  # a little above (joints)
        out = {
            "intensity": theta,
            "volume_fraction": fraction,
            "length_density": theta * self.length,
            "mean_square_end_to_end": self.mean_square_end_to_end(),
        }
        if self.lumen > 0.0 and not self.overlap:  # the walls of the hollow fibres
            lam, r2, r3 = self.lumen, self.radius.moment(2), self.radius.moment(3)
            wall = math.pi * (1.0 - lam**2) * r2 * self.length
            wall += 4.0 / 3.0 * math.pi * (1.0 - lam**3) * r3
            out["solid_fraction"] = theta * wall
        return out


def von_mises_fisher(rng, mean, kappa) -> np.ndarray:
    """Unit vectors drawn from von Mises-Fisher laws on the sphere, of densities
    ``kappa exp(kappa m.x) / (4 pi sinh kappa)``, one per row of ``mean`` (unit vectors
    ``m``) and entry of ``kappa`` (>= 0; 0 is uniform): the cosine ``w = m.x`` by inversion,
    ``w = 1 + ln(u + (1 - u) e^(-2 kappa)) / kappa``, the azimuth about ``m`` uniform."""
    mean = np.asarray(mean, dtype=float).reshape(-1, 3)
    n = len(mean)
    kappa = np.broadcast_to(np.asarray(kappa, dtype=float), (n,))
    u = 1.0 - rng.random(n)  # in (0, 1]
    flat = kappa < 1e-8
    k = np.where(flat, 1.0, kappa)
    w = np.where(flat, 2.0 * u - 1.0, 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * k)) / k)
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    e1, e2 = orthonormal_basis(mean)
    sin = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
    x = w[:, None] * mean + sin[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def multivariate_von_mises_fisher_walk(rng, main, kappa1: float, kappa2: float, steps: int):
    """Directions (n x steps x 3) of the random walks of Altendorf & Jeulin (2011): step
    ``k`` follows the multivariate von Mises-Fisher law of density proportional to
    ``exp(kappa1 mu.t + kappa2 t_(k-1).t)``, ``mu`` the main direction (a row of ``main``)
    and ``t_0 = mu``, i.e. the von Mises-Fisher law about ``kappa1 mu + kappa2 t_(k-1)`` of
    concentration its length."""
    main = np.asarray(main, dtype=float).reshape(-1, 3)
    out = np.empty((len(main), steps, 3))
    prev = main
    for k in range(steps):
        v = kappa1 * main + kappa2 * prev
        kappa = np.linalg.norm(v, axis=1)
        mean = np.where(kappa[:, None] > 0.0, v / np.maximum(kappa, 1e-300)[:, None], main)
        prev = out[:, k] = von_mises_fisher(rng, mean, kappa)
    return out


def _turn(vectors, a, b):
    """``vectors`` (n x 3) turned by the rotation taking the unit vector ``a`` to ``b``
    about their common normal (Rodrigues' formula)."""
    axis = np.cross(a, b)
    sin, cos = float(np.linalg.norm(axis)), float(np.dot(a, b))
    if sin < 1e-12:
        if cos > 0.0:
            return vectors
        axis, sin, cos = orthonormal_basis(a)[0][0], 0.0, -1.0  # half a turn
    else:
        axis = axis / sin
    return (
        vectors * cos + np.cross(axis, vectors) * sin + np.outer(vectors @ axis, axis) * (1.0 - cos)
    )


def _chain_volumes(x, first, radii) -> np.ndarray:
    """Volumes of chains of spherocylinders that overlap only at their joints: ``pi r^2 L
    + 4/3 pi r^3`` less ``2/3 r^3 (2 tan(d/2) - d)`` at a joint turning by ``d`` (the
    cylinders overlap inside the bend, the ball of the joint fills its outside)."""
    out = np.empty(len(radii))
    for f, r in enumerate(radii):
        t = np.diff(x[first[f] : first[f + 1]], axis=0)
        lengths = np.linalg.norm(t, axis=1)
        out[f] = math.pi * r * r * lengths.sum() + 4.0 / 3.0 * math.pi * r**3
        if len(t) > 1:
            u = t / np.maximum(lengths, 1e-300)[:, None]
            d = np.arccos(np.clip(np.einsum("ij,ij->i", u[:-1], u[1:]), -1.0, 1.0))
            out[f] -= 2.0 / 3.0 * r**3 * float(np.sum(2.0 * np.tan(0.5 * d) - d))
    return out


END_STEPS = ("repulsion", "shrink")


@register
@dataclass(frozen=True)
class RandomWalkFibers(Model):
    """Bending fibres packed without overlap by forces (Altendorf & Jeulin 2011).

    As in the papers (Phys. Rev. E 83, 041804; ISMM 2011): random walks, then a
    force-biased packing, to high volume fractions (0.51 for isotropic fibres of aspect
    ratio 11, 0.64 nearly aligned). A fibre of radius ``r`` (from ``radius``) and contour
    length ``L`` (from ``length``) is a chain of ``m = round(L / (step r))`` steps of
    length ``L / m``. Its main direction ``mu`` follows ``orientation`` (in the papers
    Schladitz's beta law, ``{type: beta, beta: 0.1}`` about z); each step follows the
    multivariate von Mises-Fisher law of density proportional to ``exp(kappa1 mu.t +
    kappa2 t'.t)``, ``t'`` the previous step (``mu`` for the first): ``kappa1``, the
    reliability to the main direction, holds the fibre to it, ``kappa2``, the reliability
    to the previous step, sets the smoothness of its bending. The chain is turned about
    its first node, uniform in the box, so that its end-to-end vector follows ``mu``; as
    many fibres as bring their total volume ``pi r^2 L + 4/3 pi r^3`` closest to
    ``volume_fraction`` (or ``number`` of them). As in Altendorf, Jeulin & Willot (Int. J.
    Solids Struct. 51, 3807, 2014): ``placement_trials > 1`` places the fibres one after
    another at the best of that many uniform positions, the one of least overlap with the
    fibres placed before it (the sum of the depths of the overlapping pairs of segments;
    :func:`morphogen.kernels.force_biased.place`), which shortens the packing (2.5 to 9
    times fewer steps for 30 trials, the more the looser the packing); the orthotropic
    laws of their glass fibres are mixtures of beta laws (``{type: mixture, laws: [...],
    weights: [...]}``).

    ``packing: true``: the chains are packed by the forces of the paper
    (:mod:`morphogen.kernels.force_biased`), repulsions of the overlapping segments plus
    ``recovery`` times the springs on their lengths and the open angular springs on the
    joints, until the total force falls below ``0.04 r`` per fibre and the largest overlap
    below ``0.1 r_min`` (else, after ``max_steps`` steps, the fraction is out of reach);
    the overlaps left are removed by steps of repulsions alone (``end_step: repulsion``,
    the radii kept but for a residue) or by reducing the radius of each fibre by half its
    largest overlap (``shrink``, the paper's end step). ``packing: false``: the
    overlapping walks, a Boolean model whose ``volume_fraction`` is that of the union.

    ``interaction_distance > 0``: the contact packing of Keilmann, Redenbach & Willot
    (Comput. Mater. Sci. 264, 114458, 2026), which raises the number of contacts. After
    the packing, the pairs of segments of different fibres whose surfaces are closer than
    ``contact_distance + interaction_distance`` are candidates, shortlisted so that a
    segment keeps only its closest candidate of each other fibre; the packing runs again
    with the contact forces, each shortlisted pair drawn together by ``contact_weight``
    times half its excess over ``contact_distance`` (smoothed by the friction ramp over
    ``contact_distance``), until the total force changes by less than 0.001 % over a step
    (1000 steps at most); the pairs that leave a force above ``contact_threshold`` times
    the mean radius on one of their nodes (conflicting forces) are released and the
    packing run once more, before the end step. The summary counts the contacts after
    the packing (and before the contact packing, ``*_before``): the pairs of fibres closer
    than ``contact_distance`` (plus 1 % of the smallest radius), their mean number per
    fibre, and the clots, connected components of the segments in contact (consecutive
    ones of a fibre joined) that hold more than two fibres; the theory gives Toll's
    estimate for straight fibres of the same lengths, the contacts per fibre of their
    Boolean model.

    Choices (the papers leave them open or differ): chains of spherocylinders instead of
    balls ``r/2`` apart (smooth fibres, any ``step``, no overlap meaning axes at least
    ``r_a + r_b + min_gap`` apart), the push of a segment shared by its nodes with the
    barycentric weights of its closest point; one radius per fibre; segments of a fibre
    at most ``ceil(2.5 / step)`` steps apart exempt (the paper's five balls); the moves
    capped to a quarter of the shortest segment or radius, a segment stretched beyond
    twice its rest length shortened to it after a step; a joint folded beyond its
    chord moved towards the middle of the chord; the paper's tolerance ``0.002 s`` per
    fibre (window ``s = 20 r``) made ``0.04 r``. The realized fraction (``volume_fraction``
    of the summary) departs from the target by the change of the contour lengths, which
    the springs leave free within 5 %. Generated in a periodic box, which fibres may
    outgrow (they bend around their images); non-periodic domains are cut from a periodic
    box enlarged by half the reach of a fibre on every side.
    """

    name = "random_walk_fibers"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "fibres"}
    count_label = "fibres"

    radius: Distribution = param(
        Constant(0.5), doc="Fibre radius (one per fibre)", kind="distribution", unit="length"
    )
    length: Distribution = param(
        Constant(10.0),
        doc="Contour length of the fibre axes (round caps)",
        kind="distribution",
        unit="length",
    )
    orientation: OrientationDistribution = param(
        Isotropic(),
        doc="Distribution of the main directions of the fibres (isotropic, aligned, planar, "
        "{type: beta, beta: 0.1, axis: [0, 0, 1]}, Schladitz's beta law, or a mixture of "
        "laws, {type: mixture, laws: [...], weights: [...]})",
        kind="orientation",
    )
    kappa1: float = param(
        10.0, doc="Reliability of the steps to the main direction of their fibre (>= 0)"
    )
    kappa2: float = param(
        100.0, doc="Reliability of the steps to the previous one: smoothness of the bending"
    )
    step: float = param(
        0.5, doc="Length of the steps of the walks (segments) over the fibre radius (paper: 0.5)"
    )
    volume_fraction: float | None = param(None, doc="Target volume fraction", kind="fraction")
    number: int | None = param(None, doc="Number of fibres in the generation box", kind="int")
    packing: bool = param(
        True, doc="Force-biased packing without overlaps; false: the overlapping walks", kind="bool"
    )
    recovery: float = param(
        0.2,
        doc="Weight of the recovery forces (segment lengths, joint angles) against the "
        "repulsions (epsilon of the paper)",
        kind="fraction",
    )
    max_steps: int = param(10_000, doc="Maximal number of packing steps", kind="int")
    end_step: str = param(
        "repulsion",
        doc="Removal of the last overlaps: repulsion (repulsions alone, radii kept) or "
        "shrink (each radius reduced by half the largest overlap of its fibre)",
        kind="str",
    )
    min_gap: float = param(0.0, doc="Minimal distance between fibre surfaces", unit="length")
    lumen: float = param(
        0.0,
        doc="Hollow fibres: inner radius of their lumens over their radius (0: solid fibres; "
        "the lumens belong to phase 0)",
        kind="fraction",
    )
    interaction_distance: float = param(
        0.0,
        doc="Contact packing (Keilmann et al. 2026): fibres closer than this (between "
        "surfaces) after the packing are drawn into contact (0: none)",
        unit="length",
    )
    contact_distance: float = param(
        0.0, doc="Distance between surfaces below which fibres are in contact", unit="length"
    )
    contact_weight: float = param(
        0.5, doc="Weight of the contact forces (paper: 0.5)", kind="fraction"
    )
    contact_threshold: float = param(
        0.05,
        doc="Contact pairs left with a force above this times the mean radius on a node are "
        "released (paper: 0.1 voxel for a radius of 2 voxels)",
    )
    placement_trials: int = param(
        1,
        doc="Initial placement (Altendorf, Jeulin & Willot 2014): each fibre at the best of "
        "this many uniform positions, the one of least overlap with the fibres placed "
        "before it (1: uniform positions; more with packing only)",
        kind="int",
    )

    def _validate(self) -> None:
        exactly_one("volume_fraction", self.volume_fraction, "number", self.number)
        if not 1 <= self.placement_trials <= 10_000:
            raise ValueError("placement_trials: from 1 (uniform positions) to 10000")
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        elif self.number < 0:
            raise ValueError("number: must be non-negative")
        check_positive(self.interaction_distance, "interaction_distance", allow_zero=True)
        check_positive(self.contact_distance, "contact_distance", allow_zero=True)
        check_fraction(self.contact_weight, "contact_weight", open_interval=False)
        check_positive(self.contact_threshold, "contact_threshold")
        if self.interaction_distance > 0.0 and not self.packing:
            raise ValueError("interaction_distance: the contact packing needs packing: true")
        if self.placement_trials > 1 and not self.packing:  # it would thin their overlaps
            raise ValueError(
                "placement_trials: the placement of least overlap needs packing: true "
                "(unpacked fibres overlap as a Boolean model, its volume fraction that of "
                "uniform positions)"
            )
        check_positive(self.kappa1, "kappa1", allow_zero=True)
        check_positive(self.kappa2, "kappa2", allow_zero=True)
        check_positive(self.step, "step")
        check_fraction(self.recovery, "recovery", open_interval=False)
        if self.max_steps < 0:
            raise ValueError("max_steps: must be non-negative")
        if self.end_step not in END_STEPS:
            raise ValueError(f"end_step: one of {', '.join(END_STEPS)}")
        check_positive(self.min_gap, "min_gap", allow_zero=True)
        _check_lumen(self.lumen)

    mean_volume = HardcoreFibers.mean_volume
    mean_surface = HardcoreFibers.mean_surface

    def _draw(self, rng, box_volume: float) -> tuple[np.ndarray, np.ndarray]:
        """Radii and lengths: the given number of fibres, or as many as bring their total
        volume closest to the target, ``p V`` (packed) or ``-ln(1 - p) V`` (overlapping)."""
        if self.number is not None:
            return self.radius.sample(rng, self.number), self.length.sample(rng, self.number)
        p = self.volume_fraction
        target = (p if self.packing else -math.log1p(-p)) * box_volume
        radii, lengths = np.empty(0), np.empty(0)
        while True:
            m = int(1.1 * target / self.mean_volume) + 16
            radii = np.concatenate([radii, self.radius.sample(rng, m)])
            lengths = np.concatenate([lengths, self.length.sample(rng, m)])
            cum = np.cumsum(math.pi * radii**2 * lengths + 4.0 / 3.0 * math.pi * radii**3)
            if cum[-1] >= target:
                break
        k = int(np.searchsorted(cum, target))
        n = k + 1 if k == 0 or cum[k] - target <= target - cum[k - 1] else k
        return radii[:n], lengths[:n]

    def walks(self, rng, radii, lengths) -> tuple[np.ndarray, np.ndarray]:
        """The nodes of the walks of fibres of the given radii and contour lengths (from
        their first nodes), all chains one after another, and the index of the first node
        of each chain (then the number of nodes)."""
        radii, lengths = np.asarray(radii, dtype=float), np.asarray(lengths, dtype=float)
        if np.any(radii <= 0.0) or np.any(lengths <= 0.0):
            raise ValueError("random_walk_fibers: positive radii and lengths")
        n = len(radii)
        m = np.maximum(1, np.rint(lengths / (self.step * radii))).astype(np.int64)
        first = np.zeros(n + 1, dtype=np.int64)
        first[1:] = np.cumsum(m + 1)
        nodes = np.zeros((int(first[-1]), 3))
        if n == 0:
            return nodes, first
        main = self.orientation.sample(rng, n).reshape(n, 3)
        steps = multivariate_von_mises_fisher_walk(rng, main, self.kappa1, self.kappa2, m.max())
        for f in range(n):
            t = steps[f, : m[f]] * (lengths[f] / m[f])
            chord = t.sum(axis=0)
            norm = float(np.linalg.norm(chord))
            if norm > 0.0:  # the end-to-end vector along the main direction
                t = _turn(t, chord / norm, main[f])
            nodes[first[f] + 1 : first[f + 1]] = np.cumsum(t, axis=0)
        return nodes, first

    def _pack(self, x, first, radii, box: Domain) -> tuple[np.ndarray, dict]:
        """Force-biased packing of the chains ``x`` (in ``[0, box)``, moved in place): the
        radii (less half the largest overlap left of each fibre) and the statistics."""
        from morphogen.kernels.force_biased import overlaps, pack

        n = len(radii)
        m = np.diff(first) - 1
        seg_fibre = np.repeat(np.arange(n), m)
        seg_index = np.arange(len(seg_fibre)) - np.repeat(first[:-1] - np.arange(n), m)
        seg_node = seg_index + first[seg_fibre]
        rest = np.linalg.norm(x[seg_node + 1] - x[seg_node], axis=1)
        e1, e2 = x[:-2] - x[1:-1], x[2:] - x[1:-1]
        cos = np.einsum("ij,ij->i", e1, e2)
        cos /= np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-300)
        rest_angle = np.full(len(x), math.pi)  # those of the ends of the chains: unused
        rest_angle[1:-1] = np.arccos(np.clip(cos, -1.0, 1.0))
        related = np.full(n, math.ceil(2.5 / self.step - 1e-9), dtype=np.int64)
        lengths = np.asarray(box.lengths, dtype=float)
        gap, r_min = float(self.min_gap), float(radii.min())
        if 2.0 * (2.0 * radii.max() + gap + 1.5 * rest.max()) >= lengths.min():
            raise ValueError(
                f"random_walk_fibers: the box (edge {lengths.min():g}) must exceed twice the "
                "reach of two segments, 2 (2 r + min_gap + 1.5 step r)"
            )
        args = (first, seg_node, seg_fibre, seg_index, radii, related, rest, rest_angle,
                lengths, gap, float(self.recovery), 0.25 * min(r_min, rest.min()))  # fmt: skip
        tol = 0.04 * float(radii.mean()) * n
        steps, force, largest = pack(x, *args, tol, 0.1 * r_min, int(self.max_steps), False)
        if largest > 0.1 * r_min:
            raise RuntimeError(
                f"random_walk_fibers: overlaps of {largest:.3g} left after {steps} packing "
                f"steps, above 0.1 r_min = {0.1 * r_min:.3g}; lower the volume fraction or "
                "raise max_steps"
            )
        stats = {}
        if self.interaction_distance > 0.0:
            stats = {f"{k}_before": v for k, v in self._contacts(x, args, radii).items()}
            largest, more = self._contact_packing(x, args, radii)
            stats.update(more)
        end = 0
        if self.end_step == "repulsion" and largest > 0.0:
            end, _, largest = pack(x, *args, 0.0, 1e-9 * r_min, max(self.max_steps, 1000), True)
        worst = np.zeros(n)
        overlaps(x, seg_node, seg_fibre, seg_index, radii, related, lengths, gap,
                 np.zeros((0, 3)), worst)  # fmt: skip
        shrunk = radii - 0.5 * worst  # then r_i + r_j + gap <= the distance of any two
        stats.update(self._contacts(x, args, shrunk))  # those of the fibres returned
        return shrunk, {
            "packing_steps": int(steps),
            "packing_force": float(force),
            "end_steps": int(end),
            "radius_reduction": float(np.max(0.5 * worst / radii)),
            "volume_fraction": float(np.sum(_chain_volumes(x, first, shrunk)) / box.volume),
            **stats,
        }

    def _contact_packing(self, x, args, radii) -> tuple[float, dict]:
        """The contact packing of Keilmann et al. (2026) of the packed chains ``x`` (moved
        in place): the largest overlap left and the statistics."""
        from morphogen.kernels.force_biased import near_pairs, pack_contacts

        seg_node, seg_fibre, lengths, gap = args[1], args[2], args[8], args[9]
        eps = float(self.contact_distance)
        reach = gap + eps + float(self.interaction_distance)
        s, t, d = near_pairs(x, seg_node, seg_fibre, radii, lengths, reach)
        keep = _shortlist(s, t, d, seg_fibre, len(radii))
        s, t = s[keep], t[keep]
        lim = radii[seg_fibre[s]] + radii[seg_fibre[t]] + gap + eps
        active = np.ones(len(s), dtype=np.bool_)
        force = np.zeros(len(x))
        run = (s, t, active, lim, eps, float(self.contact_weight), 1e-5, 1000, force)
        steps, _, largest = pack_contacts(x, *args, *run)
        a, b = seg_node[s], seg_node[t]
        loads = np.max(np.stack([force[a], force[a + 1], force[b], force[b + 1]]), axis=0)
        stuck = loads > self.contact_threshold * float(radii.mean())
        if np.any(stuck):  # conflicting forces: those pairs released, the packing run again
            active[stuck] = False
            more, _, largest = pack_contacts(x, *args, *run)
            steps += more
        return largest, {
            "contact_pairs": len(s),
            "released_pairs": int(np.sum(stuck)),
            "contact_steps": int(steps),
        }

    def _contacts(self, x, args, radii) -> dict:
        """Contacts of the chains: the pairs of fibres of which two segments are closer
        than ``r_a + r_b + min_gap + contact_distance`` (plus 1 % of the smallest radius),
        their number and mean number per fibre, and the clots of Keilmann et al. (2026,
        sec. 3.1), the connected components of the segments in contact, consecutive ones
        of a fibre joined, that hold more than two fibres."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        from morphogen.kernels.force_biased import near_pairs

        seg_node, seg_fibre, lengths, gap = args[1], args[2], args[8], args[9]
        n_seg = len(seg_node)
        tol = gap + float(self.contact_distance) + 0.01 * float(radii.min())
        s, t, _ = near_pairs(x, seg_node, seg_fibre, radii, lengths, tol)
        touching = np.zeros(n_seg, dtype=bool)
        touching[s] = touching[t] = True
        k = np.flatnonzero((seg_fibre[1:] == seg_fibre[:-1]) & touching[1:] & touching[:-1])
        rows, cols = np.concatenate([s, k]), np.concatenate([t, k + 1])
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_seg, n_seg))
        label = connected_components(graph, directed=False)[1]
        ids = np.flatnonzero(touching)
        members = np.unique(np.stack([label[ids], seg_fibre[ids]], axis=1), axis=0)
        sizes = np.unique(members[:, 0], return_counts=True)[1]
        fs, ft = seg_fibre[s], seg_fibre[t]
        pairs = len(np.unique(np.minimum(fs, ft) * len(radii) + np.maximum(fs, ft)))
        return {
            "contacts": int(pairs),
            "contacts_per_fibre": 2.0 * pairs / len(radii),
            "clots": int(np.sum(sizes > 2)),
        }

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        rng, seed_info = make_rng(seed)
        reach = self.length.upper() + 2.0 * self.radius.upper() + self.min_gap
        box = domain if domain.periodic else domain.dilated(0.5 * reach)
        radii, lengths = self._draw(rng, box.volume)
        nodes, first = self.walks(rng, radii, lengths)
        n = len(radii)
        summary = {"fibres": n}
        if self.placement_trials == 1 or n == 0:
            x = nodes + np.repeat(rng.random((n, 3)) * box.lengths, np.diff(first), axis=0)
        else:  # the best of the trials, the one of least overlap with those placed before
            from morphogen.kernels.force_biased import place

            x = nodes.copy()
            shifts = rng.random((n, self.placement_trials, 3)) * box.lengths
            lengths_box = np.asarray(box.lengths, dtype=float)
            depth = place(x, first, radii, lengths_box, float(self.min_gap), shifts)
            summary["placement_overlap"] = float(depth / max(float(radii.mean()), 1e-300))
        if self.packing and n:
            radii, stats = self._pack(x, first, radii, box)
            summary.update(stats)
        seg_fibre = np.repeat(np.arange(n), np.diff(first) - 1)
        seg_node = np.arange(len(seg_fibre)) + seg_fibre  # no segment from the last node
        start, end = x[seg_node], x[seg_node + 1]
        period = np.asarray(box.lengths, dtype=float)  # the midpoints moved into the box
        shift = box.lower - period * np.floor(0.5 * (start + end) / period)
        segments = Segments(start + shift, end + shift, radii[seg_fibre])
        return _fibres(domain, segments, self.lumen, self, seed_info, summary)

    def expected_count(self, domain: Domain) -> float | None:
        """Mean number of fibres in the domain (given a volume fraction)."""
        if self.volume_fraction is None:
            return None
        p = self.volume_fraction if self.packing else -math.log1p(-self.volume_fraction)
        return p * domain.volume / self.mean_volume

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        if self.volume_fraction is not None:
            p = self.volume_fraction
            total = p if self.packing else -math.log1p(-p)  # fibre volume per unit volume
        elif domain is not None and domain.periodic:
            total = self.number * self.mean_volume / domain.volume
            p = total if self.packing else -math.expm1(-total)
        else:
            return {}
        intensity = total / self.mean_volume
        out = {"intensity": intensity, "length_density": intensity * self.length.moment(1)}
        # packed: without overlaps, the segments within the 5-10 % of their springs
        out["volume_fraction"] = p
        if self.packing:
            out["specific_surface"] = p * self.mean_surface / self.mean_volume
        gap = self.min_gap + self.contact_distance
        reach = 2.0 * self.radius.moment(1) + gap
        ell, sine = self.length.moment(1), mean_sine(self.orientation)
        if self.radius.is_constant:
            excluded = excluded_volume(reach, ell, ell * ell, sine)
        else:  # averaged over the distance D = r_a + r_b + gap of two fibres in contact
            d1, d2, d3 = sizes.sum_moments(self.radius, gap)
            excluded = (
                4.0 / 3.0 * math.pi * d3 + 2.0 * math.pi * d2 * ell + 2.0 * d1 * ell**2 * sine
            )
        out["toll_contacts_per_fibre"] = intensity * excluded
        return out


def _shortlist(s, t, d, seg_fibre, n_fibres: int) -> np.ndarray:
    """The contact candidates kept (Keilmann et al. 2026, eq. 18): a pair of segments
    ``(s, t)`` at distance ``d`` is kept when no other candidate pair of the same two fibres
    sharing one of its segments is closer, so that a segment keeps at most one pair per
    other fibre."""
    m = len(s)
    if m == 0:
        return np.zeros(0, dtype=bool)
    ends = np.concatenate([s, t]).astype(np.int64)
    other = np.concatenate([seg_fibre[t], seg_fibre[s]]).astype(np.int64)
    inverse = np.unique(ends * n_fibres + other, return_inverse=True)[1]
    both = np.concatenate([d, d])
    best = np.full(int(inverse.max()) + 1, np.inf)
    np.minimum.at(best, inverse, both)
    closest = both <= best[inverse]
    return closest[:m] & closest[m:]


def mean_sine(orientation: OrientationDistribution, pairs: int = 200_000) -> float:
    """``E|sin g|``, ``g`` the angle between two independent axes of the law: pi/4
    isotropic, 2/pi planar, 0 aligned, otherwise by Monte Carlo over ``pairs`` pairs drawn
    from a fixed seed (deterministic, to about 0.2 %)."""
    if isinstance(orientation, Isotropic):
        return math.pi / 4.0
    if isinstance(orientation, Planar):
        return 2.0 / math.pi
    if isinstance(orientation, Aligned):
        return 0.0
    rng = np.random.default_rng(2026)
    u = orientation.sample(rng, pairs).reshape(-1, 3)
    v = orientation.sample(rng, pairs).reshape(-1, 3)
    return float(np.mean(np.linalg.norm(np.cross(u, v), axis=1)))


def excluded_volume(reach: float, length: float, length2: float, sine: float) -> float:
    """Mean volume of the centres of a straight fibre whose axis comes closer than
    ``reach`` to that of another one (axes of mean length ``length`` and mean product of
    two lengths ``length2``, ``E|sin g| = sine`` between them): the parallelogram swept by
    one axis along the other dilated by a ball (Steiner), ``4/3 pi D^3 + 2 pi D^2 l +
    2 D l1 l2 E|sin g|`` (Onsager's excluded volume for ``D = 2r``)."""
    return (
        4.0 / 3.0 * math.pi * reach**3
        + 2.0 * math.pi * reach**2 * length
        + 2.0 * reach * length2 * sine
    )


def _closest_parameters(r, a, b) -> tuple[np.ndarray, np.ndarray]:
    """Parameters ``(s, t)`` in [0, 1] of the closest points of the segments ``p + s a`` and
    ``q + t b``, ``r = p - q``, rows of pairs (Ericson 2005, sec. 5.1.9, as in
    :mod:`morphogen.kernels.force_biased`; parallel segments: the middle of the span of
    ``a`` facing ``b``)."""
    A = np.einsum("ij,ij->i", a, a)
    E = np.einsum("ij,ij->i", b, b)
    B = np.einsum("ij,ij->i", a, b)
    C = np.einsum("ij,ij->i", a, r)
    F = np.einsum("ij,ij->i", b, r)
    den = A * E - B * B
    lo, hi = np.clip(-C / A, 0.0, 1.0), np.clip((B - C) / A, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(den > 1e-9 * A * E, np.clip((B * F - C * E) / den, 0.0, 1.0), 0.5 * (lo + hi))
    t = (B * s + F) / E
    s = np.where(t < 0.0, lo, np.where(t > 1.0, hi, s))
    return s, np.clip(t, 0.0, 1.0)


class _Bonded(Combined):
    """Fibres combined (with their cross-links of another phase, or less their lumens),
    with the summary of the model."""

    def summary(self) -> dict:
        return {**super().summary(), **self.metadata.get("summary", {})}


@register
@dataclass(frozen=True)
class CrossLinkedFibers(Model):
    """Entangled cross-linked fibres: straight fibres bonded by cross-links at a fraction
    of their contacts (Mezeix et al. 2009, 2016; Chatti et al. 2018, 2020).

    As in Chatti et al. (Comput. Mater. Sci. 151, 14, 2018; Int. J. Solids Struct. 184,
    221, 2020), the volume elements of carbon fibres sprayed with epoxy: straight fibres of
    radius ``radius``, in contact where their axes come closer than a diameter (plus
    ``contact_gap``: a contact is a crossing), each contact bonded by a cross-link with
    the probability ``bonded_fraction`` (theirs: 1/3 to 1). ``length: null``: infinite
    fibres on Poisson lines whose directions follow ``orientation`` (the fibres of
    Chatti et al., 12 mm long, cross their 1 mm volume element; non-periodic samples, the
    exact restriction of the infinite medium); a ``length``: a Boolean model of
    spherocylinders of that cylindrical length (Mikado networks; periodic or not).
    ``orientation``: isotropic, or for their moulded material ``{type: girdle, sigma:
    0.4}``, their Gaussian law of the angle to the axis of compression (eq. 3). A
    cross-link is a ball of radius ``link_radius`` centred between the closest points of
    the axes, of phase ``link_phase`` outside the fibres (2; 1: merged with them). All the
    contacts bonded, the networks cross-linked at every crossing of Ban et al. (2016),
    Shahsavari & Picu (2015) or Bosco et al. (2017).

    The contacts along a fibre axis form, for Poisson lines, a Poisson process of
    intensity ``2 D L_V E|sin g|`` (``D = 2r + contact_gap``, ``g`` the angle between two
    fibres, ``E|sin g| = pi/4`` isotropic): the contacts are on average ``1 / (pi r L_V)``
    apart along an isotropic fibre, about ``d / 2f`` at small fractions ``f`` (Mezeix et al.
    2016, eq. 16, after Toll), the cross-links ``1 / bonded_fraction`` times farther; a
    finite fibre has on average ``theta (4/3 pi D^3 + 2 pi D^2 l + 2 D l^2 E|sin g|)``
    contacts (its excluded volume). The summary counts the contacts and cross-links
    centred in the domain and their spacings along the axes inside it.

    The radius is a number (monodisperse fibres) or a distribution (polydisperse: one
    radius per fibre). Two fibres of radii ``r_a`` and ``r_b`` are then in contact where
    their axes come closer than ``D = r_a + r_b + contact_gap``, and the closed forms take
    the moments of ``D`` (``E[D] = 2 E[R] + contact_gap`` along the lines, ``E[D^2]`` and
    ``E[D^3]`` in the excluded volume) and of ``R`` (``pi E[R^2]``, ``4/3 pi E[R^3]`` in the
    volumes of the fibres).
    """

    name = "cross_linked_fibers"
    phases: ClassVar[dict[int, str]] = {0: "pores", 1: "fibres", 2: "cross-links"}
    count_label = "fibres"

    radius: Distribution = param(
        Constant(0.5),
        doc="Fibre radius: a number (monodisperse) or a distribution (polydisperse, one "
        "radius per fibre)",
        kind="distribution",
        unit="length",
    )
    length: float | None = param(
        None,
        doc="Length of the cylindrical part of the fibres (round caps); none: infinite "
        "fibres on Poisson lines (non-periodic domains)",
        unit="length",
    )
    orientation: OrientationDistribution = param(
        Isotropic(),
        doc="Distribution of the fibre axes (isotropic, planar, or {type: girdle, sigma: "
        "0.4, axis: [0, 0, 1]}, the moulded fibres of Chatti et al.)",
        kind="orientation",
    )
    volume_fraction: float | None = param(
        None, doc="Volume fraction of the fibres", kind="fraction"
    )
    length_density: float | None = param(
        None, doc="Mean length of fibre axes per unit volume L_V", unit="1/length^2"
    )
    contact_gap: float = param(
        0.0,
        doc="Fibres are in contact where their surfaces are closer than this (0: where they cross)",
        unit="length",
    )
    bonded_fraction: float = param(
        1.0, doc="Fraction of the contacts bonded by a cross-link", kind="fraction"
    )
    link_radius: float = param(
        0.75, doc="Radius of the cross-links, balls centred at the contacts", unit="length"
    )
    link_phase: int = param(
        2, doc="Phase of the cross-links (1: merged with the fibres)", kind="int"
    )

    def _validate(self) -> None:
        if self.length is not None:
            check_positive(self.length, "length")
        exactly_one("volume_fraction", self.volume_fraction, "length_density", self.length_density)
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        else:
            check_positive(self.length_density, "length_density")
        check_positive(self.contact_gap, "contact_gap", allow_zero=True)
        check_fraction(self.bonded_fraction, "bonded_fraction", open_interval=False)
        check_positive(self.link_radius, "link_radius")
        if not 1 <= self.link_phase <= 255:
            raise ValueError("link_phase: 1 (with the fibres) or a phase index from 2 to 255")

    @property
    def supports_periodic(self) -> bool:  # infinite lines cannot be made periodic
        return self.length is not None

    def active_phases(self) -> dict[int, str]:
        if self.link_phase == 1:
            return {0: "pores", 1: "fibres and cross-links"}
        return {0: "pores", 1: "fibres", self.link_phase: "cross-links"}

    @property
    def grain_volume(self) -> float:
        """Mean volume of a fibre: an infinite cylinder per unit length, ``pi E[R^2]``, or
        a spherocylinder, ``pi E[R^2] l + 4/3 pi E[R^3]``."""
        law = self.radius
        if law.is_constant:  # as for a fixed radius, to the last bit
            r = law.moment(1)
            if self.length is None:
                return math.pi * r * r
            return math.pi * r * r * self.length + 4.0 / 3.0 * math.pi * r**3
        r2, r3 = law.moment(2), law.moment(3)
        if self.length is None:
            return math.pi * r2
        return math.pi * r2 * self.length + 4.0 / 3.0 * math.pi * r3

    @property
    def lv(self) -> float:
        """Mean length of fibre axes (cylindrical parts) per unit volume."""
        if self.length_density is not None:
            return self.length_density
        density = -math.log1p(-self.volume_fraction) / self.grain_volume
        return density if self.length is None else density * self.length

    @property
    def contact_reach(self) -> float:
        """Distance between axes below which two fibres are in contact (polydisperse
        fibres: its mean, ``2 E[R] + contact_gap``; two fibres ``r_a + r_b + contact_gap``)."""
        return 2.0 * self.radius.moment(1) + self.contact_gap

    @property
    def _largest_reach(self) -> float:
        """The largest distance between the axes of two fibres in contact."""
        return 2.0 * self.radius.upper() + self.contact_gap

    def _lines(self, domain: Domain, rng, reach: float) -> tuple[np.ndarray, np.ndarray]:
        """Ends of the Poisson lines within ``reach`` of the box, clipped to the box
        dilated by ``reach``: as many as hit the ball containing it (Poisson, of mean
        ``L_V pi rho^2`` for any law of directions), through points uniform in the disc of
        that ball normal to their directions."""
        rho = 0.5 * domain.diagonal + reach
        n = rng.poisson(self.lv * math.pi * rho * rho)
        directions = self.orientation.sample(rng, n).reshape(n, 3)
        e1, e2 = orthonormal_basis(directions)
        radial = rho * np.sqrt(rng.random(n))
        phi = rng.uniform(0.0, 2.0 * math.pi, n)
        points = (
            domain.center
            + (radial * np.cos(phi))[:, None] * e1
            + (radial * np.sin(phi))[:, None] * e2
        )
        t0, t1 = _clip_lines(points, directions, domain.lower - reach, domain.upper + reach)
        keep = t0 <= t1
        a = points[keep] + t0[keep, None] * directions[keep]
        return a, points[keep] + t1[keep, None] * directions[keep]

    def contacts(self, start, end, domain: Domain, radii=None) -> tuple[np.ndarray, np.ndarray]:
        """The contacts of the fibres of axes ``start -> end`` (and ``radii``, by default
        the mean radius): the pairs ``(i, j)`` of fibres whose axes come closer than ``r_i +
        r_j + contact_gap`` (nearest periodic images in a periodic domain) and the middles
        of their closest points (wrapped into a periodic domain)."""
        start = np.asarray(start, float).reshape(-1, 3)
        vec = np.asarray(end, float).reshape(-1, 3) - start
        n = len(start)
        if radii is None:
            radii = np.full(n, self.radius.moment(1))
        radii = np.asarray(radii, dtype=float).reshape(-1)
        if self.length is None:  # lines: every pair, 256 rows at a time
            blocks = []
            for k in range(0, max(n - 1, 0), 256):
                rows = np.arange(k, min(k + 256, n - 1))
                count = n - 1 - rows
                i = np.repeat(rows, count)
                j = i + 1 + np.arange(len(i)) - np.repeat(np.cumsum(count) - count, count)
                blocks.append(self._close(start, vec, i, j, domain, radii))
        else:
            from scipy import spatial

            mid = start + 0.5 * vec
            if domain.periodic:
                box = domain.lengths
                tree = spatial.cKDTree(np.mod(mid - domain.lower, box), boxsize=box)
            else:
                tree = spatial.cKDTree(mid)
            largest = 2.0 * float(radii.max(initial=0.0)) + self.contact_gap
            pairs = tree.query_pairs(self.length + largest, output_type="ndarray")
            blocks = [self._close(start, vec, pairs[:, 0], pairs[:, 1], domain, radii)]
        if not blocks:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3))
        return np.concatenate([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks])

    def _close(self, start, vec, i, j, domain: Domain, radii) -> tuple[np.ndarray, np.ndarray]:
        """The pairs ``(i, j)`` of axes in contact and the middles of their closest points."""
        reach = radii[i] + radii[j] + self.contact_gap
        shift = np.zeros((len(i), 3))
        if domain.periodic:  # the nearest image of the middle of j
            box = domain.lengths
            gap = (start[j] + 0.5 * vec[j]) - (start[i] + 0.5 * vec[i])
            shift = -box * np.round(gap / box)
        s, t = _closest_parameters(start[i] - (start[j] + shift), vec[i], vec[j])
        p = start[i] + s[:, None] * vec[i]
        q = start[j] + shift + t[:, None] * vec[j]
        close = np.einsum("ij,ij->i", p - q, p - q) < reach * reach
        middles = 0.5 * (p[close] + q[close])
        if domain.periodic:
            middles = domain.lower + np.mod(middles - domain.lower, domain.lengths)
        return np.stack([i[close], j[close]], axis=1), middles

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        self._check_domain(domain)
        rng, seed_info = make_rng(seed)
        ell, link = self.length, self.link_radius
        reach = 0.5 * self._largest_reach + link  # a cross-link reaches this far from an axis
        if ell is None:
            start, end = self._lines(domain, rng, reach)
        else:
            if domain.periodic and ell + self._largest_reach >= 0.5 * min(domain.size):
                raise ValueError(
                    f"{self.name}: fibres (length {ell:g} plus a diameter and the gap) must "
                    f"be shorter than half the periodic box (edge {min(domain.size):g})"
                )
            window = _germ_window(domain, 0.5 * ell + reach, self.name)
            n = rng.poisson(self.lv / ell * window.volume)
            centres = window.lower + rng.random((n, 3)) * window.lengths
            half = 0.5 * ell * self.orientation.sample(rng, n).reshape(n, 3)
            start, end = centres - half, centres + half
        radii = sizes.draw(self.radius, rng, len(start))  # one per fibre
        pairs, middles = self.contacts(start, end, domain, radii)
        bonded = rng.random(len(pairs)) < self.bonded_fraction
        lo, hi = domain.lower, domain.upper
        near = np.all((middles >= lo - link) & (middles <= hi + link), axis=1)
        inside = np.all((middles >= lo) & (middles < hi), axis=1)
        segments = _crop(Segments(start, end, radii, round_caps=True), domain)
        inner = segments.lengths if domain.periodic else _inner_lengths(segments, domain)
        summary = {
            "fibres": int(np.sum(inner > 0.0)),  # axes crossing the domain
            "contacts": int(np.sum(inside)),
            "cross_links": int(np.sum(inside & bonded)),
        }
        summary.update(sizes.size_summary("radius", segments.radii[inner > 0.0], self.radius))
        axes = float(np.sum(inner))
        for key, count in (
            ("contact", summary["contacts"]),
            ("cross_link", summary["cross_links"]),
        ):
            summary[f"{key}_spacing"] = axes / (2.0 * count) if count else math.inf
        balls = middles[near & bonded]
        links = Spheres(balls, np.full(len(balls), link))
        if self.link_phase == 1:
            micro = PrimitiveUnion(
                domain, spheres=links, segments=segments, model=self, seed=seed_info
            )
        else:
            fibres = PrimitiveUnion(domain, segments=segments, model=self, seed=seed_info)
            bonds = PrimitiveUnion(domain, spheres=links, model=self, seed=seed_info)
            resin = Combined(domain, "difference", [bonds, fibres], model=self, seed=seed_info)
            parts = [fibres, resin]
            micro = _Bonded(
                domain, "inclusions", parts, [self.link_phase], model=self, seed=seed_info
            )
        micro.metadata["summary"] = summary
        return micro

    def expected_count(self, domain: Domain) -> float:
        """Mean number of fibre axes crossing the domain: ``theta V`` in a periodic box;
        else, ``A`` the mean area of the projection of the box along the fibres (``S / 4``
        if isotropic), ``L_V A`` for lines and ``theta (V + l A)`` for segments."""
        if domain.periodic:
            return self.lv / self.length * domain.volume
        if isinstance(self.orientation, Isotropic):
            m = np.full(3, 0.5)  # E|u_k|
        else:
            u = self.orientation.sample(np.random.default_rng(2026), 200_000)
            m = np.abs(u.reshape(-1, 3)).mean(axis=0)
        lx, ly, lz = domain.size
        area = float(ly * lz * m[0] + lx * lz * m[1] + lx * ly * m[2])
        if self.length is None:
            return self.lv * area
        return self.lv / self.length * (domain.volume + self.length * area)

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        lv, reach = self.lv, self.contact_reach
        sine = mean_sine(self.orientation)
        b = self.bonded_fraction
        out = {"length_density": lv, "mean_sine": sine}
        if self.length is None:
            per_length = 2.0 * reach * lv * sine  # contacts per unit length of axis: E[D]
            if self.radius.is_constant:
                r = self.radius.moment(1)
                out["volume_fraction"] = -math.expm1(-lv * math.pi * r * r)
            else:
                out["volume_fraction"] = -math.expm1(-lv * self.grain_volume)
            out["contacts_per_length"] = per_length
            out["contact_intensity"] = 0.5 * lv * per_length
        else:
            theta, ell = lv / self.length, self.length
            if self.radius.is_constant:
                per_fibre = theta * excluded_volume(reach, ell, ell * ell, sine)
            else:  # the excluded volume averaged over D = r_a + r_b + contact_gap
                d1, d2, d3 = sizes.sum_moments(self.radius, self.contact_gap)
                mean = 4.0 / 3.0 * math.pi * d3 + 2.0 * math.pi * d2 * ell
                per_fibre = theta * (mean + 2.0 * d1 * ell * ell * sine)
            per_length = 2.0 * reach * lv * sine  # without the ends of the fibres
            out["intensity"] = theta
            out["volume_fraction"] = -math.expm1(-theta * self.grain_volume)
            out["contacts_per_fibre"] = per_fibre
            out["contact_intensity"] = 0.5 * theta * per_fibre
        out["cross_link_intensity"] = b * out["contact_intensity"]
        out["contact_spacing"] = 1.0 / per_length if per_length > 0.0 else math.inf
        out["cross_link_spacing"] = 1.0 / (b * per_length) if b * per_length > 0 else math.inf
        return out


def _inner_lengths(segments: Segments, domain: Domain) -> np.ndarray:
    """Lengths of the parts of the segments inside a box."""
    vec = segments.end - segments.start
    t0, t1 = _clip_lines(segments.start, vec, domain.lower, domain.upper)
    t0, t1 = np.clip(t0, 0.0, 1.0), np.clip(t1, 0.0, 1.0)
    return np.maximum(t1 - t0, 0.0) * np.linalg.norm(vec, axis=1)
