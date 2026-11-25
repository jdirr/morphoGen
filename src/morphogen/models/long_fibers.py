"""Long fibres along an axis of the box: the unidirectional fibre composites, their fibres
parallel or misaligned (wavy), their sections a random packing or a periodic array.

The fibres cross the whole box along ``axis`` and close on themselves across its periodic
faces: they are the infinite fibres of the periodic medium. Their sections, discs of radii
drawn one per fibre, are packed in the plane normal to the axis without overlap, a random
packing (inflated from a looser one, the overlaps removed at each step: the random fibre
distributions of unidirectional composites, Melro et al. 2008, Vaughan & McCarthy 2010,
Yang et al. 2013) or a hexagonal or square array, its sites displaced at random by a
fraction of the spacing; misaligned fibres then wave about their mean positions along
smooth random periodic paths whose segments make a given RMS angle with the axis, and are
freed of their overlaps in space (:mod:`morphogen.kernels.long_fibres`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen._utils import check_fraction, check_positive, exactly_one
from morphogen.distributions import Constant, Distribution
from morphogen.domain import Domain
from morphogen.models.base import Model, param, register
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments
from morphogen.seeding import SeedLike, make_rng

ARRANGEMENTS = ("random", "hexagonal", "square")
#: The largest volume fraction of the random packings of the sections (discs jam near 0.82)
RANDOM_LIMIT = 0.8
#: The largest RMS angle of the misalignment, in degrees
MAX_MISALIGNMENT = 30.0
#: The volume fraction of the sections at the start of their inflation, and the factor of
#: the fraction at each of its steps
START_FRACTION = 0.3
GROWTH = 1.02
#: The spectral variance below which the modes of the waviness are left out (relative to
#: a mode of zero wave number): ``exp(-(k l)^2 / 2) >= 1e-6``, ``k l <= 5.257``
SPECTRUM_CUT = math.sqrt(2.0 * math.log(1e6))
#: Segments per wavelength of the shortest mode of the waviness
SEGMENTS_PER_WAVE = 8
#: The passes of the filter that smooths the corrections of wavy fibres along them
SMOOTHING = 2


_SECANTS: dict[float, float] = {}


def secant_mean(theta: float) -> float:
    """``E[sec a]``, the mean contour length of the fibres per unit length of the axis, for
    angles ``a`` to the axis whose tangents follow a Rayleigh law (the slopes of the
    waviness, Gaussian in the plane normal to the axis), of the scale that gives them the
    RMS angle ``theta`` (radians): ``1 + theta^2/2 + 5 theta^4/12 + O(theta^6)``. The section
    of a fibre tilted by ``a`` in a plane normal to the axis is ``pi r^2 sec a``: the volume
    fraction of the fibres is that of their sections in such a plane times this mean."""
    theta = float(theta)
    if theta <= 0.0:
        return 1.0
    if theta not in _SECANTS:
        from scipy.optimize import brentq

        s, w = np.polynomial.laguerre.laggauss(64)  # E[f(R)] = sum w f(sigma sqrt(2 s))

        def mean_square(sigma):
            return float(w @ np.arctan(sigma * np.sqrt(2.0 * s)) ** 2)

        sigma = brentq(lambda x: mean_square(x) - theta * theta, 1e-12, 1e3, xtol=1e-14)
        _SECANTS[theta] = float(w @ np.sqrt(1.0 + 2.0 * sigma * sigma * s))
    return _SECANTS[theta]


def chain_volumes(nodes: np.ndarray, h: float, radii: np.ndarray) -> np.ndarray:
    """Volumes of closed chains of spherocylinders (fibres x nodes x 2: the nodes in their
    planes, ``h`` apart along the axis, the last joined to the first a period further):
    ``pi r^2`` times the contour length, less ``2/3 r^3 (2 tan(d/2) - d)`` at each joint
    turning by ``d`` (the cylinders overlap inside the bend, the ball of the joint fills
    its outside; :func:`morphogen.models.fibers._chain_volumes` for open chains)."""
    t = np.roll(nodes, -1, axis=1) - nodes
    t = np.concatenate([t, np.full((*t.shape[:2], 1), float(h))], axis=2)
    lengths = np.linalg.norm(t, axis=2)
    u = t / lengths[..., None]
    cos = np.einsum("fki,fki->fk", np.roll(u, 1, axis=1), u)
    d = np.arccos(np.clip(cos, -1.0, 1.0))
    joints = np.sum(2.0 * np.tan(0.5 * d) - d, axis=1)
    return math.pi * radii**2 * lengths.sum(axis=1) - 2.0 / 3.0 * radii**3 * joints


def segment_angles(nodes: np.ndarray, h: float) -> np.ndarray:
    """The angles (radians) of the segments of closed chains to the axis (fibres x
    segments)."""
    step = np.linalg.norm(np.roll(nodes, -1, axis=1) - nodes, axis=2)
    return np.arctan2(step, float(h))


@register
@dataclass(frozen=True)
class LongFibers(Model):
    """Long fibres along an axis of the box: a unidirectional fibre composite, its fibres
    parallel or misaligned, monodisperse or polydisperse.

    The fibres cross the box along ``axis`` and close on themselves across its periodic
    faces, chains of spherocylinders of a radius per fibre (drawn from ``radius``: a number
    for monodisperse fibres, a distribution for polydisperse ones). Their sections, discs in
    the plane normal to the axis, are packed without coming closer than ``min_gap``:

    * ``arrangement: random``, a random packing: the discs at uniform positions, shrunk to
      a fraction of 0.3, their overlaps removed by steps that push every overlapping pair
      apart along the line of its centres, then grown by 2 % of their fraction at a time,
      the overlaps removed again at each growth, up to their radii (as the random fibre
      distributions of Melro et al. 2008, Vaughan & McCarthy 2010 and Yang et al. 2013, to
      the fractions of real composites, 0.5 to 0.7, which random sequential addition, jammed
      near 0.55, does not reach; at most 0.8, near the jamming of the discs);
    * ``hexagonal`` or ``square``, the periodic arrays of the micromechanics: the spacing of
      the volume fraction, the array stretched to hold a whole number of its cells in the
      box (an even number of rows of the hexagonal one), its sites displaced by up to
      ``disorder`` spacings at random (uniform in a disc), the overlaps removed.

    ``misalignment > 0``: each fibre waves about the position of its section along a smooth
    random periodic path, the value at its section of a random field of the box, the sum of
    Fourier modes whose variances follow the Gaussian covariance of correlation lengths
    ``waviness_length`` along the axis and ``transverse_correlation`` across it (by default
    the same: neighbouring fibres wave together, as the misalignments measured in composites
    do, Yurgartis 1987; 0: each fibre on its own), its two transverse components
    independent; the paths are scaled so that the RMS angle of the segments of the
    realization to the axis is ``misalignment`` (degrees), then freed of their overlaps in
    space, their nodes moved in their planes by the pushes of the closest points of the
    segments that overlap, the pushes smoothed along the fibres. The
    volume fraction is that of the sections in a plane normal to the axis times the mean
    contour length per unit length, ``E[sec a] = 1 + theta^2/2 + ...`` (:func:`secant_mean`):
    the number of fibres takes it into account. The summary gives the realized fraction
    (the exact volume of the chains), the RMS and largest angles of the segments after the
    corrections, the steps of the packing and of the corrections, and the smallest gap
    between two fibres.

    Generated in a periodic box; a non-periodic domain is a window of a periodic box
    enlarged by the reach of a fibre (two radii, the gap and three amplitudes of the
    waviness) across the axis, and by two correlation lengths along it.
    """

    name = "long_fibers"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "fibres"}
    count_label = "fibres"

    radius: Distribution = param(
        Constant(0.5),
        doc="Fibre radius, one per fibre (a number: monodisperse fibres; a distribution: "
        "polydisperse fibres)",
        kind="distribution",
        unit="length",
    )
    volume_fraction: float | None = param(None, doc="Volume fraction of fibres", kind="fraction")
    number: int | None = param(
        None, doc="Number of fibres in the generation box (random arrangement)", kind="int"
    )
    axis: int = param(2, doc="Axis of the fibres (0: x, 1: y, 2: z)", kind="int")
    arrangement: str = param(
        "random",
        doc="Sections: random (a random packing), hexagonal or square (periodic arrays)",
        kind="str",
    )
    disorder: float = param(
        0.0,
        doc="hexagonal, square: random displacement of the sites of the array, in spacings "
        "(then the overlaps removed)",
        kind="fraction",
    )
    misalignment: float = param(
        0.0,
        doc="RMS angle of the segments of the fibres to the axis, in degrees (0: straight "
        "parallel fibres)",
        unit="deg",
    )
    waviness_length: float | None = param(
        None,
        doc="Correlation length of the waviness along the fibres (default: a quarter of the "
        "length of the domain along the axis)",
        unit="length",
    )
    transverse_correlation: float | None = param(
        None,
        doc="Correlation length of the waviness across the fibres: neighbouring fibres wave "
        "together (default: waviness_length; 0: each fibre waves independently)",
        unit="length",
    )
    segments: int | None = param(
        None,
        doc="Segments per fibre over the period along the axis (default: 8 per wavelength of "
        "the shortest mode of the waviness; 2 for straight fibres)",
        kind="int",
    )
    min_gap: float = param(0.0, doc="Minimal distance between fibre surfaces", unit="length")
    max_steps: int = param(
        20_000,
        doc="Maximal number of steps of the packing of the sections, and of the corrections "
        "of wavy fibres",
        kind="int",
    )

    def _validate(self) -> None:
        exactly_one("volume_fraction", self.volume_fraction, "number", self.number)
        if self.arrangement not in ARRANGEMENTS:
            raise ValueError(f"arrangement: one of {', '.join(ARRANGEMENTS)}")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis: 0, 1 or 2")
        check_positive(self.min_gap, "min_gap", allow_zero=True)
        if not 0.0 <= self.misalignment <= MAX_MISALIGNMENT:
            raise ValueError(f"misalignment: an RMS angle from 0 to {MAX_MISALIGNMENT:g} degrees")
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
            largest = self.largest_fraction()
            if self.volume_fraction > largest:
                what = (
                    "random packings of the sections jam near 0.82"
                    if self.arrangement == "random"
                    else f"the {self.arrangement} array of these radii and gap"
                )
                raise ValueError(f"volume_fraction: at most {largest:.4g} ({what})")
        elif self.arrangement != "random":
            raise ValueError("number: the arrays (hexagonal, square) take a volume_fraction")
        elif self.number < 0:
            raise ValueError("number: must be non-negative")
        if not 0.0 <= self.disorder < 1.0:
            raise ValueError("disorder: a fraction of the spacing, in [0, 1)")
        if self.disorder > 0.0 and self.arrangement == "random":
            raise ValueError(
                "disorder: the displacement of the sites of the arrays (hexagonal, square)"
            )
        if self.waviness_length is not None:
            check_positive(self.waviness_length, "waviness_length")
        if self.transverse_correlation is not None:
            check_positive(self.transverse_correlation, "transverse_correlation", allow_zero=True)
        if self.segments is not None and self.segments < 2:
            raise ValueError("segments: at least 2 per fibre")
        if self.max_steps < 1:
            raise ValueError("max_steps: must be positive")

    # -- geometry of the arrangements ------------------------------------------------------

    def largest_fraction(self) -> float:
        """The largest volume fraction of the arrangement: ``RANDOM_LIMIT`` for the random
        packings; for the arrays, that of their touching fibres of the largest radius, with
        the gap between their surfaces (``pi/(2 sqrt 3)`` hexagonal, ``pi/4`` square, of
        equal radii without a gap), times the mean contour length per unit length."""
        if self.arrangement == "random":
            return RANDOM_LIMIT
        r2, d = self.radius.moment(2), 2.0 * self.radius.upper() + self.min_gap
        cell = math.sqrt(3.0) / 2.0 * d * d if self.arrangement == "hexagonal" else d * d
        return min(1.0, math.pi * r2 / cell * self.contour_factor)

    @property
    def contour_factor(self) -> float:
        """The mean contour length of the fibres per unit length of the axis,
        :func:`secant_mean` of the misalignment."""
        return secant_mean(math.radians(self.misalignment))

    def spacing(self) -> float | None:
        """The spacing of the array of the volume fraction (None for the random packings):
        ``a^2 = 2 pi E[r^2] c / (sqrt(3) p)`` hexagonal, ``pi E[r^2] c / p`` square (``c`` the
        :attr:`contour_factor`), before it is stretched to the box."""
        if self.arrangement == "random" or self.volume_fraction is None:
            return None
        area = math.pi * self.radius.moment(2) * self.contour_factor / self.volume_fraction
        return math.sqrt(2.0 * area / math.sqrt(3.0) if self.arrangement == "hexagonal" else area)

    def array_shape(self, section) -> tuple[int, int]:
        """The cells of the array along the two transverse axes of a section of sides
        ``section`` (sites per row, rows: an even number of rows for the hexagonal array)."""
        a = self.spacing()
        lu, lv = (float(s) for s in section)
        if self.arrangement == "hexagonal":
            return max(1, round(lu / a)), 2 * max(1, round(lv / (math.sqrt(3.0) * a)))
        return max(1, round(lu / a)), max(1, round(lv / a))

    def _sites(self, section) -> tuple[np.ndarray, float]:
        """The sites of the array in a section (n x 2) and the nearest distance of two."""
        nu, nv = self.array_shape(section)
        au, av = float(section[0]) / nu, float(section[1]) / nv
        c, r = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
        c, r = c.ravel().astype(float), r.ravel().astype(float)
        if self.arrangement == "hexagonal":
            sites = np.stack([(c + 0.25 + 0.5 * (r % 2)) * au, (r + 0.5) * av], axis=1)
            near = min(au, math.hypot(0.5 * au, av))
        else:
            sites = np.stack([(c + 0.5) * au, (r + 0.5) * av], axis=1)
            near = min(au, av)
        return sites, near

    # -- the waviness ------------------------------------------------------------------------

    def correlation_length(self, length: float) -> float:
        """The correlation length of the waviness along the fibres: ``waviness_length``, or
        a quarter of ``length``, that of the domain along the axis."""
        return self.waviness_length if self.waviness_length is not None else 0.25 * length

    @staticmethod
    def _modes(period: float, ell: float) -> int:
        """The Fourier modes of the waviness: those whose variance is above 1e-6 of that of
        a mode of zero wave number (at least one)."""
        return max(1, int(SPECTRUM_CUT * period / (2.0 * math.pi * ell)))

    def _segments(self, period: float, ell: float) -> int:
        if self.segments is not None:
            return int(self.segments)
        if self.misalignment == 0.0:
            return 2
        return max(SEGMENTS_PER_WAVE, SEGMENTS_PER_WAVE * self._modes(period, ell))

    def transverse_length(self, ell: float) -> float:
        """The correlation length of the waviness across the fibres:
        ``transverse_correlation``, or that along them (0: the fibres wave independently)."""
        return ell if self.transverse_correlation is None else float(self.transverse_correlation)

    def waviness(self, rng, x0, section, period: float, m: int, ell: float) -> np.ndarray:
        """The offsets of the nodes of the fibres from the positions ``x0`` of their sections
        (n x m x 2), ``m`` nodes over the period along the axis: a random field of the plane
        normal to the axis and of the axis, periodic in the box, the sum of Fourier modes of
        Gaussian amplitudes whose variances ``exp(-(|k_t|^2 l_t^2 + k^2 l^2) / 2)`` are those
        of the Gaussian covariance of correlation lengths ``l_t`` across the axis
        (:meth:`transverse_length`) and ``l`` along it, of wave numbers ``k = 2 pi j /
        period`` with ``j >= 1`` along the axis (no static distortion of the sections), its
        two transverse components independent; ``l_t = 0``, a field of the axis for each
        fibre, independent of the others. It is scaled so that the RMS angle of the segments
        to the axis is the misalignment."""
        theta = math.radians(self.misalignment)
        n = len(x0)
        out = np.zeros((n, m, 2))
        if n == 0 or theta == 0.0:
            return out
        z = np.arange(m) * (period / m)
        k = 2.0 * math.pi * np.arange(1, self._modes(period, ell) + 1) / period
        cz, sz = np.cos(np.outer(k, z)), np.sin(np.outer(k, z))
        lt = self.transverse_length(ell)
        if lt == 0.0:  # a field of the axis per fibre
            std = np.exp(-0.25 * (k * ell) ** 2)
            a = rng.normal(size=(n, len(k), 2)) * std[None, :, None]
            b = rng.normal(size=(n, len(k), 2)) * std[None, :, None]
            out = np.einsum("njc,jk->nkc", a, cz) + np.einsum("njc,jk->nkc", b, sz)
        else:
            lu, lv = (float(v) for v in section)
            mu = int(SPECTRUM_CUT * lu / (2.0 * math.pi * lt))
            mv = int(SPECTRUM_CUT * lv / (2.0 * math.pi * lt))
            ju, jv = np.meshgrid(np.arange(-mu, mu + 1), np.arange(-mv, mv + 1), indexing="ij")
            kt = np.stack([2.0 * math.pi * ju.ravel() / lu, 2.0 * math.pi * jv.ravel() / lv], 1)
            q = (kt**2).sum(axis=1)[:, None] * lt * lt + (k * ell)[None, :] ** 2
            std = np.where(q <= SPECTRUM_CUT**2, np.exp(-0.25 * q), 0.0)
            a = rng.normal(size=(*q.shape, 2)) * std[..., None]
            b = rng.normal(size=(*q.shape, 2)) * std[..., None]
            # cos(kt.x + k z) = Ct Cz - St Sz, sin(kt.x + k z) = St Cz + Ct Sz
            first = np.einsum("tjc,jk->tkc", a, cz) + np.einsum("tjc,jk->tkc", b, sz)
            second = np.einsum("tjc,jk->tkc", b, cz) - np.einsum("tjc,jk->tkc", a, sz)
            phase = np.asarray(x0, dtype=float) @ kt.T
            out = np.cos(phase) @ first.reshape(len(kt), -1) + np.sin(phase) @ second.reshape(
                len(kt), -1
            )
            out = out.reshape(n, m, 2)
        slope = np.linalg.norm(np.roll(out, -1, axis=1) - out, axis=2) / (period / m)
        top = float(slope.max())
        if top <= 0.0:
            return out
        from scipy.optimize import brentq

        def excess(c):
            return math.sqrt(float(np.mean(np.arctan(c * slope) ** 2))) - theta

        hi = 1.0 / top
        while excess(hi) < 0.0:
            hi *= 2.0
        return out * brentq(excess, 0.0, hi, xtol=1e-14 * hi)

    # -- generation --------------------------------------------------------------------------

    def _draw(self, rng, area: float) -> np.ndarray:
        """The radii of the fibres: the given number, or as many as bring the total area of
        their sections, times the mean contour length, closest to ``p area``."""
        if self.number is not None:
            return self.radius.sample(rng, self.number)
        target = self.volume_fraction * area / self.contour_factor
        radii = np.empty(0)
        while True:
            m = int(1.1 * target / (math.pi * self.radius.moment(2))) + 16
            radii = np.concatenate([radii, self.radius.sample(rng, m)])
            cum = np.cumsum(math.pi * radii**2)
            if cum[-1] >= target:
                break
        k = int(np.searchsorted(cum, target))
        n = k + 1 if k == 0 or cum[k] - target <= target - cum[k - 1] else k
        return radii[:n]

    def _random_section(self, rng, radii, box, gap: float) -> tuple[np.ndarray, int]:
        """A random packing of the sections in the periodic rectangle ``box``: uniform
        centres, inflated from a fraction of :data:`START_FRACTION` by :data:`GROWTH`, the
        overlaps removed at each step (:func:`morphogen.kernels.long_fibres.relax_discs`)."""
        from morphogen.kernels.long_fibres import relax_discs

        x = rng.random((len(radii), 2)) * box
        if len(radii) == 0:
            return x, 0
        r_min = float(radii.min())
        p = math.pi * float(np.sum(radii**2)) / float(box[0] * box[1])
        scale = min(1.0, math.sqrt(START_FRACTION / p))
        total = 0
        while True:
            tol = 1e-7 * r_min if scale >= 1.0 else 1e-3 * r_min
            steps, largest = relax_discs(x, radii, scale, gap, box, tol,
                                         self.max_steps - total, 0.25 * scale * r_min)  # fmt: skip
            total += steps
            if largest > tol:
                raise RuntimeError(
                    f"long_fibers: overlaps of {largest:.3g} left in the sections after "
                    f"{total} steps (at {scale:.3f} of their radii); lower the volume "
                    "fraction or raise max_steps"
                )
            if scale >= 1.0:
                return x, total
            scale = min(1.0, scale * math.sqrt(GROWTH))

    def _section(self, rng, box, gap: float) -> tuple[np.ndarray, np.ndarray, int]:
        """The centres and radii of the sections, and the steps of their packing."""
        from morphogen.kernels.long_fibres import disc_overlaps, relax_discs

        if self.arrangement == "random":
            radii = self._draw(rng, float(box[0] * box[1]))
            p = math.pi * float(np.sum(radii**2)) * self.contour_factor / float(box[0] * box[1])
            if self.number is not None and p > RANDOM_LIMIT:
                raise ValueError(
                    f"number: {self.number} fibres fill {p:.3g} of the box, above the "
                    f"{RANDOM_LIMIT:g} that random packings of their sections reach"
                )
            x, steps = self._random_section(rng, radii, box, gap)
            return x, radii, steps
        sites, near = self._sites(box)
        radii = self.radius.sample(rng, len(sites))
        if self.disorder > 0.0:  # uniform in a disc of radius disorder x spacing
            rho = self.disorder * near * np.sqrt(rng.random(len(sites)))
            phi = rng.uniform(0.0, 2.0 * math.pi, len(sites))
            sites = np.mod(sites + np.stack([rho * np.cos(phi), rho * np.sin(phi)], 1), box)
        largest, _ = disc_overlaps(sites, radii, 1.0, gap, box, np.zeros((0, 2)))
        if largest <= 0.0:
            return sites, radii, 0
        r_min = float(radii.min())
        steps, largest = relax_discs(sites, radii, 1.0, gap, box, 1e-7 * r_min,
                                     self.max_steps, 0.25 * r_min)  # fmt: skip
        if largest > 1e-7 * r_min:
            raise RuntimeError(
                f"long_fibers: overlaps of {largest:.3g} left in the {self.arrangement} array "
                f"after {steps} steps; lower the volume fraction, the disorder or the spread "
                "of the radii"
            )
        return sites, radii, steps

    def _box(self, domain: Domain) -> tuple[np.ndarray, np.ndarray, float, float]:
        """The generation box: its lower corner, its section (the sides across the axis),
        its period along the axis and the correlation length of the waviness."""
        a, (u, v) = self.axis, self._transverse()
        lengths = np.asarray(domain.lengths, dtype=float)
        ell = self.correlation_length(float(lengths[a]))
        if domain.periodic:
            return domain.lower.copy(), lengths[[u, v]], float(lengths[a]), ell
        amplitude = ell * math.tan(math.radians(self.misalignment))
        margin = np.zeros(3)
        margin[[u, v]] = 2.0 * self.radius.upper() + self.min_gap + 3.0 * amplitude
        margin[a] = 2.0 * ell if self.misalignment > 0.0 else 0.0
        lower = domain.lower - margin
        size = lengths + 2.0 * margin
        return lower, size[[u, v]], float(size[a]), ell

    def _transverse(self) -> tuple[int, int]:
        return (self.axis + 1) % 3, (self.axis + 2) % 3

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        from morphogen.kernels.long_fibres import chain_overlaps, relax_chains

        rng, seed_info = make_rng(seed)
        lower, box, period, ell = self._box(domain)
        gap = float(self.min_gap)
        reach = 2.0 * self.radius.upper() + gap
        if 2.0 * reach >= box.min():
            raise ValueError(
                f"long_fibers: the section of the box ({box.min():g} across) must exceed "
                f"twice the reach of two fibres, 2 (2 r + min_gap) = {2.0 * reach:g}"
            )
        margin_gap = gap + 1e-6 * float(self.radius.moment(1))
        x0, radii, packing = self._section(rng, box, margin_gap)
        n = len(radii)
        m = self._segments(period, ell)
        h = period / m
        nodes = x0[:, None, :] + self.waviness(rng, x0, box, period, m, ell)
        corrections = 0
        if n and self.misalignment > 0.0:
            lim = 2.0 * float(radii.max()) + gap
            if 2.0 * (lim + float(np.ptp(nodes, axis=1).max())) >= box.min():
                raise ValueError(
                    f"long_fibers: the section of the box ({box.min():g} across) must exceed "
                    "twice the reach of two wavy fibres (2 r + min_gap plus their excursion "
                    "across the axis): a wider box or a shorter waviness_length"
                )
            span = int(lim / h) + 1
            r_min = float(radii.min())
            corrections, largest = relax_chains(nodes, radii, margin_gap, box, h, span,
                                                1e-7 * r_min, int(self.max_steps), SMOOTHING,
                                                0.25 * r_min)  # fmt: skip
            if largest > 1e-7 * r_min:
                raise RuntimeError(
                    f"long_fibers: overlaps of {largest:.3g} left between wavy fibres after "
                    f"{corrections} steps; lower the volume fraction or the misalignment, "
                    "lengthen the waviness or raise max_steps"
                )
        summary: dict = {"fibres": n}
        if n:
            span = int((2.0 * float(radii.max()) + gap) / h) + 1
            _, _, clearance = chain_overlaps(nodes, radii, 0.0, box, h, span, np.zeros((0, 0, 2)))
            angles = segment_angles(nodes, h)
            volume = float(np.sum(chain_volumes(nodes, h, radii)))
            summary.update(
                volume_fraction=volume / (float(box[0] * box[1]) * period),
                misalignment_rms=math.degrees(math.sqrt(float(np.mean(angles**2)))),
                misalignment_max=math.degrees(float(angles.max())),
                packing_steps=int(packing),
                relaxation_steps=int(corrections),
            )
            if np.isfinite(clearance):
                summary["smallest_gap"] = float(clearance)
        micro = PrimitiveUnion(
            domain, segments=self._segments_of(nodes, radii, lower, box, period, domain),
            model=self, seed=seed_info,
        )  # fmt: skip
        micro.metadata["summary"] = summary
        return micro

    def _segments_of(self, nodes, radii, lower, box, period: float, domain: Domain) -> Segments:
        """The segments of the chains in the domain: the midpoints of a periodic domain
        moved into it, those of a window cropped to it."""
        a, (u, v) = self.axis, self._transverse()
        n, m = nodes.shape[:2]
        closed = np.concatenate([nodes, nodes[:, :1]], axis=1)
        coords = np.empty((n, m + 1, 3))
        coords[..., a] = np.arange(m + 1) * (period / m)
        coords[..., u] = closed[..., 0]
        coords[..., v] = closed[..., 1]
        start = coords[:, :-1].reshape(-1, 3)
        end = coords[:, 1:].reshape(-1, 3)
        r = np.repeat(np.asarray(radii, dtype=float), m)
        if domain.periodic:
            sizes = np.empty(3)
            sizes[a], sizes[[u, v]] = period, box
            shift = -sizes * np.floor(0.5 * (start + end) / sizes)
            return Segments(lower + start + shift, lower + end + shift, r, round_caps=True)
        segments = Segments(lower + start, lower + end, r, round_caps=True)
        lo, hi = segments.bounds()
        return segments.subset(np.all((hi >= domain.lower) & (lo <= domain.upper), axis=1))

    # -- theory ------------------------------------------------------------------------------

    def _section_area(self, domain: Domain) -> float:
        lengths = np.asarray(domain.lengths, dtype=float)
        u, v = self._transverse()
        return float(lengths[u] * lengths[v])

    def expected_count(self, domain: Domain) -> float | None:
        """Mean number of fibres crossing a section of the domain normal to the axis."""
        if self.number is not None:
            return float(self.number) if domain.periodic else None
        if self.arrangement != "random" and domain.periodic:
            nu, nv = self.array_shape(np.asarray(domain.lengths)[list(self._transverse())])
            return float(nu * nv)
        p = self.volume_fraction
        area = math.pi * self.radius.moment(2) * self.contour_factor
        return p * self._section_area(domain) / area

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """The volume fraction (that of the array in a periodic box, a whole number of its
        cells), the number of fibres per unit area of a section normal to the axis, their
        length per unit volume ``L_V = p / (pi E[r^2])``, the specific surface ``2 p E[r] /
        E[r^2]`` (their lateral surface; the joints of wavy chains left out), the mean
        contour length per unit length of the axis; for the arrays their spacing and largest
        fraction."""
        c = self.contour_factor
        r1, r2 = self.radius.moment(1), self.radius.moment(2)
        if (
            domain is not None
            and domain.periodic
            and (self.number is not None or self.arrangement != "random")
        ):
            p = self.expected_count(domain) * math.pi * r2 * c / self._section_area(domain)
        elif self.volume_fraction is not None:
            p = self.volume_fraction
        else:
            return {}
        out = {
            "volume_fraction": p,
            "intensity": p / (math.pi * r2 * c),
            "length_density": p / (math.pi * r2),
            "specific_surface": 2.0 * p * r1 / r2,
            "contour_factor": c,
        }
        if self.arrangement != "random":
            out["spacing"] = self.spacing()
            out["largest_fraction"] = self.largest_fraction()
        return out
