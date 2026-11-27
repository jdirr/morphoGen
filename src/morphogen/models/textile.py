"""Woven textile composites: yarns swept along the crimped paths of their weave.

Yarns (tows) of fibres interlace in a plane: the warp yarns run along the first axis of
the plane of the fabric, the weft yarns along the second, and each passes over and under
the others as its weave says. A yarn is a section swept along its centreline: in the
plane normal to the centreline, the section of half width ``a`` along the width axis
``n`` (horizontal, normal to the centreline) and of half thickness ``c`` along the
thickness axis ``b = t x n`` (``t``: the tangent), an ellipse, a lens (the intersection
of two discs) or a power ellipse ``|u/a|^q + |v/c|^q <= 1``. The centreline of a yarn
lies in a vertical plane; along it, the stations (the crossings with the yarns of the
other direction, one spacing apart) have levels, and the path passes from one level to
the next by a half cosine over a ``transition`` share of the spacing, level elsewhere
(the idealization of Kuhn & Charalambides 1999 and of the geometric preprocessors,
WiseTex and TexGen: Lomov et al. 2001, Sherburn 2007); in the orthogonal weaves the
binders go through the thickness by arcs and vertical lines, in the gaps between the
weft yarns.

* 2D weaves, by their interlacing matrix ``W`` (warp ``i`` over weft ``j`` where
  ``W[i mod R_y, j mod R_x]``): ``plain``; ``twill`` of ``pattern [n, m]`` (over ``n``
  wefts, under ``m``, shifted by one weft from a warp to the next); ``satin`` of
  ``pattern [H, s]`` (the warp goes under one weft in ``H``, at the move ``s``,
  coprime with ``H``); ``basket`` of ``pattern [k]`` (``k`` yarns woven together as one
  in a plain weave). At a crossing, the centrelines of the warp and of the weft are
  ``c_w + c_f`` apart about the mid-plane of the ply, so that their sections touch;
  ``layers`` plies are stacked, each shifted by ``ply_shift`` spacings from the one
  below (or by random shifts), ``ply_gap`` of resin between them.
* 3D weaves (Ansar, Xinwei & Chouwei 2011): ``layers`` straight warp layers between
  ``layers + 1`` straight weft layers, tied by binder (z) yarns along the warp, in
  columns between those of the warps (adjacent binders out of phase): ``orthogonal``
  binders over the top weft layer and under the bottom one, through the thickness in
  the gaps between the wefts; ``layer_to_layer`` angle interlock, a binder for each warp
  layer, over a weft of its layer and under one of the layer below; ``through_thickness``
  angle interlock, a binder that goes down one layer at each weft column, from the top to
  the bottom and back. The binding depth (``binder_depth``; Boussu, Cristian & Nauman
  2015) is the number of weft layers that a binder ties together: all of them in the
  orthogonal weave and 2 in the layer-to-layer one by default; shallower orthogonal
  binders tie the layers in turn from one column to the next. Without stuffers
  (``stuffers``), the angle interlocks have no straight warps.

The tows of real fabrics depart from their idealized paths, and their sections change along
them, as the tomography of textile composites measures (Blacklock et al. 2012; Vanaerschot,
Cox, Lomov & Vandepitte 2013, 2014): the deviations of the centrelines across the yarns in
the plane of the fabric and through its thickness, and the widths and thicknesses of their
sections (lognormal, of the mean of their yarn), are stationary Gaussian processes along
the direction of the yarns (:func:`process_spectrum`), of a gaussian or exponential
covariance, periodic over the box, correlated between neighbouring yarns of a ply or layer
(:func:`neighbour_correlations`). The nodes of a yarn that varies move by its deviations,
its frames turn with them, and the half sizes of its section vary linearly between its
nodes.

The yarns are a partition of their union: a point inside two yarns (the idealized paths
and sections may interpenetrate, near the crossings of wide yarns) belongs to the yarn
it is deeper in, by the functions of their sections; the share of the volume of the
yarns where two of them overlap is in the summary of a realization. Each segment of the
polyline of a yarn is a region of its own (``labels``) with the orientation of its
material (``orientations``): the axis 3 along the yarn, the axis 1 across its width, the
axis 2 through its thickness, so that transversely isotropic or orthotropic yarns are
turned along their paths by the solvers that take grains and orientations.

The conforming meshes of a textile come from the exact functions of its phases at any
point (:meth:`TextileStructure.phase_level_sets_at`; the adaptive route of
:mod:`morphogen.mesh.exact`, no background grid): the interfaces between the matrix and
the yarns, and between the yarns of different phases, are faces of the mesh, within its
Hausdorff distance, and each element takes the label of the segment at its centroid and
its orientation (``mesh.cell``, ``mesh.orientations``), with which the finite elements
turn the yarns along their paths as the FFT solver does on the voxels.

The domain: a periodic box holds whole repeats of the weave in its plane (an error names
the sizes that do), the fabric centred along the stacking ``axis`` in a box at least as
thick (the rest is resin, between the stacks of a periodic box); a non-periodic domain is
a window of the infinite fabric.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen._utils import check_fraction, check_positive
from morphogen.distributions import Constant, Distribution
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.kernels import textile as kernel
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.models.planar import plane_axes
from morphogen.seeding import SeedLike, make_rng

WEAVES_2D = ("plain", "twill", "satin", "basket")
WEAVES_3D = ("orthogonal", "layer_to_layer", "through_thickness")
WEAVES = WEAVES_2D + WEAVES_3D
SECTIONS = {"ellipse": kernel.ELLIPSE, "lenticular": kernel.LENTICULAR, "power": kernel.POWER}
FAMILIES = {1: "warp", 2: "weft", 3: "binder"}
#: The patterns of the weaves that have one, by default
PATTERNS = {"twill": (2, 2), "satin": (5, 2), "basket": (2,)}
#: Upper bounds on the cell grid of the yarns
_MAX_CELLS_PER_AXIS = 256
_MAX_CELLS = 2**22


def interlacing(weave: str, pattern=None) -> np.ndarray:
    """The interlacing matrix of a 2D weave: ``W[i, j]`` true where the warp ``i`` goes
    over the weft ``j``, over one repeat (``R_y`` warps by ``R_x`` wefts). ``plain``;
    ``twill`` ``[n, m]``: ``(j - i) mod (n + m) < n``; ``satin`` ``[H, s]``: under the weft
    ``j = s i mod H`` only; ``basket`` ``[k]``: the plain weave of groups of ``k``."""
    if weave not in WEAVES_2D:
        raise ValueError(f"interlacing: a 2D weave, one of {', '.join(WEAVES_2D)}")
    p = tuple(int(v) for v in (PATTERNS.get(weave, ()) if pattern is None else pattern))
    if weave == "plain":
        if p:
            raise ValueError("pattern: none for a plain weave")
        i, j = np.indices((2, 2))
        return (i + j) % 2 == 0
    if weave == "twill":
        if len(p) != 2 or min(p) < 1:
            raise ValueError("pattern of a twill: [over, under], two positive integers")
        r = p[0] + p[1]
        i, j = np.indices((r, r))
        return (j - i) % r < p[0]
    if weave == "satin":
        if len(p) != 2:
            raise ValueError("pattern of a satin: [harness, move], two integers")
        h, s = p
        if h < 5 or not 1 < s < h - 1 or math.gcd(h, s) != 1:
            raise ValueError(
                "pattern of a satin: a harness of 5 or more and a move between 2 and "
                "harness - 2, coprime with it (e.g. [5, 2], [8, 3])"
            )
        i, j = np.indices((h, h))
        return j != (s * i) % h
    if len(p) != 1 or p[0] < 1:
        raise ValueError("pattern of a basket weave: [k], the yarns woven together")
    k = p[0]
    i, j = np.indices((2 * k, 2 * k))
    return (i // k + j // k) % 2 == 0


def section_area(a, c, section: str = "ellipse", power: float = 2.0):
    """Area of the section of half width ``a`` and half thickness ``c``: ``pi a c`` (an
    ellipse), ``4 a c Gamma(1 + 1/q)^2 / Gamma(1 + 2/q)`` (a power ellipse of exponent
    ``q``), ``2 (R^2 acos(1 - c/R) - (R - c) a)`` with ``R = (a^2 + c^2) / (2 c)`` (a
    lens: two circular segments)."""
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    if section == "ellipse":
        return math.pi * a * c
    if section == "power":
        q = float(power)
        return 4.0 * a * c * math.gamma(1.0 + 1.0 / q) ** 2 / math.gamma(1.0 + 2.0 / q)
    if section == "lenticular":
        r = 0.5 * (a * a + c * c) / c
        return 2.0 * (r * r * np.arccos(np.clip(1.0 - c / r, -1.0, 1.0)) - (r - c) * a)
    raise ValueError(f"section: one of {', '.join(SECTIONS)}")


def _inside_section(u, v, a, c, section: str, power: float) -> np.ndarray:
    """Whether the points ``(u, v)`` lie in the section (numpy)."""
    if section == "lenticular":
        r = 0.5 * (a * a + c * c) / c
        e = r - c
        return (u * u + (v + e) ** 2 <= r * r) & (u * u + (v - e) ** 2 <= r * r)
    q = 2.0 if section == "ellipse" else float(power)
    return np.abs(u / a) ** q + np.abs(v / c) ** q <= 1.0


# -- the paths of the yarns ---------------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    """A piece of the path of a yarn in its vertical plane, from ``(s0, z0)`` to ``(s1,
    z1)`` (``s`` along the yarn, ``z`` along the stacking axis): a ``line``, a half
    ``cosine`` ``z0 + (z1 - z0) (1 - cos(pi (s - s0) / (s1 - s0))) / 2``, or an ``arc`` of
    the circle of centre ``(sc, zc)`` and radius ``r``, from the angle ``a0`` to ``a1``."""

    kind: str
    s0: float
    z0: float
    s1: float
    z1: float
    sc: float = 0.0
    zc: float = 0.0
    r: float = 0.0
    a0: float = 0.0
    a1: float = 0.0

    @property
    def length(self) -> float:
        """Arc length: the chord of a line, ``r |a1 - a0|`` of an arc, ``(2 L / pi)
        E(-(pi dz / (2 L))^2)`` of a half cosine of horizontal length ``L`` and rise
        ``dz`` (``E`` the complete elliptic integral of the second kind)."""
        if self.kind == "line":
            return math.hypot(self.s1 - self.s0, self.z1 - self.z0)
        if self.kind == "arc":
            return self.r * abs(self.a1 - self.a0)
        from scipy.special import ellipe

        span = self.s1 - self.s0
        k = math.pi * (self.z1 - self.z0) / (2.0 * span)
        return 2.0 * span / math.pi * float(ellipe(-k * k))

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``n + 1`` points of the piece, both ends included: ``s``, ``z`` and the unit
        tangent ``(ts, tz)``."""
        w = np.linspace(0.0, 1.0, n + 1)
        if self.kind == "line":
            ds, dz = self.s1 - self.s0, self.z1 - self.z0
            norm = math.hypot(ds, dz)
            return (self.s0 + w * ds, self.z0 + w * dz, np.full(n + 1, ds / norm),
                    np.full(n + 1, dz / norm))  # fmt: skip
        if self.kind == "arc":
            a = self.a0 + w * (self.a1 - self.a0)
            sign = 1.0 if self.a1 > self.a0 else -1.0
            return (self.sc + self.r * np.cos(a), self.zc + self.r * np.sin(a),
                    -sign * np.sin(a), sign * np.cos(a))  # fmt: skip
        span = self.s1 - self.s0
        s = self.s0 + w * span
        z = self.z0 + (self.z1 - self.z0) * 0.5 * (1.0 - np.cos(math.pi * w))
        slope = (self.z1 - self.z0) * math.pi / (2.0 * span) * np.sin(math.pi * w)
        norm = np.sqrt(1.0 + slope * slope)
        return s, z, 1.0 / norm, slope / norm


def _rounded(s0: float, z0: float, s1: float, z1: float, radius: float) -> list[Piece]:
    """A rise from ``(s0, z0)`` to ``(s1, z1)``, level at both ends, by an arc of
    ``radius``, a line and an arc (two arcs when the rise is too low for a line): the
    angle ``theta`` of the line solves ``2 r + (|dz| - 2 r) cos(theta) = L sin(theta)``."""
    span, rise = s1 - s0, abs(z1 - z0)
    sign = 1.0 if z1 > z0 else -1.0
    r = min(float(radius), 0.5 * span)
    big = math.hypot(span, rise - 2.0 * r)
    theta = math.atan2(rise - 2.0 * r, span) + math.asin(min(1.0, 2.0 * r / big))
    line = (rise - 2.0 * r * (1.0 - math.cos(theta))) / math.sin(theta)
    if line < 0.0:  # two arcs, tangent where they meet
        theta = 2.0 * math.atan2(rise, span)
        r = span / (2.0 * math.sin(theta))
        line = 0.0
    q = 0.5 * math.pi
    first = Piece("arc", s0, z0, s0 + r * math.sin(theta), z0 + sign * r * (1.0 - math.cos(theta)),
                  s0, z0 + sign * r, r, -sign * q, -sign * q + sign * theta)  # fmt: skip
    second_start = (s1 - r * math.sin(theta), z1 - sign * r * (1.0 - math.cos(theta)))
    out = [first]
    if line > 1e-12 * span:
        out.append(Piece("line", first.s1, first.z1, *second_start))
    out.append(Piece("arc", *second_start, s1, z1, s1, z1 - sign * r, r,
                     sign * q + sign * theta, sign * q))  # fmt: skip
    return out


def profile(levels, first: float, spacing: float, transition: float = 1.0,
            radius: float | None = None) -> list[Piece]:  # fmt: skip
    """The path through stations ``first + k spacing`` at ``levels[k]``: level over the
    stations, from one level to the next over a ``transition`` share of the spacing in the
    middle of the interval, by a half cosine (or, with a ``radius``, by arcs of that
    radius and a line, horizontal at both ends); consecutive level pieces merged."""
    pieces: list[Piece] = []
    levels = [float(v) for v in levels]
    flat = 0.5 * (1.0 - transition) * spacing
    for k in range(len(levels) - 1):
        sa = first + k * spacing
        sb = sa + spacing
        za, zb = levels[k], levels[k + 1]
        if abs(zb - za) <= 1e-12 * spacing:
            pieces.append(Piece("line", sa, za, sb, za))
            continue
        p0, p1 = sa + flat, sb - flat
        if flat > 0.0:
            pieces.append(Piece("line", sa, za, p0, za))
        if radius is None:
            pieces.append(Piece("cosine", p0, za, p1, zb))
        else:
            pieces.extend(_rounded(p0, za, p1, zb, radius))
        if flat > 0.0:
            pieces.append(Piece("line", p1, zb, sb, zb))
    merged: list[Piece] = []
    for piece in pieces:
        last = merged[-1] if merged else None
        if (last is not None and piece.kind == "line" and last.kind == "line"
                and piece.z0 == piece.z1 == last.z1 == last.z0):  # fmt: skip
            merged[-1] = Piece("line", last.s0, last.z0, piece.s1, piece.z1)
        else:
            merged.append(piece)
    return merged


def path_length(pieces) -> float:
    """Arc length of a path (exact, piece by piece)."""
    return float(sum(p.length for p in pieces))


@dataclass
class _Path:
    """A yarn before its sampling: its phase, the canonical axis along which it runs (0:
    the warp direction, 1: the weft direction), its transverse coordinate, its pieces,
    whether it closes on itself over the period of the box, its half sizes, and its group
    (its family and its ply or layer: the yarns side by side whose variations correlate)."""

    phase: int
    travel: int
    fixed: float
    pieces: list
    periodic: bool
    period: float
    half_width: float
    half_thickness: float
    group: tuple = ()

    @property
    def straight(self) -> float:
        return self.pieces[-1].s1 - self.pieces[0].s0

    @property
    def crimp(self) -> float:
        return path_length(self.pieces) / self.straight - 1.0


# -- the variability of the yarns -------------------------------------------------------------

COVARIANCES = ("gaussian", "exponential")
#: The quantities that vary along the yarns: the deviations of the centrelines across the
#: yarns in the plane of the fabric and through its thickness, the widths, the thicknesses
VARIATIONS = ("in_plane", "through", "width", "thickness")
#: Segments of a varying yarn per correlation length (at least)
_PER_LENGTH = 6


def process_spectrum(period: float, length: float, covariance: str = "gaussian",
                     modes: int | None = None) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """The angular frequencies ``omega_k = 2 pi k / period`` (``k = 0, ..., K``) and the
    weights ``w_k`` of a stationary Gaussian process of unit variance on a circle of
    perimeter ``period`` whose covariance is the periodization of ``exp(-d^2 / 2 length^2)``
    (``gaussian``) or of ``exp(-|d| / length)`` (``exponential``): ``w_0 = c_0`` and ``w_k =
    2 c_k``, ``c_k = C^(omega_k) / period`` the Fourier coefficients of the periodized
    covariance (``C^`` the Fourier transform of the covariance: ``length sqrt(2 pi)
    exp(-(omega length)^2 / 2)``, ``2 length / (1 + (omega length)^2)``), truncated to ``K``
    modes and normalized to sum to 1. By default ``K`` reaches ``omega length = 7.2`` for the
    gaussian covariance (the weights left out below 10^-11) and three modes per correlation
    length for the exponential one (the Nyquist frequency of segments of a sixth of it). The
    process ``G(s) = sum_k sqrt(w_k) (A_k cos(omega_k s) + B_k sin(omega_k s))``, ``A_k``,
    ``B_k`` independent standard normal variables, has the covariance ``sum_k w_k cos(omega_k
    d)`` (:func:`process_covariance`)."""
    if covariance not in COVARIANCES:
        raise ValueError(f"covariance: one of {', '.join(COVARIANCES)}")
    period, length = float(period), float(length)
    if modes is None:
        if covariance == "gaussian":
            modes = math.ceil(7.2 * period / (2.0 * math.pi * length))
        else:
            modes = math.ceil(0.5 * _PER_LENGTH * period / length)
    k = np.arange(int(min(max(modes, 1), 4096)) + 1)
    omega = 2.0 * math.pi * k / period
    if covariance == "gaussian":
        spectrum = length * math.sqrt(2.0 * math.pi) * np.exp(-0.5 * (omega * length) ** 2)
    else:
        spectrum = 2.0 * length / (1.0 + (omega * length) ** 2)
    weights = np.where(k == 0, 1.0, 2.0) * spectrum / period
    return omega, weights / weights.sum()


def process_covariance(spectrum, d) -> np.ndarray:
    """The covariance ``sum_k w_k cos(omega_k d)`` of the process of a spectrum
    (:func:`process_spectrum`) at the lags ``d``."""
    omega, weights = spectrum
    d = np.asarray(d, dtype=float)
    return np.cos(d[..., None] * omega) @ weights


def process_values(spectrum, coefficients, s) -> tuple[np.ndarray, np.ndarray]:
    """The process ``G(s)`` of a spectrum and its coefficients ``(A_k, B_k)`` (rows), and its
    derivative, at the coordinates ``s``."""
    omega, weights = spectrum
    amplitude = np.sqrt(weights)[:, None] * np.asarray(coefficients, dtype=float)
    phase = np.asarray(s, dtype=float)[:, None] * omega
    cos, sin = np.cos(phase), np.sin(phase)
    values = cos @ amplitude[:, 0] + sin @ amplitude[:, 1]
    slopes = (cos @ (omega * amplitude[:, 1])) - (sin @ (omega * amplitude[:, 0]))
    return values, slopes


@dataclass
class _Variation:
    """The variations of one yarn as functions of its coordinate ``s`` along its direction
    (from ``origin``): the spectra of the centrelines and of the sections, the coefficients
    of the processes of :data:`VARIATIONS` (unit variance) and their scales (the standard
    deviations of the deviations; the standard deviations of the logarithms of the
    sizes)."""

    origin: float
    centrelines: tuple
    sections: tuple
    coefficients: dict
    scales: dict

    def at(self, s):
        """The deviations across the yarn and through the thickness, their derivatives along
        ``s``, and the factors of the width and of the thickness (lognormal, of mean 1) at
        the coordinates ``s``."""
        s = np.asarray(s, dtype=float) - self.origin
        out = []
        for name in ("in_plane", "through"):
            g, dg = process_values(self.centrelines, self.coefficients[name], s)
            out.append((self.scales[name] * g, self.scales[name] * dg))
        factors = []
        for name in ("width", "thickness"):
            sigma = self.scales[name]
            g, _ = process_values(self.sections, self.coefficients[name], s)
            factors.append(np.exp(sigma * g - 0.5 * sigma * sigma))
        (dn, gn), (dz, gz) = out
        return dn, dz, gn, gz, factors[0], factors[1]


def neighbour_correlations(n: int, rho: float, periodic: bool) -> np.ndarray:
    """The correlations of the variations of ``n`` yarns side by side, ``rho^d`` for yarns
    ``d`` apart; in a periodic box, where the last is the neighbour of the first, its
    periodization ``(rho^d + rho^(n - d)) / (1 + rho^n)`` (``d`` the distance on the ring)."""
    i = np.arange(int(n))
    d = np.abs(i[:, None] - i[None, :]).astype(float)
    if rho <= 0.0:
        return np.eye(len(i))
    if periodic and n > 1:
        d = np.minimum(d, n - d)
        return (rho**d + rho ** (n - d)) / (1.0 + rho**n)
    return rho**d


# -- the realizations -------------------------------------------------------------------------


@dataclass
class Yarn:
    """One yarn of a realization: its phase (1 warp, 2 weft, 3 binder; 1 all of them
    when merged) and family, the nodes of its centreline (m x 3, in the axes of the
    sample), the width axis ``normals`` and the thickness axis ``binormals`` at the nodes,
    the translation that closes a periodic yarn on itself (zeros: an open yarn, the
    polyline of its nodes), its half width and half thickness, the crimp of its path and,
    for the sections that vary along the yarn, the half widths ``widths`` and half
    thicknesses ``thicknesses`` at its nodes (None: the constant half sizes; between two
    nodes the sizes vary linearly)."""

    family: str
    phase: int
    nodes: np.ndarray
    normals: np.ndarray
    binormals: np.ndarray
    period: np.ndarray
    half_width: float
    half_thickness: float
    crimp: float = 0.0
    widths: np.ndarray | None = None
    thicknesses: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.nodes = np.ascontiguousarray(self.nodes, dtype=float).reshape(-1, 3)
        self.normals = np.ascontiguousarray(self.normals, dtype=float).reshape(-1, 3)
        self.binormals = np.ascontiguousarray(self.binormals, dtype=float).reshape(-1, 3)
        self.period = np.asarray(self.period, dtype=float).reshape(3)
        self.half_width = float(self.half_width)
        self.half_thickness = float(self.half_thickness)
        for name in ("widths", "thicknesses"):
            value = getattr(self, name)
            if value is not None:
                value = np.ascontiguousarray(value, dtype=float).reshape(-1)
                if len(value) != len(self.nodes) or not np.all(value > 0.0):
                    raise ValueError(f"{name}: one positive half size per node of the yarn")
                setattr(self, name, value)

    @property
    def varies(self) -> bool:
        """Whether its section varies along the yarn."""
        return self.widths is not None or self.thicknesses is not None

    @property
    def half_widths(self) -> np.ndarray:
        """The half width at each node."""
        if self.widths is None:
            return np.full(len(self.nodes), self.half_width)
        return self.widths

    @property
    def half_thicknesses(self) -> np.ndarray:
        """The half thickness at each node."""
        if self.thicknesses is None:
            return np.full(len(self.nodes), self.half_thickness)
        return self.thicknesses

    def segment_sizes(self) -> tuple[np.ndarray, np.ndarray]:
        """The half widths and half thicknesses at the two ends of each segment (one row per
        segment: start, end)."""
        nxt = np.roll(np.arange(len(self.nodes)), -1)
        a, c = self.half_widths, self.half_thicknesses
        aw = np.stack([a, a[nxt]], axis=1)
        ct = np.stack([c, c[nxt]], axis=1)
        if not self.closed:
            aw, ct = aw[:-1], ct[:-1]
        return np.ascontiguousarray(aw), np.ascontiguousarray(ct)

    @property
    def closed(self) -> bool:
        return bool(np.any(self.period != 0.0))

    @property
    def n_segments(self) -> int:
        return len(self.nodes) if self.closed else len(self.nodes) - 1

    def ends(self) -> tuple[np.ndarray, ...]:
        """The two ends of each segment and their frames: start, end, n at both, b at
        both."""
        nxt = np.roll(np.arange(len(self.nodes)), -1)
        start, end = self.nodes, self.nodes[nxt]
        n0, n1 = self.normals, self.normals[nxt]
        b0, b1 = self.binormals, self.binormals[nxt]
        if self.closed:
            end = end.copy()
            end[-1] += self.period
            return start, end, n0, n1, b0, b1
        return start[:-1], end[:-1], n0[:-1], n1[:-1], b0[:-1], b1[:-1]

    @property
    def length(self) -> float:
        """Length of the polyline (one period of a periodic yarn)."""
        start, end = self.ends()[:2]
        return float(np.linalg.norm(end - start, axis=1).sum())

    def to_dict(self) -> dict:
        out = {
            "family": self.family,
            "phase": self.phase,
            "nodes": self.nodes.tolist(),
            "normals": self.normals.tolist(),
            "binormals": self.binormals.tolist(),
            "period": self.period.tolist(),
            "half_width": self.half_width,
            "half_thickness": self.half_thickness,
            "crimp": self.crimp,
        }
        for name in ("widths", "thicknesses"):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name).tolist()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> Yarn:
        return cls(**{k: data[k] for k in ("family", "phase", "nodes", "normals",
                                           "binormals", "period", "half_width",
                                           "half_thickness")},
                   crimp=float(data.get("crimp", 0.0)), widths=data.get("widths"),
                   thicknesses=data.get("thicknesses"))  # fmt: skip


class _Accelerator:
    """The segments of the yarns (and their periodic images) in a grid of cells, for the
    kernels of :mod:`morphogen.kernels.textile`."""

    def __init__(self, micro: TextileStructure, band: float) -> None:
        from morphogen.kernels.raster import build_cells

        domain = micro.domain
        parts = []
        label = 0
        for k, yarn in enumerate(micro.yarns):
            start, end, n0, n1, b0, b1 = yarn.ends()
            m = len(start)
            half_w, half_t = yarn.segment_sizes()
            parts.append((start, end, n0, b0, n1, b1, np.full(m, k), half_w, half_t,
                          np.arange(label, label + m), np.full(m, yarn.phase)))  # fmt: skip
            label += m
        arrays = [np.concatenate(a) for a in zip(*parts, strict=True)]
        start, end = arrays[0], arrays[1]
        reach = np.maximum(arrays[7].max(axis=1), arrays[8].max(axis=1)) + float(band)
        lo = np.minimum(start, end) - reach[:, None]
        hi = np.maximum(start, end) + reach[:, None]
        if domain.periodic:
            lengths = domain.lengths
            span = np.max((hi - lo) / lengths)
            m = int(np.ceil(span)) + 1
            lower, upper = domain.lower - band, domain.upper + band
            pieces, shifts = [], []
            for shift in itertools.product(range(-m, m + 1), repeat=3):
                s = np.array(shift, dtype=float) * lengths
                keep = np.flatnonzero(np.all((hi + s >= lower) & (lo + s <= upper), axis=1))
                if len(keep):
                    pieces.append(keep)
                    shifts.append(np.repeat(s[None], len(keep), axis=0))
            index = np.concatenate(pieces)
            offset = np.concatenate(shifts)
            arrays = [a[index] for a in arrays]
            arrays[0] = arrays[0] + offset
            arrays[1] = arrays[1] + offset
            reach = reach[index]
            box_lo, box_hi = domain.lower - band, domain.upper + band
        else:
            margin = float(np.max(reach)) if len(reach) else float(band)
            box_lo, box_hi = domain.lower - margin, domain.upper + margin
        order = np.argsort(arrays[6], kind="stable")  # the segments of a yarn consecutive
        arrays = [np.ascontiguousarray(a[order]) for a in arrays]
        reach = np.ascontiguousarray(reach[order])
        (self.start, self.end, self.na, self.ba, self.nb, self.bb, yarn, half_w, half_t,
         labels, phase) = arrays  # fmt: skip
        self.yarn = yarn.astype(np.int64)
        self.half_w, self.half_t = half_w.astype(float), half_t.astype(float)
        self.label = labels.astype(np.int64)
        self.phase = phase.astype(np.int64)
        self.reach = reach.astype(float)
        self.band = float(band)
        lengths = np.asarray(box_hi - box_lo, dtype=float)
        typical = float(np.median(np.maximum(half_w.max(axis=1), half_t.max(axis=1)))
                        if len(half_w) else 1.0)  # fmt: skip
        n = np.clip(np.floor(lengths / max(typical, 1e-12)), 1, _MAX_CELLS_PER_AXIS)
        n = n.astype(np.int64)
        while np.prod(n) > _MAX_CELLS:
            n = np.maximum(1, n // 2)
        self.cell_n = n
        self.cell_lo = np.asarray(box_lo, dtype=float).copy()
        self.cell_h = lengths / n
        self.offsets, self.items = build_cells(
            "segments", (self.start, self.end, self.reach), 0.0, self.cell_lo, self.cell_h, n
        )

    def args(self, micro: TextileStructure) -> tuple:
        return (self.offsets, self.items, self.start, self.end, self.na, self.ba, self.nb,
                self.bb, self.yarn, self.half_w, self.half_t, self.reach, self.label,
                self.phase, micro.section_code, micro.power)  # fmt: skip


@register_kind
class TextileStructure(Microstructure):
    """Yarns swept along polylines (:class:`Yarn`), a partition of their union by the
    functions of their sections, in a matrix (phase 0). The label of each segment of the
    yarns is its index (the segments of the yarns one after another), that of the matrix
    the last (:attr:`n_labels` - 1); :attr:`orientations` turns the axes of the material
    of each segment: the axis 3 along it, the axis 1 across the width of its yarn."""

    kind = "woven_textile"
    has_labels = True
    #: The functions of two yarns that interpenetrate meet by their difference
    lipschitz = 2.0

    def __init__(self, domain, yarns, section: str = "ellipse", power: float = 2.0, **kwargs):
        super().__init__(domain, **kwargs)
        self.yarns = [y if isinstance(y, Yarn) else Yarn.from_dict(y) for y in yarns]
        if section not in SECTIONS:
            raise ValueError(f"section: one of {', '.join(SECTIONS)}")
        self.section = section
        self.section_code = SECTIONS[section]
        self.power = float(power)
        self._accelerators: dict = {}
        self._functions: dict = {}
        self._orientations = None

    # -- the segments ----------------------------------------------------------------------

    @property
    def n_yarns(self) -> int:
        return len(self.yarns)

    @property
    def n_segments(self) -> int:
        return int(sum(y.n_segments for y in self.yarns))

    @property
    def n_labels(self) -> int:
        """The labels: one per segment, and the matrix."""
        return self.n_segments + 1

    @property
    def matrix_label(self) -> int:
        return self.n_segments

    def _accelerator(self, band: float) -> _Accelerator:
        key = round(float(band), 12)
        if key not in self._accelerators:
            self._accelerators.clear()
            self._accelerators[key] = _Accelerator(self, float(band))
        return self._accelerators[key]

    @property
    def orientations(self) -> np.ndarray:
        """Bunge angles (degrees) of the axes of the material of each label: of a segment,
        the axis 3 along it, the axis 1 along the width axis of its yarn (normal to the
        segment), the axis 2 through its thickness; zeros for the matrix."""
        if self._orientations is None:
            from morphogen.orientations import matrix_to_euler

            out = np.zeros((self.n_labels, 3))
            k = 0
            for yarn in self.yarns:
                start, end, n0, n1 = yarn.ends()[:4]
                t = end - start
                t /= np.linalg.norm(t, axis=1, keepdims=True)
                n = n0 + n1
                n -= np.einsum("ij,ij->i", n, t)[:, None] * t
                n /= np.linalg.norm(n, axis=1, keepdims=True)
                g = np.stack([n, np.cross(t, n), t], axis=1)  # rows: the axes 1, 2, 3
                out[k : k + len(t)] = matrix_to_euler(g)
                k += len(t)
            self._orientations = out
        return self._orientations

    @property
    def cell_phases(self) -> np.ndarray:
        """The phase of each label (uint8)."""
        out = np.zeros(self.n_labels, dtype=np.uint8)
        k = 0
        for yarn in self.yarns:
            out[k : k + yarn.n_segments] = yarn.phase
            k += yarn.n_segments
        return out

    def implicit_phases(self) -> list[int]:
        return [0, *sorted({int(y.phase) for y in self.yarns})]

    @property
    def phase_names(self) -> dict[int, str]:
        """The names of the phases: those of the model (yarns merged or not), else of the
        families of the yarns."""
        if self.model is not None:
            return dict(self.model.active_phases())
        names = {0: "matrix"}
        for yarn in self.yarns:
            names.setdefault(int(yarn.phase), yarn.family)
        return dict(sorted(names.items()))

    @property
    def _max_phase(self) -> int:
        return max((int(y.phase) for y in self.yarns), default=1)

    # -- evaluations -----------------------------------------------------------------------

    def _points(self, points, band: float):
        """Phase, label, functions of the phases of the yarns (phases x points) and the
        number of yarns at points."""
        pts = np.ascontiguousarray(points, dtype=float).reshape(-1, 3)
        if self.domain.periodic:
            pts = np.ascontiguousarray(self.domain.wrap(pts))
        n = len(pts)
        phase = np.zeros(n, dtype=np.uint8)
        label = np.zeros(n, dtype=np.int32)
        depth = np.zeros((self._max_phase, n), dtype=float)
        count = np.zeros(n, dtype=np.int32)
        if not self.yarns:
            label[:] = self.matrix_label
            depth[:] = band
            return phase, label, depth, count
        acc = self._accelerator(band)
        kernel.yarn_points(*acc.args(self), float(band), self.matrix_label, acc.cell_lo,
                           acc.cell_h, acc.cell_n, pts, phase, label, depth, count)  # fmt: skip
        return phase, label, depth, count

    def _grid(self, grid: Grid, band: float, *, phase=False, label=False, depth=False,
              count=False):  # fmt: skip
        """The arrays of the kernel at the voxel centres of a grid (those asked)."""
        shape = tuple(int(n) for n in grid.shape)

        def array(wanted, dims, dtype):
            return np.zeros(dims if wanted else (0,) * len(dims), dtype=dtype)

        out_phase = array(phase, shape, np.uint8)
        out_label = array(label, shape, np.int32)
        out_depth = array(depth, (self._max_phase, *shape), np.float32)
        out_count = array(count, shape, np.uint8)
        if not self.yarns:
            out_label[...] = self.matrix_label
            out_depth[...] = band
            return out_phase, out_label, out_depth, out_count
        acc = self._accelerator(band)
        kernel.yarn_grid(*acc.args(self), float(band), self.matrix_label, acc.cell_lo,
                         acc.cell_h, acc.cell_n, np.asarray(grid.origin, dtype=float),
                         np.asarray(grid.spacing, dtype=float), np.asarray(shape, np.int64),
                         out_phase, out_label, out_depth, out_count)  # fmt: skip
        return out_phase, out_label, out_depth, out_count

    @staticmethod
    def _combined(depth: np.ndarray, phases: list[int]) -> dict[int, np.ndarray]:
        """The functions of the phases from the smallest functions of the yarns of each
        phase ``D_p``: the partition by the smallest of them, the matrix that of the
        constant 0: ``phi_0 = -min_p D_p``, ``phi_p = D_p - min(0, min_(q != p) D_q)``."""
        lowest = np.min(depth, axis=0)
        out = {0: -lowest}
        for p in phases:
            if p == 0:
                continue
            others = [depth[q - 1] for q in phases if q not in (0, p)]
            rival = np.minimum(0.0, np.min(others, axis=0)) if others else 0.0
            out[p] = depth[p - 1] - rival
        return out

    def phase_of(self, points) -> np.ndarray:
        """The phase of each point (uint8)."""
        return self._points(points, 0.0)[0]

    def label_of(self, points) -> np.ndarray:
        """The label of each point (int32): its segment, or the matrix."""
        return self._points(points, 0.0)[1]

    def yarns_at(self, points) -> np.ndarray:
        """The number of yarns that hold each point (2 or more where they interpenetrate)."""
        return self._points(points, 0.0)[3]

    def contains(self, points) -> np.ndarray:
        return self.phase_of(points) == 1

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if ss == 1:
            data = self._grid(grid, 0.0, phase=True)[0]
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        counts = np.zeros(grid.shape, dtype=np.int64)
        if self.yarns:
            acc = self._accelerator(0.0)
            kernel.yarn_count(*acc.args(self), 1, acc.cell_lo, acc.cell_h, acc.cell_n,
                              np.asarray(grid.origin, dtype=float),
                              np.asarray(grid.spacing, dtype=float),
                              np.asarray(grid.shape, np.int64), ss, counts)  # fmt: skip
        data = (counts / float(ss**3)).astype(np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def labels(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Image of the labels (int32): the segment of the yarn that holds each voxel, the
        matrix label elsewhere."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        data = self._grid(grid, 0.0, label=True)[1]
        return VoxelImage.from_grid(data, grid, name="cell", metadata=self.image_metadata())

    def yarn_count(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Image of the number of yarns that hold each voxel (uint8)."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        data = self._grid(grid, 0.0, count=True)[3]
        return VoxelImage.from_grid(data, grid, name="yarns", metadata=self.image_metadata())

    def _grid_functions(self, grid: Grid) -> dict[int, np.ndarray]:
        from morphogen.models.implicit import band_of, grid_key

        key = grid_key(grid)
        if key not in self._functions:
            band = band_of(grid)
            depth = self._grid(grid, band, depth=True)[2]
            out = self._combined(depth, self.implicit_phases())
            self._functions.clear()
            self._functions[key] = {
                k: np.clip(v, -band, band).astype(np.float32) for k, v in out.items()
            }
        return self._functions[key]

    def phase_level_set(
        self, phase: int, shape=None, *, voxel_size=None, grid: Grid | None = None
    ) -> VoxelImage:
        """The implicit function of a phase (see :meth:`Microstructure.phase_level_set`):
        from the functions of the sections of the yarns, about the distances to their
        boundaries, the smallest of those of each phase ``D_p``: ``-min_p D_p`` for the
        matrix, ``D_p - min(0, min_(q != p) D_q)`` for the yarns of a phase (zero where
        their yarns meet those of another phase as deep), clipped to the band of the
        grid."""
        phase = int(phase)
        phases = self.implicit_phases()
        if phase not in phases:
            raise ValueError(f"phase {phase}: the phases of this textile are {phases}")
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        data = self._grid_functions(grid)[phase]
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """The implicit function of phase 1 (:meth:`phase_level_set`): the warp yarns, or
        all the yarns when they are merged."""
        return self.phase_level_set(1, shape, voxel_size=voxel_size, grid=grid)

    def phase_level_sets_at(self, points, reach: float = np.inf) -> dict[int, np.ndarray]:
        """The functions of :meth:`phase_level_set` at the points, clipped to ``[-reach,
        reach]`` (by default the largest half size of the yarns)."""
        if not np.isfinite(reach):
            reach = max((max(y.half_width, y.half_thickness) for y in self.yarns), default=1.0)
        reach = float(reach)
        depth = self._points(points, reach)[2]
        out = self._combined(depth, self.implicit_phases())
        return {k: np.clip(v, -reach, reach) for k, v in out.items()}

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """The function of phase 1 at the points (:meth:`phase_level_sets_at`)."""
        return self.phase_level_sets_at(points, reach)[1]

    # -- description -----------------------------------------------------------------------

    def _segment_volumes(self, yarn: Yarn) -> np.ndarray:
        """The volume of each segment of a yarn of varying section: its length times the
        mean area of its section, whose half sizes vary linearly along it (Gauss–Legendre,
        four points)."""
        start, end = yarn.ends()[:2]
        lengths = np.linalg.norm(end - start, axis=1)
        aw, ct = yarn.segment_sizes()
        x, w = np.polynomial.legendre.leggauss(4)
        t = 0.5 * (x + 1.0)
        a = aw[:, :1] * (1.0 - t) + aw[:, 1:] * t
        c = ct[:, :1] * (1.0 - t) + ct[:, 1:] * t
        area = section_area(a, c, self.section, self.power)
        return lengths * (0.5 * area @ w)

    def volumes(self) -> np.ndarray:
        """The volume of each yarn: the area of its section times the length of its
        polyline (one period of a periodic yarn), overlaps counted twice; segment by
        segment for the sections that vary along the yarns."""
        out = []
        for y in self.yarns:
            if y.varies:
                out.append(float(self._segment_volumes(y).sum()))
            else:
                area = section_area(y.half_width, y.half_thickness, self.section, self.power)
                out.append(float(area) * y.length)
        return np.array(out)

    def sampled(self, points: int = 200_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """For each yarn, the share of its volume (:meth:`volumes`) in the domain and the
        share of that part which another yarn holds, from points drawn uniformly in it
        (along its segments by their lengths, and in its section; a fixed ``seed``, apart
        from that of the realization). The yarns of a periodic box are whole in it."""
        n = self.n_yarns
        inside, shared = np.ones(n), np.zeros(n)
        if not n:
            return inside, shared
        rng = np.random.default_rng(seed)
        volumes = self.volumes()
        total = float(volumes.sum())
        for k, (yarn, volume) in enumerate(zip(self.yarns, volumes, strict=True)):
            m = max(64, round(points * volume / total))
            start, end, n0, n1, b0, b1 = yarn.ends()
            lengths = np.linalg.norm(end - start, axis=1)
            if yarn.varies:
                s, t, u, v = self._draw_varying(yarn, m, rng)
                t = t[:, None]
            else:
                a, c = yarn.half_width, yarn.half_thickness
                u = v = np.empty(0)
                while len(u) < m:
                    du = rng.uniform(-a, a, 2 * m)
                    dv = rng.uniform(-c, c, 2 * m)
                    keep = _inside_section(du, dv, a, c, self.section, self.power)
                    u, v = np.concatenate([u, du[keep]]), np.concatenate([v, dv[keep]])
                u, v = u[:m], v[:m]
                s = rng.choice(len(lengths), size=m, p=lengths / lengths.sum())
                t = rng.random(m)[:, None]
            normal = (1.0 - t) * n0[s] + t * n1[s]
            binormal = (1.0 - t) * b0[s] + t * b1[s]
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            binormal /= np.linalg.norm(binormal, axis=1, keepdims=True)
            pts = start[s] + t * (end[s] - start[s]) + u[:, None] * normal + v[:, None] * binormal
            if not self.domain.periodic:
                within = self.domain.contains(pts)
                inside[k] = float(np.mean(within))
                pts = pts[within]
            if len(pts):
                shared[k] = float(np.mean(self.yarns_at(pts) >= 2))
        return inside, shared

    def _draw_varying(self, yarn: Yarn, m: int, rng):
        """``m`` points uniform in a yarn of varying section: the segment by its volume, the
        position along it by the area of its section (rejection under the area of its
        largest sizes), the point in the section there (rejection in its bounding box).
        Returns the segments, the positions along them and the coordinates in the
        sections."""
        volumes = self._segment_volumes(yarn)
        aw, ct = yarn.segment_sizes()
        top = section_area(aw.max(axis=1), ct.max(axis=1), self.section, self.power)
        segments, positions = np.empty(0, dtype=np.int64), np.empty(0)
        while len(segments) < m:
            s = rng.choice(len(volumes), size=2 * m, p=volumes / volumes.sum())
            t = rng.random(2 * m)
            a = aw[s, 0] * (1.0 - t) + aw[s, 1] * t
            c = ct[s, 0] * (1.0 - t) + ct[s, 1] * t
            keep = rng.random(2 * m) * top[s] <= section_area(a, c, self.section, self.power)
            segments = np.concatenate([segments, s[keep]])
            positions = np.concatenate([positions, t[keep]])
        s, t = segments[:m], positions[:m]
        a = aw[s, 0] * (1.0 - t) + aw[s, 1] * t
        c = ct[s, 0] * (1.0 - t) + ct[s, 1] * t
        u, v = np.zeros(m), np.zeros(m)
        missing = np.arange(m)
        while len(missing):
            du = rng.uniform(-1.0, 1.0, len(missing)) * a[missing]
            dv = rng.uniform(-1.0, 1.0, len(missing)) * c[missing]
            keep = _inside_section(du, dv, a[missing], c[missing], self.section, self.power)
            u[missing[keep]], v[missing[keep]] = du[keep], dv[keep]
            missing = missing[~keep]
        return s, t, u, v

    def interpenetration(self, points: int = 200_000, seed: int = 0) -> float:
        """The share of the volume of the yarns in the domain where two of them overlap,
        ``sum_k V_k f_k / (2 sum_k V_k)`` over the parts ``V_k`` of the yarns in the domain,
        ``f_k`` the share of each that another yarn holds (:meth:`sampled`): the fraction
        of the yarns is their nominal fraction times one minus it."""
        inside, shared = self.sampled(points, seed)
        volumes = self.volumes() * inside
        total = float(volumes.sum())
        return float(np.sum(volumes * shared) / (2.0 * total)) if total > 0 else 0.0

    def summary(self) -> dict:
        out = {"n_yarns": self.n_yarns, "n_segments": self.n_segments}
        for phase_name in FAMILIES.values():
            n = sum(1 for y in self.yarns if y.family == phase_name)
            if n:
                out[f"n_{phase_name}"] = n
        out.update(self.metadata.get("summary", {}))
        return out

    def _geometry(self) -> dict:
        return {
            "section": self.section,
            "power": self.power,
            "yarns": [y.to_dict() for y in self.yarns],
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            [Yarn.from_dict(y) for y in geometry["yarns"]],
            section=geometry.get("section", "ellipse"),
            power=geometry.get("power", 2.0),
            **kwargs,
        )


# -- the model ------------------------------------------------------------------------------------


def _mean(d: Distribution | None, default: float) -> float:
    return default if d is None else float(d.moment(1))


def _realized(infos: list) -> dict[str, float]:
    """The statistics of the variations of the yarns at their nodes, each weighted by its
    share of the length along the direction of its yarn: the RMS deviations, the
    coefficients of variation of the widths and thicknesses, and the correlation of their
    logarithms."""
    infos = [i for i in infos if i is not None]
    w = np.concatenate([i["weights"] for i in infos])
    total = float(w.sum())

    def mean(x):
        return float(np.sum(w * x) / total)

    values = {k: np.concatenate([i[k] for i in infos]) for k in VARIATIONS}
    out = {f"deviation_{k}": math.sqrt(mean(values[k] ** 2)) for k in ("in_plane", "through")}
    logs = {}
    for k in ("width", "thickness"):
        f = values[k]
        out[f"{k}_variation"] = math.sqrt(max(mean((f - mean(f)) ** 2), 0.0)) / mean(f)
        logs[k] = np.log(f) - mean(np.log(f))
    sw, st = (math.sqrt(mean(logs[k] ** 2)) for k in ("width", "thickness"))
    if sw > 0.0 and st > 0.0:
        out["size_correlation"] = mean(logs["width"] * logs["thickness"]) / (sw * st)
    return out


@register
@dataclass(frozen=True)
class WovenTextile(Model):
    """Woven textile composite: warp and weft yarns (and binders) swept along the crimped
    paths of a 2D or 3D weave in a matrix (see the module).

    ``weave``: plain, twill, satin or basket (2D: ``layers`` plies), orthogonal,
    layer_to_layer or through_thickness (3D: ``layers`` warp layers, ``layers + 1`` weft
    layers, binders). ``warp_spacing``, ``weft_spacing``: the distances between the axes of
    neighbouring warps (along the weft) and wefts (along the warp). The sizes of the yarns
    are numbers (the same yarns) or distributions (one size per yarn); their means set the
    levels of the paths, the thickness of a ply (``t_warp + t_weft``) and of a 3D fabric
    (``N (t_warp + t_weft) + t_weft + 2 t_binder``; ``t_binder`` for ``t_warp`` without
    stuffers). ``binder_depth``: the weft layers that a binder ties together (orthogonal
    and layer-to-layer weaves; Boussu, Cristian & Nauman 2015); ``stuffers``: the straight
    warps of the angle interlocks.

    The variability of the yarns along their paths (zero by default): the deviations of
    their centrelines (``deviation_in_plane``, ``deviation_through``: standard deviations)
    and the variations of the widths and thicknesses of their sections
    (``width_variation``, ``thickness_variation``: coefficients of variation, lognormal,
    of correlation ``size_correlation`` at a point), stationary Gaussian processes along
    the yarns of correlation length ``correlation_length`` and covariance
    ``variation_covariance``, periodic over the box, correlated between neighbouring yarns
    by ``neighbour_correlation``."""

    name = "woven_textile"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "warp", 2: "weft", 3: "binder"}
    count_label = "yarns"

    weave: str = param(
        "plain",
        doc="plain, twill, satin, basket (2D weaves), orthogonal, layer_to_layer or "
        "through_thickness (3D weaves)",
        kind="str",
    )
    pattern: tuple | None = param(
        None,
        doc="twill: [over, under] (default [2, 2]); satin: [harness, move] (default [5, 2]); "
        "basket: [yarns woven together] (default [2])",
        kind="ints",
    )
    layers: int = param(
        1, doc="Plies of a 2D weave; warp layers of a 3D weave (with one more weft layer)",
        kind="int",
    )  # fmt: skip
    warp_spacing: float = param(
        2.0, doc="Distance between the axes of neighbouring warp yarns", unit="length"
    )
    weft_spacing: float | None = param(
        None, doc="Distance between the axes of neighbouring weft yarns (default: the warp's)",
        unit="length",
    )  # fmt: skip
    yarn_width: Distribution = param(
        Constant(1.6), doc="Width of the warp yarns (and of the wefts by default)",
        kind="distribution", unit="length",
    )  # fmt: skip
    yarn_thickness: Distribution = param(
        Constant(0.3), doc="Thickness of the warp yarns (and of the wefts by default)",
        kind="distribution", unit="length",
    )  # fmt: skip
    weft_width: Distribution | None = param(
        None, doc="Width of the weft yarns (default: yarn_width)", kind="distribution",
        unit="length",
    )  # fmt: skip
    weft_thickness: Distribution | None = param(
        None, doc="Thickness of the weft yarns (default: yarn_thickness)",
        kind="distribution", unit="length",
    )  # fmt: skip
    binder_width: Distribution | None = param(
        None, doc="Width of the binders of 3D weaves (default: 80 % of the room between the "
        "warps)", kind="distribution", unit="length",
    )  # fmt: skip
    binder_thickness: Distribution | None = param(
        None, doc="Thickness of the binders of 3D weaves (default: half the yarn thickness, "
        "at most 80 % of the room between the wefts)", kind="distribution", unit="length",
    )  # fmt: skip
    section: str = param(
        "ellipse", doc="Section of the yarns: ellipse, lenticular (two circular arcs) or power "
        "(power ellipse |u/a|^q + |v/c|^q <= 1)", kind="str",
    )  # fmt: skip
    section_power: float = param(
        3.0, doc="Exponent q of the power ellipse (2: an ellipse; larger: squarer, >= 1)"
    )
    transition: float = param(
        1.0,
        doc="Share of the spacing over which a crimped yarn passes from one level to the next "
        "(1: half cosines between the crossings; smaller: level over the crossed yarns)",
        kind="fraction",
    )
    ply_shift: tuple | None = param(
        None, doc="2D weaves: in-plane shift of each ply from the one below, in spacings "
        "(along the warp, along the weft; default: none)", kind="floats",
    )  # fmt: skip
    random_shift: bool = param(
        False, doc="2D weaves: plies shifted at random in their plane (over the repeat)",
        kind="bool",
    )  # fmt: skip
    ply_gap: float = param(
        0.0, doc="2D weaves: thickness of resin between consecutive plies", unit="length"
    )
    merge_yarns: bool = param(
        False, doc="All the yarns phase 1 (two phases) instead of warp 1, weft 2, binder 3",
        kind="bool",
    )  # fmt: skip
    samples: int = param(
        16, doc="Segments of the centrelines per change of level of a yarn", kind="int"
    )
    fibre_fraction_in_yarns: float | None = param(
        None, doc="Volume fraction of the fibres inside the yarns (theory: that of the "
        "composite)", kind="fraction",
    )  # fmt: skip
    axis: int = param(2, doc="Stacking direction (0: x, 1: y, 2: z)", kind="int")
    binder_depth: int | None = param(
        None, doc="3D weaves: the weft layers that a binder ties together, its binding depth "
        "(orthogonal: all, layers + 1, by default, fewer: the binders of the columns in turn "
        "between different layers; layer_to_layer: 2 by default)", kind="int",
    )  # fmt: skip
    stuffers: bool = param(
        True, doc="3D weaves: the straight warp layers (stuffers); false: the angle "
        "interlocks of binders and wefts only", kind="bool",
    )  # fmt: skip
    deviation_in_plane: float = param(
        0.0, doc="Standard deviation of the centrelines across the yarns, in the plane of "
        "the fabric (variability of the yarns)", unit="length",
    )  # fmt: skip
    deviation_through: float = param(
        0.0, doc="Standard deviation of the centrelines through the thickness of the fabric",
        unit="length",
    )  # fmt: skip
    width_variation: float = param(
        0.0, doc="Coefficient of variation of the width along the yarns (lognormal)",
        kind="fraction",
    )  # fmt: skip
    thickness_variation: float = param(
        0.0, doc="Coefficient of variation of the thickness along the yarns (lognormal)",
        kind="fraction",
    )  # fmt: skip
    size_correlation: float = param(
        0.0, doc="Correlation of the logarithms of the width and of the thickness at a point "
        "of a yarn (-1 to 1)",
    )  # fmt: skip
    correlation_length: float | tuple | None = param(
        None, doc="Correlation length of the variations along the yarns: one value, or two "
        "(the centrelines, the sections); default: twice the warp spacing", kind="vector",
        unit="length",
    )  # fmt: skip
    neighbour_correlation: float = param(
        0.0, doc="Correlation of the variations of neighbouring yarns of a ply or layer "
        "(rho^d for yarns d apart; 0 to 1)", kind="fraction",
    )  # fmt: skip
    variation_covariance: str = param(
        "gaussian", doc="Covariance of the variations along the yarns: gaussian (smooth "
        "paths) or exponential (a Markov process)", kind="str",
    )  # fmt: skip

    # -- parameters --------------------------------------------------------------------------

    def _validate(self) -> None:
        if self.weave not in WEAVES:
            raise ValueError(f"weave: one of {', '.join(WEAVES)}")
        if self.is_3d and self.pattern is not None:
            raise ValueError(f"pattern: none for the 3D weave {self.weave}")
        if not self.is_3d:
            interlacing(self.weave, self.pattern)  # its errors
        if self.layers < 1:
            raise ValueError("layers: at least 1")
        check_positive(self.warp_spacing, "warp_spacing")
        check_positive(self._weft_spacing, "weft_spacing")
        if self.section not in SECTIONS:
            raise ValueError(f"section: one of {', '.join(SECTIONS)}")
        if not self.section_power >= 1.0:
            raise ValueError("section_power: at least 1 (convex sections)")
        check_fraction(self.transition, "transition", open_interval=False)
        if self.transition <= 0.0:
            raise ValueError("transition: in (0, 1]")
        check_positive(self.ply_gap, "ply_gap", allow_zero=True)
        if self.ply_shift is not None and len(self.ply_shift) != 2:
            raise ValueError("ply_shift: two numbers, along the warp and along the weft")
        if self.is_3d and (self.ply_gap or self.ply_shift is not None or self.random_shift):
            raise ValueError("ply_shift, random_shift, ply_gap: the plies of 2D weaves")
        if not 2 <= self.samples <= 256:
            raise ValueError("samples: from 2 to 256 segments per change of level")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis: 0, 1 or 2")
        if self.fibre_fraction_in_yarns is not None:
            check_fraction(self.fibre_fraction_in_yarns, "fibre_fraction_in_yarns",
                           open_interval=False)  # fmt: skip
        if self.binder_depth is not None:
            if self.weave not in ("orthogonal", "layer_to_layer"):
                raise ValueError("binder_depth: the orthogonal and layer_to_layer weaves")
            if not 2 <= self.binder_depth <= self.layers + 1:
                raise ValueError(
                    f"binder_depth: the weft layers that a binder ties, from 2 to "
                    f"layers + 1 = {self.layers + 1}"
                )
        if not self.stuffers and self.weave not in ("layer_to_layer", "through_thickness"):
            raise ValueError(
                "stuffers: false for the angle interlocks only (layer_to_layer, through_thickness)"
            )
        for name in ("deviation_in_plane", "deviation_through"):
            check_positive(getattr(self, name), name, allow_zero=True)
        for name in ("width_variation", "thickness_variation"):
            check_fraction(getattr(self, name), name, open_interval=False)
        if not -1.0 < self.size_correlation < 1.0:
            raise ValueError("size_correlation: in (-1, 1)")
        if not 0.0 <= self.neighbour_correlation < 1.0:
            raise ValueError("neighbour_correlation: in [0, 1)")
        if self.variation_covariance not in COVARIANCES:
            raise ValueError(f"variation_covariance: one of {', '.join(COVARIANCES)}")
        if self.correlation_length is not None:
            values = np.atleast_1d(np.asarray(self.correlation_length, dtype=float))
            if values.shape not in ((1,), (2,)) or not np.all(values > 0.0):
                raise ValueError(
                    "correlation_length: one positive length, or two (the centrelines, the "
                    "sections)"
                )
        sizes = self.sizes()
        s_w, s_f = self.warp_spacing, self._weft_spacing
        if self.section == "lenticular":
            for family, (width, thick) in sizes.items():
                if thick.moment(1) > width.moment(1):
                    raise ValueError(f"{family}: lenticular sections no thicker than wide")
        w_w, w_f = sizes["warp"][0].upper(), sizes["weft"][0].upper()
        if not self.is_3d:
            if w_w > s_w * (1 + 1e-12):
                raise ValueError(
                    f"yarn_width: the warps side by side, at most warp_spacing = {s_w:g} "
                    f"(the widest drawn: {w_w:g})"
                )
            if w_f > s_f * (1 + 1e-12):
                raise ValueError(
                    f"weft_width: the wefts side by side, at most weft_spacing = {s_f:g} "
                    f"(the widest drawn: {w_f:g})"
                )
            return
        w_b, t_b = sizes["binder"][0].upper(), sizes["binder"][1].upper()
        if not self.stuffers:
            if w_b > s_w * (1 + 1e-12):
                raise ValueError(
                    f"binder_width: the binders side by side, at most warp_spacing = {s_w:g} "
                    f"(the widest drawn: {w_b:g})"
                )
        elif w_w + w_b > s_w * (1 + 1e-12):
            raise ValueError(
                f"warp and binder widths: the binders between the warps, their widths at most "
                f"warp_spacing = {s_w:g} together (the widest drawn: {w_w:g} + {w_b:g})"
            )
        if w_f + t_b >= s_f:
            raise ValueError(
                f"weft width and binder thickness: the binders pass between the wefts, their "
                f"sum below weft_spacing = {s_f:g} (the largest drawn: {w_f:g} + {t_b:g})"
            )
        pitch = self._warp_level_thickness() + _mean(sizes["weft"][1], 0)
        if t_b > pitch * (1 + 1e-12):
            raise ValueError(
                f"binder_thickness: at most the thickness of a warp and a weft, {pitch:g} "
                "(the binders pass between the weft layers)"
            )

    @property
    def is_3d(self) -> bool:
        return self.weave in WEAVES_3D

    @property
    def _weft_spacing(self) -> float:
        return self.warp_spacing if self.weft_spacing is None else float(self.weft_spacing)

    @property
    def spacings(self) -> tuple[float, float]:
        """The spacings of the warps and of the wefts."""
        return float(self.warp_spacing), self._weft_spacing

    @property
    def has_level_set(self) -> bool:
        """Two phases (one level set) when the yarns are merged."""
        return self.merge_yarns

    def sizes(self) -> dict[str, tuple[Distribution, Distribution]]:
        """The width and thickness of the yarns of each family (3D weaves without stuffers:
        the binders take those of the warps by default)."""
        out = {
            "warp": (self.yarn_width, self.yarn_thickness),
            "weft": (
                self.weft_width or self.yarn_width,
                self.weft_thickness or self.yarn_thickness,
            ),
        }
        if self.is_3d:
            if not self.stuffers:
                out["binder"] = (self.binder_width or self.yarn_width,
                                 self.binder_thickness or self.yarn_thickness)  # fmt: skip
                return out
            room_w = self.warp_spacing - self.yarn_width.moment(1)
            room_f = self._weft_spacing - out["weft"][0].moment(1)
            width = self.binder_width or Constant(max(0.8 * room_w, 1e-6))
            half = 0.5 * self.yarn_thickness.moment(1)
            thick = self.binder_thickness or Constant(max(min(half, 0.8 * room_f), 1e-6))
            out["binder"] = (width, thick)
        return out

    def _warp_level_thickness(self) -> float:
        """The mean thickness of the layers between the weft layers of a 3D weave (and of
        the warps of a 2D weave): that of the warps, or of the binders without stuffers."""
        family = "warp" if self.stuffers or not self.is_3d else "binder"
        return _mean(self.sizes()[family][1], 0)

    def active_phases(self) -> dict[int, str]:
        if self.merge_yarns:
            return {0: "matrix", 1: "yarns"}
        if self.is_3d:
            if not self.stuffers:
                return {0: "matrix", 2: "weft", 3: "binder"}
            return dict(self.phases)
        return {0: "matrix", 1: "warp", 2: "weft"}

    @property
    def varies(self) -> bool:
        """Whether the yarns vary along their paths."""
        names = ("deviation_in_plane", "deviation_through", "width_variation",
                 "thickness_variation")  # fmt: skip
        return any(getattr(self, name) > 0.0 for name in names)

    def correlation_lengths(self) -> tuple[float, float]:
        """The correlation lengths of the variations of the centrelines and of the sections
        (by default twice the warp spacing)."""
        if self.correlation_length is None:
            return 2.0 * self.warp_spacing, 2.0 * self.warp_spacing
        values = np.atleast_1d(np.asarray(self.correlation_length, dtype=float))
        return float(values[0]), float(values[-1])

    def _scales(self) -> dict[str, float]:
        """The scales of the processes of :data:`VARIATIONS`: the standard deviations of the
        deviations, and of the logarithms of the lognormal factors of the sizes,
        ``sqrt(ln(1 + cv^2))``."""
        return {
            "in_plane": float(self.deviation_in_plane),
            "through": float(self.deviation_through),
            "width": math.sqrt(math.log1p(float(self.width_variation) ** 2)),
            "thickness": math.sqrt(math.log1p(float(self.thickness_variation) ** 2)),
        }

    def _depth(self) -> int:
        """The warp layers that a binder crosses: its binding depth minus 1."""
        if self.binder_depth is not None:
            return int(self.binder_depth) - 1
        return self.layers if self.weave == "orthogonal" else 1

    # -- the weave -------------------------------------------------------------------------

    def repeat(self) -> tuple[int, int]:
        """The repeat of the weave in its plane: weft columns along the warp, warp rows
        along the weft (an orthogonal weave of binders shallower than the fabric: the
        least common multiple of 2 and of the ``N + 2 - D`` levels of its binders)."""
        if not self.is_3d:
            w = interlacing(self.weave, self.pattern)
            return int(w.shape[1]), int(w.shape[0])
        if self.weave == "orthogonal":  # the columns in turn between N + 2 - D pairs of slots
            return self._binder_period(), math.lcm(2, self.layers - self._depth() + 1)
        return self._binder_period(), 2

    def _binder_period(self) -> int:
        return 2 * (self.layers + 1) if self.weave == "through_thickness" else 2

    def thickness(self) -> float:
        """Thickness of the fabric (by the mean sizes): ``layers (t_w + t_f)`` plus the
        resin between the plies (2D), ``N (t_w + t_f) + t_f + 2 t_b`` (3D; ``t_b`` for
        ``t_w`` without stuffers)."""
        sizes = self.sizes()
        t_w, t_f = self._warp_level_thickness(), _mean(sizes["weft"][1], 0)
        if not self.is_3d:
            return self.layers * (t_w + t_f) + (self.layers - 1) * self.ply_gap
        t_b = _mean(sizes["binder"][1], 0)
        return self.layers * (t_w + t_f) + t_f + 2.0 * t_b

    def _levels(self):
        """The heights (from the bottom of the fabric) of the plies (2D) or of the weft
        layers, the warp layers and the slots of the binders (3D)."""
        sizes = self.sizes()
        c_w, c_f = 0.5 * self._warp_level_thickness(), 0.5 * _mean(sizes["weft"][1], 0)
        if not self.is_3d:
            ply = 2.0 * (c_w + c_f)
            middles = [k * (ply + self.ply_gap) + 0.5 * ply for k in range(self.layers)]
            return {"plies": middles, "offset": 0.5 * (c_w + c_f)}
        c_b = 0.5 * _mean(sizes["binder"][1], 0)
        pitch = 2.0 * (c_w + c_f)
        n = self.layers
        wefts = [2.0 * c_b + c_f + m * pitch for m in range(n + 1)]
        warps = [wefts[k] + 0.5 * pitch for k in range(n)]
        slots = [c_b, *warps, wefts[-1] + c_f + c_b]
        return {"wefts": wefts, "warps": warps, "slots": slots}

    def _binder_levels(self, i: int) -> list[int]:
        """The levels (the lowest slots) of the binders of the column ``i``: one binder in an
        orthogonal weave, the columns in turn between the ``N + 2 - D`` pairs of slots ``D``
        apart (``D`` the binding depth); one per pair in a layer-to-layer weave; one through
        the thickness."""
        n, k = self.layers, self._depth()
        if self.weave == "orthogonal":
            return [i % (n - k + 1)]
        if self.weave == "layer_to_layer":
            return list(range(n - k + 1))
        return [0]

    def _slot(self, i: int, j: int, level: int) -> int:
        """The slot of the binder of the column ``i`` and of the ``level`` at the weft
        column ``j``: the slots ``level`` and ``level + D`` in turn (orthogonal and
        layer-to-layer weaves of binding depth ``D``, adjacent columns out of phase), or down
        one slot at each weft column from the top to the bottom and back (through the
        thickness)."""
        n = self.layers
        if self.weave in ("orthogonal", "layer_to_layer"):
            return level + self._depth() + 1 if (i + j) % 2 == 0 else level
        period = 2 * (n + 1)
        jj = (j + (n + 1) * i) % period
        return n + 1 - jj if jj <= n + 1 else jj - (n + 1)

    def _paths(self, lengths: np.ndarray, periodic: bool, rng) -> list[_Path]:
        """The paths of the yarns in canonical axes (0: along the warp, 1: along the weft,
        2: through the thickness, from the lower corner of the domain), their sizes drawn."""
        s_w, s_f = self.warp_spacing, self._weft_spacing
        lx, ly, height = (float(v) for v in lengths)
        sizes = self.sizes()
        base = 0.5 * (height - self.thickness())
        margin = max(max(w.upper(), t.upper()) for w, t in sizes.values()) + max(s_w, s_f)

        def size(family):
            width, thick = sizes[family]
            return 0.5 * float(width.sample(rng, 1)[0]), 0.5 * float(thick.sample(rng, 1)[0])

        def indices(length, spacing, shift):
            if periodic:
                return list(range(round(length / spacing)))
            lo = math.floor((-margin - shift) / spacing - 0.5) - 1
            hi = math.ceil((length + margin - shift) / spacing - 0.5) + 1
            return list(range(lo, hi + 1))

        def stations(length, spacing, shift):
            """The stations of a yarn: one period and the first of the next (periodic), or
            those across the window."""
            ks = indices(length, spacing, shift)
            if periodic:
                ks = [*ks, len(ks)]
            return ks

        out: list[_Path] = []
        levels = self._levels()
        tau = self.transition
        if not self.is_3d:
            w = interlacing(self.weave, self.pattern)
            ry, rx = w.shape
            h = levels["offset"]
            for k, middle in enumerate(levels["plies"]):
                dx, dy = self._shift(k, rng, rx, ry)
                z0 = base + middle
                for i in indices(ly, s_w, dy):
                    js = stations(lx, s_f, dx)
                    zs = [z0 + (h if w[i % ry, j % rx] else -h) for j in js]
                    pieces = profile(zs, (js[0] + 0.5) * s_f + dx, s_f, tau)
                    a, c = size("warp")
                    out.append(_Path(1, 0, (i + 0.5) * s_w + dy, pieces, periodic, lx, a, c,
                                     (1, k)))  # fmt: skip
                for j in indices(lx, s_f, dx):
                    iis = stations(ly, s_w, dy)
                    zs = [z0 - (h if w[i % ry, j % rx] else -h) for i in iis]
                    pieces = profile(zs, (iis[0] + 0.5) * s_w + dy, s_w, tau)
                    a, c = size("weft")
                    out.append(_Path(2, 1, (j + 0.5) * s_f + dx, pieces, periodic, ly, a, c,
                                     (2, k)))  # fmt: skip
            return out
        for m, z in enumerate(levels["warps"] if self.stuffers else []):
            for i in indices(ly, s_w, 0.0):
                js = stations(lx, s_f, 0.0)
                pieces = profile([base + z] * len(js), (js[0] + 0.5) * s_f, s_f)
                a, c = size("warp")
                out.append(_Path(1, 0, (i + 0.5) * s_w, pieces, periodic, lx, a, c, (1, m)))
        for m, z in enumerate(levels["wefts"]):
            for j in indices(lx, s_f, 0.0):
                iis = stations(ly, s_w, 0.0)
                pieces = profile([base + z] * len(iis), (iis[0] + 0.5) * s_w, s_w)
                a, c = size("weft")
                out.append(_Path(2, 1, (j + 0.5) * s_f, pieces, periodic, ly, a, c, (2, m)))
        slots = levels["slots"]
        a_f = 0.5 * _mean(sizes["weft"][0], 0)
        gap = s_f - 2.0 * a_f
        for i in indices(ly, s_w, 0.0):
            for level in self._binder_levels(i):
                js = stations(lx, s_f, 0.0)
                zs = [base + slots[self._slot(i, j, level)] for j in js]
                if self.weave == "orthogonal":
                    pieces = profile(zs, (js[0] + 0.5) * s_f, s_f, gap / s_f, 0.5 * gap)
                else:
                    pieces = profile(zs, (js[0] + 0.5) * s_f, s_f, tau)
                a, c = size("binder")
                group = (3, level if self.weave == "layer_to_layer" else 0)
                out.append(_Path(3, 0, (i + 1.0) * s_w, pieces, periodic, lx, a, c, group))
        return out

    def _shift(self, k: int, rng, rx: int, ry: int) -> tuple[float, float]:
        """The in-plane shift of the ply ``k`` (along the warp, along the weft)."""
        if self.random_shift:
            if k == 0:
                return 0.0, 0.0
            u, v = rng.random(2)
            return float(u * rx * self._weft_spacing), float(v * ry * self.warp_spacing)
        if self.ply_shift is None:
            return 0.0, 0.0
        return (k * float(self.ply_shift[0]) * self._weft_spacing,
                k * float(self.ply_shift[1]) * self.warp_spacing)  # fmt: skip

    def _check_box(self, lengths: np.ndarray) -> None:
        """A periodic box holds whole repeats in its plane and the fabric in its height."""
        rx, ry = self.repeat()
        for length, spacing, count, name, along in (
            (lengths[0], self._weft_spacing, rx, "weft_spacing", "the warp"),
            (lengths[1], self.warp_spacing, ry, "warp_spacing", "the weft"),
        ):
            unit = count * spacing
            n = length / unit
            if abs(n - round(n)) > 1e-6 * max(n, 1.0) or round(n) < 1:
                low = max(1, math.floor(n)) * unit
                high = max(1, math.ceil(n)) * unit
                sizes = f"{low:g}" if math.isclose(low, high) else f"{low:g} or {high:g}"
                raise ValueError(
                    f"woven_textile: a periodic box holds whole repeats of the weave: along "
                    f"{along}, multiples of {count} x {name} = {unit:g}; the box is "
                    f"{length:g} long (nearest: {sizes})"
                )
        thickness = self.thickness()
        if lengths[2] < thickness * (1 - 1e-9):
            raise ValueError(
                f"woven_textile: the periodic box is {lengths[2]:g} thick along the stacking "
                f"axis, the fabric {thickness:g}: give it at least that"
            )

    def generate(self, domain: Domain, seed: SeedLike = None) -> TextileStructure:
        rng, seed_info = make_rng(seed)
        b, c = plane_axes(self.axis)
        order = (b, c, self.axis)  # the axes of the sample of the canonical ones
        lengths = np.asarray(domain.lengths, dtype=float)[list(order)]
        if domain.periodic:
            self._check_box(lengths)
        paths = self._paths(lengths, domain.periodic, rng)
        variations = self._variations(paths, lengths, domain.periodic, rng)
        made = [self._yarn(p, order, domain, v) for p, v in zip(paths, variations, strict=True)]
        yarns = [yarn for yarn, _ in made]
        micro = TextileStructure(domain, yarns, self.section, self.section_power, model=self,
                                 seed=seed_info)  # fmt: skip
        inside, shared = micro.sampled()
        volumes = micro.volumes() * inside  # the parts in the domain
        summary = {"thickness": self.thickness()}
        if self.varies:
            summary.update(_realized([info for _, info in made]))
        for phase, family in FAMILIES.items():
            chosen = [k for k, p in enumerate(paths) if p.phase == phase]
            if chosen:
                summary[f"{family}_fraction"] = float(volumes[chosen].sum() / domain.volume)
                summary[f"{family}_crimp"] = float(np.mean([paths[k].crimp for k in chosen]))
        summary["yarn_fraction"] = float(volumes.sum() / domain.volume)
        total = float(volumes.sum())
        overlap = float(np.sum(volumes * shared))
        summary["interpenetration"] = overlap / (2.0 * total) if total > 0 else 0.0
        micro.metadata["summary"] = summary
        return micro

    def _variations(self, paths: list[_Path], lengths, periodic: bool, rng) -> list:
        """The variations of each yarn (None for all when the yarns do not vary): for each
        group of yarns side by side (a family in a ply or a layer), in increasing order, the
        coefficients of the four processes of :data:`VARIATIONS` of its yarns (in the order
        of their transverse coordinates), independent standard normal variables mixed
        between the yarns by the Cholesky factor of :func:`neighbour_correlations`; the
        thickness correlated with the width at a point by ``size_correlation``. The
        processes are functions of the coordinate along the direction of the yarns,
        periodic over the box (or over the window and six correlation lengths, ten for the
        exponential covariance)."""
        out: list = [None] * len(paths)
        if not self.varies:
            return out
        l_c, l_s = self.correlation_lengths()
        kind = self.variation_covariance
        scales = self._scales()
        rho = float(self.size_correlation)
        groups: dict[tuple, list[int]] = {}
        for k, path in enumerate(paths):
            groups.setdefault(path.group, []).append(k)
        for key in sorted(groups):
            members = sorted(groups[key], key=lambda k: paths[k].fixed)
            first = paths[members[0]]
            if periodic:
                origin, period = 0.0, float(lengths[first.travel])
            else:
                origin = first.pieces[0].s0
                extent = first.pieces[-1].s1 - origin
                period = extent + (6.0 if kind == "gaussian" else 10.0) * max(l_c, l_s)
            spectra = (process_spectrum(period, l_c, kind), process_spectrum(period, l_s, kind))
            mixing = np.linalg.cholesky(
                neighbour_correlations(len(members), self.neighbour_correlation, periodic)
            )
            white = {}
            for name, spectrum in zip(VARIATIONS, (spectra[0],) * 2 + (spectra[1],) * 2,
                                      strict=True):  # fmt: skip
                draws = rng.standard_normal((len(members), len(spectrum[0]), 2))
                white[name] = np.einsum("ij,jkl->ikl", mixing, draws)
            white["thickness"] = (
                rho * white["width"] + math.sqrt(1.0 - rho * rho) * white["thickness"]
            )
            for row, k in enumerate(members):
                coefficients = {name: white[name][row] for name in VARIATIONS}
                out[k] = _Variation(origin, spectra[0], spectra[1], coefficients, scales)
        return out

    def _yarn(self, path: _Path, order, domain: Domain, variation=None):
        """A path sampled into the nodes of a yarn, in the axes of the sample, and its
        variations at the nodes (None without). A yarn that varies has segments of a sixth
        of the correlation lengths at most along its direction; its nodes move by its
        deviations (across the yarn along the width axis, through the thickness along the
        stacking axis), its tangents turn with them, its width axes stay normal to them,
        and its half sizes are those of its path times the factors of its sizes."""
        s, z, ts, tz = [], [], [], []
        step = min(self.correlation_lengths()) / _PER_LENGTH if variation is not None else None
        for piece in path.pieces:
            if piece.kind == "line":
                n = 1
            elif piece.kind == "cosine":
                n = self.samples
            else:
                n = max(2, math.ceil(self.samples * abs(piece.a1 - piece.a0) / math.pi))
            if step is not None:
                n = max(n, math.ceil(abs(piece.s1 - piece.s0) / step))
            ps, pz, pts, ptz = piece.sample(n)
            keep = slice(0 if not s else 1, None)  # the shared node once
            s.append(ps[keep])
            z.append(pz[keep])
            ts.append(pts[keep])
            tz.append(ptz[keep])
        s, z = np.concatenate(s), np.concatenate(z)
        ts, tz = np.concatenate(ts), np.concatenate(tz)
        if path.periodic:  # the last node is the first of the next period
            s, z, ts, tz = s[:-1], z[:-1], ts[:-1], tz[:-1]
        m = len(s)
        canonical = np.zeros((m, 3))
        tangent = np.zeros((m, 3))
        canonical[:, path.travel] = s
        canonical[:, 1 - path.travel] = path.fixed
        canonical[:, 2] = z
        tangent[:, path.travel] = ts
        tangent[:, 2] = tz
        width = np.zeros(3)
        width[1 - path.travel] = 1.0 if path.travel == 0 else -1.0  # e_z x (the travel)
        normal = np.broadcast_to(width, (m, 3))
        info = widths = thicknesses = None
        if variation is not None:
            up = np.array([0.0, 0.0, 1.0])
            dn, dz, gn, gz, fw, ft = variation.at(s)
            canonical += dn[:, None] * width + dz[:, None] * up
            # d/d(arc length) of the deviations: their slopes along s times ds/d(arc) = ts
            tangent += ts[:, None] * (gn[:, None] * width + gz[:, None] * up)
            tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
            normal = width - (tangent @ width)[:, None] * tangent
            normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
            if self.width_variation > 0.0:
                widths = path.half_width * fw
            if self.thickness_variation > 0.0:
                thicknesses = path.half_thickness * ft
            if path.periodic:  # the share of each node of the length along s
                ds = np.diff(np.append(s, s[0] + path.period))
                weights = 0.5 * (ds + np.roll(ds, 1))
            else:
                ds = np.diff(s)
                weights = 0.5 * (np.append(ds, 0.0) + np.insert(ds, 0, 0.0))
            info = {"in_plane": dn, "through": dz, "width": fw, "thickness": ft,
                    "weights": weights}  # fmt: skip
        period = np.zeros(3)
        if path.periodic:
            period[path.travel] = path.period
        nodes = np.zeros((m, 3))
        t = np.zeros((m, 3))
        n = np.zeros((m, 3))
        p = np.zeros(3)
        for k, a in enumerate(order):
            nodes[:, a] = canonical[:, k]
            t[:, a] = tangent[:, k]
            n[:, a] = normal[:, k]
            p[a] = period[k]
        nodes += domain.lower
        binormals = np.cross(t, n)
        phase = 1 if self.merge_yarns else path.phase
        yarn = Yarn(FAMILIES[path.phase], phase, nodes, n, binormals, p, path.half_width,
                    path.half_thickness, path.crimp, widths, thicknesses)  # fmt: skip
        return yarn, info

    # -- theory ----------------------------------------------------------------------------

    def expected_count(self, domain: Domain) -> float:
        """Number of yarns across the domain."""
        b, c = plane_axes(self.axis)
        lx, ly = float(domain.lengths[b]), float(domain.lengths[c])
        rows, cols = ly / self.warp_spacing, lx / self._weft_spacing
        if not self.is_3d:
            return self.layers * (rows + cols)
        per_column = self.layers - self._depth() + 1 if self.weave == "layer_to_layer" else 1
        warps = self.layers * rows if self.stuffers else 0.0
        return warps + (self.layers + 1) * cols + rows * per_column

    def _section_factors(self, n: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The lognormal factors of the width and the thickness of the sections at the nodes
        of a Gauss–Hermite quadrature of their correlated logarithms, and its weights."""
        scales = self._scales()
        s_w, s_t, rho = scales["width"], scales["thickness"], float(self.size_correlation)
        x, w = np.polynomial.hermite_e.hermegauss(n)
        w = w / w.sum()
        g1, g2 = (a.ravel() for a in np.meshgrid(x, x, indexing="ij"))
        width = np.exp(s_w * g1 - 0.5 * s_w * s_w)
        thick = np.exp(s_t * (rho * g1 + math.sqrt(1.0 - rho * rho) * g2) - 0.5 * s_t * s_t)
        return width, thick, np.outer(w, w).ravel()

    def _mean_area(self, family: str) -> float:
        """The mean area of the sections of a family: over the laws of the sizes of the
        yarns and the lognormal factors of their variations (``exp(rho s_w s_t)`` times the
        area at the mean sizes for the ellipse and the power ellipse, whose areas are
        proportional to ``a c``; quadratures for the lens)."""
        width, thick = self.sizes()[family]
        scales = self._scales()
        if self.section in ("ellipse", "power"):
            factor = math.exp(float(self.size_correlation) * scales["width"] * scales["thickness"])
            return factor * float(section_area(0.5 * width.moment(1), 0.5 * thick.moment(1),
                                               self.section, self.section_power))  # fmt: skip
        xa, wa = width.quadrature(64)
        xc, wc = thick.quadrature(64)
        if scales["width"] == 0.0 and scales["thickness"] == 0.0:
            area = section_area(0.5 * xa[:, None], 0.5 * xc[None, :], self.section)
            return float(np.sum(wa[:, None] * wc[None, :] * area))
        fw, ft, wf = self._section_factors()
        area = section_area(0.5 * xa[:, None, None] * fw, 0.5 * xc[None, :, None] * ft,
                            self.section)  # fmt: skip
        return float(np.einsum("i,j,k,ijk->", wa, wc, wf, area))

    def _lengthening(self, path: _Path, lam2: float) -> float:
        """The mean lengthening of a path by the deviations of its centreline, to the second
        order: ``(lambda_2 / 2) int (t_s^2 s_n^2 + t_s^4 s_z^2) d(arc)``, ``t_s`` the component
        of its tangent along its direction, ``lambda_2`` the second spectral moment of the
        process of the centrelines (``1 / l^2`` for the gaussian covariance)."""
        s_n, s_z = float(self.deviation_in_plane), float(self.deviation_through)
        if s_n == 0.0 and s_z == 0.0:
            return 0.0
        x, w = np.polynomial.legendre.leggauss(32)
        u, w = 0.5 * (x + 1.0), 0.5 * w
        total = 0.0
        for piece in path.pieces:
            if piece.kind == "line":
                ts = abs(piece.s1 - piece.s0) / piece.length if piece.length > 0 else 0.0
                total += piece.length * (ts**2 * s_n**2 + ts**4 * s_z**2)
            elif piece.kind == "cosine":  # d(arc) = span / t_s dw
                span = piece.s1 - piece.s0
                slope = (piece.z1 - piece.z0) * math.pi / (2.0 * span) * np.sin(math.pi * u)
                ts = 1.0 / np.sqrt(1.0 + slope * slope)
                total += span * float(w @ (ts * s_n**2 + ts**3 * s_z**2))
            else:  # an arc: t_s = |sin a|, d(arc) = r da
                a = piece.a0 + u * (piece.a1 - piece.a0)
                ts = np.abs(np.sin(a))
                total += (
                    piece.r
                    * abs(piece.a1 - piece.a0)
                    * float(w @ (ts**2 * s_n**2 + ts**4 * s_z**2))
                )
        return 0.5 * lam2 * total

    def tow_material(self, fibre, matrix, *, transverse_shear: str = "gscs"):
        """The material of the tows, aligned ``fibre`` in ``matrix`` at the fraction
        ``fibre_fraction_in_yarns`` (:func:`~morphogen.homogenization.unidirectional.
        composite_cylinders`): transversely isotropic, its axis 3 along the fibres, as the
        orientations of the segments of the yarns turn it."""
        from morphogen.homogenization.materials import Material
        from morphogen.homogenization.unidirectional import composite_cylinders

        if self.fibre_fraction_in_yarns is None:
            raise ValueError("tow_material: give the fibre_fraction_in_yarns of the model")
        return composite_cylinders(
            Material.from_config(fibre), Material.from_config(matrix),
            self.fibre_fraction_in_yarns, transverse_shear=transverse_shear,
        ).material()  # fmt: skip

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """The nominal fractions of the yarns of each family (the mean area of their
        sections times their arc length per unit volume, overlaps counted twice), their
        crimps (arc length over straight length, minus 1), the thickness of the fabric and,
        given the fraction of the fibres in the yarns, the fibre volume fraction. Yarns that
        vary: the mean area of their varying sections, and their arc lengths lengthened by
        the deviations of their centrelines (:meth:`_lengthening`, the processes periodic
        over the box of ``domain`` when it is periodic, over one repeat otherwise)."""
        rx, ry = self.repeat()
        lx, ly = rx * self._weft_spacing, ry * self.warp_spacing
        thickness = self.thickness()
        height = thickness
        if domain is not None:
            height = max(thickness, float(domain.lengths[self.axis]))
        lengths = np.array([lx, ly, height])
        paths = self._paths(lengths, True, np.random.default_rng(0))
        volume = lx * ly * height
        lam2 = (0.0, 0.0)
        if self.varies:
            periods = (lx, ly)
            if domain is not None and domain.periodic:
                b, c = plane_axes(self.axis)
                periods = (float(domain.lengths[b]), float(domain.lengths[c]))
            length = self.correlation_lengths()[0]
            lam2 = tuple(float(np.sum(w * om**2)) for om, w in
                         (process_spectrum(p, length, self.variation_covariance)
                          for p in periods))  # fmt: skip
        out: dict[str, float] = {"thickness": thickness}
        total = 0.0
        for phase, family in FAMILIES.items():
            chosen = [p for p in paths if p.phase == phase]
            if not chosen:
                continue
            arc = sum(path_length(p.pieces) + self._lengthening(p, lam2[p.travel])
                      for p in chosen)  # fmt: skip
            fraction = self._mean_area(family) * arc / volume
            out[f"{family}_fraction"] = fraction
            out[f"{family}_crimp"] = float(np.mean([p.crimp for p in chosen]))
            total += fraction
        out["yarn_fraction"] = total
        out["volume_fraction"] = total if self.merge_yarns else out.get("warp_fraction", 0.0)
        if self.fibre_fraction_in_yarns is not None:
            out["fibre_volume_fraction"] = total * self.fibre_fraction_in_yarns
        return out
