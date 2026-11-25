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
  the bottom and back.

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
    whether it closes on itself over the period of the box, its half sizes."""

    phase: int
    travel: int
    fixed: float
    pieces: list
    periodic: bool
    period: float
    half_width: float
    half_thickness: float

    @property
    def straight(self) -> float:
        return self.pieces[-1].s1 - self.pieces[0].s0

    @property
    def crimp(self) -> float:
        return path_length(self.pieces) / self.straight - 1.0


# -- the realizations -------------------------------------------------------------------------


@dataclass
class Yarn:
    """One yarn of a realization: its phase (1 warp, 2 weft, 3 binder; 1 all of them
    when merged) and family, the nodes of its centreline (m x 3, in the axes of the
    sample), the width axis ``normals`` and the thickness axis ``binormals`` at the nodes,
    the translation that closes a periodic yarn on itself (zeros: an open yarn, the
    polyline of its nodes), its half width and half thickness, and the crimp of its path."""

    family: str
    phase: int
    nodes: np.ndarray
    normals: np.ndarray
    binormals: np.ndarray
    period: np.ndarray
    half_width: float
    half_thickness: float
    crimp: float = 0.0

    def __post_init__(self) -> None:
        self.nodes = np.ascontiguousarray(self.nodes, dtype=float).reshape(-1, 3)
        self.normals = np.ascontiguousarray(self.normals, dtype=float).reshape(-1, 3)
        self.binormals = np.ascontiguousarray(self.binormals, dtype=float).reshape(-1, 3)
        self.period = np.asarray(self.period, dtype=float).reshape(3)
        self.half_width = float(self.half_width)
        self.half_thickness = float(self.half_thickness)

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
        return {
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

    @classmethod
    def from_dict(cls, data: dict) -> Yarn:
        return cls(**{k: data[k] for k in ("family", "phase", "nodes", "normals",
                                           "binormals", "period", "half_width",
                                           "half_thickness")},
                   crimp=float(data.get("crimp", 0.0)))  # fmt: skip


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
            parts.append((start, end, n0, b0, n1, b1, np.full(m, k), np.full(m, yarn.half_width),
                          np.full(m, yarn.half_thickness), np.arange(label, label + m),
                          np.full(m, yarn.phase)))  # fmt: skip
            label += m
        arrays = [np.concatenate(a) for a in zip(*parts, strict=True)]
        start, end = arrays[0], arrays[1]
        reach = np.maximum(arrays[7], arrays[8]) + float(band)
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
        typical = float(np.median(np.maximum(half_w, half_t))) if len(half_w) else 1.0
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

    def volumes(self) -> np.ndarray:
        """The volume of each yarn: the area of its section times the length of its
        polyline (one period of a periodic yarn), overlaps counted twice."""
        areas = [section_area(y.half_width, y.half_thickness, self.section, self.power)
                 for y in self.yarns]  # fmt: skip
        return np.array([float(a) * y.length for a, y in zip(areas, self.yarns, strict=True)])

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
    (``N (t_warp + t_weft) + t_weft + 2 t_binder``)."""

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
        if w_w + w_b > s_w * (1 + 1e-12):
            raise ValueError(
                f"warp and binder widths: the binders between the warps, their widths at most "
                f"warp_spacing = {s_w:g} together (the widest drawn: {w_w:g} + {w_b:g})"
            )
        if w_f + t_b >= s_f:
            raise ValueError(
                f"weft width and binder thickness: the binders pass between the wefts, their "
                f"sum below weft_spacing = {s_f:g} (the largest drawn: {w_f:g} + {t_b:g})"
            )
        pitch = _mean(sizes["warp"][1], 0) + _mean(sizes["weft"][1], 0)
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
        """The width and thickness of the yarns of each family."""
        out = {
            "warp": (self.yarn_width, self.yarn_thickness),
            "weft": (
                self.weft_width or self.yarn_width,
                self.weft_thickness or self.yarn_thickness,
            ),
        }
        if self.is_3d:
            room_w = self.warp_spacing - self.yarn_width.moment(1)
            room_f = self._weft_spacing - out["weft"][0].moment(1)
            width = self.binder_width or Constant(max(0.8 * room_w, 1e-6))
            half = 0.5 * self.yarn_thickness.moment(1)
            thick = self.binder_thickness or Constant(max(min(half, 0.8 * room_f), 1e-6))
            out["binder"] = (width, thick)
        return out

    def active_phases(self) -> dict[int, str]:
        if self.merge_yarns:
            return {0: "matrix", 1: "yarns"}
        if self.is_3d:
            return dict(self.phases)
        return {0: "matrix", 1: "warp", 2: "weft"}

    # -- the weave -------------------------------------------------------------------------

    def repeat(self) -> tuple[int, int]:
        """The repeat of the weave in its plane: weft columns along the warp, warp rows
        along the weft."""
        if not self.is_3d:
            w = interlacing(self.weave, self.pattern)
            return int(w.shape[1]), int(w.shape[0])
        return self._binder_period(), 2

    def _binder_period(self) -> int:
        return 2 * (self.layers + 1) if self.weave == "through_thickness" else 2

    def thickness(self) -> float:
        """Thickness of the fabric (by the mean sizes): ``layers (t_w + t_f)`` plus the
        resin between the plies (2D), ``N (t_w + t_f) + t_f + 2 t_b`` (3D)."""
        sizes = self.sizes()
        t_w, t_f = (_mean(sizes[k][1], 0) for k in ("warp", "weft"))
        if not self.is_3d:
            return self.layers * (t_w + t_f) + (self.layers - 1) * self.ply_gap
        t_b = _mean(sizes["binder"][1], 0)
        return self.layers * (t_w + t_f) + t_f + 2.0 * t_b

    def _levels(self):
        """The heights (from the bottom of the fabric) of the plies (2D) or of the weft
        layers, the warp layers and the slots of the binders (3D)."""
        sizes = self.sizes()
        c_w, c_f = (0.5 * _mean(sizes[k][1], 0) for k in ("warp", "weft"))
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

    def _slot(self, i: int, j: int, level: int) -> int:
        """The slot of the binder of the column ``i`` (and of the ``level`` of a
        layer-to-layer weave) at the weft column ``j``."""
        n = self.layers
        if self.weave == "orthogonal":
            return n + 1 if (i + j) % 2 == 0 else 0
        if self.weave == "layer_to_layer":
            return level + 2 if (i + j) % 2 == 0 else level
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
                    out.append(_Path(1, 0, (i + 0.5) * s_w + dy, pieces, periodic, lx, a, c))
                for j in indices(lx, s_f, dx):
                    iis = stations(ly, s_w, dy)
                    zs = [z0 - (h if w[i % ry, j % rx] else -h) for i in iis]
                    pieces = profile(zs, (iis[0] + 0.5) * s_w + dy, s_w, tau)
                    a, c = size("weft")
                    out.append(_Path(2, 1, (j + 0.5) * s_f + dx, pieces, periodic, ly, a, c))
            return out
        for z in levels["warps"]:
            for i in indices(ly, s_w, 0.0):
                js = stations(lx, s_f, 0.0)
                pieces = profile([base + z] * len(js), (js[0] + 0.5) * s_f, s_f)
                a, c = size("warp")
                out.append(_Path(1, 0, (i + 0.5) * s_w, pieces, periodic, lx, a, c))
        for z in levels["wefts"]:
            for j in indices(lx, s_f, 0.0):
                iis = stations(ly, s_w, 0.0)
                pieces = profile([base + z] * len(iis), (iis[0] + 0.5) * s_w, s_w)
                a, c = size("weft")
                out.append(_Path(2, 1, (j + 0.5) * s_f, pieces, periodic, ly, a, c))
        slots = levels["slots"]
        a_f = 0.5 * _mean(sizes["weft"][0], 0)
        gap = s_f - 2.0 * a_f
        binder_levels = range(self.layers) if self.weave == "layer_to_layer" else [0]
        for i in indices(ly, s_w, 0.0):
            for level in binder_levels:
                js = stations(lx, s_f, 0.0)
                zs = [base + slots[self._slot(i, j, level)] for j in js]
                if self.weave == "orthogonal":
                    pieces = profile(zs, (js[0] + 0.5) * s_f, s_f, gap / s_f, 0.5 * gap)
                else:
                    pieces = profile(zs, (js[0] + 0.5) * s_f, s_f, tau)
                a, c = size("binder")
                out.append(_Path(3, 0, (i + 1.0) * s_w, pieces, periodic, lx, a, c))
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
        yarns = [self._yarn(p, order, domain) for p in paths]
        micro = TextileStructure(domain, yarns, self.section, self.section_power, model=self,
                                 seed=seed_info)  # fmt: skip
        inside, shared = micro.sampled()
        volumes = micro.volumes() * inside  # the parts in the domain
        summary = {"thickness": self.thickness()}
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

    def _yarn(self, path: _Path, order, domain: Domain) -> Yarn:
        """A path sampled into the nodes of a yarn, in the axes of the sample."""
        s, z, ts, tz = [], [], [], []
        for piece in path.pieces:
            if piece.kind == "line":
                n = 1
            elif piece.kind == "cosine":
                n = self.samples
            else:
                n = max(2, math.ceil(self.samples * abs(piece.a1 - piece.a0) / math.pi))
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
            n[:, a] = width[k]
            p[a] = period[k]
        nodes += domain.lower
        binormals = np.cross(t, n)
        phase = 1 if self.merge_yarns else path.phase
        return Yarn(FAMILIES[path.phase], phase, nodes, n, binormals, p, path.half_width,
                    path.half_thickness, path.crimp)  # fmt: skip

    # -- theory ----------------------------------------------------------------------------

    def expected_count(self, domain: Domain) -> float:
        """Number of yarns across the domain."""
        b, c = plane_axes(self.axis)
        lx, ly = float(domain.lengths[b]), float(domain.lengths[c])
        rows, cols = ly / self.warp_spacing, lx / self._weft_spacing
        if not self.is_3d:
            return self.layers * (rows + cols)
        binders = rows * (self.layers if self.weave == "layer_to_layer" else 1)
        return self.layers * rows + (self.layers + 1) * cols + binders

    def _mean_area(self, family: str) -> float:
        width, thick = self.sizes()[family]
        if self.section in ("ellipse", "power"):
            return float(section_area(0.5 * width.moment(1), 0.5 * thick.moment(1),
                                      self.section, self.section_power))  # fmt: skip
        xa, wa = width.quadrature(64)
        xc, wc = thick.quadrature(64)
        area = section_area(0.5 * xa[:, None], 0.5 * xc[None, :], self.section)
        return float(np.sum(wa[:, None] * wc[None, :] * area))

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """The nominal fractions of the yarns of each family (the mean area of their
        sections times their arc length per unit volume, overlaps counted twice), their
        crimps (arc length over straight length, minus 1), the thickness of the fabric and,
        given the fraction of the fibres in the yarns, the fibre volume fraction."""
        rx, ry = self.repeat()
        lx, ly = rx * self._weft_spacing, ry * self.warp_spacing
        thickness = self.thickness()
        height = thickness
        if domain is not None:
            height = max(thickness, float(domain.lengths[self.axis]))
        lengths = np.array([lx, ly, height])
        paths = self._paths(lengths, True, np.random.default_rng(0))
        volume = lx * ly * height
        out: dict[str, float] = {"thickness": thickness}
        total = 0.0
        for phase, family in FAMILIES.items():
            chosen = [p for p in paths if p.phase == phase]
            if not chosen:
                continue
            arc = sum(path_length(p.pieces) for p in chosen)
            fraction = self._mean_area(family) * arc / volume
            out[f"{family}_fraction"] = fraction
            out[f"{family}_crimp"] = float(np.mean([p.crimp for p in chosen]))
            total += fraction
        out["yarn_fraction"] = total
        out["volume_fraction"] = total if self.merge_yarns else out["warp_fraction"]
        if self.fibre_fraction_in_yarns is not None:
            out["fibre_volume_fraction"] = total * self.fibre_fraction_in_yarns
        return out
