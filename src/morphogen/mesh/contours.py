"""The exact contours of the sections of bodies in a plane: oblique and wavy fibres.

A union of spheres and segments (:class:`~morphogen.models.union.PrimitiveUnion`: the long
fibres of :class:`~morphogen.models.long_fibers.LongFibers`, misaligned or not, chains of
spherocylinders) cut by the plane ``z``. Only the axes within a radius of the plane reach
it: clipped to that slab, they keep the section (:func:`section_capsules`). The capsules
that meet are grouped into bodies, whose sections, those of disjoint sets, do not meet;
each body's contours are traced on its own distance function, the least of the distances
to its axes less their radii — the marching squares on a grid of its own, of a
:data:`TRACE_CELLS`-th of its least radius, their vertices moved along their edges of the
grid onto the surface (:func:`section_contours`), the contours then those of the marching
squares, which do not cross. No gap between two bodies closes, however narrow, where the
level sets of the whole section (:func:`~morphogen.mesh.triangles.section_mesh`) close
those narrower than their background: the sections of misaligned fibres packed at the
fractions of composites. Where the capsules of a body cross, its contours have corners,
found where they pass from the surface of a capsule to that of another (Newton's method on
the two distances), or, where a third covers that point, through the third; where they
pass to a capsule that they do not cross, they bridge a gap narrower than a cell, which
grids four, sixteen and sixty-four times as fine follow, their distances computed only in
a band about the surface. A wedge narrower than a cell about a node of the grid, which the marching
squares close into a loop of its own, is left to the contour that passes around it. The
flat caps of cylinders that reach the plane cut their sections along lines: the distance
of a finite cylinder, and the corners of the rims where the contours pass from its lateral
surface to a cap, the nodes on the face of their edges.

:func:`contour_mesh` meshes them as :func:`~morphogen.mesh.triangles.disc_mesh` meshes
circles: the contours cut by the sides of the box and at their corners, nodes on them at
sizes graded from the gaps between the bodies (their distances, exact), within a contour
and between those of a body, from their curvatures and toward the sharp corners, the sides
of a periodic box matching; their edges checked, ``mmg2d`` meshes the regions within them,
which keep their nodes, each triangle the phase of its centroid; six-node elements have
their middle nodes on the surfaces, each edge on the capsule of its two ends.
"""

from __future__ import annotations

import itertools
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from morphogen.io import medit
from morphogen.mesh.triangles import (
    CIRCLE_SEGMENTS,
    CORNER_REFINEMENT,
    GAP_ELEMENTS,
    REF_CIRCLE,
    SIDE_REFS,
    TRI6_EDGES,
    TRI6_NODES,
    TRI_GAUSS,
    TriMesh,
    _kept,
    _marched,
    _mmg_args,
    _point,
    _PointSizes,
    _run,
    corner_gradients,
    jacobian_ratios,
)

#: The cells of the grid on which a body's contours are traced, across its least radius
TRACE_CELLS = 8
#: Newton's steps onto the surface of a body
NEWTON_STEPS = 6
#: The finer grids on which a body is traced again where its contours bridge a gap between
#: two of its capsules narrower than a cell: each a quarter of the last, its distances
#: computed only in the cells of the last that the surface may cross
TRACE_REFINEMENTS = 3
#: The most nodes of a grid at which the distance of a body is computed
TRACE_POINTS = 4_000_000
#: The least ratio of the area element of a curved six-node element to that of its straight
#: triangle: the middle nodes of the elements that would fold give way toward the middles
MIN_JACOBIAN = 0.3
#: The least angle (degrees) between the normals of the two surfaces at a corner of a
#: contour for the sizes to be graded toward it (``CORNER_REFINEMENT``)
CORNER_ANGLE = 10.0
#: The angle (degrees) within which a contour runs along a side that it crosses for its gap
#: to the side to grade its sizes, as the gaps to the sides that it does not cross: the thin
#: wedges and lenses of a contour that crosses a side at a grazing angle, whose triangles
#: would otherwise span them flat
GRAZING = 15.0
#: The angle (degrees) below which the contours of two capsules that meet are tangent, not a
#: corner (the round end of a capsule and the side of the next where a chain bends)
TANGENT = 0.5


def section_capsules(micro, z: float | None = None):
    """The spheres and segments of a union that reach the plane ``z`` (default: the middle
    of its box; their periodic images along ``z`` taken): their axes clipped to the slab
    within their radius of the plane, which keeps their sections — the points of the plane
    within a radius of an axis are within a radius of its part in the slab —, as
    ``(start, end, radii)``, a sphere an axis of no length; None for another structure, or
    for segments with flat caps that reach the plane, whose sections are not those of
    capsules."""
    axes = _section_axes(micro, z)
    if axes is None or axes[3].any():
        return None
    return axes[:3]


def _section_axes(micro, z: float | None = None, beyond: float = 0.0):
    """The axes of :func:`section_capsules` and, for each, whether its start and its end
    are flat caps that reach the plane (n x 2), kept where they are: ``(start, end, radii,
    flat)``; the ends clipped to the slab are round, which keeps the sections. ``beyond``:
    the slab wider by it (the union grown or shrunk by it: its caps moved). The union of
    unions (a combination): the axes of all. None for another structure."""
    from morphogen.models.combination import Combined
    from morphogen.models.union import PrimitiveUnion

    if isinstance(micro, Combined) and micro.operation == "union":
        parts = [_section_axes(part, z, beyond) for part in micro.parts]
        if any(part is None for part in parts):
            return None
        return tuple(np.concatenate([part[k] for part in parts]) for k in range(4))
    if not isinstance(micro, PrimitiveUnion):
        return None
    domain = micro.domain
    z = float(domain.lower[2] + 0.5 * domain.lengths[2]) if z is None else float(z)
    lz = float(domain.lengths[2])
    shifts = (-1.0, 0.0, 1.0) if domain.periodic else (0.0,)
    starts, ends, radii = [np.zeros((0, 3))], [np.zeros((0, 3))], [np.zeros(0)]
    flats = [np.zeros((0, 2), dtype=bool)]
    spheres, segments = micro.spheres, micro.segments
    for k in shifts:
        if len(spheres):
            c = spheres.centers + np.array([0.0, 0.0, k * lz])
            r = spheres.radii
            cut = (np.abs(c[:, 2] - z) < r) & (r > 0.0)
            starts.append(c[cut])
            ends.append(c[cut])
            radii.append(r[cut])
            flats.append(np.zeros((int(cut.sum()), 2), dtype=bool))
        if len(segments):
            a = segments.start + np.array([0.0, 0.0, k * lz])
            b = segments.end + np.array([0.0, 0.0, k * lz])
            r = segments.radii
            lo, hi = np.minimum(a[:, 2], b[:, 2]), np.maximum(a[:, 2], b[:, 2])
            cut = (lo - r - beyond < z) & (hi + r + beyond > z) & (r > 0.0)
            a, b, r = a[cut], b[cut], r[cut]
            slab = r + beyond
            dz = b[:, 2] - a[:, 2]
            level = np.abs(dz) <= 1e-15 * max(lz, 1.0)
            safe = np.where(level, 1.0, dz)
            t0 = np.where(level, 0.0, (z - slab - a[:, 2]) / safe)
            t1 = np.where(level, 1.0, (z + slab - a[:, 2]) / safe)
            t0, t1 = np.clip(np.minimum(t0, t1), 0.0, 1.0), np.clip(np.maximum(t0, t1), 0.0, 1.0)
            d = b - a
            starts.append(a + t0[:, None] * d)
            ends.append(a + t1[:, None] * d)
            radii.append(r)
            capped = not segments.round_caps  # (the ends kept, those in the slab)
            first = capped & (t0 == 0.0) & (np.abs(a[:, 2] - z) < slab)
            last = capped & (t1 == 1.0) & (np.abs(b[:, 2] - z) < slab)
            flats.append(np.column_stack([first, last]))
    return (np.concatenate(starts), np.concatenate(ends), np.concatenate(radii),
            np.concatenate(flats))  # fmt: skip


def section_layers(micro):
    """The unions whose contours bound the phases of a section, from the innermost, and the
    phases inside the innermost, between them and outside them: a union of spheres and
    segments itself (phase 1 inside, 0 outside), or one coated
    (:class:`~morphogen.models.nested.CoatedStructure`: the union and that grown by the
    thickness of the coating, or the union shrunk by that of the shells and the union), or
    the difference of unions (the hollow fibres of :mod:`~morphogen.models.fibers`: the
    lumens, phase 0 inside them, then the fibres; the union of those removed, then the
    first); a union of unions (a :class:`~morphogen.models.combination.Combined` union) is
    one; None for another structure."""
    from morphogen.mesh.triangles import _grown
    from morphogen.models.combination import Combined
    from morphogen.models.nested import CoatedStructure
    from morphogen.models.union import PrimitiveUnion

    if isinstance(micro, PrimitiveUnion) or (
        isinstance(micro, Combined) and micro.operation == "union"
        and _section_axes(micro) is not None
    ):  # fmt: skip
        return [micro], [1, 0]
    if isinstance(micro, Combined) and micro.operation == "difference":
        removed = (micro.parts[1] if len(micro.parts) == 2
                   else Combined(micro.domain, "union", micro.parts[1:]))  # fmt: skip
        if _section_axes(micro.parts[0]) is not None and _section_axes(removed) is not None:
            return [removed, micro.parts[0]], [0, 1, 0]
        return None
    if not (isinstance(micro, CoatedStructure) and isinstance(micro.inner, PrimitiveUnion)):
        return None
    inner, t = micro.inner, float(micro.thickness)
    if micro.side == "outside":
        return [inner, _grown(inner, t)], [1, micro.phase, 0]
    return [_grown(inner, -t), inner], [micro.core_phase, 1, 0]


def _layer_offsets(micro) -> list[float]:
    """The offsets of the unions of :func:`section_layers` from the union coated: the
    coating grown by its thickness, the cores of shells shrunk by it (none otherwise)."""
    from morphogen.models.nested import CoatedStructure

    if not isinstance(micro, CoatedStructure):
        layers = section_layers(micro)
        return [0.0] * (len(layers[0]) if layers is not None else 1)
    t = float(micro.thickness)
    return [0.0, t] if micro.side == "outside" else [-t, 0.0]


def _segment_distances(p1, q1, p2, q2, points: bool = False):
    """The distances between pairs of segments ``[p1, q1]`` and ``[p2, q2]`` (Ericson 2005,
    §5.1.9), points among them; with ``points``, also their nearest points on each."""
    d1, d2, w = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, w)
    c = np.einsum("ij,ij->i", d1, w)
    b = np.einsum("ij,ij->i", d1, d2)
    tiny = 1e-300
    denom = a * e - b * b
    s = np.where(denom > tiny, np.clip((b * f - c * e) / np.maximum(denom, tiny), 0.0, 1.0), 0.0)
    t = np.where(e > tiny, (b * s + f) / np.maximum(e, tiny), 0.0)
    low, high = t < 0.0, t > 1.0
    t = np.clip(t, 0.0, 1.0)
    s = np.where(low, np.clip(-c / np.maximum(a, tiny), 0.0, 1.0), s)
    s = np.where(high, np.clip((b - c) / np.maximum(a, tiny), 0.0, 1.0), s)
    s = np.where(a > tiny, s, 0.0)
    first, second = p1 + s[:, None] * d1, p2 + t[:, None] * d2
    distances = np.linalg.norm(first - second, axis=1)
    return (distances, first, second) if points else distances


@dataclass
class Body:
    """A body of a section: the capsules of a union that meet, their axes ``start``,
    ``end`` (clipped to the slab of the plane, translated to an image across a periodic box)
    and ``radii``, in the plane ``z``; ``flat``: whether their starts and ends are flat
    caps (n x 2; none by default), a capsule then a cylinder between the planes of its
    flat ends and its round ends balls; ``offset``: the distance by which the body is that
    of a union grown (a coating: its flat caps then moved out by it, their rims round) or
    shrunk (the cores of shells: the caps moved in, the rims sharp), its radii those grown
    or shrunk."""

    start: np.ndarray
    end: np.ndarray
    radii: np.ndarray
    z: float
    flat: np.ndarray | None = None
    offset: float = 0.0

    def _flat(self, k: int) -> tuple[bool, bool]:
        """Whether the start and the end of the capsule ``k`` are flat caps."""
        if self.flat is None:
            return False, False
        return bool(self.flat[k, 0]), bool(self.flat[k, 1])

    @property
    def capped(self) -> bool:
        """Whether some capsule of the body has a flat cap with a rim (none grown: its
        rims are round)."""
        return self.flat is not None and bool(np.any(self.flat)) and self.offset <= 0.0

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The rectangle of the plane that holds the section."""
        lo = np.minimum(self.start[:, :2], self.end[:, :2]) - self.radii[:, None]
        hi = np.maximum(self.start[:, :2], self.end[:, :2]) + self.radii[:, None]
        return lo.min(axis=0), hi.max(axis=0)

    def distance(self, xy: np.ndarray, gradient: bool = False):
        """The signed distance of points of the plane to the body (negative inside: the
        least of the distances to its axes less their radii), and its gradient in the
        plane."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        p = np.column_stack([xy, np.full(len(xy), self.z)])
        best = np.full(len(xy), np.inf)
        grad = np.zeros((len(xy), 2))
        for k, (a, b, r) in enumerate(zip(self.start, self.end, self.radii, strict=True)):
            if any(self._flat(k)):
                value, g = self._cylinder(k, p)
                closer = value < best
                best = np.where(closer, value, best)
                grad[closer] = g[closer]
                continue
            d = b - a
            length2 = float(d @ d)
            t = np.clip((p - a) @ d / length2, 0.0, 1.0) if length2 > 0.0 else np.zeros(len(p))
            v = p - a - t[:, None] * d
            n = np.linalg.norm(v, axis=1)
            value = n - r
            closer = value < best
            best = np.where(closer, value, best)
            if gradient:
                grad[closer] = v[closer, :2] / np.maximum(n[closer], 1e-300)[:, None]
        return (best, grad) if gradient else best

    def capsule(self, k: int, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The signed distance of points of the plane to the capsule ``k`` (to its axis,
        less its radius) and its gradient in the plane."""
        if any(self._flat(k)):
            xy = np.atleast_2d(np.asarray(xy, dtype=float))
            return self._cylinder(k, np.column_stack([xy, np.full(len(xy), self.z)]))
        return self.round(k, xy)

    def round(self, k: int, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The signed distance of points of the plane to the capsule ``k`` as if its ends
        were round (its lateral surface and the balls of its ends), and its gradient."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        p = np.column_stack([xy, np.full(len(xy), self.z)])
        a, b, r = self.start[k], self.end[k], float(self.radii[k])
        d = b - a
        length2 = float(d @ d)
        t = np.clip((p - a) @ d / length2, 0.0, 1.0) if length2 > 0.0 else np.zeros(len(p))
        v = p - a - t[:, None] * d
        n = np.linalg.norm(v, axis=1)
        return n - r, v[:, :2] / np.maximum(n, 1e-300)[:, None]

    def _axial(self, k: int, p: np.ndarray):
        """Along the axis of the capsule ``k``: its unit direction and length, the abscissae
        of points (of space) from its start, their offsets from the line of the axis and
        their distances to it."""
        a, b = self.start[k], self.end[k]
        d = b - a
        length = float(np.linalg.norm(d))
        axis = d / max(length, 1e-300)
        t = (p - a) @ axis
        v = p - a - t[:, None] * axis
        return axis, length, t, v, np.linalg.norm(v, axis=1)

    def _cylinder(self, k: int, p: np.ndarray):
        """The signed distance of points (of space, in the plane) to the capsule ``k`` with a
        flat cap: to its cylinder between the planes of its ends, exact (the radial and the
        axial excesses, their hypotenuse beyond the rim), and to the balls of its round
        ends, the least; its gradient in the plane."""
        r = float(self.radii[k]) - self.offset  # (the radius of the union not grown)
        axis, length, t, v, rho = self._axial(k, p)
        u, s = rho - r, np.maximum(-t, t - length)
        radial = v / np.maximum(rho, 1e-300)[:, None]
        along = np.where((t < 0.5 * length)[:, None], -axis, axis)
        both = (u > 0.0) & (s > 0.0)
        value = np.where(both, np.hypot(u, s), np.maximum(u, s))
        grad = np.where((u >= s)[:, None], radial, along)
        grad[both] = (u[both, None] * radial[both] + s[both, None] * along[both]) / value[
            both, None
        ]
        for end, flat in zip((self.start[k], self.end[k]), self._flat(k), strict=True):
            if not flat:  # a round end: the ball
                w = p - end
                n = np.linalg.norm(w, axis=1)
                closer = n - r < value
                value = np.where(closer, n - r, value)
                grad[closer] = w[closer] / np.maximum(n[closer], 1e-300)[:, None]
        return value - self.offset, grad[:, :2]

    def faces(self, xy: np.ndarray) -> np.ndarray:
        """The face of the surface of the body nearest each point: ``3 k`` the lateral
        surface of the capsule ``k`` (and its round ends), ``3 k + 1`` and ``3 k + 2`` its
        flat start and end."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        owners = self.owners(xy) if len(self.radii) > 1 else np.zeros(len(xy), dtype=int)
        out = 3 * owners
        for k in np.unique(owners).tolist():
            start, end = self._flat(k)
            if not (start or end):
                continue
            at = owners == k
            p = np.column_stack([xy[at], np.full(int(at.sum()), self.z)])
            _, length, t, _, rho = self._axial(k, p)
            u = rho - float(self.radii[k])
            first, last = -t - self.offset, t - length - self.offset  # (the planes moved)
            face = np.zeros(len(p), dtype=int)
            if start:
                face[(first >= u) & (first >= last)] = 1
            if end:
                face[(last >= u) & (last > first)] = 2
            out[at] = 3 * k + face
        return out

    def face(self, k: int, face: int, xy: np.ndarray):
        """The signed distance of points of the plane to a face of the capsule ``k`` (0: its
        lateral surface, the line of its axis less its radius; 1, 2: the planes of its
        start and of its end) and its gradient in the plane."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        p = np.column_stack([xy, np.full(len(xy), self.z)])
        axis, length, t, v, rho = self._axial(k, p)
        if face == 0:
            return rho - float(self.radii[k]), (v / np.maximum(rho, 1e-300)[:, None])[:, :2]
        if face == 1:
            return -t - self.offset, np.tile(-axis[:2], (len(p), 1))
        return t - length - self.offset, np.tile(axis[:2], (len(p), 1))

    def owners(self, xy: np.ndarray) -> np.ndarray:
        """The capsule nearest to each point (whose surface bounds the body there)."""
        values = np.array([self.capsule(k, xy)[0] for k in range(len(self.radii))])
        return values.argmin(axis=0)

    def onto(self, xy: np.ndarray, capsules=None, faces=None) -> np.ndarray:
        """Points moved onto the surface of the body along its gradient (Newton), or onto
        that of the capsule of each (``capsules``; -1: the body's), or onto a face of it
        (``faces``, :meth:`faces`; -1: by ``capsules``): its lateral surface and round
        ends, or the plane of a flat cap."""
        xy = np.array(xy, dtype=float).reshape(-1, 2)
        own = np.full(len(xy), -1) if capsules is None else np.asarray(capsules, dtype=int)
        on = np.full(len(xy), -1) if faces is None else np.asarray(faces, dtype=int)
        for k, f in np.unique(np.column_stack([own, on]), axis=0).tolist():
            at = (own == k) & (on == f)
            p = xy[at]
            for _ in range(NEWTON_STEPS):
                if f >= 0:
                    value, grad = self.face(f // 3, f % 3, p) if f % 3 else self.round(f // 3, p)
                else:
                    value, grad = self.distance(p, gradient=True) if k < 0 else self.capsule(k, p)
                g2 = np.maximum(np.einsum("ij,ij->i", grad, grad), 1e-300)
                p = p - (value / g2)[:, None] * grad
            xy[at] = p
        return xy


def section_bodies(micro, z: float | None = None, *, lower=None, upper=None,
                   margin: float = 0.0, periodic: bool | None = None,
                   offset: float = 0.0):  # fmt: skip
    """The bodies of the section of a union in the plane ``z`` (:func:`section_capsules`):
    the capsules that meet (their axes nearer than the sum of their radii) grouped, the
    images across a periodic box taken where they come within ``margin`` of the rectangle
    ``lower``, ``upper`` (by default the box's); None as :func:`section_capsules`, or for
    capsules whose sections touch, whose bodies would meet at a point (:func:`_touching`:
    those that touch in space have sections apart, bodies of their own, unless they touch
    in the plane). ``offset``: the union is that
    of a union grown by it (or shrunk), its radii grown, whose flat caps move with it
    (:class:`Body`)."""
    capsules = _section_axes(micro, z, abs(float(offset)))
    if capsules is None:
        return None
    domain = micro.domain
    z = float(domain.lower[2] + 0.5 * domain.lengths[2]) if z is None else float(z)
    lower = np.asarray(domain.lower[:2] if lower is None else lower, dtype=float)
    upper = np.asarray(domain.upper[:2] if upper is None else upper, dtype=float)
    lengths = np.asarray(domain.lengths[:2], dtype=float)
    periodic = bool(domain.periodic) if periodic is None else bool(periodic)
    start, end, radii, flat = capsules
    if periodic and len(radii):
        shifts = np.array([(i, j, 0.0) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float)
        shifts[:, :2] *= lengths
        start = (start[None] + shifts[:, None]).reshape(-1, 3)
        end = (end[None] + shifts[:, None]).reshape(-1, 3)
        radii = np.tile(radii, len(shifts))
        flat = np.tile(flat, (len(shifts), 1))
    lo = np.minimum(start[:, :2], end[:, :2]) - radii[:, None]
    hi = np.maximum(start[:, :2], end[:, :2]) + radii[:, None]
    near = np.all((hi > lower - margin) & (lo < upper + margin), axis=1)
    start, end, radii, flat = start[near], end[near], radii[near], flat[near]
    n = len(radii)
    tol = 1e-9 * float(max(lengths.max(), 1e-300))
    parent = np.arange(n)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1:
        middle = 0.5 * (start + end)
        half = 0.5 * np.linalg.norm(end - start, axis=1)
        reach = 2.0 * float(half.max() + radii.max()) + tol
        pairs = cKDTree(middle).query_pairs(reach, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            meet, first, second = _segment_distances(start[i], end[i], start[j], end[j], True)
            gap = meet - radii[i] - radii[j]
            touch = np.flatnonzero(np.abs(gap) <= tol)
            if len(touch) and _touching(start, end, radii, i[touch], j[touch], first[touch],
                                        second[touch], z):  # fmt: skip
                return None  # capsules whose sections touch
            for u, v in pairs[gap < -tol].tolist():  # (those that touch apart: their bodies)
                parent[root(u)] = root(v)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return [Body(start[k], end[k], radii[k], z, flat[k] if flat[k].any() else None,
                 float(offset)) for k in groups.values()]  # fmt: skip


def _touching(start, end, radii, i, j, first, second, z: float) -> bool:
    """Whether some pairs of capsules that touch (the nearest points of their axes ``first``
    and ``second``) have sections that touch in the plane ``z``: along a line (their axes
    parallel), or at a point within a thousandth of their least radius of the plane. Two
    capsules that touch at a point farther from it have sections apart, a gap between them
    that the plane holds."""
    d1, d2 = end[i] - start[i], end[j] - start[j]
    across = np.linalg.norm(np.cross(d1, d2), axis=1)
    lengths = np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1)
    if np.any(across <= 1e-9 * np.maximum(lengths, 1e-300)):
        return True
    apart = np.maximum(np.linalg.norm(second - first, axis=1), 1e-300)
    contact = first + (radii[i] / apart)[:, None] * (second - first)
    return bool(np.any(np.abs(contact[:, 2] - z) <= 1e-3 * np.minimum(radii[i], radii[j])))


@dataclass
class _Grid:
    """The nodes of a grid at which the distance of a body is known: the grid from
    ``lower``, its cell ``h``, ``shape`` nodes along x and y; the keys ``i * ny + j`` of the
    nodes known, sorted, and their distances ``values``."""

    lower: np.ndarray
    h: float
    shape: tuple[int, int]
    keys: np.ndarray
    values: np.ndarray

    def find(self, keys) -> np.ndarray:
        """The indices of nodes among those known, -1 for the others."""
        return _found(self.keys, keys)

    def points(self, index) -> np.ndarray:
        """The coordinates of the nodes known of these indices."""
        i, j = np.divmod(self.keys[index], self.shape[1])
        return np.column_stack([self.lower[0] + self.h * i, self.lower[1] + self.h * j])


def _found(sorted_keys: np.ndarray, keys) -> np.ndarray:
    """The indices of ``keys`` in ``sorted_keys``, -1 for those not there."""
    keys = np.asarray(keys, dtype=np.int64)
    if not len(sorted_keys):
        return np.full(keys.shape, -1, dtype=np.int64)
    k = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return np.where(sorted_keys[k] == keys, k, -1)


def _marching_loops(grid: _Grid, along=None) -> list[np.ndarray] | None:
    """The closed contours ``f = 0`` of values at the nodes of a grid (:class:`_Grid`, the
    surface across no edge to a node not known, the values positive along its border) by
    the marching squares, the saddles resolved by the mean of their corners: lists of
    their points, linear between the nodes, or ``along(starts, steps, t)`` (the points
    ``starts + t steps`` of their edges of the grid, moved along them); None where an
    edge of a cell that the surface crosses has a node not known."""
    nx, ny = grid.shape
    keys, f, h = grid.keys, grid.values, grid.h
    inside = f < 0.0
    i, j = np.divmod(keys, ny)
    across = []  # the edges (i, j)-(i+1, j), then (i, j)-(i, j+1), that the surface crosses
    for step, last in ((ny, i >= nx - 1), (1, j >= ny - 1)):
        other = grid.find(keys + step)
        other[last] = -1
        known = np.flatnonzero(other >= 0)
        cut = known[inside[known] != inside[other[known]]]
        across.append((cut, other[cut]))
    (ax, bx), (ay, by) = across
    nh = len(ax)
    if not nh + len(ay):
        return []
    t = np.concatenate([f[ax] / (f[ax] - f[bx]), f[ay] / (f[ay] - f[by])])
    starts = grid.points(np.concatenate([ax, ay]))
    steps = np.concatenate([np.tile([h, 0.0], (nh, 1)), np.tile([0.0, h], (len(ay), 1))])
    points = starts + t[:, None] * steps if along is None else along(starts, steps, t)
    xkeys, ykeys = keys[ax], keys[ay]  # (sorted, as the keys)
    cells = np.concatenate([xkeys, xkeys[j[ax] > 0] - 1, ykeys, ykeys[i[ay] > 0] - ny])
    cells = np.unique(cells)
    ci, cj = np.divmod(cells, ny)
    cells = cells[(ci < nx - 1) & (cj < ny - 1)]
    bottom, top = _found(xkeys, cells), _found(xkeys, cells + 1)
    left, right = _found(ykeys, cells), _found(ykeys, cells + ny)
    left, right = np.where(left >= 0, nh + left, -1), np.where(right >= 0, nh + right, -1)
    count = (bottom >= 0).astype(int) + (top >= 0) + (left >= 0) + (right >= 0)
    if np.any(count % 2):
        return None  # (a node not known about a cell that the surface crosses)
    edges = np.stack([bottom, right, top, left], axis=-1)
    links = [np.sort(edges[count == 2], axis=1)[:, 2:]]
    four = count == 4
    if np.any(four):
        c = cells[four]
        corner = [grid.find(c + d) for d in (0, ny, ny + 1, 1)]
        if min(int(k.min()) for k in corner) < 0:
            return None
        centre = sum(f[k] for k in corner) < 0.0
        joined = (centre == inside[corner[0]])[:, None]  # the corners 00 and 11 joined
        b, r, tp, lf = (edges[four][:, k] for k in range(4))
        links.append(np.where(joined, np.column_stack([b, r]), np.column_stack([lf, b])))
        links.append(np.where(joined, np.column_stack([tp, lf]), np.column_stack([r, tp])))
    links = np.concatenate(links)
    neighbours = np.full((len(points), 2), -1, dtype=np.int64)
    degree = np.zeros(len(points), dtype=np.int64)
    for u, v in links.tolist():
        if degree[u] > 1 or degree[v] > 1:
            return None
        neighbours[u, degree[u]] = v
        degree[u] += 1
        neighbours[v, degree[v]] = u
        degree[v] += 1
    if np.any(degree != 2):
        return None
    loops, seen = [], np.zeros(len(points), dtype=bool)
    for first in range(len(points)):
        if seen[first]:
            continue
        loop, previous, current = [first], first, int(neighbours[first, 0])
        seen[first] = True
        while current != first:
            loop.append(current)
            seen[current] = True
            a, b = neighbours[current]
            previous, current = current, int(b if a == previous else a)
        loops.append(points[loop])
    return loops


def _corner(body: Body, a: int, b: int, x: np.ndarray, tol: float):
    """The point near ``x`` on the surfaces of the capsules ``a`` and ``b`` of a body, by
    Newton's method on their two distances, and the angle (degrees) between their normals
    there; None where it does not converge (surfaces nearly tangent: no corner) or where
    another capsule covers it."""
    found = _meeting(body, a, b, x, tol)
    if found is None or body.distance(found[0][None])[0] < -tol:
        return None
    return found


def _meeting(body: Body, a: int, b: int, x: np.ndarray, tol: float):
    """The point near ``x`` on the surfaces of the capsules ``a`` and ``b``, covered by
    another capsule or not, and the angle of their normals there (:func:`_corner`). Where a
    capsule has a flat cap, whose distance is not smooth at its rim (Newton's steps then go
    from one face to the other), the nearest of the points where a face of one meets a face
    of the other, on both surfaces."""
    found = _newton_pair(lambda p: body.capsule(a, p), lambda p: body.capsule(b, p), x, tol)
    if found is None and body.capped:
        nearest = np.inf
        for fa in _face_list(body, a):
            for fb in _face_list(body, b):
                y = _newton_pair(lambda p, fa=fa: _face_value(body, a, fa, p),
                                 lambda p, fb=fb: _face_value(body, b, fb, p), x, tol)  # fmt: skip
                if y is None or max(abs(float(body.capsule(a, y)[0][0])),
                                    abs(float(body.capsule(b, y)[0][0]))) > tol:  # fmt: skip
                    continue
                if float(np.linalg.norm(y - x)) < nearest:
                    found, nearest = y, float(np.linalg.norm(y - x))
    if found is None:
        return None
    (_, ga), (_, gb) = body.capsule(a, found), body.capsule(b, found)
    # (the normals of the contours in the plane: those of the surfaces tilt out of it)
    lengths = float(np.linalg.norm(ga[0]) * np.linalg.norm(gb[0]))
    cosine = float(ga[0] @ gb[0]) / max(lengths, 1e-300)
    return found[0], float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _newton_pair(first, second, x: np.ndarray, tol: float):
    """The point near ``x`` where two distances (``first``, ``second``: their values and
    gradients at points) vanish, by Newton's method; None where it does not converge."""
    for _ in range(2 * NEWTON_STEPS):
        (va, ga), (vb, gb) = first(x), second(x)
        if max(abs(float(va[0])), abs(float(vb[0]))) <= 1e-3 * tol:
            break
        jacobian = np.array([ga[0], gb[0]])
        if abs(float(np.linalg.det(jacobian))) < 1e-6:
            return None
        x = x - np.linalg.solve(jacobian, np.array([va[0], vb[0]]))
    (va, _), (vb, _) = first(x), second(x)
    if max(abs(float(va[0])), abs(float(vb[0]))) > tol:
        return None
    return x


def _face_list(body: Body, k: int) -> list[int]:
    """The faces of the capsule ``k``: its lateral surface (0) and its flat caps (1, 2)."""
    return [0] + [f for f, flat in zip((1, 2), body._flat(k), strict=True) if flat]


def _face_value(body: Body, k: int, f: int, xy: np.ndarray):
    """The distance of points to the face ``f`` of the capsule ``k`` (:func:`_face_list`)
    and its gradient."""
    return body.face(k, f, xy) if f else body.round(k, xy)


def _corners(body: Body, a: int, b: int, x: np.ndarray, tol: float):
    """The corners near ``x`` where a contour passes from the surface of the capsule ``a``
    to that of ``b``: their point (:func:`_corner`); or, where a third capsule covers it
    (three surfaces that meet nearly at a point, the contour along the third for less than
    a cell), the corners of ``a`` with the third and of the third with ``b``. None if
    there are none."""
    found = _corner(body, a, b, x, tol)
    if found is not None:
        return [found]
    met = _meeting(body, a, b, x, tol)
    if met is None:
        return None
    point = met[0][None]
    depth = np.array([float(body.capsule(c, point)[0][0]) if c not in (a, b) else np.inf
                      for c in range(len(body.radii))])  # fmt: skip
    c = int(np.argmin(depth))
    if not depth[c] < -tol:
        return None
    first, second = _corner(body, a, c, point, tol), _corner(body, c, b, point, tol)
    if first is None or second is None:
        return None
    return [first, second]


def _face_corner(body: Body, k: int, first: int, second: int, x: np.ndarray, tol: float):
    """The point near ``x`` where two faces of the capsule ``k`` meet (:meth:`Body.face`: its
    lateral surface and a flat cap), by Newton's method on their two distances, and the
    angle (degrees) between their normals there; None where it does not converge, or off
    the surface of the capsule, or where another capsule covers it."""
    for _ in range(2 * NEWTON_STEPS):
        (va, ga), (vb, gb) = body.face(k, first, x), body.face(k, second, x)
        if max(abs(float(va[0])), abs(float(vb[0]))) <= 1e-3 * tol:
            break
        jacobian = np.array([ga[0], gb[0]])
        if abs(float(np.linalg.det(jacobian))) < 1e-9:
            return None
        x = x - np.linalg.solve(jacobian, np.array([va[0], vb[0]]))
    (va, ga), (vb, gb) = body.face(k, first, x), body.face(k, second, x)
    if max(abs(float(va[0])), abs(float(vb[0]))) > tol:
        return None
    if abs(float(body.capsule(k, x)[0][0])) > tol or body.distance(x)[0] < -tol:
        return None
    cos = float(ga[0] @ gb[0]) / max(float(np.linalg.norm(ga[0]) * np.linalg.norm(gb[0])), 1e-300)
    return x[0], float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _face_at(body: Body, k: int, x: np.ndarray, tol: float) -> int:
    """The face of the capsule ``k`` on which a point of its surface lies: a flat cap (1, 2)
    whose plane it is on, or its lateral surface (0)."""
    for f, flat in zip((1, 2), body._flat(k), strict=True):
        if flat and abs(float(body.face(k, f, x[None])[0][0])) <= tol:
            return f
    return 0


def _with_rims(body: Body, first, second, found, tol: float):
    """The corners where a contour passes from a capsule to another (``found``), with
    those of the rims of their flat caps between them and its vertices: where the corner
    of the two lies on another face of the first than its vertex before (``first``: the
    capsule and the face), or of the second than its vertex after, the contour passes
    along a cap for less than a cell. None where such a rim is not found."""
    out = list(found)
    (a, face_a), (b, face_b) = first, second
    on = _face_at(body, a, out[0][0], tol)
    if on != face_a:
        rim = _face_corner(body, a, face_a, on, out[0][0][None], tol)
        if rim is None:
            return None
        out.insert(0, rim)
    on = _face_at(body, b, out[-1][0], tol)
    if on != face_b:
        rim = _face_corner(body, b, on, face_b, out[-1][0][None], tol)
        if rim is None:
            return None
        out.append(rim)
    return out


def _joined(body: Body, a: int, b: int) -> bool:
    """Whether the capsules ``a`` and ``b`` of a body share an end (a joint of a chain: its
    surface passes smoothly from one to the other)."""
    ends_a = np.array([body.start[a], body.end[a]])
    ends_b = np.array([body.start[b], body.end[b]])
    gap = np.linalg.norm(ends_a[:, None] - ends_b[None], axis=-1).min()
    return bool(gap <= 1e-9 * float(max(body.radii[a], body.radii[b])))


def _shared(body: Body, a: int, b: int, points: np.ndarray, h: float) -> bool:
    """Whether points lie on the surfaces of both capsules ``a`` and ``b`` (within a
    thousandth of a cell: surfaces that coincide there)."""
    return bool(max(np.abs(body.capsule(a, points)[0]).max(),
                    np.abs(body.capsule(b, points)[0]).max()) <= 1e-3 * h)  # fmt: skip


def _section_polygon(body: Body, k: int, n: int = 64) -> np.ndarray:
    """Points of the boundary of the section of the capsule ``k`` of a body, convex: along
    ``n`` rays from the point of its axis nearest to the plane, by bisection."""
    a, b, r = body.start[k], body.end[k], float(body.radii[k])
    d = b - a
    t = float(np.clip((body.z - a[2]) / d[2], 0.0, 1.0)) if d[2] != 0.0 else 0.5
    centre = (a + t * d)[:2]
    if any(body._flat(k)):  # (the axis may end short of the plane: into the section)
        for _ in range(4 * NEWTON_STEPS):
            value, grad = body.capsule(k, centre)
            if value[0] < -0.25 * r:
                break
            centre = centre - (value[0] + 0.5 * r) * grad[0] / max(float(grad[0] @ grad[0]), 1e-12)
    angle = 2.0 * np.pi * np.arange(n) / n
    u = np.column_stack([np.cos(angle), np.sin(angle)])
    low, high = np.zeros(n), np.full(n, float(np.linalg.norm(d[:2])) + 2.0 * r)
    for _ in range(24):  # (to a millionth: Newton's method then finds the corners)
        middle = 0.5 * (low + high)
        inside = body.capsule(k, centre + middle[:, None] * u)[0] < 0.0
        low, high = np.where(inside, middle, low), np.where(inside, high, middle)
    return centre + (0.5 * (low + high))[:, None] * u


def _pair_corners(body: Body, a: int, b: int, tol: float, polygon=None) -> list[np.ndarray]:
    """The corners of a body where the surfaces of its capsules ``a`` and ``b`` cross and
    no other capsule covers them: from the changes of sign of the distance to ``b`` along a
    polygon of the section of ``a`` (:func:`_section_polygon`, unless given), by Newton's
    method on the two distances (:func:`_corner`)."""
    polygon = _section_polygon(body, a) if polygon is None else polygon
    value = body.capsule(b, polygon)[0]
    outside, clear = value > 0.0, np.abs(value) > 1e-6 * float(body.radii[b])
    found: list[np.ndarray] = []
    change = (outside != np.roll(outside, -1)) & clear & np.roll(clear, -1)
    for i in np.flatnonzero(change).tolist():  # (not where the surfaces coincide: a joint)
        middle = 0.5 * (polygon[i] + polygon[(i + 1) % len(polygon)])[None]
        corner = _corner(body, a, b, middle, tol)
        if corner is None or corner[1] < TANGENT:  # (two surfaces tangent: no corner)
            continue
        if all(np.linalg.norm(corner[0] - y) > 1e3 * tol for y in found):
            found.append(corner[0])
    return found


def _crossing(points: np.ndarray, edges: np.ndarray) -> bool:
    """Whether two of the edges between points cross (those that share a point do
    not)."""
    a, b = points[edges[:, 0]], points[edges[:, 1]]
    lengths = np.linalg.norm(b - a, axis=1)
    pairs = cKDTree(0.5 * (a + b)).query_pairs(float(lengths.max()), output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    apart = ((edges[i, 0] != edges[j, 0]) & (edges[i, 0] != edges[j, 1])
             & (edges[i, 1] != edges[j, 0]) & (edges[i, 1] != edges[j, 1]))  # fmt: skip
    i, j = i[apart], j[apart]

    def side(p, q, r):
        return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])

    cross = (side(a[i], b[i], a[j]) * side(a[i], b[i], b[j]) < 0.0) & (
        side(a[j], b[j], a[i]) * side(a[j], b[j], b[i]) < 0.0
    )
    return bool(np.any(cross))


def _simple(loop: np.ndarray) -> bool:
    """Whether a closed contour does not cross itself."""
    k = np.arange(len(loop))
    return not _crossing(loop, np.column_stack([k, np.roll(k, -1)]))


def _traced(body: Body) -> list[tuple[np.ndarray, list[int], list[float]]] | None:
    """The closed contours of the section of a body (see :func:`section_contours`), their
    corners among their vertices: where a contour passes from the surface of a capsule to
    that of another that crosses it, the point of both surfaces between the two vertices
    (:func:`_corner`); each contour, the indices of its corners and their angles. Where a
    contour passes from a capsule to another that it does not cross, its chord outside the
    body, it bridges a gap between them narrower than a cell: the body is traced again on
    a grid four times as fine, :data:`TRACE_REFINEMENTS` times at most; None if a bridge
    is left."""
    h = float(body.radii.min()) / TRACE_CELLS
    grids: list = []
    for _ in range(TRACE_REFINEMENTS + 1):
        out = _traced_at(body, h, grids)
        if out is not None:
            return out
        h /= 4.0
    return None


def _traced_at(body: Body, h: float, grids: list | None = None):
    """The contours of :func:`_traced` on the grid of cell ``h``; None for a bridge.
    ``grids``: those on which the body was traced before (:class:`_Grid`), the last
    coarser, to which this one is added (:func:`_grid_distances`)."""
    lo, hi = body.bounds
    lo, hi = lo - 2.0 * h, hi + 2.0 * h
    nx, ny = (int(np.ceil((hi - lo)[k] / h)) + 1 for k in (0, 1))
    if not grids and nx * ny > TRACE_POINTS:
        return None  # (a grid too large for the body)
    grid = _grid_distances(body, lo, h, (nx, ny), grids[-1] if grids else None)
    if grid is None:
        return None  # (a band about its surface too large)
    if grids is not None:
        grids.append(grid)
    tol = 1e-9 * float(body.radii.min())
    out = []

    def along(starts, steps, t):  # Newton's steps along the edges, bisection out of them
        sign, low, high = np.sign(body.distance(starts)), np.zeros(len(t)), np.ones(len(t))
        for _ in range(8 * NEWTON_STEPS):
            d, g = body.distance(starts + t[:, None] * steps, gradient=True)
            done = np.abs(d) <= 1e-3 * tol
            if done.all():
                break
            before = np.sign(d) == sign
            low, high = np.where(before, t, low), np.where(before, high, t)
            slope = np.einsum("ij,ij->i", g, steps)
            newton = t - d / np.where(slope != 0.0, slope, np.inf)
            step = np.where((newton >= low) & (newton <= high), newton, 0.5 * (low + high))
            t = np.where(done, t, step)
        return starts + t[:, None] * steps

    traced = _marching_loops(grid, along)
    if traced is None:
        return None
    for loop in traced:
        step = np.linalg.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0), axis=1)
        loop = loop[step > 1e-9 * h]  # (the points of a node of the grid on the surface)
        if len(loop) < 3:
            continue
        owner = body.owners(loop) if len(body.radii) > 1 else np.zeros(len(loop), dtype=int)
        face = body.faces(loop) if body.capped else 3 * owner
        n = len(loop)
        found_at: list[tuple[int, float, np.ndarray, float]] = []  # (edge, along, x, angle)
        for k in range(n):
            a, b = int(owner[k]), int(owner[(k + 1) % n])
            if face[k] == face[(k + 1) % n]:
                continue
            middle = 0.5 * (loop[k] + loop[(k + 1) % n])[None]
            if a == b:  # from a face of a capsule to another: the rim of a flat cap
                found = _face_corner(body, a, int(face[k]) % 3, int(face[(k + 1) % n]) % 3,
                                     middle, tol)  # fmt: skip
                if found is None:
                    return None  # a rim that the grid did not resolve
                found = [found]
            else:
                found = _corners(body, a, b, middle, tol)
                if found is not None and body.capped:
                    found = _with_rims(body, (a, int(face[k]) % 3), (b, int(face[(k + 1) % n]) % 3),
                                       found, tol)  # fmt: skip
                    if found is None:
                        return None  # a rim that the grid did not resolve
            if found is None:
                if body.distance(middle)[0] > 0.0:
                    return None  # a gap between the capsules bridged by the chord
                if not (_joined(body, a, b) or _shared(body, a, b, loop[[k, (k + 1) % n]], h)):
                    return None  # a corner that the grid did not resolve
                continue
            known = [y for *_, y, _ in found_at]  # (the same corner, found again, left out)
            group = [(x, angle) for x, angle in found
                     if not any(np.linalg.norm(x - y) < 1e-6 * h for y in known)]  # fmt: skip
            if not group:
                continue
            # the corners, in turn, on the edge near this one that they lengthen the least
            best = None
            for e in range(k - 2, k + 3):
                p, q = loop[e % n], loop[(e + 1) % n]
                path = np.array([p, *(x for x, _ in group), q])
                detour = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                               - np.linalg.norm(q - p))  # fmt: skip
                if best is None or detour < best[0]:
                    x = group[0][0]
                    along = float((x - p) @ (q - p)) / max(float((q - p) @ (q - p)), 1e-300)
                    best = (detour, e % n, along)
            if best[0] > 16.0 * h:
                return None  # a corner far off its contour: a feature that the grid cut
            for i, (x, angle) in enumerate(group):
                found_at.append((best[1], best[2] + 1e-12 * i, x, angle))
        found_at.sort(key=lambda c: (c[0], c[1]))
        points: list[np.ndarray] = []
        corners: list[int] = []
        angles: list[float] = []
        pending = 0
        for k in range(n):
            points.append(loop[k])
            while pending < len(found_at) and found_at[pending][0] == k:
                points.append(found_at[pending][2])
                corners.append(len(points) - 1)
                angles.append(found_at[pending][3])
                pending += 1
        points = np.asarray(points)
        if corners and not _simple(points):  # the corners left out of a contour they knot
            if body.capped:
                return None  # (the rims of flat caps: on a finer grid)
            points, corners, angles = loop, [], []
        if corners:  # a vertex within a millionth of a cell of a corner: the corner
            m = len(points)
            corner = np.zeros(m, dtype=bool)
            corner[corners] = True
            near = np.zeros(m, dtype=bool)
            for c in corners:
                for d in ((c - 1) % m, (c + 1) % m):
                    near[d] |= not corner[d] and np.linalg.norm(points[d] - points[c]) < 1e-6 * h
            keep = np.flatnonzero(~near)
            points, corners = points[keep], np.flatnonzero(corner[keep]).tolist()
        out.append((points, corners, angles))
    redundant = _redundant(body, [points for points, *_ in out], h)
    out = [traced for traced, extra in zip(out, redundant, strict=True) if not extra]
    # every corner of the body (two capsules that cross, no other covering it) among those
    # of its contours: else a feature smaller than a cell, a hole, that the grid missed
    found = [loop[c] for loop, corners, _ in out for c in corners]
    m = len(body.radii)
    if m > 1:
        i, j = np.triu_indices(m, 1)
        meet = _segment_distances(body.start[i], body.end[i], body.start[j], body.end[j])
        cross = meet < body.radii[i] + body.radii[j]
        polygons = {k: _section_polygon(body, k) for k in np.unique(i[cross]).tolist()}
        for a, b in zip(i[cross].tolist(), j[cross].tolist(), strict=True):
            for x in _pair_corners(body, a, b, tol, polygons[a]):
                if not found or min(np.linalg.norm(x - y) for y in found) > 1e-6 * h:
                    return None
    return out


def _grid_distances(body: Body, lower, h: float, shape, coarse: _Grid | None = None):
    """The distances of a body at the nodes of a grid (``lower``, cell ``h``, ``shape``
    nodes): a :class:`_Grid`. With the grid of a coarser trace (``coarse``, spanning this
    one), only at the nodes in its cells that the surface may cross and in their
    neighbours, so that every cell of this grid that the surface crosses has its own: the
    distance changes by no more than the distance between two points, so that a cell with
    a corner farther from the surface than its diagonal is of one sign, that of the
    corner, as are the nodes in it. None for more than :data:`TRACE_POINTS` nodes."""
    nx, ny = shape
    lower = np.asarray(lower, dtype=float)
    if coarse is None:
        keys = np.arange(nx * ny, dtype=np.int64)
    else:
        ci, cj = _near_cells(coarse)
        offsets = np.array([(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)])
        ci = (ci[:, None] + offsets[:, 0]).ravel()
        cj = (cj[:, None] + offsets[:, 1]).ravel()
        cnx, cny = coarse.shape
        valid = (ci >= 0) & (ci < cnx - 1) & (cj >= 0) & (cj < cny - 1)
        cells = np.unique(ci[valid] * cny + cj[valid])
        ci, cj = np.divmod(cells, cny)
        block = int(np.ceil(coarse.h / h)) + 3  # (a node about each, against round-off)
        i0 = np.floor((coarse.lower[0] + ci * coarse.h - lower[0]) / h).astype(np.int64) - 1
        j0 = np.floor((coarse.lower[1] + cj * coarse.h - lower[1]) / h).astype(np.int64) - 1
        steps = np.arange(block)
        i = np.clip(i0[:, None, None] + steps[None, :, None], 0, nx - 1)
        j = np.clip(j0[:, None, None] + steps[None, None, :], 0, ny - 1)
        keys = np.unique((i * ny + j).ravel())
    if len(keys) > TRACE_POINTS:
        return None
    grid = _Grid(lower, float(h), (nx, ny), keys, np.empty(0))
    grid.values = body.distance(grid.points(np.arange(len(keys))))
    return grid


def _near_cells(grid: _Grid) -> tuple[np.ndarray, np.ndarray]:
    """The cells of a grid that the surface may cross: those whose four nodes are known, none
    farther from the surface than the diagonal of the cell (the others are of one sign)."""
    nx, ny = grid.shape
    i, j = np.divmod(grid.keys, ny)
    k0 = np.flatnonzero((i < nx - 1) & (j < ny - 1))
    corners = [k0] + [grid.find(grid.keys[k0] + d) for d in (ny, 1, ny + 1)]
    complete = np.all(np.stack(corners) >= 0, axis=0)
    corners = [k[complete] for k in corners]
    farthest = np.max(np.abs(np.stack([grid.values[k] for k in corners])), axis=0)
    near = farthest <= np.sqrt(2.0) * grid.h * (1.0 + 1e-9)
    return i[corners[0][near]], j[corners[0][near]]


def _line_root(body: Body, axis: int, value: float, start: float, span: float, tol: float):
    """The point of the surface of a body on the line ``axis = value`` nearest to ``start``
    (its coordinate along the line): bracketed by the first change of sign of the distance
    from ``start`` outward, by steps of a quarter of ``span``, then Newton's steps within
    the bracket, bisection out of it; None if there is none within 64 ``span``."""
    other = 1 - axis

    def points(t) -> np.ndarray:
        p = np.empty((len(t), 2))
        p[:, axis], p[:, other] = value, t
        return p

    d0 = float(body.distance(points(np.array([start])))[0])
    if abs(d0) <= 1e-3 * tol:
        return start
    k = np.arange(1, 257)
    offsets = np.stack([-k, k], axis=1).ravel() * (0.25 * span)  # (nearest first)
    d = body.distance(points(start + offsets))
    changed = np.flatnonzero(np.sign(d) != np.sign(d0))
    if not len(changed):
        return None
    t1 = start + offsets[changed[0]]
    t0 = t1 - np.sign(offsets[changed[0]]) * 0.25 * span  # (the step before, of the sign)
    return _bracketed(body, axis, value, min(t0, t1), max(t0, t1), tol)


def _bracketed(body: Body, axis: int, value: float, low: float, high: float, tol: float):
    """The point of the surface of a body on the line ``axis = value`` between ``low`` and
    ``high`` (its coordinates along the line), where the distance changes sign: Newton's
    steps within the bracket, bisection out of it; None if it does not converge."""
    other = 1 - axis
    p = np.empty((1, 2))
    p[0, axis] = value

    def distance(t, gradient=False):
        p[0, other] = t
        return body.distance(p, gradient=gradient)

    sign_low = np.sign(float(distance(low)[0]))
    t = 0.5 * (low + high)
    for _ in range(100):
        d, g = distance(t, gradient=True)
        d, g = float(d[0]), float(g[0, other])
        if abs(d) <= 1e-3 * tol:
            return t
        if np.sign(d) == sign_low:
            low = t
        else:
            high = t
        newton = t - d / g if g != 0.0 else np.nan
        t = newton if low < newton < high else 0.5 * (low + high)
    return t if abs(float(distance(t)[0])) <= 1e3 * tol else None


def _along(loop: np.ndarray, e: int, x: np.ndarray) -> float:
    """The fraction of the edge ``e`` of a closed contour at which a point projects."""
    p, q = loop[e], loop[(e + 1) % len(loop)]
    return float((x - p) @ (q - p)) / max(float((q - p) @ (q - p)), 1e-300)


def _redundant(body: Body, loops: list[np.ndarray], h: float) -> list[bool]:
    """Whether each closed contour of a body is redundant: the trace of the grid about a node
    of it in a wedge narrower than a cell, which another contour of the body already passes
    around to its corner. Such a contour, within 16 cells, holds no other, and the phase
    well inside it is already that of the body there (its distance) between the others:
    with it, the contours would give the other phase."""
    out = []
    for k, loop in enumerate(loops):
        others = [other for j, other in enumerate(loops) if j != k]
        small = float(np.linalg.norm(loop.max(axis=0) - loop.min(axis=0))) <= 16.0 * h
        if not others or not small or any(_inside(other, loop).all() for other in others):
            out.append(False)
            continue
        q = _interior_point(loop)
        if q is None:
            out.append(False)
            continue
        between = sum(bool(_inside(q[None], other)[0]) for other in others) % 2 == 1
        out.append(between == bool(body.distance(q[None])[0] < 0.0))
    return out


def _inside(points: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Whether points lie inside a closed polygon (the rule of even and odd)."""
    points = np.atleast_2d(points)
    x, y = points[:, :1], points[:, 1:2]
    a, b = loop[None], np.roll(loop, -1, axis=0)[None]
    straddle = (a[..., 1] > y) != (b[..., 1] > y)
    rise = np.where(straddle, b[..., 1] - a[..., 1], 1.0)
    xs = a[..., 0] + (y - a[..., 1]) * (b[..., 0] - a[..., 0]) / rise
    return (straddle & (x < xs)).sum(axis=1) % 2 == 1


def _interior_point(loop: np.ndarray) -> np.ndarray | None:
    """A point well inside a closed polygon: of its centroid and of the points of a grid over
    its box, the one inside it farthest from its edges (None if none is inside)."""
    lo, hi = loop.min(axis=0), loop.max(axis=0)
    t = np.linspace(0.0, 1.0, 18)[1:-1]
    gx, gy = np.meshgrid(lo[0] + t * (hi[0] - lo[0]), lo[1] + t * (hi[1] - lo[1]))
    candidates = np.vstack([loop.mean(axis=0)[None], np.column_stack([gx.ravel(), gy.ravel()])])
    candidates = candidates[_inside(candidates, loop)]
    if not len(candidates):
        return None
    a, b = loop, np.roll(loop, -1, axis=0)
    ab = b - a
    t = np.einsum("ijk,jk->ij", candidates[:, None] - a[None], ab) / np.maximum(
        np.einsum("ij,ij->i", ab, ab), 1e-300
    )
    nearest = a[None] + np.clip(t, 0.0, 1.0)[..., None] * ab[None]
    distance = np.linalg.norm(candidates[:, None] - nearest, axis=2).min(axis=1)
    return candidates[int(np.argmax(distance))]


def section_contours(body: Body) -> list[np.ndarray]:
    """The closed contours of the section of a body: traced on a grid of its own (a
    :data:`TRACE_CELLS`-th of its least radius, finer where a gap between two of its capsules
    needs it), their vertices on its surface, its corners among them, where the surfaces of
    two capsules that cross meet."""
    traced = _traced(body)
    if traced is None:
        raise ValueError("section contours: a gap between two capsules of a body narrower "
                         "than the finest grid")  # fmt: skip
    return [loop for loop, *_ in traced]


def _self_gaps(loop: np.ndarray, ratio: float = 3.0) -> np.ndarray:
    """The gaps of a closed contour to itself: the distance of each vertex to the nearest of
    the others that are ``ratio`` times as far along it (the two sides of a narrow wedge
    at a corner, or of a neck)."""
    n = len(loop)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
    perimeter = cum[-1] + float(np.linalg.norm(loop[0] - loop[-1]))
    k = min(n, 16)
    dist, idx = cKDTree(loop).query(loop, k=k)
    along = np.abs(cum[idx] - cum[:, None])
    along = np.minimum(along, perimeter - along)
    far = (along > ratio * dist) & (idx != np.arange(n)[:, None])
    return np.where(far, dist, np.inf).min(axis=1)


def _curvature_radii(loop: np.ndarray) -> np.ndarray:
    """The radii of the circles through each vertex of a closed contour and its two
    neighbours."""
    a, b = np.roll(loop, 1, axis=0) - loop, np.roll(loop, -1, axis=0) - loop
    c = b - a
    area2 = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    lengths = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    return lengths / np.maximum(2.0 * area2, 1e-300)


def _grazing(body: Body, loop: np.ndarray, corners: list[int], lower, upper, tol: float):
    """A closed contour with the points of the surface beyond the lines of the sides that its
    chords miss, and its corners' indices among its new vertices: between two vertices on the
    same side of a line, nearer to it than their chord is long, the point of the surface (over
    the chord, onto the capsule or the face of both) farthest across the line, where it is
    across. A contour that grazes a side, its dip across the line shallower than the bulge
    of its chord, crosses it twice between two vertices: with the point, its chords do too."""
    following = np.roll(loop, -1, axis=0)
    chord = np.linalg.norm(following - loop, axis=1)
    pairs = []  # (edge, axis, value) of the chords near a line, not across it
    for axis in (0, 1):
        for value in (lower[axis], upper[axis]):
            u, v = loop[:, axis] - value, following[:, axis] - value
            near = ((u < 0.0) == (v < 0.0)) & (np.minimum(np.abs(u), np.abs(v)) < chord)
            pairs += [(e, axis, value) for e in np.flatnonzero(near).tolist()]
    if not pairs:
        return loop, corners
    edge = np.array([p[0] for p in pairs])
    axis = np.array([p[1] for p in pairs])
    value = np.array([p[2] for p in pairs], dtype=float)
    rows = np.arange(len(pairs))
    start, run = loop[edge], following[edge] - loop[edge]
    sign = np.where(start[rows, axis] >= value, 1.0, -1.0)  # (the side of the vertices)
    if len(body.radii) == 1:
        own = np.zeros(len(pairs), dtype=int)
    else:
        a, b = body.owners(start), body.owners(start + run)
        own = np.where(a == b, a, -1)
    face = np.full(len(pairs), -1)
    if body.capped:
        a, b = body.faces(start), body.faces(start + run)
        face = np.where(a == b, a, -1)

    def across(t: np.ndarray, rows=rows):
        """How far the surface over the chord at ``t`` along it is on the side of its
        vertices (negative: across the line), and its points."""
        k = rows if t.ndim == 1 else np.repeat(rows, t.shape[1])
        p = body.onto(start[k] + t.reshape(-1)[:, None] * run[k], own[k], face[k])
        return (sign[k] * (p[np.arange(len(k)), axis[k]] - value[k])).reshape(t.shape), p

    s = np.linspace(0.0, 1.0, 17)[1:-1]
    d, p = across(np.tile(s, (len(pairs), 1)))
    # the bulge of each arc from its chord: between two of its samples it strays from theirs
    # by a 256th of it, so that an arc whose samples all pass farther from the line than a
    # 16th of it is clear of the line
    over = start[:, None] + s[None, :, None] * run[:, None]  # (the points of the chords)
    bulge = np.linalg.norm(p.reshape(len(pairs), len(s), 2) - over, axis=2).max(axis=1)
    close = np.flatnonzero(d.min(axis=1) < bulge / 16.0)
    if not len(close):
        return loop, corners
    k = d[close].argmin(axis=1)
    lo = np.where(k > 0, s[np.maximum(k - 1, 0)], 0.0)
    hi = np.where(k + 1 < len(s), s[np.minimum(k + 1, len(s) - 1)], 1.0)
    golden = 0.5 * (np.sqrt(5.0) - 1.0)
    for _ in range(30):  # (the least of the distance across, by golden sections)
        c, f = hi - golden * (hi - lo), lo + golden * (hi - lo)
        left = across(c, close)[0] < across(f, close)[0]
        hi, lo = np.where(left, f, hi), np.where(left, lo, c)
    depth, points = across(0.5 * (lo + hi), close)
    found: dict[int, tuple[float, np.ndarray]] = {}
    for r in np.flatnonzero(depth < -tol).tolist():
        e = int(edge[close[r]])
        if e not in found or depth[r] < found[e][0]:
            found[e] = (float(depth[r]), points[r])
    if not found:
        return loop, corners
    edges = sorted(found)
    loop = np.insert(loop, [e + 1 for e in edges], [found[e][1] for e in edges], axis=0)
    return loop, [c + int(np.searchsorted(edges, c)) for c in corners]


def _line_crossings(body: Body, loop: np.ndarray, lower, upper, tol: float):
    """Where a contour crosses the lines of the sides of the rectangle, along it: ``(edge,
    fraction, axis, side, t)``, the edge of the contour and the fraction of it, ``t`` along
    the line, on the surface: the root between the points of the line beside the ends of the
    edge where the distance has their signs, else by Newton's method from the crossing of
    the chord, bracketed where it overshoots; None where neither converges (a contour
    tangent to a line)."""
    out = []
    following = np.roll(loop, -1, axis=0)
    for axis in (0, 1):
        other = 1 - axis
        for side, value in ((0, lower[axis]), (1, upper[axis])):
            u, v = loop[:, axis] - value, following[:, axis] - value
            for e in np.flatnonzero((u < 0.0) != (v < 0.0)).tolist():
                fraction = float(u[e] / (u[e] - v[e]))
                start = float(loop[e, other] + fraction * (following[e, other] - loop[e, other]))
                # the root between the points of the line beside the ends of the edge, where
                # the distance has their signs: that of this edge (the nearest to the
                # crossing of the chord may be that of the next, the other side of a dip)
                low, high = sorted((float(loop[e, other]), float(following[e, other])))
                p = np.empty((2, 2))
                p[:, axis], p[:, other] = value, (low, high)
                ends = body.distance(p)
                if high > low and np.sign(ends[0]) * np.sign(ends[1]) < 0.0:
                    t = _bracketed(body, axis, value, low, high, tol)
                    if t is not None:
                        out.append((e, fraction, axis, side, t))
                        continue
                t = start
                p = np.empty((1, 2))
                p[0, axis] = value
                for _ in range(NEWTON_STEPS):
                    p[0, other] = t
                    d, g = body.distance(p, gradient=True)
                    if abs(g[0, other]) < 1e-6:
                        t = np.nan
                        break
                    t -= float(d[0] / g[0, other])
                if np.isfinite(t):
                    p[0, other] = t
                if not np.isfinite(t) or abs(float(body.distance(p)[0])) > 1e3 * tol:
                    # (a contour that crosses the line at a grazing angle: Newton's steps
                    # overshoot) the root nearest to the start, bracketed
                    span = float(np.linalg.norm(following[e] - loop[e]))
                    t = _line_root(body, axis, value, start, span, tol)
                    if t is None:
                        return None
                out.append((e, fraction, axis, side, t))
    out.sort(key=lambda c: (c[0], c[1]))
    return out


def contour_mesh(micro, *, size: float, min_size: float | None = None,
                 hausdorff: float | None = None, gradation: float = 2.0, z: float | None = None,
                 phases="all", order: int = 1, curved: bool | None = None,
                 periodic: bool | None = None, executable: str | None = None, workdir=None,
                 timeout: float | None = None) -> TriMesh | None:  # fmt: skip
    """The triangular mesh of the section of a union of spheres and segments with round
    or flat caps, coated or hollow (:func:`section_layers`), in the plane ``z`` (default: the
    middle of its box) by the exact contours of its bodies (see the module) and of those
    of its coatings or bores, the phases between them: their nodes on the surfaces, at
    sizes graded from the gaps
    between the bodies (:data:`~morphogen.mesh.triangles.GAP_ELEMENTS` elements across the
    narrowest, the gaps the distances of the bodies), from their curvatures (the share of a
    circle of :data:`~morphogen.mesh.triangles.CIRCLE_SEGMENTS` of their radius) and at
    least ``min_size`` (by default a 64th of ``size``) — below it where a gap is narrower
    than the bulge of the chords —, toward the sharp corners
    (:data:`~morphogen.mesh.triangles.CORNER_REFINEMENT`), the sides of a periodic box
    matching; ``mmg2d`` meshes the regions within them, which keep their nodes. Six-node
    elements have their middle nodes on the surfaces, by default. None for another
    structure, capsules whose sections touch, a feature of a body (a gap, a
    corner) too small for the finest grid, a contour tangent to a side, or edges that
    cross."""
    from morphogen.mesh.conforming import mmg_executable

    domain = micro.domain
    size = float(size)
    if not size > 0:
        raise ValueError("size: must be positive")
    if order not in (1, 2):
        raise ValueError("order: 1 or 2")
    if curved and order == 1:
        raise ValueError("curved: the edges of six-node elements (order 2)")
    lower = np.asarray(domain.lower[:2], dtype=float)
    upper = np.asarray(domain.upper[:2], dtype=float)
    lengths = upper - lower
    periodic = bool(domain.periodic) if periodic is None else bool(periodic)
    if periodic and not domain.periodic:
        raise ValueError("matching periodic sides need a periodic structure")
    z = float(domain.lower[2] + 0.5 * domain.lengths[2]) if z is None else float(z)
    tol = 1e-9 * float(lengths.max())
    margin = size * GAP_ELEMENTS
    layers = section_layers(micro)
    if layers is None:
        return None
    unions, bands = layers
    bodies: list[Body] = []
    level: list[int] = []
    for k, (union, offset) in enumerate(zip(unions, _layer_offsets(micro), strict=True)):
        found = section_bodies(union, z, margin=margin, periodic=periodic, offset=offset)
        if found is None:
            return None
        bodies += found
        level += [k] * len(found)
    floor = size / 64.0 if min_size is None else float(min_size)
    # 1. the contours of the bodies, their corners, the lines of the sides that they cross,
    #    and the pieces of them inside the box between these marks: ("side", (axis, side,
    #    t)) where they cross a side, ("corner", (key, point)) at a corner
    loops: list[tuple[int, np.ndarray, list[int]]] = []
    pieces: list[tuple[int, np.ndarray, tuple | None, tuple | None, float]] = []
    crosses: list[set] = [set() for _ in bodies]
    sharp: list[np.ndarray] = []
    for b, body in enumerate(bodies):
        traced = _traced(body)
        if traced is None:
            return None  # (a gap within a body too narrow for the finest grid)
        for loop, corners, angles in traced:
            perimeter = float(np.linalg.norm(loop - np.roll(loop, 1, axis=0), axis=1).sum())
            if perimeter < 3.0 * floor:
                continue  # a contour within the least size: a hole or the tip of a cap
            loop, corners = _grazing(body, loop, corners, lower, upper, tol)
            loops.append((b, loop, corners))
            found = _line_crossings(body, loop, lower, upper, tol)
            if found is None:
                return None
            sharp += [loop[k] for k, angle in zip(corners, angles, strict=True)
                      if angle > CORNER_ANGLE]  # fmt: skip
            for *_, axis, side, t in found:
                if lower[1 - axis] - tol <= t <= upper[1 - axis] + tol:
                    crosses[b].add((axis, side))
            # the crossings in turn along their edges as their points on the surface (those of
            # the chord, near a corner of the box, may come in the other order)
            marks = []
            for e, _, axis, side, t in found:
                x = _point(axis, side, t, lower, upper)
                along = _along(loop, e, x)
                marks.append((e, min(max(along, 0.0), 1.0), ("side", (axis, side, t)), x, along))
            marks += [(k, 0.0, ("corner", ((len(loops), k), loop[k])), loop[k], 0.0)
                      for k in corners]  # fmt: skip
            if not marks:
                if np.all(loop[0] > lower + tol) and np.all(loop[0] < upper - tol):
                    pieces.append((b, np.vstack([loop, loop[:1]]), None, None, perimeter))
                continue
            marks.sort(key=lambda m: (m[0], m[4]))
            n = len(loop)
            for m0, m1 in zip(marks, [*marks[1:], marks[0]], strict=True):
                start, stop = m0[0] + m0[1], m1[0] + m1[1]
                if stop < start or m1 is m0:  # (around the end of the contour, or all of it)
                    stop += n
                inner = [i % n for i in range(int(np.floor(start)) + 1, int(np.ceil(stop)))]
                path = np.vstack([m0[3], loop[inner], m1[3]])
                step = np.linalg.norm(np.diff(path, axis=0), axis=1)
                if step.sum() <= 1e3 * tol:
                    continue
                cum = np.concatenate([[0.0], np.cumsum(step)])
                k = int(np.searchsorted(cum, 0.5 * cum[-1]))
                k = min(max(k, 1), len(path) - 1)
                w = (0.5 * cum[-1] - cum[k - 1]) / max(step[k - 1], 1e-300)
                middle = path[k - 1] + w * (path[k] - path[k - 1])
                if np.all(middle > lower + tol) and np.all(middle < upper - tol):
                    pieces.append((b, path, m0[2], m1[2], perimeter))
    # 2. the sizes: along the contours, from the gaps to the other bodies and to the sides
    #    that they do not cross, and from their curvatures; along the sides, from the gaps
    #    to the bodies that do not cross them
    spacing = size / 4.0
    centres = np.array([0.5 * (body.bounds[0] + body.bounds[1]) for body in bodies])
    extents = np.array([0.5 * float(np.linalg.norm(body.bounds[1] - body.bounds[0]))
                        for body in bodies])  # fmt: skip
    tree = cKDTree(centres) if len(bodies) else None
    reach = float(extents.max()) + margin if len(bodies) else 0.0
    points, h = [], []
    for b, loop, corners in loops:
        gap = np.full(len(loop), np.inf)
        for o in tree.query_ball_point(centres[b], extents[b] + reach):
            if o != b:  # (inside the body of another layer: across a coating)
                gap = np.minimum(gap, np.abs(bodies[o].distance(loop)))
        same = [other for c, other, _ in loops if c == b and other is not loop]
        if same:  # the other contours of the body (its vertices: within a cell)
            gap = np.minimum(gap, cKDTree(np.concatenate(same)).query(loop)[0])
        gap = np.minimum(gap, _self_gaps(loop))
        normal = None
        for axis in (0, 1):
            for side, value in ((0, lower[axis]), (1, upper[axis])):
                across = np.abs(loop[:, axis] - value)
                if (axis, side) not in crosses[b]:
                    gap = np.minimum(gap, across)
                    continue
                # a side that the contour crosses: its gap where the contour runs along it
                # (within GRAZING degrees), the thin wedges and lenses of a grazing crossing
                if normal is None:
                    normal = bodies[b].distance(loop, gradient=True)[1]
                    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]
                along = np.abs(normal[:, axis]) > np.cos(np.radians(GRAZING))
                gap = np.minimum(gap, np.where(along, across, np.inf))
        radii = _curvature_radii(loop)
        for k in corners:  # (a corner is no curvature: graded below if sharp)
            radii[[k - 1, k, (k + 1) % len(loop)]] = np.inf
        sized = np.clip(np.minimum(gap / GAP_ELEMENTS, 2.0 * np.pi * radii / CIRCLE_SEGMENTS),
                        floor, size)  # fmt: skip
        # below the least size where a gap is narrower than the bulge of its chords: those of
        # the size sqrt(2 rho gap) bulge by a quarter of it
        points.append(loop)
        bent = np.isfinite(radii)
        bulge = np.full(len(loop), np.inf)
        bulge[bent] = np.sqrt(2.0 * radii[bent] * gap[bent])
        h.append(np.minimum(sized, np.maximum(bulge, 1e-3 * floor)))
    for axis in (0, 1):
        other = 1 - axis
        n = max(2, int(np.ceil(lengths[other] / spacing)) + 1)
        t = np.linspace(lower[other], upper[other], n)
        for side, value in ((0, lower[axis]), (1, upper[axis])):
            p = np.empty((n, 2))
            p[:, axis], p[:, other] = value, t
            gap = np.full(n, np.inf)
            for o, body in enumerate(bodies):
                if (axis, side) not in crosses[o]:
                    lo, hi = body.bounds
                    if lo[axis] - margin < value < hi[axis] + margin:
                        gap = np.minimum(gap, np.abs(body.distance(p)))
            points.append(p)
            h.append(np.clip(gap / GAP_ELEMENTS, floor, size))
    field = _PointSizes(np.concatenate(points), np.concatenate(h), lower, upper, size, periodic)
    field.refine(sharp, CORNER_REFINEMENT, floor)  # (the fields are singular at sharp corners)
    # 3. the nodes of the sides: their corners, the crossings, the points graded between;
    #    those of the upper sides of a periodic box the translates of the lower ones
    xy: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    refs: list[int] = []
    side_nodes: dict[tuple[int, int], tuple[np.ndarray, list[int]]] = {}
    corner: dict[tuple[int, int], int] = {}
    cuts = [m[1] for piece in pieces for m in (piece[2], piece[3])
            if m is not None and m[0] == "side"]  # fmt: skip
    corner_ids: dict[tuple[int, int], int] = {}

    def corner_node(i: int, j: int) -> int:
        if (i, j) not in corner:
            corner[(i, j)] = len(xy)
            xy.append(np.array([(lower, upper)[i][0], (lower, upper)[j][1]]))
        return corner[(i, j)]

    for axis in (0, 1):
        other = 1 - axis
        for side in (0, 1):
            ts = [t for a, s, t in cuts if a == axis and (s == side or periodic)]
            ts = np.sort(np.asarray(ts, dtype=float))
            ts = ts[np.concatenate([[True], np.diff(ts) > tol])] if len(ts) else ts
            ts = ts[(ts > lower[other] + tol) & (ts < upper[other] - tol)]
            ts = np.concatenate([[lower[other]], ts, [upper[other]]])
            value = (lower, upper)[side][axis]

            def along(s, t0, axis=axis, other=other, value=value):
                p = np.empty((len(s), 2))
                p[:, axis], p[:, other] = value, t0 + s
                return p

            if periodic and side == 1:  # the translate of the lower side
                chain = list(side_nodes[(axis, 0)][0])
            else:
                chain = [ts[0]]
                for t0, t1 in itertools.pairwise(ts):
                    params = _marched(t1 - t0, lambda s, t0=t0: along(s, t0), field)
                    chain += [*(t0 + params[1:-1]), t1]
            ends = (corner_node(*((side, 0) if axis == 0 else (0, side))),
                    corner_node(*((side, 1) if axis == 0 else (1, side))))  # fmt: skip
            ids = [ends[0]]
            for t in chain[1:-1]:
                ids.append(len(xy))
                xy.append(along(np.array([t]), 0.0)[0])
            ids.append(ends[1])
            side_nodes[(axis, side)] = (np.asarray(chain), ids)
            edges += list(itertools.pairwise(ids))
            refs += [SIDE_REFS[(axis, side)]] * (len(ids) - 1)

    def node(mark) -> int:
        """The node of a mark: of its side, or of its corner (the first time, new)."""
        kind, value = mark
        if kind == "corner":
            key, point = value
            if key not in corner_ids:
                corner_ids[key] = len(xy)
                for (axis, side), (chain, ids) in side_nodes.items():
                    # (a corner on a side, where the contour crosses it: the node of the side)
                    if abs(point[axis] - (lower, upper)[side][axis]) <= 1e3 * tol:
                        k = int(np.argmin(np.abs(chain - point[1 - axis])))
                        if abs(chain[k] - point[1 - axis]) <= 1e3 * tol:
                            corner_ids[key] = ids[k]
                            break
                else:
                    xy.append(point)
            return corner_ids[key]
        axis, side, t = value
        chain, ids = side_nodes[(axis, side)]
        k = int(np.argmin(np.abs(chain - t)))
        if abs(chain[k] - t) > 1e3 * tol:  # pragma: no cover - merged above
            raise RuntimeError("contour mesh: a crossing off the nodes of its side")
        return ids[k]

    # (a piece that ends off its side, the contour leaving the box at a corner of it)
    for _, _, start, end, _ in pieces:
        for kind, value in (m for m in (start, end) if m is not None):
            if kind == "side":
                chain = side_nodes[value[:2]][0]
                if np.abs(chain - value[2]).min() > 1e3 * tol:
                    return None
    # 4. the nodes of the pieces of the contours inside the box, graded, on the surfaces
    for b, path, start, end, perimeter in pieces:
        step = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(step)])
        length = float(cum[-1])

        def at(s, path=path, cum=cum, step=step):
            k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(step) - 1)
            w = (s - cum[k]) / np.maximum(step[k], 1e-300)
            return path[k] + w[:, None] * (path[k + 1] - path[k])

        least = max(1 if length < floor else 2, int(np.ceil(CIRCLE_SEGMENTS * length / perimeter)))
        if start is None:  # a whole contour: three segments at least, no shorter than the least
            least = max(3, min(CIRCLE_SEGMENTS, int(perimeter / floor)))
        params = _marched(length, at, field, least)[1:-1]
        # each node onto the capsule of its edge of the path, whose vertices it bounds (those
        # of its marks, ambiguous at corners, those of their neighbours), the body's where
        # the edge passes from a capsule to another (smoothly: a corner is a mark)
        one = len(bodies[b].radii) == 1
        owner = np.zeros(len(path), dtype=int) if one else bodies[b].owners(path)
        if len(path) > 2:
            owner[0], owner[-1] = owner[1], owner[-2]
        k = np.clip(np.searchsorted(cum, params, side="right") - 1, 0, len(step) - 1)
        own = np.where(owner[k] == owner[k + 1], owner[k], -1) if len(path) > 2 else None
        faces = None
        if bodies[b].capped:  # (a flat cap and the lateral surface: the face of the edge)
            face = bodies[b].faces(path)
            if len(path) > 2:
                face[0], face[-1] = face[1], face[-2]
            faces = np.where(face[k] == face[k + 1], face[k], -1)
        inner = bodies[b].onto(at(params), own, faces)
        ids = [node(start) if start is not None else len(xy)]
        if start is None:
            xy.append(path[0])
        for p in inner:
            ids.append(len(xy))
            xy.append(p)
        ids.append(node(end) if end is not None else ids[0])
        edges += list(itertools.pairwise(ids))
        refs += [REF_CIRCLE] * (len(ids) - 1)
    xy = np.asarray(xy)
    edges = np.asarray(edges, dtype=np.int64)
    refs = np.asarray(refs, dtype=np.int64)
    if _crossing(xy, edges):  # pragma: no cover - the contours of bodies apart, graded
        return None
    shortest = float(np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1).min())
    exe = mmg_executable(executable, "mmg2d")
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        medit.write_mesh(folder / "contours.mesh", xy, None, np.zeros((0, 3), dtype=np.int64),
                         edges=edges, edge_refs=refs, required_vertices=np.arange(len(xy)),
                         required_edges=np.arange(len(edges)))  # fmt: skip
        args = [*_mmg_args(size, 0.5 * shortest, hausdorff, gradation), "-nr", "-nosurf"]
        try:
            out = _run(exe, folder / "contours.mesh", None, folder / "out.mesh", args, timeout)
        except RuntimeError as err:  # (left to the level sets, which may mesh it)
            first = str(err).splitlines()[0] if str(err) else type(err).__name__
            warnings.warn(f"contour mesh: {first}; the section is left to the level sets",
                          RuntimeWarning, stacklevel=2)  # fmt: skip
            return None
    nodes, tri = out["Vertices"], out["Triangles"]
    centroids = nodes[tri].mean(axis=1)
    innermost = np.full(len(tri), len(unions))
    if len(bodies):  # the triangles inside a body: the innermost layer that holds them
        near = cKDTree(centroids)
        for body, k_level, c, e in zip(bodies, level, centres, extents, strict=True):
            k = np.asarray(near.query_ball_point(c, e), dtype=np.int64)
            if len(k):
                k = k[body.distance(centroids[k]) < 0.0]
                innermost[k] = np.minimum(innermost[k], k_level)
    phase = np.asarray(bands, dtype=np.int64)[innermost]
    outermost = {b for b, *_ in pieces if level[b] == len(unions) - 1}
    mesh = TriMesh(nodes, tri, phase, lower, upper, periodic,
                   {"mesher": "exact contours (mmg2d)", "size": size,
                    "bodies": len(outermost), "layers": len(unions), "z": z}, z=z,
                   thickness=float(domain.lengths[2]))  # fmt: skip
    mesh = _kept(mesh, phases)
    if order == 1:
        return mesh
    if curved is False:
        return mesh.quadratic()
    return _curved(mesh, bodies)


def _curved(mesh: TriMesh, bodies: list[Body]) -> TriMesh:
    """The six-node mesh of a mesh of contours whose middle nodes on the contours lie on
    the surfaces: each edge between the phases curved onto the capsule on whose surface its
    two ends lie, its middle moved along the gradient of that capsule's distance (that of
    a body, the least of its capsules', would take the surface of the other near a corner);
    the middle nodes of the elements that would fold (:data:`MIN_JACOBIAN`) halved toward
    their middles, eight times at most, then straight."""
    t0 = time.perf_counter()
    quad = mesh.quadratic()
    nodes = quad.nodes.copy()
    ends = quad.elements[:, np.array(TRI6_EDGES)]
    mids = quad.elements[:, 3:]
    flat = mids.ravel()
    low = np.full(quad.n_nodes, np.iinfo(np.int64).max)
    high = np.full(quad.n_nodes, -1)
    np.minimum.at(low, flat, np.repeat(quad.phase, 3))
    np.maximum.at(high, flat, np.repeat(quad.phase, 3))
    chosen = np.flatnonzero((high >= 0) & (low != high))  # the middles between the phases
    if not len(chosen) or not bodies:
        return quad
    pair = np.zeros((quad.n_nodes, 2), dtype=np.int64)
    pair[flat] = ends.reshape(-1, 2)
    a, b = nodes[pair[chosen, 0]], nodes[pair[chosen, 1]]
    middle = 0.5 * (a + b)
    target = middle.copy()
    best = np.full(len(chosen), np.inf)
    tree = cKDTree(middle)
    for body in bodies:
        lo, hi = body.bounds
        near = np.asarray(tree.query_ball_point(0.5 * (lo + hi), 0.5 * np.linalg.norm(hi - lo)),
                          dtype=np.int64)  # fmt: skip
        if not len(near):
            continue
        for k in range(len(body.radii)):
            off = np.maximum(
                np.abs(body.capsule(k, a[near])[0]), np.abs(body.capsule(k, b[near])[0])
            )
            better = off < best[near]
            if not np.any(better):
                continue
            take = near[better]
            best[take] = off[better]
            target[take] = body.onto(middle[take], np.full(len(take), k))
            if body.capped and any(body._flat(k)):  # the face of both ends: a cap or not
                face = np.zeros(len(take), dtype=int)
                for f in (1, 2):
                    if body._flat(k)[f - 1]:
                        on = np.maximum(np.abs(body.face(k, f, a[take])[0]),
                                        np.abs(body.face(k, f, b[take])[0]))  # fmt: skip
                        face[on <= 1e-9 * float(body.radii.max())] = f
                target[take] = body.onto(middle[take], faces=3 * k + face)
    move = target - middle
    scale = np.ones(len(chosen))
    index = np.full(quad.n_nodes, -1)
    index[chosen] = np.arange(len(chosen))
    bent = np.flatnonzero((index[mids] >= 0).any(axis=1))
    gradients = corner_gradients(nodes, quad.elements[bent])
    points = np.vstack([TRI6_NODES, TRI_GAUSS])
    for round_ in range(9):
        trial = nodes.copy()
        trial[chosen] = middle + scale[:, None] * move
        deviation = trial[mids[bent]] - 0.5 * (trial[ends[bent, :, 0]] + trial[ends[bent, :, 1]])
        bad = bent[jacobian_ratios(gradients, deviation, points).min(axis=1) < MIN_JACOBIAN]
        if not len(bad):
            break
        mine = index[mids[bad]].ravel()
        mine = np.unique(mine[mine >= 0])
        scale[mine] = 0.0 if round_ == 7 else 0.5 * scale[mine]
    nodes[chosen] = middle + scale[:, None] * move
    out = quad._with_nodes(nodes)
    size = float(mesh.metadata.get("size", 1.0))
    on = int(np.count_nonzero((scale == 1.0) & (best <= 1e-9 * size)))
    deviation = float(np.linalg.norm(scale[:, None] * move, axis=1).max())
    out.metadata["curved"] = {"nodes": on, "interface_nodes": len(chosen),
                              "max_deviation": deviation, "sampling": 0.0,
                              "time_s": round(time.perf_counter() - t0, 3)}  # fmt: skip
    return out
