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
the two distances); where they pass to a capsule that they do not cross, they bridge a gap
narrower than a cell, which grids four and sixteen times as fine follow.

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
#: two of its capsules narrower than a cell: each a quarter of the last
TRACE_REFINEMENTS = 2
#: The most nodes of the grid of a body
TRACE_POINTS = 2_000_000
#: The least ratio of the area element of a curved six-node element to that of its straight
#: triangle: the middle nodes of the elements that would fold give way toward the middles
MIN_JACOBIAN = 0.3
#: The least angle (degrees) between the normals of the two surfaces at a corner of a
#: contour for the sizes to be graded toward it (``CORNER_REFINEMENT``)
CORNER_ANGLE = 10.0


def section_capsules(micro, z: float | None = None):
    """The spheres and segments of a union that reach the plane ``z`` (default: the middle
    of its box; their periodic images along ``z`` taken): their axes clipped to the slab
    within their radius of the plane, which keeps their sections — the points of the plane
    within a radius of an axis are within a radius of its part in the slab —, as
    ``(start, end, radii)``, a sphere an axis of no length; None for another structure, or
    for segments with flat caps that reach the plane, whose sections are not those of
    capsules."""
    from morphogen.models.union import PrimitiveUnion

    if not isinstance(micro, PrimitiveUnion):
        return None
    domain = micro.domain
    z = float(domain.lower[2] + 0.5 * domain.lengths[2]) if z is None else float(z)
    lz = float(domain.lengths[2])
    shifts = (-1.0, 0.0, 1.0) if domain.periodic else (0.0,)
    starts, ends, radii = [np.zeros((0, 3))], [np.zeros((0, 3))], [np.zeros(0)]
    spheres, segments = micro.spheres, micro.segments
    for k in shifts:
        if len(spheres):
            c = spheres.centers + np.array([0.0, 0.0, k * lz])
            r = spheres.radii
            cut = (np.abs(c[:, 2] - z) < r) & (r > 0.0)
            starts.append(c[cut])
            ends.append(c[cut])
            radii.append(r[cut])
        if len(segments):
            a = segments.start + np.array([0.0, 0.0, k * lz])
            b = segments.end + np.array([0.0, 0.0, k * lz])
            r = segments.radii
            lo, hi = np.minimum(a[:, 2], b[:, 2]), np.maximum(a[:, 2], b[:, 2])
            cut = (lo - r < z) & (hi + r > z) & (r > 0.0)
            a, b, r = a[cut], b[cut], r[cut]
            if not segments.round_caps and np.any(
                (np.abs(a[:, 2] - z) < r) | (np.abs(b[:, 2] - z) < r)
            ):
                return None  # a flat cap in the slab
            dz = b[:, 2] - a[:, 2]
            flat = np.abs(dz) <= 1e-15 * max(lz, 1.0)
            safe = np.where(flat, 1.0, dz)
            t0 = np.where(flat, 0.0, (z - r - a[:, 2]) / safe)
            t1 = np.where(flat, 1.0, (z + r - a[:, 2]) / safe)
            t0, t1 = np.clip(np.minimum(t0, t1), 0.0, 1.0), np.clip(np.maximum(t0, t1), 0.0, 1.0)
            d = b - a
            starts.append(a + t0[:, None] * d)
            ends.append(a + t1[:, None] * d)
            radii.append(r)
    return np.concatenate(starts), np.concatenate(ends), np.concatenate(radii)


def section_layers(micro):
    """The unions whose contours bound the phases of a section, from the innermost, and the
    phases inside the innermost, between them and outside them: a union of spheres and
    segments itself (phase 1 inside, 0 outside), or one coated
    (:class:`~morphogen.models.nested.CoatedStructure`: the union and that grown by the
    thickness of the coating, or the union shrunk by that of the shells and the union);
    None for another structure."""
    from morphogen.mesh.triangles import _grown
    from morphogen.models.nested import CoatedStructure
    from morphogen.models.union import PrimitiveUnion

    if isinstance(micro, PrimitiveUnion):
        return [micro], [1, 0]
    if not (isinstance(micro, CoatedStructure) and isinstance(micro.inner, PrimitiveUnion)):
        return None
    inner, t = micro.inner, float(micro.thickness)
    if micro.side == "outside":
        return [inner, _grown(inner, t)], [1, micro.phase, 0]
    return [_grown(inner, -t), inner], [micro.core_phase, 1, 0]


def _segment_distances(p1, q1, p2, q2) -> np.ndarray:
    """The distances between pairs of segments ``[p1, q1]`` and ``[p2, q2]`` (Ericson 2005,
    §5.1.9), points among them."""
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
    gap = p1 + s[:, None] * d1 - p2 - t[:, None] * d2
    return np.linalg.norm(gap, axis=1)


@dataclass
class Body:
    """A body of a section: the capsules of a union that meet, their axes ``start``,
    ``end`` (clipped to the slab of the plane, translated to an image across a periodic box)
    and ``radii``, in the plane ``z``."""

    start: np.ndarray
    end: np.ndarray
    radii: np.ndarray
    z: float

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
        for a, b, r in zip(self.start, self.end, self.radii, strict=True):
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
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        p = np.column_stack([xy, np.full(len(xy), self.z)])
        a, b, r = self.start[k], self.end[k], float(self.radii[k])
        d = b - a
        length2 = float(d @ d)
        t = np.clip((p - a) @ d / length2, 0.0, 1.0) if length2 > 0.0 else np.zeros(len(p))
        v = p - a - t[:, None] * d
        n = np.linalg.norm(v, axis=1)
        return n - r, v[:, :2] / np.maximum(n, 1e-300)[:, None]

    def owners(self, xy: np.ndarray) -> np.ndarray:
        """The capsule nearest to each point (whose surface bounds the body there)."""
        values = np.array([self.capsule(k, xy)[0] for k in range(len(self.radii))])
        return values.argmin(axis=0)

    def onto(self, xy: np.ndarray, capsules=None) -> np.ndarray:
        """Points moved onto the surface of the body along its gradient (Newton), or onto
        that of the capsule of each (``capsules``; -1: the body's)."""
        xy = np.array(xy, dtype=float).reshape(-1, 2)
        own = np.full(len(xy), -1) if capsules is None else np.asarray(capsules, dtype=int)
        for k in np.unique(own).tolist():
            at = own == k
            p = xy[at]
            for _ in range(NEWTON_STEPS):
                value, grad = self.distance(p, gradient=True) if k < 0 else self.capsule(k, p)
                g2 = np.maximum(np.einsum("ij,ij->i", grad, grad), 1e-300)
                p = p - (value / g2)[:, None] * grad
            xy[at] = p
        return xy


def section_bodies(micro, z: float | None = None, *, lower=None, upper=None,
                   margin: float = 0.0, periodic: bool | None = None):  # fmt: skip
    """The bodies of the section of a union in the plane ``z`` (:func:`section_capsules`):
    the capsules that meet (their axes nearer than the sum of their radii) grouped, the
    images across a periodic box taken where they come within ``margin`` of the rectangle
    ``lower``, ``upper`` (by default the box's); None as :func:`section_capsules`, or for
    capsules that touch, whose bodies would meet at a point."""
    capsules = section_capsules(micro, z)
    if capsules is None:
        return None
    domain = micro.domain
    z = float(domain.lower[2] + 0.5 * domain.lengths[2]) if z is None else float(z)
    lower = np.asarray(domain.lower[:2] if lower is None else lower, dtype=float)
    upper = np.asarray(domain.upper[:2] if upper is None else upper, dtype=float)
    lengths = np.asarray(domain.lengths[:2], dtype=float)
    periodic = bool(domain.periodic) if periodic is None else bool(periodic)
    start, end, radii = capsules
    if periodic and len(radii):
        shifts = np.array([(i, j, 0.0) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float)
        shifts[:, :2] *= lengths
        start = (start[None] + shifts[:, None]).reshape(-1, 3)
        end = (end[None] + shifts[:, None]).reshape(-1, 3)
        radii = np.tile(radii, len(shifts))
    lo = np.minimum(start[:, :2], end[:, :2]) - radii[:, None]
    hi = np.maximum(start[:, :2], end[:, :2]) + radii[:, None]
    near = np.all((hi > lower - margin) & (lo < upper + margin), axis=1)
    start, end, radii = start[near], end[near], radii[near]
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
            gap = _segment_distances(start[i], end[i], start[j], end[j]) - radii[i] - radii[j]
            if np.any(np.abs(gap) <= tol):
                return None  # capsules that touch
            for u, v in pairs[gap < 0.0].tolist():
                parent[root(u)] = root(v)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return [Body(start[k], end[k], radii[k], z) for k in groups.values()]


def _marching_loops(f: np.ndarray, x0: float, y0: float, h: float,
                    along=None) -> list[np.ndarray]:  # fmt: skip
    """The closed contours ``f = 0`` of values on a grid (``f[i, j]`` at ``(x0 + i h, y0 +
    j h)``, positive along its border) by the marching squares, the saddles resolved by the
    mean of their corners: lists of their points, linear between the nodes, or
    ``along(starts, steps, t)`` (the points ``starts + t steps`` of their edges of the grid,
    moved along them)."""
    inside = f < 0.0
    across_x = inside[:-1, :] != inside[1:, :]  # the edges (i, j)-(i+1, j)
    across_y = inside[:, :-1] != inside[:, 1:]  # the edges (i, j)-(i, j+1)
    hx = np.full(across_x.shape, -1, dtype=np.int64)
    hy = np.full(across_y.shape, -1, dtype=np.int64)
    nh = int(across_x.sum())
    hx[across_x] = np.arange(nh)
    hy[across_y] = nh + np.arange(int(across_y.sum()))
    i, j = np.nonzero(across_x)
    tx = f[i, j] / (f[i, j] - f[i + 1, j])
    sx = np.column_stack([x0 + i * h, y0 + j * h])
    i, j = np.nonzero(across_y)
    ty = f[i, j] / (f[i, j] - f[i, j + 1])
    sy = np.column_stack([x0 + i * h, y0 + j * h])
    starts, t = np.concatenate([sx, sy]), np.concatenate([tx, ty])
    if not len(t):
        return []
    steps = np.concatenate([np.tile([h, 0.0], (len(tx), 1)), np.tile([0.0, h], (len(ty), 1))])
    points = starts + t[:, None] * steps if along is None else along(starts, steps, t)
    bottom, top, left, right = hx[:, :-1], hx[:, 1:], hy[:-1, :], hy[1:, :]
    count = (bottom >= 0).astype(int) + (top >= 0) + (left >= 0) + (right >= 0)
    edges = np.stack([bottom, right, top, left], axis=-1)
    links = [np.sort(edges[count == 2], axis=1)[:, 2:]]
    four = count == 4
    if np.any(four):
        ci, cj = np.nonzero(four)
        centre = (f[ci, cj] + f[ci + 1, cj] + f[ci + 1, cj + 1] + f[ci, cj + 1]) < 0.0
        joined = (centre == inside[ci, cj])[:, None]  # the corners 00 and 11 joined
        b, r, tp, lf = (edges[four][:, k] for k in range(4))
        links.append(np.where(joined, np.column_stack([b, r]), np.column_stack([lf, b])))
        links.append(np.where(joined, np.column_stack([tp, lf]), np.column_stack([r, tp])))
    links = np.concatenate(links)
    neighbours = np.full((len(points), 2), -1, dtype=np.int64)
    degree = np.zeros(len(points), dtype=np.int64)
    for u, v in links.tolist():
        neighbours[u, degree[u]] = v
        degree[u] += 1
        neighbours[v, degree[v]] = u
        degree[v] += 1
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
    for _ in range(2 * NEWTON_STEPS):
        (va, ga), (vb, gb) = body.capsule(a, x), body.capsule(b, x)
        if max(abs(float(va[0])), abs(float(vb[0]))) <= 1e-3 * tol:
            break
        jacobian = np.array([ga[0], gb[0]])
        if abs(float(np.linalg.det(jacobian))) < 1e-6:
            return None
        x = x - np.linalg.solve(jacobian, np.array([va[0], vb[0]]))
    (va, ga), (vb, gb) = body.capsule(a, x), body.capsule(b, x)
    if max(abs(float(va[0])), abs(float(vb[0]))) > tol or body.distance(x)[0] < -tol:
        return None
    return x[0], float(np.degrees(np.arccos(np.clip(ga[0] @ gb[0], -1.0, 1.0))))


def _section_polygon(body: Body, k: int, n: int = 64) -> np.ndarray:
    """Points of the boundary of the section of the capsule ``k`` of a body, convex: along
    ``n`` rays from the point of its axis nearest to the plane, by bisection."""
    a, b, r = body.start[k], body.end[k], float(body.radii[k])
    d = b - a
    t = float(np.clip((body.z - a[2]) / d[2], 0.0, 1.0)) if d[2] != 0.0 else 0.5
    centre = (a + t * d)[:2]
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
        if corner is not None and all(np.linalg.norm(corner[0] - y) > 1e3 * tol for y in found):
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
    for _ in range(TRACE_REFINEMENTS + 1):
        out = _traced_at(body, h)
        if out is not None:
            return out
        h /= 4.0
    return None


def _traced_at(body: Body, h: float) -> list[tuple[np.ndarray, list[int], list[float]]] | None:
    """The contours of :func:`_traced` on the grid of cell ``h``; None for a bridge."""
    lo, hi = body.bounds
    lo, hi = lo - 2.0 * h, hi + 2.0 * h
    nx, ny = (int(np.ceil((hi - lo)[k] / h)) + 1 for k in (0, 1))
    if nx * ny > TRACE_POINTS:
        return None  # (a grid too large for the body)
    gx, gy = np.meshgrid(lo[0] + h * np.arange(nx), lo[1] + h * np.arange(ny), indexing="ij")
    values = body.distance(np.column_stack([gx.ravel(), gy.ravel()])).reshape(nx, ny)
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

    for loop in _marching_loops(values, float(lo[0]), float(lo[1]), h, along):
        step = np.linalg.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0), axis=1)
        loop = loop[step > 1e-9 * h]  # (the points of a node of the grid on the surface)
        if len(loop) < 3:
            continue
        owner = body.owners(loop) if len(body.radii) > 1 else np.zeros(len(loop), dtype=int)
        n = len(loop)
        found_at: list[tuple[int, float, np.ndarray, float]] = []  # (edge, along, x, angle)
        for k in range(n):
            a, b = int(owner[k]), int(owner[(k + 1) % n])
            if a == b:
                continue
            middle = 0.5 * (loop[k] + loop[(k + 1) % n])[None]
            found = _corner(body, a, b, middle, tol)
            if found is None:
                if body.distance(middle)[0] > 0.0:
                    return None  # a gap between the capsules bridged by the chord
                continue
            x, angle = found
            if any(np.linalg.norm(x - y) < 1e-6 * h for *_, y, _ in found_at):
                continue  # (the same corner, found again)
            # the corner on the edge near this one that it lengthens the least
            best = None
            for e in range(k - 2, k + 3):
                p, q = loop[e % n], loop[(e + 1) % n]
                detour = np.linalg.norm(x - p) + np.linalg.norm(q - x) - np.linalg.norm(q - p)
                if best is None or detour < best[0]:
                    along = float((x - p) @ (q - p)) / max(float((q - p) @ (q - p)), 1e-300)
                    best = (detour, e % n, along)
            if best[0] > 16.0 * h:
                return None  # a corner far off its contour: a feature that the grid cut
            found_at.append((best[1], best[2], x, angle))
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


def _line_crossings(body: Body, loop: np.ndarray, lower, upper, tol: float):
    """Where a contour crosses the lines of the sides of the rectangle, along it: ``(edge,
    fraction, axis, side, t)``, the edge of the contour and the fraction of it, ``t`` along
    the line, moved onto the surface by Newton's method; None where it does not converge
    (a contour tangent to a line)."""
    out = []
    following = np.roll(loop, -1, axis=0)
    for axis in (0, 1):
        other = 1 - axis
        for side, value in ((0, lower[axis]), (1, upper[axis])):
            u, v = loop[:, axis] - value, following[:, axis] - value
            for e in np.flatnonzero((u < 0.0) != (v < 0.0)).tolist():
                fraction = float(u[e] / (u[e] - v[e]))
                t = float(loop[e, other] + fraction * (following[e, other] - loop[e, other]))
                p = np.empty((1, 2))
                p[0, axis] = value
                for _ in range(NEWTON_STEPS):
                    p[0, other] = t
                    d, g = body.distance(p, gradient=True)
                    if abs(g[0, other]) < 1e-6:
                        return None
                    t -= float(d[0] / g[0, other])
                p[0, other] = t
                if abs(float(body.distance(p)[0])) > 1e3 * tol:
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
    caps, coated or hollow (:func:`section_layers`), in the plane ``z`` (default: the
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
    structure, flat caps in the plane, capsules that touch, a feature of a body (a gap, a
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
    for k, union in enumerate(unions):
        found = section_bodies(union, z, margin=margin, periodic=periodic)
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
            loops.append((b, loop, corners))
            found = _line_crossings(body, loop, lower, upper, tol)
            if found is None:
                return None
            sharp += [loop[k] for k, angle in zip(corners, angles, strict=True)
                      if angle > CORNER_ANGLE]  # fmt: skip
            for *_, axis, side, t in found:
                if lower[1 - axis] - tol <= t <= upper[1 - axis] + tol:
                    crosses[b].add((axis, side))
            marks = [(e, f, ("side", (axis, side, t)), _point(axis, side, t, lower, upper))
                     for e, f, axis, side, t in found]  # fmt: skip
            marks += [(k, 0.0, ("corner", ((len(loops), k), loop[k])), loop[k])
                      for k in corners]  # fmt: skip
            if not marks:
                if np.all(loop[0] > lower + tol) and np.all(loop[0] < upper - tol):
                    pieces.append((b, np.vstack([loop, loop[:1]]), None, None, perimeter))
                continue
            marks.sort(key=lambda m: (m[0], m[1]))
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
        for axis in (0, 1):
            for side, value in ((0, lower[axis]), (1, upper[axis])):
                if (axis, side) not in crosses[b]:
                    gap = np.minimum(gap, np.abs(loop[:, axis] - value))
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
                xy.append(point)
            return corner_ids[key]
        axis, side, t = value
        chain, ids = side_nodes[(axis, side)]
        k = int(np.argmin(np.abs(chain - t)))
        if abs(chain[k] - t) > 1e3 * tol:  # pragma: no cover - merged above
            raise RuntimeError("contour mesh: a crossing off the nodes of its side")
        return ids[k]

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
        inner = bodies[b].onto(at(params), own)
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
        out = _run(exe, folder / "contours.mesh", None, folder / "out.mesh", args, timeout)
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
