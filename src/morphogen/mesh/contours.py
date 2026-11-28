"""The exact contours of the sections of bodies in a plane: oblique and wavy fibres.

A union of spheres and segments (:class:`~morphogen.models.union.PrimitiveUnion`: the long
fibres of :class:`~morphogen.models.long_fibers.LongFibers`, misaligned or not, chains of
spherocylinders) cut by the plane ``z``. Only the axes within a radius of the plane reach
it: clipped to that slab, they keep the section (:func:`section_capsules`). The capsules
that meet are grouped into bodies, whose sections, those of disjoint sets, do not meet;
each body's contours are traced on its own distance function, the least of the distances
to its axes less their radii — the marching squares on a grid of its own, of a
:data:`TRACE_CELLS`-th of its least radius, their vertices moved onto the surface by
Newton's method (:func:`section_contours`). No gap between two bodies closes, however
narrow, where the level sets of the whole section
(:func:`~morphogen.mesh.triangles.section_mesh`) close those narrower than their
background: the sections of misaligned fibres packed at the fractions of composites.

:func:`contour_mesh` meshes them as :func:`~morphogen.mesh.triangles.disc_mesh` meshes
circles: the contours cut by the sides of the box, nodes on them at sizes graded from the
gaps between the bodies (their distances, exact) and from their curvatures, the sides of a
periodic box matching; ``mmg2d`` meshes the regions within them, which keep their nodes,
each triangle the phase of its centroid; six-node elements have their middle nodes on the
surfaces (:func:`~morphogen.mesh.curved.curved_triangles`).
"""

from __future__ import annotations

import itertools
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from morphogen.io import medit
from morphogen.mesh.triangles import (
    CIRCLE_SEGMENTS,
    GAP_ELEMENTS,
    GRADE,
    REF_CIRCLE,
    SIDE_REFS,
    TriMesh,
    _DiscSizes,
    _kept,
    _marched,
    _mmg_args,
    _point,
    _run,
)

#: The cells of the grid on which a body's contours are traced, across its least radius
TRACE_CELLS = 8
#: Newton's steps onto the surface of a body
NEWTON_STEPS = 6


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
            cut = np.abs(c[:, 2] - z) < r
            starts.append(c[cut])
            ends.append(c[cut])
            radii.append(r[cut])
        if len(segments):
            a = segments.start + np.array([0.0, 0.0, k * lz])
            b = segments.end + np.array([0.0, 0.0, k * lz])
            r = segments.radii
            lo, hi = np.minimum(a[:, 2], b[:, 2]), np.maximum(a[:, 2], b[:, 2])
            cut = (lo - r < z) & (hi + r > z)
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

    def onto(self, xy: np.ndarray) -> np.ndarray:
        """Points moved onto the surface of the body along its gradient (Newton)."""
        xy = np.array(xy, dtype=float)
        for _ in range(NEWTON_STEPS):
            value, grad = self.distance(xy, gradient=True)
            g2 = np.maximum(np.einsum("ij,ij->i", grad, grad), 1e-300)
            xy -= (value / g2)[:, None] * grad
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


def _marching_loops(f: np.ndarray, x0: float, y0: float, h: float) -> list[np.ndarray]:
    """The closed contours ``f = 0`` of values on a grid (``f[i, j]`` at ``(x0 + i h, y0 +
    j h)``, positive along its border) by the marching squares, the saddles resolved by the
    mean of their corners: lists of their points, linear between the nodes."""
    inside = f < 0.0
    across_x = inside[:-1, :] != inside[1:, :]  # the edges (i, j)-(i+1, j)
    across_y = inside[:, :-1] != inside[:, 1:]  # the edges (i, j)-(i, j+1)
    hx = np.full(across_x.shape, -1, dtype=np.int64)
    hy = np.full(across_y.shape, -1, dtype=np.int64)
    nh = int(across_x.sum())
    hx[across_x] = np.arange(nh)
    hy[across_y] = nh + np.arange(int(across_y.sum()))
    i, j = np.nonzero(across_x)
    t = f[i, j] / (f[i, j] - f[i + 1, j])
    px = np.column_stack([x0 + (i + t) * h, y0 + j * h])
    i, j = np.nonzero(across_y)
    t = f[i, j] / (f[i, j] - f[i, j + 1])
    py = np.column_stack([x0 + i * h, y0 + (j + t) * h])
    points = np.concatenate([px, py])
    if not len(points):
        return []
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


def section_contours(body: Body) -> list[np.ndarray]:
    """The closed contours of the section of a body: traced on a grid of its own (a
    :data:`TRACE_CELLS`-th of its least radius), their vertices on its surface."""
    h = float(body.radii.min()) / TRACE_CELLS
    lo, hi = body.bounds
    lo, hi = lo - 2.0 * h, hi + 2.0 * h
    nx, ny = (int(np.ceil((hi - lo)[k] / h)) + 1 for k in (0, 1))
    gx, gy = np.meshgrid(lo[0] + h * np.arange(nx), lo[1] + h * np.arange(ny), indexing="ij")
    values = body.distance(np.column_stack([gx.ravel(), gy.ravel()])).reshape(nx, ny)
    loops = []
    for loop in _marching_loops(values, float(lo[0]), float(lo[1]), h):
        loop = body.onto(loop)
        step = np.linalg.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0), axis=1)
        loop = loop[step > 1e-9 * h]  # (the points of a node of the grid on the surface)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


class _ContourSizes(_DiscSizes):
    """The size field of :class:`~morphogen.mesh.triangles._DiscSizes` from its points and
    their sizes, given."""

    def __init__(self, points, h, lower, upper, size, periodic):  # (not the circles')
        self.size = float(size)
        self.lower = np.asarray(lower, dtype=float)
        self.lengths = np.asarray(upper, dtype=float) - self.lower
        self.periodic = bool(periodic)
        self.points, self.h = np.asarray(points, dtype=float), np.asarray(h, dtype=float)
        self.reach = self.size / GRADE
        self._index()


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
    caps in the plane ``z`` (default: the middle of its box) by the exact contours of its
    bodies (see the module): their nodes on the surfaces, at sizes graded from the gaps
    between the bodies (:data:`~morphogen.mesh.triangles.GAP_ELEMENTS` elements across the
    narrowest, the gaps the distances of the bodies), from their curvatures (the share of a
    circle of :data:`~morphogen.mesh.triangles.CIRCLE_SEGMENTS` of their radius) and at
    least ``min_size`` (by default a 64th of ``size``), the sides of a periodic box
    matching; ``mmg2d`` meshes the regions within them, which keep their nodes. Six-node
    elements have their middle nodes on the surfaces
    (:func:`~morphogen.mesh.curved.curved_triangles`), by default. None for another
    structure, flat caps in the plane, capsules that touch, a body larger than the box (a
    cluster that percolates), or a contour tangent to a side."""
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
    bodies = section_bodies(micro, z, margin=margin, periodic=periodic)
    if bodies is None or any(np.any(np.subtract(*b.bounds[::-1]) > lengths) for b in bodies):
        return None  # (or a body larger than the box: a cluster that percolates)
    floor = size / 64.0 if min_size is None else float(min_size)
    # 1. the contours of the bodies, the lines of the sides that they cross, and the pieces
    #    of them inside the box between their crossings
    loops: list[tuple[int, np.ndarray]] = []
    pieces: list[tuple[int, np.ndarray, tuple | None, tuple | None, float]] = []
    crosses: list[set] = [set() for _ in bodies]
    for b, body in enumerate(bodies):
        for loop in section_contours(body):
            loops.append((b, loop))
            found = _line_crossings(body, loop, lower, upper, tol)
            if found is None:
                return None
            perimeter = float(np.linalg.norm(loop - np.roll(loop, 1, axis=0), axis=1).sum())
            if not found:
                if np.all(loop[0] > lower + tol) and np.all(loop[0] < upper - tol):
                    pieces.append((b, np.vstack([loop, loop[:1]]), None, None, perimeter))
                continue
            for *_, axis, side, t in found:
                if lower[1 - axis] - tol <= t <= upper[1 - axis] + tol:
                    crosses[b].add((axis, side))
            n = len(loop)
            for c0, c1 in zip(found, [*found[1:], found[0]], strict=True):
                p0 = _point(c0[2], c0[3], c0[4], lower, upper)
                p1 = _point(c1[2], c1[3], c1[4], lower, upper)
                k0, k1 = c0[0], c1[0]
                inner = [(k0 + 1 + i) % n for i in range((k1 - k0) % n)]
                if k1 == k0 and c1[1] <= c0[1]:  # around the whole contour
                    inner = [(k0 + 1 + i) % n for i in range(n)]
                path = np.vstack([p0, loop[inner], p1]) if inner else np.vstack([p0, p1])
                step = np.linalg.norm(np.diff(path, axis=0), axis=1)
                if step.sum() <= 1e3 * tol:
                    continue
                cum = np.concatenate([[0.0], np.cumsum(step)])
                k = int(np.searchsorted(cum, 0.5 * cum[-1]))
                k = min(max(k, 1), len(path) - 1)
                w = (0.5 * cum[-1] - cum[k - 1]) / max(step[k - 1], 1e-300)
                middle = path[k - 1] + w * (path[k] - path[k - 1])
                if np.all(middle > lower + tol) and np.all(middle < upper - tol):
                    pieces.append((b, path, c0[2:], c1[2:], perimeter))
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
    for b, loop in loops:
        gap = np.full(len(loop), np.inf)
        for o in tree.query_ball_point(centres[b], extents[b] + reach):
            if o != b:
                gap = np.minimum(gap, bodies[o].distance(loop))
        same = [other for c, other in loops if c == b and other is not loop]
        if same:  # the other contours of the body (its vertices: within a cell)
            gap = np.minimum(gap, cKDTree(np.concatenate(same)).query(loop)[0])
        for axis in (0, 1):
            for side, value in ((0, lower[axis]), (1, upper[axis])):
                if (axis, side) not in crosses[b]:
                    gap = np.minimum(gap, np.abs(loop[:, axis] - value))
        bend = 2.0 * np.pi * _curvature_radii(loop) / CIRCLE_SEGMENTS
        points.append(loop)
        h.append(np.minimum(gap / GAP_ELEMENTS, bend))
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
                        gap = np.minimum(gap, body.distance(p))
            points.append(p)
            h.append(gap / GAP_ELEMENTS)
    field = _ContourSizes(np.concatenate(points), np.clip(np.concatenate(h), floor, size),
                          lower, upper, size, periodic)  # fmt: skip
    # 3. the nodes of the sides: their corners, the crossings, the points graded between;
    #    those of the upper sides of a periodic box the translates of the lower ones
    xy: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    refs: list[int] = []
    side_nodes: dict[tuple[int, int], tuple[np.ndarray, list[int]]] = {}
    corner: dict[tuple[int, int], int] = {}
    cuts = [c for piece in pieces for c in (piece[2], piece[3]) if c is not None]

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

    def side_node(cut) -> int:
        axis, side, t = cut
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

        least = max(2, int(np.ceil(CIRCLE_SEGMENTS * length / perimeter)))
        inner = bodies[b].onto(at(_marched(length, at, field, least))[1:-1])
        ids = [side_node(start) if start is not None else len(xy)]
        if start is None:
            xy.append(path[0])
        for p in inner:
            ids.append(len(xy))
            xy.append(p)
        ids.append(side_node(end) if end is not None else ids[0])
        edges += list(itertools.pairwise(ids))
        refs += [REF_CIRCLE] * (len(ids) - 1)
    xy = np.asarray(xy)
    edges = np.asarray(edges, dtype=np.int64)
    refs = np.asarray(refs, dtype=np.int64)
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
    phase = np.zeros(len(tri), dtype=np.int64)
    if len(bodies):  # the triangles inside a body
        near = cKDTree(centroids)
        for body, c, e in zip(bodies, centres, extents, strict=True):
            k = np.asarray(near.query_ball_point(c, e), dtype=np.int64)
            if len(k):
                phase[k[body.distance(centroids[k]) < 0.0]] = 1
    inside = {b for b, *_ in pieces}
    mesh = TriMesh(nodes, tri, phase, lower, upper, periodic,
                   {"mesher": "exact contours (mmg2d)", "size": size, "bodies": len(inside),
                    "z": z}, z=z, thickness=float(domain.lengths[2]))  # fmt: skip
    mesh = _kept(mesh, phases)
    if order == 1:
        return mesh
    if curved is False:
        return mesh.quadratic()
    from morphogen.mesh.curved import curved_triangles

    return curved_triangles(mesh, micro)
