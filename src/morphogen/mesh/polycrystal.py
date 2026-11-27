"""Conforming tetrahedral meshes of polycrystals (Voronoi and Laguerre tessellations).

The exact cells of the tessellation (:mod:`morphogen.mesh.polyhedra`) are convex
polyhedra with planar facets; the features smaller than ``min_feature`` are removed
first (:mod:`morphogen.mesh.regularize`). Each piece is then split into tetrahedra, a cone
from its centroid over its triangulated facets: a valid mesh of the whole complex,
conforming at the grain boundaries, with poor elements. MMG (``mmg3d``) remeshes it to
the requested size, keeping the grain boundaries (interfaces between subdomains of
different references), their triple lines (non-manifold edges) and quadruple points.

Periodic polycrystals get matching opposite faces. The edges of the box are subdivided
once for the four parallel edges; each pair of opposite faces is meshed in two
dimensions from the traces of the grain boundaries (the faces of the regularized complex
are translates of each other), subdivided at the element size: ``mmg2d`` remeshes their
constrained Delaunay triangulation, keeping the traces. The same triangles are used on
both faces and the volume remeshing keeps them unchanged.

Quality: on Voronoi tessellations of 30 to 300 cells, the mean quality of the elements
is 0.72-0.74 and 0.1 % of them are below 0.05-0.1; the few worst ones (down to 0.004)
sit in the thin wedges between grain boundaries and faces of the box that remain.
"""

from __future__ import annotations

import itertools
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from morphogen.domain import Domain
from morphogen.io import medit
from morphogen.mesh.conforming import _run_mmg, mmg_executable
from morphogen.mesh.polyhedra import CellComplex, _merge_points
from morphogen.mesh.regularize import regularize
from morphogen.mesh.tetra import TetMesh, tet_volumes

log = logging.getLogger("morphogen")

#: Reference of the grain-boundary triangles (the faces of the box are 1-6).
REF_GRAIN_BOUNDARY = 7


def _others(axis: int) -> tuple[int, int]:
    return tuple(d for d in range(3) if d != axis)  # type: ignore[return-value]


class _Points:
    """Growing array of vertices."""

    def __init__(self, xyz: np.ndarray):
        self.blocks = [np.asarray(xyz, dtype=float)]
        self.n = len(xyz)

    def add(self, xyz) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        ids = np.arange(self.n, self.n + len(xyz))
        self.blocks.append(xyz)
        self.n += len(xyz)
        return ids

    def array(self) -> np.ndarray:
        if len(self.blocks) > 1:
            self.blocks = [np.concatenate(self.blocks)]
        return self.blocks[0]


def _fan(loop, centre: int) -> np.ndarray:
    """Triangles of a polygon: a fan from its centre (one triangle for three vertices)."""
    loop = np.asarray(loop, dtype=np.int64)
    if len(loop) == 3:
        return loop[None, :].copy()
    return np.column_stack([np.full(len(loop), centre), loop, np.roll(loop, -1)])


def _point_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Winding-number test of 2D points against a closed polygon."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    cross = (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)
    up = (y0 <= y) & (y1 > y) & (cross > 0)
    down = (y0 > y) & (y1 <= y) & (cross < 0)
    return (up.sum(axis=1) - down.sum(axis=1)) != 0


def _split_segments(points: np.ndarray, segments: list, tol: float) -> tuple[np.ndarray, list]:
    """Planar straight-line graph without crossings: segments split at the points lying
    on them and at their intersections."""
    pts = [p for p in points]
    segs = {tuple(sorted(s)) for s in segments if s[0] != s[1]}
    changed = True
    while changed:
        changed = False
        arr = np.array(pts)
        tree = cKDTree(arr)
        out = set()
        for a, b in segs:
            pa, pb = arr[a], arr[b]
            ab = pb - pa
            length = float(np.hypot(*ab))
            near = tree.query_ball_point(0.5 * (pa + pb), 0.5 * length + tol)
            inside = []
            for k in near:
                if k in (a, b):
                    continue
                t = float((arr[k] - pa) @ ab) / length**2
                if (
                    0.0 < t < 1.0
                    and abs(ab[0] * (arr[k] - pa)[1] - ab[1] * (arr[k] - pa)[0]) <= tol * length
                ):
                    inside.append((t, k))
            if inside:
                chain = [a] + [k for _, k in sorted(inside)] + [b]
                out.update(tuple(sorted(s)) for s in itertools.pairwise(chain))
                changed = True
            else:
                out.add((a, b))
        segs = out
        # proper crossings
        seg_list = sorted(segs)
        sa = np.array([arr[a] for a, _ in seg_list])
        sb = np.array([arr[b] for _, b in seg_list])
        lo, hi = np.minimum(sa, sb), np.maximum(sa, sb)
        crossings = []
        for i in range(len(seg_list)):
            cand = (
                np.flatnonzero(
                    np.all(lo[i + 1 :] <= hi[i] + tol, axis=1)
                    & np.all(hi[i + 1 :] >= lo[i] - tol, axis=1)
                )
                + i
                + 1
            )
            for j in cand:
                if set(seg_list[i]) & set(seg_list[j]):
                    continue
                d1, d2 = sb[i] - sa[i], sb[j] - sa[j]
                den = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(den) <= 1e-14:
                    continue
                w = sa[j] - sa[i]
                t = (w[0] * d2[1] - w[1] * d2[0]) / den
                u = (w[0] * d1[1] - w[1] * d1[0]) / den
                if 1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9:
                    crossings.append(sa[i] + t * d1)
        if crossings:
            pts.extend(crossings)
            changed = True
    return np.array(pts), sorted(segs)


def _periodic_faces(cx: CellComplex, pts: _Points, size, sizing2d, exe2d, folder, timeout):
    """Matching triangulations of opposite faces of the box (triangles of each box facet,
    in global vertex ids)."""
    lo, hi = cx.lower, cx.upper
    length = hi - lo
    tol = 1e-9 * float(length.max())
    xyz = pts.array()
    n0 = len(xyz)
    side = np.stack([np.abs(xyz - lo) <= tol, np.abs(xyz - hi) <= tol])  # (2, n, 3)
    # 1. the twelve edges of the box, subdivided alike along parallel edges
    for d in range(3):
        b, c = _others(d)
        params = []
        for sb in (0, 1):
            for sc in (0, 1):
                on = np.flatnonzero(side[sb, :n0, b] & side[sc, :n0, c])
                params.extend(xyz[on, d] - lo[d])
        t = np.unique(np.round(np.array([*params, 0.0, length[d]]) / tol) * tol)
        full = [t[0]]
        for t0, t1 in itertools.pairwise(t):
            m = int(np.ceil((t1 - t0) / size - 1e-9))
            full += [t0 + (t1 - t0) * j / m for j in range(1, m)] + [t1]
        full = np.array(full)
        for sb in (0, 1):
            for sc in (0, 1):
                on = np.flatnonzero(side[sb, :n0, b] & side[sc, :n0, c])
                have = xyz[on, d] - lo[d]
                missing = [s for s in full if not np.any(np.abs(have - s) <= 1e3 * tol)]
                if missing:
                    p = np.zeros((len(missing), 3))
                    p[:, d] = lo[d] + np.array(missing)
                    p[:, b] = (lo, hi)[sb][b]
                    p[:, c] = (lo, hi)[sc][c]
                    pts.add(p)
    xyz = pts.array()
    face_tris: dict[int, list] = {}
    for axis in range(3):
        b, c = _others(axis)
        uvw = [b, c]
        on_face = [np.flatnonzero(np.abs(xyz[:, axis] - (lo, hi)[s][axis]) <= tol) for s in (0, 1)]
        facets = [np.flatnonzero(cx.facet_box == 2 * axis + 1 + s) for s in (0, 1)]
        # 2D points of both faces (merged), and the segments of their polygons
        all2d = np.concatenate([xyz[on_face[0]][:, uvw], xyz[on_face[1]][:, uvw]])
        inverse = _merge_points(all2d, 100 * tol)
        first = np.zeros(int(inverse.max()) + 1, dtype=np.int64)
        first[inverse[::-1]] = np.arange(len(inverse))[::-1]
        pts2d = all2d[first]
        local = [
            dict(zip(on_face[0].tolist(), inverse[: len(on_face[0])].tolist(), strict=True)),
            dict(zip(on_face[1].tolist(), inverse[len(on_face[0]) :].tolist(), strict=True)),
        ]
        segments = []
        for s in (0, 1):
            for f in facets[s]:
                loop = [local[s][int(v)] for v in cx.facets[f]]
                segments += list(zip(loop, loop[1:] + loop[:1], strict=True))
        pts2d, segments = _split_segments(pts2d, segments, 1e3 * tol)
        bnd = set()
        for k in range(len(pts2d)):
            p = pts2d[k]
            if (
                (abs(p[0] - lo[b]) <= 1e3 * tol)
                or (abs(p[0] - hi[b]) <= 1e3 * tol)
                or (abs(p[1] - lo[c]) <= 1e3 * tol)
                or (abs(p[1] - hi[c]) <= 1e3 * tol)
            ):
                bnd.add(k)
        # boundary edges: consecutive points along each side of the square
        box_segs = []
        for coord, value in ((0, lo[b]), (0, hi[b]), (1, lo[c]), (1, hi[c])):
            on = [k for k in bnd if abs(pts2d[k][coord] - value) <= 1e3 * tol]
            on.sort(key=lambda k: pts2d[k][1 - coord])
            box_segs += [tuple(sorted(e)) for e in itertools.pairwise(on)]
        box_set = set(box_segs)
        inner = [
            s
            for s in segments
            if s not in box_set
            and not (s[0] in bnd and s[1] in bnd and _same_side(pts2d, s, lo, hi, b, c, 1e3 * tol))
        ]
        # the traces subdivided at the element size: mmg2d keeps them straight (it would
        # place the points it inserts on a curve through their vertices)
        pts2d, inner = _subdivide(pts2d, inner, size)
        edges = np.array(box_segs + inner, dtype=np.int64)
        # mmg2d remeshes a triangulation of the traces: building one itself (inserting
        # the segments in a triangulation of their points) fails on some inputs
        medit.write_mesh(
            folder / f"face{axis}.mesh",
            pts2d,
            triangles=_triangulate_pslg(pts2d, edges),
            edges=edges,
            edge_refs=np.array([1] * len(box_segs) + [2] * len(inner)),
            required_edges=np.arange(len(edges)),
            required_vertices=np.arange(len(pts2d)),
        )
        out = _run_mmg(
            exe2d,
            folder / f"face{axis}.mesh",
            None,
            folder / f"face{axis}.o.mesh",
            [*sizing2d, "-nosurf"],
            timeout,
        )
        ov, ot = out["Vertices"], out["Triangles"]
        oe = out.get("Edges", np.zeros((0, 2), dtype=np.int64))
        if len(oe):
            dist, _ = cKDTree(pts2d).query(ov[np.unique(oe)])
            if dist.max() > 1e3 * tol:
                raise RuntimeError(
                    "polycrystal mesh: mmg2d moved the traces of the grain boundaries"
                )
        # mmg2d is not deterministic (its triangles differ by a few flips from one run to
        # the next): the constrained Delaunay triangulation of its points is
        dist, where = cKDTree(ov).query(pts2d)
        if dist.max() > 1e3 * tol:
            raise RuntimeError("polycrystal mesh: mmg2d lost points of the traces")
        ot = _constrained_delaunay(ov, ot, where[edges])
        kept = {tuple(sorted(e)) for t in ot.tolist() for e in itertools.combinations(t, 2)}
        if not all(tuple(sorted(e)) in kept for e in where[edges].tolist()):
            raise RuntimeError("polycrystal mesh: mmg2d lost segments of the traces")
        centres = ov[ot].mean(axis=1)
        for s in (0, 1):
            plane = (lo, hi)[s][axis]
            # global ids of the output vertices on this face: existing vertices or new ones
            existing = on_face[s]
            tree = cKDTree(xyz[existing][:, uvw])
            dist, near = tree.query(ov)
            gid = np.where(dist <= 1e3 * tol, existing[near], -1)
            new = np.flatnonzero(gid < 0)
            if new.size:
                p = np.zeros((new.size, 3))
                p[:, axis] = plane
                p[:, b], p[:, c] = ov[new, 0], ov[new, 1]
                gid[new] = pts.add(p)
            xyz = pts.array()
            # each region of the face mesh lies in one polygon of this face
            for f in facets[s]:
                poly = xyz[cx.facets[f]][:, uvw]
                inside = _point_in_polygon(centres, poly)
                tris = gid[ot[inside]]
                face_tris.setdefault(int(f), []).append(tris)
        covered = sum(len(np.concatenate(face_tris[int(f)])) for s in (0, 1) for f in facets[s])
        if covered != 2 * len(ot):
            raise RuntimeError(
                "polycrystal mesh: the face meshes do not cover the polygons of the faces"
            )
    return {f: np.concatenate(t) for f, t in face_tris.items()}


def _orient(pa, pb, pc) -> float:
    """Twice the signed area of the triangle ``(pa, pb, pc)``, positive when it is
    counter-clockwise; its sign is exact (recomputed in rational arithmetic when rounding
    could change it, with the error bound of Shewchuk 1997)."""
    left = (pa[0] - pc[0]) * (pb[1] - pc[1])
    right = (pa[1] - pc[1]) * (pb[0] - pc[0])
    det = left - right
    if abs(det) > 1e-15 * (abs(left) + abs(right)):
        return det
    from fractions import Fraction

    ax, ay, bx, by, cx, cy = map(Fraction, (*pa, *pb, *pc))
    return float((ax - cx) * (by - cy) - (ay - cy) * (bx - cx))


def _triangulate_pslg(points: np.ndarray, segments) -> np.ndarray:
    """Constrained Delaunay triangulation of a planar straight-line graph whose segments
    meet only at their ends and include the boundary of the convex hull of the points:
    the Delaunay triangulation of the points, the missing segments recovered by edge
    flips (Sloan 1993), then Lawson's flips. Exact orientation tests make the flips
    valid whatever the input; a segment through a point is split there."""
    from collections import deque

    from scipy.spatial import Delaunay

    points = np.asarray(points, dtype=float)
    dt = Delaunay(points)
    if len(dt.coplanar):
        raise RuntimeError("polycrystal mesh: points of a face too close to triangulate")
    pts = [tuple(p) for p in points.tolist()]
    tris = [list(map(int, t)) for t in dt.simplices]
    owner: dict = {}  # directed edge -> triangle (counter-clockwise) holding it
    adj = [set() for _ in pts]
    for k, t in enumerate(tris):
        if _orient(pts[t[0]], pts[t[1]], pts[t[2]]) < 0:
            t[1], t[2] = t[2], t[1]
        a, b, c = t
        owner[(a, b)] = owner[(b, c)] = owner[(c, a)] = k
        adj[a] |= {b, c}
        adj[b] |= {a, c}
        adj[c] |= {a, b}

    def third(k, a, b):
        return next(v for v in tris[k] if v != a and v != b)

    def crossed(a, b):
        """Edges crossed by the segment from ``a`` to ``b`` (right end first), up to a
        point lying on it (or ``b``)."""
        pa, pb = pts[a], pts[b]
        for p in adj[a]:
            pp = pts[p]
            ahead = (pp[0] - pa[0]) * (pb[0] - pa[0]) + (pp[1] - pa[1]) * (pb[1] - pa[1]) > 0
            if ahead and _orient(pa, pb, pp) == 0:
                return [], p  # along an edge of the triangulation
        for p in adj[a]:
            k = owner.get((a, p))
            if k is None:
                continue
            q = third(k, a, p)
            if _orient(pa, pts[p], pb) > 0 and _orient(pa, pts[q], pb) < 0:
                break
        else:
            raise RuntimeError("polycrystal mesh: a segment of a face leaves its square")
        out = [(p, q)]
        while True:
            r = third(owner[(q, p)], p, q)
            if r == b:
                return out, b
            side = _orient(pa, pb, pts[r])
            if side == 0:
                return out, r
            if side > 0:
                q = r
            else:
                p = r
            out.append((p, q))

    fixed: set = set()
    todo = [tuple(map(int, s)) for s in np.asarray(segments).tolist()]
    while todo:
        a, b = todo.pop()
        if a == b:
            continue
        if b in adj[a]:
            fixed.add((min(a, b), max(a, b)))
            continue
        edges, end = crossed(a, b)
        if end != b:  # through a point: two segments
            todo += [(a, end), (end, b)]
            continue
        if any((min(e), max(e)) in fixed for e in edges):
            raise RuntimeError("polycrystal mesh: segments of a face cross each other")
        pa, pb = pts[a], pts[b]
        queue = deque(edges)
        budget = 10 * (len(edges) + 2) ** 3  # Sloan: O(n^2) flips, each after O(n) tries
        while queue:
            budget -= 1
            if budget < 0:
                raise RuntimeError("polycrystal mesh: the traces of a face cannot be inserted")
            u, v = queue.popleft()
            k1, k2 = owner[(u, v)], owner[(v, u)]
            w, x = third(k1, u, v), third(k2, u, v)
            if not (_orient(pts[u], pts[x], pts[w]) > 0 and _orient(pts[x], pts[v], pts[w]) > 0):
                queue.append((u, v))  # not convex: later
                continue
            # (u, v, w) and (v, u, x) become (u, x, w) and (x, v, w)
            del owner[(u, v)], owner[(v, u)]
            tris[k1], tris[k2] = [u, x, w], [x, v, w]
            for e in ((u, x), (x, w), (w, u)):
                owner[e] = k1
            for e in ((x, v), (v, w), (w, x)):
                owner[e] = k2
            adj[u].discard(v)
            adj[v].discard(u)
            adj[w].add(x)
            adj[x].add(w)
            if {w, x} & {a, b}:
                continue
            if _orient(pa, pb, pts[w]) * _orient(pa, pb, pts[x]) < 0:
                queue.append((w, x))
        fixed.add((min(a, b), max(a, b)))
    return _constrained_delaunay(points, np.array(tris, dtype=np.int64), sorted(fixed))


def _incircle(pa, pb, pc, pd) -> float:
    """Positive when ``pd`` lies inside the circle through the counter-clockwise
    triangle ``(pa, pb, pc)``."""
    adx, ady = pa[0] - pd[0], pa[1] - pd[1]
    bdx, bdy = pb[0] - pd[0], pb[1] - pd[1]
    cdx, cdy = pc[0] - pd[0], pc[1] - pd[1]
    ad, bd, cd = adx * adx + ady * ady, bdx * bdx + bdy * bdy, cdx * cdx + cdy * cdy
    return adx * (bdy * cd - bd * cdy) - ady * (bdx * cd - bd * cdx) + ad * (bdx * cdy - bdy * cdx)


def _constrained_delaunay(points: np.ndarray, triangles: np.ndarray, fixed) -> np.ndarray:
    """The constrained Delaunay triangulation of a triangulated polygon: Lawson's edge
    flips from any triangulation (counter-clockwise triangles) whose edges include the
    ``fixed`` ones. It is unique (but for cocircular points), so that triangulations of
    the same points differing by flips give the same result; triangles in a canonical
    order."""
    pts = [tuple(p) for p in np.asarray(points, dtype=float)]
    tris = [list(map(int, t)) for t in triangles]
    for t in tris:  # counter-clockwise
        (ax, ay), (bx, by), (cx, cy) = pts[t[0]], pts[t[1]], pts[t[2]]
        if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0:
            t[1], t[2] = t[2], t[1]
    owner = {}
    for k, (a, b, c) in enumerate(tris):
        owner[(a, b)] = owner[(b, c)] = owner[(c, a)] = k
    locked = {(min(a, b), max(a, b)) for a, b in np.asarray(fixed, dtype=np.int64).tolist()}
    stack = [(a, b) for (a, b) in owner if a < b and (a, b) not in locked]

    def third(t, a, b):
        return next(v for v in t if v != a and v != b)

    while stack:
        a, b = stack.pop()
        if (a, b) in locked:
            continue
        t1, t2 = owner.get((a, b)), owner.get((b, a))
        if t1 is None or t2 is None:
            continue
        c, d = third(tris[t1], a, b), third(tris[t2], a, b)
        pa, pb, pc, pd = pts[a], pts[b], pts[c], pts[d]
        scale = max((p[0] - pd[0]) ** 2 + (p[1] - pd[1]) ** 2 for p in (pa, pb, pc)) ** 2
        if _incircle(pa, pb, pc, pd) <= 1e-10 * scale:
            continue
        # (a, b, c) and (b, a, d) become (a, d, c) and (d, b, c)
        for e in ((a, b), (b, a), (b, c), (c, a), (a, d), (d, b)):
            owner.pop(e, None)
        tris[t1], tris[t2] = [a, d, c], [d, b, c]
        for e in ((a, d), (d, c), (c, a)):
            owner[e] = t1
        for e in ((d, b), (b, c), (c, d)):
            owner[e] = t2
        stack += [(min(e), max(e)) for e in ((a, d), (d, b), (b, c), (c, a))]
    out = np.array(tris, dtype=np.int64)
    first = np.argmin(out, axis=1)  # rotate the smallest vertex first, keep the orientation
    out = np.stack([out[np.arange(len(out)), (first + k) % 3] for k in range(3)], axis=1)
    return out[np.lexsort(out.T[::-1])]


def _subdivide(points: np.ndarray, segments: list, size: float) -> tuple[np.ndarray, list]:
    """Segments split into equal parts no longer than ``size``."""
    new, out = [], []
    n = len(points)
    for a, b in segments:
        pa, pb = points[a], points[b]
        m = int(np.ceil(float(np.hypot(*(pb - pa))) / size - 1e-9))
        if m <= 1:
            out.append((a, b))
            continue
        ids = [a, *range(n + len(new), n + len(new) + m - 1), b]
        new.extend(pa + (pb - pa) * j / m for j in range(1, m))
        out.extend(itertools.pairwise(ids))
    if new:
        points = np.concatenate([points, np.array(new)])
    return points, out


def _same_side(pts2d, s, lo, hi, b, c, tol) -> bool:
    """Whether a segment lies along a side of the square."""
    p, q = pts2d[s[0]], pts2d[s[1]]
    for coord, value in ((0, lo[b]), (0, hi[b]), (1, lo[c]), (1, hi[c])):
        if abs(p[coord] - value) <= tol and abs(q[coord] - value) <= tol:
            return True
    return False


def _insert_face_points(cx: CellComplex, pts: _Points, tol: float) -> list:
    """Loops of the facets with the points of the faces of the box lying on their edges
    (the face meshes subdivide the traces of the grain boundaries)."""
    xyz = pts.array()
    lo, hi = cx.lower, cx.upper
    loops = []
    trees = {}
    for axis in range(3):
        for s in (0, 1):
            ids = np.flatnonzero(np.abs(xyz[:, axis] - (lo, hi)[s][axis]) <= tol)
            trees[(axis, s)] = (ids, cKDTree(xyz[ids]) if ids.size else None)
    for f, loop in enumerate(cx.facets):
        if cx.facet_box[f] > 0:
            loops.append(loop)
            continue
        out = []
        for a, b in zip(loop, np.roll(loop, -1), strict=True):
            out.append(int(a))
            pa, pb = xyz[a], xyz[b]
            inside = {}
            for axis in range(3):
                for s in (0, 1):
                    plane = (lo, hi)[s][axis]
                    if abs(pa[axis] - plane) > tol or abs(pb[axis] - plane) > tol:
                        continue
                    ids, tree = trees[(axis, s)]
                    if tree is None:
                        continue
                    ab = pb - pa
                    length = float(np.linalg.norm(ab))
                    near = tree.query_ball_point(0.5 * (pa + pb), 0.5 * length + tol)
                    for k in near:
                        w = int(ids[k])
                        if w in (a, b):
                            continue
                        t = float((xyz[w] - pa) @ ab) / length**2
                        if 0.0 < t < 1.0 and np.linalg.norm(pa + t * ab - xyz[w]) <= 1e3 * tol:
                            inside[w] = t
            out.extend(sorted(inside, key=inside.get))
        loops.append(np.array(out, dtype=np.int64))
    return loops


def initial_mesh(cx: CellComplex, *, size, sizing, periodic, folder, timeout):
    """Tetrahedra of the pieces (cones from their centroids over their triangulated
    facets), with the cell + 1 of each tetrahedron, the triangles of the faces of the box
    (references 1-6, first ``n_box`` of them) and of the grain boundaries."""
    tol = 1e-9 * float((cx.upper - cx.lower).max())
    pts = _Points(cx.vertices)
    face_tris: dict = {}
    if periodic:
        face_tris = _periodic_faces(
            cx, pts, size, sizing, mmg_executable(None, "mmg2d"), folder, timeout
        )
        loops = _insert_face_points(cx, pts, tol)
    else:
        loops = list(cx.facets)
    xyz = pts.array()
    facet_tris = []
    for f, loop in enumerate(loops):
        if f in face_tris:
            facet_tris.append(face_tris[f])
            continue
        # fan from the centre of the facet's own vertices (as checked by the
        # regularization), over its loop with the points added on its edges
        centre = -1
        if len(loop) > 3:
            centre = int(pts.add(xyz[cx.facets[f]].mean(axis=0))[0])
        facet_tris.append(_fan(loop, centre))
    xyz = pts.array()
    tets, refs = [], []
    for p, pf in enumerate(cx.piece_facets):
        members = np.unique(np.concatenate([cx.facets[f] for f in pf]))
        g = int(pts.add(xyz[members].mean(axis=0))[0])
        for f in pf:
            t = facet_tris[f]
            tets.append(np.column_stack([t, np.full(len(t), g)]))
            refs.append(np.full(len(t), cx.piece_cell[p] + 1))
    xyz = pts.array()
    tets = np.concatenate(tets)
    refs = np.concatenate(refs)
    vol = tet_volumes(xyz, tets)
    tets[vol < 0] = tets[vol < 0][:, [0, 2, 1, 3]]
    tri, tri_refs = [], []
    for f in range(cx.n_facets):
        if cx.facet_box[f] > 0:
            tri.append(facet_tris[f])
            tri_refs.append(np.full(len(facet_tris[f]), cx.facet_box[f]))
    n_box = sum(len(t) for t in tri)
    for f in cx.interface_facets():
        tri.append(facet_tris[f])
        tri_refs.append(np.full(len(facet_tris[f]), REF_GRAIN_BOUNDARY))
    return xyz, tets, refs, np.concatenate(tri), np.concatenate(tri_refs), n_box


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from 2D points to the nearest of the segments ``a[k] b[k]``."""
    ab = b - a
    t = ((points[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-300)
    proj = a[None] + np.clip(t, 0.0, 1.0)[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - proj, axis=-1).min(axis=1)


def _inside(triangles: np.ndarray, m: int) -> np.ndarray:
    """Triangles (counter-clockwise) inside the polygon of the points ``0, ..., m - 1``
    (counter-clockwise), reached from its edges without crossing them: unlike a test of
    their centroids, this leaves out the flat triangles of nearly collinear points that a
    triangulation of the convex hull lays along the edges."""
    owner = {}
    for k, (a, b, c) in enumerate(triangles.tolist()):
        owner[(a, b)] = owner[(b, c)] = owner[(c, a)] = k
    boundary = {(k, (k + 1) % m) for k in range(m)}
    blocked = boundary | {(b, a) for a, b in boundary}
    inside = np.zeros(len(triangles), dtype=bool)
    stack = [owner[e] for e in boundary if e in owner]
    while stack:
        k = stack.pop()
        if inside[k]:
            continue
        inside[k] = True
        a, b, c = triangles[k].tolist()
        for e in ((a, b), (b, c), (c, a)):
            if e not in blocked:
                nb = owner.get((e[1], e[0]))
                if nb is not None and not inside[nb]:
                    stack.append(nb)
    return inside


def _mesh_polygon(loop: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of a (nearly planar) polygon at the element size: the points of a
    triangular lattice of spacing ``size`` inside it, at least half of it from the
    boundary, and the constrained Delaunay triangulation of its corners with them; the
    points subdividing its sides, collinear but for rounding (which could make flat
    triangles of three of them), are then joined to the vertex facing their side, before
    Lawson's flips. Returns the interior points, lifted onto the fan of the loop from its
    centroid (exact for a planar polygon), and the triangles (indices: the loop points,
    then the interior points), counter-clockwise around the normal of the loop."""
    c = loop.mean(axis=0)
    normal = np.cross(loop - c, np.roll(loop, -1, axis=0) - c).sum(axis=0)
    normal /= np.linalg.norm(normal)
    u = loop[0] - c
    u -= normal * (u @ normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    p2 = np.column_stack([(loop - c) @ u, (loop - c) @ v])
    m = len(p2)
    back, ahead = p2 - np.roll(p2, 1, axis=0), np.roll(p2, -1, axis=0) - p2
    turn = back[:, 0] * ahead[:, 1] - back[:, 1] * ahead[:, 0]
    bend = np.linalg.norm(back, axis=1) * np.linalg.norm(ahead, axis=1)
    corners = np.flatnonzero((np.abs(turn) > 1e-6 * bend) | ((back * ahead).sum(axis=1) < 0))
    if len(corners) < 3:
        raise RuntimeError("polycrystal mesh: a facet has less than three corners")
    lo, hi = p2.min(axis=0), p2.max(axis=0)
    dy = 0.5 * np.sqrt(3.0) * size
    rows = []
    for i, y in enumerate(np.arange(lo[1], hi[1] + dy, dy)):
        xs = np.arange(lo[0] + (0.5 * size if i % 2 else 0.0), hi[0] + size, size)
        rows.append(np.column_stack([xs, np.full(len(xs), y)]))
    cand = np.concatenate(rows)
    cand = cand[_point_in_polygon(cand, p2)]
    if len(cand):
        cand = cand[_segment_distance(cand, p2, np.roll(p2, -1, axis=0)) >= 0.5 * size]
    mc = len(corners)
    ring = np.column_stack([np.arange(mc), (np.arange(mc) + 1) % mc])
    tri = _triangulate_pslg(np.vstack([p2[corners], cand]), ring)
    index = np.concatenate([corners, m + np.arange(len(cand))])
    tris = [list(map(int, t)) for t in index[tri[_inside(tri, mc)]]]
    owner = {}
    for k, (a, b, w) in enumerate(tris):
        owner[(a, b)] = owner[(b, w)] = owner[(w, a)] = k
    for i in range(mc):
        a, b = int(corners[i]), int(corners[(i + 1) % mc])
        side = [a, *((a + j) % m for j in range(1, (b - a) % m)), b]
        if len(side) == 2:
            continue
        k = owner.pop((a, b))
        w = next(x for x in tris[k] if x != a and x != b)
        fan = [[p, q, w] for p, q in itertools.pairwise(side)]
        slots = [k, *range(len(tris), len(tris) + len(fan) - 1)]
        tris.extend(fan[1:])
        tris[k] = fan[0]
        for slot, (p, q, x) in zip(slots, fan, strict=True):
            owner[(p, q)] = owner[(q, x)] = owner[(x, p)] = slot
    points = np.vstack([p2, cand])
    edges = np.column_stack([np.arange(m), (np.arange(m) + 1) % m])
    tri = _constrained_delaunay(points, np.array(tris, dtype=np.int64), edges)
    # lift: barycentric coordinates in the fan triangle (centroid, loop[k], loop[k + 1])
    lifted = np.zeros((len(cand), 3))
    if len(cand):
        a2, b2 = p2, np.roll(p2, -1, axis=0)
        a3, b3 = loop, np.roll(loop, -1, axis=0)
        e1, e2 = a2[None], b2[None]
        x = cand[:, None, :]
        den = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
        la = (x[..., 0] * e2[..., 1] - x[..., 1] * e2[..., 0]) / den
        lb = (e1[..., 0] * x[..., 1] - e1[..., 1] * x[..., 0]) / den
        k = np.argmax(np.minimum(np.minimum(la, lb), 1.0 - la - lb), axis=1)
        idx = np.arange(len(cand))
        la, lb = la[idx, k], lb[idx, k]
        lifted = c + la[:, None] * (a3[k] - c) + lb[:, None] * (b3[k] - c)
    return lifted, tri


def _seen(xyz: np.ndarray, triangles: np.ndarray, apex: np.ndarray, sign: float) -> bool:
    """Whether the cones from ``apex`` over the triangles all have the orientation
    ``sign`` (the triangles are seen from the same side as the fan of their facet)."""
    p = xyz[triangles]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), apex - p[:, 0])
    return bool(np.all(sign * vol > 1e-9 * np.abs(vol).sum()))


def whole_initial_mesh(cx: CellComplex, *, size: float):
    """Tetrahedra of whole periodic cells (cones from their centroids over their
    triangulated facets), with the cell + 1 of each tetrahedron, the triangles bounding
    their union (first ``n_outer`` of them) and those of the grain boundaries. The edges are
    subdivided at the element size; each pair of translated facets bounding the union is
    meshed once (:func:`_mesh_polygon`), the translate taking the same triangles. The
    regularization checked that the fans of the facets from their centres are seen from
    the centroids of the pieces; a triangulation of a warped facet that is not (one of its
    triangles seen from the other side) is replaced by that fan."""
    from morphogen.mesh.polyhedra import torus_classes

    size_box = cx.upper - cx.lower
    tol = 1e-9 * float(size_box.max())
    pts = _Points(cx.vertices)
    vertices = cx.vertices
    apex = np.array(
        [vertices[np.unique(np.concatenate([cx.facets[f] for f in pf]))].mean(axis=0)
         for pf in cx.piece_facets]
    )  # fmt: skip
    subdivided: dict = {}
    for loop in cx.facets:
        for a, b in zip(loop.tolist(), np.roll(loop, -1).tolist(), strict=True):
            key = (min(a, b), max(a, b))
            if key in subdivided:
                continue
            pa, pb = vertices[key[0]], vertices[key[1]]
            m = int(np.ceil(float(np.linalg.norm(pb - pa)) / size - 1e-9))
            inner = [pa + (pb - pa) * j / m for j in range(1, m)]
            subdivided[key] = list(pts.add(np.array(inner))) if inner else []

    def full_loop(f: int) -> np.ndarray:
        out = []
        loop = cx.facets[f].tolist()
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
            out.append(a)
            inner = subdivided[(min(a, b), max(a, b))]
            out.extend(inner if a < b else inner[::-1])
        return np.array(out, dtype=np.int64)

    outer = np.flatnonzero(cx.facet_pieces[:, 1] < 0)
    labels, _ = torus_classes(vertices, cx.lower, size_box, 0.1 * tol)  # exact translates
    pairs: dict = {}
    for f in outer:
        pairs.setdefault(tuple(sorted(labels[cx.facets[f]].tolist())), []).append(int(f))
    facet_tris: dict = {}
    for group in pairs.values():
        if len(group) != 2:
            raise RuntimeError("polycrystal mesh: a facet bounding the cells has no translate")
        f, g = sorted(group)
        loop_f, loop_g = full_loop(f), full_loop(g)
        xyz = pts.array()
        lifted, tri = _mesh_polygon(xyz[loop_f], size)
        # the translation from f to g, from a vertex and its translate
        v0 = cx.facets[f][0]
        w0 = next(w for w in cx.facets[g] if labels[w] == labels[v0])
        offset = vertices[w0] - vertices[v0]
        dist, near = cKDTree(xyz[loop_g]).query(xyz[loop_f] + offset)
        if dist.max() > tol:
            raise RuntimeError("polycrystal mesh: translated facets differ")
        # the cones over the fan of f (from the centre of its vertices) and over the
        # triangles, from the centroids of the pieces of f and g
        centre = vertices[cx.facets[f]].mean(axis=0)
        m = len(loop_f)
        local = np.vstack([xyz[loop_f], lifted, centre])  # loop, interior points, centre
        fan = np.column_stack([np.full(m, len(local) - 1), np.arange(m), (np.arange(m) + 1) % m])
        ok = True
        for facet, shift in ((f, 0.0), (g, offset)):
            a = apex[cx.facet_pieces[facet, 0]] - shift
            p = local[fan]
            vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), a - p[:, 0])
            ok = ok and _seen(local, tri, a, np.sign(vol.sum()))
        if not ok:  # the fan of the facet and its translate
            c_f, c_g = pts.add(centre)[0], pts.add(centre + offset)[0]
            facet_tris[f] = _fan(loop_f, c_f)
            facet_tris[g] = _fan(loop_g[near], c_g)[:, ::-1]
            continue
        ids_f = np.concatenate([loop_f, pts.add(lifted) if len(lifted) else []]).astype(np.int64)
        ids_g = np.concatenate(
            [loop_g[near], pts.add(lifted + offset) if len(lifted) else []]
        ).astype(np.int64)
        facet_tris[f] = ids_f[tri]
        facet_tris[g] = ids_g[tri][:, ::-1]  # the translate is seen from the other side
    xyz = pts.array()
    for f in range(cx.n_facets):
        if f in facet_tris:
            continue
        loop = full_loop(f)
        centre = -1
        if len(loop) > 3:
            centre = int(pts.add(xyz[cx.facets[f]].mean(axis=0))[0])
        facet_tris[f] = _fan(loop, centre)
    xyz = pts.array()
    tets, refs = [], []
    for p, pf in enumerate(cx.piece_facets):
        members = np.unique(np.concatenate([cx.facets[f] for f in pf]))
        g = int(pts.add(xyz[members].mean(axis=0))[0])
        for f in pf:
            t = facet_tris[int(f)]
            tets.append(np.column_stack([t, np.full(len(t), g)]))
            refs.append(np.full(len(t), cx.piece_cell[p] + 1))
    xyz = pts.array()
    tets = np.concatenate(tets)
    refs = np.concatenate(refs)
    vol = tet_volumes(xyz, tets)
    tets[vol < 0] = tets[vol < 0][:, [0, 2, 1, 3]]
    tri = [facet_tris[int(f)] for f in outer]
    tri_refs = [np.full(len(t), 1) for t in tri]
    n_outer = sum(len(t) for t in tri)
    for f in cx.interface_facets():
        tri.append(facet_tris[int(f)])
        tri_refs.append(np.full(len(facet_tris[int(f)]), REF_GRAIN_BOUNDARY))
    return xyz, tets, refs, np.concatenate(tri), np.concatenate(tri_refs), n_outer


def _ridges(cx: CellComplex) -> np.ndarray:
    """The edges (vertex pairs) of the facets that MMG sees — the faces of the box, the
    boundary of whole cells and the facets between cells — where facets of two different
    planes meet: MMG keeps them as ridges, so that the surfaces stay the planar facets of
    the geometry (its detection by angles would miss the edges of small dihedral angles)."""
    seen = (cx.facet_box > 0) | (cx.facet_pieces[:, 1] < 0)
    seen[cx.interface_facets()] = True
    area, _ = cx.facet_vectors()
    normal = area / np.maximum(np.linalg.norm(area, axis=1, keepdims=True), 1e-300)
    edge_facets: dict = {}
    for f in np.flatnonzero(seen):
        loop = cx.facets[f]
        for a, b in zip(loop, np.roll(loop, -1), strict=True):
            edge_facets.setdefault((min(a, b), max(a, b)), []).append(f)
    out = []
    for edge, faces in edge_facets.items():
        n = normal[faces]
        if len(faces) == 1 or np.any(np.abs(n @ n[0]) < 1.0 - 1e-9):
            out.append(edge)
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def polycrystal_mesh(
    micro,
    *,
    size: float,
    min_size: float | None = None,
    hausdorff: float | None = None,
    gradation: float = 2.0,
    min_feature: float | None = None,
    phases: str = "all",
    order: int = 1,
    periodic: bool | None = None,
    cells: str = "whole",
    exact: bool = False,
    executable: str | None = None,
    workdir=None,
    timeout: float | None = None,
) -> TetMesh:
    """Conforming tetrahedral mesh of a Voronoi or Laguerre tessellation, one subdomain
    per cell: each element knows its cell (``mesh.cell``), its phase and, through its
    cell, its crystal orientation (``mesh.orientations``). The grains of seeds
    (:class:`~morphogen.models.seeded.SeededTessellation`) are the subdomains, the cells of
    their spheres merged (but for the exact facets of whole cells, meshed cell by cell):
    ``mesh.cell`` is then the grain.

    Parameters
    ----------
    size
        Largest element size (``-hmax``).
    min_size, hausdorff, gradation
        Smallest element size (default ``size / 10``), Hausdorff distance (default
        ``size / 20``; ``1e-6 size`` when ``exact``) and size gradation of MMG.
    min_feature
        Features of the tessellation smaller than this length (short edges, caps of
        cells on the faces of the box) are removed before meshing (default
        ``size / 4``, 0 when ``exact``; 0 keeps the exact cells, at the price of flat
        elements next to their small features).
    exact
        Keep the facets of the cells exactly: the edges where they bend given to MMG as
        ridges and their ends as corners, and its Hausdorff distance a millionth of the
        size by default (the facets are planar: nothing to approximate, while its
        smoothing moves the nodes of the surfaces within that distance). Not with a
        positive ``min_feature``: the cells are then regularized, their facets bent.
    phases
        ``"all"``, or ``"solid"``: the cells of phase 1 only.
    periodic
        Periodic mesh (default: when the tessellation is periodic).
    cells
        Periodic meshes: ``"whole"`` (default) meshes the whole cells, periodized at the
        design: their union fills the volume of the box and its boundary is made of
        pairs of translated facets with the same triangles, so that every node of the
        boundary has its translates (``mesh.periodic_links()``); ``"box"`` cuts the cells
        by the faces of the box, meshed alike on opposite faces.
    """
    from morphogen.models.tessellation import Tessellation

    if not isinstance(micro, Tessellation):
        raise TypeError("polycrystal_mesh: a Voronoi or Laguerre tessellation")
    if phases not in ("all", "solid"):
        raise ValueError("phases: 'all' or 'solid'")
    if order not in (1, 2):
        raise ValueError("order: 1 or 2")
    size = float(size)
    if not size > 0:
        raise ValueError("size: must be positive")
    domain: Domain = micro.domain
    periodic = domain.periodic if periodic is None else bool(periodic)
    if periodic and not domain.periodic:
        raise ValueError("matching periodic faces need a periodic tessellation")
    if cells not in ("whole", "box"):
        raise ValueError("cells: 'whole' or 'box'")
    whole = periodic and cells == "whole"
    hmin = float(min_size) if min_size is not None else size / 10.0
    if min_feature is not None:
        feature = float(min_feature)
    else:
        feature = 0.0 if exact else size / 4.0
    # the facets kept: those of the exact cells (planar: regularized cells have bent ones)
    exact = bool(exact) and feature == 0.0
    if hausdorff is not None:
        hausd = float(hausdorff)
    else:
        hausd = 1e-6 * size if exact else size / 20.0
    exe = mmg_executable(executable)
    t0 = time.perf_counter()
    cx = micro.cell_complex(whole=whole)
    grains = np.asarray(micro.cell_grains(), dtype=np.int64)
    # grains of several cells (seeds): merged, MMG free inside them; but for the exact facets
    # of whole cells, meshed cell by cell (MMG does not end with their merged grains)
    by_cell = not np.array_equal(grains, np.arange(len(grains))) and exact and whole
    if not by_cell and not np.array_equal(grains, np.arange(len(grains))):
        cx = cx.merged(grains)
    grain_phases = np.zeros(int(grains.max()) + 1 if len(grains) else 0, dtype=np.int64)
    grain_phases[grains] = micro.cell_phases
    if periodic != cx.periodic:
        raise ValueError("the cells of the tessellation and the mesh differ in periodicity")
    cx = regularize(cx, feature) if feature > 0 else cx
    t_cells = time.perf_counter() - t0
    tol = 1e-9 * float(domain.lengths.max())
    sizing = ["-hmax", f"{size:.10g}", "-hmin", f"{hmin:.10g}", "-hausd", f"{hausd:.10g}"]
    sizing += ["-hgrad", f"{float(gradation):.10g}", "-v", "-1"]
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        t1 = time.perf_counter()
        if whole:
            xyz, tets, refs, tri, tri_refs, n_box = whole_initial_mesh(cx, size=size)
        else:
            xyz, tets, refs, tri, tri_refs, n_box = initial_mesh(
                cx, size=size, sizing=sizing, periodic=periodic, folder=folder, timeout=timeout
            )
        t_mmg = time.perf_counter() - t1
        used = np.unique(tets)
        renum = np.full(len(xyz), -1, dtype=np.int64)
        renum[used] = np.arange(len(used))
        required_vertices = None
        if whole:  # the vertices of the triangles bounding the cells
            required_vertices = np.unique(renum[tri[:n_box]])
        elif periodic:
            on_face = np.zeros(len(used), dtype=bool)
            for d in range(3):
                on_face |= np.abs(xyz[used, d] - domain.lower[d]) <= tol
                on_face |= np.abs(xyz[used, d] - domain.upper[d]) <= tol
            required_vertices = np.flatnonzero(on_face)
        ridges = corners = None
        if exact:  # the bends of the facets as ridges, their ends as corners
            ridges = _ridges(cx)
            ridges = ridges[np.all(renum[ridges] >= 0, axis=1)]
            corners = renum[np.unique(ridges)]
            required_vertices = (
                corners if required_vertices is None else np.union1d(required_vertices, corners)
            )
            if not len(ridges):
                ridges = corners = None
        medit.write_mesh(
            folder / "cells.mesh",
            xyz[used],
            renum[tets],
            renum[tri],
            tet_refs=refs,
            tri_refs=tri_refs,
            required_triangles=np.arange(n_box) if periodic else None,
            required_vertices=required_vertices,
            edges=None if ridges is None else renum[ridges],
            ridges=None if ridges is None else np.arange(len(ridges)),
            corners=corners,
        )
        t1 = time.perf_counter()
        data = _run_mmg(exe, folder / "cells.mesh", None, folder / "out.mesh", sizing, timeout)
        t_mmg += time.perf_counter() - t1
    nodes = data["Vertices"]
    for d in range(3 if not whole else 0):  # nodes of the faces exactly on the faces
        nodes[np.abs(nodes[:, d] - domain.lower[d]) <= tol, d] = domain.lower[d]
        nodes[np.abs(nodes[:, d] - domain.upper[d]) <= tol, d] = domain.upper[d]
    elements = data["Tetrahedra"]
    cell = data["Tetrahedra_refs"] - 1  # the grain (the cell, but for the grains of seeds)
    if by_cell:
        cell = grains[cell]
    phase = grain_phases[cell]
    flip = tet_volumes(nodes, elements) < 0
    elements[flip] = elements[flip][:, [0, 2, 1, 3]]
    used, inverse = np.unique(elements, return_inverse=True)
    mesh = TetMesh(
        nodes=nodes[used],
        elements=inverse.reshape(elements.shape),
        phase=phase,
        lower=domain.lower,
        upper=domain.upper,
        periodic=periodic,
        cell=cell,
        orientations=micro.orientations,
        whole=whole,
        metadata={
            "mesher": "polycrystal (exact cells, mmg3d)",
            "cells": "whole" if whole else "box",
            "size": size,
            "min_feature": feature,
            "exact_facets": exact,
            "flattened_caps": int(cx.metadata.get("flattened_caps", 0)),
            "relabelled_caps": int(cx.metadata.get("relabelled_caps", 0)),
            "contracted_edges": int(cx.metadata.get("contracted_edges", 0)),
            "n_cells": int(micro.n_cells),
            "n_grains": len(grain_phases),
            "phases": phases,
            "matching_faces": periodic,
            "time_cells_s": round(t_cells, 3),
            "time_mmg_s": round(t_mmg, 3),
        },
    )
    if phases == "solid":
        mesh = mesh.subset([1])
    if order == 2:
        mesh = mesh.quadratic()
    log.info(
        "polycrystal mesh: %d elements, %d nodes (MMG %.1f s)", mesh.n_elements, mesh.n_nodes, t_mmg
    )
    return mesh
