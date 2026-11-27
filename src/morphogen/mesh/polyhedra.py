"""Exact cells of Voronoi and Laguerre (power) tessellations restricted to a box.

The cell of germ ``g_i`` (weight ``w_i``) is the convex polyhedron of the points whose
power distance ``|x - g_i|^2 - w_i`` is the smallest: the intersection of the
half-spaces ``2 (g_j - g_i) . x <= |g_j|^2 - |g_i|^2 - w_j + w_i`` bounded by the
bisector planes of its neighbours. Each cell is computed as a half-space intersection
(Qhull) with the planes of nearby germs, then checked: no other germ may be closer to one
of its vertices, otherwise the missing germs are added and the cell computed again.

Restricted to the box, a tessellation is a complex of convex *pieces*: one per cell, or
several for the cells of a periodic tessellation that cross the faces of the box (one
piece per periodic image). Pieces share their *facets*: bisector polygons between two
pieces, or polygons on the faces of the box. Vertices computed by neighbouring pieces are
merged, so that the facets are shared exactly; for periodic tessellations, the vertices
of opposite faces of the box are made translates of each other.

A periodic tessellation can also be taken as a set of *whole cells* (:func:`periodic_cells`):
each cell once, around its germ in the box, uncut. Their union fills the volume of the
box and tiles space by the translations of the box; its boundary is made of facets
between a cell and a translate of another one, which come in pairs of translates, so
that a mesh of the union naturally has matching nodes on them (periodization at the
design rather than at the meshing).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import HalfspaceIntersection, QhullError, cKDTree

#: Box faces: x0, x1, y0, y1, z0, z1 (references 1 to 6).
BOX_FACES = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))


@dataclass
class CellComplex:
    """Convex pieces of the cells of a tessellation restricted to a box.

    ``facets[f]`` is a loop of vertex indices, counter-clockwise around the normal that
    points from piece ``facet_pieces[f, 0]`` to piece ``facet_pieces[f, 1]`` (``-1``:
    outside the box, for the facets on its faces). ``facet_box[f]`` is 0 inside the box,
    1-6 on the faces x0, x1, y0, y1, z0, z1. ``piece_cell[p]`` is the cell (germ index)
    of piece ``p``, ``piece_facets[p]`` its facets.

    ``whole``: the pieces are the whole cells of a periodic tessellation
    (:func:`periodic_cells`); the facets with no second piece then bound their union, and
    ``facet_shift[f]`` is the translation (in box lengths) of the cell beyond them.
    """

    vertices: np.ndarray
    facets: list[np.ndarray]
    facet_pieces: np.ndarray
    facet_box: np.ndarray
    piece_cell: np.ndarray
    piece_facets: list[np.ndarray]
    lower: np.ndarray
    upper: np.ndarray
    periodic: bool = False
    metadata: dict = field(default_factory=dict)
    whole: bool = False
    facet_shift: np.ndarray | None = None

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_facets(self) -> int:
        return len(self.facets)

    @property
    def n_pieces(self) -> int:
        return len(self.piece_cell)

    def facet_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Area vectors (normal times area, oriented as the loops) and centroids."""
        area = np.zeros((self.n_facets, 3))
        centre = np.zeros((self.n_facets, 3))
        for f, loop in enumerate(self.facets):
            p = self.vertices[loop]
            c = p.mean(axis=0)
            q = np.roll(p, -1, axis=0)
            cross = np.cross(p - c, q - c)
            a = 0.5 * cross.sum(axis=0)
            tri_area = 0.5 * np.linalg.norm(cross, axis=1)
            total = tri_area.sum()
            area[f] = a
            # area-weighted centroid of the fan triangles
            centre[f] = (
                ((p + q + c) / 3.0 * tri_area[:, None]).sum(axis=0) / total if total > 0 else c
            )
        return area, centre

    def piece_volumes(self) -> np.ndarray:
        """Volumes of the pieces (divergence theorem over their facets)."""
        area, centre = self.facet_vectors()
        flux = np.einsum("ij,ij->i", area, centre) / 3.0
        vol = np.zeros(self.n_pieces)
        np.add.at(vol, self.facet_pieces[:, 0], flux)
        inner = self.facet_pieces[:, 1] >= 0
        np.add.at(vol, self.facet_pieces[inner, 1], -flux[inner])
        return vol

    def cell_volumes(self, n_cells: int | None = None) -> np.ndarray:
        """Volume of each cell inside the box (summed over its pieces)."""
        n = int(self.piece_cell.max()) + 1 if n_cells is None else int(n_cells)
        out = np.zeros(n)
        np.add.at(out, self.piece_cell, self.piece_volumes())
        return out

    def grain_boundary_area(self) -> float:
        """Area of the facets between different cells."""
        area, _ = self.facet_vectors()
        a, b = self.facet_pieces[:, 0], self.facet_pieces[:, 1]
        inner = (b >= 0) & (self.piece_cell[a] != self.piece_cell[np.maximum(b, 0)])
        return float(np.linalg.norm(area[inner], axis=1).sum())

    def interface_facets(self) -> np.ndarray:
        """Facets between pieces of different cells (grain boundaries)."""
        a, b = self.facet_pieces[:, 0], self.facet_pieces[:, 1]
        return np.flatnonzero((b >= 0) & (self.piece_cell[a] != self.piece_cell[np.maximum(b, 0)]))

    def merged(self, groups) -> CellComplex:
        """The complex whose cells are groups of its cells (``groups[c]``, the group of the
        cell ``c``): the same pieces and facets, those between cells of a group no longer
        between cells (the grains of seeds, unions of the cells of their spheres)."""
        return replace(self, piece_cell=np.asarray(groups, dtype=np.int64)[self.piece_cell])

    def edges(self) -> np.ndarray:
        """Unique edges (sorted vertex pairs) of the facets."""
        pairs = [np.stack([loop, np.roll(loop, -1)], axis=1) for loop in self.facets]
        e = np.sort(np.concatenate(pairs), axis=1)
        return np.unique(e, axis=0)

    def summary(self) -> dict:
        e = self.edges()
        lengths = np.linalg.norm(self.vertices[e[:, 0]] - self.vertices[e[:, 1]], axis=1)
        return {
            "n_cells": int(np.unique(self.piece_cell).size),
            "n_pieces": self.n_pieces,
            "n_facets": self.n_facets,
            "n_vertices": self.n_vertices,
            "shortest_edge": float(lengths.min()) if lengths.size else float("nan"),
        }


# -- geometry of one piece -------------------------------------------------------------------


def _chebyshev_centre(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and radius of the largest ball inside ``a x <= b``."""
    norms = np.linalg.norm(a, axis=1)
    res = linprog(
        c=[0.0, 0.0, 0.0, -1.0],
        A_ub=np.column_stack([a, norms]),
        b_ub=b,
        bounds=[(None, None)] * 3 + [(0.0, None)],
        method="highs",
    )
    if res.status != 0:
        return np.zeros(3), 0.0
    return res.x[:3], float(res.x[3])


def _loop_on_plane(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Order of planar points counter-clockwise around ``normal``."""
    c = points.mean(axis=0)
    u = points[0] - c
    if np.linalg.norm(u) < 1e-300:
        u = np.array([1.0, 0.0, 0.0])
    n = normal / np.linalg.norm(normal)
    u = u - n * (u @ n)
    if np.linalg.norm(u) < 1e-300:
        u = np.cross(n, [1.0, 0.0, 0.0])
        if np.linalg.norm(u) < 1e-12:
            u = np.cross(n, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    d = points - c
    return np.argsort(np.arctan2(d @ v, d @ u), kind="stable")


def _piece(a: np.ndarray, b: np.ndarray, interior: np.ndarray, tol: float):
    """Vertices of ``a x <= b`` and, for each plane, its facet (vertex indices ordered
    counter-clockwise around the outward normal ``a``), or ``None`` if empty."""
    try:
        hs = HalfspaceIntersection(np.column_stack([a, -b]), interior)
    except QhullError:
        try:
            hs = HalfspaceIntersection(np.column_stack([a, -b]), interior, qhull_options="QJ")
        except QhullError:
            return None
    pts = hs.intersections
    # merge duplicates (degenerate vertices are produced once per simplicial facet)
    labels = _merge_points(pts, tol)
    n_v = int(labels.max()) + 1
    sums = np.zeros((n_v, 3))
    np.add.at(sums, labels, pts)
    verts = sums / np.bincount(labels, minlength=n_v)[:, None]
    norms = np.linalg.norm(a, axis=1)
    dist = (verts @ a.T - b) / norms  # (n_vertices, n_planes), <= 0 inside
    facets = {}
    for k in range(len(b)):
        on = np.flatnonzero(np.abs(dist[:, k]) <= 4.0 * tol)
        if on.size >= 3:
            loop = on[_loop_on_plane(verts[on], a[k])]
            facets[k] = loop
    if len(facets) < 4:
        return None
    return verts, facets


# -- the complex -----------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def _merge_points(points: np.ndarray, tol: float) -> np.ndarray:
    """Cluster index of each point (points closer than ``tol`` are merged)."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    pairs = cKDTree(points).query_pairs(tol, output_type="ndarray")
    n = len(points)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def power_complex(
    germs,
    weights=None,
    lower=(0.0, 0.0, 0.0),
    upper=(1.0, 1.0, 1.0),
    periodic: bool = False,
    *,
    neighbours: int = 30,
) -> CellComplex:
    """Exact cells of the power diagram of ``germs`` (Voronoi without ``weights``)
    restricted to the box ``[lower, upper]``, periodic or not."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    size = upper - lower
    g = np.asarray(germs, dtype=float).reshape(-1, 3) - lower
    n = len(g)
    if n == 0:
        raise ValueError("power_complex: no germs")
    w = np.zeros(n) if weights is None else np.asarray(weights, dtype=float).reshape(n)
    scale = float(size.max())
    tol = 1e-9 * scale
    if periodic:
        g = np.mod(g, size)
        shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
        pts = (g[None, :, :] + shifts[:, None, :] * size).reshape(-1, 3)
        ids = np.tile(np.arange(n), len(shifts))
        wts = np.tile(w, len(shifts))
    else:
        pts, ids, wts = g, np.arange(n), w
    wmax = float(wts.max())
    lift = np.sqrt(np.maximum(wmax - wts, 0.0))
    tree = cKDTree(np.column_stack([pts, lift]))
    # largest power distance from a point of the box to its cell: bounds the germs whose
    # cells reach the box (sampled on a grid, plus the sampling error)
    m = max(4, min(40, round(2.0 * n ** (1.0 / 3.0)) + 4))
    axes = [np.linspace(0.0, s, m) for s in size]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    d, _ = tree.query(np.column_stack([grid, np.zeros(len(grid))]))
    power_max = float(np.max(d**2)) - wmax
    spacing = float(np.linalg.norm(size / (m - 1)))
    reach = math.sqrt(max(power_max + wmax, 0.0)) + spacing
    outside = np.maximum(np.maximum(-pts, pts - size), 0.0)
    candidates = np.flatnonzero(np.linalg.norm(outside, axis=1) <= reach + tol)
    box_a = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], float)
    box_b = np.array([0.0, size[0], 0.0, size[1], 0.0, size[2]])
    k = min(neighbours, len(pts) - 1)
    raw = []  # (candidate, vertices, {plane key: loop})
    for c in candidates:
        pc, wc = pts[c], wts[c]
        _, near = tree.query(np.concatenate([pc, [lift[c]]]), k=k + 1)
        near = [j for j in np.atleast_1d(near) if j != c and j < len(pts)]
        while True:
            nb = np.asarray(near, dtype=np.int64)
            a = np.vstack([2.0 * (pts[nb] - pc), box_a]) if nb.size else box_a.copy()
            b = np.concatenate(
                [(pts[nb] ** 2).sum(axis=1) - pc @ pc - wts[nb] + wc, box_b] if nb.size else [box_b]
            )
            inside = (
                np.all(pc > tol)
                and np.all(pc < size - tol)
                and (nb.size == 0 or np.all(a[: nb.size] @ pc < b[: nb.size] - tol * scale))
            )
            if inside:
                centre, radius = pc, 1.0
            else:
                centre, radius = _chebyshev_centre(a, b)
            if radius <= 10.0 * tol:
                piece = None
                break
            piece = _piece(a, b, centre, tol)
            if piece is None:
                break
            verts, _ = piece
            # a vertex closer (in power distance) to another germ: that germ was missing
            dist, owner = tree.query(np.column_stack([verts, np.zeros(len(verts))]))
            own = ((verts - pc) ** 2).sum(axis=1) - wc + wmax
            missing = set(
                int(o)
                for o, dd, oo in zip(owner, dist**2, own, strict=True)
                if dd < oo - 1e-9 * scale**2 and o != c
            )
            missing -= set(near)
            if not missing:
                break
            near += sorted(missing)
        if piece is None:
            continue
        verts, facets = piece
        keyed = {}
        for plane, loop in facets.items():
            key = ("n", int(nb[plane])) if plane < nb.size else ("b", plane - nb.size + 1)
            keyed[key] = loop
        raw.append((int(c), verts, keyed))
    return _assemble(raw, ids, size, lower, periodic, tol)


def periodic_cells(
    germs,
    weights=None,
    lower=(0.0, 0.0, 0.0),
    upper=(1.0, 1.0, 1.0),
    *,
    neighbours: int = 30,
) -> CellComplex:
    """Whole cells of the periodic power diagram of ``germs`` (Voronoi without
    ``weights``) of period ``upper - lower``: one piece per non-empty cell, around its germ
    brought into the box (see the module). Facets between two cells of the set are shared;
    the others bound the union and come in pairs of translates."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    size = upper - lower
    g = np.mod(np.asarray(germs, dtype=float).reshape(-1, 3) - lower, size)
    g[g >= size] = 0.0
    n = len(g)
    if n == 0:
        raise ValueError("periodic_cells: no germs")
    w = np.zeros(n) if weights is None else np.asarray(weights, dtype=float).reshape(n)
    scale = float(size.max())
    tol = 1e-9 * scale
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)
    pts = (g[None, :, :] + shifts[:, None, :] * size).reshape(-1, 3)
    ids = np.tile(np.arange(n), len(shifts))
    image = np.repeat(shifts, n, axis=0)
    wts = np.tile(w, len(shifts))
    zero = int(np.flatnonzero(np.all(shifts == 0, axis=1))[0])
    wmax = float(wts.max())
    lift = np.sqrt(np.maximum(wmax - wts, 0.0))
    tree = cKDTree(np.column_stack([pts, lift]))
    box_a = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], float)
    voronoi = bool(np.all(w == w[0]))
    raw = []
    for c in range(n):
        own_index = zero * n + c
        pc, wc = pts[own_index], wts[own_index]
        # a box around the germ keeps the first intersections bounded; it must not bound
        # the cell in the end
        box_b = np.concatenate(
            [[-(pc[d] - 2.0 * size[d]), pc[d] + 2.0 * size[d]] for d in range(3)]
        )
        k = min(neighbours, len(pts) - 1)
        _, near = tree.query(np.concatenate([pc, [lift[own_index]]]), k=k + 1)
        near = [int(j) for j in np.atleast_1d(near) if j != own_index and j < len(pts)]
        piece = None
        for _ in range(20):
            nb = np.asarray(near, dtype=np.int64)
            a = np.vstack([2.0 * (pts[nb] - pc), box_a])
            b = np.concatenate([(pts[nb] ** 2).sum(axis=1) - pc @ pc - wts[nb] + wc, box_b])
            centre, radius = (pc, 1.0) if voronoi else _chebyshev_centre(a, b)
            if radius <= 10.0 * tol:
                piece = None
                break
            piece = _piece(a, b, centre, tol)
            if piece is None:
                break
            verts, facets = piece
            dist, owner = tree.query(np.column_stack([verts, np.zeros(len(verts))]))
            own = ((verts - pc) ** 2).sum(axis=1) - wc + wmax
            missing = {
                int(o)
                for o, dd, oo in zip(owner, dist**2, own, strict=True)
                if dd < oo - 1e-9 * scale**2 and o != own_index
            } - set(near)
            bounded = all(plane < nb.size for plane in facets)
            if not missing and bounded:
                break
            if missing:
                near += sorted(missing)
            else:  # the box bounds the cell: more neighbours
                k = min(2 * k, len(pts) - 1)
                _, more = tree.query(np.concatenate([pc, [lift[own_index]]]), k=k + 1)
                near = [int(j) for j in np.atleast_1d(more) if j != own_index and j < len(pts)]
        else:
            raise RuntimeError("periodic_cells: a cell could not be closed by its neighbours")
        if piece is None:
            continue  # an empty Laguerre cell
        verts, facets = piece
        keyed = {int(nb[plane]): loop for plane, loop in facets.items() if plane < nb.size}
        raw.append((c, verts, keyed))
    return _assemble_whole(raw, ids, image, size, lower, tol)


def _merge_periodic(points: np.ndarray, size: np.ndarray, tol: float) -> np.ndarray:
    """Cluster index of points on the torus of period ``size`` (closer than ``tol``)."""
    pairs = cKDTree(points, boxsize=size).query_pairs(tol, output_type="ndarray")
    n = len(points)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def torus_classes(points: np.ndarray, lower, size, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Points that are translates of each other by the periods ``size``: the class of
    each point and its translation (in periods) from the first point of its class."""
    p = np.asarray(points, dtype=float) - np.asarray(lower, dtype=float)
    size = np.asarray(size, dtype=float)
    wrapped = np.mod(p, size)
    wrapped[wrapped >= size] = 0.0
    labels = _merge_periodic(wrapped, size, tol)
    first = np.full(int(labels.max()) + 1 if len(labels) else 0, -1)
    first[labels[::-1]] = np.arange(len(labels))[::-1]
    shift = np.rint((p - p[first[labels]]) / size).astype(np.int64)
    return labels, shift


def _assemble_whole(raw, ids, image, size, lower, tol) -> CellComplex:
    """Vertices and facets of whole cells: the vertices are translates of torus points by
    whole periods, so that the facets bounding the union are exact translates."""
    all_pts = np.concatenate([v for _, v, _ in raw])
    offsets = np.cumsum([0] + [len(v) for _, v, _ in raw])
    labels, shift = torus_classes(all_pts, np.zeros(3), size, 50.0 * tol)
    n_lab = int(labels.max()) + 1
    # one position per torus point (the mean of its copies brought back by their shifts)
    base = all_pts - shift * size
    sums = np.zeros((n_lab, 3))
    np.add.at(sums, labels, base)
    canonical = sums / np.bincount(labels, minlength=n_lab)[:, None]
    key_index: dict = {}
    vertex_of = np.empty(len(all_pts), dtype=np.int64)
    for i, (lab, s) in enumerate(zip(labels.tolist(), map(tuple, shift.tolist()), strict=True)):
        vertex_of[i] = key_index.setdefault((lab, s), len(key_index))
    verts = np.empty((len(key_index), 3))
    for (lab, s), v in key_index.items():
        verts[v] = canonical[lab] + np.asarray(s) * size
    piece_of = {c: p for p, (c, _, _) in enumerate(raw)}
    facet_index: dict = {}
    facets: list[np.ndarray] = []
    facet_pieces: list[list[int]] = []
    facet_shift: list[tuple] = []
    piece_facets: list[list[int]] = []
    for p, (c, _, keyed) in enumerate(raw):
        mine = []
        for j, loop in keyed.items():
            gl = vertex_of[offsets[p] + loop]
            gl = gl[np.concatenate([[True], gl[1:] != gl[:-1]])]
            if gl.size > 1 and gl[0] == gl[-1]:
                gl = gl[:-1]
            if np.unique(gl).size < 3:
                continue
            other, s = int(ids[j]), tuple(int(v) for v in image[j])
            if s == (0, 0, 0):
                key = ("in", min(c, other), max(c, other))
                if key in facet_index:
                    f = facet_index[key]
                    facet_pieces[f][1] = p
                    merged = np.union1d(facets[f], gl)
                    if merged.size != np.unique(facets[f]).size:
                        facets[f] = _reloop(verts, merged, facets[f])
                    mine.append(f)
                    continue
                facet_pieces.append([p, -2 if other in piece_of else -3])
            else:
                key = ("out", c, other, s)
                facet_pieces.append([p, -1])
            f = len(facets)
            facet_index[key] = f
            facets.append(gl)
            facet_shift.append(s)
            mine.append(f)
        piece_facets.append(mine)
    facet_pieces = np.array(facet_pieces, dtype=np.int64)
    # boundary facets without their translate, interior facets with one side: slivers
    # of degenerate vertices (dropped if their area is below the tolerance)
    drop = set(np.flatnonzero(facet_pieces[:, 1] < -1).tolist())
    for key, f in facet_index.items():
        if (
            key[0] == "out"
            and ("out", key[2], key[1], tuple(-v for v in key[3])) not in facet_index
        ):
            drop.add(f)
    scale = float(size.max())
    if any(_loop_area(verts[facets[f]]) > 1e-6 * scale**2 for f in drop):
        raise RuntimeError(
            "periodic_cells: a facet has no counterpart; the tessellation is degenerate"
        )
    keep = np.array([f not in drop for f in range(len(facets))])
    remap = np.cumsum(keep) - 1
    cx = CellComplex(
        vertices=verts + lower,
        facets=[f for f, k in zip(facets, keep, strict=True) if k],
        facet_pieces=facet_pieces[keep],
        facet_box=np.zeros(int(keep.sum()), dtype=np.int64),
        piece_cell=np.array([ids[c] for c, _, _ in raw], dtype=np.int64),
        piece_facets=[
            np.array([remap[f] for f in pf if keep[f]], dtype=np.int64) for pf in piece_facets
        ],
        lower=lower,
        upper=lower + size,
        periodic=True,
        whole=True,
        facet_shift=np.array(
            [s for s, k in zip(facet_shift, keep, strict=True) if k], dtype=np.int64
        ).reshape(-1, 3),
    )
    _orient_and_close(cx)
    return cx


def _assemble(raw, ids, size, lower, periodic, tol) -> CellComplex:
    """Global vertices and shared facets of the pieces."""
    scale = float(size.max())
    # global vertices: merge the copies computed by neighbouring pieces
    all_pts = np.concatenate([v for _, v, _ in raw])
    offsets = np.cumsum([0] + [len(v) for _, v, _ in raw])
    labels = _merge_points(all_pts, 50.0 * tol)
    n_v = int(labels.max()) + 1
    sums = np.zeros((n_v, 3))
    np.add.at(sums, labels, all_pts)
    verts = sums / np.bincount(labels, minlength=n_v)[:, None]
    for d in range(3):  # exactly on the faces of the box
        verts[np.abs(verts[:, d]) <= 100.0 * tol, d] = 0.0
        verts[np.abs(verts[:, d] - size[d]) <= 100.0 * tol, d] = size[d]
    if periodic:
        _match_periodic_faces(verts, size, 1e3 * tol)
    piece_of = {c: p for p, (c, _, _) in enumerate(raw)}
    facet_index: dict = {}
    facets: list[np.ndarray] = []
    facet_pieces: list[list[int]] = []
    facet_box: list[int] = []
    piece_facets: list[list[int]] = []
    for p, (c, _, keyed) in enumerate(raw):
        mine = []
        for key, loop in keyed.items():
            gl = labels[offsets[p] + loop]
            gl = gl[np.concatenate([[True], gl[1:] != gl[:-1]])]
            if gl.size > 1 and gl[0] == gl[-1]:
                gl = gl[:-1]
            if np.unique(gl).size < 3:
                continue
            if key[0] == "b":
                f = len(facets)
                facets.append(gl)
                facet_pieces.append([p, -1])
                facet_box.append(key[1])
                mine.append(f)
                continue
            other = key[1]
            pair = (min(c, other), max(c, other))
            if pair in facet_index:
                f = facet_index[pair]
                facet_pieces[f][1] = p
                # the loop seen from the second piece: union of the two vertex sets
                merged = np.union1d(facets[f], gl)
                if merged.size != np.unique(facets[f]).size:
                    facets[f] = _reloop(verts, merged, facets[f])
            else:
                f = len(facets)
                facet_index[pair] = f
                facets.append(gl)
                facet_pieces.append([p, -2 if other in piece_of else -3])
                facet_box.append(0)
            mine.append(f)
        piece_facets.append(mine)
    facet_pieces = np.array(facet_pieces, dtype=np.int64)
    unmatched = np.flatnonzero(facet_pieces[:, 1] < -1)
    if unmatched.size:
        # facets whose second piece was not computed: tiny sliver facets of a vertex
        # shared by several cells, dropped (their area is below the tolerance)
        area = np.array([_loop_area(verts[facets[f]]) for f in unmatched])
        if np.any(area > 1e-6 * scale**2):
            raise RuntimeError(
                "power_complex: a facet has a single piece; the tessellation is degenerate"
            )
        drop = set(unmatched.tolist())
        keep = np.array([f not in drop for f in range(len(facets))])
        remap = np.cumsum(keep) - 1
        facets = [f for f, k in zip(facets, keep, strict=True) if k]
        facet_pieces = facet_pieces[keep]
        facet_box = [b for b, k in zip(facet_box, keep, strict=True) if k]
        piece_facets = [[int(remap[f]) for f in pf if keep[f]] for pf in piece_facets]
    complex_ = CellComplex(
        vertices=verts + lower,
        facets=facets,
        facet_pieces=facet_pieces,
        facet_box=np.array(facet_box, dtype=np.int64),
        piece_cell=np.array([ids[c] for c, _, _ in raw], dtype=np.int64),
        piece_facets=[np.array(pf, dtype=np.int64) for pf in piece_facets],
        lower=lower,
        upper=lower + size,
        periodic=periodic,
    )
    _orient_and_close(complex_)
    return complex_


def _loop_area(p: np.ndarray) -> float:
    c = p.mean(axis=0)
    return 0.5 * float(np.linalg.norm(np.cross(p - c, np.roll(p, -1, axis=0) - c).sum(axis=0)))


def _reloop(verts, members, reference) -> np.ndarray:
    """Loop of ``members`` (coplanar) oriented as the loop ``reference``."""
    p = verts[reference]
    c = p.mean(axis=0)
    normal = np.cross(p - c, np.roll(p, -1, axis=0) - c).sum(axis=0)
    return members[_loop_on_plane(verts[members], normal)]


def _match_periodic_faces(verts: np.ndarray, size: np.ndarray, tol: float) -> None:
    """Make the vertices of opposite faces of the box exact translates of each other."""
    for axis in range(3):
        others = [d for d in range(3) if d != axis]
        lo = np.flatnonzero(verts[:, axis] == 0.0)
        hi = np.flatnonzero(verts[:, axis] == size[axis])
        if lo.size == 0 or hi.size == 0:
            continue
        dist, near = cKDTree(verts[lo][:, others]).query(verts[hi][:, others])
        ok = dist <= tol
        verts[hi[ok][:, None], others] = verts[lo[near[ok]]][:, others]


def _orient_and_close(cx: CellComplex) -> None:
    """Orient every facet from its first piece outwards, and check that every piece is a
    closed surface (each edge shared by two of its facets, in opposite directions)."""
    centres = np.zeros((cx.n_pieces, 3))
    for p, pf in enumerate(cx.piece_facets):
        centres[p] = np.concatenate([cx.vertices[cx.facets[f]] for f in pf]).mean(axis=0)
    for f, loop in enumerate(cx.facets):
        p = cx.vertices[loop]
        c = p.mean(axis=0)
        normal = np.cross(p - c, np.roll(p, -1, axis=0) - c).sum(axis=0)
        if normal @ (c - centres[cx.facet_pieces[f, 0]]) < 0:
            cx.facets[f] = loop[::-1].copy()
    bad = 0
    for p, pf in enumerate(cx.piece_facets):
        edges = {}
        for f in pf:
            loop = cx.facets[f] if cx.facet_pieces[f, 0] == p else cx.facets[f][::-1]
            for a, b in zip(loop, np.roll(loop, -1), strict=True):
                edges[(int(a), int(b))] = edges.get((int(a), int(b)), 0) + 1
        for (a, b), count in edges.items():
            if count != 1 or edges.get((b, a), 0) != 1:
                bad += 1
                break
    cx.metadata["open_pieces"] = bad
