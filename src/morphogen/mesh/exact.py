"""Conforming meshes of the exact geometry of microstructures, without a background grid.

The level-set route of :func:`morphogen.mesh.conforming.conforming_mesh` samples the level
set of a microstructure at the nodes of a regular grid: the interface is as accurate as the
grid is fine, and features thinner than its spacing are lost. The meshes here are built on
the geometry itself.

**Faceted microstructures** (:func:`polyhedral_mesh`): microstructures made of convex
polyhedra that partition their box — the plates and voids of scale-rich networks, the
extruded ligaments of planar ones, the cubes of fractal sponges, the slabs of laminates —
give their pieces (``convex_pieces()``). Their faces need not match: a plate is bounded on
one side by a single face, on the other by the faces of every void and plate that ends on
it. The faces are grouped by plane, and the pieces facing each other across a plane share
the overlay of their faces (each face clipped by those of the other side, Sutherland-Hodgman
on the plane), so that every facet is shared by exactly two pieces (or lies on a face of the
box); the vertices lying on the edges of a facet (T-junctions) are inserted into its loop.
This face-to-face complex is split into tetrahedra (cones from the centroids of the pieces
over their fanned facets, :func:`morphogen.mesh.polycrystal.initial_mesh`), and MMG
remeshes it to the requested size keeping the facets between phases and the faces of the
box (their edges as ridges, within a Hausdorff distance of a millionth of the size: planar
facets have nothing to approximate, and its smoothing would bend them): the interfaces are
those of the exact geometry, sharp edges and corners included, at any thickness.

**Implicit microstructures** (:func:`adaptive_mesh`): the others give their level set at
any point of space (``level_set_at``: the signed distance to their boundary, or a function
with the same zero set), and the mesh is refined where the interface needs it, without a
grid. An octree of the box (:mod:`morphogen.mesh.octree`) splits its cells along whose
lines of three points ``a``, ``m``, ``b`` (edges, middle lines of the faces, axes) the
level set crosses zero twice (a feature thinner than the cell), crosses it once at a point
that its linear interpolation misplaces by more than the Hausdorff distance (the deviation
``e = phi_m - (phi_a + phi_b) / 2`` of the parabola through the three values, over the
slope), or bends toward zero by more than that distance while the interface may be near
(``|phi_a| + |phi_m|`` or ``|phi_m| + |phi_b|`` below ``L l / 2``, ``L`` the slope of the
level set: 1 for a distance, ``l`` the length of the line): a feature hidden between the
samples; or where the level set, at the zero of its linear interpolation along a segment
of the cell that may be an edge of its tetrahedra (:data:`morphogen.mesh.octree.SEGMENTS`),
exceeds the Hausdorff distance (a kink between the ends, the medial surface of a sheet,
that the parabola misses). A distance bends away from zero on the medial surfaces of the
phases (the middle of a sheet), where nothing else is refined. The octree is balanced and
split into tetrahedra (the six of the Kuhn triangulation in its plain cells, cones from the
centres of the others over their fanned faces), then MMG discretizes the zero level set
(``mmg3d -ls``) and remeshes the result within the Hausdorff distance, without detecting
ridges by angles (the facets of the cut bend at the scale of the octree: their ridges
would keep flat elements; the edges of the box are given); the nodes of the interface are
finally moved onto the exact zero level set (Newton's method along the gradient), as near
as the elements allow without flipping or flattening. Periodic meshes:
the octree is balanced across the periodic faces, so that opposite faces get the same
triangles; their edges that the interface crosses are split at its exact zeros (or their
ends moved onto them), alike on translated faces, and MMG keeps these faces as they are:
they match, and their nodes lie on the interface.
"""

from __future__ import annotations

import itertools
import logging
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from morphogen.io import medit
from morphogen.mesh.conforming import (
    REF_INSIDE,
    REF_OUTSIDE,
    _matching_refs,
    _on_faces,
    _periodic_axes,
    _run_mmg,
    mmg_executable,
)
from morphogen.mesh.polyhedra import CellComplex, _merge_points
from morphogen.mesh.tetra import TetMesh, tet_quality, tet_volumes

log = logging.getLogger("morphogen")

#: The values of the level sets within this fraction of the Hausdorff distance of zero, at
#: the nodes of the octree, are moved this far off it before the cut (no interface
#: through a node)
_OFF_NODES = 1e-3

#: Relative tolerance of the geometry (times the largest side of the box): points closer
#: are one, faces on planes closer are coplanar
TOLERANCE = 1e-9


# -- convex polygons of a plane of space -------------------------------------------------------


@njit(cache=True)
def _clip_coplanar(a, b, nb, tol):
    """The part of the convex polygon ``a`` inside the convex polygon ``b`` of the same
    plane (``b`` counter-clockwise about ``nb``): Sutherland-Hodgman against the half-planes
    of the edges of ``b``, the points within ``tol`` of an edge kept inside."""
    out = a.copy()
    m = b.shape[0]
    for j in range(m):
        k = out.shape[0]
        if k == 0:
            break
        q0 = b[j]
        q1 = b[(j + 1) % m]
        ex, ey, ez = q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]
        ix = nb[1] * ez - nb[2] * ey  # nb x e: inward for a counter-clockwise polygon
        iy = nb[2] * ex - nb[0] * ez
        iz = nb[0] * ey - nb[1] * ex
        s = np.sqrt(ix * ix + iy * iy + iz * iz)
        if s == 0.0:
            continue
        ix, iy, iz = ix / s, iy / s, iz / s
        new = np.empty((k + m + 2, 3))
        n = 0
        for i in range(k):
            p0 = out[i]
            p1 = out[(i + 1) % k]
            d0 = (p0[0] - q0[0]) * ix + (p0[1] - q0[1]) * iy + (p0[2] - q0[2]) * iz
            d1 = (p1[0] - q0[0]) * ix + (p1[1] - q0[1]) * iy + (p1[2] - q0[2]) * iz
            if d0 >= -tol:
                new[n] = p0
                n += 1
                if d1 < -tol:
                    t = d0 / (d0 - d1)
                    new[n] = p0 + t * (p1 - p0)
                    n += 1
            elif d1 >= -tol:
                t = d0 / (d0 - d1)
                new[n] = p0 + t * (p1 - p0)
                n += 1
        out = new[:n]
    # consecutive points that coincide
    k = out.shape[0]
    keep = np.ones(k, dtype=np.bool_)
    for i in range(k):
        p, q = out[i], out[(i + 1) % k]
        if abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol and abs(p[2] - q[2]) <= tol:
            keep[i] = False
    return out[keep]


def _area_vector(loop: np.ndarray) -> np.ndarray:
    """Half the sum of the cross products of a planar loop: its area times its normal."""
    p = np.asarray(loop, dtype=float)
    c = p.mean(axis=0)
    return 0.5 * np.cross(p - c, np.roll(p, -1, axis=0) - c).sum(axis=0)


# -- the face-to-face complex of convex pieces -----------------------------------------------


def _faces_of(pieces) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """The faces of the pieces (``(vertices, face offsets, labels, normals, planes)``):
    their loops (counter-clockwise about their outward normals), pieces, outward unit
    normals and offsets (the planes given: exact; faces of no area dropped)."""
    loops, owner, normals, offsets = [], [], [], []
    for p, (verts, fofs, _labels, face_normals, face_planes) in enumerate(pieces):
        for f in range(len(fofs) - 1):
            loop = np.asarray(verts[fofs[f] : fofs[f + 1]], dtype=float)
            if len(loop) < 3:
                continue
            area = _area_vector(loop)
            if not np.linalg.norm(area) > 0.0:
                continue
            n = np.asarray(face_normals[f], dtype=float)
            if area @ n < 0:  # the loop turns the other way
                loop = loop[::-1].copy()
            loops.append(loop)
            owner.append(p)
            normals.append(n)
            offsets.append(float(face_planes[f]))
    return loops, np.asarray(owner, dtype=np.int64), np.asarray(normals), np.asarray(offsets)


def _planes(normals: np.ndarray, offsets: np.ndarray, scale: float, tol: float):
    """The plane of each face (cluster index) and the side of the face on it (+1 or -1):
    the normals turned so that their largest component is positive, clustered with the
    offsets (relative to ``scale``) within ``tol``."""
    k = np.argmax(np.abs(normals), axis=1)
    side = np.sign(normals[np.arange(len(normals)), k])
    side[side == 0] = 1.0
    key = np.column_stack([normals * side[:, None], (offsets * side / scale)[:, None]])
    pairs = cKDTree(key).query_pairs(tol, output_type="ndarray")
    n = len(key)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, cluster = connected_components(graph, directed=False)
    return cluster, side.astype(np.int64), key


def _box_face(key: np.ndarray, lower, upper, scale: float, tol: float) -> int:
    """The face of the box (1-6: x0, x1, y0, y1, z0, z1) of a plane (its canonical normal
    and offset over ``scale``), or 0."""
    for d in range(3):
        e = np.zeros(3)
        e[d] = 1.0
        if np.linalg.norm(key[:3] - e) > 1e3 * tol:
            continue
        for s, bound in ((0, lower[d]), (1, upper[d])):
            if abs(key[3] * scale - bound) <= 1e3 * tol * scale:
                return 1 + 2 * d + s
    return 0


def face_to_face(pieces, cells, lower, upper, *, tol: float | None = None) -> CellComplex:
    """The face-to-face complex of convex polyhedra partitioning the box ``[lower, upper]``
    (``(vertices, face offsets, labels, normals, planes)`` tuples, the faces counter-
    clockwise about their outward normals), ``cells[p]`` the cell of piece ``p``: the
    overlay of the faces facing each other on every plane, the vertices on the edges of
    the facets inserted into their loops (see the module)."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    scale = float(np.max(upper - lower))
    tol = TOLERANCE * scale if tol is None else float(tol)
    loops, owner, normals, offsets = _faces_of(pieces)
    cluster, side, key = _planes(normals, offsets, scale, TOLERANCE * 10)
    facets, facet_pieces, facet_box = [], [], []
    order = np.argsort(cluster, kind="stable")
    bounds = np.flatnonzero(np.diff(cluster[order])) + 1
    for group in np.split(order, bounds):
        a = group[side[group] > 0]
        b = group[side[group] < 0]
        if len(a) == 0 or len(b) == 0:  # a face of the box
            box = _box_face(key[group[0]], lower, upper, scale, TOLERANCE * 10)
            if box == 0:
                raise ValueError("face_to_face: a face of a piece is neither shared nor on "
                                 "the box: the pieces do not partition the box")  # fmt: skip
            for f in group:
                facets.append(loops[f])
                facet_pieces.append((owner[f], -1))
                facet_box.append(box)
            continue
        lo_b = np.array([loops[f].min(axis=0) for f in b]) - tol
        hi_b = np.array([loops[f].max(axis=0) for f in b]) + tol
        for fa in a:
            la = loops[fa]
            lo_a, hi_a = la.min(axis=0), la.max(axis=0)
            near = np.flatnonzero(np.all(lo_b <= hi_a, axis=1) & np.all(hi_b >= lo_a, axis=1))
            for j in near:
                fb = b[j]
                part = _clip_coplanar(la, loops[fb], normals[fb], tol)
                if len(part) < 3 or np.linalg.norm(_area_vector(part)) <= tol * tol:
                    continue
                facets.append(part)
                facet_pieces.append((owner[fa], owner[fb]))
                facet_box.append(0)
    # the vertices: merged, then inserted into the loops whose edges they lie on
    counts = [len(f) for f in facets]
    points = np.concatenate(facets)
    label = _merge_points(points, tol)
    first = np.full(label.max() + 1, -1, dtype=np.int64)
    first[label[::-1]] = np.arange(len(label))[::-1]
    vertices = points[first]
    index = np.split(label, np.cumsum(counts)[:-1])
    tree = cKDTree(vertices)
    loops_idx = [_simple(_with_t_junctions(_simple(np.asarray(ix, dtype=np.int64)), vertices,
                                           tree, tol)) for ix in index]  # fmt: skip
    keep = [i for i, ix in enumerate(loops_idx) if len(ix) >= 3]
    facets_idx = [loops_idx[i] for i in keep]
    facet_pieces = np.asarray(facet_pieces, dtype=np.int64)[keep]
    facet_box = np.asarray(facet_box, dtype=np.int64)[keep]
    piece_facets = [[] for _ in pieces]
    for f, (pa, pb) in enumerate(facet_pieces):
        piece_facets[pa].append(f)
        if pb >= 0:
            piece_facets[pb].append(f)
    return CellComplex(
        vertices=vertices,
        facets=facets_idx,
        facet_pieces=facet_pieces,
        facet_box=facet_box,
        piece_cell=np.asarray(cells, dtype=np.int64),
        piece_facets=[np.asarray(f, dtype=np.int64) for f in piece_facets],
        lower=lower,
        upper=upper,
        periodic=False,
    )


def _with_t_junctions(loop: np.ndarray, vertices: np.ndarray, tree, tol: float) -> np.ndarray:
    """The loop with the vertices that lie on its edges (strictly between their ends)
    inserted in order (``tree``: a k-d tree of the vertices)."""
    out = []
    for a, b in zip(loop, np.roll(loop, -1), strict=True):
        out.append(int(a))
        pa, pb = vertices[a], vertices[b]
        ab = pb - pa
        length2 = float(ab @ ab)
        if length2 <= tol * tol:
            continue
        near = tree.query_ball_point(0.5 * (pa + pb), 0.5 * np.sqrt(length2) + tol)
        inside = {}
        for w in near:
            if w in (a, b):
                continue
            t = float((vertices[w] - pa) @ ab) / length2
            if 0.0 < t < 1.0 and np.linalg.norm(pa + t * ab - vertices[w]) <= 10.0 * tol:
                inside[w] = t
        out.extend(sorted(inside, key=inside.get))
    return np.asarray(out, dtype=np.int64)


def _simple(loop: np.ndarray) -> np.ndarray:
    """The loop without repeated consecutive vertices."""
    if len(loop) == 0:
        return loop
    keep = loop != np.roll(loop, -1)
    return loop[keep]


@njit(cache=True)
def _greedy_boxes(labels):
    """A partition of a grid of labels into boxes of one label each (greedily: the first
    free cell grown along the last axis, then the second, then the first): rows ``(i0, j0,
    k0, i1, j1, k1, label)``, the upper bounds excluded."""
    nx, ny, nz = labels.shape
    done = np.zeros(labels.shape, dtype=np.bool_)
    out = np.empty((labels.size, 7), dtype=np.int64)
    n = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if done[i, j, k]:
                    continue
                lab = labels[i, j, k]
                k1 = k + 1
                while k1 < nz and not done[i, j, k1] and labels[i, j, k1] == lab:
                    k1 += 1
                j1 = j + 1
                while j1 < ny:
                    ok = True
                    for kk in range(k, k1):
                        if done[i, j1, kk] or labels[i, j1, kk] != lab:
                            ok = False
                            break
                    if not ok:
                        break
                    j1 += 1
                i1 = i + 1
                while i1 < nx:
                    ok = True
                    for jj in range(j, j1):
                        for kk in range(k, k1):
                            if done[i1, jj, kk] or labels[i1, jj, kk] != lab:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                    i1 += 1
                done[i:i1, j:j1, k:k1] = True
                out[n] = (i, j, k, i1, j1, k1, lab)
                n += 1
    return out[:n]


def grid_boxes(labels, edges) -> tuple[list, np.ndarray]:
    """Convex pieces of a structure made of the cells of a rectilinear grid (``labels``: a
    phase per cell; ``edges``: the coordinates of the planes of the grid along each axis):
    boxes of one phase each (:func:`_greedy_boxes`), and their phases."""
    from morphogen.models.scale_rich_3d import box_polyhedron

    boxes = _greedy_boxes(np.ascontiguousarray(labels, dtype=np.int64))
    lo = np.column_stack([edges[d][boxes[:, d]] for d in range(3)])
    hi = np.column_stack([edges[d][boxes[:, 3 + d]] for d in range(3)])
    return [box_polyhedron(a, b) for a, b in zip(lo, hi, strict=True)], boxes[:, 6]


# -- the meshes ------------------------------------------------------------------------------


def _width(poly) -> float:
    """Twice the radius of the largest ball in a convex polyhedron (its Chebyshev ball)."""
    from morphogen.mesh.polyhedra import _chebyshev_centre

    _, _, _, normals, planes = poly
    return 2.0 * _chebyshev_centre(np.asarray(normals, float), np.asarray(planes, float))[1]


def pieces_mesh(
    pieces,
    phase,
    lower,
    upper,
    *,
    size: float,
    min_size: float | None = None,
    hausdorff: float | None = None,
    gradation: float = 2.0,
    phases: str = "all",
    cells=None,
    thin: float | None = None,
    order: int = 1,
    periodic: bool = False,
    executable: str | None = None,
    workdir=None,
    timeout: float | None = None,
) -> TetMesh:
    """Conforming tetrahedral mesh of convex pieces partitioning the box (the
    ``(vertices, face offsets, labels, normals, planes)`` of each, its ``phase``): the
    face-to-face complex of the pieces (:func:`face_to_face`), cones from their centroids,
    remeshed by MMG keeping their facets and the faces of the box (every element knows its
    phase, and its cell when ``cells`` are given: ``mesh.cell``). Pieces thinner than the
    elements (plates) get flat elements, one through their thickness, of aspect ratio
    ``size`` over the thickness; ``thin``: the elements of a piece at most ``thin`` times
    its width (twice the radius of its largest ball) and at least ``min_size`` (default
    ``size / 10``), meshed through their thickness (as many elements as the square of the
    ratio of the size to the thickness, per unit area). ``phases``: ``"all"``,
    ``"solid"`` (phase 1) or ``"matrix"`` (phase 0); ``periodic``: matching opposite
    faces (meshed alike by ``mmg2d``, kept by MMG). ``hausdorff``: how far MMG may move
    the nodes off the facets, which are planar (default ``1e-6 size``: not at all, the
    facets exact; with the usual ``size / 20``, its smoothing bends their ridges into
    curves, their nodes a few thousandths of the size off the planes)."""
    from morphogen.mesh.polycrystal import _ridges, initial_mesh

    if phases not in ("all", "solid", "matrix"):
        raise ValueError("phases: 'all', 'solid' (phase 1) or 'matrix' (phase 0)")
    periodic = bool(periodic)
    if order not in (1, 2):
        raise ValueError("order: 1 or 2")
    size = float(size)
    if not size > 0:
        raise ValueError("size: must be positive")
    phase = np.asarray(phase, dtype=np.int64)
    given = cells is not None
    cells = np.asarray(cells, dtype=np.int64) if given else phase
    if len(phase) != len(pieces) or len(cells) != len(pieces):
        raise ValueError("one phase and one cell per piece")
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    hmin = float(min_size) if min_size is not None else size / 10.0
    hausd = float(hausdorff) if hausdorff is not None else 1e-6 * size
    exe = mmg_executable(executable)
    t0 = time.perf_counter()
    # a reference per piece: MMG keeps every facet (with references by phase, its swaps
    # cross some of the interfaces between the pieces of different phases)
    cx = face_to_face(pieces, np.arange(len(pieces)), lower, upper)
    t_cx = time.perf_counter() - t0
    sizing = ["-hmax", f"{size:.10g}", "-hmin", f"{hmin:.10g}", "-hausd", f"{hausd:.10g}",
              "-hgrad", f"{float(gradation):.10g}", "-v", "-1"]  # fmt: skip
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        # periodic: matching triangulations of opposite faces (mmg2d), kept by MMG
        xyz, tets, refs, tri, tri_refs, n_box = initial_mesh(
            cx, size=size, sizing=sizing, periodic=periodic, folder=folder, timeout=timeout
        )
        used = np.unique(tets)
        renum = np.full(len(xyz), -1, dtype=np.int64)
        renum[used] = np.arange(len(used))
        ridges = _ridges(cx)  # the edges where the interfaces or faces of the box bend
        corners = np.unique(ridges)
        required = renum[corners]
        if periodic:
            tol = TOLERANCE * float((upper - lower).max())
            pts = xyz[used]
            on_face = np.any((np.abs(pts - lower) <= tol) | (np.abs(pts - upper) <= tol), axis=1)
            required = np.union1d(required, np.flatnonzero(on_face))
        sol = None
        if thin is not None:
            # sizes: at most a multiple of the width of the thinnest piece around each
            # vertex, so that thin pieces get elements of their thickness
            width = np.array([_width(poly) for poly in pieces])
            local = np.clip(width[refs - 1] * float(thin), hmin, size)
            h = np.full(len(xyz), size)
            for k in range(4):
                np.minimum.at(h, tets[:, k], local)
            sol = folder / "pieces.sol"
            medit.write_sol(sol, h[used])
        medit.write_mesh(folder / "pieces.mesh", xyz[used], renum[tets], renum[tri],
                         tet_refs=refs, tri_refs=tri_refs, edges=renum[ridges],
                         ridges=np.arange(len(ridges)), corners=renum[corners],
                         required_vertices=required,
                         required_triangles=np.arange(n_box) if periodic else None)  # fmt: skip
        t1 = time.perf_counter()
        data = _run_mmg(exe, folder / "pieces.mesh", sol, folder / "out.mesh", sizing, timeout)
        t_mmg = time.perf_counter() - t1
    nodes = data["Vertices"]
    tol = TOLERANCE * float((upper - lower).max())
    for d in range(3):  # the nodes of the faces exactly on them
        nodes[np.abs(nodes[:, d] - lower[d]) <= tol, d] = lower[d]
        nodes[np.abs(nodes[:, d] - upper[d]) <= tol, d] = upper[d]
    elements = data["Tetrahedra"]
    piece = data["Tetrahedra_refs"] - 1
    flip = tet_volumes(nodes, elements) < 0
    elements[flip] = elements[flip][:, [0, 2, 1, 3]]
    used, inverse = np.unique(elements, return_inverse=True)
    mesh = TetMesh(
        nodes=nodes[used],
        elements=inverse.reshape(elements.shape),
        phase=phase[piece],
        lower=lower,
        upper=upper,
        periodic=periodic,
        cell=cells[piece] if given else None,
        metadata={
            "mesher": "exact convex pieces (mmg3d)",
            "size": size,
            "pieces": len(pieces),
            "facets": cx.n_facets,
            "phases": phases,
            "matching_faces": periodic,
            "time_complex_s": round(t_cx, 3),
            "time_mmg_s": round(t_mmg, 3),
        },
    )
    if phases == "solid":
        mesh = mesh.subset([1])
    elif phases == "matrix":
        mesh = mesh.subset([0])
    if order == 2:
        mesh = mesh.quadratic()
    log.info("exact mesh of %d pieces: %d elements, %d nodes (MMG %.1f s)", len(pieces),
             mesh.n_elements, mesh.n_nodes, t_mmg)  # fmt: skip
    return mesh


def polyhedral_mesh(micro, *, size: float, by_piece: bool = False, **options) -> TetMesh:
    """Conforming tetrahedral mesh of a faceted microstructure from its exact convex pieces
    (``micro.convex_pieces()``: the pieces, their phases and cells; ``by_piece``: every
    element knows the cell of its piece, ``mesh.cell``), periodic (matching faces) when
    the microstructure is; see :func:`pieces_mesh` for the options."""
    if not callable(getattr(micro, "convex_pieces", None)):
        raise TypeError(f"{type(micro).__name__}: no convex pieces, no polyhedral mesh")
    pieces, phase, cells = micro.convex_pieces()
    domain = micro.domain
    if options.get("periodic") is None:
        options["periodic"] = bool(domain.periodic)
    if options["periodic"] and not domain.periodic:
        raise ValueError("matching periodic faces need a periodic microstructure")
    return pieces_mesh(pieces, phase, domain.lower, domain.upper, size=size,
                       cells=cells if by_piece else None, **options)  # fmt: skip


# -- the adaptive meshes of level sets --------------------------------------------------------


def _to_split(fa, fm, fb, length, *, hmin: float, hausd: float, lipschitz: float):
    """Whether segments (the level set at their ends and middles, their lengths) must be
    split to resolve the interface (see the module): two crossings (a thin feature), one
    crossing whose linear interpolation is off by more than ``hausd`` (the deviation of
    the parabola through the three values, over the slope), or no crossing but a bend
    toward zero larger than ``hausd`` where the slope ``lipschitz`` allows the interface
    (a feature between the samples; the kinks of a distance on the medial surfaces of the
    phases bend away from zero: they stay); none shorter than 1.5 ``hmin``. A kink between
    a sample and a crossing misplaces the zero beyond what the parabola tells: the level
    set at the linear zeros finds it (in :func:`adaptive_mesh`)."""
    e = fm - 0.5 * (fa + fb)
    sa, sm, sb = np.sign(fa), np.sign(fm), np.sign(fb)
    half = 0.5 * lipschitz * length
    same = (sa == sm) & (sm == sb)
    covered = same & (np.abs(fa) + np.abs(fm) >= half) & (np.abs(fm) + np.abs(fb) >= half)
    two = (sa == sb) & (sm != sa)
    one = sa != sb
    jump = np.where(one, fa - fb, 1.0)  # (not zero where the line crosses)
    t = np.clip(fa / jump, 0.0, 1.0)
    shift = np.abs(4.0 * t * (1.0 - t) * e / jump) * length
    toward = same & ~covered & (e * sa < 0) & (np.abs(e) > hausd)
    return (two | (one & (shift > hausd)) | toward) & (length > 1.5 * hmin)


def _unique_edges(simplices: np.ndarray) -> np.ndarray:
    """The edges (sorted pairs of vertices) of triangles or tetrahedra."""
    k = simplices.shape[1]
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    edges = np.sort(simplices[:, pairs].reshape(-1, 2), axis=1)
    return np.unique(edges, axis=0)


def _cut_faces(phi, nodes, tets, tri, refs, faces, group, *, earlier=(), zero=0.0, near=0.15,
               iterations=60):  # fmt: skip
    """The mesh with the triangles of the ``faces`` (references) of the box cut exactly
    along the zero level set ``phi``: their edges between nodes of opposite signs split at
    the zero (bisection of ``phi`` along the edge), or one of their ends moved to it when it
    lies within ``near`` of the edge (along the edge, on the faces of the box it lies on),
    alike for the translates of an edge (``group``: a label per node, shared by the
    translates); the tetrahedra around a split edge split in two. ``earlier``: the values
    at the nodes of the level sets cut before (several phases), whose zeros never move and
    whose negative triangles (their phases) are not cut; the nodes added or moved take
    their linear interpolation, so that their traces stay as they were. Values within
    ``zero`` of zero are zeros. Returns the nodes, tetrahedra, triangles, their references,
    the values of ``phi`` at the nodes (zero on the trace), those of the earlier level sets,
    and the numbers of nodes moved and added."""
    values = np.asarray(phi(nodes), dtype=float)
    values[np.abs(values) <= zero] = 0.0
    first = np.full(group.max() + 1, -1, dtype=np.int64)
    first[group[::-1]] = np.arange(len(group))[::-1]
    values = values[first[group]]  # the same on the translates
    earlier = [np.asarray(e, dtype=float).copy() for e in earlier]
    nodes, tets, tri, refs, group = nodes.copy(), tets.copy(), tri.copy(), refs.copy(), group.copy()
    moved = added = 0

    def active():
        """The triangles of the faces out of the phases cut before."""
        on = tri[np.isin(refs, faces)]
        for e in earlier:
            on = on[np.all(e[on] >= 0.0, axis=1)]
        return on

    def crossing():
        edges = _unique_edges(active())
        fa, fb = values[edges[:, 0]], values[edges[:, 1]]
        return edges[fa * fb < 0]

    def root(edges):
        a, b = nodes[edges[:, 0]], nodes[edges[:, 1]]
        lo, hi = np.zeros(len(edges)), np.ones(len(edges))
        flo = values[edges[:, 0]]
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            fm = phi(a + mid[:, None] * (b - a))
            same = np.sign(fm) == np.sign(flo)
            lo, flo, hi = np.where(same, mid, lo), np.where(same, fm, flo), np.where(same, hi, mid)
        return 0.5 * (lo + hi)

    def faces_of(points):
        lo, hi = nodes.min(axis=0), nodes.max(axis=0)
        tol = TOLERANCE * float((hi - lo).max())
        return (np.abs(points - lo) <= tol) | (np.abs(points - hi) <= tol)

    frozen = np.zeros(len(nodes), dtype=bool)  # the nodes of the earlier traces
    for e in earlier:
        frozen |= e == 0.0
    # 1. the ends close to a zero move onto it (along an edge of the same faces), one end
    # per crossed edge at most, and only where nodes of its sign remain around it on the
    # faces (the trace keeps its topology: no node of one sign erased)
    edges = crossing()
    if len(edges):
        t = root(edges)
        side = faces_of(nodes)
        on = active()
        face_edges = _unique_edges(on)
        same = np.zeros(len(nodes), dtype=np.int64)  # neighbours of the same sign
        sa, sb = np.sign(values[face_edges[:, 0]]), np.sign(values[face_edges[:, 1]])
        np.add.at(same, face_edges[:, 0], (sa == sb) & (sa != 0))
        np.add.at(same, face_edges[:, 1], (sa == sb) & (sa != 0))
        # the changes of sign around a node (its triangles whose two other nodes have
        # opposite signs): more than two, it is a saddle of the trace, and it stays
        changes = np.zeros(len(nodes), dtype=np.int64)
        sign = np.sign(values[on])
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            np.add.at(changes, on[:, i], sign[:, j] * sign[:, k] < 0)
        cand = []
        for end, frac in ((0, t), (1, 1.0 - t)):
            v = edges[:, end]
            other = edges[:, 1 - end]
            ok = (frac <= near) & np.all(side[v] <= side[other], axis=1)  # stays on its faces
            ok &= ~np.all(side[v], axis=1) & (same[v] > 0) & ~frozen[v] & (changes[v] <= 2)
            cand += [(frac[k], int(v[k]), k) for k in np.flatnonzero(ok)]
        accepted: dict[int, int] = {}
        blocked = set()
        for frac, v, k in sorted(cand):
            if v in accepted or (v in blocked and frac > 1e-3):
                continue  # (a zero within a thousandth of a node is taken there)
            accepted[v] = k
            blocked.update((int(edges[k, 0]), int(edges[k, 1])))  # its edge: one end
        chosen = np.array(sorted(accepted), dtype=np.int64)
        target = np.zeros((len(nodes), 3))
        interp = [np.zeros(len(nodes)) for _ in earlier]
        for v, k in accepted.items():
            a, b = edges[k]
            target[v] = nodes[a] + t[k] * (nodes[b] - nodes[a])
            for e, out in zip(earlier, interp, strict=True):
                out[v] = e[a] + t[k] * (e[b] - e[a])
        # one node of each group moves, the translates follow; no move that flips
        rep = np.full(group.max() + 1, -1, dtype=np.int64)
        rep[group[chosen]] = chosen
        shift = np.zeros((group.max() + 1, 3))
        shift[group[chosen]] = target[rep[group[chosen]]] - nodes[rep[group[chosen]]]
        follow = np.flatnonzero(np.isin(group, group[chosen]))
        trial = nodes.copy()
        trial[follow] += shift[group[follow]]
        around = np.flatnonzero(np.isin(tets, follow).any(axis=1))
        bad = tet_volumes(trial, tets[around]) <= 0
        if bad.any():
            undo = np.isin(group, group[np.unique(tets[around][bad])])
            trial[undo] = nodes[undo]
            follow = follow[~undo[follow]]
        before = values.copy()
        values[follow] = 0.0
        # a trace that touches itself, or branches: the moves of its nodes are undone (the
        # edges they cross are split instead)
        for _ in range(20):
            pinched = _pinched(active(), values)
            back = np.isin(group, group[np.intersect1d(pinched, follow)])
            if not back.any():
                break
            trial[back] = nodes[back]
            values[back] = before[back]
            follow = follow[~back[follow]]
        for e, out in zip(earlier, interp, strict=True):
            lead = rep[group[follow]]
            e[follow] = out[lead]
        nodes = trial
        moved = len(follow)
    # 2. the other crossed edges split at their zeros, in rounds of disjoint stars
    for _ in range(64):
        edges = crossing()
        if not len(edges):
            break
        t = root(edges)
        # the parameter of an edge for all its translates: that of its first one
        key = np.sort(group[edges], axis=1)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        lead = np.full(inv.max() + 1, -1, dtype=np.int64)
        lead[inv[::-1]] = np.arange(len(inv))[::-1]
        a_lead = group[edges[lead[inv], 0]]
        t = np.where(a_lead == group[edges[:, 0]], t[lead[inv]], 1.0 - t[lead[inv]])
        tet_edges = np.sort(tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]], axis=2)
        m = len(nodes)
        tkey = (tet_edges[..., 0] * m + tet_edges[..., 1]).ravel()
        ekey = edges[:, 0] * m + edges[:, 1]
        order = np.argsort(tkey, kind="stable")
        lo = np.searchsorted(tkey[order], ekey, side="left")
        hi = np.searchsorted(tkey[order], ekey, side="right")
        used = np.zeros(len(tets), dtype=bool)
        new_tets, new_tri, new_nodes = [], [], []
        new_earlier: list = [[] for _ in earlier]
        drop_tets = np.zeros(len(tets), dtype=bool)
        tri_of: dict = {}
        for f, (i, j, k_) in enumerate(np.sort(tri, axis=1).tolist()):
            for e in ((i, j), (i, k_), (j, k_)):
                tri_of.setdefault(e, []).append(f)
        # an edge and its translates split in the same round (splits of two edges of one
        # triangle in turn leave it split along a diagonal that depends on their order)
        units = np.split(np.argsort(inv, kind="stable"),
                         np.flatnonzero(np.diff(np.sort(inv))) + 1)  # fmt: skip
        for unit in units:
            arounds = [order[lo[k] : hi[k]] // 6 for k in unit]
            if any(used[around].any() for around in arounds):
                continue
            for k, around in zip(unit, arounds, strict=True):
                used[around] = True
                a, b = edges[k]
                p = len(nodes) + len(new_nodes)
                new_nodes.append(nodes[a] + t[k] * (nodes[b] - nodes[a]))
                for e, out in zip(earlier, new_earlier, strict=True):
                    out.append(e[a] + t[k] * (e[b] - e[a]))
                for e in around:
                    tet = tets[e]
                    drop_tets[e] = True
                    new_tets.append(np.where(tet == b, p, tet))
                    new_tets.append(np.where(tet == a, p, tet))
                for f in tri_of.get((int(min(a, b)), int(max(a, b))), []):
                    face = tri[f]
                    new_tri.append((f, np.where(face == b, p, face),
                                    np.where(face == a, p, face)))  # fmt: skip
        if not new_nodes:
            break
        nodes = np.vstack([nodes, np.array(new_nodes)])
        tets = np.vstack([tets[~drop_tets], np.array(new_tets, dtype=np.int64)])
        gone = np.zeros(len(tri), dtype=bool)
        extra, extra_refs = [], []
        for f, t1, t2 in new_tri:
            gone[f] = True
            extra += [t1, t2]
            extra_refs += [refs[f], refs[f]]
        tri = np.vstack([tri[~gone], np.array(extra, dtype=np.int64).reshape(-1, 3)])
        refs = np.concatenate([refs[~gone], np.array(extra_refs, dtype=np.int64)])
        values = np.concatenate([values, np.zeros(len(new_nodes))])
        earlier = [np.concatenate([e, np.array(out)]) for e, out in
                   zip(earlier, new_earlier, strict=True)]  # fmt: skip
        frozen = np.concatenate([frozen, np.zeros(len(new_nodes), dtype=bool)])
        group = np.concatenate([group, np.full(len(new_nodes), -1)])
        added += len(new_nodes)
    flip = tet_volumes(nodes, tets) < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    return nodes, tets, tri, refs, values, earlier, moved, added


def _split_across(phi, nodes, tets, tri, face, done, index, iterations=60):
    """The edges of tetrahedra between two nodes of the faces (``face``) that are not edges
    of the faces (they cross the box near its edges) and that the level set ``phi`` (the
    ``index``-th of ``done``, its values at the nodes) crosses, split at its zero; the
    values of the others at the new nodes (inside the box): their own. Returns the nodes,
    tetrahedra, the number of nodes added and the values."""
    edges = _unique_edges(tets)
    edges = edges[face[edges[:, 0]] & face[edges[:, 1]]]
    on = _unique_edges(tri)
    m = len(nodes)
    inner = ~np.isin(edges[:, 0] * m + edges[:, 1], on[:, 0] * m + on[:, 1])
    edges = edges[inner]
    values = done[index]
    edges = edges[values[edges[:, 0]] * values[edges[:, 1]] < 0]
    if not len(edges):
        return nodes, tets, 0, done
    a, b = nodes[edges[:, 0]], nodes[edges[:, 1]]
    lo, hi = np.zeros(len(edges)), np.ones(len(edges))
    flo = values[edges[:, 0]]
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        fm = phi(a + mid[:, None] * (b - a))
        same = np.sign(fm) == np.sign(flo)
        lo, flo, hi = np.where(same, mid, lo), np.where(same, fm, flo), np.where(same, hi, mid)
    points = a + (0.5 * (lo + hi))[:, None] * (b - a)
    tets = tets.copy()
    for (i, j), point in zip(edges.tolist(), points, strict=True):
        p = len(nodes)
        nodes = np.vstack([nodes, point])
        around = np.flatnonzero(np.any(tets == i, axis=1) & np.any(tets == j, axis=1))
        halves = [np.where(tets[around] == j, p, tets[around]),
                  np.where(tets[around] == i, p, tets[around])]  # fmt: skip
        keep = np.ones(len(tets), dtype=bool)
        keep[around] = False
        tets = np.vstack([tets[keep], *halves])
    added = len(edges)
    done = [np.concatenate([d, np.full(added, np.nan)]) for d in done]
    done[index][-added:] = 0.0
    flip = tet_volumes(nodes, tets) < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    return nodes, tets, added, done


def _manifold_balls(tets, values, nodes_at, tiny) -> int:
    """The values of the level set made manifold around the zeros ``nodes_at`` (the nodes
    of the matching faces that the traces go through): around such a node, the nodes of its
    tetrahedra of each sign must form one component (MMG moves the zero off a node where
    two sheets of the interface touch, and would cut the faces that it keeps); the nodes
    of the smaller components, but those of the faces, take the other sign (``tiny``).
    Returns the number of nodes changed."""
    if not len(nodes_at):
        return 0
    order = np.argsort(tets.ravel(), kind="stable")
    owners = order // 4
    start = np.searchsorted(tets.ravel()[order], nodes_at, side="left")
    stop = np.searchsorted(tets.ravel()[order], nodes_at, side="right")
    fixed = np.zeros(len(values), dtype=bool)
    fixed[nodes_at] = True
    changed = 0
    for a, lo, hi in zip(nodes_at.tolist(), start.tolist(), stop.tolist(), strict=True):
        ball = tets[owners[lo:hi]]
        link = [t[t != a] for t in ball]
        for sign in (-1.0, 1.0):
            parent: dict = {}

            def find(x, parent=parent):
                while parent.setdefault(x, x) != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for tri in link:
                members = [int(x) for x in tri if values[x] * sign > 0]
                for x in members:
                    find(x)
                for x, y in itertools.combinations(members, 2):
                    parent[find(x)] = find(y)
            groups: dict = {}
            for x in parent:
                groups.setdefault(find(x), []).append(x)
            if len(groups) < 2:
                continue
            parts = sorted(groups.values(), key=len, reverse=True)
            keep = next((g for g in parts if fixed[g].any()), parts[0])
            for g in parts:
                if g is keep or fixed[g].any():
                    continue
                values[g] = -sign * tiny
                changed += len(g)
    return changed


def _pinched(tri, values) -> np.ndarray:
    """The nodes where the zero level set of the piecewise linear interpolation of
    ``values`` on triangles is not a set of simple curves: the ends of the edges of zeros
    whose triangles all have one sign on their third vertex (the trace touches itself), the
    vertices of triangles of zeros, and the zeros where more than two edges of zeros
    meet."""
    tri = np.asarray(tri, dtype=np.int64)
    a = tri[:, [0, 1, 0]].ravel()
    b = tri[:, [1, 2, 2]].ravel()
    c = tri[:, [2, 0, 1]].ravel()
    zero = (values[a] == 0.0) & (values[b] == 0.0)
    if not zero.any():
        return np.zeros(0, dtype=np.int64)
    key = np.column_stack([np.minimum(a, b), np.maximum(a, b)])[zero]
    side = np.sign(values[c[zero]])
    edges, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    pos = np.zeros(len(edges), dtype=bool)
    neg = np.zeros(len(edges), dtype=bool)
    np.logical_or.at(pos, inverse, side > 0)
    np.logical_or.at(neg, inverse, side < 0)
    bad = [edges[pos != neg].ravel()]
    bad.append(tri[np.all(values[tri] == 0.0, axis=1)].ravel())  # triangles of zeros
    degree = np.bincount(edges.ravel(), minlength=len(values))  # branching traces
    bad.append(np.flatnonzero(degree > 2))
    return np.unique(np.concatenate(bad))


def _groups(points, lower, upper, axes, tol) -> np.ndarray:
    """A label per point, shared by the translates of the points of the faces along the
    periodic axes (their positions folded onto the lower faces)."""
    folded = np.asarray(points, dtype=float).copy()
    for d in range(3):
        if axes[d]:
            folded[np.abs(folded[:, d] - upper[d]) <= tol, d] = lower[d]
    key = np.round((folded - lower) / (1e-3 * tol)).astype(np.int64)
    return np.unique(key, axis=0, return_inverse=True)[1].ravel()


def _box_ridges(nodes, tri, lower, upper, tol):
    """The edges of the triangles of the faces of the box (``tri``) along its edges, and
    its corners: MMG's ridges and corners, so that the box keeps its edges when MMG does not
    detect ridges by angles."""
    on = np.zeros((len(nodes), 6), dtype=bool)
    for d in range(3):
        on[:, 2 * d] = np.abs(nodes[:, d] - lower[d]) <= tol
        on[:, 2 * d + 1] = np.abs(nodes[:, d] - upper[d]) <= tol
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    edges = np.unique(edges, axis=0)
    along = (on[edges[:, 0]] & on[edges[:, 1]]).sum(axis=1) >= 2
    return edges[along], np.flatnonzero(on.sum(axis=1) == 3)


def _project(phi, nodes, elements, moving, *, step, limit, lower, upper, tol, group=None,
             rounds=4):  # fmt: skip
    """The nodes ``moving`` moved onto the zero level set by Newton's method along its
    gradient (central differences of ``step``), on the faces of the box they lie on; the
    moves larger than ``limit`` undone, and those that would flip an element or flatten it
    (to a quality below 0.02 and a fifth of its own) halved until they do not, four times
    at most, then undone (for all the nodes of a ``group``: the translates of a node of a
    periodic face move alike). ``phi`` may give several functions at each point (a column
    each, the surfaces of the pieces that meet along a sharp edge): the nodes then move to
    where they all vanish, by the smallest move at each step (the pseudo-inverse of their
    Jacobian). Returns the nodes, the number moved and the largest move."""
    if group is not None:  # one node of each group moves, the others follow
        rep = np.full(group.max() + 1, -1, dtype=np.int64)
        rep[group[moving][::-1]] = moving[::-1]
        moving = np.unique(rep[group[moving]])
    x0 = nodes[moving]
    x = x0.copy()
    fixed = np.zeros(x.shape, dtype=bool)
    for d in range(3):
        fixed[:, d] = (np.abs(x[:, d] - lower[d]) <= tol) | (np.abs(x[:, d] - upper[d]) <= tol)
    for _ in range(rounds):
        f = np.asarray(phi(x), dtype=float)
        if f.ndim == 1:
            g = np.empty_like(x)
            for d in range(3):
                e = np.zeros(3)
                e[d] = step
                g[:, d] = (phi(x + e) - phi(x - e)) / (2.0 * step)
            g[fixed] = 0.0
            g2 = np.einsum("ij,ij->i", g, g)
            ok = g2 > 1e-12
            x[ok] -= (f[ok] / g2[ok])[:, None] * g[ok]
            continue
        jacobian = np.empty((len(x), f.shape[1], 3))
        for d in range(3):
            e = np.zeros(3)
            e[d] = step
            jacobian[:, :, d] = (np.asarray(phi(x + e)) - np.asarray(phi(x - e))) / (2.0 * step)
        jacobian *= ~fixed[:, None, :]
        x -= np.einsum("ndk,nk->nd", np.linalg.pinv(jacobian), f)
    move = np.linalg.norm(x - x0, axis=1)
    ok = np.isfinite(move) & (move <= limit)
    shift = np.zeros_like(nodes)
    shift[moving[ok]] = x[ok] - x0[ok]
    touched = np.zeros(len(nodes), dtype=bool)
    touched[moving[ok]] = True
    if group is not None:  # the translates follow
        by_group = np.zeros((group.max() + 1, 3))
        by_group[group[moving[ok]]] = shift[moving[ok]]
        member = np.zeros(group.max() + 1, dtype=bool)
        member[group[moving[ok]]] = True
        touched = member[group]
        shift[touched] = by_group[group[touched]]
    # the moves that flip an element, or flatten it (below a fiftieth, and a fifth of its
    # quality), halved (the node as near the surface as the elements allow), then undone
    near = np.flatnonzero(touched[elements].any(axis=1))
    q0 = tet_quality(nodes, elements[near])
    scale = touched.astype(float)  # the fraction of its move that each node makes
    out = nodes + scale[:, None] * shift
    for _ in range(100):
        out = nodes + scale[:, None] * shift
        vol = tet_volumes(out, elements[near])
        q = tet_quality(out, elements[near])
        bad = (vol <= 0) | ((q < 0.02) & (q < 0.2 * q0))
        if not bad.any():
            break
        back = np.unique(elements[near][bad])
        back = back[scale[back] > 0]
        if group is not None:
            back = np.flatnonzero(np.isin(group, group[back]) & (scale > 0))
        scale[back] = np.where(scale[back] > 0.1, 0.5 * scale[back], 0.0)
    else:  # (never seen) the moves of the nodes of the elements still bad undone
        while bad.any():
            back = np.unique(elements[near][bad])
            if group is not None:
                back = np.flatnonzero(np.isin(group, group[back]))
            scale[back] = 0.0
            out = nodes + scale[:, None] * shift
            q = tet_quality(out, elements[near])
            bad = (tet_volumes(out, elements[near]) <= 0) | ((q < 0.02) & (q < 0.2 * q0))
    touched = scale > 0
    moved = np.flatnonzero(touched)
    largest = float(np.linalg.norm(out[moved] - nodes[moved], axis=1).max()) if moved.size else 0.0
    return out, int(moved.size), largest


class _Cut:
    """A phase cut out of what the cuts before left by its level set ``function``: one of
    the pieces of :func:`morphogen.mesh.conforming._mmg_pieces`, which cuts them in turn
    without remeshing, then remeshes once."""

    def __init__(self, phase: int, function):
        self.phase, self.function = int(phase), function

    def at(self, points, band: float) -> np.ndarray:
        return np.clip(self.function(np.asarray(points, dtype=float)), -band, band)


def _kept(phases, present: list[int]) -> tuple[str, list[int] | None]:
    """The phases kept of a structure of the phases ``present``: ``"all"``, ``"solid"``
    or ``"matrix"`` (of the phases 0 and 1, kept by MMG), or a list (kept from the whole
    mesh). Returns the name of the two-phase request (``"all"`` otherwise) and the list of
    the phases to keep (None: all)."""
    if isinstance(phases, str):
        if phases not in ("all", "solid", "matrix"):
            raise ValueError(
                "phases: 'all', 'solid' (phase 1), 'matrix' (phase 0) or the phases to keep"
            )
        if phases != "all" and present != [0, 1]:
            raise ValueError(
                f"phases: 'all' or the phases to keep among {present} (a structure of "
                "several phases)"
            )
        return phases, None
    wanted = sorted({int(k) for k in phases})
    unknown = sorted(set(wanted) - set(present))
    if unknown:
        raise ValueError(f"phases: {unknown} are not phases of this structure ({present})")
    if present == [0, 1]:  # two phases: MMG keeps one
        return {(1,): "solid", (0,): "matrix"}.get(tuple(wanted), "all"), None
    return "all", wanted


def _sharp_groups(sharp, sharp_pieces, n: int) -> dict[tuple, np.ndarray]:
    """The nodes of the sharp edges by the pieces that meet there: two for the nodes of
    one sharp edge, more where sharp edges join."""
    found: dict[int, set] = {}
    for (a, b), pair in zip(sharp.tolist(), sharp_pieces.tolist(), strict=True):
        for node in (a, b):
            found.setdefault(node, set()).update(pair)
    groups: dict[tuple, list[int]] = {}
    for node, pieces in found.items():
        groups.setdefault(tuple(sorted(pieces)), []).append(node)
    return {key: np.array(sorted(nodes), dtype=np.int64) for key, nodes in groups.items()}


def adaptive_mesh(
    micro,
    *,
    size: float,
    hausdorff: float | None = None,
    min_size: float | None = None,
    gradation: float = 1.8,
    phases="all",
    order: int = 1,
    periodic=None,
    lipschitz: float | None = None,
    project: bool = True,
    remove_islands: float | None = None,
    ridges: bool = False,
    sharp_edges: bool | None = None,
    retries: int = 2,
    executable: str | None = None,
    workdir=None,
    timeout: float | None = None,
) -> TetMesh:
    """Conforming tetrahedral mesh of the exact implicit functions of the phases of a
    microstructure (``micro.phase_level_sets_at``; an object with a ``level_set_at``
    only: its two phases), refined where they need it, without a grid (see the module).

    Parameters
    ----------
    size
        Largest element size.
    hausdorff
        Largest distance between the meshed and the exact interfaces (default ``size /
        20``): the edges near the interface are split until the level set departs from its
        linear interpolation by at most this distance.
    min_size
        Smallest element size (default: ``hausdorff``): features thinner than about it are
        meshed less accurately.
    gradation
        Largest ratio of the sizes of neighbouring elements.
    phases
        ``"all"``, ``"solid"`` (phase 1) or ``"matrix"`` (phase 0) of a structure of the
        phases 0 and 1; or the phases to keep (a list), of any structure.
    order
        1 (4-node tetrahedra) or 2 (10-node tetrahedra, curved onto the interfaces by
        :func:`morphogen.mesh.curved.curved_mesh` when ``project``, straight otherwise).
    periodic
        Matching opposite faces (default: when the microstructure is periodic), or three
        booleans: along those axes.
    lipschitz
        Bound of the slope of the level set (default: the microstructure's
        ``lipschitz``, 1 for a signed distance): the edges farther from the interface than
        it allows are not refined.
    project
        Move the nodes of the interface onto the exact zero level set.
    remove_islands
        Relative volume below which MMG removes isolated components (``-rmc``; none by
        default: the geometry is exact; non-periodic meshes of two phases only).
    ridges
        Let MMG detect ridges on the cut (dihedral angles above 45°): the sharp edges of the
        geometry (the intersections of spheres, the rims of dead leaves) stay sharp, at the
        price of flat elements where the facets of the cut bend at the scale of the octree
        (0.5 % of the elements of a sheet gyroid below a quality of 10⁻³). Off by default:
        those edges are rounded within the Hausdorff distance, the edges of the box kept.
    sharp_edges
        Follow the sharp edges of the phases of the structures that give their smooth
        pieces (:meth:`~morphogen.models.base.Microstructure.implicit_pieces`: the unions
        of spheres and segments, their coatings and bores) with edges of the mesh: the
        octree is refined on the surfaces of the pieces, the box cut by each in turn, and
        the lines where two cuts meet kept as ridges (``mesh.sharp``), their nodes moved
        onto both surfaces. By default when two pieces of a phase meet; false rounds these
        edges within the Hausdorff distance. Two primitives that enter each other by less
        than the Hausdorff distance are in the same piece (their edge is rounded).
    retries
        Periodic meshes whose opposite faces do not match (MMG remeshes a face where the
        interface touches itself on it) are made again on another octree (one more root
        cell along each axis), up to this number of times.
    """
    return _adaptive_mesh(
        micro,
        size=size,
        hausdorff=hausdorff,
        min_size=min_size,
        gradation=gradation,
        phases=phases,
        order=order,
        periodic=periodic,
        lipschitz=lipschitz,
        project=project,
        remove_islands=remove_islands,
        ridges=ridges,
        sharp_edges=sharp_edges,
        retries=retries,
        executable=executable,
        workdir=workdir,
        timeout=timeout,
        attempt=0,
    )


def _adaptive_mesh(
    micro,
    *,
    size: float,
    hausdorff: float | None = None,
    min_size: float | None = None,
    gradation: float = 1.8,
    phases="all",
    order: int = 1,
    periodic=None,
    lipschitz: float | None = None,
    project: bool = True,
    remove_islands: float | None = None,
    ridges: bool = False,
    sharp_edges: bool | None = None,
    retries: int = 2,
    executable: str | None = None,
    workdir=None,
    timeout: float | None = None,
    attempt: int,
    at_once: bool = False,
) -> TetMesh:
    """:func:`adaptive_mesh`, on an octree of ``attempt`` more root cells per side
    (``at_once``: the two phases cut first and remeshed once, as several are)."""
    from morphogen.mesh.conforming import REF_PHASE, _mmg_pieces
    from morphogen.mesh.octree import LINES, SEGMENTS, Octree
    from morphogen.models.implicit import functions_at, implicit_phases_of

    if order not in (1, 2):
        raise ValueError("order: 1 or 2")
    size = float(size)
    if not size > 0:
        raise ValueError("size: must be positive")
    hausd = float(hausdorff) if hausdorff is not None else size / 20.0
    hmin = float(min_size) if min_size is not None else hausd
    if not (hausd > 0 and 0 < hmin <= size):
        raise ValueError("hausdorff and min_size: positive, min_size at most size")
    present = implicit_phases_of(micro)
    two, keep = _kept(phases, present)
    domain = micro.domain
    axes = _periodic_axes(domain.periodic if periodic is None else periodic)
    if any(axes) and not domain.periodic:
        raise ValueError("matching periodic faces need a periodic microstructure")
    lip = float(lipschitz if lipschitz is not None else getattr(micro, "lipschitz", 1.0))
    lower = np.asarray(domain.lower, dtype=float)
    upper = np.asarray(domain.upper, dtype=float)
    tol = TOLERANCE * float((upper - lower).max())
    reach = max(lip, 1.0) * size  # the lines of the octree are at most ``size`` long
    # the phases cut one after the other out of the rest: phase 0, or the last one
    rest = 0 if 0 in present else present[-1]
    keys = [k for k in present if k != rest]
    if not keys:
        raise ValueError(f"{type(micro).__name__}: a single phase, no interface to mesh")
    several = len(keys) > 1
    try:
        functions_at(micro, np.reshape(lower, (1, 3)), reach)
    except NotImplementedError:
        raise TypeError(
            f"{type(micro).__name__}: no point-wise functions of its phases "
            "(phase_level_sets_at), so no adaptive mesh; see morphogen.mesh.exact.exact_mesh"
        ) from None
    # the smooth pieces of the phases, whose sharp edges the mesh follows
    pieces = None
    if sharp_edges is None or sharp_edges:
        found = getattr(micro, "implicit_pieces", lambda depth: None)(hausd)
        if found is None and sharp_edges:
            raise ValueError(
                f"sharp_edges: {type(micro).__name__} does not give the smooth pieces of its "
                "phases (the unions of spheres and segments and their coatings do)"
            )
        if found is not None:
            counts: dict[int, int] = {}
            for piece in found[0]:
                counts[piece.phase] = counts.get(piece.phase, 0) + 1
            if sharp_edges or max(counts.values(), default=0) > 1:
                pieces, rest = found

    def phis(points, r: float = reach) -> dict:
        """The functions of the phases at the points, exact within ``r`` of the
        interfaces."""
        out = functions_at(micro, points, r)
        sets = {k: np.asarray(out[k], dtype=float) for k in present if k in out}
        if rest not in sets:  # the phase that the others leave
            sets[rest] = -np.minimum.reduce([sets[k] for k in keys])
        return sets

    def level(k):
        """The level set that cuts phase ``k`` out of the rest: half its own minus the
        smallest of those of the phases cut after it and of the rest — the boundaries of
        ``k`` with these only, clearly negative on its boundaries with the phases cut
        before (two phases: that of the phase cut)."""
        later = [*keys[keys.index(k) + 1 :], rest]

        def function(points):
            sets = phis(points)
            return 0.5 * (sets[k] - np.minimum.reduce([sets[j] for j in later]))

        return function

    def between(a, b, r: float = reach):
        """Half the difference of the level sets of two phases: zero on their interface."""

        def function(points):
            sets = phis(points, r)
            return 0.5 * (sets[a] - sets[b])

        return function

    def surfaces(points, r: float) -> list:
        """The functions that the octree resolves, exact within ``r``: those of the phases
        cut, or of the smooth pieces."""
        if pieces is not None:
            return [piece.at(points, r) for piece in pieces]
        sets = phis(points, r)
        return [sets[k] for k in keys]

    exe = mmg_executable(executable)
    t0 = time.perf_counter()
    # the octree: root cells of about the size, split where their lines miss a level set
    n = np.maximum(1, np.ceil((upper - lower) / size - 1e-9)).astype(int) + attempt
    depth = int(np.ceil(np.log2(float(np.max((upper - lower) / n)) / hmin) - 1e-9))
    tree = Octree(lower, upper, n, axes, max(depth, 0))

    def test(points, sides):
        # the cells of a level together: their lines need the level sets within their
        # length, beyond which the tests see them clipped (the cost of the functions of
        # many objects, leaves or primitives, grows as the cube of that reach)
        r = min(reach, max(lip, 1.0) * float(np.max(sides)))
        fields = surfaces(points.reshape(-1, 3), r)
        split = np.zeros(len(points), dtype=bool)
        for values in fields:
            f = values.reshape(-1, 27)
            for d, (i0, i1, i2) in LINES:
                split |= _to_split(f[:, i0], f[:, i1], f[:, i2], sides[:, d], hmin=hmin,
                                   hausd=hausd, lipschitz=lip)  # fmt: skip
        # the others: along the segments that may be edges of their tetrahedra, a kink
        # between the samples misplaces the zero of the linear interpolation, which the
        # level set tells
        rest_cells = np.flatnonzero(~split & (sides.min(axis=1) > 1.5 * hmin))
        for chunk in np.array_split(rest_cells, max(1, -(-len(rest_cells) // 8192))):
            cells, where, which = [], [], []
            for i, values in enumerate(fields):
                f = values.reshape(-1, 27)[chunk]
                fa, fb = f[:, SEGMENTS[:, 0]], f[:, SEGMENTS[:, 1]]
                c, j = np.nonzero(fa * fb < 0)
                t = fa[c, j] / (fa[c, j] - fb[c, j])
                a, b = points[chunk[c], SEGMENTS[j, 0]], points[chunk[c], SEGMENTS[j, 1]]
                cells.append(chunk[c])
                where.append(a + t[:, None] * (b - a))
                which.append(np.full(len(c), i))
            cells = np.concatenate(cells)
            if len(cells):
                which = np.concatenate(which)
                at = surfaces(np.concatenate(where), r)
                value = np.zeros(len(cells))
                for i in range(len(fields)):
                    value[which == i] = at[i][which == i]
                split[cells[np.abs(value) > hausd]] = True
        return split

    splits = tree.refine(test)
    balanced = tree.balance()
    nodes, tets, tri, tri_refs, _ = tree.tetrahedra()
    n_octree = len(tets)
    t_tree = time.perf_counter() - t0
    required: list[int] = _matching_refs(axes)
    on_faces = _on_faces(axes, lower, upper)
    moved = added = 0
    sharp = sharp_pieces = None
    # several phases, or pieces: cut in turn without remeshing, then remeshed once
    once = pieces is not None or several or at_once
    t1 = time.perf_counter()
    if (several or at_once) and pieces is None:
        # the lines where three phases meet kept as ridges, the faces of the box cut by MMG
        # alike on opposite sides (see morphogen.mesh.conforming._mmg_pieces)
        data, _, _ = _mmg_pieces(
            exe, nodes, tets, tri, tri_refs, [_Cut(k, level(k)) for k in keys], rest,
            (lower, upper), axes, on_faces, reach, size=size, min_size=hmin,
            hausdorff=hausd, gradation=gradation, workdir=workdir, timeout=timeout,
        )  # fmt: skip
    elif pieces is not None:
        # the box cut by the surface of each piece in turn, the lines where two cuts meet
        # kept as ridges (see morphogen.mesh.conforming._mmg_pieces)
        try:
            data, sharp, sharp_pieces = _mmg_pieces(
                exe, nodes, tets, tri, tri_refs, pieces, rest, (lower, upper), axes, on_faces,
                reach, size=size, min_size=hmin, hausdorff=hausd, gradation=gradation,
                workdir=workdir, timeout=timeout,
            )  # fmt: skip
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            if sharp_edges:
                raise
            log.warning("adaptive mesh: the sharp edges could not be followed (%s); they are "
                        "rounded instead", str(error).splitlines()[0])  # fmt: skip
            return _adaptive_mesh(
                micro, size=size, hausdorff=hausdorff, min_size=min_size, gradation=gradation,
                phases=phases, order=order, periodic=periodic, lipschitz=lipschitz,
                project=project, remove_islands=remove_islands, ridges=ridges,
                sharp_edges=False, retries=retries, executable=executable, workdir=workdir,
                timeout=timeout, attempt=attempt, at_once=at_once,
            )  # fmt: skip
    else:
        try:
            cut = _cut_two(
                exe, nodes, tets, tri, tri_refs, level(keys[0]), (keys[0], rest), axes,
                required, on_faces, lower, upper, tol, hausd, hmin, size, gradation, two,
                ridges, remove_islands, workdir, timeout,
            )  # fmt: skip
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            # MMG could not remesh along the cut: cut first, then remeshed once
            log.warning("adaptive mesh: MMG failed on the cut of the phases (%s); cut without "
                        "remeshing, then remeshed once", str(error).splitlines()[0])  # fmt: skip
            once = True
            data, _, _ = _mmg_pieces(
                exe, nodes, tets, tri, tri_refs, [_Cut(k, level(k)) for k in keys], rest,
                (lower, upper), axes, on_faces, reach, size=size, min_size=hmin,
                hausdorff=hausd, gradation=gradation, workdir=workdir, timeout=timeout,
            )  # fmt: skip
        else:
            nodes, tets, phase, moved, added = cut
    if once:
        nodes, tets = data["Vertices"], data["Tetrahedra"]
        for d in range(3):  # the nodes of the faces exactly on them
            nodes[np.abs(nodes[:, d] - lower[d]) <= tol, d] = lower[d]
            nodes[np.abs(nodes[:, d] - upper[d]) <= tol, d] = upper[d]
        phase = data["Tetrahedra_refs"] - REF_PHASE
    t_cut = time.perf_counter() - t1
    xyz, elements = nodes, tets
    flip = tet_volumes(xyz, elements) < 0
    elements[flip] = elements[flip][:, [0, 2, 1, 3]]
    used, inverse = np.unique(elements, return_inverse=True)
    xyz, elements = xyz[used], inverse.reshape(elements.shape)
    if sharp is not None:  # the sharp edges, between the nodes of the mesh
        numbers = np.full(len(nodes), -1, dtype=np.int64)
        numbers[used] = np.arange(len(used))
        there = (numbers[sharp] >= 0).all(axis=1)
        sharp, sharp_pieces = numbers[sharp[there]], sharp_pieces[there]
    projected, largest = 0, 0.0
    if project:
        # the nodes lie within a few Hausdorff distances of their surfaces: the functions
        # there (and as far as the moves allowed, a quarter of the size)
        near = min(reach, max(lip, 1.0) * max(8.0 * hausd, 2.0 * hmin, 0.3 * size))
        projected, largest, xyz = _onto_surfaces(
            xyz, elements, phase, present, two, between, once, pieces, sharp, sharp_pieces,
            near, on_faces, axes, lower, upper, tol, hausd, hmin, size,
        )  # fmt: skip
    metadata = {
        "mesher": "adaptive level set (octree, mmg3d -ls)",
        "size": size,
        "hausdorff": hausd,
        "min_size": hmin,
        "lipschitz": lip,
        "level_sets": [int(k) for k in keys],
        "octree_splits": splits,
        "octree_balance_splits": balanced,
        "octree_elements": n_octree,
        "face_nodes_moved": moved,
        "face_nodes_added": added,
        "projected_nodes": projected,
        "largest_projection": largest,
        "phases": phases if isinstance(phases, str) else [int(k) for k in phases],
        "matching_faces": [d for d in range(3) if axes[d]],
        "time_octree_s": round(t_tree, 3),
        "time_mmg_s": round(t_cut, 3),
    }
    if pieces is not None:
        metadata["sharp_edges"] = {"pieces": len(pieces), "edges": len(sharp), "depth": hausd}
    elif once:
        metadata["cut_once"] = True  # several phases, or the cut of two failed
    mesh = TetMesh(
        nodes=xyz,
        elements=elements,
        phase=phase,
        lower=lower,
        upper=upper,
        periodic=all(axes),
        metadata=metadata,
        sharp=sharp,
        sharp_pieces=sharp_pieces,
    )
    if any(axes) and not mesh.matching_faces():
        if not once:  # the faces cut exactly before MMG: cut by MMG alike, as on the grid
            log.info("adaptive mesh: the faces do not match; the phases cut at once")
            return _adaptive_mesh(
                micro, size=size, hausdorff=hausdorff, min_size=min_size, gradation=gradation,
                phases=phases, order=order, periodic=periodic, lipschitz=lipschitz,
                project=project, remove_islands=remove_islands, ridges=ridges,
                sharp_edges=sharp_edges, retries=retries, executable=executable,
                workdir=workdir, timeout=timeout, attempt=attempt, at_once=True,
            )  # fmt: skip
        if attempt < retries:
            log.info("adaptive mesh: the faces do not match; again on another octree")
            return _adaptive_mesh(
                micro,
                size=size,
                hausdorff=hausdorff,
                min_size=min_size,
                gradation=gradation,
                phases=phases,
                order=order,
                periodic=periodic,
                lipschitz=lipschitz,
                project=project,
                remove_islands=remove_islands,
                ridges=ridges,
                sharp_edges=sharp_edges,
                retries=retries,
                executable=executable,
                workdir=workdir,
                timeout=timeout,
                attempt=attempt + 1,
                at_once=at_once,
            )
        log.warning("adaptive mesh: the opposite faces of the box do not match")
    if two != "all" and once:  # the cuts made at once keep every phase
        keep = [1] if two == "solid" else [0]
    if order == 2:  # curved on the interfaces (the nodes of the sharp edges on both)
        from morphogen.mesh.curved import curved_mesh

        mesh = curved_mesh(mesh, micro) if project else mesh.quadratic()
    if keep is not None:
        mesh = mesh.subset(keep)
    log.info("adaptive mesh: %d elements, %d nodes (octree of %d elements, %.1f s)",
             mesh.n_elements, mesh.n_nodes, n_octree, time.perf_counter() - t0)  # fmt: skip
    return mesh


def _cut_two(exe, nodes, tets, tri, tri_refs, function, phases, axes, required, on_faces,
             lower, upper, tol, hausd, hmin, size, gradation, two, ridges, remove_islands,
             workdir, timeout):  # fmt: skip
    """The octree of a structure of two phases cut by the zero of its level set
    ``function`` (negative in the first of the two ``phases``) and remeshed by MMG in one
    pass, the matching faces cut exactly before (see the module). Returns the nodes, the
    tetrahedra, their phases, and the numbers of nodes of the faces moved and added."""
    moved = added = 0
    face_points = np.zeros((0, 3))
    if any(axes):  # the matching faces cut exactly, alike: MMG keeps them
        group = _groups(nodes, lower, upper, axes, tol)
        nodes, tets, tri, tri_refs, values, _, moved, added = _cut_faces(
            function, nodes, tets, tri, tri_refs, required, group, zero=1e-6 * hausd
        )
        # the edges between nodes of the faces through the box (near its edges) that the
        # level set crosses: split at its zeros (MMG frees the nodes of the faces at the
        # ends of a cut edge, and remeshes the faces there)
        face = on_faces(nodes)
        nodes, tets, more, done = _split_across(function, nodes, tets, tri, face, [values], 0)
        face = np.concatenate([face, np.zeros(more, dtype=bool)])
        added += more
        # the values of the faces as the cut left them: the pass of MMG takes them there
        face_points, face_values = nodes[face], done[0][face]
    values = np.asarray(function(nodes), dtype=float).copy()
    if len(face_points):
        dist, where = cKDTree(face_points).query(nodes, distance_upper_bound=10.0 * tol)
        hit = dist <= 10.0 * tol
        values[hit] = face_values[where[hit]]
    tiny = _OFF_NODES * hausd  # no interface through a node, but on the cut faces
    trace = (values == 0.0) & on_faces(nodes)
    values = np.where(np.abs(values) < tiny, np.where(values < 0, -tiny, tiny), values)
    values[trace] = 0.0
    if trace.any():  # the traces of the faces are nodes: the interface manifold there
        _manifold_balls(tets, values, np.flatnonzero(trace), tiny)
    sizing = ["-hmax", f"{size:.10g}", "-hmin", f"{hmin:.10g}", "-hausd", f"{hausd:.10g}",
              "-hgrad", f"{gradation:.10g}", "-v", "-1"]  # fmt: skip
    if two != "all":  # MMG keeps one subdomain
        sizing += ["-nsd", str(REF_INSIDE if two == "solid" else REF_OUTSIDE)]
    if not ridges:  # the edges of the box given, no detection by angles (see the module)
        sizing += ["-nr"]
    fixed = np.isin(tri_refs, required)
    if remove_islands and not fixed.any():
        sizing += ["-rmc", f"{float(remove_islands):.10g}"]
    edges, corners = _box_ridges(nodes, tri, lower, upper, tol)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        medit.write_mesh(folder / "cut.mesh", nodes, tets, tri,
                         tet_refs=np.ones(len(tets), dtype=np.int64), tri_refs=tri_refs,
                         required_triangles=np.flatnonzero(fixed) if fixed.any() else None,
                         required_vertices=np.flatnonzero(on_faces(nodes)) if fixed.any()
                         else None, edges=edges, ridges=np.arange(len(edges)),
                         corners=corners)  # fmt: skip
        medit.write_sol(folder / "cut.sol", values)
        data = _run_mmg(exe, folder / "cut.mesh", folder / "cut.sol", folder / "cut.o.mesh",
                        ["-ls", "0", *sizing], timeout)  # fmt: skip
    nodes, tets, tet_refs = data["Vertices"], data["Tetrahedra"], data["Tetrahedra_refs"]
    for d in range(3):  # the nodes of the faces exactly on them
        nodes[np.abs(nodes[:, d] - lower[d]) <= tol, d] = lower[d]
        nodes[np.abs(nodes[:, d] - upper[d]) <= tol, d] = upper[d]
    phase = np.where(tet_refs == REF_INSIDE, phases[0], phases[1])
    return nodes, tets, phase, moved, added


def _onto_surfaces(xyz, elements, phase, present, two, between, once, pieces, sharp,
                   sharp_pieces, reach, on_faces, axes, lower, upper, tol, hausd, hmin,
                   size):  # fmt: skip
    """The nodes of the interfaces moved onto their exact surfaces (:func:`_project`):
    the nodes between two phases (or on the boundary of the one that MMG kept) onto the
    interface of the two; those of the sharp edges onto the surfaces of the pieces that
    meet there. Those where three phases meet stay, as do the nodes of the matching faces
    cut exactly before MMG (they lie on the interfaces already); the nodes of the faces
    that the cuts made at ``once`` left move within their faces, alike on the opposite
    ones. Returns the number of nodes moved, the largest move and the nodes."""
    count = np.zeros((len(xyz), len(present)), dtype=bool)
    slot = {k: i for i, k in enumerate(present)}
    for k, i in slot.items():
        count[np.unique(elements[phase == k]), i] = True
    n_phases = count.sum(axis=1)
    if two != "all" and not once:  # MMG kept one phase: the interface is a boundary
        boundary = np.zeros(len(xyz), dtype=bool)
        boundary[np.unique(_free_faces(elements))] = True
        boundary &= ~_on_box(xyz, lower, upper, tol) | on_faces(xyz)
        n_phases = np.where(boundary, 2, n_phases)
        count[boundary, slot[0] if two == "solid" else slot[1]] = True
    candidates = n_phases == 2
    if not once:  # the matching faces: cut exactly already
        candidates &= ~on_faces(xyz)
    groups = {}
    if sharp is not None and len(sharp):
        groups = _sharp_groups(sharp, sharp_pieces, len(xyz))
        for nodes_at in groups.values():
            candidates[nodes_at] = False
    group = _groups(xyz, lower, upper, axes, tol) if any(axes) else None
    step = min(hausd, hmin) / 10.0
    projected, largest = 0, 0.0
    for a, b in itertools.combinations(present, 2):  # onto the interface of the two
        moving = np.flatnonzero(candidates & count[:, slot[a]] & count[:, slot[b]])
        if not moving.size:
            continue
        xyz, done, most = _project(between(b, a, reach), xyz, elements, moving, step=step,
                                   limit=0.25 * size, lower=lower, upper=upper, tol=tol,
                                   group=group)  # fmt: skip
        projected, largest = projected + done, max(largest, most)
    for meeting, moving in groups.items():  # onto the surfaces of the pieces that meet
        surfaces = [pieces[k] for k in meeting]

        def where(points, surfaces=surfaces):
            return np.stack([piece.at(points, reach) for piece in surfaces], axis=1)

        xyz, done, most = _project(where, xyz, elements, moving, step=step, limit=0.25 * size,
                                   lower=lower, upper=upper, tol=tol, group=group)  # fmt: skip
        projected, largest = projected + done, max(largest, most)
    return projected, largest, xyz


#: Barycentric coordinates of the nodes of a 10-node tetrahedron (corners, then the middles
#: of the edges in the order of :data:`morphogen.mesh.tetra.TET_EDGES`)
_P2_POINTS = np.vstack([np.eye(4), 0.5 * (np.eye(4)[[0, 1, 0, 0, 1, 2]] +
                                         np.eye(4)[[1, 2, 2, 3, 3, 3]])])  # fmt: skip


def _p2_jacobians(nodes, elements, points=_P2_POINTS) -> np.ndarray:
    """The determinants of the Jacobians of 10-node tetrahedra at points (barycentric
    coordinates; default: their ten nodes)."""
    from morphogen.mesh.tetra import TET_EDGES

    grad_l = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    x = nodes[elements]  # (m, 10, 3)
    out = np.empty((len(elements), len(points)))
    for p, lam in enumerate(points):
        grads = [(4.0 * lam[i] - 1.0) * grad_l[i] for i in range(4)]
        grads += [4.0 * (lam[j] * grad_l[i] + lam[i] * grad_l[j]) for i, j in TET_EDGES]
        jac = np.einsum("mad,ae->mde", x, np.array(grads))
        out[:, p] = np.linalg.det(jac)
    return out


def _free_faces(elements: np.ndarray) -> np.ndarray:
    """The triangles of tetrahedra that belong to a single one."""
    faces = np.concatenate([elements[:, [0, 2, 1]], elements[:, [0, 1, 3]],
                            elements[:, [1, 2, 3]], elements[:, [0, 3, 2]]])  # fmt: skip
    key = np.sort(faces, axis=1)
    _, first, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[first[counts == 1]]


def _on_box(points, lower, upper, tol) -> np.ndarray:
    """Whether points lie on a face of the box."""
    return np.any((np.abs(points - lower) <= tol) | (np.abs(points - upper) <= tol), axis=1)


# -- the route of each microstructure --------------------------------------------------------


def exact_route(micro, grains: bool | None = None) -> str:
    """How :func:`exact_mesh` meshes a microstructure: ``"cells"`` (the exact cells of
    Voronoi and Laguerre tessellations, grain by grain), ``"pieces"`` (exact convex pieces:
    faceted microstructures), ``"level_set"`` (the adaptive mesh of the exact point-wise
    functions of its phases, :meth:`~morphogen.models.base.Microstructure.
    phase_level_sets_at`: the other structures, of two phases or more, the tessellations
    with a grain-boundary phase, and those meshed by their phases, ``grains`` false), or
    ``""`` (none: the structures whose functions are those of grids only, and those whose
    cells carry crystal orientations but tessellations, whose meshes come from their voxel
    images)."""
    from morphogen.models.implicit import point_functions
    from morphogen.models.tessellation import Tessellation

    if isinstance(micro, Tessellation):
        if not micro.boundary_thickness > 0 and grains is not False:
            return "cells"
        return "level_set" if point_functions(micro) else ""
    if callable(getattr(micro, "convex_pieces", None)):
        return "pieces"
    if getattr(micro, "orientations", None) is not None:
        return ""  # crystals turned by their cells: a mesh by cell (their images)
    return "level_set" if point_functions(micro) else ""


def exact_mesh(
    micro,
    *,
    size: float,
    min_size: float | None = None,
    hausdorff: float | None = None,
    gradation: float | None = None,
    phases="all",
    order: int = 1,
    periodic=None,
    grains: bool | None = None,
    sharp_edges: bool | None = None,
    executable: str | None = None,
    workdir=None,
    timeout: float | None = None,
    **options,
) -> TetMesh:
    """Conforming tetrahedral mesh of the exact geometry of a microstructure, without a
    background grid (see the module): the exact cells of tessellations
    (:func:`morphogen.mesh.polycrystal.polycrystal_mesh`), the exact convex pieces of
    faceted microstructures (:func:`polyhedral_mesh`), or the adaptive mesh of the exact
    functions of the phases of the others (:func:`adaptive_mesh`); ``options`` go to the
    mesher of the route (:func:`exact_route`, which ``grains`` false turns from the cells of
    a tessellation to the functions of its phases). ``phases``: ``"all"``, ``"solid"``,
    ``"matrix"`` or the phases to keep (a list). ``hausdorff``: the distance of the curved
    interfaces of level sets to their facets; the planar facets of cells and pieces are
    kept, whatever it is. ``sharp_edges``: the sharp edges of the smooth pieces of level
    sets (:func:`adaptive_mesh`); the edges of cells and convex pieces are those of their
    facets, kept."""
    route = exact_route(micro, grains)
    keep = None
    if not isinstance(phases, str) and route in ("cells", "pieces"):  # kept from the whole
        keep, phases = sorted({int(k) for k in phases}), "all"
    common = dict(size=size, min_size=min_size, phases=phases, order=order,
                  executable=executable, workdir=workdir, timeout=timeout)  # fmt: skip
    if gradation is not None:
        common["gradation"] = gradation
    if route == "cells":
        from morphogen.mesh.polycrystal import polycrystal_mesh

        mesh = polycrystal_mesh(micro, periodic=periodic, exact=True, **common, **options)
    elif route == "pieces":
        if periodic is not None and np.ndim(periodic) == 0:
            options["periodic"] = bool(periodic)
        mesh = polyhedral_mesh(micro, **common, **options)
    elif route == "level_set":
        common["hausdorff"] = hausdorff
        return adaptive_mesh(micro, periodic=periodic, sharp_edges=sharp_edges, **common,
                             **options)  # fmt: skip
    else:
        raise TypeError(
            f"{type(micro).__name__}: neither exact cells, convex pieces nor point-wise "
            "functions of its phases, so no exact mesh (mesh its voxel image instead)"
        )
    if keep is not None:
        unknown = sorted(set(keep) - {int(k) for k in np.unique(mesh.phase)})
        if unknown:
            raise ValueError(f"phases: {unknown} are not phases of this structure")
        mesh = mesh.subset(keep)
    return mesh
