"""Exact triangular meshes of prismatic cells in their plane: honeycombs.

The walls of :class:`~morphogen.models.honeycomb.PrismaticCells` are the points of each
cell nearer to its sides than half its wall: the lumen of a cell is its Voronoi polygon
offset inward by half its wall — the intersection of its half planes moved in by it, an
exact convex polygon —, and the walls are the rest of the box. Prisms along ``z`` do not
vary along it: the mesh of their plane is the structure itself, whose stiffness and
conductivity in space the finite elements of the plane give in generalized plane strain
(:mod:`morphogen.homogenization.plane`). The edges of the lumens are graded across the
walls (:data:`~morphogen.mesh.triangles.GAP_ELEMENTS` elements across the thinnest) and
from their gaps to the sides of the box; ``mmg2d`` meshes the walls and the lumens within
them, the sides of a periodic box matching."""

from __future__ import annotations

import itertools
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from morphogen.io import medit
from morphogen.mesh.triangles import (
    GAP_ELEMENTS,
    SIDE_REFS,
    TriMesh,
    _clip,
    _kept,
    _marched,
    _mmg_args,
    _PointSizes,
    _rectangle,
    _run,
    _shoelace,
)

#: The reference of the edges between the walls and the lumens
REF_LUMEN = 10


def cell_lumens(micro) -> list[tuple[int, np.ndarray]] | None:
    """The lumens of the prismatic cells of a honeycomb whose prisms run along ``z``, in its
    box: each cell (the Voronoi polygon of its germ) offset inward by half its wall, cut by
    the box — in a periodic box, it and its translates —, as ``(cell, polygon)``, the
    vertices counterclockwise; None for another structure (the radial files of wood, prisms
    along another axis)."""
    from morphogen.models.honeycomb import PrismaticCells

    if not isinstance(micro, PrismaticCells) or micro.axis != 2 or micro.files is not None:
        return None
    domain = micro.domain
    lower = np.asarray(domain.lower[:2], dtype=float)
    upper = np.asarray(domain.upper[:2], dtype=float)
    lengths = upper - lower
    germs = np.asarray(micro.germs, dtype=float)
    periodic = bool(domain.periodic)
    shifts = (np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float)
              if periodic else np.zeros((1, 2)))  # fmt: skip
    every = (germs[None, :, :] + shifts[:, None, :] * lengths).reshape(-1, 2)
    tree = cKDTree(every)
    big = _rectangle(np.minimum(lower, germs.min(axis=0)) - lengths,
                     np.maximum(upper, germs.max(axis=0)) + lengths)  # fmt: skip
    out = []
    for i, c in enumerate(germs):
        k = min(len(every), 24)
        while True:  # the Voronoi cell: the half planes of the nearest germs, enough of them
            _, near = tree.query(c, k=k)
            cell = big.copy()
            for j in np.atleast_1d(near):
                d = every[j] - c
                if np.any(np.abs(d) > 0):
                    cell = _clip(cell, d, 0.5 * (every[j] @ every[j] - c @ c))
            reach = float(np.max(np.linalg.norm(cell - c, axis=1))) if len(cell) else 0.0
            far = float(np.linalg.norm(every[np.atleast_1d(near)[-1]] - c))
            if far >= 2.0 * reach or k >= len(every):
                break
            k = min(len(every), 2 * k)
        half = 0.5 * float(micro.walls[i])
        lumen = cell
        for a, b in zip(cell, np.roll(cell, -1, axis=0), strict=True):  # (each side moved in)
            d = b - a
            length = float(np.linalg.norm(d))
            if length > 0.0:
                normal = np.array([d[1], -d[0]]) / length  # (outward: counterclockwise)
                lumen = _clip(lumen, normal, float(normal @ a) - half)
        if len(lumen) < 3:
            continue  # (a cell all wall)
        for shift in shifts:
            piece = lumen + shift * lengths
            for axis in (0, 1):
                e = np.zeros(2)
                e[axis] = 1.0
                piece = _clip(piece, e, upper[axis])
                piece = _clip(piece, -e, -lower[axis])
            if len(piece) >= 3 and abs(_shoelace(piece)) > 1e-12 * float(np.prod(lengths)):
                out.append((i, piece))
    return out


def cells_mesh(micro, *, size: float, min_size: float | None = None,
               hausdorff: float | None = None, gradation: float = 2.0, z: float | None = None,
               phases="all", order: int = 1, curved: bool | None = None,
               periodic: bool | None = None, executable: str | None = None, workdir=None,
               timeout: float | None = None) -> TriMesh | None:  # fmt: skip
    """The triangular mesh of the prismatic cells of a honeycomb in their plane (see the
    module): the lumens (phase 0) exact polygons (:func:`cell_lumens`), the walls (phase 1)
    the rest of the box, the edges of the lumens graded across the walls and from their gaps
    to the sides (at least ``min_size``, by default a 64th of ``size``), the sides of a
    periodic box matching; ``mmg2d`` meshes the regions within them. Six-node elements are
    straight, as the edges (``curved`` has nothing to curve). ``z`` names the plane of the
    mesh (any: the prisms do not vary along their axis). None for another structure, or
    where ``mmg2d`` fails twice."""
    from morphogen.mesh.conforming import mmg_executable

    lumens = cell_lumens(micro)
    if lumens is None:
        return None
    size = float(size)
    if not size > 0:
        raise ValueError("size: must be positive")
    if order not in (1, 2):
        raise ValueError("order: 1 or 2")
    domain = micro.domain
    lower = np.asarray(domain.lower[:2], dtype=float)
    upper = np.asarray(domain.upper[:2], dtype=float)
    lengths = upper - lower
    periodic = bool(domain.periodic) if periodic is None else bool(periodic)
    if periodic and not domain.periodic:
        raise ValueError("matching periodic sides need a periodic structure")
    z = float(domain.lower[2] + 0.5 * domain.lengths[2]) if z is None else float(z)
    tol = 1e-9 * float(lengths.max())
    floor = size / 64.0 if min_size is None else float(min_size)
    pieces = []
    for cell, polygon in lumens:  # (the vertices on the sides exactly on them)
        polygon = polygon.copy()
        for axis in (0, 1):
            for value in (lower[axis], upper[axis]):
                polygon[np.abs(polygon[:, axis] - value) <= tol, axis] = value
        pieces.append((cell, polygon))

    def on_sides(p) -> list[tuple[int, int]]:
        return [(axis, side) for (axis, side) in SIDE_REFS
                if p[axis] == (lower, upper)[side][axis]]  # fmt: skip

    # 1. the size field: across the thinnest wall, and from the gaps of the edges of the
    #    lumens to the sides that they do not touch
    across = min(size, max(float(np.min(micro.walls)) / GAP_ELEMENTS, floor))
    points, h = [], []
    cuts: dict[tuple[int, int], list[float]] = {key: [] for key in SIDE_REFS}
    for _, polygon in pieces:
        for p in polygon:
            for axis, side in on_sides(p):
                cuts[(axis, side)].append(float(p[1 - axis]))
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            n = max(2, int(np.ceil(4.0 * np.linalg.norm(b - a) / size)) + 1)
            samples = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
            sized = np.full(n, across)
            touched = set(on_sides(a)) | set(on_sides(b))
            for axis, side in SIDE_REFS:
                if (axis, side) not in touched:
                    gap = np.abs(samples[:, axis] - (lower, upper)[side][axis])
                    sized = np.minimum(sized, np.maximum(gap / GAP_ELEMENTS, floor))
            points.append(samples)
            h.append(sized)
    if not points:  # (no lumen: all walls)
        points, h = [lower[None]], [np.array([size])]
    field = _PointSizes(np.concatenate(points), np.concatenate(h), lower, upper, size, periodic)
    # 2. the nodes of the sides: their corners, the vertices of the lumens on them, the
    #    points graded between; those of the upper sides of a periodic box the translates
    xy: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    refs: list[int] = []
    corner: dict[tuple[int, int], int] = {}
    side_nodes: dict[tuple[int, int], tuple[np.ndarray, list[int]]] = {}

    def corner_node(i: int, j: int) -> int:
        if (i, j) not in corner:
            corner[(i, j)] = len(xy)
            xy.append(np.array([(lower, upper)[i][0], (lower, upper)[j][1]]))
        return corner[(i, j)]

    for axis in (0, 1):
        other = 1 - axis
        for side in (0, 1):
            value = (lower, upper)[side][axis]
            ts = cuts[(axis, side)] + (cuts[(axis, 1 - side)] if periodic else [])
            ts = np.sort(np.asarray(ts, dtype=float))
            ts = ts[np.concatenate([[True], np.diff(ts) > tol])] if len(ts) else ts
            ts = ts[(ts > lower[other] + tol) & (ts < upper[other] - tol)]
            ts = np.concatenate([[lower[other]], ts, [upper[other]]])

            def along(s, t0, axis=axis, other=other, value=value):
                p = np.empty((len(s), 2))
                p[:, axis], p[:, other] = value, t0 + s
                return p

            if periodic and side == 1:
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

    def node(p) -> int:
        """The node of a vertex of a lumen: of its side, or new."""
        for axis, side in on_sides(p):
            chain, ids = side_nodes[(axis, side)]
            k = int(np.argmin(np.abs(chain - p[1 - axis])))
            if abs(chain[k] - p[1 - axis]) <= 1e3 * tol:
                return ids[k]
        xy.append(np.asarray(p, dtype=float))
        return len(xy) - 1

    # 3. the edges of the lumens off the sides, graded
    for _, polygon in pieces:
        ids = [node(p) for p in polygon]
        for k in range(len(polygon)):
            a, b = polygon[k], polygon[(k + 1) % len(polygon)]
            if set(on_sides(a)) & set(on_sides(b)):
                continue  # (along a side: the edges of the side)
            length = float(np.linalg.norm(b - a))

            def at(s, a=a, b=b, length=length):
                return a + (np.asarray(s)[:, None] / length) * (b - a)

            chain = [ids[k]]
            for p in at(_marched(length, at, field)[1:-1]):
                chain.append(len(xy))
                xy.append(p)
            chain.append(ids[(k + 1) % len(polygon)])
            edges += list(itertools.pairwise(chain))
            refs += [REF_LUMEN] * (len(chain) - 1)
    xy = np.asarray(xy)
    edges = np.asarray(edges, dtype=np.int64)
    refs = np.asarray(refs, dtype=np.int64)
    shortest = float(np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1).min())
    exe = mmg_executable(executable, "mmg2d")
    args = [*_mmg_args(size, 0.5 * shortest, hausdorff, gradation), "-nr", "-nosurf"]
    sequence = np.arange(len(xy))
    for attempt in range(2):  # (mmg2d inserts the nodes in their order: once more in another)
        with tempfile.TemporaryDirectory(dir=workdir) as tmp:
            folder = Path(tmp)
            place = np.empty_like(sequence)
            place[sequence] = np.arange(len(sequence))
            medit.write_mesh(folder / "cells.mesh", xy[sequence], None,
                             np.zeros((0, 3), dtype=np.int64), edges=place[edges],
                             edge_refs=refs, required_vertices=np.arange(len(xy)),
                             required_edges=np.arange(len(edges)))  # fmt: skip
            try:
                out = _run(exe, folder / "cells.mesh", None, folder / "out.mesh", args, timeout)
                break
            except RuntimeError as err:
                if attempt:
                    first = str(err).splitlines()[0].rstrip(":") if str(err) else "failed"
                    warnings.warn(f"cells mesh: {first}; the cells are left to the level sets",
                                  RuntimeWarning, stacklevel=2)  # fmt: skip
                    return None
                sequence = np.random.default_rng(0).permutation(len(xy))
    nodes, tri = out["Vertices"], out["Triangles"]
    centroids = nodes[tri].mean(axis=1)
    phase = np.ones(len(tri), dtype=np.int64)  # the walls, but in the lumens
    near = cKDTree(centroids)
    for _, polygon in pieces:
        middle = polygon.mean(axis=0)
        k = np.asarray(near.query_ball_point(middle, float(np.linalg.norm(polygon - middle,
                                                                          axis=1).max())),
                       dtype=np.int64)  # fmt: skip
        if not len(k):
            continue
        a, run = polygon[None], (np.roll(polygon, -1, axis=0) - polygon)[None]
        to = centroids[k, None, :] - a  # (the centroids seen from the vertices)
        cross = run[..., 0] * to[..., 1] - run[..., 1] * to[..., 0]
        phase[k[(cross >= 0.0).all(axis=1)]] = 0
    mesh = TriMesh(nodes, tri, phase, lower, upper, periodic,
                   {"mesher": "exact cells (mmg2d)", "size": size, "cells": len(micro.germs),
                    "lumens": len(pieces), "z": z}, z=z,
                   thickness=float(domain.lengths[2]))  # fmt: skip
    mesh = _kept(mesh, phases)
    return mesh if order == 1 else mesh.quadratic()


__all__ = ["REF_LUMEN", "cell_lumens", "cells_mesh"]
