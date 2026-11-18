"""Tessellations of a plane: the cells of the layered and prismatic bio-inspired models
(the tablets of nacre, honeycombs, the cells of wood).

A point ``x`` of the plane belongs to the cell of its nearest germ ``g_i``. Its distance
to the side shared with the cell of ``g_j`` is ``(|x - g_j|^2 - |x - g_i|^2) /
(2 |g_j - g_i|)``, and the smallest of these is its distance to the boundary of its
(convex) cell; the germs whose side is closer than ``r`` lie within ``|x - g_i| + 2 r``
of the point, so that the nearest germs suffice (as for the tessellations of space).
On a periodic rectangle the germs are taken once: a cell is not bounded by its own
images, so that a rectangle one cell wide makes the cells continuous across it.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from morphogen import settings
from morphogen.kernels.rsa import random_sequential_addition

#: Fraction of the plane covered by discs at the jamming limit of their random sequential
#: addition (Feder 1980): the densest random packing that RSA reaches.
RSA_JAMMING_2D = 0.547

#: Points per k-d tree query (bounds the memory of large grids).
_CHUNK = 1 << 18


class PlanarCells:
    """Voronoi cells of germs in a rectangle of the plane, periodic or not (the germs of
    a window larger than the rectangle, then)."""

    def __init__(self, germs, lengths, periodic: bool, origin=(0.0, 0.0)) -> None:
        self.germs = np.asarray(germs, dtype=float).reshape(-1, 2)
        self.lengths = np.asarray(lengths, dtype=float).reshape(2)
        self.origin = np.asarray(origin, dtype=float).reshape(2)
        self.periodic = bool(periodic)
        if not len(self.germs):
            raise ValueError("planar cells: no germs")
        self._tree: cKDTree | None = None

    def _local(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=float).reshape(-1, 2) - self.origin
        if self.periodic:
            p = np.mod(p, self.lengths)
            p[p >= self.lengths] = 0.0
        return p

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            box = self.lengths if self.periodic else None
            self._tree = cKDTree(self._local(self.germs), boxsize=box)
        return self._tree

    def locate(self, points, reach: float = math.inf) -> tuple[np.ndarray, np.ndarray]:
        """Cell (germ index) of each point and its distance to the boundary of the cell,
        exact up to ``reach`` and at least ``reach`` beyond (``inf``: a single cell)."""
        q = self._local(points)
        n = len(self.germs)
        labels = np.zeros(len(q), dtype=np.int64)
        out = np.full(len(q), np.inf)
        if n == 1 or not len(q):
            return labels, out
        germs = self._local(self.germs)
        for start in range(0, len(q), _CHUNK):
            todo, k = np.arange(start, min(len(q), start + _CHUNK)), min(n, 12)
            while todo.size:
                dist, idx = self.tree.query(q[todo], k=k, workers=settings.workers())
                labels[todo] = idx[:, 0]
                v = germs[idx] - q[todo, None, :]  # the germs seen from the points
                if self.periodic:
                    v -= self.lengths * np.round(v / self.lengths)
                gap = np.linalg.norm(v[:, 1:] - v[:, :1], axis=2)
                d = (dist[:, 1:] ** 2 - dist[:, :1] ** 2) / (2.0 * np.maximum(gap, 1e-300))
                out[todo] = d.min(axis=1)
                # the farthest germ found must lie beyond l_i + 2 min(d, reach)
                bound = dist[:, 0] + 2.0 * np.minimum(out[todo], reach)
                again = (dist[:, -1] <= bound) & (k < n)
                todo, k = todo[again], min(n, 2 * k)
        return labels, out


def hardcore_germs(n: int, lengths, distance: float, rng) -> np.ndarray:
    """``n`` points of the periodic rectangle ``[0, lengths)``, at least ``distance``
    apart (random sequential addition of discs; uniform points for no distance)."""
    lengths = np.asarray(lengths, dtype=float).reshape(2)
    if n <= 0:
        return np.empty((0, 2))
    if not distance > 0:
        return rng.random((n, 2)) * lengths
    thin = 1e-9 * float(lengths.min())  # a slab: the distances are those in the plane
    box = np.array([lengths[0], lengths[1], thin])
    pos = random_sequential_addition(
        np.full(n, 0.5 * distance), box, 0.0, rng, max_attempts=4000 * n + 100_000
    )
    return pos[:, :2]


def jamming_distance(n: int, area: float) -> float:
    """Distance between ``n`` points of an ``area`` at the jamming limit of their
    random sequential addition."""
    return math.sqrt(4.0 * RSA_JAMMING_2D * area / (math.pi * max(n, 1)))


def plane_axes(axis: int) -> tuple[int, int]:
    """The two axes orthogonal to ``axis``, in increasing order."""
    b, c = (a for a in range(3) if a != axis)
    return b, c


def pixel_centres(grid, axes: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """Centres of the voxels of a grid projected on the plane of ``axes``: an array of
    points (row-major over the two axes) and the shape of the pixel image."""
    b, c = axes
    u = grid.domain.lower[b] + (np.arange(grid.shape[b]) + 0.5) * grid.spacing[b]
    v = grid.domain.lower[c] + (np.arange(grid.shape[c]) + 0.5) * grid.spacing[c]
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.column_stack([uu.ravel(), vv.ravel()]), (len(u), len(v))


def axis_centres(grid, axis: int) -> np.ndarray:
    """Coordinates of the voxel centres along ``axis``."""
    return grid.domain.lower[axis] + (np.arange(grid.shape[axis]) + 0.5) * grid.spacing[axis]
