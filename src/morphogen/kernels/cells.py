"""Implicit functions of unions of convex cells: the cells of a tiling that have a phase,
each given by the planes of its faces."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def cells_implicit(
    points,
    cell,
    indptr,
    indices,
    centres,
    kind,
    planes,
    counts,
    inside,
    lengths,
    dims,
    band,
    out,
):
    """The implicit function of the union of the cells that are ``inside`` at ``points``
    (n, 3), each in the cell ``cell`` of a tiling whose cells around each (``indptr``,
    ``indices``: those sharing a vertex with it) are convex: ``planes[kind[c]]`` holds
    ``counts[kind[c]]`` planes ``(n, h)`` of the faces of the cell ``c`` from its centre
    (``n . x <= h`` inside; ``dims``: the coordinates that count, 2 for the prisms through
    a plate). The gauge of a cell at a point is its largest distance to these planes,
    negative inside the cell; the function is the smallest gauge of the cells inside
    minus the smallest of the others, among the cell of the point and those around it:
    it vanishes on the faces between a cell inside and a cell outside, near which half of
    it is the distance to the face; that half is returned, clipped to ``band``.
    ``lengths``: the periods of a
    periodic tiling along the axes (0: not periodic), for the nearest image of the
    cells."""
    for p in prange(points.shape[0]):
        c0 = cell[p]
        g_in = band
        g_out = band
        for q in range(indptr[c0] - 1, indptr[c0 + 1]):
            c = c0 if q < indptr[c0] else indices[q]
            rel = np.empty(3)
            for a in range(3):
                d = points[p, a] - centres[c, a]
                if lengths[a] > 0.0:
                    d -= lengths[a] * np.floor(d / lengths[a] + 0.5)
                rel[a] = d
            t = kind[c]
            g = -np.inf
            for f in range(counts[t]):
                v = -planes[t, f, 3]
                for a in range(dims):
                    v += planes[t, f, a] * rel[a]
                if v > g:
                    g = v
            if inside[c]:
                if g < g_in:
                    g_in = g
            elif g < g_out:
                g_out = g
        value = 0.5 * (g_in - g_out)
        if value > band:
            value = band
        elif value < -band:
            value = -band
        out[p] = value
