"""Voxels inside closed triangle surfaces: the parity of the crossings of the columns of
voxels along z (Numba)."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def surface_parity(vertices, faces, shape):
    """The voxels inside a closed surface on a local grid of ``shape`` voxels: the voxels
    of a column (i, j) whose centres lie between pairs of crossings of the column with the
    faces. ``vertices`` are in the continuous indices of the grid (the centre of voxel i at
    i); each face crossed by a column toggles the first voxel above the crossing, and the
    toggles summed modulo 2 up the column give the inside. The columns pass a little off
    the centres (1e-6 and 1.3e-6 of a voxel), so that a column never meets an edge or a
    vertex of the surface."""
    nx, ny, nz = shape
    toggles = np.zeros((nx, ny, nz + 1), dtype=np.uint8)
    for f in range(faces.shape[0]):
        a = vertices[faces[f, 0]]
        b = vertices[faces[f, 1]]
        c = vertices[faces[f, 2]]
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-14:  # a face seen edge-on: no column crosses it
            continue
        i0 = max(0, math.ceil(min(a[0], b[0], c[0])))
        i1 = min(nx - 1, math.floor(max(a[0], b[0], c[0])))
        j0 = max(0, math.ceil(min(a[1], b[1], c[1])))
        j1 = min(ny - 1, math.floor(max(a[1], b[1], c[1])))
        for i in range(i0, i1 + 1):
            px = i + 1e-6
            for j in range(j0, j1 + 1):
                py = j + 1.3e-6
                l1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / d
                l2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / d
                l3 = 1.0 - l1 - l2
                if l1 < 0.0 or l2 < 0.0 or l3 < 0.0:
                    continue
                w = l1 * a[2] + l2 * b[2] + l3 * c[2]
                k = math.ceil(w)
                if k < 0:
                    k = 0
                elif k > nz:
                    k = nz
                toggles[i, j, k] ^= 1
    out = np.zeros((nx, ny, nz), dtype=np.uint8)
    for i in range(nx):
        for j in range(ny):
            s = 0
            for k in range(nz):
                s ^= toggles[i, j, k]
                out[i, j, k] = s
    return out
