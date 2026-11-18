"""First covering leaf of every voxel centre (perfect simulation of dead leaves): spheres,
or spheroids of a given aspect ratio about their axes; and the field of transparent
leaves, accumulated in their order of fall."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _offset(index, a, s, centers, lower, spacing, length, periodic):
    """Coordinate of a voxel centre minus that of a leaf centre (nearest image)."""
    d = lower[a] + (index + 0.5) * spacing[a] - centers[s, a]
    if periodic:
        d -= length * math.floor(d / length + 0.5)
    return d


@njit(cache=True)
def first_cover(centers, radii, axes, aspect, first, lower, spacing, shape, periodic, out):
    """Give each voxel of ``out`` (-1: not covered yet) the index (from ``first``) of the
    first leaf containing its centre; returns the number of voxels still uncovered. The
    leaves are spheres of radii ``radii`` (``aspect`` 1), or spheroids of equatorial
    semi-axes ``radii`` and polar semi-axes ``aspect * radii`` along the unit ``axes``."""
    n = np.array([shape[0], shape[1], shape[2]], dtype=np.int64)
    length = np.array([n[0] * spacing[0], n[1] * spacing[1], n[2] * spacing[2]])
    sphere = aspect == 1.0
    left = 0
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                if out[i, j, k] < 0:
                    left += 1
    lo = np.empty(3, dtype=np.int64)
    hi = np.empty(3, dtype=np.int64)
    half = np.empty(3)
    for s in range(len(radii)):
        if left == 0:
            break
        r = radii[s]
        r2 = r * r
        c2 = (aspect * r) ** 2
        for a in range(3):
            # the half extent of the leaf along the axis a of the grid
            u2 = 0.0 if sphere else axes[s, a] ** 2
            half[a] = r if sphere else math.sqrt(r2 * (1.0 - u2) + c2 * u2)
            lo[a] = math.ceil((centers[s, a] - half[a] - lower[a]) / spacing[a] - 0.5)
            hi[a] = math.floor((centers[s, a] + half[a] - lower[a]) / spacing[a] - 0.5)
            if not periodic:
                lo[a] = max(lo[a], 0)
                hi[a] = min(hi[a], n[a] - 1)
            elif hi[a] - lo[a] >= n[a]:  # wider than the box: each voxel once
                hi[a] = lo[a] + n[a] - 1
        for i in range(lo[0], hi[0] + 1):
            dx = _offset(i, 0, s, centers, lower, spacing, length[0], periodic)
            ii = i % n[0] if periodic else i
            for j in range(lo[1], hi[1] + 1):
                dy = _offset(j, 1, s, centers, lower, spacing, length[1], periodic)
                d2 = dx * dx + dy * dy
                if sphere and d2 > r2:
                    continue
                jj = j % n[1] if periodic else j
                for k in range(lo[2], hi[2] + 1):
                    dz = _offset(k, 2, s, centers, lower, spacing, length[2], periodic)
                    if sphere:
                        if d2 + dz * dz > r2:
                            continue
                    else:
                        t = dx * axes[s, 0] + dy * axes[s, 1] + dz * axes[s, 2]
                        if (d2 + dz * dz - t * t) / r2 + t * t / c2 > 1.0:
                            continue
                    kk = k % n[2] if periodic else k
                    if out[ii, jj, kk] < 0:
                        out[ii, jj, kk] = first + s
                        left -= 1
    return left


@njit(cache=True)
def transparent_cover(centers, radii, values, alpha, lower, spacing, shape, periodic, z, w, tol):
    """Transparent leaves (spheres) in their order of fall backwards (the first on top):
    each leaf covering a voxel centre whose weight ``w`` exceeds ``tol`` adds ``alpha * w``
    times its value to the field ``z`` and multiplies the weight by ``1 - alpha``; returns
    the number of voxels whose weight still exceeds ``tol``."""
    n = np.array([shape[0], shape[1], shape[2]], dtype=np.int64)
    length = np.array([n[0] * spacing[0], n[1] * spacing[1], n[2] * spacing[2]])
    keep = 1.0 - alpha
    left = 0
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                if w[i, j, k] > tol:
                    left += 1
    lo = np.empty(3, dtype=np.int64)
    hi = np.empty(3, dtype=np.int64)
    for s in range(len(radii)):
        if left == 0:
            break
        r = radii[s]
        r2 = r * r
        for a in range(3):
            lo[a] = math.ceil((centers[s, a] - r - lower[a]) / spacing[a] - 0.5)
            hi[a] = math.floor((centers[s, a] + r - lower[a]) / spacing[a] - 0.5)
            if not periodic:
                lo[a] = max(lo[a], 0)
                hi[a] = min(hi[a], n[a] - 1)
            elif hi[a] - lo[a] >= n[a]:
                hi[a] = lo[a] + n[a] - 1
        for i in range(lo[0], hi[0] + 1):
            dx = _offset(i, 0, s, centers, lower, spacing, length[0], periodic)
            ii = i % n[0] if periodic else i
            for j in range(lo[1], hi[1] + 1):
                dy = _offset(j, 1, s, centers, lower, spacing, length[1], periodic)
                d2 = dx * dx + dy * dy
                if d2 > r2:
                    continue
                jj = j % n[1] if periodic else j
                for k in range(lo[2], hi[2] + 1):
                    dz = _offset(k, 2, s, centers, lower, spacing, length[2], periodic)
                    if d2 + dz * dz > r2:
                        continue
                    kk = k % n[2] if periodic else k
                    weight = w[ii, jj, kk]
                    if weight > tol:
                        z[ii, jj, kk] += alpha * weight * values[s]
                        w[ii, jj, kk] = weight * keep
                        if weight * keep <= tol:
                            left -= 1
    return left
