"""Rasterization of unions of spheres and segments (Numba kernels).

Primitives are registered in a uniform grid of cells (compressed sparse rows: the
primitives of cell ``c`` are ``items[offsets[c]:offsets[c + 1]]``). A primitive is
registered in every cell that may contain a point closer to it than its radius plus
an extra reach (the level-set band), so that each query point only tests the
candidates of its own cell. Loops over voxels are parallel over x-slices; each voxel
is written by a single thread.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _axis_cell(v, lo, h, n):
    c = math.floor((v - lo) / h)
    if c < 0:
        return 0
    if c >= n:
        return n - 1
    return c


@njit(cache=True, inline="always")
def _cell_of(x, y, z, cell_lo, cell_h, cell_n):
    ci = _axis_cell(x, cell_lo[0], cell_h[0], cell_n[0])
    cj = _axis_cell(y, cell_lo[1], cell_h[1], cell_n[1])
    ck = _axis_cell(z, cell_lo[2], cell_h[2], cell_n[2])
    return (ci * cell_n[1] + cj) * cell_n[2] + ck


@njit(cache=True, inline="always")
def _segment_param(px, py, pz, ax, ay, az, bx, by, bz):
    """Parameter t of the projection of p on the line (a, b), unclamped."""
    ux, uy, uz = bx - ax, by - ay, bz - az
    uu = ux * ux + uy * uy + uz * uz
    if uu == 0.0:
        return 0.0
    return ((px - ax) * ux + (py - ay) * uy + (pz - az) * uz) / uu


@njit(cache=True, inline="always")
def _dist2_to_point_on_axis(px, py, pz, ax, ay, az, bx, by, bz, t):
    qx = ax + t * (bx - ax) - px
    qy = ay + t * (by - ay) - py
    qz = az + t * (bz - az) - pz
    return qx * qx + qy * qy + qz * qz


@njit(cache=True)
def _register_spheres(centers, radii, extra, cell_lo, cell_h, cell_n, counts, offsets, items, fill):
    n = centers.shape[0]
    for s in range(n):
        reach = radii[s] + extra
        r2 = reach * reach
        cx, cy, cz = centers[s, 0], centers[s, 1], centers[s, 2]
        i0 = _axis_cell(cx - reach, cell_lo[0], cell_h[0], cell_n[0])
        i1 = _axis_cell(cx + reach, cell_lo[0], cell_h[0], cell_n[0])
        j0 = _axis_cell(cy - reach, cell_lo[1], cell_h[1], cell_n[1])
        j1 = _axis_cell(cy + reach, cell_lo[1], cell_h[1], cell_n[1])
        k0 = _axis_cell(cz - reach, cell_lo[2], cell_h[2], cell_n[2])
        k1 = _axis_cell(cz + reach, cell_lo[2], cell_h[2], cell_n[2])
        for i in range(i0, i1 + 1):
            bx0 = cell_lo[0] + i * cell_h[0]
            dx = max(bx0 - cx, 0.0, cx - (bx0 + cell_h[0]))
            for j in range(j0, j1 + 1):
                by0 = cell_lo[1] + j * cell_h[1]
                dy = max(by0 - cy, 0.0, cy - (by0 + cell_h[1]))
                for k in range(k0, k1 + 1):
                    bz0 = cell_lo[2] + k * cell_h[2]
                    dz = max(bz0 - cz, 0.0, cz - (bz0 + cell_h[2]))
                    if dx * dx + dy * dy + dz * dz <= r2:
                        cell = (i * cell_n[1] + j) * cell_n[2] + k
                        if fill:
                            items[offsets[cell] + counts[cell]] = s
                        counts[cell] += 1


@njit(cache=True)
def _register_segments(a, b, radii, extra, cell_lo, cell_h, cell_n, counts, offsets, items, fill):
    n = a.shape[0]
    half_diag = 0.5 * math.sqrt(cell_h[0] ** 2 + cell_h[1] ** 2 + cell_h[2] ** 2)
    for s in range(n):
        reach = radii[s] + extra
        lim = reach + half_diag
        lim2 = lim * lim
        ax, ay, az = a[s, 0], a[s, 1], a[s, 2]
        bx, by, bz = b[s, 0], b[s, 1], b[s, 2]
        i0 = _axis_cell(min(ax, bx) - reach, cell_lo[0], cell_h[0], cell_n[0])
        i1 = _axis_cell(max(ax, bx) + reach, cell_lo[0], cell_h[0], cell_n[0])
        j0 = _axis_cell(min(ay, by) - reach, cell_lo[1], cell_h[1], cell_n[1])
        j1 = _axis_cell(max(ay, by) + reach, cell_lo[1], cell_h[1], cell_n[1])
        k0 = _axis_cell(min(az, bz) - reach, cell_lo[2], cell_h[2], cell_n[2])
        k1 = _axis_cell(max(az, bz) + reach, cell_lo[2], cell_h[2], cell_n[2])
        for i in range(i0, i1 + 1):
            px = cell_lo[0] + (i + 0.5) * cell_h[0]
            for j in range(j0, j1 + 1):
                py = cell_lo[1] + (j + 0.5) * cell_h[1]
                for k in range(k0, k1 + 1):
                    pz = cell_lo[2] + (k + 0.5) * cell_h[2]
                    t = _segment_param(px, py, pz, ax, ay, az, bx, by, bz)
                    t = min(max(t, 0.0), 1.0)
                    if _dist2_to_point_on_axis(px, py, pz, ax, ay, az, bx, by, bz, t) <= lim2:
                        cell = (i * cell_n[1] + j) * cell_n[2] + k
                        if fill:
                            items[offsets[cell] + counts[cell]] = s
                        counts[cell] += 1


def build_cells(kind, arrays, extra, cell_lo, cell_h, cell_n):
    """CSR cell lists for ``kind`` in {"spheres", "segments"}."""
    ncell = int(np.prod(cell_n))
    counts = np.zeros(ncell, dtype=np.int64)
    offsets = np.zeros(ncell + 1, dtype=np.int64)
    dummy = np.zeros(0, dtype=np.int64)
    register = _register_spheres if kind == "spheres" else _register_segments
    register(*arrays, float(extra), cell_lo, cell_h, cell_n, counts, offsets, dummy, False)
    np.cumsum(counts, out=offsets[1:])
    items = np.empty(offsets[-1], dtype=np.int64)
    counts[:] = 0
    register(*arrays, float(extra), cell_lo, cell_h, cell_n, counts, offsets, items, True)
    return offsets, items


@njit(cache=True, inline="always")
def _inside(x, y, z, cell, sc, sr, so, si, ga, gb, gr, round_caps, go, gi):
    for t in range(so[cell], so[cell + 1]):
        s = si[t]
        dx = x - sc[s, 0]
        dy = y - sc[s, 1]
        dz = z - sc[s, 2]
        if dx * dx + dy * dy + dz * dz <= sr[s] * sr[s]:
            return True
    for t in range(go[cell], go[cell + 1]):
        s = gi[t]
        ax, ay, az = ga[s, 0], ga[s, 1], ga[s, 2]
        bx, by, bz = gb[s, 0], gb[s, 1], gb[s, 2]
        u = _segment_param(x, y, z, ax, ay, az, bx, by, bz)
        if round_caps:
            u = min(max(u, 0.0), 1.0)
        elif u < 0.0 or u > 1.0:
            continue
        if _dist2_to_point_on_axis(x, y, z, ax, ay, az, bx, by, bz, u) <= gr[s] * gr[s]:
            return True
    return False


@njit(cache=True, inline="always")
def _distance(x, y, z, cell, band, sc, sr, so, si, ga, gb, gr, round_caps, go, gi):
    """Signed distance to the union, clipped to [-band, band] (exact outside)."""
    d = band
    for t in range(so[cell], so[cell + 1]):
        s = si[t]
        dx = x - sc[s, 0]
        dy = y - sc[s, 1]
        dz = z - sc[s, 2]
        v = math.sqrt(dx * dx + dy * dy + dz * dz) - sr[s]
        if v < d:
            d = v
    for t in range(go[cell], go[cell + 1]):
        s = gi[t]
        ax, ay, az = ga[s, 0], ga[s, 1], ga[s, 2]
        bx, by, bz = gb[s, 0], gb[s, 1], gb[s, 2]
        u = _segment_param(x, y, z, ax, ay, az, bx, by, bz)
        if round_caps:
            uc = min(max(u, 0.0), 1.0)
            v = math.sqrt(_dist2_to_point_on_axis(x, y, z, ax, ay, az, bx, by, bz, uc)) - gr[s]
        else:
            length = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
            radial = math.sqrt(_dist2_to_point_on_axis(x, y, z, ax, ay, az, bx, by, bz, u)) - gr[s]
            axial = (abs(u - 0.5) - 0.5) * length
            ox = max(radial, 0.0)
            oy = max(axial, 0.0)
            v = math.sqrt(ox * ox + oy * oy) + min(max(radial, axial), 0.0)
        if v < d:
            d = v
    if d < -band:
        d = -band
    return d


@njit(parallel=True, cache=True)
def rasterize_count(
    sc,
    sr,
    so,
    si,
    ga,
    gb,
    gr,
    round_caps,
    go,
    gi,
    cell_lo,
    cell_h,
    cell_n,
    grid_lo,
    spacing,
    shape,
    ss,
    out,
):
    """Number of the ``ss**3`` sub-samples of each voxel lying in the union."""
    inv = 1.0 / ss
    for i in prange(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                cnt = 0
                for a in range(ss):
                    x = grid_lo[0] + (i + (a + 0.5) * inv) * spacing[0]
                    for b in range(ss):
                        y = grid_lo[1] + (j + (b + 0.5) * inv) * spacing[1]
                        for c in range(ss):
                            z = grid_lo[2] + (k + (c + 0.5) * inv) * spacing[2]
                            cell = _cell_of(x, y, z, cell_lo, cell_h, cell_n)
                            if _inside(
                                x, y, z, cell, sc, sr, so, si, ga, gb, gr, round_caps, go, gi
                            ):
                                cnt += 1
                out[i, j, k] = cnt


@njit(parallel=True, cache=True)
def rasterize_distance(
    sc,
    sr,
    so,
    si,
    ga,
    gb,
    gr,
    round_caps,
    go,
    gi,
    cell_lo,
    cell_h,
    cell_n,
    grid_lo,
    spacing,
    shape,
    band,
    out,
):
    """Signed distance at voxel centres, clipped to [-band, band]."""
    for i in prange(shape[0]):
        x = grid_lo[0] + (i + 0.5) * spacing[0]
        for j in range(shape[1]):
            y = grid_lo[1] + (j + 0.5) * spacing[1]
            for k in range(shape[2]):
                z = grid_lo[2] + (k + 0.5) * spacing[2]
                cell = _cell_of(x, y, z, cell_lo, cell_h, cell_n)
                out[i, j, k] = _distance(
                    x, y, z, cell, band, sc, sr, so, si, ga, gb, gr, round_caps, go, gi
                )


@njit(parallel=True, cache=True)
def points_inside(
    sc, sr, so, si, ga, gb, gr, round_caps, go, gi, cell_lo, cell_h, cell_n, pts, out
):
    for p in prange(pts.shape[0]):
        x, y, z = pts[p, 0], pts[p, 1], pts[p, 2]
        cell = _cell_of(x, y, z, cell_lo, cell_h, cell_n)
        out[p] = _inside(x, y, z, cell, sc, sr, so, si, ga, gb, gr, round_caps, go, gi)


@njit(parallel=True, cache=True)
def points_distance(
    sc, sr, so, si, ga, gb, gr, round_caps, go, gi, cell_lo, cell_h, cell_n, pts, band, out
):
    for p in prange(pts.shape[0]):
        x, y, z = pts[p, 0], pts[p, 1], pts[p, 2]
        cell = _cell_of(x, y, z, cell_lo, cell_h, cell_n)
        out[p] = _distance(x, y, z, cell, band, sc, sr, so, si, ga, gb, gr, round_caps, go, gi)
