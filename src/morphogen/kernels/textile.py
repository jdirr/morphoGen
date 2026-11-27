"""Yarns of woven textiles: sections swept along polylines (Numba kernels).

A yarn is a polyline of segments; each segment carries the frame of its two ends (the
width axis ``n`` and the thickness axis ``b``, in the plane normal to the centreline), the
half sizes of its section at its two ends (``half_w`` and ``half_t``, one row per segment:
the start and the end; equal for the yarns of constant sections), its label and its
phase. The segments are registered in the
uniform grid of cells of :mod:`morphogen.kernels.raster` as capsules whose radius is the
larger half size plus the band of the functions. The segments of a yarn are consecutive,
and the candidates of a cell are listed in increasing order: they come in runs, one per
yarn. At a point, the segment of each run closest to it gives the closest point of the
centreline; the coordinates ``(u, v)`` of the point in the frame there and the half sizes
of the section there (both interpolated between the ends of the segment) give the
function of the section, about the signed distance to its boundary
(:func:`section_function`). The deepest yarn (the smallest
function) takes the point; the matrix takes the points that no yarn holds.

Loops over voxels are parallel over x-slices; each voxel is written by a single thread.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from morphogen.kernels.raster import _cell_of

#: The shapes of the sections
ELLIPSE, LENTICULAR, POWER = 0, 1, 2


@njit(cache=True, inline="always")
def section_function(u, v, a, c, shape, q):
    """About the signed distance from ``(u, v)`` to the boundary of the section of half
    width ``a`` (along ``u``) and half thickness ``c`` (along ``v``), negative inside:
    the first-order distance ``(g - 1) / |grad g|`` of the gauge ``g`` of the ellipse
    ``(u/a)^2 + (v/c)^2 <= 1`` and of the power ellipse ``|u/a|^q + |v/c|^q <= 1`` (exact on
    the boundary and along the axes); for the lens, the intersection of two discs of
    radius ``R = (a^2 + c^2) / (2 c)`` centred ``R - c`` from the middle along ``v``, the
    larger of the signed distances to the discs (exact inside, and outside but near the
    two corners)."""
    if shape == LENTICULAR:
        r = 0.5 * (a * a + c * c) / c
        e = r - c
        d1 = math.sqrt(u * u + (v + e) * (v + e)) - r
        d2 = math.sqrt(u * u + (v - e) * (v - e)) - r
        return max(d1, d2)
    au = abs(u) / a
    av = abs(v) / c
    if shape == ELLIPSE:
        g = math.sqrt(au * au + av * av)
        if g < 1e-12:
            return -min(a, c)
        gu = au / a
        gv = av / c
        return (g - 1.0) * g / math.sqrt(gu * gu + gv * gv)
    s = au**q + av**q
    if s < 1e-300:
        return -min(a, c)
    g = s ** (1.0 / q)
    f = g ** (1.0 - q)
    gu = f * au ** (q - 1.0) / a
    gv = f * av ** (q - 1.0) / c
    norm = math.sqrt(gu * gu + gv * gv)
    if norm < 1e-300:
        return -min(a, c)
    return (g - 1.0) / norm


@njit(cache=True)
def _evaluate(
    x,
    y,
    z,
    cell,
    offsets,
    items,
    start,
    end,
    na,
    ba,
    nb,
    bb,
    yarn,
    half_w,
    half_t,
    reach,
    label,
    phase,
    shape,
    q,
    depth,
):
    """The yarns at one point: ``depth[p - 1]`` lowered to the smallest function of the
    yarns of phase ``p`` (left as given without one), and the function, label and phase of
    the deepest yarn and the number of yarns that hold the point."""
    best_f = np.inf
    best_label = -1
    best_phase = 0
    count = 0
    k = offsets[cell]
    stop = offsets[cell + 1]
    while k < stop:
        current = yarn[items[k]]
        chosen = -1
        chosen_d2 = np.inf
        chosen_t = 0.0
        rx = 0.0
        ry = 0.0
        rz = 0.0
        while k < stop and yarn[items[k]] == current:
            s = items[k]
            ax, ay, az = start[s, 0], start[s, 1], start[s, 2]
            dx, dy, dz = end[s, 0] - ax, end[s, 1] - ay, end[s, 2] - az
            dd = dx * dx + dy * dy + dz * dz
            t = 0.0
            if dd > 0.0:
                t = ((x - ax) * dx + (y - ay) * dy + (z - az) * dz) / dd
                t = min(max(t, 0.0), 1.0)
            px = x - (ax + t * dx)
            py = y - (ay + t * dy)
            pz = z - (az + t * dz)
            d2 = px * px + py * py + pz * pz
            if d2 < chosen_d2:
                chosen_d2 = d2
                chosen = s
                chosen_t = t
                rx, ry, rz = px, py, pz
            k += 1
        if chosen < 0 or chosen_d2 > reach[chosen] * reach[chosen]:
            continue
        t = chosen_t
        n0 = (1.0 - t) * na[chosen, 0] + t * nb[chosen, 0]
        n1 = (1.0 - t) * na[chosen, 1] + t * nb[chosen, 1]
        n2 = (1.0 - t) * na[chosen, 2] + t * nb[chosen, 2]
        b0 = (1.0 - t) * ba[chosen, 0] + t * bb[chosen, 0]
        b1 = (1.0 - t) * ba[chosen, 1] + t * bb[chosen, 1]
        b2 = (1.0 - t) * ba[chosen, 2] + t * bb[chosen, 2]
        nn = math.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
        nbb = math.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
        u = (rx * n0 + ry * n1 + rz * n2) / nn
        v = (rx * b0 + ry * b1 + rz * b2) / nbb
        a = half_w[chosen, 0] + t * (half_w[chosen, 1] - half_w[chosen, 0])
        c = half_t[chosen, 0] + t * (half_t[chosen, 1] - half_t[chosen, 0])
        f = section_function(u, v, a, c, shape, q)
        p = phase[chosen]
        if f < depth[p - 1]:
            depth[p - 1] = f
        if f <= 0.0:
            count += 1
        if f < best_f:
            best_f = f
            best_label = label[chosen]
            best_phase = p
    return best_f, best_label, best_phase, count


@njit(parallel=True, cache=True)
def yarn_grid(
    offsets,
    items,
    start,
    end,
    na,
    ba,
    nb,
    bb,
    yarn,
    half_w,
    half_t,
    reach,
    label,
    phase,
    shape,
    q,
    band,
    matrix_label,
    cell_lo,
    cell_h,
    cell_n,
    grid_lo,
    spacing,
    grid_shape,
    out_phase,
    out_label,
    out_depth,
    out_count,
):
    """The yarns at the voxel centres of a grid: the phase (``out_phase``, uint8) and the
    label (``out_label``, int32; ``matrix_label`` in the matrix) of the deepest yarn, the
    smallest function of the yarns of each phase (``out_depth[p - 1]``, float32, ``band``
    without one) and the number of yarns that hold the voxel (``out_count``, uint8). The
    outputs of size 0 are not written."""
    n_phases = out_depth.shape[0] if out_depth.size else 3
    for i in prange(grid_shape[0]):
        depth = np.empty(max(n_phases, 1))
        x = grid_lo[0] + (i + 0.5) * spacing[0]
        for j in range(grid_shape[1]):
            y = grid_lo[1] + (j + 0.5) * spacing[1]
            for k in range(grid_shape[2]):
                z = grid_lo[2] + (k + 0.5) * spacing[2]
                depth[:] = band
                cell = _cell_of(x, y, z, cell_lo, cell_h, cell_n)
                f, lab, ph, cnt = _evaluate(
                    x, y, z, cell, offsets, items, start, end, na, ba, nb, bb, yarn,
                    half_w, half_t, reach, label, phase, shape, q, depth,
                )  # fmt: skip
                if f > 0.0:
                    lab = matrix_label
                    ph = 0
                if out_phase.size:
                    out_phase[i, j, k] = ph
                if out_label.size:
                    out_label[i, j, k] = lab
                if out_depth.size:
                    for p in range(n_phases):
                        out_depth[p, i, j, k] = min(depth[p], band)
                if out_count.size:
                    out_count[i, j, k] = min(cnt, 255)


@njit(parallel=True, cache=True)
def yarn_points(
    offsets,
    items,
    start,
    end,
    na,
    ba,
    nb,
    bb,
    yarn,
    half_w,
    half_t,
    reach,
    label,
    phase,
    shape,
    q,
    band,
    matrix_label,
    cell_lo,
    cell_h,
    cell_n,
    pts,
    out_phase,
    out_label,
    out_depth,
    out_count,
):
    """The same at points (rows of ``pts``): ``out_depth`` of shape (phases, points)."""
    n_phases = out_depth.shape[0] if out_depth.size else 3
    for m in prange(pts.shape[0]):
        depth = np.empty(max(n_phases, 1))
        depth[:] = band
        x, y, z = pts[m, 0], pts[m, 1], pts[m, 2]
        cell = _cell_of(x, y, z, cell_lo, cell_h, cell_n)
        f, lab, ph, cnt = _evaluate(
            x, y, z, cell, offsets, items, start, end, na, ba, nb, bb, yarn, half_w,
            half_t, reach, label, phase, shape, q, depth,
        )  # fmt: skip
        if f > 0.0:
            lab = matrix_label
            ph = 0
        out_phase[m] = ph
        out_label[m] = lab
        if out_depth.size:
            for p in range(n_phases):
                out_depth[p, m] = min(depth[p], band)
        out_count[m] = cnt


@njit(parallel=True, cache=True)
def yarn_count(
    offsets,
    items,
    start,
    end,
    na,
    ba,
    nb,
    bb,
    yarn,
    half_w,
    half_t,
    reach,
    label,
    phase,
    shape,
    q,
    target,
    cell_lo,
    cell_h,
    cell_n,
    grid_lo,
    spacing,
    grid_shape,
    ss,
    out,
):
    """Number of the ``ss**3`` sub-samples of each voxel that the yarns of phase
    ``target`` take (the deepest yarn there being of that phase)."""
    inv = 1.0 / ss
    for i in prange(grid_shape[0]):
        depth = np.empty(3)
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                cnt = 0
                for a in range(ss):
                    x = grid_lo[0] + (i + (a + 0.5) * inv) * spacing[0]
                    for b in range(ss):
                        y = grid_lo[1] + (j + (b + 0.5) * inv) * spacing[1]
                        for c in range(ss):
                            z = grid_lo[2] + (k + (c + 0.5) * inv) * spacing[2]
                            depth[:] = 0.0
                            cell = _cell_of(x, y, z, cell_lo, cell_h, cell_n)
                            f, _, ph, _ = _evaluate(
                                x, y, z, cell, offsets, items, start, end, na, ba, nb, bb,
                                yarn, half_w, half_t, reach, label, phase, shape, q, depth,
                            )  # fmt: skip
                            if f <= 0.0 and ph == target:
                                cnt += 1
                out[i, j, k] = cnt
