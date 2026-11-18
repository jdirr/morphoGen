"""Evaluation of triply periodic minimal surface (TPMS) level-set functions."""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

GYROID, SCHWARZ_P, SCHWARZ_D, NEOVIUS = 0, 1, 2, 3
SURFACES = {"gyroid": GYROID, "schwarz_p": SCHWARZ_P, "schwarz_d": SCHWARZ_D, "neovius": NEOVIUS}


@njit(cache=True, inline="always")
def tpms_value(code, X, Y, Z):
    """phi(X, Y, Z) with X = 2 pi x / a (and similarly Y, Z)."""
    if code == GYROID:
        return math.sin(X) * math.cos(Y) + math.sin(Y) * math.cos(Z) + math.sin(Z) * math.cos(X)
    if code == SCHWARZ_P:
        return math.cos(X) + math.cos(Y) + math.cos(Z)
    if code == SCHWARZ_D:
        sx, sy, sz = math.sin(X), math.sin(Y), math.sin(Z)
        cx, cy, cz = math.cos(X), math.cos(Y), math.cos(Z)
        return sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
    # Neovius
    cx, cy, cz = math.cos(X), math.cos(Y), math.cos(Z)
    return 3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz


@njit(cache=True, inline="always")
def tpms_gradient_norm(code, X, Y, Z, kx, ky, kz):
    """|grad phi| with respect to physical coordinates (k = 2 pi / a)."""
    sx, sy, sz = math.sin(X), math.sin(Y), math.sin(Z)
    cx, cy, cz = math.cos(X), math.cos(Y), math.cos(Z)
    if code == GYROID:
        gx = cx * cy - sz * sx
        gy = -sx * sy + cy * cz
        gz = -sy * sz + cz * cx
    elif code == SCHWARZ_P:
        gx, gy, gz = -sx, -sy, -sz
    elif code == SCHWARZ_D:
        gx = cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz
        gy = sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz
        gz = sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz
    else:
        gx = -3.0 * sx - 4.0 * sx * cy * cz
        gy = -3.0 * sy - 4.0 * cx * sy * cz
        gz = -3.0 * sz - 4.0 * cx * cy * sz
    return math.sqrt((kx * gx) ** 2 + (ky * gy) ** 2 + (kz * gz) ** 2)


@njit(cache=True, inline="always")
def _signed(code, sheet, level, X, Y, Z):
    """Negative inside the solid: phi - t (network) or |phi| - t (sheet)."""
    v = tpms_value(code, X, Y, Z)
    if sheet:
        return abs(v) - level
    return v - level


@njit(parallel=True, cache=True)
def tpms_count(code, sheet, level, k, offset, grid_lo, spacing, shape, ss, out):
    inv = 1.0 / ss
    for i in prange(shape[0]):
        for j in range(shape[1]):
            for kk in range(shape[2]):
                cnt = 0
                for a in range(ss):
                    X = k[0] * (grid_lo[0] + (i + (a + 0.5) * inv) * spacing[0] - offset[0])
                    for b in range(ss):
                        Y = k[1] * (grid_lo[1] + (j + (b + 0.5) * inv) * spacing[1] - offset[1])
                        for c in range(ss):
                            Z = k[2] * (
                                grid_lo[2] + (kk + (c + 0.5) * inv) * spacing[2] - offset[2]
                            )
                            if _signed(code, sheet, level, X, Y, Z) <= 0.0:
                                cnt += 1
                out[i, j, kk] = cnt


@njit(parallel=True, cache=True)
def tpms_level_set(code, sheet, level, k, offset, grid_lo, spacing, shape, out):
    """First-order distance estimate (phi - t) / |grad phi| at voxel centres."""
    for i in prange(shape[0]):
        X = k[0] * (grid_lo[0] + (i + 0.5) * spacing[0] - offset[0])
        for j in range(shape[1]):
            Y = k[1] * (grid_lo[1] + (j + 0.5) * spacing[1] - offset[1])
            for kk in range(shape[2]):
                Z = k[2] * (grid_lo[2] + (kk + 0.5) * spacing[2] - offset[2])
                g = tpms_gradient_norm(code, X, Y, Z, k[0], k[1], k[2])
                out[i, j, kk] = _signed(code, sheet, level, X, Y, Z) / max(g, 1e-12)


@njit(parallel=True, cache=True)
def tpms_level_points(code, sheet, level, k, offset, pts, out):
    """The first-order distance of :func:`tpms_level_set` at points."""
    for p in prange(pts.shape[0]):
        X = k[0] * (pts[p, 0] - offset[0])
        Y = k[1] * (pts[p, 1] - offset[1])
        Z = k[2] * (pts[p, 2] - offset[2])
        g = tpms_gradient_norm(code, X, Y, Z, k[0], k[1], k[2])
        out[p] = _signed(code, sheet, level, X, Y, Z) / max(g, 1e-12)


@njit(parallel=True, cache=True)
def fourier_points(modes, re, im, lengths, origin, pts, out):
    """``Re sum_m (re_m + i im_m) exp(2 i pi m.(x - origin) / lengths)`` at points: the
    powers of ``exp(2 i pi (x_a - origin_a) / lengths_a)`` by recurrence, then a product
    per mode."""
    n_max = np.zeros(3, dtype=np.int64)
    for j in range(modes.shape[0]):
        for a in range(3):
            n_max[a] = max(n_max[a], abs(modes[j, a]))
    width = 2 * n_max.max() + 1
    for p in prange(pts.shape[0]):
        cr = np.empty((3, width))
        ci = np.empty((3, width))
        for a in range(3):
            t = 2.0 * math.pi * (pts[p, a] - origin[a]) / lengths[a]
            c, s = math.cos(t), math.sin(t)
            m0 = n_max[a]
            cr[a, m0], ci[a, m0] = 1.0, 0.0
            for q in range(1, m0 + 1):
                r0, i0 = cr[a, m0 + q - 1], ci[a, m0 + q - 1]
                cr[a, m0 + q], ci[a, m0 + q] = r0 * c - i0 * s, r0 * s + i0 * c
                cr[a, m0 - q], ci[a, m0 - q] = cr[a, m0 + q], -ci[a, m0 + q]
        total = 0.0
        for j in range(modes.shape[0]):
            i0, i1, i2 = modes[j, 0] + n_max[0], modes[j, 1] + n_max[1], modes[j, 2] + n_max[2]
            xr, xi = cr[0, i0], ci[0, i0]
            yr, yi = cr[1, i1], ci[1, i1]
            zr, zi = cr[2, i2], ci[2, i2]
            ar, ai = xr * yr - xi * yi, xr * yi + xi * yr
            br, bi = ar * zr - ai * zi, ar * zi + ai * zr
            total += re[j] * br - im[j] * bi
        out[p] = total


@njit(parallel=True, cache=True)
def tpms_points(code, sheet, level, k, offset, pts, out):
    for p in prange(pts.shape[0]):
        X = k[0] * (pts[p, 0] - offset[0])
        Y = k[1] * (pts[p, 1] - offset[1])
        Z = k[2] * (pts[p, 2] - offset[2])
        out[p] = _signed(code, sheet, level, X, Y, Z) <= 0.0
