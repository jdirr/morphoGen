"""Random sequential addition (RSA) of non-overlapping spheres in a periodic box."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def rsa_insert(radii, gap, box, cell_h, cell_n, head, nxt, pos, placed, candidates):
    """Try ``candidates`` in turn for the sphere ``placed``, then the next ones.

    Cells form a periodic grid with linked lists (``head[cell]`` is the last sphere
    inserted in the cell, ``nxt[s]`` the previous one). Cells are at least as large as
    the largest interaction distance, so only the 27 neighbouring cells are searched.
    Returns the new number of placed spheres and the number of candidates used.
    """
    n_total = radii.shape[0]
    rmax = 0.0
    for s in range(n_total):
        rmax = max(rmax, radii[s])
    used = 0
    for c in range(candidates.shape[0]):
        if placed >= n_total:
            break
        used += 1
        x, y, z = candidates[c, 0], candidates[c, 1], candidates[c, 2]
        r = radii[placed]
        ci = min(int(x / cell_h[0]), cell_n[0] - 1)
        cj = min(int(y / cell_h[1]), cell_n[1] - 1)
        ck = min(int(z / cell_h[2]), cell_n[2] - 1)
        ok = True
        # with fewer than 3 cells along an axis, visit each cell once
        ni = min(3, cell_n[0])
        nj = min(3, cell_n[1])
        nk = min(3, cell_n[2])
        for di in range(ni):
            ii = (ci + di - 1) % cell_n[0] if ni == 3 else di
            for dj in range(nj):
                jj = (cj + dj - 1) % cell_n[1] if nj == 3 else dj
                for dk in range(nk):
                    kk = (ck + dk - 1) % cell_n[2] if nk == 3 else dk
                    s = head[(ii * cell_n[1] + jj) * cell_n[2] + kk]
                    while s >= 0:
                        dx = x - pos[s, 0]
                        dy = y - pos[s, 1]
                        dz = z - pos[s, 2]
                        dx -= box[0] * math.floor(dx / box[0] + 0.5)
                        dy -= box[1] * math.floor(dy / box[1] + 0.5)
                        dz -= box[2] * math.floor(dz / box[2] + 0.5)
                        lim = r + radii[s] + gap
                        if dx * dx + dy * dy + dz * dz < lim * lim:
                            ok = False
                            break
                        s = nxt[s]
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            pos[placed, 0] = x
            pos[placed, 1] = y
            pos[placed, 2] = z
            cell = (ci * cell_n[1] + cj) * cell_n[2] + ck
            nxt[placed] = head[cell]
            head[cell] = placed
            placed += 1
    return placed, used


def random_sequential_addition(radii, box, gap, rng, max_attempts, batch=4096):
    """Positions (in ``[0, box)``) of non-overlapping spheres, inserted in the given order.

    Raises ``RuntimeError`` when the spheres cannot all be placed within
    ``max_attempts`` trials (jamming).
    """
    radii = np.ascontiguousarray(radii, dtype=float)
    box = np.asarray(box, dtype=float)
    n = len(radii)
    pos = np.zeros((n, 3))
    if n == 0:
        return pos
    reach = 2.0 * radii.max() + gap
    cell_n = np.maximum(1, np.floor(box / reach)).astype(np.int64)
    cell_h = box / cell_n
    head = np.full(int(np.prod(cell_n)), -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    placed, attempts = 0, 0
    while placed < n and attempts < max_attempts:
        m = int(min(batch, max_attempts - attempts))
        cand = rng.random((m, 3)) * box
        placed, used = rsa_insert(
            radii, float(gap), box, cell_h, cell_n, head, nxt, pos, placed, cand
        )
        attempts += used
    if placed < n:
        raise RuntimeError(
            f"random sequential addition jammed: {placed} of {n} spheres placed after "
            f"{attempts} attempts; lower the volume fraction (the RSA jamming limit is "
            f"about 0.38 for equal spheres) or raise max_attempts"
        )
    return pos


# -- fibres: spherocylinders -------------------------------------------------------------


@njit(cache=True, inline="always")
def _clamp(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@njit(cache=True)
def segment_distance2(p1, d1, p2, d2) -> float:
    """Squared distance between the segments ``p1 + s d1`` and ``p2 + t d2`` (s, t in
    [0, 1]): the closest points of Ericson (2005, sec. 5.1.9), robust for parallel or
    degenerate segments."""
    rx, ry, rz = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
    a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]
    e = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]
    f = d2[0] * rx + d2[1] * ry + d2[2] * rz
    eps = 1e-300
    if a <= eps and e <= eps:
        s, t = 0.0, 0.0
    elif a <= eps:
        s, t = 0.0, _clamp(f / e)
    else:
        c = d1[0] * rx + d1[1] * ry + d1[2] * rz
        if e <= eps:
            s, t = _clamp(-c / a), 0.0
        else:
            b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]
            denom = a * e - b * b
            s = _clamp((b * f - c * e) / denom) if denom > 1e-14 * a * e else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, _clamp(-c / a)
            elif t > 1.0:
                t, s = 1.0, _clamp((b - c) / a)
    dx = rx + d1[0] * s - d2[0] * t
    dy = ry + d1[1] * s - d2[1] * t
    dz = rz + d1[2] * s - d2[2] * t
    return dx * dx + dy * dy + dz * dz


@njit(cache=True)
def rsa_insert_fibres(radii, halves, gap, box, cell_h, cell_n, head, nxt, pos, axes, placed,
                      candidates, directions):  # fmt: skip
    """Try ``candidates`` (centres) with ``directions`` (unit axes) in turn for the fibre
    ``placed`` (a spherocylinder of radius ``radii[i]`` and half-length ``halves[i]``),
    then the next ones, in a periodic box with linked cell lists (see :func:`rsa_insert`;
    cells at least as large as the reach of two fibres). Fibres closer than ``gap`` are
    rejected; the nearest periodic image is tested. Returns the new number of placed fibres
    and the number of candidates used."""
    n_total = radii.shape[0]
    used = 0
    q = np.empty(3)
    p1 = np.empty(3)
    d1 = np.empty(3)
    p2 = np.empty(3)
    d2 = np.empty(3)
    for c in range(candidates.shape[0]):
        if placed >= n_total:
            break
        used += 1
        x, y, z = candidates[c, 0], candidates[c, 1], candidates[c, 2]
        q[0], q[1], q[2] = x, y, z
        r, h = radii[placed], halves[placed]
        for k in range(3):
            d1[k] = 2.0 * h * directions[c, k]
            p1[k] = q[k] - 0.5 * d1[k]
        ci = min(int(x / cell_h[0]), cell_n[0] - 1)
        cj = min(int(y / cell_h[1]), cell_n[1] - 1)
        ck = min(int(z / cell_h[2]), cell_n[2] - 1)
        ok = True
        ni = min(3, cell_n[0])
        nj = min(3, cell_n[1])
        nk = min(3, cell_n[2])
        for di in range(ni):
            ii = (ci + di - 1) % cell_n[0] if ni == 3 else di
            for dj in range(nj):
                jj = (cj + dj - 1) % cell_n[1] if nj == 3 else dj
                for dk in range(nk):
                    kk = (ck + dk - 1) % cell_n[2] if nk == 3 else dk
                    s = head[(ii * cell_n[1] + jj) * cell_n[2] + kk]
                    while s >= 0:
                        # the image of fibre s nearest to the candidate
                        for k in range(3):
                            dc = pos[s, k] - q[k]
                            dc -= box[k] * math.floor(dc / box[k] + 0.5)
                            d2[k] = 2.0 * halves[s] * axes[s, k]
                            p2[k] = q[k] + dc - 0.5 * d2[k]
                        lim = r + radii[s] + gap
                        if segment_distance2(p1, d1, p2, d2) < lim * lim:
                            ok = False
                            break
                        s = nxt[s]
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            pos[placed, 0] = x
            pos[placed, 1] = y
            pos[placed, 2] = z
            for k in range(3):
                axes[placed, k] = directions[c, k]
            cell = (ci * cell_n[1] + cj) * cell_n[2] + ck
            nxt[placed] = head[cell]
            head[cell] = placed
            placed += 1
    return placed, used


def random_sequential_addition_fibres(radii, lengths, box, gap, rng, orientation, max_attempts,
                                      batch=4096):  # fmt: skip
    """Centres (in ``[0, box)``) and unit axes of non-overlapping spherocylinders of
    radii ``radii`` and lengths ``lengths`` (of their cylindrical parts), inserted in the
    given order; candidate axes come from ``orientation`` (an orientation distribution).
    Raises ``RuntimeError`` when the fibres cannot all be placed within ``max_attempts``
    trials (jamming)."""
    radii = np.ascontiguousarray(radii, dtype=float)
    halves = np.ascontiguousarray(0.5 * np.asarray(lengths, dtype=float))
    box = np.asarray(box, dtype=float)
    n = len(radii)
    pos = np.zeros((n, 3))
    axes = np.zeros((n, 3))
    if n == 0:
        return pos, axes
    reach = 2.0 * (halves.max() + radii.max()) + gap
    cell_n = np.maximum(1, np.floor(box / reach)).astype(np.int64)
    cell_h = box / cell_n
    head = np.full(int(np.prod(cell_n)), -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    placed, attempts = 0, 0
    while placed < n and attempts < max_attempts:
        m = int(min(batch, max_attempts - attempts))
        cand = rng.random((m, 3)) * box
        dirs = np.ascontiguousarray(orientation.sample(rng, m), dtype=float)
        placed, used = rsa_insert_fibres(radii, halves, float(gap), box, cell_h, cell_n, head,
                                         nxt, pos, axes, placed, cand, dirs)  # fmt: skip
        attempts += used
    if placed < n:
        raise RuntimeError(
            f"random sequential addition jammed: {placed} of {n} fibres placed after "
            f"{attempts} attempts; lower the volume fraction (the jamming limit of RSA "
            "falls as the fibres lengthen) or raise max_attempts"
        )
    return pos, axes


@njit(cache=True)
def rsa_insert_polylines(offsets, extents, radii, gap, box, cell_h, cell_n, head, nxt, pos,
                         placed, candidates):  # fmt: skip
    """Try ``candidates`` (centres) in turn for the fibre ``placed``, a chain of
    spherocylinders of radius ``radii[placed]`` whose axis passes through the points
    ``offsets[placed]`` (relative to its centre; ``extents``: their largest distance to
    it), then the next fibres, in a periodic box with linked cell lists (cells at least
    as large as the reach of two fibres). A candidate is rejected when one of its
    segments comes closer than ``r_a + r_b + gap`` to a segment of the nearest periodic
    image of a fibre already placed. Returns the new number of placed fibres and the
    number of candidates used."""
    n_total, m1 = offsets.shape[0], offsets.shape[1]
    used = 0
    q = np.empty(3)
    dc = np.empty(3)
    a0 = np.empty(3)
    da = np.empty(3)
    b0 = np.empty(3)
    db = np.empty(3)
    for c in range(candidates.shape[0]):
        if placed >= n_total:
            break
        used += 1
        for k in range(3):
            q[k] = candidates[c, k]
        ci = min(int(q[0] / cell_h[0]), cell_n[0] - 1)
        cj = min(int(q[1] / cell_h[1]), cell_n[1] - 1)
        ck = min(int(q[2] / cell_h[2]), cell_n[2] - 1)
        ok = True
        ni = min(3, cell_n[0])
        nj = min(3, cell_n[1])
        nk = min(3, cell_n[2])
        for di in range(ni):
            ii = (ci + di - 1) % cell_n[0] if ni == 3 else di
            for dj in range(nj):
                jj = (cj + dj - 1) % cell_n[1] if nj == 3 else dj
                for dk in range(nk):
                    kk = (ck + dk - 1) % cell_n[2] if nk == 3 else dk
                    s = head[(ii * cell_n[1] + jj) * cell_n[2] + kk]
                    while s >= 0:
                        d2 = 0.0
                        for k in range(3):
                            d = pos[s, k] - q[k]
                            d -= box[k] * math.floor(d / box[k] + 0.5)
                            dc[k] = d
                            d2 += d * d
                        lim = radii[placed] + radii[s] + gap
                        reach = extents[placed] + extents[s] + lim
                        if d2 < reach * reach:
                            for i in range(m1 - 1):
                                for k in range(3):
                                    a0[k] = q[k] + offsets[placed, i, k]
                                    da[k] = offsets[placed, i + 1, k] - offsets[placed, i, k]
                                for j in range(m1 - 1):
                                    for k in range(3):
                                        b0[k] = q[k] + dc[k] + offsets[s, j, k]
                                        db[k] = offsets[s, j + 1, k] - offsets[s, j, k]
                                    if segment_distance2(a0, da, b0, db) < lim * lim:
                                        ok = False
                                        break
                                if not ok:
                                    break
                        if not ok:
                            break
                        s = nxt[s]
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            for k in range(3):
                pos[placed, k] = q[k]
            cell = (ci * cell_n[1] + cj) * cell_n[2] + ck
            nxt[placed] = head[cell]
            head[cell] = placed
            placed += 1
    return placed, used


def random_sequential_addition_polylines(offsets, radius, box, gap, rng, max_attempts,
                                         batch=4096):  # fmt: skip
    """Centres (in ``[0, box)``) of non-overlapping curved fibres: chains of
    spherocylinders of radius ``radius`` (one value, or one per fibre) along the polylines
    ``offsets`` (n x points x 3, relative to the centres), inserted in the given order with
    uniformly random centres. Raises ``RuntimeError`` when they cannot all be placed within
    ``max_attempts`` trials (jamming)."""
    offsets = np.ascontiguousarray(offsets, dtype=float)
    box = np.asarray(box, dtype=float)
    n = len(offsets)
    pos = np.zeros((n, 3))
    if n == 0:
        return pos
    radii = np.ascontiguousarray(np.broadcast_to(np.asarray(radius, dtype=float), (n,)))
    extents = np.ascontiguousarray(np.sqrt((offsets**2).sum(axis=-1)).max(axis=1))
    reach = 2.0 * extents.max() + 2.0 * float(radii.max()) + gap
    cell_n = np.maximum(1, np.floor(box / reach)).astype(np.int64)
    cell_h = box / cell_n
    head = np.full(int(np.prod(cell_n)), -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    placed, attempts = 0, 0
    while placed < n and attempts < max_attempts:
        m = int(min(batch, max_attempts - attempts))
        cand = rng.random((m, 3)) * box
        placed, used = rsa_insert_polylines(offsets, extents, radii, float(gap), box,
                                            cell_h, cell_n, head, nxt, pos, placed,
                                            cand)  # fmt: skip
        attempts += used
    if placed < n:
        raise RuntimeError(
            f"random sequential addition jammed: {placed} of {n} curved fibres placed after "
            f"{attempts} attempts; lower the volume fraction or raise max_attempts"
        )
    return pos
