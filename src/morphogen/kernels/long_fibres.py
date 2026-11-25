"""Long fibres of a periodic medium: the sections of the fibres packed in the plane normal to
their axis, and their wavy axes freed of overlaps (:mod:`morphogen.models.long_fibers`).

The sections are discs in a periodic rectangle ``[0, box)``: :func:`relax_discs` removes
their overlaps by steps that push every overlapping pair apart along the line of its
centres, each disc by half the overlap ``δ = s (r_i + r_j) + gap - d`` (nearest periodic
images, linked cells), the discs of radii ``s r`` (``s`` the scale of an inflation that
brings them to their radii from a looser packing).

The axes of wavy fibres are closed chains of ``K`` nodes in the planes normal to the axis
(``x[i, k]``, the node ``k`` of the fibre ``i`` at the coordinate ``k h`` along the axis,
the node ``K`` the node 0 a period ``K h`` further: the fibre closes on itself). The nodes
stay in their planes: :func:`relax_chains` pushes the segments of different fibres closer
than ``r_i + r_j + gap`` apart along the transverse part of the line of their closest
points (Ericson 2005, sec. 5.1.9; nearest periodic images in the plane and along the axis),
the push of a segment shared by its two nodes in proportion to the barycentric weights of
its closest point, the pushes smoothed along each fibre so that a push bends a fibre over a
few segments rather than kinking it. Pairs are found in linked cells of the nodes of each
plane, wide enough for the segments of the planes within reach along the axis.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from morphogen.kernels.force_biased import _closest


@njit(cache=True, inline="always")
def _wrap(d, period):
    """The nearest periodic image of a difference ``d``."""
    return d - period * math.floor(d / period + 0.5)


@njit(cache=True)
def _disc_cells(x, box, size):
    """Linked cells at least ``size`` wide over the discs: their numbers along the axes,
    the heads and links of their lists and the cell of each disc."""
    n = x.shape[0]
    cell_n = np.empty(2, np.int64)
    cell_h = np.empty(2)
    for k in range(2):
        cell_n[k] = max(1, int(box[k] / size))
        cell_h[k] = box[k] / cell_n[k]
    head = np.full(cell_n[0] * cell_n[1], -1, np.int64)
    nxt = np.empty(n, np.int64)
    cell = np.empty((n, 2), np.int64)
    for i in range(n):
        for k in range(2):
            m = x[i, k] - box[k] * math.floor(x[i, k] / box[k])
            cell[i, k] = min(int(m / cell_h[k]), cell_n[k] - 1)
        c = cell[i, 0] * cell_n[1] + cell[i, 1]
        nxt[i] = head[c]
        head[c] = i
    return cell_n, head, nxt, cell


@njit(cache=True)
def disc_overlaps(x, radii, scale, gap, box, push):
    """One pass over the pairs of discs closer than ``scale (r_i + r_j) + gap`` (nearest
    periodic images): each disc pushed by half the overlap away from the other, added to
    ``push`` (none if it has no rows). Returns the largest overlap (0 without any) and the
    number of overlapping pairs."""
    n = x.shape[0]
    size = 2.0 * scale * radii.max() + gap
    cell_n, head, nxt, cell = _disc_cells(x, box, size)
    ni, nj = min(3, cell_n[0]), min(3, cell_n[1])
    forces = push.shape[0] > 0
    largest = 0.0
    count = 0
    for i in range(n):
        for di in range(ni):
            ii = (cell[i, 0] + di - 1) % cell_n[0] if ni == 3 else di
            for dj in range(nj):
                jj = (cell[i, 1] + dj - 1) % cell_n[1] if nj == 3 else dj
                j = head[ii * cell_n[1] + jj]
                while j >= 0:
                    if j > i:
                        dx = _wrap(x[j, 0] - x[i, 0], box[0])
                        dy = _wrap(x[j, 1] - x[i, 1], box[1])
                        lim = scale * (radii[i] + radii[j]) + gap
                        d2 = dx * dx + dy * dy
                        if d2 < lim * lim:
                            d = math.sqrt(d2)
                            delta = lim - d
                            count += 1
                            largest = max(largest, delta)
                            if forces:
                                if d <= 1e-12 * lim:  # coincident centres: any direction
                                    dx, dy, d = math.cos(i + 0.5 * j), math.sin(i + 0.5 * j), 1.0
                                g = 0.5 * delta / d
                                push[i, 0] -= g * dx
                                push[i, 1] -= g * dy
                                push[j, 0] += g * dx
                                push[j, 1] += g * dy
                    j = nxt[j]
    return largest, count


@njit(cache=True)
def relax_discs(x, radii, scale, gap, box, tol, max_steps, max_move):
    """Steps of :func:`disc_overlaps` on the centres ``x`` (moved in place, wrapped into the
    box), each move capped to ``max_move``, until the largest overlap falls to ``tol`` or
    below. Returns the steps taken and the largest overlap left."""
    n = x.shape[0]
    push = np.zeros((n, 2))
    largest = 0.0
    for step in range(max_steps):
        push[:] = 0.0
        largest, _ = disc_overlaps(x, radii, scale, gap, box, push)
        if largest <= tol:
            return step, largest
        for i in range(n):
            m = math.sqrt(push[i, 0] ** 2 + push[i, 1] ** 2)
            f = 1.0 if m <= max_move else max_move / m
            for k in range(2):
                v = x[i, k] + f * push[i, k]
                x[i, k] = v - box[k] * math.floor(v / box[k])
    push[:] = 0.0
    largest, _ = disc_overlaps(x, radii, scale, gap, box, np.zeros((0, 2)))
    return max_steps, largest


@njit(cache=True)
def _chain_cells(x, box, size):
    """Linked cells at least ``size`` wide over the nodes of each plane ``k``: their numbers
    along the axes, the heads (plane, cell) and links (plane, fibre) of their lists and the
    cell of each node."""
    n, m = x.shape[0], x.shape[1]
    cell_n = np.empty(2, np.int64)
    cell_h = np.empty(2)
    for a in range(2):
        cell_n[a] = max(1, int(box[a] / size))
        cell_h[a] = box[a] / cell_n[a]
    head = np.full((m, cell_n[0] * cell_n[1]), -1, np.int64)
    nxt = np.empty((m, n), np.int64)
    cell = np.empty((n, m, 2), np.int64)
    for k in range(m):
        for i in range(n):
            for a in range(2):
                w = x[i, k, a] - box[a] * math.floor(x[i, k, a] / box[a])
                cell[i, k, a] = min(int(w / cell_h[a]), cell_n[a] - 1)
            c = cell[i, k, 0] * cell_n[1] + cell[i, k, 1]
            nxt[k, i] = head[k, c]
            head[k, c] = i
    return cell_n, head, nxt, cell


@njit(cache=True)
def _extent(x, span):
    """The largest transverse distance between two nodes of a fibre at most ``span``
    planes apart along it."""
    n, m = x.shape[0], x.shape[1]
    out = 0.0
    for i in range(n):
        for k in range(m):
            for d in range(1, span + 1):
                kk = (k + d) % m
                dx = x[i, kk, 0] - x[i, k, 0]
                dy = x[i, kk, 1] - x[i, k, 1]
                out = max(out, math.sqrt(dx * dx + dy * dy))
    return out


@njit(cache=True)
def chain_overlaps(x, radii, gap, box, h, span, push):
    """One pass over the pairs of segments of different fibres closer than ``r_i + r_j +
    gap`` (nearest periodic images in the plane and along the axis), the segments of the
    fibre ``j`` at most ``span`` planes from that of the fibre ``i``: their pushes, along the
    transverse part of the line of their closest points, added to ``push`` (fibres x planes
    x 2; none if it has no rows). Returns the largest overlap (0 without any), the number
    of overlapping pairs and the smallest ``d - r_i - r_j`` of the pairs examined."""
    n, m = x.shape[0], x.shape[1]
    lim_max = 2.0 * radii.max() + gap
    size = lim_max + 2.0 * _extent(x, span + 1)
    cell_n, head, nxt, cell = _chain_cells(x, box, size)
    ni, nj = min(3, cell_n[0]), min(3, cell_n[1])
    forces = push.shape[0] > 0
    largest = 0.0
    count = 0
    clearance = np.inf
    for k in range(m):
        k1 = (k + 1) % m
        for i in range(n):
            px, py = x[i, k, 0], x[i, k, 1]
            ax, ay, az = x[i, k1, 0] - px, x[i, k1, 1] - py, h
            for di in range(ni):
                ii = (cell[i, k, 0] + di - 1) % cell_n[0] if ni == 3 else di
                for dj in range(nj):
                    jj = (cell[i, k, 1] + dj - 1) % cell_n[1] if nj == 3 else dj
                    j = head[k, ii * cell_n[1] + jj]
                    while j >= 0:
                        if j > i:
                            # the image of the fibre j nearest to i in the plane k
                            sx = _wrap(x[j, k, 0] - px, box[0]) - (x[j, k, 0] - px)
                            sy = _wrap(x[j, k, 1] - py, box[1]) - (x[j, k, 1] - py)
                            lim = radii[i] + radii[j] + gap
                            for d in range(-span, span + 1):
                                b0 = (k + d) % m
                                b1 = (b0 + 1) % m
                                qx = x[j, b0, 0] + sx
                                qy = x[j, b0, 1] + sy
                                bx, by, bz = x[j, b1, 0] + sx - qx, x[j, b1, 1] + sy - qy, h
                                rx, ry, rz = px - qx, py - qy, -d * h  # r = p - q
                                u, v = _closest(rx, ry, rz, ax, ay, az, bx, by, bz)
                                wx = rx + u * ax - v * bx
                                wy = ry + u * ay - v * by
                                wz = rz + u * az - v * bz
                                dist = math.sqrt(wx * wx + wy * wy + wz * wz)
                                clearance = min(clearance, dist - radii[i] - radii[j])
                                if dist < lim:
                                    delta = lim - dist
                                    count += 1
                                    largest = max(largest, delta)
                                    if forces:
                                        t = math.sqrt(wx * wx + wy * wy)
                                        if t <= 1e-9 * lim:  # no transverse direction
                                            wx = math.cos(i + 0.5 * j + k)
                                            wy = math.sin(i + 0.5 * j + k)
                                            t = dist = 1.0
                                        g = 0.5 * delta / max(dist, t)
                                        push[i, k, 0] += (1.0 - u) * g * wx
                                        push[i, k, 1] += (1.0 - u) * g * wy
                                        push[i, k1, 0] += u * g * wx
                                        push[i, k1, 1] += u * g * wy
                                        push[j, b0, 0] -= (1.0 - v) * g * wx
                                        push[j, b0, 1] -= (1.0 - v) * g * wy
                                        push[j, b1, 0] -= v * g * wx
                                        push[j, b1, 1] -= v * g * wy
                        j = nxt[k, j]
    return largest, count, clearance


@njit(cache=True)
def relax_chains(x, radii, gap, box, h, span, tol, max_steps, passes, max_move):
    """Steps on the nodes ``x`` of the closed chains (fibres x planes x 2, moved in place
    in their planes): the pushes of :func:`chain_overlaps`, smoothed along each fibre by
    ``passes`` passes of the filter (1/4, 1/2, 1/4) of the neighbouring nodes (which keeps
    their sum: a push bends the fibre over a few segments instead of kinking it), each move
    capped to ``max_move``, until the largest overlap falls to ``tol`` or below. Returns the
    steps taken and the largest overlap left."""
    n, m = x.shape[0], x.shape[1]
    push = np.zeros((n, m, 2))
    tmp = np.zeros((m, 2))
    largest = 0.0
    for step in range(max_steps):
        push[:] = 0.0
        largest, _, _ = chain_overlaps(x, radii, gap, box, h, span, push)
        if largest <= tol:
            return step, largest
        for i in range(n):
            for _ in range(passes):
                for k in range(m):
                    km, kp = (k - 1) % m, (k + 1) % m
                    for a in range(2):
                        tmp[k, a] = 0.5 * push[i, k, a] + 0.25 * (push[i, km, a] + push[i, kp, a])
                push[i] = tmp
            for k in range(m):
                s = math.sqrt(push[i, k, 0] ** 2 + push[i, k, 1] ** 2)
                f = 1.0 if s <= max_move else max_move / s
                x[i, k, 0] += f * push[i, k, 0]
                x[i, k, 1] += f * push[i, k, 1]
    largest, _, _ = chain_overlaps(x, radii, gap, box, h, span, np.zeros((0, 0, 2)))
    return max_steps, largest
