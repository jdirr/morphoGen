"""Force-biased packing of fibres, chains of spherocylinders (Altendorf & Jeulin 2011).

The fibres are chains of nodes in a periodic box ``[0, box)`` (unwrapped coordinates:
consecutive nodes of a chain are joined by a segment, the axis of a spherocylinder of the
radius of its fibre). At every step, three displacements ("forces", Altendorf & Jeulin
2011, sec. II B) are computed from the current configuration and applied together:

* repulsion: two segments whose axes are closer than the sum of their radii plus ``gap``
  (nearest periodic images) by ``δ`` are each pushed apart by ``δ/2`` along the line of
  their closest points (Ericson 2005, sec. 5.1.9; the middle of the common span for
  parallel segments), the push shared by the two nodes of a segment in proportion to the
  barycentric weights of its closest point; segments of a fibre at most ``related``
  steps apart along it are exempt (they touch at the joints);
* springs: a segment whose length departs from its rest length ``D0`` by ``ΔD`` moves
  each of its nodes by ``f(|ΔD| / D0) ΔD`` along it, ``f`` the friction ramp of the paper,
  0 below 5 %, 1 above 10 % and a half cosine between;
* open angular springs: a joint more bent than at rest (its angle ``α`` below its rest
  angle ``α0``) moves its node towards the chord of its two neighbours by
  ``f(α0 - α) (z - z0)``, ``z`` its distance to the chord and ``z0`` the distance at which
  the angle would be ``α0`` (ramp from 0.1 to 0.2 rad), or, folded beyond the chord (the
  foot of the node outside it), by ``f`` times its distance to the middle of the chord;
  straightened joints are free.

The total displacement of a node is the repulsion plus ``recovery`` (the paper's ε)
times the springs and angles, capped to ``max_move``. Segments are found in linked cells
at least as large as their reach, ``2 r_max + gap`` plus the longest segment.

The contact packing of Keilmann, Redenbach & Willot (2026) adds contact forces
(:func:`contact_pulls`): shortlisted pairs of segments of different fibres (found by
:func:`near_pairs`) farther apart than in contact are drawn together, the complement of
the repulsions (:func:`pack_contacts`).

Before the packing, the chains may be placed one after another where they overlap the
chains already placed the least (:func:`place`), the initial placement of Altendorf,
Jeulin & Willot (2014, sec. 2.3), which shortens the packing.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

#: Friction ramps of the recovery forces (Altendorf & Jeulin 2011): relative change of
#: length of a segment, deviation of the angle of a joint (radians), from no force to all
SPRING_RAMP = (0.05, 0.10)
ANGLE_RAMP = (0.1, 0.2)


@njit(cache=True, inline="always")
def _clamp(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@njit(cache=True, inline="always")
def _ramp(x, lo, hi):
    """The friction factor: 0 below ``lo``, 1 above ``hi``, a half cosine between."""
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return 0.5 - 0.5 * math.cos(math.pi * (x - lo) / (hi - lo))


@njit(cache=True, inline="always")
def _closest(rx, ry, rz, ax, ay, az, bx, by, bz):
    """Parameters ``(s, t)`` in [0, 1] of the closest points of the segments ``p + s a``
    and ``q + t b``, ``r = p - q`` (Ericson 2005, sec. 5.1.9); for (nearly) parallel
    segments, the middle of the span of ``a`` facing ``b``."""
    A = ax * ax + ay * ay + az * az
    E = bx * bx + by * by + bz * bz
    F = bx * rx + by * ry + bz * rz
    if A <= 1e-300 and E <= 1e-300:
        return 0.0, 0.0
    if A <= 1e-300:
        return 0.0, _clamp(F / E)
    C = ax * rx + ay * ry + az * rz
    if E <= 1e-300:
        return _clamp(-C / A), 0.0
    B = ax * bx + ay * by + az * bz
    den = A * E - B * B
    if den > 1e-9 * A * E:
        s = _clamp((B * F - C * E) / den)
    else:
        s = 0.5 * (_clamp(-C / A) + _clamp((B - C) / A))
    t = (B * s + F) / E
    if t < 0.0:
        t, s = 0.0, _clamp(-C / A)
    elif t > 1.0:
        t, s = 1.0, _clamp((B - C) / A)
    return s, t


@njit(cache=True, inline="always")
def _share(push, node, g, wx, wy, wz):
    push[node, 0] += g * wx
    push[node, 1] += g * wy
    push[node, 2] += g * wz


@njit(cache=True)
def _reach(x, seg_node, radius, gap):
    """Largest radius and largest length of a segment, and the size of the cells."""
    longest = 0.0
    for s in range(seg_node.shape[0]):
        a = seg_node[s]
        d2 = 0.0
        for k in range(3):
            d = x[a + 1, k] - x[a, k]
            d2 += d * d
        longest = max(longest, math.sqrt(d2))
    return 2.0 * radius.max() + gap + longest


@njit(cache=True)
def _cells(x, seg_node, box, size):
    """Linked cells at least ``size`` wide over the segments: their numbers along the
    axes, the heads and links of their lists, and the vectors, wrapped midpoints, half
    lengths and cells of the segments."""
    n_seg = seg_node.shape[0]
    cell_n = np.empty(3, np.int64)
    cell_h = np.empty(3)
    for k in range(3):
        cell_n[k] = max(1, int(box[k] / size))
        cell_h[k] = box[k] / cell_n[k]
    head = np.full(cell_n[0] * cell_n[1] * cell_n[2], -1, np.int64)
    nxt = np.empty(n_seg, np.int64)
    mid = np.empty((n_seg, 3))
    vec = np.empty((n_seg, 3))
    half = np.empty(n_seg)
    cell = np.empty((n_seg, 3), np.int64)
    for s in range(n_seg):
        a = seg_node[s]
        d2 = 0.0
        for k in range(3):
            d = x[a + 1, k] - x[a, k]
            vec[s, k] = d
            d2 += d * d
            m = x[a, k] + 0.5 * d
            m -= box[k] * math.floor(m / box[k])
            mid[s, k] = m
            cell[s, k] = min(int(m / cell_h[k]), cell_n[k] - 1)
        half[s] = 0.5 * math.sqrt(d2)
        c = (cell[s, 0] * cell_n[1] + cell[s, 1]) * cell_n[2] + cell[s, 2]
        nxt[s] = head[c]
        head[c] = s
    return cell_n, head, nxt, mid, vec, half, cell


@njit(cache=True)
def overlaps(x, seg_node, seg_fibre, seg_index, radius, related, box, gap, push, worst):
    """One pass over the pairs of segments closer than ``r_a + r_b + gap`` (nearest
    periodic images; same-fibre pairs at most ``related[fibre]`` steps apart skipped):
    their repulsions added to ``push`` (nodes x 3; none if it has no rows) and the
    largest such ``δ`` of each fibre kept in ``worst``. Returns the largest ``δ`` (0
    without overlaps) and the number of overlapping pairs."""
    n_seg = seg_node.shape[0]
    size = _reach(x, seg_node, radius, gap)
    cell_n, head, nxt, mid, vec, half, cell = _cells(x, seg_node, box, size)
    ni, nj, nk = min(3, cell_n[0]), min(3, cell_n[1]), min(3, cell_n[2])
    forces = push.shape[0] > 0
    largest = 0.0
    count = 0
    for s in range(n_seg):
        a = seg_node[s]
        fs = seg_fibre[s]
        rs = radius[fs]
        ax, ay, az = vec[s, 0], vec[s, 1], vec[s, 2]
        for di in range(ni):
            ii = (cell[s, 0] + di - 1) % cell_n[0] if ni == 3 else di
            for dj in range(nj):
                jj = (cell[s, 1] + dj - 1) % cell_n[1] if nj == 3 else dj
                for dk in range(nk):
                    kk = (cell[s, 2] + dk - 1) % cell_n[2] if nk == 3 else dk
                    t = head[(ii * cell_n[1] + jj) * cell_n[2] + kk]
                    while t >= 0:
                        if t > s:
                            ft = seg_fibre[t]
                            near = ft == fs and abs(seg_index[t] - seg_index[s]) <= related[fs]
                            lim = rs + radius[ft] + gap
                            mx = mid[t, 0] - mid[s, 0]
                            my = mid[t, 1] - mid[s, 1]
                            mz = mid[t, 2] - mid[s, 2]
                            mx -= box[0] * math.floor(mx / box[0] + 0.5)
                            my -= box[1] * math.floor(my / box[1] + 0.5)
                            mz -= box[2] * math.floor(mz / box[2] + 0.5)
                            reach = lim + half[s] + half[t]
                            if not near and mx * mx + my * my + mz * mz < reach * reach:
                                bx, by, bz = vec[t, 0], vec[t, 1], vec[t, 2]
                                # r = p - q, p and q the starts of s and of the image of t
                                rx = -mx - 0.5 * ax + 0.5 * bx
                                ry = -my - 0.5 * ay + 0.5 * by
                                rz = -mz - 0.5 * az + 0.5 * bz
                                u, v = _closest(rx, ry, rz, ax, ay, az, bx, by, bz)
                                wx = rx + u * ax - v * bx
                                wy = ry + u * ay - v * by
                                wz = rz + u * az - v * bz
                                d = math.sqrt(wx * wx + wy * wy + wz * wz)
                                if d < lim:
                                    delta = lim - d
                                    count += 1
                                    largest = max(largest, delta)
                                    worst[fs] = max(worst[fs], delta)
                                    worst[ft] = max(worst[ft], delta)
                                    if forces:
                                        if d <= 1e-12 * lim:  # crossing axes
                                            wx = ay * bz - az * by
                                            wy = az * bx - ax * bz
                                            wz = ax * by - ay * bx
                                            d = math.sqrt(wx * wx + wy * wy + wz * wz)
                                            if d <= 1e-300:  # parallel too
                                                wx, wy, wz = -ay - az, ax - az, ax + ay
                                                d = math.sqrt(wx * wx + wy * wy + wz * wz)
                                        g = 0.5 * delta / d
                                        b = seg_node[t]
                                        _share(push, a, (1.0 - u) * g, wx, wy, wz)
                                        _share(push, a + 1, u * g, wx, wy, wz)
                                        _share(push, b, (v - 1.0) * g, wx, wy, wz)
                                        _share(push, b + 1, -v * g, wx, wy, wz)
                        t = nxt[t]
    return largest, count


@njit(cache=True)
def near_pairs(x, seg_node, seg_fibre, radius, box, gap):
    """The pairs of segments of different fibres whose axes are closer than ``r_a + r_b +
    gap`` (nearest periodic images): the indices ``s < t`` and the distances."""
    n_seg = seg_node.shape[0]
    size = _reach(x, seg_node, radius, gap)
    cell_n, head, nxt, mid, vec, half, cell = _cells(x, seg_node, box, size)
    ni, nj, nk = min(3, cell_n[0]), min(3, cell_n[1]), min(3, cell_n[2])
    found_s = np.empty(64, np.int64)
    found_t = np.empty(64, np.int64)
    found_d = np.empty(64)
    count = 0
    for s in range(n_seg):
        fs = seg_fibre[s]
        ax, ay, az = vec[s, 0], vec[s, 1], vec[s, 2]
        for di in range(ni):
            ii = (cell[s, 0] + di - 1) % cell_n[0] if ni == 3 else di
            for dj in range(nj):
                jj = (cell[s, 1] + dj - 1) % cell_n[1] if nj == 3 else dj
                for dk in range(nk):
                    kk = (cell[s, 2] + dk - 1) % cell_n[2] if nk == 3 else dk
                    t = head[(ii * cell_n[1] + jj) * cell_n[2] + kk]
                    while t >= 0:
                        if t > s and seg_fibre[t] != fs:
                            lim = radius[fs] + radius[seg_fibre[t]] + gap
                            mx = mid[t, 0] - mid[s, 0]
                            my = mid[t, 1] - mid[s, 1]
                            mz = mid[t, 2] - mid[s, 2]
                            mx -= box[0] * math.floor(mx / box[0] + 0.5)
                            my -= box[1] * math.floor(my / box[1] + 0.5)
                            mz -= box[2] * math.floor(mz / box[2] + 0.5)
                            reach = lim + half[s] + half[t]
                            if mx * mx + my * my + mz * mz < reach * reach:
                                bx, by, bz = vec[t, 0], vec[t, 1], vec[t, 2]
                                rx = -mx - 0.5 * ax + 0.5 * bx
                                ry = -my - 0.5 * ay + 0.5 * by
                                rz = -mz - 0.5 * az + 0.5 * bz
                                u, v = _closest(rx, ry, rz, ax, ay, az, bx, by, bz)
                                wx = rx + u * ax - v * bx
                                wy = ry + u * ay - v * by
                                wz = rz + u * az - v * bz
                                d = math.sqrt(wx * wx + wy * wy + wz * wz)
                                if d < lim:
                                    if count == found_s.shape[0]:  # grown by doubling
                                        found_s = np.concatenate((found_s, np.empty_like(found_s)))
                                        found_t = np.concatenate((found_t, np.empty_like(found_t)))
                                        found_d = np.concatenate((found_d, np.empty_like(found_d)))
                                    found_s[count], found_t[count], found_d[count] = s, t, d
                                    count += 1
                        t = nxt[t]
    return found_s[:count].copy(), found_t[:count].copy(), found_d[:count].copy()


@njit(cache=True)
def place(x, first, radius, box, gap, shifts):
    """The initial placement of least overlap: the chains ``x`` (each from its first node,
    moved in place) translated one after another by the best of their candidate
    ``shifts`` (fibres x trials x 3), the one whose segments overlap those of the chains
    already placed the least (the sum of the depths ``δ`` of the pairs closer than ``r_a +
    r_b + gap``, nearest periodic images; a trial stopped once it exceeds the best, the
    first without overlap taken at once). Returns the sum of the depths of the overlaps
    left, each pair counted once."""
    n = first.shape[0] - 1
    n_seg = x.shape[0] - n
    longest = 0.0
    for f in range(n):
        for a in range(first[f], first[f + 1] - 1):
            d2 = 0.0
            for k in range(3):
                d = x[a + 1, k] - x[a, k]
                d2 += d * d
            longest = max(longest, math.sqrt(d2))
    size = 2.0 * radius.max() + gap + longest
    cell_n = np.empty(3, np.int64)
    cell_h = np.empty(3)
    for k in range(3):
        cell_n[k] = max(1, int(box[k] / size))
        cell_h[k] = box[k] / cell_n[k]
    ni, nj, nk = min(3, cell_n[0]), min(3, cell_n[1]), min(3, cell_n[2])
    head = np.full(cell_n[0] * cell_n[1] * cell_n[2], -1, np.int64)
    nxt = np.empty(n_seg, np.int64)
    mid = np.empty((n_seg, 3))
    vec = np.empty((n_seg, 3))
    half = np.empty(n_seg)
    fibre = np.empty(n_seg, np.int64)
    m = np.empty(3)
    c = np.empty(3, np.int64)
    placed = 0
    total = 0.0
    for f in range(n):
        lo, hi = first[f], first[f + 1]
        best, best_t = np.inf, 0
        for t in range(shifts.shape[1]):
            score = 0.0
            for a in range(lo, hi - 1):
                ax, ay, az = x[a + 1, 0] - x[a, 0], x[a + 1, 1] - x[a, 1], x[a + 1, 2] - x[a, 2]
                hs = 0.5 * math.sqrt(ax * ax + ay * ay + az * az)
                m[0] = x[a, 0] + 0.5 * ax + shifts[f, t, 0]
                m[1] = x[a, 1] + 0.5 * ay + shifts[f, t, 1]
                m[2] = x[a, 2] + 0.5 * az + shifts[f, t, 2]
                for k in range(3):
                    m[k] -= box[k] * math.floor(m[k] / box[k])
                    c[k] = min(int(m[k] / cell_h[k]), cell_n[k] - 1)
                for di in range(ni):
                    ii = (c[0] + di - 1) % cell_n[0] if ni == 3 else di
                    for dj in range(nj):
                        jj = (c[1] + dj - 1) % cell_n[1] if nj == 3 else dj
                        for dk in range(nk):
                            kk = (c[2] + dk - 1) % cell_n[2] if nk == 3 else dk
                            q = head[(ii * cell_n[1] + jj) * cell_n[2] + kk]
                            while q >= 0:
                                lim = radius[f] + radius[fibre[q]] + gap
                                mx = mid[q, 0] - m[0]
                                my = mid[q, 1] - m[1]
                                mz = mid[q, 2] - m[2]
                                mx -= box[0] * math.floor(mx / box[0] + 0.5)
                                my -= box[1] * math.floor(my / box[1] + 0.5)
                                mz -= box[2] * math.floor(mz / box[2] + 0.5)
                                reach = lim + hs + half[q]
                                if mx * mx + my * my + mz * mz < reach * reach:
                                    bx, by, bz = vec[q, 0], vec[q, 1], vec[q, 2]
                                    rx = -mx - 0.5 * ax + 0.5 * bx
                                    ry = -my - 0.5 * ay + 0.5 * by
                                    rz = -mz - 0.5 * az + 0.5 * bz
                                    u, v = _closest(rx, ry, rz, ax, ay, az, bx, by, bz)
                                    wx = rx + u * ax - v * bx
                                    wy = ry + u * ay - v * by
                                    wz = rz + u * az - v * bz
                                    d = math.sqrt(wx * wx + wy * wy + wz * wz)
                                    if d < lim:
                                        score += lim - d
                                q = nxt[q]
                if score >= best:
                    break
            if score < best:
                best, best_t = score, t
            if best <= 0.0:
                break
        total += best
        for a in range(lo, hi):  # moved to the best position, its segments then listed
            for k in range(3):
                x[a, k] += shifts[f, best_t, k]
        for a in range(lo, hi - 1):
            d2 = 0.0
            for k in range(3):
                d = x[a + 1, k] - x[a, k]
                vec[placed, k] = d
                d2 += d * d
                mk = x[a, k] + 0.5 * d
                mk -= box[k] * math.floor(mk / box[k])
                mid[placed, k] = mk
                c[k] = min(int(mk / cell_h[k]), cell_n[k] - 1)
            half[placed] = 0.5 * math.sqrt(d2)
            fibre[placed] = f
            cell = (c[0] * cell_n[1] + c[1]) * cell_n[2] + c[2]
            nxt[placed] = head[cell]
            head[cell] = placed
            placed += 1
    return total


@njit(cache=True)
def contact_pulls(x, seg_node, pair_s, pair_t, active, lim, box, smooth, weight, pull):
    """The contact forces of Keilmann et al. (2026, eqs. 20-22) on the active pairs of
    segments, added to ``pull``: two segments farther apart than ``lim`` (that pair's
    ``r_a + r_b + gap + epsilon``) by ``e > 0`` are each drawn ``weight f(e) e / 2`` towards
    the other along the line of their closest points (nearest periodic images), ``f`` the
    friction ramp from 0 at ``e = 0`` to 1 at ``smooth`` (1 if 0), the pull shared by the
    nodes of a segment as the repulsions are."""
    for p in range(pair_s.shape[0]):
        if not active[p]:
            continue
        a, b = seg_node[pair_s[p]], seg_node[pair_t[p]]
        ax, ay, az = x[a + 1, 0] - x[a, 0], x[a + 1, 1] - x[a, 1], x[a + 1, 2] - x[a, 2]
        bx, by, bz = x[b + 1, 0] - x[b, 0], x[b + 1, 1] - x[b, 1], x[b + 1, 2] - x[b, 2]
        mx = x[b, 0] + 0.5 * bx - x[a, 0] - 0.5 * ax
        my = x[b, 1] + 0.5 * by - x[a, 1] - 0.5 * ay
        mz = x[b, 2] + 0.5 * bz - x[a, 2] - 0.5 * az
        mx -= box[0] * math.floor(mx / box[0] + 0.5)
        my -= box[1] * math.floor(my / box[1] + 0.5)
        mz -= box[2] * math.floor(mz / box[2] + 0.5)
        rx = -mx - 0.5 * ax + 0.5 * bx
        ry = -my - 0.5 * ay + 0.5 * by
        rz = -mz - 0.5 * az + 0.5 * bz
        u, v = _closest(rx, ry, rz, ax, ay, az, bx, by, bz)
        wx = rx + u * ax - v * bx  # from the closest point of b to that of a
        wy = ry + u * ay - v * by
        wz = rz + u * az - v * bz
        d = math.sqrt(wx * wx + wy * wy + wz * wz)
        e = d - lim[p]
        if e <= 0.0:
            continue
        g = 0.5 * weight * e / d * (_ramp(e, 0.0, smooth) if smooth > 0.0 else 1.0)
        _share(pull, a, -(1.0 - u) * g, wx, wy, wz)
        _share(pull, a + 1, -u * g, wx, wy, wz)
        _share(pull, b, (1.0 - v) * g, wx, wy, wz)
        _share(pull, b + 1, v * g, wx, wy, wz)


@njit(cache=True, inline="always")
def _rest_distance(h1, h2, angle):
    """Distance ``z0`` of a node to the chord of its neighbours (whose foot is at the
    distances ``h1`` and ``h2`` from them) at which the angle of the joint is ``angle``:
    the positive root of ``z² - (h1 + h2) cot(angle) z - h1 h2 = 0`` (0 for a straight
    rest angle)."""
    if angle >= math.pi - 1e-12:
        return 0.0
    b = (h1 + h2) * math.cos(angle) / math.sin(angle)
    disc = math.sqrt(b * b + 4.0 * h1 * h2)
    return 0.5 * (b + disc) if b >= 0.0 else 2.0 * h1 * h2 / (disc - b)


@njit(cache=True)
def recoveries(x, first, rest, rest_angle, out):
    """The spring and open angular spring displacements of the nodes, added to ``out``."""
    d_lo, d_hi = SPRING_RAMP
    a_lo, a_hi = ANGLE_RAMP
    for f in range(first.shape[0] - 1):
        lo, hi = first[f], first[f + 1]
        for a in range(lo, hi - 1):
            s = a - f  # the segments of fibre f follow those of the previous fibres
            vx, vy, vz = x[a, 0] - x[a + 1, 0], x[a, 1] - x[a + 1, 1], x[a, 2] - x[a + 1, 2]
            length = math.sqrt(vx * vx + vy * vy + vz * vz)
            change = rest[s] - length
            g = _ramp(abs(change) / rest[s], d_lo, d_hi)
            if g > 0.0 and length > 0.0:
                g *= change / length
                out[a, 0] += g * vx
                out[a, 1] += g * vy
                out[a, 2] += g * vz
                out[a + 1, 0] -= g * vx
                out[a + 1, 1] -= g * vy
                out[a + 1, 2] -= g * vz
        for p in range(lo + 1, hi - 1):
            e1x, e1y, e1z = x[p - 1, 0] - x[p, 0], x[p - 1, 1] - x[p, 1], x[p - 1, 2] - x[p, 2]
            e2x, e2y, e2z = x[p + 1, 0] - x[p, 0], x[p + 1, 1] - x[p, 1], x[p + 1, 2] - x[p, 2]
            n1 = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
            n2 = math.sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
            if n1 <= 0.0 or n2 <= 0.0:
                continue
            cos = (e1x * e2x + e1y * e2y + e1z * e2z) / (n1 * n2)
            deviation = rest_angle[p] - math.acos(max(-1.0, min(1.0, cos)))
            if deviation <= a_lo:
                continue
            cx, cy, cz = e2x - e1x, e2y - e1y, e2z - e1z  # the chord q1 -> q2
            lc = math.sqrt(cx * cx + cy * cy + cz * cz)
            if lc <= 0.0:
                continue
            cx, cy, cz = cx / lc, cy / lc, cz / lc
            h1 = -(e1x * cx + e1y * cy + e1z * cz)  # foot of the node: q1 + h1 c
            h2 = lc - h1
            g = _ramp(deviation, a_lo, a_hi)
            if h1 <= 0.0 or h2 <= 0.0:  # folded beyond the chord: towards its middle
                out[p, 0] += 0.5 * g * (e1x + e2x)
                out[p, 1] += 0.5 * g * (e1y + e2y)
                out[p, 2] += 0.5 * g * (e1z + e2z)
                continue
            wx, wy, wz = e1x + h1 * cx, e1y + h1 * cy, e1z + h1 * cz  # node -> foot
            z = math.sqrt(wx * wx + wy * wy + wz * wz)
            z0 = _rest_distance(h1, h2, rest_angle[p])
            if z > z0 and z > 1e-14 * lc:
                g *= (z - z0) / z
                out[p, 0] += g * wx
                out[p, 1] += g * wy
                out[p, 2] += g * wz


@njit(cache=True)
def _step(x, push, seg_node, rest, max_move):
    """Moves the nodes by their displacements capped to ``max_move``, then shortens to
    its rest length about its middle a segment stretched beyond twice that length."""
    for i in range(x.shape[0]):
        norm = math.sqrt(push[i, 0] ** 2 + push[i, 1] ** 2 + push[i, 2] ** 2)
        scale = max_move / norm if norm > max_move else 1.0
        for k in range(3):
            x[i, k] += scale * push[i, k]
    for s in range(seg_node.shape[0]):
        a = seg_node[s]
        dx, dy, dz = x[a + 1, 0] - x[a, 0], x[a + 1, 1] - x[a, 1], x[a + 1, 2] - x[a, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d > 2.0 * rest[s]:
            g = 0.5 - rest[s] / d
            x[a, 0] += g * dx
            x[a, 1] += g * dy
            x[a, 2] += g * dz
            x[a + 1, 0] -= g * dx
            x[a + 1, 1] -= g * dy
            x[a + 1, 2] -= g * dz


@njit(cache=True)
def pack(x, first, seg_node, seg_fibre, seg_index, radius, related, rest, rest_angle, box,
         gap, recovery, max_move, force_tol, overlap_tol, max_steps, repulsion_only):  # fmt: skip
    """Force-biased steps on the nodes ``x`` (moved in place) until the largest overlap
    falls to ``overlap_tol`` and the total force (the sum of the lengths of the
    displacements before capping) to ``force_tol``, or, with ``repulsion_only`` (the end
    step: repulsions alone), the overlap alone; at most ``max_steps`` steps. A segment
    stretched beyond twice its rest length after a step is shortened to it about its
    middle (a fraction out of reach tears the fibres apart, which would swell the cells).
    Returns the number of steps, the total force and the largest overlap ``δ`` of the last
    configuration."""
    n = x.shape[0]
    push = np.zeros((n, 3))
    back = np.zeros((n, 3))
    worst = np.zeros(radius.shape[0])
    steps = 0
    while True:
        push[:] = 0.0
        largest, _ = overlaps(
            x, seg_node, seg_fibre, seg_index, radius, related, box, gap, push, worst
        )
        if not repulsion_only:
            back[:] = 0.0
            recoveries(x, first, rest, rest_angle, back)
        total = 0.0
        for i in range(n):
            fx = push[i, 0] + recovery * back[i, 0]
            fy = push[i, 1] + recovery * back[i, 1]
            fz = push[i, 2] + recovery * back[i, 2]
            norm = math.sqrt(fx * fx + fy * fy + fz * fz)
            total += norm
            push[i, 0], push[i, 1], push[i, 2] = fx, fy, fz
        done = largest <= overlap_tol and (repulsion_only or total <= force_tol)
        if done or steps >= max_steps:
            return steps, total, largest
        _step(x, push, seg_node, rest, max_move)
        steps += 1


@njit(cache=True)
def pack_contacts(x, first, seg_node, seg_fibre, seg_index, radius, related, rest, rest_angle,
                  box, gap, recovery, max_move, pair_s, pair_t, active, lim, smooth, weight,
                  rel_tol, max_steps, force):  # fmt: skip
    """The contact packing of Keilmann et al. (2026, eq. 23, Algorithm 1): force-biased
    steps whose displacements are the repulsions, ``recovery`` times the springs and
    angles and the contact pulls of the active pairs (weighted by ``weight`` in
    :func:`contact_pulls`), until the total force changes over a step by less than
    ``rel_tol`` of itself (the paper's 0.001 %) or vanishes, at most ``max_steps`` steps.
    ``force`` receives the length of the displacement of each node in the last
    configuration. Returns the number of steps, the total force and the largest
    overlap."""
    n = x.shape[0]
    push = np.zeros((n, 3))
    back = np.zeros((n, 3))
    pull = np.zeros((n, 3))
    worst = np.zeros(radius.shape[0])
    steps = 0
    previous = np.inf
    while True:
        push[:] = 0.0
        largest, _ = overlaps(
            x, seg_node, seg_fibre, seg_index, radius, related, box, gap, push, worst
        )
        back[:] = 0.0
        recoveries(x, first, rest, rest_angle, back)
        pull[:] = 0.0
        contact_pulls(x, seg_node, pair_s, pair_t, active, lim, box, smooth, weight, pull)
        total = 0.0
        for i in range(n):
            fx = push[i, 0] + recovery * back[i, 0] + pull[i, 0]
            fy = push[i, 1] + recovery * back[i, 1] + pull[i, 1]
            fz = push[i, 2] + recovery * back[i, 2] + pull[i, 2]
            force[i] = math.sqrt(fx * fx + fy * fy + fz * fz)
            total += force[i]
            push[i, 0], push[i, 1], push[i, 2] = fx, fy, fz
        stalled = abs(previous - total) < rel_tol * previous
        if steps >= max_steps or stalled or total <= 1e-12 * max_move * n:
            return steps, total, largest
        previous = total
        _step(x, push, seg_node, rest, max_move)
        steps += 1
