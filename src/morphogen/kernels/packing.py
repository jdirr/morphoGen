"""Dense packings of hard spheres: the Lubachevsky–Stillinger algorithm (1990), as used
for the maximally random jammed state (Torquato, Truskett & Debenedetti 2000; Skoge,
Donev, Stillinger & Torquato 2006).

Event-driven molecular dynamics of hard spheres in a periodic box whose radii grow,
``r_i(t) = s(t) ρ_i`` with ``s(t) = s_0 + γ t``: the spheres fly in straight lines between
collisions, predicted exactly (the roots of a quadratic, the sum of the radii growing at
``σ' = γ (ρ_i + ρ_j)``). At a contact, the normal relative velocity ``u_n`` becomes ``2σ'
- u_n``, so that the gap opens at the rate at which it closed: momentum is kept, and the
growth heats the system, whose velocities are rescaled to the initial temperature after
every ``n`` events (``n`` the number of spheres). Each sphere has one event at a time, its
next collision or the crossing of a face of its cell (cells of edge at least the largest
diameter at the end, 27 neighbouring cells), in a binary heap; collision counts discard
the predictions made stale by the collisions of the partners. The positions are updated
lazily, at the events of each sphere.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

_NEVER = 1.0e300


@njit(cache=True, inline="always")
def _image(d, length):
    return d - length * math.floor(d / length + 0.5)


@njit(cache=True)
def _contact_time(dx, dy, dz, ux, uy, uz, sigma, rate):
    """Time to the contact of two spheres at the offset ``d``, of relative velocity ``u``,
    whose sum of radii ``sigma`` grows at ``rate`` (-1: none)."""
    a = ux * ux + uy * uy + uz * uz - rate * rate
    b = dx * ux + dy * uy + dz * uz - sigma * rate
    c = dx * dx + dy * dy + dz * dz - sigma * sigma
    if c <= 0.0:  # in contact: a collision now if the gap closes
        return 0.0 if b < 0.0 else -1.0
    if b >= 0.0 and a >= 0.0:
        return -1.0
    disc = b * b - a * c
    if disc < 0.0:
        return -1.0
    den = -b + math.sqrt(disc)
    return c / den if den > 0.0 else 0.0


@njit(cache=True, inline="always")
def _sift_up(heap, where, key, k):
    item = heap[k]
    while k > 0:
        parent = (k - 1) >> 1
        p = heap[parent]
        if key[p] <= key[item]:
            break
        heap[k] = p
        where[p] = k
        k = parent
    heap[k] = item
    where[item] = k


@njit(cache=True, inline="always")
def _sift_down(heap, where, key, k, n):
    item = heap[k]
    while True:
        child = 2 * k + 1
        if child >= n:
            break
        if child + 1 < n and key[heap[child + 1]] < key[heap[child]]:
            child += 1
        c = heap[child]
        if key[c] >= key[item]:
            break
        heap[k] = c
        where[c] = k
        k = child
    heap[k] = item
    where[item] = k


@njit(cache=True)
def _insert(i, c, cell, head, nxt, prv):
    cell[i] = c
    prv[i] = -1
    nxt[i] = head[c]
    if head[c] >= 0:
        prv[head[c]] = i
    head[c] = i


@njit(cache=True)
def _remove(i, cell, head, nxt, prv):
    c = cell[i]
    if prv[i] >= 0:
        nxt[prv[i]] = nxt[i]
    else:
        head[c] = nxt[i]
    if nxt[i] >= 0:
        prv[nxt[i]] = prv[i]


@njit(cache=True)
def _advance(i, t, pos, vel, tp):
    dt = t - tp[i]
    for a in range(3):
        pos[i, a] += vel[i, a] * dt
    tp[i] = t


@njit(cache=True)
def _predict(i, t, s0, growth, rho, pos, vel, tp, box, m, width, cell, head, nxt, count,
             ev_time, ev_partner, ev_count):  # fmt: skip
    """The next event of the sphere ``i`` (at the time ``t``, where its position is)."""
    cx = cell[i] % m[0]
    cy = (cell[i] // m[0]) % m[1]
    cz = cell[i] // (m[0] * m[1])
    best, partner = _NEVER, -1
    idx = (cx, cy, cz)
    for a in range(3):  # the faces of its cell
        v = vel[i, a]
        if v > 0.0:
            dt = ((idx[a] + 1) * width[a] - pos[i, a]) / v
            code = -4 - a
        elif v < 0.0:
            dt = (idx[a] * width[a] - pos[i, a]) / v
            code = -1 - a
        else:
            continue
        if t + max(dt, 0.0) < best:
            best, partner = t + max(dt, 0.0), code
    s = s0 + growth * t
    for di in range(-1, 2):
        ii = (cx + di) % m[0]
        for dj in range(-1, 2):
            jj = (cy + dj) % m[1]
            for dk in range(-1, 2):
                kk = (cz + dk) % m[2]
                j = head[ii + m[0] * (jj + m[1] * kk)]
                while j >= 0:
                    if j != i:
                        dtj = t - tp[j]
                        dx = _image(pos[j, 0] + vel[j, 0] * dtj - pos[i, 0], box[0])
                        dy = _image(pos[j, 1] + vel[j, 1] * dtj - pos[i, 1], box[1])
                        dz = _image(pos[j, 2] + vel[j, 2] * dtj - pos[i, 2], box[2])
                        tau = _contact_time(dx, dy, dz, vel[j, 0] - vel[i, 0],
                                            vel[j, 1] - vel[i, 1], vel[j, 2] - vel[i, 2],
                                            s * (rho[i] + rho[j]),
                                            growth * (rho[i] + rho[j]))  # fmt: skip
                        if tau >= 0.0 and t + tau < best:
                            best, partner = t + tau, j
                    j = nxt[j]
    ev_time[i] = best
    ev_partner[i] = partner
    ev_count[i] = count[partner] if partner >= 0 else 0


@njit(cache=True)
def grow_packing(pos, vel, rho, box, s0, growth, t_end, max_events):
    """Grow the spheres of relative radii ``rho`` from the scale ``s0`` (``pos``: their
    non-overlapping positions in ``[0, box)``, ``vel``: their velocities) for the time
    ``t_end``, until ``max_events`` events, or until they jam (the scale grows by less than
    ``1e-10`` of itself over ten rounds of ``n`` events). Returns the time reached, the
    number of events and of collisions; ``pos`` holds the positions at that time."""
    n = len(rho)
    rmax = 0.0
    for i in range(n):
        rmax = max(rmax, rho[i])
    s_end = s0 + growth * t_end
    m = np.empty(3, dtype=np.int64)
    width = np.empty(3)
    for a in range(3):
        m[a] = max(3, int(box[a] / (2.0 * s_end * rmax * (1.0 + 1e-9))))
        width[a] = box[a] / m[a]
    ncell = m[0] * m[1] * m[2]
    head = np.full(ncell, -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    prv = np.full(n, -1, dtype=np.int64)
    cell = np.empty(n, dtype=np.int64)
    for i in range(n):
        c = 0
        f = 1
        for a in range(3):
            ia = min(int(pos[i, a] / width[a]), m[a] - 1)
            c += ia * f
            f *= m[a]
        _insert(i, c, cell, head, nxt, prv)
    tp = np.zeros(n)
    count = np.zeros(n, dtype=np.int64)
    ev_time = np.empty(n)
    ev_partner = np.empty(n, dtype=np.int64)
    ev_count = np.empty(n, dtype=np.int64)
    heap = np.arange(n)
    where = np.arange(n)
    energy = 0.0
    for i in range(n):
        for a in range(3):
            energy += vel[i, a] * vel[i, a]
    t = 0.0
    events, collisions = 0, 0
    restart = True
    t_last, stalls = 0.0, 0  # jammed: the radii no longer grow
    while True:
        if restart:  # all the spheres at the time t, all the events predicted again
            for i in range(n):
                _advance(i, t, pos, vel, tp)
            for i in range(n):
                _predict(i, t, s0, growth, rho, pos, vel, tp, box, m, width, cell, head, nxt,
                         count, ev_time, ev_partner, ev_count)  # fmt: skip
            for k in range(n):
                heap[k] = k
                where[k] = k
            for k in range(n // 2 - 1, -1, -1):
                _sift_down(heap, where, ev_time, k, n)
            restart = False
        i = heap[0]
        te = ev_time[i]
        if te >= t_end:
            t = t_end
            break
        if events >= max_events:
            break
        t = te
        events += 1
        j = ev_partner[i]
        if j >= 0:
            if count[j] == ev_count[i]:  # a collision, unless j collided since
                _advance(i, t, pos, vel, tp)
                _advance(j, t, pos, vel, tp)
                dx = _image(pos[j, 0] - pos[i, 0], box[0])
                dy = _image(pos[j, 1] - pos[i, 1], box[1])
                dz = _image(pos[j, 2] - pos[i, 2], box[2])
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                nx, ny, nz = dx / d, dy / d, dz / d
                rate = growth * (rho[i] + rho[j])
                un = ((vel[j, 0] - vel[i, 0]) * nx + (vel[j, 1] - vel[i, 1]) * ny
                      + (vel[j, 2] - vel[i, 2]) * nz)  # fmt: skip
                delta = rate - un
                vel[i, 0] -= delta * nx
                vel[i, 1] -= delta * ny
                vel[i, 2] -= delta * nz
                vel[j, 0] += delta * nx
                vel[j, 1] += delta * ny
                vel[j, 2] += delta * nz
                count[i] += 1
                count[j] += 1
                collisions += 1
                _predict(j, t, s0, growth, rho, pos, vel, tp, box, m, width, cell, head, nxt,
                         count, ev_time, ev_partner, ev_count)  # fmt: skip
                _sift_up(heap, where, ev_time, where[j])
                _sift_down(heap, where, ev_time, where[j], n)
            else:
                _advance(i, t, pos, vel, tp)
        else:  # the crossing of a face of the cell
            _advance(i, t, pos, vel, tp)
            code = -1 - j
            axis = code % 3
            up = code >= 3
            c = cell[i]
            cx = c % m[0]
            cy = (c // m[0]) % m[1]
            cz = c // (m[0] * m[1])
            idx = [cx, cy, cz]
            idx[axis] += 1 if up else -1
            if idx[axis] == m[axis]:
                idx[axis] = 0
                pos[i, axis] -= box[axis]
            elif idx[axis] < 0:
                idx[axis] = m[axis] - 1
                pos[i, axis] += box[axis]
            pos[i, axis] = idx[axis] * width[axis] if up else (idx[axis] + 1) * width[axis]
            _remove(i, cell, head, nxt, prv)
            _insert(i, idx[0] + m[0] * (idx[1] + m[1] * idx[2]), cell, head, nxt, prv)
        _predict(i, t, s0, growth, rho, pos, vel, tp, box, m, width, cell, head, nxt, count,
                 ev_time, ev_partner, ev_count)  # fmt: skip
        _sift_up(heap, where, ev_time, where[i])
        _sift_down(heap, where, ev_time, where[i], n)
        if events % (5 * n) == 0:  # back to the initial temperature
            if growth * (t - t_last) < 1e-10 * (s0 + growth * t):
                stalls += 1
                if stalls >= 4:  # jammed: four rounds of 5 n events without growth
                    break
            else:
                stalls = 0
            t_last = t
            for k in range(n):
                _advance(k, t, pos, vel, tp)
            now = 0.0
            for k in range(n):
                for a in range(3):
                    now += vel[k, a] * vel[k, a]
            f = math.sqrt(energy / now) if now > 0.0 else 1.0
            for k in range(n):
                for a in range(3):
                    vel[k, a] *= f
            restart = True
    for i in range(n):
        _advance(i, t, pos, vel, tp)
        for a in range(3):
            pos[i, a] -= box[a] * math.floor(pos[i, a] / box[a])
    return t, events, collisions
