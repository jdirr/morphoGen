"""The placement of seeds made of spheres, the largest first, each at the first of its
candidate positions where its spheres overlap those of the seeds already placed by less
than a tolerance (MicroStructPy: Hart & Rimoli 2020, CMAME 370, 113242 and SoftwareX 12,
100595).

A seed ``s`` is the set of spheres ``first[s] .. first[s + 1]`` of the arrays ``offsets``
(their centres relative to the centre of the seed) and ``radii``; a position ``c`` clears
when every sphere ``k`` of the seed and every sphere ``q`` placed obey
``|c + o_k - x_q| + rtol min(r_k, r_q) >= r_k + r_q``: the depth of their overlap at most
``rtol`` times the smaller radius (0: no overlap; 1: the centre of the smaller sphere on
the surface of the other). A seed that clears at none of its candidates is left out
(MicroStructPy), or placed at the candidate of the least excess,
``max (r_k + r_q - |c + o_k - x_q|) / min(r_k, r_q) - rtol`` over its pairs (``crowded``).
The spheres placed are kept in the linked lists of a grid of cells at least as large as
twice the largest radius (``head[cell]`` the last sphere put in the cell, ``nxt[q]`` the
one before), so that only the 27 neighbouring cells are searched; in a periodic box the
cells and the distances wrap around, in a window that is not periodic the cells are
clamped to its grid.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _excess(offsets, radii, first, s, cx, cy, cz, box, periodic, rtol, cell_h, cell_n,
            head, nxt, pos, stop_at):  # fmt: skip
    """The largest excess of overlap of the seed ``s`` at ``(cx, cy, cz)`` over the spheres
    placed (negative: it clears), searched until it reaches ``stop_at``."""
    worst = -math.inf
    for k in range(first[s], first[s + 1]):
        p0 = cx + offsets[k, 0]
        p1 = cy + offsets[k, 1]
        p2 = cz + offsets[k, 2]
        if periodic:
            p0 = p0 % box[0]
            p1 = p1 % box[1]
            p2 = p2 % box[2]
        ci = min(max(int(p0 / cell_h[0]), 0), cell_n[0] - 1)
        cj = min(max(int(p1 / cell_h[1]), 0), cell_n[1] - 1)
        ck = min(max(int(p2 / cell_h[2]), 0), cell_n[2] - 1)
        ni = min(3, cell_n[0])
        nj = min(3, cell_n[1])
        nk = min(3, cell_n[2])
        for di in range(ni):
            if periodic:
                ii = (ci + di - 1) % cell_n[0] if ni == 3 else di
            else:
                ii = ci + di - 1 if ni == 3 else di
                if ii < 0 or ii >= cell_n[0]:
                    continue
            for dj in range(nj):
                if periodic:
                    jj = (cj + dj - 1) % cell_n[1] if nj == 3 else dj
                else:
                    jj = cj + dj - 1 if nj == 3 else dj
                    if jj < 0 or jj >= cell_n[1]:
                        continue
                for dk in range(nk):
                    if periodic:
                        kk = (ck + dk - 1) % cell_n[2] if nk == 3 else dk
                    else:
                        kk = ck + dk - 1 if nk == 3 else dk
                        if kk < 0 or kk >= cell_n[2]:
                            continue
                    q = head[(ii * cell_n[1] + jj) * cell_n[2] + kk]
                    while q >= 0:
                        d0 = p0 - pos[q, 0]
                        d1 = p1 - pos[q, 1]
                        d2 = p2 - pos[q, 2]
                        if periodic:
                            d0 -= box[0] * np.rint(d0 / box[0])
                            d1 -= box[1] * np.rint(d1 / box[1])
                            d2 -= box[2] * np.rint(d2 / box[2])
                        gap = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
                        small = min(radii[k], radii[q])
                        excess = (radii[k] + radii[q] - gap) / small - rtol
                        if excess > worst:
                            worst = excess
                            if worst >= stop_at:
                                return worst
                        q = nxt[q]
    return worst


@njit(cache=True)
def _put(offsets, first, s, cx, cy, cz, box, periodic, cell_h, cell_n, head, nxt, pos):
    """Put the spheres of the seed ``s`` at ``(cx, cy, cz)`` into the grid."""
    for k in range(first[s], first[s + 1]):
        p0 = cx + offsets[k, 0]
        p1 = cy + offsets[k, 1]
        p2 = cz + offsets[k, 2]
        if periodic:
            p0 = p0 % box[0]
            p1 = p1 % box[1]
            p2 = p2 % box[2]
        pos[k, 0] = p0
        pos[k, 1] = p1
        pos[k, 2] = p2
        ci = min(max(int(p0 / cell_h[0]), 0), cell_n[0] - 1)
        cj = min(max(int(p1 / cell_h[1]), 0), cell_n[1] - 1)
        ck = min(max(int(p2 / cell_h[2]), 0), cell_n[2] - 1)
        cell = (ci * cell_n[1] + cj) * cell_n[2] + ck
        nxt[k] = head[cell]
        head[cell] = k


@njit(cache=True)
def insert_seeds(offsets, radii, first, start, stop, box, periodic, rtol, cell_h, cell_n,
                 head, nxt, pos, centres, placed, crowded, candidates, per_seed, settle,
                 place_crowded):  # fmt: skip
    """Try the ``per_seed`` candidate positions of each seed ``start .. stop - 1`` in turn
    (``candidates[(s - start) * per_seed + t]``), placing it at the first that clears:
    ``centres[s]``, ``placed[s]``, the positions ``pos`` of its spheres (in the box when
    periodic) linked into the grid. A seed that clears at none is settled (``settle``):
    placed at the candidate of the least excess with ``place_crowded`` (``crowded[s]``),
    left out otherwise; without ``settle``, it is left to the caller, to be tried at more
    positions. Returns the first seed not settled (``stop`` once all are)."""
    for s in range(start, stop):
        best = math.inf
        best_t = -1
        cleared = False
        for t in range(per_seed):
            row = (s - start) * per_seed + t
            worst = _excess(offsets, radii, first, s, candidates[row, 0], candidates[row, 1],
                            candidates[row, 2], box, periodic, rtol, cell_h, cell_n, head,
                            nxt, pos, best)  # fmt: skip
            if worst <= 0.0:
                best_t = t
                cleared = True
                break
            if worst < best:
                best = worst
                best_t = t
        if not cleared:
            if not settle:
                return s
            if not place_crowded or best_t < 0:
                continue
            crowded[s] = True
        row = (s - start) * per_seed + best_t
        cx, cy, cz = candidates[row, 0], candidates[row, 1], candidates[row, 2]
        _put(offsets, first, s, cx, cy, cz, box, periodic, cell_h, cell_n, head, nxt, pos)
        centres[s, 0] = cx
        centres[s, 1] = cy
        centres[s, 2] = cz
        placed[s] = True
    return stop


#: The positions first drawn for each seed: most clear at one of them, the others are
#: tried at the rest of their attempts
QUICK = 32


def place_seeds(offsets, radii, first, box, rng, *, periodic: bool, rtol: float,
                attempts: int = 10000, batch: int = 64, crowded: bool = True):  # fmt: skip
    """The centres of the seeds (``first``: the ranges of their spheres in ``offsets`` and
    ``radii``, the seeds in the order of their placement) placed in the box ``[0, box)``
    (see the module), each tried at ``attempts`` uniform positions at most, those that clear
    at none left out or (``crowded``) placed at the least excess: the centres, whether each
    seed was placed, whether it was placed crowded, the positions of all their spheres. The
    positions are drawn :data:`QUICK` per seed for ``batch`` seeds at once, and the rest of
    its attempts for a seed that clears at none of them."""
    offsets = np.ascontiguousarray(offsets, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    first = np.ascontiguousarray(first, dtype=np.int64)
    box = np.asarray(box, dtype=np.float64)
    attempts = int(attempts)
    n = len(first) - 1
    m = len(radii)
    reach = 2.0 * float(radii.max()) if m else 1.0
    cell_n = np.maximum(1, np.floor(box / reach)).astype(np.int64)
    cell_h = box / cell_n
    head = np.full(int(np.prod(cell_n)), -1, dtype=np.int64)
    nxt = np.full(m, -1, dtype=np.int64)
    pos = np.zeros((m, 3))
    centres = np.zeros((n, 3))
    placed = np.zeros(n, dtype=np.bool_)
    crowd = np.zeros(n, dtype=np.bool_)
    quick = min(attempts, QUICK)
    grid = (box, bool(periodic), float(rtol), cell_h, cell_n, head, nxt, pos, centres, placed,
            crowd)  # fmt: skip
    start = 0
    while start < n:
        stop = min(n, start + batch)
        candidates = rng.random(((stop - start) * quick, 3)) * box
        s = insert_seeds(offsets, radii, first, start, stop, *grid, candidates, quick,
                         quick == attempts, bool(crowded))  # fmt: skip
        if s < stop:  # cleared at none of its first positions: all its attempts
            row = (s - start) * quick
            rest = np.vstack([candidates[row : row + quick],
                              rng.random((attempts - quick, 3)) * box])  # fmt: skip
            insert_seeds(offsets, radii, first, s, s + 1, *grid, rest, attempts, True,
                         bool(crowded))  # fmt: skip
            s += 1
        start = s
    return centres, placed, crowd, pos


__all__ = ["insert_seeds", "place_seeds"]
