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

The spheres placed are kept in the linked lists of grids of cells (``head[cell]`` the last
sphere put in the cell, ``nxt[q]`` the one before), one grid per level of sizes: the level
``l`` holds the spheres of radii up to ``r_max / 2^l`` in cells at least twice as large, so
that a sphere of radius ``r`` meets those of a level within ``(r + r_l) / h_l`` cells of its
own, ``r_l`` the largest radius placed there — the small seeds that fill the box last search
a few cells of the fine levels, not the cells of the largest spheres. In a periodic box the
cells and the distances wrap around; in a window that is not periodic the cells are clamped
to its grid.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

#: The most levels of sizes, and the most cells of a level
MAX_LEVELS = 8
MAX_CELLS = 1 << 22


@njit(cache=True)
def _excess(offsets, radii, first, s, cx, cy, cz, box, periodic, rtol, level, lev_h, lev_n,
            lev_off, lev_r, head, nxt, pos, stop_at):  # fmt: skip
    """The largest excess of overlap of the seed ``s`` at ``(cx, cy, cz)`` over the spheres
    placed (negative: it clears), searched until it reaches ``stop_at``: the spheres of the
    level of each sphere of the seed first, then the coarser and the finer ones, the cells
    from its own outwards (where the overlaps that end the search lie)."""
    worst = -math.inf
    levels = len(lev_r)
    for k in range(first[s], first[s + 1]):
        p0 = cx + offsets[k, 0]
        p1 = cy + offsets[k, 1]
        p2 = cz + offsets[k, 2]
        if periodic:
            p0 = p0 % box[0]
            p1 = p1 % box[1]
            p2 = p2 % box[2]
        rk = radii[k]
        own = level[k]
        for t in range(levels):  # own, own - 1, ..., 0, own + 1, ...
            lev = own - t if t <= own else t
            if lev_r[lev] <= 0.0:  # no sphere there yet
                continue
            n0, n1, n2 = lev_n[lev, 0], lev_n[lev, 1], lev_n[lev, 2]
            reach = rk + lev_r[lev]
            r0 = math.ceil(reach / lev_h[lev, 0])
            r1 = math.ceil(reach / lev_h[lev, 1])
            r2 = math.ceil(reach / lev_h[lev, 2])
            c0 = min(max(int(p0 / lev_h[lev, 0]), 0), n0 - 1)
            c1 = min(max(int(p1 / lev_h[lev, 1]), 0), n1 - 1)
            c2 = min(max(int(p2 / lev_h[lev, 2]), 0), n2 - 1)
            all0, all1, all2 = 2 * r0 + 1 >= n0, 2 * r1 + 1 >= n1, 2 * r2 + 1 >= n2
            m0 = n0 if all0 else 2 * r0 + 1
            m1 = n1 if all1 else 2 * r1 + 1
            m2 = n2 if all2 else 2 * r2 + 1
            for a in range(m0):  # from its own cell outwards: 0, 1, -1, 2, -2, ...
                ii = a if all0 else c0 + ((a + 1) // 2 if a % 2 else -(a // 2))
                if not all0:
                    if periodic:
                        ii %= n0
                    elif ii < 0 or ii >= n0:
                        continue
                for b in range(m1):
                    jj = b if all1 else c1 + ((b + 1) // 2 if b % 2 else -(b // 2))
                    if not all1:
                        if periodic:
                            jj %= n1
                        elif jj < 0 or jj >= n1:
                            continue
                    for c in range(m2):
                        kk = c if all2 else c2 + ((c + 1) // 2 if c % 2 else -(c // 2))
                        if not all2:
                            if periodic:
                                kk %= n2
                            elif kk < 0 or kk >= n2:
                                continue
                        q = head[lev_off[lev] + (ii * n1 + jj) * n2 + kk]
                        while q >= 0:
                            d0 = p0 - pos[q, 0]
                            d1 = p1 - pos[q, 1]
                            d2 = p2 - pos[q, 2]
                            if periodic:
                                d0 -= box[0] * np.rint(d0 / box[0])
                                d1 -= box[1] * np.rint(d1 / box[1])
                                d2 -= box[2] * np.rint(d2 / box[2])
                            gap = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
                            small = min(rk, radii[q])
                            excess = (rk + radii[q] - gap) / small - rtol
                            if excess > worst:
                                worst = excess
                                if worst >= stop_at:
                                    return worst
                            q = nxt[q]
    return worst


@njit(cache=True)
def _put(offsets, radii, first, s, cx, cy, cz, box, periodic, level, lev_h, lev_n, lev_off,
         lev_r, head, nxt, pos):  # fmt: skip
    """Put the spheres of the seed ``s`` at ``(cx, cy, cz)`` into the grids of their
    levels."""
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
        lev = level[k]
        n0, n1, n2 = lev_n[lev, 0], lev_n[lev, 1], lev_n[lev, 2]
        ci = min(max(int(p0 / lev_h[lev, 0]), 0), n0 - 1)
        cj = min(max(int(p1 / lev_h[lev, 1]), 0), n1 - 1)
        ck = min(max(int(p2 / lev_h[lev, 2]), 0), n2 - 1)
        cell = lev_off[lev] + (ci * n1 + cj) * n2 + ck
        nxt[k] = head[cell]
        head[cell] = k
        if radii[k] > lev_r[lev]:
            lev_r[lev] = radii[k]


@njit(cache=True)
def insert_seeds(offsets, radii, first, start, stop, box, periodic, rtol, level, lev_h, lev_n,
                 lev_off, lev_r, head, nxt, pos, centres, placed, crowded, candidates, per_seed,
                 settle, place_crowded):  # fmt: skip
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
                            candidates[row, 2], box, periodic, rtol, level, lev_h, lev_n,
                            lev_off, lev_r, head, nxt, pos, best)  # fmt: skip
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
        _put(offsets, radii, first, s, cx, cy, cz, box, periodic, level, lev_h, lev_n, lev_off,
             lev_r, head, nxt, pos)  # fmt: skip
        centres[s, 0] = cx
        centres[s, 1] = cy
        centres[s, 2] = cz
        placed[s] = True
    return stop


def _levels(radii, box):
    """The levels of sizes of the spheres: the level of each sphere (``floor(log2(r_max /
    r))``, the finest for the smallest), and the cells of each level, at least ``2 r_max /
    2^l`` wide (at most :data:`MAX_LEVELS` levels of :data:`MAX_CELLS` cells)."""
    big = float(radii.max()) if len(radii) else 1.0
    small = float(radii.min()) if len(radii) else 1.0
    sizes, counts = [], []
    for lev in range(MAX_LEVELS):
        width = 2.0 * big / 2.0**lev
        n = np.maximum(1, np.floor(box / width)).astype(np.int64)
        if lev and (int(np.prod(n)) > MAX_CELLS or width < small):
            break
        sizes.append(box / n)
        counts.append(n)
    lev_h = np.array(sizes)
    lev_n = np.array(counts, dtype=np.int64)
    lev_off = np.concatenate([[0], np.cumsum(np.prod(lev_n, axis=1))]).astype(np.int64)
    ratio = big / np.maximum(np.asarray(radii, dtype=float), 1e-300)
    level = np.clip(np.floor(np.log2(ratio)), 0, len(lev_h) - 1).astype(np.int64)
    return level, lev_h, lev_n, lev_off


#: The positions first drawn for each seed: most clear at one of them, the others are
#: tried at the rest of their attempts
QUICK = 32


def place_seeds(offsets, radii, first, box, rng, *, periodic: bool, rtol: float,
                attempts: int = 10000, batch: int = 64, crowded: bool = True,
                planar: bool = False):  # fmt: skip
    """The centres of the seeds (``first``: the ranges of their spheres in ``offsets`` and
    ``radii``, the seeds in the order of their placement) placed in the box ``[0, box)``
    (see the module), each tried at ``attempts`` uniform positions at most, those that clear
    at none left out or (``crowded``) placed at the least excess: the centres, whether each
    seed was placed, whether it was placed crowded, the positions of all their spheres. The
    positions are drawn :data:`QUICK` per seed for ``batch`` seeds at once, and the rest of
    its attempts for a seed that clears at none of them; ``planar``: in the middle plane z
    of the box (the seeds of the plane)."""
    offsets = np.ascontiguousarray(offsets, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    first = np.ascontiguousarray(first, dtype=np.int64)
    box = np.asarray(box, dtype=np.float64)
    attempts = int(attempts)
    n = len(first) - 1
    m = len(radii)
    level, lev_h, lev_n, lev_off = _levels(radii, box)
    lev_r = np.zeros(len(lev_h))
    head = np.full(int(lev_off[-1]), -1, dtype=np.int64)
    nxt = np.full(m, -1, dtype=np.int64)
    pos = np.zeros((m, 3))
    centres = np.zeros((n, 3))
    placed = np.zeros(n, dtype=np.bool_)
    crowd = np.zeros(n, dtype=np.bool_)
    quick = min(attempts, QUICK)
    grid = (box, bool(periodic), float(rtol), level, lev_h, lev_n, lev_off, lev_r, head, nxt,
            pos, centres, placed, crowd)  # fmt: skip

    def draw(k):  # uniform positions, in the middle plane for the seeds of the plane
        if not planar:
            return rng.random((k, 3)) * box
        return np.column_stack([rng.random((k, 2)) * box[:2], np.full(k, 0.5 * box[2])])

    start = 0
    while start < n:
        stop = min(n, start + batch)
        candidates = draw((stop - start) * quick)
        s = insert_seeds(offsets, radii, first, start, stop, *grid, candidates, quick,
                         quick == attempts, bool(crowded))  # fmt: skip
        if s < stop:  # cleared at none of its first positions: all its attempts
            row = (s - start) * quick
            rest = np.vstack([candidates[row : row + quick], draw(attempts - quick)])
            insert_seeds(offsets, radii, first, s, s + 1, *grid, rest, attempts, True,
                         bool(crowded))  # fmt: skip
            s += 1
        start = s
    return centres, placed, crowd, pos


__all__ = ["insert_seeds", "place_seeds"]
