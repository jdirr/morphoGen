"""Constraint solvers for tile assemblies.

``wave_function_collapse``
    Random compatible assemblies of any size: arc consistency (each tile left in the
    domain of a cell has a compatible tile in every neighbouring domain), then cells of
    least entropy are given a tile at random (with the tile weights) and the choice is
    propagated, backtracking on contradictions (domains are bit sets, Numba).
``TransferMatrix``
    Exact counting and exact uniform sampling of the compatible assemblies of a box,
    with a transfer matrix between the admissible cross-sections along x (statistical
    physics). It needs few admissible cross-sections: small sections, or constrained
    tile sets.
``count_polarized_blocks``
    Exact number of compatible assemblies of polarized blocks, for larger boxes: the
    strain sign along every line of blocks alternates, which reduces the problem to
    sign variables on the faces of the box (a gauge), summed by dynamic programming.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy import sparse

from morphogen.design.tiles import TileSet

MAX_TILES = 64  # bit sets of 64 bits


# -- wave function collapse ----------------------------------------------------------------


@njit(cache=True)
def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
def _propagate(
    dom,
    sw,
    swl,
    nb,
    allowed,
    w,
    wlogw,
    n_tiles,
    stack,
    n_stack,
    in_stack,
    trail_c,
    trail_d,
    trail_w,
    trail_l,
    n_trail,
):
    """Arc consistency from the cells on the stack; returns (ok, n_trail)."""
    one = np.uint64(1)
    while n_stack > 0:
        n_stack -= 1
        c = stack[n_stack]
        in_stack[c] = 0
        dc = dom[c]
        for d in range(6):
            q = nb[c, d]
            if q < 0:
                continue
            sup = np.uint64(0)
            for t in range(n_tiles):
                if (dc >> np.uint64(t)) & one:
                    sup |= allowed[d, t]
            old = dom[q]
            new = old & sup
            if new == old:
                continue
            if new == 0:
                # leave the stack clean for the next propagation
                while n_stack > 0:
                    n_stack -= 1
                    in_stack[stack[n_stack]] = 0
                return False, n_trail
            trail_c[n_trail] = q
            trail_d[n_trail] = old
            trail_w[n_trail] = sw[q]
            trail_l[n_trail] = swl[q]
            n_trail += 1
            removed = old & ~new
            for t in range(n_tiles):
                if (removed >> np.uint64(t)) & one:
                    sw[q] -= w[t]
                    swl[q] -= wlogw[t]
            dom[q] = new
            if not in_stack[q]:
                stack[n_stack] = q
                n_stack += 1
                in_stack[q] = 1
    return True, n_trail


@njit(cache=True)
def _undo(dom, sw, swl, trail_c, trail_d, trail_w, trail_l, n_trail, target):
    while n_trail > target:
        n_trail -= 1
        c = trail_c[n_trail]
        dom[c] = trail_d[n_trail]
        sw[c] = trail_w[n_trail]
        swl[c] = trail_l[n_trail]
    return n_trail


@njit(cache=True)
def _wfc(dom, nb, allowed, w, n_tiles, prio, rng, max_backtracks):
    """Search from the domains ``dom`` (modified in place). Status: 1 solved, 0 gave up
    (too many backtracks), -1 no solution."""
    n = dom.size
    one = np.uint64(1)
    wlogw = np.zeros(n_tiles)
    for t in range(n_tiles):
        if w[t] > 0:
            wlogw[t] = w[t] * math.log(w[t])
    sw = np.zeros(n)
    swl = np.zeros(n)
    for c in range(n):
        for t in range(n_tiles):
            if (dom[c] >> np.uint64(t)) & one:
                sw[c] += w[t]
                swl[c] += wlogw[t]
    size = n * (n_tiles + 1) + 1
    trail_c = np.empty(size, dtype=np.int64)
    trail_d = np.empty(size, dtype=np.uint64)
    trail_w = np.empty(size)
    trail_l = np.empty(size)
    stack = np.empty(n, dtype=np.int64)
    in_stack = np.zeros(n, dtype=np.uint8)
    n_stack = 0
    for c in range(n):
        if dom[c] == 0:
            return -1
        stack[n_stack] = c
        n_stack += 1
        in_stack[c] = 1
    ok, n_trail = _propagate(
        dom,
        sw,
        swl,
        nb,
        allowed,
        w,
        wlogw,
        n_tiles,
        stack,
        n_stack,
        in_stack,
        trail_c,
        trail_d,
        trail_w,
        trail_l,
        0,
    )
    if not ok:
        return -1
    n_trail = 0  # the initial reduction is never undone
    dec_cell = np.empty(n, dtype=np.int64)
    dec_tried = np.zeros(n, dtype=np.uint64)
    dec_trail = np.zeros(n, dtype=np.int64)
    depth = 0
    backtracks = 0
    while True:
        best = -1
        best_h = np.inf
        for c in range(n):
            if _popcount(dom[c]) > 1:
                h = math.log(sw[c]) - swl[c] / sw[c] + 1e-7 * prio[c]
                if h < best_h:
                    best_h = h
                    best = c
        if best < 0:
            return 1
        dec_cell[depth] = best
        dec_tried[depth] = 0
        dec_trail[depth] = n_trail
        depth += 1
        while True:
            k = depth - 1
            n_trail = _undo(dom, sw, swl, trail_c, trail_d, trail_w, trail_l, n_trail, dec_trail[k])
            c = dec_cell[k]
            options = dom[c] & ~dec_tried[k]
            total = 0.0
            for t in range(n_tiles):
                if (options >> np.uint64(t)) & one:
                    total += w[t]
            if total <= 0.0:
                depth -= 1
                backtracks += 1
                if depth == 0:
                    return -1
                if backtracks > max_backtracks:
                    return 0
                continue
            r = rng.random() * total
            choice = -1
            for t in range(n_tiles):
                if (options >> np.uint64(t)) & one:
                    choice = t
                    r -= w[t]
                    if r < 0.0:
                        break
            bit = one << np.uint64(choice)
            dec_tried[k] |= bit
            trail_c[n_trail] = c
            trail_d[n_trail] = dom[c]
            trail_w[n_trail] = sw[c]
            trail_l[n_trail] = swl[c]
            n_trail += 1
            dom[c] = bit
            sw[c] = w[choice]
            swl[c] = wlogw[choice]
            stack[0] = c
            in_stack[c] = 1
            ok, n_trail = _propagate(
                dom,
                sw,
                swl,
                nb,
                allowed,
                w,
                wlogw,
                n_tiles,
                stack,
                1,
                in_stack,
                trail_c,
                trail_d,
                trail_w,
                trail_l,
                n_trail,
            )
            if ok:
                break


def _neighbours(shape, periodic: bool, dims: int) -> np.ndarray:
    """Index of the neighbour of every cell in the directions -x, +x, -y, +y, -z, +z
    (-1: none)."""
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    nb = np.full((idx.size, 6), -1, dtype=np.int64)
    for a in range(dims):
        for side, step in ((0, 1), (1, -1)):
            rolled = np.roll(idx, step, axis=a)  # side 0: value at i-1; side 1: at i+1
            ok = np.ones(shape, dtype=bool)
            if not periodic:
                edge = [slice(None)] * 3
                edge[a] = 0 if side == 0 else -1
                ok[tuple(edge)] = False
            if shape[a] == 1 and not periodic:
                ok[:] = False
            nb[:, 2 * a + side] = np.where(ok, rolled, -1).ravel()
    return nb


def _allowed_bits(tileset: TileSet) -> np.ndarray:
    """``allowed[d, t]``: bit set of the tiles allowed next to tile t in direction d."""
    n = tileset.n_tiles
    compat = tileset.compatibility()
    bits = (np.uint64(1) << np.arange(n, dtype=np.uint64)).astype(np.uint64)
    allowed = np.zeros((6, n), dtype=np.uint64)
    for a in range(3):
        allowed[2 * a + 1] = np.bitwise_or.reduce(np.where(compat[a], bits[None, :], 0), axis=1)
        allowed[2 * a] = np.bitwise_or.reduce(np.where(compat[a].T, bits[None, :], 0), axis=1)
    return allowed


def initial_domains(
    tileset: TileSet, shape, *, periodic: bool = True, boundary=None, allowed=None
) -> np.ndarray:
    """Boolean domains ``(n0, n1, n2, n_tiles)``: tiles of positive weight, restricted
    by ``allowed`` (same shape) and, on the faces of non-periodic boxes, to tiles whose
    outer faces match the label ``boundary``."""
    shape = tuple(int(n) for n in shape)
    dom = np.broadcast_to(tileset.weights > 0, (*shape, tileset.n_tiles)).copy()
    if allowed is not None:
        dom &= np.asarray(allowed, dtype=bool).reshape(dom.shape)
    if boundary is not None and not periodic:
        for a in range(tileset.dims):
            for side in (0, 1):
                face = [slice(None)] * 3
                face[a] = 0 if side == 0 else shape[a] - 1
                dom[tuple(face)] &= tileset.boundary_tiles(a, side, boundary)
    return dom


def wave_function_collapse(
    tileset: TileSet,
    shape,
    rng: np.random.Generator,
    *,
    periodic: bool = True,
    boundary: int | None = None,
    allowed=None,
    max_backtracks: int | None = None,
    restarts: int = 5,
) -> np.ndarray:
    """A random compatible assembly of ``shape`` tiles (``(n0, n1, 1)`` for planar sets).

    ``allowed`` (boolean, ``shape + (n_tiles,)``) restricts the tiles of each cell, e.g.
    to grade a design in space; ``boundary`` is the label the outer faces of a
    non-periodic box must match. Raises ValueError when no assembly exists, or when the
    search gave up after ``restarts`` attempts of ``max_backtracks`` backtracks.
    """
    if tileset.n_tiles > MAX_TILES:
        raise ValueError(f"wave function collapse handles up to {MAX_TILES} tiles")
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("shape: three positive numbers of tiles")
    if tileset.dims == 2 and shape[2] != 1:
        raise ValueError("planar tile sets: shape (n0, n1, 1)")
    dom_bool = initial_domains(
        tileset, shape, periodic=periodic, boundary=boundary, allowed=allowed
    )
    bits = np.uint64(1) << np.arange(tileset.n_tiles, dtype=np.uint64)
    dom0 = np.bitwise_or.reduce(np.where(dom_bool, bits, np.uint64(0)), axis=-1).ravel()
    nb = _neighbours(shape, periodic, tileset.dims)
    allowed_bits = _allowed_bits(tileset)
    n_cells = dom0.size
    limit = 10 * n_cells + 100 if max_backtracks is None else int(max_backtracks)
    w = tileset.weights.astype(float)
    for _ in range(max(1, int(restarts))):
        dom = dom0.astype(np.uint64).copy()
        prio = rng.random(n_cells)
        status = _wfc(dom, nb, allowed_bits, w, tileset.n_tiles, prio, rng, limit)
        if status == 1:
            tiles = np.array([int(d).bit_length() - 1 for d in dom], dtype=np.int64)
            return tiles.reshape(shape)
        if status == -1:
            raise ValueError("no compatible assembly exists for these tiles and constraints")
    raise ValueError(
        f"no compatible assembly found after {restarts} restarts of {limit} backtracks"
    )


# -- transfer matrices ---------------------------------------------------------------------


@njit(cache=True)
def _enumerate_slices(n1, n2, n_tiles, compat, usable, periodic, max_states, out):
    npos = n1 * n2
    cur = np.full(npos, -1, dtype=np.int64)
    count = 0
    p = 0
    while p >= 0:
        cur[p] += 1
        if cur[p] >= n_tiles:
            cur[p] = -1
            p -= 1
            continue
        t = cur[p]
        if not usable[t]:
            continue
        j = p // n2
        k = p % n2
        if j > 0 and not compat[1, cur[p - n2], t]:
            continue
        if k > 0 and not compat[2, cur[p - 1], t]:
            continue
        if periodic and j == n1 - 1 and not compat[1, t, cur[k]]:
            continue
        if periodic and k == n2 - 1 and not compat[2, t, cur[j * n2]]:
            continue
        if p == npos - 1:
            if count >= max_states:
                return -1
            for q in range(npos):
                out[count, q] = cur[q]
            count += 1
            continue
        p += 1
    return count


class TransferMatrix:
    """Admissible cross-sections ``(n1, n2)`` of a tile set and the transfer matrix
    between consecutive sections along x (``T[s, s'] = 1`` when section s' may follow
    section s).

    Counting and sampling use exact integer arithmetic. Periodic boxes need the closed
    walks of the transfer matrix (one sum per section): keep the number of sections
    modest (``max_states``).
    """

    def __init__(
        self, tileset: TileSet, section, *, periodic: bool = False, max_states: int = 20000
    ):
        self.tileset = tileset
        self.section = (int(section[0]), int(section[1]))
        if tileset.dims == 2 and self.section[1] != 1:
            raise ValueError("planar tile sets: sections (n1, 1)")
        self.periodic = bool(periodic)
        n1, n2 = self.section
        compat = tileset.compatibility()
        out = np.empty((max_states, n1 * n2), dtype=np.int64)
        count = _enumerate_slices(
            n1,
            n2,
            tileset.n_tiles,
            compat,
            tileset.weights > 0,
            self.periodic,
            int(max_states),
            out,
        )
        if count < 0:
            raise ValueError(
                f"more than {max_states} admissible sections: the box is too wide for "
                "exact counting (use wave_function_collapse to sample)"
            )
        self.states = out[:count].copy()
        self.matrix = self._build(compat[0])

    def _build(self, compat_x) -> sparse.csr_matrix:
        s = self.states
        n = len(s)
        rows, cols = [], []
        block = max(1, 2_000_000 // max(1, n))
        for start in range(0, n, block):
            ok = np.ones((min(block, n - start), n), dtype=bool)
            for c in range(s.shape[1]):
                ok &= compat_x[s[start : start + block, c]][:, s[:, c]]
            r, q = np.nonzero(ok)
            rows.append(r + start)
            cols.append(q)
        r = np.concatenate(rows) if rows else np.empty(0, dtype=int)
        q = np.concatenate(cols) if cols else np.empty(0, dtype=int)
        return sparse.csr_matrix((np.ones(len(r), dtype=np.int8), (r, q)), shape=(n, n))

    @property
    def n_states(self) -> int:
        return len(self.states)

    def _step(self, v: np.ndarray) -> np.ndarray:
        """(T v)[s] = sum of v over the sections s' allowed after s (exact integers)."""
        m = self.matrix
        out = np.zeros(len(v), dtype=object)
        nonempty = np.diff(m.indptr) > 0
        if np.any(nonempty):
            out[nonempty] = np.add.reduceat(v[m.indices], m.indptr[:-1][nonempty])
        return out

    def _backward(self, length: int, end=None) -> list[np.ndarray]:
        """b[k][s]: number of ways to complete the sections k+1..length-1 after section s
        at position k (ending at section ``end`` when given)."""
        n = self.n_states
        last = np.ones(n, dtype=object)
        if end is not None:
            last = np.zeros(n, dtype=object)
            # completions must end with a section followed by `end` (periodic closure)
            col = self.matrix[:, end].toarray().ravel()
            last[col > 0] = 1
        b = [last]
        for _ in range(length - 1):
            b.append(self._step(b[-1]))
        return b[::-1]

    def count(self, length: int) -> int:
        """Number of compatible assemblies of ``length`` sections (exact)."""
        length = int(length)
        if length < 1:
            raise ValueError("length: at least 1")
        if not self.periodic:
            return int(self._backward(length)[0].sum())
        return int(sum(self._closed_walks(length)))

    def _closed_walks(self, length: int) -> list[int]:
        """Number of periodic assemblies starting with each section."""
        out = []
        for s in range(self.n_states):
            b = self._backward(length, end=s)
            out.append(int(b[0][s]))
        return out

    def sample(self, length: int, rng: np.random.Generator) -> np.ndarray:
        """An assembly ``(length, n1, n2)`` drawn uniformly among the compatible ones."""
        length = int(length)
        if self.periodic:
            weights = self._closed_walks(length)
            first = _choose(weights, rng)
            b = self._backward(length, end=first)
        else:
            b = self._backward(length)
            first = _choose(list(b[0]), rng)
        path = [first]
        m = self.matrix
        for k in range(1, length):
            s = path[-1]
            nxt = m.indices[m.indptr[s] : m.indptr[s + 1]]
            path.append(int(nxt[_choose([b[k][q] for q in nxt], rng)]))
        n1, n2 = self.section
        return self.states[np.array(path)].reshape(length, n1, n2)


def _choose(weights, rng: np.random.Generator) -> int:
    """Index drawn with probability proportional to non-negative integer weights."""
    total = int(sum(int(w) for w in weights))
    if total <= 0:
        raise ValueError("no compatible assembly")
    nbits = total.bit_length()
    while True:  # uniform integer in [0, total) by rejection
        r = int.from_bytes(rng.bytes((nbits + 7) // 8), "little") >> ((8 - nbits % 8) % 8)
        if r < total:
            break
    for i, w in enumerate(weights):
        r -= int(w)
        if r < 0:
            return i
    return len(weights) - 1  # pragma: no cover


def count_assemblies(
    tileset: TileSet, shape, *, periodic: bool = False, max_states: int = 20000
) -> int:
    """Exact number of compatible assemblies of a box of ``shape`` tiles."""
    shape = tuple(int(n) for n in shape)
    tm = TransferMatrix(tileset, shape[1:], periodic=periodic, max_states=max_states)
    return tm.count(shape[0])


def sample_uniform(
    tileset: TileSet,
    shape,
    rng: np.random.Generator,
    *,
    periodic: bool = False,
    max_states: int = 20000,
) -> np.ndarray:
    """A compatible assembly drawn uniformly at random (exact, by transfer matrix)."""
    shape = tuple(int(n) for n in shape)
    tm = TransferMatrix(tileset, shape[1:], periodic=periodic, max_states=max_states)
    return tm.sample(shape[0], rng)


# -- polarized blocks ----------------------------------------------------------------------


@njit(cache=True)
def _polarized_sum(nx, ny, nz):
    """Sum over the signs F_x(j, k) of R_0^ceil(nx/2) R_1^floor(nx/2) (see below)."""
    n_fx = 1 << (ny * nz)
    total = 0.0
    exact = np.int64(0)
    big = False
    for fx in range(n_fx):
        r = np.zeros(2, dtype=np.int64)
        for parity in range(2):
            acc = np.int64(0)
            for y in range(1 << nz):
                prod = np.int64(1)
                for j in range(ny):
                    # c = 2 - |{X_k (-1)^k : X_k = Y_k}| for the column (i, j)
                    seen_p = False
                    seen_m = False
                    for k in range(nz):
                        xk = 1 if (fx >> (j * nz + k)) & 1 else -1
                        if parity:
                            xk = -xk
                        yk = 1 if (y >> k) & 1 else -1
                        if j % 2 == 1:
                            yk = -yk
                        if xk == yk:
                            v = xk if k % 2 == 0 else -xk
                            if v > 0:
                                seen_p = True
                            else:
                                seen_m = True
                    c = 2 - (1 if seen_p else 0) - (1 if seen_m else 0)
                    prod *= c
                    if prod == 0:
                        break
                acc += prod
            r[parity] = acc
        term = 1.0
        term_exact = np.int64(1)
        for i in range(nx):
            term *= float(r[i % 2])
            term_exact *= r[i % 2]
        total += term
        exact += term_exact
        if total > 2.0**62:
            big = True
    return exact, total, big


def count_polarized_blocks(shape, *, periodic: bool = False) -> int:
    """Number of compatible assemblies of polarized blocks in a box of ``shape`` blocks.

    Writing the strain signs of the block (i, j, k) as ``e_x = F_x(j, k) (-1)^i``,
    ``e_y = F_y(i, k) (-1)^j``, ``e_z = F_z(i, j) (-1)^k`` (alternation along every
    line), an assembly is a choice of the sign arrays such that no block has three equal
    signs. The sum over ``F_z`` factorizes over the columns, then over ``F_y`` over the
    rows i, which leaves a sum over ``F_x`` (``2^(ny nz)`` terms, the smallest section
    is chosen). Periodic boxes need even numbers of blocks. Exact up to 2^62.
    """
    nx, ny, nz = sorted((int(n) for n in shape), reverse=True)
    if min(nx, ny, nz) < 1:
        raise ValueError("shape: positive numbers of blocks")
    if periodic and any(n % 2 for n in (nx, ny, nz)):
        return 0
    if ny * nz > 24:
        raise ValueError("box too wide: the two smallest edges must hold at most 24 blocks")
    exact, total, big = _polarized_sum(nx, ny, nz)
    if big:
        raise OverflowError(f"about {total:.4g} assemblies: beyond exact 64-bit counting")
    return int(exact)
