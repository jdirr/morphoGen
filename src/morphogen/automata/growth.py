"""Lattice growth processes: nucleation and growth, grain growth and aggregation.

Nucleation and growth
    Nuclei appear at random cells, all at once (site saturation) or at a constant rate
    in the untransformed volume (continuous nucleation), and grow at a constant velocity
    until they impinge on each other (Johnson-Mehl-Avrami-Kolmogorov kinetics). The
    automaton is event-driven: each cell is transformed by the first grain reaching it,
    and grains are propagated from cell to neighbouring cell (26 neighbours) with the
    position of their nucleus, so that arrival times are exact distances in the growth
    metric (spheres, or cubes and octahedra oriented by the crystal orientation of each
    grain), without the anisotropy of lattice distances. Grains are star-shaped with
    respect to their nucleus, which makes the propagation exact up to the resolution of
    the lattice. Nuclei falling in transformed cells are discarded (phantom nuclei).
Grain growth
    Monte Carlo Potts model (Anderson, Srolovitz and Grest 1984): a cell takes the label
    of a random neighbour if this does not increase the number of unlike neighbours (or
    with the Boltzmann probability at temperature T > 0). Grain boundaries move with
    their mean curvature: the mean grain size grows as the square root of time
    (parabolic growth). Cells are swept by independent sublattices in parallel, with
    counter-based random numbers (results do not depend on the number of threads).
Aggregation
    Particles join a growing aggregate: clusters grown from seeds, or deposits on a
    substrate (the face z = 0; y = 0 in 2D). Diffusion-limited aggregation (random
    walkers, Witten and Sander 1981) gives fractal clusters of dimension about 2.5 in 3D
    and 1.71 in 2D; Eden growth (random perimeter cells) compact clusters with rough
    surfaces; ballistic aggregation (straight trajectories) dense porous aggregates.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from numba import njit, prange

#: Growth metrics of nucleation and growth.
GROWTH_SHAPES = ("sphere", "cube", "octahedron")
#: Aggregation processes and geometries.
AGGREGATION_PROCESSES = ("dla", "eden", "ballistic")
AGGREGATION_GEOMETRIES = ("cluster", "deposit")


def shape_volume_coefficient(shape: str, dims: int = 3) -> float:
    """Volume (area in 2D) of the growth shape of radius 1: ``V(r) = c r^d``."""
    if dims == 3:
        return {"sphere": 4.0 * math.pi / 3.0, "cube": 8.0, "octahedron": 4.0 / 3.0}[shape]
    return {"sphere": math.pi, "cube": 4.0, "octahedron": 2.0}[shape]


# -- nucleation and growth -----------------------------------------------------------------


@njit(cache=True)
def _metric(d0, d1, d2, rot, l, use_rot, norm):
    if use_rot:
        e0 = rot[l, 0, 0] * d0 + rot[l, 0, 1] * d1 + rot[l, 0, 2] * d2
        e1 = rot[l, 1, 0] * d0 + rot[l, 1, 1] * d1 + rot[l, 1, 2] * d2
        e2 = rot[l, 2, 0] * d0 + rot[l, 2, 1] * d1 + rot[l, 2, 2] * d2
    else:
        e0, e1, e2 = d0, d1, d2
    if norm == 0:
        return math.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
    if norm == 1:
        return max(abs(e0), abs(e1), abs(e2))
    return abs(e0) + abs(e1) + abs(e2)


@njit(cache=True)
def _grow(nx, ny, nz, h, cells, times, rot, use_rot, norm, rate, periodic, dims, labels, arrival):
    n_offs = 26 if dims == 3 else 8
    offs = np.empty((n_offs, 3), dtype=np.int64)
    m = 0
    for a in range(-1, 2):
        for b in range(-1, 2):
            for c in range(-1, 2):
                if (a == 0 and b == 0 and c == 0) or (dims == 2 and c != 0):
                    continue
                offs[m, 0] = a
                offs[m, 1] = b
                offs[m, 2] = c
                m += 1
    lx, ly, lz = nx * h[0], ny * h[1], nz * h[2]
    n_cells = nx * ny * nz
    best = np.full(n_cells, np.inf)
    done = np.zeros(n_cells, dtype=np.uint8)
    real = np.zeros(len(times), dtype=np.uint8)
    heap = [(0.0, np.int64(0), np.int64(0))]
    heap.pop()
    for i in range(len(times)):
        c = (cells[i, 0] * ny + cells[i, 1]) * nz + cells[i, 2]
        heapq.heappush(heap, (times[i], c, np.int64(-i - 1)))
    while len(heap) > 0:
        t, c, o = heapq.heappop(heap)
        if done[c]:
            continue  # reached earlier (a nucleus there is a phantom)
        g = o
        if o < 0:
            g = -o - 1
            real[g] = 1
        done[c] = 1
        labels[c] = g
        arrival[c] = t
        i = c // (ny * nz)
        j = (c // nz) % ny
        k = c % nz
        for m in range(n_offs):
            ii = i + offs[m, 0]
            jj = j + offs[m, 1]
            kk = k + offs[m, 2]
            if periodic:
                ii %= nx
                jj %= ny
                kk %= nz
            elif ii < 0 or ii >= nx or jj < 0 or jj >= ny or kk < 0 or kk >= nz:
                continue
            cn = (ii * ny + jj) * nz + kk
            if done[cn]:
                continue
            d0 = (ii - cells[g, 0]) * h[0]
            d1 = (jj - cells[g, 1]) * h[1]
            d2 = (kk - cells[g, 2]) * h[2]
            if periodic:
                d0 -= lx * np.rint(d0 / lx)
                d1 -= ly * np.rint(d1 / ly)
                d2 -= lz * np.rint(d2 / lz)
            tn = times[g] + _metric(d0, d1, d2, rot, g, use_rot, norm) / rate
            if tn < best[cn]:
                best[cn] = tn
                heapq.heappush(heap, (tn, cn, g))
    return real


def grow_grains(
    shape,
    cell_size,
    nuclei,
    times,
    *,
    growth_rate: float = 1.0,
    growth_shape: str = "sphere",
    rotations=None,
    periodic: bool = True,
    dims: int = 3,
):
    """Transformation of a lattice by grains growing from nuclei.

    Parameters
    ----------
    shape, cell_size
        Lattice ``(nx, ny, nz)`` (``nz = 1`` in 2D) and edge(s) of its cells.
    nuclei, times
        Cell indices ``(n, 3)`` of the nuclei and their nucleation times.
    growth_rate
        Radius gained by a grain per unit time (half-edge of cubes, half-diagonal of
        octahedra).
    growth_shape
        ``sphere`` (isotropic growth), ``cube`` or ``octahedron`` (faceted growth).
    rotations
        Optional rotation matrices ``(n, 3, 3)`` (crystal = g @ sample) orienting the
        faceted shapes.

    Returns
    -------
    tuple
        ``(labels, arrival, real)``: the index of the nucleus of the grain of each cell
        (int32), the time at which each cell is transformed (inf if never), and whether
        each nucleus formed a grain (False for phantom nuclei, which fell in transformed
        cells).
    """
    if growth_shape not in GROWTH_SHAPES:
        raise ValueError(f"growth_shape: one of {GROWTH_SHAPES}, got {growth_shape!r}")
    nx, ny, nz = (int(n) for n in shape)
    h = np.broadcast_to(np.asarray(cell_size, dtype=float), (3,)).copy()
    cells = np.ascontiguousarray(nuclei, dtype=np.int64).reshape(-1, 3)
    t = np.ascontiguousarray(times, dtype=float).reshape(-1)
    if len(cells) != len(t):
        raise ValueError("nuclei and times: same length")
    if not growth_rate > 0:
        raise ValueError("growth_rate: must be positive")
    use_rot = rotations is not None
    rot = (
        np.ascontiguousarray(rotations, dtype=float).reshape(-1, 3, 3)
        if use_rot
        else np.zeros((1, 3, 3))
    )
    labels = np.full(nx * ny * nz, -1, dtype=np.int32)
    arrival = np.full(nx * ny * nz, np.inf)
    norm = GROWTH_SHAPES.index(growth_shape)
    real = _grow(
        nx,
        ny,
        nz,
        h,
        cells,
        t,
        rot,
        use_rot,
        norm,
        float(growth_rate),
        bool(periodic),
        int(dims),
        labels,
        arrival,
    )
    return labels.reshape(nx, ny, nz), arrival.reshape(nx, ny, nz), real.astype(bool)


# -- grain growth --------------------------------------------------------------------------


@njit(cache=True)
def _mix(x):
    """splitmix64 finalizer (a bijective hash of 64-bit integers)."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@njit(cache=True)
def _uniform(seed, a, b):
    """Uniform number in [0, 1) depending only on (seed, a, b)."""
    x = _mix(np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15) * np.uint64(a + 1))
    x = _mix(x ^ np.uint64(b))
    return (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _colour(i, n, c):
    """Colour of index i along an axis of n cells: parity, and 2 for the last cell of
    odd periodic axes, so that neighbours never share a colour."""
    if n % 2 == 1 and i == n - 1:
        return 2 == c
    return i % 2 == c


@njit(parallel=True, cache=True)
def _potts_sweep(labels, offs, temperature, seed, step, colour):
    nx, ny, nz = labels.shape
    n_offs = offs.shape[0]
    cx = colour // 9
    cy = (colour // 3) % 3
    cz = colour % 3
    for i in prange(nx):
        if not _colour(i, nx, cx):
            continue
        for j in range(ny):
            if not _colour(j, ny, cy):
                continue
            for k in range(nz):
                if not _colour(k, nz, cz):
                    continue
                cell = (i * ny + j) * nz + k
                r = _uniform(seed, 2 * step, cell)
                m = int(r * n_offs)
                q = labels[(i + offs[m, 0]) % nx, (j + offs[m, 1]) % ny, (k + offs[m, 2]) % nz]
                s = labels[i, j, k]
                if q == s:
                    continue
                de = 0
                for o in range(n_offs):
                    v = labels[(i + offs[o, 0]) % nx, (j + offs[o, 1]) % ny, (k + offs[o, 2]) % nz]
                    de += (v != q) - (v != s)
                if de <= 0 or (
                    temperature > 0
                    and _uniform(seed, 2 * step + 1, cell) < math.exp(-de / temperature)
                ):
                    labels[i, j, k] = q


def potts_coarsening(
    labels, steps: int, *, temperature: float = 0.5, seed: int = 0, dims: int = 3, callback=None
) -> np.ndarray:
    """Normal grain growth of a periodic label image by ``steps`` Monte Carlo sweeps of
    the Potts model (Moore neighbourhood: 26 neighbours in 3D, 8 in 2D).

    ``temperature`` is in units of the energy of a pair of unlike neighbours; small
    positive values (about 0.5) avoid the pinning of boundaries by the lattice.
    ``callback(step, labels)`` is called after every sweep.
    """
    from morphogen.automata.lattice import offsets

    lab = np.ascontiguousarray(labels, dtype=np.int32).copy()
    if lab.ndim != 3:
        raise ValueError("labels: a 3D array (nx, ny, 1) in 2D")
    if np.any(lab < 0):
        raise ValueError("grain growth needs a fully labelled lattice (no label -1)")
    if steps < 0 or temperature < 0:
        raise ValueError("steps and temperature: non-negative")
    offs = offsets("moore", 1, dims=dims)
    for step in range(int(steps)):
        for colour in range(27):
            _potts_sweep(lab, offs, float(temperature), int(seed), step * 27 + colour, colour)
        if callback is not None:
            callback(step + 1, lab)
    return lab


# -- aggregation ---------------------------------------------------------------------------

_DLA, _EDEN, _BALLISTIC = 0, 1, 2


@njit(cache=True)
def _unit_vector(rng, dims):
    while True:
        v0 = 2.0 * rng.random() - 1.0
        v1 = 2.0 * rng.random() - 1.0
        v2 = 2.0 * rng.random() - 1.0 if dims == 3 else 0.0
        r2 = v0 * v0 + v1 * v1 + v2 * v2
        if 1e-6 < r2 <= 1.0:
            r = math.sqrt(r2)
            return v0 / r, v1 / r, v2 / r


@njit(cache=True)
def _wrap(p, n, deposit, g):
    for a in range(3):
        if deposit and a == g:
            continue
        p[a] %= n[a]


@njit(cache=True)
def _inside(p, n):
    return 0 <= p[0] < n[0] and 0 <= p[1] < n[1] and 0 <= p[2] < n[2]


@njit(cache=True)
def _centre_distance(p, centre, n, dims):
    s = 0.0
    for a in range(dims):
        d = p[a] - centre[a]
        d -= n[a] * np.rint(d / n[a])
        s += d * d
    return math.sqrt(s)


@njit(cache=True)
def _occupy(p, occ, owner, order, contact, dirs, n, deposit, g, index, n_trees):
    """Add the cell p to the aggregate; returns the updated number of trees."""
    label = -1
    for d in range(dirs.shape[0]):
        q0 = p[0] + dirs[d, 0]
        q1 = p[1] + dirs[d, 1]
        q2 = p[2] + dirs[d, 2]
        if deposit and ((g == 2 and (q2 < 0 or q2 >= n[2])) or (g == 1 and (q1 < 0 or q1 >= n[1]))):
            continue
        q0 %= n[0]
        q1 %= n[1]
        q2 %= n[2]
        if occ[q0, q1, q2] and label < 0:
            label = owner[q0, q1, q2]
        contact[q0, q1, q2] += 1
    if label < 0:
        label = n_trees
        n_trees += 1
    occ[p[0], p[1], p[2]] = 1
    owner[p[0], p[1], p[2]] = label
    order[p[0], p[1], p[2]] = index
    return n_trees


@njit(cache=True)
def _aggregate(occ, owner, order, process, deposit, n_target, sticking, dims, single, rng):
    nx, ny, nz = occ.shape
    n = np.array([nx, ny, nz], dtype=np.int64)
    g = 2 if dims == 3 else 1  # growth axis of deposits
    nd = 2 * dims
    dirs = np.zeros((nd, 3), dtype=np.int64)
    for a in range(dims):
        dirs[2 * a, a] = 1
        dirs[2 * a + 1, a] = -1
    contact = np.zeros(occ.shape, dtype=np.int16)
    n_trees = 0
    centre = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if occ[i, j, k]:
                    n_trees = max(n_trees, owner[i, j, k] + 1)
                    centre[0] = i
                    centre[1] = j
                    centre[2] = k
                    for d in range(nd):
                        q = np.array([i + dirs[d, 0], j + dirs[d, 1], k + dirs[d, 2]])
                        if deposit and (q[g] < 0 or q[g] >= n[g]):
                            continue
                        _wrap(q, n, False, g)
                        contact[q[0], q[1], q[2]] += 1
    top = -1
    if deposit:
        for i in range(nx):
            for j in range(ny if g == 2 else 1):
                if g == 2:
                    contact[i, j, 0] += 1
                else:
                    contact[i, 0, 0] += 1
    r_max = 0.0
    lateral = max(n[0], n[1]) if g == 2 else n[0]
    r_cap = 0.5 * min(n[0], n[1], n[2] if dims == 3 else n[1]) - 2.0
    added = 0
    index = 1
    p = np.zeros(3, dtype=np.int64)
    q = np.zeros(3, dtype=np.int64)
    # Eden: perimeter list
    perim = np.empty(0, dtype=np.int64)
    in_perim = np.zeros(0, dtype=np.uint8)
    n_perim = 0
    if process == _EDEN:
        perim = np.empty(nx * ny * nz, dtype=np.int64)
        in_perim = np.zeros(nx * ny * nz, dtype=np.uint8)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if not occ[i, j, k] and contact[i, j, k] > 0:
                        c = (i * ny + j) * nz + k
                        perim[n_perim] = c
                        in_perim[c] = 1
                        n_perim += 1
    while True:
        if deposit and top >= n[g] - 1:
            break
        if n_target >= 0 and added >= n_target:
            break
        if process == _EDEN:
            if n_perim == 0:
                break
            r = rng.integers(0, n_perim)
            c = perim[r]
            n_perim -= 1
            perim[r] = perim[n_perim]
            in_perim[c] = 0
            p[0] = c // (ny * nz)
            p[1] = (c // nz) % ny
            p[2] = c % nz
            n_trees = _occupy(p, occ, owner, order, contact, dirs, n, deposit, g, index, n_trees)
            index += 1
            added += 1
            top = max(top, p[g])
            for d in range(nd):
                for a in range(3):
                    q[a] = p[a] + dirs[d, a]
                if deposit and (q[g] < 0 or q[g] >= n[g]):
                    continue
                _wrap(q, n, False, g)
                cq = (q[0] * ny + q[1]) * nz + q[2]
                if not occ[q[0], q[1], q[2]] and not in_perim[cq]:
                    perim[n_perim] = cq
                    in_perim[cq] = 1
                    n_perim += 1
            continue
        # launch a particle
        launched = False
        attempts = 0
        sphere = single and not deposit and r_max + 3.0 < r_cap
        u0, u1, u2 = 0.0, 0.0, 0.0
        x0, x1, x2 = 0.0, 0.0, 0.0
        while not launched:
            attempts += 1
            if deposit:
                for a in range(3):
                    p[a] = rng.integers(0, n[a]) if a != g and a < dims else 0
                p[g] = top + 2
                if process == _BALLISTIC:
                    u0, u1, u2 = 0.0, 0.0, 0.0
                    if g == 2:
                        u2 = -1.0
                    else:
                        u1 = -1.0
            elif sphere:
                rl = r_max + 3.0
                v0, v1, v2 = _unit_vector(rng, dims)
                if process == _BALLISTIC:
                    # straight line of direction -v and random impact parameter
                    w0, w1, w2 = _unit_vector(rng, dims)
                    dot = w0 * v0 + w1 * v1 + w2 * v2
                    w0 -= dot * v0
                    w1 -= dot * v1
                    w2 -= dot * v2
                    wn = math.sqrt(w0 * w0 + w1 * w1 + w2 * w2)
                    if wn < 1e-9:
                        continue
                    b = rl * (math.sqrt(rng.random()) if dims == 3 else rng.random())
                    x0 = centre[0] + (rl + 1.0) * v0 + b * w0 / wn
                    x1 = centre[1] + (rl + 1.0) * v1 + b * w1 / wn
                    x2 = centre[2] + (rl + 1.0) * v2 + b * w2 / wn
                    u0, u1, u2 = -v0, -v1, -v2
                    p[0] = math.floor(x0 + 0.5)
                    p[1] = math.floor(x1 + 0.5)
                    p[2] = math.floor(x2 + 0.5)
                else:
                    p[0] = int(np.rint(centre[0] + rl * v0))
                    p[1] = int(np.rint(centre[1] + rl * v1))
                    p[2] = int(np.rint(centre[2] + rl * v2))
                _wrap(p, n, False, g)
            else:
                p[0] = rng.integers(0, nx)
                p[1] = rng.integers(0, ny)
                p[2] = rng.integers(0, nz)
                if process == _BALLISTIC:
                    u0, u1, u2 = _unit_vector(rng, dims)
            if _inside(p, n):
                if occ[p[0], p[1], p[2]]:
                    continue
                if contact[p[0], p[1], p[2]] > 0 and attempts < 1000:
                    continue
            launched = True
            if process == _BALLISTIC and not sphere:
                x0, x1, x2 = float(p[0]), float(p[1]), float(p[2])
        stuck = False
        steps = 0
        if process == _DLA:
            z_kill = top + 2 + max(4, lateral // 2)
            while not stuck:
                if _inside(p, n) and contact[p[0], p[1], p[2]] > 0:
                    if sticking >= 1.0 or rng.random() < sticking:
                        stuck = True
                        break
                if sphere:
                    dist = _centre_distance(p, centre, n, dims)
                    if dist > min(2.0 * (r_max + 3.0) + 10.0, r_cap + 1.0):
                        break  # killed: launch again
                    if dist > r_max + 6.0:
                        rho = dist - r_max - 3.0
                        v0, v1, v2 = _unit_vector(rng, dims)
                        p[0] = int(np.rint(p[0] + rho * v0))
                        p[1] = int(np.rint(p[1] + rho * v1))
                        p[2] = int(np.rint(p[2] + rho * v2))
                        _wrap(p, n, False, g)
                        continue
                d = rng.integers(0, nd)
                for a in range(3):
                    q[a] = p[a] + dirs[d, a]
                if deposit:
                    if q[g] < 0:
                        continue
                    if q[g] > z_kill:
                        break  # wandered away: launch again
                _wrap(q, n, deposit, g)
                if _inside(q, n) and occ[q[0], q[1], q[2]]:
                    continue
                for a in range(3):
                    p[a] = q[a]
        else:  # ballistic: straight flight (voxel traversal)
            max_steps = 4 * (nx + ny + nz)
            step0 = 1 if u0 > 0 else -1
            step1 = 1 if u1 > 0 else -1
            step2 = 1 if u2 > 0 else -1
            c0, c1, c2 = p[0], p[1], p[2]
            fx0 = x0 - (math.floor(x0 + 0.5) - 0.5)  # position inside the cell, in [0, 1)
            fx1 = x1 - (math.floor(x1 + 0.5) - 0.5)
            fx2 = x2 - (math.floor(x2 + 0.5) - 0.5)
            inf = 1e300
            t0 = ((1.0 - fx0) / u0 if u0 > 0 else fx0 / -u0) if u0 != 0 else inf
            t1 = ((1.0 - fx1) / u1 if u1 > 0 else fx1 / -u1) if u1 != 0 else inf
            t2 = ((1.0 - fx2) / u2 if u2 > 0 else fx2 / -u2) if u2 != 0 else inf
            dt0 = 1.0 / abs(u0) if u0 != 0 else inf
            dt1 = 1.0 / abs(u1) if u1 != 0 else inf
            dt2 = 1.0 / abs(u2) if u2 != 0 else inf
            p[0], p[1], p[2] = c0, c1, c2
            while steps < max_steps:
                steps += 1
                if _inside(p, n) and contact[p[0], p[1], p[2]] > 0:
                    if sticking >= 1.0 or rng.random() < sticking:
                        stuck = True
                        break
                for a in range(3):
                    q[a] = p[a]
                if t0 <= t1 and t0 <= t2:
                    q[0] += step0
                    t0 += dt0
                elif t1 <= t2:
                    q[1] += step1
                    t1 += dt1
                else:
                    q[2] += step2
                    t2 += dt2
                if deposit and q[g] < 0:
                    stuck = _inside(p, n)
                    break
                _wrap(q, n, deposit, g)
                if _inside(q, n) and occ[q[0], q[1], q[2]]:
                    stuck = _inside(p, n)  # blocked: stays where it is
                    break
                for a in range(3):
                    p[a] = q[a]
                if (
                    sphere
                    and _centre_distance(p, centre, n, dims) > r_max + 6.0
                    and steps > 4 * (r_max + 4.0)
                ):
                    break  # missed the cluster
        if stuck and _inside(p, n) and not occ[p[0], p[1], p[2]]:
            n_trees = _occupy(p, occ, owner, order, contact, dirs, n, deposit, g, index, n_trees)
            index += 1
            added += 1
            top = max(top, p[g])
            if single:
                r_max = max(r_max, _centre_distance(p, centre, n, dims))
    return added


def aggregate(
    shape,
    rng: np.random.Generator,
    *,
    process: str = "dla",
    geometry: str = "cluster",
    n_seeds: int = 1,
    n_particles: int | None = None,
    sticking: float = 1.0,
    dims: int = 3,
):
    """Grow an aggregate on a lattice.

    Parameters
    ----------
    shape
        Lattice ``(nx, ny, nz)`` (``nz = 1`` in 2D). Clusters live on a periodic
        lattice; deposits on a lattice periodic along the substrate (the face z = 0, or
        y = 0 in 2D).
    process
        ``dla`` (random walkers), ``eden`` (random perimeter cells) or ``ballistic``
        (straight trajectories: random directions for clusters, normal to the substrate
        for deposits).
    geometry
        ``cluster``: growth from ``n_seeds`` random cells (one seed: the centre of the
        lattice, walkers launched from a sphere around the cluster); ``deposit``:
        growth from the substrate.
    n_particles
        Number of particles to add (default for deposits: until the deposit reaches the
        opposite face).
    sticking
        Probability that a particle in contact with the aggregate sticks to it (dla,
        ballistic): lower values give denser aggregates.

    Returns
    -------
    occupied : numpy.ndarray
        The aggregate (uint8).
    owner : numpy.ndarray
        The index of the seed or substrate tree of each particle (int32, -1 elsewhere).
    order : numpy.ndarray
        The order of arrival of each particle (int32: 0 for seeds, -1 elsewhere).
    """
    if process not in AGGREGATION_PROCESSES:
        raise ValueError(f"process: one of {AGGREGATION_PROCESSES}, got {process!r}")
    if geometry not in AGGREGATION_GEOMETRIES:
        raise ValueError(f"geometry: one of {AGGREGATION_GEOMETRIES}, got {geometry!r}")
    if not 0.0 < sticking <= 1.0:
        raise ValueError("sticking: must lie in (0, 1]")
    shape = tuple(int(n) for n in shape)
    occ = np.zeros(shape, dtype=np.uint8)
    owner = np.full(shape, -1, dtype=np.int32)
    order = np.full(shape, -1, dtype=np.int32)
    deposit = geometry == "deposit"
    single = not deposit and n_seeds == 1
    if not deposit:
        if n_seeds < 1:
            raise ValueError("n_seeds: at least 1")
        if single:
            cells = np.array([[n // 2 for n in shape]])
        else:
            flat = rng.choice(int(np.prod(shape)), size=int(n_seeds), replace=False)
            cells = np.stack(np.unravel_index(flat, shape), axis=1)
        for s, c in enumerate(cells):
            occ[tuple(c)] = 1
            owner[tuple(c)] = s
            order[tuple(c)] = 0
    if n_particles is None and not deposit:
        raise ValueError("clusters need a number of particles")
    target = -1 if n_particles is None else int(n_particles)
    _aggregate(
        occ,
        owner,
        order,
        AGGREGATION_PROCESSES.index(process),
        deposit,
        target,
        float(sticking),
        int(dims),
        single,
        rng,
    )
    return occ, owner, order
