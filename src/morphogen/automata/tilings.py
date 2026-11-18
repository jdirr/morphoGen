"""Tilings of the plane and of space: lattices of cellular automata beyond the square grid.

A :class:`Tiling` is a set of cells (convex polygons in 2D, convex polyhedra in 3D) with
their centres, areas or volumes, shapes and adjacency: the CSR lists of the cells sharing
a face (an edge in 2D), an edge or a vertex with each cell.

``square``
    squares (cubes in 3D): the grid of :mod:`morphogen.automata.lattice`, as a tiling;
``triangular``
    equilateral triangles: 3 edge and 12 vertex neighbours;
``hexagonal``
    regular hexagons (honeycomb cells): 6 neighbours, each sharing an edge;
``penrose``
    the thick and thin rhombi of Penrose's quasi-periodic tiling, by de Bruijn's
    pentagrid method: 4 edge neighbours, 7 to 11 vertex neighbours; finite patches only;
``ammann_beenker``
    the squares and 45-degree rhombi of the octagonal Ammann-Beenker tiling, by the same
    method on four grids (the cut-and-project of Z^4): 4 edge neighbours; finite patches;
``bcc``, ``fcc`` (3D)
    the Voronoi cells of the body- and face-centred cubic lattices: truncated octahedra
    (14 faces: 8 hexagons towards the nearest neighbours, 6 squares towards the second
    ones) and rhombic dodecahedra (12 faces; 18 vertex neighbours);
``icosahedral`` (3D)
    the prolate and oblate golden rhombohedra of the icosahedral tiling (Ammann,
    Kramer & Neri), by the dual method on six grids of planes normal to the fivefold
    axes of the icosahedron (the cut-and-project of Z^6): 6 face neighbours; finite
    patches only. Its 24 vertex stars (:meth:`Tiling.vertex_stars`) have the frequencies
    of :data:`ICOSAHEDRAL_STARS`, the shares of its acceptance window
    (:func:`window_stars`).

In 3D, the planar tilings are stacked in layers of prisms (``hexagonal``: hexagonal
prisms, 8 face and 20 vertex neighbours; ``square``: cubes). Cells have the area
``cell_size**2`` (planar tilings; the mean area for quasi-periodic tilings) or the
volume ``cell_size**3``, and prisms are ``cell_size`` high: every lattice has about as
many cells as the square grid of the same cell size.

Periodic tilings fill a box with a whole number of unit cells, scaled along each axis to
fit it exactly (the cells are then slightly distorted, like the cells of the square grid
of a box whose edges are not multiples of the cell size): the tiling is the affine image
of a reference tiling, the Voronoi tessellation of its centres (the triangles are the
Voronoi cells of the honeycomb of their centroids), in which points are located by their
nearest centre. Quasi-periodic tilings have no period: they cover boxes as finite
patches whose cells at the edges have fewer neighbours (free boundaries). Their periodic
approximants (``approximant``: an order ``k``) fill periodic boxes with the same tiles:
the rational approximants of the cut-and-project (a phason strain), whose grids replace
the golden mean τ by ``F_{k+1}/F_k`` (Fibonacci numbers; Penrose rhombi, golden
rhombohedra: the cubic approximants 1/1, 2/1, 3/2...) or the silver mean ``1 + √2`` by
``P_{k+1}/P_k`` (Pell numbers; Ammann-Beenker tiles), periodic in a rectangle or a cube
(:func:`approximant_period`) of which the box holds a whole number, fitted to it like
the unit cells of the periodic tilings.

:meth:`Tiling.diffusion_weights` gives the finite-volume Laplacian of every tiling (the
flux through each face: its length or area over the distance between the centres), on
which :func:`morphogen.automata.reaction.gray_scott_tiling` runs reaction-diffusion.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from numba import njit, prange
from scipy import sparse
from scipy.spatial import cKDTree

from morphogen.automata.lattice import NEIGHBOURHOODS

#: Lattices of cells: the planar tilings (prisms in 3D), then the 3D lattices.
LATTICES = (
    "square", "triangular", "hexagonal", "penrose", "ammann_beenker", "bcc", "fcc",
    "icosahedral",
)  # fmt: skip
PLANAR = ("square", "triangular", "hexagonal", "penrose", "ammann_beenker")
#: Quasi-periodic tilings: finite patches, never periodic boxes.
QUASIPERIODIC = ("penrose", "ammann_beenker", "icosahedral")
GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0
SILVER = 1.0 + math.sqrt(2.0)
_EPS = 1e-6

#: Grids of lines of the planar quasi-periodic tilings (de Bruijn): their number and the
#: angle between consecutive ones (unit vectors e_j at angles j * step).
MULTIGRIDS = {"penrose": (5, 2 * math.pi / 5), "ammann_beenker": (4, math.pi / 4)}


def _icosahedral_axes(t: float = GOLDEN) -> np.ndarray:
    """The six fivefold axes of the icosahedron, unit vectors ``(0, ±1, τ)`` and their
    cyclic permutations (pairwise at ``arccos(±1/√5)``). With ``t = -1/τ``, the conjugate
    of the golden mean: their images in the perpendicular space of the cut-and-project of
    Z^6, whose cosines have the opposite signs."""
    a = np.array([[1, t, 0], [-1, t, 0], [0, 1, t], [0, -1, t], [t, 0, 1], [-t, 0, 1]])
    return a / np.linalg.norm(a, axis=1, keepdims=True)


#: Volumes of the prolate and oblate golden rhombohedra of unit edge (in the ratio τ).
GOLDEN_RHOMBOHEDRA = (
    2.0 * math.sqrt(GOLDEN / (5.0 * math.sqrt(5.0))),
    2.0 / math.sqrt(5.0 * math.sqrt(5.0) * GOLDEN),
)


def _tau(*powers: int) -> float:
    """``τ^-n`` summed over the powers ``n`` (a negative one: ``-τ^n``)."""
    return sum(math.copysign(GOLDEN ** -abs(n), n) for n in powers)


#: The 24 vertex stars of the icosahedral tiling (the kinds of
#: :meth:`Tiling.vertex_stars`) and their frequencies in the infinite tiling: the shares
#: of its acceptance window, the rhombic triacontahedron, in which the images of the
#: vertices of each star fall (measured there by :func:`window_stars`; these expressions
#: of the golden mean sum to 1 and give the mean coordination 6 exactly).
ICOSAHEDRAL_STARS = {
    ((0, 2, 2, 0), (3, 3, 3, 3)): _tau(3),
    ((0, 4, 1, 1), (3, 3, 4, 4, 4)): _tau(3),
    ((0, 6, 0, 2), (3, 3, 4, 4, 5, 5)): _tau(3),
    ((1, 3, 1, 3), (3, 3, 4, 4, 5, 5)): _tau(6),
    ((1, 5, 0, 4), (3, 3, 4, 4, 5, 5, 6)): _tau(6),
    ((2, 4, 0, 6), (3, 3, 4, 4, 5, 5, 5, 7)): _tau(7, 7),
    ((3, 3, 0, 8), (3, 3, 3, 4, 5, 5, 5, 6, 8)): _tau(9, 9),
    ((5, 3, 0, 6), (3, 4, 4, 4, 4, 5, 5, 6, 7)): _tau(9, 12),
    ((4, 2, 0, 10), (3, 3, 3, 3, 5, 5, 5, 5, 7, 9)): _tau(12, 12),
    ((6, 2, 0, 8), (3, 3, 4, 4, 4, 5, 5, 5, 7, 8)): _tau(12, 15),
    ((8, 2, 0, 6), (4, 4, 4, 4, 4, 4, 5, 5, 7, 7)): _tau(12),
    ((6, 0, 0, 14), (3, 3, 3, 3, 3, 3, 5, 5, 7, 7, 8, 10)): _tau(12, 15),
    ((8, 0, 0, 12), (3, 3, 3, 3, 3, 4, 5, 6, 6, 7, 7, 10)): _tau(12),
    ((8, 0, 0, 12), (3, 3, 3, 3, 4, 4, 4, 5, 7, 7, 8, 9)): _tau(12, -15),
    ((10, 0, 0, 10), (3, 3, 3, 3, 3, 5, 6, 6, 6, 6, 6, 10)): _tau(12),
    ((10, 0, 0, 10), (3, 3, 3, 4, 4, 4, 4, 6, 6, 7, 7, 9)): _tau(12),
    ((10, 0, 0, 10), (3, 3, 4, 4, 4, 4, 4, 4, 7, 7, 8, 8)): _tau(12, 12, 15),
    ((12, 0, 0, 8), (3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 9)): _tau(15),
    ((12, 0, 0, 8), (3, 4, 4, 4, 4, 4, 4, 5, 6, 7, 7, 8)): _tau(12, 15),
    ((14, 0, 0, 6), (3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 8)): _tau(12, -15),
    ((14, 0, 0, 6), (4, 4, 4, 4, 4, 4, 5, 5, 5, 7, 7, 7)): _tau(12, 12),
    ((16, 0, 0, 4), (4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 7)): _tau(12, 15),
    ((18, 0, 0, 2), (4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6)): _tau(12),
    ((20, 0, 0, 0), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)): _tau(9),
}


def tile_edge(name: str, cell_size: float = 1.0) -> float:
    """Edge of the tiles of the quasi-periodic tiling ``name`` whose mean area (volume
    for ``icosahedral``) is ``cell_size**2`` (``**3``): Penrose's thick and thin rhombi
    (areas ``sin 72°`` and ``sin 36°`` times the squared edge, in the ratio of the golden
    mean τ), Ammann-Beenker's squares and 45° rhombi (areas 1 and ``1/√2``, in the
    ratio ``1 : √2``), the prolate and oblate golden rhombohedra (in the ratio τ)."""
    if name == "penrose":
        mean = (GOLDEN * math.sin(2 * math.pi / 5) + math.sin(math.pi / 5)) / (GOLDEN + 1)
        return float(cell_size) / math.sqrt(mean)
    if name == "ammann_beenker":
        return float(cell_size) / math.sqrt(2.0 / SILVER)
    if name == "icosahedral":
        prolate, oblate = GOLDEN_RHOMBOHEDRA
        return float(cell_size) / ((GOLDEN * prolate + oblate) / (GOLDEN + 1)) ** (1 / 3)
    raise ValueError(f"tile_edge: one of {QUASIPERIODIC}, got {name!r}")


def penrose_edge(cell_size: float = 1.0) -> float:
    """Edge of the Penrose rhombi whose mean area is ``cell_size**2`` (thick and thin
    rhombi, of areas ``sin 72°`` and ``sin 36°`` times the squared edge, in the ratio of
    the golden mean)."""
    return tile_edge("penrose", cell_size)


#: Largest distance between the centres of two cells sharing a vertex, in cell sizes:
#: the reach of one step of a radius-1 neighbourhood (along an axis for the grid).
#: Rhombi and rhombohedra: twice their largest half-diagonal (the thin Penrose rhombus,
#: the 45° rhombus, the prolate rhombohedron along its threefold axis).
REACH = {
    "square": 1.0,
    "triangular": 4.0 / 3**0.75,
    "hexagonal": (2.0 / 3**0.5) ** 0.5,
    "penrose": 2.0 * math.cos(math.pi / 10) * penrose_edge(),
    "ammann_beenker": 2.0 * math.cos(math.pi / 8) * tile_edge("ammann_beenker"),
    "bcc": 2.0 ** (1 / 3),
    "fcc": 4.0 ** (1 / 3),
    "icosahedral": math.sqrt(3.0 + 6.0 / math.sqrt(5.0)) * tile_edge("icosahedral"),
}


def _star(name: str) -> np.ndarray:
    """Unit vectors ``e_j`` of the grids of a quasi-periodic tiling (rows): the edges of
    its tiles."""
    if name == "icosahedral":
        return _icosahedral_axes()
    n, step = MULTIGRIDS[name]
    t = step * np.arange(n)
    return np.c_[np.cos(t), np.sin(t)]


# -- periodic approximants -----------------------------------------------------------------


def _fibonacci(k: int) -> tuple[int, int, int]:
    """Fibonacci numbers ``F_{k-1}, F_k, F_{k+1}`` (``F_0 = 0``, ``F_1 = 1``)."""
    a, b = 0, 1
    for _ in range(k - 1):
        a, b = b, a + b
    return a, b, a + b


def _pell(k: int) -> tuple[int, int]:
    """Pell number ``P_k`` and its companion ``H_k = P_k + P_{k-1}``: ``(1 + √2)^k = H_k +
    P_k √2``."""
    a, b = 0, 1  # P_0, P_1
    for _ in range(k - 1):
        a, b = b, 2 * b + a
    return b, a + b


def _orders(name: str, order) -> tuple[int, ...]:
    """The order of the approximant along each axis of its period (``order``: one for
    every axis, or one per axis)."""
    d = 3 if name == "icosahedral" else 2
    values = (order,) * d if np.ndim(order) == 0 else tuple(order)
    if len(values) != d or not all(
        not isinstance(v, (bool, np.bool_)) and float(v).is_integer() and v >= 1 for v in values
    ):
        raise ValueError(
            f"approximant: an order of at least 1, or {d} of them (one per axis), got {order!r}"
        )
    return tuple(int(v) for v in values)


def approximant_grid(name: str, order) -> np.ndarray:
    """Integer grid vectors ``g_j`` (rows, one per grid) of the periodic approximant of
    order ``order`` (one, or one per axis) of the quasi-periodic tiling ``name``.

    They are the unit vectors ``e_j`` of its grids (:func:`make_tiling`) with their
    irrational ratios made rational, scaled along each axis to integers: τ replaced by
    ``F_{k+1}/F_k`` in the components ``(1, (τ - 1)/2, -τ/2, -τ/2, (τ - 1)/2)`` (times
    ``2 F_k``) and ``(0, 1, 1/τ, -1/τ, -1)`` (times ``F_{k+1}``) of the Penrose grids, and in
    the axes ``(±1, τ, 0)`` and their cyclic permutations (times ``F_k``) of the icosahedral
    ones; ``√2`` replaced by ``H_k/P_k`` (Pell numbers, so that ``1 + √2`` becomes
    ``P_{k+1}/P_k``) in the Ammann-Beenker grids. Column ``a`` is the period along axis
    ``a`` of the multigrid ``x . g_j + γ_j ∈ Z``, in Z^n: the multigrid has the period 1
    along each axis of ``x``, and its dual tiling, of edges ``e_j`` (the same tiles), the
    period ``sum_j g_ja e_j``, along axis ``a`` (:func:`approximant_period`).
    """
    k = _orders(name, order)
    if name == "penrose":
        a0, a1, a2 = _fibonacci(k[0])
        _, b1, b2 = _fibonacci(k[1])
        return np.array([[2 * a1, 0], [a0, b2], [-a2, b1], [-a2, -b1], [a0, -b2]])
    if name == "ammann_beenker":
        pa, ha = _pell(k[0])
        pb, hb = _pell(k[1])
        return np.array([[ha, 0], [pa, pb], [0, hb], [-pa, pb]])
    if name == "icosahedral":
        _, x1, x2 = _fibonacci(k[0])
        _, y1, y2 = _fibonacci(k[1])
        _, z1, z2 = _fibonacci(k[2])
        return np.array(
            [[x1, y2, 0], [-x1, y2, 0], [0, y1, z2], [0, -y1, z2], [x2, 0, z1], [-x2, 0, z1]]
        )
    raise ValueError(
        f"approximant: one of the quasi-periodic tilings {QUASIPERIODIC}, got {name!r}"
    )


def approximant_period(name: str, order, cell_size: float = 1.0) -> np.ndarray:
    """Edges of the rectangular or cuboid period of the approximant of order ``order``
    (one, or one per axis) of the quasi-periodic tiling ``name`` whose tiles are those of
    mean area ``cell_size**2`` (volume ``**3``), :func:`tile_edge`: ``√5 τ^k`` by ``2 sin 72°
    τ^k`` edges for Penrose rhombi, ``(1 + √2)^k`` for Ammann-Beenker tiles (a square), and
    the cube of edge ``2 τ^(k+1) / √(2 + τ)`` for golden rhombohedra. A periodic box holds a
    whole number of periods along each axis (:func:`make_tiling`)."""
    period = _star(name).T @ approximant_grid(name, order)
    return np.diag(period) * tile_edge(name, cell_size)


def _reduction(periods: np.ndarray):
    """Reduction of integer vectors (in rows) modulo the lattice of the columns of
    ``periods`` (``n x d`` integers): ``K - periods @ c``, ``c = floor(adj(B) K_I / det B)``
    for the ``d x d`` minor ``B`` of rows ``I`` of largest determinant, the representative
    of its class with ``0 <= adj(B) K_I < det B``, in exact integer arithmetic (the
    vertices of a tiling of the torus, numbered without tolerance)."""
    periods = np.asarray(periods, dtype=np.int64)
    n, d = periods.shape
    rows = list(max(itertools.combinations(range(n), d), key=lambda r: abs(_det(periods[list(r)]))))
    minor = periods[rows]
    det = _det(minor)
    if d == 2:
        adj = np.array([[minor[1, 1], -minor[0, 1]], [-minor[1, 0], minor[0, 0]]])
    else:
        adj = np.array([np.cross(minor[1], minor[2]), np.cross(minor[2], minor[0]),
                        np.cross(minor[0], minor[1])]).T  # fmt: skip
    if det < 0:
        det, adj = -det, -adj

    def reduce(keys: np.ndarray) -> np.ndarray:
        c = (keys[..., rows] @ adj.T) // det
        return keys - c @ periods.T

    return reduce


def _det(m: np.ndarray) -> int:
    """Determinant of a 2 x 2 or 3 x 3 integer matrix, exactly."""
    if len(m) == 2:
        return int(m[0, 0]) * int(m[1, 1]) - int(m[0, 1]) * int(m[1, 0])
    return int(np.dot(m[0], np.cross(m[1], m[2])))


def _periodic_dual_cells(e, grid, gamma, counts, families, reduce, what: str):
    """Cells of the periodic tiling dual to the multigrid ``x . g_j + gamma_j = k`` (the
    integer rows ``g_j`` of ``grid``, of period 1 along each axis of ``x``) over the box
    ``[0, counts)`` of grid periods, as :func:`_dual_cells` gives them over a patch: for
    each tuple of ``families`` (of hyperplanes ``g_r`` in the same orientation as their
    edges ``e_r``, refused otherwise), the integer indices ``K`` of the corners of the
    cells reduced by ``reduce`` modulo the periods (:func:`_reduction`), one per
    intersection of the torus (the box is enlarged by 1e-7 grid periods so that none is
    missed, the images met twice kept once, and the count checked against ``|det g|
    prod(counts)``).
    """
    counts = np.asarray(counts)
    gamma = np.asarray(gamma, dtype=float)
    span = grid * counts  # the values g_j . x over the box: between the sums of its signs
    lo = np.minimum(span, 0).sum(axis=1) + gamma
    hi = np.maximum(span, 0).sum(axis=1) + gamma
    for fam in families:
        fam = list(fam)
        det = _det(grid[fam])
        if det == 0:  # parallel hyperplanes: no cell of these edges
            continue
        if det * np.linalg.det(e[fam]) < 0:
            raise ValueError(f"approximant: grids {fam} of the other orientation than their tiles")
        ranges = [np.arange(math.floor(lo[r]) - 1, math.ceil(hi[r]) + 2) for r in fam]
        k = np.stack([g.ravel() for g in np.meshgrid(*ranges, indexing="ij")], axis=1)
        z = np.linalg.solve(grid[fam].astype(float), (k - gamma[fam]).T).T
        keep = np.all((z > -1e-7) & (z < counts + 1e-7), axis=1)
        k, z = k[keep], z[keep]
        w = z @ grid.T + gamma
        _check_regular(np.delete(w, fam, axis=1), gamma, what)
        big_k = np.ceil(w).astype(np.int64)
        big_k[:, fam] = k
        big_k = np.unique(reduce(big_k), axis=0)
        if len(big_k) != abs(det) * int(np.prod(counts)):
            raise RuntimeError(
                f"approximant: {len(big_k)} cells of the grids {fam}, "
                f"{abs(det) * int(np.prod(counts))} expected"
            )
        yield fam, big_k


# -- sparse adjacency --------------------------------------------------------------------


def _csr(n: int, rows, cols, mirror: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric CSR lists ``(indptr, indices)`` of the pairs ``(rows, cols)``, with
    their mirror images unless the pairs are symmetric already (duplicates and loops
    dropped, indices sorted)."""
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    keep = rows != cols
    r, c = rows[keep], cols[keep]
    if mirror:
        r, c = np.concatenate([r, c]), np.concatenate([c, r])
    m = sparse.csr_matrix((np.ones(len(r), dtype=np.int32), (r, c)), shape=(n, n))
    m.sum_duplicates()
    m.sort_indices()
    return m.indptr.astype(np.int64), m.indices.astype(np.int64)


def _lattice_adjacency(n, basis, cell, radius: float, periodic: bool):
    """Pairs of points of a lattice closer than ``radius`` (CSR): ``basis`` fractions of a
    rectangular unit ``cell``, ``n`` cells per axis, the point ``i`` of the cell ``c`` (C
    order) numbered ``c * len(basis) + i``. The neighbours of each basis point are found
    once in a block of cells, then translated to every cell."""
    n = np.asarray(n)
    d, nb = len(n), len(basis)
    m = math.ceil(radius / float(np.min(cell))) + 1
    span = np.arange(-m, m + 1)
    shifts = np.stack(np.meshgrid(*[span] * d, indexing="ij"), -1).reshape(-1, d)
    units = np.stack(np.meshgrid(*(np.arange(k) for k in n), indexing="ij"), -1).reshape(-1, d)
    rows, cols = [], []
    for b0 in range(nb):
        dist = np.linalg.norm((shifts[:, None, :] + basis[None] - basis[b0]) * cell, axis=-1)
        shift, b1 = np.nonzero((dist <= radius) & (dist > 1e-9 * radius))
        target = units[:, None, :] + shifts[shift][None]
        if periodic:
            target %= n
            valid = np.ones(target.shape[:2], dtype=bool)
        else:
            valid = np.all((target >= 0) & (target < n), axis=-1)
            target = np.where(valid[..., None], target, 0)
        flat = np.ravel_multi_index(tuple(np.moveaxis(target, -1, 0)), tuple(n)) * nb + b1
        rows.append(np.broadcast_to((np.arange(len(units)) * nb + b0)[:, None], flat.shape)[valid])
        cols.append(flat[valid])
    return _csr(len(units) * nb, np.concatenate(rows), np.concatenate(cols), mirror=False)


def _pairs(indptr, indices) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(np.arange(len(indptr) - 1), np.diff(indptr)), np.asarray(indices)


def _power(indptr, indices, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Cells at graph distance 1 to ``m`` (CSR)."""
    n = len(indptr) - 1
    a = sparse.csr_matrix((np.ones(len(indices), dtype=np.int32), indices, indptr), shape=(n, n))
    reach = a.copy()
    for _ in range(m - 1):
        reach = reach + reach @ a
        reach.data[:] = 1
    c = reach.tocoo()
    return _csr(n, c.row, c.col)


def _shells(points, box, radius) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of points closer than ``radius`` (periodic in ``box`` unless it is None)."""
    if box is None:
        tree = cKDTree(points)
    else:
        tree = cKDTree(np.mod(points, box), boxsize=box)
    p = tree.query_pairs(radius, output_type="ndarray")
    return _csr(len(points), p[:, 0], p[:, 1])


@njit(parallel=True, cache=True)
def _csr_count(state, indptr, indices, out):
    for i in prange(out.shape[0]):
        s = 0
        for p in range(indptr[i], indptr[i + 1]):
            s += state[indices[p]]
        out[i] = s


@njit(parallel=True, cache=True)
def _csr_differ(state, indptr, indices):
    """Number of (ordered) pairs of neighbours in different states."""
    n = indptr.shape[0] - 1
    total = 0
    for i in prange(n):
        s = 0
        for p in range(indptr[i], indptr[i + 1]):
            s += state[indices[p]] != state[i]
        total += s
    return total


@njit(cache=True)
def _greedy_colours(indptr, indices):
    """Greedy colouring: neighbours never share a colour."""
    n = indptr.shape[0] - 1
    degree = 0
    for i in range(n):
        degree = max(degree, indptr[i + 1] - indptr[i])
    colour = -np.ones(n, dtype=np.int64)
    mark = -np.ones(degree + 2, dtype=np.int64)
    for i in range(n):
        for p in range(indptr[i], indptr[i + 1]):
            c = colour[indices[p]]
            if c >= 0:
                mark[c] = i
        c = 0
        while mark[c] == i:
            c += 1
        colour[i] = c
    return colour


@njit(parallel=True, cache=True)
def _in_polygons(points, candidates, centres, poly_ptr, poly, box, periodic, out):
    """First candidate cell whose polygon (offsets from its centre, counterclockwise)
    contains each point; the nearest candidate when none does (round-off)."""
    for p in prange(points.shape[0]):
        out[p] = candidates[p, 0]
        for q in range(candidates.shape[1]):
            c = candidates[p, q]
            if c < 0 or c >= centres.shape[0]:
                break
            dx = points[p, 0] - centres[c, 0]
            dy = points[p, 1] - centres[c, 1]
            if periodic:
                dx -= box[0] * np.floor(dx / box[0] + 0.5)
                dy -= box[1] * np.floor(dy / box[1] + 0.5)
            a = poly_ptr[c]
            b = poly_ptr[c + 1]
            inside = True
            for v in range(a, b):
                w = v + 1 if v + 1 < b else a
                ex = poly[w, 0] - poly[v, 0]
                ey = poly[w, 1] - poly[v, 1]
                if ex * (dy - poly[v, 1]) - ey * (dx - poly[v, 0]) < -1e-9 * (abs(ex) + abs(ey)):
                    inside = False
                    break
            if inside:
                out[p] = c
                break


# -- the tiling ----------------------------------------------------------------------------


class Tiling:
    """Cells of a tiling: centres, shapes, areas or volumes and adjacency.

    Build tilings with :func:`make_tiling`. Cells are numbered layer by layer for prisms
    (``layer * n_planar + planar_cell``). ``adjacency(kind)`` gives the CSR lists of the
    cells sharing a ``"faces"`` (an edge in 2D), ``"edges"`` or ``"vertices"`` with each
    cell; :meth:`neighbours` the neighbourhoods of the automata.
    """

    def __init__(self, spec: dict, plane: dict | None, solid: dict | None) -> None:
        self.spec = dict(spec)
        self.name = spec["name"]
        self.dims = int(spec["dims"])
        self.cell_size = float(spec["cell_size"])
        self.periodic = bool(spec["periodic"])
        self.origin = np.asarray(spec["origin"], dtype=float)
        self.size = np.asarray(spec["size"], dtype=float)
        self._plane = plane
        self._solid = solid
        self._cache: dict = {}
        if plane is not None:
            m = len(plane["centres"])
            layers = plane["layers"]
            hz = plane["layer_height"]
            sx, sy, sz = plane["scale"]
            xy = plane["centres"] * (sx, sy)
            z = (np.arange(layers) + 0.5) * hz * sz
            self.centres = np.empty((m * layers, 3))
            self.centres[:, :2] = np.tile(xy, (layers, 1))
            self.centres[:, 2] = np.repeat(z, m)
            self.centres += self.origin
            area = plane["areas"] * sx * sy
            measure = area if self.dims == 2 else area * hz * sz
            self.volumes = np.tile(measure, layers)
            self.kinds = np.tile(plane["kinds"], layers)
        elif "volumes" in solid:  # cells of several shapes (icosahedral)
            self.centres = solid["centres"] + self.origin
            self.volumes = solid["volumes"]
            self.kinds = solid["kinds"]
        else:
            self.centres = solid["centres"] * solid["scale"] + self.origin
            self.volumes = np.full(len(self.centres), solid["volume"] * np.prod(solid["scale"]))
            self.kinds = np.zeros(len(self.centres), dtype=np.int8)
        self._adjacency: dict = {}

    def _build_adjacency(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        if self._plane is not None:
            return _prism_adjacency(self._plane, self.dims, self.periodic, kind)
        s = self._solid
        if kind in s:  # from the vertices shared by the cells
            return s[kind]
        r = s["shells"][kind]
        key = ("shell", r)
        if key not in self._cache:
            self._cache[key] = _lattice_adjacency(*s["lattice"], r * (1 + _EPS), self.periodic)
        return self._cache[key]

    # -- description -------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return len(self.centres)

    def adjacency(self, kind: str = "faces") -> tuple[np.ndarray, np.ndarray]:
        """CSR lists ``(indptr, indices)`` of the cells sharing a face (an edge in 2D),
        an edge or a vertex (``kind``: faces, edges or vertices) with each cell."""
        if kind not in ("faces", "edges", "vertices"):
            raise ValueError(f"adjacency: faces, edges or vertices, got {kind!r}")
        if kind not in self._adjacency:
            self._adjacency[kind] = self._build_adjacency(kind)
        return self._adjacency[kind]

    def neighbours(
        self, neighbourhood: str = "moore", radius: float = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Neighbourhood of every cell (CSR), the counterpart of the grid's offsets.

        ``von_neumann``, ``edges`` and ``moore``: the cells within ``floor(radius)``
        steps across faces (edges in 2D), edges or vertices (on the square grid: the
        L1, edge and max-norm balls); ``euclidean``: the cells whose centres lie within
        ``radius`` cell sizes, refused when it reaches none (the nearest centres lie 1.075,
        1.091 and 1.122 cell sizes apart on undistorted hexagonal, BCC and FCC lattices).
        """
        if neighbourhood not in NEIGHBOURHOODS:
            raise ValueError(f"neighbourhood: one of {NEIGHBOURHOODS}, got {neighbourhood!r}")
        r = float(radius)
        if not r >= 1:
            raise ValueError(f"radius: must be at least 1, got {radius!r}")
        key = (neighbourhood, r if neighbourhood == "euclidean" else int(r))
        if key not in self._cache:
            if neighbourhood == "euclidean":
                ball = self.ball(r, include_self=False)
                if ball[0][-1] == 0:  # the automaton would change nothing
                    d = 2 if self.dims == 2 else 3
                    pts = self.centres[:, :d] - self.origin[:d]
                    box = self.size[:d] if self.periodic else None
                    tree = cKDTree(pts) if box is None else cKDTree(np.mod(pts, box), boxsize=box)
                    gap = float(tree.query(tree.data, k=2)[0][:, 1].min()) / self.cell_size
                    raise ValueError(
                        f"radius {r:g}: no cell centre within it of another on the {self.name} "
                        f"lattice, the nearest {gap:.4g} cell sizes apart (take a larger radius)"
                    )
                self._cache[key] = ball
            else:
                kind = {"von_neumann": "faces", "edges": "edges", "moore": "vertices"}
                base = self.adjacency(kind[neighbourhood])
                self._cache[key] = base if int(r) == 1 else _power(*base, int(r))
        return self._cache[key]

    def ball(self, radius: float, include_self: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Cells whose centres lie within ``radius`` cell sizes of each centre (CSR)."""
        key = ("ball", float(radius), bool(include_self))
        if key not in self._cache:
            d = 2 if self.dims == 2 else 3
            pts = self.centres[:, :d] - self.origin[:d]
            box = self.size[:d] if self.periodic else None
            indptr, indices = _shells(pts, box, float(radius) * self.cell_size * (1 + 1e-9))
            if include_self:
                rows, cols = _pairs(indptr, indices)
                n = self.n_cells
                a = sparse.csr_matrix(
                    (np.ones(len(rows) + n, dtype=np.int32),
                     (np.concatenate([rows, np.arange(n)]), np.concatenate([cols, np.arange(n)]))),
                    shape=(n, n),
                )  # fmt: skip
                a.sort_indices()
                indptr, indices = a.indptr.astype(np.int64), a.indices.astype(np.int64)
            self._cache[key] = (indptr, indices)
        return self._cache[key]

    def counts(self, state, indptr, indices) -> np.ndarray:
        """Number of cells in state 1 in the CSR neighbourhoods ``(indptr, indices)``."""
        out = np.empty(len(indptr) - 1, dtype=np.int32)
        _csr_count(np.ascontiguousarray(state, dtype=np.uint8), indptr, indices, out)
        return out

    def colours(self, indptr, indices) -> np.ndarray:
        """Colour of each cell such that neighbours (CSR) never share one (greedy)."""
        return _greedy_colours(np.asarray(indptr, np.int64), np.asarray(indices, np.int64))

    def face_measures(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Length (2D) or area (3D) of the face shared by each pair of face neighbours:
        CSR lists ``(indptr, indices, measures)`` aligned with ``adjacency("faces")``.

        Periodic tilings are the affine images ``S`` of Voronoi tilings: a face of area
        ``A`` normal to the offset ``n`` of the centres becomes ``A |det S| |S^-1 n|``
        (Nanson). Rhombi and rhombohedra measure the edges and faces they share: the
        tiles' edge, and the area ``(2 / √5) a^2`` of golden rhombi, unless an approximant
        is fitted to its box (their images by ``S``)."""
        if "face_measures" not in self._cache:
            indptr, indices = self.adjacency("faces")
            rows, cols = _pairs(indptr, indices)
            if self._plane is not None:
                measure = self._prism_faces(rows, cols)
            else:
                s = self._solid
                if "frames" in s:  # golden rhombi
                    measure = _shared_measures(*_rhombohedron_faces(s), rows, cols)
                else:  # Voronoi cells of the BCC and FCC lattices
                    d = s["centres"][cols] - s["centres"][rows]
                    if self.periodic:
                        d -= s["box"] * np.round(d / s["box"])
                    r = np.linalg.norm(d, axis=1)
                    a = float(s["lattice"][2][0])
                    if self.name == "bcc":  # hexagons (nearest), squares (second)
                        area = np.where(r < 0.9 * a, 3 * math.sqrt(3) / 16, 1 / 8) * a**2
                    else:  # rhombi
                        area = np.full(len(r), a**2 / (4 * math.sqrt(2)))
                    scale = s["scale"]
                    measure = area * np.prod(scale) * np.linalg.norm(d / r[:, None] / scale, axis=1)
            self._cache["face_measures"] = (indptr, indices, measure)
        return self._cache["face_measures"]

    def _prism_faces(self, rows, cols) -> np.ndarray:
        """Edges of planar cells (lengths) or faces of prisms (areas: the edges times the
        layer height across the sides, the cell's area across the bases)."""
        p = self._plane
        sx, sy, sz = p["scale"]
        m = len(p["centres"])
        a, b = rows % m, cols % m
        if "vertex_ids" in p:  # rhombi: the edges they share, scaled
            keys, owner, sides = _polygon_sides(p)
            length = _shared_measures(keys, owner, np.hypot(sx * sides[:, 0], sy * sides[:, 1]),
                                      a, b)  # fmt: skip
        else:
            d = p["centres"][b] - p["centres"][a]
            if self.periodic:
                d -= p["box"] * np.round(d / p["box"])  # minimum image (reference tiling)
            r = np.linalg.norm(d, axis=1)
            r[r == 0] = 1.0  # bases of prisms
            n = d / r[:, None]
            edge = float(np.linalg.norm(p["poly"][1] - p["poly"][0]))  # every edge
            length = edge * np.hypot(sy * n[:, 0], sx * n[:, 1])
        if self.dims == 2:
            return length
        return np.where(a == b, p["areas"][a] * sx * sy, length * p["layer_height"] * sz)

    def diffusion_weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weights of the finite-volume Laplacian of the tiling, in units of the cell
        size: CSR lists ``(indptr, indices, w)`` with ``w = A / d`` for the face of measure
        ``A`` (length or area) between centres ``d`` apart (two-point fluxes), so that
        ``lap(u)_i = sum_j w_ij (u_j - u_i) / V_i`` (``V_i`` the cell's measure, also in
        cell sizes): the 5- and 7-point Laplacians on the square tiling."""
        if "diffusion" not in self._cache:
            indptr, indices, measure = self.face_measures()
            rows, cols = _pairs(indptr, indices)
            d = self.centres[cols] - self.centres[rows]
            if self.periodic:
                d -= self.size * np.round(d / self.size)
            k = 2 if self.dims == 2 else 3
            dist = np.linalg.norm(d[:, :k], axis=1)
            h = self.cell_size
            self._cache["diffusion"] = (indptr, indices, measure / h ** (k - 1) / (dist / h))
        return self._cache["diffusion"]

    def interface_density(self, state) -> float:
        """Fraction of the pairs of cells sharing a face that are in different states."""
        indptr, indices = self.adjacency("faces")
        if not len(indices):
            return 0.0
        s = np.ascontiguousarray(state, dtype=np.uint8)
        return _csr_differ(s, indptr, indices) / len(indices)

    def cell_vertices(self, i: int) -> np.ndarray:
        """Vertices of cell ``i``: its polygon (counterclockwise, ``(k, 2)``) for planar
        tilings in 2D, its corners ``(k, 3)`` in 3D."""
        if self._plane is not None:
            p = self._plane
            m = len(p["centres"])
            c = int(i) % m
            off = p["poly"][p["poly_ptr"][c] : p["poly_ptr"][c + 1]] * p["scale"][:2]
            xy = self.centres[i, :2] + off
            if self.dims == 2:
                return xy
            hz = 0.5 * p["layer_height"] * p["scale"][2]
            z = self.centres[i, 2]
            return np.concatenate(
                [np.c_[xy, np.full(len(xy), z - hz)], np.c_[xy, np.full(len(xy), z + hz)]]
            )
        s = self._solid
        if "frames" in s:  # the corners of a rhombohedron, in the order of _CUBE
            return self.origin + s["corner"][i] + _CUBE @ s["frames"][s["triple"][i]]
        return self.centres[i] + s["prototype"] * s["scale"]

    @property
    def circumradius(self) -> float:
        """Largest distance from a cell centre to its vertices."""
        if "circumradius" not in self._cache:
            if self._plane is not None:
                p = self._plane
                r = float(np.max(np.linalg.norm(p["poly"] * p["scale"][:2], axis=1)))
                if self.dims == 3:
                    r = math.hypot(r, 0.5 * p["layer_height"] * p["scale"][2])
            elif "circumradius" in self._solid:
                r = self._solid["circumradius"]
            else:
                s = self._solid
                r = float(np.max(np.linalg.norm(s["prototype"] * s["scale"], axis=1)))
            self._cache["circumradius"] = r
        return self._cache["circumradius"]

    # -- point location ----------------------------------------------------------------

    def locate(self, points) -> np.ndarray:
        """Index of the cell containing each point (``(n, 3)``; the nearest cell for
        points outside a finite patch)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if self._plane is None:
            s = self._solid
            ref = (pts - self.origin) / s["scale"]
            tree = self._tree()
            if self.periodic:
                ref = np.mod(ref, s["box"])
            if "frames" in s:
                return self._locate_cells(ref)
            return tree.query(ref, k=1)[1].astype(np.int64)
        c2 = self._locate_plane(pts[:, :2])
        return self._layer(pts[:, 2]) * len(self._plane["centres"]) + c2

    def _locate_cells(self, ref) -> np.ndarray:
        """Rhombohedra containing the points (tested among the 16 nearest centres, then
        among all the centres closer than the circumradius for the points left)."""
        s = self._solid
        tree = self._tree()
        k = min(16, self.n_cells)
        cand = tree.query(ref, k=k)[1].reshape(len(ref), k).astype(np.int64)
        out = np.empty(len(ref), dtype=np.int64)
        found = np.empty(len(ref), dtype=np.bool_)
        box = s["box"] if self.periodic else np.ones(3)
        args = (s["centres"], s["corner"], s["triple"].astype(np.int64), s["inverse"], box,
                self.periodic)  # fmt: skip
        _in_parallelepipeds(ref, cand, *args, out, found)
        left = np.flatnonzero(~found)
        if len(left):
            near = tree.query_ball_point(ref[left], self.circumradius * (1 + 1e-9))
            width = max(1, max(len(c) for c in near))
            more = np.full((len(left), width), -1, dtype=np.int64)
            for row, c in enumerate(near):
                more[row, : len(c)] = c
            more[:, 0] = np.where(more[:, 0] < 0, out[left], more[:, 0])
            sub, hit = np.empty(len(left), np.int64), np.empty(len(left), np.bool_)
            _in_parallelepipeds(ref[left], more, *args, sub, hit)
            out[left[hit]] = sub[hit]  # outside the patch: the nearest centre
        return out

    def _tree(self) -> cKDTree:
        if "tree" not in self._cache:
            g = self._plane if self._plane is not None else self._solid
            box = g["box"] if self.periodic else None
            data = np.mod(g["centres"], box) if box is not None else g["centres"]
            self._cache["tree"] = cKDTree(data, boxsize=box)
        return self._cache["tree"]

    def _locate_plane(self, xy) -> np.ndarray:
        p = self._plane
        ref = (np.asarray(xy, dtype=float) - self.origin[:2]) / p["scale"][:2]
        box = p["box"] if self.periodic else np.ones(2)
        if self.periodic:
            ref = np.mod(ref, box)
        k = min(12, len(p["centres"]))
        cand = self._tree().query(ref, k=k)[1].reshape(len(ref), k).astype(np.int64)
        out = np.empty(len(ref), dtype=np.int64)
        _in_polygons(ref, cand, p["centres"], p["poly_ptr"], p["poly"], box, self.periodic, out)
        return out

    def _layer(self, z) -> np.ndarray:
        p = self._plane
        layers = p["layers"]
        if self.dims == 2:
            return np.zeros(len(z), dtype=np.int64)
        u = np.floor((np.asarray(z) - self.origin[2]) / (p["layer_height"] * p["scale"][2]))
        u = u.astype(np.int64)
        return np.mod(u, layers) if self.periodic else np.clip(u, 0, layers - 1)

    def cell_index(self, grid) -> np.ndarray:
        """Index of the cell containing the centre of every voxel of ``grid`` (int32
        image)."""
        nx, ny, nz = grid.shape
        if self._plane is not None:
            x, y = np.meshgrid(grid.centers(0), grid.centers(1), indexing="ij")
            c2 = self._locate_plane(np.c_[x.ravel(), y.ravel()]).reshape(nx, ny)
            layer = self._layer(grid.centers(2)) * len(self._plane["centres"])
            return (c2[:, :, None] + layer[None, None, :]).astype(np.int32)
        out = np.empty((nx, ny, nz), dtype=np.int32)
        step = max(1, (1 << 20) // max(1, nx * ny))
        for k0 in range(0, nz, step):
            k1 = min(nz, k0 + step)
            out[:, :, k0:k1] = self.locate(grid.center_points(k0, k1)).reshape(nx, ny, k1 - k0)
        return out

    # -- quasi-periodic tilings ----------------------------------------------------------

    def vertex_stars(self) -> dict[tuple, int]:
        """Vertex stars of a Penrose, Ammann-Beenker or icosahedral tiling: the number of
        interior vertices of each kind.

        Penrose tilings: kinds ``(index, angles)``. ``angles`` is the cyclic sequence of
        the angles of the rhombi around the vertex, in units of 36 degrees (thick rhombi:
        2 and 3; thin ones: 1 and 4), smallest over rotations and reflections; ``index``
        is de Bruijn's index ``sum_j K_j`` of the vertex, folded to ``min(k, 5 - k)`` (1
        or 2 for Penrose tilings), which separates the stars of five thick rhombi into his
        S and S5. Ammann-Beenker tilings: kinds ``angles``, in units of 45 degrees
        (squares: 2; rhombi: 1 and 3).

        Icosahedral tilings: the star of a vertex is the set of the golden rhombohedra
        around it; kinds ``(corners, valences)``. ``corners``: the numbers of the cells
        that meet the vertex by a tip of a prolate rhombohedron (the corner of its three
        acute angles), by another corner of a prolate one, by a tip of an oblate one
        (three obtuse angles) and by another corner of an oblate one; ``valences``: the
        number of cells around each of its edges, sorted (as many as its coordination
        ``Z``; its cells number ``2 Z - 4``). These tell apart the 24 stars of the tiling
        up to the 120 symmetries of the icosahedron (:data:`ICOSAHEDRAL_STARS`, their
        frequencies in the infinite tiling); the directions of the edges alone would give
        9 kinds. The interior vertices are those around which the solid angles of the
        cells add up to the whole sphere (all of them in a periodic approximant, whose
        phason strain leaves a few stars of other kinds)."""
        if self.name == "icosahedral":
            return _icosahedral_stars(self._solid)
        p = self._plane
        if p is None or "vertex_ids" not in p:
            raise ValueError(
                "vertex stars are defined for Penrose, Ammann-Beenker and icosahedral tilings"
            )
        families = MULTIGRIDS[self.name][0]
        ids, poly, ptr = p["vertex_ids"], p["poly"], p["poly_ptr"]
        nxt = np.arange(len(poly)) + 1
        prv = np.arange(len(poly)) - 1
        ends = ptr[1:] - 1
        starts = ptr[:-1]
        nxt[ends] = starts
        prv[starts] = ends
        u = poly[prv] - poly
        v = poly[nxt] - poly
        cos = np.einsum("ij,ij->i", u, v) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
        unit = 180.0 / families  # degrees
        units = np.rint(np.degrees(np.arccos(np.clip(cos, -1, 1))) / unit).astype(int)
        # direction from the vertex to the cell centre orders the cells around it
        direction = np.arctan2(-poly[:, 1], -poly[:, 0])
        order = np.lexsort((direction, ids))
        out: dict[tuple, int] = {}
        bounds = np.flatnonzero(np.diff(ids[order])) + 1
        for group in np.split(order, bounds):
            seq = tuple(int(a) for a in units[group])
            if sum(seq) != 2 * families:
                continue  # vertex on the edge of the patch
            if self.name == "penrose":
                k = int(p["vertex_index"][group[0]])
                key = (min(k, 5 - k), _canonical(seq))
            else:
                key = _canonical(seq)
            out[key] = out.get(key, 0) + 1
        return out


def _canonical(seq: tuple[int, ...]) -> tuple[int, ...]:
    """Smallest rotation of a cyclic sequence or of its reverse."""
    k = len(seq)
    forms = [seq[i:] + seq[:i] for i in range(k)]
    rev = seq[::-1]
    forms += [rev[i:] + rev[:i] for i in range(k)]
    return min(forms)


def _icosahedral_corners() -> tuple[np.ndarray, np.ndarray]:
    """The 160 corners that golden rhombohedra can have at a vertex, numbered ``8 * triple
    + corner`` (the triples of axes in the order of ``itertools.combinations``, the
    corners in that of the vertex numbers of the cells): the three edges that leave each,
    numbered ``2 j + s`` (along ``+e_j``: ``s = 0``; along ``-e_j``: ``s = 1``), and its
    species: 0, a tip of a prolate rhombohedron (three acute angles); 1, another of its
    corners; 2, a tip of an oblate one (three obtuse angles); 3, another of its corners."""
    e = _icosahedral_axes()
    triples = itertools.combinations(range(6), 3)
    edges = np.array(
        [[2 * fam[k] + (c >> k & 1) for k in range(3)] for fam in triples for c in range(8)]
    )
    signed = np.stack([e, -e], axis=1).reshape(12, 3)  # the edge 2 j + s
    a, b, c = (signed[edges[:, k]] for k in range(3))
    acute = sum(
        (np.einsum("ij,ij->i", u, v) > 0).astype(np.int64) for u, v in ((a, b), (b, c), (c, a))
    )
    return edges, np.array([2, 1, 3, 0])[acute]


def _star_key(row) -> tuple:
    """The kind of an icosahedral star from its four numbers of corners and its twelve
    sorted valences (zero for the edges that it has not)."""
    return tuple(int(v) for v in row[:4]), tuple(int(v) for v in row[4:] if v)


def _icosahedral_stars(solid: dict) -> dict[tuple, int]:
    """The stars of the interior vertices of an icosahedral tiling (see
    :meth:`Tiling.vertex_stars`)."""
    edges, species = _icosahedral_corners()
    ids = solid["vertex_ids"].ravel()
    corner = (8 * solid["triple"].astype(np.int64)[:, None] + np.arange(8)).ravel()
    n = int(ids.max()) + 1
    # the solid angle of each corner from the edge vectors of its cell (those of an
    # approximant are scaled with its box; van Oosterom & Strackee 1983)
    unit = solid["frames"] / np.linalg.norm(solid["frames"], axis=2, keepdims=True)
    a, b, c = ((1.0 - 2.0 * _CUBE[None, :, k, None]) * unit[:, None, k, :] for k in range(3))
    volume = np.abs(np.einsum("tci,tci->tc", a, np.cross(b, c)))
    cosines = sum(np.einsum("tci,tci->tc", u, v) for u, v in ((a, b), (b, c), (c, a)))
    angles = (2.0 * np.arctan2(volume, 1.0 + cosines)).ravel()
    whole = np.bincount(ids, weights=angles[corner], minlength=n)
    interior = np.abs(whole - 4.0 * math.pi) < 1e-6
    corners = np.zeros((n, 4), dtype=np.int64)
    np.add.at(corners, (ids, species[corner]), 1)
    valences = np.zeros((n, 12), dtype=np.int64)
    np.add.at(valences, (ids[:, None], edges[corner]), 1)
    valences.sort(axis=1)
    rows, counts = np.unique(np.hstack([corners, valences])[interior], axis=0, return_counts=True)
    return {_star_key(row): int(count) for row, count in zip(rows, counts, strict=True)}


@njit(parallel=True, cache=True)
def _window_star_rows(points, normals, low, high, species, edges, rows):
    """The star of the vertex whose image in the perpendicular space is each of
    ``points``: the cells that have it for a corner are those of the 160 corners whose
    eight vertices fall in the window, the intervals ``[low, high]`` of its coordinates
    along the 15 ``normals``; ``rows``: the numbers of corners of each species and the
    sorted valences of the twelve edges."""
    for p in prange(points.shape[0]):
        along = np.zeros(15)
        for k in range(15):
            along[k] = (
                normals[k, 0] * points[p, 0]
                + normals[k, 1] * points[p, 1]
                + normals[k, 2] * points[p, 2]
            )
        valences = np.zeros(12, dtype=np.int64)
        for corner in range(160):
            inside = True
            for k in range(15):
                if along[k] < low[corner, k] or along[k] > high[corner, k]:
                    inside = False
                    break
            if inside:
                rows[p, species[corner]] += 1
                for j in range(3):
                    valences[edges[corner, j]] += 1
        rows[p, 4:] = np.sort(valences)


def window_stars(samples: int = 1 << 20, rng: np.random.Generator | None = None) -> dict:
    """Frequencies of the vertex stars of the icosahedral tiling by cut and projection
    (the kinds of :meth:`Tiling.vertex_stars`): the tiling is the projection of the points
    of Z^6 whose images in the perpendicular space fall in a window, the rhombic
    triacontahedron (the zonotope of the six perpendicular axes); a rhombohedron is a
    cell when its eight corners do, so that the star of a vertex depends only on its
    image, and the frequency of a star is the share of the window in which the images of
    its vertices fall. That share is measured here on ``samples`` points drawn uniformly
    in the box of the window (about 0.59 of them fall in it): an estimate within
    ``sqrt(f (1 - f) / (0.59 samples))`` of each frequency ``f``, independent of the
    dual method that builds the tilings (:data:`ICOSAHEDRAL_STARS`: the 24 stars and
    their frequencies)."""
    rng = np.random.default_rng() if rng is None else rng
    axes = _icosahedral_axes(-1.0 / GOLDEN)  # of the perpendicular space
    signed = np.stack([axes, -axes], axis=1).reshape(12, 3)
    # the faces of the zonotope: normal to the 15 pairs of axes
    normals = np.array([np.cross(axes[a], axes[b]) for a, b in itertools.combinations(range(6), 2)])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    support = 0.5 * np.abs(normals @ axes.T).sum(axis=1)
    edges, species = _icosahedral_corners()
    # a corner of a cell: the vertex and the sums of its three edges, all in the window
    sums = np.einsum("mk,ckj->cmj", _CUBE, signed[edges])  # corners x 8 x 3
    along = sums @ normals.T
    low, high = -support - along.min(axis=1), support - along.max(axis=1)
    reach = 0.5 * np.abs(axes).sum(axis=0)
    points = (2.0 * rng.random((int(samples), 3)) - 1.0) * reach
    points = points[np.all(np.abs(points @ normals.T) <= support, axis=1)]
    if not len(points):
        raise ValueError("samples: too few points to fall in the window")
    rows = np.zeros((len(points), 16), dtype=np.int64)
    _window_star_rows(points, normals, low, high, species, edges, rows)
    weights = 21 ** np.arange(4) * 11**12  # one number per row: fewer than 21 and 11 of each
    codes = rows[:, :4] @ weights + rows[:, 4:] @ 11 ** np.arange(12)
    _, first, counts = np.unique(codes, return_index=True, return_counts=True)
    return {_star_key(rows[i]): n / len(points) for i, n in zip(first, counts, strict=True)}


def _prism_adjacency(plane: dict, dims: int, periodic: bool, kind: str):
    """Adjacency ``kind`` of the layers of prisms over a planar tiling (one layer in 2D):
    prisms share a face across a planar edge or a base, an edge across a planar vertex or
    the edges of a base, a vertex across a planar vertex within a layer or the next."""
    m = len(plane["centres"])
    layers = plane["layers"] if dims == 3 else 1
    if layers == 1:
        return plane["faces"] if kind == "faces" else plane["vertices"]
    faces2 = _pairs(*plane["faces"])
    vertices2 = _pairs(*plane["vertices"])
    same_cell = (np.arange(m), np.arange(m))

    def join(*parts):
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def stack(within, across):
        rows, cols = [], []
        for layer in range(layers):
            rows.append(within[0] + layer * m)
            cols.append(within[1] + layer * m)
            up = layer + 1
            if up == layers:
                if not (periodic and layers > 2):
                    continue
                up = 0
            rows.append(across[0] + layer * m)
            cols.append(across[1] + up * m)
        return _csr(m * layers, np.concatenate(rows), np.concatenate(cols))

    if kind == "faces":
        return stack(faces2, same_cell)
    if kind == "edges":
        return stack(vertices2, join(faces2, same_cell))
    return stack(vertices2, join(vertices2, same_cell))


class TilingContext:
    """Neighbourhood of the cells of a tiling on which a rule acts: the counterpart of
    :class:`~morphogen.automata.rules.LatticeContext`, for states holding one value per
    cell. ``n_neighbours`` is a number, or an array when cells differ (Penrose tilings,
    edges of finite patches)."""

    def __init__(self, tiling: Tiling, neighbourhood: str = "moore", radius: float = 1):
        self.tiling = tiling
        self.periodic = tiling.periodic
        self.dims = tiling.dims
        self.boundary = 0
        self.indptr, self.indices = tiling.neighbours(neighbourhood, radius)
        sizes = np.diff(self.indptr)
        regular = len(sizes) > 0 and bool(np.all(sizes == sizes[0]))
        self.n_neighbours = int(sizes[0]) if regular else sizes
        self.cache: dict = {}

    def counts(self, state: np.ndarray) -> np.ndarray:
        return self.tiling.counts(state, self.indptr, self.indices)

    def interface(self, state: np.ndarray) -> float:
        return self.tiling.interface_density(state)

    def sublattices(self) -> list[np.ndarray]:
        """Classes of cells that are not neighbours (greedy colouring)."""
        colour = self.tiling.colours(self.indptr, self.indices)
        return [colour == c for c in np.unique(colour)]


# -- generators ----------------------------------------------------------------------------


def _regular_plane(name: str, size, h: float) -> dict:
    """Square, triangular or hexagonal tiling of whole rectangular unit cells about
    ``size``: the cell ``i`` of the unit cell ``c`` (C order) is ``c * n_basis + i``."""
    if name == "square":
        cell = np.array([h, h])
        basis = np.array([[0.5, 0.5]])
        kinds = np.zeros(1, dtype=np.int8)
        shapes = [0.5 * h * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])]
        shells = {"faces": h, "vertices": math.sqrt(2) * h}
    elif name == "hexagonal":
        a = h * math.sqrt(2 / math.sqrt(3))  # centre spacing of hexagons of area h^2
        cell = np.array([a, math.sqrt(3) * a])
        basis = np.array([[0.25, 0.25], [0.75, 0.75]])
        kinds = np.zeros(2, dtype=np.int8)
        t = np.radians(90 + 60 * np.arange(6))
        shapes = [a / math.sqrt(3) * np.c_[np.cos(t), np.sin(t)]]
        shells = {"faces": a, "vertices": a}
    else:  # triangles: the Voronoi cells of the honeycomb of their centroids
        b = 2 * h / 3**0.25  # edge of the triangles of area h^2
        rise = math.sqrt(3) / 2 * b
        cell = np.array([b, 2 * rise])
        basis = np.array([[0.5, 1 / 6], [0.0, 1 / 3], [0.0, 2 / 3], [0.5, 5 / 6]])
        kinds = np.array([0, 1, 0, 1], dtype=np.int8)  # pointing up, down
        shapes = [
            np.array([[-b / 2, -rise / 3], [b / 2, -rise / 3], [0.0, 2 * rise / 3]]),
            np.array([[0.0, -2 * rise / 3], [b / 2, rise / 3], [-b / 2, rise / 3]]),
        ]
        shells = {"faces": b / math.sqrt(3), "vertices": 2 * b / math.sqrt(3)}
    n = np.maximum(1, np.rint(np.asarray(size[:2], dtype=float) / cell)).astype(int)
    units = np.stack(np.meshgrid(*(np.arange(m) for m in n), indexing="ij"), -1).reshape(-1, 2)
    kinds = np.tile(kinds, len(units))
    k = np.array([len(shapes[c]) for c in kinds])
    return {
        "centres": ((units[:, None, :] + basis[None]) * cell).reshape(-1, 2),
        "poly": np.concatenate([shapes[c] for c in kinds]),
        "poly_ptr": np.concatenate([[0], np.cumsum(k)]).astype(np.int64),
        "areas": np.full(len(kinds), h * h),
        "kinds": kinds,
        "box": n * cell,
        "lattice": (n, basis, cell),
        "shells": shells,
    }


def _dual_cells(e: np.ndarray, gamma, half, families):
    """Cells of the tiling dual to the multigrid ``x . e_j + gamma_j = k`` (``k``
    integer, ``e`` the unit vectors of the grids, in rows) whose centres lie within
    ``half`` (edges) of the origin: one cell per intersection of hyperplanes (lines in 2D,
    planes in 3D) of the ``d`` families of each tuple of ``families``.

    The mesh of a point ``x`` maps to the vertex ``sum_j K_j e_j``, ``K_j = ceil(x . e_j +
    gamma_j)``, and an intersection to the cell of edges ``e_r`` (``r`` in its families)
    whose corners are those of the meshes around it (de Bruijn 1981; Kramer & Neri 1984).
    Since ``sum_j (x . e_j) e_j = (n / d) x`` for these grids, the tiling is the grid
    scaled by ``n / d``, up to a displacement of at most ``sum_j (1 + |gamma_j|)``, which
    bounds the hyperplanes to generate. Yields, for each tuple, its integer indices ``K``
    of the corner of the cell (of the meshes on the negative side of all its
    hyperplanes), its centre and the other grids' values at the intersection.
    """
    n, d = e.shape
    zr = (float(np.linalg.norm(half)) + n + float(np.abs(gamma).sum())) / (n / d) + 1.0
    for fam in families:
        fam = list(fam)
        ranges = [np.arange(math.floor(gamma[r] - zr), math.ceil(gamma[r] + zr) + 1) for r in fam]
        k = np.stack([g.ravel() for g in np.meshgrid(*ranges, indexing="ij")], axis=1)
        z = np.linalg.solve(e[fam], (k - gamma[fam]).T).T
        w = z @ e.T + gamma
        big_k = np.ceil(w).astype(np.int64)
        big_k[:, fam] = k
        centre = big_k @ e + 0.5 * e[fam].sum(axis=0)
        keep = np.all(np.abs(centre) <= half, axis=1)
        yield fam, big_k[keep], centre[keep], np.delete(w[keep], fam, axis=1)


def _check_regular(other, gamma, what: str) -> None:
    """Refuse singular grids: more hyperplanes than the dimension through one point."""
    if np.any(np.abs(other - np.round(other)) < 1e-9):
        raise ValueError(
            f"offsets {np.round(gamma, 12).tolist()}: a singular {what}, whose cells would "
            "overlap (offsets drawn at random are regular)"
        )


def _vertex_ids(keys: np.ndarray) -> np.ndarray:
    """Number of the vertex of each row of integer indices ``keys`` (the same numbers as
    the rows' lexicographic order: the indices are packed into one integer)."""
    lo = keys.min(axis=0)
    base = int((keys.max(axis=0) - lo).max()) + 1
    if base ** keys.shape[1] >= 2**62:
        return np.unique(keys, axis=0, return_inverse=True)[1].ravel().astype(np.int64)
    packed = np.zeros(len(keys), dtype=np.int64)
    for j in range(keys.shape[1]):
        packed = packed * base + (keys[:, j] - lo[j])
    return np.unique(packed, return_inverse=True)[1].ravel().astype(np.int64)


def _shared_vertices(vertex_ids: np.ndarray, n: int):
    """Number of vertices shared by the pairs of cells (COO, the cells sharing one at
    least) from the vertex numbers of each cell (``(n, k)``)."""
    k = vertex_ids.shape[1]
    incidence = sparse.csr_matrix(
        (np.ones(k * n, dtype=np.int32), (np.repeat(np.arange(n), k), vertex_ids.ravel())),
        shape=(n, int(vertex_ids.max()) + 1),
    )
    return (incidence @ incidence.T).tocoo()


def _polygon_sides(plane: dict):
    """Sides of the rhombi of a multigrid tiling: their vertex numbers (sorted pairs), the
    cell of each and its vector (in the reference tiling)."""
    ids = plane["vertex_ids"].reshape(-1, 4)
    poly = plane["poly"].reshape(-1, 4, 2)
    keys = np.sort(np.stack([ids, np.roll(ids, -1, axis=1)], axis=2), axis=2).reshape(-1, 2)
    owner = np.repeat(np.arange(len(ids)), 4)
    return keys, owner, (np.roll(poly, -1, axis=1) - poly).reshape(-1, 2)


def _rhombohedron_faces(solid: dict):
    """Faces of the rhombohedra: their vertex numbers (sorted), the cell of each and its
    area (the cross product of the two edges spanning it)."""
    ids = solid["vertex_ids"]
    keys = np.sort(ids[:, PARALLELEPIPED_FACES], axis=2).reshape(-1, 4)
    owner = np.repeat(np.arange(len(ids)), 6)
    f = solid["frames"][solid["triple"]]  # edge vectors in rows
    span = np.array([[1, 2], [1, 2], [0, 2], [0, 2], [0, 1], [0, 1]])  # of each face
    areas = np.linalg.norm(np.cross(f[:, span[:, 0]], f[:, span[:, 1]]), axis=2).ravel()
    return keys, owner, areas


def _shared_measures(keys, owner, measures, rows, cols) -> np.ndarray:
    """Measures of the sides (or faces) shared by the pairs of cells ``(rows, cols)``: the
    sides of the cells ``owner`` met twice among ``keys`` (their vertex numbers), summed
    when a pair shares several (on the smallest tori); 0 for pairs sharing none."""
    order = np.lexsort(keys.T[::-1])
    same = np.all(keys[order[1:]] == keys[order[:-1]], axis=1)
    first, second = order[:-1][same], order[1:][same]
    i, j, w = owner[first], owner[second], measures[first]
    n = int(owner.max()) + 1
    m = sparse.csr_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n))
    if not len(rows):
        return np.zeros(0)
    return np.asarray(m[rows, cols]).ravel()


def _multigrid_plane(name: str, size, h: float, gamma) -> dict:
    """Rhombi of the Penrose or Ammann-Beenker tiling dual to the multigrid of offsets
    ``gamma`` (the pentagrid, or four grids at 45°; de Bruijn 1981) whose centres lie
    within one edge of the box ``[0, size]``.

    The lines ``x . e_j + gamma_j = k`` (``e_j`` at angles ``2 pi j / 5`` or ``pi j / 4``,
    ``k`` integer) split the plane into meshes, mapped to the vertices of the tiling; the
    intersection of lines of the families ``r < s`` maps to the rhombus of edges
    ``e_r, e_s``: thick when ``s - r`` is 1 or 4, thin otherwise (Penrose); a square when
    ``s - r = 2``, a 45° rhombus otherwise (Ammann-Beenker). Kind 0 is the larger tile.
    """
    edge = tile_edge(name, h)
    e = _star(name)
    gamma = np.asarray(gamma, dtype=float)
    half = np.asarray(size[:2], dtype=float) / (2 * edge) + 1.0  # in edges
    pairs = list(itertools.combinations(range(len(e)), 2))
    rhombi = _Rhombi(name, e)
    for (r, s), big_k, centre, third in _dual_cells(e, gamma, half, pairs):
        _check_regular(third, gamma, rhombi.what)
        rhombi.add(r, s, big_k, centre)
    return rhombi.plane(edge, 0.5 * np.asarray(size[:2], dtype=float), None, lambda keys: keys)


def _approximant_plane(name: str, size, h: float, gamma, order) -> dict:
    """Rhombi of the periodic approximant of order ``order`` of the Penrose or
    Ammann-Beenker tiling (the multigrid of :func:`approximant_grid` and offsets
    ``gamma``) filling a whole number of periods, about ``size``: the box of the tiling.

    The intersections of the lines of families ``r < s`` in the box of grid periods map
    to the rhombi of edges ``e_r, e_s`` as on the patches of :func:`_multigrid_plane`,
    ``|det(g_r, g_s)|`` per period; the vertices, integer vectors reduced modulo the
    periods, are those of the torus, whose adjacency wraps across the faces of the box.
    """
    edge = tile_edge(name, h)
    e = _star(name)
    grid = approximant_grid(name, order)
    counts = _period_counts(name, order, size[:2], h)
    reduce = _reduction(grid * counts)
    rhombi = _Rhombi(name, e)
    pairs = list(itertools.combinations(range(len(e)), 2))
    cells = _periodic_dual_cells(e, grid, gamma, counts, pairs, reduce, rhombi.what)
    for (r, s), big_k in cells:
        rhombi.add(r, s, big_k, big_k @ e + 0.5 * (e[r] + e[s]))
    box = np.diag(e.T @ grid) * counts * edge
    plane = rhombi.plane(edge, np.zeros(2), box, reduce)
    m = len(plane["centres"])
    corners = plane["poly"].reshape(m, 4, 2)
    why = _torus_defect(plane["vertex_ids"].reshape(m, 4), [_RHOMBUS_SIDES], corners,
                        plane["centres"], box)  # fmt: skip
    if why is not None:
        raise _degenerate(name, order, counts, h, why)
    return plane


#: Corners of the sides of a rhombus (its polygon, counterclockwise).
_RHOMBUS_SIDES = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])


def _torus_defect(ids, groups, corners, centres, box) -> str | None:
    """What makes a tiling of the torus degenerate, or None: ``ids``, the vertex numbers
    of the corners of each cell (``n x k``); ``groups``, the corners of its sides, then of
    its faces in 3D (index arrays); ``corners``, their positions from the cell centres.

    On a torus too small for its cells, a cell meets its own image (a vertex twice among
    its corners), or two cells meet through several images: the vertices they share,
    counted before the adjacency is deduplicated, are then not those of a single common
    vertex, side or face, and their CSR neighbourhoods would be wrong (several faces, or
    two vertices taken for an edge); or two cells share a face through another image than
    the nearest, which the fluxes of the Laplacian take."""
    n = len(ids)
    ranked = np.sort(ids, axis=1)
    if np.any(ranked[:, 1:] == ranked[:, :-1]):
        return "a cell meets its own image"
    shared = _shared_vertices(ids, n)  # with multiplicity: before any deduplication
    other = shared.row != shared.col
    rows, cols, count = shared.row[other], shared.col[other], shared.data[other]
    common = []
    for g in groups:  # sides (faces) shared by the pairs of cells sharing vertices
        keys = _vertex_ids(np.sort(ids[:, g], axis=2).reshape(-1, g.shape[1]))
        m = _shared_vertices(keys.reshape(n, len(g)), n).tocsr()
        common.append(np.asarray(m[rows, cols]).ravel())
    single = (count == 1) | ((count == 2) & (common[0] == 1))
    if len(groups) == 2:  # rhombohedra: a vertex, an edge or a face
        single |= (count == 4) & (common[1] == 1)
    if not np.all(single):
        return "two cells meet through several images: several faces, edges or vertices"
    g = groups[-1]  # each face, seen from its two cells through the nearest image
    keys = np.sort(ids[:, g], axis=2).reshape(-1, g.shape[1])
    at = corners[:, g].mean(axis=2).reshape(-1, corners.shape[2])
    owner = np.repeat(np.arange(n), len(g))
    order = np.lexsort(keys.T[::-1])
    pair = np.all(keys[order[1:]] == keys[order[:-1]], axis=1)
    p, q = order[:-1][pair], order[1:][pair]
    d = centres[owner[q]] - centres[owner[p]]
    d -= box * np.round(d / box)
    if np.abs(at[p] - d - at[q]).max() > 1e-9 * np.abs(corners).max():
        return "two cells share a face through another image than the nearest"
    return None


def _degenerate(name: str, order, counts, h: float, why: str) -> ValueError:
    period = approximant_period(name, order, h)
    return ValueError(
        f"approximant {order} of the {name} tiling: {np.asarray(counts).tolist()} periods of "
        f"{np.round(period, 3).tolist()} make a torus too small for its cells ({why}); take "
        "a larger box (more periods) or a higher order"
    )


def _period_counts(name: str, order, size, h: float) -> np.ndarray:
    """Number of periods of an approximant along each axis of a box: the nearest whole
    number, refused when the box is less than half a period."""
    period = approximant_period(name, order, h)
    counts = np.rint(np.asarray(size, dtype=float) / period).astype(int)
    if np.any(counts < 1):
        raise ValueError(
            f"approximant {order}: the box {np.round(size, 4).tolist()} is less than half its "
            f"period {np.round(period, 4).tolist()} (take a lower order or a larger box)"
        )
    return counts


class _Rhombi:
    """Rhombi of a planar multigrid tiling, gathered by families (:func:`_dual_cells`,
    :func:`_periodic_dual_cells`): kind 0 the larger tile."""

    def __init__(self, name: str, e: np.ndarray) -> None:
        n, step = MULTIGRIDS[name]
        self.n, self.e = n, e
        self.sine = np.abs(np.sin(step * (np.arange(n)[None] - np.arange(n)[:, None])))
        self.what = (
            f"{'pentagrid' if n == 5 else 'multigrid'}, three of its lines through one point"
        )
        self.centres, self.polys, self.keys, self.kinds, self.areas = [], [], [], [], []

    def add(self, r: int, s: int, big_k: np.ndarray, centre: np.ndarray) -> None:
        """Rhombi of edges ``e_r, e_s``, their corners ``big_k`` and centres (in edges)."""
        e, n, sine = self.e, self.n, self.sine
        corner = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        if e[r, 0] * e[s, 1] - e[r, 1] * e[s, 0] < 0:
            corner = corner[::-1]
        offsets = corner @ np.array([e[r], e[s]]) - 0.5 * (e[r] + e[s])
        unit = np.zeros((4, n), dtype=np.int64)
        unit[:, r] = corner[:, 0]
        unit[:, s] = corner[:, 1]
        self.centres.append(centre)
        self.polys.append(np.broadcast_to(offsets, (len(centre), 4, 2)))
        self.keys.append(big_k[:, None, :] + unit[None])
        kind = 0 if sine[r, s] > sine.max() - 1e-9 else 1
        self.kinds.append(np.full(len(centre), kind, np.int8))
        self.areas.append(np.full(len(centre), sine[r, s]))

    def plane(self, edge: float, shift, box, reduce) -> dict:
        """The planar tiling (centres shifted by ``shift``, wrapped into ``box`` when it is
        periodic, the corners reduced by ``reduce`` modulo its periods)."""
        centres = np.concatenate(self.centres) * edge + shift
        if box is not None:
            centres = np.mod(centres, box)
        m = len(centres)
        keys = reduce(np.concatenate(self.keys)).reshape(-1, self.n)
        vertex_ids = _vertex_ids(keys)
        shared = _shared_vertices(vertex_ids.reshape(m, 4), m)
        edge_pairs = shared.data >= 2
        return {
            "centres": centres,
            "poly": np.concatenate(self.polys).reshape(-1, 2) * edge,
            "poly_ptr": 4 * np.arange(m + 1, dtype=np.int64),
            "areas": np.concatenate(self.areas) * edge**2,
            "kinds": np.concatenate(self.kinds),
            "box": box,
            "edge": edge,
            "vertex_ids": vertex_ids,
            "vertex_index": keys.sum(axis=1),
            "faces": _csr(m, shared.row[edge_pairs], shared.col[edge_pairs]),
            "vertices": _csr(m, shared.row, shared.col),
        }


def _cubic_solid(name: str, size, h: float) -> dict:
    """Body- or face-centred cubic lattice of whole cubic cells, about ``size``."""
    if name == "bcc":
        a = 2 ** (1 / 3) * h  # two cells of volume h^3 per cube
        basis = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
        corners = set()
        for p in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.add(tuple(np.array([0.0, sy, 2 * sz])[list(p)]))
        prototype = np.array(sorted(corners)) * a / 4  # truncated octahedron
        shells = {"faces": a, "edges": a, "vertices": a}
        volume = a**3 / 2
    else:
        a = 4 ** (1 / 3) * h  # four cells of volume h^3 per cube
        basis = 0.25 + np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        axes = np.concatenate([np.eye(3), -np.eye(3)]) * a / 2
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1], indexing="ij")).reshape(3, -1).T
        prototype = np.concatenate([axes, signs * a / 4])  # rhombic dodecahedron
        shells = {"faces": a / math.sqrt(2), "edges": a / math.sqrt(2), "vertices": a}
        volume = a**3 / 4
    n = np.maximum(1, np.rint(np.asarray(size, dtype=float) / a)).astype(int)
    cells = np.stack(np.meshgrid(*(np.arange(m) for m in n), indexing="ij"), -1).reshape(-1, 3)
    centres = ((cells[:, None, :] + basis[None]) * a).reshape(-1, 3)
    return {
        "centres": centres,
        "box": n * a,
        "lattice": (n, basis, np.full(3, a)),
        "scale": np.asarray(size, dtype=float) / (n * a),
        "shells": shells,
        "prototype": prototype,
        "volume": volume,
    }


#: Corners of the unit cube, the bits of the corners of a parallelepiped (x fastest).
_CUBE = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=np.int64)
#: The six faces of a parallelepiped, as indices of its corners in :data:`_CUBE`
#: (around each face: x = 0, 1, y = 0, 1, z = 0, 1).
PARALLELEPIPED_FACES = np.array(
    [[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]]
)
#: Its twelve edges: the corners whose bits differ by one.
_PARALLELEPIPED_EDGES = np.array(
    [(i, i | 1 << b) for b in range(3) for i in range(8) if not i >> b & 1]
)


def _icosahedral_solid(size, h: float, gamma) -> dict:
    """Golden rhombohedra of the icosahedral tiling dual to the six grids of planes
    ``x . e_j + gamma_j = k`` normal to the fivefold axes ``e_j`` of the icosahedron, whose
    centres lie within one edge of the box ``[0, size]`` (the generalized dual method of
    de Bruijn, Kramer & Neri 1984, Socolar & Steinhardt 1986).

    The intersection of planes of the families ``r < s < t`` maps to the rhombohedron of
    edges ``e_r, e_s, e_t``: prolate (kind 0) when the product of their pairwise cosines
    (``±1/√5``) is positive, oblate (kind 1) otherwise. Its corners are integer vectors
    of Z^6, which number the vertices shared by the cells without tolerance.
    """
    edge = tile_edge("icosahedral", h)
    e = _icosahedral_axes()
    gamma = np.asarray(gamma, dtype=float)
    half = np.asarray(size, dtype=float) / (2 * edge) + 1.0  # in edges
    triples = list(itertools.combinations(range(6), 3))
    cells = []
    for fam, big_k, centre, other in _dual_cells(e, gamma, half, triples):
        _check_regular(other, gamma, _ICOSAHEDRAL_SINGULAR)
        cells.append((fam, big_k, centre))
    shift = 0.5 * np.asarray(size, dtype=float)
    return _rhombohedra(e, cells, edge, np.ones(3), shift, None, lambda keys: keys)


_ICOSAHEDRAL_SINGULAR = "icosahedral grid, four of its planes through one point"


def _approximant_solid(size, h: float, gamma, order) -> dict:
    """Golden rhombohedra of the periodic (cubic) approximant of order ``order`` of the
    icosahedral tiling (the grids of planes of :func:`approximant_grid`, offsets
    ``gamma``) filling a whole number of periods, scaled to the box ``size`` along each
    axis (the scale is part of the cells: edge vectors, corners, volumes). The
    intersections of planes of the families ``r < s < t`` in the box of grid periods map
    to the rhombohedra of edges ``e_r, e_s, e_t``, ``|det(g_r, g_s, g_t)|`` per period, and
    their vertices are reduced modulo the periods (those of the 3-torus)."""
    edge = tile_edge("icosahedral", h)
    e = _icosahedral_axes()
    grid = approximant_grid("icosahedral", order)
    size = np.asarray(size, dtype=float)
    counts = _period_counts("icosahedral", order, size, h)
    reduce = _reduction(grid * counts)
    triples = list(itertools.combinations(range(6), 3))
    cells = [
        (fam, big_k, big_k @ e + 0.5 * e[fam].sum(axis=0))
        for fam, big_k in _periodic_dual_cells(
            e, grid, gamma, counts, triples, reduce, _ICOSAHEDRAL_SINGULAR
        )
    ]
    scale = size / (np.diag(e.T @ grid) * counts * edge)
    solid = _rhombohedra(e, cells, edge, scale, np.zeros(3), size, reduce)
    frames = solid["frames"]  # the corners from the centres, for each triple
    rel = np.einsum("vk,tkj->tvj", _CUBE, frames) - 0.5 * frames.sum(axis=1)[:, None]
    groups = [_PARALLELEPIPED_EDGES, PARALLELEPIPED_FACES]
    why = _torus_defect(solid["vertex_ids"], groups, rel[solid["triple"]], solid["centres"], size)
    if why is not None:
        raise _degenerate("icosahedral", order, counts, h, why)
    return solid


def _rhombohedra(e, cells, edge: float, scale, shift, box, reduce) -> dict:
    """The rhombohedra of the triples ``fam`` of ``cells`` (``(fam, corners K, centres)``,
    in edges), scaled along the axes by ``scale`` and shifted by ``shift``: corners are
    integer vectors of Z^6 (reduced by ``reduce`` modulo the periods of a periodic
    ``box``, into which the centres are wrapped) that number the vertices shared by the
    cells without tolerance."""
    triples = list(itertools.combinations(range(6), 3))
    frames = np.array([e[list(f)] for f in triples]) * edge * scale  # edge vectors, in rows
    dets = np.abs(np.linalg.det(frames))
    keys, triple, centres = [], [], []
    for fam, big_k, centre in cells:
        unit = np.zeros((8, 6), dtype=np.int64)
        unit[:, fam] = _CUBE
        centres.append(centre)
        keys.append(reduce(big_k[:, None, :] + unit[None]).astype(np.int32))  # small indices
        triple.append(np.full(len(centre), triples.index(tuple(fam)), dtype=np.int8))
    centres = np.concatenate(centres) * edge * scale + shift
    if box is not None:
        centres = np.mod(centres, box)
    n = len(centres)
    triple = np.concatenate(triple)
    vertex_ids = _vertex_ids(np.concatenate(keys).reshape(-1, 6)).reshape(n, 8)
    shared = _shared_vertices(vertex_ids, n)
    prolate = dets > dets.mean()
    # the corners of the cells, from their centres: the largest distance to them
    half = 0.5 * frames.sum(axis=1)
    reach = np.linalg.norm(_CUBE @ frames - half[:, None], axis=2).max()
    return {
        "centres": centres,
        "scale": np.ones(3),
        "box": box,
        "edge": edge,
        "volumes": dets[triple],
        "kinds": np.where(prolate[triple], 0, 1).astype(np.int8),
        "corner": centres - half[triple],
        "triple": triple,
        "frames": frames,
        "inverse": np.linalg.inv(frames),
        "circumradius": float(reach),
        "vertex_ids": vertex_ids,
        "faces": _csr(n, shared.row[shared.data == 4], shared.col[shared.data == 4]),
        "edges": _csr(n, shared.row[shared.data >= 2], shared.col[shared.data >= 2]),
        "vertices": _csr(n, shared.row, shared.col),
    }


@njit(parallel=True, cache=True)
def _in_parallelepipeds(
    points, candidates, centres, corner, triple, inverse, box, periodic, out, found
):
    """First candidate cell (parallelepiped of centre ``centres``, corner ``corner`` and
    edge vectors the rows of ``inv(inverse[triple])``) containing each point (the nearest
    image of the cell in a periodic ``box``); the nearest candidate and ``found = False``
    when none does."""
    for p in prange(points.shape[0]):
        out[p] = candidates[p, 0]
        found[p] = False
        for q in range(candidates.shape[1]):
            c = candidates[p, q]
            if c < 0 or c >= corner.shape[0]:
                break
            # the point from the corner of the nearest image of the cell
            dx = points[p, 0] - centres[c, 0]
            dy = points[p, 1] - centres[c, 1]
            dz = points[p, 2] - centres[c, 2]
            if periodic:
                dx -= box[0] * np.floor(dx / box[0] + 0.5)
                dy -= box[1] * np.floor(dy / box[1] + 0.5)
                dz -= box[2] * np.floor(dz / box[2] + 0.5)
            dx += centres[c, 0] - corner[c, 0]
            dy += centres[c, 1] - corner[c, 1]
            dz += centres[c, 2] - corner[c, 2]
            m = inverse[triple[c]]
            inside = True
            for a in range(3):  # coordinates of the point along the edges of the cell
                u = dx * m[0, a] + dy * m[1, a] + dz * m[2, a]
                if u < -1e-9 or u > 1.0 + 1e-9:
                    inside = False
                    break
            if inside:
                out[p] = c
                found[p] = True
                break


def make_tiling(
    name: str,
    size,
    cell_size: float = 1.0,
    *,
    dims: int = 2,
    periodic: bool = True,
    origin=(0.0, 0.0, 0.0),
    offsets=None,
    rng: np.random.Generator | None = None,
    approximant=None,
) -> Tiling:
    """Tiling ``name`` (see :data:`LATTICES`) of the box ``origin + [0, size]``.

    ``dims=2``: one layer of planar cells through the thickness of the box; ``dims=3``:
    layers of prisms ``cell_size`` high (planar tilings), or the ``bcc``, ``fcc`` and
    ``icosahedral`` lattices. ``offsets``: the offsets of the grids of a quasi-periodic
    tiling, drawn from ``rng`` when not given (uniform, centred): five for the pentagrid
    of a Penrose tiling (summing to 0; other sums give de Bruijn's generalized Penrose
    tilings), four for an Ammann-Beenker tiling, six for an icosahedral tiling (any
    regular offsets give tilings of the same local isomorphism class). Singular grids,
    more lines (planes) than the dimension through one point in the box, are refused.

    Quasi-periodic tilings tile no periodic box: ``approximant`` (an order ``k >= 1``, or
    one per axis) gives their periodic approximant of that order, the same tiles periodic
    in a rectangle or a cube (:func:`approximant_grid`, :func:`approximant_period`). The
    box holds the nearest whole number of periods along each axis (at least half of one),
    scaled to fit it exactly like the unit cells of the periodic tilings: the tiles keep
    their shapes when the box is a multiple of the period.
    """
    if name not in LATTICES:
        raise ValueError(f"lattice: one of {LATTICES}, got {name!r}")
    if dims not in (2, 3):
        raise ValueError("dims: 2 or 3")
    if name not in PLANAR and dims != 3:
        raise ValueError(f"the {name} lattice is three-dimensional (dims=3)")
    if approximant is not None:
        if name not in QUASIPERIODIC:
            raise ValueError(f"approximant: quasi-periodic tilings only {QUASIPERIODIC}")
        if not periodic:
            raise ValueError(
                "approximant: periodic tilings only (non-periodic boxes take patches of the "
                "quasi-periodic tiling, or windows of a larger periodic approximant)"
            )
        orders = _orders(name, approximant)
        approximant = orders[0] if len(set(orders)) == 1 else list(orders)
    elif name in QUASIPERIODIC and periodic:
        raise ValueError(
            f"the {name} tiling is quasi-periodic: it tiles no periodic box (use a non-periodic "
            "domain, or a periodic approximant: approximant=1, 2, ...)"
        )
    size = np.asarray(size, dtype=float).reshape(3)
    h = float(cell_size)
    if not (h > 0 and np.all(size > 0)):
        raise ValueError("cell_size and size: must be positive")
    spec = {
        "name": name,
        "size": size.tolist(),
        "cell_size": h,
        "dims": int(dims),
        "periodic": bool(periodic),
        "origin": [float(v) for v in origin],
    }
    if name in QUASIPERIODIC:
        k = 6 if name == "icosahedral" else MULTIGRIDS[name][0]
        if offsets is None:
            u = (np.random.default_rng() if rng is None else rng).random(k)
            offsets = u - u.mean()
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (k,):
            raise ValueError(f"offsets: {k} numbers for the {name} tiling, got {offsets.shape}")
        spec["offsets"] = offsets.tolist()
        if approximant is not None:
            spec["approximant"] = approximant
    if name == "icosahedral":
        if approximant is not None:
            return Tiling(spec, None, _approximant_solid(size, h, offsets, approximant))
        return Tiling(spec, None, _icosahedral_solid(size, h, offsets))
    if name not in PLANAR:
        return Tiling(spec, None, _cubic_solid(name, size, h))
    if approximant is not None:
        plane = _approximant_plane(name, size, h, offsets, approximant)
        scale = list(size[:2] / plane["box"])
    elif name in QUASIPERIODIC:
        plane = _multigrid_plane(name, size, h, offsets)
        scale = [1.0, 1.0]
    else:
        plane = _regular_plane(name, size, h)
        for kind, r in plane.pop("shells").items():
            plane[kind] = _lattice_adjacency(*plane["lattice"], r * (1 + _EPS), periodic)
        scale = list(size[:2] / plane["box"])
    if dims == 2:
        plane["layers"], plane["layer_height"] = 1, float(size[2])
        scale.append(1.0)
    else:
        layers = max(1, round(size[2] / h))
        plane["layers"], plane["layer_height"] = layers, h
        scale.append(float(size[2] / (layers * h)))
    plane["scale"] = np.array(scale)
    return Tiling(spec, plane, None)
