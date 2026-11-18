"""Lattices of cellular automata: neighbourhoods and neighbour sums.

An automaton lives on the cells ``(i, j, k)`` of a regular grid, with the conventions of
voxel images (arrays indexed along x, y, z). Two-dimensional automata use arrays of shape
``(nx, ny, 1)`` and planar neighbourhoods (``dims=2``).

Boundaries are periodic (the lattice is a torus) or fixed: cells outside the lattice
then hold a constant ``boundary`` value.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy import fft as sfft

from morphogen import settings

#: Neighbourhood shapes: ``|d|`` is the L1 norm (von Neumann), the max norm (Moore), both
#: (edges: faces and edges of the unit cube), or the Euclidean norm.
NEIGHBOURHOODS = ("von_neumann", "moore", "edges", "euclidean")


def offsets(
    neighbourhood: str = "moore", radius: float = 1, *, dims: int = 3, include_center: bool = False
) -> np.ndarray:
    """Integer offsets ``(k, 3)`` of the cells of a neighbourhood.

    ``von_neumann``: ``|dx| + |dy| + |dz| <= r`` (6 cells for r = 1);
    ``moore``: ``max |d| <= r`` (26 cells); ``edges``: Moore cells with
    ``|dx| + |dy| + |dz| <= 2r`` (18 cells for r = 1); ``euclidean``:
    ``dx^2 + dy^2 + dz^2 <= r^2`` (a digital ball). ``dims=2`` keeps ``dz = 0``.
    """
    if neighbourhood not in NEIGHBOURHOODS:
        raise ValueError(f"neighbourhood: one of {NEIGHBOURHOODS}, got {neighbourhood!r}")
    if dims not in (2, 3):
        raise ValueError("dims: 2 or 3")
    r = float(radius)
    if not r >= 1:
        raise ValueError(f"radius: must be at least 1, got {radius!r}")
    m = int(np.floor(r))
    span = np.arange(-m, m + 1)
    dz = span if dims == 3 else np.array([0])
    d = np.stack(np.meshgrid(span, span, dz, indexing="ij"), axis=-1).reshape(-1, 3)
    l1 = np.abs(d).sum(axis=1)
    linf = np.abs(d).max(axis=1)
    if neighbourhood == "von_neumann":
        keep = l1 <= r
    elif neighbourhood == "moore":
        keep = linf <= r
    elif neighbourhood == "edges":
        keep = (linf <= r) & (l1 <= 2 * r)
    else:
        keep = (d * d).sum(axis=1) <= r * r + 1e-9
    if not include_center:
        keep &= l1 > 0
    return np.ascontiguousarray(d[keep], dtype=np.int64)


@njit(parallel=True, cache=True)
def _count(state, offs, periodic, boundary, out):
    nx, ny, nz = state.shape
    k = offs.shape[0]
    for i in prange(nx):
        for j in range(ny):
            for l in range(nz):
                s = 0
                for o in range(k):
                    ii = i + offs[o, 0]
                    jj = j + offs[o, 1]
                    ll = l + offs[o, 2]
                    if periodic:
                        ii %= nx
                        jj %= ny
                        ll %= nz
                        s += state[ii, jj, ll]
                    elif 0 <= ii < nx and 0 <= jj < ny and 0 <= ll < nz:
                        s += state[ii, jj, ll]
                    else:
                        s += boundary
                out[i, j, l] = s


def neighbour_counts(
    state: np.ndarray, offs: np.ndarray, *, periodic: bool = True, boundary: int = 0
) -> np.ndarray:
    """Number of cells in state 1 among the neighbours ``offs`` of every cell."""
    s = np.ascontiguousarray(state, dtype=np.uint8)
    if s.ndim != 3:
        raise ValueError("the state must be a 3D array (use (nx, ny, 1) in 2D)")
    out = np.empty(s.shape, dtype=np.int32)
    _count(s, np.ascontiguousarray(offs, dtype=np.int64), bool(periodic), int(boundary), out)
    return out


def kernel_sum(
    field: np.ndarray,
    offs: np.ndarray,
    weights=None,
    *,
    periodic: bool = True,
    boundary: float = 0.0,
) -> np.ndarray:
    """Weighted sum of ``field`` over the neighbourhood ``offs`` of every cell (float).

    Computed by FFT (circular convolution), which is efficient for large neighbourhoods;
    fixed boundaries pad the field with the ``boundary`` value.
    """
    f = np.asarray(field, dtype=np.float64)
    offs = np.asarray(offs, dtype=np.int64)
    w = np.ones(len(offs)) if weights is None else np.asarray(weights, dtype=float)
    pad = 0 if periodic else int(np.abs(offs).max(initial=0))
    if pad:
        f = np.pad(f, pad, constant_values=boundary)
    shape = f.shape
    kernel = np.zeros(shape)
    # out[x] = sum_o w_o f[x + o] = (f * k)[x] with k[-o] = w_o
    np.add.at(kernel, tuple((-offs[:, a]) % shape[a] for a in range(3)), w)
    workers = settings.workers()
    out = sfft.irfftn(
        sfft.rfftn(f, workers=workers) * sfft.rfftn(kernel, workers=workers),
        s=shape,
        workers=workers,
    )
    if pad:
        out = out[pad:-pad, pad:-pad, pad:-pad]
    return out


def sublattices(neighbourhood: str, shape, dims: int = 3) -> list[np.ndarray]:
    """Partition of the cells into sublattices whose cells are not neighbours (radius 1).

    Used by sequential updates: von Neumann neighbourhoods need two colours (parity of
    ``i + j + k``), Moore and edges neighbourhoods eight (parities of i, j, k; four in
    2D). On periodic lattices with an odd number of cells along an axis, the cells on
    both sides of the seam share a colour.
    """
    i, j, k = np.indices(shape)
    if neighbourhood == "von_neumann":
        colour = (i + j + k) % 2
    else:
        colour = (i % 2) + 2 * (j % 2) + (4 * (k % 2) if dims == 3 else 0)
    return [colour == c for c in np.unique(colour)]
