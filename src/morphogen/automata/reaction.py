"""Reaction-diffusion on lattices (continuous-state automata): Gray-Scott model.

Two species ``u`` (substrate) and ``v`` (activator) react and diffuse (Pearson 1993):

    du/dt = D_u lap(u) - u v^2 + F (1 - u)
    dv/dt = D_v lap(v) + u v^2 - (F + k) v

integrated with explicit Euler steps of unit time on unit cells (7-point Laplacian in
3D, 5-point in 2D), with periodic or zero-flux boundaries. Depending on the feed rate
``F`` and the kill rate ``k``, the activator settles into spots, worms, labyrinths or
holes of a characteristic size (Turing patterns); thresholding ``v`` gives two-phase
microstructures.

:func:`gray_scott_tiling` integrates the same system on the cells of any tiling
(:mod:`morphogen.automata.tilings`) with the finite-volume Laplacian of the tiling: the
flux through each face is its length or area over the distance between the centres,
divided by the cell's area or volume (two-point fluxes; the 5- and 7-point Laplacians on
the square tiling). The free edges of finite patches have no flux.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy import ndimage, sparse

#: Feed and kill rates of a few pattern families (with D_u = 0.16, D_v = 0.08).
GRAY_SCOTT_PRESETS = {
    "spots": (0.030, 0.062),
    "worms": (0.046, 0.065),
    "labyrinth": (0.037, 0.060),
    "holes": (0.039, 0.058),
}


@njit(parallel=True, cache=True)
def _step(u, v, du, dv, feed, kill, periodic, u_out, v_out):
    nx, ny, nz = u.shape
    for i in prange(nx):
        im = i - 1
        ip = i + 1
        if periodic:
            im %= nx
            ip %= nx
        else:
            im = max(im, 0)
            ip = min(ip, nx - 1)
        for j in range(ny):
            jm = j - 1
            jp = j + 1
            if periodic:
                jm %= ny
                jp %= ny
            else:
                jm = max(jm, 0)
                jp = min(jp, ny - 1)
            for k in range(nz):
                km = k - 1
                kp = k + 1
                if periodic:
                    km %= nz
                    kp %= nz
                else:
                    km = max(km, 0)
                    kp = min(kp, nz - 1)
                uc = u[i, j, k]
                vc = v[i, j, k]
                lu = (
                    u[im, j, k]
                    + u[ip, j, k]
                    + u[i, jm, k]
                    + u[i, jp, k]
                    + u[i, j, km]
                    + u[i, j, kp]
                    - 6.0 * uc
                )
                lv = (
                    v[im, j, k]
                    + v[ip, j, k]
                    + v[i, jm, k]
                    + v[i, jp, k]
                    + v[i, j, km]
                    + v[i, j, kp]
                    - 6.0 * vc
                )
                uvv = uc * vc * vc
                u_out[i, j, k] = uc + du * lu - uvv + feed * (1.0 - uc)
                v_out[i, j, k] = vc + dv * lv + uvv - (feed + kill) * vc


@njit(parallel=True, cache=True)
def _step_tiling(u, v, indptr, indices, weights, inv_volume, du, dv, feed, kill, dt, u_out, v_out):
    for i in prange(u.shape[0]):
        uc = u[i]
        vc = v[i]
        su = 0.0
        sv = 0.0
        w = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            c = weights[p]
            su += c * u[indices[p]]
            sv += c * v[indices[p]]
            w += c
        lu = (su - w * uc) * inv_volume[i]
        lv = (sv - w * vc) * inv_volume[i]
        uvv = uc * vc * vc
        u_out[i] = uc + dt * du * lu - dt * uvv + dt * feed * (1.0 - uc)
        v_out[i] = vc + dt * dv * lv + dt * uvv - dt * (feed + kill) * vc


def _seeds(noise: np.ndarray, seed_fraction: float, rng: np.random.Generator):
    """Initial fields: ``u = 1/2, v = 1/4`` in the blobs of the highest values of the
    smoothed ``noise``, ``u = 1, v = 0`` elsewhere, plus 1 % noise."""
    seeds = noise > np.quantile(noise, 1.0 - seed_fraction)
    u = np.where(seeds, 0.5, 1.0) + 0.01 * rng.standard_normal(noise.shape)
    v = np.where(seeds, 0.25, 0.0) + 0.01 * rng.standard_normal(noise.shape)
    return u, v


def _initial(initial, shape) -> tuple[np.ndarray, np.ndarray]:
    u, v = (np.array(a, dtype=float) for a in initial)
    if u.shape != shape or v.shape != shape:
        raise ValueError(f"initial: two fields of shape {shape}")
    return u, v


def gray_scott(
    shape,
    rng: np.random.Generator,
    *,
    feed: float = 0.037,
    kill: float = 0.060,
    du: float = 0.16,
    dv: float = 0.08,
    steps: int = 3000,
    seed_fraction: float = 0.1,
    seed_size: float = 2.0,
    periodic: bool = True,
    initial=None,
    callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fields ``(u, v)`` after ``steps`` explicit steps from a perturbed uniform state.

    The initial state is ``u = 1, v = 0`` except in random blobs (a fraction
    ``seed_fraction`` of the cells: the highest values of a noise smoothed over
    ``seed_size`` cells) where ``u = 1/2, v = 1/4``, plus 1 % noise; isolated seed cells
    would vanish by diffusion. ``initial``: the fields ``(u, v)`` to start from instead.
    Stability of the explicit scheme requires ``D <= 1/6`` (3D) or ``1/4`` (2D).
    """
    shape = tuple(int(n) for n in shape)
    limit = 1.0 / (2 * sum(1 for n in shape if n > 1))
    if not (0 < du <= limit and 0 < dv <= limit):
        raise ValueError(f"diffusion coefficients must lie in (0, {limit:g}] for stability")
    if not 0.0 < seed_fraction < 1.0:
        raise ValueError("seed_fraction: must lie in (0, 1)")
    if initial is not None:
        u, v = _initial(initial, shape)
    else:
        sigma = [float(seed_size) if n > 1 else 0.0 for n in shape]
        noise = ndimage.gaussian_filter(
            rng.standard_normal(shape), sigma, mode="wrap" if periodic else "nearest"
        )
        u, v = _seeds(noise, seed_fraction, rng)
    u2 = np.empty_like(u)
    v2 = np.empty_like(v)
    for s in range(int(steps)):
        _step(u, v, float(du), float(dv), float(feed), float(kill), bool(periodic), u2, v2)
        u, u2 = u2, u
        v, v2 = v2, v
        if callback is not None:
            callback(s + 1, u, v)
    return u, v


def _operator(tiling):
    """CSR weights, the inverse cell measures (in cell sizes) and the largest rate
    ``max_i sum_j w_ij / V_i`` of the finite-volume Laplacian of a tiling."""
    indptr, indices, w = tiling.diffusion_weights()
    k = 2 if tiling.dims == 2 else 3
    inv_volume = tiling.cell_size**k / np.asarray(tiling.volumes, dtype=float)
    rows = np.repeat(np.arange(len(inv_volume)), np.diff(indptr))
    total = np.bincount(rows, weights=w, minlength=len(inv_volume))
    return indptr, indices, w, inv_volume, float(np.max(total * inv_volume, initial=0.0))


def _substeps(rate: float, du: float, dv: float) -> int:
    return max(1, math.ceil(max(float(du), float(dv)) * rate - 1e-9))


def stable_substeps(tiling, du: float, dv: float) -> int:
    """Number of equal substeps of the unit time step that keep the explicit scheme
    stable on a tiling: ``D dt max_i(sum_j w_ij / V_i) <= 1`` (the eigenvalues of the
    Laplacian lie within twice that rate, Gershgorin); 1 on the square tiling for
    ``D <= 1/(2d)``, as on the grid."""
    return _substeps(_operator(tiling)[4], du, dv)


def gray_scott_tiling(
    tiling,
    rng: np.random.Generator,
    *,
    feed: float = 0.037,
    kill: float = 0.060,
    du: float = 0.16,
    dv: float = 0.08,
    steps: int = 3000,
    seed_fraction: float = 0.1,
    seed_size: float = 2.0,
    initial=None,
    callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fields ``(u, v)`` (one value per cell of ``tiling``) after ``steps`` time steps of
    the Gray-Scott system with the finite-volume Laplacian of the tiling.

    Diffusion coefficients are in cell sizes squared per step, as on the grid; each unit
    step is split into :func:`stable_substeps` explicit Euler substeps (one on the square
    tiling, where the scheme is the grid's, to round-off). The initial blobs are the
    highest values of a white noise smoothed by the heat equation over ``seed_size``
    cell sizes (a heat kernel of standard deviation ``seed_size``); ``initial``: the
    fields ``(u, v)`` to start from instead. ``callback(step, u, v)`` after each step.
    """
    if not (du > 0 and dv > 0):
        raise ValueError("diffusion coefficients must be positive")
    if not 0.0 < seed_fraction < 1.0:
        raise ValueError("seed_fraction: must lie in (0, 1)")
    indptr, indices, w, inv_volume, rate = _operator(tiling)
    n = tiling.n_cells
    if initial is not None:
        u, v = _initial(initial, (n,))
    else:
        noise = rng.standard_normal(n)
        t = 0.5 * float(seed_size) ** 2  # heat kernel of variance 2 t = seed_size^2
        if t > 0 and rate > 0:
            rows = np.repeat(np.arange(n), np.diff(indptr))
            lap = sparse.csr_matrix((w * inv_volume[rows], indices, indptr), shape=(n, n))
            lap = lap - sparse.diags(np.asarray(lap.sum(axis=1)).ravel())
            m = math.ceil(t * rate)
            for _ in range(m):
                noise = noise + (t / m) * (lap @ noise)
        u, v = _seeds(noise, seed_fraction, rng)
    sub = _substeps(rate, du, dv)
    dt = 1.0 / sub
    u2 = np.empty_like(u)
    v2 = np.empty_like(v)
    args = (indptr, indices, np.asarray(w, dtype=float), inv_volume, float(du), float(dv))
    for s in range(int(steps)):
        for _ in range(sub):
            _step_tiling(u, v, *args, float(feed), float(kill), dt, u2, v2)
            u, u2 = u2, u
            v, v2 = v2, v
        if callback is not None:
            callback(s + 1, u, v)
    return u, v
