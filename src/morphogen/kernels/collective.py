"""Stealthy hyperuniform point patterns by the collective-coordinate method (Uche,
Stillinger & Torquato 2004; Batten, Stillinger & Torquato 2008).

``n`` points in a periodic box are moved, from uniform random positions, to a ground
state of the energy ``Φ = Σ_{k ∈ Q} |ρ(k)|^2``, ``ρ(k) = Σ_j e^{i k·x_j}`` the collective
coordinates on the wave vectors ``k = 2π m / L`` of the box (``m`` integer) with ``0 < |k|
<= K``, one of each pair ``±k`` (``ρ(-k)`` is the conjugate of ``ρ(k)``). At ``Φ = 0`` the
structure factor ``S(k) = |ρ(k)|^2 / n`` vanishes on the ball of radius ``K``: the pattern
is stealthy (it scatters no wave of those wave vectors) and hyperuniform. The ratio of
the constrained wave vectors to the degrees of freedom, ``χ = M / (3 (n - 1))``, sets the
order: ground states are disordered below ``χ = 1/2`` and crystallize beyond. ``Φ`` and
its gradient ``∂Φ/∂x_j = 2 Σ_k k (Im ρ(k) cos(k·x_j) - Re ρ(k) sin(k·x_j))`` are
minimized by L-BFGS.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


def stealthy_wave_vectors(n: int, box, chi: float) -> np.ndarray:
    """The wave vectors ``k`` (rows) of the half space whose number is the largest up to
    ``χ · 3 (n - 1)`` over the complete shells of ``|k|``."""
    box = np.asarray(box, dtype=float)
    target = chi * 3.0 * (n - 1)
    k0 = 2.0 * math.pi / box
    # enough integer vectors: a ball holding twice the target in its half
    radius = (3.0 * 2.0 * target / (2.0 * math.pi) * np.prod(k0)) ** (1.0 / 3.0) + k0.max()
    span = np.ceil(radius / k0).astype(int)
    m = np.stack(np.meshgrid(*[np.arange(-s, s + 1) for s in span], indexing="ij"), -1)
    m = m.reshape(-1, 3)
    half = (
        (m[:, 0] > 0)
        | ((m[:, 0] == 0) & (m[:, 1] > 0))
        | ((m[:, 0] == 0) & (m[:, 1] == 0) & (m[:, 2] > 0))
    )
    k = m[half] * k0
    size = np.linalg.norm(k, axis=1)
    order = np.argsort(size, kind="stable")
    k, size = k[order], size[order]
    shells = np.flatnonzero(np.diff(np.round(size, 9)) > 0) + 1  # ends of complete shells
    ends = np.concatenate([shells, [len(size)]])
    keep = ends[ends <= target]
    if not len(keep):
        raise ValueError(f"stealthy: no complete shell of wave vectors for chi = {chi} and {n} "
                         "points (raise chi or the number of points)")  # fmt: skip
    return k[: int(keep[-1])]


def _energy(x, k):
    """``Φ`` and its gradient for the positions ``x`` (flattened) and wave vectors ``k``."""
    phase = x.reshape(-1, 3) @ k.T
    c, s = np.cos(phase), np.sin(phase)
    re, im = c.sum(axis=0), s.sum(axis=0)
    grad = 2.0 * ((c * im - s * re) @ k)
    return float(np.sum(re * re + im * im)), grad.ravel()


def stealthy_points(n: int, box, chi: float, rng, *, tries: int = 3, tolerance: float = 1e-10):
    """``n`` points of a stealthy hyperuniform ground state in the periodic box of edges
    ``box`` (see the module), their ``S(k)`` below ``tolerance`` on the constrained wave
    vectors; returns the points (in ``[0, box)``), the wave vectors and the ``χ`` reached.
    Raises ``RuntimeError`` when ``tries`` runs from random positions miss it."""
    if not 0.0 < chi < 0.5:
        raise ValueError("stealthy: chi in (0, 0.5), the disordered ground states")
    box = np.asarray(box, dtype=float)
    k = stealthy_wave_vectors(n, box, chi)
    worst = math.inf
    for _ in range(tries):
        x0 = rng.random((n, 3)) * box
        options = {"maxiter": 20_000, "maxcor": 30, "ftol": 0.0, "gtol": 0.0}
        res = minimize(_energy, x0.ravel(), args=(k,), jac=True, method="L-BFGS-B",
                       options=options)  # fmt: skip
        energy, _ = _energy(res.x, k)
        worst = min(worst, energy / n)
        if energy / n <= tolerance * len(k):
            pts = np.mod(res.x.reshape(n, 3), box)
            pts[pts >= box] = 0.0
            phase = pts @ k.T
            s = (np.cos(phase).sum(0) ** 2 + np.sin(phase).sum(0) ** 2) / n
            if s.max() <= tolerance:
                return pts, k, len(k) / (3.0 * (n - 1))
    raise RuntimeError(f"stealthy: the ground state was not reached (S(k) up to {worst:.2e}); "
                       "lower chi, or raise the number of points")  # fmt: skip
