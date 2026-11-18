"""Transient diffusion in voxel images: moisture, oxygen, solvents.

The activity ``u`` of the mobile species (relative humidity, oxygen pressure over its
value outside) is continuous across interfaces; each phase holds the concentration
``c = S u`` (``S`` its solubility, the concentration at equilibrium with the outside)
and lets it through with the permeability ``P = D S`` (``D`` its diffusivity):

    S du/dt = div(P grad u) - k S u

(``k`` an optional first-order consumption). Finite volumes on the voxels: the flux
between two neighbours goes through the harmonic mean of their permeabilities (exact
for layers in series), the exposed faces hold ``u`` (a value or a function of time)
half a voxel from the centres of their voxels, the other faces are impermeable or, in
periodic images, periodic; implicit Euler steps. The analytic solutions of a slab
(Crank 1975, eq. 4.17 and 4.18) check it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from morphogen.analysis.basic import as_array
from morphogen.image import VoxelImage

FACES = ("x-", "x+", "y-", "y+", "z-", "z+")
#: Voxels up to which the steps are solved by sparse LU (one factorization per time step
#: length); preconditioned conjugate gradients beyond.
DIRECT_LIMIT = 4096


@dataclass
class DiffusionResult:
    """Concentrations ``c = S u`` at the requested ``times`` and the uptake: the mean
    concentration over the image at each time, and at equilibrium (``saturation``)."""

    times: np.ndarray
    fields: list[np.ndarray]
    uptake: np.ndarray
    saturation: float

    @property
    def relative_uptake(self) -> np.ndarray:
        """``M(t) / M_inf``."""
        return (
            self.uptake / self.saturation if self.saturation else np.full_like(self.uptake, np.nan)
        )


def _per_phase(labels: np.ndarray, values, name: str, default=None) -> np.ndarray:
    phases = np.unique(labels)
    if values is None:
        if default is None:
            raise ValueError(f"{name}: one value per phase")
        values = dict.fromkeys(phases.tolist(), default)
    elif np.isscalar(values):
        values = dict.fromkeys(phases.tolist(), float(values))
    missing = [int(p) for p in phases if int(p) not in values]
    if missing:
        raise ValueError(f"{name}: no value for the phases {missing}")
    table = np.zeros(int(phases.max()) + 1)
    for p, v in values.items():
        if int(p) <= phases.max():
            table[int(p)] = float(v)
    if np.any(table[phases] < 0):
        raise ValueError(f"{name}: non-negative values")
    return table[labels]


def _harmonic(a, b):
    s = a + b
    return np.divide(2.0 * a * b, s, out=np.zeros_like(s), where=s > 0)


def _operator(perm, spacing, periodic_axes, exposed):
    """Conductance matrix K (SPD) and the conductances g of the exposed faces."""
    shape = perm.shape
    n = perm.size
    index = np.arange(n).reshape(shape)
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    boundary = np.zeros(n)
    for axis in range(3):
        h = spacing[axis]
        area = np.prod([spacing[a] for a in range(3) if a != axis])
        if shape[axis] > 1 or axis in periodic_axes:
            a = perm
            b = np.roll(perm, -1, axis=axis)
            g = _harmonic(a, b) * area / h
            i = index
            j = np.roll(index, -1, axis=axis)
            if axis not in periodic_axes:  # no link across the last face
                sl = [slice(None)] * 3
                sl[axis] = slice(0, shape[axis] - 1)
                g, i, j = g[tuple(sl)], i[tuple(sl)], j[tuple(sl)]
            if shape[axis] == 1 and axis in periodic_axes:
                g = np.zeros_like(g)  # a voxel and itself
            g, i, j = g.ravel(), i.ravel(), j.ravel()
            rows += [i, j]
            cols += [j, i]
            vals += [-g, -g]
            np.add.at(diag, i, g)
            np.add.at(diag, j, g)
        for side, face in ((0, f"{'xyz'[axis]}-"), (-1, f"{'xyz'[axis]}+")):
            if face in exposed:
                sl = [slice(None)] * 3
                sl[axis] = side
                cells = index[tuple(sl)].ravel()
                np.add.at(boundary, cells, perm.ravel()[cells] * area / (0.5 * h))
    if vals:
        data, ij = np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))
    else:
        data, ij = np.zeros(0), (np.zeros(0, int), np.zeros(0, int))
    k = sparse.csr_matrix((data, ij), shape=(n, n))
    k = k + sparse.diags(diag + boundary)
    return k.tocsc(), boundary


def transient_diffusion(
    image,
    diffusivity,
    times,
    *,
    solubility=None,
    exposed=("x-", "x+"),
    boundary: float | Callable[[float], float] = 1.0,
    initial: float = 0.0,
    consumption=None,
    steps: int = 40,
    periodic=None,
) -> DiffusionResult:
    """Diffusion of a species into the phases of ``image`` from its ``exposed`` faces,
    held at the activity ``boundary`` (a number or a function of time), from a uniform
    initial activity ``initial``.

    ``diffusivity``, ``solubility`` (default 1) and ``consumption`` (first-order rate,
    default 0): one number, or a mapping phase -> value. ``times``: the output times,
    each interval split into ``steps`` implicit Euler steps (the first one into steps
    growing geometrically, for the sqrt(t) start). Axes without exposed faces are
    periodic in periodic images, impermeable otherwise.
    """
    labels = np.asarray(as_array(image)).astype(np.int64)
    spacing = np.asarray(image.spacing if isinstance(image, VoxelImage) else (1.0, 1.0, 1.0), float)
    per = (
        bool(image.periodic)
        if periodic is None and isinstance(image, VoxelImage)
        else bool(periodic)
    )
    exposed = tuple(exposed)
    for face in exposed:
        if face not in FACES:
            raise ValueError(f"exposed: faces among {', '.join(FACES)}")
    d = _per_phase(labels, diffusivity, "diffusivity")
    s = _per_phase(labels, solubility, "solubility", default=1.0)
    k = _per_phase(labels, consumption, "consumption", default=0.0)
    axes_exposed = {FACES.index(f) // 2 for f in exposed}
    periodic_axes = {a for a in range(3) if per and a not in axes_exposed}
    voxel = float(np.prod(spacing))
    cap = (s * voxel).ravel()
    big_k, g = _operator(d * s, spacing, periodic_axes, exposed)
    big_k = big_k + sparse.diags((k * s).ravel() * voxel)
    value = boundary if callable(boundary) else (lambda t, b=float(boundary): b)
    times = np.asarray(times, dtype=float)
    if np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError("times: increasing and non-negative")
    u = np.full(labels.size, float(initial))
    t = 0.0
    fields, uptake = [], []
    cache: dict[float, object] = {}
    for t_out in times:
        span = t_out - t
        if span > 0:
            if t == 0.0:  # geometric steps from about span / 10^4
                r = 1.25
                first = span * (r - 1.0) / (r**steps - 1.0)
                dts = first * r ** np.arange(steps)
            else:
                dts = np.full(steps, span / steps)
            for dt in dts:
                key = float(dt)
                if key not in cache:
                    a = (sparse.diags(cap / dt) + big_k).tocsc()
                    cache.clear()
                    cache[key] = splinalg.splu(a) if labels.size <= DIRECT_LIMIT else a
                solver = cache[key]
                t += dt
                rhs = cap / dt * u + g * value(t)
                if isinstance(solver, splinalg.SuperLU):
                    u = solver.solve(rhs)
                else:
                    diag = solver.diagonal()
                    pre = splinalg.LinearOperator(solver.shape, matvec=lambda x, dg=diag: x / dg)
                    u, info = splinalg.cg(solver, rhs, x0=u, rtol=1e-11, maxiter=5000, M=pre)
                    if info:
                        raise RuntimeError(
                            "transient_diffusion: conjugate gradients did not converge"
                        )
            t = float(t_out)
        c = (s.ravel() * u).reshape(labels.shape)
        fields.append(c)
        uptake.append(float(c.mean()))
    saturation = float(s.mean()) * float(value(times[-1]))
    return DiffusionResult(times, fields, np.asarray(uptake), saturation)


def slab_uptake(t, thickness: float, diffusivity: float, terms: int = 400) -> np.ndarray:
    """``M(t) / M_inf`` of a slab exposed on both faces (Crank 1975, eq. 4.18):
    ``1 - sum 8 / ((2n + 1)^2 pi^2) exp(-D (2n + 1)^2 pi^2 t / L^2)``."""
    t = np.asarray(t, dtype=float)
    m = 2 * np.arange(terms) + 1
    e = np.exp(-diffusivity * (m * np.pi / thickness) ** 2 * t[..., None])
    return 1.0 - np.sum(8.0 / (m * np.pi) ** 2 * e, axis=-1)


def slab_profile(z, t: float, thickness: float, diffusivity: float, terms: int = 400) -> np.ndarray:
    """``c / c_s`` at the distance ``z`` from the middle of the slab (Crank 1975, eq. 4.17)."""
    z = np.asarray(z, dtype=float)
    n = np.arange(terms)
    m = 2 * n + 1
    e = np.exp(-diffusivity * (m * np.pi / thickness) ** 2 * t)
    series = np.sum((-1.0) ** n / m * e * np.cos(m * np.pi * z[..., None] / thickness), axis=-1)
    return 1.0 - 4.0 / np.pi * series
