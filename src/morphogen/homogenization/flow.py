"""Viscous flow through the pores of a voxel image: the permeability of Darcy's law.

A fluid of viscosity ``eta`` creeps through the pores of a rigid solid under a
macroscopic pressure gradient ``G``: the Stokes problem ``eta lap(v) - grad p = G``,
``div v = 0`` in the pores, ``v = 0`` on their walls, the pressure ``p`` and the velocity
periodic. Its mean velocity over the whole cell (the filtration velocity) obeys Darcy's
law,

    <v> = -(k / eta) G,

with the permeability tensor ``k`` (a squared length: that of the pores), symmetric and
positive, independent of the fluid.

**Method.** The lattice Boltzmann method with two relaxation times (Ginzburg 2005) on the
voxels themselves (lattice D3Q19): populations ``f_i`` move along the 19 velocities
``c_i`` and relax towards the linear equilibrium of a Stokes flow, ``w_i (rho + 3 c_i . j)``
— their symmetric parts ``(f_i + f_-i) / 2`` at the rate ``1 / tau+``, which sets the
viscosity ``nu = (tau+ - 1/2) / 3``, their antisymmetric parts at the rate ``1 / tau-``
— and bounce back from the solid voxels (no slip). The product ``(tau+ - 1/2)(tau- -
1/2)`` is kept at 3/16, for which the wall of a channel along the lattice lies exactly
midway between the last fluid node and the first solid one, and the steady solution does
not depend on the viscosity: the permeability is a property of the voxels alone. The
flow is driven by a uniform body force (the pressure gradient), with the half-force
correction of the momentum, and marched to its steady state; its mean velocity is the
mass flux of the populations that move between the nodes of the pores, conserved exactly
(the same through every section, zero where the pores are closed), which takes a number of
steps of the order of ``w^2 / nu`` for pores ``w`` voxels wide (hence a large ``tau+``,
4 by default); the changes of the mean velocity decay geometrically, which gives the
error left and the limit itself (Aitken's extrapolation), accepted when it holds at two
checks in a row.

A channel of ``H`` voxels between two plane walls has the exact parabolic profile at its
nodes and the permeability ``H^2 / 12 + 1 / 24`` (per unit of its width, in voxels: the
mean of the parabola sampled at the nodes). Curved walls are drawn as stairs: a few
percent of error when the pores are ten voxels wide, less when they are wider.

The voxels must be cubic. The pores are the void phases of the materials (a fluid has no
`Material`); closed pores hold a fluid at rest.

References: H. Darcy (1856); I. Ginzburg, Adv. Water Resour. 28 (2005) 1171; I. Ginzburg,
F. Verhaeghe, D. d'Humières, Commun. Comput. Phys. 3 (2008) 427; L. Talon et al., Water
Resour. Res. 48 (2012) W04526; A. A. Zick, G. M. Homsy, J. Fluid Mech. 115 (1982) 13; A.
S. Sangani, A. Acrivos, Int. J. Multiphase Flow 8 (1982) 193.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from morphogen.homogenization.core import VOIGT_NAMES
from morphogen.homogenization.elements import VOIGT
from morphogen.image import VoxelImage
from morphogen.stopping import checkpoint

log = logging.getLogger("morphogen")

__all__ = [
    "ENTRIES",
    "MAGIC",
    "StokesFlow",
    "permeability_tensor",
    "pore_mask",
    "stokes_flow",
]

#: The scalar results of :func:`permeability_tensor` (columns of the tables of results and
#: quantities of the statistics of RVE studies)
ENTRIES = ("mean", "porosity", *(f"k{name}" for name in VOIGT_NAMES))
#: The product ``(tau+ - 1/2)(tau- - 1/2)`` of the two relaxation times: walls midway
#: between the nodes for the flow of a channel along the lattice
MAGIC = 3.0 / 16.0

# D3Q19: the rest velocity, the six of the axes, the twelve of the diagonals of the faces
_C = np.array(
    [[0, 0, 0],
     [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
     [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
     [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
     [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]],
    dtype=np.int64,
)  # fmt: skip
_W = np.array([1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12)
#: The velocity opposite to each one (they come in pairs after the rest velocity)
_OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int64)


@njit(cache=True, parallel=True)
def _step(old, new, fluid, force, om_plus, om_minus, c, w, opp):  # pragma: no cover
    """One step of the populations (x, y, z, 19): those of ``old`` (after their
    collision) pulled along their velocities, or bounced back from the solid, then
    collided into ``new``."""
    nx, ny, nz = fluid.shape
    gx, gy, gz = force[0], force[1], force[2]
    source = 1.0 - 0.5 * om_minus
    for x in prange(nx):
        p = np.empty(19)
        # the neighbours x - c for c = -1, 0, 1 (periodic)
        xa = np.array([(x + 1) % nx, x, (x - 1) % nx])
        ya = np.empty(3, dtype=np.int64)
        za = np.empty(3, dtype=np.int64)
        for y in range(ny):
            ya[0], ya[1], ya[2] = (y + 1) % ny, y, (y - 1) % ny
            for z in range(nz):
                if not fluid[x, y, z]:
                    continue
                za[0], za[1], za[2] = (z + 1) % nz, z, (z - 1) % nz
                for i in range(19):
                    xs = xa[c[i, 0] + 1]
                    ys = ya[c[i, 1] + 1]
                    zs = za[c[i, 2] + 1]
                    if fluid[xs, ys, zs]:
                        p[i] = old[xs, ys, zs, i]
                    else:
                        p[i] = old[x, y, z, opp[i]]
                rho = 0.0
                jx = 0.5 * gx
                jy = 0.5 * gy
                jz = 0.5 * gz
                for i in range(19):
                    rho += p[i]
                    jx += c[i, 0] * p[i]
                    jy += c[i, 1] * p[i]
                    jz += c[i, 2] * p[i]
                new[x, y, z, 0] = p[0] - om_plus * (p[0] - w[0] * rho)
                for i in range(1, 19, 2):
                    o = i + 1  # the opposite velocity
                    sym = 0.5 * (p[i] + p[o]) - w[i] * rho
                    cj = c[i, 0] * jx + c[i, 1] * jy + c[i, 2] * jz
                    anti = 0.5 * (p[i] - p[o]) - 3.0 * w[i] * cj
                    cg = c[i, 0] * gx + c[i, 1] * gy + c[i, 2] * gz
                    push = source * 3.0 * w[i] * cg
                    new[x, y, z, i] = p[i] - om_plus * sym - om_minus * anti + push
                    new[x, y, z, o] = p[o] - om_plus * sym + om_minus * anti - push


@njit(cache=True, parallel=True)
def _momentum(pops, fluid, force, c, out):  # pragma: no cover
    """The momentum of each fluid node (``out``: x, y, z, 3; zero in the solid) from its
    populations after their collision: the collision has added the whole force to the
    momentum of the populations that arrived, and the momentum of the fluid is theirs
    plus half of it."""
    nx, ny, nz = fluid.shape
    for x in prange(nx):
        for y in range(ny):
            for z in range(nz):
                if not fluid[x, y, z]:
                    out[x, y, z, 0] = out[x, y, z, 1] = out[x, y, z, 2] = 0.0
                    continue
                jx = -0.5 * force[0]
                jy = -0.5 * force[1]
                jz = -0.5 * force[2]
                for i in range(1, 19):
                    v = pops[x, y, z, i]
                    jx += c[i, 0] * v
                    jy += c[i, 1] * v
                    jz += c[i, 2] * v
                out[x, y, z, 0] = jx
                out[x, y, z, 1] = jy
                out[x, y, z, 2] = jz


@njit(cache=True, parallel=True)
def _flux(pops, fluid, c, out):  # pragma: no cover
    """The mass flux of the fluid (``out``: its three components summed over the nodes):
    that of the populations which, after their collision, leave a node for a neighbour in
    the pores. Mass is conserved exactly: at the steady state this flux is the same
    through every section of the cell, and zero where the pores are closed (which the
    momentum of the nodes is not, next to a wall that the force pushes on)."""
    nx, ny, nz = fluid.shape
    sx = np.zeros(nx)
    sy = np.zeros(nx)
    sz = np.zeros(nx)
    for x in prange(nx):
        ax, ay, az = 0.0, 0.0, 0.0
        for y in range(ny):
            for z in range(nz):
                if not fluid[x, y, z]:
                    continue
                for i in range(1, 19):
                    if fluid[(x + c[i, 0]) % nx, (y + c[i, 1]) % ny, (z + c[i, 2]) % nz]:
                        v = pops[x, y, z, i]
                        ax += c[i, 0] * v
                        ay += c[i, 1] * v
                        az += c[i, 2] * v
        sx[x], sy[x], sz[x] = ax, ay, az
    out[0], out[1], out[2] = sx.sum(), sy.sum(), sz.sum()


def pore_mask(image: VoxelImage, materials=None, pores=None) -> np.ndarray:
    """The voxels of the pores of an image (a boolean array): those of the phases
    ``pores`` (a phase or several), or of the void phases of ``materials``."""
    if not isinstance(image, VoxelImage):
        raise TypeError("the flow through the pores is solved on a VoxelImage")
    data = np.asarray(image.data)
    if data.dtype.kind not in "iub":
        raise ValueError("the flow needs an integer phase image")
    if pores is None:
        if materials is None:
            raise ValueError("give the materials (their voids are the pores) or the pores")
        pores = [p for p, m in materials.items() if getattr(m, "void", False)]
        if not pores:
            raise ValueError("flow: no pore (a void phase) in the materials")
    phases = np.atleast_1d(np.asarray(pores)).astype(np.int64)
    return np.isin(data, phases)


@dataclass
class StokesFlow:
    """The steady creeping flow of a periodic voxel image under a unit macroscopic
    pressure gradient along ``-direction`` (a unit vector), in the units of the image:
    ``velocity``, the filtration velocity times the viscosity (``-k . G``: a row of the
    permeability for a direction along an axis); ``porosity``; ``steps`` of the lattice
    Boltzmann method and the estimated relative ``error`` of the mean velocity left;
    ``field``: ``eta v`` at each voxel (x, y, z, 3; zero in the solid) when asked for."""

    direction: np.ndarray
    velocity: np.ndarray
    porosity: float
    steps: int
    error: float
    time: float = 0.0
    field: np.ndarray | None = field(default=None, repr=False)


def stokes_flow(
    image: VoxelImage,
    direction,
    *,
    materials=None,
    pores=None,
    tol: float = 1e-4,
    max_steps: int | None = None,
    relaxation: float = 4.0,
    fields: bool = False,
) -> StokesFlow:
    """The creeping flow through the pores of a periodic image driven along
    ``direction`` (see the module). ``pores``: the phases of the fluid (by default the
    void phases of ``materials``); ``tol``: the relative error of the mean velocity at
    which the march to the steady state stops; ``max_steps``: at most that many steps (by
    default 200 times the largest size of the image in voxels, at least 20000);
    ``relaxation``: the time ``tau+`` (viscosity ``(tau+ - 1/2) / 3`` of the lattice: the
    steady flow does not depend on it, the number of steps does); ``fields``: keep the
    velocity field."""
    fluid = np.ascontiguousarray(pore_mask(image, materials, pores))
    spacing = np.asarray(image.spacing, dtype=float)
    if not np.allclose(spacing, spacing[0], rtol=1e-9):
        raise ValueError(f"flow: cubic voxels (the lattice), got the spacing {tuple(spacing)}")
    if not image.periodic:
        log.info("flow: the image is treated as the cell of a periodic medium")
    if not relaxation > 0.5:
        raise ValueError("relaxation: tau+ > 1/2")
    g = np.asarray(direction, dtype=float).ravel()
    if g.shape != (3,) or not np.linalg.norm(g) > 0:
        raise ValueError("direction: a vector of three components")
    g = g / np.linalg.norm(g)
    h = float(spacing[0])
    porosity = float(fluid.mean())
    t0 = time.perf_counter()
    if porosity == 0.0:
        return StokesFlow(g, np.zeros(3), 0.0, 0, 0.0)
    nu = (relaxation - 0.5) / 3.0
    om_plus = 1.0 / relaxation
    om_minus = 1.0 / (0.5 + MAGIC / (relaxation - 0.5))
    scale = 1e-6  # of the force: the flow is linear, the populations stay near rest
    force = scale * g
    old = np.empty((*fluid.shape, 19))
    old[...] = _W
    new = np.empty_like(old)
    total = np.zeros(3)
    limit = int(max_steps) if max_steps else max(20000, 200 * max(fluid.shape))
    check = 50
    previous = np.zeros(3)  # the mean momentum at the last check, and its change
    change = np.zeros(3)
    limit_value, error, steps = None, float("inf"), 0
    reached = False  # the tolerance at the last check: it must hold twice in a row
    velocity = np.zeros(3)
    while steps < limit:
        checkpoint()
        for _ in range(check // 2):  # the two arrays in turn
            _step(old, new, fluid, force, om_plus, om_minus, _C, _W, _OPP)
            _step(new, old, fluid, force, om_plus, om_minus, _C, _W, _OPP)
        steps += check
        _flux(old, fluid, _C, total)
        mean = total / fluid.size
        delta = mean - previous
        along, d, dc = float(mean @ g), float(delta @ g), float(change @ g)
        velocity = mean
        if steps >= 200 and abs(along) * nu / scale < 1e-9:
            # no path through the pores (a permeability below 1e-9 voxel^2): a fluid at rest
            velocity, error = np.zeros(3), 0.0
            break
        if along > 0.0 and dc != 0.0:
            ratio = d / dc
            if 0.0 <= ratio < 1.0:
                # the changes decay geometrically: what is left of them (Aitken), added
                # to the mean unless the field itself is asked for
                rest = ratio / (1.0 - ratio)
                error = abs(d) * rest / along
                if not fields:
                    velocity = mean + delta * rest
                    value = float(velocity @ g)
                    if limit_value is not None and value != 0.0:
                        error = min(error, abs(value - limit_value) / abs(value))
                    limit_value = value
            else:
                error, limit_value = abs(d) / along, None
            if error <= tol and reached:
                break
            reached = error <= tol
        previous, change = mean, delta
    velocity = velocity * (nu / scale) * h * h
    out = StokesFlow(g, velocity, porosity, steps, float(error), time.perf_counter() - t0)
    if fields:
        momentum = np.empty((*fluid.shape, 3))
        _momentum(old, fluid, force, _C, momentum)
        out.field = momentum * (nu / scale) * h * h
    return out


def permeability_tensor(
    image: VoxelImage, materials=None, *, pores=None, axes=(0, 1, 2), **options
) -> dict:
    """The permeability tensor of a periodic voxel image (see the module): one flow per
    axis of ``axes`` (the others are left out: their rows and columns zero), symmetrized.

    Returns ``"tensor"`` (3 x 3, in the squared unit of the spacing of the image), its
    entries ``k11`` ... ``k12``, ``"mean"`` (the mean of the diagonal over the ``axes``),
    the ``"porosity"``, the ``"steps"`` of the three flows and the largest ``"error"``
    estimated. ``options``: those of :func:`stokes_flow` (``tol``, ``max_steps``,
    ``relaxation``)."""
    axes = tuple(int(a) for a in axes)
    if not axes or any(a not in (0, 1, 2) for a in axes) or len(set(axes)) != len(axes):
        raise ValueError("axes: among 0, 1, 2")
    tensor = np.zeros((3, 3))
    steps, error, porosity = 0, 0.0, 0.0
    for a in axes:
        unit = np.zeros(3)
        unit[a] = 1.0
        flow = stokes_flow(image, unit, materials=materials, pores=pores, **options)
        tensor[:, a] = flow.velocity
        steps += flow.steps
        error = max(error, flow.error)
        porosity = flow.porosity
    keep = np.zeros(3, dtype=bool)
    keep[list(axes)] = True
    tensor[~keep] = 0.0
    tensor = 0.5 * (tensor + tensor.T)
    out = {
        "tensor": tensor,
        "mean": float(np.mean([tensor[a, a] for a in axes])),
        "porosity": porosity,
        "steps": int(steps),
        "error": float(error),
    }
    out.update({f"k{n}": float(tensor[i, j]) for (i, j), n in zip(VOIGT, VOIGT_NAMES, strict=True)})
    return out
