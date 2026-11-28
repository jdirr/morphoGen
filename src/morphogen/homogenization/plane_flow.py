"""Creeping flow through the pores of a structure of the plane: the permeability of Darcy's
law by the finite elements of the plane.

A fluid of viscosity ``eta`` creeps through the pores of a rigid solid that does not vary
along ``z`` — a bed of parallel fibres, the channels between rods, the section of any
structure — under a macroscopic pressure gradient: the Stokes problem ``eta lap(v) - grad p
+ f = 0``, ``div v = 0`` in the pores, ``v = 0`` on their walls, the velocity and the
pressure periodic, ``f`` the uniform force of the gradient. The mean velocity over the whole
cell, solid included, obeys Darcy's law,

    <v> = (k / eta) f,

with the permeability tensor ``k`` (a squared length), symmetric and positive.

**Method.** Across the fibres the flow is plane: the elements of Taylor & Hood, the velocity
quadratic on the six-node triangles of the pores (curved along their walls on the sections
of structures that give their implicit functions, :func:`~morphogen.mesh.curved.
curved_triangles`; isoparametric, as the elements of :mod:`morphogen.homogenization.plane`),
the pressure linear on their corners and continuous — stable (they satisfy the inf-sup
condition), the velocity of the third order in its values. Along the fibres the flow is the
anti-plane flow ``eta lap(w) + f3 = 0``, ``w = 0`` on the walls, on the same quadratic
elements: the flows across and along a structure that does not vary along ``z`` are
uncoupled (``k13 = k23 = 0``). The systems are factorized (the saddle point of the plane
flow by an LU factorization, both of its loads at once); the pressure is fixed at a node of
each connected part of the pores, so that the closed ones hold a fluid at rest.

The pores are the void phases of the materials (``Material(void=True)``), or the phases
``pores``, meshed with the solid: their walls are the edges between them and the other
phases, or the edges of a mesh of the pores alone inside the box. The mesh is periodic,
its opposite sides matching. A channel of height ``H`` between two plane walls, across a
periodic cell of height ``L``: the parabola of Poiseuille in the quadratic elements exactly,
``k = H^3 / (12 L)`` along the channel and along ``z``, zero across.

References: C. Taylor, P. Hood, Comput. Fluids 1 (1973) 73; F. Brezzi, M. Fortin, Mixed
and Hybrid Finite Element Methods (1991); A. S. Sangani, A. Acrivos, Int. J. Multiphase
Flow 8 (1982) 193; J. E. Drummond, M. I. Tahir, Int. J. Multiphase Flow 10 (1984) 515.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu

from morphogen.homogenization.core import VOIGT_NAMES
from morphogen.homogenization.elements import VOIGT
from morphogen.homogenization.plane import GAUSS, _gradients
from morphogen.mesh.triangles import TRI6_EDGES, TriMesh

log = logging.getLogger("morphogen")

__all__ = ["plane_permeability", "pore_phases"]


def pore_phases(mesh: TriMesh, materials=None, pores=None) -> list[int]:
    """The phases of the pores of a mesh of the plane: ``pores``, or the void phases of
    ``materials`` among those of the mesh."""
    present = [int(p) for p in np.unique(mesh.phase)]
    if pores is not None:
        chosen = [int(p) for p in np.atleast_1d(pores)]
    elif materials is not None:
        chosen = [p for p in present if p in materials and materials[p].void]
    else:
        raise ValueError("the pores: the phases 'pores', or materials whose voids they are")
    return [p for p in chosen if p in present]


def _representatives(mesh: TriMesh) -> np.ndarray:
    """The node that stands for each node and its images across the sides of the box."""
    rep = np.arange(mesh.n_nodes)
    pairs = [mesh.periodic_pairs(d) for d in (0, 1)]
    for _ in range(3):  # the corners of the box have three images
        for pair in pairs:
            if len(pair):
                rep[pair[:, 1]] = rep[pair[:, 0]]
    while True:
        further = rep[rep]
        if np.array_equal(further, rep):
            return rep
        rep = further


def _walls(mesh: TriMesh) -> np.ndarray:
    """The nodes of the walls of a mesh of the pores: those of the edges of a single
    element that do not lie on a side of the box (corners and middle nodes)."""
    c = mesh.elements
    ends = np.concatenate([c[:, [a, b]] for a, b in TRI6_EDGES])
    middle = np.concatenate([c[:, 3 + k] for k in range(3)])
    key = np.sort(ends, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    single = counts[inverse.ravel()] == 1
    x = mesh.nodes[ends]
    tol = mesh.tolerance
    on_box = np.zeros(len(ends), dtype=bool)
    for d in range(2):
        for target in (mesh.lower[d], mesh.upper[d]):
            on_box |= np.all(np.abs(x[:, :, d] - target) <= tol, axis=1)
    wall = single & ~on_box
    out = np.zeros(mesh.n_nodes, dtype=bool)
    out[ends[wall].ravel()] = True
    out[middle[wall]] = True
    return out


def _element_arrays(mesh: TriMesh):
    """The arrays of the elements at their integration points: the stiffness of the
    Laplacian ``(m, 6, 6)``, the divergence against the linear pressures ``(m, 3, 6, 2)`` and
    the integrals of the quadratic functions ``(m, 6)``."""
    grads, weights = _gradients(mesh)  # (m, q, 6, 2), (m, q)
    points, _ = GAUSS[2]
    shapes = np.empty((len(points), 6))
    for q, lam in enumerate(points):
        shapes[q, :3] = lam * (2.0 * lam - 1.0)
        for k, (a, b) in enumerate(TRI6_EDGES):
            shapes[q, 3 + k] = 4.0 * lam[a] * lam[b]
    laplacian = np.einsum("mq,mqai,mqbi->mab", weights, grads, grads, optimize=True)
    divergence = np.einsum("mq,qi,mqac->miac", weights, points, grads, optimize=True)
    integrals = np.einsum("mq,qa->ma", weights, shapes, optimize=True)
    return laplacian, divergence, integrals


def plane_permeability(
    mesh: TriMesh, materials=None, *, pores=None, viscosity: float = 1.0, fields: bool = False
) -> dict:
    """The permeability tensor of the pores of a periodic mesh of the plane (see the
    module): the flows across the plane for the forces along ``x`` and ``y``, that along
    ``z`` for the force along ``z``, of the fluid of viscosity ``viscosity`` (the tensor
    does not depend on it).

    Returns ``"tensor"`` (3 x 3, in the squared unit of the coordinates; symmetrized), its
    entries ``k11`` ... ``k12``, ``"mean"`` (the mean of its diagonal), the
    ``"porosity"``, the ``"unknowns"`` of the plane flow, the connected ``"parts"`` of the
    pores and the ``"time_s"``; with ``fields``, the mesh of the pores (``"pores"``) and the
    velocities of its nodes under the unit forces along ``x``, ``y`` and ``z``
    (``"velocity"``, ``(3, nodes, 3)``)."""
    if not isinstance(mesh, TriMesh):
        raise TypeError("the flow of the plane needs a TriMesh (morphogen.mesh.triangles)")
    t0 = time.perf_counter()
    phases = pore_phases(mesh, materials, pores)
    box = mesh.box_area()
    empty = {"tensor": np.zeros((3, 3)), "mean": 0.0, "porosity": 0.0, "unknowns": 0}
    if not phases:
        if float(np.abs(mesh.areas()).sum()) < (1.0 - 1e-9) * box:
            raise ValueError("the flow of the plane: the pores are left out of the mesh; "
                             "mesh them with the solid (phases: all)")  # fmt: skip
        log.warning("no pores in the mesh of the plane: its permeability is zero")
        empty.update({f"k{n}": 0.0 for n in VOIGT_NAMES})
        return empty
    if not mesh.periodic or not mesh.matching_sides():
        raise ValueError("the flow through the pores of a mesh of the plane: periodic "
                         "conditions, a mesh with matching sides (periodic=True)")  # fmt: skip
    fluid = mesh.subset(phases)
    if fluid.order == 1:
        fluid = fluid.quadratic()
    n = fluid.n_nodes
    rep = _representatives(fluid)
    wall = np.zeros(n, dtype=bool)
    np.logical_or.at(wall, rep, _walls(fluid))  # a node of a wall: its images too
    wall = wall[rep]
    if not wall.any():
        raise ValueError("pores without walls (no solid in the mesh): their flow is not bounded")
    # the unknowns: the velocities of the nodes off the walls, by their representatives
    own = rep == np.arange(n)
    free = np.flatnonzero(own & ~wall)
    number = np.full(n, -1)
    number[free] = np.arange(len(free))
    velocity_of = number[rep]  # -1 on the walls
    corners = np.unique(fluid.elements[:, :3])
    pressure_nodes = np.unique(rep[corners])
    pressure_of = np.full(n, -1)
    pressure_of[pressure_nodes] = np.arange(len(pressure_nodes))
    pressure_of = pressure_of[rep]
    # a pressure fixed in each connected part of the pores
    tri = pressure_of[fluid.elements[:, :3]]
    links = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    graph = sp.coo_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])),
                          shape=(len(pressure_nodes),) * 2)  # fmt: skip
    parts, label = csgraph.connected_components(graph, directed=False)
    fixed = np.zeros(len(pressure_nodes), dtype=bool)
    fixed[np.unique(label, return_index=True)[1]] = True
    kept = np.flatnonzero(~fixed)
    renumber = np.full(len(pressure_nodes), -1)
    renumber[kept] = np.arange(len(kept))
    pressure_of = np.where(pressure_of >= 0, renumber[np.maximum(pressure_of, 0)], -1)
    nv, npr = len(free), len(kept)

    laplacian, divergence, integrals = _element_arrays(fluid)
    eta = float(viscosity)
    vel = velocity_of[fluid.elements]  # (m, 6)
    pre = pressure_of[fluid.elements[:, :3]]  # (m, 3)
    # the Laplacian of one component, on the free nodes
    ok = (vel[:, :, None] >= 0) & (vel[:, None, :] >= 0)
    rows = np.broadcast_to(vel[:, :, None], laplacian.shape)[ok]
    cols = np.broadcast_to(vel[:, None, :], laplacian.shape)[ok]
    lap = sp.coo_matrix((laplacian[ok], (rows, cols)), shape=(nv, nv)).tocsr()
    # the divergence: rows the pressures, columns the components of the velocities
    okb = (pre[:, :, None, None] >= 0) & (vel[:, None, :, None] >= 0)
    okb = np.broadcast_to(okb, divergence.shape)
    prow = np.broadcast_to(pre[:, :, None, None], divergence.shape)[okb]
    vcol = (2 * np.broadcast_to(vel[:, None, :, None], divergence.shape)
            + np.arange(2)[None, None, None, :])[okb]  # fmt: skip
    div = sp.coo_matrix((divergence[okb], (prow, vcol)), shape=(npr, 2 * nv)).tocsr()
    # the integral of each function of the free nodes over the pores: the load of a unit
    # force, and the mean velocity times the area of the box
    load = np.zeros(nv)
    okl = vel >= 0
    np.add.at(load, vel[okl], integrals[okl])
    viscous = sp.kron(eta * lap, sp.identity(2), format="csr")
    system = sp.bmat([[viscous, -div.T], [-div, None]], format="csc")
    tensor = np.zeros((3, 3))
    velocities = np.zeros((3, n, 3))
    if nv and system.shape[0]:
        rhs = np.zeros((system.shape[0], 2))
        rhs[0 : 2 * nv : 2, 0] = load
        rhs[1 : 2 * nv : 2, 1] = load
        solution = splu(system).solve(rhs)
        on = velocity_of >= 0
        for a in (0, 1):
            v = solution[: 2 * nv, a].reshape(nv, 2)
            tensor[:2, a] = eta * (load @ v) / box
            velocities[a, on, :2] = v[velocity_of[on]]
        axial = splu((eta * lap).tocsc()).solve(load)
        tensor[2, 2] = eta * (load @ axial) / box
        velocities[2, on, 2] = axial[velocity_of[on]]
    tensor = 0.5 * (tensor + tensor.T)
    porosity = float(np.abs(fluid.areas()).sum() / box)
    out = {
        "tensor": tensor,
        "mean": float(np.trace(tensor) / 3.0),
        "porosity": porosity,
        "unknowns": int(system.shape[0]),
        "parts": int(parts),
        "time_s": round(time.perf_counter() - t0, 3),
    }
    if fields:
        out.update({"pores": fluid, "velocity": velocities})
    out.update({f"k{n}": float(tensor[i, j]) for (i, j), n in zip(VOIGT, VOIGT_NAMES, strict=True)})
    log.info("flow of the plane: %d unknowns, k11 %.4g, k22 %.4g, k33 %.4g (%.1f s)",
             out["unknowns"], tensor[0, 0], tensor[1, 1], tensor[2, 2], out["time_s"])  # fmt: skip
    return out
