"""Creeping flow through the pores of a conforming mesh: the permeability of Darcy's law by
the finite elements of Taylor and Hood in space.

A fluid of viscosity ``eta`` creeps through the pores of a rigid solid under a macroscopic
pressure gradient: the Stokes problem ``eta lap(v) - grad p + f = 0``, ``div v = 0`` in
the pores, ``v = 0`` on their walls, the velocity and the pressure periodic, ``f`` the
uniform force of the gradient. The mean velocity over the whole cell obeys Darcy's law,
``<v> = (k / eta) f``. The lattice Boltzmann method solves it on voxels, the walls stairs
(:mod:`morphogen.homogenization.flow`); here the pores are the tetrahedra of a conforming
mesh, whose walls follow the interfaces — curved along them with ten-node elements
(:func:`~morphogen.mesh.curved.curved_mesh`).

**Method.** The elements of Taylor and Hood, as in the plane
(:mod:`morphogen.homogenization.plane_flow`): the velocity quadratic on the ten-node
tetrahedra of the pores (isoparametric), the pressure linear on their corners and
continuous; the weak form ``eta int grad v : grad w - int p div w = int f . w``,
``int q div v = 0``, integrated at the four points of the tetrahedra (exact on straight
ones). The walls are the faces between the pores and the other phases, or the free faces
of a mesh of the pores alone inside the box; the nodes of the faces of the box are those of
their images; the pressure is fixed in each connected part of the pores. The saddle point
is factorized (LU) for the three forces at once up to :data:`DIRECT_LIMIT` unknowns, and
solved beyond by MINRES, preconditioned by a block diagonal: a V-cycle of smoothed
aggregation on the viscous block of each component, and the inverse of the lumped mass
matrix of the pressures over the viscosity (the Schur complement of Stokes is spectrally
equivalent to it): about a hundred iterations per force whatever the size of the mesh.

References: C. Taylor, P. Hood, Comput. Fluids 1 (1973) 73; H. C. Elman, D. J. Silvester,
A. J. Wathen, Finite Elements and Fast Iterative Solvers (2005); A. A. Zick, G. M. Homsy,
J. Fluid Mech. 115 (1982) 13.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import csgraph

from morphogen.homogenization.core import VOIGT_NAMES
from morphogen.homogenization.elements import VOIGT
from morphogen.homogenization.fe import GAUSS
from morphogen.homogenization.plane_flow import pore_phases
from morphogen.mesh.tetra import (
    TET_EDGES,
    TET_FACE_MIDS,
    TET_FACES,
    TetMesh,
    corner_gradients,
    edge_deviations,
)

log = logging.getLogger("morphogen")

__all__ = ["DIRECT_LIMIT", "mesh_permeability"]

#: The unknowns of the saddle point up to which it is factorized (beyond: MINRES; the
#: factorization of the saddle points of space fills fast: 140 s for 27 500 unknowns, where
#: MINRES takes 9 s)
DIRECT_LIMIT = 8_000


def _representatives(mesh: TetMesh) -> np.ndarray:
    """The node that stands for each node and its images across the faces of the box."""
    rep = np.arange(mesh.n_nodes)
    pairs = [mesh.periodic_pairs(d) for d in range(3)]
    for _ in range(4):  # the corners of the box have seven images
        for pair in pairs:
            if len(pair):
                rep[pair[:, 1]] = rep[pair[:, 0]]
    while True:
        further = rep[rep]
        if np.array_equal(further, rep):
            return rep
        rep = further


def _walls(mesh: TetMesh) -> np.ndarray:
    """The nodes of the walls of a mesh of the pores: those of the faces of a single
    element that do not lie on a face of the box (corners and mid-edge nodes)."""
    e = mesh.elements
    corners = e[:, TET_FACES].reshape(-1, 3)
    middles = e[:, TET_FACE_MIDS].reshape(-1, 3)
    key = np.sort(corners, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    single = counts[inverse.ravel()] == 1
    x = mesh.nodes[corners]
    tol = mesh.tolerance()
    on_box = np.zeros(len(corners), dtype=bool)
    for d in range(3):
        for target in (mesh.lower[d], mesh.upper[d]):
            on_box |= np.all(np.abs(x[:, :, d] - target) <= tol, axis=1)
    wall = single & ~on_box
    out = np.zeros(mesh.n_nodes, dtype=bool)
    out[corners[wall].ravel()] = True
    out[middles[wall].ravel()] = True
    return out


def _element_arrays(mesh: TetMesh):
    """The arrays of the ten-node tetrahedra at their four integration points: the
    stiffness of the Laplacian ``(m, 10, 10)``, the divergence against the linear pressures
    ``(m, 4, 10, 3)``, the integrals of the quadratic functions ``(m, 10)`` and the lumped
    masses of the corners ``(m, 4)``. Curved elements are their straight tetrahedra moved
    by ``sum_k N_k d_k``: their gradients those of the straight one times ``F^-1``, their
    weights times ``det F``."""
    e = mesh.elements
    dl = corner_gradients(mesh.nodes, e)  # (m, 4, 3)
    p = mesh.nodes[e[:, :4]]
    volume = np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                       np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6.0  # fmt: skip
    points, weights = GAUSS[2]
    dev = edge_deviations(mesh.nodes, e)
    bent = np.flatnonzero(np.abs(dev).max(axis=(1, 2)) > 0.0)
    m, q = len(e), len(points)
    grads = np.empty((m, q, 10, 3))
    w = np.empty((m, q))
    shapes = np.empty((q, 10))
    i, j = TET_EDGES[:, 0], TET_EDGES[:, 1]
    for k, lam in enumerate(points):
        shapes[k, :4] = lam * (2.0 * lam - 1.0)
        shapes[k, 4:] = 4.0 * lam[i] * lam[j]
        g = np.empty((m, 10, 3))
        g[:, :4] = (4.0 * lam - 1.0)[None, :, None] * dl
        g[:, 4:] = 4.0 * (lam[i][None, :, None] * dl[:, j] + lam[j][None, :, None] * dl[:, i])
        w[:, k] = volume * weights[k]
        if len(bent):
            f = np.eye(3) + np.einsum("eka,ekb->eab", dev[bent], g[bent, 4:])
            g[bent] = np.einsum("enb,eba->ena", g[bent], np.linalg.inv(f))
            w[bent, k] *= np.linalg.det(f)
        grads[:, k] = g
    laplacian = np.einsum("mq,mqai,mqbi->mab", w, grads, grads, optimize=True)
    divergence = np.einsum("mq,qi,mqac->miac", w, points, grads, optimize=True)
    integrals = np.einsum("mq,qa->ma", w, shapes, optimize=True)
    lumped = np.einsum("mq,qi->mi", w, points, optimize=True)
    return laplacian, divergence, integrals, lumped


def mesh_permeability(
    mesh: TetMesh,
    materials=None,
    *,
    pores=None,
    viscosity: float = 1.0,
    solver: str = "auto",
    tol: float = 1e-8,
    maxiter: int | None = None,
    fields: bool = False,
) -> dict:
    """The permeability tensor of the pores of a periodic conforming mesh (see the
    module): the flows of the fluid of viscosity ``viscosity`` for the unit forces along
    ``x``, ``y`` and ``z`` (the tensor does not depend on it). ``solver``: ``"auto"`` (a
    factorization up to :data:`DIRECT_LIMIT` unknowns, MINRES beyond), ``"direct"`` or
    ``"minres"`` (to the relative residual ``tol``).

    Returns ``"tensor"`` (3 x 3, in the squared unit of the coordinates; symmetrized), its
    entries ``k11`` ... ``k12``, ``"mean"`` (the mean of its diagonal), the
    ``"porosity"``, the ``"unknowns"``, the connected ``"parts"`` of the pores, the
    ``"solver"`` and its ``"iterations"``, and the ``"time_s"``; with ``fields``, the mesh
    of the pores (``"pores"``) and the velocities of its nodes under the three forces
    (``"velocity"``, ``(3, nodes, 3)``)."""
    if not isinstance(mesh, TetMesh):
        raise TypeError("the flow of a conforming mesh needs a TetMesh (morphogen.mesh.tetra)")
    if solver not in ("auto", "direct", "minres"):
        raise ValueError("solver: 'auto', 'direct' or 'minres'")
    t0 = time.perf_counter()
    phases = pore_phases(mesh, materials, pores)
    box = float(np.prod(mesh.size))
    if not phases:
        if float(np.abs(mesh.volumes()).sum()) < (1.0 - 1e-9) * box:
            raise ValueError("the flow of a mesh: the pores are left out of the mesh; mesh "
                             "them with the solid (phases: all)")  # fmt: skip
        log.warning("no pores in the mesh: its permeability is zero")
        out = {"tensor": np.zeros((3, 3)), "mean": 0.0, "porosity": 0.0, "unknowns": 0}
        out.update({f"k{n}": 0.0 for n in VOIGT_NAMES})
        return out
    if getattr(mesh, "whole", False) or not mesh.periodic or not mesh.matching_faces():
        raise ValueError("the flow through the pores of a mesh: periodic conditions, a mesh "
                         "of the box with matching faces (periodic=True)")  # fmt: skip
    fluid = mesh.subset(phases)
    if fluid.order == 1:
        fluid = fluid.quadratic()
    n = fluid.n_nodes
    rep = _representatives(fluid)
    wall = np.zeros(n, dtype=bool)
    np.logical_or.at(wall, rep, _walls(fluid))
    wall = wall[rep]
    if not wall.any():
        raise ValueError("pores without walls (no solid in the mesh): their flow is not bounded")
    own = rep == np.arange(n)
    free = np.flatnonzero(own & ~wall)
    number = np.full(n, -1)
    number[free] = np.arange(len(free))
    velocity_of = number[rep]
    corners = np.unique(fluid.elements[:, :4])
    pressure_nodes = np.unique(rep[corners])
    pressure_of = np.full(n, -1)
    pressure_of[pressure_nodes] = np.arange(len(pressure_nodes))
    pressure_of = pressure_of[rep]
    tet = pressure_of[fluid.elements[:, :4]]
    links = np.concatenate([tet[:, [a, b]] for a, b in TET_EDGES])
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

    laplacian, divergence, integrals, lumped = _element_arrays(fluid)
    eta = float(viscosity)
    vel = velocity_of[fluid.elements]  # (m, 10)
    pre = pressure_of[fluid.elements[:, :4]]  # (m, 4)
    ok = (vel[:, :, None] >= 0) & (vel[:, None, :] >= 0)
    rows = np.broadcast_to(vel[:, :, None], laplacian.shape)[ok]
    cols = np.broadcast_to(vel[:, None, :], laplacian.shape)[ok]
    lap = sp.coo_matrix((laplacian[ok], (rows, cols)), shape=(nv, nv)).tocsr()
    okb = np.broadcast_to((pre[:, :, None, None] >= 0) & (vel[:, None, :, None] >= 0),
                          divergence.shape)  # fmt: skip
    prow = np.broadcast_to(pre[:, :, None, None], divergence.shape)[okb]
    vcol = (3 * np.broadcast_to(vel[:, None, :, None], divergence.shape)
            + np.arange(3)[None, None, None, :])[okb]  # fmt: skip
    div = sp.coo_matrix((divergence[okb], (prow, vcol)), shape=(npr, 3 * nv)).tocsr()
    load = np.zeros(nv)
    okl = vel >= 0
    np.add.at(load, vel[okl], integrals[okl])
    mass = np.zeros(npr)
    okp = pre >= 0
    np.add.at(mass, pre[okp], lumped[okp])
    viscous = sp.kron(eta * lap, sp.identity(3), format="csr")
    system = sp.bmat([[viscous, -div.T], [-div, None]], format="csc")
    size = system.shape[0]
    rhs = np.zeros((size, 3))
    for c in range(3):
        rhs[c : 3 * nv : 3, c] = load
    method = solver if solver != "auto" else ("direct" if size <= DIRECT_LIMIT else "minres")
    iterations = 0
    if method == "direct":
        solution = spla.splu(system).solve(rhs)
    else:
        from morphogen.homogenization.amg import SmoothedAggregation

        cycle = SmoothedAggregation(eta * lap)
        inverse_mass = eta / np.maximum(mass, 1e-300)

        def precondition(r):
            r = np.asarray(r).ravel()
            out = np.empty_like(r)
            u = r[: 3 * nv].reshape(nv, 3)
            out[: 3 * nv] = np.column_stack([cycle(u[:, c]) for c in range(3)]).ravel()
            out[3 * nv :] = inverse_mass * r[3 * nv :]
            return out

        preconditioner = spla.LinearOperator((size, size), matvec=precondition)
        solution = np.empty((size, 3))
        counts = [0]

        def count(_):
            counts[0] += 1

        for c in range(3):
            x, info = spla.minres(system.tocsr(), rhs[:, c], M=preconditioner, rtol=tol,
                                  maxiter=maxiter or 10 * size, callback=count)  # fmt: skip
            if info > 0:
                log.warning("MINRES: the flow along %s did not reach the tolerance in %d "
                            "iterations", "xyz"[c], info)  # fmt: skip
            solution[:, c] = x
        iterations = counts[0]
    tensor = np.zeros((3, 3))
    velocities = np.zeros((3, n, 3))
    on = velocity_of >= 0
    for c in range(3):
        v = solution[: 3 * nv, c].reshape(nv, 3)
        tensor[:, c] = eta * (load @ v) / box
        velocities[c, on] = v[velocity_of[on]]
    tensor = 0.5 * (tensor + tensor.T)
    out = {
        "tensor": tensor,
        "mean": float(np.trace(tensor) / 3.0),
        "porosity": float(np.abs(fluid.volumes()).sum() / box),
        "unknowns": int(size),
        "parts": int(parts),
        "solver": method,
        "iterations": int(iterations),
        "time_s": round(time.perf_counter() - t0, 3),
    }
    out.update({f"k{n}": float(tensor[i, j]) for (i, j), n in zip(VOIGT, VOIGT_NAMES, strict=True)})
    if fields:
        out.update({"pores": fluid, "velocity": velocities})
    log.info("flow of the mesh: %d unknowns (%s), k11 %.4g, k22 %.4g, k33 %.4g (%.1f s)",
             size, method, tensor[0, 0], tensor[1, 1], tensor[2, 2], out["time_s"])  # fmt: skip
    return out
