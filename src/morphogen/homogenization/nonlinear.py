"""Nonlinear homogenization along loading paths: the periodic element.

The unknowns of a step are the fluctuation ``w`` of the displacement ``u = E x + w``
(periodic, or zero on the boundary for kinematic conditions) and the components of the
macroscopic strain ``E`` that are not prescribed: the macroscopic degrees of freedom of
the periodic element of Z-set. Their equations are

* equilibrium, ``R_w = int B^T sigma dV = 0`` on the reduced unknowns of ``w``;
* the macroscopic stress, ``R_E = int sigma dV - V Sigma = 0`` on the components of
  ``E`` prescribed as stresses (``V``: the volume of the box, voids included). The
  reactions of the other components of ``E`` are ``V Sigma``.

The stresses come from the behaviours of the phases (von Mises elasto-viscoplasticity,
:mod:`morphogen.homogenization.behaviours`; crystal plasticity,
:mod:`morphogen.homogenization.crystal`), integrated implicitly over each step. Newton's
method solves the equations with the consistent tangent:

    [ K_ww   K_wE ] [dw]      [R_w]
    [ K_Ew   K_EE ] [dE] = -  [R_E],    K_ww = int B^T C B, K_wE = int B^T C, K_EE = int C

whose macroscopic block is condensed: ``K_ww Y = K_wE`` (one solve per unknown
component of ``E``, from the solutions of the previous iteration) gives the 6 x 6 (at
most) system ``(K_EE - K_Ew Y) dE = -R_E + K_Ew K_ww^-1 R_w``. The linear solvers are
those of the linear problems (sparse LU, or conjugate gradients with algebraic
multigrid, whose hierarchy is built once per step). The consistent tangents of the
points are symmetrized for them when the laws of all the phases have symmetric ones;
otherwise (``tangent``, :data:`TANGENTS`) they are kept whole, ``K_Ew`` is no longer
``K_wE^T`` and the tangent is solved by LU or by GMRES (with the multigrid of its
symmetric part), the Krylov solvers symmetrizing them again at the linearizations where
their measured asymmetry is below :data:`ASYMMETRY`. A step that does not converge, whose
local integration fails, or whose corrections by GMRES stagnate at :data:`_STALLED`
Newton iterations in a row, is divided by two (at most ``max_cuts`` times), and the step
grows back after two converged steps.

In finite strains (``kinematics="finite"``, :mod:`morphogen.homogenization.finite`) the
equations are written on the reference configuration with the first Piola-Kirchhoff
stress ``P`` and its tangent ``dP/dF``; the macroscopic gradient ``F = I + E`` is upper
triangular, its components ``E`` (nominal strains and simple shears) conjugate to those
of the mean of ``P`` (nominal stresses). The outputs add the Cauchy stress and the
logarithmic strain.

For linear behaviours in small strains the solutions are those of the condensation of
:func:`morphogen.homogenization.core.superpose`.

The time stepping and the Newton iterations (:class:`_PathSolver`) are shared with the
FFT solver of voxel images (:mod:`morphogen.homogenization.fft_nonlinear`), which gives
its own discretization and linear solver.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import scipy.sparse as sp
from numba import njit, prange

from morphogen.homogenization import dilatation
from morphogen.homogenization import finite as fin
from morphogen.homogenization.behaviours import (
    CRYSTAL,
    ELASTIC,
    HILL,
    MFRONT,
    N_PARAMS,
    PLASTIC,
    POROUS,
    STATE_P,
    mises,
    update_points,
)
from morphogen.homogenization.behaviours import state_width as j2_width
from morphogen.homogenization.crystal import (
    _BROKEN,
    CP_ALPHA,
    CP_KIND,
    CP_WIDTH,
    OPENING,
    SLIP,
    CrystalPlasticity,
    turn_twins,
)
from morphogen.homogenization.crystal import state_width as cp_width
from morphogen.homogenization.elements import voigt_to_tensor
from morphogen.homogenization.fe import (
    CHUNKS,
    GAUSS,
    FEElasticProblem,
    Reduction,
    _expand_columns,
    _geometry,
    _point_gradients,
    _strain_of,
    chunk_slots,
    element_colours,
    node_pattern,
)
from morphogen.homogenization.materials import Material, MaterialTable, material_table
from morphogen.homogenization.paths import LoadingPath, PathResult
from morphogen.homogenization.porous import P_F0, STATE_FN, porosity
from morphogen.homogenization.porous import state_width as porous_width
from morphogen.homogenization.solvers import choose_method, dot, norm, times
from morphogen.homogenization.tensors import symmetrize
from morphogen.settings import without_blas_threads

log = logging.getLogger("morphogen")

#: Kinematics of the nonlinear problems
KINEMATICS = ("small", "finite")
#: Consistent tangents of the points in the Newton iterations of the volume elements:
#: symmetrized and solved by conjugate gradients, or full and solved by GMRES (sparse LU
#: by finite elements); ``auto`` symmetrizes them when the laws of all the phases have
#: symmetric tangents, and otherwise, for the Krylov solvers, where their measured
#: asymmetry is below :data:`ASYMMETRY`
TANGENTS = ("auto", "symmetric", "full")
#: With ``tangent="auto"`` and laws whose tangents are not all symmetric, the volume
#: elements solved by Krylov methods (voxels; finite elements with multigrid) measure the
#: asymmetry of the consistent tangents of the points of each phase at each linearization
#: (:func:`asymmetry`) and symmetrize them where the largest is below this value
#: (conjugate gradients, MINRES); above it they solve the full tangents by GMRES
ASYMMETRY = 2e-2
#: Engineering strains of the tensor components of the outputs
_HALF_SHEAR = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
#: Relative residual of the solutions of the macroscopic coupling ``K_ww Y = K_wE``, and
#: largest one of the corrections of an inexact Newton iteration (the residual of the
#: equilibrium decides the convergence)
_MACRO_TOL = 1e-4
_MAX_FORCING = 1e-2
#: The macroscopic coupling is computed again at this Newton iteration of a step, and at
#: the start of the next step when it took more iterations
_REFRESH_AFTER = 3
#: A step is predicted at the second order (:func:`_predicted`) where, on the last step,
#: the error of that prediction was below this share of the error of the first-order one,
#: itself below this share of the increments of the step (a response that does not turn)
_SECOND = 0.5
#: The condensed block of the macroscopic unknowns kept from an earlier linearization is
#: updated by the secant of a correction (:func:`_secant`) that has brought the merit
#: below this share of what it was (a tenth of the residuals: a correction of the
#: converging iteration, whose step the linearization describes)
_TRUSTED = 1e-2
#: Powell's damping of that update: the curvature of the secant along the step between
#: this share of the curvature of the block and its inverse
_DAMPING = 0.2
#: A step whose corrections by GMRES (full tangents) stop above their tolerance at this
#: many Newton iterations in a row is divided: its tangents are too far from definite for
#: the Krylov solvers (the softening voxels of a step too large), and its iterations make
#: little progress
_STALLED = 3
#: Gradient (9 components) of each macroscopic strain component (6, engineering shears)
_MACRO = fin.gradient_of(np.eye(6)).T
#: The round-off of the stresses of a material point relative to its stiffness times its
#: strain: the residual it reaches at zero stress (a thousand times the machine epsilon)
_ROUND_OFF = 1e3 * float(np.finfo(float).eps)
#: In finite strains, the round-off of the stresses of a gradient ``I + H`` relative to
#: the stiffness, which does not vanish with ``H``: what is left of the stress of a point
#: brought back to no strain after a plastic history (ten times the machine epsilon)
_ROUND_OFF_UNIT = 10.0 * float(np.finfo(float).eps)
#: The share of the largest scales of the residuals reached along a path that the scales
#: of a state keep: an unloaded state (an elastic cell brought back to zero strain) has no
#: force of its own with which to compare its residuals
_UNLOADED = 1e-6


def kinematics_of(value) -> bool:
    """Whether the kinematics ``"small"`` or ``"finite"`` is finite."""
    if value not in KINEMATICS:
        raise ValueError(f"kinematics: one of {KINEMATICS}, got {value!r}")
    return value == "finite"


# -- kernels ------------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _point_strains(nodes, elements, order, u, gp, gw, eps, wts):
    """Strains (engineering) and weights of the integration points of every element."""
    n_el, nn = elements.shape
    ng = gp.shape[0]
    for e in prange(n_el):
        x = np.empty((4, 3))
        gl = np.empty((4, 3))
        dev = np.empty((6, 3))
        grad = np.empty((nn, 3))
        tmp = np.empty(6)
        vol, curved = _geometry(nodes, elements, e, x, gl, dev)
        for g in range(ng):
            wts[e * ng + g] = gw[g] * vol * _point_gradients(order, curved, gl, dev, gp[g], grad)
            _strain_of(grad, elements, e, u, tmp)
            for r in range(6):
                eps[e * ng + g, r] = tmp[r]


@njit(parallel=True, cache=True)
def _internal_forces(nodes, elements, order, gp, sigma, wts, by_colour, colours, forces):
    """Nodal forces ``int B^T sigma`` (into ``forces``), the integral of the stress, and
    the norm of the contributions of the integration points (before they cancel); the
    elements of a colour in parallel (``by_colour``, ``colours``:
    :func:`~morphogen.homogenization.fe.element_colours`), the sums over all the elements
    chunk by chunk, in the same order for any number of threads."""
    nn = elements.shape[1]
    ng = gp.shape[0]
    for i in prange(forces.size):
        forces[i] = 0.0
    n_colours = colours.size - 1
    # the sums of each chunk (rows wide enough for two chunks not to share a cache line)
    slots = chunk_slots(colours)
    sums = np.zeros((slots[n_colours], 16))
    xs = np.empty((CHUNKS, 4, 3))
    gls = np.empty((CHUNKS, 4, 3))
    devs = np.empty((CHUNKS, 6, 3))
    grads = np.empty((CHUNKS, nn, 3))
    for colour in range(n_colours):
        first = colours[colour]
        count = colours[colour + 1] - first
        chunks = min(count, CHUNKS)
        for chunk in prange(chunks):
            x = xs[chunk]
            gl = gls[chunk]
            dev = devs[chunk]
            grad = grads[chunk]
            part = sums[slots[colour] + chunk]
            for q in range(first + chunk * count // chunks, first + (chunk + 1) * count // chunks):
                e = by_colour[q]
                _, curved = _geometry(nodes, elements, e, x, gl, dev)
                for g in range(ng):
                    _point_gradients(order, curved, gl, dev, gp[g], grad)
                    i = e * ng + g
                    w = wts[i]
                    s = sigma[i]
                    for r in range(6):
                        part[r] += w * s[r]
                    for b in range(nn):
                        k = elements[e, b]
                        gx, gy, gz = grad[b, 0], grad[b, 1], grad[b, 2]
                        fx = w * (gx * s[0] + gz * s[4] + gy * s[5])
                        fy = w * (gy * s[1] + gz * s[3] + gx * s[5])
                        fz = w * (gz * s[2] + gy * s[3] + gx * s[4])
                        forces[3 * k] += fx
                        forces[3 * k + 1] += fy
                        forces[3 * k + 2] += fz
                        part[6] += fx * fx + fy * fy + fz * fz
    total = np.zeros(6)
    norm2 = 0.0
    for k in range(sums.shape[0]):
        for r in range(6):
            total[r] += sums[k, r]
        norm2 += sums[k, 6]
    return total, math.sqrt(norm2)


@njit(parallel=True, cache=True)
def _fill_tangent(nodes, elements, order, gp, tangent, wts, indptr_n, indices_n, by_colour,
                  colours, data):  # fmt: skip
    """CSR values of ``sum_e int B^T C B`` with the tangent ``C`` of every integration
    point (node-level pattern ``indptr_n``, ``indices_n``), the elements of a colour in
    parallel (``by_colour``, ``colours``:
    :func:`~morphogen.homogenization.fe.element_colours`)."""
    nn = elements.shape[1]
    ng = gp.shape[0]
    for i in prange(data.size):
        data[i] = 0.0
    xs = np.empty((CHUNKS, 4, 3))
    gls = np.empty((CHUNKS, 4, 3))
    devs = np.empty((CHUNKS, 6, 3))
    grads = np.empty((CHUNKS, nn, 3))
    dbs = np.empty((CHUNKS, nn, 6, 3))
    kes = np.empty((CHUNKS, nn * 3, nn * 3))
    places = np.empty((CHUNKS, nn, nn), dtype=np.int64)
    for colour in range(colours.size - 1):
        first = colours[colour]
        count = colours[colour + 1] - first
        chunks = min(count, CHUNKS)
        for chunk in prange(chunks):
            x = xs[chunk]
            gl = gls[chunk]
            dev = devs[chunk]
            grad = grads[chunk]
            db = dbs[chunk]
            ke = kes[chunk]
            pos = places[chunk]
            for q in range(first + chunk * count // chunks, first + (chunk + 1) * count // chunks):
                e = by_colour[q]
                _, curved = _geometry(nodes, elements, e, x, gl, dev)
                ke[:, :] = 0.0
                for g in range(ng):
                    _point_gradients(order, curved, gl, dev, gp[g], grad)
                    i = e * ng + g
                    w = wts[i]
                    d = tangent[i]
                    for b in range(nn):
                        gx, gy, gz = grad[b, 0], grad[b, 1], grad[b, 2]
                        for r in range(6):
                            db[b, r, 0] = d[r, 0] * gx + d[r, 4] * gz + d[r, 5] * gy
                            db[b, r, 1] = d[r, 1] * gy + d[r, 3] * gz + d[r, 5] * gx
                            db[b, r, 2] = d[r, 2] * gz + d[r, 3] * gy + d[r, 4] * gx
                    for a in range(nn):
                        gx, gy, gz = grad[a, 0], grad[a, 1], grad[a, 2]
                        for b in range(nn):
                            for c2 in range(3):
                                ke[3 * a, 3 * b + c2] += w * (
                                    gx * db[b, 0, c2] + gz * db[b, 4, c2] + gy * db[b, 5, c2]
                                )
                                ke[3 * a + 1, 3 * b + c2] += w * (
                                    gy * db[b, 1, c2] + gz * db[b, 3, c2] + gx * db[b, 5, c2]
                                )
                                ke[3 * a + 2, 3 * b + c2] += w * (
                                    gz * db[b, 2, c2] + gy * db[b, 3, c2] + gx * db[b, 4, c2]
                                )
                for a in range(nn):
                    na = elements[e, a]
                    lo = indptr_n[na]
                    hi = indptr_n[na + 1]
                    for b in range(nn):
                        nb = elements[e, b]
                        left, right = lo, hi
                        while left < right:
                            mid = (left + right) // 2
                            if indices_n[mid] < nb:
                                left = mid + 1
                            else:
                                right = mid
                        pos[a, b] = left - lo
                for a in range(nn):
                    na = elements[e, a]
                    length = indptr_n[na + 1] - indptr_n[na]
                    for c in range(3):
                        start = 9 * indptr_n[na] + c * 3 * length
                        for b in range(nn):
                            p = pos[a, b]
                            for c2 in range(3):
                                data[start + p * 3 + c2] += ke[3 * a + c, 3 * b + c2]


@njit(parallel=True, cache=True)
def _coupling(nodes, elements, order, gp, tangent, wts, by_colour, colours, out, kee):
    """``int B^T C`` (nodal forces of the six unit macroscopic strains, into ``out``,
    n_dofs x 6) and ``int C`` (into ``kee``); the elements of a colour in parallel, as
    :func:`_internal_forces`."""
    nn = elements.shape[1]
    ng = gp.shape[0]
    for i in prange(out.shape[0]):
        for j in range(6):
            out[i, j] = 0.0
    n_colours = colours.size - 1
    slots = chunk_slots(colours)
    sums = np.zeros((slots[n_colours], 48))  # (36 values: the rows of two chunks apart)
    xs = np.empty((CHUNKS, 4, 3))
    gls = np.empty((CHUNKS, 4, 3))
    devs = np.empty((CHUNKS, 6, 3))
    grads = np.empty((CHUNKS, nn, 3))
    for colour in range(n_colours):
        first = colours[colour]
        count = colours[colour + 1] - first
        chunks = min(count, CHUNKS)
        for chunk in prange(chunks):
            x = xs[chunk]
            gl = gls[chunk]
            dev = devs[chunk]
            grad = grads[chunk]
            part = sums[slots[colour] + chunk]
            for q in range(first + chunk * count // chunks, first + (chunk + 1) * count // chunks):
                e = by_colour[q]
                _, curved = _geometry(nodes, elements, e, x, gl, dev)
                for g in range(ng):
                    _point_gradients(order, curved, gl, dev, gp[g], grad)
                    i = e * ng + g
                    w = wts[i]
                    d = tangent[i]
                    for r in range(6):
                        for c in range(6):
                            part[6 * r + c] += w * d[r, c]
                    for b in range(nn):
                        k = elements[e, b]
                        gx, gy, gz = grad[b, 0], grad[b, 1], grad[b, 2]
                        for j in range(6):
                            out[3 * k, j] += w * (gx * d[0, j] + gz * d[4, j] + gy * d[5, j])
                            out[3 * k + 1, j] += w * (gy * d[1, j] + gz * d[3, j] + gx * d[5, j])
                            out[3 * k + 2, j] += w * (gz * d[2, j] + gy * d[3, j] + gx * d[4, j])
    kee[:, :] = 0.0
    for k in range(sums.shape[0]):
        for r in range(6):
            for c in range(6):
                kee[r, c] += sums[k, 6 * r + c]


@njit(parallel=True, cache=True)
def _group_sums(group, n_groups, wts, sigma, eps, p):
    """Integrals over each group of the stress, the strain (engineering), the cumulated
    plastic strain ``p`` and the volume."""
    n = wts.shape[0]
    n_chunks = min(max(n, 1), 256)
    partial = np.zeros((n_chunks, n_groups, 14))
    for ch in prange(n_chunks):
        for i in range(ch * n // n_chunks, (ch + 1) * n // n_chunks):
            acc = partial[ch, group[i]]
            w = wts[i]
            for r in range(6):
                acc[r] += w * sigma[i, r]
                acc[6 + r] += w * eps[i, r]
            acc[12] += w * p[i]
            acc[13] += w
    return partial.sum(axis=0)


# -- behaviours of the elements -------------------------------------------------------------


@dataclass
class Behaviours:
    """Behaviours of the elements of phases: the kind of each element (elastic, von Mises,
    crystal, MFront), its row of parameters (``pmat``: of :attr:`params`, of the crystal
    arrays or of :attr:`mfront`; -1 for elastic elements), the data of the integration
    kernels (see :func:`morphogen.homogenization.behaviours.update_points`), the number of
    internal variables of a point, the crystal plasticity of each crystal row and the laws
    of the points integrated outside the kernels (:attr:`mfront`: the MFront behaviours
    loaded by MGIS, :class:`~morphogen.homogenization.mfront.MFrontLaw`, and the learned
    laws, :class:`~morphogen.learning.behaviour.LearnedLaw`). :attr:`symmetric`: whether
    the consistent tangents of all the laws are symmetric (see :func:`behaviour_table`)."""

    kind: np.ndarray
    pmat: np.ndarray
    data: tuple
    width: int
    crystals: list = field(default_factory=list)
    mfront: list = field(default_factory=list)
    symmetric: bool = True

    @property
    def nonlinear(self) -> bool:
        return bool(np.any(self.kind != ELASTIC))


def behaviour_table(
    phase, materials: dict[int, Material], table: MaterialTable, *, finite: bool = False
) -> Behaviours:
    """Behaviours of the elements of phases ``phase`` with their materials ``table``
    (:func:`morphogen.homogenization.materials.material_table`): the slip systems of the
    crystals turn with the orientation of their material.

    The consistent tangents are symmetric (:attr:`Behaviours.symmetric`) for elasticity,
    for von Mises and Hill plasticity without Armstrong-Frederick terms (``D = 0``), for
    crystal plasticity in small strains without isotropic hardening (``Q b = 0``), for the
    learned potentials and standard models and for the MFront behaviours that declare it;
    not for the Armstrong-Frederick hardening (the backstresses turn the direction of
    flow), the isotropic hardening of the slip systems and the multiplicative crystal
    plasticity of finite strains, porous plasticity (the growth of the porosity), the
    learned perceptrons and recurrent networks, and the other MFront behaviours."""
    phase = np.asarray(phase).ravel().astype(np.int64)
    kind = np.full(phase.size, ELASTIC, dtype=np.int64)
    pmat = np.full(phase.size, -1, dtype=np.int64)
    rows, n_kin, porous_kin = [], 0, -1
    symmetric = True
    crystals: list[CrystalPlasticity] = []
    laws = []
    for p in np.unique(phase):
        m = materials[int(p)]
        sel = phase == p
        if m.void:
            continue
        if m.mfront is not None:
            from morphogen.homogenization.mfront import load_behaviour

            kind[sel] = MFRONT
            pmat[sel] = len(laws)
            laws.append(load_behaviour(m.mfront, finite=finite))
        elif m.learned is not None:  # integrated outside the kernels too, by its model
            from morphogen.learning.behaviour import learned_law

            kind[sel] = MFRONT
            pmat[sel] = len(laws)
            heat = m.expansion_tensor() if m.expands else None  # in the axes of the material
            laws.append(learned_law(m.learned, finite=finite, expansion=heat))
        elif m.analytic:  # viscoelasticity, hyperelasticity, damage: likewise, in closed form
            from morphogen.homogenization.laws import analytic_law

            kind[sel] = MFRONT
            pmat[sel] = len(laws)
            heat = m.expansion_tensor() if m.expands else None
            laws.append(analytic_law(m, finite=finite, expansion=heat))
        elif m.plastic:
            kind[sel] = PLASTIC if m.plasticity.hill is None else HILL
            pmat[sel] = len(rows)
            rows.append(m.plasticity.params(*m.lame))
            n_kin = max(n_kin, len(m.plasticity.kinematic))
            symmetric &= all(c == 0.0 or d == 0.0 for c, d in m.plasticity.kinematic)
        elif m.porous:
            kind[sel] = POROUS
            pmat[sel] = len(rows)
            rows.append(m.porous_plasticity.params(*m.lame))
            porous_kin = max(porous_kin, len(m.porous_plasticity.kinematic))
            symmetric = False
        elif m.crystal_plastic:
            kind[sel] = CRYSTAL
            pmat[sel] = len(crystals)
            crystals.append(m.crystal_plasticity)
            par = m.crystal_plasticity.params_array()
            symmetric &= not finite and bool(np.all((par[:, 1] == 0.0) | (par[:, 2] == 0.0)))
            # the twins slow each other and the systems of a cleavage plane soften together
            symmetric &= not m.crystal_plasticity.extended
    # the rows of the porous and Hill points are longer (the GPU pads them all to ROW_WIDTH)
    n_par = max([N_PARAMS] + [len(r) for r in rows])
    params = (
        np.array([np.pad(r, (0, n_par - len(r))) for r in rows])
        if rows
        else np.zeros((1, N_PARAMS))
    )
    n_max = max((c.n_systems for c in crystals), default=1)
    n_cp = max(len(crystals), 1)
    # the kinds of the systems after their parameters, for the crystals that twin or cleave
    cp_par = np.zeros((n_cp, n_max, CP_WIDTH if any(c.extended for c in crystals) else CP_KIND))
    cp_h = np.zeros((n_cp, n_max, n_max))
    cp_n = np.zeros(n_cp, dtype=np.int64)
    for c, cp in enumerate(crystals):
        ns = cp.n_systems
        cp_par[c, :ns, :CP_KIND] = cp.params_array()
        if cp_par.shape[2] > CP_KIND:
            cp_par[c, :ns, CP_KIND:] = cp.kinds_array()
        cp_h[c, :ns, :ns] = cp.interaction_matrix()
        cp_n[c] = ns
    n_mats = table.matrices.shape[0]
    systems = np.zeros((max(n_mats, 1), n_max, 6))
    for c, cp in enumerate(crystals):
        normals, directions = cp.systems()
        mats = np.unique(table.mat[(kind == CRYSTAL) & (pmat == c)])
        for m in mats[mats >= 0]:
            g = table.rotations[m]
            # vectors of the crystal in the axes of the sample: g^T v
            systems[m, : cp.n_systems, :3] = directions @ g
            systems[m, : cp.n_systems, 3:] = normals @ g
    width = max([j2_width(n_kin)]
                + [cp_width(c.n_systems, finite, c.reorients, c.micro_width) for c in crystals]
                + [law.width for law in laws]
                + ([porous_width(porous_kin)] if porous_kin >= 0 else []))  # fmt: skip
    data = (
        np.ascontiguousarray(table.matrices),
        np.ascontiguousarray(table.eigen),
        np.ascontiguousarray(params),
        int(n_kin),
        np.ascontiguousarray(systems),
        cp_par,
        cp_h,
        cp_n,
    )
    symmetric = bool(symmetric and all(law.symmetric for law in laws))
    return Behaviours(kind=kind, pmat=pmat, data=data, width=int(width), crystals=crystals,
                      mfront=laws, symmetric=symmetric)  # fmt: skip


def _lattice_turns(crystal, finite: bool, grads, state) -> np.ndarray:
    """The rotations (points x 3 x 3) that the lattices of crystal points have turned by:
    ``R_e`` of the polar decomposition of ``F_e`` in finite strains (none in small
    strains), after that of the twin that turned the lattice, ``R_e R_t``. The lattice
    of orientation ``g`` is then ``g (R_e R_t)^T``."""
    if finite:
        off = CP_ALPHA + 2 * crystal.n_systems
        turns = fin.lattice_rotations(grads, state[:, off : off + 9] + np.eye(3).ravel())
    else:
        turns = np.broadcast_to(np.eye(3), (len(state), 3, 3))
    if crystal.reorients:
        first = cp_width(crystal.n_systems, finite)
        turns = turns @ (state[:, first : first + 9].reshape(-1, 3, 3) + np.eye(3))
    return turns


def _majority(lattices) -> np.ndarray:
    """The points of each element that have the lattice of most of its points (1, the
    others 0): ``lattices``, the rotations of their twins less the identity (elements x
    points x 9; the same numbers for the points that the same twin turned). The lattice
    of the grain unless more than half of the points are turned; then that of the twin
    of most of them (the first of the twins of as many points)."""
    moved = np.any(lattices != 0.0, axis=-1)
    turned = 2 * moved.sum(axis=1) > lattices.shape[1]
    same = np.all(lattices[:, :, None, :] == lattices[:, None, :, :], axis=-1)
    # the points that have the lattice of each point, among those turned (or not) as most
    count = np.where(moved == turned[:, None], same.sum(axis=2), 0)
    chosen = np.argmax(count, axis=1)
    return same[np.arange(len(lattices)), chosen].astype(float)


def _mfront_points(solver):
    """The points of the MFront behaviours of a solver of paths, integrated by MGIS
    (:class:`~morphogen.homogenization.mfront.MFrontPoints`, built at the first call; their
    tangents symmetrized as the solver's), or None without them."""
    beh = solver.behaviours
    if not beh.mfront:
        return None
    points = solver.__dict__.get("_mfront_points")
    if points is None:
        points = external_points(beh.mfront, solver.kind, solver.pmat, solver.point_mat,
                                 solver.table.rotations, solver.finite,
                                 symmetric=solver.symmetrize)  # fmt: skip
        solver.__dict__["_mfront_points"] = points
    return points


def external_points(laws, kind, pmat, point_mat, rotations, finite: bool, *,
                    symmetric: bool = True):  # fmt: skip
    """The points of the behaviours integrated outside the kernels (of the kind
    ``MFRONT``, of the ``laws`` ``pmat``): those of the MFront behaviours by MGIS
    (:class:`~morphogen.homogenization.mfront.MFrontPoints`), those of the learned laws by
    their models (:class:`~morphogen.learning.behaviour.LearnedPoints`), together, their
    tangents symmetrized with ``symmetric`` (for the conjugate gradients); None without
    them."""
    if not laws:
        return None
    from morphogen.learning.behaviour import LearnedLaw, LearnedPoints

    kind, pmat = np.asarray(kind), np.asarray(pmat)
    external = kind == MFRONT
    learned = np.array([isinstance(law, LearnedLaw) for law in laws])
    parts = []
    for flag in (False, True):
        own = np.flatnonzero(learned == flag)
        if own.size == 0:
            continue
        index = np.full(len(laws), -1)
        index[own] = np.arange(own.size)
        row = index[np.where(external, pmat, 0)]
        mine = external & (row >= 0)
        args = ([laws[i] for i in own], np.where(mine, MFRONT, ELASTIC),
                np.where(mine, row, -1), point_mat, rotations, finite)  # fmt: skip
        if flag:
            parts.append(LearnedPoints(*args, symmetric=symmetric))
        else:
            from morphogen.homogenization.mfront import MFrontPoints

            parts.append(MFrontPoints(*args, symmetric=symmetric))
    return parts[0] if len(parts) == 1 else _ExternalPoints(parts)


class _ExternalPoints:
    """The points of MFront behaviours and of learned laws of one solver together."""

    def __init__(self, parts):
        self.parts = parts
        self.points = np.sort(np.concatenate([part.points for part in parts]))

    def integrate(self, grads, dtemp, dt, old, new, stress, tangent) -> int:
        return sum(p.integrate(grads, dtemp, dt, old, new, stress, tangent) for p in self.parts)

    def integrate_compact(self, grads, dtemp, dt, old, new, stress, tangent) -> int:
        failed = 0
        for part in self.parts:
            pos = np.searchsorted(self.points, part.points)
            out = [new[pos], stress[pos], tangent[pos]]
            failed += part.integrate_compact(grads[pos], dtemp, dt, old[pos], *out)
            new[pos], stress[pos], tangent[pos] = out
        return failed


def _update(finite, beh_kind, mat, pmat, data, grads, dtemp, dt, old, new, stress, tangent, guess,
            ratio, symmetric: bool = True) -> int:  # fmt: skip
    """Integration of the behaviours of the points in small or finite strains, their
    tangents symmetrized with ``symmetric``."""
    update = fin.update_points_finite if finite else update_points
    return update(beh_kind, mat, pmat, data, grads, dtemp, dt, old, new, stress, tangent,
                  int(guess), float(ratio), bool(symmetric))  # fmt: skip


@njit(parallel=True, cache=True)
def _tangent_norms(tangent, out):
    """``out[p] = (|C_p - C_p^T|^2, |C_p|^2)``: the squared Frobenius norms of twice the
    skew part of the tangent ``C_p`` of each point and of the tangent."""
    nc = tangent.shape[1]
    for p in prange(tangent.shape[0]):
        s = 0.0
        c = 0.0
        for i in range(nc):
            for j in range(nc):
                a = tangent[p, i, j]
                d = a - tangent[p, j, i]
                c += a * a
                s += d * d
        out[p, 0] = s
        out[p, 1] = c


@njit(parallel=True, cache=True)
def _symmetrize_points(tangent):
    for p in prange(tangent.shape[0]):
        symmetrize(tangent[p])


def asymmetry(tangent, wts, groups=None) -> float:
    """The asymmetry of the consistent tangents ``C_p`` of the points (points x nc x nc,
    weights ``wts``) that decides ``tangent="auto"`` (:data:`ASYMMETRY`): ``|C - C^T| /
    |C|`` for the block-diagonal operator ``C`` of the points, ``sqrt(sum_p w_p |C_p -
    C_p^T|^2 / sum_p w_p |C_p|^2)`` (Frobenius norms), the stiffer points weighing more;
    with ``groups`` (the phase of each point), the largest of those of the phases, so that
    a small phase whose tangents are far from symmetric is not diluted by the others (a
    learned law at 0.6 % of the voxels: 1.3e-2 for the whole volume element, 0.2 for its
    phase). Point by point, ``|C_p - C_p^T| / |C_p|`` reaches 0.1 to 0.2 at a few points of
    morphoGen's own laws (porous points near their softening, crystals whose slip systems
    harden each other strongly) whose tangents are soft, where the operator stays nearly
    symmetric."""
    norms = np.empty((tangent.shape[0], 2))
    _tangent_norms(np.ascontiguousarray(tangent), norms)
    w = np.abs(np.asarray(wts, dtype=float))
    if groups is None:
        return _asymmetry_of(dot(w, norms[:, 0]), dot(w, norms[:, 1]))
    groups = np.asarray(groups)
    largest = 0.0
    for g in np.unique(groups):
        points = np.flatnonzero(groups == g)
        wg = w[points]
        largest = max(largest, _asymmetry_of(dot(wg, norms[points, 0]),
                                             dot(wg, norms[points, 1])))  # fmt: skip
    return largest


def _asymmetry_of(skew: float, full: float) -> float:
    """The asymmetry of :func:`asymmetry` from the sums of the squared norms of the ``C_p -
    C_p^T`` and of the ``C_p`` (weighted alike)."""
    return math.sqrt(skew / full) if full > 0.0 else 0.0


class _PointArrays:
    """The arrays of the integration points of a discretization of the loading paths (see
    :class:`_PathSolver`) in the memory of the CPU, and the integration of their behaviours
    there (:func:`_update`); :class:`morphogen.homogenization.gpu.VoxelPaths` keeps them on
    the GPU instead. ``problem``: the solver of the paths."""

    problem: _PathSolver

    def arrays(self, n_pts: int, nc: int, width: int) -> tuple:
        """Measures of strain, weights, stresses, tangents and two sets of internal
        variables of ``n_pts`` points (measures and stresses of ``nc`` components,
        ``width`` internal variables)."""
        return (np.zeros((n_pts, nc)), np.zeros(n_pts), np.zeros((n_pts, nc)),
                np.zeros((n_pts, nc, nc)), np.zeros((n_pts, width)),
                np.zeros((n_pts, width)))  # fmt: skip

    def integrate(self, grads, dtemp, dt, old, new, stress, tangent, guess, ratio) -> int:
        """Integration of the behaviours of the points over a step (see :func:`_update`;
        MGIS integrates the points of MFront behaviours), their tangents symmetrized as
        the problem says (:attr:`_PathSolver.symmetrize`): the number of points whose
        integration failed."""
        p = self.problem
        failed = _update(p.finite, p.kind, p.point_mat, p.pmat, p.behaviours.data, grads,
                         dtemp, dt, old, new, stress, tangent, guess, ratio,
                         p.symmetrize)  # fmt: skip
        points = _mfront_points(p)
        if points is not None:
            failed += points.integrate(grads, dtemp, dt, old, new, stress, tangent)
        return failed

    def stress_scale(self, sigma, wts) -> float:
        """Integral of the norm of the stresses of the points: the scale of the residuals of
        the macroscopic unknowns."""
        return float(np.sum(np.abs(wts) * np.sqrt(np.einsum("ij,ij->i", sigma, sigma))))

    def asymmetry(self, tangent, wts) -> float:
        """The largest asymmetry of the tangents of the points of the phases
        (:func:`asymmetry`)."""
        return asymmetry(tangent, wts, self.problem.point_group)

    def symmetrize(self, tangent) -> None:
        """The tangents of the points replaced by their symmetric parts, in place (to the
        bit those of the kernels that symmetrize them)."""
        _symmetrize_points(tangent)

    def host(self, *arrays) -> tuple:
        """The arrays in the memory of the CPU (for the outputs)."""
        return arrays


#: The number of iterates whose largest residual bounds the line search of Newton's
#: corrections where cleavage planes are regularized by their microdamage
_MICRO_WINDOW = 4
#: The share of the largest scales of the residuals reached along a path that the scales
#: of a state keep where cleavage planes are regularized by their microdamage, in place of
#: :data:`_UNLOADED`: a cell that a crack cuts through keeps no stress with which to
#: compare its residuals, and the local iterations of the planes resolve their stresses
#: to 1e-10 of their strength, no further
_CUT = 1e-3
#: The equilibria of a step between two updates of the microdamage are solved to this
#: share of the last change of the microdamage (over the damage that breaks a plane), no
#: looser than :data:`_LOOSE` and no tighter than the tolerance of Newton's iteration,
#: to which the last one is solved: the damage of an equilibrium is known to about the
#: error of its residual, which the next update takes over
_SHARE = 0.1
_LOOSE = 1e-2
#: The loosest relative residual of the solutions of the macroscopic coupling between the
#: updates of a microdamage (ten times the tolerance of the equilibria, at most this)
_COARSE = 1e-2
#: The acceleration of the alternate scheme of the microdamage (after Storvik et al.
#: 2021): the relaxation of its updates once their residual has grown over this many
#: passes in a row, and how far an extrapolated update may go past the plain one, in
#: units of the residual of each element
_RELAXATION = 1.6
_GROWING = 3
_REACH = 10.0
#: The share of the merit of the last iterate above which a correction was a poor one:
#: the condensation of the macroscopic unknowns is then computed again (with regularized
#: cleavage planes: where one breaks, the tangents of the cell are no longer those of
#: the condensation that the iterations keep)
_POOR = 0.25


@dataclass
class _Planes:
    """The cleavage planes of a crystal that their microdamage regularizes, at the points
    of its ``elements``: the columns of the cumulated slips of the systems of each plane
    (``members``: their sum is its damage) and of the slip systems (``slips``), that of
    the microdamage (``slot``; the marks of the planes follow it), the rate of the
    softening and the strength of each plane, the micromorphic modulus, the rate at which
    the slip lowers the strength, and the damage that breaks a plane."""

    elements: np.ndarray
    points: np.ndarray
    members: list
    slips: np.ndarray
    slot: int
    softening: np.ndarray
    strength: np.ndarray
    modulus: float
    coupling: float
    broken: float


class _Microdamage:
    """The microdamage of the crystals whose cleavage planes it regularizes (Aslan &
    Forest 2009; :attr:`~morphogen.homogenization.crystal.CrystalPlasticity.regularized`):
    a field ``chi_d`` that follows the damage ``d`` of the planes of the points over a
    length, ``chi_d - ell^2 lap chi_d = d``, and that their thresholds take off the damage
    (an internal variable of their points). The equilibrium of a step is solved for a
    given microdamage, which is then that of the damage reached, and again until it
    settles (a fixed point, as the alternate minimization of a phase field): the points
    harden at a fixed microdamage and soften as it follows them.

    A plane whose strength the microdamage has brought to nothing is broken through
    (:meth:`_break`): it is marked in its point, keeps no strength, and what it opens
    since is the opening of a crack, not a damage — the microdamage keeps the damage that
    the plane had, and does not spread from a crack that opens.

    ``solver``: the solver of the paths, whose ``_smooth(elements, values, length)``
    solves the equation of the microdamage for the damage ``values`` of its ``elements``
    (the means of their points)."""

    def __init__(self, solver):
        beh = solver.behaviours
        self.solver, self.ng = solver, solver.ng
        self.tol, self.most = float(solver.microdamage_tol), int(solver.microdamage_passes)
        self.parts: list[_Planes] = []
        #: the crystals of each length: they share a microdamage, that of the damage of
        #: them all (a crack goes from the one to the other)
        self.fields: dict[float, list[int]] = {}
        for c, cp in enumerate(beh.crystals):
            if not cp.regularized:
                continue
            elements = np.flatnonzero((beh.kind == CRYSTAL) & (beh.pmat == c))
            if elements.size == 0:
                continue
            n = cp.n_systems
            kinds, par = cp.kinds_array(), cp.params_array()
            first = CP_ALPHA + n  # the cumulated slips of the systems
            planes = range(cp.n_planes)
            opening = [
                int(np.flatnonzero((kinds[:, 3] == p) & (kinds[:, 0] == OPENING))[0])
                for p in planes
            ]
            modulus, length = cp.cleavage_regularization
            self.fields.setdefault(float(length), []).append(len(self.parts))
            self.parts.append(
                _Planes(
                    elements=elements,
                    points=(elements[:, None] * self.ng + np.arange(self.ng)).ravel(),
                    members=[first + np.flatnonzero(kinds[:, 3] == p) for p in planes],
                    slips=first + np.flatnonzero(kinds[:, 0] == SLIP),
                    slot=cp_width(n, solver.finite, cp.reorients),
                    softening=kinds[opening, 1],
                    strength=par[opening, 0],
                    modulus=float(modulus),
                    coupling=float(cp.cleavage_coupling),
                    broken=float(1.0 / kinds[opening, 1].max()),
                )
            )
        if self.parts and type(solver)._smooth is _PathSolver._smooth:
            solver._smooth(np.zeros(0, dtype=np.int64), np.zeros(0), 1.0)  # (refuses them)
        self.kept: list = []
        #: the equilibria solved, and the steps kept before the microdamage had settled
        self.passes = 0
        self.unsettled = 0
        self.accelerated = bool(solver.microdamage_acceleration)
        self._restart()

    def _restart(self) -> None:
        """The acceleration starts again, from the plain updates: no earlier pass (a new
        step, or planes broken through: another fixed point)."""
        self._last = None  # the microdamage of the last equilibrium and its residual
        self._norms: list[float] = []  # of the residuals of the passes
        self._growing = 0  # the passes in a row over which the residual has grown
        #: how far the microdamage of the equilibrium was extrapolated past its plain
        #: update, at the points of each part
        self._excess = [0.0] * len(self.parts)

    def __bool__(self) -> bool:
        return bool(self.parts)

    @staticmethod
    def _block(part: _Planes) -> slice:
        """The microdamage of the points of a part and the marks of their planes."""
        return slice(part.slot, part.slot + 1 + len(part.members))

    def begin(self, old) -> None:
        """The microdamage and the marks of the last converged step (those of the points
        ``old``), kept for the steps that are divided."""
        self.kept = [old[part.points, self._block(part)].copy() for part in self.parts]
        self._restart()

    def rollback(self, old) -> None:
        for part, kept in zip(self.parts, self.kept, strict=True):
            old[part.points, self._block(part)] = kept
        self._restart()

    @staticmethod
    def _damage(part: _Planes, state):
        """The damage of each plane of the points of a part in ``state`` (points x
        planes), and what the point gives to the microdamage: the damage of its planes,
        that which they had for those broken through."""
        block = state[part.points]
        own = np.stack([block[:, columns].sum(axis=1) for columns in part.members], axis=1)
        marks = block[:, part.slot + 1 : part.slot + 1 + len(part.members)]
        return own, np.where(marks > 0.5, marks - 1.0, own).sum(axis=1)

    def _strength(self, part: _Planes, state, source, micro):
        """The share of its strength that each plane of the points of a part keeps
        (points x planes) for the damage ``d`` of the planes of the points (``source``:
        what they give to the microdamage) and the microdamage ``micro``: ``fall - w d +
        chi_H (d - chi_d) / Y_0`` (``fall``: of the cumulated slip of ``state``)."""
        fall = 1.0
        if part.coupling > 0.0:
            slipped = state[part.points][:, part.slips].sum(axis=1)
            fall = np.exp(-part.coupling * slipped)[:, None]
        ahead = (part.modulus / part.strength) * (source - micro)[:, None]
        return fall - part.softening * source[:, None] + ahead

    def _break(self, part: _Planes, old, new, micro, excess=0.0) -> bool:
        """The planes that an equilibrium breaks through. Those that end it without
        strength, ``Y_0 (fall - w d) + chi_H (d - chi_d) <= 0`` at the damage reached
        (``new``) — for the microdamage of the equilibrium (that of ``old``: a plane that
        has softened to nothing, or one that a crack nearby has unloaded and left free
        to open) or for the one that follows it (``micro``: the microdamage takes what
        was left of its strength); nothing: what the tolerance of the microdamage
        resolves of a strength, ``chi_H tol / w``. And those that have opened with a
        damage further ahead of the microdamage than anywhere in the band that breaks,
        ``d - chi_d > (Y_0 / chi_H) / sqrt(1 - w Y_0 / chi_H)`` (the centre of the band
        as its stress comes to nothing): planes that a crack forces open beside those
        that it has broken — in the next element —, and which would harden against it,
        since the microdamage does not follow one element. No plane is broken through
        where the microdamage of the equilibrium had been extrapolated (``excess``: past
        its plain update) beyond that of the damage reached: it was ahead of the damage.

        These planes are marked in the points, for good, with the damage that they had
        then: that of the start of the step for a plane that had no strength from the
        start of the equilibrium to its end (what it has opened is the opening of a
        crack), that reached otherwise (less what it has opened beyond the bound)."""
        marks = slice(part.slot + 1, part.slot + 1 + len(part.members))
        own, source = self._damage(part, new)
        before, given = self._damage(part, old)
        last = old[part.points, part.slot]  # the microdamage of the equilibrium
        ended = self._strength(part, new, source, last)
        ratio = part.softening * part.strength / part.modulus
        lost = np.maximum(self.tol * part.modulus / (part.softening * part.strength), _BROKEN)
        breaks = np.minimum(ended, self._strength(part, new, source, micro)) <= lost
        beyond = (source - micro)[:, None] - part.strength / (part.modulus * np.sqrt(1.0 - ratio))
        forced = (beyond > 0.0) & (own > before)
        breaks = (breaks | forced) & (old[part.points, marks] < 0.5)
        slack = self.tol * part.broken
        ahead = (np.asarray(excess) > slack) & (last - micro > slack)
        breaks &= ~np.broadcast_to(ahead, last.shape)[:, None]
        if not breaks.any():
            return False
        started = self._strength(part, old, given, last)
        kept = np.where(forced, np.maximum(own - beyond, before), own)
        kept = np.where((started <= _BROKEN) & (ended <= lost), before, kept)
        marked = np.where(breaks, 1.0 + kept, old[part.points, marks])
        old[part.points, marks] = marked
        new[part.points, marks] = marked
        return True

    def _accelerate(self, current, target, weight):
        """The microdamage of the next equilibrium from that of the last one
        (``current``) and that of the damage that it reached (``target``, the plain
        update), for the elements of all the parts (``weight``: their volumes). After
        the scheme of Storvik et al. (2021) for the alternate minimization of a phase
        field, by the residual ``target - current`` of the fixed point: where it has
        decreased over two passes, Anderson's acceleration of depth one — the secant
        extrapolation of the last two updates, within :data:`_REACH` residuals of the
        plain one —, that of a fixed point close by; where it has grown over
        :data:`_GROWING` passes (a crack that starts to run: no fixed point close by), a
        relaxation of the update, :data:`_RELAXATION`; the plain update otherwise."""
        residual = target - current
        norm = math.sqrt(dot(weight, residual * residual))
        last, self._last = self._last, (current, residual)
        norms = self._norms
        norms.append(norm)
        if last is None or len(norms) < 2:
            return target
        if norm > norms[-2]:  # the residual grows
            self._growing += 1
            if self._growing >= _GROWING:
                return current + _RELAXATION * residual
            return target
        self._growing = 0
        if len(norms) < 3 or norms[-2] > norms[-3]:
            return target
        change = residual - last[1]
        scale = dot(weight, change * change)
        if scale <= 0.0:
            return target
        gamma = dot(weight, change * residual) / scale
        step = -gamma * (current - last[0] + change)  # past the plain update
        reach = _REACH * np.abs(residual)
        return target + np.clip(step, -reach, reach)

    def settle(self, old, new, wts, last: bool = False) -> float:
        """The microdamage of the damage reached (``new``), given to the points of the
        step (``old``, which the kernels read, and ``new``): its largest change, over the
        damage that breaks a plane (more than any tolerance where it breaks planes: the
        step is then solved again). Unless the microdamage has settled, or the pass is
        the ``last`` one of the step, the points are given an accelerated update
        (:meth:`_accelerate`)."""
        change = 0.0
        ng = self.ng
        broke = False
        found = [None] * len(self.parts)  # the microdamage of the damage reached (elements)
        for length, members in self.fields.items():
            means = []
            for k in members:
                part = self.parts[k]
                shape = (part.elements.size, ng)
                source = self._damage(part, new)[1].reshape(shape)
                weights = np.abs(np.asarray(wts)[part.points]).reshape(shape)
                means.append((weights * source).sum(axis=1) / weights.sum(axis=1))
            elements = np.concatenate([self.parts[k].elements for k in members])
            smooth = self.solver._smooth(elements, np.concatenate(means), length)
            start = 0
            for k in members:
                part = self.parts[k]
                found[k] = smooth[start : start + part.elements.size]
                start += part.elements.size
                micro = np.repeat(found[k], ng)
                moved = float(np.abs(micro - old[part.points, part.slot]).max())
                change = max(change, moved / part.broken)
                broke |= self._break(part, old, new, micro, self._excess[k])
        self.passes += 1
        update = found
        self._excess = [0.0] * len(self.parts)
        if broke:  # another fixed point: the step again, from the plain update
            change = max(change, 1.0)
            self._restart()
        elif self.accelerated and not last and change > self.tol:
            sizes = np.cumsum([part.elements.size for part in self.parts])[:-1]
            current = np.concatenate([old[p.points, p.slot][::ng] for p in self.parts])
            volumes = np.concatenate(
                [np.abs(np.asarray(wts)[p.points]).reshape(-1, ng).sum(axis=1) for p in self.parts]
            )
            update = np.split(self._accelerate(current, np.concatenate(found), volumes), sizes)
            self._excess = [np.repeat(u - f, ng) for u, f in zip(update, found, strict=True)]
        for part, values in zip(self.parts, update, strict=True):
            micro = np.repeat(values, ng)
            old[part.points, part.slot] = micro
            new[part.points, part.slot] = micro
        return change


# -- driver ------------------------------------------------------------------------------


class _Steps:
    """Time stepping along a path: the planned steps, divided by two after a failure
    (``level`` divisions) and grown back after two converged steps."""

    def __init__(self, path: LoadingPath, max_cuts: int):
        self.ends = path.schedule()
        self.start = float(path.times[0])
        self.segment_ends = np.asarray(path.times[1:])
        self.max_cuts = int(max_cuts)
        self.k = 0
        self.t = self.start
        self.level = 0
        self.successes = 0
        self.cuts = 0

    @property
    def done(self) -> bool:
        return self.k >= len(self.ends)

    def _planned(self) -> float:
        """Length of the current planned step."""
        start = self.ends[self.k - 1] if self.k else self.start
        return float(self.ends[self.k] - start)

    def next(self) -> float:
        """End of the next attempted step (the end of the planned step, rather than a
        sliver before it)."""
        end = float(self.ends[self.k])
        dt = self._planned() / 2.0**self.level
        t_try = self.t + dt
        return end if t_try >= end - 1e-9 * dt else t_try

    def accept(self, t_new: float) -> None:
        self.t = t_new
        if t_new == self.ends[self.k]:
            self.k += 1
        self.successes += 1
        if self.successes >= 2 and self.level > 0:
            self.level -= 1
            self.successes = 0

    def reject(self) -> None:
        self.level += 1
        self.successes = 0
        self.cuts += 1
        if self.level > self.max_cuts:
            raise RuntimeError(
                f"nonlinear solution: no convergence at time {self.t:.6g} after dividing the "
                f"step {self.max_cuts} times"
            )

    def segment(self, t: float) -> int:
        return int(np.searchsorted(self.segment_ends, t - 1e-12 * max(1.0, abs(t))))


#: Halvings of the Newton corrections in the line search
_MAX_HALVINGS = 6


def _line_search(evaluate, x, unknown, dx, merit0: float, merit):
    """Backtracking line search along the Newton correction ``dx`` of the components
    ``unknown`` of ``x``: the first of the steps 1, 1/2, ... that decreases the merit
    (Armijo), or the shortest one if it decreases it at all. Returns the new point and its
    evaluation (None when the evaluation fails or the merit does not decrease).

    ``merit0`` is the merit at ``x``, or, for a nonmonotone search (Grippo, Lampariello &
    Lucidi 1986), the largest merit of the last iterates: the merit may then rise from
    ``x``, below that largest value. The residual of unilateral contacts is not smooth:
    where the contacts change along the correction, it rises along a direction that is
    the right one — that of the solution of the contacts as they are —, and the next
    linearization, with the contacts as they have become, corrects it (an active-set
    iteration, which the short steps that decrease the residual would let change its set
    only little by little); the bound keeps the iteration from cycling."""
    t = 1.0
    for k in range(_MAX_HALVINGS + 1):
        x_t = x.copy()
        x_t[unknown] += t * dx
        out = evaluate(x_t)
        if out is not None:
            m = merit(out)
            if m <= (1.0 - 1e-4 * t) * merit0 or (k == _MAX_HALVINGS and m < merit0):
                return x_t, out
        t *= 0.5
    return x, None


def _point_merit(out) -> float:
    return dot(out[2], out[2])


def _fe_merit(out) -> float:
    """Merit of the residuals of a trial state: the equilibrium relative to the nodal
    forces, the macroscopic stress relative to the stresses."""
    r_q, r_e, fscale, escale = out
    # squares bounded below, not the scales: (1e-300)^2 underflows to zero
    return dot(r_q, r_q) / max(fscale * fscale, 1e-300) + dot(r_e, r_e) / max(
        escale * escale, 1e-300
    )


def _split(evaluate, n_q, temp, dt, values, x):
    """Evaluation of a trial state ``x = (q, E)`` (the slips from the current iterate)."""
    return evaluate(x[:n_q], x[n_q:], temp, dt, values, 1, 1.0)


def _predicted(dt: float, last, before=None):
    """Increments of the unknowns ``(q, E)`` over a step ``dt``, predicted from the last
    converged step ``last = (segment, dt, dq, dE)``: its rate (first order); with the
    step ``before`` it, of the same segment, its rate and the change of the rates of the
    two steps, which are those of their middles (second order: the unknowns extrapolated
    as a parabola through the ends of the two steps)."""
    ratio = dt / last[1]
    dq, de = ratio * last[2], ratio * last[3]
    if before is not None and before[0] == last[0]:
        c = dt * (dt + last[1]) / (last[1] + before[1])
        dq = dq + c * (last[2] / last[1] - before[2] / before[1])
        de = de + c * (last[3] / last[1] - before[3] / before[1])
    return dq, de


def _second_order(step, last, before) -> bool:
    """Whether the next step is predicted at the second order: where the second-order
    prediction (:func:`_predicted`) of the converged ``step`` from the two steps before
    it was clearly closer to it than the first-order one (:data:`_SECOND`: half its
    error, the errors of the fluctuation and of the macroscopic unknowns each relative
    to its increment), and the first-order one was not far from it (the same share of
    the increments). A response that turns (a yield, a crack that starts, a contact
    that closes, twins that start) is predicted at the first order: the second one
    would overshoot, and the step be divided."""
    if last is None or before is None or not (step[0] == last[0] == before[0]):
        return False

    def error(prediction) -> float:
        total = 0.0
        for found, guess in zip(step[2:], prediction, strict=True):
            scale = norm(found)
            if scale > 0.0:
                total += norm(found - guess) / scale
        return total

    first = error(_predicted(step[1], last))
    return first < _SECOND and error(_predicted(step[1], last, before)) < _SECOND * first


def _secant(schur, step, before, after, symmetric: bool):
    """The condensed block of the macroscopic unknowns of an earlier linearization (their
    Schur complement, kept from a correction to the next), updated by the secant of a
    correction: their ``step`` and their condensed residuals ``before`` and ``after``
    it. The BFGS update of a symmetric block with Powell's damping (1978): the curvature
    of the secant along the step is kept between a fifth of and five times that of the
    block (:data:`_DAMPING`), so that the block stays positive definite and one update
    changes it by a bounded factor. Broyden's update otherwise, its change bounded by
    the image of the step likewise."""
    change = after - before
    image = schur @ step
    if symmetric:
        energy = float(step @ image)
        if energy <= 0.0:
            return schur
        curvature = float(change @ step)
        low, high = _DAMPING * energy, energy / _DAMPING
        if not low <= curvature <= high:  # (the secant mixed with the image of the block)
            bound = low if curvature < low else high
            share = (bound - energy) / (curvature - energy)
            change = share * change + (1.0 - share) * image
            curvature = bound
        return schur - np.outer(image, image) / energy + np.outer(change, change) / curvature
    length = float(step @ step)
    if length == 0.0:
        return schur
    defect = change - image
    size, most = float(np.linalg.norm(defect)), float(np.linalg.norm(image)) / _DAMPING
    if size > most > 0.0:
        defect = defect * (most / size)
    return schur + np.outer(defect, step) / length


@dataclass
class StepState:
    """A converged step of a loading path, given to the ``monitor`` of
    :meth:`_PathSolver.run`: its time, the macroscopic strain (6), the unknowns of the
    fluctuation, the indices of the macroscopic components solved for, and the
    discretization with the arrays of its integration points (measures of strain,
    weights, stresses and consistent tangents at the end of the step)."""

    time: float
    strain: np.ndarray
    q: np.ndarray
    unknown: np.ndarray
    disc: object
    grads: np.ndarray
    wts: np.ndarray
    sigma: np.ndarray
    tangent: np.ndarray


def _macro_stress(finite: bool, stress, tangent):
    """Macroscopic stress (6) of a point and its tangent with respect to the macroscopic
    strain (6 x 6): the nominal stress in finite strains."""
    if not finite:
        return stress, tangent
    return fin.macro_part(stress), _MACRO.T @ tangent @ _MACRO


def simulate_point(
    material: Material,
    path,
    *,
    name: str = "point",
    orientation=None,
    kinematics: str = "small",
    tol: float = 1e-10,
    max_newton: int = 30,
    max_cuts: int = 12,
    tangent: str = "auto",
) -> PathResult:
    """Response of one material point (homogeneous strain and stress) to a loading
    path: the components prescribed as stresses are solved for by Newton's method with
    the consistent tangent (a material point simulator, as ``Zsim``). ``orientation``:
    Bunge angles (degrees) of a crystal; ``kinematics``: ``"small"`` or ``"finite"``
    strains (the strain then holds the components of the upper triangular displacement
    gradient, the stress those of the first Piola-Kirchhoff stress). In finite strains,
    and where its twins turn it, the result has the ``orientation`` of the lattice of a
    crystal at each step: its Bunge angles ``g R_e^T``, with the rotation ``R_e`` of the
    polar decomposition of ``F_e`` (the material lines along the first axis stay along
    it, and the planes normal to the third stay normal to it: the grips of a tension
    along the first axis, the platens of a compression along the third). ``tangent``
    (:data:`TANGENTS`): Newton's method solves a dense system, with by default
    (``"auto"``) the exact tangents of the laws that are not symmetric (those of the learned
    laws and of the MFront behaviours too) and the symmetrized ones of the others (which
    only lose their round-off); ``"symmetric"`` symmetrizes them all, ``"full"`` keeps them
    all exact."""
    if tangent not in TANGENTS:
        raise ValueError(f"tangent: one of {TANGENTS}, got {tangent!r}")
    path = LoadingPath.from_spec(path)
    finite = kinematics_of(kinematics)
    if material.void:
        raise ValueError("simulate_point: the material is a void")
    mats = {0: material}
    if orientation is None:
        table = material_table([0], mats, "elastic")
    else:
        euler = np.asarray(orientation, dtype=float).reshape(1, 3)
        table = material_table([0], mats, "elastic", grains=[0], orientations=euler)
    beh = behaviour_table([0], mats, table, finite=finite)
    symmetric = tangent == "symmetric" or (tangent == "auto" and beh.symmetric)
    points = external_points(beh.mfront, beh.kind, beh.pmat, table.mat, table.rotations, finite,
                             symmetric=tangent == "symmetric")  # fmt: skip
    nc = 9 if finite else 6
    old, new = np.zeros((1, beh.width)), np.zeros((1, beh.width))
    stress, tangents = np.zeros((1, nc)), np.zeros((1, nc, nc))
    unknown = np.flatnonzero(~path.strain_controlled)
    controls = path.strain_controlled
    strain = np.zeros(6)
    rec = {"t": [path.times[0]], "e": [strain.copy()], "s": [np.zeros(6)], "T": [0.0], "p": [0.0]}
    rec.update({"cauchy": [np.zeros(6)], "full": [np.zeros(nc)]})
    porous = bool(beh.kind[0] == POROUS)
    rec["f"] = [float(beh.data[2][beh.pmat[0], P_F0])] if porous else []
    # the twins and the cleavage planes of a crystal: the curves of their fields
    crystal = beh.crystals[0] if beh.crystals and beh.crystals[0].extended else None
    if crystal is not None:
        rec["crystal"] = {name: [0.0] for name in crystal.field_names}
        if crystal.regularized:
            # a point alone has no gradient: its microdamage is its damage (a negative
            # value tells its kernel), and it softens with it, as fast as it may
            crystal.check_softening(material.matrix("elastic"), force=True)
            slot = cp_width(crystal.n_systems, finite, crystal.reorients)
            old[0, slot] = new[0, slot] = -1.0
    # the lattice of a crystal, where it turns: its rotation in finite strains, its twins
    lattice = beh.crystals[beh.pmat[0]] if beh.kind[0] == CRYSTAL else None
    if lattice is not None and not (finite or lattice.reorients):
        lattice = None
    if lattice is not None:
        from morphogen.orientations import matrix_to_euler

        initial = table.rotations[table.mat[0]]
        rec["euler"] = [matrix_to_euler(initial)[0]]
    iterations = []
    steps = _Steps(path, max_cuts)
    history = None  # (segment, dt, increment of the unknowns) of the last converged step

    def evaluate(e, dt, temp, values, guess, ratio):
        """Macroscopic stress, tangent and residual of a trial strain (None on failure)."""
        grads = fin.gradient_of(e)[None] if finite else e[None]
        failed = _update(finite, beh.kind, table.mat, beh.pmat, beh.data, grads, temp, dt, old,
                         new, stress, tangents, guess, ratio, symmetric)  # fmt: skip
        if points is not None:
            failed += points.integrate(grads, temp, dt, old, new, stress, tangents)
        if failed:
            return None
        s6, k6 = _macro_stress(finite, stress[0], tangents[0])
        return s6, k6, s6[unknown] - values[unknown]

    t0 = time.perf_counter()
    while not steps.done:
        t_new = steps.next()
        dt = t_new - steps.t
        values, temp = path.at(t_new)
        seg = steps.segment(t_new)
        e = strain.copy()
        e[controls] = values[controls]
        same = history is not None and history[0] == seg
        if same:  # extrapolation of the unknown components
            e[unknown] += dt / history[1] * history[2]
        res = evaluate(e, dt, temp, values, 2 if same else 0, dt / history[1] if same else 1.0)
        ok, its = False, 0
        for its in range(max_newton):  # noqa: B007
            if res is None:
                break
            s6, k6, r = res
            scale = float(np.abs(s6).max() + np.abs(values[unknown]).max(initial=0.0))
            # the round-off of the stresses of the strain: reached by a solution at zero
            # stress (a free thermal expansion), where the relative tolerance asks for zeros;
            # in finite strains that of the gradient I + H, which does not vanish with H (a
            # crack that has closed again at no strain: no stress, but for the round-off)
            stiff = float(np.abs(k6).max())
            floor = _ROUND_OFF * stiff * float(np.abs(e).max())
            if finite:
                floor += _ROUND_OFF_UNIT * stiff
            if np.all(np.abs(r) <= max(tol * scale, floor) + 1e-300):
                ok = True
                break
            de = -np.linalg.solve(k6[np.ix_(unknown, unknown)], r)
            trial = partial(evaluate, dt=dt, temp=temp, values=values, guess=1, ratio=1.0)
            e, res = _line_search(trial, e, unknown, de, float(r @ r), _point_merit)
        if not ok:
            steps.reject()
            history = None
            continue
        iterations.append(its)
        steps.accept(t_new)
        history = (seg, dt, e[unknown] - strain[unknown])
        strain = e
        old, new = new, old
        if crystal is not None and crystal.by_grain:  # the lattice that its twins turn
            turn_twins(crystal, old, new, table.mat, beh.data[4], finite=finite)
        rec["t"].append(t_new)
        rec["e"].append(strain.copy())
        rec["s"].append(s6.copy())
        rec["full"].append(stress[0].copy())
        rec["T"].append(temp)
        rec["p"].append(float(old[0, STATE_P]))
        if porous:
            rec["f"].append(float(porosity(old, beh.data[2], beh.pmat)[0]))
        if crystal is not None:
            turned = crystal.state_fields(old, finite, table.rotations[table.mat])
            for name, values in turned.items():
                rec["crystal"][name].append(float(values[0]))
        if lattice is not None:
            grads = fin.gradient_of(strain)[None] if finite else strain[None]
            turn = _lattice_turns(lattice, finite, grads, old)[0]
            rec["euler"].append(matrix_to_euler(initial @ turn.T)[0])
    p = np.array(rec["p"])
    extra = {"porosity": np.array(rec["f"])} if porous else {}
    if lattice is not None:
        extra["orientation"] = np.array(rec["euler"])
    if crystal is not None:
        extra["series"] = {name: np.array(values) for name, values in rec["crystal"].items()}
    if finite:
        grads = fin.gradient_of(np.array(rec["e"]))
        extra["cauchy"] = fin.cauchy(np.array(rec["full"]), grads)
        extra["log_strain"] = fin.log_strain(grads)
    return PathResult(
        name=name,
        bc="point",
        path=path,
        times=np.array(rec["t"]),
        strain=np.array(rec["e"]),
        stress=np.array(rec["s"]),
        temperature=np.array(rec["T"]),
        plastic=p,
        iterations=np.array(iterations, dtype=int),
        info={"cuts": steps.cuts, "time_s": round(time.perf_counter() - t0, 4)},
        kinematics=kinematics,
        **extra,
    )


class _PathSolver:
    """Loading paths of the periodic element: the time stepping and Newton's method
    shared by the finite element solver (:class:`NonlinearFEProblem`) and the FFT solver
    of voxel images (:class:`morphogen.homogenization.fft_nonlinear.NonlinearFFTProblem`).

    The solvers give the behaviours of their integration points
    (:meth:`_point_behaviours`), the options of the iterations (:meth:`_path_options`)
    and their discretization (:meth:`_discretization`), an object with:

    * ``n_q``: the number of unknowns of the fluctuation;
    * ``measures(q, e, grads, wts)``: the measures of strain of the integration points
      (and their weights) for the fluctuation ``q`` and the macroscopic strain ``e``;
    * ``forces(sigma, wts)``: the residual of the equilibrium on the unknowns, the
      integral of the stress and the norm of the nodal forces before they cancel (it may
      replace the stresses of the points by those conjugate to its measures of strain,
      as B-bar and F-bar do; the outputs then report them);
    * ``linearize(tangent, wts, first, symmetric=None)``: the solver of the tangent
      operator (``first``: the first iteration of a step), with ``solve(b, x0=None,
      tol=None)`` and ``method``: conjugate gradients (or LU) on symmetrized tangents
      (``symmetric``, by default :attr:`symmetrize`), GMRES (or LU) on full ones;
    * ``iterative``: whether that solver is a Krylov method, ``asymmetry(tangent, wts)``
      the asymmetry of the tangents of the points (:func:`asymmetry`) and
      ``symmetrize(tangent)`` their symmetrization in place, for the tangents that
      :attr:`measured` decides;
    * ``coupling(tangent, wts, unknown)``: the columns ``unknown`` of ``K_wE`` on the
      unknowns, and ``K_EE`` (6 x 6);
    * ``macro_response(tangent, wts, v)``: ``K_Ew v``, the integral of the stress
      increments of the fluctuation ``v`` (6), which is ``K_wE^T v`` for symmetric
      tangents;
    * ``displacement(q, e)``: the displacement field of the outputs;
    * the arrays of the points and the integration of their behaviours, as
      :class:`_PointArrays` provides them on the CPU.
    """

    #: Largest order of the prediction of a step from the steps before it (the starting
    #: point of its Newton iterations): 1, the rate of the last step; 2, that rate and its
    #: change over the last two steps, where it would have predicted the last step better
    #: (:func:`_second_order`)
    prediction = 2
    #: The microdamage of regularized cleavage planes (:class:`_Microdamage`): the largest
    #: change, over the damage that breaks a plane, at which it has settled, and the
    #: most passes of a step (beyond, the step is kept as it is and the microdamage goes on
    #: with the next one: a crack that runs across the cell at a fixed loading takes some
    #: passes per element that it crosses)
    microdamage_tol = 1e-3
    microdamage_passes = 200
    #: Whether the alternate scheme of the microdamage is accelerated: its updates
    #: extrapolated or relaxed (:meth:`_Microdamage._accelerate`), and the equilibria
    #: between them solved to a share of its last change (:data:`_SHARE`), the last one of
    #: a step to the tolerance of Newton's iteration
    microdamage_acceleration = True

    def microdamage_options(self, tolerance=None, passes=None, acceleration=None) -> None:
        """The options of the alternate scheme of the microdamage of regularized
        cleavage planes (those left as they are when not given): the ``tolerance`` on
        the change of the microdamage between two equilibria of a step, over the damage
        that breaks a plane (1e-3); the most ``passes`` of a step (200); whether the
        scheme is accelerated (true)."""
        if tolerance is not None:
            if isinstance(tolerance, bool) or not 0.0 < float(tolerance) < 1.0:
                raise ValueError("microdamage: a tolerance in (0, 1)")
            self.microdamage_tol = float(tolerance)
        if passes is not None:
            if isinstance(passes, bool) or int(passes) != passes or passes < 1:
                raise ValueError("microdamage: a positive number of passes")
            self.microdamage_passes = int(passes)
        if acceleration is not None:
            if not isinstance(acceleration, bool):
                raise ValueError("microdamage: acceleration, true or false")
            self.microdamage_acceleration = acceleration

    def _smooth(self, elements, values, length) -> np.ndarray:
        """The solution of ``chi - length^2 lap chi = values`` at the ``elements`` (the
        equation of the microdamage of cleavage planes): the solvers of voxel images and
        of meshes have it."""
        raise ValueError(
            "the microdamage of cleavage planes (cleavage.regularization): with the solvers "
            "of the loading paths on voxel images (FFT) and on meshes (FE)"
        )

    def _path_options(self, newton_tol, max_newton, max_cuts, kinematics,
                      tangent: str = "auto") -> None:  # fmt: skip
        if not 0.0 < newton_tol < 1.0:
            raise ValueError("newton_tol: a number in (0, 1)")
        if tangent not in TANGENTS:
            raise ValueError(f"tangent: one of {TANGENTS}, got {tangent!r}")
        self.finite = kinematics_of(kinematics)
        self.kinematics = kinematics
        self.newton_tol = float(newton_tol)
        self.max_newton = int(max_newton)
        self.max_cuts = int(max_cuts)
        self.tangent = tangent
        #: Whether the consistent tangents of the points are symmetrized (conjugate
        #: gradients), or kept full (GMRES): :data:`TANGENTS`, ``auto`` decided by the laws
        #: of the phases (:attr:`Behaviours.symmetric`), and then by the asymmetry of their
        #: tangents (:attr:`measured`)
        self.symmetrize = tangent != "full"
        self.measured = False

    def _point_behaviours(self, beh: Behaviours, mat, group, ng: int) -> None:
        """Behaviours of the integration points: ``ng`` per element, the elements of
        materials ``mat`` and of groups ``group`` (indices of :attr:`groups`)."""
        self.behaviours = beh
        self.symmetrize = self.tangent == "symmetric" or (self.tangent == "auto" and beh.symmetric)
        #: Whether the Krylov solvers decide at each linearization from the measured
        #: asymmetry of the full tangents of the points (:data:`ASYMMETRY`): ``auto`` with
        #: laws that are not all symmetric
        self.measured = self.tangent == "auto" and not beh.symmetric
        self.ng = int(ng)
        self.kind = np.ascontiguousarray(np.repeat(beh.kind, ng))
        self.pmat = np.ascontiguousarray(np.repeat(beh.pmat, ng))
        self.point_mat = np.ascontiguousarray(np.repeat(mat, ng))
        self.point_group = np.ascontiguousarray(np.repeat(group, ng))

    def _discretization(self):
        raise NotImplementedError

    def _support(self):
        """The support of the fields of the outputs (mesh or image)."""
        raise NotImplementedError

    @without_blas_threads
    def run(self, path, *, name: str = "path", fields=False, monitor=None) -> PathResult:
        """Response of the volume element to a loading path (:class:`LoadingPath`, or a
        mapping for :meth:`LoadingPath.from_spec`). ``fields``: true for the element
        fields at the end of each segment of the path, ``"steps"`` at the end of each of
        its planned steps, or a list of such times. ``monitor``: a function called with the
        :class:`StepState` of every converged step (the stability analyses of
        :func:`morphogen.homogenization.bifurcation.path_bifurcation`); a true return
        value ends the path there."""
        path = LoadingPath.from_spec(path)
        t_start = time.perf_counter()
        if self.empty:
            return self._empty_path(path, name)
        finite = self.finite
        beh = self.behaviours
        disc = self._discretization()
        n_pts = self.kind.size
        nc = 9 if finite else 6
        grads, wts, sigma, tangent, old, new = disc.arrays(n_pts, nc, beh.width)
        controls = path.strain_controlled
        unknown = np.flatnonzero(~controls)
        volume = self.box_volume
        n_groups = len(self.groups)
        field_times = self._field_times(path, fields)
        # the crystals whose cleavage planes are regularized by their microdamage: the
        # step is solved again with the microdamage of the damage reached, until it settles
        micro = _Microdamage(self) or None
        unloaded = _UNLOADED if micro is None else _CUT

        def evaluate(q, e, dtemp, dt, target, guess, ratio):
            """Residuals of a trial state (None if the integration fails)."""
            disc.measures(q, e, grads, wts)
            failed = disc.integrate(grads, dtemp, dt, old, new, sigma, tangent, guess, ratio)
            if failed:
                state["why"] = f"the integration failed at {failed} points"
                return None
            r_q, total, fscale = disc.forces(sigma, wts)
            if finite:
                total = fin.macro_part(total)
            r_e = total[unknown] - volume * target[unknown]
            escale = disc.stress_scale(sigma, wts)
            escale += volume * float(np.abs(target[unknown]).max(initial=0.0))
            # an unloaded state keeps a share of the largest scales reached before it
            fscale = max(fscale, unloaded * state["fmax"])
            escale = max(escale, unloaded * state["emax"])
            return r_q, r_e, fscale, escale

        state = {"y": None, "schur": None, "stale": True, "lin": 0, "why": "",
                 "asymmetry": 0.0, "full": 0, "stalled": 0, "fmax": 0.0, "emax": 0.0,
                 "secant": None, "rho": None}  # fmt: skip
        measured = self.measured and disc.iterative

        def correction(r_q, r_e, first: bool, refresh: bool, eta: float):
            """Newton correction of the fluctuation and of the unknown components of E,
            the equilibrium solved to the relative residual ``eta``. The macroscopic block
            is condensed with ``Y = K_ww^-1 K_wE``, computed again (from the previous
            ``Y``) when ``refresh``, and kept otherwise: a quasi-Newton iteration for the
            macroscopic unknowns. ``K_Ew`` is ``K_wE^T`` with symmetrized tangents, the
            integral of the stress increments of the fluctuations with full ones. Measured
            tangents (:attr:`measured`) less asymmetric than :data:`ASYMMETRY` are
            symmetrized here."""
            symmetric = self.symmetrize
            if measured:
                value = disc.asymmetry(tangent, wts)
                state["asymmetry"] = max(state["asymmetry"], value)
                symmetric = value < ASYMMETRY
                if symmetric:
                    disc.symmetrize(tangent)
                else:
                    state["full"] += 1
            solver = disc.linearize(tangent, wts, first, symmetric)
            y0, info = solver.solve(-r_q, tol=eta)
            state["lin"] += info.iterations
            if not symmetric and disc.iterative:  # GMRES: the corrections that stagnate
                state["stalled"] = state["stalled"] + 1 if info.residual > eta else 0
                if state["stalled"] >= _STALLED:
                    state["why"] = f"GMRES stagnated at {_STALLED} iterations in a row"
                    return None, None
            if unknown.size == 0:
                return y0, np.zeros(0)
            g, kee = disc.coupling(tangent, wts, unknown)

            def k_ew(v):
                if symmetric:
                    return times(g.T, v)
                return disc.macro_response(tangent, wts, v)[unknown]

            renewed = refresh or state["schur"] is None or solver.method == "direct"
            if renewed:
                previous = state["y"]
                ys = []
                coarse = max(self.tol, _MACRO_TOL)
                if micro is not None:  # between its updates: as loosely as the equilibrium
                    coarse = max(coarse, min(_COARSE, 10.0 * slack))
                for j in range(unknown.size):
                    x0 = None if previous is None else previous[:, j]
                    y, info = solver.solve(g[:, j], x0, tol=coarse)
                    state["lin"] += info.iterations
                    ys.append(y)
                state["y"] = np.column_stack(ys)
                state["stale"] = False
            if renewed or micro is not None:
                # (with a microdamage the tangents change at every pass: the condensed
                # block is that of the tangents as they are, for the ``Y`` kept)
                if symmetric:
                    coupled = times(g.T, state["y"])
                else:
                    coupled = np.column_stack([k_ew(y) for y in state["y"].T])
                state["schur"] = kee[np.ix_(unknown, unknown)] - coupled
            rho = r_e + k_ew(y0)  # the condensed residual of the macroscopic unknowns
            if state["secant"] is not None and not renewed and micro is None:
                # the block kept from an earlier linearization: updated by the secant of
                # the last correction (the tangents have changed since it was computed)
                state["schur"] = _secant(state["schur"], *state["secant"], rho, symmetric)
            state["rho"] = rho
            de = np.linalg.solve(state["schur"], -rho)
            return y0 - times(state["y"], de), de

        slack = self.newton_tol  # (the tolerance of the equilibrium being solved)
        q = np.zeros(disc.n_q)
        e = np.zeros(6)
        n_q = q.size
        moving = np.concatenate([np.arange(n_q), n_q + unknown])  # the unknowns in (q, E)
        rec = {"t": [], "e": [], "s": [], "T": [], "p": [], "groups": [], "fields": {}}
        rec.update({"cauchy": [], "log": [], "f": [], "crystals": {}})
        self._record(rec, path.times[0], e, 0.0, None, None, None, None, n_groups, first=True)
        self._record_porosity(rec, None, None, n_groups)
        self._record_crystals(rec, None, None)
        steps = _Steps(path, self.max_cuts)
        iterations: list[int] = []
        history = None  # (segment, dt, dq, de) of the last converged step
        earlier = None  # of the step before it
        second = False  # whether the next step is predicted at the second order
        # the hooks of the problems whose own state follows the solution: ``_settle(state)``
        # updates it on a converged state and tells whether to solve the step again;
        # ``_rollback()`` restores it when a step is divided
        settle = getattr(self, "_settle", None)
        rollback = getattr(self, "_rollback", None)
        # the crystals whose lattices are turned by their twins grain by grain, after a step
        turning = any(cp.by_grain for cp in beh.crystals)
        scales = [0.0, 0.0]

        def merit(out) -> float:
            """The merit of the residuals of a trial state, with scales no smaller than
            those kept (the largest ones of the iterates of a step where planes break:
            the stresses of a cell that a crack cuts fall within one correction, and its
            residuals with them — relative to the scales of the trial state, they would
            not decrease along the correction that brings it there)."""
            r_q, r_e, fscale, escale = out
            return _fe_merit((r_q, r_e, max(fscale, scales[0]), max(escale, scales[1])))

        while not steps.done:
            t_new = steps.next()
            dt = t_new - steps.t
            values, temp = path.at(t_new)
            seg = steps.segment(t_new)
            q_try = q.copy()
            e_try = e.copy()
            e_try[controls] = values[controls]
            same = history is not None and history[0] == seg
            if same:
                dq, de = _predicted(dt, history, earlier if second else None)
                q_try += dq
                e_try[unknown] += de
            guess, gratio = (2, dt / history[1]) if same else (0, 1.0)
            state["why"], state["stalled"] = "", 0
            total = 0  # the Newton iterations of the step, over its settlings
            # the merits of the last iterates: one for a monotone line search, several
            # (``_window``) for the problems whose residual is not smooth (interfaces)
            window = getattr(self, "_window", 1)
            if micro is not None:
                # (a plane that breaks is a kink of the residual: as the interfaces)
                window = max(window, _MICRO_WINDOW)
                micro.begin(old)
            merits = deque(maxlen=window)
            scales[0] = scales[1] = 0.0
            passes = 0
            # the tolerance of the equilibria of the step: that of Newton's iteration, or a
            # looser one between the updates of a microdamage that has not settled
            slack = self.newton_tol
            closing = False  # the last equilibrium of a step whose passes are spent
            while True:
                state["secant"] = None  # (no secant across two equilibria)
                res = evaluate(q_try, e_try, temp, dt, values, guess, gratio)
                if micro is not None:  # its passes after the first: from the state reached
                    guess, gratio = 1, 1.0
                ok, its = False, 0
                for its in range(self.max_newton + 1):
                    if res is None:
                        break
                    r_q, r_e, fscale, escale = res
                    if micro is not None:
                        scales[0], scales[1] = max(scales[0], fscale), max(scales[1], escale)
                    # (a pass makes one correction at least, unless it has converged)
                    tol = slack if its else self.newton_tol
                    if norm(r_q) <= tol * fscale and np.all(np.abs(r_e) <= tol * escale):
                        ok = True
                        break
                    if its == self.max_newton:
                        break
                    # inexact Newton: the error of the correction below the expected residual
                    ratio_r = norm(r_q) / max(fscale, 1e-300)
                    eta = min(
                        _MAX_FORCING, max(self.tol, ratio_r, 0.1 * slack / max(ratio_r, 1e-300))
                    )
                    refresh = (its == 0 and state["stale"]) or its == _REFRESH_AFTER
                    current = merit(res)
                    if micro is not None and its and current > _POOR * merits[-1]:
                        # (a plane that breaks changes the tangents of its cell wholesale)
                        refresh = True
                    dq, de = correction(r_q, r_e, first=its == 0, refresh=refresh, eta=eta)
                    if dq is None or not (np.all(np.isfinite(dq)) and np.all(np.isfinite(de))):
                        break
                    trial = partial(_split, evaluate, n_q, temp, dt, values)
                    x0, dx = np.concatenate([q_try, e_try]), np.concatenate([dq, de])
                    merits.append(current)
                    x, found = _line_search(trial, x0, moving, dx, max(merits), merit)
                    if found is None and micro is not None and not refresh and unknown.size:
                        # a correction of no use with the condensation kept from an
                        # earlier pass: again, with that of the tangents as they are
                        state["why"] = ""
                        dq, de = correction(r_q, r_e, first=False, refresh=True, eta=eta)
                        if dq is not None and np.all(np.isfinite(dq)) and np.all(np.isfinite(de)):
                            dx = np.concatenate([dq, de])
                            x, found = _line_search(trial, x0, moving, dx, max(merits), merit)
                    res = found
                    if res is None and not state["why"]:
                        state["why"] = "no decrease of the residual along the correction"
                    # the step of the macroscopic unknowns and their condensed residual
                    # before it: the secant of the next correction, if this one converges
                    state["secant"] = None
                    if (found is not None and unknown.size and state["rho"] is not None
                            and merit(found) <= _TRUSTED * current):  # fmt: skip
                        state["secant"] = (x[n_q:][unknown] - e_try[unknown], state["rho"])
                    q_try, e_try = x[:n_q], x[n_q:]
                total += its
                if ok and micro is not None and not closing:
                    passes += 1
                    change = micro.settle(old, new, wts, passes >= micro.most)
                    # (the condensation is kept from a pass to the next: a poor
                    # correction, or one of no use, computes it again)
                    if change > micro.tol and passes < micro.most:
                        if micro.accelerated:
                            slack = min(_LOOSE, max(self.newton_tol, _SHARE * change))
                        continue
                    if change > micro.tol:
                        # the passes of the step are spent (a crack that runs across the
                        # cell at a fixed loading): its last equilibrium is kept, and the
                        # microdamage goes on with the next step
                        micro.unsettled += 1
                        closing = True
                        log.info(
                            "step to t = %.6g: the microdamage has not settled in %d passes "
                            "(it goes on with the next step)",
                            t_new,
                            passes,
                        )
                    if slack > self.newton_tol:  # the last equilibrium: to the tolerance
                        slack = self.newton_tol
                        continue
                # a problem whose own state follows the solution (the damage of a phase
                # field) settles it on the state reached: the step is then solved again
                if not ok or settle is None:
                    break
                if not settle(StepState(t_new, e_try.copy(), q_try.copy(), unknown, disc,
                                        grads, wts, sigma, tangent)):  # fmt: skip
                    break
                state["stale"] = True
            if not ok:
                stalled = state["stalled"] >= _STALLED
                why = state["why"] if res is None or stalled else f"{its} iterations"
                log.info("step to t = %.6g not converged (%s): divided", t_new, why)
                steps.reject()
                if rollback is not None:
                    rollback()
                if micro is not None:
                    micro.rollback(old)
                history, earlier, second = None, None, False
                state["stale"] = True
                continue
            its = total
            if its > _REFRESH_AFTER:
                state["stale"] = True
            step = (seg, dt, q_try - q, e_try[unknown] - e[unknown])
            second = self.prediction > 1 and _second_order(step, history, earlier)
            history, earlier = step, history
            state["fmax"] = max(state["fmax"], float(res[2]))
            state["emax"] = max(state["emax"], float(res[3]))
            q, e = q_try, e_try
            old, new = new, old
            if turning:
                self._turn_twins(old, new, wts)
            steps.accept(t_new)
            iterations.append(its)
            self._record(rec, t_new, e, temp, *disc.host(wts, sigma, grads, old[:, STATE_P]),
                         n_groups)  # fmt: skip
            if np.any(self.kind == POROUS):  # the porosity of the porous points, on the host
                self._record_porosity(rec, *disc.host(wts, old[:, : STATE_FN + 1]), n_groups)
            if rec["crystals"]:  # the twins and the cleavage planes of the crystals
                self._record_crystals(rec, *disc.host(wts, old))
            if field_times.size and np.any(np.isclose(t_new, field_times, rtol=1e-12, atol=0)):
                u = disc.displacement(q, e)
                rec["fields"][float(t_new)] = self._fields(*disc.host(sigma, grads, old), u)
            if monitor is not None and monitor(
                StepState(t_new, e.copy(), q.copy(), unknown, disc, grads, wts, sigma, tangent)
            ):
                break
        result = self._result(
            rec, path, name, iterations, steps.cuts, state["lin"], n_q, unknown.size,
            time.perf_counter() - t_start,
        )  # fmt: skip
        if measured:  # the largest asymmetry, and the linearizations of full tangents
            result.info.update(asymmetry=float(state["asymmetry"]), full_tangents=state["full"])
        if micro is not None:  # the equilibria solved for the microdamage of the planes
            result.info["microdamage_passes"] = micro.passes
            result.info["microdamage_unsettled"] = micro.unsettled
        recycling = getattr(disc, "recycling", None)
        if recycling is not None and recycling.solves:  # GMRES solves from recycled vectors
            result.info["recycled"] = recycling.solves
        return result

    # -- outputs -------------------------------------------------------------------------

    def _extra_stress(self):
        """A part of the macroscopic stress that the points do not carry (the forces of
        contacts through the faces of a cell: :mod:`morphogen.contact.periodic`), in the
        components of their stresses; none here."""
        return 0.0

    def _field_times(self, path: LoadingPath, fields) -> np.ndarray:
        if isinstance(fields, str):
            if fields != "steps":
                raise ValueError("fields: true, 'steps' or a list of times of the path")
            return path.schedule()
        if fields is True:
            return np.asarray(path.times[1:], dtype=float)
        if not fields:
            return np.zeros(0)
        times = np.asarray(fields, dtype=float).ravel()
        planned = path.schedule()
        missing = [t for t in times if not np.any(np.isclose(t, planned, rtol=1e-12, atol=0))]
        if missing:
            raise ValueError(f"fields: times {missing} are not ends of the steps of the path")
        return times

    def _element_mean(self, values) -> np.ndarray:
        return values.reshape(-1, self.ng, *values.shape[1:]).mean(axis=1)

    def _point_porosity(self, state) -> np.ndarray | None:
        """The porosity of the porous points (NaN at the others), or None without them;
        ``state`` None: the initial porosity."""
        mask = self.kind == POROUS
        if not mask.any():
            return None
        params = self.behaviours.data[2]
        out = np.full(self.kind.size, np.nan)
        if state is None:
            out[mask] = params[self.pmat[mask], P_F0]
        else:
            out[mask] = porosity(state[mask], params, self.pmat[mask])
        return out

    def _crystal_values(self, state) -> dict[str, np.ndarray]:
        """The fields of the crystals that twin or cleave (:data:`~morphogen.
        homogenization.crystal.FIELDS`) at the integration points, NaN at the others;
        none without such crystals."""
        out: dict[str, np.ndarray] = {}
        for c, cp in enumerate(self.behaviours.crystals):
            if not cp.extended:
                continue
            pts = np.flatnonzero((self.kind == CRYSTAL) & (self.pmat == c))
            if pts.size == 0:
                continue
            axes = self.table.rotations[self.point_mat[pts]]  # of the grains of the points
            for name, values in cp.state_fields(state[pts], self.finite, axes).items():
                out.setdefault(name, np.full(self.kind.size, np.nan))[pts] = values
        return out

    def _turn_twins(self, state, before, wts) -> None:
        """The lattices that the twins of the grains turn at the end of a step (``state``;
        ``before``: at its start), for the crystals that turn them grain by grain
        (:func:`~morphogen.homogenization.crystal.turn_twins`): the grains are the
        materials of the points, each with its orientation."""
        beh = self.behaviours
        for c, cp in enumerate(beh.crystals):
            if not cp.by_grain:
                continue
            pts = np.flatnonzero((self.kind == CRYSTAL) & (self.pmat == c))
            if pts.size:
                turn_twins(cp, state, before, self.point_mat[pts], beh.data[4], points=pts,
                           finite=self.finite, weights=np.asarray(wts)[pts])  # fmt: skip

    def _record_crystals(self, rec, wts, state) -> None:
        """The means of the fields of the crystals that twin or cleave over their points,
        at the end of a step (zero without ``state``: the initial ones)."""
        if not any(cp.extended for cp in self.behaviours.crystals):
            return
        if state is None:
            names = [name for cp in self.behaviours.crystals for name in cp.field_names]
            for name in dict.fromkeys(names):
                rec["crystals"].setdefault(name, []).append(0.0)
            return
        for name, values in self._crystal_values(state).items():
            there = ~np.isnan(values)
            w = np.ones(int(there.sum())) if wts is None else np.abs(np.asarray(wts)[there])
            rec["crystals"].setdefault(name, []).append(float(w @ values[there] / w.sum()))

    def _record_porosity(self, rec, wts, state, n_groups) -> None:
        """The integrals of the porosity and the volumes of the porous points of each group
        at the end of a step (the initial porosity without ``state``)."""
        f = self._point_porosity(state)
        if f is None:
            return
        mask = ~np.isnan(f)
        w = np.ones(mask.sum()) if wts is None else np.asarray(wts)[mask]
        g = self.point_group[mask]
        rec["f"].append(np.stack([np.bincount(g, w * f[mask], minlength=n_groups),
                                  np.bincount(g, w, minlength=n_groups)]))  # fmt: skip

    def _fields(self, sigma, grads, state, u) -> dict[str, np.ndarray]:
        """Element means of the stress (Cauchy), strain (logarithmic in finite strains)
        and plastic strain (tensor components), cumulated plastic strain and von Mises
        stress; the displacement of the nodes; in finite strains, and where their twins
        turn them, the orientation of the lattice of the crystals (Bunge angles in
        degrees, NaN elsewhere)."""
        if self.finite:
            stress = self._element_mean(fin.cauchy(sigma, grads))
            strain = self._element_mean(fin.log_strain(grads) * _HALF_SHEAR)
        else:
            stress = self._element_mean(sigma)
            strain = self._element_mean(grads * _HALF_SHEAR)
        out = {
            "stress": stress,
            "strain": strain,
            "plastic_strain": self._element_mean(state[:, :6]),
            "p": self._element_mean(state[:, STATE_P]),
            "mises": mises(stress),
            "displacement": u.reshape(-1, 3),
        }
        f = self._point_porosity(state)
        if f is not None:
            out["porosity"] = self._element_mean(f)
        for name, values in self._crystal_values(state).items():
            out[name] = self._element_mean(values)
        crystals = self.behaviours.crystals
        if crystals and (self.finite or any(cp.reorients for cp in crystals)):
            out["orientation"] = self._orientations(grads, state)
        return out

    def _orientations(self, grads, state) -> np.ndarray:
        """Current orientation of the lattice of each crystal element: ``g R_e^T`` with
        the rotation ``R_e`` of the lattice in finite strains (none in small strains), ``g
        R_t^T R_e^T`` for a lattice that its twins turned by ``R_t``. An element whose
        points do not have the same lattice (some turned by a twin, others not) has that
        of most of them."""
        from morphogen.orientations import matrix_to_euler

        beh = self.behaviours
        out = np.full((beh.kind.size, 3), np.nan)
        rotations = self.table.rotations
        mats = self.point_mat[:: self.ng]
        for c, cp in enumerate(beh.crystals):
            el = np.flatnonzero((beh.kind == CRYSTAL) & (beh.pmat == c))
            if el.size == 0:
                continue
            pts = (el[:, None] * self.ng + np.arange(self.ng)).ravel()
            if self.finite:
                off = CP_ALPHA + 2 * cp.n_systems
                fp_inv = state[pts, off : off + 9] + np.eye(3).ravel()
                re = fin.lattice_rotations(grads[pts], fp_inv)
            else:
                re = np.broadcast_to(np.eye(3), (pts.size, 3, 3))
            share = np.ones((el.size, self.ng))  # of the points in the mean of their element
            if cp.reorients:  # the lattice of the twin, then the rotation of the lattice
                first = cp_width(cp.n_systems, self.finite)
                twin = state[pts, first : first + 9]
                re = re @ (twin.reshape(-1, 3, 3) + np.eye(3))
                share = _majority(twin.reshape(el.size, self.ng, 9))
            re = re.reshape(el.size, self.ng, 3, 3)
            # mean rotation of the element: the rotation of the mean matrix
            u, _, vt = np.linalg.svd(np.einsum("ep,epij->eij", share, re))
            mean = u @ vt
            g0 = rotations[mats[el]]
            out[el] = matrix_to_euler(g0 @ np.swapaxes(mean, -1, -2))
        return out

    def _record(self, rec, t, e, temp, wts, sigma, grads, p, n_groups, first=False):
        """The macroscopic state and the integrals over the groups at the end of a step:
        ``p`` is the cumulated plastic strain of the points (unused, as the arrays, for the
        initial state: ``first``)."""
        rec["t"].append(float(t))
        rec["e"].append(np.array(e, dtype=float))
        rec["T"].append(float(temp))
        width = 21 if self.finite else 14
        if first:
            sums = np.zeros((n_groups, width))
            sums[:, 13] = 1.0
            if self.finite:
                sums[:, 20] = 1.0
            full = np.zeros(9 if self.finite else 6)
        elif self.finite:
            sums = fin.group_sums(self.point_group, n_groups, wts, sigma, grads, p)
            full = np.einsum("i,ij->j", wts, sigma) / self.box_volume + self._extra_stress()
        else:
            sums = _group_sums(self.point_group, n_groups, wts, sigma, grads, p)
            full = np.einsum("i,ij->j", wts, sigma) / self.box_volume + self._extra_stress()
        rec["groups"].append(sums)
        vol = sums[:, 13].sum()
        rec["p"].append(float(sums[:, 12].sum() / vol) if vol > 0 else 0.0)
        if self.finite:
            grad = fin.gradient_of(e)
            rec["s"].append(fin.macro_part(full))
            rec["cauchy"].append(fin.cauchy(full, grad))
            rec["log"].append(fin.log_strain(grad))
        else:
            rec["s"].append(full)

    def _result(self, rec, path, name, iterations, cuts, linear, n_q, n_macro, elapsed):
        groups = np.array(rec["groups"])  # steps x groups x 14 (21)
        phases = {}
        for g, phase in enumerate(self.groups):
            vol = groups[:, g, 13]
            if not np.any(vol > 0):
                continue
            safe = np.where(vol > 0, vol, 1.0)[:, None]
            phases[int(phase)] = {
                "stress": groups[:, g, 0:6] / safe,
                "strain": groups[:, g, 6:12] / safe,
                "plastic": groups[:, g, 12] / safe[:, 0],
            }
            if self.finite:
                jac = groups[:, g, 20]
                phases[int(phase)]["cauchy"] = (
                    groups[:, g, 14:20] / np.where(jac > 0, jac, 1.0)[:, None]
                )
        info = {
            "n_dofs": int(n_q + n_macro),
            "cuts": int(cuts),
            "linear_iterations": int(linear),
            "time_s": round(float(elapsed), 4),
        }
        extra = {}
        if self.finite:
            extra = {"cauchy": np.array(rec["cauchy"]), "log_strain": np.array(rec["log"])}
        if rec["f"]:
            sums = np.array(rec["f"])  # steps x 2 x groups
            for g, phase in enumerate(self.groups):
                vol = sums[:, 1, g]
                if int(phase) in phases and np.all(vol > 0):
                    phases[int(phase)]["porosity"] = sums[:, 0, g] / vol
            extra["porosity"] = sums[:, 0].sum(axis=1) / sums[:, 1].sum(axis=1)
        if rec.get("crystals"):
            extra["series"] = {name: np.array(values) for name, values in rec["crystals"].items()}
        return PathResult(
            name=name,
            bc=self.bc,
            path=path,
            times=np.array(rec["t"]),
            strain=np.array(rec["e"]),
            stress=np.array(rec["s"]),
            temperature=np.array(rec["T"]),
            plastic=np.array(rec["p"]),
            phases=phases,
            iterations=np.array(iterations, dtype=int),
            info=info,
            fields=rec["fields"],
            support=self._support() if rec["fields"] else None,
            kinematics=self.kinematics,
            **extra,
        )

    def _empty_path(self, path, name) -> PathResult:
        """Response of a sample without load-carrying material: zero stress."""
        times = np.concatenate([[path.times[0]], path.schedule()])
        strain = np.zeros((times.size, 6))
        controls = path.strain_controlled
        for k, t in enumerate(times):
            strain[k, controls] = path.at(t)[0][controls]
        extra = {}
        if self.finite:
            extra = {
                "cauchy": np.zeros((times.size, 6)),
                "log_strain": fin.log_strain(fin.gradient_of(strain)),
            }
        return PathResult(
            name=name,
            bc=self.bc,
            path=path,
            times=times,
            strain=strain,
            stress=np.zeros((times.size, 6)),
            temperature=np.array([path.at(t)[1] for t in times]),
            plastic=np.zeros(times.size),
            info={"n_dofs": 0},
            kinematics=self.kinematics,
            **extra,
        )


class _FEPaths(_PointArrays):
    """Discretization of the loading paths by finite elements (see :class:`_PathSolver`):
    the unknowns of the fluctuation reduced by the periodic links or the kinematic
    conditions; the tangent assembled at each iteration and solved by sparse LU, or by
    conjugate gradients with algebraic multigrid whose hierarchy is built at the first
    iteration of each step (GMRES with the multigrid of its symmetric part for full
    tangents). With the integration ``"bbar"`` of the problem, the measures of strain of
    the 10-node tetrahedra are those of the mean dilatation, the stresses their
    conjugates and the tangent that of their energy
    (:mod:`morphogen.homogenization.dilatation`)."""

    def __init__(self, problem: NonlinearFEProblem):
        m = problem.mesh
        self.problem = problem
        self.mesh = m
        self.finite = problem.finite
        self.gp, self.gw = GAUSS[m.order]
        self.key = (problem.bc,)
        self.t_mat, self.rep = problem._reduction(self.key)
        self.t_tr = self.t_mat.T.tocsr()
        self.n_q = self.t_mat.shape[1]
        self.indptr_n, self.indices_n = node_pattern(m.elements, m.n_nodes)
        # the elements by colours: those of a colour are assembled in parallel
        self.by_colour, self.colours = element_colours(m.elements, m.n_nodes)
        self.data = np.zeros(9 * int(self.indptr_n[-1]))
        indices = _expand_columns(self.indptr_n, self.indices_n, 3)
        lengths = np.repeat(np.diff(self.indptr_n) * 3, 3)
        indptr = np.zeros(3 * m.n_nodes + 1, dtype=np.int64)
        np.cumsum(lengths, out=indptr[1:])
        if indptr[-1] < 2**31 - 1:
            indices, indptr = indices.astype(np.int32), indptr.astype(np.int32)
        self.indices, self.indptr = indices, indptr
        self.n_dofs = 3 * m.n_nodes
        self.nodal = np.zeros(self.n_dofs)
        self.columns = np.zeros((self.n_dofs, 6))
        self.kee = np.zeros((6, 6))
        self.solver = None
        #: Multigrid with conjugate gradients or GMRES, rather than LU (whose iterations
        #: are exact Newton ones with full tangents, whatever their asymmetry)
        self.iterative = choose_method(problem.solver_method, self.n_q) != "direct"
        #: The volume change of each element as its mean (10-node tetrahedra)
        self.bbar = problem.integration == "bbar" and m.order == 2
        keep = m.n_elements * self.gp.shape[0] if self.bbar and self.finite else 1
        # F-bar: the gradients of the points and the stresses of their behaviours
        self.base = np.zeros((keep, 9))
        self.kept = np.zeros((keep, 9))
        self._bar = None  # the moduli and the vectors of the tangent of B-bar
        self._k = None  # the tangent of all the degrees of freedom, as last linearized
        self._plan = False  # the reduction of the tangents (none: by sparse products)
        self._unit = None  # the displacements of the six unit macroscopic strains

    def displacement(self, q, e) -> np.ndarray:
        if self.finite:
            macro = fin.gradient_of(e).reshape(3, 3)
        else:
            macro = voigt_to_tensor(e, strain=True)
        return times(self.mesh.nodes, macro.T).ravel() + self.t_mat @ q

    def measures(self, q, e, grads, wts) -> None:
        m = self.mesh
        u = self.displacement(q, e)
        kernel = fin.point_gradients if self.finite else _point_strains
        kernel(m.nodes, m.elements, m.order, u, self.gp, self.gw, grads, wts)
        if self.bbar:
            if self.finite:
                dilatation.mean_gradients(grads, wts, self.gp.shape[0], self.base)
            else:
                dilatation.mean_strains(grads, wts, self.gp.shape[0])

    def forces(self, sigma, wts):
        """With ``bbar``, the stresses of the points are first replaced by their
        conjugates: the outputs report them."""
        m = self.mesh
        if self.bbar:
            if self.finite:
                dilatation.conjugate_piola(sigma, wts, self.gp.shape[0], self.base, self.kept)
            else:
                dilatation.conjugate_stresses(sigma, wts, self.gp.shape[0])
        kernel = fin.internal_forces if self.finite else _internal_forces
        total, fscale = kernel(m.nodes, m.elements, m.order, self.gp, sigma, wts,
                               self.by_colour, self.colours, self.nodal)  # fmt: skip
        return self.t_tr @ self.nodal, total, fscale

    def _fill(self, tangent, wts, data) -> None:
        """The values of the tangent of all the degrees of freedom, into ``data``."""
        m = self.mesh
        if not self.bbar:
            fill = fin.fill_tangent if self.finite else _fill_tangent
            fill(m.nodes, m.elements, m.order, self.gp, tangent, wts, self.indptr_n,
                 self.indices_n, self.by_colour, self.colours, data)  # fmt: skip
            return
        nc = 9 if self.finite else 6
        if self._bar is None:
            n = wts.size
            self._bar = (np.zeros((n, nc, nc)), np.zeros((n, nc)), np.zeros((n, nc)),
                         np.zeros(n), np.zeros((n if self.finite else 1, 9)))  # fmt: skip
        moduli, vl, vr, kappa, fvec = self._bar
        if self.finite:
            dilatation.bar_moduli_finite(tangent, wts, self.gp.shape[0], self.base, self.kept,
                                         moduli, vl, vr, kappa, fvec)  # fmt: skip
        else:
            dilatation.bar_moduli(tangent, moduli, vl, vr, kappa)
        dilatation.fill_bar(m.nodes, m.elements, m.order, self.gp, wts, self.finite, moduli,
                            vl, vr, kappa, fvec, self.indptr_n, self.indices_n, self.by_colour,
                            self.colours, data)  # fmt: skip

    def _macro_fields(self) -> np.ndarray:
        """The nodal displacements of the six unit macroscopic strains (n_dofs x 6)."""
        if self._unit is None:
            unit = np.zeros((self.n_dofs, 6))
            for j in range(6):
                e = np.zeros(6)
                e[j] = 1.0
                unit[:, j] = self.displacement(np.zeros(self.n_q), e)
            self._unit = unit
        return self._unit

    def linearize(self, tangent, wts, first: bool, symmetric: bool | None = None):
        self._fill(tangent, wts, self.data)
        k = sp.csr_matrix((self.data, self.indices, self.indptr), shape=(self.n_dofs, self.n_dofs))
        self._k = k
        if self._plan is False:  # (found at the first linearization: the pattern is fixed)
            self._plan = Reduction.of(self.t_mat, self.indptr, self.indices)
        if self._plan is None:
            kqq = self.problem._reduce(self.key, k, self.t_mat, self.rep)
        else:
            kqq = self.problem._regularized(self._plan(self.data))
        symmetric = self.problem.symmetrize if symmetric is None else symmetric
        # the preconditioner of the last step is kept while it serves (its sweeps take
        # the values of each tangent): built again at the start of a step once it is worn
        if self.solver is None or (first and self.solver.worn()):
            self.solver = self.problem._linear_solver(kqq, self.t_mat, self.rep, symmetric)
        else:
            self.solver = self.solver.updated(kqq, symmetric)
        return self.solver

    def macro_response(self, tangent, wts, v) -> np.ndarray:
        """``K_Ew v``: the integral of the stress increments ``C : B v`` of the points (``A :
        grad v`` in finite strains, its components of the macroscopic strains) for the
        reduced unknowns ``v`` of the fluctuation."""
        m = self.mesh
        u = self.t_mat @ v
        if self.bbar:  # the work of the nodal forces on the fields of the unit strains
            return times(self._macro_fields().T, self._k @ u)
        nc = 9 if self.finite else 6
        grads, weights = np.zeros((wts.size, nc)), np.zeros(wts.size)
        kernel = fin.point_gradients if self.finite else _point_strains
        kernel(m.nodes, m.elements, m.order, u, self.gp, self.gw, grads, weights)
        total = np.einsum("p,pij,pj->i", wts, tangent, grads)
        return fin.macro_part(total) if self.finite else total

    def coupling(self, tangent, wts, unknown):
        m = self.mesh
        if self.bbar:  # with the tangent of the last linearization
            unit = self._macro_fields()
            self.columns[:, :] = self._k @ unit
            self.kee[:, :] = times(unit.T, self.columns)
            return self.t_tr @ self.columns[:, unknown], self.kee
        couple = fin.coupling if self.finite else _coupling
        couple(m.nodes, m.elements, m.order, self.gp, tangent, wts, self.by_colour, self.colours,
               self.columns, self.kee)  # fmt: skip
        return self.t_tr @ self.columns[:, unknown], self.kee

    def full_tangent(self, tangent, wts) -> sp.csr_matrix:
        """The tangent stiffness of all the degrees of freedom of the mesh, before the
        reduction by the boundary conditions: ``sum_e int B^T C B``, or ``int grad N . A .
        grad N`` in finite strains (its geometric part included)."""
        data = np.zeros_like(self.data)
        self._fill(tangent, wts, data)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n_dofs, self.n_dofs))

    def macro_columns(self, tangent, wts) -> tuple[np.ndarray, np.ndarray]:
        """The nodal forces of the six unit macroscopic strains on all the degrees of
        freedom (``int B^T C``, n_dofs x 6) and ``int C`` (6 x 6)."""
        m = self.mesh
        if self.bbar:
            unit = self._macro_fields()
            columns = self.full_tangent(tangent, wts) @ unit
            return columns, times(unit.T, columns)
        couple = fin.coupling if self.finite else _coupling
        columns, kee = np.zeros((self.n_dofs, 6)), np.zeros((6, 6))
        couple(m.nodes, m.elements, m.order, self.gp, tangent, wts, self.by_colour, self.colours,
               columns, kee)  # fmt: skip
        return columns, kee


class Extended:
    """The nodes of a cell and the images of some of them across its faces: ``node`` the
    node of each extended node (the nodes of the mesh first), ``offset`` its translation
    (zero for the nodes themselves) and ``reference`` its position; ``fold``, the dofs of
    the extended nodes from those of the nodes. The image of a node of displacement ``u``
    translated by ``L`` is displaced by ``u + H L``, ``H`` the macroscopic displacement
    gradient: the cell deforms with it."""

    def __init__(self, nodes: np.ndarray, node=None, offset=None):
        n = len(nodes)
        extra = np.zeros(0, dtype=np.int64) if node is None else np.asarray(node, dtype=np.int64)
        shifts = np.zeros((0, 3)) if offset is None else np.asarray(offset, dtype=float)
        self.node = np.concatenate([np.arange(n), extra])
        self.offset = np.concatenate([np.zeros((n, 3)), shifts.reshape(-1, 3)])
        self.reference = np.ascontiguousarray(nodes[self.node] + self.offset)
        n_ext = len(self.node)
        columns = (3 * self.node[:, None] + np.arange(3)).ravel()
        self.fold = sp.csr_matrix(
            (np.ones(3 * n_ext), (np.arange(3 * n_ext), columns)), shape=(3 * n_ext, 3 * n)
        )

    def positions(self, u, macro) -> np.ndarray:
        """The positions of the extended nodes for the displacements ``u`` of the nodes
        and the macroscopic displacement gradient ``macro``."""
        return np.ascontiguousarray(
            self.reference + u.reshape(-1, 3)[self.node] + self.offset @ macro.T
        )


class _SurfacePaths(_FEPaths):
    """The discretization of the loading paths of a cell with forces between the nodes of
    its surfaces and their images across its faces: the contact of the walls of its pores
    (:mod:`morphogen.contact.periodic`), the cohesive interfaces of its phases
    (:mod:`morphogen.fracture.periodic`).

    The problem gives its extended nodes (``_extended``: :class:`Extended`) and the forces
    on them, ``_surface_forces(x, tangent, first)``: the forces (and their tangent, or
    None) at the positions ``x`` of the extended nodes; ``first``: the first linearization
    of a step. These forces are added to those of the points; their work in a change of the
    macroscopic gradient, ``Σ f ⊗ X`` over the extended nodes (nothing for opposite forces
    at one place; the force that crosses a face of the cell times the period), is added to
    the integral of the stress, kept in ``_virial`` for the macroscopic stress of the
    results, and its derivatives go into the tangent of the macroscopic unknowns.
    ``_floor`` and ``_floor_stress``: floors of the scales of the residuals, for parts
    that nothing loads yet; ``_dt``: the time step being tried."""

    def __init__(self, problem):
        super().__init__(problem)
        self.cell = problem
        self._u = np.zeros(self.n_dofs)
        self._macro = np.zeros((3, 3))
        self._terms = None  # of the tangent: K_s U, U^T K_s, U^T K_s U
        ext = problem._extended
        unit = self._macro_fields()
        shifted = np.zeros((len(ext.node), 3, 6))
        for j in range(6):
            e = np.zeros(6)
            e[j] = 1.0
            shifted[:, :, j] = ext.offset @ self._gradient(e).T
        #: the displacements of the extended nodes for the six unit macroscopic strains
        self.unit_ext = ext.fold @ unit + shifted.reshape(-1, 6)

    def _gradient(self, e) -> np.ndarray:
        if self.finite:
            return fin.gradient_of(e).reshape(3, 3)
        return voigt_to_tensor(e, strain=True)

    def measures(self, q, e, grads, wts) -> None:
        super().measures(q, e, grads, wts)
        self._u = self.displacement(q, e)
        self._macro = self._gradient(e)

    def integrate(self, grads, dtemp, dt, old, new, stress, tangent, guess, ratio) -> int:
        self.cell._dt = float(dt)
        return super().integrate(grads, dtemp, dt, old, new, stress, tangent, guess, ratio)

    def forces(self, sigma, wts):
        r_q, total, fscale = super().forces(sigma, wts)
        cell, ext = self.cell, self.cell._extended
        f, _ = cell._surface_forces(ext.positions(self._u, self._macro), False, False)
        if self.finite:  # the work of the forces in the nine components of the gradient
            virial = np.einsum("ni,nj->ij", f.reshape(-1, 3), ext.reference).ravel()
        else:
            virial = self.unit_ext.T @ f
        cell._virial = virial
        f = ext.fold.T @ f
        # the scale of the forces: with a floor, for parts that nothing loads until they
        # touch (their forces are zero, and no residual is small against them)
        fscale += float(np.linalg.norm(f)) + cell._floor
        return r_q + self.t_tr @ f, total + virial, fscale

    def stress_scale(self, sigma, wts) -> float:
        # with a floor, as the scale of the forces: a cell that nothing loads yet
        return super().stress_scale(sigma, wts) + self.cell._floor_stress

    def linearize(self, tangent, wts, first: bool, symmetric: bool | None = None):
        from morphogen.fracture.cohesive import plus

        self._fill(tangent, wts, self.data)
        k = sp.csr_matrix((self.data, self.indices, self.indptr), shape=(self.n_dofs, self.n_dofs))
        cell, ext = self.cell, self.cell._extended
        _, ks = cell._surface_forces(ext.positions(self._u, self._macro), True, first)
        self._terms = None
        if ks is not None:
            columns = ks @ self.unit_ext  # the forces of the unit macroscopic strains
            self._terms = (
                ext.fold.T @ columns,
                (ks.T @ self.unit_ext).T @ ext.fold,
                self.unit_ext.T @ columns,
            )
            k = plus(k, (ext.fold.T @ ks @ ext.fold).tocsr())
        self._k = k
        kqq = self.problem._reduce(self.key, k, self.t_mat, self.rep)
        symmetric = self.problem.symmetrize if symmetric is None else symmetric
        if first or self.solver is None:
            self.solver = self.problem._linear_solver(kqq, self.t_mat, self.rep, symmetric)
        else:
            self.solver = self.solver.updated(kqq, symmetric)
        return self.solver

    def macro_response(self, tangent, wts, v) -> np.ndarray:
        out = super().macro_response(tangent, wts, v)
        if self._terms is not None:
            out = out + self._terms[1] @ (self.t_mat @ v)
        return out

    def coupling(self, tangent, wts, unknown):
        g, kee = super().coupling(tangent, wts, unknown)
        if self._terms is None:
            return g, kee
        return g + self.t_tr @ self._terms[0][:, unknown], kee + self._terms[2]


class NonlinearFEProblem(_PathSolver, FEElasticProblem):
    """Nonlinear behaviours of the phases (von Mises elasto-(visco)plasticity, crystal
    plasticity) on a tetrahedral mesh, loaded along paths: periodic conditions (the
    periodic element, whose macroscopic strain components are degrees of freedom) or
    kinematic ones (``kubc``: ``u = E x`` on the boundary, ``E`` likewise), in small or
    finite strains (``kinematics``).

    ``newton_tol``: relative residual of convergence of the equilibrium (to the norm of
    the nodal forces before they cancel) and of the macroscopic stress (to the integral
    of the norm of the stress); ``max_newton`` iterations per step; ``max_cuts``
    divisions of a step by two. ``tangent``: the consistent tangents of the points
    symmetrized (``"symmetric"``: conjugate gradients with multigrid), full (``"full"``:
    GMRES with the multigrid of their symmetric part; LU either way for small systems)
    or, by default (``"auto"``), symmetrized when the laws of all the phases have
    symmetric tangents (:func:`behaviour_table`), and otherwise kept full by LU and, with
    multigrid, symmetrized at each linearization where their asymmetry is below
    :data:`ASYMMETRY`.

    ``integration``: ``"full"``, the behaviours integrated with the strain of each point,
    or ``"bbar"``, with the volume change of each 10-node tetrahedron as its mean
    (``B``-bar; ``F``-bar in finite strains: :mod:`morphogen.homogenization.dilatation`),
    for nearly incompressible phases — elastomers — and isochoric plastic flow.
    """

    def __init__(
        self,
        mesh,
        materials: dict[int, Material],
        bc: str = "periodic",
        *,
        solver: str = "auto",
        tol: float = 1e-8,
        maxiter: int | None = None,
        newton_tol: float = 1e-6,
        max_newton: int = 25,
        max_cuts: int = 10,
        kinematics: str = "small",
        tangent: str = "auto",
        integration: str = "full",
    ):
        super().__init__(
            mesh, materials, bc, solver=solver, tol=tol, maxiter=maxiter, integration=integration
        )
        if self.bc not in ("periodic", "kubc"):
            raise ValueError(
                "loading paths: periodic or kinematic (kubc) boundary conditions; static "
                "and mixed conditions prescribe the strain or the stress only"
            )
        self._path_options(newton_tol, max_newton, max_cuts, kinematics, tangent)
        self.materials = materials
        self._smoothing: dict = {}
        if not self.empty:
            gp, _ = GAUSS[self.mesh.order]
            beh = behaviour_table(self.mesh.phase, materials, self.table, finite=self.finite)
            self._point_behaviours(beh, self.mat, self.group, gp.shape[0])
            if self.mesh.order == 1 and beh.nonlinear:
                log.warning(
                    "linear tetrahedra lock under isochoric plastic flow: prefer quadratic "
                    "elements (mesh order 2) for plasticity"
                )

    def _discretization(self) -> _FEPaths:
        return _FEPaths(self)

    def _support(self):
        return self.mesh

    def _smoother(self, length: float):
        """The solver of the equation of the microdamage for a length (kept): the
        factorized operator ``M + length^2 K`` of the nodes (the masters of the periodic
        conditions), the loads of the nodes for a unit value in each element (nodes x
        elements) and the volumes of the elements."""
        found = self._smoothing.get(length)
        if found is not None:
            return found
        from scipy.sparse.linalg import splu

        from morphogen.homogenization.bloch import MASS, mass_matrix
        from morphogen.homogenization.fe import assemble, periodic_map

        mesh = self.mesh
        count = mesh.n_elements
        volumes = np.abs(mesh.volumes())
        share = MASS[mesh.order].sum(axis=1)  # the integral of each function over a unit volume
        nodes = mesh.elements.shape[1]
        load = sp.csr_matrix(
            (
                (volumes[:, None] * share[None, :]).ravel(),
                (mesh.elements.ravel(), np.repeat(np.arange(count), nodes)),
            ),
            shape=(mesh.n_nodes, count),
        )
        stiffness = assemble(mesh, 1, np.zeros(count, dtype=np.int64), np.eye(3)[None])
        operator = mass_matrix(mesh, np.ones(count)) + length**2 * stiffness
        reduce = None
        if self.bc == "periodic":
            reduce = periodic_map(mesh)[0]
            operator = reduce.T @ operator @ reduce
        factor = splu(sp.csc_matrix(operator))

        def solve(rhs):
            if reduce is None:
                return factor.solve(rhs)
            return reduce @ factor.solve(reduce.T @ rhs)

        found = self._smoothing[length] = (solve, load, volumes)
        return found

    def _smooth(self, elements, values, length) -> np.ndarray:
        """The equation of the microdamage on the mesh, ``chi - length^2 lap chi =
        values`` with the ``values`` of the ``elements`` (nothing in the others), by
        finite elements of the order of the mesh: the field of the nodes — periodic with
        the conditions of the cell, and with no flux through the other boundaries, those
        of the pores and the faces of the box under kinematic conditions — and its means
        in the elements."""
        solve, load, volumes = self._smoother(float(length))
        source = np.zeros(volumes.size)
        source[elements] = values
        field = solve(load @ source)
        return (load.T @ field)[elements] / volumes[elements]
