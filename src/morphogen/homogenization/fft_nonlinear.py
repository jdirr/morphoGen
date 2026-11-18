"""Nonlinear homogenization of periodic voxel images along loading paths: the periodic
element of :mod:`morphogen.homogenization.nonlinear` on the voxels of the FFT solver.

Each voxel is the trilinear hexahedron of :mod:`morphogen.homogenization.fft`, its 2 x 2
x 2 Gauss points the integration points of the behaviours. Fully integrated, these
elements lock under isochoric plastic flow (the macroscopic flow stress of a composite
with a plastic matrix converges from far above as the voxels are refined): by default the
volumetric part of the measures is averaged over each voxel, the B-bar method in small
strains (Hughes 1980) and the F-bar method in finite strains (de Souza Neto, Perić,
Dutko & Owen 1996), in their variational form (the forces derive from the energy of
these measures), so that the tangent stays symmetric. The unknowns of a step are the
periodic nodal fluctuation ``w`` and the components of the macroscopic strain prescribed
as stresses (the macroscopic degrees of freedom of the periodic element), solved by the
Newton iterations of :class:`~morphogen.homogenization.nonlinear._PathSolver`: the
macroscopic block is condensed, the corrections are inexact, the steps are divided when
they do not converge.

Each linearized equilibrium ``K_t dw = -R_w`` is solved by conjugate gradients
preconditioned by the inverse of the stiffness of a homogeneous elastic reference medium,
applied in Fourier space (:class:`~morphogen.homogenization.fft.FourierPreconditioner`): a
Newton-Krylov scheme (Gélébart & Mondon-Cancel 2013; Kabel, Böhlke & Schneider 2014) on
the discretization of Schneider, Merkert & Kabel (2017). The tangent operator is applied
matrix-free from the consistent tangents of the points: stress increments ``C_t : B v`` at
the points, then nodal forces gathered node by node; nothing is assembled. The tangents
are symmetrized, so that conjugate gradients apply, when the laws of all the phases have
symmetric ones; otherwise (crystal and porous plasticity, the Armstrong-Frederick
hardening, learned laws that are not potentials, MFront behaviours without a symmetric
tangent) they are kept full, symmetrized again at the linearizations where their measured
asymmetry is small (:data:`~morphogen.homogenization.nonlinear.ASYMMETRY`), and GMRES
with deflated restarts and the same preconditioner solves the corrections of the others
(:func:`~morphogen.homogenization.solvers.gmres`; ``tangent`` chooses), recycling from a
solve to the next of the path the space of the first one that restarts (GCRO-DR,
:class:`~morphogen.homogenization.solvers.Recycling`) and stopping a solve that stagnates
(an inexact correction). Pores
have no integration points and, as in the linear solver, need no ersatz material.

The behaviours are those of the finite elements: von Mises elasto-(visco)plasticity, and
crystal plasticity with the slip systems of each grain turned by its orientation, in
small or finite strains (total Lagrangian, upper triangular macroscopic gradient).

Memory: per integration point, the measure of strain, the stress, the tangent (36 or 81
values) and the internal variables twice (a crystal with 12 slip systems: about 1 kB in
small strains, 1.7 kB in finite strains), with 8 points per voxel: about 2 GB for a 64³
polycrystal in small strains.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from functools import partial

import numpy as np

from morphogen.homogenization import finite as fin
from morphogen.homogenization import kernels
from morphogen.homogenization.elements import voigt_to_tensor
from morphogen.homogenization.fft import FFTElasticProblem, FourierPreconditioner
from morphogen.homogenization.materials import Material
from morphogen.homogenization.nonlinear import _PathSolver, _PointArrays, behaviour_table
from morphogen.homogenization.solvers import (
    GMRES_STAGNATION,
    NEGLIGIBLE,
    Recycling,
    SolveInfo,
    conjugate_gradient,
    gmres,
    norm,
)
from morphogen.image import VoxelImage

log = logging.getLogger("morphogen")

#: Integration of the voxels: B-bar (F-bar in finite strains), or full
INTEGRATIONS = ("bbar", "full")


class _KrylovSolver:
    """Conjugate gradients on a tangent operator applied matrix-free (``apply``),
    preconditioned in Fourier space (``precondition``), from a starting point when one is
    given (the solver of :meth:`_VoxelPaths.linearize`); GMRES on an operator that is not
    ``symmetric``, recycling the space of ``recycle`` (shared by the solves of a path: the
    operator is a new one for it) and stopped where it stagnates
    (:data:`~morphogen.homogenization.solvers.GMRES_STAGNATION`)."""

    method = "fft"

    def __init__(self, apply, precondition, tol: float, maxiter: int, symmetric: bool = True,
                 recycle: Recycling | None = None):  # fmt: skip
        self.apply = apply
        self.precondition = precondition
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self.krylov = conjugate_gradient
        if not symmetric:
            self.krylov = partial(gmres, recycle=recycle, stagnation=GMRES_STAGNATION)
        if recycle is not None:
            recycle.renew()

    def solve(self, rhs, x0=None, tol=None) -> tuple[np.ndarray, SolveInfo]:
        """Solution of ``K x = rhs`` to the relative residual ``tol`` (default: that of
        the solver), from ``x0`` if given. The nodal forces of a periodic grid sum to
        zero (the translations are the kernel of ``K``): the mean of ``rhs``, round-off,
        is removed."""
        b = np.asarray(rhs, dtype=float).reshape(-1, 3)
        b = (b - b.mean(axis=0)).ravel()
        n = b.size
        t0 = time.perf_counter()
        tol = self.tol if tol is None else float(tol)
        norm_b = norm(b)
        if norm_b == 0.0:
            return np.zeros(n), SolveInfo(self.method, n, 0, 0.0, 0.0, 0.0)
        start = None if x0 is None or not np.any(x0) else np.asarray(x0, dtype=float)
        r = b if start is None else b - self.apply(start)
        norm_r = norm(r)
        iterations, residual = 0, norm_r / norm_b
        if norm_r <= tol * norm_b:
            x = np.zeros(n) if start is None else start.copy()
        else:
            x, iterations, rel = self.krylov(
                self.apply, self.precondition, r, tol * norm_b / norm_r, self.maxiter
            )
            residual = rel * norm_r / norm_b
            if start is not None:
                x += start
        if residual > 10 * tol:
            log.warning(
                "FFT solver stopped at relative residual %.2e (tolerance %.1e)", residual, tol
            )
        info = SolveInfo(self.method, n, iterations, residual, 0.0, time.perf_counter() - t0)
        return x, info


class _VoxelPaths(_PointArrays):
    """Discretization of the loading paths on the voxels (see
    :class:`~morphogen.homogenization.nonlinear._PathSolver`): the periodic nodal
    fluctuation (3 unknowns per voxel), the integration points of the voxels that carry
    material, the tangent applied matrix-free and solved by FFT-preconditioned conjugate
    gradients (GMRES for full tangents); on the CPU (on the GPU:
    :class:`morphogen.homogenization.gpu.VoxelPaths`)."""

    def __init__(self, problem: NonlinearFFTProblem):
        self.problem = problem
        self.shape = problem.shape
        self.finite = problem.finite
        self.elem = problem.elem
        self.slot = problem.slot
        el = problem.element
        self.g_gauss = np.array(el.g_gauss)
        self.weight = float(el.weight)
        self.n_q = 3 * int(np.prod(self.shape))
        self.reference = problem._reference_element()
        self.bbar = problem.integration == "bbar"
        self.tol = problem.tol
        self.maxiter = problem.maxiter or 20 * int(max(self.shape)) + 1000
        #: The space that the GMRES solves of the path recycle (empty until one restarts)
        self.recycling = Recycling()
        self._buffers(problem.kind.size, 9 if self.finite else 6)

    def _buffers(self, n_pts: int, nc: int) -> None:
        """The preconditioner and the work arrays of ``n_pts`` points (``nc`` components)."""
        self.precondition = FourierPreconditioner.from_element(self.shape, self.reference, 3)
        self.work = np.zeros((n_pts, nc))
        # F-bar: the gradients of the points and the stresses of their behaviours
        keep = self.finite and self.bbar
        self.base = np.zeros((n_pts if keep else 1, 9))
        self.stress = np.zeros((n_pts if keep else 1, nc))
        self.zero = np.zeros(9)

    def _nodal(self, s) -> tuple[np.ndarray, float]:
        """Nodal forces of the point stresses ``s`` and their norm before they cancel;
        forces below ``NEGLIGIBLE`` times that norm are round-off (uniform stresses, as
        in a homogeneous medium), set to zero."""
        nx, ny, nz = self.shape
        out = np.empty(self.n_q)
        scale = kernels.voxel_forces(
            nx, ny, nz, self.slot, s, self.weight, self.g_gauss, self.finite, out
        )
        if norm(out) <= NEGLIGIBLE * scale:
            out[:] = 0.0
        return out, scale

    def _macro(self, e) -> np.ndarray:
        """Macroscopic gradient (9, row by row) of the macroscopic components ``e``:
        upper triangular in finite strains, symmetric in small strains."""
        if self.finite:
            return fin.gradient_of(e)
        return voigt_to_tensor(e, strain=True).ravel()

    def measures(self, q, e, grads, wts) -> None:
        nx, ny, nz = self.shape
        kernels.voxel_measures(nx, ny, nz, self.elem, q, self._macro(e), self.g_gauss,
                               self.finite, self.bbar, grads, self.base)  # fmt: skip
        wts.fill(self.weight)

    def forces(self, sigma, wts):
        """With B-bar (F-bar), the stresses of the points are first replaced by their
        conjugates: the outputs report them."""
        if self.bbar:
            if self.finite:
                np.copyto(self.stress, sigma)
            kernels.voxel_effective(self.elem.size, self.finite, sigma, self.base)
        out, scale = self._nodal(sigma)
        return out, self.weight * sigma.sum(axis=0), scale

    def _increments(self, v, direction, tangent) -> np.ndarray:
        """Stress increments (conjugate ones with B-bar) of the points for the nodal
        displacements ``v`` and the uniform gradient ``direction``."""
        nx, ny, nz = self.shape
        kernels.voxel_tangent_stress(nx, ny, nz, self.elem, v, direction, tangent, self.stress,
                                     self.base, self.g_gauss, self.finite, self.bbar,
                                     self.work)  # fmt: skip
        return self.work

    #: The linear solver is a Krylov method (see :attr:`_PathSolver.measured`)
    iterative = True

    def linearize(self, tangent, wts, first: bool, symmetric: bool | None = None):
        def apply(v):
            return self._nodal(self._increments(v, self.zero, tangent))[0]

        symmetric = self.problem.symmetrize if symmetric is None else symmetric
        return _KrylovSolver(apply, self.precondition, self.tol, self.maxiter, symmetric,
                             self.recycling)  # fmt: skip

    def macro_response(self, tangent, wts, v) -> np.ndarray:
        """``K_Ew v``: the integral of the stress increments (conjugate ones with B-bar and
        F-bar) of the points for the nodal fluctuation ``v``."""
        total = self.weight * self._increments(v, self.zero, tangent).sum(axis=0)
        return fin.macro_part(total) if self.finite else total

    def coupling(self, tangent, wts, unknown):
        g = np.empty((self.n_q, unknown.size))
        kee = np.zeros((6, 6))
        still = np.zeros(self.n_q)
        for k, j in enumerate(unknown):
            unit = np.zeros(6)
            unit[j] = 1.0
            ds = self._increments(still, self._macro(unit), tangent)
            g[:, k] = self._nodal(ds)[0]
            total = self.weight * ds.sum(axis=0)
            kee[:, j] = fin.macro_part(total) if self.finite else total
        return g, kee

    def displacement(self, q, e) -> np.ndarray:
        """Displacement at the centres of all the voxels (voxels x 3)."""
        image = self.problem.image
        nx, ny, nz = self.shape
        if self.finite:
            macro = fin.gradient_of(e).reshape(3, 3)
        else:
            macro = voigt_to_tensor(e, strain=True)
        axes = [
            image.origin[d] + (np.arange(n) + 0.5) * image.spacing[d]
            for d, n in enumerate(self.shape)
        ]
        x = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        mean = np.empty((nx * ny * nz, 3))
        kernels.voxel_centres(nx, ny, nz, q, mean)
        return x @ macro.T + mean


class NonlinearFFTProblem(_PathSolver, FFTElasticProblem):
    """Nonlinear behaviours of the phases (von Mises elasto-(visco)plasticity, crystal
    plasticity) on a periodic voxel image, loaded along paths: the periodic element on
    trilinear voxels, whose macroscopic strain components are degrees of freedom, each
    Newton correction solved by FFT-preconditioned conjugate gradients (GMRES with full
    tangents), in small or finite strains (``kinematics``). Crystals are turned by the
    orientations of their grains (``grains``: image or array of grain labels;
    ``orientations``: Bunge angles of each grain).

    ``integration``: ``"bbar"`` (default) averages the volumetric part of the measures
    over each voxel (B-bar in small strains, F-bar in finite strains), so that the
    trilinear voxels do not lock under isochoric plastic flow; ``"full"`` integrates every
    component at the 8 Gauss points (the discretization of the linear FFT solver).
    ``device``: ``"gpu"`` runs the iterations on the GPU, where the arrays of the points
    stay (the measures, the integration of the behaviours, the nodal forces and the
    conjugate gradients; see :mod:`morphogen.homogenization.gpu`), ``"auto"`` when there
    is one. ``tangent``: the consistent tangents of the points symmetrized and solved by
    conjugate gradients (``"symmetric"``), full and solved by GMRES (``"full"``), or by
    default (``"auto"``) symmetrized when the laws of all the phases have symmetric
    tangents (:func:`~morphogen.homogenization.nonlinear.behaviour_table`:
    :attr:`symmetrize` tells which), and otherwise at each linearization where their
    asymmetry is below :data:`~morphogen.homogenization.nonlinear.ASYMMETRY`
    (:attr:`measured`; the outputs report the largest asymmetry and the linearizations
    solved with the full tangents).
    ``tol``: relative residual of the conjugate gradients (the Newton corrections are
    solved inexactly above it); ``maxiter``: their iterations; ``reference``: Lamé
    coefficients of the reference medium of the preconditioner (default: the means of the
    extreme elastic moduli of the phases); ``newton_tol``, ``max_newton`` and
    ``max_cuts``: as for :class:`~morphogen.homogenization.nonlinear.NonlinearFEProblem`.
    """

    def __init__(
        self,
        image: VoxelImage,
        materials: dict[int, Material],
        bc: str = "periodic",
        *,
        tol: float = 1e-8,
        maxiter: int | None = None,
        reference: tuple[float, float] | None = None,
        grains=None,
        orientations=None,
        newton_tol: float = 1e-6,
        max_newton: int = 25,
        max_cuts: int = 10,
        kinematics: str = "small",
        integration: str = "bbar",
        device: str = "cpu",
        tangent: str = "auto",
    ):
        if integration not in INTEGRATIONS:
            raise ValueError(f"integration: one of {INTEGRATIONS}, got {integration!r}")
        self.integration = integration
        super().__init__(
            image,
            materials,
            bc,
            tol=tol,
            maxiter=maxiter,
            reference=reference,
            grains=grains,
            orientations=orientations,
            device=device,
        )
        self._path_options(newton_tol, max_newton, max_cuts, kinematics, tangent)
        self.materials = materials
        self._smoothing: dict = {}  # the Fourier symbols of the equation of the microdamage
        self.groups: tuple[int, ...] = ()
        if not self.empty:
            # the voxels that carry material, and the phases they belong to
            self.elem = np.flatnonzero(self.active)
            self.slot = np.full(self.mat.size, -1, dtype=np.int64)
            self.slot[self.elem] = np.arange(self.elem.size)
            present = np.unique(self.stat[self.elem])
            self.groups = tuple(self.table.phases[g] for g in present)
            group = np.searchsorted(present, self.stat[self.elem])
            table = dataclasses.replace(self.table, mat=self.mat[self.elem], stat=group)
            phase = np.asarray(image.data).ravel()[self.elem]
            beh = behaviour_table(phase, materials, table, finite=self.finite)
            self._point_behaviours(beh, self.mat[self.elem], group, 8)

    def _discretization(self) -> _VoxelPaths:
        if self.device == "gpu":
            from morphogen.homogenization import gpu

            return gpu.VoxelPaths(self)
        return _VoxelPaths(self)

    def _support(self) -> VoxelImage:
        return self.image

    def _smooth(self, elements, values, length) -> np.ndarray:
        """The equation of the microdamage on the grid of the voxels, ``chi - length^2
        lap chi = values`` with the ``values`` of the voxels ``elements`` (among those
        that carry material; nothing elsewhere) and the seven-point Laplacian, solved in
        Fourier space: the inverse of a positive operator, so that the microdamage of a
        damage is positive, and spreads through the voxels of the other phases."""
        voxels = self.elem[elements]
        grid = np.zeros(self.mat.size)
        grid[voxels] = values
        symbol = self._smoothing.get(float(length))
        if symbol is None:
            symbol = np.ones(1)
            for axis, (n, h) in enumerate(zip(self.shape, self.image.spacing, strict=True)):
                last = axis == len(self.shape) - 1
                k = np.arange(n // 2 + 1) if last else np.arange(n)
                term = (length / h) ** 2 * (2.0 - 2.0 * np.cos(2.0 * np.pi * k / n))
                shape = [1] * len(self.shape)
                shape[axis] = term.size
                symbol = symbol + term.reshape(shape)
            self._smoothing[float(length)] = symbol
        axes = tuple(range(len(self.shape)))
        spectrum = np.fft.rfftn(grid.reshape(self.shape), axes=axes) / symbol
        smooth = np.fft.irfftn(spectrum, s=self.shape, axes=axes)
        return smooth.ravel()[voxels]

    def _fields(self, sigma, grads, state, u) -> dict[str, np.ndarray]:
        """The fields of the finite elements on the grid of the image (voxels x
        components; NaN in the pores), the displacement at the centres of the voxels."""
        out = {}
        for name, values in super()._fields(sigma, grads, state, u).items():
            if name == "displacement":
                out[name] = values.reshape(*self.shape, 3)
                continue
            full = np.full((self.mat.size, *values.shape[1:]), np.nan)
            full[self.elem] = values
            out[name] = full.reshape(*self.shape, *values.shape[1:])
        return out
