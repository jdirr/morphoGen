"""FFT-based homogenization of periodic voxel images.

Each voxel is a trilinear hexahedron and the fluctuation ``w`` of ``u = E x + w``
(``T = G . x + w``) is periodic on the grid. The equilibrium equations ``K w = f`` are
solved by conjugate gradients without assembling ``K`` (each product loops over the
voxels), preconditioned by the inverse of the stiffness ``K0`` of a homogeneous
reference medium. ``K0`` is a circulant operator: it is diagonal (3 x 3 blocks in
elasticity) in Fourier space, and its inverse costs a few FFTs (Schneider, Merkert &
Kabel 2017, "FFT-based homogenization for microstructures discretized by linear
hexahedral elements", Int. J. Numer. Meth. Engng 109, 1461-1489). This is a
Lippmann-Schwinger scheme whose discrete Green operator is that of trilinear elements:
the fields have no ringing artefacts, and pores (zero stiffness) are handled without an
ersatz material.

The number of iterations grows with the contrast between the phases (as its square
root, at worst); the memory is a few vectors of the size of the grid. With
``device="gpu"`` the conjugate gradients run on the GPU (:mod:`morphogen.homogenization.
gpu`); the solutions are those of the CPU to the tolerance of the iterations.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from numba import njit, prange
from scipy import fft as sfft

from morphogen import gpu, settings
from morphogen.homogenization import kernels
from morphogen.homogenization.core import (
    Loading,
    MacroscopicLoadings,
    PhaseStatistics,
    Solution,
    boundary_condition,
)
from morphogen.homogenization.elements import hex_element, voigt_to_tensor
from morphogen.homogenization.materials import Material, material_table, transport_physics
from morphogen.homogenization.solvers import NEGLIGIBLE, SolveInfo, conjugate_gradient, norm
from morphogen.image import VoxelImage

log = logging.getLogger("morphogen")


@njit(parallel=True, cache=True)
def apply_operator(nx, ny, nz, nd, group, mats, w, out):
    """out = K w on the periodic grid; ``mats[g]`` is the element matrix of material
    ``g`` of the voxels with ``group`` g (-1: no stiffness). Loops over nodes, so no
    atomics are needed."""
    corners = kernels.CORNERS
    size = 8 * nd
    for n in prange(nx * ny * nz):
        k = n % nz
        j = (n // nz) % ny
        i = n // (ny * nz)
        acc = np.zeros(3)
        we = np.empty(size)
        for a in range(2):
            ei = i - a if i - a >= 0 else nx - 1
            for b in range(2):
                ej = j - b if j - b >= 0 else ny - 1
                for c in range(2):
                    ek = k - c if k - c >= 0 else nz - 1
                    g = group[(ei * ny + ej) * nz + ek]
                    if g < 0:
                        continue
                    ln = kernels.CORNER_INDEX[a, b, c]
                    for q in range(8):
                        qi = ei + corners[q, 0]
                        qj = ej + corners[q, 1]
                        qk = ek + corners[q, 2]
                        if qi == nx:
                            qi = 0
                        if qj == ny:
                            qj = 0
                        if qk == nz:
                            qk = 0
                        m = (qi * ny + qj) * nz + qk
                        for c2 in range(nd):
                            we[q * nd + c2] = w[m * nd + c2]
                    mat = mats[g]
                    for r in range(nd):
                        row = ln * nd + r
                        s = 0.0
                        for col in range(size):
                            s += mat[row, col] * we[col]
                        acc[r] += s
        for r in range(nd):
            out[n * nd + r] = acc[r]


def _stencil(k: np.ndarray, nd: int) -> np.ndarray:
    """Blocks (3, 3, 3, nd, nd) of the homogeneous operator of element matrix ``k`` for
    node offsets -1, 0, 1."""
    blocks = np.zeros((3, 3, 3, nd, nd))
    corners = kernels.CORNERS
    for ln in range(8):
        for q in range(8):
            off = corners[q] - corners[ln] + 1
            rows = slice(ln * nd, ln * nd + nd)
            cols = slice(q * nd, q * nd + nd)
            blocks[off[0], off[1], off[2]] += k[rows, cols]
    return blocks


class FourierPreconditioner:
    """Inverse of a homogeneous reference operator, applied in Fourier space: the
    element matrix ``coef_a * ka + coef_b * kb`` (see :meth:`from_element`)."""

    def __init__(self, shape, ka, kb, coef_a: float, coef_b: float, nd: int):
        self._build(shape, coef_a * np.asarray(ka) + coef_b * np.asarray(kb), nd)

    @classmethod
    def from_element(cls, shape, k_ref, nd: int) -> FourierPreconditioner:
        """Preconditioner of the homogeneous medium of element matrix ``k_ref``."""
        self = cls.__new__(cls)
        self._build(shape, np.asarray(k_ref, dtype=float), nd)
        return self

    def _build(self, shape, k, nd: int) -> None:
        self.shape = tuple(int(s) for s in shape)
        self.nd = nd
        self.blocks = np.ascontiguousarray(_stencil(k, nd))
        nx, ny, nz = self.shape
        # the mean of the fluctuation is set to zero: the inverse is zero at frequency 0
        self.inverse = np.empty((nx, ny, nz // 2 + 1, nd, nd), dtype=np.complex128)
        kernels.fourier_inverse(nx, ny, nz, self.blocks, self.inverse)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        nd, shape = self.nd, self.shape
        field = r.reshape(*shape, nd)
        workers = settings.workers()
        hat = sfft.rfftn(field, axes=(0, 1, 2), workers=workers)
        hat = np.einsum("...ij,...j->...i", self.inverse, hat)
        return sfft.irfftn(hat, s=shape, axes=(0, 1, 2), workers=workers).reshape(-1)


class _FFTProblem(MacroscopicLoadings):
    physics = ""
    nd = 0

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
        device: str = "cpu",
        physics: str | None = None,
    ):
        if not isinstance(image, VoxelImage):
            raise TypeError("the FFT solver needs a VoxelImage")
        if physics is not None:
            self._set_physics(physics)
        self.device = gpu.device_of(device)
        if boundary_condition(bc) != "periodic":
            raise ValueError(
                "the FFT solver of voxel images uses periodic boundary conditions; for "
                "kubc, mixed or subc conditions, mesh the microstructure (finite elements, "
                "see morphogen.mesh.conforming)"
            )
        if image.data.dtype.kind not in "iub":
            raise ValueError("homogenization needs an integer phase image")
        if not image.periodic:
            log.info("FFT solver: the image is treated as the cell of a periodic medium")
        if grains is not None:
            grains = np.asarray(grains.data if isinstance(grains, VoxelImage) else grains)
            if grains.shape != image.shape:
                raise ValueError("grains: one grain label per voxel of the image")
        self.bc = "periodic"
        self.image = image
        self.shape = image.shape
        self.box_volume = float(image.volume)
        self.table = material_table(
            image.data, materials, self.physics, grains=grains, orientations=orientations
        )
        # material of every voxel (-1: void), phase of every voxel (statistics)
        self.mat = np.ascontiguousarray(self.table.mat)
        self.stat = np.ascontiguousarray(self.table.stat)
        self.active = self.mat >= 0
        self.empty = not np.any(self.active) or not np.any(self.table.matrices)
        if self.empty:
            log.warning("no load-carrying material in the sample: its apparent properties are zero")
        self.element = hex_element(tuple(image.spacing))
        self.element_matrices = self._element_matrices(self.table.matrices)
        self.counts = np.bincount(self.mat[self.active], minlength=len(self.table.matrices))
        self.tol = float(tol)
        self.maxiter = maxiter
        self.reference = reference
        self._preconditioner = None
        self._units: dict = {}

    def _set_physics(self, physics: str) -> None:
        """The physics of the problem, among those its class solves (elasticity; any of
        the transport phenomena of a scalar potential for the thermal problems)."""
        if transport_physics(physics) != (self.nd == 1):
            kind = "a transport phenomenon" if self.nd == 1 else "'elastic'"
            raise ValueError(f"{type(self).__name__}: physics {kind}, got {physics!r}")
        self.physics = physics

    def _element_matrices(self, matrices) -> np.ndarray:
        raise NotImplementedError

    def _reference_element(self) -> np.ndarray:
        raise NotImplementedError

    def _load_scale(self, loads) -> float:
        """Norm of the element loads (without cancellation between elements)."""
        return float(np.sqrt(np.sum(self.counts * np.sum(loads * loads, axis=1))))

    def _solve(self, rhs, scale: float):
        nx, ny, nz = self.shape
        nd = self.nd
        if norm(np.ravel(rhs)) <= NEGLIGIBLE * scale:
            return np.zeros_like(rhs), SolveInfo("fft", rhs.size, 0, 0.0, 0.0, 0.0)
        if self._preconditioner is None and self.device != "gpu":
            self._preconditioner = FourierPreconditioner.from_element(
                self.shape, self._reference_element(), nd
            )
        mats = self.element_matrices

        def apply(v):
            out = np.empty_like(v)
            apply_operator(nx, ny, nz, nd, self.mat, mats, v, out)
            return out

        maxiter = self.maxiter or 20 * int(max(self.shape)) + 1000
        t0 = time.perf_counter()
        if self.device == "gpu":
            from morphogen.homogenization import gpu as gpu_solver

            x, it, res = gpu_solver.solve_linear(self, rhs, self.tol, maxiter)
        else:
            x, it, res = conjugate_gradient(apply, self._preconditioner, rhs, self.tol, maxiter)
        if res > self.tol:
            log.warning("FFT solver stopped at relative residual %.2e after %d iterations", res, it)
        method = "fft-gpu" if self.device == "gpu" else "fft"
        return x, SolveInfo(method, rhs.size, it, res, 0.0, time.perf_counter() - t0)

    def _phase_statistics(self, sums, n_mean, names, sizes) -> list[PhaseStatistics]:
        out = []
        vol_index = sums.shape[1] - 1
        for g, phase in enumerate(self.table.phases):
            vol = sums[g, vol_index]
            if vol <= 0:
                continue
            mean, std, pos = {}, {}, 0
            for name, size in zip(names, sizes, strict=True):
                m = sums[g, pos : pos + size] / vol
                sq = sums[g, n_mean + pos : n_mean + pos + size] / vol
                mean[name] = m
                std[name] = np.sqrt(np.maximum(sq - m * m, 0.0))
                pos += size
            out.append(PhaseStatistics(phase, float(vol / self.box_volume), mean, std))
        return out

    def _empty_solution(self, loading: Loading, t0: float) -> Solution:
        elastic = self.physics == "elastic"
        values = np.where(loading.strain_controlled, loading.array, 0.0)
        # the phases inert for the physics (voids, insulators) fill the image
        strain = voigt_to_tensor(values, strain=True) if elastic else values
        return Solution(
            physics=self.physics,
            bc=self.bc,
            loading=loading,
            strain=strain,
            stress=np.zeros((3, 3)) if elastic else np.zeros(3),
            energy=0.0,
            material_fraction=0.0,
            phases=[],
            info=SolveInfo("none", 0, 0, 0.0, 0.0, time.perf_counter() - t0),
        )


#: Mandel weights: sqrt(2) on the shear components
_MANDEL = np.array([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])


def reference_moduli(matrices, isotropic: bool) -> tuple[float, float]:
    """Lame coefficients of the isotropic reference medium of the FFT preconditioner:
    the mean of the extreme moduli of the materials (bulk moduli and shear stiffnesses
    of the crystals)."""
    d = np.asarray(matrices, dtype=float)
    if isotropic:
        lam, mu = d[:, 0, 1], d[:, 3, 3]
        return 0.5 * (lam.min() + lam.max()), 0.5 * (mu.min() + mu.max())
    m = d * _MANDEL[None, :, None] * _MANDEL[None, None, :]
    n = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) / np.sqrt(3.0)
    bulk = np.einsum("i,mij,j->m", n, m, n) / 3.0
    proj = np.eye(6) - np.outer(n, n)
    dev = np.linalg.eigvalsh(proj @ m @ proj)[:, 1:] / 2.0  # without the hydrostatic mode
    k0 = 0.5 * (bulk.min() + bulk.max())
    mu0 = 0.5 * (dev.min() + dev.max())
    return k0 - 2.0 * mu0 / 3.0, mu0


class FFTElasticProblem(_FFTProblem):
    """Periodic linear elasticity of a voxel image (FFT-preconditioned CG); crystals are
    turned by the orientations of their grains (``grains``: label image)."""

    physics = "elastic"
    nd = 3

    def _element_matrices(self, matrices) -> np.ndarray:
        b = self.element.b_gauss
        k = self.element.weight * np.einsum("gri,mrs,gsj->mij", b, matrices, b, optimize=True)
        return np.ascontiguousarray(0.5 * (k + np.swapaxes(k, 1, 2)))

    def _reference_element(self) -> np.ndarray:
        if self.reference is not None:
            lam, mu = self.reference
        else:
            used = self.table.matrices[self.counts > 0]
            lam, mu = reference_moduli(used, self.table.isotropic)
        return self.element.stiffness(lam, mu)

    def _thermal_loads(self) -> np.ndarray:
        """Element load vectors of a unit change of temperature, ``int B^T D alpha``."""
        el = self.element
        sig = np.einsum("mrs,ms->mr", self.table.matrices, self.table.eigen)
        return el.weight * np.einsum("grc,mr->mc", el.b_gauss, sig)

    def _field(self, x: np.ndarray, dt: float):
        """Periodic fluctuation for the macroscopic strain ``x`` (Voigt order) and the
        change of temperature ``dt``."""
        nx, ny, nz = self.shape
        u_loc = self.element.affine_displacement(x)
        loads = self.element_matrices @ u_loc
        if dt:
            loads = loads - dt * self._thermal_loads()
        loads = np.ascontiguousarray(loads)
        dof = np.arange(3 * nx * ny * nz, dtype=np.int64)
        rhs = kernels.element_load(nx, ny, nz, True, self.mat, dof, 3, loads, dof.size)
        return self._solve(rhs, self._load_scale(loads))

    def _sums(self, w, x, dt) -> np.ndarray:
        el = self.element
        nx, ny, nz = self.shape
        return kernels.elastic_sums(
            nx,
            ny,
            nz,
            True,
            self.mat,
            self.stat,
            len(self.table.phases),
            self.table.matrices,
            self.table.eigen,
            float(dt),
            w,
            el.affine_displacement(x),
            el.b_gauss,
            el.weight,
        )

    def _dual(self, w, x, dt) -> np.ndarray:
        return self._sums(w, x, dt).sum(axis=0)[6:12] / self.box_volume

    def _evaluate(self, loading, w, x, dt, info, fields, t0) -> Solution:
        el = self.element
        nx, ny, nz = self.shape
        sums = self._sums(w, x, dt)
        total = sums.sum(axis=0)
        fld = None
        if fields:
            e_strain, e_stress, e_energy = kernels.elastic_fields(
                nx,
                ny,
                nz,
                True,
                self.mat,
                self.table.matrices,
                self.table.eigen,
                float(dt),
                w,
                el.affine_displacement(x),
                el.b_mean,
                el.b_gauss,
            )
            fld = {
                "strain": e_strain.reshape(*self.shape, 6),
                "stress": e_stress.reshape(*self.shape, 6),
                "energy": e_energy.reshape(self.shape),
            }
        info.solve_s = time.perf_counter() - t0
        return Solution(
            physics="elastic",
            bc=self.bc,
            loading=loading,
            strain=voigt_to_tensor(x, strain=True),
            stress=voigt_to_tensor(total[6:12] / self.box_volume, strain=False),
            energy=float(total[12] / self.box_volume),
            material_fraction=float(total[26] / self.box_volume),
            phases=self._phase_statistics(sums, 13, ("strain", "stress", "energy"), (6, 6, 1)),
            info=info,
            fields=fld,
            support=self.image if fields else None,
        )


class FFTThermalProblem(_FFTProblem):
    """Periodic steady heat conduction of a voxel image (FFT-preconditioned CG), or with
    ``physics`` any other transport phenomenon of a scalar potential
    (:data:`~morphogen.homogenization.materials.TRANSPORT`: ``"electrical"``,
    ``"dielectric"``, ``"magnetic"``, ``"diffusion"``), with the coefficient of the
    materials for it; the phases whose coefficient is zero (insulators) are voids."""

    physics = "thermal"
    nd = 1

    def _element_matrices(self, matrices) -> np.ndarray:
        g = self.element.g_gauss
        k = self.element.weight * np.einsum("gdi,mde,gej->mij", g, matrices, g, optimize=True)
        return np.ascontiguousarray(0.5 * (k + np.swapaxes(k, 1, 2)))

    def _reference_element(self) -> np.ndarray:
        if self.reference is not None:
            k0 = self.reference[0]
        else:
            used = self.table.matrices[self.counts > 0]
            eig = np.linalg.eigvalsh(used)
            k0 = 0.5 * (eig.min() + eig.max())
        return k0 * self.element.k_thermal

    def _field(self, x: np.ndarray, dt: float = 0.0):
        """Periodic fluctuation of temperature for the macroscopic gradient ``x``."""
        nx, ny, nz = self.shape
        t_loc = self.element.affine_temperature(x)
        loads = np.ascontiguousarray(self.element_matrices @ t_loc)
        dof = np.arange(nx * ny * nz, dtype=np.int64)
        rhs = kernels.element_load(nx, ny, nz, True, self.mat, dof, 1, loads, dof.size)
        return self._solve(rhs, self._load_scale(loads))

    def _sums(self, t, x) -> np.ndarray:
        el = self.element
        nx, ny, nz = self.shape
        return kernels.thermal_sums(
            nx,
            ny,
            nz,
            True,
            self.mat,
            self.stat,
            len(self.table.phases),
            self.table.matrices,
            t,
            el.affine_temperature(x),
            el.g_gauss,
            el.weight,
        )

    def _dual(self, t, x, dt=0.0) -> np.ndarray:
        return self._sums(t, x).sum(axis=0)[3:6] / self.box_volume

    def _evaluate(self, loading, t, x, dt, info, fields, t0) -> Solution:
        el = self.element
        nx, ny, nz = self.shape
        sums = self._sums(t, x)
        total = sums.sum(axis=0)
        fld = None
        if fields:
            e_grad, e_flux, e_diss = kernels.thermal_fields(
                nx,
                ny,
                nz,
                True,
                self.mat,
                self.table.matrices,
                t,
                el.affine_temperature(x),
                el.g_mean,
                el.g_gauss,
            )
            fld = {
                "gradient": e_grad.reshape(*self.shape, 3),
                "flux": e_flux.reshape(*self.shape, 3),
                "dissipation": e_diss.reshape(self.shape),
            }
        info.solve_s = time.perf_counter() - t0
        return Solution(
            physics=self.physics,
            bc=self.bc,
            loading=loading,
            strain=np.array(x, dtype=float),
            stress=total[3:6] / self.box_volume,
            energy=float(total[6] / self.box_volume),
            material_fraction=float(total[14] / self.box_volume),
            phases=self._phase_statistics(sums, 7, ("gradient", "flux", "dissipation"), (3, 3, 1)),
            info=info,
            fields=fld,
            support=self.image if fields else None,
        )
