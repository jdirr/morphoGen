"""Homogenized properties of a periodic voxel cell with graded moduli, and their
sensitivities.

Every voxel ``e`` is a trilinear element of the base material with its stiffness (or
conductance) scaled by ``s_e``: ``K = sum_e s_e K0_e``. For each unit macroscopic strain
``E^i`` (temperature gradient in conduction), the periodic fluctuation ``w^i`` solves
``K w^i = -sum_e s_e K0_e u_aff^i`` (FFT-preconditioned conjugate gradients, see
:mod:`morphogen.homogenization.fft`; previous solutions are reused as initial guesses).
With ``u^i = u_aff^i + w^i``, the homogenized tensor and its derivatives are the
mutual energies

    C_ij = (1 / V) sum_e s_e u^i_e . K0 u^j_e,        dC_ij / ds_e = (1 / V) u^i_e . K0 u^j_e

(the problem is self-adjoint: no adjoint solve is needed; Sigmund 1994, Andreassen and
Andreasen 2014). Elastic tensors use the Voigt order 11, 22, 33, 23, 31, 12 with
engineering shear strains.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

from morphogen.homogenization import kernels
from morphogen.homogenization.elements import hex_element
from morphogen.homogenization.fft import FourierPreconditioner
from morphogen.homogenization.materials import Material
from morphogen.homogenization.solvers import NEGLIGIBLE, dot, norm


@njit(parallel=True, cache=True)
def apply_scaled(nx, ny, nz, nd, scale, mat, w, out):
    """out = sum_e scale_e K0 w_e on the periodic node grid (loop over nodes)."""
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
                    s = scale[(ei * ny + ej) * nz + ek]
                    if s == 0.0:
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
                    for r in range(nd):
                        row = ln * nd + r
                        t = 0.0
                        for col in range(size):
                            t += mat[row, col] * we[col]
                        acc[r] += s * t
        for r in range(nd):
            out[n * nd + r] = acc[r]


@njit(parallel=True, cache=True)
def scaled_load(nx, ny, nz, nd, scale, v, out):
    """out = -sum_e scale_e v (v: element vector K0 u_aff, the same in every voxel)."""
    for n in prange(nx * ny * nz):
        k = n % nz
        j = (n // nz) % ny
        i = n // (ny * nz)
        for r in range(nd):
            out[n * nd + r] = 0.0
        for a in range(2):
            ei = i - a if i - a >= 0 else nx - 1
            for b in range(2):
                ej = j - b if j - b >= 0 else ny - 1
                for c in range(2):
                    ek = k - c if k - c >= 0 else nz - 1
                    s = scale[(ei * ny + ej) * nz + ek]
                    if s == 0.0:
                        continue
                    ln = kernels.CORNER_INDEX[a, b, c]
                    for r in range(nd):
                        out[n * nd + r] -= s * v[ln * nd + r]


@njit(parallel=True, cache=True)
def element_energies(nx, ny, nz, nd, mat, w, uaff, out):
    """out[p, e] = u^i_e . K0 u^j_e for the pairs p = (i, j) of load cases (upper
    triangle, row by row); ``w[i]`` are the fluctuations, ``uaff[i]`` the affine element
    vectors."""
    n_cases = w.shape[0]
    size = 8 * nd
    corners = kernels.CORNERS
    for e in prange(nx * ny * nz):
        k = e % nz
        j = (e // nz) % ny
        i = e // (ny * nz)
        u = np.empty((n_cases, size))
        for q in range(8):
            qi = i + corners[q, 0]
            qj = j + corners[q, 1]
            qk = k + corners[q, 2]
            if qi == nx:
                qi = 0
            if qj == ny:
                qj = 0
            if qk == nz:
                qk = 0
            m = (qi * ny + qj) * nz + qk
            for c in range(n_cases):
                for d in range(nd):
                    u[c, q * nd + d] = uaff[c, q * nd + d] + w[c, m * nd + d]
        ku = np.empty((n_cases, size))
        for c in range(n_cases):
            for r in range(size):
                t = 0.0
                for col in range(size):
                    t += mat[r, col] * u[c, col]
                ku[c, r] = t
        p = 0
        for a in range(n_cases):
            for b in range(a, n_cases):
                t = 0.0
                for r in range(size):
                    t += u[a, r] * ku[b, r]
                out[p, e] = t
                p += 1


def _pcg(apply, precondition, rhs, x0, tol, maxiter):
    """Preconditioned conjugate gradient from ``x0``; relative residual to ``rhs``."""
    norm_b = norm(rhs)
    if norm_b == 0.0:
        return np.zeros_like(rhs), 0, 0.0
    x = x0.copy()
    r = rhs - apply(x)
    res = norm(r) / norm_b
    if res <= tol:
        return x, 0, res
    z = precondition(r)
    p = z.copy()
    rz = dot(r, z)
    it = 0
    for it in range(1, maxiter + 1):  # noqa: B007
        ap = apply(p)
        pap = dot(p, ap)
        if pap <= 0.0:
            break
        alpha = rz / pap
        x += alpha * p
        r -= alpha * ap
        res = norm(r) / norm_b
        if res <= tol:
            break
        z = precondition(r)
        rz_new = dot(r, z)
        if rz_new <= 0.0:  # the residual is round-off the preconditioner cancels
            break
        p *= rz_new / rz
        p += z
        rz = rz_new
    return x, it, res


ELASTIC_CASES = 6
THERMAL_CASES = 3


class PeriodicCell:
    """Periodic cell of ``shape`` voxels of edge ``spacing`` made of a base material whose
    moduli are scaled voxel by voxel.

    ``physics`` is ``elastic`` (stiffness tensor 6 x 6) or ``thermal`` (conductivity
    tensor 3 x 3). ``cases`` selects the unit loadings solved (default: all); the tensor
    entries between unsolved cases are NaN.
    """

    def __init__(
        self,
        shape,
        spacing=(1.0, 1.0, 1.0),
        *,
        physics: str = "elastic",
        material: Material | None = None,
        cases=None,
        tol: float = 1e-7,
        maxiter: int | None = None,
    ):
        if physics not in ("elastic", "thermal"):
            raise ValueError("physics: elastic or thermal")
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError("shape: three positive numbers of voxels")
        self.spacing = tuple(float(h) for h in np.broadcast_to(spacing, (3,)))
        self.physics = physics
        self.material = material or Material(young=1.0, poisson=0.3, conductivity=1.0)
        el = hex_element(self.spacing)
        self.element = el
        n_all = ELASTIC_CASES if physics == "elastic" else THERMAL_CASES
        self.cases = tuple(range(n_all)) if cases is None else tuple(sorted(set(cases)))
        if not self.cases or min(self.cases) < 0 or max(self.cases) >= n_all:
            raise ValueError(f"cases: indices among 0..{n_all - 1}")
        self.nd = 3 if physics == "elastic" else 1
        if physics == "elastic":
            lam, mu = self.material.lame
            self.ka, self.kb = el.k_lambda, el.k_mu
            self.coef = (lam, mu)
            self.k0 = np.ascontiguousarray(lam * el.k_lambda + mu * el.k_mu)
            basis = np.eye(6)
            self.uaff = np.array([el.affine_displacement(basis[c]) for c in self.cases])
        else:
            k = self.material.conductivity
            if k is None:
                raise ValueError("thermal problems need the conductivity of the material")
            self.ka, self.kb = el.k_thermal, np.zeros_like(el.k_thermal)
            self.coef = (k, 0.0)
            self.k0 = np.ascontiguousarray(k * el.k_thermal)
            self.uaff = np.array([el.affine_temperature(np.eye(3)[c]) for c in self.cases])
        self.n_voxels = int(np.prod(self.shape))
        self.volume = self.n_voxels * float(np.prod(self.spacing))
        self.tol = float(tol)
        self.maxiter = maxiter or 20 * max(self.shape) + 1000
        self._w = np.zeros((len(self.cases), self.nd * self.n_voxels))
        self.iterations = 0
        self.solve_time = 0.0

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    def pairs(self) -> list[tuple[int, int]]:
        """Pairs of solved cases, in the order of the energies (upper triangle)."""
        return [(a, b) for a in range(self.n_cases) for b in range(a, self.n_cases)]

    def solve(self, scale) -> tuple[np.ndarray, np.ndarray]:
        """Homogenized tensor and the mutual energies ``(n_pairs, n_voxels)``, whose
        products with ``1 / V`` are the derivatives ``dC_ij / ds_e``."""
        s = np.ascontiguousarray(np.asarray(scale, dtype=float).reshape(-1))
        if s.size != self.n_voxels or np.any(s < 0):
            raise ValueError("scale: one non-negative value per voxel")
        nx, ny, nz = self.shape
        nd = self.nd
        t0 = time.perf_counter()
        active = s > 0
        smin = float(s[active].min()) if active.any() else 1.0
        sref = 0.5 * (smin + float(s.max(initial=1.0)))
        pre = FourierPreconditioner(
            self.shape, self.ka, self.kb, sref * self.coef[0], sref * self.coef[1], nd
        )

        def apply(v):
            out = np.empty_like(v)
            apply_scaled(nx, ny, nz, nd, s, self.k0, v, out)
            return out

        rhs = np.empty(nd * self.n_voxels)
        for c in range(self.n_cases):
            load = self.k0 @ self.uaff[c]
            scaled_load(nx, ny, nz, nd, s, load, rhs)
            if norm(rhs) <= NEGLIGIBLE * norm(s) * norm(load):
                # the element loads cancel out (e.g. shear of a laminate): no fluctuation
                w, it = np.zeros_like(rhs), 0
            else:
                w, it, _ = _pcg(apply, pre, rhs, self._w[c], self.tol, self.maxiter)
            w = w.reshape(-1, nd)
            self._w[c] = (w - w.mean(axis=0)).reshape(-1)  # zero-mean fluctuation
            self.iterations += it
        energies = np.empty((len(self.pairs()), self.n_voxels))
        element_energies(nx, ny, nz, nd, self.k0, self._w, self.uaff, energies)
        n_all = ELASTIC_CASES if self.physics == "elastic" else THERMAL_CASES
        tensor = np.full((n_all, n_all), np.nan)
        for p, (a, b) in enumerate(self.pairs()):
            value = float(s @ energies[p]) / self.volume
            tensor[self.cases[a], self.cases[b]] = tensor[self.cases[b], self.cases[a]] = value
        self.solve_time += time.perf_counter() - t0
        return tensor, energies

    def derivative(self, energies: np.ndarray, i: int, j: int) -> np.ndarray:
        """dC_ij / ds_e for all voxels (i, j: case indices of the full tensor)."""
        a, b = sorted((self.cases.index(i), self.cases.index(j)))
        return energies[self.pairs().index((a, b))] / self.volume
