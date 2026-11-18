"""Trilinear hexahedra (Q1) on a regular voxel grid.

Every voxel is one 8-node element. The corners follow the VTK/Abaqus order of
:data:`morphogen.mesh.voxel.HEX_CORNERS`; the degrees of freedom of an elastic element
are ordered node by node, ``(u1, u2, u3)`` at corner 0, then corner 1, etc.

Strains and stresses use the Voigt order ``(11, 22, 33, 23, 31, 12)`` with engineering
shear strains (``gamma_23 = 2 eps_23``) and tensor shear stresses, so that
``sigma = D eps`` and ``sigma . eps`` is the energy density.

All the matrices are integrated exactly with 2 x 2 x 2 Gauss points: the integrands are
polynomials of degree at most 2 in each coordinate on a rectangular voxel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from morphogen.mesh.voxel import HEX_CORNERS

VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))
VOIGT_NAMES = ("11", "22", "33", "23", "31", "12")
#: m in D = lambda m m^T + mu diag(2, 2, 2, 1, 1, 1)
VOIGT_TRACE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
SHEAR_WEIGHTS = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])

_GAUSS_1D = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))


def gauss_points() -> np.ndarray:
    """The 8 Gauss points of the reference cube [0, 1]^3."""
    g = _GAUSS_1D
    return np.array([[a, b, c] for c in g for b in g for a in g])


def shape_gradients(xi: np.ndarray, spacing) -> np.ndarray:
    """Gradients (3 x 8) of the 8 shape functions at the local point ``xi`` in [0, 1]^3."""
    h = np.asarray(spacing, dtype=float)
    grad = np.empty((3, 8))
    for a, corner in enumerate(HEX_CORNERS):
        phi = np.where(corner == 1, xi, 1.0 - xi)
        dphi = np.where(corner == 1, 1.0, -1.0)
        for d in range(3):
            others = [phi[e] for e in range(3) if e != d]
            grad[d, a] = dphi[d] / h[d] * others[0] * others[1]
    return grad


def strain_matrix(grad: np.ndarray) -> np.ndarray:
    """Strain-displacement matrix B (6 x 24) built from the shape gradients (3 x 8)."""
    b = np.zeros((6, 24))
    for a in range(8):
        gx, gy, gz = grad[:, a]
        c = 3 * a
        b[0, c] = gx
        b[1, c + 1] = gy
        b[2, c + 2] = gz
        b[3, c + 1], b[3, c + 2] = gz, gy
        b[4, c], b[4, c + 2] = gz, gx
        b[5, c], b[5, c + 1] = gy, gx
    return b


def isotropic_stiffness(lam: float, mu: float) -> np.ndarray:
    """Elasticity matrix D (6 x 6, Voigt, engineering shear strains)."""
    return lam * np.outer(VOIGT_TRACE, VOIGT_TRACE) + mu * np.diag(SHEAR_WEIGHTS)


@dataclass(frozen=True)
class HexElement:
    """Reference matrices of one voxel element of edge lengths ``spacing``.

    The stiffness of an isotropic element is ``lam * k_lambda + mu * k_mu``; the
    conductance of a conducting element is ``k * k_thermal``.
    """

    spacing: tuple[float, float, float]
    volume: float
    weight: float
    b_gauss: np.ndarray
    b_mean: np.ndarray
    k_lambda: np.ndarray
    k_mu: np.ndarray
    g_gauss: np.ndarray
    g_mean: np.ndarray
    k_thermal: np.ndarray

    def stiffness(self, lam: float, mu: float) -> np.ndarray:
        return lam * self.k_lambda + mu * self.k_mu

    def affine_displacement(self, strain_voigt) -> np.ndarray:
        """Nodal displacements (24) of the affine field E x, x measured from corner 0."""
        e = voigt_to_tensor(strain_voigt, strain=True)
        x = HEX_CORNERS * np.asarray(self.spacing)
        return (x @ e.T).ravel()

    def affine_temperature(self, gradient) -> np.ndarray:
        """Nodal temperatures (8) of the affine field G . x, x measured from corner 0."""
        return HEX_CORNERS * np.asarray(self.spacing) @ np.asarray(gradient, dtype=float)


@lru_cache(maxsize=32)
def hex_element(spacing: tuple[float, float, float]) -> HexElement:
    """Matrices of the voxel element (cached by spacing)."""
    h = tuple(float(s) for s in spacing)
    volume = h[0] * h[1] * h[2]
    weight = volume / 8.0
    grads = np.array([shape_gradients(xi, h) for xi in gauss_points()])
    b_gauss = np.array([strain_matrix(g) for g in grads])
    m = np.outer(VOIGT_TRACE, VOIGT_TRACE)
    s = np.diag(SHEAR_WEIGHTS)
    k_lambda = weight * sum(b.T @ m @ b for b in b_gauss)
    k_mu = weight * sum(b.T @ s @ b for b in b_gauss)
    k_thermal = weight * sum(g.T @ g for g in grads)
    for mat in (k_lambda, k_mu, k_thermal, b_gauss, grads):
        mat.setflags(write=False)
    b_mean = b_gauss.mean(axis=0)
    g_mean = grads.mean(axis=0)
    b_mean.setflags(write=False)
    g_mean.setflags(write=False)
    return HexElement(
        spacing=h,
        volume=volume,
        weight=weight,
        b_gauss=b_gauss,
        b_mean=b_mean,
        k_lambda=0.5 * (k_lambda + k_lambda.T),
        k_mu=0.5 * (k_mu + k_mu.T),
        g_gauss=grads,
        g_mean=g_mean,
        k_thermal=0.5 * (k_thermal + k_thermal.T),
    )


def voigt_to_tensor(v, *, strain: bool) -> np.ndarray:
    """3 x 3 tensor of a Voigt vector (engineering shear components if ``strain``)."""
    v = np.asarray(v, dtype=float)
    t = np.empty((3, 3))
    f = 0.5 if strain else 1.0
    for i, (a, b) in enumerate(VOIGT):
        val = v[i] if i < 3 else f * v[i]
        t[a, b] = t[b, a] = val
    return t


def tensor_to_voigt(t, *, strain: bool) -> np.ndarray:
    """Voigt vector of a symmetric 3 x 3 tensor (engineering shears if ``strain``)."""
    t = np.asarray(t, dtype=float)
    if t.shape != (3, 3):
        raise ValueError(f"expected a 3 x 3 tensor, got shape {t.shape}")
    if not np.allclose(t, t.T, rtol=1e-12, atol=1e-14 * max(1.0, float(np.abs(t).max()))):
        raise ValueError("the tensor must be symmetric")
    f = 2.0 if strain else 1.0
    return np.array([t[a, b] if i < 3 else f * t[a, b] for i, (a, b) in enumerate(VOIGT)])
