"""Finite strains of the micromorphic models (total Lagrangian).

*Macroscopic deformation* (as the finite strains of :mod:`morphogen.homogenization.finite`,
Miehe, Apel & Lambrecht 2002): the logarithmic strain ``ε = 1/2 ln C`` of the right
Cauchy-Green tensor ``C = Fᵀ F`` replaces the small strain in the behaviour of a point,
whose stress ``T`` is conjugate to it; the second Piola-Kirchhoff stress is ``S = T : 2
∂ε/∂C``, the first ``P = F S``, and their derivatives follow from the first and second
divided differences of ``1/2 ln`` at the eigenvalues of ``C`` (Daleckii-Krein): the
plastic flow is additive in ``ε``, the update of the point that of small strains, and the
formulation reduces to small strains for small strains. A scalar internal variable ``p``
of the point (a cumulated plastic strain) gets ``∂p/∂F = ∂p/∂ε : ∂ε/∂F`` the same way.

*Micromorphic deformation* (Eringen & Suhubi 1964; Forest & Sievert 2003, Forest 2009):
the microdeformation ``χ = 1 + χ̃`` (``χ̃`` the generalized degrees of freedom, zero at
rest) maps the directions of the microstructure of the reference configuration to the
current one, so that a superposed rotation ``Q`` turns it as ``F`` and its gradient ``G =
Grad χ`` (``G_ijk = χ_ij,k``). The Lagrangian measures ``Υ = χ⁻¹ F`` (the relative
deformation, ``e = Υ - 1``) and ``K = χ⁻¹ G`` (``K_ijk = (χ⁻¹)_ia G_ajk``) do not change
under it (objective); for small strains ``e ≈ ∇u - χ̃`` and ``K ≈ ∇χ̃``, the measures of
small strains. The micromorphic energy is quadratic in them, ``ψ = 1/2 e : D : e + 1/2 K
⋮ A ⋮ K`` with the relative and higher-order moduli of small strains (``s = λ_s tr e 1 +
2μ_s e^sym + 2μ_c e^skew``, ``S_ijk = l_c² (λ δ_ij K_llk + 2μ K_ijk)``); its derivatives by
``F``, ``χ`` and ``G``, ``∂ψ/∂F = χ⁻ᵀ s``, ``∂ψ/∂χ = -χ⁻ᵀ s Υᵀ - χ⁻ᵀ S : Kᵀ`` (contracted on
the two last indices of ``S`` and ``K``) and ``∂ψ/∂G = χ⁻ᵀ S``, are exact, its Hessian their
central differences (the energy is smooth: rational in ``χ``, quadratic in ``F`` and
``G``).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from morphogen.homogenization.tensors import eigh3, half_log, log_divided, log_divided2

#: Mandel components: (11, 22, 33, 23, 31, 12), shears times sqrt(2)
MAP = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]], dtype=np.int64)
W = np.array([1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0)])
PAIRS = np.array([[0, 0], [1, 1], [2, 2], [1, 2], [2, 0], [0, 1]], dtype=np.int64)
#: Components of the micromorphic arguments: F (9), χ (9), G (27)
N_Z = 45


@njit(cache=True)
def hencky(hu, mu, q, eps6) -> int:
    """The logarithmic strain (Mandel ``eps6``) of the displacement gradient ``hu`` (3 x
    3), the eigenvalues ``mu`` of ``C - 1`` and the eigenvectors (columns of ``q``) of
    ``C``; returns 1 if ``C`` is not positive definite."""
    a = np.empty((3, 3))
    for i in range(3):  # C - 1 = H + Hᵀ + Hᵀ H, without cancellation for small strains
        for j in range(3):
            acc = hu[i, j] + hu[j, i]
            for k in range(3):
                acc += hu[k, i] * hu[k, j]
            a[i, j] = acc
    eigh3(a, mu, q)
    for k in range(3):
        if not (mu[k] > -1.0 and math.isfinite(mu[k])):
            return 1
    for m in range(6):
        i, j = PAIRS[m, 0], PAIRS[m, 1]
        acc = 0.0
        for k in range(3):
            acc += q[i, k] * q[j, k] * half_log(mu[k])
        eps6[m] = acc * W[m]
    return 0


@njit(cache=True)
def pull_back(hu, mu, q, t6, tan6, dp6, pk1, tangent, dp9):
    """From the stress ``t6`` conjugate to the logarithmic strain, its tangent ``tan6``
    and the derivative ``dp6`` of a scalar by it (all Mandel; see the module): the first
    Piola-Kirchhoff stress ``pk1`` (3 x 3), its derivative ``tangent`` (9 x 9, rows and
    columns ``3 i + j``) and the derivative ``dp9`` of the scalar by ``F``."""
    f = np.eye(3) + hu
    f1 = np.empty((3, 3))
    f2 = np.empty((3, 3, 3))
    for a in range(3):
        for b in range(3):
            f1[a, b] = log_divided(mu[a], mu[b])
            for c in range(3):
                f2[a, b, c] = log_divided2(mu[a], mu[b], mu[c])
    t = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            t[i, j] = t6[MAP[i, j]] / W[MAP[i, j]]
    tq = q.T @ t @ q
    sq = 2.0 * f1 * tq
    s = q @ sq @ q.T
    p = f @ s
    for i in range(3):
        for j in range(3):
            pk1[i, j] = p[i, j]
    dc = np.empty((3, 3))
    de = np.empty((3, 3))
    d6 = np.empty(6)
    dt = np.empty((3, 3))
    for kl in range(9):
        k, ll = kl // 3, kl % 3
        for i in range(3):  # dC = dFᵀ F + Fᵀ dF for dF = e_k ⊗ e_l
            for j in range(3):
                v = 0.0
                if i == ll:
                    v += f[k, j]
                if j == ll:
                    v += f[k, i]
                dc[i, j] = v
        dcq = q.T @ dc @ q
        de[:, :] = q @ (f1 * dcq) @ q.T
        for m in range(6):
            d6[m] = de[PAIRS[m, 0], PAIRS[m, 1]] * W[m]
        dt6 = tan6 @ d6
        for i in range(3):
            for j in range(3):
                dt[i, j] = dt6[MAP[i, j]] / W[MAP[i, j]]
        dtq = q.T @ dt @ q
        dsq = 2.0 * f1 * dtq
        for a in range(3):
            for b in range(3):
                acc = 0.0
                for c in range(3):
                    acc += f2[a, b, c] * (dcq[a, c] * tq[c, b] + tq[a, c] * dcq[c, b])
                dsq[a, b] += 2.0 * acc
        dp = f @ (q @ dsq @ q.T)
        for j in range(3):
            dp[k, j] += s[ll, j]
        for i in range(3):
            for j in range(3):
                tangent[3 * i + j, kl] = dp[i, j]
        dp9[kl] = dp6 @ d6


@njit(cache=True)
def micro_gradient(z, mp, g) -> float:
    """The micromorphic energy (see the module) at ``z = (F, χ, G)`` (45: ``F_ij`` at ``3 i +
    j``, ``χ_ij`` at ``9 + 3 i + j``, ``G_ijk`` at ``18 + 9 i + 3 j + k``) for the moduli
    ``mp = (l_c², λ, μ, λ_s, μ_s, μ_c)``, its gradient into ``g``."""
    l2, lam_h, mu_h, lam_s, mu_s, mu_c = mp[0], mp[1], mp[2], mp[3], mp[4], mp[5]
    f = np.empty((3, 3))
    chi = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            f[i, j] = z[3 * i + j]
            chi[i, j] = z[9 + 3 * i + j]
    a = np.linalg.inv(chi)
    y = a @ f
    tr = y[0, 0] + y[1, 1] + y[2, 2] - 3.0
    s = np.empty((3, 3))
    psi = 0.0
    for i in range(3):
        for j in range(3):
            eij = y[i, j] - (1.0 if i == j else 0.0)
            eji = y[j, i] - (1.0 if i == j else 0.0)
            s[i, j] = mu_s * (eij + eji) + mu_c * (eij - eji) + (lam_s * tr if i == j else 0.0)
            psi += 0.5 * s[i, j] * eij
    big_k = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for m in range(3):
                acc = 0.0
                for b in range(3):
                    acc += a[i, b] * z[18 + 9 * b + 3 * j + m]
                big_k[i, j, m] = acc
    big_s = np.empty((3, 3, 3))
    for m in range(3):
        trk = big_k[0, 0, m] + big_k[1, 1, m] + big_k[2, 2, m]
        for i in range(3):
            for j in range(3):
                v = 2.0 * mu_h * big_k[i, j, m] + (lam_h * trk if i == j else 0.0)
                big_s[i, j, m] = l2 * v
                psi += 0.5 * big_s[i, j, m] * big_k[i, j, m]
    gf = a.T @ s
    gc = -(gf @ y.T)
    for k in range(3):
        for ll in range(3):
            acc = 0.0
            for i in range(3):
                for j in range(3):
                    for m in range(3):
                        acc += a[i, k] * big_s[i, j, m] * big_k[ll, j, m]
            gc[k, ll] -= acc
    for i in range(3):
        for j in range(3):
            g[3 * i + j] = gf[i, j]
            g[9 + 3 * i + j] = gc[i, j]
    for b in range(3):
        for j in range(3):
            for m in range(3):
                acc = 0.0
                for i in range(3):
                    acc += a[i, b] * big_s[i, j, m]
                g[18 + 9 * b + 3 * j + m] = acc
    return psi


@njit(cache=True)
def micro_hessian(z, mp, g, h) -> float:
    """The energy, its gradient ``g`` (45) and its Hessian ``h`` (45 x 45; central
    differences of the gradient, symmetrized) at ``z`` (see :func:`micro_gradient`)."""
    psi = micro_gradient(z, mp, g)
    gp = np.empty(N_Z)
    gm = np.empty(N_Z)
    zz = z.copy()
    for c in range(N_Z):
        step = 1e-6 * max(1.0, abs(z[c]))
        zz[c] = z[c] + step
        micro_gradient(zz, mp, gp)
        zz[c] = z[c] - step
        micro_gradient(zz, mp, gm)
        zz[c] = z[c]
        for r in range(N_Z):
            h[r, c] = (gp[r] - gm[r]) / (2.0 * step)
    for r in range(N_Z):
        for c in range(r + 1, N_Z):
            v = 0.5 * (h[r, c] + h[c, r])
            h[r, c] = v
            h[c, r] = v
    return psi
