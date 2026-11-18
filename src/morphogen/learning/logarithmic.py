"""The logarithmic strain of finite strain data and the stresses conjugate to it, for the
learned laws of ``strain="logarithmic"`` (Numba kernels).

Such a law sees the logarithmic strain ``H = 1/2 ln C`` (``C = I + 2 E``) of the
Green-Lagrange strain ``E`` of its data, the space of the Hencky points of morphoGen
(Miehe, Apel & Lambrecht 2002), and gives the stress ``T`` conjugate to it; the second
Piola-Kirchhoff stress follows from ``S : dE = T : dH``,

    S = J^T T,    dS/dE = J^T (dT/dH) J + T : d2H/dE2,    J = dH/dE (6 x 6),

with ``J`` and the geometric term ``T : d2H/dE2`` from the first and second divided
differences of ``1/2 ln`` in the eigenbasis of ``C`` (Daleckii-Krein; those of
:mod:`morphogen.homogenization.tensors`, exact for equal eigenvalues). ``H`` is an
isotropic function of ``C``: ``H(R C R^T) = R H(C) R^T``. The strains and stresses are
normalized by the scales of a model (``x = E / s``, ``h = H / s``), Voigt components with
engineering shears for the strains.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from morphogen.homogenization.tensors import (
    eigh3,
    half_log,
    log_divided,
    log_divided2,
    strain_voigt,
    stress_tensor,
)
from morphogen.learning.data import PAIRS


@njit(cache=True)
def _logarithmic(x, scale, h, jac, q, mu, dq):
    """The logarithmic strain ``H = 1/2 ln C`` (``C = I + 2 E``) of a normalized
    Green-Lagrange strain ``x = E / scale`` (Voigt, engineering shears), normalized alike
    (``h``), and its derivative ``jac = dH/dE`` (6 x 6): ``J_ij = w_i sum_ab D_i,ab f[c_a,
    c_b] D_j,ab`` in the eigenbasis of ``C`` (eigenvalues ``mu`` of ``C - I``, vectors the
    columns of ``q``; ``f[.,.]`` the divided differences of ``1/2 ln``), ``D_j = Q^T dC/dE_j
    Q`` kept in ``dq`` (``w_i`` 1/2 for the normal components, 1 for the shears). Returns 1
    where ``C`` is not positive definite."""
    a = np.empty((3, 3))
    for k in range(6):
        i, j = PAIRS[k]
        v = scale * x[k] * (2.0 if k < 3 else 1.0)  # C - I = 2 E
        a[i, j] = v
        a[j, i] = v
    eigh3(a, mu, q)
    for c in range(3):
        if not (mu[c] > -1.0 and math.isfinite(mu[c])):
            return 1
    for i in range(3):
        for j in range(3):
            s = 0.0
            for c in range(3):
                s += q[i, c] * q[j, c] * half_log(mu[c])
            a[i, j] = s
    strain_voigt(a, h)
    for k in range(6):
        h[k] /= scale
    f1 = np.empty((3, 3))
    for c in range(3):
        for d in range(c, 3):
            f1[c, d] = log_divided(mu[c], mu[d])
            f1[d, c] = f1[c, d]
    for k in range(6):
        m, n = PAIRS[k]
        for c in range(3):
            for d in range(3):
                if k < 3:
                    dq[k, c, d] = 2.0 * q[m, c] * q[m, d]
                else:
                    dq[k, c, d] = q[m, c] * q[n, d] + q[n, c] * q[m, d]
    for i in range(6):
        for j in range(i, 6):
            s = 0.0
            for c in range(3):
                for d in range(3):
                    s += dq[i, c, d] * f1[c, d] * dq[j, c, d]
            jac[i, j] = s * (0.5 if i < 3 else 1.0)
            jac[j, i] = s * (0.5 if j < 3 else 1.0)
    return 0


@njit(cache=True)
def _geometric(t, q, mu, dq, scale, out):
    """Adds to ``out`` (6 x 6) the normalized ``T : d2H/dE2`` of the normalized stress ``t``
    conjugate to the logarithmic strain (the Hessian of ``T . H(E)`` at fixed ``T``):
    ``2 sum_abc f[c_a, c_b, c_c] T_ca D_j,ab D_k,bc`` in the eigenbasis of ``C`` (the second
    divided differences of ``1/2 ln``, Daleckii-Krein), times ``scale``."""
    tt = np.empty((3, 3))
    stress_tensor(t, tt)
    tq = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            s = 0.0
            for i in range(3):
                for j in range(3):
                    s += q[i, a] * tt[i, j] * q[j, b]
            tq[a, b] = s
    f2 = np.empty((3, 3, 3))
    for a in range(3):  # symmetric in its arguments: 10 values
        for b in range(a, 3):
            for c in range(b, 3):
                v = log_divided2(mu[a], mu[b], mu[c])
                f2[a, b, c] = v
                f2[a, c, b] = v
                f2[b, a, c] = v
                f2[b, c, a] = v
                f2[c, a, b] = v
                f2[c, b, a] = v
    r = np.empty((6, 3, 3))
    for j in range(6):
        for b in range(3):
            for c in range(3):
                s = 0.0
                for a in range(3):
                    s += f2[a, b, c] * tq[c, a] * dq[j, a, b]
                r[j, b, c] = s
    for j in range(6):
        for k in range(j, 6):
            s = 0.0
            for b in range(3):
                for c in range(3):
                    s += r[j, b, c] * dq[k, b, c]
            out[j, k] += 2.0 * scale * s
            if k != j:
                out[k, j] += 2.0 * scale * s


@njit(cache=True)
def _conjugate(jac, t, kmat, y, out):
    """The stress conjugate to the Green-Lagrange strain of the stress ``t`` conjugate to
    ``H``, ``y = J^T t``, and ``out = J^T K J`` of the tangent ``K = dT/dH``."""
    tmp = np.empty((6, 6))
    for i in range(6):
        s = 0.0
        for k in range(6):
            s += jac[k, i] * t[k]
        y[i] = s
        for j in range(6):
            s = 0.0
            for k in range(6):
                s += kmat[i, k] * jac[k, j]
            tmp[i, j] = s
    for i in range(6):
        for j in range(6):
            s = 0.0
            for k in range(6):
                s += jac[k, i] * tmp[k, j]
            out[i, j] = s


@njit(cache=True)
def _symmetrize(c):
    for r in range(6):
        for k in range(r):
            s = 0.5 * (c[r, k] + c[k, r])
            c[r, k] = s
            c[k, r] = s


@njit(cache=True)
def _measures(x, scale):
    """The normalized logarithmic strains (n x 6) and their derivatives ``dH/dE`` (n x 6 x
    6) of normalized Green-Lagrange strains ``x`` (n x 6); NaN where ``C`` is not positive
    definite."""
    n = x.shape[0]
    h, jac = np.empty((n, 6)), np.empty((n, 6, 6))
    q, mu, dq = np.empty((3, 3)), np.empty(3), np.empty((6, 3, 3))
    for p in range(n):
        if _logarithmic(x[p], scale, h[p], jac[p], q, mu, dq):
            h[p] = np.nan
            jac[p] = np.nan
    return h, jac


@njit(cache=True)
def _pullback(x, t, k, scale, symmetric):
    """The normalized second Piola-Kirchhoff stresses ``J^T t`` (n x 6) and their tangents
    ``J^T k J + T : d2H/dE2`` (n x 6 x 6, normalized) of normalized Green-Lagrange strains
    ``x`` (n x 6), of the normalized stresses ``t`` conjugate to their logarithmic strains
    (n x 6) and of the tangents ``k = dt/dh`` of a law (n x 6 x 6), symmetrized with
    ``symmetric`` (a law of a potential); NaN where ``C`` is not positive definite."""
    n = x.shape[0]
    y, tangent = np.empty((n, 6)), np.empty((n, 6, 6))
    h, jac, q, mu = np.empty(6), np.empty((6, 6)), np.empty((3, 3)), np.empty(3)
    dq = np.empty((6, 3, 3))
    for p in range(n):
        if _logarithmic(x[p], scale, h, jac, q, mu, dq):
            y[p] = np.nan
            tangent[p] = np.nan
            continue
        _conjugate(jac, t[p], k[p], y[p], tangent[p])
        _geometric(t[p], q, mu, dq, scale, tangent[p])
        if symmetric:
            _symmetrize(tangent[p])
    return y, tangent
