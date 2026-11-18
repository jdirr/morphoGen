"""Small dense algebra of the integration kernels (numba): 3 x 3 tensors, their Voigt
components, the eigen-decomposition of symmetric tensors and the divided differences of
the logarithm (functions of symmetric tensors, Daleckii-Krein), and Gaussian elimination.

Voigt order ``(11, 22, 33, 23, 31, 12)``: strains with engineering shears, stresses with
their tensor components. Second-order tensors of 9 components (deformation gradients,
first Piola-Kirchhoff stresses) are stored row by row: ``iJ -> 3 i + J``.
"""

from __future__ import annotations

import math

from numba import njit

#: Index pairs of the Voigt components
_VI = (0, 1, 2, 1, 2, 0)
_VJ = (0, 1, 2, 2, 0, 1)


@njit(cache=True, inline="always")
def matmul3(a, b, out):
    """``out = a b``."""
    for i in range(3):
        for j in range(3):
            out[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j]


@njit(cache=True, inline="always")
def tmatmul3(a, b, out):
    """``out = a^T b``."""
    for i in range(3):
        for j in range(3):
            out[i, j] = a[0, i] * b[0, j] + a[1, i] * b[1, j] + a[2, i] * b[2, j]


@njit(cache=True, inline="always")
def matmult3(a, b, out):
    """``out = a b^T``."""
    for i in range(3):
        for j in range(3):
            out[i, j] = a[i, 0] * b[j, 0] + a[i, 1] * b[j, 1] + a[i, 2] * b[j, 2]


@njit(cache=True, inline="always")
def strain_voigt(t, out):
    """Engineering Voigt components of a symmetric tensor (a strain)."""
    out[0] = t[0, 0]
    out[1] = t[1, 1]
    out[2] = t[2, 2]
    out[3] = t[1, 2] + t[2, 1]
    out[4] = t[2, 0] + t[0, 2]
    out[5] = t[0, 1] + t[1, 0]


@njit(cache=True, inline="always")
def stress_tensor(v, out):
    """Symmetric tensor of Voigt stress components."""
    out[0, 0] = v[0]
    out[1, 1] = v[1]
    out[2, 2] = v[2]
    out[1, 2] = v[3]
    out[2, 1] = v[3]
    out[2, 0] = v[4]
    out[0, 2] = v[4]
    out[0, 1] = v[5]
    out[1, 0] = v[5]


@njit(cache=True, inline="always")
def elastic_stress(d, strain, out, e6, s6):
    """``out = D : strain`` (tensors) for the elasticity matrix ``d`` (Voigt)."""
    strain_voigt(strain, e6)
    for r in range(6):
        acc = 0.0
        for c in range(6):
            acc += d[r, c] * e6[c]
        s6[r] = acc
    stress_tensor(s6, out)


@njit(cache=True, inline="always")
def project(lv, nv, t):
    """``l . t . n``."""
    acc = 0.0
    for i in range(3):
        acc += lv[i] * (t[i, 0] * nv[0] + t[i, 1] * nv[1] + t[i, 2] * nv[2])
    return acc


@njit(cache=True, inline="always")
def symmetrize(a):
    """A square matrix replaced by its symmetric part, in place (the consistent tangents
    of the points for the conjugate gradients)."""
    n = a.shape[0]
    for r in range(n):
        for c in range(r + 1, n):
            m = 0.5 * (a[r, c] + a[c, r])
            a[r, c] = m
            a[c, r] = m


@njit(cache=True)
def inverse3(a, out):
    """Inverse of a 3 x 3 matrix; returns its determinant."""
    c00 = a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]
    c01 = a[1, 2] * a[2, 0] - a[1, 0] * a[2, 2]
    c02 = a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]
    det = a[0, 0] * c00 + a[0, 1] * c01 + a[0, 2] * c02
    inv = 1.0 / det
    out[0, 0] = c00 * inv
    out[1, 0] = c01 * inv
    out[2, 0] = c02 * inv
    out[0, 1] = (a[0, 2] * a[2, 1] - a[0, 1] * a[2, 2]) * inv
    out[1, 1] = (a[0, 0] * a[2, 2] - a[0, 2] * a[2, 0]) * inv
    out[2, 1] = (a[0, 1] * a[2, 0] - a[0, 0] * a[2, 1]) * inv
    out[0, 2] = (a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1]) * inv
    out[1, 2] = (a[0, 2] * a[1, 0] - a[0, 0] * a[1, 2]) * inv
    out[2, 2] = (a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]) * inv
    return det


@njit(cache=True)
def gauss_solve(a, b, n, m):
    """Solve ``a[:n, :n] x = b[:n, :m]`` in place by Gaussian elimination with partial
    pivoting: ``b`` receives ``x``, ``a`` is overwritten. Returns 1 if ``a`` is singular."""
    for k in range(n):
        piv = k
        big = abs(a[k, k])
        for i in range(k + 1, n):
            if abs(a[i, k]) > big:
                big = abs(a[i, k])
                piv = i
        if not (big > 1e-300) or not math.isfinite(big):
            return 1
        if piv != k:
            for j in range(n):
                tmp = a[k, j]
                a[k, j] = a[piv, j]
                a[piv, j] = tmp
            for j in range(m):
                tmp = b[k, j]
                b[k, j] = b[piv, j]
                b[piv, j] = tmp
        inv = 1.0 / a[k, k]
        for i in range(k + 1, n):
            f = a[i, k] * inv
            if f != 0.0:
                for j in range(k + 1, n):
                    a[i, j] -= f * a[k, j]
                for j in range(m):
                    b[i, j] -= f * b[k, j]
    for k in range(n - 1, -1, -1):
        inv = 1.0 / a[k, k]
        for j in range(m):
            acc = b[k, j]
            for i in range(k + 1, n):
                acc -= a[k, i] * b[i, j]
            b[k, j] = acc * inv
    return 0


@njit(cache=True)
def eigh3(a, w, v):
    """Eigenvalues ``w`` and orthonormal eigenvectors (columns of ``v``) of a symmetric 3 x 3
    matrix, by Jacobi's method (accurate for close eigenvalues); ``a`` is overwritten."""
    for i in range(3):
        for j in range(3):
            v[i, j] = 1.0 if i == j else 0.0
    for _ in range(60):
        off = a[0, 1] * a[0, 1] + a[0, 2] * a[0, 2] + a[1, 2] * a[1, 2]
        scale = a[0, 0] * a[0, 0] + a[1, 1] * a[1, 1] + a[2, 2] * a[2, 2]
        if off <= 1e-36 * scale or off == 0.0:
            break
        for p in range(2):
            for q in range(p + 1, 3):
                apq = a[p, q]
                if apq == 0.0:
                    continue
                theta = (a[q, q] - a[p, p]) / (2.0 * apq)
                if abs(theta) > 1e150:
                    t = 0.5 / theta
                else:
                    t = 1.0 / (abs(theta) + math.sqrt(theta * theta + 1.0))
                    if theta < 0.0:
                        t = -t
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(3):
                    akp = a[k, p]
                    akq = a[k, q]
                    a[k, p] = c * akp - s * akq
                    a[k, q] = s * akp + c * akq
                for k in range(3):
                    apk = a[p, k]
                    aqk = a[q, k]
                    a[p, k] = c * apk - s * aqk
                    a[q, k] = s * apk + c * aqk
                for k in range(3):
                    vkp = v[k, p]
                    vkq = v[k, q]
                    v[k, p] = c * vkp - s * vkq
                    v[k, q] = s * vkp + c * vkq
    for i in range(3):
        w[i] = a[i, i]


@njit(cache=True, inline="always")
def half_log(mu):
    """``f(1 + mu) = 1/2 ln(1 + mu)``."""
    return 0.5 * math.log1p(mu)


@njit(cache=True)
def log_divided(x, y):
    """First divided difference ``f[1 + x, 1 + y]`` of ``f = 1/2 ln`` (``x, y > -1``: the
    eigenvalues of ``C - I``), exact for close and equal arguments."""
    z = (x - y) / (1.0 + y)
    if abs(z) < 1e-4:
        return 0.5 / (1.0 + y) * (1.0 - z * (0.5 - z * (1.0 / 3.0 - 0.25 * z)))
    return 0.5 * math.log1p(z) / (x - y)


@njit(cache=True)
def log_divided2(x, y, z):
    """Second divided difference ``f[1 + x, 1 + y, 1 + z]`` of ``f = 1/2 ln``, symmetric in
    its arguments; a Taylor series about their mean when they are close."""
    lo = min(x, min(y, z))
    hi = max(x, max(y, z))
    mid = x + y + z - lo - hi
    m = 1.0 + (x + y + z) / 3.0
    if hi - lo <= 1e-4 * m:
        d0 = 1.0 + x - m
        d1 = 1.0 + y - m
        d2 = 1.0 + z - m
        h2 = 0.5 * (d0 * d0 + d1 * d1 + d2 * d2)
        h3 = d0 * d1 * d2
        m2 = m * m
        return -0.25 / m2 - h2 / (8.0 * m2 * m2) + h3 / (10.0 * m2 * m2 * m)
    return (log_divided(hi, mid) - log_divided(mid, lo)) / (hi - lo)
