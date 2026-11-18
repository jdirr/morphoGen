"""Tensile tests of laser-architectured sheets: finite elements of a membrane in plane
stress, with large rotations and the flow law of each point.

The sheet is a membrane of quadrilaterals (bilinear, 2 x 2 Gauss points), each Gauss
point split into the layers of the treatment through the thickness, each layer in plane
stress with its own flow law (:mod:`morphogen.laser.materials`): a point treated on one
face only is a stack of layers of different hardness that stretch together (bending is
not modelled).

**Kinematics** (total Lagrangian displacements, corotational stresses). At each Gauss
point the increment ``ΔF = F F_n⁻¹`` of the in-plane deformation gradient splits into
its rotation and stretch, ``ΔF = ΔR ΔU`` (polar decomposition); the strain increment is
``ln ΔU``, the old Cauchy stress is updated in the frame of ``F_n`` then turned by
``ΔR`` (an incrementally objective scheme: exact for rigid rotations, the logarithmic
strains of proportional paths). The thickness strain of each layer comes from its plane
stress, ``h = h0 exp(Σ Δε33)``. The internal forces are ``∫ Bᵀ τ h0 dA0`` with the
Kirchhoff stress ``τ = J σ`` (``J = det F λ3``) and the spatial gradients of the shape
functions; the tangent, ``∫ (Bᵀ J C B + Gᵀ τ G) h0 dA0``, adds the geometric stiffness
of the stresses (the large rotations of the corrugated reinforcements that unbend).

**Plasticity** of each layer: von Mises, isotropic hardening ``σ_y(p)``; the radial
return of the trial stress in 3D with the thickness strain increment ``Δε33`` found by
Newton's method so that ``σ33 = 0``; the consistent tangent of the radial return (Simo &
Hughes 1998), condensed on the in-plane components,
``C_ps = C_ab − C_a3 C_3b / C_33``.

**Tensile test** (:func:`tensile_test`): the grips pull the ends of the specimen along x
(displacement control, lateral displacements of the gripped ends held, or free for a
strip with free ends); a virtual extensometer reads the elongation of the parallel part
(the mean axial displacement of two cross-sections). Results: the engineering curve, the
0.2 % proof stress, the tensile strength and the uniform elongation at the maximum force
(diffuse necking, Considère's criterion of the specimen), the energy absorbed up to it,
and the fields.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange
from scipy import sparse
from scipy.sparse import linalg as splinalg

from morphogen.laser.materials import LUDWIK_OFFSET
from morphogen.laser.specimen import QuadMesh, Specimen

# -- material point --------------------------------------------------------------------


@njit(cache=True)
def _flow(kind, s0, a, b, p, offset):  # pragma: no cover - compiled
    """Flow stress and hardening modulus: 0 Ludwik, 1 Voce, 2 Swift (Hollomon from the
    yield stress ``s0``: ``K (ε0 + p)^n``, ``ε0 = (s0/K)^(1/n)``)."""
    if kind == 0:
        q = p + offset
        return s0 + a * (q**b - offset**b), b * a * q ** (b - 1.0)
    if kind == 2:
        q = (s0 / a) ** (1.0 / b) + p
        return a * q**b, b * a * q ** (b - 1.0)
    e = math.exp(-b * p)
    return s0 + a * (1.0 - e), a * b * e


@njit(cache=True)
def _flowm(law, p, offset):  # pragma: no cover - compiled
    """The mixture ``(1 − w) σ_1(p) + w σ_2(p)`` of the two laws of a point
    (``law``: kind, σ0, a, b of each, then w)."""
    w = law[8]
    s1, h1 = _flow(int(law[0]), law[1], law[2], law[3], p, offset)
    if w == 0.0:
        return s1, h1
    s2, h2 = _flow(int(law[4]), law[5], law[6], law[7], p, offset)
    return (1.0 - w) * s1 + w * s2, (1.0 - w) * h1 + w * h2


@njit(cache=True)
def _return_3d(sig, de, p_old, law, G, K, offset, out, C):  # pragma: no cover
    """Radial return of the trial stress ``sig + C_e de`` (Voigt 11, 22, 33, 23, 13,
    12; engineering shears in ``de``); writes the stress in ``out`` and the consistent
    tangent in ``C``; returns the plastic strain increment."""
    lam = K - 2.0 * G / 3.0
    tr = de[0] + de[1] + de[2]
    for i in range(3):
        out[i] = sig[i] + lam * tr + 2.0 * G * de[i]
    for i in range(3, 6):
        out[i] = sig[i] + G * de[i]
    mean = (out[0] + out[1] + out[2]) / 3.0
    s = np.empty(6)
    for i in range(3):
        s[i] = out[i] - mean
    for i in range(3, 6):
        s[i] = out[i]
    norm = math.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2 + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2))
    q = math.sqrt(1.5) * norm
    sy, h = _flowm(law, p_old, offset)
    for i in range(6):
        for j in range(6):
            C[i, j] = 0.0
    for i in range(3):
        for j in range(3):
            C[i, j] = lam
        C[i, i] = lam + 2.0 * G
    for i in range(3, 6):
        C[i, i] = G
    if q - sy <= 1e-10 * max(sy, 1.0):
        return 0.0
    # Newton on the plastic strain increment, bracketed in [0, q / 3G]
    lo, hi = 0.0, q / (3.0 * G)
    dp = (q - sy) / (3.0 * G + h)
    for _ in range(100):
        if dp <= lo or dp >= hi:
            dp = 0.5 * (lo + hi)
        sy, h = _flowm(law, p_old + dp, offset)
        g = q - 3.0 * G * dp - sy
        if abs(g) <= 1e-12 * max(sy, 1.0):
            break
        if g > 0:
            lo = dp
        else:
            hi = dp
        dp = dp + g / (3.0 * G + h)
    sy, h = _flowm(law, p_old + dp, offset)
    ratio = 3.0 * G * dp / q
    for i in range(6):
        out[i] = (mean if i < 3 else 0.0) + (1.0 - ratio) * s[i]
    theta = 1.0 - ratio
    theta_bar = 1.0 / (1.0 + h / (3.0 * G)) - ratio
    n = np.empty(6)
    for i in range(6):
        n[i] = s[i] / norm
    for i in range(6):
        for j in range(6):
            C[i, j] = 0.0
    for i in range(3):
        for j in range(3):
            C[i, j] = K - 2.0 * G * theta / 3.0
        C[i, i] += 2.0 * G * theta
    for i in range(3, 6):
        C[i, i] = G * theta
    for i in range(6):
        for j in range(6):
            C[i, j] -= 2.0 * G * theta_bar * n[i] * n[j]
    return dp


@njit(cache=True)
def _kinematics(F, Fn):  # pragma: no cover - compiled
    """The increment ``ΔF = F Fn⁻¹`` split into its rotation ``ΔR`` (cosine, sine) and
    the logarithm of its stretch ``ln ΔU`` (11, 22, 12) in the frame of ``Fn``;
    ``ok`` is false if the increment inverts the element."""
    det_n = Fn[0, 0] * Fn[1, 1] - Fn[0, 1] * Fn[1, 0]
    inv00, inv01 = Fn[1, 1] / det_n, -Fn[0, 1] / det_n
    inv10, inv11 = -Fn[1, 0] / det_n, Fn[0, 0] / det_n
    d00 = F[0, 0] * inv00 + F[0, 1] * inv10
    d01 = F[0, 0] * inv01 + F[0, 1] * inv11
    d10 = F[1, 0] * inv00 + F[1, 1] * inv10
    d11 = F[1, 0] * inv01 + F[1, 1] * inv11
    # polar decomposition dF = R U
    th = math.atan2(d10 - d01, d00 + d11)
    c, s = math.cos(th), math.sin(th)
    u00 = c * d00 + s * d10
    u01 = c * d01 + s * d11
    u10 = -s * d00 + c * d10
    u11 = -s * d01 + c * d11
    u01 = 0.5 * (u01 + u10)
    # logarithm of U (symmetric 2 x 2)
    m = 0.5 * (u00 + u11)
    r = math.sqrt(0.25 * (u00 - u11) ** 2 + u01 * u01)
    l1, l2 = m + r, m - r
    if l2 <= 0.0:
        return c, s, 0.0, 0.0, 0.0, False
    if r < 1e-14:
        return c, s, math.log(l1), math.log(l2), 0.0, True
    phi = 0.5 * math.atan2(2.0 * u01, u00 - u11)
    cp, sp = math.cos(phi), math.sin(phi)
    g1, g2 = math.log(l1), math.log(l2)
    return (
        c,
        s,
        g1 * cp * cp + g2 * sp * sp,
        g1 * sp * sp + g2 * cp * cp,
        (g1 - g2) * cp * sp,
        True,
    )


#: The coefficients of Hill's 1948 criterion of an isotropic point (von Mises): F, G, H,
#: N (normalized by G + H = 1) and the angle of its rolling direction (radians).
ISOTROPIC = (0.5, 0.5, 0.5, 1.5, 0.0)


def hill_from_r(r0: float, r45: float, r90: float, angle: float = 0.0) -> tuple:
    """Hill's 1948 coefficients ``(F, G, H, N)`` of a sheet of Lankford coefficients
    ``r0``, ``r45``, ``r90`` (normalized by ``G + H = 1``: the yield stress along the
    rolling direction), and the angle of its rolling direction from the axis of the
    specimen (degrees; radians in the tuple): the anisotropy of a point of a
    :class:`Membrane`."""
    r0, r45, r90 = float(r0), float(r45), float(r90)
    if min(r0, r45, r90) <= 0:
        raise ValueError("hill_from_r: positive r values")
    G = 1.0 / (1.0 + r0)
    H = r0 / (1.0 + r0)
    F = r0 / (r90 * (1.0 + r0))
    N = (r0 + r90) * (1.0 + 2.0 * r45) / (2.0 * r90 * (1.0 + r0))
    return (F, G, H, N, math.radians(angle))


def yld2000_profile(alpha, a: float, angles=(0.0, 45.0, 90.0)) -> dict:
    """The yield stresses (over that along the rolling direction) and the Lankford
    coefficients of Yld2000-2d ``(alpha, a)`` in uniaxial tension at ``angles``
    (degrees), and in equibiaxial tension (``"b"``): ``{"stress": [...], "r": [...],
    "stress_b", "r_b"}``."""
    al = np.asarray(alpha, dtype=float)
    T = np.eye(3)
    n = np.empty(3)
    out = {"stress": [], "r": []}
    s0 = _yld_eq(np.array([1.0, 0.0, 0.0]), al, float(a), T, n)
    for ang in np.atleast_1d(angles):
        th = math.radians(float(ang))
        c, s = math.cos(th), math.sin(th)
        seq = _yld_eq(np.array([c * c, s * s, c * s]), al, float(a), T, n)
        out["stress"].append(s0 / seq)
        out["r"].append(-(n[0] * s * s + n[1] * c * c - n[2] * c * s) / (n[0] + n[1]))
    seq = _yld_eq(np.array([1.0, 1.0, 0.0]), al, float(a), T, n)
    out["stress_b"] = s0 / seq
    out["r_b"] = n[1] / n[0]
    return out


def yld2000_from_data(
    s0: float,
    s45: float,
    s90: float,
    sb: float,
    r0: float,
    r45: float,
    r90: float,
    rb: float = 1.0,
    a: float = 6.0,
    angle: float = 0.0,
) -> tuple:
    """Barlat's Yld2000-2d coefficients α1 ... α8 fitted exactly to the yield stresses
    ``s0``, ``s45``, ``s90`` of uniaxial tension at 0°, 45°, 90° from the rolling
    direction, ``sb`` of equibiaxial tension, the Lankford coefficients ``r0``, ``r45``,
    ``r90`` and the biaxial ratio ``rb`` (Barlat et al. 2003; the stresses in any unit,
    the equivalent stress being normalized to ``s0``), with the exponent ``a`` (6 for
    cubic-centred metals, 8 for face-centred), and the angle of the rolling direction
    from the axis of the specimen (degrees): the anisotropy of a point of a
    :class:`Membrane` (10 values)."""
    from scipy.optimize import least_squares

    data = np.array([1.0, s0 / s45, s0 / s90, s0 / sb, r0, r45, r90, rb], dtype=float)
    if not np.all(np.isfinite(data)) or min(s0, s45, s90, sb, r0, r45, r90, rb) <= 0:
        raise ValueError("yld2000_from_data: positive stresses and r values")
    T = np.eye(3)
    n = np.empty(3)
    tests = []
    for ang in (0.0, 45.0, 90.0):
        th = math.radians(ang)
        c, s = math.cos(th), math.sin(th)
        tests.append((np.array([c * c, s * s, c * s]), c, s))

    def residuals(x):
        out = np.empty(8)
        for k, (u, _c, _s) in enumerate(tests):
            out[k] = _yld_eq(u, x, a, T, n) - data[k]
        out[3] = _yld_eq(np.array([1.0, 1.0, 0.0]), x, a, T, n) - data[3]
        for k, (u, c, s) in enumerate(tests):
            _yld_eq(u, x, a, T, n)
            out[4 + k] = -(n[0] * s * s + n[1] * c * c - n[2] * c * s) / (n[0] + n[1]) - data[4 + k]
        _yld_eq(np.array([1.0, 1.0, 0.0]), x, a, T, n)
        out[7] = n[1] / n[0] - data[7]
        return out

    sol = least_squares(residuals, np.ones(8), method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    worst = float(np.max(np.abs(sol.fun)))
    if not worst < 1e-8:
        raise ValueError(f"yld2000_from_data: no exact fit (residual {worst:.1e})")
    return (*(float(v) for v in sol.x), float(a), math.radians(angle))


@njit(cache=True)
def _isotropic(aniso):  # pragma: no cover - compiled
    return (
        abs(aniso[0] - 0.5) < 1e-12
        and abs(aniso[1] - 0.5) < 1e-12
        and abs(aniso[2] - 0.5) < 1e-12
        and abs(aniso[3] - 1.5) < 1e-12
    )


@njit(cache=True)
def _inv3(A, out):  # pragma: no cover - compiled
    a, b, c = A[0, 0], A[0, 1], A[0, 2]
    d, e, f = A[1, 0], A[1, 1], A[1, 2]
    g, h, i = A[2, 0], A[2, 1], A[2, 2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    out[0, 0], out[0, 1], out[0, 2] = (
        (e * i - f * h) / det,
        (c * h - b * i) / det,
        (b * f - c * e) / det,
    )
    out[1, 0], out[1, 1], out[1, 2] = (
        (f * g - d * i) / det,
        (a * i - c * g) / det,
        (c * d - a * f) / det,
    )
    out[2, 0], out[2, 1], out[2, 2] = (
        (d * h - e * g) / det,
        (b * g - a * h) / det,
        (a * e - b * d) / det,
    )
    return det


@njit(cache=True)
def _hill_matrix(aniso, phi, P):  # pragma: no cover - compiled
    """``σ_eq² = σᵀ P σ`` (σ = 11, 22, 12) of Hill's 1948 criterion in plane stress, its
    rolling direction at the angle ``phi`` from the axis 1."""
    pm = np.zeros((3, 3))
    pm[0, 0] = aniso[1] + aniso[2]
    pm[0, 1] = pm[1, 0] = -aniso[2]
    pm[1, 1] = aniso[0] + aniso[2]
    pm[2, 2] = 2.0 * aniso[3]
    c, s = math.cos(phi), math.sin(phi)
    T = np.empty((3, 3))  # the components in the axes of the rolling direction
    T[0, 0], T[0, 1], T[0, 2] = c * c, s * s, 2.0 * c * s
    T[1, 0], T[1, 1], T[1, 2] = s * s, c * c, -2.0 * c * s
    T[2, 0], T[2, 1], T[2, 2] = -c * s, c * s, c * c - s * s
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                for m in range(3):
                    acc += T[k, i] * pm[k, m] * T[m, j]
            P[i, j] = acc


@njit(cache=True)
def _hill_state(dg, tr, C, P, out):  # pragma: no cover - compiled
    """The stress ``(I + dg C P)⁻¹ tr`` in ``out``; returns its equivalent stress."""
    A = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += C[i, k] * P[k, j]
            A[i, j] = (1.0 if i == j else 0.0) + dg * acc
    Ai = np.empty((3, 3))
    _inv3(A, Ai)
    for i in range(3):
        out[i] = Ai[i, 0] * tr[0] + Ai[i, 1] * tr[1] + Ai[i, 2] * tr[2]
    q = 0.0
    for i in range(3):
        for j in range(3):
            q += out[i] * P[i, j] * out[j]
    return math.sqrt(max(q, 0.0))


@njit(cache=True)
def _hill_update(
    sig_old, de, p_old, law, aniso, phi, E, nu, offset, out, Cp, mloc
):  # pragma: no cover
    """The plane-stress return mapping of Hill's 1948 criterion (associated flow
    ``Δε_p = Δγ P σ``, ``Δp = Δγ σ_eq``; de Souza Neto, Perić & Owen 2008, ch. 9-10):
    the stress (11, 22, 12) in ``out``, the consistent tangent in ``Cp``, ``d ln J / dε``
    in ``mloc``; returns ``(Δp, Δε33)``."""
    P = np.empty((3, 3))
    _hill_matrix(aniso, phi, P)
    k = E / (1.0 - nu * nu)
    C = np.zeros((3, 3))
    C[0, 0] = C[1, 1] = k
    C[0, 1] = C[1, 0] = k * nu
    C[2, 2] = 0.5 * k * (1.0 - nu)
    tr = np.empty(3)
    for i in range(3):
        tr[i] = sig_old[i] + C[i, 0] * de[0] + C[i, 1] * de[1] + C[i, 2] * de[2]
    q = 0.0
    for i in range(3):
        for j in range(3):
            q += tr[i] * P[i, j] * tr[j]
    q = math.sqrt(max(q, 0.0))
    sy, h = _flowm(law, p_old, offset)
    if q - sy <= 1e-10 * max(sy, 1.0):  # elastic
        for i in range(3):
            out[i] = tr[i]
            for j in range(3):
                Cp[i, j] = C[i, j]
        r = nu / (1.0 - nu)
        mloc[0] = mloc[1] = 1.0 - r
        mloc[2] = 0.0
        return 0.0, -r * (de[0] + de[1])
    # Newton on the multiplier, bracketed (the residual falls with it)
    G = 0.5 * E / (1.0 + nu)
    lo, hi = 0.0, 2.0 * (q - sy) / (3.0 * G * max(sy, 1.0))
    sig = np.empty(3)
    for _ in range(60):
        seq = _hill_state(hi, tr, C, P, sig)
        if seq - _flowm(law, p_old + hi * seq, offset)[0] < 0.0:
            break
        lo, hi = hi, 2.0 * hi
    dg = 0.5 * (lo + hi)
    for _ in range(80):
        seq = _hill_state(dg, tr, C, P, sig)
        res = seq - _flowm(law, p_old + dg * seq, offset)[0]
        if abs(res) <= 1e-11 * max(sy, 1.0):
            break
        if res > 0.0:
            lo = dg
        else:
            hi = dg
        step = 1e-6 * max(dg, 1e-12)
        seq2 = _hill_state(dg + step, tr, C, P, out)
        res2 = seq2 - _flowm(law, p_old + (dg + step) * seq2, offset)[0]
        slope = (res2 - res) / step
        new = dg - res / slope if slope < 0.0 else 0.5 * (lo + hi)
        if new <= lo or new >= hi:
            new = 0.5 * (lo + hi)
        dg = new
    seq = _hill_state(dg, tr, C, P, sig)
    dp = dg * seq
    sy, h = _flowm(law, p_old + dp, offset)
    # the consistent tangent: E_xi = (C^-1 + dg P)^-1, n = P sigma
    Ci = np.empty((3, 3))
    _inv3(C, Ci)
    for i in range(3):
        for j in range(3):
            Ci[i, j] += dg * P[i, j]
    Ex = np.empty((3, 3))
    _inv3(Ci, Ex)
    n = np.empty(3)
    for i in range(3):
        n[i] = P[i, 0] * sig[0] + P[i, 1] * sig[1] + P[i, 2] * sig[2]
    En = np.empty(3)
    for i in range(3):
        En[i] = Ex[i, 0] * n[0] + Ex[i, 1] * n[1] + Ex[i, 2] * n[2]
    a = n[0] * En[0] + n[1] * En[1] + n[2] * En[2]
    beta = max(1.0 - h * dg, 1e-6)
    coef = beta / (h * seq * seq + a * beta)
    for i in range(3):
        out[i] = sig[i]
        for j in range(3):
            Cp[i, j] = Ex[i, j] - coef * En[i] * En[j]
    # the thickness: elastic (plane stress) and plastic (incompressible) parts
    de33 = -nu / E * (sig[0] - sig_old[0] + sig[1] - sig_old[1]) - dg * (n[0] + n[1])
    for j in range(3):
        d33 = -nu / E * (Cp[0, j] + Cp[1, j]) - (n[0] + n[1]) * coef * En[j]
        for i in range(2):
            for m in range(3):
                d33 -= dg * P[i, m] * Cp[m, j]
        mloc[j] = (1.0 if j < 2 else 0.0) + d33
    return dp, de33


#: The isotropic point of Barlat's Yld2000-2d criterion: α1 ... α8 = 1, the exponent 2
#: (von Mises), the angle of the rolling direction (radians).
YLD_ISOTROPIC = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0)


@njit(cache=True)
def _yld_phi(sx, sy, sxy, al, a, g):  # pragma: no cover - compiled
    """Barlat's Yld2000-2d function ``φ = |X'1 − X'2|^a + |2X''2 + X''1|^a + |2X''1 +
    X''2|^a`` (``X' = L' σ``, ``X'' = L'' σ``, their principal values; Barlat et al.
    2003) of the plane stress ``(sx, sy, sxy)`` in the axes of the rolling direction, its
    gradient ``(∂φ/∂sx, ∂φ/∂sy, ∂φ/∂sxy)`` in ``g``; ``al``: α1 ... α8."""
    a1, a2, a3, a4, a5, a6, a7, a8 = al[0], al[1], al[2], al[3], al[4], al[5], al[6], al[7]
    l00, l01 = 2.0 * a1 / 3.0, -a1 / 3.0
    l10, l11 = -a2 / 3.0, 2.0 * a2 / 3.0
    x0 = l00 * sx + l01 * sy
    x1 = l10 * sx + l11 * sy
    x2 = a7 * sxy
    dx = x0 - x1
    d1 = math.sqrt(dx * dx + 4.0 * x2 * x2)
    phi = d1**a
    if d1 > 0.0:
        f = a * d1 ** (a - 2.0)
    elif a <= 2.0 + 1e-12:
        f = a
    else:
        f = 0.0
    gx0, gx1, gx2 = f * dx, -f * dx, 4.0 * f * x2
    g[0] = l00 * gx0 + l10 * gx1
    g[1] = l01 * gx0 + l11 * gx1
    g[2] = a7 * gx2
    m00 = (-2.0 * a3 + 2.0 * a4 + 8.0 * a5 - 2.0 * a6) / 9.0
    m01 = (a3 - 4.0 * a4 - 4.0 * a5 + 4.0 * a6) / 9.0
    m10 = (4.0 * a3 - 4.0 * a4 - 4.0 * a5 + a6) / 9.0
    m11 = (-2.0 * a3 + 8.0 * a4 + 2.0 * a5 - 2.0 * a6) / 9.0
    y0 = m00 * sx + m01 * sy
    y1 = m10 * sx + m11 * sy
    y2 = a8 * sxy
    S = y0 + y1
    dy = y0 - y1
    D = math.sqrt(dy * dy + 4.0 * y2 * y2)
    A = 0.5 * (3.0 * S - D)
    B = 0.5 * (3.0 * S + D)
    phi += abs(A) ** a + abs(B) ** a
    dA = a * abs(A) ** (a - 1.0) * (1.0 if A >= 0.0 else -1.0)
    dB = a * abs(B) ** (a - 1.0) * (1.0 if B >= 0.0 else -1.0)
    dS = 1.5 * (dA + dB)
    if D > 1e-8 * (abs(y0) + abs(y1) + abs(y2) + 1e-300):
        k = 0.5 * (dB - dA) / D
    else:  # the limit of (dB - dA) / 2D as the principal values meet
        k = 0.5 * a * (a - 1.0) * abs(1.5 * S) ** (a - 2.0)
    gy0 = dS + k * dy
    gy1 = dS - k * dy
    gy2 = 4.0 * k * y2
    g[0] += m00 * gy0 + m10 * gy1
    g[1] += m01 * gy0 + m11 * gy1
    g[2] += a8 * gy2
    return phi


@njit(cache=True)
def _yld_axes(phi, T):  # pragma: no cover - compiled
    """The change of the stress components (11, 22, 12) to the axes of the rolling
    direction at ``phi`` (radians) from the axis 1."""
    c, s = math.cos(phi), math.sin(phi)
    T[0, 0], T[0, 1], T[0, 2] = c * c, s * s, 2.0 * c * s
    T[1, 0], T[1, 1], T[1, 2] = s * s, c * c, -2.0 * c * s
    T[2, 0], T[2, 1], T[2, 2] = -c * s, c * s, c * c - s * s


@njit(cache=True)
def _yld_eq(sig, al, a, T, n):  # pragma: no cover - compiled
    """The equivalent stress ``(φ/2)^(1/a)`` of Yld2000-2d (homogeneous of degree one:
    the yield stress along the rolling direction when the α are fitted to it) and its
    gradient ``n`` (engineering shear: ``Δε_p = Δγ n``, ``σ·n = σ_eq``)."""
    sm0 = T[0, 0] * sig[0] + T[0, 1] * sig[1] + T[0, 2] * sig[2]
    sm1 = T[1, 0] * sig[0] + T[1, 1] * sig[1] + T[1, 2] * sig[2]
    sm2 = T[2, 0] * sig[0] + T[2, 1] * sig[1] + T[2, 2] * sig[2]
    gm = np.empty(3)
    phi = _yld_phi(sm0, sm1, sm2, al, a, gm)
    if phi <= 0.0:
        n[0] = n[1] = n[2] = 0.0
        return 0.0
    seq = (0.5 * phi) ** (1.0 / a)
    fac = seq / (a * phi)
    for i in range(3):
        n[i] = fac * (T[0, i] * gm[0] + T[1, i] * gm[1] + T[2, i] * gm[2])
    return seq


@njit(cache=True)
def _yld_hessian(sig, al, a, T, N):  # pragma: no cover - compiled
    """``∂n/∂σ`` by central differences of the analytical gradient, symmetrized."""
    h = 1e-6 * (abs(sig[0]) + abs(sig[1]) + abs(sig[2]) + 1e-3)
    sp = np.empty(3)
    npl = np.empty(3)
    nmi = np.empty(3)
    for j in range(3):
        for i in range(3):
            sp[i] = sig[i]
        sp[j] = sig[j] + h
        _yld_eq(sp, al, a, T, npl)
        sp[j] = sig[j] - h
        _yld_eq(sp, al, a, T, nmi)
        for i in range(3):
            N[i, j] = (npl[i] - nmi[i]) / (2.0 * h)
    for i in range(3):
        for j in range(i + 1, 3):
            m = 0.5 * (N[i, j] + N[j, i])
            N[i, j] = m
            N[j, i] = m


@njit(cache=True)
def _yld_return(tr, p_old, law, al, a, T, Ci, offset, sig, n, N):  # pragma: no cover
    """The closest-point return of Yld2000-2d from the trial stress ``tr``: Newton's
    method on ``C⁻¹(σ − σ*) + Δγ n(σ) = 0``, ``σ_eq(σ) = σ_y(p + Δγ)`` with a
    backtracking line search on the residual (the large curvature of the biaxial region:
    Zhang et al. 2018); returns ``(Δγ, converged)``, the stress in ``sig``."""
    seq = _yld_eq(tr, al, a, T, n)
    sy, h = _flowm(law, p_old, offset)
    scale = max(sy, 1.0)
    C = np.empty((3, 3))
    _inv3(Ci, C)
    Cn = np.empty(3)
    for i in range(3):
        Cn[i] = C[i, 0] * n[0] + C[i, 1] * n[1] + C[i, 2] * n[2]
    nCn = n[0] * Cn[0] + n[1] * Cn[1] + n[2] * Cn[2]
    dg = max((seq - sy) / (nCn + max(h, 0.0)), 0.0)
    for i in range(3):
        sig[i] = tr[i] - dg * Cn[i]
    J = np.empty((4, 4))
    R = np.empty(4)
    st = np.empty(3)
    nt = np.empty(3)
    Et = C[0, 0]
    dgt = dg
    for _ in range(80):
        seq = _yld_eq(sig, al, a, T, n)
        syk, hk = _flowm(law, p_old + dg, offset)
        err = 0.0
        for i in range(3):
            R[i] = (
                Ci[i, 0] * (sig[0] - tr[0])
                + Ci[i, 1] * (sig[1] - tr[1])
                + Ci[i, 2] * (sig[2] - tr[2])
                + dg * n[i]
            )
            err += (R[i] * Et) ** 2
        R[3] = seq - syk
        err = math.sqrt(err + R[3] ** 2) / scale
        if err < 1e-10:
            return dg, True
        _yld_hessian(sig, al, a, T, N)
        for i in range(3):
            for j in range(3):
                J[i, j] = Ci[i, j] + dg * N[i, j]
            J[i, 3] = n[i]
            J[3, i] = n[i]
        J[3, 3] = -hk
        dx = np.linalg.solve(J, -R)
        t = 1.0
        accepted = False
        for _ls in range(30):
            dgt = dg + t * dx[3]
            if dgt >= 0.0:
                for i in range(3):
                    st[i] = sig[i] + t * dx[i]
                seqt = _yld_eq(st, al, a, T, nt)
                syt = _flowm(law, p_old + dgt, offset)[0]
                errt = 0.0
                for i in range(3):
                    ri = (
                        Ci[i, 0] * (st[0] - tr[0])
                        + Ci[i, 1] * (st[1] - tr[1])
                        + Ci[i, 2] * (st[2] - tr[2])
                        + dgt * nt[i]
                    )
                    errt += (ri * Et) ** 2
                errt = math.sqrt(errt + (seqt - syt) ** 2) / scale
                if errt < (1.0 - 1e-4 * t) * err:
                    accepted = True
                    break
            t *= 0.5
        if not accepted:
            return dg, False
        for i in range(3):
            sig[i] = st[i]
        dg = dgt
    return dg, False


@njit(cache=True)
def _yld_update(
    sig_old, de, p_old, law, aniso, phi, E, nu, offset, out, Cp, mloc
):  # pragma: no cover
    """The plane-stress return mapping of Yld2000-2d (associated flow ``Δε_p = Δγ n``,
    ``Δp = Δγ``; the increment split in substeps if Newton's method fails): the stress
    (11, 22, 12) in ``out``, the consistent tangent ``Ξ − (Ξn)(Ξn)ᵀ/(nᵀΞn + h)``,
    ``Ξ = (C⁻¹ + Δγ ∂n/∂σ)⁻¹`` (of the last substep), in ``Cp``, ``d ln J / dε`` in
    ``mloc``; returns ``(Δp, Δε33)``."""
    al = aniso[:8]
    a = aniso[8]
    T = np.empty((3, 3))
    _yld_axes(phi, T)
    k = E / (1.0 - nu * nu)
    C = np.zeros((3, 3))
    C[0, 0] = C[1, 1] = k
    C[0, 1] = C[1, 0] = k * nu
    C[2, 2] = 0.5 * k * (1.0 - nu)
    Ci = np.empty((3, 3))
    _inv3(C, Ci)
    n = np.empty(3)
    N = np.empty((3, 3))
    sig = np.empty(3)
    start = np.empty(3)
    tr = np.empty(3)
    nsub = 1
    p_cur = p_old
    de33p = 0.0
    dg = 0.0
    last_plastic = False
    while True:
        for i in range(3):
            start[i] = sig_old[i]
        p_cur = p_old
        de33p = 0.0
        dg = 0.0
        last_plastic = False
        ok = True
        for _sub in range(nsub):
            for i in range(3):
                tr[i] = start[i] + (C[i, 0] * de[0] + C[i, 1] * de[1] + C[i, 2] * de[2]) / nsub
            q = _yld_eq(tr, al, a, T, n)
            sy = _flowm(law, p_cur, offset)[0]
            if q - sy <= 1e-10 * max(sy, 1.0):  # an elastic substep
                for i in range(3):
                    start[i] = tr[i]
                last_plastic = False
                continue
            dg, conv = _yld_return(tr, p_cur, law, al, a, T, Ci, offset, sig, n, N)
            if not conv:
                ok = False
                break
            de33p -= dg * (n[0] + n[1])  # incompressible
            p_cur += dg
            for i in range(3):
                start[i] = sig[i]
            last_plastic = True
        if ok or nsub >= 64:
            break
        nsub *= 2
    de33 = -nu / E * (start[0] - sig_old[0] + start[1] - sig_old[1]) + de33p
    if not last_plastic:
        for i in range(3):
            out[i] = start[i]
            for j in range(3):
                Cp[i, j] = C[i, j]
        r = nu / (1.0 - nu)
        mloc[0] = mloc[1] = 1.0 - r
        mloc[2] = 0.0
        return p_cur - p_old, de33
    _yld_eq(start, al, a, T, n)
    _yld_hessian(start, al, a, T, N)
    h = _flowm(law, p_cur, offset)[1]
    A = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            A[i, j] = Ci[i, j] + dg * N[i, j]
    Xi = np.empty((3, 3))
    _inv3(A, Xi)
    Xn = np.empty(3)
    for i in range(3):
        Xn[i] = Xi[i, 0] * n[0] + Xi[i, 1] * n[1] + Xi[i, 2] * n[2]
    denom = n[0] * Xn[0] + n[1] * Xn[1] + n[2] * Xn[2] + max(h, 0.0)
    for i in range(3):
        out[i] = start[i]
        for j in range(3):
            Cp[i, j] = Xi[i, j] - Xn[i] * Xn[j] / denom
    for j in range(3):
        d33 = -nu / E * (Cp[0, j] + Cp[1, j]) - (n[0] + n[1]) * Xn[j] / denom
        for i in range(2):
            for m in range(3):
                d33 -= dg * N[i, m] * Cp[m, j]
        mloc[j] = (1.0 if j < 2 else 0.0) + d33
    return p_cur - p_old, de33


@njit(cache=True)
def _point(F, Fn, sig_old, p_old, lam_old, law, aniso, E, nu, offset, sig_out, Cps, mvec):
    """Update of a layer of a Gauss point from ``Fn`` to ``F`` (2 x 2): the new in-plane
    Cauchy stress (11, 22, 12) in ``sig_out``, the condensed spatial tangent in ``Cps``,
    the derivative of ``ln J`` (in-plane trace plus thickness strain) in ``mvec``;
    returns ``(p, λ3, ok)``. ``aniso``: Hill's coefficients and the rolling direction of
    the point (5 values; :data:`ISOTROPIC`: von Mises, by the 3D radial return), or the
    coefficients α1 ... α8 of Yld2000-2d, its exponent and the rolling direction (10;
    :data:`YLD_ISOTROPIC`: von Mises)."""
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    c, s, e00, e11, e01, ok = _kinematics(F, Fn)
    if not ok:
        sig_out[:] = 0.0
        return p_old, lam_old, False
    Cp = np.empty((3, 3))
    mloc = np.empty(3)
    outv = np.empty(3)
    yld = aniso.shape[0] == 10
    if yld:
        iso = abs(aniso[8] - 2.0) < 1e-12
        for i in range(8):
            if abs(aniso[i] - 1.0) > 1e-12:
                iso = False
    else:
        iso = _isotropic(aniso)
    if iso:
        sig3 = np.zeros(6)
        sig3[0], sig3[1], sig3[5] = sig_old[0], sig_old[1], sig_old[2]
        de = np.zeros(6)
        de[0], de[1], de[5] = e00, e11, 2.0 * e01
        # thickness strain increment: Newton on sigma_33 = 0, bracketed
        de[2] = -nu / (1.0 - nu) * (e00 + e11)
        out = np.empty(6)
        C = np.empty((6, 6))
        lo, hi = -1.0, 1.0
        dp = 0.0
        ok = False
        for _ in range(60):
            dp = _return_3d(sig3, de, p_old, law, G, K, offset, out, C)
            f = out[2]
            scale = abs(out[0]) + abs(out[1]) + abs(out[5]) + 1.0
            if abs(f) <= 1e-10 * scale:
                ok = True
                break
            if f > 0:
                hi = min(hi, de[2])
            else:
                lo = max(lo, de[2])
            step = de[2] - f / C[2, 2]
            if step <= lo or step >= hi:
                step = 0.5 * (lo + hi)
            de[2] = step
        # condensed tangent on (11, 22, 12), and d(ln J)/d(eps): 1 + d(eps33)/d(eps)
        idx = (0, 1, 5)
        for i in range(3):
            for j in range(3):
                Cp[i, j] = C[idx[i], idx[j]] - C[idx[i], 2] * C[2, idx[j]] / C[2, 2]
            mloc[i] = (1.0 if i < 2 else 0.0) - C[2, idx[i]] / C[2, 2]
        outv[0], outv[1], outv[2] = out[0], out[1], out[5]
        de33 = de[2]
    else:
        # the rolling direction turns with the material: the rotation of Fn
        turn = math.atan2(Fn[1, 0] - Fn[0, 1], Fn[0, 0] + Fn[1, 1])
        dvec = np.empty(3)
        dvec[0], dvec[1], dvec[2] = e00, e11, 2.0 * e01
        if yld:
            dp, de33 = _yld_update(
                sig_old, dvec, p_old, law, aniso, aniso[9] + turn, E, nu, offset, outv, Cp, mloc
            )
        else:
            dp, de33 = _hill_update(
                sig_old, dvec, p_old, law, aniso, aniso[4] + turn, E, nu, offset, outv, Cp, mloc
            )
        ok = True
    # turn the stress and the tangent by the rotation of the increment
    s00, s11, s01 = outv[0], outv[1], outv[2]
    sig_out[0] = c * c * s00 - 2.0 * c * s * s01 + s * s * s11
    sig_out[1] = s * s * s00 + 2.0 * c * s * s01 + c * c * s11
    sig_out[2] = c * s * (s00 - s11) + (c * c - s * s) * s01
    T = np.empty((3, 3))
    T[0, 0], T[0, 1], T[0, 2] = c * c, s * s, -2.0 * c * s
    T[1, 0], T[1, 1], T[1, 2] = s * s, c * c, 2.0 * c * s
    T[2, 0], T[2, 1], T[2, 2] = c * s, -c * s, c * c - s * s
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                for m2 in range(3):
                    acc += T[i, k] * Cp[k, m2] * T[j, m2]
            Cps[i, j] = acc
    for i in range(3):
        acc = 0.0
        for k in range(3):
            acc += T[i, k] * mloc[k]
        mvec[i] = acc
    return p_old + dp, lam_old * math.exp(de33), ok


@njit(parallel=True, cache=True)
def _elements(
    coords,
    quads,
    dndX,
    w0,
    u,
    Fn,
    sig_n,
    p_n,
    lam_n,
    law,
    aniso,
    E,
    nu,
    offset,
    h0,
    Ke,
    fe,
    sig,
    p,
    lam,
    Fg,
    bad,
):  # pragma: no cover - compiled
    """Element stiffness matrices and internal forces (8 dofs: x0, y0, x1, y1...), the
    states of the Gauss points and layers at the displacements ``u`` (``E``: Young's
    modulus of each Gauss point and layer)."""
    ne = quads.shape[0]
    nl = sig_n.shape[2]
    hl = h0 / nl
    for e in prange(ne):
        for i in range(8):
            fe[e, i] = 0.0
            for j in range(8):
                Ke[e, i, j] = 0.0
        for g in range(4):
            F = np.eye(2)
            for an in range(4):
                node = quads[e, an]
                for i in range(2):
                    for j in range(2):
                        F[i, j] += u[2 * node + i] * dndX[e, g, an, j]
            for i in range(2):
                for j in range(2):
                    Fg[e, g, i, j] = F[i, j]
            det = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
            if det <= 0.0:
                bad[e] = True
                continue
            inv = np.empty((2, 2))
            inv[0, 0], inv[0, 1] = F[1, 1] / det, -F[0, 1] / det
            inv[1, 0], inv[1, 1] = -F[1, 0] / det, F[0, 0] / det
            dndx = np.empty((4, 2))
            for an in range(4):
                for j in range(2):
                    dndx[an, j] = dndX[e, g, an, 0] * inv[0, j] + dndX[e, g, an, 1] * inv[1, j]
            N = np.zeros(3)
            D = np.zeros((3, 3))
            so = np.empty(3)
            Cl = np.empty((3, 3))
            ml = np.empty(3)
            for layer in range(nl):
                pl, ll, ok = _point(
                    F,
                    Fn[e, g],
                    sig_n[e, g, layer],
                    p_n[e, g, layer],
                    lam_n[e, g, layer],
                    law[e, g, layer],
                    aniso[e, g, layer],
                    E[e, g, layer],
                    nu,
                    offset,
                    so,
                    Cl,
                    ml,
                )
                if not ok:
                    bad[e] = True
                p[e, g, layer] = pl
                lam[e, g, layer] = ll
                for i in range(3):
                    sig[e, g, layer, i] = so[i]
                jl = det * ll
                # the Truesdell tangent of the Kirchhoff stress: J (C + sigma m) - S(tau)
                t11, t22, t12 = jl * so[0], jl * so[1], jl * so[2]
                for i in range(3):
                    N[i] += hl * jl * so[i]
                    for j in range(3):
                        D[i, j] += hl * jl * (Cl[i, j] + so[i] * ml[j])
                D[0, 0] -= hl * 2.0 * t11
                D[1, 1] -= hl * 2.0 * t22
                D[0, 2] -= hl * t12
                D[1, 2] -= hl * t12
                D[2, 0] -= hl * t12
                D[2, 1] -= hl * t12
                D[2, 2] -= hl * 0.5 * (t11 + t22)
            wg = w0[e, g]
            B = np.zeros((3, 8))
            for an in range(4):
                B[0, 2 * an] = dndx[an, 0]
                B[1, 2 * an + 1] = dndx[an, 1]
                B[2, 2 * an] = dndx[an, 1]
                B[2, 2 * an + 1] = dndx[an, 0]
            for i in range(8):
                acc = 0.0
                for k in range(3):
                    acc += B[k, i] * N[k]
                fe[e, i] += acc * wg
            DB = np.zeros((3, 8))
            for k in range(3):
                for j in range(8):
                    acc = 0.0
                    for m in range(3):
                        acc += D[k, m] * B[m, j]
                    DB[k, j] = acc
            for i in range(8):
                for j in range(8):
                    acc = 0.0
                    for k in range(3):
                        acc += B[k, i] * DB[k, j]
                    Ke[e, i, j] += acc * wg
            # geometric stiffness of the membrane forces
            for an in range(4):
                for bn in range(4):
                    gab = dndx[an, 0] * (N[0] * dndx[bn, 0] + N[2] * dndx[bn, 1]) + dndx[an, 1] * (
                        N[2] * dndx[bn, 0] + N[1] * dndx[bn, 1]
                    )
                    Ke[e, 2 * an, 2 * bn] += gab * wg
                    Ke[e, 2 * an + 1, 2 * bn + 1] += gab * wg


@njit(parallel=True, cache=True)
def _replay(Fh, law, aniso, w0, h0, E, nu, offset, work, plastic):  # pragma: no cover
    """Replay of the deformation gradients ``Fh`` (steps, e, 4, 2, 2) of the Gauss points
    with the flow laws ``law`` (e, 4, layers, 9): the stress work of each element up to
    each step (``work``, steps x e, N mm = mJ) and the plastic strain of its points
    (``plastic``, steps x e x 4 x layers)."""
    ns, ne = Fh.shape[0], Fh.shape[1]
    nl = law.shape[2]
    hl = h0 / nl
    for e in prange(ne):
        so = np.empty(3)
        Cl = np.empty((3, 3))
        ml = np.empty(3)
        for k in range(ns):
            work[k, e] = 0.0
        for g in range(4):
            for layer in range(nl):
                sig = np.zeros(3)
                p, lam, j_old, w = 0.0, 1.0, 1.0, 0.0
                lw = law[e, g, layer]
                aw = aniso[e, g, layer]
                vol = w0[e, g] * hl
                plastic[0, e, g, layer] = 0.0
                for k in range(1, ns):
                    F = Fh[k, e, g]
                    c, s, e00, e11, e01, ok = _kinematics(F, Fh[k - 1, e, g])
                    if ok:
                        p, lam, ok = _point(
                            F,
                            Fh[k - 1, e, g],
                            sig,
                            p,
                            lam,
                            lw,
                            aw,
                            E[e, g, layer],
                            nu,
                            offset,
                            so,
                            Cl,
                            ml,
                        )
                    if ok:
                        # the new stress back in the frame of the old one, where ln dU is
                        t00 = c * c * so[0] + 2.0 * c * s * so[2] + s * s * so[1]
                        t11 = s * s * so[0] - 2.0 * c * s * so[2] + c * c * so[1]
                        t01 = -c * s * (so[0] - so[1]) + (c * c - s * s) * so[2]
                        j_new = (F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]) * lam
                        dw = (
                            (sig[0] + t00) * e00 + (sig[1] + t11) * e11 + 2.0 * (sig[2] + t01) * e01
                        )
                        w += 0.25 * dw * (j_old + j_new)
                        j_old = j_new
                        for i in range(3):
                            sig[i] = so[i]
                    work[k, e] += w * vol
                    plastic[k, e, g, layer] = p


class Fracture:
    """The fracture of the brittle points of a membrane (finite ``critical`` plastic
    strains, ``(e, 4, layers)``): ``local``, a point breaks at its critical strain; or
    nonlocal over a ``radius`` (mm), the plastic strain of the brittle points of each
    element averaged with that of its neighbours of the same critical strain (the same
    phase) within the radius (weights ``radius − d`` between centres times their brittle
    volume; Pijaudier-Cabot & Bažant 1987),
    against the critical strain of the element (the least of its points): the crack
    crosses a region of the size of a track, not a point of its edge."""

    def __init__(self, centroids, w0, critical, radius: float | None = None):
        crit = np.asarray(critical, dtype=float)
        vol = np.broadcast_to(np.asarray(w0, dtype=float)[:, :, None], crit.shape)
        self.critical = crit
        self.finite = np.isfinite(crit)
        self.any = bool(self.finite.any())
        self.vol = np.where(self.finite, vol, 0.0)
        bvol = self.vol.sum(axis=(1, 2))
        self.share = bvol / vol.sum(axis=(1, 2))
        self.element_critical = np.where(self.finite, crit, np.inf).min(axis=(1, 2))
        self.radius = radius
        self.weights = None
        if radius and self.any:
            from scipy.spatial import cKDTree

            c = np.asarray(centroids, dtype=float)
            pairs = cKDTree(c).query_pairs(radius, output_type="ndarray")
            d = np.linalg.norm(c[pairs[:, 0]] - c[pairs[:, 1]], axis=1)
            n = len(c)
            rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
            cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
            vals = np.concatenate([radius - d, radius - d, np.full(n, radius)])
            self.weights = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
            self._bvol = bvol

    def element_strain(self, plastic) -> np.ndarray:
        """The mean plastic strain of the brittle points of each element (``(..., e, 4,
        layers)`` to ``(..., e)``)."""
        tot = self.vol.sum(axis=(1, 2))
        num = (np.asarray(plastic) * self.vol).sum(axis=(-2, -1))
        return np.where(tot > 0, num / np.where(tot > 0, tot, 1.0), 0.0)

    def ratios(self, plastic, alive=None) -> np.ndarray:
        """The ratio of the (averaged) plastic strain to the critical one of each
        element, ``(..., e)`` from the plastic strains ``(..., e, 4, layers)``; only the
        ``alive`` elements (all by default) are averaged and have a ratio (the broken
        ones of a growing crack, :class:`Membrane`, none)."""
        p = np.asarray(plastic, dtype=float)
        if not self.any:
            return np.zeros(p.shape[:-2])
        live = np.ones(len(self.share), dtype=bool) if alive is None else np.asarray(alive)
        if self.weights is None:
            r = np.where(self.finite, p / np.where(self.finite, self.critical, 1.0), 0.0)
            return np.where(live, r.max(axis=(-2, -1)), 0.0)
        pe = self.element_strain(p)
        flat = pe.reshape(-1, pe.shape[-1]).T  # e x steps
        avg = np.zeros_like(flat)
        for group in self._groups():  # each phase averaged with itself
            bv = self._bvol * (group & live)
            num = self.weights @ (flat * bv[:, None])
            den = (self.weights @ bv)[:, None]
            g = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
            avg = np.where(group[:, None], g, avg)
        avg = avg.T.reshape(pe.shape)
        brittle = (self.share > 0) & live
        return np.where(brittle, avg / np.where(brittle, self.element_critical, 1.0), 0.0)

    def _groups(self) -> list:
        """The elements averaged together: those of one critical strain (a phase), or
        all the breakable ones when the critical strains vary continuously (densities)."""
        if getattr(self, "_cache", None) is None:
            crit = self.element_critical
            finite = np.isfinite(crit)
            values = np.unique(np.round(crit[finite], 12))
            if 1 < len(values) <= 8:
                self._cache = [finite & np.isclose(crit, v) for v in values]
            else:
                self._cache = [finite]
        return self._cache


# -- the problem -----------------------------------------------------------------------


@dataclass
class TensileResult:
    """A tensile test: the engineering ``strain`` (extensometer) and ``stress`` (force
    over the initial section of the parallel part), the ``force`` (N) and grip
    ``displacement`` (mm) at each converged step, their measures and the final fields."""

    strain: np.ndarray
    stress: np.ndarray
    force: np.ndarray
    displacement: np.ndarray
    yield_strength: float
    tensile_strength: float
    uniform_elongation: float
    energy: float
    necked: bool
    steps: int
    iterations: int
    time_s: float
    fields: dict = field(default_factory=dict, repr=False)
    failure_strain: float = math.nan
    history: list = field(default_factory=list, repr=False)
    width_strain: np.ndarray | None = field(default=None, repr=False)
    thickness_strain: np.ndarray | None = field(default=None, repr=False)

    def r_value(self, strain: float | None = None) -> float:
        """Lankford's coefficient of the gauge, the true width strain over the true
        thickness strain (their means over the extensometer) at an engineering strain
        (default: 10 %, or 0.8 of the uniform elongation if smaller)."""
        if self.width_strain is None or len(self.strain) < 3:
            return math.nan
        e = 0.1 if strain is None else float(strain)
        if strain is None and math.isfinite(self.uniform_elongation):
            e = min(e, 0.8 * self.uniform_elongation)
        e = min(e, float(self.strain[-1]))
        ew = float(np.interp(e, self.strain, self.width_strain))
        et = float(np.interp(e, self.strain, self.thickness_strain))
        return ew / et if et != 0 else math.nan

    @property
    def ductility(self) -> float:
        """The strain at which the specimen fails: the uniform elongation, or the strain at
        which a point reaches its critical plastic strain if earlier."""
        if math.isfinite(self.failure_strain):
            return min(self.uniform_elongation, self.failure_strain)
        return self.uniform_elongation

    def work(self, strain: float | None = None) -> float:
        """The work absorbed per unit volume of the parallel part (MJ/m³) up to an
        engineering strain (the last one reached by default; the failure strain if
        earlier): the area under the engineering curve."""
        e, s = self.strain, self.stress
        end = float(e[-1]) if strain is None else float(strain)
        if math.isfinite(self.failure_strain):
            end = min(end, self.failure_strain)
        end = min(end, float(e[-1]))
        keep = e <= end
        ee = np.append(e[keep], end)
        ss = np.append(s[keep], np.interp(end, e, s))
        return float(np.trapezoid(ss, ee))

    def summary(self) -> dict:
        """The measures as a mapping (MPa, engineering strain, MJ/m³)."""
        return {
            "yield_strength": self.yield_strength,
            "tensile_strength": self.tensile_strength,
            "uniform_elongation": self.uniform_elongation,
            "energy": self.energy,
            "strength_ductility": self.tensile_strength * self.uniform_elongation,
            "failure_strain": self.failure_strain,
            "ductility": self.ductility,
            "r_value": self.r_value(),
            "necked": self.necked,
        }


def curve_measures(strain, stress, *, young: float | None = None, offset: float = 0.002):
    """Measures of an engineering curve: the ``offset`` proof stress (the line of slope
    ``young``, default the initial slope, shifted by the offset), the maximum (parabola
    through the three points around it) and its strain, and the energy absorbed up to it
    (MJ/m³ = MPa); ``necked``: the curve fell after its maximum."""
    e = np.asarray(strain, dtype=float)
    s = np.asarray(stress, dtype=float)
    if len(e) < 3:
        return math.nan, float(s.max(initial=0.0)), math.nan, math.nan, False
    k = int(np.argmax(s))
    uts, eu = float(s[k]), float(e[k])
    if 0 < k < len(s) - 1:
        x, y = e[k - 1 : k + 2], s[k - 1 : k + 2]
        c = np.polyfit(x - x[1], y, 2)
        if c[0] < 0:
            xm = -c[1] / (2 * c[0])
            if abs(xm) <= max(abs(x[0] - x[1]), abs(x[2] - x[1])):
                eu, uts = float(x[1] + xm), float(np.polyval(c, xm))
    if young is None:
        j = max(1, min(3, len(e) - 1))
        young = float(s[j] / e[j]) if e[j] > 0 else math.nan
    line = s - young * (e - offset)
    cross = np.flatnonzero((line[:-1] > 0) & (line[1:] <= 0))
    if cross.size:
        i = cross[0]
        f = line[i] / (line[i] - line[i + 1])
        ys = float(s[i] + f * (s[i + 1] - s[i]))
    else:
        ys = math.nan
    mask = e <= eu
    ee = np.append(e[mask], eu)
    ss = np.append(s[mask], uts)
    energy = float(np.trapezoid(ss, ee))
    necked = bool(k < len(s) - 1 and s[-1] < uts * (1.0 - 1e-3))
    return ys, uts, eu, energy, necked


class Membrane:
    """The finite element problem of a specimen: its mesh, the flow laws of its Gauss
    points and layers ``(kind, σ0, a, b)``, each ``(elements, 4, layers)`` — or two laws
    and the weight ``w`` of the second, ``(kind1, σ01, a1, b1, kind2, σ02, a2, b2, w)``,
    the flow stress of a point then ``(1 − w) σ_1(p) + w σ_2(p)`` (the phases of an
    ideal design, the densities of a topology optimization) —, elasticity (MPa; Young's
    modulus one value or one per point and layer, ``(elements, 4, layers)``: phases of
    different stiffness), thickness (mm) and the critical plastic strains of the points
    (``critical``: the fracture of a brittle phase; infinite by default), local or averaged over
    ``fracture_radius`` (:class:`Fracture`); ``springs``: linear bars ``(i, j, k, e)``
    between nodes (stiffness ``k`` N/mm along the unit vector ``e``: the pins of a
    specimen, which turn freely about them); ``anisotropy``: Hill's 1948 coefficients and
    the angle of the rolling direction of the points ``(..., 5)`` (:func:`hill_from_r`;
    :data:`ISOTROPIC` by default: von Mises), or Barlat's Yld2000-2d coefficients, its
    exponent and the angle ``(..., 10)`` (:func:`yld2000_from_data`)."""

    def __init__(
        self,
        mesh: QuadMesh,
        flow,
        *,
        young=210000.0,
        poisson: float = 0.3,
        thickness: float = 1.0,
        critical=None,
        fracture_radius: float | None = None,
        springs=None,
        anisotropy=None,
    ):
        self.mesh = mesh
        self.coords = np.ascontiguousarray(mesh.nodes, dtype=float)
        self.quads = np.ascontiguousarray(mesh.quads, dtype=np.int64)
        self.points, self.dndX, self.w0 = mesh.gauss()
        parts = [np.asarray(v, dtype=float) for v in flow]
        if len(parts) == 4:
            parts = parts + parts + [np.zeros_like(parts[0])]
        if len(parts) != 9:
            raise ValueError("flow: 4 arrays (one law) or 9 (two laws and a weight)")
        ne = len(self.quads)
        shape = parts[0].shape
        if len(shape) != 3 or shape[:2] != (ne, 4):
            raise ValueError("flow: arrays of shape (elements, 4, layers)")
        self.law = np.ascontiguousarray(np.stack(np.broadcast_arrays(*parts), axis=-1))
        self.layers = shape[2]
        self.critical = (
            np.full(shape, np.inf)
            if critical is None
            else np.broadcast_to(np.asarray(critical, dtype=float), shape).copy()
        )
        #: Young's modulus of each point and layer; ``young``: their mean over the area
        #: (the modulus of the sheet along stiff and soft bands, the scale of the forces)
        self.moduli = np.ascontiguousarray(
            np.broadcast_to(np.asarray(young, dtype=float), shape), dtype=float
        )
        if not np.all(self.moduli > 0):
            raise ValueError("young: positive moduli")
        self.young = float((self.moduli.mean(axis=2) * self.w0).sum() / self.w0.sum())
        self.poisson, self.thickness = float(poisson), float(thickness)
        self.fracture_radius = fracture_radius
        self.centroids = self.coords[self.quads].mean(axis=1)
        self.fracture = Fracture(self.centroids, self.w0, self.critical, fracture_radius)
        aniso = np.asarray(ISOTROPIC if anisotropy is None else anisotropy, dtype=float)
        if aniso.shape[-1] not in (5, 10):
            raise ValueError("anisotropy: 5 values (Hill 1948) or 10 (Yld2000-2d) a point")
        self.aniso = np.ascontiguousarray(np.broadcast_to(aniso, (*shape, aniso.shape[-1])))
        self.springs = None
        if springs is not None:  # bars (i, j, stiffness, unit direction): pins, links
            i, j, k, e = (np.asarray(v) for v in springs)
            i, j = i.astype(np.int64), j.astype(np.int64)
            k = np.broadcast_to(np.asarray(k, dtype=float), i.shape).astype(float)
            e = np.asarray(e, dtype=float).reshape(-1, 2)
            blocks = k[:, None, None] * e[:, :, None] * e[:, None, :]  # n x 2 x 2
            dofs = np.stack([2 * i, 2 * i + 1, 2 * j, 2 * j + 1], axis=1)
            kb = np.zeros((len(i), 4, 4))
            kb[:, :2, :2], kb[:, 2:, 2:] = blocks, blocks
            kb[:, :2, 2:], kb[:, 2:, :2] = -blocks, -blocks
            self.springs = (dofs, kb)
        dofs = np.empty((ne, 8), dtype=np.int64)
        dofs[:, 0::2] = 2 * self.quads
        dofs[:, 1::2] = 2 * self.quads + 1
        self.dofs = dofs
        self._rows = np.repeat(dofs, 8, axis=1).ravel()
        self._cols = np.tile(dofs, (1, 8)).ravel()
        self.ndof = 2 * len(self.coords)
        self.reset()

    #: The stiffness left to a broken element, a share of its elastic one.
    residual = 1e-6

    def reset(self) -> None:
        ne, nl = len(self.quads), self.layers
        self.alive = np.ones(ne, dtype=bool)  # the elements a crack has not broken
        self.u = np.zeros(self.ndof)
        self.Fn = np.tile(np.eye(2), (ne, 4, 1, 1))
        self.sig_n = np.zeros((ne, 4, nl, 3))
        self.p_n = np.zeros((ne, 4, nl))
        self.lam_n = np.ones((ne, 4, nl))

    def evaluate(self, u):
        """Internal forces, tangent matrix and the trial states at displacements ``u``."""
        ne, nl = len(self.quads), self.layers
        Ke = np.empty((ne, 8, 8))
        fe = np.empty((ne, 8))
        sig = np.empty((ne, 4, nl, 3))
        p = np.empty((ne, 4, nl))
        lam = np.empty((ne, 4, nl))
        Fg = np.empty((ne, 4, 2, 2))
        bad = np.zeros(ne, dtype=np.bool_)
        _elements(
            self.coords,
            self.quads,
            self.dndX,
            self.w0,
            u,
            self.Fn,
            self.sig_n,
            self.p_n,
            self.lam_n,
            self.law,
            self.aniso,
            self.moduli,
            self.poisson,
            LUDWIK_OFFSET,
            self.thickness,
            Ke,
            fe,
            sig,
            p,
            lam,
            Fg,
            bad,
        )
        self.inverted = bad.copy()  # the elements this displacement turns inside out
        dead = ~self.alive
        if dead.any():  # broken elements: a residual elastic stiffness, their state frozen
            Kd = self.residual * self.elastic_stiffness()[dead]
            Ke[dead] = Kd
            fe[dead] = np.einsum("eij,ej->ei", Kd, u[self.dofs[dead]])
            sig[dead] = 0.0
            p[dead], lam[dead], Fg[dead] = self.p_n[dead], self.lam_n[dead], self.Fn[dead]
            bad[dead] = False
        f = np.bincount(self.dofs.ravel(), weights=fe.ravel(), minlength=self.ndof)
        K = sparse.csr_matrix((Ke.ravel(), (self._rows, self._cols)), shape=(self.ndof, self.ndof))
        if self.springs is not None:  # linear bars, their directions fixed
            dofs, kb = self.springs
            fs = np.einsum("nab,nb->na", kb, u[dofs])
            f = f + np.bincount(dofs.ravel(), weights=fs.ravel(), minlength=self.ndof)
            rows = np.repeat(dofs, 4, axis=1).ravel()
            cols = np.tile(dofs, (1, 4)).ravel()
            K = K + sparse.csr_matrix((kb.ravel(), (rows, cols)), shape=K.shape)
        return f, K, (sig, p, lam, Fg), bool(bad.any())

    def commit(self, u, trial) -> None:
        sig, p, lam, Fg = trial
        self.u = u.copy()
        self.sig_n, self.p_n, self.lam_n, self.Fn = sig, p, lam, Fg

    def elastic_stiffness(self) -> np.ndarray:
        """The small-strain elastic stiffness of each element in its initial shape
        (``(e, 8, 8)``, plane stress over the thickness; computed once)."""
        if getattr(self, "_kel", None) is None:
            nu = self.poisson
            D = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]) / (
                1.0 - nu * nu
            )
            ne = len(self.quads)
            B = np.zeros((ne, 4, 3, 8))
            B[:, :, 0, 0::2] = self.dndX[..., 0]
            B[:, :, 1, 1::2] = self.dndX[..., 1]
            B[:, :, 2, 0::2] = self.dndX[..., 1]
            B[:, :, 2, 1::2] = self.dndX[..., 0]
            wE = self.w0 * self.moduli.mean(axis=2)  # the modulus of each point
            self._kel = np.einsum("egai,ab,egbj,eg->eij", B, D, B, wE) * self.thickness
        return self._kel


@dataclass
class Replay:
    """The replay of the history of a tensile test with other flow laws: at each
    recorded step, the extensometer ``strain``, the grip ``displacement`` (mm), the
    stress ``work`` of each element (steps x elements, mJ), the ``plastic`` strains of
    the points (steps x elements x 4 x layers) and the ``ratio`` of each element to
    fracture (:class:`Fracture`)."""

    strain: np.ndarray
    displacement: np.ndarray
    work: np.ndarray
    plastic: np.ndarray
    ratio: np.ndarray


def replay(membrane: Membrane, history, law=None, critical=None, anisotropy=None) -> Replay:
    """Replay the deformation gradients of the Gauss points recorded by
    :func:`tensile_test` (``record_history``) with the flow laws ``law`` (the
    membrane's by default; an array ``(elements, 4, layers, 9)`` like
    ``Membrane.law``) and the ``critical`` plastic strains (the fracture of the
    membrane, its radius): the work and the fracture each element would have seen had
    it followed the same path with another material (the kinematics held, the
    first-order effect of a change of material: the sensitivities of
    :mod:`morphogen.laser.cells`)."""
    if not history:
        raise ValueError("replay: no history (tensile_test(record_history=True))")
    strain = np.array([h[0] for h in history], dtype=float)
    disp = np.array([h[1] for h in history], dtype=float)
    Fh = np.ascontiguousarray(np.stack([h[2] for h in history]))
    law = membrane.law if law is None else np.ascontiguousarray(law, dtype=float)
    if law.shape != membrane.law.shape:
        raise ValueError(f"replay: law of shape {membrane.law.shape}")
    if critical is None:
        fracture = membrane.fracture
    else:
        crit = np.broadcast_to(np.asarray(critical, dtype=float), law.shape[:3])
        fracture = Fracture(membrane.centroids, membrane.w0, crit, membrane.fracture_radius)
    ns, ne = Fh.shape[0], Fh.shape[1]
    work = np.zeros((ns, ne))
    plastic = np.zeros((ns, *law.shape[:3]))
    if anisotropy is None:
        aniso = membrane.aniso
    else:
        aniso = np.asarray(anisotropy, dtype=float)
        aniso = np.ascontiguousarray(np.broadcast_to(aniso, (*law.shape[:3], aniso.shape[-1])))
    _replay(
        Fh,
        law,
        aniso,
        membrane.w0,
        membrane.thickness,
        membrane.moduli,
        membrane.poisson,
        LUDWIK_OFFSET,
        work,
        plastic,
    )
    return Replay(strain, disp, work, plastic, fracture.ratios(plastic))


def tensile_test(
    specimen: Specimen,
    membrane: Membrane,
    *,
    max_strain: float = 0.3,
    step: float = 0.002,
    max_step: float = 0.01,
    drop: float = 0.02,
    beyond: float = 0.02,
    tol: float = 1e-5,
    max_iterations: int = 20,
    max_steps: int = 600,
    callback=None,
    keep_fields: bool = True,
    failure_margin: float | None = None,
    record_history: bool = False,
) -> TensileResult:
    """Pull the specimen (see the module): steps of grip displacement of ``step`` times
    the parallel length at first, grown up to ``max_step`` while Newton's method
    converges in few iterations (shrunk when it needs many, halved when it fails), back
    to ``step`` after the maximum force; until the extensometer reads ``max_strain``,
    or the force has fallen by ``drop`` after its maximum, or the strain exceeds that of
    the maximum by ``beyond`` (the neck has formed: the measures are taken). A point
    whose plastic strain reaches its critical one (``Membrane.critical``: a brittle
    phase) sets the ``failure_strain`` of the specimen (interpolated between the steps);
    the test stops ``failure_margin`` beyond it if given; ``record_history`` keeps the
    deformation gradients of the Gauss points at every step (the replays of
    :func:`replay`)."""
    t0 = time.perf_counter()
    mesh = membrane.mesh
    nodes = mesh.nodes
    left, right = mesh.left, mesh.right
    fixed = [2 * left, 2 * right]
    if specimen.kind == "strip" and not specimen.clamped:
        mid = left[len(left) // 2]
        fixed.append(np.array([2 * mid + 1]))
    else:
        fixed += [2 * left + 1, 2 * right + 1]
    fixed = np.unique(np.concatenate(fixed))
    free = np.setdiff1d(np.arange(membrane.ndof), fixed)
    pull = 2 * right  # the x dofs of the moving grip
    float(nodes[right, 0].mean() - nodes[left, 0].mean())
    xa, xb = specimen.extensometer_ends()
    gauge = xb - xa
    section = specimen.width * specimen.thickness

    membrane.reset()
    u = membrane.u.copy()
    disp = 0.0
    record = {"strain": [0.0], "stress": [0.0], "force": [0.0], "displacement": [0.0]}
    inc = step * specimen.length
    fmax, e_peak, iterations, steps = 0.0, 0.0, 0, 0
    min_inc = 1e-4 * step * specimen.length
    f, K, _, _ = membrane.evaluate(u)  # the tangent of the converged state
    history = [(0.0, 0.0, membrane.Fn.copy())] if record_history else []
    has_critical = membrane.fracture.any
    failure, ratio_prev = math.nan, 0.0
    xs_all = np.unique(np.round(nodes[:, 0], 9))
    sections = {}
    for xc in (xa, xb):
        j = int(np.clip(np.searchsorted(xs_all, xc) - 1, 0, len(xs_all) - 2))
        wgt = (xc - xs_all[j]) / (xs_all[j + 1] - xs_all[j])
        sections[xc] = [
            (np.flatnonzero(np.abs(nodes[:, 0] - xs_all[j]) < 1e-9), 1.0 - wgt),
            (np.flatnonzero(np.abs(nodes[:, 0] - xs_all[j + 1]) < 1e-9), wgt),
        ]

    def ext_disp(v):
        def at(xc):
            return sum(wv * v[2 * sel].mean() for sel, wv in sections[xc])

        return at(xb) - at(xa)

    # the width and the thickness of the gauge (Lankford's coefficient)
    top, bottom = mesh.top, mesh.bottom
    lateral = len(top) > 0 and len(top) == len(bottom)
    if lateral:
        inside = (nodes[top, 0] >= xa - 1e-9) & (nodes[top, 0] <= xb + 1e-9)
        top, bottom = top[inside], bottom[inside]
        lateral = len(top) > 1
    gp_x = membrane.points[..., 0]
    gw = np.where((gp_x >= xa) & (gp_x <= xb), membrane.w0, 0.0)
    record["width_strain"], record["thickness_strain"] = [0.0], [0.0]

    def lateral_strains(v):
        if not lateral:
            return math.nan, math.nan
        w = (nodes[top, 1] + v[2 * top + 1]) - (nodes[bottom, 1] + v[2 * bottom + 1])
        w0 = nodes[top, 1] - nodes[bottom, 1]
        ew = float(np.log(np.mean(w) / np.mean(w0)))
        et = float((np.log(membrane.lam_n).mean(axis=2) * gw).sum() / max(gw.sum(), 1e-300))
        return ew, et

    while True:
        target = disp + inc
        # tangent predictor: the grip moves, the free dofs follow the tangent
        trial = u.copy()
        dpull = np.zeros(membrane.ndof)
        dpull[pull] = target - disp
        rhs = -(f[free] + K[free][:, pull] @ dpull[pull])
        Kff = K[free][:, free].tocsc()
        try:
            trial[free] += splinalg.spsolve(Kff, rhs)
        except RuntimeError:
            pass
        trial[fixed] = 0.0
        trial[pull] = target
        converged = False
        state = None
        it = 0
        for it in range(max_iterations):  # noqa: B007 (it counts the iterations)
            fi, Ki, state, bad = membrane.evaluate(trial)
            iterations += 1
            if bad or not np.all(np.isfinite(fi)):
                break
            r = fi[free]
            scale = max(np.linalg.norm(fi[fixed]), 1e-9 * membrane.young * section)
            if np.linalg.norm(r) <= tol * scale:
                converged = True
                break
            try:
                du = splinalg.spsolve(Ki[free][:, free].tocsc(), -r)
            except RuntimeError:
                break
            if not np.all(np.isfinite(du)):
                break
            trial[free] += du
        if not converged:
            inc *= 0.5
            if inc < min_inc:
                break
            continue
        u, f, K = trial, fi, Ki
        membrane.commit(u, state)
        disp = target
        steps += 1
        force = float(f[pull].sum())
        strain = ext_disp(u) / gauge
        if record_history:
            history.append((strain, disp, membrane.Fn.copy()))
        ratio = float(membrane.fracture.ratios(membrane.p_n).max()) if has_critical else 0.0
        if ratio >= 1.0 and not math.isfinite(failure):
            e_prev = record["strain"][-1]
            f_ = (1.0 - ratio_prev) / max(ratio - ratio_prev, 1e-12)
            failure = e_prev + min(max(f_, 0.0), 1.0) * (strain - e_prev)
        ratio_prev = ratio
        record["strain"].append(strain)
        ew_, et_ = lateral_strains(u)
        record["width_strain"].append(ew_)
        record["thickness_strain"].append(et_)
        record["stress"].append(force / section)
        record["force"].append(force)
        record["displacement"].append(disp)
        if callback is not None:
            callback(steps, strain, force / section)
        flat = force - record["force"][-2] < 0.004 * abs(force)  # near the maximum
        if force > fmax:
            fmax, e_peak = force, strain
        past = force < fmax
        if (
            strain >= max_strain
            or (past and force < (1.0 - drop) * fmax)
            or (past and strain > e_peak + beyond)
            or steps >= max_steps
            or (failure_margin is not None and strain >= failure + failure_margin)
        ):
            break
        if it <= 4:
            inc *= 1.5
        elif it >= 8:
            inc *= 0.7
        inc = min(inc, (step if past or flat else max_step) * specimen.length)
        inc = max(inc, min_inc)
    arr = {k: np.asarray(v) for k, v in record.items()}
    ys, uts, eu, energy, necked = curve_measures(arr["strain"], arr["stress"], young=membrane.young)
    fields = {}
    if keep_fields:
        fields = {
            "mesh": (mesh.nodes, mesh.quads),
            "displacement": u.reshape(-1, 2).copy(),
            "plastic_strain": membrane.p_n.mean(axis=2),
            "stress": membrane.sig_n.mean(axis=2),
            "thickness": membrane.lam_n.mean(axis=2) * membrane.thickness,
        }
    return TensileResult(
        strain=arr["strain"],
        stress=arr["stress"],
        force=arr["force"],
        displacement=arr["displacement"],
        yield_strength=ys,
        tensile_strength=uts,
        uniform_elongation=eu,
        energy=energy,
        necked=necked,
        steps=steps,
        iterations=iterations,
        time_s=time.perf_counter() - t0,
        failure_strain=failure,
        history=history,
        fields=fields,
        width_strain=arr["width_strain"],
        thickness_strain=arr["thickness_strain"],
    )
