"""Micromorphic ductile damage: von Mises elasto-(visco)plasticity whose damage is driven by
a micromorphic counterpart of the cumulated plastic strain, which regularizes the
localization of ductile fracture, in small and finite strains (Forest 2009, the
micromorphic approach; the implicit gradient-enhanced ductile damage of Engelen, Geers &
Baaijens 2003 and Geers, Peerlings, Brekelmans & de Borst).

*Plasticity of the effective stress.* ``σ̃ = C : (ε - ε^p)`` (isotropic elasticity), the
yield function ``f = σ̃_eq - R(p)`` with ``R = σ0 + H p + Σ Q_i (1 - exp(-b_i p))``, the
normality ``ε̇^p = ṗ 3/2 s̃ / σ̃_eq``, rate-independent or with Norton's overstress ``σ̃_eq - R
= K ṗ^(1/N)``: the radial return, its consistent tangent ``C - (2μ)² θ n ⊗ n - (2μ)² Δp /
σ̃_eq^tr (3/2 I_dev - n ⊗ n)`` (``n = 3/2 s̃ / σ̃_eq``, ``θ = 1 / (3μ + R' + ∂V/∂Δp)``) and the
derivative of the cumulated plastic strain by the strain, ``∂p/∂ε = 2μ θ n``.

*Triaxiality.* The damage is driven by the equivalent plastic strain ``e``, ``ė = w(η)
ṗ``, weighted by a function of the triaxiality ``η = σ̃_m / σ̃_eq`` of the effective stress
(of the stress conjugate to the logarithmic strain in finite strains, nearly Kirchhoff's):
the ratio ``ε_f(1/3) / ε_f(η)`` of the strains to failure of uniaxial tension and of the
current state, so that ``e`` is the plastic strain of uniaxial tension that would damage
as much (``w(1/3) = 1``: the thresholds of the damage keep their meaning in tension).
``"none"``: ``w = 1``, ``e = p``; ``"rice_tracey"``: ``w = exp(β (η - 1/3))``, the growth
rate of a void of Rice & Tracey (1969, ``β = 3/2``); ``"johnson_cook"``: ``ε_f = D1 + D2
exp(D3 η)`` (Johnson & Cook 1985); ``"lemaitre"``: ``w = R_ν = 2/3 (1 + ν) + 3 (1 - 2ν) η²``,
the triaxiality function of the elastic energy (Lemaitre 1985). Pure shear damages less
than tension, the triaxial stress ahead of a notch in plane strain more: the damage
concentrates there and the crack runs from the notch across the ligament (mode I). Its
increment ``Δe = w(η) Δp`` takes the triaxiality at the end of the step (backward Euler),
and ``∂e/∂ε = w ∂p/∂ε + Δp w'(η) ∂η/∂ε`` that of the consistent tangent, ``∂η/∂ε =
(∂σ̃_m/∂ε - η ∂σ̃_eq/∂ε) / σ̃_eq``.

*Micromorphic damage.* A scalar micromorphic variable ``χ`` (a degree of freedom), the
counterpart of ``e``, with the generalized balance ``H (χ - e) - div(H ℓ² ∇χ) = 0``: the
Helmholtz equation ``χ - ℓ² Δχ = e`` of the implicit gradient models, ``∇χ · n = 0`` on
the boundary (``H`` scales the equation only). The damage follows its largest value ``κ =
max χ``, ``d = d_max (1 - exp(-(κ - κ0) / κ_c))`` beyond the onset ``κ0``, and degrades the
stress, ``σ = (1 - d) σ̃``. The plasticity, that of the effective stress, hardens and stays
regular; the softening comes from the damage, nonlocal through ``χ``, so that the band of
damage has the width set by ``ℓ``, not by the elements, and a crack is the band where ``d``
reaches ``d_max``: a ligament fully damaged carries no load.

*Finite strains* (total Lagrangian; :mod:`morphogen.micromorphic.finite`): the plasticity
acts on the logarithmic strain and its conjugate stress, ``P = (1 - d) P̃`` with ``P̃`` the
first Piola-Kirchhoff stress pulled back from it, and ``∂p/∂F`` likewise.

*Elements* (:mod:`morphogen.micromorphic.space`): quadratic displacements and a linear
``χ``; the tangent, with the derivatives of the damage by ``χ`` and of ``p`` by the strain,
is not symmetric. The time of a viscous step is its fraction of the ``duration`` of
:meth:`~morphogen.micromorphic.space.MixedProblem.run`, whose iterations start here from
the increment that the last tangent predicts (its ``predictor``): from the converged
values with only the prescribed ones moved, the points along a pulled boundary yield and
the iterations stall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from morphogen.homogenization.fe import _corner_gradients, _shape_gradients
from morphogen.mesh.tetra import TetMesh
from morphogen.micromorphic.finite import MAP, W, hencky, pull_back
from morphogen.micromorphic.space import (
    CHUNK,
    POINTS,
    WEIGHTS,
    MixedProblem,
    MixedSpace,
    PathResult,
)

#: The internal variables of a point: the plastic strain (Mandel), p, κ, d, the equivalent
#: plastic strain of the damage e and the triaxiality of the last step
N_STATE = 11
#: The weights of the triaxiality
TRIAXIALITY = {"none": 0, "rice_tracey": 1, "johnson_cook": 2, "lemaitre": 3}
#: The generalized degree of freedom: the micromorphic cumulated plastic strain
CHI = 3


@dataclass(frozen=True)
class MicromorphicDuctile:
    """A material of the micromorphic ductile damage model (see the module): elasticity
    ``young`` and ``poisson``; plasticity ``yield_stress`` σ0, the linear ``hardening`` H,
    ``voce`` ((Q1, b1), (Q2, b2)), Norton's ``viscosity`` K and ``exponent`` N (None:
    rate-independent); the damage ``onset`` κ0, its scale ``critical`` κ_c and
    ``max_damage``; the ``length`` ℓ and the ``penalty`` H of the equation of ``χ`` (by
    default ``yield_stress``); the weight of the ``triaxiality`` (``"none"``,
    ``"rice_tracey"`` with ``triaxiality_parameters`` (β,), default 1.5, ``"johnson_cook"``
    with (D1, D2, D3), ``"lemaitre"``). The hardening must keep the plasticity of the
    effective stress stable (``H + Σ Q_i b_i > 0``)."""

    young: float
    poisson: float
    yield_stress: float
    length: float
    hardening: float = 0.0
    voce: tuple = ()
    viscosity: float | None = None
    exponent: float | None = None
    onset: float = 0.0
    critical: float = 0.1
    max_damage: float = 0.99
    penalty: float | None = None
    triaxiality: str = "none"
    triaxiality_parameters: tuple = ()

    def __post_init__(self) -> None:
        name = "MicromorphicDuctile"
        if not (self.young > 0 and -1 < self.poisson < 0.5 and self.yield_stress > 0):
            raise ValueError(f"{name}: positive young and yield_stress, -1 < poisson < 0.5")
        if not (self.length > 0 and self.critical > 0 and 0 < self.max_damage < 1):
            raise ValueError(f"{name}: positive length and critical, 0 < max_damage < 1")
        if len(self.voce) > 2 or self.hardening < 0 or self.onset < 0:
            raise ValueError(f"{name}: at most two Voce terms, hardening and onset >= 0")
        if (self.viscosity is None) != (self.exponent is None):
            raise ValueError(f"{name}: Norton's viscosity and exponent together")
        if self.triaxiality not in TRIAXIALITY:
            raise ValueError(f"{name}: triaxiality among {sorted(TRIAXIALITY)}")
        needed = {"none": 0, "rice_tracey": 1, "johnson_cook": 3, "lemaitre": 0}
        given = len(self.triaxiality_parameters)
        if given not in (
            (0, 1) if self.triaxiality == "rice_tracey" else (needed[self.triaxiality],)
        ):
            raise ValueError(f"{name}: {needed[self.triaxiality]} triaxiality parameters")
        if (
            self.triaxiality == "johnson_cook"
            and not min(self._failure(np.array([-1.0 / 3.0, 3.0]))) > 0
        ):
            raise ValueError(f"{name}: Johnson-Cook strains to failure must stay positive")

    def _failure(self, eta) -> np.ndarray:
        d1, d2, d3 = self.triaxiality_parameters
        return d1 + d2 * np.exp(d3 * np.asarray(eta, dtype=float))

    def weight(self, eta) -> np.ndarray:
        """The weight ``w(η)`` of the triaxiality (see the module; 1 in uniaxial tension)."""
        eta = np.asarray(eta, dtype=float)
        kind = self.triaxiality
        if kind == "rice_tracey":
            beta = self.triaxiality_parameters[0] if self.triaxiality_parameters else 1.5
            return np.exp(beta * (eta - 1.0 / 3.0))
        if kind == "johnson_cook":
            return self._failure(1.0 / 3.0) / self._failure(eta)
        if kind == "lemaitre":
            nu = self.poisson
            return 2.0 / 3.0 * (1.0 + nu) + 3.0 * (1.0 - 2.0 * nu) * eta**2
        return np.ones_like(eta)

    def parameters(self) -> np.ndarray:
        lam = self.young * self.poisson / ((1 + self.poisson) * (1 - 2 * self.poisson))
        mu = self.young / (2 * (1 + self.poisson))
        voce = list(self.voce) + [(0.0, 0.0)] * (2 - len(self.voce))
        k, n = (0.0, 1.0)
        if self.viscosity is not None:
            k, n = float(self.viscosity), float(self.exponent)
        h = self.yield_stress if self.penalty is None else float(self.penalty)
        law = [0.0, 0.0, 0.0]
        if self.triaxiality == "rice_tracey":
            law[0] = self.triaxiality_parameters[0] if self.triaxiality_parameters else 1.5
        elif self.triaxiality == "johnson_cook":
            law = [float(v) for v in self.triaxiality_parameters]
        elif self.triaxiality == "lemaitre":
            law[0] = self.poisson
        return np.array([lam, mu, self.yield_stress, self.hardening, voce[0][0], voce[0][1],
                         voce[1][0], voce[1][1], k, n, self.onset, self.critical,
                         self.max_damage, self.length**2, h, TRIAXIALITY[self.triaxiality],
                         *law])  # fmt: skip

    def damage(self, kappa) -> np.ndarray:
        """``d(κ)`` of the damage law."""
        k = np.asarray(kappa, dtype=float)
        d = self.max_damage * (1.0 - np.exp(-(k - self.onset) / self.critical))
        return np.where(k > self.onset, d, 0.0)


@njit(cache=True)
def _hardening(par, p):
    """``R(p)`` and ``R'(p)``."""
    e1 = math.exp(-par[5] * p)
    e2 = math.exp(-par[7] * p)
    r = par[2] + par[3] * p + par[4] * (1.0 - e1) + par[6] * (1.0 - e2)
    return r, par[3] + par[4] * par[5] * e1 + par[6] * par[7] * e2


@njit(cache=True)
def _damage_law(par, kappa):
    """``d(κ)`` and ``d'(κ)``."""
    if kappa <= par[10]:
        return 0.0, 0.0
    ex = math.exp(-(kappa - par[10]) / par[11])
    return par[12] * (1.0 - ex), par[12] * ex / par[11]


@njit(cache=True)
def _weight(par, eta):
    """``w(η)`` and ``w'(η)`` of the triaxiality (see the module)."""
    kind = int(par[15])
    if kind == 1:  # Rice & Tracey
        w = math.exp(par[16] * (eta - 1.0 / 3.0))
        return w, par[16] * w
    if kind == 2:  # Johnson & Cook
        d1, d2, d3 = par[16], par[17], par[18]
        tension = d1 + d2 * math.exp(d3 / 3.0)
        failure = max(d1 + d2 * math.exp(d3 * eta), 1e-12 * tension)
        slope = d2 * d3 * math.exp(d3 * eta)
        return tension / failure, -tension * slope / failure**2
    if kind == 3:  # Lemaitre
        nu = par[16]
        return 2.0 / 3.0 * (1.0 + nu) + 3.0 * (1.0 - 2.0 * nu) * eta * eta, 6.0 * (
            1.0 - 2.0 * nu
        ) * eta
    return 1.0, 0.0


@njit(cache=True)
def _driving(par, sig, tan, dp, dp_increment, de):
    """The increment of the equivalent plastic strain of the damage, ``w(η) Δp``, and its
    derivative by the strain into ``de`` (Mandel; see the module), from the effective stress
    ``sig``, its tangent ``tan``, ``∂p/∂ε`` ``dp`` and ``Δp``; returns ``(Δe, η)``."""
    mean = (sig[0] + sig[1] + sig[2]) / 3.0
    s = sig.copy()
    for r in range(3):
        s[r] -= mean
    q = math.sqrt(1.5 * (s @ s))
    if int(par[15]) == 0 or not dp_increment > 0.0 or not q > 0.0:
        de[:] = dp
        return dp_increment, (mean / q if q > 0.0 else 0.0)
    eta = mean / q
    w, dw = _weight(par, eta)
    for c in range(6):
        dmean = (tan[0, c] + tan[1, c] + tan[2, c]) / 3.0
        dq = 0.0
        for r in range(6):
            dq += 1.5 * s[r] / q * tan[r, c]
        de[c] = w * dp[c] + dp_increment * dw * (dmean - eta * dq) / q
    return w * dp_increment, eta


@njit(cache=True)
def _j2_point(par, eps, epsp_old, p_old, dt, sig, epsp, tan, dp):
    """The radial return of a point (Mandel vectors; see the module): the effective
    stress, the plastic strain, the consistent tangent and ``∂p/∂ε`` into ``sig``,
    ``epsp``, ``tan``, ``dp``; returns ``p``."""
    lam, mu = par[0], par[1]
    ee = eps - epsp_old
    tr = ee[0] + ee[1] + ee[2]
    for r in range(6):
        sig[r] = 2.0 * mu * ee[r] + (lam * tr if r < 3 else 0.0)
        for c in range(6):
            tan[r, c] = (2.0 * mu if r == c else 0.0) + (lam if r < 3 and c < 3 else 0.0)
    mean = (sig[0] + sig[1] + sig[2]) / 3.0
    s = sig.copy()
    for r in range(3):
        s[r] -= mean
    q_tr = math.sqrt(1.5 * (s @ s))
    r0, dr0 = _hardening(par, p_old)
    epsp[:] = epsp_old
    dp[:] = 0.0
    if not math.isfinite(q_tr):
        sig[:] = np.nan
        return p_old
    if not q_tr > r0 * (1.0 + 1e-12):
        return p_old
    k, n_exp = par[8], par[9]
    lo, hi = 0.0, q_tr / (3.0 * mu)
    x = min(max((q_tr - r0) / (3.0 * mu + max(dr0, 0.0)), 1e-12 * hi), 0.999 * hi)
    slope = 3.0 * mu
    for _ in range(100):
        rr, dr = _hardening(par, p_old + x)
        g = q_tr - 3.0 * mu * x - rr
        slope = 3.0 * mu + dr
        if k > 0.0:
            rate = x / dt
            g -= k * rate ** (1.0 / n_exp)
            slope += k / (n_exp * dt) * rate ** (1.0 / n_exp - 1.0)
        if abs(g) <= 1e-13 * q_tr:
            break
        if g > 0.0:
            lo = x
        else:
            hi = x
        new = x + g / slope
        if not lo < new < hi:
            new = 0.5 * (lo + hi)
        if abs(new - x) <= 1e-15 * hi:
            x = new
            break
        x = new
    theta = 1.0 / slope
    for r in range(6):
        nr = 1.5 * s[r] / q_tr
        sig[r] -= 2.0 * mu * x * nr
        epsp[r] = epsp_old[r] + x * nr
        dp[r] = 2.0 * mu * theta * nr
    a = (2.0 * mu) ** 2 * theta
    b = (2.0 * mu) ** 2 * x / q_tr
    for r in range(6):
        nr = 1.5 * s[r] / q_tr
        for c in range(6):
            nc = 1.5 * s[c] / q_tr
            dev = (1.0 if r == c else 0.0) - (1.0 / 3.0 if r < 3 and c < 3 else 0.0)
            tan[r, c] -= a * nr * nc + b * (1.5 * dev - nr * nc)
    return p_old + x


@njit(cache=True)
def point_update(par, eps, epsp_old, p_old, dt):
    """A point (Mandel vectors): returns the effective stress, the plastic strain, ``p``,
    the consistent tangent and ``∂p/∂ε`` (for checks)."""
    sig = np.empty(6)
    epsp = np.empty(6)
    tan = np.empty((6, 6))
    dp = np.empty(6)
    p = _j2_point(par, eps, epsp_old, p_old, dt, sig, epsp, tan, dp)
    return sig, epsp, p, tan, dp


@njit(cache=True)
def driving_update(par, eps, epsp_old, p_old, e_old, dt):
    """A point with the equivalent plastic strain of the damage (Mandel vectors): returns
    ``e``, the triaxiality ``η`` and ``∂e/∂ε`` (for checks)."""
    sig = np.empty(6)
    epsp = np.empty(6)
    tan = np.empty((6, 6))
    dp = np.empty(6)
    de = np.empty(6)
    p = _j2_point(par, eps, epsp_old, p_old, dt, sig, epsp, tan, dp)
    increment, eta = _driving(par, sig, tan, dp, p - p_old, de)
    return e_old + increment, eta, de


@njit(parallel=True, cache=True)
def _ductile_chunk(nodes, elements, first, mat, par, finite, values, offset, gp, gw, dt,
                   state_old, state_new, fe, ke, tangent):  # fmt: skip
    """Element vectors and matrices of the elements ``first...`` (see the module)."""
    r2 = math.sqrt(2.0)
    for ee in prange(fe.shape[0]):
        e = first + ee
        el = elements[e]
        pm = par[mat[e]]
        l2, hs = pm[13], pm[14]
        x = np.empty((4, 3))
        for a in range(4):
            x[a] = nodes[el[a]]
        gl = np.empty((4, 3))
        vol = abs(_corner_gradients(x, gl))
        ul = np.empty((10, 3))
        for a in range(10):
            for i in range(3):
                ul[a, i] = values[offset[el[a]] + i]
        gv = np.empty(4)
        for b in range(4):
            gv[b] = values[offset[el[b]] + 3]
        f = fe[ee]
        f[:] = 0.0
        if tangent:
            ke[ee, :, :] = 0.0
        grad = np.empty((10, 3))
        hu = np.empty((3, 3))
        mu3 = np.empty(3)
        q = np.empty((3, 3))
        eps = np.empty(6)
        sig = np.empty(6)
        epsp = np.empty(6)
        tan = np.empty((6, 6))
        dp6 = np.empty(6)
        de6 = np.empty(6)
        ptil = np.empty((3, 3))
        dpdf = np.empty((9, 9))
        dp9 = np.empty(9)
        for g in range(gp.shape[0]):
            bary = gp[g]
            _shape_gradients(2, gl, bary, grad)
            w = gw[g] * vol
            for i in range(3):
                for j in range(3):
                    v = 0.0
                    for a in range(10):
                        v += ul[a, i] * grad[a, j]
                    hu[i, j] = v
            chi = 0.0
            gchi = np.zeros(3)
            for b in range(4):
                chi += bary[b] * gv[b]
                for k in range(3):
                    gchi[k] += gv[b] * gl[b, k]
            so = state_old[e, g]
            sn = state_new[e, g]
            if finite:
                if hencky(hu, mu3, q, eps) != 0:  # an inverted point: the step is divided
                    f[:] = np.nan
                    sn[:] = so
                    continue
            else:
                eps[0], eps[1], eps[2] = hu[0, 0], hu[1, 1], hu[2, 2]
                eps[3] = (hu[1, 2] + hu[2, 1]) / r2
                eps[4] = (hu[2, 0] + hu[0, 2]) / r2
                eps[5] = (hu[0, 1] + hu[1, 0]) / r2
            p = _j2_point(pm, eps, so[:6], so[6], dt, sig, epsp, tan, dp6)
            increment, eta = _driving(pm, sig, tan, dp6, p - so[6], de6)
            e_drive = so[9] + increment
            if finite:
                pull_back(hu, mu3, q, sig, tan, de6, ptil, dpdf, dp9)
            else:
                for i in range(3):
                    for j in range(3):
                        a6 = MAP[i, j]
                        ptil[i, j] = sig[a6] / W[a6]
                        dp9[3 * i + j] = de6[a6] / W[a6]
                        for qq in range(3):
                            for rr in range(3):
                                b6 = MAP[qq, rr]
                                dpdf[3 * i + j, 3 * qq + rr] = tan[a6, b6] / (W[a6] * W[b6])
            kappa_old = so[7]
            loading = chi > kappa_old
            kappa = chi if loading else kappa_old
            d, dd = _damage_law(pm, kappa)
            if not loading:
                dd = 0.0
            sn[:6] = epsp
            sn[6] = p
            sn[7] = kappa
            sn[8] = d
            sn[9] = e_drive
            sn[10] = eta
            for a in range(10):
                for i in range(3):
                    v = 0.0
                    for j in range(3):
                        v += ptil[i, j] * grad[a, j]
                    f[3 * a + i] += w * (1.0 - d) * v
            for b in range(4):
                dot = gchi[0] * gl[b, 0] + gchi[1] * gl[b, 1] + gchi[2] * gl[b, 2]
                f[30 + b] += w * hs * ((chi - e_drive) * bary[b] + l2 * dot)
            if not tangent:
                continue
            kk = ke[ee]
            for a in range(10):
                for i in range(3):
                    for c in range(10):
                        for qq in range(3):
                            v = 0.0
                            for j in range(3):
                                for rr in range(3):
                                    v += grad[a, j] * dpdf[3 * i + j, 3 * qq + rr] * grad[c, rr]
                            kk[3 * a + i, 3 * c + qq] += w * (1.0 - d) * v
                    v = 0.0
                    for j in range(3):
                        v += grad[a, j] * ptil[i, j]
                    for b in range(4):
                        kk[3 * a + i, 30 + b] -= w * dd * v * bary[b]
                for qq in range(3):
                    v = 0.0
                    for rr in range(3):
                        v += dp9[3 * qq + rr] * grad[a, rr]
                    for b in range(4):
                        kk[30 + b, 3 * a + qq] -= w * hs * bary[b] * v
            for b in range(4):
                for c in range(4):
                    dot = gl[b, 0] * gl[c, 0] + gl[b, 1] * gl[c, 1] + gl[b, 2] * gl[c, 2]
                    kk[30 + b, 30 + c] += w * hs * (bary[b] * bary[c] + l2 * dot)


class DuctileProblem(MixedProblem):
    """A body of :class:`MicromorphicDuctile` materials (one for every phase, or a dict of
    the phases), in ``"small"`` or ``"finite"`` strains: its generalized degree of freedom
    (component :data:`CHI` = 3 of the corner nodes) is ``χ``, the micromorphic cumulated
    plastic strain of the damage (the counterpart of ``e``, the plastic strain weighted by
    the triaxiality)."""

    def __init__(self, mesh: TetMesh, materials, *, kinematics: str = "small"):
        if kinematics not in ("small", "finite"):
            raise ValueError("kinematics: 'small' or 'finite'")
        if isinstance(materials, MicromorphicDuctile):
            materials = {int(p): materials for p in np.unique(mesh.phase)}
        self.space = MixedSpace(mesh, 1)
        self.mesh = self.space.mesh
        self.finite = kinematics == "finite"
        phases = sorted(materials)
        if not np.isin(self.mesh.phase, phases).all():
            raise ValueError("DuctileProblem: a material for every phase of the mesh")
        index = {p: k for k, p in enumerate(phases)}
        self.materials = materials
        self._mat = np.array([index[int(p)] for p in self.mesh.phase], dtype=np.int64)
        self._par = np.array([materials[p].parameters() for p in phases])

    def run(
        self, prescribed, load: float = 1.0, *, predictor: bool = True, **options
    ) -> PathResult:
        """:meth:`~morphogen.micromorphic.space.MixedProblem.run`, by default with its
        ``predictor`` (the iterations of a step from the increment that the last tangent
        predicts: points yielding along the loaded boundary would stall them)."""
        return super().run(prescribed, load, predictor=predictor, **options)

    def _initial_state(self):
        return np.zeros((self.mesh.n_elements, len(WEIGHTS), N_STATE))

    def _assemble(self, values, state, tangent):
        space = self.space
        n_el = self.mesh.n_elements
        rhs = np.zeros(space.n_dofs)
        data = space.new_matrix_data() if tangent else np.zeros(0)
        new_state = np.empty_like(state)
        n_loc = space.n_local
        dt = float(getattr(self, "_dt", 1.0))
        for first in range(0, n_el, CHUNK):
            count = min(CHUNK, n_el - first)
            fe = np.empty((count, n_loc))
            ke = np.empty((count, n_loc, n_loc)) if tangent else np.empty((count, 1, 1))
            _ductile_chunk(self.mesh.nodes, space.elements, first, self._mat, self._par,
                           self.finite, values, space.offset, POINTS, WEIGHTS, dt, state,
                           new_state, fe, ke, tangent)  # fmt: skip
            space.add(first, ke, fe, data, rhs)
        return rhs, (data if tangent else None), new_state

    def damage(self, state) -> np.ndarray:
        """The damage of the elements (the mean of their points)."""
        return state[:, :, 8].mean(axis=1)

    def plastic_strain(self, state) -> np.ndarray:
        """The cumulated plastic strain of the elements (the mean of their points)."""
        return state[:, :, 6].mean(axis=1)

    def driving(self, state) -> np.ndarray:
        """The equivalent plastic strain of the damage ``e`` of the elements (the mean of
        their points)."""
        return state[:, :, 9].mean(axis=1)

    def triaxiality(self, state) -> np.ndarray:
        """The triaxiality of the effective stress of the elements at their last plastic
        step (the mean of their points)."""
        return state[:, :, 10].mean(axis=1)
