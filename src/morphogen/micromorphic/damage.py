"""A micromorphic damage model to crack with: cracks as localized bands of damage, whose
width a length sets independently of the elements, in small and finite strains (Forest
2009, the micromorphic approach; the implicit gradient model of Peerlings, de Borst,
Brekelmans & de Vree 1996).

*Model.* A scalar micromorphic variable ``χ`` (a degree of freedom), the micromorphic
counterpart of a local equivalent strain ``ε_eq``, with the free energy ``ψ = (1 - d(κ))
w(ε) + H/2 (χ - ε_eq(ε))² + H ℓ²/2 |∇χ|²`` (``w`` the elastic energy) and the
generalized balance ``div b - a = 0`` of its stresses ``a = ∂ψ/∂χ = H (χ - ε_eq)`` and ``b =
∂ψ/∂∇χ = H ℓ² ∇χ``: the Helmholtz equation ``χ - ℓ² Δχ = ε_eq`` of the implicit gradient
model (for any penalty ``H``), with ``∇χ · n = 0`` on the boundary. The damage follows
the largest value of ``χ``, ``κ = max χ``: ``d = 1 - (κ0/κ) (1 - α + α exp(-β (κ - κ0)))``
beyond the threshold ``κ0`` (Peerlings et al. 1998), at most ``d_max``. The stress is that
of Peerlings et al., ``σ = (1 - d) C : ε``; the derivative of the free energy, ``(1 - d) C
: ε - H (χ - ε_eq) ∂ε_eq/∂ε`` (``variational``), keeps in the band the stiffness of the
penalty, which does not degrade: a bar then damages throughout instead of localizing
(stress locking), unless ``H`` is small. The
equivalent strain is the modified von Mises one (de Vree et al. 1995), ``ε_eq = (k - 1)
I1 / (2k (1 - 2ν)) + sqrt(((k - 1) I1 / (1 - 2ν))² + 12 k J2 / (1 + ν)²) / (2k)``, the
uniaxial strain in tension and ``1/k`` of it in compression.

*Finite strains* (total Lagrangian): the compressible neo-Hookean energy ``w = μ/2 (I1 -
3) - μ ln J + λ/2 (ln J)²`` and the equivalent strain of the Green-Lagrange strain ``E``,
so that ``P = (1 - d) P0(F) - H (χ - ε_eq) F ∂ε_eq/∂E``.

*Elements* (:mod:`morphogen.micromorphic.space`): quadratic displacements and a linear
``χ``, one order lower, as Peerlings et al. recommend (equal orders make the stresses
oscillate); the tangent, with the derivative of the damage, is not symmetric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from morphogen.homogenization.fe import _corner_gradients, _shape_gradients
from morphogen.mesh.tetra import TetMesh
from morphogen.micromorphic.microfoam import _MAP, _W
from morphogen.micromorphic.space import CHUNK, POINTS, WEIGHTS, MixedProblem, MixedSpace

#: The internal variables of a point: κ and d
N_STATE = 2
#: The generalized degree of freedom: the micromorphic equivalent strain
CHI = 3


@dataclass(frozen=True)
class MicromorphicDamage:
    """A material of the micromorphic damage model (see the module): elasticity
    ``young`` and ``poisson``; the damage ``threshold`` κ0, ``alpha`` and ``beta``, the
    strength ratio ``ratio`` k of the equivalent strain and the largest damage
    ``max_damage``; the ``length`` ℓ, the ``penalty`` H (by default ``young``, which scales
    the equation of ``χ`` only) and ``variational`` (the coupling term of the stress: see
    the module)."""

    young: float
    poisson: float
    threshold: float
    length: float
    alpha: float = 0.99
    beta: float = 300.0
    ratio: float = 10.0
    max_damage: float = 0.9999
    penalty: float | None = None
    variational: bool = False

    def __post_init__(self) -> None:
        if not (self.young > 0 and -1 < self.poisson < 0.5 and self.threshold > 0):
            raise ValueError("MicromorphicDamage: positive young and threshold, -1 < poisson < 0.5")
        if not (self.length > 0 and self.ratio >= 1 and 0 < self.max_damage < 1):
            raise ValueError("MicromorphicDamage: positive length, ratio >= 1, 0 < max_damage < 1")

    def parameters(self) -> np.ndarray:
        h = self.young if self.penalty is None else float(self.penalty)
        lam = self.young * self.poisson / ((1 + self.poisson) * (1 - 2 * self.poisson))
        mu = self.young / (2 * (1 + self.poisson))
        coupling = h if self.variational else 0.0
        return np.array([lam, mu, self.poisson, self.threshold, self.alpha, self.beta, self.ratio,
                         self.max_damage, self.length**2, h, coupling])  # fmt: skip

    def damage(self, kappa) -> np.ndarray:
        """``d(κ)`` of the damage law."""
        k = np.maximum(np.asarray(kappa, dtype=float), self.threshold)
        d = 1 - self.threshold / k * (
            1 - self.alpha + self.alpha * np.exp(-self.beta * (k - self.threshold))
        )
        return np.minimum(d, self.max_damage)


@njit(cache=True)
def _damage_law(par, kappa):
    """``d(κ)`` and ``d'(κ)``."""
    k0, alpha, beta, dmax = par[3], par[4], par[5], par[7]
    if kappa <= k0:
        return 0.0, 0.0
    ex = math.exp(-beta * (kappa - k0))
    d = 1.0 - k0 / kappa * (1.0 - alpha + alpha * ex)
    if d >= dmax:
        return dmax, 0.0
    dd = k0 / (kappa * kappa) * (1.0 - alpha + alpha * ex) + k0 / kappa * alpha * beta * ex
    return d, dd


@njit(cache=True)
def _equivalent(par, eps, grad, hess):
    """The modified von Mises equivalent strain of a strain (Mandel), its gradient and
    its Hessian (Mandel)."""
    nu, k = par[2], par[6]
    a = (k - 1.0) / (2.0 * k * (1.0 - 2.0 * nu))
    c = (k - 1.0) / (1.0 - 2.0 * nu)
    g = 12.0 * k / (1.0 + nu) ** 2
    i1 = eps[0] + eps[1] + eps[2]
    dev = eps.copy()
    for i in range(3):
        dev[i] -= i1 / 3.0
    j2 = 0.5 * (dev @ dev)
    small = (1e-9 * par[3]) ** 2
    r = math.sqrt(c * c * i1 * i1 + g * j2 + small)
    q = np.empty(6)
    for i in range(6):
        q[i] = 0.5 * g * dev[i] + (c * c * i1 if i < 3 else 0.0)
    for i in range(6):
        grad[i] = (a if i < 3 else 0.0) + q[i] / (2.0 * k * r)
    for i in range(6):
        for j in range(6):
            v = 0.0
            if i < 3 and j < 3:
                v += c * c - g / 6.0
            if i == j:
                v += 0.5 * g
            hess[i, j] = v / (2.0 * k * r) - q[i] * q[j] / (2.0 * k * r**3)
    return a * i1 + r / (2.0 * k)


@njit(parallel=True, cache=True)
def _damage_chunk(nodes, elements, first, mat, par, finite, values, offset, gp, gw, state_old,
                  state_new, fe, ke, tangent):  # fmt: skip
    """Element vectors and matrices of the elements ``first...`` (see the module)."""
    for ee in prange(fe.shape[0]):
        e = first + ee
        el = elements[e]
        pm = par[mat[e]]
        lam, mu, l2, hs, hc = pm[0], pm[1], pm[8], pm[9], pm[10]
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
        strain = np.empty(6)
        deq = np.empty(6)
        heq = np.empty((6, 6))
        p = np.empty((3, 3))
        duu = np.zeros((3, 3, 3, 3))
        duc = np.empty((3, 3))
        dcu = np.empty((3, 3))
        r2 = math.sqrt(2.0)
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
            if finite:  # the Green-Lagrange strain
                fm = np.eye(3) + hu
                cg = fm.T @ fm
                strain[0], strain[1], strain[2] = (
                    0.5 * (cg[0, 0] - 1),
                    0.5 * (cg[1, 1] - 1),
                    0.5 * (cg[2, 2] - 1),
                )
                strain[3], strain[4], strain[5] = cg[1, 2] / r2, cg[2, 0] / r2, cg[0, 1] / r2
            else:
                strain[0], strain[1], strain[2] = hu[0, 0], hu[1, 1], hu[2, 2]
                strain[3] = (hu[1, 2] + hu[2, 1]) / r2
                strain[4] = (hu[2, 0] + hu[0, 2]) / r2
                strain[5] = (hu[0, 1] + hu[1, 0]) / r2
            eq = _equivalent(pm, strain, deq, heq)
            kappa_old = state_old[e, g, 0]
            loading = chi > kappa_old
            kappa = chi if loading else kappa_old
            d, dd = _damage_law(pm, kappa)
            if not loading:
                dd = 0.0
            state_new[e, g, 0] = kappa
            state_new[e, g, 1] = d
            rel = chi - eq
            # the derivative of eq (3 x 3 tensor) and its push forward in finite strains
            gt = np.empty((3, 3))
            for i in range(3):
                for j in range(3):
                    k6 = _MAP[i, j]
                    gt[i, j] = deq[k6] / _W[k6]
            if finite:
                fm = np.eye(3) + hu
                finv = np.linalg.inv(fm)
                jac = np.linalg.det(fm)
                lnj = math.log(jac) if jac > 0.0 else -30.0
                p0 = mu * (fm - finv.T) + lam * lnj * finv.T
                fg = fm @ gt
                for i in range(3):
                    for j in range(3):
                        p[i, j] = (1.0 - d) * p0[i, j] - hc * rel * fg[i, j]
            else:
                tr = hu[0, 0] + hu[1, 1] + hu[2, 2]
                p0 = np.empty((3, 3))
                for i in range(3):
                    for j in range(3):
                        p0[i, j] = mu * (hu[i, j] + hu[j, i]) + (lam * tr if i == j else 0.0)
                        p[i, j] = (1.0 - d) * p0[i, j] - hc * rel * gt[i, j]
                fg = gt
            ach = hs * rel
            for a in range(10):
                for i in range(3):
                    v = 0.0
                    for j in range(3):
                        v += p[i, j] * grad[a, j]
                    f[3 * a + i] += w * v
            for b in range(4):
                f[30 + b] += w * (
                    ach * bary[b]
                    + hs * l2 * (gchi[0] * gl[b, 0] + gchi[1] * gl[b, 1] + gchi[2] * gl[b, 2])
                )
            if not tangent:
                continue
            # dP/dgrad u
            for i in range(3):
                for j in range(3):
                    for q in range(3):
                        for r in range(3):
                            a6 = _MAP[i, j]
                            b6 = _MAP[q, r]
                            v = -hc * rel * heq[a6, b6] / (_W[a6] * _W[b6])
                            if finite:
                                hv = 0.0
                                for bb in range(3):
                                    for dd2 in range(3):
                                        hv += (
                                            fm[i, bb]
                                            * fm[q, dd2]
                                            * heq[_MAP[bb, j], _MAP[r, dd2]]
                                            / (_W[_MAP[bb, j]] * _W[_MAP[r, dd2]])
                                        )
                                v = -hc * rel * (hv + (gt[r, j] if i == q else 0.0))
                                c1 = mu - lam * lnj
                                v += (1.0 - d) * (
                                    c1 * finv[r, i] * finv[j, q]
                                    + lam * finv[j, i] * finv[r, q]
                                    + (mu if i == q and j == r else 0.0)
                                )
                            else:
                                el6 = 0.0
                                if i == j and q == r:
                                    el6 += lam
                                if i == q and j == r:
                                    el6 += mu
                                if i == r and j == q:
                                    el6 += mu
                                v += (1.0 - d) * el6
                            v += hc * fg[i, j] * fg[q, r]
                            duu[i, j, q, r] = v
                    duc[i, j] = -dd * p0[i, j] - hc * fg[i, j]
                    dcu[i, j] = -hs * fg[i, j]
            kk = ke[ee]
            for a in range(10):
                for i in range(3):
                    for c in range(10):
                        for q in range(3):
                            v = 0.0
                            for j in range(3):
                                for r in range(3):
                                    v += grad[a, j] * duu[i, j, q, r] * grad[c, r]
                            kk[3 * a + i, 3 * c + q] += w * v
                    v = 0.0
                    for j in range(3):
                        v += grad[a, j] * duc[i, j]
                    for b in range(4):
                        kk[3 * a + i, 30 + b] += w * v * bary[b]
                for q in range(3):
                    v = 0.0
                    for r in range(3):
                        v += dcu[q, r] * grad[a, r]
                    for b in range(4):
                        kk[30 + b, 3 * a + q] += w * bary[b] * v
            for b in range(4):
                for c in range(4):
                    dot = gl[b, 0] * gl[c, 0] + gl[b, 1] * gl[c, 1] + gl[b, 2] * gl[c, 2]
                    kk[30 + b, 30 + c] += w * hs * (bary[b] * bary[c] + l2 * dot)


class DamageProblem(MixedProblem):
    """A body of :class:`MicromorphicDamage` materials (one for every phase, or a dict of
    the phases), in ``"small"`` or ``"finite"`` strains: its generalized degree of
    freedom (component :data:`CHI` = 3 of the corner nodes) is ``χ``."""

    def __init__(self, mesh: TetMesh, materials, *, kinematics: str = "small"):
        if kinematics not in ("small", "finite"):
            raise ValueError("kinematics: 'small' or 'finite'")
        if isinstance(materials, MicromorphicDamage):
            materials = {int(p): materials for p in np.unique(mesh.phase)}
        self.space = MixedSpace(mesh, 1)
        self.mesh = self.space.mesh
        self.finite = kinematics == "finite"
        phases = sorted(materials)
        if not np.isin(self.mesh.phase, phases).all():
            raise ValueError("DamageProblem: a material for every phase of the mesh")
        index = {p: k for k, p in enumerate(phases)}
        self.materials = materials
        self._mat = np.array([index[int(p)] for p in self.mesh.phase], dtype=np.int64)
        self._par = np.array([materials[p].parameters() for p in phases])

    def _initial_state(self):
        return np.zeros((self.mesh.n_elements, len(WEIGHTS), N_STATE))

    def _assemble(self, values, state, tangent):
        space = self.space
        n_el = self.mesh.n_elements
        rhs = np.zeros(space.n_dofs)
        data = space.new_matrix_data() if tangent else np.zeros(0)
        new_state = np.empty_like(state)
        n_loc = space.n_local
        for first in range(0, n_el, CHUNK):
            count = min(CHUNK, n_el - first)
            fe = np.empty((count, n_loc))
            ke = np.empty((count, n_loc, n_loc)) if tangent else np.empty((count, 1, 1))
            _damage_chunk(self.mesh.nodes, space.elements, first, self._mat, self._par, self.finite,
                          values, space.offset, POINTS, WEIGHTS, state, new_state, fe, ke,
                          tangent)  # fmt: skip
            space.add(first, ke, fe, data, rhs)
        return rhs, (data if tangent else None), new_state

    def damage(self, state) -> np.ndarray:
        """The damage of the elements (the mean of their points)."""
        return state[:, :, 1].mean(axis=1)
