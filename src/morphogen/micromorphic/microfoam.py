"""The microfoam model: anisotropic compressible plasticity of metallic foams in a
micromorphic continuum (Dillard, Forest & Ienny 2006, after Forest & Lorentz 2004 and
Forest et al. 2005), in small strains.

*Compressible plasticity* (Badiche et al. 2000). ``ε = ε^e + ε^p``, ``σ = C : ε^e`` with
orthotropic elasticity ``C``; the yield function ``f = σ_eq - Y(p)`` with ``σ_eq² = (3c/2)
σ' : H : σ' + F (P : σ)²`` (``σ'`` the deviator; in the axes of the sheet, ``H =
diag(H_a, H_b, H_c)`` on the normal components and ``H_d, H_e, H_f`` on the shears 12,
23, 31, ``P = diag(P, Q, R)``): Hill's criterion when ``F = 0``, the isotropic elliptic
criterion of powder metallurgy and foams (Besson & Abouaf 1989; Deshpande & Fleck 2000)
when ``H = 1``, ``P = 1``. Associative flow ``ε̇^p = ṗ ∂σ_eq/∂σ`` (``σ : ε̇^p = σ_eq ṗ``),
hardening ``Y = σ0 + Q1 (1 - exp(-b1 p)) + Q2 (1 - exp(-b2 p)) + H p`` (at least 0.1 % of
σ0), and for fracture a
softening beyond a critical ``p_c``: ``Y`` times ``r + (1 - r) exp(-((p - p_c)/w)²)``
(the article's branch, "a parabolic and an exponential term", is not given in closed
form: ours starts with a zero slope and tends to the residual fraction ``r``). The
quadratic criterion makes the implicit update a scalar equation: with ``λ = Δp /
σ_eq`` and ``M`` the matrix of the quadratic form, ``σ = (C⁻¹ + λ M)⁻¹ : (ε - ε^p_n)``
and ``σ_eq(λ) = Y(p_n + λ σ_eq(λ))``, solved by a safeguarded Newton method; the
consistent tangent is ``Ξ - (1 - λY') (Ξn) ⊗ (Ξn) / (Y' + (1 - λY') n:Ξ:n)``, ``Ξ = (C⁻¹ +
λM)⁻¹``, ``n = M:σ / σ_eq``.

*Micromorphic continuum* (Eringen & Suhubi 1964, Mindlin 1964; Forest & Sievert 2003):
the displacement ``u`` and a full, non-symmetric microdeformation ``χ``; the strain ``ε``,
the relative deformation ``e = ∇u - χ`` and the microdeformation gradient ``K = ∇χ``
(``K_ijk = χ_ij,k``), with the power of internal forces ``σ : ε̇ + s : ė + S ⋮ K̇``; the
balance equations ``div(σ + s) = 0`` and ``div S + s = 0``, the tractions ``(σ + s) · n``
and the double tractions ``S · n``. The plasticity concerns ``σ`` only; the relative and
higher-order stresses are linear and isotropic, ``s = λ_s tr(e) 1 + 2μ_s e^sym + 2μ_c
e^skew`` and ``S_ijk = l_c² (λ δ_ij K_llk + 2μ K_ijk)``: the moduli ``λ_s, μ_s, μ_c``
penalize the departure of the microdeformation from the deformation (the model tends to
a second gradient one as they grow), ``λ, μ`` are of the order of the elastic moduli and
``l_c`` is the characteristic length. In one dimension (``χ_22(x_2)`` only), the
microdeformation follows ``χ''' = ω² χ'`` with ``ω² = k_s E_t / ((k_s + E_t) A l_c²)``,
``k_s = λ_s + 2μ_s``, ``A = λ + 2μ`` and ``E_t`` the tangent modulus: boundary layers of
length ``1/ω`` in hardening, sine-shaped localization bands of wavelength ``2π/|ω|``
when ``E_t < 0`` (the article's Appendix A, ``λ_s = λ = 0``, ``μ_s = μ_c = μ``).

*Elements* (:mod:`morphogen.micromorphic.space`): quadratic displacements and linear
microdeformations on tetrahedra; the article used quadratic elements with reduced
integration in plane stress. A classical continuum (no ``l_c``) has the displacements
only.

*Finite strains* (``kinematics="finite"``, total Lagrangian; :mod:`morphogen.micromorphic.
finite`, after Forest & Sievert 2003 and Forest 2009): the compressible plasticity acts on
the logarithmic strain ``ε = 1/2 ln C`` and its conjugate stress ``T`` exactly as on the
small strain and the Cauchy stress (orthotropic elasticity, the criterion, the flow and
the hardening; the plastic strain additive in ``ε``, its volume change ``tr ε^p`` the
logarithm of the plastic change of volume: the densification of the foam), pulled back to
the Piola-Kirchhoff stresses; the microdeformation ``χ = 1 + χ̃`` enters through the
objective measures ``Υ = χ⁻¹ F`` (relative, ``e = Υ - 1``) and ``K = χ⁻¹ Grad χ`` (higher
order), with the linear isotropic laws of small strains in them. The generalized degrees
of freedom remain ``χ̃`` (zero at rest), and the formulation reduces to that of small
strains for small strains. The tangent of an element is exact for the macroscopic part
and the central differences of the exact gradient of the micromorphic energy at its
points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from morphogen.homogenization.fe import _corner_gradients, _shape_gradients
from morphogen.mesh.tetra import TetMesh
from morphogen.micromorphic.finite import N_Z, PAIRS, hencky, micro_hessian, pull_back
from morphogen.micromorphic.space import CHUNK, POINTS, WEIGHTS, MixedProblem, MixedSpace

#: Mandel components: (11, 22, 33, 23, 31, 12), shears times sqrt(2)
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))
_MAP = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]], dtype=np.int64)
_W = np.array([1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0)])
#: The internal variables of a point: the plastic strain (Mandel) and p
N_STATE = 7
#: Components of the microdeformation: chi_ij is generalized degree of freedom 3 i + j
CHI = {f"chi{i + 1}{j + 1}": 3 + 3 * i + j for i in range(3) for j in range(3)}


def _mandel(voigt: np.ndarray) -> np.ndarray:
    """A Voigt matrix (engineering shear strains) as a Mandel matrix."""
    return _W[:, None] * np.asarray(voigt, dtype=float) * _W[None, :]


def _mandel_rotation(axes: np.ndarray) -> np.ndarray:
    """The Mandel matrix of the change of axes ``A_global = R^T A_material R`` (rows of
    ``R``: the material axes in the global frame)."""
    r = np.asarray(axes, dtype=float)
    q = np.empty((6, 6))
    for j, (k, m) in enumerate(_PAIRS):
        basis = np.zeros((3, 3))
        basis[k, m] = basis[m, k] = 1.0 / _W[j] if k != m else 1.0
        turned = r.T @ basis @ r
        q[:, j] = [turned[a, b] * _W[i] for i, (a, b) in enumerate(_PAIRS)]
    return q


@dataclass(frozen=True)
class Microfoam:
    """A material of the microfoam model (see the module), in MPa and mm or any
    consistent units: elasticity ``stiffness`` (6 x 6 Voigt matrix in the material axes,
    order (11, 22, 33, 23, 31, 12), engineering shear strains) or ``young`` and
    ``poisson``; the criterion ``c``, ``F``, ``H`` (``H_a, H_b, H_c, H_d, H_e, H_f``: the
    normal components, then the shears 12, 23, 31, as the article) and ``P`` (``P, Q,
    R``); hardening ``yield_stress`` σ0 (infinite: elastic), ``voce`` ((Q1, b1), (Q2,
    b2)...) and the linear ``hardening`` H; ``softening`` (p_c, r, w) or None; the
    material ``axes`` (rows, in the global frame; default: the global axes); and the
    micromorphic ``length`` l_c (None: a classical continuum), the higher-order moduli
    ``micro`` (λ, μ) and the relative moduli ``relative`` (λ_s, μ_s, μ_c)."""

    young: float | None = None
    poisson: float | None = None
    stiffness: tuple | None = None
    yield_stress: float = math.inf
    voce: tuple = ()
    hardening: float = 0.0
    c: float = 1.0
    F: float = 0.0
    H: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    P: tuple = (1.0, 1.0, 1.0)
    softening: tuple | None = None
    axes: tuple | None = None
    length: float | None = None
    micro: tuple = (0.0, 0.0)
    relative: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.stiffness is None and (self.young is None or self.poisson is None):
            raise ValueError("Microfoam: a stiffness, or young and poisson")
        if len(self.voce) > 2:
            raise ValueError("Microfoam: at most two Voce terms")
        if len(self.H) != 6 or len(self.P) != 3:
            raise ValueError("Microfoam: six H and three P")
        if self.length is not None and not self.length >= 0:
            raise ValueError("Microfoam: a non-negative length")

    @classmethod
    def nickel_foam(cls, length: float | None = 0.1, axes=None, softening=None) -> Microfoam:
        """The nickel foam of relative density 0.035 of the article (MPa, mm; axes RD, TD,
        ND): its Tables 2 (the in-plane elastic moduli of plane stress, completed by a
        decoupled thickness, ``C13 = C23 = 0``) and 3 (``l_c = 0.1`` mm from the hole size
        effect; 0.06 mm from the cracked panels)."""
        stiffness = np.zeros((6, 6))
        stiffness[0, 0], stiffness[0, 1], stiffness[1, 1] = 464.0, 172.0, 239.0
        stiffness[1, 0] = 172.0
        stiffness[2, 2] = 239.0
        stiffness[3, 3] = stiffness[4, 4] = stiffness[5, 5] = 105.0
        return cls(
            stiffness=tuple(map(tuple, stiffness)),
            yield_stress=0.3,
            voce=((0.51, 611.0), (7442.0, 0.0013)),
            c=1.0,
            F=0.000682,
            H=(0.41, 1.8, 1.75, 1.49, 1.0, 1.0),
            P=(1.0, 36.6, 22.6),
            softening=softening,
            axes=axes,
            length=length,
            micro=(111.0, 167.0),
            relative=(11000.0, 17000.0, 10000.0),
        )

    def elastic_matrix(self) -> np.ndarray:
        """The Mandel elasticity matrix in the global axes."""
        if self.stiffness is not None:
            voigt = np.asarray(self.stiffness, dtype=float)
        else:
            lam = self.young * self.poisson / ((1 + self.poisson) * (1 - 2 * self.poisson))
            mu = self.young / (2 * (1 + self.poisson))
            voigt = np.zeros((6, 6))
            voigt[:3, :3] = lam
            voigt[np.arange(3), np.arange(3)] += 2 * mu
            voigt[np.arange(3, 6), np.arange(3, 6)] = mu
        c = _mandel(voigt)
        if self.axes is not None:
            q = _mandel_rotation(self.axes)
            c = q @ c @ q.T
        return c

    def criterion_matrix(self) -> np.ndarray:
        """The Mandel matrix ``M`` of ``σ_eq² = σ : M : σ`` in the global axes."""
        ha, hb, hc, hd, he, hf = (float(v) for v in self.H)
        dev = np.eye(3) - 1.0 / 3.0
        m = np.zeros((6, 6))
        m[:3, :3] = 1.5 * self.c * dev @ np.diag([ha, hb, hc]) @ dev
        m[3, 3], m[4, 4], m[5, 5] = 1.5 * self.c * he, 1.5 * self.c * hf, 1.5 * self.c * hd
        p = np.array([*map(float, self.P), 0.0, 0.0, 0.0])
        m += self.F * np.outer(p, p)
        if self.axes is not None:
            q = _mandel_rotation(self.axes)
            m = q @ m @ q.T
        return m

    def hardening_parameters(self) -> np.ndarray:
        voce = list(self.voce) + [(0.0, 0.0)] * (2 - len(self.voce))
        pc, r, w = self.softening if self.softening is not None else (0.0, 1.0, 1.0)
        return np.array([self.yield_stress, voce[0][0], voce[0][1], voce[1][0], voce[1][1],
                         self.hardening, pc, r, w])  # fmt: skip

    def micro_parameters(self) -> np.ndarray:
        length = 0.0 if self.length is None else float(self.length)
        return np.array([length**2, *map(float, self.micro), *map(float, self.relative)])

    def flow(self, stress) -> np.ndarray:
        """The direction ``∂σ_eq/∂σ`` (3 x 3, global axes) of the plastic flow at a stress
        (3 x 3)."""
        s = np.asarray(stress, dtype=float)
        vec = np.array([s[a, b] * _W[i] for i, (a, b) in enumerate(_PAIRS)])
        m = self.criterion_matrix()
        n = m @ vec / math.sqrt(vec @ m @ vec)
        out = np.empty((3, 3))
        for i, (a, b) in enumerate(_PAIRS):
            out[a, b] = out[b, a] = n[i] / _W[i]
        return out

    def equivalent(self, stress) -> float:
        """``σ_eq`` of a stress (3 x 3)."""
        s = np.asarray(stress, dtype=float)
        vec = np.array([s[a, b] * _W[i] for i, (a, b) in enumerate(_PAIRS)])
        return float(math.sqrt(vec @ self.criterion_matrix() @ vec))


def characteristic_length(material: Microfoam, tangent_modulus: float) -> float:
    """``1/ω`` of the one-dimensional micromorphic problem (see the module) for a
    tangent modulus ``E_t`` of the uniaxial response: a boundary layer (``E_t > 0``), or
    ``1/|ω|`` with the wavelength ``2π/|ω|`` of a localization band (``E_t < 0``)."""
    lam_h, mu_h = material.micro
    lam_s, mu_s, _ = material.relative
    ks, a = lam_s + 2 * mu_s, lam_h + 2 * mu_h
    omega2 = ks * tangent_modulus / ((ks + tangent_modulus) * a * material.length**2)
    return 1.0 / math.sqrt(abs(omega2))


# -- the point -------------------------------------------------------------------------------


@njit(cache=True)
def _yield(h, p):
    """``Y(p)`` and ``Y'(p)`` (see the module), at least 0.1 % of σ0."""
    yh = h[0] + h[1] * (1.0 - math.exp(-h[2] * p)) + h[3] * (1.0 - math.exp(-h[4] * p)) + h[5] * p
    dyh = h[1] * h[2] * math.exp(-h[2] * p) + h[3] * h[4] * math.exp(-h[4] * p) + h[5]
    if h[6] > 0.0 and p > h[6]:
        z = (p - h[6]) / h[8]
        ez = math.exp(-z * z)
        g = h[7] + (1.0 - h[7]) * ez
        dg = (1.0 - h[7]) * ez * (-2.0 * z / h[8])
        yh, dyh = yh * g, dyh * g + yh * dg
    if yh < 1e-3 * h[0]:
        return 1e-3 * h[0], 0.0
    return yh, dyh


@njit(cache=True)
def _plastic_point(cm, cinv, mm, h, eps, epsp_old, p_old, sig, epsp, tan):
    """The implicit update of a point (Mandel vectors): the stress, plastic strain and
    consistent tangent into ``sig``, ``epsp``, ``tan``; returns ``p``."""
    de = eps - epsp_old
    sig_tr = cm @ de
    seq_tr = math.sqrt(max(sig_tr @ (mm @ sig_tr), 0.0))
    y0, dy0 = _yield(h, p_old)
    if not math.isfinite(seq_tr):  # a diverging trial state: no update, the step is divided
        sig[:] = np.nan
        epsp[:] = epsp_old
        tan[:, :] = cm
        return p_old
    if not seq_tr > y0 * (1.0 + 1e-12):
        sig[:] = sig_tr
        epsp[:] = epsp_old
        tan[:, :] = cm
        return p_old
    n_tr = (mm @ sig_tr) / seq_tr
    denom = n_tr @ (cm @ n_tr) + dy0
    lam = max((seq_tr - y0) / denom, 0.0) / y0 if denom > 0.0 else 1e-6 / y0
    lo, hi = 0.0, math.inf
    xi = np.empty((6, 6))
    s = np.empty(6)
    ms = np.empty(6)
    seq, dy = y0, dy0
    for _ in range(100):
        xi[:, :] = np.linalg.inv(cinv + lam * mm)
        s[:] = xi @ de
        ms[:] = mm @ s
        seq = math.sqrt(max(s @ ms, 1e-300))
        p = p_old + lam * seq
        y, dy = _yield(h, p)
        g = seq - y
        if abs(g) <= 1e-12 * max(y, 1e-300):
            break
        if g > 0.0:
            lo = lam
        else:
            hi = lam
        dseq = -(ms @ (xi @ ms)) / seq
        dg = dseq - dy * (seq + lam * dseq)
        new = lam - g / dg if dg < 0.0 else -1.0
        if not lo < new < hi:
            new = 0.5 * (lo + hi) if hi < math.inf else 2.0 * lo + 1e-12
        if abs(new - lam) <= 1e-15 * max(lam, 1e-300):
            break
        if not new < 1e300:  # no plastic solution (an infinite flow)
            sig[:] = np.nan
            epsp[:] = epsp_old
            tan[:, :] = cm
            return p_old
        lam = new
    sig[:] = s
    epsp[:] = epsp_old + lam * ms
    n = ms / seq
    xn = xi @ n
    a = 1.0 - lam * dy
    tan[:, :] = xi - a * np.outer(xn, xn) / (dy + a * (n @ xn))
    return p_old + lam * seq


@njit(cache=True)
def plastic_update(cm, cinv, mm, h, eps, epsp_old, p_old):
    """A point (Mandel vectors): returns the stress, the plastic strain, ``p`` and the
    consistent tangent (for checks)."""
    sig = np.empty(6)
    epsp = np.empty(6)
    tan = np.empty((6, 6))
    p = _plastic_point(cm, cinv, mm, h, eps, epsp_old, p_old, sig, epsp, tan)
    return sig, epsp, p, tan


# -- the elements ----------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _microfoam_chunk(nodes, elements, first, mat, cm, cinv, mm, hard, micro, plastic, values,
                     offset, n_g, gp, gw, state_old, state_new, fe, ke, tangent):  # fmt: skip
    """Element vectors and matrices of the elements ``first...``, the internal variables
    of their points into ``state_new``."""
    for ee in prange(fe.shape[0]):
        e = first + ee
        el = elements[e]
        m = mat[e]
        x = np.empty((4, 3))
        for a in range(4):
            x[a] = nodes[el[a]]
        gl = np.empty((4, 3))
        vol = abs(_corner_gradients(x, gl))
        ul = np.empty((10, 3))
        for a in range(10):
            for i in range(3):
                ul[a, i] = values[offset[el[a]] + i]
        gv = np.zeros((4, 9))
        for b in range(4):
            for k in range(n_g):
                gv[b, k] = values[offset[el[b]] + 3 + k]
        l2, lam_h, mu_h = micro[m, 0], micro[m, 1], micro[m, 2]
        lam_s, mu_s, mu_c = micro[m, 3], micro[m, 4], micro[m, 5]
        ds = np.zeros((9, 9))
        if n_g > 0:
            for i in range(3):
                for j in range(3):
                    for q in range(3):
                        for r in range(3):
                            v = 0.0
                            if i == j and q == r:
                                v += lam_s
                            if i == q and j == r:
                                v += mu_s + mu_c
                            if i == r and j == q:
                                v += mu_s - mu_c
                            ds[3 * i + j, 3 * q + r] = v
        f = fe[ee]
        f[:] = 0.0
        if tangent:
            ke[ee, :, :] = 0.0
        grad = np.empty((10, 3))
        hu = np.empty((3, 3))
        eps = np.empty(6)
        sig = np.empty(6)
        epsp = np.empty(6)
        tan = np.empty((6, 6))
        t = np.empty((3, 3))
        s = np.zeros((3, 3))
        big_s = np.zeros((3, 3, 3))
        d = np.empty((9, 9))
        gmat = np.empty((3, 9))
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
            r2 = math.sqrt(2.0)
            eps[0], eps[1], eps[2] = hu[0, 0], hu[1, 1], hu[2, 2]
            eps[3] = (hu[1, 2] + hu[2, 1]) / r2
            eps[4] = (hu[2, 0] + hu[0, 2]) / r2
            eps[5] = (hu[0, 1] + hu[1, 0]) / r2
            so = state_old[e, g]
            sn = state_new[e, g]
            if plastic[m]:
                p = _plastic_point(
                    cm[m], cinv[m], mm[m], hard[m], eps, so[:6], so[6], sig, epsp, tan
                )
                sn[:6] = epsp
                sn[6] = p
            else:
                sig[:] = cm[m] @ eps
                tan[:, :] = cm[m]
                sn[:] = so
            for i in range(3):
                for j in range(3):
                    k6 = _MAP[i, j]
                    t[i, j] = sig[k6] / _W[k6]
            if n_g > 0:
                tr = 0.0
                for i in range(3):
                    tr += hu[i, i]
                    for b in range(4):
                        tr -= bary[b] * gv[b, 4 * i]
                for i in range(3):
                    for j in range(3):
                        eij = hu[i, j]
                        eji = hu[j, i]
                        for b in range(4):
                            eij -= bary[b] * gv[b, 3 * i + j]
                            eji -= bary[b] * gv[b, 3 * j + i]
                        s[i, j] = mu_s * (eij + eji) + mu_c * (eij - eji)
                        if i == j:
                            s[i, j] += lam_s * tr
                        t[i, j] += s[i, j]
                for k in range(3):
                    trk = 0.0
                    for q in range(3):
                        for b in range(4):
                            trk += gv[b, 4 * q] * gl[b, k]
                    for i in range(3):
                        for j in range(3):
                            kij = 0.0
                            for b in range(4):
                                kij += gv[b, 3 * i + j] * gl[b, k]
                            big_s[i, j, k] = l2 * (
                                2.0 * mu_h * kij + (lam_h * trk if i == j else 0.0)
                            )
            for a in range(10):
                for i in range(3):
                    v = 0.0
                    for j in range(3):
                        v += t[i, j] * grad[a, j]
                    f[3 * a + i] += w * v
            if n_g > 0:
                for b in range(4):
                    for i in range(3):
                        for j in range(3):
                            v = -s[i, j] * bary[b]
                            for k in range(3):
                                v += big_s[i, j, k] * gl[b, k]
                            f[30 + 9 * b + 3 * i + j] += w * v
            if not tangent:
                continue
            kk = ke[ee]
            for i in range(3):
                for j in range(3):
                    for q in range(3):
                        for r in range(3):
                            a6 = _MAP[i, j]
                            b6 = _MAP[q, r]
                            d[3 * i + j, 3 * q + r] = (
                                tan[a6, b6] / (_W[a6] * _W[b6]) + ds[3 * i + j, 3 * q + r]
                            )
            for a in range(10):
                for i in range(3):
                    for qr in range(9):
                        v = 0.0
                        for j in range(3):
                            v += grad[a, j] * d[3 * i + j, qr]
                        gmat[i, qr] = v
                for c in range(10):
                    for i in range(3):
                        for q in range(3):
                            v = 0.0
                            for r in range(3):
                                v += gmat[i, 3 * q + r] * grad[c, r]
                            kk[3 * a + i, 3 * c + q] += w * v
            if n_g == 0:
                continue
            for a in range(10):
                for i in range(3):
                    for qr in range(9):
                        v = 0.0
                        for j in range(3):
                            v += grad[a, j] * ds[3 * i + j, qr]
                        for b in range(4):
                            kk[3 * a + i, 30 + 9 * b + qr] -= w * bary[b] * v
                            kk[30 + 9 * b + qr, 3 * a + i] -= w * bary[b] * v
            for b in range(4):
                for c in range(4):
                    dot = gl[b, 0] * gl[c, 0] + gl[b, 1] * gl[c, 1] + gl[b, 2] * gl[c, 2]
                    for ij in range(9):
                        for qr in range(9):
                            v = bary[b] * bary[c] * ds[ij, qr]
                            if ij == qr:
                                v += l2 * 2.0 * mu_h * dot
                            if ij % 4 == 0 and qr % 4 == 0:
                                v += l2 * lam_h * dot
                            kk[30 + 9 * b + ij, 30 + 9 * c + qr] += w * v


@njit(parallel=True, cache=True)
def _microfoam_finite_chunk(nodes, elements, first, mat, cm, cinv, mm, hard, micro, plastic,
                            values, offset, n_g, gp, gw, state_old, state_new, fe, ke,
                            tangent):  # fmt: skip
    """The finite-strain counterpart of :func:`_microfoam_chunk` (see the module): at
    each point the gradient and Hessian of the energy by ``z = (F, χ, Grad χ)``, the
    macroscopic part exact, then ``Bᵀ g`` and ``Bᵀ H B`` with the linear map ``B`` of the
    degrees of freedom of the element to ``z``."""
    n_z = N_Z if n_g > 0 else 9
    n_loc = 30 + 4 * n_g
    for ee in prange(fe.shape[0]):
        e = first + ee
        el = elements[e]
        m = mat[e]
        x = np.empty((4, 3))
        for a in range(4):
            x[a] = nodes[el[a]]
        gl = np.empty((4, 3))
        vol = abs(_corner_gradients(x, gl))
        ul = np.empty((10, 3))
        for a in range(10):
            for i in range(3):
                ul[a, i] = values[offset[el[a]] + i]
        gv = np.zeros((4, 9))
        for b in range(4):
            for k in range(n_g):
                gv[b, k] = values[offset[el[b]] + 3 + k]
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
        none6 = np.zeros(6)
        pk1 = np.empty((3, 3))
        dpdf = np.empty((9, 9))
        dp9 = np.empty(9)
        z = np.empty(N_Z)
        gz = np.zeros(N_Z)
        hz = np.zeros((N_Z, N_Z))
        g45 = np.empty(N_Z)
        h45 = np.empty((N_Z, N_Z))
        bmat = np.zeros((n_z, n_loc))
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
            so = state_old[e, g]
            sn = state_new[e, g]
            if hencky(hu, mu3, q, eps) != 0:  # an inverted point: the step is divided
                f[:] = np.nan
                sn[:] = so
                continue
            if plastic[m]:
                p = _plastic_point(cm[m], cinv[m], mm[m], hard[m], eps, so[:6], so[6], sig, epsp,
                                   tan)  # fmt: skip
                sn[:6] = epsp
                sn[6] = p
            else:
                sig[:] = cm[m] @ eps
                tan[:, :] = cm[m]
                sn[:] = so
            pull_back(hu, mu3, q, sig, tan, none6, pk1, dpdf, dp9)
            gz[:] = 0.0
            hz[:, :] = 0.0
            for i in range(3):
                for j in range(3):
                    gz[3 * i + j] = pk1[i, j]
            hz[:9, :9] = dpdf
            bmat[:, :] = 0.0
            for i in range(3):
                for j in range(3):
                    for a in range(10):
                        bmat[3 * i + j, 3 * a + i] = grad[a, j]
            if n_g > 0:
                for i in range(3):
                    for j in range(3):
                        z[3 * i + j] = hu[i, j] + (1.0 if i == j else 0.0)
                        v = 1.0 if i == j else 0.0
                        for b in range(4):
                            v += bary[b] * gv[b, 3 * i + j]
                        z[9 + 3 * i + j] = v
                        for k in range(3):
                            v = 0.0
                            for b in range(4):
                                v += gv[b, 3 * i + j] * gl[b, k]
                            z[18 + 9 * i + 3 * j + k] = v
                micro_hessian(z, micro[m], g45, h45)
                gz += g45
                hz += h45
                for ij in range(9):
                    for b in range(4):
                        bmat[9 + ij, 30 + 9 * b + ij] = bary[b]
                        for k in range(3):
                            bmat[18 + 3 * ij + k, 30 + 9 * b + ij] = gl[b, k]
            bt = np.ascontiguousarray(bmat.T)
            f += w * (bt @ np.ascontiguousarray(gz[:n_z]))
            if tangent:
                ke[ee] += w * (bt @ (np.ascontiguousarray(hz[:n_z, :n_z]) @ bmat))


@njit(parallel=True, cache=True)
def _point_fields_finite(nodes, elements, mat, cm, values, offset, n_g, gp, state, out):
    """At the points of every element in finite strains: the logarithmic strain (Mandel),
    the Cauchy stress (Mandel), ``p``, the microdeformation ``χ̃`` (9) and ``J`` into
    ``out`` (elements x points x 23)."""
    for e in prange(elements.shape[0]):
        el = elements[e]
        m = mat[e]
        x = np.empty((4, 3))
        for a in range(4):
            x[a] = nodes[el[a]]
        gl = np.empty((4, 3))
        _corner_gradients(x, gl)
        grad = np.empty((10, 3))
        mu3 = np.empty(3)
        q = np.empty((3, 3))
        eps = np.empty(6)
        pk1 = np.empty((3, 3))
        dpdf = np.empty((9, 9))
        dp9 = np.empty(9)
        none6 = np.zeros(6)
        for g in range(gp.shape[0]):
            _shape_gradients(2, gl, gp[g], grad)
            hu = np.zeros((3, 3))
            for a in range(10):
                for i in range(3):
                    for j in range(3):
                        hu[i, j] += values[offset[el[a]] + i] * grad[a, j]
            o = out[e, g]
            if hencky(hu, mu3, q, eps) != 0:
                o[:] = np.nan
                continue
            sig = cm[m] @ (eps - state[e, g, :6])
            pull_back(hu, mu3, q, sig, cm[m], none6, pk1, dpdf, dp9)
            fm = np.eye(3) + hu
            jac = np.linalg.det(fm)
            cauchy = pk1 @ fm.T / jac
            o[:6] = eps
            for k in range(6):
                o[6 + k] = cauchy[PAIRS[k, 0], PAIRS[k, 1]] * _W[k]
            o[12] = state[e, g, 6]
            for k in range(9):
                v = 0.0
                if n_g > 0:
                    for b in range(4):
                        v += gp[g, b] * values[offset[el[b]] + 3 + k]
                o[13 + k] = v
            o[22] = jac


@njit(parallel=True, cache=True)
def _point_fields(nodes, elements, mat, cm, values, offset, n_g, gp, state, out):
    """At the points of every element: the strain (Mandel), the stress ``σ`` (Mandel),
    ``p`` and the microdeformation (9) into ``out`` (elements x points x 22)."""
    for e in prange(elements.shape[0]):
        el = elements[e]
        m = mat[e]
        x = np.empty((4, 3))
        for a in range(4):
            x[a] = nodes[el[a]]
        gl = np.empty((4, 3))
        _corner_gradients(x, gl)
        grad = np.empty((10, 3))
        eps = np.empty(6)
        r2 = math.sqrt(2.0)
        for g in range(gp.shape[0]):
            _shape_gradients(2, gl, gp[g], grad)
            hu = np.zeros((3, 3))
            for a in range(10):
                for i in range(3):
                    for j in range(3):
                        hu[i, j] += values[offset[el[a]] + i] * grad[a, j]
            eps[0], eps[1], eps[2] = hu[0, 0], hu[1, 1], hu[2, 2]
            eps[3] = (hu[1, 2] + hu[2, 1]) / r2
            eps[4] = (hu[2, 0] + hu[0, 2]) / r2
            eps[5] = (hu[0, 1] + hu[1, 0]) / r2
            o = out[e, g]
            o[:6] = eps
            o[6:12] = cm[m] @ (eps - state[e, g, :6])
            o[12] = state[e, g, 6]
            for k in range(9):
                v = 0.0
                if n_g > 0:
                    for b in range(4):
                        v += gp[g, b] * values[offset[el[b]] + 3 + k]
                o[13 + k] = v


class MicrofoamProblem(MixedProblem):
    """A body of microfoam materials (see the module): ``materials`` a
    :class:`Microfoam` (every phase) or a dict of the phases of the mesh, all
    micromorphic (with a ``length``) or all classical, in ``"small"`` or ``"finite"``
    strains. The generalized degrees of freedom of a corner are the components ``chi_ij``
    (:data:`CHI`: component ``3 + 3 i + j``; in finite strains those of ``χ̃ = χ - 1``);
    :meth:`run` prescribes them as the displacements."""

    def __init__(self, mesh: TetMesh, materials, *, kinematics: str = "small"):
        if kinematics not in ("small", "finite"):
            raise ValueError("kinematics: 'small' or 'finite'")
        self.finite = kinematics == "finite"
        if isinstance(materials, Microfoam):
            materials = {int(p): materials for p in np.unique(mesh.phase)}
        lengths = {m.length is None for m in materials.values()}
        if len(lengths) != 1:
            raise ValueError("MicrofoamProblem: every material micromorphic, or none")
        self.micromorphic = not lengths.pop()
        self.space = MixedSpace(mesh, 9 if self.micromorphic else 0)
        self.mesh = self.space.mesh
        phases = sorted(materials)
        index = {p: k for k, p in enumerate(phases)}
        if not np.isin(self.mesh.phase, phases).all():
            raise ValueError("MicrofoamProblem: a material for every phase of the mesh")
        self.materials = materials
        self._mat = np.array([index[int(p)] for p in self.mesh.phase], dtype=np.int64)
        mats = [materials[p] for p in phases]
        self._cm = np.array([m.elastic_matrix() for m in mats])
        self._cinv = np.array([np.linalg.inv(c) for c in self._cm])
        self._mm = np.array([m.criterion_matrix() for m in mats])
        self._hard = np.array([m.hardening_parameters() for m in mats])
        self._micro = np.array([m.micro_parameters() for m in mats])
        self._plastic = np.array([math.isfinite(m.yield_stress) for m in mats])

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
            chunk = _microfoam_finite_chunk if self.finite else _microfoam_chunk
            chunk(self.mesh.nodes, space.elements, first, self._mat, self._cm, self._cinv,
                  self._mm, self._hard, self._micro, self._plastic, values, space.offset,
                  space.n_g, POINTS, WEIGHTS, state, new_state, fe, ke, tangent)  # fmt: skip
            space.add(first, ke, fe, data, rhs)
        return rhs, (data if tangent else None), new_state

    def fields(self, values, state) -> dict:
        """The fields of the elements (means of their points): ``strain`` and ``stress``
        (elements x 3 x 3; in finite strains the logarithmic strain and the Cauchy
        stress), the cumulated plastic strain ``p``, the microdeformation ``chi``
        (elements x 3 x 3; ``χ̃`` in finite strains) and, in finite strains, the change of
        volume ``J``."""
        space = self.space
        width = 23 if self.finite else 22
        out = np.empty((self.mesh.n_elements, len(WEIGHTS), width))
        point_fields = _point_fields_finite if self.finite else _point_fields
        point_fields(self.mesh.nodes, space.elements, self._mat, self._cm, values, space.offset,
                     space.n_g, POINTS, state, out)  # fmt: skip
        mean = out.mean(axis=1)
        strain = np.empty((len(mean), 3, 3))
        stress = np.empty((len(mean), 3, 3))
        for i, (a, b) in enumerate(_PAIRS):
            strain[:, a, b] = strain[:, b, a] = mean[:, i] / _W[i]
            stress[:, a, b] = stress[:, b, a] = mean[:, 6 + i] / _W[i]
        fields = {"strain": strain, "stress": stress, "p": mean[:, 12],
                  "chi": mean[:, 13:22].reshape(-1, 3, 3)}  # fmt: skip
        if self.finite:
            fields["J"] = mean[:, 22]
        return fields
