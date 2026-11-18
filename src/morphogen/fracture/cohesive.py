"""Cohesive zone elements: the interfaces of composites and fracture along given surfaces,
in small and finite strains.

Traction-separation laws, per unit area of the reference interface (``δ = u⁺ - u⁻`` the
separation, ``n`` the normal from side ``-`` to side ``+``, ``δ_n = δ · n`` and ``δ_s = δ -
δ_n n``):

* :class:`Bilinear` (Camanho & Dávila 2002; Turon, Camanho, Costa & Dávila 2006): a penalty
  stiffness ``K``, the onset tractions ``N`` (normal) and ``S`` (shear), the fracture
  energies ``G_Ic`` and ``G_IIc`` and the exponent ``η`` of the mixed-mode criterion of
  Benzeggagh & Kenane. The equivalent separation ``λ = sqrt(<δ_n>² + |δ_s|²)`` and the mode
  mixity ``B = |δ_s|² / λ²`` give the separations of onset, ``λ0² = δn0² + (δs0² - δn0²)
  B^η``, and of failure, ``λf = (δn0 δnf + (δs0 δsf - δn0 δnf) B^η) / λ0`` (``δn0 = N/K``,
  ``δs0 = S/K``, ``δnf = 2 G_Ic / N``, ``δsf = 2 G_IIc / S``), hence the damage ``d = λf (λ -
  λ0) / (λ (λf - λ0))``, irreversible (its largest value); the traction is ``(1 - d) K δ``,
  except in compression, ``t_n = K δ_n`` (no interpenetration, no damage). The energy of
  a separation at fixed mixity is ``G_Ic + (G_IIc - G_Ic) B^η``.
* :class:`Exponential` (Ortiz & Pandolfi 1999): the effective opening ``δ = sqrt(<δ_n>² +
  β² |δ_s|²)`` and the effective traction ``e σc (δ/δc) exp(-δ/δc)`` along its envelope
  (``G_c = e σc δc``), unloading linearly to the origin; ``t = (t_eff / δ) (<δ_n> n + β²
  δ_s)``, and a penalty ``K_c`` in compression.
* :class:`Contact`: the contact of the lips of a crack, a penalty against their
  interpenetration and no normal traction otherwise (:mod:`morphogen.fracture.mixed`),
  frictionless or with Coulomb's friction.

Friction (the three laws take a coefficient ``friction`` μ; Alfano & Sacco 2006): a point of
damage ``d`` is a share ``1 - d`` of sound interface, which carries the traction of its law,
and a share ``d`` of broken interface, whose lips touch under a negative ``δ_n`` (the
pressure ``p = K <-δ_n>`` of the penalty) and rub: an elastic stick, ``τ = K_t (δ_s - δ_s^p)``,
within Coulomb's cone ``|τ| <= μ p``, and beyond it a slip ``δ_s^p`` along ``τ`` (a
non-associated flow: the slip opens nothing), integrated by a radial return, ``τ = μ p τ_trial
/ |τ_trial|`` and ``δ_s^p = δ_s - τ / K_t``. The shear traction is ``(1 - d) K δ_s + d τ``: that
of the sound law alone while the interface is intact, Coulomb's ``μ p`` once it is broken,
and the lips of a crack (:class:`Contact`: ``d = 1``) stick or slide. Lips apart carry
nothing and forget their slip. The slip of each point (two components in its frame, which
turns with the interface in finite strains) and the energy that friction dissipated there
per unit area, ``∫ d τ · dδ_s^p``, are kept with the history of its law; the tangent, the
derivative of the tractions of the radial return, is not symmetric where the lips slide
(``∂τ/∂δ_n = -μ K τ/|τ|``).

Interface elements: zero-thickness elements of six nodes between the faces of two linear
tetrahedra, integrated at three points of their mid-surface; of twelve nodes between the
faces of two 10-node tetrahedra (the six nodes of each face: the separation is quadratic,
as the displacements of the two sides), integrated at six points (a rule exact to degree
4: three points would leave openings without traction), with the normal and the area
element of each point: their faces may be curved, as those of the meshes whose nodes lie
on the interfaces of the phases (:mod:`morphogen.mesh.curved`). In finite strains the
normal and tangents are those of the current mid-surface (the tractions turn with the
interface: frame-indifferent under rigid rotations), and the tangent of an element is the
derivative of its forces by central differences.

Insertion (:func:`insert_interfaces`): the faces between elements of different phases
(the interfaces of composites), or the faces chosen by a mask or a function of their
centroids (a crack path), are opened by duplicating their nodes: the elements around each
of their nodes are grouped by the faces (through that node) by which they remain
connected, and each group gets its own copy, so that the nodes of a crack front, around
which the two sides remain connected, are not split. A mesh of 10-node elements is opened
by its corners, and the mid-edge nodes of the two lips made again where they were (on the
curved interfaces of the phases).

Solver (:class:`CohesiveProblem`): quasi-static, the displacements prescribed on node sets
(and dead loads) grown with a load factor; Newton's method on the bulk (linear elasticity
of the materials in small strains; a compressible neo-Hookean solid, ``W = μ/2 (I1 - 3) -
μ ln J + λ/2 (ln J)²``, total Lagrangian, in finite strains, both with 4- or 10-node
elements; or the behaviours of morphoGen's points, elasto-viscoplastic, crystalline or
MFront, with their internal variables, on 4-node elements:
:mod:`morphogen.fracture.bulk`) and the interfaces, with a
line search and the steps divided when they fail; results: the reactions, the damage and
the cracked area at each step, and optionally the displacements. The local instabilities
of softening interfaces (an element failing at once: no equilibrium beyond a limit point
under the prescribed displacements) are passed by the viscous regularization of Gao &
Bower (2004), the load factor as time: ``d = (d_n + Δt/ζ d_∞) / (1 + Δt/ζ)``, the steps
being divided until the damage, slowed, lets Newton's method converge; the solution tends
to the rate-independent one as the viscosity ``ζ`` tends to 0.

Calibration (Turon, Dávila, Camanho & Costa 2007; Linke & Lammering 2023): the length of
the cohesive zone ``l_cz = M E G / T²``, the strength that coarse meshes need, ``T =
sqrt(9π E G / (32 N_e l_e))``, and the corrected beam theory of the double cantilever
beam (load and crack length against the opening).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit, prange

from morphogen.homogenization.fe import GAUSS, _geometry, _point_gradients
from morphogen.mesh.tetra import TET_EDGES, TET_FACES, TetMesh

BILINEAR = 0
EXPONENTIAL = 1
CONTACT = 2
#: The values kept at an interface point: the history of its law (the damage, or the
#: largest effective opening), the two components of the frictional slip of its broken
#: share, the energy dissipated there by friction per unit area, and its normal separation
#: (negative or zero: lips that touch)
STATE = 5
#: Damage above which an integration point is broken (its area counts as cracked)
BROKEN = 0.999
_GAUSS = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
#: The rule of the triangle exact to degree 4 (Dunavant): ξ, η and the weights (sum 1) of
#: its six points, for the interface elements of six nodes a side
_GA, _GB = 0.445948490915965, 0.091576213509771
_GAUSS6 = np.array(
    [
        [_GA, _GA, 0.223381589678011],
        [1.0 - 2.0 * _GA, _GA, 0.223381589678011],
        [_GA, 1.0 - 2.0 * _GA, 0.223381589678011],
        [_GB, _GB, 0.109951743655322],
        [1.0 - 2.0 * _GB, _GB, 0.109951743655322],
        [_GB, 1.0 - 2.0 * _GB, 0.109951743655322],
    ]
)


@dataclass(frozen=True)
class Bilinear:
    """The bilinear mixed-mode law (see the module): ``stiffness`` K (penalty),
    ``normal_strength`` N, ``shear_strength`` S, the fracture energies ``mode_I`` and
    ``mode_II`` (G_Ic, G_IIc) and the exponent ``eta`` of Benzeggagh & Kenane."""

    stiffness: float
    normal_strength: float
    shear_strength: float
    mode_I: float
    mode_II: float
    eta: float = 2.0
    friction: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.stiffness,
            self.normal_strength,
            self.shear_strength,
            self.mode_I,
            self.mode_II,
            self.eta,
        )
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Bilinear: positive stiffness, strengths, energies and eta")
        _check_friction("Bilinear", self.friction)
        for strength, energy, name in (
            (self.normal_strength, self.mode_I, "normal"),
            (self.shear_strength, self.mode_II, "shear"),
        ):
            if 2 * energy / strength <= strength / self.stiffness:
                raise ValueError(
                    f"Bilinear: the {name} failure separation 2 G / T must exceed "
                    "the onset T / K (a stiffer penalty or more energy)"
                )

    def params(self) -> np.ndarray:
        scale = min(self.normal_strength, self.shear_strength) / self.stiffness
        return np.array(
            [
                self.stiffness,
                self.normal_strength,
                self.shear_strength,
                self.mode_I,
                self.mode_II,
                self.eta,
                0.0,
                1e-6 * scale,
                self.friction,
                self.stiffness,
                0.0,
            ]
        )

    kind = BILINEAR


@dataclass(frozen=True)
class Exponential:
    """The irreversible exponential law of Ortiz & Pandolfi (see the module): the cohesive
    ``strength`` σc, the fracture ``energy`` G_c, the shear-to-normal ratio ``beta``, the
    ``penalty`` stiffness in compression (by default ``10 e σc / δc``), which is also that
    of the stick of the broken share, and the coefficient of ``friction`` of that share."""

    strength: float
    energy: float
    beta: float = 1.0
    penalty: float | None = None
    friction: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (self.strength, self.energy, self.beta)):
            raise ValueError("Exponential: positive strength, energy and beta")
        _check_friction("Exponential", self.friction)

    @property
    def critical_opening(self) -> float:
        """``δc = G_c / (e σc)``."""
        return self.energy / (math.e * self.strength)

    def params(self) -> np.ndarray:
        dc = self.critical_opening
        penalty = self.penalty if self.penalty is not None else 10 * math.e * self.strength / dc
        return np.array(
            [self.strength, self.energy, self.beta, penalty, 0.0, 0.0, 0.0, 1e-6 * dc,
             self.friction, penalty, 0.0]
        )  # fmt: skip

    kind = EXPONENTIAL


@dataclass(frozen=True)
class Contact:
    """Contact of the lips of a crack: a penalty ``stiffness`` against their
    interpenetration (``t_n = K min(δ_n, 0)``), no normal traction otherwise; with a
    coefficient of ``friction``, lips in contact stick (the ``tangential`` stiffness of
    the stick, by default that of the penalty) or slide under Coulomb's ``μ |t_n|`` (see
    the module). Points whose history is negative (:meth:`CohesiveProblem.run`) are
    bonded by the two stiffnesses, in tension as in compression: the ligament ahead of
    the front of a crack whose lips rub (:func:`morphogen.fracture.mixed.solve_crack`)."""

    stiffness: float
    friction: float = 0.0
    tangential: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.stiffness) and self.stiffness > 0):
            raise ValueError("Contact: a positive stiffness")
        _check_friction("Contact", self.friction)
        if self.tangential is not None and not (
            math.isfinite(self.tangential) and self.tangential > 0
        ):
            raise ValueError("Contact: a positive tangential stiffness")

    def params(self) -> np.ndarray:
        stick = self.stiffness if self.tangential is None else self.tangential
        return np.array(
            [self.stiffness, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-9, self.friction, stick, 0.0]
        )

    kind = CONTACT


def _check_friction(name: str, friction: float) -> None:
    if not (math.isfinite(friction) and friction >= 0):
        raise ValueError(f"{name}: a coefficient of friction of 0 or more")


# -- laws ------------------------------------------------------------------------------------


@njit(cache=True)
def _friction(mu, stick, pressure, s1, s2, old, new, predict):
    """Coulomb's friction of the broken share of a point under the ``pressure`` of its
    lips (see the module): the shear traction of the tangential separation ``(s1, s2)``
    from the slip of ``old``, by a radial return; the slip of ``new``. ``predict``: lips
    that stick, whatever their pressure (the predictor of a step). A negative ``stick``
    is an elastic slip: the stiffness of the stick is then that which reaches Coulomb's
    bound at this slip, ``μ p / |stick|`` — the bound is reached over one length whatever
    the pressure, and lips that come to touch, whose pressure and slip both start from
    nothing, stick first (the direction of a slip that starts has no meaning)."""
    if mu <= 0.0:
        new[1], new[2] = old[1], old[2]
        return 0.0, 0.0
    if pressure <= 0.0 and (not predict or stick < 0.0):
        # lips apart: no traction, the slip forgotten
        new[1], new[2] = s1, s2
        return 0.0, 0.0
    if stick < 0.0:
        stick = mu * pressure / -stick
    t1 = stick * (s1 - old[1])
    t2 = stick * (s2 - old[2])
    norm = math.sqrt(t1 * t1 + t2 * t2)
    limit = mu * pressure
    if norm <= limit or predict:
        new[1], new[2] = old[1], old[2]
        return t1, t2
    t1 *= limit / norm
    t2 *= limit / norm
    new[1] = s1 - t1 / stick
    new[2] = s2 - t2 / stick
    return t1, t2


@njit(cache=True)
def _law(kind, par, d, old, new):
    """Traction (local components: normal, shears) of the separation ``d`` from the state
    ``old`` of the point (:data:`STATE` values: the history of the law, bilinear: the
    damage, exponential: the largest effective opening; the frictional slip; the energy
    dissipated by friction; the normal separation), and its state ``new``. With the
    switch ``par[10]``, the law of the predictor of a step: the lips that touched at its
    start (a normal separation of 0 or less) stay closed (the penalty, whatever the sign
    of their separation) and stick."""
    t = np.empty(3)
    dn, s1, s2 = d[0], d[1], d[2]
    dt2 = s1 * s1 + s2 * s2
    dnp = dn if dn > 0.0 else 0.0
    hist = old[0]
    new[4] = dn
    predict = par[10] > 0.0 and old[4] <= 0.0
    if kind == 0:
        k, nn, ss, gi, gii, eta = par[0], par[1], par[2], par[3], par[4], par[5]
        lam2 = dnp * dnp + dt2
        lam = math.sqrt(lam2)
        dn0, ds0 = nn / k, ss / k
        dnf, dsf = 2.0 * gi / nn, 2.0 * gii / ss
        b = dt2 / lam2 if lam2 > 0.0 else 0.0
        bh = b**eta
        lam0 = math.sqrt(dn0 * dn0 + (ds0 * ds0 - dn0 * dn0) * bh)
        lamf = (dn0 * dnf + (ds0 * dsf - dn0 * dnf) * bh) / lam0
        dmg = 0.0
        if lam > lam0:
            dmg = min(1.0, lamf * (lam - lam0) / (lam * (lamf - lam0)))
        dmg = max(dmg, hist)
        if par[6] > 0.0:  # viscous regularization (Gao & Bower 2004): the damage lags
            dmg = (hist + par[6] * dmg) / (1.0 + par[6])
        new[0] = dmg
        tau1, tau2 = _friction(
            par[8], par[9], -k * dn if dn < 0.0 else 0.0, s1, s2, old, new, predict
        )
        t[0] = (1.0 - dmg) * k * dn if dn > 0.0 and not predict else k * dn
        t[1] = (1.0 - dmg) * k * s1 + dmg * tau1
        t[2] = (1.0 - dmg) * k * s2 + dmg * tau2
        new[3] = old[3] + dmg * (tau1 * (new[1] - old[1]) + tau2 * (new[2] - old[2]))
        return t
    if kind == 2:  # contact: interpenetration only, and the friction of the lips
        new[0] = hist
        if hist < 0.0:  # a bonded point (the ligament ahead of a front): the penalty alone
            t[0] = par[0] * dn
            t[1] = par[9] * s1
            t[2] = par[9] * s2
            new[1], new[2], new[3] = old[1], old[2], old[3]
            return t
        tau1, tau2 = _friction(
            par[8], par[9], -par[0] * dn if dn < 0.0 else 0.0, s1, s2, old, new, predict
        )
        t[0] = par[0] * dn if dn < 0.0 or predict else 0.0
        t[1] = tau1
        t[2] = tau2
        new[3] = old[3] + tau1 * (new[1] - old[1]) + tau2 * (new[2] - old[2])
        return t
    sc, gc, beta, kc = par[0], par[1], par[2], par[3]
    dc = gc / (math.e * sc)
    de = math.sqrt(dnp * dnp + beta * beta * dt2)
    kap = max(hist, de)
    if par[6] > 0.0:
        kap = (hist + par[6] * kap) / (1.0 + par[6])
    new[0] = kap
    decay = math.exp(-kap / dc)
    ratio = math.e * sc / dc * decay  # the secant at the largest opening
    tau1, tau2 = _friction(par[8], par[9], -kc * dn if dn < 0.0 else 0.0, s1, s2, old, new, predict)
    t[0] = ratio * dnp + (kc * dn if dn < 0.0 or predict else 0.0)
    t[1] = ratio * beta * beta * s1 + (1.0 - decay) * tau1
    t[2] = ratio * beta * beta * s2 + (1.0 - decay) * tau2
    new[3] = old[3] + (1.0 - decay) * (tau1 * (new[1] - old[1]) + tau2 * (new[2] - old[2]))
    return t


@njit(cache=True)
def _law_tangent(kind, par, d, old):
    """Derivative of the traction by the separation (central differences, the state of
    the start of the step)."""
    h = par[7]
    out = np.empty((3, 3))
    new = np.empty(STATE)
    for j in range(3):
        dp = d.copy()
        dp[j] += h
        tp = _law(kind, par, dp, old, new)
        dm = d.copy()
        dm[j] -= h
        tm = _law(kind, par, dm, old, new)
        for i in range(3):
            out[i, j] = (tp[i] - tm[i]) / (2.0 * h)
    return out


@njit(cache=True)
def _damage(kind, par, hist):
    if kind == 2:
        return 1.0 if hist >= 0.0 else 0.0  # the lips of a crack: broken; bonded points: sound
    if kind == 0:
        return hist
    dc = par[1] / (math.e * par[0])
    return 1.0 - math.exp(-hist / dc)


def traction(law, separation, history=0.0, slip=None):
    """The traction (local components: normal, shear, shear) of a law for a separation
    (local components) from a ``history`` (damage or largest opening), and the new
    history: a material point of the interfaces. With ``slip`` (the frictional slip of
    the point at the start: two components, zeros for a new point), also the new slip
    and the energy that friction dissipated per unit area on the way."""
    old = np.zeros(STATE)
    old[0] = float(history)
    if slip is not None:
        old[1:3] = np.asarray(slip, dtype=float)
    new = np.zeros(STATE)
    t = _law(law.kind, law.params(), np.asarray(separation, dtype=float), old, new)
    if slip is None:
        return t, float(new[0])
    return t, float(new[0]), new[1:3].copy(), float(new[3])


@njit(cache=True)
def _point_tractions(kind, par, local, state, out):
    """The tractions (local components) of the separations ``local`` (elements x points x
    3) of points in the states ``state``."""
    new = np.empty(STATE)
    for e in range(local.shape[0]):
        for g in range(local.shape[1]):
            out[e, g] = _law(kind, par, local[e, g], state[e, g], new)


def lip_tractions(nodes, interfaces, displacement, law, result=None, finite: bool = False):
    """The tractions that the points of interface elements of three nodes a side carry at
    the end of a solution: ``(tractions, jumps)``, elements x 3 points x 3, in global
    components; the traction is that on the side ``-`` (its opposite on the side ``+``),
    of the separation of ``displacement`` (nodes x 3) from the history and the frictional
    slip of ``result`` (a :class:`CohesiveResult`; none: new points)."""
    minus, plus = interfaces.minus[:, :3], interfaces.plus[:, :3]
    u = np.asarray(displacement, dtype=float).reshape(-1, 3)
    x = nodes[minus]
    if finite:  # the frame of the current mid-surface
        x = x + 0.5 * (u[minus] + u[plus])
    a, b = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    t1 = a / np.linalg.norm(a, axis=1, keepdims=True)
    frame = np.stack([normal, t1, np.cross(normal, t1)], axis=1)  # rows: n, t1, t2
    shapes = np.array([[1.0 - xi - eta, xi, eta] for xi, eta in _GAUSS])
    jumps = np.einsum("gi,eic->egc", shapes, u[plus] - u[minus])
    local = np.ascontiguousarray(np.einsum("eab,egb->ega", frame, jumps))
    state = np.zeros((len(minus), 3, STATE))
    if result is not None and len(minus):
        state[..., 0] = result.history
        if result.slip is not None:
            state[..., 1:3] = result.slip
    out = np.zeros_like(local)
    if len(minus):
        _point_tractions(law.kind, law.params(), local, state, out)
    return np.einsum("eab,ega->egb", frame, out), jumps


# -- interface elements ------------------------------------------------------------------------


@njit(cache=True)
def _frame(x0, x1, x2):
    """The normal, two tangents and twice the area of a triangle."""
    a = x1 - x0
    b = x2 - x0
    nrm = np.array(
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    )
    area2 = math.sqrt(nrm[0] ** 2 + nrm[1] ** 2 + nrm[2] ** 2)
    n = nrm / area2
    t1 = a / math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    t2 = np.array(
        [n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]]
    )
    r = np.empty((3, 3))
    r[0], r[1], r[2] = n, t1, t2
    return r, area2


@njit(cache=True)
def _element_forces(xr, ue, finite, kind, par, hist, hist_new, gauss):
    """Forces (18: the three nodes of side - then of side +) of an interface element of
    reference positions ``xr`` (3 x 3, its mid-surface) and nodal displacements ``ue`` (6
    x 3), and the new state of its points (``hist``, ``hist_new``: points x STATE)."""
    x = xr.copy()
    if finite:  # the current mid-surface
        for i in range(3):
            x[i] = xr[i] + 0.5 * (ue[i] + ue[3 + i])
    r, _ = _frame(x[0], x[1], x[2])
    _, area_ref = _frame(xr[0], xr[1], xr[2])
    f = np.zeros(18)
    for g in range(3):
        xi, eta = gauss[g, 0], gauss[g, 1]
        shape = np.array([1.0 - xi - eta, xi, eta])
        w = area_ref / 6.0
        delta = np.zeros(3)
        for i in range(3):
            delta += shape[i] * (ue[3 + i] - ue[i])
        local = r @ delta
        t = _law(kind, par, local, hist[g], hist_new[g])
        tg = r.T @ t
        for i in range(3):
            for c in range(3):
                f[9 + 3 * i + c] += w * shape[i] * tg[c]
                f[3 * i + c] -= w * shape[i] * tg[c]
    return f


@njit(parallel=True, cache=True)
def _interfaces(nodes, minus, plus, u, finite, kind, par, hist, hist_new, gauss, tangent, fe, ke):
    """Forces and tangents of all the interface elements (rows: elements)."""
    for e in prange(minus.shape[0]):
        xr = np.empty((3, 3))
        ue = np.empty((6, 3))
        for i in range(3):
            xr[i] = nodes[minus[e, i]]
            ue[i] = u[minus[e, i]]
            ue[3 + i] = u[plus[e, i]]
        hn = np.empty((3, STATE))
        fe[e] = _element_forces(xr, ue, finite, kind, par, hist[e], hn, gauss)
        hist_new[e, :, :] = hn
        if not tangent:
            continue
        if not finite:  # B^T D B, D the derivative of the law in the reference frame
            r, area2 = _frame(xr[0], xr[1], xr[2])
            k = np.zeros((18, 18))
            for g in range(3):
                xi, eta = gauss[g, 0], gauss[g, 1]
                shape = np.array([1.0 - xi - eta, xi, eta])
                delta = np.zeros(3)
                for i in range(3):
                    delta += shape[i] * (ue[3 + i] - ue[i])
                d = r.T @ _law_tangent(kind, par, r @ delta, hist[e, g]) @ r
                w = area2 / 6.0
                for i in range(3):
                    for j in range(3):
                        for a in range(3):
                            for b in range(3):
                                v = w * shape[i] * shape[j] * d[a, b]
                                k[9 + 3 * i + a, 9 + 3 * j + b] += v
                                k[3 * i + a, 3 * j + b] += v
                                k[9 + 3 * i + a, 3 * j + b] -= v
                                k[3 * i + a, 9 + 3 * j + b] -= v
            ke[e] = k
        else:  # central differences of the forces: the turning frame included
            scale = 1e-7 * math.sqrt(area2_of(xr))
            dummy = np.empty((3, STATE))
            for j in range(18):
                node, c = j // 3, j % 3
                up = ue.copy()
                up[node, c] += scale
                fp = _element_forces(xr, up, finite, kind, par, hist[e], dummy, gauss)
                um = ue.copy()
                um[node, c] -= scale
                fm = _element_forces(xr, um, finite, kind, par, hist[e], dummy, gauss)
                for i in range(18):
                    ke[e, i, j] = (fp[i] - fm[i]) / (2.0 * scale)


@njit(cache=True)
def area2_of(xr):
    _, a2 = _frame(xr[0], xr[1], xr[2])
    return a2


@njit(cache=True)
def _shape6(xi, eta, shape, dxi, deta):
    """The functions of the six nodes of a triangle (corners, then the middles of the
    edges 0-1, 1-2 and 2-0) at a point, and their derivatives."""
    zeta = 1.0 - xi - eta
    shape[0] = zeta * (2.0 * zeta - 1.0)
    shape[1] = xi * (2.0 * xi - 1.0)
    shape[2] = eta * (2.0 * eta - 1.0)
    shape[3] = 4.0 * zeta * xi
    shape[4] = 4.0 * xi * eta
    shape[5] = 4.0 * eta * zeta
    dxi[0] = 1.0 - 4.0 * zeta
    dxi[1] = 4.0 * xi - 1.0
    dxi[2] = 0.0
    dxi[3] = 4.0 * (zeta - xi)
    dxi[4] = 4.0 * eta
    dxi[5] = -4.0 * eta
    deta[0] = 1.0 - 4.0 * zeta
    deta[1] = 0.0
    deta[2] = 4.0 * eta - 1.0
    deta[3] = -4.0 * xi
    deta[4] = 4.0 * xi
    deta[5] = 4.0 * (zeta - eta)


@njit(cache=True)
def _frame6(x, dxi, deta):
    """The normal and two tangents of a triangle of six nodes at a point (the derivatives
    of its functions there), and twice its area element."""
    a = np.zeros(3)
    b = np.zeros(3)
    for i in range(6):
        for c in range(3):
            a[c] += dxi[i] * x[i, c]
            b[c] += deta[i] * x[i, c]
    nrm = np.array(
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    )
    area2 = math.sqrt(nrm[0] ** 2 + nrm[1] ** 2 + nrm[2] ** 2)
    n = nrm / area2
    t1 = a / math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    t2 = np.array(
        [n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]]
    )
    r = np.empty((3, 3))
    r[0], r[1], r[2] = n, t1, t2
    return r, area2


@njit(cache=True)
def _element_forces6(xr, ue, finite, kind, par, hist, hist_new, gauss):
    """Forces (36: the six nodes of side - then of side +) of an interface element of
    reference positions ``xr`` (6 x 3, its mid-surface, which may be curved) and nodal
    displacements ``ue`` (12 x 3), and the new state of its six points."""
    x = xr.copy()
    if finite:  # the current mid-surface
        for i in range(6):
            x[i] = xr[i] + 0.5 * (ue[i] + ue[6 + i])
    shape = np.empty(6)
    dxi = np.empty(6)
    deta = np.empty(6)
    f = np.zeros(36)
    for g in range(6):
        _shape6(gauss[g, 0], gauss[g, 1], shape, dxi, deta)
        r, _ = _frame6(x, dxi, deta)
        _, area_ref = _frame6(xr, dxi, deta)
        w = 0.5 * gauss[g, 2] * area_ref
        delta = np.zeros(3)
        for i in range(6):
            delta += shape[i] * (ue[6 + i] - ue[i])
        t = _law(kind, par, r @ delta, hist[g], hist_new[g])
        tg = r.T @ t
        for i in range(6):
            for c in range(3):
                f[18 + 3 * i + c] += w * shape[i] * tg[c]
                f[3 * i + c] -= w * shape[i] * tg[c]
    return f


@njit(parallel=True, cache=True)
def _interfaces6(nodes, minus, plus, u, finite, kind, par, hist, hist_new, gauss, tangent, fe, ke):
    """Forces and tangents of all the interface elements of six nodes a side."""
    for e in prange(minus.shape[0]):
        xr = np.empty((6, 3))
        ue = np.empty((12, 3))
        for i in range(6):
            xr[i] = nodes[minus[e, i]]
            ue[i] = u[minus[e, i]]
            ue[6 + i] = u[plus[e, i]]
        hn = np.empty((6, STATE))
        fe[e] = _element_forces6(xr, ue, finite, kind, par, hist[e], hn, gauss)
        hist_new[e, :, :] = hn
        if not tangent:
            continue
        if not finite:  # B^T D B, D the derivative of the law in the frame of each point
            shape = np.empty(6)
            dxi = np.empty(6)
            deta = np.empty(6)
            k = np.zeros((36, 36))
            for g in range(6):
                _shape6(gauss[g, 0], gauss[g, 1], shape, dxi, deta)
                r, area2 = _frame6(xr, dxi, deta)
                delta = np.zeros(3)
                for i in range(6):
                    delta += shape[i] * (ue[6 + i] - ue[i])
                d = r.T @ _law_tangent(kind, par, r @ delta, hist[e, g]) @ r
                w = 0.5 * gauss[g, 2] * area2
                for i in range(6):
                    for j in range(6):
                        for a in range(3):
                            for b in range(3):
                                v = w * shape[i] * shape[j] * d[a, b]
                                k[18 + 3 * i + a, 18 + 3 * j + b] += v
                                k[3 * i + a, 3 * j + b] += v
                                k[18 + 3 * i + a, 3 * j + b] -= v
                                k[3 * i + a, 18 + 3 * j + b] -= v
            ke[e] = k
        else:  # central differences of the forces: the turning frames included
            scale = 1e-7 * math.sqrt(area2_of(xr))
            dummy = np.empty((6, STATE))
            for j in range(36):
                node, c = j // 3, j % 3
                up = ue.copy()
                up[node, c] += scale
                fp = _element_forces6(xr, up, finite, kind, par, hist[e], dummy, gauss)
                um = ue.copy()
                um[node, c] -= scale
                fm = _element_forces6(xr, um, finite, kind, par, hist[e], dummy, gauss)
                for i in range(36):
                    ke[e, i, j] = (fp[i] - fm[i]) / (2.0 * scale)


# -- bulk in finite strains --------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _neo_hookean(grads, weights, elements, u, mu, lam, fe, ke, tangent):
    """Forces and tangents of tetrahedra of a compressible neo-Hookean solid (total
    Lagrangian): ``P = μ (F - F^-T) + λ ln J F^-T`` at the integration points of each
    element (``grads``: elements x points x nodes x 3, the gradients of the functions of
    its nodes in the reference configuration; ``weights``: their volumes)."""
    nn = elements.shape[1]
    for e in prange(elements.shape[0]):
        out = np.zeros(3 * nn)
        k = np.zeros((3 * nn, 3 * nn))
        m, l = mu[e], lam[e]
        for q in range(grads.shape[1]):
            g = grads[e, q]
            f = np.eye(3)
            for a in range(nn):
                ua = u[elements[e, a]]
                for i in range(3):
                    for j in range(3):
                        f[i, j] += ua[i] * g[a, j]
            jac = np.linalg.det(f)
            if jac <= 0.0:
                jac = 1e-12  # inverted: the residual becomes large, the step is divided
            finv = np.linalg.inv(f)
            lnj = math.log(jac)
            p = m * (f - finv.T) + l * lnj * finv.T
            v = weights[e, q]
            for a in range(nn):
                for i in range(3):
                    s = 0.0
                    for j in range(3):
                        s += p[i, j] * g[a, j]
                    out[3 * a + i] += v * s
            if not tangent:
                continue
            c1 = m - l * lnj
            # A_ijkl g_j g_l = μ δ_ik (g_a · g_b) + c1 (F^-1 g_b)_i (F^-1 g_a)_k
            #                  + λ (F^-1 g_a)_i (F^-1 g_b)_k, with (F^-1 g)_i = g_j F^-1_ji
            h = np.zeros((nn, 3))
            for a in range(nn):
                for i in range(3):
                    for jj in range(3):
                        h[a, i] += g[a, jj] * finv[jj, i]
            for a in range(nn):
                for b in range(nn):
                    dot = g[a, 0] * g[b, 0] + g[a, 1] * g[b, 1] + g[a, 2] * g[b, 2]
                    for i in range(3):
                        for kk in range(3):
                            s = c1 * h[b, i] * h[a, kk] + l * h[a, i] * h[b, kk]
                            if i == kk:
                                s += m * dot
                            k[3 * a + i, 3 * b + kk] += v * s
        fe[e] = out
        if tangent:
            ke[e] = k


@njit(cache=True)
def _reference_gradients(nodes, elements, order, gp, gw, grads, weights):
    """The gradients of the functions of the nodes of each element at its integration
    points (those of the reference configuration; curved 10-node elements: with the
    Jacobian of each point) and the volumes of the points."""
    x = np.empty((4, 3))
    gl = np.empty((4, 3))
    dev = np.zeros((6, 3))
    out = np.empty((elements.shape[1], 3))
    for e in range(elements.shape[0]):
        vol, curved = _geometry(nodes, elements, e, x, gl, dev)
        for q in range(gp.shape[0]):
            ratio = _point_gradients(order, curved, gl, dev, gp[q], out)
            grads[e, q] = out
            weights[e, q] = abs(vol) * gw[q] * ratio


def _tet_gradients(nodes, elements):
    """Gradients of the shape functions (elements x 4 x 3) and volumes of linear tetrahedra."""
    x = nodes[elements]
    dm = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)
    inv = np.linalg.inv(dm)  # rows: gradients of N1, N2, N3
    grads = np.concatenate([-inv.sum(axis=1, keepdims=True), inv], axis=1)
    return np.ascontiguousarray(grads), np.abs(np.linalg.det(dm)) / 6.0


def _gradients_at_points(nodes, elements):
    """Gradients of the functions of the nodes (elements x points x nodes x 3) and volumes
    (elements x points) at the integration points of tetrahedra of 4 nodes (one point) or
    of 10 nodes (four)."""
    order = 1 if elements.shape[1] == 4 else 2
    gp, gw = GAUSS[order]
    shape = (len(elements), len(gw))
    grads = np.zeros((*shape, elements.shape[1], 3))
    weights = np.zeros(shape)
    _reference_gradients(
        np.ascontiguousarray(nodes, dtype=float),
        np.ascontiguousarray(elements, dtype=np.int64),
        order,
        gp,
        gw,
        grads,
        weights,
    )
    return grads, weights


# -- insertion ---------------------------------------------------------------------------------


@dataclass
class Interfaces:
    """Interface elements: the nodes of their side ``minus`` and of their side ``plus``
    (matching; elements x 3 between 4-node tetrahedra, elements x 6 between 10-node ones:
    the corners, then the mid-edge nodes of the edges 0-1, 1-2 and 2-0), the phases of the
    two sides (elements x 2; -1 unknown); the normal goes from side - to side +."""

    minus: np.ndarray
    plus: np.ndarray
    phases: np.ndarray

    @property
    def n_elements(self) -> int:
        return len(self.minus)

    @property
    def order(self) -> int:
        """1 for faces of three nodes, 2 for faces of six."""
        return 1 if self.minus.shape[1] == 3 else 2

    @property
    def weights(self) -> np.ndarray:
        """The shares of the area of an element of its integration points."""
        return np.full(3, 1.0 / 3.0) if self.order == 1 else _GAUSS6[:, 2].copy()

    def subset(self, keep) -> Interfaces:
        """The interface elements ``keep`` (mask or indices)."""
        keep = np.asarray(keep)
        return Interfaces(self.minus[keep], self.plus[keep], self.phases[keep])

    def centroids(self, mesh: TetMesh) -> np.ndarray:
        return mesh.nodes[self.minus[:, :3]].mean(axis=1)

    def areas(self, mesh: TetMesh) -> np.ndarray:
        """The areas of the elements (of their side -; curved faces of six nodes: the
        integral of their area element)."""
        x = mesh.nodes[self.minus]
        if self.order == 1:
            return 0.5 * np.linalg.norm(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=1)
        shape, dxi, deta = np.empty(6), np.empty(6), np.empty(6)
        out = np.zeros(len(x))
        for xi, eta, weight in _GAUSS6:
            _shape6(xi, eta, shape, dxi, deta)
            a = np.einsum("i,eic->ec", dxi, x)
            b = np.einsum("i,eic->ec", deta, x)
            out += 0.5 * weight * np.linalg.norm(np.cross(a, b), axis=1)
        return out


def _interior_faces(tets: np.ndarray):
    """The faces shared by two tetrahedra: the sorted corners of the faces of the elements
    (4 per element), and for each shared face its two places among them (the face ``k %
    4`` of the element ``k // 4``)."""
    keys = np.sort(tets[:, np.asarray(TET_FACES)].reshape(-1, 3), axis=1)
    order = np.lexsort(keys.T[::-1])
    sorted_keys = keys[order]
    same = np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
    return keys, order[:-1][same], order[1:][same]


def interior_faces(mesh: TetMesh) -> tuple[np.ndarray, np.ndarray]:
    """The faces that two elements of a mesh share, in the order of the masks of
    :func:`insert_interfaces`: their corners (faces x 3) and their two elements (faces x
    2)."""
    tets = np.asarray(mesh.elements[:, :4], dtype=np.int64)
    keys, first, second = _interior_faces(tets)
    return keys[first], np.stack([first // 4, second // 4], axis=1)


def insert_interfaces(mesh: TetMesh, faces="phases", *, sides=None) -> tuple[TetMesh, Interfaces]:
    """Open faces of a tetrahedral mesh (see the module): ``faces`` is ``"phases"`` (the
    faces between different phases; side - the lower phase), a boolean mask of the
    interior faces or a function of the centroids of their corners (rows) returning one.
    Returns the mesh with the duplicated nodes and the interface elements of the opened
    faces (all of them: keep a subset for the cohesive ones, the others being free crack
    faces). ``sides``: a label per element, the faces kept only between different labels
    (the two sides of a crack). A mesh of 10-node elements is opened by its corners; its
    mid-edge nodes are made again, each lip with its own, where they were (the curved
    edges of the interfaces of the phases stay curved), and the interface elements have
    six nodes a side."""
    if mesh.order == 2:
        opened, interfaces = insert_interfaces(mesh.linear(), faces, sides=sides)
        return _with_mid_nodes(mesh, opened, interfaces)
    tets = np.array(mesh.elements, dtype=np.int64)
    local = np.asarray(TET_FACES)
    keys, first, second = _interior_faces(tets)
    e1, f1, e2, f2 = first // 4, first % 4, second // 4, second % 4
    centroids = mesh.nodes[keys[first]].mean(axis=1)
    if isinstance(faces, str):
        if faces != "phases":
            raise ValueError("insert_interfaces: 'phases', a mask or a function of centroids")
        selected = mesh.phase[e1] != mesh.phase[e2]
        swap = mesh.phase[e1] > mesh.phase[e2]  # side -: the lower phase (the matrix, say)
        e1, e2 = np.where(swap, e2, e1), np.where(swap, e1, e2)
        f1, f2 = np.where(swap, f2, f1), np.where(swap, f1, f2)
    elif callable(faces):
        selected = np.asarray(faces(centroids), dtype=bool)
    else:
        selected = np.asarray(faces, dtype=bool)
    if selected.shape != (len(first),):
        raise ValueError(f"insert_interfaces: one value per interior face ({len(first)})")
    if sides is not None:
        sides = np.asarray(sides)
        selected = selected & (sides[e1] != sides[e2])
    face_key = {tuple(k): i for i, k in enumerate(keys[first])}
    split_faces = set(np.flatnonzero(selected).tolist())
    on = np.unique(keys[first[selected]])
    # the elements around each node
    counts = np.bincount(tets.ravel(), minlength=mesh.n_nodes)
    start = np.concatenate([[0], np.cumsum(counts)])
    owners = np.argsort(tets.ravel(), kind="stable") // 4
    nodes = [mesh.nodes]
    next_id = mesh.n_nodes
    new_tets = tets.copy()
    for v in on.tolist():
        star = owners[start[v] : start[v + 1]]
        parent = {int(e): int(e) for e in star}

        def find(a, parent=parent):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for e in star.tolist():
            for lf in range(4):
                face = tets[e, local[lf]]
                if v not in face:
                    continue
                index = face_key.get(tuple(np.sort(face)))
                if index is None or index in split_faces:
                    continue  # a boundary face, or an opened one
                other = int(e2[index]) if int(e1[index]) == e else int(e1[index])
                parent[find(e)] = find(other)
        groups: dict[int, list[int]] = {}
        for e in star.tolist():
            groups.setdefault(find(e), []).append(e)
        for members in list(groups.values())[1:]:
            nodes.append(mesh.nodes[v][None])
            for e in members:
                new_tets[e][tets[e] == v] = next_id
            next_id += 1
    sel = np.flatnonzero(selected)
    minus = np.empty((len(sel), 3), dtype=np.int64)
    plus = np.empty((len(sel), 3), dtype=np.int64)
    all_nodes = np.concatenate(nodes)
    for k, index in enumerate(sel.tolist()):
        a, b, la = int(e1[index]), int(e2[index]), int(f1[index])
        original = tets[a, local[la]]
        for i, o in enumerate(original):
            minus[k, i] = new_tets[a][tets[a] == o][0]
            plus[k, i] = new_tets[b][tets[b] == o][0]
        x = mesh.nodes[original]
        normal = np.cross(x[1] - x[0], x[2] - x[0])
        towards = mesh.nodes[tets[b]].mean(axis=0) - mesh.nodes[tets[a]].mean(axis=0)
        if normal @ towards < 0:  # the normal from side - (element a) to side +
            minus[k, [1, 2]] = minus[k, [2, 1]]
            plus[k, [1, 2]] = plus[k, [2, 1]]
    phases = np.stack([mesh.phase[e1[sel]], mesh.phase[e2[sel]]], axis=1)
    opened = TetMesh(
        all_nodes,
        new_tets,
        mesh.phase,
        mesh.lower,
        mesh.upper,
        periodic=False,
        metadata=dict(mesh.metadata),
        cell=mesh.cell,
        orientations=mesh.orientations,
        side=mesh.side,
    )
    return opened, Interfaces(minus, plus, phases)


def _with_mid_nodes(mesh: TetMesh, opened: TetMesh, interfaces: Interfaces):
    """The 10-node mesh of the opened corners ``opened`` of the 10-node ``mesh`` (the same
    elements), its mid-edge nodes where those of ``mesh`` are, and the interface elements
    with the mid-edge nodes of their two faces."""
    quad = opened.quadratic()
    nodes = quad.nodes.copy()
    nodes[quad.elements[:, 4:]] = mesh.nodes[mesh.elements[:, 4:]]
    quad = quad._derived(nodes, quad.elements)
    # the mid-edge node of each pair of corners
    ends = np.sort(quad.elements[:, TET_EDGES].reshape(-1, 2), axis=1)
    keys, first = np.unique(ends[:, 0] * quad.n_nodes + ends[:, 1], return_index=True)
    mids = quad.elements[:, 4:].ravel()[first]

    def six(corners: np.ndarray) -> np.ndarray:
        out = [corners]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = corners[:, i], corners[:, j]
            key = np.minimum(a, b) * quad.n_nodes + np.maximum(a, b)
            out.append(mids[np.searchsorted(keys, key)][:, None])
        return np.concatenate(out, axis=1)

    return quad, Interfaces(six(interfaces.minus), six(interfaces.plus), interfaces.phases)


# -- the problem -------------------------------------------------------------------------------


def plus(a, b):
    """The sum of two sparse matrices with the stored entries of both, zeros included: the
    sum of SciPy drops the zeros, and with them the full blocks of the three dofs of each
    pair of nodes, on which the sparse factorization is several times faster."""
    a, b = a.tocoo(), b.tocoo()
    return sp.csr_matrix(
        (
            np.concatenate([a.data, b.data]),
            (np.concatenate([a.row, b.row]), np.concatenate([a.col, b.col])),
        ),
        shape=a.shape,
    )


@dataclass
class CohesiveResult:
    """The steps of a cohesive problem: load factors, reactions (steps x prescribed sets x
    3, summed over the nodes of each set), cracked area and largest damage, Newton
    iterations, the history of the interface points at the end (elements x 3, or x 6
    for elements of six nodes a side) and their damage, the final displacement (nodes x
    3), with ``fields`` the displacements of the steps, and the works at each step
    (``work``: ``"external"``, of the prescribed displacements and the forces; ``"bulk"``
    and ``"interfaces"``, of their internal forces: the energy stored and dissipated by
    each, which add up to the external work; with a law that rubs, ``"friction"``: the
    part of the work of the interfaces that friction dissipated). ``slip``: the
    frictional slip of the points at the end (elements x points x 2, in the frame of
    each point: the tangents of its mid-surface), and ``friction`` the energy that
    friction dissipated at each per unit area."""

    loads: np.ndarray
    reactions: np.ndarray
    cracked: np.ndarray
    damage_max: np.ndarray
    iterations: np.ndarray
    history: np.ndarray
    damage: np.ndarray
    displacement: np.ndarray
    fields: list = field(default_factory=list, repr=False)
    time_s: float = 0.0
    work: dict = field(default_factory=dict, repr=False)
    slip: np.ndarray | None = field(default=None, repr=False)
    friction: np.ndarray | None = field(default=None, repr=False)

    def rows(self) -> list[dict]:
        """One row per step: load factor, the reactions of each prescribed set, the cracked
        area and the largest damage."""
        out = []
        for k, lam in enumerate(self.loads):
            row = {"load": float(lam)}
            for s in range(self.reactions.shape[1]):
                for c, name in enumerate("xyz"):
                    row[f"R{s + 1}{name}"] = float(self.reactions[k, s, c])
            row["cracked_area"] = float(self.cracked[k])
            row["damage_max"] = float(self.damage_max[k])
            for name, values in self.work.items():
                row[f"work_{name}"] = float(values[k])
            out.append(row)
        return out


class CohesiveProblem:
    """A body with interface elements (see the module): ``mesh`` (tetrahedra of 4 or 10
    nodes, after :func:`insert_interfaces`), the cohesive ``interfaces`` (of three or six
    nodes a side, as the faces of the mesh), the ``materials`` of the phases, the cohesive
    ``law`` and the ``kinematics`` (``"small"`` or ``"finite"`` strains).

    ``bulk``: ``"elastic"``, isotropic elasticity (Young's modulus and Poisson's ratio,
    which also give the neo-Hookean solid of finite strains); ``"points"``, the behaviours
    of morphoGen's points (:class:`~morphogen.fracture.bulk.PointBulk`: von Mises
    elasto-(visco)plasticity, crystal plasticity with the orientations of the grains,
    MFront laws, and Hencky's elasticity in finite strains), with their internal variables
    committed at each converged step, on 4-node elements; ``"auto"``, the points when a
    phase is inelastic, or anisotropic in finite strains."""

    def __init__(
        self,
        mesh: TetMesh,
        interfaces: Interfaces,
        materials,
        law,
        *,
        kinematics="small",
        bulk: str = "auto",
    ):
        if kinematics not in ("small", "finite"):
            raise ValueError("kinematics: 'small' or 'finite'")
        if bulk not in ("auto", "elastic", "points"):
            raise ValueError("bulk: 'auto', 'elastic' or 'points'")
        if interfaces.n_elements and interfaces.order != mesh.order:
            raise ValueError(
                "CohesiveProblem: interface elements of three nodes a side for 4-node "
                "tetrahedra, of six for 10-node ones (insert_interfaces on the mesh itself)"
            )
        self.mesh, self.interfaces, self.law = mesh, interfaces, law
        self.finite = kinematics == "finite"
        self.materials = materials
        self.par = law.params()
        n = mesh.n_nodes
        self.n_dofs = 3 * n
        side = np.hstack([interfaces.minus, interfaces.plus])
        self._idofs = (3 * side[:, :, None] + np.arange(3)).reshape(len(side), 3 * side.shape[1])
        self._tdofs = (3 * mesh.elements[:, :, None] + np.arange(3)).reshape(
            mesh.n_elements, 3 * mesh.elements.shape[1]
        )
        self._points = 3 if side.shape[1] == 6 else 6  # of an interface element
        from morphogen.fracture.bulk import PointBulk, nonlinear_materials

        inelastic = nonlinear_materials(materials, mesh.phase)
        if bulk == "elastic" and inelastic:
            raise ValueError("bulk 'elastic': the materials are inelastic")
        self.bulk = None
        self._dt = 1.0
        self._parts = (np.zeros(self.n_dofs), np.zeros(self.n_dofs))
        present = [materials[int(p)] for p in np.unique(mesh.phase) if int(p) in materials]
        anisotropic = self.finite and any(m.young is None for m in present if not m.void)
        if bulk == "points" or (bulk == "auto" and (inelastic or anisotropic)):
            if mesh.order != 1:
                raise ValueError(
                    "CohesiveProblem: the behaviours of the points (inelastic materials, "
                    "anisotropic ones in finite strains) take 4-node tetrahedra (mesh.linear())"
                )
            self.bulk = PointBulk(mesh, materials, self.finite)
        elif self.finite:
            grads, volumes = _gradients_at_points(mesh.nodes, mesh.elements)
            self._grads, self._volumes = grads, volumes
            mu = np.zeros(mesh.n_elements)
            lam = np.zeros(mesh.n_elements)
            for p, m in materials.items():
                if m.void or m.young is None:
                    continue
                sel = mesh.phase == p
                mu[sel] = m.young / (2 * (1 + m.poisson))
                lam[sel] = m.young * m.poisson / ((1 + m.poisson) * (1 - 2 * m.poisson))
            if np.any((mu == 0) & np.isin(mesh.phase, list(materials))):
                raise ValueError("finite strains: isotropic materials (young, poisson)")
            self._mu, self._lam = mu, lam
        else:
            from morphogen.homogenization.fe import assemble
            from morphogen.homogenization.materials import material_table

            table = material_table(
                mesh.phase, materials, "elastic", grains=mesh.cell, orientations=mesh.orientations
            )  # crystals: their grains
            self._kbulk = assemble(mesh, 3, table.mat, table.matrices).tocsr()

    # -- forces --------------------------------------------------------------------------

    def _forces(self, u, hist, tangent: bool, rate: float = 0.0, predict: bool = False):
        """The internal forces of the displacement ``u``, their tangent and the new state
        of the interface points. ``predict``: the tangent of the interfaces is that of
        the predictor of a step, lips that touched at its start closed and sticking
        (:func:`_law`), the forces those of the law."""
        n = self.n_dofs
        u3 = np.ascontiguousarray(u.reshape(-1, 3))
        ni = self.interfaces.n_elements
        nd = self._idofs.shape[1]  # 18, or 36 for six nodes a side
        fe = np.zeros((ni, nd))
        ke = np.zeros((ni, nd, nd)) if tangent else np.zeros((1, nd, nd))
        hist_new = np.zeros_like(hist)
        if ni:
            par = self.par.copy()
            par[6] = rate
            kernel, gauss = (_interfaces, _GAUSS) if nd == 18 else (_interfaces6, _GAUSS6)
            shared = (self.mesh.nodes, self.interfaces.minus, self.interfaces.plus, u3,
                      self.finite, self.law.kind)  # fmt: skip
            predict = predict and tangent
            kernel(*shared, par, hist, hist_new, gauss, tangent and not predict, fe, ke)
            if predict:  # the tangent of the lips that stay closed and stick
                par = par.copy()
                par[10] = 1.0
                kernel(*shared, par, hist, np.zeros_like(hist), gauss, True, np.zeros_like(fe), ke)
        f = np.bincount(self._idofs.ravel(), fe.ravel(), minlength=n) if ni else np.zeros(n)
        k = None
        if tangent and ni:
            rows = np.repeat(self._idofs, nd, axis=1).ravel()
            cols = np.tile(self._idofs, (1, nd)).ravel()
            k = sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(n, n))
        f_interfaces = f.copy()
        if self.bulk is not None:
            out = self.bulk.evaluate(u, self._dt, tangent)
            if out is None:  # a point failed: no residual, the step is divided
                return np.full(n, np.nan), sp.identity(n, format="csr"), hist_new
            fb, kb = out
            f += fb
            if tangent:
                k = kb if k is None else plus(k, kb)
        elif self.finite:
            nt, nb = self._tdofs.shape  # 12 dofs an element, or 30
            fb = np.zeros((nt, nb))
            kb = np.zeros((nt, nb, nb)) if tangent else np.zeros((1, nb, nb))
            _neo_hookean(
                self._grads,
                self._volumes,
                self.mesh.elements,
                u3,
                self._mu,
                self._lam,
                fb,
                kb,
                tangent,
            )
            f += np.bincount(self._tdofs.ravel(), fb.ravel(), minlength=n)
            if tangent:
                rows = np.repeat(self._tdofs, nb, axis=1).ravel()
                cols = np.tile(self._tdofs, (1, nb)).ravel()
                kbulk = sp.csr_matrix((kb.ravel(), (rows, cols)), shape=(n, n))
                k = kbulk if k is None else plus(k, kbulk)
        else:
            f += self._kbulk @ u
            if tangent:
                k = self._kbulk if k is None else plus(self._kbulk, k)
        self._parts = (f_interfaces, f - f_interfaces)
        return f, k, hist_new

    def _new_state(self) -> np.ndarray:
        """The state of the points of the interfaces before any loading (elements x
        points x :data:`STATE`)."""
        return np.zeros((self.interfaces.n_elements, self._points, STATE))

    def _damage_of(self, hist) -> np.ndarray:
        """The damage of the points (elements x points) of their states."""
        if not hist.size:
            return hist[..., 0]
        return np.vectorize(lambda h: _damage(self.law.kind, self.par, h))(hist[..., 0])

    def _areas(self) -> np.ndarray:
        if getattr(self, "_area", None) is None:
            self._area = self.interfaces.areas(self.mesh)
        return self._area

    def _cracked(self, hist) -> tuple[float, float]:
        if not self.interfaces.n_elements:
            return 0.0, 0.0
        damage = self._damage_of(hist)
        broken = (damage >= BROKEN) @ self.interfaces.weights  # the shares of the points
        return float(self._areas() @ broken), float(damage.max())

    def _rubbed(self, hist) -> float:
        """The energy dissipated by friction over the interfaces."""
        if not self.interfaces.n_elements:
            return 0.0
        return float(self._areas() @ (hist[..., 3] @ self.interfaces.weights))

    # -- solution --------------------------------------------------------------------------

    # -- the seams of the solvers built on this one (morphogen.contact) -----------------------

    def _predictor(self) -> bool:
        """Whether the first tangent of a step is that of its predictor (lips that touch
        or rub)."""
        return self.law.kind == CONTACT or self.par[8] > 0.0

    def _part_names(self) -> tuple:
        """The names of the parts of the internal forces (``_parts``), whose works are
        results."""
        return ("interfaces", "bulk")

    def _begin(self, u, load: float) -> None:
        """Before the first step, from the displacement ``u`` at the load factor ``load``."""

    def _attempt(self, target: float) -> None:
        """A step is tried towards the load factor ``target``."""

    def _augment(self) -> bool:
        """After Newton's method converged: whether the step is to be solved again."""
        return False

    def _accept(self, u) -> None:
        """A step converged at the displacement ``u``."""

    def _newton(self, u_try, hist, dofs, free, load_now, rate, tol, max_newton):
        """Newton's method with a line search on the residual under the dead loads
        ``load_now`` of the free dofs: whether it converged, its iterations, the
        displacement, the forces, the new history of the interfaces and these loads."""
        ok, iterations, scale = False, 0, None
        f = hist_try = None
        predictor = self._predictor()
        for iterations in range(1, max_newton + 1):
            f, k, hist_try = self._forces(u_try, hist, True, rate, predictor and iterations == 1)
            r = f[free] - load_now
            norm = float(np.linalg.norm(r))
            if scale is None:
                scale = max(
                    float(np.linalg.norm(f[dofs])),
                    float(np.linalg.norm(f)),
                    float(np.linalg.norm(load_now)),
                    1e-30,
                )
            if norm <= tol * scale and iterations > 1:
                ok = True
                break
            kff = k[free][:, free].tocsc()
            # the ordering of the factorization: that of matrices of symmetric pattern is
            # three to five times faster on 10-node elements, slower on 4-node ones
            ordering = "MMD_AT_PLUS_A" if self.mesh.order == 2 else "COLAMD"
            try:
                du = spla.spsolve(kff, -r, permc_spec=ordering)
            except RuntimeError:
                break
            if not np.all(np.isfinite(du)):
                break
            alpha = 1.0
            for _ in range(4):  # a backtracking line search on the residual
                trial = u_try.copy()
                trial[free] += alpha * du
                f_t, _, _ = self._forces(trial, hist, False, rate)
                if np.linalg.norm(f_t[free] - load_now) < norm or alpha < 0.2:
                    break
                alpha *= 0.5
            u_try[free] += alpha * du
            scale = max(scale, float(np.linalg.norm(f_t[dofs])))
        return ok, iterations, u_try, f, hist_try, load_now

    def run(
        self,
        prescribed,
        load: float = 1.0,
        *,
        steps: int = 20,
        tol: float = 1e-6,
        max_newton: int = 25,
        max_cuts: int = 12,
        fields: bool = False,
        viscosity: float = 0.0,
        forces=None,
        duration: float | None = None,
        start: float = 0.0,
        initial=None,
        equilibrate: bool = True,
        history=None,
    ) -> CohesiveResult:
        """Grow the prescribed displacements to ``load`` times their values in ``steps``
        steps (divided when Newton's method fails, ``max_cuts`` times in a row at most).
        ``prescribed``: entries ``(nodes, components, value)``, the displacement of the
        components (an index or a list) of the nodes being ``load factor x value`` (0:
        fixed; an array of the nodes x components: one value each), or ``value(load
        factor)`` when ``value`` is a function: loadings in several stages, such as a
        pressure held while a shear grows. ``forces``: nodal forces (3 x nodes, dead loads)
        grown with them, or a function of the load factor returning them. ``viscosity``:
        the ``ζ`` of the viscous regularization (in units of the load factor; 0: none). The
        load factor is the time of the viscous behaviours of the bulk, or the loading lasts
        ``duration``.

        ``initial``: a displacement (nodes x 3) at the load factor ``start``, from which the
        loading resumes, the internal variables of the bulk as they are (those of an earlier
        run, or a state transferred from another mesh: :func:`morphogen.fracture.ductile.
        transfer_state`); ``equilibrate``: first bring it to equilibrium at that load, over
        a vanishing time (no viscous flow), the plastic correction of the points included
        but dissipating nothing (the bulk keeps the dissipated part of its work density, its
        free energy that of the equilibrium; ``result.equilibration``: the Newton
        iterations).

        ``history``: that of the interface points at the start (one value per element, or
        elements x points): a damage of 1 for the part of a bilinear interface that is
        already broken, a negative value for the bonded points of a :class:`Contact`."""
        t0 = time.perf_counter()
        n = self.n_dofs
        dofs, values, sets = [], [], []
        for nodes, comps, value in prescribed:
            nodes = np.atleast_1d(np.asarray(nodes, dtype=np.int64))
            comps = np.atleast_1d(np.asarray(comps, dtype=np.int64))
            d = (3 * nodes[:, None] + comps[None, :]).ravel()
            dofs.append(d)
            shape = (len(nodes), len(comps))
            if callable(value):  # a function of the load factor

                def course(t, value=value, shape=shape):
                    return np.broadcast_to(np.asarray(value(t), dtype=float), shape).ravel()

            else:
                amplitude = np.broadcast_to(np.asarray(value, dtype=float), shape).ravel()

                def course(t, amplitude=amplitude):
                    return t * amplitude

            values.append(course)
            sets.append(d)
        dofs = np.concatenate(dofs)
        if len(np.unique(dofs)) != len(dofs):
            raise ValueError("prescribed: a degree of freedom is given twice")

        def values_at(t: float) -> np.ndarray:
            """The prescribed displacements at the load factor ``t``."""
            return np.concatenate([course(t) for course in values])

        if callable(forces):

            def external_at(t: float) -> np.ndarray:
                out = np.asarray(forces(t), dtype=float).ravel()
                if out.shape != (n,):
                    raise ValueError(f"forces: one per degree of freedom ({n})")
                return out

        else:
            external = np.zeros(n) if forces is None else np.asarray(forces, dtype=float).ravel()
            if external.shape != (n,):
                raise ValueError(f"forces: one per degree of freedom ({n})")

            def external_at(t: float) -> np.ndarray:
                return t * external

        free = np.setdiff1d(np.arange(n), dofs)
        hist = self._new_state()
        if history is not None:
            history = np.asarray(history, dtype=float)
            if history.shape[:1] != hist.shape[:1] or history.shape not in (
                hist.shape[:1],
                hist.shape[:2],
            ):
                raise ValueError(
                    "history: one value per interface element, or one per point "
                    f"({hist.shape[0]} x {hist.shape[1]})"
                )
            hist[..., 0] = history if history.ndim == 2 else history[:, None]
        rubs = self.par[8] > 0.0  # a law with friction
        if initial is None:
            u, lam = np.zeros(n), 0.0
            if self.bulk is not None:
                self.bulk.reset()
        else:
            u = np.array(initial, dtype=float).ravel()
            if u.shape != (n,):
                raise ValueError(f"initial: a displacement of the {n // 3} nodes")
            lam = float(start)
            u[dofs] = values_at(lam)
        clock = 1.0 if duration is None else float(duration) / float(load)
        names = self._part_names()
        parts = tuple(np.zeros(n) for _ in names)  # the forces of the parts at the last step
        applied_last = np.zeros(n)
        equilibration = 0
        self._begin(u, lam)
        if initial is not None and equilibrate:
            self._attempt(lam)
            self._dt = 1e-9 * clock * max(abs(load - lam), abs(load), 1e-30) / steps
            ok, equilibration, u, f, hist, load_now = self._newton(
                u, hist, dofs, free, external_at(lam)[free], 0.0, tol, max_newton
            )
            if not ok:
                raise RuntimeError(f"cohesive problem: no equilibrium of the state at load {lam:g}")
            if self.bulk is not None:  # dissipating nothing: the dissipated energy kept
                dissipated = self.bulk.work - self.bulk.free_energy()
                self.bulk.commit()
                psi = self.bulk.free_energy()
                self.bulk.work[...] = np.where(np.isfinite(psi), dissipated + psi, self.bulk.work)
            parts = tuple(part.copy() for part in self._parts)
            applied_last[dofs] = f[dofs]
            applied_last[free] = load_now
        work = {"external": [0.0], "bulk": [0.0], "interfaces": [0.0]}
        work.update({name: [0.0] for name in names if name not in work})
        if rubs:
            work["friction"] = [0.0]
        loads, reactions, cracked, dmax, its, snaps = [lam], [], [0.0], [0.0], [equilibration], []
        reactions.append(
            np.array([[applied_last[s[s % 3 == c]].sum() for c in range(3)] for s in sets])
        )
        if fields:
            snaps.append(u.reshape(-1, 3).copy())
        span = (load - lam) / steps
        step = span
        cuts = 0
        while lam < load * (1 - 1e-12):
            step = min(step, load - lam)
            target = lam + step
            u_try = u.copy()
            u_try[dofs] = values_at(target)
            self._dt = step * clock
            rate = step / viscosity if viscosity > 0 else 0.0
            self._attempt(target)
            ok, iterations, u_try, f, hist_try, load_now = self._newton(
                u_try, hist, dofs, free, external_at(target)[free], rate, tol, max_newton
            )
            while ok and self._augment():  # solved again (the multipliers of a contact)
                ok, more, u_try, f, hist_try, load_now = self._newton(
                    u_try, hist, dofs, free, external_at(target)[free], rate, tol, max_newton
                )
                iterations += more
            if not ok:
                cuts += 1
                if cuts > max_cuts:
                    raise RuntimeError(f"cohesive problem: no convergence at load {target:g}")
                step *= 0.5
                continue
            du_step = u_try - u
            applied = np.zeros(n)  # the reactions and the dead loads
            applied[dofs] = f[dofs]
            applied[free] = load_now
            work["external"].append(work["external"][-1] + 0.5 * (applied_last + applied) @ du_step)
            for name, now, before in zip(names, self._parts, parts, strict=True):
                work[name].append(work[name][-1] + 0.5 * (before + now) @ du_step)
            parts = tuple(part.copy() for part in self._parts)
            applied_last = applied
            if self.bulk is not None:
                self.bulk.commit()
            self._accept(u_try)
            u, hist, lam = u_try, hist_try, target
            if rubs:
                work["friction"].append(self._rubbed(hist))
            cuts = 0  # the divisions of one step only
            loads.append(lam)
            reactions.append(np.array([[f[s[s % 3 == c]].sum() for c in range(3)] for s in sets]))
            area, damage = self._cracked(hist)
            cracked.append(area)
            dmax.append(damage)
            its.append(iterations)
            if fields:
                snaps.append(u.reshape(-1, 3).copy())
            if iterations <= 4:
                step = min(step * 1.5, span)
        result = CohesiveResult(
            np.array(loads),
            np.array(reactions),
            np.array(cracked),
            np.array(dmax),
            np.array(its),
            hist[..., 0].copy(),
            self._damage_of(hist),
            u.reshape(-1, 3),
            snaps,
            time.perf_counter() - t0,
            {name: np.array(values) for name, values in work.items()},
            hist[..., 1:3].copy(),
            hist[..., 3].copy(),
        )
        result.equilibration = equilibration
        return result

    def write(self, path, result: CohesiveResult, step: int = -1):
        """The bulk (``<path>.vtu``: the displacement of ``step`` with ``fields``, else the
        final one) and the interfaces (``<path>_interfaces.vtu``: wedges of six nodes with
        the damage of their points and its mean, and with a law that rubs the mean slip of
        their points and the energy that friction dissipated per unit area); returns both
        paths."""
        from pathlib import Path

        from morphogen.io.meshes import write_vtu_tet
        from morphogen.io.vtk import write_vtu

        path = Path(path)
        u = result.fields[step] if result.fields else result.displacement
        bulk = write_vtu_tet(path.with_suffix(".vtu"), self.mesh, point_data={"displacement": u})
        # wedges of the corners (the mid-edge nodes of faces of six nodes are left out)
        cells = np.hstack([self.interfaces.minus[:, :3], self.interfaces.plus[:, :3]])
        data = {"damage": result.damage @ self.interfaces.weights, "damage_points": result.damage}
        if self.par[8] > 0.0 and result.slip is not None:  # the mean slip of the points
            data["slip"] = np.linalg.norm(result.slip, axis=2) @ self.interfaces.weights
            data["friction"] = result.friction @ self.interfaces.weights
        wedges = write_vtu(
            path.with_name(path.stem + "_interfaces.vtu"), self.mesh.nodes + u, cells, 13, data
        )
        return bulk, wedges


# -- calibration and the double cantilever beam -------------------------------------------------


def cohesive_zone_length(
    young: float, energy: float, strength: float, factor: float = 1.0
) -> float:
    """``l_cz = M E G / T²`` (``factor`` M: 1 after Hillerborg, 9π/32 after Rice for a
    bilinear law in mode I, 0.21 after Falk; Turon et al. 2007)."""
    return factor * young * energy / strength**2


def turon_strength(young: float, energy: float, element_size: float, elements: int = 3) -> float:
    """The cohesive strength that keeps ``elements`` elements of size ``element_size`` in
    the cohesive zone (Turon, Dávila, Camanho & Costa 2007): ``sqrt(9π E G / (32 N_e
    l_e))``."""
    return math.sqrt(9 * math.pi * young * energy / (32 * elements * element_size))


def dcb_beam_theory(
    young: float, poisson: float, width: float, arm: float, precrack: float, energy: float, opening
):
    """The corrected beam theory of the double cantilever beam (Williams 1989; Reeder et
    al.; Linke & Lammering 2023, eqs. 2 to 6): the load and the crack length against the
    opening ``opening`` of the loading line, for arms of thickness ``arm`` and ``width``
    of an isotropic material, a precrack ``precrack`` and the mode I fracture energy:
    ``P = 3 E I d / (2 (a + χ h)³)`` with ``a = a0`` until the crack grows, then ``a = (9 d² E
    I / (4 w G))^(1/4) - χ h``; ``χ = sqrt(E / (11 S) (3 - 2 (Γ / (1 + Γ))²))``, ``Γ = 1.18 E /
    S``."""
    d = np.asarray(opening, dtype=float)
    shear = young / (2 * (1 + poisson))
    gamma = 1.18 * young / shear
    chi = math.sqrt(young / (11 * shear) * (3 - 2 * (gamma / (1 + gamma)) ** 2))
    inertia = width * arm**3 / 12
    grown = (9 * d**2 * young * inertia / (4 * width * energy)) ** 0.25 - chi * arm
    crack = np.maximum(precrack, grown)
    load = 3 * young * inertia * d / (2 * (crack + chi * arm) ** 3)
    return load, crack


def dcb_specimen(
    length: float, width: float, arm: float, precrack: float, size: float, order: int = 1
):
    """A double cantilever beam of two arms of thickness ``arm`` (``length`` x ``width``),
    meshed by six tetrahedra per voxel of edge ``size`` (of 4 nodes, or of 10 with ``order``
    2), opened along its mid-plane: the mesh, the interface elements of the bonded part
    (beyond the precrack) and the node sets of the loading lines (``"top"``, ``"bottom"``:
    the nodes at ``x = 0`` on the mid-plane of each arm)."""
    from morphogen.homogenization.bloch import voxel_tet_mesh
    from morphogen.image import VoxelImage

    shape = (round(length / size), max(1, round(width / size)), 2 * round(arm / size))
    if shape[2] % 4:
        raise ValueError("dcb_specimen: an even number of voxels across each arm")
    image = VoxelImage(np.zeros(shape, np.uint8), spacing=(size,) * 3, periodic=False)
    mesh = voxel_tet_mesh(image)
    if order == 2:
        mesh = mesh.quadratic()
    elif order != 1:
        raise ValueError("dcb_specimen: order 1 or 2")
    plane = shape[2] // 2 * size
    opened, interfaces = insert_interfaces(mesh, lambda c: np.abs(c[:, 2] - plane) < 1e-6 * size)
    bonded = interfaces.subset(interfaces.centroids(opened)[:, 0] > precrack)
    x = opened.nodes
    at_end = np.abs(x[:, 0]) < 1e-9 * size
    top = np.flatnonzero(at_end & (np.abs(x[:, 2] - 1.5 * plane) < 1e-6 * size))
    bottom = np.flatnonzero(at_end & (np.abs(x[:, 2] - 0.5 * plane) < 1e-6 * size))
    return opened, bonded, {"top": top, "bottom": bottom}
