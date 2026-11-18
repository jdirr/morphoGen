"""Viscoelasticity, hyperelasticity and damage: behaviours of the phases given in closed
form, integrated at all the points of a phase at once.

They follow the seam of the laws integrated outside the compiled kernels (the MFront
behaviours and the learned laws, :mod:`morphogen.learning.behaviour`): a material whose
``viscoelasticity``, ``hyperelasticity`` or ``damage`` is set has its points integrated by
one of the laws of this module, at a material point, by finite elements, on the voxels of
the FFT solver and in the bulk of the crack solvers; the rest of the solvers is unchanged
(the periodic element, Newton's method, the outputs). The elastic constants of the
material (``young``, ``poisson``) are the instantaneous ones: those of the linear
problems, of the reference medium of the FFT solver and of the first instant of a path.

**Viscoelasticity** (:class:`Viscoelasticity`): the generalized Maxwell model, a Prony
series of the shear and bulk relaxation moduli,

    G(t) = G0 (1 - sum_i g_i (1 - exp(-t / tau_i))),   K(t) likewise with (k_j, tau_j),

relative to the instantaneous moduli (the convention of Abaqus). The stress is ``s = 2
G_inf e + sum_i q_i`` (deviators), each branch relaxing as ``dq_i/dt + q_i / tau_i = 2 g_i
G0 de/dt``, integrated exactly for a strain that varies linearly over the step (the
recurrence of Taylor, Pister & Goudreau 1970): ``q_i <- exp(-x) q_i + 2 g_i G0 phi(x) de``,
``x = dt / tau_i``, ``phi(x) = (1 - exp(-x)) / x``; the tangent is the elasticity of the
moduli ``G_inf + sum_i g_i G0 phi(x_i)``. Thermo-rheologically simple materials shift
their times by ``a_T`` (Williams-Landel-Ferry, or Arrhenius). In finite strains the same
law holds in the logarithmic strain space, as the other behaviours of morphoGen; with a
``hyperelasticity``, it is the finite linear viscoelasticity of Simo (1987): the branches
relax the isochoric and volumetric second Piola-Kirchhoff stresses of the hyperelastic
law, ``Q_i <- exp(-x) Q_i + g_i phi(x) (S_iso^(n+1) - S_iso^n)``.
:meth:`Viscoelasticity.complex_moduli` gives the storage and loss moduli of a frequency
(the correspondence principle: :mod:`morphogen.homogenization.harmonic`).

**Hyperelasticity** (:class:`Hyperelasticity`; finite strains): a strain energy ``W =
psi(I1b, I2b) + U(J)`` of the isochoric invariants ``I1b = J^(-2/3) I1``, ``I2b =
J^(-4/3) I2`` of ``C = F^T F`` and of ``J = det F`` (neo-Hookean, the polynomial of
Rivlin with its forms of Mooney and Rivlin and of Yeoh, Arruda-Boyce and the eight chains,
Gent, van der Waals, Fung), of the isochoric principal stretches (Ogden), of the stretches
of families of fibres in a neo-Hookean matrix (Holzapfel, Gasser and Ogden), compressible
without a split (the foams of Blatz and Ko and of Ogden, Hill and Storakers), or the law
of Saint-Venant and Kirchhoff. The law gives the second Piola-Kirchhoff stress ``S = 2
dW/dC`` and its tangent ``dS/dE = 4 d2W/dC2`` (symmetric), which the points turn into the
first Piola-Kirchhoff stress and ``dP/dF``. Its isochoric stress may soften as the
rubbers of Mullins (Ogden & Roxburgh 1999). In small strains the material is elastic (its
tangent at the natural state: the fibres of Holzapfel do not count there).
:meth:`_Hyperelastic.nominal_stress` gives the tests of rubbers (uniaxial, equibiaxial
and planar tension, simple shear) and :func:`fit_hyperelasticity` fits a law to them.

**Damage** (:class:`Damage`): the isotropic damage of quasi-brittle solids, ``sigma =
(1 - d) C : eps``, the damage ``d`` a function of the largest equivalent strain ``kappa``
reached (the energy norm of Marigo and Lemaitre, Mazars' positive principal strains, or
the modified von Mises strain of de Vree et al. 1995), with a linear or exponential
softening. The stress may keep the stiffness of closed cracks: the bulk response in
compression (the volumetric split of Amor et al. 2009), or that of the negative principal
strains (the spectral split of Miehe et al. 2010, ``sigma = (1 - d) sigma+ + sigma-`` with
``sigma± = lambda <tr eps>± I + 2 mu eps±``: the unilateral effect, a crack that closes
carries the compression across it). Softening makes the dissipation of a local law
vanish with the size of the elements: the fracture energy ``G_f`` and the size ``length``
of the elements set the ultimate strain so that a band of one element dissipates ``G_f``
per unit area (the crack band of Bazant & Oh 1983), and a ``viscosity`` delays the damage
(``d_v <- (d_v + d dt / eta) / (1 + dt / eta)``), which keeps the incremental problems
well posed. The nonlocal alternative, whose bands have a width of their own, is the
micromorphic damage of :mod:`morphogen.micromorphic`. In finite strains the law holds in
the logarithmic strain space.

**Concrete** (:class:`Mazars`): the law of Mazars (1984, 1986) with a damage of tension and
one of compression, ``d = alpha_t^beta d_t(kappa) + alpha_c^beta d_c(kappa)``, each ``d_i =
1 - kappa_0 (1 - A_i) / kappa - A_i exp(-B_i (kappa - kappa_0))`` of the largest equivalent
strain ``kappa`` (the norm of the positive principal strains), weighted by the shares
``alpha_t`` and ``alpha_c = 1 - alpha_t`` of the positive strains that the tensions and
the compressions of the effective stress make: 1 and 0 in a uniaxial tension, 0 and 1 in
a uniaxial compression (which only extends the point sideways).

**Damage coupled to plasticity** (:class:`DuctileDamage`, with the ``plasticity`` of the
material): the elasto-(visco)plasticity of von Mises of
:mod:`morphogen.homogenization.behaviours` (isotropic and kinematic hardenings, Norton's
flow) is that of the effective stress ``sigma~ = sigma / (1 - D)`` (the strain
equivalence of Lemaitre & Chaboche), and the damage ``D`` follows

* ``lemaitre``: ``dD = (Y / S)^s dp`` beyond the cumulated plastic strain ``threshold``
  (Lemaitre 1985), with ``Y`` the energy release rate of the effective stress, ``Y =
  sigma~_eq^2 R_nu / (2 E)``, ``R_nu = 2/3 (1 + nu) + 3 (1 - 2 nu) (sigma~_m /
  sigma~_eq)^2`` the triaxiality function; with the ``closure`` ``h`` of the microcracks
  (Ladeveze & Lemaitre 1984), the negative principal stresses count for ``h`` of
  themselves in ``Y`` (0.2 for metals: compression damages little). Ductile fracture,
  and the fatigue of a few cycles: the damage of a cycle is that of its plastic strain;
* ``kachanov``: ``dD/dt = (chi(sigma) / A)^r (1 - D)^(-k)`` (Kachanov 1958; Rabotnov
  1969), the damage of creep, with Hayhurst's (1972) stress ``chi = alpha sigma_I + 3
  beta sigma_m + (1 - alpha - beta) sigma_eq`` of the largest principal stress, the mean
  stress and von Mises' stress: under a constant stress the time to rupture is ``1 /
  ((k + 1) (chi / A)^r)`` and ``D = 1 - (1 - t / t_R)^(1 / (k + 1))``, the creep of the
  effective stress speeding up as the section is lost (tertiary creep).

A point whose damage reaches ``critical`` is broken: it keeps the ``residual`` fraction
of its stress. The laws are local: once the stress falls their bands are those of the
elements (the micromorphic ductile damage of :mod:`morphogen.micromorphic` has a width
of its own), and the incremental problems of a cell lose their solution; a ``viscosity``
delays the damage that degrades the stress behind that of the law, ``dD/dt = (D_law -
D) / eta``, which keeps them well posed.

Stresses use the Voigt order 11 22 33 23 31 12, strains the engineering shears; the
variable ``p`` of the outputs of a damaged phase is its damage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from morphogen.homogenization.materials import isotropic_matrix

#: Hyperelastic models and their parameters (those in brackets have a default: the
#: shear and bulk moduli of the material)
HYPERELASTIC_MODELS = {
    "neo_hookean": "psi = mu/2 (I1b - 3)                                  [mu]",
    "mooney_rivlin": "psi = sum cij (I1b - 3)^i (I2b - 3)^j, i + j <= 3     c10, c01, [cij]",
    "polynomial": "psi = sum cij (I1b - 3)^i (I2b - 3)^j, i + j <= 6        cij",
    "yeoh": "psi = c10 (I1b - 3) + c20 (I1b - 3)^2 + c30 (I1b - 3)^3      c10, [c20, c30]",
    "arruda_boyce": "psi = mu sum_i a_i / lock^(2i-2) (I1b^i - 3^i)        [mu], lock",
    "eight_chain": "psi' = mu/6 (3 - x) / (1 - x), x = I1b / (3 lock^2)    [mu], lock",
    "gent": "psi = -mu Jm / 2 ln(1 - (I1b - 3) / Jm)                      [mu], lock (Jm)",
    "van_der_waals": "psi = mu (-(lock^2 - 3) (ln(1 - e) + e) - 2a/3 ((I - 3)/2)^1.5)"
    "   [mu], lock, [a, beta]",
    "fung": "psi = mu / (2 b) (exp(b (I1b - 3)) - 1)                      [mu], b",
    "ogden": "psi = sum_p mu_p / alpha_p (l1^a + l2^a + l3^a - 3)         terms",
    "holzapfel": "psi = mu/2 (I1b - 3) + k1/(2 k2) sum_f (exp(k2 E_f^2) - 1)"
    "   [mu], k1, k2, [kappa], fibres",
    "blatz_ko": "W = mu f/2 (I1 - 3 + (J^(-2b) - 1)/b) + mu (1-f)/2 (I2/J^2 - 3 + (J^(2b) - 1)/b)"
    "   [mu, f]",
    "hyperfoam": "W = sum_p 2 mu_p / alpha_p^2 (l1^a + l2^a + l3^a - 3 + (J^(-a b) - 1)/b)   terms",
    "saint_venant_kirchhoff": "S = lambda tr(E) I + 2 mu E                (none)",
}
#: The parameters that each model takes and those it must be given (the polynomials take
#: besides any coefficient ``cij`` up to their order, :data:`_ORDERS`)
_PARAMETERS = {
    "neo_hookean": ({"mu"}, set()),
    "mooney_rivlin": (set(), {"c10", "c01"}),
    "polynomial": (set(), set()),
    "yeoh": ({"c10", "c20", "c30"}, {"c10"}),
    "arruda_boyce": ({"mu", "lock"}, {"lock"}),
    "eight_chain": ({"mu", "lock"}, {"lock"}),
    "gent": ({"mu", "lock"}, {"lock"}),
    "van_der_waals": ({"mu", "lock", "a", "beta"}, {"lock"}),
    "fung": ({"mu", "b"}, {"b"}),
    "ogden": ({"terms"}, {"terms"}),
    "holzapfel": ({"mu", "k1", "k2", "kappa", "fibres"}, {"k1", "k2", "fibres"}),
    "blatz_ko": ({"mu", "f"}, set()),
    "hyperfoam": ({"terms"}, {"terms"}),
    "saint_venant_kirchhoff": (set(), set()),
}
#: The largest order ``i + j`` of the coefficients ``cij`` of the polynomials of Rivlin
_ORDERS = {"mooney_rivlin": 3, "polynomial": 6}
#: The models whose isochoric energy is a polynomial of the invariants
_POLYNOMIALS = ("mooney_rivlin", "polynomial", "yeoh")
#: The compressible laws: no split of the energy, their Poisson's ratio from the bulk
#: modulus
_COMPRESSIBLE = ("blatz_ko", "hyperfoam")
#: The homogeneous tests of rubbers (:meth:`_Hyperelastic.nominal_stress`)
TESTS = ("uniaxial", "equibiaxial", "planar", "shear")
#: The models that :func:`fit_hyperelasticity` fits to tests of incompressible rubbers
FITTED = ("neo_hookean", "mooney_rivlin", "polynomial", "yeoh", "arruda_boyce", "eight_chain",
          "gent", "van_der_waals", "fung", "ogden")  # fmt: skip
#: Volumetric energies ``U(J)`` of the hyperelastic laws (``K``: the bulk modulus)
VOLUMETRIC = ("quadratic", "logarithmic", "simo_miehe")
#: Equivalent strains of the damage law
EQUIVALENT_STRAINS = ("energy", "mazars", "mises")
#: Softening laws of the damage
SOFTENINGS = ("exponential", "linear")
#: Splits of the stress of a damaged point: all of it degraded, the bulk response in
#: compression kept (Amor, Marigo & Maurini 2009), or that of the negative principal
#: strains (Miehe, Hofacker & Welschinger 2010)
SPLITS = ("none", "volumetric", "spectral")
#: The laws of the damage coupled to plasticity (:class:`DuctileDamage`)
DUCTILE_LAWS = ("lemaitre", "kachanov")

_M = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
#: Tensor components of the engineering strains
_HALF = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))
_I0 = np.array([i for i, _ in _PAIRS])
_I1 = np.array([j for _, j in _PAIRS])
#: The Voigt matrices of I (x) I - I_sym and of the deviatoric projector (stresses of
#: engineering strains)
_A2 = np.outer(_M, _M) - np.diag(_HALF)
#: Points integrated at once (the memory of the arrays of a step)
_CHUNK = 1 << 15
#: Coefficients of the series of Arruda and Boyce
_ARRUDA = (0.5, 1.0 / 20.0, 11.0 / 1050.0, 19.0 / 7000.0, 519.0 / 673750.0)


def _phi(x):
    """``(1 - exp(-x)) / x`` (1 at 0)."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-6
    safe = np.where(small, 1.0, x)
    return np.where(small, 1.0 - 0.5 * x + x * x / 6.0, -np.expm1(-safe) / safe)


def _pairs(value, name: str, first: str, second: str) -> tuple[tuple[float, float], ...]:
    """Terms given as pairs, mappings ``{first: .., second: ..}`` or a list of them."""
    if value is None:
        return ()
    items = [value] if isinstance(value, dict) else list(value)
    if items and not isinstance(items[0], (dict, list, tuple)):
        items = [items]
    out = []
    for item in items:
        if isinstance(item, dict):
            if set(item) != {first, second}:
                raise ValueError(f"{name}: each term gives {first} and {second}")
            item = (item[first], item[second])
        if len(item) != 2:
            raise ValueError(f"{name}: each term is a pair ({first}, {second})")
        a, b = float(item[0]), float(item[1])
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError(f"{name}: finite terms")
        out.append((a, b))
    return tuple(out)


# -- viscoelasticity -----------------------------------------------------------------------


@dataclass(frozen=True)
class Viscoelasticity:
    """A generalized Maxwell model (see the module): ``shear`` and ``bulk``, the terms
    ``(g_i, tau_i)`` and ``(k_j, tau_j)`` of the Prony series of the shear and bulk
    relaxation moduli, relative to the instantaneous moduli of the material (each sum
    below 1; no ``bulk`` term: an elastic bulk response). ``wlf``: the constants ``(C1,
    C2)`` of Williams, Landel and Ferry, ``log10 a_T = -C1 dT / (C2 + dT)`` for the change
    of temperature ``dT`` of the loading paths from their reference state; or
    ``arrhenius``: the activation energy over the gas constant (K), ``ln a_T = arrhenius
    (1 / T - 1 / T_ref)`` with ``reference_temperature`` (K) that of the reference state."""

    shear: tuple = ()
    bulk: tuple = ()
    wlf: tuple | None = None
    arrhenius: float | None = None
    reference_temperature: float = 293.15

    def __post_init__(self) -> None:
        for name, weight in (("shear", "g"), ("bulk", "k")):
            terms = _pairs(getattr(self, name), f"viscoelasticity.{name}", weight, "tau")
            if any(not (w > 0 and tau > 0) for w, tau in terms):
                raise ValueError(f"viscoelasticity.{name}: positive weights and times")
            if sum(w for w, _ in terms) >= 1.0:
                raise ValueError(
                    f"viscoelasticity.{name}: the weights of the terms sum to less than 1 "
                    "(a solid: its long-term modulus is positive)"
                )
            object.__setattr__(self, name, terms)
        if not self.shear and not self.bulk:
            raise ValueError("viscoelasticity: at least one term (shear or bulk)")
        if self.wlf is not None:
            wlf = tuple(float(v) for v in self.wlf)
            if len(wlf) != 2 or not wlf[1] > 0:
                raise ValueError("viscoelasticity.wlf: the constants (C1, C2), C2 positive")
            object.__setattr__(self, "wlf", wlf)
            if self.arrhenius is not None:
                raise ValueError("viscoelasticity: wlf or arrhenius, not both")
        if self.arrhenius is not None:
            object.__setattr__(self, "arrhenius", float(self.arrhenius))
        t = float(self.reference_temperature)
        if not t > 0:
            raise ValueError("viscoelasticity.reference_temperature: an absolute temperature")
        object.__setattr__(self, "reference_temperature", t)

    @classmethod
    def from_config(cls, spec) -> Viscoelasticity:
        """A mapping ``shear: [{g: .., tau: ..}, ...]`` (or pairs), ``bulk: [{k: ..,
        tau: ..}]``, ``wlf: [C1, C2]``, ``arrhenius``, ``reference_temperature``."""
        if isinstance(spec, Viscoelasticity):
            return spec
        if not isinstance(spec, dict):
            raise ValueError("viscoelasticity: a mapping (shear, bulk, wlf, arrhenius)")
        names = {"shear", "bulk", "wlf", "arrhenius", "reference_temperature"}
        unknown = sorted(set(spec) - names)
        if unknown:
            raise ValueError(f"viscoelasticity: unknown keys {unknown}; allowed: {sorted(names)}")
        return cls(**spec)

    def to_config(self) -> dict:
        out: dict = {}
        if self.shear:
            out["shear"] = [{"g": g, "tau": tau} for g, tau in self.shear]
        if self.bulk:
            out["bulk"] = [{"k": k, "tau": tau} for k, tau in self.bulk]
        if self.wlf is not None:
            out["wlf"] = list(self.wlf)
        if self.arrhenius is not None:
            out["arrhenius"] = self.arrhenius
            out["reference_temperature"] = self.reference_temperature
        return out

    def shift(self, dtemp: float = 0.0) -> float:
        """The shift factor ``a_T`` of the relaxation times at the change of temperature
        ``dtemp`` from the reference state (1 without a shift law)."""
        dtemp = float(dtemp)
        if self.wlf is not None:
            c1, c2 = self.wlf
            if c2 + dtemp <= 0:
                raise ValueError("viscoelasticity.wlf: C2 + dT must stay positive")
            return float(10.0 ** (-c1 * dtemp / (c2 + dtemp)))
        if self.arrhenius is not None:
            t0 = self.reference_temperature
            return float(math.exp(self.arrhenius * (1.0 / (t0 + dtemp) - 1.0 / t0)))
        return 1.0

    def long_term(self, bulk: float, shear: float) -> tuple[float, float]:
        """The relaxed bulk and shear moduli of instantaneous moduli ``bulk``, ``shear``."""
        return (
            bulk * (1.0 - sum(k for k, _ in self.bulk)),
            shear * (1.0 - sum(g for g, _ in self.shear)),
        )

    def relaxation(self, times, bulk: float, shear: float, dtemp: float = 0.0):
        """The bulk and shear relaxation moduli ``K(t)``, ``G(t)`` at ``times``."""
        t = np.asarray(times, dtype=float) / self.shift(dtemp)
        k = bulk * (1.0 - sum(w * -np.expm1(-t / tau) for w, tau in self.bulk))
        g = shear * (1.0 - sum(w * -np.expm1(-t / tau) for w, tau in self.shear))
        return k + 0.0 * t, g + 0.0 * t

    def complex_moduli(self, omega, bulk: float, shear: float, dtemp: float = 0.0):
        """The complex bulk and shear moduli ``K*``, ``G*`` at the angular frequencies
        ``omega`` (rad per unit time): ``G* = G0 (1 - sum_i g_i / (1 + i omega tau_i))``,
        their real parts the storage moduli, their imaginary parts the loss moduli."""
        w = np.asarray(omega, dtype=float) * self.shift(dtemp)
        k = bulk * (1.0 - sum(c / (1.0 + 1j * w * tau) for c, tau in self.bulk))
        g = shear * (1.0 - sum(c / (1.0 + 1j * w * tau) for c, tau in self.shear))
        return k + 0j * w, g + 0j * w


class _Maxwell:
    """The integration of a :class:`Viscoelasticity` in small strains (the law of the
    module); without branches, isotropic elasticity."""

    kind = "viscoelastic"
    symmetric = True
    thermal = True

    def __init__(self, visco: Viscoelasticity | None, bulk: float, shear: float):
        self.visco = visco
        self.bulk, self.shear = float(bulk), float(shear)
        self.g_terms = visco.shear if visco is not None else ()
        self.k_terms = visco.bulk if visco is not None else ()
        self.n_state = 6 * len(self.g_terms) + len(self.k_terms)

    def step(self, e_new, e_old, h_old, dt, dtemp: float = 0.0):
        e_new = np.asarray(e_new, dtype=float)
        n = e_new.shape[0]
        a_t = self.visco.shift(dtemp) if self.visco is not None else 1.0
        de = e_new - e_old
        tr, dtr = e_new[:, :3].sum(axis=1), de[:, :3].sum(axis=1)
        dev = (e_new - tr[:, None] / 3.0 * _M) * _HALF
        ddev = (de - dtr[:, None] / 3.0 * _M) * _HALF
        k_inf = self.bulk * (1.0 - sum(k for k, _ in self.k_terms))
        g_inf = self.shear * (1.0 - sum(g for g, _ in self.g_terms))
        h_new = np.empty((n, self.n_state))
        s = 2.0 * g_inf * dev
        g_alg = g_inf
        for i, (g, tau) in enumerate(self.g_terms):
            x = dt / (a_t * tau)
            f = float(_phi(x))
            q = math.exp(-x) * h_old[:, 6 * i : 6 * i + 6] + (2.0 * self.shear * g * f) * ddev
            h_new[:, 6 * i : 6 * i + 6] = q
            s = s + q
            g_alg += self.shear * g * f
        p = k_inf * tr
        k_alg = k_inf
        at = 6 * len(self.g_terms)
        for j, (k, tau) in enumerate(self.k_terms):
            x = dt / (a_t * tau)
            f = float(_phi(x))
            r = math.exp(-x) * h_old[:, at + j] + (self.bulk * k * f) * dtr
            h_new[:, at + j] = r
            p = p + r
            k_alg += self.bulk * k * f
        stress = s + p[:, None] * _M
        tangent = np.broadcast_to(isotropic_matrix(k_alg - 2.0 * g_alg / 3.0, g_alg), (n, 6, 6))
        return stress, tangent, h_new


# -- hyperelasticity -----------------------------------------------------------------------


@dataclass(frozen=True)
class Hyperelasticity:
    """A hyperelastic law (see the module): ``model`` (:data:`HYPERELASTIC_MODELS`) and
    its ``parameters`` — ``mu`` (neo-Hookean, Arruda-Boyce, eight chains, Gent, van der
    Waals, Fung, Holzapfel, Blatz-Ko), the coefficients ``cij`` of the polynomial of
    Rivlin (``c10`` and ``c01`` of Mooney and Rivlin, with ``c11``, ``c20``, ``c02`` and
    the terms of the third order for their forms of five and nine parameters; ``c10``,
    ``c20``, ``c30`` of Yeoh; any ``cij`` up to the sixth order for ``polynomial``),
    ``lock`` (the locking stretch of Arruda and Boyce, of the eight chains and of van der
    Waals; ``Jm`` of Gent), ``a`` and ``beta`` (van der Waals: the interaction of the
    chains, the share of the second invariant), ``b`` (Fung: the rate of the exponential),
    ``terms`` (Ogden and the hyperfoam: the pairs ``(mu_p, alpha_p)``; the shear modulus
    is ``sum_p mu_p alpha_p / 2`` for Ogden, ``sum_p mu_p`` for the hyperfoam), ``k1``,
    ``k2``, ``kappa`` and ``fibres`` (Holzapfel: the stiffness and the rate of the
    families of fibres, their dispersion, their directions in the axes of the sample),
    ``f`` (Blatz-Ko: the share of the first invariant) — ``bulk``, the bulk modulus of
    the volumetric energy, and ``volumetric`` (:data:`VOLUMETRIC`: ``K/2 (J - 1)^2``,
    ``K/2 (ln J)^2`` or ``K/4 (J^2 - 1 - 2 ln J)``; the compressible laws, Blatz-Ko and
    the hyperfoam, take their Poisson's ratio from the bulk modulus). ``mullins``: the
    softening of Ogden and Roxburgh, ``(r, m, beta)``. The shear and bulk moduli not
    given are those of the material (``young``, ``poisson``); given, they must agree
    with them (:meth:`initial_moduli`), or the material takes those of the law."""

    model: str = "neo_hookean"
    parameters: tuple = ()
    bulk: float | None = None
    volumetric: str = "quadratic"
    mullins: tuple | None = None

    def __post_init__(self) -> None:
        if self.model not in HYPERELASTIC_MODELS:
            raise ValueError(
                f"hyperelasticity.model: one of {tuple(HYPERELASTIC_MODELS)}, got {self.model!r}"
            )
        if self.volumetric not in VOLUMETRIC:
            raise ValueError(f"hyperelasticity.volumetric: one of {VOLUMETRIC}")
        params = dict(self.parameters)
        allowed, required = _PARAMETERS[self.model]
        order = _ORDERS.get(self.model)
        unknown = sorted(
            name
            for name in params
            if name not in allowed and not (order and 1 <= sum(_powers(name) or (9,)) <= order)
        )
        if unknown:
            names = sorted(allowed) + ([f"cij (i + j <= {order})"] if order else [])
            raise ValueError(
                f"hyperelasticity ({self.model}): unknown parameters {unknown}; allowed: {names}"
            )
        missing = sorted(required - set(params))
        if missing:
            raise ValueError(f"hyperelasticity ({self.model}): give {missing}")
        if order and not params:
            raise ValueError(f"hyperelasticity ({self.model}): give its coefficients cij")
        if "terms" in params:
            terms = _pairs(params["terms"], "hyperelasticity.terms", "mu", "alpha")
            if not terms or any(a == 0.0 for _, a in terms):
                raise ValueError("hyperelasticity.terms: pairs (mu_p, alpha_p), alpha_p not zero")
            params["terms"] = terms
        if "fibres" in params:
            params["fibres"] = _directions(params["fibres"])
        for name, value in params.items():
            if name not in ("terms", "fibres"):
                params[name] = float(value)
                if not math.isfinite(params[name]):
                    raise ValueError(f"hyperelasticity.{name}: a finite value")
        if params.get("lock", 1.0) <= 0:
            raise ValueError("hyperelasticity.lock: positive")
        if self.model == "van_der_waals":
            if params["lock"] ** 2 <= 3.0:
                raise ValueError("hyperelasticity.lock (van der Waals): above the square root of 3")
            if not 0.0 <= params.get("beta", 0.0) <= 1.0 or params.get("a", 0.0) < 0:
                raise ValueError(
                    "hyperelasticity (van der Waals): beta from 0 to 1, a not negative"
                )
        if self.model == "eight_chain" and params["lock"] <= 1.0:
            raise ValueError("hyperelasticity.lock (eight chains): above 1")
        if self.model == "fung" and not params["b"] > 0:
            raise ValueError("hyperelasticity.b: positive")
        if self.model == "holzapfel":
            if params["k1"] < 0 or not params["k2"] > 0:
                raise ValueError("hyperelasticity (Holzapfel): k1 not negative, k2 positive")
            if not 0.0 <= params.get("kappa", 0.0) <= 1.0 / 3.0:
                raise ValueError(
                    "hyperelasticity.kappa: from 0 (aligned fibres) to 1/3 (isotropic)"
                )
        if self.model == "blatz_ko" and not 0.0 <= params.get("f", 0.0) <= 1.0:
            raise ValueError("hyperelasticity.f: from 0 to 1")
        object.__setattr__(self, "parameters", tuple(sorted(params.items())))
        if self.bulk is not None:
            if not (math.isfinite(self.bulk) and self.bulk > 0):
                raise ValueError("hyperelasticity.bulk: a positive bulk modulus")
            object.__setattr__(self, "bulk", float(self.bulk))
        if self.model in _COMPRESSIBLE and self.volumetric != "quadratic":
            raise ValueError(
                f"hyperelasticity ({self.model}): a compressible law, without a volumetric "
                "energy of its own to choose"
            )
        if self.mullins is not None:
            try:
                r, m, beta = (float(v) for v in self.mullins)
            except (TypeError, ValueError):
                raise ValueError("hyperelasticity.mullins: three numbers (r, m, beta)") from None
            if not (r > 1.0 and m >= 0.0 and beta >= 0.0 and m + beta > 0.0):
                raise ValueError(
                    "hyperelasticity.mullins: r above 1, m and beta not negative and not both zero"
                )
            if self.model == "saint_venant_kirchhoff":
                raise ValueError("hyperelasticity.mullins: not for Saint-Venant and Kirchhoff")
            object.__setattr__(self, "mullins", (r, m, beta))

    @classmethod
    def from_config(cls, spec) -> Hyperelasticity:
        """The name of a model, or a mapping ``model`` with its parameters (``mu``,
        ``c10``..., ``terms``, ``fibres``), ``bulk``, ``volumetric`` and ``mullins``."""
        if isinstance(spec, Hyperelasticity):
            return spec
        if isinstance(spec, str):
            return cls(spec)
        if not isinstance(spec, dict):
            raise ValueError("hyperelasticity: the name of a model, or a mapping")
        spec = dict(spec)
        model = spec.pop("model", "neo_hookean")
        bulk = spec.pop("bulk", None)
        volumetric = spec.pop("volumetric", "quadratic")
        mullins = spec.pop("mullins", None)
        return cls(model, tuple(spec.items()), bulk, volumetric, mullins)

    def to_config(self) -> dict:
        out: dict = {"model": self.model}
        for name, value in self.parameters:
            out[name] = [list(t) for t in value] if name in ("terms", "fibres") else value
        if self.bulk is not None:
            out["bulk"] = self.bulk
        if self.volumetric != "quadratic":
            out["volumetric"] = self.volumetric
        if self.mullins is not None:
            out["mullins"] = list(self.mullins)
        return out

    def initial_moduli(self) -> tuple[float | None, float | None]:
        """The bulk and shear moduli of the law at the natural state (None for those it
        takes from the material)."""
        p = dict(self.parameters)
        shear = None
        if self.model in _POLYNOMIALS:
            shear = 2.0 * (p.get("c10", 0.0) + p.get("c01", 0.0))
        elif self.model == "ogden":
            shear = 0.5 * sum(m * a for m, a in p["terms"])
        elif self.model == "hyperfoam":
            shear = sum(m for m, _ in p["terms"])
        elif self.model == "arruda_boyce" and "mu" in p:
            shear = p["mu"] * _arruda_factor(p["lock"])
        elif self.model == "eight_chain" and "mu" in p:
            shear = p["mu"] * _chain_factor(p["lock"])
        elif "mu" in p:
            shear = p["mu"]
        if shear is not None and not shear > 0:
            raise ValueError("hyperelasticity: the shear modulus at the natural state is positive")
        return self.bulk, shear

    def law(self, bulk: float | None = None, shear: float | None = None):
        """The law of these parameters on its own (stresses, tangents, energies, the
        responses to the tests of rubbers: :meth:`_Hyperelastic.nominal_stress`), with
        the ``bulk`` and ``shear`` moduli that it does not give."""
        own_bulk, own_shear = self.initial_moduli()
        bulk = own_bulk if own_bulk is not None else bulk
        shear = own_shear if own_shear is not None else shear
        if bulk is None or shear is None:
            raise ValueError("hyperelasticity: give the bulk and shear moduli that the law has not")
        return _Hyperelastic(self, float(bulk), float(shear), bulk - 2.0 * shear / 3.0)


def _powers(name: str) -> tuple[int, int] | None:
    """The powers ``(i, j)`` of a coefficient ``cij`` of the polynomial of Rivlin."""
    if len(name) == 3 and name[0] == "c" and name[1:].isdigit():
        return int(name[1]), int(name[2])
    return None


def _directions(value) -> tuple[tuple[float, float, float], ...]:
    """The unit directions of families of fibres: one vector, or several."""
    try:
        array = np.asarray(value, dtype=float).reshape(-1, 3)
    except (TypeError, ValueError):
        raise ValueError("hyperelasticity.fibres: directions [x, y, z], one per family") from None
    norms = np.linalg.norm(array, axis=1)
    if not len(array) or not np.all(np.isfinite(array)) or np.any(norms == 0.0):
        raise ValueError("hyperelasticity.fibres: directions [x, y, z], one per family")
    return tuple(tuple(float(v) for v in row) for row in array / norms[:, None])


def _arruda_factor(lock: float) -> float:
    """The initial shear modulus of the series of Arruda and Boyce over its ``mu``."""
    return sum(2.0 * i * a * 3.0 ** (i - 1) / lock ** (2 * i - 2)
               for i, a in enumerate(_ARRUDA, start=1))  # fmt: skip


def _chain_factor(lock: float) -> float:
    """The initial shear modulus of the eight chains (Cohen's approximant of the inverse
    Langevin function) over their ``mu``."""
    x2 = 1.0 / (lock * lock)
    return (3.0 - x2) / (3.0 * (1.0 - x2))


def _outer(a, b):
    return a[:, :, None] * b[:, None, :]


def _box(cinv):
    """The Voigt matrices (n, 6, 6) of ``1/2 (Cinv_ik Cinv_jl + Cinv_il Cinv_jk)``."""
    a = cinv[:, _I0[:, None], _I0[None, :]] * cinv[:, _I1[:, None], _I1[None, :]]
    b = cinv[:, _I0[:, None], _I1[None, :]] * cinv[:, _I1[:, None], _I0[None, :]]
    return 0.5 * (a + b)


def _invariant_law(c6, cinv6, cinv, i1, j, w):
    """The second Piola-Kirchhoff stresses (n, 6) and their tangents ``dS/dE`` (n, 6, 6)
    of an energy ``w(I1, I2, J)`` whose first and second derivatives are ``w = (w1, w2,
    wj, w11, w12, w1j, w22, w2j, wjj)``: ``S = 2 (w1 A1 + w2 A2 + wj AJ)`` with ``A1 = I``,
    ``A2 = I1 I - C``, ``AJ = J/2 C^-1`` and ``dS/dE = 4 [sum_ab w_ab A_a (x) A_b + w2 (I
    (x) I - I_sym) + wj (J/4 C^-1 (x) C^-1 - J/2 C^-1 (.) C^-1)]``."""
    w1, w2, wj, w11, w12, w1j, w22, w2j, wjj = w
    n = len(i1)
    a1 = np.broadcast_to(_M, (n, 6))
    a2 = i1[:, None] * _M - c6
    aj = 0.5 * j[:, None] * cinv6
    s = 2.0 * (w1[:, None] * a1 + w2[:, None] * a2 + wj[:, None] * aj)
    t = w11[:, None, None] * _outer(a1, a1) + w22[:, None, None] * _outer(a2, a2)
    t += wjj[:, None, None] * _outer(aj, aj)
    t += w12[:, None, None] * (_outer(a1, a2) + _outer(a2, a1))
    t += w1j[:, None, None] * (_outer(a1, aj) + _outer(aj, a1))
    t += w2j[:, None, None] * (_outer(a2, aj) + _outer(aj, a2))
    t += w2[:, None, None] * _A2
    t += (wj * j)[:, None, None] * (0.25 * _outer(cinv6, cinv6) - 0.5 * _box(cinv))
    return s, 4.0 * t


def _spectral(vec, lam2, s_a, d):
    """The stresses and tangents of an isotropic law in the principal axes ``vec`` of
    ``C`` (of eigenvalues ``lam2``): ``S = sum_a S_a N_a (x) N_a`` and ``dS/dE = sum_ab
    D_ab N_aa (x) N_bb + sum_(a<b) 4 G_ab P_ab (x) P_ab`` with ``D_ab = (1/l_b)
    dS_a/dl_b``, ``G_ab = (S_b - S_a) / (l_b^2 - l_a^2)`` (its limit ``(D_bb - D_ab) / 2``
    for equal stretches) and ``P_ab`` the symmetric part of ``N_a (x) N_b``."""
    m = vec[:, _I0, :] * vec[:, _I1, :]  # N_a (x) N_a in Voigt components: (n, 6, a)
    s = np.einsum("nka,na->nk", m, s_a)
    t = np.einsum("nia,nab,njb->nij", m, d, m)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        gap = lam2[:, b] - lam2[:, a]
        close = np.abs(gap) < 1e-7 * (lam2[:, a] + lam2[:, b])
        g = np.where(close, 0.5 * (d[:, b, b] - d[:, a, b]),
                     (s_a[:, b] - s_a[:, a]) / np.where(close, 1.0, gap))  # fmt: skip
        pab = 0.5 * (vec[:, _I0, a] * vec[:, _I1, b] + vec[:, _I0, b] * vec[:, _I1, a])
        t += (4.0 * g)[:, None, None] * _outer(pab, pab)
    return s, t


class _Hyperelastic:
    """The integration of a :class:`Hyperelasticity` (a law of the Green-Lagrange strain:
    the second Piola-Kirchhoff stress and ``dS/dE``), with the branches of a
    :class:`Viscoelasticity` on its isochoric and volumetric stresses (Simo 1987), or the
    softening of Mullins on its isochoric stress (Ogden & Roxburgh 1999). State of a
    viscoelastic point: the isochoric and volumetric stresses at the start of the step,
    then those of the branches; of a point that softens: the largest isochoric energy
    that it has reached."""

    kind = "hyperelastic"
    symmetric = True
    thermal = True

    def __init__(self, hyper: Hyperelasticity, bulk: float, shear: float, lam: float,
                 visco: Viscoelasticity | None = None):  # fmt: skip
        self.hyper, self.visco = hyper, visco
        self.model = hyper.model
        p = dict(hyper.parameters)
        self.k = float(hyper.bulk if hyper.bulk is not None else bulk)
        self.mu = float(p.get("mu", shear))
        if "mu" not in p:  # the shear modulus of the material
            if self.model == "arruda_boyce":
                self.mu = shear / _arruda_factor(p["lock"])
            elif self.model == "eight_chain":
                self.mu = shear / _chain_factor(p["lock"])
        self.lam = float(lam)
        self.shear = float(shear)
        self.p = p
        if self.model in _POLYNOMIALS:  # the polynomial of Rivlin: (i, j, cij)
            self.coefficients = tuple((*_powers(name), c) for name, c in p.items() if c != 0.0)
        if self.model in ("blatz_ko", "hyperfoam"):
            # their Poisson's ratio: K = mu (2/3 + 2 beta), beta = nu / (1 - 2 nu)
            first = self.mu if self.model == "blatz_ko" else sum(m for m, _ in p["terms"])
            self.beta = 0.5 * (self.k / first - 2.0 / 3.0)
            if self.beta <= -1.0 / 3.0:
                raise ValueError(f"hyperelasticity ({self.model}): a positive bulk modulus")
        self.structure = ()
        if self.model == "holzapfel":
            kappa = p.get("kappa", 0.0)
            self.structure = tuple(
                (kappa * np.eye(3) + (1.0 - 3.0 * kappa) * np.outer(a, a))[_I0, _I1]
                for a in np.asarray(p["fibres"])
            )
        self.mullins = hyper.mullins
        if self.mullins is not None and visco is not None:
            raise ValueError("hyperelasticity.mullins: not with a viscoelasticity")
        self.g_terms = visco.shear if visco is not None else ()
        self.k_terms = visco.bulk if visco is not None else ()
        self.n_state = 6 * (2 + len(self.g_terms) + len(self.k_terms)) if visco else 0
        if self.mullins is not None:
            self.n_state = 1

    # -- energies --------------------------------------------------------------------------

    def _psi(self, i1b, i2b):
        """The derivatives ``(psi1, psi2, psi11, psi12, psi22)`` of the isochoric energy
        and its value."""
        zero = np.zeros_like(i1b)
        p, x = self.p, i1b - 3.0
        if self.model in ("neo_hookean", "holzapfel"):
            return (zero + 0.5 * self.mu, zero, zero, zero, zero), 0.5 * self.mu * x
        if self.model in _POLYNOMIALS:
            y = i2b - 3.0
            d1, d2, d11, d12, d22, value = (zero.copy() for _ in range(6))
            for i, j, c in self.coefficients:
                xi, yj = x**i, y**j
                value += c * xi * yj
                if i >= 1:
                    d1 += c * i * x ** (i - 1) * yj
                if j >= 1:
                    d2 += c * j * xi * y ** (j - 1)
                if i >= 2:
                    d11 += c * i * (i - 1) * x ** (i - 2) * yj
                if i >= 1 and j >= 1:
                    d12 += c * i * j * x ** (i - 1) * y ** (j - 1)
                if j >= 2:
                    d22 += c * j * (j - 1) * xi * y ** (j - 2)
            return (d1, d2, d11, d12, d22), value
        if self.model == "arruda_boyce":
            lock = p["lock"]
            d1, d11, value = zero.copy(), zero.copy(), zero.copy()
            for i, a in enumerate(_ARRUDA, start=1):
                c = self.mu * a / lock ** (2 * i - 2)
                value += c * (i1b**i - 3.0**i)
                d1 += c * i * i1b ** (i - 1)
                if i > 1:
                    d11 += c * i * (i - 1) * i1b ** (i - 2)
            return (d1, zero, d11, zero, zero), value
        if self.model == "eight_chain":
            # Cohen's approximant x (3 - x^2) / (1 - x^2) of the inverse Langevin function
            # of the stretch of the chains over their locking stretch
            scale = 3.0 * p["lock"] ** 2
            u = i1b / scale
            u = np.where(u < 1.0, u, np.nan)  # beyond the locking: a failed point
            u0 = 3.0 / scale
            d1 = self.mu / 6.0 * (3.0 - u) / (1.0 - u)
            d11 = self.mu / 6.0 * 2.0 / ((1.0 - u) ** 2 * scale)
            value = self.mu * scale / 6.0 * (u - u0 - 2.0 * np.log((1.0 - u) / (1.0 - u0)))
            return (d1, zero, d11, zero, zero), value
        if self.model == "van_der_waals":
            span = p["lock"] ** 2 - 3.0
            a, beta = p.get("a", 0.0), p.get("beta", 0.0)
            # the mixture of the invariants from 3 (its square root is not differentiable
            # at the natural state, where the limits of the terms are those of this floor)
            mix = np.maximum((1.0 - beta) * i1b + beta * i2b - 3.0, 1e-12)
            eta = np.sqrt(mix / span)
            eta = np.where(eta < 1.0, eta, np.nan)  # beyond the locking: a failed point
            root = np.sqrt(0.5 * mix)
            rest = (1.0 - beta) * i1b + beta * i2b - 3.0 <= 1e-12  # the natural state
            d = np.where(rest, 0.5 * self.mu, self.mu * (0.5 / (1.0 - eta) - 0.5 * a * root))
            dd = self.mu * (0.25 / (span * eta * (1.0 - eta) ** 2) - 0.125 * a / root)
            dd = np.where(rest, 0.0, dd)
            value = self.mu * (-span * (np.log1p(-eta) + eta) - (2.0 / 3.0) * a * root**3)
            return (
                (1.0 - beta) * d,
                beta * d,
                (1.0 - beta) ** 2 * dd,
                (1.0 - beta) * beta * dd,
                beta * beta * dd,
            ), value
        if self.model == "fung":
            ex = np.exp(p["b"] * x)
            return (
                0.5 * self.mu * ex,
                zero,
                0.5 * self.mu * p["b"] * ex,
                zero,
                zero,
            ), 0.5 * self.mu / p["b"] * (ex - 1.0)
        jm = p["lock"]  # Gent
        rest = jm - x
        rest = np.where(rest > 0, rest, np.nan)  # beyond the locking: a failed point
        d1 = 0.5 * self.mu * jm / rest
        return (d1, zero, d1 / rest, zero, zero), -0.5 * self.mu * jm * np.log(rest / jm)

    def _volumetric(self, j):
        """``U'``, ``U''`` and ``U`` of the volumetric energy."""
        k = self.k
        if self.hyper.volumetric == "quadratic":
            return k * (j - 1.0), k + 0.0 * j, 0.5 * k * (j - 1.0) ** 2
        log = np.log(j)
        if self.hyper.volumetric == "logarithmic":
            return k * log / j, k * (1.0 - log) / (j * j), 0.5 * k * log * log
        return (
            0.5 * k * (j - 1.0 / j),
            0.5 * k * (1.0 + 1.0 / (j * j)),
            0.25 * k * (j * j - 1.0 - 2.0 * log),
        )

    def _blatz_ko(self, i1, i2, j):
        """The derivatives of the energy of Blatz and Ko with respect to ``(I1, I2, J)``
        (the order of :func:`_invariant_law`) and its value."""
        mu, f, beta = self.mu, self.p.get("f", 0.0), self.beta
        zero = np.zeros_like(j)
        if abs(beta) > 1e-9:
            down, up = j ** (-2.0 * beta), j ** (2.0 * beta)
            vol_down, vol_up = (down - 1.0) / beta, (up - 1.0) / beta
        else:  # no lateral contraction: the limits of the powers
            down, up = np.ones_like(j), np.ones_like(j)
            vol_down, vol_up = -2.0 * np.log(j), 2.0 * np.log(j)
        j2 = j * j
        w = (
            zero + 0.5 * mu * f,
            0.5 * mu * (1.0 - f) / j2,
            -mu * f * down / j + mu * (1.0 - f) * (up / j - i2 / (j2 * j)),
            zero,
            zero,
            zero,
            zero,
            -mu * (1.0 - f) / (j2 * j),
            mu * f * (2.0 * beta + 1.0) * down / j2
            + mu * (1.0 - f) * ((2.0 * beta - 1.0) * up / j2 + 3.0 * i2 / (j2 * j2)),
        )
        value = 0.5 * mu * f * (i1 - 3.0 + vol_down) + 0.5 * mu * (1.0 - f) * (
            i2 / j2 - 3.0 + vol_up
        )
        return w, value

    def _principal(self, c, j):
        """The stresses, tangents and energy of the laws of the principal stretches, in
        the principal axes of ``C`` (:func:`_spectral`): the isochoric part of Ogden's
        law, of the stretches ``J^(-1/3) l_a``, or the whole of the hyperfoam, of the
        stretches themselves."""
        n = c.shape[0]
        lam2, vec = np.linalg.eigh(c)
        lam2 = np.maximum(lam2, 1e-300)
        lam = np.sqrt(lam2)
        s_a = np.zeros((n, 3))
        d = np.zeros((n, 3, 3))
        w = np.zeros(n)
        diagonal = [0, 1, 2]
        if self.model == "hyperfoam":
            beta = self.beta
            for mu, alpha in self.p["terms"]:
                pw = lam**alpha
                if abs(beta) > 1e-9:
                    jp = j ** (-alpha * beta)
                    w += 2.0 * mu / alpha**2 * (pw.sum(axis=1) - 3.0 + (jp - 1.0) / beta)
                else:
                    jp = np.ones(n)
                    w += 2.0 * mu / alpha**2 * (pw.sum(axis=1) - 3.0 - alpha * np.log(j))
                s_a += 2.0 * mu / alpha * (pw - jp[:, None]) / lam2
                d += (2.0 * mu * beta * jp)[:, None, None] / (lam2[:, :, None] * lam2[:, None, :])
                d[:, diagonal, diagonal] += 2.0 * mu * pw / (lam2 * lam2)
        else:
            lb = lam * (j ** (-1.0 / 3.0))[:, None]
            for mu, alpha in self.p["terms"]:
                pw = lb**alpha
                mean = pw.mean(axis=1, keepdims=True)
                w += mu / alpha * (pw.sum(axis=1) - 3.0)
                s_a += mu * (pw - mean) / lam2
                core = -(pw[:, :, None] + pw[:, None, :]) / 3.0 + mean[:, :, None] / 3.0
                core[:, diagonal, diagonal] += pw
                d += mu * alpha * core / (lam2[:, :, None] * lam2[:, None, :])
        d[:, diagonal, diagonal] -= 2.0 * s_a / lam2
        s, t = _spectral(vec, lam2, s_a, d)
        return s, t, w

    def _fibres(self, c6, cinv, cinv6, j):
        """The stresses, tangents and energy of the families of fibres of Holzapfel,
        Gasser and Ogden: ``k1 / (2 k2) (exp(k2 E^2) - 1)`` of the strain ``E = J^(-2/3) C
        : H - 1`` of each family, ``H = kappa I + (1 - 3 kappa) a (x) a`` its structure
        tensor, where it is positive (the fibres carry no compression)."""
        k1, k2 = self.p["k1"], self.p["k2"]
        n = len(j)
        s, t, w = np.zeros((n, 6)), np.zeros((n, 6, 6)), np.zeros(n)
        scale = j ** (-2.0 / 3.0)
        pairs = _outer(cinv6, cinv6)
        box = _box(cinv)
        for h6 in self.structure:
            q = c6 @ (h6 / _HALF)  # C : H
            strain = np.maximum(scale * q - 1.0, 0.0)
            ex = np.exp(k2 * strain * strain)
            first = k1 * strain * ex
            second = np.where(strain > 0, k1 * (1.0 + 2.0 * k2 * strain * strain) * ex, 0.0)
            g = scale[:, None] * (h6 - (q / 3.0)[:, None] * cinv6)  # dE/dC
            h = np.broadcast_to(h6, (n, 6))
            curvature = (
                -(_outer(h, cinv6) + _outer(cinv6, h)) / 3.0
                + (q / 9.0)[:, None, None] * pairs
                + (q / 3.0)[:, None, None] * box
            )
            s += 2.0 * first[:, None] * g
            t += second[:, None, None] * _outer(g, g)
            t += (first * scale)[:, None, None] * curvature
            w += 0.5 * k1 / k2 * (ex - 1.0)
        return s, 4.0 * t, w

    # -- stresses --------------------------------------------------------------------------

    def _parts(self, e):
        """The isochoric and volumetric stresses and tangents of strains ``e`` (n, 6), and
        the energies of both parts (the whole of the compressible laws in the first);
        NaN where ``C`` is not positive definite."""
        n = e.shape[0]
        if self.model == "saint_venant_kirchhoff":
            d = isotropic_matrix(self.lam, self.shear)
            s = e @ d
            zero = np.zeros((n, 6))
            return (
                s,
                np.broadcast_to(d, (n, 6, 6)),
                zero,
                np.zeros((n, 6, 6)),
                (
                    0.5 * np.einsum("ni,ni->n", s, e),
                    np.zeros(n),
                ),
            )
        c = np.empty((n, 3, 3))
        for k, (a, b) in enumerate(_PAIRS):
            c[:, a, b] = c[:, b, a] = e[:, k] * (2.0 if a == b else 1.0)
        c[:, [0, 1, 2], [0, 1, 2]] += 1.0
        det = np.linalg.det(c)
        ok = np.isfinite(det) & (det > 0)
        if not ok.all():
            c = np.where(ok[:, None, None], c, np.eye(3))
        cinv = np.linalg.inv(c)
        j = np.sqrt(np.where(ok, det, 1.0))
        c6, cinv6 = c[:, _I0, _I1], cinv[:, _I0, _I1]
        i1 = c[:, 0, 0] + c[:, 1, 1] + c[:, 2, 2]
        zero = np.zeros(n)
        if self.model in _COMPRESSIBLE:  # no split: the whole law in the first part
            if self.model == "blatz_ko":
                i2 = 0.5 * (i1 * i1 - np.einsum("nij,nij->n", c, c))
                derivatives, w = self._blatz_ko(i1, i2, j)
                s_iso, t_iso = _invariant_law(c6, cinv6, cinv, i1, j, derivatives)
            else:
                s_iso, t_iso, w = self._principal(c, j)
            s_vol, t_vol, u = np.zeros((n, 6)), np.zeros((n, 6, 6)), zero
        else:
            du, ddu, u = self._volumetric(j)
            vol = (zero, zero, du, zero, zero, zero, zero, zero, ddu)
            s_vol, t_vol = _invariant_law(c6, cinv6, cinv, i1, j, vol)
            if self.model == "ogden":
                s_iso, t_iso, w = self._principal(c, j)
            else:
                i2 = 0.5 * (i1 * i1 - np.einsum("nij,nij->n", c, c))
                a, b = j ** (-2.0 / 3.0), j ** (-4.0 / 3.0)
                i1b, i2b = a * i1, b * i2
                (p1, p2, p11, p12, p22), w = self._psi(i1b, i2b)
                d1, d2 = -2.0 / 3.0 * i1b / j, -4.0 / 3.0 * i2b / j
                iso = (
                    p1 * a,
                    p2 * b,
                    p1 * d1 + p2 * d2,
                    p11 * a * a,
                    p12 * a * b,
                    a * (p11 * d1 + p12 * d2) - 2.0 / 3.0 * p1 * a / j,
                    p22 * b * b,
                    b * (p12 * d1 + p22 * d2) - 4.0 / 3.0 * p2 * b / j,
                    p11 * d1 * d1 + 2.0 * p12 * d1 * d2 + p22 * d2 * d2
                    + p1 * (10.0 / 9.0) * i1b / (j * j) + p2 * (28.0 / 9.0) * i2b / (j * j),
                )  # fmt: skip
                s_iso, t_iso = _invariant_law(c6, cinv6, cinv, i1, j, iso)
                if self.structure:
                    s_f, t_f, w_f = self._fibres(c6, cinv, cinv6, j)
                    s_iso, t_iso, w = s_iso + s_f, t_iso + t_f, w + w_f
        if not ok.all():
            for array in (s_iso, t_iso, s_vol, t_vol):
                array[~ok] = np.nan
        return s_iso, t_iso, s_vol, t_vol, (w, u)

    def energy(self, e) -> np.ndarray:
        """The strain energy per unit reference volume of Green-Lagrange strains ``e``
        (n, 6; engineering shears), that of the law without softening."""
        e = np.atleast_2d(np.asarray(e, dtype=float))
        w, u = self._parts(e)[4]
        return w + u

    def step(self, e_new, e_old, h_old, dt, dtemp: float = 0.0):
        e_new = np.asarray(e_new, dtype=float)
        n = e_new.shape[0]
        stress, tangent = np.empty((n, 6)), np.empty((n, 6, 6))
        h_new = np.empty((n, self.n_state))
        for lo in range(0, n, _CHUNK):
            sl = slice(lo, min(lo + _CHUNK, n))
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                s_iso, t_iso, s_vol, t_vol, (w, _) = self._parts(e_new[sl])
            if self.mullins is not None:
                # the isochoric stress times eta = 1 - erf((Wm - W) / (m + beta Wm)) / r,
                # Wm the largest isochoric energy reached: 1 while it grows
                r, m, beta = self.mullins
                reached = np.fmax(h_old[sl, 0], w)
                room = m + beta * reached
                x = (reached - w) / np.where(room > 0, room, 1.0)
                eta = 1.0 - erf(x) / r
                slope = np.where(reached > w, 2.0 / (r * math.sqrt(math.pi)) * np.exp(-x * x), 0.0)
                slope = slope / np.where(room > 0, room, 1.0)
                stress[sl] = eta[:, None] * s_iso + s_vol
                tangent[sl] = eta[:, None, None] * t_iso + t_vol
                tangent[sl] += slope[:, None, None] * _outer(s_iso, s_iso)
                h_new[sl, 0] = reached
                continue
            if self.visco is None:
                stress[sl], tangent[sl] = s_iso + s_vol, t_iso + t_vol
                continue
            a_t = self.visco.shift(dtemp)
            old = h_old[sl]
            new = h_new[sl]
            new[:, 0:6], new[:, 6:12] = s_iso, s_vol
            factors = []
            at = 12
            for terms, part, start in ((self.g_terms, s_iso, 0), (self.k_terms, s_vol, 6)):
                weight = 1.0 - sum(c for c, _ in terms)
                total = weight * part
                for c, tau in terms:
                    x = dt / (a_t * tau)
                    f = float(_phi(x))
                    q = math.exp(-x) * old[:, at : at + 6] + c * f * (
                        part - old[:, start : start + 6]
                    )
                    new[:, at : at + 6] = q
                    total = total + q
                    weight += c * f
                    at += 6
                factors.append((total, weight))
            stress[sl] = factors[0][0] + factors[1][0]
            tangent[sl] = factors[0][1] * t_iso + factors[1][1] * t_vol
        return stress, tangent, h_new

    # -- the tests of rubbers ----------------------------------------------------------------

    def nominal_stress(self, mode: str, stretch, *, incompressible: bool = False) -> np.ndarray:
        """The nominal stress of a homogeneous test (:data:`TESTS`) at the stretches
        ``stretch`` (``shear``: the amounts of shear): the force over the initial section
        along the stretch of ``uniaxial`` tension (the two other stresses zero), of
        ``equibiaxial`` tension (two equal stretches, the stress across them zero) and of
        ``planar`` tension (the width held: pure shear), and the shear stress of a simple
        ``shear``. The lateral stretches are those that cancel the lateral stresses of
        the law with its bulk modulus (Newton's iterations on its tangent); for an
        ``incompressible`` solid, the closed forms of the isochoric energy (the laws of
        the invariants and of Ogden; isotropic laws only: the fibres of Holzapfel have
        their directions)."""
        if mode not in TESTS:
            raise ValueError(f"mode: one of {TESTS}")
        if self.structure:
            raise ValueError("nominal_stress: isotropic laws (the fibres have directions)")
        lam = np.atleast_1d(np.asarray(stretch, dtype=float))
        if incompressible:
            return self._incompressible(mode, lam)
        n = len(lam)
        e = np.zeros((n, 6))
        if mode == "shear":  # F = 1 + g e1 (x) e2: E22 = g^2/2, 2 E12 = g
            e[:, 1], e[:, 5] = 0.5 * lam * lam, lam
            s = self._parts(e)[0] + self._parts(e)[2]
            return s[:, 5] + lam * s[:, 1]
        if np.any(lam <= 0):
            raise ValueError("stretch: positive")
        e[:, 0] = 0.5 * (lam * lam - 1.0)
        # the axes stretched with the first and those free of stress
        driven = [0, 1] if mode == "equibiaxial" else [0]
        free = [2] if mode in ("equibiaxial", "planar") else [1, 2]
        e[:, driven] = e[:, [0]]
        # the stretch of the free axes, from that of the incompressible solid
        side = lam ** {"uniaxial": -0.5, "equibiaxial": -2.0, "planar": -1.0}[mode]
        for _ in range(60):
            e[:, free] = 0.5 * (side * side - 1.0)[:, None]
            s_iso, t_iso, s_vol, t_vol, _ = self._parts(e)
            residual = (s_iso + s_vol)[:, free[0]]
            slope = (t_iso + t_vol)[:, free[0]][:, free].sum(axis=1) * side
            step = residual / slope
            side = np.maximum(side - step, 0.5 * side)
            if np.all(np.abs(step) <= 1e-13 * side):
                break
        e[:, free] = 0.5 * (side * side - 1.0)[:, None]
        s_iso, _, s_vol, _, _ = self._parts(e)
        return lam * (s_iso + s_vol)[:, 0]

    def _incompressible(self, mode: str, lam) -> np.ndarray:
        """The closed forms of the tests for an incompressible solid."""
        if self.model in (*_COMPRESSIBLE, "saint_venant_kirchhoff"):
            raise ValueError(f"nominal_stress: the {self.model} law is compressible")
        if self.model == "ogden":
            total = np.zeros_like(lam)
            for mu, alpha in self.p["terms"]:
                if mode == "uniaxial":
                    total += mu * (lam ** (alpha - 1.0) - lam ** (-0.5 * alpha - 1.0))
                elif mode == "equibiaxial":
                    total += mu * (lam ** (alpha - 1.0) - lam ** (-2.0 * alpha - 1.0))
                elif mode == "planar":
                    total += mu * (lam ** (alpha - 1.0) - lam ** (-alpha - 1.0))
                else:  # the principal stretch of the shear: l - 1/l = g
                    top = 0.5 * lam + np.sqrt(1.0 + 0.25 * lam * lam)
                    total += mu * (top**alpha - top**-alpha) / (top + 1.0 / top)
            return total
        l2 = lam * lam
        if mode == "uniaxial":
            i1, i2 = l2 + 2.0 / lam, 2.0 * lam + 1.0 / l2
        elif mode == "equibiaxial":
            i1, i2 = 2.0 * l2 + 1.0 / (l2 * l2), l2 * l2 + 2.0 / l2
        elif mode == "planar":
            i1 = i2 = l2 + 1.0 / l2 + 1.0
        else:
            i1 = i2 = 3.0 + l2
        (p1, p2, _, _, _), _ = self._psi(i1, i2)
        if mode == "uniaxial":
            return 2.0 * (lam - 1.0 / l2) * (p1 + p2 / lam)
        if mode == "equibiaxial":
            return 2.0 * (lam - 1.0 / (l2 * l2 * lam)) * (p1 + l2 * p2)
        if mode == "planar":
            return 2.0 * (lam - 1.0 / (l2 * lam)) * (p1 + p2)
        return 2.0 * lam * (p1 + p2)


def fit_hyperelasticity(
    model: str,
    tests,
    *,
    start: dict | None = None,
    fixed: dict | None = None,
    relative: bool = False,
    terms: int = 2,
    **options,
) -> Hyperelasticity:
    """The parameters of a hyperelastic ``model`` fitted to tests of an incompressible
    rubber: ``tests`` maps the modes (:data:`TESTS`) to their ``(stretches, nominal
    stresses)`` (``shear``: the amounts of shear and the shear stresses), fitted together
    by least squares on the stresses (``relative``: on their ratios to the measures, which
    weighs the small strains as the large ones). The polynomials of Rivlin (``mooney_rivlin``
    with the coefficients of ``start``, by default ``c10`` and ``c01``; ``yeoh``;
    ``polynomial``) are linear in their coefficients: one linear solve. The others start
    from ``start`` (by default from the slope of the first points) and keep the
    parameters ``fixed``; ``ogden``: ``terms`` pairs. Returns the law (its bulk modulus,
    that of the material, is not fitted: the tests are those of an incompressible
    solid); ``options`` go to :func:`scipy.optimize.least_squares`."""
    from scipy.optimize import least_squares

    if model not in FITTED:
        raise ValueError(f"fit_hyperelasticity: a model among {FITTED}")
    data = []
    for mode, (stretches, stresses) in dict(tests).items():
        if mode not in TESTS:
            raise ValueError(f"tests: modes among {TESTS}")
        x, y = np.asarray(stretches, dtype=float).ravel(), np.asarray(stresses, dtype=float).ravel()
        if x.shape != y.shape or not len(x):
            raise ValueError(f"tests[{mode!r}]: as many stretches as stresses, at least one")
        data.append((mode, x, y))
    if not data:
        raise ValueError("tests: at least one test")
    measured = np.concatenate([y for _, _, y in data])
    weights = 1.0 / np.abs(measured) if relative else np.ones_like(measured)
    if relative and not np.all(np.isfinite(weights)):
        raise ValueError("tests: no zero stress with relative=True")
    fixed = dict(fixed or {})

    def stresses_of(parameters: dict) -> np.ndarray:
        law = _Hyperelastic(Hyperelasticity(model, tuple(parameters.items())), 1.0, 1.0, 1.0 / 3.0)
        return np.concatenate([law.nominal_stress(m, x, incompressible=True) for m, x, _ in data])

    if model in _POLYNOMIALS:  # linear in the coefficients
        names = sorted(start) if start else {"mooney_rivlin": ["c01", "c10"],
                                             "yeoh": ["c10", "c20", "c30"]}.get(model)  # fmt: skip
        if not names:
            raise ValueError("fit_hyperelasticity (polynomial): the coefficients of start, e.g. "
                             "{'c10': 0, 'c01': 0, 'c20': 0}")  # fmt: skip
        names = [name for name in names if name not in fixed]
        every = dict.fromkeys([*names, *fixed], 0.0)
        basis = np.column_stack([stresses_of({**every, name: 1.0}) for name in names])
        offset = stresses_of({**every, **fixed}) if fixed else 0.0
        solution = np.linalg.lstsq(basis * weights[:, None], (measured - offset) * weights,
                                   rcond=None)[0]  # fmt: skip
        found = {**dict(zip(names, (float(v) for v in solution), strict=True)), **fixed}
        return Hyperelasticity(model, tuple(found.items()))
    # the slope of the smallest strain of the first test: its shear modulus (three, six
    # and four times it in uniaxial, equibiaxial and planar tension)
    mode, x, y = data[0]
    strain = x if mode == "shear" else x - 1.0
    moving = np.flatnonzero((strain != 0) & (y != 0))
    shear = 1.0
    if len(moving):
        k = moving[np.argmin(np.abs(strain[moving]))]
        slope = {"uniaxial": 3.0, "equibiaxial": 6.0, "planar": 4.0, "shear": 1.0}[mode]
        shear = abs(float(y[k] / strain[k])) / slope
    top = max(1.5, *(float(np.max(np.abs(x))) for _, x, _ in data))
    defaults = {
        "neo_hookean": {"mu": shear},
        "arruda_boyce": {"mu": shear, "lock": 2.0 * top},
        "eight_chain": {"mu": shear, "lock": 2.0 * top},
        "gent": {"mu": shear, "lock": 10.0 * top * top},
        "van_der_waals": {"mu": shear, "lock": 2.0 * top, "a": 0.1},
        "fung": {"mu": shear, "b": 0.1},
    }
    if model == "ogden":
        guess = [(shear, 2.0), (0.01 * shear, 4.0), (-0.01 * shear, -2.0)][: max(1, int(terms))]
        initial = dict(start or {})
        pairs = initial.get("terms", guess)
        names = [f"{kind}{p}" for p in range(len(pairs)) for kind in ("mu", "alpha")]
        x0 = np.array([v for pair in pairs for v in pair], dtype=float)

        def build(values):
            return {"terms": [[values[2 * p], values[2 * p + 1]] for p in range(len(pairs))]}

    else:
        initial = {**defaults[model], **(start or {})}
        names = [name for name in initial if name not in fixed]
        x0 = np.array([initial[name] for name in names], dtype=float)

        def build(values):
            return {**dict(zip(names, values, strict=True)), **fixed}

    def residual(values):
        try:
            with np.errstate(all="ignore"):
                found = stresses_of(build(values))
        except ValueError:  # parameters out of their range
            return np.full_like(measured, 1e6 * (1.0 + np.abs(measured).max()))
        found = np.where(np.isfinite(found), found, 1e6 * (1.0 + np.abs(measured).max()))
        return (found - measured) * weights

    solution = least_squares(residual, x0, **{"x_scale": "jac", **options})
    return Hyperelasticity(model, tuple(build(solution.x).items()))


# -- damage --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Damage:
    """An isotropic damage law (see the module): the damage starts at the equivalent
    strain ``threshold`` (or at the uniaxial stress ``strength``: the threshold is
    ``strength / E``) and grows by the ``softening`` law (:data:`SOFTENINGS`) up to the
    equivalent strain ``ultimate`` (linear: no stress beyond it; exponential: the stress
    decays as ``exp(-(kappa - kappa_0) / (ultimate - kappa_0))``), or so that a band of the
    width ``length`` (the size of the elements) dissipates the ``fracture_energy`` per unit
    area. ``equivalent``: the equivalent strain (:data:`EQUIVALENT_STRAINS`), with
    ``ratio`` the compressive strength over the tensile one of the modified von Mises
    strain; ``split`` (:data:`SPLITS`); ``viscosity``: the time of the delayed damage (0:
    none); ``residual``: the fraction of the stiffness that the damage leaves."""

    threshold: float | None = None
    strength: float | None = None
    ultimate: float | None = None
    fracture_energy: float | None = None
    length: float | None = None
    softening: str = "exponential"
    equivalent: str = "mazars"
    ratio: float = 10.0
    split: str = "none"
    viscosity: float = 0.0
    residual: float = 1e-6

    def __post_init__(self) -> None:
        if (self.threshold is None) == (self.strength is None):
            raise ValueError("damage: give the threshold (a strain) or the strength (a stress)")
        for name in ("threshold", "strength", "ultimate", "fracture_energy", "length"):
            value = getattr(self, name)
            if value is not None:
                if not (math.isfinite(value) and value > 0):
                    raise ValueError(f"damage.{name}: positive")
                object.__setattr__(self, name, float(value))
        if (self.ultimate is None) == (self.fracture_energy is None):
            raise ValueError(
                "damage: give the ultimate strain, or the fracture energy and the length of "
                "the elements"
            )
        if self.fracture_energy is not None and self.length is None:
            raise ValueError(
                "damage.length: the size of the elements (the voxels) over which the "
                "fracture energy is dissipated"
            )
        if self.softening not in SOFTENINGS:
            raise ValueError(f"damage.softening: one of {SOFTENINGS}")
        if self.equivalent not in EQUIVALENT_STRAINS:
            raise ValueError(f"damage.equivalent: one of {EQUIVALENT_STRAINS}")
        if self.split not in SPLITS:
            raise ValueError(f"damage.split: one of {SPLITS}")
        if not self.ratio >= 1.0:
            raise ValueError("damage.ratio: the compressive strength over the tensile one (>= 1)")
        if not self.viscosity >= 0.0:
            raise ValueError("damage.viscosity: non-negative")
        if not 0.0 < self.residual < 1.0:
            raise ValueError("damage.residual: a fraction of the stiffness in (0, 1)")
        for name in ("ratio", "viscosity", "residual"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @classmethod
    def from_config(cls, spec) -> Damage:
        """A mapping of the parameters of :class:`Damage`."""
        if isinstance(spec, Damage):
            return spec
        if not isinstance(spec, dict):
            raise ValueError("damage: a mapping of parameters")
        names = set(cls.__dataclass_fields__)
        unknown = sorted(set(spec) - names)
        if unknown:
            raise ValueError(f"damage: unknown keys {unknown}; allowed: {sorted(names)}")
        return cls(**spec)

    def to_config(self) -> dict:
        defaults = Damage(threshold=1.0, ultimate=2.0)
        out = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            keep = name in ("threshold", "strength", "ultimate", "fracture_energy", "length")
            if value is not None and (keep or value != getattr(defaults, name)):
                out[name] = value
        return out

    def strains(self, young: float) -> tuple[float, float]:
        """The threshold ``kappa_0`` and the ultimate strain ``kappa_f`` of the softening
        for the Young's modulus ``young``: those given, or from the strength and the
        fracture energy over the length of the elements (linear: ``kappa_f = 2 G_f / (h E
        kappa_0)``; exponential: ``kappa_f = kappa_0 / 2 + G_f / (h E kappa_0)``)."""
        k0 = self.threshold if self.threshold is not None else self.strength / young
        if self.ultimate is not None:
            kf = self.ultimate
        else:
            density = self.fracture_energy / self.length
            if self.softening == "linear":
                kf = 2.0 * density / (young * k0)
            else:
                kf = 0.5 * k0 + density / (young * k0)
        if not kf > k0:
            largest = 2.0 * (self.fracture_energy or 0.0) / (young * k0 * k0)
            raise ValueError(
                "damage: the ultimate strain must exceed the threshold; with a fracture "
                f"energy, elements smaller than {largest:.4g} (the stress would snap back)"
            )
        return float(k0), float(kf)

    def damage(self, kappa, young: float):
        """The damage ``d(kappa)`` and its derivative, for equivalent strains ``kappa``."""
        k0, kf = self.strains(young)
        kappa = np.asarray(kappa, dtype=float)
        k = np.maximum(kappa, k0)
        if self.softening == "linear":
            d = kf * (k - k0) / (k * (kf - k0))
            dd = kf * k0 / (k * k * (kf - k0))
        else:
            decay = np.exp(-(k - k0) / (kf - k0))
            d = 1.0 - k0 / k * decay
            dd = k0 / (k * k) * decay + k0 / k * decay / (kf - k0)
        top = 1.0 - self.residual
        growing = (kappa > k0) & (d < top)
        return np.where(kappa > k0, np.minimum(d, top), 0.0), np.where(growing, dd, 0.0)


class _DamageLaw:
    """The integration of a :class:`Damage` in small strains. State of a point: the
    largest equivalent strain reached, and the (delayed) damage."""

    kind = "damage"
    thermal = False
    n_state = 2

    def __init__(self, damage: Damage, young: float, poisson: float):
        self.damage = damage
        self.young, self.poisson = float(young), float(poisson)
        self.k = young / (3.0 * (1.0 - 2.0 * poisson))
        self.mu = young / (2.0 * (1.0 + poisson))
        self.d = isotropic_matrix(self.k - 2.0 * self.mu / 3.0, self.mu)
        damage.strains(young)  # checked once
        # the tangent of a growing damage is not symmetric, but for the energy norm
        self.symmetric = damage.equivalent == "energy"

    def scalar(self, h) -> np.ndarray:
        """The damage of the points (the variable ``p`` of the outputs)."""
        return h[:, 1]

    def _equivalent(self, e, tr, plus):
        """The equivalent strains (n) and their derivatives with respect to the strains
        (n, 6: stress-like components, their product with engineering strains a full
        contraction); ``plus``: the part of the effective stress that the damage degrades."""
        dmg = self.damage
        n = e.shape[0]
        if dmg.equivalent == "energy":
            energy = np.einsum("ni,ni->n", plus, e)  # twice the degraded energy density
            eq = np.sqrt(np.maximum(energy, 0.0) / self.young)
            grad = plus / (self.young * np.where(eq > 0, eq, 1.0))[:, None]
            return eq, np.where((eq > 0)[:, None], grad, 0.0)
        if dmg.equivalent == "mazars":
            t = np.empty((n, 3, 3))
            for k, (a, b) in enumerate(_PAIRS):
                t[:, a, b] = t[:, b, a] = e[:, k] * (1.0 if a == b else 0.5)
            w, v = np.linalg.eigh(t)
            pos = np.maximum(w, 0.0)
            eq = np.sqrt(np.einsum("na,na->n", pos, pos))
            m = v[:, _I0, :] * v[:, _I1, :]
            grad = np.einsum("nka,na->nk", m, pos) / np.where(eq > 0, eq, 1.0)[:, None]
            return eq, np.where((eq > 0)[:, None], grad, 0.0)
        # the modified von Mises strain (de Vree, Brekelmans & van Gils 1995)
        k, nu = dmg.ratio, self.poisson
        dev = (e - tr[:, None] / 3.0 * _M) * _HALF
        j2 = 0.5 * (np.einsum("ni,ni->n", dev[:, :3], dev[:, :3])
                    + 2.0 * np.einsum("ni,ni->n", dev[:, 3:], dev[:, 3:]))  # fmt: skip
        a = (k - 1.0) / (2.0 * k * (1.0 - 2.0 * nu))
        b = (k - 1.0) / (1.0 - 2.0 * nu)
        c = 12.0 * k / (1.0 + nu) ** 2
        root = np.sqrt(b * b * tr * tr + c * j2)
        eq = a * tr + root / (2.0 * k)
        safe = np.where(root > 0, root, 1.0)
        grad = a * _M + ((2.0 * b * b * tr)[:, None] * _M + c * dev) / (4.0 * k * safe)[:, None]
        return eq, np.where((root > 0)[:, None], grad, 0.0)

    def step(self, e_new, e_old, h_old, dt, dtemp: float = 0.0):
        dmg = self.damage
        e = np.asarray(e_new, dtype=float)
        n = e.shape[0]
        tr = e[:, :3].sum(axis=1)
        if dmg.split == "volumetric":  # the bulk response in compression is kept
            open_ = tr > 0
            dev = (e - tr[:, None] / 3.0 * _M) * _HALF
            plus = 2.0 * self.mu * dev + (self.k * np.where(open_, tr, 0.0))[:, None] * _M
            minus = (self.k * np.where(open_, 0.0, tr))[:, None] * _M
            dev_matrix = isotropic_matrix(-2.0 * self.mu / 3.0, self.mu)
            bulk = self.k * np.outer(_M, _M)
            c_plus = dev_matrix[None] + np.where(open_, 1.0, 0.0)[:, None, None] * bulk
            c_minus = np.where(open_, 0.0, 1.0)[:, None, None] * bulk
        elif dmg.split == "spectral":  # the negative principal strains keep their stiffness
            plus, c_plus, minus, c_minus = _principal_split(e, tr, self.k, self.mu)
        else:
            plus = e @ self.d
            minus = 0.0
            c_plus = np.broadcast_to(self.d, (n, 6, 6))
            c_minus = 0.0
        eq, grad = self._equivalent(e, tr, plus)
        kappa_old = h_old[:, 0]
        loading = eq > kappa_old
        kappa = np.where(loading, eq, kappa_old)
        d, slope = dmg.damage(kappa, self.young)
        rate = 1.0
        if dmg.viscosity > 0.0 and dt > 0.0:
            ratio = dt / dmg.viscosity
            rate = ratio / (1.0 + ratio)
            d = (h_old[:, 1] + ratio * d) / (1.0 + ratio)
        elif dmg.viscosity > 0.0:  # an instantaneous step: no growth of the damage
            rate = 0.0
            d = h_old[:, 1].copy()
        d = np.maximum(d, h_old[:, 1])
        stress = (1.0 - d)[:, None] * plus + minus
        tangent = (1.0 - d)[:, None, None] * c_plus + c_minus
        growth = np.where(loading, rate * slope, 0.0)
        tangent = tangent - growth[:, None, None] * _outer(plus, grad)
        return stress, tangent, np.stack([kappa, d], axis=1)


def _principal_strains(e):
    """The principal strains (n, 3) and axes (n, 3, 3) of engineering strains (n, 6)."""
    t = np.empty((e.shape[0], 3, 3))
    for k, (a, b) in enumerate(_PAIRS):
        t[:, a, b] = t[:, b, a] = e[:, k] * (1.0 if a == b else 0.5)
    return np.linalg.eigh(t)


def _principal_split(e, tr, bulk: float, shear: float):
    """The stresses of the positive and of the negative principal strains of an isotropic
    elastic point (Miehe et al. 2010), ``sigma± = lambda <tr eps>± I + 2 mu eps±``, and
    their tangents (see :func:`_spectral`)."""
    lam = bulk - 2.0 * shear / 3.0
    w, v = _principal_strains(e)
    out = []
    for sign in (1.0, -1.0):
        on = (sign * w > 0).astype(float)  # the principal strains of this sign
        whole = (sign * tr > 0).astype(float)
        s_a = (lam * whole * tr)[:, None] + 2.0 * shear * on * w
        d = (lam * whole)[:, None, None] * np.ones((1, 3, 3)) + 2.0 * shear * (
            on[:, :, None] * np.eye(3)[None]
        )
        out.extend(_spectral(v, 1.0 + 2.0 * w, s_a, d))
    return tuple(out)


# -- concrete: the law of Mazars -------------------------------------------------------------


@dataclass(frozen=True)
class Mazars:
    """Mazars' law of the damage of concrete (see the module): the ``threshold`` strain
    ``kappa_0`` (about 1e-4), the parameters ``tension`` ``(A_t, B_t)`` and ``compression``
    ``(A_c, B_c)`` of its two damages (0.7 to 1.2 and 1e4 to 5e4; 1 to 1.5 and 1e3 to 2e3
    for usual concretes), the exponent ``beta`` of their shares (1; 1.06 lessens the
    damage in shear) and the ``residual`` fraction of the stiffness that the damage
    leaves."""

    threshold: float
    tension: tuple[float, float]
    compression: tuple[float, float]
    beta: float = 1.0
    residual: float = 1e-6

    def __post_init__(self) -> None:
        if not (math.isfinite(self.threshold) and self.threshold > 0):
            raise ValueError("damage.threshold: the strain at which the damage starts, positive")
        object.__setattr__(self, "threshold", float(self.threshold))
        for name in ("tension", "compression"):
            pair = getattr(self, name)
            try:
                a, b = (float(x) for x in pair)
            except (TypeError, ValueError):
                raise ValueError(f"damage.{name}: the pair (A, B) of Mazars' law") from None
            if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
                raise ValueError(f"damage.{name}: A and B positive")
            object.__setattr__(self, name, (a, b))
        if not (math.isfinite(self.beta) and self.beta >= 1.0):
            raise ValueError("damage.beta: the exponent of the shares, 1 or more")
        if not 0.0 < self.residual < 1.0:
            raise ValueError("damage.residual: a fraction of the stiffness in (0, 1)")
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(self, "residual", float(self.residual))

    def to_config(self) -> dict:
        out = {"law": "mazars", "threshold": self.threshold, "tension": list(self.tension),
               "compression": list(self.compression)}  # fmt: skip
        if self.beta != 1.0:
            out["beta"] = self.beta
        if self.residual != 1e-6:
            out["residual"] = self.residual
        return out

    def strains(self, young: float) -> tuple[float, float]:
        """The threshold of the damage (and no ultimate strain: the laws decay)."""
        return self.threshold, math.inf

    def branch(self, kappa, pair):
        """A damage of Mazars ``1 - kappa_0 (1 - A) / kappa - A exp(-B (kappa - kappa_0))``
        and its derivative, for strains ``kappa`` beyond the threshold."""
        a, b = pair
        k0 = self.threshold
        k = np.maximum(kappa, k0)
        decay = np.exp(-b * (k - k0))
        d = 1.0 - k0 * (1.0 - a) / k - a * decay
        return np.clip(d, 0.0, 1.0), k0 * (1.0 - a) / (k * k) + a * b * decay

    def uniaxial(self, strain, young: float, poisson: float):
        """The stress of a monotonic uniaxial tension (positive strains) or compression
        (negative ones), in closed form."""
        e = np.asarray(strain, dtype=float)
        kappa = np.where(e >= 0, e, -math.sqrt(2.0) * poisson * e)
        d_t, _ = self.branch(kappa, self.tension)
        d_c, _ = self.branch(kappa, self.compression)
        d = np.where(kappa > self.threshold, np.where(e >= 0, d_t, d_c), 0.0)
        return (1.0 - np.minimum(d, 1.0 - self.residual)) * young * e


class _MazarsLaw:
    """The integration of :class:`Mazars` in small strains. State of a point: the largest
    equivalent strain reached, and the damage."""

    kind = "damage"
    thermal = False
    symmetric = False
    n_state = 2

    def __init__(self, damage: Mazars, young: float, poisson: float):
        self.damage = damage
        self.young, self.poisson = float(young), float(poisson)
        self.k = young / (3.0 * (1.0 - 2.0 * poisson))
        self.mu = young / (2.0 * (1.0 + poisson))
        self.d = isotropic_matrix(self.k - 2.0 * self.mu / 3.0, self.mu)

    def scalar(self, h) -> np.ndarray:
        """The damage of the points (the variable ``p`` of the outputs)."""
        return h[:, 1]

    def _shares(self, w):
        """The share ``alpha_t`` of the positive principal strains ``w`` (n, 3) that the
        tensions of the effective stress make, and its derivatives by the principal
        strains (n, 3): ``alpha_t = sum_i eps^t_i <eps_i>+ / eps_eq^2`` with ``eps^t`` the
        strains of the positive principal stresses."""
        lam = self.k - 2.0 * self.mu / 3.0
        young, nu = self.young, self.poisson
        tr = w.sum(axis=1)
        s = lam * tr[:, None] + 2.0 * self.mu * w  # the principal effective stresses
        sp = np.maximum(s, 0.0)
        hs = (s > 0).astype(float)
        et = (1.0 + nu) / young * sp - nu / young * sp.sum(axis=1)[:, None]
        pos = np.maximum(w, 0.0)
        hp = (w > 0).astype(float)
        eq2 = np.einsum("na,na->n", pos, pos)
        safe = np.where(eq2 > 0, eq2, 1.0)
        top = np.einsum("na,na->n", et, pos)
        alpha = np.where(eq2 > 0, top / safe, 0.0)
        # d et_i / d w_j = (1 + nu)/E hs_i ds_i/dw_j - nu/E sum_k hs_k ds_k/dw_j
        ds = lam * np.ones((1, 3, 3)) + 2.0 * self.mu * np.eye(3)[None]  # ds_i / dw_j
        hds = hs[:, :, None] * ds
        det = (1.0 + nu) / young * hds - nu / young * hds.sum(axis=1)[:, None, :]
        dtop = np.einsum("nij,ni->nj", det, pos) + et * hp
        dalpha = (dtop - 2.0 * alpha[:, None] * pos) / safe[:, None]
        alpha = np.clip(alpha, 0.0, 1.0)
        inside = ((alpha > 0.0) & (alpha < 1.0) & (eq2 > 0))[:, None]
        return alpha, np.where(inside, dalpha, 0.0)

    def step(self, e_new, e_old, h_old, dt, dtemp: float = 0.0):
        dmg = self.damage
        e = np.asarray(e_new, dtype=float)
        w, v = _principal_strains(e)
        pos = np.maximum(w, 0.0)
        eq = np.sqrt(np.einsum("na,na->n", pos, pos))
        m = v[:, _I0, :] * v[:, _I1, :]  # N_a (x) N_a in Voigt components
        kappa_old = np.maximum(h_old[:, 0], dmg.threshold)
        loading = eq > kappa_old
        kappa = np.where(loading, eq, kappa_old)
        alpha, dalpha = self._shares(w)
        d_t, slope_t = dmg.branch(kappa, dmg.tension)
        d_c, slope_c = dmg.branch(kappa, dmg.compression)
        beta = dmg.beta
        a_t, a_c = alpha**beta, (1.0 - alpha) ** beta
        d = a_t * d_t + a_c * d_c
        top = 1.0 - dmg.residual
        d = np.where(kappa > dmg.threshold, np.minimum(d, top), 0.0)
        growing = (d > h_old[:, 1]) & (d < top)
        d = np.maximum(d, h_old[:, 1])
        effective = e @ self.d
        stress = (1.0 - d)[:, None] * effective
        tangent = (1.0 - d)[:, None, None] * self.d[None]
        # the derivative of the damage by the principal strains: by kappa (while it
        # grows) and by the shares of the tension and of the compression
        safe = np.where(eq > 0, eq, 1.0)
        dkappa = np.where(loading[:, None], pos / safe[:, None], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            da_t = np.where(alpha > 0, beta * alpha ** (beta - 1.0), 0.0)
            da_c = np.where(alpha < 1, beta * (1.0 - alpha) ** (beta - 1.0), 0.0)
        dd = (a_t * slope_t + a_c * slope_c)[:, None] * dkappa + (da_t * d_t - da_c * d_c)[
            :, None
        ] * dalpha
        grad = np.einsum("nka,na->nk", m, np.where(growing[:, None], dd, 0.0))
        tangent = tangent - _outer(effective, grad)
        return stress, tangent, np.stack([kappa, d], axis=1)


# -- damage coupled to plasticity ----------------------------------------------------------


@dataclass(frozen=True)
class DuctileDamage:
    """A damage coupled to the von Mises plasticity of its material (see the module).

    ``law="lemaitre"``: the strength ``S`` of the damage (a stress: of the order of the
    energy release rate at failure) and its exponent ``s`` (1 by default), the cumulated
    plastic strain ``threshold`` at which it starts (0), and the ``closure`` ``h`` of the
    microcracks in compression (1: none; 0.2 for metals).

    ``law="kachanov"``: the resistance ``A`` (a stress), the exponents ``r`` of the
    stress and ``k`` of ``1 - D`` (by default ``r``), and Hayhurst's weights ``alpha`` of
    the largest principal stress and ``beta`` of the mean stress (0: von Mises' stress).

    ``critical``: the damage at which a point breaks (by default it grows up to ``1 -
    residual``); ``residual``: the fraction of its stress that a broken point keeps;
    ``viscosity``: the time of the delayed damage (0: none)."""

    law: str
    S: float | None = None
    s: float = 1.0
    threshold: float = 0.0
    closure: float = 1.0
    A: float | None = None
    r: float | None = None
    k: float | None = None
    alpha: float = 0.0
    beta: float = 0.0
    critical: float | None = None
    residual: float = 1e-6
    viscosity: float = 0.0

    def __post_init__(self) -> None:
        if self.law not in DUCTILE_LAWS:
            raise ValueError(f"damage.law: one of {DUCTILE_LAWS}")

        def positive(name, value):
            if value is None or not (math.isfinite(value) and value > 0):
                raise ValueError(f"damage.{name}: positive")
            return float(value)

        own = {"lemaitre": ("S", "s", "threshold", "closure"),
               "kachanov": ("A", "r", "k", "alpha", "beta")}  # fmt: skip
        defaults = {"S": None, "s": 1.0, "threshold": 0.0, "closure": 1.0, "A": None,
                    "r": None, "k": None, "alpha": 0.0, "beta": 0.0}  # fmt: skip
        for other in DUCTILE_LAWS:
            if other != self.law:
                given = [n for n in own[other] if getattr(self, n) != defaults[n]]
                if given:
                    raise ValueError(f"damage: {given} are parameters of the law {other}")
        if self.law == "lemaitre":
            object.__setattr__(self, "S", positive("S", self.S))
            object.__setattr__(self, "s", positive("s", self.s))
            if not (math.isfinite(self.threshold) and self.threshold >= 0):
                raise ValueError("damage.threshold: a cumulated plastic strain, 0 or more")
            if not 0.0 <= self.closure <= 1.0:
                raise ValueError("damage.closure: the share of the compressions, in [0, 1]")
            object.__setattr__(self, "threshold", float(self.threshold))
            object.__setattr__(self, "closure", float(self.closure))
        else:
            object.__setattr__(self, "A", positive("A", self.A))
            object.__setattr__(self, "r", positive("r", self.r))
            k = self.r if self.k is None else self.k
            if not (math.isfinite(k) and k >= 0):
                raise ValueError("damage.k: the exponent of 1 - D, 0 or more")
            object.__setattr__(self, "k", float(k))
            if not (self.alpha >= 0 and self.beta >= 0 and self.alpha + self.beta <= 1.0):
                raise ValueError("damage: Hayhurst's weights alpha and beta, in [0, 1], add "
                                 "up to 1 at most")  # fmt: skip
            object.__setattr__(self, "alpha", float(self.alpha))
            object.__setattr__(self, "beta", float(self.beta))
        if not 0.0 < self.residual < 1.0:
            raise ValueError("damage.residual: a fraction of the stress in (0, 1)")
        object.__setattr__(self, "residual", float(self.residual))
        if not (math.isfinite(self.viscosity) and self.viscosity >= 0.0):
            raise ValueError("damage.viscosity: non-negative")
        object.__setattr__(self, "viscosity", float(self.viscosity))
        if self.critical is not None:
            if not 0.0 < self.critical < 1.0:
                raise ValueError("damage.critical: the damage of a broken point, in (0, 1)")
            object.__setattr__(self, "critical", float(self.critical))

    def to_config(self) -> dict:
        out: dict = {"law": self.law}
        if self.law == "lemaitre":
            out.update(S=self.S)
            for name, default in (("s", 1.0), ("threshold", 0.0), ("closure", 1.0)):
                if getattr(self, name) != default:
                    out[name] = getattr(self, name)
        else:
            out.update(A=self.A, r=self.r)
            if self.k != self.r:
                out["k"] = self.k
            for name in ("alpha", "beta"):
                if getattr(self, name):
                    out[name] = getattr(self, name)
        if self.critical is not None:
            out["critical"] = self.critical
        if self.residual != 1e-6:
            out["residual"] = self.residual
        if self.viscosity:
            out["viscosity"] = self.viscosity
        return out

    @property
    def top(self) -> float:
        """The largest damage of a point that is not broken."""
        return 1.0 - self.residual if self.critical is None else self.critical

    def rupture_time(self, stress: float) -> float:
        """Kachanov's time to rupture under a constant uniaxial stress, ``1 / ((k + 1)
        (stress / A)^r)`` (of the damage 1; of ``critical``: ``1 - (1 - critical)^(k + 1)``
        of it)."""
        if self.law != "kachanov":
            raise ValueError("rupture_time: of the law of Kachanov")
        whole = 1.0 / ((self.k + 1.0) * (stress / self.A) ** self.r)
        if self.critical is None:
            return whole
        return whole * (1.0 - (1.0 - self.critical) ** (self.k + 1.0))

    def release_rate(self, stress, young: float, poisson: float):
        """The energy release rate ``Y`` of effective stresses (n, 6) and its derivative
        by them (n, 6, as strains: its contraction with a stress increment is ``dY``):
        ``(1 + nu) / (2 E) sum_i g(s_i) - nu / (2 E) g(tr s)`` with ``g(x) = <x>+^2 + h
        <x>-^2`` of the principal stresses."""
        s = np.asarray(stress, dtype=float)
        t = np.empty((s.shape[0], 3, 3))
        for k, (a, b) in enumerate(_PAIRS):
            t[:, a, b] = t[:, b, a] = s[:, k]
        w, v = np.linalg.eigh(t)
        h = self.closure
        tr = w.sum(axis=1)

        def g(x):
            return np.where(x > 0, x * x, h * x * x), np.where(x > 0, 2.0 * x, 2.0 * h * x)

        gw, dgw = g(w)
        gt, dgt = g(tr)
        a, b = (1.0 + poisson) / (2.0 * young), poisson / (2.0 * young)
        y = a * gw.sum(axis=1) - b * gt
        dy = a * dgw - (b * dgt)[:, None]  # by the principal stresses
        m = v[:, _I0, :] * v[:, _I1, :]
        grad = np.einsum("nka,na->nk", m, dy)
        grad[:, 3:] *= 2.0  # strain-like: the shears count twice in the contraction
        return np.maximum(y, 0.0), grad

    def creep_stress(self, stress):
        """Hayhurst's stress ``alpha s_I + 3 beta s_m + (1 - alpha - beta) s_eq`` of
        stresses (n, 6), its positive part, and its derivative (n, 6, strain-like)."""
        s = np.asarray(stress, dtype=float)
        n = s.shape[0]
        mean = s[:, :3].mean(axis=1)
        dev = s - mean[:, None] * _M
        j2 = 0.5 * (np.einsum("ni,ni->n", dev[:, :3], dev[:, :3])
                    + 2.0 * np.einsum("ni,ni->n", dev[:, 3:], dev[:, 3:]))  # fmt: skip
        seq = np.sqrt(3.0 * j2)
        safe = np.where(seq > 0, seq, 1.0)
        dseq = 1.5 * dev / safe[:, None]
        dseq[:, 3:] *= 2.0
        chi = 3.0 * self.beta * mean + (1.0 - self.alpha - self.beta) * seq
        grad = self.beta * _M[None] + (1.0 - self.alpha - self.beta) * np.where(
            (seq > 0)[:, None], dseq, 0.0
        )
        if self.alpha > 0:
            t = np.empty((n, 3, 3))
            for k, (a, b) in enumerate(_PAIRS):
                t[:, a, b] = t[:, b, a] = s[:, k]
            w, v = np.linalg.eigh(t)
            chi = chi + self.alpha * w[:, 2]
            first = v[:, _I0, 2] * v[:, _I1, 2]
            first[:, 3:] *= 2.0
            grad = grad + self.alpha * first
        on = chi > 0
        return np.where(on, chi, 0.0), np.where(on[:, None], grad, 0.0)


class _PlasticDamage:
    """The integration of a :class:`DuctileDamage` with the plasticity of its material: the
    kernel of von Mises, or of Hill's criterion, on the effective stress, then the damage
    (see the module). State of a point: that of the plasticity (plastic strain, ``p``,
    backstresses), the damage of the law, and the (delayed) damage that degrades the
    stress."""

    kind = "damage"
    thermal = False
    symmetric = False

    def __init__(self, plasticity, damage: DuctileDamage, young: float, poisson: float):
        from morphogen.homogenization import behaviours as beh

        if plasticity is None:
            raise ValueError(
                f"damage: the law {damage.law} is that of a plastic material (plasticity)"
            )
        if damage.law == "kachanov" and not plasticity.viscous:
            raise ValueError("damage: Kachanov's law is that of a viscoplastic material "
                             "(plasticity.norton)")  # fmt: skip
        self.damage = damage
        self.young, self.poisson = float(young), float(poisson)
        bulk = young / (3.0 * (1.0 - 2.0 * poisson))
        shear = young / (2.0 * (1.0 + poisson))
        lam = bulk - 2.0 * shear / 3.0
        self.nk = len(plasticity.kinematic)
        self.width = beh.state_width(self.nk)
        self.n_state = self.width + 2
        self._p = beh.STATE_P
        elastic = isotropic_matrix(lam, shear)
        self._data = (
            np.ascontiguousarray(elastic[None]),
            np.zeros((1, 6)),
            np.ascontiguousarray(plasticity.params(lam, shear)[None]),
            int(self.nk),
            np.zeros((1, 1, 6)),
            np.zeros((1, 1, 7)),
            np.zeros((1, 1, 1)),
            np.zeros(1, dtype=np.int64),
        )
        self._update = beh.update_points
        self._plastic = beh.PLASTIC if plasticity.hill is None else beh.HILL
        # the cumulated plastic strain of a plastic strain increment: the norm dual to the
        # criterion, dp² = dep . a⁺ . dep with J² = x . a . x (a = 3/2 P diag(h) P in
        # Mandel's components; 2/3 dep : dep for von Mises), here for tensor components
        h = np.ones(6)
        if plasticity.hill is not None:
            for name, value in zip(beh.HILL_COMPONENTS, plasticity.hill, strict=True):
                h[beh._HILL_VOIGT[name]] = value
        deviatoric = np.eye(3) - 1.0 / 3.0
        criterion = np.zeros((6, 6))
        criterion[:3, :3] = 1.5 * deviatoric @ np.diag(h[:3]) @ deviatoric
        criterion[3:, 3:] = 1.5 * np.diag(h[3:])
        mandel = np.diag([1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0)])
        self._dual = mandel @ np.linalg.pinv(criterion) @ mandel
        # the tensor strains of a stress (the compliance), and of engineering strains
        self._compliance = (1.0 + poisson) / young * np.eye(6) - poisson / young * np.outer(_M, _M)

    def scalar(self, h) -> np.ndarray:
        """The damage of the points (the variable ``p`` of the outputs)."""
        return h[:, -1]

    def step(self, e_new, e_old, h_old, dt, dtemp: float = 0.0):
        dmg = self.damage
        e = np.ascontiguousarray(e_new, dtype=float)
        n = e.shape[0]
        old = np.ascontiguousarray(h_old[:, : self.width])
        new = np.zeros_like(old)
        eff = np.zeros((n, 6))
        c_eff = np.zeros((n, 6, 6))
        kind = np.full(n, self._plastic, dtype=np.int64)
        zero = np.zeros(n, dtype=np.int64)
        failed = self._update(kind, zero, zero, self._data, e, 0.0, float(dt), old, new, eff,
                              c_eff, 0, 1.0, False)  # fmt: skip
        if failed:
            raise FloatingPointError("damage: the plasticity of the effective stress failed")
        law_old, d_old = h_old[:, -2], h_old[:, -1]
        full = 1.0 - dmg.residual
        broken = law_old >= full  # no more to lose
        if dmg.law == "lemaitre":
            grow, d_grow = self._lemaitre(e, old, new, eff, c_eff)
        else:
            grow, d_grow = self._kachanov(dt, law_old, eff, c_eff)
        law = np.where(broken, law_old, law_old + grow)
        breaks = ~broken & (law >= dmg.top)  # at its critical damage the point breaks
        law = np.where(breaks, full, law)
        active = ~broken & ~breaks
        rate = 1.0
        if dmg.viscosity > 0.0:  # the damage of the stress follows that of the law
            ratio = dt / dmg.viscosity if dt > 0.0 else 0.0
            rate = ratio / (1.0 + ratio)
        d = np.maximum(d_old + rate * (law - d_old), d_old)
        stress = (1.0 - d)[:, None] * eff
        tangent = (1.0 - d)[:, None, None] * c_eff
        tangent = tangent - _outer(eff, np.where(active[:, None], rate * d_grow, 0.0))
        return stress, tangent, np.column_stack([new, law, d])

    def _lemaitre(self, e, old, new, eff, c_eff):
        """The growth of Lemaitre's damage over a step, ``(Y / S)^s dp`` beyond the
        threshold, and its derivative by the strains (n, 6)."""
        dmg = self.damage
        p_old, p_new = old[:, self._p], new[:, self._p]
        dp = np.maximum(p_new - np.maximum(p_old, dmg.threshold), 0.0)
        y, dy = dmg.release_rate(eff, self.young, self.poisson)
        rate = (y / dmg.S) ** dmg.s
        grow = rate * dp
        # dp = sqrt(dep . a⁺ . dep) (sqrt(2/3 dep : dep) for von Mises), dep = eps - C^-1
        # sigma~ - ep_old (tensor components)
        dep = new[:, :6] - old[:, :6]
        whole = np.maximum(p_new - p_old, 0.0)
        safe = np.where(whole > 0, whole, 1.0)
        contract = dep @ self._dual / safe[:, None]
        ddep = np.diag(_HALF)[None] - np.einsum("ij,njk->nik", self._compliance, c_eff)
        ddp = np.einsum("ni,nik->nk", contract, ddep)
        ddp = np.where(((dp > 0) & (whole > 0))[:, None], ddp, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            drate = np.where(y > 0, dmg.s * rate / np.where(y > 0, y, 1.0), 0.0)
        dy_de = np.einsum("ni,nik->nk", dy, c_eff)
        return grow, (drate * dp)[:, None] * dy_de + rate[:, None] * ddp

    def _kachanov(self, dt, d_old, eff, c_eff):
        """The growth of Kachanov's damage over a step: the law integrated exactly for the
        stress of the end of the step, ``(1 - D)^q = (1 - D_old)^q - q (chi / A)^r dt``
        with ``q = k - r + 1`` (the stress ``chi`` is that of ``(1 - D) sigma~``), and its
        derivative by the strains (n, 6)."""
        dmg = self.damage
        chi, dchi = dmg.creep_stress(eff)
        q = dmg.k - dmg.r + 1.0
        rate = (chi / dmg.A) ** dmg.r
        left = 1.0 - np.minimum(d_old, 1.0 - 1e-12)
        if abs(q) > 1e-12:
            power = left**q - q * rate * dt
            done = power <= 0.0 if q > 0 else np.zeros_like(power, dtype=bool)
            new_left = np.where(done, 0.0, np.abs(np.where(done, 1.0, power)) ** (1.0 / q))
        else:  # q = 0: ln(1 - D) falls at the rate (chi / A)^r
            done = np.zeros(len(left), dtype=bool)
            new_left = left * np.exp(-rate * dt)
        grow = np.maximum(left - new_left, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            safe = np.where(chi > 0, chi, 1.0)
            slope = np.where(
                (chi > 0) & ~done & (new_left > 0),
                dt * dmg.r * rate / safe * np.where(new_left > 0, new_left, 1.0) ** (1.0 - q),
                0.0,
            )
        return grow, slope[:, None] * np.einsum("ni,nik->nk", dchi, c_eff)


def damage_from_config(spec):
    """The damage law of a mapping of a configuration: Mazars' law of concrete (``law:
    mazars``), a damage coupled to plasticity (``law: lemaitre`` or ``kachanov``), or the
    isotropic damage of quasi-brittle solids (:class:`Damage`, without ``law``)."""
    if isinstance(spec, (Damage, Mazars, DuctileDamage)):
        return spec
    if not isinstance(spec, dict):
        raise ValueError("damage: a mapping of parameters")
    law = spec.get("law")
    if law is None:
        return Damage.from_config(spec)
    values = {k: v for k, v in spec.items() if k != "law"}
    if law == "mazars":
        names = set(Mazars.__dataclass_fields__)
        missing = sorted({"threshold", "tension", "compression"} - set(values))
        if missing:
            raise ValueError(f"damage: Mazars' law needs {missing}")
        cls, kwargs = Mazars, values
    elif law in DUCTILE_LAWS:
        names = set(DuctileDamage.__dataclass_fields__) - {"law"}
        cls, kwargs = DuctileDamage, {"law": law, **values}
    else:
        raise ValueError(f"damage.law: mazars, or one of {DUCTILE_LAWS} with a plasticity")
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"damage: unknown keys {unknown}; allowed: {sorted(names)}")
    return cls(**kwargs)


# -- the laws in the solvers -----------------------------------------------------------------


class _Logarithmic:
    """A law of small strains in the logarithmic strain space of finite strains: it sees
    ``H = 1/2 ln C`` and gives the stress ``T`` conjugate to it, pulled back to the second
    Piola-Kirchhoff stress ``S = J^T T`` and its tangent (:mod:`morphogen.learning.
    logarithmic`), as the Hencky points of morphoGen (Miehe, Apel & Lambrecht 2002)."""

    def __init__(self, law):
        self.law = law
        self.kind, self.symmetric, self.n_state = law.kind, law.symmetric, law.n_state
        self.thermal = law.thermal
        if hasattr(law, "scalar"):
            self.scalar = law.scalar

    def step(self, e_new, e_old, h_old, dt, dtemp: float = 0.0):
        from morphogen.learning.logarithmic import _measures, _pullback

        e_new = np.ascontiguousarray(e_new, dtype=float)
        h = _measures(e_new, 1.0)[0]
        h0 = _measures(np.ascontiguousarray(e_old, dtype=float), 1.0)[0]
        with np.errstate(invalid="ignore"):
            t, k, state = self.law.step(h, h0, h_old, dt, dtemp)
        s, c = _pullback(e_new, np.ascontiguousarray(t), np.ascontiguousarray(k), 1.0,
                         bool(self.symmetric))  # fmt: skip
        return s, c, state


@dataclass(frozen=True)
class _Spec:
    """What the points of the solvers ask of the law of a material (as a learned law):
    the laws of this module are isotropic, their axes those of the sample."""

    oriented: bool = False


def analytic(material) -> bool:
    """Whether a material has a behaviour of this module."""
    return (
        material.viscoelasticity is not None
        or material.hyperelasticity is not None
        or material.damage is not None
    )


def analytic_model(material, *, finite: bool):
    """The law of the points of a material with a ``viscoelasticity``, a
    ``hyperelasticity`` or a ``damage``, in small strains or (``finite``) for the
    Green-Lagrange strain and the second Piola-Kirchhoff stress: a step ``(strain at the
    end, strain and state at the start, time step, change of temperature) -> (stress,
    tangent, state)`` of all the points at once."""
    bulk, shear = material.moduli()
    young, poisson = material.young, material.poisson
    lam = bulk - 2.0 * shear / 3.0
    visco, hyper, damage = material.viscoelasticity, material.hyperelasticity, material.damage
    if hyper is not None:
        if finite:
            return _Hyperelastic(hyper, bulk, shear, lam, visco)
        return _Maxwell(visco, bulk, shear)  # its elasticity at the natural state
    if visco is not None:
        law = _Maxwell(visco, bulk, shear)
    elif isinstance(damage, DuctileDamage):
        law = _PlasticDamage(material.plasticity, damage, young, poisson)
    elif isinstance(damage, Mazars):
        law = _MazarsLaw(damage, young, poisson)
    else:
        law = _DamageLaw(damage, young, poisson)
    return _Logarithmic(law) if finite else law


def analytic_law(material, *, finite: bool, expansion=None):
    """The law of the points of a material of this module for the solvers (the seam of
    the learned laws, :class:`~morphogen.learning.behaviour.LearnedLaw`); ``expansion``:
    its thermal expansion tensor, the thermal strain taken out of the strain it sees."""
    from morphogen.learning.behaviour import LearnedLaw

    if expansion is not None:
        expansion = np.asarray(expansion, dtype=float).reshape(3, 3)
        expansion = 0.5 * (expansion + expansion.T) if np.any(expansion) else None
    return LearnedLaw(_Spec(), analytic_model(material, finite=finite), bool(finite), expansion)
