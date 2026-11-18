"""Porous elasto-(visco)plasticity: the ``porous_plastic`` behaviours of Z-mat (Besson,
Cailletaud, Chaboche & Forest, *Non-linear mechanics of materials*, 2010, chap. 7), for the
ductile damage of metals (Gurson-Tvergaard-Needleman, Rousselier) and the compaction of
powders and foams (the elliptic criterion of Abouaf, Green and Shima & Oyane).

A porous material has a matrix, the "real material", whose flow stress ``σ*`` (the matrix
stress) is defined from the macroscopic stress ``σ`` and the porosity ``f`` by a potential
``φ(σ, σ*, f) = 0``, an isotropic function of the mean stress ``Σm = tr σ / 3`` and of the
von Mises stress ``Σeq = sqrt(3/2 s : s)``:

* Gurson-Tvergaard-Needleman: ``(Σeq / σ*)² + 2 q1 f* cosh(3 q2 Σm / (2 σ*)) - 1 - q3 f*² =
  0`` (Gurson 1977; Tvergaard 1981; Tvergaard & Needleman 1984), the effective porosity
  ``f*`` a piecewise linear function of ``f`` (``f`` itself by default; the coalescence of
  Tvergaard & Needleman, ``f* = fc + (1/q1 - fc) (f - fc) / (fF - fc)`` beyond ``fc``),
  kept below ``0.995 / q1`` (the surface vanishes at ``1 / q1``);
* Rousselier: ``σ* = Σeq / (1 - f) + D σ1 f exp(q Σm / ((1 - f) σ1))`` (Rousselier 1987,
  ``q = 1``; the modified one of Z-mat, ``q2``), which grows the voids in pure shear;
* elliptic: ``σ*² = C(f) Σeq² + F(f) I1²`` (``I1 = 3 Σm``: Abouaf 1985, Abouaf, Chenot,
  Raisson & Bauduin 1988 for the hot compaction of metal powders; Green 1972; Shima &
  Oyane 1976; foams), ``C`` and ``F`` constants or piecewise linear functions of ``f``
  (``C = 1``, ``F = 0`` for the dense material).

At ``f = 0`` the three are von Mises. The overstress ``σ* - R(p)`` drives the flow of the
matrix: rate-independent plasticity ``σ* = R(p)``, or Norton's viscoplasticity ``ṗ = <(σ*
- R) / K>^N``, with the isotropic hardening ``R(p) = R0 + H p + Σ Q_i (1 - exp(-b_i p))``
of the matrix equivalent plastic strain ``p``. The plastic strain is normal to the
potential, ``ε̇p = Λ̇ ∂σ*/∂σ``, and its rate follows from the equivalence of the plastic
work of the matrix and of the porous material, ``(1 - f) σ* ṗ = σ : ε̇p`` (``Λ̇ = (1 - f)
ṗ`` for the potentials homogeneous of degree 1 in ``σ``, GTN and elliptic). The porosity
is the sum of the growth porosity, by the conservation of the mass of the incompressible
matrix, ``ḟg = (1 - f) tr ε̇p``, and of the nucleated one, ``ḟn = A(p) ṗ``: a constant
``A``, Chu & Needleman's (1980) Gaussian ``A = fN / (sN sqrt(2π)) exp(-((p - εN) / sN)² /
2)`` or an exponential ``A = B exp(-p / ε0)``, none below the plastic strain ``p_n``.

*Kinematic hardening* (Besson & Guillemer-Neel 2003, the extension of the Gurson and Green
models; Besson, Cailletaud, Chaboche & Forest 2010, chap. 7), for the potentials
homogeneous of degree 1 in the stress, GTN and elliptic: the potential is that of the
relative stress ``σ - X``, ``X = Σ X_i``, the Armstrong-Frederick terms ``Ẋ_i = 2/3 C_i
ε̇p - D_i X_i ṗ`` (``D_i = 0``: Prager) of the plastic strain normal to the potential at ``σ
- X``, and the work equivalence is that of the relative stress, ``(1 - f) σ* ṗ = (σ - X) :
ε̇p`` (``Λ̇ = (1 - f) ṗ``). The plastic strain has the volume change of the growth of the
voids: the backstresses have a hydrostatic part, ``tr Ẋ_i = 2/3 C_i tr ε̇p - D_i tr X_i ṗ``,
and the mean stress of the potential is ``Σm - tr X / 3``: the voids grown in hydrostatic
tension move the surface towards the tensile mean stresses (a hydrostatic Bauschinger
effect: the reversed, compressive mean stress yields earlier), as the deviatoric part moves
it along the deviator like that of von Mises points. At ``f = 0`` without nucleation the
model is von Mises's with the same kinematic terms. Rousselier's potential, whose
exponential term has the constant stress ``σ1`` (not homogeneous), takes none.

*Integration* (backward Euler, Aravas 1987): the trial stress ``σ_tr`` of the elastic
predictor (isotropic elasticity, bulk modulus ``K_b``, shear modulus ``μ``) is corrected
along the invariants, ``Σm = Σm_tr - K_b Δεv`` and ``Σeq = Σeq_tr - 3 μ Δεq`` (the
deviatoric direction ``n = 3/2 s_tr / Σeq_tr`` of the trial stress, isotropic potentials),
with ``Δεp = Δεv / 3 I + Δεq n``. Four unknowns, ``Δεv``, ``Δεq``, ``Δp`` (or the viscous
stress over ``K``, better conditioned near no flow) and ``f``, solve four equations: the
normality ``Δεv ∂σ*/∂Σeq - Δεq ∂σ*/∂Σm = 0``, the flow rule, the equivalence of the plastic
works and the evolution of the porosity, by Newton's method with their analytical
Jacobian (the first and second derivatives of ``σ*``: by the implicit function theorem
on the potential of GTN, in closed form for the others) and a line search, the matrix
stress of GTN by its own scalar Newton iteration (bracketed from that of the current
point at the points of the line search: a few evaluations instead of a dozen).
With backstresses (``X_i = θ_i (X_i,n + 2/3 C_i Δεp)``, ``θ_i = 1 / (1 + D_i Δp)``: the
implicit recovery) the relative stress is the relative trial stress ``σ_tr - Σ θ_i X_i,n``
(its deviator ``ξ*``, von Mises stress ``J*`` and mean stress ``m*``) corrected along its
invariants with the moduli ``3 μ + c`` and ``K_b + 2 c / 9`` (``c = Σ θ_i C_i``): ``Σeq =
J* - (3 μ + c) Δεq``, ``Σm = m* - (K_b + 2 c / 9) Δεv``, its deviator in the direction ``n
= 3/2 ξ* / J*``, which ``Δp`` alone sets: the same four unknowns and equations, whose
invariants depend on ``Δp`` too (``dJ*/dΔp = n : Σ D_i θ_i² X_i,n``, ``dm*/dΔp = Σ D_i
θ_i² tr X_i,n / 3``, ``dc/dΔp = -Σ D_i θ_i² C_i``) in the analytical Jacobian.
At the vertex of Rousselier's surface (``Σeq = 0`` under a hydrostatic stress, where its
deviatoric gradient jumps) the normality gives way to ``Σeq = 0``. The
consistent tangent comes from the sensitivity of the solution to the trial invariants
(``dσ = dΣm I + 2/3 dΣeq n + 2 μ Σeq / Σeq_tr (dε_dev - 2/3 (n : dε_dev) n)``), not
symmetric (the porosity grows with the volumetric flow: 10⁻⁵ to 2·10⁻² of its norm),
symmetrized for the conjugate gradient solvers like that of the von Mises points or kept
whole for GMRES (``sym``); with backstresses
the trial invariant of the deviator is ``J*`` (``dJ* = n : ds_tr``, ``n`` that of ``ξ*``),
the deviator ``s = s_tr - 2 μ Δεq n`` and the direction turns with ``Δp``: ``dn = 3 / (2
J*) (dξ* - 2/3 (n : dξ*) n)``, ``dξ* = ds_tr + Σ D_i θ_i² X_i,n dΔp``.

In finite strains (:mod:`morphogen.homogenization.finite`) the same update works in the
space of the logarithmic strain, where the plastic volume change is ``ln J_p``: the
growth of the porosity is that of the volume of the pores of an incompressible matrix.

The kernels of the CPU have their CUDA port for the FFT solvers on the GPU
(``behaviours.cu``, compared with them point by point).

Internal variables of a point: the plastic strain (6, tensor components), ``p``, the
growth of the porosity from its initial value ``f0``, the nucleated porosity, the
matrix stress (an output), ``f = f0 + state[7] + state[8]``, then the backstresses (6 each,
tensor components: :func:`state_width`).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from morphogen.homogenization.tensors import symmetrize

#: Criteria of the porous potentials
GURSON = 0
ROUSSELIER = 1
ELLIPTIC = 2
CRITERIA = {"gurson": GURSON, "rousselier": ROUSSELIER, "elliptic": ELLIPTIC}
#: Nucleation of porosity
NUCLEATION = {"none": 0, "constant": 1, "gaussian": 2, "exponential": 3}
#: Points of the tables of the coefficients depending on the porosity
MAX_TABLE = 4
MAX_ISOTROPIC = 3
#: Most kinematic terms (C, D) of the matrix, as for the von Mises points
MAX_KINEMATIC = 4
#: Layout of a row of parameters (the first twelve as those of the von Mises points)
P_CRITERION = 12
P_F0 = 13
P_Q1, P_Q2, P_Q3 = 14, 15, 16
P_SIGMA1, P_D, P_QR = 17, 18, 19
P_NUCLEATION = 20  # type, then three coefficients and the threshold of p
P_FSTAR = 25  # count, then (f, f*) pairs
P_C = P_FSTAR + 1 + 2 * MAX_TABLE
P_F = P_C + 1 + 2 * MAX_TABLE
P_CAP = P_F + 1 + 2 * MAX_TABLE
P_NKIN = P_CAP + 1  # the number of kinematic terms, then their (C, D) pairs
P_KIN = P_NKIN + 1
N_POROUS = P_KIN + 2 * MAX_KINEMATIC
#: Internal variables: plastic strain (6), p, growth of the porosity, nucleated porosity,
#: matrix stress, then the backstresses (6 each); the width without them
STATE_P = 6
STATE_FG = 7
STATE_FN = 8
STATE_SSTAR = 9
STATE_BACK = 10
POROUS_WIDTH = STATE_BACK


def state_width(n_kinematic: int = 0) -> int:
    """Number of internal variables of a porous point with ``n_kinematic`` backstresses."""
    return STATE_BACK + 6 * int(n_kinematic)


def _table(value, name: str) -> tuple[tuple[float, float], ...]:
    """A coefficient depending on the porosity: a number, or pairs ``(f, value)`` (a
    piecewise linear function, extrapolated linearly), at most :data:`MAX_TABLE`."""
    if value is None:
        return ()
    if np.ndim(value) == 0:
        return ((0.0, float(value)),)
    pairs = [tuple(map(float, item)) for item in value]
    if not 1 <= len(pairs) <= MAX_TABLE or any(len(p) != 2 for p in pairs):
        raise ValueError(f"{name}: a number or 1 to {MAX_TABLE} pairs (porosity, value)")
    pairs.sort()
    if any(b[0] <= a[0] for a, b in itertools.pairwise(pairs)):
        raise ValueError(f"{name}: increasing porosities")
    return tuple(pairs)


def _pairs(value, name: str, first: str = "Q", second: str = "b") -> tuple:
    from morphogen.homogenization.behaviours import _pairs as pairs

    return pairs(value, name, first, second)


@dataclass(frozen=True)
class PorousPlasticity:
    """Porous elasto-(visco)plasticity of an isotropic elastic material (see the module).

    ``criterion``: ``"gurson"`` (GTN: ``q1``, ``q2``, ``q3``, by default ``q1²``, the
    effective porosity ``effective`` as pairs ``(f, f*)``, or the coalescence ``fc``,
    ``fF``), ``"rousselier"`` (``sigma1``, ``D``, the factor ``q2`` of the mean stress of
    the modified criterion) or ``"elliptic"`` (``C``, ``F``: numbers or pairs ``(f,
    value)``). ``porosity``: the initial porosity ``f0``. The matrix: ``yield_stress`` (R0),
    ``hardening`` (H), ``isotropic`` (pairs Q, b), Norton's ``viscosity`` (K) and
    ``exponent`` (N) or rate-independent plasticity. ``nucleation``: ``None``, or a
    mapping with the ``type`` (``constant``: ``A``; ``gaussian``: ``fN``, ``eN``, ``sN``;
    ``exponential``: ``B``, ``e0``) and the threshold ``yield`` of ``p``. ``kinematic``:
    the Armstrong-Frederick terms (pairs C, D; ``D = 0``: Prager) of the backstresses of
    the criteria ``gurson`` and ``elliptic``.
    """

    criterion: str
    yield_stress: float
    porosity: float = 0.0
    hardening: float = 0.0
    isotropic: tuple[tuple[float, float], ...] = ()
    viscosity: float | None = None
    exponent: float | None = None
    q1: float = 1.5
    q2: float = 1.0
    q3: float | None = None
    effective: tuple = ()
    fc: float | None = None
    fF: float | None = None
    sigma1: float | None = None
    D: float | None = None
    C: tuple | float = 1.0
    F: tuple | float = 0.0
    nucleation: dict | None = None
    kinematic: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        crit = str(self.criterion).lower()
        if crit not in CRITERIA:
            raise ValueError(f"porous plasticity: criterion {self.criterion!r}: {list(CRITERIA)}")
        object.__setattr__(self, "criterion", crit)
        for name in ("yield_stress", "porosity", "hardening"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v >= 0.0):
                raise ValueError(f"porous plasticity: {name} must be non-negative")
            object.__setattr__(self, name, v)
        if self.porosity >= 1.0:
            raise ValueError("porous plasticity: an initial porosity below 1")
        iso = _pairs(self.isotropic, "isotropic")
        if len(iso) > MAX_ISOTROPIC or any(q < 0 or not b > 0 for q, b in iso):
            raise ValueError(f"isotropic: at most {MAX_ISOTROPIC} terms (Q >= 0, b > 0)")
        object.__setattr__(self, "isotropic", iso)
        kin = _pairs(self.kinematic, "kinematic", "C", "D")
        if len(kin) > MAX_KINEMATIC:
            raise ValueError(f"kinematic: at most {MAX_KINEMATIC} terms")
        if not all(math.isfinite(c) and math.isfinite(d) and c >= 0 and d >= 0 for c, d in kin):
            raise ValueError("kinematic: terms (C, D) with C >= 0 and D >= 0")
        if kin and crit == "rousselier":
            raise ValueError(
                "porous plasticity: kinematic hardening with the criteria gurson and elliptic "
                "(Besson & Guillemer-Neel's extension of the potentials homogeneous of degree "
                "1 in the stress; Rousselier's, with the constant sigma1 of its exponential "
                "term, is not)"
            )
        object.__setattr__(self, "kinematic", kin)
        if (self.viscosity is None) != (self.exponent is None):
            raise ValueError("norton: give both K and n (or neither: rate-independent)")
        if self.viscosity is not None:
            k, n = float(self.viscosity), float(self.exponent)
            if not (k > 0 and n >= 1.0 and math.isfinite(k) and math.isfinite(n)):
                raise ValueError("norton: K > 0 and n >= 1")
            object.__setattr__(self, "viscosity", k)
            object.__setattr__(self, "exponent", n)
        if crit == "gurson":
            q1, q2 = float(self.q1), float(self.q2)
            q3 = q1 * q1 if self.q3 is None else float(self.q3)
            if not (q1 > 0 and q2 > 0 and q3 > 0):
                raise ValueError("gurson: q1, q2, q3 > 0")
            object.__setattr__(self, "q1", q1)
            object.__setattr__(self, "q2", q2)
            object.__setattr__(self, "q3", q3)
            effective = _table(self.effective or None, "gurson.effective")
            if self.fc is not None or self.fF is not None:
                if effective:
                    raise ValueError("gurson: give the effective porosity or fc and fF, not both")
                fc, ff = float(self.fc), float(self.fF)
                if not 0.0 < fc < ff:
                    raise ValueError("gurson: 0 < fc < fF")
                effective = ((0.0, 0.0), (fc, fc), (ff, 1.0 / q1))
            object.__setattr__(self, "effective", effective)
        elif crit == "rousselier":
            if self.sigma1 is None or self.D is None:
                raise ValueError("rousselier: sigma1 and D")
            s1, d = float(self.sigma1), float(self.D)
            if not (s1 > 0 and d >= 0):
                raise ValueError("rousselier: sigma1 > 0 and D >= 0")
            object.__setattr__(self, "sigma1", s1)
            object.__setattr__(self, "D", d)
            object.__setattr__(self, "q2", float(self.q2))
        else:
            object.__setattr__(self, "C", _table(self.C, "elliptic.C"))
            object.__setattr__(self, "F", _table(self.F, "elliptic.F"))
            if any(v < 0 for _, v in self.C + self.F):
                raise ValueError("elliptic: C and F >= 0")
        if self.nucleation is not None:
            spec = dict(self.nucleation)
            kind = str(spec.pop("type", "")).lower()
            needed = {"constant": ("A",), "gaussian": ("fN", "eN", "sN"),
                      "exponential": ("B", "e0")}  # fmt: skip
            if kind not in needed:
                raise ValueError(f"nucleation: type {list(needed)}")
            threshold = float(spec.pop("yield", 0.0))
            unknown = sorted(set(spec) - set(needed[kind]))
            if unknown or any(k not in spec for k in needed[kind]):
                raise ValueError(f"nucleation {kind}: the coefficients {list(needed[kind])}")
            values = {k: float(spec[k]) for k in needed[kind]}
            if kind == "gaussian" and not values["sN"] > 0:
                raise ValueError("nucleation gaussian: sN > 0")
            if kind == "exponential" and not values["e0"] > 0:
                raise ValueError("nucleation exponential: e0 > 0")
            object.__setattr__(self, "nucleation", {"type": kind, "yield": threshold, **values})

    @property
    def viscous(self) -> bool:
        return self.viscosity is not None

    def params(self, lam: float, mu: float) -> np.ndarray:
        """Row of parameters of the integration kernels (see the module)."""
        row = np.zeros(N_POROUS)
        row[:4] = (lam, mu, self.yield_stress, self.hardening)
        if self.viscous:
            row[4:6] = (self.viscosity, self.exponent)
        for i, (q, b) in enumerate(self.isotropic):
            row[6 + 2 * i : 8 + 2 * i] = (q, b)
        row[P_CRITERION] = CRITERIA[self.criterion]
        row[P_F0] = self.porosity
        row[P_Q1], row[P_Q2], row[P_Q3] = self.q1, self.q2, self.q3 or 0.0
        if self.criterion == "rousselier":
            row[P_SIGMA1], row[P_D], row[P_QR] = self.sigma1, self.D, self.q2
        if self.nucleation is not None:
            nuc = self.nucleation
            kind = nuc["type"]
            row[P_NUCLEATION] = NUCLEATION[kind]
            names = {"constant": ("A",), "gaussian": ("fN", "eN", "sN"), "exponential": ("B", "e0")}
            for k, name in enumerate(names[kind]):
                row[P_NUCLEATION + 1 + k] = nuc[name]
            row[P_NUCLEATION + 4] = nuc["yield"]
        tables = [(P_FSTAR, self.effective if self.criterion == "gurson" else ())]
        if self.criterion == "elliptic":
            tables += [(P_C, self.C), (P_F, self.F)]
        for at, table in tables:
            row[at] = len(table)
            for k, (x, y) in enumerate(table):
                row[at + 1 + 2 * k : at + 3 + 2 * k] = (x, y)
        row[P_CAP] = 0.995 / self.q1 if self.criterion == "gurson" else 0.999
        row[P_NKIN] = len(self.kinematic)
        for k, (c, d) in enumerate(self.kinematic):
            row[P_KIN + 2 * k : P_KIN + 2 * k + 2] = (c, d)
        return row

    def to_config(self) -> dict:
        out: dict = {"criterion": self.criterion, "yield": self.yield_stress}
        if self.porosity:
            out["porosity"] = self.porosity
        if self.hardening:
            out["hardening"] = self.hardening
        if self.isotropic:
            out["isotropic"] = [{"Q": q, "b": b} for q, b in self.isotropic]
        if self.viscous:
            out["norton"] = {"K": self.viscosity, "n": self.exponent}
        if self.criterion == "gurson":
            out.update({"q1": self.q1, "q2": self.q2, "q3": self.q3})
            if self.effective:
                out["effective"] = [list(p) for p in self.effective]
        elif self.criterion == "rousselier":
            out.update({"sigma1": self.sigma1, "D": self.D, "q2": self.q2})
        else:
            for name in ("C", "F"):
                table = getattr(self, name)
                out[name] = table[0][1] if len(table) == 1 else [list(p) for p in table]
        if self.nucleation is not None:
            out["nucleation"] = dict(self.nucleation)
        if self.kinematic:
            out["kinematic"] = [{"C": c, "D": d} for c, d in self.kinematic]
        return out

    @classmethod
    def from_config(cls, spec) -> PorousPlasticity:
        """``{criterion, yield (R0), porosity, hardening (H), isotropic: [{Q, b}...],
        kinematic: [{C, D}...], norton: {K, n}, q1, q2, q3, effective | fc and fF, sigma1,
        D, C, F, nucleation}``."""
        if isinstance(spec, PorousPlasticity):
            return spec
        if not isinstance(spec, dict):
            raise ValueError("porous_plasticity: a mapping (criterion, yield, porosity...)")
        allowed = {"criterion", "yield", "R0", "porosity", "hardening", "H", "isotropic",
                   "kinematic", "norton", "q1", "q2", "q3", "effective", "fc", "fF", "sigma1",
                   "D", "C", "F", "nucleation"}  # fmt: skip
        unknown = sorted(set(spec) - allowed)
        if unknown:
            raise ValueError(
                f"porous_plasticity: unknown keys {unknown}; allowed: {sorted(allowed)}"
            )
        if ("yield" in spec) == ("R0" in spec):
            raise ValueError("porous_plasticity: the yield stress of the matrix (yield, or R0)")
        if "criterion" not in spec:
            raise ValueError(f"porous_plasticity: the criterion ({list(CRITERIA)})")
        norton = spec.get("norton")
        k = n = None
        if norton is not None:
            if not isinstance(norton, dict) or set(norton) != {"K", "n"}:
                raise ValueError("porous_plasticity.norton: {K: viscosity, n: exponent}")
            k, n = float(norton["K"]), float(norton["n"])
        names = ("q1", "q2", "q3", "effective", "fc", "fF", "sigma1", "D", "C", "F", "nucleation")
        kwargs = {key: spec[key] for key in names if key in spec}
        return cls(
            criterion=spec["criterion"],
            yield_stress=float(spec.get("yield", spec.get("R0"))),
            porosity=float(spec.get("porosity", 0.0)),
            hardening=float(spec.get("hardening", spec.get("H", 0.0))),
            isotropic=spec.get("isotropic") or (),
            viscosity=k,
            exponent=n,
            kinematic=spec.get("kinematic") or (),
            **kwargs,
        )


# -- kernels ------------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _ddot(a, b):
    """a : b of two symmetric tensors given by their Voigt components."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + 2.0 * (a[3] * b[3] + a[4] * b[4] + a[5] * b[5])


@njit(cache=True)
def _hardening(par, p):
    """R(p) = R0 + H p + sum Q (1 - exp(-b p))."""
    r = par[2] + par[3] * p
    for i in range(MAX_ISOTROPIC):
        q = par[6 + 2 * i]
        if q != 0.0:
            r += q * (1.0 - math.exp(-par[7 + 2 * i] * p))
    return r


@njit(cache=True)
def _hardening_slope(par, p):
    """R'(p)."""
    r = par[3]
    for i in range(MAX_ISOTROPIC):
        q = par[6 + 2 * i]
        if q != 0.0:
            b = par[7 + 2 * i]
            r += q * b * math.exp(-b * p)
    return r


@njit(cache=True)
def _table_value(par, at, x, default):
    """The piecewise linear function of the table at ``par[at]`` (linear extrapolation);
    ``default`` without points."""
    n = int(par[at])
    if n == 0:
        return default
    if n == 1:
        return par[at + 2]
    k = 0
    while k < n - 2 and x > par[at + 1 + 2 * (k + 1)]:
        k += 1
    x0 = par[at + 1 + 2 * k]
    y0 = par[at + 2 + 2 * k]
    x1 = par[at + 3 + 2 * k]
    y1 = par[at + 4 + 2 * k]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@njit(cache=True)
def _table_slope(par, at, x, default):
    """The slope of :func:`_table_value` at ``x``; ``default`` without points."""
    n = int(par[at])
    if n == 0:
        return default
    if n == 1:
        return 0.0
    k = 0
    while k < n - 2 and x > par[at + 1 + 2 * (k + 1)]:
        k += 1
    dy = par[at + 4 + 2 * k] - par[at + 2 + 2 * k]
    return dy / (par[at + 3 + 2 * k] - par[at + 1 + 2 * k])


@njit(cache=True)
def _gurson_stress(q1, q2, q3, fs, seq, sm, guess):
    """The matrix stress of the GTN potential and its derivatives by Σeq and Σm: the root
    bracketed from ``guess`` if it is positive (the matrix stress of a nearby point; by
    factors from 1 + 1e-6 up to 2), else from an estimate (by factors of 2), then a
    safeguarded Newton iteration."""
    if fs <= 0.0:
        return seq, 1.0, 0.0  # von Mises
    if seq == 0.0 and sm == 0.0:
        return 0.0, 1.0, 0.0
    # Φ(s) = (seq/s)^2 + 2 q1 fs cosh(1.5 q2 sm/s) - 1 - q3 fs^2, decreasing in s
    c = 1.0 + q3 * fs * fs
    if c - 2.0 * q1 * fs <= 0.0:
        return math.inf, 0.0, 0.0  # the surface has vanished
    if guess > 0.0 and math.isfinite(guess):
        s = guess
        grow = 1.0 + 1e-6
    else:
        s = math.sqrt(seq * seq + (1.5 * q2 * sm) ** 2) / max(1.0 - q1 * fs, 0.05)
        grow = 2.0
    lo, hi = 0.0, 0.0
    for _ in range(200):
        t = 1.5 * q2 * sm / s
        phi = (seq / s) ** 2 + 2.0 * q1 * fs * math.cosh(t) - c
        if phi > 0.0:
            lo = s
            if hi == 0.0:
                s *= grow
                grow = 1.0 + min(8.0 * (grow - 1.0), 1.0)
                continue
        else:
            hi = s
            if lo == 0.0:
                s /= grow
                grow = 1.0 + min(8.0 * (grow - 1.0), 1.0)
                continue
        break
    for _ in range(100):
        t = 1.5 * q2 * sm / s
        phi = (seq / s) ** 2 + 2.0 * q1 * fs * math.cosh(t) - c
        if phi > 0.0:
            lo = s
        else:
            hi = s
        dphi = -2.0 / s * ((seq / s) ** 2 + q1 * fs * t * math.sinh(t))  # dΦ/ds < 0
        step = s - phi / dphi
        if not (lo < step < hi):
            step = 0.5 * (lo + hi)
        if abs(step - s) <= 1e-15 * s:
            s = step
            break
        s = step
    t = 1.5 * q2 * sm / s
    den = (seq / s) ** 2 + q1 * fs * t * math.sinh(t)
    return s, (seq / s) / den, 1.5 * q1 * q2 * fs * math.sinh(t) / den


@njit(cache=True)
def matrix_stress(par, seq, sm, f):
    """The matrix stress ``σ*`` of the potential of a point (see the module) and its
    derivatives by ``Σeq`` and ``Σm``, at the porosity ``f``."""
    return _matrix_stress(par, seq, sm, f, 0.0)


@njit(cache=True)
def _matrix_stress(par, seq, sm, f, guess):
    """:func:`matrix_stress`, Gurson's iteration started from ``guess`` if it is positive."""
    kind = int(par[P_CRITERION])
    if kind == GURSON:
        fs = _table_value(par, P_FSTAR, f, f)
        fs = min(max(fs, 0.0), par[P_CAP])
        return _gurson_stress(par[P_Q1], par[P_Q2], par[P_Q3], fs, seq, sm, guess)
    if kind == ROUSSELIER:
        fe = min(max(f, 0.0), par[P_CAP])
        s1 = par[P_SIGMA1]
        e = par[P_D] * s1 * fe * math.exp(min(par[P_QR] * sm / ((1.0 - fe) * s1), 700.0))
        return seq / (1.0 - fe) + e, 1.0 / (1.0 - fe), e * par[P_QR] / ((1.0 - fe) * s1)
    fe = max(f, 0.0)
    cc = max(_table_value(par, P_C, fe, 1.0), 0.0)
    ff = max(_table_value(par, P_F, fe, 0.0), 0.0)
    s2 = cc * seq * seq + 9.0 * ff * sm * sm
    if s2 <= 0.0:
        return 0.0, 1.0, 0.0
    s = math.sqrt(s2)
    return s, cc * seq / s, 9.0 * ff * sm / s


@njit(cache=True)
def _nucleation(par, p):
    """The rate ``A(p)`` of the nucleation of porosity per unit of ``ṗ``."""
    kind = int(par[P_NUCLEATION])
    if kind == 0 or p < par[P_NUCLEATION + 4]:
        return 0.0
    a, b, c = par[P_NUCLEATION + 1], par[P_NUCLEATION + 2], par[P_NUCLEATION + 3]
    if kind == 1:
        return a
    if kind == 2:
        z = (p - b) / c
        return a / (c * math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
    return a * math.exp(-p / b)


@njit(cache=True)
def _nucleation_slope(par, p):
    """A'(p), the derivative of :func:`_nucleation` (zero below its threshold)."""
    kind = int(par[P_NUCLEATION])
    if kind <= 1 or p < par[P_NUCLEATION + 4]:
        return 0.0
    a, b, c = par[P_NUCLEATION + 1], par[P_NUCLEATION + 2], par[P_NUCLEATION + 3]
    if kind == 2:
        z = (p - b) / c
        return -a / (c * math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z) * z / c
    return -a / b * math.exp(-p / b)


@njit(cache=True)
def _matrix_derivatives(par, seq, sm, f, s, d):
    """The derivatives of the matrix stress ``s`` of ``(seq, sm, f)`` in ``d``: by Σeq, Σm
    and f, then the second ones by (Σeq, Σeq), (Σeq, Σm), (Σm, Σm), (Σeq, f) and (Σm, f);
    those of GTN by the implicit function theorem on its potential."""
    for i in range(8):
        d[i] = 0.0
    kind = int(par[P_CRITERION])
    if kind == GURSON:
        raw = _table_value(par, P_FSTAR, f, f)
        cap = par[P_CAP]
        fs = min(max(raw, 0.0), cap)
        dfs = _table_slope(par, P_FSTAR, f, 1.0) if 0.0 <= raw < cap else 0.0
        if not s > 0.0:
            d[0] = 1.0
            return
        q1 = par[P_Q1]
        q3 = par[P_Q3]
        k = 1.5 * par[P_Q2]
        u = k * sm / s
        ch = math.cosh(u)
        sh = math.sinh(u)
        a = seq / s
        # Φ = (Σeq/s)² + 2 q1 fs cosh(k Σm/s) - 1 - q3 fs² and its derivatives
        p_s = -2.0 * (a * a + q1 * fs * u * sh) / s
        p_q = 2.0 * a / s
        p_m = 2.0 * q1 * fs * k * sh / s
        p_f = 2.0 * (q1 * ch - q3 * fs)
        s2 = s * s
        p_qq = 2.0 / s2
        p_qs = -4.0 * a / s2
        p_mm = 2.0 * q1 * fs * k * k * ch / s2
        p_ms = -2.0 * q1 * fs * k * (u * ch + sh) / s2
        p_mf = 2.0 * q1 * k * sh / s
        p_fs = -2.0 * q1 * u * sh / s
        p_ss = (6.0 * a * a + 2.0 * q1 * fs * (2.0 * u * sh + u * u * ch)) / s2
        s_q = -p_q / p_s
        s_m = -p_m / p_s
        s_f = -p_f / p_s
        d[0] = s_q
        d[1] = s_m
        d[2] = s_f * dfs
        d[3] = -(p_qq + 2.0 * p_qs * s_q + p_ss * s_q * s_q) / p_s
        d[4] = -(p_qs * s_m + p_ms * s_q + p_ss * s_q * s_m) / p_s
        d[5] = -(p_mm + 2.0 * p_ms * s_m + p_ss * s_m * s_m) / p_s
        d[6] = -(p_qs * s_f + p_fs * s_q + p_ss * s_q * s_f) / p_s * dfs
        d[7] = -(p_mf + p_ms * s_f + p_fs * s_m + p_ss * s_m * s_f) / p_s * dfs
        return
    if kind == ROUSSELIER:
        cap = par[P_CAP]
        fe = min(max(f, 0.0), cap)
        dfe = 1.0 if 0.0 <= f < cap else 0.0
        s1 = par[P_SIGMA1]
        qr = par[P_QR]
        a = 1.0 / (1.0 - fe)
        arg = qr * sm * a / s1
        free = arg < 700.0
        ex = math.exp(arg if free else 700.0)
        g = par[P_D] * s1 * fe * ex
        dg = par[P_D] * s1 * ex + (g * qr * sm * a * a / s1 if free else 0.0)  # d g / d fe
        d[0] = a
        d[1] = g * qr * a / s1
        d[2] = (seq * a * a + dg) * dfe
        d[5] = d[1] * qr * a / s1 if free else 0.0
        d[6] = a * a * dfe
        d[7] = qr / s1 * (dg * a + g * a * a) * dfe
        return
    fe = max(f, 0.0)
    dfe = 1.0 if f >= 0.0 else 0.0
    c_raw = _table_value(par, P_C, fe, 1.0)
    f_raw = _table_value(par, P_F, fe, 0.0)
    cc = max(c_raw, 0.0)
    ff = max(f_raw, 0.0)
    dc = _table_slope(par, P_C, fe, 0.0) * dfe if c_raw > 0.0 else 0.0
    dff = _table_slope(par, P_F, fe, 0.0) * dfe if f_raw > 0.0 else 0.0
    if not s > 0.0:
        d[0] = 1.0
        return
    s3 = s * s * s
    d[0] = cc * seq / s
    d[1] = 9.0 * ff * sm / s
    d[2] = (seq * seq * dc + 9.0 * sm * sm * dff) / (2.0 * s)
    d[3] = 9.0 * cc * ff * sm * sm / s3
    d[4] = -9.0 * cc * ff * seq * sm / s3
    d[5] = 9.0 * ff * cc * seq * seq / s3
    d[6] = dc * seq / s - cc * seq * d[2] / (s * s)
    d[7] = 9.0 * dff * sm / s - 9.0 * ff * sm * d[2] / (s * s)


@njit(cache=True)
def _relative(par, kin, sm_tr, seq_tr, dp):
    """The invariants of the relative trial stress ``σ_tr - Σ θ_i X_i,n`` after a plastic
    increment ``Δp`` (``θ_i = 1 / (1 + D_i Δp)``, see the module): its von Mises stress
    ``J*`` and mean stress ``m*``, ``c = Σ θ_i C_i``, and their derivatives by ``Δp``;
    ``kin``: the trial deviator (row 0) and the deviators and mean stresses of the
    backstresses at the start of the step (rows 1...); without backstresses, ``(Σeq_tr,
    Σm_tr, 0...)``."""
    nk = int(par[P_NKIN])
    if nk == 0:
        return seq_tr, sm_tr, 0.0, 0.0, 0.0, 0.0
    m, dm, c, dc = sm_tr, 0.0, 0.0, 0.0
    for k in range(nk):
        d = par[P_KIN + 2 * k + 1]
        th = 1.0 / (1.0 + d * dp)
        c += th * par[P_KIN + 2 * k]
        dc -= d * th * th * par[P_KIN + 2 * k]
        m -= th * kin[1 + k, 6]
        dm += d * th * th * kin[1 + k, 6]
    xx = 0.0
    xw = 0.0
    for r in range(6):
        xi = kin[0, r]
        w = 0.0
        for k in range(nk):
            d = par[P_KIN + 2 * k + 1]
            th = 1.0 / (1.0 + d * dp)
            xi -= th * kin[1 + k, r]
            w += d * th * th * kin[1 + k, r]
        g = 1.0 if r < 3 else 2.0
        xx += g * xi * xi
        xw += g * xi * w
    j = math.sqrt(max(1.5 * xx, 0.0))
    return j, m, c, (1.5 * xw / j if j > 0.0 else 0.0), dm, dc


@njit(cache=True)
def _relative_deviator(par, kin, dp, xi, w):
    """The deviator ``ξ*`` of the relative trial stress after a plastic increment ``Δp``
    (:func:`_relative`) and its derivative ``w = Σ D_i θ_i² X_i,n`` by ``Δp``."""
    for r in range(6):
        xi[r] = kin[0, r]
        w[r] = 0.0
    for k in range(int(par[P_NKIN])):
        d = par[P_KIN + 2 * k + 1]
        th = 1.0 / (1.0 + d * dp)
        for r in range(6):
            xi[r] -= th * kin[1 + k, r]
            w[r] += d * th * th * kin[1 + k, r]


@njit(cache=True)
def _residual(par, x, sm_tr, seq_tr, kb, mu, f_old, p_old, dt, vertex, out, guess, kin):
    """The four equations of the update (see the module) at ``x = (Δεv, Δεq, z, f)``, the
    matrix stress sought from ``guess`` (if positive: that of a nearby point), ``kin`` the
    trial deviator and the backstresses (:func:`_relative`); returns the matrix stress, or
    -1 when ``x`` is out of bounds."""
    dv, dq, z, f = x[0], x[1], x[2], x[3]
    if z < 0.0 or not (0.0 <= f < 0.9999):
        return -1.0
    viscous = par[4] > 0.0
    dp = dt * z ** par[5] if viscous else z
    jstar, mstar, c, _, _, _ = _relative(par, kin, sm_tr, seq_tr, dp)
    sm = mstar - (kb + 2.0 * c / 9.0) * dv
    seq = jstar - (3.0 * mu + c) * dq
    s, s_q, s_m = _matrix_stress(par, seq, sm, f, guess)
    if not math.isfinite(s):
        return -1.0
    p = p_old + dp
    out[0] = dq - jstar / (3.0 * mu + c) if vertex else dv * s_q - dq * s_m
    out[1] = s - _hardening(par, p) - (par[4] * z if viscous else 0.0)
    out[2] = (1.0 - f) * s * dp - sm * dv - seq * dq
    out[3] = f - f_old - (1.0 - f) * dv - _nucleation(par, p) * dp
    return s


@njit(cache=True)
def _solve4(a, b, n_rhs, m):
    """Solves ``a x = b`` (4 x 4, ``n_rhs`` columns of ``b``) by Gaussian elimination with
    partial pivoting, in place in ``b`` (``m``: a work matrix of 4 x 4); returns False if
    singular."""
    for r in range(4):
        for c in range(4):
            m[r, c] = a[r, c]
    for col in range(4):
        piv = col
        for r in range(col + 1, 4):
            if abs(m[r, col]) > abs(m[piv, col]):
                piv = r
        if abs(m[piv, col]) < 1e-300:
            return False
        if piv != col:
            for c in range(4):
                m[col, c], m[piv, c] = m[piv, c], m[col, c]
            for c in range(n_rhs):
                b[col, c], b[piv, c] = b[piv, c], b[col, c]
        for r in range(col + 1, 4):
            g = m[r, col] / m[col, col]
            if g != 0.0:
                for c in range(col, 4):
                    m[r, c] -= g * m[col, c]
                for c in range(n_rhs):
                    b[r, c] -= g * b[col, c]
    for col in range(3, -1, -1):
        for c in range(n_rhs):
            acc = b[col, c]
            for k in range(col + 1, 4):
                acc -= m[col, k] * b[k, c]
            b[col, c] = acc / m[col, col]
    return True


@njit(cache=True)
def _scaled_norm(r, e_scale, s_scale):
    return max(abs(r[0]) / e_scale, abs(r[1]) / s_scale, abs(r[2]) / (s_scale * e_scale),
               abs(r[3]))  # fmt: skip


@njit(cache=True)
def _jacobian(par, x, sm_tr, seq_tr, kb, mu, f_old, p_old, dt, vertex, s, d, jac, dr, kin):
    """The Jacobian ``jac`` of the residual (:func:`_residual`) at ``x``, whose matrix
    stress is ``s``, and in ``dr`` (4 x 2) its derivatives by the trial invariants Σm_tr
    and Σeq_tr (with backstresses: m* and J*); ``d``: a work vector of 8
    (:func:`_matrix_derivatives`)."""
    dv, dq, z, f = x[0], x[1], x[2], x[3]
    kv = par[4]
    viscous = kv > 0.0
    nv = par[5]
    if viscous:
        dp = dt * z**nv
        dpz = nv * dt * z ** (nv - 1.0) if z > 0.0 else (dt if nv == 1.0 else 0.0)
    else:
        dp = z
        dpz = 1.0
    jstar, mstar, c, dj, dm, dc = _relative(par, kin, sm_tr, seq_tr, dp)
    kbe = kb + 2.0 * c / 9.0
    mue = 3.0 * mu + c
    sm = mstar - kbe * dv
    seq = jstar - mue * dq
    # the derivatives of the relative invariants by Δp (the recovery of the backstresses)
    seq_p = dj - dc * dq
    sm_p = dm - 2.0 * dc / 9.0 * dv
    _matrix_derivatives(par, seq, sm, f, s, d)
    s_q, s_m, s_f = d[0], d[1], d[2]
    s_qq, s_qm, s_mm, s_qf, s_mf = d[3], d[4], d[5], d[6], d[7]
    s_p = s_q * seq_p + s_m * sm_p
    p = p_old + dp
    if vertex:  # dq - J* / (3 μ + c)
        jac[0, 0] = 0.0
        jac[0, 1] = 1.0
        jac[0, 2] = -dpz * (dj - jstar * dc / mue) / mue
        jac[0, 3] = 0.0
        dr[0, 0] = 0.0
        dr[0, 1] = -1.0 / mue
    else:  # dv S_q - dq S_m
        nq = dv * s_qq - dq * s_qm
        nm = dv * s_qm - dq * s_mm
        jac[0, 0] = s_q - kbe * nm
        jac[0, 1] = -s_m - mue * nq
        jac[0, 2] = dpz * (nq * seq_p + nm * sm_p)
        jac[0, 3] = dv * s_qf - dq * s_mf
        dr[0, 0] = nm
        dr[0, 1] = nq
    # σ* - R(p) - K z
    jac[1, 0] = -kbe * s_m
    jac[1, 1] = -mue * s_q
    jac[1, 2] = (s_p - _hardening_slope(par, p)) * dpz - (kv if viscous else 0.0)
    jac[1, 3] = s_f
    dr[1, 0] = s_m
    dr[1, 1] = s_q
    # (1 - f) σ* dp - Σm dv - Σeq dq
    g = 1.0 - f
    jac[2, 0] = -g * dp * kbe * s_m + kbe * dv - sm
    jac[2, 1] = -mue * g * dp * s_q + mue * dq - seq
    jac[2, 2] = (g * s + g * dp * s_p - dv * sm_p - dq * seq_p) * dpz
    jac[2, 3] = (g * s_f - s) * dp
    dr[2, 0] = g * dp * s_m - dv
    dr[2, 1] = g * dp * s_q - dq
    # f - f_old - (1 - f) dv - A(p) dp
    jac[3, 0] = -g
    jac[3, 1] = 0.0
    jac[3, 2] = -dpz * (_nucleation(par, p) + _nucleation_slope(par, p) * dp)
    jac[3, 3] = 1.0 + dv
    dr[3, 0] = 0.0
    dr[3, 1] = 0.0


@njit(cache=True)
def _porous_works(count):
    """``count`` sets of the work arrays of :func:`_porous_update`: the vectors of 6 and
    of 4, the matrices of 4 x 4, the right-hand sides, the derivatives of the residual by
    the trial invariants, those of the matrix stress, and the trial deviator with the
    backstresses (:func:`_relative`), with one more axis, the first one. The loops over
    the points allocate a set per chunk of points before their parallel loop, and take
    that of each chunk (:func:`_porous_set`): rows of arrays allocated in the loop would
    count their references (:func:`morphogen.homogenization.crystal.work_block`)."""
    return (np.empty((count, 6, 6)), np.empty((count, 4, 4)), np.empty((count, 2, 4, 4)),
            np.empty((count, 4, 2)), np.empty((count, 4, 2)), np.empty((count, 8)),
            np.zeros((count, 1 + MAX_KINEMATIC, 7)))  # fmt: skip


@njit(cache=True)
def _porous_set(works, k):
    """The set ``k`` of the work arrays of :func:`_porous_works` (views)."""
    return (works[0][k], works[1][k], works[2][k], works[3][k], works[4][k], works[5][k],
            works[6][k])  # fmt: skip


@njit(cache=True)
def _porous_work():
    """Work arrays of :func:`_porous_update` (a set of :func:`_porous_works`)."""
    return _porous_set(_porous_works(1), 0)


@njit(cache=True)
def porous_update(par, eps, eth, dt, old, new, sig, tan):
    """Implicit update of a porous point over a step ``dt`` (see the module): strain
    ``eps`` (engineering), thermal strain ``eth`` (tensor components). Fills the stress,
    the new internal variables and the tangent; returns 1 if the local iteration fails."""
    return _porous_update(par, eps, eth, dt, old, new, sig, tan, _porous_work())


@njit(cache=True)
def _porous_update(par, eps, eth, dt, old, new, sig, tan, wk, sym=True):
    """:func:`porous_update` with its work arrays ``wk`` (:func:`_porous_work`), the
    tangent symmetrized with ``sym``."""
    lam = par[0]
    mu = par[1]
    kb = lam + 2.0 * mu / 3.0
    viscous = par[4] > 0.0
    vec6, vec4, mat4, rhs, dr, ds, kin = wk
    ee = vec6[0]
    s_tr = vec6[1]
    nrm = vec6[2]
    d = vec6[3]
    xi = vec6[4]
    w = vec6[5]
    x = vec4[0]
    res = vec4[1]
    trial = vec4[2]
    rt = vec4[3]
    jac = mat4[0]
    m4 = mat4[1]
    for r in range(6):
        e = eps[r] if r < 3 else 0.5 * eps[r]
        ee[r] = e - eth[r] - old[r]
    tr = ee[0] + ee[1] + ee[2]
    sm_tr = kb * tr
    for r in range(6):
        s_tr[r] = 2.0 * mu * (ee[r] - tr / 3.0 if r < 3 else ee[r])
    seq_tr = math.sqrt(max(1.5 * _ddot(s_tr, s_tr), 0.0))
    for r in range(old.shape[0]):
        new[r] = old[r]
    # the trial deviator and the deviators and mean stresses of the backstresses
    nk = int(par[P_NKIN])
    for r in range(6):
        kin[0, r] = s_tr[r]
    for k in range(nk):
        base = STATE_BACK + 6 * k
        xm = (old[base] + old[base + 1] + old[base + 2]) / 3.0
        for r in range(6):
            kin[1 + k, r] = old[base + r] - (xm if r < 3 else 0.0)
        kin[1 + k, 6] = xm
    f_old = max(par[P_F0] + old[STATE_FG] + old[STATE_FN], 0.0)
    p_old = old[STATE_P]
    # the relative trial stress (the trial one without backstresses)
    jstar0, mstar0, c0, _, _, _ = _relative(par, kin, sm_tr, seq_tr, 0.0)
    s0, sq0, smd0 = matrix_stress(par, jstar0, mstar0, f_old)
    r_old = _hardening(par, p_old)
    # the stresses of the trial state (not σ*, whose exponential term of Rousselier may be
    # out of all proportion with them)
    s_scale = max(seq_tr, abs(sm_tr), jstar0, abs(mstar0), r_old, 1e-300)
    tiny = 1e-14 * s_scale
    inv = 1.5 / seq_tr if seq_tr > tiny else 0.0
    for r in range(6):
        nrm[r] = inv * s_tr[r]
    if not math.isfinite(s0):
        return 1
    if s0 - r_old <= 1e-13 * s_scale:
        for r in range(6):
            sig[r] = s_tr[r] + (sm_tr if r < 3 else 0.0)
            for c in range(6):
                tan[r, c] = 0.0
        for r in range(3):
            for c in range(3):
                tan[r, c] = lam
            tan[r, r] += 2.0 * mu
            tan[3 + r, 3 + r] = mu
        new[STATE_SSTAR] = s0
        return 0
    # plastic flow: initial guess, the cutting-plane step along the normal at the trial
    # stress, d(σ*)/dΛ = K_b S_m² + 3 μ S_q² + R' dp/dΛ, dp/dΛ = (σ - X) : N / ((1 - f) σ*)
    # (the backstresses: K_b + 2 c / 9 and 3 μ + c)
    e_scale = s_scale / (3.0 * mu)
    work = jstar0 * sq0 + mstar0 * smd0
    dp_dl = work / ((1.0 - f_old) * s0) if work > 0.0 else 1.0 / (1.0 - f_old)
    slope = (_hardening(par, p_old + 1e-6) - _hardening(par, max(p_old - 1e-6, 0.0))) / (
        p_old + 1e-6 - max(p_old - 1e-6, 0.0)
    )
    kbe = kb + 2.0 * c0 / 9.0
    mue = 3.0 * mu + c0
    lam0 = (s0 - r_old) / (kbe * smd0 * smd0 + mue * sq0 * sq0 + slope * dp_dl)
    dp0 = lam0 * dp_dl
    x[0] = lam0 * smd0
    x[1] = min(lam0 * sq0, 0.5 * jstar0 / mue)
    x[3] = f_old
    if viscous:  # the viscous stress of the rate-independent increment, at most the overstress
        bound = (s0 - r_old) / par[4]
        x[2] = min((dp0 / dt) ** (1.0 / par[5]), bound) if dt > 0.0 else bound
    else:
        x[2] = dp0
    vertex = jstar0 <= tiny and int(par[P_CRITERION]) == ROUSSELIER
    converged = False
    for _attempt in range(2):
        # sx: the matrix stress of x, the guess of the nearby points
        sx = _residual(par, x, sm_tr, seq_tr, kb, mu, f_old, p_old, dt, vertex, res, 0.0, kin)
        if sx < 0.0:
            x[0] = 0.0
            x[1] = 0.0
            x[2] = 0.0
            x[3] = f_old
            sx = _residual(par, x, sm_tr, seq_tr, kb, mu, f_old, p_old, dt, vertex, res, 0.0,
                           kin)  # fmt: skip
            if sx < 0.0:
                return 1
        norm = _scaled_norm(res, e_scale, s_scale)
        for _ in range(80):
            if norm <= 1e-12:
                converged = True
                break
            _jacobian(par, x, sm_tr, seq_tr, kb, mu, f_old, p_old, dt, vertex, sx, ds, jac, dr,
                      kin)  # fmt: skip
            for i in range(4):
                rhs[i, 0] = -res[i]
            if not _solve4(jac, rhs, 1, m4):
                return 1
            # the porosity falls at most by 90 % per iteration: it approaches full density
            # (the elliptic surface grows without bound as F(f) vanishes) without passing it
            alpha = 1.0
            if rhs[3, 0] < 0.0 and x[3] > 0.0:
                alpha = min(1.0, 0.9 * x[3] / -rhs[3, 0])
            accepted = False
            n_t = norm
            for _ in range(30):
                for i in range(4):
                    trial[i] = x[i] + alpha * rhs[i, 0]
                trial[3] = max(trial[3], 0.0)  # a dense point stays dense (no roundoff below)
                inside = _residual(par, trial, sm_tr, seq_tr, kb, mu, f_old, p_old, dt, vertex, rt,
                                   sx, kin)  # fmt: skip
                if inside >= 0.0:
                    n_t = _scaled_norm(rt, e_scale, s_scale)
                    if n_t < (1.0 - 1e-4 * alpha) * norm or alpha < 1e-6:
                        accepted = True
                        break
                alpha *= 0.5
            if not accepted:
                return 1
            for i in range(4):
                x[i] = trial[i]
                res[i] = rt[i]
            sx = inside
            norm = n_t
        if not converged:
            return 1
        dp = dt * x[2] ** par[5] if viscous else x[2]
        jstar, _, c, _, _, _ = _relative(par, kin, sm_tr, seq_tr, dp)
        seq = jstar - (3.0 * mu + c) * x[1]
        if seq >= -1e-10 * s_scale or vertex:
            break
        # Rousselier's vertex: the hydrostatic return
        vertex = True
        converged = False
        x[1] = jstar / (3.0 * mu + c)
    if not converged:
        return 1
    dv, dq, z, f = x[0], x[1], x[2], x[3]
    if viscous:
        dp = dt * z ** par[5]
        dpz = par[5] * dt * z ** (par[5] - 1.0) if z > 0.0 else (dt if par[5] == 1.0 else 0.0)
    else:
        dp = z
        dpz = 1.0
    jstar, mstar, c, _, _, _ = _relative(par, kin, sm_tr, seq_tr, dp)
    sm = mstar - (kb + 2.0 * c / 9.0) * dv
    seq = max(jstar - (3.0 * mu + c) * dq, 0.0)
    s, _, _ = matrix_stress(par, seq, sm, f)
    # the deviator of the relative trial stress, whose direction the plastic strain takes
    _relative_deviator(par, kin, dp, xi, w)
    if nk > 0:
        inv = 1.5 / jstar if jstar > tiny else 0.0
        for r in range(6):
            nrm[r] = inv * xi[r]
    for r in range(6):
        new[r] = old[r] + (dv / 3.0 if r < 3 else 0.0) + dq * nrm[r]
    new[STATE_P] = p_old + dp
    new[STATE_FN] = old[STATE_FN] + _nucleation(par, p_old + dp) * dp
    # no porosity below zero (a dense point whose volume change is a roundoff)
    new[STATE_FG] = max(old[STATE_FG] + (1.0 - f) * dv, -par[P_F0] - new[STATE_FN])
    new[STATE_SSTAR] = s
    for k in range(nk):  # X = θ (X_n + 2/3 C Δεp), its hydrostatic part included
        th = 1.0 / (1.0 + par[P_KIN + 2 * k + 1] * dp)
        cc = 2.0 / 3.0 * par[P_KIN + 2 * k]
        base = STATE_BACK + 6 * k
        for r in range(6):
            dep = (dv / 3.0 if r < 3 else 0.0) + dq * nrm[r]
            new[base + r] = th * (old[base + r] + cc * dep)
    # σ = σ_tr - K_b Δεv I - 2 μ Δεq n: the deviator (s_tr - ξ*) + (1 - 3 μ Δεq / J*) ξ*
    ratio_s = max(jstar - 3.0 * mu * dq, 0.0) / jstar if jstar > tiny else 1.0
    sm_abs = sm_tr - kb * dv
    for r in range(6):
        sig[r] = (s_tr[r] - xi[r]) + ratio_s * xi[r] + (sm_abs if r < 3 else 0.0)
    # consistent tangent: the sensitivity of the solution to the trial invariants,
    # jac dx = -dr d(Σm_tr, Σeq_tr) (with backstresses: d(m*, J*), dJ* = n : ds_tr)
    _jacobian(par, x, sm_tr, seq_tr, kb, mu, f_old, p_old, dt, vertex, sx, ds, jac, dr, kin)
    for i in range(4):
        rhs[i, 0] = -dr[i, 0]
        rhs[i, 1] = -dr[i, 1]
    if not _solve4(jac, rhs, 2, m4):
        return 1
    gq = rhs[1, 1]
    ratio = ratio_s if jstar > tiny else 1.0 - 3.0 * mu * gq
    # the turn of the direction with Δp: dn = 3 / (2 J*) (w - 2/3 (n : w) n) dΔp
    turn = 3.0 * mu * dq / jstar if nk > 0 and jstar > tiny else 0.0
    nw = _ddot(nrm, w)
    for j in range(6):
        for r in range(6):
            d[r] = 0.0
        d[j] = 1.0 if j < 3 else 0.5
        trd = 1.0 if j < 3 else 0.0
        nd = _ddot(nrm, d)
        dsm_tr = kb * trd
        dseq_tr = 2.0 * mu * nd
        ddv = rhs[0, 0] * dsm_tr + rhs[0, 1] * dseq_tr
        ddq = rhs[1, 0] * dsm_tr + rhs[1, 1] * dseq_tr
        ddp = turn * dpz * (rhs[2, 0] * dsm_tr + rhs[2, 1] * dseq_tr)
        dsm = dsm_tr - kb * ddv
        dseq = dseq_tr - 3.0 * mu * ddq
        for r in range(6):
            dev = d[r] - trd / 3.0 if r < 3 else d[r]
            value = 2.0 / 3.0 * dseq * nrm[r] + 2.0 * mu * ratio * (dev - 2.0 / 3.0 * nd * nrm[r])
            if turn != 0.0:
                value -= ddp * (w[r] - 2.0 / 3.0 * nw * nrm[r])
            if r < 3:
                value += dsm
            tan[r, j] = value
    if sym:
        symmetrize(tan)
    return 0


def porosity(state, params, pmat) -> np.ndarray:
    """The porosity ``f`` of porous points from their internal variables (points x width),
    the rows of parameters and the row of each point."""
    state = np.asarray(state)
    return params[np.asarray(pmat), P_F0] + state[:, STATE_FG] + state[:, STATE_FN]
