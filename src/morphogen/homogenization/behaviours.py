"""Nonlinear behaviours of the phases: von Mises elasto-(visco)plasticity, and the
integration of the behaviours of all the points (crystal plasticity:
:mod:`morphogen.homogenization.crystal`; porous plasticity:
:mod:`morphogen.homogenization.porous`).

The model of Lemaitre & Chaboche (1990), in small strains, with isotropic elasticity
(``sigma = D (eps - eps_th - eps_p)``):

* yield function ``f = J(sigma - X) - R(p) - R0``, with ``J`` the von Mises norm
  ``sqrt(3/2 s : s)`` of the deviator;
* isotropic hardening ``R(p) = H p + sum_i Q_i (1 - exp(-b_i p))`` of the cumulated
  plastic strain ``p``;
* kinematic hardening ``X = sum_k X_k``, ``dX_k = 2/3 C_k deps_p - D_k X_k dp``
  (Armstrong-Frederick; ``D_k = 0``: linear, Prager);
* normality ``deps_p = dp n``, ``n = 3/2 (s - X) / J(sigma - X)``;
* Norton's viscous flow ``dp/dt = <f / K>^N`` (viscoplasticity), or the consistency
  condition ``f = 0`` when there is no viscosity (rate-independent plasticity).

The equations are integrated implicitly (backward Euler) over each time step. The
Armstrong-Frederick backstresses then read ``X_k = theta_k (X_k^n + 2/3 C_k dp n)`` with
``theta_k = 1 / (1 + D_k dp)``; the relative stress ``s - X`` is parallel to ``xi* = s_tr
- sum theta_k X_k^n`` (``s_tr``: the elastic trial deviator), and the whole update reduces
to one scalar equation for the increment ``dp`` (or, for viscoplasticity, for the
viscous stress, which is better conditioned near ``dp = 0``), solved by Newton's method
safeguarded by bisection. Its linearization gives the consistent tangent operator
``d sigma / d eps`` of the step (Simo & Taylor 1985), symmetric but for the
Armstrong-Frederick terms (the backstresses turn the direction of flow: about 1 % of
its largest entries). The kernels symmetrize it for the conjugate gradient solvers
(which only removes the round-off of the symmetric laws), or keep it whole for GMRES
(``symmetric=False``: :mod:`morphogen.homogenization.nonlinear` decides by the laws of
the phases).

Stresses and the internal tensors are stored by their Voigt components ``(11, 22, 33,
23, 31, 12)``; strains in the engineering convention of the solvers (shears doubled).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from morphogen.homogenization.crystal import cp_small, work_block, work_blocks
from morphogen.homogenization.porous import (
    N_POROUS,
    _porous_set,
    _porous_update,
    _porous_works,
)
from morphogen.homogenization.tensors import symmetrize

#: Revision of the kernels of the other modules compiled into :func:`update_points`
#: (crystal, porous). Numba dates the cache of a compiled function by the file that
#: defines it alone: a change of those kernels comes with a change of this number, and
#: so of this file, which makes the caches of the earlier versions stale (2: twins and
#: cleavage planes among the systems of the crystals).
_KERNELS = 22

#: Most nonlinear isotropic terms ``Q (1 - exp(-b p))`` and kinematic terms ``(C, D)``.
MAX_ISOTROPIC = 3
MAX_KINEMATIC = 4
#: Layout of a row of parameters: lambda, mu, R0, H, K, N, then the (Q, b) and (C, D) pairs.
_KIN = 6 + 2 * MAX_ISOTROPIC
N_PARAMS = _KIN + 2 * MAX_KINEMATIC
#: The rows of parameters of the GPU: those of von Mises, then the six coefficients of Hill
PARAMS_WIDTH = N_PARAMS + 6
#: Rows of parameters of the kernels of the GPU: those of von Mises and Hill, or of a
#: porous point
ROW_WIDTH = max(PARAMS_WIDTH, N_POROUS)
#: Internal variables of a point: plastic strain (6, tensor components), p, backstresses.
STATE_P = 6
STATE_X = 7
#: Behaviours of the points: elastic, von Mises, crystal plasticity, an MFront behaviour
#: (integrated by MGIS: :mod:`morphogen.homogenization.mfront`; the kernels skip them),
#: porous plasticity
ELASTIC = 0
PLASTIC = 1
CRYSTAL = 2
MFRONT = 3
POROUS = 4
HILL = 5  # Hill's criterion
#: The components of Hill's coefficients in the configurations (those of Z-set's ``hill1``
#: ... ``hill6``), and their places in the Voigt order of the kernels (11, 22, 33, 23, 31, 12)
HILL_COMPONENTS = ("11", "22", "33", "12", "23", "31")
_HILL_VOIGT = {"11": 0, "22": 1, "33": 2, "23": 3, "31": 4, "12": 5}


def kernel_constants(ns: int, width: int) -> dict:
    """The constants of the C ports of the kernels of the points (``behaviours.cu``, for
    the GPU and the MFront sources of crystals in finite strains): the slip systems of the
    arrays of the crystals (NS), the internal variables of a point (WIDTH) and the
    constants of the Python modules."""
    from morphogen.homogenization import crystal as cp
    from morphogen.homogenization import porous as pp

    assert pp.STATE_P == STATE_P
    porous = {name: getattr(pp, name) for name in (
        "P_CRITERION", "P_F0", "P_Q1", "P_Q2", "P_Q3", "P_SIGMA1", "P_D", "P_QR",
        "P_NUCLEATION", "P_FSTAR", "P_C", "P_F", "P_CAP", "P_NKIN", "P_KIN", "GURSON",
        "ROUSSELIER", "ELLIPTIC", "STATE_FG", "STATE_FN", "STATE_SSTAR", "STATE_BACK")}  # fmt: skip
    return {
        "NS": int(ns),
        "WIDTH": int(width),
        "CP_ALPHA": cp.CP_ALPHA,
        "STATE_P": STATE_P,
        "STATE_X": STATE_X,
        "KIN": _KIN,
        "N_PARAMS": N_PARAMS,
        "PARAMS_WIDTH": PARAMS_WIDTH,
        "ROW_WIDTH": ROW_WIDTH,
        "N_CP": len(cp.PARAMS),
        "MAX_ISOTROPIC": MAX_ISOTROPIC,
        "ELASTIC": ELASTIC,
        "PLASTIC": PLASTIC,
        "CRYSTAL": CRYSTAL,
        "MFRONT": MFRONT,
        "HILL": HILL,
        "POROUS": POROUS,
        "POROUS_ISOTROPIC": pp.MAX_ISOTROPIC,
        "POROUS_KINEMATIC": pp.MAX_KINEMATIC,
        **porous,
        "HILL_IT": _HILL_IT,
        "SLIP_FLOOR": float(cp._SLIP_FLOOR),
        "LOCAL_TOL": float(cp._LOCAL_TOL),
        "MAX_LOCAL": cp._MAX_LOCAL,
        "MAX_ROUNDS": cp._MAX_ROUNDS,
        "SHRINK": float(cp._SHRINK),
        "SUBSTEP_LEVELS": cp._SUBSTEP_LEVELS,
    }


def _pairs(value, name: str, first: str, second: str) -> tuple[tuple[float, float], ...]:
    """Pairs of parameters given as a mapping ``{first: .., second: ..}``, a list of such
    mappings, or a list of pairs."""
    if value is None:
        return ()
    items = [value] if isinstance(value, dict) else list(value)
    if items and not isinstance(items[0], (dict, list, tuple)):
        items = [items]  # a single pair
    out = []
    for item in items:
        if isinstance(item, dict):
            unknown = set(item) - {first, second}
            if unknown or first not in item or second not in item:
                raise ValueError(f"{name}: each term gives {first} and {second}")
            pair = (item[first], item[second])
        else:
            if len(item) != 2:
                raise ValueError(f"{name}: each term is a pair ({first}, {second})")
            pair = tuple(item)
        out.append((float(pair[0]), float(pair[1])))
    return tuple(out)


@dataclass(frozen=True)
class Plasticity:
    """Von Mises elasto-(visco)plasticity of an isotropic elastic material.

    ``yield_stress`` (R0), linear isotropic hardening ``hardening`` (H), nonlinear
    isotropic terms ``isotropic`` (pairs Q, b), kinematic terms ``kinematic`` (pairs
    C, D), and Norton's viscosity ``viscosity`` (K) and exponent ``exponent`` (N) for
    viscoplasticity (rate-independent plasticity without them). Stresses in the units
    of the elastic moduli, times in those of the loading paths.
    """

    yield_stress: float
    hardening: float = 0.0
    isotropic: tuple[tuple[float, float], ...] = ()
    kinematic: tuple[tuple[float, float], ...] = ()
    viscosity: float | None = None
    exponent: float | None = None
    hill: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        r0 = float(self.yield_stress)
        if not (math.isfinite(r0) and r0 >= 0.0):
            raise ValueError(f"yield: a non-negative yield stress, got {self.yield_stress!r}")
        object.__setattr__(self, "yield_stress", r0)
        h = float(self.hardening)
        if not (math.isfinite(h) and h >= 0.0):
            raise ValueError("hardening: a non-negative linear isotropic hardening modulus")
        object.__setattr__(self, "hardening", h)
        iso = _pairs(self.isotropic, "isotropic", "Q", "b")
        kin = _pairs(self.kinematic, "kinematic", "C", "D")
        if len(iso) > MAX_ISOTROPIC:
            raise ValueError(f"isotropic: at most {MAX_ISOTROPIC} terms")
        if len(kin) > MAX_KINEMATIC:
            raise ValueError(f"kinematic: at most {MAX_KINEMATIC} terms")
        for q, b in iso:
            if not (math.isfinite(q) and math.isfinite(b) and b > 0):
                raise ValueError("isotropic: terms (Q, b) with b > 0")
            if q < 0:
                raise ValueError("isotropic: Q >= 0 (no softening)")
        for c, d in kin:
            if not (math.isfinite(c) and math.isfinite(d) and c >= 0 and d >= 0):
                raise ValueError("kinematic: terms (C, D) with C >= 0 and D >= 0")
        object.__setattr__(self, "isotropic", iso)
        object.__setattr__(self, "kinematic", kin)
        if (self.viscosity is None) != (self.exponent is None):
            raise ValueError("norton: give both K and n (or neither: rate-independent)")
        if self.viscosity is not None:
            k, n = float(self.viscosity), float(self.exponent)
            if not (math.isfinite(k) and k > 0):
                raise ValueError("norton: K > 0")
            if not (math.isfinite(n) and n >= 1.0):
                raise ValueError("norton: n >= 1")
            object.__setattr__(self, "viscosity", k)
            object.__setattr__(self, "exponent", n)
        if self.hill is not None:
            object.__setattr__(self, "hill", _hill_coefficients(self.hill))

    @property
    def viscous(self) -> bool:
        return self.viscosity is not None

    def yield_stresses(self) -> dict[str, float]:
        """The initial yield stresses in tension along the axes and in shear in their planes
        (``"11"`` ... ``"12"``...): ``R0`` and ``R0 / sqrt(3)`` for von Mises; for Hill's
        criterion, ``R0 sqrt(6 / (4 h11 + h22 + h33))`` along 1 (and its permutations)
        and ``R0 / sqrt(3 h12)`` in the plane 12."""
        h = dict(zip(HILL_COMPONENTS, self.hill or (1.0,) * 6, strict=True))
        r0 = self.yield_stress
        out = {}
        for i, j, k in (("11", "22", "33"), ("22", "33", "11"), ("33", "11", "22")):
            out[i] = r0 * math.sqrt(6.0 / (4.0 * h[i] + h[j] + h[k]))
        for c in ("12", "23", "31"):
            out[c] = r0 / math.sqrt(3.0 * h[c])
        return out

    def params(self, lam: float, mu: float) -> np.ndarray:
        """Row of parameters of the integration kernels (with Hill's coefficients after
        those of von Mises, in the order of the kernels)."""
        row = np.zeros(N_PARAMS + (6 if self.hill is not None else 0))
        if self.hill is not None:
            for c, value in zip(HILL_COMPONENTS, self.hill, strict=True):
                row[N_PARAMS + _HILL_VOIGT[c]] = value
        row[:4] = (lam, mu, self.yield_stress, self.hardening)
        if self.viscous:
            row[4:6] = (self.viscosity, self.exponent)
        for i, (q, b) in enumerate(self.isotropic):
            row[6 + 2 * i : 8 + 2 * i] = (q, b)
        for k, (c, d) in enumerate(self.kinematic):
            row[_KIN + 2 * k : _KIN + 2 * k + 2] = (c, d)
        return row

    def to_config(self) -> dict:
        out: dict = {"yield": self.yield_stress}
        if self.hardening:
            out["hardening"] = self.hardening
        if self.isotropic:
            out["isotropic"] = [{"Q": q, "b": b} for q, b in self.isotropic]
        if self.kinematic:
            out["kinematic"] = [{"C": c, "D": d} for c, d in self.kinematic]
        if self.viscous:
            out["norton"] = {"K": self.viscosity, "n": self.exponent}
        if self.hill is not None:
            out["hill"] = dict(zip(HILL_COMPONENTS, self.hill, strict=True))
        return out

    @classmethod
    def from_config(cls, spec) -> Plasticity:
        """``{yield: R0, hardening: H, isotropic: [{Q, b}...], kinematic: [{C, D}...],
        norton: {K, n}, hill: {11: h11, 22: h22, 33: h33, 12: h12, 23: h23, 31: h31}}``
        (aliases ``R0`` and ``H``; the coefficients of Hill not given are 1)."""
        if isinstance(spec, Plasticity):
            return spec
        if not isinstance(spec, dict):
            raise ValueError(
                "plasticity: a mapping (yield, hardening, isotropic, kinematic, norton)"
            )
        allowed = {"yield", "R0", "hardening", "H", "isotropic", "kinematic", "norton", "hill"}
        unknown = sorted(set(spec) - allowed)
        if unknown:
            raise ValueError(f"plasticity: unknown keys {unknown}; allowed: {sorted(allowed)}")
        if ("yield" in spec) == ("R0" in spec):
            raise ValueError("plasticity: give the yield stress (yield, or R0)")
        if "hardening" in spec and "H" in spec:
            raise ValueError("plasticity: give hardening or H, not both")
        norton = spec.get("norton")
        k = n = None
        if norton is not None:
            if not isinstance(norton, dict) or set(norton) != {"K", "n"}:
                raise ValueError("plasticity.norton: {K: viscosity, n: exponent}")
            k, n = float(norton["K"]), float(norton["n"])
        return cls(
            yield_stress=float(spec.get("yield", spec.get("R0"))),
            hardening=float(spec.get("hardening", spec.get("H", 0.0))),
            isotropic=spec.get("isotropic") or (),
            kinematic=spec.get("kinematic") or (),
            viscosity=k,
            exponent=n,
            hill=spec.get("hill"),
        )


def _hill_coefficients(spec) -> tuple[float, ...]:
    """Hill's coefficients in the order of :data:`HILL_COMPONENTS`, from a mapping of the
    components (those not given are 1) or from six numbers in that order."""
    if isinstance(spec, dict):
        keys = {str(k): v for k, v in spec.items()}
        unknown = sorted(set(keys) - set(HILL_COMPONENTS))
        if unknown:
            raise ValueError(
                f"hill: unknown components {unknown}; the components are "
                f"{', '.join(HILL_COMPONENTS)}"
            )
        values = [keys.get(c, 1.0) for c in HILL_COMPONENTS]
    else:
        values = list(spec)
        if len(values) != 6:
            raise ValueError(
                f"hill: six coefficients ({', '.join(HILL_COMPONENTS)}), or a mapping of them"
            )
    out = tuple(float(v) for v in values)
    if not all(math.isfinite(v) and v > 0.0 for v in out):
        raise ValueError("hill: positive coefficients (1 everywhere: von Mises)")
    return out


# -- integration kernels -----------------------------------------------------------------


@njit(cache=True, inline="always")
def _ddot(a, b):
    """a : b of two symmetric tensors given by their Voigt components."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + 2.0 * (a[3] * b[3] + a[4] * b[4] + a[5] * b[5])


@njit(cache=True, inline="always")
def _hardening(par, p):
    """R(p) and R'(p)."""
    r = par[3] * p
    dr = par[3]
    for i in range(MAX_ISOTROPIC):
        q = par[6 + 2 * i]
        if q != 0.0:
            b = par[7 + 2 * i]
            e = math.exp(-b * p)
            r += q * (1.0 - e)
            dr += q * b * e
    return r, dr


@njit(cache=True, inline="always")
def _phi(par, nk, s_tr, old, dp, xi, nrm, w):
    """Yield function after a plastic increment ``dp``, without the viscous stress, and
    its derivative; fills ``xi`` (xi*), the flow direction ``nrm`` and ``w = sum D_k
    theta_k^2 X_k^n``. Returns (phi, dphi/ddp, J(xi*))."""
    mu = par[1]
    csum = 0.0
    c2sum = 0.0
    for r in range(6):
        xi[r] = s_tr[r]
        w[r] = 0.0
    for k in range(nk):
        c = par[_KIN + 2 * k]
        d = par[_KIN + 2 * k + 1]
        th = 1.0 / (1.0 + d * dp)
        csum += th * c
        c2sum += th * th * c
        base = STATE_X + 6 * k
        for r in range(6):
            xk = old[base + r]
            xi[r] -= th * xk
            w[r] += d * th * th * xk
    jm = math.sqrt(max(1.5 * _ddot(xi, xi), 0.0))
    inv = 1.5 / jm if jm > 0.0 else 0.0
    for r in range(6):
        nrm[r] = inv * xi[r]
    rr, dr = _hardening(par, old[STATE_P] + dp)
    phi = jm - (3.0 * mu + csum) * dp - rr - par[2]
    dphi = _ddot(nrm, w) - 3.0 * mu - c2sum - dr
    return phi, dphi, jm


@njit(cache=True)
def _elastic_tangent(lam, mu, tan):
    for r in range(6):
        for c in range(6):
            tan[r, c] = 0.0
    for r in range(3):
        for c in range(3):
            tan[r, c] = lam
        tan[r, r] += 2.0 * mu
        tan[3 + r, 3 + r] = mu


@njit(cache=True)
def _j2_update(par, nk, eps, eth, dt, old, new, sig, tan, buf, sym=True):
    """Implicit update of a von Mises point over a step ``dt``: strain ``eps``
    (engineering), thermal strain ``eth`` (tensor components). Fills the stress, the new
    internal variables and the tangent (symmetrized with ``sym``); returns 1 if the local
    Newton iteration fails."""
    lam = par[0]
    mu = par[1]
    kv = par[4]
    nv = par[5]
    ee = buf[0]
    s_tr = buf[1]
    xi = buf[2]
    nrm = buf[3]
    w = buf[4]
    d = buf[5]
    dxi = buf[6]
    for r in range(6):
        e = eps[r] if r < 3 else 0.5 * eps[r]
        ee[r] = e - eth[r] - old[r]
    tr = ee[0] + ee[1] + ee[2]
    for r in range(6):
        sig[r] = 2.0 * mu * ee[r]
    for r in range(3):
        sig[r] += lam * tr
    mean = (sig[0] + sig[1] + sig[2]) / 3.0
    for r in range(6):
        s_tr[r] = sig[r] - mean if r < 3 else sig[r]
    n_state = old.shape[0]
    for r in range(n_state):
        new[r] = old[r]
    f_tr, dphi, jm = _phi(par, nk, s_tr, old, 0.0, xi, nrm, w)
    rr0, _ = _hardening(par, old[STATE_P])
    scale = jm + par[2] + abs(rr0) + 1e-300
    if f_tr <= 1e-13 * scale:
        _elastic_tangent(lam, mu, tan)
        return 0
    tol = 1e-12 * scale
    dp = 0.0
    y = 0.0
    converged = False
    if kv > 0.0:
        # viscoplastic: unknown y = (dp / dt)^(1/N), the viscous stress over K
        lo, hi = 0.0, f_tr / kv
        y = 0.5 * hi
        for _ in range(200):
            dp = dt * y**nv
            phi, dphi, jm = _phi(par, nk, s_tr, old, dp, xi, nrm, w)
            res = phi - kv * y
            if abs(res) <= tol:
                converged = True
                break
            if res > 0.0:
                lo = y
            else:
                hi = y
            der = dphi * nv * dt * y ** (nv - 1.0) - kv
            step = y - res / der if der < 0.0 else 0.5 * (lo + hi)
            if not (lo < step < hi):
                step = 0.5 * (lo + hi)
            if abs(step - y) <= 1e-15 * hi:
                y = step
                converged = True
                break
            y = step
        dp = dt * y**nv
    else:
        # rate-independent: unknown dp, bracketed by doubling
        lo, hi = 0.0, f_tr / (3.0 * mu)
        for _ in range(200):
            phi, dphi, jm = _phi(par, nk, s_tr, old, hi, xi, nrm, w)
            if phi < 0.0:
                break
            lo = hi
            hi *= 2.0
        dp = lo + 0.5 * (hi - lo)
        for _ in range(200):
            phi, dphi, jm = _phi(par, nk, s_tr, old, dp, xi, nrm, w)
            if abs(phi) <= tol:
                converged = True
                break
            if phi > 0.0:
                lo = dp
            else:
                hi = dp
            step = dp - phi / dphi if dphi < 0.0 else 0.5 * (lo + hi)
            if not (lo < step < hi):
                step = 0.5 * (lo + hi)
            if abs(step - dp) <= 1e-15 * hi:
                dp = step
                converged = True
                break
            dp = step
    if not converged or not math.isfinite(dp):
        return 1
    phi, dphi, jm = _phi(par, nk, s_tr, old, dp, xi, nrm, w)
    if jm <= 0.0:
        return 1
    # new state and stress
    for r in range(6):
        new[r] = old[r] + dp * nrm[r]
        sig[r] -= 2.0 * mu * dp * nrm[r]
    new[STATE_P] = old[STATE_P] + dp
    for k in range(nk):
        c = par[_KIN + 2 * k]
        th = 1.0 / (1.0 + par[_KIN + 2 * k + 1] * dp)
        base = STATE_X + 6 * k
        for r in range(6):
            new[base + r] = th * (old[base + r] + 2.0 / 3.0 * c * dp * nrm[r])
    # consistent tangent: d dp = 2 mu / h n : d eps
    h = -dphi
    if kv > 0.0 and y > 0.0:
        lg = (1.0 - nv) * math.log(y)
        h = h + (kv / (nv * dt) * math.exp(lg) if lg < 700.0 else math.inf)
    inv_h = 1.0 / h if (h > 0.0 and math.isfinite(h)) else 0.0
    for j in range(6):
        for r in range(6):
            d[r] = 0.0
        d[j] = 1.0 if j < 3 else 0.5
        trd = 1.0 if j < 3 else 0.0
        ddp = 2.0 * mu * _ddot(nrm, d) * inv_h
        for r in range(6):
            dev = d[r] - trd / 3.0 if r < 3 else d[r]
            dxi[r] = 2.0 * mu * dev + w[r] * ddp
        ndxi = _ddot(nrm, dxi)
        for r in range(6):
            dn = 1.5 / jm * (dxi[r] - 2.0 / 3.0 * nrm[r] * ndxi)
            ds = 2.0 * mu * d[r] - 2.0 * mu * (ddp * nrm[r] + dp * dn)
            if r < 3:
                ds += lam * trd
            tan[r, j] = ds
    if sym:
        symmetrize(tan)
    return 0


# -- Hill's criterion --------------------------------------------------------------------

_SQ2 = math.sqrt(2.0)
_HILL_IT = 60  # Newton iterations of a Hill point


@njit(cache=True)
def _hill_matrix(h, a):
    """``a = 3/2 P M P`` in Mandel's notation (``P``: the deviatoric projector, ``M =
    diag(h)``), so that ``J_H^2 = x . a . x`` for the relative stress ``x``."""
    for r in range(6):
        for c in range(6):
            a[r, c] = 0.0
    for r in range(3):
        for c in range(3):
            acc = 0.0
            for k in range(3):
                acc += ((r == k) - 1.0 / 3.0) * h[k] * ((c == k) - 1.0 / 3.0)
            a[r, c] = 1.5 * acc
    for r in range(3, 6):
        a[r, r] = 1.5 * h[r]


@njit(cache=True)
def _hill_system(par, nk, a, s_tr, xn, p0, de, q, dt, res, jac, sr, nrm, w, nn):
    """Residuals of a Hill update ``(de, q)`` and their Jacobian (Mandel's notation):
    ``de - dp n`` and ``J_H - R(p) - R0 - K q``, the unknown ``q`` being ``dp``, or the
    viscous stress over K, ``(dp / dt)^(1/N)``, with a viscosity. Fills the relative stress
    ``sr``, the normal ``nrm``, ``w = d sr / d dp`` and ``nn = d n / d sr``; returns
    ``(J_H, dp, d dp / d q)``, or J_H = 0 where the normal is not defined."""
    lam = par[0]
    mu = par[1]
    kv = par[4]
    nv = par[5]
    if kv > 0.0:
        dp = dt * q**nv
        ddp = nv * dt * q ** (nv - 1.0) if q > 0.0 else (dt if nv == 1.0 else 0.0)
    else:
        dp = q
        ddp = 1.0
    cb = 0.0
    cdb = 0.0
    for r in range(6):
        sr[r] = s_tr[r]
        w[r] = 0.0
    for k in range(nk):
        c = par[_KIN + 2 * k]
        d = par[_KIN + 2 * k + 1]
        th = 1.0 / (1.0 + d * dp)
        cb += 2.0 / 3.0 * th * c
        cdb += 2.0 / 3.0 * c * d * th * th
        for r in range(6):
            sr[r] -= th * xn[k, r]
            w[r] += d * th * th * xn[k, r]
    tr = de[0] + de[1] + de[2]
    for r in range(6):
        b = (2.0 * mu + cb) * de[r] + (lam * tr if r < 3 else 0.0)
        sr[r] -= b
        w[r] += cdb * de[r]
    jj = 0.0
    for r in range(6):
        acc = 0.0
        for c in range(6):
            acc += a[r, c] * sr[c]
        nrm[r] = acc
        jj += acc * sr[r]
    if not jj > 0.0:
        return 0.0, dp, ddp
    jh = math.sqrt(jj)
    for r in range(6):
        nrm[r] /= jh
    for r in range(6):
        for c in range(6):
            nn[r, c] = (a[r, c] - nrm[r] * nrm[c]) / jh
    rr, dr = _hardening(par, p0 + dp)
    for r in range(6):
        res[r] = de[r] - dp * nrm[r]
    res[6] = jh - rr - par[2] - (kv * q if kv > 0.0 else 0.0)
    # d res / d de = I + dp nn B, with B = (2 mu + cb) I + lam 1 1 (on the normal terms)
    for r in range(6):
        rows = 0.0
        for c in range(3):
            rows += nn[r, c]
        for c in range(6):
            jac[r, c] = dp * ((2.0 * mu + cb) * nn[r, c] + (lam * rows if c < 3 else 0.0))
        jac[r, r] += 1.0
        acc = 0.0
        for c in range(6):
            acc += nn[r, c] * w[c]
        jac[r, 6] = (-nrm[r] - dp * acc) * ddp
    ntr = nrm[0] + nrm[1] + nrm[2]
    nw = 0.0
    for c in range(6):
        jac[6, c] = -((2.0 * mu + cb) * nrm[c] + (lam * ntr if c < 3 else 0.0))
        nw += nrm[c] * w[c]
    jac[6, 6] = (nw - dr) * ddp - (kv if kv > 0.0 else 0.0)
    return jh, dp, ddp


@njit(cache=True)
def _solve_in_place(a, b):
    """``a x = b`` for the columns of ``b``, by Gauss's elimination with partial pivoting
    (``a`` and ``b`` overwritten, ``x`` left in ``b``); False if ``a`` is singular."""
    n = a.shape[0]
    for k in range(n):
        piv = k
        for r in range(k + 1, n):
            if abs(a[r, k]) > abs(a[piv, k]):
                piv = r
        if not abs(a[piv, k]) > 0.0:
            return False
        if piv != k:
            for c in range(n):
                a[k, c], a[piv, c] = a[piv, c], a[k, c]
            for c in range(b.shape[1]):
                b[k, c], b[piv, c] = b[piv, c], b[k, c]
        for r in range(k + 1, n):
            f = a[r, k] / a[k, k]
            if f != 0.0:
                for c in range(k, n):
                    a[r, c] -= f * a[k, c]
                for c in range(b.shape[1]):
                    b[r, c] -= f * b[k, c]
    for k in range(n - 1, -1, -1):
        for c in range(b.shape[1]):
            acc = b[k, c]
            for j in range(k + 1, n):
                acc -= a[k, j] * b[j, c]
            b[k, c] = acc / a[k, k]
    return True


@njit(cache=True)
def _hill_works(count):
    """``count`` sets of the work arrays of :func:`_hill_update`: the vectors, the
    matrices of 6 x 6 and of 7 x 7, the backstresses, the residuals, the right-hand sides
    of the tangent and the Newton step, with one more axis, the first one. The loops over
    the points allocate a set per chunk of points before their parallel loop, and take
    that of each chunk (:func:`_hill_set`): rows of arrays allocated in the loop would
    count their references (:func:`morphogen.homogenization.crystal.work_block`)."""
    return (
        np.empty((count, 9, 6)),
        np.empty((count, 3, 6, 6)),
        np.empty((count, MAX_KINEMATIC, 6)),
        np.empty((count, 7)),
        np.empty((count, 2, 7, 7)),
        np.empty((count, 7, 6)),
        np.empty((count, 7, 1)),
    )


@njit(cache=True)
def _hill_set(works, k):
    """The set ``k`` of the work arrays of :func:`_hill_works` (views)."""
    return (works[0][k], works[1][k], works[2][k], works[3][k], works[4][k], works[5][k],
            works[6][k])  # fmt: skip


@njit(cache=True)
def _hill_work():
    """Work arrays of :func:`_hill_update` (a set of :func:`_hill_works`)."""
    return _hill_set(_hill_works(1), 0)


@njit(cache=True)
def _hill_update(par, nk, eps, eth, dt, old, new, sig, tan, wk, sym=True):
    """Implicit update of a point of Hill's criterion over a step ``dt`` (see
    :class:`Plasticity`): strain ``eps`` (engineering), thermal strain ``eth`` (tensor
    components), work arrays ``wk`` (:func:`_hill_work`). Fills the stress, the new internal
    variables and the tangent (symmetrized with ``sym``); returns 1 if the local Newton
    iteration fails."""
    lam = par[0]
    mu = par[1]
    kv = par[4]
    nv = par[5]
    vec, mat6, xn, res, mat7, rhs, step = wk
    f = vec[0]  # Mandel's factors
    h = vec[1]
    s_tr = vec[2]
    ee = vec[3]
    sr = vec[4]
    nrm = vec[5]
    w = vec[6]
    de = vec[7]
    trial_de = vec[8]
    a = mat6[0]
    nn = mat6[1]
    cm = mat6[2]
    jac = mat7[0]
    work = mat7[1]
    for r in range(6):
        f[r] = 1.0 if r < 3 else _SQ2
        h[r] = par[N_PARAMS + r]
    _hill_matrix(h, a)
    # the elastic trial (Mandel)
    for r in range(6):
        e = eps[r] if r < 3 else 0.5 * eps[r]
        ee[r] = e - eth[r] - old[r]
    tr = ee[0] + ee[1] + ee[2]
    for r in range(6):
        s_tr[r] = f[r] * (2.0 * mu * ee[r] + (lam * tr if r < 3 else 0.0))
    for k in range(nk):
        for r in range(6):
            xn[k, r] = f[r] * old[STATE_X + 6 * k + r]
    for r in range(old.shape[0]):
        new[r] = old[r]
    p0 = old[STATE_P]
    for r in range(6):
        de[r] = 0.0
    jh, dp, _ddp = _hill_system(par, nk, a, s_tr, xn, p0, de, 0.0, dt, res, jac, sr, nrm, w, nn)
    f_tr = res[6] if jh > 0.0 else -1.0
    rr0, dr0 = _hardening(par, p0)
    scale = jh + par[2] + abs(rr0) + 1e-300
    if f_tr <= 1e-13 * scale:
        _elastic_tangent(lam, mu, tan)
        for r in range(6):
            sig[r] = s_tr[r] / f[r]
        return 0
    # first guess: the plastic increment of the elastic predictor along its normal
    stiff = 0.0
    ntr = nrm[0] + nrm[1] + nrm[2]
    for r in range(6):
        stiff += nrm[r] * (2.0 * mu * nrm[r] + (lam * ntr if r < 3 else 0.0))
    for k in range(nk):
        stiff += 2.0 / 3.0 * par[_KIN + 2 * k] * (nrm @ nrm)
    dp = f_tr / (stiff + max(dr0, 0.0))
    q = min((dp / dt) ** (1.0 / nv), f_tr / kv) if kv > 0.0 else dp
    if kv > 0.0:
        dp = dt * q**nv
    for r in range(6):
        de[r] = dp * nrm[r]
    e_scale = f_tr / (3.0 * mu) + 1e-300
    jh, dp, _ddp = _hill_system(par, nk, a, s_tr, xn, p0, de, q, dt, res, jac, sr, nrm, w, nn)
    if jh <= 0.0:
        return 1
    converged = False
    for _it in range(_HILL_IT):
        merit = 0.0
        for r in range(6):
            merit += (res[r] / e_scale) ** 2
        merit += (res[6] / scale) ** 2
        big = 0.0
        for r in range(6):
            big = max(big, abs(res[r]))
        if big <= 1e-12 * e_scale and abs(res[6]) <= 1e-12 * scale:
            converged = True
            break
        for r in range(7):
            step[r, 0] = -res[r]
            for c in range(7):
                work[r, c] = jac[r, c]
        if not _solve_in_place(work, step):
            return 1
        # backtracking on the scaled residuals: the system of the accepted point is that of
        # the next iteration
        alpha = 1.0
        accepted = False
        qt = q
        dpt = dp
        for _ls in range(16):
            qt = q + alpha * step[6, 0]
            if qt < 0.0:
                alpha *= 0.5
                continue
            for r in range(6):
                trial_de[r] = de[r] + alpha * step[r, 0]
            jt, dpt, _ddpt = _hill_system(par, nk, a, s_tr, xn, p0, trial_de, qt, dt, res, jac,
                                        sr, nrm, w, nn)  # fmt: skip
            if jt > 0.0:
                m = 0.0
                for r in range(6):
                    m += (res[r] / e_scale) ** 2
                m += (res[6] / scale) ** 2
                if m <= (1.0 - 1e-4 * alpha) * merit:
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            return 1
        q = qt
        dp = dpt
        for r in range(6):
            de[r] = trial_de[r]
    if not converged or not math.isfinite(dp):
        return 1
    # new state and stress (tensor components)
    tr = de[0] + de[1] + de[2]
    for r in range(6):
        new[r] = old[r] + de[r] / f[r]
        sig[r] = (s_tr[r] - (2.0 * mu * de[r] + (lam * tr if r < 3 else 0.0))) / f[r]
    new[STATE_P] = p0 + dp
    for k in range(nk):
        th = 1.0 / (1.0 + par[_KIN + 2 * k + 1] * dp)
        c = par[_KIN + 2 * k]
        for r in range(6):
            new[STATE_X + 6 * k + r] = th * (xn[k, r] + 2.0 / 3.0 * c * de[r]) / f[r]
    # consistent tangent: jac d(de, q) = -d res / d eps_m, d sigma = C (d eps_m - d de)
    for r in range(6):
        for c in range(6):
            cm[r, c] = (2.0 * mu if r == c else 0.0) + (lam if r < 3 and c < 3 else 0.0)
    for c in range(6):
        for r in range(6):
            acc = 0.0
            for j in range(6):
                acc += nn[r, j] * cm[j, c]
            rhs[r, c] = dp * acc
        acc = 0.0
        for j in range(6):
            acc += nrm[j] * cm[j, c]
        rhs[6, c] = -acc
    for r in range(7):
        for c in range(7):
            work[r, c] = jac[r, c]
    if not _solve_in_place(work, rhs):
        return 1
    for r in range(6):
        for c in range(6):
            acc = cm[r, c]
            for j in range(6):
                acc -= cm[r, j] * rhs[j, c]
            tan[r, c] = acc / (f[r] * f[c])
    if sym:
        symmetrize(tan)
    return 0


@njit(cache=True)
def point_chunks(n):
    """Number of the chunks of consecutive points that the threads integrate, each with
    work arrays of its own: 64 points or more per chunk, 128 chunks at most."""
    return min(max(n // 64, 1), 128)


@njit(parallel=True, cache=True)
def update_points(kind, mat, pmat, data, eps, dtemp, dt, old, new, sigma, tangent, guess, ratio,
                  symmetric=True):  # fmt: skip
    """Stress, internal variables and tangent of every integration point for the strains
    ``eps`` (engineering) at the end of a step ``dt``, from the internal variables
    ``old`` at its start, at the change of temperature ``dtemp``.

    ``data``: the elasticity matrices of the materials (in the axes of the sample) and
    their thermal strains per unit change of temperature (engineering), the rows of the
    von Mises and porous parameters and the number of backstresses, and for crystal
    plasticity the
    slip systems of each material (slip direction, plane normal; axes of the sample), the
    parameters of each system, the interaction matrices and the numbers of systems
    (:func:`morphogen.homogenization.nonlinear.behaviour_table`). Points use the material
    ``mat`` and the row of parameters ``pmat`` of their behaviour ``kind``. ``guess`` and
    ``ratio``: initial slips of the crystal points (:func:`morphogen.homogenization.
    crystal.cp_small`). ``symmetric``: the consistent tangents symmetrized (for the
    conjugate gradients), or kept whole (for a nonsymmetric solver). Returns the number of
    points whose integration failed."""
    matrices, eigen, params, nk, systems, cp_par, cp_h, cp_n = data
    n = eps.shape[0]
    n_chunks = point_chunks(n)
    failed = np.zeros(n_chunks, dtype=np.int64)
    # the work arrays of the laws, a set per chunk of points: allocated before the
    # parallel loop, which holds them as plain memory (crystal.work_block)
    blocks = work_blocks(n_chunks, cp_par.shape[1])
    bufs = np.empty((n_chunks, 7, 6))
    vectors = np.empty((n_chunks, 2, 6))
    hws = _hill_works(n_chunks)
    pws = _porous_works(n_chunks)
    for ch in prange(n_chunks):
        buf = bufs[ch]
        mech = vectors[ch, 0]
        eth = vectors[ch, 1]
        work = work_block(blocks, ch)
        hw = _hill_set(hws, ch)
        pw = _porous_set(pws, ch)
        for i in range(ch * n // n_chunks, (ch + 1) * n // n_chunks):
            m = mat[i]
            k = kind[i]
            if k == MFRONT:
                continue
            if k == ELASTIC:
                dm = matrices[m]
                for r in range(6):
                    mech[r] = eps[i, r] - dtemp * eigen[m, r]
                for r in range(6):
                    s = 0.0
                    for c in range(6):
                        s += dm[r, c] * mech[c]
                        tangent[i, r, c] = dm[r, c]
                    sigma[i, r] = s
                continue
            for r in range(6):
                eth[r] = dtemp * eigen[m, r] * (1.0 if r < 3 else 0.5)
            if k == PLASTIC:
                failed[ch] += _j2_update(params[pmat[i]], nk, eps[i], eth, dt, old[i], new[i],
                                         sigma[i], tangent[i], buf, symmetric)  # fmt: skip
            elif k == HILL:
                failed[ch] += _hill_update(params[pmat[i]], nk, eps[i], eth, dt, old[i],
                                           new[i], sigma[i], tangent[i], hw,
                                           symmetric)  # fmt: skip
            elif k == POROUS:
                failed[ch] += _porous_update(params[pmat[i]], eps[i], eth, dt, old[i], new[i],
                                             sigma[i], tangent[i], pw, symmetric)  # fmt: skip
            else:
                c = pmat[i]
                # (the Schmid tensors of a material are kept from a point to the next)
                failed[ch] += cp_small(systems[m], cp_par[c], cp_h[c], cp_n[c], matrices[m],
                                       eps[i], eth, dt, old[i], new[i], sigma[i], tangent[i],
                                       guess, ratio, work, symmetric, m)  # fmt: skip
    return int(failed.sum())


def mises(stress) -> np.ndarray:
    """Von Mises equivalent stress of Voigt stress components (..., 6)."""
    s = np.asarray(stress, dtype=float)
    mean = s[..., :3].mean(axis=-1, keepdims=True)
    dev = s[..., :3] - mean
    shear = s[..., 3:]
    return np.sqrt(1.5 * (np.sum(dev * dev, axis=-1) + 2.0 * np.sum(shear * shear, axis=-1)))


def state_width(n_kinematic: int) -> int:
    """Number of internal variables of a point with ``n_kinematic`` backstresses."""
    return STATE_X + 6 * int(n_kinematic)
