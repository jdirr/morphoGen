"""Materials of laser-architectured sheets: flow laws from tensile properties, the
strength of each hardness, and the steels of the PIMM studies.

**Flow law** (:class:`FlowLaw`). The true stress of the equivalent plastic strain,
Ludwik's ``σ(p) = σ0 + K((p + δ)^n − δ^n)`` (``δ = 10⁻⁴`` keeps the initial slope
finite) or Voce's ``σ0 + Q(1 − exp(−bp))``. Given the 0.2 % proof stress ``R_p`` (the
stress at a plastic strain of 0.2 %), the engineering tensile strength ``R_m`` and the
uniform elongation ``A_g`` (engineering) of a tensile test, Considère's condition
``dσ/dp = σ`` at the true plastic uniform strain ``p_u = ln(1 + A_g) − σ_u/E``, with
``σ(p_u) = σ_u = R_m (1 + A_g)``, fixes the two other parameters
(:func:`flow_parameters`): Ludwik's law when an exponent fits, Voce's otherwise (the
large strength ratios of martensite at small elongations).

**Strength of a hardness** (:class:`StrengthTable`). The tensile properties of the
states of a material (base, treated at several energies, carburized...) against their
Vickers hardness, measured; between them linear interpolation, beyond the last rows the
correlations of Pavlina & Van Tyne (2008) for steels (``R_p = −90.7 + 2.876 HV``,
``R_m = −99.8 + 3.734 HV``, MPa) shifted to meet the table, and the uniform elongation
extrapolated linearly (at least 2 %). Each point of a treated sheet gets the flow law
of its hardness.

**Presets** (:data:`PRESETS`, :func:`sheet_material`):

* ``aisi430``: ferritic stainless steel AISI 430, 0.5 mm (Wang et al. 2022): C 0.026,
  Cr 18.7; base 160 HV, R_p 346 MPa, R_m 503 MPa, A_g 19.4 %; fully treated by the double
  laser in argon at 15 and 20 kJ/m: 211.5 and 241.8 HV, R_p 489 and 540 MPa, R_m 624 and
  703 MPa, A_g 15.5 and 13.7 % (the table); carburized (Wang 2022, thesis ch. 4): +130 %
  HV, R_p +90 %, R_m +85 % (A_g assumed 10 %). Hardening by peak temperature (onset Ac1
  = 1148 K, 310 HV near Tm = 1698 K for a single track), its gain at 1300 °C and its
  tempering fitted to the mean hardness of the sheets fully treated by ten tracks at
  1 mm pitch (211.5, 241.8 and 233.6 HV at 15, 20 and 25 kJ/m; computed 212, 240 and
  235).
* ``annealed_ms1500``: MS1500 annealed at 850 °C for 1 h (ferrite–pearlite), 1 mm
  (Wang 2022, thesis ch. 5): C 0.228, Si 0.242, Mn 1.73, Cr 0.19; 167 HV, R_p 300 MPa,
  R_m 534 MPa, A_g 16.7 %; laser martensite as the as-received MS1500 (about 480 HV,
  R_p 1300 MPa, R_m 1600 MPa, A_g 4 %; Lapouge et al. 2019, Wang 2020).
* ``dp1180``: dual-phase DP1180, 1 mm (Lapouge et al. 2019): 85 % martensite, 380 HV,
  R_p 850 MPa, R_m 1200 MPa, A_g 5 %; tempered at Ac1, R_m 816 MPa (about 255 HV).
  Laser tempering: soft tracks in a hard sheet.

Thermal properties: AISI 430 with the mean of the Eurocode stainless-steel laws used by
Wang et al. (conductivity 14.6 + 0.0127 θ, specific heat about 560 J/kg/K over the
cycle), the C–Mn steels with Lapouge et al.'s (28 W/m/K, 560 J/kg/K, 7860 kg/m³);
absorptivity 0.42 under argon (Wang et al. 2022), 200 W/m²/K on the faces (gas jets).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from morphogen.laser.heat import ThermalProperties
from morphogen.laser.metallurgy import (
    PeakHardening,
    PeakSoftening,
    Tempering,
    TransformationHardening,
    model_from_config,
)

#: Offset of the Ludwik law (keeps its initial slope finite).
LUDWIK_OFFSET = 1e-4
#: Kinds of flow laws: Ludwik's power law, Voce's saturating law, Swift's (Hollomon's law
#: from the yield stress).
LAWS = ("ludwik", "voce", "swift")


@dataclass(frozen=True)
class FlowLaw:
    """True stress (MPa) of the equivalent plastic strain: ``ludwik``,
    ``σ0 + a((p + δ)^b − δ^b)`` (``a = K``, ``b = n``), ``voce``,
    ``σ0 + a(1 − exp(−b p))`` (``a = Q``, ``b`` the rate), or ``swift``,
    ``K (ε0 + p)^n`` (``a = K``, ``b = n``, ``ε0 = (σ0/K)^(1/n)``: Hollomon's ``σ = K ε^n``
    from the yield stress ``σ0``, as Wang et al. 2023 for their phases)."""

    kind: str
    sigma0: float
    a: float
    b: float
    offset: float = LUDWIK_OFFSET

    def __post_init__(self):
        if self.kind not in LAWS:
            raise ValueError(f"flow law: one of {LAWS}")

    @property
    def code(self) -> int:
        """The index of the kind in the kernels."""
        return LAWS.index(self.kind)

    @classmethod
    def hollomon(cls, yield_strength: float, k: float, n: float) -> FlowLaw:
        """Hollomon's ``σ = K ε^n`` beyond the yield stress (a Swift law)."""
        return cls("swift", float(yield_strength), float(k), float(n))

    def stress(self, p):
        p = np.asarray(p, dtype=float)
        if self.kind == "ludwik":
            d = self.offset
            return self.sigma0 + self.a * ((p + d) ** self.b - d**self.b)
        if self.kind == "swift":
            return self.a * ((self.sigma0 / self.a) ** (1.0 / self.b) + p) ** self.b
        return self.sigma0 + self.a * (1.0 - np.exp(-self.b * p))

    def slope(self, p):
        p = np.asarray(p, dtype=float)
        if self.kind == "ludwik":
            return self.b * self.a * (p + self.offset) ** (self.b - 1.0)
        if self.kind == "swift":
            q = (self.sigma0 / self.a) ** (1.0 / self.b) + p
            return self.b * self.a * q ** (self.b - 1.0)
        return self.a * self.b * np.exp(-self.b * p)

    @classmethod
    def from_tensile(
        cls,
        yield_strength: float,
        tensile_strength: float,
        uniform_elongation: float,
        young: float = 210000.0,
    ) -> FlowLaw:
        """The law of a tensile test (see the module): Considère at the uniform
        elongation."""
        kind, s0, a, b = flow_parameters(
            yield_strength, tensile_strength, uniform_elongation, young=young
        )
        return cls(LAWS[int(kind[0])], float(s0[0]), float(a[0]), float(b[0]))

    def necking(self) -> tuple[float, float]:
        """True plastic strain at Considère's condition (rigid-plastic) and the
        engineering tensile strength of the law."""
        lo, hi = 1e-7, 5.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.slope(mid) > self.stress(mid):
                lo = mid
            else:
                hi = mid
        p = 0.5 * (lo + hi)
        return p, float(self.stress(p) * math.exp(-p))


def _bisect(fun, lo, hi, iterations: int = 60):
    """Vectorized bisection of a decreasing function on ``[lo, hi]``."""
    lo, hi = lo.copy(), hi.copy()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        up = fun(mid) > 0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def flow_parameters(
    yield_strength,
    tensile_strength,
    uniform_elongation,
    *,
    young: float = 210000.0,
    offset: float = LUDWIK_OFFSET,
    strict: bool = True,
):
    """Flow laws of tensile properties, vectorized: ``(kind, σ0, a, b)`` (kind 0:
    Ludwik, 1: Voce). The law gives the proof stress at a plastic strain of 0.2 %,
    ``σ(0.002) = R_p``, and at the true plastic strain ``p_u = ln(1 + A_g) − σ_u/E`` of
    the uniform elongation ``σ(p_u) = σ_u = R_m (1 + A_g)`` and ``dσ/dp = σ_u``
    (Considère). Ludwik: ``K = σ_u (p_u + δ)^(1−n)/n``, ``n`` and ``σ0`` by nested
    bisections; where no exponent fits (a large ratio ``R_m/R_p`` at a small elongation,
    martensite), Voce: ``x/(e^x − 1) = σ_u p_u/(σ_u − σ0)`` with ``x = b p_u`` and
    ``Q = (σ_u − σ0)/(1 − e^(−x))``. ``strict``: raise on properties that no law fits
    (else their parameters are NaN)."""
    rp = np.atleast_1d(np.asarray(yield_strength, dtype=float))
    rm = np.atleast_1d(np.asarray(tensile_strength, dtype=float))
    ag = np.atleast_1d(np.asarray(uniform_elongation, dtype=float))
    rp, rm, ag = (np.array(v, dtype=float) for v in np.broadcast_arrays(rp, rm, ag))
    bad = (ag <= 0) | (rm * (1.0 + ag) <= rp)
    if strict and bad.any():
        raise ValueError("tensile properties: A_g > 0 and R_m (1 + A_g) > R_p")
    ag = np.where(bad, 0.1, ag)
    rm = np.where(bad, 2.0 * rp, rm)
    su = rm * (1.0 + ag)
    pu = np.maximum(np.log1p(ag) - su / young, 1e-3)
    steep = su * (1.0 - pu) <= rp
    if strict and steep.any():
        raise ValueError(
            "tensile properties: a hardening exponent above 1 (R_p too high "
            "for the strength and elongation)"
        )
    bad |= steep
    d = float(offset)
    q = pu + d
    n_lo, n_hi = np.full_like(su, 1e-3), np.full_like(su, 1.0)

    def ludwik(s0):
        """Exponent, modulus and whether a root exists, for the initial stresses s0."""
        f = lambda n: s0 + su / n * (q - d**n * q ** (1.0 - n)) - su  # noqa: E731
        ok = (f(n_lo) > 0) & (f(n_hi) < 0)
        n = _bisect(f, n_lo, n_hi)
        return n, su * q ** (1.0 - n) / n, ok

    def g_ludwik(s0):  # the proof stress minus R_p: increasing in s0
        n, k, ok = ludwik(s0)
        return np.where(ok, rp - (s0 + k * ((0.002 + d) ** n - d**n)), -1.0)

    s_lo, s_hi = 0.2 * rp, rp.copy()
    s0_l = _bisect(g_ludwik, s_lo, s_hi)
    n, k, ok = ludwik(s0_l)
    ok &= np.abs(g_ludwik(s0_l)) < 1e-6 * rp

    def voce(s0):
        r = su * pu / (su - s0)
        x = _bisect(
            lambda x: x / np.expm1(x) - r, np.full_like(su, 1e-6), np.full_like(su, 60.0), 80
        )
        return x / pu, (su - s0) / (-np.expm1(-x))

    def g_voce(s0):
        b, qv = voce(s0)
        return rp - (s0 + qv * (-np.expm1(-b * 0.002)))

    s0_v = _bisect(g_voce, s_lo, s_hi)
    b_v, q_v = voce(s0_v)
    bad |= ~ok & (np.abs(g_voce(s0_v)) > 1e-3 * rp)
    if strict and bad.any():
        raise ValueError("tensile properties: no Ludwik or Voce law fits them")
    kind = np.where(ok, 0, 1).astype(np.int8)
    out = [np.where(ok, s0_l, s0_v), np.where(ok, k, q_v), np.where(ok, n, b_v)]
    out = [np.where(bad, np.nan, v) for v in out]
    return (kind, *out)


@dataclass
class StrengthTable:
    """Tensile properties against hardness: rows ``(HV, R_p, R_m, A_g)`` (MPa, MPa,
    engineering fraction), interpolated linearly, extrapolated by Pavlina & Van Tyne's
    slopes (see the module)."""

    rows: tuple

    def __post_init__(self):
        tab = np.asarray(self.rows, dtype=float).reshape(-1, 4)
        tab = tab[np.argsort(tab[:, 0])]
        if len(tab) < 1 or np.any(np.diff(tab[:, 0]) <= 0):
            raise ValueError("strength table: rows of distinct hardnesses")
        self.rows = tuple(map(tuple, tab))
        self._t = tab

    def properties(self, hv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``R_p``, ``R_m`` and ``A_g`` of hardnesses (arrays)."""
        hv = np.asarray(hv, dtype=float)
        t = self._t
        if len(t) == 1:
            rp = t[0, 1] + 2.876 * (hv - t[0, 0])
            rm = t[0, 2] + 3.734 * (hv - t[0, 0])
            ag = np.full_like(hv, t[0, 3])
        else:
            rp = np.interp(hv, t[:, 0], t[:, 1])
            rm = np.interp(hv, t[:, 0], t[:, 2])
            ag = np.interp(hv, t[:, 0], t[:, 3])
            lo, hi = hv < t[0, 0], hv > t[-1, 0]
            rp = np.where(lo, t[0, 1] + 2.876 * (hv - t[0, 0]), rp)
            rm = np.where(lo, t[0, 2] + 3.734 * (hv - t[0, 0]), rm)
            rp = np.where(hi, t[-1, 1] + 2.876 * (hv - t[-1, 0]), rp)
            rm = np.where(hi, t[-1, 2] + 3.734 * (hv - t[-1, 0]), rm)
            slope = (t[-1, 3] - t[-2, 3]) / (t[-1, 0] - t[-2, 0])
            ag = np.where(hi, t[-1, 3] + slope * (hv - t[-1, 0]), ag)
        ag = np.clip(ag, 0.02, 0.6)
        rp = np.maximum(rp, 50.0)
        rm = np.maximum(rm, 1.02 * rp / (1.0 + ag) + 1.0)
        return rp, rm, ag

    def flow(self, hv, young: float = 210000.0) -> tuple:
        """Flow laws ``(kind, σ0, a, b)`` of hardnesses (see :func:`flow_parameters`),
        those of the nearest hardness of a grid of 0.5 HV (cached)."""
        hv = np.asarray(hv, dtype=float)
        key = float(young)
        cache = self.__dict__.setdefault("_flow_cache", {})
        if key not in cache:
            lo = max(50.0, 0.6 * self._t[0, 0])
            hi = max(1.6 * self._t[-1, 0], self._t[-1, 0] + 200.0)
            grid = np.arange(lo, hi + 1e-9, 0.5)
            params = flow_parameters(*self.properties(grid), young=young, strict=False)
            valid = np.isfinite(params[1])
            if not valid.any():
                raise ValueError("strength table: no flow law fits its properties")
            near = np.flatnonzero(valid)[
                np.clip(
                    np.searchsorted(np.flatnonzero(valid), np.arange(len(grid))), 0, valid.sum() - 1
                )
            ]
            cache[key] = (lo, tuple(v[near] for v in params))
        lo, params = cache[key]
        i = np.clip(np.rint((hv - lo) / 0.5).astype(np.int64), 0, len(params[0]) - 1)
        return tuple(v[i] for v in params)

    @classmethod
    def from_config(cls, data) -> StrengthTable:
        if isinstance(data, StrengthTable):
            return data
        return cls(tuple(tuple(row) for row in data))


_YLD_FITS: dict = {}


def _yld2000_coefficients(material) -> tuple:
    """α1 ... α8 of the Yld2000-2d texture of a material (fitted once per data)."""
    from morphogen.laser.mechanics import yld2000_from_data

    a = material.anisotropy
    key = tuple(
        float(a[k]) if k in a else None
        for k in ("s0", "s45", "s90", "sb", "r0", "r45", "r90", "rb", "a")
    )
    if key not in _YLD_FITS:
        missing = [k for k in ("s0", "s45", "s90", "sb", "r0", "r45", "r90") if k not in a]
        if missing:
            raise ValueError(f"anisotropy yld2000: give {missing}")
        _YLD_FITS[key] = yld2000_from_data(
            a["s0"],
            a["s45"],
            a["s90"],
            a["sb"],
            a["r0"],
            a["r45"],
            a["r90"],
            a.get("rb", a["r0"] / a["r90"]),
            a.get("a", 6.0),
        )[:8]
    return _YLD_FITS[key]


@dataclass
class SheetMaterial:
    """A sheet: its thermal properties, its metallurgical response to the cycles of the
    tracks, its strength against hardness, its elasticity (MPa), its density (kg/m³), its
    usual thickness (mm) and absorptivity, its rolling texture (``anisotropy``: the
    Lankford coefficients ``{"r0", "r45", "r90"}`` of Hill's 1948 criterion, or with
    ``"criterion": "yld2000"`` the yield stresses ``"s0", "s45", "s90", "sb"`` and the
    coefficients ``"r0", "r45", "r90", "rb"`` of Barlat's Yld2000-2d and its exponent
    ``"a"``; the share ``"memory"`` of the texture the treated points keep;
    :meth:`yield_criterion`),
    the ``fracture`` of its hard points (``{"hardness":
    HV, "critical": plastic strain, "radius": mm}``: the points at least that hard break
    at that plastic strain, averaged over the radius; :class:`~morphogen.laser.mechanics.Fracture`)
    and the sources of the data."""

    name: str
    thermal: ThermalProperties
    metallurgy: TransformationHardening | PeakHardening
    strength: StrengthTable
    young: float = 210000.0
    poisson: float = 0.3
    thickness: float = 1.0
    absorptivity: float = 0.42
    description: str = ""
    references: tuple = ()
    extra: dict = field(default_factory=dict)
    fracture: dict | None = None
    anisotropy: dict | None = None

    def yield_criterion(self, treated, angle: float = 0.0) -> np.ndarray:
        """The yield criterion of points of treated share ``treated``: Hill's 1948
        (``(..., 5)``, :meth:`hill`) or, with ``anisotropy["criterion"] = "yld2000"``,
        Barlat's Yld2000-2d (``(..., 10)``: α1 ... α8 fitted to the data of the sheet,
        :func:`~morphogen.laser.mechanics.yld2000_from_data`, the exponent ``a``, 6 by
        default, and the angle of the rolling direction): the texture where untreated,
        the isotropic coefficients (α = 1, the same exponent) where the laser
        transformed the steel, or the share ``memory`` of the texture's departure from
        them, mixed in proportion between."""
        a = self.anisotropy or {}
        if a.get("criterion", "hill48") == "hill48":
            return self.hill(treated, angle)
        if a["criterion"] != "yld2000":
            raise ValueError("anisotropy criterion: hill48 or yld2000")
        coef = np.asarray(_yld2000_coefficients(self), dtype=float)
        expo = float(a.get("a", 6.0))
        t = np.clip(np.asarray(treated, dtype=float), 0.0, 1.0)[..., None]
        treated_coef = 1.0 + float(a.get("memory", 0.0)) * (coef - 1.0)
        al = (1.0 - t) * coef + t * treated_coef
        tail = np.empty((*al.shape[:-1], 2))
        tail[..., 0] = expo
        tail[..., 1] = np.radians(angle)
        return np.concatenate([al, tail], axis=-1)

    def hill(self, treated, angle: float = 0.0) -> np.ndarray:
        """Hill's 1948 coefficients and the angle of the rolling direction of points of
        treated share ``treated`` (``(..., 5)``, see
        :func:`~morphogen.laser.mechanics.hill_from_r`): the sheet's texture
        (``anisotropy``: ``{"r0", "r45", "r90"}``, the Lankford coefficients of the
        rolled sheet) where it is untreated; where the laser transformed it, isotropic
        (the new phases form with a weak texture) or, with ``"memory"`` ``m`` in
        ``anisotropy``, the isotropic coefficients plus ``m`` times the texture's
        departure from them (1: the texture inherited whole, as when the treatment only
        dissolves precipitates); mixed in proportion between; the rolling direction at
        ``angle`` degrees from the axis of the specimen."""
        from morphogen.laser.mechanics import ISOTROPIC, hill_from_r

        t = np.clip(np.asarray(treated, dtype=float), 0.0, 1.0)[..., None]
        iso = np.asarray(ISOTROPIC[:4])
        if not self.anisotropy:
            base = treated_coef = iso
        else:
            a = self.anisotropy
            base = np.asarray(hill_from_r(a["r0"], a["r45"], a["r90"])[:4])
            treated_coef = iso + float(a.get("memory", 0.0)) * (base - iso)
        coef = (1.0 - t) * base + t * treated_coef
        ang = np.full((*coef.shape[:-1], 1), np.radians(angle))
        return np.concatenate([coef, ang], axis=-1)

    def critical_strain(self, hardness):
        """The critical plastic strain of points of the given hardness (None: ductile
        sheet): ``fracture["hardness"]`` and ``["critical"]`` are a threshold and its
        strain, or tables between which the brittleness (the inverse of the critical
        strain) varies linearly (a partly transformed zone is less brittle)."""
        if not self.fracture:
            return None
        hv = np.asarray(hardness, dtype=float)
        h = np.atleast_1d(np.asarray(self.fracture["hardness"], dtype=float))
        c = np.atleast_1d(np.asarray(self.fracture["critical"], dtype=float))
        if h.size == 1:
            return np.where(hv >= h[0], c[0], np.inf)
        with np.errstate(divide="ignore"):
            brittle = np.interp(hv, h, 1.0 / c, left=0.0)
            return np.where(brittle > 0, 1.0 / np.where(brittle > 0, brittle, 1.0), np.inf)

    @property
    def density(self) -> float:
        return self.thermal.density

    @property
    def base_hardness(self) -> float:
        """Hardness of the untreated sheet."""
        m = self.metallurgy
        return float(m.hardness(m.initial_state(1))[0])

    def with_(self, **changes) -> SheetMaterial:
        return replace(self, **changes)

    @classmethod
    def from_config(cls, data) -> SheetMaterial:
        """A preset name, or a mapping: ``preset`` (optional, the defaults) and any of
        the fields (``thermal``, ``metallurgy`` with its ``kind``, ``strength`` rows...)."""
        if isinstance(data, SheetMaterial):
            return data
        if isinstance(data, str):
            return sheet_material(data)
        data = dict(data)
        base = sheet_material(data.pop("preset")) if "preset" in data else None
        if "thermal" in data:
            th = data["thermal"]
            if base is not None and isinstance(th, dict):
                th = replace(base.thermal, **th)
            data["thermal"] = ThermalProperties.from_config(th)
        if "metallurgy" in data:
            met = data["metallurgy"]
            if base is not None and isinstance(met, dict) and "kind" not in met:
                kind = (
                    "softening"
                    if isinstance(base.metallurgy, PeakSoftening)
                    else "peak"
                    if isinstance(base.metallurgy, PeakHardening)
                    else "transformation"
                )
                params = {
                    k: getattr(base.metallurgy, k)
                    for k in type(base.metallurgy).__dataclass_fields__
                }
                params.update(met)
                met = {"kind": kind, **params}
            data["metallurgy"] = model_from_config(met)
        if "strength" in data:
            data["strength"] = StrengthTable.from_config(data["strength"])
        if base is not None:
            return replace(base, **data)
        missing = {"name", "thermal", "metallurgy", "strength"} - set(data)
        if missing:
            raise ValueError(f"material: missing {sorted(missing)} (or give a preset)")
        return cls(**data)


def _aisi430() -> SheetMaterial:
    return SheetMaterial(
        name="aisi430",
        thermal=ThermalProperties(
            conductivity=23.5, density=7700.0, heat_capacity=560.0, convection=200.0
        ),
        metallurgy=PeakHardening(
            table=((875.0, 0.0), (1082.0, 40.0), (1300.0, 110.0), (1425.0, 150.0), (1700.0, 150.0)),
            base_hardness=160.0,
            solidus=1425.0,
            composition={"C": 0.026, "Si": 0.30, "Mn": 0.50, "Cr": 18.7, "Ni": 0.14},
            carbon_hardening=560.0,
            tempering=Tempering(start=8300.0, end=26800.0, floor=0.0),
        ),
        strength=StrengthTable(
            (
                (160.0, 346.0, 502.9, 0.1939),
                (211.5, 488.7, 623.5, 0.1552),
                (241.8, 539.7, 702.5, 0.1365),
                (370.0, 657.0, 930.0, 0.10),
            )
        ),
        thickness=0.5,
        absorptivity=0.42,
        description="ferritic stainless steel AISI 430 (18.7 Cr), hardened by martensite "
        "and Cr carbides near its melting point; carburized from a graphite coating",
        references=(
            "Wang, Dirrenberger, Lapouge, Dubent (2022) Mater. Sci. Eng. A 831, 142205",
            "Wang et al. (2022) J. Eng. Mater. Technol. 144, 041005",
            "Wang (2022) PhD thesis, HESAM Université (ENSAM, PIMM)",
        ),
    )


def _annealed_ms1500() -> SheetMaterial:
    comp = {"C": 0.228, "Si": 0.242, "Mn": 1.730, "Cr": 0.190}
    return SheetMaterial(
        name="annealed_ms1500",
        thermal=ThermalProperties(
            conductivity=28.0, density=7860.0, heat_capacity=560.0, convection=200.0
        ),
        metallurgy=TransformationHardening(
            composition=comp,
            superheat=50.0,
            solidus=1480.0,
            martensite=0.0,
            matrix_hardness=167.0,
            critical_time=3.0,
            slow_time=30.0,
            tempering=Tempering(constant=20.0, start=9000.0, end=20000.0, floor=0.0, minimum=190.0),
        ),
        # the martensite of the tracks: Hollomon's law identified by Wang et al. (2023,
        # K 1966 MPa, n 0.06), uniform elongation from Considère (the 0.036 of the bulk
        # martensite tested alone is its fracture, which the matrix delays)
        strength=StrengthTable(
            (
                (167.0, 300.0, 534.0, 0.167),
                (480.0, 1300.0, 1560.0, 0.062),
            )
        ),
        thickness=1.0,
        absorptivity=0.42,
        description="MS1500EZ annealed at 850 °C for 1 h (ferrite-pearlite, 0.23 C, 1.7 Mn), "
        "hardened by laser tracks of martensite",
        references=(
            "Wang, Bouaziz, Dirrenberger, Lapouge (2023) steel res. int. 94, 2200695",
            "Lapouge et al. (2019) Mater. Sci. Eng. A 752, 128-135",
            "Maynier, Jungmann, Dollet (1978) Hardenability concepts with applications to "
            "steel, AIME, 518-545",
        ),
    )


def _dp1180() -> SheetMaterial:
    comp = {"C": 0.15, "Si": 0.2, "Mn": 1.8}
    return SheetMaterial(
        name="dp1180",
        thermal=ThermalProperties(
            conductivity=28.0, density=7860.0, heat_capacity=560.0, convection=200.0
        ),
        metallurgy=TransformationHardening(
            composition=comp,
            ac1=727.0,
            ac3=827.0,
            superheat=50.0,
            solidus=1480.0,
            martensite=0.85,
            martensite_hardness=420.0,
            matrix_hardness=165.0,
            critical_time=3.0,
            slow_time=30.0,
            tempering=Tempering(constant=20.0, start=9000.0, end=20000.0, floor=0.0, minimum=200.0),
        ),
        strength=StrengthTable(
            (
                (165.0, 300.0, 480.0, 0.20),
                (255.0, 600.0, 816.0, 0.08),
                (380.0, 850.0, 1200.0, 0.05),
                (480.0, 1300.0, 1600.0, 0.04),
            )
        ),
        thickness=1.0,
        absorptivity=0.42,
        description="dual-phase DP1180 (85 % martensite), softened by laser tempering",
        references=("Lapouge et al. (2019) Mater. Sci. Eng. A 752, 128-135",),
    )


def _22mnb5() -> SheetMaterial:
    comp = {"C": 0.22, "Si": 0.25, "Mn": 1.25, "Cr": 0.2}
    return SheetMaterial(
        name="22mnb5",
        thermal=ThermalProperties(
            conductivity=34.0, density=7850.0, heat_capacity=650.0, convection=200.0
        ),
        metallurgy=TransformationHardening(
            composition=comp,
            superheat=50.0,
            solidus=1480.0,
            martensite=0.0,
            matrix_hardness=185.0,
            critical_time=3.0,
            slow_time=30.0,
            tempering=Tempering(constant=20.0, start=9000.0, end=20000.0, floor=0.0, minimum=200.0),
        ),
        # as delivered (75 % ferrite, 25 % pearlite: R_p about 400, R_m 450-550 MPa,
        # uniform elongation 18 %) and quenched (martensite up to 605 HV10, R_m
        # 1590-1600 MPa; its uniform elongation 3.25 % measured alone, 0.06 from
        # Considère as for the annealed MS1500): Karbasian & Tekkaya (2010), Merklein et
        # al. (2006)
        strength=StrengthTable(
            (
                (185.0, 400.0, 520.0, 0.18),
                (550.0, 1150.0, 1590.0, 0.06),
            )
        ),
        thickness=1.5,
        absorptivity=0.4,
        description="boron steel 22MnB5 as delivered (ferrite and pearlite), hardened by "
        "laser tracks of martensite (the steel of hot stamping; laser-patterned tubes of "
        "Kim & Yoon 2021)",
        references=(
            "Karbasian H., Tekkaya A.E. (2010) J. Mater. Process. Technol. 210, 2103-2118",
            "Kim H.-g., Yoon J. (2021) Compos. Struct. 262, 113633",
        ),
    )


def _aa6082_t6() -> SheetMaterial:
    return SheetMaterial(
        name="aa6082_t6",
        thermal=ThermalProperties(
            conductivity=170.0, density=2700.0, heat_capacity=950.0, convection=200.0
        ),
        # T6 (105 HV); overaged near 350-400 °C (about 60 HV, no ageing again);
        # dissolved above about 480 °C then naturally aged (55 HV just after, about
        # 80 HV after days): Piccininni & Palumbo (2020), Myhr & Grong (1991)
        metallurgy=PeakSoftening(
            base_hardness=105.0,
            solidus=585.0,
            table=(
                (200.0, 0.0),
                (250.0, -8.0),
                (300.0, -25.0),
                (350.0, -40.0),
                (400.0, -45.0),
                (450.0, -38.0),
                (500.0, -25.0),
                (600.0, -25.0),
            ),
        ),
        # the solution-treated state just after the laser (yield about 70 MPa), the
        # naturally aged one (T4-like) and T6
        strength=StrengthTable(
            (
                (55.0, 70.0, 170.0, 0.20),
                (80.0, 150.0, 240.0, 0.15),
                (105.0, 290.0, 330.0, 0.07),
            )
        ),
        young=70000.0,
        poisson=0.33,
        thickness=1.0,
        absorptivity=0.3,
        description="age-hardened aluminium alloy AA6082-T6 softened by laser tracks "
        "(the precipitates dissolved above about 250 °C; tailor heat-treated blanks)",
        references=(
            "Piccininni A., Palumbo G. (2020) Materials 13, 1576",
            "Myhr O.R., Grong O. (1991) Acta Metall. Mater. 39, 2693-2702",
        ),
    )


#: The materials of the PIMM studies (see the module) and alternative ones.
PRESETS = {
    "aisi430": _aisi430,
    "annealed_ms1500": _annealed_ms1500,
    "dp1180": _dp1180,
    "22mnb5": _22mnb5,
    "aa6082_t6": _aa6082_t6,
}


def sheet_material(name: str) -> SheetMaterial:
    """A preset material (:data:`PRESETS`)."""
    if name not in PRESETS:
        raise ValueError(f"material: one of {sorted(PRESETS)}, got {name!r}")
    return PRESETS[name]()
