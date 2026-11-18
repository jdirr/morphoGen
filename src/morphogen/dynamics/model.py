"""The model of an explicit dynamic simulation of a morphology: the unit system of its
inputs (:class:`UnitSystem`), the materials of its phases (:class:`ExplicitMaterial`),
its loading (:class:`ExplicitLoading`), its mesh (:func:`explicit_mesh`) and the stable
time step of that mesh (:func:`critical_time_step`), from which the cost of a run is
known before it starts.

OpenRadioss has no unit system of its own: the values are given in a consistent system,
which the Starter input names (``/BEGIN``) so that the solver knows it, e.g. Mg, mm and
s: stresses in MPa, densities in Mg/mm³ (7.85e-9 for a steel), velocities in mm/s,
forces in N and energies in mJ. :meth:`UnitSystem.from_si` converts values given in SI
units (a density of 7850 kg/m³, a speed of 10 m/s) into the system."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The units of mass, length and time that the Starter of Radioss names (``/BEGIN``),
#: by their value in kg, m and s
MASS_UNITS = {"Mg": 1e3, "kg": 1.0, "g": 1e-3, "mg": 1e-6, "ug": 1e-9}
LENGTH_UNITS = {"m": 1.0, "dm": 1e-1, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
#: The names of the derived units of the usual systems, by their value in SI units
_NAMES = {
    "stress": {1.0: "Pa", 1e3: "kPa", 1e6: "MPa", 1e9: "GPa", 1e12: "TPa"},
    "force": {1e-6: "µN", 1e-3: "mN", 1.0: "N", 1e3: "kN", 1e6: "MN"},
    "energy": {1e-9: "nJ", 1e-6: "µJ", 1e-3: "mJ", 1.0: "J", 1e3: "kJ"},
    "velocity": {1e-6: "µm/s", 1e-3: "mm/s", 1.0: "m/s", 1e3: "km/s"},
}
#: The kinds of loading of :class:`ExplicitLoading`
LOADINGS = ("compression", "tension", "shear", "impact")
AXES = "xyz"


@dataclass(frozen=True)
class UnitSystem:
    """A consistent system of units of mass, length and time, named as Radioss names
    them (:data:`MASS_UNITS`, :data:`LENGTH_UNITS`, :data:`TIME_UNITS`): every value of
    the inputs is given in it, stresses in mass / (length time²), densities in mass /
    length³, velocities in length / time."""

    mass: str = "Mg"
    length: str = "mm"
    time: str = "s"

    def __post_init__(self) -> None:
        for name, value, known in (
            ("mass", self.mass, MASS_UNITS),
            ("length", self.length, LENGTH_UNITS),
            ("time", self.time, TIME_UNITS),
        ):
            if value not in known:
                raise ValueError(f"units: the {name} in {', '.join(known)}, got {value!r}")

    @classmethod
    def parse(cls, value) -> UnitSystem:
        """A system from three names (``"Mg mm s"``, ``("kg", "mm", "ms")``) or itself."""
        if isinstance(value, UnitSystem):
            return value
        if value in (None, ""):
            return cls()
        names = value.replace(",", " ").split() if isinstance(value, str) else list(value)
        if len(names) != 3:
            raise ValueError(f"units: the units of mass, length and time, got {value!r}")
        return cls(*[str(n) for n in names])

    @property
    def names(self) -> tuple[str, str, str]:
        return (self.mass, self.length, self.time)

    def si(self, quantity: str) -> float:
        """The value in SI units of the unit of ``quantity`` in this system: ``mass``,
        ``length``, ``time``, ``density``, ``stress``, ``velocity``, ``force``,
        ``energy`` or ``strain_rate``."""
        m, l, t = MASS_UNITS[self.mass], LENGTH_UNITS[self.length], TIME_UNITS[self.time]
        values = {
            "mass": m,
            "length": l,
            "time": t,
            "density": m / l**3,
            "stress": m / (l * t * t),
            "velocity": l / t,
            "force": m * l / (t * t),
            "energy": m * l * l / (t * t),
            "strain_rate": 1.0 / t,
        }
        if quantity not in values:
            raise ValueError(f"quantity: one of {', '.join(values)}")
        return values[quantity]

    def from_si(self, value, quantity: str):
        """``value`` given in SI units, in this system (a number or an array)."""
        factor = self.si(quantity)
        return np.asarray(value, dtype=float) / factor if np.ndim(value) else float(value) / factor

    def to_si(self, value, quantity: str):
        """``value`` given in this system, in SI units (a number or an array)."""
        factor = self.si(quantity)
        return np.asarray(value, dtype=float) * factor if np.ndim(value) else float(value) * factor

    def unit(self, quantity: str) -> str:
        """The name of the unit of ``quantity`` (e.g. ``"MPa"``; ``"Mg/mm³"``)."""
        if quantity in ("mass", "length", "time"):
            return getattr(self, quantity)
        if quantity == "density":
            return f"{self.mass}/{self.length}³"
        if quantity == "strain_rate":
            return f"1/{self.time}"
        value = self.si(quantity)
        for known, name in _NAMES.get(quantity, {}).items():
            if math.isclose(value, known, rel_tol=1e-9):
                return name
        return {
            "stress": f"{self.mass}/({self.length} {self.time}²)",
            "velocity": f"{self.length}/{self.time}",
            "force": f"{self.mass} {self.length}/{self.time}²",
            "energy": f"{self.mass} {self.length}²/{self.time}²",
        }[quantity]

    def describe(self) -> str:
        """The system and its derived units, in words."""
        return (
            f"{self.mass}, {self.length}, {self.time}: stresses in {self.unit('stress')}, "
            f"densities in {self.unit('density')}, velocities in {self.unit('velocity')}, "
            f"forces in {self.unit('force')}, energies in {self.unit('energy')}"
        )


# -- materials -----------------------------------------------------------------------------


@dataclass(frozen=True)
class JohnsonCook:
    """The yield stress of Johnson and Cook, ``(a + b p^n) (1 + c ln(ṗ / ṗ0))`` (p the
    plastic strain), Radioss's LAW2 without its terms of temperature: ``a`` the yield
    stress, ``b`` and ``n`` the hardening, ``c`` the sensitivity to the strain rate above
    the reference rate ``rate`` (ṗ0, in 1 / time), ``max_stress`` a cap of the yield
    stress (none by default)."""

    a: float
    b: float = 0.0
    n: float = 1.0
    c: float = 0.0
    rate: float = 1.0
    max_stress: float | None = None

    def __post_init__(self) -> None:
        values = {"a": self.a, "b": self.b, "n": self.n, "c": self.c, "rate": self.rate}
        for name, value in values.items():
            if not (math.isfinite(float(value)) and float(value) >= 0.0):
                raise ValueError(f"johnson_cook: {name} >= 0, got {value!r}")
            object.__setattr__(self, name, float(value))
        if self.a <= 0.0:
            raise ValueError("johnson_cook: a, the yield stress, > 0")
        if self.n <= 0.0:
            raise ValueError("johnson_cook: n > 0")
        if self.c > 0.0 and self.rate <= 0.0:
            raise ValueError("johnson_cook: a reference strain rate > 0 with c > 0")
        if self.max_stress is not None:
            if not float(self.max_stress) > self.a:
                raise ValueError("johnson_cook: max_stress above a")
            object.__setattr__(self, "max_stress", float(self.max_stress))

    def yield_stress(self, strain, rate: float = 0.0) -> np.ndarray:
        """The yield stress at the plastic strains ``strain`` and the strain rate."""
        p = np.asarray(strain, dtype=float)
        out = self.a + self.b * p**self.n
        if self.c > 0.0 and rate > self.rate:
            out = out * (1.0 + self.c * math.log(rate / self.rate))
        if self.max_stress is not None:
            out = np.minimum(out, self.max_stress)
        return out

    @classmethod
    def from_config(cls, spec) -> JohnsonCook:
        """``{a, b, n, c, rate, max_stress}`` (aliases A, B, C, ``reference_rate``)."""
        if isinstance(spec, JohnsonCook):
            return spec
        if not isinstance(spec, dict):
            raise ValueError("johnson_cook: a mapping {a, b, n, c, rate}")
        aliases = {"A": "a", "B": "b", "C": "c", "N": "n", "reference_rate": "rate"}
        spec = {aliases.get(k, k): v for k, v in spec.items()}
        unknown = sorted(set(spec) - {"a", "b", "n", "c", "rate", "max_stress"})
        if unknown:
            raise ValueError(f"johnson_cook: unknown keys {unknown}")
        if "a" not in spec:
            raise ValueError("johnson_cook: a, the yield stress")
        return cls(**spec)


#: The strain rates (in 1 / time) of the yield curves of a viscoplastic material (Norton's
#: overstress tabulated at each, LAW36 interpolating between them)
NORTON_RATES = (0.0, *(10.0**k for k in range(-4, 7)))
#: The plastic strains of the tabulated yield curves (denser near 0, where the
#: saturating terms bend)
CURVE_STRAINS = tuple(round(2.0 * (k / 40.0) ** 2, 10) for k in range(41))


@dataclass(frozen=True)
class ExplicitMaterial:
    """The material of a phase in an explicit dynamic simulation: isotropic elasticity
    (``young``, ``poisson``) and its ``density`` (mass / length³), elastic (Radioss's
    LAW1), or elastoplastic: von Mises with the isotropic hardening of morphoGen's
    :class:`~morphogen.homogenization.behaviours.Plasticity` (the yield stress R0, the
    linear hardening H, the terms Q (1 - exp(-b p)); Norton's viscosity K ṗ^(1/n) as
    yield curves at several strain rates, :data:`NORTON_RATES`), tabulated in LAW36, or
    the law of Johnson and Cook (:class:`JohnsonCook`, LAW2). ``failure_strain``: the
    plastic strain at which an element fails and is deleted (none by default)."""

    young: float
    poisson: float
    density: float
    plasticity: object | None = None
    johnson_cook: JohnsonCook | None = None
    failure_strain: float | None = None

    def __post_init__(self) -> None:
        e, nu, rho = float(self.young), float(self.poisson), float(self.density)
        if not (math.isfinite(e) and e > 0):
            raise ValueError(f"young: > 0, got {self.young!r}")
        if not -1.0 < nu < 0.5:
            raise ValueError(f"poisson: in (-1, 0.5), got {self.poisson!r}")
        if not (math.isfinite(rho) and rho > 0):
            raise ValueError(
                f"density: > 0, got {self.density!r} (explicit dynamics needs the mass)"
            )
        object.__setattr__(self, "young", e)
        object.__setattr__(self, "poisson", nu)
        object.__setattr__(self, "density", rho)
        if self.plasticity is not None:
            from morphogen.homogenization.behaviours import Plasticity

            plastic = Plasticity.from_config(self.plasticity)
            if plastic.kinematic:
                raise ValueError(
                    "plasticity: kinematic hardening has no equivalent among the laws "
                    "written for OpenRadioss (isotropic hardening only)"
                )
            if plastic.hill is not None:
                raise ValueError("plasticity: Hill's criterion is not written for OpenRadioss")
            object.__setattr__(self, "plasticity", plastic)
        if self.johnson_cook is not None:
            object.__setattr__(self, "johnson_cook", JohnsonCook.from_config(self.johnson_cook))
            if self.plasticity is not None:
                raise ValueError("give plasticity or johnson_cook, not both")
        if self.failure_strain is not None:
            eps = float(self.failure_strain)
            if not (math.isfinite(eps) and eps > 0):
                raise ValueError(f"failure_strain: > 0, got {self.failure_strain!r}")
            if self.plasticity is None and self.johnson_cook is None:
                raise ValueError("failure_strain: a plastic strain, which needs a plastic law")
            object.__setattr__(self, "failure_strain", eps)

    @property
    def law(self) -> str:
        """The law of Radioss that it is written as: LAW1, LAW2 or LAW36."""
        if self.johnson_cook is not None:
            return "LAW2"
        return "LAW36" if self.plasticity is not None else "LAW1"

    @property
    def wave_speed(self) -> float:
        """The speed of the longitudinal (dilatational) waves, sqrt(M / ρ), M the
        P-wave modulus."""
        e, nu = self.young, self.poisson
        modulus = e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return math.sqrt(modulus / self.density)

    def yield_curve(self, strains=None, rate: float = 0.0) -> np.ndarray:
        """The yield stress at the plastic strains ``strains`` (by default
        :data:`CURVE_STRAINS`) and the plastic strain rate ``rate`` (its plastic law)."""
        p = np.asarray(CURVE_STRAINS if strains is None else strains, dtype=float)
        if self.johnson_cook is not None:
            return self.johnson_cook.yield_stress(p, rate)
        plastic = self.plasticity
        if plastic is None:
            raise ValueError("an elastic material has no yield curve")
        out = plastic.yield_stress + plastic.hardening * p
        for q, b in plastic.isotropic:
            out = out + q * (1.0 - np.exp(-b * p))
        if plastic.viscous and rate > 0.0:
            out = out + plastic.viscosity * rate ** (1.0 / plastic.exponent)
        return out

    def rates(self) -> tuple[float, ...]:
        """The strain rates of its tabulated yield curves: several for a viscoplastic
        material, one (0) otherwise."""
        viscous = self.plasticity is not None and self.plasticity.viscous
        return NORTON_RATES if viscous else (0.0,)

    def to_config(self) -> dict:
        out: dict = {"young": self.young, "poisson": self.poisson, "density": self.density}
        if self.plasticity is not None:
            out["plasticity"] = self.plasticity.to_config()
        if self.johnson_cook is not None:
            jc = self.johnson_cook
            out["johnson_cook"] = {"a": jc.a, "b": jc.b, "n": jc.n, "c": jc.c, "rate": jc.rate}
            if jc.max_stress is not None:
                out["johnson_cook"]["max_stress"] = jc.max_stress
        if self.failure_strain is not None:
            out["failure_strain"] = self.failure_strain
        return out

    @classmethod
    def from_config(cls, spec, *, density: float | None = None) -> ExplicitMaterial | None:
        """A material from its configuration: the keys of morphoGen's materials
        (:meth:`~morphogen.homogenization.Material.from_config`: ``young`` and ``poisson``
        or ``bulk`` and ``shear``, ``density``, ``plasticity``), ``johnson_cook``
        (:meth:`JohnsonCook.from_config`) and ``failure_strain``; ``"void"`` (or a void
        material) gives None. ``density``: the density of a material that gives none.
        Its thermal properties are not used."""
        from morphogen.homogenization.materials import Material

        if isinstance(spec, ExplicitMaterial):
            return spec
        extra: dict = {}
        if isinstance(spec, dict):
            spec = dict(spec)
            for key in ("johnson_cook", "failure_strain"):
                if key in spec:
                    extra[key] = spec.pop(key)
        material = Material.from_config(spec)
        if material.void:
            if extra:
                raise ValueError("a void has no plastic law nor failure")
            return None
        if material.young is None:
            raise ValueError(
                "explicit dynamics: isotropic elasticity (young and poisson, or bulk and shear)"
            )
        for name in ("crystal_plasticity", "porous_plasticity", "mfront", "learned"):
            if getattr(material, name) is not None:
                raise ValueError(f"{name}: not written for OpenRadioss")
        rho = material.density if material.density is not None else density
        if rho is None:
            raise ValueError("density: explicit dynamics needs the density of each material")
        return cls(material.young, material.poisson, rho, plasticity=material.plasticity, **extra)


def explicit_materials(spec, *, density: float | None = None) -> dict[int, ExplicitMaterial | None]:
    """The material of each phase (None for the voids) from a mapping phase -> material
    (:meth:`ExplicitMaterial.from_config`; the phases may be strings)."""
    if not isinstance(spec, dict) or not spec:
        raise ValueError("materials: a mapping phase -> material")
    out: dict[int, ExplicitMaterial | None] = {}
    for key, value in spec.items():
        try:
            phase = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"materials: phases are integers, got {key!r}") from None
        try:
            out[phase] = ExplicitMaterial.from_config(value, density=density)
        except ValueError as exc:
            raise ValueError(f"materials[{phase}]: {exc}") from None
    return out


# -- loading -------------------------------------------------------------------------------


def _axis(value, name: str = "axis") -> int:
    if isinstance(value, str) and value.lower() in AXES:
        return AXES.index(value.lower())
    try:
        axis = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: x, y, z (or 0, 1, 2), got {value!r}") from None
    if axis not in (0, 1, 2):
        raise ValueError(f"{name}: x, y, z (or 0, 1, 2), got {value!r}")
    return axis


@dataclass(frozen=True)
class ExplicitLoading:
    """The loading of a morphology in an explicit dynamic simulation.

    * ``compression`` and ``tension``: the lower face normal to ``axis`` held along it,
      the upper face driven along it at the nominal strain rate ``strain_rate`` (in
      1 / time) up to the nominal ``strain``;
    * ``shear``: the lower face held, the upper face driven along ``shear_axis`` (by
      default the next axis) at the rate ``strain_rate`` up to the engineering shear
      ``strain``, held along the other axes;
    * ``impact``: the morphology thrown at ``velocity`` (length / time) along -``axis``
      against a fixed rigid wall at its lower face, for ``duration`` (by default the
      time to travel ``strain`` times its length at that velocity).

    The driven velocities grow smoothly from 0 over the share ``ramp`` of the duration,
    which tempers the first waves. ``lateral``: ``fixed`` (the loaded faces held across
    the axis, as by bonded platens) or ``free`` (they slide freely across it, a node of
    the lower face held against drifting); ``contact``: the self
    contact of the outer surface of the solid (folding struts, cells closing; Radioss's
    TYPE24) with the ``friction`` coefficient."""

    kind: str = "compression"
    axis: int = 2
    strain: float = 0.2
    strain_rate: float = 100.0
    shear_axis: int | None = None
    velocity: float = 0.0
    duration: float | None = None
    ramp: float = 0.05
    lateral: str = "fixed"
    contact: bool = True
    friction: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in LOADINGS:
            raise ValueError(f"loading: one of {', '.join(LOADINGS)}, got {self.kind!r}")
        object.__setattr__(self, "axis", _axis(self.axis))
        shear = self.shear_axis
        if self.kind == "shear":
            shear = (self.axis + 1) % 3 if shear is None else _axis(shear, "shear_axis")
            if shear == self.axis:
                raise ValueError("shear_axis: another axis than the normal of the faces")
        object.__setattr__(self, "shear_axis", shear)
        strain = float(self.strain)
        if not (math.isfinite(strain) and strain > 0.0):
            raise ValueError(f"strain: > 0, got {self.strain!r}")
        if self.kind == "compression" and strain >= 1.0:
            raise ValueError("strain: a nominal compression below 1")
        object.__setattr__(self, "strain", strain)
        rate = float(self.strain_rate)
        if self.kind != "impact" and not (math.isfinite(rate) and rate > 0.0):
            raise ValueError(f"strain_rate: > 0, got {self.strain_rate!r}")
        object.__setattr__(self, "strain_rate", rate)
        v = float(self.velocity)
        if self.kind == "impact" and not (math.isfinite(v) and v > 0.0):
            raise ValueError("velocity: the speed of the impact, > 0")
        object.__setattr__(self, "velocity", v)
        if self.duration is not None:
            t = float(self.duration)
            if not (math.isfinite(t) and t > 0.0):
                raise ValueError(f"duration: > 0, got {self.duration!r}")
            object.__setattr__(self, "duration", t)
        ramp = float(self.ramp)
        if not 0.0 <= ramp < 1.0:
            raise ValueError(f"ramp: a share of the duration in [0, 1), got {self.ramp!r}")
        object.__setattr__(self, "ramp", ramp)
        if self.lateral not in ("free", "fixed"):
            raise ValueError("lateral: free or fixed")
        friction = float(self.friction)
        if not (math.isfinite(friction) and friction >= 0.0):
            raise ValueError("friction: >= 0")
        object.__setattr__(self, "friction", friction)
        object.__setattr__(self, "contact", bool(self.contact))

    @property
    def driven(self) -> int:
        """The axis along which the upper face is driven."""
        return self.shear_axis if self.kind == "shear" else self.axis

    def speed(self, lengths) -> float:
        """The speed of the upper face once the ramp is over (that of the impact)."""
        if self.kind == "impact":
            return self.velocity
        return self.strain_rate * float(np.asarray(lengths, dtype=float)[self.axis])

    def sign(self) -> float:
        """The direction of the motion along :attr:`driven`: -1 for a compression and
        an impact (toward the lower face), +1 otherwise."""
        return -1.0 if self.kind in ("compression", "impact") else 1.0

    def end_time(self, lengths) -> float:
        """The duration of the simulation: the nominal strain reached at the rate after
        the ramp (whose smooth step travels half as far as at full speed)."""
        if self.duration is not None:
            return self.duration
        length = float(np.asarray(lengths, dtype=float)[self.axis])
        if self.kind == "impact":
            return self.strain * length / self.velocity
        return self.strain / (self.strain_rate * (1.0 - 0.5 * self.ramp))

    def velocity_function(self, end_time: float, points: int = 11) -> np.ndarray:
        """The velocity of the upper face over time, a share of :meth:`speed`: the smooth
        step 3 s² - 2 s³ (s = t / t_ramp) over the ramp, then 1 (to ten times the end,
        so that the solver never extrapolates); rows (t, share)."""
        t_ramp = self.ramp * end_time
        if t_ramp <= 0.0:
            return np.array([[0.0, 1.0], [10.0 * end_time, 1.0]])
        s = np.linspace(0.0, 1.0, points)
        rows = [(t_ramp * x, 3.0 * x**2 - 2.0 * x**3) for x in s]
        rows.append((10.0 * end_time, 1.0))
        return np.array(rows, dtype=float)

    def displacement(self, times, lengths) -> np.ndarray:
        """The displacement of the upper face along :attr:`driven` at ``times`` (signed;
        the impact: the distance travelled at the initial velocity)."""
        t = np.asarray(times, dtype=float)
        v = self.sign() * self.speed(lengths)
        if self.kind == "impact":
            return v * t
        t_ramp = self.ramp * self.end_time(lengths)
        if t_ramp <= 0.0:
            return v * t
        s = np.clip(t / t_ramp, 0.0, 1.0)
        within = t_ramp * (s**3 - 0.5 * s**4)  # the integral of 3 s² - 2 s³
        return v * np.where(t <= t_ramp, within, 0.5 * t_ramp + (t - t_ramp))

    def to_config(self) -> dict:
        out = {"kind": self.kind, "axis": AXES[self.axis]}
        if self.kind == "impact":
            out["velocity"] = self.velocity
        else:
            out["strain_rate"] = self.strain_rate
        out["strain"] = self.strain
        if self.kind == "shear":
            out["shear_axis"] = AXES[self.shear_axis]
        if self.duration is not None:
            out["duration"] = self.duration
        out.update(
            ramp=self.ramp, lateral=self.lateral, contact=self.contact, friction=self.friction
        )
        return out

    @classmethod
    def from_config(cls, spec) -> ExplicitLoading:
        """A loading from its configuration (the fields, axes as x, y, z)."""
        if isinstance(spec, ExplicitLoading):
            return spec
        if spec in (None, ""):
            return cls()
        if isinstance(spec, str):
            return cls(kind=spec)
        if not isinstance(spec, dict):
            raise ValueError("loading: a mapping {kind, axis, strain, strain_rate, ...}")
        known = {
            "kind",
            "axis",
            "strain",
            "strain_rate",
            "shear_axis",
            "velocity",
            "duration",
            "ramp",
            "lateral",
            "contact",
            "friction",
        }
        unknown = sorted(set(spec) - known)
        if unknown:
            raise ValueError(f"loading: unknown keys {unknown} (known: {', '.join(sorted(known))})")
        return cls(**spec)


# -- meshes and time steps -----------------------------------------------------------------


def explicit_mesh(
    micro,
    materials: dict,
    *,
    mesh: str = "voxels",
    resolution: int = 24,
    element_size: float | None = None,
    order: int = 1,
):
    """The mesh of a morphology for an explicit simulation, of its phases that are not
    voids (``materials``: phase -> :class:`ExplicitMaterial` or None): ``"voxels"``,
    8-node hexahedra of its voxel image of ``resolution`` voxels along its longest edge
    (:func:`morphogen.mesh.voxel.voxel_mesh`), or ``"conforming"``, tetrahedra of
    ``order`` 1 or 2 conforming to its level set, ``element_size`` the largest
    (:func:`morphogen.mesh.conforming_mesh`; 1/12 of the longest edge by default)."""
    voids = {int(q) for q, m in materials.items() if m is None}
    if mesh == "voxels":
        from morphogen.mesh.voxel import voxel_mesh

        image = micro.voxelize(shape=int(resolution))
        present = {int(q) for q in np.unique(image.data)}
        missing = sorted(present - set(materials))
        if missing:
            raise ValueError(f"phases {missing}: no material (give theirs, or 'void')")
        solid = sorted(present - voids)
        if not solid:
            raise ValueError("every phase of the image is void")
        labels = micro.labels(grid=image.grid()) if getattr(micro, "has_labels", False) else None
        return voxel_mesh(image, solid if voids & present else None, labels=labels)
    if mesh != "conforming":
        raise ValueError("mesh: voxels or conforming")
    from morphogen.mesh import conforming_mesh

    edge = float(np.max(micro.domain.lengths))
    size = float(element_size) if element_size else edge / 12.0
    phases = "solid" if 0 in voids and 1 not in voids else "all"
    tets = conforming_mesh(micro, size=size, order=int(order), phases=phases)
    present = {int(q) for q in np.unique(tets.phase)}
    missing = sorted(present - set(materials))
    if missing:
        raise ValueError(f"phases {missing}: no material (give theirs, or 'void')")
    if present & voids:
        if not present - voids:
            raise ValueError("every phase of the mesh is void")
        tets = tets.subset(sorted(present - voids))
    return tets


def element_lengths(mesh) -> np.ndarray:
    """The characteristic length of each element of a mesh: the smallest edge of the
    voxels of a voxel mesh; the smallest altitude (3 V / the largest face) of the
    tetrahedra, halved for the quadratic ones (whose nodes lie twice as close)."""
    elements = np.asarray(mesh.elements)
    if elements.shape[1] == 8:
        return np.full(len(elements), float(np.min(mesh.spacing)))
    nodes = np.asarray(mesh.nodes, dtype=float)
    x = nodes[elements[:, :4]]
    volume = (
        np.abs(
            np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])
        )
        / 6.0
    )
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    area = np.max(
        [
            0.5 * np.linalg.norm(np.cross(x[:, b] - x[:, a], x[:, c] - x[:, a]), axis=1)
            for a, b, c in faces
        ],
        axis=0,
    )
    lengths = 3.0 * volume / np.maximum(area, 1e-300)
    return lengths if elements.shape[1] == 4 else 0.5 * lengths


def critical_time_step(mesh, materials: dict, *, scale: float = 0.9) -> float:
    """The stable time step of an explicit integration of a mesh, about: ``scale`` times
    the smallest time of a longitudinal wave across an element (its characteristic
    length, :func:`element_lengths`, over the wave speed of its material). Radioss
    computes its own, a little smaller with its bulk viscosity."""
    lengths = element_lengths(mesh)
    phase = np.asarray(mesh.phase)
    speed = np.zeros(len(phase))
    for q in np.unique(phase):
        material = materials.get(int(q))
        if material is None:
            raise ValueError(f"phase {int(q)}: no material")
        speed[phase == q] = material.wave_speed
    return float(scale * np.min(lengths / speed))
