"""Types shared by the solvers: boundary conditions, loadings, solutions."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np

from morphogen.homogenization.elements import tensor_to_voigt
from morphogen.homogenization.materials import TRANSPORT
from morphogen.homogenization.solvers import SolveInfo

BOUNDARY_CONDITIONS = ("kubc", "subc", "mixed", "periodic")
_ALIASES = {
    "kubc": "kubc",
    "dirichlet": "kubc",
    "kinematic": "kubc",
    "utg": "kubc",
    "subc": "subc",
    "neumann": "subc",
    "static": "subc",
    "uhf": "subc",
    "mixed": "mixed",
    "mbc": "mixed",
    "mtbc": "mixed",
    "periodic": "periodic",
    "pbc": "periodic",
}


def boundary_condition(name: str) -> str:
    """Canonical name of a boundary condition (accepts aliases such as 'pbc', 'utg')."""
    key = str(name).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"unknown boundary condition {name!r}; use one of {BOUNDARY_CONDITIONS} "
            "(aliases: dirichlet, utg, neumann, uhf, mbc, mtbc, pbc)"
        )
    return _ALIASES[key]


#: Names of the Voigt components (elasticity) and of the vector components (conduction).
VOIGT_NAMES = ("11", "22", "33", "23", "31", "12")
VECTOR_NAMES = ("1", "2", "3")
_ALIASES_VOIGT = {"32": "23", "13": "31", "21": "12"}


@dataclass(frozen=True)
class Loading:
    """Macroscopic loading.

    ``kind`` is ``strain`` or ``stress`` (6 Voigt components, order 11 22 33 23 31 12,
    engineering shear strains) for elasticity, ``gradient`` or ``flux`` (3 components)
    for the transport phenomena of a scalar potential (heat or electrical conduction,
    dielectric or magnetic polarization, diffusion), or ``mixed``: component ``i`` is a
    strain (gradient) where ``controls[i]`` is true, a stress (flux) where it is false.
    The components of the macroscopic strain are then unknowns of the problem, dual to
    those of the macroscopic stress (their reactions are ``V Sigma``), as the degrees of
    freedom of the periodic element of Z-set (see :func:`superpose`). ``temperature`` is a
    uniform change of temperature from the stress-free state (elasticity: thermal strains
    of the materials that expand). ``physics``: ``"elastic"`` for the loadings of 6
    components; for those of 3, the transport phenomenon they load
    (:data:`~morphogen.homogenization.materials.TRANSPORT`, ``"thermal"`` by default): a
    problem takes the gradients and fluxes of any of them, and :func:`~morphogen.
    homogenization.apparent.homogenize` solves each named loading with the problem of its
    physics.
    """

    kind: str
    value: tuple[float, ...]
    controls: tuple[bool, ...] | None = None
    temperature: float = 0.0
    physics: str | None = None

    def __post_init__(self) -> None:
        sizes = {"strain": 6, "stress": 6, "gradient": 3, "flux": 3, "mixed": (3, 6)}
        if self.kind not in sizes:
            raise ValueError(f"loading kind: one of {tuple(sizes)}, got {self.kind!r}")
        v = tuple(float(x) for x in np.asarray(self.value, dtype=float).ravel())
        n = sizes[self.kind]
        if (len(v) not in n) if isinstance(n, tuple) else len(v) != n:
            raise ValueError(f"a {self.kind} loading has {n} components")
        if not all(np.isfinite(v)):
            raise ValueError("loading values must be finite")
        object.__setattr__(self, "value", v)
        if self.kind == "mixed":
            if self.controls is None or len(self.controls) != len(v):
                raise ValueError("a mixed loading tells which component is a strain (controls)")
            object.__setattr__(self, "controls", tuple(bool(c) for c in self.controls))
        elif self.controls is not None:
            raise ValueError("controls: mixed loadings only")
        elastic = len(v) == 6
        if self.physics is None:
            object.__setattr__(self, "physics", "elastic" if elastic else "thermal")
        elif self.physics not in (("elastic",) if elastic else tuple(TRANSPORT)):
            allowed = "'elastic'" if elastic else f"one of {tuple(TRANSPORT)}"
            raise ValueError(
                f"physics of a loading of {len(v)} components: {allowed}, got {self.physics!r}"
            )
        t = float(self.temperature)
        if not np.isfinite(t):
            raise ValueError("temperature: a finite change of temperature")
        if t and not elastic:
            raise ValueError("temperature: a change of temperature loads elastic problems")
        object.__setattr__(self, "temperature", t)

    @classmethod
    def strain(cls, value, *, temperature: float = 0.0) -> Loading:
        """Imposed strain: a 3 x 3 tensor or 6 Voigt components (engineering shears)."""
        v = np.asarray(value, dtype=float)
        v = tensor_to_voigt(v, strain=True) if v.shape == (3, 3) else v
        return cls("strain", v, temperature=temperature)

    @classmethod
    def stress(cls, value, *, temperature: float = 0.0) -> Loading:
        v = np.asarray(value, dtype=float)
        v = tensor_to_voigt(v, strain=False) if v.shape == (3, 3) else v
        return cls("stress", v, temperature=temperature)

    @classmethod
    def gradient(cls, value, *, physics: str = "thermal") -> Loading:
        """Imposed gradient of the potential of a transport phenomenon (3 components)."""
        return cls("gradient", value, physics=physics)

    @classmethod
    def flux(cls, value, *, physics: str = "thermal") -> Loading:
        """Imposed flux of a transport phenomenon (3 components; ``flux = -K gradient``)."""
        return cls("flux", value, physics=physics)

    @classmethod
    def mixed(
        cls,
        strain=None,
        stress=None,
        *,
        gradient=None,
        flux=None,
        temperature: float = 0.0,
        physics: str | None = None,
    ) -> Loading:
        """Strain prescribed on some components, stress on the others (or gradient and
        flux): each as a mapping (``{"11": 0.01, "23": 0}``; conduction ``{"1": ...}``)
        or a sequence with ``None`` where the other one is prescribed. Every component
        must be prescribed exactly once. Example, uniaxial tension along 1 with free
        lateral contraction: ``Loading.mixed(strain={"11": 0.01}, stress={"22": 0, "33":
        0, "23": 0, "31": 0, "12": 0})``."""
        elastic = gradient is None and flux is None
        if not elastic and (strain is not None or stress is not None):
            raise ValueError("a mixed loading: strain and stress, or gradient and flux")
        names = VOIGT_NAMES if elastic else VECTOR_NAMES
        first, second = (strain, stress) if elastic else (gradient, flux)
        a, b = _components(first, names), _components(second, names)
        both = [n for n, x, y in zip(names, a, b, strict=True) if x is not None and y is not None]
        neither = [n for n, x, y in zip(names, a, b, strict=True) if x is None and y is None]
        if both or neither:
            raise ValueError(
                f"mixed loading: components {both} prescribed twice, {neither} not prescribed"
            )
        values = [x if x is not None else y for x, y in zip(a, b, strict=True)]
        controls = [x is not None for x in a]
        return cls(
            "mixed", values, controls=tuple(controls), temperature=temperature, physics=physics
        )

    @classmethod
    def uniaxial(cls, axis: int, *, strain=None, stress=None, temperature: float = 0.0):
        """Tension along ``axis``: its strain (or stress) prescribed, the other stress
        components zero (free lateral contraction)."""
        if (strain is None) == (stress is None):
            raise ValueError("uniaxial loading: give the strain or the stress along the axis")
        name = VOIGT_NAMES[int(axis)]
        free = {n: 0.0 for n in VOIGT_NAMES if n != name}
        if strain is not None:
            return cls.mixed(strain={name: strain}, stress=free, temperature=temperature)
        values = [stress if n == name else 0.0 for n in VOIGT_NAMES]
        return cls.stress(values, temperature=temperature)

    @classmethod
    def from_spec(cls, spec) -> Loading:
        """Loading of a configuration mapping: ``strain`` and/or ``stress`` (elasticity)
        or ``gradient`` and/or ``flux`` (a transport phenomenon: ``physics``, thermal by
        default), each a mapping of components or a full list, and ``temperature``.
        Components not given are free: zero stress (flux). ``{"strain": {"11": 0.01}}`` is
        a uniaxial tension, ``{"temperature": 1}`` a free thermal expansion, ``{"physics":
        "electrical", "gradient": {"1": 1}}`` a potential gradient along x, the currents
        across it free."""
        if isinstance(spec, Loading):
            return spec
        if not isinstance(spec, dict):
            raise ValueError(f"a loading is a mapping, got {spec!r}")
        unknown = set(spec) - {"strain", "stress", "gradient", "flux", "temperature", "physics"}
        if unknown:
            raise ValueError(f"loading: unknown keys {sorted(unknown)}")
        physics = spec.get("physics")
        elastic = not ({"gradient", "flux"} & set(spec)) and physics in (None, "elastic")
        if not elastic and ({"strain", "stress"} & set(spec)):
            raise ValueError("a loading: strain and stress, or gradient and flux")
        names = VOIGT_NAMES if elastic else VECTOR_NAMES
        first = _components(spec.get("strain" if elastic else "gradient"), names)
        second = _components(spec.get("stress" if elastic else "flux"), names)
        values, controls = [], []
        for name, a, b in zip(names, first, second, strict=True):
            if a is not None and b is not None:
                raise ValueError(f"loading: component {name} prescribed twice")
            values.append(a if a is not None else (b or 0.0))
            controls.append(a is not None)
        temperature = float(spec.get("temperature", 0.0))
        if all(controls):
            kind = "strain" if elastic else "gradient"
        elif not any(controls):
            kind = "stress" if elastic else "flux"
        else:
            kind = "mixed"
        return cls(
            kind,
            values,
            controls=tuple(controls) if kind == "mixed" else None,
            temperature=temperature,
            physics=physics,
        )

    @classmethod
    def parse(cls, text: str) -> Loading:
        """Loading of a compact text: ``E11=0.01 S22=0 dT=10`` (strain ``E``, stress
        ``S``), ``G1=1 Q2=0`` (gradient, flux; of the transport phenomenon
        ``physics=electrical``, thermal by default); components not given are free."""
        spec: dict = {}
        keys = {"E": "strain", "S": "stress", "G": "gradient", "Q": "flux"}
        for item in text.replace(",", " ").split():
            name, _, value = item.partition("=")
            if not value:
                raise ValueError(f"loading: expected NAME=VALUE, got {item!r}")
            if name.lower() == "physics":
                spec["physics"] = value
            elif name.lower() in ("dt", "temperature"):
                spec["temperature"] = float(value)
            elif name[:1].upper() in keys and name[1:]:
                spec.setdefault(keys[name[:1].upper()], {})[name[1:]] = float(value)
            else:
                raise ValueError(f"loading: unknown component {name!r} (E11, S12, G1, Q3, dT)")
        return cls.from_spec(spec)

    @property
    def array(self) -> np.ndarray:
        return np.array(self.value)

    @property
    def strain_controlled(self) -> np.ndarray:
        """Components prescribed as strains (gradients); the others are stresses (fluxes)."""
        if self.kind == "mixed":
            return np.array(self.controls)
        return np.full(len(self.value), self.kind in ("strain", "gradient"))

    def to_dict(self) -> dict:
        out = {"kind": self.kind, "value": list(self.value)}
        if self.controls is not None:
            out["controls"] = list(self.controls)
        if self.temperature:
            out["temperature"] = self.temperature
        if self.physics not in ("elastic", "thermal"):
            out["physics"] = self.physics
        return out


def _components(spec, names) -> list:
    """Values of the named components (None where not given)."""
    out = [None] * len(names)
    if spec is None:
        return out
    if isinstance(spec, dict):
        for key, value in spec.items():
            name = _ALIASES_VOIGT.get(str(key), str(key))
            if name not in names:
                raise ValueError(f"unknown component {key!r}; use {list(names)}")
            out[names.index(name)] = None if value is None else float(value)
        return out
    values = list(spec)
    if len(values) != len(names):
        raise ValueError(f"expected {len(names)} components (None where not prescribed)")
    return [None if v is None else float(v) for v in values]


def superpose(problem, loading: Loading) -> tuple[np.ndarray, np.ndarray, SolveInfo]:
    """Solution of a loading that mixes the components a boundary condition prescribes
    (``problem.native``: strains, or stresses for static conditions) with their duals.

    The macroscopic native components are the extra unknowns of the problem, dual to the
    others: their reactions are the volume times the averages of the dual field, ``R =
    V Sigma`` for strain components. A linear problem condenses them exactly: the
    responses to unit native loadings (``problem._unit(j)``, cached) and to a unit change
    of temperature give the matrix ``M`` (the apparent stiffness, or compliance) and the
    vector ``r`` of ``dual = M x + r dT``; the unknown components of ``x`` solve the
    prescribed rows, and the field is the sum of the unit fields. Returns ``x``, the field
    and the work done."""
    n = len(loading.value)
    native = problem.native_mask(loading)
    values = loading.array
    dt = loading.temperature
    x = np.where(native, values, 0.0)
    unknown = ~native
    need = np.flatnonzero(unknown | (x != 0))
    units = {int(j): problem._unit(int(j)) for j in need}
    m = np.zeros((n, n))
    for j, (_, response, _) in units.items():
        m[:, j] = response
    r = np.zeros(n)
    thermal = problem._unit(-1) if dt else None
    if thermal is not None:
        r = thermal[1]
    if unknown.any():
        rhs = values[unknown] - m[np.ix_(unknown, native)] @ x[native] - r[unknown] * dt
        x[unknown] = np.linalg.solve(m[np.ix_(unknown, unknown)], rhs)
    parts = [(x[j], units[j][0]) for j in units if x[j] != 0]
    if thermal is not None:
        parts.append((dt, thermal[0]))
    state = sum((c * f for c, f in parts[1:]), parts[0][0] * parts[0][1]) if parts else None
    infos = [u[2] for u in units.values()] + ([thermal[2]] if thermal is not None else [])
    info = SolveInfo(
        infos[0].method if infos else "none",
        infos[0].n_dofs if infos else 0,
        sum(i.iterations for i in infos),
        max((i.residual for i in infos), default=0.0),
        sum(i.setup_s for i in infos),
        0.0,
    )
    return x, state, info


@dataclass
class PhaseStatistics:
    """Volume fraction, means and standard deviations of the fields in one phase."""

    phase: int
    fraction: float
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "fraction": self.fraction,
            "mean": {k: np.asarray(v).tolist() for k, v in self.mean.items()},
            "std": {k: np.asarray(v).tolist() for k, v in self.std.items()},
        }


@dataclass
class Solution:
    """Volume averages over the box of one solved problem.

    Elasticity: ``strain`` is the macroscopic strain ``E`` (imposed, or the average
    strain under SUBC), ``stress`` the average stress ``Sigma`` (voids count as zero),
    both 3 x 3 tensors, and ``energy`` the average of ``sigma : eps`` (twice the
    elastic energy density). Conduction, and every transport phenomenon of a scalar
    potential (``physics``: :data:`~morphogen.homogenization.materials.TRANSPORT`):
    ``strain`` is the macroscopic gradient ``G``, ``stress`` the average flux ``Q``
    (``q = -k grad T``) and ``energy`` the average of ``k grad T . grad T``.
    """

    physics: str
    bc: str
    loading: Loading
    strain: np.ndarray
    stress: np.ndarray
    energy: float
    material_fraction: float
    phases: list[PhaseStatistics]
    info: SolveInfo
    #: local fields (with ``fields=True``): element means of strain and stress (or
    #: gradient and flux) and energy density, plus the nodal displacement (temperature)
    #: of finite element solutions
    fields: dict[str, np.ndarray] | None = field(default=None, repr=False)
    #: the mesh (TetMesh, porous phases removed) or image (VoxelImage) of the fields
    support: object | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "physics": self.physics,
            "bc": self.bc,
            "loading": self.loading.to_dict(),
            "strain": self.strain.tolist(),
            "stress": self.stress.tolist(),
            "energy": self.energy,
            "material_fraction": self.material_fraction,
            "phases": [p.to_dict() for p in self.phases],
            "info": self.info.to_dict(),
        }


class MacroscopicLoadings:
    """Solution of macroscopic loadings by a problem that solves for the components its
    boundary condition prescribes (``_field(x, dt)``) and evaluates the averages of a
    field (``_dual``, ``_evaluate``); other components are condensed by
    :func:`superpose`."""

    physics = ""
    bc = ""

    @property
    def native(self) -> str:
        """What the boundary condition prescribes: strains (gradients), or stresses
        (fluxes) for static conditions."""
        if self.bc == "subc":
            return "stress" if self.physics == "elastic" else "flux"
        return "strain" if self.physics == "elastic" else "gradient"

    def native_mask(self, loading: Loading) -> np.ndarray:
        """Components of a loading that the boundary condition prescribes directly."""
        if (loading.physics == "elastic") != (self.physics == "elastic"):
            raise ValueError(f"the loading is {loading.physics}, the problem {self.physics}")
        strain = loading.strain_controlled
        return strain if self.native in ("strain", "gradient") else ~strain

    def _unit(self, j: int):
        """Field, dual response and work of the unit native loading ``j`` (``-1``: a
        unit change of temperature), cached."""
        if j not in self._units:
            x = np.zeros(6 if self.physics == "elastic" else 3)
            dt = 0.0
            if j >= 0:
                x[j] = 1.0
            else:
                dt = 1.0
            state, info = self._field(x, dt)
            self._units[j] = (state, self._dual(state, x, dt), info)
        return self._units[j]

    def solve(self, loading: Loading, *, fields: bool = False) -> Solution:
        """Solve for a macroscopic loading: what the boundary condition prescribes
        (strains, or stresses under ``subc``), or any mix of strain and stress components
        under ``kubc``, ``periodic`` and ``subc``, by condensation of the macroscopic
        unknowns (:func:`superpose`)."""
        t0 = time.perf_counter()
        native = self.native_mask(loading)
        if self.empty:
            return self._empty_solution(loading, t0)
        dt = loading.temperature
        if native.all():
            x = loading.array
            unit = np.flatnonzero(x)
            if dt == 0.0 and unit.size == 1 and x[unit[0]] == 1.0:  # e.g. tensors: cached
                state, _, info = self._unit(int(unit[0]))
                info = dataclasses.replace(info)
            elif dt == 1.0 and not x.any():
                state, _, info = self._unit(-1)
                info = dataclasses.replace(info)
            else:
                state, info = self._field(x, dt)
        else:
            if self.bc == "mixed":
                raise ValueError(
                    "mixed boundary conditions prescribe a strain (gradient): stress or mixed "
                    "loadings need kubc, periodic or subc conditions"
                )
            x, state, info = superpose(self, loading)
        return self._evaluate(loading, state, x, dt, info, fields, t0)
