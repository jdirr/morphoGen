"""Loading paths: macroscopic loadings that vary in time, and the responses to them.

A path gives, at increasing times, the value of each macroscopic component, prescribed
as a strain or as a stress (the other one is then an unknown, solved for), and the
change of temperature; values vary linearly in time between them. The first time is
the natural state: zero strain, stress and change of temperature. Uniaxial tension at
a strain rate, creep under a stress, relaxation, cycles or thermal histories are
paths.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from morphogen.homogenization.core import _ALIASES, _ALIASES_VOIGT, VOIGT_NAMES


def _index(key) -> int:
    """Voigt index of a component name (``"11"`` ... ``"12"``, or ``"21"`` ...)."""
    name = _ALIASES_VOIGT.get(str(key), str(key))
    if name not in VOIGT_NAMES:
        raise ValueError(f"unknown component {key!r}; use {list(VOIGT_NAMES)}")
    return VOIGT_NAMES.index(name)


def _series(value, n_times: int, name: str) -> list[float]:
    """Values of a component at the times: a list (the first one may be omitted, it is
    zero), or one value reached at the end of the first segment and held."""
    if np.ndim(value) == 0:
        v = float(value)
        return [0.0] + [v] * (n_times - 1)
    values = [float(v) for v in value]
    if len(values) == n_times - 1:
        values = [0.0, *values]
    if len(values) != n_times:
        raise ValueError(f"{name}: one value per time ({n_times}), or one value held")
    return values


@dataclass(frozen=True)
class LoadingPath:
    """History of a macroscopic loading of elasticity.

    At the ``times`` (increasing; the first one is the natural state), ``values`` gives
    the six Voigt components (11, 22, 33, 23, 31, 12; engineering shear strains),
    prescribed as strains where ``controls`` is true and as stresses elsewhere, and
    ``temperature`` the change of temperature. Each segment between two times is
    divided into ``increments`` steps (the solver divides a step further when it does
    not converge).
    """

    times: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    controls: tuple[bool, ...]
    temperature: tuple[float, ...] = ()
    increments: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        times = tuple(float(t) for t in self.times)
        if len(times) < 2 or not all(math.isfinite(t) for t in times):
            raise ValueError("times: at least two finite times")
        if any(b <= a for a, b in itertools.pairwise(times)):
            raise ValueError("times: strictly increasing")
        values = np.asarray(self.values, dtype=float)
        if values.shape != (len(times), 6) or not np.all(np.isfinite(values)):
            raise ValueError("values: six finite components at each time")
        if np.any(values[0] != 0):
            raise ValueError("a path starts from the natural state: zero values at the first time")
        controls = tuple(bool(c) for c in self.controls)
        if len(controls) != 6:
            raise ValueError("controls: one per component (true: strain, false: stress)")
        temps = tuple(float(t) for t in self.temperature) or (0.0,) * len(times)
        if len(temps) != len(times) or not all(math.isfinite(t) for t in temps):
            raise ValueError("temperature: one finite value per time")
        if temps[0] != 0.0:
            raise ValueError("temperature: zero change at the first time (natural state)")
        incs = self.increments or (10,)
        incs = tuple(int(i) for i in (incs * (len(times) - 1) if len(incs) == 1 else incs))
        if len(incs) != len(times) - 1 or min(incs) < 1:
            raise ValueError("increments: a positive number per segment (or one for all)")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", tuple(tuple(row) for row in values.tolist()))
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "temperature", temps)
        object.__setattr__(self, "increments", incs)

    # -- construction -------------------------------------------------------------------

    @classmethod
    def from_spec(cls, spec) -> LoadingPath:
        """Path of a configuration mapping: ``times``, ``increments`` (one per segment, or
        one for all), ``strain`` and ``stress`` (component -> values at the times, or one
        value reached at the end of the first segment and held; the first value, zero,
        may be omitted) and ``temperature`` (likewise). Components not given are free:
        zero stress. ``{"times": [0, 10], "strain": {"11": 0.02}}`` is a uniaxial tension
        at the strain rate 0.002."""
        if isinstance(spec, LoadingPath):
            return spec
        if not isinstance(spec, dict):
            raise ValueError(f"a path is a mapping, got {spec!r}")
        unknown = set(spec) - {"times", "increments", "strain", "stress", "temperature"}
        if unknown:
            raise ValueError(f"path: unknown keys {sorted(unknown)}")
        if "times" not in spec:
            raise ValueError("path: give the times")
        times = [float(t) for t in spec["times"]]
        n = len(times)

        def series(key):
            raw = spec.get(key) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"path.{key}: a mapping component -> values")
            return {_index(k): _series(v, n, f"path.{key}.{k}") for k, v in raw.items()}

        strains, stresses = series("strain"), series("stress")
        twice = sorted(set(strains) & set(stresses))
        if twice:
            raise ValueError(f"path: components {[VOIGT_NAMES[i] for i in twice]} given twice")
        values = np.zeros((n, 6))
        controls = [False] * 6
        for i, v in strains.items():
            values[:, i] = v
            controls[i] = True
        for i, v in stresses.items():
            values[:, i] = v
        temperature = spec.get("temperature")
        temps = () if temperature is None else tuple(_series(temperature, n, "path.temperature"))
        incs = spec.get("increments", 10)
        incs = (int(incs),) if np.ndim(incs) == 0 else tuple(int(i) for i in incs)
        return cls(times, values, controls, temps, incs)

    @classmethod
    def parse(cls, text: str) -> LoadingPath:
        """Path of a compact text: ``t=0,10,20 n=10 E11=0,0.02,0 S12=0 dT=0,50,50``: the
        times ``t``, the increments ``n`` (one, or one per segment), strain ``E`` and
        stress ``S`` components and the change of temperature ``dT`` at the times (or one
        value reached at the end of the first segment and held); the components not given
        are free (zero stress)."""
        spec: dict = {}
        for item in text.replace(";", " ").split():
            name, _, value = item.partition("=")
            if not value:
                raise ValueError(f"path: expected NAME=VALUES, got {item!r}")
            numbers = [float(v) for v in value.split(",") if v.strip()]
            one = numbers[0] if len(numbers) == 1 else numbers
            if name == "t":
                spec["times"] = numbers
            elif name == "n":
                spec["increments"] = [int(v) for v in numbers] if len(numbers) > 1 else int(one)
            elif name.lower() in ("dt", "temperature"):
                spec["temperature"] = one
            elif name[:1] in ("E", "S") and name[1:]:
                key = "strain" if name[0] == "E" else "stress"
                spec.setdefault(key, {})[name[1:]] = one
            else:
                raise ValueError(f"path: unknown item {name!r} (t, n, E11, S12, dT)")
        return cls.from_spec(spec)

    @classmethod
    def uniaxial(
        cls, axis: int, *, times, strain=None, stress=None, increments=10, temperature=None
    ) -> LoadingPath:
        """Tension (compression) along ``axis``: its strain (or stress) at the ``times``,
        the other stress components zero."""
        if (strain is None) == (stress is None):
            raise ValueError("uniaxial path: give the strain or the stress along the axis")
        name = VOIGT_NAMES[int(axis)]
        spec = {"times": list(times), "increments": increments}
        spec["strain" if strain is not None else "stress"] = {
            name: strain if strain is not None else stress
        }
        if temperature is not None:
            spec["temperature"] = temperature
        return cls.from_spec(spec)

    # -- evaluation --------------------------------------------------------------------

    @property
    def strain_controlled(self) -> np.ndarray:
        return np.array(self.controls)

    @property
    def duration(self) -> float:
        return self.times[-1] - self.times[0]

    def at(self, t: float) -> tuple[np.ndarray, float]:
        """Values of the components and change of temperature at the time ``t``."""
        values = np.asarray(self.values)
        v = np.array([np.interp(t, self.times, values[:, i]) for i in range(6)])
        return v, float(np.interp(t, self.times, self.temperature))

    def schedule(self) -> np.ndarray:
        """Times at the end of the planned steps."""
        ends = [
            np.linspace(a, b, n + 1)[1:]
            for a, b, n in zip(self.times[:-1], self.times[1:], self.increments, strict=True)
        ]
        return np.concatenate(ends)

    def to_dict(self) -> dict:
        series = {n: [r[i] for r in self.values] for i, n in enumerate(VOIGT_NAMES)}
        strain = {n: v for i, (n, v) in enumerate(series.items()) if self.controls[i]}
        stress = {n: v for i, (n, v) in enumerate(series.items()) if not self.controls[i]}
        out = {"times": list(self.times), "increments": list(self.increments)}
        if strain:
            out["strain"] = strain
        if stress:
            out["stress"] = stress
        if any(self.temperature):
            out["temperature"] = list(self.temperature)
        return out


@dataclass
class PathResult:
    """Macroscopic response of a volume element (or of a material point) along a path.

    ``times`` (n), ``strain`` and ``stress`` (n x 6, Voigt; engineering shear strains),
    ``temperature`` and ``plastic`` (the mean cumulated plastic strain of the material)
    at each converged step, the first one the natural state; ``phases``: for each
    phase, its mean ``stress``, ``strain`` and ``plastic`` strain at each step;
    ``iterations``: Newton iterations of each step; ``fields``: element fields at some
    times (``time -> name -> array``) on ``support``.

    In finite strains (``kinematics="finite"``), ``strain`` holds the components 11, 22,
    33, 23, 13, 12 of the upper triangular macroscopic displacement gradient ``F - I``
    (nominal strains and simple shears) and ``stress`` those of the mean first
    Piola-Kirchhoff stress (nominal stresses); ``cauchy`` holds the Cauchy (true) stress
    and ``log_strain`` the logarithmic (true) strain, and each phase its mean ``cauchy``
    stress. With porous plastic materials, ``porosity`` holds the mean porosity of their
    points at each step, and each porous phase its mean ``porosity``.

    ``series``: the other curves that a solver records at each step (name -> values:
    the area in contact of the walls of a cell that close,
    :class:`~morphogen.contact.periodic.ContactCell`), columns of :meth:`rows` and
    quantities of :meth:`value` as the others.

    ``orientation``: for a material point that is a crystal, in finite strains or where
    its twins turn it, the Bunge angles (degrees) of its lattice at each step (n x 3;
    the columns ``phi1``, ``Phi`` and ``phi2`` of :meth:`rows`). The lattices of the
    crystals of a volume element are in its ``fields`` (``orientation``):
    :meth:`texture`.
    """

    name: str
    bc: str
    path: LoadingPath
    times: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    temperature: np.ndarray
    plastic: np.ndarray
    phases: dict = field(default_factory=dict)
    iterations: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    info: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict, repr=False)
    support: object | None = field(default=None, repr=False)
    kinematics: str = "small"
    cauchy: np.ndarray | None = None
    log_strain: np.ndarray | None = None
    porosity: np.ndarray | None = None
    series: dict = field(default_factory=dict)
    #: other scalar results of the path, by name (the fatigue of its cycle): final values
    #: (:meth:`components`) and quantities of :meth:`value`
    scalars: dict = field(default_factory=dict)
    orientation: np.ndarray | None = None

    @property
    def n_steps(self) -> int:
        return len(self.times) - 1

    @property
    def finite(self) -> bool:
        return self.kinematics == "finite"

    def _true(self, k: int) -> dict[str, float]:
        """Logarithmic strains ``LE11 ...`` and Cauchy stresses ``TS11 ...`` of step
        ``k`` (finite strains)."""
        if self.log_strain is None or self.cauchy is None:
            return {}
        out = {f"LE{n}": float(v) for n, v in zip(VOIGT_NAMES, self.log_strain[k], strict=True)}
        out.update({f"TS{n}": float(v) for n, v in zip(VOIGT_NAMES, self.cauchy[k], strict=True)})
        return out

    def _row(self, k: int) -> dict:
        """The curves at step ``k`` (see :meth:`rows`)."""
        row = {"time": float(self.times[k])}
        strain, stress = self.strain[k], self.stress[k]
        row.update({f"E{n}": float(v) for n, v in zip(VOIGT_NAMES, strain, strict=True)})
        row.update({f"S{n}": float(v) for n, v in zip(VOIGT_NAMES, stress, strict=True)})
        row.update(self._true(k))
        row["temperature"] = float(self.temperature[k])
        row["p"] = float(self.plastic[k])
        if self.porosity is not None:
            row["f"] = float(self.porosity[k])
        for phase, series in self.phases.items():
            for n, v in zip(VOIGT_NAMES, series["stress"][k], strict=True):
                row[f"S{n}_{phase}"] = float(v)
            if "cauchy" in series:
                for n, v in zip(VOIGT_NAMES, series["cauchy"][k], strict=True):
                    row[f"TS{n}_{phase}"] = float(v)
            row[f"p_{phase}"] = float(series["plastic"][k])
            if "porosity" in series:
                row[f"f_{phase}"] = float(series["porosity"][k])
        row.update({name: float(values[k]) for name, values in self.series.items()})
        if self.orientation is not None:
            angles = zip(("phi1", "Phi", "phi2"), self.orientation[k], strict=True)
            row.update({name: float(v) for name, v in angles})
        return row

    def rows(self) -> list[dict]:
        """The curves, one row per step: time, ``E11 ... E12``, ``S11 ... S12``, in finite
        strains the logarithmic strains ``LE11 ...`` and the Cauchy stresses ``TS11 ...``,
        the change of temperature, the mean cumulated plastic strain (and porosity ``f``),
        the mean stress (and Cauchy stress) and plastic strain (and porosity) of each
        phase, and the other curves recorded (``series``)."""
        return [self._row(k) for k in range(len(self.times))]

    def driven(self) -> int:
        """The component of the loading: that whose prescribed strain varies most along
        the path (of its prescribed stresses, without a prescribed strain)."""
        values = np.asarray(self.path.values)
        controls = self.path.strain_controlled
        spans = np.ptp(values, axis=0)
        if controls.any():
            spans = np.where(controls, spans, -1.0)
        return int(np.argmax(spans))

    def texture(self, time: float | None = None):
        """The lattices of the crystals at a time of the path (the nearest of those of
        its fields; the last one by default): their orientations (Bunge angles in
        degrees, m x 3) and their volumes (m), for the pole figures of
        :mod:`morphogen.texture`. The crystals are the elements of a mesh or the voxels
        of an image whose fields have an ``orientation`` (crystals in finite strains,
        or whose twins turn their lattices; the paths run with ``fields``), or the
        material point of :attr:`orientation`."""
        if self.orientation is not None and not self.fields:
            k = len(self.times) - 1 if time is None else int(np.argmin(np.abs(self.times - time)))
            return self.orientation[k : k + 1].copy(), np.ones(1)
        if not self.fields:
            raise ValueError("texture: the path kept no fields (run it with fields)")
        at = max(self.fields) if time is None else min(self.fields, key=lambda t: abs(t - time))
        arrays = self.fields[at]
        if "orientation" not in arrays:
            raise ValueError(
                "texture: the fields have no orientation of the lattices (those of crystals "
                "in finite strains, or whose twins turn them)"
            )
        euler = np.asarray(arrays["orientation"], dtype=float).reshape(-1, 3)
        volumes = getattr(self.support, "volumes", None)
        weights = np.asarray(volumes(), dtype=float) if callable(volumes) else np.ones(len(euler))
        if weights.shape != (len(euler),):
            raise ValueError("texture: the fields and their support do not match")
        crystals = ~np.isnan(euler).any(axis=1)
        return euler[crystals], weights[crystals]

    def strength(self, component: int | None = None, upto: int | None = None) -> float:
        """The largest stress (in absolute value) reached along the path, up to the step
        ``upto``, of a component: by default that of the loading (:meth:`driven`). The
        strength of a volume element that softens (damage, fracture), its flow stress
        otherwise."""
        c = self.driven() if component is None else int(component)
        stop = len(self.times) if upto is None else int(upto) + 1
        return float(np.abs(self.stress[:stop, c]).max())

    def work(self, upto: int | None = None) -> float:
        """The work of the macroscopic stress per unit volume along the path, ``int Sigma
        : dE`` (trapezoids over the steps; nominal stresses and strains in finite
        strains), up to the step ``upto``: the energy stored and dissipated; at the end of a
        path that breaks the volume element, its work of fracture."""
        stop = len(self.times) if upto is None else int(upto) + 1
        mean = 0.5 * (self.stress[1:stop] + self.stress[: stop - 1])
        return float(np.einsum("ti,ti->", mean, np.diff(self.strain[:stop], axis=0)))

    def value(self, name: str, time: float | None = None) -> float:
        """Value of a quantity of the curves (a column of :meth:`rows`: ``E11``, ``S11``,
        ``TS11``, ``p``, ``S11_1``...) at the end of the path, or at ``time``: the end of
        one of its steps; or ``strength`` and ``work``, the largest stress of the loading
        and the work done up to there (:meth:`strength`, :meth:`work`)."""
        k = len(self.times) - 1
        if time is not None:
            hits = np.flatnonzero(np.isclose(self.times, float(time), rtol=1e-9, atol=1e-12))
            if hits.size == 0:
                raise ValueError(f"path {self.name!r}: no step ends at the time {time:g}")
            k = int(hits[-1])
        if name == "strength":
            return self.strength(upto=k)
        if name == "work":
            return self.work(upto=k)
        if name in self.scalars:
            return float(self.scalars[name])
        row = self._row(k)
        if name not in row or name == "time":
            columns = [c for c in row if c != "time"] + ["strength", "work"]
            raise ValueError(f"path {self.name!r}: unknown quantity {name!r}; use one of {columns}")
        return row[name]

    def components(self) -> dict[str, float]:
        """Final values: ``E11 ... E12``, ``S11 ... S12`` (in finite strains also ``LE11
        ...`` and ``TS11 ...``), the mean cumulated plastic strain ``p`` and the other
        curves recorded (``series``)."""
        out = {f"E{n}": float(v) for n, v in zip(VOIGT_NAMES, self.strain[-1], strict=True)}
        out.update({f"S{n}": float(v) for n, v in zip(VOIGT_NAMES, self.stress[-1], strict=True)})
        out.update(self._true(len(self.times) - 1))
        out["p"] = float(self.plastic[-1])
        if self.porosity is not None:
            out["f"] = float(self.porosity[-1])
        out.update({name: float(values[-1]) for name, values in self.series.items()})
        out.update({name: float(value) for name, value in self.scalars.items()})
        return out

    def row(self) -> dict:
        """Summary record (long format): final values and solver data."""
        out = {"property": self.name, "bc": self.bc, **self.components()}
        out["steps"] = self.n_steps
        out["iterations"] = int(np.sum(self.iterations))
        out.update(self.info)
        return out


@dataclass(frozen=True)
class PathQuantity:
    """A quantity of the curves of a loading path, for the statistics of RVE studies: a
    column of :meth:`PathResult.rows` (``S11``, ``TS11``, ``p``, ``S11_1``...) at the end
    of the path, or at ``time``, the end of one of its steps (the flow stress at a given
    strain, say), under every boundary condition, or under ``bc`` only.

    Written ``<path>.<quantity>`` or ``<path>.<quantity>@<time>``: ``tension.S11@5`` is
    ``S11`` of the path ``tension`` at the time 5; ``.<bc>`` follows for one boundary
    condition (``tension.S11@5.periodic``).
    """

    path: str
    quantity: str
    time: float | None = None
    bc: str | None = None

    @classmethod
    def parse(cls, text) -> PathQuantity:
        name = str(text).strip()
        bc = None
        head, dot, tail = name.rpartition(".")
        if dot and tail.lower() in _ALIASES:
            name, bc = head, _ALIASES[tail.lower()]
        body, at, when = name.partition("@")
        path, dot, quantity = body.partition(".")
        if not (dot and path and quantity) or "." in quantity:
            raise ValueError(
                f"{text!r}: a quantity of a path is <path>.<quantity>[@<time>][.<bc>], "
                "e.g. tension.S11@5"
            )
        time = None
        if at:
            try:
                time = float(when)
            except ValueError:
                raise ValueError(f"{text!r}: a time after @, e.g. tension.S11@5") from None
        return cls(path, quantity, time, bc)

    @property
    def name(self) -> str:
        """Name of the series of the quantity (without the boundary condition)."""
        name = f"{self.path}.{self.quantity}"
        return name if self.time is None else f"{name}@{self.time:.12g}"

    @property
    def key(self) -> str:
        return self.name if self.bc is None else f"{self.name}.{self.bc}"

    def of(self, result: PathResult) -> float:
        """The quantity in a response to the path."""
        return result.value(self.quantity, self.time)
