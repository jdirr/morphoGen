"""Design of laser-architectured sheets: the simulation of a case and the optimization of
its laser path.

A :class:`SheetCase` gathers the choices of a study: the base material and its
thickness, the treatment (hardening, or carburizing, nitriding, alloying in the melt
pool), the laser configuration (one beam, or two aligned beams on both faces), the
specimen and its mesh, and the pattern of the tracks. :meth:`SheetCase.evaluate` runs
the chain: the path drawn on the parallel part of the specimen, the treatment at the
Gauss points of the mesh (:mod:`morphogen.laser.process`), the flow law of each point
and layer (:mod:`morphogen.laser.materials`), the tensile test
(:mod:`morphogen.laser.mechanics`), and the measures (:data:`MEASURES`) — or, with the
loading ``cell``, the periodic cell of the sheet treated all over and its homogenized
tensile tests (:mod:`morphogen.laser.periodic`).

**Objectives** (:data:`OBJECTIVES`, maximized): ``strength_ductility`` the product of
the tensile strength and the uniform elongation (MPa, the balance of the "banana" chart
of AHSS), ``energy`` the work absorbed up to necking (MJ/m³, the area under the
engineering curve up to the maximum force), ``uniform_elongation``,
``tensile_strength``, ``yield_strength``; ``target``: the opposite of the squared
relative distances to target measures. **Constraints** bound any measure, e.g. the
reinforced volume fraction, the yield strength, the smallest gap between tracks (their
neighbours temper them) or the smallest radius of curvature of a track (the
quasi-steady thermal cycles need radii larger than the heated width), the length of the
path in the specimen (the time and energy of the process); a violation adds a penalty.

**Optimization** (:func:`optimize`) of the parameters of the pattern and of the laser
(dotted names: ``pattern.amplitude``, ``pattern.coefficients.1.0``, ``beam.power``,
``bottom.power``...), each between bounds, by ``grid`` (the full factorial of the
thesis's parametric studies), ``nelder_mead``, ``differential_evolution`` (SciPy),
``genetic`` (:func:`morphogen.design.search.genetic_algorithm`) or ``bayesian``: a
Gaussian process (Matérn 5/2, automatic relevance, hyperparameters by the marginal
likelihood) fitted to the penalized costs of a Latin hypercube of first designs, then
the design of largest expected improvement at each evaluation (Jones et al. 1998), for
the costly finite element evaluations.
"""

from __future__ import annotations

import copy
import itertools
import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import optimize as sopt
from scipy.stats import norm, qmc

from morphogen.laser.bending import BendingResult, three_point_bending
from morphogen.laser.heat import Beam
from morphogen.laser.materials import SheetMaterial
from morphogen.laser.mechanics import Membrane, TensileResult, tensile_test
from morphogen.laser.patterns import LaserPath, make_pattern
from morphogen.laser.process import LaserTreatment, TreatmentResult
from morphogen.laser.specimen import Specimen

#: Measures of a case: of the tensile test (MPa, engineering strain, MJ/m³), of the
#: bending test (N, N/mm, mJ), of the treated parallel part (fractions, HV, °C), of the
#: process (mm, s, J), and with several directions the anisotropy (the spread of a
#: measure over the directions over its mean); the measures of one direction ``d``
#: take the suffix ``_d`` (``yield_strength_45``).
MEASURES = (
    "yield_strength",
    "tensile_strength",
    "uniform_elongation",
    "energy",
    "strength_ductility",
    "failure_strain",
    "ductility",
    "bending_load",
    "bending_stiffness",
    "bending_load_at_deflection",
    "bending_energy",
    "volume_fraction",
    "area_fraction",
    "mean_hardness",
    "max_hardness",
    "melted_fraction",
    "peak_temperature",
    "path_length",
    "laser_time",
    "laser_energy",
    "min_gap",
    "min_radius",
    "yield_anisotropy",
    "strength_anisotropy",
    "elongation_anisotropy",
    "r_value",
    "r_anisotropy",
    "r_bar",
    "delta_r",
    "hill_f",
    "hill_g",
    "hill_h",
    "hill_n",
)
#: Objectives (maximized): measures, ``isotropy`` (the opposite of the anisotropy of
#: the yield strength), ``flow_isotropy`` (of the Lankford coefficient) and ``target``
#: (the opposite of the squared relative distances to target measures).
OBJECTIVES = (
    "strength_ductility",
    "energy",
    "uniform_elongation",
    "ductility",
    "tensile_strength",
    "yield_strength",
    "bending_load_at_deflection",
    "bending_energy",
    "bending_load",
    "isotropy",
    "flow_isotropy",
    "target",
)
#: Load cases.
LOADINGS = ("tension", "bending", "cell")
#: The keys of the loading ``cell`` that set the cell (the others go to its test).
CELL_KEYS = ("kind", "vectors", "element", "along")
#: How far around the periodic cell its pattern is drawn (mm): the heat of the tracks.
CELL_REACH = 5.0
#: Optimization methods.
METHODS = ("grid", "nelder_mead", "differential_evolution", "genetic", "bayesian")


def hill48(r_values: dict) -> dict:
    """Hill's 1948 criterion of a sheet from its Lankford coefficients at 0°, 45° and
    90° (``{direction: r}``): ``F``, ``G``, ``H``, ``N`` normalized by ``G + H = 1``
    (``F = r0 / (r90 (1 + r0))``, ``G = 1/(1 + r0)``, ``H = r0/(1 + r0)``, ``N = (r0 +
    r90)(1 + 2 r45) / (2 r90 (1 + r0))``; isotropy: 0.5, 0.5, 0.5, 1.5), the normal
    anisotropy ``r̄ = (r0 + 2 r45 + r90)/4`` and the planar ``Δr = (r0 − 2 r45 + r90)/2``;
    NaN without the three directions."""
    keys = ("r_bar", "delta_r", "hill_f", "hill_g", "hill_h", "hill_n")
    try:
        r0, r45, r90 = (float(r_values[d]) for d in (0, 45, 90))
    except (KeyError, TypeError, ValueError):
        return dict.fromkeys(keys, math.nan)
    if not all(np.isfinite([r0, r45, r90])) or r90 == 0 or r0 == -1:
        return dict.fromkeys(keys, math.nan)
    return {
        "r_bar": (r0 + 2 * r45 + r90) / 4,
        "delta_r": (r0 - 2 * r45 + r90) / 2,
        "hill_f": r0 / (r90 * (1 + r0)),
        "hill_g": 1 / (1 + r0),
        "hill_h": r0 / (1 + r0),
        "hill_n": (r0 + r90) * (1 + 2 * r45) / (2 * r90 * (1 + r0)),
    }


@dataclass
class SheetResult:
    """The evaluation of a case: its path, the treated sheet at the Gauss points (or at
    the fibres of the bending sections), the tensile test or the bending test, the
    measures; with several directions, the result of each (``directions``, by angle),
    the others being those of the first."""

    path: LaserPath
    treatment: TreatmentResult
    tensile: TensileResult | None
    measures: dict
    time_s: float = 0.0
    bending: BendingResult | None = None
    directions: dict = field(default_factory=dict)


@dataclass
class SheetCase:
    """The choices of a study (see the module). ``beam``: the beam of the upper face
    (and of both faces in the ``symmetric`` configuration unless ``bottom`` is given);
    ``pattern``: the kind and parameters of
    :func:`~morphogen.laser.patterns.make_pattern`; ``element``: the size of the elements
    (mm); ``loading``: the load case, ``kind`` ``tension`` (the default; the other keys
    are options of :func:`~morphogen.laser.mechanics.tensile_test`) or ``bending``
    (three-point bending of the strip: ``span``, ``deflection`` the reference
    deflection, ``sections`` along the span, the fibres across the width at about
    ``element``; :func:`~morphogen.laser.bending.three_point_bending`) or ``cell`` (the
    periodic cell of the sheet treated all over by the pattern repeated: its lattice
    from the pattern or ``vectors``, its elements of ``element`` mm; the other keys are
    options of :func:`~morphogen.laser.periodic.cell_test`; the measures of the treatment
    over the cell, those of the path over the parallel part of the specimen);
    ``directions``: angles (degrees) of the tensile axis from the x axis of the pattern
    (the rolling direction), a specimen for each (the pattern turned by the opposite
    angle; the cell is pulled in each direction instead): the measures are their means,
    each one's with its suffix, and the anisotropies."""

    material: SheetMaterial | str = "annealed_ms1500"
    specimen: Specimen = field(default_factory=Specimen)
    beam: Beam = field(default_factory=lambda: Beam(150.0, 10.0, 0.5, 0.55))
    bottom: Beam | None = field(
        default_factory=lambda: Beam(100.0, 10.0, 0.6, 0.55, profile="tophat")
    )
    configuration: str = "symmetric"
    treatment: str = "hardening"
    alloying: dict | None = None
    layers: int = 4
    pattern: dict = field(default_factory=lambda: {"kind": "straight", "count": 1})
    pause: float = 30.0
    element: float = 0.5
    loading: dict = field(default_factory=dict)
    ambient: float = 20.0
    directions: tuple | None = None

    def __post_init__(self):
        self.material = SheetMaterial.from_config(self.material)
        self.specimen = Specimen.from_config(self.specimen)
        if self.configuration == "single":
            self.bottom = None
        self.pattern = dict(self.pattern)
        if "kind" not in self.pattern:
            raise ValueError("pattern: a kind")
        self.loading = dict(self.loading)
        if self.loading.get("kind", "tension") not in LOADINGS:
            raise ValueError(f"loading kind: one of {LOADINGS}")
        if self.directions is not None:
            self.directions = tuple(float(d) for d in np.atleast_1d(self.directions))
        self._treatments: dict = {}

    def laser_treatment(self) -> LaserTreatment:
        """The treatment of the sheet (cached: its tabulated cycles are reused)."""
        key = (
            self.material.name,
            self.specimen.thickness,
            self.configuration,
            self.layers,
            self.treatment,
            repr(self.alloying),
            self.ambient,
        )
        if key not in self._treatments:
            self._treatments[key] = LaserTreatment(
                self.material,
                thickness=self.specimen.thickness,
                configuration=self.configuration,
                layers=self.layers,
                treatment=self.treatment,
                alloying=self.alloying,
                ambient=self.ambient,
            )
        return self._treatments[key]

    def path(self) -> LaserPath:
        """The laser path drawn on the parallel part of the specimen."""
        params = dict(self.pattern)
        kind = params.pop("kind")
        if kind == "none":
            return LaserPath([])
        return make_pattern(
            kind,
            self.specimen.band(),
            beam=self.beam,
            bottom=self.bottom,
            pause=self.pause,
            **params,
        )

    def with_values(self, values: dict) -> SheetCase:
        """A copy with parameters set by dotted names (``pattern.amplitude``,
        ``beam.power``, ``bottom.radius``, ``specimen.width``, ``layers``...)."""
        new = copy.copy(self)
        new.pattern = copy.deepcopy(self.pattern)
        for name, value in values.items():
            head, _, rest = name.partition(".")
            if head == "pattern":
                _set_dotted(new.pattern, rest, value)
            elif head in ("beam", "bottom", "specimen"):
                obj = getattr(new, head)
                if obj is None:
                    raise ValueError(f"{name}: no {head}")
                setattr(new, head, replace(obj, **{rest: value}))
            elif head == "loading" and rest:
                new.loading = {**new.loading, rest: value}
            elif hasattr(new, head) and not rest:
                setattr(new, head, value)
            else:
                raise ValueError(f"unknown parameter {name!r}")
        if new.bottom is not None and "beam.speed" in values:
            new.bottom = replace(new.bottom, speed=new.beam.speed)
        return new

    def evaluate(self, *, keep_fields: bool = False) -> SheetResult:
        """Run the chain (see the module), in each direction if several."""
        if self.loading.get("kind") == "cell":
            return self._evaluate_cell(keep_fields)
        if not self.directions:
            return self._evaluate(keep_fields)
        t0 = time.perf_counter()
        results = {}
        for d in self.directions:
            sub = copy.copy(self)
            sub.directions = None
            sub.pattern = dict(self.pattern)
            sub.pattern["rotation"] = float(self.pattern.get("rotation", 0.0)) - d
            sub._direction = float(d)  # the specimen cut at d from the rolling direction
            results[d] = sub._evaluate(keep_fields)
        return self._combine(results, t0)

    def _combine(self, results: dict, t0: float) -> SheetResult:
        """The result of several directions: the first's, with the means, the values of
        each direction and the anisotropies."""
        first = results[next(iter(results))]
        measures = {}
        spreads = (
            "yield_anisotropy",
            "strength_anisotropy",
            "elongation_anisotropy",
            "r_anisotropy",
            "r_bar",
            "delta_r",
            "hill_f",
            "hill_g",
            "hill_h",
            "hill_n",
        )
        for k, v in first.measures.items():
            vals = np.array([r.measures[k] for r in results.values()], dtype=float)
            measures[k] = bool(v) if isinstance(v, bool) else float(np.mean(vals))
            if not isinstance(v, bool) and k not in spreads:
                for d, r in results.items():
                    measures[f"{k}_{d:g}"] = float(r.measures[k])
        for name, key in (
            ("yield_anisotropy", "yield_strength"),
            ("strength_anisotropy", "tensile_strength"),
            ("elongation_anisotropy", "uniform_elongation"),
            ("r_anisotropy", "r_value"),
        ):
            vals = np.array([r.measures[key] for r in results.values()], dtype=float)
            mean = float(np.mean(vals))
            measures[name] = float((vals.max() - vals.min()) / mean) if mean else math.nan
        measures.update(hill48({d: r.measures["r_value"] for d, r in results.items()}))
        return replace(
            first, measures=measures, time_s=time.perf_counter() - t0, directions=results
        )

    def cell(self):
        """The periodic cell of the treated sheet
        (:class:`~morphogen.laser.periodic.PeriodicCell`): the lattice of the pattern
        (its length ``along`` straight tracks from the loading if given), or the
        ``vectors`` of the loading, its elements of ``element`` mm (or the loading's
        ``element``), centred on the parallel part of the specimen."""
        from morphogen.laser.periodic import PeriodicCell, lattice, reduce_basis

        vectors = self.loading.get("vectors")
        if vectors is None:
            a1, a2 = lattice(self.pattern, self.loading.get("along"))
        else:
            a1, a2 = reduce_basis(*vectors)
        x0, x1, y0, y1 = self.specimen.band()
        return PeriodicCell.build(
            a1,
            a2,
            element=float(self.loading.get("element", self.element)),
            centre=(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
        )

    def _evaluate_cell(self, keep_fields: bool) -> SheetResult:
        """The periodic cell of the sheet (loading ``cell``): its treatment once, its
        tensile test in each direction (:func:`~morphogen.laser.periodic.cell_test`)."""
        from morphogen.laser.periodic import cell_test

        t0 = time.perf_counter()
        cell = self.cell()
        load = {k: v for k, v in self.loading.items() if k not in CELL_KEYS}
        params = dict(self.pattern)
        kind = params.pop("kind")
        sheet = (  # the pattern over the sheet around the cell, as far as it heats it
            LaserPath([])
            if kind == "none"
            else make_pattern(
                kind,
                cell.box(CELL_REACH),
                beam=self.beam,
                bottom=self.bottom,
                pause=self.pause,
                cover=True,
                **params,
            )
        )
        pts, _, wts = cell.mesh.gauss()
        treat = self.laser_treatment().apply(sheet, pts.reshape(-1, 2), wts.reshape(-1))
        ne = len(cell.mesh.quads)
        flow = tuple(v.reshape(ne, 4, -1) for v in treat.flow())
        m = self.material
        critical, radius = m.critical_strain(treat.hardness), None
        if critical is not None:
            critical = critical.reshape(ne, 4, -1)
            radius = m.fracture.get("radius")
        aniso = None
        if m.anisotropy:  # the rolling direction along x, the tests at their angles
            aniso = m.yield_criterion(treat.treated.reshape(ne, 4, -1), angle=0.0)
        membrane = Membrane(
            cell.mesh,
            flow,
            young=m.young,
            poisson=m.poisson,
            thickness=self.specimen.thickness,
            critical=critical,
            fracture_radius=radius,
            anisotropy=aniso,
        )
        path = self.path()  # the measures of the path: on the specimen's parallel part
        everywhere = np.ones(len(treat.points), dtype=bool)
        results = {}
        for d in self.directions or (0.0,):
            t1 = time.perf_counter()
            tens = cell_test(membrane, cell, float(d), keep_fields=keep_fields, **load)
            measures = case_measures(self, path, treat, tens, inside=everywhere)
            results[d] = SheetResult(path, treat, tens, measures, time.perf_counter() - t1)
        if not self.directions:
            return replace(results[0.0], time_s=time.perf_counter() - t0)
        return self._combine(results, t0)

    def _evaluate(self, keep_fields: bool) -> SheetResult:
        t0 = time.perf_counter()
        path = self.path()
        load = dict(self.loading)
        if load.pop("kind", "tension") == "bending":
            return self._bend(path, load, t0)
        mesh = self.specimen.mesh(self.element)
        pts, _, wts = mesh.gauss()
        treat = self.laser_treatment().apply(path, pts.reshape(-1, 2), wts.reshape(-1))
        flow = tuple(v.reshape(len(mesh.quads), 4, -1) for v in treat.flow())
        m = self.material
        critical, radius = m.critical_strain(treat.hardness), None
        if critical is not None:
            critical = critical.reshape(len(mesh.quads), 4, -1)
            radius = m.fracture.get("radius")
        aniso = None
        if m.anisotropy:  # the rolling direction at -direction in the specimen
            share = treat.treated.reshape(len(mesh.quads), 4, -1)
            aniso = m.yield_criterion(share, angle=-getattr(self, "_direction", 0.0))
        membrane = Membrane(
            mesh,
            flow,
            young=m.young,
            poisson=m.poisson,
            thickness=self.specimen.thickness,
            critical=critical,
            fracture_radius=radius,
            anisotropy=aniso,
        )
        tens = tensile_test(self.specimen, membrane, keep_fields=keep_fields, **load)
        measures = case_measures(self, path, treat, tens)
        return SheetResult(path, treat, tens, measures, time.perf_counter() - t0)

    def _bend(self, path, load: dict, t0: float) -> SheetResult:
        """Three-point bending of the strip: the treatment at the fibres of its sections."""
        x0, x1, y0, y1 = self.specimen.band()
        span = float(load.get("span", 0.8 * (x1 - x0)))
        if not 0 < span <= x1 - x0:
            raise ValueError("bending: 0 < span <= the length of the strip")
        ns = int(load.get("sections", max(21, round(span / self.element) + 1)))
        ns += 1 - ns % 2  # odd: a section under the load
        ny = max(4, round((y1 - y0) / self.element))
        xc = 0.5 * (x0 + x1)
        xs = np.linspace(xc - 0.5 * span, xc + 0.5 * span, ns)
        ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
        pts = np.column_stack([np.repeat(xs, ny), np.tile(ys, ns)])
        dx = span / (ns - 1)
        wts = np.full(len(pts), dx * (y1 - y0) / ny)
        treat = self.laser_treatment().apply(path, pts, wts)
        kind, s0, a, b = (v.reshape(ns, ny, -1) for v in treat.flow())
        bend = three_point_bending(
            kind,
            s0,
            a,
            b,
            span=span,
            width=y1 - y0,
            thickness=self.specimen.thickness,
            young=self.material.young,
            deflection=load.get("deflection"),
        )
        measures = case_measures(self, path, treat, None, bend)
        return SheetResult(path, treat, None, measures, time.perf_counter() - t0, bending=bend)


def _set_dotted(mapping, dotted: str, value) -> None:
    keys = dotted.split(".")
    target = mapping
    for i, k in enumerate(keys[:-1]):
        nxt = keys[i + 1]
        if isinstance(target, list):
            k = int(k)
        if isinstance(target, dict) and k not in target:
            target[k] = [] if nxt.isdigit() else {}
        target = target[k]
    last = keys[-1]
    if isinstance(target, list):
        i = int(last)
        while len(target) <= i:
            target.append(0.0)
        target[i] = value
    else:
        target[last] = value
    return None


def _coerce_coefficients(pattern: dict) -> None:
    if "coefficients" in pattern:
        pattern["coefficients"] = [
            list(map(float, row)) if isinstance(row, (list, tuple)) else row
            for row in pattern["coefficients"]
        ]


def case_measures(
    case: SheetCase,
    path: LaserPath,
    treat: TreatmentResult,
    tens: TensileResult | None,
    bend: BendingResult | None = None,
    *,
    inside=None,
) -> dict:
    """The measures (:data:`MEASURES`) of an evaluated case: those of the tensile or of
    the bending test (NaN for the other), and those of the treated parallel part (or of
    the points ``inside``: the periodic cell) and of the path."""
    x0, x1, y0, y1 = case.specimen.band()
    p = treat.points
    if inside is None:
        inside = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
    w = treat.weights[inside]
    share = treat.treated[inside]
    hard = treat.hardness[inside]
    out = dict.fromkeys(MEASURES, math.nan)
    if tens is not None:
        out.update(tens.summary())
        out.pop("necked", None)
        out["necked"] = bool(tens.necked)
    if bend is not None:
        out.update(bend.summary())
    wsum = max(w.sum(), 1e-300)
    out["volume_fraction"] = float((w[:, None] * share).sum() / (treat.layers * wsum))
    out["area_fraction"] = float((w * (share.mean(axis=1) >= 0.5)).sum() / wsum)
    out["mean_hardness"] = float((w[:, None] * hard).sum() / (treat.layers * wsum))
    out["max_hardness"] = float(hard.max()) if hard.size else math.nan
    out["melted_fraction"] = float(
        (w[:, None] * treat.melted[inside]).sum() / (treat.layers * wsum)
    )
    out["peak_temperature"] = float(treat.peak.max()) if treat.peak.size else math.nan
    lo, hi = (x0, y0), (x1, y1)
    out["path_length"] = path.clipped_length(lo, hi) if len(path) else 0.0
    out["laser_time"] = path.time if len(path) else 0.0
    out["laser_energy"] = path.energy if len(path) else 0.0
    out["min_gap"] = path.min_gap(lo, hi) if len(path) > 1 else math.inf
    out["min_radius"] = path.min_radius() if len(path) else math.inf
    return out


# -- optimization -------------------------------------------------------------------------


@dataclass
class Parameter:
    """A design variable: a dotted name (see :meth:`SheetCase.with_values`), its bounds,
    whether it takes integer values."""

    name: str
    low: float
    high: float
    integer: bool = False

    def __post_init__(self):
        if not self.high > self.low:
            raise ValueError(f"parameter {self.name}: high > low")

    def value(self, u: float):
        v = self.low + float(np.clip(u, 0.0, 1.0)) * (self.high - self.low)
        return round(v) if self.integer else float(v)

    def unit(self, value) -> float:
        return (float(value) - self.low) / (self.high - self.low)

    @classmethod
    def from_config(cls, name: str, spec) -> Parameter:
        if isinstance(spec, Parameter):
            return spec
        if isinstance(spec, dict):
            return cls(
                name, float(spec["low"]), float(spec["high"]), bool(spec.get("integer", False))
            )
        low, high = spec[:2]
        integer = (
            bool(spec[2]) if len(spec) > 2 else (isinstance(low, int) and isinstance(high, int))
        )
        return cls(name, float(low), float(high), integer)


@dataclass
class OptimizationResult:
    """The best values found, their evaluation, the objective and the history of every
    evaluation (values, measures, objective, penalty, cost)."""

    values: dict
    result: SheetResult | None
    objective: float
    feasible: bool
    history: list
    method: str
    evaluations: int
    time_s: float
    settings: dict = field(default_factory=dict)


def _is_measure(name: str) -> bool:
    """A measure, or a measure of one direction (``yield_strength_45``)."""
    if name in MEASURES:
        return True
    base, _, angle = name.rpartition("_")
    try:
        float(angle)
    except ValueError:
        return False
    return base in MEASURES


class _Problem:
    """The penalized cost of normalized designs, with a cache of the evaluations."""

    def __init__(self, case, parameters, objective, constraints, target, penalty, callback):
        self.case = case
        self.parameters = parameters
        self.objective = objective
        self.constraints = {k: tuple(v) for k, v in (constraints or {}).items()}
        for k in self.constraints:
            if not _is_measure(k):
                raise ValueError(f"constraint on {k!r}: one of the measures {MEASURES}")
        self.target = dict(target or {})
        if objective == "target":
            if not self.target:
                raise ValueError("objective target: give the target measures")
            for k in self.target:
                if not _is_measure(k):
                    raise ValueError(f"target {k!r}: one of the measures {MEASURES}")
        elif objective not in OBJECTIVES:
            raise ValueError(f"objective: one of {OBJECTIVES}")
        self.penalty = float(penalty)
        self.callback = callback
        self.cache: dict = {}
        self.history: list = []
        self.results: dict = {}
        self.scale = None

    def values(self, u) -> dict:
        return {p.name: p.value(x) for p, x in zip(self.parameters, np.atleast_1d(u), strict=True)}

    def score(self, m: dict) -> float:
        if self.objective == "target":
            return -sum(((m[k] - v) / (abs(v) or 1.0)) ** 2 for k, v in self.target.items())
        if self.objective == "isotropy":
            return -float(m["yield_anisotropy"])
        if self.objective == "flow_isotropy":
            return -float(m["r_anisotropy"])
        return float(m[self.objective])

    def violation(self, m: dict) -> float:
        total = 0.0
        for k, (lo, hi) in self.constraints.items():
            v = m[k]
            if not np.isfinite(v):
                v = 1e300 if v > 0 else -1e300
            if lo is not None and v < lo:
                total += min((lo - v) / (abs(lo) or 1.0), 10.0)
            if hi is not None and v > hi:
                total += min((v - hi) / (abs(hi) or 1.0), 10.0)
        return total

    def __call__(self, u) -> float:
        vals = self.values(u)
        key = tuple(round(v, 9) if isinstance(v, float) else v for v in vals.values())
        if key in self.cache:
            return self.cache[key]
        case = self.case.with_values(vals)
        _coerce_coefficients(case.pattern)
        try:
            res = case.evaluate()
            m = res.measures
            obj = self.score(m)
            bad = not np.isfinite(obj)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            res, m, obj, bad = None, {"error": str(exc)}, math.nan, True
        viol = self.violation(m) if not bad else 10.0
        if self.scale is None and not bad and self.objective != "target":
            self.scale = max(abs(obj), 1e-12)
        scale = self.scale or 1.0
        cost = (-obj / scale if not bad else 1.0) + self.penalty * viol
        self.cache[key] = cost
        row = {
            "evaluation": len(self.history) + 1,
            **vals,
            **{k: m.get(k, math.nan) for k in MEASURES},
            **{k: v for k, v in m.items() if k not in MEASURES and _is_measure(k)},
            "objective": obj,
            "violation": viol,
            "cost": cost,
            "feasible": (viol == 0.0 and not bad),
        }
        if "error" in m:
            row["error"] = m["error"]
        self.history.append(row)
        self.results[key] = res
        if self.callback is not None:
            self.callback(row)
        return cost


def optimize(
    case: SheetCase,
    parameters,
    *,
    objective: str = "strength_ductility",
    constraints=None,
    target=None,
    method: str = "bayesian",
    evaluations: int = 40,
    initial: int | None = None,
    levels=5,
    penalty: float = 100.0,
    seed: int = 0,
    callback=None,
    **options,
) -> OptimizationResult:
    """Optimize the parameters of a case (see the module). ``parameters``: a mapping of
    dotted names to ``[low, high]`` (``[low, high, true]`` for integers) or
    :class:`Parameter` s; ``constraints``: a mapping of measures to ``[low, high]``
    (None for no bound); ``target``: the target measures of the objective ``target``;
    ``evaluations``: the budget of evaluations (``grid``: ``levels`` per parameter
    instead); ``initial``: the first designs of the Bayesian optimization (Latin
    hypercube; default 2 d + 2); ``options``: those of the SciPy methods."""
    if method not in METHODS:
        raise ValueError(f"method: one of {METHODS}")
    params = [Parameter.from_config(k, v) for k, v in dict(parameters).items()]
    if not params:
        raise ValueError("parameters: at least one")
    prob = _Problem(case, params, objective, constraints, target, penalty, callback)
    rng = np.random.default_rng(seed)
    d = len(params)
    t0 = time.perf_counter()
    if method == "grid":
        lv = [levels] * d if np.ndim(levels) == 0 else list(levels)
        axes = []
        for p, n in zip(params, lv, strict=True):
            n = int(n)
            if p.integer:
                vals = np.arange(int(p.low), int(p.high) + 1)
                axes.append(
                    [p.unit(v) for v in vals]
                    if len(vals) <= n or n < 2
                    else list(np.linspace(0.0, 1.0, n))
                )
            else:
                axes.append(list(np.linspace(0.0, 1.0, max(n, 1))) if n > 1 else [0.5])
        for u in itertools.product(*axes):
            prob(np.array(u))
    elif method == "nelder_mead":
        x0 = np.full(d, 0.5)

        def boxed(u):
            out = np.clip(u, 0.0, 1.0)
            return prob(out) + 10.0 * float(np.sum((u - out) ** 2))

        simplex = np.vstack([x0] + [x0 + 0.25 * np.eye(d)[i] for i in range(d)])
        sopt.minimize(
            boxed,
            x0,
            method="Nelder-Mead",
            options={
                "maxfev": evaluations,
                "initial_simplex": simplex,
                "xatol": 1e-3,
                "fatol": 1e-6,
                **options,
            },
        )
    elif method == "differential_evolution":
        popsize = int(options.pop("popsize", 6))
        maxiter = max(1, evaluations // (popsize * d) - 1)
        sopt.differential_evolution(
            prob,
            [(0.0, 1.0)] * d,
            maxiter=maxiter,
            popsize=popsize,
            polish=False,
            init="latinhypercube",
            tol=0.0,
            **_seeded(sopt.differential_evolution, seed),
            **options,
        )
    elif method == "genetic":
        from morphogen.design.search import genetic_algorithm

        size = int(options.pop("population", max(6, 2 * d + 2)))
        gens = max(1, evaluations // size - 1)
        pop = [rng.random(d) for _ in range(size)]

        def mutate(x, r):
            return np.clip(x + r.normal(0.0, 0.1, d), 0.0, 1.0)

        def crossover(a, b, r):
            w = r.random(d)
            return w * a + (1 - w) * b

        genetic_algorithm(
            pop,
            prob,
            rng,
            mutate=mutate,
            crossover=crossover,
            generations=gens,
            elite=1,
            tournament=2,
        )
    else:
        _bayesian(prob, d, evaluations, initial if initial is not None else 2 * d + 2, rng)
    hist = prob.history
    if not hist:
        raise RuntimeError("no evaluation")
    feasible = [h for h in hist if h["feasible"]]
    pool = feasible or hist
    best = min(pool, key=lambda h: h["cost"])
    values = {p.name: best[p.name] for p in params}
    key = tuple(round(v, 9) if isinstance(v, float) else v for v in values.values())
    return OptimizationResult(
        values=values,
        result=prob.results.get(key),
        objective=best["objective"],
        feasible=bool(best["feasible"]),
        history=hist,
        method=method,
        evaluations=len(hist),
        time_s=time.perf_counter() - t0,
        settings={
            "objective": objective,
            "constraints": prob.constraints,
            "target": prob.target,
            "parameters": {p.name: [p.low, p.high, p.integer] for p in params},
        },
    )


# -- Bayesian optimization ----------------------------------------------------------------


def _seeded(func, value) -> dict:
    """The seed keyword of a SciPy function: ``rng`` from SciPy 1.15, ``seed`` before."""
    import inspect

    return {"rng" if "rng" in inspect.signature(func).parameters else "seed": value}


def _matern52(x1, x2, ls):
    d = np.sqrt(np.maximum(((x1[:, None, :] - x2[None, :, :]) / ls) ** 2, 0.0).sum(axis=-1))
    s5 = math.sqrt(5.0) * d
    return (1.0 + s5 + 5.0 / 3.0 * d * d) * np.exp(-s5)


class GaussianProcess:
    """Gaussian process regression of costs on the unit cube: a constant mean, a Matérn
    5/2 kernel with a length scale per dimension, a nugget; the hyperparameters maximize
    the marginal likelihood (L-BFGS-B from a few starts)."""

    def __init__(self, x, y, *, seed: int = 0):
        self.x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu, self.sd = float(y.mean()), float(y.std() or 1.0)
        self.y = (y - self.mu) / self.sd
        d = self.x.shape[1]
        rng = np.random.default_rng(seed)
        best = None
        starts = [np.r_[np.full(d, np.log(0.3)), np.log(1e-3)]]
        starts += [np.r_[rng.uniform(np.log(0.05), np.log(2.0), d), np.log(1e-2)] for _ in range(3)]
        bounds = [(np.log(0.02), np.log(10.0))] * d + [(np.log(1e-8), np.log(0.5))]
        for s in starts:
            try:
                r = sopt.minimize(self._nll, s, method="L-BFGS-B", bounds=bounds)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if best is None or r.fun < best.fun:
                best = r
        theta = best.x if best is not None else starts[0]
        self._set(theta)

    def _factor(self, theta):
        d = self.x.shape[1]
        ls, nug = np.exp(theta[:d]), np.exp(theta[d])
        k = _matern52(self.x, self.x, ls) + (nug + 1e-10) * np.eye(len(self.x))
        return np.linalg.cholesky(k), ls

    def _nll(self, theta):
        try:
            c, _ = self._factor(theta)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = np.linalg.solve(c.T, np.linalg.solve(c, self.y))
        return 0.5 * self.y @ alpha + np.log(np.diag(c)).sum()

    def _set(self, theta):
        self.chol, self.ls = self._factor(theta)
        self.alpha = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, self.y))

    def predict(self, xs) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation of the costs at points of the cube."""
        ks = _matern52(np.atleast_2d(xs), self.x, self.ls)
        mean = ks @ self.alpha
        v = np.linalg.solve(self.chol, ks.T)
        var = np.maximum(1.0 - (v * v).sum(axis=0), 1e-12)
        return self.mu + self.sd * mean, self.sd * np.sqrt(var)


def expected_improvement(mean, std, best: float) -> np.ndarray:
    """Expected improvement of costs below ``best`` (minimization)."""
    z = (best - mean) / std
    return (best - mean) * norm.cdf(z) + std * norm.pdf(z)


def _bayesian(prob, d: int, evaluations: int, initial: int, rng) -> None:
    initial = max(2, min(int(initial), evaluations))
    sampler = qmc.LatinHypercube(d=d, **_seeded(qmc.LatinHypercube, rng))
    xs = list(sampler.random(initial))
    ys = [prob(x) for x in xs]
    while len(prob.history) < evaluations:
        x_arr, y_arr = np.array(xs), np.array(ys)
        ok = np.isfinite(y_arr)
        gp = GaussianProcess(x_arr[ok], y_arr[ok], seed=int(rng.integers(1 << 31)))
        best = float(y_arr[ok].min())
        cand = qmc.Sobol(d=d, **_seeded(qmc.Sobol, rng)).random(1024)
        xb = x_arr[ok][np.argmin(y_arr[ok])]
        local = np.clip(xb + rng.normal(0.0, 0.05, (256, d)), 0.0, 1.0)
        cand = np.vstack([cand, local])
        m, s = gp.predict(cand)
        ei = expected_improvement(m, s, best)
        # keep away from the evaluated designs (the cache would return them)
        dist = np.min(np.linalg.norm(cand[:, None, :] - x_arr[None, :, :], axis=-1), axis=1)
        ei[dist < 1e-3] = -1.0
        x_new = cand[int(np.argmax(ei))]
        xs.append(x_new)
        before = len(prob.history)
        ys.append(prob(x_new))
        if len(prob.history) == before:  # a cached design: a random one instead
            x_new = rng.random(d)
            xs.append(x_new)
            ys.append(prob(x_new))
