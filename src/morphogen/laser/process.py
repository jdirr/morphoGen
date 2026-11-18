"""The laser treatment of a sheet: the hardness that a path leaves at given points.

:class:`LaserTreatment` holds the sheet (material, thickness, layers through the
thickness), the treatment (``hardening``, or a thermochemical one with its
:class:`~morphogen.laser.metallurgy.Alloying`) and the laser configuration (``single``:
the upper face; ``symmetric``: the same beam on both faces, or the beam of each track's
``bottom``). :meth:`LaserTreatment.apply` runs a
:class:`~morphogen.laser.patterns.LaserPath` track after track at the points of the sheet
(the Gauss points of a mesh, the pixels of a map), each point split into ``layers``
through the thickness:

1. the distance of each point to the track gives its thermal cycle at each depth, from
   the quasi-steady cycles of a long straight track
   (:class:`~morphogen.laser.heat.TrackKernel`, tabulated once per pair of beams): the
   moving beam heats the band around its axis
   as if the track were straight and long (curved tracks: their radius of curvature
   should exceed the heated width; their ends overhang the specimen);
2. a thermochemical treatment mixes the coating (or nitrogen) into the points that the
   track melts (:meth:`~morphogen.laser.metallurgy.Alloying.mix`), the coating of the faces melted
   being consumed;
3. the metallurgical model integrates each cycle (:mod:`morphogen.laser.metallurgy`): the
   points never touched before take the response of the first track tabulated once per
   kernel; the
   others (the tracks that overlap, temper or re-harden earlier ones) integrate their
   cycle step by step.

The tracks are separated by pauses long enough for the sheet to cool (30 s in Wang's
experiments): each cycle starts from ambient. :meth:`LaserTreatment.transient` runs the
same path through the finite-volume solver of the whole blank instead (the heat that the
previous tracks leave, the edges), for checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from morphogen.laser.heat import CONFIGURATIONS, Beam, TrackKernel, TransientHeat, track_kernel
from morphogen.laser.materials import SheetMaterial
from morphogen.laser.metallurgy import TREATMENTS, Alloying, PeakHardening
from morphogen.laser.patterns import LaserPath, distance_to_polyline

#: Distances of the response of an untouched sheet to a track (its shortcut).
RESPONSE_POINTS = 512
#: State variables that define the material of a point (the others are bookkeeping).
_MATERIAL_KEYS = (
    "martensite",
    "bainite",
    "hm0",
    "ptemper",
    "retained",
    "gain",
    "carbon",
    "nitrogen",
)


@dataclass
class TreatmentResult:
    """The treated sheet at its points: ``hardness``, ``peak`` temperature (°C),
    ``treated`` share (of the transformed phases, or of the full gain), ``melted``,
    ``carbon`` and ``nitrogen`` (wt%), all ``(points, layers)``; ``weights``: the area
    (mm²) of each point; the path and its measures."""

    points: np.ndarray
    weights: np.ndarray
    layers: int
    hardness: np.ndarray
    peak: np.ndarray
    treated: np.ndarray
    melted: np.ndarray
    carbon: np.ndarray
    nitrogen: np.ndarray
    material: SheetMaterial
    path: LaserPath | None = None
    state: dict = field(default_factory=dict, repr=False)

    @property
    def mean_hardness(self) -> np.ndarray:
        """Hardness averaged through the thickness (``points``)."""
        return self.hardness.mean(axis=1)

    def area_fraction(self, threshold: float = 0.5) -> float:
        """Fraction of the area whose treated share (through the thickness) exceeds
        ``threshold``."""
        share = self.treated.mean(axis=1)
        return float(np.sum(self.weights * (share >= threshold)) / np.sum(self.weights))

    def volume_fraction(self) -> float:
        """Mean treated share over the sheet: the reinforced volume fraction."""
        return float(
            np.sum(self.weights[:, None] * self.treated) / (self.layers * np.sum(self.weights))
        )

    def flow(self) -> tuple:
        """Flow laws of the points ``(kind, σ0, a, b)``, each ``(points, layers)``."""
        kind, s0, a, b = self.material.strength.flow(
            self.hardness.ravel(), young=self.material.young
        )
        shape = self.hardness.shape
        return kind.reshape(shape), s0.reshape(shape), a.reshape(shape), b.reshape(shape)


class LaserTreatment:
    """A treatment of a sheet (see the module): ``material`` (a :class:`SheetMaterial`
    or a preset name), ``thickness`` (mm; the material's by default), ``configuration``,
    ``layers`` through the thickness, ``treatment`` (``hardening``, ``carburizing``,
    ``nitriding``, ``alloying``) and its ``alloying`` parameters, ``ambient`` (°C)."""

    def __init__(
        self,
        material,
        *,
        thickness: float | None = None,
        configuration: str = "symmetric",
        layers: int = 4,
        treatment: str = "hardening",
        alloying=None,
        ambient: float = 20.0,
        distances: int = 48,
        times: int = 240,
    ):
        self.material = SheetMaterial.from_config(material)
        self.thickness = float(thickness if thickness is not None else self.material.thickness)
        if configuration not in CONFIGURATIONS:
            raise ValueError(f"configuration: one of {CONFIGURATIONS}")
        self.configuration = configuration
        self.layers = int(layers)
        if self.layers < 1:
            raise ValueError("layers: at least one")
        if treatment not in TREATMENTS:
            raise ValueError(f"treatment: one of {TREATMENTS}")
        self.treatment = treatment
        if treatment == "hardening":
            self.alloying = None
        else:
            spec = (
                {"kind": treatment, **dict(alloying or {})}
                if not isinstance(alloying, Alloying)
                else alloying
            )
            self.alloying = Alloying.from_config(spec)
        self.ambient = float(ambient)
        self._kernel_options = {"distances": int(distances), "times": int(times)}
        self._kernels: dict = {}
        self._responses: dict = {}

    @property
    def model(self):
        return self.material.metallurgy

    def beams(self, beam: Beam, bottom: Beam | None) -> tuple[Beam, Beam | None]:
        """The beams of a track: its ``bottom`` beam, or the same beam on the lower
        face in the ``symmetric`` configuration, none in the ``single`` one."""
        if self.configuration == "single":
            return beam, None
        return beam, bottom if bottom is not None else beam

    def kernel(self, beam: Beam, bottom: Beam | None = None) -> TrackKernel:
        """The tabulated cycles of a track of these beams (cached)."""
        top, low = self.beams(beam, bottom)
        key = (top, low)
        if key not in self._kernels:
            self._kernels[key] = track_kernel(
                top,
                self.thickness,
                self.material.thermal,
                bottom=low,
                layers=self.layers,
                ambient=self.ambient,
                cutoff=self._cutoff(),
                **self._kernel_options,
            )
        return self._kernels[key]

    def _cutoff(self) -> float:
        """The lowest peak temperature that changes a point: the onset of tempering when
        the sheet holds martensite, else of the transformations."""
        m = self.model
        low = m.tempering.t_min + 100.0
        if isinstance(m, PeakHardening) or getattr(m, "martensite", 0.0) > 0:
            return max(low, self.ambient + 50.0)
        return max(min(m.onset - 100.0, low + 300.0), self.ambient + 50.0)

    def _response(self, kern: TrackKernel) -> tuple[np.ndarray, dict]:
        """The state that the cycles of a kernel leave in an untouched sheet, on a fine
        grid of distances (:data:`RESPONSE_POINTS`) x layers: the thresholds of the
        transformations stay sharp."""
        key = id(kern)
        if key not in self._responses:
            dist = np.linspace(0.0, kern.reach, RESPONSE_POINTS)
            n = len(dist) * kern.layers
            state = self.model.initial_state(n)
            cyc = kern.cycles(dist)  # distances x times x layers
            temps = np.transpose(cyc, (1, 0, 2)).reshape(len(kern.times), n)
            self._integrate(state, temps, kern.times)
            self._responses[key] = (dist, state)
        return self._responses[key]

    def _integrate(self, state, temps, times) -> None:
        """Integrate cycles ``temps`` (times x points) from ambient, then cool."""
        m = self.model
        prev = np.full(temps.shape[1], self.ambient)
        t_prev = None
        for j, tj in enumerate(times):
            dt = float(tj - t_prev) if t_prev is not None else 0.0
            if dt > 0:
                m.step(state, prev, temps[j], dt)
            prev = temps[j]
            t_prev = tj
        m.step(state, prev, np.full_like(prev, self.ambient), max(1.0, float(times[-1] - times[0])))
        m.finish(state)

    def apply(self, path: LaserPath, points, weights=None) -> TreatmentResult:
        """Run the path at the points (n x 2, mm) of areas ``weights`` (mm²; equal by
        default): see the module."""
        pts = np.ascontiguousarray(np.asarray(points, dtype=float).reshape(-1, 2))
        n, nl = len(pts), self.layers
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).reshape(n)
        m = self.model
        state = m.initial_state(n * nl)
        init = m.initial_state(1)
        keys = [k for k in _MATERIAL_KEYS if k in state]
        coated = {"top": np.ones(n, bool), "bottom": np.ones(n, bool)}
        if self.alloying is not None:
            if self.alloying.faces == "top":
                coated["bottom"][:] = False
            elif self.alloying.faces == "bottom":
                coated["top"][:] = False
        dz = self.thickness / nl
        for track in path:
            kern = self.kernel(track.beam, track.bottom)
            d = distance_to_polyline(pts, track.points)
            near = np.flatnonzero(d < kern.reach)
            if not near.size:
                continue
            dn = d[near]
            rows = (near[:, None] * nl + np.arange(nl)[None, :]).ravel()  # point-layers
            # 1. composition of the melt
            changed = np.zeros(rows.size, dtype=bool)
            if self.alloying is not None:
                peaks = kern.peak_at(dn)  # near x layers
                melted = peaks >= m.solidus
                if melted.any():
                    surface = {}
                    for face, layer in (("top", 0), ("bottom", nl - 1)):
                        hit = melted[:, layer] & coated[face][near]
                        surface[face] = float(w[near][hit].sum())
                        coated[face][near[hit]] = False
                    mask = np.zeros(n * nl, dtype=bool)
                    mask[rows[melted.ravel()]] = True
                    self.alloying.mix(
                        state,
                        mask,
                        np.repeat(w, nl) * dz,
                        surface,
                        density=self.material.density,
                        melt_temperature=float(peaks.max()),
                        chromium=m.composition.get("Cr", 0.0),
                    )
                    changed = melted.ravel()
            # 2. metallurgy: untouched points take the tabulated response
            pristine = np.ones(rows.size, dtype=bool)
            for k in keys:
                pristine &= state[k][rows] == init[k][0]
            pristine &= ~changed
            if pristine.any():
                self._take_response(
                    state,
                    rows[pristine],
                    kern,
                    np.repeat(dn, nl)[pristine],
                    np.tile(np.arange(nl), dn.size)[pristine],
                )
            sel = ~pristine
            todo = rows[sel]
            if todo.size:
                sub = {k: v[todo] for k, v in state.items() if k != "clock"}
                sub["clock"] = state["clock"].copy()
                cyc = kern.cycles(dn)  # near x times x layers
                temps = np.transpose(cyc, (1, 0, 2)).reshape(len(kern.times), -1)
                self._integrate(sub, temps[:, sel], kern.times)
                for k, v in sub.items():
                    if k != "clock":
                        state[k][todo] = v
        hard = m.hardness(state).reshape(n, nl)
        treated = m.treated(state).reshape(n, nl)
        return TreatmentResult(
            points=pts,
            weights=w,
            layers=nl,
            hardness=hard,
            peak=state["tmax"].reshape(n, nl).copy(),
            treated=treated,
            melted=state["melted"].reshape(n, nl).copy(),
            carbon=state["carbon"].reshape(n, nl).copy(),
            nitrogen=state["nitrogen"].reshape(n, nl).copy(),
            material=self.material,
            path=path,
            state=state,
        )

    def _take_response(self, state, rows, kern: TrackKernel, dist, layer) -> None:
        """The tabulated response of an untouched sheet at distances ``dist`` and
        ``layer`` s: linear interpolation of the continuous variables in the distance,
        the nearest distance for the others."""
        grid, resp = self._response(kern)
        nd, nl = len(grid), kern.layers
        x = np.clip(dist, 0.0, grid[-1])
        i = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, nd - 2)
        f = (x - grid[i]) / (grid[i + 1] - grid[i])
        r0, r1 = i * nl + layer, (i + 1) * nl + layer
        nearest = np.where(f < 0.5, r0, r1)
        for k, table in resp.items():
            if k == "clock":
                continue
            if table.dtype == bool or k in ("t8",):
                state[k][rows] = table[nearest]
            else:
                a, b = table[r0], table[r1]
                ok = np.isfinite(a) & np.isfinite(b)
                state[k][rows] = np.where(ok, (1.0 - f) * a + f * b, table[nearest])

    def transient(
        self,
        path: LaserPath,
        size,
        *,
        origin=(0.0, 0.0),
        cell: float = 0.1,
        cooling: float | None = None,
    ) -> tuple:
        """The path run by the finite-volume solver on the blank ``size`` (mm) at
        ``origin``, cells of ``cell`` mm, the layers of the treatment through the
        thickness: ``(TransientHeat, TransientResult)`` with the metallurgical state of
        the cells (hardening only)."""
        if self.alloying is not None:
            raise ValueError(
                "transient runs: hardening only (the melt pools of the "
                "thermochemical treatments come from the kernels)"
            )
        solver = TransientHeat(
            size,
            self.thickness,
            self.material.thermal,
            cell=cell,
            layers=self.layers,
            origin=origin,
            ambient=self.ambient,
        )
        segs = []
        for t in path:
            top, low = self.beams(t.beam, t.bottom)
            segs.append((t.points, top, low, t.pause))
        res = solver.run(
            segs,
            configuration="single",
            model=self.model,
            cooling=cooling if cooling is not None else self.ambient + 30.0,
        )
        return solver, res
