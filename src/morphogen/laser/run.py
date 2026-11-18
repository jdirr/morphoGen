"""Configuration files of laser-architectured sheets (``morphogen laser``).

Example (the reference case of Wang 2022, thesis ch. 5, and its optimization)::

    laser_sheet:
      material: annealed_ms1500        # aisi430, dp1180, or a mapping (preset + changes)
      treatment: hardening             # carburizing, nitriding, alloying
      configuration: symmetric         # single: the upper face only
      layers: 4                        # through the thickness
      laser:
        beam: {power: 150, speed: 10, radius: 0.5, absorptivity: 0.55}
        bottom: {power: 100, speed: 10, radius: 0.6, absorptivity: 0.55, profile: tophat}
        pause: 30                      # s between tracks
      specimen: {kind: dogbone, length: 30, width: 5, thickness: 1.0}
      element: 0.5                     # mm
      pattern: {kind: corrugated, count: 1, amplitude: 1.0, period: 15}
      loading: {kind: tension}         # or {kind: bending, span: 30, deflection: 1.0}
      directions: [0, 45, 90]          # optional: a specimen per direction (anisotropy)
      compare:                         # reference patterns, evaluated too
        base: {kind: none}
        straight: {kind: straight, count: 1}
    optimization:                      # optional
      method: bayesian                 # grid, nelder_mead, differential_evolution, genetic
      evaluations: 30
      objective: strength_ductility    # energy, uniform_elongation, tensile_strength...
      parameters: {pattern.amplitude: [0.0, 2.5], pattern.period: [6.0, 30.0]}
      constraints: {yield_strength: [400, null]}
      check_element: 0.25              # the best design again on a finer mesh
    cellular:                          # optional: a cellular design (laser.cells)
      phases: wang                     # ferrite and martensite of Wang et al. 2023
      size: 0.5                        # mm, the cells
      objective: ductility             # energy, tensile_strength, strength_ductility
      constraints: {tensile_strength: [700, null]}
      start: {amplitude: 1.5, period: 10}  # a track drawn by a spot of min_width cells
      method: exchange                 # swap (BESO), hca, topology
      symmetry: [point]
      period: 10                       # mm: the cells change by periods
      min_width: 2                     # cells: the spot
      iterations: 40
      realize: true                    # its tracks treated by the process above
    seed: 1
    outputs:
      directory: out/laser
      figures: true                    # PNG figures (matplotlib)
      fields: [vti]                    # hardness and peak temperature maps
      gcode: true                      # the laser path of the best design

Results: ``laser.json`` (the settings, the measures of each design: the pattern, the
references, the optimum and its check), ``curves.csv`` (the engineering curves),
``history.csv`` (the evaluations of the optimization), ``path.gcode`` and
``path.csv`` (the tracks), ``fields.vti`` (hardness, treated share and peak temperature
of the sheet, a pixel per 0.05 mm, a layer per slice) and the figures; with a cellular
design, ``cells.csv`` (its history), ``cells_design.csv`` (the phase of each cell, a
row per row of cells), ``cells_path.gcode`` (its tracks) and ``cells.png``; the measures
of its tracks treated by the process are the design ``cellular`` of ``laser.json``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from morphogen._utils import to_jsonable
from morphogen.config import ConfigError, _check_keys, read_config_file
from morphogen.laser.design import METHODS, OBJECTIVES, SheetCase, optimize
from morphogen.laser.heat import Beam
from morphogen.laser.materials import SheetMaterial
from morphogen.laser.specimen import Specimen

SECTION = {
    "directions",
    "material",
    "treatment",
    "alloying",
    "configuration",
    "layers",
    "laser",
    "specimen",
    "element",
    "pattern",
    "compare",
    "loading",
    "ambient",
}
OPTIMIZATION = {
    "method",
    "evaluations",
    "objective",
    "parameters",
    "constraints",
    "target",
    "initial",
    "levels",
    "penalty",
    "check_element",
    "options",
}
OUTPUTS = {"directory", "figures", "fields", "gcode", "resolution"}
CELLULAR = {
    "case",
    "phases",
    "critical",
    "size",
    "element",
    "objective",
    "constraints",
    "start",
    "method",
    "budget",
    "symmetry",
    "period",
    "min_width",
    "candidates",
    "iterations",
    "patience",
    "realize",
    "fracture_radius",
}
CELL_METHODS = ("exchange", "swap", "hca", "topology")


@dataclass
class LaserConfig:
    """A validated configuration: the case, the references, the optimization."""

    case: SheetCase
    compare: dict
    optimization: dict | None
    seed: int
    cellular: dict | None
    directory: Path
    figures: bool
    fields: list
    gcode: bool
    resolution: float
    source: dict = field(default_factory=dict)


def load_laser_config(source, *, base_dir=None) -> LaserConfig:
    """Validate a configuration (file path or mapping)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = read_config_file(path)
        base = path.parent if base_dir is None else Path(base_dir)
    else:
        data = dict(source)
        base = Path.cwd() if base_dir is None else Path(base_dir)
    _check_keys(
        "configuration",
        data,
        {"laser_sheet", "optimization", "cellular", "seed", "outputs", "version", "threads"},
        {"laser_sheet"},
    )
    sec = dict(data["laser_sheet"] or {})
    _check_keys("laser_sheet", sec, SECTION)
    try:
        case = case_from_config(sec)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"laser_sheet: {exc}") from None
    compare = dict(sec.get("compare") or {})
    for name, pat in compare.items():
        if not isinstance(pat, dict) or "kind" not in pat:
            raise ConfigError(f"laser_sheet.compare.{name}: a pattern with a kind")
    opt = data.get("optimization")
    if opt is not None:
        opt = dict(opt)
        _check_keys("optimization", opt, OPTIMIZATION, {"parameters"})
        if opt.get("method", "bayesian") not in METHODS:
            raise ConfigError(f"optimization.method: one of {METHODS}")
        if opt.get("objective", "strength_ductility") not in OBJECTIVES:
            raise ConfigError(f"optimization.objective: one of {OBJECTIVES}")
        if not isinstance(opt["parameters"], dict) or not opt["parameters"]:
            raise ConfigError("optimization.parameters: a mapping of names to [low, high]")
    cel = data.get("cellular")
    if cel is not None:
        cel = dict(cel or {})
        _check_keys("cellular", cel, CELLULAR)
        if cel.get("method", "exchange") not in CELL_METHODS:
            raise ConfigError(f"cellular.method: one of {CELL_METHODS}")
        try:
            cellular_problem(cel, case)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"cellular: {exc}") from None
    seed = data.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"seed: a non-negative integer, got {seed!r}")
    out = dict(data.get("outputs") or {})
    _check_keys("outputs", out, OUTPUTS)
    directory = Path(out.get("directory", "morphogen_laser"))
    if not directory.is_absolute():
        directory = base / directory
    fields_ = out.get("fields", ["vti"]) or []
    fields_ = [fields_] if isinstance(fields_, str) else list(fields_)
    for f in fields_:
        if f not in ("vti", "vtk", "npz"):
            raise ConfigError(f"outputs.fields: vti, vtk or npz, got {f!r}")
    return LaserConfig(
        case=case,
        compare=compare,
        optimization=opt,
        seed=seed,
        cellular=cel,
        directory=directory,
        figures=bool(out.get("figures", True)),
        fields=fields_,
        gcode=bool(out.get("gcode", True)),
        resolution=float(out.get("resolution", 0.05)),
        source=data,
    )


def case_from_config(sec: dict) -> SheetCase:
    """A :class:`SheetCase` of a ``laser_sheet`` section."""
    laser = dict(sec.get("laser") or {})
    _check_keys("laser_sheet.laser", laser, {"beam", "bottom", "pause"})
    material = SheetMaterial.from_config(sec.get("material", "annealed_ms1500"))
    configuration = sec.get("configuration", "symmetric")
    beam_cfg = dict(laser.get("beam") or {"power": 150.0, "speed": 10.0, "radius": 0.5})
    beam_cfg.setdefault("absorptivity", material.absorptivity)
    beam = Beam.from_config(beam_cfg)
    bottom = None
    if configuration != "single" and laser.get("bottom") is not None:
        bcfg = dict(laser["bottom"])
        bcfg.setdefault("absorptivity", material.absorptivity)
        bcfg.setdefault("speed", beam.speed)
        bottom = Beam.from_config(bcfg)
    spec = dict(sec.get("specimen") or {})
    spec.setdefault("thickness", material.thickness)
    return SheetCase(
        material=material,
        specimen=Specimen.from_config(spec),
        beam=beam,
        bottom=bottom,
        configuration=configuration,
        treatment=sec.get("treatment", "hardening"),
        alloying=sec.get("alloying"),
        layers=int(sec.get("layers", 4)),
        pattern=dict(sec.get("pattern") or {"kind": "straight", "count": 1}),
        pause=float(laser.get("pause", 30.0)),
        element=float(sec.get("element", 0.5)),
        loading=dict(sec.get("loading") or {}),
        ambient=float(sec.get("ambient", 20.0)),
        directions=sec.get("directions"),
    )


def cellular_problem(cel: dict, case: SheetCase):
    """The :class:`~morphogen.laser.cells.CellProblem` of a ``cellular`` section on the
    specimen of the case."""
    from morphogen.laser.cells import CellProblem, Phase, wang_phases

    phases = cel.get("phases", "wang")
    if phases == "wang":
        phases = wang_phases(**({"critical": float(cel["critical"])} if "critical" in cel else {}))
    elif isinstance(phases, (list, tuple)):
        phases = tuple(Phase.from_config(p) for p in phases)
    else:
        raise ValueError("phases: wang, or a list of phases {name, law, critical, cost}")
    extra = {}
    if "fracture_radius" in cel:
        extra["fracture_radius"] = cel["fracture_radius"]
    if cel.get("case") is not None:  # a structure: compact tension, L-shaped plate
        from morphogen.laser.structures import case_from_config as structure

        extra["case"] = structure(dict(cel["case"]))
    return CellProblem(
        specimen=case.specimen,
        phases=phases,
        size=float(cel.get("size", 0.5)),
        element=float(cel.get("element", case.element)),
        objective=cel.get("objective", "ductility"),
        constraints=dict(cel.get("constraints") or {}),
        **extra,
    )


def run_cellular(cel: dict, case: SheetCase, *, log=None):
    """Run the cellular design of a ``cellular`` section: the start (a track drawn by
    the spot), the automaton or the densities, and the tracks of the result treated by
    the process of the case (``realize``). Returns the optimization, the start and the
    evaluated case of the tracks (or None)."""
    from morphogen.laser.cells import beso, corrugation, design_tracks, hca, topology

    say = log or (lambda *a: None)
    problem = cellular_problem(cel, case)
    grid = problem.grid
    start_cfg = dict(cel.get("start") or {})
    width = int(cel.get("min_width", 2) or 1)
    if "tracks" in start_cfg:  # polylines (mm) drawn by the spot
        lines = [np.asarray(t, dtype=float) for t in start_cfg["tracks"]]
    else:
        period = start_cfg.get("period")
        lines = corrugation(
            problem.specimen,
            float(start_cfg.get("amplitude", 0.0)),
            float(period) if period else None,
            count=int(start_cfg.get("count", 1)),
        )
    start = grid.stamp(lines, width=max(width, 1))
    method = cel.get("method", "exchange")
    budget = cel.get("budget")
    iterations = int(cel.get("iterations", 40))
    say(f"cellular design ({method}, cells of {problem.size} mm)")
    if method in ("exchange", "swap"):
        run = beso(
            problem,
            start,
            budget=budget,
            moves=method,
            symmetry=tuple(cel.get("symmetry") or ()),
            period=cel.get("period"),
            min_width=width if width > 1 else None,
            candidates=int(cel.get("candidates", 6)),
            iterations=iterations,
            patience=int(cel.get("patience", 8)),
        )
    else:
        fun = hca if method == "hca" else topology
        run = fun(
            problem,
            start.astype(float),
            budget=float(start.mean() if budget is None else budget),
            iterations=iterations,
        )
    realized = None
    if cel.get("realize", True) and problem.case.kind == "tension":
        tracks = design_tracks(grid, run.design)
        sub = case.with_values({})
        sub.pattern = {"kind": "polylines", "tracks": [t.tolist() for t in tracks]}
        say(f"treating the {len(tracks)} track(s) of the design")
        realized = sub.evaluate(keep_fields=True)
    return run, start, realized


def run_laser(cfg: LaserConfig, *, callback=None, log=None) -> tuple[dict, Path]:
    """Evaluate the designs of a configuration, optimize when asked, write the results;
    returns the summary and the folder."""
    t0 = time.perf_counter()
    say = log or (lambda *a: None)
    designs = {}
    say("evaluating the pattern of the configuration")
    designs["pattern"] = cfg.case.evaluate(keep_fields=True)
    for name, pat in cfg.compare.items():
        say(f"evaluating the reference {name}")
        ref = cfg.case.with_values({})  # shares the cached kernels of the case
        ref.pattern = dict(pat)
        designs[name] = ref.evaluate(keep_fields=True)
    opt_out = None
    if cfg.optimization is not None:
        o = dict(cfg.optimization)
        check = o.pop("check_element", None)
        options = dict(o.pop("options", {}) or {})
        say(f"optimizing ({o.get('method', 'bayesian')}, {o.get('evaluations', 40)} evaluations)")
        opt_out = optimize(
            cfg.case, o.pop("parameters"), seed=cfg.seed, callback=callback, **o, **options
        )
        best_case = cfg.case.with_values(opt_out.values)
        designs["optimum"] = best_case.evaluate(keep_fields=True)
        if check:
            say(f"checking the optimum on elements of {check} mm")
            fine = best_case.with_values({"element": float(check)})
            designs["optimum_check"] = fine.evaluate(keep_fields=True)
    cells_out = None
    if cfg.cellular is not None:
        cells_out = run_cellular(cfg.cellular, cfg.case, log=say)
        if cells_out[2] is not None:
            designs["cellular"] = cells_out[2]
    summary = write_laser_results(cfg, designs, opt_out)
    if cells_out is not None:
        summary["cellular"] = write_cellular_results(cfg, cells_out)
    summary["time_s"] = time.perf_counter() - t0
    (cfg.directory / "laser.json").write_text(
        json.dumps(to_jsonable(summary), indent=2), encoding="utf-8"
    )
    return summary, cfg.directory


def write_laser_results(cfg: LaserConfig, designs: dict, opt_out) -> dict:
    """Write the files of the results (see the module); returns the summary."""
    from morphogen import pipeline

    out = cfg.directory
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "versions": pipeline.versions(),
        "config": cfg.source,
        "material": cfg.case.material.name,
        "designs": {
            k: {
                "measures": v.measures,
                "time_s": v.time_s,
                "tracks": len(v.path),
                "steps": v.tensile.steps if v.tensile is not None else 0,
            }
            for k, v in designs.items()
        },
    }
    # curves
    rows = []
    for name, res in designs.items():
        for d, r in (res.directions or {None: res}).items():
            tag = name if d is None else f"{name}_{d:g}"
            if r.tensile is not None:
                t = r.tensile
                for e, st, f, u in zip(t.strain, t.stress, t.force, t.displacement, strict=True):
                    rows.append(
                        {"design": tag, "strain": e, "stress": st, "force": f, "displacement": u}
                    )
            if r.bending is not None:
                b = r.bending
                for u, f in zip(b.deflection, b.load, strict=True):
                    rows.append({"design": tag, "deflection": u, "load": f})
    pipeline.write_rows(out / "curves.csv", rows)
    if opt_out is not None:
        pipeline.write_rows(out / "history.csv", opt_out.history)
        summary["optimization"] = {
            "method": opt_out.method,
            "values": opt_out.values,
            "objective": opt_out.objective,
            "feasible": opt_out.feasible,
            "evaluations": opt_out.evaluations,
            "time_s": opt_out.time_s,
            "settings": opt_out.settings,
        }
    best = designs.get("optimum", designs["pattern"])
    best_case = cfg.case if "optimum" not in designs else cfg.case.with_values(opt_out.values)
    if cfg.gcode and len(best.path):
        (out / "path.gcode").write_text(best.path.to_gcode(), encoding="utf-8")
        prow = []
        for k, t in enumerate(best.path):
            for x, y in t.points:
                prow.append(
                    {
                        "track": k,
                        "x": x,
                        "y": y,
                        "power": t.beam.power,
                        "bottom_power": t.bottom.power if t.bottom is not None else 0.0,
                        "speed": t.beam.speed,
                    }
                )
        pipeline.write_rows(out / "path.csv", prow)
    if cfg.fields:
        _write_fields(best_case, best, out, cfg.fields, cfg.resolution)
    if cfg.figures:
        try:
            from morphogen.laser import figures

            figures.write_figures(best_case, designs, opt_out, out, resolution=cfg.resolution)
        except ImportError:  # matplotlib missing
            summary["figures"] = "matplotlib is needed for the figures"
    return summary


def write_cellular_results(cfg: LaserConfig, cells_out) -> dict:
    """Write the files of a cellular design (see the module); returns its summary."""
    from morphogen import pipeline

    run, start, realized = cells_out
    out = cfg.directory
    out.mkdir(parents=True, exist_ok=True)
    pipeline.write_rows(out / "cells.csv", run.history)
    np.savetxt(out / "cells_design.csv", np.asarray(run.design), fmt="%d", delimiter=",")
    if cfg.gcode and realized is not None and len(realized.path):
        (out / "cells_path.gcode").write_text(realized.path.to_gcode(), encoding="utf-8")
    first = run.history[0] if run.history else {}
    summary = {
        "method": run.method,
        "evaluations": run.evaluations,
        "time_s": run.time_s,
        "start": {
            k: first.get(k)
            for k in ("value", "tensile_strength", "failure_strain", "energy", "budget")
        },
        "best": {"value": run.evaluation.value, **run.evaluation.measures},
    }
    if cfg.figures:
        try:
            from morphogen.laser.figures import cells_figure

            problem = cellular_problem(cfg.cellular, cfg.case)
            grid = problem.grid
            curves = {}
            for name, d in (("start", start), ("design", run.design)):
                ev = problem.evaluate(d)
                curves[name] = (ev.result.strain, ev.result.stress, ev.measures["failure_strain"])
            cells_figure(grid, {"start": start, "design": run.design}, curves, out / "cells.png")
        except ImportError:  # matplotlib missing
            summary["figures"] = "matplotlib is needed for the figures"
    return summary


def hardness_map(case: SheetCase, path, resolution: float = 0.05) -> dict:
    """The treated parallel part of a case on a grid of pixels (``resolution`` mm): the
    hardness, treated share and peak temperature of each layer, ``(nx, ny, layers)``."""
    x0, x1, y0, y1 = case.specimen.band()
    nx = max(2, round((x1 - x0) / resolution))
    ny = max(2, round((y1 - y0) / resolution))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    pts = np.column_stack([np.repeat(xs, ny), np.tile(ys, nx)])
    res = case.laser_treatment().apply(path, pts)
    shape = (nx, ny, res.layers)
    return {
        "x": xs,
        "y": ys,
        "spacing": ((x1 - x0) / nx, (y1 - y0) / ny),
        "hardness": res.hardness.reshape(shape),
        "treated": res.treated.reshape(shape),
        "peak": res.peak.reshape(shape),
        "carbon": res.carbon.reshape(shape),
    }


def _write_fields(case: SheetCase, res, out: Path, formats, resolution: float) -> None:
    from morphogen import io
    from morphogen.image import VoxelImage

    maps = hardness_map(case, res.path, resolution)
    dz = case.specimen.thickness / case.layers
    spacing = (*maps["spacing"], dz)
    images = [
        VoxelImage(maps[k].astype(np.float32), spacing=spacing, name=k)
        for k in ("hardness", "treated", "peak", "carbon")
    ]
    for fmt in formats:
        if fmt == "npz":
            np.savez_compressed(out / "fields.npz", **{k: maps[k] for k in maps})
        else:
            io.save_image(images, out / f"fields.{fmt}")
