"""Command-line interface: ``morphogen <command>`` (see ``morphogen --help``)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

from morphogen import analysis, gpu, io, pipeline, settings
from morphogen._platform import full_speed
from morphogen._utils import to_jsonable
from morphogen._version import __version__
from morphogen.config import ConfigError, RVESpec, _mfront_paths, load_config
from morphogen.models.base import get_model, list_models
from morphogen.rve import report as rve_report

log = logging.getLogger("morphogen")


def _templates() -> dict[str, str]:
    folder = resources.files("morphogen") / "templates"
    return {
        p.name[: -len(".yaml")]: p.read_text(encoding="utf-8")
        for p in folder.iterdir()
        if p.name.endswith(".yaml")
    }


#: Environment variables holding the index of a task of a job array, and their origin.
ARRAY_INDEX = (
    ("SLURM_ARRAY_TASK_ID", 0),
    ("PBS_ARRAY_INDEX", 0),
    ("PBS_ARRAYID", 0),
    ("SGE_TASK_ID", 1),
    ("OAR_ARRAY_INDEX", 1),
    ("LSB_JOBINDEX", 1),
)


def _task_id(value: str) -> int:
    """``--task-id``: an index, or ``auto`` (from the job array of the scheduler)."""
    if value != "auto":
        return int(value)
    for name, origin in ARRAY_INDEX:
        if os.environ.get(name, "").strip().isdigit():
            return int(os.environ[name]) - origin
    raise ValueError("--task-id auto: no job array index in the environment")


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    if args.output is not None:
        cfg.outputs.directory = Path(args.output)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.threads is not None:
        cfg.threads = args.threads
    if args.db is not None:
        cfg.outputs.database = Path(args.db)
    if args.mpi:
        from morphogen import parallel

        outdir = parallel.run_mpi(cfg, jobs=args.jobs, resume=args.resume)
    else:
        outdir = pipeline.run(
            cfg,
            task_id=_task_id(args.task_id),
            n_tasks=args.n_tasks,
            jobs=args.jobs,
            resume=args.resume,
        )
    print(f"results written to {outdir}")
    report = outdir / "rve_report.md"
    if report.exists() and not args.quiet:
        print(report.read_text(encoding="utf-8"))
    return 0


def _amitex_export(args) -> int:
    from morphogen.homogenization import Loading, LoadingPath, materials_from_config
    from morphogen.homogenization.materials import any_oriented

    materials = materials_from_config(_material_spec(args))
    path = Path(args.target)
    grains = {}
    if path.suffix.lower() == ".json":
        if args.voxels is None:
            raise ValueError("--voxels N: the voxels of the image of the geometry")
        micro = io.load_microstructure(path)
        image = micro.voxelize(args.voxels)
        orientations = getattr(micro, "orientations", None)
        if orientations is not None and any_oriented(materials):
            grains = {"grains": micro.labels(grid=image.grid()).data,
                      "orientations": orientations}  # fmt: skip
    else:
        image = io.load_image(path, name=args.array)
    loadings = {}
    for k, text in enumerate(args.loading or []):
        name, sep, rest = text.partition(":")
        name, rest = (name.strip(), rest) if sep else (f"loading{k + 1}", text)
        loadings[name] = Loading.parse(rest)
    paths = {}
    if args.paths:
        loaded = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))
        loaded = loaded.get("paths", loaded) if isinstance(loaded, dict) else None
        if not isinstance(loaded, dict):
            raise ValueError("--paths: a YAML/JSON file mapping names to loading paths")
        paths.update({str(k): LoadingPath.from_spec(v) for k, v in loaded.items()})
    for k, text in enumerate(args.path or []):
        name, sep, rest = text.partition(":")
        name, rest = (name.strip(), rest) if sep else (f"path{k + 1}", text)
        paths[name] = LoadingPath.parse(rest)
    stiffness = True if args.stiffness else None
    export = io.write_amitex(
        args.output or f"{path.stem}_amitex",
        image,
        materials,
        loadings=loadings,
        paths=paths,
        stiffness=stiffness,
        name=args.name or path.stem,
        fields=args.fields,
        tolerance=args.tol,
        processes=args.processes,
        library=args.library or "",
        elastic_only=args.elastic_only,
        kinematics=args.kinematics,
        **grains,
    )
    if args.json:
        print(json.dumps({"directory": str(export.directory), "name": export.name,
                          "runs": export.runs}, indent=2))  # fmt: skip
    else:
        print(f"AMITEX runs of {path.name} ({image.shape} voxels) in {export.directory}: "
              f"{', '.join(export.runs)}; run {export.name}_run.sh there")  # fmt: skip
    return 0


def _amitex_read(args) -> int:
    from morphogen.homogenization.postprocess import NOTATION, compliance, matrix_entries

    run = io.read_amitex_run(args.target, args.name)
    out = {"name": run.manifest["name"], "volume": run.manifest["volume"], "runs": {}}
    for key, res in run.results.items():
        out["runs"][key] = {
            "times": res.times.tolist(),
            "stress": res.stress.tolist(),
            "strain": res.strain.tolist(),
        }
    if run.stiffness is not None:
        out["stiffness"] = run.stiffness.tolist()
        out["compliance"] = compliance(run.stiffness).tolist()
        out["notation"] = NOTATION
        out["properties"] = run.properties
    missing = sorted(set(run.manifest["runs"]) - set(run.results))
    if missing:
        out["missing"] = missing
    if args.output:
        if run.stiffness is None:
            raise ValueError("-o: a table of the apparent stiffness, which needs its run")
        row = {"name": out["name"], "volume": out["volume"], **run.properties}
        row.update(matrix_entries(run.stiffness, "C"))
        row.update(matrix_entries(out["compliance"], "S"))
        target = Path(args.output)
        new = not target.exists()
        with open(target, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"{out['name']}: runs with results {sorted(run.results) or 'none'}"
          + (f", without {missing}" if missing else ""))  # fmt: skip
    if run.stiffness is not None:
        print("apparent", end=" ")
        _print_elastic(run.stiffness, out["compliance"])
        for key, value in run.properties.items():
            print(f"{key:>16s}: {value:.6g}")
    return 0


def cmd_amitex(args) -> int:
    return _amitex_export(args) if args.action == "export" else _amitex_read(args)


def cmd_db(args) -> int:
    from morphogen.results import ResultsDatabase, write_records

    with ResultsDatabase(args.db) as db:
        if args.action == "add":
            for directory in args.items:
                n = db.add_run(directory, name=args.name)
                print(f"{directory}: {n} records added to {args.db}")
            return 0
        if args.action == "remove":
            for run in args.items:
                if not db.remove(run):
                    raise ValueError(f"no run {run!r} in {args.db}")
            return 0
        records = db.records(args.where or [])
        if args.action == "export":
            if not args.output:
                raise ValueError("export: give the output file (-o results.csv)")
            write_records(records, args.output)
            print(f"{len(records)} records written to {args.output}")
            return 0
        if args.action == "keys":
            keys = db.keys(args.where or [])
            if args.json:
                print(json.dumps(keys, indent=2))
            else:
                for k, v in keys.items():
                    kind = "number" if v["numeric"] else "text"
                    print(f"{k:60s} {v['count']:7d}  {kind}")
            return 0
        runs = db.runs()
        if args.json:
            print(json.dumps({"runs": runs, "records": len(records)}, indent=2))
            return 0
        for r in runs:
            line = f"{r['id']:4d}  {r['name']:24s} {r['records']:6d} records  {r['added']}"
            print(f"{line}  {r['path']}")
        print(f"{len(records)} records" + (" match the conditions" if args.where else ""))
    return 0


def cmd_ashby(args) -> int:
    from morphogen.results import ResultsDatabase, ashby_chart, ashby_points, write_records

    with ResultsDatabase(args.db) as db:
        records = db.records(args.where or [])
    if not records:
        raise ValueError("no record (add runs with 'morphogen db add', or relax --where)")
    by = None if args.by in ("", "none") else args.by
    path = ashby_chart(
        records,
        args.output,
        args.x,
        args.y,
        by,
        log=not args.linear,
        guides=args.guides,
        groups=args.groups,
        theme=args.theme,
        title=args.title,
    )
    data = ashby_points(records, args.x, args.y, by, log=not args.linear, groups=args.groups)
    if args.table:
        keys = ["run_name", "realization", *([by] if by else []), data["x"], data["y"]]
        write_records(records, args.table, keys)
    shown = sum(len(g["points"]) for g in data["groups"])
    print(f"{path}: {shown} of {len(records)} records, {data['y']} against {data['x']}")
    return 0


def cmd_merge(args) -> int:
    from morphogen import parallel

    outdir = parallel.merge(args.directory, report=not args.no_report)
    print(f"results merged in {outdir}")
    report = Path(outdir) / "rve_report.md"
    if report.exists() and not args.quiet and not args.no_report:
        print(report.read_text(encoding="utf-8"))
    return 0


def cmd_status(args) -> int:
    from morphogen import parallel

    info = parallel.status(args.directory)
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    print(f"{info['directory']}: {info['done']} of {info['expected']} realizations done")
    if info["done"]:
        print(f"mean time per realization: {info['mean_time_s']:.2f} s")
    if info["missing"]:
        shown = ", ".join(info["missing"][:5]) + (" ..." if len(info["missing"]) > 5 else "")
        print(f"missing: {shown}")
    print("merged" if info["merged"] else "not merged yet (morphogen merge <directory>)")
    return 0 if not info["missing"] else 1


def cmd_submit(args) -> int:
    from morphogen import parallel

    cfg = load_config(args.config)
    outdir = Path(args.output) if args.output else cfg.outputs.directory
    seed = args.seed if args.seed is not None else cfg.seed
    if seed is None:  # one seed for all the tasks
        seed = int(np.random.SeedSequence().entropy)
    opts = parallel.JobOptions(
        tasks=args.tasks,
        jobs=args.jobs,
        threads=args.threads,
        time=args.time,
        memory=args.memory,
        queue=args.queue,
        account=args.account,
        name=args.name or f"morphogen-{cfg.outputs.name}",
        setup=list(args.setup or []),
    )
    array, merging = parallel.job_script(args.scheduler, args.config, outdir, opts, seed=seed)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{args.scheduler}_array.sh", outdir / f"{args.scheduler}_merge.sh"]
    for path, text in zip(paths, (array, merging), strict=True):
        path.write_text(text, encoding="utf-8", newline="\n")
        path.chmod(0o755)
    print(f"job array script: {paths[0]}\nmerging script: {paths[1]}")
    if args.submit:
        ids = parallel.submit(args.scheduler, *paths)
        print(f"submitted: job array {ids[0]}, merging job {ids[1]}")
    else:
        print(f"submit the array, then '{paths[1].name}' after it (or use --submit)")
    return 0


def cmd_fit(args) -> int:
    rows = pipeline.collect_results(args.results)
    fe_rows = pipeline.collect_homogenization(args.results)
    if args.volume_column != "volume":
        for row in rows + fe_rows:
            if args.volume_column not in row:
                raise ValueError(f"no column {args.volume_column!r} in the results")
            row["volume"] = row[args.volume_column]
    if args.bootstrap < 0:
        raise ValueError("--bootstrap: a non-negative number of replicates")
    spec = RVESpec(
        relative_errors=args.eps,
        n_realizations=args.n,
        gamma=args.gamma,
        min_volume=args.min_volume,
        max_volume=args.max_volume,
        bootstrap=args.bootstrap,
        weights=args.weights,
    )
    report = pipeline.build_report(
        rows, spec, args.property, fe_rows=fe_rows or None, group_by=args.group_by
    )
    if args.output:
        files = rve_report.write(report, Path(args.output))
        print(f"report written to {args.output} ({', '.join(f.name for f in files)})")
    print(rve_report.markdown(report))
    return 0


def _material_spec(args) -> dict:
    """The materials of the phases of ``--materials`` (a file; the sources of MFront
    behaviours relative to it) and ``--material`` (on the command line)."""
    spec = {}
    if args.materials:
        loaded = yaml.safe_load(Path(args.materials).read_text(encoding="utf-8"))
        loaded = loaded.get("materials", loaded) if isinstance(loaded, dict) else {}
        spec.update(_mfront_paths(loaded, Path(args.materials).parent))
    for text in args.material or []:
        phase, value = _parse_material(text)
        spec[phase] = value
    if not spec:
        raise ValueError("give the materials of the phases (--material or --materials)")
    return spec


def cmd_mfront(args) -> int:
    from morphogen.homogenization import materials_from_config
    from morphogen.io.mfront import identifier, write_mfront

    materials = materials_from_config(_material_spec(args))
    stem = identifier(args.name or (Path(args.materials).stem if args.materials else "material"))
    directory = Path(args.output or ".")
    written = []
    for p, m in sorted(materials.items()):
        # the nonlinear materials; in finite strains, the elastic ones too (Hencky)
        if m.void or m.mfront is not None or not (m.nonlinear or args.kinematics == "finite"):
            continue
        target = directory / f"{stem}Phase{p}.mfront"
        written.append(write_mfront(m, target, kinematics=args.kinematics))
        print(f"phase {p}: {target}")
    if not written:
        raise ValueError("no material to write (plasticity, porous_plasticity, "
                         "crystal_plasticity, learned; in finite strains, elasticity)")  # fmt: skip
    names = " ".join(p.name for p in written)
    print(f"compile: cd {directory} && mfront --obuild --interface=INTERFACE {names}")
    print("  (INTERFACE: generic for MGIS and morphoGen, umat for AMITEX_FFTP and Cast3M, "
          "abaqus, aster...)")  # fmt: skip
    return 0


def _apparent_properties() -> tuple[str, ...]:
    """The apparent properties of ``morphogen homogenize --property``."""
    from morphogen.homogenization.apparent import COUPLED, FLOW, HARMONIC, PROPERTIES, TENSORS

    scalars = [p for p in PROPERTIES if p not in ("bulk_modulus", "shear_modulus")]
    return ("bulk_modulus", "shear_modulus", "stiffness", "thermal_expansion", *scalars,
            *(t for t in TENSORS if t not in ("stiffness", "thermal_expansion")),
            *COUPLED, *HARMONIC, *FLOW)  # fmt: skip


def _parse_material(text: str):
    """``PHASE=void`` or ``PHASE=key:value,key:value`` (young, poisson, bulk, shear,
    conductivity, expansion, density, electrical_conductivity, permittivity, permeability,
    diffusivity; crystals: ``cubic:c11/c12/c44``, ``hexagonal:c11/c12/c13/c33/c44``,
    conductivities ``k1/k2/k3``)."""
    phase, sep, spec = text.partition("=")
    if not sep:
        raise ValueError(f"--material {text!r}: expected PHASE=void or PHASE=young:...,poisson:...")
    spec = spec.strip()
    if spec.lower() == "void":
        return phase.strip(), "void"
    values = {}
    for item in spec.split(","):
        key, sep, value = item.partition(":")
        if not sep:
            raise ValueError(f"--material {text!r}: items are key:value")
        numbers = [float(v) for v in value.split("/")]
        values[key.strip()] = numbers[0] if len(numbers) == 1 else numbers
    return phase.strip(), values


def _homogenization_sample(args, materials):
    """Mesh or image to homogenize: a geometry (.json) is meshed conformingly, a
    tetrahedral mesh (.vtu, .mesh) is used as is (finite elements), an image is solved
    by FFT."""
    from morphogen.homogenization.materials import any_oriented
    from morphogen.mesh.conforming import conforming_mesh

    path = Path(args.sample)
    ext = path.suffix.lower()
    if ext == ".json":
        micro = io.load_microstructure(path)
        if args.voxels is not None:
            # FFT on the image of the geometry; crystals turned by its grains
            image = micro.voxelize(args.voxels)
            orientations = getattr(micro, "orientations", None)
            if orientations is not None and any_oriented(materials):
                labels = micro.labels(grid=image.grid())
                return image, {"grains": labels, "orientations": orientations}
            return image, {}
        size, hausdorff, light = args.size, args.hausdorff, {}
        if size is None:  # finite elements by default: the voxels are the last resort
            from morphogen.config import default_mesh

            spec = default_mesh(micro.domain)
            size, light = spec.size, {"min_size": spec.min_size}
            hausdorff = spec.hausdorff if hausdorff is None else hausdorff
        phases = args.mesh_phases
        if phases == "auto":
            phases = "solid" if any(m.void for m in materials.values()) else "all"
        periodic = True if args.periodic else None
        extra = dict(light)
        if args.min_feature is not None:
            extra["min_feature"] = args.min_feature
        mesh = conforming_mesh(
            micro,
            size=size,
            hausdorff=hausdorff,
            order=args.order,
            phases=phases,
            periodic=periodic,
            cells=args.cells,
            method=args.mesh_method,
            **extra,
        )
        return mesh, {}
    if ext in (".vtu", ".mesh"):
        return io.read_tet_mesh(path, periodic=True if args.periodic else None), {}
    image = io.load_image(path, name=args.array)
    if args.periodic:
        image.periodic = True
    return image, {}


def _matrix_text(matrix) -> str:
    """A matrix as the commands print it (the round-off below 1e-12 of its largest entry
    as zeros)."""
    m = np.asarray(matrix, dtype=float)
    m = np.where(np.abs(m) < 1e-12 * np.abs(m).max(), 0.0, m)
    return np.array2string(m, formatter={"float_kind": "{:11.5g}".format})


def _print_harmonic(res: dict) -> None:
    """The scalar entries of a harmonic property (a row per frequency or wavelength)."""
    axis = "wavelengths" if "wavelengths" in res else "frequencies"
    values = np.asarray(res[axis])
    names = sorted({k.rsplit("_", 1)[0] for k, v in res.items()
                    if isinstance(v, float) and k.rsplit("_", 1)[-1].isdigit()})  # fmt: skip
    print("  " + f"{axis[:-1]:>12s} " + " ".join(f"{n:>16s}" for n in names))
    for i, x in enumerate(values):
        print("  " + f"{x:12.6g} " + " ".join(f"{res[f'{n}_{i + 1}']:16.6g}" for n in names))


def _print_elastic(stiffness, compliance=None, units=("of the moduli", "their inverse")):
    """The stiffness ``C`` and its compliance ``S = C^-1`` as the commands print them, with
    their notation (:func:`morphogen.homogenization.postprocess.notation`) and ``units``
    (those of C, those of S)."""
    from morphogen.homogenization.postprocess import compliance as inverse
    from morphogen.homogenization.postprocess import notation

    c = np.asarray(stiffness, dtype=float)
    s = inverse(c) if compliance is None else np.asarray(compliance, dtype=float)
    print(f"stiffness C ({notation(c)}), units {units[0]}:")
    print(_matrix_text(c))
    print(f"compliance S = C^-1, units {units[1]}:")
    print(_matrix_text(s))


def _contact_sample(args, materials):
    """The mesh of a test of contact (``morphogen compress``, ``morphogen indent``): a
    geometry (.json) meshed conformingly, or a tetrahedral mesh (.vtu, .mesh); its solid
    alone when phases are voids. The surfaces that touch are the faces of a mesh."""
    from morphogen.mesh.tetra import TetMesh

    if Path(args.sample).suffix.lower() not in (".json", ".vtu", ".mesh"):
        raise ValueError(
            "a geometry (.json, meshed with --size) or a tetrahedral mesh (.vtu, .mesh): the "
            "surfaces that touch are the faces of a mesh (morphogen mesh makes one of an image)"
        )
    mesh, _ = _homogenization_sample(args, materials)
    if not isinstance(mesh, TetMesh):  # pragma: no cover - the suffixes above
        raise ValueError("a tetrahedral mesh")
    voids = {int(k) for k, m in materials.items() if m.void}
    phases = {int(k) for k in np.unique(mesh.phase)}
    if phases & voids:
        mesh = mesh.subset(sorted(phases - voids))
    missing = sorted(phases - voids - {int(k) for k in materials})
    if missing:
        raise ValueError(f"no material for the phases {missing} of the mesh")
    return mesh


def _contact_report(args, mesh, values: dict, rows: list, curve: str, lines: list) -> int:
    """The results of a test of contact: its curve (.csv), its report (``-o``, ``--json``)
    or its lines on the terminal."""
    target = Path(args.curve or curve)
    target.parent.mkdir(parents=True, exist_ok=True)
    pipeline.write_rows(target, rows)
    out = {
        "file": str(args.sample),
        "elements": mesh.n_elements,
        "nodes": mesh.n_nodes,
        "order": mesh.order,
        **values,
        "curve": str(target),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(to_jsonable(out), indent=2) + "\n", "utf-8")
    if args.json:
        print(json.dumps(to_jsonable(out), indent=2))
        return 0
    print(
        f"{args.sample}: {mesh.n_elements} tetrahedra ({mesh.order * 6 - 2}-node), "
        f"{mesh.n_nodes} nodes"
    )
    for line in lines:
        print(line)
    print(f"curve {target}")
    return 0


def cmd_compress(args) -> int:
    from morphogen.contact import compression
    from morphogen.homogenization import materials_from_config

    materials = materials_from_config(_material_spec(args))
    mesh = _contact_sample(args, materials)
    res = compression(
        mesh,
        materials,
        args.strain,
        axis=args.axis,
        friction=args.friction,
        self_contact=not args.no_self_contact,
        points=args.points,
        steps=args.steps,
        kinematics="small" if args.small else "finite",
    )
    fields = None
    if args.fields:
        fields = str(res.problem.write(args.fields, res.result))
    tangent = res.tangent()
    values = {
        "height": res.height,
        "section": res.section,
        "strain": float(res.strain[-1]),
        "stress": float(res.stress[-1]),
        "force": float(res.force[-1]),
        "modulus": res.modulus,
        "tangent_modulus": float(tangent[-1]),
        "area_fixed": float(res.area[-1, 0]),
        "area_moving": float(res.area[-1, 1]),
        "self_area": float(res.self_area[-1]),
        "friction": float(res.friction[-1]),
        "steps": len(res.strain) - 1,
        "iterations": int(np.sum(res.result.iterations)),
        "fields": fields,
    }
    lines = [
        f"compressed by {-100 * res.strain[-1]:.4g} % between two platens "
        f"(friction {args.friction:g}): nominal stress {res.stress[-1]:.6g}",
        f"apparent modulus {res.modulus:.6g} at the first step, {tangent[-1]:.6g} at the last",
        f"in contact: {res.area[-1, 0]:.6g} and {res.area[-1, 1]:.6g} with the platens, "
        f"{res.self_area[-1]:.6g} of the sample with itself",
    ]
    stem = Path(args.sample).stem
    return _contact_report(args, mesh, values, res.rows(), f"{stem}_compression.csv", lines)


def cmd_indent(args) -> int:
    from morphogen.contact import indentation
    from morphogen.homogenization import materials_from_config

    materials = materials_from_config(_material_spec(args))
    mesh = _contact_sample(args, materials)
    res = indentation(
        mesh,
        materials,
        args.depth,
        indenter=args.indenter,
        radius=args.radius,
        half_angle=args.half_angle,
        point=args.point,
        axis=args.axis,
        friction=args.friction,
        steps=args.steps,
        unload=args.unload,
        base=args.base,
        kinematics="small" if args.small else "finite",
    )
    fields = None
    if args.fields:
        fields = str(res.problem.write(args.fields, res.result, step=len(res.depth) - 1))
    peak = int(np.argmax(res.depth))
    values = {
        "indenter": res.indenter,
        "depth": float(res.depth[peak]),
        "force": float(res.force[peak]),
        "area": float(res.area[peak]),
        "radius": float(res.radius[peak]),
        "pressure": float(res.pressure[peak]),
        "hardness": float(res.hardness[peak]),
        "modulus": float(res.modulus[peak]),
        "stiffness": res.stiffness,
        "unloading_modulus": res.unloading_modulus,
        "steps": len(res.depth) - 1,
        "iterations": int(np.sum(res.result.iterations)),
        "fields": fields,
    }
    lines = [
        f"{res.indenter} pressed to {res.depth[peak]:.6g}: force {res.force[peak]:.6g}, "
        f"contact of radius {res.radius[peak]:.6g}, mean pressure {res.hardness[peak]:.6g}",
        f"reduced modulus read from the force: {res.modulus[peak]:.6g} (a half-space)",
    ]
    if args.poisson is not None and res.indenter == "sphere":
        values["layer_modulus"] = float(res.layer_modulus(args.poisson)[peak])
        lines.append(
            f"with the base of the sample (a layer of thickness {res.thickness:.6g}): "
            f"{values['layer_modulus']:.6g}"
        )
    if res.unloading_modulus is not None:
        lines.append(f"from the slope of the unloading: {res.unloading_modulus:.6g}")
    stem = Path(args.sample).stem
    return _contact_report(args, mesh, values, res.rows(), f"{stem}_indentation.csv", lines)


def _interface_options(text: str) -> dict:
    """The law of the interfaces of ``--interfaces``: ``KEY:VALUE`` pairs separated by
    commas (``normal_strength:30,mode_I:0.05,friction:0.3``); ``faces`` is ``phases``,
    ``grains`` or pairs of phases (``faces:0-1+0-2``)."""
    out: dict = {}
    for item in text.split(","):
        key, sep, value = item.partition(":")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            raise ValueError(f"--interfaces {text!r}: KEY:VALUE pairs separated by commas")
        if key == "law":
            out[key] = value
        elif key == "faces":
            if value in ("phases", "grains"):
                out[key] = value
            else:
                try:
                    pairs = [[int(p) for p in pair.split("-")] for pair in value.split("+")]
                except ValueError:
                    pairs = []
                if not pairs or any(len(pair) != 2 for pair in pairs):
                    raise ValueError(
                        f"--interfaces: faces is phases, grains or pairs of phases (0-1+0-2), "
                        f"got {value!r}"
                    )
                out[key] = pairs
        else:
            try:
                out[key] = float(value)
            except ValueError:
                raise ValueError(f"--interfaces: {key} is a number, got {value!r}") from None
    return out


def _nonlinear_options(args) -> dict | None:
    """The options of the loading paths of ``morphogen homogenize``: finite strains
    (``--finite``), the contact of the walls of the pores (``--contact [MU]``) and the
    interfaces of the phases that debond (``--interfaces KEY:VALUE,...``)."""
    out = {}
    if args.finite:
        out["kinematics"] = "finite"
    if args.contact is not None:
        if not args.contact >= 0:
            raise ValueError("--contact: a coefficient of friction of 0 or more")
        out["contact"] = {"friction": args.contact}
    if args.interfaces is not None:
        out["interfaces"] = _interface_options(args.interfaces)
    return out or None


def cmd_homogenize(args) -> int:
    from morphogen.homogenization import (
        Apparent,
        Loading,
        LoadingPath,
        PathResult,
        Response,
        bounds,
        homogenize,
        materials_from_config,
    )
    from morphogen.homogenization.apparent import COUPLED, HARMONIC
    from morphogen.homogenization.coupled import NOTATION as COUPLED_NOTATION
    from morphogen.homogenization.postprocess import notation
    from morphogen.mesh.tetra import TetMesh

    materials = materials_from_config(_material_spec(args))
    sample, grains = _homogenization_sample(args, materials)
    if isinstance(sample, TetMesh) and args.save_mesh:
        io.save_mesh(sample, args.save_mesh)
    loadings = {}
    for k, text in enumerate(args.loading or []):
        name, sep, rest = text.partition(":")
        name, rest = (name.strip(), rest) if sep else (f"loading{k + 1}", text)
        loadings[name] = Loading.parse(rest)
    paths = {}
    if args.paths:
        loaded = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))
        loaded = loaded.get("paths", loaded) if isinstance(loaded, dict) else None
        if not isinstance(loaded, dict):
            raise ValueError("--paths: a YAML/JSON file mapping names to loading paths")
        paths.update({str(k): LoadingPath.from_spec(v) for k, v in loaded.items()})
    for k, text in enumerate(args.path or []):
        name, sep, rest = text.partition(":")
        name, rest = (name.strip(), rest) if sep else (f"path{k + 1}", text)
        paths[name] = LoadingPath.parse(rest)
    results = homogenize(
        sample,
        materials,
        bcs=args.bc,
        properties=args.property,
        solver=args.solver,
        tol=args.tol,
        fields=bool(args.fields),
        loadings=loadings,
        paths=paths,
        nonlinear=_nonlinear_options(args),
        integration=args.integration,
        device=args.device,
        frequencies=args.frequencies,
        wavelengths=args.wavelengths,
        fluid_bulk_modulus=args.fluid_bulk_modulus,
        **grains,
    )
    if isinstance(sample, TetMesh):
        fractions = pipeline._mesh_fractions(sample, materials)
        described = (
            f"{sample.n_elements} tetrahedra ({sample.order * 6 - 2}-node), "
            f"{sample.n_nodes} nodes, finite elements"
        )
        size = {"elements": sample.n_elements, "nodes": sample.n_nodes, "order": sample.order}
    else:
        fractions = analysis.phase_fractions(sample)
        described = f"{sample.shape} voxels, FFT solver"
        size = {"shape": list(sample.shape)}
    out = {"file": str(args.sample), **size, "phase_fractions": fractions}
    table = []
    responses = {}
    curves = {}
    for key, res in results.items():
        if isinstance(res, PathResult):
            folder = Path(args.curves)
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{Path(args.sample).stem}_{key.replace('.', '_')}.csv"
            pipeline.write_rows(target, res.rows())
            curves[key] = {**res.components(), "steps": res.n_steps, "curve": str(target)}
            if args.fields and res.fields:
                folder = Path(args.fields)
                folder.mkdir(parents=True, exist_ok=True)
                for k, arrays in enumerate(res.fields[t] for t in sorted(res.fields)):
                    stem = f"{Path(args.sample).stem}_{key.replace('.', '_')}_{k + 1}.vtu"
                    pipeline.write_path_fields(folder / stem, res.support, arrays)
            continue
        if isinstance(res, (Apparent, Response)):
            if isinstance(res, Apparent):
                table.append(res.row())
            else:
                responses[key] = res.components()
            if args.fields:
                folder = Path(args.fields)
                folder.mkdir(parents=True, exist_ok=True)
                stem = f"{Path(args.sample).stem}_{key.replace('.', '_')}"
                pipeline._write_fields(res, folder / stem)
        elif key.partition(".")[0] in COUPLED:  # the tensors of a coupled law
            out[key] = {**res, "notation": COUPLED_NOTATION}
        elif key.partition(".")[0] in HARMONIC:
            out[key] = res
        elif isinstance(res, dict) and "compliance" in res:  # a stiffness, its compliance
            out[key] = {**res, "notation": notation(res["tensor"])}
        else:
            out[key] = res
    out["results"] = table
    if responses:
        out["loadings"] = responses
    if curves:
        out["paths"] = curves
    try:
        out["bounds"] = bounds(fractions, materials)
    except (ValueError, KeyError):
        out["bounds"] = {}
    if args.output:
        Path(args.output).write_text(json.dumps(to_jsonable(out), indent=2) + "\n", "utf-8")
    if args.json:
        print(json.dumps(to_jsonable(out), indent=2))
        return 0
    print(
        f"{args.sample}: {described}; phase fractions "
        + ", ".join(f"{p}: {f:.4f}" for p, f in fractions.items())
    )
    width = max([15] + [len(row["property"]) for row in table])
    print(
        f"{'property':>{width}s} {'bc':>9s} {'value':>12s} {'energy':>12s} {'isotropy':>9s} "
        f"{'dofs':>9s} {'iter':>5s} {'time_s':>7s}"
    )
    for row in table:
        print(
            f"{row['property']:>{width}s} {row['bc']:>9s} {row['value']:12.6g} "
            f"{row['energy_value']:12.6g} {row['isotropy']:9.3g} {row['n_dofs']:9d} "
            f"{row['iterations']:5d} {row['solve_s']:7.2f}"
        )
    for key, res in results.items():
        if isinstance(res, Response):
            print(f"{key}: " + " ".join(f"{n} {v:.6g}" for n, v in res.components().items()))
        elif isinstance(res, PathResult):
            final = " ".join(f"{n} {v:.6g}" for n, v in res.components().items())
            print(f"{key}: {res.n_steps} steps, final {final}; curve {curves[key]['curve']}")
        elif key.partition(".")[0] in HARMONIC:  # its scalar entries at each frequency
            print(f"{key}:")
            _print_harmonic(res)
        elif key.partition(".")[0] in COUPLED:  # its tensors by name, then its constants
            print(f"{key} ({COUPLED_NOTATION}):")
            for name, value in res.items():
                if name != "tensor" and isinstance(value, np.ndarray) and value.ndim == 2:
                    print(f"  {name}:")
                    print(_matrix_text(value))
            constants = {n: v for n, v in res.items() if isinstance(v, float) and n != "residual"}
            print("  constants: " + ", ".join(f"{n} {v:.6g}" for n, v in constants.items()))
        elif key.partition(".")[0] in ("hydraulic_permeability", "acoustic"):
            print(f"{key} (lengths in the unit of the voxels, permeabilities in its square):")
            tensor = res.get("tensor", res.get("permeability_tensor"))
            print("  permeability tensor k (<v> = -k grad p / viscosity):")
            print(_matrix_text(tensor))
            constants = {n: v for n, v in res.items() if isinstance(v, float) and len(n) > 3}
            print("  " + ", ".join(f"{n} {v:.6g}" for n, v in constants.items()))
        elif key.partition(".")[0] == "poroelastic":  # drained C and S, Biot, Gassmann
            print(f"{key} (Sigma = C E - b p; porosity change = b : E + p / N):")
            _print_elastic(res["tensor"], res["compliance"])
            print("  tensor of Biot b:")
            print(_matrix_text(res["biot"]))
            if "skempton" in res:
                print("  undrained stiffness:")
                print(_matrix_text(res["undrained_stiffness"]))
                print("  tensor of Skempton B (p = -B : Sigma / 3):")
                print(_matrix_text(res["skempton"]))
            constants = {n: v for n, v in res.items() if isinstance(v, float) and len(n) > 3}
            print("  constants: " + ", ".join(f"{n} {v:.6g}" for n, v in constants.items()))
        elif isinstance(res, dict) and "compliance" in res:  # C, S = C^-1, the constants
            print(f"{key}:")
            _print_elastic(res["tensor"], res["compliance"])
            names = [f"E{i}" for i in (1, 2, 3)] + ["G23", "G31", "G12"]
            names += [f"nu{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3) if i != j]
            print("engineering constants: " + ", ".join(f"{n} {res[n]:.6g}" for n in names))
        elif not isinstance(res, Apparent):
            print(f"{key}:")
            print(_matrix_text(res["tensor"]))
    for prop, b in out["bounds"].items():
        print(f"bounds of {prop}: " + ", ".join(f"{k} {v:.6g}" for k, v in b.items()))
    return 0


def cmd_bands(args) -> int:
    """The band structure of the Bloch waves of a periodic cell (an image or a mesh)."""
    from morphogen.homogenization import band_structure, materials_from_config
    from morphogen.homogenization.bloch import POINTS

    materials = materials_from_config(_material_spec(args))
    path = Path(args.sample)
    if path.suffix.lower() in (".vtu", ".mesh"):
        sample = io.read_tet_mesh(path, periodic=True)
    else:
        sample = io.load_image(path, name=args.array)
        sample.periodic = True
    unknown = [c for c in args.path or () if c not in POINTS]
    if unknown or (args.path and len(args.path) < 2):
        raise ValueError(
            f"--path: at least two corners of the Brillouin zone among {', '.join(POINTS)}"
        )
    bands = band_structure(
        sample,
        materials,
        tuple(args.path) if args.path else None,
        bands=args.bands,
        points=args.points,
        order=args.order,
        phases=args.phases,
        waves=args.waves,
        coarsen=args.coarsen,
    )
    names = [f"omega{k + 1}" for k in range(bands.frequencies.shape[1])]
    rows = [
        {"distance": float(d), "kx": float(k[0]), "ky": float(k[1]), "kz": float(k[2]),
         **{n: float(f) for n, f in zip(names, row, strict=True)}}
        for d, k, row in zip(bands.distance, bands.wavevectors, bands.frequencies, strict=True)
    ]  # fmt: skip
    gaps = [{"below": n, "low": low, "high": high} for n, low, high in bands.gaps()]
    if args.output:
        pipeline.write_rows(Path(args.output), rows)
    if args.figure:
        ax = bands.plot(ylabel="angular frequency")
        ax.figure.savefig(args.figure, dpi=150)
    out = {
        "file": str(args.sample),
        "waves": args.waves,
        "path": [name for _, name in bands.labels],
        "wavevectors": len(rows),
        "bands": bands.frequencies.shape[1],
        "gaps": gaps,
        "highest": float(bands.frequencies.max()),
    }
    if args.json:
        print(json.dumps(to_jsonable({**out, "frequencies": bands.frequencies}), indent=2))
        return 0
    corners = " - ".join(out["path"])
    print(f"{args.sample}: {out['bands']} bands of the {args.waves} waves at {len(rows)} wave "
          f"vectors along {corners}")  # fmt: skip
    for (_, name), i in zip(bands.labels, _label_rows(bands), strict=True):
        values = " ".join(f"{f:.6g}" for f in bands.frequencies[i])
        print(f"  {name:>2s}: {values}")
    if gaps:
        for gap in gaps:
            low, high, n = gap["low"], gap["high"], gap["below"]
            width = 200 * (high - low) / (high + low)
            print(f"band gap between the bands {n} and {n + 1}: {low:.6g} to {high:.6g} "
                  f"({width:.1f} % of its middle)")  # fmt: skip
    else:
        print("no band gap along this path")
    for target in (args.output, args.figure):
        if target:
            print(f"written: {target}")
    return 0


def _label_rows(bands) -> list[int]:
    """The rows of the wave vectors of the corners of the path of a band structure."""
    return [int(np.argmin(np.abs(bands.distance - position))) for position, _ in bands.labels]


def _indices(text: str) -> tuple[int, ...]:
    """The indices of a family of planes written without spaces: 111, 10-10, 0001."""
    import re

    found = re.findall(r"-?\d", text.replace(" ", "").replace(",", ""))
    if not found or "".join(found) != text.replace(" ", "").replace(",", ""):
        raise argparse.ArgumentTypeError(
            f"expected Miller indices such as 111 or 10-10, got {text!r}"
        )
    return tuple(int(v) for v in found)


def _sample_direction(text: str) -> tuple[float, float, float]:
    """A direction of the sample: x, y, z or three numbers separated by commas."""
    named = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    if text.lower() in named:
        return named[text.lower()]
    try:
        values = tuple(float(v) for v in text.split(","))
    except ValueError:
        values = ()
    if len(values) != 3 or not any(values):
        raise argparse.ArgumentTypeError(f"expected x, y, z or three numbers a,b,c, got {text!r}")
    return values


def _fields_texture(args) -> int:
    """``morphogen texture`` on the fields of a loading path: the texture of the lattices
    of its crystals (a .vti or a .vtu file with their orientations)."""
    from morphogen import texture

    euler, weights = pipeline.read_path_texture(args.map)
    family = {"m-3": "cubic", "6/m": "hexagonal"}.get(args.lattice, args.lattice)
    defaults = {"cubic": ((1, 0, 0), (1, 1, 0), (1, 1, 1)),
                "hexagonal": ((0, 0, 0, 1), (1, 0, -1, 0), (1, 0, -1, 1))}  # fmt: skip
    shown = [texture.pole_figure(euler, indices, family, weights=weights,
                                 halfwidth=args.halfwidth)
             for indices in args.poles or defaults[family]]  # fmt: skip
    inverse = texture.inverse_pole_figure(euler, args.direction, family, weights=weights,
                                          halfwidth=args.halfwidth)  # fmt: skip
    names = ["".join(str(int(v)) for v in f.indices) for f in shown]
    summary = {
        "file": str(args.map),
        "crystals": len(euler),
        "lattice": family,
        "pole_figures": {n: f.maximum for n, f in zip(names, shown, strict=True)},
        "inverse_pole_figure": inverse.maximum,
        "files": [],
    }
    if args.output:
        from morphogen.viz import figures
        from morphogen.viz.export import save_figure

        folder = Path(args.output)
        name = {(1.0, 0.0, 0.0): "x", (0.0, 1.0, 0.0): "y", (0.0, 0.0, 1.0): "z"}.get(
            tuple(args.direction), "d"
        )
        written = [
            save_figure(figures.pole_figures_figure(shown), folder / f"pole_figures.{args.format}"),
            save_figure(figures.inverse_pole_figure_figure(inverse),
                        folder / f"inverse_pole_figure_{name}.{args.format}"),
        ]  # fmt: skip
        summary["files"] = [str(p) for p in written]
    if args.json:
        print(json.dumps(to_jsonable(summary), indent=2))
        return 0
    print(f"{args.map}: the lattices of {len(euler)} crystals ({family})")
    maxima = ", ".join(f"{{{n}}} {f.maximum:.1f}" for n, f in zip(names, shown, strict=True))
    print(f"  pole figures, largest density in multiples of random: {maxima}")
    print(f"  inverse pole figure of the direction {tuple(args.direction)}: {inverse.maximum:.1f}")
    for path in summary["files"]:
        print(f"written: {path}")
    return 0


def cmd_alloy(args) -> int:
    """The scales of an alloy (morphogen.interop.alloys): their configurations, written
    for ``morphogen run``, and their effective elasticity."""
    from morphogen.interop import alloys

    alloy = alloys.read_alloy(args.alloy)
    scales = alloys.alloy_scales(alloy)
    print(f"{alloy.alloy}: {', '.join(s['name'] for s in scales)}")
    if args.config:
        folder = Path(args.config)
        folder.mkdir(parents=True, exist_ok=True)
        for s in scales:
            cfg = alloys.alloy_configuration(alloy, s["name"], shape=args.shape)
            path = folder / f"alloy_{s['name']}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            print(f"written {path}")
    if args.homogenize:
        out = alloys.homogenize_alloy(alloy, shape=args.shape, seed=args.seed)
        for name, entry in out["scales"].items():
            shares = ", ".join(f"{k}: {v:.3f}" for k, v in entry["fractions"].items())
            print(f"{name}: E = {entry['young']:.6g}, nu = {entry['poisson']:.4f} ({shares})")
        if "predicted_young" in out:
            gap = 100 * out["departure"]
            print(f"predicted E = {out['predicted_young']:.6g}: departure {gap:+.2f} %")
        if args.json:
            Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


def cmd_texture(args) -> int:
    """The grains and the texture of an orientation map (an .ang or a .ctf file), or the
    texture of the lattices of the fields of a loading path (.vti, .vtu)."""
    from morphogen.io import ebsd as maps
    from morphogen.viz.png import encode_png, upscale

    if Path(args.map).suffix.lower() in (".vti", ".vtu"):
        return _fields_texture(args)
    measured = maps.read_ebsd(args.map, axes=args.axes, setting=args.setting)
    grains = measured.grains(args.threshold, min_size=args.min_size)
    ny, nx = measured.shape
    summary = {
        "file": str(args.map),
        "points": [nx, ny],
        "step": list(measured.step),
        "size": list(measured.size),
        "indexed": float(measured.indexed.mean()),
        "threshold": args.threshold,
        "min_size": args.min_size,
        **grains.summary(poles=args.poles, halfwidth=args.halfwidth),
    }
    written = []
    if args.output:
        folder = Path(args.output)
        folder.mkdir(parents=True, exist_ok=True)
        written.append(io.write_table(folder / "grains.csv", grains.table()))
        pairs, lengths, angles = grains.boundaries()
        boundaries = {"grain_a": pairs[:, 0], "grain_b": pairs[:, 1], "length": lengths,
                      "disorientation": angles}  # fmt: skip
        written.append(io.write_table(folder / "boundaries.csv", boundaries))
        name = {(1.0, 0.0, 0.0): "x", (0.0, 1.0, 0.0): "y", (0.0, 0.0, 1.0): "z"}.get(
            tuple(args.direction), "d"
        )
        colours = np.rint(255.0 * measured.colours(args.direction)).astype(np.uint8)
        target = folder / f"map_ipf_{name}.png"
        target.write_bytes(encode_png(upscale(colours, max(1, 512 // max(nx, ny)))))
        written.append(target)
        written += _texture_figures(folder, measured, grains, args, name)
    if args.convert:
        cleaned = measured
        if args.clean:  # the map of the grains: their mean orientations at every point
            cleaned = maps.EBSDMap.from_grains(grains.labels, grains.orientations, grains.step,
                                               phase=grains.phase, phases=grains.phases,
                                               unit=grains.unit)  # fmt: skip
        written.append(maps.write_ebsd(args.convert, cleaned, axes=args.axes, setting=args.setting))
    summary["files"] = [str(p) for p in written]
    if args.json:
        print(json.dumps(to_jsonable(summary), indent=2))
        return 0
    print(f"{args.map}: {nx} x {ny} points every {measured.step[0]:g} x {measured.step[1]:g} "
          f"{measured.unit}, {100.0 * summary['indexed']:.1f} % indexed")  # fmt: skip
    print(f"grains (neighbouring points less than {args.threshold:g} degrees apart, "
          f"{args.min_size} points at least): {grains.count}")  # fmt: skip
    for entry in summary["phases"]:
        print(f"  {entry['name']} ({entry['lattice']}): {entry['grains']} grains, "
              f"{100.0 * entry['area_fraction']:.1f} % of the area")  # fmt: skip
        print(f"    mean diameter {entry['mean_diameter']:.3g} {measured.unit} "
              f"({entry['mean_diameter_by_area']:.3g} by area), spread inside the grains "
              f"{entry['mean_spread']:.2f} degrees")  # fmt: skip
        if "mean_disorientation" in entry:
            random = " (40.7 for crystals at random)" if entry["lattice"] == "cubic" else ""
            print(f"    {entry['boundaries']} boundaries, mean disorientation "
                  f"{entry['mean_disorientation']:.1f} degrees{random}")  # fmt: skip
        if entry.get("pole_figures"):
            maxima = ", ".join(f"{{{k}}} {v:.1f}" for k, v in entry["pole_figures"].items())
            print(f"    pole figures, largest density in multiples of random: {maxima}")
    for path in written:
        print(f"written: {path}")
    return 0


def _texture_figures(folder: Path, measured, grains, args, name: str) -> list[Path]:
    """The figures of ``morphogen texture`` (matplotlib): the map with its boundaries,
    the pole figures and the inverse pole figure of each cubic or hexagonal phase, the
    disorientations of the boundaries."""
    from morphogen import texture

    try:
        from morphogen.viz import figures
        from morphogen.viz.export import save_figure

        figures._figure()
    except ImportError:
        log.warning("no figures without matplotlib: pip install 'morphogen[plot]'")
        return []
    written = []

    def save(fig, stem: str) -> None:
        written.append(save_figure(fig, folder / f"{stem}.{args.format}"))

    save(figures.ipf_map_figure(measured, args.direction, grains=grains), f"map_ipf_{name}_grains")
    defaults = {"cubic": ((1, 0, 0), (1, 1, 0), (1, 1, 1)),
                "hexagonal": ((0, 0, 0, 1), (1, 0, -1, 0), (1, 0, -1, 1))}  # fmt: skip
    pairs, lengths, angles = grains.boundaries()
    for k, phase in enumerate(grains.phases, start=1):
        of = grains.phase == k
        if not of.any():
            continue
        suffix = f"_phase{k}" if len(grains.phases) > 1 else ""
        lattice = grains.lattice(k)
        family = {"m-3": "cubic", "6/m": "hexagonal"}.get(lattice, lattice)
        if family in defaults:
            ratio = phase.c_over_a if family == "hexagonal" else None
            shown = [texture.pole_figure(grains.orientations[of], indices, family,
                                         weights=grains.areas[of], halfwidth=args.halfwidth,
                                         c_over_a=ratio)
                     for indices in args.poles or defaults[family]]  # fmt: skip
            save(figures.pole_figures_figure(shown, title=phase.name or None),
                 f"pole_figures{suffix}")  # fmt: skip
            inverse = texture.inverse_pole_figure(grains.orientations[of], args.direction, family,
                                                  weights=grains.areas[of],
                                                  halfwidth=args.halfwidth)  # fmt: skip
            points = texture.inverse_poles(grains.orientations[of], args.direction, family)
            save(figures.inverse_pole_figure_figure(inverse, points=points),
                 f"inverse_pole_figure_{name}{suffix}")  # fmt: skip
        between = of[pairs[:, 0]] if len(pairs) else np.zeros(0, dtype=bool)
        if between.any():
            save(figures.disorientation_figure(angles[between], lengths[between], lattice=lattice),
                 f"disorientations{suffix}")  # fmt: skip
    return written


def cmd_analyze(args) -> int:
    image = io.load_image(args.image, name=args.array)
    labels = None
    if "cells" in args.measure:
        labels = image
    out = analysis.measure(image, [m for m in args.measure], labels=labels)
    out = {"file": str(args.image), "shape": list(image.shape), **out}
    if args.json:
        print(json.dumps(to_jsonable(out), indent=2))
    else:
        for key, value in out.items():
            print(f"{key:>20s}: {value}")
    return 0


def _sizes(text: str) -> list[int]:
    """Box sizes, in voxels per edge: integers separated by commas."""
    try:
        return [int(v) for v in text.replace(" ", "").split(",") if v]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected integers separated by commas, got {text!r}"
        ) from None


def _percolation_scaling(args, cfg) -> int:
    """``morphogen percolation --sizes``: the thresholds of the boxes, extrapolated."""
    from morphogen.analysis.percolation import family_from_config, percolation_scaling

    result = percolation_scaling(
        family_from_config(cfg, args.parameter, sized=True),
        args.sizes,
        args.lower,
        args.upper,
        nu=args.nu,
        fit_nu=args.fit_nu,
        phase=args.phase,
        axis=args.axis,
        connectivity=args.connectivity,
        periodic=args.periodic,
        method=args.method,
        realizations=args.realizations,
        tol=args.tol,
        seed=cfg.seed if args.seed is None else args.seed,
    )
    if args.json:
        out = {
            "config": str(args.config),
            "model": cfg.model.name,
            "parameter": args.parameter,
            "method": args.method,
            "threshold": result.threshold,
            "stderr": result.stderr,
            "amplitude": result.amplitude,
            "nu": result.nu,
            "nu_stderr": result.nu_stderr,
            "realizations": args.realizations,
            "sizes": result.sizes,
            "thresholds": result.thresholds,
            "brackets": result.brackets,
            "residuals": result.residuals,
        }
        print(json.dumps(to_jsonable(out), indent=2))
        return 0
    along = {"any": "any axis", "all": "the three axes"}.get(args.axis, args.axis)
    nu = f"nu = {result.nu:.4g}"
    if result.nu_stderr is not None:
        nu += f" +/- {result.nu_stderr:.2g} fitted"
    print(
        f"percolation threshold of phase {args.phase} of {cfg.model.name} along {along}, "
        f"extrapolated from {len(result.sizes)} box sizes: {args.parameter} = "
        f"{result.threshold:.4g} +/- {result.stderr:.2g} (a = {result.amplitude:.4g}, {nu}; "
        f"{args.realizations} realizations{', forest fire' if args.method == 'fire' else ''})"
    )
    print(f"{'size':>8s}  {args.parameter:>20s}  {'bracket':>21s}  residual")
    for size, threshold, (lo, hi), residual in zip(
        result.sizes, result.thresholds, result.brackets, result.residuals, strict=True
    ):
        print(f"{size:8d}  {threshold:20.4g}  {f'[{lo:.4g}, {hi:.4g}]':>21s}  {residual:+.2e}")
    return 0


def _directions(text: str):
    """Directions of percolation: axes, lattice (the axes and the diagonals) or a number."""
    if text in ("axes", "lattice"):
        return text
    try:
        count = int(text)
    except ValueError:
        count = 0
    if count < 1:
        raise argparse.ArgumentTypeError(f"expected axes, lattice or a number, got {text!r}")
    return count


def _percolation_anisotropy(args, cfg) -> int:
    """``morphogen percolation --directions``: the thresholds along directions, their
    anisotropy and the threshold tensor."""
    from morphogen.analysis.percolation import family_from_config, percolation_anisotropy

    r = percolation_anisotropy(
        family_from_config(cfg, args.parameter),
        args.lower,
        args.upper,
        directions=args.directions,
        phase=args.phase,
        connectivity=args.connectivity,
        realizations=args.realizations,
        tol=args.tol,
        seed=cfg.seed if args.seed is None else args.seed,
    )
    names = ("directions", "thresholds", "stderr", "deviations", "deviation_stderr", "ratio",
             "ratio_stderr", "tensor", "eigenvalues", "eigenvalue_stderr", "principal",
             "residuals", "samples")  # fmt: skip
    if args.json:
        out = {"config": str(args.config), "model": cfg.model.name, "parameter": args.parameter,
               "realizations": r.realizations, **{k: getattr(r, k) for k in names}}  # fmt: skip
        print(json.dumps(to_jsonable(out), indent=2))
        return 0

    def vector(v):
        return "(" + ", ".join(f"{c:+.3f}" for c in v) + ")"

    print(
        f"percolation thresholds of phase {args.phase} of {cfg.model.name} along "
        f"{len(r.directions)} directions (across the cores of the inscribed ball, forest fire, "
        f"{r.realizations} realizations): largest over smallest {r.ratio:.4g} +/- "
        f"{r.ratio_stderr:.2g}"
    )
    if np.all(np.isfinite(r.eigenvalues)):
        axes = zip(r.eigenvalues, r.eigenvalue_stderr, r.principal, strict=True)
        terms = (f"{v:.4g} +/- {e:.2g} along {vector(a)}" for v, e, a in axes)
        print("threshold tensor: " + "; ".join(terms))
    print(f"{'direction':>24s}  {args.parameter:>20s}  {'stderr':>8s}  {'deviation':>10s}  paired")
    for n, t, e, d, p in zip(r.directions, r.thresholds, r.stderr, r.deviations,
                             r.deviation_stderr, strict=True):  # fmt: skip
        print(f"{vector(n):>24s}  {t:20.4g}  {e:8.2g}  {d:+10.2e}  {p:.2g}")
    return 0


def _fractions(text: str) -> list[float]:
    """Fractions of a sweep: numbers separated by commas."""
    try:
        return [float(v) for v in text.replace(" ", "").split(",") if v]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected numbers separated by commas, got {text!r}"
        ) from None


#: The solvers of the voxels of ``morphogen percolation --solver``, as the output names them
_SOLVED = {"fft": "FFT", "fe": "voxel FE"}


def _rigidity(args, cfg, family):
    """The rigidity threshold of the --sizes: where the elastic moduli of the phase vanish
    (``--rigidity``, and the threshold of ``--exponent elastic``)."""
    from morphogen.analysis.percolation import rigidity_threshold

    return rigidity_threshold(
        family,
        args.sizes,
        args.lower,
        args.upper,
        nu=args.nu,
        fit_nu=args.fit_nu and not args.exponent,
        phase=args.phase,
        axis=args.axis,
        cut=args.cut,
        realizations=args.realizations,
        tol=args.tol,
        device=args.device,
        solver=args.solver,
        seed=cfg.seed if args.seed is None else args.seed,
    )


def _percolation_rigidity(args, cfg) -> int:
    """``morphogen percolation --rigidity``: the rigidity threshold of the boxes, where the
    elastic moduli of the phase vanish, extrapolated."""
    from morphogen.analysis.percolation import family_from_config

    if not args.sizes:
        raise ValueError("--rigidity: give the box sizes of the finite-size scaling (--sizes)")
    if args.axis not in ("x", "y", "z"):
        raise ValueError(f"--rigidity: along one axis (x, y or z), got {args.axis!r}")
    r = _rigidity(args, cfg, family_from_config(cfg, args.parameter, sized=True))
    if args.json:
        names = ("threshold", "stderr", "amplitude", "nu", "nu_stderr", "sizes", "thresholds",
                 "brackets", "residuals", "cut", "contrast", "floppy", "rigid", "solver",
                 "solutions", "decided", "unresolved", "iterations", "residual")  # fmt: skip
        out = {"config": str(args.config), "model": cfg.model.name, "parameter": args.parameter,
               "criterion": "rigidity", "realizations": args.realizations,
               **{k: getattr(r, k) for k in names},
               "probabilities": [one.probabilities for one in r.results],
               "fractions": [one.fractions for one in r.results]}  # fmt: skip
        print(json.dumps(to_jsonable(out), indent=2))
        return 0
    nu = f"nu = {r.nu:.4g}"
    if r.nu_stderr is not None:
        nu += f" +/- {r.nu_stderr:.2g} fitted"
    print(
        f"rigidity threshold of phase {args.phase} of {cfg.model.name} along {args.axis}, "
        f"extrapolated from {len(r.sizes)} box sizes: {args.parameter} = {r.threshold:.4g} "
        f"+/- {r.stderr:.2g} (a = {r.amplitude:.4g}, {nu}; {args.realizations} realizations "
        f"bearing a uniaxial strain above {r.cut:.2g} of the Young's modulus of the phase, the "
        "other phases void)"
    )
    print(
        f"{r.solutions} {_SOLVED[r.solver]} solutions ({r.decided} images decided by their "
        f"connectivity or inclusion, {r.unresolved} energies above the cut unresolved by the "
        f"stress average), at most {r.iterations} iterations; moduli solved: floppy up to "
        f"{r.floppy:.2g}, rigid from {r.rigid:.2g}"
    )
    print(f"{'size':>8s}  {args.parameter:>20s}  {'bracket':>21s}  residual")
    for size, threshold, (lo, hi), residual in zip(
        r.sizes, r.thresholds, r.brackets, r.residuals, strict=True
    ):
        print(f"{size:8d}  {threshold:20.4g}  {f'[{lo:.4g}, {hi:.4g}]':>21s}  {residual:+.2e}")
    return 0


def _percolation_exponent(args, cfg) -> int:
    """``morphogen percolation --exponent``: the critical exponent of the conductivity or of
    the elastic moduli, by finite-size scaling at the threshold (extrapolated first by the
    same sizes unless given: that of the connectivity for the conductivity, that of the
    moduli, the rigidity, for the elastic moduli)."""
    from morphogen.analysis.percolation import (
        _extrapolated_threshold,
        family_from_config,
        percolation_exponent,
        percolation_scaling,
    )

    if not args.sizes:
        raise ValueError("--exponent: give the box sizes of the finite-size scaling (--sizes)")
    if args.axis not in ("x", "y", "z"):
        raise ValueError(f"--exponent: along one axis (x, y or z), got {args.axis!r}")
    family = family_from_config(cfg, args.parameter, sized=True)
    seed = cfg.seed if args.seed is None else args.seed
    scaling, threshold, stderr = None, args.threshold, args.threshold_stderr
    criterion = "given" if threshold is not None else "connectivity"
    if threshold is None and args.exponent == "elastic":
        criterion, scaling = "rigidity", _rigidity(args, cfg, family)
        threshold, error = _extrapolated_threshold(scaling, args.lower, args.upper)
        stderr = error if stderr is None else stderr
    elif threshold is None:
        scaling = percolation_scaling(
            family,
            args.sizes,
            args.lower,
            args.upper,
            nu=args.nu,
            phase=args.phase,
            axis=args.axis,
            connectivity=args.connectivity,
            periodic=args.periodic,
            method=args.method,
            realizations=args.realizations,
            tol=args.tol,
            seed=seed,
        )
        threshold, error = _extrapolated_threshold(scaling, args.lower, args.upper)
        stderr = error if stderr is None else stderr
    r = percolation_exponent(
        family,
        args.sizes,
        threshold,
        threshold_stderr=stderr or 0.0,
        fractions=args.fractions or (),
        physics=args.exponent,
        phase=args.phase,
        axis=args.axis,
        contrast=args.contrast,
        nu=args.nu,
        realizations=args.realizations,
        device=args.device,
        solver=args.solver,
        seed=None if seed is None else seed + 1,
    )
    name = "t" if r.physics == "thermal" else "f"
    total = float(np.hypot(r.stderr, r.threshold_error))
    if args.json:
        out = {"config": str(args.config), "model": cfg.model.name, "parameter": args.parameter,
               "physics": r.physics, "threshold": r.threshold,
               "threshold_stderr": stderr or 0.0, "exponent": r.exponent, "stderr": r.stderr,
               "threshold_error": r.threshold_error, "error": total, "ratio": r.ratio,
               "amplitude": r.amplitude, "nu": r.nu, "contrast": r.contrast,
               "realizations": r.realizations, "sizes": r.sizes, "means": r.means,
               "mean_stderr": r.mean_stderr, "residuals": r.residuals,
               "fractions": r.fractions, "values": r.values, "value_stderr": r.value_stderr,
               "sweep": r.sweep, "sweep_exponents": r.sweep_exponents,
               "sweep_stderr": r.sweep_stderr, "sweep_threshold_error": r.sweep_threshold_error,
               "solver": r.solver, "iterations": r.iterations, "residual": r.residual}  # fmt: skip
        out["threshold_criterion"] = criterion
        if scaling is not None:
            out["threshold_sizes"], out["thresholds"] = scaling.sizes, scaling.thresholds
        print(json.dumps(to_jsonable(out), indent=2))
        return 0
    prop = "conductivity" if r.physics == "thermal" else "uniaxial-strain modulus"
    origin = {"given": "given", "connectivity": "extrapolated",
              "rigidity": "extrapolated from the moduli"}[criterion]  # fmt: skip
    print(
        f"critical exponent of the {prop} of phase {args.phase} of {cfg.model.name} along "
        f"{args.axis}: {name} = {r.exponent:.3f} +/- {total:.2g} ({name}/nu = {r.ratio:.4g}, "
        f"nu = {r.nu:.4g}; statistical {r.stderr:.2g}, from the threshold {r.threshold_error:.2g})"
    )
    print(
        f"at the threshold {args.parameter} = {r.threshold:.4g} +/- {stderr or 0.0:.2g} "
        f"({origin}), contrast {r.contrast:.2g}, {r.realizations} realizations per size, at "
        f"most {r.iterations} iterations of the {_SOLVED[r.solver]} solver"
    )
    print(f"{'size':>8s}  {'mean':>12s}  {'stderr':>10s}  residual")
    for size, mean, error, residual in zip(r.sizes, r.means, r.mean_stderr, r.residuals,
                                           strict=True):  # fmt: skip
        print(f"{size:8d}  {mean:12.4e}  {error:10.2e}  {residual:+.2e}")
    if r.sweep.size:
        fractions = ", ".join(f"{f:.4g}" for f in r.sweep)
        print(f"sweep over {args.parameter} = {fractions}:")
        for size, e, s, d in zip(r.sizes, r.sweep_exponents, r.sweep_stderr,
                                 r.sweep_threshold_error, strict=True):  # fmt: skip
            print(f"{size:8d}  {name} = {e:.3f} +/- {s:.2g} (from the threshold {d:.2g})")
    return 0


def cmd_percolation(args) -> int:
    from morphogen.analysis.percolation import family_from_config, percolation_threshold

    cfg = load_config(args.config)
    if args.rigidity and args.exponent == "thermal":
        raise ValueError("--rigidity: the threshold of the elastic moduli, for --exponent elastic")
    if args.connectivity is None:  # the FFT solvers join voxels through their nodes
        args.connectivity = 26 if args.exponent else 6
    if args.exponent:
        if args.periodic is None:  # and are periodic
            args.periodic = True
        return _percolation_exponent(args, cfg)
    if args.rigidity:
        return _percolation_rigidity(args, cfg)
    if args.directions is not None:
        return _percolation_anisotropy(args, cfg)
    if args.sizes:
        return _percolation_scaling(args, cfg)
    result = percolation_threshold(
        family_from_config(cfg, args.parameter),
        args.lower,
        args.upper,
        phase=args.phase,
        axis=args.axis,
        connectivity=args.connectivity,
        periodic=args.periodic,
        method=args.method,
        realizations=args.realizations,
        tol=args.tol,
        seed=cfg.seed if args.seed is None else args.seed,
    )
    lo, hi = result.bracket
    if args.json:
        out = {
            "config": str(args.config),
            "model": cfg.model.name,
            "parameter": args.parameter,
            "method": args.method,
            "threshold": result.threshold,
            "bracket": [lo, hi],
            "realizations": result.realizations,
            "fractions": result.fractions,
            "probabilities": result.probabilities,
        }
        print(json.dumps(to_jsonable(out), indent=2))
        return 0
    along = {"any": "any axis", "all": "the three axes"}.get(args.axis, args.axis)
    print(
        f"percolation threshold of phase {args.phase} of {cfg.model.name} along {along}: "
        f"{args.parameter} = {result.threshold:.4g} in [{lo:.4g}, {hi:.4g}] "
        f"({result.realizations} realizations{', forest fire' if args.method == 'fire' else ''})"
    )
    print(f"{args.parameter:>20s}  probability")
    for fraction, probability in zip(result.fractions, result.probabilities, strict=True):
        print(f"{fraction:20.4g}  {probability:11.3f}")
    return 0


#: The theories of the beams of ``morphogen frame-stiffness``
BEAM_THEORIES = ("euler-bernoulli", "timoshenko")


def _realization(path: Path, seed: int | None, realization: int):
    """A realization of a file and its configuration: a geometry (.json, as ``morphogen
    run`` writes them; None) or a configuration, whose model is realized in its domain as
    the realization ``realization`` of its batch run (``seed``: the configuration's when
    None)."""
    from morphogen.models.base import Microstructure
    from morphogen.seeding import derive_seed

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "kind" in data:
            if seed is not None or realization:
                raise ValueError("--seed, --realization: the realizations of a configuration")
            return Microstructure.from_dict(data), None
    cfg = load_config(path)
    if realization < 0:
        raise ValueError("--realization: an index from 0")
    seed = cfg.seed if seed is None else seed
    return cfg.model.generate(cfg.domain, derive_seed(seed, realization)), cfg


def cmd_frame_stiffness(args) -> int:
    """The stiffness of the frame of beams of a realization of a network model (its
    framework, kept in its metadata): the stiffness, engineering constants, energy split,
    affine ratio and non-affinity, and Ban et al.'s softening coefficients, against
    replicas of the stiffness of the beams on request."""
    from morphogen.homogenization.frames import cowper_coefficient
    from morphogen.homogenization.networks import (
        frame_stiffness,
        network_frame,
        softening,
        softening_coefficients,
    )
    from morphogen.homogenization.postprocess import notation

    if args.replicas:
        if not args.softening:
            raise ValueError("--replicas: of the modes of --softening")
        if args.replicas < 2:
            raise ValueError("--replicas: at least 2 (0: the coefficients alone)")
        if min(args.cv) < 0 or (args.distribution == "bimodal" and max(args.cv) >= 1):
            raise ValueError("--cv: non-negative (below 1 for the bimodal law)")
    elif args.table or args.figure:
        raise ValueError("--table, --figure: of the replicas (--replicas N)")
    micro, cfg = _realization(Path(args.file), args.seed, args.realization)
    timoshenko = args.beams == "timoshenko"
    if args.shear_coefficient is not None and not timoshenko:
        raise ValueError("--shear-coefficient: of Timoshenko beams (--beams timoshenko)")
    kappa = None
    if timoshenko:
        kappa = "cowper" if args.shear_coefficient is None else args.shear_coefficient
    frame = network_frame(micro, young=args.young, poisson=args.poisson, shear_coefficient=kappa,
                          rigid_joints=args.rigid_joints)  # fmt: skip
    result = frame_stiffness(frame)
    dim = result.dimension
    axis = "xyz".index(args.axis)
    if axis >= dim:
        raise ValueError("--axis: x or y for a plane network")
    out = {
        "file": str(args.file),
        "model": micro.model.name if micro.model is not None else micro.kind,
        "seed": None if cfg is None else (cfg.seed if args.seed is None else args.seed),
        "realization": None if cfg is None else args.realization,
        "dimension": dim,
        "nodes": len(frame.nodes),
        "beams": len(frame.beams),
        "radius": float(np.sqrt(frame.area[0] / np.pi)),
        "theory": args.beams,
        "shear_coefficient": None,
        "rigid_joints": args.rigid_joints,
        "young": args.young,
        "poisson": args.poisson,
        "fibre_volume_fraction": result.density,
        "mean_length": result.mean_length,
        "stiffness": result.stiffness,
        "compliance": result.compliance(),
        "notation": notation(result.stiffness),
        "affine": result.affine,
        "engineering": result.engineering(),
        "axes": result.by_axis(),
        "summary": result.summary(args.young),
    }
    if timoshenko:
        cowper = kappa == "cowper"
        out["shear_coefficient"] = float(cowper_coefficient(args.poisson) if cowper else kappa)
    studies = []
    if args.softening:
        out["softening"] = {"axis": args.axis}
        for mode in dict.fromkeys(args.softening):
            if args.replicas:
                options = {"law": args.distribution, "axis": axis, "seed": args.replica_seed}
                study = softening(frame, mode, args.cv, replicas=args.replicas, **options)
                studies.append(study)
                c = study.coefficients
            else:
                c = softening_coefficients(frame, mode, axis)
            out["softening"][mode] = {"young": c["young"], "s": c["mean"], "v": c["variance"]}
    written = []
    if studies:
        rows = [row for study in studies for row in study.rows()]
        out["replicas"] = {"axis": args.axis, "distribution": args.distribution,
                           "replicas": args.replicas, "seed": args.replica_seed,
                           "cv": list(args.cv), "rows": rows}  # fmt: skip
        if args.table:
            with open(args.table, "w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            written.append(f"table of the replicas written to {args.table}")
        if args.figure:
            from morphogen.viz.export import save_figure
            from morphogen.viz.figures import softening_figure

            title = (f"{out['model']}: replicas along {args.axis}, {args.distribution} law, "
                     f"{args.replicas} per CV")  # fmt: skip
            save_figure(softening_figure(studies, title=title), args.figure)
            written.append(f"figure of the replicas written to {args.figure}")
    if args.json:
        print(json.dumps(to_jsonable(out), indent=2))
    else:
        _print_frame_stiffness(out)
        for line in written:
            print(line)
    return 0


def _print_frame_stiffness(out: dict) -> None:
    """The text of ``morphogen frame-stiffness`` (``out``: its JSON output)."""
    plane = out["dimension"] == 2
    seed = ""
    if out["realization"] is not None:
        seed = f", realization {out['realization']}"
        seed += "" if out["seed"] is None else f" of seed {out['seed']}"
    beams = "Timoshenko" if out["theory"] == "timoshenko" else "Euler-Bernoulli"
    if out["shear_coefficient"] is not None:
        beams += f" beams (shear coefficient {out['shear_coefficient']:.4g})"
    else:
        beams += " beams"
    joints = ""
    if out["rigid_joints"] is not None:
        joints = f", rigid over {out['rigid_joints']:g} radii at the joints"
    print(
        f"{out['model']}{seed}: a {'plane ' if plane else ''}frame of {out['beams']} {beams} "
        f"of radius {out['radius']:.4g} between {out['nodes']} nodes{joints} (fibres "
        f"E = {out['young']:g}, nu = {out['poisson']:g}); fibre volume "
        f"{'per unit area' if plane else 'fraction'} {out['fibre_volume_fraction']:.4g}, mean "
        f"length {out['mean_length']:.4g}"
    )
    units = ("of the modulus of the fibres", "its inverse")
    if plane:
        units = ("of the modulus of the fibres times a length (per unit area)", "its inverse")
    _print_elastic(out["stiffness"], out["compliance"], units)
    constants = ", ".join(f"{k} {v:.5g}" for k, v in out["engineering"].items())
    print(f"engineering constants: {constants}")
    rows = out["axes"]
    parts = [k for k in ("stretching", "bending", "torsion", "shear") if k in rows[0]]
    print(f"{'uniaxial stress':>15s} {'E':>11s} {'nu':>7s} {'E/E_affine':>10s}"
          + "".join(f" {k:>10s}" for k in parts) + f" {'non-affinity':>12s}")  # fmt: skip
    for row in rows:
        print(f"{'along ' + row['axis']:>15s} {row['young']:11.5g} {row['poisson']:7.4f} "
              f"{row['affinity']:10.4f}" + "".join(f" {row[k]:10.4f}" for k in parts)
              + f" {row['nonaffinity']:12.4g}")  # fmt: skip
    means = ", ".join(f"{k} {v:.5g}" for k, v in out["summary"].items())
    print(f"means over the axes, the moduli relative to that of the fibres: {means}")
    if "softening" in out:
        print("softening of non-identical beams (Ban et al. 2016) under a uniaxial stress "
              f"along {out['softening']['axis']}: (E0 - <E>)/E0 = s CV^2, var(E)/E0^2 = v "
              "CV^2")  # fmt: skip
        print(f"{'mode':>10s} {'s':>9s} {'v':>11s}")
        for mode, c in out["softening"].items():
            if mode != "axis":
                print(f"{mode:>10s} {c['s']:9.4g} {c['v']:11.4g}")
    if "replicas" in out:
        r = out["replicas"]
        print(f"replicas ({r['distribution']} law, {r['replicas']} per CV, seed {r['seed']}): "
              "the mean of E/E0, plain and corrected by the control variate, against 1 - s "
              "CV^2; var(E)/E0^2 against v CV^2")  # fmt: skip
        print(f"{'mode':>10s} {'CV':>6s} {'<E>/E0':>9s} {'+-':>8s} {'corrected':>9s} {'+-':>8s} "
              f"{'1 - s CV^2':>10s} {'var/E0^2':>10s} {'+-':>8s} {'v CV^2':>10s}")  # fmt: skip
        for row in r["rows"]:
            print(f"{row['mode']:>10s} {row['cv']:6.3g} {row['mean']:9.6f} "
                  f"{row['mean_error']:8.2g} {row['corrected']:9.6f} "
                  f"{row['corrected_error']:8.2g} {row['theory']:10.6f} {row['scatter']:10.4g} "
                  f"{row['scatter_error']:8.2g} {row['theory_scatter']:10.4g}")  # fmt: skip


def cmd_fit_rule(args) -> int:
    """Cellular-automaton rule whose structures match the morphology of an image, on the
    square grid or on the cells of another tiling (``--tiling``)."""
    from morphogen.automata.tilings import PLANAR, QUASIPERIODIC
    from morphogen.design.search import fit_rule

    image = io.load_image(args.image, name=args.array)
    if image.data.dtype.kind == "f":
        raise ValueError("fit-rule needs a phase image (not partial volumes)")
    planar = image.shape[2] == 1
    if planar and args.tiling not in PLANAR:
        raise ValueError(f"--tiling {args.tiling}: three-dimensional images only")
    n = args.lattice or (64 if planar else 24)
    shape = (n, n, 1) if planar else (n, n, n)

    def progress(generation, genome, cost):
        log.info("generation %d: cost %.4g", generation, cost)

    fit = fit_rule(
        image,
        neighbourhood=args.neighbourhood,
        dims=2 if planar else 3,
        lattice=args.tiling,
        shape=shape,
        steps=args.steps,
        initial_fraction=args.initial_fraction,
        population=args.population,
        generations=args.generations,
        rng=np.random.default_rng(args.seed),
        callback=progress,
    )
    names = ["volume_fraction", "interface_density"] + [
        f"correlation_{h}" for h in range(1, len(fit.target) - 1)
    ]
    out = {
        "rule": fit.rule,
        "tiling": args.tiling,
        "initial_fraction": fit.initial_fraction,
        "cost": fit.cost,
        "evaluations": fit.evaluations,
        "target": dict(zip(names, fit.target.tolist(), strict=True)),
        "fitted": dict(zip(names, fit.descriptors.tolist(), strict=True)),
    }
    if args.output:
        h = float(np.mean(image.spacing[:2] if planar else image.spacing))
        lengths = [float(v) for v in np.asarray(image.shape) * np.asarray(image.spacing)]
        config = {
            "model": {
                "type": "cellular_automaton",
                "rule": fit.rule,
                "neighbourhood": args.neighbourhood,
                "steps": args.steps,
                "initial_fraction": round(fit.initial_fraction, 4),
                "cell_size": h,
                "dimension": 2 if planar else 3,
            },
            "domain": {"size": lengths, "periodic": args.tiling not in QUASIPERIODIC},
            "seed": 1,
            "voxels": {"shape": list(image.shape)},
            "outputs": {"directory": "out/fitted_rule", "name": "automaton", "image": ["vti"]},
            "analysis": ["volume_fraction", "specific_surface", "connectivity"],
        }
        if args.tiling != "square":
            config["model"]["lattice"] = args.tiling
        Path(args.output).write_text(
            f"# Cellular automaton fitted to {Path(args.image).name} (morphogen fit-rule)\n"
            + yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        out["config"] = str(args.output)
    if args.json:
        print(json.dumps(to_jsonable(out), indent=2))
    else:
        on = f", {args.tiling} tiling" if args.tiling != "square" else ""
        print(f"rule: {fit.rule}  (initial fraction {fit.initial_fraction:.3f}{on})")
        print(f"cost: {fit.cost:.4g} after {fit.evaluations} evaluations")
        print(f"{'descriptor':>18s} {'target':>9s} {'fitted':>9s}")
        for name in names:
            print(f"{name:>18s} {out['target'][name]:9.4f} {out['fitted'][name]:9.4f}")
        if args.output:
            print(f"configuration written to {args.output}")
    return 0


def cmd_learn(args) -> int:
    """A behaviour learned from data (:mod:`morphogen.learning`) of a configuration file:
    the data, the training, the errors, the file of the model."""
    from morphogen.learning import learn

    path = Path(args.config)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = loaded.get("learning", loaded) if isinstance(loaded, dict) else None
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: a mapping of the learning (or a 'learning' mapping)")
    cfg = dict(cfg)
    # the files of the command line from the current directory, those of the file from it
    for key, value in (("output", args.output), ("save_data", args.data)):
        if value is not None:
            cfg[key] = str(Path(value).resolve())
    cfg.setdefault("output", f"{path.stem}.npz")
    overrides = (("model", "kind", args.model), ("training", "epochs", args.epochs),
                 ("data", "paths", args.paths), ("data", "seed", args.seed),
                 ("model", "seed", args.seed), ("training", "seed", args.seed))  # fmt: skip
    for section, key, value in overrides:
        if value is not None:
            cfg[section] = {**(cfg.get(section) or {}), key: value}

    def progress(epoch, train_loss, validation_loss):
        if epoch % 10 == 0:
            log.info("epoch %d: loss %.4g, validation %.4g", epoch, train_loss, validation_loss)

    _, out = learn(cfg, base=path.parent, callback=progress)
    if args.json:
        print(json.dumps(to_jsonable(out), indent=2))
        return 0
    t, v = out["training"], out["validation"]
    print(
        f"{out['model']} model, {out['paths']} paths of {out['points']} points: {t['epochs']} "
        f"epochs ({t['time_s']:.1f} s), the best {t['best_epoch']}; validation paths: relative "
        f"error {v['relative']:.4f} (mean {v['mean']:.4f}, max {v['max']:.4f})"
    )
    for name, row in out.get("tests", {}).items():
        failed = f" (failed: {row['failed']})" if "failed" in row else ""
        print(
            f"path {name}: stress gap {row['stress']:.4f}, strain gap {row['strain']:.4f}{failed}"
        )
    print(f"model: {out['output']}")
    if "save_data" in out:
        print(f"data: {out['save_data']}")
    return 0


def cmd_optimize(args) -> int:
    """Topology optimization of a periodic cell from a configuration file."""
    from morphogen.optimization.run import load_optimization_config, run_optimization

    cfg = load_optimization_config(args.config)
    if args.output is not None:
        cfg.directory = Path(args.output)
    if args.seed is not None:
        cfg.seed = args.seed

    def progress(iteration, density, objective):
        log.info(
            "iteration %d: objective %.6g, volume %.4f", iteration, objective, float(density.mean())
        )

    design, outdir = run_optimization(cfg, callback=progress)
    props = design.properties()
    print(f"results written to {outdir}")
    state = "converged" if design.converged else "not converged"
    print(
        f"{design.iterations} iterations ({state}), {design.time_s:.1f} s; volume fraction "
        f"{design.volume_fraction:.4f} (thresholded design: {design.binary_fraction:.4f})"
    )
    refined = design.check.get("properties", {})
    if refined:
        print(f"{'':>16s}  {'design grid':>12s}  {design.check['refinement']} x finer")
    for key, value in props.items():
        line = f"{key:>16s}: {value:12.6g}"
        if key in refined:
            line += f"  {refined[key]:12.6g}"
        print(line)
    tensor = design.binary_tensor
    if np.shape(tensor) == (6, 6) and np.all(np.isfinite(tensor)):  # an elastic design
        print("thresholded design:", end=" ")
        try:
            _print_elastic(tensor, units=("of the material", "its inverse"))
        except np.linalg.LinAlgError:
            print("a singular stiffness (no compliance)")
    if design.hinges:
        print(
            f"warning: {design.hinges} blocks of voxels touch only along an edge or at a "
            "corner (hinges): check the constants on the finer grid, or use a larger "
            "filter radius or a softer projection"
        )
    return 0


def cmd_laser(args) -> int:
    """Laser-architectured sheets: the designs of a configuration file, its optimization,
    the results."""
    from morphogen.laser.run import load_laser_config, run_laser

    cfg = load_laser_config(args.config)
    if args.output is not None:
        cfg.directory = Path(args.output)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.evaluations is not None and cfg.optimization is not None:
        cfg.optimization["evaluations"] = args.evaluations
    if args.no_optimization:
        cfg.optimization = None

    def progress(row):
        log.info(
            "evaluation %d: objective %.6g%s", row["evaluation"], row["objective"],
            "" if row["feasible"] else " (infeasible)",
        )  # fmt: skip

    summary, outdir = run_laser(cfg, callback=progress, log=log.info)
    if args.json:
        print(json.dumps(to_jsonable(summary), indent=2))
        return 0
    print(f"results written to {outdir}")
    if cfg.case.loading.get("kind", "tension") == "bending":
        keys = ("bending_load_at_deflection", "bending_load", "bending_stiffness",
                "bending_energy", "volume_fraction", "path_length")  # fmt: skip
        heads = ("F(d) (N)", "F max (N)", "k (N/mm)", "W (mJ)", "treated", "path (mm)")
    else:
        keys = ("yield_strength", "tensile_strength", "uniform_elongation", "energy",
                "strength_ductility", "volume_fraction", "path_length")  # fmt: skip
        heads = ("Rp (MPa)", "Rm (MPa)", "Ag", "W (MJ/m3)", "Rm Ag", "treated", "path (mm)")

    print(f"{'design':>14s}  " + "  ".join(f"{h:>10s}" for h in heads))
    for name, d in summary["designs"].items():
        m = d["measures"]
        print(f"{name:>14s}  " + "  ".join(f"{m[k]:10.4g}" for k in keys))
    opt = summary.get("optimization")
    if opt is not None:
        vals = ", ".join(f"{k} = {v:.4g}" for k, v in opt["values"].items())
        state = "" if opt["feasible"] else " (no feasible design: the least infeasible)"
        print(f"optimum ({opt['method']}, {opt['evaluations']} evaluations): {vals}{state}")
    return 0


def _view_source(path: Path, shape):
    """Microstructure or image of a file: geometry (.json), configuration (first
    realization of its model) or voxel image."""
    from morphogen.models.base import Microstructure

    ext = path.suffix.lower()
    if ext in (".json", ".yaml", ".yml", ".toml"):
        data = json.loads(path.read_text(encoding="utf-8")) if ext == ".json" else None
        if data is not None and "kind" in data:
            return Microstructure.from_dict(data), None
        cfg = load_config(path)
        return cfg.model.generate(cfg.domain, cfg.seed), cfg
    return io.load_image(path), None


def cmd_view(args) -> int:
    """Figures of a microstructure, an image or the formation of a structure."""
    from morphogen.viz import figures

    path = Path(args.file)
    micro, cfg = _view_source(path, args.shape)
    stem = path.with_suffix("")
    if args.animate:
        if cfg is None or not hasattr(cfg.model, "evolution"):
            raise ValueError("--animate needs a configuration of an automaton or growth model")
        frames = cfg.model.evolution(cfg.domain, cfg.seed, args.frames)
        axis_frames = [np.flipud(np.take(f, f.shape[2] // 2, axis=2).T) for f in frames]
        kind = "labels" if frames[-1].dtype == np.int32 else "auto"
        figures.save_gif(axis_frames, args.animate, fps=args.fps, kind=kind)
        print(f"animation written to {args.animate} ({len(frames)} frames)")
        return 0
    if args.surface:
        fig = figures.surface_figure(micro, shape=args.shape)
    elif args.tile:
        fig = figures.tiled_figure(micro, tiles=args.tile, shape=args.shape)
    elif args.history:
        history = getattr(micro, "attributes", {}).get("history")
        if history is None:
            raise ValueError("this structure has no history (automata record one)")
        fig = figures.history_figure(history)
    elif args.kinetics:
        fig = figures.kinetics_figure(micro)
    else:
        title = micro.model.name if getattr(micro, "model", None) is not None else path.name
        fig = figures.slices_figure(micro, shape=args.shape, labels=args.labels, title=title)
    out = Path(args.output) if args.output else Path(f"{stem}_view.png")
    fig.savefig(out)
    print(f"figure written to {out}")
    return 0


def _export_source(path: Path, array: str | None):
    """The object of a file to export and the arrays of a mesh: a microstructure (.json
    geometry), a table, a tetrahedral mesh (.vtu with its point and cell arrays, .mesh)
    or a voxel image (the array ``array``, or the first one)."""
    ext = path.suffix.lower()
    if ext == ".json":
        return io.load_microstructure(path), {}
    if ext in (".csv", ".txt", ".tsv", ".dat", ".xlsx"):
        return io.read_table(path), {}
    if ext in (".vtu", ".mesh"):
        arrays = {}
        if ext == ".vtu":
            from morphogen.io.vtk import read_vtu

            data = read_vtu(path)
            arrays = {**data["point_data"], **data["cell_data"]}
        return io.read_tet_mesh(path), arrays
    return io.load_image(path, array), {}


def _export_options(obj, arrays: dict, args, video: bool) -> dict:
    """The options of :func:`morphogen.export` given on the command line, for the kind of
    object and of output."""
    from morphogen.image import VoxelImage
    from morphogen.mesh.tetra import TetMesh
    from morphogen.models.base import Microstructure

    def given(**kw) -> dict:
        return {k: v for k, v in kw.items() if v is not None}

    size = args.size or []
    pixels = tuple(size) if len(size) == 2 else ((size[0], size[0] * 3 // 4) if size else None)
    side = max(size) if size else None
    lo, hi = args.clim or (None, None)
    clim = tuple(args.clim) if args.clim else None

    def array(name: str):
        if name not in arrays:
            raise ValueError(f"no array {name!r} in the mesh (arrays: {', '.join(arrays)})")
        return arrays[name]

    if args.particles is not None:  # one phase without its matrix, for the DEM
        if not isinstance(obj, (VoxelImage, Microstructure)):
            raise ValueError("--particles: the particles of a geometry (.json) or an image")
        options = {"particles": args.particles}
        if args.format == "yade":
            options["format"] = "yade"
        if args.tetrahedra:
            options["elements"] = "tetrahedra"
        if args.element_size is not None:
            options["size"] = args.element_size
        if isinstance(obj, Microstructure):
            options.update(given(shape=args.shape, voxel_size=args.voxel_size))
        return options
    if args.format == "yade":
        raise ValueError("--format yade: the spheres of the particles of a phase (--particles)")
    if args.tetrahedra or args.element_size is not None:
        raise ValueError("--tetrahedra, --element-size: the particles of a phase (--particles)")
    options = given(fps=args.fps, mode=args.mode, repeat=args.repeat) if video else {}
    if isinstance(obj, Microstructure):
        options.update(given(shape=args.shape, voxel_size=args.voxel_size))
    if isinstance(obj, (VoxelImage, Microstructure)) and video:
        kind = args.kind or "sections"
        options["kind"] = kind
        if kind == "sections":
            options.update(given(axis=args.axis, size=side, cmap=args.cmap, vmin=lo, vmax=hi))
        elif kind == "orthoslices":
            options.update(given(frames=args.frames, size=side, cmap=args.cmap, vmin=lo, vmax=hi))
        else:
            options.update(given(phase=args.phase, frames=args.frames, size=pixels))
    elif isinstance(obj, TetMesh) and (
        video or Path(args.output).suffix.lower() in (".png", ".tif", ".jpg")
    ):
        field = None
        if args.field:
            field = "phase" if args.field == "phase" else array(args.field)
        options.update(given(field=field, size=pixels, cmap=args.cmap, clim=clim))
        if args.field:
            options.setdefault("title", args.field)
        if video:
            kind = args.kind or ("field" if args.displacement else "turntable")
            options.update(kind=kind, **given(frames=args.frames))
            if kind == "sweep":
                options.update(given(axis=args.axis))
            elif kind == "field":
                if not args.displacement:
                    raise ValueError("--kind field: give the displacement (--displacement NAME)")
                options.update(
                    displacement=array(args.displacement),
                    **given(
                        magnification=args.magnification,
                        wave=args.wave,
                        section=tuple(args.section) if args.section else None,
                    ),
                )
    return options


def cmd_export(args) -> int:
    """Convert a file (image, mesh, geometry, table) to another format, a figure or a
    video."""
    from morphogen.io.export import VIDEOS, export

    source = Path(args.input)
    obj, arrays = _export_source(source, args.array)
    out = Path(args.output)
    video = out.suffix.lower() in VIDEOS or args.format == "frames"
    written = export(obj, out, **_export_options(obj, arrays, args, video))
    files = written if isinstance(written, list) else [written]
    print("written to " + ", ".join(str(f) for f in files))
    return 0


def cmd_zmat(args) -> int:
    """How morphoGen reads a material file of Z-mat, or what of Z-mat it covers."""
    from morphogen.homogenization import zmat

    if args.coverage:
        sys.stdout.write(zmat.coverage_markdown())
        return 0
    if not args.file:
        raise ValueError("zmat: give a material file of Z-mat (or --coverage)")
    out = zmat.describe(Path(args.file), args.behaviour)
    if args.output and out["translation"] == "mfront":
        stem = zmat.mfront_name(Path(args.file).stem)
        b = zmat.read_zmat(Path(args.file))[args.behaviour]
        target = Path(args.output) / f"{stem}.mfront"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(zmat.zmat_mfront_source(b, stem), encoding="utf-8")
        out["written"] = str(target)
    if args.json:
        print(json.dumps(to_jsonable(out), indent=2))
    elif out["translation"] == "native":
        print("morphoGen's kernels (finite elements, FFT, GPU); the material of a configuration:")
        for note in out.get("notes", []):
            print(f"  note: {note}")
        print(yaml.safe_dump(to_jsonable(out["material"]), sort_keys=False).rstrip())
    else:
        print(f"an MFront behaviour ({out['reason']}): MFront compiles it, MGIS integrates it;")
        print(out.get("written") or out["source"])
    return 0


def cmd_gui(args) -> int:
    """Graphical interface in the web browser."""
    from morphogen.gui import serve

    serve(args.port, open_browser=not args.no_browser)
    return 0


def cmd_desktop(args) -> int:
    """The graphical interface in a window of its own (the stand-alone application)."""
    from morphogen.gui.desktop import run

    return run(session_dir=args.session, debug=args.debug, browser=args.browser)


def _radioss_print(out: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(to_jsonable(out), indent=2))
        return
    for key, value in out.items():
        if isinstance(value, float):
            value = f"{value:.6g}"
        print(f"{key:>34s}: {value}")


def _radioss_mesh(args, materials):
    """The mesh of the source of ``morphogen radioss export``: a geometry (.json) meshed by
    its voxels or conformingly, or the voxels of an image."""
    from morphogen.dynamics import explicit_mesh

    path = Path(args.source)
    if path.suffix.lower() == ".json":
        micro = io.load_microstructure(path)
        mesh = explicit_mesh(
            micro,
            materials,
            mesh=args.mesh,
            resolution=args.voxels or 24,
            element_size=args.element_size,
            order=args.order,
        )
        names = {int(k): str(v) for k, v in (getattr(micro, "phase_names", None) or {}).items()}
        return mesh, names, getattr(micro, "kind", None) or path.stem
    if args.mesh != "voxels":
        raise ValueError("--mesh conforming: a geometry (.json), not an image")
    from morphogen.mesh.voxel import voxel_mesh

    image = io.load_image(path, name=args.array)
    present = {int(q) for q in np.unique(image.data)}
    missing = sorted(present - set(materials))
    if missing:
        raise ValueError(f"phases {missing} of the image: no material (give theirs, or 'void')")
    solid = sorted(q for q in present if materials[q] is not None)
    if not solid:
        raise ValueError("every phase of the image is void")
    return voxel_mesh(image, solid if len(solid) < len(present) else None), {}, path.stem


def _radioss_loading(args):
    from morphogen.dynamics import ExplicitLoading

    spec = {
        "kind": args.loading,
        "axis": args.axis,
        "strain": args.strain,
        "lateral": args.lateral,
        "contact": not args.no_contact,
        "friction": args.friction,
    }
    if args.rate is not None:
        spec["strain_rate"] = args.rate
    if args.velocity is not None:
        spec["velocity"] = args.velocity
    if args.duration is not None:
        spec["duration"] = args.duration
    if args.shear_axis is not None:
        spec["shear_axis"] = args.shear_axis
    return ExplicitLoading.from_config(spec)


def _radioss_summary(directory: Path, args) -> int:
    from morphogen.dynamics import read_explicit_run, summary, write_frames, write_tables

    result = read_explicit_run(directory)
    out = summary(result)
    out["forces from"] = result.source
    write_tables(result, directory)
    if args.frames_vtu:
        out["frames written"] = len(write_frames(result, directory / "frames"))
    for note in result.warnings:
        print(f"warning: {note}", file=sys.stderr)
    _radioss_print(out, args.json)
    return 0


def cmd_radioss(args) -> int:
    """OpenRadioss, the explicit solver: its installation, the inputs of a morphology,
    their runs and their results."""
    from morphogen import dynamics

    if args.action == "info":
        try:
            install = dynamics.find_openradioss(args.path)
        except FileNotFoundError as exc:
            if args.json:
                print(json.dumps({"found": False, "reason": str(exc)}, indent=2))
            else:
                print(f"OpenRadioss: not found ({exc})")
            return 1
        _radioss_print({"found": True, **install.describe()}, args.json)
        return 0
    if args.action == "install":

        def report(fraction, message):
            print(f"\r{100 * fraction:5.1f} % {message[:70]:<70s}", end="", file=sys.stderr)

        install = dynamics.install_openradioss(args.output, tag=args.tag, progress=report)
        print(file=sys.stderr)
        _radioss_print(install.describe(), args.json)
        return 0
    if args.action == "export":
        if not args.materials:
            raise ValueError("--materials: a YAML/JSON file of the materials of the phases")
        spec = yaml.safe_load(Path(args.materials).read_text(encoding="utf-8"))
        spec = spec.get("materials", spec) if isinstance(spec, dict) else spec
        materials = dynamics.explicit_materials(spec, density=args.density)
        mesh, names, stem = _radioss_mesh(args, materials)
        loading = _radioss_loading(args)
        target = Path(args.output or f"{Path(args.source).stem}_openradioss")
        inputs = io.write_radioss(
            target,
            mesh,
            {q: m for q, m in materials.items() if m is not None},
            loading,
            name=args.name or stem,
            units=args.units,
            frames=args.frames,
            time_step=args.time_step,
            length_scale=args.length_scale,
            phase_names=names,
            title=f"{loading.kind} of {stem}",
        )
        m = inputs.manifest
        out = {
            "directory": str(inputs.directory),
            "name": inputs.name,
            "elements": f"{m['n_elements']} {m['elements']}",
            "nodes": m["n_nodes"],
            "end time": m["end_time"],
            "stable time step (estimate)": m["critical_time_step"],
            "cycles (estimate)": round(m["cycles"]),
        }
        if args.run:
            dynamics.run_openradioss(
                target, threads=args.threads, mpi=args.mpi, installation=args.path
            )
            return _radioss_summary(Path(target), args)
        _radioss_print(out, args.json)
        return 0
    directory = Path(args.source)
    if args.action == "run":

        def report(fraction, message):
            if args.verbose:
                print(f"\r{100 * fraction:5.1f} % {message[:60]:<60s}", end="", file=sys.stderr)

        run = dynamics.run_openradioss(
            directory, threads=args.threads, mpi=args.mpi, installation=args.path, progress=report
        )
        if args.verbose:
            print(file=sys.stderr)
        for note in run.warnings:
            print(f"warning: {note}", file=sys.stderr)
    elif args.convert:  # read: the outputs converted first when they can be
        try:
            install = dynamics.find_openradioss(args.path)
        except FileNotFoundError as exc:
            print(f"warning: outputs not converted ({exc})", file=sys.stderr)
        else:
            from morphogen.io.radioss import read_manifest

            run = dynamics.RadiossRun(directory, read_manifest(directory)["name"], [], None, {})
            dynamics.convert_outputs(run, install)
    return _radioss_summary(directory, args)


def cmd_models(args) -> int:
    if args.name is None:
        for name in list_models():
            doc = (get_model(name).__doc__ or "").strip().splitlines()[0]
            print(f"{name:20s} {doc}")
        return 0
    cls = get_model(args.name)
    params = cls.parameters()
    if args.json:
        print(json.dumps({"name": cls.name, "parameters": params}, indent=2))
        return 0
    print(f"{cls.name}: {(cls.__doc__ or '').strip()}")
    print()
    print("phases: " + ", ".join(f"{k} = {v}" for k, v in cls.phases.items()))
    print("parameters:")
    for p in params:
        default = "required" if p["required"] else f"default {p['default']!r}"
        unit = f" [{p['unit']}]" if p["unit"] else ""
        print(f"  {p['name']} ({p['kind']}{unit}, {default}): {p['doc']}")
    return 0


def cmd_template(args) -> int:
    templates = _templates()
    if args.name is None:
        for name in sorted(templates):
            print(name)
        return 0
    if args.name not in templates:
        raise ValueError(
            f"unknown template {args.name!r}; available: {', '.join(sorted(templates))}"
        )
    text = templates[args.name]
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"template written to {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_info(args) -> int:
    import numba

    from morphogen.homogenization import mfront
    from morphogen.mesh.conforming import mmg_executable

    for key, value in pipeline.versions().items():
        print(f"{key:>10s}: {value}")
    try:
        mmg = mmg_executable()
    except (ImportError, FileNotFoundError) as exc:
        mmg = f"not found ({exc})"
    print(f"{'mmg3d':>10s}: {mmg}")
    from morphogen.dynamics import find_openradioss

    try:
        radioss = find_openradioss()
        radioss = f"{radioss.root}" + (f" ({radioss.version})" if radioss.version else "")
    except FileNotFoundError:
        radioss = "not found (morphogen radioss install)"
    print(f"{'openradioss':>10s}: {radioss}")
    print(f"{'threads':>10s}: {numba.get_num_threads()}")
    device = gpu.info()
    if device["gpu"] is None:
        print(f"{'gpu':>10s}: not available ({device['reason']})")
    else:
        print(f"{'gpu':>10s}: {device['gpu']} ({device['memory_gb']} GB, CuPy {device['cupy']})")
    behaviours = mfront.info()
    if not behaviours["mgis"]:
        print(f"{'mfront':>10s}: not available ({behaviours['reason']})")
    elif behaviours["mfront"] is None:
        print(f"{'mfront':>10s}: MGIS without the mfront generator (compiled libraries only)")
    else:
        print(f"{'mfront':>10s}: TFEL {behaviours['tfel']} and MGIS ({behaviours['mfront']})")
    print(f"{'models':>10s}: {', '.join(list_models())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from morphogen.automata.tilings import LATTICES
    from morphogen.homogenization.networks import LAWS as NETWORK_LAWS
    from morphogen.homogenization.networks import MODES as NETWORK_MODES

    parser = argparse.ArgumentParser(
        prog="morphogen",
        description="Morphological modelling and generation of virtual microstructures.",
    )
    parser.add_argument("--version", action="version", version=f"morphogen {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="print progress")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="generate realizations or run an RVE study from a config file")
    p.add_argument("config", help="configuration file (.yaml, .json or .toml)")
    p.add_argument("-o", "--output", help="output directory (overrides outputs.directory)")
    p.add_argument("--seed", type=int, help="base seed (overrides the configuration)")
    p.add_argument(
        "--task-id",
        default="0",
        help="index of this task when the run is split (e.g. $SLURM_ARRAY_TASK_ID), or "
        "'auto': the index of the job array of the scheduler",
    )
    p.add_argument("--n-tasks", type=int, default=1, help="number of tasks of a split run")
    p.add_argument("-j", "--jobs", type=int, default=1, help="processes running the realizations")
    p.add_argument(
        "--threads", type=int, help="number of threads (shared by the processes; default all)"
    )
    p.add_argument(
        "--resume", action="store_true", help="skip the realizations already done (records)"
    )
    p.add_argument(
        "--mpi", action="store_true", help="one task per MPI rank (mpirun ...; needs mpi4py)"
    )
    p.add_argument("-q", "--quiet", action="store_true", help="do not print the RVE report")
    p.add_argument("--db", help="add the run to this database of results (outputs.database)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "amitex", help="runs of AMITEX_FFTP on an image (export), and their results (read)"
    )
    p.add_argument("action", choices=("export", "read"), help="write the inputs, or read results")
    p.add_argument(
        "target", help="export: an image, or a geometry (.json) with --voxels; read: the "
        "directory of an export"
    )  # fmt: skip
    p.add_argument("--material", nargs="+", help="materials, as for morphogen homogenize")
    p.add_argument("--materials", help="YAML/JSON file of the materials of the phases")
    p.add_argument("--voxels", type=int, help="voxels along the longest edge of a geometry")
    p.add_argument("--array", help="array of the image file")
    p.add_argument("--loading", nargs="+", help="named macroscopic loadings ('tension: S11=1')")
    p.add_argument("--path", nargs="+", help="loading paths, as for morphogen homogenize")
    p.add_argument("--paths", help="YAML/JSON file of loading paths")
    p.add_argument(
        "--stiffness",
        action="store_true",
        help="also the six unit strains of the stiffness (default without loadings)",
    )
    p.add_argument("--fields", action="store_true", help="stress and strain fields of the runs")
    p.add_argument("--tol", type=float, help="convergence criterion of AMITEX (default 1e-4)")
    p.add_argument("--processes", type=int, default=4, help="MPI processes of the script")
    p.add_argument("--library", help="path of libUmatAmitex.so (default: through AMITEX_PATH)")
    p.add_argument(
        "--elastic-only",
        action="store_true",
        help="export the elasticity of the nonlinear materials (default: their MFront laws)",
    )
    p.add_argument("--kinematics", choices=("small", "finite"), default="small",
                   help="finite: paths in finite strains, every phase an MFront law")  # fmt: skip
    p.add_argument("--name", help="name of the files (default: the stem of the sample)")
    p.add_argument("-o", "--output", help="export: directory; read: CSV table to append to")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(func=cmd_amitex)

    p = sub.add_parser(
        "mfront", help="MFront sources of the materials (von Mises, porous plasticity, "
        "crystals, learned laws; in finite strains, elasticity too)"
    )  # fmt: skip
    p.add_argument("--material", nargs="+", help="materials, as for morphogen homogenize")
    p.add_argument("--materials", help="YAML/JSON file of the materials of the phases")
    p.add_argument("--kinematics", choices=("small", "finite"), default="small",
                   help="finite: Hencky elasticity and von Mises plasticity in the "
                   "logarithmic strain space, crystals by morphoGen's kernel")  # fmt: skip
    p.add_argument("--name", help="stem of the behaviours (default: that of --materials)")
    p.add_argument("-o", "--output", help="directory of the sources (default: here)")
    p.set_defaults(func=cmd_mfront)

    p = sub.add_parser(
        "db", help="database of the configurations tested and their results (SQLite)"
    )
    p.add_argument(
        "action",
        choices=("add", "list", "keys", "export", "remove"),
        help="add run directories, list the runs, list the keys of the records, export "
        "the records (CSV), remove runs",
    )
    p.add_argument("items", nargs="*", help="run directories (add) or run ids (remove)")
    p.add_argument("--db", default="morphogen.db", help="database file (default morphogen.db)")
    p.add_argument("--name", help="name of the runs added (default: outputs.name)")
    p.add_argument(
        "--where",
        nargs="+",
        help="conditions on the records, e.g. 'model.type == gyroid' 'solid_fraction < 0.3'",
    )
    p.add_argument("-o", "--output", help="output file of export (.csv)")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(func=cmd_db)

    p = sub.add_parser("ashby", help="Ashby chart of the records of a database")
    p.add_argument("-o", "--output", default="ashby.png", help="chart file (.png, .svg, .pdf)")
    p.add_argument("--db", default="morphogen.db", help="database file (default morphogen.db)")
    p.add_argument("--x", help="key of the x axis (default solid_fraction)")
    p.add_argument("--y", help="key of the y axis (default: a relative modulus)")
    p.add_argument("--by", default="model.type", help="key of the families ('none': one)")
    p.add_argument("--groups", nargs="+", help="families shown first (three in colour)")
    p.add_argument("--where", nargs="+", help="conditions on the records (see 'db')")
    p.add_argument("--linear", action="store_true", help="linear axes (default logarithmic)")
    p.add_argument(
        "--guides", nargs="*", type=float, help="exponents n of the lines y = x^n (default 1 2)"
    )
    p.add_argument("--theme", choices=("light", "dark"), default="light", help="colours")
    p.add_argument("--title", help="title of the chart")
    p.add_argument("--table", help="also write the points as a table (.csv)")
    p.set_defaults(func=cmd_ashby)

    p = sub.add_parser("merge", help="gather the results of the tasks of a split run")
    p.add_argument("directory", help="output directory of the run")
    p.add_argument("--no-report", action="store_true", help="do not write the RVE report")
    p.add_argument("-q", "--quiet", action="store_true", help="do not print the RVE report")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("status", help="realizations done and missing in a run directory")
    p.add_argument("directory", help="output directory of the run")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser(
        "submit",
        help="job array of a configuration on a cluster (SLURM, PBS, Torque, SGE, OAR, LSF)",
    )
    p.add_argument("config", help="configuration file (.yaml, .json or .toml)")
    p.add_argument(
        "--scheduler",
        required=True,
        choices=("slurm", "pbs", "torque", "sge", "oar", "lsf"),
        help="batch scheduler",
    )
    p.add_argument("--tasks", type=int, required=True, help="number of tasks of the array")
    p.add_argument("-j", "--jobs", type=int, default=1, help="processes per task")
    p.add_argument("--threads", type=int, default=1, help="threads per process")
    p.add_argument("--time", default="01:00:00", help="time limit of a task (HH:MM:SS)")
    p.add_argument("--memory", help="memory of a task (e.g. 8G, scheduler syntax)")
    p.add_argument("--queue", help="partition or queue")
    p.add_argument("--account", help="account or project")
    p.add_argument("--name", help="job name")
    p.add_argument(
        "--setup",
        nargs="+",
        help="shell lines run before morphoGen (e.g. 'module load python')",
    )
    p.add_argument("-o", "--output", help="output directory (overrides outputs.directory)")
    p.add_argument("--seed", type=int, help="seed of the run (default: the configuration's)")
    p.add_argument(
        "--submit", action="store_true", help="submit the jobs (default: write the scripts)"
    )
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("fit", help="RVE scaling-law fit of results tables (morphoGen or external)")
    p.add_argument("results", nargs="+", help="results CSV files or directories (results*.csv)")
    p.add_argument("--property", nargs="+", help="columns to analyse (default: all measures)")
    p.add_argument("--volume-column", default="volume", help="column holding the sample volume")
    p.add_argument("--eps", nargs="+", type=float, default=[0.01, 0.05], help="relative errors")
    p.add_argument("--n", nargs="+", type=int, default=[1, 10], help="numbers of realizations")
    p.add_argument("--gamma", type=float, help="impose the scaling exponent")
    p.add_argument("--min-volume", type=float, help="smallest volume used in the fit")
    p.add_argument("--max-volume", type=float, help="largest volume used in the fit")
    p.add_argument(
        "--group-by", help="column splitting the rows into series (e.g. a boundary condition)"
    )
    p.add_argument(
        "--bootstrap", type=int, default=1000, help="bootstrap replicates (0: no intervals)"
    )
    p.add_argument(
        "--weights",
        choices=("none", "n"),
        default="none",
        help="weights of the sizes in the fit ('n': (n - 1) / 2)",
    )
    p.add_argument("-o", "--output", help="directory for rve_report.json/.md/.tex and figures")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser(
        "homogenize",
        help="apparent elastic and thermal properties of a sample: finite elements on a "
        "conforming mesh, or FFT on a voxel image",
    )
    p.add_argument(
        "sample",
        help="geometry (.json, meshed with --size), tetrahedral mesh (.vtu, .mesh) or phase "
        "image (.vti, .vtk, .mhd, .npz, .npy, .tif: FFT solver, periodic conditions)",
    )
    p.add_argument("--array", help="array name in multi-array image files")
    p.add_argument(
        "--size",
        type=float,
        help="element size of the conforming mesh of a geometry (.json; default: the shortest "
        "edge of its domain over 10, a light mesh): finite elements, unless --voxels",
    )
    p.add_argument(
        "--voxels",
        type=int,
        help="FFT solver on an image of N voxels per edge of the geometry (.json) instead of "
        "a conforming mesh",
    )
    p.add_argument("--hausdorff", type=float, help="Hausdorff distance of the interface mesh")
    p.add_argument(
        "--mesh-method",
        choices=("auto", "exact", "grid"),
        default="auto",
        help="conforming mesh of the exact geometry, without a grid (cells, convex pieces or "
        "adaptive level set), or of the level set sampled on a background grid; auto "
        "(default): exact when the model has an exact geometry, the grid otherwise",
    )
    p.add_argument(
        "--min-feature",
        type=float,
        help="tessellations: features of the cells smaller than this length are removed "
        "before meshing (default size / 4; 0 keeps the exact cells)",
    )
    p.add_argument(
        "--cells",
        choices=("whole", "box"),
        default="whole",
        help="periodic tessellations: mesh the whole grains, periodized at the design "
        "(default), or the grains cut by the faces of the box",
    )
    p.add_argument(
        "--order",
        type=int,
        choices=(1, 2),
        default=2,
        help="elements of the conforming mesh: 2 (default), 10-node tetrahedra whose edges "
        "follow the interfaces; 1, 4-node tetrahedra (faster, stiffer)",
    )
    p.add_argument(
        "--mesh-phases",
        choices=("auto", "all", "solid"),
        default="auto",
        help="mesh both phases, or phase 1 only (auto: solid when a phase is a void)",
    )
    p.add_argument(
        "--integration",
        choices=("full", "bbar", "reduced"),
        help="of the elements: full (every point), or bbar (also: reduced), the volume "
        "change of each element as its mean (B-bar, F-bar), for nearly incompressible "
        "phases and plastic flow; default: full on a mesh, bbar for the loading paths of "
        "an image",
    )
    p.add_argument(
        "--loading",
        action="append",
        metavar="[NAME:] COMPONENTS",
        help="a macroscopic loading to solve, e.g. 'tension: E11=0.01' (strain E, stress S, "
        "gradient G, flux Q, change of temperature dT; free components: zero stress or "
        "flux); repeatable",
    )
    p.add_argument(
        "--path",
        action="append",
        metavar="[NAME:] ITEMS",
        help="a loading path followed in time with the nonlinear behaviours (finite "
        "elements, periodic or kubc; FFT on images, periodic), e.g. "
        "'tension: t=0,10 n=20 E11=0,0.02' (times t, "
        "increments n, strain E and stress S components and change of temperature dT at the "
        "times; free components: zero stress); repeatable",
    )
    p.add_argument("--paths", help="YAML/JSON file mapping names to loading paths")
    p.add_argument(
        "--finite",
        action="store_true",
        help="follow the loading paths in finite strains (strains E: components of the upper "
        "triangular gradient F - I, nominal strains and simple shears; stresses S: nominal "
        "stresses; the curves add the logarithmic strains LE and the Cauchy stresses TS)",
    )
    p.add_argument(
        "--contact",
        nargs="?",
        type=float,
        const=0.0,
        metavar="MU",
        help="the walls of the pores that close along the loading paths press on each other "
        "(and on their images across the faces of a periodic cell), with Coulomb's friction "
        "MU (default 0): finite elements on the mesh of the solid, in finite strains; the "
        "curves add the area in contact, the largest pressure, the deepest penetration and "
        "the energy that friction dissipated",
    )
    p.add_argument(
        "--interfaces",
        metavar="KEY:VALUE,...",
        help="the interfaces of the phases of a mesh debond along the loading paths: their "
        "traction-separation law (normal_strength:N,mode_I:G for the bilinear law, with "
        "shear_strength, mode_II, eta, stiffness; law:exponential,strength:N,energy:G; "
        "law:contact for lips without adhesion), Coulomb's friction of the broken "
        "interfaces (friction:MU), the faces that are interfaces (faces:phases, grains or "
        "pairs 0-1+0-2), the time of the viscous regularization (viscosity:T) and the "
        "elastic slip of the friction (slip:D); the "
        "curves add the debonded area, the mean damage and the largest opening of the "
        "interfaces and the energy that friction dissipated",
    )
    p.add_argument(
        "--curves", default=".", help="directory for the curves of the loading paths (.csv)"
    )
    p.add_argument("--save-mesh", help="write the conforming mesh to this file (.vtu, .inp...)")
    p.add_argument(
        "--material",
        nargs="+",
        help="PHASE=void or PHASE=young:E,poisson:nu[,conductivity:k] (or bulk:k,shear:mu; "
        "crystals: cubic:C11/C12/C44, hexagonal:C11/C12/C13/C33/C44; thermal expansion "
        "expansion:a, density:rho; the other transport coefficients electrical_conductivity:s, "
        "permittivity:e, permeability:m, diffusivity:d, each one value or k1/k2/k3)",
    )
    p.add_argument(
        "--materials",
        help="YAML/JSON file mapping phases to materials (with their plasticity or crystal "
        "plasticity, if any)",
    )
    p.add_argument(
        "--bc",
        nargs="+",
        default=["periodic"],
        help="boundary conditions: kubc, mixed, periodic, subc (aliases utg, mtbc, pbc, uhf)",
    )
    p.add_argument(
        "--property",
        nargs="+",
        default=["bulk_modulus", "shear_modulus"],
        choices=_apparent_properties(),
        metavar="PROPERTY",
        help="apparent properties: " + ", ".join(_apparent_properties()),
    )
    p.add_argument(
        "--frequencies",
        nargs="+",
        type=float,
        help="angular frequencies of the harmonic properties (dynamic_stiffness, "
        "complex_conductivity)",
    )
    p.add_argument(
        "--wavelengths",
        nargs="+",
        type=float,
        help="wavelengths of the effective refractive index (refractive_index)",
    )
    p.add_argument(
        "--fluid-bulk-modulus",
        type=float,
        help="bulk modulus of the fluid of the pores: the undrained response of the "
        "property poroelastic (Gassmann, Skempton)",
    )
    p.add_argument(
        "--solver",
        choices=("auto", "direct", "amg", "cg", "fenicsx"),
        default="auto",
        help="linear solver of the finite element problems (fenicsx: FEniCSx, Linux/WSL)",
    )
    p.add_argument("--tol", type=float, default=1e-8, help="relative residual of iterative solvers")
    p.add_argument("--periodic", action="store_true", help="treat the sample as periodic")
    p.add_argument("--fields", help="directory for the local fields (.vtu or .vti)")
    p.add_argument(
        "--device",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
        help="device of the FFT solver of images: the GPU needs CuPy (morphogen info)",
    )
    p.add_argument("-o", "--output", help="write the results to this JSON file")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.add_argument("--threads", type=int, help="number of threads")
    p.set_defaults(func=cmd_homogenize)

    def contact_test(name: str, help_text: str):
        """The parser of a test of contact, with the arguments of its sample."""
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "sample",
            help="geometry (.json, meshed with --size) or tetrahedral mesh (.vtu, .mesh) of the "
            "sample; with void phases, its solid alone",
        )
        p.add_argument("--size", type=float, help="element size of the conforming mesh (.json)")
        p.add_argument("--hausdorff", type=float, help="Hausdorff distance of the interface mesh")
        p.add_argument(
            "--order", type=int, choices=(1, 2), default=2,
            help="elements of the conforming mesh: 10-node tetrahedra (2, default) or 4-node "
            "ones (1)",
        )  # fmt: skip
        p.add_argument(
            "--mesh-phases", choices=("auto", "all", "solid"), default="auto",
            help="mesh both phases, or phase 1 only (auto: solid when a phase is a void)",
        )  # fmt: skip
        p.add_argument("--material", nargs="+", help="materials, as for morphogen homogenize")
        p.add_argument("--materials", help="YAML/JSON file of the materials of the phases")
        p.add_argument("--axis", type=int, choices=(0, 1, 2), default=2,
                       help="the axis of the test (default 2: z)")  # fmt: skip
        p.add_argument("--friction", type=float, default=0.0,
                       help="Coulomb's coefficient of friction (default 0)")  # fmt: skip
        p.add_argument("--steps", type=int, default=20, help="steps of the loading (default 20)")
        p.add_argument("--small", action="store_true",
                       help="small strains (default: finite strains)")  # fmt: skip
        p.add_argument("--curve", help="file of the curve (.csv; default: <sample>_<test>.csv)")
        p.add_argument("--fields", help="write the deformed sample and the contact pressure "
                       "at the end of the loading to this file (.vtu)")  # fmt: skip
        p.add_argument("-o", "--output", help="write the results to this JSON file")
        p.add_argument("--json", action="store_true", help="JSON output")
        p.set_defaults(voxels=None, periodic=False, cells="box", min_feature=None, array=None)
        return p

    p = contact_test(
        "compress",
        "compression of a sample between two rigid platens, its walls free to close on "
        "themselves (contact and friction): nominal stress against nominal strain",
    )
    p.add_argument("--strain", type=float, required=True,
                   help="nominal strain at the end, negative: the approach of the platens over "
                   "the height of the sample (-0.2: compressed by 20 %%)")  # fmt: skip
    p.add_argument("--no-self-contact", action="store_true",
                   help="the surface of the sample does not press on itself (its walls go "
                   "through each other)")  # fmt: skip
    p.add_argument("--points", choices=("gauss", "nodes"), default="gauss",
                   help="the contact points of the faces of the sample: inside them (gauss, the "
                   "default), or at their nodes (nodes: a sample whose corners and edges meet "
                   "the platens, or its own walls, first)")  # fmt: skip
    p.set_defaults(func=cmd_compress)

    p = contact_test(
        "indent",
        "indentation of the upper face of a sample by a rigid sphere or cone: force against "
        "depth, area of the contact, and the modulus that the force reads",
    )
    p.add_argument("--depth", type=float, required=True, help="depth of the indentation")
    p.add_argument("--indenter", choices=("sphere", "cone"), default="sphere")
    p.add_argument("--radius", type=float, help="radius of the sphere")
    p.add_argument("--half-angle", type=float, default=70.3,
                   help="half-angle of the cone, degrees from its axis (default 70.3: the cone "
                   "of the area of the pyramids of Berkovich and Vickers)")  # fmt: skip
    p.add_argument("--point", type=float, nargs=2, metavar=("U", "V"),
                   help="where on the face (its two coordinates; default: its middle)")  # fmt: skip
    p.add_argument("--unload", action="store_true",
                   help="then withdraw the indenter in as many steps (the slope of the "
                   "unloading: Oliver and Pharr)")  # fmt: skip
    p.add_argument("--base", choices=("fixed", "smooth"), default="fixed",
                   help="the lower face of the sample: glued (default) or sliding")  # fmt: skip
    p.add_argument("--poisson", type=float,
                   help="Poisson's ratio of the sample, to read the modulus of a sphere with "
                   "the rigid base under the sample (a layer of finite thickness)")  # fmt: skip
    p.set_defaults(func=cmd_indent, steps=10)

    p = sub.add_parser(
        "bands",
        help="band structure of the Bloch waves of a periodic cell: elastic waves, the "
        "scalar waves of sonic and planar photonic crystals, or light in a "
        "three-dimensional photonic crystal, and their band gaps",
    )
    p.add_argument(
        "sample",
        help="phase image of the cell (.vti, .vtk, .mhd, .npz, .npy, .tif: cut into "
        "tetrahedra; its voxels carry the plane waves of the electromagnetic waves) or "
        "tetrahedral mesh (.vtu, .mesh) with matching opposite faces",
    )
    p.add_argument("--array", help="array name in multi-array image files")
    p.add_argument(
        "--material",
        nargs="+",
        help="materials, as for morphogen homogenize (with density:rho for the elastic, "
        "acoustic and shear waves; permittivity:e and permeability:m for tm, te and "
        "electromagnetic)",
    )
    p.add_argument("--materials", help="YAML/JSON file of the materials of the phases")
    p.add_argument(
        "--waves",
        choices=("elastic", "acoustic", "shear", "tm", "te", "electromagnetic"),
        default="elastic",
        help="elastic waves of the solid; scalar waves: acoustic (the pressure of fluids), "
        "shear (antiplane), tm and te (light in the plane of a cell invariant along z); "
        "electromagnetic: light in a three-dimensional cell (plane waves on the voxels "
        "of an image)",
    )
    p.add_argument(
        "--path",
        nargs="+",
        metavar="CORNER",
        help="corners of the Brillouin zone: G, X, Y, Z, M, R (default: G X M G R; G X M G "
        "in the plane of an image one voxel thick)",
    )
    p.add_argument(
        "--coarsen",
        type=int,
        default=1,
        metavar="N",
        help="electromagnetic waves: plane waves on a grid N times coarser than the voxels, "
        "whose cells cut by the interfaces take an averaged permittivity tensor (voxelize "
        "2 to 4 times finer than the plane waves)",
    )
    p.add_argument("--bands", type=int, default=8, help="number of bands (the lowest)")
    p.add_argument("--points", type=int, default=8, help="wave vectors per segment of the path")
    p.add_argument("--order", type=int, choices=(1, 2), default=2, help="1: 4-node, 2: 10-node")
    p.add_argument("--phases", type=int, nargs="+", help="phases of an image kept in the cell")
    p.add_argument("-o", "--output", help="CSV file of the bands (a row per wave vector)")
    p.add_argument("--figure", help="figure of the bands with their gaps (.png, .svg, .pdf)")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(func=cmd_bands)

    p = sub.add_parser(
        "alloy",
        help="the scales of an alloy (its phases from a tool of alloy design) as microstructures",
        description="An alloy (JSON or YAML: composition, grain size, phases with their roles, "
        "fractions, radii and elastic constants, the properties predicted; see "
        "morphogen.interop.alloys): its scales (the precipitates in the matrix, the grains), "
        "their configurations for morphogen run, and their effective elasticity by FFT.",
    )
    p.add_argument("alloy", help="the alloy (.json, .yaml)")
    p.add_argument("--config", metavar="DIR", help="write the configuration of each scale here")
    p.add_argument("--homogenize", action="store_true", help="the effective elasticity by FFT")
    p.add_argument("--shape", type=int, default=32, help="voxels per edge (default 32)")
    p.add_argument("--seed", type=int, default=1, help="seed of the realizations (default 1)")
    p.add_argument("--json", help="write the results of --homogenize to this JSON file")
    p.set_defaults(func=cmd_alloy)

    p = sub.add_parser("analyze", help="morphological measurements on a voxel image")
    p.add_argument("image", help="image file (.vti, .vtk, .mhd, .npz, .npy, .tif)")
    p.add_argument("--array", help="array name in multi-array files")
    p.add_argument(
        "--measure",
        nargs="+",
        default=["volume_fraction", "specific_surface"],
        choices=sorted(analysis.MEASURES),
        help="measures to compute",
    )
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser(
        "texture",
        help="grains and texture of an orientation map (EBSD): pole figures, inverse pole "
        "figure, disorientations of the boundaries; texture of the lattices of the fields "
        "of a loading path",
    )
    p.add_argument(
        "map",
        help="orientation map: an .ang (EDAX/TSL) or a .ctf (Oxford/HKL) file; or the "
        "fields of a loading path with the orientations of the lattices (.vti, .vtu)",
    )
    p.add_argument(
        "--lattice",
        default="cubic",
        choices=("cubic", "hexagonal"),
        help="lattice of the crystals of the fields of a path (a map gives its own)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="disorientation (degrees) below which two neighbouring points are in one grain",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=5,
        help="points: smaller grains are given to their neighbours",
    )
    p.add_argument(
        "--axes",
        choices=("map", "file"),
        default="map",
        help="map: the orientations brought into the axes of the image of the map (x to "
        "the right, y upwards, z towards the viewer); file: as the file gives them",
    )
    p.add_argument(
        "--setting",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="axes of the orientations of an .ang file: their first axis down the image "
        "(1, the usual setting), up (2), to the right (3) or to the left (4)",
    )
    p.add_argument(
        "--poles",
        nargs="+",
        type=_indices,
        metavar="HKL",
        help="families of planes of the pole figures: 100 110 111 (cubic), 0001 10-10 "
        "10-11 (hexagonal) by default",
    )
    p.add_argument(
        "--direction",
        type=_sample_direction,
        default=(0.0, 0.0, 1.0),
        help="direction of the sample of the inverse pole figure and of the colours of "
        "the map: x, y, z (default) or a,b,c",
    )
    p.add_argument(
        "--halfwidth",
        type=float,
        default=10.0,
        help="half-width (degrees) of the kernel that spreads each pole",
    )
    p.add_argument(
        "-o",
        "--output",
        help="folder for the tables (grains.csv, boundaries.csv), the map in the colours "
        "of its orientations and, with matplotlib, the figures",
    )
    p.add_argument(
        "--format",
        default="png",
        choices=("png", "svg", "pdf"),
        help="format of the figures",
    )
    p.add_argument(
        "--convert",
        metavar="FILE",
        help="also write the map as this .ang or .ctf file (the other format, other axes)",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="with --convert: write the map of the grains (each point with the mean "
        "orientation of its grain, the points that were not indexed filled)",
    )
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(func=cmd_texture)

    p = sub.add_parser(
        "percolation",
        help="percolation threshold of the model of a configuration against a parameter",
    )
    p.add_argument("config", help="configuration file: its model, domain and voxels sections")
    p.add_argument(
        "--parameter",
        default="volume_fraction",
        help="parameter of the model varied (default volume_fraction)",
    )
    p.add_argument("--lower", type=float, default=0.05, help="lower bound (default 0.05)")
    p.add_argument("--upper", type=float, default=0.95, help="upper bound (default 0.95)")
    p.add_argument("--phase", type=int, default=1, help="phase that percolates (default 1)")
    p.add_argument(
        "--axis",
        choices=("x", "y", "z", "any", "all"),
        default="x",
        help="joined along this axis, any of them or all three (default x)",
    )
    p.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 6, 8, 18, 26),
        help="neighbours: 6 through faces, 18 and edges, 26 and corners; 4 or 8 in 2-D (one "
        "voxel along z; default 6, which is 4 there; 26 with --exponent thermal, the voxels "
        "of the FFT solvers joining through their nodes)",
    )
    p.add_argument(
        "--periodic",
        action=argparse.BooleanOptionalAction,
        help="wrapping around the box, or (--no-periodic) joining two faces (default: as "
        "the domain; periodic with --exponent, as the FFT solvers)",
    )
    p.add_argument(
        "--directions",
        type=_directions,
        help="anisotropy: the thresholds across the cores of the box along these directions "
        "(axes; lattice: the axes and the diagonals, 4 in 2-D and 13 in 3-D; or a number "
        "sampled over the half circle or sphere), their ratio and threshold tensor",
    )
    p.add_argument(
        "--method",
        choices=("labels", "fire"),
        default="labels",
        help="components labelled, or a forest fire lit on a face reaching the other "
        "(open along the axis, periodic or not across it; default labels)",
    )
    p.add_argument("--realizations", type=int, default=9, help="images at each fraction")
    p.add_argument("--tol", type=float, default=0.005, help="width of the final bracket")
    p.add_argument("--seed", type=int, help="seed (default: the configuration's)")
    p.add_argument(
        "--sizes",
        type=_sizes,
        help="finite-size scaling: the thresholds of cubes of these sizes (voxels per edge, "
        "separated by commas, at least 3) at the voxel size of the configuration, "
        "extrapolated to the infinite medium as p_c + a L^(-1/nu)",
    )
    p.add_argument(
        "--nu",
        type=float,
        help="exponent of the correlation length (default that of random percolation: "
        "0.8765 in 3-D, 4/3 in 2-D)",
    )
    p.add_argument("--fit-nu", action="store_true", help="fit nu too (at least 4 sizes)")
    p.add_argument(
        "--rigidity",
        action="store_true",
        help="the threshold of the elastic moduli over the --sizes, where the phase becomes "
        "rigid (its voxels joined at corners and edges turn about them, so that no "
        "connectivity gives it): where half the realizations bear a uniaxial strain along the "
        "axis with a modulus above --cut, the other phases void, by the --solver of the voxels "
        "(on --device); that of --exponent elastic",
    )
    p.add_argument(
        "--cut",
        type=float,
        default=None,
        help="rigidity: the modulus above which a realization bears, over the Young's modulus "
        "of the phase (default 1e-8 with --solver fft, 1e-12 with fe, whose energies resolve "
        "the moduli down to the round-off)",
    )
    p.add_argument(
        "--exponent",
        choices=("thermal", "elastic"),
        help="critical exponent of the conductivity (t) or of the elastic moduli (f) of the "
        "phase, by the --solver of the voxels: the finite-size scaling of their means at the "
        "threshold over the --sizes, L^(-t/nu)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        help="threshold of the exponent (default: extrapolated first over the --sizes, for t "
        "with the criterion, through 26 neighbours and wrapping around the box by default, "
        "for f by the moduli as with --rigidity)",
    )
    p.add_argument(
        "--threshold-stderr",
        type=float,
        help="standard error of the threshold, whose effect on the exponent is computed "
        "(default: that of the extrapolation; 0 with --threshold)",
    )
    p.add_argument(
        "--fractions",
        type=_fractions,
        help="exponent: a sweep too, over these fractions above the threshold (at least 3, "
        "separated by commas)",
    )
    p.add_argument(
        "--contrast",
        type=float,
        default=1e-8,
        help="exponent: the properties of the other phases over those of the phase (default "
        "1e-8; 0: void)",
    )
    p.add_argument(
        "--device",
        choices=("cpu", "gpu", "auto"),
        default="auto",
        help="exponent and rigidity: the device of the FFT solver (default auto: the GPU when "
        "there is one)",
    )
    p.add_argument(
        "--solver",
        choices=("fft", "fe"),
        default="fft",
        help="exponent and rigidity: the solver of the trilinear voxels, fft (default, on "
        "--device) or fe, their matrix assembled on the voxels of the materials and solved by "
        "conjugate gradients preconditioned by its sparse factorization, on the CPU: the same "
        "moduli in a few iterations where the FFT solver takes thousands, near the rigidity "
        "threshold with void around (--contrast 0)",
    )
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(func=cmd_percolation)

    p = sub.add_parser(
        "frame-stiffness",
        help="stiffness of the frame of beams of a fibre network, open foam or random lattice "
        "(a realization of delaunay_network, open_foam or random_lattice)",
    )
    p.add_argument(
        "file",
        help="configuration of a network model (a realization of it), or the geometry (.json) "
        "of a realization",
    )
    p.add_argument("--seed", type=int, help="seed (default: the configuration's)")
    p.add_argument(
        "--realization",
        type=int,
        default=0,
        help="the realization of the configuration, seeded as in its batch run (default 0)",
    )
    p.add_argument(
        "--beams",
        choices=BEAM_THEORIES,
        default="euler-bernoulli",
        help="Euler-Bernoulli beams (default), or Timoshenko beams (shear stiffness kappa G A)",
    )
    p.add_argument(
        "--shear-coefficient",
        type=float,
        help="kappa of Timoshenko beams (default Cowper's, 6(1 + nu)/(7 + 6 nu) for circles)",
    )
    p.add_argument(
        "--rigid-joints",
        type=float,
        metavar="C",
        help="beams rigid over C of their radii from each node (default none; a third "
        "matches the continuum for Timoshenko beams)",
    )
    p.add_argument("--young", type=float, default=1.0, help="Young's modulus of the fibres")
    p.add_argument("--poisson", type=float, default=0.3, help="Poisson's ratio of the fibres")
    p.add_argument(
        "--softening",
        nargs="+",
        choices=NETWORK_MODES,
        help="Ban et al.'s coefficients s and v of the softening of non-identical beams, the "
        "stiffness of a mode varying from beam to beam: axial (EA), bending (EI), modulus (E) "
        "or area (EA and kappa G A)",
    )
    p.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="x",
        help="axis of the uniaxial stress of the softening coefficients (default x)",
    )
    p.add_argument(
        "--replicas",
        type=int,
        default=0,
        metavar="N",
        help="N replicas per CV of the stiffness of the beams of each mode of --softening, on "
        "the same geometry, against s and v (default 0: the coefficients alone; one solve each)",
    )
    p.add_argument(
        "--cv",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3],
        help="coefficients of variation of the replicas (default 0.1 0.2 0.3)",
    )
    p.add_argument(
        "--distribution",
        choices=NETWORK_LAWS,
        default="normal",
        help="law of the stiffness of the replicas: normal (draws below 1e-3 raised to it, the "
        "default), lognormal or bimodal (1 - CV or 1 + CV)",
    )
    p.add_argument("--replica-seed", type=int, default=1, help="seed of the replicas (default 1)")
    p.add_argument("--table", help="write the table of the replicas (.csv)")
    p.add_argument("--figure", help="figure of the replicas (.png, .svg, .pdf)")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(func=cmd_frame_stiffness)

    p = sub.add_parser(
        "fit-rule",
        help="find a cellular-automaton rule whose structures match the morphology of an "
        "image (volume fraction, interface density, correlation function)",
    )
    p.add_argument("image", help="phase image (phase 1: state 1); one voxel per cell")
    p.add_argument("--array", help="array name in multi-array image files")
    p.add_argument(
        "--neighbourhood",
        default="moore",
        choices=("von_neumann", "moore", "edges", "euclidean"),
        help="neighbourhood of the automaton",
    )
    p.add_argument("--steps", type=int, default=15, help="updates from the initial noise")
    p.add_argument(
        "--initial-fraction", type=float, help="density of the initial noise (default: searched)"
    )
    p.add_argument("--lattice", type=int, help="cells per edge of the trial lattices")
    p.add_argument(
        "--tiling",
        default="square",
        choices=LATTICES,
        help="cells of the automaton: square (the grid), triangular, hexagonal, penrose, "
        "ammann_beenker (prisms in 3-D), bcc, fcc or icosahedral (3-D); quasi-periodic "
        "tilings run on free patches and their configurations on non-periodic domains",
    )
    p.add_argument("--population", type=int, default=24, help="rules per generation")
    p.add_argument("--generations", type=int, default=15, help="generations of the search")
    p.add_argument("--seed", type=int, default=0, help="seed of the search")
    p.add_argument("-o", "--output", help="write a configuration of the fitted automaton")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(func=cmd_fit_rule)

    p = sub.add_parser(
        "optimize",
        help="topology optimization of a periodic cell (inverse homogenization, FFT solver)",
    )
    p.add_argument("config", help="configuration file with an 'optimization' section")
    p.add_argument("-o", "--output", help="output directory (overrides outputs.directory)")
    p.add_argument("--seed", type=int, help="seed (random initial designs)")
    p.add_argument("--threads", type=int, help="number of threads")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser(
        "laser",
        help="architectured sheets by localized laser treatment: treatment, tensile test and "
        "optimization of the laser path",
    )
    p.add_argument("config", help="configuration file with a 'laser_sheet' section")
    p.add_argument("-o", "--output", help="output directory (overrides outputs.directory)")
    p.add_argument("--seed", type=int, help="seed of the optimization")
    p.add_argument("--evaluations", type=int, help="evaluations of the optimization")
    p.add_argument("--no-optimization", action="store_true", help="evaluate the designs only")
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    p.set_defaults(func=cmd_laser)

    p = sub.add_parser(
        "learn",
        help="learn a behaviour from data: random paths, a network trained, evaluated, saved",
    )
    p.add_argument("config", help="configuration file (a 'learning' section, or its keys)")
    p.add_argument("-o", "--output", help="file of the model (.npz; overrides output)")
    p.add_argument("--data", help="also save the data (.npz; overrides save_data)")
    p.add_argument("--model", choices=["feedforward", "potential", "recurrent", "standard"],
                   help="kind of model (overrides model.kind)")  # fmt: skip
    p.add_argument("--epochs", type=int, help="epochs of the training")
    p.add_argument("--paths", type=int, help="random paths of the data")
    p.add_argument("--seed", type=int, help="seed of the paths, of the model and of the training")
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser(
        "view",
        help="figures of a microstructure (sections, tiling, 3D surface, history) or "
        "animation of its formation",
    )
    p.add_argument(
        "file",
        help="geometry (.json), configuration (its first realization) or voxel image",
    )
    p.add_argument("--shape", type=int, help="voxels per edge to voxelize a geometry")
    p.add_argument("--labels", action="store_true", help="show the cell (grain) labels")
    p.add_argument(
        "--tile",
        type=int,
        help="tile the middle sections of a periodic cell (those normal to x, y and z; the "
        "section of a planar one)",
    )
    p.add_argument("--surface", action="store_true", help="3D view of the phase-1 boundary")
    p.add_argument("--history", action="store_true", help="history of an automaton")
    p.add_argument("--kinetics", action="store_true", help="transformation kinetics (JMAK)")
    p.add_argument("--animate", help="write an animated GIF of the formation (configuration)")
    p.add_argument("--frames", type=int, default=24, help="frames of the animation")
    p.add_argument("--fps", type=int, default=8, help="frames per second of the animation")
    p.add_argument("-o", "--output", help="figure file (default: <file>_view.png)")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser(
        "export",
        help="convert an image, a mesh, a geometry or a table to another format, a figure or "
        "a video (mp4, gif)",
        description="The output format follows the suffix: data (.vti, .vtu, .msh, .inp, .xdmf, "
        ".stl...), tables (.csv, .txt, .xlsx), figures (.png, .tif, .svg, .pdf, .tex: TikZ), "
        "videos (.mp4, .gif; or --format frames and a folder). With --particles PHASE, the "
        "particles of that phase without its matrix, for the discrete element method: a "
        "closed surface per particle (.stl, .obj, .vtu; .msh, Gmsh 2.2 as LMGC90 reads it, "
        "its tetrahedra with --tetrahedra), whole across periodic faces; a table (.csv, "
        ".txt, .xlsx); the spheres of a union of spheres and segments as a LAMMPS / "
        "LIGGGHTS data file (.data) or YADE's columns (--format yade).",
    )
    p.add_argument(
        "input",
        help="image, mesh (.vtu with its arrays, .mesh), geometry (.json) "
        "or table (.csv, .txt, .xlsx)",
    )
    p.add_argument("output", help="output file (its suffix chooses the format) or folder")
    p.add_argument(
        "--format",
        choices=("frames", "yade"),
        help="PNG frames in the output folder; yade: the spheres of --particles as YADE reads them",
    )
    p.add_argument(
        "--particles",
        type=int,
        metavar="PHASE",
        help="the particles of this phase without its matrix (DEM): surfaces, tables, spheres",
    )
    p.add_argument(
        "--tetrahedra",
        action="store_true",
        help="--particles in a .msh file: their tetrahedra (gmsh), not their triangles",
    )
    p.add_argument(
        "--element-size",
        type=float,
        metavar="H",
        help="--particles: their surfaces remeshed by MMG (mmgs) at triangles of this size",
    )
    p.add_argument("--array", help="array of a multi-array image")
    p.add_argument(
        "--kind",
        choices=("sections", "orthoslices", "turntable", "sweep", "field"),
        help="video: sections, orthoslices or turntable (images); turntable, sweep or field "
        "(meshes)",
    )
    p.add_argument("--axis", type=int, choices=(0, 1, 2), help="axis of sections and sweeps")
    p.add_argument("--phase", type=int, help="phase of the surface of an image (turntable)")
    p.add_argument("--field", help="array colouring a mesh ('phase': the phases)")
    p.add_argument("--displacement", help="point array displacing a mesh (kind field)")
    p.add_argument("--magnification", type=float, help="of the displacement (default: auto)")
    p.add_argument("--wave", choices=("ramp", "sine"), help="load growing or oscillating")
    p.add_argument(
        "--section",
        nargs=2,
        type=float,
        metavar=("AXIS", "FRACTION"),
        help="cut the mesh by a plane normal to AXIS at FRACTION of the box",
    )
    p.add_argument("--frames", type=int, help="number of frames")
    p.add_argument("--fps", type=float, help="frames per second")
    p.add_argument("--mode", choices=("once", "loop", "bounce"), help="once, loop, back and forth")
    p.add_argument("--repeat", type=int, help="loops of the video")
    p.add_argument("--cmap", help="matplotlib colormap (viridis, inferno, coolwarm...)")
    p.add_argument("--clim", nargs=2, type=float, metavar=("LOW", "HIGH"), help="colour range")
    p.add_argument(
        "--size", nargs="+", type=int, help="pixels: W [H] of 3D views, side of sections"
    )
    p.add_argument("--shape", type=int, help="voxels per edge to voxelize a geometry")
    p.add_argument("--voxel-size", type=float, help="voxel size to voxelize a geometry")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser(
        "zmat",
        help="read a material file of Z-mat (Z-set): its translation to morphoGen, or the "
        "coverage of Z-mat's library",
    )
    p.add_argument("file", nargs="?", help="material file of Z-mat (***behavior gen_evp ...)")
    p.add_argument("--behaviour", type=int, default=0, help="index of the behaviour in the file")
    p.add_argument("-o", "--output", help="folder for the MFront source (when one is needed)")
    p.add_argument("--coverage", action="store_true", help="the coverage of Z-mat (Markdown)")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(func=cmd_zmat)

    p = sub.add_parser("gui", help="graphical interface in the web browser (local server)")
    p.add_argument("--port", type=int, default=8765, help="port of the local server")
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser(
        "desktop",
        help="the graphical interface as a stand-alone application, in a window of its own "
        "(pip install 'morphogen[desktop]')",
    )
    p.add_argument(
        "--session", help="folder of the files of the session (default: a temporary one)"
    )
    p.add_argument("--debug", action="store_true", help="the developer tools of the page")
    p.add_argument(
        "--browser",
        action="store_true",
        help="in the web browser rather than a window (as morphogen gui)",
    )
    p.set_defaults(func=cmd_desktop)

    p = sub.add_parser(
        "radioss",
        help="explicit dynamics with OpenRadioss: its installation, the inputs of a morphology "
        "(crushing, tension, shear, impact), their runs and their results",
    )
    p.add_argument(
        "action",
        choices=("info", "install", "export", "run", "read"),
        help="info: the installation found; install: a release from GitHub; export: the "
        "inputs of a geometry (.json) or an image; run: run inputs written by export; read: "
        "the results of a run (made here or elsewhere)",
    )
    p.add_argument("source", nargs="?", help="export: geometry or image; run, read: folder")
    p.add_argument("-o", "--output", help="export: folder of the inputs; install: folder")
    p.add_argument("--path", help="folder of OpenRadioss (default: found)")
    p.add_argument("--tag", help="install: the tag of a release (default: the latest)")
    p.add_argument(
        "--materials",
        help="export: YAML/JSON file of the materials of the phases "
        "(young, poisson, density, plasticity, johnson_cook, failure_strain; 'void')",
    )
    p.add_argument("--density", type=float, help="export: density of the materials that give none")
    p.add_argument(
        "--units",
        default="Mg mm s",
        help="export: units of mass, length and time of "
        "every value (default 'Mg mm s': MPa, Mg/mm³, mm/s)",
    )
    p.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help="export: the length unit of the units per unit of the geometry",
    )
    p.add_argument("--mesh", choices=("voxels", "conforming"), default="voxels")
    p.add_argument("--voxels", type=int, help="export: voxels per edge of a geometry (24)")
    p.add_argument(
        "--element-size", type=float, help="export: conforming meshes, the largest element"
    )
    p.add_argument("--order", type=int, choices=(1, 2), default=1, help="conforming: tetrahedra")
    p.add_argument("--array", help="export: the array of an image (default: its first)")
    p.add_argument(
        "--loading", choices=("compression", "tension", "shear", "impact"), default="compression"
    )
    p.add_argument("--axis", default="z", help="normal to the loaded faces (x, y, z)")
    p.add_argument("--shear-axis", help="shear: the direction of the drive")
    p.add_argument("--strain", type=float, default=0.2, help="final nominal strain (0.2)")
    p.add_argument("--rate", type=float, help="nominal strain rate, in 1/time (default 100)")
    p.add_argument("--velocity", type=float, help="impact: speed, in length/time")
    p.add_argument("--duration", type=float, help="duration (default: from the strain)")
    p.add_argument(
        "--lateral",
        choices=("fixed", "free"),
        default="fixed",
        help="the loaded faces held across the axis, or free to slide",
    )
    p.add_argument("--friction", type=float, default=0.0, help="of the contacts")
    p.add_argument("--no-contact", action="store_true", help="no self contact")
    p.add_argument("--frames", type=int, default=50, help="export: animations over the run")
    p.add_argument("--time-step", type=float, help="export: minimum time step (mass scaling)")
    p.add_argument("--name", help="export: name of the run")
    p.add_argument("--run", action="store_true", help="export: run the inputs at once")
    p.add_argument("--threads", type=int, help="run: OpenMP threads (default: all the cores)")
    p.add_argument("--mpi", type=int, help="run: MPI processes")
    p.add_argument(
        "--no-convert",
        dest="convert",
        action="store_false",
        help="read: do not convert the outputs with OpenRadioss's tools",
    )
    p.add_argument(
        "--frames-vtu",
        action="store_true",
        help="run, read: the animations as .vtu files in <folder>/frames",
    )
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(func=cmd_radioss)

    p = sub.add_parser("models", help="list the models or describe one")
    p.add_argument("name", nargs="?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("template", help="print an example configuration")
    p.add_argument("name", nargs="?", help="template name (omit to list them)")
    p.add_argument("-o", "--output", help="write to this file")
    p.set_defaults(func=cmd_template)

    p = sub.add_parser("info", help="versions and settings")
    p.set_defaults(func=cmd_info)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    if getattr(args, "threads", None):
        settings.set_num_threads(args.threads)
    full_speed()  # Windows: the long runs kept off the efficiency cores (EcoQoS)
    np.seterr(all="ignore")
    try:
        return int(args.func(args) or 0)
    except (ConfigError, ValueError, FileNotFoundError, RuntimeError, ImportError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
