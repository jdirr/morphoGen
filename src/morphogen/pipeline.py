"""Execution of configurations: batches of realizations and RVE studies.

Outputs of a run (in ``outputs.directory``):

* ``<name>_<index>.<ext>`` files of each realization (geometry, images, voxel meshes...;
  conforming tetrahedral meshes are ``<name>_<index>_tet.<ext>``, and local fields
  ``<name>_<index>_<property>_<bc>.vtu`` or ``.vti``);
* ``results.csv`` (``results_task<i>of<n>.csv`` for split runs): one row per realization
  with its seed, measurements and apparent properties (columns ``<property>.<bc>``; of a
  stiffness, the upper triangles of the stiffness ``stiffness.<bc>.C11`` ... ``C66`` and
  of its compliance ``stiffness.<bc>.S11`` ... ``S66``, in the notation that ``run.json``
  states, beside its engineering constants), the solid fraction, and the relative,
  normalized (per unit of solid fraction) and specific values of the moduli and
  conductivities (:mod:`morphogen.homogenization.postprocess`);
* ``homogenization.csv`` (``homogenization_task<i>of<n>.csv``): one row per realization,
  apparent property (or named loading or path) and boundary condition, with the
  macroscopic response, the energy estimate, the isotropy criterion and the solver data;
* ``<name>_<index>_<path>_<bc>.csv``: the macroscopic curves of each loading path (time,
  strain, stress, temperature, mean cumulated plastic strain, and the mean stress and
  plastic strain of each phase), and with ``fields`` the element fields at the end of
  each of its segments (``..._<k>.vtu``);
* with ``outputs.tables``, these tables also as ``.txt`` (tab-separated) or ``.xlsx``;
* with a ``visualization`` section, the figures and videos of the first realizations
  (``<name>_<index>_<view>.<format>``, see :mod:`morphogen.viz.workflow`): sections and
  orthoslices of the image, views of the mesh, fields on the deformed mesh, curves;
* with ``outputs.code_aster``, the study of Code_Aster of the conforming mesh of each
  realization (``<name>_<index>_aster/``: its mesh, command and export files,
  :mod:`morphogen.io.code_aster`); with ``outputs.openfoam``, the case of OpenFOAM of its
  image or of its conforming mesh (``<name>_<index>_openfoam/``, :mod:`morphogen.io.openfoam`);
* ``run.json``: configuration, theoretical values, versions and timing (and the notation
  of the stiffness and compliance, ``stiffness_notation``, when they are computed);
* with ``outputs.database``, the run is added to that database of the configurations
  tested (:mod:`morphogen.results`);
* for RVE studies, ``rve_report.json`` / ``.md`` / ``.tex`` and figures (when matplotlib
  is installed): statistics, scaling-law fits and RVE sizes (see :mod:`morphogen.rve`);
  with the stiffness, per boundary condition and size the mean stiffness ``Cbar``, its
  compliance ``S = Cbar^-1`` and the standard deviation of each component, with the RVE
  size of each component (``rve_stiffness.csv``, ``rve_stiffness_<bc>.png``; the
  conductivity tensor likewise, ``rve_conductivity_tensor.csv``);
* for window studies, ``results.csv`` has one row per window size (``window_edge``,
  ``volume``, the number ``n`` of windows): the statistics of the volume fraction
  (``mean``, ``variance``, ``relative_error``) and, of each tensor of the windows
  (:mod:`morphogen.rve.windows`: the homogenization computing ``stiffness`` or
  ``conductivity_tensor`` by FFT), the components of their mean ``<tensor>.periodic.C11``
  ..., of its inverse ``S11`` ... and their standard deviations ``C11_std`` ...; ``run.json``
  holds the apparent tensors of the whole realization (``apparent``). With ``rve.shifts``
  (> 1), the tilings shifted around the periodic realization (:mod:`morphogen.rve.scaling`)
  add the column ``tilings`` after ``n``, then the windows of one tiling: the means over all
  the windows, the variances pooled within the tilings, from the same solutions.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np

from morphogen import io, parallel, rve, settings
from morphogen._utils import to_jsonable
from morphogen._version import __version__
from morphogen.analysis.basic import phase_fractions
from morphogen.analysis.measures import measure
from morphogen.config import (
    HomogenizationSpec,
    RunConfig,
    RVESpec,
    path_quantities,
    property_entries,
)
from morphogen.domain import Domain
from morphogen.homogenization import Apparent, PathResult, Response, bounds, homogenize
from morphogen.homogenization.apparent import COUPLED, FLOW, HARMONIC
from morphogen.homogenization.beremin import path_beremin
from morphogen.homogenization.coupled import coupled_entries
from morphogen.homogenization.fatigue import path_fatigue
from morphogen.homogenization.materials import any_oriented
from morphogen.homogenization.postprocess import NOTATION, matrix_entries, normalized_values
from morphogen.homogenization.xfem import LevelSetSample
from morphogen.io.tables import write_table
from morphogen.mesh.conforming import conforming_mesh
from morphogen.mesh.surface import surface_mesh
from morphogen.mesh.tetra import TetMesh
from morphogen.mesh.triangles import TriMesh, plane_mesh
from morphogen.mesh.voxel import voxel_mesh
from morphogen.models.base import Microstructure, Model
from morphogen.models.implicit import implicit_functions
from morphogen.rve import report as rve_report
from morphogen.seeding import derive_seed
from morphogen.stopping import checkpoint
from morphogen.viz.workflow import element_arrays

log = logging.getLogger("morphogen")


def versions() -> dict:
    """The versions of morphoGen, Python, NumPy, SciPy, Numba (and pymmg) and the platform, recorded
    with the results."""
    import numba
    import scipy

    out = {
        "morphogen": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
    }
    try:
        from importlib.metadata import version

        out["pymmg"] = version("pymmg")
    except Exception:
        pass
    return out


def write_rows(path: Path, rows: list[dict]) -> Path:
    """Rows (mappings) written as a CSV file, the union of their keys as columns in their order of
    appearance."""
    keys: list[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    return path


def read_rows(path) -> list[dict]:
    """The rows of a CSV file as mappings, its numbers as floats."""

    def parse(v):
        try:
            return float(v)
        except ValueError:
            return v

    with open(Path(path), encoding="utf-8", newline="") as f:
        return [{k: parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _task_name(stem: str, task_id: int, n_tasks: int) -> str:
    return f"{stem}.csv" if n_tasks == 1 else f"{stem}_task{task_id}of{n_tasks}.csv"


def _results_name(task_id: int, n_tasks: int) -> str:
    return _task_name("results", task_id, n_tasks)


_COMPONENTS = ("11", "22", "33", "23", "31", "12")


def _field_arrays(result: Apparent) -> dict[str, np.ndarray]:
    """Derived local fields of a solution (element values, last axis: components)."""
    return element_arrays(result.solution.fields, result.solution.physics)


def _point_orientations(sample, grains) -> np.ndarray | None:
    """The Bunge angles of the crystal of each point of the fields of a sample (the
    voxels of an image with its ``grains``: labels and orientations; the elements of a
    mesh of grains), or None without crystals."""
    if grains is not None:
        labels = np.asarray(getattr(grains[0], "data", grains[0]))
        return np.asarray(grains[1], dtype=float)[labels]
    cell = getattr(sample, "cell", None)
    orientations = getattr(sample, "orientations", None)
    if cell is not None and orientations is not None:
        return np.asarray(orientations, dtype=float)[np.asarray(cell)]
    return None


def write_path_fields(target: Path, support, arrays: dict) -> Path:
    """Fields of a loading path at one time (:attr:`PathResult.fields`) on their support:
    ``.vtu`` of a mesh, the displacement of the nodes as point data, the other fields as
    cell data; ``.vti`` of an image, one array per component (``stress_11`` ...,
    ``displacement_1`` ..., the displacement at the centres of the voxels). The suffix of
    ``target`` is replaced by that of the format; returns the path of the file written."""
    values = {n: np.asarray(v, dtype=np.float32) for n, v in arrays.items()}
    if isinstance(support, TetMesh):
        cell = {n: v for n, v in values.items() if n != "displacement"}
        nodal = {n: v for n, v in values.items() if n == "displacement"}
        return io.write_vtu_tet(Path(target).with_suffix(".vtu"), support, cell, nodal)
    images = []
    for name, v in values.items():
        if v.ndim == 3:
            images.append(support.with_data(v, name=name))
            continue
        labels = _COMPONENTS if v.shape[-1] == 6 else [str(c + 1) for c in range(v.shape[-1])]
        for c, label in enumerate(labels):
            component = np.ascontiguousarray(v[..., c])
            images.append(support.with_data(component, name=f"{name}_{label}"))
    return io.write_vti(Path(target).with_suffix(".vti"), images)


def read_path_texture(path) -> tuple[np.ndarray, np.ndarray]:
    """The lattices of the crystals in a file of the fields of a loading path
    (:func:`write_path_fields`: a ``.vti`` of the voxels of an image, a ``.vtu`` of the
    elements of a mesh): their orientations (Bunge angles in degrees, m x 3) and their
    volumes (m; those of the elements, or one per voxel). The fields have these
    orientations for crystals in finite strains, or whose twins turn their lattices."""
    path = Path(path)
    if path.suffix.lower() == ".vti":
        from morphogen.io.vtk import vti_names

        names = [f"orientation_{c}" for c in "123"]
        missing = [n for n in names if n not in vti_names(path)]
        if missing:
            raise ValueError(f"{path}: no orientation of the lattices among its fields")
        arrays = {image.name: image.data for image in io.read_vti(path, names=names)}
        euler = np.stack([np.asarray(arrays[n], dtype=float).ravel() for n in names], axis=1)
        weights = np.ones(len(euler))
    elif path.suffix.lower() == ".vtu":
        from morphogen.mesh.tetra import element_volumes

        grid = io.read_vtu(path)
        if "orientation" not in grid["cell_data"]:
            raise ValueError(f"{path}: no orientation of the lattices among its fields")
        euler = np.asarray(grid["cell_data"]["orientation"], dtype=float).reshape(-1, 3)
        weights = np.abs(element_volumes(grid["points"], grid["cells"][:, :4]))
    else:
        raise ValueError(f"{path}: the fields of a path are a .vti or a .vtu file")
    crystals = ~np.isnan(euler).any(axis=1)
    if not crystals.any():
        raise ValueError(f"{path}: no crystal has an orientation")
    return euler[crystals], weights[crystals]


def _write_fields(result: Apparent, path_stem: Path) -> Path:
    """Local fields of a solution: ``.vtu`` on its mesh, ``.vti`` on its image."""
    support = result.solution.support
    arrays = _field_arrays(result)
    if isinstance(support, (TetMesh, TriMesh)):
        fields = result.solution.fields
        nodal = {}
        if "displacement" in fields:
            u = np.asarray(fields["displacement"], dtype=float)
            if u.shape[1] == 2:  # plane strain: the vectors of ParaView have three components
                u = np.column_stack([u, np.zeros(len(u))])
            nodal["displacement"] = u
        if "temperature" in fields:
            nodal["temperature"] = fields["temperature"]
        cell = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
        write = io.write_vtu_tri if isinstance(support, TriMesh) else io.write_vtu_tet
        return write(path_stem.with_suffix(".vtu"), support, cell, nodal)
    images = [support.with_data(np.asarray(v, dtype=np.float32), name=k) for k, v in arrays.items()]
    return io.write_vti(path_stem.with_suffix(".vti"), images)


def _write_flow(pores: TriMesh, velocity: np.ndarray, path_stem: Path) -> Path:
    """The velocities of the three flows of the plane (under the unit forces along x, y and
    z) at the nodes of the triangles of the pores: ``.vtu``."""
    nodal = {f"velocity_{axis}": velocity[k] for k, axis in enumerate("xyz")}
    return io.write_vtu_tri(path_stem.with_suffix(".vtu"), pores, {}, nodal)


def _mesh_fractions(mesh: TetMesh | TriMesh, materials) -> dict[int, float]:
    """Volume fractions of the phases of a mesh; a phase left out of the mesh (the pores
    of porous media) takes the rest of the box."""
    fractions = mesh.phase_fractions()
    absent = [p for p in materials if p not in fractions]
    if len(absent) == 1:
        fractions[absent[0]] = max(0.0, 1.0 - sum(fractions.values()))
    return dict(sorted(fractions.items()))


def _homogenize(
    sample,
    spec: HomogenizationSpec,
    stem: str,
    outdir: Path,
    grains=None,
    quantities=(),
    *,
    keep=False,
    tables=(),
    entries=(),
):
    """Apparent properties of one sample (mesh or image; ``grains``: labels and
    orientations of the grains of an image): wide values, long rows and the results.
    ``quantities`` (:class:`~morphogen.homogenization.paths.PathQuantity`) of the loading
    paths add long rows (``property``, ``bc``, ``value``) for the statistics of RVE
    studies, as do the ``entries`` ``(property, entry)`` of the strong couplings and of the
    harmonic properties (rows ``<property>.<entry>``). ``keep``: the fields kept for the
    figures and videos (true, or ``"steps"`` for the loading paths); ``tables``: formats of
    the curves of the paths besides .csv."""
    extra = {} if grains is None else {"grains": grains[0], "orientations": grains[1]}
    if isinstance(sample, TriMesh):  # the elastic problems of the plane
        extra["plane"] = spec.plane
    # the fatigue of a cycle and the criterion of Beremin read the stresses of each step
    judged = spec.fatigue or spec.beremin
    fields = "steps" if keep == "steps" or judged else bool(spec.fields or keep)
    results = homogenize(sample, spec.materials, **spec.options(), fields=fields, **extra)
    values: dict = {}
    rows: list[dict] = []
    for key, res in results.items():
        if isinstance(res, PathResult):
            if spec.fatigue and res.name == spec.fatigue["path"] and res.fields:
                cycle = path_fatigue(res, spec.fatigue["criteria"], cycle=spec.fatigue["cycle"])
                res.scalars.update({f"fatigue_{n}": v for n, v in cycle.components().items()})
                # with the fields of the end of the path, for its figures
                res.fields[max(res.fields)].update(cycle.fields())
                target = outdir / f"{stem}_{key.replace('.', '_')}_fatigue.vtu"
                written = write_path_fields(target, res.support, cycle.fields())
                log.info(
                    "fatigue of %s: largest utilization %.4g, factor of safety %.4g, %.3g %% "
                    "of the points beyond their limit; fields: %s",
                    key, cycle.largest, cycle.safety, 100 * cycle.fraction(), written.name,
                )  # fmt: skip
            if spec.beremin and res.name == spec.beremin["path"] and res.fields:
                found = path_beremin(res, spec.beremin["criterion"],
                                     orientations=_point_orientations(sample, grains))  # fmt: skip
                res.series.update(found.series(res.times) or {})
                res.fields[max(res.fields)].update(found.fields())
                target = outdir / f"{stem}_{key.replace('.', '_')}_beremin.vtu"
                written = write_path_fields(target, res.support, found.fields())
                log.info(
                    "cleavage of %s (Beremin): largest Weibull stress %.5g, probability of "
                    "failure %.4g; field: %s",
                    key, found.largest, float(found.probability[-1]), written.name,
                )  # fmt: skip
            values.update({f"{key}.{n}": v for n, v in res.components().items()})
            rows.append(res.row())
            for q in quantities:
                if q.path == res.name and q.bc in (None, res.bc):
                    rows.append({"property": q.name, "bc": res.bc, "value": q.of(res)})
            write_rows(outdir / f"{stem}_{key.replace('.', '_')}.csv", res.rows())
            for fmt in tables:
                write_table(outdir / f"{stem}_{key.replace('.', '_')}.{fmt}", res.rows())
            ends = res.path.times[1:]  # the fields of the files: at the ends of the segments
            kept = [(t, a) for t, a in sorted(res.fields.items()) if np.isclose(t, ends).any()]
            for k, (t, arrays) in enumerate(kept if spec.fields else []):
                target = outdir / f"{stem}_{key.replace('.', '_')}_{k + 1}.vtu"
                written = write_path_fields(target, res.support, arrays)
                log.info("fields of %s at time %g: %s", key, t, written.name)
            continue
        if isinstance(res, (Apparent, Response)):
            if isinstance(res, Apparent):
                values[key] = res.value
            else:
                values.update({f"{key}.{n}": v for n, v in res.components().items()})
            rows.append(res.row())
            if spec.fields and res.solution.fields is not None:
                _write_fields(res, outdir / f"{stem}_{key.replace('.', '_')}")
        else:
            pores, velocity = res.pop("pores", None), res.pop("velocity", None)
            if spec.fields and pores is not None and velocity is not None:
                written = _write_flow(pores, velocity, outdir / f"{stem}_{key.replace('.', '_')}")
                log.info("the velocities of the flows of %s: %s", key, written.name)
            named = {}
            prop, _, bc = key.partition(".")
            if prop in COUPLED:  # the tensors of a coupled law, by name
                named = coupled_entries(res)
            elif prop in HARMONIC or prop in FLOW:  # their scalar entries, by name
                named = {}
            elif prop not in ("thermal_expansion", "swelling"):  # theirs are named A.., B..
                named = matrix_entries(res["tensor"], "C")
            if "compliance" in res:  # of the stiffness: S = C^-1, in the same notation
                named.update(matrix_entries(res["compliance"], "S"))
            named.update({n: v for n, v in res.items() if isinstance(v, float)})
            values.update({f"{key}.{k}": v for k, v in named.items()})
            for name, entry in entries:  # the scalar results of the statistics of RVE studies
                if name == prop and entry in named:
                    rows.append({"property": f"{prop}.{entry}", "bc": bc, "value": named[entry]})
    if isinstance(sample, (TetMesh, TriMesh)):
        fractions = _mesh_fractions(sample, spec.materials)
    elif isinstance(sample, LevelSetSample):
        fractions = sample.phase_fractions()
    else:
        fractions = phase_fractions(sample)
    for phase, fraction in fractions.items():
        values.setdefault(f"fraction_{phase}", fraction)
    present = {p: f for p, f in fractions.items() if p in spec.materials}
    values.update(normalized_values(values, present, spec.materials))
    return values, rows, results


def _conforming_mesh(micro: Microstructure, cfg: RunConfig) -> tuple[TetMesh | TriMesh, dict]:
    """The conforming mesh of a realization: tetrahedra, or triangles of the plane
    (``mesh.plane``, by default for the columnar grains of tessellations of coplanar
    germs), and its summary."""
    homog = cfg.homogenization
    porous = homog.void_phases if homog is not None else set()
    if cfg.mesh.on_plane(micro):
        if homog is not None and "hydraulic_permeability" in homog.properties:
            porous = set()  # the flow through the pores: meshed with the solid
        mesh = plane_mesh(micro, **cfg.mesh.plane_kwargs(porous))
    else:
        mesh = conforming_mesh(micro, **cfg.mesh.kwargs(porous))
    q = mesh.quality()
    info = {
        "mesh_nodes": mesh.n_nodes,
        "mesh_elements": mesh.n_elements,
        "mesh_order": mesh.order,
        "mesh_quality_min": float(q.min()) if q.size else float("nan"),
        "mesh_time_s": sum(  # the background, octree, cells or complex, and MMG
            float(v) for k, v in mesh.metadata.items() if k.startswith("time_") and k.endswith("_s")
        ),
    }
    return mesh, info


def _process(
    micro: Microstructure,
    cfg: RunConfig,
    stem: str,
    outdir: Path,
    measures,
    index: int = 0,
    *,
    homogenize: bool = True,
) -> tuple[dict, list[dict]]:
    """Write the files of one realization (the ``index``-th of the run: the first ones
    have figures and videos); return its measurements and the rows of its apparent
    properties (none without ``homogenize``: a window study solves its realization
    itself)."""
    out = cfg.outputs
    for fmt in out.geometry:
        io.save_microstructure(micro, outdir / f"{stem}.{fmt}")
    homog = cfg.homogenization
    fe = homog is not None and homog.method == "fe"
    xfem = homog is not None and homog.method == "xfem"
    viz = cfg.visualization
    viz = viz if viz is not None and index < viz.realizations else None
    values: dict = {}
    mesh = None
    shown = viz is not None and viz.shows("mesh")
    foam = out.openfoam is not None and out.openfoam.get("sample") == "mesh"
    if cfg.mesh is not None and (
        fe
        or out.conforming_mesh
        or "volume_fraction" in measures
        or shown
        or out.code_aster is not None
        or foam
    ):
        mesh, info = _conforming_mesh(micro, cfg)
        checkpoint()  # a stop after the mesh (morphogen.stopping)
        values.update(info)
        plane = isinstance(mesh, TriMesh)
        if plane and (out.code_aster is not None or foam):
            raise ValueError(
                f"{stem}: a mesh of the plane (the columnar grains of coplanar germs); the "
                "files of Code_Aster and of OpenFOAM take tetrahedra (mesh.plane: false)"
            )
        kind = "tri" if plane else "tet"
        for fmt in out.conforming_mesh:
            # Abaqus inputs with the materials of the phases (MFront laws for the nonlinear,
            # and for all in finite strains; the elastic ones of the plane elements)
            extra = {}
            if fmt == "inp" and homog is not None:
                extra = {"materials": homog.materials}
                if not plane:
                    extra["kinematics"] = _kinematics(homog)
            target = outdir / f"{stem}_{kind}.{fmt}"
            try:
                io.save_mesh(mesh, target, **extra)
            except ValueError as error:
                if not (plane and extra):
                    raise
                log.warning("%s: the plane elements without materials (%s)", target.name, error)
                io.save_mesh(mesh, target)
        if out.periodic_pairs and mesh.periodic:
            io.write_periodic_pairs(outdir / f"{stem}_{kind}_periodic.csv", mesh)
        if out.periodic_equations and mesh.periodic:
            io.write_periodic_equations(outdir / f"{stem}_{kind}_equations.inp", mesh)
        if out.code_aster is not None:  # the study of the apparent properties of the mesh
            io.write_code_aster(outdir / f"{stem}_aster", mesh, homog.materials, name=stem,
                                **out.code_aster)  # fmt: skip
        if foam:
            options = {k: v for k, v in out.openfoam.items() if k != "sample"}
            io.write_openfoam(outdir / f"{stem}_openfoam", mesh, **options)
    if out.neper:
        io.write_neper(outdir / stem, micro)
    exact = [m for m in measures if m in _exact_names()]
    if exact:  # on the implicit geometry of the realization itself, without voxels
        from morphogen.morphology.measures import measure_exact

        values.update(measure_exact(micro, exact, seed=index))
    morph = [m for m in measures if m in _measure_names()]
    use_image = cfg.voxels is not None and (
        out.needs_voxels
        or bool(morph)
        or (homog is not None and not fe and not xfem and homogenize)
        or cfg.diffusion is not None
        or (viz is not None and viz.shows("image", "labels"))
    )
    image = labels = None
    if use_image:
        shown = viz is not None and viz.shows("labels")
        image = _voxel_outputs(micro, cfg, stem, outdir, measures, labels=shown)
        labels = image[1]
        image = image[0]
        if morph:
            values.update(measure(image, morph, labels=labels))
    elif "volume_fraction" in measures and mesh is not None:
        values["volume_fraction"] = _mesh_fractions(
            mesh, homog.materials if homog is not None else {0: None, 1: None}
        ).get(1, 0.0)
    if cfg.diffusion is not None:
        values.update(_diffuse(image, cfg.diffusion, stem, outdir))
    fe_rows: list[dict] = []
    results = None
    if homog is not None and homogenize:
        sample = mesh if fe else image
        if xfem:  # the level set on the grid of the voxels section, cut by the interfaces
            grid = cfg.voxels
            sample = LevelSetSample.of(micro, grid.shape, spacing=grid.voxel_size)
        grains = None
        if not fe and not xfem and _oriented(micro, homog):
            # crystals turned by the orientations of the grains of the image
            if labels is None:
                labels = micro.labels(grid=image.grid())
            grains = (labels, micro.orientations)
        quantities = path_quantities(cfg.rve.properties, homog) if cfg.rve is not None else []
        entries = property_entries(cfg.rve.properties, homog) if cfg.rve is not None else []
        keep = viz.fields() if viz is not None else False
        fe_values, fe_rows, results = _homogenize(
            sample,
            homog,
            stem,
            outdir,
            grains,
            quantities,
            keep=keep,
            tables=out.tables,
            entries=entries,
        )
        values.update(fe_values)
    if viz is not None:
        from morphogen.viz.workflow import visualize

        visualize(viz, stem, outdir, image=image, labels=labels, mesh=mesh, results=results)
    return values, fe_rows


def _oriented(micro: Microstructure, homog: HomogenizationSpec) -> bool:
    """Whether the materials include crystals and the microstructure orients its grains."""
    return any_oriented(homog.materials) and getattr(micro, "orientations", None) is not None


def _kinematics(homog: HomogenizationSpec) -> str:
    """The kinematics of the loading paths of a run (and of its exports)."""
    return homog.nonlinear.get("kinematics", "small")


def _diffuse(image, spec, stem: str, outdir: Path) -> dict:
    """Transient diffusion in the image: the uptake curve (``<stem>_diffusion.csv``: the
    mean concentration, its ratio to saturation and the mean of each phase), the fields
    at the times (``<stem>_diffusion_<k>.vti``); returns the relative uptakes."""
    from morphogen.coupling.diffusion import transient_diffusion

    res = transient_diffusion(
        image,
        spec.diffusivity,
        spec.times,
        solubility=spec.solubility,
        exposed=spec.exposed,
        boundary=spec.boundary(),
        initial=spec.initial,
        consumption=spec.consumption,
        steps=spec.steps,
    )
    labels = np.asarray(image.data)
    rows = []
    for k, t in enumerate(res.times):
        row = {
            "time": float(t),
            "uptake": float(res.uptake[k]),
            "relative_uptake": float(res.relative_uptake[k]),
        }
        for p in np.unique(labels):
            row[f"c_{int(p)}"] = float(res.fields[k][labels == p].mean())
        rows.append(row)
        if spec.fields:
            target = outdir / f"{stem}_diffusion_{k + 1}.vti"
            io.write_vti(
                target, image.with_data(res.fields[k].astype(np.float32), name="concentration")
            )
    write_rows(outdir / f"{stem}_diffusion.csv", rows)
    return {
        f"diffusion.relative_uptake_{k + 1}": float(v) for k, v in enumerate(res.relative_uptake)
    }


def _voxel_outputs(
    micro: Microstructure, cfg: RunConfig, stem: str, outdir: Path, measures, labels=False
):
    """Voxel image of a realization and the files derived from it; returns the image
    and the cell labels (tessellations, grain structures) when needed (``labels``: for
    the figures and videos). With ``voxels.section``, the image, the labels and the
    level sets are those of one layer of voxels of the volume (the surfaces stay those
    of the volume)."""
    out = cfg.outputs
    vox = cfg.voxels
    volume = micro.voxelize(**vox.kwargs())
    grid = volume.grid()  # of the volume: the fields of a section are its layers
    image = vox.layer(volume)
    wanted = labels or out.labels or out.mesh or out.damask or "cells" in measures
    labels = None
    if micro.has_labels and wanted:
        labels = vox.layer(micro.labels(grid=grid))
    if out.damask:
        homog = cfg.homogenization
        io.write_damask(
            outdir / f"{stem}_damask",
            labels,
            micro.orientations,
            micro.cell_phases,
            materials=homog.materials if homog is not None else None,
        )
    if out.amitex is not None:
        homog = cfg.homogenization
        grains = {}
        if _oriented(micro, homog):
            if labels is None:
                labels = vox.layer(micro.labels(grid=grid))
            grains = {"grains": labels.data, "orientations": micro.orientations}
        # the nonlinear materials go to AMITEX as MFront laws, along the paths only (all of
        # them in finite strains, whose runs are the paths)
        finite = _kinematics(homog) == "finite"
        nonlinear = not out.amitex.get("elastic_only") and any(
            m.nonlinear for m in homog.materials.values()
        )
        stiffness = not (nonlinear or finite) and (
            "stiffness" in homog.properties or not (homog.loadings or homog.paths)
        )
        io.write_amitex(outdir / f"{stem}_amitex", image, homog.materials,
                        loadings={} if finite else homog.loadings, paths=homog.paths,
                        stiffness=stiffness, name=stem, kinematics=_kinematics(homog), **grains,
                        **out.amitex)  # fmt: skip
    if out.openfoam is not None and out.openfoam.get("sample") != "mesh":
        options = {k: v for k, v in out.openfoam.items() if k != "sample"}
        io.write_openfoam(outdir / f"{stem}_openfoam", image, **options)
    exported = (
        vox.layer(micro.voxelize(grid=grid, supersampling=vox.supersampling))
        if (out.image and vox.supersampling > 1)
        else image
    )
    for fmt in out.image:
        io.save_image(exported, outdir / f"{stem}.{fmt}")
    for fmt in out.labels:
        if labels is None:
            raise ValueError(
                "outputs.labels: only tessellations and grain structures have cell labels"
            )
        io.save_image(labels, outdir / f"{stem}_labels.{fmt}")
    several = micro.implicit_phases() != [0, 1]  # one implicit function per phase
    if out.level_set:
        for suffix, field in implicit_functions(micro, grid=grid):
            for fmt in out.level_set:
                io.save_image(vox.layer(field), outdir / f"{stem}_levelset{suffix}.{fmt}")
    if out.mesh:
        mesh = voxel_mesh(image, phases=out.mesh_phases, labels=labels)
        homog = cfg.homogenization
        extra = {}
        if homog is not None:  # Abaqus inputs with the materials (and grains) of the run
            extra.update(materials=homog.materials, kinematics=_kinematics(homog))
            if labels is not None and _oriented(micro, homog):
                extra["orientations"] = micro.orientations
        for fmt in out.mesh:
            io.save_mesh(mesh, outdir / f"{stem}.{fmt}", **(extra if fmt == "inp" else {}))
        if out.periodic_pairs:
            io.write_periodic_pairs(outdir / f"{stem}_periodic.csv", mesh)
        if out.periodic_equations and image.periodic and out.mesh_phases is None:
            io.write_periodic_equations(outdir / f"{stem}_equations.inp", mesh)
    for fmt in out.surface:
        if several:  # the surface of each phase but the last (that of the others together)
            for k in micro.implicit_phases()[:-1]:
                field = micro.phase_level_set(k, grid=grid)
                io.save_mesh(surface_mesh(field), outdir / f"{stem}_phase{k}.{fmt}")
            continue
        try:
            field = micro.level_set(grid=grid)
        except NotImplementedError:
            field = micro.voxelize(grid=grid, supersampling=max(2, vox.supersampling))
        io.save_mesh(surface_mesh(field), outdir / f"{stem}.{fmt}")
    return image, labels


def _measure_names():
    from morphogen.analysis.measures import MEASURES

    return MEASURES


def _exact_names():
    from morphogen.morphology.measures import EXACT_MEASURES

    return EXACT_MEASURES


def _manifest(cfg: RunConfig, outdir: Path, extra: dict) -> Path:
    try:
        theory = cfg.model.theory(cfg.domain)
    except (ValueError, NotImplementedError):
        theory = {}
    data = {
        "versions": versions(),
        "config": cfg.source,
        "seed": cfg.seed,
        "model": cfg.model.to_config(),
        "theory": theory,
        **extra,
    }
    homog = cfg.homogenization
    if homog is not None and "stiffness" in homog.properties:  # of its columns C.., S..
        data["stiffness_notation"] = NOTATION
    if cfg.outputs.database is not None:
        data["database"] = str(Path(cfg.outputs.database).resolve())
    path = outdir / "run.json"
    if "task" in extra and extra["task"]["n"] > 1:
        path = outdir / f"run_task{extra['task']['id']}of{extra['task']['n']}.json"
    path.write_text(json.dumps(to_jsonable(data), indent=2) + "\n", encoding="utf-8")
    return path


def _check_task(task_id: int, n_tasks: int) -> None:
    if n_tasks < 1 or not 0 <= task_id < n_tasks:
        raise ValueError(f"task id {task_id} must lie in [0, {n_tasks})")


def run(
    cfg: RunConfig,
    *,
    task_id: int = 0,
    n_tasks: int = 1,
    jobs: int = 1,
    resume: bool = False,
) -> Path:
    """Execute a configuration; returns the output directory.

    ``n_tasks`` > 1 runs the share ``task_id`` of the realizations (a task of a job
    array; :func:`morphogen.parallel.merge` gathers the tasks); ``jobs`` processes run
    them on this machine; ``resume`` skips the realizations already done (see
    :mod:`morphogen.parallel`). Scripts calling it with ``jobs`` > 1 must guard their main
    code with ``if __name__ == "__main__":`` (processes are started with ``spawn``).
    """
    _check_task(task_id, n_tasks)
    if jobs < 1:
        raise ValueError("jobs: a positive number of processes")
    if cfg.visualization is not None:
        from morphogen.viz.workflow import check

        check(cfg.visualization)  # before any realization
    if cfg.threads is not None:
        settings.set_num_threads(cfg.threads)
    if cfg.seed is None:
        if n_tasks > 1:
            raise ValueError(
                "a run split in tasks needs a seed shared by the tasks: set 'seed' in the "
                "configuration or give --seed"
            )
        cfg.seed = int(np.random.SeedSequence().entropy)
        log.info("seed of the run: %d", cfg.seed)
    outdir = cfg.outputs.directory
    outdir.mkdir(parents=True, exist_ok=True)
    if cfg.rve is not None and cfg.rve.method == "windows":
        run_windows(cfg, outdir)
    else:
        run_items(cfg, outdir, task_id, n_tasks, jobs=jobs, resume=resume)
    if cfg.outputs.database is not None and n_tasks == 1:
        from morphogen.results import record_run

        record_run(outdir, cfg.outputs.database)
    return outdir


def _write_results(outdir, rows, fe_rows, task_id, n_tasks, tables=()) -> None:
    write_rows(outdir / _results_name(task_id, n_tasks), rows)
    if fe_rows:
        write_rows(outdir / _task_name("homogenization", task_id, n_tasks), fe_rows)
    for fmt in tables:  # the same tables as .txt or .xlsx
        for stem, table in (("results", rows), ("homogenization", fe_rows)):
            if table:
                name = Path(_task_name(stem, task_id, n_tasks)).with_suffix(f".{fmt}")
                write_table(outdir / name, table)


def run_items(
    cfg: RunConfig, outdir: Path, task_id: int, n_tasks: int, *, jobs: int = 1, resume=False
) -> None:
    """Realizations of a batch or of an ensemble RVE study (the share of one task)."""
    t0 = time.time()
    todo = parallel.task_items(parallel.items(cfg), task_id, n_tasks)
    records = parallel.execute(cfg, todo, outdir, jobs=jobs, resume=resume)
    rows, fe_rows = parallel.rows_of(records)
    _write_results(outdir, rows, fe_rows, task_id, n_tasks, cfg.outputs.tables)
    task = {"id": task_id, "n": n_tasks, "jobs": jobs}
    _manifest(cfg, outdir, {"task": task, "elapsed_s": time.time() - t0})
    if cfg.rve is not None and n_tasks == 1:
        write_report(rows, outdir, cfg.rve, cfg, fe_rows=fe_rows)


def run_batch(cfg: RunConfig, outdir: Path, task_id: int, n_tasks: int) -> None:
    """The items of a batch configuration that fall to a task (``task_id`` among ``n_tasks``)
    generated and processed, their outputs and records in ``outdir``."""
    run_items(cfg, outdir, task_id, n_tasks)


def run_ensemble(cfg: RunConfig, outdir: Path, task_id: int, n_tasks: int) -> None:
    """The realizations of an RVE study by the ensemble method that fall to a task (``task_id``
    among ``n_tasks``) generated and processed, their outputs and records in ``outdir``."""
    run_items(cfg, outdir, task_id, n_tasks)


def run_windows(cfg: RunConfig, outdir: Path) -> None:
    """An RVE study by the window method: one realization of the size of the study, measured in
    windows of the sizes given, its statistics, results and report in ``outdir``: of the
    volume fraction, and of the tensors of the windows when the homogenization computes
    them (:func:`morphogen.rve.window_tensors`: the realization solved once per unit load
    by the FFT solver, each window its mean mutual energies); with ``rve.shifts``, over the
    tilings shifted around the periodic realization, from the same solutions."""
    spec = cfg.rve
    t0 = time.time()
    size = spec.size
    domain = Domain(
        size=(size, size, size),
        origin=cfg.domain.origin,
        periodic=cfg.domain.periodic,
        unit=cfg.domain.unit,
    )
    micro = cfg.model.generate(domain, derive_seed(cfg.seed, 0))
    _process(micro, cfg, f"{cfg.outputs.name}_L{size:g}", outdir, [], homogenize=False)
    image = micro.voxelize(**cfg.voxels.kwargs())
    h = float(np.mean(image.spacing))
    edges = sorted({max(1, round(w / h)) for w in spec.window_sizes})
    options = study_options(spec)
    options.bootstrap = 0
    stats = rve.window_statistics(image, edges, phase=1, shifts=spec.shifts)
    rows = [{"window_edge": e * h, **row} for e, row in zip(edges, stats.rows(), strict=True)]
    analyses = []
    if "volume_fraction" in spec.properties:
        theory = _theory(cfg, domain)
        analyses.append(rve.analyze_statistics(stats, "volume_fraction", options, theory=theory))
    else:  # the windows alone (and their tilings)
        keep = ("window_edge", "volume", "n", "tilings")
        rows = [{k: row[k] for k in keep if k in row} for row in rows]
    tensors, apparent = [], {}
    for name in [p for p in spec.properties if rve.is_tensor(p)]:
        prop = name.partition(".")[0]
        windows = _window_tensors(micro, image, cfg.homogenization, edges, prop, spec.shifts)
        t = windows.statistics(options)
        tensors.append(t)
        apparent[t.key] = windows.apparent
        for row, entries in zip(rows, t.volume_rows(), strict=True):
            row.update({f"{t.key}.{k}": v for k, v in entries.items() if k[0] in "CS"})
    write_rows(outdir / "results.csv", rows)
    extra = {"task": {"id": 0, "n": 1}, "elapsed_s": time.time() - t0}
    if apparent:  # of the whole realization, from its mean stresses (fluxes)
        extra["apparent"] = apparent
    _manifest(cfg, outdir, extra)
    report = rve.StudyReport(analyses, {}, options, method="windows", tensors=tensors)
    rve_report.write(report, outdir)


def _window_tensors(micro, image, homog: HomogenizationSpec, edges, prop: str, shifts=1):
    """The tensors ``prop`` of the windows of a realization (its voxel ``image``; crystals
    turned by the orientations of its grains), by the FFT solver of the homogenization, over
    ``shifts`` tilings per axis."""
    grains = {}
    if _oriented(micro, homog):
        grains = {"grains": micro.labels(grid=image.grid()), "orientations": micro.orientations}
    loads = 3 if prop != "stiffness" or image.shape[2] == 1 else 6
    log.info("window tensors %s: %d unit loads on %s voxels", prop, loads, image.shape)
    return rve.window_tensors(image, homog.materials, edges, property=prop,
                              tol=homog.tolerance, maxiter=homog.max_iterations,
                              device=homog.device, shifts=shifts, **grains)  # fmt: skip


def _theory(cfg: RunConfig, domain: Domain) -> dict:
    try:
        return cfg.model.theory(domain)
    except (ValueError, NotImplementedError):
        return {}


def study_options(spec: RVESpec) -> rve.StudyOptions:
    """The options of an RVE study (:class:`morphogen.rve.StudyOptions`) from the section ``rve`` of
    a configuration."""
    return rve.StudyOptions(
        gamma=spec.gamma,
        min_volume=spec.min_volume,
        max_volume=spec.max_volume,
        relative_errors=tuple(spec.relative_errors),
        n_realizations=tuple(spec.n_realizations),
        bootstrap=spec.bootstrap,
        weights=spec.weights,
    )


def _counter(model: Model | None):
    """Mean number of objects of the model in a cube of the given volume."""
    if model is None:
        return None
    try:
        if model.expected_count(Domain.cube(1.0)) is None:
            return None
    except (ValueError, NotImplementedError):
        return None

    def count(volume: float) -> float:
        return float(model.expected_count(Domain.cube(volume ** (1.0 / 3.0))))

    return count


def _bounds(rows: list[dict], homog: HomogenizationSpec | None) -> dict:
    """Bounds from the mean phase fractions of the largest volume."""
    if homog is None or not rows:
        return {}
    volumes = np.array([float(r.get("volume", np.nan)) for r in rows])
    largest = [r for r, v in zip(rows, volumes, strict=True) if v == np.nanmax(volumes)]
    fractions = {}
    for key in dict.fromkeys(k for r in largest for k in r if k.startswith("fraction_")):
        # a phase absent from a realization: a zero fraction there
        values = [r.get(key) for r in largest]
        values = [0.0 if v is None or v == "" else float(v) for v in values]
        fractions[int(key.split("_")[1])] = float(np.nanmean(values))
    try:
        return bounds(fractions, homog.materials) if fractions else {}
    except (ValueError, KeyError):
        return {}


def build_report(
    rows: list[dict],
    spec: RVESpec,
    properties=None,
    theory=None,
    *,
    fe_rows: list[dict] | None = None,
    count=None,
    bound_values: dict | None = None,
    group_by: str | None = None,
) -> rve.StudyReport:
    """RVE report of results tables (wide ``rows``; long ``fe_rows`` from homogenization),
    with the mean stiffness and compliance per volume of each boundary condition whose
    stiffness the rows hold (:func:`morphogen.rve.tensors_from_rows`; the conductivity
    tensor likewise): all of them when ``properties`` is None, else those it names
    (``stiffness``, ``stiffness.<bc>``, ``conductivity_tensor``...)."""
    if not rows and not fe_rows:
        raise ValueError("no results to analyse")
    wanted = list(properties) if properties is not None else None
    options = study_options(spec)
    asked = None if wanted is None else [p for p in wanted if rve.is_tensor(p)]
    tensors = []
    if rows and asked != []:
        tensors = rve.tensors_from_rows(rows, properties=asked, options=options)
        if asked and not tensors:
            what = " or ".join(sorted({p.partition(".")[0] for p in asked}))
            raise ValueError(
                f"no {what} in the results for {asked}: the columns <tensor>.<bc>.C11 ... "
                "(C66 of a stiffness, C33 of a conductivity) of a homogenization computing it"
            )
    if wanted is not None:
        wanted = [p for p in wanted if not rve.is_tensor(p)]
    series: list[rve.Series] = []
    fe_names = set()
    if fe_rows:
        fe_series = rve.series_from_rows(fe_rows)
        fe_names = {s.key for s in fe_series}
        series += [
            s for s in fe_series if wanted is None or s.key in wanted or s.property in wanted
        ]
    if rows:
        morph = None
        if wanted is not None:
            morph = [
                p
                for p in wanted
                if p not in fe_names and not any(n.startswith(p + ".") for n in fe_names)
            ]
        if morph is None or morph:
            try:
                wide = rve.series_from_rows(rows, properties=morph, group_by=group_by)
            except ValueError:
                if morph:
                    raise
                wide = []
            series += [s for s in wide if s.key not in fe_names]
    if not series and not tensors:
        raise ValueError(f"no series to analyse for the properties {wanted}")
    return rve.analyze_study(
        series, options, count=count, bounds=bound_values, theory=theory, tensors=tensors
    )


def write_report(
    rows, outdir: Path, spec, cfg: RunConfig | None = None, properties=None, *, fe_rows=None
) -> rve.StudyReport:
    """Write rve_report.json/.md/.tex (and plots) of a study."""
    theory = _theory(cfg, cfg.domain) if cfg is not None else None
    if properties is None and cfg is not None and cfg.rve is not None:
        properties = cfg.rve.properties
    report = build_report(
        rows,
        spec,
        properties,
        theory,
        fe_rows=fe_rows,
        count=_counter(cfg.model) if cfg is not None else None,
        bound_values=_bounds(rows, cfg.homogenization if cfg is not None else None),
    )
    rve_report.write(report, outdir)
    return report


def report_markdown(report: rve.StudyReport) -> str:
    """The Markdown report of an RVE study (:func:`morphogen.rve.report.markdown`)."""
    return rve_report.markdown(report)


def collect_results(paths, stem: str = "results") -> list[dict]:
    """Rows of one or several results files (a directory: its ``<stem>.csv``, or all its
    ``<stem>*.csv`` files of split runs that were not merged)."""
    files: list[Path] = []
    for p in [paths] if isinstance(paths, (str, Path)) else paths:
        p = Path(p)
        if p.is_dir():
            merged = p / f"{stem}.csv"
            files.extend([merged] if merged.exists() else sorted(p.glob(f"{stem}*.csv")))
        else:
            files.append(p)
    if not files:
        raise ValueError(f"no {stem} files found in {paths}")
    rows = []
    for f in files:
        rows.extend(read_rows(f))
    return rows


def collect_homogenization(paths) -> list[dict]:
    """Rows of the homogenization files of result directories (empty if none)."""
    dirs = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
    dirs = [d for d in dirs if d.is_dir()]
    if not any(next(d.glob("homogenization*.csv"), None) for d in dirs):
        return []
    try:
        return collect_results(dirs, stem="homogenization")
    except ValueError:
        return []
