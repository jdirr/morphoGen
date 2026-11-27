"""The tools of the tab Solvers & formats: the inputs of other solvers for a morphology of
the workspace, and its files in the formats of other programs (the common files, geometry,
sections, VTK images, STL surface and conforming mesh .vtu, are those of the tab Export):

* AMITEX_FFTP (:func:`morphogen.io.write_amitex`): the voxel image of a morphology, the
  materials of its phases (AMITEX's elastic laws; MFront laws of the ``umat`` interface
  for the nonlinear behaviours, and for every phase in finite strains), the six unit
  strains of the apparent stiffness and a loading path; and the results of its runs read
  back (:func:`morphogen.io.read_amitex_run`: the averages of each run, the apparent
  stiffness and its moduli);
* Abaqus (:func:`morphogen.io.write_inp_tet`, :func:`morphogen.io.write_inp`): the
  conforming or voxel mesh of a morphology with the materials of its phases
  (``*ELASTIC``, isotropic or in the axes of the crystal of each grain with its
  ``*ORIENTATION``; MFront laws of the ``abaqus`` interface), the periodic equations of
  three reference nodes (:func:`morphogen.io.write_periodic_equations`; uniform strains on
  the faces of the box, KUBC, when the morphology is not periodic) and a step per segment
  of a loading path;
* Code_Aster (:func:`morphogen.io.write_code_aster`): the conforming mesh of a morphology
  in Code_Aster's format with the groups of its phases, grains and faces, the materials
  of its phases, the six unit strains and three unit gradients of its apparent stiffness
  and conductivity under periodic conditions (kinematic ones when the morphology is not
  periodic), the command and export files of ``run_aster``;
* OpenFOAM (:func:`morphogen.io.write_openfoam`): the polyMesh of the voxels of a phase of
  a morphology and the case of the flow through it (Darcy's permeability, ``simpleFoam``)
  or of the diffusion through it (``laplacianFoam``);
* meshes and images in other formats: Gmsh, Medit, FEniCSx, the periodic node pairs, Neper
  and DAMASK for tessellations, TIFF stacks and MetaImage;
* the MFront sources of morphoGen's behaviours (:func:`morphogen.io.mfront_source`), and
  the material files of Z-set translated, and those of the behaviours of the workspace
  written (:mod:`morphogen.homogenization.zmat`).

The tools write their files in the folder of their result, with a README of the commands
to run and an archive of them all (``.zip``, their folders kept); they list them for the
page (``files``: the inputs of the solvers are of kinds the page does not know). The
loadings are given in morphoGen's components (Voigt order 11, 22, 33, 23, 31, 12 with
engineering shear strains; in finite strains the components 11, 22, 33, 23, 13, 12 of the
gradient, upper triangular, and of the nominal stress): strains ``E`` or stresses ``S`` at
the times of a path (:meth:`morphogen.homogenization.LoadingPath.parse`)."""

from __future__ import annotations

import re
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np

from morphogen.gui.tools import Field, elastic_tables, tab, tool

tab(
    "interop",
    "Solvers & formats",
    "The inputs of other solvers for a morphology of the workspace: AMITEX_FFTP (its voxel "
    "image, the materials of its phases and the loadings; the results of the runs read "
    "back), Abaqus (its conforming or voxel mesh, materials, crystal orientations, "
    "periodic equations and loading steps), Code_Aster (its conforming mesh and groups, "
    "materials and the loadings of its apparent properties) and OpenFOAM (the flow through "
    "its pores, the diffusion through a phase); its meshes and images in other formats (Gmsh, "
    "Medit, FEniCSx, Neper, DAMASK, TIFF, MetaImage); the MFront sources of morphoGen's "
    "behaviours and the material files of Z-set. Each result has a README of the commands "
    "to run and an archive of its files.",
    order=85,
)

#: The kinematics of the exports of materials (MFront laws for all in finite strains)
KINEMATICS = (
    ("small", "small strains"),
    ("finite", "finite strains (every phase an MFront law)"),
)
#: The keys of a material of the configurations that give its elasticity: a behaviour of
#: the form with none of them completes the elastic constants of its phase, else replaces
#: them
ELASTIC_KEYS = frozenset(
    {"young", "poisson", "bulk", "shear", "cubic", "hexagonal", "stiffness", "zmat"}
)
BEHAVIOURS_DOC = (
    "phase -> a material of the configurations, completing or replacing its elastic "
    'constants: {"1": {"plasticity": {"yield": 300, "hardening": 1000}}} (von Mises), '
    '{"1": {"cubic": [168e3, 121e3, 75e3], "crystal_plasticity": {"tau0": 40, "K": 20, '
    '"n": 12}}} (a crystal, turned by the orientation of each grain), {"1": {"zmat": '
    '"alloy.mat"}} (Z-set)'
)
COMPONENTS_DOC = (
    "strains E11 ... E12 (engineering shears; in finite strains the gradient) and "
    "stresses S11 ... S12 at the times after the first (one value: reached at the end of "
    "the first segment and held; E11=0.01,0,-0.01: a cycle); the components not given "
    "are free (zero stress)"
)
#: The kinds of the files of the results, for the page (the others: data)
FILE_KINDS = {
    ".zip": "archive",
    ".md": "report",
    ".csv": "table",
    ".png": "figure",
    ".svg": "figure",
    ".mp4": "video",
    ".gif": "video",
    ".inp": "configuration",
    ".xml": "configuration",
    ".sh": "configuration",
    ".mfront": "configuration",
    ".yaml": "configuration",
    ".json": "configuration",
    ".mat": "configuration",
    ".comm": "configuration",
    ".export": "configuration",
    ".mail": "data",
}
#: Voxels per edge of the voxel meshes (8-node hexahedra, written as text)
MAX_VOXEL_MESH = 128
#: morphoGen's Voigt components
VOIGT = ("11", "22", "33", "23", "31", "12")
#: The row i (degree of freedom) and column j (reference node REF_X, REF_Y, REF_Z) of the
#: displacement gradient H, upper triangular, of each Voigt component: u_i(REF_j) =
#: H_ij L_j, and the force on that degree of freedom is A_j sigma_ij (A_j: the area of
#: the face normal to j)
GRADIENT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
REFERENCES = ("REF_X", "REF_Y", "REF_Z")
#: Lines of a source shown in the page (the file holds it all)
SHOWN_LINES = 120


# -- common ----------------------------------------------------------------------------------


def _name(value, item: dict | None = None) -> str:
    """The stem of the files of a result: the name of the form, or the model of the
    morphology (letters, digits, '_' and '-')."""
    model = (item or {}).get("model") or {}
    text = str(value or "").strip() or model.get("type") or "morphology"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "morphology"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _head(text: str, name: str) -> str:
    """The first :data:`SHOWN_LINES` lines of a text (the file ``name`` holds it all)."""
    lines = text.splitlines()
    if len(lines) <= SHOWN_LINES:
        return text
    rest = len(lines) - SHOWN_LINES
    return "\n".join([*lines[:SHOWN_LINES], f"... ({rest} more lines in {name})"]) + "\n"


def _edge(micro) -> float:
    return float(np.max(micro.domain.lengths))


def _image(micro, n, what: str, limit: int | None = None):
    """The voxel image of the phases of a morphology, ``n`` voxels along its longest edge
    (the memory checked)."""
    from morphogen.gui.api import MAX_SHAPE, _voxels

    limit = limit or MAX_SHAPE
    n = int(n)
    if not 2 <= n <= limit:
        raise ValueError(f"voxels per edge: from 2 to {limit}")
    _voxels(micro.domain, n, 24.0, what)
    return micro.voxelize(shape=n)


#: A behaviour of the workspace given to one phase (with its own elasticity)
KEPT_FIELDS = (
    Field(
        "kept",
        "item",
        label="behaviour of the workspace",
        items=("behaviour",),
        none="none: the materials above",
        doc="a material point or an import of Z-set (B), given to the phase below with its "
        "own elasticity",
    ),
    Field("kept_phase", "int", 1, label="its phase"),
)


def _specs(p: dict, phases) -> dict:
    """The materials of the form for the ``phases`` (their configurations): the elastic
    constants (a contrast of 10 without them), completed or replaced by the behaviours,
    and by the behaviour of the workspace chosen for one phase."""
    phases = sorted(int(q) for q in phases)
    spec = {int(k): v for k, v in (p.get("materials") or {}).items()}
    if not spec:
        spec = {q: {"young": 1.0 if k == 0 else 10.0, "poisson": 0.3} for k, q in enumerate(phases)}
    extra = p.get("behaviours") or {}
    if not isinstance(extra, dict):
        raise ValueError('behaviours: a mapping phase -> material, {"1": {...}}')
    for key, value in extra.items():
        try:
            phase = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"behaviours: the phases are integers, got {key!r}") from None
        base = spec.get(phase)
        if isinstance(value, dict) and isinstance(base, dict) and not ELASTIC_KEYS & set(value):
            value = {**base, **value}
        spec[phase] = value
    if p.get("kept"):
        phase = p.get("kept_phase")
        if phase is None or int(phase) not in phases:
            raise ValueError(f"its phase: one of the phases {phases}")
        spec[int(phase)] = p["kept"]["material"]
    return spec


def _materials(p: dict, phases, *, required: bool = True) -> dict:
    """The :class:`~morphogen.homogenization.Material` of each of the ``phases`` (see
    :func:`_specs`); a phase without one is an error (``required``) or left out."""
    from morphogen.homogenization.materials import materials_from_config

    spec = _specs(p, phases)
    phases = sorted(int(q) for q in phases)
    missing = [q for q in phases if q not in spec]
    if missing and required:
        raise ValueError(f"phases {missing}: no material (give their elastic constants)")
    chosen = {q: spec[q] for q in phases if q in spec}
    return materials_from_config(chosen) if chosen else {}


def _path(p: dict):
    """The loading path of the form: the components at the times, the increments of each
    segment."""
    from morphogen.homogenization.paths import LoadingPath

    components = str(p.get("components") or "").strip()
    if not components:
        raise ValueError(f"components: {COMPONENTS_DOC}")
    times = p.get("times") or [0.0, 1.0]
    increments = p.get("increments") or [10]
    text = (
        f"t={','.join(repr(float(t)) for t in times)} "
        f"n={','.join(str(int(n)) for n in increments)} {components}"
    )
    path = LoadingPath.parse(text)
    if any(path.temperature):
        raise ValueError("components: the changes of temperature (dT) are not exported")
    return path


def _path_fields(when: dict) -> list[Field]:
    return [
        Field("components", "str", "E11=0.01", when=when, doc=COMPONENTS_DOC),
        Field("times", "floats", [0.0, 1.0], when=when, doc="the times of the path, from 0"),
        Field(
            "increments",
            "ints",
            [10],
            when=when,
            doc="the increments of each segment (one number: the same for all)",
        ),
    ]


def _segments(path) -> list[dict]:
    """The rows of a table of the segments of a path."""
    rows = []
    for k in range(len(path.times) - 1):
        row = {
            "segment": k + 1,
            "from": path.times[k],
            "to": path.times[k + 1],
            "increments": path.increments[k],
        }
        for name, value, strain in zip(VOIGT, path.values[k + 1], path.controls, strict=True):
            row[("E" if strain else "S") + name] = value
        rows.append(row)
    return rows


def _section(ctx, data, kind: str, name: str, title: str) -> None:
    """A figure of the middle x-y section of a voxel array."""
    from morphogen.viz.png import slice_of, to_rgb, upscale

    rgb = to_rgb(slice_of(np.asarray(data), 2), kind)
    ctx.figure(upscale(rgb, max(1, 320 // max(rgb.shape[:2]))), name, title)


def _listing(ctx) -> list[dict]:
    """The files of a result for the page, those of its folders too, by kind."""
    out = []
    for path in sorted(ctx.folder.rglob("*")):
        if path.is_file():
            name = path.relative_to(ctx.folder).as_posix()
            kind = FILE_KINDS.get(path.suffix.lower(), "data")
            out.append({"name": name, "kind": kind, "url": ctx.url(name)})
    return out


def _finish(ctx, archive: str, out: dict) -> dict:
    """An archive of the files of a result (``<archive>.zip``, their folders kept) and the
    list of the files (the inputs of the solvers, of kinds the page does not know)."""
    files = [p for p in sorted(ctx.folder.rglob("*")) if p.is_file()]
    target = ctx.path(f"{archive}.zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ctx.folder).as_posix())
    out["archive"] = ctx.url(target.name)
    out["files"] = _listing(ctx)
    return out


def _conforming(micro, p: dict, materials: dict | None = None):
    """The conforming mesh of a morphology (its grains for tessellations), of its phases
    that are not voids: phase 1 alone when phase 0 is void, otherwise the mesh of all its
    phases without the voids."""
    from morphogen.mesh import conforming_mesh

    edge = _edge(micro)
    size = p.get("element_size") or edge / 12
    if not edge / 200 <= size <= edge:
        raise ValueError(f"element size: between {edge / 200:g} and {edge:g}")
    void = {q for q, m in (materials or {}).items() if m.void}
    phases = "solid" if 0 in void and 1 not in void else "all"
    mesh = conforming_mesh(micro, size=float(size), order=int(p.get("order") or 1), phases=phases)
    present = set(np.unique(mesh.phase).tolist())
    if present & void:
        if not present - void:
            raise ValueError("every phase of the mesh is void")
        mesh = mesh.subset(sorted(present - void))
    return mesh


# -- AMITEX_FFTP -----------------------------------------------------------------------------


@tool(
    "amitex_inputs",
    "interop",
    "AMITEX_FFTP inputs",
    "The inputs of AMITEX_FFTP, the FFT solver of the CEA, for a morphology: the material "
    "and zone of each voxel of its image (legacy VTK), the materials of its phases (the "
    "laws elasiso and elasaniso; crystals turned by the orientation of each grain; the "
    "nonlinear behaviours, and every phase in finite strains, MFront laws of the umat "
    "interface, their sources in mfront/), the algorithm, a loading file per run (the six "
    "unit strains of the apparent stiffness, a loading path) and the script compiling the "
    "laws and running them all with MPI.",
    [
        Field("source", "item", label="morphology", required=True),
        Field(
            "resolution", "int", 32, label="voxels per edge", doc="of the image (its longest edge)"
        ),
        Field("materials", "materials", of="source"),
        Field("behaviours", "json", label="behaviours of the phases", doc=BEHAVIOURS_DOC),
        *KEPT_FIELDS,
        Field("kinematics", "choice", "small", choices=KINEMATICS),
        Field(
            "loading",
            "choice",
            "stiffness",
            label="runs",
            choices=(
                ("stiffness", "the six unit strains: the apparent stiffness (elastic phases)"),
                ("path", "a loading path"),
                ("both", "the unit strains and a loading path (elastic phases)"),
            ),
        ),
        *_path_fields({"loading": ["path", "both"]}),
        Field(
            "elastic_only",
            "bool",
            False,
            label="elasticity only",
            doc="the elasticity of the nonlinear phases instead of their MFront laws",
        ),
        Field("fields", "bool", False, doc="the stress and strain fields of every increment"),
        Field(
            "tolerance", "float", doc="convergence criterion, below 1e-3 (blank: AMITEX's, 1e-4)"
        ),
        Field(
            "processes", "int", 4, label="MPI processes", doc="of the script (NP=... overrides it)"
        ),
        Field(
            "library",
            "str",
            "",
            label="library of the native laws",
            doc="the path of libUmatAmitex.so (blank: found through AMITEX_PATH)",
        ),
        Field("name", "str", "", doc="of the files (blank: the model of the morphology)"),
    ],
    order=1,
    menu="AMITEX_FFTP",
)
def amitex_inputs(p: dict, ctx) -> dict:
    import json

    from morphogen import io
    from morphogen.homogenization.materials import any_oriented

    item = p["source"]
    micro = item["micro"]
    name = _name(p["name"], item)
    loading = p["loading"]
    paths = {"path": _path(p)} if loading in ("path", "both") else {}
    tolerance = p["tolerance"]
    if tolerance is not None and not 0.0 < tolerance < 1e-3:
        raise ValueError("tolerance: a positive number below 1e-3 (AMITEX)")
    if p["processes"] < 1:
        raise ValueError("MPI processes: at least one")
    ctx.progress(0.05, "voxel image")
    image = _image(micro, p["resolution"], "AMITEX image")
    materials = _materials(p, np.unique(image.data))
    grains = {}
    orientations = getattr(micro, "orientations", None)
    if any_oriented(materials) and orientations is not None and micro.has_labels:
        grains = {"grains": micro.labels(grid=image.grid()).data, "orientations": orientations}
    if not micro.domain.periodic:
        warnings.warn(
            f"{item['title']} is not periodic: AMITEX's FFT solver makes its image periodic",
            stacklevel=2,
        )
    ctx.progress(0.3, "AMITEX files")
    export = io.write_amitex(
        ctx.folder,
        image,
        materials,
        paths=paths,
        stiffness=loading in ("stiffness", "both"),
        name=name,
        fields=p["fields"],
        tolerance=tolerance,
        library=p["library"] or "",
        processes=p["processes"],
        elastic_only=p["elastic_only"],
        kinematics=p["kinematics"],
        **grains,
    )
    manifest = json.loads((ctx.folder / f"{name}_amitex.json").read_text(encoding="utf-8"))
    ctx.progress(0.8, "README and archive")
    names = item.get("phases") or {}
    rows = [
        {
            "material": m["material"],
            "phase": m["phase"],
            "name": names.get(str(m["phase"]), ""),
            "law": m.get("law", ""),
            "zones": m.get("zones", 1),
            "MFront source": m.get("mfront", ""),
        }
        for m in manifest["materials"]
    ]
    ctx.table(rows, "materials", "materials of the export")
    if paths:
        ctx.table(_segments(paths["path"]), "path", "segments of the loading path")
    _section(ctx, image.data, "phase", "image", "x-y section of the image (phases)")
    if grains:
        _section(ctx, grains["grains"], "labels", "grains", "x-y section of the grains (zones)")
    readme = _amitex_readme(item, export, manifest)
    ctx.path("README.md").write_text(readme, encoding="utf-8")
    values = {
        "voxels": " x ".join(str(n) for n in manifest["shape"]),
        "materials": len(manifest["materials"]),
        "runs": ", ".join(manifest["runs"]),
        "MFront laws": len(list((ctx.folder / "mfront").glob("*.mfront"))),
        "kinematics": manifest["kinematics"],
    }
    if grains:
        values["grains"] = len(np.unique(grains["grains"]))
    return _finish(ctx, f"{name}_amitex", {"values": values, "text": readme})


def _amitex_readme(item: dict, export, manifest: dict) -> str:
    name = export.name
    script = (export.directory / f"{name}_run.sh").read_text(encoding="utf-8").splitlines()
    commands = [line for line in script if "mfront --obuild" in line or line.startswith("mpirun")]
    shape = " x ".join(str(n) for n in manifest["shape"])
    spacing = ", ".join(f"{v:.6g}" for v in manifest["spacing"])
    runs = ", ".join(f"{key} ({run['loading']})" for key, run in manifest["runs"].items())
    laws = sorted({m["mfront"] for m in manifest["materials"] if "mfront" in m})
    lines = [
        f"# AMITEX_FFTP runs of {item['title']}",
        "",
        f"Written by morphoGen {manifest['morphogen']}, in {manifest['kinematics']} strains: "
        f"the material and zone of each of the {shape} voxels (of {spacing}) in "
        f"{name}_materials.vtk and {name}_zones.vtk, "
        f"{_count(len(manifest['materials']), 'material')} in {name}_material.xml, the "
        f"algorithm in {name}_algorithm.xml and a loading file per run: {runs}. "
        f"{name}_amitex.json describes the export.",
        "",
        "## Run",
        "",
        "On a machine with AMITEX_FFTP and MPI (this folder, or its archive unzipped):",
        "",
        "    source <AMITEX>/env_amitex.sh    # AMITEX_PATH",
        f"    sh {name}_run.sh    # NP=16 sh {name}_run.sh: 16 MPI processes",
        "",
        "The script runs, from its folder:",
        "",
        *[f"    {line}" for line in commands],
        "",
    ]
    if laws:
        lines += [
            f"The MFront laws ({', '.join(laws)}) are compiled first by TFEL's mfront (on "
            "the PATH) with the umat interface (mfront/src/libUmatBehaviour.so); their "
            "internal variables start at zero.",
            "",
        ]
    lines += [
        "## Results",
        "",
        f"AMITEX writes the averages of each run in results/{name}_<run>.std (per material: "
        ".mstd, per zone: .zstd; the fields: .vtk). Read them back with the tool AMITEX_FFTP "
        "results of the tab Solvers & formats (the path of this folder), or",
        "",
        "    morphogen amitex read <this folder>",
        "",
        "which gives the averages of the runs and the apparent stiffness of the unit strains "
        "with its moduli. AMITEX orders the components xx, yy, zz, xy, xz, yz (engineering "
        "shear strains), morphoGen 11, 22, 33, 23, 31, 12.",
    ]
    return "\n".join(lines) + "\n"


@tool(
    "amitex_results",
    "interop",
    "AMITEX_FFTP results",
    "The results of the runs of an AMITEX_FFTP export, read back from its folder (the "
    "averages of each run over the unit cell, in morphoGen's components; the apparent "
    "stiffness of the six unit strains, its moduli and engineering constants; the averages "
    "of the materials), or of a single run (its .std file).",
    [
        Field(
            "directory",
            "str",
            required=True,
            label="folder of the export",
            doc="where <name>_amitex.json is, with results/ after the runs; or a .std file",
            browse="folder",
        ),
        Field("name", "str", "", label="name of the export", doc="blank: the only export there"),
    ],
    order=2,
    menu="AMITEX_FFTP",
)
def amitex_results(p: dict, ctx) -> dict:
    from morphogen import io

    target = Path(str(p["directory"]).strip().strip('"')).expanduser()
    if target.suffix == ".std":
        if not target.is_file():
            raise ValueError(f"no file {target}")
        prefix = target.with_suffix("")
        runs = {prefix.name: (io.read_amitex(prefix), prefix)}
        kinds = {prefix.name: "stiffness" if prefix.name.endswith("_stiffness") else "path"}
        stiffness, properties, missing = None, {}, []
    else:
        if not target.is_dir():
            raise ValueError(f"no folder {target}")
        run = io.read_amitex_run(target, p["name"] or None)
        manifest = run.manifest
        runs = {k: (r, target / manifest["runs"][k]["results"]) for k, r in run.results.items()}
        kinds = {k: v["kind"] for k, v in manifest["runs"].items()}
        stiffness, properties = run.stiffness, run.properties
        missing = sorted(set(manifest["runs"]) - set(run.results))
        if not runs:
            raise ValueError(
                f"no results yet in {target / 'results'} (runs {', '.join(manifest['runs'])}): "
                f"run {manifest['name']}_run.sh there first"
            )
    if missing:
        warnings.warn(f"runs without results: {', '.join(missing)}", stacklevel=2)
    if stiffness is not None:  # in morphoGen's order (read_amitex turns AMITEX's xy, xz, yz)
        elastic_tables(ctx, stiffness, "(AMITEX_FFTP)", kind="apparent")
    for k, (key, (res, prefix)) in enumerate(runs.items()):
        ctx.progress(k / len(runs), key)
        _amitex_run(ctx, key, res, prefix, kinds.get(key) != "stiffness")
    values = {k: float(v) for k, v in properties.items()}
    values["runs"] = ", ".join(runs)
    return {"values": values}


def _amitex_run(ctx, key: str, res, prefix: Path, curves: bool) -> None:
    """The tables of the averages of a run (of its materials too, when AMITEX wrote them)
    and the stress-strain curves of the components that vary along it (``curves``)."""
    from morphogen.io.amitex import read_amitex_materials

    finite = res.kinematics == "finite"
    names = ("11", "22", "33", "23", "13", "12") if finite else VOIGT
    strain, stress = ("H", "P") if finite else ("E", "S")
    stem = _name(key)  # the names of the files of the result

    def columns(r, step):
        out = {strain + c: float(r.strain[step, i]) for i, c in enumerate(names)}
        out.update({stress + c: float(r.stress[step, i]) for i, c in enumerate(names)})
        return out

    rows = []
    for step, t in enumerate(res.times):
        row = {"time": float(t), **columns(res, step)}
        if res.iterations is not None:
            row["iterations"] = int(res.iterations[step])
        rows.append(row)
    ctx.table(rows, f"run_{stem}", f"averages of the run {key}")
    if (prefix.parent / f"{prefix.name}.mstd").exists():
        per = read_amitex_materials(prefix, kinematics=res.kinematics)
        rows = [
            {"material": m, "time": float(r.times[-1]), **columns(r, -1)} for m, r in per.items()
        ]
        ctx.table(rows, f"materials_{stem}", f"averages of the materials at the end of {key}")
    if not curves:
        return
    try:
        from matplotlib.figure import Figure
    except ImportError:  # an optional dependency: the tables only
        warnings.warn("no curves without matplotlib: pip install 'morphogen[plot]'", stacklevel=2)
        return
    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    for i, c in enumerate(names):
        if np.ptp(res.strain[:, i]) > 0 or np.ptp(res.stress[:, i]) > 0:
            ax.plot(res.strain[:, i], res.stress[:, i], marker=".", lw=1.4, label=c)
    ax.set_xlabel(f"{strain}ij")
    ax.set_ylabel(f"{stress}ij")
    ax.set_title(f"run {key}", fontsize=10)
    ax.grid(alpha=0.3)
    if ax.lines:
        ax.legend(fontsize=8, frameon=False)
    ctx.figure(fig, f"curves_{stem}", f"stress-strain curves of {key}")


# -- Abaqus ----------------------------------------------------------------------------------


@tool(
    "abaqus_input",
    "interop",
    "Abaqus input",
    "The input of Abaqus for a morphology: its conforming mesh (tetrahedra; the grains of "
    "tessellations) or voxel mesh (hexahedra), the materials of its phases (*ELASTIC, "
    "isotropic or anisotropic in the axes of the crystal of each grain with its "
    "*ORIENTATION; the nonlinear behaviours, and every phase in finite strains, MFront laws "
    "of the abaqus interface, their sources in <name>_mfront/), the periodic equations of "
    "the reference nodes REF_X, REF_Y, REF_Z (uniform strains on the faces, KUBC, when the "
    "morphology is not periodic) and a step per segment of a loading path.",
    [
        Field("source", "item", label="morphology", required=True),
        Field(
            "mesh",
            "choice",
            "conforming",
            choices=(
                ("conforming", "conforming tetrahedra (the grains of tessellations)"),
                ("voxels", "voxels: 8-node hexahedra"),
            ),
        ),
        Field(
            "element_size",
            "float",
            unit="length",
            when={"mesh": "conforming"},
            doc="the largest element (blank: 1/12 of the longest edge)",
        ),
        Field(
            "order",
            "choice",
            "1",
            when={"mesh": "conforming"},
            choices=(("1", "linear: C3D4"), ("2", "quadratic: C3D10")),
        ),
        Field(
            "resolution",
            "int",
            24,
            label="voxels per edge",
            when={"mesh": "voxels"},
            doc=f"of the voxel mesh (at most {MAX_VOXEL_MESH})",
        ),
        Field("materials", "materials", of="source"),
        Field("behaviours", "json", label="behaviours of the phases", doc=BEHAVIOURS_DOC),
        *KEPT_FIELDS,
        Field("kinematics", "choice", "small", choices=KINEMATICS),
        Field(
            "loading",
            "choice",
            "path",
            label="steps",
            choices=(
                ("path", "a loading path: a step per segment"),
                ("none", "none: the model only (mesh, materials, equations)"),
            ),
        ),
        *_path_fields({"loading": "path"}),
        Field("name", "str", "", doc="of the files (blank: the model of the morphology)"),
    ],
    order=3,
    menu="Abaqus and meshes",
)
def abaqus_input(p: dict, ctx) -> dict:
    from morphogen import io

    item = p["source"]
    micro = item["micro"]
    name = _name(p["name"], item)
    kinematics = p["kinematics"]
    path = _path(p) if p["loading"] == "path" else None
    target = ctx.path(f"{name}.inp")
    ctx.progress(0.05, "mesh")
    mesh, materials, element = _abaqus_mesh(p, item, target, ctx)
    ctx.progress(0.6, "boundary conditions")
    lower = np.asarray(micro.domain.lower, dtype=float)
    lengths = np.asarray(micro.domain.lengths, dtype=float)
    periodic, equations = False, 0
    if micro.domain.periodic:
        scratch = ctx.path(f"{name}_equations.inp")
        try:
            io.write_periodic_equations(scratch, mesh)
        except ValueError as exc:
            warnings.warn(
                f"no periodic equations ({exc}): the faces under uniform strains (KUBC)",
                stacklevel=2,
            )
        else:
            text = scratch.read_text(encoding="ascii")
            equations = text.count("*EQUATION")
            with target.open("a", encoding="ascii", newline="\n") as f:
                f.write(text)
            periodic = True
        finally:
            scratch.unlink(missing_ok=True)
    if path is not None:
        # the nodes start at 293.15 K when a material expands: the steps keep them there (the
        # paths have no change of temperature), CalculiX failing on a step without one
        initial = re.search(r"^\*INITIAL CONDITIONS, TYPE=TEMPERATURE\n(.+)$",
                            target.read_text(encoding="ascii"), re.MULTILINE)  # fmt: skip
        with target.open("a", encoding="ascii", newline="\n") as f:
            f.write(_abaqus_steps(path, mesh, kinematics, periodic, lower, lengths,
                                  initial.group(1) if initial else None))  # fmt: skip
        ctx.table(_segments(path), "steps", "steps of the loading path")
    finite = kinematics == "finite"
    names = item.get("phases") or {}
    rows = []
    for q, m in materials.items():
        if m.nonlinear or finite:
            kind = "MFront law (abaqus interface)"
        elif m.stiffness is not None:
            kind = "elastic, anisotropic in the axes of the crystal"
        else:
            kind = "elastic, isotropic"
        rows.append({"phase": q, "name": names.get(str(q), ""), "material": kind})
    ctx.table(rows, "materials", "materials of the phases (the voids have no element)")
    laws = sorted(f.name for f in (ctx.folder / f"{name}_mfront").glob("*.mfront"))
    if periodic:
        conditions = "periodic equations"
    else:
        conditions = "KUBC (the faces)" if path is not None else "none"
    readme = _abaqus_readme(item, target, mesh, element, conditions, equations, path, laws, finite)
    ctx.path("README.md").write_text(readme, encoding="utf-8")
    values = {
        "nodes": int(mesh.n_nodes),
        "elements": int(mesh.n_elements),
        "element type": element,
        "boundary conditions": conditions,
        "equations": equations,
        "steps": 0 if path is None else len(path.times) - 1,
        "MFront laws": len(laws),
    }
    cells = getattr(mesh, "cell", None)
    cells = getattr(mesh, "label", None) if cells is None else cells
    if cells is not None:
        values["grains"] = len(np.unique(cells))
    return _finish(ctx, f"{name}_abaqus", {"values": values, "text": readme})


def _abaqus_mesh(p: dict, item: dict, target: Path, ctx):
    """The mesh of the tool Abaqus input written to ``target`` with the materials of its
    phases (the voids left out): its mesh, its materials and its type of elements."""
    from morphogen import io
    from morphogen.homogenization.materials import any_oriented

    micro = item["micro"]
    kinematics = p["kinematics"]
    if p["mesh"] == "voxels":
        from morphogen.mesh.voxel import voxel_mesh

        image = _image(micro, p["resolution"], "voxel mesh", MAX_VOXEL_MESH)
        materials = _materials(p, np.unique(image.data))
        solid = sorted(q for q, m in materials.items() if not m.void)
        if not solid:
            raise ValueError("every phase is void")
        labels = micro.labels(grid=image.grid()) if micro.has_labels else None
        mesh = voxel_mesh(image, None if len(solid) == len(materials) else solid, labels=labels)
        orientations = getattr(micro, "orientations", None)
        oriented = labels is not None and orientations is not None and any_oriented(materials)
        ctx.progress(0.4, "Abaqus input")
        io.write_inp(
            target,
            mesh,
            materials=materials,
            kinematics=kinematics,
            label_sets=labels is not None,
            orientations=orientations if oriented else None,
        )
        return mesh, {q: materials[q] for q in solid}, "C3D8"
    candidates = {int(k) for k in (item.get("phases") or {})} | {0, 1}
    candidates |= {int(k) for k in (p.get("materials") or {})}
    materials = _materials(p, candidates, required=False)
    mesh = _conforming(micro, p, materials)
    present = sorted(set(np.unique(mesh.phase).tolist()))
    missing = [q for q in present if q not in materials]
    if missing:
        raise ValueError(f"phases {missing}: no material (give their elastic constants)")
    materials = {q: materials[q] for q in present}
    ctx.progress(0.4, "Abaqus input")
    io.write_inp_tet(target, mesh, materials=materials, kinematics=kinematics)
    return mesh, materials, "C3D4" if mesh.order == 1 else "C3D10"


def _faces(mesh) -> list[str]:
    """The node sets of the upper faces of a mesh that it writes (those with nodes)."""
    if hasattr(mesh, "node_sets"):
        return [k.upper() for k in ("x1", "y1", "z1") if len(mesh.node_sets.get(k, ()))]
    return [f"{'XYZ'[a]}1" for a in range(3) if mesh.face_nodes(a, 1).size]


def _abaqus_steps(path, mesh, kinematics: str, periodic: bool, lower, lengths,
                  temperature: str | None = None) -> str:  # fmt: skip
    """The steps of a loading path, one per segment: on the reference nodes of the
    periodic equations (their rotation and a corner fixed; strains as displacements,
    stresses as forces), or on the nodes of the faces (KUBC: strains only).
    ``temperature``: the data line of ``*TEMPERATURE`` in each step (the nodes at their
    initial temperature when a material expands; CalculiX asks each step for one)."""
    if not periodic and not all(path.controls):
        raise ValueError(
            "stresses: the reference nodes of the periodic equations (a periodic morphology); "
            "the faces under uniform strains (KUBC) take strains only"
        )
    area = np.prod(lengths) / lengths
    boundary = None
    if not periodic:
        if hasattr(mesh, "boundary_nodes"):
            boundary = mesh.boundary_nodes()
        else:
            boundary = np.unique(np.concatenate(list(mesh.node_sets.values())))
    nlgeom = "YES" if kinematics == "finite" else "NO"
    lines = [
        "**",
        "** the loading path: the macroscopic displacement gradient H, upper triangular (the",
        "** strains, engineering shears in small strains), and the mean stresses",
    ]
    for k in range(len(path.times) - 1):
        t0, t1, n = path.times[k], path.times[k + 1], path.increments[k]
        period = t1 - t0
        dt = period / n
        values = path.values[k + 1]
        lines += [
            "**",
            f"** segment {k + 1}: from t = {t0:g} to {t1:g}, {n} increments",
            f"*STEP, NAME=SEGMENT_{k + 1}, NLGEOM={nlgeom}, INC={max(100, 10 * n)}",
            "*STATIC",
            f"{dt:.9g}, {period:.9g}, {dt * 1e-5:.9g}, {dt:.9g}",
            "*BOUNDARY",
        ]
        if periodic:
            lines += ["CORNER, 1, 3, 0.", "REF_X, 2, 3, 0.", "REF_Y, 3, 3, 0."]
            loads = []
            for (i, j), value, strain in zip(GRADIENT, values, path.controls, strict=True):
                if strain:
                    lines.append(f"{REFERENCES[j]}, {i + 1}, {i + 1}, {value * lengths[j]:.12g}")
                else:
                    loads.append(f"{REFERENCES[j]}, {i + 1}, {value * area[j]:.12g}")
            if loads:
                lines += ["*CLOAD", *loads]
        else:
            h = np.zeros((3, 3))
            for (i, j), value in zip(GRADIENT, values, strict=True):
                h[i, j] = value
            u = (np.asarray(mesh.nodes)[boundary] - lower) @ h.T
            for node, row in zip(boundary, u, strict=True):
                lines += [f"{node + 1}, {i + 1}, {i + 1}, {row[i]:.12g}" for i in range(3)]
        if temperature:
            lines += ["*TEMPERATURE", temperature]
        lines.append("*OUTPUT, FIELD, VARIABLE=PRESELECT")
        if periodic:
            for ref in REFERENCES:
                lines += [f"*NODE PRINT, NSET={ref}", "U, RF"]
        else:
            for face in _faces(mesh):
                lines += [f"*NODE PRINT, NSET={face}, TOTALS=YES", "RF"]
        lines.append("*END STEP")
    return "\n".join(lines) + "\n"


def _abaqus_readme(item, target: Path, mesh, element, conditions, equations, path, laws, finite):
    """The README of an Abaqus input (``target``)."""
    name = target.stem
    steps = 0 if path is None else len(path.times) - 1
    text = target.read_text(encoding="ascii")
    sets = "PHASE_<p>"
    if "*ELSET, ELSET=CELL_" in text:
        sets += " and CELL_<c> (the grains"
        sets += ", their orientations ORI_<c>)" if "*ORIENTATION" in text else ")"
    if getattr(mesh, "whole", False):
        faces = "OUTER, the boundary of the whole periodic cells"
    else:
        faces = "of the faces X0, X1, Y0, Y1, Z0, Z1"
    lines = [
        f"# Abaqus input of {item['title']}",
        "",
        f"{name}.inp: {mesh.n_nodes} nodes, {mesh.n_elements} {element} elements in the "
        f"element sets {sets}, the materials and their solid sections (the voids have no "
        f"element), the node sets {faces}"
        + (f", {equations} periodic equations" if equations else "")
        + f", {_count(steps, 'step')}.",
        "",
        "## Run",
        "",
    ]
    if laws:
        lines += [
            f"The {'' if finite else 'nonlinear '}materials are MFront laws ({name}_mfront/: "
            f"{', '.join(laws)}); compile them with TFEL, then run with the umat of MFront "
            "from this folder (the laws read their <behaviour>-parameters.txt there):",
            "",
            f"    cd {name}_mfront && mfront --obuild --interface=abaqus {' '.join(laws)} && cd ..",
            f"    export LD_LIBRARY_PATH=$PWD/{name}_mfront/src:$LD_LIBRARY_PATH",
            f"    abaqus job={name} input={name}.inp "
            "user=<TFEL>/share/doc/mfront/abaqus/umat.cpp interactive",
        ]
    else:
        lines.append(f"    abaqus job={name} input={name}.inp interactive")
    calculix = (
        f"CalculiX (ccx -i {name}) reads it as well with its elastic materials (it ignores "
        "the parameter NAME of *STEP and those of *OUTPUT)"
    )
    if conditions == "periodic equations":
        calculix += (
            ", but its reaction forces leave out those of the equations: average the "
            "stresses of the elements instead (*EL PRINT: S, with EVOL for their volumes)"
        )
    lines += ["", calculix + ".", "", "## Loading", ""]
    mean = "the mean nominal stresses" if finite else "the mean stresses"
    if conditions == "periodic equations":
        lines.append(
            "The periodic equations tie each node of the upper faces to its image and to the "
            "reference nodes REF_X, REF_Y, REF_Z: u_i(REF_j) = H_ij L_j, H the macroscopic "
            "displacement gradient, upper triangular (its rotation fixed: REF_X 2 and 3, "
            "REF_Y 3), and CORNER is fixed. A strain prescribes the displacement of a "
            "reference node, a stress the force A_j S_ij on it (A_j: the area of the face "
            f"normal to j); in Abaqus, their reaction forces over A_j are {mean} (*NODE PRINT "
            "of REF_X, REF_Y, REF_Z)."
        )
    elif conditions != "none":
        lines.append(
            "The nodes of the faces of the box follow the uniform strain u = H (x - x0) "
            f"(KUBC); the total reaction forces of the faces X1, Y1, Z1 over their areas are "
            f"{mean}."
        )
    else:
        lines.append("No step: add yours after the model.")
    if path is not None:
        lines += [
            "",
            "Components: morphoGen's (11, 22, 33, 23, 31, 12; engineering shears in small "
            "strains, the gradient in finite strains, NLGEOM=YES); a step per segment of the "
            "path, its times as the times of the steps.",
        ]
    return "\n".join(lines) + "\n"


# -- other formats ---------------------------------------------------------------------------

FORMATS = (
    ("msh", "Gmsh .msh: the conforming mesh"),
    ("medit", "Medit .mesh (MMG): the conforming mesh"),
    ("xdmf", "FEniCSx .xdmf: the conforming mesh, its cell and facet tags"),
    ("pairs", "periodic node pairs .csv of the conforming mesh"),
    ("voxel_msh", "Gmsh .msh: the voxel mesh (hexahedra)"),
    ("voxel_pairs", "periodic node pairs .csv of the voxel mesh"),
    ("neper", "Neper: the seeds, weights and orientations of a tessellation, its command"),
    ("damask", "DAMASK: the grid .vti and material.yaml of a polycrystal"),
    ("tif", "TIFF stack: a page per z section"),
    ("mhd", "MetaImage: .mhd header and .raw data"),
)
_CONFORMING = {"msh", "medit", "xdmf", "pairs"}
_VOXELS = {"voxel_msh", "voxel_pairs", "damask", "tif", "mhd"}
_NOTES = {
    "msh": "Gmsh 2.2 (ASCII): the physical volumes phase_<p> (tag p + 1; the elementary "
    "tag: the grain + 1), the physical surfaces x0 ... z1 (tags 101-106) or outer (107: "
    "the boundary of whole periodic cells); Gmsh, meshio, Code_Aster, Elmer",
    "mesh": "Medit (MMG): the tetrahedra of reference phase + 1 (grain + 1), the boundary "
    "triangles 1-6 (faces x0, x1, ..., z1) or 1 (whole cells); MMG, meshio, FreeFEM",
    "xdmf": 'FEniCSx: XDMFFile(MPI.COMM_WORLD, path, "r"), read_mesh(name="mesh"), '
    'read_meshtags(mesh, name="phase") (and "grain"), the facets "faces" (1-6: x0 ... z1)',
    "csv": "periodic node pairs (1-based ids, as in .msh and .inp files): axis, the node on "
    "the lower face, the node on the upper face; whole periodic cells: node, image and shift "
    "(x_node = x_image + shift * edges)",
    "neper": "Neper: sh <name>_neper.sh builds the tessellation (-morphooptiini from the "
    "seeds and weights, the orientations as Bunge angles)",
    "damask": "DAMASK: DAMASK_grid --geom <name>_damask.vti --material "
    "<name>_damask_material.yaml --load <load>.yaml (complete the phases of the material "
    "file: lattice, elastic and plastic behaviours)",
    "tif": "TIFF stack (ImageJ, Fiji: a page per z section, the voxel size in its metadata)",
    "mhd": "MetaImage: ITK, ParaView, Fiji, 3D Slicer (the .mhd header, the .raw data)",
}


@tool(
    "mesh_formats",
    "interop",
    "Meshes and images in other formats",
    "The meshes and images of a morphology in the formats of other programs: its "
    "conforming mesh (Gmsh, Medit, FEniCSx; its periodic node pairs), its voxel mesh "
    "(Gmsh, periodic node pairs), the files of Neper (tessellations) and DAMASK "
    "(polycrystals), its voxel image as a TIFF stack or a MetaImage.",
    [
        Field("source", "item", label="morphology", required=True),
        Field("formats", "choices", ["msh", "mhd"], choices=FORMATS),
        Field(
            "element_size",
            "float",
            unit="length",
            doc="conforming meshes: the largest element (blank: 1/12 of the longest edge)",
        ),
        Field(
            "order",
            "choice",
            "1",
            choices=(("1", "linear tetrahedra"), ("2", "quadratic tetrahedra (Gmsh)")),
        ),
        Field(
            "resolution",
            "int",
            64,
            label="voxels per edge",
            doc=f"images and voxel meshes (voxel meshes: at most {MAX_VOXEL_MESH})",
        ),
        Field(
            "image",
            "choice",
            "phases",
            label="image of",
            choices=(
                ("phases", "the phases (8 bits)"),
                ("labels", "the cells of tessellations and grain structures (16 or 32 bits)"),
                (
                    "level_set",
                    "the level set (32-bit floats; the implicit function of each phase of a "
                    "morphology of several phases, one file per phase)",
                ),
            ),
        ),
    ],
    order=4,
    menu="Abaqus and meshes",
)
def mesh_formats(p: dict, ctx) -> dict:
    from morphogen import io
    from morphogen.models.tessellation import Tessellation

    item = p["source"]
    micro = item["micro"]
    name = _name("", item)
    chosen = set(p["formats"] or [])
    if not chosen:
        raise ValueError("formats: choose at least one")
    orientations = getattr(micro, "orientations", None)
    if "neper" in chosen and not isinstance(micro, Tessellation):
        raise ValueError("Neper: the seeds of tessellations (Voronoi, Laguerre)")
    if "damask" in chosen and (not micro.has_labels or orientations is None):
        raise ValueError("DAMASK: polycrystals (tessellations with crystal orientations)")
    images = {"tif", "mhd"} & chosen
    if images and p["image"] == "labels" and not micro.has_labels:
        raise ValueError("image of the cells: tessellations and grain structures only")
    values: dict = {}
    notes: list[tuple[str, str]] = []
    if _CONFORMING & chosen:
        ctx.progress(0.05, "conforming mesh")
        mesh = _conforming(micro, p)
        values.update({"mesh nodes": int(mesh.n_nodes), "mesh elements": int(mesh.n_elements)})
        if "msh" in chosen:
            io.write_msh_tet(ctx.path(f"{name}.msh"), mesh)
            notes.append((f"{name}.msh", _NOTES["msh"]))
        if "medit" in chosen:
            io.write_medit_tet(ctx.path(f"{name}.mesh"), mesh)
            notes.append((f"{name}.mesh", _NOTES["mesh"]))
        if "xdmf" in chosen:
            io.write_fenicsx(ctx.path(f"{name}.xdmf"), mesh)
            notes.append((f"{name}.xdmf", _NOTES["xdmf"]))
        if "pairs" in chosen:
            if not mesh.periodic:
                warnings.warn(
                    "the conforming mesh is not periodic: its pairs are the nodes facing "
                    "each other",
                    stacklevel=2,
                )
            io.write_periodic_pairs(ctx.path(f"{name}_periodic.csv"), mesh)
            notes.append((f"{name}_periodic.csv", _NOTES["csv"]))
    if _VOXELS & chosen:
        ctx.progress(0.5, "voxel image")
        meshed = {"voxel_msh", "voxel_pairs"} & chosen
        image = _image(micro, p["resolution"], "voxel image", MAX_VOXEL_MESH if meshed else None)
        values["voxels"] = " x ".join(str(n) for n in image.shape)
        labels = micro.labels(grid=image.grid()) if micro.has_labels else None
        if meshed:
            from morphogen.mesh.voxel import voxel_mesh

            vmesh = voxel_mesh(image, labels=labels)
            values["voxel mesh nodes"] = int(vmesh.n_nodes)
            values["voxel mesh elements"] = int(vmesh.n_elements)
            if "voxel_msh" in chosen:
                io.write_msh(ctx.path(f"{name}_voxels.msh"), vmesh)
                notes.append((f"{name}_voxels.msh", _NOTES["msh"]))
            if "voxel_pairs" in chosen:
                io.write_periodic_pairs(ctx.path(f"{name}_voxels_periodic.csv"), vmesh)
                notes.append((f"{name}_voxels_periodic.csv", _NOTES["csv"]))
        if "damask" in chosen:
            phases = {int(k): v for k, v in (item.get("phases") or {}).items()}
            io.write_damask(
                ctx.path(f"{name}_damask"),
                labels,
                orientations,
                getattr(micro, "cell_phases", None),
                phase_names=phases,
            )
            notes.append((f"{name}_damask.vti, {name}_damask_material.yaml", _NOTES["damask"]))
        if images:
            kind = {"phases": "phase", "labels": "labels", "level_set": "field"}[p["image"]]
            for suffix, data in _images_of(micro, image, labels, p["image"]):
                if "tif" in chosen:
                    io.write_tiff(ctx.path(f"{name}{suffix}.tif"), data)
                    notes.append((f"{name}{suffix}.tif", _NOTES["tif"]))
                if "mhd" in chosen:
                    io.write_mhd(ctx.path(f"{name}{suffix}.mhd"), data)
                    notes.append((f"{name}{suffix}.mhd, {name}{suffix}.raw", _NOTES["mhd"]))
                what = f"{p['image']}{suffix.replace('_', ' ')}"
                _section(
                    ctx, data.data, kind, f"image{suffix}", f"x-y section of the image ({what})"
                )
    if "neper" in chosen:
        io.write_neper(ctx.path(name), micro)
        notes.append((f"{name}_seeds.txt, {name}_neper.sh", _NOTES["neper"]))
    lines = [f"# {item['title']} in other formats", ""]
    lines += [f"* {files}: {text.replace('<name>', name)}." for files, text in notes]
    readme = "\n".join(lines) + "\n"
    ctx.path("README.md").write_text(readme, encoding="utf-8")
    return _finish(ctx, f"{name}_formats", {"values": values, "text": readme})


# -- Code_Aster and OpenFOAM ----------------------------------------------------------------


@tool(
    "code_aster_study",
    "interop",
    "Code_Aster study",
    "The study of Code_Aster of the apparent properties of a morphology: its conforming mesh "
    "in Code_Aster's format (its voids left out; the groups of its phases, of the grains of "
    "its crystals, of the faces of the box and of the periodic nodes), the materials of its "
    "phases (ELAS, ELAS_ORTH of crystals turned by their grains; THER, THER_ORTH), the six "
    "unit strains and the three unit gradients of the apparent stiffness and conductivity "
    "under periodic conditions (LIAISON_GROUP of the nodes of the upper faces and their "
    "images; kinematic conditions when the morphology is not periodic), the tables of the "
    "integrals of the stresses and fluxes, and the export file of run_aster; read back by "
    "morphogen code-aster read.",
    [
        Field("source", "item", label="morphology", required=True),
        Field(
            "element_size",
            "float",
            unit="length",
            doc="the largest element (blank: 1/12 of the longest edge)",
        ),
        Field(
            "order",
            "choice",
            "1",
            choices=(("1", "linear: TETRA4"), ("2", "quadratic: TETRA10")),
        ),
        Field("materials", "materials", of="source"),
        Field(
            "physics",
            "choice",
            "auto",
            choices=(
                ("auto", "those that the materials have"),
                ("elastic", "the stiffness"),
                ("thermal", "the conductivity"),
            ),
        ),
        Field(
            "conditions",
            "choice",
            "auto",
            label="boundary conditions",
            choices=(
                ("auto", "periodic for a periodic morphology, kinematic otherwise"),
                ("periodic", "periodic (LIAISON_GROUP)"),
                ("kinematic", "kinematic: affine on the faces"),
            ),
        ),
        Field(
            "mesh_format",
            "choice",
            "aster",
            label="mesh",
            choices=(("aster", "Code_Aster's format (.mail)"), ("med", "MED (needs h5py)")),
        ),
        Field("name", "str", "", doc="of the files (blank: the model of the morphology)"),
    ],
    order=4,
    menu="Code_Aster and OpenFOAM",
)
def code_aster_study(p: dict, ctx) -> dict:
    from morphogen import io

    item = p["source"]
    micro = item["micro"]
    name = _name(p["name"], item)
    ctx.progress(0.05, "mesh")
    candidates = {int(k) for k in (item.get("phases") or {})} | {0, 1}
    candidates |= {int(k) for k in (p.get("materials") or {})}
    materials = _materials(p, candidates, required=False)
    mesh = _conforming(micro, p, materials)
    present = sorted(set(np.unique(mesh.phase).tolist()))
    missing = [q for q in present if q not in materials]
    if missing:
        raise ValueError(f"phases {missing}: no material (give their elastic constants)")
    materials = {q: materials[q] for q in present}
    ctx.progress(0.5, "Code_Aster study")
    study = io.write_code_aster(
        ctx.folder,
        mesh,
        materials,
        physics=None if p["physics"] == "auto" else p["physics"],
        conditions=None if p["conditions"] == "auto" else p["conditions"],
        name=name,
        mesh_format=p["mesh_format"],
    )
    import json

    manifest = json.loads((ctx.folder / f"{name}_aster.json").read_text(encoding="utf-8"))
    conditions = manifest["conditions"]
    rows = [{"loading": case, "physics": spec["physics"], "table": spec["table"],
             "unit": spec["unit"]} for case, spec in study.cases.items()]  # fmt: skip
    ctx.table(rows, "loadings", "the loadings of the study and the tables of their results")
    readme = (
        f"# Code_Aster study of {name}\n\n"
        f"The conforming mesh of the morphology ({mesh.n_elements} tetrahedra of "
        f"{mesh.order * 6 - 2} nodes, in {name}.{'mail' if p['mesh_format'] == 'aster' else 'med'}"
        f"), the materials of its phases and the {len(study.cases)} loadings of its apparent "
        f"properties under {conditions} conditions ({name}.comm). Run it with\n\n"
        f"    run_aster {name}.export\n\n"
        "and read the apparent stiffness and conductivity from the tables it writes with\n\n"
        "    morphogen code-aster read .\n"
    )
    ctx.path("README.md").write_text(readme, encoding="utf-8")
    values = {
        "nodes": int(mesh.n_nodes),
        "elements": int(mesh.n_elements),
        "boundary conditions": conditions,
        "loadings": len(study.cases),
    }
    return _finish(ctx, f"{name}_code_aster", {"values": values, "text": readme})


@tool(
    "openfoam_case",
    "interop",
    "OpenFOAM case",
    "The case of OpenFOAM of a phase of a morphology: the polyMesh of its voxels (a hexahedral "
    "cell per voxel; the faces of the box, the walls of the phase, cyclic pairs of a periodic "
    "morphology), and the steady flow through it under a pressure drop (simpleFoam: Darcy's "
    "permeability from the flow rate through the outlet) or the steady diffusion through it "
    "(laplacianFoam: its effective diffusivity); read back by morphogen openfoam read.",
    [
        Field("source", "item", label="morphology", required=True),
        Field(
            "resolution", "int", 32, label="voxels per edge", doc="of the image (its longest edge)"
        ),
        Field(
            "kind",
            "choice",
            "flow",
            choices=(
                ("flow", "flow through the phase: the permeability"),
                ("diffusion", "diffusion through the phase: the effective diffusivity"),
            ),
        ),
        Field("phase", "int", 0, doc="the phase of the cells (0: the pores)"),
        Field("axis", "choice", "0", choices=(("0", "x"), ("1", "y"), ("2", "z"))),
        Field(
            "lateral",
            "choice",
            "auto",
            label="other faces",
            choices=(
                ("auto", "cyclic for a periodic morphology, symmetry planes otherwise"),
                ("symmetryPlane", "symmetry planes"),
                ("cyclic", "cyclic pairs"),
            ),
        ),
        Field("viscosity", "float", 1.0, when={"kind": "flow"}, doc="kinematic viscosity"),
        Field(
            "pressure_drop",
            "float",
            1.0,
            when={"kind": "flow"},
            doc="kinematic pressure (p / rho) at the inlet, 0 at the outlet",
        ),
        Field("diffusivity", "float", 1.0, when={"kind": "diffusion"}, doc="DT"),
        Field("name", "str", "", doc="of the case (blank: the model of the morphology)"),
    ],
    order=5,
    menu="Code_Aster and OpenFOAM",
)
def openfoam_case(p: dict, ctx) -> dict:
    from morphogen import io

    item = p["source"]
    micro = item["micro"]
    name = _name(p["name"], item)
    image = _image(micro, p["resolution"], "OpenFOAM case", MAX_VOXEL_MESH)
    ctx.progress(0.3, "OpenFOAM case")
    options = {}
    if p["kind"] == "flow":
        options.update(viscosity=float(p["viscosity"]), pressure_drop=float(p["pressure_drop"]))
    else:
        options["diffusivity"] = float(p["diffusivity"])
    export = io.write_openfoam(
        ctx.folder / name,
        image,
        kind=p["kind"],
        phases=(int(p["phase"]),),
        axis=int(p["axis"]),
        lateral=None if p["lateral"] == "auto" else p["lateral"],
        **options,
    )
    m = export.manifest
    ctx.table(
        [{"patch": q.name, "type": q.type, "faces": q.size} for q in export.mesh.patches],
        "patches",
        "the patches of the mesh",
    )
    solver = "simpleFoam" if p["kind"] == "flow" else "laplacianFoam"
    readme = (
        f"# OpenFOAM case of {name}\n\n"
        f"The {p['kind']} through the phase {p['phase']} of the morphology "
        f"({_count(export.mesh.n_cells, 'cell')}, {m['connected_fraction']:.4g} of the box: what "
        f"joins {m['inlet']} to {m['outlet']}), from {m['inlet']} to {m['outlet']}, the other "
        f"faces {m['lateral']}. Run it ({solver}) in {name}/, with the environment of OpenFOAM, "
        "by\n\n    ./Allrun\n\nand read the "
        f"{'permeability' if p['kind'] == 'flow' else 'effective diffusivity'} with\n\n"
        f"    morphogen openfoam read {name}\n"
    )
    ctx.path("README.md").write_text(readme, encoding="utf-8")
    values = {
        "cells": export.mesh.n_cells,
        "faces": len(export.mesh.faces),
        "fraction of the phase": m["fraction"],
        "fraction meshed": m["connected_fraction"],
    }
    return _finish(ctx, f"{name}_openfoam", {"values": values, "text": readme})


def _images_of(micro, image, labels, kind: str) -> list[tuple[str, object]]:
    """The images to write on the grid of ``image``, each with the suffix of its files:
    the image of the phases, of the cells (16 bits when they fit) or of the level set
    (32-bit floats; for a morphology of several phases, the implicit function of each
    phase, ``_phase<k>``)."""
    from morphogen.models.implicit import implicit_functions

    if kind == "phases":
        return [("", image)]
    if kind == "labels":
        data = np.asarray(labels.data)
        cells = labels.with_data(data.astype(np.uint16 if data.max() < 2**16 else np.int32))
        return [("", cells)]
    try:
        functions = implicit_functions(micro, grid=image.grid())
    except NotImplementedError:
        raise ValueError(f"{type(micro).__name__}: no level set; choose its phases") from None
    return [(s, f.with_data(np.asarray(f.data, dtype=np.float32))) for s, f in functions]


# -- MFront and Z-set ------------------------------------------------------------------------

#: The default names of the behaviours of the tool MFront sources
BEHAVIOUR_NAMES = {
    "vonmises": "MorphogenVonMises",
    "crystal": "MorphogenCrystal",
    "elastic": "MorphogenHencky",
    "material": "MorphogenMaterial",
    "workspace": "MorphogenBehaviour",
}
_CRYSTAL = {"behaviour": "crystal"}
_MISES = {"behaviour": "vonmises"}
FAMILIES = (
    ("octahedral", "octahedral {111}<110> (FCC)"),
    ("cube", "cube {100}<110>"),
    ("bcc110", "{110}<111> (BCC)"),
    ("bcc112", "{112}<111> (BCC)"),
)


@tool(
    "mfront_sources",
    "interop",
    "MFront sources",
    "The MFront source of a behaviour of morphoGen, for the codes of MFront's interfaces "
    "(AMITEX_FFTP and Cast3M with umat, Abaqus, Code_Aster, MGIS): von Mises "
    "elasto-(visco)plasticity (the brick StandardElastoViscoPlasticity: linear and Voce "
    "isotropic hardening, Armstrong-Frederick and Prager kinematic hardening, Norton's "
    "flow), Méric-Cailletaud crystal plasticity (morphoGen's slip systems and interactions; "
    "in finite strains its kernel), Hencky's elasticity in finite strains, or a material of "
    "the configurations; with the material of morphoGen whose behaviour it is.",
    [
        Field(
            "behaviour",
            "choice",
            "vonmises",
            choices=(
                ("vonmises", "von Mises elasto-(visco)plasticity"),
                ("crystal", "Méric-Cailletaud crystal plasticity"),
                ("elastic", "Hencky elasticity (finite strains)"),
                ("material", "a material of the configurations (JSON)"),
                ("workspace", "a behaviour of the workspace (B: a material point, Z-set)"),
            ),
        ),
        Field(
            "kept",
            "item",
            label="behaviour of the workspace",
            items=("behaviour",),
            required=True,
            when={"behaviour": "workspace"},
        ),
        Field(
            "kinematics",
            "choice",
            "small",
            choices=(("small", "small strains"), ("finite", "finite strains")),
        ),
        Field("name", "str", "", doc="of the behaviour: letters and digits (blank: its kind)"),
        Field(
            "elasticity",
            "choice",
            "isotropic",
            when={"behaviour": ["vonmises", "crystal", "elastic"]},
            choices=(("isotropic", "isotropic: E, ν"), ("cubic", "cubic crystal: C11, C12, C44")),
        ),
        Field(
            "young", "float", 200000.0, label="E", unit="stress", when={"elasticity": "isotropic"}
        ),
        Field("poisson", "float", 0.3, label="ν", when={"elasticity": "isotropic"}),
        Field(
            "cubic",
            "floats",
            [168000.0, 121000.0, 75000.0],
            label="C11, C12, C44",
            unit="stress",
            when={"elasticity": "cubic"},
        ),
        Field("yield_stress", "float", 300.0, label="yield stress R0", unit="stress", when=_MISES),
        Field("hardening", "float", 1000.0, label="linear hardening H", unit="stress", when=_MISES),
        Field(
            "voce",
            "floats",
            label="Voce terms Q, b",
            when=_MISES,
            doc="pairs Q1, b1, Q2, b2 (at most three): R = R0 + H p + Σ Q (1 - exp(-b p))",
        ),
        Field(
            "kinematic",
            "floats",
            label="kinematic terms C, D",
            when=_MISES,
            doc="pairs C1, D1, ... (at most four; D = 0: Prager): X = 2/3 C α",
        ),
        Field(
            "norton",
            "floats",
            label="Norton K, n",
            when=_MISES,
            doc="viscoplasticity, dp/dt = <f / K>^n (blank: rate-independent)",
        ),
        Field(
            "families",
            "choices",
            ["octahedral"],
            label="slip families",
            choices=FAMILIES,
            when=_CRYSTAL,
        ),
        Field(
            "tau0",
            "float",
            40.0,
            label="τ0",
            unit="stress",
            when=_CRYSTAL,
            doc="critical resolved shear stress",
        ),
        Field(
            "Q",
            "float",
            0.0,
            unit="stress",
            when=_CRYSTAL,
            doc="r = τ0 + Q Σ h_sr (1 - exp(-b v_r))",
        ),
        Field("b", "float", 0.0, when=_CRYSTAL),
        Field("C", "float", 0.0, unit="stress", when=_CRYSTAL, doc="x = C α, dα = dγ - D α |dγ|"),
        Field("D", "float", 0.0, when=_CRYSTAL),
        Field(
            "K", "float", 20.0, unit="stress", when=_CRYSTAL, doc="dγ/dt = <(|τ - x| - r) / K>^n"
        ),
        Field("n", "float", 12.0, when=_CRYSTAL),
        Field(
            "interaction",
            "floats",
            when=_CRYSTAL,
            doc="h_self, h_latent, or the six coefficients self, coplanar, Hirth, collinear, "
            "glissile, Lomer of the octahedral systems (blank: all 1)",
        ),
        Field(
            "material",
            "json",
            when={"behaviour": "material"},
            doc='e.g. {"young": 70e3, "poisson": 0.33, "plasticity": {"yield": 100, '
            '"isotropic": [{"Q": 50, "b": 20}], "norton": {"K": 50, "n": 5}}}',
        ),
        Field(
            "compile",
            "bool",
            False,
            label="compile it",
            doc="with MFront's generic interface, when mfront is on the PATH",
        ),
    ],
    order=5,
    menu="Behaviours of materials",
)
def mfront_sources(p: dict, ctx) -> dict:
    import yaml

    from morphogen._utils import to_jsonable
    from morphogen.io.mfront import (
        identifier,
        mfront_state_size,
        mfront_strain_measure,
        mfront_symmetry,
        umat_function,
        write_mfront,
    )

    kinematics = p["kinematics"]
    if p["behaviour"] == "elastic" and kinematics == "small":
        raise ValueError(
            "Hencky elasticity: finite strains (in small strains, the elastic phases need no "
            "MFront law: the codes have their own)"
        )
    material = _behaviour(p)
    name = identifier(p["name"] or BEHAVIOUR_NAMES[p["behaviour"]])
    source = write_mfront(material, ctx.path(f"{name}.mfront"), name=name, kinematics=kinematics)
    config = {"materials": {1: to_jsonable(material.to_config())}}
    ctx.path("material.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    states = mfront_state_size(material, kinematics)
    symmetry = mfront_symmetry(material, kinematics)
    measure = mfront_strain_measure(material, kinematics)
    gradient = "the deformation gradient" if kinematics == "finite" else "the small strain"
    values = {
        "behaviour": name,
        "kinematics": kinematics,
        "symmetry": symmetry,
        "internal state variables": states,
        "strain measure": measure or gradient,
        "umat function": umat_function(name, measure),
        "Abaqus material": f"ABAQUSBEHAVIOUR_{name.upper()}_3D",
    }
    if p["compile"]:
        if shutil.which("mfront") is None:
            warnings.warn("mfront is not on the PATH: the source is not compiled", stacklevel=2)
        else:
            from morphogen.homogenization.mfront import compile_mfront

            ctx.progress(0.3, "compiling with MFront")
            values["library"] = str(compile_mfront(source))
    readme = _mfront_readme(name, kinematics, symmetry, states, measure, values)
    ctx.path("README.md").write_text(readme, encoding="utf-8")
    text = _head(source.read_text(encoding="utf-8"), source.name)
    return _finish(ctx, name, {"values": values, "text": readme + "\n" + text})


def _behaviour(p: dict):
    """The material of the form of the tool MFront sources."""
    from morphogen.homogenization.crystal import PARAMS
    from morphogen.homogenization.materials import Material

    kind = p["behaviour"]
    if kind == "workspace":
        return Material.from_config(p["kept"]["material"])
    if kind == "material":
        if not p["material"]:
            raise ValueError("material: a material of the configurations (JSON)")
        return Material.from_config(p["material"])
    if p["elasticity"] == "cubic":
        constants = p["cubic"] or []
        if len(constants) != 3:
            raise ValueError("C11, C12, C44: the three constants of the cubic crystal")
        spec: dict = {"cubic": constants}
    else:
        spec = {"young": p["young"], "poisson": p["poisson"]}
    if kind == "vonmises":
        plasticity: dict = {"yield": p["yield_stress"], "hardening": p["hardening"] or 0.0}
        for key, terms, a, b in (
            ("isotropic", p["voce"], "Q", "b"),
            ("kinematic", p["kinematic"], "C", "D"),
        ):
            terms = terms or []
            if len(terms) % 2:
                raise ValueError(f"{key} terms: pairs {a}, {b}")
            if terms:
                pairs = zip(terms[::2], terms[1::2], strict=True)
                plasticity[key] = [{a: x, b: y} for x, y in pairs]
        if p["norton"]:
            if len(p["norton"]) != 2:
                raise ValueError("Norton: K and n")
            plasticity["norton"] = {"K": p["norton"][0], "n": p["norton"][1]}
        spec["plasticity"] = plasticity
    elif kind == "crystal":
        cp = {"families": p["families"] or ["octahedral"], **{k: p[k] for k in PARAMS}}
        if p["interaction"]:
            cp["interaction"] = p["interaction"]
        spec["crystal_plasticity"] = cp
    return Material.from_config(spec)


def _mfront_readme(name, kinematics, symmetry, states, measure, values) -> str:
    finite = kinematics == "finite"
    orthotropic = symmetry == "orthotropic"
    axes = " + 6, the axes of the crystal" if finite and orthotropic else ""
    lines = [
        f"# MFront behaviour {name}",
        "",
        f"{name}.mfront: {kinematics} strains, {symmetry}, {states} internal state variables"
        + (f", the strain measure {measure}" if measure else "")
        + ". material.yaml: the material of morphoGen's configurations whose behaviour it is "
        "(here, of phase 1; its elastic constants for the linear problems and the reference "
        "media), which morphoGen integrates with its own kernels.",
        "",
        "## Compile",
        "",
        f"    mfront --obuild --interface=generic {name}.mfront   # MGIS",
        f"    mfront --obuild --interface=umat {name}.mfront      # AMITEX_FFTP, Cast3M",
        f"    mfront --obuild --interface=abaqus {name}.mfront    # Abaqus",
        f"    mfront --obuild --interface=aster {name}.mfront     # Code_Aster",
        "",
        "## Use",
        "",
        f"* morphoGen: the material {{mfront: {{source: {name}.mfront}}}} beside its elastic "
        "constants (MGIS integrates it, compiled on first use);",
        f"* umat (AMITEX_FFTP, Cast3M): the function {values['umat function']} of "
        "src/libUmatBehaviour.so, with the coefficients that the interface imposes (YOUN, NU, "
        "RHO, ALPH; orthotropic: YG1 ... G13, the axes V1, V2 of the crystal, RHO, ALP1 ... "
        f"ALP3) and {max(states, 1)} internal variables;",
        f"* Abaqus: the material {values['Abaqus material']} (*DEPVAR {states}{axes}; "
        "*USER MATERIAL, CONSTANTS=1: 0.) with the umat.cpp of TFEL (share/doc/mfront/abaqus) "
        "and src/libABAQUSBEHAVIOUR.so on LD_LIBRARY_PATH"
        + ("; the axes of each grain: *ORIENTATION" if orthotropic and not finite else "")
        + (": its steps need NLGEOM=YES" if finite else "")
        + ".",
    ]
    return "\n".join(lines) + "\n"


@tool(
    "zmat_import",
    "interop",
    "Z-set material files",
    "A material file of Z-mat (Z-set) translated: a gen_evp behaviour of one von Mises "
    "potential (Norton or rate-independent flow, the isotropic and kinematic hardenings of "
    "morphoGen) and the porous_plastic behaviours run on morphoGen's kernels; the other "
    "gen_evp behaviours (Hill's criterion, other flows and hardenings, several potentials) "
    "become an MFront behaviour of the brick StandardElastoViscoPlasticity; linear_elastic: "
    "any elasticity. With the material of the configurations, the MFront source and the "
    "coverage of Z-mat's library. The behaviour joins the workspace (B), for the other "
    "tools.",
    [
        Field(
            "text",
            "text",
            required=True,
            label="material file",
            doc="the text of a material file of Z-mat (***behavior gen_evp ... ***return)",
        ),
        Field("behaviour", "int", 0, label="behaviour number", doc="in the file, from 0"),
        Field(
            "native",
            "bool",
            True,
            label="morphoGen's kernels",
            doc="when they integrate it (otherwise MFront: for the exports to other codes)",
        ),
        Field("name", "str", "", doc="of the MFront behaviour (blank: ZmatBehaviour)"),
    ],
    order=6,
    menu="Behaviours of materials",
)
def zmat_import(p: dict, ctx) -> dict:
    import yaml

    from morphogen._utils import to_jsonable
    from morphogen.gui.api import behaviour_summary
    from morphogen.homogenization import zmat

    text = str(p["text"])
    behaviours = zmat.read_zmat(text)
    index = p["behaviour"]
    if not 0 <= index < len(behaviours):
        raise ValueError(f"behaviour number: the file has {len(behaviours)} (from 0)")
    b = behaviours[index]
    stem = zmat.mfront_name(p["name"] or "ZmatBehaviour")
    ctx.path("zmat.mat").write_text(text, encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        material = zmat.zmat_material(
            text, behaviour=index, name=stem, directory=tmp, native=p["native"]
        )
        config = to_jsonable(material.to_config())
        if material.mfront is not None:
            shutil.copyfile(material.mfront.source, ctx.path(f"{stem}.mfront"))
            config["mfront"]["source"] = f"{stem}.mfront"
    described = zmat.describe(text, index)
    porous = b.kind == "porous_plastic"
    if material.mfront is not None:
        translation = "mfront"
        why = described.get("reason") or "native: false, MFront for the exports to other codes"
    else:
        translation = "native"
        why = "morphoGen's kernels"
        if b.potentials and not porous:  # the brick of MFront, for the other codes
            ctx.path(f"{stem}.mfront").write_text(zmat.zmat_mfront_source(b, stem), "utf-8")
    kind = "elastic" if not b.potentials else ("porous" if porous else "plastic")
    material_text = yaml.safe_dump({"materials": {1: config}}, sort_keys=False)
    ctx.path("material.yaml").write_text(material_text, encoding="utf-8")
    ctx.table(_bricks(b), "bricks", f"the behaviour {b.kind} read")
    coverage = [
        {"category": a, "Z-mat": z, "morphoGen": s, "how": h} for a, z, s, h in zmat.COVERAGE
    ]
    ctx.table(coverage, "coverage", "coverage of Z-mat's library", shown=len(coverage))
    lines = [
        f"# The behaviour {index} ({b.kind}) of a material file of Z-mat",
        "",
        f"Translation: {translation} ({why}).",
        *[f"Note: {note}." for note in described.get("notes") or []],
        "",
        "material.yaml, the material of a configuration (here, of phase 1):",
        "",
        *[f"    {line}" for line in material_text.splitlines()],
    ]
    if ctx.path(f"{stem}.mfront").exists():
        what = (
            "its behaviour"
            if translation == "mfront"
            else "the same behaviour for the codes of MFront (AMITEX_FFTP, Abaqus, Code_Aster)"
        )
        lines += [
            "",
            f"{stem}.mfront: {what}, compiled by mfront --obuild --interface=generic (MGIS), "
            "umat, abaqus or aster.",
        ]
    readme = "\n".join(lines) + "\n"
    ctx.path("README.md").write_text(readme, encoding="utf-8")
    values = {
        "translation": translation,
        "kind": kind,
        "behaviours in the file": len(behaviours),
        "potentials": len(b.potentials),
    }
    # the behaviour joins the workspace (B), the run with it: for the other tools
    label, info = behaviour_summary(config)
    item = ctx.session.add(ctx.key, "behaviour", f"{label} (Z-set {b.kind} {index})",
                           material=config, info=info)  # fmt: skip
    out = {"values": values, "text": readme, "items": [ctx.session.public(item)]}
    return _finish(ctx, "zmat", out)


@tool(
    "zmat_export",
    "interop",
    "Z-set file of a behaviour",
    "A behaviour of the workspace (B: identified on the response of a morphology, kept by a "
    "material point, imported) written as a material file of Z-mat for Z-set: "
    "linear_elastic (isotropic, cubic or anisotropic elasticity), gen_evp of one potential "
    "of von Mises or Hill (a Norton or rate-independent flow, the isotropic hardening constant, "
    "linear, nonlinear, linear_nonlinear or nonlinear_sum, the kinematic terms linear or "
    "nonlinear) with its isotropic thermal strain, or porous_plastic (Gurson with its "
    "coalescence, Rousselier, the elliptic criterion, the nucleation, the initial "
    "porosity); the numbers in full, read back as the same material. Crystal plasticity, "
    "MFront behaviours: their MFront sources.",
    [
        Field(
            "behaviour",
            "item",
            items=("behaviour",),
            required=True,
            doc="identified, kept by a material point or imported",
        ),
        Field("name", "str", "", label="file name", doc="blank: behaviour.mat"),
        Field(
            "digits",
            "int",
            0,
            label="significant digits",
            doc="of the numbers (0: in full, read back exactly)",
        ),
    ],
    order=7,
    menu="Behaviours of materials",
)
def zmat_export(p: dict, ctx) -> dict:
    from morphogen.homogenization import zmat

    item = p["behaviour"]
    name = (p["name"] or "").strip().removesuffix(".mat")
    stem = re.sub(r"[^\w-]", "_", name).strip("_") or "behaviour"  # one name, in the folder
    digits = p["digits"] or None
    text = zmat.write_zmat(item["material"], ctx.path(f"{stem}.mat"), title=item["title"],
                           digits=digits)  # fmt: skip
    (behaviour,) = zmat.read_zmat(text)
    lines = [
        f"# {item['title']}: a material file of Z-mat",
        "",
        f"{stem}.mat: the behaviour {behaviour.kind} for Z-set, its numbers "
        + (f"to {digits} significant digits" if digits else "in full")
        + " (a material file, or its lines after ***material in an input file).",
        "",
        *[f"    {line}" for line in text.splitlines()],
    ]
    readme = "\n".join(lines) + "\n"
    ctx.path("README.md").write_text(readme, encoding="utf-8")
    values = {"behaviour": behaviour.kind, "potentials": len(behaviour.potentials)}
    return _finish(ctx, stem, {"values": values, "text": readme})


def _coefficients(values: dict) -> str:
    out = []
    for key, value in values.items():
        if isinstance(value, tuple) and value[:1] == ("table",):
            out.append(f"{key}: table of {value[1]} ({len(value[2])} rows)")
        elif isinstance(value, (int, float)):
            out.append(f"{key} {value:g}")
        else:
            out.append(f"{key} {value}")
    return ", ".join(out)


def _bricks(b) -> list[dict]:
    """The bricks of a behaviour of Z-mat as read, for a table."""
    rows = [
        {
            "brick": "elasticity",
            "type": b.elasticity[0],
            "coefficients": _coefficients(b.elasticity[1]),
        }
    ]
    if b.thermal_strain is not None:
        kind, coefficients = b.thermal_strain
        rows.append(
            {"brick": "thermal_strain", "type": kind, "coefficients": _coefficients(coefficients)}
        )
    for pot in b.potentials:
        bricks = [("criterion", pot.criterion), ("flow", pot.flow)]
        bricks += [("isotropic", x) for x in pot.isotropic]
        bricks += [("kinematic", x) for x in pot.kinematic]
        if pot.nucleation is not None:
            bricks.append(("nucleation", pot.nucleation[:2]))
        for brick, (kind, coefficients) in bricks:
            rows.append(
                {
                    "brick": f"potential {pot.name}: {brick}",
                    "type": kind,
                    "coefficients": _coefficients(coefficients),
                }
            )
    if b.modifier:
        rows.append({"brick": "modifier", "type": b.modifier, "coefficients": ""})
    return rows
