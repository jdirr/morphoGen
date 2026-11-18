"""The tools of the tab Explicit dynamics: the crushing, tension, shear and impact of a
morphology simulated by OpenRadioss, the explicit finite element solver
(:mod:`morphogen.dynamics`), its results read back from a run made elsewhere, and the
installation of OpenRadioss from its releases on GitHub.

The forms take engineering units, converted to the consistent system Mg, mm, s of the
inputs of OpenRadioss: moduli and stresses in MPa, densities in kg/m³, velocities in m/s,
strain rates in 1/s, durations in ms, the lengths of the morphology in the unit chosen.
The results come in MPa, J, ms and MJ/m³ (the absorbed energy per unit volume of the box)
and J/g (per unit mass). Each run keeps the inputs of OpenRadioss, a README of the
commands that rerun them elsewhere, the curves (``curves.csv``: the nominal strain and
stress of the morphology in morphoGen's components too, E33 and S33 for a compression
along z, so that the Comparisons tab draws them with loading paths), the energies, the
frames of its animations (shown in 3D, the last one first) and a video when VTK renders."""

from __future__ import annotations

import os
import shutil
import warnings
import zipfile
from pathlib import Path

import numpy as np

from morphogen.gui.tools import Field, tab, tool

tab(
    "dynamics",
    "Explicit dynamics",
    "Explicit dynamics with OpenRadioss, the open-source crash and impact solver: the "
    "crushing, tension, shear or impact of a morphology at high strain rates, on its voxels "
    "or a conforming mesh, its phases elastic, von Mises elastoplastic (with Norton's "
    "viscosity) or Johnson–Cook, with failure, and the self contact of its folding struts "
    "and closing cells; its nominal stress–strain curves, energies, plateau stress, "
    "densification strain and absorbed energy, and its fields over time in 3D. OpenRadioss "
    "is installed from its releases on GitHub by the tool OpenRadioss installation; a "
    "run made elsewhere (a cluster) is read back.",
    order=67,
)

#: The units of the lengths of the morphologies, by their size in mm
LENGTH_UNITS = (("mm", "mm"), ("um", "µm"), ("cm", "cm"), ("m", "m"))
TO_MM = {"mm": 1.0, "um": 1e-3, "cm": 10.0, "m": 1e3}
LOADINGS = (
    ("compression", "compression: the upper face driven toward the lower one"),
    ("tension", "tension: the upper face pulled away"),
    ("shear", "shear: the upper face driven across"),
    ("impact", "impact: the morphology thrown against a rigid wall"),
)
AXES = (("x", "x"), ("y", "y"), ("z", "z"))
MATERIALS_DOC = (
    "phase -> a material of the configurations in MPa and kg/m³, completing or replacing "
    'the fields above: {"1": {"johnson_cook": {"a": 300, "b": 450, "n": 0.6, "c": 0.01, '
    '"rate": 1}}} (Johnson–Cook), {"1": {"plasticity": {"yield": 250, "isotropic": '
    '[{"Q": 100, "b": 15}], "norton": {"K": 50, "n": 5}}}} (Voce, viscous), {"2": '
    '{"young": 3000, "poisson": 0.35, "density": 1200}}'
)
#: The rows of the curves shown in the page (the file holds them all)
SHOWN_ROWS = 25
#: Run on this machine with all the cores but one by default
THREADS = max(1, (os.cpu_count() or 2) - 1)

FIELDS = [
    Field("source", "item", label="morphology", required=True),
    Field("length_unit", "choice", "mm", label="unit of its lengths", choices=LENGTH_UNITS),
    Field(
        "mesh",
        "choice",
        "voxels",
        choices=(
            ("voxels", "voxels: 8-node bricks (robust, any morphology)"),
            ("conforming", "conforming tetrahedra of its level set (MMG)"),
        ),
    ),
    Field(
        "resolution",
        "int",
        24,
        label="voxels per edge",
        when={"mesh": "voxels"},
        doc="of its longest edge; the time step falls as the voxels shrink",
    ),
    Field(
        "element_size",
        "float",
        unit="length",
        when={"mesh": "conforming"},
        doc="the largest element, in the unit of the morphology (blank: 1/12 of the edge)",
    ),
    Field(
        "order",
        "choice",
        "1",
        when={"mesh": "conforming"},
        choices=(("1", "linear: TETRA4 (nodal pressures)"), ("2", "quadratic: TETRA10")),
    ),
    Field(
        "materials",
        "materials",
        of="source",
        default={"young": 70000, "poisson": 0.33},
        doc="Young's moduli in MPa (an aluminium: 70000, a steel: 210000)",
    ),
    Field("density", "float", 2700.0, unit="kg/m³", doc="of the solid phases"),
    Field(
        "yield_stress",
        "float",
        unit="MPa",
        label="yield stress",
        doc="of the solid phases, von Mises (blank: elastic)",
    ),
    Field("hardening", "float", 0.0, unit="MPa", doc="linear isotropic hardening modulus"),
    Field(
        "failure_strain",
        "float",
        label="failure strain",
        doc="plastic strain at which an element fails and is deleted (blank: none)",
    ),
    Field("behaviours", "json", label="materials of the phases", doc=MATERIALS_DOC),
    Field(
        "kept",
        "item",
        label="behaviour of the workspace",
        items=("behaviour",),
        none="none: the materials above",
        doc="a material point or an identification (B) in MPa, given to the phase below with "
        "the density above",
    ),
    Field("kept_phase", "int", 1, label="its phase"),
    Field("loading", "choice", "compression", choices=LOADINGS),
    Field("axis", "choice", "z", doc="normal to the loaded faces", choices=AXES),
    Field(
        "shear_axis",
        "choice",
        "x",
        label="shear along",
        when={"loading": "shear"},
        choices=AXES,
    ),
    Field(
        "strain",
        "float",
        0.3,
        label="final strain",
        doc="nominal (shear: engineering); impact: the share of the length travelled at "
        "the impact speed (the duration, unless given)",
    ),
    Field(
        "strain_rate",
        "float",
        500.0,
        unit="1/s",
        label="strain rate",
        when={"loading": ["compression", "tension", "shear"]},
        doc="nominal, after a smooth start; quasi-static below about 1/s, impacts above 100/s",
    ),
    Field("velocity", "float", 10.0, unit="m/s", label="impact speed", when={"loading": "impact"}),
    Field("duration", "float", unit="ms", doc="blank: the final strain at the rate or speed"),
    Field(
        "lateral",
        "choice",
        "fixed",
        label="loaded faces",
        when={"loading": ["compression", "tension"]},
        choices=(
            ("fixed", "held across the axis (bonded platens)"),
            ("free", "free to slide across the axis"),
        ),
    ),
    Field("contact", "bool", True, label="self contact", doc="folding struts, closing cells"),
    Field("friction", "float", 0.1, doc="Coulomb's coefficient of the contacts"),
    Field(
        "frames",
        "int",
        40,
        doc="animations written over the run (the curves of the faces "
        "from the time history, of 1000 samples)",
    ),
    Field(
        "time_step",
        "float",
        unit="µs",
        label="minimum time step",
        doc="kept by adding mass to the smallest elements (mass scaling: faster, but heavier "
        "nodes); blank: the stable time step",
    ),
    Field("run", "bool", True, label="run OpenRadioss here", doc="else write its inputs only"),
    Field("threads", "int", THREADS, label="OpenMP threads", doc="of the Engine"),
    Field("mpi", "int", 1, label="MPI processes", doc="above 1: mpiexec and the Engine with MPI"),
    Field(
        "installation",
        "str",
        "",
        label="folder of OpenRadioss",
        doc="blank: OPENRADIOSS_PATH, ~/morphogen/openradioss or the PATH",
        browse="folder",
    ),
    Field("name", "str", "", doc="of the run (blank: the model of the morphology)"),
]


def _materials(p: dict, item: dict) -> dict:
    """The materials of the phases in Mg, mm, s: the moduli of the form (MPa), the
    density (kg/m³ in Mg/mm³), the plasticity and failure of the solid phases, then the
    materials of the JSON field and the behaviour of the workspace."""
    density = float(p["density"]) * 1e-12
    if density <= 0:
        raise ValueError("density: > 0 kg/m³")
    base = {int(k): v for k, v in (p.get("materials") or {}).items()}
    if not base:  # no editor yet: the phases of the morphology, a void kept void
        phases = item.get("phases") or {"0": "void", "1": "solid"}
        base = {
            int(k): ("void" if v == "void" else {"young": 70000.0, "poisson": 0.33})
            for k, v in phases.items()
        }
    out: dict = {}
    for q, spec in base.items():
        if spec == "void":
            out[q] = "void"
            continue
        material = {
            "young": float(spec["young"]),
            "poisson": float(spec["poisson"]),
            "density": density,
        }
        if p.get("yield_stress"):
            material["plasticity"] = {
                "yield": float(p["yield_stress"]),
                "hardening": float(p.get("hardening") or 0.0),
            }
        if p.get("failure_strain") and "plasticity" in material:
            material["failure_strain"] = float(p["failure_strain"])
        out[q] = material
    extra = p.get("behaviours") or {}
    if not isinstance(extra, dict):
        raise ValueError('materials of the phases: a mapping phase -> material, {"1": {...}}')
    for key, value in extra.items():
        try:
            q = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"materials of the phases: phases are integers, got {key!r}") from None
        if isinstance(value, dict):
            value = dict(value)
            if "density" in value:
                value["density"] = float(value["density"]) * 1e-12
            base_q = out.get(q)
            if isinstance(base_q, dict) and not {"young", "poisson", "bulk", "shear"} & set(value):
                value = {**base_q, **value}
            value.setdefault("density", density)
            if "johnson_cook" in value:
                value.pop("plasticity", None)
        out[q] = value
    if p.get("kept"):
        q = int(p.get("kept_phase") or 1)
        material = dict(p["kept"]["material"])
        material.pop("expansion", None)
        material.setdefault("density", density)
        if "density" in p["kept"]["material"]:  # a behaviour of the workspace keeps its own
            material["density"] = float(p["kept"]["material"]["density"])
        out[q] = material
    return out


def _loading(p: dict):
    from morphogen.dynamics import ExplicitLoading

    duration = float(p["duration"]) * 1e-3 if p.get("duration") else None
    return ExplicitLoading(
        kind=p["loading"],
        axis=p["axis"],
        strain=float(p["strain"]),
        strain_rate=float(p.get("strain_rate") or 1.0),
        shear_axis=p.get("shear_axis") if p["loading"] == "shear" else None,
        velocity=float(p.get("velocity") or 0.0) * 1e3,  # m/s in mm/s
        duration=duration,
        lateral=p.get("lateral") or "fixed",
        contact=bool(p.get("contact")),
        friction=float(p.get("friction") or 0.0),
    )


def _scripts(folder: Path, name: str, threads: int) -> None:
    """The scripts that run the inputs of a folder elsewhere (bash and Windows)."""
    sh = folder / "run_openradioss.sh"
    sh.write_text(
        "#!/bin/bash\n"
        f"# OpenRadioss on the inputs of {name}: set OPENRADIOSS_PATH to its folder\n"
        "set -e\n"
        ': "${OPENRADIOSS_PATH:?set OPENRADIOSS_PATH to the folder of OpenRadioss}"\n'
        "export RAD_CFG_PATH=$OPENRADIOSS_PATH/hm_cfg_files\n"
        "export RAD_H3D_PATH=$OPENRADIOSS_PATH/extlib/h3d/lib/linux64\n"
        "export LD_LIBRARY_PATH=$OPENRADIOSS_PATH/extlib/hm_reader/linux64:"
        "$OPENRADIOSS_PATH/extlib/h3d/lib/linux64:$LD_LIBRARY_PATH\n"
        f"export OMP_STACKSIZE=400m OMP_NUM_THREADS=${{THREADS:-{threads}}}\n"
        'cd "$(dirname "$0")"\n'
        f"$OPENRADIOSS_PATH/exec/starter_linux64_gf -i {name}_0000.rad -np 1\n"
        f"$OPENRADIOSS_PATH/exec/engine_linux64_gf -i {name}_0001.rad -nt $OMP_NUM_THREADS\n"
        f"for f in {name}A[0-9][0-9][0-9]; do\n"
        '  $OPENRADIOSS_PATH/exec/anim_to_vtk_linux64_gf "$f" > "$f.vtk"\n'
        "done\n"
        f"$OPENRADIOSS_PATH/exec/th_to_csv_linux64_gf {name}T01\n",
        encoding="utf-8",
        newline="\n",
    )
    sh.chmod(0o755)
    bat = folder / "run_openradioss.bat"
    bat.write_text(
        "@echo off\r\n"
        f"rem OpenRadioss on the inputs of {name}: set OPENRADIOSS_PATH to its folder\r\n"
        "set RAD_CFG_PATH=%OPENRADIOSS_PATH%\\hm_cfg_files\r\n"
        "set RAD_H3D_PATH=%OPENRADIOSS_PATH%\\extlib\\h3d\\lib\\win64\r\n"
        "set PATH=%OPENRADIOSS_PATH%\\extlib\\hm_reader\\win64;"
        "%OPENRADIOSS_PATH%\\extlib\\intelOneAPI_runtime\\win64;%PATH%\r\n"
        f"set KMP_STACKSIZE=400m\r\nset OMP_NUM_THREADS={threads}\r\n"
        'cd /d "%~dp0"\r\n'
        f"%OPENRADIOSS_PATH%\\exec\\starter_win64.exe -i {name}_0000.rad -np 1\r\n"
        f"%OPENRADIOSS_PATH%\\exec\\engine_win64.exe -i {name}_0001.rad "
        "-nt %OMP_NUM_THREADS%\r\n"
        f"for %%f in ({name}A???) do "
        "%OPENRADIOSS_PATH%\\exec\\anim_to_vtk_win64.exe %%f > %%f.vtk\r\n"
        f"%OPENRADIOSS_PATH%\\exec\\th_to_csv_win64.exe {name}T01\r\n",
        encoding="utf-8",
        newline="",
    )


def _readme(manifest: dict, folder: Path, ran: bool) -> str:
    """The README of the inputs of a run."""
    from morphogen.dynamics import UnitSystem

    m = manifest
    name = m["name"]
    units = UnitSystem(*m["units"])
    loading = m["loading"]
    laws = ", ".join(f"phase {q}: {law}" for q, law in m["laws"].items())
    lines = [
        f"# OpenRadioss inputs: {m['title']}",
        "",
        f"* {name}_0000.rad: the Starter input, {m['n_nodes']} nodes, {m['n_elements']} "
        f"{m['elements']} elements (a part per phase), the materials ({laws}), the "
        f"loading ({loading['kind']} along {'xyz'[m['axis']]}), "
        + ("the self contact of the outer surface (TYPE24), " if loading.get("contact") else "")
        + "the time histories;",
        f"* {name}_0001.rad: the Engine input, to t = {m['end_time']:.6g} s, "
        f"{m['frames']} animations;",
        f"* {name}_mesh.npz and morphogen_radioss.json: what morphoGen reads the results "
        "back with.",
        "",
        f"Units: {units.describe()}. The stable time step is about "
        f"{m['critical_time_step']:.3g} s: about {m['cycles']:.3g} cycles"
        + (f" (mass scaling to {m['time_step']:.3g} s)." if m.get("time_step") else "."),
        "",
        "## Run",
        "",
        "With morphoGen (an installation found, or `morphogen radioss install`):",
        "",
        f"    morphogen radioss run {folder}",
        "",
        "or with the scripts of this folder (OPENRADIOSS_PATH set to the folder of OpenRadioss):",
        "",
        "    ./run_openradioss.sh          # Linux",
        "    run_openradioss.bat           # Windows",
        "",
        "then read the results back (the curves, energies and frames):",
        "",
        f"    morphogen radioss read {folder}",
        "",
        "or with the tool OpenRadioss results of the interface.",
    ]
    if ran:
        lines += [
            "",
            f"The run is in {folder} (the animations {name}A*, the time history "
            f"{name}T01 and their conversions, the listings {name}_0000.out and "
            f"{name}_0001.out).",
        ]
    return "\n".join(lines) + "\n"


def _archive(ctx, folder: Path, name: str) -> str:
    """An archive of the inputs of a run (not its outputs, which can be large)."""
    target = ctx.path(f"{name}_openradioss_inputs.zip")
    keep = (
        f"{name}_0000.rad",
        f"{name}_0001.rad",
        f"{name}_mesh.npz",
        "morphogen_radioss.json",
        "run_openradioss.sh",
        "run_openradioss.bat",
        "README.md",
    )
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in keep:
            if (folder / file).is_file():
                zf.write(folder / file, file)
    return ctx.url(target.name)


#: morphoGen's Voigt components of a normal and of a shear loading
_NORMAL = ("11", "22", "33")
_SHEAR = {frozenset((0, 1)): "12", frozenset((1, 2)): "23", frozenset((0, 2)): "31"}


def _report(ctx, result, *, title: str, extra: dict) -> dict:
    """The page of the results of a run: the summary (MPa, J, ms, MJ/m³, J/g), the curves
    and energies, their figures, the frames for the 3D view, a video."""
    from morphogen.dynamics import UnitSystem, summary, write_frames

    m = result.manifest
    units = UnitSystem(*m["units"])
    to_mpa = units.si("stress") / 1e6
    to_j = units.si("energy")
    to_ms = units.si("time") * 1e3
    values = dict(extra)
    s = summary(result)
    labels = {
        "end time": ("end time (ms)", to_ms),
        "stable time step (estimate)": ("stable time step, estimate (µs)", to_ms * 1e3),
        "mass": ("mass (g)", units.si("mass") * 1e3),
        "final stress": ("final stress (MPa)", to_mpa),
        "peak stress": ("peak stress (MPa)", to_mpa),
        "plateau stress (20-40 %)": ("plateau stress, 20-40 % (MPa)", to_mpa),
        "absorbed energy per volume": ("absorbed energy per volume (MJ/m³)", to_mpa),
        "absorbed energy per mass": (
            "absorbed energy per mass (J/g)",
            units.si("energy") / units.si("mass") * 1e-3,
        ),
        "peak force on the wall": ("peak force on the wall (N)", units.si("force")),
        "internal energy at the end": ("internal energy at the end (J)", to_j),
        "kinetic energy at the end": ("kinetic energy at the end (J)", to_j),
        "hourglass energy at the end": ("hourglass energy at the end (J)", to_j),
        "initial kinetic energy": ("initial kinetic energy (J)", to_j),
    }
    for key, value in s.items():
        label, factor = labels.get(key, (key, None))
        values[label] = value * factor if factor is not None and isinstance(value, float) else value
    values["forces from"] = result.source
    for note in result.warnings:
        warnings.warn(note, stacklevel=2)
    # the frames in a folder of their own, the last one beside the files for the 3D view
    frames = write_frames(result, ctx.path("frames"))
    if frames:
        shutil.copyfile(frames[-1], ctx.path("final_state.vtu"))
    c = result.curves
    tables = []
    if c:
        columns = {}
        for key, value in c.items():
            if key == "time":
                columns["time (ms)"] = np.asarray(value) * to_ms
            elif key.startswith("stress"):
                columns[f"{key} (MPa)"] = np.asarray(value) * to_mpa
            elif key.startswith("force"):
                columns[f"{key} (N)"] = np.asarray(value) * units.si("force")
            elif key == "displacement":
                columns["displacement (mm)"] = np.asarray(value) * units.si("length") * 1e3
            else:
                columns[key] = np.asarray(value)
        if "stress_upper" in c:  # morphoGen's components, for the Comparisons tab
            axis, drive = int(m["axis"]), int(m["driven"])
            comp = _NORMAL[axis] if axis == drive else _SHEAR[frozenset((axis, drive))]
            sign = float(m["sign"])
            columns[f"E{comp}"] = sign * np.asarray(c["strain"])
            columns[f"S{comp}"] = sign * np.asarray(c["stress_upper"]) * to_mpa
            columns["time"] = np.asarray(c["time"]) * to_ms
        tables.append(
            _table(
                ctx,
                columns,
                "curves",
                "curves of the run (nominal strain and stress "
                "of the faces; E and S in morphoGen's components, MPa)",
            )
        )
    if result.energies:
        columns = {
            ("time (ms)" if k == "time" else f"{k} (J)"): np.asarray(v)
            * (to_ms if k == "time" else to_j)
            for k, v in result.energies.items()
            if not k.startswith("momentum")
        }
        if len(columns) > 1:
            tables.append(_table(ctx, columns, "energies", "energies of the parts over time"))
    _figures(ctx, result, to_mpa, to_ms, to_j)
    _video(ctx, result, to_mpa, to_ms)
    values["frames"] = len(frames)
    return {"title": title, "values": values, "tables": tables}


def _table(ctx, columns: dict, name: str, title: str) -> dict:
    """A table of curves written whole as .csv, every n-th row shown in the page."""
    from morphogen.io.tables import write_table

    path = write_table(ctx.path(f"{name}.csv"), columns)
    n = len(next(iter(columns.values())))
    rows = np.unique(np.linspace(0, n - 1, min(n, SHOWN_ROWS)).round().astype(int)) if n else []
    names = list(columns)
    from morphogen.gui.tools import _cell

    return {
        "title": title,
        "columns": names,
        "rows": [[_cell(float(columns[k][i])) for k in names] for i in rows],
        "count": n,
        "url": ctx.url(path.name),
    }


def _figures(ctx, result, to_mpa: float, to_ms: float, to_j: float) -> None:
    try:
        from matplotlib.figure import Figure
    except ImportError:
        warnings.warn("no figures without matplotlib: pip install 'morphogen[plot]'", stacklevel=2)
        return
    c = result.curves
    if "stress_upper" in c:
        fig = Figure(figsize=(5.6, 3.9), layout="constrained")
        ax = fig.subplots()
        strain = np.asarray(c["strain"])
        ax.plot(
            strain,
            np.asarray(c["stress_upper"]) * to_mpa,
            lw=1.8,
            color="#2a78d6",
            label="driven face",
        )
        ax.plot(
            strain,
            np.asarray(c["stress_lower"]) * to_mpa,
            lw=1.2,
            ls="--",
            color="#eb6834",
            label="held face",
        )
        kind = result.manifest["loading"]["kind"]
        ax.set_xlabel("nominal shear strain" if kind == "shear" else "nominal strain")
        ax.set_ylabel("nominal stress (MPa)")
        ax.set_title(
            f"{kind} at {result.manifest['loading'].get('strain_rate', 0):g} /s", fontsize=10
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
        ctx.figure(fig, "stress_strain", "nominal stress–strain curves of the faces")
    elif "force_wall" in c:
        fig = Figure(figsize=(5.6, 3.9), layout="constrained")
        ax = fig.subplots()
        from morphogen.dynamics import UnitSystem

        units = UnitSystem(*result.manifest["units"])
        ax.plot(
            np.asarray(c["time"]) * to_ms,
            np.asarray(c["force_wall"]) * units.si("force"),
            lw=1.6,
            color="#2a78d6",
        )
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("force on the wall (N)")
        ax.grid(alpha=0.3)
        ctx.figure(fig, "wall_force", "force on the wall (the change of the momentum)")
    e = result.energies
    shown = [k for k in ("internal", "kinetic", "hourglass") if k in e]
    if shown:
        fig = Figure(figsize=(5.6, 3.6), layout="constrained")
        ax = fig.subplots()
        colours = {"internal": "#2a78d6", "kinetic": "#eb6834", "hourglass": "#1baf7a"}
        for k in shown:
            ax.plot(
                np.asarray(e["time"]) * to_ms,
                np.asarray(e[k]) * to_j,
                lw=1.5,
                color=colours[k],
                label=k,
            )
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("energy (J)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
        ctx.figure(fig, "energies", "energies of the parts")


def _video(ctx, result, to_mpa: float, to_ms: float) -> None:
    """The deformed morphology over time, coloured by its von Mises stress (VTK)."""
    if len(result.frames) < 2 or result.frames[0].displacement is None:
        return
    try:
        from morphogen.viz.video import history_frames, vtk_can_render

        if not vtk_can_render():
            return
        from morphogen.dynamics.results import frames_mesh

        mesh, owner = frames_mesh(result)
        fields = None
        if all("von_mises" in f.cells for f in result.frames):
            fields = [np.nan_to_num(f.cells["von_mises"][owner]) * to_mpa for f in result.frames]
        images = history_frames(
            mesh,
            [f.displacement for f in result.frames],
            fields,
            magnification=1.0,
            title="von Mises (MPa)" if fields is not None else "|u|",
            captions=[f"t = {f.time * to_ms:.3g} ms" for f in result.frames],
        )
        ctx.video(images, "deformation", "the morphology over time (von Mises stress)", mode="loop")
    except (ImportError, RuntimeError, ValueError) as exc:
        warnings.warn(f"no video of the run: {exc}", stacklevel=2)


@tool(
    "radioss_test",
    "dynamics",
    "Dynamic test (OpenRadioss)",
    "The crushing, tension, shear or impact of a morphology simulated by OpenRadioss: its "
    "mesh (voxels or conforming tetrahedra), the materials of its phases (elastic, von "
    "Mises elastoplastic with a linear, Voce or viscous hardening, Johnson–Cook; failure "
    "by element deletion), the loaded faces held and driven at the strain rate after a "
    "smooth start (or the morphology thrown against a rigid wall), the self contact of its "
    "outer surface; its nominal stress–strain curves (both faces: their gap tells whether "
    "the test is quasi-static), energies, peak and plateau stresses, densification strain, "
    "absorbed energy per volume and mass, its fields over time in 3D and a video. Without "
    "OpenRadioss here, the inputs and the scripts to run them elsewhere.",
    FIELDS,
    order=1,
    groups=(
        ("Specimen", "source"),
        ("Materials", "materials"),
        ("Loading", "loading"),
        ("Solver and outputs", "frames", True),
    ),
)
def radioss_test(p: dict, ctx) -> dict:
    from morphogen import dynamics

    item = p["source"]
    micro = item.get("micro")
    if micro is None:
        raise ValueError(f"{item['title']}: no microstructure in this session")
    installation = None
    if p["run"]:  # before the mesh: say at once that OpenRadioss is missing
        try:
            installation = dynamics.find_openradioss(p.get("installation") or None)
        except FileNotFoundError as exc:
            raise ValueError(
                f"{exc} — or uncheck 'run OpenRadioss here' to write its inputs"
            ) from None
    materials = _materials(p, item)
    loading = _loading(p)
    model = (item.get("model") or {}).get("type") or getattr(micro, "kind", "") or "morphology"
    folder = ctx.path("openradioss")
    threads = int(p.get("threads") or THREADS)
    step = float(p["time_step"]) * 1e-6 if p.get("time_step") else None
    test = dynamics.explicit_test(
        micro,
        materials,
        loading,
        folder,
        mesh=p["mesh"],
        resolution=int(p.get("resolution") or 24),
        element_size=p.get("element_size"),
        order=int(p.get("order") or 1),
        units=dynamics.UnitSystem("Mg", "mm", "s"),
        length_scale=TO_MM[p["length_unit"]],
        name=p.get("name") or model,
        frames=int(p.get("frames") or 40),
        time_step=step,
        run=bool(p["run"]),
        threads=threads,
        mpi=int(p.get("mpi") or 1),
        installation=installation,
        progress=lambda f, message: ctx.progress(0.02 + 0.9 * f, message),
        phase_names={int(k): v for k, v in (item.get("phases") or {}).items()},
    )
    manifest = test.inputs.manifest
    name = test.inputs.name
    _scripts(folder, name, threads)
    readme = _readme(manifest, folder, bool(p["run"]))
    (folder / "README.md").write_text(readme, encoding="utf-8")
    shutil.copyfile(folder / "README.md", ctx.path("README.md"))
    archive = _archive(ctx, folder, name)
    extra = {"OpenRadioss": installation.version or str(installation.root)} if installation else {}
    if not p["run"]:
        values = {
            "nodes": manifest["n_nodes"],
            "elements": f"{manifest['n_elements']} {manifest['elements']}",
            "end time (ms)": manifest["end_time"] * 1e3,
            "stable time step, estimate (µs)": manifest["critical_time_step"] * 1e6,
            "cycles (estimate)": round(manifest["cycles"]),
            "mass (g)": manifest["mass"] * 1e6,
        }
        return {
            "title": f"OpenRadioss inputs of {model}",
            "values": values,
            "text": readme,
            "archive": archive,
        }
    extra["run time (s)"] = round(test.run.elapsed, 1)
    out = _report(ctx, test.result, title=f"Dynamic {loading.kind} of {model}", extra=extra)
    out.update(text=readme, archive=archive)
    return out


@tool(
    "radioss_results",
    "dynamics",
    "OpenRadioss results",
    "The results of a run of OpenRadioss made elsewhere (a cluster, the scripts of a Dynamic "
    "test): the folder of its inputs written by morphoGen, with its animations and time "
    "history, converted here when they are not (anim_to_vtk, th_to_csv of the installation "
    "found); its curves, energies, summary and frames in 3D, as a Dynamic test shows them.",
    [
        Field(
            "directory",
            "str",
            required=True,
            label="folder of the run",
            doc="the folder with morphogen_radioss.json, the inputs and the outputs",
            browse="folder",
        ),
        Field("convert", "bool", True, label="convert the outputs", doc="with OpenRadioss's tools"),
        Field(
            "installation",
            "str",
            "",
            label="folder of OpenRadioss",
            browse="folder",
            doc="blank: found as for a Dynamic test",
        ),
    ],
    order=2,
)
def radioss_results(p: dict, ctx) -> dict:
    from morphogen import dynamics
    from morphogen.io.radioss import read_manifest

    folder = Path(str(p["directory"]).strip().strip('"')).expanduser()
    manifest = read_manifest(folder)
    if p["convert"]:
        try:
            install = dynamics.find_openradioss(p.get("installation") or None)
        except FileNotFoundError as exc:
            warnings.warn(f"outputs not converted: {exc}", stacklevel=1)
        else:
            run = dynamics.RadiossRun(folder, manifest["name"], [], None, {})
            dynamics.convert_outputs(run, install, progress=ctx.progress)
            for note in run.warnings:
                warnings.warn(note, stacklevel=1)
    result = dynamics.read_explicit_run(folder)
    if not result.curves and not result.frames:
        raise ValueError(
            f"no results in {folder}: run its inputs first ({manifest['name']}_0000.rad)"
        )
    return _report(
        ctx,
        result,
        title=f"OpenRadioss results of {manifest['title']}",
        extra={"folder": str(folder)},
    )


@tool(
    "radioss_install",
    "dynamics",
    "OpenRadioss installation",
    "The installation of OpenRadioss that the Dynamic tests run: the one found (its "
    "folder, executables and release), or a release downloaded from GitHub (the latest, or "
    "a tag) and unpacked into a folder (~/morphogen/openradioss by default), which the "
    "tests then find. Releases are published for Linux and Windows; elsewhere build "
    "OpenRadioss from its sources and set OPENRADIOSS_PATH. OpenRadioss is free software "
    "(AGPL-3.0) of the OpenRadioss project, run as a separate program.",
    [
        Field(
            "action",
            "choice",
            "check",
            choices=(
                ("check", "show the installation found"),
                ("install", "download and install a release from GitHub"),
            ),
        ),
        Field(
            "target",
            "str",
            "",
            label="folder",
            when={"action": "install"},
            browse="folder",
            doc="blank: ~/morphogen/openradioss (replaced)",
        ),
        Field(
            "tag",
            "str",
            "",
            label="release",
            when={"action": "install"},
            doc="the tag of a release (blank: the latest)",
        ),
        Field(
            "installation",
            "str",
            "",
            label="folder of OpenRadioss",
            when={"action": "check"},
            browse="folder",
            doc="blank: OPENRADIOSS_PATH, ~/morphogen/openradioss or the PATH",
        ),
    ],
    order=3,
)
def radioss_install(p: dict, ctx) -> dict:
    from morphogen import dynamics

    if p["action"] == "install":
        install = dynamics.install_openradioss(
            p.get("target") or None, tag=p.get("tag") or None, progress=ctx.progress
        )
        title = f"OpenRadioss {install.version or ''} installed"
    else:
        try:
            install = dynamics.find_openradioss(p.get("installation") or None)
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from None
        title = "OpenRadioss found"
    return {"title": title, "values": install.describe()}
