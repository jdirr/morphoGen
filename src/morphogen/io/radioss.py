"""Inputs and outputs of OpenRadioss, the explicit finite element solver (Radioss Block
Format of the input version 2022, which OpenRadioss reads; Radioss Reference Guide).

* :func:`write_radioss` writes the Starter input ``<name>_0000.rad`` of a mesh: its unit
  system (``/BEGIN``), its nodes, its elements by part (``/BRICK`` for the voxels,
  ``/TETRA4`` or ``/TETRA10`` for conforming tetrahedra; a part per phase), the solid
  property (``/PROP/SOLID``: the stabilized 8-node brick, Isolid 24; the linear
  tetrahedra with nodal pressures against locking), the materials of the phases
  (``/MAT/LAW1`` elastic, ``/MAT/LAW36`` von Mises with tabulated yield curves,
  ``/MAT/LAW2`` Johnson–Cook), the node groups of the loaded faces, the loading
  (``/BCS``, ``/IMPVEL`` and its ``/FUNCT``; ``/INIVEL`` and a fixed ``/RWALL`` for an
  impact), the self contact of the outer surface (``/INTER/TYPE24``) and the time
  histories (``/TH/NODE`` of the reactions of the loaded faces, ``/TH/PART`` of the
  energies and momenta); and the Engine input ``<name>_0001.rad`` (``/RUN``,
  ``/TFILE``, ``/ANIM``: displacements, velocities, reaction and contact forces, von
  Mises stresses, plastic strains, energy densities, pressures and the status of the
  elements; ``/DT/NODA/CST`` for a mass scaling). It writes too the mesh and the
  manifest that morphoGen reads the results back with (:mod:`morphogen.dynamics`).
* :func:`read_radioss_listing` reads the listing of a run (``<name>_0000.out``,
  ``<name>_0001.out``): its cycles and times (the progress of a run), its errors and
  whether it ended normally.
* :func:`read_radioss_th` reads a time history converted to CSV by OpenRadioss's
  ``th_to_csv``; the animations converted by ``anim_to_vtk`` are legacy VTK files
  (:func:`morphogen.io.vtk.read_vtk_grid`).

The fields of the cards are those of the reference guide: integers of 10 characters,
reals of 20, comments after ``#``. The 4-node tetrahedra are written with their natural
coordinates r, s, t running from node 4 toward nodes 1, 2, 3 (positive Jacobian,
``det[x1 - x4, x2 - x4, x3 - x4] > 0``), the 10-node ones with their mid-side nodes 5 to
10 on the edges 1-2, 2-3, 3-1, 1-4, 2-4 and 3-4 (morphoGen's and VTK's order), the bricks
with their nodes 1-4 on a face whose normal (right-hand rule) points toward nodes 5-8."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from morphogen._version import __version__

#: The input version of the cards (``/BEGIN``)
INPUT_VERSION = 2022
#: The solid property: the 8-node brick with physical stabilization (Isolid 24), the
#: small strains and the constant pressure chosen by Radioss for the law (-1), the
#: linear tetrahedra with nodal pressures against volumetric locking (Itetra4 3), the
#: quadratic ones by default
SOLID_PROPERTY = {
    "Isolid": 24,
    "Ismstr": -1,
    "Icpre": -1,
    "Itetra10": 0,
    "Itetra4": 3,
    "Iframe": -1,
}
#: The animated results of the Engine input
ANIM_VECTORS = ("DISP", "VEL", "FREAC", "CONT")
ANIM_ELEMENTS = ("VONM", "EPSP", "ENER", "P", "OFF")
#: The kinds of elements: the keyword of their block and their nodes
ELEMENTS = {"brick": ("BRICK", 8), "tetra4": ("TETRA4", 4), "tetra10": ("TETRA10", 10)}
#: The order of the nodes of a 10-node tetrahedron whose corners 1 and 2 are swapped
_SWAP10 = np.array([1, 0, 2, 3, 4, 6, 5, 8, 7, 9])
#: The values (nodes times variables times samples) of the time histories, at most: a
#: mesh with large loaded faces is sampled less often (its forces are also drawn from
#: the reaction forces of the animations)
MAX_HISTORY_VALUES = 20_000_000
#: The ruler of the columns of the cards (a comment)
RULER = "#---1----|" + "".join(f"----{k}----|" for k in range(2, 10)) + "---10----|"
MANIFEST = "morphogen_radioss.json"


def _i(value) -> str:
    """An integer field (10 characters)."""
    text = f"{int(value):10d}"
    if len(text) > 10:
        raise ValueError(f"{value}: more than 10 digits")
    return text


def _f(value) -> str:
    """A real field (20 characters, 12 significant digits)."""
    return f"{float(value):20.12g}"


def _ascii(text: str) -> str:
    """A text in ASCII, as the cards are read (accents dropped, ³ as 3, µ as u)."""
    text = unicodedata.normalize("NFKD", str(text)).replace("μ", "u").replace("µ", "u")
    return text.encode("ascii", "ignore").decode("ascii")


def _title(text: str) -> str:
    return re.sub(r"\s+", " ", _ascii(text)).strip()[:100] or "morphoGen"


def run_name(text: str) -> str:
    """A name for the files of a run: letters, digits, '_' and '-', 4 characters at least
    (the Engine's ``/RUN`` asks for 4 to 80)."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text or "")).strip("_") or "morphogen"
    if len(name) < 4:
        name = f"{name}_run"
    return name[:60]


def _ids(values, per_line: int = 10) -> list[str]:
    values = [int(v) for v in values]
    return [
        "".join(_i(v) for v in values[k : k + per_line]) for k in range(0, len(values), per_line)
    ]


def element_kind(mesh) -> str:
    """``brick``, ``tetra4`` or ``tetra10``: the elements of a voxel or tetrahedral mesh."""
    n = np.asarray(mesh.elements).shape[1]
    kinds = {8: "brick", 4: "tetra4", 10: "tetra10"}
    if n not in kinds:
        raise ValueError(f"meshes of 8-node hexahedra or 4- or 10-node tetrahedra, got {n} nodes")
    return kinds[n]


def oriented(mesh) -> np.ndarray:
    """The elements of a mesh (0-based nodes) in the orientation of Radioss: the
    tetrahedra whose ``det[x1 - x4, x2 - x4, x3 - x4]`` is negative get their corners 1
    and 2 swapped (and their mid-side nodes with them); the hexahedra of voxel meshes
    are kept (morphoGen's order is Radioss's)."""
    elements = np.array(mesh.elements, dtype=np.int64, copy=True)
    if elements.shape[1] == 8:
        return elements
    x = np.asarray(mesh.nodes, dtype=float)[elements[:, :4]]
    det = np.einsum("ij,ij->i", x[:, 0] - x[:, 3], np.cross(x[:, 1] - x[:, 3], x[:, 2] - x[:, 3]))
    flip = det < 0
    if elements.shape[1] == 4:
        elements[flip, :2] = elements[flip, 1::-1]
    else:
        elements[flip] = elements[flip][:, _SWAP10]
    return elements


def box_of(mesh) -> tuple[np.ndarray, np.ndarray]:
    """The box of a mesh: that of a tetrahedral mesh, the extent of the nodes of a voxel
    mesh."""
    if hasattr(mesh, "lower") and hasattr(mesh, "upper"):
        return np.asarray(mesh.lower, dtype=float), np.asarray(mesh.upper, dtype=float)
    nodes = np.asarray(mesh.nodes, dtype=float)
    return nodes.min(axis=0), nodes.max(axis=0)


def face_nodes(mesh, axis: int, side: int) -> np.ndarray:
    """The nodes (0-based) on the face of the box of a mesh normal to ``axis``, lower
    (``side`` 0) or upper (1)."""
    if getattr(mesh, "whole", False):
        raise ValueError(
            "a mesh of whole periodic cells has no faces of the box to load: mesh the box"
        )
    sets = getattr(mesh, "node_sets", None)
    name = f"{'xyz'[axis]}{side}"
    if sets is not None and name in sets:
        return np.asarray(sets[name], dtype=np.int64)
    lower, upper = box_of(mesh)
    target = lower[axis] if side == 0 else upper[axis]
    tol = 1e-9 * float(np.max(upper - lower))
    return np.flatnonzero(np.abs(np.asarray(mesh.nodes)[:, axis] - target) <= tol)


def element_volumes(mesh) -> np.ndarray:
    """The volume of each element: the voxels', or that of the corners of the
    tetrahedra (the straight ones)."""
    elements = np.asarray(mesh.elements)
    if elements.shape[1] == 8:
        return np.full(len(elements), float(np.prod(mesh.spacing)))
    x = np.asarray(mesh.nodes, dtype=float)[elements[:, :4]]
    return (
        np.abs(
            np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])
        )
        / 6.0
    )


@dataclass
class RadiossInputs:
    """The inputs of a run of OpenRadioss written by :func:`write_radioss`: their
    ``directory``, the run ``name``, the Starter and Engine inputs, the manifest that
    the results are read back with (its contents in ``manifest``)."""

    directory: Path
    name: str
    starter: Path
    engine: Path
    manifest: dict = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return float(self.manifest["end_time"])


def _material_cards(
    mat_id: int, title: str, material, funct_start: int
) -> tuple[list[str], list[str], int]:
    """The cards of a material (and of the functions of its yield curves, numbered from
    ``funct_start``): the lines of the material, those of its functions, the next
    number of function."""
    e, nu, rho = material.young, material.poisson, material.density
    eps_max = material.failure_strain or 0.0  # 0: Radioss's default, no failure
    law = material.law
    head = [f"/MAT/{law}/{mat_id}", _title(title), "#              RHO_I", _f(rho)]
    functs: list[str] = []
    if law == "LAW1":
        return (
            [*head, "#                  E                  nu", _f(e) + _f(nu)],
            functs,
            funct_start,
        )
    if law == "LAW2":
        jc = material.johnson_cook
        lines = [
            *head,
            "#                  E                  Nu     Iflag",
            _f(e) + _f(nu) + _i(0),
            "#                  a                   b         "
            "          n             EPS_max            SIG_max0",
            _f(jc.a) + _f(jc.b) + _f(jc.n) + _f(eps_max) + _f(jc.max_stress or 0.0),
            "#                  c           EPS_DOT_0      "
            " ICC   Fsmooth               F_cut               Chard",
            _f(jc.c) + _f(jc.rate if jc.c > 0 else 0.0) + _i(0) + _i(0) + _f(0.0) + _f(0.0),
            "#                  m              T_melt              rhoC_p                 T_r",
            _f(0.0) * 4,
        ]
        return lines, functs, funct_start
    rates = material.rates()
    ids = list(range(funct_start, funct_start + len(rates)))
    strains = np.asarray(material_curve_strains(), dtype=float)
    for fid, rate in zip(ids, rates, strict=True):
        stress = material.yield_curve(strains, rate)
        functs += [
            f"/FUNCT/{fid}",
            _title(f"{title}: yield stress at the plastic strain rate {rate:g}"),
            "#                  X                   Y",
        ]
        functs += [_f(p) + _f(s) for p, s in zip(strains, stress, strict=True)]
    viscous = len(rates) > 1
    lines = [
        *head,
        "#                  E                  Nu         "
        "  Eps_p_max               Eps_t               Eps_m",
        _f(e) + _f(nu) + _f(eps_max) + _f(0.0) + _f(0.0),
        "#  N_funct  F_smooth              C_hard         "
        "      F_cut               Eps_f                  VP",
        _i(len(rates)) + _i(0) + _f(0.0) + _f(0.0) + _f(0.0) + " " * 10 + _i(1 if viscous else 0),
        "#  fct_IDp              Fscale   Fct_IDE                EInf                  CE",
        _i(0) + _f(0.0) + _i(0) + _f(0.0) + _f(0.0),
        "# func_ID1  func_ID2  func_ID3  func_ID4  func_ID5",
        *_ids(ids, 5),
        "#           Fscale_1            Fscale_2         "
        "   Fscale_3            Fscale_4            Fscale_5",
        *["".join(_f(1.0) for _ in ids[k : k + 5]) for k in range(0, len(ids), 5)],
        "#        Eps_dot_1          Eps_dot_2          "
        " Eps_dot_3           Eps_dot_4           Eps_dot_5",
        *["".join(_f(r) for r in rates[k : k + 5]) for k in range(0, len(rates), 5)],
    ]
    return lines, functs, funct_start + len(rates)


def material_curve_strains() -> tuple[float, ...]:
    """The plastic strains of the tabulated yield curves (:data:`morphogen.dynamics.
    model.CURVE_STRAINS`)."""
    from morphogen.dynamics.model import CURVE_STRAINS

    return CURVE_STRAINS


def write_radioss(
    directory,
    mesh,
    materials: dict,
    loading,
    *,
    name: str = "morphogen",
    units=None,
    frames: int = 50,
    history: int = 1000,
    time_step: float | None = None,
    length_scale: float = 1.0,
    phase_names: dict | None = None,
    title: str = "",
) -> RadiossInputs:
    """Write the inputs of a run of OpenRadioss for a mesh in ``directory``.

    ``mesh``: a :class:`~morphogen.mesh.voxel.VoxelMesh` (bricks) or a
    :class:`~morphogen.mesh.tetra.TetMesh` (4- or 10-node tetrahedra) of the box of a
    morphology; ``materials``: phase -> :class:`~morphogen.dynamics.ExplicitMaterial`
    of each phase of the mesh; ``loading``: an :class:`~morphogen.dynamics.
    ExplicitLoading`; ``units``: the :class:`~morphogen.dynamics.UnitSystem` of every
    value (Mg, mm, s by default). ``frames``: the animations written over the run (and
    the initial one), ``history``: the samples of the time histories; ``time_step``: a
    minimum time step kept by adding mass to the nodes (``/DT/NODA/CST``; none by
    default, the run then going at the stable time step of its smallest element).
    ``length_scale``: the length unit of the system per length unit of the mesh (1000 for
    a mesh in metres written in mm). ``phase_names``: the names of the phases, for the
    titles of the parts.

    Returns the :class:`RadiossInputs`; besides the two inputs, the directory holds
    ``<name>_mesh.npz`` (the mesh and the nodes of its loaded faces) and the manifest
    ``morphogen_radioss.json`` that :func:`morphogen.dynamics.read_explicit_run` reads
    the results back with.

    The loaded faces: ``lateral`` ``fixed`` holds the lower face and the upper one
    across the drive (a compression between bonded platens), ``free`` holds the lower
    face along the axis only and one of its nodes, the nearest to its centre, across it
    too (no rigid drift); a shear holds the lower face and the upper one but along the
    drive. The reactions of the nodes of the loaded faces along the axis and the drive
    are written to the time history, sampled less often when the faces are large
    (:data:`MAX_HISTORY_VALUES`).
    """
    from morphogen.dynamics.model import AXES, ExplicitLoading, UnitSystem, critical_time_step

    units = UnitSystem.parse(units)
    loading = ExplicitLoading.from_config(loading)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = run_name(name)
    kind = element_kind(mesh)
    keyword, per = ELEMENTS[kind]
    scale = float(length_scale)
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"length_scale: > 0, got {length_scale!r}")
    nodes = np.asarray(mesh.nodes, dtype=float) * scale
    elements = oriented(mesh)
    phase = np.asarray(mesh.phase, dtype=np.int64)
    present = [int(q) for q in np.unique(phase)]
    missing = [q for q in present if materials.get(q) is None]
    if missing:
        raise ValueError(f"phases {missing} of the mesh: no material (voids are not meshed)")
    if int(frames) < 1 or int(history) < 2:
        raise ValueError("frames >= 1 and history >= 2")
    names = {int(k): str(v) for k, v in (phase_names or {}).items()}
    lower, upper = box_of(mesh)
    lower, upper = lower * scale, upper * scale
    lengths = upper - lower
    axis = loading.axis
    end = loading.end_time(lengths)
    speed = loading.speed(lengths)
    volumes = element_volumes(mesh) * scale**3
    density = np.array([materials[int(q)].density for q in phase])
    mass = float(np.sum(volumes * density))
    dt = critical_time_step(mesh, materials) * scale
    n_nodes = len(nodes)
    lines = [
        "#RADIOSS STARTER",
        f"# morphoGen {__version__}: {_title(title or name)}",
        _ascii(
            f"# units {' '.join(units.names)}: stresses in {units.unit('stress')}, densities in "
            f"{units.unit('density')}, velocities in {units.unit('velocity')}"
        ),
        _ascii(f"#   forces in {units.unit('force')}, energies in {units.unit('energy')}"),
        RULER,
        "/BEGIN",
        name,
        _i(INPUT_VERSION) + _i(0),
        "".join(f"{u:>20s}" for u in units.names),
        "".join(f"{u:>20s}" for u in units.names),
        RULER,
        "/NODE",
        "#  node_ID                  Xc                  Yc                  Zc",
    ]
    ids = np.arange(1, n_nodes + 1)
    lines += [
        f"{i:10d}{x:20.12g}{y:20.12g}{z:20.12g}" for i, (x, y, z) in zip(ids, nodes, strict=True)
    ]
    # parts: one per phase (an element type per mesh), numbered like the materials
    parts: dict[int, int] = {}
    order = np.argsort(phase, kind="stable")
    numbers = np.empty(len(phase), dtype=np.int64)
    numbers[order] = np.arange(1, len(phase) + 1)
    funct = 1
    material_lines: list[str] = []
    function_lines: list[str] = []
    for k, q in enumerate(present, start=1):
        parts[q] = k
        label = f"phase {q}" + (f" ({names[q]})" if q in names else "")
        lines += [
            RULER,
            f"/PART/{k}",
            _title(label),
            "#  prop_ID    mat_ID subset_ID",
            _i(1) + _i(k) + _i(0),
        ]
        own = np.flatnonzero(phase == q)
        lines.append(f"/{keyword}/{k}")
        if kind == "tetra10":
            lines.append("#tetra_ID")
            for e in own:
                lines.append(_i(numbers[e]))
                lines.append("".join(_i(n + 1) for n in elements[e]))
        else:
            lines.append(
                f"#{'elem_ID':>9s}" + "".join(f"{'node_ID' + str(j + 1):>10s}" for j in range(per))
            )
            for e in own:
                lines.append(_i(numbers[e]) + "".join(_i(n + 1) for n in elements[e]))
        cards, functs, funct = _material_cards(k, label, materials[q], funct)
        material_lines += [RULER, *cards]
        function_lines += functs
    prop = SOLID_PROPERTY
    lines += [
        RULER,
        "/PROP/SOLID/1",
        "solid",
        "#   Isolid    Ismstr      Iale     Icpre "
        " Itetra10     Inpts   Itetra4    Iframe                  dn",
        _i(prop["Isolid"])
        + _i(prop["Ismstr"])
        + " " * 10
        + _i(prop["Icpre"])
        + _i(prop["Itetra10"])
        + _i(0)
        + _i(prop["Itetra4"])
        + _i(prop["Iframe"])
        + _f(0.0),
        "#                 qa                  qb         "
        "          h            LAMBDA_V                MU_V",
        _f(0.0) * 5,
        "#         deltaT_min",
        _f(0.0),
        "#     Ndir sphpartID",
        _i(0) + _i(0),
    ]
    lines += material_lines
    if function_lines:
        lines += [RULER, *function_lines]
    # node groups, the loading, the contact and the time histories
    lower_nodes = face_nodes(mesh, axis, 0)
    upper_nodes = face_nodes(mesh, axis, 1)
    if loading.kind != "impact" and (not len(lower_nodes) or not len(upper_nodes)):
        raise ValueError(
            f"no solid on the {'lower' if not len(lower_nodes) else 'upper'} face normal to "
            f"{AXES[axis]}: nothing to hold or drive (another axis, or a morphology that "
            "spans the box)"
        )
    groups: dict[str, int] = {}

    def group(title_: str, node_ids) -> int:
        gid = len(groups) + 1
        groups[title_] = gid
        lines.extend([RULER, f"/GRNOD/NODE/{gid}", _title(title_), *_ids(np.asarray(node_ids) + 1)])
        return gid

    histories: list[str] = []
    th = 0

    def node_history(title_: str, node_ids, variables, prefix: str) -> None:
        nonlocal th
        th += 1
        histories.extend(
            [RULER, f"/TH/NODE/{th}", _title(title_), "".join(f"{v:>10s}" for v in variables)]
        )
        histories.extend(_i(n + 1) + _i(0) + f"{prefix}{n + 1}" for n in node_ids)

    drive = loading.driven
    sign = loading.sign()
    wall = None
    if loading.kind == "impact":
        gid = len(groups) + 1
        groups["all nodes"] = gid
        lines += [RULER, f"/GRNOD/PART/{gid}", "all nodes", *_ids(parts.values())]
        velocity = np.zeros(3)
        velocity[axis] = sign * speed
        lines += [
            RULER,
            "/INIVEL/TRA/1",
            "impact velocity",
            "#                 Vx                  Vy                  Vz  grnod_ID   skew_ID",
            _f(velocity[0]) + _f(velocity[1]) + _f(velocity[2]) + _i(gid) + _i(0),
        ]
        center = 0.5 * (lower + upper)
        point = center.copy()
        point[axis] = lower[axis]
        normal = np.zeros(3)
        normal[axis] = 1.0
        slide = 2 if loading.friction > 0 else 0
        wall = {"point": point.tolist(), "normal": normal.tolist()}
        lines += [
            RULER,
            "/RWALL/PLANE/1",
            "fixed wall at the lower face",
            "#  node_ID     Slide  grnod_ID1  grnod_ID2",
            _i(0) + _i(slide) + _i(0) + _i(0),
            "#            Dsearch                fric         "
            "   Diameter                ffac       ifq",
            _f(1.5 * float(np.max(lengths))) + _f(loading.friction) + _f(0.0) + _f(0.0) + _i(0),
            "#                 XM                  YM                  ZM",
            _f(point[0]) + _f(point[1]) + _f(point[2]),
            "#                X_M1                Y_M1                Z_M1",
            _f(point[0] + normal[0]) + _f(point[1] + normal[1]) + _f(point[2] + normal[2]),
        ]
        th += 1
        histories += [RULER, f"/TH/RWALL/{th}", "wall", f"{'DEF':>10s}", _i(1)]
    else:
        free = loading.kind != "shear" and loading.lateral == "free"
        held_nodes, anchor = lower_nodes, None
        if free:  # one node of the lower face held across too: no rigid drift
            face = np.asarray(nodes[lower_nodes], dtype=float)
            anchor = int(lower_nodes[np.argmin(np.linalg.norm(face - face.mean(axis=0), axis=1))])
            held_nodes = lower_nodes[lower_nodes != anchor]
        g_lower = group("lower face", held_nodes) if len(held_nodes) else None
        g_upper = group("upper face", upper_nodes)
        held = np.ones(3, dtype=int)
        if free:
            held[:] = 0
            held[axis] = 1
        upper_held = np.zeros(3, dtype=int) if free else np.ones(3, dtype=int)
        upper_held[drive] = 0
        bcs = []
        if g_lower is not None:
            bcs.append(("lower face held", held, g_lower))
        if anchor is not None:
            bcs.append(
                ("a node of the lower face held", np.ones(3, dtype=int), group("anchor", [anchor]))
            )
        if upper_held.any():
            bcs.append(("upper face held across the drive", upper_held, g_upper))
        for k, (text, dofs, gid) in enumerate(bcs, start=1):
            lines += [
                RULER,
                f"/BCS/{k}",
                text,
                "#  Tra rot   skew_ID  grnod_ID",
                f"   {dofs[0]}{dofs[1]}{dofs[2]} 000" + _i(0) + _i(gid),
            ]
        table = loading.velocity_function(end)
        lines += [
            RULER,
            "/FUNCT/" + str(funct),
            "velocity of the upper face (share of its speed)",
            "#                  X                   Y",
        ]
        lines += [_f(t) + _f(s) for t, s in table]
        lines += [
            RULER,
            "/IMPVEL/1",
            "upper face driven",
            "#funct_IDT       Dir   skew_ID   sens_ID  grnod_ID  frame_ID     Icoor",
            _i(funct) + f"{'XYZ'[drive]:>10s}" + _i(0) + _i(0) + _i(g_upper) + _i(0) + _i(0),
            "#           Ascale_x            Fscale_Y              Tstart               Tstop",
            _f(1.0) + _f(sign * speed) + _f(0.0) + _f(10.0 * end),
        ]
        reactions = tuple(f"REAC{'XYZ'[a]}" for a in dict.fromkeys((axis, drive)))
        node_history("reactions of the lower face", lower_nodes, reactions, "L")
        node_history("reactions of the upper face", upper_nodes, reactions, "U")
        values = len(reactions) * (len(lower_nodes) + len(upper_nodes))
        history = int(max(100, min(int(history), MAX_HISTORY_VALUES // max(values, 1))))
    if loading.contact:
        lines += [
            RULER,
            "/SURF/PART/EXT/1",
            "outer surface of the solid",
            *_ids(parts.values()),
            RULER,
            "/INTER/TYPE24/1",
            "self contact of the outer surface",
            "# Surf_ID1  Surf_ID2      Istf                   "
            "            Irem_i2                Idel",
            _i(1) + _i(0) + _i(0) + " " * 20 + _i(0) + " " * 10 + _i(0),
            "#grnod_IDs                         Iedge         "
            " Edge_angle           Gap_max_s           Gap_max_m",
            _i(0) + " " * 20 + _i(0) + _f(0.0) + _f(0.0) + _f(0.0),
            "#              Stmin               Stmax     Igap0     Ipen0            Ipen_max",
            _f(0.0) + _f(0.0) + _i(0) + _i(0) + _f(0.0),
            "#              Stfac                Fric         "
            "                         Tstart               Tstop",
            _f(0.0) + _f(loading.friction) + " " * 20 + _f(0.0) + _f(10.0 * end),
            "#      IBC                    Inacti             "
            "   VISs                                   Tpressfit",
            " " * 30 + _i(0) + _f(0.0) + " " * 20 + _f(0.0),
            "#    Ifric    Ifiltr               Xfreq         "
            "    sens_ID                              fric_ID",
            _i(0) + _i(0) + _f(0.0) + " " * 10 + _i(0) + " " * 30 + _i(0),
        ]
    th += 1
    histories += [RULER, f"/TH/PART/{th}", "parts", f"{'DEF':>10s}", *_ids(parts.values())]
    lines += histories
    lines += [RULER, "/END"]
    starter = directory / f"{name}_0000.rad"
    starter.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    # the Engine input
    cycles = end / dt
    every = max(1, round(min(cycles, 1e9) / 200))
    engine_lines = [
        "#RADIOSS ENGINE",
        f"# morphoGen {__version__}: {_title(title or name)}",
        RULER,
        f"/RUN/{name}/1",
        _f(end),
        "/TFILE",
        _f(end / int(history)),
        "/ANIM/DT",
        _f(0.0) + _f(end / int(frames)),
        *[f"/ANIM/VECT/{v}" for v in ANIM_VECTORS],
        *[f"/ANIM/ELEM/{v}" for v in ANIM_ELEMENTS],
        f"/PRINT/-{every}",
    ]
    if time_step:
        engine_lines += [
            "/DT/NODA/CST/0",
            "#            DT_scale              DT_min",
            _f(0.9) + _f(float(time_step)),
        ]
    engine_lines.append(RULER)
    engine = directory / f"{name}_0001.rad"
    engine.write_text("\n".join(engine_lines) + "\n", encoding="ascii", newline="\n")
    np.savez_compressed(
        directory / f"{name}_mesh.npz",
        nodes=nodes,
        elements=elements,
        phase=phase,
        numbers=numbers,
        lower_nodes=lower_nodes,
        upper_nodes=upper_nodes,
        kind=np.array(kind),
    )
    area = float(np.prod(np.delete(lengths, axis)))
    manifest = {
        "morphogen": __version__,
        "name": name,
        "title": _title(title or name),
        "units": list(units.names),
        "loading": loading.to_config(),
        "axis": axis,
        "driven": drive,
        "sign": sign,
        "speed": speed,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "lengths": lengths.tolist(),
        "area": area,
        "volume": float(np.prod(lengths)),
        "solid_volume": float(volumes.sum()),
        "mass": mass,
        "end_time": end,
        "frames": int(frames),
        "history": int(history),
        "critical_time_step": dt,
        "time_step": float(time_step) if time_step else None,
        "cycles": float(end / max(dt, float(time_step or 0.0))),
        "elements": kind,
        "n_nodes": n_nodes,
        "n_elements": len(elements),
        "parts": {str(q): k for q, k in parts.items()},
        "materials": {str(q): materials[q].to_config() for q in present},
        "laws": {str(q): materials[q].law for q in present},
        "phase_names": {str(k): v for k, v in names.items()},
        "groups": groups,
        "faces": {"lower": len(lower_nodes), "upper": len(upper_nodes)},
        "wall": wall,
    }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return RadiossInputs(directory, name, starter, engine, manifest)


# -- outputs -------------------------------------------------------------------------------

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"
_CYCLE = re.compile(rf"^\s*(\d+)\s+({_NUMBER})\s+({_NUMBER})\s+[A-Za-z]")


@dataclass
class RadiossListing:
    """What the listing of a run of Radioss tells: its cycles, times and time steps (the
    lines printed every few cycles), its errors and whether it ended normally (None
    while it runs, or when the listing does not say)."""

    cycles: np.ndarray
    times: np.ndarray
    steps: np.ndarray
    errors: list[str]
    normal: bool | None

    @property
    def time(self) -> float:
        return float(self.times[-1]) if len(self.times) else 0.0


def read_radioss_listing(path) -> RadiossListing:
    """The cycles, times and errors of the listing of a Starter or Engine run
    (``<name>_0000.out``, ``<name>_0001.out``)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    cycles, times, steps = [], [], []
    for line in text.splitlines():
        m = _CYCLE.match(line)
        if m:
            cycles.append(int(m.group(1)))
            times.append(float(m.group(2).replace("D", "E").replace("d", "e")))
            steps.append(float(m.group(3).replace("D", "E").replace("d", "e")))
    errors = []
    lines = text.splitlines()
    for k, line in enumerate(lines):
        if re.search(r"\bERROR\s+ID\b|^\s*\*+\s*ERROR\b|\bERROR\s*:", line):
            detail = " ".join(s.strip() for s in lines[k : k + 4] if s.strip())
            errors.append(re.sub(r"\s+", " ", detail)[:300])
    upper = text.upper()
    if "ERROR TERMINATION" in upper:
        normal = False
    elif "NORMAL TERMINATION" in upper:
        normal = True
    else:
        normal = None
    return RadiossListing(
        np.array(cycles, dtype=np.int64), np.array(times), np.array(steps), errors, normal
    )


def read_radioss_th(path) -> dict[str, np.ndarray]:
    """The columns of a time history converted to CSV (``th_to_csv``): name -> values,
    the first column, the time, under ``"time"`` as well. The names are those of the
    header (stripped of their quotes and spaces); a header given twice gets ``#2``..."""
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        except csv.Error:
            dialect = csv.excel
        rows = [r for r in csv.reader(f, dialect) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path}: an empty time history")

    def numeric(row):
        try:
            [float(c) for c in row if c.strip()]
        except ValueError:
            return False
        return True

    header = None
    if not numeric(rows[0]):
        header, rows = rows[0], rows[1:]
    rows = [[float(c) for c in r if c.strip()] for r in rows if numeric(r)]
    if not rows:
        raise ValueError(f"{path}: no values")
    width = min(len(r) for r in rows)
    data = np.array([r[:width] for r in rows], dtype=float)
    names = []
    for k in range(width):
        base = header[k].strip().strip('"').strip() if header and k < len(header) else ""
        base = base or ("time" if k == 0 else f"column {k}")
        candidate, n = base, 1
        while candidate in names:
            n += 1
            candidate = f"{base}#{n}"
        names.append(candidate)
    out = {n: data[:, k] for k, n in enumerate(names)}
    out.setdefault("time", data[:, 0])
    return out


def read_manifest(directory) -> dict:
    """The manifest of a run written by :func:`write_radioss`."""
    path = Path(directory) / MANIFEST
    if not path.is_file():
        raise ValueError(f"no run of morphoGen in {directory} (no {MANIFEST})")
    return json.loads(path.read_text(encoding="utf-8"))
