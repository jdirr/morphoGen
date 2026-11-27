"""Mesh files: Abaqus/CalculiX (.inp), Gmsh 2.2 (.msh), VTK (.vtu), Medit (.mesh), STL,
and meshio.

Voxel meshes (8-node hexahedra) and conforming tetrahedral meshes (4- or 10-node
tetrahedra) are written with their phases (element sets, physical groups or cell data)
and the nodes or element faces lying on the six faces of the box, or, for meshes of
whole periodic cells, on the boundary of the cells (``outer``); triangular meshes of the
plane (3- or 6-node triangles) with their phases and the nodes or edges of the four sides
of their rectangle.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from morphogen._version import __version__
from morphogen.io import medit
from morphogen.io.vtk import write_vtu
from morphogen.mesh.surface import SurfaceMesh
from morphogen.mesh.tetra import TetMesh
from morphogen.mesh.triangles import TriMesh
from morphogen.mesh.voxel import FACE_NAMES, VoxelMesh

#: Physical tags of the box faces in Gmsh files (``outer``: boundary of whole cells).
FACE_TAGS = {**{name: 101 + i for i, name in enumerate(FACE_NAMES)}, "outer": 107}
#: Physical tags of the sides of the rectangle in the Gmsh files of triangular meshes.
SIDE_TAGS = {"x0": 101, "x1": 102, "y0": 103, "y1": 104}


def _write_rows(f, rows: np.ndarray, fmt: str) -> None:
    np.savetxt(f, rows, fmt=fmt)


def _write_id_list(f, ids: np.ndarray, per_line: int = 16) -> None:
    ids = np.asarray(ids, dtype=np.int64)
    for start in range(0, len(ids), per_line):
        f.write(", ".join(str(int(i)) for i in ids[start : start + per_line]) + "\n")


def write_inp(
    path,
    mesh: VoxelMesh,
    *,
    element_type: str = "C3D8",
    label_sets: bool = False,
    materials: dict | None = None,
    orientations=None,
    kinematics: str = "small",
):
    """Abaqus / CalculiX input file: nodes, elements by phase, face node sets.

    Element sets ``PHASE_<p>`` (and ``CELL_<id>`` with ``label_sets``), node sets
    ``X0, X1, Y0, Y1, Z0, Z1``. ``orientations`` (Bunge angles of each label) adds the
    sets of the cells and their orientations ``ORI_<id>``; ``materials`` the materials
    and the solid sections, as :func:`write_inp_tet` (one per cell, with its orientation,
    when orientations are given; one per phase otherwise), in small or finite strains
    (``kinematics``).
    """
    path = Path(path)
    if orientations is not None and mesh.label is None:
        raise ValueError("orientations: the elements need the labels of their cells")
    mats = _abaqus_materials(path, _present(materials, mesh.phase), kinematics)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("*HEADING\n")
        f.write(f"morphoGen {__version__} voxel mesh: {mesh.n_elements} elements\n")
        _mfront_header(f, mats.sources, finite=kinematics == "finite", axes=bool(mats.axes))
        f.write("*NODE\n")
        ids = np.arange(1, mesh.n_nodes + 1)
        _write_rows(f, np.column_stack([ids, mesh.nodes]), "%d, %.12g, %.12g, %.12g")
        eid = 1
        for p in np.unique(mesh.phase):
            sel = np.flatnonzero(mesh.phase == p)
            f.write(f"*ELEMENT, TYPE={element_type}, ELSET=PHASE_{p}\n")
            rows = np.column_stack([np.arange(eid, eid + len(sel)), mesh.elements[sel] + 1])
            _write_rows(f, rows, ", ".join(["%d"] * 9))
            eid += len(sel)
        cells = label_sets or orientations is not None
        _write_cells(f, mesh.phase, mesh.label if cells else None, orientations, mats,
                     per_cell=orientations is not None)  # fmt: skip
        for name in FACE_NAMES:
            nodes = mesh.node_sets.get(name)
            if nodes is not None and len(nodes):
                f.write(f"*NSET, NSET={name.upper()}\n")
                _write_id_list(f, nodes + 1)
        _initial_temperature(f, mesh.n_nodes, mats)
    return path


def write_msh(path, mesh: VoxelMesh, *, boundary: bool = True):
    """Gmsh 2.2 ASCII file: hexahedra tagged by phase, optional boundary quadrangles.

    Physical volumes ``phase_<p>`` have tag p + 1; the elementary tag is the cell label
    + 1 when labels are attached. Physical surfaces ``x0 ... z1`` (tags 101-106) hold the
    element faces lying on the box faces.
    """
    path = Path(path)
    quads = {name: mesh.boundary_faces(name) for name in FACE_NAMES} if boundary else {}
    quads = {k: v for k, v in quads.items() if len(v)}
    phases = np.unique(mesh.phase)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n")
        f.write(f"{len(phases) + len(quads)}\n")
        for p in phases:
            f.write(f'3 {p + 1} "phase_{p}"\n')
        for name in quads:
            f.write(f'2 {FACE_TAGS[name]} "{name}"\n')
        f.write("$EndPhysicalNames\n$Nodes\n")
        f.write(f"{mesh.n_nodes}\n")
        _write_rows(
            f, np.column_stack([np.arange(1, mesh.n_nodes + 1), mesh.nodes]), "%d %.12g %.12g %.12g"
        )
        f.write("$EndNodes\n$Elements\n")
        n_quads = sum(len(q) for q in quads.values())
        f.write(f"{mesh.n_elements + n_quads}\n")
        eid = 1
        for name, q in quads.items():
            tag = FACE_TAGS[name]
            rows = np.column_stack(
                [np.arange(eid, eid + len(q)), np.full((len(q), 4), [3, 2, tag, tag]), q + 1]
            )
            _write_rows(f, rows, " ".join(["%d"] * 9))
            eid += len(q)
        geom = (mesh.label if mesh.label is not None else mesh.phase) + 1
        rows = np.column_stack(
            [
                np.arange(eid, eid + mesh.n_elements),
                np.full(mesh.n_elements, 5),
                np.full(mesh.n_elements, 2),
                mesh.phase + 1,
                geom,
                mesh.elements + 1,
            ]
        )
        _write_rows(f, rows, " ".join(["%d"] * 13))
        f.write("$EndElements\n")
    return path


def read_msh(path) -> dict:
    """Minimal reader of Gmsh 2.2 ASCII files (nodes and elements by type)."""
    lines = Path(path).read_text(encoding="ascii").splitlines()
    out = {"nodes": None, "elements": {}}
    i = 0
    while i < len(lines):
        if lines[i] == "$Nodes":
            n = int(lines[i + 1])
            arr = np.array([lines[i + 2 + k].split() for k in range(n)], dtype=float)
            out["nodes"] = arr[:, 1:]
            i += n + 2
        elif lines[i] == "$Elements":
            n = int(lines[i + 1])
            for k in range(n):
                vals = [int(v) for v in lines[i + 2 + k].split()]
                etype, ntags = vals[1], vals[2]
                out["elements"].setdefault(etype, []).append(
                    vals[3 : 3 + ntags] + vals[3 + ntags :]
                )
            i += n + 2
        else:
            i += 1
    out["elements"] = {k: np.array(v) for k, v in out["elements"].items()}
    return out


def write_vtu_mesh(path, mesh: VoxelMesh, *, compress: bool = True):
    """A voxel mesh written as a VTK unstructured grid (.vtu) of hexahedra, with the phase of each
    element (and its cell, if labelled)."""
    data = {"phase": mesh.phase.astype(np.int32)}
    if mesh.label is not None:
        data["cell"] = mesh.label.astype(np.int32)
    return write_vtu(path, mesh.nodes, mesh.elements, 12, data, compress=compress)


def write_periodic_pairs(path, mesh: VoxelMesh | TetMesh | TriMesh) -> Path:
    """CSV of facing nodes on opposite faces (1-based ids, as in .inp and .msh files).
    Meshes of whole periodic cells: each node of their boundary that is a translate of
    another, that node and the periods between them (``x_node = x_image + shift * edges``).
    """
    path = Path(path)
    if isinstance(mesh, TetMesh) and mesh.whole:
        slave, master, shift = mesh.periodic_links()
        rows = np.column_stack([slave + 1, master + 1, shift])
        with open(path, "w", encoding="ascii", newline="\n") as f:
            f.write("node,image,shift_x,shift_y,shift_z\n")
            _write_rows(f, rows, "%d,%d,%d,%d,%d")
        return path
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("axis,node_lower,node_upper\n")
        for axis, name in enumerate("xy" if isinstance(mesh, TriMesh) else "xyz"):
            pairs = mesh.periodic_pairs(axis) + 1
            for a, b in pairs:
                f.write(f"{name},{a},{b}\n")
    return path


#: Gmsh and VTK numbers of the 10-node tetrahedron: Gmsh swaps the last two mid-edge nodes.
_GMSH_TET10 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])


def _tet_face_nodes(mesh: TetMesh) -> dict[str, np.ndarray]:
    if mesh.whole:
        return {"outer": mesh.boundary_nodes()}
    return {
        f"{'xyz'[axis]}{side}": mesh.face_nodes(axis, side) for axis in range(3) for side in (0, 1)
    }


def _tet_face_elements(mesh: TetMesh) -> dict[str, np.ndarray]:
    if mesh.whole:
        return {"outer": mesh.outer_faces()}
    return {
        f"{'xyz'[axis]}{side}": mesh.boundary_faces(axis, side)
        for axis in range(3)
        for side in (0, 1)
    }


#: morphoGen's Voigt index of each of Abaqus's components (11, 22, 33, 12, 13, 23)
_ABAQUS = np.array([0, 1, 2, 5, 4, 3])


def abaqus_anisotropic(stiffness) -> list[float]:
    """The 21 constants of Abaqus's ``*ELASTIC, TYPE=ANISOTROPIC``: D1111, D1122, D2222,
    D1133, D2233, D3333, D1112, D2212, D3312, D1212, D1113, D2213, D3313, D1213, D1313,
    D1123, D2223, D3323, D1223, D1323, D2323 (Abaqus's components 11, 22, 33, 12, 13, 23,
    engineering shear strains), of a stiffness in morphoGen's Voigt order (11, 22, 33,
    23, 31, 12)."""
    d = np.asarray(stiffness, dtype=float)[np.ix_(_ABAQUS, _ABAQUS)]
    return [float(d[i, j]) for j in range(6) for i in range(j + 1)]


def _numbers(values, per_line: int = 8) -> str:
    """Data lines of at most ``per_line`` numbers, as Abaqus reads them."""
    text = [f"{float(v):.15g}" for v in values]
    return "".join(", ".join(text[k : k + per_line]) + "\n" for k in range(0, len(text), per_line))


class _AbaqusMaterials(NamedTuple):
    """The materials of an Abaqus input (:func:`_abaqus_materials`)."""

    names: dict  # the material of each phase
    lines: list  # the lines of their definitions
    sources: list  # the MFront sources of their laws
    axes: set  # the phases whose laws read the axes of their crystals in their state
    unoriented: set  # the phases whose sections take no orientation (MFront's finite strains)
    depvar: int  # the largest number of state variables of the laws
    expanding: set  # the phases whose materials expand (from the initial temperature)


def _abaqus_materials(path: Path, materials: dict, kinematics: str = "small") -> _AbaqusMaterials:
    """The material of each phase (a name, or a :class:`Material` defined here), the
    lines of the definitions, the MFront sources of the nonlinear materials, and in
    finite strains of all of them (in ``<stem>_mfront`` next to the input; the
    parameters of the user's laws in ``<behaviour>-parameters.txt``, next to it too),
    the phases whose laws read the axes of their crystals in their first six state
    variables, those whose sections take no orientation (the laws of finite strains:
    MFront's abaqus interface takes no ``*ORIENTATION`` there), and the largest number of
    state variables of the laws."""
    import shutil

    from morphogen.io.mfront import (
        ABAQUS_AXES,
        identifier,
        mfront_brick,
        mfront_state_size,
        mfront_symmetry,
        user_behaviour,
        write_mfront,
        write_parameters,
    )

    if kinematics not in ("small", "finite"):
        raise ValueError("kinematics: 'small' or 'finite'")
    finite = kinematics == "finite"
    names: dict[int, str] = {}
    lines: list[str] = []
    sources: list[Path] = []
    parameters: dict[str, tuple] = {}
    axes: set[int] = set()
    unoriented: set[int] = set()
    expanding: set[int] = set()
    depvar = 0
    folder = path.parent / f"{path.stem}_mfront"
    for p, m in sorted(materials.items()):
        p = int(p)
        if isinstance(m, str):
            names[p] = m
            continue
        if m.void:
            lines.append(f"** phase {p}: void, no material and no section\n")
            continue
        n_state = 0
        if m.nonlinear and m.mfront is not None:
            # the user's law, with its material properties as constants (one at least for
            # the interface); the dispatcher of MFront takes ABAQUSBEHAVIOUR_<behaviour>_3D
            # and ignores what follows (a material per phase)
            user = user_behaviour(m, kinematics)
            if "_" in user.name:
                raise ValueError(
                    f"phase {p}: the dispatcher of MFront in Abaqus splits the names of the "
                    f"materials at '_'; rename the behaviour {user.name}"
                )
            if finite and user.orthotropic and not user.abaqus_axes:
                raise ValueError(
                    f"phase {p}: MFront's abaqus interface takes orthotropic laws of finite "
                    f"strains with the axes in their state variables: add {ABAQUS_AXES} to "
                    f"the source of {user.name}"
                )
            target = folder / user.source.name
            if target not in sources:
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(user.source, target)
                sources.append(target)
            if parameters.setdefault(user.name, m.mfront.parameters) != m.mfront.parameters:
                raise ValueError(f"phase {p}: the parameters of {user.name} are global")
            write_parameters(path.parent, user.name, m.mfront.parameters)
            name = f"ABAQUSBEHAVIOUR_{user.name.upper()}_3D_PHASE{p}"
            values = dict(m.mfront.material_properties)
            constants = [values[k] for k in user.properties] or [0.0]
            n_state = user.internal_variables + (6 if user.abaqus_axes else 0)
            if user.abaqus_axes:
                axes.add(p)
            if user.abaqus_axes or finite:
                unoriented.add(p)
            lines.append(f"*MATERIAL, NAME={name}\n")
            if n_state:
                lines += ["*DEPVAR\n", f"{n_state},\n"]
            lines += [f"*USER MATERIAL, CONSTANTS={len(constants)}, UNSYMM\n",
                      _numbers(constants)]  # fmt: skip
        elif m.nonlinear or finite:
            if finite and m.expands and mfront_brick(m, kinematics):
                raise ValueError(
                    f"phase {p}: MFront's abaqus interface computes no thermal expansion "
                    "of the laws of finite strains of its brick (von Mises, Hill's "
                    "criterion, GTN's porous plasticity): it has no stress-free expansion in "
                    "finite strains"
                )
            # the dispatcher of MFront splits ABAQUSBEHAVIOUR_<behaviour>_3D at its '_'
            behaviour = f"{identifier(path.stem)[:40]}Phase{p}"
            sources.append(write_mfront(m, folder / f"{behaviour}.mfront", name=behaviour,
                                        kinematics=kinematics))  # fmt: skip
            name = f"ABAQUSBEHAVIOUR_{behaviour.upper()}_3D"
            # in finite strains, the axes of the orthotropic laws are their first six state
            # variables; the abaqus interface of MFront asks one constant of the laws
            # without material properties (whose values are in their sources)
            oriented = finite and mfront_symmetry(m, kinematics) == "orthotropic"
            n_state = mfront_state_size(m, kinematics) + (6 if oriented else 0)
            if oriented:
                axes.add(p)
            # in finite strains the axes are state variables; a learned law in the axes of
            # the sample (not oriented) is not turned by the grains, as morphoGen runs it
            if finite or (m.learned is not None and not m.learned.oriented):
                unoriented.add(p)
            lines.append(f"*MATERIAL, NAME={name}\n")
            if n_state:
                lines += ["*DEPVAR\n", f"{n_state},\n"]
            lines += ["*USER MATERIAL, CONSTANTS=1, UNSYMM\n", "0.\n"]
        else:
            if not m.elastic:
                raise ValueError(f"phase {p}: the material has no elastic parameters")
            name = f"PHASE_{p}"
            lines.append(f"*MATERIAL, NAME={name}\n")
            if m.stiffness is None:
                lines += ["*ELASTIC\n", _numbers((m.young, m.poisson))]
            else:  # in the axes of the crystal (the orientation of its section)
                lines += ["*ELASTIC, TYPE=ANISOTROPIC\n", _numbers(abaqus_anisotropic(m.stiffness))]
            if m.expands:
                if not isinstance(m.expansion, tuple):
                    lines += ["*EXPANSION\n", _numbers([m.expansion])]
                else:
                    a = np.asarray(m.expansion, dtype=float)
                    if np.abs(a - np.diag(np.diag(a))).max() > 1e-12 * np.abs(a).max():
                        raise ValueError(
                            f"phase {p}: an expansion tensor with shear components in the "
                            "axes of the crystal (Abaqus's orthotropic expansion)"
                        )
                    lines += ["*EXPANSION, TYPE=ORTHO\n", _numbers(np.diag(a))]
        if m.density:
            lines += ["*DENSITY\n", _numbers([m.density])]
        if m.expands and m.mfront is None:  # *EXPANSION, or the MFront law of the phase
            expanding.add(p)
        names[p] = name
        depvar = max(depvar, n_state)
    return _AbaqusMaterials(names, lines, sources, axes, unoriented, depvar, expanding)


def _present(materials: dict | None, phases) -> dict:
    """The materials of the phases of a mesh."""
    present = {int(p) for p in phases}
    return {p: m for p, m in (materials or {}).items() if int(p) in present}


def _mfront_header(f, sources: list[Path], *, finite: bool = False, axes: bool = False) -> None:
    """The comments of an Abaqus input on its MFront laws: compilation and run."""
    if sources:
        folder = sources[0].parent.name
        what = "the materials" if finite else "the nonlinear materials"
        f.write(
            f"** {what} are MFront laws ({folder}/), compiled by\n"
            f"**   cd {folder} && mfront --obuild --interface=abaqus "
            f"{' '.join(s.name for s in sources)}\n"
            "** and run with the umat of MFront (share/doc/mfront/abaqus/umat.cpp of TFEL)\n"
            f"** and {folder}/src (libABAQUSBEHAVIOUR.so) on LD_LIBRARY_PATH, from this\n"
            "** directory (the <behaviour>-parameters.txt of the laws are read there)\n"
        )
        if finite:
            f.write("** the laws are of finite strains: their steps need NLGEOM=YES\n")
        if axes:
            f.write(
                "** the orthotropic laws read the axes of their crystals in their first six\n"
                "** state variables (*INITIAL CONDITIONS, TYPE=SOLUTION), not *ORIENTATION\n"
            )


def _initial_axes(f, entries: list, depvar: int) -> None:
    """``*INITIAL CONDITIONS, TYPE=SOLUTION``: the axes [100] and [010] of the crystal of
    each element set in its first six state variables, then zeros, ``depvar`` values for
    each (Abaqus reads the same number of values for every element: seven on the line of
    the set, eight on each following line)."""
    if not entries:
        return
    f.write("*INITIAL CONDITIONS, TYPE=SOLUTION\n")
    for label, rot in entries:
        values = [f"{v:.12g}" for v in (*rot[0], *rot[1])] + ["0."] * (depvar - 6)
        f.write(", ".join([label, *values[:7]]) + "\n")
        for k in range(7, len(values), 8):
            f.write(", ".join(values[k : k + 8]) + "\n")


def _initial_temperature(f, n_nodes: int, materials: _AbaqusMaterials) -> None:
    """``*INITIAL CONDITIONS, TYPE=TEMPERATURE``: the nodes at 293.15 K when a material
    expands. It is the natural temperature of morphoGen's MFront laws, from which they count
    their thermal expansion (MFront's reference temperatures), and ``*EXPANSION`` counts it
    from the initial temperature: the steps then give absolute temperatures, 293.15 K plus
    morphoGen's change of temperature (Abaqus's nodes start at 0 otherwise)."""
    if not materials.expanding:
        return
    f.write(
        "** the expanding materials count their expansion from the initial temperature of\n"
        "** the nodes, 293.15 K (the natural temperature of morphoGen's MFront laws): the\n"
        "** steps give absolute temperatures, 293.15 K plus morphoGen's change of temperature\n"
        "*NSET, NSET=ALL_NODES, GENERATE\n"
        f"1, {n_nodes}, 1\n"
        "*INITIAL CONDITIONS, TYPE=TEMPERATURE\n"
        "ALL_NODES, 293.15\n"
    )


def _write_cells(f, phase, cell, orientations, materials: _AbaqusMaterials | None = None, *,
                 per_cell: bool = True) -> None:  # fmt: skip
    """The element sets of the cells (grains) ``CELL_<c>`` and the orientation of each,
    ``ORI_<c>`` (rectangular: the axes [100] and [010] of the crystal in the axes of the
    box), the materials, and the solid sections: one per cell (``per_cell``) when the
    elements have cells, with its orientation (each cell of one phase), one per phase
    otherwise. The sections of the laws of finite strains have no orientation; those
    whose laws read the axes of their crystals in their first six state variables get
    them from the initial conditions (:func:`_initial_axes`). The elements are numbered
    phase after phase."""
    from morphogen.orientations import euler_to_matrix

    if materials is None:
        materials = _AbaqusMaterials({}, [], [], set(), set(), 0, set())
    names, axes, unoriented = materials.names, materials.axes, materials.unoriented
    phase = np.asarray(phase)
    phases = np.unique(phase)
    cells = np.unique(cell) if cell is not None else np.zeros(0, dtype=np.int64)
    if cells.size:
        order = np.concatenate([np.flatnonzero(phase == p) for p in phases])
        numbers = np.empty(phase.size, dtype=np.int64)
        numbers[order] = np.arange(1, phase.size + 1)
        for c in cells:
            f.write(f"*ELSET, ELSET=CELL_{c}\n")
            _write_id_list(f, np.sort(numbers[cell == c]))
    oriented = cells.size > 0 and orientations is not None
    rotations = {}
    if oriented:
        g = euler_to_matrix(np.asarray(orientations, dtype=float)[cells])
        for c, rot in zip(cells, g.reshape(-1, 3, 3), strict=True):
            rotations[int(c)] = rot
            if not set(np.unique(phase[cell == c]).tolist()) <= unoriented:
                f.write(f"*ORIENTATION, NAME=ORI_{c}, SYSTEM=RECTANGULAR\n")
                f.write(", ".join(f"{v:.12g}" for v in (*rot[0], *rot[1])) + "\n")
    f.writelines(materials.lines)
    if not names:
        return
    initial = []  # the element sets whose laws read their axes in their state variables
    if cells.size and per_cell:
        for c in cells:
            own = np.unique(phase[cell == c])
            if own.size > 1:
                raise ValueError(
                    f"cell {c} spans the phases {own.tolist()}: a section has one material"
                )
            name = names.get(int(own[0]))
            if name is None:
                continue
            if int(own[0]) in axes:
                initial.append((f"CELL_{c}", rotations.get(int(c), np.eye(3))))
            extra = f", ORIENTATION=ORI_{c}" if oriented and int(own[0]) not in unoriented else ""
            f.write(f"*SOLID SECTION, ELSET=CELL_{c}, MATERIAL={name}{extra}\n")
    else:
        for p in phases:
            if names.get(int(p)) is not None:
                f.write(f"*SOLID SECTION, ELSET=PHASE_{p}, MATERIAL={names[int(p)]}\n")
                if int(p) in axes:
                    initial.append((f"PHASE_{p}", np.eye(3)))
    _initial_axes(f, initial, materials.depvar)


def write_inp_tet(path, mesh: TetMesh, *, materials: dict | None = None,
                  kinematics: str = "small") -> Path:  # fmt: skip
    """Abaqus / CalculiX input file of a tetrahedral mesh (C3D4 or C3D10 elements).

    Element sets ``PHASE_<p>``, node sets ``X0, X1, Y0, Y1, Z0, Z1`` (``OUTER``, the
    boundary of whole periodic cells). Meshes of
    polycrystals add one element set per grain, ``CELL_<id>``, and the orientation of its
    crystal axes, ``ORI_<id>`` (rectangular system: the axes [100] and [010] of the crystal
    in the axes of the box). ``materials`` writes the solid sections, one per grain with
    its orientation for polycrystals: phase -> the name of a material defined elsewhere,
    or phase -> :class:`~morphogen.homogenization.Material`, defined in the file:
    ``*ELASTIC`` (isotropic, or the 21 constants of ``TYPE=ANISOTROPIC`` in the axes of
    the crystal: see :func:`abaqus_anisotropic`), ``*EXPANSION``, ``*DENSITY``, and the
    nonlinear materials as the MFront laws of the abaqus interface
    (:mod:`morphogen.io.mfront`, their sources in ``<stem>_mfront``): the material
    ``ABAQUSBEHAVIOUR_<behaviour>_3D``, its ``*DEPVAR`` and ``*USER MATERIAL`` (one
    constant, zero, which the interface asks of laws without material properties). Voids
    have no section. In finite strains (``kinematics``), every material is an MFront law
    (Hencky's elasticity for the elastic ones; steps with ``NLGEOM=YES``), and the
    orthotropic laws read the axes of their crystals in their first six state variables,
    which ``*INITIAL CONDITIONS, TYPE=SOLUTION`` gives (MFront's abaqus interface takes no
    ``*ORIENTATION`` in finite strains). When a material expands, the nodes start at
    293.15 K (``*INITIAL CONDITIONS, TYPE=TEMPERATURE``), from which the MFront laws
    count their expansion: the steps give absolute temperatures.
    """
    path = Path(path)
    etype = "C3D4" if mesh.order == 1 else "C3D10"
    nn = mesh.elements.shape[1]
    phases = np.unique(mesh.phase)
    mats = _abaqus_materials(path, _present(materials, phases), kinematics)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("*HEADING\n")
        f.write(f"morphoGen {__version__} conforming mesh: {mesh.n_elements} {etype} elements\n")
        _mfront_header(f, mats.sources, finite=kinematics == "finite", axes=bool(mats.axes))
        f.write("*NODE\n")
        ids = np.arange(1, mesh.n_nodes + 1)
        _write_rows(f, np.column_stack([ids, mesh.nodes]), "%d, %.12g, %.12g, %.12g")
        eid = 1
        for p in phases:
            sel = np.flatnonzero(mesh.phase == p)
            f.write(f"*ELEMENT, TYPE={etype}, ELSET=PHASE_{p}\n")
            rows = np.column_stack([np.arange(eid, eid + len(sel)), mesh.elements[sel] + 1])
            if nn <= 15:
                _write_rows(f, rows, ", ".join(["%d"] * (nn + 1)))
            else:  # pragma: no cover - only 4- and 10-node elements exist
                raise ValueError("unsupported element")
            eid += len(sel)
        _write_cells(f, mesh.phase, mesh.cell, mesh.orientations, mats)
        for name, nodes in _tet_face_nodes(mesh).items():
            if len(nodes):
                f.write(f"*NSET, NSET={name.upper()}\n")
                _write_id_list(f, nodes + 1)
        _initial_temperature(f, mesh.n_nodes, mats)
    return path


def write_msh_tet(path, mesh: TetMesh, *, boundary: bool = True) -> Path:
    """Gmsh 2.2 ASCII file of a tetrahedral mesh (element types 4 and 11).

    Physical volumes ``phase_<p>`` have tag p + 1; the elementary tag is the grain + 1 for
    polycrystals; physical surfaces ``x0 ... z1`` (tags 101-106) hold the element faces
    (3- or 6-node triangles) lying on the box faces, ``outer`` (tag 107) those bounding
    whole periodic cells.
    """
    path = Path(path)
    order = mesh.order
    tet_type, tri_type = (4, 2) if order == 1 else (11, 9)
    elements = mesh.elements if order == 1 else mesh.elements[:, _GMSH_TET10]
    faces = _tet_face_elements(mesh) if boundary else {}
    faces = {k: v for k, v in faces.items() if len(v)}
    phases = np.unique(mesh.phase)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n")
        f.write(f"{len(phases) + len(faces)}\n")
        for p in phases:
            f.write(f'3 {p + 1} "phase_{p}"\n')
        for name in faces:
            f.write(f'2 {FACE_TAGS[name]} "{name}"\n')
        f.write("$EndPhysicalNames\n$Nodes\n")
        f.write(f"{mesh.n_nodes}\n")
        _write_rows(
            f, np.column_stack([np.arange(1, mesh.n_nodes + 1), mesh.nodes]), "%d %.12g %.12g %.12g"
        )
        f.write("$EndNodes\n$Elements\n")
        f.write(f"{mesh.n_elements + sum(len(v) for v in faces.values())}\n")
        eid = 1
        for name, tri in faces.items():
            tag = FACE_TAGS[name]
            rows = np.column_stack(
                [
                    np.arange(eid, eid + len(tri)),
                    np.full((len(tri), 4), [tri_type, 2, tag, tag]),
                    tri + 1,
                ]
            )
            _write_rows(f, rows, " ".join(["%d"] * rows.shape[1]))
            eid += len(tri)
        geom = (mesh.cell if mesh.cell is not None else mesh.phase) + 1
        rows = np.column_stack(
            [
                np.arange(eid, eid + mesh.n_elements),
                np.full(mesh.n_elements, tet_type),
                np.full(mesh.n_elements, 2),
                mesh.phase + 1,
                geom,
                elements + 1,
            ]
        )
        _write_rows(f, rows, " ".join(["%d"] * rows.shape[1]))
        f.write("$EndElements\n")
    return path


def write_vtu_tet(
    path,
    mesh: TetMesh,
    cell_data: dict | None = None,
    point_data: dict | None = None,
    *,
    compress: bool = True,
) -> Path:
    """VTK unstructured grid of a tetrahedral mesh (cell types 10 and 24) with its phases,
    the grains of polycrystals (``cell``) and their orientations (``euler``: Bunge angles
    in degrees), and optional fields."""
    data = {"phase": mesh.phase.astype(np.int32)}
    if mesh.cell is not None:
        data["cell"] = mesh.cell.astype(np.int32)
        if mesh.orientations is not None:
            data["euler"] = mesh.orientations[mesh.cell]
    if mesh.whole:
        data["outer"] = _outer_bits(mesh)
    data.update(cell_data or {})
    cell_type = 10 if mesh.order == 1 else 24
    return write_vtu(
        path,
        mesh.nodes,
        mesh.elements,
        cell_type,
        data,
        point_data=point_data,
        field_data=_box_fields(mesh),
        compress=compress,
    )


def _outer_bits(mesh: TetMesh) -> np.ndarray:
    """The faces of each element bounding whole periodic cells, as bits 0-3."""
    return (mesh.outer.astype(np.int32) << np.arange(4, dtype=np.int32)).sum(axis=1)


def _box_fields(mesh: TetMesh) -> dict:
    return {
        "box_lower": mesh.lower,
        "box_upper": mesh.upper,
        "periodic": np.array([int(mesh.periodic)], dtype=np.int32),
        "whole": np.array([int(mesh.whole)], dtype=np.int32),
    }


def _box_comment(mesh: TetMesh) -> str:
    box = " ".join(f"{v:.17g}" for v in (*mesh.lower, *mesh.upper))
    return f"morphogen box {box} periodic {int(mesh.periodic)} whole {int(mesh.whole)}"


def write_medit_tet(path, mesh: TetMesh) -> Path:
    """Medit file (MMG): tetrahedra with reference phase + 1, or grain + 1 for
    polycrystals (the phase of each grain is then given in a comment), corners only for
    10-node meshes; boundary triangles with references 1-6 (faces x0, x1, ..., z1), or
    1 for the faces bounding whole periodic cells."""
    lin = mesh.linear()
    tri, refs = [], []
    if lin.whole:
        tri.append(lin.outer_faces())
        refs.append(np.ones(len(tri[0]), dtype=np.int64))
    else:
        for axis in range(3):
            for side in (0, 1):
                faces = lin.boundary_faces(axis, side)
                tri.append(faces)
                refs.append(np.full(len(faces), 1 + 2 * axis + side))
    comments = [_box_comment(mesh)]
    if lin.cell is not None:
        cell_phase = np.full(int(lin.cell.max(initial=-1)) + 1, -1, dtype=np.int64)
        cell_phase[lin.cell] = lin.phase
        comments.append("morphogen references cell phases " + " ".join(map(str, cell_phase)))
    return medit.write_mesh(
        path,
        lin.nodes,
        lin.elements,
        np.concatenate(tri),
        tet_refs=(lin.cell if lin.cell is not None else lin.phase) + 1,
        tri_refs=np.concatenate(refs),
        comments=comments,
    )


def read_tet_mesh(path, *, box=None, periodic: bool | None = None) -> TetMesh:
    """Tetrahedral mesh written by morphoGen: ``.vtu`` (cell types 10 or 24 and a
    ``phase`` cell array; ``cell`` and ``euler`` for polycrystals) or Medit ``.mesh``
    (tetrahedra references = phase + 1, or grain + 1 for polycrystals).

    The box (``(lower, upper)``) and the periodicity are read from the file when it
    records them (files written by morphoGen do); otherwise the box is the bounding box
    of the nodes, which is wrong when the meshed phases do not touch every face. Meshes
    of whole periodic cells keep the faces bounding them.
    """
    from morphogen.io.vtk import read_vtu
    from morphogen.mesh.tetra import TET_FACES

    path = Path(path)
    ext = path.suffix.lower()
    stored_box, stored_periodic, whole = None, False, False
    cell = orientations = outer = None
    if ext == ".vtu":
        data = read_vtu(path)
        types = np.unique(data["types"])
        if not set(types.tolist()) <= {10, 24} or len(types) != 1:
            raise ValueError(f"{path}: expected only 4-node or only 10-node tetrahedra")
        nodes, elements = data["points"], data["cells"]
        phase = data["cell_data"].get("phase", np.zeros(len(elements)))
        if "cell" in data["cell_data"]:
            cell = np.asarray(data["cell_data"]["cell"], dtype=np.int64)
            euler = data["cell_data"].get("euler")
            if euler is not None and cell.size:
                orientations = np.zeros((int(cell.max()) + 1, 3))
                orientations[cell] = np.asarray(euler, dtype=float).reshape(-1, 3)
        fields = data.get("field_data", {})
        if "box_lower" in fields and "box_upper" in fields:
            stored_box = (fields["box_lower"], fields["box_upper"])
        stored_periodic = bool(fields.get("periodic", [0])[0])
        whole = bool(fields.get("whole", [0])[0])
        if whole and "outer" in data["cell_data"]:
            bits = np.asarray(data["cell_data"]["outer"], dtype=np.int64)
            outer = (bits[:, None] >> np.arange(4)) & 1
    elif ext == ".mesh":
        data = medit.read_mesh(path)
        if "Tetrahedra" not in data:
            raise ValueError(f"{path}: no tetrahedra")
        nodes, elements = data["Vertices"], data["Tetrahedra"]
        phase = data["Tetrahedra_refs"] - 1
        for line in data.get("comments", []):
            words = line.split()
            if words[:2] == ["morphogen", "box"] and len(words) >= 8:
                values = [float(w) for w in words[2:8]]
                stored_box = (values[:3], values[3:])
                if len(words) >= 10 and words[8] == "periodic":
                    stored_periodic = bool(int(words[9]))
                if len(words) >= 12 and words[10] == "whole":
                    whole = bool(int(words[11]))
            elif words[:4] == ["morphogen", "references", "cell", "phases"]:
                cell_phase = np.array([int(w) for w in words[4:]], dtype=np.int64)
                cell = phase
                phase = cell_phase[cell]
        if whole and len(data.get("Triangles", [])):  # the faces of the elements
            corners = np.sort(elements[:, TET_FACES].reshape(-1, 3), axis=1)
            tri = np.sort(data["Triangles"], axis=1)
            both, inverse = np.unique(np.concatenate([tri, corners]), axis=0, return_inverse=True)
            on = np.zeros(len(both), dtype=bool)
            on[inverse.ravel()[: len(tri)]] = True
            outer = on[inverse.ravel()[len(tri) :]].reshape(-1, 4)
    else:
        raise ValueError(f"{path}: tetrahedral meshes are read from .vtu or .mesh files")
    used, inverse = np.unique(elements, return_inverse=True)
    nodes = np.asarray(nodes, dtype=float)[used]
    if box is None:
        box = stored_box if stored_box is not None else (nodes.min(axis=0), nodes.max(axis=0))
    return TetMesh(
        nodes=nodes,
        elements=inverse.reshape(elements.shape),
        phase=np.asarray(phase, dtype=np.int64),
        lower=np.asarray(box[0], dtype=float),
        upper=np.asarray(box[1], dtype=float),
        periodic=stored_periodic if periodic is None else bool(periodic),
        metadata={"source": str(path)},
        cell=cell,
        orientations=orientations,
        whole=whole,
        outer=outer,
    )


# ----------------------------------------------------------------------------------------
# Triangular meshes of the plane


def _tri_sides(mesh: TriMesh) -> dict[str, np.ndarray]:
    return {f"{'xy'[axis]}{side}": mesh.side_edges(axis, side)
            for axis in (0, 1) for side in (0, 1)}  # fmt: skip


def _tri_box_fields(mesh: TriMesh) -> dict:
    return {
        "box_lower": mesh.lower,
        "box_upper": mesh.upper,
        "periodic": np.array([int(mesh.periodic)], dtype=np.int32),
        "plane_z": np.array([mesh.z]),
        "thickness": np.array([mesh.thickness]),
    }


def _tri_comment(mesh: TriMesh) -> str:
    box = " ".join(f"{v:.17g}" for v in (*mesh.lower, *mesh.upper))
    return (f"morphogen plane box {box} periodic {int(mesh.periodic)} z {mesh.z:.17g} "
            f"thickness {mesh.thickness:.17g}")  # fmt: skip


def _plane_materials(mesh: TriMesh, materials: dict, plane: str):
    """The materials of a triangular mesh for Abaqus: the lines of their definitions and
    the sections, (element set, material): one per phase, or one per grain of the oriented
    phases (``CELL_<id>``: their stiffness turned into the axes of the box, as plane
    elements take no orientation out of their plane)."""
    from morphogen.homogenization.materials import material_table

    lines: list[str] = []
    sections: list[tuple[str, str]] = []
    for p, m in sorted(materials.items()):
        p = int(p)
        if isinstance(m, str):
            sections.append((f"PHASE_{p}", m))
            continue
        if m.void:
            lines.append(f"** phase {p}: void, no material and no section\n")
            continue
        if m.nonlinear or m.mfront is not None:
            raise ValueError(f"phase {p}: the plane elements are written with elastic "
                             "materials (nonlinear ones: an extruded mesh in space)")  # fmt: skip
        if not m.elastic:
            raise ValueError(f"phase {p}: the material has no elastic parameters")
        if m.stiffness is None:
            lines += [f"*MATERIAL, NAME=PHASE_{p}\n", "*ELASTIC\n", _numbers((m.young, m.poisson))]
            if m.expands:
                if isinstance(m.expansion, tuple):
                    raise ValueError(f"phase {p}: an isotropic material expands alike")
                lines += ["*EXPANSION\n", _numbers([m.expansion])]
            if m.density:
                lines += ["*DENSITY\n", _numbers([m.density])]
            sections.append((f"PHASE_{p}", f"PHASE_{p}"))
            continue
        if plane == "stress":
            raise ValueError(f"phase {p}: Abaqus's plane stress elements take no anisotropic "
                             "elasticity; plane strain, or an extruded mesh")  # fmt: skip
        if m.expands:
            raise ValueError(f"phase {p}: the expansion of crystals in the plane elements "
                             "(their axes turned); an extruded mesh in space")  # fmt: skip
        sel = np.flatnonzero(mesh.phase == p)
        grains = None if mesh.cell is None else mesh.cell[sel]
        table = material_table(np.full(len(sel), p), {p: m}, "elastic", grains=grains,
                               orientations=mesh.orientations)  # fmt: skip
        for k, matrix in enumerate(table.matrices):
            name = f"PHASE_{p}"
            if len(table.matrices) > 1:  # one per grain
                name = f"CELL_{int(grains[table.mat == k][0])}"
            lines += [f"*MATERIAL, NAME={name}\n", "*ELASTIC, TYPE=ANISOTROPIC\n",
                      _numbers(abaqus_anisotropic(matrix))]  # fmt: skip
            if m.density:
                lines += ["*DENSITY\n", _numbers([m.density])]
            sections.append((name, name))
    return lines, sections


def write_inp_tri(path, mesh: TriMesh, *, materials: dict | None = None,
                  plane: str = "strain") -> Path:  # fmt: skip
    """Abaqus / CalculiX input file of a triangular mesh of the plane: elements CPE3 or
    CPE6 in plane strain (``plane``), CPS3 or CPS6 in plane stress, the nodes in the plane.

    Element sets ``PHASE_<p>`` (and ``CELL_<id>`` for the grains of polycrystals), node
    sets ``X0, X1, Y0, Y1`` (the sides). ``materials`` writes the solid sections, of the
    thickness of the mesh: phase -> the name of a material defined elsewhere, or phase ->
    an elastic :class:`~morphogen.homogenization.Material` defined in the file
    (``*ELASTIC``, ``*EXPANSION``, ``*DENSITY``; the crystals of a polycrystal one per
    grain, their stiffness turned into the axes of the box, ``TYPE=ANISOTROPIC``, in plane
    strain).
    """
    path = Path(path)
    if plane not in ("strain", "stress"):
        raise ValueError("plane: 'strain' or 'stress'")
    etype = {("strain", 1): "CPE3", ("strain", 2): "CPE6", ("stress", 1): "CPS3",
             ("stress", 2): "CPS6"}[(plane, mesh.order)]  # fmt: skip
    nn = mesh.elements.shape[1]
    phases = np.unique(mesh.phase)
    lines, sections = _plane_materials(mesh, _present(materials, phases), plane)
    order = np.concatenate([np.flatnonzero(mesh.phase == p) for p in phases])
    numbers = np.empty(mesh.n_elements, dtype=np.int64)
    numbers[order] = np.arange(1, mesh.n_elements + 1)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("*HEADING\n")
        f.write(f"morphoGen {__version__} mesh of the plane: {mesh.n_elements} {etype} "
                f"elements, thickness {mesh.thickness:.12g}\n")  # fmt: skip
        f.write("*NODE\n")
        ids = np.arange(1, mesh.n_nodes + 1)
        _write_rows(f, np.column_stack([ids, mesh.nodes]), "%d, %.12g, %.12g")
        for p in phases:
            sel = np.flatnonzero(mesh.phase == p)
            f.write(f"*ELEMENT, TYPE={etype}, ELSET=PHASE_{p}\n")
            rows = np.column_stack([numbers[sel], mesh.elements[sel] + 1])
            _write_rows(f, rows, ", ".join(["%d"] * (nn + 1)))
        if mesh.cell is not None:
            for c in np.unique(mesh.cell):
                f.write(f"*ELSET, ELSET=CELL_{c}\n")
                _write_id_list(f, np.sort(numbers[mesh.cell == c]))
        for name, edges in _tri_sides(mesh).items():
            nodes = np.unique(edges)
            if len(nodes):
                f.write(f"*NSET, NSET={name.upper()}\n")
                _write_id_list(f, nodes + 1)
        f.writelines(lines)
        for elset, material in sections:
            f.write(f"*SOLID SECTION, ELSET={elset}, MATERIAL={material}\n"
                    f"{mesh.thickness:.12g},\n")  # fmt: skip
    return path


def write_msh_tri(path, mesh: TriMesh, *, boundary: bool = True) -> Path:
    """Gmsh 2.2 ASCII file of a triangular mesh (element types 2 and 9), the nodes at the
    height ``z`` of its plane.

    Physical surfaces ``phase_<p>`` have tag p + 1; the elementary tag is the grain + 1 for
    polycrystals; physical lines ``x0, x1, y0, y1`` (tags 101-104) hold the edges (2- or
    3-node lines) of the sides.
    """
    path = Path(path)
    tri_type, line_type = (2, 1) if mesh.order == 1 else (9, 8)
    sides = {k: v for k, v in _tri_sides(mesh).items() if len(v)} if boundary else {}
    phases = np.unique(mesh.phase)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n")
        f.write(f"{len(phases) + len(sides)}\n")
        for p in phases:
            f.write(f'2 {p + 1} "phase_{p}"\n')
        for name in sides:
            f.write(f'1 {SIDE_TAGS[name]} "{name}"\n')
        f.write("$EndPhysicalNames\n$Nodes\n")
        f.write(f"{mesh.n_nodes}\n")
        rows = np.column_stack([np.arange(1, mesh.n_nodes + 1), mesh.nodes,
                                np.full(mesh.n_nodes, mesh.z)])  # fmt: skip
        _write_rows(f, rows, "%d %.12g %.12g %.12g")
        f.write("$EndNodes\n$Elements\n")
        f.write(f"{mesh.n_elements + sum(len(v) for v in sides.values())}\n")
        eid = 1
        for name, edges in sides.items():
            tag = SIDE_TAGS[name]
            rows = np.column_stack([np.arange(eid, eid + len(edges)),
                                    np.full((len(edges), 4), [line_type, 2, tag, tag]),
                                    edges + 1])  # fmt: skip
            _write_rows(f, rows, " ".join(["%d"] * rows.shape[1]))
            eid += len(edges)
        geom = (mesh.cell if mesh.cell is not None else mesh.phase) + 1
        rows = np.column_stack([np.arange(eid, eid + mesh.n_elements),
                                np.full(mesh.n_elements, tri_type), np.full(mesh.n_elements, 2),
                                mesh.phase + 1, geom, mesh.elements + 1])  # fmt: skip
        _write_rows(f, rows, " ".join(["%d"] * rows.shape[1]))
        f.write("$EndElements\n")
    return path


def write_vtu_tri(path, mesh: TriMesh, cell_data: dict | None = None,
                  point_data: dict | None = None, *, compress: bool = True) -> Path:  # fmt: skip
    """VTK unstructured grid of a triangular mesh (cell types 5 and 22), the nodes at the
    height ``z`` of its plane, with its phases, the grains of polycrystals (``cell``) and
    their orientations (``euler``: Bunge angles in degrees), and optional fields."""
    data = {"phase": mesh.phase.astype(np.int32)}
    if mesh.cell is not None:
        data["cell"] = mesh.cell.astype(np.int32)
        if mesh.orientations is not None:
            data["euler"] = mesh.orientations[mesh.cell]
    data.update(cell_data or {})
    points = np.column_stack([mesh.nodes, np.full(mesh.n_nodes, mesh.z)])
    return write_vtu(path, points, mesh.elements, 5 if mesh.order == 1 else 22, data,
                     point_data=point_data, field_data=_tri_box_fields(mesh),
                     compress=compress)  # fmt: skip


def write_medit_tri(path, mesh: TriMesh) -> Path:
    """Medit file of the plane (``Dimension 2``, MMG): triangles with reference phase + 1,
    or grain + 1 for polycrystals (the phase of each grain then given in a comment),
    corners only for 6-node meshes; the edges of the sides with references 1-4 (x0, x1,
    y0, y1)."""
    lin = mesh.linear()
    sides = _tri_sides(lin)
    edges = np.concatenate([e[:, :2] for e in sides.values()])
    refs = np.concatenate([np.full(len(e), 1 + k) for k, e in enumerate(sides.values())])
    comments = [_tri_comment(mesh)]
    if lin.cell is not None:
        cell_phase = np.full(int(lin.cell.max(initial=-1)) + 1, -1, dtype=np.int64)
        cell_phase[lin.cell] = lin.phase
        comments.append("morphogen references cell phases " + " ".join(map(str, cell_phase)))
    return medit.write_mesh(path, lin.nodes, None, lin.elements, edges=edges,
                            tri_refs=(lin.cell if lin.cell is not None else lin.phase) + 1,
                            edge_refs=refs, comments=comments)  # fmt: skip


def read_tri_mesh(path, *, box=None, periodic: bool | None = None) -> TriMesh:
    """Triangular mesh written by morphoGen: ``.vtu`` (cell types 5 or 22 and a ``phase``
    cell array; ``cell`` and ``euler`` for polycrystals) or Medit ``.mesh`` of dimension 2
    (triangle references = phase + 1, or grain + 1 for polycrystals). The rectangle
    (``(lower, upper)``), the periodicity, the height of the plane and the thickness are
    read from the file when it records them (files written by morphoGen do); otherwise the
    rectangle is the bounding box of the nodes."""
    from morphogen.io.vtk import read_vtu

    path = Path(path)
    ext = path.suffix.lower()
    stored_box, stored_periodic, z, thickness = None, False, 0.0, 1.0
    cell = orientations = None
    if ext == ".vtu":
        data = read_vtu(path)
        types = np.unique(data["types"])
        if not set(types.tolist()) <= {5, 22} or len(types) != 1:
            raise ValueError(f"{path}: expected only 3-node or only 6-node triangles")
        points, elements = np.asarray(data["points"], dtype=float), data["cells"]
        nodes = points[:, :2]
        phase = data["cell_data"].get("phase", np.zeros(len(elements)))
        if "cell" in data["cell_data"]:
            cell = np.asarray(data["cell_data"]["cell"], dtype=np.int64)
            euler = data["cell_data"].get("euler")
            if euler is not None and cell.size:
                orientations = np.zeros((int(cell.max()) + 1, 3))
                orientations[cell] = np.asarray(euler, dtype=float).reshape(-1, 3)
        fields = data.get("field_data", {})
        if "box_lower" in fields and "box_upper" in fields:
            stored_box = (fields["box_lower"][:2], fields["box_upper"][:2])
        stored_periodic = bool(fields.get("periodic", [0])[0])
        z = float(fields.get("plane_z", [points[:, 2].mean() if len(points) else 0.0])[0])
        thickness = float(fields.get("thickness", [1.0])[0])
    elif ext == ".mesh":
        data = medit.read_mesh(path)
        if "Triangles" not in data or data["Vertices"].shape[1] != 2:
            raise ValueError(f"{path}: no triangles of the plane (Dimension 2)")
        nodes, elements = data["Vertices"], data["Triangles"]
        phase = data["Triangles_refs"] - 1
        for line in data.get("comments", []):
            words = line.split()
            if words[:3] == ["morphogen", "plane", "box"] and len(words) >= 13:
                values = [float(w) for w in words[3:7]]
                stored_box = (values[:2], values[2:])
                stored_periodic = bool(int(words[8]))
                z, thickness = float(words[10]), float(words[12])
            elif words[:4] == ["morphogen", "references", "cell", "phases"]:
                cell_phase = np.array([int(w) for w in words[4:]], dtype=np.int64)
                cell = phase
                phase = cell_phase[cell]
    else:
        raise ValueError(f"{path}: triangular meshes are read from .vtu or .mesh files")
    used, inverse = np.unique(elements, return_inverse=True)
    nodes = np.asarray(nodes, dtype=float)[used]
    if box is None:
        box = stored_box if stored_box is not None else (nodes.min(axis=0), nodes.max(axis=0))
    return TriMesh(nodes, inverse.reshape(np.shape(elements)), np.asarray(phase, dtype=np.int64),
                   np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float),
                   stored_periodic if periodic is None else bool(periodic),
                   {"source": str(path)}, cell=cell, orientations=orientations, z=z,
                   thickness=thickness)  # fmt: skip


def write_stl(path, surface: SurfaceMesh, *, name: str = "morphogen") -> Path:
    """Binary STL."""
    path = Path(path)
    tri = surface.vertices[surface.faces].astype(np.float32)
    normals = surface.face_normals().astype(np.float32)
    record = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    data = np.zeros(len(tri), dtype=record)
    data["n"] = normals
    data["v"] = tri
    header = f"{name} (morphoGen {__version__})".encode("ascii", "replace")[:80].ljust(80, b" ")
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.uint32(len(tri)).tobytes())
        f.write(data.tobytes())
    return path


def read_stl(path) -> SurfaceMesh:
    """Binary STL reader (vertices are not merged)."""
    raw = Path(path).read_bytes()
    n = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
    record = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    data = np.frombuffer(raw, dtype=record, count=n, offset=84)
    verts = data["v"].reshape(-1, 3).astype(float)
    return SurfaceMesh(verts, np.arange(3 * n).reshape(n, 3))


def write_meshio(path, mesh: VoxelMesh | TetMesh | TriMesh | SurfaceMesh) -> Path:
    """Any other format supported by the optional ``meshio`` package."""
    try:
        import meshio
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"format of {path} needs the optional 'meshio' package: pip install 'morphogen[io]'"
        ) from None
    if isinstance(mesh, SurfaceMesh):
        m = meshio.Mesh(mesh.vertices, [("triangle", mesh.faces)])
    elif isinstance(mesh, TriMesh):
        kind = "triangle" if mesh.order == 1 else "triangle6"
        data = {"phase": [mesh.phase]}
        if mesh.cell is not None:
            data["cell"] = [mesh.cell]
            if mesh.orientations is not None:
                data["euler"] = [mesh.orientations[mesh.cell]]
        points = np.column_stack([mesh.nodes, np.full(mesh.n_nodes, mesh.z)])
        m = meshio.Mesh(points, [(kind, mesh.elements)], cell_data=data)
    elif isinstance(mesh, TetMesh):
        kind = "tetra" if mesh.order == 1 else "tetra10"
        data = {"phase": [mesh.phase]}
        if mesh.cell is not None:
            data["cell"] = [mesh.cell]
            if mesh.orientations is not None:
                data["euler"] = [mesh.orientations[mesh.cell]]
        if mesh.whole:
            data["outer"] = [_outer_bits(mesh)]
        m = meshio.Mesh(mesh.nodes, [(kind, mesh.elements)], cell_data=data)
    else:
        data = {"phase": [mesh.phase]}
        if mesh.label is not None:
            data["cell"] = [mesh.label]
        m = meshio.Mesh(mesh.nodes, [("hexahedron", mesh.elements)], cell_data=data)
    meshio.write(str(path), m)
    return Path(path)
