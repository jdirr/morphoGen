"""Studies of Code_Aster, the finite element code of EDF (https://code-aster.org), on the
conforming meshes of morphoGen: the apparent stiffness and conductivity of a volume
element, and their results.

:func:`write_code_aster` writes the files of a study of a conforming tetrahedral mesh
(:class:`~morphogen.mesh.tetra.TetMesh`, 4- or 10-node elements):

* the mesh, ``<name>.mail`` in the native format of Code_Aster (``LIRE_MAILLAGE``,
  ``FORMAT='ASTER'``; :func:`write_aster_mesh`) or ``<name>.med`` (MED through meshio and
  h5py, its groups as families): the nodes ``N1``, ``N2``..., the elements ``M1``...
  (``TETRA4`` or ``TETRA10``, then the triangles ``TRIA3`` or ``TRIA6`` of the faces of
  the box), numbered as in VTK and morphoGen (the corners, then the middles of the edges
  1-2, 2-3, 3-1, 1-4, 2-4 and 3-4 of a tetrahedron; 1-2, 2-3, 3-1 of a triangle), which
  is Code_Aster's order (MED numbers the corners of a tetrahedron with the second and
  the third swapped, its mirror image, and its mid-edge nodes accordingly: its tetrahedra
  are written so); the groups of elements ``PHASE_<p>`` of the phases (voids are
  left out of the mesh), ``GRAIN_<c>`` of the grains of oriented crystals, ``XMIN``,
  ``XMAX``... ``ZMAX`` of the triangles of the faces of the box (``OUTER`` for whole
  periodic cells); the groups of nodes ``N_XMIN``... ``N_ZMAX`` of the faces,
  ``BOUNDARY`` of all of them, ``ORIGIN`` (the node that the periodic conditions fix,
  the nearest to the lower corner of the box among those tied to no other) and, for the
  periodic conditions, ``PS_<s>`` and ``PM_<s>`` (below);
* the command file ``<name>.comm``: ``LIRE_MAILLAGE``, the model of the phases
  (``AFFE_MODELE``, ``MODELISATION='3D'``, mechanical and thermal), the material of each
  phase (``DEFI_MATERIAU``: ``ELAS`` of its Young's modulus and Poisson's ratio,
  ``ELAS_ORTH`` of the nine engineering constants of an orthotropic crystal in its axes;
  ``THER`` of its conductivity, ``THER_ORTH`` of the principal conductivities of a
  crystal), assigned by group (``AFFE_MATERIAU``), the axes of the crystals
  (``AFFE_CARA_ELEM``, ``MASSIF``: ``ANGL_EULER``, the Bunge angles of the grain, which
  Code_Aster's Euler angles are: its local frame is the frame of the crystal), a loading
  per unit strain (per unit gradient of temperature), ``MECA_STATIQUE``
  (``THER_LINEAIRE``, stationary at the one instant of its ``INCREMENT``, then
  ``CALC_CHAMP`` of ``FLUX_ELGA``), the integrals of the stresses ``SIEF_ELGA`` (of the
  fluxes) over the elements (``POST_ELEM``, ``INTEGRALE``) and their tables
  (``IMPR_TABLE``, ``FORMAT='TABLEAU'``, a file per loading, ``<name>_<case>.table``), and
  with ``fields`` the results in MED;
* ``<name>.export``, the export file of ``run_aster`` (``run_aster <name>.export``): the
  parameters of the run (``P``), the data and result files with their logical units
  (``F``);
* ``<name>_aster.json``, the description of the study, which :func:`read_code_aster`
  reads back with the tables.

**Periodic conditions** (``conditions="periodic"``, a mesh with matching opposite faces):
the displacement ``u = E x + w`` has a periodic fluctuation ``w``, so that a node ``x`` of
an upper face and its image ``x - s L`` on the lower faces (``s``: 1 along each axis
whose upper face holds the node, 0 along the others; ``L``: the edges of the box) obey
``u(x) - u(x - s L) = E (s L)``. The nodes translated along the same axes form the group
``PS_<s>`` (``s`` written as three digits, ``M`` for -1 in whole periodic cells) and
their images the group ``PM_<s>``: one ``LIAISON_GROUP`` per group and component pairs
each node of ``GROUP_NO_1`` translated by ``TRAN = -s L`` with the nearest node of
``GROUP_NO_2`` and imposes ``u_i(x_1) - u_i(x_2) = (E s L)_i`` (``COEF_IMPO``). Every node
of the upper faces is in one group and tied to one image on the lower faces, which is
tied to no other: the equations are independent, those of the edges and corners too,
and ``ORIGIN`` is fixed (the rigid translations). The temperature likewise, ``T(x_1) -
T(x_2) = G . (s L)``. **Kinematic conditions** (``conditions="kinematic"``): ``u = E x``
(``T = G . x``) at the nodes of the boundary, functions of the coordinates
(``FORMULE`` of ``X``, ``Y``, ``Z``; ``AFFE_CHAR_MECA_F``, ``AFFE_CHAR_THER_F``).

**Results.** The table of a loading gives the integrals ``INTE_SIXX``... of the stresses
over the 3D elements of the model (``TOUT='OUI'``); :func:`read_code_aster` divides them
by the volume of the box (a void carries no stress), which gives the mean stress of the
unit strain, a column of the apparent stiffness (Voigt order 11, 22, 33, 23, 31, 12,
engineering shears: the shear strains of the loadings are ``E_ij = 1/2``), and minus the
mean flux of a unit gradient, a column of the apparent conductivity. The nonlinear
materials are written by their elasticity; the expansion of the materials is not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from morphogen._version import __version__
from morphogen.mesh.tetra import TetMesh

#: Code_Aster's names of the components of the stresses, in morphoGen's Voigt order (11,
#: 22, 33, 23, 31, 12), and of the heat fluxes
STRESSES = ("SIXX", "SIYY", "SIZZ", "SIYZ", "SIXZ", "SIXY")
FLUXES = ("FLUX", "FLUY", "FLUZ")
#: The names of the faces of the box, by axis and side
FACES = ("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX")
#: The unit strains of the stiffness (Voigt order) and the unit gradients of the
#: conductivity: the names of the loadings
STRAIN_CASES = ("E11", "E22", "E33", "E23", "E31", "E12")
GRADIENT_CASES = ("G1", "G2", "G3")
#: The physics of a study
PHYSICS = ("elastic", "thermal")
CONDITIONS = ("periodic", "kinematic")
MESH_FORMATS = ("aster", "med")
#: The nodes of the tetrahedra of MED in the order of VTK's and Code_Aster's (indices of
#: MED's nodes; the permutation is its own inverse)
MED_TETRA4 = (0, 2, 1, 3)
MED_TETRA10 = (0, 2, 1, 3, 6, 5, 4, 7, 9, 8)
#: The logical units of the mesh and of the first table
_MESH_UNIT, _FIRST_TABLE, _FIELDS_UNIT = 20, 81, 80
#: The longest line and word of the files of Code_Aster's format
_LINE, _WORD = 80, 8


def _unit_strain(k: int) -> np.ndarray:
    """The tensor of the unit strain ``k`` (Voigt order, engineering shear strains)."""
    pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))
    i, j = pairs[k]
    e = np.zeros((3, 3))
    e[i, j] = e[j, i] = 1.0 if i == j else 0.5
    return e


# -- the mesh ---------------------------------------------------------------------------


def _wrapped(words: list[str], first: str = "") -> list[str]:
    """Lines of at most 80 characters holding the words, the first line starting with
    ``first`` (an element name); the next lines continue the list."""
    lines, line = [], first
    for w in words:
        if line and len(line) + 1 + len(w) > _LINE:
            lines.append(line)
            line = " " + w if first else w
        else:
            line = f"{line} {w}" if line else w
    if line:
        lines.append(line)
    return lines


def _check_count(n: int, what: str) -> None:
    if n >= 10 ** (_WORD - 1):
        raise ValueError(
            f"Code_Aster's format: names of at most {_WORD} characters, at most "
            f"{10 ** (_WORD - 1) - 1} {what}; write the mesh in MED"
        )


def _box_faces(mesh: TetMesh) -> dict[str, np.ndarray]:
    """The triangles of the faces of the box (whole periodic cells: of their boundary)."""
    if mesh.whole:
        return {"OUTER": mesh.outer_faces()}
    return {
        FACES[2 * axis + side]: mesh.boundary_faces(axis, side)
        for axis in range(3)
        for side in (0, 1)
    }


@dataclass
class AsterGroups:
    """The groups of a mesh written for Code_Aster: the triangles of the faces of the box
    (``faces``), the groups of elements (``elements``: indices into the tetrahedra, then
    the triangles, numbered after them) and of nodes, and the periodic classes (``links``:
    the name of each class, its translation ``s`` and its pairs of nodes)."""

    faces: dict[str, np.ndarray]
    elements: dict[str, np.ndarray]
    nodes: dict[str, np.ndarray]
    links: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    origin: int = 0


def _class_name(shift) -> str:
    return "".join({-1: "M", 0: "0", 1: "1"}[int(s)] for s in shift)


def aster_groups(mesh: TetMesh, *, periodic: bool = False, grains=()) -> AsterGroups:
    """The groups of the elements and nodes of a mesh written for Code_Aster (see the
    module): those of its phases, of the grains ``grains`` (cell labels), of the faces of
    the box and of their nodes, of the boundary, the node ``ORIGIN`` and, with
    ``periodic``, the classes ``PS_<s>`` / ``PM_<s>`` of the nodes of the upper faces and
    of their images (a mesh with matching opposite faces, or of whole periodic cells)."""
    from morphogen.io.solvers import periodic_images

    faces = _box_faces(mesh)
    n_tet = mesh.n_elements
    elements: dict[str, np.ndarray] = {}
    for p in np.unique(mesh.phase):
        elements[f"PHASE_{int(p)}"] = np.flatnonzero(mesh.phase == p)
    for c in grains:
        elements[f"GRAIN_{int(c)}"] = np.flatnonzero(mesh.cell == c)
    start = n_tet
    for name, tri in faces.items():
        if len(tri):
            elements[name] = start + np.arange(len(tri))
        start += len(tri)
    nodes: dict[str, np.ndarray] = {}
    if not mesh.whole:
        for axis in range(3):
            for side in (0, 1):
                found = mesh.face_nodes(axis, side)
                if len(found):
                    nodes[f"N_{FACES[2 * axis + side]}"] = found
    nodes["BOUNDARY"] = mesh.boundary_nodes()
    links: dict[str, tuple] = {}
    slaves = np.zeros(0, dtype=np.int64)
    if periodic:
        slaves, masters, shifts = periodic_images(mesh)
        keys = [tuple(int(v) for v in s) for s in shifts]
        for key in sorted(set(keys)):
            sel = np.array([k == key for k in keys], dtype=bool)
            name = _class_name(key)
            links[name] = (np.asarray(key, dtype=np.int64), slaves[sel], masters[sel])
            nodes[f"PS_{name}"] = slaves[sel]
            nodes[f"PM_{name}"] = masters[sel]
    free = np.setdiff1d(np.arange(mesh.n_nodes), slaves)
    origin = int(free[np.argmin(np.linalg.norm(mesh.nodes[free] - mesh.lower, axis=1))])
    nodes["ORIGIN"] = np.array([origin])
    return AsterGroups(faces, elements, nodes, links, origin)


def write_aster_mesh(path, mesh: TetMesh, *, groups: AsterGroups | None = None) -> Path:
    """A conforming tetrahedral mesh in Code_Aster's native format (``.mail``,
    ``LIRE_MAILLAGE(FORMAT='ASTER')``): the title, the coordinates (``COOR_3D``), the
    tetrahedra (``TETRA4``, ``TETRA10``), the triangles of the faces of the box
    (``TRIA3``, ``TRIA6``) and the groups (``GROUP_MA``, ``GROUP_NO``, ``NOM=``) of
    :func:`aster_groups` (by default those of a mesh without periodic classes). Words of
    at most eight characters, lines of at most 80 (the element lists continue on the next
    lines), comments after ``%``, each block closed by ``FINSF`` and the file by ``FIN``.
    """
    path = Path(path)
    if groups is None:
        groups = aster_groups(mesh)
    faces = [tri for tri in groups.faces.values() if len(tri)]
    n_faces = sum(len(t) for t in faces)
    _check_count(mesh.n_nodes, "nodes")
    _check_count(mesh.n_elements + n_faces, "elements")
    quadratic = mesh.order == 2
    lines = [
        "TITRE",
        f"morphoGen {__version__}: conforming mesh of {mesh.n_elements} tetrahedra",
        "box lower " + " ".join(f"{v:.9g}" for v in mesh.lower),
        "box upper " + " ".join(f"{v:.9g}" for v in mesh.upper),
        "FINSF",
        "%",
        "COOR_3D",
    ]
    lines += [f"N{i + 1} {x:.12g} {y:.12g} {z:.12g}" for i, (x, y, z) in enumerate(mesh.nodes)]
    lines += ["FINSF", "%", "TETRA10" if quadratic else "TETRA4"]
    for e, row in enumerate(mesh.elements):
        lines += _wrapped([f"N{v + 1}" for v in row], first=f"M{e + 1}")
    lines.append("FINSF")
    if faces:
        lines += ["%", "TRIA6" if quadratic else "TRIA3"]
        number = mesh.n_elements
        for tri in faces:
            for row in tri:
                number += 1
                lines += _wrapped([f"N{v + 1}" for v in row], first=f"M{number}")
        lines.append("FINSF")
    for name, ids in groups.elements.items():
        lines += ["%", f"GROUP_MA NOM={name}", *_wrapped([f"M{i + 1}" for i in ids]), "FINSF"]
    for name, ids in groups.nodes.items():
        lines += ["%", f"GROUP_NO NOM={name}", *_wrapped([f"N{i + 1}" for i in ids]), "FINSF"]
    lines += ["%", "FIN"]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def read_aster_mesh(path) -> dict:
    """A mesh in Code_Aster's native format, as :func:`write_aster_mesh` writes it: the
    coordinates of the nodes (``nodes``, in the order of their names), the elements by
    type (``elements``: type -> (names, node indices)), and the groups (``groups_ma``,
    ``groups_no``: name -> element names or node indices)."""
    text = Path(path).read_text(encoding="ascii")
    lines = [line.split("%", 1)[0].strip() for line in text.splitlines()]
    node_names: list[str] = []
    coords: list[list[float]] = []
    elements: dict[str, tuple[list[str], list[list[str]]]] = {}
    groups_ma: dict[str, list[str]] = {}
    groups_no: dict[str, list[str]] = {}
    sizes = {"TETRA4": 4, "TETRA10": 10, "TRIA3": 3, "TRIA6": 6}
    k = 0
    while k < len(lines):
        words = lines[k].split()
        k += 1
        if not words:
            continue
        key = words[0].upper()
        if key == "FIN":
            break
        block = []
        while k < len(lines) and lines[k].upper() != "FINSF":
            block += lines[k].split()
            k += 1
        k += 1
        if key == "COOR_3D":
            for i in range(0, len(block), 4):
                node_names.append(block[i])
                coords.append([float(v) for v in block[i + 1 : i + 4]])
        elif key in sizes:
            n = sizes[key]
            names, rows = elements.setdefault(key, ([], []))
            for i in range(0, len(block), n + 1):
                names.append(block[i])
                rows.append(block[i + 1 : i + 1 + n])
        elif key in ("GROUP_MA", "GROUP_NO"):
            name = words[1].split("=", 1)[1] if len(words) > 1 else block.pop(0)
            (groups_ma if key == "GROUP_MA" else groups_no)[name] = block
    index = {name: i for i, name in enumerate(node_names)}
    return {
        "nodes": np.array(coords, dtype=float).reshape(-1, 3),
        "node_names": node_names,
        "elements": {
            t: (names, np.array([[index[v] for v in r] for r in rows], dtype=np.int64))
            for t, (names, rows) in elements.items()
        },
        "groups_ma": groups_ma,
        "groups_no": {
            g: np.array([index[v] for v in ids], dtype=np.int64) for g, ids in groups_no.items()
        },
    }


def _families(labels: list[tuple[str, ...]], sign: int) -> tuple[np.ndarray, dict]:
    """The MED families of the items of a mesh, from the groups of each item: one family
    per set of groups (0 for the items in none; negative numbers for the elements,
    positive for the nodes)."""
    numbers: dict[tuple[str, ...], int] = {}
    out = np.zeros(len(labels), dtype=np.int64)
    for i, key in enumerate(labels):
        if key:
            out[i] = numbers.setdefault(key, sign * (len(numbers) + 1))
    return out, {v: list(k) for k, v in numbers.items()}


def _write_med(path: Path, mesh: TetMesh, groups: AsterGroups) -> Path:
    """The mesh and its groups in MED (meshio and h5py): the groups as families."""
    try:
        import h5py  # noqa: F401 - meshio's MED writer needs it
        import meshio
    except ImportError as exc:  # pragma: no cover - depends on the installation
        raise ImportError(
            "MED meshes need meshio and h5py (python -m pip install meshio h5py); or write "
            "the mesh in Code_Aster's format (mesh_format='aster')"
        ) from exc
    quadratic = mesh.order == 2
    tri = [t for t in groups.faces.values() if len(t)]
    tri = np.concatenate(tri) if tri else np.zeros((0, 6 if quadratic else 3), np.int64)
    n_tet, n_all = mesh.n_elements, mesh.n_elements + len(tri)
    member: list[list[str]] = [[] for _ in range(n_all)]
    for name, ids in groups.elements.items():
        for i in ids:
            member[i].append(name)
    tags, cell_tags = _families([tuple(m) for m in member], -1)
    node_member: list[list[str]] = [[] for _ in range(mesh.n_nodes)]
    for name, ids in groups.nodes.items():
        for i in ids:
            node_member[i].append(name)
    node_tags, point_tags = _families([tuple(m) for m in node_member], 1)
    # MED numbers the nodes of the tetrahedra otherwise than VTK and Code_Aster: 1 3 2 4
    # (the mid-edge nodes 7 6 5 8 10 9), a permutation that is its own inverse, which
    # Code_Aster applies when it reads them
    order = MED_TETRA10 if quadratic else MED_TETRA4
    cells = [("tetra10" if quadratic else "tetra", mesh.elements[:, order])]
    data = [tags[:n_tet]]
    if len(tri):
        cells.append(("triangle6" if quadratic else "triangle", tri))
        data.append(tags[n_tet:])
    out = meshio.Mesh(
        mesh.nodes, cells, point_data={"point_tags": node_tags}, cell_data={"cell_tags": data}
    )
    out.cell_tags = cell_tags
    out.point_tags = point_tags
    meshio.write(path, out, file_format="med")
    return path


# -- the materials ------------------------------------------------------------------------


def _number(value: float) -> str:
    """A real of a command (a decimal point always: the reals of the catalogues)."""
    value = float(value) + 0.0  # no negative zero
    text = f"{value:.12g}"
    return text if any(c in text for c in ".en") else text + ".0"


def _orthotropic(stiffness, phase: int) -> dict[str, float]:
    """``ELAS_ORTH``: the engineering constants of an orthotropic stiffness in its axes."""
    from morphogen.io.mfront import orthotropic_constants

    try:
        e1, e2, e3, nu12, nu23, nu13, g12, g23, g13 = orthotropic_constants(stiffness)
    except ValueError:
        raise ValueError(
            f"phase {phase}: Code_Aster's elasticity of crystals is orthotropic (ELAS_ORTH); "
            "this stiffness couples normal and shear components in the axes of the crystal"
        ) from None
    return {
        "E_L": e1,
        "E_T": e2,
        "E_N": e3,
        "NU_LT": nu12,
        "NU_LN": nu13,
        "NU_TN": nu23,
        "G_LT": g12,
        "G_LN": g13,
        "G_TN": g23,
    }


def _behaviours(material, phase: int, physics) -> list[str]:
    """The keywords of ``DEFI_MATERIAU`` of a material for the physics of a study."""
    parts = []
    if "elastic" in physics:
        if not material.elastic:
            raise ValueError(f"phase {phase}: the material has no elastic parameters")
        if material.stiffness is None:
            parts.append(f"ELAS=_F(E={_number(material.young)}, NU={_number(material.poisson)})")
        else:
            values = ", ".join(
                f"{k}={_number(v)}" for k, v in _orthotropic(material.stiffness, phase).items()
            )
            parts.append(f"ELAS_ORTH=_F({values})")
    if "thermal" in physics:
        if not material.thermal:
            raise ValueError(f"phase {phase}: the material has no conductivity")
        if material.conductivities is None:
            parts.append(f"THER=_F(LAMBDA={_number(material.conductivity)})")
        else:
            k = np.asarray(material.conductivities, dtype=float)
            if np.abs(k - np.diag(np.diag(k))).max() > 1e-12 * np.abs(k).max():
                raise ValueError(
                    f"phase {phase}: Code_Aster's conduction of crystals is orthotropic "
                    "(THER_ORTH); this conductivity is not diagonal in the axes of the crystal"
                )
            lam = np.diag(k)
            parts.append(
                f"THER_ORTH=_F(LAMBDA_L={_number(lam[0])}, LAMBDA_T={_number(lam[1])}, "
                f"LAMBDA_N={_number(lam[2])})"
            )
    return parts


def _oriented(material, physics) -> bool:
    return ("elastic" in physics and material.stiffness is not None) or (
        "thermal" in physics and material.conductivities is not None
    )


# -- the command file ---------------------------------------------------------------------


def _tuple(values) -> str:
    items = [f'"{v}"' if isinstance(v, str) else _number(v) for v in values]
    return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"


def _call(name: str, target: str | None, args: list[str]) -> list[str]:
    """A command of the command file, its keywords one per line."""
    head = f"{target} = {name}(" if target else f"{name}("
    return [head, *[f"    {a}," for a in args], ")"]


def _factors(items: list[str]) -> str:
    """A tuple of ``_F`` factor keywords on several lines."""
    if len(items) == 1:
        return items[0]
    return "(\n" + "".join(f"        {i},\n" for i in items) + "    )"


@dataclass
class CodeAsterExport:
    """The files of a study of Code_Aster (:func:`write_code_aster`): the directory, the
    name of the files, the files written and the loadings (their physics, unit strain or
    gradient, table and logical unit)."""

    directory: Path
    name: str
    files: list[Path]
    cases: dict[str, dict]


def _formula(name: str, coefficients) -> str:
    terms = " + ".join(f"({_number(c)})*{x}" for c, x in zip(coefficients, "XYZ", strict=True))
    return f'{name} = FORMULE(VALE="{terms}", NOM_PARA=("X", "Y", "Z"))'


def write_code_aster(
    directory,
    mesh: TetMesh,
    materials: dict,
    *,
    physics=None,
    conditions: str | None = None,
    name: str = "aster",
    mesh_format: str = "aster",
    fields: bool = False,
    memory: float = 4096.0,
    time_limit: float = 3600.0,
    threads: int = 1,
) -> CodeAsterExport:
    """The files of a study of Code_Aster of the apparent properties of a mesh (see the
    module): ``materials`` (phase -> :class:`~morphogen.homogenization.Material`; voids
    are left out of the mesh), ``physics`` (``"elastic"``, ``"thermal"`` or both; by
    default those that every material of the phases has), ``conditions``
    (``"periodic"``, by default for a periodic mesh with matching faces, or
    ``"kinematic"``), the mesh in Code_Aster's format (``mesh_format="aster"``) or in MED
    (``"med"``), ``fields`` (the results in MED, ``<name>_fields.rmed``), and the
    parameters of the run in the export file (``memory`` in MB, ``time_limit`` in
    seconds, ``threads``)."""
    from morphogen.io.solvers import periodic_images

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"name: letters, digits, '_', '-' or '.', got {name!r}")
    if mesh_format not in MESH_FORMATS:
        raise ValueError(f"mesh_format: one of {MESH_FORMATS}")
    present = {int(p) for p in np.unique(mesh.phase)}
    missing = sorted(present - {int(p) for p in materials})
    if missing:
        raise ValueError(f"no material for phase(s) {missing}; phases present: {sorted(present)}")
    mats = {int(p): m for p, m in materials.items() if int(p) in present}
    solid = sorted(p for p, m in mats.items() if not m.void)
    if not solid:
        raise ValueError("the phases of the mesh are voids")
    if len(solid) < len(mats):  # the voids out of the mesh
        mesh = mesh.subset(solid)
    if physics is None:
        physics = [
            k
            for k, has in (
                ("elastic", all(mats[p].elastic for p in solid)),
                ("thermal", all(mats[p].thermal for p in solid)),
            )
            if has
        ]
        if not physics:
            raise ValueError("the materials have neither elastic parameters nor conductivities")
    physics = [physics] if isinstance(physics, str) else list(physics)
    if not physics or any(k not in PHYSICS for k in physics):
        raise ValueError(f"physics: some of {PHYSICS}")
    if conditions is None:
        matching = mesh.periodic and mesh.matching_faces()
        conditions = "periodic" if matching else "kinematic"
    if conditions not in CONDITIONS:
        raise ValueError(f"conditions: one of {CONDITIONS}")
    periodic = conditions == "periodic"
    if periodic:
        periodic_images(mesh)  # an error names the faces that do not match
    oriented = [p for p in solid if _oriented(mats[p], physics)]
    grains = []
    if oriented and mesh.cell is not None and mesh.orientations is not None:
        for c in np.unique(mesh.cell):
            own = np.unique(mesh.phase[mesh.cell == c])
            if own.size > 1:
                raise ValueError(f"cell {c} spans the phases {own.tolist()}: a grain, one material")
            if int(own[0]) in oriented:
                grains.append(int(c))
    groups = aster_groups(mesh, periodic=periodic, grains=grains)
    files = []
    if mesh_format == "aster":
        mesh_file = write_aster_mesh(directory / f"{name}.mail", mesh, groups=groups)
    else:
        mesh_file = _write_med(directory / f"{name}.med", mesh, groups)
    files.append(mesh_file)
    lower, upper = mesh.lower, mesh.upper
    length = upper - lower
    volume = float(np.prod(length))
    phases = [f"PHASE_{p}" for p in solid]

    comm = [
        f"# morphoGen {__version__}: the apparent properties of a volume element by Code_Aster",
        f"# box: lower {' '.join(_number(v) for v in lower)}, edges "
        f"{' '.join(_number(v) for v in length)}, volume {_number(volume)}",
        f"# {mesh.n_elements} {'TETRA10' if mesh.order == 2 else 'TETRA4'} elements, "
        f"phases {', '.join(str(p) for p in solid)}; {conditions} conditions",
        "# the tables of the integrals of the stresses (fluxes) of each loading over the elements:",
        "# their sums over the volume of the box are the mean stresses (fluxes), read back by",
        "# morphogen.io.read_code_aster",
        "DEBUT()",
        "",
    ]
    fmt = '"ASTER"' if mesh_format == "aster" else '"MED"'
    extra = [] if mesh_format == "aster" else ['NOM_MED="mesh"']
    comm += _call("LIRE_MAILLAGE", "MESH", [f"FORMAT={fmt}", f"UNITE={_MESH_UNIT}", *extra])
    comm.append("")
    for p in solid:
        comm += _call("DEFI_MATERIAU", f"MAT_{p}", _behaviours(mats[p], p, physics))
    assign = [f'_F(GROUP_MA="PHASE_{p}", MATER=MAT_{p})' for p in solid]
    comm += _call("AFFE_MATERIAU", "MATS", ["MAILLAGE=MESH", f"AFFE={_factors(assign)}"])
    comm.append("")

    cases: dict[str, dict] = {}
    unit = _FIRST_TABLE
    results = []
    for kind in physics:
        thermal = kind == "thermal"
        model = "MODELTH" if thermal else "MODEL"
        phenomenon = "THERMIQUE" if thermal else "MECANIQUE"
        comm += _call(
            "AFFE_MODELE",
            model,
            [
                "MAILLAGE=MESH",
                f'AFFE=_F(GROUP_MA={_tuple(phases)}, PHENOMENE="{phenomenon}", MODELISATION="3D")',
            ],
        )
        cara = None
        if oriented:
            cara = "CARATH" if thermal else "CARA"
            massif = []
            if grains:
                eulers = np.asarray(mesh.orientations, dtype=float)
                massif += [
                    f'_F(GROUP_MA="GRAIN_{c}", ANGL_EULER={_tuple(eulers[c])})' for c in grains
                ]
            else:  # the axes of the crystals are those of the sample
                massif += [f'_F(GROUP_MA="PHASE_{p}", ANGL_REP=(0.0, 0.0, 0.0))' for p in oriented]
            comm += _call("AFFE_CARA_ELEM", cara, [f"MODELE={model}", f"MASSIF={_factors(massif)}"])
        if thermal:  # the instant of the stationary solutions (INCREMENT is required)
            comm += _call("DEFI_LIST_REEL", "TIMES", ["VALE=(0.0,)"])
        comm.append("")
        names = GRADIENT_CASES if thermal else STRAIN_CASES
        for k, case in enumerate(names):
            if thermal:
                g = np.eye(3)[k]
                rows = [g]  # the temperature: one degree of freedom
            else:
                e = _unit_strain(k)
                rows = list(e)  # the components of u
            dofs = ("TEMP",) if thermal else ("DX", "DY", "DZ")
            load = f"LOAD_{case}"
            comm.append(f"# unit {'gradient' if thermal else 'strain'} {case}")
            if periodic:
                fixed = (
                    '_F(GROUP_NO="ORIGIN", TEMP=0.0)'
                    if thermal
                    else '_F(GROUP_NO="ORIGIN", DX=0.0, DY=0.0, DZ=0.0)'
                )
                links = []
                for cls, (shift, _, _) in groups.links.items():
                    jump = shift * length  # x_slave - x_master
                    for dof, row in zip(dofs, rows, strict=True):
                        links.append(
                            f'_F(GROUP_NO_1="PS_{cls}", GROUP_NO_2="PM_{cls}", '
                            f'TRAN={_tuple(-jump)}, DDL_1="{dof}", COEF_MULT_1=1.0, '
                            f'DDL_2="{dof}", COEF_MULT_2=-1.0, '
                            f"COEF_IMPO={_number(float(row @ jump))})"
                        )
                command = "AFFE_CHAR_THER" if thermal else "AFFE_CHAR_MECA"
                imposed = "TEMP_IMPO" if thermal else "DDL_IMPO"
                comm += _call(
                    command,
                    load,
                    [f"MODELE={model}", f"{imposed}={fixed}", f"LIAISON_GROUP={_factors(links)}"],
                )
            else:
                names_f = []
                for dof, row in zip(dofs, rows, strict=True):
                    fname = f"T_{case}" if thermal else f"U{dof[1]}_{case}"
                    comm.append(_formula(fname, row))
                    names_f.append((dof, fname))
                values = ", ".join(f"{dof}={fname}" for dof, fname in names_f)
                command = "AFFE_CHAR_THER_F" if thermal else "AFFE_CHAR_MECA_F"
                imposed = "TEMP_IMPO" if thermal else "DDL_IMPO"
                comm += _call(
                    command,
                    load,
                    [f"MODELE={model}", f'{imposed}=_F(GROUP_NO="BOUNDARY", {values})'],
                )
            resu = f"RESU_{case}"
            args = [f"MODELE={model}", "CHAM_MATER=MATS"]
            if cara:
                args.append(f"CARA_ELEM={cara}")
            args.append(f"EXCIT=_F(CHARGE={load})")
            if thermal:
                comm += _call(
                    "THER_LINEAIRE",
                    resu,
                    [*args, 'TYPE_CALCUL="STAT"', "INCREMENT=_F(LIST_INST=TIMES)"],
                )
                comm += _call(
                    "CALC_CHAMP",
                    resu,
                    [f"reuse={resu}", f"RESULTAT={resu}", 'THERMIQUE=("FLUX_ELGA",)'],
                )
                field_name, components = "FLUX_ELGA", FLUXES
            else:
                comm += _call("MECA_STATIQUE", resu, args)
                field_name, components = "SIEF_ELGA", STRESSES
            table = f"INTE_{case}"
            comm += _call(
                "POST_ELEM",
                table,
                [
                    f"RESULTAT={resu}",
                    f"MODELE={model}",
                    f'INTEGRALE=_F(TOUT="OUI", NOM_CHAM="{field_name}", '
                    f"NOM_CMP={_tuple(components)}, "
                    'TYPE_MAILLE="3D")',
                ],
            )
            comm += _call(
                "IMPR_TABLE", None, [f"TABLE={table}", f"UNITE={unit}", 'FORMAT="TABLEAU"']
            )
            comm.append("")
            cases[case] = {
                "physics": kind,
                "table": f"{name}_{case}.table",
                "unit": unit,
                "load": (g if thermal else e).tolist(),
            }
            results.append(resu)
            unit += 1
    if fields:
        resu = [f"_F(RESULTAT={r})" for r in results]
        comm += _call(
            "IMPR_RESU", None, ['FORMAT="MED"', f"UNITE={_FIELDS_UNIT}", f"RESU={_factors(resu)}"]
        )
        comm.append("")
    comm.append("FIN()")
    comm_file = directory / f"{name}.comm"
    comm_file.write_text("\n".join(comm) + "\n", encoding="ascii")
    files.append(comm_file)

    export = [
        f"P time_limit {time_limit:g}",
        f"P memory_limit {memory:g}",
        f"P ncpus {int(threads)}",
        "P mpi_nbcpu 1",
        f"F comm {comm_file.name} D 1",
        f"F {'mail' if mesh_format == 'aster' else 'mmed'} {mesh_file.name} D {_MESH_UNIT}",
        f"F mess {name}.mess R 6",
        f"F resu {name}.resu R 8",
        *[f"F libr {c['table']} R {c['unit']}" for c in cases.values()],
    ]
    if fields:
        export.append(f"F rmed {name}_fields.rmed R {_FIELDS_UNIT}")
    export_file = directory / f"{name}.export"
    export_file.write_text("\n".join(export) + "\n", encoding="ascii")
    files.append(export_file)
    manifest = {
        "morphogen": __version__,
        "name": name,
        "physics": physics,
        "conditions": conditions,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "volume": volume,
        "phases": solid,
        "elements": int(mesh.n_elements),
        "order": int(mesh.order),
        "mesh": mesh_file.name,
        "cases": cases,
    }
    target = directory / f"{name}_aster.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files.append(target)
    return CodeAsterExport(directory, name, files, cases)


# -- the results ------------------------------------------------------------------------


def read_aster_table(path) -> dict[str, list]:
    """A table written by ``IMPR_TABLE`` in the format ``TABLEAU``: its columns, by name
    (numbers as floats, the other cells as strings). The lines of comments (``#``), the
    lines of dashes and the blank lines are skipped; the line of the names is the first
    that holds an integral (``INTE_``) or the name of a parameter of the tables of
    morphoGen's studies, and the rows are the following lines of as many cells."""
    header: list[str] | None = None
    columns: dict[str, list] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or set(text) <= {"-", " "}:
            continue
        words = text.split()
        if header is None:
            if any(w.startswith(("INTE_", "MOYE_")) for w in words) or "VOL" in words:
                header = words
                columns = {w: [] for w in header}
            continue
        if len(words) != len(header):
            continue
        for key, w in zip(header, words, strict=True):
            try:
                columns[key].append(float(w.replace("D", "E")))
            except ValueError:
                columns[key].append(w)
    if header is None:
        raise ValueError(f"{path}: no table of integrals")
    return columns


@dataclass
class CodeAsterRun:
    """The results of a study (:func:`read_code_aster`): its description, the mean
    stress (Voigt order) or flux of each loading whose table exists, the apparent
    stiffness and conductivity when all their loadings have results, and their
    properties."""

    manifest: dict
    averages: dict[str, np.ndarray] = field(default_factory=dict)
    stiffness: np.ndarray | None = None
    conductivity: np.ndarray | None = None
    properties: dict[str, float] = field(default_factory=dict)


def read_code_aster(directory, name: str | None = None) -> CodeAsterRun:
    """The results of a study written by :func:`write_code_aster` (``<name>_aster.json``
    in ``directory``; the only one there by default): the tables of its loadings, the
    mean stress or flux of each (the sums of its integrals over the volume of the box),
    the apparent stiffness of the unit strains (its column ``k`` the mean stress of the
    strain ``k``) with the bulk modulus (``C_ij`` summed over the normal components, over
    9), the shear modulus (``C66``), the moduli of its isotropic projection and its
    engineering constants, and the apparent conductivity of the unit gradients (minus the
    mean fluxes) with its mean."""
    from morphogen.homogenization.apparent import isotropic_projection
    from morphogen.homogenization.postprocess import engineering_constants

    directory = Path(directory)
    if name is None:
        found = sorted(directory.glob("*_aster.json"))
        if len(found) != 1:
            raise ValueError(f"{directory}: give the name of the study ({len(found)} found)")
        target = found[0]
    else:
        target = directory / f"{name}_aster.json"
    manifest = json.loads(target.read_text(encoding="utf-8"))
    out = CodeAsterRun(manifest=manifest)
    volume = float(manifest["volume"])
    for case, spec in manifest["cases"].items():
        path = directory / spec["table"]
        if not path.exists():
            continue
        table = read_aster_table(path)
        names = FLUXES if spec["physics"] == "thermal" else STRESSES
        out.averages[case] = np.array([float(np.sum(table[f"INTE_{c}"])) / volume for c in names])
    if all(c in out.averages for c in STRAIN_CASES):
        c = np.column_stack([out.averages[k] for k in STRAIN_CASES])
        out.stiffness = 0.5 * (c + c.T)
        bulk, shear = isotropic_projection(out.stiffness)
        out.properties.update(
            bulk_modulus=float(out.stiffness[:3, :3].sum() / 9.0),
            shear_modulus=float(out.stiffness[5, 5]),
            isotropic_bulk=float(bulk),
            isotropic_shear=float(shear),
            **engineering_constants(out.stiffness),
        )
    if all(c in out.averages for c in GRADIENT_CASES):
        k = -np.column_stack([out.averages[g] for g in GRADIENT_CASES])
        out.conductivity = 0.5 * (k + k.T)
        out.properties["conductivity"] = float(np.trace(out.conductivity) / 3.0)
    return out
