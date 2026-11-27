"""Inputs of other programs: periodic boundary conditions for Abaqus and CalculiX,
grids and materials for DAMASK, seeds for Neper.

* :func:`write_periodic_equations` ties every node of the upper faces of a mesh with
  matching opposite faces (conforming tetrahedra or voxels) to its image on the lower
  faces and to three reference nodes, whose displacements (temperatures) are the
  macroscopic strain (gradient) times the edges of the box. Meshes of whole periodic
  cells tie the translates of every node of their boundary alike.
* :func:`write_damask` writes the grid of a polycrystal (``.vti`` with the ``material``
  of each voxel) and its ``material.yaml``: one material per grain, with the quaternion
  of its orientation (DAMASK convention, P = -1) and its phase.
* :func:`write_neper` writes the seeds, weights and orientations of a tessellation, and
  the Neper command that builds it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from morphogen._version import __version__
from morphogen.image import VoxelImage
from morphogen.mesh.tetra import TetMesh
from morphogen.mesh.voxel import VoxelMesh


def _box(mesh) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, TetMesh):
        return np.asarray(mesh.lower, float), np.asarray(mesh.upper, float)
    return mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)


def _upper(mesh, axis: int) -> np.ndarray:
    """Nodes of the upper face along ``axis``."""
    if isinstance(mesh, TetMesh):
        return mesh.face_nodes(axis, 1)
    return np.flatnonzero(mesh.node_ijk[:, axis] == mesh.grid_shape[axis])


def periodic_images(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes of the upper faces (``slaves``), their images on the lower faces
    (``masters``, translated along every axis whose upper face holds the node) and the
    periods between them, ``x_slave = x_master + shift * edges`` (shift: 0 or 1 along
    each axis). Meshes of whole periodic cells: every node of their boundary that is a
    translate of another, ``mesh.periodic_links()``."""
    if not isinstance(mesh, (TetMesh, VoxelMesh)):
        raise TypeError("periodic equations need a TetMesh or a VoxelMesh")
    if isinstance(mesh, TetMesh) and mesh.whole:
        if not mesh.matching_faces():
            raise ValueError(
                "periodic equations need a mesh of all the cells (their boundary made of "
                "pairs of translated faces)"
            )
        return mesh.periodic_links()
    n = mesh.n_nodes
    image = []
    on_upper = []
    for axis in range(3):
        pairs = mesh.periodic_pairs(axis)
        upper = _upper(mesh, axis)
        if len(pairs) != len(upper):
            raise ValueError(
                "periodic equations need matching opposite faces (a periodic conforming "
                "mesh, or a voxel mesh of all the voxels)"
            )
        m = np.full(n, -1, dtype=np.int64)
        m[pairs[:, 1]] = pairs[:, 0]
        image.append(m)
        flag = np.zeros(n, dtype=bool)
        flag[upper] = True
        on_upper.append(flag)
    slaves = np.flatnonzero(on_upper[0] | on_upper[1] | on_upper[2])
    masters = slaves.copy()
    for axis in range(3):
        sel = on_upper[axis][slaves]
        masters[sel] = image[axis][masters[sel]]
        if np.any(masters < 0):  # pragma: no cover - guarded by the face check
            raise ValueError("a node of an upper face has no periodic image")
    return slaves, masters, np.stack([flag[slaves] for flag in on_upper], axis=1).astype(np.int64)


def write_periodic_equations(path, mesh, *, physics: str = "elastic") -> Path:
    """Abaqus/CalculiX include file of periodic boundary conditions (``*INCLUDE`` it
    after the mesh): reference nodes ``REF_X``, ``REF_Y``, ``REF_Z`` (numbered after the
    nodes of the mesh), the node set ``CORNER`` (a node to fix against translation) and
    ``*EQUATION`` blocks ``u(n) - u(image of n) - sum_d s_d u(REF_d) = 0``, with ``s_d``
    the periods from the image to the node along each axis (1 for the upper faces of a
    mesh of the box holding the node; -1, 0 or 1 for whole periodic cells), degrees of
    freedom 1-3, or 11 for ``physics='thermal'``."""
    if physics not in ("elastic", "thermal"):
        raise ValueError("physics: 'elastic' or 'thermal'")
    path = Path(path)
    slaves, masters, shifts = periodic_images(mesh)
    lower, upper = _box(mesh)
    length = upper - lower
    n = mesh.n_nodes
    refs = n + 1 + np.arange(3)  # 1-based ids of the reference nodes
    dofs = (1, 2, 3) if physics == "elastic" else (11,)
    corner = int(np.argmin(np.linalg.norm(mesh.nodes - lower, axis=1)))
    names = ("REF_X", "REF_Y", "REF_Z")
    area = np.prod(length) / length  # faces normal to each axis
    lines = [
        f"** morphoGen {__version__}: periodic boundary conditions ({physics})",
        "** Include after the mesh. Macroscopic strain E (gradient G): impose on REF_<d> the",
        "** displacement E . L_d e_d (temperature G . L_d e_d), and fix CORNER, e.g.",
        (
            f"**   *BOUNDARY: REF_X, 1, 1, E11*{length[0]:.9g}; REF_X, 2, 2, E21*...; ..."
            if physics == "elastic"
            else f"**   *BOUNDARY: REF_X, 11, 11, G1*{length[0]:.9g}; REF_Y, 11, 11, G2*..."
        ),
        (
            "** Macroscopic stress S (the reference nodes are the macroscopic degrees of "
            "freedom):\n** fix the rotation (REF_X 2 and 3, REF_Y 3) and apply on REF_<d> the "
            "force A_d S . e_d,\n** A_d the area of the face normal to d: "
            + ", ".join(f"A_{n} = {a:.9g}" for n, a in zip("xyz", area, strict=True))
            + "; e.g. *CLOAD: REF_X, 1, S11*A_x. Mixed: prescribe either per component."
            if physics == "elastic"
            else "** Macroscopic flux Q: apply on REF_<d> the concentrated flux -A_d Q_d, "
            "A_d the area of the face normal to d: "
            + ", ".join(f"A_{n} = {a:.9g}" for n, a in zip("xyz", area, strict=True))
        ),
        f"** Box: lower {' '.join(f'{v:.9g}' for v in lower)}, edges "
        f"{' '.join(f'{v:.9g}' for v in length)}.",
        "*NODE",
    ]
    for d in range(3):
        x = upper + length * 0.1 * (np.arange(3) == d)  # outside the box, for display
        lines.append(f"{refs[d]}, {x[0]:.12g}, {x[1]:.12g}, {x[2]:.12g}")
    for d, name in enumerate(names):
        lines += [f"*NSET, NSET={name}", str(refs[d])]
    lines += ["*NSET, NSET=CORNER", str(corner + 1)]
    for s, m, shift in zip(slaves, masters, shifts.tolist(), strict=True):
        terms_ref = [(refs[d], -float(shift[d])) for d in range(3) if shift[d]]
        for dof in dofs:
            terms = [(s + 1, dof, 1.0), (m + 1, dof, -1.0)] + [(r, dof, c) for r, c in terms_ref]
            lines += ["*EQUATION", str(len(terms))]
            for start in range(0, len(terms), 4):  # at most four terms per line
                chunk = terms[start : start + 4]
                lines.append(", ".join(f"{a}, {b}, {c:g}" for a, b, c in chunk))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


# -- DAMASK --------------------------------------------------------------------------------


def euler_to_quaternion(euler) -> np.ndarray:
    """Unit quaternions (w, x, y, z), w >= 0, of Bunge Euler angles in degrees, with the
    conventions of DAMASK (passive rotation, P = -1; Rowenhorst et al. 2015)."""
    e = 0.5 * np.radians(np.asarray(euler, dtype=float).reshape(-1, 3))
    c, s = np.cos(e[:, 1]), np.sin(e[:, 1])
    q = np.stack(
        [
            c * np.cos(e[:, 0] + e[:, 2]),
            s * np.cos(e[:, 0] - e[:, 2]),
            s * np.sin(e[:, 0] - e[:, 2]),
            c * np.sin(e[:, 0] + e[:, 2]),
        ],
        axis=1,
    )
    q[q[:, 0] < 0] *= -1.0
    return q


def _yaml_list(values) -> str:
    return "[" + ", ".join(f"{float(v):.12g}" for v in values) + "]"


def write_damask(
    stem,
    grains: VoxelImage,
    orientations,
    grain_phases=None,
    *,
    phase_names: dict | None = None,
    materials: dict | None = None,
) -> list[Path]:
    """DAMASK grid solver inputs: ``<stem>.vti`` (``material`` of each voxel: its grain,
    numbered from 0 over the grains present) and ``<stem>_material.yaml`` (homogenization
    ``SX``, one material per grain with its orientation and phase). ``grain_phases``
    gives the phase of each grain label (default 1); ``phase_names`` the names of the
    phases; ``materials`` (phase -> :class:`~morphogen.homogenization.Material`) fills
    the elastic constants of the crystals (``Hooke``, in the units of the materials:
    DAMASK expects pascals); the plastic behaviour is left to the user."""
    from morphogen.io.vtk import write_vti

    stem = Path(stem)
    labels = np.asarray(grains.data)
    if labels.dtype.kind not in "iu":
        raise ValueError("DAMASK grids need an image of grain labels")
    used, inverse = np.unique(labels, return_inverse=True)
    orientations = np.asarray(orientations, dtype=float)
    if orientations.ndim != 2 or orientations.shape[1] != 3 or len(orientations) <= used.max():
        raise ValueError("orientations: Bunge angles (degrees) of every grain label")
    phases = np.ones(int(used.max()) + 1, dtype=np.int64)
    if grain_phases is not None:
        phases = np.asarray(grain_phases, dtype=np.int64)
    names = {int(p): f"phase_{int(p)}" for p in np.unique(phases[used])}
    names.update({int(k): str(v) for k, v in (phase_names or {}).items()})
    grid = grains.with_data(inverse.reshape(labels.shape).astype(np.int32), name="material")
    vti = write_vti(stem.with_suffix(".vti"), [grid])
    quats = euler_to_quaternion(orientations[used])
    lines = [
        f"# morphoGen {__version__}: DAMASK material configuration ({len(used)} grains)",
        "homogenization:",
        "  SX:",
        "    N_constituents: 1",
        "    mechanical: {type: pass}",
        "phase:",
    ]
    for p, name in names.items():
        lines += [
            f"  {name}:",
            "    lattice: cF  # to be checked: cF, cI, hP, tI...",
            "    mechanical:",
        ]
        lines.append("      output: [F, P, F_e, F_p, L_p, O]")
        mat = (materials or {}).get(p)
        if mat is not None and getattr(mat, "stiffness", None) is not None:
            d = np.array(mat.stiffness)
            keys = {"C_11": d[0, 0], "C_12": d[0, 1], "C_44": d[3, 3]}
            if mat.symmetry == "hexagonal":
                lines[-3] = "    lattice: hP"
                keys.update({"C_13": d[0, 2], "C_33": d[2, 2]})
            inner = ", ".join(f"{k}: {v:.12g}" for k, v in keys.items())
            lines.append(f"      elastic: {{type: Hooke, {inner}}}")
        else:
            lines.append("      elastic: {type: Hooke}  # to be completed")
    lines.append("material:")
    for g, q in zip(used, quats, strict=True):
        lines += [
            "  - homogenization: SX",
            "    constituents:",
            f"      - phase: {names[int(phases[g])]}",
            "        v: 1.0",
            f"        O: {_yaml_list(q)}",
            f"    # grain {int(g)}",
        ]
    yaml_path = stem.parent / f"{stem.name}_material.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [vti, yaml_path]


# -- Neper -----------------------------------------------------------------------------------


def write_neper(stem, tessellation) -> list[Path]:
    """Seeds (``<stem>_seeds.txt``, coordinates in the box), weights (``_weights.txt``:
    the radii of a Laguerre tessellation, sqrt of the morphoGen weights) and orientations
    (``_ori.txt``: Bunge angles in degrees) of a tessellation, and ``<stem>_neper.sh``,
    the command building it with Neper (``-morphooptiini``). Germs outside the box (the
    window of a non-periodic tessellation) have no Neper equivalent: they are left out,
    which changes the cells next to the faces. The grains of seeds, unions of the cells of
    their spheres, are not a tessellation of Neper (their meshes come from morphoGen)."""
    grains = np.asarray(tessellation.cell_grains())
    if not np.array_equal(grains, np.arange(len(grains))):
        raise ValueError(
            "Neper: the grains of seeds are unions of the Laguerre cells of their spheres, "
            "which a tessellation of Neper does not describe (mesh them with morphoGen)"
        )
    stem = Path(stem)
    domain = tessellation.domain
    germs = np.asarray(tessellation.germs, dtype=float) - np.asarray(domain.lower)
    inside = np.all((germs >= 0) & (germs <= np.asarray(domain.lengths)), axis=1)
    files = []
    seeds = stem.parent / f"{stem.name}_seeds.txt"
    np.savetxt(seeds, germs[inside], fmt="%.12g")
    files.append(seeds)
    ini = f"coo:file({seeds.name})"
    if tessellation.weights is not None:
        weights = stem.parent / f"{stem.name}_weights.txt"
        np.savetxt(weights, np.sqrt(np.maximum(tessellation.weights[inside], 0.0)), fmt="%.12g")
        files.append(weights)
        ini += f",weight:file({weights.name})"
    options = [
        "-T",
        f"-n {int(inside.sum())}",
        '-domain "cube({})"'.format(",".join(f"{v:.12g}" for v in domain.lengths)),
        f'-morphooptiini "{ini}"',
    ]
    if domain.periodic:
        options.append("-periodicity all")
    if tessellation.orientations is not None:
        ori = stem.parent / f"{stem.name}_ori.txt"
        np.savetxt(ori, np.asarray(tessellation.orientations)[inside], fmt="%.9g")
        files.append(ori)
        options.append(f'-ori "file({ori.name},des=euler-bunge)"')
    options.append(f"-o {stem.name}")
    script = stem.parent / f"{stem.name}_neper.sh"
    lost = int((~inside).sum())
    note = f"# {lost} germs outside the box left out\n" if lost else ""
    script.write_text(
        f"#!/bin/sh\n# morphoGen {__version__}: the tessellation with Neper (shifted to the "
        f"origin)\n{note}neper " + " \\\n  ".join(options) + "\n",
        encoding="utf-8",
    )
    files.append(script)
    return files


# -- FEniCSx ---------------------------------------------------------------------------------


def _xml_data(values: np.ndarray, number: str, precision: int) -> str:
    values = np.asarray(values)
    fmt = "%d" if number == "Int" else "%.17g"
    rows = "\n".join(" ".join(fmt % v for v in row) for row in values.reshape(len(values), -1))
    shape = " ".join(str(s) for s in values.shape)
    return (
        f'<DataItem Dimensions="{shape}" NumberType="{number}" Precision="{precision}" '
        f'Format="XML">\n{rows}\n</DataItem>'
    )


def write_fenicsx(path, mesh: TetMesh) -> Path:
    """XDMF file of a tetrahedral mesh and its tags, as written and read by FEniCSx
    (``dolfinx.io.XDMFFile``, ASCII encoding): the grid ``mesh`` (linear tetrahedra), the
    cell tags ``phase`` (and ``grain`` for polycrystals) and the facet tags ``faces``
    (the faces of the box: 1-6 for x0, x1, y0, y1, z0, z1; 1 for the faces bounding
    whole periodic cells). Reading::

        with XDMFFile(MPI.COMM_WORLD, path, "r") as f:
            mesh = f.read_mesh(name="mesh")
            phase = f.read_meshtags(mesh, name="phase")
            mesh.topology.create_connectivity(2, 3)
            faces = f.read_meshtags(mesh, name="faces")
    """
    path = Path(path)
    lin = mesh.linear()
    n = lin.n_elements
    topology = _xml_data(lin.elements, "Int", 8)
    tri, refs = [], []
    if lin.whole:
        tri.append(lin.outer_faces())
        refs.append(np.ones(len(tri[0]), dtype=np.int64))
    else:
        for axis in range(3):
            for side in (0, 1):
                faces = lin.boundary_faces(axis, side)
                tri.append(faces[:, :3])
                refs.append(np.full(len(faces), 1 + 2 * axis + side))
    tri, refs = np.concatenate(tri), np.concatenate(refs)
    geometry_ref = "<xi:include xpointer=\"xpointer(/Xdmf/Domain/Grid[@Name='mesh']/Geometry)\"/>"

    def tags(name, topo_type, cells, values):
        return (
            f'    <Grid Name="{name}" GridType="Uniform">\n'
            f"      {geometry_ref}\n"
            f'      <Topology TopologyType="{topo_type}" NumberOfElements="{len(cells)}" '
            f'NodesPerElement="{cells.shape[1]}">\n'
            f"{_xml_data(cells, 'Int', 8)}\n"
            "      </Topology>\n"
            f'      <Attribute Name="{name}" AttributeType="Scalar" Center="Cell">\n'
            f"{_xml_data(values, 'Int', 4)}\n"
            "      </Attribute>\n"
            "    </Grid>\n"
        )

    grids = [tags("phase", "Tetrahedron", lin.elements, lin.phase)]
    if lin.cell is not None:
        grids.append(tags("grain", "Tetrahedron", lin.elements, lin.cell))
    grids.append(tags("faces", "Triangle", tri, refs))
    text = (
        '<?xml version="1.0"?>\n'
        '<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n'
        "  <Domain>\n"
        '    <Grid Name="mesh" GridType="Uniform">\n'
        f'      <Topology TopologyType="Tetrahedron" NumberOfElements="{n}" NodesPerElement="4">\n'
        f"{topology}\n"
        "      </Topology>\n"
        '      <Geometry GeometryType="XYZ">\n'
        f"{_xml_data(lin.nodes, 'Float', 8)}\n"
        "      </Geometry>\n"
        "    </Grid>\n" + "".join(grids) + "  </Domain>\n</Xdmf>\n"
    )
    path.write_text(text, encoding="ascii")
    return path
