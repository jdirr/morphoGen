"""Cases of OpenFOAM, the finite volume code (of the OpenFOAM Foundation,
https://openfoam.org, and of OpenCFD, https://www.openfoam.com), on the voxel images and
the conforming meshes of morphoGen: the steady flow through the pores (Darcy's
permeability) and the steady diffusion through a phase (its effective diffusivity, or
conductivity), and their results.

:func:`polymesh` makes the mesh of OpenFOAM (``constant/polyMesh``) of the voxels of some
phases of an image (a hexahedral cell per voxel) or of the elements of some phases of a
conforming tetrahedral mesh (a cell per element):

* the faces, the internal ones first, by owner then by neighbour (the upper-triangular
  order), the owner the lower cell and the normal (the order of the points, right-handed)
  from the owner to the neighbour; then the boundary faces patch by patch, their normals
  outward;
* the patches: the six faces of the box, ``XMIN``, ``XMAX``... ``ZMAX`` (``OUTER``, the
  boundary of whole periodic cells), of type ``patch``, ``symmetryPlane`` or, in pairs
  along the axes given, ``cyclic``; then ``WALLS``, the faces between the cells and the
  voxels (elements) of the other phases, of type ``wall``. The face ``k`` of a cyclic
  patch faces the face ``k`` of its neighbour, translated by the edge of the box, its
  points in the reverse order from the same first point, as OpenFOAM matches them, and
  the patch gives ``transform translational`` and ``separationVector``, its centre minus
  that of its neighbour (the names that both lines of OpenFOAM read);
* the cell zones, one per phase (``phase<p>``).

:func:`write_polymesh` writes its files (``points``, ``faces``, ``owner``, ``neighbour``,
``boundary``, ``cellZones``; FoamFile headers, the counts of the mesh in the note of
``owner`` and ``neighbour``, the cell zones of the class ``dictionary``, which every line
of OpenFOAM reads) and :func:`read_polymesh` reads them back.

:func:`write_openfoam` writes a case directory of a sample, its flow axis from the face
``<axis>MIN`` (the inlet) to ``<axis>MAX`` (the outlet), the other faces symmetry planes
or cyclic pairs: ``kind="flow"``, the steady flow of a Newtonian fluid of kinematic
viscosity ``nu`` through the pores under the pressure drop ``dp`` (kinematic pressure,
``p / rho``) between the inlet and the outlet, no slip on the walls (``simpleFoam``; the
solver ``incompressibleFluid`` of ``foamRun`` in the versions of the Foundation that have
it), and the flow rate through the outlet (the function object ``surfaceFieldValue``,
``sum(phi)``); ``kind="diffusion"``, the steady diffusion of a scalar ``T`` of
diffusivity ``DT`` through the phase, ``T = dT`` at the inlet and 0 at the outlet, no flux
through the walls (``laplacianFoam``), and the integral of its gradient over the cells
(the function objects ``grad`` and ``volFieldValue``, ``volIntegrate``). It writes
``system/controlDict``, ``fvSchemes``, ``fvSolution``, the properties of the fluid in the
files of both lines (``constant/transportProperties``, ``physicalProperties``,
``turbulenceProperties`` and ``momentumTransport``: each reads its own), the fields of
``0/``, ``Allrun``, an empty ``<case>.foam`` for ParaView and ``morphogen_openfoam.json``,
the description of the case. The components of the phase that do not join the inlet to
the outlet (closed pores, where the pressure or ``T`` would be undetermined) are left out
by default (``connected``).

:func:`read_openfoam` reads the last values written by the function objects
(``postProcessing/``): Darcy's permeability ``k = nu Q L / (A dp)`` (``Q`` the flow rate,
``L`` the length of the box along the axis, ``A`` the area of its section: the mean
velocity over the whole section is ``(k / nu) dp / L``), or the effective diffusivity
``D_eff = -DT I L / (V dT)`` (``I`` the integral of the gradient along the axis over the
cells, ``V`` the volume of the box: the mean flux over the box is ``-DT I / V``), and its
ratio to ``DT``, the inverse of the formation factor of the phase.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from morphogen._version import __version__
from morphogen.image import VoxelImage
from morphogen.mesh.tetra import TET_FACES, TetMesh

FACES = ("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX")
WALLS = "WALLS"
KINDS = ("flow", "diffusion")
LATERAL = ("symmetryPlane", "cyclic")
_BANNER = (
    "/*--------------------------------*- C++ -*----------------------------------*\\\n"
    f"  morphoGen {__version__}\n"
    "\\*---------------------------------------------------------------------------*/\n"
)
_RULE = "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"


@dataclass
class Patch:
    """A patch of the boundary of a mesh of OpenFOAM: its name, type, first face and number
    of faces, and the other entries of its dictionary (those of cyclic patches)."""

    name: str
    type: str
    start: int
    size: int
    entries: dict = field(default_factory=dict)


@dataclass
class PolyMesh:
    """A mesh of OpenFOAM (:func:`polymesh`): its points, its faces (rows of point
    indices, all of three or all of four points), the owner of each face and the
    neighbour of each internal face, the patches of the boundary faces and the cell
    zones."""

    points: np.ndarray
    faces: np.ndarray
    owner: np.ndarray
    neighbour: np.ndarray
    patches: list[Patch]
    zones: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(max(self.owner.max(initial=-1), self.neighbour.max(initial=-1)) + 1)

    @property
    def n_internal(self) -> int:
        return len(self.neighbour)

    def patch(self, name: str) -> Patch:
        for p in self.patches:
            if p.name == name:
                return p
        raise KeyError(f"no patch {name!r}; patches: {[p.name for p in self.patches]}")

    def face_areas(self) -> np.ndarray:
        """The area vectors of the faces (their normals by the order of their points)."""
        x = self.points[self.faces]
        centre = x.mean(axis=1)
        nxt = np.roll(x, -1, axis=1)
        return 0.5 * np.cross(x - centre[:, None], nxt - centre[:, None]).sum(axis=1)

    def face_centres(self) -> np.ndarray:
        """The centres of the faces (exact for the planar triangles and quadrangles of the
        meshes of morphoGen: the mean of their points)."""
        return self.points[self.faces].mean(axis=1)

    def cell_volumes(self) -> np.ndarray:
        """The volumes of the cells by the divergence theorem, ``(1/3) sum x_f . S_f`` over
        their faces, outward."""
        s = self.face_areas()
        flux = np.einsum("ij,ij->i", self.face_centres(), s) / 3.0
        out = np.zeros(self.n_cells)
        np.add.at(out, self.owner, flux)
        np.add.at(out, self.neighbour, -flux[: self.n_internal])
        return out


def _axes(axis: int) -> tuple[int, int]:
    """The two other axes, ``(b, c)``, with ``e_b x e_c = e_axis``."""
    return (axis + 1) % 3, (axis + 2) % 3


def _cells_of(kept: np.ndarray) -> np.ndarray:
    """The cell of each voxel kept (-1 elsewhere), numbered with x varying fastest."""
    flat = kept.transpose(2, 1, 0).ravel()
    ids = np.full(flat.shape, -1, dtype=np.int64)
    ids[flat] = np.arange(int(flat.sum()))
    return ids.reshape(kept.shape[::-1]).transpose(2, 1, 0)


def _voxel_face(axis: int, plane: np.ndarray, b: np.ndarray, c: np.ndarray, shape, outward: int):
    """The points (node indices of the grid of the voxels) of faces normal to ``axis`` at the
    node planes ``plane``, of corner ``(b, c)`` along the two other axes, their normal along
    ``outward`` times the axis."""
    nx, ny = shape[0] + 1, shape[1] + 1
    ax_b, ax_c = _axes(axis)

    def node(db, dc):
        idx = [None, None, None]
        idx[axis] = plane
        idx[ax_b] = b + db
        idx[ax_c] = c + dc
        return idx[0] + nx * (idx[1] + ny * idx[2])

    if outward > 0:  # (b0 c0) (b1 c0) (b1 c1) (b0 c1): e_b x e_c = +axis
        return np.stack([node(0, 0), node(1, 0), node(1, 1), node(0, 1)], axis=1)
    return np.stack([node(0, 0), node(0, 1), node(1, 1), node(1, 0)], axis=1)


def _flowing(kept: np.ndarray, axis: int, cyclic: tuple[int, ...]) -> np.ndarray:
    """The voxels kept that belong to a component (face neighbours, through the faces of
    the cyclic axes) joining the face ``axis = 0`` to the opposite one."""
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    labels, n = ndimage.label(kept)
    if n == 0:
        return kept
    rows, cols = [], []
    for a in cyclic:  # components joined across the periodic faces
        lo, hi = np.take(labels, 0, axis=a), np.take(labels, -1, axis=a)
        both = (lo > 0) & (hi > 0)
        rows.append(lo[both])
        cols.append(hi[both])
    if rows:
        r, c = np.concatenate(rows), np.concatenate(cols)
        graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(n + 1, n + 1))
        _, merged = connected_components(graph, directed=False)
    else:
        merged = np.arange(n + 1)
    first = np.unique(merged[np.take(labels, 0, axis=axis)[np.take(labels, 0, axis=axis) > 0]])
    last = np.unique(merged[np.take(labels, -1, axis=axis)[np.take(labels, -1, axis=axis) > 0]])
    joined = np.intersect1d(first, last)
    return kept & np.isin(merged[labels], joined) & (labels > 0)


def _voxel_polymesh(image: VoxelImage, phases, cyclic, connected_axis):
    data = np.asarray(image.data)
    kept = np.isin(data, list(phases))
    if connected_axis is not None:
        kept = _flowing(kept, connected_axis, tuple(cyclic))
    if not kept.any():
        raise ValueError(
            f"no voxel of the phases {list(phases)} (that joins the inlet to the outlet)"
        )
    shape = kept.shape
    cell = _cells_of(kept)
    internal_o, internal_n, internal_f = [], [], []
    boundary: dict[str, list] = {name: [] for name in FACES}
    walls_o, walls_f = [], []
    for axis in range(3):
        ax_b, ax_c = _axes(axis)
        n_a = shape[axis]
        # the faces between the layers i and i + 1 along the axis, at the node plane i + 1:
        # internal between two cells, walls between a cell and a voxel of another phase
        lo = np.take(kept, np.arange(n_a - 1), axis=axis)
        hi = np.take(kept, np.arange(1, n_a), axis=axis)
        c_lo = np.take(cell, np.arange(n_a - 1), axis=axis)
        c_hi = np.take(cell, np.arange(1, n_a), axis=axis)
        idx = np.nonzero(lo & hi)
        if idx[0].size:
            internal_o.append(c_lo[idx])
            internal_n.append(c_hi[idx])
            internal_f.append(_voxel_face(axis, idx[axis] + 1, idx[ax_b], idx[ax_c], shape, 1))
        for mask, owner, sign in ((lo & ~hi, c_lo, 1), (~lo & hi, c_hi, -1)):
            idx = np.nonzero(mask)
            if idx[0].size:
                walls_o.append(owner[idx])
                walls_f.append(_voxel_face(axis, idx[axis] + 1, idx[ax_b], idx[ax_c], shape, sign))
        # the faces of the box: a patch, or along a cyclic axis a cyclic face where the
        # voxel across the periodic face is kept too and a wall where it is not
        others = [d for d in range(3) if d != axis]
        first, last = np.take(kept, 0, axis=axis), np.take(kept, n_a - 1, axis=axis)
        c_first, c_last = np.take(cell, 0, axis=axis), np.take(cell, n_a - 1, axis=axis)
        for side, here, there, owner in ((0, first, last, c_first), (1, last, first, c_last)):
            opened = here & there if axis in cyclic else here
            shut = here & ~there if axis in cyclic else np.zeros_like(here)
            for mask, target in ((opened, boundary[FACES[2 * axis + side]]), (shut, None)):
                jk = np.nonzero(mask)
                if not jk[0].size:
                    continue
                full = [None, None, None]
                full[others[0]], full[others[1]] = jk
                plane = np.full(jk[0].size, 0 if side == 0 else n_a, dtype=np.int64)
                faces = _voxel_face(axis, plane, full[ax_b], full[ax_c], shape, 1 if side else -1)
                # in the order of their position on the face: cyclic pairs face each other
                order = np.argsort(full[ax_c] * shape[ax_b] + full[ax_b], kind="stable")
                if target is None:
                    walls_o.append(owner[jk][order])
                    walls_f.append(faces[order])
                else:
                    target.append((owner[jk][order], faces[order]))
    o = np.concatenate(internal_o) if internal_o else np.zeros(0, np.int64)
    nb = np.concatenate(internal_n) if internal_n else np.zeros(0, np.int64)
    f = np.concatenate(internal_f) if internal_f else np.zeros((0, 4), np.int64)
    order = np.lexsort((nb, o))
    groups = [(name, *_joined(items, 4)) for name, items in boundary.items()]
    groups.append((WALLS, *_joined(list(zip(walls_o, walls_f, strict=True)), 4)))
    spacing = np.asarray(image.spacing, dtype=float)
    origin = np.asarray(image.origin, dtype=float)
    grid = np.array(np.meshgrid(*[np.arange(n + 1) for n in shape], indexing="ij"))
    points = origin + grid.reshape(3, -1, order="F").T * spacing
    zones = {
        f"phase{int(p)}": np.sort(cell[(data == p) & kept])
        for p in sorted(phases)
        if np.any((data == p) & kept)
    }
    return points, o[order], nb[order], f[order], groups, zones, kept


def _joined(items, k: int):
    if not items:
        return np.zeros(0, np.int64), np.zeros((0, k), np.int64)
    return np.concatenate([a for a, _ in items]), np.concatenate([b for _, b in items])


def _tet_polymesh(mesh: TetMesh, phases, cyclic):
    if mesh.order != 1:
        mesh = mesh.linear()
    keep = np.isin(mesh.phase, list(phases))
    if not keep.any():
        raise ValueError(f"no element of the phases {list(phases)}")
    elements = mesh.elements[keep]
    phase = mesh.phase[keep]
    n_cells = len(elements)
    tri = elements[:, TET_FACES].reshape(-1, 3)
    cell_of = np.repeat(np.arange(n_cells), 4)
    key = np.sort(tri, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    if counts.max() > 2:
        raise ValueError("a face of more than two elements: not a conforming mesh")
    order = np.argsort(inverse, kind="stable")
    inv_sorted = inverse[order]
    starts = np.flatnonzero(np.r_[True, inv_sorted[1:] != inv_sorted[:-1]])
    pairs = counts[inv_sorted[starts]] == 2
    a = order[starts[pairs]]
    b = order[starts[pairs] + 1]
    owner = np.minimum(cell_of[a], cell_of[b])
    neighbour = np.maximum(cell_of[a], cell_of[b])
    faces = tri[a]
    centres = mesh.nodes[elements].mean(axis=1)
    x = mesh.nodes
    normal = np.cross(x[faces[:, 1]] - x[faces[:, 0]], x[faces[:, 2]] - x[faces[:, 0]])
    flip = np.einsum("ij,ij->i", normal, centres[neighbour] - centres[owner]) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    srt = np.lexsort((neighbour, owner))
    owner, neighbour, faces = owner[srt], neighbour[srt], faces[srt]
    single = order[starts[~pairs]]
    b_faces, b_owner = tri[single], cell_of[single]
    normal = np.cross(x[b_faces[:, 1]] - x[b_faces[:, 0]], x[b_faces[:, 2]] - x[b_faces[:, 0]])
    out = np.einsum("ij,ij->i", normal, x[b_faces].mean(axis=1) - centres[b_owner]) < 0
    b_faces[out] = b_faces[out][:, [0, 2, 1]]
    tol = mesh.tolerance()
    placed = np.full(len(b_faces), -1)
    if not mesh.whole:
        for axis in range(3):
            for side in (0, 1):
                target = mesh.lower[axis] if side == 0 else mesh.upper[axis]
                on = np.all(np.abs(x[b_faces][:, :, axis] - target) <= tol, axis=1)
                placed[on & (placed < 0)] = 2 * axis + side
    groups = []
    names = ["OUTER"] if mesh.whole else list(FACES)
    for k, name in enumerate(names):
        sel = np.flatnonzero(placed == k) if not mesh.whole else np.zeros(0, np.int64)
        groups.append((name, b_owner[sel], b_faces[sel]))
    walls = np.flatnonzero(placed < 0)
    if mesh.whole:  # the faces bounding whole cells, and the walls of the phases left out
        outer = _outer_triangles(mesh, b_faces[walls])
        groups[0] = ("OUTER", b_owner[walls[outer]], b_faces[walls[outer]])
        walls = walls[~outer]
    groups.append((WALLS, b_owner[walls], b_faces[walls]))
    for axis in cyclic:  # the faces of the upper patch in the order of their images below
        if mesh.whole:
            raise ValueError(
                "cyclic patches: a mesh of the box (whole cells have no faces of the box)"
            )
        lo_name, hi_name = FACES[2 * axis], FACES[2 * axis + 1]
        lo = next(g for g in groups if g[0] == lo_name)
        hi = next(g for g in groups if g[0] == hi_name)
        hi_owner, hi_faces = _matched(x, lo[2], hi[1], hi[2], axis, mesh.size[axis], tol)
        groups[[g[0] for g in groups].index(hi_name)] = (hi_name, hi_owner, hi_faces)
    zones = {f"phase{int(p)}": np.flatnonzero(phase == p) for p in np.unique(phase)}
    return x, owner, neighbour, faces, groups, zones


def _outer_triangles(mesh: TetMesh, triangles: np.ndarray) -> np.ndarray:
    """Which free faces of a mesh of some phases of whole periodic cells bound the cells
    (its ``outer`` faces), not the phases left out."""
    outer = mesh.outer_faces()[:, :3]
    key = {tuple(sorted(f)) for f in outer.tolist()}
    return np.array([tuple(sorted(f)) in key for f in triangles.tolist()], dtype=bool)


def _matched(x, lo_faces, hi_owner, hi_faces, axis, length, tol):
    """The faces of an upper patch reordered to face those of the lower one (``k`` facing
    ``k``), each with its points in the reverse order from the image of the first point of
    its partner."""
    from scipy.spatial import cKDTree

    if len(lo_faces) != len(hi_faces):
        raise ValueError(
            f"cyclic patches along {'xyz'[axis]}: {len(lo_faces)} and {len(hi_faces)} faces "
            "(the opposite faces of the mesh must match)"
        )
    shift = np.zeros(3)
    shift[axis] = length
    lo_c = x[lo_faces].mean(axis=1) + shift
    dist, near = cKDTree(x[hi_faces].mean(axis=1)).query(lo_c)
    if dist.size and (dist.max() > 1e3 * tol or len(np.unique(near)) != len(near)):
        raise ValueError(f"cyclic patches along {'xyz'[axis]}: the opposite faces do not match")
    faces = hi_faces[near]
    out = np.empty_like(faces)
    m = faces.shape[1]
    reverse = [0, *range(m - 1, 0, -1)]  # the points of a face in the reverse order
    for k, (a, b) in enumerate(zip(lo_faces, faces, strict=True)):
        image = x[a] + shift
        j = np.argmin(np.linalg.norm(x[b] - image[0], axis=1))
        out[k] = [b[(j + i) % m] for i in range(m)]  # outward both: a rotation of its points
        if np.abs(x[out[k]] - image[reverse]).max() > 1e3 * tol:
            raise ValueError(f"cyclic patches along {'xyz'[axis]}: faces of other shapes")
    return hi_owner[near], out


def polymesh(
    sample,
    phases=(0,),
    *,
    cyclic=(),
    symmetry=(),
    connected_axis: int | None = None,
) -> PolyMesh:
    """The mesh of OpenFOAM of the voxels (a :class:`~morphogen.image.VoxelImage`) or of the
    elements (a :class:`~morphogen.mesh.tetra.TetMesh`) of ``phases`` (see the module):
    ``cyclic``, the axes whose faces are cyclic pairs (a periodic sample; a conforming mesh
    with matching opposite faces); ``symmetry``, those whose faces are symmetry planes (the
    others are patches); ``connected_axis`` (voxels): only the components that join the
    two faces normal to this axis (through the cyclic faces)."""
    cyclic = tuple(sorted({int(a) for a in cyclic}))
    symmetry = tuple(sorted({int(a) for a in symmetry}))
    if set(cyclic) & set(symmetry):
        raise ValueError("an axis is either cyclic or a symmetry, not both")
    if isinstance(sample, VoxelImage):
        points, owner, neighbour, faces, groups, zones, _ = _voxel_polymesh(
            sample, phases, cyclic, connected_axis
        )
        lengths = np.asarray(sample.spacing, dtype=float) * np.asarray(sample.shape)
    elif isinstance(sample, TetMesh):
        points, owner, neighbour, faces, groups, zones = _tet_polymesh(sample, phases, cyclic)
        lengths = sample.size
    else:
        raise TypeError("polymesh: a VoxelImage or a TetMesh")
    all_faces = [faces]
    all_owner = [owner]
    patches = []
    start = len(faces)
    for name, own, fc in groups:
        kind = "wall" if name == WALLS else "patch"
        entries = {}
        if name in FACES:
            axis = FACES.index(name) // 2
            if axis in symmetry:
                kind = "symmetryPlane"
            elif axis in cyclic:
                kind = "cyclic"
                lower = FACES.index(name) % 2 == 0
                partner = FACES[2 * axis + (1 if lower else 0)]
                separation = np.zeros(3)
                separation[axis] = -lengths[axis] if lower else lengths[axis]
                entries = {
                    "inGroups": "List<word> 1(cyclic)",
                    "matchTolerance": "0.0001",
                    "transform": "translational",
                    "neighbourPatch": partner,
                    "separationVector": "(" + " ".join(f"{v:.12g}" for v in separation) + ")",
                }
        patches.append(Patch(name, kind, start, len(fc), entries))
        all_faces.append(fc)
        all_owner.append(own)
        start += len(fc)
    faces = np.concatenate(all_faces).astype(np.int64)
    owner = np.concatenate(all_owner).astype(np.int64)
    used, inverse = np.unique(faces, return_inverse=True)  # the points of the faces only
    return PolyMesh(
        points=np.asarray(points, dtype=float)[used],
        faces=inverse.reshape(faces.shape),
        owner=owner,
        neighbour=np.asarray(neighbour, dtype=np.int64),
        patches=patches,
        zones=zones,
    )


# -- the files -----------------------------------------------------------------------


def _header(cls: str, obj: str, location: str | None = None, note: str | None = None) -> str:
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        f"    class       {cls};",
    ]
    if note:
        lines.append(f'    note        "{note}";')
    if location:
        lines.append(f'    location    "{location}";')
    lines += [f"    object      {obj};", "}"]
    return _BANNER + "\n".join(lines) + "\n" + _RULE + "\n"


def _write_list(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text(
        header + f"{len(rows)}\n(\n" + "\n".join(rows) + ("\n" if rows else "") + ")\n\n" + _RULE,
        encoding="ascii",
    )
    return path


def write_polymesh(directory, mesh: PolyMesh) -> list[Path]:
    """The files of a mesh of OpenFOAM in ``directory/constant/polyMesh``."""
    folder = Path(directory) / "constant" / "polyMesh"
    folder.mkdir(parents=True, exist_ok=True)
    location = "constant/polyMesh"
    note = (
        f"nPoints:{len(mesh.points)}  nCells:{mesh.n_cells}  nFaces:{len(mesh.faces)}  "
        f"nInternalFaces:{mesh.n_internal}"
    )
    files = [
        _write_list(
            folder / "points",
            _header("vectorField", "points", location),
            [f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in mesh.points],
        ),
        _write_list(
            folder / "faces",
            _header("faceList", "faces", location),
            [f"{len(f)}({' '.join(str(v) for v in f)})" for f in mesh.faces.tolist()],
        ),
        _write_list(
            folder / "owner",
            _header("labelList", "owner", location, note),
            [str(v) for v in mesh.owner.tolist()],
        ),
        _write_list(
            folder / "neighbour",
            _header("labelList", "neighbour", location, note),
            [str(v) for v in mesh.neighbour.tolist()],
        ),
    ]
    entries = []
    for p in mesh.patches:
        body = [f"    {p.name}", "    {", f"        type            {p.type};"]
        if p.type == "wall":
            body.append("        inGroups        List<word> 1(wall);")
        for key in ("inGroups", "matchTolerance", "transform", "neighbourPatch"):
            if key in p.entries:
                body.append(f"        {key:<15} {p.entries[key]};")
        body += [f"        nFaces          {p.size};", f"        startFace       {p.start};"]
        if "separationVector" in p.entries:
            body.append(f"        separationVector {p.entries['separationVector']};")
        body.append("    }")
        entries.append("\n".join(body))
    files.append(
        _write_list(folder / "boundary", _header("polyBoundaryMesh", "boundary", location), entries)
    )
    zones = []
    for name, cells in mesh.zones.items():
        labels = "\n".join(str(v) for v in np.asarray(cells).tolist())
        zones.append(
            f"{name}\n{{\n    type cellZone;\ncellLabels      List<label> {len(cells)}\n"
            f"(\n{labels}\n)\n;\n}}"
        )
    files.append(
        _write_list(folder / "cellZones", _header("dictionary", "cellZones", location), zones)
    )
    return files


def _body(path: Path) -> str:
    """The text of a file of OpenFOAM without its comments and its FoamFile header."""
    text = Path(path).read_text(encoding="ascii")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"FoamFile\s*\{.*?\}", "", text, count=1, flags=re.S)


def _list(path: Path) -> tuple[str, int]:
    """The text inside the outer list of a file of OpenFOAM, and its declared length."""
    body = _body(path)
    head = re.search(r"(\d+)\s*\(", body)
    if head is None:
        raise ValueError(f"{path}: no list")
    inner = body[head.end() : body.rindex(")")]
    return inner, int(head.group(1))


def read_polymesh(directory) -> PolyMesh:
    """A mesh of OpenFOAM (``directory/constant/polyMesh``, ASCII files) as
    :func:`write_polymesh` writes it."""
    folder = Path(directory) / "constant" / "polyMesh"
    inner, n = _list(folder / "points")
    rows = re.findall(r"\(([^()]*)\)", inner)
    points = np.array([[float(v) for v in p.split()] for p in rows]).reshape(n, 3)
    inner, n = _list(folder / "faces")
    faces = [[int(v) for v in f.split()] for f in re.findall(r"\d+\(([^()]*)\)", inner)]
    faces = np.array(faces, dtype=np.int64)
    owner = np.array(_list(folder / "owner")[0].split(), dtype=np.int64)
    neighbour = np.array(_list(folder / "neighbour")[0].split(), dtype=np.int64)
    inner, _ = _list(folder / "boundary")
    patches = []
    for name, block in re.findall(r"(\w+)\s*\{([^{}]*)\}", inner):
        entries = dict(re.findall(r"(\w+)\s+([^;]+);", block))
        patches.append(
            Patch(
                name,
                entries.pop("type"),
                int(entries.pop("startFace")),
                int(entries.pop("nFaces")),
                entries,
            )
        )
    zones = {}
    zone_file = folder / "cellZones"
    if zone_file.exists():
        inner, _ = _list(zone_file)
        for name, block in re.findall(r"(\w+)\s*\{(.*?)\}", inner, flags=re.S):
            values = re.search(r"List<label>\s*\d+\s*\(([^()]*)\)", block)
            zones[name] = np.array(values.group(1).split() if values else [], dtype=np.int64)
    return PolyMesh(points, faces, owner, neighbour, patches, zones)


# -- the cases -------------------------------------------------------------------------


@dataclass
class OpenFoamExport:
    """The files of a case of OpenFOAM (:func:`write_openfoam`): its directory, kind, mesh
    and files, and its description (``manifest``)."""

    directory: Path
    kind: str
    mesh: PolyMesh
    files: list[Path]
    manifest: dict


def _dictionary(cls: str, obj: str, body: str, location: str | None = None) -> str:
    return _header(cls, obj, location) + body.strip("\n") + "\n\n" + _RULE


def _field(cls, obj, dims, internal, conditions: dict) -> str:
    entries = "\n".join(
        f"    {name}\n    {{\n" + "".join(f"        {line}\n" for line in lines) + "    }"
        for name, lines in conditions.items()
    )
    body = (
        f"dimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n"
        f"{entries}\n}}"
    )
    return _dictionary(cls, obj, body, "0")


_FUNCTIONS = {
    "flow": """
    flowRate
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        log             true;
        writeFields     false;
        regionType      patch;
        name            {outlet};
        select          patch;
        patch           {outlet};
        operation       sum;
        fields          (phi);
    }}
""",
    "diffusion": """
    gradT
    {{
        type            grad;
        libs            ("libfieldFunctionObjects.so");
        field           T;
        executeControl  timeStep;
        writeControl    writeTime;
    }}
    gradientIntegral
    {{
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        log             true;
        writeFields     false;
        regionType      all;
        select          all;
        operation       volIntegrate;
        fields          (grad(T));
    }}
""",
}


def write_openfoam(
    directory,
    sample,
    *,
    kind: str = "flow",
    phases=(0,),
    axis: int = 0,
    lateral: str | None = None,
    viscosity: float = 1.0,
    pressure_drop: float = 1.0,
    diffusivity: float = 1.0,
    difference: float = 1.0,
    iterations: int = 2000,
    tolerance: float = 1e-7,
    connected: bool = True,
) -> OpenFoamExport:
    """A case of OpenFOAM of a voxel image or a conforming mesh (see the module): ``kind``
    ``"flow"`` (the pores, ``phases``: by default the phase 0) or ``"diffusion"``, along
    ``axis`` from the face ``<axis>MIN`` to ``<axis>MAX``; ``lateral``, the faces of the
    other axes: ``"symmetryPlane"`` or ``"cyclic"`` (by default for a periodic sample);
    ``viscosity`` (kinematic) and ``pressure_drop`` (kinematic) of the flow;
    ``diffusivity`` and ``difference`` (of ``T`` between the inlet and the outlet) of the
    diffusion; ``iterations`` (the largest number of iterations of SIMPLE) and
    ``tolerance`` (the residuals at which they stop); ``connected``: only the components
    of the phases that join the inlet to the outlet (voxels). The lengths are those of the
    sample, in metres for OpenFOAM."""
    if kind not in KINDS:
        raise ValueError(f"kind: one of {KINDS}")
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis: 0, 1 or 2")
    periodic = bool(getattr(sample, "periodic", False))
    if isinstance(sample, TetMesh):
        if sample.whole:
            raise ValueError(
                "write_openfoam: a mesh of the box (whole periodic cells have no inlet and outlet "
                "faces); mesh it with cells='box'"
            )
        periodic = periodic and sample.matching_faces()
    if lateral is None:
        lateral = "cyclic" if periodic else "symmetryPlane"
    if lateral not in LATERAL:
        raise ValueError(f"lateral: one of {LATERAL}")
    others = tuple(a for a in range(3) if a != axis)
    cyclic = others if lateral == "cyclic" else ()
    symmetry = others if lateral == "symmetryPlane" else ()
    connect = axis if connected and isinstance(sample, VoxelImage) else None
    mesh = polymesh(sample, phases, cyclic=cyclic, symmetry=symmetry, connected_axis=connect)
    directory = Path(directory)
    files = write_polymesh(directory, mesh)
    inlet, outlet = FACES[2 * axis], FACES[2 * axis + 1]
    if isinstance(sample, VoxelImage):
        lengths = np.asarray(sample.spacing, dtype=float) * np.asarray(sample.shape)
        total = int(np.asarray(sample.data).size)
        fraction = float(np.isin(np.asarray(sample.data), list(phases)).sum() / total)
        cells_fraction = mesh.n_cells / total
    else:
        lengths = np.asarray(sample.size, dtype=float)
        vol = sample.volumes()
        fraction = float(vol[np.isin(sample.phase, list(phases))].sum() / sample.box_volume)
        cells_fraction = float(mesh.cell_volumes().sum() / sample.box_volume)
    volume = float(np.prod(lengths))
    area = volume / float(lengths[axis])

    sides = [FACES[2 * a + side] for a in others for side in (0, 1)]
    side_bc = [f"type            {lateral};"]  # the lateral faces: symmetry planes or cyclic
    walls = WALLS if mesh.patch(WALLS).size else None
    if kind == "flow":
        application = "simpleFoam"
        u_bc = {inlet: ["type            zeroGradient;"], outlet: ["type            zeroGradient;"]}
        p_bc = {
            inlet: [
                "type            fixedValue;",
                f"value           uniform {pressure_drop:.12g};",
            ],
            outlet: ["type            fixedValue;", "value           uniform 0;"],
        }
        u_bc.update(dict.fromkeys(sides, side_bc))
        p_bc.update(dict.fromkeys(sides, side_bc))
        if walls:
            u_bc[walls] = ["type            noSlip;"]
            p_bc[walls] = ["type            zeroGradient;"]
        fields = {
            "U": _field("volVectorField", "U", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)", u_bc),
            "p": _field("volScalarField", "p", "[0 2 -2 0 0 0 0]", "uniform 0", p_bc),
        }
        schemes = """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
    div((nuEff*dev(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""
        solution = f"""
solvers
{{
    p
    {{
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-10;
        relTol          0.01;
    }}
    U
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-10;
        relTol          0.01;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors {0 if isinstance(sample, VoxelImage) else 2};
    consistent      yes;
    residualControl
    {{
        p               {tolerance:g};
        U               {tolerance:g};
    }}
}}

relaxationFactors
{{
    equations
    {{
        U               0.9;
        ".*"            0.9;
    }}
    fields
    {{
        p               1;
    }}
}}
"""
        properties = {
            "transportProperties": (
                f"transportModel  Newtonian;\n\nnu              [0 2 -1 0 0 0 0] {viscosity:.12g};"
            ),
            "physicalProperties": (
                f"viscosityModel  constant;\n\nnu              [0 2 -1 0 0 0 0] {viscosity:.12g};"
            ),
            "turbulenceProperties": "simulationType  laminar;",
            "momentumTransport": "simulationType  laminar;",
        }
        end, write = int(iterations), int(iterations)
        run = (
            "if command -v simpleFoam >/dev/null 2>&1; then\n"
            "    simpleFoam > log.simpleFoam 2>&1\nelse\n"
            "    foamRun -solver incompressibleFluid > log.foamRun 2>&1\nfi\n"
        )
    else:
        application = "laplacianFoam"
        t_bc = {
            inlet: ["type            fixedValue;", f"value           uniform {difference:.12g};"],
            outlet: ["type            fixedValue;", "value           uniform 0;"],
        }
        t_bc.update(dict.fromkeys(sides, side_bc))
        if walls:
            t_bc[walls] = ["type            zeroGradient;"]
        fields = {"T": _field("volScalarField", "T", "[0 0 0 1 0 0 0]", "uniform 0", t_bc)}
        schemes = """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(T)         Gauss linear;
}

divSchemes
{
    default         none;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""
        correctors = 0 if isinstance(sample, VoxelImage) else 3
        solution = f"""
solvers
{{
    T
    {{
        solver          PCG;
        preconditioner  DIC;
        tolerance       {tolerance * 1e-3:g};
        relTol          0;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors {correctors};
}}
"""
        dt = f"DT              [0 2 -1 0 0 0 0] {diffusivity:.12g};"
        properties = {"transportProperties": dt, "physicalProperties": dt}
        end, write = 1, 1
        run = "laplacianFoam > log.laplacianFoam 2>&1\n"
    functions = _FUNCTIONS[kind].format(outlet=outlet)
    solver = "\nsolver          incompressibleFluid;\n" if kind == "flow" else ""
    control = f"""
application     {application};
{solver}
startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         {end};

deltaT          1;

writeControl    timeStep;

writeInterval   {write};

purgeWrite      0;

writeFormat     ascii;

writePrecision  12;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

functions
{{{functions}}}
"""
    system = directory / "system"
    constant = directory / "constant"
    zero = directory / "0"
    for folder in (system, constant, zero):
        folder.mkdir(parents=True, exist_ok=True)
    texts = {
        system / "controlDict": _dictionary("dictionary", "controlDict", control, "system"),
        system / "fvSchemes": _dictionary("dictionary", "fvSchemes", schemes, "system"),
        system / "fvSolution": _dictionary("dictionary", "fvSolution", solution, "system"),
    }
    for obj, body in properties.items():
        texts[constant / obj] = _dictionary("dictionary", obj, body, "constant")
    for obj, text in fields.items():
        texts[zero / obj] = text
    for path, text in texts.items():
        path.write_text(text, encoding="ascii")
        files.append(path)
    allrun = directory / "Allrun"
    allrun.write_text(
        "#!/bin/sh\n"
        f"# morphoGen {__version__}: the case run by OpenFOAM (its environment sourced)\n"
        'cd "${0%/*}" || exit 1\n' + run,
        encoding="ascii",
    )
    allrun.chmod(0o755)
    foam = directory / f"{directory.name or 'case'}.foam"
    foam.write_text("", encoding="ascii")
    files += [allrun, foam]
    manifest = {
        "morphogen": __version__,
        "kind": kind,
        "axis": axis,
        "inlet": inlet,
        "outlet": outlet,
        "lateral": lateral,
        "phases": [int(p) for p in phases],
        "lengths": [float(v) for v in lengths],
        "length": float(lengths[axis]),
        "area": float(area),
        "volume": volume,
        "cells": mesh.n_cells,
        "faces": len(mesh.faces),
        "fraction": fraction,
        "connected_fraction": float(cells_fraction),
        "viscosity": float(viscosity),
        "pressure_drop": float(pressure_drop),
        "diffusivity": float(diffusivity),
        "difference": float(difference),
    }
    target = directory / "morphogen_openfoam.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files.append(target)
    return OpenFoamExport(directory, kind, mesh, files, manifest)


# -- the results -----------------------------------------------------------------------


def _last_values(folder: Path) -> list[float] | None:
    """The numbers of the last line of the outputs of a function object (the files of its
    last start time), or None."""
    if not folder.is_dir():
        return None
    starts = sorted(
        (p for p in folder.iterdir() if p.is_dir()),
        key=lambda p: float(p.name) if re.fullmatch(r"[-+.\deE]+", p.name) else 0.0,
    )
    for start in reversed(starts):
        for data in sorted(start.glob("*.dat")):
            rows = [
                line
                for line in data.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if rows:
                return [
                    float(v)
                    for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", rows[-1])
                ]
    return None


@dataclass
class OpenFoamRun:
    """The results of a case (:func:`read_openfoam`): its description, the last values of
    its function objects and the properties they give."""

    manifest: dict
    values: dict[str, list[float]] = field(default_factory=dict)
    properties: dict[str, float] = field(default_factory=dict)


def read_openfoam(directory) -> OpenFoamRun:
    """The results of a case written by :func:`write_openfoam` and run by OpenFOAM: the
    flow rate through the outlet and Darcy's permeability along the axis (flow), or the
    integral of the gradient of ``T`` over the cells, the effective diffusivity along the
    axis and its ratio to the diffusivity of the phase (diffusion); see the module."""
    directory = Path(directory)
    manifest = json.loads((directory / "morphogen_openfoam.json").read_text(encoding="utf-8"))
    out = OpenFoamRun(manifest=manifest)
    post = directory / "postProcessing"
    length, area, volume = manifest["length"], manifest["area"], manifest["volume"]
    if manifest["kind"] == "flow":
        values = _last_values(post / "flowRate")
        if values is not None:
            out.values["flowRate"] = values
            rate = abs(values[-1])
            out.properties.update(
                flow_rate=rate,
                mean_velocity=rate / area,
                permeability=manifest["viscosity"]
                * rate
                * length
                / (area * manifest["pressure_drop"]),
            )
    else:
        values = _last_values(post / "gradientIntegral")
        if values is not None:
            out.values["gradientIntegral"] = values
            integral = values[-3 + manifest["axis"]]
            d_eff = -manifest["diffusivity"] * integral * length / (volume * manifest["difference"])
            out.properties.update(
                gradient_integral=integral,
                effective_diffusivity=d_eff,
                relative_diffusivity=d_eff / manifest["diffusivity"],
            )
    return out
