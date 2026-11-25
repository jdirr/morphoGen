"""Particles for the discrete element method (DEM): one phase of a microstructure without
its matrix.

The particles of a phase are the connected parts of its voxels (:func:`particles`), each
a closed surface cut from the implicit function of the phase (marching cubes): whole
across the faces of periodic boxes, where the medium goes on (a particle that crosses a
face is put back together from its pieces), cut and closed by the faces of non-periodic
windows. The structures made of spheres and segments (:class:`PrimitiveUnion`) also give
their exact spheres, the segments as clumps of spheres along their axes
(:func:`particle_spheres`), which the codes of discrete elements take as they are.

:func:`write_particles` writes them in the format of the suffix of its path:

============================  ===============================================================
suffix                        what is written
============================  ===============================================================
``.stl``                      the surfaces of the particles, one solid ``particle_<k>`` each
                              (ASCII); with ``separate=True``, a folder of one binary STL each
``.obj``                      the same, one object each
``.vtu``                      the same, triangles with the number of the particle of each
``.msh``                      Gmsh 2.2 (ASCII): the triangles of each particle, or its
                              tetrahedra (``elements="tetrahedra"``), as an elementary entity
                              and a physical group ``particle_<k>`` (``particle_<k>_cut`` for a
                              particle that is not whole): the meshes that LMGC90 reads
                              (``readMesh``, then ``separateMeshes`` and ``surfacicMeshToRigid3D``
                              or ``volumicMeshToRigid3D``)
``.csv`` ``.txt`` ``.xlsx``   a table: the spheres (centre, radius, clump) when the structure
                              has them, else the particles (volume, area, centroid, radius of
                              the sphere of the same volume, whether whole)
``.data``                     a LAMMPS / LIGGGHTS data file of the spheres (atom style
                              ``sphere``; ``hybrid sphere molecular`` with the clumps as
                              molecules, for ``fix rigid``)
``format="yade"``             the columns ``x y z r`` (``clump`` too) that YADE's
                              ``ymport.text`` (``ymport.textClumps``) reads
============================  ===============================================================

The surfaces of the exact primitives are those of their spheres (icospheres) and of the
spherocylinders of their segments (capsules), scaled to their volumes. ``size`` remeshes
every surface with MMG (``mmgs``, of the package ``pymmg``) at triangles of that size:
the cubes of the voxels leave thin triangles, and the codes of polyhedra (LMGC90's
``POLYR``) detect their contacts the faster for fewer faces. The tetrahedra of a particle
fill its surface, whose triangles are their boundary (:func:`particle_tetrahedra`: gmsh's
Delaunay mesher, or without the module ``gmsh`` the fan of the triangles to the centre of a
star-shaped particle).
"""

from __future__ import annotations

import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from morphogen._version import __version__
from morphogen.domain import Domain
from morphogen.image import VoxelImage
from morphogen.mesh.surface import SurfaceMesh, _marching_cubes

#: The suffixes of the surfaces, of the tables and of the sphere files
SURFACE_FORMATS = (".stl", ".obj", ".vtu", ".msh")
TABLE_FORMATS = (".csv", ".txt", ".xlsx")
SPHERE_FORMATS = (".data",)


@dataclass
class Particle:
    """One particle of a phase: its closed surface (outward normals, in the coordinates of
    the domain: a whole particle of a periodic box may reach beyond its faces) and the
    number of its voxels. ``whole`` is False for a particle cut by the faces of a
    non-periodic window, and for a part that winds around the periodic box (an unbounded
    cluster of the medium), both closed by the faces of the box."""

    surface: SurfaceMesh
    voxels: int
    whole: bool

    @property
    def volume(self) -> float:
        return self.surface.volume()

    @property
    def area(self) -> float:
        return self.surface.area()

    @property
    def centroid(self) -> np.ndarray:
        """The centre of the volume enclosed (the divergence theorem on the tetrahedra of
        the faces and the origin)."""
        return _centroid(self.surface)

    @property
    def equivalent_radius(self) -> float:
        """The radius of the sphere of the same volume."""
        return (3.0 * max(self.volume, 0.0) / (4.0 * math.pi)) ** (1.0 / 3.0)


def _centroid(surface: SurfaceMesh) -> np.ndarray:
    """The centre of the volume a closed surface encloses (the signed tetrahedra of its
    faces and the origin)."""
    v = surface.vertices[surface.faces]
    six = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))
    total = float(six.sum())
    if total == 0.0:
        return surface.vertices.mean(axis=0)
    return (six[:, None] * v.sum(axis=1)).sum(axis=0) / (4.0 * total)


def _field(obj, phase: int, shape, voxel_size) -> tuple[np.ndarray, VoxelImage]:
    """The implicit function of the phase on a grid (negative inside), and the image whose
    grid it is: that of a microstructure, or of a voxel image (a phase image: half a
    voxel's step about its boundary; a level set: as it is)."""
    from morphogen.models.base import Microstructure

    if isinstance(obj, Microstructure):
        if shape is None and voxel_size is None:
            shape = 64
        image = obj.phase_level_set(phase, shape, voxel_size=voxel_size)
        return np.asarray(image.data, dtype=np.float32), image
    if not isinstance(obj, VoxelImage):
        raise TypeError("particles: a microstructure or a voxel image")
    data = np.asarray(obj.data)
    if data.dtype.kind in "iub":
        return (0.5 - (data == phase)).astype(np.float32), obj
    if obj.name == "fraction":
        return (0.5 - data).astype(np.float32), obj
    return data.astype(np.float32), obj  # a level set of the phase


def _unwrapped(labels: np.ndarray, n: int, structure: np.ndarray):
    """The components of ``labels`` (labelled inside the box) joined across the periodic
    faces: the root of each label, the translation (in boxes) of each label's copy in its
    root's, and whether each root winds around the box (a cycle of neighbours through the
    faces whose translations do not cancel)."""
    shape = labels.shape
    parent = np.arange(n + 1)
    frame = np.zeros((n + 1, 3), dtype=np.int64)
    wraps = np.zeros(n + 1, dtype=bool)

    def find(a):
        path = []
        while parent[a] != a:
            path.append(a)
            a = parent[a]
        for b in reversed(path[:-1]):  # path compression, frames made relative to the root
            frame[b] += frame[parent[b]]
            parent[b] = a
        return a

    edges = []
    for d in np.argwhere(structure) - 1:
        if tuple(d) <= (0, 0, 0):  # one offset of each opposite pair, not the centre
            continue
        for a in np.flatnonzero(d):  # voxels whose neighbour x + d is across the face normal to a
            index = [np.arange(m) for m in shape]
            index[a] = np.array([shape[a] - 1 if d[a] > 0 else 0])
            moved = [i + k for i, k in zip(index, d, strict=True)]
            u = labels[np.ix_(*index)]
            v = labels[np.ix_(*[m % s for m, s in zip(moved, shape, strict=True)])]
            shift = np.ix_(*[m // s for m, s in zip(moved, shape, strict=True)])
            shift = np.stack(np.broadcast_arrays(*shift), axis=-1)
            keep = (u > 0) & (v > 0)
            edges.append(np.column_stack([u[keep], v[keep], shift[keep]]))
    if edges:
        for u, v, *s in np.unique(np.concatenate(edges), axis=0).tolist():
            ru, rv = find(u), find(v)
            winding = frame[u] + np.asarray(s) - frame[v]
            if ru == rv:
                if np.any(winding != 0):
                    wraps[ru] = True
            else:
                parent[rv] = ru
                frame[rv] = winding
                wraps[ru] |= wraps[rv]
    roots = np.zeros(n + 1, dtype=np.int64)
    frames = np.zeros((n + 1, 3), dtype=np.int64)
    for a in range(n + 1):  # after find, the frame of a label is that of its root's copy
        roots[a] = find(a)
        if roots[a] != a:
            frames[a] = frame[a]
    return roots, frames, wraps


def particles(
    obj,
    phase: int = 1,
    *,
    shape=None,
    voxel_size=None,
    connectivity: int = 6,
    min_voxels: int = 1,
) -> list[Particle]:
    """The particles of a phase: its connected parts, each a closed surface cut from the
    implicit function of the phase on a grid (``shape`` voxels along the longest edge of
    a microstructure, 64 by default, or ``voxel_size``; a voxel image as it is),
    ``connectivity`` 6, 18 or 26 neighbours, those of fewer than ``min_voxels`` voxels
    left out. On a periodic domain the parts that cross its faces are put back together
    (whole particles that reach beyond the box); a part that winds around the periodic
    box is no particle but an unbounded cluster, cut and closed by the faces of the box
    (``whole`` False), as are the particles cut by the faces of a non-periodic window."""
    from morphogen.analysis.connectivity import _structure

    field, image = _field(obj, phase, shape, voxel_size)
    marching_cubes = _marching_cubes()
    structure = _structure(connectivity)
    labels, n = ndimage.label(field < 0.0, structure=structure)
    dims = np.array(field.shape)
    spacing = np.asarray(image.spacing, dtype=float)
    lower = np.asarray(image.origin, dtype=float)
    upper = lower + spacing * dims
    if image.periodic:
        roots, frames, wraps = _unwrapped(labels, n, structure)
    else:
        roots, frames, wraps = np.arange(n + 1), np.zeros((n + 1, 3), int), np.zeros(n + 1, bool)
    boxes = ndimage.find_objects(labels)
    members: dict[int, list[int]] = {}
    for a in range(1, n + 1):
        members.setdefault(int(roots[a]), []).append(a)
    out = []
    for root in sorted(members):
        parts = []
        for a in members[root]:
            box = boxes[a - 1]
            local = np.argwhere(labels[box] == a) + np.array([s.start for s in box])
            shift = 0 if wraps[root] else frames[a] * dims
            parts.append(local + shift)
        voxels = np.concatenate(parts)
        if len(voxels) < max(int(min_voxels), 1):
            continue
        lo = voxels.min(axis=0) - 1
        hi = voxels.max(axis=0) + 1
        idx = [np.arange(lo[k], hi[k] + 1) for k in range(3)]
        values = field[np.ix_(*[i % dims[k] for k, i in enumerate(idx)])]
        mine = np.zeros(values.shape, dtype=bool)
        mine[tuple((voxels - lo).T)] = True
        sub = np.where(mine, values, np.abs(values) + np.float32(1e-6))
        if not image.periodic or wraps[root]:  # outside the box: outside the particle
            outside = [(i < 0) | (i >= dims[k]) for k, i in enumerate(idx)]
            beyond = (
                outside[0][:, None, None] | outside[1][None, :, None] | outside[2][None, None, :]
            )
            sub = np.where(beyond, np.abs(sub) + np.float32(1.0), sub)
        top = float(np.abs(sub).max()) + 1.0
        sub = np.pad(sub, 1, mode="constant", constant_values=np.float32(top))
        # values exactly at the level put vertices on grid nodes (see surface_mesh)
        scale = float(np.abs(sub).max())
        sub = np.where(sub == 0.0, np.float32(1e-3 * max(scale, 1e-30)), sub)
        verts, faces, _, _ = marching_cubes(
            sub, level=0.0, spacing=tuple(spacing), gradient_direction="ascent",
            allow_degenerate=False,
        )  # fmt: skip
        verts = verts + lower + (lo - 1 + 0.5) * spacing
        cut = not image.periodic and (np.any(voxels == 0) or np.any(voxels == dims - 1))
        if cut or wraps[root]:  # the caps on the faces of the box, as surface_mesh puts them
            first, last, eps = lower + 0.5 * spacing, upper - 0.5 * spacing, 1e-9 * spacing
            verts = np.where(verts < first - eps, lower, verts)
            verts = np.where(verts > last + eps, upper, verts)
        surface = SurfaceMesh(verts, faces.astype(np.int64))
        if surface.volume() < 0.0:
            surface = SurfaceMesh(verts, faces[:, ::-1].astype(np.int64))
        out.append(Particle(surface, len(voxels), not (cut or wraps[root])))
    return out


def particle_spheres(micro, phase: int = 1, *, clump_spacing: float = 0.5) -> dict:
    """The exact spheres of a structure made of spheres and segments (a
    :class:`~morphogen.models.union.PrimitiveUnion`, phase 1): ``centre`` (n x 3),
    ``radius``, ``clump`` (the primitive of each sphere) and ``segment`` (whether it is a
    bead of a segment, whose first and last are its ends). A sphere is a clump of one;
    a segment is a clump of spheres of its radius along its axis, ``clump_spacing`` radii
    apart at most (round caps: its ends are its caps; the beads leave a groove of
    ``r - sqrt(r^2 - s^2/4)`` between them, 3 % of the radius for 0.5). Periodic domains:
    each primitive once, its centre (or the middle of its axis) in the box; non-periodic
    ones: the primitives that reach the window."""
    from morphogen.models.union import PrimitiveUnion
    from morphogen.primitives import restricted_to_box

    if not isinstance(micro, PrimitiveUnion) or int(phase) != 1:
        raise ValueError(
            "particle_spheres: the spheres of a union of spheres and segments (phase 1); "
            "the other structures give the surfaces of their particles (particles)"
        )
    if not clump_spacing > 0:
        raise ValueError("clump_spacing: positive (in radii)")
    domain: Domain = micro.domain
    spheres, segments = micro.spheres, micro.segments
    if not domain.periodic:
        spheres = restricted_to_box(spheres, domain)
        segments = restricted_to_box(segments, domain)
    centres, radii, clumps, beads = [], [], [], []
    shift = np.zeros((len(spheres), 3))
    if domain.periodic and len(spheres):
        shift = domain.wrap(spheres.centers) - spheres.centers
    centres.append(np.asarray(spheres.centers, dtype=float) + shift)
    radii.append(np.asarray(spheres.radii, dtype=float))
    clumps.append(np.arange(len(spheres)))
    beads.append(np.zeros(len(spheres), dtype=bool))
    first = len(spheres)
    for k in range(len(segments)):
        a, b = np.asarray(segments.start[k], float), np.asarray(segments.end[k], float)
        r = float(segments.radii[k])
        if domain.periodic:
            middle = 0.5 * (a + b)
            moved = domain.wrap(middle) - middle
            a, b = a + moved, b + moved
        length = float(np.linalg.norm(b - a))
        m = max(1, math.ceil(length / (clump_spacing * r)))
        t = np.linspace(0.0, 1.0, m + 1)
        centres.append(a + t[:, None] * (b - a))
        radii.append(np.full(m + 1, r))
        clumps.append(np.full(m + 1, first + k))
        beads.append(np.ones(m + 1, dtype=bool))
    return {
        "centre": np.concatenate(centres).reshape(-1, 3),
        "radius": np.concatenate(radii),
        "clump": np.concatenate(clumps).astype(np.int64),
        "segment": np.concatenate(beads),
    }


def _has_spheres(obj, phase: int) -> bool:
    from morphogen.models.union import PrimitiveUnion

    return isinstance(obj, PrimitiveUnion) and int(phase) == 1


def particle_table(found: list[Particle]) -> dict:
    """The table of the particles: their number, volume, area, centroid, the radius of the
    sphere of the same volume, the voxels they hold and whether they are whole."""
    c = np.array([p.centroid for p in found]).reshape(-1, 3)
    return {
        "particle": np.arange(1, len(found) + 1),
        "volume": np.array([p.volume for p in found]),
        "area": np.array([p.area for p in found]),
        "x": c[:, 0],
        "y": c[:, 1],
        "z": c[:, 2],
        "equivalent_radius": np.array([p.equivalent_radius for p in found]),
        "voxels": np.array([p.voxels for p in found], dtype=np.int64),
        "whole": np.array([p.whole for p in found]),
    }


def _sphere_table(spheres: dict) -> dict:
    c = spheres["centre"]
    return {
        "sphere": np.arange(1, len(c) + 1),
        "x": c[:, 0],
        "y": c[:, 1],
        "z": c[:, 2],
        "radius": spheres["radius"],
        "clump": spheres["clump"] + 1,
    }


def _write_stl_ascii(path: Path, surfaces: list[SurfaceMesh]) -> Path:
    lines = []
    for k, s in enumerate(surfaces, start=1):
        lines.append(f"solid particle_{k}")
        tri = s.vertices[s.faces]
        for n, t in zip(s.face_normals(), tri, strict=True):
            lines.append(f"  facet normal {n[0]:.7e} {n[1]:.7e} {n[2]:.7e}\n    outer loop")
            lines.extend(f"      vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}" for v in t)
            lines.append("    endloop\n  endfacet")
        lines.append(f"endsolid particle_{k}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _write_obj(path: Path, surfaces: list[SurfaceMesh]) -> Path:
    lines = [f"# particles of morphoGen {__version__}: one object each"]
    offset = 1
    for k, s in enumerate(surfaces, start=1):
        lines.append(f"o particle_{k}")
        lines.extend(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}" for v in s.vertices)
        lines.extend(f"f {a + offset} {b + offset} {c + offset}" for a, b, c in s.faces)
        offset += len(s.vertices)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _write_lammps(path: Path, spheres: dict, domain: Domain, density: float) -> Path:
    """A LAMMPS / LIGGGHTS data file of the spheres: atom style sphere, or hybrid sphere
    molecular when segments make clumps (the clump of each sphere its molecule)."""
    c, r, clump = spheres["centre"], spheres["radius"], spheres["clump"]
    clumped = len(np.unique(clump)) < len(clump)
    lo = np.minimum(domain.lower, (c - r[:, None]).min(axis=0)) if len(c) else domain.lower
    hi = np.maximum(domain.upper, (c + r[:, None]).max(axis=0)) if len(c) else domain.upper
    if domain.periodic:  # the box of the periodic medium
        lo, hi = domain.lower, domain.upper
    style = "hybrid sphere molecular" if clumped else "sphere"
    lines = [
        f"LAMMPS data file: particles of morphoGen {__version__} (atom_style {style})",
        "",
        f"{len(c)} atoms",
        "1 atom types",
        "",
        f"{lo[0]:.9g} {hi[0]:.9g} xlo xhi",
        f"{lo[1]:.9g} {hi[1]:.9g} ylo yhi",
        f"{lo[2]:.9g} {hi[2]:.9g} zlo zhi",
        "",
        f"Atoms # {'hybrid' if clumped else 'sphere'}",
        "",
    ]
    for k in range(len(c)):
        x, y, z = c[k]
        if clumped:  # atom-ID atom-type x y z diameter density molecule-ID
            lines.append(f"{k + 1} 1 {x:.9g} {y:.9g} {z:.9g} {2 * r[k]:.9g} {density:.9g} "
                         f"{clump[k] + 1}")  # fmt: skip
        else:  # atom-ID atom-type diameter density x y z
            lines.append(f"{k + 1} 1 {2 * r[k]:.9g} {density:.9g} {x:.9g} {y:.9g} {z:.9g}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _write_yade(path: Path, spheres: dict) -> Path:
    c, r, clump = spheres["centre"], spheres["radius"], spheres["clump"]
    clumped = len(np.unique(clump)) < len(clump)
    head = "# x y z r clumpId" if clumped else "# x y z r"
    rows = [f"{x:.9g} {y:.9g} {z:.9g} {rr:.9g}" + (f" {int(k)}" if clumped else "")
            for (x, y, z), rr, k in zip(c, r, clump, strict=True)]  # fmt: skip
    path.write_text(
        "\n".join([f"{head} (morphoGen {__version__})", *rows]) + "\n", encoding="ascii"
    )
    return path


def write_particles(
    path,
    obj,
    phase: int = 1,
    *,
    shape=None,
    voxel_size=None,
    connectivity: int = 6,
    min_voxels: int = 1,
    exact: bool | None = None,
    separate: bool = False,
    clump_spacing: float = 0.5,
    density: float = 1.0,
    format: str | None = None,
    elements: str = "triangles",
    size: float | None = None,
) -> Path | list[Path]:
    """The particles of a phase of ``obj`` (a microstructure or a voxel image) without
    its matrix, in the format of the suffix of ``path`` (see the module) or ``format``
    ("yade"). The surfaces follow :func:`particles` (``shape``, ``voxel_size``,
    ``connectivity``, ``min_voxels``), or are those of the exact primitives
    (:func:`sphere_surfaces`); the tables, sphere and YADE files the exact spheres of
    :func:`particle_spheres` when the structure has them (``exact``, by default for them),
    else the particles' table (the spheres of the same volume at their centroids for the
    sphere files). ``density`` is that of the spheres of LAMMPS files. ``elements``
    "tetrahedra" fills the particles of a Gmsh file with tetrahedra
    (:func:`particle_tetrahedra`); ``size`` remeshes the surfaces at triangles of that
    size (:func:`remesh_surfaces`)."""
    from morphogen.io.tables import write_table

    path = Path(path)
    suffix = path.suffix.lower()
    kind = (format or "").lower() or None
    if kind not in (None, "yade"):
        raise ValueError("format: 'yade', or the suffix of the path")
    if elements not in ("triangles", "tetrahedra"):
        raise ValueError("elements: 'triangles' or 'tetrahedra'")
    if elements == "tetrahedra" and (suffix != ".msh" or kind):
        raise ValueError(
            "elements='tetrahedra': the volumes of the particles in a Gmsh file (.msh)"
        )
    if size is not None and not float(size) > 0:
        raise ValueError("size: a positive length (the triangles of the surfaces remeshed)")
    spheres_known = _has_spheres(obj, phase)
    exact = spheres_known if exact is None else bool(exact)
    if exact and not spheres_known:
        raise ValueError("exact: the spheres of a union of spheres and segments (phase 1) only")
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "yade" or suffix in TABLE_FORMATS + SPHERE_FORMATS:
        if exact:
            spheres = particle_spheres(obj, phase, clump_spacing=clump_spacing)
        else:
            found = particles(obj, phase, shape=shape, voxel_size=voxel_size,
                              connectivity=connectivity, min_voxels=min_voxels)  # fmt: skip
            if suffix in TABLE_FORMATS and kind is None:
                return write_table(path, particle_table(found))
            spheres = {
                "centre": np.array([p.centroid for p in found]).reshape(-1, 3),
                "radius": np.array([p.equivalent_radius for p in found]),
                "clump": np.arange(len(found)),
            }
        domain = obj.domain if isinstance(getattr(obj, "domain", None), Domain) else obj.domain()
        if domain.periodic and len(spheres["centre"]):  # the centres in the periodic box
            spheres = {**spheres, "centre": domain.wrap(spheres["centre"])}
        if kind == "yade":
            return _write_yade(path, spheres)
        if suffix == ".data":
            return _write_lammps(path, spheres, domain, float(density))
        return write_table(path, _sphere_table(spheres))
    if suffix not in SURFACE_FORMATS:
        raise ValueError(
            f"particles: the suffixes {', '.join(SURFACE_FORMATS + TABLE_FORMATS + SPHERE_FORMATS)}"
            " or format='yade'"
        )
    if exact:
        surfaces = sphere_surfaces(particle_spheres(obj, phase, clump_spacing=clump_spacing))
        whole = [True] * len(surfaces)
    else:
        found = particles(obj, phase, shape=shape, voxel_size=voxel_size,
                          connectivity=connectivity, min_voxels=min_voxels)  # fmt: skip
        surfaces, whole = [p.surface for p in found], [p.whole for p in found]
    if size is not None:
        surfaces = remesh_surfaces(surfaces, float(size))
    if suffix == ".msh":
        if separate:
            raise ValueError("separate: one binary STL per particle (.stl)")
        if elements == "tetrahedra":
            return _write_msh(path, particle_tetrahedra(surfaces), whole, 3)
        return _write_msh(path, [(s.vertices, s.faces) for s in surfaces], whole, 2)
    if separate:
        if suffix != ".stl":
            raise ValueError("separate: one binary STL per particle (.stl)")
        from morphogen.io.meshes import write_stl

        folder = path.with_suffix("")
        folder.mkdir(parents=True, exist_ok=True)
        return [write_stl(folder / f"particle_{k:05d}.stl", s, name=f"particle_{k}")
                for k, s in enumerate(surfaces, start=1)]  # fmt: skip
    if suffix == ".stl":
        return _write_stl_ascii(path, surfaces)
    if suffix == ".obj":
        return _write_obj(path, surfaces)
    from morphogen.io.vtk import write_vtu

    points, cells, number, entire, offset = [np.zeros((0, 3))], [], [], [], 0
    for k, (s, w) in enumerate(zip(surfaces, whole, strict=True), start=1):
        points.append(s.vertices)
        cells.append(s.faces + offset)
        number.append(np.full(len(s.faces), k, dtype=np.int64))
        entire.append(np.full(len(s.faces), w, dtype=np.uint8))
        offset += len(s.vertices)
    cells = np.concatenate(cells) if cells else np.zeros((0, 3), dtype=np.int64)
    data = {"particle": np.concatenate(number or [np.zeros(0, dtype=np.int64)]),
            "whole": np.concatenate(entire or [np.zeros(0, dtype=np.uint8)])}  # fmt: skip
    return write_vtu(path, np.concatenate(points), cells, 5, data)


def _icosphere(level: int = 2) -> SurfaceMesh:
    """The unit sphere as an icosahedron subdivided ``level`` times (outward normals),
    scaled to the volume of the sphere (the mass of a particle kept)."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]  # fmt: skip
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]  # fmt: skip
    v = [np.asarray(p, dtype=float) / np.linalg.norm(p) for p in verts]
    f = list(faces)

    def middle(a, b, cache):  # the vertex in the middle of an edge, on the sphere
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = v[a] + v[b]
            v.append(m / np.linalg.norm(m))
            cache[key] = len(v) - 1
        return cache[key]

    for _ in range(int(level)):
        cache: dict[tuple[int, int], int] = {}
        split = []
        for a, b, c in f:
            ab, bc, ca = middle(a, b, cache), middle(b, c, cache), middle(c, a, cache)
            split += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = split
    mesh = SurfaceMesh(np.array(v), np.array(f, dtype=np.int64))
    scale = (4.0 * math.pi / 3.0 / mesh.volume()) ** (1.0 / 3.0)
    return SurfaceMesh(scale * mesh.vertices, mesh.faces)


def _capsule_mesh(start, end, radius: float, rings: int) -> SurfaceMesh:
    """The spherocylinder of axis ``start``-``end``: two hemispherical caps of ``rings``
    rings of latitude, the cylinder between them in bands about as high, ``4 rings``
    meridians (outward normals)."""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    length = float(np.linalg.norm(b - a))
    u = (b - a) / length
    e1 = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    m = 4 * rings
    theta = 2.0 * math.pi * np.arange(m) / m
    around = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2  # (m, 3)
    phi = 0.5 * math.pi * np.arange(1, rings + 1) / rings  # from the pole to the equator
    bands = max(1, round(length / (radius * 0.5 * math.pi / rings)))
    circles = [b + radius * (math.cos(f) * u + math.sin(f) * around) for f in phi]
    circles += [b - (t / bands) * length * u + radius * around for t in range(1, bands + 1)]
    circles += [a + radius * (-math.cos(f) * u + math.sin(f) * around) for f in phi[-2::-1]]
    top, bottom = b + radius * u, a - radius * u
    vertices = np.vstack([top, *circles, bottom])
    ring = np.arange(m)
    faces = [np.column_stack([np.zeros(m, dtype=np.int64), 1 + ring, 1 + (ring + 1) % m])]
    for j in range(len(circles) - 1):
        p, q = 1 + j * m + ring, 1 + (j + 1) * m + ring
        p1, q1 = 1 + j * m + (ring + 1) % m, 1 + (j + 1) * m + (ring + 1) % m
        faces += [np.column_stack([p, q, q1]), np.column_stack([p, q1, p1])]
    last = 1 + (len(circles) - 1) * m
    faces.append(np.column_stack([np.full(m, len(vertices) - 1), last + (ring + 1) % m,
                                  last + ring]))  # fmt: skip
    mesh = SurfaceMesh(vertices, np.concatenate(faces).astype(np.int64))
    if mesh.volume() < 0.0:
        mesh = SurfaceMesh(vertices, mesh.faces[:, ::-1].copy())
    return mesh


def _capsule(start, end, radius: float, level: int = 2) -> SurfaceMesh:
    """The closed surface of a spherocylinder (outward normals), its radius scaled so that
    it holds the volume ``pi r^2 l + 4/3 pi r^3`` (the mass of the particle kept): the
    prism of the cylinder holds ``c2 s^2`` and the caps ``c3 s^3`` for a radius ``s``."""
    rings = 2 ** int(level) + 1
    r = float(radius)
    length = float(np.linalg.norm(np.asarray(end, float) - np.asarray(start, float)))
    m = 4 * rings
    c2 = 0.5 * m * math.sin(2.0 * math.pi / m) * length
    c3 = (_capsule_mesh(start, end, r, rings).volume() - c2 * r * r) / r**3
    target = math.pi * r * r * length + 4.0 / 3.0 * math.pi * r**3
    s = r
    for _ in range(50):  # Newton on c2 s^2 + c3 s^3 = target (increasing in s)
        step = (c2 * s * s + c3 * s**3 - target) / (2.0 * c2 * s + 3.0 * c3 * s * s)
        s -= step
        if abs(step) < 1e-15 * s:
            break
    return _capsule_mesh(start, end, s, rings)


def sphere_surfaces(spheres: dict, level: int = 2) -> list[SurfaceMesh]:
    """The surfaces of the clumps of spheres of :func:`particle_spheres`, one each: a
    sphere's icosphere of ``level`` subdivisions (320 triangles for 2), the beads of a
    segment (``segment``) the closed surface of its spherocylinder (a capsule from its
    first bead to its last), both scaled to their volumes; the spheres of another clump in
    one surface (overlapping shells, their union not cut out)."""
    unit = _icosphere(level)
    beads = spheres.get("segment")
    out = []
    for k in np.unique(spheres["clump"]):
        picked = np.flatnonzero(spheres["clump"] == k)
        centres, radii = spheres["centre"][picked], spheres["radius"][picked]
        if beads is not None and len(picked) > 1 and bool(np.all(beads[picked])):
            out.append(_capsule(centres[0], centres[-1], float(radii[0]), level))
            continue
        verts = [c + r * unit.vertices for c, r in zip(centres, radii, strict=True)]
        faces = [unit.faces + j * len(unit.vertices) for j in range(len(picked))]
        out.append(SurfaceMesh(np.concatenate(verts), np.concatenate(faces)))
    return out


def _remeshed(surface: SurfaceMesh, size: float, exe: str) -> SurfaceMesh:
    from morphogen.io import medit
    from morphogen.mesh.conforming import _run_mmg

    with tempfile.TemporaryDirectory(prefix="morphogen_mmgs_") as folder:
        folder = Path(folder)
        medit.write_mesh(folder / "in.mesh", surface.vertices, None, surface.faces)
        args = ["-hmax", f"{size:.9g}", "-hmin", f"{size / 4:.9g}",
                "-hausd", f"{size / 10:.9g}", "-hgrad", "1.3"]  # fmt: skip
        out = _run_mmg(exe, folder / "in.mesh", None, folder / "out.mesh", args, 600)
    mesh = SurfaceMesh(np.asarray(out["Vertices"], float), np.asarray(out["Triangles"]))
    if mesh.volume() < 0.0:
        mesh = SurfaceMesh(mesh.vertices, mesh.faces[:, ::-1].copy())
    return mesh


def remesh_surfaces(surfaces: list[SurfaceMesh], size: float) -> list[SurfaceMesh]:
    """The closed surfaces remeshed by MMG (``mmgs``, of the package ``pymmg``) at triangles
    of ``size`` (``-hmax``; ``-hmin`` a quarter of it, the Hausdorff distance a tenth, its
    ridges kept: the faces of the box that close a cut particle), outward: the thin
    triangles that the cubes of the voxels leave become regular ones, and a coarse size
    leaves few faces (the contacts of polyhedra the faster to detect). The surfaces are
    remeshed at once in threads."""
    from morphogen.mesh.conforming import mmg_executable

    if not float(size) > 0:
        raise ValueError("size: a positive length")
    if not surfaces:
        return []
    exe = mmg_executable(program="mmgs")
    with ThreadPoolExecutor(max_workers=min(len(surfaces), os.cpu_count() or 1)) as pool:
        return list(pool.map(lambda s: _remeshed(s, float(size), exe), surfaces))


def _fan(surface: SurfaceMesh) -> tuple[np.ndarray, np.ndarray] | None:
    """The tetrahedra of the triangles of a closed surface and its centre, when they are
    all positive: then each ray from the centre crosses the surface once (the solid
    angles of the triangles, all positive, add up to 4 pi), the surface is star-shaped
    about it and the fan fills it."""
    centre = _centroid(surface)
    nodes = np.vstack([surface.vertices, centre])
    tets = np.column_stack([np.full(len(surface.faces), len(surface.vertices)), surface.faces])
    p = nodes[tets]
    six = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    if not len(six) or six.min() <= 1e-12 * six.sum():
        return None
    return nodes, tets.astype(np.int64)


def _gmsh_tetrahedra(gmsh, surface: SurfaceMesh, size: float | None):
    """The tetrahedra of gmsh's Delaunay mesher in a closed surface, its triangles the
    boundary (a discrete surface, a volume of the kernel geo); None if it fails."""
    gmsh.model.add("particle")
    try:
        n = len(surface.vertices)
        gmsh.model.addDiscreteEntity(2, 1)
        gmsh.model.mesh.addNodes(2, 1, np.arange(1, n + 1), surface.vertices.ravel())
        gmsh.model.mesh.addElementsByType(1, 2, [], (surface.faces + 1).ravel())
        gmsh.model.geo.addSurfaceLoop([1], 1)
        gmsh.model.geo.addVolume([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1e22 if size is None else float(size))
        gmsh.model.mesh.generate(3)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        types, _, conn = gmsh.model.mesh.getElements(3, 1)
    except Exception:  # gmsh raises its own errors (boundary not recovered...)
        return None
    finally:
        gmsh.model.remove()
    if 4 not in list(types):
        return None
    index = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    index[tags.astype(np.int64)] = np.arange(len(tags))
    tets = index[np.asarray(conn[list(types).index(4)], dtype=np.int64).reshape(-1, 4)]
    nodes = np.asarray(xyz, dtype=float).reshape(-1, 3)
    p = nodes[tets]
    six = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    tets[six < 0] = tets[six < 0][:, [0, 2, 1, 3]]
    if abs(abs(six).sum() / 6.0 - surface.volume()) > 1e-9 * abs(surface.volume()):
        return None  # not the volume of the surface: its triangles were not kept
    return nodes, tets


def particle_tetrahedra(
    surfaces: list[SurfaceMesh], *, size: float | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The tetrahedra that fill each closed surface (outward normals), its triangles their
    boundary, positively oriented: ``(nodes, tetrahedra)`` per surface. gmsh's Delaunay
    mesher (the module ``gmsh``; ``size`` its largest element, by default the sizes of
    the triangles), or without the module the fan of the triangles to the centre of a
    star-shaped surface (spheres, spherocylinders, convex particles: exact volumes and
    inertias, for rigid bodies; thin elements for deformable ones). The volume of the
    tetrahedra is that of the surface; a surface that neither fills raises an error."""
    try:
        import gmsh
    except ImportError:  # pragma: no cover - depends on the environment
        gmsh = None
    out = []
    started = False
    try:
        if gmsh is not None and not gmsh.isInitialized():
            gmsh.initialize(readConfigFiles=False, interruptible=False)
            started = True
        if gmsh is not None:
            gmsh.option.setNumber("General.Terminal", 0)
        for k, surface in enumerate(surfaces, start=1):
            filled = _gmsh_tetrahedra(gmsh, surface, size) if gmsh is not None else None
            filled = filled if filled is not None else _fan(surface)
            if filled is None and gmsh is not None:
                raise RuntimeError(
                    f"particle {k}: gmsh did not fill its surface, which is not star-shaped "
                    "about its centre"
                )
            if filled is None:
                raise ImportError(
                    f"particle {k}: its surface is not star-shaped about its centre; gmsh's "
                    "mesher fills any closed surface: pip install gmsh"
                )
            out.append(filled)
    finally:
        if started:
            gmsh.finalize()
    return out


def _write_msh(path: Path, parts: list, whole: list[bool], dimension: int) -> Path:
    """A Gmsh 2.2 ASCII file of the triangles (``dimension`` 2) or tetrahedra (3) of the
    particles: the elements of the particle ``k`` in the elementary entity ``k`` and the
    physical group ``k``, named ``particle_<k>`` (``particle_<k>_cut`` if it is not whole);
    no blank line (LMGC90's reader splits every line)."""
    kind, width = (2, 3) if dimension == 2 else (4, 4)
    n_nodes = sum(len(nodes) for nodes, _ in parts)
    n_cells = sum(len(cells) for _, cells in parts)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$PhysicalNames\n{len(parts)}\n")
        for k, w in enumerate(whole, start=1):
            f.write(f'{dimension} {k} "particle_{k}{"" if w else "_cut"}"\n')
        f.write(f"$EndPhysicalNames\n$Nodes\n{n_nodes}\n")
        first = 1
        for nodes, _ in parts:
            ids = np.arange(first, first + len(nodes))
            if len(nodes):
                np.savetxt(f, np.column_stack([ids, nodes]), fmt=["%d"] + ["%.15g"] * 3)
            first += len(nodes)
        f.write(f"$EndNodes\n$Elements\n{n_cells}\n")
        first, element = 1, 1
        for k, (nodes, cells) in enumerate(parts, start=1):
            cells = np.asarray(cells, dtype=np.int64).reshape(-1, width)
            if len(cells):
                rows = np.column_stack([
                    np.arange(element, element + len(cells)), np.full(len(cells), kind),
                    np.full(len(cells), 2), np.full(len(cells), k), np.full(len(cells), k),
                    cells + first,
                ])  # fmt: skip
                np.savetxt(f, rows, fmt="%d")
            element += len(cells)
            first += len(nodes)
        f.write("$EndElements\n")
    return path
