"""Linear elastic fracture mechanics of 3D cracks: cracks described by two level sets, their
energy release rate, and their propagation, either on meshes conforming to the crack,
remeshed at each step as in Z-cracks, or by the extended finite element method (XFEM) on
a fixed mesh.

A crack (:class:`Crack`) is the zero of ``φ``, the signed distance to its plane, where
``ψ``, the signed distance to its front within the plane, is negative: a straight front
(a through or edge crack of length ``size`` from ``point`` along ``direction``) or a circular
one (a penny of radius ``size``). It grows by moving its front (:meth:`Crack.grown`).

*Conforming meshes* (Z-cracks): the body is meshed by MMG conforming to the plane of the
crack (:func:`crack_mesh`, or any mesh whose faces lie on it, such as six tetrahedra per
voxel), and the faces behind the front are opened by duplicating their nodes
(:func:`morphogen.fracture.cohesive.insert_interfaces`): free crack faces.

*Morphologies*: the crack meshes of a morphology carry its phases, those of the materials
(``mesh.phase``), and the sides of the crack apart (``mesh.side``), through the opened
meshes, the solvers and the transfers of their state: each element in the phase of its
centroid (:func:`with_phases`: staircase interfaces, as the grains of polycrystals), or the
interfaces of the phases conforming as well: the background mesh cut by MMG along the
level set of the morphology without remeshing (:func:`phase_mesh`, once), then along the
plane of each crack and remeshed in MMG's multi-material mode, each phase split into its two
sides (:func:`cut_mesh`). The elements of void phases (pores) are removed
(:func:`without_voids`); the walls of the pores and the interfaces of the phases keep their
nodes when the front is snapped.

*XFEM* (:class:`XFEM`; Moës, Dolbow & Belytschko 1999; Sukumar, Moës, Moran & Belytschko
2000): the displacement of linear tetrahedra is enriched, with shifted functions, by the
jump ``H(φ) - H(φ_i)`` at the nodes of the elements that the crack cuts through, and by
the branch functions ``√r (sin θ/2, cos θ/2, sin θ/2 sin θ, cos θ/2 sin θ)`` (``r =
sqrt(φ² + ψ²)``, ``θ = atan2(φ, ψ)``) at the nodes within ``tip_radius`` of the front; the
enriched elements are integrated on sub-tetrahedra of their two sides, refined near the
front.

*Energy release rate* (:func:`release_rate`): under prescribed forces, the potential
energy is ``-W/2`` with ``W = fᵀ u``, so that ``G = (W(a + δ) - W(a - δ)) / (2 (A(a + δ) -
A(a - δ)))``, the areas those of the crack inside the body (the opened faces, or the
crack surface in the cut elements): the mean of ``G`` along the front, for its uniform
advance; in mode I, ``K = sqrt(E' G)``.

*Quadratic elements* (``order=2`` of :func:`conforming_work`): the opened mesh is made of
10-node tetrahedra, and the mid-edge nodes of the edges that leave the front are moved to
a quarter of their length from it (:func:`quarter_points`; Barsoum 1976, Henshell & Shaw
1975): along such an edge the displacement of the isoparametric element is then a
polynomial of the square root of the distance to the front, the leading term of the
fields of a crack, which 4-node elements, of constant strain, only approach by their
refinement. The XFEM, whose enrichment carries that term, keeps its 4-node elements.

*Propagation* (:func:`propagate`): the front advances by steps, its rate given by Paris'
law ``da/dN = C ΔK^m`` (the cycles of each step) or by Griffith (the load factor at which
``G`` reaches ``G_c``, ``G`` growing as its square); the mesh is rebuilt at each step
(conforming) or kept (XFEM).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from morphogen.mesh.tetra import TET_EDGES, TET_FACES, TetMesh
from morphogen.mesh.tetra import tet_volumes as _tet_volumes

_Q4 = np.array(
    [
        [0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
        [0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
        [0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
        [0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685],
    ]
)


@dataclass(frozen=True)
class Crack:
    """A planar crack (see the module): ``kind`` ``"straight"`` (from ``point`` along
    ``direction`` over ``size``) or ``"penny"`` (centre ``point``, radius ``size``), its
    plane through ``point`` normal to ``normal``."""

    kind: str
    point: tuple
    normal: tuple
    size: float
    direction: tuple | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("straight", "penny"):
            raise ValueError("Crack: kind 'straight' or 'penny'")
        n = np.asarray(self.normal, dtype=float)
        object.__setattr__(self, "normal", tuple(n / np.linalg.norm(n)))
        object.__setattr__(self, "point", tuple(float(v) for v in self.point))
        if self.kind == "straight":
            if self.direction is None:
                raise ValueError("Crack: the direction of a straight crack")
            d = np.asarray(self.direction, dtype=float)
            d = d - (d @ n) * n / (n @ n)
            object.__setattr__(self, "direction", tuple(d / np.linalg.norm(d)))
        if not self.size > 0:
            raise ValueError("Crack: a positive size")

    def phi(self, x) -> np.ndarray:
        """Signed distance to the plane of the crack."""
        return (np.asarray(x, dtype=float) - self.point) @ np.asarray(self.normal)

    def psi(self, x) -> np.ndarray:
        """Signed distance to the front within the plane (negative on the crack)."""
        y = np.asarray(x, dtype=float) - self.point
        if self.kind == "straight":
            return y @ np.asarray(self.direction) - self.size
        n = np.asarray(self.normal)
        inplane = y - np.outer(y @ n, n) if y.ndim == 2 else y - (y @ n) * n
        return np.linalg.norm(inplane, axis=-1) - self.size

    def grown(self, da: float) -> Crack:
        """The crack whose front has moved by ``da``."""
        return replace(self, size=self.size + float(da))

    def frame(self, x):
        """At points ``x`` (rows): the distance ``r`` to the front, the axes of the nearest
        point of the front, ``n`` (normal to the crack), ``d`` (in its plane, the direction
        of propagation) and ``t = d × n`` (along the front), and the coordinate ``s`` along
        the front (a length, or the angle of a penny from the axis ``n × x``, ``n × y`` for
        a normal along ``x``)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = np.broadcast_to(np.asarray(self.normal), x.shape).copy()
        r = np.hypot(self.phi(x), self.psi(x))
        if self.kind == "straight":
            d = np.broadcast_to(np.asarray(self.direction), x.shape).copy()
            t = np.cross(d, n)
            return r, n, d, t, (x - self.point) @ t[0]
        y = x - self.point
        inplane = y - np.outer(y @ n[0], n[0])
        d = inplane / np.maximum(np.linalg.norm(inplane, axis=1, keepdims=True), 1e-300)
        e1 = np.cross(n[0], [1.0, 0.0, 0.0] if abs(n[0, 0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n[0], e1)
        return r, n, d, np.cross(d, n), np.arctan2(inplane @ e2, inplane @ e1)

    def front(self, lower, upper, count: int = 400):
        """Points of the front inside the box ``[lower, upper]`` and the length (or angle)
        of front that each represents."""
        if self.kind == "penny":
            n = np.asarray(self.normal)
            e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            a = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
            pts = np.asarray(self.point) + self.size * (
                np.outer(np.cos(a), e1) + np.outer(np.sin(a), e2)
            )
            step = np.full(count, 2 * np.pi * self.size / count)
        else:
            d, n = np.asarray(self.direction), np.asarray(self.normal)
            return line_in_box(np.asarray(self.point) + self.size * d, np.cross(d, n), lower, upper,
                               count)  # fmt: skip
        inside = np.all(
            (pts >= np.asarray(lower) - 1e-9) & (pts <= np.asarray(upper) + 1e-9), axis=1
        )
        return pts[inside], step[inside]


def line_in_box(point, direction, lower, upper, count: int = 400):
    """Points (midpoints of ``count`` equal parts) of the segment of the line through ``point``
    along ``direction`` inside the box ``[lower, upper]``, and the length of each part."""
    p0, t = np.asarray(point, dtype=float), np.asarray(direction, dtype=float)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    s_min, s_max = -np.inf, np.inf
    for i in range(3):
        if abs(t[i]) < 1e-12:
            if not lo[i] - 1e-9 <= p0[i] <= hi[i] + 1e-9:
                return np.zeros((0, 3)), np.zeros(0)
            continue
        a, b = sorted(((lo[i] - p0[i]) / t[i], (hi[i] - p0[i]) / t[i]))
        s_min, s_max = max(s_min, a), min(s_max, b)
    if not s_max > s_min:
        return np.zeros((0, 3)), np.zeros(0)
    step = (s_max - s_min) / count
    s = s_min + (np.arange(count) + 0.5) * step
    return p0 + np.outer(s, t), np.full(count, step)


# -- conforming meshes -----------------------------------------------------------------------


class _Plane:
    """A body whose level set is that of the plane of a crack (for MMG)."""

    def __init__(self, domain, crack: Crack):
        self.domain, self.crack = domain, crack

    def level_set(self, shape=None, *, voxel_size=None, grid=None):
        from morphogen.image import VoxelImage

        values = self.crack.phi(grid.center_points()).reshape(grid.shape)
        return VoxelImage(values, spacing=tuple(grid.spacing), origin=tuple(grid.origin))


def crack_mesh(lower, upper, crack: Crack, size: float, grains=None, *, morphology=None,
               voids=(), conforming: bool = False, front_size: float | None = None,
               front_radius: float | None = None, **options) -> TetMesh:  # fmt: skip
    """A tetrahedral mesh of the box ``[lower, upper]`` conforming to the plane of the
    crack (MMG; ``options`` those of :func:`morphogen.mesh.conforming.conforming_mesh`):
    the remeshing step of the conforming approach. Its two phases are the sides of the
    crack. ``grains``: a tessellation (a polycrystal) whose cells the elements take
    (:func:`with_grains`).

    ``morphology``: a microstructure whose phases the elements take (``mesh.phase``, the
    phases of the materials), the sides of the crack kept apart (``mesh.side``): each
    element in the phase of its centroid (:func:`with_phases`: staircase interfaces), or,
    with ``conforming``, a mesh conforming to the interfaces of its phases too, the
    background mesh cut by its level set (:func:`phase_mesh`) cut by the plane of the crack
    and remeshed by MMG (:func:`cut_mesh`; ``options`` among ``background``, ``min_size``,
    ``hausdorff``, ``gradation``, ``executable``, ``workdir``, ``timeout``); ``morphology``
    may be that mesh of its phases, made once for all the cracks of a growth. ``voids``:
    phases whose elements are removed (pores; :func:`without_voids`).

    ``front_size``: the size of the elements within ``front_radius`` of the front (by
    default four times that size), where the mesh is refined by a second pass of MMG with
    a map of sizes (:func:`refine_front`): the singular fields of a curved front resolved
    without a fine mesh of the whole body."""
    from morphogen.domain import Domain
    from morphogen.mesh.conforming import conforming_mesh

    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if morphology is None and len(voids):
        raise ValueError("crack_mesh: voids are phases of a morphology")
    if morphology is not None and (conforming or isinstance(morphology, TetMesh)):
        unknown = set(options) - set(_CUT_OPTIONS) - {"background"}
        if unknown:
            raise TypeError(f"crack_mesh of the phases: unexpected options {sorted(unknown)}")
        common = {k: v for k, v in options.items() if k in ("executable", "workdir", "timeout")}
        phases = morphology
        if not isinstance(phases, TetMesh):
            phases = phase_mesh(morphology, lower, upper, size,
                                background=options.get("background"), **common)  # fmt: skip
        elif not (np.allclose(phases.lower, lower) and np.allclose(phases.upper, upper)):
            raise ValueError("crack_mesh: the mesh of the phases is not that of the box")
        cut = {k: v for k, v in options.items() if k in _CUT_OPTIONS}
        mesh = without_voids(cut_mesh(phases, crack, size, **cut), voids)
        if front_size is not None:
            mesh = refine_front(mesh, crack, front_size, front_radius, **common)
        mesh = snap_front(mesh, crack)
    else:
        domain = Domain(size=tuple(upper - lower), origin=tuple(lower), periodic=False)
        mesh = conforming_mesh(
            _Plane(domain, crack), size=size, phases="all", periodic=False, **options
        )
        if front_size is not None:
            common = {k: v for k, v in options.items() if k in ("executable", "workdir", "timeout")}
            mesh = refine_front(mesh, crack, front_size, front_radius, **common)
        mesh = snap_front(mesh, crack)
        if morphology is not None:
            mesh = without_voids(with_phases(mesh, morphology, 0.25 * float(size)), voids)
    if grains is not None:
        mesh = with_grains(mesh, grains)
    # its two phases are the sides of the crack, or those of its materials and its sides
    mesh.metadata["crack_sides"] = "phase" if mesh.side is None else "side"
    return mesh


#: The options of :func:`cut_mesh` among those of :func:`crack_mesh`
_CUT_OPTIONS = ("min_size", "hausdorff", "gradation", "executable", "workdir", "timeout")


def phase_mesh(morphology, lower, upper, size: float, *, background: float | None = None,
               executable=None, workdir=None, timeout=None) -> TetMesh:  # fmt: skip
    """The background mesh of the box ``[lower, upper]`` (six tetrahedra per cube of a
    grid of spacing ``background``, by default ``size / 2``) cut by MMG along the zero level
    set of a morphology (two phases: negative in phase 1), without remeshing: a mesh
    conforming to the interfaces of its phases, its elements of any quality, to be cut by
    each crack and remeshed once (:func:`cut_mesh`; :func:`crack_mesh` with
    ``conforming``). The level set is sampled and moved away from zero near the nodes of
    the grid as for :func:`morphogen.mesh.conforming.conforming_mesh`; the plane of a
    crack is then cut on the same grid as by the crack meshes of a homogeneous body."""
    import tempfile
    from pathlib import Path

    from morphogen.domain import Domain
    from morphogen.io import medit
    from morphogen.mesh.conforming import (
        REF_INSIDE,
        SNAP,
        _Part,
        _run_mmg,
        _snap_level_set,
        background_mesh,
        mmg_executable,
        node_level_set,
    )

    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    box = Domain(size=tuple(upper - lower), origin=tuple(lower), periodic=False)
    body = _Part(morphology, box)
    spacing = float(background) if background is not None else 0.5 * float(size)
    values, shape = node_level_set(body, spacing)
    values = _snap_level_set(values, SNAP)
    nodes, tets, tri, tri_refs = background_mesh(shape, lower, upper)
    exe = mmg_executable(executable)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        medit.write_mesh(folder / "bg.mesh", nodes, tets, tri, tri_refs=tri_refs)
        medit.write_sol(folder / "bg.sol", values.ravel())
        data = _run_mmg(exe, folder / "bg.mesh", folder / "bg.sol", folder / "cut.mesh",
                        ["-ls", "0", "-nosurf", "-noinsert", "-noswap", "-nomove", "-v", "-1"],
                        timeout)  # fmt: skip
    xyz, elements = data["Vertices"], data["Tetrahedra"]
    tol = 1e-9 * float(np.max(upper - lower))
    for d in range(3):  # the nodes of the faces exactly on them
        xyz[:, d] = np.where(np.abs(xyz[:, d] - lower[d]) <= tol, lower[d], xyz[:, d])
        xyz[:, d] = np.where(np.abs(xyz[:, d] - upper[d]) <= tol, upper[d], xyz[:, d])
    flip = _tet_volumes(xyz, elements) < 0
    elements[flip] = elements[flip][:, [0, 2, 1, 3]]
    used, inverse = np.unique(elements, return_inverse=True)
    phase = np.where(data["Tetrahedra_refs"] == REF_INSIDE, 1, 0)
    spacings = [float(v) for v in (upper - lower) / np.array(shape)]
    metadata = {"mesher": "mmg3d -ls, cut without remeshing", "size": float(size),
                "background_spacing": spacings}  # fmt: skip
    return TetMesh(xyz[used], inverse.reshape(elements.shape), phase, lower, upper,
                   metadata=metadata)  # fmt: skip


def cut_mesh(mesh: TetMesh, crack, size: float, *, min_size=None, hausdorff=None,
             gradation: float = 2.0, executable=None, workdir=None,
             timeout=None) -> TetMesh:  # fmt: skip
    """The mesh of the phases of a morphology (linear tetrahedra) cut by the plane of a
    crack, ``φ = 0``, and remeshed by MMG in its multi-material mode, each phase split into
    its two sides (the sizes those of :func:`morphogen.mesh.conforming.conforming_mesh`):
    a mesh conforming to the interfaces of the phases and to the crack, its phases in
    ``mesh.phase`` and the sides of the crack in ``mesh.side`` (1 where ``φ < 0``, as the
    phases of :func:`crack_mesh`), the faces of the box kept plane. The values of ``φ``
    at the nodes closer to zero than a fraction of their variation along the edges are
    moved away from it, as on the grids of the conforming meshes (no sliver)."""
    import tempfile
    from pathlib import Path

    from morphogen.io import medit
    from morphogen.mesh.conforming import SNAP, _run_mmg, mmg_executable

    if mesh.order != 1:
        raise ValueError("cut_mesh: linear tetrahedra")
    x, tets = mesh.nodes, mesh.elements
    phi = crack.phi(x)
    edges = tets[:, TET_EDGES].reshape(-1, 2)
    jump = np.abs(phi[edges[:, 0]] - phi[edges[:, 1]])
    variation = np.zeros(len(x))
    np.maximum.at(variation, edges[:, 0], jump)
    np.maximum.at(variation, edges[:, 1], jump)
    threshold = SNAP * variation
    near = np.abs(phi) < threshold
    phi = np.where(near, np.where(phi < 0, -threshold, threshold), phi)
    faces = boundary_faces(mesh)
    lower, upper = mesh.lower, mesh.upper
    tol = mesh.tolerance()
    refs = np.full(len(faces), 7, dtype=np.int64)  # the walls of pores, if any
    for axis in range(3):
        for side, value in ((0, lower[axis]), (1, upper[axis])):
            on = np.all(np.abs(x[faces][:, :, axis] - value) <= tol, axis=1)
            refs[on] = 1 + 2 * axis + side
    phases = np.unique(mesh.phase)
    # each phase p (reference 1 + p) split into 101 + 2p where φ < 0 and 100 + 2p elsewhere
    split = "".join(f"{1 + p} {101 + 2 * p} {100 + 2 * p}\n" for p in phases.tolist())
    sizing = ["-hmax", f"{size:.10g}",
              "-hmin", f"{(min_size if min_size is not None else size / 10):.10g}",
              "-hausd", f"{(hausdorff if hausdorff is not None else size / 20):.10g}",
              "-hgrad", f"{float(gradation):.10g}", "-v", "-1"]  # fmt: skip
    exe = mmg_executable(executable)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        medit.write_mesh(folder / "cut.mesh", x, tets, faces, tet_refs=1 + mesh.phase,
                         tri_refs=refs)  # fmt: skip
        medit.write_sol(folder / "cut.sol", phi)
        (folder / "cut.mmg3d").write_text(f"LSReferences\n{len(phases)}\n{split}")
        data = _run_mmg(exe, folder / "cut.mesh", folder / "cut.sol", folder / "out.mesh",
                        ["-ls", "0", *sizing], timeout)  # fmt: skip
    xyz = data["Vertices"]
    for d in range(3):  # the nodes of the faces exactly on them
        xyz[:, d] = np.where(np.abs(xyz[:, d] - lower[d]) <= tol, lower[d], xyz[:, d])
        xyz[:, d] = np.where(np.abs(xyz[:, d] - upper[d]) <= tol, upper[d], xyz[:, d])
    elements, labels = data["Tetrahedra"], data["Tetrahedra_refs"] - 100
    if np.any(labels < 0):
        raise RuntimeError("cut_mesh: MMG left elements of no phase")
    flip = _tet_volumes(xyz, elements) < 0
    elements[flip] = elements[flip][:, [0, 2, 1, 3]]
    used, inverse = np.unique(elements, return_inverse=True)
    metadata = {**mesh.metadata, "mesher": "mmg3d -ls, multi-material", "size": float(size)}
    return TetMesh(xyz[used], inverse.reshape(elements.shape), labels // 2, lower, upper,
                   metadata=metadata, side=labels % 2)  # fmt: skip


def phases_at(morphology, points, spacing: float | None = None) -> np.ndarray:
    """The phase of a morphology at points (rows): its point queries (two phases), or the
    voxel of an image of it whose voxels are about ``spacing`` (at most 256 per edge; by
    default 1/128 of its longest edge); ``morphology`` may also be a function giving the
    phases of points."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if callable(morphology) and not hasattr(morphology, "domain"):
        return np.asarray(morphology(points), dtype=np.int64)
    names = getattr(morphology, "phase_names", None) or {0: "0", 1: "1"}
    if set(names) <= {0, 1}:
        try:
            return np.asarray(morphology.contains(points), dtype=bool).astype(np.int64)
        except NotImplementedError:
            pass
    lengths = np.asarray(morphology.domain.lengths, dtype=float)
    spacing = float(lengths.max()) / 128.0 if spacing is None else float(spacing)
    image = morphology.voxelize(voxel_size=max(spacing, float(lengths.max()) / 256.0))
    data = np.asarray(image.data)
    if data.dtype.kind not in "iub":
        raise ValueError("phases_at: the morphology gives no phase image")
    steps = np.asarray(image.spacing, dtype=float)
    index = np.floor((points - np.asarray(image.origin, dtype=float)) / steps).astype(np.int64)
    index = np.clip(index, 0, np.asarray(data.shape) - 1)
    return data[index[:, 0], index[:, 1], index[:, 2]].astype(np.int64)


def with_phases(mesh: TetMesh, morphology, spacing: float | None = None) -> TetMesh:
    """A crack mesh (its phases the sides of the crack) with the phases of a morphology:
    each element in the phase of the morphology at its centroid (``mesh.phase``, by
    :func:`phases_at`), the sides of the crack kept in ``mesh.side``. The interfaces of
    the phases are then staircases of element faces, as the grain boundaries of
    :func:`with_grains`."""
    centroids = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    side = mesh.phase if mesh.side is None else mesh.side
    return TetMesh(mesh.nodes, mesh.elements, phases_at(morphology, centroids, spacing),
                   mesh.lower, mesh.upper, periodic=mesh.periodic, metadata=dict(mesh.metadata),
                   cell=mesh.cell, orientations=mesh.orientations, side=side)  # fmt: skip


def without_voids(mesh: TetMesh, voids=(), islands: float = 0.01) -> TetMesh:
    """A mesh without the elements of the phases ``voids`` (pores), and without the parts
    of it they leave apart whose volume is below ``islands`` times that of the largest
    (more than one large part: an error, a body that falls apart)."""
    voids = [int(p) for p in voids]
    if not voids or not np.isin(mesh.phase, voids).any():
        return mesh
    out = mesh.select(~np.isin(mesh.phase, voids))
    if not out.n_elements:
        raise ValueError("void phases: no element left")
    labels, count = out.components()
    if count > 1:
        volumes = np.bincount(labels, np.abs(out.volumes()), minlength=count)
        large = np.flatnonzero(volumes >= islands * volumes.max())
        if len(large) > 1:
            raise ValueError(f"void phases {voids}: the body left falls apart into "
                             f"{len(large)} parts")  # fmt: skip
        out = out.select(labels == large[0])
    return out


def with_grains(mesh: TetMesh, grains) -> TetMesh:
    """The mesh with the grains of a tessellation (``grains``: its ``cell_of(points)`` and
    its ``orientations``): each element in the cell of its centroid (``mesh.cell``), the
    crystal orientations of the cells (``mesh.orientations``). The grain boundaries are
    then staircases of element faces, whose steps shrink with the elements; the crack
    meshes of a polycrystal stay conforming to the crack alone (MMG cuts one level set)."""
    centroids = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    cell = np.asarray(grains.cell_of(centroids), dtype=np.int64)
    return TetMesh(mesh.nodes, mesh.elements, mesh.phase, mesh.lower, mesh.upper,
                   periodic=mesh.periodic, metadata=dict(mesh.metadata), cell=cell,
                   orientations=getattr(grains, "orientations", None),
                   side=mesh.side)  # fmt: skip


def interface_nodes(mesh: TetMesh) -> np.ndarray:
    """The nodes (a mask) of the interfaces between the phases of a mesh and of its free
    faces off the faces of its box (the walls of its pores)."""
    tets = mesh.elements[:, :4]
    held = np.zeros(mesh.n_nodes, dtype=bool)
    phases = np.unique(mesh.phase)
    if len(phases) > 1:
        count = np.zeros(mesh.n_nodes, dtype=np.int64)
        for p in phases:
            touched = np.zeros(mesh.n_nodes, dtype=bool)
            touched[tets[mesh.phase == p].ravel()] = True
            count += touched
        held |= count > 1
    free = boundary_faces(mesh)
    x, tol = mesh.nodes[free], mesh.tolerance()
    on_box = np.zeros(len(free), dtype=bool)
    for axis in range(3):
        for value in (mesh.lower[axis], mesh.upper[axis]):
            on_box |= np.all(np.abs(x[:, :, axis] - value) <= tol, axis=1)
    held[free[~on_box].ravel()] = True
    return held


def refine_front(mesh: TetMesh, crack, size: float, radius: float | None = None, *,
                 gradation: float = 1.5, executable=None, workdir=None,
                 timeout=None) -> TetMesh:  # fmt: skip
    """A crack mesh (linear tetrahedra; its phases, or its phases and its sides) remeshed
    by MMG with elements of ``size`` within ``radius`` of the front of the crack (by
    default four times that size), growing beyond it by the ratio ``gradation`` between
    neighbours up to the sizes of the mesh: the surfaces between its labels (the surface
    of the crack, the interfaces of the phases, the walls of the pores) and the faces of
    the box are kept."""
    import tempfile
    from pathlib import Path

    from morphogen.io import medit
    from morphogen.mesh.conforming import _run_mmg, mmg_executable

    if mesh.order != 1:
        raise ValueError("refine_front: linear tetrahedra")
    size = float(size)
    radius = 4.0 * size if radius is None else float(radius)
    if not (size > 0 and radius > 0):
        raise ValueError("refine_front: a positive size and radius")
    x, tets = mesh.nodes, mesh.elements
    edges = tets[:, TET_EDGES].reshape(-1, 2)
    lengths = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1)
    counts = np.maximum(np.bincount(edges.ravel(), minlength=len(x)), 1)
    local = np.bincount(edges.ravel(), np.repeat(lengths, 2), minlength=len(x)) / counts
    r = crack.frame(x)[0]
    # the size of the front within the radius, then that of the mesh (MMG grades between)
    wanted = np.where(r <= radius, np.minimum(size, local), local)
    faces = boundary_faces(mesh)
    lower, upper = mesh.lower, mesh.upper
    tol = mesh.tolerance()
    refs = np.full(len(faces), 7, dtype=np.int64)  # the walls of pores, if any
    for axis in range(3):
        for side, value in ((0, lower[axis]), (1, upper[axis])):
            on = np.all(np.abs(x[faces][:, :, axis] - value) <= tol, axis=1)
            refs[on] = 1 + 2 * axis + side
    labels = mesh.phase if mesh.side is None else 2 * mesh.phase + mesh.side
    exe = mmg_executable(executable)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        medit.write_mesh(folder / "front.mesh", x, tets, faces, tet_refs=1 + labels,
                         tri_refs=refs)  # fmt: skip
        medit.write_sol(folder / "front.sol", wanted)
        args = ["-hgrad", f"{float(gradation):.10g}", "-hausd", f"{0.05 * size:.10g}", "-v", "-1"]
        data = _run_mmg(exe, folder / "front.mesh", folder / "front.sol", folder / "out.mesh",
                        args, timeout)  # fmt: skip
    xyz = data["Vertices"]
    for d in range(3):  # the nodes of the faces exactly on them
        xyz[:, d] = np.where(np.abs(xyz[:, d] - lower[d]) <= tol, lower[d], xyz[:, d])
        xyz[:, d] = np.where(np.abs(xyz[:, d] - upper[d]) <= tol, upper[d], xyz[:, d])
    elements, found = data["Tetrahedra"], data["Tetrahedra_refs"] - 1
    flip = _tet_volumes(xyz, elements) < 0
    elements[flip] = elements[flip][:, [0, 2, 1, 3]]
    used, inverse = np.unique(elements, return_inverse=True)
    metadata = {**mesh.metadata, "front_size": size, "front_radius": radius}
    if mesh.side is None:
        phase, side = found, None
    else:
        phase, side = found // 2, found % 2
    return TetMesh(xyz[used], inverse.reshape(elements.shape), phase, lower, upper,
                   periodic=mesh.periodic, metadata=metadata, side=side)  # fmt: skip


def snap_front(mesh: TetMesh, crack, fraction: float = 0.5, quality: float = 0.2) -> TetMesh:
    """The mesh with the nodes of the crack surface (between its two phases, the sides of
    the crack, or its two sides ``mesh.side``) within ``fraction`` of the length of their
    edges from the front moved onto it, in the surface: the opened faces then end on the
    front, instead of a zigzag that lengthens or shortens the crack locally by a fraction
    of an element (the remeshing of Z-cracks, whose meshes have the front along their
    edges). A node stays if the move would leave an element around it below ``quality``
    times its volume, a triangle of the surface below that fraction of its area, or a node
    of a face of the box off it; and, with its sides apart from its phases, the nodes of
    the interfaces of the phases and of the walls of the pores (:func:`interface_nodes`)."""
    labels = mesh.phase if mesh.side is None else mesh.side
    phases = np.unique(labels)
    if len(phases) != 2:
        return mesh
    x = mesh.nodes.copy()
    tets = mesh.elements[:, :4]
    sides = [np.zeros(len(x), dtype=bool) for _ in phases]
    for side, p in zip(sides, phases, strict=True):
        side[tets[labels == p].ravel()] = True
    surface = sides[0] & sides[1]
    if mesh.side is not None:
        surface &= ~interface_nodes(mesh)
    edges = tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2)
    lengths = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1)
    counts = np.maximum(np.bincount(edges.ravel(), minlength=len(x)), 1)
    size = np.bincount(edges.ravel(), np.repeat(lengths, 2), minlength=len(x)) / counts
    psi = crack.psi(x)
    candidates = np.flatnonzero(surface & (np.abs(psi) < fraction * size) & (psi != 0))
    if not len(candidates):
        return mesh
    directions = crack.frame(x[candidates])[2]
    per_node = np.bincount(tets.ravel(), minlength=len(x))
    start = np.concatenate([[0], np.cumsum(per_node)])
    owners = np.argsort(tets.ravel(), kind="stable") // 4
    span = float(np.max(mesh.upper - mesh.lower))
    faces_of = np.asarray(TET_FACES)
    moved = 0
    for k in np.argsort(np.abs(psi[candidates])):
        v = candidates[k]
        old = x[v].copy()
        new = old - psi[v] * directions[k]
        boxed = (np.abs(old - mesh.lower) < 1e-9 * span) | (np.abs(old - mesh.upper) < 1e-9 * span)
        if np.any(boxed & (np.abs(new - old) > 1e-12 * span)):
            continue
        star = tets[owners[start[v] : start[v + 1]]]
        faces = star[:, faces_of].reshape(-1, 3)
        faces = faces[np.any(faces == v, axis=1) & np.all(surface[faces], axis=1)]

        def shape(points, star=star, faces=faces):
            p = points[faces]
            return _tet_volumes(points, star), np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])

        volumes, normals = shape(x)
        x[v] = new
        moved_volumes, moved_normals = shape(x)
        kept = np.einsum("ij,ij->i", moved_normals, normals) > quality * np.einsum(
            "ij,ij->i", normals, normals
        )
        if np.all(moved_volumes / volumes > quality) and np.all(kept):
            moved += 1
        else:
            x[v] = old
    out = TetMesh(x, mesh.elements, mesh.phase, mesh.lower, mesh.upper, periodic=mesh.periodic,
                  metadata=dict(mesh.metadata), cell=mesh.cell,
                  orientations=mesh.orientations, side=mesh.side)  # fmt: skip
    out.metadata["front_nodes_moved"] = moved
    return out


def open_crack(mesh: TetMesh, crack: Crack, sides=None):
    """The mesh with the crack opened and the area of the crack (see :func:`crack_faces`)."""
    opened, faces = crack_faces(mesh, crack, sides)
    return opened, float(faces.areas(opened).sum())


def crack_faces(mesh: TetMesh, crack, sides=None):
    """The mesh with the faces of the crack opened, those between its two sides behind its
    front, and the pairs of faces of its lips (interface elements). ``sides``: a label per
    element (by default the sides ``mesh.side`` of a crack mesh of a morphology, or the two
    phases of a mesh from :func:`crack_mesh`, whose interface MMG may have moved slightly,
    otherwise the sign of ``φ`` at the centroids of the elements)."""
    from morphogen.fracture.cohesive import insert_interfaces

    if sides is None:
        if mesh.side is not None:
            sides = mesh.side
        elif mesh.metadata.get("crack_sides") == "phase":
            sides = mesh.phase
        else:
            sides = crack.phi(mesh.nodes[mesh.elements].mean(axis=1)) > 0
    return insert_interfaces(mesh, lambda c: crack.psi(c) < 0.0, sides=sides)


def boundary_faces(mesh: TetMesh) -> np.ndarray:
    """The faces of a tetrahedral mesh that belong to one element (rows of three nodes)."""
    faces = mesh.elements[:, np.asarray(TET_FACES)].reshape(-1, 3)
    keys = np.sort(faces, axis=1)
    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    return faces[first[counts == 1]]


def face_forces(mesh: TetMesh, where: Callable, traction) -> np.ndarray:
    """Nodal forces (3 n) of a uniform ``traction`` on the boundary faces whose centroids
    (those of their corners) satisfy ``where``: a third of the force of each face to each
    of its nodes for 4-node elements; for 10-node ones the consistent forces of the six
    nodes of the face, ``∫ N_a dA`` (a third to each mid-edge node of a plane face with
    straight edges, nothing to its corners)."""
    traction = np.asarray(traction, dtype=float)
    f = np.zeros((mesh.n_nodes, 3))
    if mesh.order == 1:
        faces = boundary_faces(mesh)
        x = mesh.nodes[faces]
        keep = np.asarray(where(x.mean(axis=1)), dtype=bool)
        areas = 0.5 * np.linalg.norm(
            np.cross(x[keep, 1] - x[keep, 0], x[keep, 2] - x[keep, 0]), axis=1
        )
        for k in range(3):
            np.add.at(f, faces[keep, k], np.outer(areas / 3.0, traction))
        return f.ravel()
    from morphogen.fracture.cohesive import _GAUSS6, _shape6

    faces = _boundary_faces6(mesh)
    x = mesh.nodes[faces]
    keep = np.asarray(where(x[:, :3].mean(axis=1)), dtype=bool)
    faces, x = faces[keep], x[keep]
    shape, dxi, deta = np.empty(6), np.empty(6), np.empty(6)
    for xi, eta, weight in _GAUSS6:
        _shape6(xi, eta, shape, dxi, deta)
        a = np.einsum("i,eic->ec", dxi, x)
        b = np.einsum("i,eic->ec", deta, x)
        area = 0.5 * weight * np.linalg.norm(np.cross(a, b), axis=1)
        for k in range(6):
            np.add.at(f, faces[:, k], np.outer(area * shape[k], traction))
    return f.ravel()


def _boundary_faces6(mesh: TetMesh) -> np.ndarray:
    """The faces of a mesh of 10-node tetrahedra that belong to one element: their
    corners, then the mid-edge nodes of their edges 0-1, 1-2 and 2-0."""
    corners = mesh.elements[:, np.asarray(TET_FACES)].reshape(-1, 3)
    keys = np.sort(corners, axis=1)
    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    first = first[counts == 1]
    element, face = first // 4, first % 4
    # the mid-edge node of each pair of corners of an element
    local = {(int(i), int(j)): 4 + k for k, (i, j) in enumerate(np.asarray(TET_EDGES))}
    local.update({(j, i): k for (i, j), k in list(local.items())})
    mids = np.empty((len(first), 3), dtype=np.int64)
    for f in range(4):
        at = face == f
        c = np.asarray(TET_FACES)[f]
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            mids[at, k] = mesh.elements[element[at], local[(int(c[i]), int(c[j]))]]
    return np.concatenate([corners[first], mids], axis=1)


def quarter_points(mesh: TetMesh, front) -> TetMesh:
    """The mesh of 10-node tetrahedra whose mid-edge nodes on the edges that leave the
    nodes ``front`` (those of a crack front: one end on it, the other not) are moved to a
    quarter of the edge from the front (Barsoum 1976): the displacements of these
    elements vary as the square root of the distance to the front along these edges, and
    their strains as its inverse, as the fields of a crack. The edges along the front and
    the others keep their mid-edge nodes."""
    if mesh.order != 2:
        raise ValueError("quarter_points: 10-node tetrahedra (mesh.quadratic())")
    on = np.zeros(mesh.n_nodes, dtype=bool)
    on[np.asarray(front, dtype=np.int64)] = True
    ends = mesh.elements[:, np.asarray(TET_EDGES)]  # (elements, 6, 2)
    mids = mesh.elements[:, 4:]
    a, b = ends[..., 0], ends[..., 1]
    nodes = mesh.nodes.copy()
    for tip, other in ((a, b), (b, a)):
        leave = on[tip] & ~on[other]
        nodes[mids[leave]] = 0.75 * mesh.nodes[tip[leave]] + 0.25 * mesh.nodes[other[leave]]
    return mesh._derived(nodes, mesh.elements)


def _isotropic(young: float, poisson: float) -> np.ndarray:
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = young / (2 * (1 + poisson))
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[np.arange(3), np.arange(3)] += 2 * mu
    d[np.arange(3, 6), np.arange(3, 6)] = mu
    return d


def _solve(k: sp.spmatrix, f: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    n = k.shape[0]
    free = np.setdiff1d(np.arange(n), fixed)
    u = np.zeros(n)
    u[free] = spla.spsolve(k[free][:, free].tocsc(), f[free])
    return u


def conforming_work(
    mesh: TetMesh,
    material,
    crack: Crack,
    forces,
    supports,
    *,
    order: int = 1,
    quarter_point: bool = True,
) -> tuple[float, float]:
    """``W = fᵀ u`` of the body with the crack opened on its (conforming) mesh, and the
    area of the crack: ``forces(mesh)`` the nodal forces and ``supports(mesh)`` the fixed
    dofs (functions of the opened mesh). ``order`` 2: on the 10-node tetrahedra of the
    mesh, those of the front with their mid-edge nodes at the quarter of their edges
    (``quarter_point``; see :func:`quarter_points`)."""
    from morphogen.homogenization.fe import assemble

    if order not in (1, 2):
        raise ValueError("conforming_work: order 1 or 2")
    if order == 1:
        opened, area = open_crack(mesh.linear(), crack)
    else:
        opened, faces = crack_faces(mesh.quadratic(), crack)
        area = float(faces.areas(opened).sum())
        if quarter_point:  # the nodes that the two lips share: the front
            front = np.intersect1d(faces.minus[:, :3], faces.plus[:, :3])
            opened = quarter_points(opened, front)
    d = _isotropic(material.young, material.poisson)
    k = assemble(opened, 3, np.zeros(opened.n_elements, dtype=np.int64), d[None]).tocsr()
    f = forces(opened)
    u = _solve(k, f, np.asarray(supports(opened), dtype=np.int64))
    return float(f @ u), area


# -- XFEM ----------------------------------------------------------------------------------


def _split(x: np.ndarray, phi: np.ndarray):
    """Sub-tetrahedra of a tetrahedron cut by the plane ``φ = 0`` (linear ``φ``): rows of
    four vertices, and the side (+1, -1) of each."""
    pos = phi > 0
    if pos.all() or (~pos).all():
        return [x], [1.0 if pos.all() else -1.0]

    def cut(i, j):
        return x[i] + phi[i] / (phi[i] - phi[j]) * (x[j] - x[i])

    def prism(a, b, c, pa, pb, pc):  # triangles (a, b, c) and (pa, pb, pc) facing each other
        return [np.array([a, b, c, pa]), np.array([b, c, pa, pb]), np.array([c, pa, pb, pc])]

    ip = np.flatnonzero(pos)
    im = np.flatnonzero(~pos)
    if len(ip) in (1, 3):
        lone, others = (ip[0], im) if len(ip) == 1 else (im[0], ip)
        side_lone = 1.0 if len(ip) == 1 else -1.0
        p = [cut(lone, o) for o in others]
        tets = [np.array([x[lone], p[0], p[1], p[2]])]
        sides = [side_lone]
        rest = prism(x[others[0]], x[others[1]], x[others[2]], p[0], p[1], p[2])
        return tets + rest, sides + [-side_lone] * 3
    a, b = ip
    c, d = im
    pac, pad, pbc, pbd = cut(a, c), cut(a, d), cut(b, c), cut(b, d)
    plus = prism(x[a], pac, pad, x[b], pbc, pbd)
    minus = prism(x[c], pac, pbc, x[d], pad, pbd)
    return plus + minus, [1.0] * 3 + [-1.0] * 3


def _refine(tet: np.ndarray) -> list:
    """Eight sub-tetrahedra by the midpoints of the edges."""
    a, b, c, d = tet
    ab, ac, ad, bc, bd, cd = (
        (a + b) / 2,
        (a + c) / 2,
        (a + d) / 2,
        (b + c) / 2,
        (b + d) / 2,
        (c + d) / 2,
    )
    return [
        np.array(t)
        for t in (
            [a, ab, ac, ad],
            [ab, b, bc, bd],
            [ac, bc, c, cd],
            [ad, bd, cd, d],
            [ab, ac, ad, bd],
            [ab, ac, bc, bd],
            [ac, ad, bd, cd],
            [ac, bc, bd, cd],
        )
    ]


def _branch(phi, psi, gphi, gpsi):
    """The four branch functions and their gradients (rows) at points of level sets
    ``phi``, ``psi`` (gradients ``gphi``, ``gpsi``)."""
    r = np.sqrt(phi**2 + psi**2)
    r = np.maximum(r, 1e-14)
    th = np.arctan2(phi, psi)
    sr = np.sqrt(r)
    s2, c2, s, c = np.sin(th / 2), np.cos(th / 2), np.sin(th), np.cos(th)
    f = np.stack([sr * s2, sr * c2, sr * s2 * s, sr * c2 * s], axis=-1)
    dfr = np.stack([s2 / (2 * sr), c2 / (2 * sr), s2 * s / (2 * sr), c2 * s / (2 * sr)], axis=-1)
    dft = np.stack(
        [sr * c2 / 2, -sr * s2 / 2, sr * (c2 / 2 * s + s2 * c), sr * (-s2 / 2 * s + c2 * c)],
        axis=-1,
    )
    dr = (phi[:, None] * gphi + psi[:, None] * gpsi) / r[:, None]
    dt = (psi[:, None] * gphi - phi[:, None] * gpsi) / r[:, None] ** 2
    grad = dfr[:, :, None] * dr[:, None, :] + dft[:, :, None] * dt[:, None, :]
    return f, grad


def _b_rows(c: int, g: np.ndarray) -> np.ndarray:
    """Voigt strains (11, 22, 33, 23, 31, 12; engineering shears) of the displacement
    component ``c`` with the gradients ``g`` (points x 3): points x 6."""
    out = np.zeros((len(g), 6))
    out[:, c] = g[:, c]
    pairs = {3: (1, 2), 4: (2, 0), 5: (0, 1)}
    for row, (i, j) in pairs.items():
        if c == i:
            out[:, row] += g[:, j]
        if c == j:
            out[:, row] += g[:, i]
    return out


class XFEM:
    """A cracked body by the extended finite element method (see the module): a linear
    tetrahedral ``mesh`` of an isotropic ``material``, the ``crack``, the radius
    ``tip_radius`` within which the nodes get the branch functions (by default twice the
    mean edge) and the number of refinements of the sub-tetrahedra of the elements near the
    front (``refine``)."""

    def __init__(self, mesh: TetMesh, material, crack: Crack, *, tip_radius=None, refine: int = 0):
        from morphogen.fracture.cohesive import _tet_gradients

        if mesh.order != 1:
            raise ValueError("XFEM: linear tetrahedra")
        self.mesh, self.crack = mesh, crack
        self.d = _isotropic(material.young, material.poisson)
        x, elems = mesh.nodes, mesh.elements
        self.grads, self.volumes = _tet_gradients(x, elems)
        h = float(np.mean(np.linalg.norm(x[elems[:, 1]] - x[elems[:, 0]], axis=1)))
        self.tip_radius = 2.0 * h if tip_radius is None else float(tip_radius)
        self.refine = int(refine)
        phi = crack.phi(x)
        phi = np.where(np.abs(phi) < 1e-9 * h, 1e-9 * h, phi)  # no node on the plane
        psi = crack.psi(x)
        self.phi, self.psi = phi, psi
        pe, se = phi[elems], psi[elems]
        cut = (pe.min(axis=1) < 0) & (pe.max(axis=1) > 0)
        near = np.sqrt(phi**2 + psi**2) < self.tip_radius
        tip_elements = cut & (se.min(axis=1) <= 0) & (se.max(axis=1) >= 0)
        tip = np.zeros(mesh.n_nodes, dtype=bool)
        tip[np.unique(elems[tip_elements])] = True
        tip |= near
        split = cut & (se.max(axis=1) < 0)
        jump = np.zeros(mesh.n_nodes, dtype=bool)
        jump[np.unique(elems[split])] = True
        jump &= ~tip
        self.jump, self.tip = jump, tip
        offset = np.full(mesh.n_nodes, -1, dtype=np.int64)
        count = 3 * mesh.n_nodes
        for i in range(mesh.n_nodes):
            if jump[i]:
                offset[i], count = count, count + 3
            elif tip[i]:
                offset[i], count = count, count + 12
        self.offset, self.n_dofs = offset, count
        self.enriched = np.flatnonzero((jump | tip)[elems].any(axis=1))
        self.cut = cut

    def _points(self, e: int):
        """Quadrature points (rows), weights and sides of an enriched element."""
        x = self.mesh.nodes[self.mesh.elements[e]]
        phi = self.phi[self.mesh.elements[e]]
        subs, sides = _split(x, phi)
        near_front = self.tip[self.mesh.elements[e]].any()
        if near_front:
            for _ in range(self.refine):
                refined, refined_sides = [], []
                for t, s in zip(subs, sides, strict=True):
                    pieces = _refine(t)
                    refined += pieces
                    refined_sides += [s] * len(pieces)
                subs, sides = refined, refined_sides
        pts, wts, sgn = [], [], []
        for t, s in zip(subs, sides, strict=True):
            vol = abs(np.linalg.det(np.stack([t[1] - t[0], t[2] - t[0], t[3] - t[0]]))) / 6.0
            if vol <= 0:
                continue
            pts.append(_Q4 @ t)
            wts.append(np.full(4, vol / 4.0))
            sgn.append(np.full(4, s))
        return np.concatenate(pts), np.concatenate(wts), np.concatenate(sgn)

    def stiffness(self) -> sp.csr_matrix:
        """The stiffness of all the dofs: the standard ones (``3 n``) then the enriched."""
        from morphogen.homogenization.fe import assemble

        mesh = self.mesh
        k_std = assemble(mesh, 3, np.zeros(mesh.n_elements, dtype=np.int64), self.d[None])
        rows, cols, vals = [], [], []
        for e in self.enriched:
            nodes = mesh.elements[e]
            g = self.grads[e]
            pts, wts, sgn = self._points(e)
            x0 = mesh.nodes[nodes[0]]
            # barycentric coordinates of the points: N = 1 - sum, from the gradients
            nval = np.empty((len(pts), 4))
            nval[:, 1:] = (pts - x0) @ g[1:].T
            nval[:, 0] = 1.0 - nval[:, 1:].sum(axis=1)
            phi_q, psi_q = nval @ self.phi[nodes], nval @ self.psi[nodes]
            gphi, gpsi = self.phi[nodes] @ g, self.psi[nodes] @ g
            columns, dofs = [], []
            for a, i in enumerate(nodes):
                for c in range(3):
                    columns.append(_b_rows(c, np.tile(g[a], (len(pts), 1))))
                    dofs.append(3 * i + c)
                if self.jump[i]:
                    hi = 1.0 if self.phi[i] > 0 else -1.0
                    for c in range(3):
                        columns.append(_b_rows(c, (sgn - hi)[:, None] * g[a][None]))
                        dofs.append(self.offset[i] + c)
                elif self.tip[i]:
                    f, gf = _branch(phi_q, psi_q, gphi[None], gpsi[None])
                    fi, _ = _branch(
                        self.phi[i : i + 1], self.psi[i : i + 1], gphi[None], gpsi[None]
                    )
                    for kf in range(4):
                        grad = (f[:, kf] - fi[0, kf])[:, None] * g[a][None] + nval[
                            :, a : a + 1
                        ] * gf[:, kf]
                        for c in range(3):
                            columns.append(_b_rows(c, grad))
                            dofs.append(self.offset[i] + 3 * kf + c)
            b = np.stack(columns, axis=2)  # points x 6 x columns
            ke = np.einsum("q,qia,ij,qjb->ab", wts, b, self.d, b)
            dofs = np.asarray(dofs)
            enriched = dofs >= 3 * mesh.n_nodes
            keep = enriched[:, None] | enriched[None, :]  # the standard block is in k_std
            r, cc = np.meshgrid(dofs, dofs, indexing="ij")
            rows.append(r[keep])
            cols.append(cc[keep])
            vals.append(ke[keep])
        n = self.n_dofs
        k = (
            sp.csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
            )
            if rows
            else sp.csr_matrix((n, n))
        )
        k_std = sp.csr_matrix((k_std.data, k_std.indices, k_std.indptr), shape=k_std.shape)
        k_std.resize((n, n))
        return (k + k_std).tocsr()

    def crack_area(self) -> float:
        """The area of the crack inside the body: the parts behind the front of its sections
        by the cut elements (polygons clipped by ``ψ = 0``)."""
        total = 0.0
        x = self.mesh.nodes
        for e in np.flatnonzero(self.cut):
            nodes = self.mesh.elements[e]
            p, s = self.phi[nodes], self.psi[nodes]
            points, values = [], []
            for i in range(4):
                for j in range(i + 1, 4):
                    if (p[i] > 0) != (p[j] > 0):
                        t = p[i] / (p[i] - p[j])
                        points.append(x[nodes[i]] + t * (x[nodes[j]] - x[nodes[i]]))
                        values.append(s[i] + t * (s[j] - s[i]))
            pts = np.array(points)
            centre = pts.mean(axis=0)
            normal = np.asarray(self.crack.normal)
            ref = pts[0] - centre
            ang = np.arctan2(np.cross(ref, pts - centre) @ normal, (pts - centre) @ ref)
            order = np.argsort(ang)
            poly, vals = pts[order], np.asarray(values)[order]
            clipped = []  # Sutherland-Hodgman on psi < 0
            for k in range(len(poly)):
                a, b = poly[k], poly[(k + 1) % len(poly)]
                va, vb = vals[k], vals[(k + 1) % len(poly)]
                if va < 0:
                    clipped.append(a)
                if (va < 0) != (vb < 0):
                    clipped.append(a + va / (va - vb) * (b - a))
            if len(clipped) >= 3:
                c = np.array(clipped)
                for k in range(1, len(c) - 1):
                    total += 0.5 * np.linalg.norm(np.cross(c[k] - c[0], c[k + 1] - c[0]))
        return total

    def work(self, forces: np.ndarray, supports) -> float:
        """``W = fᵀ u`` under the nodal forces (``3 n``) with the fixed standard dofs (the
        enriched dofs of the same components of their nodes fixed too)."""
        n3 = 3 * self.mesh.n_nodes
        fixed = np.asarray(supports, dtype=np.int64)
        extra = []
        for dof in fixed:
            i, c = int(dof) // 3, int(dof) % 3
            if self.offset[i] >= 0:
                functions = 1 if self.jump[i] else 4
                extra.extend(self.offset[i] + 3 * k + c for k in range(functions))
        f = np.zeros(self.n_dofs)
        f[:n3] = forces
        u = _solve(self.stiffness(), f, np.concatenate([fixed, np.asarray(extra, dtype=np.int64)]))
        self.solution = u
        return float(f @ u)


# -- energy release rate and propagation ------------------------------------------------------


def release_rate(
    work_area: Callable[[Crack], tuple[float, float]], crack: Crack, delta: float
) -> float:
    """The mean energy release rate along the front, ``(W(a + δ) - W(a - δ)) / (2 (A(a + δ) -
    A(a - δ)))``, from a function giving ``W = fᵀ u`` and the area of the crack."""
    w_plus, a_plus = work_area(crack.grown(delta))
    w_minus, a_minus = work_area(crack.grown(-delta))
    if not a_plus > a_minus:
        raise ValueError("release_rate: the crack area must grow (a larger delta for the mesh)")
    return 0.5 * (w_plus - w_minus) / (a_plus - a_minus)


def xfem_work(mesh: TetMesh, material, forces, supports, **options) -> Callable:
    """``crack -> (W, area)`` by XFEM on a fixed mesh (``forces``, ``supports``: functions of
    the mesh)."""
    f = forces(mesh)
    fixed = supports(mesh)

    def run(crack: Crack):
        model = XFEM(mesh, material, crack, **options)
        return model.work(f, fixed), model.crack_area()

    return run


def conforming_work_of(
    mesh_of: Callable[[Crack], TetMesh], material, forces, supports, **options
) -> Callable:
    """``crack -> (W, area)`` on conforming meshes, ``mesh_of(crack)`` meshing (or remeshing,
    :func:`crack_mesh`) the body for each crack; ``options``: those of
    :func:`conforming_work` (``order=2`` for 10-node elements, with quarter points)."""

    def run(crack: Crack):
        return conforming_work(mesh_of(crack), material, crack, forces, supports, **options)

    return run


@dataclass
class Propagation:
    """The steps of a propagation: crack sizes, energy release rates, stress intensity
    factors (mode I), cycles (Paris) or critical load factors (Griffith)."""

    sizes: np.ndarray
    rates: np.ndarray
    factors: np.ndarray
    cycles: np.ndarray
    loads: np.ndarray

    def rows(self) -> list[dict]:
        return [
            {"a": float(a), "G": float(g), "K": float(k), "N": float(n), "load": float(lam)}
            for a, g, k, n, lam in zip(
                self.sizes, self.rates, self.factors, self.cycles, self.loads, strict=True
            )
        ]


def propagate(
    work_area: Callable[[Crack], tuple[float, float]],
    crack: Crack,
    *,
    young: float,
    poisson: float,
    increment: float,
    steps: int,
    delta: float | None = None,
    paris: tuple[float, float] | None = None,
    toughness: float | None = None,
    plane_strain: bool = True,
) -> Propagation:
    """Grow a crack by ``steps`` uniform advances of its front by ``increment`` (see the
    module): ``G`` at each size by :func:`release_rate` (``delta``: half the advance of its
    differences, by default ``increment / 2``), ``K = sqrt(E' G)`` (``E' = E / (1 - ν²)`` in
    plane strain), the cycles of Paris' law ``da/dN = C K^m`` (``paris = (C, m)``, the
    loading of the forces as the range) and the load factor ``sqrt(G_c / G)`` at which
    Griffith's criterion is met (``toughness`` G_c)."""
    e_prime = young / (1 - poisson**2) if plane_strain else young
    delta = increment / 2 if delta is None else delta
    sizes, rates, factors, cycles, loads = [], [], [], [], []
    n = 0.0
    for _ in range(int(steps) + 1):
        g = release_rate(work_area, crack, delta)
        k = math.sqrt(max(e_prime * g, 0.0))
        sizes.append(crack.size)
        rates.append(g)
        factors.append(k)
        loads.append(math.sqrt(toughness / g) if toughness else math.nan)
        cycles.append(n)
        if paris is not None:
            c, m = paris
            n += increment / (c * k**m)
        crack = crack.grown(increment)
    return Propagation(
        np.array(sizes), np.array(rates), np.array(factors), np.array(cycles), np.array(loads)
    )


def sent_factor(ratio: float) -> float:
    """The geometry factor of the single edge notched plate in uniform tension, ``K = σ
    sqrt(π a) F(a/W)`` (Tada, Paris & Irwin; 0.5 % for a/W < 0.6)."""
    r = float(ratio)
    return 1.12 - 0.231 * r + 10.55 * r**2 - 21.72 * r**3 + 30.39 * r**4
