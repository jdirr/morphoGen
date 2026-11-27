"""Triangular meshes of planar structures, in the plane ``z`` of their box.

A structure of the plane — the section of a model at a height ``z``, the columnar grains of
seeds of the plane (``seeded_laguerre`` with ``dimension: 2``), a tessellation whose germs
lie in one plane — is meshed in that plane by triangles (:class:`TriMesh`: three- or
six-node triangles, a phase each, a grain each for polycrystals with their crystal
orientations), conforming to its interfaces and, for a periodic structure, with matching
nodes on opposite sides, so that periodic conditions are exact (:func:`plane_mesh`):

* the grains of a tessellation whose germs lie in one plane are exact polygons: the facets
  of its exact cells on the face ``z0`` of the box
  (:meth:`~morphogen.models.tessellation.Tessellation.cell_complex`), the cells of the
  spheres of a seed merged into its grain. Their edges between grains and the sides of the
  box are subdivided at the element size — the sides alike on opposite sides — and the
  polygons fanned from their centroids; ``mmg2d`` remeshes this triangulation, keeping the
  corners of the polygons and, for a periodic structure, the edges of the sides
  (:func:`tessellation_mesh`);
* any other structure (:func:`section_mesh`): the implicit functions of its phases are
  sampled at the nodes of a background triangulation of the rectangle (a grid of squares
  cut by alternating diagonals) in the plane ``z``, and cut one after another by the
  level sets of ``mmg2d`` (the multi-material level sets of MMG: a phase is where its
  function is the least, the phase of the largest index the rest), without remeshing;
  ``mmg2d`` then remeshes the triangulation within the Hausdorff distance of the
  interfaces, keeping the edges of the sides of a periodic structure, which the cut left
  matching (the same values on opposite sides).

:meth:`TriMesh.extruded` makes a layer of tetrahedra of it (prisms cut into three), whose
finite elements in space reproduce those of the plane in plane strain (the verification of
:mod:`morphogen.homogenization.plane`).
"""

from __future__ import annotations

import itertools
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from morphogen.domain import Domain, Grid
from morphogen.io import medit

log = logging.getLogger("morphogen")

#: The references of the edges of the sides of the rectangle: x0, x1, y0, y1
SIDE_REFS = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
#: MMG's references of the triangles cut by a level set, and of the edges of the interface
REF_INSIDE, REF_OUTSIDE, REF_INTERFACE = 3, 2, 10
#: The references of the triangles of a mesh of several phases: this plus their phase
REF_PHASE = 20
#: Level-set values closer to zero than this fraction of an edge of the background
#: triangulation are moved away from zero (no interface through a node: no slivers)
SNAP = 0.2
#: The growth of the sizes of the elements along the edges of the polygons, away from their
#: small features (an element at most 1 + GRADE times its neighbour)
GRADE = 0.25
#: The middles of the edges of a six-node triangle: nodes 3, 4, 5 on the edges 01, 12, 20
TRI6_EDGES = ((0, 1), (1, 2), (2, 0))


def _signed_areas(nodes: np.ndarray, corners: np.ndarray) -> np.ndarray:
    a = nodes[corners[:, 1]] - nodes[corners[:, 0]]
    b = nodes[corners[:, 2]] - nodes[corners[:, 0]]
    return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


@dataclass
class TriMesh:
    """Triangular mesh of the rectangle ``[lower, upper]`` of the plane ``z``, a phase per
    element: three-node triangles counterclockwise, or six-node ones (the corners, then the
    middles of the edges 01, 12, 20). Meshes of polycrystals give the grain of each element
    (``cell``) and the crystal orientation of each grain (``orientations``: Bunge angles in
    degrees). ``thickness``: the extent of the structure across its plane (the sections of
    the files of solids, and the volumes of :meth:`extruded`); ``periodic``: whether the
    structure is periodic (its opposite sides then have matching nodes)."""

    nodes: np.ndarray
    elements: np.ndarray
    phase: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    periodic: bool = False
    metadata: dict = field(default_factory=dict)
    cell: np.ndarray | None = None
    orientations: np.ndarray | None = None
    z: float = 0.0
    thickness: float = 1.0

    def __post_init__(self) -> None:
        self.nodes = np.ascontiguousarray(self.nodes, dtype=np.float64).reshape(-1, 2)
        self.elements = np.array(self.elements, dtype=np.int64)  # a copy: reoriented below
        self.phase = np.ascontiguousarray(self.phase, dtype=np.int64)
        self.lower = np.asarray(self.lower, dtype=np.float64).reshape(2)
        self.upper = np.asarray(self.upper, dtype=np.float64).reshape(2)
        if self.elements.ndim != 2 or self.elements.shape[1] not in (3, 6):
            raise ValueError("triangles need 3 (linear) or 6 (quadratic) nodes")
        if self.phase.shape != (len(self.elements),):
            raise ValueError("one phase per element")
        if self.cell is not None:
            self.cell = np.ascontiguousarray(self.cell, dtype=np.int64)
            if self.cell.shape != (len(self.elements),):
                raise ValueError("one cell label per element")
        if self.orientations is not None:
            self.orientations = np.ascontiguousarray(self.orientations, dtype=np.float64)
            if self.orientations.ndim != 2 or self.orientations.shape[1] != 3:
                raise ValueError("orientations: three Euler angles per cell")
        flip = _signed_areas(self.nodes, self.elements[:, :3]) < 0
        if flip.any():  # counterclockwise
            order = [0, 2, 1] if self.order == 1 else [0, 2, 1, 5, 4, 3]
            self.elements[flip] = self.elements[flip][:, order]
        self.z = float(self.z)
        self.thickness = float(self.thickness)

    # -- sizes ---------------------------------------------------------------------------

    @property
    def order(self) -> int:
        return 1 if self.elements.shape[1] == 3 else 2

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def size(self) -> np.ndarray:
        return self.upper - self.lower

    def box_area(self) -> float:
        return float(np.prod(self.size))

    @property
    def tolerance(self) -> float:
        return 1e-9 * float(self.size.max())

    def areas(self) -> np.ndarray:
        """The areas of the elements (of their straight corners)."""
        return _signed_areas(self.nodes, self.elements[:, :3])

    def quality(self) -> np.ndarray:
        """``4 √3 A / Σ ℓ²`` of each element: 1 for an equilateral triangle, 0 for a flat
        one."""
        p = self.nodes[self.elements[:, :3]]
        lengths = sum(np.sum((p[:, (k + 1) % 3] - p[:, k]) ** 2, axis=1) for k in range(3))
        return 4.0 * np.sqrt(3.0) * self.areas() / np.maximum(lengths, 1e-300)

    def min_angles(self) -> np.ndarray:
        """The smallest angle of each element (degrees)."""
        p = self.nodes[self.elements[:, :3]]
        out = np.full(self.n_elements, np.inf)
        for k in range(3):
            u = p[:, (k + 1) % 3] - p[:, k]
            v = p[:, (k + 2) % 3] - p[:, k]
            cos = np.einsum("ij,ij->i", u, v) / np.maximum(
                np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300
            )
            out = np.minimum(out, np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return out

    def phase_fractions(self) -> dict[int, float]:
        a = self.areas()
        return {int(p): float(a[self.phase == p].sum() / a.sum()) for p in np.unique(self.phase)}

    # -- edges and sides -----------------------------------------------------------------

    def edges(self) -> np.ndarray:
        """The edges of the elements (sorted pairs of corners), each once."""
        c = self.elements[:, :3]
        pairs = np.concatenate([c[:, [a, b]] for a, b in TRI6_EDGES])
        return np.unique(np.sort(pairs, axis=1), axis=0)

    def boundary_edges(self) -> np.ndarray:
        """The edges of a single element (pairs of corners, counterclockwise about it)."""
        c = self.elements[:, :3]
        pairs = np.concatenate([c[:, [a, b]] for a, b in TRI6_EDGES])
        key = np.sort(pairs, axis=1)
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        return pairs[counts[inverse.ravel()] == 1]

    def side_nodes(self, axis: int, side: int) -> np.ndarray:
        """The nodes on the side ``x = lower`` (``axis`` 0, ``side`` 0), ``x = upper``,
        ``y = lower`` or ``y = upper``."""
        value = self.lower[axis] if side == 0 else self.upper[axis]
        return np.flatnonzero(np.abs(self.nodes[:, axis] - value) <= self.tolerance)

    def side_edges(self, axis: int, side: int) -> np.ndarray:
        """The edges of the elements on a side (:meth:`side_nodes`): their pairs of corners
        counterclockwise about the mesh, then their middle nodes for six-node meshes
        (``(k, 2)`` or ``(k, 3)``)."""
        value = self.lower[axis] if side == 0 else self.upper[axis]
        on = np.abs(self.nodes[:, axis] - value) <= self.tolerance
        c = self.elements
        out = []
        for k, (i, j) in enumerate(TRI6_EDGES):
            hit = on[c[:, i]] & on[c[:, j]]
            out.append(c[hit][:, [i, j] if self.order == 1 else [i, j, 3 + k]])
        return np.concatenate(out)

    def periodic_pairs(self, axis: int) -> np.ndarray:
        """The pairs of nodes facing each other across the sides normal to ``axis``
        (``(k, 2)``: on the side 0, on the side 1)."""
        lo, hi = self.side_nodes(axis, 0), self.side_nodes(axis, 1)
        if not len(lo) or not len(hi):
            return np.zeros((0, 2), dtype=np.int64)
        other = 1 - axis
        dist, j = cKDTree(self.nodes[hi, other][:, None]).query(self.nodes[lo, other][:, None])
        keep = dist <= 1e3 * self.tolerance
        return np.column_stack([lo[keep], hi[j[keep]]])

    def matching_sides(self) -> bool:
        """Whether every node of a side has its translate on the opposite side."""
        for axis in (0, 1):
            pairs = self.periodic_pairs(axis)
            lo, hi = self.side_nodes(axis, 0), self.side_nodes(axis, 1)
            if len(pairs) != len(lo) or len(pairs) != len(hi):
                return False
        return True

    # -- derived meshes ------------------------------------------------------------------

    def select(self, keep) -> TriMesh:
        """The mesh of the elements ``keep`` (a mask or indices; unused nodes removed)."""
        elements = self.elements[keep]
        used, inverse = np.unique(elements, return_inverse=True)
        return TriMesh(self.nodes[used], inverse.reshape(elements.shape), self.phase[keep],
                       self.lower, self.upper, self.periodic, dict(self.metadata),
                       None if self.cell is None else self.cell[keep], self.orientations,
                       self.z, self.thickness)  # fmt: skip

    def subset(self, phases) -> TriMesh:
        """The mesh of the elements of the given phases."""
        return self.select(np.isin(self.phase, list(phases)))

    def quadratic(self) -> TriMesh:
        """The six-node mesh: a node at the middle of each edge, shared."""
        if self.order == 2:
            return self
        c = self.elements
        pairs = np.concatenate([c[:, [a, b]] for a, b in TRI6_EDGES])
        key = np.sort(pairs, axis=1)
        unique, inverse = np.unique(key, axis=0, return_inverse=True)
        mids = self.n_nodes + inverse.ravel().reshape(3, -1).T
        nodes = np.vstack([self.nodes, 0.5 * (self.nodes[unique[:, 0]] + self.nodes[unique[:, 1]])])
        return TriMesh(nodes, np.hstack([c, mids]), self.phase, self.lower, self.upper,
                       self.periodic, dict(self.metadata), self.cell, self.orientations,
                       self.z, self.thickness)  # fmt: skip

    def linear(self) -> TriMesh:
        """The three-node mesh of the corners."""
        if self.order == 1:
            return self
        nodes, corners = _corner_mesh(self.nodes, self.elements[:, :3])
        return TriMesh(nodes, corners, self.phase, self.lower, self.upper, self.periodic,
                       dict(self.metadata), self.cell, self.orientations, self.z,
                       self.thickness)  # fmt: skip

    def extruded(self, thickness: float | None = None, z0: float | None = None):
        """A layer of tetrahedra of thickness ``thickness`` (default :attr:`thickness`) from
        ``z0`` (default :attr:`z` minus half of it): each prism over a triangle cut into
        three tetrahedra along the diagonals that start at the least node of each of its
        sides (the cuts of neighbouring prisms then match), a
        :class:`~morphogen.mesh.tetra.TetMesh` of the box of that thickness, periodic as
        this mesh. With uniform fields across the layer (plane strain, periodic along z),
        its linear finite elements are those of this mesh."""
        from morphogen.mesh.tetra import TetMesh

        lin = self.linear()
        t = float(self.thickness if thickness is None else thickness)
        base = float(self.z - 0.5 * t if z0 is None else z0)
        n = lin.n_nodes
        nodes = np.vstack([np.column_stack([lin.nodes, np.full(n, base)]),
                           np.column_stack([lin.nodes, np.full(n, base + t)])])  # fmt: skip
        # the prism (a, b, c) below, (a', b', c') above (a < b < c): cut from the least node
        a, b, c = np.sort(lin.elements, axis=1).T
        cuts = [(a, b, c, c + n), (a, b, c + n, b + n), (a, b + n, c + n, a + n)]
        tets = np.stack([np.column_stack(t) for t in cuts], axis=1).reshape(-1, 4)
        lower = np.array([*lin.lower, base])
        upper = np.array([*lin.upper, base + t])
        mesh = TetMesh(nodes, tets, np.repeat(lin.phase, 3), lower, upper,
                       periodic=lin.periodic, metadata={"extruded_from": "triangles"},
                       cell=None if lin.cell is None else np.repeat(lin.cell, 3),
                       orientations=lin.orientations)  # fmt: skip
        vol = mesh.volumes()
        if np.any(vol <= 0):
            flip = vol < 0
            mesh.elements[flip] = mesh.elements[flip][:, [0, 2, 1, 3]]
        return mesh

    def summary(self) -> dict:
        q = self.quality()
        return {
            "nodes": self.n_nodes,
            "elements": self.n_elements,
            "order": self.order,
            "min_quality": float(q.min()) if q.size else float("nan"),
            "min_angle": float(self.min_angles().min()) if self.n_elements else float("nan"),
            "phase_fractions": self.phase_fractions(),
            "periodic": bool(self.periodic),
            "matching_sides": self.matching_sides() if self.periodic else False,
        }


def _corner_mesh(nodes: np.ndarray, corners: np.ndarray):
    used, inverse = np.unique(corners, return_inverse=True)
    return nodes[used], inverse.reshape(corners.shape)


# ----------------------------------------------------------------------------------------
# The background triangulation and the level sets of the plane


def background_triangulation(shape, lower, upper):
    """The triangulation of a grid of ``shape`` squares of the rectangle: each square cut by
    a diagonal alternating with its parity (the triangulation symmetric about the lines of
    the grid), and the edges of its sides with their references (:data:`SIDE_REFS`)."""
    nx, ny = (int(k) for k in shape)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    x = np.linspace(lower[0], upper[0], nx + 1)
    y = np.linspace(lower[1], upper[1], ny + 1)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    i, j = i.ravel(), j.ravel()
    a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
    even = (i + j) % 2 == 0
    tri = np.concatenate([
        np.column_stack([a, b, c])[even], np.column_stack([a, c, d])[even],
        np.column_stack([a, b, d])[~even], np.column_stack([b, c, d])[~even],
    ])  # fmt: skip
    edges, refs = [], []
    for (axis, side), ref in SIDE_REFS.items():
        line = idx[0 if side == 0 else nx, :] if axis == 0 else idx[:, 0 if side == 0 else ny]
        edges.append(np.column_stack([line[:-1], line[1:]]))
        refs.append(np.full(len(line) - 1, ref))
    return nodes, tri, np.concatenate(edges), np.concatenate(refs)


def plane_level_set(micro, spacing: float, z: float, phase: int | None = None):
    """The values of the level-set function of the phase 1 (or of the implicit function of
    ``phase``) at the nodes of a grid of the rectangle of the box in the plane ``z``:
    ``(nx + 1, ny + 1)`` values and the shape ``(nx, ny)`` of the grid."""
    domain: Domain = micro.domain
    lengths = np.asarray(domain.lengths[:2], dtype=float)
    n = np.maximum(1, np.ceil(lengths / spacing - 1e-9)).astype(int)
    h = lengths / n
    thin = 1e-6 * float(lengths.max())
    origin = (*(domain.lower[:2] - h / 2), z - thin / 2)
    sampling = Domain(size=(*(lengths + h), thin), origin=origin, periodic=False)
    grid = Grid(sampling, (int(n[0]) + 1, int(n[1]) + 1, 1))
    try:
        field = (
            micro.level_set(grid=grid) if phase is None else micro.phase_level_set(phase, grid=grid)
        )
    except NotImplementedError:
        raise ValueError(f"{type(micro).__name__}: no level-set function, so no mesh of its "
                         "section") from None  # fmt: skip
    return np.asarray(field.data, dtype=float)[:, :, 0], (int(n[0]), int(n[1]))


def _snap(values: np.ndarray, fraction: float) -> np.ndarray:
    """Values closer to zero than ``fraction`` of their variation along the grid moved away
    from it."""
    grad = np.zeros_like(values)
    for axis in (0, 1):
        d = np.abs(np.diff(values, axis=axis))
        pad = [(0, 0), (0, 0)]
        pad[axis] = (1, 0)
        grad = np.maximum(grad, np.pad(d, pad, mode="edge"))
        pad[axis] = (0, 1)
        grad = np.maximum(grad, np.pad(d, pad, mode="edge"))
    limit = fraction * grad
    close = np.abs(values) < limit
    return np.where(close, np.where(values < 0, -limit, limit), values)


def _matched(values: np.ndarray, periodic: bool) -> np.ndarray:
    if periodic:
        values = values.copy()
        values[-1, :] = values[0, :]
        values[:, -1] = values[:, 0]
    return values


def _linear_values(values: np.ndarray, lower, upper, points: np.ndarray) -> np.ndarray:
    """The values at ``points`` of the function linear in the triangles of
    :func:`background_triangulation` that takes ``values`` at its nodes."""
    n = np.array(values.shape) - 1
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    u = (points - lower) / (upper - lower) * n
    cell = np.clip(np.floor(u).astype(np.int64), 0, n - 1)
    s, t = np.clip(u - cell, 0.0, 1.0).T
    i, j = cell.T
    a, b = values[i, j], values[i + 1, j]
    c, d = values[i + 1, j + 1], values[i, j + 1]
    even = (i + j) % 2 == 0
    # even squares: (a, b, c) where s >= t, (a, c, d) else; odd: (a, b, d) where s + t <= 1
    out_even = np.where(s >= t, a + s * (b - a) + t * (c - b), a + t * (d - a) + s * (c - d))
    out_odd = np.where(s + t <= 1.0, a + s * (b - a) + t * (d - a),
                       c + (1.0 - s) * (d - c) + (1.0 - t) * (b - c))  # fmt: skip
    return np.where(even, out_even, out_odd)


def _mmg_args(size, min_size, hausdorff, gradation) -> list[str]:
    return ["-hmax", f"{size:.10g}",
            "-hmin", f"{(min_size if min_size is not None else size / 10):.10g}",
            "-hausd", f"{(hausdorff if hausdorff is not None else size / 20):.10g}",
            "-hgrad", f"{float(gradation):.10g}", "-v", "-1"]  # fmt: skip


def _run(exe, mesh_in, sol_in, mesh_out, args, timeout) -> dict:
    from morphogen.mesh.conforming import _run_mmg

    return _run_mmg(exe, mesh_in, sol_in, mesh_out, args, timeout)


def section_mesh(micro, *, size: float, min_size: float | None = None,
                 hausdorff: float | None = None, gradation: float = 2.0,
                 background: float | None = None, z: float | None = None,
                 phases="all", order: int = 1, periodic: bool | None = None,
                 executable: str | None = None, workdir=None,
                 timeout: float | None = None) -> TriMesh:  # fmt: skip
    """The conforming triangular mesh of the section of a structure in the plane ``z``
    (default: the middle of its box) from the implicit functions of its phases (see the
    module): ``size`` the largest edge, ``hausdorff`` (default ``size / 20``) the distance
    within which the edges follow the interfaces, ``background`` the spacing of the grid of
    the sampled functions (default ``size / 2``); ``phases``: ``"all"``, ``"solid"`` (phase
    1), or the phases to keep; ``order`` 1 or 2 (the middles of the straight edges)."""
    from morphogen.mesh.conforming import mmg_executable

    domain: Domain = micro.domain
    size = float(size)
    if not size > 0:
        raise ValueError("size: must be positive")
    if order not in (1, 2):
        raise ValueError("order: 1 or 2")
    z = float(domain.lower[2] + 0.5 * domain.lengths[2]) if z is None else float(z)
    periodic = bool(domain.periodic) if periodic is None else bool(periodic)
    if periodic and not domain.periodic:
        raise ValueError("matching periodic sides need a periodic structure")
    exe = mmg_executable(executable, "mmg2d")
    spacing = float(background) if background is not None else 0.5 * size
    present = (list(micro.implicit_phases()) if callable(getattr(micro, "implicit_phases", None))
               else [0, 1])  # fmt: skip
    several = present != [0, 1]
    args = _mmg_args(size, min_size, hausdorff, gradation)
    lower, upper = domain.lower[:2], domain.upper[:2]
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        if several:
            sampled = {}
            for k in present:
                sampled[k], shape = plane_level_set(micro, spacing, z, k)
            fields = []
            for k in present[:-1]:
                others = np.minimum.reduce([sampled[q] for q in present if q != k])
                fields.append((k, _matched(_snap(sampled[k] - others, SNAP), periodic)))
            rest = present[-1]
        else:
            values, shape = plane_level_set(micro, spacing, z)
            fields = [(1, _matched(_snap(values, SNAP), periodic))]
            rest = 0
        nodes, tri, edges, edge_refs = background_triangulation(shape, lower, upper)
        mesh = {"Vertices": nodes, "Triangles": tri,
                "Triangles_refs": np.full(len(tri), REF_PHASE + rest),
                "Edges": edges, "Edges_refs": edge_refs}  # fmt: skip
        done: list[int] = []
        for step, (phase, values) in enumerate(fields):
            name = folder / f"cut{step}.mesh"
            medit.write_mesh(name, mesh["Vertices"], None, mesh["Triangles"],
                             edges=mesh["Edges"], tri_refs=mesh["Triangles_refs"],
                             edge_refs=mesh["Edges_refs"])  # fmt: skip
            at = values.ravel() if step == 0 else _linear_values(values, lower, upper,
                                                                 mesh["Vertices"])  # fmt: skip
            medit.write_sol(folder / f"cut{step}.sol", at, dimension=2)
            lines = ["LSReferences", str(1 + len(done)),
                     f"{REF_PHASE + rest} {REF_PHASE + phase} {REF_PHASE + rest}"]  # fmt: skip
            lines += [f"{REF_PHASE + k} nosplit" for k in done]
            (folder / f"cut{step}.mmg2d").write_text("\n".join(lines) + "\n", encoding="ascii")
            cut = _run(exe, name, folder / f"cut{step}.sol", folder / f"cut{step + 1}_in.mesh",
                       ["-ls", "0", "-nosurf", "-noinsert", "-noswap", "-nomove", "-v", "-1"],
                       timeout)  # fmt: skip
            mesh = {k: cut[k] for k in ("Vertices", "Triangles", "Triangles_refs")}
            mesh["Edges"] = cut.get("Edges", np.zeros((0, 2), dtype=np.int64))
            mesh["Edges_refs"] = cut.get("Edges_refs", np.zeros(0, dtype=np.int64))
            done.append(int(phase))
        sides = np.isin(mesh["Edges_refs"], list(SIDE_REFS.values()))
        keep = {}
        if periodic:  # the cut sides match: kept as they are
            corners = _side_vertices(mesh["Vertices"], lower, upper)
            keep = {"required_edges": np.flatnonzero(sides), "required_vertices": corners}
        medit.write_mesh(folder / "cut.mesh", mesh["Vertices"], None, mesh["Triangles"],
                         edges=mesh["Edges"], tri_refs=mesh["Triangles_refs"],
                         edge_refs=mesh["Edges_refs"], **keep)  # fmt: skip
        out = _run(exe, folder / "cut.mesh", None, folder / "out.mesh", args, timeout)
    refs = out["Triangles_refs"]
    phase_of = refs - REF_PHASE
    if np.any(phase_of < 0):  # pragma: no cover - MMG kept the references
        raise RuntimeError("section mesh: unexpected references of the triangles")
    mesh = TriMesh(out["Vertices"], out["Triangles"], phase_of, lower, upper, periodic,
                   {"mesher": "section (mmg2d level sets)", "size": size, "z": z},
                   z=z, thickness=float(domain.lengths[2]))  # fmt: skip
    mesh = _kept(mesh, phases)
    return mesh.quadratic() if order == 2 else mesh


def _side_vertices(vertices: np.ndarray, lower, upper) -> np.ndarray:
    tol = 1e-9 * float(np.max(np.asarray(upper) - np.asarray(lower)))
    on = np.zeros(len(vertices), dtype=bool)
    for axis in (0, 1):
        on |= np.abs(vertices[:, axis] - lower[axis]) <= tol
        on |= np.abs(vertices[:, axis] - upper[axis]) <= tol
    return np.flatnonzero(on)


def _kept(mesh: TriMesh, phases) -> TriMesh:
    if isinstance(phases, str):
        if phases == "all":
            return mesh
        if phases == "solid":
            return mesh.subset([1])
        if phases == "matrix":
            return mesh.subset([0])
        raise ValueError("phases: 'all', 'solid', 'matrix' or the phases to keep")
    phases = [int(k) for k in phases]
    if not phases:
        raise ValueError("phases: at least one phase to keep")
    return mesh.subset(phases)


# ----------------------------------------------------------------------------------------
# The exact grains of planar tessellations


def planar_tessellation(micro) -> bool:
    """Whether a structure is a tessellation whose germs lie in one plane (the columnar
    grains of seeds of the plane, or any tessellation of coplanar germs)."""
    from morphogen.models.tessellation import Tessellation

    if not isinstance(micro, Tessellation) or not micro.n_cells:
        return False
    if (getattr(micro, "metadata", None) or {}).get("dimension") == 2:
        return True
    z = np.asarray(micro.germs)[:, 2]
    return bool(np.ptp(z) <= 1e-12 * float(np.max(micro.domain.lengths)))


def _clip(polygon: np.ndarray, normal: np.ndarray, bound: float) -> np.ndarray:
    """The part of a convex polygon (counterclockwise vertices) where ``normal . x <=
    bound`` (Sutherland and Hodgman)."""
    if not len(polygon):
        return polygon
    side = polygon @ normal - bound
    out = []
    for k in range(len(polygon)):
        p, q = polygon[k], polygon[(k + 1) % len(polygon)]
        sp, sq = side[k], side[(k + 1) % len(polygon)]
        if sp <= 0:
            out.append(p)
        if (sp < 0 < sq) or (sq < 0 < sp):
            out.append(p + sp / (sp - sq) * (q - p))
    return np.asarray(out).reshape(-1, 2)


def _rectangle(lower, upper) -> np.ndarray:
    return np.array([[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]],
                     [lower[0], upper[1]]], dtype=float)  # fmt: skip


def planar_pieces(micro) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """The power diagram in the plane of a tessellation of coplanar germs, cut by its box:
    the vertices of the pieces (merged), each piece as a loop of them (counterclockwise),
    and the cell of each piece. A cell is the polygon of the points of least power distance
    ``|x - x_k|² - w_k`` to its germ, clipped by the half planes of its neighbours (those
    within the reach of its farthest vertex, the periodic images of the germs among them);
    in a periodic box, it and its translates are cut by the box (the pieces of a cell on
    opposite sides)."""
    domain: Domain = micro.domain
    lower, upper = np.asarray(domain.lower[:2], dtype=float), np.asarray(domain.upper[:2])
    lengths = upper - lower
    germs = np.asarray(micro.germs, dtype=float)[:, :2]
    n = len(germs)
    weights = np.zeros(n) if micro.weights is None else np.asarray(micro.weights, dtype=float)
    periodic = bool(domain.periodic)
    shifts = (np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float)
              if periodic else np.zeros((1, 2)))  # fmt: skip
    every = (germs[None, :, :] + shifts[:, None, :] * lengths).reshape(-1, 2)
    every_w = np.tile(weights, len(shifts))
    tree = cKDTree(every)
    w_max = float(weights.max()) if n else 0.0
    big = (
        _rectangle(lower - lengths, upper + lengths)
        if periodic
        else _rectangle(
            np.minimum(lower, germs.min(axis=0)) - lengths,
            np.maximum(upper, germs.max(axis=0)) + lengths,
        )
    )
    pieces, owner = [], []
    for i in range(n):
        c, w = germs[i], weights[i]
        k = min(len(every), 24)
        while True:
            _, near = tree.query(c, k=k)
            poly = big.copy()
            for j in np.atleast_1d(near):
                d = every[j] - c
                if not np.any(np.abs(d) > 0):
                    continue
                bound = 0.5 * (every[j] @ every[j] - c @ c - every_w[j] + w)
                poly = _clip(poly, d, bound)
                if not len(poly):
                    break
            if not len(poly):
                break
            reach = float(np.max(np.linalg.norm(poly - c, axis=1)))
            limit = reach + np.sqrt(max(reach * reach - w + w_max, 0.0))
            far = float(np.linalg.norm(every[np.atleast_1d(near)[-1]] - c))
            if far >= limit or k >= len(every):
                break
            k = min(len(every), 2 * k)
        if len(poly) < 3:
            continue
        for shift in shifts if periodic else np.zeros((1, 2)):
            piece = poly + shift * lengths
            for axis in (0, 1):
                e = np.zeros(2)
                e[axis] = 1.0
                piece = _clip(piece, e, upper[axis])
                piece = _clip(piece, -e, -lower[axis])
            if len(piece) >= 3 and abs(_shoelace(piece)) > 1e-12 * float(np.prod(lengths)):
                pieces.append(piece)
                owner.append(i)
    # the vertices merged within a tolerance, each loop without repeats
    stack = np.vstack(pieces)
    tol = 1e-9 * float(lengths.max())
    pairs = cKDTree(stack).query_pairs(tol, output_type="ndarray")
    parent = np.arange(len(stack))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(a) for a in range(len(stack))])
    unique, number = np.unique(roots, return_inverse=True)
    vertices = stack[unique]
    loops, start = [], 0
    for piece in pieces:
        ids = number[start : start + len(piece)]
        start += len(piece)
        keep = np.concatenate([[True], ids[1:] != ids[:-1]])
        ids = ids[keep]
        if len(ids) > 1 and ids[0] == ids[-1]:
            ids = ids[:-1]
        loops.append(ids)
    return vertices, loops, np.asarray(owner, dtype=np.int64)


def _shoelace(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The distances of points to segments ``ab`` (row by row)."""
    d = b - a
    t = np.einsum("ij,ij->i", points - a, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-300)
    return np.linalg.norm(a + np.clip(t, 0.0, 1.0)[:, None] * d - points, axis=1)


class _SizeField:
    """The size of the elements along the edges of the polygons: the least of the size and
    of ``h_k + GRADE |x - p_k|`` over points ``p_k`` along the edges, whose sizes ``h_k``
    are their distances to the nearest edge not adjacent to theirs (thin regions, close
    corners), at most the lengths of the edges of the vertices among them; in a periodic
    box, between the periodic images of the points and of the edges (the vertices of the
    sides identified with their translates)."""

    def __init__(self, xy, segments, ids, size, lower, lengths, periodic, spacing=None):
        self.size = float(size)
        self.lower = np.asarray(lower, dtype=float)
        self.lengths = np.asarray(lengths, dtype=float)
        self.periodic = bool(periodic)
        seg = np.asarray(segments, dtype=np.int64).reshape(-1, 2)
        a, b = xy[seg[:, 0]], xy[seg[:, 1]]
        ell = np.linalg.norm(b - a, axis=1)
        spacing = self.size / 4.0 if spacing is None else float(spacing)
        counts = np.maximum(np.ceil(ell / spacing - 1e-9).astype(np.int64), 1)
        owner = np.repeat(np.arange(len(seg)), counts + 1)
        t = np.concatenate([np.arange(c + 1) / c for c in counts]) if len(seg) else np.zeros(0)
        points = a[owner] + t[:, None] * (b - a)[owner]
        h = np.full(len(points), self.size)
        ends = (t == 0.0) | (t == 1.0)
        h[ends] = np.minimum(h[ends], ell[owner[ends]])
        tree = self._tree(points)
        pairs = tree.query_pairs(self.size + spacing, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            si, sj = owner[i], owner[j]
            ca, cb = ids[seg[si]], ids[seg[sj]]
            adjacent = ((ca[:, 0] == cb[:, 0]) | (ca[:, 0] == cb[:, 1])
                        | (ca[:, 1] == cb[:, 0]) | (ca[:, 1] == cb[:, 1]))  # fmt: skip
            i, j, si, sj = i[~adjacent], j[~adjacent], si[~adjacent], sj[~adjacent]
            shift = self._image(points[i] - points[j])  # the image of j nearest to i
            np.minimum.at(h, i, _point_segment(points[i], a[sj] + shift, b[sj] + shift))
            np.minimum.at(h, j, _point_segment(points[j], a[si] - shift, b[si] - shift))
        self.points, self.h = points, np.maximum(h, 1e-12 * float(self.lengths.max()))
        self.tree = self._tree(points)
        self.reach = self.size / GRADE

    def _image(self, d: np.ndarray) -> np.ndarray:
        if not self.periodic:
            return np.zeros_like(d)
        return self.lengths * np.round(d / self.lengths)

    def _tree(self, points: np.ndarray) -> cKDTree:
        if not self.periodic:
            return cKDTree(points)
        wrapped = np.mod(points - self.lower, self.lengths)
        wrapped[wrapped >= self.lengths] = 0.0
        return cKDTree(wrapped, boxsize=self.lengths)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self.periodic:
            x = np.mod(x - self.lower, self.lengths)
            x[x >= self.lengths] = 0.0
        k = min(len(self.points), 48)
        dist, idx = self.tree.query(x, k=k, distance_upper_bound=self.reach)
        dist, idx = dist.reshape(len(x), k), idx.reshape(len(x), k)
        found = idx < len(self.points)
        value = np.where(found, self.h[np.minimum(idx, len(self.points) - 1)] + GRADE * dist,
                         np.inf)  # fmt: skip
        return np.minimum(self.size, value.min(axis=1))


def _graded(starts: np.ndarray, ends: np.ndarray, field: _SizeField) -> list[np.ndarray]:
    """The points strictly inside each segment from ``starts`` to ``ends``, at the sizes of
    the field (marched from the start, the steps then scaled to end on the end)."""
    vector = ends - starts
    length = np.linalg.norm(vector, axis=1)
    unit = vector / np.maximum(length, 1e-300)[:, None]
    steps = [[] for _ in range(len(starts))]
    t = np.zeros(len(starts))
    live = np.flatnonzero(length > 0)
    while len(live):
        h = field(starts[live] + t[live, None] * unit[live])
        for k, hk in zip(live, h, strict=True):
            steps[k].append(hk)
        t[live] += h
        live = live[t[live] < length[live]]
    out = []
    for k in range(len(starts)):
        s = np.asarray(steps[k])
        if len(s) <= 1:
            out.append(np.zeros((0, 2)))
            continue
        before = float(s[:-1].sum())  # the steps before the last, which passes the end
        n = max(1, round(len(s) - 1 + (length[k] - before) / s[-1]))
        cum = np.cumsum(s[:n])
        cum = cum[: n - 1] * length[k] / cum[n - 1] if n > 1 else np.zeros(0)
        out.append(starts[k] + cum[:, None] * unit[k])
    return out


def tessellation_mesh(micro, *, size: float, min_size: float | None = None,
                      hausdorff: float | None = None, gradation: float = 2.0,
                      phases="all", order: int = 1, periodic: bool | None = None,
                      executable: str | None = None, workdir=None,
                      timeout: float | None = None) -> TriMesh:  # fmt: skip
    """The conforming triangular mesh of the grains of a planar tessellation (see the
    module): one region per grain, the phase and the orientation of its grain to each
    element (``cell``: the grain). The polygons are exact: their edges between grains and
    on the sides are subdivided at the sizes of a field graded from their small features
    (:class:`_SizeField`: short edges, thin grains, close corners) up to ``size``, and
    ``mmg2d`` remeshes the grains within them without moving them (``-nosurf``)."""
    from morphogen.mesh.conforming import mmg_executable

    if not planar_tessellation(micro):
        raise ValueError("tessellation mesh: the germs of a tessellation in one plane")
    domain: Domain = micro.domain
    size = float(size)
    if not size > 0:
        raise ValueError("size: must be positive")
    if order not in (1, 2):
        raise ValueError("order: 1 or 2")
    periodic = bool(domain.periodic) if periodic is None else bool(periodic)
    if periodic and not domain.periodic:
        raise ValueError("matching periodic sides need a periodic tessellation")
    exe = mmg_executable(executable, "mmg2d")
    grains = np.asarray(micro.cell_grains(), dtype=np.int64)
    n_grains = int(grains.max()) + 1 if len(grains) else 0
    grain_phase = np.zeros(n_grains, dtype=np.int64)
    grain_phase[grains] = micro.cell_phases
    lower = np.asarray(domain.lower[:2], dtype=float)
    upper = np.asarray(domain.upper[:2], dtype=float)
    lengths = upper - lower
    tol = 1e-9 * float(domain.lengths.max())
    # the polygons: the pieces of the cells of the power diagram in the box, their grains
    vertices, loops, piece_cell = planar_pieces(micro)
    owner = grains[piece_cell]
    xy = vertices.copy()
    for axis in (0, 1):  # the vertices of the sides exactly on them
        for value in (lower[axis], upper[axis]):
            xy[np.abs(xy[:, axis] - value) <= tol, axis] = value
    ids = np.arange(len(xy))  # each vertex, or the one of the lower sides it translates
    twins = [np.arange(len(xy)), np.arange(len(xy))]  # along x, along y
    if periodic:
        for axis in (0, 1):
            on_hi = np.flatnonzero(xy[:, axis] == upper[axis])
            on_lo = np.flatnonzero(xy[:, axis] == lower[axis])
            if len(on_hi) != len(on_lo):
                raise RuntimeError("tessellation mesh: the sides of the cells do not match")
            other = 1 - axis
            dist, j = cKDTree(xy[on_lo, other][:, None]).query(xy[on_hi, other][:, None])
            if len(dist) and dist.max() > 1e3 * tol:
                raise RuntimeError("tessellation mesh: the sides of the cells do not match")
            xy[on_hi, other] = xy[on_lo[j], other]
            twins[axis][on_hi] = on_lo[j]
        ids = twins[0][twins[1][twins[0]]]  # the corners of the box: one vertex
    # the edges of the pieces: between grains, on the sides, or inside a grain
    pieces_of: dict[tuple[int, int], list[int]] = {}
    for p, loop in enumerate(loops):
        for a, b in zip(loop, np.roll(loop, -1), strict=True):
            pieces_of.setdefault((min(int(a), int(b)), max(int(a), int(b))), []).append(p)

    def side_of(key):
        pa, pb = xy[key[0]], xy[key[1]]
        for (axis, side), ref in SIDE_REFS.items():
            value = lower[axis] if side == 0 else upper[axis]
            if pa[axis] == value and pb[axis] == value:
                return ref
        return 0

    edge_ref: dict[tuple[int, int], int] = {}
    for key, ps in pieces_of.items():
        ref = side_of(key)
        if ref:
            if len(ps) != 1:
                raise RuntimeError("tessellation mesh: an edge of a side between two pieces")
            edge_ref[key] = ref
        elif len(ps) != 2:
            raise RuntimeError("tessellation mesh: an edge inside the box not between two "
                               "pieces")  # fmt: skip
        elif owner[ps[0]] != owner[ps[1]]:
            edge_ref[key] = 100
    kept = list(edge_ref)
    # the points along the kept edges, graded from their small features; the upper sides of
    # a periodic box the translates of the lower ones
    field = _SizeField(xy, kept, ids, size, lower, lengths, periodic)
    marched = [k for k in kept if not (periodic and edge_ref[k] in (2, 4))]
    starts = xy[[k[0] for k in marched]].reshape(-1, 2)
    ends = xy[[k[1] for k in marched]].reshape(-1, 2)
    inner = dict(zip(marched, _graded(starts, ends, field), strict=True))
    if periodic:
        for key in kept:
            ref = edge_ref[key]
            if ref not in (2, 4):
                continue
            axis = 0 if ref == 2 else 1
            low = (int(twins[axis][key[0]]), int(twins[axis][key[1]]))
            twin = (min(low), max(low))
            pts = inner[twin] if low[0] == twin[0] else inner[twin][::-1]
            pts = pts.copy()
            pts[:, axis] = upper[axis]
            inner[key] = pts
    count = len(xy)
    between: dict[tuple[int, int], list[int]] = {}
    extra = []
    for key in kept:
        between[key] = list(range(count, count + len(inner[key])))
        extra.append(inner[key])
        count += len(inner[key])
    xy = np.vstack([xy, *extra])
    # the triangles: each polygon fanned from its centroid
    tris, refs, centres = [], [], []
    for loop, g in zip(loops, owner, strict=True):
        ring = []
        for a, b in zip(loop, np.roll(loop, -1), strict=True):
            mids = between.get((min(a, b), max(a, b)), [])
            ring.append(int(a))
            ring += mids if a < b else mids[::-1]
        c = len(xy) + len(centres)
        centres.append(xy[loop].mean(axis=0))
        ring = np.asarray(ring)
        tris.append(np.column_stack([ring, np.roll(ring, -1), np.full(len(ring), c)]))
        refs.append(np.full(len(ring), g + 1))
    xy = np.vstack([xy, np.asarray(centres).reshape(-1, 2)])
    tris = np.concatenate(tris)
    refs = np.concatenate(refs)
    flip = _signed_areas(xy, tris) < 0  # counterclockwise
    tris[flip] = tris[flip][:, [0, 2, 1]]
    # the edges given to MMG, kept as they are: between grains and on the sides
    edges, edge_refs = [], []
    for key in kept:
        chain = [key[0], *between[key], key[1]]
        edges += list(itertools.pairwise(chain))
        edge_refs += [edge_ref[key]] * (len(chain) - 1)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edge_refs = np.asarray(edge_refs, dtype=np.int64)
    shortest = float(np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1).min())
    hmin = 0.5 * shortest if min_size is None else min(float(min_size), 0.5 * shortest)
    corners = np.unique(edges)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        medit.write_mesh(folder / "grains.mesh", xy, None, tris, edges=edges, tri_refs=refs,
                         edge_refs=edge_refs, required_vertices=corners,
                         required_edges=np.arange(len(edges)))  # fmt: skip
        args = [*_mmg_args(size, hmin, hausdorff, gradation), "-nr", "-nosurf"]
        out = _run(exe, folder / "grains.mesh", None, folder / "out.mesh", args, timeout)
    cell = out["Triangles_refs"] - 1
    mesh = TriMesh(out["Vertices"], out["Triangles"], grain_phase[cell], lower, upper,
                   periodic, {"mesher": "tessellation (exact polygons, mmg2d)", "size": size,
                              "n_grains": n_grains},
                   cell=cell, orientations=micro.orientations,
                   z=float(domain.lower[2] + 0.5 * domain.lengths[2]),
                   thickness=float(domain.lengths[2]))  # fmt: skip
    mesh = _kept(mesh, phases)
    return mesh.quadratic() if order == 2 else mesh


def plane_mesh(micro, *, size: float, **options) -> TriMesh:
    """The triangular mesh of a planar structure: the exact grains of a planar tessellation
    (:func:`tessellation_mesh`), the section of any other structure in its plane
    (:func:`section_mesh`)."""
    if planar_tessellation(micro):
        options.pop("background", None)
        options.pop("z", None)
        return tessellation_mesh(micro, size=size, **options)
    return section_mesh(micro, size=size, **options)


__all__ = ["REF_PHASE", "SIDE_REFS", "TRI6_EDGES", "TriMesh", "background_triangulation",
           "planar_pieces", "planar_tessellation", "plane_level_set", "plane_mesh",
           "section_mesh", "tessellation_mesh"]  # fmt: skip
