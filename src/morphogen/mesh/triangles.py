"""Triangular meshes of planar structures, in the plane ``z`` of their box.

A structure of the plane — the columnar grains of seeds of the plane (``seeded_laguerre``
with ``dimension: 2``) or of any tessellation whose germs lie in one plane, the section of a
structure of space at a height ``z``, an orientation map — is meshed in that plane by
triangles (:class:`TriMesh`: three- or six-node triangles, a phase each, a grain each for
polycrystals with their crystal orientations; the six-node ones isoparametric, curved
along the interfaces of sections), conforming to its interfaces and, for a
periodic structure, with matching nodes on opposite sides, so that periodic conditions are
exact (:func:`plane_mesh`):

* the grains of a tessellation whose germs lie in one plane, and the sections of any
  tessellation of space, are exact polygons: the pieces of the power diagram of the plane
  (:func:`planar_pieces`; for a section, the weights less the squares of the distances of
  the germs to the plane, :func:`section_weights`), the cells of the spheres of a seed
  gathered into its grain. Their features smaller than ``min_feature`` are removed (short
  edges collapsed, the vertices near the sides moved onto them or onto their nearest
  vertices), their edges between grains and on the sides subdivided at sizes graded from
  what small features remain — the upper
  sides of a periodic box the translates of the lower ones —, and ``mmg2d`` remeshes the
  grains within them without moving them (:func:`tessellation_mesh`);
* the discs of the sections of straight fibres along ``z`` and of spheres, coated or hollow
  (:func:`disc_layers`), by their circles, discretized exactly at sizes graded from the gaps
  between them — those of plain discs that cross each other cut where they cross, their
  arcs inside other discs left out: the boundary of their union —, ``mmg2d`` meshing the
  regions within them (:func:`disc_mesh`);
* a map of grains (a lattice field of one layer with the label of a grain per cell: the
  orientation maps of EBSD) by its pixels, two triangles each (:func:`lattice_mesh`);
* any other structure (:func:`section_mesh`): the implicit functions of its phases are
  sampled at the nodes of a background triangulation of the rectangle (a grid of squares
  cut by alternating diagonals) in the plane ``z``, and cut one after another by the
  level sets of ``mmg2d`` (the multi-material level sets of MMG: a phase is where its
  function is the least, the phase of the largest index the rest), without remeshing;
  ``mmg2d`` then remeshes the triangulation within the Hausdorff distance of the
  interfaces, keeping the edges of the sides of a periodic structure, which the cut left
  matching (the same values on opposite sides); six-node elements have their nodes of
  the interfaces on the curves of the section (:func:`~morphogen.mesh.curved.
  curved_triangles`).

:meth:`TriMesh.extruded` makes layers of tetrahedra of it (prisms cut into three), whose
periodic finite elements in space are those of the plane (:meth:`TriMesh.layer`: the
loading paths of :mod:`morphogen.homogenization.plane`).
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
#: The six nodes of the reference triangle (barycentric coordinates: the corners, then the
#: middles of the edges 01, 12, 20), and its three integration points (exact to the second
#: degree, that of the area element of a curved six-node triangle)
TRI6_NODES = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                       [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])  # fmt: skip
TRI_GAUSS = np.array([[4.0, 1.0, 1.0], [1.0, 4.0, 1.0], [1.0, 1.0, 4.0]]) / 6.0


def _signed_areas(nodes: np.ndarray, corners: np.ndarray) -> np.ndarray:
    a = nodes[corners[:, 1]] - nodes[corners[:, 0]]
    b = nodes[corners[:, 2]] - nodes[corners[:, 0]]
    return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def corner_gradients(nodes: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """The gradients of the barycentric coordinates of the straight triangles ``corners``
    (``(m, 3, 2)``)."""
    p = nodes[corners[:, :3]]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    det = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])[:, None]
    g1 = np.column_stack([e2[:, 1], -e2[:, 0]]) / det
    g2 = np.column_stack([-e1[:, 1], e1[:, 0]]) / det
    return np.stack([-g1 - g2, g1, g2], axis=1)


def edge_deviations(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """The departures (``(m, 3, 2)``) of the middle nodes of six-node triangles from the
    middles of their straight edges (zero where the elements are straight)."""
    ends = elements[:, np.array(TRI6_EDGES)]
    return nodes[elements[:, 3:]] - 0.5 * (nodes[ends[..., 0]] + nodes[ends[..., 1]])


def curved_maps(gradients: np.ndarray, deviations: np.ndarray, point) -> np.ndarray:
    """The gradients ``F = I + sum_k d_k (x) grad N_k`` (``(m, 2, 2)``) of the maps of the
    straight triangles onto their curved six-node elements at a point (barycentric
    coordinates): ``d_k`` the departures of their middle nodes, ``N_k`` the functions of
    these nodes on the straight triangles (``gradients``: those of the barycentric
    coordinates)."""
    lam = np.asarray(point, dtype=float)
    f = np.zeros((len(gradients), 2, 2))
    f[:, 0, 0] = f[:, 1, 1] = 1.0
    for k, (a, b) in enumerate(TRI6_EDGES):
        g = 4.0 * (lam[a] * gradients[:, b] + lam[b] * gradients[:, a])
        f += deviations[:, k, :, None] * g[:, None, :]
    return f


def jacobian_ratios(gradients: np.ndarray, deviations: np.ndarray, points) -> np.ndarray:
    """``det F`` (:func:`curved_maps`) at the points (rows of barycentric coordinates) of
    six-node triangles: the ratios of the area elements of the curved elements to those of
    their straight triangles (``(m, p)``)."""
    out = np.empty((len(gradients), len(points)))
    for q, point in enumerate(points):
        f = curved_maps(gradients, deviations, point)
        out[:, q] = f[:, 0, 0] * f[:, 1, 1] - f[:, 0, 1] * f[:, 1, 0]
    return out


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
        """The areas of the elements: those of their straight corners, of the curved
        six-node elements exactly (their area elements, :func:`jacobian_ratios`,
        integrated)."""
        a = _signed_areas(self.nodes, self.elements[:, :3])
        if self.order == 2 and self.n_elements:
            dev = self.deviations()
            bent = np.flatnonzero(np.abs(dev).max(axis=(1, 2)) > 0.0)
            if len(bent):
                grads = corner_gradients(self.nodes, self.elements[bent])
                a[bent] *= jacobian_ratios(grads, dev[bent], TRI_GAUSS).mean(axis=1)
        return a

    def quality(self) -> np.ndarray:
        """``4 √3 A / Σ ℓ²`` of each element (of its straight corners): 1 for an equilateral
        triangle, 0 for a flat one."""
        p = self.nodes[self.elements[:, :3]]
        lengths = sum(np.sum((p[:, (k + 1) % 3] - p[:, k]) ** 2, axis=1) for k in range(3))
        area = _signed_areas(self.nodes, self.elements[:, :3])
        return 4.0 * np.sqrt(3.0) * area / np.maximum(lengths, 1e-300)

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

    # -- curved edges --------------------------------------------------------------------

    def deviations(self) -> np.ndarray | None:
        """The departures (elements x 3 x 2) of the middle nodes of a six-node mesh from the
        middles of their straight edges (zero where the elements are straight); None for a
        three-node mesh."""
        if self.order == 1:
            return None
        return edge_deviations(self.nodes, self.elements)

    @property
    def curved(self) -> bool:
        """Whether some six-node elements have curved edges (middle nodes off the middles
        of their edges, beyond the round-off of coordinates read from a file):
        isoparametric elements, each edge the parabola through its three nodes."""
        if self.order == 1 or not self.n_elements:
            return False
        dev = np.abs(self.deviations()).max(axis=(1, 2))
        scale = np.sqrt(np.abs(_signed_areas(self.nodes, self.elements[:, :3])))
        return bool(np.any(dev > 1e-10 * scale))

    def min_jacobian(self) -> float:
        """The smallest ratio of the area element of the six-node elements to that of their
        straight triangles, over their six nodes and their three integration points (1 for
        a straight mesh; positive for valid elements)."""
        if not self.curved:
            return 1.0
        dev = self.deviations()
        bent = np.flatnonzero(np.abs(dev).max(axis=(1, 2)) > 0.0)
        grads = corner_gradients(self.nodes, self.elements[bent])
        return float(jacobian_ratios(grads, dev[bent], np.vstack([TRI6_NODES, TRI_GAUSS])).min())

    def straight(self) -> TriMesh:
        """The same six-node mesh with straight edges (its middle nodes at the middles of
        their edges); the mesh itself if it has no curved edge."""
        if not self.curved:
            return self
        nodes = self.nodes.copy()
        ends = self.elements[:, np.array(TRI6_EDGES)]
        nodes[self.elements[:, 3:]] = 0.5 * (self.nodes[ends[..., 0]] + self.nodes[ends[..., 1]])
        out = self._with_nodes(nodes)
        out.metadata.pop("curved", None)
        return out

    def _with_nodes(self, nodes: np.ndarray) -> TriMesh:
        """The same mesh with its nodes at ``nodes``."""
        return TriMesh(nodes, self.elements, self.phase, self.lower, self.upper, self.periodic,
                       dict(self.metadata), self.cell, self.orientations, self.z,
                       self.thickness)  # fmt: skip

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

    def extruded(self, thickness: float | None = None, z0: float | None = None,
                 layers: int = 1):  # fmt: skip
        """Layers of tetrahedra of the total thickness ``thickness`` (default
        :attr:`thickness`) from ``z0`` (default :attr:`z` minus half of it): each prism over a
        triangle cut into three tetrahedra along the diagonals that start at the least node of
        each of its sides (the cuts of neighbouring prisms then match), a
        :class:`~morphogen.mesh.tetra.TetMesh` of the box of that thickness, periodic as this
        mesh, its faces ``z`` matching. With uniform fields across the layers (plane strain,
        periodic along z), its linear finite elements are those of this mesh."""
        from morphogen.mesh.tetra import TetMesh

        lin = self.linear()
        t = float(self.thickness if thickness is None else thickness)
        base = float(self.z - 0.5 * t if z0 is None else z0)
        layers = int(layers)
        if layers < 1:
            raise ValueError("layers: one at least")
        n = lin.n_nodes
        heights = base + t * np.arange(layers + 1) / layers
        heights[-1] = base + t
        nodes = np.vstack([np.column_stack([lin.nodes, np.full(n, h)]) for h in heights])
        # the prism (a, b, c) below, (a', b', c') above (a < b < c): cut from the least node
        a, b, c = np.sort(lin.elements, axis=1).T
        cuts = [(a, b, c, c + n), (a, b, c + n, b + n), (a, b + n, c + n, a + n)]
        one = np.stack([np.column_stack(t) for t in cuts], axis=1).reshape(-1, 4)
        tets = np.concatenate([one + k * n for k in range(layers)])
        lower = np.array([*lin.lower, base])
        upper = np.array([*lin.upper, base + t])
        order = np.tile(np.repeat(np.arange(lin.n_elements), 3), layers)  # their triangles
        mesh = TetMesh(nodes, tets, lin.phase[order], lower, upper,
                       periodic=lin.periodic, metadata={"extruded_from": "triangles",
                                                        "layers": layers},
                       cell=None if lin.cell is None else lin.cell[order],
                       orientations=lin.orientations)  # fmt: skip
        vol = mesh.volumes()
        if np.any(vol <= 0):
            flip = vol < 0
            mesh.elements[flip] = mesh.elements[flip][:, [0, 2, 1, 3]]
        return mesh

    def layer(self):
        """The layer of tetrahedra in space whose periodic problems are those of this mesh
        (generalized plane strain): one layer of the thickness of its mean edge (the fields
        do not vary across it, so that its thickness does not matter; elements of good
        shape), of ten-node tetrahedra for six-node triangles (:meth:`extruded`)."""
        lin = self.linear()
        edges = lin.edges()
        mean = float(np.linalg.norm(lin.nodes[edges[:, 1]] - lin.nodes[edges[:, 0]], axis=1).mean())
        mesh = lin.extruded(thickness=mean, z0=self.z - 0.5 * mean)
        if self.order == 1:
            return mesh
        mesh = mesh.quadratic()
        if self.curved:  # the edges of the prisms over the curved edges, curved as they are
            mesh = self._curved_layer(mesh, lin.n_nodes)
        return mesh

    def _curved_layer(self, mesh, n: int):
        """The ten-node tetrahedra of the prisms over the triangles (:meth:`extruded`, the
        corners of each layer of nodes numbered as those of :meth:`linear`) whose edges over
        the curved edges of the triangles are curved as they are: the middle node of an
        edge between the corners ``i`` and ``j`` of a triangle, at whatever heights, at the
        place of the middle node of that edge in the plane (the faces of the prisms then
        the curved edges extruded, exactly)."""
        from morphogen.mesh.tetra import TET_EDGES

        _, inverse = np.unique(self.elements[:, :3], return_inverse=True)
        corners = inverse.reshape(-1, 3)
        pairs = np.concatenate([corners[:, [a, b]] for a, b in TRI6_EDGES])
        middles = np.concatenate([self.elements[:, 3 + k] for k in range(3)])
        pairs = np.sort(pairs, axis=1)
        codes, first = np.unique(pairs[:, 0] * n + pairs[:, 1], return_index=True)
        ends = mesh.elements[:, TET_EDGES] % n  # the corners of the triangles
        across = ends[..., 0] != ends[..., 1]  # not along the thickness
        lo, hi = np.minimum(ends[..., 0], ends[..., 1]), np.maximum(ends[..., 0], ends[..., 1])
        where = np.searchsorted(codes, (lo * n + hi)[across])
        nodes = mesh.nodes.copy()
        nodes[mesh.elements[:, 4:][across], :2] = self.nodes[middles[first[where]]]
        return mesh._derived(nodes, mesh.elements)

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
                 phases="all", order: int = 1, curved: bool | None = None,
                 periodic: bool | None = None, executable: str | None = None, workdir=None,
                 timeout: float | None = None) -> TriMesh:  # fmt: skip
    """The conforming triangular mesh of the section of a structure in the plane ``z``
    (default: the middle of its box) from the implicit functions of its phases (see the
    module): ``size`` the largest edge, ``hausdorff`` (default ``size / 20``) the distance
    within which the edges follow the interfaces, ``background`` the spacing of the grid of
    the sampled functions (default ``size / 2``); ``phases``: ``"all"``, ``"solid"`` (phase
    1), or the phases to keep; ``order`` 1 or 2. Six-node elements have curved edges along
    the interfaces, their nodes on the curves of the section
    (:func:`~morphogen.mesh.curved.curved_triangles`), by default; ``curved`` false keeps
    the straight edges of the mesher, their middle nodes at their middles."""
    from morphogen.mesh.conforming import mmg_executable

    domain: Domain = micro.domain
    size = float(size)
    if not size > 0:
        raise ValueError("size: must be positive")
    if order not in (1, 2):
        raise ValueError("order: 1 or 2")
    if curved and order == 1:
        raise ValueError("curved: the edges of six-node elements (order 2)")
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
    if order == 1:
        return mesh
    if curved is False:
        return mesh.quadratic()
    from morphogen.mesh.curved import curved_triangles

    return curved_triangles(mesh, micro)


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
    if side.max() <= 0:  # the half plane holds the whole polygon (most neighbours)
        return polygon
    if side.min() > 0:
        return polygon[:0]
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


def section_weights(micro, z: float) -> np.ndarray:
    """The weights of the power diagram of the plane ``z`` that cuts a tessellation in space:
    ``w_k - (z - z_k)²``, the distance to the plane of the nearest image of each germ along
    ``z`` in a periodic box — the power distance of a point of the plane to a germ is its
    distance in the plane less that weight, and the cells of the plane are the sections of
    the cells in space (the germs of the cells that the plane misses have empty cells)."""
    germs = np.asarray(micro.germs, dtype=float)
    weights = (np.zeros(len(germs)) if micro.weights is None
               else np.asarray(micro.weights, dtype=float))  # fmt: skip
    d = float(z) - germs[:, 2]
    if micro.domain.periodic:
        length = float(micro.domain.lengths[2])
        d = d - length * np.round(d / length)
    return weights - d * d


def planar_pieces(micro, z: float | None = None) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """The power diagram in the plane of a tessellation of coplanar germs, cut by its box:
    the vertices of the pieces (merged), each piece as a loop of them (counterclockwise),
    and the cell of each piece. A cell is the polygon of the points of least power distance
    ``|x - x_k|² - w_k`` to its germ, clipped by the half planes of its neighbours (those
    within the reach of its farthest vertex, the periodic images of the germs among them);
    in a periodic box, it and its translates are cut by the box (the pieces of a cell on
    opposite sides). With ``z``, the section of any tessellation in space by the plane at
    that height: the power diagram of the weights of :func:`section_weights`."""
    domain: Domain = micro.domain
    lower, upper = np.asarray(domain.lower[:2], dtype=float), np.asarray(domain.upper[:2])
    lengths = upper - lower
    germs = np.asarray(micro.germs, dtype=float)[:, :2]
    n = len(germs)
    if z is not None:
        weights = section_weights(micro, z)
    else:
        weights = (np.zeros(n) if micro.weights is None
                   else np.asarray(micro.weights, dtype=float))  # fmt: skip
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


def _find(parent: np.ndarray, a: int) -> int:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _merged(points: np.ndarray, tol: float) -> np.ndarray:
    """The index of each point among the distinct ones (those closer than ``tol`` merged)."""
    parent = np.arange(len(points))
    for a, b in cKDTree(points).query_pairs(tol, output_type="ndarray"):
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([_find(parent, a) for a in range(len(points))], dtype=np.int64)
    return np.unique(roots, return_inverse=True)[1]


def _kept_edges(loops, owner) -> dict[tuple[int, int], list[int]]:
    """The pieces of each edge of the loops (sorted pairs of vertices)."""
    pieces_of: dict[tuple[int, int], list[int]] = {}
    for p, loop in enumerate(loops):
        for a, b in zip(loop, np.roll(loop, -1), strict=True):
            pieces_of.setdefault((min(int(a), int(b)), max(int(a), int(b))), []).append(p)
    return pieces_of


def _regularized(xy, loops, owner, lower, upper, periodic: bool, delta: float, tol: float):
    """The polygons of the pieces without their features smaller than ``delta``, as the
    cells in space (:func:`morphogen.mesh.regularize.regularize`): the edges between grains
    or on the sides shorter than it collapsed (their ends merged, on the torus of a periodic
    box: the copies of a vertex on opposite sides alike; on the side, or the corner, of a
    vertex of one), the vertices nearer than it to a side moved onto the side (the thin
    caps of the cells that the box cuts), the pieces left without area removed, and the
    vertices of the sides of a periodic box given their translates on the opposite sides
    (inserted in the edges of those sides). Returns the vertices, the loops and their
    grains, or None when the polygons fold (the exact ones are then kept)."""
    lengths = upper - lower
    n = len(xy)
    side = np.zeros((n, 2), dtype=np.int8)  # -1 on the lower side of an axis, +1 the upper
    for axis in (0, 1):
        side[xy[:, axis] == lower[axis], axis] = -1
        side[xy[:, axis] == upper[axis], axis] = 1
    canon = np.arange(n)  # each vertex, or the copy on the lower sides that it translates
    if periodic:
        for axis in (0, 1):
            hi, lo = np.flatnonzero(side[:, axis] == 1), np.flatnonzero(side[:, axis] == -1)
            if len(hi) and len(lo):
                other = 1 - axis
                d, j = cKDTree(xy[lo, other][:, None]).query(xy[hi, other][:, None])
                good = d <= 1e3 * tol
                canon[hi[good]] = lo[j[good]]
        for _ in range(3):
            canon = canon[canon]
    # the short edges between grains or on the sides: their ends merged
    parent = np.arange(n)
    for (a, b), ps in _kept_edges(loops, owner).items():
        kept = len(ps) == 1 or owner[ps[0]] != owner[ps[1]]
        if kept and np.linalg.norm(xy[a] - xy[b]) < delta:
            ra, rb = _find(parent, int(canon[a])), _find(parent, int(canon[b]))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([_find(parent, int(canon[v])) for v in range(n)], dtype=np.int64)
    position = {}
    for r in np.unique(roots):
        members = np.unique(canon[roots == r])
        p = xy[members].copy()
        if periodic:
            p -= lengths * np.round((p - xy[r]) / lengths)
        target = p.mean(axis=0)
        for axis in (0, 1):
            fixed = side[members, axis] != 0
            if fixed.any():
                target[axis] = p[fixed, axis][0]
        position[int(r)] = target
    new = np.empty_like(xy)
    for v in range(n):
        c = int(canon[v])
        target = position[int(roots[v])]
        if periodic:  # the image of the merged vertex nearest to the canonical copy
            target = target + lengths * np.round((xy[c] - target) / lengths)
        new[v] = target + (xy[v] - xy[c])
    # the vertices near a side, onto it (the caps of the cells cut by the box), or onto the
    # vertex of the side nearer than delta (no short edge of the side)
    for axis in (0, 1):
        other = 1 - axis
        for value in (lower[axis], upper[axis]):
            near = np.flatnonzero((np.abs(new[:, axis] - value) < delta) & (new[:, axis] != value))
            if not len(near):
                continue
            targets = new[near].copy()
            targets[:, axis] = value
            on = np.flatnonzero(new[:, axis] == value)
            if len(on):
                d, j = cKDTree(new[on, other][:, None]).query(targets[:, other][:, None])
                close = d < delta
                targets[close] = new[on[j[close]]]
            new[near] = targets
    for axis in (0, 1):  # exactly on the sides
        for value in (lower[axis], upper[axis]):
            new[np.abs(new[:, axis] - value) <= tol, axis] = value
    index = _merged(new, tol)
    xy2 = np.zeros((index.max() + 1, 2))
    xy2[index] = new
    out_loops, out_owner = [], []
    box = float(np.prod(lengths))
    for loop, g in zip(loops, owner, strict=True):
        ids = _without_spikes(xy2, index[np.asarray(loop)], tol)
        if len(ids) < 3:
            continue
        area = _shoelace(xy2[ids])
        if area <= 1e-14 * box:
            if area < -1e-12 * box:
                return None  # a piece folded over
            continue
        out_loops.append(ids)
        out_owner.append(g)
    total = sum(_shoelace(xy2[loop]) for loop in out_loops)
    if abs(total - box) > 1e-9 * box:
        return None
    if periodic:  # the vertices of the sides and their translates on the opposite sides
        xy2, out_loops = _matched_sides(xy2, out_loops, lower, upper, tol)
    used, inverse = np.unique(np.concatenate(out_loops), return_inverse=True)
    starts = np.cumsum([0] + [len(loop) for loop in out_loops])
    out_loops = [inverse[a:b] for a, b in itertools.pairwise(starts)]
    return xy2[used], out_loops, np.asarray(out_owner, dtype=np.int64)


def _without_spikes(xy: np.ndarray, ids: np.ndarray, tol: float) -> np.ndarray:
    """A loop without its repeated vertices and the vertices where it turns back along a
    line (the spikes that a vertex moved onto a side, or merged, leaves: a cap gone)."""
    ids = list(int(k) for k in ids)
    scale = float(np.max(np.abs(xy))) if len(xy) else 1.0
    changed = True
    while changed and len(ids) >= 3:
        changed = False
        for i in range(len(ids)):
            a, b, c = ids[i - 1], ids[i], ids[(i + 1) % len(ids)]
            if b == a or b == c:
                del ids[i]
                changed = True
                break
            u, w = xy[b] - xy[a], xy[c] - xy[b]
            flat = abs(u[0] * w[1] - u[1] * w[0]) <= tol * scale * (
                np.abs(u).sum() + np.abs(w).sum()
            )
            if flat and u @ w < 0:
                del ids[i]
                changed = True
                break
    return np.asarray(ids, dtype=np.int64)


def _matched_sides(xy, loops, lower, upper, tol):
    """The vertices of each side translated onto the opposite side where it has none there,
    inserted in the edge of that side that holds them."""
    xy = [tuple(p) for p in xy]
    loops = [list(loop) for loop in loops]
    for axis in (0, 1):
        other = 1 - axis
        for here, there in ((lower[axis], upper[axis]), (upper[axis], lower[axis])):
            used = sorted({int(k) for loop in loops for k in loop})  # (not those of no loop)
            mine = [k for k in used if xy[k][axis] == here]
            theirs = np.array([xy[k][other] for k in used if xy[k][axis] == there])
            for k in mine:
                t = xy[k][other]
                if len(theirs) and np.min(np.abs(theirs - t)) <= 1e3 * tol:
                    continue
                point = [0.0, 0.0]
                point[axis], point[other] = there, t
                new = len(xy)
                for loop in loops:
                    hit = False
                    for i in range(len(loop)):
                        a, b = loop[i], loop[(i + 1) % len(loop)]
                        pa, pb = xy[a], xy[b]
                        low, high = sorted((pa[other], pb[other]))
                        if pa[axis] == there and pb[axis] == there and low < t < high:
                            loop.insert(i + 1, new)
                            hit = True
                            break
                    if hit:
                        break
                xy.append(tuple(point))
                theirs = np.append(theirs, t)
    return np.asarray(xy, dtype=float), [np.asarray(loop, dtype=np.int64) for loop in loops]


def _ear_clipped(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """The triangles of a simple polygon (counterclockwise ring of indices of ``points``) by
    ear clipping."""
    ring = list(int(k) for k in ring)
    out = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(ring) > 3 and guard < 10 * len(ring) ** 2:
        guard += 1
        for i in range(len(ring)):
            a, b, c = ring[i - 1], ring[i], ring[(i + 1) % len(ring)]
            pa, pb, pc = points[a], points[b], points[c]
            if cross(pa, pb, pc) <= 0:
                continue  # a reflex or flat corner
            inside = False
            for k in ring:
                if k in (a, b, c):
                    continue
                q = points[k]
                if cross(pa, pb, q) >= 0 and cross(pb, pc, q) >= 0 and cross(pc, pa, q) >= 0:
                    inside = True
                    break
            if not inside:
                out.append((a, b, c))
                del ring[i]
                break
        else:
            break
    if len(ring) == 3:
        out.append(tuple(ring))
    return np.asarray(out, dtype=np.int64).reshape(-1, 3)


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
                      z: float | None = None, min_feature: float | None = None,
                      executable: str | None = None, workdir=None,
                      timeout: float | None = None) -> TriMesh:  # fmt: skip
    """The conforming triangular mesh of the grains of a planar tessellation (see the
    module): one region per grain, the phase and the orientation of its grain to each
    element (``cell``: the grain). The polygons are exact: their edges between grains and
    on the sides are subdivided at the sizes of a field graded from their small features
    (:class:`_SizeField`: short edges, thin grains, close corners) up to ``size``, and
    ``mmg2d`` remeshes the grains within them without moving them (``-nosurf``). With
    ``z``, the exact section of a tessellation in space by the plane at that height (its
    grains cut by the plane: :func:`planar_pieces`). ``min_feature`` (default ``size / 4``,
    as the cells in space): the edges between grains shorter than it collapsed and the
    vertices nearer than it to a side moved onto it (:func:`_regularized`); 0 keeps the
    exact polygons, at the price of flat elements next to their small features (the thin
    caps of the cells that the box cuts). Where the polygons would fold, the features are
    removed at half that size, three times at most, before the exact polygons."""
    from morphogen.mesh.conforming import mmg_executable

    if z is None and not planar_tessellation(micro):
        raise ValueError("tessellation mesh: the germs of a tessellation in one plane, or the "
                         "height z of the section of a tessellation in space")  # fmt: skip
    if z is not None and not exact_section(micro):
        raise ValueError("tessellation mesh: the exact sections of the tessellations without "
                         "a boundary phase")  # fmt: skip
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
    vertices, loops, piece_cell = planar_pieces(micro, z)
    owner = grains[piece_cell]
    xy = vertices.copy()
    for axis in (0, 1):  # the vertices of the sides exactly on them
        for value in (lower[axis], upper[axis]):
            xy[np.abs(xy[:, axis] - value) <= tol, axis] = value
    feature = size / 4.0 if min_feature is None else float(min_feature)
    for _ in range(4):  # the features removed at half the size where the polygons fold
        if not feature > 0:
            break
        regular = _regularized(xy, loops, owner, lower, upper, periodic, feature, tol)
        if regular is not None:
            xy, loops, owner = regular
            break
        log.info("tessellation mesh: the polygons fold when their features below %g are "
                 "removed; below half of it instead", feature)  # fmt: skip
        feature /= 2.0
    else:
        log.warning("tessellation mesh: the polygons fold when their small features are "
                    "removed; the exact polygons instead")  # fmt: skip
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
        ring = np.asarray(ring)
        centre = xy[loop].mean(axis=0)
        fan = np.column_stack([ring, np.roll(ring, -1)])
        folds = _signed_areas(np.vstack([xy, centre]), np.column_stack(
            [fan, np.full(len(ring), len(xy))])) <= 0  # fmt: skip
        if folds.any():  # a piece that its regularization left not star-shaped
            ears = _ear_clipped(xy, ring)
            tris.append(ears)
            refs.append(np.full(len(ears), g + 1))
            continue
        c = len(xy) + len(centres)
        centres.append(centre)
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
    height = float(domain.lower[2] + 0.5 * domain.lengths[2]) if z is None else float(z)
    what = "tessellation (exact polygons, mmg2d)" if z is None else "section of a tessellation"
    mesh = TriMesh(out["Vertices"], out["Triangles"], grain_phase[cell], lower, upper,
                   periodic, {"mesher": what, "size": size, "n_grains": n_grains},
                   cell=cell, orientations=micro.orientations, z=height,
                   thickness=float(domain.lengths[2]))  # fmt: skip
    mesh = _kept(mesh, phases)
    return mesh.quadratic() if order == 2 else mesh


# ----------------------------------------------------------------------------------------
# The exact sections of discs


#: The elements across the narrowest gap between two discs, or between a disc and a side
GAP_ELEMENTS = 3
#: The fewest segments of a whole circle
CIRCLE_SEGMENTS = 24
#: The sizes at the corners of a union of discs, where the fields are singular: those of the
#: elements around them over this
CORNER_REFINEMENT = 8
#: The reference of the edges of the circles (in the files of mmg2d)
REF_CIRCLE = 10


def disc_sections(micro, z: float | None = None):
    """The discs of the section of a union of spheres and of segments along ``z`` (the
    straight fibres along ``z`` of :class:`~morphogen.models.long_fibers.LongFibers`) in the
    plane ``z`` (default: the middle of its box), the periodic images along ``z`` taken:
    their centres ``(n, 2)`` and radii, the concentric discs of the segments of a fibre
    merged; None for another structure, or for segments that are not along ``z``, whose
    sections are not discs."""
    from morphogen.models.union import PrimitiveUnion

    if not isinstance(micro, PrimitiveUnion):
        return None
    domain = micro.domain
    z = float(domain.lower[2] + 0.5 * domain.lengths[2]) if z is None else float(z)
    lz = float(domain.lengths[2])
    shifts = (-1.0, 0.0, 1.0) if domain.periodic else (0.0,)
    centres, radii = [np.zeros((0, 2))], [np.zeros(0)]
    spheres, segments = micro.spheres, micro.segments
    if len(spheres):
        c, r = spheres.centers, spheres.radii
        for k in shifts:
            d = z - (c[:, 2] + k * lz)
            cut = np.abs(d) < r
            centres.append(c[cut, :2])
            radii.append(np.sqrt(r[cut] ** 2 - d[cut] ** 2))
    if len(segments):
        a, b, r = segments.start, segments.end, segments.radii
        if np.abs(a[:, :2] - b[:, :2]).max() > 1e-12 * float(domain.lengths.max()):
            return None  # oblique segments: their sections are not discs
        lo, hi = np.minimum(a[:, 2], b[:, 2]), np.maximum(a[:, 2], b[:, 2])
        for k in shifts:
            height = z - k * lz
            d = np.maximum(np.maximum(lo - height, height - hi), 0.0)  # to the axis
            cut = d < r if segments.round_caps else d == 0.0
            centres.append(a[cut, :2])
            radii.append(np.sqrt(np.maximum(r[cut] ** 2 - d[cut] ** 2, 0.0)))
    centres, radii = np.concatenate(centres), np.concatenate(radii)
    keep = radii > 0.0
    centres, radii = centres[keep], radii[keep]
    if not len(radii):
        return centres, radii
    tol = 1e-9 * float(domain.lengths.max())
    key = np.round(centres / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    largest = np.zeros(len(first))
    np.maximum.at(largest, inverse.ravel(), radii)
    return centres[first], largest


def _grown(union, by: float):
    """The union of the same spheres and segments, their radii grown by ``by`` (the
    coating), or shrunk (the bores of shells)."""
    from morphogen.models.union import PrimitiveUnion
    from morphogen.primitives import Segments, Spheres

    spheres, segments = union.spheres, union.segments
    return PrimitiveUnion(
        union.domain,
        Spheres(spheres.centers, np.maximum(spheres.radii + by, 0.0)),
        Segments(segments.start, segments.end, np.maximum(segments.radii + by, 0.0),
                 segments.round_caps),
    )  # fmt: skip


def disc_layers(micro, z: float | None = None):
    """The discs of the section in the plane ``z`` of a union of spheres and of segments
    along ``z`` (:func:`disc_sections`), or of such a union coated (a coating outside, or
    the shells of hollowed members: :class:`~morphogen.models.nested.CoatedStructure`):
    their centres ``(n, 2)``, the radii of their concentric circles ``(n, m)`` from the
    innermost (0 where a disc has no such circle: the cut of a sphere in its coating
    alone), and the phases ``(m + 1,)`` inside the innermost circle, between the circles,
    and outside them; None for another structure."""
    from morphogen.models.nested import CoatedStructure

    if not isinstance(micro, CoatedStructure):
        found = disc_sections(micro, z)
        if found is None:
            return None
        centres, radii = found
        return centres, radii[:, None], np.array([1, 0])
    inner = micro.inner
    if disc_sections(inner, z) is None:
        return None
    t = float(micro.thickness)
    if micro.side == "outside":
        circles = (disc_sections(inner, z), disc_sections(_grown(inner, t), z))
        phases = np.array([1, micro.phase, 0])
    else:
        circles = (disc_sections(_grown(inner, -t), z), disc_sections(inner, z))
        phases = np.array([micro.core_phase, 1, 0])
    outer_c, outer_r = circles[1]
    inner_c, inner_r = circles[0]
    radii = np.zeros((len(outer_r), 2))
    radii[:, 1] = outer_r
    if len(inner_r) and len(outer_r):
        d, j = cKDTree(outer_c).query(inner_c)
        tol = 1e-9 * float(micro.domain.lengths.max())
        if np.any(d > tol):
            return None  # (concentric circles only)
        radii[j, 0] = inner_r
    return outer_c, radii, phases


def _crossings(c, r, lower, upper, tol):
    """Where a circle crosses the sides of the rectangle: ``(axis, side, t)``, ``t`` along
    the side; tangent circles do not cross."""
    out = []
    for axis in (0, 1):
        other = 1 - axis
        for side, value in ((0, lower[axis]), (1, upper[axis])):
            d = value - c[axis]
            if abs(d) >= r - tol:
                continue
            half = np.sqrt(r * r - d * d)
            for t in (c[other] - half, c[other] + half):
                if lower[other] - tol <= t <= upper[other] + tol:
                    out.append((axis, side, float(np.clip(t, lower[other], upper[other]))))
    return out


def _point(axis: int, side: int, t: float, lower, upper) -> np.ndarray:
    """The point at ``t`` along a side of the rectangle."""
    p = np.empty(2)
    p[axis], p[1 - axis] = (lower, upper)[side][axis], t
    return p


class _Union:
    """The points where the circles of discs that meet cross each other (``points``: two per
    pair, keyed ``(i, j, k)``), whether each lies on the boundary of the union of the discs
    (``valid``: outside the other discs), the pairs that cross (``crossing``), the discs
    inside others (``hidden``), and the test of points of the plane."""

    def __init__(self, cs: np.ndarray, rs: np.ndarray, meeting: np.ndarray, tol: float):
        self.cs, self.rs, self.tol = cs, rs, tol
        self.tree = cKDTree(cs) if len(cs) else None
        self.reach = float(rs.max()) if len(rs) else 0.0
        self.points: dict[tuple[int, int, int], np.ndarray] = {}
        self.valid: dict[tuple[int, int, int], bool] = {}
        self.crossing: set[tuple[int, int]] = set()
        self.hidden = np.zeros(len(cs), dtype=bool)
        self._by: dict[int, list] = {}
        for i, j in np.sort(meeting, axis=1).tolist():
            d = cs[j] - cs[i]
            dist = float(np.hypot(*d))
            if dist <= abs(rs[i] - rs[j]):  # a disc inside the other
                self.hidden[i if rs[i] < rs[j] else j] = True
                continue
            self.crossing.add((i, j))
            a = (rs[i] ** 2 - rs[j] ** 2 + dist**2) / (2.0 * dist)
            h = np.sqrt(max(rs[i] ** 2 - a * a, 0.0))
            base, normal = cs[i] + a * d / dist, np.array([-d[1], d[0]]) / dist
            for k, sign in enumerate((1.0, -1.0)):
                point = base + sign * h * normal
                key = (i, j, k)
                self.points[key] = point
                self.valid[key] = self.outside(point, (i, j))
                self._by.setdefault(i, []).append((key, point))
                self._by.setdefault(j, []).append((key, point))

    def outside(self, point: np.ndarray, own=()) -> bool:
        """Whether a point lies outside the discs other than ``own``."""
        if self.tree is None:
            return True
        for k in self.tree.query_ball_point(point, self.reach):
            if k not in own and np.hypot(*(point - self.cs[k])) < self.rs[k] - self.tol:
                return False
        return True

    def of(self, i: int):
        """The points where the circle ``i`` crosses others."""
        return self._by.get(i, [])


class _DiscSizes:
    """The size of the elements along the circles and the sides: the least of the size and
    of ``h_k + GRADE |x - p_k|`` over points ``p_k`` along the circles and the sides, whose
    sizes ``h_k`` are a :data:`GAP_ELEMENTS`-th of their gaps to the nearest other circle
    (and, for the points of a circle, to the sides it does not cross; for those of a side,
    to the circles that do not cross it), at most the share of a circle of
    :data:`CIRCLE_SEGMENTS`, at least ``floor``; two circles that cross each other (the
    pairs ``crossing``) meet at a point of the boundary, not across a gap. At the corners of
    a union of discs (``corners``), where the fields are singular, the sizes are a
    :data:`CORNER_REFINEMENT`-th of what they would be."""

    def __init__(self, cs, rs, lower, upper, size, periodic, floor, crossing=frozenset(),
                 corners=()):  # fmt: skip
        self.size = float(size)
        self.lower = np.asarray(lower, dtype=float)
        self.lengths = np.asarray(upper, dtype=float) - self.lower
        self.periodic = bool(periodic)
        spacing = self.size / 4.0
        points, h = [], []
        tree = cKDTree(cs) if len(cs) else None
        reach = float(rs.max()) + self.size * GAP_ELEMENTS if len(rs) else 0.0
        for i, (c, r) in enumerate(zip(cs, rs, strict=True)):
            m = max(16, int(np.ceil(2.0 * np.pi * r / spacing)))
            a = 2.0 * np.pi * np.arange(m) / m
            p = c + r * np.column_stack([np.cos(a), np.sin(a)])
            gap = np.full(m, np.inf)
            for j in tree.query_ball_point(c, r + reach):
                if j != i and (min(i, j), max(i, j)) not in crossing:  # (concentric too)
                    gap = np.minimum(gap, np.abs(np.linalg.norm(p - cs[j], axis=1) - rs[j]))
            for axis in (0, 1):
                for value in (lower[axis], upper[axis]):
                    if abs(c[axis] - value) >= r:  # a side that the circle does not cross
                        gap = np.minimum(gap, np.abs(p[:, axis] - value))
            points.append(p)
            h.append(np.minimum(gap / GAP_ELEMENTS, 2.0 * np.pi * r / CIRCLE_SEGMENTS))
        for axis in (0, 1):
            other = 1 - axis
            n = max(2, int(np.ceil(self.lengths[other] / spacing)) + 1)
            t = np.linspace(lower[other], upper[other], n)
            for value in (lower[axis], upper[axis]):
                p = np.empty((n, 2))
                p[:, axis], p[:, other] = value, t
                gap = np.full(n, np.inf)
                apart = np.abs(cs[:, axis] - value) >= rs  # the circles that do not cross it
                for c, r in zip(cs[apart], rs[apart], strict=True):
                    gap = np.minimum(gap, np.abs(np.linalg.norm(p - c, axis=1) - r))
                points.append(p)
                h.append(gap / GAP_ELEMENTS)
        self.points = np.concatenate(points)
        self.h = np.clip(np.concatenate(h), floor, self.size)
        self.reach = self.size / GRADE
        self._index()
        corners = np.asarray(corners, dtype=float).reshape(-1, 2)
        if len(corners):
            refined = np.maximum(self(corners) / CORNER_REFINEMENT, floor)
            self.points = np.concatenate([self.points, corners])
            self.h = np.concatenate([self.h, refined])
            self._index()

    def _index(self) -> None:
        """The tree of the points (of their images in a periodic box)."""
        if self.periodic:
            wrapped = np.mod(self.points - self.lower, self.lengths)
            wrapped[wrapped >= self.lengths] = 0.0
            self.tree = cKDTree(wrapped, boxsize=self.lengths)
        else:
            self.tree = cKDTree(self.points)

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


def _marched(length: float, at, field: _DiscSizes, least: int = 1) -> np.ndarray:
    """The arc lengths (from 0 to ``length``) of the nodes of a curve whose points are
    ``at(s)``, at the sizes of the field: marched from its start, the steps then scaled to
    end on its end; ``least`` segments at least."""
    steps, s = [], 0.0
    while s < length:
        h = float(field(at(np.array([s])))[0])
        steps.append(h)
        s += h
    steps = np.asarray(steps)
    n = max(least, round(len(steps) - 1 + (length - float(steps[:-1].sum())) / steps[-1]))
    cum = np.concatenate([[0.0], np.cumsum(np.resize(steps, n))])
    return cum * length / cum[-1]


def disc_mesh(micro, discs=None, *, size: float, min_size: float | None = None,
              hausdorff: float | None = None, gradation: float = 2.0, z: float | None = None,
              phases="all", order: int = 1, curved: bool | None = None,
              periodic: bool | None = None, executable: str | None = None, workdir=None,
              timeout: float | None = None) -> TriMesh | None:  # fmt: skip
    """The triangular mesh of a section made of discs (the straight fibres along ``z`` of a
    union of segments, the sections of spheres, coated or hollow: :func:`disc_layers`; or
    ``discs``, their centres and radii, or their centres, the radii of their concentric
    circles and the phases of the bands between them): their circles discretized exactly,
    at sizes graded from the gaps between them (:data:`GAP_ELEMENTS` elements across the
    narrowest, and across the coatings; at least ``min_size``, by default a 64th of
    ``size``), the sides of a periodic box matching; ``mmg2d`` meshes the regions within
    them, which keep their nodes. Plain discs that cross each other (a Boolean model of
    discs) are meshed by the boundary of their union: their circles cut where they cross,
    these corners nodes of both, the sizes graded toward them (:data:`CORNER_REFINEMENT`),
    the arcs inside other discs and the discs inside others left out. Six-node elements
    have their middle nodes on the circles
    (:func:`~morphogen.mesh.curved.curved_triangles`), by default. None for discs that
    touch, outside or inside, and for coated ones that meet (the level sets of their union
    then)."""
    from morphogen.mesh.conforming import mmg_executable

    domain = micro.domain
    size = float(size)
    if not size > 0:
        raise ValueError("size: must be positive")
    if order not in (1, 2):
        raise ValueError("order: 1 or 2")
    if curved and order == 1:
        raise ValueError("curved: the edges of six-node elements (order 2)")
    lower = np.asarray(domain.lower[:2], dtype=float)
    upper = np.asarray(domain.upper[:2], dtype=float)
    lengths = upper - lower
    periodic = bool(domain.periodic) if periodic is None else bool(periodic)
    if periodic and not domain.periodic:
        raise ValueError("matching periodic sides need a periodic structure")
    z = float(domain.lower[2] + 0.5 * domain.lengths[2]) if z is None else float(z)
    discs = disc_layers(micro, z) if discs is None else discs
    if discs is None:
        return None
    if len(discs) == 2:  # the centres and radii of plain discs
        centres, layers, bands = discs[0], np.asarray(discs[1])[:, None], [1, 0]
    else:
        centres, layers, bands = discs
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    layers = np.asarray(layers, dtype=float).reshape(len(centres), -1)
    bands = np.asarray(bands, dtype=np.int64)
    tol = 1e-9 * float(lengths.max())
    # 1. the discs that reach the box, and their images across a periodic box
    if periodic and len(centres):
        shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float)
        ds = (centres[None, :, :] + (shifts * lengths)[:, None, :]).reshape(-1, 2)
        dl = np.tile(layers, (len(shifts), 1))
    else:
        ds, dl = centres, layers
    do = dl.max(axis=1) if dl.size else np.zeros(len(ds))
    margin = size * GAP_ELEMENTS
    near = np.all((ds + do[:, None] > lower - margin) & (ds - do[:, None] < upper + margin),
                  axis=1)  # fmt: skip
    ds, dl, do = ds[near], dl[near], do[near]
    meeting = np.zeros((0, 2), dtype=np.int64)
    if len(do) > 1:
        pairs = cKDTree(ds).query_pairs(2.0 * float(do.max()) + tol, output_type="ndarray")
        if len(pairs):
            distance = np.linalg.norm(ds[pairs[:, 0]] - ds[pairs[:, 1]], axis=1)
            gap = distance - do[pairs[:, 0]] - do[pairs[:, 1]]
            meeting = pairs[gap <= tol]
            nested = np.abs(do[pairs[:, 0]] - do[pairs[:, 1]])
            if len(meeting) and (layers.shape[1] > 1  # coatings that meet
                                 or np.any(np.abs(gap) <= tol)  # tangent discs
                                 or np.any(np.abs(distance - nested) <= tol)):  # fmt: skip
                return None  # the level sets of their union
    union = _Union(ds, do, meeting, tol)  # of the discs (one layer each where they meet)
    # the circles: the layers of the discs, but those of discs inside others
    which, layer = np.nonzero((dl > 0.0) & ~union.hidden[:, None])
    cs, rs = ds[which], dl[which, layer]
    reach = np.flatnonzero(np.all((cs + rs[:, None] > lower + tol)
                                  & (cs - rs[:, None] < upper - tol), axis=1))  # fmt: skip
    floor = size / 64.0 if min_size is None else float(min_size)
    circle = {int(k): n for n, k in enumerate(which)}  # (one each where discs meet)
    crossing = {(circle[i], circle[j]) for i, j in union.crossing
                if i in circle and j in circle}  # fmt: skip
    corners = [p for key, p in union.points.items() if union.valid[key]]
    field = _DiscSizes(cs, rs, lower, upper, size, periodic, floor, crossing, corners)
    cuts = [[(a, s, t) for a, s, t in _crossings(cs[i], rs[i], lower, upper, tol)
             if union.outside(_point(a, s, t, lower, upper), (which[i],))]
            for i in reach]  # fmt: skip
    # 2. the nodes of the sides: their corners, the crossings, the points graded between;
    #    those of the upper sides of a periodic box the translates of the lower ones
    xy: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    refs: list[int] = []
    side_nodes: dict[tuple[int, int], tuple[np.ndarray, list[int]]] = {}
    corner: dict[tuple[int, int], int] = {}

    def corner_node(i: int, j: int) -> int:
        if (i, j) not in corner:
            corner[(i, j)] = len(xy)
            xy.append(np.array([(lower, upper)[i][0], (lower, upper)[j][1]]))
        return corner[(i, j)]

    for axis in (0, 1):
        other = 1 - axis
        for side in (0, 1):
            ts = [t for found in cuts for a, s, t in found
                  if a == axis and (s == side or periodic)]  # fmt: skip
            ts = np.sort(np.asarray(ts, dtype=float))  # (those of both sides, merged)
            ts = ts[np.concatenate([[True], np.diff(ts) > tol])] if len(ts) else ts
            ts = ts[(ts > lower[other] + tol) & (ts < upper[other] - tol)]
            ts = np.concatenate([[lower[other]], ts, [upper[other]]])
            value = (lower, upper)[side][axis]

            def along(s, t0, axis=axis, other=other, value=value):
                p = np.empty((len(s), 2))
                p[:, axis], p[:, other] = value, t0 + s
                return p

            if periodic and side == 1:  # the translate of the lower side
                chain = list(side_nodes[(axis, 0)][0])
            else:
                chain = [ts[0]]
                for t0, t1 in itertools.pairwise(ts):
                    params = _marched(t1 - t0, lambda s, t0=t0: along(s, t0), field)
                    chain += [*(t0 + params[1:-1]), t1]
            ends = (corner_node(*((side, 0) if axis == 0 else (0, side))),
                    corner_node(*((side, 1) if axis == 0 else (1, side))))  # fmt: skip
            ids = [ends[0]]
            for t in chain[1:-1]:
                ids.append(len(xy))
                xy.append(along(np.array([t]), 0.0)[0])
            ids.append(ends[1])
            side_nodes[(axis, side)] = (np.asarray(chain), ids)
            edges += list(itertools.pairwise(ids))
            refs += [SIDE_REFS[(axis, side)]] * (len(ids) - 1)
    # 3. the points where the circles of discs that meet cross each other: the corners of
    #    their union, nodes where they are on its boundary within the box
    cusp: dict[tuple[int, int, int], int] = {}
    for key, point in union.points.items():
        if union.valid[key] and np.all(point > lower + tol) and np.all(point < upper - tol):
            cusp[key] = len(xy)
            xy.append(point)
    # 4. the arcs of the circles inside the box and outside the other discs, between their
    #    crossings of the sides and of the other circles
    for i, found in zip(reach, cuts, strict=True):
        c, r = cs[i], rs[i]
        marks = []  # (angle, node): the nodes of the points on the boundary of the union
        for axis, side, t in _crossings(c, r, lower, upper, tol):
            chain, ids = side_nodes[(axis, side)]
            k = int(np.argmin(np.abs(chain - t)))
            on = (axis, side, t) in found and abs(chain[k] - t) <= 1e3 * tol
            point = _point(axis, side, t, lower, upper)
            marks.append((float(np.arctan2(*(point - c)[::-1])), ids[k] if on else None))
        for key, point in union.of(which[i]):
            marks.append((float(np.arctan2(*(point - c)[::-1])), cusp.get(key)))
        marks.sort(key=lambda m: m[0])
        merged: list[tuple[float, int | None]] = []
        for angle, node in marks:  # a crossing at a corner, or of a side at a corner, once
            if merged and angle - merged[-1][0] <= 1e3 * tol / r:
                merged[-1] = (merged[-1][0], merged[-1][1] if merged[-1][1] is not None
                              else node)  # fmt: skip
                continue
            merged.append((angle, node))
        if merged:
            spans = [(n0, n1, a0, a1) for (a0, n0), (a1, n1)
                     in zip(merged, [*merged[1:], (merged[0][0] + 2.0 * np.pi, merged[0][1])],
                            strict=True)]  # fmt: skip
        else:
            spans = [(None, None, 0.0, 2.0 * np.pi)]
        for start, end, a0, a1 in spans:
            middle = c + r * np.array([np.cos(0.5 * (a0 + a1)), np.sin(0.5 * (a0 + a1))])
            if np.any(middle <= lower + tol) or np.any(middle >= upper - tol):
                continue  # an arc outside the box
            if not union.outside(middle, (which[i],)):
                continue  # an arc inside another disc
            if merged and (start is None or end is None):  # pragma: no cover - degenerate
                return None

            def on(s, a0=a0, c=c, r=r):
                angle = a0 + s / r
                return c + r * np.column_stack([np.cos(angle), np.sin(angle)])

            least = max(2, int(np.ceil((a1 - a0) / (2.0 * np.pi) * CIRCLE_SEGMENTS)))
            points = on(_marched(r * (a1 - a0), on, field, least))
            ids = [start if start is not None else len(xy)]
            if start is None:
                xy.append(points[0])
            for p in points[1:-1]:
                ids.append(len(xy))
                xy.append(p)
            ids.append(end if end is not None else ids[0])
            edges += list(itertools.pairwise(ids))
            refs += [REF_CIRCLE] * (len(ids) - 1)
    xy = np.asarray(xy)
    edges = np.asarray(edges, dtype=np.int64)
    refs = np.asarray(refs, dtype=np.int64)
    shortest = float(np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1).min())
    exe = mmg_executable(executable, "mmg2d")
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        folder = Path(tmp)
        medit.write_mesh(folder / "discs.mesh", xy, None, np.zeros((0, 3), dtype=np.int64),
                         edges=edges, edge_refs=refs, required_vertices=np.arange(len(xy)),
                         required_edges=np.arange(len(edges)))  # fmt: skip
        args = [*_mmg_args(size, 0.5 * shortest, hausdorff, gradation), "-nr", "-nosurf"]
        out = _run(exe, folder / "discs.mesh", None, folder / "out.mesh", args, timeout)
    nodes, tri = out["Vertices"], out["Triangles"]
    phase = np.full(len(tri), bands[-1], dtype=np.int64)
    if len(do):  # the triangles inside a disc: the band of their centroids
        centroids = cKDTree(nodes[tri].mean(axis=1))
        close = centroids.sparse_distance_matrix(cKDTree(ds), float(do.max()),
                                                 output_type="ndarray")  # fmt: skip
        inside = close["v"] < do[close["j"]]
        i, j, d = close["i"][inside], close["j"][inside], close["v"][inside]
        phase[i] = bands[(dl[j] < d[:, None]).sum(axis=1)]
    mesh = TriMesh(nodes, tri, phase, lower, upper, periodic,
                   {"mesher": "exact discs (mmg2d)", "size": size, "discs": len(centres),
                    "meeting": len(meeting), "z": z}, z=z,
                   thickness=float(domain.lengths[2]))  # fmt: skip
    mesh = _kept(mesh, phases)
    if order == 1:
        return mesh
    if curved is False:
        return mesh.quadratic()
    from morphogen.mesh.curved import curved_triangles

    return curved_triangles(mesh, micro)


def lattice_map(micro) -> bool:
    """Whether a structure is a map of grains: a lattice field of one layer through the
    thickness of its box, with the label of a grain per cell (the orientation maps of EBSD:
    the model ``ebsd``)."""
    from morphogen.models.lattice_field import LatticeField

    return (isinstance(micro, LatticeField) and micro.shape[2] == 1
            and micro.cell_labels is not None)  # fmt: skip


def lattice_mesh(micro, *, order: int = 1, phases="all", periodic: bool | None = None) -> TriMesh:
    """The mesh of the plane of a map of grains (:func:`lattice_map`): two triangles per
    cell (pixel), along the diagonals that alternate with its parity (the triangulation
    symmetric about the lines of the lattice, as :func:`background_triangulation`), each the
    phase and the grain of its cell, the grains their orientations — the map as measured,
    its grain boundaries the edges of its pixels. The cells without a grain (label -1) are
    one more grain of no orientation (zero angles)."""
    if not lattice_map(micro):
        raise ValueError("lattice mesh: a lattice field of one layer with grains (a map)")
    if order not in (1, 2):
        raise ValueError("order: 1 or 2")
    domain: Domain = micro.domain
    periodic = bool(domain.periodic) if periodic is None else bool(periodic)
    nx, ny, _ = micro.shape
    lower = np.asarray(domain.lower[:2], dtype=float)
    upper = np.asarray(domain.upper[:2], dtype=float)
    x = np.linspace(lower[0], upper[0], nx + 1)
    y = np.linspace(lower[1], upper[1], ny + 1)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    i, j = i.ravel(), j.ravel()
    a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
    even = (i + j) % 2 == 0
    first = np.where(even[:, None], np.column_stack([a, b, c]), np.column_stack([a, b, d]))
    second = np.where(even[:, None], np.column_stack([a, c, d]), np.column_stack([b, c, d]))
    tri = np.stack([first, second], axis=1).reshape(-1, 3)  # the two of each cell together
    cells = np.repeat(np.arange(nx * ny), 2)
    phase = micro.phases[:, :, 0].ravel()[cells].astype(np.int64)
    labels = micro.cell_labels[:, :, 0].ravel()[cells].astype(np.int64)
    orientations = micro.orientations
    if labels.min(initial=0) < 0:  # the cells of no grain: one more grain
        extra = int(labels.max(initial=-1)) + 1
        labels = np.where(labels < 0, extra, labels)
        if orientations is not None:
            orientations = np.vstack([orientations, np.zeros((extra + 1 - len(orientations), 3))])
    mesh = TriMesh(nodes, tri, phase, lower, upper, periodic,
                   {"mesher": "map of grains (two triangles per cell)", "cells": nx * ny},
                   cell=labels, orientations=orientations,
                   z=float(domain.lower[2] + 0.5 * domain.lengths[2]),
                   thickness=float(domain.lengths[2]))  # fmt: skip
    mesh = _kept(mesh, phases)
    return mesh.quadratic() if order == 2 else mesh


def exact_section(micro) -> bool:
    """Whether the sections of a structure are exact polygons: a tessellation (its grains
    gathered from its cells) without a boundary phase."""
    from morphogen.models.tessellation import Tessellation

    return (isinstance(micro, Tessellation) and bool(micro.n_cells)
            and not micro.boundary_thickness > 0)  # fmt: skip


def plane_mesh(micro, *, size: float, **options) -> TriMesh:
    """The triangular mesh of a planar structure: the exact grains of a planar tessellation
    (:func:`tessellation_mesh`), the exact section of a tessellation in space in the plane
    ``z`` (by default the middle of its box), the exact discs of the section of straight
    fibres along ``z`` and of spheres, coated or hollow, and the union of plain ones that
    cross (:func:`disc_mesh`), the section of any other structure in that plane
    (:func:`section_mesh`), a map of grains by its pixels (:func:`lattice_mesh`).
    ``min_feature``: that of the polygons of tessellations; ``curved``: the six-node
    elements of the sections with curved edges along their interfaces (by default; the
    edges of polygons and pixels are straight)."""
    curved = options.get("curved")
    if curved and options.get("order", 1) == 1:
        raise ValueError("curved: the edges of six-node elements (order 2)")
    if lattice_map(micro):
        return lattice_mesh(micro, order=options.get("order", 1),
                            phases=options.get("phases", "all"),
                            periodic=options.get("periodic"))  # fmt: skip
    if planar_tessellation(micro) or exact_section(micro):
        options.pop("background", None)
        options.pop("curved", None)
        z = options.pop("z", None)
        if not planar_tessellation(micro):
            domain = micro.domain
            z = float(domain.lower[2] + 0.5 * domain.lengths[2]) if z is None else float(z)
        return tessellation_mesh(micro, size=size, z=z, **options)
    options.pop("min_feature", None)  # (the level sets have no features to remove)
    discs = disc_layers(micro, options.get("z"))
    if discs is not None:
        exact = {k: v for k, v in options.items() if k != "background"}
        mesh = disc_mesh(micro, discs, size=size, **exact)
        if mesh is not None:
            return mesh
        log.info(
            "plane mesh: discs that touch or coatings that meet, meshed by the level "
            "sets of their union"
        )
    return section_mesh(micro, size=size, **options)


__all__ = ["REF_PHASE", "SIDE_REFS", "TRI6_EDGES", "TRI6_NODES", "TRI_GAUSS", "TriMesh",
           "background_triangulation", "corner_gradients", "curved_maps", "disc_layers",
           "disc_mesh", "disc_sections", "edge_deviations", "exact_section", "jacobian_ratios",
           "lattice_map", "lattice_mesh",
           "planar_pieces", "planar_tessellation", "plane_level_set", "plane_mesh",
           "section_mesh", "section_weights", "tessellation_mesh"]  # fmt: skip
