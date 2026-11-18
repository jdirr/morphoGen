"""Tetrahedral meshes of a box: linear (4-node) or quadratic (10-node) elements.

Quadratic elements follow the VTK / Abaqus C3D10 node order: corners 0-3, then the
mid-edge nodes of edges (0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3). They are
isoparametric: a mid-edge node may leave the middle of its straight edge, and the edge
is then the parabola through its three nodes (:func:`morphogen.mesh.curved.curved_mesh`
puts the nodes of the interfaces on the implicit surfaces of the phases). An element is
*straight* when its mid-edge nodes are the middles of its edges, *curved* otherwise
(:meth:`TetMesh.deviations`, :attr:`TetMesh.curved`).

A mesh of the box is bounded by its faces. A mesh of whole periodic cells (``whole``)
fills the volume of the box without being cut by its faces: its outer boundary is made
of pairs of translated facets, and every node on it has translates by the periods of the
box (:meth:`TetMesh.periodic_links`). The faces of the elements on that boundary are
recorded (``outer``) when the mesh is made, so that they are known in a mesh of some of
its phases too, where the walls of the pores are free faces as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

#: Corner pairs of the six edges of a tetrahedron, in mid-node order.
TET_EDGES = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]])
#: Corner triples of the four faces (outward normals for positive tetrahedra), and the
#: mid-edge nodes of each face (indices into the 10-node element).
TET_FACES = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
TET_FACE_MIDS = np.array([[6, 5, 4], [4, 8, 7], [5, 9, 8], [7, 9, 6]])


#: A rule of the tetrahedron exact to degree 3 (barycentric coordinates, weights): the
#: Jacobian of a 10-node element is a polynomial of that degree
_CUBIC_RULE = (
    np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 1 / 6, 1 / 6, 1 / 6],
            [1 / 6, 0.5, 1 / 6, 1 / 6],
            [1 / 6, 1 / 6, 0.5, 1 / 6],
            [1 / 6, 1 / 6, 1 / 6, 0.5],
        ]
    ),
    np.array([-0.8, 0.45, 0.45, 0.45, 0.45]),
)
#: Barycentric coordinates of the ten nodes of a quadratic tetrahedron
TET10_NODES = np.concatenate(
    [np.eye(4), 0.5 * (np.eye(4)[TET_EDGES[:, 0]] + np.eye(4)[TET_EDGES[:, 1]])]
)


def tet_volumes(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Signed volumes of the tetrahedra of corners 0-3 (straight edges)."""
    p = nodes[elements[:, :4]]
    a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    return np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0


def edge_deviations(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """The departures (elements x 6 x 3) of the mid-edge nodes of 10-node tetrahedra from
    the middles of their straight edges: zero for straight elements."""
    ends = nodes[elements[:, TET_EDGES]]  # (n, 6, 2, 3)
    return nodes[elements[:, 4:]] - 0.5 * (ends[:, :, 0] + ends[:, :, 1])


def corner_gradients(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Gradients (elements x 4 x 3) of the barycentric coordinates of the straight
    tetrahedra of corners 0-3."""
    p = nodes[elements[:, :4]]
    edges = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)
    inverse = np.linalg.inv(edges)  # its rows: the gradients of L1, L2, L3
    return np.concatenate([-inverse.sum(axis=1, keepdims=True), inverse], axis=1)


def jacobian_ratios(gradients: np.ndarray, deviations: np.ndarray, bary) -> np.ndarray:
    """The ratio of the volume element of curved 10-node tetrahedra to that of their
    straight ones at points of barycentric coordinates ``bary`` (points x 4): elements x
    points. The curved element is the straight one moved by ``sum_k N_k(L) d_k``, with
    ``N_k = 4 L_i L_j`` the function of the mid-edge node of the edge (i, j) and ``d_k``
    its departure: the ratio is ``det(I + sum_k d_k (x) grad N_k)``."""
    bary = np.atleast_2d(np.asarray(bary, dtype=float))
    i, j = TET_EDGES[:, 0], TET_EDGES[:, 1]
    # grad N_k = 4 (L_i grad L_j + L_j grad L_i): (elements, points, 6, 3)
    grad = 4.0 * (
        bary[None, :, i, None] * gradients[:, None, j, :]
        + bary[None, :, j, None] * gradients[:, None, i, :]
    )
    f = np.einsum("eka,eqkb->eqab", deviations, grad)
    f += np.eye(3)
    return np.linalg.det(f)


def element_volumes(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Signed volumes of the elements: of the tetrahedra of their corners, or, for 10-node
    elements with curved edges, the integral of their Jacobian (a polynomial of degree 3
    in the barycentric coordinates, by a rule exact for it)."""
    volumes = tet_volumes(nodes, elements)
    if elements.shape[1] != 10:
        return volumes
    dev = edge_deviations(nodes, elements)
    scale = np.abs(volumes) ** (1.0 / 3.0)
    curved = np.flatnonzero(np.abs(dev).max(axis=(1, 2)) > 1e-10 * scale)
    if curved.size:
        points, weights = _CUBIC_RULE
        ratios = jacobian_ratios(corner_gradients(nodes, elements[curved]), dev[curved], points)
        volumes[curved] *= ratios @ weights
    return volumes


def tet_quality(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Mean-ratio-like quality: 1 for a regular tetrahedron, 0 for a flat one.

    ``q = 6 sqrt(2) V / l_rms^3`` with ``l_rms`` the root mean square edge length.
    """
    p = nodes[elements[:, :4]]
    l2 = sum(((p[:, b] - p[:, a]) ** 2).sum(axis=1) for a, b in TET_EDGES)
    vol = np.abs(tet_volumes(nodes, elements))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 6.0 * np.sqrt(2.0) * vol / (l2 / 6.0) ** 1.5
    return np.nan_to_num(q)


@dataclass
class TetMesh:
    """Tetrahedral mesh of the box ``[lower, upper]`` with a phase per element.

    ``periodic`` tells whether the meshed microstructure is periodic (periodic boundary
    conditions are then meaningful); the opposite faces need not have matching nodes.
    Meshes of polycrystals also give the cell (grain) of each element, ``cell``, and the
    crystal orientation of each cell, ``orientations`` (Bunge Euler angles in degrees).
    The crack meshes of a morphology (:func:`morphogen.fracture.lefm.crack_mesh`), whose
    phases are those of its materials, give the side of the crack of each element,
    ``side`` (0 or 1).
    """

    nodes: np.ndarray
    elements: np.ndarray
    phase: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    periodic: bool = False
    metadata: dict = field(default_factory=dict)
    cell: np.ndarray | None = None
    orientations: np.ndarray | None = None
    whole: bool = False
    #: whole periodic cells: which faces of each element (``TET_FACES`` order) bound the
    #: volume element; by default, the free faces of the mesh (all the cells meshed)
    outer: np.ndarray | None = None
    #: crack meshes of a morphology: the side of the crack of each element (0 or 1)
    side: np.ndarray | None = None
    #: meshes that follow the sharp edges of the phases: these edges (pairs of corner
    #: nodes), and the two pieces of the structure that meet along each
    #: (:meth:`~morphogen.models.base.Microstructure.implicit_pieces`)
    sharp: np.ndarray | None = None
    sharp_pieces: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.nodes = np.ascontiguousarray(self.nodes, dtype=np.float64)
        self.elements = np.ascontiguousarray(self.elements, dtype=np.int64)
        self.phase = np.ascontiguousarray(self.phase, dtype=np.int64)
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)
        if self.elements.ndim != 2 or self.elements.shape[1] not in (4, 10):
            raise ValueError("tetrahedra need 4 (linear) or 10 (quadratic) nodes")
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
            if (
                self.cell is not None
                and self.cell.size
                and self.cell.max() >= len(self.orientations)
            ):
                raise ValueError("orientations: one row per cell label")
        if self.whole and self.outer is None:
            self.outer = self._free_mask()
        if self.outer is not None:
            self.outer = np.ascontiguousarray(self.outer, dtype=bool)
            if self.outer.shape != (len(self.elements), 4):
                raise ValueError("outer: four flags (faces) per element")
        if self.side is not None:
            self.side = np.ascontiguousarray(self.side, dtype=np.int64)
            if self.side.shape != (len(self.elements),):
                raise ValueError("one side of the crack per element")
        if self.sharp is not None:
            self.sharp = np.ascontiguousarray(self.sharp, dtype=np.int64).reshape(-1, 2)
            if self.sharp_pieces is not None:
                self.sharp_pieces = np.ascontiguousarray(self.sharp_pieces, dtype=np.int64)
                if self.sharp_pieces.shape != self.sharp.shape:
                    raise ValueError("sharp_pieces: two pieces per sharp edge")

    def _derived(self, nodes, elements, keep=None, numbers=None) -> TetMesh:
        """A mesh with new nodes and elements and the same attributes (elements
        ``keep`` of this one; ``numbers``: the new numbers of its nodes, -1 for those
        that are left out)."""
        sel = slice(None) if keep is None else keep
        sharp, sharp_pieces = self.sharp, self.sharp_pieces
        if sharp is not None and numbers is not None:
            moved = numbers[sharp]
            there = (moved >= 0).all(axis=1)
            sharp = moved[there]
            sharp_pieces = None if sharp_pieces is None else sharp_pieces[there]
        return TetMesh(
            nodes=nodes,
            elements=elements,
            phase=self.phase[sel],
            lower=self.lower,
            upper=self.upper,
            periodic=self.periodic,
            metadata=dict(self.metadata),
            cell=None if self.cell is None else self.cell[sel],
            orientations=self.orientations,
            whole=self.whole,
            outer=None if self.outer is None else self.outer[sel],
            side=None if self.side is None else self.side[sel],
            sharp=sharp,
            sharp_pieces=sharp_pieces,
        )

    @property
    def order(self) -> int:
        return 1 if self.elements.shape[1] == 4 else 2

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def size(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def box_volume(self) -> float:
        return float(np.prod(self.size))

    def volumes(self) -> np.ndarray:
        """Signed volumes of the elements (those of curved 10-node elements exactly)."""
        return element_volumes(self.nodes, self.elements)

    def deviations(self) -> np.ndarray | None:
        """The departures (elements x 6 x 3) of the mid-edge nodes of a 10-node mesh from
        the middles of their straight edges (zero where the elements are straight); None
        for a 4-node mesh."""
        if self.order == 1:
            return None
        return edge_deviations(self.nodes, self.elements)

    @property
    def curved(self) -> bool:
        """Whether some 10-node elements have curved edges (mid-edge nodes off the
        middles of their edges, beyond the round-off of coordinates read from a file)."""
        if self.order == 1:
            return False
        if "curved" not in self._cache:
            dev = np.abs(self.deviations()).max(axis=(1, 2)) if self.n_elements else np.zeros(0)
            scale = np.abs(tet_volumes(self.nodes, self.elements)) ** (1.0 / 3.0)
            self._cache["curved"] = bool(np.any(dev > 1e-10 * scale))
        return self._cache["curved"]

    def min_jacobian(self) -> float:
        """The smallest ratio of the volume element of the 10-node elements to that of
        their straight tetrahedra, over their ten nodes and their four integration points
        (1 for a straight mesh; positive for valid elements)."""
        if not self.curved:
            return 1.0
        a, b = 0.5854101966249685, 0.1381966011250105
        gauss = np.full((4, 4), b) + (a - b) * np.eye(4)
        dev = self.deviations()
        bent = np.flatnonzero(np.abs(dev).max(axis=(1, 2)) > 0.0)
        ratios = jacobian_ratios(
            corner_gradients(self.nodes, self.elements[bent]),
            dev[bent],
            np.concatenate([TET10_NODES, gauss]),
        )
        return float(ratios.min())

    def quality(self) -> np.ndarray:
        return tet_quality(self.nodes, self.elements)

    def phase_fractions(self) -> dict[int, float]:
        """Volume of each phase over the volume of the box (missing phases: voids)."""
        vol = self.volumes()
        return {
            int(p): float(vol[self.phase == p].sum() / self.box_volume)
            for p in np.unique(self.phase)
        }

    def tolerance(self) -> float:
        return 1e-9 * float(np.max(self.size))

    def face_nodes(self, axis: int, side: int) -> np.ndarray:
        """Nodes on the face of the box normal to ``axis`` (``side`` 0: lower, 1: upper)."""
        target = self.lower[axis] if side == 0 else self.upper[axis]
        return np.flatnonzero(np.abs(self.nodes[:, axis] - target) <= self.tolerance())

    def boundary_nodes(self) -> np.ndarray:
        """Nodes on the boundary of the volume element: the faces of the box, or the
        outer boundary of whole periodic cells."""
        if self.whole:
            return np.unique(self.outer_faces())
        on = np.zeros(self.n_nodes, dtype=bool)
        for axis in range(3):
            for side in (0, 1):
                on[self.face_nodes(axis, side)] = True
        return np.flatnonzero(on)

    def _free_mask(self) -> np.ndarray:
        """Faces of each element (``TET_FACES`` order) that belong to no other element."""
        key = np.sort(self.elements[:, TET_FACES].reshape(-1, 3), axis=1)
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        return (counts[inverse.ravel()] == 1).reshape(-1, 4)

    def _faces(self, mask: np.ndarray) -> np.ndarray:
        """The faces flagged in an (elements, 4) mask: (k, 3) corners or (k, 6) for P2
        (corners then mid-edge nodes), oriented with outward normals."""
        elem, local = np.nonzero(mask)
        faces = self.elements[elem[:, None], TET_FACES[local]]
        if self.order == 2:
            faces = np.concatenate(
                [faces, self.elements[elem[:, None], TET_FACE_MIDS[local]]], axis=1
            )
        return faces

    def free_faces(self) -> np.ndarray:
        """Element faces belonging to a single element: (k, 3) corners or (k, 6) for P2
        (corners then mid-edge nodes), oriented with outward normals."""
        if "free_faces" not in self._cache:
            self._cache["free_faces"] = self._faces(self._free_mask())
        return self._cache["free_faces"]

    def _torus(self, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Classes of translates (by the periods of the box) among nodes, and the shift
        of each node from the first of its class (translates agree to the rounding of
        their coordinates, distinct nodes may be a micrometre apart in a unit box)."""
        from morphogen.mesh.polyhedra import torus_classes

        return torus_classes(self.nodes[candidates], self.lower, self.size, self.tolerance())

    def outer_faces(self) -> np.ndarray:
        """Faces bounding the volume element, with outward normals: those on the faces of
        the box, or the ``outer`` faces of whole periodic cells."""
        if not self.whole:
            return np.concatenate([self.boundary_faces(a, s) for a in range(3) for s in (0, 1)])
        if "outer_faces" not in self._cache:
            self._cache["outer_faces"] = self._faces(self.outer)
        return self._cache["outer_faces"]

    def periodic_links(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nodes of the boundary that are translates of others by the periods of the box:
        ``(slave, master, shift)`` with ``x_slave = x_master + shift * size`` (shift in
        periods), each slave linked to the first node of its class."""
        if "links" not in self._cache:
            candidates = self.boundary_nodes()
            if candidates.size == 0:
                empty = np.zeros(0, dtype=np.int64)
                return empty, empty, np.zeros((0, 3), dtype=np.int64)
            labels, shift = self._torus(candidates)
            first = np.full(int(labels.max()) + 1, -1)
            first[labels[::-1]] = np.arange(len(labels))[::-1]
            slave = np.flatnonzero(first[labels] != np.arange(len(labels)))
            master = first[labels[slave]]
            self._cache["links"] = (
                candidates[slave],
                candidates[master],
                shift[slave] - shift[master],
            )
        return self._cache["links"]

    def boundary_faces(self, axis: int, side: int) -> np.ndarray:
        """Element faces lying on a face of the box: (k, 3) corners or (k, 6) for P2
        (corners then mid-edge nodes), oriented with outward normals."""
        target = self.lower[axis] if side == 0 else self.upper[axis]
        tol = self.tolerance()
        on_plane = np.abs(self.nodes[:, axis] - target) <= tol
        out = []
        for f, mids in zip(TET_FACES, TET_FACE_MIDS, strict=True):
            sel = np.all(on_plane[self.elements[:, f]], axis=1)
            if not np.any(sel):
                continue
            faces = (
                self.elements[sel][:, f]
                if self.order == 1
                else np.concatenate([self.elements[sel][:, f], self.elements[sel][:, mids]], axis=1)
            )
            out.append(faces)
        if not out:
            return np.zeros((0, 3 if self.order == 1 else 6), dtype=np.int64)
        return np.concatenate(out)

    def periodic_pairs(self, axis: int) -> np.ndarray:
        """Pairs (node on the lower face, node on the upper face) at the same position
        across the box along ``axis`` (all face nodes for matching faces)."""

        lower = self.face_nodes(axis, 0)
        upper = self.face_nodes(axis, 1)
        if lower.size == 0 or upper.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        others = [d for d in range(3) if d != axis]
        dist, near = cKDTree(self.nodes[lower][:, others]).query(self.nodes[upper][:, others])
        ok = dist <= 1e-7 * float(np.max(self.size))
        return np.stack([lower[near[ok]], upper[ok]], axis=1)

    def matching_faces(self) -> bool:
        """Whether every node of a face has an image on the opposite face. Whole periodic
        cells: whether the faces bounding them come in translated pairs (in a mesh of
        some of the cells, the faces facing the others have none)."""
        if self.whole:
            faces = self.outer_faces()[:, :3]
            if not len(faces):
                return True
            nodes, local = np.unique(faces, return_inverse=True)
            labels, shift = self._torus(nodes)
            labels, shift = labels[local.reshape(-1, 3)], shift[local.reshape(-1, 3)]
            order = np.argsort(labels, axis=1)
            labels = np.take_along_axis(labels, order, axis=1)
            shift = np.take_along_axis(shift, order[..., None], axis=1)
            # a face and its translate: the same classes, the same shifts between corners
            key = np.concatenate([labels, (shift[:, 1:] - shift[:, :1]).reshape(-1, 6)], axis=1)
            _, counts = np.unique(key, axis=0, return_counts=True)
            return bool(np.all(counts == 2))
        for axis in range(3):
            n_pairs = len(self.periodic_pairs(axis))
            if n_pairs != self.face_nodes(axis, 0).size or n_pairs != self.face_nodes(axis, 1).size:
                return False
        return True

    def subset(self, phases) -> TetMesh:
        """Mesh of the elements of the given phases (unused nodes removed)."""
        return self.select(np.isin(self.phase, list(phases)))

    def select(self, keep) -> TetMesh:
        """Mesh of the elements ``keep`` (a mask or indices; unused nodes removed)."""
        elements = self.elements[keep]
        used, inverse = np.unique(elements, return_inverse=True)
        numbers = np.full(self.n_nodes, -1, dtype=np.int64)
        numbers[used] = np.arange(len(used))
        return self._derived(self.nodes[used], inverse.reshape(elements.shape), keep, numbers)

    def quadratic(self) -> TetMesh:
        """The same mesh with 10-node elements (mid-edge nodes on straight edges)."""
        if self.order == 2:
            return self
        edges = self.elements[:, TET_EDGES].reshape(-1, 2)
        edges.sort(axis=1)
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        mids = 0.5 * (self.nodes[unique[:, 0]] + self.nodes[unique[:, 1]])
        # mid-edge nodes on a face of the box must lie exactly on it
        nodes = np.concatenate([self.nodes, mids])
        mid_ids = self.n_nodes + inverse.reshape(-1, 6)
        return self._derived(nodes, np.concatenate([self.elements, mid_ids], axis=1))

    def straight(self) -> TetMesh:
        """The same 10-node mesh with straight edges (its mid-edge nodes at the middles
        of their edges); the mesh itself if it has no curved edge."""
        if not self.curved:
            return self
        nodes = self.nodes.copy()
        ends = self.elements[:, TET_EDGES]  # (n, 6, 2)
        nodes[self.elements[:, 4:]] = 0.5 * (self.nodes[ends[..., 0]] + self.nodes[ends[..., 1]])
        out = self._derived(nodes, self.elements)
        out.metadata.pop("curved", None)
        return out

    def linear(self) -> TetMesh:
        """The same mesh with 4-node elements (mid-edge nodes dropped)."""
        if self.order == 1:
            return self
        corners = self.elements[:, :4]
        used, inverse = np.unique(corners, return_inverse=True)
        numbers = np.full(self.n_nodes, -1, dtype=np.int64)
        numbers[used] = np.arange(len(used))
        return self._derived(self.nodes[used], inverse.reshape(corners.shape), None, numbers)

    def components(self) -> tuple[np.ndarray, int]:
        """Connected components of the elements (sharing at least one node)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        m = self.n_elements
        rows = np.repeat(np.arange(m), self.elements.shape[1])
        incidence = coo_matrix(
            (np.ones(rows.size), (rows, self.elements.ravel())), shape=(m, self.n_nodes)
        ).tocsr()
        n_comp, labels = connected_components(incidence @ incidence.T, directed=False)
        return labels, n_comp

    def summary(self) -> dict:
        q = self.quality()
        out = {
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "order": self.order,
            "quality_min": float(q.min()) if q.size else float("nan"),
            "quality_mean": float(q.mean()) if q.size else float("nan"),
            **{f"mesh_fraction_{p}": v for p, v in self.phase_fractions().items()},
        }
        if self.cell is not None:
            out["n_grains"] = int(np.unique(self.cell).size)
        return out
