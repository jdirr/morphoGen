"""The surfaces that touch: faces of a tetrahedral mesh, pairs of them, rigid obstacles, and
bodies meshed apart brought into one mesh."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from morphogen.contact import kernels
from morphogen.mesh.tetra import TetMesh


@dataclass
class Surface:
    """Faces of a tetrahedral mesh that may touch others: rows of three nodes (4-node
    tetrahedra) or of six (10-node ones: the corners, then the middles of the edges 0-1,
    1-2 and 2-0), their normals out of the body."""

    faces: np.ndarray

    def __post_init__(self) -> None:
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        if self.faces.ndim != 2 or self.faces.shape[1] not in (3, 6):
            raise ValueError("Surface: rows of three or six nodes")

    @classmethod
    def of(cls, mesh: TetMesh, where=None, phases=None) -> Surface:
        """The free faces of a mesh (those of one element only: its outside, the walls of
        its pores, the lips of its cracks): all of them, those of the elements of the
        ``phases`` given, those whose centroids (rows) ``where`` keeps."""
        mask = mesh._free_mask()
        if phases is not None:
            mask = mask & np.isin(mesh.phase, np.atleast_1d(phases))[:, None]
        faces = mesh._faces(mask)
        if where is not None:
            keep = np.asarray(where(mesh.nodes[faces[:, :3]].mean(axis=1)), dtype=bool)
            if keep.shape != (len(faces),):
                raise ValueError("Surface.of: where returns one value per face")
            faces = faces[keep]
        return cls(faces)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def order(self) -> int:
        """1 for faces of three nodes, 2 for faces of six."""
        return 1 if self.faces.shape[1] == 3 else 2

    def subset(self, keep) -> Surface:
        """The faces ``keep`` (mask or indices)."""
        keep = np.asarray(keep)
        return Surface(self.faces[keep if keep.dtype == bool else keep.astype(np.int64)])

    def centroids(self, nodes: np.ndarray) -> np.ndarray:
        """The centroids of the corners of the faces."""
        return nodes[self.faces[:, :3]].mean(axis=1)

    def normals(self, nodes: np.ndarray) -> np.ndarray:
        """The outward unit normals of the triangles of the corners."""
        x = nodes[self.faces[:, :3]]
        normal = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        return normal / np.linalg.norm(normal, axis=1, keepdims=True)

    def points(self, nodes: np.ndarray, rule: str = "gauss"):
        """The integration points of the faces: their faces, coordinates (ξ, η) and
        weights (their shares of the areas of the faces in the positions ``nodes``: the
        area element of each on a curved face of six nodes), three per face of three
        nodes, six per face of six. ``rule``: ``"gauss"``, points inside the faces, or
        ``"nodes"``: the vertices of the faces of three nodes; the vertices, the middles
        of the edges and the centroids of those of six (a corner or an edge of a body is
        then a contact point)."""
        if rule not in kernels.RULES:
            raise ValueError(f"contact points: one of {kernels.RULES}, got {rule!r}")
        if rule == "nodes":
            rule = kernels.NODES3 if self.order == 1 else kernels.NODES6
        else:
            rule = kernels.POINTS3 if self.order == 1 else kernels.POINTS6
        n = self.faces.shape[1]
        count = len(rule)
        face = np.repeat(np.arange(self.n_faces), count)
        xi = np.tile(rule[:, 0], self.n_faces)
        eta = np.tile(rule[:, 1], self.n_faces)
        x = nodes[self.faces]
        weight = np.empty((self.n_faces, count))
        shape, dxi, deta = np.empty(n), np.empty(n), np.empty(n)
        for g, (u, v, w) in enumerate(rule):
            kernels.shapes(n, u, v, shape, dxi, deta)
            a = np.einsum("i,eic->ec", dxi, x)
            b = np.einsum("i,eic->ec", deta, x)
            weight[:, g] = 0.5 * w * np.linalg.norm(np.cross(a, b), axis=1)
        return face, xi, eta, weight.ravel()

    def positions(self, nodes: np.ndarray, face, xi, eta) -> np.ndarray:
        """The positions of points of the faces (rows) in the positions ``nodes``."""
        n = self.faces.shape[1]
        zeta = 1.0 - xi - eta
        if n == 3:
            shape = np.stack([zeta, xi, eta], axis=1)
        else:
            shape = np.stack(
                [
                    zeta * (2 * zeta - 1),
                    xi * (2 * xi - 1),
                    eta * (2 * eta - 1),
                    4 * zeta * xi,
                    4 * xi * eta,
                    4 * eta * zeta,
                ],
                axis=1,
            )
        return np.einsum("qi,qic->qc", shape, nodes[self.faces[face]])

    def areas(self, nodes: np.ndarray) -> np.ndarray:
        """The areas of the faces in the positions ``nodes``."""
        return self.points(nodes)[3].reshape(self.n_faces, -1).sum(axis=1)

    def rims(self) -> np.ndarray:
        """Which of the edges of each face (0-1, 1-2 and 2-0 of its corners; faces x 3)
        no other face of the surface shares: the rim of the surface, none on the closed
        surface of a body."""
        corners = self.faces[:, :3]
        edges = np.sort(np.stack([corners, np.roll(corners, -1, axis=1)], axis=2), axis=2)
        _, inverse, counts = np.unique(
            edges.reshape(-1, 2), axis=0, return_inverse=True, return_counts=True
        )
        return (counts[inverse.ravel()] == 1).reshape(-1, 3)

    def size(self, nodes: np.ndarray) -> float:
        """The mean length of the edges of the triangles of the corners."""
        x = nodes[self.faces[:, :3]]
        edges = np.linalg.norm(x - np.roll(x, -1, axis=1), axis=2)
        return float(edges.mean())


def _check_contact(name: str, friction, stiffness, tangential, depth, points="gauss") -> None:
    if not (math.isfinite(friction) and friction >= 0):
        raise ValueError(f"{name}: a coefficient of friction of 0 or more")
    if points not in kernels.RULES:
        raise ValueError(f"{name}: contact points among {kernels.RULES}, got {points!r}")
    for value, what in ((stiffness, "stiffness"), (tangential, "tangential stiffness"),
                        (depth, "depth")):  # fmt: skip
        if value is not None and not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name}: a positive {what}")


@dataclass
class Pair:
    """Two surfaces that may touch: the integration points of the faces of ``slave`` are
    kept out of ``master`` (and those of ``master`` out of ``slave`` too with
    ``symmetric``: two passes, for bodies of like stiffness and meshes). ``slave`` alone
    (``master`` None): a surface that may touch itself, the walls of pores or the struts
    of a lattice (every face against the others, those that share a node with it left
    out).

    ``friction``: Coulomb's coefficient. ``stiffness``: the penalty, a pressure per unit
    of penetration (by default ten times the stiffest Young's modulus over the size of
    the faces; the two passes of a symmetric pair, or the two sides of a surface that
    touches itself, share it); ``tangential``: that of the stick (by default the
    penalty). ``depth``: the deepest penetration looked for (by default half the size of
    the faces): a point deeper behind a face is taken to be across a wall, not through
    it. ``points``: the slave points, ``"gauss"`` (inside the faces) or ``"nodes"`` (at
    their nodes: for a surface whose corners or edges meet the master first, which points
    inside its faces would let through; the weight of a point is then the share of its
    face that the master sees, the cosine of the angle of their normals, and a point of
    the rim of the slave still faces the master a twentieth of a face beyond its rim)."""

    slave: Surface
    master: Surface | None = None
    friction: float = 0.0
    stiffness: float | None = None
    tangential: float | None = None
    depth: float | None = None
    symmetric: bool = False
    points: str = "gauss"

    def __post_init__(self) -> None:
        _check_contact("Pair", self.friction, self.stiffness, self.tangential, self.depth,
                       self.points)  # fmt: skip
        if self.master is not None and self.master.order != self.slave.order:
            raise ValueError("Pair: two surfaces of the same order (three or six nodes a face)")

    @property
    def own(self) -> bool:
        """Whether the surface touches itself."""
        return self.master is None


@dataclass
class Obstacle:
    """A rigid obstacle that the integration points of the faces of ``surface`` stay out
    of, with Coulomb's ``friction`` on it; ``stiffness`` and ``tangential``: as for a
    :class:`Pair`. ``motion``: its translation, a vector grown with the load factor
    (``load factor x motion``) or a function of the load factor returning one; the force
    that the body exerts on it is a result of each step.

    The kinds (:meth:`plane`, :meth:`sphere`, :meth:`cylinder`, :meth:`cone`): a plane (a
    point and its normal, out of the obstacle: a platen), a sphere (its centre and
    radius), a cylinder without ends (a point of its axis, its axis and radius), a cone
    (its apex, its axis from the apex into the cone and its half-angle, in degrees: an
    indenter). ``inside``: the body is held within the sphere or the cylinder (a die).
    ``points``: the points of the surface kept out of it, ``"gauss"`` or ``"nodes"`` (as
    for a :class:`Pair`: a body that meets a platen by a corner or an edge)."""

    kind: int
    surface: Surface
    point: tuple
    axis: tuple = (0.0, 0.0, 1.0)
    size: float = 0.0
    inside: bool = False
    friction: float = 0.0
    stiffness: float | None = None
    tangential: float | None = None
    motion: object = None
    name: str = field(default="", compare=False)
    points: str = "gauss"

    def __post_init__(self) -> None:
        _check_contact("Obstacle", self.friction, self.stiffness, self.tangential, None,
                       self.points)  # fmt: skip
        self.point = tuple(float(v) for v in self.point)
        axis = np.asarray(self.axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if len(self.point) != 3 or axis.shape != (3,) or not norm > 0:
            raise ValueError("Obstacle: a point and an axis of three components")
        self.axis = tuple(axis / norm)
        if self.kind in (kernels.SPHERE, kernels.CYLINDER) and not self.size > 0:
            raise ValueError("Obstacle: a positive radius")
        if self.kind == kernels.CONE and not 0 < self.size < 0.5 * math.pi:
            raise ValueError("Obstacle: a half-angle of the cone between 0 and 90 degrees")
        if self.inside and self.kind not in (kernels.SPHERE, kernels.CYLINDER):
            raise ValueError("Obstacle: only a sphere or a cylinder holds a body inside")
        if not self.name:
            self.name = ("plane", "sphere", "cylinder", "cone")[self.kind]

    @classmethod
    def plane(cls, surface: Surface, point, normal, **options) -> Obstacle:
        return cls(kernels.PLANE, surface, point, normal, **options)

    @classmethod
    def sphere(cls, surface: Surface, centre, radius: float, **options) -> Obstacle:
        return cls(kernels.SPHERE, surface, centre, size=float(radius), **options)

    @classmethod
    def cylinder(cls, surface: Surface, point, axis, radius: float, **options) -> Obstacle:
        return cls(kernels.CYLINDER, surface, point, axis, float(radius), **options)

    @classmethod
    def cone(cls, surface: Surface, apex, axis, half_angle: float, **options) -> Obstacle:
        return cls(kernels.CONE, surface, apex, axis, math.radians(float(half_angle)), **options)

    def translation(self, t: float) -> np.ndarray:
        """The translation of the obstacle at the load factor ``t``."""
        if self.motion is None:
            return np.zeros(3)
        if callable(self.motion):
            out = np.asarray(self.motion(t), dtype=float)
        else:
            out = t * np.asarray(self.motion, dtype=float)
        if out.shape != (3,):
            raise ValueError("Obstacle: a motion of three components")
        return out

    def geometry(self, t: float = 0.0) -> np.ndarray:
        """The array of the kernels: the point moved to the load factor ``t``, the axis,
        the size and the sign of the side."""
        return np.array(
            [*(np.asarray(self.point) + self.translation(t)), *self.axis, self.size,
             -1.0 if self.inside else 1.0]
        )  # fmt: skip

    def distance(self, points: np.ndarray, t: float = 0.0) -> np.ndarray:
        """The signed distances of points (rows) to the obstacle at the load factor ``t``
        (positive outside it)."""
        geometry, normal = self.geometry(t), np.empty(3)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return np.array([kernels.obstacle_distance(self.kind, geometry, p, normal) for p in points])


def assemble(meshes, phases=None) -> TetMesh:
    """Bodies meshed apart as one mesh, for their contact: the nodes and elements of the
    ``meshes`` one after the other (nothing is merged: nodes at the same place stay
    apart), in the box of them all; ``phases``: a phase per mesh in place of their own
    (by default each mesh keeps its phases)."""
    meshes = list(meshes)
    if not meshes:
        raise ValueError("assemble: at least one mesh")
    if len({m.order for m in meshes}) != 1:
        raise ValueError("assemble: meshes of the same order (4- or 10-node tetrahedra)")
    if phases is not None and len(phases) != len(meshes):
        raise ValueError("assemble: a phase per mesh")
    nodes, elements, phase, offset = [], [], [], 0
    for k, m in enumerate(meshes):
        nodes.append(m.nodes)
        elements.append(m.elements + offset)
        phase.append(m.phase if phases is None else np.full(m.n_elements, int(phases[k])))
        offset += m.n_nodes
    nodes = np.concatenate(nodes)
    return TetMesh(
        nodes,
        np.concatenate(elements),
        np.concatenate(phase),
        nodes.min(axis=0),
        nodes.max(axis=0),
        periodic=False,
        metadata={"bodies": [m.n_nodes for m in meshes]},
    )
