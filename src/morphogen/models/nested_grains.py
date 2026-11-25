"""Grains with subgrains, or another structure inside the grains of a polycrystal.

``nested_grains`` nests a structure in the grains of a tessellation (``grains``: a Voronoi
or Laguerre tessellation, isotropic or not, with its phases, the crystal orientations of
its grains and its grain-boundary phase):

* an ``inner`` tessellation divides the grains into subgrains. The germs of the inner
  tessellation that fall in a grain are the germs of its subgrains, and each point of the
  grain belongs to the nearest of them in the power distance of the inner tessellation:
  each subgrain holds its germ (but for Laguerre cells of overlapping spheres) and the
  sub-boundaries stop at the grain boundaries, as the walls of dislocations that recovery
  builds within each grain (Humphreys & Hatherly 2004); a grain that holds no germ is a
  single subgrain. Each subgrain is the crystal of its grain turned by an angle drawn from
  ``misorientation`` about a uniformly random axis, ``g_s = g ΔR``: low-angle boundaries
  between the subgrains of a grain, the grain boundaries keep the disorientations of the
  grains. The labels are those of the subgrains, grain after grain (``parents``: the grain
  of each subgrain); the phases are those of the grains.
* any other ``inner`` model fills the grains: its phase 1 becomes a phase of the grains
  (``inner_phase``, by default the next free phase), its phase 0 leaves them as they are —
  lamellar colonies, precipitates in the grains, a two-phase structure turned grain by
  grain. The labels are those of the grains, with their orientations.

With ``turned`` (the default), each grain sees the inner structure in the axes of its
crystal: a point ``x`` of grain ``i`` sees the point ``q = g_i (x - c_i) + s_i`` of the
inner realization (``crystal = g @ sample``, ``c_i`` the germ of the grain, ``x - c_i``
the shortest periodic image — of the images of a point, the one in the whole cell of the
grain is the nearest to its germ in the power distance of the cells —, ``s_i`` a shift
uniform in the box): lamellae lie on planes of the crystal, laths of an anisotropic inner
tessellation along a direction of the crystal, and the pattern changes from grain to grain
(grains without crystal orientations are turned at random). The grains share one
realization of the inner model in a periodic box (the domain), each seeing it at its own
shift and rotation: places of the box drawn independently, so that the patterns of two
grains are independent as long as the grains are small compared with the box. Without
``turned``, one realization in the axes of the sample is cut by the grains (the germs of
an inner tessellation belong to the grain they fall in).

The phases and labels are computed at any point (k-d trees of the germs and the point
queries of the inner structure), so that the voxel images, sections and slabs of any grid
are exact. The functions of the phases are those of the grains for subgrains; for filled
grains, the union over the grains of the inner structure that each sees, cut by its cell,
at any point (the signed distances of the voxels of each phase when the inner structure
gives its functions on grids only).

The exact meshes (:func:`morphogen.mesh.exact.exact_mesh`) follow both: the subgrains are
exact convex pieces (:meth:`NestedGrainStructure.subgrain_pieces`: the pieces of the cells
of the grains cut by the planes of equal power distance to two germs of the inner
tessellation, affine in the frame of each grain, and by those of the grain-boundary
phase), meshed with their faces, each element with its subgrain and its crystal; filled
grains are meshed by the functions of their phases, each element with the grain of its
centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.spatial import cKDTree

from morphogen import settings
from morphogen.distributions import Constant, Distribution
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.affine import AffineStructure
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.models.implicit import band_of
from morphogen.models.nested import _child_seeds, _edt, _supersampled
from morphogen.models.tessellation import Tessellation, _TessellationModel
from morphogen.orientations import (
    euler_to_matrix,
    matrix_to_euler,
    random_orientations,
    uniform_unit_vectors,
)
from morphogen.seeding import SeedLike, make_rng

#: Points processed at once (bounds the memory of large grids)
_CHUNK = 1 << 21
#: Resolution (cells along the longest edge) of the voxels that stand for an inner
#: structure without point queries, when no grid gives one
_INNER_RESOLUTION = 256


def _turns(axes, angles) -> np.ndarray:
    """The rotations ``(n, 3, 3)`` of ``angles`` (radians) about ``axes`` ``(n, 3)``
    (Rodrigues' formula)."""
    u = np.asarray(axes, dtype=float).reshape(-1, 3)
    u = u / np.linalg.norm(u, axis=1)[:, None]
    t = np.asarray(angles, dtype=float).reshape(-1)
    k = np.zeros((len(u), 3, 3))
    k[:, 0, 1], k[:, 0, 2] = -u[:, 2], u[:, 1]
    k[:, 1, 0], k[:, 1, 2] = u[:, 2], -u[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -u[:, 1], u[:, 0]
    return np.eye(3) + np.sin(t)[:, None, None] * k + (1.0 - np.cos(t))[:, None, None] * (k @ k)


class _Cells:
    """The cells of a tessellation, or of its image by an affine map
    (:class:`~morphogen.models.affine.AffineStructure`), in the axes of the sample: the
    cell of points, the germs, the vertices of the cells."""

    def __init__(self, micro: Microstructure):
        if isinstance(micro, AffineStructure) and isinstance(micro.inner, Tessellation):
            self.tessellation, self.matrix = micro.inner, np.asarray(micro.matrix, dtype=float)
        elif isinstance(micro, Tessellation):
            self.tessellation, self.matrix = micro, None
        else:
            raise ValueError(f"{type(micro).__name__}: not the cells of a tessellation")
        self.inverse = None if self.matrix is None else np.linalg.inv(self.matrix)

    @property
    def n_cells(self) -> int:
        return self.tessellation.n_cells

    def preimage(self, points) -> np.ndarray:
        """The points in the space of the isotropic tessellation."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        return p if self.inverse is None else p @ self.inverse.T

    def cell_of(self, points) -> np.ndarray:
        return self.tessellation.cell_of(self.preimage(points))

    @property
    def germs(self) -> np.ndarray:
        g = self.tessellation.germs
        return g if self.matrix is None else g @ self.matrix.T

    @property
    def weights(self) -> np.ndarray | None:
        return self.tessellation.weights

    def vertices(self) -> list[np.ndarray | None]:
        """The vertices of each cell (the whole cells of a periodic tessellation, around
        their germs; else the pieces of the cells in the box), None for the cells without
        a piece."""
        tess = self.tessellation
        cx = tess.cell_complex(whole=tess.domain.periodic)
        loops: list[list[np.ndarray]] = [[] for _ in range(tess.n_cells)]
        for piece, cell in enumerate(cx.piece_cell):
            loops[int(cell)].extend(cx.facets[f] for f in cx.piece_facets[piece])
        out: list[np.ndarray | None] = []
        for cell_loops in loops:
            if not cell_loops:
                out.append(None)
                continue
            v = cx.vertices[np.unique(np.concatenate(cell_loops))]
            out.append(v if self.matrix is None else v @ self.matrix.T)
        return out


def _wrap(d: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """The shortest periodic images of the vectors ``d``."""
    return d - lengths * np.round(d / lengths)


# -- exact cells: convex polyhedra ``(vertices, face offsets, labels, normals, planes)``,
# their faces counter-clockwise about their outward unit normals ``n . x = c`` (those of
# :func:`morphogen.mesh.exact.pieces_mesh`) ------------------------------------------------

#: The label of the faces of the pieces (what lies beyond them: unused by the meshes)
_FACE = -1


def _polyhedron(loops) -> tuple:
    """A convex polyhedron from the loops of its faces, counter-clockwise about their
    outward normals."""
    normals, planes = [], []
    for loop in loops:
        area = 0.5 * np.cross(loop, np.roll(loop, -1, axis=0)).sum(axis=0)
        n = area / np.linalg.norm(area)
        normals.append(n)
        planes.append(float(n @ loop.mean(axis=0)))
    offsets = np.concatenate([[0], np.cumsum([len(loop) for loop in loops])])
    return (
        np.ascontiguousarray(np.concatenate(loops), dtype=float),
        offsets.astype(np.int64),
        np.full(len(loops), _FACE, dtype=np.int64),
        np.asarray(normals, dtype=float),
        np.asarray(planes, dtype=float),
    )


def _piece_loops(complex_, piece: int, faces=None) -> list[np.ndarray]:
    """The loops of the facets of a piece of a cell complex (``faces``: some of them),
    counter-clockwise about their normals out of the piece."""
    loops = []
    for f in complex_.piece_facets[piece] if faces is None else faces:
        loop = complex_.vertices[complex_.facets[f]]
        loops.append(loop if complex_.facet_pieces[f, 0] == piece else loop[::-1])
    return loops


def _mapped(poly, matrix: np.ndarray) -> tuple:
    """The image of a convex polyhedron by ``x = A y`` (its normals by ``A^-T``)."""
    verts, offsets, labels, normals, _ = poly
    v = np.ascontiguousarray(verts @ matrix.T)
    n = normals @ np.linalg.inv(matrix)
    n /= np.linalg.norm(n, axis=1)[:, None]
    return v, offsets, labels, np.ascontiguousarray(n), np.einsum("ij,ij->i", n, v[offsets[:-1]])


def _clipped(poly, normals, offsets, tol: float):
    """The part of a convex polyhedron where ``n_k . x <= c_k`` for every half-space (unit
    normals), None when nothing of it is left. A plane within ``tol`` of every vertex leaves
    the polyhedron on the side it lies on: two pieces cut by one plane from its two sides
    share its trace."""
    from morphogen.models.scale_rich_3d import clip_polyhedron

    v, o, lab, nrm, pln = poly
    for n, c in zip(np.asarray(normals, dtype=float), np.asarray(offsets, dtype=float),
                    strict=True):  # fmt: skip
        d = v @ n - c
        if d.max() <= tol:
            continue  # wholly inside
        if d.min() >= -tol:
            return None  # wholly outside, or touching the plane only
        v, o, lab, nrm, pln = clip_polyhedron(v, o, lab, nrm, pln, n[0], n[1], n[2], float(c),
                                              _FACE, tol)  # fmt: skip
        if len(o) < 5:  # fewer than four faces: nothing left
            return None
    return v, o, lab, nrm, pln


@register_kind
class NestedGrainStructure(Microstructure):
    """The grains of a tessellation with subgrains, or filled with another structure (see
    the module): the realization of the grains, that of the inner model, the frame in
    which each grain sees it, and the subgrains (their grains, germs and orientations)."""

    kind = "nested_grains"
    has_labels = True

    def __init__(
        self,
        domain: Domain,
        grains: Microstructure,
        inner: Microstructure,
        *,
        turned: bool,
        centres,
        filled_phases,
        rotations=None,
        shifts=None,
        inner_phase: int | None = None,
        parents=None,
        subgrain_germs=None,
        subgrain_weights=None,
        orientations=None,
        misorientations=None,
        **kwargs,
    ) -> None:
        super().__init__(domain, **kwargs)
        self.grains = grains
        self.inner = inner
        self.cells = _Cells(grains)
        self.turned = bool(turned)
        n = self.cells.n_cells
        self.centres = np.asarray(centres, dtype=float).reshape(n, 3)
        self.filled_phases = sorted(int(k) for k in filled_phases)
        if self.turned:
            self.rotations = np.asarray(rotations, dtype=float).reshape(n, 3)
            self.matrices = euler_to_matrix(self.rotations)
            self.shifts = np.asarray(shifts, dtype=float).reshape(n, 3)
        else:
            self.rotations = None
            self.matrices = np.broadcast_to(np.eye(3), (n, 3, 3))
            self.shifts = self.centres
        self.inner_phase = None if inner_phase is None else int(inner_phase)
        self.subgrains = parents is not None
        if self.subgrains:
            self.parents = np.asarray(parents, dtype=np.int64).reshape(-1)
            if np.any(np.diff(self.parents) < 0) or np.unique(self.parents).size != n:
                raise ValueError("parents: the grain of each subgrain, grain after grain")
            m = len(self.parents)
            self.first = np.searchsorted(self.parents, np.arange(n + 1))
            self.subgrain_germs = np.asarray(subgrain_germs, dtype=float).reshape(m, 3)
            self.subgrain_weights = (
                None
                if subgrain_weights is None
                else np.asarray(subgrain_weights, dtype=float).reshape(m)
            )
            self._orientations = np.asarray(orientations, dtype=float).reshape(m, 3)
            self.misorientations = (
                np.zeros(m)
                if misorientations is None
                else np.asarray(misorientations, dtype=float).reshape(m)
            )
            if self.turned and not inner.domain.periodic:
                raise ValueError("turned subgrains: the inner tessellation is periodic")
            inner_cells = _Cells(inner)
            self._metric = inner_cells.inverse  # the metric of the inner power distance
        else:
            self.parents = np.arange(n, dtype=np.int64)
            self.first = np.arange(n + 1)
        self._trees: dict[int, tuple] = {}
        self._inner_voxels: dict = {}
        self._filled_cells = np.isin(self.cells.tessellation.cell_phases, self.filled_phases)
        self._pieces: tuple | None = None
        self._planes: list | None = None
        self._bridge_faces: list | None = None
        self._point_inner: bool | None = None
        # anisotropic grains of a window whose preimage is no box: their cells, mapped, are
        # cut by the faces of the domain
        self._window = isinstance(grains, AffineStructure) and grains.box_stretch is None

    # -- the grains, their subgrains and their crystals -----------------------------------------

    @property
    def n_grains(self) -> int:
        return self.cells.n_cells

    @property
    def n_cells(self) -> int:
        """The number of labels, those of the subgrains or of the grains."""
        return len(self.parents)

    @property
    def orientations(self) -> np.ndarray | None:
        """The crystal orientations of the labels (Bunge angles, degrees): of the subgrains,
        or of the grains (None if the grains have none)."""
        if self.subgrains:
            return self._orientations
        return getattr(self.grains, "orientations", None)

    @property
    def cell_phases(self) -> np.ndarray | None:
        """The phase of each subgrain (that of its grain); None for filled grains, whose
        labels hold several phases."""
        if not self.subgrains:
            return None
        return np.asarray(self.cells.tessellation.cell_phases)[self.parents]

    @property
    def germs(self) -> np.ndarray:
        """A point of each label in the axes of the sample: the germ of each subgrain (in
        the box, for a periodic domain), or the germ of each grain."""
        if not self.subgrains:
            return self.cells.germs
        g = self.parents
        local = np.einsum("nji,nj->ni", self.matrices[g], self.subgrain_germs - self.shifts[g])
        x = self.centres[g] + local
        if self.domain.periodic:
            x = self.domain.wrap(x)
        return x

    def _local(self, points, grains):
        """The points seen by their grains: ``q = g (x - c) + s``, ``x - c`` the shortest
        periodic image."""
        d = np.asarray(points, dtype=float).reshape(-1, 3) - self.centres[grains]
        if self.domain.periodic:
            d = _wrap(d, self.domain.lengths)
        return np.einsum("nij,nj->ni", self.matrices[grains], d) + self.shifts[grains]

    def _tree(self, grain: int):
        """The k-d tree of the germs of the subgrains of a grain, lifted by their weights,
        in the space of the inner tessellation."""
        if grain not in self._trees:
            k0, k1 = int(self.first[grain]), int(self.first[grain + 1])
            data = self.subgrain_germs[k0:k1]
            if self._metric is not None:
                data = data @ self._metric.T
            lifted = self.subgrain_weights is not None
            if lifted:
                w = self.subgrain_weights[k0:k1]
                data = np.column_stack([data, np.sqrt(np.maximum(w.max() - w, 0.0))])
            self._trees[grain] = (cKDTree(data), lifted)
        return self._trees[grain]

    def _subgrains_of(self, points, grains) -> np.ndarray:
        """The subgrain of each point, given its grain."""
        out = self.first[grains].astype(np.int64)
        several = (self.first[grains + 1] - self.first[grains]) > 1
        if not several.any():
            return out
        index = np.flatnonzero(several)
        q = self._local(np.asarray(points, dtype=float).reshape(-1, 3)[index], grains[index])
        if self._metric is not None:
            q = q @ self._metric.T
        order = np.argsort(grains[index], kind="stable")
        sorted_grains = grains[index][order]
        cuts = np.flatnonzero(np.diff(sorted_grains)) + 1
        for chunk in np.split(order, cuts):
            grain = int(grains[index[chunk[0]]])
            tree, lifted = self._tree(grain)
            z = q[chunk]
            if lifted:
                z = np.column_stack([z, np.zeros(len(z))])
            _, nearest = tree.query(z, k=1, workers=settings.workers())
            out[index[chunk]] += nearest
        return out

    def cell_of(self, points) -> np.ndarray:
        """The label of each point: its subgrain, or its grain."""
        grains = self.cells.cell_of(points)
        if not self.subgrains:
            return grains
        return self._subgrains_of(points, grains)

    # -- the exact cells of the subgrains ---------------------------------------------------

    @property
    def convex_pieces(self):
        """Subgrains: the method that gives their exact cells as convex pieces
        (:meth:`subgrain_pieces`), for the exact meshes of
        :func:`morphogen.mesh.exact.polyhedral_mesh`; None for filled grains (their
        phases are curved: :meth:`phase_level_sets_at`)."""
        return self.subgrain_pieces if self.subgrains else None

    @property
    def point_cells(self) -> bool:
        """Filled grains: the exact meshes cut the phases by their point-wise functions and
        give each element the label of its centre (:func:`morphogen.mesh.exact.exact_mesh`);
        the faces between two grains of one phase are not followed."""
        return not self.subgrains

    def subgrain_pieces(self) -> tuple[list[tuple], np.ndarray, np.ndarray]:
        """The subgrains as exact convex pieces that partition the box: the pieces of the
        cells of the grains (their parts in the box, :meth:`Tessellation.cell_complex`,
        mapped by an anisotropic tessellation), each cut by the planes where the power
        distances of the inner tessellation to the germs of two of its subgrains are equal
        (affine in ``x``: ``q = g (x - c - t) + s``, ``t`` the periodic image of the grain
        the piece belongs to), and by the planes of the grain-boundary phase (the cell
        shrunk by half its thickness along the faces it shares with other cells, and the
        slabs between). Returns the pieces (``(vertices, face offsets, labels, normals,
        planes)``), their phases and their labels (the subgrains)."""
        if not self.subgrains:
            raise ValueError("filled grains: no convex pieces (their phases are curved)")
        if self._pieces is not None:
            return self._pieces
        from morphogen.models.scale_rich_3d import polyhedron_volume

        tess = self.cells.tessellation
        lengths = self.domain.lengths
        tol = 1e-9 * float(np.max(lengths))
        complex_ = tess.cell_complex()
        cell_phases = np.asarray(tess.cell_phases, dtype=np.int64)
        thickness = float(tess.boundary_thickness)
        planes = self._cell_planes() if thickness > 0 else None
        box = None
        if self._window:
            lo, hi = self.domain.lower, self.domain.upper
            box = (np.vstack([np.eye(3), -np.eye(3)]), np.concatenate([hi, -lo]))
        pieces, phases, labels = [], [], []
        for p, grain in enumerate(complex_.piece_cell):
            grain = int(grain)
            whole = _polyhedron(_piece_loops(complex_, p))
            parts = [(whole, int(cell_phases[grain]))]
            if planes is not None:
                parts = self._boundary_parts(whole, planes[grain], int(cell_phases[grain]),
                                             thickness, tol)  # fmt: skip
            k0, k1 = int(self.first[grain]), int(self.first[grain + 1])
            for part, phase in parts:
                if self.cells.matrix is not None:
                    part = _mapped(part, self.cells.matrix)
                    if box is not None:
                        part = _clipped(part, *box, tol)
                        if part is None:
                            continue
                if k1 - k0 == 1:
                    pieces.append(part)
                    phases.append(phase)
                    labels.append(k0)
                    continue
                t = np.zeros(3)
                if self.domain.periodic:  # the image of the grain this piece belongs to
                    d = part[0].mean(axis=0) - self.centres[grain]
                    t = lengths * np.round(d / lengths)
                for j, (n, c) in enumerate(self._subgrain_halfspaces(grain, t)):
                    sub = _clipped(part, n, c, tol)
                    if sub is None or not polyhedron_volume(sub[0], sub[1], sub[3], sub[4]) > 0:
                        continue
                    pieces.append(sub)
                    phases.append(phase)
                    labels.append(k0 + j)
        self._pieces = (pieces, np.asarray(phases, dtype=np.int64),
                        np.asarray(labels, dtype=np.int64))  # fmt: skip
        return self._pieces

    def _subgrain_halfspaces(self, grain: int, t) -> list[tuple[np.ndarray, np.ndarray]]:
        """For each subgrain of a grain, the half-spaces ``n . x <= c`` of the sample (unit
        normals) where the power distance ``|M (q - y_j)|^2 - w_j`` of the inner
        tessellation to its germ is at most that to every other germ of the grain, for the
        points ``x`` of the image of the grain translated by ``t``: ``q = g (x - c - t) +
        s``, so that ``2 M^T M (y_k - y_j) . q <= P_k - P_j`` (``P = |M y|^2 - w``) is a
        half-space in ``x``."""
        k0, k1 = int(self.first[grain]), int(self.first[grain + 1])
        y = self.subgrain_germs[k0:k1]
        my = y if self._metric is None else y @ self._metric.T
        w = np.zeros(k1 - k0) if self.subgrain_weights is None else self.subgrain_weights[k0:k1]
        power = np.einsum("ij,ij->i", my, my) - w
        rotation, shift = self.matrices[grain], self.shifts[grain]
        origin = self.centres[grain] + np.asarray(t, dtype=float)
        out = []
        for j in range(k1 - k0):
            others = np.arange(k1 - k0) != j
            v = my[others] - my[j]
            a = 2.0 * (v if self._metric is None else v @ self._metric)  # 2 M^T v, in q
            n = a @ rotation  # g^T a: the normal in x
            c = power[others] - power[j] - a @ shift + n @ origin
            norm = np.linalg.norm(n, axis=1)
            keep = norm > 0.0  # (two germs at one place: one cell)
            out.append((n[keep] / norm[keep, None], c[keep] / norm[keep]))
        return out

    def _cell_planes(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """The faces that each grain shares with other cells, in the space of the
        tessellation: their outward unit normals and offsets, its germ (in the frame of the
        cell: the periodic image of the points is the one nearest to it, as for
        :meth:`_local`) and the farthest of its vertices from it — the whole cells of a
        periodic tessellation, about which its pieces in the box are translates; the cells
        in the box of a window."""
        if self._planes is not None:
            return self._planes
        tess = self.cells.tessellation
        empty = (np.zeros((0, 3)), np.zeros(0), np.zeros(3), 0.0)
        out = [empty] * tess.n_cells
        periodic = tess.domain.periodic
        complex_ = tess.cell_complex(whole=True) if periodic else tess.cell_complex()
        germs = np.asarray(tess.germs, dtype=float)
        box = tess.domain.lengths
        for p, grain in enumerate(complex_.piece_cell):
            vertices = complex_.vertices[np.unique(np.concatenate(
                [complex_.facets[f] for f in complex_.piece_facets[p]]))]  # fmt: skip
            centre = germs[int(grain)]
            if periodic:  # the image of the germ in the frame of the whole cell
                middle = vertices.mean(axis=0)
                centre = middle - _wrap(middle - centre, box)
            radius = float(np.max(np.linalg.norm(vertices - centre, axis=1)))
            faces = [f for f in complex_.piece_facets[p] if periodic or complex_.facet_box[f] == 0]
            if not faces:  # a single cell: no face shared
                out[int(grain)] = (np.zeros((0, 3)), np.zeros(0), centre, radius)
                continue
            whole = _polyhedron(_piece_loops(complex_, p, faces))
            out[int(grain)] = (whole[3], whole[4], centre, radius)
        self._planes = out
        return out

    def _boundary_parts(self, poly, planes, phase: int, thickness: float, tol: float) -> list:
        """A piece of a cell (the space of the tessellation) divided by its grain-boundary
        phase, at the distance ``thickness / 2`` of the faces the cell shares with others
        (the distance to the nearest face of a convex cell is that to its plane): the cell
        shrunk by it (the phase of the cell) and the disjoint slabs between, each convex (the
        part beyond the shifted plane of a face and within those of the faces before it;
        the boundary phase)."""
        normals, offsets, centre, _ = planes
        if not len(offsets):
            return [(poly, phase)]
        tess = self.cells.tessellation
        t = np.zeros(3)
        if tess.domain.periodic:  # the translate of the whole cell that holds this piece
            box = tess.domain.lengths
            t = box * np.round((poly[0].mean(axis=0) - centre) / box)
        shifted = offsets + normals @ t - 0.5 * thickness
        parts = []
        core = _clipped(poly, normals, shifted, tol)
        if core is not None:
            parts.append((core, phase))
        for k in range(len(shifted)):
            slab = _clipped(poly, np.vstack([-normals[k : k + 1], normals[:k]]),
                            np.concatenate([[-shifted[k]], shifted[:k]]), tol)  # fmt: skip
            if slab is not None:
                parts.append((slab, int(tess.boundary_phase)))
        return parts

    # -- phases -----------------------------------------------------------------------------

    def _grain_phases(self, points, grains) -> tuple[np.ndarray, np.ndarray | None]:
        """The phases of the grains at the points, and where their boundary phase is."""
        tess = self.cells.tessellation
        phases = np.asarray(tess.cell_phases, dtype=np.uint8)[grains]
        boundary = None
        if tess.boundary_thickness > 0:
            half = 0.5 * tess.boundary_thickness
            boundary = tess.boundary_distance(self.cells.preimage(points), half) < half
            phases[boundary] = tess.boundary_phase
        return phases, boundary

    def _inner_contains(self, q, spacing=None) -> np.ndarray:
        """Whether the points of the inner realization are in its phase 1: its point
        queries, else the nearest voxel of an image of it (as fine as ``spacing``)."""
        box = self.inner.domain
        if box.periodic:
            q = box.wrap(q)
        try:
            return np.asarray(self.inner.contains(q), dtype=bool).reshape(-1)
        except NotImplementedError:
            pass
        h = float(np.min(spacing)) if spacing is not None else None
        if h is None:
            h = float(np.max(box.lengths)) / _INNER_RESOLUTION
        if h not in self._inner_voxels:
            self._inner_voxels[h] = self.inner.voxelize(voxel_size=h)
        image = self._inner_voxels[h]
        index = np.floor((q - np.asarray(image.origin)) / np.asarray(image.spacing)).astype(int)
        n = np.asarray(image.shape)
        index = np.mod(index, n) if box.periodic else np.clip(index, 0, n - 1)
        return image.data[index[:, 0], index[:, 1], index[:, 2]] == 1

    def phases_at(self, points, spacing=None) -> np.ndarray:
        """The phase at each point (uint8): that of its grain, its boundary phase, or the
        inner phase where the inner structure fills the grain."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        grains = self.cells.cell_of(pts)
        phases, boundary = self._grain_phases(pts, grains)
        if self.subgrains or not len(pts):
            return phases
        filled = self._filled_cells[grains]
        if boundary is not None:
            filled &= ~boundary
        index = np.flatnonzero(filled)
        if index.size:
            q = self._local(pts[index], grains[index])
            inside = self._inner_contains(q, spacing)
            phases[index[inside]] = self.inner_phase
        return phases

    def contains(self, points) -> np.ndarray:
        return self.phases_at(points) == 1

    # -- discretization ---------------------------------------------------------------------

    def _on_grid(self, grid: Grid, values, dtype) -> np.ndarray:
        """``values(points)`` at the centres of the voxels of ``grid``, slab by slab."""
        nx, ny, nz = grid.shape
        out = np.empty(grid.shape, dtype=dtype)
        k_step = max(1, _CHUNK // (nx * ny))
        for k0 in range(0, nz, k_step):
            k1 = min(nz, k0 + k_step)
            out[:, :, k0:k1] = values(grid.center_points(k0, k1)).reshape(nx, ny, k1 - k0)
        return out

    def _phases_on(self, grid: Grid) -> np.ndarray:
        return self._on_grid(grid, lambda p: self.phases_at(p, grid.spacing), np.uint8)

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if ss == 1:
            data = self._phases_on(grid)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        data = _supersampled((self._phases_on(fine) == 1).astype(np.float32), grid, ss)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def labels(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Image of the labels (int32): of the subgrains, grain after grain, or of the
        grains."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        data = self._on_grid(grid, self.cell_of, np.int32)
        return VoxelImage.from_grid(data, grid, name="cell", metadata=self.image_metadata())

    def grain_labels(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Image of the labels of the grains (int32), the parents of the subgrains."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        data = self._on_grid(grid, self.cells.cell_of, np.int32)
        return VoxelImage.from_grid(data, grid, name="grain", metadata=self.image_metadata())

    # -- implicit functions -----------------------------------------------------------------

    def implicit_phases(self) -> list[int]:
        phases = set(self.grains.implicit_phases())
        if not self.subgrains:
            phases.add(self.inner_phase)
        return sorted(phases)

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """That of the grains (subgrains keep their phases); for filled grains, the
        function of phase 1 (:meth:`phase_level_set`)."""
        if self.subgrains:
            return self.grains.level_set(shape, voxel_size=voxel_size, grid=grid)
        if 1 in self.implicit_phases() and len(self.implicit_phases()) > 1:
            return self.phase_level_set(1, shape, voxel_size=voxel_size, grid=grid)
        return super().level_set(shape, voxel_size=voxel_size, grid=grid)

    def phase_level_set(
        self, phase: int, shape=None, *, voxel_size=None, grid: Grid | None = None
    ) -> VoxelImage:
        """The implicit function of a phase (see :meth:`Microstructure.phase_level_set`):
        that of the grains for subgrains; for filled grains, the point-wise functions of
        :meth:`phase_level_sets_at` at the centres of the voxels, clipped to the band of
        the grid, or, when the inner structure has no point-wise functions, the signed
        distance from the centres of the voxels of the phase to those of the others, less
        half a voxel (the interfaces between the voxels)."""
        phase = int(phase)
        if phase not in self.implicit_phases():
            raise ValueError(f"phase {phase}: the phases are {self.implicit_phases()}")
        if self.subgrains:
            return self.grains.phase_level_set(phase, shape, voxel_size=voxel_size, grid=grid)
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        band = band_of(grid)
        if self._inner_functions():
            data = self._on_grid(grid, lambda p: self.phase_level_sets_at(p, band)[phase],
                                 np.float32)  # fmt: skip
            data = np.clip(data, -band, band).astype(np.float32)
            return VoxelImage.from_grid(data, grid, name="level_set",
                                        metadata=self.image_metadata())  # fmt: skip
        inside = self._phases_on(grid) == phase
        mode = "wrap" if self.domain.periodic else "nearest"
        h = grid.spacing
        half = 0.5 * float(np.min(h))
        outside_d = _edt(~inside, h, mode, band + half)
        inside_d = _edt(inside, h, mode, band + half)
        data = np.where(inside, -(inside_d - half), outside_d - half)
        data = np.clip(data, -band, band).astype(np.float32)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def _inner_functions(self) -> bool:
        """Whether the inner structure gives the function of its phase 1 at any point."""
        from morphogen.models.implicit import point_functions

        if self._point_inner is None:
            self._point_inner = point_functions(self.inner)
        return self._point_inner

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """That of the grains, for subgrains; for filled grains, the function of their phase
        1 (:meth:`phase_level_sets_at`)."""
        if self.subgrains:
            return self.grains.level_set_at(points, reach)
        functions = self.phase_level_sets_at(points, reach)
        if 1 not in functions:
            raise NotImplementedError("filled grains without a phase 1: no level set")
        return functions[1]

    def phase_level_sets_at(self, points, reach: float = np.inf) -> dict[int, np.ndarray]:
        """The functions of the phases at points: those of the grains, for subgrains, which
        keep their phases. Filled grains: the inner phase is the union over the filled
        grains ``B`` of the inner structure seen by ``B`` and cut by its cell,
        ``F = min_B max(c_B, phi(q_B))`` (:meth:`_inner_union`: ``c_B`` the function of the
        cell, shrunk by half the thickness of a grain-boundary phase, ``phi`` that of the
        phase 1 of the inner structure at the point ``q_B = g_B (x - c_B) + s_B`` that
        ``B`` sees, a rigid motion of ``x``: distances are kept), continuous across the
        faces of the grains, whose patterns differ; without turns, one realization in the
        axes of the sample, ``F = max(D, phi(x))``, ``D = min_k G_k`` over the filled phases
        (negative across the faces between filled cells). A filled phase ``k`` of the
        grains, of function ``G_k``, is ``max(G_k, -F)``, the others ``G_k``.
        NotImplementedError for an inner structure whose functions are those of grids
        only."""
        if self.subgrains:
            return self.grains.phase_level_sets_at(points, reach)
        from morphogen.models.implicit import functions_at

        if not self._inner_functions():
            raise NotImplementedError(
                f"filled grains: {type(self.inner).__name__} gives the functions of its phases "
                "on grids only"
            )
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        reach = float(reach)
        far = reach if np.isfinite(reach) else 2.0 * float(np.max(self.domain.lengths))
        try:
            grains = functions_at(self.grains, pts, reach)
        except NotImplementedError:  # a tessellation of one phase: no interface
            phases = self.grains.implicit_phases()
            if len(phases) != 1:
                raise
            grains = {phases[0]: np.full(len(pts), -far)}
        filled = [k for k in self.filled_phases if k in grains]
        if self.turned:
            inner = self._inner_union(pts, reach, far)
        else:  # one realization for all the grains: max(D, phi), D the filled cells
            cells = (np.minimum.reduce([np.asarray(grains[k], dtype=float) for k in filled])
                     if filled else np.full(len(pts), far))  # fmt: skip
            box = self.inner.domain
            q = box.wrap(pts) if box.periodic else pts
            phi = np.asarray(functions_at(self.inner, q, reach)[1], dtype=float)
            inner = np.clip(np.maximum(cells, phi), -far, far)
        out = {}
        for k, g in grains.items():
            g = np.asarray(g, dtype=float)
            out[int(k)] = np.maximum(g, -inner) if k in filled else g
        if self.inner_phase in out:  # the inner phase joins that phase of the grains
            out[self.inner_phase] = np.minimum(out[self.inner_phase], inner)
        else:
            out[self.inner_phase] = inner
        return {k: np.clip(v, -reach, reach) for k, v in out.items()}

    def _faces_between_filled(self) -> list[tuple]:
        """The faces shared by two filled grains, in the box of the tessellation: their
        centre, the farthest of their vertices from it, their plane (unit normal from the
        first grain to the second, offset), the planes of the other faces of the two cells
        (the pieces of the whole cells at that place) and the two grains."""
        if self._bridge_faces is not None:
            return self._bridge_faces
        tess = self.cells.tessellation
        complex_ = tess.cell_complex()
        planes = self._cell_planes()
        box = tess.domain.lengths
        tol = 1e-9 * float(np.max(box))
        means = {}
        out = []
        for f, (pa, pb) in enumerate(complex_.facet_pieces):
            if pb < 0:
                continue
            ca, cb = int(complex_.piece_cell[pa]), int(complex_.piece_cell[pb])
            if ca == cb or not (self._filled_cells[ca] and self._filled_cells[cb]):
                continue
            loop = complex_.vertices[complex_.facets[f]]
            area = 0.5 * np.cross(loop, np.roll(loop, -1, axis=0)).sum(axis=0)
            normal = area / np.linalg.norm(area)
            centre = loop.mean(axis=0)
            offset = float(normal @ centre)
            sides = []
            for piece, grain, sign in ((pa, ca, 1.0), (pb, cb, -1.0)):
                normals, offsets, anchor, _ = planes[grain]
                shifted = offsets.copy()
                if tess.domain.periodic:  # the planes of the whole cell at this piece
                    if piece not in means:
                        means[piece] = complex_.vertices[np.unique(np.concatenate(
                            [complex_.facets[g] for g in complex_.piece_facets[piece]]))
                        ].mean(axis=0)  # fmt: skip
                    shifted = shifted + normals @ (box * np.round((means[piece] - anchor) / box))
                same = (np.linalg.norm(normals - sign * normal, axis=1) < 1e-6) & (
                    np.abs(shifted - sign * offset) < tol * 1e3
                )
                sides.append((normals[~same], shifted[~same]))
            radius = float(np.max(np.linalg.norm(loop - centre, axis=1)))
            out.append((centre, radius, normal, offset, sides, ca, cb))
        self._bridge_faces = out
        return out

    def _inner_union(self, pts: np.ndarray, reach: float, far: float) -> np.ndarray:
        """``F = min_B max(c_B, phi(q_B))`` over the filled grains ``B`` within ``reach`` of
        the points (see :meth:`phase_level_sets_at`): ``c_B`` the largest of the signed
        distances to the planes of the faces that ``B`` shares with other cells (in the
        space of the tessellation; exact inside the convex cell, a lower bound of the
        distance outside), shifted by half the thickness of a grain-boundary phase; the
        points of a periodic box taken at the image nearest to the cell. Two grains that
        share a face would leave ``F`` zero on it where both see the inner phase (each
        function vanishes on the boundary of its cell): a bridge over each such face,
        ``max(|n . y - o| - e, a, b, phi(q_A), phi(q_B))`` (``a`` and ``b`` the planes of the
        other faces of the two cells, so that the bridge lies in their union; ``e`` the
        reach), negative only where the inner structure seen by both grains is, fills it.
        Clipped to ``[-far, far]``."""
        from morphogen.models.implicit import functions_at

        tess = self.cells.tessellation
        y = self.cells.preimage(pts)
        inverse = self.cells.inverse
        stretch = 1.0 if inverse is None else float(np.linalg.norm(inverse, 2))
        out = np.full(len(pts), far)
        planes = self._cell_planes()
        half = 0.5 * float(tess.boundary_thickness)
        box = tess.domain.lengths
        inner_box = self.inner.domain
        for grain in np.flatnonzero(self._filled_cells):
            normals, offsets, centre, radius = planes[int(grain)]
            d = y - centre
            if tess.domain.periodic:
                d = d - box * np.round(d / box)
            near = np.einsum("ij,ij->i", d, d) <= (radius + stretch * min(reach, far)) ** 2
            index = np.flatnonzero(near)
            if not index.size:
                continue
            if len(offsets):
                cell = ((d[index] + centre) @ normals.T - (offsets - half)).max(axis=1)
            else:  # a single cell: everywhere
                cell = np.full(index.size, -far)
            q = self._local(pts[index], np.full(index.size, int(grain)))
            if inner_box.periodic:
                q = inner_box.wrap(q)
            phi = np.asarray(functions_at(self.inner, q, reach)[1], dtype=float)
            out[index] = np.minimum(out[index], np.maximum(cell, phi))
        if half > 0.0:  # the grain-boundary phase parts the grains: no face shared
            return np.clip(out, -far, far)
        width = stretch * min(reach, far)
        for centre, radius, normal, offset, sides, ca, cb in self._faces_between_filled():
            d = y - centre
            if tess.domain.periodic:
                d = d - box * np.round(d / box)
            index = np.flatnonzero(np.einsum("ij,ij->i", d, d) <= (radius + width) ** 2)
            if not index.size:
                continue
            local = d[index] + centre
            bridge = np.abs(local @ normal - offset) - width
            for normals, offsets in sides:
                if len(offsets):
                    bridge = np.maximum(bridge, (local @ normals.T - offsets).max(axis=1))
            for grain in (ca, cb):
                q = self._local(pts[index], np.full(index.size, grain))
                if inner_box.periodic:
                    q = inner_box.wrap(q)
                bridge = np.maximum(bridge, functions_at(self.inner, q, reach)[1])
            out[index] = np.minimum(out[index], bridge)
        return np.clip(out, -far, far)

    @property
    def lipschitz(self) -> float:
        """A bound of the slopes of the functions: that of the grains, and for filled grains
        those of the inner structure (moved rigidly) and of the planes of the cells (in the
        space of an anisotropic tessellation: the norm of the inverse of its map); maxima
        and minima keep the bound."""
        bound = float(getattr(self.grains, "lipschitz", 1.0))
        if not self.subgrains:
            bound = max(bound, float(getattr(self.inner, "lipschitz", 1.0)))
            if self.cells.inverse is not None:
                bound = max(bound, float(np.linalg.norm(self.cells.inverse, 2)))
        return bound

    # -- description ------------------------------------------------------------------------

    def summary(self) -> dict:
        """The numbers of grains (with those of the window of a non-periodic domain, as a
        tessellation counts its cells) and of subgrains, the mean misorientation of the
        subgrains (degrees), or the number of filled grains."""
        out = {"n_grains": self.n_grains}
        if self.subgrains:
            out["n_subgrains"] = self.n_cells
            out["subgrains_per_grain"] = self.n_cells / max(self.n_grains, 1)
            turned = self.misorientations > 0.0
            if turned.any():
                out["mean_misorientation"] = float(self.misorientations[turned].mean())
        else:
            out["filled_grains"] = int(self._filled_cells.sum())
        return out

    def _geometry(self) -> dict:
        geo: dict = {
            "grains": self.grains.to_dict(),
            "inner": self.inner.to_dict(),
            "turned": self.turned,
            "centres": self.centres.tolist(),
            "filled_phases": list(self.filled_phases),
        }
        if self.turned:
            geo["rotations"] = self.rotations.tolist()
            geo["shifts"] = self.shifts.tolist()
        if self.subgrains:
            geo["parents"] = self.parents.tolist()
            geo["subgrain_germs"] = self.subgrain_germs.tolist()
            if self.subgrain_weights is not None:
                geo["subgrain_weights"] = self.subgrain_weights.tolist()
            geo["orientations"] = self._orientations.tolist()
            geo["misorientations"] = self.misorientations.tolist()
        else:
            geo["inner_phase"] = self.inner_phase
        return geo

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            Microstructure.from_dict(geometry["grains"]),
            Microstructure.from_dict(geometry["inner"]),
            turned=geometry["turned"],
            centres=geometry["centres"],
            filled_phases=geometry["filled_phases"],
            rotations=geometry.get("rotations"),
            shifts=geometry.get("shifts"),
            inner_phase=geometry.get("inner_phase"),
            parents=geometry.get("parents"),
            subgrain_germs=geometry.get("subgrain_germs"),
            subgrain_weights=geometry.get("subgrain_weights"),
            orientations=geometry.get("orientations"),
            misorientations=geometry.get("misorientations"),
            **kwargs,
        )


def _is_tessellation(model) -> bool:
    return isinstance(model, _TessellationModel)


@register
@dataclass(frozen=True)
class NestedGrains(Model):
    """Grains with subgrains, or another structure inside the grains of a polycrystal.

    The grains of a Voronoi or Laguerre tessellation (``grains``) hold the structure of
    another model (``inner``): an inner tessellation divides them into subgrains, each the
    crystal of its grain turned by a small angle (``misorientation``, about a uniformly
    random axis), the sub-boundaries stopping at the grain boundaries; any other model
    fills them with its phase 1 (lamellar colonies, precipitates). With ``turned``, each
    grain sees the inner structure in the axes of its crystal, at its own place of one
    realization in a periodic box; otherwise one realization in the axes of the sample is
    cut by the grains. See :mod:`morphogen.models.nested_grains`.
    """

    name = "nested_grains"
    phases: ClassVar[dict[int, str]] = {
        0: "phase 0 of the grains",
        1: "grains",
        2: "phase 1 of the inner model",
    }
    count_label = "grains"

    grains: Model = param(
        doc="The grains: a Voronoi or Laguerre tessellation, with its phases, crystal "
        "orientations and grain boundaries",
        kind="model",
    )
    inner: Model = param(
        doc="What the grains hold: a tessellation gives their subgrains; any other model "
        "fills them with its phase 1",
        kind="model",
    )
    misorientation: Distribution | None = param(
        Constant(2.0),
        doc="Subgrains: the angle by which each is turned from the crystal of its grain, "
        "about a uniformly random axis (a number or a distribution; none: not turned)",
        kind="distribution",
        unit="degrees",
    )
    turned: bool = param(
        True,
        doc="The inner structure seen by each grain in the axes of its crystal, at its own "
        "place of the realization (false: one realization in the axes of the sample, cut "
        "by the grains)",
        kind="bool",
    )
    inner_phase: int | None = param(
        None,
        doc="Filled grains: the phase of the phase 1 of the inner model inside them "
        "(default: the next free phase)",
        kind="int",
    )
    filled_phases: tuple[int, ...] | None = param(
        None,
        doc="The phases of the grains that the inner structure divides or fills (default: "
        "every phase of the cells)",
        kind="ints",
    )

    def _validate(self) -> None:
        if not _is_tessellation(self.grains):
            raise ValueError(
                f"grains: a Voronoi or Laguerre tessellation (type voronoi or laguerre), not "
                f"{self.grains.name}"
            )
        if self.subgrains:
            inner = self.inner
            if inner.phase_fractions is not None or inner.boundary_thickness > 0:
                raise ValueError(
                    "inner: a tessellation gives subgrains, which keep the phases of their "
                    "grains: no phase_fractions nor boundary_thickness"
                )
            if not self.grains.crystal_orientations:
                raise ValueError(
                    "grains: subgrains turn the crystals of their grains; give the grains "
                    "crystal orientations (crystal_orientations: true)"
                )
            if self.inner_phase is not None:
                raise ValueError(
                    "inner_phase: the phase of what fills the grains (an inner model that is "
                    "not a tessellation)"
                )
        if self.misorientation is not None and self.misorientation.upper() > 90.0:
            raise ValueError("misorientation: angles of at most 90 degrees")
        if self.inner_phase is not None and not 0 <= self.inner_phase <= 255:
            raise ValueError("inner_phase: a phase index from 0 to 255")
        if not self.subgrains and self.phase_of_inner > 255:
            raise ValueError("inner_phase: a phase index from 0 to 255")
        if self.filled_phases is not None:
            cells = self.grain_phases
            wrong = sorted(set(self.filled_phases) - set(cells))
            if wrong or not self.filled_phases:
                raise ValueError(
                    f"filled_phases: among the phases of the cells {sorted(cells)}, not {wrong}"
                )
        if self.turned and not self.inner.supports_periodic:
            raise ValueError(
                f"inner: {self.inner.name} cannot be periodic, which turned grains need (each "
                "sees its own place of a periodic box); use turned: false"
            )

    # -- what the parameters choose -----------------------------------------------------------

    @property
    def subgrains(self) -> bool:
        """Whether the inner model divides the grains into subgrains (a tessellation)."""
        return _is_tessellation(self.inner)

    @property
    def grain_phases(self) -> list[int]:
        """The phases of the cells of the grains (their boundary phase left out)."""
        fractions = self.grains.phase_fractions
        if fractions is None:
            return [1]
        return [i for i, p in enumerate(fractions) if p > 0]

    @property
    def divided_phases(self) -> list[int]:
        """The phases of the grains that the inner structure divides or fills."""
        if self.filled_phases is None:
            return self.grain_phases
        return sorted(int(k) for k in self.filled_phases)

    @property
    def phase_of_inner(self) -> int:
        """The phase of the inner phase 1 inside filled grains."""
        if self.inner_phase is not None:
            return int(self.inner_phase)
        return max(self.grains.active_phases()) + 1

    @property
    def has_level_set(self) -> bool:
        return bool(self.grains.has_level_set) if self.subgrains else False

    @property
    def supports_periodic(self) -> bool:
        return self.turned or self.inner.supports_periodic

    def active_phases(self) -> dict[int, str]:
        out = self.grains.active_phases()
        if not self.subgrains:
            out.setdefault(self.phase_of_inner, f"{self.inner.name} in the grains")
        return out

    def expected_count(self, domain: Domain) -> float:
        """Mean number of grains in the domain."""
        return self.grains.expected_count(domain)

    def inner_domain(self, domain: Domain) -> Domain:
        """The box of the inner realization: periodic like the domain, or periodic for
        turned grains."""
        if not self.turned:
            return domain
        return Domain(size=domain.size, origin=domain.origin, periodic=True, unit=domain.unit)

    # -- realizations ---------------------------------------------------------------------------

    def generate(self, domain: Domain, seed: SeedLike = None) -> NestedGrainStructure:
        self._check_domain(domain)
        rng, seed_info = make_rng(seed)
        seeds = _child_seeds(rng, seed_info, 3)
        grains = self.grains.generate(domain, seeds[0])
        inner = self.inner.generate(self.inner_domain(domain), seeds[1])
        local = np.random.default_rng(seeds[2])
        cells = _Cells(grains)
        n = cells.n_cells
        # the centre of a grain is its germ: of the periodic images of a point, the one in
        # the whole cell is the nearest to the germ (in the power distance of the cells)
        centres = cells.germs
        reach = np.zeros(n)  # the farthest point of each cell from its germ
        if self.subgrains:
            for i, v in enumerate(cells.vertices()):
                if v is not None:
                    reach[i] = float(np.max(np.linalg.norm(v - centres[i], axis=1)))
        orientations = getattr(grains, "orientations", None)
        rotations = shifts = None
        if self.turned:
            rotations = (
                np.asarray(orientations, dtype=float)
                if orientations is not None
                else random_orientations(local, n)
            )
            box = inner.domain
            shifts = box.lower + local.random((n, 3)) * box.lengths
        common = {
            "turned": self.turned,
            "centres": centres,
            "filled_phases": self.divided_phases,
            "rotations": rotations,
            "shifts": shifts,
            "model": self,
            "seed": seed_info,
        }
        if not self.subgrains:
            return NestedGrainStructure(
                domain, grains, inner, inner_phase=self.phase_of_inner, **common
            )
        # the frame of each grain: q = g (x - c) + s
        if self.turned:
            frame = (centres, euler_to_matrix(rotations), shifts)
        else:
            frame = (centres, np.broadcast_to(np.eye(3), (n, 3, 3)), centres)
        divided = np.isin(cells.tessellation.cell_phases, self.divided_phases)
        sub = self._subgrain_germs(domain, cells, inner, frame, reach, divided)
        parents, sub_germs, sub_weights = sub
        euler = np.asarray(orientations, dtype=float)[parents]
        angles = np.zeros(len(parents))
        turns = divided[parents]
        if self.misorientation is not None and turns.any():
            m = int(turns.sum())
            angles[turns] = self.misorientation.sample(local, m)
            axes = uniform_unit_vectors(local, m)
            g = euler_to_matrix(euler[turns]) @ _turns(axes, np.radians(angles[turns]))
            euler[turns] = matrix_to_euler(g)
        return NestedGrainStructure(
            domain,
            grains,
            inner,
            parents=parents,
            subgrain_germs=sub_germs,
            subgrain_weights=sub_weights,
            orientations=euler,
            misorientations=angles,
            **common,
        )

    def _subgrain_germs(self, domain, cells, inner, frame, reach, divided):
        """The grain of each subgrain (sorted), its germ in the frame of its grain
        (``frame``: the centres, rotations and shifts of the grains) and its weight: the
        germs of the inner tessellation that the divided grains see in their cells; a
        single subgrain for the other grains and those that see none."""
        n = cells.n_cells
        centres, matrices, shifts = frame
        inner_cells = _Cells(inner)
        y = inner_cells.germs
        w = inner_cells.weights
        grain = np.zeros(0, dtype=np.int64)
        germ = np.zeros((0, 3))
        index = np.zeros(0, dtype=np.int64)
        seeing = np.flatnonzero(divided & (reach > 0))
        if self.turned and len(y) and seeing.size:
            box = inner.domain  # the germs within the reach of the grain about its shift,
            lengths = box.lengths  # in every periodic image of the box that comes so close
            y = box.wrap(y)
            images = math.ceil(float(reach.max()) / float(lengths.min()))
            span = range(-images, images + 1)
            offsets = np.array([(a, b, c) for a in span for b in span for c in span],
                               dtype=float) * lengths  # fmt: skip
            g = np.repeat(seeing, len(offsets))
            offset = np.tile(offsets, (len(seeing), 1))
            near = cKDTree(y).query_ball_point(shifts[g] - offset, reach[g])
            counts = np.array([len(found) for found in near], dtype=np.int64)
            if counts.sum():
                index = np.concatenate([np.asarray(f, dtype=np.int64) for f in near if f])
                g, offset = np.repeat(g, counts), np.repeat(offset, counts, axis=0)
                q = y[index] + offset
                # back in the sample: x = c + g^T (q - s), in the cell of its grain about c
                d = np.einsum("nji,nj->ni", matrices[g], q - shifts[g])
                keep = cells.cell_of(centres[g] + d) == g
                if domain.periodic:
                    keep &= np.all(np.abs(d) < 0.5 * domain.lengths, axis=1)
                grain, germ, index = g[keep], q[keep], index[keep]
        elif len(y) and seeing.size:
            g = cells.cell_of(y)
            d = y - centres[g]
            if domain.periodic:
                d = _wrap(d, domain.lengths)
            keep = divided[g]
            grain, germ, index = g[keep], centres[g[keep]] + d[keep], np.flatnonzero(keep)
        weight = None if w is None else np.asarray(w, dtype=float)[index]
        alone = np.setdiff1d(np.arange(n), grain)  # a single subgrain: the grain itself
        grain = np.concatenate([grain, alone]).astype(np.int64)
        germ = np.concatenate([germ, shifts[alone]])
        index = np.concatenate([index, np.full(len(alone), -1)])
        if weight is not None:
            weight = np.concatenate([weight, np.zeros(len(alone))])
        order = np.lexsort((index, grain))
        return grain[order], germ[order], None if weight is None else weight[order]

    # -- theory ---------------------------------------------------------------------------------

    def _pairs(self, n: int = 128):
        """Quadrature of two independent misorientations: half angles (radians) and
        weights."""
        if self.misorientation is None:
            return np.zeros(1), np.zeros(1), np.ones(1)
        x, w = self.misorientation.quadrature(n)
        half = 0.5 * np.radians(np.asarray(x, dtype=float))
        a, b = np.meshgrid(half, half, indexing="ij")
        return a.ravel(), b.ravel(), np.outer(w, w).ravel() / float(np.sum(w)) ** 2

    def subgrain_disorientation_cdf(self, angles, n: int = 128) -> np.ndarray:
        """The law of the angle between the crystals of two subgrains of one grain (their
        disorientation, below 45 degrees for cubic crystals and 30 for hexagonal ones), at
        ``angles`` (degrees): for subgrains turned by ``t_a`` and ``t_b`` about independent
        uniform axes, ``cos(psi / 2)`` is uniform between ``cos((t_a + t_b) / 2)`` and
        ``cos((t_a - t_b) / 2)``, averaged over the two misorientations."""
        x = np.radians(np.asarray(angles, dtype=float))
        a, b, w = self._pairs(n)
        lo, hi = np.cos(a + b), np.cos(a - b)
        flat = (hi - lo) <= 1e-14
        span = np.where(flat, 1.0, hi - lo)
        out = np.empty(x.size)
        flat_x = x.reshape(-1)
        step = max(1, (1 << 22) // len(w))  # angles at a time: bounded memory
        for k in range(0, flat_x.size, step):
            t = flat_x[k : k + step, None]
            p = np.clip((hi - np.cos(0.5 * t)) / span, 0.0, 1.0)
            p = np.where(flat, (t >= 2.0 * (a + b) - 1e-12).astype(float), p)
            out[k : k + step] = p @ w
        return out.reshape(x.shape)

    def subgrain_disorientation_moments(self, n: int = 128) -> tuple[float, float]:
        """The mean and the mean square (degrees, degrees squared) of the angle between
        two subgrains of one grain (see :meth:`subgrain_disorientation_cdf`), from the
        primitives ``u arccos u - sqrt(1 - u^2)`` and ``u arccos^2 u - 2 sqrt(1 - u^2)
        arccos u - 2 u`` of ``arccos u`` and ``arccos^2 u``."""
        a, b, w = self._pairs(n)
        lo, hi = np.cos(a + b), np.cos(a - b)
        flat = (hi - lo) <= 1e-14

        def f1(u):
            return u * np.arccos(u) - np.sqrt(np.maximum(1.0 - u * u, 0.0))

        def f2(u):
            s = np.sqrt(np.maximum(1.0 - u * u, 0.0))
            return u * np.arccos(u) ** 2 - 2.0 * s * np.arccos(u) - 2.0 * u

        span = np.where(flat, 1.0, hi - lo)
        mean = np.where(flat, 2.0 * (a + b), 2.0 * (f1(hi) - f1(lo)) / span)
        square = np.where(flat, 4.0 * (a + b) ** 2, 4.0 * (f2(hi) - f2(lo)) / span)
        return math.degrees(float(np.sum(w * mean))), math.degrees(1.0) ** 2 * float(
            np.sum(w * square)
        )

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """The characteristics of the grains (their phases: subgrains keep them); for
        subgrains, the law of the misorientations, the mean angle between two subgrains
        of one grain and the mean number of germs of subgrains in a grain (that of the
        inner germs in its volume, ``lambda_inner / lambda_grains``: the number of
        subgrains, but for the grains that hold none, which count one); for filled grains,
        the fractions of the phases, ``f p`` for the inner phase (``f`` the fraction of
        the filled grains, ``p`` that of the inner model: each point of a grain sees a
        point of the inner realization uniform in its box)."""
        try:
            grains = self.grains.theory(domain)
        except (ValueError, NotImplementedError):
            grains = {}
        if self.subgrains:
            out = dict(grains)
            if self.misorientation is not None:
                out["misorientation_mean"] = self.misorientation.mean()
            out["subgrain_disorientation_mean"] = self.subgrain_disorientation_moments()[0]
            try:
                inner = self.inner.theta(self.inner_domain(domain) if domain else None)
                outer = self.grains.theta(domain)
            except ValueError:
                return out
            out["subgrain_germs_per_grain"] = inner / outer
            return out
        out = {}
        try:
            box = None if domain is None else self.inner_domain(domain)
            p = self.inner.theory(box).get("volume_fraction")
        except (ValueError, NotImplementedError):
            p = None
        if p is None or self.grains.boundary_thickness > 0:
            return out
        out["inner_fraction"] = p
        fractions = self.grains.phase_fractions
        cells = {1: 1.0} if fractions is None else dict(enumerate(map(float, fractions)))
        divided = set(self.divided_phases)
        phases: dict[int, float] = {}
        for k, f in cells.items():
            kept = f * (1.0 - p) if k in divided else f
            phases[k] = phases.get(k, 0.0) + kept
        inner = self.phase_of_inner
        filled = sum(f for k, f in cells.items() if k in divided)
        phases[inner] = phases.get(inner, 0.0) + filled * p
        for k, f in sorted(phases.items()):
            out[f"volume_fraction_{k}"] = f
        if 1 in phases:
            out["volume_fraction"] = phases[1]
        return out


__all__ = ["NestedGrainStructure", "NestedGrains"]
