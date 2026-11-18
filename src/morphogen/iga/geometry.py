"""NURBS patches and their assemblies: the geometry of an isogeometric analysis.

A patch is a tensor product of B-spline bases (:mod:`morphogen.iga.basis`) with control
points ``P_a`` and positive weights ``w_a``: a point of parameters ``xi`` is::

    x(xi) = sum_a R_a(xi) P_a,    R_a = N_a w_a / sum_b N_b w_b,

``N_a`` the product of the functions of each direction. Rational functions give conics —
circles, ellipses — exactly; with unit weights the patch is a polynomial spline. A patch
of two parameters in the plane is a surface (plane stress or plane strain), of three
parameters a solid; their sides are patches of one parameter less.

A :class:`Multipatch` assembles patches that share whole sides: their control points
there coincide (with their weights and their knots, so that the patches fit along the
whole side), and are one control point of the assembly. The unknowns of an analysis are
attached to these control points (the isoparametric concept: the functions of the
geometry are those of the fields).

Refinement keeps the geometry (:meth:`Patch.refined`): every element cut in parts
(*h*), the degree raised (*p*), or the degree raised then the elements cut (*k*, which
keeps the continuity of the higher degree across the new knots). The control points of
the refined patch are linear in those of the coarse one for given weights:
:meth:`Multipatch.refinement` returns the matrix of that map, by which a design made of
few control points drives the many of its analysis (:mod:`morphogen.iga.optimization`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from morphogen.iga import basis as bs


def _per_direction(value, ndim: int, name: str) -> tuple[int, ...]:
    values = (int(value),) * ndim if np.isscalar(value) else tuple(int(v) for v in value)
    if len(values) != ndim:
        raise ValueError(f"{name}: one value, or one per direction ({ndim})")
    return values


def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Whether points of the plane are inside a closed polygon (the parity of the
    crossings of a half-line)."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for (a, b), (c, d) in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            inside ^= ((b > y) != (d > y)) & (x < (c - a) * (y - b) / (d - b) + a)
    return inside


def _lattice(lattice, dim: int) -> np.ndarray | None:
    """The periods of a cell as an array (one vector per row), checked."""
    if lattice is None:
        return None
    lattice = np.array(lattice, dtype=float)
    if lattice.shape != (dim, dim):
        raise ValueError(f"lattice: {dim} vectors of {dim} coordinates")
    if abs(np.linalg.det(lattice)) < 1e-9 * np.abs(lattice).max() ** dim:
        raise ValueError("lattice: independent vectors")
    return lattice


@dataclass
class Patch:
    """A NURBS patch: the ``degrees`` and the ``knots`` of its directions, its control
    ``points`` (``n_1 x ... x n_d x dim``) and their ``weights`` (``n_1 x ... x n_d``; unit
    weights by default), and the ``phase`` of its material."""

    degrees: tuple
    knots: tuple
    points: np.ndarray
    weights: np.ndarray | None = None
    phase: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.degrees = tuple(int(p) for p in self.degrees)
        self.knots = tuple(
            bs.check_knots(k, p) for k, p in zip(self.knots, self.degrees, strict=True)
        )
        self.points = np.array(self.points, dtype=float)
        shape = tuple(bs.n_functions(k, p) for k, p in zip(self.knots, self.degrees, strict=True))
        if self.points.shape[:-1] != shape:
            raise ValueError(
                f"points: {shape} control points for these knots and degrees, got "
                f"{self.points.shape[:-1]}"
            )
        if self.weights is None:
            self.weights = np.ones(shape)
        self.weights = np.array(self.weights, dtype=float)
        if self.weights.shape != shape or np.any(self.weights <= 0):
            raise ValueError("weights: one positive weight per control point")
        self.phase = int(self.phase)

    # -- sizes -------------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """The number of parameters: 1 (curve), 2 (surface) or 3 (solid)."""
        return len(self.degrees)

    @property
    def dim(self) -> int:
        """The dimension of the space of the control points."""
        return self.points.shape[-1]

    @property
    def shape(self) -> tuple[int, ...]:
        """The numbers of control points along the directions."""
        return self.weights.shape

    @property
    def n_elements(self) -> int:
        return int(np.prod([len(bs.breakpoints(k)) - 1 for k in self.knots]))

    @property
    def rational(self) -> bool:
        return bool(np.ptp(self.weights) > 0.0)

    # -- evaluation --------------------------------------------------------------------

    def _collocations(self, params, derivative=None) -> list[np.ndarray]:
        """The matrices of the functions of each direction at the parameters (points x
        ndim), with those of ``derivative`` differentiated once."""
        params = np.atleast_2d(np.asarray(params, dtype=float))
        if params.shape[1] != self.ndim:
            raise ValueError(f"parameters: {self.ndim} per point")
        return [
            bs.collocation(self.knots[d], self.degrees[d], params[:, d], int(d == derivative))
            for d in range(self.ndim)
        ]

    def _contract(self, matrices, coefficients) -> np.ndarray:
        """``sum_a prod_d C_d[m, a_d] c_a``: the spline of coefficients ``c`` at the
        points (rows of the matrices)."""
        letters = "abc"[: self.ndim]
        operands = ",".join(f"m{letter}" for letter in letters)
        trailing = "k" if coefficients.ndim > self.ndim else ""
        return np.einsum(f"{operands},{letters}{trailing}->m{trailing}", *matrices, coefficients)

    def at(self, params) -> np.ndarray:
        """The points of the patch at parameters (points x ndim): points x dim."""
        matrices = self._collocations(params)
        weighted = self.weights[..., None] * self.points
        return self._contract(matrices, weighted) / self._contract(matrices, self.weights)[:, None]

    def jacobian(self, params) -> tuple[np.ndarray, np.ndarray]:
        """The points at parameters and the derivatives of the position with respect to
        the parameters there (points x dim x ndim)."""
        weighted = self.weights[..., None] * self.points
        matrices = self._collocations(params)
        den = self._contract(matrices, self.weights)
        x = self._contract(matrices, weighted) / den[:, None]
        jac = np.empty((len(x), self.dim, self.ndim))
        for d in range(self.ndim):
            derived = self._collocations(params, d)
            da = self._contract(derived, weighted)
            dw = self._contract(derived, self.weights)
            jac[:, :, d] = (da - x * dw[:, None]) / den[:, None]
        return x, jac

    def sample(self, counts=8) -> np.ndarray:
        """The points of a grid of parameters, ``counts`` per element and direction
        (and the end of the last element): ``m_1 x ... x m_d x dim``."""
        counts = _per_direction(counts, self.ndim, "counts")
        axes = []
        for knots, count in zip(self.knots, counts, strict=True):
            ends = bs.breakpoints(knots)
            inside = ends[:-1, None] + np.diff(ends)[:, None] * np.arange(count)[None, :] / count
            axes.append(np.append(inside.ravel(), ends[-1]))
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        return self.at(grid.reshape(-1, self.ndim)).reshape(*grid.shape[:-1], self.dim)

    # -- sides ---------------------------------------------------------------------------

    def side_indices(self, axis: int, end: int) -> np.ndarray:
        """The control points of a side (``end`` 0: the start of direction ``axis``, 1:
        its end): their indices in the raveled arrays of the patch."""
        index = np.arange(self.weights.size).reshape(self.shape)
        return np.take(index, 0 if end == 0 else -1, axis=axis).ravel()

    def side(self, axis: int, end: int) -> Patch:
        """A side of the patch: a patch of one parameter less."""
        if self.ndim < 2:
            raise ValueError("side: of a surface or a solid")
        take = 0 if end == 0 else -1
        others = [d for d in range(self.ndim) if d != axis]
        return Patch(
            tuple(self.degrees[d] for d in others),
            tuple(self.knots[d] for d in others),
            np.take(self.points, take, axis=axis),
            np.take(self.weights, take, axis=axis),
            self.phase,
        )

    # -- refinement ----------------------------------------------------------------------

    def refinement_operators(self, subdivisions=1, elevate=0) -> tuple[list, list, list]:
        """The matrices of the refinement of each direction (new functions x old ones;
        :func:`morphogen.iga.basis.embedding`), with the new degrees and knots."""
        subdivisions = _per_direction(subdivisions, self.ndim, "subdivisions")
        elevate = _per_direction(elevate, self.ndim, "elevate")
        operators, degrees, knots = [], [], []
        for d in range(self.ndim):
            new_knots = bs.refined_knots(
                self.knots[d], self.degrees[d], subdivisions[d], elevate[d]
            )
            new_degree = self.degrees[d] + elevate[d]
            if subdivisions[d] == 1 and elevate[d] == 0:
                operators.append(np.eye(self.shape[d]))
            else:
                operators.append(
                    bs.embedding(self.knots[d], self.degrees[d], new_knots, new_degree)
                )
            degrees.append(new_degree)
            knots.append(new_knots)
        return operators, degrees, knots

    def refined(self, subdivisions=1, elevate=0) -> Patch:
        """The same patch with every element cut into ``subdivisions`` parts along each
        direction and its degrees raised by ``elevate`` (one value, or one per
        direction): the degree is raised first (*k*-refinement)."""
        operators, degrees, knots = self.refinement_operators(subdivisions, elevate)
        homogeneous = np.concatenate(
            [self.weights[..., None] * self.points, self.weights[..., None]], axis=-1
        )
        for d, matrix in enumerate(operators):
            homogeneous = np.moveaxis(np.tensordot(matrix, homogeneous, axes=(1, d)), 0, d)
        weights = homogeneous[..., -1]
        points = homogeneous[..., :-1] / weights[..., None]
        return Patch(tuple(degrees), tuple(knots), points, weights, self.phase, dict(self.metadata))

    def with_points(self, points) -> Patch:
        """The patch with other control points (the same weights and knots)."""
        return Patch(self.degrees, self.knots, points, self.weights, self.phase,
                     dict(self.metadata))  # fmt: skip


class Multipatch:
    """Patches that share whole sides (see the module): ``patches``, the control points
    of the assembly (``nodes``, ``weights``) and, for each patch, the indices of its
    control points among them (``ids``, of the shape of the patch). Control points
    closer than ``tolerance`` times the size of the assembly are one.

    ``lattice`` (optional; one vector per row) gives the periods of an assembly that is
    the cell of a periodic medium without being a box — a parallelogram, or any set of
    patches that tiles the plane (the space) by these translations: the sides of its
    boundary then match by a period (:mod:`morphogen.iga.homogenization`)."""

    def __init__(self, patches, tolerance: float = 1e-8, lattice=None):
        patches = [patches] if isinstance(patches, Patch) else list(patches)
        if not patches:
            raise ValueError("Multipatch: at least one patch")
        if len({(p.ndim, p.dim) for p in patches}) != 1:
            raise ValueError("Multipatch: patches of the same dimensions")
        self.patches = patches
        self.tolerance = float(tolerance)
        self.lattice = _lattice(lattice, patches[0].dim)
        points = np.concatenate([p.points.reshape(-1, p.dim) for p in patches])
        weights = np.concatenate([p.weights.ravel() for p in patches])
        size = float(np.ptp(points, axis=0).max()) or 1.0
        pairs = cKDTree(points).query_pairs(self.tolerance * size, output_type="ndarray")
        graph = sp.coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2
        )
        count, labels = connected_components(graph, directed=False)
        # the control points of the assembly in the order of their first patch
        _, first = np.unique(labels, return_index=True)
        order = np.argsort(first)
        rank = np.empty(count, dtype=np.int64)
        rank[order] = np.arange(count)
        labels = rank[labels]
        self.nodes = np.zeros((count, points.shape[1]))
        self.nodes[labels] = points
        self.weights = np.zeros(count)
        self.weights[labels] = weights
        if np.abs(self.weights[labels] - weights).max() > 1e-9:
            raise ValueError(
                "Multipatch: control points shared by two patches have different weights"
            )
        self.ids, start = [], 0
        for p in patches:
            self.ids.append(labels[start : start + p.weights.size].reshape(p.shape))
            start += p.weights.size

    # -- sizes -------------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        return self.patches[0].ndim

    @property
    def dim(self) -> int:
        return self.patches[0].dim

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return sum(p.n_elements for p in self.patches)

    @property
    def lower(self) -> np.ndarray:
        """The lower corner of the box of the control points."""
        return self.nodes.min(axis=0)

    @property
    def upper(self) -> np.ndarray:
        return self.nodes.max(axis=0)

    @property
    def phases(self) -> list[int]:
        return sorted({p.phase for p in self.patches})

    def side_nodes(self, patch: int, axis: int, end: int) -> np.ndarray:
        """The control points of a side of a patch (indices in the assembly)."""
        return self.ids[patch].ravel()[self.patches[patch].side_indices(axis, end)]

    def boundary_nodes(self) -> np.ndarray:
        """The control points of the sides that belong to one patch only: the boundary
        of the assembly (the outer one and those of its holes)."""
        sides: dict[tuple, int] = {}
        for k, p in enumerate(self.patches):
            for axis in range(p.ndim):
                for end in (0, 1):
                    key = tuple(sorted(self.side_nodes(k, axis, end).tolist()))
                    sides[key] = sides.get(key, 0) + 1
        nodes = [n for key, count in sides.items() if count == 1 for n in key]
        return np.unique(np.array(nodes, dtype=np.int64))

    def free_sides(self) -> list[tuple[int, int, int]]:
        """The sides (patch, axis, end) that belong to one patch only."""
        sides: dict[tuple, list] = {}
        for k, p in enumerate(self.patches):
            for axis in range(p.ndim):
                for end in (0, 1):
                    key = tuple(sorted(self.side_nodes(k, axis, end).tolist()))
                    sides.setdefault(key, []).append((k, axis, end))
        return [found[0] for found in sides.values() if len(found) == 1]

    # -- other geometries of the same assembly -----------------------------------------------

    def with_nodes(self, nodes) -> Multipatch:
        """The assembly with its control points moved to ``nodes`` (the same weights,
        knots and connections)."""
        nodes = np.asarray(nodes, dtype=float)
        if nodes.shape != self.nodes.shape:
            raise ValueError(
                f"nodes: {self.nodes.shape[0]} control points of {self.dim} coordinates"
            )
        out = Multipatch.__new__(Multipatch)
        out.patches = [
            p.with_points(nodes[ids]) for p, ids in zip(self.patches, self.ids, strict=True)
        ]
        out.tolerance = self.tolerance
        out.lattice = self.lattice
        out.nodes = nodes.copy()
        out.weights = self.weights.copy()
        out.ids = [ids.copy() for ids in self.ids]
        return out

    def refined(self, subdivisions=1, elevate=0) -> Multipatch:
        """The assembly of the refined patches (:meth:`Patch.refined`)."""
        return Multipatch(
            [p.refined(subdivisions, elevate) for p in self.patches], self.tolerance, self.lattice
        )

    def refinement(self, subdivisions=1, elevate=0) -> tuple[Multipatch, sp.csr_matrix]:
        """The refined assembly and the matrix ``T`` of its control points in terms of
        those of this one, for given weights: ``refined.nodes = T @ nodes`` (and for any
        other positions of the control points of this assembly:
        ``refined.with_nodes(T @ other)``)."""
        fine = self.refined(subdivisions, elevate)
        rows, cols, values = [], [], []
        seen = np.zeros(fine.n_nodes, dtype=bool)
        for p, ids, fine_ids in zip(self.patches, self.ids, fine.ids, strict=True):
            operators, _, _ = p.refinement_operators(subdivisions, elevate)
            matrix = operators[0]
            for other in operators[1:]:
                matrix = np.kron(matrix, other)
            scaled = matrix * p.weights.ravel()[None, :]
            scaled /= scaled.sum(axis=1, keepdims=True)  # the weights of the refined points
            targets = fine_ids.ravel()
            new = ~seen[targets]
            seen[targets] = True
            r, c = np.nonzero(scaled[new])
            rows.append(targets[new][r])
            cols.append(ids.ravel()[c])
            values.append(scaled[new][r, c])
        matrix = sp.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(fine.n_nodes, self.n_nodes),
        ).tocsr()
        return fine, matrix

    def jacobians(self, counts: int = 6) -> tuple[float, float]:
        """The smallest and the largest determinants of the Jacobian of the patches, on
        grids of parameters that hold the ends of the elements (``counts`` per element
        and direction): a negative one where a patch folds or turns over — at its
        boundary first, where the points of integration of an analysis do not look."""
        if self.ndim != self.dim:
            raise ValueError("jacobians: of surfaces of the plane or of solids")
        low, high = np.inf, -np.inf
        for p in self.patches:
            axes = []
            for knots in p.knots:
                ends = bs.breakpoints(knots)
                inside = ends[:-1, None] + np.diff(ends)[:, None] * np.arange(counts) / counts
                axes.append(np.append(inside.ravel(), ends[-1]))
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, p.ndim)
            det = np.linalg.det(p.jacobian(grid)[1])
            low, high = min(low, float(det.min())), max(high, float(det.max()))
        return low, high

    def polygons(self, counts: int = 12) -> list[np.ndarray]:
        """The boundary of each patch as a closed polygon (for figures: the patches
        filled): surfaces only."""
        if self.ndim != 2:
            raise ValueError("polygons: of an assembly of surfaces")
        out = []
        for p in self.patches:
            sides = [
                p.side(1, 0).sample(counts),
                p.side(0, 1).sample(counts),
                p.side(1, 1).sample(counts)[::-1],
                p.side(0, 0).sample(counts)[::-1],
            ]
            out.append(np.concatenate([side[:-1] for side in sides]))
        return out

    def voxelize(self, shape, lower=None, upper=None, hole: int | None = None,
                 depth: int = 1, counts: int = 16):  # fmt: skip
        """The image of an assembly of surfaces on a grid of ``shape`` pixels over the
        box from ``lower`` to ``upper`` (that of its control points by default): each
        pixel takes the phase of the patch that holds its centre, ``hole`` where there
        is none (by default the largest phase plus one). With a ``lattice``, the copies
        of the patches by its periods are drawn as well, so that a box that is a period
        of the medium gives a periodic image — the square of two spacings for
        :func:`~morphogen.iga.shapes.alternating_holes`. Curved sides are drawn as
        polygons of ``counts`` points per element.

        Returns a periodic :class:`~morphogen.image.VoxelImage` of ``depth`` voxels along
        the third axis: the shape for the voxel solvers (plane strain), the nonlinear
        behaviours and the exports."""
        from morphogen.image import VoxelImage

        if self.ndim != 2 or self.dim != 2:
            raise ValueError("voxelize: an assembly of surfaces of the plane")
        shape = (int(shape),) * 2 if np.isscalar(shape) else tuple(int(n) for n in shape)
        lower = self.lower if lower is None else np.asarray(lower, dtype=float)
        upper = self.upper if upper is None else np.asarray(upper, dtype=float)
        if len(shape) != 2 or min(shape) < 1 or np.any(upper <= lower):
            raise ValueError("voxelize: a shape of two counts and a box lower < upper")
        step = (upper - lower) / shape
        hole = max(self.phases) + 1 if hole is None else int(hole)
        image = np.full(shape, hole, dtype=np.uint8)
        shifts = np.zeros((1, 2))
        if self.lattice is not None:
            reach = int(np.ceil(np.abs((upper - lower) @ np.linalg.inv(self.lattice)).max())) + 2
            combos = np.array(np.meshgrid(*[np.arange(-reach, reach + 1)] * 2)).reshape(2, -1).T
            shifts = combos @ self.lattice
        for patch, polygon in zip(self.patches, self.polygons(counts), strict=True):
            for shift in shifts:
                moved = polygon + shift
                first = np.maximum(np.floor((moved.min(axis=0) - lower) / step).astype(int), 0)
                last = np.minimum(np.ceil((moved.max(axis=0) - lower) / step).astype(int), shape)
                if np.any(last <= first):
                    continue
                axes = [lower[d] + (np.arange(first[d], last[d]) + 0.5) * step[d] for d in (0, 1)]
                centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
                found = _inside(centres, moved).reshape(len(axes[0]), len(axes[1]))
                image[first[0] : last[0], first[1] : last[1]][found] = patch.phase
        return VoxelImage(
            np.repeat(image[:, :, None], int(depth), axis=2),
            spacing=(step[0], step[1], step[0]),
            origin=(lower[0], lower[1], 0.0),
            periodic=True,
            metadata={"source": "morphogen.iga", "hole": hole},
        )

    def outline(self, counts: int = 12) -> list[np.ndarray]:
        """The sides of the patches as polylines (for figures): surfaces only."""
        if self.ndim != 2:
            raise ValueError("outline: of an assembly of surfaces")
        lines = []
        for p in self.patches:
            for axis in range(2):
                for end in (0, 1):
                    lines.append(p.side(axis, end).sample(counts))
        return lines


__all__ = ["Multipatch", "Patch"]
