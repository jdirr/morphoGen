"""Microstructures defined cell by cell on a tiling other than the square grid.

Cellular automata (and reaction-diffusion, thresholded cell by cell) on triangles,
hexagons, Penrose rhombi, Ammann-Beenker squares and rhombi, prisms, truncated octahedra,
rhombic dodecahedra or golden rhombohedra (:mod:`morphogen.automata.tilings`) give a
phase per cell. Images are rasterized at any resolution, each voxel taking the phase of
the cell containing its centre (partial volumes by supersampling), so that every analysis
and solver applies. The realization keeps the specification of its tiling (lattice, box,
cell size and, for quasi-periodic tilings, the offsets of their grids and the order of a
periodic approximant), from which the tiling is rebuilt, and the phases of the cells
meeting the domain.
"""

from __future__ import annotations

import numpy as np

from morphogen.automata.tilings import Tiling, make_tiling
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.base import Microstructure, register_kind
from morphogen.models.lattice_field import _decode, _encode

#: Voxels per cell size of the default images (2D, 3D).
VOXELS_PER_CELL = {2: 4, 3: 2}


def default_grid(domain: Domain, tiling: Tiling) -> Grid:
    """Grid of the default images of a tiling: 4 voxels per cell size in 2D (one through
    the thickness), 2 in 3D."""
    n = np.rint(domain.lengths * VOXELS_PER_CELL[tiling.dims] / tiling.cell_size)
    n = np.maximum(1, n).astype(int)
    if tiling.dims == 2:
        n[2] = 1
    return Grid(domain, tuple(int(v) for v in n))


@register_kind
class TilingField(Microstructure):
    """A phase per cell of a tiling covering the domain.

    Parameters
    ----------
    tiling
        The :class:`~morphogen.automata.tilings.Tiling`, or its ``spec``.
    phases
        ``uint8`` phase of each cell (of each of ``cells`` when given).
    cells
        Indices of the cells kept (those meeting a window cut from a larger tiling);
        all cells when None.
    attributes
        JSON-able data (number of steps, history of the run...).
    """

    kind = "tiling_field"

    def __init__(
        self,
        domain: Domain,
        tiling: Tiling | dict,
        phases,
        *,
        cells=None,
        attributes: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(domain, **kwargs)
        self.tiling = tiling if isinstance(tiling, Tiling) else make_tiling(**tiling)
        self.phases = np.ascontiguousarray(phases, dtype=np.uint8).ravel()
        self.cells = None if cells is None else np.ascontiguousarray(cells, dtype=np.int64)
        n = self.tiling.n_cells if self.cells is None else len(self.cells)
        if len(self.phases) != n:
            raise ValueError(f"phases: one per cell ({n}), got {len(self.phases)}")
        self.attributes = dict(attributes or {})
        self._position = None

    @property
    def default_grid(self) -> Grid:
        return default_grid(self.domain, self.tiling)

    @property
    def binary(self) -> bool:
        return bool(self.phases.max(initial=0) <= 1)

    def phase_of(self, cells) -> np.ndarray:
        """Phases of cells of the tiling (0 for cells not kept: outside the window)."""
        cells = np.asarray(cells)
        if self.cells is None:
            return self.phases[cells]
        if self._position is None:
            pos = np.full(self.tiling.n_cells, len(self.phases), dtype=np.int64)
            pos[self.cells] = np.arange(len(self.cells))
            self._position = pos
        return np.append(self.phases, np.uint8(0))[self._position[cells]]

    def cell_image(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Image of the index (in the tiling) of the cell containing each voxel centre."""
        grid = self._resolve(shape, voxel_size, grid)
        data = self.tiling.cell_index(grid)
        return VoxelImage.from_grid(data, grid, name="cell", metadata=self.image_metadata())

    def _resolve(self, shape, voxel_size, grid) -> Grid:
        if shape is None and voxel_size is None and grid is None:
            grid = self.default_grid
        return self.grid(shape, voxel_size=voxel_size, grid=grid)

    # -- discretization ----------------------------------------------------------------

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self._resolve(shape, voxel_size, grid)
        ss = int(supersampling)
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        meta = self.image_metadata()
        if ss == 1:
            data = self.phase_of(self.tiling.cell_index(grid))
            return VoxelImage.from_grid(data, grid, name="phase", metadata=meta)
        if not self.binary:
            raise ValueError("partial volumes are defined for two-phase structures")
        sub = (ss, ss, 1 if self.tiling.dims == 2 else ss)
        fine = Grid(grid.domain, tuple(n * s for n, s in zip(grid.shape, sub, strict=True)))
        inside = (self.phase_of(self.tiling.cell_index(fine)) == 1).astype(np.float32)
        nx, ny, nz = grid.shape
        data = inside.reshape(nx, sub[0], ny, sub[1], nz, sub[2]).mean(
            axis=(1, 3, 5), dtype=np.float32
        )
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=meta)

    def contains(self, points) -> np.ndarray:
        return self.phase_of(self.tiling.locate(points)) == 1

    # -- implicit functions ----------------------------------------------------------------

    def implicit_phases(self) -> list[int]:
        phases = {int(k) for k in np.unique(self.phases)} | {0, 1}
        if self.model is not None:
            phases |= {int(k) for k in self.model.active_phases()}
        return sorted(phases)

    def _cell_planes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The planes ``(n, h)`` of the faces of the cells from their centres (``n . x <=
        h`` inside), by kind of cell (the cells of a kind are translated copies): the kind
        of each cell, the planes of each kind (padded) and their numbers."""
        if getattr(self, "_planes", None) is not None:
            return self._planes
        t = self.tiling
        if t._plane is not None:  # polygons (prisms through the layers in 3D)
            p = t._plane
            m = len(p["centres"])
            kind = np.arange(t.n_cells) % m
            ptr = p["poly_ptr"]
            poly = p["poly"] * p["scale"][:2]
            extra = 2 if t.dims == 3 else 0
            width = int(np.diff(ptr).max()) + extra
            planes = np.zeros((m, width, 4))
            counts = np.zeros(m, dtype=np.int64)
            hz = 0.5 * p["layer_height"] * p["scale"][2]
            for c in range(m):
                v = poly[ptr[c] : ptr[c + 1]]
                edge = np.roll(v, -1, axis=0) - v  # counterclockwise: the outside on the right
                normal = np.c_[edge[:, 1], -edge[:, 0]]
                normal /= np.linalg.norm(normal, axis=1, keepdims=True)
                k = len(v)
                planes[c, :k, :2] = normal
                planes[c, :k, 3] = np.einsum("ij,ij->i", normal, v)
                if extra:
                    planes[c, k, 2], planes[c, k + 1, 2] = 1.0, -1.0
                    planes[c, k : k + 2, 3] = hz
                counts[c] = k + extra
        else:
            from scipy.spatial import ConvexHull

            s = t._solid
            if "frames" in s:  # rhombohedra: a kind for each triple of edges
                kind = s["triple"].astype(np.int64)
                corners = [
                    np.array(list(np.ndindex(2, 2, 2)), dtype=float) @ f - 0.5 * f.sum(axis=0)
                    for f in s["frames"]
                ]
            else:  # the Voronoi cells of a lattice: all alike
                kind = np.zeros(t.n_cells, dtype=np.int64)
                corners = [s["prototype"] * s["scale"]]
            hulls = [ConvexHull(v).equations for v in corners]
            width = max(len(h) for h in hulls)
            planes = np.zeros((len(hulls), width, 4))
            counts = np.array([len(h) for h in hulls], dtype=np.int64)
            for c, h in enumerate(hulls):
                planes[c, : len(h), :3] = h[:, :3]
                planes[c, : len(h), 3] = -h[:, 3]
        self._planes = (kind, planes, counts)
        return self._planes

    def phase_level_set(
        self, phase: int, shape=None, *, voxel_size=None, grid: Grid | None = None
    ) -> VoxelImage:
        """The implicit function of a phase (see :meth:`Microstructure.phase_level_set`):
        the cells of the phase are convex, each the points on the inner side of the planes
        of its faces; the function is half the smallest gauge of the cells of the phase
        (the largest distance of the point to the planes of a cell, negative in it) minus
        half the smallest gauge of the other cells, among the cell of the point and those
        that share a vertex with it: zero on the faces between a cell of the phase and
        another, the distance to the face near it, clipped to a band of four voxels."""
        from morphogen.kernels.cells import cells_implicit
        from morphogen.models.implicit import band_of

        phase = int(phase)
        if phase not in self.implicit_phases():
            raise ValueError(f"phase {phase}: the phases are {self.implicit_phases()}")
        grid = self._resolve(shape, voxel_size, grid)
        t = self.tiling
        kind, planes, counts = self._cell_planes()
        indptr, indices = t.adjacency("vertices")
        inside = self.phase_of(np.arange(t.n_cells)) == phase
        lengths = np.where(t.periodic, t.size, 0.0).astype(float)
        if t.dims == 2:
            lengths[2] = 0.0
        band = band_of(grid)
        nx, ny, nz = grid.shape
        out = np.empty(grid.shape, dtype=np.float32)
        step = max(1, (1 << 20) // max(1, nx * ny))
        for k0 in range(0, nz, step):
            k1 = min(nz, k0 + step)
            pts = grid.center_points(k0, k1)
            values = np.empty(len(pts))
            cells_implicit(
                pts,
                t.locate(pts).astype(np.int64),
                indptr.astype(np.int64),
                indices.astype(np.int64),
                t.centres,
                kind,
                planes,
                counts,
                inside,
                lengths,
                t.dims,
                band,
                values,
            )
            out[:, :, k0:k1] = values.reshape(nx, ny, k1 - k0)
        return VoxelImage.from_grid(out, grid, name="level_set", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """The implicit function of phase 1 (:meth:`phase_level_set`)."""
        return self.phase_level_set(1, shape, voxel_size=voxel_size, grid=grid)

    def phase_level_sets_at(self, points, reach: float = np.inf) -> dict[int, np.ndarray]:
        """The functions of :meth:`phase_level_set` at the points: half the smallest gauge
        of the cells of each phase minus half the smallest gauge of the other cells, among
        the cell of the point and those that share a vertex with it, within ``[-reach,
        reach]`` (default: the diagonal of the box)."""
        from morphogen.kernels.cells import cells_implicit

        pts = np.ascontiguousarray(np.asarray(points, dtype=float).reshape(-1, 3))
        band = float(self.domain.diagonal) if not np.isfinite(reach) else float(reach)
        t = self.tiling
        kind, planes, counts = self._cell_planes()
        indptr, indices = t.adjacency("vertices")
        lengths = np.where(t.periodic, t.size, 0.0).astype(float)
        if t.dims == 2:
            lengths[2] = 0.0
        cell = t.locate(pts).astype(np.int64)
        phase_of = self.phase_of(np.arange(t.n_cells))
        out = {}
        for k in self.implicit_phases():
            values = np.empty(len(pts))
            cells_implicit(pts, cell, indptr.astype(np.int64), indices.astype(np.int64),
                           t.centres, kind, planes, counts, phase_of == k, lengths, t.dims,
                           band, values)  # fmt: skip
            out[k] = values
        return out

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """The function of phase 1 at the points (:meth:`phase_level_sets_at`)."""
        return self.phase_level_sets_at(points, reach)[1]

    # -- description ---------------------------------------------------------------------

    def summary(self) -> dict:
        out = {"lattice": self.tiling.name, "n_cells": len(self.phases)}
        if "steps" in self.attributes:
            out["steps"] = self.attributes["steps"]
        return out

    def _geometry(self) -> dict:
        geo = {"tiling": self.tiling.spec, "phases": _encode(self.phases)}
        if self.cells is not None:
            geo["cells"] = _encode(self.cells)
        if self.attributes:
            geo["attributes"] = self.attributes
        return geo

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            geometry["tiling"],
            _decode(geometry["phases"]),
            cells=_decode(geometry["cells"]) if "cells" in geometry else None,
            attributes=geometry.get("attributes"),
            **kwargs,
        )
