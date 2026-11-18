"""Microstructures made of a union of spheres and segments (grains, fibres, struts)."""

from __future__ import annotations

import numpy as np

from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.kernels.union import UnionAccelerator
from morphogen.models.base import Microstructure, register_kind
from morphogen.primitives import Segments, Spheres, restricted_to_box, with_periodic_images


@register_kind
class PrimitiveUnion(Microstructure):
    """Phase 1 is the union of the spheres and segments, phase 0 its complement.

    For periodic domains the primitives are stored once (reference points inside the
    box) and their periodic images are added when the structure is rasterized.
    """

    kind = "primitive_union"

    def __init__(
        self,
        domain: Domain,
        spheres: Spheres | None = None,
        segments: Segments | None = None,
        **kwargs,
    ) -> None:
        super().__init__(domain, **kwargs)
        self.spheres = spheres if spheres is not None else Spheres(np.empty((0, 3)), np.empty(0))
        self.segments = (
            segments
            if segments is not None
            else Segments(np.empty((0, 3)), np.empty((0, 3)), np.empty(0))
        )
        self._cache: dict = {}

    # -- acceleration structures -----------------------------------------------------

    def _covering(self, margin: float) -> tuple[Spheres, Segments]:
        """Primitives (with periodic images if needed) covering the box plus ``margin``."""
        if self.domain.periodic:
            return (
                with_periodic_images(self.spheres, self.domain, margin),
                with_periodic_images(self.segments, self.domain, margin),
            )
        return (
            restricted_to_box(self.spheres, self.domain, margin),
            restricted_to_box(self.segments, self.domain, margin),
        )

    def _accelerator(self, band: float = 0.0, min_cell: float = 0.0) -> UnionAccelerator:
        key = (round(band, 12), round(min_cell, 12))
        if key not in self._cache:
            spheres, segments = self._covering(band)
            lo = self.domain.lower - band
            hi = self.domain.upper + band
            self._cache.clear()
            self._cache[key] = UnionAccelerator(
                spheres, segments, lo, hi, band=band, min_cell=min_cell
            )
        return self._cache[key]

    # -- discretization ----------------------------------------------------------------

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        acc = self._accelerator(0.0, 2.0 * float(grid.spacing.max()))
        counts = acc.count(grid.origin, grid.spacing, grid.shape, supersampling)
        if supersampling == 1:
            return VoxelImage.from_grid(counts, grid, name="phase", metadata=self.image_metadata())
        data = counts.astype(np.float32) / np.float32(supersampling**3)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def level_set(
        self, shape=None, *, voxel_size=None, grid: Grid | None = None, band: float | None = None
    ) -> VoxelImage:
        """Signed distance to the union (negative inside), exact outside the grains and
        clipped to ``[-band, band]`` (default: 4 voxels)."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        band = 4.0 * float(grid.spacing.max()) if band is None else float(band)
        acc = self._accelerator(band, 2.0 * float(grid.spacing.max()))
        data = acc.distance(grid.origin, grid.spacing, grid.shape)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def contains(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.domain.periodic:
            pts = self.domain.wrap(pts)
        return self._accelerator(0.0).inside(pts)

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """The signed distance to the union (negative inside), clipped to ``[-reach,
        reach]`` (default: a tenth of the smallest side of the box)."""
        if not np.isfinite(reach):
            reach = 0.1 * float(self.domain.lengths.min())
        return self.signed_distance(points, float(reach))

    def signed_distance(self, points, band: float) -> np.ndarray:
        """Signed distance at arbitrary points, clipped to ``[-band, band]``."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.domain.periodic:
            pts = self.domain.wrap(pts)
        return self._accelerator(float(band)).signed_distance(pts)

    def implicit_pieces(self, depth: float = 0.0) -> tuple[list, int] | None:
        """The primitives grouped so that no two of a group overlap: the pieces of phase
        1, in what phase 0 leaves (see :mod:`morphogen.models.pieces`); None for the
        unions of more phases."""
        from morphogen.models.pieces import Piece, union_pieces

        if self.implicit_phases() != [0, 1]:
            return None
        return [Piece(1, part) for part in union_pieces(self, 0.0, depth)], 0

    # -- description -------------------------------------------------------------------

    def summary(self) -> dict:
        out = {}
        if len(self.spheres):
            out["n_spheres"] = len(self.spheres)
        if len(self.segments):
            out["n_segments"] = len(self.segments)
        out.update(self.metadata.get("summary", {}))
        return out

    def _geometry(self) -> dict:
        geo = {}
        if len(self.spheres):
            geo["spheres"] = self.spheres.to_dict()
        if len(self.segments):
            geo["segments"] = self.segments.to_dict()
        return geo

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        spheres = Spheres.from_dict(geometry["spheres"]) if "spheres" in geometry else None
        segments = Segments.from_dict(geometry["segments"]) if "segments" in geometry else None
        return cls(domain, spheres, segments, **kwargs)
