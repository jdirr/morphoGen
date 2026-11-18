"""Python front-end of the union rasterization kernels."""

from __future__ import annotations

import numpy as np

from morphogen.kernels import raster
from morphogen.primitives import Segments, Spheres

#: Upper bounds on the cell grid used to accelerate queries.
MAX_CELLS_PER_AXIS = 256
MAX_CELLS = 2**22


class UnionAccelerator:
    """Spheres and segments registered in a cell grid covering ``[lower, upper]``.

    ``band`` is the distance up to which signed distances must be exact; membership
    queries only need ``band = 0``.
    """

    def __init__(
        self,
        spheres: Spheres | None,
        segments: Segments | None,
        lower,
        upper,
        *,
        band: float = 0.0,
        min_cell: float = 0.0,
    ) -> None:
        spheres = spheres if spheres is not None else Spheres(np.empty((0, 3)), np.empty(0))
        segments = (
            segments
            if segments is not None
            else Segments(np.empty((0, 3)), np.empty((0, 3)), np.empty(0))
        )
        self.spheres = spheres
        self.segments = segments
        self.band = float(band)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        lengths = upper - lower

        typical = []
        if len(spheres):
            typical.append(2.0 * float(np.median(spheres.radii)))
        if len(segments):
            typical.append(2.0 * float(np.median(segments.radii)))
        target = max(min(typical) if typical else lengths.max(), float(min_cell), 1e-12)
        n = np.clip(np.floor(lengths / target), 1, MAX_CELLS_PER_AXIS).astype(np.int64)
        while np.prod(n) > MAX_CELLS:
            n = np.maximum(1, n // 2)
        self.cell_n = n
        self.cell_lo = lower.copy()
        self.cell_h = lengths / n

        self.s_off, self.s_items = raster.build_cells(
            "spheres", (spheres.centers, spheres.radii), self.band, self.cell_lo, self.cell_h, n
        )
        self.g_off, self.g_items = raster.build_cells(
            "segments",
            (segments.start, segments.end, segments.radii),
            self.band,
            self.cell_lo,
            self.cell_h,
            n,
        )

    def _args(self):
        s, g = self.spheres, self.segments
        return (
            s.centers,
            s.radii,
            self.s_off,
            self.s_items,
            g.start,
            g.end,
            g.radii,
            g.round_caps,
            self.g_off,
            self.g_items,
            self.cell_lo,
            self.cell_h,
            self.cell_n,
        )

    def count(self, grid_lo, spacing, shape, supersampling: int = 1, out=None) -> np.ndarray:
        """Sub-samples (``supersampling**3`` per voxel) inside the union, as uint8."""
        ss = int(supersampling)
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        shape = np.asarray(shape, dtype=np.int64)
        if out is None:
            out = np.zeros(tuple(shape), dtype=np.uint8)
        raster.rasterize_count(
            *self._args(),
            np.asarray(grid_lo, dtype=float),
            np.asarray(spacing, dtype=float),
            shape,
            ss,
            out,
        )
        return out

    def distance(self, grid_lo, spacing, shape, out=None) -> np.ndarray:
        """Signed distance at voxel centres (float32), clipped to ``[-band, band]``."""
        if not self.band > 0:
            raise ValueError("signed distances need a positive band")
        shape = np.asarray(shape, dtype=np.int64)
        if out is None:
            out = np.empty(tuple(shape), dtype=np.float32)
        raster.rasterize_distance(
            *self._args(),
            np.asarray(grid_lo, dtype=float),
            np.asarray(spacing, dtype=float),
            shape,
            self.band,
            out,
        )
        return out

    def inside(self, points) -> np.ndarray:
        pts = np.ascontiguousarray(points, dtype=float).reshape(-1, 3)
        out = np.zeros(len(pts), dtype=np.bool_)
        raster.points_inside(*self._args(), pts, out)
        return out

    def signed_distance(self, points) -> np.ndarray:
        if not self.band > 0:
            raise ValueError("signed distances need a positive band")
        pts = np.ascontiguousarray(points, dtype=float).reshape(-1, 3)
        out = np.empty(len(pts), dtype=float)
        raster.points_distance(*self._args(), pts, self.band, out)
        return out
