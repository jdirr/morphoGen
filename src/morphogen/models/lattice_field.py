"""Microstructures defined cell by cell on a regular lattice.

Cellular automata and lattice growth processes (nucleation and growth, aggregation)
produce a phase per cell, and sometimes a label per cell (grains). Such a realization
is resolution-independent like the other microstructures:

* on its own lattice, it voxelizes to its cells exactly;
* elsewhere, the indicator of phase 1 (or a continuous field, such as a concentration,
  when the process has one) is interpolated trilinearly, after an optional Gaussian
  smoothing of ``smoothing`` cells, and thresholded; labels are taken from the nearest
  cell;
* its level set is smooth across the interface, so that conforming meshes (MMG)
  follow the cells' boundaries, rounded at the corners.
"""

from __future__ import annotations

import base64
import math
import zlib

import numpy as np
from numba import njit, prange
from scipy import ndimage

from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.base import Microstructure, register_kind


@njit(parallel=True, cache=True)
def _trilinear(field, lower, h, points, periodic, out):
    """Trilinear interpolation of a cell-centred field at points."""
    nx, ny, nz = field.shape
    n = (nx, ny, nz)
    for p in prange(points.shape[0]):
        idx0 = np.empty(3, dtype=np.int64)
        idx1 = np.empty(3, dtype=np.int64)
        t = np.empty(3)
        for a in range(3):
            u = (points[p, a] - lower[a]) / h[a] - 0.5
            i0 = math.floor(u)
            ta = u - i0
            i1 = i0 + 1
            if periodic:
                i0 %= n[a]
                i1 %= n[a]
            else:
                if i0 < 0:
                    i0 = 0
                    ta = 0.0
                if i1 > n[a] - 1:
                    i1 = n[a] - 1
                if i0 > n[a] - 1:
                    i0 = n[a] - 1
                if i1 < 0:
                    i1 = 0
            idx0[a] = i0
            idx1[a] = i1
            t[a] = ta
        v = 0.0
        for c in range(8):
            w = 1.0
            ii = idx1[0] if c & 1 else idx0[0]
            w *= t[0] if c & 1 else 1.0 - t[0]
            jj = idx1[1] if c & 2 else idx0[1]
            w *= t[1] if c & 2 else 1.0 - t[1]
            kk = idx1[2] if c & 4 else idx0[2]
            w *= t[2] if c & 4 else 1.0 - t[2]
            v += w * field[ii, jj, kk]
        out[p] = v


def _encode(arr: np.ndarray) -> dict:
    a = np.ascontiguousarray(arr)
    return {
        "dtype": a.dtype.str,
        "shape": list(a.shape),
        "data": base64.b64encode(zlib.compress(a.tobytes(), 6)).decode("ascii"),
    }


def _decode(spec: dict) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(spec["data"]))
    return np.frombuffer(raw, dtype=np.dtype(spec["dtype"])).reshape(spec["shape"]).copy()


@register_kind
class LatticeField(Microstructure):
    """A phase (and optionally a label) per cell of a regular lattice of the domain.

    Parameters
    ----------
    phases
        ``uint8`` array ``(nx, ny, nz)`` (``nz = 1`` for 2D structures, extruded
        through the thickness of the domain).
    labels
        Optional ``int32`` array of the same shape: grain index per cell (``-1``:
        none, e.g. untransformed matrix).
    orientations
        Optional crystal orientations of the labels (Bunge Euler angles, degrees).
    smoothing
        Gaussian smoothing, in cells, of the indicator of phase 1 before interpolation
        (0: plain trilinear interpolation, which keeps one-cell features).
    field, threshold
        Optional continuous field per cell (e.g. a concentration) whose excursion
        ``field > threshold`` is phase 1: it is interpolated instead of the indicator,
        which gives smoother interfaces.
    attributes
        Per-label arrays (e.g. nucleation times) and other JSON-able data.
    """

    kind = "lattice_field"

    def __init__(
        self,
        domain: Domain,
        phases,
        *,
        labels=None,
        orientations=None,
        smoothing: float = 0.0,
        field=None,
        threshold: float = 0.0,
        attributes: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(domain, **kwargs)
        self.phases = np.ascontiguousarray(phases, dtype=np.uint8)
        if self.phases.ndim != 3:
            raise ValueError("phases: expected a 3D array (nx, ny, nz)")
        self.cell_labels = None if labels is None else np.ascontiguousarray(labels, dtype=np.int32)
        if self.cell_labels is not None and self.cell_labels.shape != self.phases.shape:
            raise ValueError("labels: same shape as phases")
        self.orientations = (
            None if orientations is None else np.asarray(orientations, dtype=float).reshape(-1, 3)
        )
        self.smoothing = float(smoothing)
        if self.smoothing < 0:
            raise ValueError("smoothing: must be non-negative")
        self.field = None if field is None else np.ascontiguousarray(field, dtype=np.float32)
        if self.field is not None and self.field.shape != self.phases.shape:
            raise ValueError("field: same shape as phases")
        self.threshold = float(threshold)
        self.attributes = dict(attributes or {})
        self._smoothed = None

    # -- lattice -----------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.phases.shape)

    @property
    def lattice(self) -> Grid:
        return Grid(self.domain, self.shape)

    @property
    def cell_size(self) -> np.ndarray:
        return self.lattice.spacing

    @property
    def binary(self) -> bool:
        return bool(self.phases.max(initial=0) <= 1)

    @property
    def has_labels(self) -> bool:
        return self.cell_labels is not None

    def _level_cells(self) -> np.ndarray:
        """Cell values of the level set: negative in phase 1."""
        if self._smoothed is None:
            if self.field is not None:
                lev = self.threshold - self.field.astype(np.float64)
            else:
                lev = (self.phases == 1).astype(np.float64)
            if self.smoothing > 0:
                sigma = [self.smoothing if n > 1 else 0.0 for n in self.shape]
                mode = "wrap" if self.domain.periodic else "nearest"
                lev = ndimage.gaussian_filter(lev, sigma, mode=mode)
            if self.field is None:
                scale = float(np.mean(self.cell_size)) * max(
                    1.0, math.sqrt(2 * math.pi) * self.smoothing
                )
                lev = (0.5 - lev) * scale  # about the signed distance near the interface
            self._smoothed = np.ascontiguousarray(lev)
        return self._smoothed

    @property
    def lipschitz(self) -> float:
        """The largest slope of the trilinear interpolant of the lattice values."""
        lev = self._level_cells()
        slopes = []
        for a in range(3):
            if lev.shape[a] > 1:
                d = np.abs(np.diff(lev, axis=a))
                slopes.append(float(d.max()) / float(self.cell_size[a]))
        return float(np.sqrt(np.sum(np.square(slopes)))) if slopes else 1.0

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """:meth:`level_at`: the level set of the lattice interpolated at the points."""
        return self.level_at(points)

    def level_at(self, points) -> np.ndarray:
        """Interpolated level set at points (negative in phase 1)."""
        pts = np.ascontiguousarray(np.asarray(points, dtype=float).reshape(-1, 3))
        out = np.empty(len(pts))
        _trilinear(
            self._level_cells(), self.domain.lower, self.cell_size, pts, self.domain.periodic, out
        )
        return out

    def _nearest(self, points) -> tuple[np.ndarray, ...]:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        u = (pts - self.domain.lower) / self.cell_size
        idx = np.floor(u).astype(np.int64)
        n = np.array(self.shape)
        if self.domain.periodic:
            idx %= n
        else:
            idx = np.clip(idx, 0, n - 1)
        return idx[:, 0], idx[:, 1], idx[:, 2]

    def _on_lattice(self, grid: Grid) -> bool:
        return tuple(grid.shape) == self.shape and grid.domain == self.domain

    def _sample(self, grid: Grid, values_of) -> np.ndarray:
        """Evaluate ``values_of(points)`` slab by slab on the voxel centres of ``grid``."""
        nx, ny, nz = grid.shape
        out = None
        step = max(1, (1 << 21) // max(1, nx * ny))
        for k0 in range(0, nz, step):
            k1 = min(nz, k0 + step)
            vals = values_of(grid.center_points(k0, k1))
            if out is None:
                out = np.empty(grid.shape, dtype=vals.dtype)
            out[:, :, k0:k1] = vals.reshape(nx, ny, k1 - k0)
        return out

    # -- discretization ----------------------------------------------------------------

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        if shape is None and voxel_size is None and grid is None:
            grid = self.lattice
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        meta = self.image_metadata()
        if ss == 1 and self._on_lattice(grid):
            return VoxelImage.from_grid(self.phases.copy(), grid, name="phase", metadata=meta)
        if not self.binary:
            if ss > 1:
                raise ValueError("partial volumes are defined for two-phase structures")
            data = self._sample(grid, lambda p: self.phases[self._nearest(p)])
            return VoxelImage.from_grid(data, grid, name="phase", metadata=meta)
        if ss == 1:
            data = self._sample(grid, lambda p: (self.level_at(p) < 0).astype(np.uint8))
            return VoxelImage.from_grid(data, grid, name="phase", metadata=meta)
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        inside = self._sample(fine, lambda p: (self.level_at(p) < 0).astype(np.float32))
        nx, ny, nz = grid.shape
        data = inside.reshape(nx, ss, ny, ss, nz, ss).mean(axis=(1, 3, 5), dtype=np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=meta)

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Interpolated level set, negative in phase 1: ``threshold - field`` for
        structures with a continuous field, otherwise ``(1/2 - indicator) * scale``,
        about the signed distance to the interface near it (``scale`` is the cell size,
        times ``sqrt(2 pi)`` times the smoothing when it exceeds one cell)."""
        if not self.binary:
            raise NotImplementedError("level sets are defined for two-phase structures")
        if shape is None and voxel_size is None and grid is None:
            grid = self.lattice
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        data = self._sample(grid, lambda p: self.level_at(p).astype(np.float32))
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def contains(self, points) -> np.ndarray:
        if self.binary:
            return self.level_at(points) < 0
        return self.phases[self._nearest(points)] == 1

    def labels(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Image of the cell labels (int32 grain indices; -1: no grain), sampled from
        the nearest cell."""
        if self.cell_labels is None:
            raise ValueError("this structure has no labels")
        if shape is None and voxel_size is None and grid is None:
            grid = self.lattice
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        if self._on_lattice(grid):
            data = self.cell_labels.copy()
        else:
            data = self._sample(grid, lambda p: self.cell_labels[self._nearest(p)])
        return VoxelImage.from_grid(data, grid, name="cell", metadata=self.image_metadata())

    # -- description ---------------------------------------------------------------------

    def summary(self) -> dict:
        out = {"lattice": "x".join(str(n) for n in self.shape)}
        if self.cell_labels is not None:
            labels = self.cell_labels
            kind = self.attributes.get("label_kind", "grains")
            out[f"n_{kind}"] = int(np.unique(labels[labels >= 0]).size)
        for key in ("steps", "n_particles"):
            if key in self.attributes:
                out[key] = self.attributes[key]
        return out

    def _geometry(self) -> dict:
        geo = {"phases": _encode(self.phases), "smoothing": self.smoothing}
        if self.field is not None:
            geo["field"] = _encode(self.field)
            geo["threshold"] = self.threshold
        if self.cell_labels is not None:
            geo["labels"] = _encode(self.cell_labels)
        if self.orientations is not None:
            geo["orientations"] = self.orientations.tolist()
        if self.attributes:
            geo["attributes"] = self.attributes
        return geo

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            _decode(geometry["phases"]),
            labels=_decode(geometry["labels"]) if "labels" in geometry else None,
            orientations=geometry.get("orientations"),
            smoothing=geometry.get("smoothing", 0.0),
            field=_decode(geometry["field"]) if "field" in geometry else None,
            threshold=geometry.get("threshold", 0.0),
            attributes=geometry.get("attributes"),
            **kwargs,
        )


def lattice_shape(domain: Domain, cell_size: float, dimension: int = 3) -> tuple[int, int, int]:
    """Number of cells of edge about ``cell_size`` along each axis of the domain
    (one cell through the thickness for 2D structures)."""
    n = np.maximum(1, np.rint(domain.lengths / float(cell_size))).astype(int)
    if dimension == 2:
        n[2] = 1
    return tuple(int(v) for v in n)
