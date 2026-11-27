"""Voronoi and Laguerre (power) tessellations of Poisson germs.

A grain-boundary phase of thickness ``t`` can be added between the cells: the points
closer than ``t / 2`` to a face shared by their cell and another. The faces of the cell
of germ ``g_i`` containing ``x`` lie in the planes where the power distances
``p_j(x) = |x - g_j|^2 - w_j`` of two germs are equal; ``x`` is at distance
``(p_j(x) - p_i(x)) / (2 |g_j - g_i|)`` from the plane of germ ``j``, and the smallest of
these distances is its distance to the boundary of its (convex) cell. The germs whose
plane is closer than ``r`` are among those of lifted distance below ``l_i + 2 r``
(``l_i``: lifted distance to the own germ), so that the nearest germs suffice.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.spatial import cKDTree

from morphogen import settings, theory
from morphogen._utils import check_positive
from morphogen.distributions import Constant, Distribution
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.kernels.collective import stealthy_points
from morphogen.kernels.rsa import random_sequential_addition
from morphogen.models.affine import ANISOTROPY, Anisotropy, anisotropic
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.orientations import (
    OrientationDistribution,
    fibre_orientations,
    random_orientations,
)
from morphogen.seeding import SeedLike, make_rng

#: Points processed per k-d tree query (bounds the memory of large grids).
_CHUNK = 1 << 21


@register_kind
class Tessellation(Microstructure):
    """Cells of germs ``g_i`` with weights ``w_i``: x belongs to the cell minimizing the
    power distance ``|x - g_i|^2 - w_i`` (Voronoi when all weights vanish).

    Each cell carries a phase index and optionally crystal orientations (Bunge Euler
    angles in degrees). The cell label of a voxel is the index of its germ. With a
    ``boundary_thickness``, the points closer than half of it to a face shared by two
    cells are of phase ``boundary_phase`` (they keep the label of their cell).
    """

    kind = "tessellation"
    has_labels = True

    def __init__(
        self,
        domain: Domain,
        germs,
        weights=None,
        cell_phases=None,
        orientations=None,
        boundary_thickness: float = 0.0,
        boundary_phase: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(domain, **kwargs)
        self.boundary_thickness = float(boundary_thickness)
        self.boundary_phase = int(boundary_phase)
        self.germs = np.ascontiguousarray(germs, dtype=float).reshape(-1, 3)
        n = len(self.germs)
        self.weights = None if weights is None else np.asarray(weights, dtype=float).reshape(n)
        self.cell_phases = (
            np.ones(n, dtype=np.uint8)
            if cell_phases is None
            else np.asarray(cell_phases, dtype=np.uint8).reshape(n)
        )
        self.orientations = (
            None if orientations is None else np.asarray(orientations, dtype=float).reshape(n, 3)
        )
        self._tree = None
        self._by_phase = None
        self._complex = None
        self._whole = None
        self._wmax = 0.0 if self.weights is None else float(self.weights.max(initial=0.0))
        self.margin = float(self.metadata.get("germ_margin", math.inf))

    @property
    def n_cells(self) -> int:
        return len(self.germs)

    @property
    def n_grains(self) -> int:
        """The number of grains: the cells (the grains of
        :class:`~morphogen.models.seeded.SeededTessellation` gather several)."""
        return self.n_cells

    def cell_grains(self) -> np.ndarray:
        """The grain of each cell: the cell itself (the grains of
        :class:`~morphogen.models.seeded.SeededTessellation` gather the cells of several
        germs)."""
        return np.arange(self.n_cells, dtype=np.int64)

    def grain_of(self, points) -> np.ndarray:
        """The grain of each point: that of its cell, the cell of least power distance."""
        return self.cell_grains()[self.cell_of(points)]

    def grain_shapes(self) -> dict[str, np.ndarray]:
        """The exact shapes of the grains (the cells; for the grains of seeds the unions of
        the cells of their spheres), from the moments of the exact cells
        (:meth:`~morphogen.mesh.polyhedra.CellComplex.piece_moments`): their ``volume``,
        ``centroid``, equivalent ``diameter`` ``(6 V / π)^(1/3)`` and the ellipsoid of the
        same second moments, ``C = R diag(a², b², c²) Rᵀ / 5`` (``semi_axes`` a >= b >= c,
        ``axes`` the columns of ``R``). The columnar grains of a tessellation of the plane
        (``metadata["dimension"] = 2``: germs in the middle plane z) have their ``area``,
        the ``diameter`` of the disc of that area and the ellipses of the same second moments
        in the plane, ``C = R diag(a², b²) Rᵀ / 4`` (two semi-axes, two axes). In a periodic
        box the whole cells of a grain are gathered about its first one (grains smaller than
        half the box); in a box that is not periodic, ``inside`` tells the grains that its
        faces do not cut (NaN shapes for those outside it)."""
        periodic = bool(self.domain.periodic)
        cx = self.cell_complex(whole=periodic)
        volumes, centres, second = cx.piece_moments()
        n = self.n_grains
        g = np.asarray(self.cell_grains(), dtype=np.int64)[np.asarray(cx.piece_cell)]
        lengths = self.domain.lengths
        if periodic and len(g):  # each grain about the centroid of its first piece
            order = np.argsort(g, kind="stable")
            heads = order[np.r_[0, np.flatnonzero(np.diff(g[order])) + 1]]
            ref = np.zeros((n, 3))
            ref[g[heads]] = centres[heads]
            centres = centres - lengths * np.round((centres - ref[g]) / lengths)
        volume = np.bincount(g, volumes, minlength=n)
        filled = volume > 0
        safe = np.where(filled, volume, 1.0)
        centroid = np.stack([np.bincount(g, volumes * centres[:, i], minlength=n)
                             for i in range(3)], axis=1) / safe[:, None]  # fmt: skip
        d = centres - centroid[g]
        moments = np.zeros((n, 3, 3))
        np.add.at(moments, g, second + volumes[:, None, None] * np.einsum("ni,nj->nij", d, d))
        planar = self.metadata.get("dimension") == 2
        if planar:  # the ellipses in the plane of the sections of the columnar grains
            w, v = np.linalg.eigh(moments[:, :2, :2] / safe[:, None, None])
            semi_axes = np.sqrt(4.0 * np.clip(w[:, ::-1], 0.0, None))
            axes = np.zeros((n, 3, 2))
            axes[:, :2, :] = v[:, :, ::-1]
        else:
            w, v = np.linalg.eigh(moments / safe[:, None, None])
            semi_axes = np.sqrt(5.0 * np.clip(w[:, ::-1], 0.0, None))
            axes = v[:, :, ::-1]
        inside = filled.copy()
        if periodic:
            centroid = self.domain.lower + np.mod(centroid - self.domain.lower, lengths)
        else:  # the grains cut by the faces of the box (but those of the plane, z)
            box = np.asarray(cx.facet_box)
            cut = np.zeros(cx.n_pieces, dtype=bool)
            faces = (box > 0) & (box <= 4) if planar else box > 0
            cut[np.asarray(cx.facet_pieces)[faces, 0]] = True
            inside &= np.bincount(g, cut.astype(float), minlength=n) == 0
        centroid[~filled] = semi_axes[~filled] = np.nan
        axes[~filled] = np.nan
        out = {"volume": volume, "centroid": centroid, "semi_axes": semi_axes, "axes": axes,
               "inside": inside}  # fmt: skip
        if planar:
            out["area"] = volume / float(lengths[2])
            out["diameter"] = np.sqrt(4.0 * out["area"] / math.pi)
        else:
            out["diameter"] = np.cbrt(6.0 * volume / math.pi)
        return out

    # -- cell assignment -------------------------------------------------------------

    def _lift(self, points: np.ndarray) -> np.ndarray:
        """Coordinates for the k-d tree (4D lifting turns power cells into Voronoi cells)."""
        p = points - self.domain.lower
        if self.domain.periodic:
            p = np.mod(p, self.domain.lengths)
            p[p >= self.domain.lengths] = 0.0
        return p

    def _get_tree(self) -> cKDTree:
        if self._tree is None:
            data = self._lift(self.germs)
            boxsize = None
            if self.weights is not None:
                extra = np.sqrt(np.maximum(self._wmax - self.weights, 0.0))[:, None]
                data = np.hstack([data, extra])
            if self.domain.periodic:
                boxsize = list(self.domain.lengths)
                if self.weights is not None:
                    # effectively non-periodic 4th coordinate
                    boxsize.append(4.0 * (math.sqrt(self._wmax) + 1.0) + 1.0)
            self._tree = cKDTree(data, boxsize=boxsize)
        return self._tree

    def cell_of(self, points) -> np.ndarray:
        """Label (germ index) of the cell containing each point."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        tree = self._get_tree()
        labels = np.empty(len(pts), dtype=np.int64)
        worst = 0.0
        for start in range(0, len(pts), _CHUNK):
            q = self._lift(pts[start : start + _CHUNK])
            if self.weights is not None:
                q = np.hstack([q, np.zeros((len(q), 1))])
            dist, idx = tree.query(q, k=1, workers=settings.workers())
            labels[start : start + len(q)] = idx
            if len(q):
                worst = max(worst, float(np.max(dist)))
        self._check_margin(worst)
        return labels

    def boundary_distance(self, points, reach: float = math.inf) -> np.ndarray:
        """Distance from each point to the nearest face shared by its cell and another
        (``inf`` for a single cell); exact up to ``reach``, at least ``reach`` beyond."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        out = np.full(len(pts), np.inf)
        n = self.n_cells
        if n < 2 or not len(pts):
            return out
        tree = self._get_tree()
        lengths = self.domain.lengths
        k0 = min(n, 16)
        for start in range(0, len(pts), max(1, _CHUNK // k0)):
            todo = np.arange(start, min(len(pts), start + max(1, _CHUNK // k0)))
            k = k0
            while todo.size:
                q = self._lift(pts[todo])
                if self.weights is not None:
                    q = np.hstack([q, np.zeros((len(q), 1))])
                dist, idx = tree.query(q, k=k, workers=settings.workers())
                v = self.germs[idx] - pts[todo, None, :]  # germs seen from the points
                if self.domain.periodic:
                    v -= lengths * np.round(v / lengths)
                gap = np.linalg.norm(v[:, 1:] - v[:, :1], axis=2)
                d = (dist[:, 1:] ** 2 - dist[:, :1] ** 2) / (2.0 * np.maximum(gap, 1e-300))
                out[todo] = d.min(axis=1)
                # the farthest germ found must be beyond l_i + 2 min(d, reach)
                bound = dist[:, :1].ravel() + 2.0 * np.minimum(out[todo], reach)
                again = (dist[:, -1] <= bound) & (k < n)
                todo, k = todo[again], min(n, 2 * k)
        return out

    def _check_margin(self, worst: float) -> None:
        """Warn if germs outside the generation window could own some points."""
        if self.domain.periodic or not math.isfinite(self.margin):
            return
        # lifted distance^2 = |x - g|^2 + w_max - w: a germ outside the window (farther than
        # the margin from every point of the domain) has a lifted distance above the margin
        if worst > self.margin:
            warnings.warn(
                "tessellation: some points are farther from their germ than the germ "
                "window margin; cells near the boundary may be inexact",
                RuntimeWarning,
                stacklevel=3,
            )

    def labels(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Image of cell labels (int32 germ indices)."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        nx, ny, nz = grid.shape
        out = np.empty(grid.shape, dtype=np.int32)
        k_step = max(1, _CHUNK // (nx * ny))
        for k0 in range(0, nz, k_step):
            k1 = min(nz, k0 + k_step)
            lab = self.cell_of(grid.center_points(k0, k1))
            out[:, :, k0:k1] = lab.reshape(nx, ny, k1 - k0)
        return VoxelImage.from_grid(out, grid, name="cell", metadata=self.image_metadata())

    def _boundary_image(self, grid: Grid, reach: float) -> np.ndarray:
        """Distance of the voxel centres to the grain boundaries (exact up to ``reach``)."""
        nx, ny, nz = grid.shape
        out = np.empty(grid.shape)
        k_step = max(1, _CHUNK // (16 * nx * ny))
        for k0 in range(0, nz, k_step):
            k1 = min(nz, k0 + k_step)
            d = self.boundary_distance(grid.center_points(k0, k1), reach)
            out[:, :, k0:k1] = d.reshape(nx, ny, k1 - k0)
        return out

    def _phases_on(self, grid: Grid) -> np.ndarray:
        data = self.cell_phases[self.labels(grid=grid).data]
        if self.boundary_thickness > 0:
            half = 0.5 * self.boundary_thickness
            inside = self._boundary_image(grid, half) < half
            data[inside] = self.boundary_phase
        return data

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if ss == 1:
            data = self._phases_on(grid)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        ph = (self._phases_on(fine) == 1).astype(np.float32)
        nx, ny, nz = grid.shape
        data = ph.reshape(nx, ss, ny, ss, nz, ss).mean(axis=(1, 3, 5), dtype=np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def contains(self, points) -> np.ndarray:
        phase = self.cell_phases[self.cell_of(points)]
        if self.boundary_thickness > 0:
            half = 0.5 * self.boundary_thickness
            phase = np.where(
                self.boundary_distance(points, half) < half, self.boundary_phase, phase
            )
        return phase == 1

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Grains of phase 1 in a grain-boundary phase 0: half the thickness of the
        boundaries minus the distance to them (negative in the grains); the cells of
        phase 1 among cells of other phases: :meth:`phase_level_set`."""
        if (
            self.boundary_thickness > 0
            and self.boundary_phase == 0
            and np.all(self.cell_phases == 1)
        ):
            grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
            half = 0.5 * self.boundary_thickness
            reach = half + 4.0 * float(np.max(grid.spacing))  # exact around the interface
            data = (half - self._boundary_image(grid, reach)).astype(np.float32)
            return VoxelImage.from_grid(
                data, grid, name="level_set", metadata=self.image_metadata()
            )
        if 1 in self.implicit_phases() and len(self.implicit_phases()) > 1:
            return self.phase_level_set(1, shape, voxel_size=voxel_size, grid=grid)
        return super().level_set(shape, voxel_size=voxel_size, grid=grid)

    def implicit_phases(self) -> list[int]:
        phases = {int(k) for k in np.unique(self.cell_phases)}
        if self.boundary_thickness > 0:
            phases.add(self.boundary_phase)
        return sorted(phases)

    def _phase_trees(self) -> dict:
        """A k-d tree of the lifted germs of each phase of the cells."""
        if getattr(self, "_by_phase", None) is None:
            tree = self._get_tree()
            self._by_phase = {
                int(k): cKDTree(tree.data[self.cell_phases == k], boxsize=tree.boxsize)
                for k in np.unique(self.cell_phases)
            }
        return self._by_phase

    def phase_level_set(
        self, phase: int, shape=None, *, voxel_size=None, grid: Grid | None = None
    ) -> VoxelImage:
        """The implicit function of a phase (see :meth:`Microstructure.phase_level_set`):
        for the cells of a phase, the smallest power distance to their germs minus the
        smallest to the germs of the cells of the other phases, over twice the mean
        spacing of the germs: zero on the faces between cells of the phase and cells of
        another (the planes where the two power distances are equal), near which it is
        the distance to the face times the ratio of the distance between their germs to
        the mean spacing. A grain-boundary phase is the layer within half its thickness
        of the faces of the cells, which the cells lose."""
        phase = int(phase)
        phases = self.implicit_phases()
        if phase not in phases:
            raise ValueError(f"phase {phase}: the phases of this tessellation are {phases}")
        from morphogen.models.implicit import band_of

        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        band = band_of(grid)
        trees = self._phase_trees()
        scale = 2.0 * (self.domain.volume / max(self.n_cells, 1)) ** (1.0 / 3.0)
        nx, ny, nz = grid.shape
        cells = None
        if phase in trees:
            cells = np.full(grid.shape, -band, dtype=np.float32)
            if len(trees) > 1:
                k_step = max(1, _CHUNK // (nx * ny))
                for k0 in range(0, nz, k_step):
                    k1 = min(nz, k0 + k_step)
                    q = self._lift(grid.center_points(k0, k1))
                    if self.weights is not None:
                        q = np.hstack([q, np.zeros((len(q), 1))])
                    own = other = None
                    for k, tree in trees.items():
                        d2 = tree.query(q, k=1, workers=settings.workers())[0] ** 2
                        if k == phase:
                            own = d2
                        else:
                            other = d2 if other is None else np.minimum(other, d2)
                    cells[:, :, k0:k1] = ((own - other) / scale).reshape(nx, ny, k1 - k0)
                np.clip(cells, -band, band, out=cells)
        data = cells
        if self.boundary_thickness > 0:
            half = 0.5 * self.boundary_thickness
            layer = (self._boundary_image(grid, half + band) - half).astype(np.float32)
            np.clip(layer, -band, band, out=layer)
            if phase == self.boundary_phase:
                data = layer if cells is None else np.minimum(cells, layer)
            else:
                data = np.maximum(cells, -layer)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """Grains of phase 1 in a grain-boundary phase 0: half the thickness of the
        boundaries minus the distance to them at the points; the other tessellations of
        several phases: the function of phase 1 (:meth:`phase_level_sets_at`)."""
        if not (self.boundary_thickness > 0 and self.boundary_phase == 0
                and np.all(self.cell_phases == 1)):  # fmt: skip
            if 1 not in self.implicit_phases():
                return super().level_set_at(points, reach)
            return self.phase_level_sets_at(points, reach)[1]
        half = 0.5 * self.boundary_thickness
        reach = float(min(reach, float(self.domain.diagonal))) + half
        return half - self.boundary_distance(np.asarray(points, dtype=float).reshape(-1, 3), reach)

    @property
    def lipschitz(self) -> float:
        """A bound of the slopes of :meth:`phase_level_sets_at`: the function of the cells
        of a phase, the difference of the squares of the distances to the nearest germs of
        two phases over twice the mean spacing ``l``, has the slope ``|c_o - c_k| / l``,
        the distance of those germs over ``l``, about two for neighbouring cells (1.97
        measured on a mosaic): 2 for a mosaic of several phases, 1 for the layer of a
        grain-boundary phase alone."""
        return 2.0 if len(self._phase_trees()) > 1 else 1.0

    def phase_level_sets_at(self, points, reach: float = np.inf) -> dict[int, np.ndarray]:
        """The functions of :meth:`phase_level_set` at the points: for the cells of a
        phase, the smallest power distance to their germs minus the smallest to the germs
        of the other phases, over twice the mean spacing of the germs (zero on the faces
        between cells of the phase and cells of another); a grain-boundary phase, the layer
        within half its thickness of the faces of the cells. A tessellation of one phase
        without a boundary phase has no interface (and none: its exact geometry is that of
        its cells)."""
        phases = self.implicit_phases()
        if len(phases) < 2:
            raise NotImplementedError("a tessellation of one phase has no interface")
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        spacing = (self.domain.volume / max(self.n_cells, 1)) ** (1.0 / 3.0)
        reach = float(spacing) if not np.isfinite(reach) else float(reach)
        trees = self._phase_trees()
        cells: dict[int, np.ndarray] = {}
        if len(trees) > 1:
            q = self._lift(pts)
            if self.weights is not None:
                q = np.hstack([q, np.zeros((len(q), 1))])
            d2 = {k: tree.query(q, k=1, workers=settings.workers())[0] ** 2
                  for k, tree in trees.items()}  # fmt: skip
            for k in trees:
                other = np.minimum.reduce([d2[j] for j in trees if j != k])
                cells[k] = np.clip((d2[k] - other) / (2.0 * spacing), -reach, reach)
        else:
            cells = {k: np.full(len(pts), -reach) for k in trees}
        if not self.boundary_thickness > 0:
            return {k: cells[k] for k in phases}
        half = 0.5 * self.boundary_thickness
        layer = np.clip(self.boundary_distance(pts, half + reach) - half, -reach, reach)
        out = {}
        for k in phases:
            if k == self.boundary_phase:
                out[k] = layer if k not in cells else np.minimum(cells[k], layer)
            else:
                out[k] = np.maximum(cells[k], -layer)
        return out

    def cell_complex(self, whole: bool = False):
        """Exact cells restricted to the domain: convex pieces with shared facets
        (:class:`morphogen.mesh.polyhedra.CellComplex`), computed once. With ``whole``
        (periodic tessellations), the whole cells, uncut by the faces of the box."""
        from morphogen.mesh.polyhedra import periodic_cells, power_complex

        if whole:
            if not self.domain.periodic:
                raise ValueError("whole cells: a periodic tessellation")
            if self._whole is None:
                self._whole = periodic_cells(
                    self.germs, self.weights, self.domain.lower, self.domain.upper
                )
            return self._whole
        if self._complex is None:
            self._complex = power_complex(
                self.germs, self.weights, self.domain.lower, self.domain.upper, self.domain.periodic
            )
        return self._complex

    # -- description -------------------------------------------------------------------

    def cell_table(self) -> dict[str, np.ndarray]:
        """Per-cell attributes: id, germ coordinates, weight, phase, orientation."""
        table = {
            "id": np.arange(self.n_cells),
            "x": self.germs[:, 0],
            "y": self.germs[:, 1],
            "z": self.germs[:, 2],
            "phase": self.cell_phases.astype(int),
        }
        if self.weights is not None:
            table["weight"] = self.weights
        if self.orientations is not None:
            table["phi1"] = self.orientations[:, 0]
            table["Phi"] = self.orientations[:, 1]
            table["phi2"] = self.orientations[:, 2]
        return table

    def summary(self) -> dict:
        return {"n_cells": self.n_cells}

    def _geometry(self) -> dict:
        geo = {"germs": self.germs.tolist(), "cell_phases": self.cell_phases.tolist()}
        if self.weights is not None:
            geo["weights"] = self.weights.tolist()
        if self.orientations is not None:
            geo["orientations"] = self.orientations.tolist()
        if self.boundary_thickness > 0:
            geo["boundary_thickness"] = self.boundary_thickness
            geo["boundary_phase"] = self.boundary_phase
        return geo

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            geometry["germs"],
            geometry.get("weights"),
            geometry.get("cell_phases"),
            geometry.get("orientations"),
            boundary_thickness=geometry.get("boundary_thickness", 0.0),
            boundary_phase=geometry.get("boundary_phase", 0),
            **kwargs,
        )


@dataclass(frozen=True)
class _TessellationModel(Model):
    phases: ClassVar[dict[int, str]] = {0: "phase 0", 1: "phase 1"}
    count_label = "cells"
    #: (the orientation map of a measured texture)
    path_parameters: ClassVar[tuple[str, ...]] = ("texture.file",)

    n_cells: int | None = param(
        None,
        doc="Number of cells: exactly this many germs in a periodic box, Poisson germs "
        "of intensity n_cells / volume otherwise",
        kind="int",
    )
    intensity: float | None = param(None, doc="Germs per unit volume", unit="1/length^3")
    phase_fractions: tuple[float, ...] | None = param(
        None,
        doc="Probabilities of phases 0, 1, ... for each cell (default: every cell is phase 1)",
        kind="floats",
    )
    crystal_orientations: bool = param(
        False, doc="Draw a uniformly random crystal orientation per cell", kind="bool"
    )
    texture: OrientationDistribution | None = param(
        None,
        doc="The texture of the crystal orientations. A fibre texture: the law of the "
        "third axis of the crystals (the axis c of hexagonal ones), which are turned at "
        "random about it, e.g. {type: von_mises_fisher, kappa: 30, axis: [0, 0, 1]}. A "
        "measured texture: the orientations of the grains of an orientation map (EBSD), "
        "drawn with their areas, e.g. {type: measured, file: map.ctf, spread: 2.0} "
        "(default: uniform orientations)",
        kind="orientation",
        kw_only=True,
    )
    boundary_thickness: float = param(
        0.0,
        doc="Thickness of a grain-boundary phase: the points closer than half of it to a "
        "face shared by two cells (0: none)",
        unit="length",
    )
    boundary_phase: int | None = param(
        None,
        doc="Phase of the grain boundaries (default: 0 when every cell is phase 1, the "
        "phase after those of the cells otherwise)",
        kind="int",
    )

    def _validate(self) -> None:
        if (self.n_cells is None) == (self.intensity is None):
            raise ValueError("give exactly one of 'n_cells' and 'intensity'")
        if self.n_cells is not None and self.n_cells < 1:
            raise ValueError("n_cells: must be a positive integer")
        if self.intensity is not None:
            check_positive(self.intensity, "intensity")
        if self.phase_fractions is not None:
            p = np.asarray(self.phase_fractions, dtype=float)
            if p.ndim != 1 or len(p) < 1 or np.any(p < 0) or abs(p.sum() - 1.0) > 1e-9:
                raise ValueError("phase_fractions: non-negative probabilities summing to 1")
            if len(p) > 255:
                raise ValueError("phase_fractions: at most 255 phases")
        check_positive(self.boundary_thickness, "boundary_thickness", allow_zero=True)
        if self.boundary_phase is not None and not 0 <= self.boundary_phase <= 255:
            raise ValueError("boundary_phase: a phase index from 0 to 255")

    @property
    def grain_boundary_phase(self) -> int | None:
        """Phase of the grain boundaries (None without them)."""
        if not self.boundary_thickness > 0:
            return None
        if self.boundary_phase is not None:
            return self.boundary_phase
        return 0 if self.phase_fractions is None else len(self.phase_fractions)

    @property
    def has_level_set(self) -> bool:
        """Grains of phase 1 in a grain-boundary phase 0 have a level-set function."""
        return self.phase_fractions is None and self.grain_boundary_phase == 0

    @property
    def meshed_by_grains(self) -> bool:
        """Cells meshed grain by grain, unless a grain-boundary phase separates them or an
        anisotropy maps them (their voxel images only)."""
        mapped = not getattr(getattr(self, "anisotropy", None), "is_identity", True)
        return self.grain_boundary_phase is None and not mapped

    def active_phases(self) -> dict[int, str]:
        if self.phase_fractions is None:
            out = {1: "cells"}
        else:
            out = {i: f"phase {i}" for i, p in enumerate(self.phase_fractions) if p > 0}
        if self.grain_boundary_phase is not None:
            out[self.grain_boundary_phase] = "grain boundaries"
        return out

    def theta(self, domain: Domain | None) -> float:
        """Intensity of the germs (``n_cells / volume`` when the number is given)."""
        if self.intensity is not None:
            return self.intensity
        if domain is None:
            raise ValueError("the intensity implied by n_cells depends on the domain")
        return self.n_cells / domain.volume

    def expected_count(self, domain: Domain) -> float:
        """Mean number of germs (cells) in the domain."""
        return self.theta(domain) * domain.volume

    def _germs(self, rng, window: Domain, n: int):
        """Germs (uniform in the window) and their weights."""
        return window.lower + rng.random((n, 3)) * window.lengths, self._weights(rng, n)

    def _count(self, rng, mean: float) -> int:
        """The number of germs of mean ``mean`` in a window (Poisson)."""
        return int(rng.poisson(mean))

    def _weights(self, rng, n):
        return None

    def _max_weight(self, theta: float) -> float:
        return 0.0

    def generate(self, domain: Domain, seed: SeedLike = None) -> Tessellation:
        rng, seed_info = make_rng(seed)
        theta = self.theta(domain)
        if domain.periodic:
            window, margin = domain, math.inf
            mean = theta * domain.volume
            n = self.n_cells if self.n_cells is not None else self._count(rng, mean)
        else:
            margin = 3.0 * theta ** (-1.0 / 3.0) + 2.0 * math.sqrt(self._max_weight(theta))
            window = domain.dilated(margin)
            n = self._count(rng, theta * window.volume)
        germs, weights = self._germs(rng, window, n)
        if self.phase_fractions is None:
            phases = np.ones(n, dtype=np.uint8)
        else:
            p = np.asarray(self.phase_fractions, dtype=float)
            phases = rng.choice(len(p), size=n, p=p / p.sum()).astype(np.uint8)
        orientations = None
        if self.crystal_orientations and self.texture is not None:
            orientations = fibre_orientations(rng, n, self.texture)
        elif self.crystal_orientations:
            orientations = random_orientations(rng, n)
        return Tessellation(
            domain,
            germs,
            weights,
            phases,
            orientations,
            boundary_thickness=self.boundary_thickness,
            boundary_phase=self.grain_boundary_phase or 0,
            model=self,
            seed=seed_info,
            metadata={"germ_margin": margin} if math.isfinite(margin) else None,
        )


@register
@anisotropic
@dataclass(frozen=True)
class Voronoi(_TessellationModel):
    """Voronoi tessellation (mosaic): each point belongs to its nearest germ. The germs
    are Poisson points, or a stealthy hyperuniform pattern (``germs: stealthy``: the
    collective coordinates of Torquato and coworkers, :mod:`morphogen.kernels.collective`),
    a fixed number of them in the box, whose cells are disordered but far more regular."""

    name = "voronoi"

    germs: str = param(
        "poisson",
        doc="Germs: poisson (independent points) or stealthy (a stealthy hyperuniform pattern)",
        kind="str",
    )
    stealthiness: float = param(
        0.3,
        doc="Stealthy germs: the constrained wave vectors per degree of freedom (chi, below "
        "0.5: disordered ground states; more: more regular cells)",
        kind="fraction",
    )
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        super()._validate()
        if self.germs not in ("poisson", "stealthy"):
            raise ValueError(f"germs: poisson or stealthy, got {self.germs!r}")
        if not 0.0 < self.stealthiness < 0.5:
            raise ValueError("stealthiness: in (0, 0.5), the disordered ground states")

    def _count(self, rng, mean: float) -> int:
        return super()._count(rng, mean) if self.germs == "poisson" else round(mean)

    def _germs(self, rng, window: Domain, n: int):
        if self.germs == "poisson":
            return super()._germs(rng, window, n)
        points, _, _ = stealthy_points(n, window.lengths, self.stealthiness, rng)
        return window.lower + points, None

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        if self.germs == "stealthy":  # the constants of the Poisson germs do not hold
            out = {}
            if self.intensity is not None or domain is not None:
                theta = self.theta(domain)
                out.update(intensity=theta, mean_cell_volume=1.0 / theta)
            if self.phase_fractions is not None and not self.boundary_thickness > 0:
                for i, p in enumerate(self.phase_fractions):
                    out[f"volume_fraction_{i}"] = float(p)
            return out
        out = {
            "mean_faces": theory.VORONOI_MEAN_FACES,
            "cell_volume_cv2": theory.VORONOI_VOLUME_CV2,
        }
        if self.intensity is not None or domain is not None:
            theta = self.theta(domain)
            out.update(
                {
                    "intensity": theta,
                    "mean_cell_volume": 1.0 / theta,
                    "integral_range": (1.0 + theory.VORONOI_VOLUME_CV2) / theta,
                    "boundary_surface_density": theory.VORONOI_SURFACE_COEFF * theta ** (1 / 3),
                }
            )
            if self.boundary_thickness > 0:  # S_V t: exact to first order in t
                out["boundary_fraction_first_order"] = (
                    out["boundary_surface_density"] * self.boundary_thickness
                )
        else:
            out["integral_range_in_cells"] = 1.0 + theory.VORONOI_VOLUME_CV2
        if self.phase_fractions is not None and not self.boundary_thickness > 0:
            for i, p in enumerate(self.phase_fractions):
                out[f"volume_fraction_{i}"] = float(p)
        return out


@register
@anisotropic
@dataclass(frozen=True)
class Laguerre(_TessellationModel):
    """Laguerre (power) tessellation of germ spheres, the weights being their ``r^2``.

    The germs are Poisson points with independent radii, or the centres of a packing of
    non-overlapping spheres (random sequential addition, the largest first) whose radii
    follow the distribution up to a common factor filling ``packing_fraction`` of the
    box: the cells then follow the sizes of the spheres closely (e.g. Fan et al. 2004).
    """

    name = "laguerre"

    radius: Distribution = param(
        Constant(1.0),
        doc="Radius r of the germ spheres (weight r^2; relative sizes for packed germs)",
        kind="distribution",
        unit="length",
    )
    germs: str = param(
        "poisson", doc="Germs: poisson (independent points) or packing (hard spheres)", kind="str"
    )
    packing_fraction: float = param(
        0.3,
        doc="Packed germs: volume fraction of the spheres (random sequential addition jams "
        "near 0.38 for equal spheres)",
        kind="fraction",
    )
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        super()._validate()
        if self.germs not in ("poisson", "packing"):
            raise ValueError(f"germs: poisson or packing, got {self.germs!r}")
        if not 0.0 < self.packing_fraction < 0.6:
            raise ValueError("packing_fraction: between 0 and 0.6")

    def _scale(self, theta: float) -> float:
        """Factor of the radii of packed germs (expected value, given the intensity)."""
        return (self.packing_fraction / (theta * 4.0 / 3.0 * math.pi * self.radius.moment(3))) ** (
            1.0 / 3.0
        )

    def _germs(self, rng, window: Domain, n: int):
        if self.germs == "poisson":
            return super()._germs(rng, window, n)
        radii = np.sort(self.radius.sample(rng, n))[::-1]
        if n:
            radii = radii * (
                self.packing_fraction * window.volume / (4.0 / 3.0 * math.pi * np.sum(radii**3))
            ) ** (1.0 / 3.0)
        if n and 2.0 * radii[0] >= 0.5 * float(np.min(window.lengths)):
            raise ValueError("laguerre: the packed spheres are too large for the box")
        local = random_sequential_addition(radii, window.lengths, 0.0, rng, 10_000 * max(n, 1))
        return window.lower + local, radii**2

    def _weights(self, rng, n):
        return self.radius.sample(rng, n) ** 2

    def _max_weight(self, theta: float) -> float:
        if self.germs == "packing":  # cells close to their spheres: a few mean radii
            return (3.0 * self._scale(theta) * self.radius.mean()) ** 2
        return self.radius.upper() ** 2
