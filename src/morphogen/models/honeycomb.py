"""Honeycombs and wood: prismatic cells with walls.

Prismatic cellular solids, the honeycombs of bees and the cells of wood, of cork or of
plant stems, are stiff and strong along their axis for their weight, and compliant
across it (Gibson and Ashby 1997). The model extrudes along an axis a tessellation of
the plane of germs whose cells are bounded by walls of thickness ``t``: the points of a
cell closer than ``t / 2`` to its sides (a wall between two cells is the sum of their
halves). The germs are

* ``hexagonal`` or ``square``: lattices whose cells are hexagons ``d`` wide between
  opposite walls, or squares of side ``d``, displaced at random by up to ``disorder``
  cell (a periodic box holds whole cells: the lattice stretches to fit it);
* ``voronoi``: points of a random sequential addition at least ``regularity`` jamming
  distance apart (Poisson points for 0), one per area ``sqrt(3) d^2 / 2`` of a hexagon;
* ``wood``: the radial files of softwoods, rectangular cells rather than the cells of
  germs. Along the tangential axis (the second of the plane), files ``d`` wide whose
  radial walls are straight; along the radial axis (the first), cells whose size and
  wall thickness vary through the growth rings of width ``ring_width``: from large
  thin-walled earlywood cells (``d`` across) to small thick-walled latewood cells
  (``latewood_size`` and ``latewood_wall`` times those of earlywood, in the last
  ``latewood_fraction`` of the ring, after a transition of half that width); each file
  is shifted by a random fraction of its cells, so that the tangential walls are
  staggered. Loaded radially, the straight radial walls carry the load; tangentially,
  the radial walls bend between the staggered tangential ones: with uniform cells the
  radial modulus is several times the tangential one. Dense latewood bands, continuous
  along the tangential axis, stiffen it.

  Rays (``ray_fraction`` of the volume) are ribbons of parenchyma cells that run
  radially across the rings between two files: ``ray_width`` cells across (1: the
  uniseriate rays of softwoods, more for the multiseriate rays of hardwoods) and
  ``ray_height`` high along the grain, at random places, never closer than one ray cell
  along the grain to another ray between the same files or between neighbouring ones.
  Their cells are boxes ``ray_cell_length`` long radially, ``ray_cell_width`` wide and
  ``ray_cell_height`` high, with walls ``ray_wall`` thick. The tracheids pass beside
  the rays: over the height of a ray, its walls replace the radial wall between the two
  files, narrowed by half its width each. Rays make the structure three-dimensional.

Walls of uniform thickness ``t`` around hexagons or squares of width ``d`` take the
fraction ``1 - (1 - t / d)^2`` of the volume, exactly; around the cells of Poisson germs,
``t L_A = 2 t sqrt(lambda)`` to first order in ``t`` (``L_A``: length of the sides per
unit area, ``lambda``: germs per unit area).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen._utils import check_positive
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.models.planar import (
    PlanarCells,
    axis_centres,
    hardcore_germs,
    jamming_distance,
    pixel_centres,
    plane_axes,
)
from morphogen.seeding import SeedLike, make_rng

ARRANGEMENTS = ("hexagonal", "square", "voronoi", "wood")


class RadialFiles:
    """Rectangular cells in files along the first axis of the plane: file ``j`` spans
    ``[origin + j width, origin + (j + 1) width)`` across, its cells the intervals between
    its consecutive ``bounds[j]`` along it (on a periodic box, one period: the last bound
    is the first plus the length of the box)."""

    def __init__(self, origin, width, bounds, lengths, periodic: bool, lower) -> None:
        self.origin = float(origin)
        self.width = float(width)
        self.bounds = [np.asarray(b, dtype=float).reshape(-1) for b in bounds]
        self.lengths = np.asarray(lengths, dtype=float).reshape(2)
        self.lower = np.asarray(lower, dtype=float).reshape(2)
        self.periodic = bool(periodic)
        self.first = np.cumsum([0] + [len(b) - 1 for b in self.bounds])  # cell ids

    @property
    def n_files(self) -> int:
        return len(self.bounds)

    def _wrap(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        if self.periodic:
            p = self.lower + np.mod(p - self.lower, self.lengths)
        return p

    def parts(self, points):
        """Cell and file ``j`` of each point, and its distances to the sides of the cell:
        along the file, and across it to its radial walls below and above (on the
        boundaries ``j`` and ``j + 1`` of the files)."""
        p = self._wrap(points)
        n = len(self.bounds)
        y = p[:, 1] - self.origin
        j = np.floor(y / self.width).astype(np.int64)
        below = np.clip(y - j * self.width, 0.0, self.width)
        j = np.mod(j, n) if self.periodic else np.clip(j, 0, n - 1)
        labels = np.zeros(len(p), dtype=np.int64)
        along = np.zeros(len(p))
        for f in np.unique(j):
            sel = j == f
            b = self.bounds[f]
            u = p[sel, 0]
            if self.periodic:  # into the period of the file
                u = b[0] + np.mod(u - b[0], b[-1] - b[0])
            k = np.clip(np.searchsorted(b, u, side="right") - 1, 0, len(b) - 2)
            along[sel] = np.minimum(u - b[k], b[k + 1] - u)
            labels[sel] = self.first[f] + k
        return labels, j, along, below, self.width - below

    def above(self, j: np.ndarray) -> np.ndarray:
        """Boundary above the files ``j`` (the first one again on a periodic box)."""
        return np.mod(j + 1, len(self.bounds)) if self.periodic else j + 1

    def locate(self, points, reach: float = math.inf) -> tuple[np.ndarray, np.ndarray]:
        """Cell of each point and its distance to the sides of the cell (exact)."""
        labels, _, along, below, above = self.parts(points)
        return labels, np.minimum(along, np.minimum(below, above))


class WoodRays:
    """Rays between radial files: ray ``r`` lies on the boundary ``boundary[r]`` of the
    files (the radial wall between the files ``boundary[r] - 1`` and ``boundary[r]``),
    ``width`` wide across it, from ``z[r]`` to ``z[r] + height`` along the grain (the prism
    axis, of ``period`` on a periodic box). Its cells are boxes ``cell_length`` long
    radially (their boundaries at ``u[r]`` plus multiples of it), ``width / cells`` wide
    and ``cell_height`` high, with walls ``thickness`` thick."""

    def __init__(self, boundary, z, u, cells, width, height, cell_length, cell_height,
                 thickness, period=None) -> None:  # fmt: skip
        self.boundary = np.asarray(boundary, dtype=np.int64).reshape(-1)
        self.z = np.asarray(z, dtype=float).reshape(-1)
        self.u = np.asarray(u, dtype=float).reshape(-1)
        self.cells = int(cells)
        self.width = float(width)
        self.height = float(height)
        self.cell_length = float(cell_length)
        self.cell_height = float(cell_height)
        self.thickness = float(thickness)
        self.period = None if period is None else float(period)

    def __len__(self) -> int:
        return len(self.z)

    def locate(self, boundary: np.ndarray, z: np.ndarray):
        """For points given by a boundary of the files and their coordinate along the
        grain: the ray on the boundary at their height (-1: none), their height in it, and
        their distance along the grain to the rays of the boundary (0 within the height of
        one, inf: none)."""
        ray = np.full(len(z), -1, dtype=np.int64)
        rel = np.zeros(len(z))
        gap = np.full(len(z), np.inf)
        for r in range(len(self)):
            on = boundary == self.boundary[r]
            if self.period is None:
                dz = z - self.z[r]
                out = np.where(dz < 0.0, -dz, np.maximum(dz - self.height, 0.0))
            else:
                dz = np.mod(z - self.z[r], self.period)
                out = np.where(
                    dz < self.height, 0.0, np.minimum(dz - self.height, self.period - dz)
                )
            inside = on & (dz >= 0.0) & (dz < self.height)
            ray[inside] = r
            rel[inside] = dz[inside]
            gap[on] = np.minimum(gap[on], out[on])
        return ray, rel, gap

    def cell_distance(self, ray: np.ndarray, u: np.ndarray, rel: np.ndarray, across):
        """Distance of points in rays to the walls of their cells: radially, along the
        grain and across (``across``: the position of the points across their ray, from 0
        to its width)."""
        w = self.width / self.cells
        dy = np.mod(across, w)
        du = np.mod(u - self.u[ray], self.cell_length)
        dz = np.mod(rel, self.cell_height)
        return np.minimum.reduce(
            [np.minimum(dy, w - dy), np.minimum(du, self.cell_length - du),
             np.minimum(dz, self.cell_height - dz)]
        )  # fmt: skip

    def to_dict(self) -> dict:
        return {
            "boundary": self.boundary.tolist(),
            "z": self.z.tolist(),
            "u": self.u.tolist(),
            "cells": self.cells,
            "width": self.width,
            "height": self.height,
            "cell_length": self.cell_length,
            "cell_height": self.cell_height,
            "thickness": self.thickness,
        }


@register_kind
class PrismaticCells(Microstructure):
    """Prisms along ``axis`` over cells of the plane: the walls (phase 1) are the points
    closer to the sides of their cell than half its wall thickness ``walls[i]``, the
    lumens of the cells phase 0. The cells are those of germs (Voronoi), or rectangles
    in radial ``files`` (wood; the germs are then their centres), between which rays may
    run (``files["rays"]``, :class:`WoodRays`): the structure is then three-dimensional."""

    kind = "prismatic_cells"

    def __init__(self, domain, axis, germs, walls, files=None, **kwargs):
        super().__init__(domain, **kwargs)
        self.axis = int(axis)
        self.germs = np.asarray(germs, dtype=float).reshape(-1, 2)
        self.walls = np.broadcast_to(np.asarray(walls, dtype=float), len(self.germs)).copy()
        self.files = files
        b, c = plane_axes(self.axis)
        d = self.domain
        lengths, lower = d.lengths[[b, c]], d.lower[[b, c]]
        if files is None:
            self.cells = PlanarCells(self.germs, lengths, d.periodic, lower)
        else:
            self.cells = RadialFiles(
                files["origin"], files["width"], files["bounds"], lengths, d.periodic, lower
            )
            if self.cells.first[-1] != len(self.germs):
                raise ValueError("prismatic cells: one germ per cell of the files")
        self.rays = None
        if files is not None and files.get("rays"):
            period = float(d.lengths[self.axis]) if d.periodic else None
            self.rays = WoodRays(**files["rays"], period=period)

    @property
    def n_cells(self) -> int:
        return len(self.germs)

    @property
    def n_rays(self) -> int:
        return 0 if self.rays is None else len(self.rays)

    @property
    def ray_fraction(self) -> float:
        """Fraction of the box in rays (their cells and walls), exact: rays run across the
        box radially, so that it is their share of the tangential plane (across the files
        and along the grain)."""
        if self.rays is None:
            return 0.0
        r, f, d, ax = self.rays, self.cells, self.domain, self.axis
        c = plane_axes(ax)[1]
        if d.periodic:
            return len(r) * r.width * r.height / float(d.lengths[c] * d.lengths[ax])

        def overlap(lo, hi, a, b):
            return np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)

        centre = f.origin + r.boundary * f.width
        across = overlap(centre - 0.5 * r.width, centre + 0.5 * r.width, d.lower[c], d.upper[c])
        along = overlap(r.z, r.z + r.height, d.lower[ax], d.upper[ax])
        return float(np.sum(across * along) / (d.lengths[c] * d.lengths[ax]))

    def _beside_rays(self, along, below, above, half, u, lower, upper) -> np.ndarray:
        """Signed field at points of the files: ``along`` their distance to the sides of
        their cells along the file, ``below`` and ``above`` to its radial walls, ``half``
        the half walls of their cells, ``u`` their radial coordinate, and ``lower`` and
        ``upper`` the rays on these walls at their height (:meth:`WoodRays.locate`). Over
        its height, a ray replaces the radial wall and narrows the files beside it; in a
        ray, the field is the distance to the walls of its cells."""
        r = self.rays
        hw = 0.5 * r.width
        (ray_lo, rel_lo, gap_lo), (ray_hi, rel_hi, gap_hi) = lower, upper
        out = np.minimum.reduce(
            [along, np.where(ray_lo >= 0, np.inf, below), np.where(ray_hi >= 0, np.inf, above),
             np.hypot(np.maximum(below - hw, 0.0), gap_lo),
             np.hypot(np.maximum(above - hw, 0.0), gap_hi)]
        ) - half  # fmt: skip
        for ray, rel, side, across in (
            (ray_lo, rel_lo, below, hw + below),
            (ray_hi, rel_hi, above, hw - above),
        ):
            inside = (ray >= 0) & (side < hw)
            if inside.any():
                dist = r.cell_distance(ray[inside], u[inside], rel[inside], across[inside])
                out[inside] = dist - 0.5 * r.thickness
        return out

    def _signed(self, plane: np.ndarray, reach: float, z=None) -> np.ndarray:
        """Distance to the sides of the cell minus half its wall (negative in walls), at
        points of the plane, or of space (``z``: along the axis) when rays run between the
        files."""
        if self.rays is None or z is None:
            label, d = self.cells.locate(plane, reach)
            return d - 0.5 * self.walls[label]
        plane = np.asarray(plane, dtype=float).reshape(-1, 2)
        label, j, along, below, above = self.cells.parts(plane)
        z = np.asarray(z, dtype=float).reshape(-1)
        lower, upper = self.rays.locate(j, z), self.rays.locate(self.cells.above(j), z)
        half = 0.5 * self.walls[label]
        return self._beside_rays(along, below, above, half, plane[:, 0], lower, upper)

    def _field(self, grid: Grid, reach: float) -> np.ndarray:
        """Signed field on the voxels: of the plane (extruded by the callers) without
        rays, of space with them (slice by slice along the axis)."""
        plane, shape = pixel_centres(grid, plane_axes(self.axis))
        if self.rays is None:
            label, d = self.cells.locate(plane, reach)
            return (d - 0.5 * self.walls[label]).astype(np.float32).reshape(shape)
        label, j, along, below, above = self.cells.parts(plane)
        half, up = 0.5 * self.walls[label], self.cells.above(j)
        boundaries = np.arange(self.cells.n_files + 1)
        z = axis_centres(grid, self.axis)
        out = np.empty((len(z), len(plane)), dtype=np.float32)
        for i, zi in enumerate(z):
            ray, rel, gap = self.rays.locate(boundaries, np.full(len(boundaries), zi))
            lower, upper = (ray[j], rel[j], gap[j]), (ray[up], rel[up], gap[up])
            out[i] = self._beside_rays(along, below, above, half, plane[:, 0], lower, upper)
        return np.moveaxis(out.reshape(len(z), *shape), 0, self.axis)

    def _extruded(self, image2d: np.ndarray, grid: Grid) -> np.ndarray:
        out = np.broadcast_to(image2d[:, :, None], (*image2d.shape, grid.shape[self.axis]))
        return np.ascontiguousarray(np.moveaxis(out, 2, self.axis))

    def _space_field(self, grid: Grid, reach: float) -> np.ndarray:
        field = self._field(grid, reach)
        return field if field.ndim == 3 else self._extruded(field, grid)

    @property
    def _reach(self) -> float:
        thickest = float(self.walls.max())
        return 0.5 * max(thickest, 0.0 if self.rays is None else self.rays.thickness)

    def contains(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        b, c = plane_axes(self.axis)
        out = self._signed(pts[:, [b, c]], self._reach, pts[:, self.axis]) < 0
        if not self.domain.periodic:
            out &= np.all((pts >= self.domain.lower) & (pts <= self.domain.upper), axis=1)
        return out

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """The distance to the sides of the cells minus half their walls at the points
        (exact within ``reach`` of the walls, at least ``reach`` beyond)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        b, c = plane_axes(self.axis)
        reach = float(min(reach, float(np.max(self.domain.lengths))))
        return self._signed(pts[:, [b, c]], max(reach, self._reach), pts[:, self.axis])

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if ss == 1:
            walls = (self._field(grid, self._reach) < 0).astype(np.uint8)
            data = walls if walls.ndim == 3 else self._extruded(walls, grid)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        walls = (self._field(fine, self._reach) < 0).astype(np.float32)
        if walls.ndim == 3:
            nx, ny, nz = grid.shape
            data = walls.reshape(nx, ss, ny, ss, nz, ss).mean(axis=(1, 3, 5), dtype=np.float32)
        else:
            b, c = plane_axes(self.axis)
            nb, nc = grid.shape[b], grid.shape[c]
            coarse = walls.reshape(nb, ss, nc, ss).mean(axis=(1, 3), dtype=np.float32)
            data = self._extruded(coarse, grid)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Distance to the sides of the cells minus half their walls: negative in the
        walls, the signed distance to them near their faces."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        reach = self._reach + 4.0 * float(np.max(grid.spacing))
        far = np.float32(np.max(grid.domain.lengths))
        data = np.minimum(self._space_field(grid, reach), far).astype(np.float32)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def summary(self) -> dict:
        out = {"n_cells": self.n_cells, "mean_wall": float(self.walls.mean())}
        if self.rays is not None:
            out["n_rays"] = self.n_rays
            out["ray_fraction"] = self.ray_fraction
        return out

    def _geometry(self) -> dict:
        geo = {"axis": self.axis, "germs": self.germs.tolist(), "walls": self.walls.tolist()}
        if self.files is not None:
            geo["files"] = {
                "origin": float(self.files["origin"]),
                "width": float(self.files["width"]),
                "bounds": [np.asarray(b).tolist() for b in self.files["bounds"]],
            }
            if self.rays is not None:
                geo["files"]["rays"] = self.rays.to_dict()
        return geo

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            geometry["axis"],
            geometry["germs"],
            geometry["walls"],
            geometry.get("files"),
            **kwargs,
        )


@register
@dataclass(frozen=True)
class Honeycomb(Model):
    """Prismatic cells along ``axis`` with walls of ``wall_thickness``: hexagonal or
    square honeycombs, random cells, or the radial files, growth rings and rays of wood."""

    name = "honeycomb"
    phases: ClassVar[dict[int, str]] = {0: "cells", 1: "walls"}
    count_label = "cells"

    cell_size: float = param(
        1.0, doc="Width of the cells: between opposite walls of the hexagons, the side of "
        "the squares, the width of the files of wood", unit="length",
    )  # fmt: skip
    wall_thickness: float = param(
        0.1, doc="Thickness of the walls (of the earlywood for wood)", unit="length"
    )
    arrangement: str = param(
        "hexagonal", doc="hexagonal, square, voronoi (random) or wood (radial files)", kind="str"
    )
    regularity: float = param(
        0.7, doc="voronoi: least distance of the germs, in jamming distances (0: Poisson)",
        kind="fraction",
    )  # fmt: skip
    disorder: float = param(
        0.0, doc="Random displacement of the germs of lattices and files, in cells", kind="fraction"
    )
    ring_width: float | None = param(
        None, doc="wood: width of the growth rings (default: no rings)", unit="length"
    )
    latewood_fraction: float = param(
        0.3, doc="wood: fraction of the rings in latewood", kind="fraction"
    )
    latewood_size: float = param(
        0.4, doc="wood: radial size of the latewood cells, relative to earlywood"
    )
    latewood_wall: float = param(
        2.0, doc="wood: wall thickness of the latewood cells, relative to earlywood"
    )
    ray_fraction: float = param(
        0.0, doc="wood: fraction of the volume in rays, radial ribbons of cells (0: none)",
        kind="fraction",
    )  # fmt: skip
    ray_width: int = param(
        1, doc="wood: cells across a ray (1: uniseriate rays of softwoods)", kind="int"
    )
    ray_height: float | None = param(
        None, doc="wood: height of the rays along the grain (default: 3 cell sizes)",
        unit="length",
    )  # fmt: skip
    ray_cell_length: float | None = param(
        None, doc="wood: radial length of the ray cells (default: 4 cell sizes)", unit="length"
    )
    ray_cell_width: float | None = param(
        None, doc="wood: tangential width of the ray cells (default: half a cell size)",
        unit="length",
    )  # fmt: skip
    ray_cell_height: float | None = param(
        None, doc="wood: height of the ray cells along the grain (default: 0.6 cell size)",
        unit="length",
    )  # fmt: skip
    ray_wall: float | None = param(
        None, doc="wood: thickness of the walls of the ray cells (default: wall_thickness)",
        unit="length",
    )  # fmt: skip
    axis: int = param(2, doc="Axis of the prisms, the grain of wood (0: x, 1: y, 2: z)", kind="int")

    def _validate(self) -> None:
        d = check_positive(self.cell_size, "cell_size")
        t = check_positive(self.wall_thickness, "wall_thickness")
        if self.arrangement not in ARRANGEMENTS:
            raise ValueError(f"arrangement: one of {list(ARRANGEMENTS)}")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis: 0, 1 or 2")
        if not 0.0 <= self.regularity <= 0.95:
            raise ValueError("regularity: from 0 (Poisson germs) to 0.95 (near jamming)")
        if not 0.0 <= self.disorder <= 0.5:
            raise ValueError("disorder: from 0 to 0.5 cell")
        check_positive(self.latewood_size, "latewood_size")
        check_positive(self.latewood_wall, "latewood_wall")
        if not 0.0 < self.latewood_fraction < 0.6:
            raise ValueError("latewood_fraction: in (0, 0.6), the transition taking half more")
        if self.ring_width is not None:
            check_positive(self.ring_width, "ring_width")
            if self.ring_width < 3.0 * d:
                raise ValueError("ring_width: at least three cells")
        wood = self.arrangement == "wood" and self.ring_width is not None
        thinnest = d * (min(1.0, self.latewood_size) if wood else 1.0)
        thickest = t * (max(1.0, self.latewood_wall) if wood else 1.0)
        if thickest >= thinnest:
            raise ValueError("wall_thickness: walls thinner than the cells (a lumen remains)")
        if not 0.0 <= self.ray_fraction < 0.5:
            raise ValueError("ray_fraction: from 0 (no rays) to 0.5")
        if self.ray_fraction > 0 and self.arrangement != "wood":
            raise ValueError("ray_fraction: rays cross the files of wood (arrangement: wood)")
        if self.ray_width < 1:
            raise ValueError("ray_width: at least one cell")
        names = ("ray_height", "ray_cell_length", "ray_cell_width", "ray_cell_height", "ray_wall")
        for name in names:
            if getattr(self, name) is not None:
                check_positive(getattr(self, name), name)
        if self.ray_fraction > 0:
            ray = self.ray_cells
            if min(ray["cell_length"], ray["cell_width"], ray["cell_height"]) <= ray["thickness"]:
                raise ValueError("ray cells: longer, wider and higher than their walls are thick")
            if ray["height"] < ray["cell_height"] * (1 - 1e-9):
                raise ValueError("ray_height: at least one ray cell")
            if 0.5 * ray["width"] >= d - thickest:
                raise ValueError(
                    "rays: narrower than twice the files less their walls (a lumen remains "
                    "beside them)"
                )

    # -- germs ---------------------------------------------------------------------------

    def _lattice(self, rng, lower, lengths, periodic: bool):
        """Germs of the hexagonal or square lattice over the box (or a window around it)."""
        d = self.cell_size
        if self.arrangement == "hexagonal":
            cell = np.array([d, d * math.sqrt(3.0)])  # rectangular period with two germs
            basis = np.array([[0.0, 0.0], [0.5, 0.5]])
        else:
            cell = np.array([d, d])
            basis = np.array([[0.0, 0.0]])
        if periodic:
            counts = np.maximum(1, np.rint(lengths / cell)).astype(int)
            if counts[0] < 2 or (self.arrangement == "square" and counts[1] < 2):
                raise ValueError("honeycomb: a periodic box at least two cells wide")
            cell = lengths / counts  # whole cells: the lattice stretches to fit the box
            lo, hi = np.zeros(2, dtype=int), counts
        else:
            lo = np.floor((lower - cell) / cell).astype(int)
            hi = np.ceil((lower + lengths + cell) / cell).astype(int)
        i, j = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing="ij")
        corners = np.column_stack([i.ravel(), j.ravel()])
        germs = ((corners[:, None, :] + basis[None] + 0.25) * cell).reshape(-1, 2)
        germs += lower if periodic else 0.0
        if self.disorder > 0:
            germs += (rng.random(germs.shape) - 0.5) * 2.0 * self.disorder * d
        return germs, np.full(len(germs), self.wall_thickness)

    def _random(self, rng, lower, lengths, periodic: bool):
        """Germs of random cells, at least ``regularity`` jamming distance apart."""
        d = self.cell_size
        margin = 0.0 if periodic else 3.0 * d
        window = lengths + 2.0 * margin
        area = float(np.prod(window))
        mean = area / (0.5 * math.sqrt(3.0) * d * d)
        n = max(2, round(mean)) if periodic else max(2, int(rng.poisson(mean)))
        distance = self.regularity * jamming_distance(n, area)
        germs = lower - margin + hardcore_germs(n, window, distance, rng)
        return germs, np.full(n, self.wall_thickness)

    @property
    def ray_cells(self) -> dict[str, float]:
        """Sizes of the ray cells (``cell_length``, ``cell_width``, ``cell_height``) and of
        the rays (``width``, ``height``), and the ``thickness`` of their walls."""
        d = self.cell_size

        def given(value, default: float) -> float:
            return default if value is None else float(value)

        out = {
            "cell_length": given(self.ray_cell_length, 4.0 * d),
            "cell_width": given(self.ray_cell_width, 0.5 * d),
            "cell_height": given(self.ray_cell_height, 0.6 * d),
            "height": given(self.ray_height, 3.0 * d),
            "thickness": given(self.ray_wall, self.wall_thickness),
        }
        out["width"] = self.ray_width * out["cell_width"]
        return out

    def _rays(self, rng, files: dict, thickest: float, domain: Domain) -> dict:
        """Rays between the ``files`` of a box: their boundaries, lower ends along the
        grain and the radial offsets of their cells. They are placed one after the other at
        random, never closer than one ray cell along the grain to another ray on the same
        boundary or on a neighbouring one (a file is never narrowed from both sides)."""
        ray = self.ray_cells
        ax, periodic = self.axis, domain.periodic
        radial = float(domain.lengths[plane_axes(ax)[0]])
        lower, length = float(domain.lower[ax]), float(domain.lengths[ax])
        n_files, file_width = len(files["bounds"]), float(files["width"])
        width, tall, cell_length = ray["width"], ray["height"], ray["cell_length"]
        if periodic:  # whole ray cells radially
            cell_length = radial / max(1, round(radial / cell_length))
        cell_height = tall / max(1, round(tall / ray["cell_height"]))  # whole cells per ray
        gap = cell_height
        if 0.5 * width >= file_width - thickest:
            raise ValueError("rays: narrower than twice the files less their walls")
        if periodic:
            if tall + gap > length:
                raise ValueError("rays: a periodic box longer than a ray and one of its cells")
            window, boundaries, first = length, n_files, 0
            n = round(self.ray_fraction * n_files * file_width * length / (width * tall))
            share = n * width * tall / (n_files * file_width * length)
            if abs(share - self.ray_fraction) > 0.1 * self.ray_fraction:
                warnings.warn(
                    f"honeycomb: the periodic box holds {n} whole rays, {share:.3g} of the "
                    f"volume instead of {self.ray_fraction:.3g}; for the asked fraction, make "
                    "it longer along the grain or wider",
                    RuntimeWarning,
                    stacklevel=3,
                )
        else:  # on the inner boundaries of the files, starting below the box
            window, boundaries, first = length + tall, n_files - 1, 1
            mean = self.ray_fraction * boundaries * file_width * window / (width * tall)
            n = int(rng.poisson(mean))
        where, starts = np.zeros(n, dtype=np.int64), np.zeros(n)
        count = attempts = 0
        while count < n:
            attempts += 1
            if attempts > 1000 * (n + 10):
                raise ValueError(
                    "ray_fraction: the rays do not fit (fewer, lower or narrower rays)"
                )
            j = first + int(rng.integers(boundaries))
            z = float(rng.random() * window)
            k, dz = where[:count], np.abs(z - starts[:count])
            if periodic:
                apart = np.minimum((j - k) % n_files, (k - j) % n_files)
                dz = np.minimum(dz, length - dz)
            else:
                apart = np.abs(j - k)
            if np.any((apart <= 1) & (dz < tall + gap)):
                continue
            where[count], starts[count] = j, z
            count += 1
        z0 = lower if periodic else lower - tall
        return {
            "boundary": where.tolist(),
            "z": (z0 + starts).tolist(),
            "u": (rng.random(n) * cell_length).tolist(),
            "cells": self.ray_width,
            "width": width,
            "height": tall,
            "cell_length": cell_length,
            "cell_height": cell_height,
            "thickness": ray["thickness"],
        }

    def ring_profile(self, x: np.ndarray) -> np.ndarray:
        """Latewood share at the positions ``x`` (in rings, from the start of a ring): 0 in
        earlywood, 1 in latewood, linear through the transition."""
        f = self.latewood_fraction
        x = np.mod(x, 1.0)
        return np.clip((x - (1.0 - 1.5 * f)) / (0.5 * f), 0.0, 1.0)

    def _ring_cells(self, ring: float) -> np.ndarray:
        """Boundaries of the cells of a ring (in rings, from 0 to 1): steps of the local
        cell size, scaled to end on the ring boundary."""
        d = self.cell_size
        x, bounds = 0.0, [0.0]
        while x < 1.0:
            late = float(self.ring_profile(np.array([x]))[0])
            x += d * (1.0 + (self.latewood_size - 1.0) * late) / ring
            bounds.append(x)
        bounds = np.array(bounds)
        if len(bounds) > 2 and bounds[-1] - 1.0 > 0.5 * (bounds[-1] - bounds[-2]):
            bounds = bounds[:-1]  # the closest whole number of cells
        return bounds / bounds[-1]

    def _wood(self, rng, lower, lengths, periodic: bool):
        """Radial files (along the first axis of the plane): the bounds of their cells,
        the centres of the cells and their walls."""
        d = self.cell_size
        margin = 0.0 if periodic else 3.0 * d
        lo, size = lower - margin, lengths + 2.0 * margin
        if periodic:
            n_files = max(2, round(size[1] / d))
            width = size[1] / n_files
        else:
            n_files, width = math.ceil(size[1] / d), d
        ring = d if self.ring_width is None else self.ring_width  # no rings: one cell each
        if periodic:
            n_rings = max(1, round(size[0] / ring))
            ring, start = size[0] / n_rings, 0.0
        else:
            n_rings, start = math.ceil(size[0] / ring) + 1, -rng.random()
        cells = np.array([0.0, 1.0]) if self.ring_width is None else self._ring_cells(ring)
        edges = (start + np.arange(n_rings)[:, None] + cells[None, :-1]).ravel()
        steps = np.diff(np.append(edges, edges[0] + n_rings))  # cell sizes, in rings
        bounds, germs, walls = [], [], []
        for j in range(n_files):
            b = edges + rng.random() * steps  # each file shifted by part of its cells
            if self.disorder > 0:
                least = np.minimum(steps, np.roll(steps, 1))
                b = b + (rng.random(len(b)) - 0.5) * self.disorder * least
            b = np.append(b, b[0] + n_rings)
            centres = 0.5 * (b[:-1] + b[1:])
            late = self.ring_profile(centres - start) if self.ring_width else 0.0 * centres
            bounds.append(lo[0] + b * ring)
            u = np.mod(centres, n_rings) if periodic else centres  # the centres in the box
            v = np.full(len(u), lo[1] + (j + 0.5) * width)
            germs.append(np.column_stack([lo[0] + u * ring, v]))
            walls.append(self.wall_thickness * (1.0 + (self.latewood_wall - 1.0) * late))
        files = {"origin": float(lo[1]), "width": float(width), "bounds": bounds}
        return np.concatenate(germs), np.concatenate(walls), files

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrismaticCells:
        rng, seed_info = make_rng(seed)
        b, c = plane_axes(self.axis)
        lower, lengths = domain.lower[[b, c]], domain.lengths[[b, c]]
        if domain.periodic and np.any(lengths < 1.5 * self.cell_size):
            raise ValueError("honeycomb: a periodic box at least two cells wide")
        files = None
        if self.arrangement in ("hexagonal", "square"):
            germs, walls = self._lattice(rng, lower, lengths, domain.periodic)
        elif self.arrangement == "voronoi":
            germs, walls = self._random(rng, lower, lengths, domain.periodic)
        else:
            germs, walls, files = self._wood(rng, lower, lengths, domain.periodic)
            if self.ray_fraction > 0:
                files["rays"] = self._rays(rng, files, float(walls.max()), domain)
        return PrismaticCells(domain, self.axis, germs, walls, files, model=self, seed=seed_info)

    # -- theory ------------------------------------------------------------------------------

    @property
    def cell_area(self) -> float:
        d = self.cell_size
        return 0.5 * math.sqrt(3.0) * d * d if self.arrangement != "square" else d * d

    def expected_count(self, domain: Domain) -> float:
        """Mean number of cells across the domain."""
        b, c = plane_axes(self.axis)
        return float(domain.lengths[b] * domain.lengths[c] / self.cell_area)

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        d, t = self.cell_size, self.wall_thickness
        out = {"cells_per_area": 1.0 / self.cell_area}
        if self.arrangement in ("hexagonal", "square") and self.disorder == 0:
            out["volume_fraction"] = 1.0 - (1.0 - t / d) ** 2
            out["specific_surface"] = 4.0 * (d - t) / d**2  # lumen perimeter per area
        elif self.arrangement == "voronoi" and self.regularity == 0:
            out["volume_fraction_first_order"] = 2.0 * t / math.sqrt(self.cell_area)
        if self.ray_fraction > 0:
            ray = self.ray_cells
            out["ray_fraction"] = self.ray_fraction
            out["ray_height_cells"] = float(max(1, round(ray["height"] / ray["cell_height"])))
        return out
