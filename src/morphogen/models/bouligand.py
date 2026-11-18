"""Bouligand structures: twisted plywoods of fibres.

The fibrous composites of arthropod cuticles (chitin), of fish scales and of the
osteons of bone (collagen) stack layers of parallel fibres, each layer turned by a
small angle from the one below (Bouligand 1972): the fibre direction describes a
helicoid along the stacking axis. Cracks follow the twisting fibres, which spreads the
damage and makes the dactyl club of the mantis shrimp so impact tolerant (Weaver et al.
2012); helicoidal laminates of carbon fibres borrow the arrangement.

Layer ``k`` (thickness ``h``) holds parallel cylinders of radius ``r`` at the spacing
``s``, in the direction ``theta_0 + k delta`` (``delta``: the pitch angle) at a random
offset. The fibres of a layer do not overlap (``2 r <= s``), nor those of consecutive
layers (``2 r <= h``): the volume fraction of fibres is ``pi r^2 / (s h)``.

A periodic box holds a whole number of half-turns of the helicoid, and fibre families
that are periodic in its plane: the fibres normal to ``(a / L_u, b / L_v)`` (integers
``a``, ``b``) at the spacing ``1 / |(a / L_u, b / L_v)|``. The number of layers and the
pitch are those closest to the asked thickness and pitch, and each layer takes the
periodic family closest to its direction and spacing (a larger box gets closer); the
directions and spacings taken are those of the realization.
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
from morphogen.models.planar import axis_centres, pixel_centres, plane_axes
from morphogen.seeding import SeedLike, make_rng


def periodic_families(lengths, spacing: float, least: float):
    """Fibre families that are periodic in a rectangle of sides ``lengths``: the fibre
    direction (degrees, in ``[0, 180)``) and the spacing of each, at least ``least``."""
    lu, lv = (float(x) for x in lengths)
    a_max = math.ceil(3.0 * lu / spacing) + 1
    b_max = math.ceil(3.0 * lv / spacing) + 1
    a, b = np.meshgrid(np.arange(-a_max, a_max + 1), np.arange(0, b_max + 1), indexing="ij")
    a, b = a.ravel(), b.ravel()
    keep = (b > 0) | (a > 0)
    a, b = a[keep], b[keep]
    nu, nv = a / lu, b / lv
    s = 1.0 / np.hypot(nu, nv)
    ok = s >= least
    normal = np.degrees(np.arctan2(nv[ok], nu[ok]))
    return np.mod(normal - 90.0, 180.0), s[ok]


def _angle_gap(a, b):
    """Difference of fibre directions (degrees, modulo 180)."""
    return np.abs((np.asarray(a) - b + 90.0) % 180.0 - 90.0)


@register_kind
class BouligandStructure(Microstructure):
    """Layers of parallel fibres along ``axis``: layer ``k`` spans ``[z0 + k h, z0 + (k +
    1) h)``, its fibres the cylinders of radius ``r`` around the lines of its mid-plane
    in the direction ``angles[k]`` (degrees from the first axis of the plane), at the
    spacing ``spacings[k]``, one of them through ``offsets[k]`` along their normal."""

    kind = "bouligand"

    def __init__(self, domain, axis, radius, thickness, origin, angles, spacings, offsets, **kw):
        super().__init__(domain, **kw)
        self.axis = int(axis)
        self.radius = float(radius)
        self.thickness = float(thickness)
        self.origin = float(origin)
        self.angles = np.asarray(angles, dtype=float).reshape(-1)
        self.spacings = np.asarray(spacings, dtype=float).reshape(-1)
        self.offsets = np.asarray(offsets, dtype=float).reshape(-1)
        t = np.radians(self.angles)
        self.normals = np.column_stack([-np.sin(t), np.cos(t)])

    @property
    def n_layers(self) -> int:
        return len(self.angles)

    def _layer(self, j: np.ndarray) -> np.ndarray:
        if self.domain.periodic:
            return np.mod(j, self.n_layers)
        return np.where((j >= 0) & (j < self.n_layers), j, -1)

    def _distance(self, plane: np.ndarray, z: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Distance to the nearest fibre axis of the layers ``j`` (``inf``: none)."""
        k = self._layer(j)
        out = np.full(len(z), np.inf)
        ok = k >= 0
        kk = k[ok]
        u = np.einsum("ij,ij->i", plane[ok], self.normals[kk]) - self.offsets[kk]
        du = u - self.spacings[kk] * np.round(u / self.spacings[kk])
        dz = z[ok] - (self.origin + (j[ok] + 0.5) * self.thickness)
        out[ok] = np.hypot(du, dz)
        return out

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """The signed distance to the fibres (negative inside)."""
        return self.signed_distance(points)

    def signed_distance(self, points) -> np.ndarray:
        """Distance to the fibres (negative inside): the distance to the nearest axis of
        the layers around each point, minus the radius."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.domain.periodic:
            pts = self.domain.wrap(pts)
        b, c = plane_axes(self.axis)
        z = pts[:, self.axis]
        j = np.floor((z - self.origin) / self.thickness).astype(np.int64)
        plane = pts[:, [b, c]]
        d = np.minimum.reduce([self._distance(plane, z, j + s) for s in (-1, 0, 1)])
        return d - self.radius

    def contains(self, points) -> np.ndarray:
        return self.signed_distance(points) <= 0.0

    def _field(self, grid: Grid) -> np.ndarray:
        b, c = plane_axes(self.axis)
        plane, shape = pixel_centres(grid, (b, c))
        z = axis_centres(grid, self.axis)
        out = np.empty((len(z), *shape), dtype=np.float32)
        for i, zi in enumerate(z):
            j0 = math.floor((zi - self.origin) / self.thickness)
            zz = np.full(len(plane), zi)
            d = np.minimum.reduce(
                [self._distance(plane, zz, np.full(len(plane), j0 + s)) for s in (-1, 0, 1)]
            )
            out[i] = (d - self.radius).reshape(shape)
        return np.moveaxis(out, 0, self.axis)

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if ss == 1:
            data = (self._field(grid) <= 0).astype(np.uint8)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        ph = (self._field(fine) <= 0).astype(np.float32)
        nx, ny, nz = grid.shape
        data = ph.reshape(nx, ss, ny, ss, nz, ss).mean(axis=(1, 3, 5), dtype=np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Signed distance to the fibres (exact: the union of the cylinders of the
        layers around each point)."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        far = np.float32(np.max(grid.domain.lengths))
        data = np.minimum(self._field(grid), far)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def summary(self) -> dict:
        fraction = np.mean(math.pi * self.radius**2 / (self.spacings * self.thickness))
        return {
            "n_layers": self.n_layers,
            "pitch_angle": float(self.pitch_angle),
            "fiber_fraction": float(fraction),
        }

    @property
    def pitch_angle(self) -> float:
        """Mean rotation from one layer to the next (degrees)."""
        if self.n_layers < 2:
            return 0.0
        turns = (np.diff(self.angles) + 90.0) % 180.0 - 90.0
        return float(np.mean(turns))

    def _geometry(self) -> dict:
        return {
            "axis": self.axis,
            "radius": self.radius,
            "thickness": self.thickness,
            "origin": self.origin,
            "angles": self.angles.tolist(),
            "spacings": self.spacings.tolist(),
            "offsets": self.offsets.tolist(),
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            geometry["axis"],
            geometry["radius"],
            geometry["thickness"],
            geometry["origin"],
            geometry["angles"],
            geometry["spacings"],
            geometry["offsets"],
            **kwargs,
        )


@register
@dataclass(frozen=True)
class Bouligand(Model):
    """Bouligand (twisted plywood) structure: layers of parallel fibres, each turned by
    ``pitch_angle`` from the one below, stacked along ``axis``."""

    name = "bouligand"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "fibres"}
    count_label = "layers"

    fiber_radius: float = param(0.05, doc="Radius of the fibres", unit="length")
    fiber_spacing: float | None = param(
        None, doc="Distance between the fibres of a layer (default: 2.5 radii)", unit="length"
    )
    layer_thickness: float | None = param(
        None, doc="Thickness of the layers (default: one fibre diameter)", unit="length"
    )
    pitch_angle: float = param(
        15.0, doc="Rotation of the fibres from a layer to the next (degrees)"
    )
    start_angle: float | None = param(
        None, doc="Direction of the fibres of the first layer, in degrees from the first "
        "axis of the plane (default: random)",
    )  # fmt: skip
    axis: int = param(2, doc="Stacking direction (0: x, 1: y, 2: z)", kind="int")

    def _validate(self) -> None:
        r = check_positive(self.fiber_radius, "fiber_radius")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis: 0, 1 or 2")
        if not -90.0 < self.pitch_angle <= 90.0:
            raise ValueError("pitch_angle: in (-90, 90] degrees")
        if self.spacing < 2.0 * r * (1 - 1e-12):
            raise ValueError("fiber_spacing: at least one fibre diameter (no overlap)")
        if self.thickness < 2.0 * r * (1 - 1e-12):
            raise ValueError("layer_thickness: at least one fibre diameter (no overlap)")

    @property
    def spacing(self) -> float:
        return 2.5 * self.fiber_radius if self.fiber_spacing is None else self.fiber_spacing

    @property
    def thickness(self) -> float:
        return 2.0 * self.fiber_radius if self.layer_thickness is None else self.layer_thickness

    def _stack(self, height: float) -> tuple[int, float]:
        """Layers and pitch of a periodic box: whole half-turns, the closest to the
        asked layer thickness and pitch angle."""
        h, delta = self.thickness, self.pitch_angle
        n0 = max(1, round(height / h))
        best = None
        for n in range(1, 2 * n0 + 4):
            if height / n < 2.0 * self.fiber_radius * (1 - 1e-12):
                break
            m = round(n * delta / 180.0)
            pitch = 180.0 * m / n
            if pitch <= -90.0 or pitch > 90.0:
                continue
            turn = (pitch - delta) / max(abs(delta), 1.0)  # relative, or in degrees near 0
            cost = (math.log(height / (n * h)) / 0.1) ** 2 + (turn / 0.05) ** 2
            if best is None or cost < best[0]:
                best = (cost, n, pitch)
        if best is None:
            raise ValueError("bouligand: the box is thinner than one layer of fibres")
        return best[1], best[2]

    def generate(self, domain: Domain, seed: SeedLike = None) -> BouligandStructure:
        rng, seed_info = make_rng(seed)
        ax = self.axis
        b, c = plane_axes(ax)
        height = float(domain.lengths[ax])
        start = rng.uniform(0.0, 180.0) if self.start_angle is None else self.start_angle
        if domain.periodic:
            n, pitch = self._stack(height)
            thickness = height / n
            if abs(thickness / self.thickness - 1) > 0.05 or abs(pitch - self.pitch_angle) > max(
                0.5, 0.05 * abs(self.pitch_angle)
            ):
                warnings.warn(
                    f"bouligand: the periodic box holds {n} layers {thickness:.4g} thick, "
                    f"turned by {pitch:.4g} degrees (whole half-turns); for the asked values, "
                    "make its height a multiple of 180 / pitch_angle layers",
                    RuntimeWarning,
                    stacklevel=2,
                )
            origin = float(domain.lower[ax] + rng.random() * thickness)
            wanted = start + pitch * np.arange(n)
            dirs, gaps = periodic_families(
                domain.lengths[[b, c]], self.spacing, 2.0 * self.fiber_radius
            )
            angles, spacings = [], []
            for w in wanted:
                cost = (_angle_gap(dirs, w) / 1.0) ** 2 + (np.log(gaps / self.spacing) / 0.025) ** 2
                i = int(np.argmin(cost))
                angles.append(dirs[i])
                spacings.append(gaps[i])
            angles, spacings = np.array(angles), np.array(spacings)
        else:
            thickness = self.thickness
            origin = float(domain.lower[ax] - rng.random() * thickness)
            n = math.ceil((domain.upper[ax] - origin) / thickness)
            angles = np.mod(start + self.pitch_angle * np.arange(n), 180.0)
            spacings = np.full(n, self.spacing)
        offsets = rng.random(n) * spacings
        return BouligandStructure(
            domain,
            ax,
            self.fiber_radius,
            thickness,
            origin,
            angles,
            spacings,
            offsets,
            model=self,
            seed=seed_info,
        )

    def expected_count(self, domain: Domain) -> float:
        """Number of layers across the domain."""
        return float(domain.lengths[self.axis] / self.thickness)

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        r, s, h = self.fiber_radius, self.spacing, self.thickness
        turn = 180.0 / abs(self.pitch_angle) if self.pitch_angle else math.inf
        return {
            "volume_fraction": math.pi * r * r / (s * h),
            "specific_surface": 2.0 * math.pi * r / (s * h),
            "layers_per_half_turn": turn,
            "half_turn_height": turn * h,
        }
