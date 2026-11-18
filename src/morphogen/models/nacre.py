"""Nacre: brick-and-mortar stacks of mineral tablets in an organic matrix.

Nacre (mother of pearl) is about 95 % of aragonite tablets, some 0.5 um thick and 5 to
10 um wide, in layers bonded by some 20 to 30 nm of biopolymers (the mortar); the
tablets of consecutive layers overlap, and the load passes from one to the next by
shear of the mortar between them (Jackson, Vincent and Turner 1988; Barthelat et al.
2007). Its toughness, far above that of aragonite, inspires brick-and-mortar composites.

The model stacks layers of tablets along an axis, with the period ``t + m`` (``t``: the
thickness of the tablets, ``m``: that of the mortar between layers). In each layer the
tablets are the cells of a planar tessellation shrunk by half the gap ``g`` between
neighbouring tablets:

* ``voronoi``: polygonal tablets, the cells of germs at least a fraction ``regularity``
  of the jamming distance of their random sequential addition apart (Poisson germs for
  0), independent from one layer to the next (sheet nacre);
* ``brick``: rectangles of area ``w^2`` and aspect ratio ``a/b``, each layer shifted by
  ``overlap`` tablet from the previous one, along both axes of the plane (each tablet
  over four, ``stagger: uv``) or along the first (over two, ``stagger: u``). A periodic
  box one tablet wide along an axis makes the tablets continuous across it: with
  ``stagger: u``, the two-dimensional brick-and-mortar model, extruded.

Volume fraction of tablets: ``t / (t + m)`` times the fraction of each layer they cover,
``(a - g)(b - g) / (a b)`` for bricks, and ``1 - 2 sqrt(lambda) g`` to first order in
``g`` for the cells of Poisson germs of intensity ``lambda = 1 / w^2`` (their sides are
``2 sqrt(lambda)`` long per unit area).
"""

from __future__ import annotations

import math
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

ARRANGEMENTS = ("voronoi", "brick")
STAGGERS = ("uv", "u")


@register_kind
class NacreStructure(Microstructure):
    """Layers of tablets along ``axis``: layer ``k`` spans ``[z0 + k p, z0 + (k + 1) p)``
    (``p``: the period), its tablet the slab of thickness ``t`` in its middle; in the
    plane, the tablets are the cells of the germs of the layer, shrunk by half the
    ``gap``. Periodic domains hold a whole number of layers, which wrap around."""

    kind = "nacre"

    def __init__(self, domain, axis, period, thickness, offset, gap, layers, **kwargs):
        super().__init__(domain, **kwargs)
        self.axis = int(axis)
        self.period = float(period)
        self.thickness = float(thickness)
        self.offset = float(offset)
        self.gap = float(gap)
        self.layers = [np.asarray(g, dtype=float).reshape(-1, 2) for g in layers]
        self._cells: dict[int, PlanarCells] = {}

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def axes(self) -> tuple[int, int]:
        return plane_axes(self.axis)

    def cells(self, layer: int) -> PlanarCells:
        """Tessellation of the tablets of a layer."""
        if layer not in self._cells:
            b, c = self.axes
            d = self.domain
            self._cells[layer] = PlanarCells(
                self.layers[layer], d.lengths[[b, c]], d.periodic, d.lower[[b, c]]
            )
        return self._cells[layer]

    def _layer(self, j: np.ndarray) -> np.ndarray:
        """Stored layer of the layer indices ``j`` (-1: none)."""
        if self.domain.periodic:
            return np.mod(j, self.n_layers)
        return np.where((j >= 0) & (j < self.n_layers), j, -1)

    def _slab(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Index of the layer around each coordinate and the signed distance to the slab
        of its tablet (negative inside)."""
        j = np.floor((z - self.offset) / self.period).astype(np.int64)
        centre = self.offset + (j + 0.5) * self.period
        return j, np.abs(z - centre) - 0.5 * self.thickness

    # -- point classification ----------------------------------------------------------

    def contains(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.domain.periodic:
            pts = self.domain.wrap(pts)
        b, c = self.axes
        j, slab = self._slab(pts[:, self.axis])
        layer = self._layer(j)
        out = np.zeros(len(pts), dtype=bool)
        for k in np.unique(layer[(slab <= 0) & (layer >= 0)]):
            sel = (layer == k) & (slab <= 0)
            _, d = self.cells(int(k)).locate(pts[sel][:, [b, c]], reach=0.5 * self.gap)
            out[sel] = d > 0.5 * self.gap
        return out

    # -- discretization ---------------------------------------------------------------

    def _phases_on(self, grid: Grid) -> np.ndarray:
        pts, shape = pixel_centres(grid, self.axes)
        j, slab = self._slab(axis_centres(grid, self.axis))
        layer = self._layer(j)
        out = np.zeros((len(j), *shape), dtype=np.uint8)
        tablets: dict[int, np.ndarray] = {}
        for i in np.flatnonzero((slab <= 0) & (layer >= 0)):
            k = int(layer[i])
            if k not in tablets:
                _, d = self.cells(k).locate(pts, reach=0.5 * self.gap)
                tablets[k] = (d > 0.5 * self.gap).reshape(shape).astype(np.uint8)
            out[i] = tablets[k]
        return np.moveaxis(out, 0, self.axis)

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if ss == 1:
            data = self._phases_on(grid)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        ph = self._phases_on(fine).astype(np.float32)
        nx, ny, nz = grid.shape
        data = ph.reshape(nx, ss, ny, ss, nz, ss).mean(axis=(1, 3, 5), dtype=np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Minimum over the layers around each point of ``max(slab, g / 2 - d)``: the
        signed distance to the slab of the tablets and the distance ``d`` to the sides
        of the cells (negative in the tablets, a distance near their faces)."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        pts, shape2 = pixel_centres(grid, self.axes)
        z = axis_centres(grid, self.axis)
        j0, _ = self._slab(z)
        reach = 0.5 * self.gap + 4.0 * float(np.max(grid.spacing))
        far = np.float32(np.max(grid.domain.lengths))
        out = np.full((len(z), *shape2), far, dtype=np.float32)
        planes: dict[int, np.ndarray] = {}
        for i, zi in enumerate(z):
            for j in (j0[i] - 1, j0[i], j0[i] + 1):
                k = int(self._layer(np.array([j]))[0])
                if k < 0:
                    continue
                if k not in planes:
                    _, d = self.cells(k).locate(pts, reach=reach)
                    planes[k] = (0.5 * self.gap - d).reshape(shape2).astype(np.float32)
                centre = self.offset + (j + 0.5) * self.period
                slab = np.float32(abs(zi - centre) - 0.5 * self.thickness)
                out[i] = np.minimum(out[i], np.maximum(slab, planes[k]))
        data = np.moveaxis(out, 0, self.axis)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """As :meth:`level_set`, at the points (the distances to the sides of the cells
        exact within ``reach`` of them, by default the half gap and a tenth of the period)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.domain.periodic:
            pts = self.domain.wrap(pts)
        b, c = self.axes
        z = pts[:, self.axis]
        j0, _ = self._slab(z)
        if not np.isfinite(reach):
            reach = 0.5 * self.gap + 0.1 * self.period
        out = np.full(len(pts), float(np.max(self.domain.lengths)))
        for shift in (-1, 0, 1):
            j = j0 + shift
            layer = self._layer(j)
            centre = self.offset + (j + 0.5) * self.period
            slab = np.abs(z - centre) - 0.5 * self.thickness
            for k in np.unique(layer[layer >= 0]):
                sel = np.flatnonzero(layer == k)
                _, d = self.cells(int(k)).locate(pts[sel][:, [b, c]], reach=float(reach))
                out[sel] = np.minimum(out[sel], np.maximum(slab[sel], 0.5 * self.gap - d))
        return out

    # -- description -------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "tablets_per_layer": float(np.mean([len(g) for g in self.layers])),
        }

    def _geometry(self) -> dict:
        return {
            "axis": self.axis,
            "period": self.period,
            "thickness": self.thickness,
            "offset": self.offset,
            "gap": self.gap,
            "layers": [g.tolist() for g in self.layers],
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            geometry["axis"],
            geometry["period"],
            geometry["thickness"],
            geometry["offset"],
            geometry["gap"],
            geometry["layers"],
            **kwargs,
        )


@register
@dataclass(frozen=True)
class Nacre(Model):
    """Nacre: staggered layers of mineral tablets bonded by an organic mortar.

    Tablets of ``tablet_thickness`` and in-plane ``tablet_size`` (the side of a square of
    the same area), ``voronoi`` polygons (sheet nacre) or ``brick`` rectangles in a
    staggered stack, separated by ``mortar_thickness`` between the layers and ``gap``
    between the tablets of a layer, stacked along ``axis``.
    """

    name = "nacre"
    phases: ClassVar[dict[int, str]] = {0: "mortar", 1: "tablets"}
    count_label = "tablets"

    tablet_size: float = param(
        1.0, doc="In-plane size of the tablets (square root of their area)", unit="length"
    )
    tablet_thickness: float = param(0.2, doc="Thickness of the tablets", unit="length")
    mortar_thickness: float = param(
        0.02, doc="Thickness of the mortar between the layers", unit="length"
    )
    gap: float | None = param(
        None, doc="Width of the mortar between the tablets of a layer (default: as between "
        "the layers)", unit="length",
    )  # fmt: skip
    arrangement: str = param(
        "voronoi", doc="voronoi (polygonal tablets) or brick (staggered rectangles)", kind="str"
    )
    regularity: float = param(
        0.7, doc="voronoi: least distance of the germs, in jamming distances (0: Poisson)",
        kind="fraction",
    )  # fmt: skip
    aspect: float = param(1.0, doc="brick: length over width of the tablets")
    overlap: float = param(
        0.5, doc="brick: shift of a layer from the previous one, in tablets", kind="fraction"
    )
    stagger: str = param(
        "uv", doc="brick: shift along both axes of the plane (uv) or the first (u)", kind="str"
    )
    axis: int = param(2, doc="Stacking direction (0: x, 1: y, 2: z)", kind="int")

    def _validate(self) -> None:
        check_positive(self.tablet_size, "tablet_size")
        check_positive(self.tablet_thickness, "tablet_thickness")
        check_positive(self.mortar_thickness, "mortar_thickness", allow_zero=True)
        check_positive(self.aspect, "aspect")
        if self.arrangement not in ARRANGEMENTS:
            raise ValueError(f"arrangement: one of {list(ARRANGEMENTS)}")
        if self.stagger not in STAGGERS:
            raise ValueError(f"stagger: one of {list(STAGGERS)}")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis: 0, 1 or 2")
        if not 0.0 <= self.regularity <= 0.95:
            raise ValueError("regularity: from 0 (Poisson germs) to 0.95 (near jamming)")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap: from 0 to 1 (excluded)")
        g = self.gap_width
        check_positive(g, "gap", allow_zero=True)
        a, b = self.brick_sides
        if g >= (min(a, b) if self.arrangement == "brick" else 0.5 * self.tablet_size):
            raise ValueError("gap: narrower than the tablets")

    @property
    def gap_width(self) -> float:
        return self.mortar_thickness if self.gap is None else float(self.gap)

    @property
    def period(self) -> float:
        """Thickness of a layer: a tablet and the mortar over it."""
        return self.tablet_thickness + self.mortar_thickness

    @property
    def brick_sides(self) -> tuple[float, float]:
        """Sides of the bricks along the two axes of the plane."""
        s = math.sqrt(self.aspect)
        return self.tablet_size * s, self.tablet_size / s

    # -- generation ----------------------------------------------------------------------

    def _layer_count(self, length: float) -> int:
        """Layers in a periodic box: the closest to its height in whole periods of the
        staggering (a whole number of tablets of shift)."""
        n0 = max(1, round(length / self.period))
        if self.arrangement != "brick" or self.overlap == 0:
            return n0
        fits = [
            n for n in range(1, 4 * n0 + 8)
            if abs(n * self.overlap - round(n * self.overlap)) < 1e-9
        ]  # fmt: skip
        if not fits:
            raise ValueError(
                f"nacre: no number of layers repeats a shift of {self.overlap} tablet; "
                "use a rational overlap (0.5, 0.25, 1/3...) or a non-periodic domain"
            )
        return min(fits, key=lambda n: abs(math.log(length / (n * self.period))))

    def _brick_germs(self, k: int, lower, lengths, counts, sides, periodic: bool):
        """Centres of the bricks of layer ``k``."""
        a, b = sides
        shift = self.overlap * k * np.array([a, b if self.stagger == "uv" else 0.0])
        if periodic:  # from the corner of the box, which holds whole bricks
            i, j = np.meshgrid(np.arange(counts[0]), np.arange(counts[1]), indexing="ij")
            shift = shift + lower
        else:  # the bricks over the box, and a row around it
            lo = np.floor((lower - shift) / sides).astype(int) - 1
            hi = np.ceil((lower + lengths - shift) / sides).astype(int) + 1
            i, j = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing="ij")
        return np.column_stack([(i.ravel() + 0.5) * a, (j.ravel() + 0.5) * b]) + shift

    def generate(self, domain: Domain, seed: SeedLike = None) -> NacreStructure:
        rng, seed_info = make_rng(seed)
        ax = self.axis
        b, c = plane_axes(ax)
        lower, lengths = domain.lower[[b, c]], domain.lengths[[b, c]]
        height = float(domain.lengths[ax])
        scale = 1.0
        if domain.periodic:
            n_layers = self._layer_count(height)
            scale = height / (n_layers * self.period)  # the stack fills the box
            offset = float(domain.lower[ax] + rng.random() * height)
            layer_ids = range(n_layers)
        else:
            offset = float(domain.lower[ax] - rng.random() * self.period)
            n_layers = math.ceil((domain.upper[ax] - offset) / self.period)
            layer_ids = range(n_layers)
        sides = np.array(self.brick_sides)
        counts = np.maximum(1, np.rint(lengths / sides)).astype(int)
        if domain.periodic and self.arrangement == "brick":
            sides = lengths / counts  # whole bricks in the box
        layers = []
        margin = 0.0 if domain.periodic else 3.0 * self.tablet_size
        window_lo, window = lower - margin, lengths + 2.0 * margin
        area = float(np.prod(window))
        for k in layer_ids:
            if self.arrangement == "brick":
                layers.append(self._brick_germs(k, lower, lengths, counts, sides, domain.periodic))
                continue
            if domain.periodic:
                n = max(1, round(area / self.tablet_size**2))
            else:
                n = max(1, int(rng.poisson(area / self.tablet_size**2)))
            distance = self.regularity * jamming_distance(n, area)
            layers.append(window_lo + hardcore_germs(n, window, distance, rng))
        metadata = {"scale": {"thickness": scale}} if domain.periodic else {}
        if domain.periodic and self.arrangement == "brick":
            metadata.setdefault("scale", {})["tablet_sides"] = (sides / self.brick_sides).tolist()
        return NacreStructure(
            domain,
            ax,
            self.period * scale,
            self.tablet_thickness * scale,
            offset,
            self.gap_width,
            layers,
            model=self,
            seed=seed_info,
            metadata=metadata,
        )

    # -- theory ----------------------------------------------------------------------------

    def expected_count(self, domain: Domain) -> float:
        """Mean number of tablets in the domain."""
        return float(domain.volume / (self.tablet_size**2 * self.period))

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        layer = self.tablet_thickness / self.period
        g = self.gap_width
        out = {
            "layer_fraction": layer,
            "tablet_aspect_ratio": self.tablet_size / self.tablet_thickness,
        }
        if self.arrangement == "brick":
            a, b = self.brick_sides
            out["volume_fraction"] = layer * (a - g) * (b - g) / (a * b)
        elif self.regularity == 0:  # sides 2 sqrt(lambda) long per unit area
            out["gap_fraction_first_order"] = 2.0 * g / self.tablet_size
            out["volume_fraction_first_order"] = layer * (1.0 - 2.0 * g / self.tablet_size)
        return out
