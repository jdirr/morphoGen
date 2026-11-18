"""Random laminates: layers of two phases of random thicknesses, stacked along an axis,
with holes through the layers of phase 1 (breakup of thin layers).

Nanolayered films made by multilayer coextrusion are such stacks; their thinnest
layers break up into holes and fragments, which adds an in-plane scale to the scale of
the layers (Bironeau et al. 2016). The model is an alternating renewal process along the
axis: layers of phase 1 and of phase 0, of independent thicknesses ``t`` and ``s``
drawn from their distributions; each layer of phase 1 is pierced by cylindrical holes
across its thickness, a two-dimensional Boolean model of discs covering the fraction
``q`` of its area, independent from one layer to the next.

Non-periodic boxes see a stationary stack: the layer covering the lower face is of
phase 1 with probability ``E[t] / (E[t] + E[s])``, its length is drawn from the
length-biased distribution, and the face falls uniformly in it. Along a periodic axis,
whole periods are drawn and their thicknesses scaled by a common factor (close to one
for many layers) so that they fill the box; a random shift makes the stack stationary.

With ``p = E[t] / (E[t] + E[s])`` and holes of radius ``r``:

* volume fraction: ``p (1 - q)``;
* specific surface: ``(1 - q) (2 + E[t] lambda 2 pi E[r]) / (E[t] + E[s])``, with
  ``lambda = -ln(1 - q) / (pi E[r^2])`` holes per unit area of each layer;
* without holes, the volume fraction of a box of edge ``L`` along the axis has the
  variance ``p (1 - p) A / L`` for large ``L`` (renewal-reward theorem), with the
  one-dimensional integral range ``A = (E[s]^2 Var[t] + E[t]^2 Var[s]) /
  ((E[t] + E[s]) E[t] E[s])``: ``2 E[t] E[s] / (E[t] + E[s])`` for exponential
  thicknesses, 0 for constant ones. In the plane of the layers the covariance does not
  decay: the variance decreases as ``1 / L`` only, not as the inverse of the volume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.spatial import cKDTree

from morphogen import settings
from morphogen.distributions import Distribution
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.seeding import SeedLike, make_rng


def _size_biased(dist: Distribution, rng, m: int = 4096) -> float:
    """A value of the length-biased distribution ``x f(x) / E[x]`` (by sampling-
    importance-resampling, exact for constants)."""
    if dist.is_constant:
        return float(dist.mean())
    x = dist.sample(rng, m)
    return float(x[rng.choice(m, p=x / x.sum())])


class _Discs:
    """Union of discs in the plane of a layer (periodic or not)."""

    def __init__(self, centers: np.ndarray, radii: np.ndarray, lengths, periodic: bool):
        self.centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        self.radii = np.asarray(radii, dtype=float).reshape(-1)
        self.lengths = np.asarray(lengths, dtype=float)
        self.periodic = periodic
        self.rmax = float(self.radii.max(initial=0.0))
        self._lifted = self._plane = None

    def _wrap(self, p):
        if not self.periodic:
            return p
        p = np.mod(p, self.lengths)
        p[p >= self.lengths] = 0.0  # rounding of np.mod
        return p

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Whether the points (plane coordinates from the lower corner) are in a disc."""
        if not len(self.radii):
            return np.zeros(len(points), dtype=bool)
        if self._lifted is None:  # lifting: |p - c|^2 + rmax^2 - r^2 <= rmax^2
            extra = np.sqrt(np.maximum(self.rmax**2 - self.radii**2, 0.0))[:, None]
            box = [*self.lengths, 4.0 * self.rmax + 1.0] if self.periodic else None
            self._lifted = cKDTree(np.hstack([self._wrap(self.centers), extra]), boxsize=box)
        q = np.hstack([self._wrap(points), np.zeros((len(points), 1))])
        dist, _ = self._lifted.query(q, k=1, workers=settings.workers())
        return dist <= self.rmax * (1.0 + 1e-12)

    def signed(self, points: np.ndarray) -> np.ndarray:
        """``min_j |p - c_j| - r_j``: the signed distance to the union outside it,
        negative inside (with the sign of the union everywhere)."""
        out = np.full(len(points), np.inf)
        n = len(self.radii)
        if not n:
            return out
        if self._plane is None:
            box = list(self.lengths) if self.periodic else None
            self._plane = cKDTree(self._wrap(self.centers), boxsize=box)
        todo, k = np.arange(len(points)), min(n, 8)
        while todo.size:
            dist, idx = self._plane.query(self._wrap(points[todo]), k=k)
            dist, idx = dist.reshape(len(todo), k), idx.reshape(len(todo), k)
            out[todo] = np.min(dist - self.radii[idx], axis=1)
            # farther discs reach at most rmax inside their centre distance
            again = (dist[:, -1] - self.rmax < out[todo]) & (k < n)
            todo, k = todo[again], min(n, 2 * k)
        return out


@register_kind
class Laminate(Microstructure):
    """Layers of phase 1 (intervals along ``axis``) in phase 0, with holes."""

    kind = "laminate"

    def __init__(
        self,
        domain: Domain,
        axis: int,
        starts,
        ends,
        layer_ids=None,
        hole_layers=None,
        hole_centers=None,
        hole_radii=None,
        **kwargs,
    ) -> None:
        super().__init__(domain, **kwargs)
        self.axis = int(axis)
        self.starts = np.asarray(starts, dtype=float).reshape(-1)
        self.ends = np.asarray(ends, dtype=float).reshape(-1)
        order = np.argsort(self.starts)
        self.starts, self.ends = self.starts[order], self.ends[order]
        ids = np.arange(len(order)) if layer_ids is None else np.asarray(layer_ids)[order]
        self.layer_ids = np.asarray(ids, dtype=np.int64)
        self.hole_layers = np.asarray(
            [] if hole_layers is None else hole_layers, dtype=np.int64
        ).reshape(-1)
        self.hole_centers = np.asarray(
            [] if hole_centers is None else hole_centers, dtype=float
        ).reshape(-1, 2)
        self.hole_radii = np.asarray([] if hole_radii is None else hole_radii, dtype=float)
        self._discs: dict[int, _Discs] = {}

    @property
    def plane_axes(self) -> tuple[int, int]:
        return tuple(a for a in range(3) if a != self.axis)  # type: ignore[return-value]

    @property
    def n_layers(self) -> int:
        """Number of layers of phase 1 (a layer cut by a periodic face counts once)."""
        return int(np.unique(self.layer_ids).size)

    def discs(self, layer: int) -> _Discs:
        """Holes of a layer, in plane coordinates from the lower corner of the box."""
        if layer not in self._discs:
            sel = self.hole_layers == layer
            b, c = self.plane_axes
            lo = self.domain.lower[[b, c]]
            self._discs[layer] = _Discs(
                self.hole_centers[sel] - lo,
                self.hole_radii[sel],
                self.domain.lengths[[b, c]],
                self.domain.periodic,
            )
        return self._discs[layer]

    # -- point classification ---------------------------------------------------------------

    def _layer_at(self, z: np.ndarray) -> np.ndarray:
        """Index of the interval of phase 1 containing each coordinate (-1: none)."""
        k = np.searchsorted(self.starts, z, side="right") - 1
        inside = (k >= 0) & (z < self.ends[np.maximum(k, 0)])
        return np.where(inside, k, -1)

    def _in_phase(self, z: np.ndarray, plane: np.ndarray) -> np.ndarray:
        k = self._layer_at(z)
        out = k >= 0
        for i in np.unique(k[out]):
            sel = k == i
            out[sel] = ~self.discs(int(self.layer_ids[i])).inside(plane[sel])
        return out

    def _split(self, points: np.ndarray):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        b, c = self.plane_axes
        z = pts[:, self.axis]
        if self.domain.periodic:
            lo, length = self.domain.lower[self.axis], self.domain.lengths[self.axis]
            z = lo + np.mod(z - lo, length)
        return z, pts[:, [b, c]] - self.domain.lower[[b, c]]

    def contains(self, points) -> np.ndarray:
        return self._in_phase(*self._split(points))

    # -- discretization -------------------------------------------------------------------

    def _phases_on(self, grid: Grid) -> np.ndarray:
        b, c = self.plane_axes
        centres = [grid.domain.lower[a] + (np.arange(n) + 0.5) * h for a, n, h in
                   zip(range(3), grid.shape, grid.spacing, strict=True)]  # fmt: skip
        u, v = np.meshgrid(centres[b], centres[c], indexing="ij")
        plane = np.column_stack([u.ravel(), v.ravel()]) - self.domain.lower[[b, c]]
        k = self._layer_at(centres[self.axis])
        out = np.zeros((grid.shape[self.axis], grid.shape[b], grid.shape[c]), dtype=np.uint8)
        masks: dict[int, np.ndarray] = {}
        for i, layer in enumerate(k):
            if layer < 0:
                continue
            lid = int(self.layer_ids[layer])
            if lid not in masks:
                masks[lid] = (~self.discs(lid).inside(plane)).reshape(u.shape).astype(np.uint8)
            out[i] = masks[lid]
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
        """Minimum over the layers of phase 1 of ``max(slab, -holes)``: the signed distance
        to the slab along the axis, and minus the signed distance to the holes in the
        plane (negative in phase 1); only the layers around each point count."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        b, c = self.plane_axes
        centres = [grid.domain.lower[a] + (np.arange(n) + 0.5) * h for a, n, h in
                   zip(range(3), grid.shape, grid.spacing, strict=True)]  # fmt: skip
        u, v = np.meshgrid(centres[b], centres[c], indexing="ij")
        plane = np.column_stack([u.ravel(), v.ravel()]) - self.domain.lower[[b, c]]
        starts, ends, ids = self._extended_layers()
        out = np.full((grid.shape[self.axis], *u.shape), np.inf, dtype=np.float32)
        holes: dict[int, np.ndarray] = {}
        for i, z in enumerate(centres[self.axis]):
            k = int(np.searchsorted(starts, z, side="right")) - 1
            for j in (k, k + 1):  # the layer below (or containing) and the one above
                if not 0 <= j < len(starts):
                    continue
                slab = max(starts[j] - z, z - ends[j])
                lid = int(ids[j])
                if lid not in holes:
                    holes[lid] = -self.discs(lid).signed(plane).reshape(u.shape)
                out[i] = np.minimum(out[i], np.maximum(slab, holes[lid]))
        out = np.minimum(out, np.float32(np.max(grid.domain.lengths)))  # no layer: far
        data = np.moveaxis(out, 0, self.axis)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def _extended_layers(self):
        """The layers of phase 1 (starts, ends, ids), with those of the neighbouring periods
        of a periodic domain, joined at its faces."""
        starts, ends, ids = self.starts, self.ends, self.layer_ids
        if self.domain.periodic:
            length = self.domain.lengths[self.axis]
            starts = np.concatenate([starts - length, starts, starts + length])
            ends = np.concatenate([ends - length, ends, ends + length])
            ids = np.tile(ids, 3)
            cut = np.flatnonzero(
                (np.abs(ends[:-1] - starts[1:]) <= 1e-12 * length) & (ids[:-1] == ids[1:])
            )
            ends[cut] = ends[cut + 1]
            keep = np.ones(len(starts), dtype=bool)
            keep[cut + 1] = False
            starts, ends, ids = starts[keep], ends[keep], ids[keep]
        return starts, ends, ids

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """As :meth:`level_set`, at the points: the minimum over the layers around each
        point of ``max(slab, -holes)``."""
        z, plane = self._split(points)
        starts, ends, ids = self._extended_layers()
        out = np.full(len(z), np.inf)
        k = np.searchsorted(starts, z, side="right") - 1
        for j in (k, k + 1):  # the layer below (or containing) and the one above
            ok = np.flatnonzero((j >= 0) & (j < len(starts)))
            for lid in np.unique(ids[j[ok]]):
                sel = ok[ids[j[ok]] == lid]
                slab = np.maximum(starts[j[sel]] - z[sel], z[sel] - ends[j[sel]])
                holes = -self.discs(int(lid)).signed(plane[sel])
                out[sel] = np.minimum(out[sel], np.maximum(slab, holes))
        return np.minimum(out, float(np.max(self.domain.lengths)))  # no layer: far

    # -- description ------------------------------------------------------------------------

    def summary(self) -> dict:
        return {"n_layers": self.n_layers, "n_holes": len(self.hole_radii)}

    def _geometry(self) -> dict:
        return {
            "axis": self.axis,
            "starts": self.starts.tolist(),
            "ends": self.ends.tolist(),
            "layer_ids": self.layer_ids.tolist(),
            "hole_layers": self.hole_layers.tolist(),
            "hole_centers": self.hole_centers.tolist(),
            "hole_radii": self.hole_radii.tolist(),
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            geometry["axis"],
            geometry["starts"],
            geometry["ends"],
            geometry.get("layer_ids"),
            geometry.get("hole_layers"),
            geometry.get("hole_centers"),
            geometry.get("hole_radii"),
            **kwargs,
        )


@register
@dataclass(frozen=True)
class RandomLaminate(Model):
    """Layers of random thicknesses stacked along an axis, with holes through the layers
    of phase 1 (nanolayered films, Bironeau et al. 2016)."""

    name = "random_laminate"
    phases: ClassVar[dict[int, str]] = {0: "layers of phase 0", 1: "layers of phase 1"}
    count_label = "layers"

    thickness: Distribution = param(
        doc="Thickness of the layers of phase 1", kind="distribution", unit="length"
    )
    spacing: Distribution = param(
        doc="Thickness of the layers of phase 0 between them", kind="distribution", unit="length"
    )
    axis: int = param(2, doc="Stacking direction (0: x, 1: y, 2: z)", kind="int")
    hole_fraction: float = param(
        0.0,
        doc="Fraction of the area of each layer of phase 1 taken by holes through it",
        kind="fraction",
    )
    hole_radius: Distribution | None = param(
        None,
        doc="Radius of the holes (discs across the layers)",
        kind="distribution",
        unit="length",
    )

    def _validate(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError("axis: 0, 1 or 2")
        if not 0.0 <= self.hole_fraction < 1.0:
            raise ValueError(f"hole_fraction: must lie in [0, 1), got {self.hole_fraction!r}")
        if self.hole_fraction > 0 and self.hole_radius is None:
            raise ValueError("hole_radius: needed for holes (hole_fraction > 0)")

    @property
    def period(self) -> float:
        """Mean thickness of a pair of layers."""
        return self.thickness.mean() + self.spacing.mean()

    def _hole_intensity(self) -> float:
        if not self.hole_fraction > 0:
            return 0.0
        return -math.log1p(-self.hole_fraction) / (math.pi * self.hole_radius.moment(2))

    def _stack(self, rng, lower: float, length: float, periodic: bool):
        """Intervals of phase 1 along the axis, with their layer ids."""
        t, s = self.thickness, self.spacing
        if periodic:
            pairs = []
            total = 0.0
            while total < length:
                pairs.append((float(t.sample(rng, 1)[0]), float(s.sample(rng, 1)[0])))
                total += sum(pairs[-1])
            # the number of whole periods whose total is the closest to the box
            sums = np.cumsum([a + b for a, b in pairs])
            n = len(sums)
            if n > 1 and abs(sums[-2] - length) < abs(sums[-1] - length):
                n -= 1
            scale = length / sums[n - 1]
            shift = rng.random() * length
            starts, ends, ids = [], [], []
            z = shift
            for i, (a, b) in enumerate(pairs[:n]):
                z0, z1 = z, z + a * scale
                for lo, hi in ((z0, z1), (z0 - length, z1 - length)):  # wrapped at the top
                    lo, hi = max(lo, 0.0), min(hi, length)
                    if hi > lo:
                        starts.append(lower + lo)
                        ends.append(lower + hi)
                        ids.append(i)
                z = (z1 + b * scale) % length
            return np.array(starts), np.array(ends), np.array(ids, dtype=np.int64), scale
        # stationary start: the layer covering the lower face, cut by it
        mt, ms = t.mean(), s.mean()
        phase = int(rng.random() < mt / (mt + ms))
        first = _size_biased(t if phase else s, rng) * (1.0 - rng.random())
        starts, ends = [], []
        z = 0.0
        width = first
        while z < length:
            if phase:
                starts.append(lower + z)
                ends.append(lower + min(z + width, length))
            z += width
            phase = 1 - phase
            width = float((t if phase else s).sample(rng, 1)[0])
        return np.array(starts), np.array(ends), np.arange(len(starts)), 1.0

    def generate(self, domain: Domain, seed: SeedLike = None) -> Laminate:
        self._check_domain(domain)
        rng, seed_info = make_rng(seed)
        a = self.axis
        starts, ends, ids, scale = self._stack(
            rng, float(domain.lower[a]), float(domain.lengths[a]), domain.periodic
        )
        b, c = (d for d in range(3) if d != a)
        lam = self._hole_intensity()
        layers, centres, radii = [], [], []
        if lam > 0:
            rmax = self.hole_radius.upper()
            margin = 0.0 if domain.periodic else rmax
            lo = domain.lower[[b, c]] - margin
            size = domain.lengths[[b, c]] + 2.0 * margin
            for layer in np.unique(ids):
                n = rng.poisson(lam * size[0] * size[1])
                layers.append(np.full(n, layer))
                centres.append(lo + rng.random((n, 2)) * size)
                radii.append(self.hole_radius.sample(rng, n))
        metadata = {"thickness_scale": scale} if domain.periodic else None
        return Laminate(
            domain,
            a,
            starts,
            ends,
            ids,
            np.concatenate(layers) if layers else None,
            np.concatenate(centres) if centres else None,
            np.concatenate(radii) if radii else None,
            model=self,
            seed=seed_info,
            metadata=metadata,
        )

    def expected_count(self, domain: Domain) -> float:
        """Mean number of layers of phase 1 across the domain."""
        return float(domain.lengths[self.axis]) / self.period

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        mt, ms = self.thickness.mean(), self.spacing.mean()
        p = mt / (mt + ms)
        q = self.hole_fraction
        walls = (
            mt
            * self._hole_intensity()
            * 2.0
            * math.pi
            * (self.hole_radius.mean() if q > 0 else 0.0)
        )
        out = {
            "volume_fraction": p * (1.0 - q),
            "specific_surface": (1.0 - q) * (2.0 + walls) / (mt + ms),
            "layer_period": mt + ms,
            "integral_range": math.inf,
        }
        if q == 0:
            vt, vs = self.thickness.std() ** 2, self.spacing.std() ** 2
            out["integral_range_1d"] = (ms**2 * vt + mt**2 * vs) / ((mt + ms) * mt * ms)
        return out
