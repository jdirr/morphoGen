"""Haversian (cortical) bone: osteons remodelled one after another.

Cortical bone is renewed by basic multicellular units that tunnel along the bone axis,
resorb a cylinder of bone and refill it with concentric lamellae around a vascular
(Haversian) canal: a secondary osteon, bounded by a cement line. Each new osteon cuts
through the older ones, whose fragments, with the remains of older bone, form the
interstitial bone (Currey 2002). The model is a dead-leaves model of aligned cylinders
(Matheron 1968; Jeulin 1989): osteons of radius ``R`` arrive at Poisson positions of
the cross-section in random order, the last on top; a point covered by none is
interstitial bone. Within the top osteon, it lies in the canal (``d < rho R``), the
cement line (``d > R - c``) or the lamellar bone in between (``d``: its distance to
the axis of the osteon).

With ``lambda`` osteons per unit area chosen for a ``coverage`` ``p = 1 - exp(-lambda pi
E[R^2])``, the top osteon is an osteon weighted by its area, and a point falls in each
of its parts with the share of its area (exact):

* Haversian canals (the vascular porosity): ``p rho^2``;
* cement lines: ``p E[R^2 - max(R - c, rho R)^2] / E[R^2]``;
* lamellar bone of the osteons: the rest of ``p``; interstitial bone: ``1 - p``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from morphogen._utils import check_fraction, check_positive
from morphogen.distributions import Distribution
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.models.planar import pixel_centres, plane_axes
from morphogen.seeding import SeedLike, make_rng

INTERSTITIAL, LAMELLAR, CEMENT, CANAL = 0, 1, 2, 3


@njit(cache=True)
def _osteon_levels(offsets, ptr, osteons, radii, canal_ratio, cement, reach, out):
    """The level sets of the lamellar bone, cement lines, canals and interstitial bone at
    points of the cross-section (``out``: (n, 4)), their candidate osteons from the top one down
    (``osteons[ptr[p]:ptr[p + 1]]``, the offsets of the points from their axes): for each
    phase, the minimum over the osteons ``i`` of ``max(s_i, -min_{j above i} (d_j -
    r_j))``, ``s_i`` the signed distance to the disc or annulus of the phase in osteon
    ``i``, within ``[-reach, reach]``."""
    for p in range(len(ptr) - 1):
        lowest = np.inf
        best0 = np.inf
        best1 = np.inf
        best2 = np.inf
        for k in range(ptr[p], ptr[p + 1]):
            i = osteons[k]
            d = math.hypot(offsets[k, 0], offsets[k, 1])
            r = radii[i]
            above = -lowest
            inner = canal_ratio * r
            outer = r - cement
            best0 = min(best0, max(max(inner - d, d - outer), above))  # lamellar
            best1 = min(best1, max(max(outer - d, d - r), above))  # cement line
            best2 = min(best2, max(d - inner, above))  # canal
            lowest = min(lowest, d - r)
            if lowest <= -reach:
                break
        out[p, 0] = min(max(best0, -reach), reach)
        out[p, 1] = min(max(best1, -reach), reach)
        out[p, 2] = min(max(best2, -reach), reach)
        out[p, 3] = min(max(-lowest, -reach), reach)  # interstitial: no osteon


@register_kind
class OsteonStructure(Microstructure):
    """Osteons along ``axis``: discs of the cross-section (centres, radii) in the order
    of their arrival, the last on top, with a canal of ``canal_ratio`` times their
    radius and a cement line of ``cement`` thickness."""

    kind = "osteons"

    def __init__(self, domain, axis, centres, radii, canal_ratio, cement, **kwargs):
        super().__init__(domain, **kwargs)
        self.axis = int(axis)
        self.centres = np.asarray(centres, dtype=float).reshape(-1, 2)
        self.radii = np.asarray(radii, dtype=float).reshape(-1)
        self.canal_ratio = float(canal_ratio)
        self.cement = float(cement)

    @property
    def n_osteons(self) -> int:
        return len(self.radii)

    def _top(self, plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The last osteon over each point of the cross-section (-1: none) and the
        distance of the point to its axis."""
        b, c = plane_axes(self.axis)
        lower, lengths = self.domain.lower[[b, c]], self.domain.lengths[[b, c]]
        periodic = self.domain.periodic
        q = plane - lower
        if periodic:
            q = np.mod(q, lengths)
            q[q >= lengths] = 0.0
        top = np.full(len(q), -1, dtype=np.int64)
        dist = np.full(len(q), np.inf)
        if not len(q) or not self.n_osteons:
            return top, dist
        tree = cKDTree(q, boxsize=lengths if periodic else None)
        centres = self.centres - lower
        if periodic:
            centres = np.mod(centres, lengths)
            centres[centres >= lengths] = 0.0
        for i in range(self.n_osteons):  # in order of arrival: the later on top
            hits = tree.query_ball_point(centres[i], self.radii[i], return_sorted=False)
            if not hits:
                continue
            hits = np.asarray(hits, dtype=np.int64)
            v = q[hits] - centres[i]
            if periodic:
                v -= lengths * np.round(v / lengths)
            top[hits] = i
            dist[hits] = np.hypot(v[:, 0], v[:, 1])
        return top, dist

    def _phases(self, plane: np.ndarray) -> np.ndarray:
        top, dist = self._top(plane)
        out = np.full(len(plane), INTERSTITIAL, dtype=np.uint8)
        ok = top >= 0
        r = self.radii[top[ok]]
        d = dist[ok]
        out[ok] = np.where(
            d < self.canal_ratio * r, CANAL, np.where(d > r - self.cement, CEMENT, LAMELLAR)
        )
        return out

    def phase_level_sets_at(self, points, reach: float = np.inf) -> dict[int, np.ndarray]:
        """The level sets of the lamellar bone (1), cement lines (2), canals (3) and
        interstitial bone (0) at points: in the cross-section, the parts of each osteon not
        covered by later ones, and the space no osteon covers (see :func:`_osteon_levels`),
        within ``[-reach, reach]`` (default: the largest radius)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        b, c = plane_axes(self.axis)
        lower, lengths = self.domain.lower[[b, c]], self.domain.lengths[[b, c]]
        periodic = self.domain.periodic
        rmax = float(self.radii.max(initial=0.0))
        reach = rmax if not np.isfinite(reach) else float(reach)
        out = np.full((len(pts), 4), reach)
        out[:, 3] = -reach  # no osteon: interstitial
        if self.n_osteons:
            centres = self.centres - lower
            q = pts[:, [b, c]] - lower
            if periodic:
                centres = np.mod(centres, lengths)
                centres[centres >= lengths] = 0.0
                q = np.mod(q, lengths)
                q[q >= lengths] = 0.0
            tree = cKDTree(centres, boxsize=lengths if periodic else None)
            chunk = 8192
            for s in range(0, len(q), chunk):
                lists = tree.query_ball_point(q[s : s + chunk], rmax + reach)
                counts = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
                ost = np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int64,
                                  count=int(counts.sum()))  # fmt: skip
                owner = np.repeat(np.arange(len(lists)), counts)
                order = np.lexsort((-ost, owner))  # from the top osteon down
                ost, owner = ost[order], owner[order]
                offsets = q[s : s + chunk][owner] - centres[ost]
                if periodic:
                    offsets -= lengths * np.round(offsets / lengths)
                ptr = np.concatenate([[0], np.cumsum(counts)])
                _osteon_levels(np.ascontiguousarray(offsets), ptr, ost, self.radii,
                               self.canal_ratio, self.cement, reach,
                               out[s : s + chunk])  # fmt: skip
        return {INTERSTITIAL: out[:, 3], LAMELLAR: out[:, 0], CEMENT: out[:, 1], CANAL: out[:, 2]}

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """The function of the lamellar bone (phase 1) at the points
        (:meth:`phase_level_sets_at`)."""
        return self.phase_level_sets_at(points, reach)[LAMELLAR]

    def contains(self, points) -> np.ndarray:
        """Whether points lie in the lamellar bone of the osteons (phase 1)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        b, c = plane_axes(self.axis)
        return self._phases(pts[:, [b, c]]) == LAMELLAR

    # -- implicit functions ----------------------------------------------------------------

    def implicit_phases(self) -> list[int]:
        return [INTERSTITIAL, LAMELLAR, CEMENT, CANAL]

    def _implicit(self, grid: Grid) -> np.ndarray:
        """The implicit functions of the four phases in the cross-section of a grid (kept
        for the grid last asked): the rings of each osteon (its canal, its lamellae, its
        cement line: differences of distances to its axis) where no later osteon covers
        them, and the interstitial bone that none covers."""
        from morphogen.models.implicit import band_of, grid_key, lattice_boxes, ordered_cover

        key = grid_key(grid)
        cached = getattr(self, "_fields", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        band = band_of(grid)
        axes = plane_axes(self.axis)
        lengths = self.domain.lengths if self.domain.periodic else None
        shape = tuple(grid.shape[a] for a in axes)

        def objects():
            for i in range(self.n_osteons - 1, -1, -1):  # the later on top
                radius = self.radii[i]
                canal, cement = self.canal_ratio * radius, radius - self.cement
                reach = radius + band
                for slices, (du, dv) in lattice_boxes(self.centres[i], reach, grid, lengths, axes):
                    r = np.hypot(du, dv).astype(np.float32)
                    parts = [None] * 4
                    parts[LAMELLAR] = np.maximum(canal - r, r - cement)
                    parts[CEMENT] = np.maximum(cement - r, r - radius)
                    parts[CANAL] = r - canal
                    yield slices, r - radius, parts

        parts, top = ordered_cover(shape, band, objects(), 4)
        parts[INTERSTITIAL] = -top
        self._fields = (key, parts)
        return parts

    def phase_level_set(
        self, phase: int, shape=None, *, voxel_size=None, grid: Grid | None = None
    ) -> VoxelImage:
        """The implicit function of a phase (see :meth:`Microstructure.phase_level_set`),
        the same along the axis of the osteons."""
        phase = int(phase)
        if phase not in self.implicit_phases():
            raise ValueError(f"phase {phase}: the phases are {self.implicit_phases()}")
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        plane = self._implicit(grid)[phase]
        out = np.broadcast_to(plane[:, :, None], (*plane.shape, grid.shape[self.axis]))
        data = np.ascontiguousarray(np.moveaxis(out, 2, self.axis))
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """The implicit function of the lamellar bone of the osteons (phase 1)."""
        return self.phase_level_set(LAMELLAR, shape, voxel_size=voxel_size, grid=grid)

    def _image(self, grid: Grid) -> np.ndarray:
        plane, shape = pixel_centres(grid, plane_axes(self.axis))
        image = self._phases(plane).reshape(shape)
        out = np.broadcast_to(image[:, :, None], (*shape, grid.shape[self.axis]))
        return np.ascontiguousarray(np.moveaxis(out, 2, self.axis))

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if ss == 1:
            data = self._image(grid)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        ph = (self._image(fine) == LAMELLAR).astype(np.float32)
        nx, ny, nz = grid.shape
        data = ph.reshape(nx, ss, ny, ss, nz, ss).mean(axis=(1, 3, 5), dtype=np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def summary(self) -> dict:
        return {"n_osteons": self.n_osteons}

    def _geometry(self) -> dict:
        return {
            "axis": self.axis,
            "centres": self.centres.tolist(),
            "radii": self.radii.tolist(),
            "canal_ratio": self.canal_ratio,
            "cement": self.cement,
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            geometry["axis"],
            geometry["centres"],
            geometry["radii"],
            geometry["canal_ratio"],
            geometry["cement"],
            **kwargs,
        )


@register
@dataclass(frozen=True)
class Osteons(Model):
    """Haversian bone: aligned osteons with canals and cement lines, remodelled one
    after another over a fraction ``coverage`` of the cross-section (a dead-leaves
    model of cylinders along ``axis``)."""

    name = "osteons"
    phases: ClassVar[dict[int, str]] = {
        INTERSTITIAL: "interstitial bone",
        LAMELLAR: "osteons (lamellar bone)",
        CEMENT: "cement lines",
        CANAL: "Haversian canals",
    }
    has_level_set = False
    count_label = "osteons"

    radius: Distribution = param(
        1.0, doc="Radius of the osteons", kind="distribution", unit="length"
    )
    coverage: float = param(
        0.7, doc="Fraction of the cross-section remodelled into osteons", kind="fraction"
    )
    canal_ratio: float = param(
        0.2, doc="Radius of the Haversian canals, relative to the osteons", kind="fraction"
    )
    cement_thickness: float = param(0.03, doc="Thickness of the cement lines", unit="length")
    axis: int = param(2, doc="Axis of the osteons (the bone axis: 0: x, 1: y, 2: z)", kind="int")

    def _validate(self) -> None:
        check_fraction(self.coverage, "coverage")
        check_fraction(self.canal_ratio, "canal_ratio")
        check_positive(self.cement_thickness, "cement_thickness", allow_zero=True)
        if self.axis not in (0, 1, 2):
            raise ValueError("axis: 0, 1 or 2")

    @property
    def intensity(self) -> float:
        """Osteons per unit area of the cross-section."""
        return -math.log1p(-self.coverage) / (math.pi * self.radius.moment(2))

    def generate(self, domain: Domain, seed: SeedLike = None) -> OsteonStructure:
        rng, seed_info = make_rng(seed)
        b, c = plane_axes(self.axis)
        rmax = self.radius.upper()
        margin = 0.0 if domain.periodic else rmax
        lower = domain.lower[[b, c]] - margin
        size = domain.lengths[[b, c]] + 2.0 * margin
        n = int(rng.poisson(self.intensity * size[0] * size[1]))
        centres = lower + rng.random((n, 2)) * size
        radii = self.radius.sample(rng, n)
        if domain.periodic and np.any(2.0 * radii > size.min()):
            raise ValueError("osteons: a periodic box wider than the osteons")
        return OsteonStructure(
            domain,
            self.axis,
            centres,
            radii,
            self.canal_ratio,
            self.cement_thickness,
            model=self,
            seed=seed_info,
        )

    def expected_count(self, domain: Domain) -> float:
        """Mean number of osteons across the domain (cut ones included)."""
        b, c = plane_axes(self.axis)
        return float(self.intensity * domain.lengths[b] * domain.lengths[c])

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        p, rho, t = self.coverage, self.canal_ratio, self.cement_thickness
        r2 = self.radius.moment(2)
        inner = self.radius.expect(lambda r: np.maximum(r - t, rho * r) ** 2)
        canal = p * rho * rho
        cement = p * (r2 - inner) / r2
        return {
            "intensity": self.intensity,
            "volume_fraction_0": 1.0 - p,
            "volume_fraction_1": p - canal - cement,
            "volume_fraction_2": cement,
            "volume_fraction_3": canal,
            "porosity": canal,
        }
