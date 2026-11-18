"""Dead leaves models (Matheron 1968; Jeulin 1997).

Leaves fall one after the other, each hiding what it covers; after infinitely many,
every point belongs to the last leaf that covered it. The perfect simulation of Kendall
& Thönnes (1999) runs the time backwards: leaves are drawn one by one, each going under
the earlier ones, so that a point belongs to the *first* leaf drawn that covers it, and
the drawing stops once the domain is covered. Here it stops once the voxel centres of a
check grid are covered, and a third more leaves are drawn, which leaves a negligible
part uncovered; voxels left uncovered at finer resolutions take the leaf of the
nearest centre.

The leaves are spheres, or spheroids (``shape: spheroid``: semi-axes ``a``, ``a`` and
``aspect_ratio * a`` about an axis drawn from ``orientation``), of random sizes. Each
phase may have its own law of sizes (``phase_radius``: families of leaves, the
multicomponent dead leaves of Jeulin), ``phase_fractions`` being the fractions of space
of the phases: the leaves of phase ``i`` fall in the proportion ``p_i ∝ q_i / E[V_i]``.
A finite ``coverage`` ``c`` (the mean number of leaves falling on a point) stops the
fall: the points that no leaf covers, a fraction ``exp(-c)``, keep the ``background``
phase (sequential dead leaves, from the Boolean model for small ``c`` to the dead
leaves as ``c`` grows).

The visible parts of the leaves form a random tessellation with non-convex cells: the
cell label of a voxel is the index of its leaf (the background, if any, is the last
label), and the leaves can carry crystal orientations (polycrystals of interpenetrating
grains).

Theory (Jeulin 1997). Let ``K_i(h)`` be the mean geometric covariogram of the leaves of
family ``i`` (``K_i(0) = E[V_i]``), ``κ_i = K_i / K_i(0)`` and ``c_i = c q_i`` the mean
number of its leaves over a point. Going back in time, the leaves hitting both ``x`` and
``x + h`` come at the rate ``a_i = c_i κ_i``, those hitting ``x`` only at ``b_i = c_i (1 -
κ_i)``, those hitting either at ``λ = Σ c_i (2 - κ_i)``, and those hitting ``x + h`` at
``μ = c``:

* ``P(x, x + h in the same leaf of i) = (a_i / λ) (1 - e^{-λ})``;
* ``P(x in a leaf of i, x + h in another of j) = (b_i q_j + b_j q_i)(I_1 - I_2)``, with
  ``I_1 = (1 - e^{-λ}) / λ`` and ``I_2 = (e^{-λ} - e^{-μ}) / (μ - λ)``;
* ``P(x in a leaf of i, x + h uncovered) = b_i I_2``, ``P(both uncovered) = e^{-λ}``.

With complete coverage (``c → ∞``) they become ``a_i / λ`` and ``(b_i q_j + b_j q_i) /
λ``: one family of phases of probabilities ``p_i`` gives the covariance ``p_i^2 + p_i (1 -
p_i) r(h)`` of phase ``i``, ``r(h) = K(h) / (2K(0) - K(h))``. A sphere of radius ``R`` and
volume ``V`` has ``K(h) = V (1 - 3h/(4R) + h^3/(16R^3))`` up to ``2R``; a spheroid, an
affine image of the ball, ``K(h w) = V k(h sqrt((1 - t^2)/a^2 + t^2/c^2))``, with ``t =
w · axis`` and ``k(s) = 1 - 3s/4 + s^3/16`` (``s < 2``), the covariogram of the unit ball
over its volume.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from morphogen import settings
from morphogen._utils import check_positive
from morphogen.distributions import Distribution
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.kernels.leaves import first_cover
from morphogen.models.affine import ANISOTROPY, Anisotropy, anisotropic
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.orientations import (
    Aligned,
    Isotropic,
    OrientationDistribution,
    random_orientations,
)
from morphogen.seeding import SeedLike, make_rng

#: Shapes of the leaves.
SHAPES = ("sphere", "spheroid")
_NO_AXES = np.zeros((1, 3))


@njit(cache=True)
def _visible_levels(offsets, ptr, leaves, radii, axes, aspect, leaf_phases, background, reach,
                    out):  # fmt: skip
    """The level sets of the visible parts of the leaves of each phase (columns of
    ``out``) at points (their candidate leaves in order, ``leaves[ptr[p]:ptr[p + 1]]``, and
    the offsets of the points from their centres): for each phase, the minimum over its
    leaves ``i`` of ``max(d_i, -min_{j < i} d_j)``, ``d`` the signed distance to a leaf (the
    ellipsoidal radius minus one times the smallest semi-axis for spheroids: a lower bound
    of the distance outside, 1-Lipschitz), and ``-min_j d_j`` too for the ``background``
    phase (the uncovered space; -1: none); within ``[-reach, reach]``."""
    small = min(1.0, aspect)
    n_phases = out.shape[1]
    best = np.empty(n_phases)
    for p in range(len(ptr) - 1):
        lowest = np.inf
        best[:] = np.inf
        for k in range(ptr[p], ptr[p + 1]):
            i = leaves[k]
            dx, dy, dz = offsets[k, 0], offsets[k, 1], offsets[k, 2]
            d2 = dx * dx + dy * dy + dz * dz
            r = radii[i]
            if aspect == 1.0:
                d = math.sqrt(d2) - r
            else:
                t = dx * axes[i, 0] + dy * axes[i, 1] + dz * axes[i, 2]
                q = (d2 - t * t) / (r * r) + t * t / (aspect * aspect * r * r)
                d = (math.sqrt(q) - 1.0) * r * small
            ph = leaf_phases[i]
            best[ph] = min(best[ph], max(d, -lowest))
            lowest = min(lowest, d)
            if lowest <= -reach:  # the leaves below are hidden beyond the reach
                break
        if background >= 0:
            best[background] = min(best[background], -lowest)
        for ph in range(n_phases):
            out[p, ph] = min(max(best[ph], -reach), reach)


def _cover(centers, radii, axes, aspect, grid: Grid, periodic: bool, out=None, first: int = 0):
    """Labels of the first leaves covering the voxel centres of a grid (-1: none)."""
    if out is None:
        out = np.full(grid.shape, -1, dtype=np.int32)
    left = first_cover(
        np.ascontiguousarray(centers, dtype=float),
        np.ascontiguousarray(radii, dtype=float),
        _NO_AXES if axes is None else np.ascontiguousarray(axes, dtype=float),
        float(aspect),
        first,
        np.asarray(grid.domain.lower, dtype=float),
        np.asarray(grid.spacing, dtype=float),
        np.asarray(grid.shape, dtype=np.int64),
        periodic,
        out,
    )
    return out, left


@register_kind
class DeadLeaves(Microstructure):
    """Leaves in the order they cover the space (the first one is on top), and the
    background that they leave uncovered (finite coverage; the last cell)."""

    kind = "dead_leaves"
    has_labels = True

    def __init__(
        self,
        domain: Domain,
        centers,
        radii,
        leaf_phases,
        orientations=None,
        *,
        axes=None,
        aspect: float = 1.0,
        background: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(domain, **kwargs)
        self.centers = np.ascontiguousarray(centers, dtype=float).reshape(-1, 3)
        self.radii = np.asarray(radii, dtype=float).reshape(-1)
        self.leaf_phases = np.asarray(leaf_phases, dtype=np.uint8).reshape(-1)
        n = len(self.radii)
        self.aspect = float(aspect)
        self.axes = None if axes is None else np.ascontiguousarray(axes, dtype=float).reshape(n, 3)
        self.background = None if background is None else int(background)
        # the phase of each cell: the leaves, then the background
        self.cell_phases = self.leaf_phases
        if self.background is not None:
            self.cell_phases = np.append(self.leaf_phases, np.uint8(self.background))
        self.orientations = (
            None
            if orientations is None
            else np.asarray(orientations, dtype=float).reshape(len(self.cell_phases), 3)
        )
        self._tree = None

    @property
    def n_leaves(self) -> int:
        return len(self.radii)

    def _local(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float) - self.domain.lower
        if self.domain.periodic:
            p = np.mod(p, self.domain.lengths)
            p[p >= self.domain.lengths] = 0.0
        return p

    def _centres_tree(self) -> cKDTree:
        if self._tree is None:
            if self.domain.periodic:
                self._tree = cKDTree(self._local(self.centers), boxsize=self.domain.lengths)
            else:
                self._tree = cKDTree(self.centers - self.domain.lower)
        return self._tree

    def _nearest(self, points: np.ndarray) -> np.ndarray:
        """Leaf of the nearest centre (points that no leaf covers, complete coverage)."""
        return self._centres_tree().query(self._local(points), k=1, workers=settings.workers())[1]

    def _uncovered(self, points: np.ndarray) -> np.ndarray:
        """The cell of points that no leaf covers: the nearest leaf, or the background."""
        if self.background is not None:
            return np.full(len(points), self.n_leaves, dtype=np.int64)
        return self._nearest(points)

    def labels(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Image of the visible leaves (int32 leaf indices; the background: ``n_leaves``)."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        out, left = _cover(self.centers, self.radii, self.axes, self.aspect, grid,
                           self.domain.periodic)  # fmt: skip
        if left:
            missing = out < 0
            out[missing] = self._uncovered(grid.center_points()[missing.ravel()])
        return VoxelImage.from_grid(out, grid, name="cell", metadata=self.image_metadata())

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if ss == 1:
            data = self.cell_phases[self.labels(grid=grid).data]
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        ph = (self.cell_phases[self.labels(grid=fine).data] == 1).astype(np.float32)
        nx, ny, nz = grid.shape
        data = ph.reshape(nx, ss, ny, ss, nz, ss).mean(axis=(1, 3, 5), dtype=np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def _inside(self, leaves: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Whether the offsets ``d`` from the centres of ``leaves`` lie in them."""
        r2 = self.radii[leaves] ** 2
        d2 = np.einsum("ij,ij->i", d, d)
        if self.axes is None or self.aspect == 1.0:
            return d2 <= r2
        t = np.einsum("ij,ij->i", d, self.axes[leaves])
        return (d2 - t * t) / r2 + t * t / (self.aspect**2 * r2) <= 1.0

    def leaf_of(self, points) -> np.ndarray:
        """Visible cell at each point: its leaf, or the background (``n_leaves``)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        out = np.full(len(pts), -1, dtype=np.int64)
        if self.n_leaves:
            tree = self._centres_tree()
            p = self._local(pts)
            reach = float(self.radii.max()) * max(1.0, self.aspect)
            for i, cand in enumerate(tree.query_ball_point(p, reach)):
                if not cand:
                    continue
                cand = np.asarray(cand)
                d = tree.data[cand] - p[i]
                if self.domain.periodic:
                    d -= self.domain.lengths * np.round(d / self.domain.lengths)
                inside = cand[self._inside(cand, d)]
                if inside.size:
                    out[i] = inside.min()
        missing = out < 0
        if missing.any():
            out[missing] = self._uncovered(pts[missing]) if self.n_leaves else self.n_leaves
        return out

    def contains(self, points) -> np.ndarray:
        return self.cell_phases[self.leaf_of(points)] == 1

    # -- implicit functions ----------------------------------------------------------------

    def implicit_phases(self) -> list[int]:
        return sorted(int(k) for k in np.unique(self.cell_phases))

    def _leaf_function(self, i: int, dx, dy, dz):
        """The implicit function of the leaf ``i`` at offsets from its centre: the
        distance to a sphere; for a spheroid, its radius times the gauge of the point
        minus one (zero on it, the distance to it along its equator)."""
        r = self.radii[i]
        d2 = dx * dx + dy * dy + dz * dz
        if self.axes is None or self.aspect == 1.0:
            return np.sqrt(d2) - r
        a = self.axes[i]
        t = dx * a[0] + dy * a[1] + dz * a[2]
        return r * (np.sqrt(np.maximum(d2 - t * t, 0.0) + t * t / self.aspect**2) / r - 1.0)

    def _implicit(self, grid: Grid) -> dict[int, np.ndarray]:
        """The implicit functions of all the phases on a grid (kept for the grid last
        asked): the leaves of a phase where no leaf on top of them covers them, and the
        part of what no leaf covers that the phase takes (the background, or the points
        nearest to the centres of its leaves)."""
        from morphogen.models.implicit import band_of, grid_key, lattice_boxes, ordered_cover

        key = grid_key(grid)
        cached = getattr(self, "_fields", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        band = band_of(grid)
        phases = self.implicit_phases()
        index = {k: n for n, k in enumerate(phases)}
        lengths = self.domain.lengths if self.domain.periodic else None
        stretch = max(1.0, self.aspect)

        def objects():
            for i in range(self.n_leaves):
                reach = self.radii[i] * stretch + band
                for slices, offsets in lattice_boxes(self.centers[i], reach, grid, lengths):
                    f = self._leaf_function(i, *offsets).astype(np.float32)
                    parts = [None] * len(phases)
                    parts[index[int(self.leaf_phases[i])]] = f
                    yield slices, f, parts

        parts, top = ordered_cover(grid.shape, band, objects(), len(phases))
        fields = {k: parts[n] for k, n in index.items()}
        uncovered = -top  # negative where no leaf covers
        if self.background is not None:
            np.minimum(fields[self.background], uncovered, out=fields[self.background])
        elif self.n_leaves and np.any(uncovered < 0):  # the leaf of the nearest centre
            nx, ny, nz = grid.shape
            tree = self._centres_tree()
            scale = 2.0 * float(np.mean(self.radii))
            for k in phases:
                own = cKDTree(tree.data[self.leaf_phases == k], boxsize=tree.boxsize)
                rest = self.leaf_phases != k
                share = np.full(grid.shape, -band, dtype=np.float32)
                if rest.any():
                    others = cKDTree(tree.data[rest], boxsize=tree.boxsize)
                    q = self._local(grid.center_points())
                    d = own.query(q, workers=settings.workers())[0] ** 2
                    d -= others.query(q, workers=settings.workers())[0] ** 2
                    share = np.clip(d / scale, -band, band).reshape(nx, ny, nz).astype(np.float32)
                np.minimum(fields[k], np.maximum(uncovered, share), out=fields[k])
        self._fields = (key, fields)
        return fields

    def phase_level_set(
        self, phase: int, shape=None, *, voxel_size=None, grid: Grid | None = None
    ) -> VoxelImage:
        """The implicit function of a phase (see :meth:`Microstructure.phase_level_set`):
        the leaves of the phase outside the leaves on top of them, each by the distance to
        its sphere (spheroids: a function that vanishes on them), clipped to a band of
        four voxels."""
        phase = int(phase)
        if phase not in self.implicit_phases():
            raise ValueError(f"phase {phase}: the phases are {self.implicit_phases()}")
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        data = self._implicit(grid)[phase]
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """The implicit function of phase 1 (:meth:`phase_level_set`)."""
        return self.phase_level_set(1, shape, voxel_size=voxel_size, grid=grid)

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """Two phases: the level set of the visible parts of the leaves of phase 1 (each
        leaf minus those above it), see :func:`_visible_levels`, within ``[-reach, reach]``
        (default: the largest radius); the points that no leaf covers are of phase 0 (of
        the background phase, with a finite coverage). Its zero set is the boundary of
        phase 1, sharp edges included: the exact meshes follow it. More phases: one level
        set per phase (:meth:`phase_level_sets_at`), of which this is that of phase 1."""
        if 1 not in self.implicit_phases():
            raise NotImplementedError("dead leaves without a phase 1: one level set per "
                                      "phase (phase_level_sets_at)")  # fmt: skip
        return self.phase_level_sets_at(points, reach)[1]

    def phase_level_sets_at(self, points, reach: float = np.inf) -> dict[int, np.ndarray]:
        """The functions of all the phases at the points: the visible parts of the leaves
        of each phase (the uncovered points: of the background phase, or of none), see
        :func:`_visible_levels`, within ``[-reach, reach]`` (default: the largest radius)."""
        n_phases = max(int(self.cell_phases.max(initial=0)), 1) + 1
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        rmax = float(self.radii.max(initial=0.0)) * max(1.0, self.aspect)
        reach = rmax if not np.isfinite(reach) else float(reach)
        out = np.full((len(pts), n_phases), reach)
        background = -1 if self.background is None else int(self.background)
        if not self.n_leaves:
            if background >= 0:
                out[:, background] = -reach
            return {k: out[:, k] for k in self.implicit_phases()}
        tree = self._centres_tree()
        axes = self.axes if self.axes is not None else np.zeros((self.n_leaves, 3))
        aspect = self.aspect if self.axes is not None else 1.0
        phases = self.leaf_phases.astype(np.int64)
        chunk = 4096  # the pairs of points and leaves in memory
        for s in range(0, len(pts), chunk):
            p = self._local(pts[s : s + chunk])
            lists = tree.query_ball_point(p, rmax + reach, workers=settings.workers())
            counts = np.fromiter((len(c) for c in lists), dtype=np.int64, count=len(lists))
            leaves = np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int64,
                                 count=int(counts.sum()))  # fmt: skip
            owner = np.repeat(np.arange(len(p)), counts)
            order = np.lexsort((leaves, owner))  # each point's leaves in order
            leaves, owner = leaves[order], owner[order]
            offsets = p[owner] - tree.data[leaves]
            if self.domain.periodic:
                offsets -= self.domain.lengths * np.round(offsets / self.domain.lengths)
            ptr = np.concatenate([[0], np.cumsum(counts)])
            _visible_levels(np.ascontiguousarray(offsets), ptr, leaves, self.radii, axes, aspect,
                            phases, background, reach, out[s : s + chunk])  # fmt: skip
        return {k: np.ascontiguousarray(out[:, k]) for k in self.implicit_phases()}

    def summary(self) -> dict:
        return {"n_leaves": self.n_leaves}

    def _geometry(self) -> dict:
        geo = {
            "centers": self.centers.tolist(),
            "radii": self.radii.tolist(),
            "leaf_phases": self.leaf_phases.tolist(),
        }
        if self.orientations is not None:
            geo["orientations"] = self.orientations.tolist()
        if self.axes is not None:
            geo.update(axes=self.axes.tolist(), aspect=self.aspect)
        if self.background is not None:
            geo["background"] = self.background
        return geo

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            geometry["centers"],
            geometry["radii"],
            geometry["leaf_phases"],
            geometry.get("orientations"),
            axes=geometry.get("axes"),
            aspect=geometry.get("aspect", 1.0),
            background=geometry.get("background"),
            **kwargs,
        )


def _ball(s: np.ndarray) -> np.ndarray:
    """The covariogram of the unit ball over its volume at the distance ``s``."""
    return np.where(s < 2.0, 1.0 - 0.75 * s + s**3 / 16.0, 0.0)


def _fibonacci(n: int) -> np.ndarray:
    """``n`` nearly uniform directions of the half sphere (a Fibonacci lattice)."""
    k = np.arange(n) + 0.5
    z = k / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * k
    s = np.sqrt(1.0 - z * z)
    return np.column_stack([s * np.cos(phi), s * np.sin(phi), z])


@register
@anisotropic
@dataclass(frozen=True)
class DeadLeavesModel(Model):
    """Dead leaves of spheres or spheroids, of one or several families: each point belongs
    to the last leaf covering it (or to a background, with a finite coverage)."""

    name = "dead_leaves"
    phases: ClassVar[dict[int, str]] = {0: "leaves of phase 0", 1: "leaves of phase 1"}
    count_label = "leaves"

    @property
    def has_level_set(self) -> bool:
        """Leaves of the phase 1 among those of the phase 0: one level set."""
        return sorted(self.active_phases()) == [0, 1]

    radius: Distribution | None = param(
        None,
        doc="Radius of the leaves (spheres; the equatorial semi-axis of spheroids)",
        kind="distribution",
        unit="length",
    )
    phase_fractions: tuple[float, ...] = param(
        (0.5, 0.5),
        doc="Fractions of the space of the leaves of phases 0, 1, ... (the probabilities of "
        "the leaves when they share their sizes)",
        kind="floats",
    )
    phase_radius: tuple[Distribution, ...] | None = param(
        None,
        doc="Radius of the leaves of each phase (families of leaves of their own sizes), "
        "instead of radius",
        kind="distributions",
        unit="length",
    )
    shape: str = param("sphere", doc="Shape of the leaves: sphere or spheroid", kind="str")
    aspect_ratio: float = param(
        1.0, doc="Polar over equatorial semi-axis of the spheroids (> 1 prolate, < 1 oblate)"
    )
    orientation: OrientationDistribution = param(
        Isotropic(),
        doc="Distribution of the axes of the spheroids (isotropic, aligned, planar)",
        kind="orientation",
    )
    coverage: float | None = param(
        None,
        doc="Mean number of leaves falling on a point: a finite fall over the background "
        "(default: complete coverage)",
    )
    background: int = param(
        0, doc="Phase of the points that no leaf covers (finite coverage)", kind="int"
    )
    crystal_orientations: bool = param(
        False, doc="Draw a uniformly random crystal orientation per leaf", kind="bool"
    )
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        p = np.asarray(self.phase_fractions, dtype=float)
        if p.ndim != 1 or len(p) < 1 or np.any(p < 0) or abs(p.sum() - 1.0) > 1e-9:
            raise ValueError("phase_fractions: non-negative probabilities summing to 1")
        if len(p) > 255:
            raise ValueError("phase_fractions: at most 255 phases")
        if (self.radius is None) == (self.phase_radius is None):
            raise ValueError("give the radius of the leaves, or that of each phase (phase_radius)")
        if self.phase_radius is not None and len(self.phase_radius) != len(p):
            raise ValueError("phase_radius: one distribution per phase of phase_fractions")
        if self.shape not in SHAPES:
            raise ValueError(f"shape: {' or '.join(SHAPES)}, got {self.shape!r}")
        check_positive(self.aspect_ratio, "aspect_ratio")
        if self.shape == "sphere" and self.aspect_ratio != 1.0:
            raise ValueError("aspect_ratio: the leaves are spheres (give shape: spheroid)")
        if self.coverage is not None:
            check_positive(self.coverage, "coverage")
        if not 0 <= self.background <= 255:
            raise ValueError("background: a phase between 0 and 255")

    @property
    def aspect(self) -> float:
        """The polar over the equatorial semi-axis of the leaves (1: spheres)."""
        return self.aspect_ratio if self.shape == "spheroid" else 1.0

    def _laws(self) -> list[Distribution]:
        n = len(self.phase_fractions)
        return list(self.phase_radius) if self.phase_radius is not None else [self.radius] * n

    def families(self) -> list[tuple[int, Distribution, float, float]]:
        """The families of leaves: their phase, the law of their radii, the fraction of
        space ``q`` of their leaves and their proportion ``p`` among the leaves."""
        q = np.asarray(self.phase_fractions, dtype=float)
        laws = self._laws()
        volumes = np.array([self._volume(law) for law in laws])
        w = np.where(q > 0, q / volumes, 0.0)
        p = w / w.sum()
        return [(i, laws[i], float(q[i]), float(p[i])) for i in range(len(q)) if q[i] > 0]

    def _volume(self, law: Distribution) -> float:
        return 4.0 / 3.0 * math.pi * self.aspect * law.moment(3)

    def mean_leaf_volume(self) -> float:
        """The mean volume of a falling leaf, ``1 / Σ q_i / E[V_i]``."""
        return 1.0 / sum(q / self._volume(law) for _, law, q, _ in self.families())

    def active_phases(self) -> dict[int, str]:
        out = {i: f"leaves of phase {i}" for i, p in enumerate(self.phase_fractions) if p > 0}
        if self.coverage is not None:
            out.setdefault(self.background, f"background (phase {self.background})")
        return out

    def _draw(self, rng, window: Domain, k: int):
        """``k`` leaves: centres, radii and phases (families of their own sizes)."""
        families = self.families()
        centers = window.lower + rng.random((k, 3)) * window.lengths
        if self.phase_radius is None:
            return centers, self.radius.sample(rng, k), None
        fam = rng.choice(len(families), size=k, p=np.array([f[3] for f in families]))
        radii = np.empty(k)
        for f, (_, law, _, _) in enumerate(families):
            sel = fam == f
            radii[sel] = law.sample(rng, int(sel.sum()))
        return centers, radii, np.array([f[0] for f in families], dtype=np.uint8)[fam]

    def generate(self, domain: Domain, seed: SeedLike = None) -> DeadLeaves:
        self._check_domain(domain)
        rng, seed_info = make_rng(seed)
        periodic = domain.periodic
        aspect = self.aspect
        laws = [law for _, law, _, _ in self.families()]
        reach = max(law.upper() for law in laws) * max(1.0, aspect)
        window = domain if periodic else domain.dilated(reach)
        mean_volume = self.mean_leaf_volume()
        spheroid = aspect != 1.0
        centers, radii, phases, axes = [], [], [], []
        if self.coverage is not None:  # a Poisson number of leaves over the window
            n = int(rng.poisson(self.coverage * window.volume / mean_volume))
            c, r, ph = self._draw(rng, window, n)
            centers, radii, phases = [c], [r], [ph]
            if spheroid:
                axes.append(self.orientation.sample(rng, n))
        else:
            # leaves until the centres of a check grid (a third of the smallest median
            # semi-axis) are covered, then a third more
            step = min(law.median() for law in laws) * min(1.0, aspect) / 3.0
            shape = tuple(int(np.clip(math.ceil(s / step), 8, 128)) for s in domain.lengths)
            check = Grid(domain, shape)
            cover = np.full(shape, -1, dtype=np.int32)
            batch = max(64, int(0.5 * window.volume / mean_volume))
            n, left = 0, cover.size
            while left:
                c, r, ph = self._draw(rng, window, batch)
                u = self.orientation.sample(rng, batch) if spheroid else None
                cover, left = _cover(c, r, u, aspect, check, periodic, cover, n)
                centers.append(c)
                radii.append(r)
                phases.append(ph)
                axes.append(u)
                n += batch
                if n > 10**8:
                    raise RuntimeError("dead_leaves: the leaves do not cover the domain")
            extra = n // 3 + 1
            c, r, ph = self._draw(rng, window, extra)
            centers.append(c)
            radii.append(r)
            phases.append(ph)
            axes.append(self.orientation.sample(rng, extra) if spheroid else None)
            n += extra
        n = sum(len(r) for r in radii)
        if self.phase_radius is None:
            p = np.asarray(self.phase_fractions, dtype=float)
            leaf_phases = rng.choice(len(p), size=n, p=p / p.sum()).astype(np.uint8)
        else:
            leaf_phases = np.concatenate(phases) if n else np.zeros(0, dtype=np.uint8)
        background = self.background if self.coverage is not None else None
        n_cells = n + (background is not None)
        orientations = random_orientations(rng, n_cells) if self.crystal_orientations else None
        return DeadLeaves(
            domain,
            np.concatenate(centers) if n else np.zeros((0, 3)),
            np.concatenate(radii) if n else np.zeros(0),
            leaf_phases,
            orientations,
            axes=np.concatenate(axes) if spheroid and n else None,
            aspect=aspect,
            background=background,
            model=self,
            seed=seed_info,
        )

    # -- theory --------------------------------------------------------------------------

    def _cosines(self, direction=None) -> tuple[np.ndarray, np.ndarray]:
        """Cosines of the angle between the axes of the leaves and the direction of ``h``,
        with their weights: uniform over the directions without ``direction`` (and for
        isotropic axes), that of the aligned axes, or those of 128 drawn axes."""
        if self.aspect == 1.0:
            return np.zeros(1), np.ones(1)
        if direction is None or isinstance(self.orientation, Isotropic):
            x, w = np.polynomial.legendre.leggauss(48)
            return 0.5 * (x + 1.0), 0.5 * w
        u = np.asarray(direction, dtype=float)
        u = u / np.linalg.norm(u)
        if isinstance(self.orientation, Aligned):
            return np.array([abs(float(np.dot(u, self.orientation.direction)))]), np.ones(1)
        axes = self.orientation.sample(np.random.default_rng(0), 128)
        return np.abs(axes @ u), np.full(128, 1.0 / 128)

    def normalized_covariogram(self, h, law: Distribution | None = None, direction=None):
        """``K(h) / K(0)``: the mean geometric covariogram of the leaves of a law of radii
        (that of the first family by default) over their mean volume, for the distances
        ``h`` along ``direction`` (or averaged over the directions)."""
        law = law if law is not None else self.families()[0][1]
        h = np.asarray(h, dtype=float)
        x, wx = law.quadrature(64)
        t, wt = self._cosines(direction)
        hm = h[..., None] * np.sqrt(1.0 - t * t + (t / self.aspect) ** 2)
        out = np.zeros(h.shape)
        for xi, wi in zip(x, wx, strict=True):  # the volume weights the sizes
            out += wi * xi**3 * (_ball(hm / xi) @ wt)
        return out / np.sum(wx * x**3)

    def _rates(self, h, direction=None) -> tuple[np.ndarray, np.ndarray]:
        """The fractions of space ``q`` of the families and their normalized covariograms
        ``κ`` at the distances ``h`` (last axis: the families)."""
        families = self.families()
        q = np.array([f[2] for f in families])
        kappa = np.stack([self.normalized_covariogram(h, law, direction)
                          for _, law, _, _ in families], axis=-1)  # fmt: skip
        return q, kappa

    def bivariate(self, h, direction=None) -> np.ndarray:
        """The joint probabilities of the families (and of the background, last, with a
        finite coverage) at ``x`` and ``x + h``: shape ``(..., F, F)`` or ``(..., F + 1,
        F + 1)``."""
        q, kappa = self._rates(h, direction)
        f = len(q)
        if self.coverage is None:  # the rates are those of a unit coverage
            lam = np.sum(q * (2.0 - kappa), axis=-1)[..., None]
            same, cross = q * kappa / lam, q * (1.0 - kappa) / lam
            out = np.einsum("...i,j->...ij", cross, q)
            out = out + np.swapaxes(out, -1, -2)
            out[..., range(f), range(f)] += same
            return out
        c = self.coverage
        a, b = c * q * kappa, c * q * (1.0 - kappa)
        lam = np.sum(a + 2.0 * b, axis=-1)[..., None]
        gap = lam - c  # >= 0: lam = c (h = 0) to 2c (independent points)
        i1 = -np.expm1(-lam) / lam
        safe = np.where(gap > 1e-12 * c, gap, 1.0)
        i2 = np.where(gap > 1e-12 * c, math.exp(-c) * -np.expm1(-gap) / safe, math.exp(-c))
        out = np.zeros((*np.shape(lam)[:-1], f + 1, f + 1))
        cross = np.einsum("...i,j->...ij", b, q) * (i1 - i2)[..., None]
        out[..., :f, :f] = cross + np.swapaxes(cross, -1, -2)
        out[..., range(f), range(f)] += a * -np.expm1(-lam) / lam
        out[..., :f, f] = b * i2
        out[..., f, :f] = b * i2
        out[..., f, f] = np.exp(-lam[..., 0])
        return out

    def _cells_of(self, phase: int) -> list[int]:
        """The indices of the families (and of the background) of a phase in
        :meth:`bivariate`."""
        families = self.families()
        idx = [k for k, f in enumerate(families) if f[0] == phase]
        if self.coverage is not None and self.background == phase:
            idx.append(len(families))
        return idx

    def covariance(self, h, phase: int = 1, direction=None) -> np.ndarray:
        """``P(x and x + h in the phase)`` (averaged over the directions without
        ``direction``)."""
        idx = self._cells_of(phase)
        p = self.bivariate(h, direction)
        return np.sum(p[..., idx, :][..., :, idx], axis=(-2, -1)) if idx else np.zeros(np.shape(h))

    def same_leaf_probability(self, h, direction=None) -> np.ndarray:
        """Probability that two points at the distance ``h`` see the same leaf:
        ``Σ a_i / λ`` (times ``1 - e^{-λ}`` with a finite coverage)."""
        q, kappa = self._rates(h, direction)
        lam = np.sum(q * (2.0 - kappa), axis=-1)
        same = np.sum(q * kappa, axis=-1) / lam
        if self.coverage is None:
            return same
        return same * -np.expm1(-self.coverage * lam)

    def phase_fraction(self, phase: int) -> float:
        """The fraction of space of a phase."""
        q = sum(f[2] for f in self.families() if f[0] == phase)
        if self.coverage is None:
            return float(q)
        bare = math.exp(-self.coverage)
        return float(q * (1.0 - bare) + (bare if self.background == phase else 0.0))

    def _isotropic(self) -> bool:
        return self.aspect == 1.0 or isinstance(self.orientation, Isotropic)

    def integral_range(self, phase: int = 1) -> float:
        """``∫ (C(h) - p^2) dh / (p (1 - p))`` over the space, for the phase of fraction
        ``p`` (Gauss-Legendre in the distance, over 24 directions when the leaves are not
        isotropic)."""
        p = self.phase_fraction(phase)
        if not 0.0 < p < 1.0:
            return 0.0
        reach = max(law.upper() for _, law, _, _ in self.families()) * max(1.0, self.aspect)
        x, w = np.polynomial.legendre.leggauss(256)
        h, w = reach * (x + 1.0), reach * w  # up to the largest diameter
        directions = [None] if self._isotropic() else list(_fibonacci(24))
        total = 0.0
        for u in directions:
            centred = self.covariance(h, phase, u) - p * p
            total += float(np.sum(centred * 4.0 * math.pi * h**2 * w))
        return total / len(directions) / (p * (1.0 - p))

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        out = {"mean_leaf_volume": self.mean_leaf_volume()}
        n = max(len(self.phase_fractions), self.background + 1 if self.coverage else 0)
        for i in range(n):
            out[f"volume_fraction_{i}"] = self.phase_fraction(i)
        if n > 1:
            out["volume_fraction"] = out["volume_fraction_1"]
        if self.coverage is not None:
            out["background_fraction"] = math.exp(-self.coverage)
        phase = 1 if 0.0 < out.get("volume_fraction_1", 0.0) < 1.0 else self.families()[0][0]
        out["integral_range"] = self.integral_range(phase)
        return out
