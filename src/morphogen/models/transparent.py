"""Transparent dead leaves (Jeulin 1997): a random function of falling transparent leaves,
and its excursion above a threshold.

Each leaf (a sphere) carries a value ``v`` (a grey level, uniform on [0, 1] or Bernoulli);
when it falls on a point, the value there becomes ``α v + (1 - α) Z``: the leaf lets the
fraction ``1 - α`` of what it covers show through (``α = 1``: opaque leaves, the dead
leaves). After infinitely many leaves, ``Z(x) = Σ_n α (1 - α)^{n - 1} v_n``, ``v_1`` the
value of the last leaf fallen on ``x``. Going back in time, as in the perfect simulation
of the dead leaves (:mod:`morphogen.models.deadleaves`), the leaves are drawn one by one,
each under the earlier ones, until the weight ``(1 - α)^n`` left to each centre of a
check grid is below the tolerance, then a third more; what is left of the weight of a
point takes the mean value of the leaves.

Theory. ``E[Z] = E[v]`` and ``Var Z = σ^2 α / (2 - α)`` (``σ^2 = Var v``). The leaves hitting
``x`` or ``x + h`` cover both with the probability ``r(h) = K(h) / (2K(0) - K(h))`` (``K``:
the mean geometric covariogram of the leaves), so that each of them multiplies
``E[(1 - α)^{a + b}]``, ``a`` and ``b`` the leaves fallen on each point above it, by ``m = r (1
- α)^2 + (1 - r)(1 - α)``; the shared leaves give

    Cov(Z(x), Z(x + h)) = σ^2 α^2 r / (1 - m) = σ^2 α r / (1 + r (1 - α)),

the correlation ``r (2 - α) / (1 + r (1 - α))`` (``r`` for opaque leaves). Phase 1 is
``{Z > t}``; its fraction ``P(Z > t)`` follows from the law of the series, here from 10^6
draws of its terms down to a weight of 10^-4 (the rest by its mean): the threshold of a
volume fraction to about 10^-3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

import numpy as np
from scipy.spatial import cKDTree

from morphogen import settings
from morphogen._utils import check_fraction, check_positive, exactly_one
from morphogen.distributions import Distribution
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.kernels.leaves import transparent_cover
from morphogen.models.affine import ANISOTROPY, Anisotropy, anisotropic
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.seeding import SeedLike, make_rng

#: Laws of the values of the leaves.
VALUES = ("uniform", "bernoulli")


def _field(centers, radii, values, alpha, grid: Grid, periodic: bool, mean: float, z=None, w=None,
           tol: float = 0.0):  # fmt: skip
    """The field of the leaves at the voxel centres of a grid (and the weights left)."""
    if z is None:
        z = np.zeros(grid.shape)
        w = np.ones(grid.shape)
    left = transparent_cover(
        np.ascontiguousarray(centers, dtype=float),
        np.ascontiguousarray(radii, dtype=float),
        np.ascontiguousarray(values, dtype=float),
        float(alpha),
        np.asarray(grid.domain.lower, dtype=float),
        np.asarray(grid.spacing, dtype=float),
        np.asarray(grid.shape, dtype=np.int64),
        periodic,
        z,
        w,
        float(tol),
    )
    return z, w, left


@register_kind
class TransparentDeadLeaves(Microstructure):
    """Transparent leaves in their order of fall backwards (the first on top), the
    random function ``Z`` they make, and its excursion ``Z > threshold`` (phase 1)."""

    kind = "transparent_dead_leaves"

    def __init__(self, domain: Domain, centers, radii, values, *, alpha: float,
                 threshold: float, mean_value: float, **kwargs) -> None:  # fmt: skip
        super().__init__(domain, **kwargs)
        self.centers = np.ascontiguousarray(centers, dtype=float).reshape(-1, 3)
        self.radii = np.asarray(radii, dtype=float).reshape(-1)
        self.values = np.asarray(values, dtype=float).reshape(-1)
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.mean_value = float(mean_value)
        self._tree = None

    @property
    def n_leaves(self) -> int:
        return len(self.radii)

    def field(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """The random function ``Z`` at the voxel centres (float32)."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        z, w, _ = _field(self.centers, self.radii, self.values, self.alpha, grid,
                         self.domain.periodic, self.mean_value)  # fmt: skip
        z += w * self.mean_value  # what the leaves leave of the weight: the mean value
        return VoxelImage.from_grid(z.astype(np.float32), grid, name="field",
                                    metadata=self.image_metadata())  # fmt: skip

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if ss == 1:
            data = (self.field(grid=grid).data > self.threshold).astype(np.uint8)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        ph = (self.field(grid=fine).data > self.threshold).astype(np.float32)
        nx, ny, nz = grid.shape
        data = ph.reshape(nx, ss, ny, ss, nz, ss).mean(axis=(1, 3, 5), dtype=np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def _local(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float) - self.domain.lower
        if self.domain.periodic:
            p = np.mod(p, self.domain.lengths)
            p[p >= self.domain.lengths] = 0.0
        return p

    def values_at(self, points) -> np.ndarray:
        """The random function at points: the leaves covering each, in their order."""
        pts = self._local(np.asarray(points, dtype=float).reshape(-1, 3))
        out = np.full(len(pts), self.mean_value)
        if not self.n_leaves:
            return out
        if self._tree is None:
            if self.domain.periodic:
                self._tree = cKDTree(self._local(self.centers), boxsize=self.domain.lengths)
            else:
                self._tree = cKDTree(self.centers - self.domain.lower)
        found = self._tree.query_ball_point(pts, float(self.radii.max()),
                                            workers=settings.workers())  # fmt: skip
        keep = 1.0 - self.alpha
        for i, cand in enumerate(found):
            if not cand:
                continue
            cand = np.sort(np.asarray(cand))
            d = self._tree.data[cand] - pts[i]
            if self.domain.periodic:
                d -= self.domain.lengths * np.round(d / self.domain.lengths)
            inside = cand[np.einsum("ij,ij->i", d, d) <= self.radii[cand] ** 2]
            weights = self.alpha * keep ** np.arange(len(inside))
            rest = keep ** len(inside) * self.mean_value
            out[i] = float(np.sum(weights * self.values[inside])) + rest
        return out

    def contains(self, points) -> np.ndarray:
        return self.values_at(points) > self.threshold

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """The implicit function of the excursion ``Z > threshold`` (negative inside). The
        function ``Z`` is constant between the spheres of the leaves: the boundary of the
        excursion is made of the parts of these spheres across which ``Z`` passes the
        threshold. At a point, the leaf whose sphere is such a part nearby is one that,
        added to the leaves that cover the point or taken from them, changes its side:
        the function is the distance to the nearest of these spheres, with the sign of the
        side of the point (clipped to a band of four voxels). It vanishes exactly on the
        boundary; away from it, it is a distance to a sphere near that boundary."""
        from morphogen.models.implicit import band_of, lattice_boxes, ordered_cover

        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        band = band_of(grid)
        lengths = self.domain.lengths if self.domain.periodic else None
        if self.alpha == 1.0:  # opaque leaves: the dead leaves, of the side of their value

            def objects():
                for i in range(self.n_leaves):
                    reach = self.radii[i] + band
                    for slices, (dx, dy, dz) in lattice_boxes(
                        self.centers[i], reach, grid, lengths
                    ):
                        d = (np.sqrt(dx * dx + dy * dy + dz * dz) - self.radii[i]).astype(
                            np.float32
                        )
                        yield slices, d, [d if self.values[i] > self.threshold else None]

            parts, top = ordered_cover(grid.shape, band, objects(), 1)
            data = parts[0]
            if self.mean_value > self.threshold:  # what no leaf covers
                data = np.minimum(data, -top)
            return VoxelImage.from_grid(
                data, grid, name="level_set", metadata=self.image_metadata()
            )
        z = np.asarray(self.field(grid=grid).data)
        inside = z > self.threshold
        z = z.astype(np.float64)
        keep = 1.0 - self.alpha
        head = np.zeros(grid.shape)  # the sum of the leaves above the current one
        weight = np.ones(grid.shape)  # what they let through
        dist = np.full(grid.shape, band, dtype=np.float32)
        for i in range(self.n_leaves):
            radius, value = self.radii[i], self.alpha * self.values[i]
            for slices, (dx, dy, dz) in lattice_boxes(
                self.centers[i], radius + band, grid, lengths
            ):
                d = np.sqrt(dx * dx + dy * dy + dz * dz) - radius
                covers = d <= 0
                a, w, zz = head[slices], weight[slices], z[slices]
                # the field without the leaf where it covers, with it elsewhere
                other = np.where(covers, a + (zz - a - w * value) / keep,
                                 a + w * value + keep * (zz - a))  # fmt: skip
                flips = (other > self.threshold) != inside[slices]
                view = dist[slices]
                np.minimum(view, np.where(flips, np.abs(d), band), out=view)
                a += np.where(covers, w * value, 0.0)
                w *= np.where(covers, keep, 1.0)
        data = np.where(inside, -dist, dist).astype(np.float32)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def summary(self) -> dict:
        return {"n_leaves": self.n_leaves}

    def _geometry(self) -> dict:
        return {
            "centers": self.centers.tolist(),
            "radii": self.radii.tolist(),
            "values": self.values.tolist(),
            "alpha": self.alpha,
            "threshold": self.threshold,
            "mean_value": self.mean_value,
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(domain, geometry["centers"], geometry["radii"], geometry["values"],
                   alpha=geometry["alpha"], threshold=geometry["threshold"],
                   mean_value=geometry["mean_value"], **kwargs)  # fmt: skip


@lru_cache(maxsize=64)
def _law(alpha: float, values: str, p: float) -> np.ndarray:
    """10^6 draws of ``Z`` (sorted): its terms down to a weight of 10^-4, the rest by its
    mean."""
    keep = 1.0 - alpha
    terms = 1 if keep == 0.0 else math.ceil(math.log(1e-4) / math.log(keep))
    mean = 0.5 if values == "uniform" else p
    rng = np.random.default_rng(1997)
    out = np.empty(10**6)
    weights = alpha * keep ** np.arange(terms)
    for start in range(0, len(out), 50_000):
        n = min(50_000, len(out) - start)
        if values == "uniform":
            v = rng.random((n, terms))
        else:
            v = (rng.random((n, terms)) < p).astype(float)
        out[start : start + n] = v @ weights + keep**terms * mean
    return np.sort(out)


@register
@anisotropic
@dataclass(frozen=True)
class TransparentDeadLeavesModel(Model):
    """Transparent dead leaves of spheres (Jeulin 1997): the excursion of the random
    function of leaves letting ``1 - α`` of what they cover show through."""

    name = "transparent_dead_leaves"
    phases: ClassVar[dict[int, str]] = {0: "below the threshold", 1: "above the threshold"}
    count_label = "leaves"

    radius: Distribution = param(doc="Radius of the leaves", kind="distribution", unit="length")
    transparency: float = param(
        0.5,
        doc="Weight alpha of a leaf in the value it covers (1: opaque leaves, the dead "
        "leaves; small: very transparent leaves)",
        kind="fraction",
    )
    values: str = param("uniform", doc="Values of the leaves: uniform (on [0, 1]) or bernoulli",
                        kind="str")  # fmt: skip
    value_probability: float = param(
        0.5, doc="Probability of the value 1 of bernoulli leaves", kind="fraction"
    )
    threshold: float | None = param(None, doc="Phase 1: the field above it")
    volume_fraction: float | None = param(
        None, doc="Fraction of phase 1 (sets the threshold)", kind="fraction"
    )
    tolerance: float = param(
        1e-3, doc="Weight left to a point by the leaves drawn (then its mean value)"
    )
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        check_fraction(self.transparency, "transparency", open_interval=False)
        if self.transparency == 0.0:
            raise ValueError("transparency: must lie in (0, 1]")
        if self.values not in VALUES:
            raise ValueError(f"values: {' or '.join(VALUES)}, got {self.values!r}")
        check_fraction(self.value_probability, "value_probability")
        exactly_one("threshold", self.threshold, "volume_fraction", self.volume_fraction)
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        check_positive(self.tolerance, "tolerance")
        if self.tolerance >= 0.5:
            raise ValueError("tolerance: below 0.5")

    @property
    def mean_value(self) -> float:
        return 0.5 if self.values == "uniform" else float(self.value_probability)

    @property
    def value_variance(self) -> float:
        p = self.value_probability
        return 1.0 / 12.0 if self.values == "uniform" else float(p * (1.0 - p))

    def _sample(self) -> np.ndarray:
        return _law(float(self.transparency), self.values, float(self.value_probability))

    def field_threshold(self) -> float:
        """The threshold of phase 1: given, or the quantile of the law of ``Z`` that leaves
        the volume fraction above it."""
        if self.threshold is not None:
            return float(self.threshold)
        return float(np.quantile(self._sample(), 1.0 - self.volume_fraction))

    def excursion_fraction(self) -> float:
        """``P(Z > threshold)``, from the law of ``Z``."""
        z = self._sample()
        return float(1.0 - np.searchsorted(z, self.field_threshold(), side="right") / len(z))

    def same_leaf_probability(self, h) -> np.ndarray:
        """``r(h) = K(h) / (2 K(0) - K(h))`` for spheres of the law of the radii."""
        h = np.asarray(h, dtype=float)
        x, w = self.radius.quadrature(64)
        u = h[..., None] / x
        k = np.sum(w * x**3 * np.where(u < 2.0, 1.0 - 0.75 * u + u**3 / 16.0, 0.0), axis=-1)
        k0 = np.sum(w * x**3)
        return k / (2.0 * k0 - k)

    def field_covariance(self, h) -> np.ndarray:
        """``Cov(Z(x), Z(x + h)) = σ^2 α r / (1 + r (1 - α))``."""
        r = self.same_leaf_probability(h)
        a = self.transparency
        return self.value_variance * a * r / (1.0 + r * (1.0 - a))

    def _volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius.moment(3)

    def generate(self, domain: Domain, seed: SeedLike = None) -> TransparentDeadLeaves:
        self._check_domain(domain)
        rng, seed_info = make_rng(seed)
        periodic = domain.periodic
        window = domain if periodic else domain.dilated(self.radius.upper())
        # leaves until the weight left to the centres of a check grid (a third of the
        # median radius) falls below the tolerance, then a third more
        step = self.radius.median() / 3.0
        shape = tuple(int(np.clip(math.ceil(s / step), 8, 128)) for s in domain.lengths)
        check = Grid(domain, shape)
        z, w = np.zeros(shape), np.ones(shape)
        batch = max(64, int(0.5 * window.volume / self._volume()))
        centers, radii, values, n, left = [], [], [], 0, z.size
        while left:
            c, r, v = self._draw(rng, window, batch)
            z, w, left = _field(c, r, v, self.transparency, check, periodic, self.mean_value,
                                z, w, self.tolerance)  # fmt: skip
            centers.append(c)
            radii.append(r)
            values.append(v)
            n += batch
            if n > 10**8:
                raise RuntimeError("transparent_dead_leaves: the leaves do not cover the domain")
        c, r, v = self._draw(rng, window, n // 3 + 1)
        centers.append(c)
        radii.append(r)
        values.append(v)
        return TransparentDeadLeaves(
            domain,
            np.concatenate(centers),
            np.concatenate(radii),
            np.concatenate(values),
            alpha=self.transparency,
            threshold=self.field_threshold(),
            mean_value=self.mean_value,
            model=self,
            seed=seed_info,
        )

    def _draw(self, rng, window: Domain, k: int):
        centers = window.lower + rng.random((k, 3)) * window.lengths
        radii = self.radius.sample(rng, k)
        if self.values == "uniform":
            values = rng.random(k)
        else:
            values = (rng.random(k) < self.value_probability).astype(float)
        return centers, radii, values

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        a = self.transparency
        x, _ = self.radius.quadrature(64)
        h = np.linspace(0.0, 2.0 * float(x.max()), 4001)
        rho = self.field_covariance(h) / (self.value_variance * a / (2.0 - a))
        integrand = rho * 4.0 * math.pi * h**2
        area = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(h)))
        return {
            "field_mean": self.mean_value,
            "field_variance": self.value_variance * a / (2.0 - a),
            "field_integral_range": area,
            "threshold": self.field_threshold(),
            "volume_fraction": self.excursion_fraction(),
            "mean_leaf_volume": self._volume(),
        }
