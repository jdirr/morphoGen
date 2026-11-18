"""Foams as level sets of distance fields (Sonon, François & Massart 2012, 2015; Kilingar,
Sonon & Massart 2019; Degdeg & Dirrenberger 2026).

Seeds: non-overlapping ellipsoids aligned with the axes, of short semi-axis ``a`` (a
distribution) and shape factors ``b/a`` and ``c/a`` (``aspect``): a dense packing of
polydisperse spheres (Lubachevsky–Stillinger) or their random sequential addition,
mapped by the stretch of the y and z axes, so that the packing of the ellipsoids is the
affine image of that of spheres (as the packing of Degdeg & Dirrenberger, whose aligned
ellipsoids fall into a container). Each seed defines the normalized algebraic distance

    D_i(x) = sum_m (x_m - x_im)^2 / r_im^2 - 1

(-1 at its centre, 0 on its surface), and ``D_N1 <= D_N2 <= D_N3 <= D_N4`` are the four
smallest at ``x``. The open foam is the region ``Phi <= 0`` of

    Phi = O_p - t O_s - K_c O_k
    O_p = (D_N3 + D_N2)/2 - D_N1             zero on the edges of the tessellation
    O_s = sqrt(c1 xi^4 + c2 xi^2 + 1)          thickening towards the nodes
    O_k = min(0, (delta^2 - w^2) / (2 w))      delta = (D_N3 - D_N2)/2, w = t O_s

``t`` the thickness (dimensionless; or solved for a ``volume_fraction``), ``K_c`` the
curvature of the cross-sections of the struts (``0 < K_c < 1``: the concave Plateau
borders of drained foams; ``K_c < 0``: convex), ``xi`` the position along the strut (0
at mid-span, 1/2 at the nodes): ``D_N4 - D_N3`` vanishes at the nodes and is largest at
mid-span, and ``xi = (1 - (D_N4 - D_N3)/m)/2`` with ``m`` its largest value near the
axis of the strut (the strut of the three nearest seeds), found on a fine grid at
generation. Closed cells add walls where ``D_N2 = D_N1``:
``Phi = min(Phi, (D_N2 - D_N1)/2 - t_w)``.

:func:`calibrate_seeds` scales the seeds (and, if asked, changes their shape factors)
until the mean radii of the sections of the pores by the planes of the grid match
measured ones, the cells of a packing being larger than its seeds.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.spatial import cKDTree

from morphogen._utils import check_fraction, check_positive
from morphogen.distributions import Constant, Distribution, distribution_from_config
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.kernels.rsa import random_sequential_addition
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.models.hardcore import grown_packing
from morphogen.seeding import SeedLike, make_rng

_CHUNK = 1 << 18
_NEAR = 1.0  # points of a strut: O_p below this (normalized units)


def nearest_distances(points, centres, radii, box, periodic: bool, k: int = 16):
    """The four smallest ``|y - y_i|^2 / a_i^2 - 1`` at ``points`` (y-space: the seeds are
    spheres of radii ``radii``) and the indices of their seeds, from the ``k`` nearest
    centres (more where the four could hide beyond them)."""
    n = len(centres)
    tree = cKDTree(centres, boxsize=box if periodic else None)
    pts = np.mod(points, box) if periodic else points
    a2 = radii**2
    amax2 = float(a2.max())
    dn = np.empty((len(pts), 4))
    nn = np.empty((len(pts), 4), dtype=np.int64)
    todo = np.arange(len(pts))
    kk = min(k, n)
    while todo.size:
        d, idx = tree.query(pts[todo], k=kk, workers=-1)
        ds = d * d / a2[idx] - 1.0
        order = np.argsort(ds, axis=1)[:, :4]
        rows = np.arange(len(todo))[:, None]
        dn[todo] = ds[rows, order]
        nn[todo] = idx[rows, order]
        if kk >= n:
            break
        bound = d[:, -1] ** 2 / amax2 - 1.0  # no seed beyond the k nearest is closer
        todo = todo[dn[todo, 3] > bound]
        kk = min(2 * kk, n)
    return dn, nn


def strut_keys(nn: np.ndarray, n: int) -> np.ndarray:
    """Key of the strut of the three nearest seeds (their sorted indices)."""
    s = np.sort(nn[:, :3], axis=1)
    return (s[:, 0] * n + s[:, 1]) * n + s[:, 2]


@register_kind
class DistanceFieldFoam(Microstructure):
    """A foam, the region ``Phi <= 0`` of the level-set function of its seeds (see the
    module); the same realization can be evaluated at any resolution and point."""

    kind = "distance_field_foam"

    def __init__(
        self,
        domain,
        centres,
        axes,
        box_lower,
        box_lengths,
        thickness,
        *,
        curvature: float = 0.0,
        c1: float = 0.0,
        c2: float = 0.0,
        cells: str = "open",
        wall_thickness: float = 0.0,
        periodic_box: bool = True,
        struts=None,
        **kwargs,
    ):
        super().__init__(domain, **kwargs)
        self.centres = np.asarray(centres, dtype=float).reshape(-1, 3)
        self.axes = np.asarray(axes, dtype=float).reshape(-1, 3)
        if len(self.centres) < 5:
            raise ValueError("distance_field_foam: five seeds at least")
        self.box_lower = np.asarray(box_lower, dtype=float)
        self.box_lengths = np.asarray(box_lengths, dtype=float)
        self.thickness = float(thickness)
        self.curvature = float(curvature)
        self.c1, self.c2 = float(c1), float(c2)
        self.cells = cells
        self.wall_thickness = float(wall_thickness)
        self.periodic_box = bool(periodic_box)
        # aligned ellipsoids of equal shapes: spheres in the y-space x / (1, b/a, c/a)
        self.scale = self.axes[0] / self.axes[0, 0]
        if not np.allclose(self.axes / self.axes[:, :1], self.scale, rtol=1e-9):
            raise ValueError("distance_field_foam: seeds of the same shape (aspect) only")
        self.struts = None
        if struts is not None:
            keys, top = struts
            self.struts = (np.asarray(keys, dtype=np.int64), np.asarray(top, dtype=float))

    # -- distance fields ----------------------------------------------------------------
    def distances(self, points) -> tuple[np.ndarray, np.ndarray]:
        """``D_N1..D_N4`` and the indices of their seeds at ``points`` (n x 3)."""
        y = (np.asarray(points, dtype=float) - self.box_lower) / self.scale
        cy = (self.centres - self.box_lower) / self.scale
        box = self.box_lengths / self.scale
        dn = np.empty((len(y), 4))
        nn = np.empty((len(y), 4), dtype=np.int64)
        for s in range(0, len(y), _CHUNK):
            dn[s : s + _CHUNK], nn[s : s + _CHUNK] = nearest_distances(
                y[s : s + _CHUNK], np.mod(cy, box), self.axes[:, 0], box, self.periodic_box
            )
        return dn, nn

    def learn_struts(self, dn: np.ndarray, nn: np.ndarray) -> None:
        """Largest ``D_N4 - D_N3`` of each strut, over the points near its axis."""
        op = 0.5 * (dn[:, 2] + dn[:, 1]) - dn[:, 0]
        near = op <= _NEAR
        keys = strut_keys(nn[near], len(self.centres))
        gap = dn[near, 3] - dn[near, 2]
        uniq, inverse = np.unique(keys, return_inverse=True)
        top = np.zeros(len(uniq))
        np.maximum.at(top, inverse, gap)
        self.struts = (uniq, top)

    def _position(self, dn, nn) -> np.ndarray:
        """``xi`` along the struts: 0 at mid-span, 1/2 at the nodes."""
        if self.struts is None or len(self.struts[0]) == 0 or (self.c1 == 0 and self.c2 == 0):
            return np.zeros(len(dn))
        keys, top = self.struts
        k = strut_keys(nn, len(self.centres))
        i = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
        m = np.where(keys[i] == k, top[i], float(np.median(top)))
        ratio = np.divide(dn[:, 3] - dn[:, 2], m, out=np.ones(len(dn)), where=m > 0)
        return 0.5 * (1.0 - np.clip(ratio, 0.0, 1.0))

    def phi(self, dn, nn, thickness: float | None = None) -> np.ndarray:
        """The level-set function (solid where it is not positive)."""
        t = self.thickness if thickness is None else float(thickness)
        xi = self._position(dn, nn)
        o_s = np.sqrt(self.c1 * xi**4 + self.c2 * xi**2 + 1.0)
        o_p = 0.5 * (dn[:, 2] + dn[:, 1]) - dn[:, 0]
        out = o_p - t * o_s
        if t > 0.0 and self.curvature != 0.0:
            w = t * o_s
            delta = 0.5 * (dn[:, 2] - dn[:, 1])
            out -= self.curvature * np.minimum(0.0, (delta**2 - w**2) / (2.0 * w))
        if self.cells == "closed":
            out = np.minimum(out, 0.5 * (dn[:, 1] - dn[:, 0]) - self.wall_thickness)
        return out

    def values(self, points) -> np.ndarray:
        dn, nn = self.distances(points)
        return self.phi(dn, nn)

    # -- microstructure -------------------------------------------------------------------
    def _grid_values(self, grid: Grid) -> np.ndarray:
        pts = np.stack(
            np.meshgrid(*[grid.centers(a) for a in range(3)], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        return self.values(pts).reshape(grid.shape)

    def field(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        data = self._grid_values(grid).astype(np.float32)
        return VoxelImage.from_grid(data, grid, name="phi", metadata=self.image_metadata())

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        if ss == 1:
            data = (self._grid_values(grid) <= 0.0).astype(np.uint8)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        inside = (self._grid_values(fine) <= 0.0).astype(np.float32)
        nx, ny, nz = grid.shape
        data = inside.reshape(nx, ss, ny, ss, nz, ss).mean(axis=(1, 3, 5), dtype=np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """``Phi`` times half the mean short semi-axis: negative in the solid, roughly a
        distance near its surface (``|grad D_i|`` is about ``2 / a`` there)."""
        f = self.field(shape, voxel_size=voxel_size, grid=grid)
        data = f.data * np.float32(0.5 * float(self.axes[:, 0].mean()))
        return f.with_data(data.astype(np.float32), name="level_set")

    def contains(self, points) -> np.ndarray:
        return self.values(np.asarray(points, dtype=float).reshape(-1, 3)) <= 0.0

    #: ``Phi`` grows at most twice as fast as the distance to the nearest seed
    lipschitz = 1.5

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """``Phi`` times half the mean short semi-axis at the points, as :meth:`level_set`."""
        values = self.values(np.asarray(points, dtype=float).reshape(-1, 3))
        return values * 0.5 * float(self.axes[:, 0].mean())

    def summary(self) -> dict:
        return {
            "seeds": len(self.centres),
            "thickness": self.thickness,
            "struts": 0 if self.struts is None else len(self.struts[0]),
        }

    def _geometry(self) -> dict:
        return {
            "centres": self.centres.tolist(),
            "axes": self.axes.tolist(),
            "box_lower": self.box_lower.tolist(),
            "box_lengths": self.box_lengths.tolist(),
            "thickness": self.thickness,
            "curvature": self.curvature,
            "c1": self.c1,
            "c2": self.c2,
            "cells": self.cells,
            "wall_thickness": self.wall_thickness,
            "periodic_box": self.periodic_box,
            "struts": None
            if self.struts is None
            else [self.struts[0].tolist(), self.struts[1].tolist()],
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        g = dict(geometry)
        return cls(
            domain,
            g.pop("centres"),
            g.pop("axes"),
            g.pop("box_lower"),
            g.pop("box_lengths"),
            g.pop("thickness"),
            **g,
            **kwargs,
        )


@register
@dataclass(frozen=True)
class LevelSetFoam(Model):
    """Open- or closed-cell foams as level sets of the distance fields of a packing of
    aligned ellipsoids (Sonon et al. 2012; Kilingar et al. 2019; Degdeg & Dirrenberger
    2026): struts with Plateau-border cross-sections of controlled curvature, thickening
    towards the nodes, cells elongated by the shape of the seeds (see the module)."""

    name = "level_set_foam"
    phases: ClassVar[dict[int, str]] = {0: "pores", 1: "solid"}
    count_label = "seeds"

    seed_radius: Distribution = param(
        Constant(0.1),
        doc="Short semi-axis a of the seed ellipsoids",
        kind="distribution",
        unit="length",
    )
    aspect: tuple[float, ...] = param(
        (1.0, 1.0),
        doc="Shape factors b/a and c/a of the seeds (semi-axes along y and z)",
        kind="floats",
    )
    packing: str = param(
        "dense",
        doc="Packing of the seeds: dense (Lubachevsky-Stillinger) or rsa (random sequential "
        "addition)",
        kind="str",
    )
    packing_fraction: float = param(
        0.6, doc="Volume fraction of the seed ellipsoids", kind="fraction"
    )
    thickness: float | None = param(
        None, doc="Thickness t of the struts (dimensionless; or give volume_fraction)"
    )
    volume_fraction: float | None = param(
        None, doc="Solid volume fraction, reached by solving for t", kind="fraction"
    )
    curvature: float = param(
        0.0, doc="Curvature K_c of the cross-sections: 0 < K_c < 1 concave, K_c < 0 convex"
    )
    c1: float = param(0.0, doc="Node thickening: A(xi) = c1 xi^4 + c2 xi^2 + 1")
    c2: float = param(0.0, doc="Node thickening: A(xi) = c1 xi^4 + c2 xi^2 + 1")
    cells: str = param("open", doc="open (struts) or closed (struts and walls)", kind="str")
    wall_thickness: float = param(
        0.05, doc="Thickness of the walls of closed cells (dimensionless)"
    )
    growth_rate: float = param(0.01, doc="Growth rate of the dense packing", unit="")

    def _validate(self) -> None:
        if len(self.aspect) != 2 or min(self.aspect) <= 0:
            raise ValueError("aspect: two positive shape factors, b/a and c/a")
        if self.packing not in ("dense", "rsa"):
            raise ValueError("packing: dense or rsa")
        check_fraction(self.packing_fraction, "packing_fraction")
        if (self.thickness is None) == (self.volume_fraction is None):
            raise ValueError("give exactly one of thickness and volume_fraction")
        if self.thickness is not None:
            check_positive(self.thickness, "thickness")
        else:
            check_fraction(self.volume_fraction, "volume_fraction")
        if not self.curvature < 1.0:
            raise ValueError("curvature: K_c < 1")
        if self.cells not in ("open", "closed"):
            raise ValueError("cells: open or closed")
        check_positive(self.wall_thickness, "wall_thickness", allow_zero=True)
        check_positive(self.growth_rate, "growth_rate")

    @property
    def scale(self) -> np.ndarray:
        return np.array([1.0, *self.aspect])

    def _radii(self, rng, volume: float) -> np.ndarray:
        """Short semi-axes of the seeds filling ``packing_fraction`` of ``volume``."""
        shape = float(np.prod(self.scale))
        target = self.packing_fraction * volume
        radii = np.empty(0)
        while True:
            m = int(1.2 * target / (4.0 / 3.0 * math.pi * shape * self.seed_radius.moment(3))) + 8
            radii = np.concatenate([radii, self.seed_radius.sample(rng, m)])
            cum = np.cumsum(4.0 / 3.0 * math.pi * shape * radii**3)
            if cum[-1] >= target:
                return radii[: int(np.searchsorted(cum, target)) + 1]

    def seeds(self, domain: Domain, rng) -> tuple[np.ndarray, np.ndarray, Domain]:
        """Centres and semi-axes of the seed ellipsoids, and their periodic box (the
        domain if periodic, enlarged by two long semi-axes otherwise)."""
        reach = 2.0 * float(self.seed_radius.upper()) * float(self.scale.max())
        box = domain if domain.periodic else domain.dilated(reach)
        radii = self._radii(rng, box.volume)
        lengths = box.lengths / self.scale  # the box of the spheres
        if self.packing == "dense":
            if np.any(lengths < 6.0 * radii.max()):
                raise ValueError(
                    "level_set_foam: the dense packing needs a box of three seed diameters along "
                    "each axis (the long ones included): enlarge it or use packing: rsa"
                )
            pos = grown_packing(radii, lengths, rng, growth_rate=self.growth_rate)
        else:
            order = np.argsort(-radii, kind="stable")
            pos = np.empty((len(radii), 3))
            pos[order] = random_sequential_addition(
                radii[order], lengths, 0.0, rng, 10_000 * len(radii)
            )
        centres = box.lower + pos * self.scale
        return centres, radii[:, None] * self.scale, box

    def generate(self, domain: Domain, seed: SeedLike = None) -> DistanceFieldFoam:
        rng, seed_info = make_rng(seed)
        centres, axes, box = self.seeds(domain, rng)
        foam = DistanceFieldFoam(
            domain,
            centres,
            axes,
            box.lower,
            box.lengths,
            self.thickness if self.thickness is not None else 0.0,
            curvature=self.curvature,
            c1=self.c1,
            c2=self.c2,
            cells=self.cells,
            wall_thickness=self.wall_thickness,
            periodic_box=True,
            model=self,
            seed=seed_info,
        )
        # the struts (and the thickness) on a grid of about eight voxels per short semi-axis
        h = float(self.seed_radius.moment(1)) / 8.0
        n = np.maximum(8, np.ceil(domain.lengths / h)).astype(int)
        while np.prod(n) > 2_000_000:
            n = np.maximum(8, n // 2 + 1)
        grid = Grid(domain, tuple(int(v) for v in n))
        pts = np.stack(np.meshgrid(*[grid.centers(a) for a in range(3)], indexing="ij"), axis=-1)
        dn, nn = foam.distances(pts.reshape(-1, 3))
        foam.learn_struts(dn, nn)
        if self.volume_fraction is not None:
            foam.thickness = self._solve_thickness(foam, dn, nn)
        return foam

    def _solve_thickness(self, foam, dn, nn) -> float:
        """Bisection of the solid fraction of the generation grid (increasing with t)."""
        target = self.volume_fraction
        lo, hi = 0.0, 0.5
        while np.mean(foam.phi(dn, nn, hi) <= 0.0) < target:
            lo, hi = hi, 2.0 * hi
            if hi > 64.0:
                raise ValueError("level_set_foam: the volume fraction cannot be reached")
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if np.mean(foam.phi(dn, nn, mid) <= 0.0) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def expected_count(self, domain: Domain) -> float:
        shape = float(np.prod(self.scale))
        return (
            self.packing_fraction
            * domain.volume
            / (4.0 / 3.0 * math.pi * shape * self.seed_radius.moment(3))
        )

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        shape = float(np.prod(self.scale))
        return {
            "seed_packing_fraction": self.packing_fraction,
            "seed_intensity": self.packing_fraction
            / (4.0 / 3.0 * math.pi * shape * self.seed_radius.moment(3)),
        }


#: Keys of the configurations of distributions that are lengths.
_LENGTHS = ("value", "low", "high", "mean", "std")
#: d log kappa_D / d (log s, log b/a, log c/a) of the sections normal to x, y and z of
#: ellipsoids of semi-axes a, b, c along x, y, z (kappa_x = sqrt(bc)/a, kappa_y = sqrt(c/a),
#: kappa_z = sqrt(b/a), in units of a).
_SECTION_JACOBIAN = np.array([[1.0, 0.5, 0.5], [1.0, 0.0, 0.5], [1.0, 0.5, 0.0]])


def _scaled(distribution: Distribution, factor: float) -> Distribution:
    """``distribution`` of lengths multiplied by ``factor``."""
    cfg = distribution.to_config()
    if not isinstance(cfg, dict):
        return Constant(float(cfg) * factor)
    out = {}
    for key, value in cfg.items():
        if key == "values":
            out[key] = [factor * float(v) for v in value]
        elif key in _LENGTHS:
            out[key] = factor * float(value)
        else:
            out[key] = value
    return distribution_from_config(out)


@dataclass
class SeedCalibration:
    """The result of :func:`calibrate_seeds`: the calibrated ``model``, the ``scale`` of its
    seeds relative to the initial model and their ``aspect``, the ``target`` and
    ``achieved`` mean section radii (axis -> radius), the ``history`` of (scale, aspect,
    achieved) over the iterations and the ``sections`` of the last foam."""

    model: Model
    scale: float
    aspect: tuple[float, float]
    target: dict[int, float]
    achieved: dict[int, float]
    history: list
    sections: list

    @property
    def residual(self) -> float:
        """The largest relative deviation of the achieved mean section radii."""
        return max(abs(self.achieved[a] / self.target[a] - 1.0) for a in self.target)


def calibrate_seeds(
    model: LevelSetFoam,
    domain: Domain,
    target,
    *,
    shape=None,
    voxel_size=None,
    phase: int = 0,
    fit_shape: bool = False,
    separate: bool = True,
    prominence: float | None = None,
    min_radius: float = 0.0,
    iterations: int = 8,
    tol: float = 0.01,
    seed: SeedLike = None,
) -> SeedCalibration:
    """Scale the seeds of a foam, and with ``fit_shape`` change their shape factors, until
    the mean radii of the sections of its ``phase`` (the pores) by the planes normal to the
    axes of ``target`` (a mapping axis -> mean radius, or the measured
    :class:`~morphogen.analysis.stereology.Sections`) match them within ``tol``.
    Gauss–Newton steps in the logarithms of the scale and of the shape factors with the
    Jacobian of ellipsoids, ``log r_D = log s + log kappa_D(b/a, c/a)`` (``fit_shape``
    needs the three axes; otherwise the step is the mean logarithmic deviation, and the
    iterations stop when it falls below ``tol / 2``, a shape that does not fit the
    sections leaving its misfit in :attr:`SeedCalibration.residual`); each foam
    is generated with the same ``seed`` (drawn once if None) on the grid ``shape`` or
    ``voxel_size`` and measured as the target should have been (``separate``,
    ``prominence``, ``min_radius``: see
    :func:`~morphogen.analysis.stereology.section_sizes`)."""
    from morphogen.analysis.stereology import section_sizes

    if isinstance(target, dict):
        means = {int(a): float(r) for a, r in target.items()}
    else:
        means = {s.axis: s.mean_radius() for s in target}
    axes = sorted(means)
    if fit_shape and axes != [0, 1, 2]:
        raise ValueError("calibrate_seeds: fit_shape needs the sections normal to x, y and z")
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31))
    base = model.seed_radius
    log_p = np.log([1.0, *model.aspect])
    goal = np.log([means[a] for a in axes])
    jacobian = _SECTION_JACOBIAN[axes] if fit_shape else np.ones((len(axes), 1))
    history = []
    for it in range(int(iterations) + 1):
        scale, aspect = float(np.exp(log_p[0])), tuple(float(v) for v in np.exp(log_p[1:]))
        current = dataclasses.replace(model, seed_radius=_scaled(base, scale), aspect=aspect)
        image = current.generate(domain, seed=seed).voxelize(shape, voxel_size=voxel_size)
        sections = [
            section_sizes(
                image, phase, a, separate=separate, prominence=prominence, min_radius=min_radius
            )
            for a in axes
        ]
        got = np.log([s.mean_radius() for s in sections])
        achieved = {a: float(np.exp(g)) for a, g in zip(axes, got, strict=True)}
        history.append((scale, aspect, achieved))
        residual = goal - got
        if np.abs(np.expm1(residual)).max() < tol or it == int(iterations):
            break
        step, *_ = np.linalg.lstsq(jacobian, residual, rcond=None)
        if np.abs(step).max() < 0.5 * tol:
            break  # settled: what remains is the misfit of the shape
        if fit_shape:
            log_p += step
        else:
            log_p[0] += step[0]
    return SeedCalibration(current, scale, aspect, means, achieved, history, sections)
