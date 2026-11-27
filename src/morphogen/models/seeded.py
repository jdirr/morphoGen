"""Grains grown from seeds (MicroStructPy: Hart & Rimoli 2020, *Comput. Methods Appl. Mech.
Eng.* 370, 113242; *SoftwareX* 12, 100595): the Laguerre tessellation of the spheres that
approximate the seeds, the cells of the spheres of each seed gathered into its grain.

The seeds are ellipsoids of an equivalent diameter ``d`` (the diameter of the sphere of
their volume) and of the ratios ``a/b`` and ``b/c`` of their semi-axes ``a >= b >= c``,
drawn from their laws until their volumes fill the box, their axes turned at random or
aligned with those of the box, and each of a phase drawn from the fractions of the
phases. They are placed the largest first, each at the first of uniform positions where
its spheres overlap those of the seeds placed by at most ``overlap`` times the smaller
radius (:mod:`morphogen.kernels.seeds`); by default the tolerance of Hart and Rimoli, a
rational function of the coefficient of variation of the volumes of the seeds fitted to
their simulations (with the coefficients that MicroStructPy fits to their Fig. 10). A
seed that clears at none of its positions is placed at the one of the least excess of
overlap (``crowded: place``), or left out as MicroStructPy does (``crowded: leave``), its
volume going to its neighbours: the sequential addition of elongated seeds jams early,
and leaving them out makes the grains larger than the seeds.

**Breakdown.** An ellipsoid of semi-axes ``a >= b >= c`` is approximated from inside by the
spheres centred on its medial surface, the ellipse of semi-axes ``(a² - c²)/a`` and
``(b² - c²)/b`` in its plane ``z = 0`` (the segment ``|x| <= (a² - c²)/a`` of a prolate
spheroid), each the largest sphere of the ellipsoid about its centre: from ``(x, y, 0)``
the nearest points of the surface are ``(a² x/(a² - c²), b² y/(b² - c²), ±z)``, which gives
its radius in closed form (``c`` at the centre, the radius of curvature ``c²/a`` at the
ends of the long axis). The centres are a triangular lattice of the medial ellipse and
points of its rim, at the spacing ``spacing`` times ``c`` (widened to keep at most
:data:`MOST_SPHERES` spheres per seed): the union of the spheres fills 94 to 97 % of the
ellipsoid at the spacing 0.5 and 98 to 99 % at 0.25, whatever its shape; a sphere is its
own breakdown.

**The grains.** The spheres of all the seeds are the germs of a Laguerre tessellation, their
weights the squares of their radii: a point belongs to the sphere of least power distance
``|x - x_k|² - r_k²``, and to the grain of its seed (:class:`SeededTessellation`). The
grains are thus unions of the convex cells of their spheres, exact at any point (the
k-d tree of the tessellation), and their shapes follow those of the seeds where the seeds
fill the box: their volumes those of the seeds, their elongations and their axes those of
the seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen._utils import check_positive
from morphogen.distributions import Constant, Distribution
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.base import Model, param, register, register_kind
from morphogen.models.tessellation import _CHUNK, Tessellation
from morphogen.orientations import (
    OrientationDistribution,
    euler_to_matrix,
    fibre_orientations,
    random_orientations,
)
from morphogen.seeding import SeedLike, make_rng

#: The largest number of spheres of the breakdown of a seed
MOST_SPHERES = 256
#: The orientations of the axes of the seeds
SEED_ORIENTATIONS = ("random", "aligned")
#: What becomes of a seed that clears at none of its positions
CROWDED = ("place", "leave")


def ellipsoid_spheres(a: float, b: float, c: float, *, spacing: float = 0.5,
                      most: int = MOST_SPHERES) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """The spheres inscribed in the ellipsoid ``x²/a² + y²/b² + z²/c² <= 1``
    (``a >= b >= c``) centred on its medial surface (see the module): their centres (in
    the frame of the ellipsoid) and their radii."""
    a, b, c = float(a), float(b), float(c)
    if not a >= b >= c > 0:
        raise ValueError("semi-axes: a >= b >= c > 0")
    if a - c <= 1e-12 * a:  # a sphere
        return np.zeros((1, 3)), np.array([a])
    long_ = (a * a - c * c) / a  # the medial ellipse
    wide = (b * b - c * c) / b
    prolate = wide <= 1e-9 * a
    step = float(spacing) * c
    while True:
        if prolate:  # the medial segment
            n = max(2, math.ceil(2.0 * long_ / step) + 1)
            points = np.column_stack([np.linspace(-long_, long_, n), np.zeros(n)])
        else:  # a triangular lattice of the medial ellipse, and its rim
            h = step * math.sqrt(3.0) / 2.0
            rows = []
            for k in range(-math.floor(wide / h), math.floor(wide / h) + 1):
                y = k * h
                half = long_ * math.sqrt(max(0.0, 1.0 - (y / wide) ** 2))
                shift = 0.5 * step if k % 2 else 0.0
                xs = (np.arange(-math.floor((half + shift) / step) - 1,
                                math.floor((half - shift) / step) + 2) * step + shift)  # fmt: skip
                rows += [(x, y) for x in xs[np.abs(xs) <= half]]
            m = max(8, math.ceil(math.pi * (long_ + wide) / step))
            t = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False)
            rim = np.column_stack([long_ * np.cos(t), wide * np.sin(t)])
            points = np.vstack([np.asarray(rows, dtype=float).reshape(-1, 2), rim])
        if len(points) <= most or step > 4.0 * c:
            break
        step *= 1.25
    x, y = points[:, 0], points[:, 1]
    q1 = a * a * x / (a * a - c * c)
    q2 = np.zeros_like(y) if prolate else b * b * y / (b * b - c * c)
    off = c * c * np.clip(1.0 - (q1 / a) ** 2 - (q2 / b) ** 2, 0.0, None)
    radii = np.sqrt((x - q1) ** 2 + (y - q2) ** 2 + off)
    return np.column_stack([x, y, np.zeros_like(x)]), radii


def overlap_tolerance(volumes) -> float:
    """The overlap of the seeds tolerated at their placement (relative to the smaller
    sphere) for the coefficient of variation of their volumes: the rational fit of Hart and
    Rimoli (2020, Fig. 10) in 3D, with MicroStructPy's coefficients, between 0 and 1."""
    v = np.asarray(volumes, dtype=float)
    cv = float(v.std() / v.mean()) if len(v) > 1 and v.mean() > 0 else 0.0
    numer = 0.471115 * cv * cv - 0.602324 * cv + 0.297562
    denom = cv * cv - 1.08469 * cv + 0.428216
    return float(np.clip(numer / denom, 0.0, 1.0))


@register_kind
class SeededTessellation(Tessellation):
    """The grains of seeds (see the module): the Laguerre cells of the spheres ``germs``
    (weights the squares of their radii), the sphere ``k`` of the grain ``grains[k]``;
    ``grain_phases`` and the crystal ``orientations`` (Bunge angles) of the grains;
    ``seeds``: their centres, semi-axes and the angles of their axes. The labels are those
    of the grains."""

    kind = "seeded_tessellation"

    def __init__(self, domain: Domain, germs, weights, grains, grain_phases=None,
                 orientations=None, seeds=None, **kwargs) -> None:  # fmt: skip
        grains = np.asarray(grains, dtype=np.int64).reshape(-1)
        n_grains = int(grains.max()) + 1 if len(grains) else 0
        if grain_phases is None:
            grain_phases = np.ones(n_grains, dtype=np.uint8)
        grain_phases = np.asarray(grain_phases, dtype=np.uint8).reshape(n_grains)
        super().__init__(domain, germs, weights, grain_phases[grains], None, **kwargs)
        self.grains = grains
        self.grain_phases = grain_phases
        if orientations is not None:
            orientations = np.asarray(orientations, dtype=float).reshape(n_grains, 3)
        self.orientations = orientations
        self.seeds = {k: np.asarray(v, dtype=float) for k, v in (seeds or {}).items()}

    @property
    def n_grains(self) -> int:
        """The number of grains (of seeds placed)."""
        return len(self.grain_phases)

    def cell_grains(self) -> np.ndarray:
        """The grain of each cell: that of the seed of its sphere."""
        return self.grains

    def labels(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Image of the grain labels (int32)."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        cells = Tessellation.labels(self, grid=grid)
        data = self.grains[cells.data].astype(np.int32)
        return VoxelImage.from_grid(data, grid, name="grain", metadata=self.image_metadata())

    def _phases_on(self, grid: Grid) -> np.ndarray:
        data = self.grain_phases[self.labels(grid=grid).data]
        if self.boundary_thickness > 0:
            half = 0.5 * self.boundary_thickness
            data[self._boundary_image(grid, half) < half] = self.boundary_phase
        return data

    def boundary_distance(self, points, reach: float = math.inf) -> np.ndarray:
        """The distance from each point to the nearest of the faces between its cell and
        the cells of the other grains (``inf`` for a single grain), exact up to ``reach``:
        the distance to the boundary of its grain where its own cell holds the nearest
        part, a lower bound otherwise (a face of another cell of its grain nearer)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        out = np.full(len(pts), np.inf)
        n = self.n_cells
        if self.n_grains < 2 or not len(pts):
            return out
        tree = self._get_tree()
        lengths = self.domain.lengths
        k0 = min(n, 32)
        for start in range(0, len(pts), max(1, _CHUNK // k0)):
            todo = np.arange(start, min(len(pts), start + max(1, _CHUNK // k0)))
            k = k0
            while todo.size:
                q = self._lift(pts[todo])
                if self.weights is not None:
                    q = np.hstack([q, np.zeros((len(q), 1))])
                dist, idx = tree.query(q, k=k)
                v = self.germs[idx] - pts[todo, None, :]
                if self.domain.periodic:
                    v -= lengths * np.round(v / lengths)
                gap = np.linalg.norm(v[:, 1:] - v[:, :1], axis=2)
                d = (dist[:, 1:] ** 2 - dist[:, :1] ** 2) / (2.0 * np.maximum(gap, 1e-300))
                other = self.grains[idx[:, 1:]] != self.grains[idx[:, :1]]
                out[todo] = np.where(other, d, np.inf).min(axis=1)
                bound = dist[:, :1].ravel() + 2.0 * np.minimum(out[todo], reach)
                again = (dist[:, -1] <= bound) & (k < n)
                todo, k = todo[again], min(n, 2 * k)
        return out

    def grain_volumes(self) -> np.ndarray:
        """The exact volumes of the grains in the box: the sums of the volumes of the
        Laguerre cells of their spheres (:meth:`cell_complex`)."""
        cx = self.cell_complex()
        volumes = np.bincount(self.grains[np.asarray(cx.piece_cell)], cx.piece_volumes(),
                              minlength=self.n_grains)  # fmt: skip
        return volumes

    def cell_table(self) -> dict[str, np.ndarray]:
        """Per-grain attributes: id, phase, the number of its spheres, its seed (centre,
        semi-axes, the angles of its axes) and its crystal orientation."""
        table = {
            "id": np.arange(self.n_grains),
            "phase": self.grain_phases.astype(int),
            "spheres": np.bincount(self.grains, minlength=self.n_grains),
        }
        names = {"centres": ("x", "y", "z"), "axes": ("a", "b", "c"),
                 "angles": ("axis_phi1", "axis_Phi", "axis_phi2")}  # fmt: skip
        for key, columns in names.items():
            if key in self.seeds:
                for i, column in enumerate(columns):
                    table[column] = self.seeds[key][:, i]
        if self.orientations is not None:
            table["phi1"] = self.orientations[:, 0]
            table["Phi"] = self.orientations[:, 1]
            table["phi2"] = self.orientations[:, 2]
        return table

    def summary(self) -> dict:
        return {"n_grains": self.n_grains, "n_spheres": self.n_cells}

    def _geometry(self) -> dict:
        geo = {"germs": self.germs.tolist(), "weights": self.weights.tolist(),
               "grains": self.grains.tolist(), "grain_phases": self.grain_phases.tolist(),
               "seeds": {k: v.tolist() for k, v in self.seeds.items()}}  # fmt: skip
        if self.orientations is not None:
            geo["orientations"] = self.orientations.tolist()
        return geo

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(domain, geometry["germs"], geometry.get("weights"), geometry["grains"],
                   geometry.get("grain_phases"), geometry.get("orientations"),
                   geometry.get("seeds"), **kwargs)  # fmt: skip


@register
@dataclass(frozen=True)
class SeededLaguerre(Model):
    """Grains grown from ellipsoidal seeds (MicroStructPy): the Laguerre cells of the
    spheres that approximate the seeds, gathered seed by seed (see the module)."""

    name = "seeded_laguerre"
    phases: ClassVar[dict[int, str]] = {0: "phase 0", 1: "phase 1"}
    count_label = "grains"
    path_parameters: ClassVar[tuple[str, ...]] = ("texture.file",)

    diameter: Distribution = param(
        Constant(1.0),
        doc="Equivalent diameter of the seeds: that of the sphere of their volume",
        kind="distribution",
        unit="length",
    )
    aspect_ratio: Distribution = param(
        Constant(1.0),
        doc="Ratio a/b of the longest semi-axis of a seed to the middle one (at least 1)",
        kind="distribution",
    )
    flatness: Distribution = param(
        Constant(1.0),
        doc="Ratio b/c of the middle semi-axis of a seed to the shortest (at least 1; with "
        "an aspect ratio of 1, spheres)",
        kind="distribution",
    )
    orientation: str = param(
        "random",
        doc="The axes of the seeds: random (uniform rotations) or aligned (a along x, b "
        "along y, c along z)",
        kind="str",
    )
    phase_fractions: tuple[float, ...] | None = param(
        None,
        doc="Fractions of the volume of phases 0, 1, ... among the grains (default: every "
        "grain of phase 1)",
        kind="floats",
    )
    overlap: float | None = param(
        None,
        doc="Overlap of the seeds tolerated at their placement, relative to the smaller of "
        "two spheres (0: none; default: the fit of Hart and Rimoli to the variation of "
        "the volumes of the seeds)",
        kind="fraction",
    )
    spacing: float = param(
        0.5,
        doc="Spacing of the spheres of a seed on its medial surface, in units of its "
        "shortest semi-axis (0.5: the spheres fill 94 to 97 % of the seed)",
    )
    attempts: int = param(
        10000, doc="Positions tried for a seed before it is placed crowded or left out", kind="int"
    )
    crowded: str = param(
        "place",
        doc="A seed that clears at none of its positions: place (at the one of the least "
        "excess of overlap) or leave (left out, as MicroStructPy)",
        kind="str",
    )
    crystal_orientations: bool = param(
        False, doc="Draw a uniformly random crystal orientation per grain", kind="bool"
    )
    texture: OrientationDistribution | None = param(
        None,
        doc="The texture of the crystal orientations of the grains (as for voronoi)",
        kind="orientation",
        kw_only=True,
    )

    def _validate(self) -> None:
        if not self.diameter.mean() > 0:
            raise ValueError("diameter: positive")
        for name in ("aspect_ratio", "flatness"):
            law = getattr(self, name)
            lowest = float(np.min(law.quadrature(64)[0]))
            if lowest < 1.0 - 1e-12:
                raise ValueError(f"{name}: ratios of at least 1 (the semi-axes a >= b >= c)")
        if self.orientation not in SEED_ORIENTATIONS:
            raise ValueError(f"orientation: one of {SEED_ORIENTATIONS}")
        if self.phase_fractions is not None:
            p = np.asarray(self.phase_fractions, dtype=float)
            if p.ndim != 1 or len(p) < 1 or np.any(p < 0) or abs(p.sum() - 1.0) > 1e-9:
                raise ValueError("phase_fractions: non-negative fractions summing to 1")
            if len(p) > 255:
                raise ValueError("phase_fractions: at most 255 phases")
        if self.overlap is not None and not 0.0 <= self.overlap <= 1.0:
            raise ValueError("overlap: between 0 and 1")
        check_positive(self.spacing, "spacing")
        if self.attempts < 1:
            raise ValueError("attempts: a positive number")
        if self.crowded not in CROWDED:
            raise ValueError(f"crowded: one of {CROWDED}")

    def active_phases(self) -> dict[int, str]:
        if self.phase_fractions is None:
            return {1: "grains"}
        return {i: f"phase {i}" for i, p in enumerate(self.phase_fractions) if p > 0}

    def mean_seed_volume(self) -> float:
        """The mean volume of the seeds, ``π E[d³]/6``."""
        return math.pi / 6.0 * self.diameter.moment(3)

    def _seeds(self, rng, volume: float):
        """Seeds whose volumes fill ``volume``: their semi-axes (a >= b >= c)."""
        drawn, total = [], 0.0
        batch = max(8, math.ceil(1.2 * volume / self.mean_seed_volume()))
        while total < volume:
            d = self.diameter.sample(rng, batch)
            ab = self.aspect_ratio.sample(rng, batch)
            bc = self.flatness.sample(rng, batch)
            c = 0.5 * d / np.cbrt(ab * bc * bc)  # a b c = (d/2)^3
            axes = np.column_stack([ab * bc * c, bc * c, c])
            v = math.pi / 6.0 * d**3
            room = np.cumsum(v) + total
            take = int(np.searchsorted(room, volume)) + 1  # the first that fills it
            drawn.append(axes[:take])
            total = float(room[min(take, batch) - 1])
        return np.vstack(drawn)

    def generate(self, domain: Domain, seed: SeedLike = None) -> SeededTessellation:
        from morphogen.kernels.seeds import place_seeds

        rng, seed_info = make_rng(seed)
        if domain.periodic:
            window, margin = domain, math.inf
        else:  # seeds about the box too, so that its faces cut grains
            margin = float(self.diameter.upper())
            window = domain.dilated(margin)
        axes = self._seeds(rng, window.volume)
        n = len(axes)
        volumes = 4.0 / 3.0 * math.pi * np.prod(axes, axis=1)
        if self.orientation == "random":
            angles = random_orientations(rng, n)
        else:
            angles = np.zeros((n, 3))
        rotations = np.transpose(euler_to_matrix(angles), (0, 2, 1))  # seed -> sample
        shapes: dict[tuple, tuple] = {}
        offsets, radii, first = [], [], [0]
        order = np.argsort(-volumes, kind="stable")  # the largest first
        for s in order:
            key = tuple(np.round(axes[s], 12))
            if key not in shapes:
                shapes[key] = ellipsoid_spheres(*axes[s], spacing=self.spacing)
            local, r = shapes[key]
            offsets.append(local @ rotations[s].T)
            radii.append(r)
            first.append(first[-1] + len(r))
        offsets = np.vstack(offsets)
        radii = np.concatenate(radii)
        tolerance = overlap_tolerance(volumes) if self.overlap is None else float(self.overlap)
        centres, placed, crowded, positions = place_seeds(
            offsets, radii, np.asarray(first), window.lengths, rng,
            periodic=domain.periodic, rtol=tolerance, attempts=int(self.attempts),
            crowded=self.crowded == "place",
        )  # fmt: skip
        kept = np.flatnonzero(placed)
        if not len(kept):
            raise ValueError("seeded_laguerre: no seed could be placed (a larger box)")
        grain_ids = np.repeat(np.arange(len(kept)), np.diff(first)[kept])
        rows = np.concatenate([np.arange(first[s], first[s + 1]) for s in kept])
        germs = positions[rows] + window.lower
        weights = radii[rows] ** 2
        seeds_kept = order[kept]
        phases = np.ones(len(kept), dtype=np.uint8)
        if self.phase_fractions is not None:
            p = np.asarray(self.phase_fractions, dtype=float)
            phases = rng.choice(len(p), size=len(kept), p=p / p.sum()).astype(np.uint8)
        crystals = None
        if self.crystal_orientations and self.texture is not None:
            crystals = fibre_orientations(rng, len(kept), self.texture)
        elif self.crystal_orientations:
            crystals = random_orientations(rng, len(kept))
        seeds = {"centres": centres[kept] + window.lower, "axes": axes[seeds_kept],
                 "angles": angles[seeds_kept]}  # fmt: skip
        metadata = {"overlap": tolerance, "seeds_drawn": n, "seeds_left_out": n - len(kept),
                    "seeds_crowded": int(crowded.sum())}  # fmt: skip
        if math.isfinite(margin):
            metadata["germ_margin"] = margin
        return SeededTessellation(domain, germs, weights, grain_ids, phases, crystals, seeds,
                                  model=self, seed=seed_info, metadata=metadata)  # fmt: skip

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """The mean volume of the seeds, that of the grains where they fill the box, and the
        number of grains it implies in the domain; the fractions of the phases."""
        out = {"mean_grain_volume": self.mean_seed_volume()}
        if domain is not None:
            out["expected_grains"] = domain.volume / out["mean_grain_volume"]
        if self.phase_fractions is not None:
            for i, p in enumerate(self.phase_fractions):
                out[f"volume_fraction_{i}"] = float(p)
        return out


__all__ = ["CROWDED", "MOST_SPHERES", "SEED_ORIENTATIONS", "SeededLaguerre", "SeededTessellation",
           "ellipsoid_spheres", "overlap_tolerance"]  # fmt: skip
