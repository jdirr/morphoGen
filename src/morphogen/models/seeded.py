"""Grains grown from seeds (MicroStructPy: Hart & Rimoli 2020, *Comput. Methods Appl. Mech.
Eng.* 370, 113242; *SoftwareX* 12, 100595): the Laguerre tessellation of the spheres that
approximate the seeds, the cells of the spheres of each seed gathered into its grain.

The seeds are ellipsoids of an equivalent diameter ``d`` (the diameter of the sphere of
their volume) and of the ratios ``a/b`` and ``b/c`` of their semi-axes ``a >= b >= c``,
drawn from their laws until their volumes fill the box, their axes turned at random or
aligned with those of the box. The seeds of each phase may have their own laws
(``phase_diameter``, ``phase_aspect_ratio``, ``phase_flatness``, ``phase_orientation``):
a seed is of the phase ``i`` with the probability ``p_i ∝ q_i / E[V_i]`` (MicroStructPy),
``q_i`` the fraction of the phase, ``E[V_i]`` the mean volume of its seeds, so that each
phase takes its fraction of the volume. The seeds of an amorphous phase
(``amorphous_phases``: a matrix, a glass) are merged into one region without grain
boundaries; a void is an amorphous phase that the meshes leave out (their ``phases``).
With ``dimension: 2`` the seeds are ellipses in the middle plane z of the box (the diameter
of the disc of their area, their ratio a/b, an angle in the plane), broken down as the
sections of the prolate spheroids ``(a, b, b)``: the grains are columnar, the prisms of the
power diagram of their circles in the plane. They are placed the largest first, each at
the first of uniform positions where
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
the seeds. :meth:`SeededTessellation.seed_comparison` checks it on the exact cells, as the
verification of MicroStructPy does on its meshes: the ellipsoids of the same second
moments as the grains (:meth:`~morphogen.models.tessellation.Tessellation.grain_shapes`)
against their seeds, and the grains of each phase against the laws of its seeds
(Kolmogorov–Smirnov).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import numpy as np

from morphogen._utils import check_positive
from morphogen.distributions import Constant, Distribution, ks_statistic
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


def overlap_tolerance(volumes, dimension: int = 3) -> float:
    """The overlap of the seeds tolerated at their placement (relative to the smaller
    sphere) for the coefficient of variation of their volumes (of their areas in the plane):
    the rational fit of Hart and Rimoli (2020, Fig. 10), with MicroStructPy's coefficients
    in 3D and in 2D, between 0 and 1."""
    v = np.asarray(volumes, dtype=float)
    cv = float(v.std() / v.mean()) if len(v) > 1 and v.mean() > 0 else 0.0
    if dimension == 2:
        numer = 0.362954 * cv * cv - 0.419069 * cv + 0.184959
        denom = cv * cv - 1.05989 * cv + 0.365096
    else:
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
        self.seeds = {k: np.asarray(v, dtype=float).reshape(n_grains, -1)
                      for k, v in (seeds or {}).items()}  # fmt: skip

    @property
    def n_grains(self) -> int:
        """The number of grains (of seeds placed)."""
        return len(self.grain_phases)

    def cell_grains(self) -> np.ndarray:
        """The grain of each cell: that of the seed of its sphere."""
        return self.grains

    @property
    def amorphous(self) -> np.ndarray:
        """Whether each grain is the region of an amorphous phase (the seeds of the phase
        merged; no seed of its own) rather than the crystal of a seed."""
        axes = self.seeds.get("axes")
        if axes is None or not len(axes):
            return np.zeros(self.n_grains, dtype=bool)
        return ~np.isfinite(axes[:, 0])

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

    def seed_comparison(self, shapes: dict | None = None) -> dict:
        """The grains against their seeds (the verification of MicroStructPy, from the exact
        shapes of the grains, :meth:`grain_shapes`): for each crystal (the grain of a seed,
        not an amorphous region) the equivalent diameters and the ratios a/b and b/c of its
        seed and of the ellipsoid of its grain, the angle in degrees between their
        distinguished axes (the long one, or the short one of a flat seed; NaN for a
        sphere); ``errors``: for each quantity, over the crystals that the faces of the box
        do not cut, the mean error (``bias``), the mean absolute error, the root mean square
        error, the largest error, the coefficient of determination ``R²`` and the
        Kolmogorov–Smirnov distance of the grains of each phase to the law of the seeds of
        the phase (a continuous law); ``fractions``: the fraction of the volume of each
        phase, realized and asked. In the plane, the ellipses of the seeds and of the
        sections of the grains: the diameters of the discs of their areas, their ratios
        a/b and their long axes."""
        shapes = self.grain_shapes() if shapes is None else shapes
        crystal = ~self.amorphous
        if "axes" not in self.seeds:
            raise ValueError("seed_comparison: the realization has no seeds")
        seed_axes = self.seeds["axes"]
        semi = shapes["semi_axes"]
        planar = self.metadata.get("dimension") == 2
        if planar:  # ellipses: the diameter of the disc of their area, a/b
            values = {
                "diameter": (2.0 * np.sqrt(seed_axes[:, 0] * seed_axes[:, 1]), shapes["diameter"]),
                "aspect_ratio": (seed_axes[:, 0] / seed_axes[:, 1], semi[:, 0] / semi[:, 1]),
            }
        else:
            values = {
                "diameter": (2.0 * np.cbrt(np.prod(seed_axes, axis=1)), shapes["diameter"]),
                "aspect_ratio": (seed_axes[:, 0] / seed_axes[:, 1], semi[:, 0] / semi[:, 1]),
                "flatness": (seed_axes[:, 1] / seed_axes[:, 2], semi[:, 1] / semi[:, 2]),
            }
        # the distinguished axis: the long one, or the short one of a flat seed
        rows = euler_to_matrix(np.nan_to_num(self.seeds["angles"]))  # rows: axes of the seeds
        ab = values["aspect_ratio"][0]
        bc = values["flatness"][0] if "flatness" in values else np.ones_like(ab)
        long_ = ab >= bc
        last = shapes["axes"].shape[2] - 1
        seed_axis = np.where(long_[:, None], rows[:, 0, :], rows[:, 2, :])
        grain_axis = np.where(long_[:, None], shapes["axes"][:, :, 0], shapes["axes"][:, :, last])
        cos = np.abs(np.einsum("ni,ni->n", seed_axis, grain_axis))
        angle = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
        angle[(ab < 1.0 + 1e-9) & (bc < 1.0 + 1e-9)] = np.nan
        use = crystal & shapes["inside"]
        phases = self.grain_phases.astype(np.int64)
        model = getattr(self, "model", None)
        errors = {}
        for name, (seed, grain) in values.items():
            r = (grain - seed)[use]
            spread = grain[use] - np.mean(grain[use]) if use.any() else r
            mse = float(np.mean(r * r)) if r.size else math.nan
            base = float(np.mean(spread * spread)) if r.size else math.nan
            errors[name] = {
                "bias": float(np.mean(r)) if r.size else math.nan,
                "mae": float(np.mean(np.abs(r))) if r.size else math.nan,
                "rmse": math.sqrt(mse) if r.size else math.nan,
                "max": float(np.max(np.abs(r))) if r.size else math.nan,
                "r2": 1.0 - mse / base if r.size and base > 0 else math.nan,
                "ks": {},
            }
            if isinstance(model, SeededLaguerre):
                for f in model.families():
                    mine = use & (phases == f.phase)
                    if mine.any():
                        errors[name]["ks"][f.phase] = ks_statistic(grain[mine], getattr(f, name))
        a = angle[use & np.isfinite(angle)]
        errors["axis_angle"] = {"mean": float(np.mean(a)) if a.size else math.nan,
                                "max": float(np.max(a)) if a.size else math.nan}  # fmt: skip
        volume = np.bincount(phases, shapes["volume"], minlength=int(phases.max(initial=0)) + 1)
        asked = getattr(model, "phase_fractions", None)
        if asked is None and isinstance(model, SeededLaguerre):  # every grain of phase 1
            asked = (0.0, 1.0)
        fractions = {
            int(k): (float(volume[k] / volume.sum()),
                     float(asked[k]) if asked is not None and k < len(asked) else math.nan)
            for k in np.unique(phases)
        }  # fmt: skip
        out = {"grain": np.flatnonzero(crystal), "phase": phases[crystal],
               "inside": shapes["inside"][crystal], "axis_angle": angle[crystal]}  # fmt: skip
        for name, (seed, grain) in values.items():
            out[f"seed_{name}"] = seed[crystal]
            out[f"grain_{name}"] = grain[crystal]
        out["errors"] = errors
        out["fractions"] = fractions
        return out

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
        def listed(v):  # the amorphous regions have no seed: null
            return np.where(np.isfinite(v), v, None).tolist()

        geo = {"germs": self.germs.tolist(), "weights": self.weights.tolist(),
               "grains": self.grains.tolist(), "grain_phases": self.grain_phases.tolist(),
               "seeds": {k: listed(v) for k, v in self.seeds.items()}}  # fmt: skip
        if self.orientations is not None:
            geo["orientations"] = self.orientations.tolist()
        return geo

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(domain, geometry["germs"], geometry.get("weights"), geometry["grains"],
                   geometry.get("grain_phases"), geometry.get("orientations"),
                   geometry.get("seeds"), **kwargs)  # fmt: skip


class SeedFamily(NamedTuple):
    """The seeds of a phase: their laws, the fraction of the volume of their phase and
    their share of the seeds."""

    phase: int
    diameter: Distribution
    aspect_ratio: Distribution
    flatness: Distribution
    orientation: str
    fraction: float
    share: float
    dimension: int = 3

    def mean_volume(self) -> float:
        """The mean volume of its seeds, ``π E[d³]/6`` (their mean area ``π E[d²]/4`` in
        the plane)."""
        if self.dimension == 2:
            return math.pi / 4.0 * self.diameter.moment(2)
        return math.pi / 6.0 * self.diameter.moment(3)


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
        doc="The axes of the seeds: random (uniform rotations; uniform angles in the plane) or "
        "aligned (a along x, b along y, c along z)",
        kind="str",
    )
    dimension: int = param(
        3,
        doc="3, or 2: elliptical seeds in the middle plane z of the box (their equivalent "
        "diameter that of the disc of their area, their ratio a/b, an angle in the plane), "
        "columnar grains",
        kind="int",
    )
    phase_fractions: tuple[float, ...] | None = param(
        None,
        doc="Fractions of the volume of phases 0, 1, ... among the grains (default: every "
        "grain of phase 1)",
        kind="floats",
    )
    phase_diameter: tuple[Distribution, ...] | None = param(
        None,
        doc="Equivalent diameter of the seeds of each phase of phase_fractions (seeds of "
        "their own sizes), instead of diameter",
        kind="distributions",
        unit="length",
    )
    phase_aspect_ratio: tuple[Distribution, ...] | None = param(
        None,
        doc="Ratio a/b of the seeds of each phase, instead of aspect_ratio",
        kind="distributions",
    )
    phase_flatness: tuple[Distribution, ...] | None = param(
        None,
        doc="Ratio b/c of the seeds of each phase, instead of flatness",
        kind="distributions",
    )
    phase_orientation: tuple[str, ...] | None = param(
        None,
        doc="The axes of the seeds of each phase (random or aligned), instead of orientation",
        kind="json",
    )
    amorphous_phases: tuple[int, ...] | None = param(
        None,
        doc="Amorphous phases (a matrix, a glass): the seeds of each merged into one region, "
        "without grain boundaries (a void phase: an amorphous phase that the meshes leave "
        "out)",
        kind="ints",
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
        per_phase = ("phase_diameter", "phase_aspect_ratio", "phase_flatness",
                     "phase_orientation", "amorphous_phases")  # fmt: skip
        for name in per_phase:
            if getattr(self, name) is not None and self.phase_fractions is None:
                raise ValueError(f"{name}: give the phases (phase_fractions)")
        n = 0 if self.phase_fractions is None else len(self.phase_fractions)
        for name in per_phase[:4]:
            value = getattr(self, name)
            if value is not None and len(value) != n:
                raise ValueError(f"{name}: one value per phase of phase_fractions ({n})")
        for law in self.phase_diameter or ():
            if not law.mean() > 0:
                raise ValueError("phase_diameter: positive")
        for name in ("phase_aspect_ratio", "phase_flatness"):
            for law in getattr(self, name) or ():
                if float(np.min(law.quadrature(64)[0])) < 1.0 - 1e-12:
                    raise ValueError(f"{name}: ratios of at least 1 (the semi-axes a >= b >= c)")
        for value in self.phase_orientation or ():
            if value not in SEED_ORIENTATIONS:
                raise ValueError(f"phase_orientation: each one of {SEED_ORIENTATIONS}")
        for k in self.amorphous_phases or ():
            if not 0 <= k < n:
                raise ValueError(f"amorphous_phases: phases among 0 to {n - 1}")
        if self.dimension not in (2, 3):
            raise ValueError("dimension: 3, or 2 (seeds of the plane)")
        if self.dimension == 2:
            for law in (self.flatness, *(self.phase_flatness or ())):
                if not (law.is_constant and law.mean() == 1.0):
                    raise ValueError("flatness: the seeds of the plane are ellipses (1)")
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

    def families(self) -> list[SeedFamily]:
        """The families of seeds: one per phase of a positive fraction (of the laws of the
        phase, or of those shared), the fraction of the volume ``q`` of its seeds and its
        share ``p`` of the seeds, ``p ∝ q / E[V]`` (MicroStructPy)."""
        dim = int(self.dimension)
        if self.phase_fractions is None:
            return [SeedFamily(1, self.diameter, self.aspect_ratio, self.flatness,
                               self.orientation, 1.0, 1.0, dim)]  # fmt: skip
        q = np.asarray(self.phase_fractions, dtype=float)
        rows = []
        for i, qi in enumerate(q):
            if qi > 0:
                rows.append(SeedFamily(i, *(self._law(name, i) for name in
                                            ("diameter", "aspect_ratio", "flatness",
                                             "orientation")), float(qi), 0.0, dim))  # fmt: skip
        w = np.array([f.fraction / f.mean_volume() for f in rows])
        return [f._replace(share=float(wi / w.sum())) for f, wi in zip(rows, w, strict=True)]

    def _law(self, name: str, phase: int):
        own = getattr(self, f"phase_{name}")
        return own[phase] if own is not None else getattr(self, name)

    def mean_seed_volume(self) -> float:
        """The mean volume of the seeds, ``π E[d³]/6`` over their families
        (``1 / Σ q / E[V]``; their mean area in the plane)."""
        return 1.0 / sum(f.fraction / f.mean_volume() for f in self.families())

    def _seeds(self, rng, volume: float):
        """Seeds whose volumes fill ``volume`` (their areas in the plane): their semi-axes
        (a >= b >= c; a >= b = c for the ellipses of the plane) and their families."""
        families = self.families()
        share = np.array([f.share for f in families])
        drawn, kinds, total = [], [], 0.0
        batch = max(8, math.ceil(1.2 * volume / self.mean_seed_volume()))
        while total < volume:
            if len(families) == 1:
                kind = np.zeros(batch, dtype=np.int64)
                f = families[0]
                d = f.diameter.sample(rng, batch)
                ab = f.aspect_ratio.sample(rng, batch)
                bc = f.flatness.sample(rng, batch)
            else:
                kind = rng.choice(len(families), size=batch, p=share)
                d, ab, bc = np.empty(batch), np.empty(batch), np.empty(batch)
                for k, f in enumerate(families):
                    mine = kind == k
                    d[mine] = f.diameter.sample(rng, int(mine.sum()))
                    ab[mine] = f.aspect_ratio.sample(rng, int(mine.sum()))
                    bc[mine] = f.flatness.sample(rng, int(mine.sum()))
            if self.dimension == 2:  # ellipses: a b = (d/2)^2, broken down as (a, b, b)
                b = 0.5 * d / np.sqrt(ab)
                axes = np.column_stack([ab * b, b, b])
                v = math.pi / 4.0 * d**2
            else:
                c = 0.5 * d / np.cbrt(ab * bc * bc)  # a b c = (d/2)^3
                axes = np.column_stack([ab * bc * c, bc * c, c])
                v = math.pi / 6.0 * d**3
            room = np.cumsum(v) + total
            take = int(np.searchsorted(room, volume)) + 1  # the first that fills it
            drawn.append(axes[:take])
            kinds.append(kind[:take])
            total = float(room[min(take, batch) - 1])
        return np.vstack(drawn), np.concatenate(kinds)

    def generate(self, domain: Domain, seed: SeedLike = None) -> SeededTessellation:
        from morphogen.kernels.seeds import place_seeds

        rng, seed_info = make_rng(seed)
        families = self.families()
        planar = self.dimension == 2
        if domain.periodic:
            window, margin = domain, math.inf
        else:  # seeds about the box too, so that its faces cut grains
            margin = max(float(f.diameter.upper()) for f in families)
            window = domain.dilated(margin)
            if planar:  # in the plane only
                window = Domain(size=(*window.size[:2], domain.size[2]),
                                origin=(*window.origin[:2], domain.origin[2]), periodic=False,
                                unit=domain.unit)  # fmt: skip
        lengths = window.lengths
        axes, kind = self._seeds(rng, float(lengths[0] * lengths[1]) if planar else window.volume)
        n = len(axes)
        if planar:
            volumes = math.pi * axes[:, 0] * axes[:, 1]  # their areas
        else:
            volumes = 4.0 / 3.0 * math.pi * np.prod(axes, axis=1)
        turned = np.array([f.orientation == "random" for f in families])[kind]
        angles = np.zeros((n, 3))
        if planar:  # an angle about z
            angles[turned, 0] = rng.uniform(0.0, 2.0 * math.pi, int(turned.sum()))
        elif turned.all():
            angles = random_orientations(rng, n)
        elif turned.any():
            angles[turned] = random_orientations(rng, int(turned.sum()))
        rotations = np.transpose(euler_to_matrix(angles), (0, 2, 1))  # seed -> sample
        shapes: dict[tuple, tuple] = {}
        offsets, radii, first = [], [], [0]
        order = np.argsort(-volumes, kind="stable")  # the largest first
        for s in order:
            key = tuple(np.round(axes[s], 12))
            if key not in shapes:  # the largest spheres first: their overlaps end the tests
                local, r = ellipsoid_spheres(*axes[s], spacing=self.spacing)
                big = np.argsort(-r, kind="stable")
                shapes[key] = (local[big], r[big])
            local, r = shapes[key]
            offsets.append(local @ rotations[s].T)
            radii.append(r)
            first.append(first[-1] + len(r))
        offsets = np.vstack(offsets)
        radii = np.concatenate(radii)
        tolerance = (overlap_tolerance(volumes, self.dimension) if self.overlap is None
                     else float(self.overlap))  # fmt: skip
        centres, placed, crowded, positions = place_seeds(
            offsets, radii, np.asarray(first), window.lengths, rng,
            periodic=domain.periodic, rtol=tolerance, attempts=int(self.attempts),
            crowded=self.crowded == "place", planar=planar,
        )  # fmt: skip
        kept = np.flatnonzero(placed)
        if not len(kept):
            raise ValueError("seeded_laguerre: no seed could be placed (a larger box)")
        seeds_kept = order[kept]
        phase = np.array([f.phase for f in families], dtype=np.int64)[kind[seeds_kept]]
        # the grains: a crystal per seed, then one region per amorphous phase
        merged = np.isin(phase, list(self.amorphous_phases or ()))
        regions = np.unique(phase[merged])
        n_crystals = int((~merged).sum())
        grain_of = np.empty(len(kept), dtype=np.int64)
        grain_of[~merged] = np.arange(n_crystals)
        grain_of[merged] = n_crystals + np.searchsorted(regions, phase[merged])
        n_grains = n_crystals + len(regions)
        grain_ids = np.repeat(grain_of, np.diff(first)[kept])
        rows = np.concatenate([np.arange(first[s], first[s + 1]) for s in kept])
        germs = positions[rows] + window.lower
        weights = radii[rows] ** 2
        grain_phases = np.empty(n_grains, dtype=np.uint8)
        grain_phases[grain_of] = phase
        crystals = None
        if self.crystal_orientations:  # the amorphous regions keep the identity
            crystals = np.zeros((n_grains, 3))
            if self.texture is not None:
                crystals[:n_crystals] = fibre_orientations(rng, n_crystals, self.texture)
            else:
                crystals[:n_crystals] = random_orientations(rng, n_crystals)

        def by_grain(values):  # the seed of each crystal; NaN for the amorphous regions
            out = np.full((n_grains, 3), np.nan)
            out[grain_of[~merged]] = values[~merged]
            return out

        seed_axes = axes[seeds_kept].copy()
        if planar:  # ellipses: no third semi-axis
            seed_axes[:, 2] = np.nan
        seeds = {"centres": by_grain(centres[kept] + window.lower),
                 "axes": by_grain(seed_axes),
                 "angles": by_grain(angles[seeds_kept])}  # fmt: skip
        metadata = {"overlap": tolerance, "seeds_drawn": n, "seeds_left_out": n - len(kept),
                    "seeds_crowded": int(crowded.sum())}  # fmt: skip
        if planar:
            metadata["dimension"] = 2
        if len(regions):
            metadata["amorphous_phases"] = [int(k) for k in regions]
        if math.isfinite(margin):
            metadata["germ_margin"] = margin
        return SeededTessellation(domain, germs, weights, grain_ids, grain_phases, crystals,
                                  seeds, model=self, seed=seed_info, metadata=metadata)  # fmt: skip

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """The mean volume of the seeds and that of the crystals (the grains of the seeds of
        the phases that are not amorphous, where the seeds fill the box; their areas in the
        plane), the number of grains it implies in the domain (the crystals and one region
        per amorphous phase); the fractions of the phases."""
        families = self.families()
        amorphous = set(self.amorphous_phases or ())
        crystals = [f for f in families if f.phase not in amorphous]
        size = "area" if self.dimension == 2 else "volume"
        out = {f"mean_seed_{size}": self.mean_seed_volume()}
        if crystals:
            density = sum(f.fraction / f.mean_volume() for f in crystals)
            out[f"mean_grain_{size}"] = sum(f.fraction for f in crystals) / density
        if domain is not None:
            regions = len(families) - len(crystals)
            room = domain.volume if self.dimension == 3 else float(np.prod(domain.lengths[:2]))
            out["expected_grains"] = (room * sum(f.fraction / f.mean_volume()
                                                 for f in crystals) + regions)  # fmt: skip
        if self.phase_fractions is not None:
            for i, p in enumerate(self.phase_fractions):
                out[f"volume_fraction_{i}"] = float(p)
        return out


__all__ = ["CROWDED", "MOST_SPHERES", "SEED_ORIENTATIONS", "SeedFamily", "SeededLaguerre",
           "SeededTessellation", "ellipsoid_spheres", "overlap_tolerance"]  # fmt: skip
