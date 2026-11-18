"""Nested models: structures placed in, or around, the phases of others.

* ``cox_spheres``: spheres clustered in the phase 1 of another model (aggregates of
  particles in nanocomposites, at any number of scales): a Cox (doubly stochastic)
  Boolean model, or hard-core spheres placed among its germs.
* ``coating``: a layer of constant thickness around the phase 1 of a model (an
  interphase), or a shell inside it (hollow members: spheres, fibres, struts).
* ``fractal_sponge``: the attractor of an iterated function system on a cubic grid (the
  Menger sponge, Cantor dust, the Vicsek fractal, or any set of sub-cubes), iterated a
  finite number of times.

Hierarchical lattices of hollow tubes are in :mod:`morphogen.models.hierarchical`, and
the ``nested`` operation of :mod:`morphogen.models.combination` fills the phase 1 of a
structure with another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy import ndimage

from morphogen import theory
from morphogen._utils import check_fraction, check_positive
from morphogen.distributions import Constant, Distribution
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.kernels.rsa import rsa_insert
from morphogen.models.affine import ANISOTROPY, Anisotropy, anisotropic
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.models.implicit import band_of
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments, Spheres
from morphogen.seeding import SeedLike, make_rng, seed_from_description


def _child_seeds(rng, seed_info, n: int) -> list:
    """Seeds of the parts of a nested realization, derived from its own seed."""
    if seed_info is not None:
        return list(seed_from_description(seed_info).spawn(n))
    return list(rng.spawn(n))


def _supersampled(fine: np.ndarray, grid: Grid, ss: int) -> np.ndarray:
    nx, ny, nz = grid.shape
    return fine.reshape(nx, ss, ny, ss, nz, ss).mean(axis=(1, 3, 5), dtype=np.float32)


# -- clustered spheres ----------------------------------------------------------------------


@register_kind
class ClusteredSpheres(PrimitiveUnion):
    """Spheres (phase 1) clustered in aggregates, a realization of another model; the
    aggregates outside the spheres may form a phase of their own."""

    kind = "clustered_spheres"

    def __init__(self, domain, spheres, aggregates: Microstructure, aggregate_phase=None, **kw):
        super().__init__(domain, spheres=spheres, **kw)
        self.aggregates = aggregates
        self.aggregate_phase = None if aggregate_phase is None else int(aggregate_phase)

    @property
    def phase_names(self) -> dict[int, str]:
        return self.model.active_phases() if self.model is not None else super().phase_names

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        image = super().voxelize(
            shape, voxel_size=voxel_size, grid=grid, supersampling=supersampling
        )
        if self.aggregate_phase is None or supersampling != 1:
            return image
        inside = self.aggregates.voxelize(grid=image.grid()).data == 1
        data = image.data.copy()
        data[(data == 0) & inside] = self.aggregate_phase
        return image.with_data(data)

    def implicit_phases(self) -> list[int]:
        phases = {0, 1}
        if self.aggregate_phase is not None:
            phases.add(self.aggregate_phase)
        return sorted(phases)

    def phase_level_set(
        self, phase: int, shape=None, *, voxel_size=None, grid: Grid | None = None
    ) -> VoxelImage:
        """The implicit function of a phase (see :meth:`Microstructure.phase_level_set`):
        the spheres by their signed distance; the aggregates, when a phase of their own,
        by the level set of their structure outside the spheres."""
        phase = int(phase)
        if phase not in self.implicit_phases():
            raise ValueError(f"phase {phase}: the phases are {self.implicit_phases()}")
        if self.aggregate_phase in (None, 0):
            return Microstructure.phase_level_set(self, phase, shape, voxel_size=voxel_size,
                                                  grid=grid)  # fmt: skip
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        spheres = np.asarray(self.level_set(grid=grid).data, dtype=np.float32)
        zone = np.asarray(self.aggregates.level_set(grid=grid).data, dtype=np.float32)
        if self.aggregate_phase == 1:  # one phase with the spheres
            data = np.minimum(spheres, zone)
            data = data if phase == 1 else -data
        elif phase == 1:
            data = spheres
        elif phase == self.aggregate_phase:
            data = np.maximum(zone, -spheres)
        else:
            data = -np.minimum(spheres, zone)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def phase_level_sets_at(self, points, reach: float = np.inf) -> dict[int, np.ndarray]:
        """The functions of :meth:`phase_level_set` at the points: the signed distance to
        the spheres, and the point-wise level set of the aggregates when they form a phase
        of their own."""
        if self.aggregate_phase in (None, 0):
            phi = PrimitiveUnion.level_set_at(self, points, reach)
            return {0: -phi, 1: phi}
        spheres = PrimitiveUnion.level_set_at(self, points, reach)
        zone = np.asarray(self.aggregates.level_set_at(points, reach), dtype=float)
        if self.aggregate_phase == 1:  # one phase with the spheres
            phi = np.minimum(spheres, zone)
            return {0: -phi, 1: phi}
        return {
            0: -np.minimum(spheres, zone),
            1: spheres,
            self.aggregate_phase: np.maximum(zone, -spheres),
        }

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """The function of phase 1 (:meth:`phase_level_sets_at`)."""
        return self.phase_level_sets_at(points, reach)[1]

    def implicit_pieces(self, depth: float = 0.0) -> tuple[list, int] | None:
        """The pieces of the spheres (:meth:`PrimitiveUnion.implicit_pieces`), when the
        aggregates are not a phase of their own nor of the spheres; None otherwise."""
        if self.aggregate_phase not in (None, 0):
            return None
        return super().implicit_pieces(depth)

    def summary(self) -> dict:
        out = super().summary()
        out.update({f"aggregates_{k}": v for k, v in self.aggregates.summary().items()})
        return out

    def _geometry(self) -> dict:
        geo = super()._geometry()
        geo["aggregates"] = self.aggregates.to_dict()
        if self.aggregate_phase is not None:
            geo["aggregate_phase"] = self.aggregate_phase
        return geo

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        spheres = Spheres.from_dict(geometry["spheres"]) if "spheres" in geometry else None
        aggregates = Microstructure.from_dict(geometry["aggregates"])
        return cls(domain, spheres, aggregates, geometry.get("aggregate_phase"), **kwargs)


@register
@anisotropic
@dataclass(frozen=True)
class CoxSpheres(Model):
    """Spheres clustered in the phase 1 of another model: a Cox Boolean model.

    The centres of the spheres are Poisson points of intensity ``theta_1`` in the
    aggregates (the phase 1 of ``aggregates``, any model, Cox spheres included) and
    ``theta_0`` outside, the spheres being whole (Jeulin & Moreaud 2007; Jean, Jeulin,
    Forest et al. 2011 for carbon black in rubber). With ``hardcore``, non-overlapping
    spheres are placed among such germs by random sequential addition instead. The
    fractions are local: that of the spheres within the aggregates and outside them.
    """

    name = "cox_spheres"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "spheres"}
    count_label = "spheres"

    aggregates: Model = param(
        doc="Model whose phase 1 holds the spheres (aggregates)", kind="model"
    )
    radius: Distribution = param(
        Constant(1.0), doc="Sphere radius (number or distribution)", kind="distribution",
        unit="length",
    )  # fmt: skip
    local_fraction: float | None = param(
        None, doc="Volume fraction of the spheres in the aggregates", kind="fraction"
    )
    intensity: float | None = param(
        None, doc="Number of sphere centres per unit volume of the aggregates", unit="1/length^3"
    )
    outside_fraction: float = param(
        0.0, doc="Volume fraction of the spheres outside the aggregates", kind="fraction"
    )
    hardcore: bool = param(
        False, doc="Non-overlapping spheres (random sequential addition)", kind="bool"
    )
    min_gap: float = param(0.0, doc="Hard-core spheres: minimal gap between them", unit="length")
    aggregate_phase: int | None = param(
        None, doc="Phase of the aggregates outside the spheres (default: the matrix)", kind="int"
    )
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        if (self.local_fraction is None) == (self.intensity is None):
            raise ValueError("give exactly one of 'local_fraction' and 'intensity'")
        if self.local_fraction is not None:
            check_fraction(self.local_fraction, "local_fraction")
        else:
            check_positive(self.intensity, "intensity")
            if self.hardcore:
                raise ValueError("hard-core spheres: give their local_fraction")
        if not 0.0 <= self.outside_fraction < 1.0:
            raise ValueError("outside_fraction: in [0, 1)")
        check_positive(self.min_gap, "min_gap", allow_zero=True)
        if self.aggregate_phase is not None and not 2 <= self.aggregate_phase <= 255:
            raise ValueError("aggregate_phase: a phase index from 2 to 255")

    @property
    def has_level_set(self) -> bool:
        return self.aggregate_phase is None

    @property
    def supports_periodic(self) -> bool:
        return self.aggregates.supports_periodic

    def active_phases(self) -> dict[int, str]:
        out = dict(self.phases)
        if self.aggregate_phase is not None:
            out[self.aggregate_phase] = f"aggregates ({self.aggregates.name})"
        return out

    @property
    def mean_volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius.moment(3)

    @property
    def intensities(self) -> tuple[float, float]:
        """Intensities of the germs in and out of the aggregates (Boolean spheres)."""
        inside = (
            self.intensity
            if self.intensity is not None
            else theory.boolean_intensity(self.local_fraction, self.mean_volume)
        )
        outside = theory.boolean_intensity(self.outside_fraction, self.mean_volume)
        return inside, outside

    def generate(self, domain: Domain, seed: SeedLike = None) -> ClusteredSpheres:
        self._check_domain(domain)
        rng, seed_info = make_rng(seed)
        seeds = _child_seeds(rng, seed_info, 2)
        rmax = self.radius.upper()
        reach = (2.0 * rmax + self.min_gap) if self.hardcore else rmax
        if domain.periodic:
            if 2.0 * reach >= min(domain.size):
                raise ValueError("cox_spheres: spheres too large for the periodic box")
            window = domain
        else:
            window = domain.dilated(reach)
        aggregates = self.aggregates.generate(window, seeds[0])
        local = np.random.default_rng(seeds[1])
        if self.hardcore:
            centers, radii = self._hardcore(window, aggregates, local)
        else:
            inside, outside = self.intensities
            top = max(inside, outside)
            n = local.poisson(top * window.volume)
            centers = window.lower + local.random((n, 3)) * window.lengths
            rate = np.where(aggregates.contains(centers), inside, outside)
            centers = centers[local.random(n) * top < rate]
            radii = self.radius.sample(local, len(centers))
        spheres = Spheres(centers, radii)
        if not domain.periodic:
            spheres = spheres.subset(domain.distance_to_box(spheres.centers) <= spheres.radii)
        return ClusteredSpheres(
            domain, spheres, aggregates, self.aggregate_phase, model=self, seed=seed_info
        )

    def _hardcore(self, window: Domain, aggregates: Microstructure, rng):
        """Random sequential addition among the germs of the Cox process (a periodic box,
        cropped later for non-periodic domains)."""
        probe = window.lower + rng.random((20000, 3)) * window.lengths
        p_in = float(aggregates.contains(probe).mean())
        target = window.volume * (p_in * self.local_fraction + (1 - p_in) * self.outside_fraction)
        radii = np.empty(0)
        while np.sum(4.0 / 3.0 * math.pi * radii**3) < target:
            more = self.radius.sample(rng, int(0.2 * target / self.mean_volume) + 16)
            radii = np.concatenate([radii, more])
        if radii.size == 0:
            return np.zeros((0, 3)), radii
        cum = np.cumsum(4.0 / 3.0 * math.pi * radii**3)
        radii = np.sort(radii[: int(np.searchsorted(cum, target)) + 1])[::-1].copy()
        top = max(self.local_fraction, self.outside_fraction)
        box = window.lengths
        reach = 2.0 * radii.max() + self.min_gap if radii.size else 1.0
        cell_n = np.maximum(1, np.floor(box / reach)).astype(np.int64)
        cell_h = box / cell_n
        head = np.full(int(np.prod(cell_n)), -1, dtype=np.int64)
        nxt = np.full(len(radii), -1, dtype=np.int64)
        pos = np.zeros((len(radii), 3))
        placed, attempts, budget = 0, 0, 20_000 * max(len(radii), 1)
        while placed < len(radii) and attempts < budget:
            cand = rng.random((8192, 3)) * box
            rate = np.where(
                aggregates.contains(window.lower + cand), self.local_fraction, self.outside_fraction
            )
            cand = np.ascontiguousarray(cand[rng.random(len(cand)) * top < rate])
            placed, _ = rsa_insert(
                radii, float(self.min_gap), box, cell_h, cell_n, head, nxt, pos, placed, cand
            )
            attempts += 8192
        if placed < len(radii):
            raise RuntimeError(
                f"cox_spheres: {placed} of {len(radii)} hard-core spheres placed; lower the "
                "local fraction (random sequential addition jams near 0.38 for equal spheres)"
            )
        return window.lower + pos, radii

    def expected_count(self, domain: Domain) -> float | None:
        """Mean number of sphere centres in the domain (Boolean spheres)."""
        try:
            p = self.aggregates.theory(domain)["volume_fraction"]
        except (KeyError, ValueError, NotImplementedError):
            return None
        if self.hardcore:
            return (
                domain.volume
                * (p * self.local_fraction + (1 - p) * self.outside_fraction)
                / (self.mean_volume)
            )
        inside, outside = self.intensities
        return (inside * p + outside * (1.0 - p)) * domain.volume

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """Fractions when the spheres are small compared with the aggregates (a sphere
        rarely straddles their boundary): ``1 - p exp(-theta_1 V) - (1 - p) exp(-theta_0
        V)`` for Boolean spheres, ``p f_1 + (1 - p) f_0`` for hard-core ones, ``p`` being
        the fraction of the aggregates."""
        try:
            p = self.aggregates.theory(domain).get("volume_fraction")
        except (ValueError, NotImplementedError):
            p = None
        inside, outside = self.intensities
        out = {"intensity_inside": inside, "intensity_outside": outside}
        if p is None:
            return out
        out["aggregate_fraction"] = p
        if self.hardcore:
            out["volume_fraction"] = p * self.local_fraction + (1 - p) * self.outside_fraction
            return out
        v = self.mean_volume
        q_in, q_out = math.exp(-inside * v), math.exp(-outside * v)
        out["volume_fraction"] = 1.0 - p * q_in - (1.0 - p) * q_out
        area = 4.0 * math.pi * self.radius.moment(2)
        out["specific_surface"] = area * (inside * p * q_in + outside * (1.0 - p) * q_out)
        return out


# -- coatings and shells ---------------------------------------------------------------------


def _signed_distance(inner: Microstructure, grid: Grid, band: float) -> tuple:
    """Phases of a structure on a grid and the signed distance to its phase 1 (negative
    inside): exact for unions of spheres and segments, from the Euclidean distance
    transform of the voxels otherwise."""
    phases = inner.voxelize(grid=grid).data
    if isinstance(inner, PrimitiveUnion):
        return phases, inner.level_set(grid=grid, band=band).data.astype(np.float64)
    solid = phases == 1
    h = grid.spacing
    wrap = "wrap" if inner.domain.periodic else "nearest"
    outside = _edt(~solid, h, wrap, band)
    inside = _edt(solid, h, wrap, band)
    half = 0.5 * float(np.min(h))  # from voxel centres to the interface between them
    return phases, np.where(solid, -(inside - half), outside - half)


def _edt(mask: np.ndarray, spacing, mode: str, band: float = math.inf) -> np.ndarray:
    """Distance from the voxels of ``mask`` to the nearest voxel outside it, exact up to
    ``band`` and at least ``band`` beyond (periodic: on a copy wrapped over the band)."""
    if not mask.any():
        return np.zeros(mask.shape)
    if mask.all():
        return np.full(mask.shape, np.inf)
    if mode == "wrap":
        h = np.broadcast_to(np.asarray(spacing, dtype=float), (mask.ndim,))
        pad = [min(n, math.ceil(band / s) + 1) for n, s in zip(mask.shape, h, strict=True)]
        tiled = np.pad(mask, [(p, p) for p in pad], mode="wrap")
        d = ndimage.distance_transform_edt(tiled, sampling=spacing)
        return d[tuple(slice(p, p + n) for p, n in zip(pad, mask.shape, strict=True))]
    return ndimage.distance_transform_edt(mask, sampling=spacing)


@register_kind
class CoatedStructure(Microstructure):
    """A structure with a layer around its phase 1 (outside) or a shell inside it."""

    kind = "coated"

    def __init__(self, domain, inner, thickness, side, phase, core_phase, bores=None, **kw):
        super().__init__(domain, **kw)
        self.inner = inner
        self.thickness = float(thickness)
        self.side = side
        self.phase = int(phase)
        self.core_phase = int(core_phase)
        self.bores = bores  # hollow unions of spheres and segments: the same, thinner

    @property
    def phase_names(self) -> dict[int, str]:
        return self.model.active_phases() if self.model is not None else super().phase_names

    def _phases_on(self, grid: Grid) -> np.ndarray:
        if self.bores is not None:  # union of the members minus the union of their bores
            outer = self.inner.voxelize(grid=grid).data == 1
            bore = self.bores.voxelize(grid=grid).data == 1
            out = np.zeros(grid.shape, dtype=np.uint8)
            out[outer] = 1
            out[outer & bore] = self.core_phase
            return out
        band = self.thickness + 2.0 * float(np.max(grid.spacing))
        phases, dist = _signed_distance(self.inner, grid, band)
        out = phases.astype(np.uint8, copy=True)
        if self.side == "outside":
            out[(phases == 0) & (dist <= self.thickness)] = self.phase
        else:
            out[(phases == 1) & (dist < -self.thickness)] = self.core_phase
        return out

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if ss == 1:
            data = self._phases_on(grid)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        data = _supersampled((self._phases_on(fine) == 1).astype(np.float32), grid, ss)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Hollow members (a void core): the union minus the bores (exact sets), or the
        shell ``-t <= d <= 0`` of the signed distance; a coating leaves phase 1 as is."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        if self.side == "outside":
            return self.inner.level_set(grid=grid)
        if self.core_phase != 0:
            raise NotImplementedError("a shell with a filled core has three phases")
        if self.bores is not None:
            outer = self.inner.level_set(grid=grid).data.astype(np.float32)
            bore = self.bores.level_set(grid=grid).data.astype(np.float32)
            data = np.maximum(outer, -bore)
        else:
            band = self.thickness + 4.0 * float(np.max(grid.spacing))
            _, dist = _signed_distance(self.inner, grid, band)
            data = np.maximum(dist, -dist - self.thickness).astype(np.float32)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def implicit_phases(self) -> list[int]:
        if self.side == "outside":
            return sorted({0, 1, self.phase})
        return sorted({0, 1, self.core_phase})

    def phase_level_set(
        self, phase: int, shape=None, *, voxel_size=None, grid: Grid | None = None
    ) -> VoxelImage:
        """The implicit function of a phase (see :meth:`Microstructure.phase_level_set`),
        from the signed distance ``d`` to the phase 1 of the structure: a coating of
        thickness ``t`` is ``0 < d < t``, a shell ``-t < d < 0`` and its core ``d < -t``
        (hollow unions of spheres and segments: the union minus the bores, and the
        bores). The distance is exact for unions of spheres and segments; for the other
        structures it is that of their voxels on the grid (as in their images)."""
        phase = int(phase)
        if phase not in self.implicit_phases():
            raise ValueError(f"phase {phase}: the phases are {self.implicit_phases()}")
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        clip = band_of(grid)
        t = self.thickness
        if self.bores is not None:
            outer = np.asarray(self.inner.level_set(grid=grid).data, dtype=np.float64)
            bore = np.asarray(self.bores.level_set(grid=grid).data, dtype=np.float64)
            regions = {1: np.maximum(outer, -bore), self.core_phase: np.maximum(outer, bore)}
            outside = -outer
        else:
            _, d = _signed_distance(self.inner, grid, t + clip)
            if self.side == "outside":
                regions = {1: d, self.phase: np.maximum(-d, d - t)}
                outside = t - d
            else:
                regions = {1: np.maximum(d, -d - t), self.core_phase: d + t}
                outside = -d
        # the phases that share an index are one (a core of the phase of the matrix...)
        fields: dict[int, np.ndarray] = {0: outside}
        for k in (1, *(q for q in regions if q != 1)):
            fields[k] = np.minimum(fields[k], regions[k]) if k in fields else regions[k]
        data = np.clip(fields[phase], -clip, clip).astype(np.float32)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def _exact_distance(self) -> bool:
        """Whether the distance to the phase 1 of the structure is known at any point: a
        union of spheres and segments of two phases (the distance of the other structures
        is that of their voxels, :func:`_signed_distance`)."""
        return isinstance(self.inner, PrimitiveUnion) and self.inner.implicit_phases() == [0, 1]

    def phase_level_sets_at(self, points, reach: float = np.inf) -> dict[int, np.ndarray]:
        """The functions of :meth:`phase_level_set` at the points, for the coatings and
        shells of a union of spheres and segments (whose signed distance is exact
        there): the same functions of that distance, or of those of the union and of its
        bores. The coatings of the other structures, whose distance is that of their
        voxels, have none."""
        if not self._exact_distance():
            raise NotImplementedError(
                "the coatings and shells of a structure that is not a union of spheres and "
                "segments: the distance to its phase 1 is that of its voxels"
            )
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        t = self.thickness
        if not np.isfinite(reach):
            reach = 0.1 * float(self.domain.lengths.min()) + t
        reach = float(reach)
        if self.bores is not None:
            outer = self.inner.signed_distance(pts, reach)
            bore = self.bores.signed_distance(pts, reach)
            regions = {1: np.maximum(outer, -bore), self.core_phase: np.maximum(outer, bore)}
            outside = -outer
        else:
            d = self.inner.signed_distance(pts, reach + t)
            if self.side == "outside":
                regions = {1: d, self.phase: np.maximum(-d, d - t)}
                outside = t - d
            else:
                regions = {1: np.maximum(d, -d - t), self.core_phase: d + t}
                outside = -d
        # the phases that share an index are one (a core of the phase of the matrix...)
        fields: dict[int, np.ndarray] = {0: outside}
        for k in (1, *(q for q in regions if q != 1)):
            fields[k] = np.minimum(fields[k], regions[k]) if k in fields else regions[k]
        return {k: np.clip(fields[k], -reach, reach) for k in self.implicit_phases()}

    def implicit_pieces(self, depth: float = 0.0) -> tuple[list, int] | None:
        """The pieces of a coated union of spheres and segments, or of its hollow
        members (see :mod:`morphogen.models.pieces`): the primitives, then the same with
        their radii increased by the thickness of the coating; the bores of hollow
        members, then the primitives. None for the other structures."""
        from morphogen.models.pieces import Piece, union_pieces

        inner = self.inner
        if not isinstance(inner, PrimitiveUnion) or inner.implicit_phases() != [0, 1]:
            return None
        if self.side == "outside":
            pieces = [Piece(1, part) for part in union_pieces(inner, 0.0, depth)]
            pieces += [
                Piece(self.phase, part) for part in union_pieces(inner, self.thickness, depth)
            ]
            return pieces, 0
        if self.bores is None:
            return None
        pieces = [Piece(self.core_phase, part) for part in union_pieces(self.bores, 0.0, depth)]
        pieces += [Piece(1, part) for part in union_pieces(inner, 0.0, depth)]
        return pieces, 0

    @property
    def lipschitz(self) -> float:
        parts = [self.inner] + ([self.bores] if self.bores is not None else [])
        return max(float(getattr(p, "lipschitz", 1.0)) for p in parts)

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """As :meth:`level_set`, at the points: the level set of the structure (coating),
        or the function of phase 1 of :meth:`phase_level_sets_at` (the members of a union
        less their bores, the shell of its signed distance)."""
        if self.side == "outside":
            return self.inner.level_set_at(points, reach)
        return self.phase_level_sets_at(points, reach)[1]

    def contains(self, points) -> np.ndarray:
        """Phase 1: that of the structure (coating), or its shell (hollow members of
        unions of spheres and segments)."""
        if self.side == "outside":
            return self.inner.contains(points)
        if self.bores is None:
            raise NotImplementedError("point queries of shells need unions of spheres or segments")
        inside = self.inner.contains(points)
        return inside & ~self.bores.contains(points) if self.core_phase != 1 else inside

    def summary(self) -> dict:
        return {f"inner_{k}": v for k, v in self.inner.summary().items()}

    def _geometry(self) -> dict:
        geo = {
            "inner": self.inner.to_dict(),
            "thickness": self.thickness,
            "side": self.side,
            "phase": self.phase,
            "core_phase": self.core_phase,
        }
        if self.bores is not None:
            geo["bores"] = self.bores.to_dict()
        return geo

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        bores = geometry.get("bores")
        return cls(
            domain,
            Microstructure.from_dict(geometry["inner"]),
            geometry["thickness"],
            geometry["side"],
            geometry["phase"],
            geometry["core_phase"],
            None if bores is None else Microstructure.from_dict(bores),
            **kwargs,
        )


def bores_of(union: PrimitiveUnion, thickness: float) -> PrimitiveUnion:
    """The same spheres and segments, thinner by ``thickness`` (those thinner than it
    are left out: they stay full)."""
    sph, seg = union.spheres, union.segments
    keep_s = sph.radii > thickness
    keep_g = seg.radii > thickness
    spheres = Spheres(sph.centers[keep_s], sph.radii[keep_s] - thickness)
    segments = Segments(
        seg.start[keep_g], seg.end[keep_g], seg.radii[keep_g] - thickness,
        round_caps=seg.round_caps,
    )  # fmt: skip
    return PrimitiveUnion(union.domain, spheres=spheres, segments=segments)


@register
@dataclass(frozen=True)
class Coating(Model):
    """A layer of constant thickness on the phase 1 of a model.

    ``outside``: a coating (an interphase) of phase ``phase`` around it, within the
    distance ``thickness``. ``inside``: a shell of that thickness, the core becoming the
    phase ``core_phase`` (0: hollow members). Unions of spheres and segments (Boolean
    and hard-core spheres, fibres, strut lattices...) are coated exactly, and their
    members are hollowed one by one (each minus a thinner copy: bores join where members
    meet); other structures through the distance transform of their voxels.
    """

    name = "coating"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "phase 1 of the model", 2: "coating"}

    model: Model = param(doc="Model whose phase 1 is coated or hollowed", kind="model")
    thickness: float = param(doc="Thickness of the layer", unit="length")
    side: str = param(
        "outside", doc="outside (a coating around phase 1) or inside (a shell)", kind="str"
    )
    phase: int = param(2, doc="Phase of the coating (outside)", kind="int")
    core_phase: int = param(
        0, doc="Phase of the core within the shell (inside; 0: hollow)", kind="int"
    )

    def _validate(self) -> None:
        check_positive(self.thickness, "thickness")
        if self.side not in ("outside", "inside"):
            raise ValueError("side: 'outside' (a coating) or 'inside' (a shell)")
        if not 2 <= self.phase <= 255:
            raise ValueError("phase: a phase index from 2 to 255")
        if self.core_phase == 1 or not 0 <= self.core_phase <= 255:
            raise ValueError("core_phase: 0 (hollow) or a phase index from 2 to 255")

    @property
    def has_level_set(self) -> bool:  # two phases: hollow members
        return self.side == "inside" and self.core_phase == 0 and self.model.has_level_set

    @property
    def supports_periodic(self) -> bool:
        return self.model.supports_periodic

    def active_phases(self) -> dict[int, str]:
        if self.side == "outside":
            return {0: "matrix", 1: f"phase 1 ({self.model.name})", self.phase: "coating"}
        out = {0: "matrix", 1: "shell"}
        if self.core_phase != 0:
            out[self.core_phase] = "core"
        return out

    def generate(self, domain: Domain, seed: SeedLike = None) -> CoatedStructure:
        self._check_domain(domain)
        rng, seed_info = make_rng(seed)
        inner = self.model.generate(domain, _child_seeds(rng, seed_info, 1)[0])
        bores = None
        if self.side == "inside" and isinstance(inner, PrimitiveUnion):
            bores = bores_of(inner, self.thickness)
        return CoatedStructure(
            domain,
            inner,
            self.thickness,
            self.side,
            self.phase,
            self.core_phase,
            bores,
            model=self,
            seed=seed_info,
        )

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """Boolean spheres: their coating is the dilation of a Boolean model, the Boolean
        model of the spheres of radii ``R + t``."""
        from morphogen.models.boolean import BooleanSpheres

        m = self.model
        if not isinstance(m, BooleanSpheres) or self.side != "outside":
            return {}
        theta, t = m.theta, self.thickness
        nodes, weights = m.radius.quadrature(64)
        dilated = float(np.sum(weights * 4.0 / 3.0 * math.pi * (nodes + t) ** 3))
        p = 1.0 - math.exp(-theta * m.mean_grain_volume)
        return {
            "volume_fraction": p,
            "coating_fraction": 1.0 - math.exp(-theta * dilated) - p,
        }


# -- fractal sponges ---------------------------------------------------------------------------

#: Sub-cubes kept by the classical patterns of a 3 x 3 x 3 subdivision.
_CENTRE_AND_FACES = {(1, 1, 1), (0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)}
PATTERNS = {
    "menger": sorted(set(np.ndindex(3, 3, 3)) - _CENTRE_AND_FACES),
    "vicsek": sorted(_CENTRE_AND_FACES),
    "cantor_dust": sorted(c for c in np.ndindex(3, 3, 3) if 1 not in c),
}


@register_kind
class FractalStructure(Microstructure):
    """Periodic tiling of the cube kept by an iterated function system."""

    kind = "fractal_sponge"

    def __init__(self, domain, cell_size, subdivisions, kept, level, **kw):
        super().__init__(domain, **kw)
        self.cell_size = float(cell_size)
        self.subdivisions = int(subdivisions)
        self.kept = [tuple(int(v) for v in c) for c in kept]
        self.level = int(level)
        k = self.subdivisions
        self._mask = np.zeros(k**3, dtype=bool)
        for i, j, m in self.kept:
            self._mask[(i * k + j) * k + m] = True

    def contains(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        k = self.subdivisions
        u = np.mod((pts - self.domain.lower) / self.cell_size, 1.0)
        inside = np.ones(len(pts), dtype=bool)
        for _ in range(self.level):
            d = np.minimum((u * k).astype(np.int64), k - 1)
            inside &= self._mask[(d[:, 0] * k + d[:, 1]) * k + d[:, 2]]
            u = u * k - d
        return inside

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        fine = grid if ss == 1 else Grid(grid.domain, tuple(n * ss for n in grid.shape))
        data = np.empty(fine.shape, dtype=bool)
        for k0 in range(0, fine.shape[2], 16):
            k1 = min(fine.shape[2], k0 + 16)
            data[:, :, k0:k1] = self.contains(fine.center_points(k0, k1)).reshape(
                fine.shape[0], fine.shape[1], k1 - k0
            )
        if ss == 1:
            return VoxelImage.from_grid(
                data.astype(np.uint8), grid, name="phase", metadata=self.image_metadata()
            )
        frac = _supersampled(data.astype(np.float32), grid, ss)
        return VoxelImage.from_grid(frac, grid, name="fraction", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """The signed distance to the solid in the maximum norm (negative inside),
        clipped to a band of four voxels and to the edge of the smallest cubes: the
        solid is a union of such cubes, and the distance from a point to the nearest
        cube of the other state among the 26 around its own is the largest of its
        distances to the faces that separate them."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        k = self.subdivisions
        h = self.cell_size / k**self.level
        band = min(h, band_of(grid))
        nx, ny, nz = grid.shape
        out = np.empty(grid.shape, dtype=np.float32)
        offsets = [o for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]
        step = max(1, (1 << 18) // (nx * ny))
        for k0 in range(0, nz, step):
            k1 = min(nz, k0 + step)
            pts = grid.center_points(k0, k1)
            u = (pts - self.domain.lower) / h
            cell = np.floor(u)
            frac = u - cell
            centre = (cell + 0.5) * h + self.domain.lower
            own = self.contains(centre)
            dist = np.full(len(pts), band)
            for o in offsets:
                shift = np.array(o, dtype=float) - 1.0
                other = self.contains(centre + shift * h) != own
                if not other.any():
                    continue
                # to the faces towards that cube, along the axes where it is shifted
                gaps = np.where(shift > 0, 1.0 - frac, frac) * h
                d = np.where(shift != 0, gaps, 0.0).max(axis=1)
                dist = np.where(other, np.minimum(dist, d), dist)
            out[:, :, k0:k1] = np.where(own, -dist, dist).reshape(nx, ny, k1 - k0)
        return VoxelImage.from_grid(out, grid, name="level_set", metadata=self.image_metadata())

    def convex_pieces(self):
        """The structure as boxes of one phase each (the cubes of the finest level, merged
        greedily), clipped to the box: the pieces, their phases and their cells (the
        phases), for :func:`morphogen.mesh.exact.polyhedral_mesh`."""
        from morphogen.mesh.exact import grid_boxes

        step = self.cell_size / self.subdivisions**self.level
        lo, hi = self.domain.lower, self.domain.upper
        edges = []
        for d in range(3):
            n = int(np.ceil((hi[d] - lo[d]) / step - 1e-9))
            e = lo[d] + step * np.arange(n + 1)
            e[-1] = hi[d]
            edges.append(e)
        centres = [0.5 * (e[1:] + e[:-1]) for e in edges]
        grid = np.stack(np.meshgrid(*centres, indexing="ij"), axis=-1)
        labels = self.contains(grid.reshape(-1, 3)).reshape(grid.shape[:3]).astype(np.int64)
        pieces, phase = grid_boxes(labels, edges)
        return pieces, phase, phase.copy()

    def summary(self) -> dict:
        return {"kept_cubes": len(self.kept) ** self.level, "level": self.level}

    def _geometry(self) -> dict:
        return {
            "cell_size": self.cell_size,
            "subdivisions": self.subdivisions,
            "kept": [list(c) for c in self.kept],
            "level": self.level,
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            geometry["cell_size"],
            geometry["subdivisions"],
            geometry["kept"],
            geometry["level"],
            **kwargs,
        )


@register
@dataclass(frozen=True)
class FractalSponge(Model):
    """Iterated function system on a cubic grid: the Menger sponge and its kin.

    The cube of edge ``cell_size`` is divided into ``subdivisions^3`` sub-cubes, of which
    the pattern keeps some (``menger``: all but the centre and the face centres, 20 of
    27; ``vicsek``: the centre and the face centres; ``cantor_dust``: the 8 corners; or
    ``kept``: indices (i, j, k)); each kept sub-cube is divided alike, ``level`` times.
    The attractor has the similarity dimension ``log N / log k`` (N cubes kept of k^3);
    the domain is tiled by the cube.
    """

    name = "fractal_sponge"
    phases: ClassVar[dict[int, str]] = {0: "void", 1: "solid"}
    count_label = "unit cells"

    pattern: str = param("menger", doc="menger, vicsek or cantor_dust (or 'kept')", kind="str")
    kept: tuple | None = param(
        None, doc="Sub-cubes kept, as [i, j, k] indices (overrides the pattern)", kind="json"
    )
    subdivisions: int = param(3, doc="Sub-cubes per edge at each level", kind="int")
    level: int = param(3, doc="Number of iterations", kind="int")
    cell_size: float = param(1.0, doc="Edge of the cube of the sponge", unit="length")

    def _validate(self) -> None:
        check_positive(self.cell_size, "cell_size")
        if not 2 <= self.subdivisions <= 9:
            raise ValueError("subdivisions: from 2 to 9")
        if not 1 <= self.level <= 8:
            raise ValueError("level: from 1 to 8")
        if self.kept is None:
            if self.pattern not in PATTERNS:
                raise ValueError(f"pattern: one of {sorted(PATTERNS)}, or give 'kept'")
            if self.subdivisions != 3:
                raise ValueError("the named patterns divide the cube in 3 x 3 x 3")
        else:
            cubes = {tuple(int(v) for v in c) for c in self.kept}
            k = self.subdivisions
            if not cubes or any(len(c) != 3 or not all(0 <= v < k for v in c) for c in cubes):
                raise ValueError(f"kept: sub-cube indices [i, j, k] from 0 to {k - 1}")

    @property
    def cubes(self) -> list[tuple]:
        if self.kept is not None:
            return sorted({tuple(int(v) for v in c) for c in self.kept})
        return list(PATTERNS[self.pattern])

    def generate(self, domain: Domain, seed: SeedLike = None) -> FractalStructure:
        _, seed_info = make_rng(seed)
        if domain.periodic:
            ratio = domain.lengths / self.cell_size
            if np.any(np.abs(ratio - np.rint(ratio)) > 1e-6 * np.maximum(ratio, 1.0)):
                raise ValueError("fractal_sponge: a periodic domain holds whole cubes")
        return FractalStructure(
            domain,
            self.cell_size,
            self.subdivisions,
            self.cubes,
            self.level,
            model=self,
            seed=seed_info,
        )

    def expected_count(self, domain: Domain) -> float:
        return domain.volume / self.cell_size**3

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        n, k = len(self.cubes), self.subdivisions
        return {
            "volume_fraction": (n / k**3) ** self.level,
            "fractal_dimension": math.log(n) / math.log(k),
            "smallest_cube": self.cell_size / k**self.level,
        }
