"""Combinations of models: union, intersection, difference and complement of their
phase 1, and inclusions overlaid on a matrix (a polycrystal, or any structure).

The components are independent realizations of their models in the same domain, with
seeds derived from the seed of the combination. The phase 1 of a union is the union of
the phases 1 of the components, and so on; their level sets combine by minimum, maximum
and negation, so that combinations of models with level sets are meshed conformingly.
Inclusions keep the phases of the first component (the matrix: a polycrystal keeps its
cells, their labels and orientations) and give the phase 1 of each further component a
phase of its own, the next free index by default; later components cover earlier ones.

For independent stationary random sets ``A`` and ``B`` of volume fractions ``p`` and
specific surfaces ``S`` (Matheron 1975):

* union: ``p = 1 - (1 - pA) (1 - pB)``, ``S = SA (1 - pB) + SB (1 - pA)``;
* intersection: ``p = pA pB``, ``S = SA pB + SB pA``;
* complement: ``p = 1 - pA``, ``S = SA``.

A union of a Boolean model with another is the Boolean model of both grains; the
intersection of fine inclusions with a coarse random set gives clustered (Cox-like,
multiscale) inclusions, cut by the boundary of the set (``cox_spheres`` keeps them
whole). A nested combination fills the phase 1 of each model with the next one, whose
other phases are kept: hierarchical structures (a gyroid of octet trusses...), its phase
1 being the intersection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.seeding import SeedLike, make_rng, seed_from_description

OPERATIONS = ("union", "intersection", "difference", "complement", "inclusions", "nested")


def _set_operation(operation: str, parts: list[np.ndarray]) -> np.ndarray:
    """Phase 1 of a set operation on the phase-1 indicators of the components."""
    if operation == "union":
        return np.logical_or.reduce(parts)
    if operation in ("intersection", "nested"):
        return np.logical_and.reduce(parts)
    if operation == "difference":
        return parts[0] & ~np.logical_or.reduce(parts[1:])
    return ~parts[0]  # complement


@register_kind
class Combined(Microstructure):
    """Realization of a combination: its components and how they combine."""

    kind = "combination"

    def __init__(
        self,
        domain: Domain,
        operation: str,
        parts,
        inclusion_phases=None,
        **kwargs,
    ) -> None:
        super().__init__(domain, **kwargs)
        if operation not in OPERATIONS:
            raise ValueError(f"operation: one of {', '.join(OPERATIONS)}")
        self.operation = operation
        self.parts = list(parts)
        self.inclusion_phases = (
            None if inclusion_phases is None else [int(p) for p in inclusion_phases]
        )

    @property
    def matrix(self) -> Microstructure:
        return self.parts[0]

    # -- discretization ------------------------------------------------------------------

    def _phases_on(self, grid: Grid) -> np.ndarray:
        images = [p.voxelize(grid=grid).data for p in self.parts]
        if self.operation == "nested":  # each structure fills the phase 1 of the previous
            out = images[0].astype(np.uint8, copy=True)
            for img in images[1:]:
                inside = out == 1
                out[inside] = img[inside]
            return out
        if self.operation != "inclusions":
            return _set_operation(self.operation, [img == 1 for img in images]).astype(np.uint8)
        out = images[0].astype(np.uint8, copy=True)
        for img, phase in zip(images[1:], self.inclusion_phases, strict=True):
            out[img == 1] = phase
        return out

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if ss == 1:
            data = self._phases_on(grid)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        ph = (self._phases_on(fine) == 1).astype(np.float32)
        nx, ny, nz = grid.shape
        data = ph.reshape(nx, ss, ny, ss, nz, ss).mean(axis=(1, 3, 5), dtype=np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Minimum (union), maximum (intersection) or negation (complement) of the level
        sets of the components."""
        if self.operation == "inclusions":
            return super().level_set(shape, voxel_size=voxel_size, grid=grid)
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        fields = [p.level_set(grid=grid).data.astype(np.float32) for p in self.parts]
        if self.operation == "union":
            data = np.minimum.reduce(fields)
        elif self.operation in ("intersection", "nested"):
            data = np.maximum.reduce(fields)
        elif self.operation == "difference":
            data = np.maximum(fields[0], -np.minimum.reduce(fields[1:]))
        else:
            data = -fields[0]
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    def implicit_phases(self) -> list[int]:
        if self.operation == "inclusions":
            return sorted({*self.matrix.implicit_phases(), *self.inclusion_phases})
        if self.operation == "nested":  # the phases of each structure, in the phase 1 of the last
            phases = set(self.parts[-1].implicit_phases())
            for part in self.parts[:-1]:
                phases |= set(part.implicit_phases()) - {1}
            return sorted(phases)
        return [0, 1]

    def phase_level_set(
        self, phase: int, shape=None, *, voxel_size=None, grid: Grid | None = None
    ) -> VoxelImage:
        """The implicit function of a phase (see :meth:`Microstructure.phase_level_set`).
        Inclusions: a phase is what the matrix has of it outside all the inclusions,
        and each inclusion of that phase outside the later ones, which cover it. Nested
        structures: a phase is what each structure has of it inside the phase 1 of all
        the previous ones."""
        phase = int(phase)
        if self.operation not in ("inclusions", "nested"):
            return super().phase_level_set(phase, shape, voxel_size=voxel_size, grid=grid)
        if phase not in self.implicit_phases():
            raise ValueError(f"phase {phase}: the phases are {self.implicit_phases()}")
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)

        def of(part, k):
            if k not in part.implicit_phases():
                return None
            return np.asarray(part.phase_level_set(k, grid=grid).data, dtype=np.float32)

        data = None

        def add(field):
            nonlocal data
            if field is not None:
                data = field if data is None else np.minimum(data, field)

        if self.operation == "inclusions":
            covers = [of(part, 1) for part in self.parts[1:]]
            later = None  # the smallest function of the inclusions that cover the current one
            for cover, k in zip(covers[::-1], self.inclusion_phases[::-1], strict=True):
                if k == phase:
                    add(cover if later is None else np.maximum(cover, -later))
                later = cover if later is None else np.minimum(later, cover)
            field = of(self.matrix, phase)
            if field is not None:
                add(field if later is None else np.maximum(field, -later))
        else:
            inside = None  # the largest function of the phases 1 of the previous structures
            for i, part in enumerate(self.parts):
                last = i == len(self.parts) - 1
                if phase != 1 or last:
                    field = of(part, phase)
                    if field is not None:
                        add(field if inside is None else np.maximum(field, inside))
                if not last:
                    one = of(part, 1)
                    inside = one if inside is None else np.maximum(inside, one)
        return VoxelImage.from_grid(data, grid, name="level_set", metadata=self.image_metadata())

    @property
    def lipschitz(self) -> float:
        return max(float(getattr(p, "lipschitz", 1.0)) for p in self.parts)

    def phase_level_sets_at(self, points, reach: float = np.inf) -> dict[int, np.ndarray]:
        """The functions of :meth:`phase_level_set` at the points, from those of the
        components (:meth:`~morphogen.models.base.Microstructure.phase_level_sets_at`).
        Inclusions: the visible parts of the inclusions of a phase (each minus the later
        ones, ``max(psi_i, -min_{j > i} psi_j)``) and the phase of the matrix where no
        inclusion covers it; nested structures: a phase of each structure inside the
        phase 1 of all the previous ones; the others (two phases): that of
        :meth:`level_set_at`. A component without point-wise functions (a polycrystal
        matrix of one phase) leaves the combination without them."""
        if self.operation not in ("inclusions", "nested"):
            return super().phase_level_sets_at(points, reach)
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        values = [p.phase_level_sets_at(pts, reach) for p in self.parts]
        out: dict[int, np.ndarray] = {}

        def add(phase, field):
            if field is not None:
                out[phase] = field if phase not in out else np.minimum(out[phase], field)

        if self.operation == "inclusions":
            later = None  # the smallest function of the inclusions after the current one
            covers = [v[1] for v in values[1:]]
            for cover, k in zip(covers[::-1], self.inclusion_phases[::-1], strict=True):
                add(k, cover if later is None else np.maximum(cover, -later))
                later = cover if later is None else np.minimum(later, cover)
            for k, field in values[0].items():  # the matrix where no inclusion covers it
                add(k, field if later is None else np.maximum(field, -later))
        else:
            inside = None  # the largest function of the phases 1 of the previous structures
            for i, part in enumerate(values):
                last = i == len(values) - 1
                for k, field in part.items():
                    if k != 1 or last:
                        add(k, field if inside is None else np.maximum(field, inside))
                if not last:
                    inside = part[1] if inside is None else np.maximum(inside, part[1])
        return {k: out[k] for k in self.implicit_phases() if k in out}

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """The level sets of the components at the points, combined as :meth:`level_set`
        (inclusions: the function of phase 1, see :meth:`phase_level_sets_at`)."""
        if self.operation == "inclusions":
            if 1 in self.implicit_phases():
                return self.phase_level_sets_at(points, reach)[1]
            return super().level_set_at(points, reach)
        fields = [p.level_set_at(points, reach) for p in self.parts]
        if self.operation == "union":
            return np.minimum.reduce(fields)
        if self.operation in ("intersection", "nested"):
            return np.maximum.reduce(fields)
        if self.operation == "difference":
            return np.maximum(fields[0], -np.minimum.reduce(fields[1:]))
        return -fields[0]

    def contains(self, points) -> np.ndarray:
        inside = [p.contains(points) for p in self.parts]
        if self.operation != "inclusions":
            return _set_operation(self.operation, inside)
        out = inside[0].copy()  # phase 1 of the matrix, unless covered by an inclusion
        for covered, phase in zip(inside[1:], self.inclusion_phases, strict=True):
            out[covered] = phase == 1
        return out

    # -- grains of a polycrystal matrix ----------------------------------------------------

    @property
    def has_labels(self) -> bool:
        return self.operation == "inclusions" and self.matrix.has_labels

    def labels(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Cell labels of the matrix (the inclusions keep the label of their cell)."""
        if not self.has_labels:
            raise NotImplementedError("only inclusions in a tessellation have cell labels")
        return self.matrix.labels(shape, voxel_size=voxel_size, grid=grid)

    @property
    def orientations(self):
        return getattr(self.matrix, "orientations", None) if self.has_labels else None

    # -- description ------------------------------------------------------------------------

    def summary(self) -> dict:
        out: dict = {}
        for k, part in enumerate(self.parts):
            out.update({f"c{k}_{key}": value for key, value in part.summary().items()})
        return out

    def _geometry(self) -> dict:
        geo = {"operation": self.operation, "parts": [p.to_dict() for p in self.parts]}
        if self.inclusion_phases is not None:
            geo["inclusion_phases"] = self.inclusion_phases
        return geo

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        parts = [Microstructure.from_dict(d) for d in geometry["parts"]]
        return cls(domain, geometry["operation"], parts, geometry.get("inclusion_phases"), **kwargs)


@register
@dataclass(frozen=True)
class Combination(Model):
    """Union, intersection, difference or complement of models, or inclusions in a matrix."""

    name = "combination"
    phases: ClassVar[dict[int, str]] = {0: "outside", 1: "combined phase 1"}

    operation: str = param(
        "union",
        doc="union, intersection, difference (the first minus the others), complement "
        "(of a single model), inclusions (the phase 1 of the others over the first) or "
        "nested (each model fills the phase 1 of the previous one, with its phases)",
        kind="str",
    )
    models: tuple = param(
        (), doc="Models combined: mappings with a 'type' key and their parameters", kind="models"
    )
    inclusion_phases: tuple[int, ...] | None = param(
        None,
        doc="Inclusions: phase of each model after the first (default: the next free phases)",
        kind="ints",
    )

    def _validate(self) -> None:
        if self.operation not in OPERATIONS:
            raise ValueError(f"operation: one of {', '.join(OPERATIONS)}")
        if self.operation == "complement":
            if len(self.models) != 1:
                raise ValueError("models: the complement of exactly one model")
        elif len(self.models) < 2:
            raise ValueError(f"models: at least two models for the {self.operation}")
        if self.inclusion_phases is not None:
            if self.operation != "inclusions":
                raise ValueError("inclusion_phases: only for the 'inclusions' operation")
            if len(self.inclusion_phases) != len(self.models) - 1:
                raise ValueError("inclusion_phases: one phase per model after the first")
            if not all(0 <= p <= 255 for p in self.inclusion_phases):
                raise ValueError("inclusion_phases: phase indices from 0 to 255")

    # -- properties that depend on the components -------------------------------------------

    @property
    def supports_periodic(self) -> bool:
        return all(m.supports_periodic for m in self.models)

    @property
    def has_level_set(self) -> bool:
        return self.operation != "inclusions" and all(m.has_level_set for m in self.models)

    @property
    def phases_of_inclusions(self) -> list[int]:
        """Phase of each inclusion model (the next free indices by default)."""
        if self.operation != "inclusions":
            return []
        if self.inclusion_phases is not None:
            return list(self.inclusion_phases)
        start = max(self.models[0].active_phases()) + 1
        return list(range(start, start + len(self.models) - 1))

    def active_phases(self) -> dict[int, str]:
        if self.operation == "nested":
            out = {0: "outside", 1: "phase 1 of every model"}
            for model in self.models[1:]:
                for phase, name in model.active_phases().items():
                    out.setdefault(phase, f"{name} ({model.name})")
            return out
        if self.operation != "inclusions":
            return dict(self.phases)
        out = self.models[0].active_phases()
        for model, phase in zip(self.models[1:], self.phases_of_inclusions, strict=True):
            out[phase] = f"inclusions ({model.name})"
        return out

    # -- realizations -----------------------------------------------------------------------

    def generate(self, domain: Domain, seed: SeedLike = None) -> Combined:
        self._check_domain(domain)
        rng, seed_info = make_rng(seed)
        if seed_info is not None:  # children of the seed sequence: reproducible
            seeds = list(seed_from_description(seed_info).spawn(len(self.models)))
        else:
            seeds = list(rng.spawn(len(self.models)))
        parts = [m.generate(domain, s) for m, s in zip(self.models, seeds, strict=True)]
        phases = self.phases_of_inclusions if self.operation == "inclusions" else None
        if phases is not None and max(phases) > 255:
            raise ValueError("inclusion_phases: phase indices from 0 to 255")
        return Combined(domain, self.operation, parts, phases, model=self, seed=seed_info)

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """Volume fraction and specific surface of set operations on independent
        components, when theirs are known."""
        if self.operation == "inclusions":
            return {}
        values = []
        for m in self.models:
            try:
                t = m.theory(domain)
            except (ValueError, NotImplementedError):
                return {}
            values.append(t)
        out = {}
        if all("volume_fraction" in t for t in values):
            p = [t["volume_fraction"] for t in values]
            out["volume_fraction"] = _fraction(self.operation, p)
            if all("specific_surface" in t for t in values):
                s = [t["specific_surface"] for t in values]
                out["specific_surface"] = _surface(self.operation, p, s)
        return out


def _fraction(operation: str, p: list[float]) -> float:
    if operation == "union":
        return 1.0 - math.prod(1.0 - x for x in p)
    if operation in ("intersection", "nested"):
        return math.prod(p)
    if operation == "difference":
        return p[0] * math.prod(1.0 - x for x in p[1:])
    return 1.0 - p[0]


def _surface(operation: str, p: list[float], s: list[float]) -> float:
    """Surface of the boundary of a set operation on independent sets: the boundary of
    each set where the others leave it on the boundary of the result."""
    if operation == "complement":
        return s[0]
    total = 0.0
    for i, si in enumerate(s):
        others = [x for j, x in enumerate(p) if j != i]
        if operation == "union":
            total += si * math.prod(1.0 - x for x in others)
        elif operation in ("intersection", "nested"):
            total += si * math.prod(others)
        elif i == 0:  # difference: boundary of A outside the others
            total += si * math.prod(1.0 - x for x in others)
        else:  # boundary of another set inside A and outside the rest
            rest = [x for j, x in enumerate(p[1:], start=1) if j != i]
            total += si * p[0] * math.prod(1.0 - x for x in rest)
    return total
