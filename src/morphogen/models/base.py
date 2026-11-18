"""Base classes of morphological models and of their realizations (microstructures).

A :class:`Model` holds the parameters of a random set (or of a deterministic
architecture). ``model.generate(domain, seed)`` returns a :class:`Microstructure`, the
explicit ("vector") description of one realization, which can be voxelized at any
resolution, queried point-wise, turned into level-set fields and exported.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from morphogen._utils import to_jsonable
from morphogen._version import __version__
from morphogen.distributions import Distribution, distribution_from_config
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.orientations import OrientationDistribution, orientation_from_config
from morphogen.seeding import SeedLike

# ----------------------------------------------------------------------------------
# Parameters


def param(
    default=dataclasses.MISSING,
    *,
    doc: str,
    kind: str = "float",
    unit: str = "",
    kw_only: bool = False,
):
    """Dataclass field carrying the metadata used by configs, the CLI and the GUI.

    ``kind`` is one of: float, int, bool, str, fraction, distribution, distributions (a
    list of them), orientation, vector, floats, ints, json (nested lists), model (a model
    given as a mapping with a ``type`` key), models (a list of them), anisotropy (an
    affine anisotropy, :mod:`morphogen.models.affine`).
    """
    metadata = {"doc": doc, "kind": kind, "unit": unit}
    if kw_only:  # after the positional parameters of the subclasses too
        return dataclasses.field(default=default, metadata=metadata, kw_only=True)
    return dataclasses.field(default=default, metadata=metadata)


def _int(value) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise TypeError
    return int(value)


def _convert(kind: str, value: Any, name: str):
    if value is None:
        return None
    if kind == "models":  # errors of the nested models keep their own messages
        if isinstance(value, (dict, Model)):
            raise ValueError(f"{name}: expected a list of models")
        return tuple(v if isinstance(v, Model) else model_from_config(v) for v in value)
    if kind == "json":  # nested lists of numbers or strings, kept as tuples

        def frozen(v):
            return tuple(frozen(x) for x in v) if isinstance(v, (list, tuple)) else v

        return frozen(value)
    if kind == "model":
        if isinstance(value, Model):
            return value
        if not isinstance(value, dict):
            raise ValueError(f"{name}: expected a model (a mapping with a 'type' key)")
        return model_from_config(value)
    if kind == "distributions":
        if isinstance(value, (dict, str, int, float, Distribution)):
            raise ValueError(f"{name}: expected a list of distributions (numbers or mappings)")
        return tuple(distribution_from_config(v, f"{name}[{k}]") for k, v in enumerate(value))
    try:
        if kind == "distribution":
            return distribution_from_config(value, name)
        if kind == "orientation":
            return orientation_from_config(value, name)
        if kind == "anisotropy":
            from morphogen.models.affine import anisotropy_from_config

            return anisotropy_from_config(value, name)
        if kind in ("float", "fraction"):
            if isinstance(value, bool):
                raise TypeError
            return float(value)
        if kind == "int":
            return _int(value)
        if kind == "ints":
            return tuple(_int(v) for v in value)
        if kind == "bool":
            if not isinstance(value, bool):
                raise TypeError
            return value
        if kind == "str":
            return str(value)
        if kind == "vector":
            if np.ndim(value) == 0:
                return float(value)
            return tuple(float(v) for v in value)
        if kind == "floats":
            return tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        detail = f" ({exc})" if str(exc) else ""
        raise ValueError(
            f"{name}: invalid value {value!r} for a {kind} parameter{detail}"
        ) from None
    raise ValueError(f"{name}: unsupported parameter kind {kind!r}")


# ----------------------------------------------------------------------------------
# Models

_REGISTRY: dict[str, type[Model]] = {}


def register(cls: type[Model]) -> type[Model]:
    """Class decorator adding a model to the registry under ``cls.name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no registry name")
    _REGISTRY[cls.name] = cls
    return cls


def get_model(name: str) -> type[Model]:
    """The class of a registered model, by its name (:func:`list_models`)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown model type {name!r}; available models: {', '.join(list_models())}"
        ) from None


def list_models() -> list[str]:
    """The names of the registered models, sorted."""
    return sorted(_REGISTRY)


def model_from_config(cfg: dict) -> Model:
    """Build a model from a mapping ``{"type": name, **parameters}``."""
    if not isinstance(cfg, dict) or "type" not in cfg:
        raise ValueError("model: expected a mapping with a 'type' key")
    return get_model(str(cfg["type"])).from_config(cfg)


@dataclasses.dataclass(frozen=True)
class Model(ABC):
    """Morphological model (random set or deterministic architecture)."""

    name: ClassVar[str] = ""
    #: Meaning of the phase indices of the voxel images.
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "inclusions"}
    #: False for models that cannot be periodic (e.g. infinite Poisson fibres).
    supports_periodic: ClassVar[bool] = True
    #: Whether one level set, that of the phase 1, describes the realizations (two phases:
    #: the extended finite elements, the contours of the sections). False for the models
    #: of more phases, or of other phases than 0 and 1 (cells of a tessellation, coated
    #: inclusions, osteons...): each of their phases has its implicit function
    #: (:meth:`Microstructure.phase_level_set`), from which they are meshed conformingly.
    has_level_set: ClassVar[bool] = True
    #: True for models whose realizations are meshed grain by grain from exact cells
    #: (Voronoi and Laguerre tessellations, see :mod:`morphogen.mesh.polycrystal`).
    meshed_by_grains: ClassVar[bool] = False
    #: Objects counted by :meth:`expected_count` (fibres, spheres, cells...).
    count_label: ClassVar[str] = ""
    #: The parameters that are paths of files (or the keys of parameters that are
    #: mappings: ``texture.file``): a configuration file gives them from its own folder
    #: (:func:`morphogen.config.load_config`).
    path_parameters: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        # accept plain values (numbers, mappings, strings) for every parameter kind
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            kind = f.metadata.get("kind", "float")
            converted = _convert(kind, value, f.name)
            if converted is not value:
                object.__setattr__(self, f.name, converted)
        self._validate()

    def _validate(self) -> None:  # noqa: B027 (optional hook)
        """Check parameter values (raise ValueError with an explicit message)."""

    @abstractmethod
    def generate(self, domain: Domain, seed: SeedLike = None) -> Microstructure:
        """One realization of the model in ``domain``."""

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """Theoretical morphological characteristics known in closed form (may be empty).

        Some values depend on the domain (e.g. the intensity implied by a number of cells).
        """
        return {}

    def active_phases(self) -> dict[int, str]:
        """Phases the realizations of this model can contain, with their names (the
        :attr:`phases` of the model, unless its parameters choose them)."""
        return dict(self.phases)

    def expected_count(self, domain: Domain) -> float | None:
        """Mean number of objects of the model in ``domain`` (None when undefined).

        Used to express RVE sizes as numbers of fibres, grains or cells
        (Dirrenberger et al. 2014, Table 10).
        """
        return None

    # -- configuration -------------------------------------------------------------

    def to_config(self) -> dict:
        cfg: dict[str, Any] = {"type": self.name}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, (Distribution, OrientationDistribution)):
                value = value.to_config()
            if f.metadata.get("kind") == "anisotropy":
                value = value.to_config()
            if f.metadata.get("kind") == "distributions":
                value = [d.to_config() for d in value]
            if f.metadata.get("kind") == "models":
                value = [m.to_config() for m in value]
            elif f.metadata.get("kind") == "model":
                value = value.to_config()
            cfg[f.name] = to_jsonable(value)
        return cfg

    @classmethod
    def from_config(cls, cfg: dict) -> Model:
        cfg = dict(cfg)
        cfg.pop("type", None)
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(cfg) - set(fields))
        if unknown:
            raise ValueError(
                f"model {cls.name}: unknown parameters {unknown}; "
                f"valid parameters: {', '.join(fields)}"
            )
        kwargs = {}
        for key, value in cfg.items():
            kind = fields[key].metadata.get("kind", "float")
            kwargs[key] = _convert(kind, value, f"model.{key}")
        missing = [
            f.name
            for f in fields.values()
            if f.name not in kwargs
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ]
        if missing:
            raise ValueError(f"model {cls.name}: missing parameters {missing}")
        try:
            return cls(**kwargs)
        except ValueError as exc:
            raise ValueError(f"model {cls.name}: {exc}") from None

    @classmethod
    def parameters(cls) -> list[dict]:
        """Description of the parameters (name, kind, default, unit, doc)."""
        out = []
        for f in dataclasses.fields(cls):
            default = None if f.default is dataclasses.MISSING else f.default
            if isinstance(default, (Distribution, OrientationDistribution)):
                default = default.to_config()
            out.append(
                {
                    "name": f.name,
                    "kind": f.metadata.get("kind", "float"),
                    "default": to_jsonable(default),
                    "required": f.default is dataclasses.MISSING,
                    "unit": f.metadata.get("unit", ""),
                    "doc": f.metadata.get("doc", ""),
                }
            )
        return out

    def _check_domain(self, domain: Domain) -> None:
        if domain.periodic and not self.supports_periodic:
            raise ValueError(
                f"model {self.name} cannot generate periodic microstructures; "
                "use a non-periodic domain (periodic: false)"
            )


# ----------------------------------------------------------------------------------
# Microstructures

_KINDS: dict[str, type[Microstructure]] = {}


def register_kind(cls: type[Microstructure]) -> type[Microstructure]:
    """Registers a kind of microstructure (a class decorator), which
    :meth:`Microstructure.from_dict` then reads back."""
    _KINDS[cls.kind] = cls
    return cls


class Microstructure(ABC):
    """One realization of a model in a domain."""

    kind: ClassVar[str] = ""

    def __init__(
        self,
        domain: Domain,
        *,
        model: Model | None = None,
        seed: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.domain = domain
        self.model = model
        self.seed = seed
        self.metadata = dict(metadata or {})

    # -- discretization ------------------------------------------------------------

    def grid(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> Grid:
        """Resolve the voxel grid from ``shape``, ``voxel_size`` or an explicit grid."""
        if grid is not None:
            if shape is not None or voxel_size is not None:
                raise ValueError("give either 'grid' or 'shape'/'voxel_size', not both")
            return grid
        return Grid.for_domain(self.domain, shape=shape, voxel_size=voxel_size)

    @abstractmethod
    def voxelize(
        self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling: int = 1
    ) -> VoxelImage:
        """Phase image (uint8), or partial volume fractions of phase 1 (float32) when
        ``supersampling > 1`` (``supersampling**3`` sub-samples per voxel)."""

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Field negative inside phase 1, positive outside and zero on its boundary."""
        raise NotImplementedError(f"{type(self).__name__} has no level-set representation")

    def implicit_phases(self) -> list[int]:
        """The phases of the realization, each described by an implicit function
        (:meth:`phase_level_set`), in increasing order: phase 1 and what surrounds it
        (the structures of more phases, or of others, say theirs)."""
        return [0, 1]

    def phase_level_set(
        self, phase: int, shape=None, *, voxel_size=None, grid: Grid | None = None
    ) -> VoxelImage:
        """The implicit function of a phase: a field negative inside ``phase``, positive
        outside and zero on its boundary, at the centres of the voxels of a grid. The
        interfaces are those of the model (see :mod:`morphogen.models.implicit`): the
        conforming meshes, the surfaces and the sections of the phase are cut from it.
        Phase 1 is :meth:`level_set`; the other phase of a structure of two phases, its
        opposite; the structures of more phases give each of theirs."""
        phase = int(phase)
        phases = self.implicit_phases()
        if phase not in phases:
            raise ValueError(f"phase {phase}: the phases of this structure are {phases}")
        if phase == 1:
            return self.level_set(shape, voxel_size=voxel_size, grid=grid)
        if len(phases) == 2 and 1 in phases:
            field = self.level_set(shape, voxel_size=voxel_size, grid=grid)
            return field.with_data(-np.asarray(field.data))
        raise NotImplementedError(
            f"{type(self).__name__} has no implicit function of its phase {phase}"
        )

    def implicit_at(self, phase: int, points, band: float) -> np.ndarray | None:
        """The implicit function of one phase at arbitrary points (rows of ``points``),
        clipped to ``[-band, band]``: that of :meth:`phase_level_sets_at`; None for the
        structures whose functions are those of the points of grids only
        (:meth:`phase_level_set`), and for a phase they do not have."""
        try:
            values = self.phase_level_sets_at(points, float(band))
        except NotImplementedError:
            return None
        if int(phase) not in values:
            return None
        return np.clip(np.asarray(values[int(phase)], dtype=float), -float(band), float(band))

    def implicit_pieces(self, depth: float = 0.0) -> tuple[list, int] | None:
        """The smooth pieces of the phases, for the structures whose implicit functions
        have folds (the unions of spheres and segments and their coatings, whose surfaces
        have a sharp edge where two primitives meet): pieces in the order of the cuts of
        a mesher that follows these edges, and the phase of what they leave (see
        :mod:`morphogen.models.pieces`). A point belongs to the phase of the first piece
        that holds it. Two primitives that enter each other by more than ``depth`` are
        in different pieces. None for the other structures."""
        return None

    def contains(self, points) -> np.ndarray:
        """Whether points lie in phase 1."""
        raise NotImplementedError(f"{type(self).__name__} does not support point queries")

    #: A bound of the slope of the functions of :meth:`phase_level_sets_at` (1 for signed
    #: distances, and for their minima and maxima): the adaptive meshes leave unrefined
    #: the edges that an interface cannot reach at this slope
    lipschitz: float = 1.0

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """The function of phase 1 at points of space (:meth:`level_set` at any point),
        negative in phase 1, whose zero set is its boundary: the signed distance to it,
        exact within ``reach`` of it and at least ``reach`` beyond, when the structure knows
        it (the others: a function with the same zero set, about a distance near it). The
        structures of several phases give the function of their phase 1
        (:meth:`phase_level_sets_at`), if they have one."""
        raise NotImplementedError(
            f"{type(self).__name__} has no point-wise level set (no exact mesh but its "
            "convex pieces, if any)"
        )

    def phase_level_sets_at(self, points, reach: float = np.inf) -> dict[int, np.ndarray]:
        """The implicit functions of all the phases at points of space (rows of
        ``points``): ``{k: phi_k}`` for each phase ``k`` of :meth:`implicit_phases`, negative
        in it, positive outside and zero on its boundary, exact within ``reach`` of the
        interfaces and at least ``reach`` beyond (or about a distance near them, as
        :meth:`level_set_at`). It is :meth:`phase_level_set` at any point: the same values
        at the centres of the voxels of a grid, within the band of the grid. Two phases:
        ``{1: phi, 0: -phi}`` of :meth:`level_set_at`. The exact meshes
        (:mod:`morphogen.mesh.exact`) cut them, and the curved elements
        (:mod:`morphogen.mesh.curved`) put their nodes on their zeros. Raises
        NotImplementedError for the structures whose functions are those of the points of
        grids only."""
        if self.implicit_phases() != [0, 1]:
            raise NotImplementedError(
                f"{type(self).__name__} has no point-wise functions of its phases"
            )
        phi = np.asarray(self.level_set_at(points, reach), dtype=float)
        return {0: -phi, 1: phi}

    @property
    def has_labels(self) -> bool:
        """Whether ``labels(...)`` gives an image of cell indices (cells, grains)."""
        return False

    # -- description ---------------------------------------------------------------

    @property
    def phase_names(self) -> dict[int, str]:
        return dict(self.model.phases) if self.model is not None else {0: "0", 1: "1"}

    def image_metadata(self) -> dict:
        return to_jsonable(
            {
                "morphogen": __version__,
                "model": self.model.to_config() if self.model is not None else None,
                "domain": self.domain.to_dict(),
                "seed": self.seed,
                "phases": {str(k): v for k, v in self.phase_names.items()},
            }
        )

    def summary(self) -> dict:
        """Counts and sizes describing the realization."""
        return {}

    @abstractmethod
    def _geometry(self) -> dict:
        """JSON-able explicit geometry."""

    @classmethod
    @abstractmethod
    def _from_geometry(cls, domain: Domain, geometry: dict, **kwargs) -> Microstructure:
        """Inverse of :meth:`_geometry`."""

    def to_dict(self) -> dict:
        return to_jsonable(
            {
                "morphogen": __version__,
                "kind": self.kind,
                "domain": self.domain.to_dict(),
                "model": self.model.to_config() if self.model is not None else None,
                "seed": self.seed,
                "metadata": self.metadata,
                "geometry": self._geometry(),
            }
        )

    @staticmethod
    def from_dict(data: dict) -> Microstructure:
        kind = data.get("kind")
        if kind not in _KINDS:
            raise ValueError(f"unknown microstructure kind {kind!r}")
        model = model_from_config(data["model"]) if data.get("model") else None
        return _KINDS[kind]._from_geometry(
            Domain.from_dict(data["domain"]),
            data["geometry"],
            model=model,
            seed=data.get("seed"),
            metadata=data.get("metadata"),
        )

    def __repr__(self) -> str:
        name = self.model.name if self.model is not None else self.kind
        info = ", ".join(f"{k}={v}" for k, v in self.summary().items())
        return f"<{type(self).__name__} {name} {info}>"
