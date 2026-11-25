"""The shapes of Microgen (3MAH) as a model of morphoGen: TPMS sheets and skeletons,
spinodoids, spheres, boxes, cylinders, capsules, ellipsoids, cylindrical and spherical
TPMS, described by implicit fields negative inside, as morphoGen describes its phases.

The model ``microgen`` names a class of Microgen and its arguments; its realization is
the inside of the field of that shape (phase 1), which morphoGen voxelizes, meshes from
its implicit function, measures and homogenizes like any other structure. The package
``microgen`` (Microgen 2, GPL-3.0; ``pip install --pre microgen``) is needed when a
realization is generated or read back, not to configure the model.
:mod:`morphogen.interop.microgen` gives the other way (a phase of any realization as a
shape of Microgen) and the meshes of both through PyVista.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.seeding import SeedLike, make_rng

#: The parts of a TPMS of Microgen (``Tpms.as_<part>()``)
PARTS = ("", "sheet", "lower_skeletal", "upper_skeletal", "surface")
#: Points of the field evaluated at once (memory)
_SLAB_POINTS = 1 << 20


def _microgen():
    try:
        import microgen
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the model microgen needs Microgen 2 (3MAH): pip install --pre microgen"
        ) from None
    return microgen


def build_shape(description: dict, seed: int | None = None):
    """The shape of Microgen of a description ``{"shape": class name, "arguments":
    {...}, "part": ...}``: the class of the package (``Tpms``, ``Spinodoid``, ``Sphere``,
    ``Box``...) called with the arguments, a ``surface_function`` given by its name in
    ``microgen.surface_functions``; ``center`` and ``orientation`` (Bunge angles in
    degrees, ZXZ) baked into its field (Microgen's own constructors leave them to its
    meshes); the ``part`` of a TPMS (``sheet``, ``lower_skeletal``...); ``seed`` given to
    the shapes that draw (spinodoids) when the arguments have none."""
    mg = _microgen()
    name = str(description.get("shape", ""))
    cls = getattr(mg, name, None)
    if not (inspect.isclass(cls) and issubclass(cls, mg.Shape)):
        raise ValueError(f"microgen: no shape {name!r} in Microgen")
    params = dict(description.get("arguments") or {})
    center = params.pop("center", None)
    orientation = params.pop("orientation", None)
    if isinstance(params.get("surface_function"), str):
        function = getattr(mg.surface_functions, params["surface_function"], None)
        if function is None or not callable(function):
            raise ValueError(f"microgen: no surface function {params['surface_function']!r}")
        params["surface_function"] = function
    params = {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}
    accepted = inspect.signature(cls.__init__).parameters
    if seed is not None and "seed" in accepted and params.get("seed") is None:
        params["seed"] = int(seed)
    try:
        shape = cls(**params)
    except TypeError as exc:
        raise ValueError(f"microgen: the arguments of {name}: {exc}") from None
    part = str(description.get("part") or "")
    if part:
        maker = getattr(shape, f"as_{part}", None)
        if maker is None:
            raise ValueError(f"microgen: {name} has no part {part!r} (those of the TPMS)")
        shape = maker()
    if orientation is not None:
        shape = shape.rotate(tuple(float(a) for a in orientation))
    if center is not None:
        shape = shape.translate(tuple(float(c) for c in center))
    return shape


@register_kind
class MicrogenStructure(Microstructure):
    """Phase 1 is the inside of a shape of Microgen (its field negative), phase 0 the rest;
    in a periodic domain the field is that of the points brought back into the box. Its
    implicit function is the field of the shape (a distance for the spheres, boxes,
    cylinders; about one near the surface of the TPMS and spinodoids), at the voxels and
    at any point. A structure of a description (the model ``microgen``) is saved and read
    back; one of a shape given as it is (:func:`morphogen.interop.microgen.from_microgen`)
    is not."""

    kind = "microgen"

    def __init__(self, domain: Domain, shape=None, description: dict | None = None, **kwargs):
        super().__init__(domain, **kwargs)
        self.description = dict(description or {})
        if shape is None and not self.description:
            raise ValueError("microgen: a shape of Microgen or its description")
        self._shape = shape
        self.lipschitz = self._slope()

    @property
    def shape(self):
        """The shape of Microgen (built from the description at first use)."""
        if self._shape is None:
            seed = (self.metadata or {}).get("shape_seed")
            self._shape = build_shape(self.description, seed)
        return self._shape

    def field(self, points) -> np.ndarray:
        """The field of the shape at points (rows), negative inside; the points of a
        periodic domain brought back into the box first."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.domain.periodic:
            p = self.domain.wrap(p)
        out = np.empty(len(p))
        for k in range(0, len(p), _SLAB_POINTS):
            q = p[k : k + _SLAB_POINTS]
            out[k : k + len(q)] = np.asarray(self.shape.evaluate(q[:, 0], q[:, 1], q[:, 2]))
        return out

    def _slope(self) -> float:
        """A bound of the slope of the field (for the adaptive meshes): 1.5 times the
        largest one of central differences on a coarse grid of the domain, at least 1."""
        grid = Grid.for_domain(self.domain, shape=24)
        h = float(grid.spacing.min())
        centres = grid.center_points()
        values = self.field(centres)
        largest = 0.0
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = h
            slope = np.abs(self.field(centres + step) - values) / h
            largest = max(largest, float(slope.max()))
        return max(1.0, 1.5 * largest)

    def _grid_field(self, grid: Grid) -> np.ndarray:
        out = np.empty(grid.shape, dtype=np.float32)
        nx, ny, nz = grid.shape
        per = max(1, _SLAB_POINTS // max(nx * ny, 1))
        for k0 in range(0, nz, per):
            k1 = min(nz, k0 + per)
            out[:, :, k0:k1] = self.field(grid.center_points(k0, k1)).reshape(nx, ny, k1 - k0)
        return out

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        meta = self.image_metadata()
        if supersampling == 1:
            phase = (self._grid_field(grid) < 0).astype(np.uint8)
            return VoxelImage.from_grid(phase, grid, name="phase", metadata=meta)
        s = int(supersampling)
        fine = Grid(grid.domain, tuple(n * s for n in grid.shape))
        inside = (self._grid_field(fine) < 0).astype(np.float32)
        nx, ny, nz = grid.shape
        data = inside.reshape(nx, s, ny, s, nz, s).mean(axis=(1, 3, 5)).astype(np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=meta)

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        meta = self.image_metadata()
        return VoxelImage.from_grid(self._grid_field(grid), grid, name="level_set", metadata=meta)

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        return np.clip(self.field(points), -float(reach), float(reach))

    def contains(self, points) -> np.ndarray:
        return self.field(points) < 0

    def summary(self) -> dict:
        return {"shape": self.description.get("shape", type(self._shape).__name__)}

    def _geometry(self) -> dict:
        if not self.description:
            raise ValueError(
                "microgen: a structure of a shape given as it is cannot be saved; describe "
                "it ({shape, arguments, part}, the model microgen) to save it"
            )
        return {"description": self.description}

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(domain, description=geometry["description"], **kwargs)


@register
@dataclass(frozen=True)
class Microgen(Model):
    """A shape of Microgen (3MAH): its class, its arguments, the part of a TPMS; phase 1
    is the inside of its field.

    ``shape`` is a class of Microgen 2 that has a field (``Tpms``, ``Spinodoid``,
    ``Sphere``, ``Box``, ``Cylinder``, ``Capsule``, ``Ellipsoid``, ``CylindricalTpms``,
    ``SphericalTpms``...; its strut lattices are CAD only), ``arguments`` the keyword
    arguments of its constructor (a ``surface_function`` by its name in
    ``microgen.surface_functions``: gyroid, schwarz_p, schwarz_d, neovius, schoen_iwp,
    lidinoid, fischer_koch_s...; ``center`` and ``orientation``, Bunge angles in degrees,
    put into the field), ``part`` that of a TPMS (sheet, lower_skeletal, upper_skeletal).
    A spinodoid without a seed takes one from the seed of the realization. Needs the
    package ``microgen`` (``pip install --pre microgen``) to generate."""

    name = "microgen"
    phases: ClassVar[dict[int, str]] = {0: "matrix", 1: "shape"}

    shape: str = param(
        "Tpms",
        doc="Class of the shape in Microgen (Tpms, Spinodoid, Sphere, Box, Cylinder, Capsule, "
        "Ellipsoid, CylindricalTpms...)",
        kind="str",
    )
    arguments: dict | None = param(
        None,
        doc="Keyword arguments of the class (a surface_function by its name; center and "
        "orientation in the field), e.g. {surface_function: schoen_iwp, offset: 0.3, cell_size: 1}",
        kind="json",
    )
    part: str = param(
        "",
        doc="Part of a TPMS: sheet, lower_skeletal, upper_skeletal (empty: the shape itself)",
        kind="str",
    )

    def _validate(self) -> None:
        if not str(self.shape).isidentifier():
            raise ValueError(f"shape: the name of a class of Microgen, got {self.shape!r}")
        if self.arguments is not None and not isinstance(self.arguments, dict):
            raise ValueError("arguments: a mapping of the keyword arguments of the class")
        if self.part not in PARTS:
            raise ValueError(f"part: one of {', '.join(p or repr('') for p in PARTS)}")

    def description(self) -> dict:
        return {"shape": self.shape, "arguments": dict(self.arguments or {}), "part": self.part}

    def generate(self, domain: Domain, seed: SeedLike = None) -> MicrogenStructure:
        rng, seed_info = make_rng(seed)
        shape_seed = int(rng.integers(2**31 - 1))
        shape = build_shape(self.description(), shape_seed)
        return MicrogenStructure(domain, shape, self.description(), model=self,
                                 seed=seed_info, metadata={"shape_seed": shape_seed})  # fmt: skip
