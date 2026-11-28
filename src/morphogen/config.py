"""Configuration files (YAML, JSON or TOML) describing generation runs and RVE studies.

A configuration has the sections ``model``, ``domain``, ``voxels`` (voxel images),
``mesh`` (conforming tetrahedral meshes), ``outputs``, ``analysis``, ``homogenization``
(apparent properties: finite elements on the meshes, or FFT on the voxel images),
``visualization`` (figures and videos of the images, meshes and fields of the first
realizations) and, for RVE studies, ``rve`` (see docs/architecture.md and the templates).
"""

from __future__ import annotations

import itertools
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from morphogen.analysis.measures import MEASURES
from morphogen.domain import Domain
from morphogen.gpu import DEVICES
from morphogen.homogenization.apparent import (
    CONTACT_OPTIONS,
    INTERFACE_OPTIONS,
    PHYSICS,
    PROPERTIES,
)
from morphogen.homogenization.core import VOIGT_NAMES, Loading, boundary_condition
from morphogen.homogenization.materials import Material, materials_from_config
from morphogen.homogenization.paths import LoadingPath, PathQuantity
from morphogen.homogenization.solvers import SOLVERS
from morphogen.io import GEOMETRY_FORMATS, IMAGE_FORMATS
from morphogen.models.base import Model, get_model, model_from_config
from morphogen.rve.study import TENSOR_PROPERTIES

SURFACE_FORMATS = ("stl",)
MESH_FORMATS = ("inp", "msh", "vtu", "xdmf", "med", "mesh", "mail")
HOMOGENIZATION_METHODS = ("auto", "fe", "fft", "xfem")
#: Formats of the tables of a run besides .csv (outputs.tables), of the figures and the
#: videos of the visualization section (the 3D views of VTK are bitmaps)
TABLE_FORMATS = ("csv", "txt", "xlsx")
FIGURE_FORMATS = ("png", "tif", "jpg", "svg", "pdf", "eps", "tex")
BITMAP_FORMATS = ("png", "tif", "jpg")
VIDEO_FORMATS = ("mp4", "gif", "frames")


class ConfigError(ValueError):
    """Invalid configuration (the message names the faulty entry)."""


def _check_keys(section: str, data: dict, allowed: set, required: set = frozenset()) -> None:
    if not isinstance(data, dict):
        raise ConfigError(f"{section}: expected a mapping, got {data!r}")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{section}: unknown keys {unknown}; allowed: {sorted(allowed)}")
    missing = sorted(required - set(data))
    if missing:
        raise ConfigError(f"{section}: missing keys {missing}")


def _list_of(section: str, value, allowed) -> list[str]:
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    for item in items:
        if item not in allowed:
            raise ConfigError(f"{section}: unknown value {item!r}; allowed: {list(allowed)}")
    return [str(v) for v in items]


def _positive_int(section: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{section}: expected a positive integer, got {value!r}")
    return value


@dataclass
class VoxelSpec:
    shape: int | tuple[int, int, int] | None = None
    voxel_size: float | None = None
    supersampling: int = 1
    #: the axis (0, 1, 2) normal to which the image of a run is one layer of voxels
    #: through the middle of the realization (``section: x``, ``y`` or ``z``): a planar
    #: study of a three-dimensional model
    section: int | None = None

    def kwargs(self) -> dict:
        return {"shape": self.shape, "voxel_size": self.voxel_size}

    def image(self, micro, supersampling: int = 1):
        """The voxel image of a realization: its volume, or the layer of its section."""
        extra = {"supersampling": supersampling} if supersampling > 1 else {}
        return self.layer(micro.voxelize(**self.kwargs(), **extra))

    def layer(self, image):
        """The layer of the section of an image of the volume (the image itself without
        a section)."""
        return image if self.section is None else image.layer(self.section)


@dataclass
class MeshSpec:
    """Conforming tetrahedral meshes of the realizations (see
    :func:`morphogen.mesh.conforming.conforming_mesh`). ``order``: None in a
    configuration that does not give it, then 2 (10-node tetrahedra) when the mesh is
    analysed by finite elements and 1 (4-node) when it is only written
    (:func:`load_config`). ``plane``: triangular meshes of the plane instead
    (:func:`morphogen.mesh.triangles.plane_mesh`: the exact columnar grains of a
    tessellation whose germs lie in one plane, the section of any other structure at the
    height ``z``, by default the middle of the box); None (``auto``): for those
    tessellations."""

    size: float
    min_size: float | None = None
    hausdorff: float | None = None
    gradation: float = 2.0
    background: float | None = None
    order: int | None = None
    curved: bool | None = None
    sharp_edges: bool | None = None
    phases: str = "auto"
    grains: bool | None = None
    min_feature: float | None = None
    cells: str = "whole"
    method: str = "auto"
    plane: bool | None = None
    z: float | None = None

    def on_plane(self, micro) -> bool:
        """Whether the mesh of a realization is a mesh of the plane: asked (``plane``), or
        by default the columnar grains of a tessellation whose germs lie in one plane and
        the maps of grains (the orientation maps of EBSD)."""
        if self.plane is not None:
            return bool(self.plane)
        from morphogen.mesh.triangles import lattice_map, planar_tessellation

        return planar_tessellation(micro) or lattice_map(micro)

    def plane_kwargs(self, porous=False) -> dict:
        """Arguments of :func:`morphogen.mesh.triangles.plane_mesh` (``phases`` as for
        :meth:`kwargs`)."""
        full = self.kwargs(porous)
        out = {k: full[k] for k in ("size", "min_size", "hausdorff", "gradation", "background",
                                    "order", "phases")}  # fmt: skip
        if self.z is not None:
            out["z"] = self.z
        if self.min_feature is not None:
            out["min_feature"] = self.min_feature
        if self.curved is not None:
            out["curved"] = self.curved
        return out

    def kwargs(self, porous=False) -> dict:
        """Arguments of ``conforming_mesh``; ``phases='auto'`` meshes only the solid phase
        of porous media (``porous``: the void phases, or True for a void phase 0): phase 1
        when phase 0 is the void, phase 0 when the pores are phase 1."""
        phases = self.phases
        if phases == "auto":
            voids = {0} if porous is True else set(porous or ())
            if not voids:
                phases = "all"
            else:
                phases = "matrix" if voids == {1} else "solid"
        out = {
            "size": self.size,
            "min_size": self.min_size,
            "hausdorff": self.hausdorff,
            "gradation": self.gradation,
            "background": self.background,
            "order": 1 if self.order is None else self.order,
            "phases": phases,
            "cells": self.cells,
            "method": self.method,
        }
        if self.curved is not None:
            out["curved"] = self.curved
        if self.sharp_edges is not None:
            out["sharp_edges"] = self.sharp_edges
        if self.grains is not None:
            out["grains"] = self.grains
        if self.min_feature is not None:
            out["min_feature"] = self.min_feature
        return out


@dataclass
class OutputSpec:
    directory: Path = Path("morphogen_output")
    name: str = "sample"
    geometry: list[str] = field(default_factory=list)
    image: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    level_set: list[str] = field(default_factory=list)
    mesh: list[str] = field(default_factory=list)
    mesh_phases: list[int] | None = None
    conforming_mesh: list[str] = field(default_factory=list)
    periodic_pairs: bool = False
    periodic_equations: bool = False
    surface: list[str] = field(default_factory=list)
    damask: bool = False
    neper: bool = False
    database: Path | None = None
    #: AMITEX_FFTP runs of the image (the options of morphogen.io.write_amitex), or None
    amitex: dict | None = None
    #: the study of Code_Aster of the conforming mesh (the options of
    #: morphogen.io.write_code_aster), or None
    code_aster: dict | None = None
    #: the case of OpenFOAM of the image or of the conforming mesh (the options of
    #: morphogen.io.write_openfoam, and ``sample``: "voxels" or "mesh"), or None
    openfoam: dict | None = None
    #: the tables of the run (results, homogenization, curves of the loading paths) in
    #: these formats too, besides .csv
    tables: list[str] = field(default_factory=list)

    @property
    def needs_voxels(self) -> bool:
        return bool(
            self.image
            or self.labels
            or self.mesh
            or self.surface
            or self.level_set
            or self.damask
            or self.amitex is not None
            or (self.openfoam is not None and self.openfoam.get("sample") != "mesh")
        )


#: Options of outputs.amitex (those of morphogen.io.write_amitex)
AMITEX_OPTIONS = ("fields", "tolerance", "elastic_only", "processes", "library")


def _amitex(value) -> dict | None:
    """``outputs.amitex``: true (the default options), false, or a mapping of options."""
    if value is None or value is False:
        return None
    if value is True:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"outputs.amitex: true, or a mapping of options {list(AMITEX_OPTIONS)}")
    _check_keys("outputs.amitex", value, set(AMITEX_OPTIONS))
    out = {}
    for key in ("fields", "elastic_only"):
        if key in value:
            out[key] = bool(value[key])
    if value.get("tolerance") is not None:
        tol = float(value["tolerance"])
        if not 0.0 < tol < 1e-3:
            raise ConfigError("outputs.amitex.tolerance: a positive number below 1e-3 (AMITEX)")
        out["tolerance"] = tol
    if "processes" in value:
        n = int(value["processes"])
        if n < 1:
            raise ConfigError("outputs.amitex.processes: a positive number of MPI processes")
        out["processes"] = n
    if "library" in value:
        out["library"] = str(value["library"])
    return out


#: Options of outputs.code_aster (those of morphogen.io.write_code_aster)
CODE_ASTER_OPTIONS = (
    "physics",
    "conditions",
    "mesh_format",
    "fields",
    "memory",
    "time_limit",
    "threads",
)
#: Options of outputs.openfoam (those of morphogen.io.write_openfoam, and the sample)
OPENFOAM_OPTIONS = (
    "sample",
    "kind",
    "phases",
    "axis",
    "lateral",
    "viscosity",
    "pressure_drop",
    "diffusivity",
    "difference",
    "iterations",
    "tolerance",
    "connected",
    "dual",
)


def _code_aster(value) -> dict | None:
    """``outputs.code_aster``: true (the default options), false, or a mapping of options."""
    from morphogen.io.code_aster import CONDITIONS, MESH_FORMATS, PHYSICS

    if value is None or value is False:
        return None
    if value is True:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(
            f"outputs.code_aster: true, or a mapping of options {list(CODE_ASTER_OPTIONS)}"
        )
    _check_keys("outputs.code_aster", value, set(CODE_ASTER_OPTIONS))
    out = dict(value)
    if "physics" in out:
        physics = [out["physics"]] if isinstance(out["physics"], str) else list(out["physics"])
        if not physics or any(k not in PHYSICS for k in physics):
            raise ConfigError(f"outputs.code_aster.physics: some of {list(PHYSICS)}")
        out["physics"] = physics
    if out.get("conditions") is not None and out["conditions"] not in CONDITIONS:
        raise ConfigError(f"outputs.code_aster.conditions: one of {list(CONDITIONS)}")
    if out.get("mesh_format", "aster") not in MESH_FORMATS:
        raise ConfigError(f"outputs.code_aster.mesh_format: one of {list(MESH_FORMATS)}")
    if "fields" in out:
        out["fields"] = bool(out["fields"])
    for key in ("memory", "time_limit"):
        if key in out:
            out[key] = _positive_float(f"outputs.code_aster.{key}", out[key])
    if "threads" in out:
        out["threads"] = _positive_int("outputs.code_aster.threads", out["threads"])
    return out


def _openfoam(value) -> dict | None:
    """``outputs.openfoam``: true (the default options), false, or a mapping of options."""
    from morphogen.io.openfoam import KINDS, LATERAL

    if value is None or value is False:
        return None
    if value is True:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(
            f"outputs.openfoam: true, or a mapping of options {list(OPENFOAM_OPTIONS)}"
        )
    _check_keys("outputs.openfoam", value, set(OPENFOAM_OPTIONS))
    out = dict(value)
    if out.get("sample", "voxels") not in ("voxels", "mesh"):
        raise ConfigError("outputs.openfoam.sample: 'voxels' (the image) or 'mesh' (conforming)")
    if out.get("kind", "flow") not in KINDS:
        raise ConfigError(f"outputs.openfoam.kind: one of {list(KINDS)}")
    if out.get("lateral") is not None and out["lateral"] not in LATERAL:
        raise ConfigError(f"outputs.openfoam.lateral: one of {list(LATERAL)}")
    if "axis" in out and out["axis"] not in (0, 1, 2):
        raise ConfigError("outputs.openfoam.axis: 0, 1 or 2")
    if "phases" in out:
        phases = [out["phases"]] if isinstance(out["phases"], int) else out["phases"]
        if not phases:
            raise ConfigError("outputs.openfoam.phases: the phases of the cells")
        out["phases"] = [int(p) for p in phases]
    for key in ("viscosity", "pressure_drop", "diffusivity", "difference", "tolerance"):
        if key in out:
            out[key] = _positive_float(f"outputs.openfoam.{key}", out[key])
    if "iterations" in out:
        out["iterations"] = _positive_int("outputs.openfoam.iterations", out["iterations"])
    if "connected" in out:
        out["connected"] = bool(out["connected"])
    if out.get("dual") is not None:
        out["dual"] = bool(out["dual"])
    return out


#: Options of the nonlinear solver in configurations, and their names in the code
NONLINEAR_OPTIONS = {
    "tolerance": "newton_tol",
    "max_iterations": "max_newton",
    "max_cuts": "max_cuts",
    "kinematics": "kinematics",
    "integration": "integration",
    "tangent": "tangent",
    "contact": "contact",
    "interfaces": "interfaces",
    "microdamage": "microdamage",
}


@dataclass
class HomogenizationSpec:
    """Apparent properties computed on every realization: finite elements on its
    conforming mesh (``method: fe``) or FFT on its voxel image (``method: fft``); and
    loading paths followed in time with the nonlinear behaviours of the materials, by
    either method."""

    materials: dict[int, Material]
    method: str = "auto"
    boundary_conditions: list[str] = field(default_factory=lambda: ["periodic"])
    properties: list[str] = field(default_factory=lambda: ["bulk_modulus", "shear_modulus"])
    solver: str = "auto"
    tolerance: float = 1e-8
    max_iterations: int | None = None
    fields: bool = False
    loadings: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    nonlinear: dict = field(default_factory=dict)
    device: str = "cpu"
    ntfa: dict = field(default_factory=dict)
    #: the mean absolute temperature of the thermoelectric coupling (K)
    mean_temperature: float = 300.0
    #: the angular frequencies of the harmonic properties, the wavelengths of the
    #: refractive index
    frequencies: list[float] | None = None
    wavelengths: list[float] | None = None
    #: the brittle fracture of the loading paths by the phase-field method (the
    #: parameters of morphogen.fracture.phasefield.PhaseField, and of its solver)
    fracture: dict | None = None
    #: the bulk modulus of the fluid of the pores, for the undrained response of the
    #: property 'poroelastic'
    fluid_bulk_modulus: float | None = None
    #: 'full', or 'bbar': the volume change of each element as its mean (10-node
    #: tetrahedra: the elastic problems and the loading paths; voxels: the loading paths)
    integration: str | None = None
    #: the fatigue of the points along a cycle of a loading path: its ``path``, the
    #: times of its ``cycle`` (None: the whole path) and the ``criteria`` of the phases
    #: (mappings for morphogen.homogenization.fatigue.criterion_from_config)
    fatigue: dict | None = None
    #: the criterion of cleavage of Beremin along a loading path: its ``path`` and its
    #: ``criterion`` (a mapping for morphogen.homogenization.beremin.Beremin)
    beremin: dict | None = None
    #: the elastic problems of the meshes of the plane: 'generalized' (generalized plane
    #: strain, the default) or 'strain' (plane strain); morphogen.homogenization.plane
    plane: str | None = None

    @property
    def porous(self) -> bool:
        return any(m.void for m in self.materials.values())

    @property
    def void_phases(self) -> set[int]:
        """The phases whose material is a void."""
        return {int(p) for p, m in self.materials.items() if m.void}

    def series(self) -> list[str]:
        """Names ``<property>.<bc>`` of the scalar apparent properties computed."""
        return [
            f"{p}.{bc}"
            for p in self.properties
            if p in PROPERTIES
            for bc in self.boundary_conditions
        ]

    def tensors(self) -> list[str]:
        """Names ``stiffness.<bc>`` of the stiffnesses computed (not under mixed
        conditions), whose mean and compliance per volume RVE studies give, then those
        ``conductivity_tensor.<bc>`` of the conductivities (their mean and resistivity) and
        of the tensors of the other transport phenomena (``permittivity_tensor.<bc>``...)."""
        return [
            f"{p}.{bc}"
            for p in TENSOR_PROPERTIES
            if p in self.properties
            for bc in self.boundary_conditions
            if bc != "mixed"
        ]

    def options(self) -> dict:
        return {
            "bcs": self.boundary_conditions,
            "properties": self.properties,
            "solver": self.solver,
            "tol": self.tolerance,
            "maxiter": self.max_iterations,
            "loadings": self.loadings,
            "paths": self.paths,
            "nonlinear": {NONLINEAR_OPTIONS[k]: v for k, v in self.nonlinear.items()},
            "device": self.device,
            "ntfa": self.ntfa or None,
            "mean_temperature": self.mean_temperature,
            "frequencies": self.frequencies,
            "wavelengths": self.wavelengths,
            "fracture": self.fracture,
            "fluid_bulk_modulus": self.fluid_bulk_modulus,
            "integration": self.integration,
        }


@dataclass
class RVESpec:
    method: str = "ensemble"
    sizes: list[float] = field(default_factory=list)
    realizations: list[int] = field(default_factory=list)
    size: float | None = None
    window_sizes: list[float] = field(default_factory=list)
    properties: list[str] = field(default_factory=lambda: ["volume_fraction"])
    relative_errors: list[float] = field(default_factory=lambda: [0.01, 0.05])
    n_realizations: list[int] = field(default_factory=lambda: [1, 10])
    gamma: float | None = None
    min_volume: float | None = None
    max_volume: float | None = None
    bootstrap: int = 1000
    weights: str = "none"
    #: the window method: the tilings per axis, shifted around the periodic realization
    shifts: int = 1


@dataclass
class DiffusionSpec:
    """Transient diffusion in the voxel image of each realization
    (:func:`morphogen.coupling.diffusion.transient_diffusion`): ``diffusivity``,
    ``solubility`` and ``consumption`` of each phase, the ``exposed`` faces held at the
    ``activity`` (a number, or a piecewise linear history ``(times, values)``), the
    ``initial`` activity, the output ``times``, the implicit Euler ``steps`` of each
    interval; ``fields``: the concentration fields at the times."""

    diffusivity: dict[int, float]
    solubility: dict[int, float]
    consumption: dict[int, float]
    exposed: tuple[str, ...] = ("x-", "x+")
    activity: float | tuple = 1.0
    initial: float = 0.0
    times: tuple[float, ...] = ()
    steps: int = 40
    fields: bool = False

    def boundary(self):
        if isinstance(self.activity, tuple):
            t, v = self.activity
            return lambda time: float(np.interp(time, t, v))
        return float(self.activity)


@dataclass
class ViewSpec:
    """A figure or a video of each realization (the ``visualization`` section): ``of``
    what (``image``, ``labels``, ``mesh``, or a result ``<name>.<bc>`` of the
    homogenization, ``group`` "result" or "path"), its ``kind``, the field colouring it
    (``color``) and the options of the views (see :mod:`morphogen.viz.workflow`)."""

    of: str
    group: str
    kind: str
    formats: list[str]
    color: str | None = None
    axis: int = 2
    phase: int = 1
    magnification: float | None = None
    wave: str = "ramp"
    section: tuple[int, float] | None = None
    part: str = "total"
    frames: int | None = None
    mode: str = "once"
    repeat: int = 1
    fps: float | None = None
    cmap: str | None = None
    clim: tuple[float, float] | None = None
    name: str | None = None
    #: rendered by VTK (3D views)
    vtk: bool = False

    @property
    def label(self) -> str:
        """The name of its files, after the stem of the realization."""
        if self.name:
            return self.name
        if self.group in ("image", "labels"):
            return f"{self.of}_{self.kind}"
        if self.group == "mesh":
            return "mesh_sweep" if self.kind == "sweep" else "mesh"
        base = self.of.replace(".", "_")
        return f"{base}_curves" if self.kind == "curves" else f"{base}_{self.color}"


@dataclass
class VisualizationSpec:
    """Figures and videos of the first ``realizations`` of a run, in the colormap
    ``cmap``, videos at ``fps`` frames per second, 3D views of ``size`` pixels seen from
    the ``elevation`` and ``azimuth`` of their camera (degrees), figures at ``dpi`` (see
    :mod:`morphogen.viz.workflow`)."""

    figures: list[ViewSpec] = field(default_factory=list)
    videos: list[ViewSpec] = field(default_factory=list)
    realizations: int = 1
    cmap: str = "viridis"
    fps: float = 24.0
    size: tuple[int, int] = (640, 480)
    dpi: int = 200
    elevation: float = 25.0
    azimuth: float = 30.0

    @property
    def views(self) -> list[ViewSpec]:
        return [*self.figures, *self.videos]

    def shows(self, *groups: str) -> bool:
        return any(v.group in groups for v in self.views)

    def fields(self) -> bool | str:
        """What the homogenization keeps for the views: the fields of the loading paths
        at each of their steps (``"steps"``), the fields (``True``), or nothing."""
        if any(v.group == "path" for v in self.videos):
            return "steps"
        return any(v.group in ("result", "path") and v.kind == "field" for v in self.views)


@dataclass
class RunConfig:
    model: Model
    domain: Domain
    seed: int | None = None
    realizations: int = 1
    voxels: VoxelSpec | None = None
    mesh: MeshSpec | None = None
    outputs: OutputSpec = field(default_factory=OutputSpec)
    analysis: list[str] = field(default_factory=list)
    rve: RVESpec | None = None
    homogenization: HomogenizationSpec | None = None
    diffusion: DiffusionSpec | None = None
    visualization: VisualizationSpec | None = None
    threads: int | None = None
    source: dict = field(default_factory=dict)


def read_config_file(path) -> dict:
    """The mapping of a configuration file (.yaml, .yml, .json or .toml); :class:`ConfigError` if it
    cannot be read."""
    path = Path(path)
    ext = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    try:
        if ext in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif ext == ".json":
            data = json.loads(text)
        elif ext == ".toml":
            data = tomllib.loads(text)
        else:
            raise ConfigError(f"{path}: configuration files must be .yaml, .yml, .json or .toml")
    except (yaml.YAMLError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: cannot parse the file ({exc})") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the configuration must be a mapping")
    return data


def _voxels(data) -> VoxelSpec:
    _check_keys("voxels", data, {"shape", "voxel_size", "supersampling", "section"})
    if ("shape" in data) == ("voxel_size" in data):
        raise ConfigError("voxels: give exactly one of 'shape' and 'voxel_size'")
    shape = data.get("shape")
    if shape is not None and not isinstance(shape, int):
        shape = tuple(shape)
    ss = data.get("supersampling", 1)
    if isinstance(ss, bool) or not isinstance(ss, int) or not 1 <= ss <= 6:
        raise ConfigError("voxels.supersampling: an integer between 1 and 6")
    section = data.get("section")
    if section is not None:
        if section not in ("x", "y", "z"):
            raise ConfigError(
                "voxels.section: 'x', 'y' or 'z', the normal of the layer of voxels through "
                "the middle of the realization that the run studies (a planar section)"
            )
        section = "xyz".index(section)
    return VoxelSpec(shape=shape, voxel_size=data.get("voxel_size"), supersampling=ss,
                     section=section)  # fmt: skip


def _positive_float(section: str, value, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{section}: expected a positive number, got {value!r}")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{section}: expected a positive number, got {value!r}") from None
    if not x > 0 or x == float("inf"):
        raise ConfigError(f"{section}: expected a positive number, got {value!r}")
    return x


def _mesh(data) -> MeshSpec:
    allowed = {
        "size",
        "min_size",
        "hausdorff",
        "gradation",
        "background",
        "order",
        "curved",
        "sharp_edges",
        "phases",
        "grains",
        "min_feature",
        "cells",
        "method",
        "plane",
        "z",
    }
    _check_keys("mesh", data, allowed, {"size"})
    sharp_edges = data.get("sharp_edges")
    if sharp_edges is not None and not isinstance(sharp_edges, bool):
        raise ConfigError(
            "mesh.sharp_edges: true (the mesh follows the sharp edges of the phases: the "
            "default of the structures that give their smooth pieces) or false (rounded)"
        )
    grains = data.get("grains")
    if grains is not None and not isinstance(grains, bool):
        raise ConfigError("mesh.grains: true (one subdomain per grain of a tessellation) or false")
    min_feature = data.get("min_feature")
    if min_feature is not None:
        try:
            min_feature = float(min_feature)
        except (TypeError, ValueError):
            min_feature = -1.0
        if not 0.0 <= min_feature < float("inf"):
            raise ConfigError("mesh.min_feature: a non-negative length (0 keeps the exact cells)")
    order = data.get("order")
    if order is not None and (order not in (1, 2) or isinstance(order, bool)):
        raise ConfigError(
            "mesh.order: 1 (4-node tetrahedra) or 2 (10-node tetrahedra; the default of "
            "the meshes analysed by finite elements)"
        )
    curved = data.get("curved")
    if curved is not None and not isinstance(curved, bool):
        raise ConfigError(
            "mesh.curved: true (10-node elements whose edges follow the interfaces, the "
            "default) or false (straight edges)"
        )
    if curved and order == 1:
        raise ConfigError("mesh.curved: the edges of 10-node elements (mesh.order: 2)")
    phases = data.get("phases", "auto")
    if phases not in ("auto", "all", "solid"):
        raise ConfigError("mesh.phases: 'auto', 'all' or 'solid'")
    cells = data.get("cells", "whole")
    if cells not in ("whole", "box"):
        raise ConfigError(
            "mesh.cells: 'whole' (whole periodic grains) or 'box' (grains cut by the box)"
        )
    method = data.get("method", "auto")
    if method not in ("auto", "exact", "grid"):
        raise ConfigError(
            "mesh.method: 'auto' (the exact geometry when the model has one, the grid "
            "otherwise), 'exact' (the exact geometry, without a grid) or 'grid' (the level "
            "set on a background grid)"
        )
    gradation = _positive_float("mesh.gradation", data.get("gradation", 2.0))
    if gradation < 1.0:
        raise ConfigError("mesh.gradation: a number >= 1 (size ratio of neighbouring elements)")
    plane = data.get("plane", "auto")
    if plane == "auto":
        plane = None
    elif not isinstance(plane, bool):
        raise ConfigError(
            "mesh.plane: true (triangles of the plane: the columnar grains of a tessellation "
            "of coplanar germs, the section of any structure at mesh.z), false (tetrahedra) "
            "or auto (the triangles of those tessellations)"
        )
    z = data.get("z")
    if z is not None:
        if plane is False:
            raise ConfigError("mesh.z: the height of the sections of the plane (mesh.plane)")
        try:
            z = float(z)
        except (TypeError, ValueError):
            raise ConfigError("mesh.z: the height of the plane of the sections") from None
    return MeshSpec(
        size=_positive_float("mesh.size", data["size"]),
        min_size=_positive_float("mesh.min_size", data.get("min_size"), optional=True),
        hausdorff=_positive_float("mesh.hausdorff", data.get("hausdorff"), optional=True),
        gradation=gradation,
        background=_positive_float("mesh.background", data.get("background"), optional=True),
        order=None if order is None else int(order),
        curved=curved,
        sharp_edges=sharp_edges,
        phases=phases,
        grains=grains,
        min_feature=min_feature,
        cells=cells,
        method=method,
        plane=plane,
        z=z,
    )


def _outputs(data, base: Path) -> OutputSpec:
    allowed = {
        "directory",
        "name",
        "geometry",
        "image",
        "labels",
        "level_set",
        "mesh",
        "conforming_mesh",
        "surface",
        "periodic_pairs",
        "periodic_equations",
        "damask",
        "neper",
        "database",
        "amitex",
        "code_aster",
        "openfoam",
        "tables",
    }
    _check_keys("outputs", data, allowed)
    mesh = data.get("mesh")
    phases = None
    if isinstance(mesh, dict):
        _check_keys("outputs.mesh", mesh, {"formats", "phases"})
        phases = mesh.get("phases")
        mesh = mesh.get("formats", [])
    directory = Path(data.get("directory", "morphogen_output"))
    if not directory.is_absolute():
        directory = base / directory
    database = data.get("database")
    if database is not None:
        database = Path(database)
        database = database if database.is_absolute() else base / database
    return OutputSpec(
        amitex=_amitex(data.get("amitex")),
        code_aster=_code_aster(data.get("code_aster")),
        openfoam=_openfoam(data.get("openfoam")),
        directory=directory,
        name=str(data.get("name", "sample")),
        geometry=_list_of("outputs.geometry", data.get("geometry"), GEOMETRY_FORMATS),
        image=_list_of("outputs.image", data.get("image"), IMAGE_FORMATS),
        labels=_list_of("outputs.labels", data.get("labels"), IMAGE_FORMATS),
        level_set=_list_of("outputs.level_set", data.get("level_set"), IMAGE_FORMATS),
        mesh=_list_of("outputs.mesh", mesh, MESH_FORMATS),
        mesh_phases=None if phases is None else [int(p) for p in phases],
        conforming_mesh=_list_of(
            "outputs.conforming_mesh", data.get("conforming_mesh"), MESH_FORMATS
        ),
        periodic_pairs=bool(data.get("periodic_pairs", False)),
        periodic_equations=bool(data.get("periodic_equations", False)),
        damask=bool(data.get("damask", False)),
        neper=bool(data.get("neper", False)),
        surface=_list_of("outputs.surface", data.get("surface"), SURFACE_FORMATS),
        database=database,
        tables=[
            t for t in _list_of("outputs.tables", data.get("tables"), TABLE_FORMATS) if t != "csv"
        ],
    )


def _mfront_paths(materials, base: Path):
    """The materials with the sources and libraries of their MFront behaviours, their Z-mat
    files and the files of their learned laws, relative to the directory of the
    configuration file."""
    if not isinstance(materials, dict):
        return materials
    out = {}
    for key, spec in materials.items():
        zm = spec.get("zmat") if isinstance(spec, dict) else None
        if zm is not None:  # Z-mat files relative to the configuration
            zm = {"file": zm} if isinstance(zm, str) else dict(zm)
            if "file" in zm and not Path(str(zm["file"])).is_absolute():
                zm["file"] = str(base / str(zm["file"]))
            spec = {**spec, "zmat": zm}
        mf = spec.get("mfront") if isinstance(spec, dict) else None
        if isinstance(mf, str):
            mf = {"source" if mf.endswith(".mfront") else "library": mf}
        if isinstance(mf, dict):
            mf = dict(mf)
            for name in ("source", "library"):
                if name in mf and not Path(str(mf[name])).is_absolute():
                    mf[name] = str(base / str(mf[name]))
            spec = {**spec, "mfront": mf}
        ln = spec.get("learned") if isinstance(spec, dict) else None
        model = ln.get("model") if isinstance(ln, dict) else ln
        if isinstance(model, str) and not Path(model).is_absolute():  # a learned law's file
            model = str(base / model)
            spec = {**spec, "learned": {**ln, "model": model} if isinstance(ln, dict) else model}
        out[key] = spec
    return out


def _diffusion(data) -> DiffusionSpec:
    """The ``diffusion`` section of a configuration."""
    from morphogen.coupling.diffusion import FACES

    if not isinstance(data, dict):
        raise ConfigError("diffusion: a mapping (phases, exposed, activity, times...)")
    _check_keys(
        "diffusion",
        data,
        {"phases", "exposed", "activity", "initial", "times", "steps", "fields"},
        {"phases", "times"},
    )
    phases = data["phases"]
    if not isinstance(phases, dict) or not phases:
        raise ConfigError(
            "diffusion.phases: a mapping phase -> {diffusivity, solubility, consumption}"
        )
    diff, sol, cons = {}, {}, {}
    for key, props in phases.items():
        name = f"diffusion.phases.{key}"
        if not isinstance(props, dict):
            raise ConfigError(f"{name}: a mapping (diffusivity, solubility, consumption)")
        _check_keys(name, props, {"diffusivity", "solubility", "consumption"}, {"diffusivity"})
        p = int(key)
        diff[p] = float(props["diffusivity"])
        sol[p] = float(props.get("solubility", 1.0))
        cons[p] = float(props.get("consumption", 0.0))
        if diff[p] < 0 or cons[p] < 0:
            raise ConfigError(f"{name}: non-negative diffusivity and consumption")
        if sol[p] <= 0:
            raise ConfigError(f"{name}.solubility: positive (an impermeable phase: diffusivity 0)")
    exposed = tuple(str(f) for f in (data.get("exposed") or ("x-", "x+")))
    wrong = [f for f in exposed if f not in FACES]
    if wrong:
        raise ConfigError(f"diffusion.exposed: faces among {', '.join(FACES)}, got {wrong}")
    activity = data.get("activity", 1.0)
    if isinstance(activity, dict):
        _check_keys("diffusion.activity", activity, {"times", "values"}, {"times", "values"})
        t = tuple(float(v) for v in activity["times"])
        v = tuple(float(x) for x in activity["values"])
        if len(t) != len(v) or len(t) < 2 or any(b <= a for a, b in itertools.pairwise(t)):
            raise ConfigError("diffusion.activity: increasing times and one value per time")
        activity = (t, v)
    else:
        activity = float(activity)
    times = tuple(
        float(t) for t in (data["times"] if isinstance(data["times"], list) else [data["times"]])
    )
    if not times or times[0] <= 0 or any(b <= a for a, b in itertools.pairwise(times)):
        raise ConfigError("diffusion.times: positive, increasing output times")
    steps = _positive_int("diffusion.steps", data.get("steps", 40))
    return DiffusionSpec(
        diff,
        sol,
        cons,
        exposed,
        activity,
        float(data.get("initial", 0.0)),
        times,
        steps,
        bool(data.get("fields", False)),
    )


def _integration(name: str, section) -> str:
    """The integration of a section: 'full' or 'bbar' ('reduced': the same)."""
    from morphogen.homogenization.fe import integration_of

    try:
        return integration_of(section["integration"])
    except ValueError:
        raise ConfigError(
            f"{name}: 'full' (every point of the elements) or 'bbar' (also 'reduced': the "
            "volume change of each element as its mean, B-bar and F-bar; for nearly "
            "incompressible phases and plastic flow)"
        ) from None


def _homogenization(data, base: Path | None = None) -> HomogenizationSpec:
    allowed = {
        "materials",
        "method",
        "boundary_conditions",
        "properties",
        "solver",
        "tolerance",
        "max_iterations",
        "fields",
        "loadings",
        "paths",
        "nonlinear",
        "device",
        "ntfa",
        "mean_temperature",
        "frequencies",
        "wavelengths",
        "fracture",
        "fluid_bulk_modulus",
        "integration",
        "fatigue",
        "beremin",
        "plane",
    }
    _check_keys("homogenization", data, allowed, {"materials"})
    method = data.get("method", "auto")
    if method not in HOMOGENIZATION_METHODS:
        raise ConfigError(
            "homogenization.method: 'fe' (finite elements on conforming meshes), 'fft' "
            "(voxel images, periodic), 'xfem' (extended finite elements on the grid of the "
            "voxels section, cut by the level set; periodic) or 'auto'"
        )
    try:
        raw = data["materials"] if base is None else _mfront_paths(data["materials"], base)
        materials = materials_from_config(raw)
    except ValueError as exc:
        raise ConfigError(f"homogenization.{exc}") from None
    bcs_raw = data.get("boundary_conditions", ["periodic"])
    bcs_raw = [bcs_raw] if isinstance(bcs_raw, str) else list(bcs_raw)
    bcs = []
    for b in bcs_raw:
        try:
            name = boundary_condition(b)
        except ValueError as exc:
            raise ConfigError(f"homogenization.boundary_conditions: {exc}") from None
        if name not in bcs:
            bcs.append(name)
    if not bcs:
        raise ConfigError("homogenization.boundary_conditions: give at least one")
    paths = _paths(data.get("paths"))
    props = _list_of(
        "homogenization.properties",
        data.get("properties", [] if paths else ["bulk_modulus", "shear_modulus"]),
        tuple(PHYSICS),
    )
    if not props and not paths:
        raise ConfigError("homogenization.properties: give at least one (or loading paths)")
    solver = data.get("solver", "auto")
    if solver not in SOLVERS:
        raise ConfigError(f"homogenization.solver: one of {list(SOLVERS)}")
    tol = float(data.get("tolerance", 1e-8))
    if not 0.0 < tol < 1.0:
        raise ConfigError("homogenization.tolerance: a number in (0, 1)")
    maxiter = data.get("max_iterations")
    if maxiter is not None:
        maxiter = _positive_int("homogenization.max_iterations", maxiter)
    if "subc" in bcs and any(m.void for m in materials.values()):
        raise ConfigError(
            "homogenization: static uniform (subc) conditions cannot be applied to porous "
            "media (Dirrenberger et al. 2014); use 'mixed'"
        )
    loadings_raw = data.get("loadings") or {}
    if not isinstance(loadings_raw, dict):
        raise ConfigError("homogenization.loadings: a mapping name -> loading")
    loadings = {}
    for name, spec in loadings_raw.items():
        name = str(name)
        if name in PHYSICS or "." in name:
            raise ConfigError(f"homogenization.loadings: {name!r} is not a valid name")
        try:
            loadings[name] = Loading.from_spec(spec)
        except ValueError as exc:
            raise ConfigError(f"homogenization.loadings.{name}: {exc}") from None
    twice = sorted(set(loadings) & set(paths))
    if twice:
        raise ConfigError(f"homogenization: {twice} name both a loading and a path")
    nonlinear = data.get("nonlinear") or {}
    if not isinstance(nonlinear, dict):
        raise ConfigError("homogenization.nonlinear: a mapping of options")
    _check_keys("homogenization.nonlinear", nonlinear, set(NONLINEAR_OPTIONS), set())
    nl = {}
    if "tolerance" in nonlinear:
        nl["tolerance"] = float(nonlinear["tolerance"])
        if not 0.0 < nl["tolerance"] < 1.0:
            raise ConfigError("homogenization.nonlinear.tolerance: a number in (0, 1)")
    for key in ("max_iterations", "max_cuts"):
        if key in nonlinear:
            nl[key] = _positive_int(f"homogenization.nonlinear.{key}", nonlinear[key])
    if "kinematics" in nonlinear:
        nl["kinematics"] = str(nonlinear["kinematics"])
        if nl["kinematics"] not in ("small", "finite"):
            raise ConfigError("homogenization.nonlinear.kinematics: 'small' or 'finite' strains")
    if "integration" in nonlinear:
        nl["integration"] = _integration("homogenization.nonlinear.integration", nonlinear)
    if "tangent" in nonlinear:
        nl["tangent"] = str(nonlinear["tangent"])
        if nl["tangent"] not in ("auto", "symmetric", "full"):
            raise ConfigError(
                "homogenization.nonlinear.tangent: 'auto', 'symmetric' (symmetrized, "
                "conjugate gradients) or 'full' (GMRES)"
            )
    if nonlinear.get("microdamage") is not None:
        nl["microdamage"] = _microdamage(
            "homogenization.nonlinear.microdamage", nonlinear["microdamage"]
        )
    if nonlinear.get("contact") not in (None, False):
        nl["contact"] = _contact("homogenization.nonlinear.contact", nonlinear["contact"])
        if not paths:
            raise ConfigError(
                "homogenization.nonlinear.contact: the walls of the pores touch along the "
                "loading paths (homogenization.paths)"
            )
        # walls that close move far: finite strains, unless the configuration says otherwise
        nl.setdefault("kinematics", "finite")
    if nonlinear.get("interfaces") not in (None, False):
        name = "homogenization.nonlinear.interfaces"
        nl["interfaces"] = _interfaces(name, nonlinear["interfaces"])
        if not paths:
            raise ConfigError(
                f"{name}: the interfaces debond along the loading paths (homogenization.paths)"
            )
        if "contact" in nl:
            raise ConfigError(
                f"{name}: the interfaces of the phases, or the contact of the walls of the "
                "pores (homogenization.nonlinear.contact), not both"
            )
    if paths and not ({"periodic", "kubc"} & set(bcs)):
        raise ConfigError(
            "homogenization.paths: loading paths are solved under periodic or kubc conditions"
        )
    device = str(data.get("device", "cpu"))
    if device not in DEVICES:
        raise ConfigError(f"homogenization.device: one of {list(DEVICES)} (the FFT solver)")
    ntfa = _ntfa(data.get("ntfa"), paths, base)
    mean_temperature = float(data.get("mean_temperature", 300.0))
    if not mean_temperature > 0.0:
        raise ConfigError("homogenization.mean_temperature: an absolute temperature (K)")
    spectra = {}
    for key, needed in (("frequencies", ("dynamic_stiffness", "complex_conductivity")),
                        ("wavelengths", ("refractive_index",))):  # fmt: skip
        values = data.get(key)
        if values is not None:
            values = [values] if np.ndim(values) == 0 else list(values)
            try:
                values = [float(v) for v in values]
            except (TypeError, ValueError):
                values = [-1.0]
            if not values or min(values) <= 0.0:
                raise ConfigError(f"homogenization.{key}: positive values")
        elif any(p in props for p in needed):
            asked = [p for p in needed if p in props]
            raise ConfigError(f"homogenization.{key}: those of the properties {asked}")
        spectra[key] = values
    fracture = data.get("fracture")
    if fracture is not None:
        from morphogen.fracture.phasefield import PhaseField

        if not isinstance(fracture, dict):
            raise ConfigError("homogenization.fracture: a mapping (toughness, length, ...)")
        solver_keys = {"staggered_tol", "max_staggered", "tol"}
        try:
            PhaseField.from_config({k: v for k, v in fracture.items() if k not in solver_keys})
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"homogenization.fracture: {exc}") from None
        if not paths:
            raise ConfigError(
                "homogenization.fracture: the loading paths along which the samples break "
                "(homogenization.paths)"
            )
        fracture = dict(fracture)
    fluid = data.get("fluid_bulk_modulus")
    if fluid is not None:
        try:
            fluid = float(fluid)
        except (TypeError, ValueError):
            fluid = -1.0
        if not fluid > 0.0:
            raise ConfigError("homogenization.fluid_bulk_modulus: a positive bulk modulus")
        if "poroelastic" not in props:
            raise ConfigError(
                "homogenization.fluid_bulk_modulus: that of the property 'poroelastic'"
            )
    if "poroelastic" in props:
        if not any(m.void for m in materials.values()):
            raise ConfigError(
                "homogenization.properties: 'poroelastic' needs pores (a material 'void')"
            )
        if not {"periodic", "kubc"} & set(bcs):
            raise ConfigError(
                "homogenization.properties: 'poroelastic' is computed under periodic or kubc "
                "conditions"
            )
    flows = [p for p in props if PHYSICS[p] == "flow"]
    if flows and not any(m.void for m in materials.values()):
        raise ConfigError(
            f"homogenization.properties: {flows} need pores (a material 'void'): the fluid "
            "flows through them"
        )
    if "swelling" in props and not any(m.swelling is not None for m in materials.values()):
        raise ConfigError(
            "homogenization.properties: 'swelling' needs the swelling of a material "
            "(materials: {phase: {swelling: ...}})"
        )
    return HomogenizationSpec(
        materials=materials,
        method=method,
        boundary_conditions=bcs,
        properties=props,
        solver=solver,
        tolerance=tol,
        max_iterations=maxiter,
        fields=bool(data.get("fields", False)),
        loadings=loadings,
        paths=paths,
        nonlinear=nl,
        device=device,
        ntfa=ntfa,
        mean_temperature=mean_temperature,
        fracture=fracture,
        fluid_bulk_modulus=fluid,
        integration=_integration("homogenization.integration", data)
        if data.get("integration") is not None
        else None,
        fatigue=_fatigue(data.get("fatigue"), paths, materials),
        beremin=_beremin(data.get("beremin"), paths),
        plane=_plane_hypothesis(data.get("plane")),
        **spectra,
    )


def _meshes_of_the_plane(mesh: MeshSpec | None, model) -> bool:
    """Whether the meshes of a run are those of the plane: asked, or by default for the
    models whose realizations are columnar grains (the seeds of the plane, the orientation
    maps)."""
    if mesh is not None and mesh.plane is not None:
        return bool(mesh.plane)
    from morphogen.models.ebsd import EBSD
    from morphogen.models.seeded import SeededLaguerre

    return isinstance(model, EBSD) or (isinstance(model, SeededLaguerre) and model.dimension == 2)


def _check_plane(mesh: MeshSpec, homog, outputs, viz) -> None:
    """What the meshes of the plane do not do (``mesh.plane: true``): the tetrahedra of the
    files of Code_Aster and of OpenFOAM; the loading paths without periodic conditions; the
    reduced models, contacts and interfaces; the videos of the mesh and of its fields."""
    other = [f for f in outputs.conforming_mesh if f in ("mail", "med")]
    reasons = []
    if other:
        reasons.append(f"outputs.conforming_mesh {other} (Code_Aster's tetrahedra)")
    if outputs.code_aster is not None:
        reasons.append("outputs.code_aster")
    if outputs.openfoam is not None and outputs.openfoam.get("sample") == "mesh":
        reasons.append("outputs.openfoam of the mesh")
    if homog is not None and homog.paths and "periodic" not in homog.boundary_conditions:
        reasons.append("loading paths without periodic conditions (on the layer in space)")
    if homog is not None and homog.ntfa:
        reasons.append("the reduced models")
    if homog is not None and {"contact", "interfaces"} & set(homog.nonlinear):
        reasons.append("the contacts and the interfaces")
    if homog is not None and "mixed" in homog.boundary_conditions:
        reasons.append("the mixed conditions")
    if viz is not None and viz.videos and any(v.of != "image" and v.of != "labels"
                                              for v in viz.videos):  # fmt: skip
        reasons.append("the videos of the mesh and of its fields")
    if reasons:
        raise ConfigError(
            "mesh.plane: the meshes of the plane do not take "
            + "; ".join(reasons)
            + " (mesh.plane: false for tetrahedra)"
        )


def _plane_hypothesis(value) -> str | None:
    from morphogen.homogenization.plane import PLANES

    if value is None or value in PLANES:
        return value
    raise ConfigError(
        "homogenization.plane: 'generalized' (generalized plane strain: the anti-plane "
        "fluctuation free, the default) or 'strain' (plane strain), for the meshes of the "
        "plane"
    )


#: The scalar results of the fatigue of a cycle (quantities of its loading path)
FATIGUE_QUANTITIES = ("fatigue_utilization", "fatigue_safety", "fatigue_beyond")


def _beremin(value, paths: dict) -> dict | None:
    """The criterion of cleavage of Beremin along a loading path (``homogenization.
    beremin``): its ``modulus``, ``scale`` and reference ``volume``, the cleavage
    ``planes`` of the crystals and the ``plastic`` strain of the plastic zone; the
    ``path`` (the only one, when left out)."""
    from morphogen.homogenization.beremin import Beremin

    if value is None:
        return None
    name = "homogenization.beremin"
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: a mapping (modulus, scale, volume, planes, plastic, path)")
    if not paths:
        raise ConfigError(f"{name}: the fields of a loading path (homogenization.paths)")
    value = dict(value)
    path = value.pop("path", None)
    if path is None and len(paths) != 1:
        raise ConfigError(f"{name}.path: one of the loading paths {list(paths)}")
    path = next(iter(paths)) if path is None else str(path)
    if path not in paths:
        raise ConfigError(f"{name}.path: one of the loading paths {list(paths)}, got {path!r}")
    try:
        Beremin.from_config(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name}: {str(exc).removeprefix('beremin: ')}") from None
    return {"path": path, "criterion": value}


def _fatigue(value, paths: dict, materials: dict) -> dict | None:
    """The fatigue of the points along a cycle of a loading path (``homogenization.
    fatigue``): one criterion and its limits for the solid phases (or those of
    ``phases``), or the ``criteria`` of the phases; the ``path`` (the only one, when
    left out) and the times of its ``cycle`` (the whole path)."""
    from morphogen.homogenization.fatigue import criterion_from_config

    if value is None:
        return None
    name = "homogenization.fatigue"
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: a mapping (criterion, bending, torsion, path, cycle...)")
    if not paths:
        raise ConfigError(f"{name}: the cycle of a loading path (homogenization.paths)")
    value = dict(value)
    path = value.pop("path", None)
    if path is None and len(paths) != 1:
        raise ConfigError(f"{name}.path: one of the loading paths {list(paths)}")
    path = next(iter(paths)) if path is None else str(path)
    if path not in paths:
        raise ConfigError(f"{name}.path: one of the loading paths {list(paths)}, got {path!r}")
    cycle = value.pop("cycle", None)
    steps = np.concatenate([[paths[path].times[0]], paths[path].schedule()])
    if cycle is not None:
        try:
            start, end = (float(v) for v in cycle)
        except (TypeError, ValueError):
            raise ConfigError(f"{name}.cycle: the times of its start and of its end") from None
        if not steps[0] <= start < end <= steps[-1]:
            raise ConfigError(
                f"{name}.cycle: two times of the path {path!r}, from {steps[0]:g} to "
                f"{steps[-1]:g}, the first before the second"
            )
        cycle = (start, end)
        steps = steps[(steps >= start - 1e-9) & (steps <= end + 1e-9)]
    if steps.size < 3:
        raise ConfigError(
            f"{name}: a cycle of two steps of the path {path!r} at least (its fields are "
            "kept at the ends of its steps)"
        )
    solid = sorted(p for p, m in materials.items() if not m.void)
    criteria = value.pop("criteria", None)
    phases = value.pop("phases", None)
    try:
        if criteria is not None:
            if value or phases is not None or not isinstance(criteria, dict) or not criteria:
                raise ValueError
            specs = {int(p): dict(c) for p, c in criteria.items()}
        else:
            chosen = solid if phases is None else np.atleast_1d(phases).tolist()
            specs = {int(p): dict(value) for p in chosen}
    except (TypeError, ValueError):
        raise ConfigError(
            f"{name}: one criterion and its limits (for the solid phases, or 'phases'), or "
            "'criteria', a mapping of the phases to theirs"
        ) from None
    absent = sorted(set(specs) - set(solid))
    if absent:
        raise ConfigError(f"{name}: {absent} are not solid phases of the materials {solid}")
    for phase, spec in specs.items():
        try:
            criterion_from_config(spec)
        except ValueError as exc:
            raise ConfigError(f"{name} (phase {phase}): {exc}") from None
    return {"path": path, "cycle": cycle, "criteria": specs}


def _ntfa(raw, paths, base: Path | None) -> dict:
    """Options of the reduced model (NTFA) that follows the loading paths."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("homogenization.ntfa: a mapping (training, modes, save, full_field)")
    _check_keys(
        "homogenization.ntfa", raw, {"training", "modes", "save", "full_field"}, {"training"}
    )
    if not paths:
        raise ConfigError("homogenization.ntfa: give the paths to follow (homogenization.paths)")
    try:
        training = _paths(raw["training"])
    except ConfigError as exc:
        raise ConfigError(
            str(exc).replace("homogenization.paths", "homogenization.ntfa.training")
        ) from None
    if not training:
        raise ConfigError("homogenization.ntfa.training: at least one path")
    modes = raw.get("modes", 6)
    if isinstance(modes, dict):
        modes = {
            int(k): _positive_int(f"homogenization.ntfa.modes.{k}", v) for k, v in modes.items()
        }
    else:
        modes = _positive_int("homogenization.ntfa.modes", modes)
    out = {"training": training, "modes": modes, "full_field": bool(raw.get("full_field", False))}
    if raw.get("save"):
        save = Path(str(raw["save"]))
        out["save"] = str(save if base is None or save.is_absolute() else base / save)
    return out


def _paths(raw) -> dict[str, LoadingPath]:
    """Loading paths of a configuration (name -> path)."""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError("homogenization.paths: a mapping name -> path")
    out = {}
    for name, spec in raw.items():
        name = str(name)
        if name in PHYSICS or "." in name:
            raise ConfigError(f"homogenization.paths: {name!r} is not a valid name")
        try:
            out[name] = LoadingPath.from_spec(spec)
        except ValueError as exc:
            raise ConfigError(f"homogenization.paths.{name}: {exc}") from None
    return out


def _rve_properties(value, homog: HomogenizationSpec | None) -> list[str]:
    """RVE properties: measures, apparent properties (all their BCs) or '<prop>.<bc>', the
    stiffness ('stiffness' or 'stiffness.<bc>': its mean and compliance per volume) and the
    conductivity tensor ('conductivity_tensor', '.<bc>') like the tensors of the other
    transport phenomena, quantities of the curves of the loading paths
    ('<path>.<quantity>[@<time>][.<bc>]': a column, or 'strength' and 'work') and scalar
    results of the strong couplings and of the harmonic properties
    ('<property>.<entry>[.<bc>]': 'piezoelectric.e33', 'dynamic_stiffness.shear_loss_1')."""
    series = homog.series() if homog is not None else []
    tensors = homog.tensors() if homog is not None else []
    if value is None:
        return ["volume_fraction", *series, *tensors]
    items = [value] if isinstance(value, str) else list(value)
    out = []
    for item in items:
        item = str(item)
        if item in MEASURES or item in series or item in tensors:
            names = [item]
        elif homog is not None and item in homog.properties and item in PROPERTIES:
            names = [f"{item}.{bc}" for bc in homog.boundary_conditions]
        elif item in TENSOR_PROPERTIES and any(t.startswith(f"{item}.") for t in tensors):
            names = [t for t in tensors if t.startswith(f"{item}.")]
        elif homog is not None and "." in item and item.partition(".")[0] in homog.paths:
            names = [_path_quantity(item, homog)]
        elif homog is not None and item.partition(".")[0] in _entry_properties(homog):
            names = [_property_entry(item, homog)]
        else:
            heads = [p for p in TENSOR_PROPERTIES if any(t.startswith(f"{p}.") for t in tensors)]
            allowed = list(MEASURES) + series + heads + tensors
            if homog is not None and homog.paths:
                allowed.append("<path>.<quantity>[@<time>][.<bc>] (e.g. tension.S11@5)")
            for prop in _entry_properties(homog) if homog is not None else ():
                allowed.append(
                    f"{prop}.<entry>[.<bc>] (e.g. {prop}.{_entry_names(prop, homog)[0]})"
                )
            raise ConfigError(f"rve.properties: unknown value {item!r}; allowed: {allowed}")
        out.extend(n for n in names if n not in out)
    return out


def path_quantities(names, homog: HomogenizationSpec | None) -> list[PathQuantity]:
    """The quantities of the loading paths of ``homog`` among the RVE properties ``names``."""
    if homog is None or not homog.paths:
        return []
    return [PathQuantity.parse(n) for n in names if "." in n and n.partition(".")[0] in homog.paths]


def _entry_properties(homog: HomogenizationSpec) -> list[str]:
    """The properties computed whose results are named scalars: the strong couplings and
    the harmonic properties."""
    named = ("thermal_expansion", "swelling", "poroelastic")
    families = ("coupled", "harmonic", "flow")
    return [p for p in homog.properties if PHYSICS.get(p) in families or p in named]


def _entry_names(prop: str, homog: HomogenizationSpec) -> tuple[str, ...]:
    """The scalar results of a coupled or harmonic property of ``homog``, of its
    poroelastic law and of its tensors of thermal expansion and of swelling."""
    if prop == "poroelastic":
        from morphogen.homogenization.poroelastic import POROELASTIC_ENTRIES, UNDRAINED_ENTRIES

        fluid = homog.fluid_bulk_modulus is not None
        return (*POROELASTIC_ENTRIES, *(UNDRAINED_ENTRIES if fluid else ()))
    if prop in ("thermal_expansion", "swelling"):
        letter = "A" if prop == "thermal_expansion" else "B"
        return (*(f"{letter}{name}" for name in VOIGT_NAMES), "mean")
    if prop == "hydraulic_permeability":
        from morphogen.homogenization.flow import ENTRIES

        return ENTRIES
    if prop == "acoustic":
        from morphogen.homogenization.acoustics import PARAMETERS

        return PARAMETERS
    if PHYSICS[prop] == "coupled":
        from morphogen.homogenization.coupled import entry_names

        return entry_names(prop)
    from morphogen.homogenization.harmonic import entry_names

    values = homog.wavelengths if prop == "refractive_index" else homog.frequencies
    return entry_names(prop, np.size(values))


def _property_entry(item: str, homog: HomogenizationSpec) -> str:
    """Canonical name ``<property>.<entry>[.<bc>]`` of a scalar result of a strong coupling
    or of a harmonic property, checked against the results of the property."""
    prop, _, rest = item.partition(".")
    entry, _, bc = rest.partition(".")
    names = _entry_names(prop, homog)
    if entry not in names:
        shown = ", ".join(names[:12]) + (" ..." if len(names) > 12 else "")
        raise ConfigError(
            f"rve.properties: {item!r}: unknown result {entry!r} of {prop}; use one of {shown}"
        )
    if PHYSICS[prop] == "elastic":  # under the conditions of the elastic problems
        solved = [b for b in homog.boundary_conditions if b != "mixed"]
        if prop == "poroelastic":
            solved = [b for b in solved if b != "subc"]
        if bc:
            try:
                bc = boundary_condition(bc)
            except ValueError as exc:
                raise ConfigError(f"rve.properties: {item!r}: {exc}") from None
            if bc not in solved:
                raise ConfigError(
                    f"rve.properties: {item!r}: {prop} is computed under {solved} conditions"
                )
        return f"{prop}.{entry}" + (f".{bc}" if bc else "")
    if bc and bc != "periodic":
        raise ConfigError(f"rve.properties: {item!r}: {prop} is solved under periodic conditions")
    return f"{prop}.{entry}" + (".periodic" if bc else "")


def property_entries(names, homog: HomogenizationSpec | None) -> list[tuple[str, str]]:
    """The scalar results ``(property, entry)`` of the strong couplings and of the harmonic
    properties of ``homog`` among the RVE properties ``names``."""
    if homog is None:
        return []
    props = _entry_properties(homog)
    out = []
    for name in names:
        prop, _, rest = str(name).partition(".")
        if prop in props and rest:
            out.append((prop, rest.partition(".")[0]))
    return out


def _microdamage(name: str, value) -> dict:
    """The options of the alternate scheme of the microdamage of regularized cleavage
    planes (``homogenization.nonlinear.microdamage``): its ``tolerance``, the most
    ``passes`` of a step, its ``acceleration``."""
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: a mapping (tolerance, passes, acceleration)")
    _check_keys(name, value, {"tolerance", "passes", "acceleration"}, set())
    out = {}
    if "tolerance" in value:
        tolerance = value["tolerance"]
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise ConfigError(f"{name}.tolerance: a number in (0, 1)")
        if not 0.0 < tolerance < 1.0:
            raise ConfigError(f"{name}.tolerance: a number in (0, 1)")
        out["tolerance"] = float(tolerance)
    if "passes" in value:
        out["passes"] = _positive_int(f"{name}.passes", value["passes"])
    if "acceleration" in value:
        if not isinstance(value["acceleration"], bool):
            raise ConfigError(f"{name}.acceleration: true or false")
        out["acceleration"] = value["acceleration"]
    return out


def _contact(name: str, value) -> dict:
    """The options of the contact of the walls of the pores along the loading paths
    (``true``, or a mapping of them): those of morphogen.contact.periodic.ContactCell."""
    if value is True:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: true, or a mapping of its options {list(CONTACT_OPTIONS)}")
    _check_keys(name, value, set(CONTACT_OPTIONS), set())
    out = {}
    for key in ("friction", "stiffness", "tangential", "depth", "tolerance"):
        if key not in value:
            continue
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ConfigError(f"{name}.{key}: a number, got {number!r}")
        out[key] = float(number)
        if key == "friction" and not (np.isfinite(out[key]) and out[key] >= 0):
            raise ConfigError(f"{name}.friction: Coulomb's coefficient, 0 or more")
        if key != "friction" and not (np.isfinite(out[key]) and out[key] > 0):
            raise ConfigError(f"{name}.{key}: a positive number")
    if "augmentations" in value:
        count = value["augmentations"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ConfigError(f"{name}.augmentations: a number of updates of the multipliers")
        out["augmentations"] = count
    if "points" in value:
        if value["points"] not in ("gauss", "nodes"):
            raise ConfigError(
                f"{name}.points: 'gauss' (inside the faces) or 'nodes' (their nodes: corners "
                f"and edges that touch), got {value['points']!r}"
            )
        out["points"] = value["points"]
    return out


#: The options of the cell of the interfaces, beside their law
_CELL_OPTIONS = ("faces", "viscosity", "slip")


def _interface_numbers(name: str, value: dict, keys) -> dict:
    """The numbers ``keys`` of the options of the interfaces: positive, or of 0 or more
    for the friction and the viscosity."""
    out = {}
    for key in keys:
        if key not in value or value[key] is None:
            continue
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ConfigError(f"{name}.{key}: a number, got {number!r}")
        out[key] = float(number)
        if key in ("friction", "viscosity"):
            if not (np.isfinite(out[key]) and out[key] >= 0):
                raise ConfigError(f"{name}.{key}: a number of 0 or more")
        elif not (np.isfinite(out[key]) and out[key] > 0):
            raise ConfigError(f"{name}.{key}: a positive number")
    return out


def _interface_law(name: str, value, extra=()) -> dict:
    """A law of interfaces: its kind and its parameters (morphogen.fracture.periodic.
    interface_law); ``extra``: the other keys allowed beside them."""
    from morphogen.fracture.periodic import LAW_OPTIONS

    if not isinstance(value, dict):
        raise ConfigError(f"{name}: a mapping of the law of the interfaces and its parameters")
    law = str(value.get("law", "bilinear"))
    if law not in LAW_OPTIONS:
        raise ConfigError(f"{name}.law: one of {list(LAW_OPTIONS)}, got {law!r}")
    allowed = set(LAW_OPTIONS[law]) | {"law", *extra}
    foreign = sorted(set(value) - allowed)
    if foreign:
        raise ConfigError(
            f"{name}: {foreign} are not parameters of the law {law!r}; use {sorted(allowed)}"
        )
    required = {"bilinear": ("normal_strength", "mode_I"), "exponential": ("strength", "energy")}
    missing = [key for key in required.get(law, ()) if key not in value]
    if missing:
        raise ConfigError(f"{name}: the law {law!r} needs {missing}")
    return {"law": law, **_interface_numbers(name, value, LAW_OPTIONS[law])}


def _interfaces(name: str, value) -> dict:
    """The law of the interfaces of the phases along the loading paths — or their
    ``laws``, by pair of phases (``"0-1"``) — and the options of their cell (a mapping):
    those of morphogen.fracture.periodic (interface_law and CohesiveCell)."""
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: a mapping of the law of the interfaces and its options")
    _check_keys(name, value, set(INTERFACE_OPTIONS), set())
    if "laws" in value:
        foreign = sorted(set(value) - {"laws", *_CELL_OPTIONS})
        laws = value["laws"]
        if foreign or not isinstance(laws, dict) or not laws:
            raise ConfigError(
                f"{name}: 'laws', a mapping of pairs of phases (0-1) to their laws, beside "
                f"{list(_CELL_OPTIONS)} only"
            )
        out: dict = {"laws": {}}
        for key, law in laws.items():
            try:
                p, q = sorted(int(part) for part in str(key).split("-"))
            except ValueError:
                raise ConfigError(f"{name}.laws: pairs of two phases (0-1), got {key!r}") from None
            out["laws"][f"{p}-{q}"] = _interface_law(f"{name}.laws.{key}", law)
    else:
        out = _interface_law(name, value, _CELL_OPTIONS)
    out.update(_interface_numbers(name, value, ("viscosity", "slip")))
    faces = value.get("faces", "phases")
    if isinstance(faces, str):
        if faces not in ("phases", "grains"):
            raise ConfigError(f"{name}.faces: 'phases', 'grains' or pairs of phases")
    else:
        try:
            faces = [[int(p) for p in pair] for pair in faces]
        except (TypeError, ValueError):
            faces = []
        if not faces or any(len(pair) != 2 or pair[0] == pair[1] for pair in faces):
            raise ConfigError(
                f"{name}.faces: 'phases', 'grains' or pairs of two different phases ([[0, 1]])"
            )
    out["faces"] = faces
    return out


def _path_quantity(item: str, homog: HomogenizationSpec) -> str:
    """Canonical name of a quantity of a loading path, checked against the path (its
    times), the boundary conditions and the components of the curves."""
    try:
        q = PathQuantity.parse(item)
    except ValueError as exc:
        raise ConfigError(f"rve.properties: {exc}") from None
    finite = homog.nonlinear.get("kinematics") == "finite"
    families = ["E", "S", "LE", "TS"] if finite else ["E", "S"]
    allowed = [f + n for f in families for n in VOIGT_NAMES] + ["p"]
    phases = [p for p, m in homog.materials.items() if not m.void]
    allowed += [
        f"{f}{n}_{p}"
        for p in phases
        for f in (["S", "TS"] if finite else ["S"])
        for n in VOIGT_NAMES
    ]
    allowed += [f"p_{p}" for p in phases]
    # the largest stress of the loading and the work done; the columns of a fracture
    allowed += ["strength", "work"]
    if homog.fracture:
        allowed += ["damage", "crack_surface", "elastic_energy", "fracture_energy"]
    contact = homog.nonlinear.get("contact") is not None
    if contact:
        from morphogen.contact.periodic import SERIES

        allowed += list(SERIES)
    interfaces = homog.nonlinear.get("interfaces") is not None
    if interfaces:
        from morphogen.fracture.periodic import SERIES as DEBONDING

        allowed += list(DEBONDING)
    # the twinned fraction, the opening and the damage of the cleavage planes of crystals
    crystals = [m.crystal_plasticity for m in homog.materials.values() if m.crystal_plastic]
    twinned = list(dict.fromkeys(name for cp in crystals for name in cp.field_names))
    allowed += twinned
    fatigue = homog.fatigue is not None and homog.fatigue["path"] == q.path
    if fatigue:
        allowed += list(FATIGUE_QUANTITIES)
    beremin = homog.beremin is not None and homog.beremin["path"] == q.path
    if beremin:
        from morphogen.homogenization.beremin import SERIES as BEREMIN

        allowed += list(BEREMIN)
    if q.quantity not in allowed:
        raise ConfigError(
            f"rve.properties: {item!r}: unknown quantity {q.quantity!r} of the curves; use "
            f"E11 ... S12{', LE11 ... TS12' if finite else ''}, p, a phase (S11_<phase>, "
            "p_<phase>), strength or work"
            + (", damage, crack_surface, elastic_energy, fracture_energy" if homog.fracture else "")
            + (
                ", contact_area, contact_pressure, contact_penetration, contact_dissipation"
                if contact
                else ""
            )
            + (
                ", debonded_area, interface_damage, interface_opening, interface_dissipation"
                if interfaces
                else ""
            )
            + (", fatigue_utilization, fatigue_safety, fatigue_beyond" if fatigue else "")
            + (f", {', '.join(twinned)}" if twinned else "")
            + (", beremin_weibull, beremin_probability" if beremin else "")
        )
    if q.time is not None:
        ends = homog.paths[q.path].schedule()
        if not np.any(np.isclose(ends, q.time, rtol=1e-9, atol=1e-12)):
            raise ConfigError(
                f"rve.properties: {item!r}: no step of the path {q.path!r} ends at the time "
                f"{q.time:g} (the steps end at {[float(f'{t:.6g}') for t in ends]})"
            )
    solved = [b for b in homog.boundary_conditions if b in ("periodic", "kubc")]
    if q.bc is not None and q.bc not in solved:
        raise ConfigError(
            f"rve.properties: {item!r}: the paths are solved under {solved} conditions"
        )
    return q.key


def _rve(data, homog: HomogenizationSpec | None = None) -> RVESpec:
    allowed = {
        "method",
        "sizes",
        "realizations",
        "size",
        "window_sizes",
        "properties",
        "relative_errors",
        "n_realizations",
        "gamma",
        "min_volume",
        "max_volume",
        "bootstrap",
        "weights",
        "shifts",
    }
    _check_keys("rve", data, allowed)
    method = data.get("method", "ensemble")
    if method not in ("ensemble", "windows"):
        raise ConfigError("rve.method: 'ensemble' or 'windows'")
    spec = RVESpec(method=method)
    if "shifts" in data:
        if method != "windows":
            raise ConfigError("rve.shifts: the shifted tilings of the window method")
        spec.shifts = _positive_int("rve.shifts", data["shifts"])
    spec.properties = _rve_properties(data.get("properties"), homog)
    spec.relative_errors = [float(e) for e in data.get("relative_errors", [0.01, 0.05])]
    spec.n_realizations = [int(n) for n in data.get("n_realizations", [1, 10])]
    if any(e <= 0 for e in spec.relative_errors) or any(n < 1 for n in spec.n_realizations):
        raise ConfigError("rve: relative_errors must be positive and n_realizations >= 1")
    spec.gamma = None if data.get("gamma") is None else float(data["gamma"])
    spec.min_volume = data.get("min_volume")
    spec.max_volume = data.get("max_volume")
    boot = data.get("bootstrap", 1000)
    if isinstance(boot, bool) or not isinstance(boot, int) or boot < 0:
        raise ConfigError("rve.bootstrap: a non-negative integer (number of replicates)")
    spec.bootstrap = boot
    spec.weights = str(data.get("weights", "none"))
    if spec.weights not in ("none", "n"):
        raise ConfigError("rve.weights: 'none' or 'n'")
    if method == "ensemble":
        if "sizes" not in data or "realizations" not in data:
            raise ConfigError("rve: the ensemble method needs 'sizes' and 'realizations'")
        spec.sizes = [float(s) for s in data["sizes"]]
        reals = data["realizations"]
        reals = [reals] * len(spec.sizes) if isinstance(reals, int) else list(reals)
        if len(reals) != len(spec.sizes):
            raise ConfigError("rve.realizations: one number, or one per size")
        spec.realizations = [_positive_int("rve.realizations", r) for r in reals]
    else:
        if "size" not in data or "window_sizes" not in data:
            raise ConfigError("rve: the window method needs 'size' and 'window_sizes'")
        spec.size = float(data["size"])
        spec.window_sizes = [float(w) for w in data["window_sizes"]]
        if homog is not None:
            _window_homogenization(homog)
        tensors = homog.tensors() if homog is not None else []
        bad = [p for p in spec.properties if p != "volume_fraction" and p not in tensors]
        if bad:
            raise ConfigError(
                "rve: the window method measures volume_fraction and the tensors of its "
                f"windows (stiffness, conductivity_tensor), not {bad}"
            )
    return spec


def _window_homogenization(homog: HomogenizationSpec) -> None:
    """The homogenization of a window study: the tensors of its windows (the stiffness, the
    conductivity tensor), the realization solved once per unit load by the FFT solver."""
    other = [p for p in homog.properties if p not in TENSOR_PROPERTIES]
    if other or not homog.properties:
        raise ConfigError(
            "rve: the window method computes the tensors of its windows "
            f"(homogenization.properties: stiffness, conductivity_tensor), not {other}"
        )
    if homog.loadings or homog.paths or homog.ntfa:
        raise ConfigError(
            "rve: the window method solves the unit loads of its tensors, not loadings or "
            "paths (homogenization.loadings, paths, ntfa)"
        )
    if homog.fields:
        raise ConfigError("rve: the window method writes no fields (homogenization.fields)")


def _colormap(section: str, value) -> str:
    """A matplotlib colormap name (checked when matplotlib is installed)."""
    name = str(value)
    try:
        import matplotlib
    except ImportError:  # pragma: no cover - checked when the figures are drawn
        return name
    if name not in matplotlib.colormaps:
        raise ConfigError(f"{section}: {name!r} is not a matplotlib colormap (viridis, ...)")
    return name


def _field_kind(physics: str) -> str:
    """The kind of fields of the solutions of a physics: "elastic", or "thermal" for
    every transport phenomenon of a scalar potential."""
    return "elastic" if physics == "elastic" else "thermal"


def _result_group(section: str, of: str, homog: HomogenizationSpec | None) -> tuple[str, str]:
    """The group ("result" or "path") and the kind of fields ("elastic", "thermal",
    "path") of a view of the result ``of`` = ``<name>.<bc>`` of the homogenization (the
    fields of every transport phenomenon are those of the kind "thermal": a potential,
    its gradient and its flux)."""
    key, dot, bc = of.rpartition(".")
    if not dot or not key:
        raise ConfigError(
            f"{section}.of: 'image', 'labels', 'mesh' or a result '<name>.<bc>' of the "
            f"homogenization, got {of!r}"
        )
    if homog is None:
        raise ConfigError(f"{section}.of: {of!r} is a result of the homogenization section")
    if bc not in homog.boundary_conditions:
        raise ConfigError(
            f"{section}.of: {bc!r} is not one of the boundary conditions "
            f"{homog.boundary_conditions}"
        )
    if key in homog.paths:
        if bc not in ("periodic", "kubc"):
            raise ConfigError(f"{section}.of: loading paths are followed under periodic or kubc")
        if homog.ntfa and not homog.ntfa.get("full_field"):
            raise ConfigError(f"{section}.of: the paths are followed by the NTFA model only")
        return "path", "path"
    if key in homog.loadings:
        return "result", _field_kind(homog.loadings[key].physics)
    if key in PROPERTIES and key in homog.properties:
        return "result", _field_kind(PHYSICS[key])
    names = [p for p in homog.properties if p in PROPERTIES] + [*homog.loadings, *homog.paths]
    raise ConfigError(
        f"{section}.of: {key!r} is none of the properties with fields, loadings and paths "
        f"of the homogenization: {names}"
    )


def _number(section: str, value, low: float, high: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = float("nan")
    if isinstance(value, bool) or not low < x <= high:
        raise ConfigError(f"{section}: a number in ({low:g}, {high:g}], got {value!r}")
    return x


def _view(section: str, data, homog: HomogenizationSpec | None, *, video: bool) -> ViewSpec:
    """One figure or video of the ``visualization`` section."""
    from morphogen.viz.video import MODES
    from morphogen.viz.workflow import DEFAULT_FIELD, FIELDS, KINDS, NODAL

    if isinstance(data, str):
        data = {"of": data}
    if not isinstance(data, dict) or "of" not in data:
        raise ConfigError(
            f"{section}: a mapping with 'of' (image, labels, mesh, or a result '<name>.<bc>' "
            "of the homogenization)"
        )
    of = str(data["of"])
    if of in ("image", "labels", "mesh"):
        group, physics = of, None
    else:
        group, physics = _result_group(section, of, homog)
    options = {
        "image": {"axis", "phase", "cmap"},
        "labels": {"axis"},
        "mesh": {"color", "axis"},
        "result": {"color", "axis", "magnification", "section", "part", "cmap", "clim"},
        "path": {"color", "axis", "magnification", "section", "part", "cmap", "clim"},
    }[group]
    allowed = {"of", "kind", "formats", "name"} | options
    if video:
        allowed |= {"frames", "mode", "repeat", "fps"} | ({"wave"} if group == "result" else set())
    _check_keys(section, data, allowed)
    kinds = KINDS[group][1 if video else 0]
    kind = str(data.get("kind", kinds[0]))
    if kind not in kinds:
        what = "videos" if video else "figures"
        raise ConfigError(f"{section}.kind: one of {list(kinds)} for the {what} of {of}")
    formats = _list_of(
        f"{section}.formats",
        data.get("formats", "mp4" if video else "png"),
        VIDEO_FORMATS if video else FIGURE_FORMATS,
    )
    if not formats:
        raise ConfigError(f"{section}.formats: give at least one")
    fe = homog is not None and homog.method in ("fe", "xfem")  # fields on a mesh
    vtk = (
        group == "mesh"
        or kind == "surface"
        or (group in ("result", "path") and kind == "field" and fe)
    )
    if vtk and not video:
        vector = [f for f in formats if f not in BITMAP_FORMATS]
        if vector:
            raise ConfigError(
                f"{section}.formats: the 3D views are bitmaps {list(BITMAP_FORMATS)}, not {vector}"
            )
    color = data.get("color")
    if group == "mesh":
        color = str(color or "phase")
        if color not in ("phase", "none"):
            raise ConfigError(f"{section}.color: 'phase' or 'none' (uniform)")
    elif physics is not None:
        color = str(color or DEFAULT_FIELD[physics])
        if color not in FIELDS[physics]:
            raise ConfigError(f"{section}.color: one of {list(FIELDS[physics])}")
        if not fe and color in NODAL and group == "result":
            raise ConfigError(f"{section}.color: {color!r} is on the nodes of finite elements")
    axis = data.get("axis", 2)
    if isinstance(axis, bool) or axis not in (0, 1, 2):
        raise ConfigError(f"{section}.axis: 0, 1 or 2")
    phase = data.get("phase", 1)
    if isinstance(phase, bool) or not isinstance(phase, int) or phase < 0:
        raise ConfigError(f"{section}.phase: a phase number")
    magnification = data.get("magnification")
    if magnification is not None:
        magnification = _number(f"{section}.magnification", magnification, 0.0, 1e12)
    cut = data.get("section")
    if cut is not None:
        if not fe:
            raise ConfigError(
                f"{section}.section: planes cut finite element meshes; the sections of the "
                "voxel images are normal to 'axis'"
            )
        if isinstance(cut, dict):
            _check_keys(f"{section}.section", cut, {"axis", "fraction"}, {"axis"})
            cut = (cut["axis"], cut.get("fraction", 0.5))
        if not isinstance(cut, (list, tuple)) or len(cut) != 2 or cut[0] not in (0, 1, 2):
            raise ConfigError(f"{section}.section: {{axis: 0, 1 or 2, fraction: 0 to 1}}")
        fraction = float(cut[1])
        if not 0.0 <= fraction <= 1.0:
            raise ConfigError(f"{section}.section.fraction: between 0 and 1")
        cut = (int(cut[0]), fraction)
    part = str(data.get("part", "total"))
    if part not in ("total", "fluctuation"):
        raise ConfigError(f"{section}.part: 'total' or 'fluctuation' (without E x)")
    if part == "fluctuation" and physics == "thermal":
        raise ConfigError(f"{section}.part: the fluctuations of mechanical displacements")
    wave = str(data.get("wave", "ramp"))
    if wave not in ("ramp", "sine"):
        raise ConfigError(f"{section}.wave: 'ramp' (a load growing) or 'sine' (oscillating)")
    frames = data.get("frames")
    if frames is not None:
        frames = _positive_int(f"{section}.frames", frames)
    mode = str(data.get("mode", "once"))
    if mode not in MODES:
        raise ConfigError(f"{section}.mode: one of {list(MODES)}")
    fps = data.get("fps")
    if fps is not None:
        fps = _number(f"{section}.fps", fps, 0.0, 240.0)
    cmap = data.get("cmap")
    if cmap is not None:
        cmap = _colormap(f"{section}.cmap", cmap)
    clim = data.get("clim")
    if clim is not None:
        if not isinstance(clim, (list, tuple)) or len(clim) != 2:
            raise ConfigError(f"{section}.clim: [low, high]")
        clim = (float(clim[0]), float(clim[1]))
        if not clim[0] < clim[1]:
            raise ConfigError(f"{section}.clim: low < high")
    name = data.get("name")
    if name is not None:
        name = str(name)
        if not name or any(c in name for c in '/\\:*?"<>| '):
            raise ConfigError(f"{section}.name: a file name without spaces or separators")
    return ViewSpec(
        of=of,
        group=group,
        kind=kind,
        formats=formats,
        color=color,
        axis=int(axis),
        phase=int(phase),
        magnification=magnification,
        wave=wave,
        section=cut,
        part=part,
        frames=frames,
        mode=mode,
        repeat=_positive_int(f"{section}.repeat", data.get("repeat", 1)),
        fps=fps,
        cmap=cmap,
        clim=clim,
        name=name,
        vtk=vtk,
    )


def _visualization(data, homog: HomogenizationSpec | None) -> VisualizationSpec:
    """The ``visualization`` section: figures and videos of the first realizations."""
    if not isinstance(data, dict):
        raise ConfigError("visualization: a mapping (figures, videos, realizations, cmap...)")
    _check_keys(
        "visualization",
        data,
        {"figures", "videos", "realizations", "cmap", "fps", "size", "dpi", "elevation", "azimuth"},
    )
    camera = {}
    for key in ("elevation", "azimuth"):
        if key in data:
            camera[key] = _number(f"visualization.{key}", data[key], -360.0, 360.0)

    def views(key: str, video: bool) -> list[ViewSpec]:
        items = data.get(key) or []
        items = [items] if isinstance(items, (str, dict)) else items
        if not isinstance(items, list):
            raise ConfigError(f"visualization.{key}: a list of views")
        return [
            _view(f"visualization.{key}[{k}]", v, homog, video=video) for k, v in enumerate(items)
        ]

    figures, videos = views("figures", False), views("videos", True)
    if not figures and not videos:
        raise ConfigError("visualization: give figures or videos")
    files = [(v.label, f) for v in figures + videos for f in v.formats]
    twice = sorted({label for label, f in files if files.count((label, f)) > 1})
    if twice:
        raise ConfigError(f"visualization: views write the same files {twice}; name them ('name')")
    size = data.get("size", [640, 480])
    size = [size, size] if isinstance(size, int) and not isinstance(size, bool) else size
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or not all(isinstance(n, int) and not isinstance(n, bool) and 16 <= n <= 8192 for n in size)
    ):
        raise ConfigError("visualization.size: [width, height] in pixels (16 to 8192)")
    return VisualizationSpec(
        figures=figures,
        videos=videos,
        realizations=_positive_int("visualization.realizations", data.get("realizations", 1)),
        cmap=_colormap("visualization.cmap", data.get("cmap", "viridis")),
        fps=_number("visualization.fps", data.get("fps", 24), 0.0, 240.0),
        size=(int(size[0]), int(size[1])),
        dpi=_positive_int("visualization.dpi", data.get("dpi", 200)),
        **camera,
    )


#: The element size of the conforming meshes that ``method: auto`` describes when a
#: configuration gives none: the shortest edge of the domain over this number
DEFAULT_ELEMENTS_PER_EDGE = 10
#: Their Hausdorff distance and smallest element, as fractions of that size: the curved
#: interfaces followed within an eighth of an element, no element below a third of one
#: (a mesh some times lighter than the defaults of ``conforming_mesh``, for a first run)
DEFAULT_HAUSDORFF, DEFAULT_MIN_SIZE = 1.0 / 8.0, 1.0 / 3.0


def default_mesh(domain: Domain, voxels: VoxelSpec | None = None) -> MeshSpec:
    """The conforming mesh that ``method: auto`` describes for a configuration without a
    ``mesh`` section: elements of the shortest edge of the domain over
    :data:`DEFAULT_ELEMENTS_PER_EDGE`, a Hausdorff distance and a smallest element of
    :data:`DEFAULT_HAUSDORFF` and :data:`DEFAULT_MIN_SIZE` of that size, the level set
    sampled at the voxels of a ``voxels`` section (the grid route), if any."""
    size = float(np.min(domain.lengths)) / DEFAULT_ELEMENTS_PER_EDGE
    background = None
    if voxels is not None:  # the resolution the configuration asks for its images
        if voxels.voxel_size is not None:
            background = float(voxels.voxel_size)
        elif voxels.shape is not None:
            background = float(np.max(domain.lengths)) / float(np.max(voxels.shape))
    return MeshSpec(size=size, hausdorff=DEFAULT_HAUSDORFF * size,
                    min_size=DEFAULT_MIN_SIZE * size, background=background)  # fmt: skip


def voxels_only(
    homog: HomogenizationSpec, rve=None, voxels: VoxelSpec | None = None, *, plane: bool = False
) -> list[str]:
    """What only the solvers of voxel images do among the options of a homogenization (and
    of the RVE study and voxels of its configuration): the reasons for which ``method:
    auto`` takes the FFT solver on voxels, the last resort, rather than finite elements
    on conforming meshes (``plane``: those of the plane, which give the permeability of
    their pores too)."""
    from morphogen.homogenization.apparent import PLANE_FLOW

    reasons = []
    if homog.device == "gpu":
        reasons.append("the GPU (device: gpu) runs the FFT solver")
    voxel = [p for p in homog.properties if PHYSICS[p] in ("coupled", "harmonic", "flow")
             and not (plane and p in PLANE_FLOW)]  # fmt: skip
    if voxel:
        reasons.append(f"the properties {voxel} are solved on voxel images")
    if homog.fracture is not None:
        reasons.append("the phase-field fracture runs on voxel images")
    if homog.ntfa:
        reasons.append("the reduced models (ntfa) are those of voxel images")
    if rve is not None and rve.method == "windows":
        reasons.append("the window method solves its realization by FFT")
    if voxels is not None and voxels.section is not None:
        reasons.append("a planar study is a layer of voxels (voxels.section)")
    return reasons


def _check_method(
    homog: HomogenizationSpec,
    mesh: MeshSpec | None,
    *,
    rve=None,
    voxels: VoxelSpec | None = None,
    domain: Domain | None = None,
    plane: bool = False,
) -> MeshSpec | None:
    """Resolve ``method: auto``: finite elements on conforming meshes (exact where the
    model knows its geometry), with the mesh of :func:`default_mesh` when the
    configuration describes none; the FFT solver on voxels only for what only it does
    (:func:`voxels_only`) or without a conforming mesher (MMG). Check that the boundary
    conditions suit the method; return the mesh (``plane``: the meshes are those of the
    plane)."""
    from morphogen.homogenization.apparent import PLANE_FLOW

    if homog.method == "auto":
        reasons = voxels_only(homog, rve, voxels, plane=plane)
        if not reasons:
            from morphogen.mesh.conforming import has_mmg

            if not has_mmg():
                import warnings

                warnings.warn(
                    "homogenization: no conforming mesher here (MMG: pip install pymmg); the "
                    "FFT solver on voxels instead of finite elements",
                    stacklevel=3,
                )
                reasons.append("no conforming mesher (MMG)")
        homog.method = "fft" if reasons else "fe"
        if homog.method == "fe" and mesh is None and domain is not None:
            mesh = default_mesh(domain, voxels)
    if homog.method == "fe" and mesh is None:
        raise ConfigError(
            "homogenization.method 'fe': describe the conforming meshes in a 'mesh' section "
            "(at least the element size)"
        )
    if homog.method == "fe" and homog.device == "gpu":
        raise ConfigError(
            "homogenization.device 'gpu': the GPU runs the FFT solver (method: fft); finite "
            "elements run on the CPU"
        )
    coupled = [p for p in homog.properties if PHYSICS[p] in ("coupled", "harmonic")]
    if homog.method != "fft" and coupled:
        raise ConfigError(
            f"homogenization.properties: the coupled and harmonic problems {coupled} are "
            "solved on voxel images (method: fft, periodic conditions)"
        )
    flows = [p for p in homog.properties if PHYSICS[p] == "flow"
             and not (homog.method == "fe" and p in PLANE_FLOW)]  # fmt: skip
    if homog.method != "fft" and flows:
        raise ConfigError(
            f"homogenization.properties: the flows through the pores {flows} are solved on "
            "voxel images (method: fft, periodic conditions), the permeability on the "
            "conforming meshes too (method: fe)"
        )
    if homog.fracture is not None and homog.method != "fft":
        raise ConfigError(
            "homogenization.fracture: the phase-field method runs on voxel images (method: "
            "fft, periodic conditions)"
        )
    if homog.nonlinear.get("contact") is not None:
        name = "homogenization.nonlinear.contact"
        if homog.method != "fe":
            raise ConfigError(
                f"{name}: the walls of the pores are the free faces of a conforming mesh "
                "(method: fe, with a 'mesh' section)"
            )
        if not homog.porous:
            raise ConfigError(
                f"{name}: the walls of the pores of a porous medium (a void material)"
            )
        if mesh.phases == "all":
            raise ConfigError(f"{name}: the mesh of the solid alone (mesh.phases: auto or solid)")
        if homog.solver not in ("auto", "direct"):
            raise ConfigError(f"{name}: the tangent of the contacts is factorized (solver: direct)")
    if homog.fatigue is not None and homog.method == "xfem":
        raise ConfigError(
            "homogenization.fatigue: the fields of the phases of a mesh or of an image "
            "(method: fe or fft)"
        )
    if homog.beremin is not None and homog.method == "xfem":
        raise ConfigError(
            "homogenization.beremin: the fields of the phases of a mesh or of an image "
            "(method: fe or fft)"
        )
    if homog.nonlinear.get("interfaces") is not None:
        name = "homogenization.nonlinear.interfaces"
        if homog.method != "fe":
            raise ConfigError(
                f"{name}: the interfaces of the phases are the faces of a conforming mesh "
                "(method: fe, with a 'mesh' section)"
            )
        if homog.solver not in ("auto", "direct"):
            raise ConfigError(
                f"{name}: the tangent of the interfaces is factorized (solver: direct)"
            )
    if homog.method in ("fft", "xfem"):
        other = [b for b in homog.boundary_conditions if b != "periodic"]
        if other:
            solver = "FFT solver" if homog.method == "fft" else "XFEM solver of level sets"
            raise ConfigError(
                f"homogenization: {other} conditions need finite elements on conforming "
                f"meshes (method: fe, with a 'mesh' section); the {solver} is periodic"
            )
    if homog.method == "xfem":
        if homog.device == "gpu":
            raise ConfigError("homogenization.device 'gpu': the GPU runs the FFT solver")
        if "integration" in homog.nonlinear or homog.integration is not None:
            raise ConfigError(
                "homogenization.integration: an option of the finite elements of a mesh "
                "(method: fe) and of the voxels of the loading paths (method: fft)"
            )
        if homog.ntfa:
            raise ConfigError("homogenization.ntfa: a reduced model of voxel images (method: fft)")
    return mesh


def _model_paths(model, base: Path):
    """The model of a configuration with its files (the ``path_parameters`` of its
    class: parameters, or keys of parameters that are mappings, ``texture.file``) taken
    from the folder of the configuration; the models that it holds likewise."""
    if isinstance(model, (list, tuple)):
        return [_model_paths(item, base) for item in model]
    if not isinstance(model, dict):
        return model
    out = {key: _model_paths(value, base) for key, value in model.items()}
    try:
        names = get_model(str(model.get("type"))).path_parameters
    except ValueError:
        return out
    for name in names:
        *holders, key = name.split(".")
        target = out
        for holder in holders:
            target = target.get(holder) if isinstance(target, dict) else None
        value = target.get(key) if isinstance(target, dict) else None
        if isinstance(value, str) and value and not Path(value).expanduser().is_absolute():
            target[key] = str(base / value)
    return out


def load_config(source, *, base_dir=None) -> RunConfig:
    """Validate a configuration given as a file path or a mapping."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = read_config_file(path)
        base = path.parent if base_dir is None else Path(base_dir)
    else:
        data = dict(source)
        base = Path.cwd() if base_dir is None else Path(base_dir)
    allowed = {
        "model",
        "domain",
        "seed",
        "realizations",
        "voxels",
        "outputs",
        "analysis",
        "rve",
        "homogenization",
        "diffusion",
        "mesh",
        "visualization",
        "threads",
        "version",
    }
    if "optimization" in data and "model" not in data:
        raise ConfigError("this is a topology optimization: run it with 'morphogen optimize'")
    _check_keys("configuration", data, allowed, {"model"})
    try:
        model = model_from_config(_model_paths(data["model"], base))
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    homog = None
    if data.get("homogenization") is not None:
        homog = _homogenization(data["homogenization"], base)
    rve = _rve(data["rve"], homog) if data.get("rve") is not None else None
    for count in ("n_cells", "number"):
        if rve is not None and getattr(model, count, None) is not None:
            raise ConfigError(
                f"model.{count}: RVE studies compare samples of different sizes and need "
                "a density (intensity or volume_fraction); a fixed count would change "
                "the microstructure with the sample size"
            )
    dom = dict(data.get("domain") or {})
    _check_keys("domain", dom, {"size", "origin", "periodic", "unit"})
    if rve is not None:
        if "size" in dom:
            raise ConfigError("domain.size: not used by RVE studies (see rve.sizes / rve.size)")
        dom["size"] = rve.sizes[0] if rve.method == "ensemble" else rve.size
    elif "size" not in dom:
        raise ConfigError("domain: 'size' is required")
    try:
        domain = Domain.from_dict(dom)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    if rve is not None and rve.shifts > 1 and not domain.periodic:
        raise ConfigError(
            "rve.shifts: the shifted tilings wrap around the realization, a periodic one "
            "(domain.periodic: true)"
        )
    seed = data.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ConfigError(f"seed: expected a non-negative integer, got {seed!r}")
    voxels = _voxels(data["voxels"]) if data.get("voxels") is not None else None
    diffusion = _diffusion(data["diffusion"]) if data.get("diffusion") is not None else None
    if diffusion is not None:
        if voxels is None:
            raise ConfigError("diffusion: give the voxel grid (voxels section)")
        missing = sorted(set(model.active_phases()) - set(diffusion.diffusivity))
        if missing:
            raise ConfigError(f"diffusion.phases: no properties for phase(s) {missing}")
    mesh = _mesh(data["mesh"]) if data.get("mesh") is not None else None
    outputs = _outputs(data.get("outputs") or {}, base)
    from morphogen.morphology.measures import EXACT_MEASURES

    analysis = _list_of("analysis", data.get("analysis"), {**MEASURES, **EXACT_MEASURES})
    if homog is not None:
        mesh = _check_method(homog, mesh, rve=rve, voxels=voxels, domain=domain,
                             plane=_meshes_of_the_plane(mesh, model))  # fmt: skip
    if mesh is not None and mesh.order is None:
        # quadratic elements for the finite element analyses: linear tetrahedra are stiff,
        # and lock under isochoric plastic flow; a mesh that is only written stays linear
        analysed = homog is not None and homog.method == "fe"
        mesh.order = 2 if analysed else 1
        if mesh.curved and mesh.order == 1:
            raise ConfigError("mesh.curved: the edges of 10-node elements (mesh.order: 2)")
    if homog is not None:
        if homog.method == "xfem":
            if not model.has_level_set:
                raise ConfigError(
                    f"homogenization.method 'xfem': the model {model.name} has no level set "
                    "(tessellations: method fe, grain by grain, or fft)"
                )
            if not domain.periodic:
                raise ConfigError(
                    "homogenization.method 'xfem': a periodic domain (domain.periodic: true)"
                )
        if rve is not None and rve.method == "windows" and homog.method != "fft":
            raise ConfigError(
                "rve: the window method solves its realization by the FFT solver "
                "(homogenization.method: fft, periodic conditions)"
            )
        phases = model.active_phases()
        missing = sorted(set(phases) - set(homog.materials))
        if missing:
            raise ConfigError(
                f"homogenization.materials: no material for phase(s) {missing} of model "
                f"{model.name} ({', '.join(f'{p}: {phases[p]}' for p in missing)})"
            )
    viz = None
    if data.get("visualization") is not None:
        viz = _visualization(data["visualization"], homog)
        if viz.shows("mesh") and mesh is None:
            raise ConfigError("visualization: the views of the mesh need the 'mesh' section")
        if viz.shows("image", "labels") and voxels is None:
            raise ConfigError("visualization: the views of the image need the 'voxels' section")
        if rve is not None and rve.method == "windows":
            raise ConfigError("visualization: the realizations of batches and ensemble studies")
    if outputs.conforming_mesh and mesh is None:
        raise ConfigError("outputs.conforming_mesh: give the 'mesh' section (element size...)")
    if (
        homog is not None
        and homog.plane is not None
        and (homog.method != "fe" or mesh is None or not _meshes_of_the_plane(mesh, model))
    ):
        raise ConfigError(
            "homogenization.plane: the elastic problems of the meshes of the plane (method: fe, "
            "a 'mesh' section, mesh.plane true, or auto for the seeds of the plane)"
        )
    if mesh is not None and mesh.plane:
        _check_plane(mesh, homog, outputs, viz)
    if (homog is not None and homog.method == "fe" and "hydraulic_permeability" in homog.properties
            and "periodic" not in homog.boundary_conditions):  # fmt: skip
        raise ConfigError(
            "homogenization.properties: the flow through the pores of a mesh, under periodic "
            "conditions (boundary_conditions: [periodic, ...])"
        )
    if mesh is not None and viz is not None and _meshes_of_the_plane(mesh, model):
        for view in viz.figures:  # the triangles drawn by matplotlib, not VTK
            if view.group in ("mesh", "result", "path"):
                view.vtk = False
    if (outputs.damask or outputs.neper) and not model.meshed_by_grains:
        raise ConfigError("outputs.damask, outputs.neper: Voronoi and Laguerre tessellations only")
    if outputs.damask and not getattr(model, "crystal_orientations", False):
        raise ConfigError("outputs.damask: the grains need orientations (crystal_orientations)")
    if outputs.amitex is not None:
        if homog is None:
            raise ConfigError("outputs.amitex: give the materials (homogenization section)")
        if homog.nonlinear.get("kinematics", "small") == "finite" and not homog.paths:
            raise ConfigError(
                "outputs.amitex in finite strains: the runs of AMITEX are the loading paths "
                "(homogenization.paths)"
            )
    if outputs.code_aster is not None:
        if mesh is None:
            raise ConfigError("outputs.code_aster: give the 'mesh' section (element size...)")
        if homog is None:
            raise ConfigError("outputs.code_aster: give the materials (homogenization section)")
    if outputs.openfoam is not None:
        if outputs.openfoam.get("sample") == "mesh" and mesh is None:
            raise ConfigError("outputs.openfoam.sample mesh: give the 'mesh' section")
        if outputs.openfoam.get("sample") != "mesh" and voxels is None:
            raise ConfigError("outputs.openfoam: give the 'voxels' section (or sample: mesh)")
    if outputs.periodic_equations and not domain.periodic:
        raise ConfigError("outputs.periodic_equations: a periodic domain")
    if outputs.periodic_equations and not (outputs.conforming_mesh or outputs.mesh):
        raise ConfigError(
            "outputs.periodic_equations: the equations refer to the nodes of a mesh written "
            "by outputs.conforming_mesh or outputs.mesh"
        )
    if mesh is not None and mesh.grains and not model.meshed_by_grains:
        what = (
            "are mapped by an anisotropy (voxelize them)"
            if hasattr(model, "boundary_thickness")
            and not getattr(getattr(model, "anisotropy", None), "is_identity", True)
            else "have a grain-boundary phase"
            if hasattr(model, "boundary_thickness")
            else ("are not tessellations")
        )
        raise ConfigError(f"mesh.grains: the realizations of model {model.name} {what}")
    measures = set(analysis) - set(EXACT_MEASURES)  # those on the implicit geometry: no voxels
    if rve is not None:
        measures |= {p for p in rve.properties if p in MEASURES}
    # without voxels, the volume fraction is measured on the conforming mesh
    from_mesh = {"volume_fraction"} if mesh is not None else set()
    needs_voxels = (
        outputs.needs_voxels
        or bool(measures - from_mesh)
        or (homog is not None and homog.method in ("fft", "xfem"))
        or (rve is not None and rve.method == "windows")
    )
    if voxels is not None and voxels.section is not None:
        if homog is not None and homog.method == "xfem":
            raise ConfigError(
                "voxels.section: the extended finite elements cut the grid of the volume "
                "(method: fft for a planar section)"
            )
        if rve is not None and rve.method == "windows":
            raise ConfigError("voxels.section: the window method takes its windows in the volume")
    if needs_voxels and voxels is None:
        raise ConfigError(
            "voxels: required to write images or voxel meshes, for morphological "
            "measurements and window studies, by the FFT solver and (the grid) by the "
            "extended finite elements"
        )
    if rve is not None and voxels is not None and voxels.voxel_size is None:
        raise ConfigError("voxels: RVE studies need 'voxel_size' (same resolution at all sizes)")
    threads = data.get("threads")
    if threads is not None:
        threads = _positive_int("threads", threads)
    return RunConfig(
        model=model,
        domain=domain,
        seed=seed,
        realizations=_positive_int("realizations", data.get("realizations", 1)),
        voxels=voxels,
        mesh=mesh,
        outputs=outputs,
        analysis=analysis,
        rve=rve,
        homogenization=homog,
        diffusion=diffusion,
        visualization=viz,
        threads=threads,
        source=data,
    )
