"""The computations of morphoGen as models of VIMSEO, whose tools demonstrate the credibility
of simulations: the verification of a code against references, of a solution against its
discretization, the validation against measurements with uncertain inputs, the
propagation of uncertainties, the sensitivities, the calibration.

VIMSEO (IRT Saint Exupéry, LGPL-3.0; https://github.com/vimseo/vimseo, the package
``vimseo`` 0.1.7 or later) is built on GEMSEO: a model of VIMSEO is a GEMSEO discipline of
named inputs and outputs (one-dimensional arrays of numbers), run for a load case, whose
runs are archived (inputs, outputs and metadata of each) and cached. Its tools take such
models: ``CodeVerificationAgainstData`` (the outputs against reference values at the same
inputs, by error metrics), ``DiscretizationSolutionVerification`` (the outputs on four
discretizations: Richardson's extrapolation, the observed order, the grid convergence
index of Roache), the validation cases and points (the area metric between the
distributions of the simulations and of the measurements), the plans of experiments, the
sensitivities and the calibration. VIMSEO runs on Python 3.10 to 3.12: morphoGen and
VIMSEO together on Python 3.12 (``pip install "vimseo>=0.1.7" "morphogen[all]"``).

morphoGen gives VIMSEO what it computes, as functions of named numbers
(:class:`Simulation`):

* :func:`configuration`: a configuration of ``morphogen run`` (a model, its domain, its
  voxels or mesh, the homogenization) whose entries chosen by their paths
  (``model.volume_fraction``, ``mesh.size``, ``voxels.shape``, ``homogenization.materials.1.
  young``, ``seed``) are the inputs and whose results (the apparent properties
  ``bulk_modulus.periodic``..., the measures) are the outputs, one realization run per
  execution;
* :func:`exact_morphology`: the four Minkowski densities of the realizations of a model of
  spheres, exactly (:mod:`morphogen.morphology.balls`), with :func:`theory_of` the closed
  forms of the model at the same inputs — the references of a verification of the code
  in which only the fluctuation of the realizations remains;
* :func:`discipline` and :func:`model`: the GEMSEO discipline and the VIMSEO model of a
  simulation (``BaseDisciplineModel``, the load case ``Dummy``), and :func:`dataset` the
  reference data that the tools take; :func:`solution_verification` and
  :func:`code_verification` run the two tools of verification on a model.

GEMSEO and VIMSEO are imported by these last functions only: the simulations are plain
functions, of use without them.
"""

from __future__ import annotations

import copy
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The densities of :func:`exact_morphology`
DENSITIES = ("volume_fraction", "specific_surface", "mean_curvature_density", "euler_density")
#: The properties of a configuration that are numbers (one column per boundary condition)
SCALARS = (
    "bulk_modulus",
    "shear_modulus",
    "young_modulus",
    "poisson_ratio",
    "conductivity",
    "permittivity",
    "permeability",
    "diffusivity",
)


@dataclass
class Simulation:
    """A computation as a function of named numbers: ``function(**inputs)`` returns a mapping
    holding the ``outputs``; ``inputs`` are the default values, ``integers`` the inputs taken
    as whole numbers (a seed, a number of voxels) and ``bounds`` their ranges of validity."""

    name: str
    function: Callable[..., Mapping]
    inputs: dict[str, float]
    outputs: tuple[str, ...]
    integers: tuple[str, ...] = ()
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # a variable of a GEMSEO discipline is an input or an output, not both
        both = sorted(set(self.inputs) & set(self.outputs))
        if both:
            raise ValueError(f"{self.name}: {both} both inputs and outputs; rename the inputs")

    def __call__(self, **values) -> dict[str, float]:
        unknown = sorted(set(values) - set(self.inputs))
        if unknown:
            raise ValueError(f"{self.name}: unknown inputs {unknown}; inputs: {list(self.inputs)}")
        args = {**self.inputs, **values}
        for name in self.integers:
            args[name] = round(float(args[name]))
        for name, (low, high) in self.bounds.items():
            if not low <= float(args[name]) <= high:
                raise ValueError(f"{self.name}: {name} = {args[name]} out of [{low}, {high}]")
        out = self.function(**args)
        missing = [k for k in self.outputs if k not in out]
        if missing:
            raise ValueError(f"{self.name}: the outputs {missing} were not computed")
        return {k: float(out[k]) for k in self.outputs}


# ----------------------------------------------------------------------------------
# Configurations of morphoGen


def _key(mapping: dict, part: str):
    """The key of a mapping named ``part`` (an integer key if the mapping has it: the
    phases of the materials)."""
    if part in mapping:
        return part
    try:
        number = int(part)
    except ValueError:
        return part
    return number if number in mapping else part


def get_entry(data: Mapping, path: str):
    """The entry of a nested mapping at a dotted path (``mesh.size``)."""
    node = data
    for part in path.split("."):
        if not isinstance(node, Mapping):
            raise KeyError(path)
        key = _key(node, part)
        if key not in node:
            raise KeyError(path)
        node = node[key]
    return node


def set_entry(data: dict, path: str, value) -> None:
    """Set the entry of a nested mapping at a dotted path (the mappings on the way made if
    missing)."""
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        key = _key(node, part)
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[_key(node, parts[-1])] = value


def _default_outputs(data: Mapping) -> list[str]:
    """The numbers that a configuration computes: its scalar properties under each of its
    boundary conditions, and the volume fraction."""
    homog = data.get("homogenization") or {}
    props = homog.get("properties") or ["bulk_modulus", "shear_modulus"]
    bcs = homog.get("boundary_conditions") or ["periodic"]
    names = [f"{p}.{bc}" for p in props if p in SCALARS for bc in bcs]
    return [*names, "volume_fraction"]


def configuration(config, inputs: Mapping[str, str], outputs=None, *, name: str = "morphogen",
                  base_dir=None) -> Simulation:  # fmt: skip
    """The simulation of a configuration of ``morphogen run`` (a file or a mapping): the
    ``inputs`` map names to the dotted paths of entries of the configuration (their values
    there are the defaults), the ``outputs`` are columns of the results of a realization
    (default: the scalar properties under each boundary condition, and the volume
    fraction). Each call runs one realization of the configuration with the inputs given,
    in a temporary folder, and reads its results."""
    from morphogen.config import load_config, read_config_file

    if isinstance(config, (str, Path)):
        path = Path(config)
        data = read_config_file(path)
        base = path.parent if base_dir is None else Path(base_dir)
    else:
        data = copy.deepcopy(dict(config))
        base = Path.cwd() if base_dir is None else Path(base_dir)
    defaults, integers = {}, []
    for key, entry in inputs.items():
        try:
            value = get_entry(data, entry)
        except KeyError:
            raise ValueError(f"{name}: no entry {entry!r} in the configuration") from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}: the entry {entry!r} is not a number ({value!r})")
        if isinstance(value, int):
            integers.append(key)
        defaults[key] = float(value)
    outputs = tuple(_default_outputs(data) if outputs is None else outputs)

    def run(**values):
        from morphogen.analysis import MEASURES
        from morphogen.pipeline import read_rows, run

        cfg = copy.deepcopy(data)
        for key, entry in inputs.items():
            value = values[key]
            set_entry(cfg, entry, int(value) if key in integers else float(value))
        cfg["realizations"] = 1
        cfg.pop("rve", None)
        cfg.pop("visualization", None)
        wanted = [k for k in outputs if k in MEASURES]  # the measures asked for
        cfg["analysis"] = list(dict.fromkeys([*(cfg.get("analysis") or []), *wanted]))
        with tempfile.TemporaryDirectory() as tmp:
            cfg["outputs"] = {"directory": tmp, "name": name}
            outdir = run(load_config(cfg, base_dir=base))
            row = read_rows(Path(outdir) / "results.csv")[0]
        return {k: float(v) for k, v in row.items() if _number(v)}

    return Simulation(name, run, defaults, outputs, tuple(integers))


def _number(value) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


# ----------------------------------------------------------------------------------
# The exact measures of a model and their theory


#: The prefix of the names of the parameters of a model among the inputs of
#: :func:`exact_morphology` (``model_volume_fraction``: the outputs have the names of the
#: densities)
PREFIX = "model_"


def _model_inputs(model_config: Mapping, inputs) -> dict[str, float]:
    if inputs is None:
        inputs = [k for k, v in model_config.items() if k != "type" and _scalar(v)]
    out = {}
    for k in inputs:
        if k not in model_config or not _scalar(model_config[k]):
            raise ValueError(f"exact_morphology: {k!r} is not a number of the model")
        out[PREFIX + k] = float(model_config[k])
    return out


def _parameters(values: Mapping) -> dict:
    """The parameters of a model from the inputs named with :data:`PREFIX`."""
    return {k[len(PREFIX) :]: v for k, v in values.items() if k.startswith(PREFIX)}


def _scalar(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def exact_morphology(model_config: Mapping, size: float, *, inputs=None,
                     name: str = "exact_morphology") -> Simulation:  # fmt: skip
    """The simulation of the four Minkowski densities of the realizations of a model of
    spheres (a mapping ``{type, ...}``: Boolean, hard or penetrable spheres), exactly, on
    the torus of a periodic cube of edge ``size``: the inputs are the numeric parameters of
    the model (or those named in ``inputs``), prefixed by :data:`PREFIX`
    (``model_volume_fraction``), and ``seed``; the outputs the densities :data:`DENSITIES`,
    as the theory of the models names them."""
    from morphogen.domain import Domain
    from morphogen.models import model_from_config
    from morphogen.morphology.balls import minkowski_functionals

    base = dict(model_config)
    defaults = _model_inputs(base, inputs)
    defaults["seed"] = 0.0

    def run(seed: int, **values):
        m = model_from_config({**base, **_parameters(values)})
        micro = m.generate(Domain.cube(float(size), periodic=True), seed=int(seed))
        return minkowski_functionals(micro).densities()

    return Simulation(name, run, defaults, DENSITIES, ("seed",))


def theory_of(model_config: Mapping, points: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The closed forms of the densities of a model (:data:`DENSITIES`, from its theory) at
    the ``points`` of the inputs of :func:`exact_morphology` (arrays of the same length; the
    parameters of the model prefixed by :data:`PREFIX`): its references."""
    from morphogen.models import model_from_config

    names = list(points)
    columns = [np.atleast_1d(np.asarray(points[k], dtype=float)) for k in names]
    n = len(columns[0]) if columns else 1
    out = {k: np.empty(n) for k in DENSITIES}
    for i in range(n):
        values = {k: float(c[i]) for k, c in zip(names, columns, strict=True)}
        th = model_from_config({**dict(model_config), **_parameters(values)}).theory()
        for k in DENSITIES:
            out[k][i] = th[k]
    return out


# ----------------------------------------------------------------------------------
# GEMSEO and VIMSEO


def discipline(simulation: Simulation):
    """The GEMSEO discipline of a simulation: its inputs and outputs arrays of one number, the
    defaults of its inputs, its run a call of the simulation."""
    from gemseo.core.discipline import Discipline

    class SimulationDiscipline(Discipline):
        """A simulation of morphoGen as a GEMSEO discipline."""

        def __init__(self) -> None:
            super().__init__(simulation.name)
            self.simulation = simulation
            arrays = {k: np.array([v]) for k, v in simulation.inputs.items()}
            self.input_grammar.update_from_data(arrays)
            self.output_grammar.update_from_data({k: np.array([0.0]) for k in simulation.outputs})
            self.default_input_data.update(arrays)

        def _run(self, input_data):
            values = {k: float(np.asarray(input_data[k]).ravel()[0]) for k in simulation.inputs}
            out = self.simulation(**values)
            return {k: np.array([v]) for k, v in out.items()}

    return SimulationDiscipline()


def model(simulation: Simulation, **options):
    """The VIMSEO model of a simulation (a ``BaseDisciplineModel`` of its discipline, of the load
    case ``Dummy``); ``options`` are those of the models of VIMSEO (``directory_archive_root``,
    ``directory_scratch_root``, ``cache_file_path``...)."""
    from vimseo.core.base_discipline_model import BaseDisciplineModel

    name = "".join(part.capitalize() for part in simulation.name.replace("-", "_").split("_"))
    cls = type(
        name or "MorphogenSimulation",
        (BaseDisciplineModel,),
        {"_DISCIPLINE": discipline(simulation), "_EXPECTED_LOAD_CASE": "Dummy"},
    )
    return cls("Dummy", **options)


def dataset(inputs: Mapping[str, np.ndarray], outputs: Mapping[str, np.ndarray]):
    """The GEMSEO dataset (``IODataset``) of reference values: the ``inputs`` and ``outputs``
    arrays, one row per point."""
    from gemseo.datasets.io_dataset import IODataset

    data = IODataset()
    names = list(inputs)
    data.add_input_group(np.column_stack([np.asarray(inputs[k], dtype=float) for k in names]),
                         names)  # fmt: skip
    names = list(outputs)
    data.add_output_group(np.column_stack([np.asarray(outputs[k], dtype=float) for k in names]),
                          names)  # fmt: skip
    return data


def solution_verification(vimseo_model, size: str, sizes, output: str, *, working_directory=None):
    """VIMSEO's verification of the solution against its discretization: the ``output`` of a
    model at four values ``sizes`` (decreasing) of its input ``size`` (the size of the
    elements or voxels), Richardson's extrapolation, the observed order and the grid
    convergence index (``result.extrapolation``). Returns the tool, executed."""
    from vimseo.tools.verification.solution_verification import (
        DiscretizationSolutionVerification,
    )

    kwargs = {} if working_directory is None else {"working_directory": working_directory}
    tool = DiscretizationSolutionVerification(**kwargs)
    tool.execute(model=vimseo_model, element_size_variable_name=size,
                 element_size_values=[float(s) for s in sizes], output_name=output)  # fmt: skip
    return tool


def code_verification(vimseo_model, references, *, output_names=None, metric_names=None,
                      working_directory=None):  # fmt: skip
    """VIMSEO's verification of the code against reference values (a dataset of
    :func:`dataset`): the model run at the inputs of the references and its outputs compared
    by error metrics (``result``). Returns the tool, executed."""
    from vimseo.tools.verification.verification_vs_data import CodeVerificationAgainstData

    kwargs = {} if working_directory is None else {"working_directory": working_directory}
    tool = CodeVerificationAgainstData(**kwargs)
    options = {}
    if output_names is not None:
        options["output_names"] = list(output_names)
    if metric_names is not None:
        options["metric_names"] = list(metric_names)
    tool.execute(model=vimseo_model, reference_data=references, **options)
    return tool


__all__ = [
    "DENSITIES",
    "PREFIX",
    "SCALARS",
    "Simulation",
    "code_verification",
    "configuration",
    "dataset",
    "discipline",
    "exact_morphology",
    "get_entry",
    "model",
    "set_entry",
    "solution_verification",
    "theory_of",
]
