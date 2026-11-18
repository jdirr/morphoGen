"""Learning from a configuration (``morphogen learn``): the data of a behaviour at a
material point or of a microstructure along random paths, the training of a model, its
errors on the validation paths and on named paths that the learned law runs at a
material point against the reference, the files of the model and of the data.

A configuration (a mapping, or its ``learning`` mapping)::

    material: {young: 200000, poisson: 0.3, plasticity: {yield: 200, hardening: 10000}}
    # or a microstructure: sample: image.npy, materials: {0: ..., 1: ...}, bc: periodic
    kinematics: small
    data: {paths: 128, segments: 4, increments: 8, amplitude: 0.01, volumetric: 0.2,
           kind: walk, components: null, free: false, rate: null, seed: 0,
           validation: 0.25}
    model: {kind: recurrent, hidden: 32, rate: false, seed: 0}  # or kind: standard, internal: 8
    # (and strain: logarithmic or green-lagrange, the strain a model sees in finite strains)
    training: {epochs: 300, batch: 16, learning_rate: 0.01, final: 0.1, patience: null,
               clip: 1.0, rotate: false, seed: 0}
    paths: {cycle: "t=0,1,3 n=8,16 E11=0,0.01,-0.01"}
    output: model.npz
    save_data: data.npz

``data`` are the options of :func:`~morphogen.learning.data.random_paths` and the fraction
of validation paths, ``model`` those of :func:`~morphogen.learning.networks.make_model`
(``hidden`` by default 32, or two layers of 32; ``internal``, the internal variables of a
``standard`` model, by default 8; ``strain``, the strain that a model of any kind sees in
finite strains: ``logarithmic``, the default, ``1/2 ln C`` and its conjugate stress, or
``green-lagrange``, ``E`` and ``S``), ``training`` those of
:func:`~morphogen.learning.training.train` (``rotate``: true for isotropic behaviours,
``cubic`` for a cubic crystal in its axes). The learned law of a crystal runs in the axes
of its material, turned by the orientation of each grain (``learned: {model: ...,
oriented: true}``, the default); the elastic constants and the thermal expansion of a
reference material go to its learned material.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from morphogen.homogenization.paths import LoadingPath
from morphogen.learning.behaviour import learned_material
from morphogen.learning.data import generate, random_paths, responder
from morphogen.learning.networks import Model, make_model
from morphogen.learning.training import errors, train

#: The keys of a configuration, and the defaults of its sections
KEYS = ("material", "sample", "materials", "bc", "kinematics", "data", "model", "training",
        "paths", "output", "save_data")  # fmt: skip
DATA = {"paths": 128, "segments": 4, "increments": 8, "amplitude": 0.01, "volumetric": 0.2,
        "kind": "walk", "components": None, "free": False, "rate": None, "seed": 0,
        "validation": 0.25}  # fmt: skip
MODEL = {"kind": "recurrent", "hidden": None, "rate": False, "internal": None, "strain": None,
         "seed": 0}  # fmt: skip
TRAINING = {"epochs": 300, "batch": 16, "learning_rate": 0.01, "final": 0.1, "patience": None,
            "clip": 1.0, "rotate": False, "seed": 0}  # fmt: skip


def _section(cfg: dict, name: str, defaults: dict) -> dict:
    given = cfg.get(name) or {}
    if not isinstance(given, dict):
        raise ValueError(f"learning.{name}: a mapping")
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        raise ValueError(f"learning.{name}: unknown keys {unknown}; allowed {sorted(defaults)}")
    return {**defaults, **given}


def source_of(cfg: dict, base=None, kinematics: str = "small") -> Callable:
    """The response to a path of the ``material`` (a material point) or of the ``sample``
    (a voxel image, relative to ``base``) of ``materials`` (the FFT solver of the paths,
    under the boundary condition ``bc``) of a configuration."""
    if ("material" in cfg) == ("sample" in cfg):
        raise ValueError("learning: give a material (a material point) or a sample (an image)")
    if "material" in cfg:
        return responder(cfg["material"], kinematics)
    from morphogen import io
    from morphogen.homogenization import materials_from_config
    from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem

    if "materials" not in cfg:
        raise ValueError("learning.sample: give the materials of its phases")
    sample = Path(cfg["sample"])
    image = io.load_image(sample if base is None or sample.is_absolute() else Path(base) / sample)
    problem = NonlinearFFTProblem(image, materials_from_config(cfg["materials"]),
                                  cfg.get("bc", "periodic"), kinematics=kinematics)  # fmt: skip
    return problem.run


def _at(result, key: str, times) -> np.ndarray:
    """The strains or stresses (``key``) of a response at ``times`` (times x 6)."""
    values = np.asarray(getattr(result, key))
    return np.array([np.interp(times, result.times, values[:, c]) for c in range(6)]).T


def reference_elasticity(source) -> dict:
    """The elastic constants of a reference material (young, poisson), for the linear
    problems and the reference media of its learned law, and its thermal expansion; none
    of a microstructure, nor the constants of a crystal (the learned law then takes those
    of its initial tangent)."""
    from morphogen.homogenization.materials import Material

    if callable(source) and not isinstance(source, (dict, Material)):
        return {}
    reference = Material.from_config(source)
    out = {}
    if reference.young is not None:
        out = {"young": reference.young, "poisson": reference.poisson}
    if reference.expands:
        out["expansion"] = reference.expansion
    return out


def compare_path(material, respond, path, kinematics: str = "small", orientation=None) -> tuple:
    """The response of the reference ``respond`` along ``path`` (a
    :class:`~morphogen.homogenization.paths.LoadingPath`), that of the learned ``material``
    at a material point (turned by ``orientation``, Bunge angles; None where it fails) and
    their gaps, a row of :func:`path_errors`."""
    from morphogen.homogenization.nonlinear import simulate_point

    ref = respond(path)
    try:  # a learned law may have no solution for the stresses prescribed
        got = simulate_point(material, path, kinematics=kinematics, orientation=orientation,
                             max_cuts=4)  # fmt: skip
    except RuntimeError as exc:
        return ref, None, {"stress": math.inf, "strain": math.inf, "failed": str(exc)}
    times = np.concatenate([[path.times[0]], path.schedule()])
    row = {}
    for key in ("stress", "strain"):
        a, b = _at(got, key, times), _at(ref, key, times)
        row[key] = float(np.sqrt(np.sum((a - b) ** 2) / max(np.sum(b**2), 1e-300)))
    row["final_stress"] = np.asarray(got.stress)[-1].tolist()
    row["reference_stress"] = np.asarray(ref.stress)[-1].tolist()
    return ref, got, row


def path_errors(model, source, paths, kinematics: str = "small", orientation=None) -> dict:
    """The gaps of the learned law of ``model`` at a material point to the responses of
    ``source`` (:func:`~morphogen.learning.data.responder`) along named ``paths`` (name ->
    path, its specification or its text): the root mean square of the gaps of the stresses
    (of the strains) relative to that of the stresses (of the strains) of the reference
    at the planned steps, and the final stresses of both; infinite gaps and the message
    where the learned law fails (no solution for the stresses prescribed, after four
    divisions of a step). The learned material takes the elastic constants of a
    reference material (else those of its initial tangent); ``orientation`` (Bunge
    angles) turns both, a crystal and its law learned in its axes."""
    respond = responder(source, kinematics, orientation)
    material = learned_material(model, **reference_elasticity(source))
    out = {}
    for name, spec in dict(paths).items():
        path = LoadingPath.parse(spec) if isinstance(spec, str) else LoadingPath.from_spec(spec)
        out[str(name)] = compare_path(material, respond, path, kinematics, orientation)[2]
    return out


def learn(
    config: dict, base=None, *, callback=None, model: Model | None = None
) -> tuple[Model, dict]:
    """Learn the behaviour of a configuration (see the module): the trained model (or
    ``model``, trained further) and the summary of the data, of the training, of its
    errors and of the files written (relative to ``base``). ``callback``: that of
    :func:`~morphogen.learning.training.train`."""
    cfg = config.get("learning", config) if isinstance(config, dict) else None
    if not isinstance(cfg, dict):
        raise ValueError("learning: a mapping")
    unknown = sorted(set(cfg) - set(KEYS))
    if unknown:
        raise ValueError(f"learning: unknown keys {unknown}; allowed {sorted(KEYS)}")
    kinematics = cfg.get("kinematics", "small")
    base = None if base is None else Path(base)
    data_opts = _section(cfg, "data", DATA)
    model_opts = _section(cfg, "model", MODEL)
    training = _section(cfg, "training", TRAINING)
    source = source_of(cfg, base, kinematics)
    validation = float(data_opts.pop("validation"))
    count = int(data_opts.pop("paths"))
    paths = random_paths(count, **data_opts)
    info = {"source": cfg.get("material") or {"sample": str(cfg["sample"])},
            "kinematics": kinematics, "paths": {"count": count, **data_opts}}  # fmt: skip
    data = generate(source, paths, kinematics=kinematics, info=info)
    train_set, valid = data.split(validation, seed=int(data_opts["seed"]))
    if model is None:
        kind = model_opts.pop("kind")
        hidden = model_opts.pop("hidden")
        if kind not in ("recurrent", "standard"):
            model_opts.pop("rate")
        if hidden is not None:
            model_opts["hidden"] = hidden
        for key in ("internal", "strain"):
            value = model_opts.pop(key)
            if value is not None:
                if key == "internal" and kind != "standard":
                    raise ValueError("learning.model.internal: the internal variables of a "
                                     "standard model only")  # fmt: skip
                model_opts[key] = value
        model = make_model(kind, **model_opts)
    model.metadata["data"] = info
    fit = train(model, train_set, valid, callback=callback, **training)
    out = {"model": model.kind, "options": model.options, "paths": data.n_paths,
           "points": data.n_points, "training": fit.summary()}  # fmt: skip
    out["validation"] = {k: v for k, v in errors(model, valid).items() if k != "paths"}
    model.metadata["validation"] = out["validation"]
    if cfg.get("paths"):
        out["tests"] = path_errors(model, source, cfg["paths"], kinematics)
    for key, write in (("output", model.save), ("save_data", data.save)):
        if cfg.get(key):
            target = Path(cfg[key])
            target = target if base is None or target.is_absolute() else base / target
            out[key] = str(write(target))
    return model, out
