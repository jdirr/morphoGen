"""The tools of the tab Learning: a behaviour learned from data (:mod:`morphogen.learning`).
The responses of a behaviour at a material point, or the homogenized responses of a
morphology of the workspace (FFT on its voxels), along random strain paths; a network
trained on them in the background (its losses reported epoch by epoch); its learning
curves and its law along a validation path against the reference; the model saved
(``learned.npz``) and its learned law kept in the workspace (B), from which the loading
paths of morphologies give it to a phase.

The fields of the paths, of the model and of the training are those of the sections
``data``, ``model`` and ``training`` of the configurations of ``morphogen learn``
(:mod:`morphogen.learning.run`), with their defaults; the run writes the configuration that
learns the law again (a material point, or a morphology of isotropic phases: its voxel
image)."""

from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np

from morphogen.gui.tools import Field, elastic_tables, tab, tool
from morphogen.gui.tools.mechanics import (
    BEHAVIOURS_FIELD,
    COMPONENTS,
    SHADES,
    _behaviour,
    _behaviour_fields,
    _check_voxels,
    _json,
    _materials,
    _path,
    _path_fields,
)
from morphogen.learning.run import DATA, MODEL, TRAINING

tab(
    "learning",
    "Learning",
    "Behaviours learned from data: the responses of a behaviour at a material point, or "
    "the homogenized responses of a morphology, along random strain paths; a perceptron, "
    "a learned convex potential (hyperelastic) or a recurrent network (path-dependent "
    "laws: plasticity) trained on them in the background, its learning curves and its law "
    "along a validation path against the reference; the model saved and its law kept in "
    "the workspace (B), the behaviour of a phase in the loading paths of morphologies.",
    order=45,
)

POINT = {"data": "point"}
MORPHOLOGY = {"data": "morphology"}
#: The rotations of the mini-batches of the form (``rotate`` of the training)
ROTATIONS = {"false": False, "true": True, "cubic": "cubic"}
#: The models (:data:`morphogen.learning.KINDS`)
MODELS = (
    ("recurrent", "recurrent: a gated recurrent unit, learned internal variables (plasticity)"),
    (
        "standard",
        "standard: a learned convex free energy and dissipation of learned internal "
        "variables (plasticity, viscosity; thermodynamically consistent)",
    ),
    ("potential", "potential: the stress of a learned convex energy (hyperelasticity)"),
    ("feedforward", "perceptron: the stress a function of the strain (nonlinear elasticity)"),
)


def _fields(section: dict, specs: dict) -> list[Field]:
    """The fields of a section of the configurations of ``morphogen learn``, their
    defaults those of ``section``: ``specs`` maps each key of the section to the options of
    its :class:`~morphogen.gui.tools.Field` (``name``: that of the form, by default the
    key)."""
    out = []
    for key, spec in specs.items():
        spec = dict(spec)
        out.append(Field(spec.pop("name", key), default=spec.pop("default", section[key]), **spec))
    return out


#: The fields of the random paths (the section ``data``)
PATH_FIELDS = _fields(
    DATA,
    {
        "paths": {"kind": "int", "doc": "random strain paths, their responses the data"},
        "segments": {"kind": "int", "doc": "linear segments of a path, from the natural state"},
        "increments": {"kind": "int", "label": "increments of a segment"},
        "amplitude": {
            "kind": "float",
            "label": "strain amplitude",
            "doc": "of the tensor components (a random walk held within it)",
        },
        "volumetric": {
            "kind": "float",
            "label": "volumetric ratio",
            "doc": "of the volumetric strains: 0.2 for metals (elastic, small volume "
            "changes), 1 for porous media",
        },
        "kind": {
            "kind": "choice",
            "label": "paths",
            "choices": (
                ("walk", "random walks: each strain the previous one plus a random step"),
                ("linear", "independent strains: each component uniform"),
            ),
        },
        "components": {
            "kind": "choices",
            "default": list(COMPONENTS),
            "label": "components driven",
            "choices": COMPONENTS,
            "doc": "the others held at zero strain, or free; a learned law runs the paths "
            "of the components of its data only",
        },
        "free": {
            "kind": "bool",
            "label": "the others free",
            "doc": "the components not driven at zero stress, rather than at zero strain "
            "(a tension-torsion test drives 11 and 12)",
        },
        "rate": {
            "kind": "float",
            "label": "strain rate",
            "unit": "1/time",
            "doc": "the largest rate of a component on each segment (blank: a unit time "
            "each); rate-dependent behaviours",
        },
        "seed": {"kind": "int", "label": "seed of the paths"},
        "validation": {
            "kind": "float",
            "label": "validation fraction",
            "doc": "the paths of the early stopping and of the errors, kept out of the training",
        },
    },
)
#: The fields of the model (the section ``model``)
MODEL_FIELDS = _fields(
    MODEL,
    {
        "kind": {"name": "model", "kind": "choice", "choices": MODELS},
        "hidden": {
            "kind": "ints",
            "label": "hidden sizes",
            "doc": "the size of the state of a recurrent network, the ridges of the free "
            "energy of a standard model, or the hidden layers of the others (blank: 32; 32, "
            "32)",
        },
        "rate": {
            "name": "model_rate",
            "kind": "bool",
            "label": "time step as an input",
            "when": {"model": ["recurrent", "standard"]},
            "doc": "rate-dependent behaviours (viscoplasticity; a viscous dissipation of a "
            "standard model)",
        },
        "internal": {
            "kind": "int",
            "label": "internal variables",
            "when": {"model": "standard"},
            "doc": "of a standard model (blank: 8)",
        },
        "strain": {
            "name": "energy_strain",
            "kind": "choice",
            "default": "logarithmic",
            "label": "strain of the law",
            "when": {"kinematics": "finite"},
            "choices": (
                (
                    "logarithmic",
                    "logarithmic: ½ ln C and its conjugate stress T, S = Jᵀ T, as the Hencky "
                    "points (plasticity, Hencky's elasticity)",
                ),
                ("green-lagrange", "Green-Lagrange: E and the second Piola-Kirchhoff stress"),
            ),
            "doc": "in finite strains, the strain the model sees: the input of a perceptron or "
            "of a recurrent network (its increments), the strain of a potential or of the free "
            "energy of a standard model (its internal variables additive in it)",
        },
        "seed": {"name": "model_seed", "kind": "int", "label": "seed of the parameters"},
    },
)
#: The fields of the training (the section ``training``)
TRAINING_FIELDS = _fields(
    TRAINING,
    {
        "epochs": {"kind": "int"},
        "batch": {
            "kind": "int",
            "label": "mini-batch",
            "doc": "paths of a recurrent model, points of the others",
        },
        "learning_rate": {"kind": "float", "label": "learning rate", "doc": "of Adam"},
        "final": {
            "kind": "float",
            "label": "final rate",
            "doc": "a fraction of the first, reached geometrically at the last epoch",
        },
        "patience": {
            "kind": "int",
            "doc": "epochs without a better validation loss before it stops (blank: all)",
        },
        "clip": {"kind": "float", "label": "gradient clipping", "doc": "its global norm"},
        "rotate": {
            "kind": "choice",
            "default": "false",
            "label": "random rotations",
            "choices": (
                ("false", "none"),
                ("true", "uniform: an isotropic behaviour, data of every direction"),
                ("cubic", "of the cube: a cubic crystal in its axes (24 rotations)"),
            ),
            "doc": "each mini-batch turned at each epoch by symmetries of the behaviour: "
            "data without more responses",
        },
        "seed": {"name": "training_seed", "kind": "int", "label": "seed of the shuffles"},
    },
)


@tool(
    "learn",
    "learning",
    "Learn a behaviour",
    "A behaviour learned from data: the responses of a behaviour at a material point, or "
    "the homogenized responses of a morphology (FFT on its voxels), along random strain "
    "paths, then a network trained on them in the background (Adam, early stopping on the "
    "validation paths; its losses reported at each epoch). Its learning curves, its errors "
    "on the validation paths and its law at a material point along a validation path "
    "against the reference; the model saved (learned.npz, with the data and the "
    "configuration that learns it again), and the learned law joins the workspace (B): the "
    "loading paths of morphologies give it to a phase (a behaviour of the workspace).",
    [
        Field(
            "data",
            "choice",
            "point",
            label="data of",
            choices=(
                ("point", "a behaviour at a material point"),
                ("morphology", "a morphology: its homogenized responses (FFT on its voxels)"),
            ),
        ),
        Field("source", "item", label="morphology", required=True, when=MORPHOLOGY),
        Field(
            "materials",
            "materials",
            of="source",
            when=MORPHOLOGY,
            doc="elastic constants of the phases (the unit of the stresses), or voids",
        ),
        Field(
            "phase",
            "int",
            label="phase of the behaviour",
            when=MORPHOLOGY,
            doc="blank: the first phase that is not a void",
        ),
        Field("young", "float", 1.0, label="Young's modulus E", unit="stress", when=POINT),
        Field("poisson", "float", 0.3, label="Poisson's ratio ν", when=POINT),
        *_behaviour_fields("learned: at the point, or that phase of the morphology"),
        Field(
            "material",
            "text",
            label="material of the configurations (JSON)",
            when=POINT,
            doc='replaces the behaviour of the form: e.g. {"young": 1, "poisson": 0.3, '
            '"plasticity": {"yield": 0.002, "hardening": 0.05}}',
        ),
        replace(BEHAVIOURS_FIELD, when=MORPHOLOGY),
        Field("voxels", "int", 8, label="voxels per edge", when=MORPHOLOGY),
        Field(
            "integration",
            "choice",
            "bbar",
            when=MORPHOLOGY,
            choices=(
                ("bbar", "B-bar (F-bar in finite strains): no locking under plastic flow"),
                ("full", "full: 8 Gauss points per voxel"),
            ),
        ),
        Field(
            "kinematics",
            "choice",
            "small",
            choices=(
                ("small", "small strains"),
                (
                    "finite",
                    "finite strains: the Green-Lagrange strain, the second Piola-Kirchhoff stress",
                ),
            ),
        ),
        *PATH_FIELDS,
        *MODEL_FIELDS,
        *TRAINING_FIELDS,
        *_path_fields("cycle", prefix="check_"),
        Field(
            "check_orientation",
            "floats",
            label="orientation of the validation point",
            unit="degrees",
            when=POINT,
            doc="Bunge angles φ1, Φ, φ2: the behaviour and its learned law turned by them "
            "(a crystal learned in its axes, checked in another frame); blank: the axes of "
            "the data",
        ),
    ],
    order=1,
    groups=(
        ("Data", "data"),
        ("Behaviour", "behaviour"),
        ("Solver", "voxels"),
        ("Random paths", "paths"),
        ("Model", "model"),
        ("Training", "epochs", True),
        ("Validation path", "check_loading"),
    ),
)
def learn_behaviour(p: dict, ctx) -> dict:
    from morphogen.gui.api import behaviour_summary
    from morphogen.homogenization.paths import LoadingPath
    from morphogen.learning import (
        errors,
        generate,
        learned_material,
        random_paths,
        responder,
        train,
    )
    from morphogen.learning.run import compare_path, reference_elasticity

    kinematics = p["kinematics"]
    name, check = _path(p, "check_")
    check = LoadingPath.from_spec(check)
    orientation = p.get("check_orientation") or None
    if orientation is not None and len(orientation) != 3:
        raise ValueError("orientation of the validation point: three Bunge angles φ1, Φ, φ2")
    rotate = ROTATIONS[p["rotate"]]
    respond, source, sample, origin = _source(p, ctx, kinematics, len(check.schedule()))
    options = {k: p[k] for k in ("segments", "increments", "amplitude", "volumetric", "kind")}
    options.update(components=p["components"] or None, free=bool(p["free"]), rate=p["rate"],
                   seed=p["seed"])  # fmt: skip
    if not 0.0 < p["validation"] < 1.0 or p["paths"] < 2:
        raise ValueError("validation fraction: between 0 and 1, of two paths at least")
    paths = random_paths(p["paths"], **options)
    model = _model(p)
    info = {"source": source, "kinematics": kinematics, "paths": {"count": p["paths"], **options}}

    def generated(done, total):
        ctx.progress(0.02 + 0.38 * done / total, f"data: path {done} of {total}")

    started = time.perf_counter()
    data = generate(respond, paths, kinematics=kinematics, info=info, callback=generated)
    data_s = time.perf_counter() - started
    train_set, valid = data.split(p["validation"], seed=p["seed"])
    epochs = p["epochs"]

    def trained(epoch, loss, check_loss):
        ctx.progress(0.4 + 0.55 * (epoch + 1) / epochs,
                     f"epoch {epoch + 1} of {epochs}: loss {loss:.3g}, validation "
                     f"{check_loss:.3g}")  # fmt: skip

    model.metadata["data"] = info
    fit = train(model, train_set, valid, epochs=epochs, batch=p["batch"],
                learning_rate=p["learning_rate"], final=p["final"], patience=p["patience"],
                clip=p["clip"], rotate=rotate, seed=p["training_seed"],
                callback=trained)  # fmt: skip
    scores = errors(model, valid)
    model.metadata["validation"] = {k: v for k, v in scores.items() if k != "paths"}
    path = model.save(ctx.folder / "learned.npz")
    data.save(ctx.folder / "data.npz")
    ctx.progress(0.96, f"validation path {name}")
    elastic = reference_elasticity(source) if p["data"] == "point" else {}
    material = learned_material(model, **elastic)
    if orientation is not None:  # the behaviour at the point turned with its law
        respond = responder(source, kinematics, orientation)
    ref, got, gaps = compare_path(material, respond, check, kinematics, orientation)
    ctx.figure(_curves_figure(fit), "learning_curves", "learning curves")
    ctx.figure(_check_figure(ref, got, f"validation path {name}"), "validation_path",
               f"validation path {name}: the learned law against the reference")  # fmt: skip
    ctx.table([{"epoch": k + 1, "training loss": a, "validation loss": b} for k, (a, b) in
               enumerate(zip(fit.train_loss, fit.validation_loss, strict=True))],
              "learning_curves", "learning curves")  # fmt: skip
    ctx.table([{"path": k + 1, "relative error": e} for k, e in enumerate(scores["paths"])],
              "validation_paths", "errors of the validation paths")  # fmt: skip
    ctx.table(ref.rows(), f"reference_{name}", f"reference along the path {name}")
    _elastic_comparison(ctx, model, source if p["data"] == "point" else None)
    if got is not None:
        ctx.table(got.rows(), f"learned_{name}", f"learned law along the path {name}")
    else:
        import warnings

        warnings.warn(f"the learned law along the path {name}: {gaps['failed']}", stacklevel=1)
    # the behaviour of the workspace: the model by its name in the folder of the run
    config = material.to_config()
    config["learned"] = {**config["learned"], "model": path.name}
    (ctx.folder / "material.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    _configuration(p, ctx, source, sample, options, name, check)
    label, details = behaviour_summary(config)
    details.update(model=model.kind, data=data.measure, validation=round(scores["relative"], 4))
    item = ctx.session.add(ctx.key, "behaviour", f"{label} ({model.kind}) of {origin}",
                           material=config, info=details)  # fmt: skip
    values = {
        "paths": data.n_paths,
        "points per path": data.n_points,
        "epochs": fit.epochs,
        "best epoch": fit.best_epoch + 1,
        "training loss": fit.train_loss[fit.best_epoch],
        "validation loss": fit.best_loss,
        "validation error": scores["relative"],
        "validation error, worst path": scores["max"],
        f"stress gap along {name}": gaps["stress"],
        f"strain gap along {name}": gaps["strain"],
        "data_s": round(data_s, 2),
        "training_s": round(fit.time_s, 2),
        "young": material.young,
        "poisson": material.poisson,
    }
    return {"values": values, "items": [ctx.session.public(item)]}


def _elastic_comparison(ctx, model, reference) -> None:
    """The elasticity of the learned law, its tangent at the natural state (stiffness and
    compliance, ``tangent_stiffness.csv`` and ``tangent_compliance.csv``), beside that of
    the reference material of a material point (``reference_stiffness.csv``...; none for
    a morphology, or a law without elastic constants of its own)."""
    from morphogen.homogenization.materials import Material
    from morphogen.learning.behaviour import initial_tangent

    names = ("tangent_stiffness", "tangent_compliance")
    try:
        elastic_tables(ctx, initial_tangent(model), "of the learned law at the natural state",
                       names=names, kind="tangent")  # fmt: skip
    except np.linalg.LinAlgError:  # a learned law without a stiffness at the natural state
        pass
    try:
        material = Material.from_config(reference) if reference is not None else None
    except (ValueError, TypeError, KeyError, OSError):
        material = None
    if material is not None and material.elastic:
        elastic_tables(ctx, material.matrix("elastic"), "of the reference",
                       names=("reference_stiffness", "reference_compliance"))  # fmt: skip


def _source(p: dict, ctx, kinematics: str, steps: int):
    """The response to a path of the source of the form (``respond``), its description
    (the material of a material point, or the morphology and its materials), the voxel
    image of a morphology (None at a material point, or when its grains are turned) and
    the name of the source in the title of the learned law."""
    from morphogen.gui.api import behaviour_summary
    from morphogen.homogenization.materials import Material
    from morphogen.learning import responder

    if p["data"] == "point":
        spec = _json(p.get("material"), "material of the configurations")
        kept = p["kept"] if spec is None and p["behaviour"] == "workspace" else None
        if spec is None:
            spec = _behaviour(p, {"young": p["young"], "poisson": p["poisson"]}, ctx.folder)
        Material.from_config(spec)  # checked
        origin = kept["title"] if kept else behaviour_summary(spec)[0]
        return responder(spec, kinematics), spec, None, origin
    from morphogen.homogenization import materials_from_config
    from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem
    from morphogen.homogenization.materials import any_oriented

    item = p["source"]
    micro = item["micro"]
    materials, phase = _materials(p, item, ctx.folder)
    n = p["voxels"]
    if not 4 <= n <= 64:
        raise ValueError("voxels per edge: from 4 to 64 for the data of a learned law")
    _check_voxels(micro, n, steps, "learning on voxels")
    image = micro.voxelize(shape=n)
    mats = materials_from_config(materials)
    grains = {}
    if any_oriented(mats) and getattr(micro, "orientations", None) is not None:
        grains = {"grains": micro.labels(grid=image.grid()), "orientations": micro.orientations}
    problem = NonlinearFFTProblem(image, mats, "periodic", kinematics=kinematics,
                                  integration=p["integration"], **grains)  # fmt: skip
    source = {"morphology": item["title"], "phase": phase, "voxels": n,
              "materials": {str(k): v for k, v in materials.items()}}  # fmt: skip
    return problem.run, source, None if grains else image, f"#{item['number']}"


def _model(p: dict):
    """The model of the form (:func:`~morphogen.learning.make_model`)."""
    from morphogen.learning import make_model

    options = {"seed": p["model_seed"]}
    if p.get("energy_strain"):  # in finite strains
        options["strain"] = p["energy_strain"]
    hidden = p.get("hidden")
    if p["model"] in ("recurrent", "standard"):
        options["rate"] = bool(p.get("model_rate"))
        if p["model"] == "standard" and p.get("internal"):
            options["internal"] = p["internal"]
        if hidden:
            if len(hidden) != 1:
                raise ValueError("hidden sizes: one, the size of the state of a recurrent "
                                 "network or the ridges of a standard model")  # fmt: skip
            options["hidden"] = hidden[0]
    elif hidden:
        options["hidden"] = list(hidden)
    return make_model(p["model"], **options)


def _curves_figure(fit):
    """The losses of the training and validation sets at each epoch, the best one marked."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    epochs = np.arange(1, fit.epochs + 1)
    ax.semilogy(epochs, fit.train_loss, "-", lw=1.6, color=SHADES[1], label="training paths")
    ax.semilogy(epochs, fit.validation_loss, "-", lw=1.6, color=SHADES[0],
                label="validation paths")  # fmt: skip
    best = fit.best_epoch + 1
    ax.axvline(best, color="0.5", ls=":", lw=1.0)
    ax.plot([best], [fit.best_loss], "o", color=SHADES[0], ms=5,
            label=f"kept: epoch {best}, {fit.best_loss:.3g}")  # fmt: skip
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean square of the normalized stress gaps")
    ax.set_title(f"training of the {fit.model.kind} model", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, frameon=False)
    return fig


def _check_figure(ref, got, title: str):
    """The learned law (dashed) against the reference (solid) along a path: the stress of
    the strain that varies most, and the stresses that vary against time."""
    from matplotlib.figure import Figure

    from morphogen.gui.tools.mechanics import _driven

    fig = Figure(figsize=(9.6, 3.8), layout="constrained")
    ax, ax2 = fig.subplots(1, 2)
    strain, stress, e_name, s_name, i = _driven(ref)
    j = int(np.argmax(np.ptp(stress, axis=0)))
    ax.plot(strain[:, i], stress[:, j], "k-", lw=1.8, marker=".", ms=4, label="reference")
    ptp = np.ptp(stress, axis=0)
    shown = [c for c in range(6) if ptp[c] > 0.05 * ptp.max()]
    for k, c in enumerate(shown):
        shade = SHADES[k % len(SHADES)]
        label = f"{s_name}{COMPONENTS[c]}"
        ax2.plot(ref.times, stress[:, c], "-", lw=1.8, color=shade, label=label)
    if got is not None:
        e2, s2, *_ = _driven(got)
        ax.plot(e2[:, i], s2[:, j], "--", color=SHADES[0], lw=1.8, label="learned law")
        for k, c in enumerate(shown):
            ax2.plot(got.times, s2[:, c], "--", lw=1.4, color=SHADES[k % len(SHADES)])
    ax.set_xlabel(e_name + COMPONENTS[i])
    ax.set_ylabel(s_name + COMPONENTS[j])
    ax.locator_params(axis="x", nbins=5)
    ax2.set_xlabel("time")
    ax2.set_ylabel("stress (solid: reference, dashed: learned)")
    for a in (ax, ax2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8, frameon=False)
    fig.suptitle(title + ("" if got is not None else ": the learned law failed"), fontsize=10)
    return fig


def _configuration(p: dict, ctx, source, sample, options: dict, name: str, check) -> None:
    """The configuration of ``morphogen learn`` that learns the law again (beside the files
    it names): at a material point, or on the voxel image of a morphology (not of turned
    grains)."""
    import yaml

    from morphogen import io

    cfg: dict = {}
    if p["data"] == "point":
        cfg["material"] = source
    elif sample is not None:
        io.save_image(sample, ctx.folder / "sample.vti")
        cfg.update(sample="sample.vti", materials=source["materials"], bc="periodic")
    else:
        return
    model = {"kind": p["model"], "hidden": p.get("hidden") or None, "seed": p["model_seed"]}
    if p["model"] in ("recurrent", "standard"):
        model["rate"] = bool(p.get("model_rate"))
        if model["hidden"]:
            model["hidden"] = model["hidden"][0]
    if p["model"] == "standard" and p.get("internal"):
        model["internal"] = p["internal"]
    if p.get("energy_strain"):
        model["strain"] = p["energy_strain"]
    training = {k: p[k] for k in ("epochs", "batch", "learning_rate", "final", "patience", "clip")}
    training.update(rotate=ROTATIONS[p["rotate"]], seed=p["training_seed"])
    cfg.update(kinematics=p["kinematics"],
               data={"paths": p["paths"], **options, "validation": p["validation"]},
               model=model, training=training, paths={name: check.to_dict()},
               output="learned.npz", save_data="data.npz")  # fmt: skip
    text = yaml.safe_dump({"learning": json.loads(json.dumps(cfg))}, sort_keys=False)
    (ctx.folder / "learning.yaml").write_text(text, encoding="utf-8")
