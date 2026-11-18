"""The tool of the tab Learning: its form built from the sections of the configurations of
``morphogen learn``, a behaviour learned at a material point and from a morphology in the
background (its losses reported), its curves, files and configuration, and its learned
law kept in the workspace, given to a phase by a loading path, saved and opened."""

import json

import numpy as np
import pytest
import yaml

from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.tools import ToolJob
from morphogen.learning import learn, load_model
from morphogen.learning.run import DATA, MODEL, TRAINING
from morphogen.models.base import model_from_config
from morphogen.seeding import derive_seed

E, NU = 1.0, 0.3
#: Isotropic elasticity learned by a perceptron: 0.2 % on the validation paths
ELASTIC = {"behaviour": "elastic", "paths": 10, "segments": 3, "increments": 4,
           "model": "feedforward", "hidden": "8, 8", "epochs": 50, "batch": 32,
           "check_loading": "tension", "check_amplitude": 0.004,
           "check_increments": 4}  # fmt: skip


def _workspace(tmp_path):
    """A session whose workspace holds a Boolean model of spheres."""
    session = api.Session(tmp_path)
    model = model_from_config({"type": "boolean_spheres", "radius": 0.6, "volume_fraction": 0.2})
    micro = model.generate(Domain(size=4.0, periodic=True), derive_seed(7, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "boolean_spheres (seed 7)", micro=micro, seed=7,
                model=model.to_config(), domain={"size": 4.0, "periodic": True},
                phases={"0": "matrix", "1": "spheres"})  # fmt: skip
    return session, key


def _run(session, name, params, messages=None):
    job = ToolJob({"tool": name, "params": params}, session)
    if messages is not None:  # the progress the page follows
        job.report = lambda fraction, message="": messages.append(message)
    job._run()  # synchronously
    status = job.status()
    assert status["state"] == "done", status.get("error")
    return status["result"]


def _error(session, name, params) -> str:
    try:
        job = ToolJob({"tool": name, "params": params}, session)
    except ValueError as exc:  # the errors of the form
        return str(exc)
    job._run()
    assert job.status()["state"] == "error"
    return job.status()["error"]


def test_the_form_of_the_sections_of_the_configurations():
    """The tab sits in Simulate; the fields of the paths, the model and the training take
    the defaults of the configurations of morphogen learn."""
    catalog = tools.catalog()
    tab = next(t for t in catalog["tabs"] if t["name"] == "learning")
    assert tab["section"] == "simulate" and tab["title"] == "Learning"
    (learn_tool,) = [t for t in catalog["tools"] if t["tab"] == "learning"]
    fields = {f["name"]: f for f in learn_tool["fields"]}
    assert fields["paths"]["default"] == DATA["paths"] and fields["kind"]["default"] == "walk"
    assert fields["validation"]["default"] == DATA["validation"]
    assert fields["model"]["default"] == MODEL["kind"] == "recurrent"
    assert fields["epochs"]["default"] == TRAINING["epochs"]
    assert fields["learning_rate"]["default"] == TRAINING["learning_rate"]
    assert fields["model_rate"]["when"] == {"model": ["recurrent", "standard"]}
    assert fields["internal"]["when"] == {"model": ["standard"]}
    assert fields["internal"]["default"] is MODEL["internal"] is None
    assert fields["energy_strain"]["when"] == {"kinematics": ["finite"]}  # every model
    assert fields["energy_strain"]["default"] == "logarithmic"
    assert fields["source"]["when"] == {"data": ["morphology"]} and fields["source"]["required"]
    assert fields["behaviours"]["when"] == {"data": ["morphology"]}
    assert fields["kept"]["items"] == ["behaviour"]
    assert fields["paths"]["group"] == "Random paths" and fields["epochs"]["fold"] is True
    assert fields["check_loading"]["group"] == "Validation path"


def test_elasticity_learned_at_a_material_point(tmp_path):
    """Isotropic elasticity learned by a perceptron: its losses reported epoch by epoch, its
    curves, errors and validation path against the reference within 1 %; the model, the
    data and the configuration that learns it again written; the learned law kept in the
    workspace by the name of its model in its folder."""
    session, _ = _workspace(tmp_path)
    messages = []
    out = _run(session, "learn", ELASTIC, messages)
    assert any(m.startswith("data: path 10 of 10") for m in messages)
    assert any(m.startswith("epoch 50 of 50: loss") for m in messages)
    values = out["values"]
    assert values["paths"] == 10 and values["points per path"] == 13
    assert values["validation error"] < 0.01  # 0.2 % measured
    assert values["stress gap along tension"] < 0.01 and values["strain gap along tension"] < 0.01
    assert values["young"] == E and values["poisson"] == NU
    assert [f["title"] for f in out["figures"]] == [
        "learning curves",
        "validation path tension: the learned law against the reference",
    ]
    curves = next(t for t in out["tables"] if t["title"] == "learning curves")
    # its elasticity: the tangent of the learned law at the natural state against the reference
    tables = {t["title"].split(" of ")[0]: t for t in out["tables"]}
    tangent = np.array([r[1:] for r in tables["tangent stiffness C_ij"]["rows"]])
    inverse = np.array([r[1:] for r in tables["tangent compliance S_ij = C⁻¹"]["rows"]])
    assert np.allclose(inverse @ tangent, np.eye(6), atol=1e-4)
    reference = np.array([r[1:] for r in tables["stiffness C_ij"]["rows"]])
    assert reference[3, 3] == pytest.approx(E / (2 * (1 + NU)), rel=1e-5)  # Voigt: G23
    assert np.abs(tangent - reference).max() < 0.05 * reference.max()
    assert np.array([r[1:] for r in tables["compliance S_ij = C⁻¹"]["rows"]])[0, 0] == (
        pytest.approx(1 / E, rel=1e-5)
    )
    assert curves["columns"] == ["epoch", "training loss", "validation loss"]
    assert curves["count"] == values["epochs"] == 50
    names = {f["name"] for f in out["files"]}
    assert {"learned.npz", "data.npz", "learning.yaml", "material.json",
            "learned_tension.csv", "reference_tension.csv"} <= names  # fmt: skip
    kept = session.item(out["key"], ("behaviour",))
    assert kept["title"] == "#2 learned law (feedforward) of elasticity"
    assert kept["material"] == {"young": E, "poisson": NU, "learned": {"model": "learned.npz"}}
    assert kept["info"]["model"] == "feedforward" and "curves" in api.item_traits(kept)
    folder = tmp_path / out["key"]
    material = api.behaviour_material(kept, session.root)
    assert material["learned"]["model"] == str(folder / "learned.npz")
    model = load_model(folder / "learned.npz")
    assert model.kind == "feedforward" and model.metadata["data"]["source"]["young"] == E
    # the configuration of morphogen learn learns it again
    config = yaml.safe_load((folder / "learning.yaml").read_text(encoding="utf-8"))
    config["learning"].update(output=None, save_data=None)
    again, summary = learn(config, base=folder)
    assert summary["validation"]["relative"] == pytest.approx(values["validation error"], rel=1e-9)
    assert np.array_equal(again.params["L"], model.params["L"])


def test_a_learned_law_given_to_a_phase(tmp_path):
    """A loading path gives the learned law of the workspace to the matrix of a morphology:
    the response of the elastic matrix within 1 %; saved and opened with the workspace,
    the law still runs a material point."""
    session, key = _workspace(tmp_path / "first")
    kept = session.item(_run(session, "learn", ELASTIC)["key"], ("behaviour",))
    common = {"source": key, "method": "fft", "voxels": 6, "increments": 3, "amplitude": 0.004,
              "video": False, "phase": 0,
              "materials": {"0": {"young": E, "poisson": NU}, "1": {"young": 10 * E,
                                                                   "poisson": NU}}}  # fmt: skip
    elastic = _run(session, "path", {**common, "behaviour": "elastic"})["values"]
    learned = _run(session, "path", {**common, "behaviour": "workspace", "kept": kept["key"]})
    assert learned["values"]["S11"] == pytest.approx(elastic["S11"], rel=0.01)
    saved = api.save_workspace({"directory": str(tmp_path / "saved")}, session)
    second = api.Session(tmp_path / "second")
    opened = [i for i in api.open_workspace({"directory": saved["directory"]}, second)["items"]
              if i["kind"] == "behaviour"]  # fmt: skip
    assert opened and opened[0]["tool"] == "learn"
    point = _run(second, "point", {"loading": "tension", "amplitude": 0.004, "increments": 4,
                                   "behaviour": "workspace", "kept": opened[0]["key"]})  # fmt: skip
    assert point["values"]["S11"] == pytest.approx(E * 0.004, rel=0.01)


def test_a_crystal_learned_in_its_axes(tmp_path):
    """A cubic crystal learned in its axes, its paths turned by the rotations of the cube,
    checked turned by an orientation against the crystal turned; the material point of
    Mechanics turns the learned law of the workspace by an orientation as it turns
    crystals: a uniaxial tension along another direction of the crystal."""
    from morphogen.homogenization.materials import Material

    session, _ = _workspace(tmp_path)
    crystal = {"cubic": [1.7, 1.2, 0.75]}
    out = _run(session, "learn", {**ELASTIC, "material": json.dumps(crystal), "rotate": "cubic",
                                  "check_orientation": "30, 50, 70"})  # fmt: skip
    assert out["values"]["stress gap along tension"] < 0.01
    kept = session.item(out["key"], ("behaviour",))
    tension = {"loading": "tension", "amplitude": 0.004, "increments": 4,
               "behaviour": "workspace", "kept": kept["key"]}  # fmt: skip
    turned = _run(session, "point", {**tension, "orientation": "30, 50, 70"})["values"]["S11"]
    axes = _run(session, "point", tension)["values"]["S11"]
    compliance = np.linalg.inv(Material.from_config(crystal).matrix("elastic", [30, 50, 70]))
    assert turned == pytest.approx(0.004 / compliance[0, 0], rel=0.01)
    assert axes == pytest.approx(0.004 / np.linalg.inv(Material.from_config(crystal).matrix(
        "elastic"))[0, 0], rel=0.01)  # fmt: skip
    assert abs(turned / axes - 1) > 0.2  # along [100] the crystal is softest


def test_a_standard_law_learned_at_a_material_point(tmp_path):
    """Von Mises plasticity learned by a standard model (a learned free energy and
    dissipation of internal variables): its options in the configuration that learns it
    again to the bit, the learned law kept in the workspace; in finite strains, the strain
    of its free energy, and that of a recurrent network (logarithmic by default), in the
    model and the configuration."""
    session, _ = _workspace(tmp_path)
    params = {"behaviour": "mises", "yield_stress": 0.002, "hardening": 0.05, "paths": 6,
              "segments": 2, "increments": 3, "model": "standard", "hidden": "4",
              "internal": 3, "epochs": 4, "batch": 4, "check_loading": "tension",
              "check_increments": 3}  # fmt: skip
    out = _run(session, "learn", params)
    kept = session.item(out["key"], ("behaviour",))
    assert kept["title"].startswith("#2 learned law (standard) of")
    folder = tmp_path / out["key"]
    model = load_model(folder / "learned.npz")
    assert model.kind == "standard" and model.n_state == 3
    config = yaml.safe_load((folder / "learning.yaml").read_text(encoding="utf-8"))
    assert config["learning"]["model"] == {"kind": "standard", "hidden": 4, "seed": 0,
                                           "rate": False, "internal": 3}  # fmt: skip
    config["learning"].update(output=None, save_data=None)
    again, _ = learn(config, base=folder)
    assert all(np.array_equal(again.params[k], model.params[k]) for k in model.params)
    # in finite strains, its free energy of the Green-Lagrange strain
    out = _run(session, "learn", {**params, "kinematics": "finite", "amplitude": 0.05,
                                  "energy_strain": "green-lagrange"})  # fmt: skip
    folder = tmp_path / out["key"]
    model = load_model(folder / "learned.npz")
    assert model.measure == "green-lagrange" and model.options["strain"] == "green-lagrange"
    config = yaml.safe_load((folder / "learning.yaml").read_text(encoding="utf-8"))
    assert config["learning"]["model"]["strain"] == "green-lagrange"
    assert config["learning"]["kinematics"] == "finite"
    out = _run(session, "learn", {**params, "model": "recurrent", "kinematics": "finite",
                                  "amplitude": 0.05})  # fmt: skip
    folder = tmp_path / out["key"]
    model = load_model(folder / "learned.npz")
    assert model.kind == "recurrent" and model.logarithmic
    config = yaml.safe_load((folder / "learning.yaml").read_text(encoding="utf-8"))
    assert config["learning"]["model"]["strain"] == "logarithmic"


def test_a_recurrent_law_of_a_morphology(tmp_path):
    """The homogenized responses of a morphology (FFT on its voxels) as data, a recurrent
    model trained on them: its validation path run against the morphology, its voxel image
    and configuration written; the form's errors."""
    session, key = _workspace(tmp_path)
    params = {"data": "morphology", "source": key, "voxels": 4, "behaviour": "mises",
              "yield_stress": 0.002, "hardening": 0.05, "isotropic": "0, 1", "paths": 4,
              "segments": 2, "increments": 2, "hidden": "4", "epochs": 3, "batch": 2,
              "check_loading": "tension", "check_increments": 2}  # fmt: skip
    out = _run(session, "learn", params)
    values = out["values"]
    assert values["paths"] == 4 and values["epochs"] == 3 and values["young"] > 1.0
    kept = session.item(out["key"], ("behaviour",))
    assert kept["title"] == "#2 learned law (recurrent) of #1"
    folder = tmp_path / out["key"]
    config = yaml.safe_load((folder / "learning.yaml").read_text(encoding="utf-8"))["learning"]
    assert config["sample"] == "sample.vti" and (folder / "sample.vti").is_file()
    assert config["materials"]["0"]["plasticity"]["yield"] == 0.002
    assert config["model"] == {"kind": "recurrent", "hidden": 4, "seed": 0, "rate": False}
    assert load_model(folder / "learned.npz").metadata["data"]["source"]["voxels"] == 4
    assert "one, the size of the state" in _error(session, "learn", {**params, "hidden": "4, 4"})
    assert "between 0 and 1" in _error(session, "learn", {**params, "validation": 1.0})
    assert "from 4 to 64" in _error(session, "learn", {**params, "voxels": 2})
