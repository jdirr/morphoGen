"""The tools of the tab Contact: their catalog and forms; a porous morphology compressed
between two platens (softer than its solid, the same as the test run directly, its curve,
its figure and the file of the deformed sample) and a morphology whose phases are alike
indented by a sphere (the reduced modulus of its solid with the base of the sample, the
slope of the unloading) and by a cone; the errors of the forms."""

import numpy as np
import pytest

from morphogen.contact import effective_modulus
from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.tools import ToolJob
from morphogen.mesh.conforming import has_mmg

mmg = pytest.mark.skipif(not has_mmg(), reason="MMG")
SOLID = {"young": 1000.0, "poisson": 0.3}


def _run(session, name, params):
    job = ToolJob({"tool": name, "params": params}, session)
    job._run()  # synchronously
    status = job.status()
    assert status["state"] == "done", status.get("error")
    return status["result"]


def _error(session, name, params) -> str:
    try:
        job = ToolJob({"tool": name, "params": params}, session)
    except ValueError as exc:  # an error of the form
        return str(exc)
    job._run()
    assert job.status()["state"] == "error"
    return job.status()["error"]


def _table(result, title):
    table = next(t for t in result["tables"] if t["title"].startswith(title))
    return [dict(zip(table["columns"], row, strict=True)) for row in table["rows"]]


def _spheres(session):
    """Four spheres of radius 0.6 in a box of 4, as a morphology of the workspace."""
    from morphogen.models.base import model_from_config
    from morphogen.seeding import derive_seed

    model = model_from_config({"type": "hardcore_spheres", "radius": 0.6, "number": 4})
    micro = model.generate(Domain(size=4.0, periodic=False), derive_seed(5, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "spheres", micro=micro, seed=5, model=model.to_config(),
                domain={"size": 4.0, "periodic": False},
                phases={"0": "matrix", "1": "spheres"})  # fmt: skip
    return key


def test_catalog_and_forms():
    catalog = tools.catalog()
    tab = next(t for t in catalog["tabs"] if t["name"] == "contact")
    assert tab["order"] == 52 and tab["title"] == "Contact" and tab["section"] == "simulate"
    contact = [t for t in catalog["tools"] if t["tab"] == "contact"]
    assert [t["name"] for t in contact] == ["platen_compression", "indentation_test"]
    assert [t["menu"] for t in contact] == ["Tests", "Tests"]
    for t in contact:  # the conditions of a field refer to the fields before it
        names = []
        for f in t["fields"]:
            assert set(f.get("when") or {}) <= set(names), (t["name"], f["name"])
            names.append(f["name"])
        assert t["fields"][0]["name"] == "source" and t["fields"][0]["required"]
    fields = {f["name"]: f for f in contact[1]["fields"]}
    assert fields["radius"]["when"] == {"indenter": ["sphere"]}
    assert fields["half_angle"]["default"] == 70.3 and fields["unload"]["default"] is True


@mmg
def test_compression_of_a_porous_morphology_between_platens(tmp_path):
    from morphogen.contact import compression
    from morphogen.gui.tools.contact import _sample

    session = api.Session(tmp_path)
    key = _spheres(session)
    params = {
        "source": key,
        "materials": {"0": SOLID, "1": "void"},
        "size": 0.5,
        "strain": 0.02,
        "steps": 2,
        "kinematics": "small",
    }
    out = _run(session, "platen_compression", params)
    values = out["values"]
    assert values["height of the sample"] == 4.0 and values["section of its box"] == 16.0
    assert 0.5 * 1000.0 < values["apparent modulus (first step)"] < 1000.0  # 11 % of pores
    assert values["nominal stress at the end"] < 0
    assert values["area of the sample in contact with itself"] == 0
    assert values["area in contact with the moving platen"] == pytest.approx(16.0, rel=0.05)
    rows = _table(out, "curve of the compression")
    assert len(rows) == 3 and rows[-1]["strain"] == pytest.approx(-0.02)
    assert rows[-1]["stress"] == pytest.approx(values["nominal stress at the end"], rel=1e-4)
    assert [f["title"] for f in out["figures"]] == [
        "stress and areas in contact against the compression"
    ]
    folder = tmp_path / out["key"] if "key" in out else None
    written = list(tmp_path.rglob("compressed.vtu"))
    assert len(written) == 1 and b"contact_pressure" in written[0].read_bytes()
    assert folder is None or folder.exists()
    # the test run directly on the mesh of the tool
    parsed = tools.TOOLS["platen_compression"].parse(params, session)

    class Quiet:
        def progress(self, *_):
            pass

    mesh, materials = _sample(parsed, Quiet())
    direct = compression(mesh, materials, -0.02, steps=2, kinematics="small")
    assert values["nominal stress at the end"] == pytest.approx(direct.stress[-1], rel=1e-9)
    assert values["nodes"] == mesh.n_nodes and set(np.unique(mesh.phase)) == {0}
    # with friction on the platens: stiffer
    rubbed = _run(session, "platen_compression", {**params, "friction": 0.4})["values"]
    assert rubbed["apparent modulus (first step)"] > values["apparent modulus (first step)"]
    assert rubbed["energy dissipated by friction"] > 0
    # the contact points at the nodes of the faces: the same sample between the same platens
    once = {**params, "points": "nodes", "steps": 1}  # elastic, without friction: one step
    nodes = _run(session, "platen_compression", once)["values"]
    assert nodes["nominal stress at the end"] == pytest.approx(
        values["nominal stress at the end"], rel=0.02
    )
    assert nodes["area in contact with the moving platen"] == pytest.approx(16.0, rel=0.05)
    assert "between 0 and 0.9" in _error(session, "platen_compression", {**params, "strain": 0})
    assert "every phase is a void" in _error(
        session, "platen_compression", {**params, "materials": {"0": "void", "1": "void"}}
    )
    assert "element size" in _error(session, "platen_compression", {**params, "size": 3.0})
    assert "friction" in _error(session, "platen_compression", {**params, "friction": -1})


@mmg
def test_indentation_of_a_morphology(tmp_path):
    from morphogen.contact import indentation
    from morphogen.gui.tools.contact import _sample

    session = api.Session(tmp_path)
    key = _spheres(session)
    params = {
        "source": key,
        "materials": {"0": SOLID, "1": SOLID},  # alike: a homogeneous block
        "size": 0.5,
        "radius": 10.0,
        "depth": 0.04,
        "steps": 2,
        "poisson": 0.3,
        "kinematics": "small",
    }
    out = _run(session, "indentation_test", params)
    values = out["values"]
    exact = effective_modulus(1000.0, 0.3)
    assert values["depth"] == pytest.approx(0.04) and values["radius of the sphere"] == 10.0
    assert values["force"] > 0 and values["largest pressure"] > values["mean pressure (hardness)"]
    half_space = values["reduced modulus read (a half-space)"]
    layer = values["reduced modulus read (with the base of the sample)"]
    # 4-node elements of the size of the contact: the modulus of the solid, coarsely
    assert layer < half_space and 0.7 * exact < layer < 1.6 * exact
    assert 0.5 * exact < values["reduced modulus of the unloading"] < 2.0 * exact
    rows = _table(out, "curve of the indentation")
    assert len(rows) == 5 and rows[2]["depth"] == pytest.approx(0.04) and rows[-1]["force"] == 0
    assert len(list(tmp_path.rglob("indented.vtu"))) == 1
    # the test run directly on the mesh of the tool
    parsed = tools.TOOLS["indentation_test"].parse(params, session)

    class Quiet:
        def progress(self, *_):
            pass

    mesh, materials = _sample(parsed, Quiet())
    direct = indentation(mesh, materials, 0.04, radius=10.0, steps=2, unload=True,
                         kinematics="small")  # fmt: skip
    assert values["force"] == pytest.approx(direct.force[2], rel=1e-9)
    assert layer == pytest.approx(direct.layer_modulus(0.3)[2], rel=1e-9)
    assert set(np.unique(mesh.phase)) == {0, 1}  # both phases meshed: no void
    cone = _run(session, "indentation_test", {
        "source": key, "materials": {"0": SOLID, "1": SOLID}, "size": 0.5, "indenter": "cone",
        "depth": 0.2, "steps": 2, "unload": False, "base": "smooth", "kinematics": "small",
    })["values"]  # fmt: skip
    assert "reduced modulus of the unloading" not in cone and "radius of the sphere" not in cone
    assert 0.5 * exact < cone["reduced modulus read (a half-space)"] < 2.5 * exact
    assert "two coordinates" in _error(session, "indentation_test", {**params, "point": [1.0]})
    assert "Poisson" in _error(session, "indentation_test", {**params, "poisson": 0.7})
    assert "steps" in _error(session, "indentation_test", {**params, "steps": 0})
