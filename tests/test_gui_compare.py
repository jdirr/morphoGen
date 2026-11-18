"""The tab Comparisons of the interface: the curves of several runs of the workspace (the
loading path of a morphology, material points) on one figure, with the points of a test."""

import pytest

from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.tools import Field, ToolJob
from morphogen.models.base import model_from_config
from morphogen.seeding import derive_seed

E, NU, R0 = 1.0, 0.3, 0.002
MISES = {"behaviour": "mises", "yield_stress": R0, "hardening": 0.05, "isotropic": "0, 1"}
TENSION = {"loading": "tension", "amplitude": 0.01, "increments": 8}


def _run(session, name, params):
    job = ToolJob({"tool": name, "params": params}, session)
    job._run()  # synchronously
    status = job.status()
    assert status["state"] == "done", status.get("error")
    return status["result"]


def test_curves_of_runs(tmp_path):
    session = api.Session(tmp_path)
    model = model_from_config({"type": "boolean_spheres", "radius": 0.6, "volume_fraction": 0.2})
    micro = model.generate(Domain(size=4.0, periodic=True), derive_seed(7, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "spheres", micro=micro, model=model.to_config(),
                phases={"0": "matrix", "1": "spheres"})  # fmt: skip
    point = _run(session, "point", {"young": E, "poisson": NU, **TENSION, **MISES})
    harder = _run(session, "point", {"young": E, "poisson": NU, **TENSION, **MISES,
                                     "yield_stress": 2 * R0})  # fmt: skip
    path = _run(session, "path", {"source": key, "method": "fft", "voxels": 8, "video": False,
                                  "phase": 0, **TENSION, **MISES})  # fmt: skip
    runs = [point["key"], harder["key"], path["key"]]
    traits = {i["key"]: i["traits"] for i in session.workspace()}
    assert all(traits[k] == ["curves"] for k in runs) and traits[key] == ["model"]
    out = _run(
        session, "compare_curves", {"runs": runs, "reference": "0.005, 0.0021\n0.01, 0.0023"}
    )
    assert out["title"] == "Curves of 3 runs" and [f["title"] for f in out["figures"]] == [
        "curves of the runs"
    ]
    (table,) = out["tables"]
    rows = [dict(zip(table["columns"], r, strict=True)) for r in table["rows"]]
    assert [(r["x"], r["y"]) for r in rows] == [("E11", "S11")] * 3
    assert rows[1]["largest y"] > rows[0]["largest y"]  # the harder point
    assert rows[0]["x at the end"] == pytest.approx(0.01)
    assert all(r["relative gap to the test"] >= 0 for r in rows)
    number = session.item(point["key"], ("behaviour",))["number"]
    assert out["values"][f"#{number} S11 at the end"] == pytest.approx(
        rows[0]["y at the end"], rel=1e-5
    )
    kept = session.item(out["key"], ("result",))
    assert kept["title"].startswith("#") and "(#" in kept["title"]  # the runs it took
    # a macroscopic law identified on the response of the morphology, under its own path
    law = _run(session, "identify", {"reference": "runs", "runs": [path["key"]], "young": E,
                                     "poisson": NU, "yield_stress": R0, "hardening": 0.05,
                                     "isotropic": "0, 1", "fit": ["young", "yield"]})  # fmt: skip
    assert law["values"]["gap"] < 0.5 * law["values"]["starting gap"]
    assert law["values"]["young"] > E  # the spheres stiffen the matrix
    assert law["items"][0]["kind"] == "behaviour"
    # the plastic strain along the time, and the runs a form of several items refuses
    out = _run(session, "compare_curves", {"runs": ",".join(runs[:2]), "x": "time", "y": "p"})
    assert {r[1] for r in out["tables"][0]["rows"]} == {"time"}
    linear, _ = session.folder()
    session.add(linear, "simulation", "shear modulus, FFT", source=key)
    with pytest.raises(ValueError, match="is not the run of a loading path"):
        tools.TOOLS["compare_curves"].parse({"runs": [path["key"], linear]}, session)
    with pytest.raises(ValueError, match="no simulation or behaviour or result"):
        tools.TOOLS["compare_curves"].parse({"runs": [key]}, session)
    with pytest.raises(ValueError, match="required"):
        tools.TOOLS["compare_curves"].parse({"runs": []}, session)
    assert Field("x", "items").parse(None, session) is None


def test_comparisons_in_the_catalog():
    catalog = tools.catalog()
    tab = next(t for t in catalog["tabs"] if t["name"] == "compare")
    assert tab["section"] == "share" and tab["title"] == "Comparisons"
    curves = next(t for t in catalog["tools"] if t["name"] == "compare_curves")
    runs = curves["fields"][0]
    assert runs["kind"] == "items" and runs["needs"] == ["curves"]
