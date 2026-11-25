"""The tools of the tab Explicit dynamics of the interface: the inputs of OpenRadioss of a
morphology written without running them, a dynamic test run with a stand-in of
OpenRadioss (fake_openradioss) and its page (the summary in engineering units, the
curves in morphoGen's components for the Comparisons tab, the frames of the 3D view), the
results of a run read back from its folder, the installation found, and the errors."""

from __future__ import annotations

import sys
import zipfile

import numpy as np
import pytest

from fake_openradioss import fake_installation
from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.tools import ToolJob
from morphogen.models.base import model_from_config

posix = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in runs by its shebang")


@pytest.fixture
def session(tmp_path, monkeypatch):
    from morphogen import dynamics

    for name in dynamics.openradioss.ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dynamics.openradioss, "default_home", lambda: tmp_path / "home")
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
    s = api.Session(tmp_path / "session")
    model = model_from_config(
        {"type": "tpms", "surface": "gyroid", "volume_fraction": 0.3, "cell_size": 5.0}
    )
    domain = Domain(size=10.0)
    key, _ = s.folder()
    s.add(
        key,
        "morphology",
        "gyroid",
        micro=model.generate(domain, 1),
        seed=1,
        model=model.to_config(),
        domain=domain.to_dict(),
        phases={"0": "void", "1": "solid"},
    )
    return s, key


def _run(session, name, params):
    job = ToolJob({"tool": name, "params": params}, session)
    job._run()
    status = job.status()
    assert status["state"] == "done", status.get("error")
    return status["result"], job.folder


def _error(session, name, params) -> str:
    job = ToolJob({"tool": name, "params": params}, session)
    job._run()
    status = job.status()
    assert status["state"] == "error"
    return status["error"]


def test_catalog():
    catalog = tools.catalog()
    tab = next(t for t in catalog["tabs"] if t["name"] == "dynamics")
    assert tab["section"] == "simulate" and tab["title"] == "Explicit dynamics"
    names = [t["name"] for t in catalog["tools"] if t["tab"] == "dynamics"]
    assert names == ["radioss_test", "radioss_results", "radioss_install"]
    test = next(t for t in catalog["tools"] if t["name"] == "radioss_test")
    fields = {f["name"]: f for f in test["fields"]}
    assert fields["installation"]["browse"] == "folder"
    assert fields["materials"]["default"] == {"young": 70000, "poisson": 0.33}
    assert [g for g in dict.fromkeys(f.get("group") for f in test["fields"])] == [
        "Specimen",
        "Materials",
        "Loading",
        "Solver and outputs",
    ]


def test_inputs_without_running(session, tmp_path):
    s, key = session
    params = {
        "source": key,
        "resolution": 10,
        "run": False,
        "yield_stress": 250,
        "hardening": 800,
        "failure_strain": 0.35,
        "length_unit": "um",
        "behaviours": '{"1": {"plasticity": {"yield": 250, "norton": {"K": 30, "n": 5}}}}',
    }
    result, folder = _run(s, "radioss_test", params)
    assert (
        result["values"]["elements"].endswith("brick") and result["values"]["cycles (estimate)"] > 0
    )
    names = {f["name"] for f in result["files"]}
    assert names == {"README.md", "tpms_openradioss_inputs.zip"}
    with zipfile.ZipFile(folder / "tpms_openradioss_inputs.zip") as zf:
        assert {
            "tpms_0000.rad",
            "tpms_0001.rad",
            "run_openradioss.sh",
            "run_openradioss.bat",
            "morphogen_radioss.json",
            "tpms_mesh.npz",
            "README.md",
        } <= set(zf.namelist())
    deck = (folder / "openradioss" / "tpms_0000.rad").read_text(encoding="ascii")
    assert "/MAT/LAW36/1" in deck and "/INTER/TYPE24/1" in deck
    import json

    manifest = json.loads((folder / "openradioss" / "morphogen_radioss.json").read_text())
    assert manifest["lengths"] == pytest.approx([1e-2] * 3)  # 10 µm in mm
    material = manifest["materials"]["1"]
    assert material["density"] == pytest.approx(2.7e-9) and material["failure_strain"] == 0.35
    assert material["plasticity"]["norton"] == {"K": 30.0, "n": 5.0}
    assert "morphogen radioss run" in result["text"]


def test_errors(session):
    s, key = session
    assert "OpenRadioss not found" in _error(s, "radioss_test", {"source": key})
    assert "uncheck" in _error(s, "radioss_test", {"source": key})
    error = _error(
        s,
        "radioss_test",
        {
            "source": key,
            "run": False,
            "behaviours": '{"1": {"plasticity": {"yield": 1, "kinematic": [{"C": 1, "D": 1}]}}}',
        },
    )
    assert "kinematic hardening" in error
    assert "OpenRadioss not found" in _error(s, "radioss_install", {"action": "check"})


@posix
def test_dynamic_test_and_its_page(session, tmp_path):
    s, key = session
    root = fake_installation(tmp_path / "OpenRadioss")
    params = {
        "source": key,
        "resolution": 10,
        "frames": 5,
        "strain": 0.25,
        "strain_rate": 1000,
        "yield_stress": 200,
        "hardening": 500,
        "installation": str(root),
        "threads": 1,
    }
    result, folder = _run(s, "radioss_test", params)
    values = result["values"]
    phi = values["solid fraction"]
    assert values["final stress (MPa)"] == pytest.approx(70000 * phi * 0.25, rel=1e-6)
    assert values["forces from"] == "time history" and values["frames"] == 6
    assert values["absorbed energy per volume (MJ/m³)"] == pytest.approx(
        0.5 * 70000 * phi * 0.25**2, rel=0.03
    )
    assert values["end time (ms)"] == pytest.approx(0.25 / (1000 * 0.975) * 1e3)
    names = {f["name"] for f in result["files"]}
    assert {
        "curves.csv",
        "energies.csv",
        "final_state.vtu",
        "stress_strain.png",
        "README.md",
        "tpms_openradioss_inputs.zip",
    } <= names
    assert result["archive"].endswith("tpms_openradioss_inputs.zip") and result["view3d"]
    assert len(list((folder / "frames").glob("frame_*.vtu"))) == 6
    curves = next(t for t in result["tables"] if t["url"].endswith("curves.csv"))
    assert "E33" in curves["columns"] and "S33" in curves["columns"] and curves["count"] == 1001
    from morphogen.pipeline import read_rows

    rows = read_rows(folder / "curves.csv")
    assert float(rows[-1]["E33"]) == pytest.approx(-0.25) and float(rows[-1]["S33"]) < 0
    # the 3D view of the result, and its curves compared with another run
    shown = api.surface({"key": result["key"]}, s)
    assert shown["field"] is not None and "von_mises" in shown["field"]["names"]
    # its frames are the steps of the 3D view; no video unless asked, the export makes one
    assert result["videos"] == [] and len(shown["steps"]) == 6 and shown["step"] == 5
    first = api.surface({"key": result["key"], "step": 0}, s)
    assert first["step"] == 0 and first["steps"][0]["label"].startswith("t = ")
    item = s.item(result["key"], ("result",))
    assert "curves" in api.item_traits(item)
    again, _ = _run(s, "radioss_test", {**params, "strain_rate": 500})
    compared, _ = _run(s, "compare_curves", {"runs": [result["key"], again["key"]]})
    table = compared["tables"][0]
    runs = [dict(zip(table["columns"], row, strict=True)) for row in table["rows"]]
    assert (
        len(runs) == 2 and {r["x"] for r in runs} == {"E33"} and {r["y"] for r in runs} == {"S33"}
    )
    # the run read back from its folder
    read, _ = _run(
        s, "radioss_results", {"directory": str(folder / "openradioss"), "installation": str(root)}
    )
    assert read["values"]["final stress (MPa)"] == pytest.approx(values["final stress (MPa)"])
    checked, _ = _run(s, "radioss_install", {"action": "check", "installation": str(root)})
    assert checked["values"]["engine"].endswith("engine_linux64_gf")


@posix
def test_impact_page(session, tmp_path):
    s, key = session
    root = fake_installation(tmp_path / "OpenRadioss")
    result, _ = _run(
        s,
        "radioss_test",
        {
            "source": key,
            "resolution": 10,
            "frames": 3,
            "loading": "impact",
            "velocity": 15,
            "installation": str(root),
            "threads": 1,
        },
    )
    values = result["values"]
    mass = values["mass (g)"] * 1e-3
    assert values["initial kinetic energy (J)"] == pytest.approx(0.5 * mass * 15**2, rel=1e-6)
    assert values["forces from"] == "momentum"
    assert any(f["name"] == "wall_force.png" for f in result["files"])
    assert np.isfinite(values["peak force on the wall (N)"])
