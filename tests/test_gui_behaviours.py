"""The behaviours of the workspace of the interface (B): kept by a material point or an
import of Z-set, shown again, saved and opened with the workspace, and taken by the
loading paths of morphologies, the NTFA, cracks and the inputs of other solvers."""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.tools import ToolJob, fracture, interop
from morphogen.gui.tools.mechanics import ZMAT_EXAMPLE
from morphogen.homogenization.materials import Material
from morphogen.mesh.conforming import has_mmg
from morphogen.models.base import model_from_config
from morphogen.seeding import derive_seed

E, NU, R0 = 1.0, 0.3, 0.002
MISES = {"behaviour": "mises", "yield_stress": R0, "hardening": 0.05, "isotropic": "0, 1"}


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


def _run(session, name, params):
    job = ToolJob({"tool": name, "params": params}, session)
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


def _point(session, **params):
    """The behaviour a material point keeps in the workspace, and its run."""
    tension = {"young": E, "poisson": NU, "loading": "tension", "amplitude": 0.01,
               "increments": 10}  # fmt: skip
    result = _run(session, "point", {**tension, **params})
    return session.item(result["key"], ("behaviour",)), result


def test_a_material_point_keeps_its_behaviour(tmp_path):
    session, key = _workspace(tmp_path)
    kept, result = _point(session, **MISES)
    assert kept["title"] == "#2 von Mises plasticity (material point, tension)"
    assert [i["key"] for i in result["items"]] == [kept["key"]]  # shown below the result
    assert kept["info"] == {"behaviour": "von Mises plasticity", "E": E, "ν": NU, "R0": R0}
    material = Material.from_config(kept["material"])
    assert material.plasticity.yield_stress == R0 and material.young == E
    public = next(i for i in session.workspace() if i["key"] == kept["key"])
    assert public["kind"] == "behaviour" and public["tool"] == "point"
    shown = api.tool_result(kept["key"], session)  # its run, shown again
    assert shown["params"]["behaviour"] == "mises" and shown["result"]["values"] == result["values"]
    # run again from the workspace: the same curves, and no second behaviour
    again = _run(session, "point", {"loading": "tension", "amplitude": 0.01, "increments": 10,
                                    "behaviour": "workspace", "kept": kept["key"]})  # fmt: skip
    assert again["values"]["S11"] == pytest.approx(result["values"]["S11"], rel=1e-12)
    kinds = [i["kind"] for i in session.workspace()]
    assert kinds.count("behaviour") == 1 and kinds.count("result") == 1
    assert "required" in _error(session, "point", {"behaviour": "workspace"})
    with pytest.raises(ValueError, match="no behaviour"):
        tools.TOOLS["point"].parse({"behaviour": "workspace", "kept": key}, session)


def test_a_loading_path_takes_a_behaviour_of_the_workspace(tmp_path):
    """The path of a morphology whose matrix has the behaviour of a material point is the
    path of the same behaviour given in its form."""
    session, key = _workspace(tmp_path)
    kept, _ = _point(session, **MISES)
    common = {"source": key, "method": "fft", "voxels": 8, "integration": "full",
              "increments": 4, "amplitude": 0.01, "video": False, "phase": 0,
              "materials": {"0": {"young": E, "poisson": NU},
                            "1": {"young": 10 * E, "poisson": NU}}}  # fmt: skip
    given = _run(session, "path", {**common, **MISES})["values"]
    taken = _run(session, "path", {**common, "behaviour": "workspace", "kept": kept["key"]})
    assert taken["values"]["S11"] == pytest.approx(given["S11"], rel=1e-10)
    assert taken["values"]["p"] == pytest.approx(given["p"], rel=1e-10) and given["p"] > 0
    # the NTFA reduces von Mises phases only
    crystal, _ = _point(session, behaviour="crystal")
    assert crystal["info"]["behaviour"] == "crystal plasticity"
    message = _error(session, "ntfa", {"source": key, "behaviour": "workspace",
                                       "kept": crystal["key"], "voxels": 8})  # fmt: skip
    assert "reduces isotropic von Mises phases" in message


def test_a_zset_file_kept_saved_and_opened(tmp_path):
    """An import of Z-set keeps its behaviour; saved and opened with the workspace, it
    still runs a material point."""
    session, _ = _workspace(tmp_path / "first")
    result = _run(session, "zmat_import", {"text": ZMAT_EXAMPLE})
    kept = session.item(result["key"], ("behaviour",))
    assert kept["title"] == "#2 von Mises plasticity (Z-set gen_evp 0)"
    assert kept["material"]["plasticity"]["yield"] == pytest.approx(0.002)
    saved = api.save_workspace({"directory": str(tmp_path / "saved")}, session)
    second = api.Session(tmp_path / "second")
    _, opened = api.open_workspace({"directory": saved["directory"]}, second)["items"]
    assert opened["kind"] == "behaviour" and opened["tool"] == "zmat_import"
    out = _run(second, "point", {"loading": "tension", "amplitude": 0.01, "increments": 10,
                                 "behaviour": "workspace", "kept": opened["key"]})  # fmt: skip
    assert out["values"]["p"] > 0


def test_files_of_a_behaviour_in_its_folder(tmp_path):
    """An MFront source is copied into the folder of the behaviour and named by its name;
    taken, it is given by its path in the session (a library stays where it is)."""
    source = tmp_path / "law.mfront"
    source.write_text("@DSL Default;\n", encoding="utf-8")
    folder = tmp_path / "session" / "abc"
    folder.mkdir(parents=True)
    config = api.behaviour_config({"mfront": {"source": str(source), "behaviour": "Law"}}, folder)
    assert config["mfront"]["source"] == "law.mfront" and (folder / "law.mfront").is_file()
    item = {"key": "abc", "material": config}
    taken = api.behaviour_material(item, tmp_path / "session")
    assert Path(taken["mfront"]["source"]) == folder / "law.mfront"
    assert item["material"]["mfront"]["source"] == "law.mfront"  # the item left as it was
    library = {"mfront": {"library": str(tmp_path / "libBehaviour.so"), "behaviour": "Law"}}
    assert api.behaviour_config(library, folder) == library


def test_the_inputs_of_other_solvers_take_a_behaviour(tmp_path):
    """AMITEX_FFTP gets the MFront law of a behaviour of the workspace for one phase; the
    MFront sources write that of a behaviour of the workspace."""
    session, key = _workspace(tmp_path)
    kept, _ = _point(session, **MISES)
    fields = {f["name"]: f for f in tools.TOOLS["amitex_inputs"].to_dict()["fields"]}
    assert fields["kept"]["none"] == "none: the materials above"
    params = {"source": key, "resolution": 8, "loading": "path", "components": "E11=0.01",
              "increments": [4], "kept": kept["key"], "kept_phase": 1}  # fmt: skip
    result = _run(session, "amitex_inputs", params)
    folder = session.file(result["key"], "README.md").parent
    root = ET.parse(folder / "boolean_spheres_material.xml").getroot()
    laws = [m.get("Law") for m in root.findall("Material")]
    assert laws[0] == "elasiso" and laws[1].startswith("umat")
    assert "its phase" in _error(session, "amitex_inputs", {**params, "kept_phase": 4})
    assert interop._specs({"kept": None}, [0, 1]) == {
        0: {"young": 1.0, "poisson": 0.3},
        1: {"young": 10.0, "poisson": 0.3},
    }
    mfront = _run(session, "mfront_sources", {"behaviour": "workspace", "kept": kept["key"]})
    assert mfront["values"]["behaviour"] == "MorphogenBehaviour"
    source = session.file(mfront["key"], "MorphogenBehaviour.mfront").read_text(encoding="utf-8")
    assert "StandardElastoViscoPlasticity" in source


def test_cracks_take_a_behaviour(tmp_path):
    """The bulk of a plate, or one phase of a morphology, of a behaviour of the workspace."""
    session, key = _workspace(tmp_path)
    kept, _ = _point(session, **MISES)
    parsed = tools.TOOLS["crack_growth"].parse(
        {"behaviour": "workspace", "kept": kept["key"]}, session
    )
    bulk = fracture._material(parsed)
    assert bulk.plasticity.yield_stress == R0 and bulk.young == E
    form = {"bulk": "morphology", "source": key, "phase_kept": kept["key"], "kept_phase": 0}
    parsed = tools.TOOLS["crack_growth"].parse(form, session)
    assert parsed["phase_kept"]["material"]["plasticity"]["yield"] == R0
    none = tools.TOOLS["crack_growth"].parse({**form, "phase_kept": ""}, session)
    assert none["phase_kept"] is None
    fields = {f["name"]: f for f in tools.TOOLS["crack_growth"].to_dict()["fields"]}
    assert fields["phase_kept"]["none"] and "none" not in fields["kept"]
    json.dumps(tools.catalog())  # the catalog of the page, with its new fields


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG meshes the cracks")
def test_a_crack_through_a_plastic_matrix(tmp_path):
    """A crack grows through hard-core spheres whose matrix has the plastic behaviour of a
    material point: J at the tip reaches the toughness at each advance, under a rising load."""
    session = api.Session(tmp_path)
    model = model_from_config({"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.2})
    size = (12.0, 12.0, 1.0)
    micro = model.generate(Domain(size=size, periodic=False), derive_seed(3, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "hard-core spheres", micro=micro, model=model.to_config(),
                phases={"0": "matrix", "1": "spheres"})  # fmt: skip
    kept, _ = _point(session, young=1000, behaviour="mises", yield_stress=1.0, hardening=50.0,
                     isotropic="0, 1")  # fmt: skip
    stiff = {"0": {"young": 1000, "poisson": 0.3}, "1": {"young": 10000, "poisson": 0.3}}
    params = {"bulk": "morphology", "source": key, "materials": stiff, "growth": "toughness",
              "steps": 1, "element_size": 0.8, "phase_kept": kept["key"], "kept_phase": 0,
              "control": "grips", "load": 0.01, "toughness": 0.05, "load_step": 0.5}  # fmt: skip
    result = _run(session, "crack_growth", params)
    (rows,) = [t for t in result["tables"] if t["title"].startswith("the advances")]
    advances = [dict(zip(rows["columns"], r, strict=True)) for r in rows["rows"]]
    assert len(advances) == 2 and advances[1]["load"] > advances[0]["load"]
    assert all(a["J_tip"] >= 0.05 * (1 - 1e-9) for a in advances)
    assert advances[0]["J_work"] > advances[0]["J_far"]  # the plastic work of the matrix
    assert result["values"]["phase_at_front"] == "0: matrix"


def test_a_behaviour_identified(tmp_path):
    """A von Mises behaviour identified on the curves of a material point of the workspace
    (its yield stress and isotropic hardening recovered from values 50 % off), then on the
    points of a test along the path of the form; it joins the workspace."""
    session, _ = _workspace(tmp_path)
    true, run = _point(session, behaviour="mises", yield_stress=R0, hardening=0.05,
                       isotropic="0.001, 100")  # fmt: skip
    start = {"young": E, "poisson": NU, "yield_stress": 0.003, "hardening": 0.05,
             "isotropic": "0.0015, 60", "fit": ["yield", "Q", "b"]}  # fmt: skip
    out = _run(session, "identify", {"reference": "runs", "runs": [true["key"]], **start})
    v = out["values"]
    assert v["yield"] == pytest.approx(R0, rel=1e-4) and v["b"] == pytest.approx(100, rel=1e-3)
    assert v["Q"] == pytest.approx(0.001, rel=1e-3) and v["gap"] < 1e-6 < v["starting gap"]
    (made,) = out["items"]
    kept = session.item(made["key"], ("behaviour",))
    assert kept["title"].endswith(f"(identified on {true['title']})")
    assert Material.from_config(kept["material"]).plasticity.yield_stress == pytest.approx(R0)
    # its curves compared with the reference, and identified on again (the path of its own)
    traits = {i["key"]: i["traits"] for i in session.workspace()}
    assert traits[kept["key"]] == ["curves"]
    both = _run(session, "compare_curves", {"runs": [true["key"], kept["key"]]})
    (table,) = both["tables"]
    assert len(table["rows"]) == 2
    again = _run(session, "identify", {"reference": "runs", "runs": [kept["key"]], **start})
    assert again["values"]["yield"] == pytest.approx(R0, rel=1e-4)
    # the points of a test (strain, stress) along the tension of the form
    header, *rows = session.file(run["key"], "point_tension.csv").read_text().splitlines()
    e, s = header.split(",").index("E11"), header.split(",").index("S11")
    points = "\n".join(f"{r.split(',')[e]}, {r.split(',')[s]}" for r in rows[1:])
    test = {**start, "reference": "points", "points": points, "loading": "tension",
            "amplitude": 0.01, "increments": 10}  # fmt: skip
    assert _run(session, "identify", test)["values"]["yield"] == pytest.approx(R0, rel=1e-3)
    message = _error(session, "identify", {**test, "loading": "cycle"})
    assert "monotonic" in message


def test_a_behaviour_identified_on_several_runs(tmp_path):
    """The isotropic and kinematic hardenings of a behaviour identified on two material
    points of the workspace followed together, a tension and a cycle in shear (recovered
    from values 40 % off); the runs weighted, all in small or all in finite strains."""
    session, _ = _workspace(tmp_path)
    true = {"behaviour": "mises", "yield_stress": R0, "hardening": 0.0,
            "isotropic": "0.001, 100", "kinematic": "0.2, 200"}  # fmt: skip
    tension, _ = _point(session, **true)
    shear, _ = _point(session, **true, loading="cycle", component="12", amplitude=0.015)
    start = {"young": E, "poisson": NU, "yield_stress": 0.003, "hardening": 0.0,
             "isotropic": "0.0014, 60", "kinematic": "0.12, 280",
             "fit": ["yield", "Q", "b", "C", "D"], "evaluations": 400}  # fmt: skip
    runs = [tension["key"], shear["key"]]
    out = _run(session, "identify", {"reference": "runs", "runs": runs, **start})
    v = out["values"]
    for name, value in {"yield": R0, "Q": 0.001, "b": 100, "C": 0.2, "D": 200}.items():
        assert v[name] == pytest.approx(value, rel=1e-4), name
    assert v[f"gap #{shear['number']}"] < 1e-6 < v["starting gap"]
    assert len(out["figures"]) == 1  # one panel per reference
    tables = {t["title"]: t for t in out["tables"]}
    followed = [dict(zip(tables["the references followed"]["columns"], r, strict=True))
                for r in tables["the references followed"]["rows"]]  # fmt: skip
    assert [(r["path"], r["components"]) for r in followed] == [
        ("tension", "S11"),
        ("cycle", "S12"),
    ]
    (made,) = out["items"]
    kept = session.item(made["key"], ("behaviour",))
    assert kept["title"].endswith(
        f"(identified on the runs #{tension['number']}, #{shear['number']})"
    )
    assert Material.from_config(kept["material"]).plasticity.kinematic[0][0] == pytest.approx(0.2)
    # its curves: those under the path of its first reference
    assert api.curves_of(kept, session.root).name == "identified_01_tension.csv"
    own = {**start, **true, "yield_stress": 0.003, "fit": ["yield"]}
    again = _run(session, "identify", {"reference": "runs", "runs": [kept["key"]], **own})
    assert again["values"]["yield"] == pytest.approx(R0, rel=1e-6)
    # weights, one per run; the runs all in small or all in finite strains
    message = _error(session, "identify", {"reference": "runs", "runs": runs, **start,
                                           "weights": "1"})  # fmt: skip
    assert "2 positive values, one per run" in message
    finite, _ = _point(session, **true, kinematics="finite")
    message = _error(session, "identify", {"reference": "runs", "runs": [tension["key"],
                                           finite["key"]], **start})  # fmt: skip
    assert "all in small strains or all in finite strains" in message


def test_a_hill_behaviour_identified(tmp_path):
    """A material point of Hill's criterion yields in tension along 1 at R0 sqrt(6 / (4 h11
    + h22 + h33)) and keeps its coefficients; the normal coefficients of a Hill behaviour
    identified on tensions along the three axes, from von Mises's (all 1)."""
    session, _ = _workspace(tmp_path)
    hill = [1.3, 0.8, 1.1, 0.7, 1.25, 0.9]
    perfect = {"behaviour": "mises", "yield_stress": R0, "hardening": 0.0, "isotropic": "0, 1"}
    kept, result = _point(session, **perfect, hill=", ".join(map(str, hill)))
    assert result["values"]["S11"] == pytest.approx(R0 * math.sqrt(6 / (4 * 1.3 + 0.8 + 1.1)))
    assert kept["material"]["plasticity"]["hill"]["12"] == 0.7
    runs = [_point(session, **perfect, hill="1.3, 0.8, 1.1, 1, 1, 1", component=c)[0]["key"]
            for c in ("11", "22", "33")]  # fmt: skip
    start = {"young": E, "poisson": NU, "yield_stress": R0, "hardening": 0.0,
             "isotropic": "0, 1", "fit": ["hill11", "hill22", "hill33"]}  # fmt: skip
    out = _run(session, "identify", {"reference": "runs", "runs": runs, **start})
    for name, value in (("hill11", 1.3), ("hill22", 0.8), ("hill33", 1.1)):
        assert out["values"][name] == pytest.approx(value, rel=1e-5), name
    message = _error(session, "point", {**perfect, "hill": "1, 2"})
    assert "six coefficients" in message


def test_hill_behaviours_named(tmp_path):
    """A behaviour of Hill's criterion is named so in the workspace, with its coefficients
    (its title is that of the Z-set files written from it), those not given 1; von Mises's
    keeps its name."""
    coefficients = "h11, h22, h33, h12, h23, h31"
    name, info = api.behaviour_summary(
        {"young": E, "poisson": NU, "plasticity": {"yield": R0, "hill": {"11": 1.3, 22: 0.8}}}
    )
    assert name == info["behaviour"] == "Hill plasticity"
    assert info[coefficients] == "1.3, 0.8, 1, 1, 1, 1" and info["R0"] == R0
    six = {"young": E, "plasticity": {"yield": R0, "hill": [1, 2, 3, 0.5, 1, 1]}}
    assert api.behaviour_summary(six)[1][coefficients] == "1, 2, 3, 0.5, 1, 1"
    name, info = api.behaviour_summary({"young": E, "poisson": NU, "plasticity": {"yield": R0}})
    assert name == "von Mises plasticity" and coefficients not in info
    session = api.Session(tmp_path)
    kept, _ = _point(session, **MISES, hill="1.3, 0.8, 1.1, 0.7, 1.25, 0.9")
    assert kept["title"] == "#1 Hill plasticity (material point, tension)"
    assert kept["info"][coefficients] == "1.3, 0.8, 1.1, 0.7, 1.25, 0.9"


def _layers(session):
    """Layers of two von Mises phases (yield stresses 1 and 2) normal to 3, in the
    workspace, and the form of their macroscopic law."""
    model = model_from_config({"type": "random_laminate", "thickness": 1.0, "spacing": 1.0})
    micro = model.generate(Domain(size=4.0, periodic=True), derive_seed(3, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "layers", micro=micro, seed=3, model=model.to_config(),
                domain={"size": 4.0, "periodic": True},
                phases={"0": "soft", "1": "hard"})  # fmt: skip
    hard = {"young": 1000.0, "poisson": 0.3, "plasticity": {"yield": 2.0, "hardening": 10.0}}
    return key, {"source": key, "materials": {"0": {"young": 1000.0, "poisson": 0.3},
                                              "1": {"young": 1000.0, "poisson": 0.3}},
                 "phase": 0, "behaviour": "mises", "yield_stress": 1.0, "hardening": 10.0,
                 "isotropic": "0, 1", "behaviours": json.dumps({"1": hard}), "voxels": 4,
                 "loadings": ["11", "33", "12", "31"], "strain": 0.006,
                 "increments": 8}  # fmt: skip


def test_the_macroscopic_law_of_a_laminate(tmp_path):
    """The macroscopic law of layers of two von Mises phases (yield stresses 1 and 2)
    normal to 3, in one go: four loadings of its voxels, then Hill's law from an
    estimated one, which halves the gaps and finds the shear across the layers weaker than
    in their plane (h31 > h12); the law joins the workspace, written as a Z-mat file."""
    session = api.Session(tmp_path)
    key, params = _layers(session)
    out = _run(session, "macroscopic_law", params)
    v = out["values"]
    assert v["gap"] < 0.5 * v["starting gap"] and v["hill31"] > 1.5 * v["hill12"]
    assert "hill11" not in v and v["young"] == pytest.approx(1000.0, rel=0.05)
    (made,) = out["items"]
    kept = session.item(made["key"], ("behaviour",))
    assert kept["title"].endswith(f"(macroscopic law of #{session.item(key)['number']})")
    assert set(kept["material"]["plasticity"]["hill"]) == {"33", "12", "31"}
    names = {f["name"] for f in out["files"]}
    assert {"macroscopic_law.mat", "material.json", "macroscopic_law.png"} <= names
    assert [t["title"] for t in out["tables"]][:2] == ["the parameters", "the loadings followed"]
    message = _error(session, "macroscopic_law", {**params, "loadings": []})
    assert "at least one" in message
    # cycles: a kinematic hardening too
    cycles = {**params, "loadings": ["11"], "law": "mises", "cycle": True, "increments": 4}
    v = _run(session, "macroscopic_law", cycles)["values"]
    assert {"C", "D"} <= set(v) and "hill33" not in v


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_the_macroscopic_law_by_finite_elements_and_in_finite_strains(tmp_path):
    """The macroscopic law of the same layers from the shears in their plane and across
    them, by finite elements on their conforming mesh and by FFT in finite strains: the
    shear across the layers weaker either way (h31 above h12, which stays 1)."""
    session = api.Session(tmp_path)
    _, params = _layers(session)
    shears = {**params, "loadings": ["12", "31"], "increments": 6}
    fe = {**shears, "method": "fe", "element_size": 1.0, "order": "1"}
    for form in (fe, {**shears, "kinematics": "finite"}):
        v = _run(session, "macroscopic_law", form)["values"]
        assert v["hill31"] > 1.3 and "hill12" not in v and v["gap"] < v["starting gap"], form


def test_the_gurson_law_of_a_porous_morphology(tmp_path):
    """Gurson's law of hard-core spheres as voids in a von Mises matrix, from a tension and
    a confined one of its voxels: the matrix that of the phase, the porosity that of the
    voids of its voxels, q1 and q2 found closer to the responses than Tvergaard's."""
    session = api.Session(tmp_path)
    model = model_from_config({"type": "hardcore_spheres", "radius": 0.8, "volume_fraction": 0.1})
    micro = model.generate(Domain(size=4.0, periodic=True), derive_seed(3, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "pores", micro=micro, seed=3, model=model.to_config(),
                domain={"size": 4.0, "periodic": True},
                phases={"0": "matrix", "1": "pores"})  # fmt: skip
    params = {"source": key, "materials": {"0": {"young": 1000.0, "poisson": 0.3}, "1": "void"},
              "phase": 0, "behaviour": "mises", "yield_stress": 1.0, "hardening": 10.0,
              "isotropic": "0, 1", "voxels": 6, "loadings": ["11", "confined11"],
              "strain": 0.01, "increments": 6, "law": "gurson"}  # fmt: skip
    out = _run(session, "macroscopic_law", params)
    v = out["values"]
    assert v["gap"] < v["starting gap"] and 0.3 < v["q1"] < 5.0 and 0.2 < v["q2"] < 5.0
    (made,) = out["items"]
    material = session.item(made["key"], ("behaviour",))["material"]
    assert 0.03 < material["porous_plasticity"]["porosity"] < 0.2
    assert material["porous_plasticity"]["yield"] == 1.0  # the matrix of the phase
    solid = {**params, "materials": {"0": {"young": 1000.0, "poisson": 0.3},
                                     "1": {"young": 1000.0, "poisson": 0.3}}}  # fmt: skip
    assert "a porous morphology" in _error(session, "macroscopic_law", solid)


def test_a_porous_behaviour_identified(tmp_path):
    """Gurson's q1 and q2 identified on two material points of a porous behaviour of the
    workspace, a tension and a uniaxial strain (two triaxialities), from values 30 % off;
    the behaviour found starting another identification from the workspace."""
    session, _ = _workspace(tmp_path)
    gtn = {"behaviour": "porous", "criterion": "gurson", "porosity": 0.01, "gurson": "1.5, 1",
           "yield_stress": R0, "hardening": 0.02, "isotropic": "0, 1"}  # fmt: skip
    tension, _ = _point(session, **gtn, amplitude=0.05, increments=20)
    strain = "t=0,1 n=20 E11=0,0.05 E22=0,0 E33=0,0"  # the lateral strains held
    strained, _ = _point(session, **gtn, loading="custom", path=strain)
    runs = [tension["key"], strained["key"]]
    start = {**gtn, "young": E, "poisson": NU, "gurson": "1, 1.3", "fit": ["q1", "q2"]}
    out = _run(session, "identify", {"reference": "runs", "runs": runs, **start})
    assert out["values"]["q1"] == pytest.approx(1.5, rel=1e-4)
    assert out["values"]["q2"] == pytest.approx(1.0, rel=1e-4)
    # the lateral stresses of the uniaxial strain compared too: they tell the triaxiality
    (followed,) = (t for t in out["tables"] if t["title"] == "the references followed")
    at = followed["columns"].index("components")
    assert [r[at] for r in followed["rows"]] == ["S11", "S11, S22, S33"]
    (made,) = out["items"]
    kept = session.item(made["key"], ("behaviour",))
    assert kept["title"].startswith(f"#{kept['number']} porous plasticity (identified on")
    assert kept["material"]["porous_plasticity"]["q1"] == pytest.approx(1.5, rel=1e-4)
    # from a behaviour of the workspace: its own values, its elasticity
    again = _run(session, "identify", {"reference": "runs", "runs": [strained["key"]],
                                       "behaviour": "workspace", "kept": kept["key"],
                                       "fit": ["porosity"]})  # fmt: skip
    assert again["values"]["porosity"] == pytest.approx(0.01, rel=1e-4)
