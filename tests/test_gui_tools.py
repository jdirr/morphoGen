"""The tools of the interface: their catalog, the forms they parse, their jobs, and the
tools of the tab Statistics (RVE sizes, measures, correlation functions)."""

import numpy as np
import pytest

from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.tools import Field, ToolJob
from morphogen.mesh.conforming import has_mmg
from morphogen.models.base import model_from_config
from morphogen.seeding import derive_seed


def _workspace(tmp_path):
    """A session whose workspace holds a Boolean model of spheres (without the views)."""
    session = api.Session(tmp_path)
    model = model_from_config({"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.3})
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


def test_catalog_and_forms(tmp_path):
    catalog = tools.catalog()
    assert [t["name"] for t in catalog["tabs"]][:1] == ["statistics"]
    names = {t["name"] for t in catalog["tools"]}
    assert {"rve", "measures", "curves"} <= names
    rve = next(t for t in catalog["tools"] if t["name"] == "rve")
    fields = {f["name"]: f for f in rve["fields"]}
    assert fields["source"]["kind"] == "item" and fields["source"]["required"]
    assert fields["sizes"]["when"] == {"method": ["ensemble"]}
    assert fields["materials"]["of"] == "source"
    # the values of a form: converted, checked, the hidden ones left out
    session, key = _workspace(tmp_path)
    parsed = tools.TOOLS["rve"].parse(
        {"source": key, "method": "windows", "sizes": "1, 2", "window_sizes": "0.5 1"}, session
    )
    assert parsed["source"]["key"] == key and parsed["sizes"] is None
    assert parsed["window_sizes"] == [0.5, 1.0] and parsed["relative_errors"] == [0.01, 0.05]
    with pytest.raises(ValueError, match="required"):
        tools.TOOLS["rve"].parse({}, session)
    with pytest.raises(ValueError, match="one of"):
        tools.TOOLS["rve"].parse({"source": key, "method": "both"}, session)
    with pytest.raises(ValueError, match="workspace"):
        tools.TOOLS["rve"].parse({"source": "nothing"}, session)
    with pytest.raises(ValueError, match="expected int"):
        Field("n", "int").parse("two", session)
    assert Field("m", "materials").parse(
        {"0": "void", "1": {"young": 2, "poisson": 0.2}}, None
    ) == {
        0: "void",
        1: {"young": 2.0, "poisson": 0.2},
    }
    with pytest.raises(ValueError, match="kind"):
        Field("x", "colour")
    with pytest.raises(ValueError, match="tool"):
        ToolJob({"tool": "nothing"}, session)


def test_items_with_the_traits_the_tools_need(tmp_path):
    """The forms take the items that have the traits their fields need: an RVE study the
    realization of a model, a polycrystal crack a tessellation; the page is told them."""
    session, key = _workspace(tmp_path)
    grains = model_from_config({"type": "voronoi", "n_cells": 8})
    micro = grains.generate(Domain(size=1.0, periodic=True), derive_seed(1, 0))
    poly, _ = session.folder()
    session.add(poly, "morphology", "voronoi (seed 1)", micro=micro, model=grains.to_config())
    design, _ = session.folder()
    session.add(design, "morphology", "design", micro=session.item(key)["micro"], model=None)
    traits = {i["title"]: i["traits"] for i in session.workspace()}
    assert traits == {
        "#1 boolean_spheres (seed 7)": ["model"],
        "#2 voronoi (seed 1)": ["model", "grains"],
        "#3 design": [],
    }
    catalog = {t["name"]: t for t in tools.catalog()["tools"]}
    assert catalog["rve"]["fields"][0]["needs"] == ["model"]
    assert catalog["polycrystal_crack"]["fields"][0]["needs"] == ["grains"]
    with pytest.raises(ValueError, match="#3 design is not the realization of a model"):
        tools.TOOLS["rve"].parse({"source": design}, session)
    with pytest.raises(ValueError, match="is not a polycrystal"):
        tools.TOOLS["polycrystal_crack"].parse({"source": key}, session)
    assert (
        tools.TOOLS["polycrystal_crack"].parse({"source": poly}, session)["source"]["key"] == poly
    )


def test_runs_of_tools_kept_in_the_workspace(tmp_path):
    """Each run of a tool joins the workspace as a result, titled by the items it took,
    with the parameters of its form and its result to show them again; a saved workspace
    opens with them, their links and the items they took renamed."""
    session, key = _workspace(tmp_path / "first")
    params = {"source": key, "resolution": "40", "measures": ["phase_fractions"]}
    measured = _run(session, "measures", params)
    (run,) = [i for i in session.workspace() if i["kind"] == "result"]
    assert run["title"] == "#2 Morphological measures (#1)" and run["tab"] == "statistics"
    kept = api.tool_result(run["key"], session)
    assert kept["tool"] == "measures" and kept["params"] == params
    assert kept["result"]["values"] == measured["values"]
    with pytest.raises(ValueError, match="no result"):
        api.tool_result(key, session)
    saved = api.save_workspace({"directory": str(tmp_path / "saved")}, session)
    second = api.Session(tmp_path / "second")
    morph, again = api.open_workspace({"directory": saved["directory"]}, second)["items"]
    reopened = api.tool_result(again["key"], second)
    assert reopened["params"]["source"] == morph["key"] != key
    assert reopened["result"]["values"] == measured["values"]
    urls = [f["url"] for f in reopened["result"]["files"]]
    assert urls and all(u.startswith(f"/files/{again['key']}/") for u in urls)


def test_numbers_never_given_twice(tmp_path):
    """An item removed leaves its number: the next item takes a new one."""
    session, key = _workspace(tmp_path)
    second, _ = session.folder()
    session.add(second, "morphology", "second", micro=None)
    session.remove(key)
    third, _ = session.folder()
    assert session.add(third, "morphology", "third", micro=None)["title"] == "#3 third"
    # renamed, an item keeps its number
    assert session.rename(third, "  the  third one ")["title"] == "#3 the third one"
    with pytest.raises(ValueError, match="some text"):
        session.rename(third, " ")
    with pytest.raises(ValueError, match="no item"):
        session.rename(key, "removed")


def test_closed_items_brought_back(tmp_path):
    """The cross of a chip takes an item out of the workspace, and its undo brings it back
    with its number, title and objects; only the last ones closed can come back."""
    session, key = _workspace(tmp_path)
    before = session.public(session.item(key))
    micro = session.items[key]["micro"]
    session.remove(key)
    assert session.workspace() == []
    with pytest.raises(ValueError, match="no item"):
        session.remove(key)
    restored = session.restore(key)
    assert session.public(restored) == before and restored["micro"] is micro
    with pytest.raises(ValueError, match=r"no item .* closed"):
        session.restore(key)  # once
    keys = []
    for n in range(session.CLOSED + 2):
        k, _ = session.folder()
        session.add(k, "morphology", f"item {n}", micro=None)
        session.remove(k)
        keys.append(k)
    with pytest.raises(ValueError, match="closed"):
        session.restore(keys[0])  # the earliest closes are forgotten
    assert session.restore(keys[-1])["title"].endswith(f"item {session.CLOSED + 1}")


def test_a_run_joins_the_item_its_tool_made(tmp_path):
    """A tool that makes its folder an item (the simulation of a loading path) keeps it:
    the run joins that item, which stays a simulation and shows again in its tab."""
    session, key = _workspace(tmp_path)

    def made(p, ctx):
        source = p["source"]
        ctx.session.add(
            ctx.key, "simulation", f"made from #{source['number']}", source=source["key"]
        )
        return {"values": {"x": 1.0}}

    tools.tool("demo_made", "statistics", "Made", "", [Field("source", "item", required=True)])(
        made
    )
    try:
        result = _run(session, "demo_made", {"source": key})
    finally:
        tools.TOOLS.pop("demo_made")
    item = session.item(result["key"])
    assert item["kind"] == "simulation" and item["title"] == "#2 made from #1"
    assert [i["kind"] for i in session.workspace()] == ["simulation", "morphology"]
    kept = api.tool_result(result["key"], session)
    assert kept["tool"] == "demo_made" and kept["params"] == {"source": key}
    plain, _ = session.folder()
    session.add(plain, "simulation", "plain")
    with pytest.raises(ValueError, match="not the run of a tool"):
        api.tool_result(plain, session)


@pytest.mark.skipif(not has_mmg(), reason="MMG meshes the cracks")
def test_runs_with_a_mesh_drawn_in_3d(tmp_path):
    """A run whose result has a mesh of fields (a cracked plate) is drawn by the 3D view,
    deformed by its displacement; a run without one has nothing to draw."""
    session, key = _workspace(tmp_path)
    cracked = _run(session, "crack_factors", {"length": "4.5"})
    assert cracked["view3d"] is True
    run = next(i for i in session.workspace() if i["key"] == cracked["key"])
    assert run["kind"] == "result" and run["view3d"] is True
    shown = api.surface({"key": cracked["key"], "resolution": 24}, session)
    assert shown["kind"] == "result" and shown["displacement"] and shown["field"]["names"]
    measured = _run(session, "measures", {"source": key, "resolution": "32"})
    assert measured["view3d"] is False
    with pytest.raises(ValueError, match="nothing to draw"):
        api.surface({"key": measured["key"]}, session)


def test_statistics_tools(tmp_path):
    session, key = _workspace(tmp_path)
    chosen = ["phase_fractions", "specific_surface"]
    measured = _run(session, "measures", {"source": key, "resolution": 48, "measures": chosen})
    assert measured["values"]["fraction_1"] == pytest.approx(0.3, abs=0.05)
    theory = next(t for t in measured["tables"] if t["title"] == "theory of the model")
    assert "volume_fraction" in [row[0] for row in theory["rows"]]
    # on the implicit geometry itself: no voxels, the theory beside the exact densities
    chosen = ["exact_fraction", "exact_minkowski"]
    exact = _run(session, "measures", {"source": key, "resolution": 48, "measures": chosen})
    values = exact["values"]
    assert values["exact_V_V"] == pytest.approx(values["exact_fraction"], abs=0.01)
    assert {"exact_S_V", "exact_M_V", "exact_chi_V", "exact_fraction_std"} <= set(values)
    theory = next(t for t in exact["tables"] if t["title"] == "theory of the model")
    row = next(r for r in theory["rows"] if r[0] == "specific_surface")
    assert row[-1] == pytest.approx(values["exact_S_V"], rel=1e-5)  # the table: 6 digits
    chosen = ["covariance", "chords", "capacity_ball"]
    curves = _run(session, "curves", {"source": key, "resolution": 32, "curves": chosen})
    titles = [f["title"] for f in curves["figures"]]
    assert titles == ["covariance", "chords", "capacity ball"]
    assert {"covariance.csv", "covariance.png", "covariance.svg"} <= {
        f["name"] for f in curves["files"]
    }
    covariance = next(t for t in curves["tables"] if t["title"] == "covariance")
    assert covariance["columns"][:2] == ["h", "x"] and covariance["rows"][0][1] == pytest.approx(
        measured["values"]["fraction_1"], abs=0.02
    )  # C(0) = p
    # an RVE study by the ensemble method: the realizations of the model at three sizes
    study = _run(session, "rve", {"source": key, "method": "ensemble", "sizes": [2, 3, 4],
                                  "realizations": [4, 3, 2], "voxel_size": 0.125})  # fmt: skip
    assert any("RVE sizes" in t["title"] for t in study["tables"])
    assert any(f["title"].startswith("rve volume fraction") for f in study["figures"])
    assert "rve_report.md" in {f["name"] for f in study["files"]}
    windows = _run(session, "rve", {"source": key, "method": "windows", "size": 6.0,
                                    "window_sizes": [1.5, 3.0], "voxel_size": 0.125})  # fmt: skip
    assert windows["tables"] and windows["figures"]
    # a design is not the realization of a model: refused at once, as the form is read
    design_key, _ = session.folder()
    session.add(design_key, "morphology", "design", micro=session.item(key)["micro"], model=None)
    with pytest.raises(ValueError, match="not the realization of a model"):
        ToolJob({"tool": "rve", "params": {"source": design_key}}, session)


def test_rve_tool_of_the_stiffness(tmp_path):
    """The RVE study of the stiffness: the mean C of the largest size and its compliance S =
    C⁻¹ (S C = I, Voigt order), then the tables of the report, each component per size."""
    session, key = _workspace(tmp_path)
    study = _run(session, "rve", {"source": key, "method": "ensemble", "sizes": [2, 3],
                                  "realizations": [2, 2], "voxel_size": 1.0,
                                  "property": "stiffness"})  # fmt: skip
    stiffness, compliance = study["tables"][:2]
    assert stiffness["title"].startswith("stiffness C_ij at 3³, C the mean of 2 realizations")
    assert compliance["title"].startswith("compliance S_ij = C⁻¹ at 3³, C the mean of 2")
    assert stiffness["columns"] == ["", "11", "22", "33", "23", "31", "12"]
    c = np.array([row[1:] for row in stiffness["rows"]], dtype=float)
    s = np.array([row[1:] for row in compliance["rows"]], dtype=float)
    np.testing.assert_allclose(s @ c, np.eye(6), atol=1e-4)  # rounded to 6 digits
    titles = [t["title"] for t in study["tables"]]
    assert any(t.startswith("Mean stiffness Cbar_ij +/- 2 D (PBC") for t in titles)
    assert any(t.startswith("Compliance S_ij = Cbar^-1 +/- 2 D (PBC") for t in titles)
    assert any(f["title"] == "rve stiffness periodic" for f in study["figures"])


def test_markdown_tables():
    from morphogen.gui.tools.statistics import _markdown_tables

    text = (
        "# Report\n\n**RVE sizes**\n\n| eps | n | V |\n|---|---|---|\n"
        "| 0.01 | 1 | 12.5 |\n| 0.05 | 10 | 0.3 |\n"
    )
    (t,) = _markdown_tables(text)
    assert t["title"] == "RVE sizes" and t["columns"] == ["eps", "n", "V"]
    assert t["rows"] == [["0.01", "1", "12.5"], ["0.05", "10", "0.3"]] and t["count"] == 2
    assert np.isfinite(float(t["rows"][0][2]))


def test_stereology_tool(tmp_path):
    """The sections of hard-core spheres by the planes of the grid unfold into their
    radius, and the shape consistent with the three directions is a sphere."""
    session = api.Session(tmp_path)
    model = model_from_config({"type": "hardcore_spheres", "radius": 0.5, "volume_fraction": 0.2})
    micro = model.generate(Domain(size=5.0, periodic=True), derive_seed(1, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "spheres", micro=micro, model=model.to_config(), seed=1)
    out = _run(session, "stereology", {"source": key, "resolution": 64})
    assert out["values"]["mean short semi-axis"] == pytest.approx(0.5, rel=0.1)
    assert out["values"]["consistent b/a"] == pytest.approx(1.0, abs=0.1)
    assert out["tables"][0]["count"] == 3  # the three directions


def test_table_fit_and_reconstruction(tmp_path):
    """The RVE law fitted to a pasted table whose variance falls as 1/V, and a morphology
    reconstructed from the two-point function of another one, joining the workspace."""
    session, key = _workspace(tmp_path)
    rng = np.random.default_rng(0)
    lines = ["volume;E1"]
    for v, n in ((1, 30), (8, 20), (27, 10), (64, 6)):
        lines += [f"{v};{10 + rng.normal(0, 2 / np.sqrt(v)):.5f}" for _ in range(n)]
    fit = _run(session, "fit_table", {"table": "\n".join(lines)})
    assert any("RVE sizes" in t["title"] for t in fit["tables"]) and fit["figures"]
    job = ToolJob({"tool": "fit_table", "params": {"table": "a,b\n1,2", "volume_column": "v"}},
                  session)  # fmt: skip
    job._run()
    assert "no column 'v'" in job.status()["error"]
    rebuilt = _run(session, "reconstruction", {"source": key, "resolution": 20, "sweeps": 15})
    values = rebuilt["values"]
    assert values["fraction (reconstruction)"] == pytest.approx(
        values["fraction (original)"], abs=0.02
    )
    assert values["final energy"] < 0.2 * values["initial energy"]
    (item,) = rebuilt["items"]
    assert item["kind"] == "morphology" and session.item(item["key"])["source"] == key
    # by gradient descent: the exact covariance of the Boolean model at every lag
    params = {"source": key, "resolution": 20, "method": "gradient", "iterations": 300}
    descended = _run(session, "reconstruction", params)
    values = descended["values"]
    assert values["fraction (reconstruction)"] == pytest.approx(0.3, abs=0.01)
    assert values["error of the two-point function"] < 0.05


def test_orientation_map_tool(tmp_path):
    """The tool of the orientation maps on the map that comes with morphoGen: the numbers
    of its grains and of its texture, the tables of the grains and of their boundaries,
    the figures (the map, the pole figures, the inverse pole figure, the
    disorientations); its polycrystal joins the workspace, one cell thick, with the
    orientations of its grains."""
    session = api.Session(tmp_path)
    out = _run(session, "texture", {"file": "ebsd_map.ctf", "poles": "100 1-11"})
    values = out["values"]
    assert values["points"] == "64 x 48" and values["grains"] == 56
    assert values["indexed"] == pytest.approx(0.88, abs=0.005) and values["lattice"] == "cubic"
    assert values["pole figure {100}, largest density"] == pytest.approx(5.73, abs=0.01)
    assert "pole figure {1-11}, largest density" in values
    assert values["mean disorientation of the boundaries (degrees)"] == pytest.approx(38.4, abs=0.1)
    grains, boundaries = out["tables"]
    assert grains["count"] == 56 and grains["columns"][:5] == ["grain", "phase", "phi1", "Phi",
                                                               "phi2"]  # fmt: skip
    assert boundaries["count"] == 130
    titles = [f["title"] for f in out["figures"]]
    assert len(titles) == 4 and titles[0].startswith("the map in the colours")
    assert "pole figures" in titles and "disorientations of the boundaries" in titles
    (item,) = out["items"]
    kept = session.item(item["key"])
    assert item["kind"] == "morphology" and kept["planar"] and kept["model"]["type"] == "ebsd"
    micro = kept["micro"]
    assert micro.shape == (64, 48, 1) and len(micro.orientations) == 56
    # without its polycrystal; a file that is not there; indices that are not
    again = _run(session, "texture", {"file": "ebsd_map.ctf", "add": False, "direction": "x"})
    assert again["items"] == [] and "along x" in again["figures"][0]["title"]
    wrong = [({"file": "nowhere.ctf"}, "no orientation map"),
             ({"file": "ebsd_map.ctf", "poles": "1a1"}, "indices such as")]  # fmt: skip
    for params, message in wrong:
        job = ToolJob({"tool": "texture", "params": params}, session)
        job._run()
        assert message in job.status()["error"]


def test_tabs_of_the_tools_have_their_sections():
    """Every tab of tools sits in a section of the menus: Statistics characterizes, Design
    creates, Solvers & formats shares, the others simulate."""
    tabs = {t["name"]: t["section"] for t in tools.catalog()["tabs"]}
    assert set(tabs.values()) <= set(tools.SECTIONS)
    assert tabs["statistics"] == "characterize" and tabs["design"] == "create"
    assert tabs["interop"] == "share" and tabs["fracture"] == tabs["mechanics"] == "simulate"
    with pytest.raises(ValueError, match="section"):
        tools.tab("nothing", "Nothing", "", 1, section="elsewhere")


def test_menus_of_the_tabs():
    """The menu of a tab lists its tools under the headings of what they do: every tool
    of the larger tabs has its heading, a heading holds consecutive tools in their order,
    and the catalog gives the tools as the menus list them."""
    catalog = tools.catalog()
    by_tab: dict[str, list[dict]] = {}
    for t in catalog["tools"]:
        by_tab.setdefault(t["tab"], []).append(t)
    assert [t["name"] for t in catalog["tabs"]] == list(by_tab)  # tab after tab
    for name, listed in by_tab.items():
        headings = [t["menu"] for t in listed]
        runs = [h for k, h in enumerate(headings) if k == 0 or h != headings[k - 1]]
        assert len(runs) == len(set(runs)), name  # no heading split in two
        if len(listed) >= 4:
            assert all(headings), name
            assert max(headings.count(h) for h in runs) <= 4, name
        menu = tools.menus(name)
        assert [h for h, _ in menu] == runs
        assert [t.name for _, group in menu for t in group] == [t["name"] for t in listed]
        for _, group in menu:
            assert [t.order for t in group] == sorted(t.order for t in group)

    def menu(name):
        return {h: [t.name for t in group] for h, group in tools.menus(name)}

    assert menu("mechanics") == {
        "Loading paths": ["path", "macroscopic_law", "ntfa"],
        "Behaviours of materials": ["point", "identify"],
        "Linear elasticity": ["tensors", "frame"],
    }
    assert menu("statistics") == {
        "Morphology": ["measures", "curves", "stereology", "percolation"],
        "Representative volume": ["rve", "fit_table"],
        "Reconstruction": ["reconstruction"],
        "Orientation maps": ["texture"],
    }
    assert list(menu("fracture")) == ["Cracks", "Cohesive zones", "Phase field"]
    assert list(menu("interop")) == [
        "AMITEX_FFTP",
        "Abaqus and meshes",
        "Code_Aster and OpenFOAM",
        "Behaviours of materials",
    ]
    assert menu("learning") == {"": ["learn"]} and menu("nothing") == {}
    # a heading gathers its tools wherever their orders stand among those of the others
    registered = dict(tools.TOOLS)
    try:
        for k, heading in enumerate(["B", "A", "B", ""]):
            tools.tool(f"demo{k}", "demo", f"Demo {k}", "", order=k, menu=heading)(lambda p, c: {})
        assert tools.menus("demo") == [
            ("B", [tools.TOOLS["demo0"], tools.TOOLS["demo2"]]),
            ("A", [tools.TOOLS["demo1"]]),
            ("", [tools.TOOLS["demo3"]]),
        ]
        assert tools.TOOLS["demo1"].to_dict()["menu"] == "A"
        # tools of a tab that is not declared: after the others
        assert [t["name"] for t in tools.catalog()["tools"]][-4:] == [f"demo{k}" for k in range(4)]
    finally:
        tools.TOOLS.clear()
        tools.TOOLS.update(registered)


def test_long_forms_in_boxes():
    """The long forms stand in boxes (specimen, material, loading...): each box holds
    consecutive fields, from its first field to the next box; the folded ones hold no
    required field."""
    for t in tools.catalog()["tools"]:
        boxes = [f.get("group", "") for f in t["fields"]]
        runs = [g for k, g in enumerate(boxes) if k == 0 or g != boxes[k - 1]]
        assert len(runs) == len(set(runs)), t["name"]  # no box split in two
        if len(t["fields"]) >= 25:
            assert sum(map(bool, set(runs))) >= 3, t["name"]
        for f in t["fields"]:
            assert not (f.get("fold") and f["required"]), (t["name"], f["name"])
    path = {
        f["name"]: f
        for f in next(x for x in tools.catalog()["tools"] if x["name"] == "path")["fields"]
    }
    assert path["source"]["group"] == path["phase"]["group"] == "Morphology and materials"
    assert path["zmat"]["group"] == "Behaviour" and path["clim"]["fold"] is True
    fields = [Field("a"), Field("b"), Field("c"), Field("d")]
    boxed = tools._grouped("demo", fields, [("First", "b"), ("Second", "c", True)])
    assert [(f.group, f.fold) for f in boxed] == [
        ("", False),
        ("First", False),
        ("Second", True),
        ("Second", True),
    ]
    with pytest.raises(ValueError, match="no field e"):
        tools._grouped("demo", fields, [("Nowhere", "e")])


def test_percolation_tool(tmp_path):
    """The percolation threshold of the model of the Boolean spheres of the workspace against
    their volume fraction: the probabilities tried, drawn and tabulated, cross 1/2 within
    the final bracket (the forest fire too); a parameter the model does not have is refused."""
    session, key = _workspace(tmp_path)
    params = {"source": key, "lower": 0.1, "upper": 0.6, "resolution": 24, "realizations": 3,
              "tol": 0.05, "seed": 2}  # fmt: skip
    out = _run(session, "percolation", params)
    values = out["values"]
    assert values["bracket from"] < values["threshold"] < values["bracket to"]
    assert values["bracket to"] - values["bracket from"] <= 0.05
    assert [f["title"] for f in out["figures"]] == ["percolation probability"]
    table = next(t for t in out["tables"] if t["title"] == "probabilities tried")
    assert table["columns"] == ["volume_fraction", "probability"]
    assert table["rows"][0][1] < 0.5 <= table["rows"][-1][1]
    fire = _run(session, "percolation", {**params, "method": "fire"})["values"]
    assert fire["bracket from"] < fire["threshold"] < fire["bracket to"]
    with pytest.raises(AssertionError, match="no parameter 'porosity'"):
        _run(session, "percolation", {**params, "parameter": "porosity"})


def test_percolation_tool_scaling(tmp_path):
    """With box sizes, the thresholds of cubes of these sizes at the voxel size of the
    resolution: their probabilities drawn, their fit to p_c + a L^(-1/nu) drawn and
    tabulated; fewer than 3 sizes refused."""
    session, key = _workspace(tmp_path)
    params = {"source": key, "lower": 0.1, "upper": 0.6, "resolution": 24, "realizations": 3,
              "tol": 0.05, "seed": 2, "method": "fire", "sizes": "12, 18, 24"}  # fmt: skip
    out = _run(session, "percolation", params)
    values = out["values"]
    assert {"threshold", "standard error", "amplitude", "nu", "realizations"} <= set(values)
    assert values["nu"] == 0.8765 and values["standard error"] > 0
    assert [f["title"] for f in out["figures"]] == ["percolation probability",
                                                    "finite-size scaling"]  # fmt: skip
    table = next(t for t in out["tables"] if t["title"] == "thresholds of the boxes")
    assert table["columns"] == ["size", "volume_fraction", "bracket from", "bracket to",
                                "residual"]  # fmt: skip
    assert [row[0] for row in table["rows"]] == [12, 18, 24]
    with pytest.raises(AssertionError, match="at least 3"):
        _run(session, "percolation", {**params, "sizes": "12, 24"})


def test_percolation_tool_exponent(tmp_path):
    """With a critical exponent, the means of the property at the threshold over the box
    sizes, drawn with their power law and tabulated, the exponent and its errors: the
    conductivity and the elastic modulus at a threshold given (its standard error 0); one
    extrapolated from boxes so small that it leaves the bounds within its error refused."""
    session, key = _workspace(tmp_path)
    params = {"source": key, "lower": 0.05, "upper": 0.6, "resolution": 24, "realizations": 3,
              "tol": 0.05, "seed": 2, "sizes": "9, 12, 15", "exponent": "thermal",
              "threshold": 0.25, "device": "cpu"}  # fmt: skip
    out = _run(session, "percolation", params)
    assert [f["title"] for f in out["figures"]] == ["finite-size scaling of the property"]
    table = next(t for t in out["tables"] if t["title"] == "means of the boxes at the threshold")
    assert table["columns"] == ["size", "mean", "standard error", "residual"]
    assert [row[0] for row in table["rows"]] == [9, 12, 15]
    values = out["values"]
    assert {"exponent", "standard error", "nu", "threshold", "realizations"} <= set(values)
    assert values["threshold"] == 0.25 and values["threshold standard error"] == 0
    assert values["nu"] == 0.8765 and values["error from the threshold"] == 0
    elastic = {**params, "exponent": "elastic", "threshold_stderr": 0.01, "sizes": "7, 8, 9"}
    values = _run(session, "percolation", elastic)["values"]
    assert values["threshold standard error"] == 0.01 and values["error from the threshold"] > 0
    extrapolated = {**params, "threshold": None, "connectivity": "26", "boundaries": "periodic"}
    with pytest.raises(AssertionError, match="too uncertain for an exponent"):
        _run(session, "percolation", extrapolated)


def test_percolation_tool_rigidity(tmp_path):
    """The rigidity threshold of a planar morphology (the sections of spheres: discs) over box
    sizes: the shares of the realizations that bear and their moduli against the fraction,
    the finite-size scaling and its values, the same by the voxels assembled and factorized
    in fewer iterations; the elastic exponent at this threshold when none is given; refused
    without box sizes."""
    session, _ = _workspace(tmp_path)
    model = model_from_config({"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.5})
    domain = {"size": [8.0, 8.0, 0.25], "periodic": False}
    micro = model.generate(Domain(size=(8.0, 8.0, 0.25), periodic=False), derive_seed(7, 1))
    planar, _ = session.folder()
    session.add(planar, "morphology", "discs", micro=micro, seed=7, model=model.to_config(),
                domain=domain, phases={"0": "matrix", "1": "spheres"})  # fmt: skip
    params = {"source": planar, "lower": 0.2, "upper": 0.95, "resolution": 32, "realizations": 3,
              "tol": 0.05, "seed": 4, "sizes": "12, 16, 20", "rigidity": True,
              "device": "cpu"}  # fmt: skip
    out = _run(session, "percolation", params)
    assert [f["title"] for f in out["figures"]] == ["bearing probability",
                                                    "moduli of the realizations",
                                                    "finite-size scaling"]  # fmt: skip
    values = out["values"]
    assert {"threshold", "standard error", "cut", "solutions", "decided", "unresolved"} <= set(
        values
    )
    assert values["cut"] == 1e-8 and values["smallest rigid modulus"] > 1e-8
    table = next(t for t in out["tables"] if t["title"] == "thresholds of the boxes")
    assert [row[0] for row in table["rows"]] == [12, 16, 20]
    fe = _run(session, "percolation", {**params, "solver": "fe"})["values"]
    assert fe["threshold"] == values["threshold"] and fe["solutions"] == values["solutions"]
    assert fe["iterations"] <= 10 < values["iterations"]
    elastic = {**params, "rigidity": False, "exponent": "elastic"}
    exponent = _run(session, "percolation", elastic)["values"]
    assert exponent["threshold"] == values["threshold"] and np.isfinite(exponent["exponent"])
    with pytest.raises(AssertionError, match="at least 3"):
        _run(session, "percolation", {**params, "sizes": None})


def test_percolation_tool_anisotropy(tmp_path):
    """With an anisotropy, the thresholds across the cores of the box along directions: the
    13 of the lattice of the spheres, as bars, and 6 sampled in the plane of a planar
    morphology (the sections of spheres: discs), as a rose; tabulated, with their ratio and
    the eigenvalues of their tensor (3, or 2 in the plane)."""
    session, key = _workspace(tmp_path)
    params = {"source": key, "lower": 0.1, "upper": 0.6, "resolution": 16, "realizations": 3,
              "tol": 0.05, "seed": 2, "directions": "lattice"}  # fmt: skip
    out = _run(session, "percolation", params)
    assert [f["title"] for f in out["figures"]] == ["thresholds along directions"]
    table = next(t for t in out["tables"] if t["title"] == "thresholds along directions")
    assert table["columns"][:4] == ["x", "y", "z", "volume_fraction"] and len(table["rows"]) == 13
    assert {"largest over smallest", "eigenvalue 3", "realizations"} <= set(out["values"])
    model = model_from_config({"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.5})
    domain = {"size": [4.0, 4.0, 0.125], "periodic": False}
    micro = model.generate(Domain(size=(4.0, 4.0, 0.125), periodic=False), derive_seed(7, 1))
    planar, _ = session.folder()
    session.add(planar, "morphology", "discs", micro=micro, seed=7, model=model.to_config(),
                domain=domain, phases={"0": "matrix", "1": "spheres"})  # fmt: skip
    disc = {**params, "source": planar, "resolution": 32, "lower": 0.2, "upper": 0.9,
            "directions": "sampled", "count": 6}  # fmt: skip
    out = _run(session, "percolation", disc)
    table = next(t for t in out["tables"] if t["title"] == "thresholds along directions")
    assert len(table["rows"]) == 6 and all(row[2] == 0 for row in table["rows"])
    assert {"eigenvalue 2", "realizations"} <= set(out["values"])
    assert "eigenvalue 3" not in out["values"]
