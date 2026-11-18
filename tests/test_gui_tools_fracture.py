"""The tools of the tab Fracture: their catalog and forms; the stress intensity factors of an
edge crack in a plate (against the handbook, the G-θ method against K² / E', its rings and
the energy of the body), of a centre crack, of antiplane and in-plane shear, of a morphology
whose phases are alike (the homogeneous plate) and of a penny crack (against Sneddon); the
propagation of a crack, straight in mode I and kinked in mixed mode, by its toughness (the
first advance at the load of Griffith) and in a viscoplastic plate (the resistance curve),
and through a morphology (stiff spheres at a fixed load, spheres as pores by the
toughness; the factors with pores; a laminate whose weak interfaces deflect the crack along
them, by the toughness of its phases and interfaces); cohesive zones (a ligament broken
through, its energies
balanced; a double cantilever beam against the corrected beam theory; the interfaces of a
laminate in series; the grain boundaries of a polycrystal), the growth of a crack
through a polycrystal and the phase-field fracture of a morphology (a section of stiff
spheres that do not break: the matrix breaks at a strength of the order of that of its
phase field, the stress falls and the work done is that of the cracks)."""

import math

import numpy as np
import pytest

from morphogen.domain import Domain
from morphogen.fracture import sent_factor
from morphogen.gui import api, tools
from morphogen.gui.tools import ToolJob
from morphogen.mesh.conforming import has_mmg

mmg = pytest.mark.skipif(not has_mmg(), reason="MMG")


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


def _polycrystal(session):
    """Twelve columnar grains of random orientations in a plate of 8 x 8 x 0.5 (the germs in
    its mid-plane), as a morphology of the workspace."""
    from morphogen.models.tessellation import Tessellation

    rng = np.random.default_rng(3)
    germs = np.column_stack([rng.uniform(0, 8, 12), rng.uniform(0, 8, 12), np.full(12, 0.25)])
    euler = rng.uniform(0, 1, (12, 3)) * [360.0, 180.0, 360.0]
    micro = Tessellation(Domain(size=(8.0, 8.0, 0.5), periodic=False), germs, orientations=euler)
    key, _ = session.folder()
    session.add(key, "morphology", "polycrystal", micro=micro, seed=3, model=None,
                domain={"size": [8.0, 8.0, 0.5], "periodic": False},
                phases={"1": "cells"})  # fmt: skip
    return key


def test_catalog_and_forms(tmp_path):
    catalog = tools.catalog()
    tab = next(t for t in catalog["tabs"] if t["name"] == "fracture")
    assert tab["order"] == 50 and tab["title"] == "Fracture"
    fracture = [t for t in catalog["tools"] if t["tab"] == "fracture"]
    assert [t["name"] for t in fracture] == [
        "crack_factors", "crack_growth", "front_growth", "polycrystal_crack",
        "cohesive_zones", "phase_field"]  # fmt: skip
    assert [t["menu"] for t in fracture] == ["Cracks"] * 4 + ["Cohesive zones", "Phase field"]
    for t in fracture:  # the conditions of a field refer to the fields before it
        names = []
        for f in t["fields"]:
            assert set(f.get("when", {})) <= set(names), (t["name"], f["name"])
            names.append(f["name"])
        assert len(names) == len(set(names))
    session = api.Session(tmp_path)
    parsed = tools.TOOLS["crack_factors"].parse({}, session)
    assert parsed["crack"] == "edge" and parsed["source"] is None and parsed["young"] == 1000.0
    assert parsed["plane_strain"] is True and parsed["yield_stress"] is None
    morphology = tools.TOOLS["crack_factors"].parse({"bulk": "morphology", "source": None}, session)
    assert morphology["young"] is None and morphology["behaviour"] is None
    # errors of the forms, at once or as the job starts
    assert "through" in _error(session, "crack_factors", {"crack": "through", "loading": "shear"})
    assert "porous" in _error(session, "crack_growth", {"growth": "porosity"})
    assert "elastic bulk" in _error(session, "crack_growth", {"behaviour": "plastic"})
    assert "penny" in _error(session, "crack_factors", {"crack": "penny", "length": 3.0})
    assert "polycrystal" in _error(session, "polycrystal_crack", {"source": _spheres(session)})
    assert "void" in _error(session, "crack_factors", {"bulk": "morphology",
                                                       "source": _spheres(session),
                                                       "materials": {"0": "void"}})  # fmt: skip


def _spheres(session, size=(16.0, 16.0, 1.0)):
    """A Boolean model of spheres as a morphology of the workspace."""
    from morphogen.models.base import model_from_config
    from morphogen.seeding import derive_seed

    model = model_from_config({"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3})
    micro = model.generate(Domain(size=size, periodic=False), derive_seed(3, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "spheres", micro=micro, seed=3, model=model.to_config(),
                domain={"size": list(size), "periodic": False},
                phases={"0": "matrix", "1": "spheres"})  # fmt: skip
    return key


@mmg
def test_factors_of_an_edge_crack(tmp_path):
    """The edge crack of 0.28 W in the plate of 16 x 32 in tension (plane strain, elements of
    1): K_I a few percent below the handbook (linear tetrahedra), the other modes nil, G of
    the G-θ method that of K_I and independent of its ring, close to the energy of the body
    for the crack shorter and longer by an element."""
    session = api.Session(tmp_path)
    result = _run(session, "crack_factors", {"energy_check": True})
    v = result["values"]
    handbook = math.sqrt(math.pi * 4.48) * sent_factor(4.48 / 16.0)  # the unit traction
    assert v["K_I_handbook"] == pytest.approx(handbook)
    assert 0.88 < v["K_I_ratio"] < 1.0 and v["K_I"] == pytest.approx(v["K_I_ratio"] * handbook)
    assert abs(v["K_II"]) + abs(v["K_III"]) < 0.02 * v["K_I"] and abs(v["kink_angle_deg"]) < 1
    assert v["G"] == pytest.approx(v["K_I"] ** 2 * (1 - 0.3**2) / 1000.0, rel=0.03)
    assert v["J"] == pytest.approx(v["G"], rel=0.03)
    assert v["G_ratio_energy"] == pytest.approx(1.0, abs=0.1)
    rings = _table(result, "independence of the domain")  # beyond the ring of 1 to 2 elements
    wide = [r for r in rings if r["r_out"] > 2.5 * v["element_size"]]
    assert len(wide) >= 2 and all(r["G"] == pytest.approx(v["G"], rel=0.02) for r in wide)
    assert v["crack_area"] == pytest.approx(4.48 * 1.0, rel=0.02)
    assert [f["title"] for f in result["figures"]] == [
        "mesh conforming to the crack", "von Mises stress near the front"]  # fmt: skip
    assert {"crack.vtu", "mesh.png", "fields.png", "domains.csv"} <= {
        f["name"] for f in result["files"]}  # fmt: skip
    assert "Tada" in result["text"]


@pytest.mark.slow
@mmg
def test_factors_of_other_specimens(tmp_path):
    """The centre crack of 2 x 4 in a plate of 32 (its half) in tension near the factor of a
    long strip (the plate is square: a little above it); the edge crack in antiplane shear,
    its in-plane motion held, K_III within 4 % of the exact factor of the strip; the shear of
    the edge crack, pure mode II, kinking at -70.5°."""
    session = api.Session(tmp_path)
    through = _run(session, "crack_factors", {"crack": "through"})["values"]
    assert 0.97 < through["K_I_ratio"] < 1.08 and abs(through["K_II"]) < 0.02 * through["K_I"]
    anti = _run(session, "crack_factors", {"loading": "antiplane", "poisson": 0.0})["values"]
    assert anti["K_III_ratio"] == pytest.approx(1.0, abs=0.04)
    assert abs(anti["K_I"]) + abs(anti["K_II"]) < 0.02 * anti["K_III"]
    assert anti["G_III"] == pytest.approx(anti["G"], rel=1e-3)
    shear = _run(session, "crack_factors", {"loading": "shear"})["values"]
    assert abs(shear["K_I"]) < 0.02 * shear["K_II"] and "K_I_handbook" not in shear
    assert shear["kink_angle_deg"] == pytest.approx(-70.53, abs=1.0)


@pytest.mark.slow
@mmg
def test_factors_of_a_morphology(tmp_path):
    """A morphology whose two phases have the same material: the factors of the homogeneous
    plate of its domain (the same crack mesh); stiffer spheres change them."""
    session = api.Session(tmp_path)
    key = _spheres(session)
    alike = {"0": {"young": 1000, "poisson": 0.3}, "1": {"young": 1000, "poisson": 0.3}}
    same = _run(session, "crack_factors", {"bulk": "morphology", "source": key,
                                           "materials": alike})["values"]  # fmt: skip
    plate = _run(session, "crack_factors", {"width": 16, "height": 16, "thickness": 1})["values"]
    for name in ("K_I", "K_II", "G", "elements"):
        assert same[name] == pytest.approx(plate[name], rel=1e-9, abs=1e-12)
    stiff = {"0": {"young": 1000, "poisson": 0.3}, "1": {"young": 5000, "poisson": 0.3}}
    other = _run(session, "crack_factors", {"bulk": "morphology", "source": key,
                                            "materials": stiff})  # fmt: skip
    assert abs(other["values"]["K_I"] / plate["K_I"] - 1) > 0.02
    assert "K_I_handbook" not in other["values"]


@pytest.mark.slow
@mmg
def test_factors_of_a_penny_crack(tmp_path):
    """A penny crack of radius 1 in a cube of 4 in tension: K_I within 10 % of Sneddon's
    infinite body (the cube raises it) and uniform along the front, its profile at
    stations; a 3D view when VTK renders."""
    from morphogen.viz.video import vtk_can_render

    session = api.Session(tmp_path)
    result = _run(session, "crack_factors", {"crack": "penny"})
    v = result["values"]
    assert v["K_I_handbook"] == pytest.approx(2.0 * math.sqrt(1.0 / math.pi))
    assert 1.0 < v["K_I_ratio"] < 1.1
    profile = _table(result, "along the front")
    k1 = np.array([row["K_I"] for row in profile])
    assert len(k1) >= 4 and np.std(k1) < 0.05 * np.mean(k1)
    assert v["front_length"] == pytest.approx(2 * math.pi, rel=1e-6)
    titles = [f["title"] for f in result["figures"]]
    assert "front" in titles and ("von Mises stress (3D, cut at the middle of z)" in titles) == (
        vtk_can_render()
    )


@mmg
def test_growth_stays_straight_in_mode_one(tmp_path):
    """Fatigue growth of the edge crack in tension: the path stays straight (the kinks below
    a degree, the tip on the plane of the crack), G grows with the length, Paris' law counts
    the cycles, Griffith's critical load falls; the path, the video and the last mesh."""
    session = api.Session(tmp_path)
    result = _run(session, "crack_growth", {"steps": 1, "paris_C": 1e-3, "toughness": 0.05,
                                            "element_size": 1.2})  # fmt: skip
    rows = _table(result, "the cracks")
    assert len(rows) == 2 and rows[1]["length"] == pytest.approx(4.48 + 1.8, rel=1e-6)
    assert all(abs(r["angle_deg"]) < 1.0 for r in rows)
    assert all(abs(r["y_tip"] - 16.0) < 0.05 for r in rows)
    assert rows[1]["G"] > rows[0]["G"] and rows[1]["N"] > 0
    assert rows[1]["load_critical"] < rows[0]["load_critical"]
    assert rows[0]["load_critical"] == pytest.approx(math.sqrt(0.05 / rows[0]["G"]), rel=1e-5)
    assert [v["title"] for v in result["videos"]] == ["the growth, crack by crack"]
    assert {"path.png", "last_crack.vtu", "advances.csv"} <= {f["name"] for f in result["files"]}


@pytest.mark.slow
@mmg
def test_growth_by_toughness_in_mixed_mode(tmp_path):
    """The edge crack in tension and shear (0.7), its toughness 1.2 times the G of the unit
    load: the load raised by steps of 0.25 reaches it at 1.25 (G as the square of the load),
    the crack kinks by about -41° (Erdogan & Sih), then advances at once (dead loads)."""
    session = api.Session(tmp_path)
    result = _run(session, "crack_growth", {"growth": "toughness", "loading": "mixed",
                                            "steps": 1, "element_size": 1.2})  # fmt: skip
    v = result["values"]
    rows = _table(result, "the advances")
    loading = _table(result, "every load step")
    first = next(r for r in loading if r["load"] == 1.0)
    assert v["toughness"] == pytest.approx(1.2 * first["J_tip"], rel=1e-5)
    assert rows[0]["load"] == pytest.approx(1.25)
    assert rows[0]["J_tip"] == pytest.approx(1.25**2 * first["J_tip"], rel=1e-5)
    assert -46.0 < rows[0]["angle_deg"] < -36.0
    assert v["advances"] == 1 and v["stopped"] == "after the last step"
    assert rows[1]["y_tip"] < 16.0 - 0.5  # kinked towards -y
    assert {"R-curve", "criterion", "crack path"} <= {f["title"] for f in result["figures"]}


@pytest.mark.slow
@mmg
def test_ductile_growth_of_a_viscoplastic_plate(tmp_path):
    """The plate of a viscoplastic metal pulled by its grips, its crack growing straight once
    J_tip reaches 0.03: the load of the second advance higher (the plastic wake shields the
    tip), the far J rising (the resistance curve), the field the cumulated plastic strain."""
    session = api.Session(tmp_path)
    result = _run(session, "crack_growth", {"growth": "toughness", "behaviour": "plastic",
                                            "viscosity": 1.0, "control": "grips", "load": 0.03,
                                            "toughness": 0.03, "load_step": 0.5, "kink": False,
                                            "steps": 1})  # fmt: skip
    rows = _table(result, "the advances")
    assert len(rows) == 2 and rows[1]["load"] > rows[0]["load"]
    assert all(r["J_tip"] >= 0.03 for r in rows)
    assert rows[1]["J_far"] > 1.1 * rows[0]["J_far"]
    assert all(abs(r["y_tip"] - 16.0) < 1e-9 for r in rows)


def _hardcore(session, size=(12.0, 12.0, 1.0)):
    """Hard-core spheres (radius 1, 20 %) cut by a plate, as a morphology of the workspace."""
    from morphogen.models.base import model_from_config
    from morphogen.seeding import derive_seed

    model = model_from_config({"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.2})
    micro = model.generate(Domain(size=size, periodic=False), derive_seed(3, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "hard-core spheres", micro=micro, seed=3,
                model=model.to_config(), domain={"size": list(size), "periodic": False},
                phases={"0": "matrix", "1": "spheres"})  # fmt: skip
    return key


@pytest.mark.slow
@mmg
def test_growth_through_a_morphology(tmp_path):
    """The crack grows at a fixed load through stiff hard-core spheres (10 E), on crack meshes
    conforming to them: the phase at each tip in the table, the path over the phases and the
    video of the growth, the phases on the last mesh; the factors of the spheres as pores,
    left out of the mesh: fewer elements, and under the same traction a higher K_I than with
    the stiff spheres."""
    session = api.Session(tmp_path)
    key = _hardcore(session)
    stiff = {"0": {"young": 1000, "poisson": 0.3}, "1": {"young": 10000, "poisson": 0.3}}
    params = {"bulk": "morphology", "source": key, "materials": stiff, "steps": 2,
              "element_size": 0.8}  # fmt: skip
    result = _run(session, "crack_growth", params)
    v = result["values"]
    rows = _table(result, "the cracks")
    assert len(rows) == 3 and all(r["phase"] in (0, 1) for r in rows)
    assert all(r["G"] > 0 for r in rows) and rows[2]["length"] > rows[0]["length"]
    assert v["interfaces"] == "conforming" and v["phases_at_tips"].count("/") == 2
    assert {"G along the growth", "crack path"} <= {f["title"] for f in result["figures"]}
    assert [x["title"] for x in result["videos"]] == ["the growth, crack by crack"]
    assert "last_crack.vtu" in {f["name"] for f in result["files"]}
    pores = {"0": {"young": 1000, "poisson": 0.3}, "1": "void"}
    factors = {"bulk": "morphology", "source": key, "interfaces": "conforming",
               "element_size": 0.8, "load": 1.0}  # fmt: skip
    solid = _run(session, "crack_factors", {**factors, "materials": stiff})["values"]
    porous = _run(session, "crack_factors", {**factors, "materials": pores})["values"]
    assert porous["pores"] == "1: spheres" and porous["elements"] < solid["elements"]
    assert porous["K_I"] > solid["K_I"]


@pytest.mark.slow
@mmg
def test_ductile_growth_through_a_porous_morphology(tmp_path):
    """The spheres as pores, the crack growing once its G at the tip reaches its toughness
    (1.2 times the G of the unit load): the advances at the toughness, the resistance curve
    and the criterion at each load step, the pores in the values."""
    session = api.Session(tmp_path)
    key = _hardcore(session)
    params = {"bulk": "morphology", "source": key, "growth": "toughness", "steps": 1,
              "element_size": 0.8, "materials": {"0": {"young": 1000, "poisson": 0.3},
                                                 "1": "void"}}  # fmt: skip
    result = _run(session, "crack_growth", params)
    v = result["values"]
    rows = _table(result, "the advances")
    assert len(rows) == 2 and all(r["J_tip"] >= v["toughness"] * (1 - 1e-9) for r in rows)
    assert v["pores"] == "1: spheres" and v["advances"] == 1
    assert {"R-curve", "criterion", "crack path"} <= {f["title"] for f in result["figures"]}


@pytest.mark.slow
@mmg
def test_growth_along_weak_interfaces(tmp_path):
    """Layers of 1 normal to the crack (a laminate of alike phases), their interfaces ten times
    weaker than the phases: at a fixed load the crack meets an interface within an increment
    and turns along it (the advances along interfaces and the deflections in the values, the
    bulk or the interface of each advance, what it meets, its toughness and its ratio G / Γ in
    the table, the critical load 1 / sqrt(G / Γ) of each crack and its figure); a wrong count
    of toughnesses of the phases, or a toughness of the interfaces alone, is refused."""
    from morphogen.models.base import model_from_config
    from morphogen.seeding import derive_seed

    session = api.Session(tmp_path)
    model = model_from_config(
        {"type": "random_laminate", "thickness": 1.0, "spacing": 1.0, "axis": 0}
    )
    micro = model.generate(Domain(size=(12.0, 12.0, 1.0), periodic=False), derive_seed(3, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "layers", micro=micro, seed=3, model=model.to_config(),
                domain={"size": [12.0, 12.0, 1.0], "periodic": False},
                phases={"0": "layers 0", "1": "layers 1"})  # fmt: skip
    alike = {"0": {"young": 1000, "poisson": 0.3}, "1": {"young": 1000, "poisson": 0.3}}
    params = {"bulk": "morphology", "source": key, "materials": alike, "steps": 1,
              "element_size": 0.8, "toughness_phases": "1, 1",
              "interface_toughness": 0.1}  # fmt: skip
    result = _run(session, "crack_growth", params)
    v = result["values"]
    rows = _table(result, "the cracks")
    assert v["along_interfaces"] == 1 and v["deflections"] == 1 and v["interface_toughness"] == 0.1
    assert abs(rows[0]["angle_deg"]) == pytest.approx(90.0, abs=1.0)  # into the interface
    assert abs(rows[1]["angle_deg"]) < 1.0  # then along it
    for r in rows:
        assert r["path"] == "interface" and r["met"] == "0/1" and r["toughness"] == 0.1
        assert r["load_critical"] == pytest.approx(1.0 / math.sqrt(r["ratio"]), rel=1e-4)
    assert v["path_length"] > 2.0 * v["path_extent"]
    assert {"critical loads", "crack path"} <= {f["title"] for f in result["figures"]}
    wrong = _error(session, "crack_growth", {**params, "toughness_phases": "1, 1, 1"})
    assert "one value per phase" in wrong
    alone = _error(session, "crack_growth", {**params, "toughness_phases": None})
    assert "toughness of the phases too" in alone


def test_cohesive_ligament(tmp_path):
    """The ligament of 6 ahead of a notch of 2 in a plate of 8 x 8 x 1 pulled apart: it breaks
    through (the force back to zero), its peak below the strength over the ligament, the
    works of the bulk and of the interfaces making that of the loads, the elastic bulk
    storing nothing at the end, the interfaces dissipating at least G_c per unit of area."""
    session = api.Session(tmp_path)
    result = _run(session, "cohesive_zones", {})
    v = result["values"]
    assert v["interface_area"] == pytest.approx(6.0) and v["cracked_area"] == pytest.approx(6.0)
    assert 0.5 * 10.0 * 6.0 < v["peak_force"] < 10.0 * 6.0
    assert v["work_external"] == pytest.approx(v["work_bulk"] + v["work_interfaces"], rel=1e-6)
    assert abs(v["work_bulk"]) < 1e-6 * v["work_external"] and v["dissipated_bulk"] == 0.0
    assert v["work_interfaces"] > 0.2 * 6.0 and v["G_c"] == pytest.approx(0.2)
    assert v["zone_elements"] == pytest.approx(4.0)
    steps = _table(result, "the steps")
    assert abs(steps[-1]["R1y"]) < 1e-3 * v["peak_force"]
    assert [f["title"] for f in result["figures"]] == [
        "force-displacement", "energies", "broken interfaces"]  # fmt: skip
    assert {"cohesive.vtu", "cohesive_interfaces.vtu"} <= {f["name"] for f in result["files"]}
    # the 3D view draws the bulk, not the wedges of the interfaces (named after it)
    shown = api.surface({"key": result["key"], "resolution": 24}, session)
    assert result["view3d"] and shown["kind"] == "result" and shown["displacement"]


@pytest.mark.slow
def test_cohesive_beam(tmp_path):
    """The double cantilever beam of 60 x 2, arms of 4, a precrack of 20 (E = 1000, G_Ic =
    0.5, N = 5: a cohesive zone of 20 elements): its peak load within 3 % of the corrected
    beam theory, the crack growing."""
    session = api.Session(tmp_path)
    v = _run(session, "cohesive_zones", {"specimen": "dcb"})["values"]
    assert v["G_c"] == pytest.approx(0.5) and v["zone_elements"] == pytest.approx(20.0)
    assert v["peak_force"] == pytest.approx(v["peak_beam_theory"], rel=0.03)
    assert v["cracked_area"] > 10.0


@mmg
def test_cohesive_interfaces_of_a_laminate(tmp_path):
    """Layers of 1 stacked along y (E = 1000 and 3000), their interfaces cohesive, pulled
    along y: the interfaces in series, one breaks through at the strength over the section
    (N = E / 100 of the stiffest layer: within the overshoot of the coarse steps), the bulk
    unloaded at the end."""
    from morphogen.models.base import model_from_config
    from morphogen.seeding import derive_seed

    session = api.Session(tmp_path)
    model = model_from_config(
        {"type": "random_laminate", "thickness": 1.0, "spacing": 1.0, "axis": 1}
    )
    micro = model.generate(Domain(size=(4.0, 4.0, 1.0), periodic=False), derive_seed(3, 0))
    key, _ = session.folder()
    session.add(key, "morphology", "laminate", micro=micro, seed=3, model=model.to_config(),
                phases={"0": "layers 0", "1": "layers 1"})  # fmt: skip
    params = {"specimen": "interfaces", "source": key, "interfaces": "phases", "element_size": 1}
    v = _run(session, "cohesive_zones", params)["values"]
    assert v["interface_area"] == pytest.approx(4 * 4.0)
    assert 0.95 * 30.0 * 4.0 < v["peak_force"] < 1.15 * 30.0 * 4.0
    assert v["cracked_area"] >= 4.0 * (1 - 1e-6)
    assert v["stored_bulk"] < 0.01 * v["work_external"]
    assert v["phases"] == "0: layers 0, 1: layers 1"


@mmg
def test_cohesive_grain_boundaries(tmp_path):
    """The grain boundaries of a polycrystal cohesive, pulled across: some break (a path
    across the plate), not all; the works balance."""
    session = api.Session(tmp_path)
    key = _polycrystal(session)
    params = {"specimen": "interfaces", "source": key, "interfaces": "grains", "element_size": 0.7,
              "steps": 40}  # fmt: skip
    v = _run(session, "cohesive_zones", params)["values"]
    assert v["interface_elements"] > 100 and 0.05 < v["broken_fraction"] < 0.8
    assert v["peak_force"] > 0
    assert v["work_external"] == pytest.approx(v["work_bulk"] + v["work_interfaces"], rel=1e-6)


@mmg
def test_polycrystal_crack_growth(tmp_path):
    """A crack of 2.5 from the middle of the left edge of the columnar polycrystal grows by 1
    once J_tip reaches 0.6 (crystal plasticity, plane strain): the advances at increasing
    strains, the tips in the grains of the tessellation, the path over the grains."""
    session = api.Session(tmp_path)
    key = _polycrystal(session)
    params = {"source": key, "steps": 1, "length": 2.5, "increment": 1.0, "element_size": 0.7}
    result = _run(session, "polycrystal_crack", params)
    rows = _table(result, "the advances")
    assert len(rows) == 2 and all(r["J_tip"] >= 0.6 for r in rows)
    assert rows[1]["strain"] >= rows[0]["strain"] and rows[1]["length"] == pytest.approx(3.5)
    assert all(0 <= r["grain"] < 12 for r in rows)
    assert result["values"]["orientations"] == "of the tessellation"
    assert "crack path over the grains" in {f["title"] for f in result["figures"]}


def test_phase_field_fracture_of_a_morphology(tmp_path):
    from morphogen.models.base import model_from_config
    from morphogen.seeding import derive_seed

    session = api.Session(tmp_path)
    model = model_from_config({"type": "boolean_spheres", "radius": 0.6, "volume_fraction": 0.2})
    micro = model.generate(Domain(size=4.0, periodic=True), derive_seed(7, 0))
    key, _ = session.folder()
    session.add(
        key,
        "morphology",
        "spheres",
        micro=micro,
        seed=7,
        model=model.to_config(),
        domain={"size": 4.0, "periodic": True},
        phases={"0": "matrix", "1": "spheres"},
    )
    young, toughness, voxels = 1.0, 2e-5, 12
    length = 2 * 4.0 / voxels
    params = {
        "source": key,
        "materials": {
            "0": {"young": young, "poisson": 0.25},
            "1": {"young": 10.0, "poisson": 0.25},
        },
        "toughness": [toughness],  # the spheres do not break
        "voxels": voxels,
        "amplitude": 0.012,
        "increments": 10,
    }
    out = _run(session, "phase_field", params)
    values = out["values"]
    assert values["voxels"] == voxels * voxels and values["regularization length"] == length
    # the strength: of the order of that of the homogeneous matrix, sqrt(3 E' G_c / (8 l))
    critical = math.sqrt(3 * young / (1 - 0.25**2) * toughness / (8 * length))
    assert 0.4 * critical < values["strength (largest stress)"] < 1.6 * critical
    assert values["stress at the end"] < 0.1 * values["strength (largest stress)"]
    assert 0.0 < values["strain at the strength"] < 0.012
    work = values["work of the loading per unit volume"]
    assert values["fracture energy per unit volume"] == pytest.approx(work, rel=0.5)
    assert 0.1 < values["mean damage of the phases that break"] < 1.0
    curve = _table(out, "macroscopic curve")
    assert len(curve) == 11 and curve[0]["stress"] == 0.0
    assert max(row["damage"] for row in curve) == curve[-1]["damage"]
    assert [f["title"] for f in out["figures"]] == ["curve of the loading and damage at the end"]
    # in finite strains: the same plate at these small strains, its phases as rubbers
    finite = {**params, "kinematics": "finite", "energy": "neo_hookean", "split": "volumetric"}
    large = _run(session, "phase_field", finite)["values"]
    strength = values["strength (largest stress)"]
    assert large["strength (largest stress)"] == pytest.approx(strength, rel=0.25)
    assert large["stress at the end"] < 0.1 * large["strength (largest stress)"]
    assert large["regularized area of the cracks per unit volume"] == pytest.approx(
        values["regularized area of the cracks per unit volume"], rel=0.25
    )
    # a rubber does not take the spectral split of the logarithmic strain
    assert "spectral split" in _error(session, "phase_field", {**finite, "split": "spectral"})
    # the forms
    porous = {"0": params["materials"]["0"], "1": "void"}
    unbreakable = {**params, "toughness": [0.0, 1e-5], "materials": porous}
    assert "must break" in _error(session, "phase_field", unbreakable)
    assert "voxels per edge" in _error(session, "phase_field", {**params, "voxels": 4})
    assert "at least a voxel" in _error(session, "phase_field", {**params, "length": 0.01})
    assert "at most one per phase" in _error(
        session, "phase_field", {**params, "toughness": [1e-5, 1e-5, 1e-5]}
    )


@pytest.mark.slow
@mmg
def test_growth_of_a_crack_front_in_3d(tmp_path):
    """A penny crack inclined by 30° to the tension, in the volume of a block: the three
    modes along its front near those of the infinite body (Kassir & Sih), each point
    kinking by its own angle (46° where the shear is largest); grown once, its mode II
    falls. The forms."""
    session = api.Session(tmp_path)
    params = {"major": 1.0, "inclination": 30.0, "young": 1.0, "stress": 1.0,
              "element_size": 0.7, "front_size": 0.12, "stations": 8, "steps": 1,
              "increment": 0.4}  # fmt: skip
    out = _run(session, "front_growth", params)
    v = out["values"]
    exact = v["K_I of the infinite body (Kassir & Sih)"]
    assert exact == pytest.approx(2 * math.sqrt(1 / math.pi) * 0.75)
    assert v["K_I along the first front (mean)"] == pytest.approx(exact, rel=0.08)
    k2 = v["largest K_II of the infinite body (Kassir & Sih)"]
    assert v["largest |K_II| along the first front"] == pytest.approx(k2, rel=0.12)
    assert v["largest angle of kinking of the first front (degrees)"] == pytest.approx(46.4, abs=4)
    assert (
        v["largest |K_II| along the last front"] < 0.8 * v["largest |K_II| along the first front"]
    )
    assert v["area of the last crack over that of the first"] > 1.3
    assert v["element size at the front"] < 0.2
    growth = _table(out, "the crack at each step")
    assert [row["step"] for row in growth] == [0, 1] and growth[1]["area"] > growth[0]["area"]
    factors = _table(out, "the factors and the angle")
    assert len(factors) == 16 and {row["step"] for row in factors} == {0, 1}
    assert [f["title"] for f in out["figures"]] == [
        "stress intensity factors along the front",
        "the successive fronts of the crack",
    ]
    assert out["title"] == "Growth of a penny crack inclined by 30°"
    assert "b at most a" in _error(session, "front_growth", {**params, "minor": 2.0})
    assert "three elements of the front" in _error(
        session, "front_growth", {**params, "increment": 0.1}
    )
    assert "at least 2.5 times" in _error(session, "front_growth", {**params, "block": 2.0})
