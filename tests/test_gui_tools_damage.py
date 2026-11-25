"""The tools of the tab Nonlocal damage of the interface, run once each on tiny voxel
specimens: their forms and specimens (notched plates, bars, morphologies of the workspace
without their holes), the ductile damage with the stress triaxiality (which concentrates
the damage ahead of a notch in plane strain and breaks the ligament sooner than the
plastic strain alone), the microfoam bar whose band of localization has the width of the
article's Appendix A while the classical continuum's is one element wide (and snaps back),
the micromorphic damage band widening with the length, and a 3D bar."""

import math

import numpy as np
import pytest

from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.tools import ToolJob
from morphogen.gui.tools import nonlocal_damage as nd
from morphogen.micromorphic import Microfoam, characteristic_length
from morphogen.models.base import model_from_config
from morphogen.seeding import derive_seed


def _session(tmp_path, model=None, size=4.0):
    """A session, with a morphology of a model in its workspace when one is given."""
    session = api.Session(tmp_path)
    if model is None:
        return session, None
    model = model_from_config(model)
    micro = model.generate(Domain(size=size, periodic=True), derive_seed(7, 0))
    key, _ = session.folder()
    session.add(
        key,
        "morphology",
        f"{model.name} (seed 7)",
        micro=micro,
        seed=7,
        model=model.to_config(),
        domain={"size": size, "periodic": True},
        phases={str(k): v for k, v in model.active_phases().items()},
    )
    return session, key


def _run(session, name, params):
    job = ToolJob({"tool": name, "params": params}, session)
    job._run()  # synchronously
    status = job.status()
    assert status["state"] == "done", status.get("error")
    return status["result"]


def test_catalog_forms_and_specimens(tmp_path):
    catalog = tools.catalog()
    tab = next(t for t in catalog["tabs"] if t["name"] == "damage")
    assert tab["title"] == "Nonlocal damage" and tab["order"] == 55
    names = [t["name"] for t in catalog["tools"] if t["tab"] == "damage"]
    assert names == ["microfoam", "ductile", "brittle"]
    ductile = next(t for t in catalog["tools"] if t["name"] == "ductile")
    fields = {f["name"]: f for f in ductile["fields"]}
    assert fields["source"]["when"] == {"specimen": ["morphology"]}
    assert fields["thickness"]["when"] == {"specimen": ["notched_plate"], "condition": ["3d"]}
    assert fields["compare"]["when"] == {"triaxiality": ["rice_tracey", "johnson_cook", "lemaitre"]}
    session, _ = _session(tmp_path)
    parsed = tools.TOOLS["ductile"].parse({"triaxiality": "none", "elements": "9"}, session)
    assert parsed["elements"] == 9 and parsed["source"] is None and parsed["compare"] is None
    with pytest.raises(ValueError, match="one of"):
        tools.TOOLS["ductile"].parse({"triaxiality": "gurson"}, session)
    # a notched plate: a notch of 0.3 of the width at mid-height, a voxel row at least
    p = {**parsed, "notch_depth": 0.3, "notch_width": 0.1, "width": 1.0, "height": None}
    plate = nd._specimen(p)
    assert plate.shape == (9, 9, 1) and plate.notch == pytest.approx(0.3)
    holes = np.argwhere(plate.labels[:, :, 0] == nd.HOLE)
    assert set(holes[:, 1]) == {4} and set(holes[:, 0]) == {0, 1, 2}
    assert plate.estimate(1) == 3 * 19 * 19 * 3 + 10 * 10 * 2
    # the crack from the notch: the damaged voxels joining its tip, not those elsewhere
    damage = np.where(plate.labels == nd.HOLE, np.nan, 0.0)
    damage[3:7, 4, 0] = 0.95  # from the tip (x = 3 voxels) to x = 7 voxels
    damage[8, 0, 0] = 0.99  # at a corner
    assert nd._crack(damage, plate) == pytest.approx(4 / 6)
    damage[3, 4, 0] = 0.5
    assert nd._crack(damage, plate) == 0.0
    # a bar with a weaker layer at mid-height, in 3D: a square section
    bar = nd._specimen(
        {
            **p,
            "specimen": "bar",
            "notch_depth": 0.0,
            "imperfection": 0.1,
            "condition": "3d",
            "elements": 4,
        }
    )
    assert bar.shape == (4, 8, 4) and set(np.unique(bar.labels)) == {0, nd.WEAK}
    assert np.flatnonzero((bar.labels == nd.WEAK).any(axis=(0, 2))).tolist() == [3, 4]
    with pytest.raises(ValueError, match="imperfection"):
        nd._specimen({**p, "specimen": "bar", "notch_depth": 0.0, "imperfection": 1.5})
    # only the solid joining both grips is kept (the rest would float)
    solid = np.zeros((4, 5, 1), bool)
    solid[1, :, 0] = True  # a column from y = 0 to the top
    solid[3, 1:3, 0] = True  # an island
    assert np.array_equal(nd._load_bearing(solid), solid & (np.arange(4) == 1)[:, None, None])
    # widths at half maximum: a band reaching a face ends there
    assert nd._band_width(np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]), 0.5) == pytest.approx(1.0)
    assert nd._band_width(np.array([1.0, 1.0, 0.0, 0.0]), 0.5) == pytest.approx(1.0)


@pytest.mark.slow
def test_ductile_damage_concentrates_ahead_of_the_notch(tmp_path):
    """A notched plate in plane strain: the stress ahead of the notch is triaxial, and the
    plastic strain weighted by Rice & Tracey's growth of voids concentrates the damage
    there, so that the ligament loses its load sooner than with the plastic strain
    alone (Forest 2009; examples/ductile_triaxiality.py)."""
    session, _ = _session(tmp_path)
    params = {"elements": 11, "steps": 12, "strain": 0.02, "compare": True}
    out = _run(session, "ductile", params)
    v = out["values"]
    assert v["triaxiality ahead of the notch"] > 0.6  # uniaxial tension: 1/3
    ahead_rt = v["damage ahead of the notch / mean (rice_tracey)"]
    ahead_p = v["damage ahead of the notch / mean (none)"]
    assert ahead_rt > ahead_p + 0.05 and ahead_rt > 1.2
    left_rt = v["final force / peak (rice_tracey)"]
    left_p = v["final force / peak (none)"]
    assert left_rt < left_p - 0.15
    assert v["peak force (rice_tracey)"] == pytest.approx(v["peak force (none)"], rel=0.05)
    assert [f["title"] for f in out["figures"]] == ["force-displacement", "fields at the end"]
    # no video unless asked: the fields of the steps, for the 3D view and the export
    assert out["videos"] == [] and out["fields3d"]["field"] == "damage"
    names = {f["name"] for f in out["files"]}
    assert {"ductile.vtu", "force_displacement.csv", "fields.png", "ductile_steps.json",
            "ductile_step001.vti"} <= names  # fmt: skip
    shown = api.surface({"key": out["key"], "step": 0, "field": "damage"}, session)
    assert shown["step"] == 0 and len(shown["steps"]) >= params["steps"]
    video = api.export({"source": out["key"], "outputs": [], "video": "run", "field": "damage"},
                       session)["videos"]  # fmt: skip
    assert len(video) == 1
    curve = out["tables"][0]
    assert curve["columns"] == ["triaxiality", "displacement", "force", "nominal_stress"]
    assert curve["count"] == 2 * (v["steps"] + 1)


def test_microfoam_band_width_set_by_the_length(tmp_path):
    """A softening microfoam bar (ν = 0 and F = c/2: one-dimensional, the article's
    Appendix A) with a weaker layer: the band of the cumulated plastic strain has the
    width at half maximum of a sine arch, π / |ω| (Dillard, Forest & Ienny 2006), while
    the classical continuum localizes in one element and snaps back (the steps stop, the
    converged ones are kept)."""
    session, _ = _session(tmp_path)
    params = {
        "condition": "plane_stress",
        "width": 0.025,
        "height": 0.6,
        "elements": 1,
        "imperfection": 0.02,
        "grips": "clamped",
        "loading": "tension",
        "strain": 0.0125,
        "steps": 15,
        "young": 100.0,
        "poisson": 0.0,
        "F": 0.5,
        "yield_stress": 1.0,
        "hardening": -10.0,
        "length": 0.01,
        "classical": True,
        "video": "p",
    }
    out = _run(session, "microfoam", params)
    v = out["values"]
    material = Microfoam(
        young=100.0,
        poisson=0.0,
        F=0.5,
        length=0.01,
        micro=(0.0, 50.0),
        relative=(0.0, 1000.0, 1000.0),
    )  # the moduli of the form's blanks
    tangent = 100.0 * -10.0 / (-10.0 + 1.5 * 100.0)
    expected = math.pi * characteristic_length(material, tangent)
    assert v["band width (p at half maximum, along y)"] == pytest.approx(expected, rel=0.1)
    assert v["band width, classical continuum"] == pytest.approx(0.025)  # one voxel
    # the uniaxial yield of the weaker layer: σ_eq = sqrt(1.5) σ
    assert v["peak nominal stress (micromorphic)"] == pytest.approx(0.98 / math.sqrt(1.5), rel=0.02)
    assert v["final nominal stress (micromorphic)"] > 0.8 * v["peak nominal stress (micromorphic)"]
    assert any("classical continuum: stopped" in w for w in out["warnings"])
    titles = [f["title"] for f in out["figures"]]
    assert titles == ["force-displacement", "fields at the end", "micromorphic and classical bands"]
    assert out["videos"][0]["title"] == "p along the loading"


def test_micromorphic_damage_band_widens_with_the_length(tmp_path):
    """A bar of the micromorphic damage model with a weaker layer: a band of damage across
    it (d near 1, the force falling) whose width at half maximum of κ grows with the
    length ℓ (Peerlings et al. 1996; tests/test_micromorphic.py)."""
    session, _ = _session(tmp_path)
    base = {
        "condition": "plane_stress",
        "width": 0.025,
        "height": 1.0,
        "elements": 1,
        "grips": "clamped",
        "poisson": 0.0,
        "strain": 0.006,
        "steps": 20,
        "video": "none",
    }
    narrow = _run(session, "brittle", {**base, "length": 0.05})["values"]
    wide = _run(session, "brittle", {**base, "length": 0.1})["values"]
    key = "band width (κ at half maximum, along y)"
    assert wide[key] > 1.4 * narrow[key]
    assert narrow["largest damage"] > 0.95 and narrow["final force / peak"] < 0.3


def test_specimens_of_morphologies_and_in_3d(tmp_path):
    """The section of a porous morphology of the workspace (its spheres removed), and a
    notched bar in 3D (a groove all around it)."""
    model = {"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.2}
    session, key = _session(tmp_path, model)
    params = {
        "specimen": "morphology",
        "source": key,
        "holes": [1],
        "resolution": 12,
        "triaxiality": "none",
        "strain": 0.01,
        "steps": 6,
        "video": "none",
    }
    out = _run(session, "ductile", params)
    v = out["values"]
    assert 0 < v["elements"] < 6 * 12 * 12 and v["largest damage (none)"] > 0
    assert "band width (κ at half maximum, along y)" in v
    with pytest.raises(ValueError, match="no path for the load"):
        nd._specimen({**tools.TOOLS["ductile"].parse(params, session), "holes": [0, 1]})
    out = _run(
        session,
        "brittle",
        {"condition": "3d", "elements": 3, "notch_depth": 0.34, "steps": 8, "video": "damage"},
    )
    assert out["values"]["elements"] < 6 * 3 * 6 * 3  # the groove removed
    assert out["videos"] and {"damage.vtu", "force_displacement.png"} <= {
        f["name"] for f in out["files"]
    }
