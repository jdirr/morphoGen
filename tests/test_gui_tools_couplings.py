"""The tools of the tab Couplings of the interface, run on small voxel images: the
transient diffusion into a morphology of the workspace (Fick's √t at short times, Crank's
slab and the steady effective diffusivity of a homogeneous image, both within the bounds
of a two-phase one, the Thiele steady state with a consumption), the thermal oxidation
of an epoxy plate (an oxidized layer of a few hundred micrometres, Ernault et al. 2019)
and the hygromorph multilayers (Timoshenko's bilayer, bending towards the layer that
shrinks, a layer of aligned fibres bending across them)."""

import math

import numpy as np
import pytest

from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.tools import ToolJob
from morphogen.models.base import model_from_config
from morphogen.seeding import derive_seed

SPHERES = {"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.3}
#: Fibres along x (in a cube of edge 8)
FIBRES = {
    "type": "boolean_cylinders",
    "radius": 0.6,
    "length": 3.5,
    "volume_fraction": 0.3,
    "orientation": {"type": "aligned", "direction": [1, 0, 0]},
}


def _session(tmp_path, *models, size=4.0):
    """A session whose workspace holds a morphology of each model (periodic cubes)."""
    session = api.Session(tmp_path)
    keys = []
    for spec in models:
        model = model_from_config(spec)
        micro = model.generate(Domain(size=size, periodic=True), derive_seed(3, 0))
        key, _ = session.folder()
        session.add(
            key,
            "morphology",
            model.name,
            micro=micro,
            seed=3,
            model=model.to_config(),
            domain={"size": size, "periodic": True},
            phases={str(k): v for k, v in model.active_phases().items()},
        )
        keys.append(key)
    return session, keys


def _run(session, name, params):
    job = ToolJob({"tool": name, "params": params}, session)
    job._run()  # synchronously
    status = job.status()
    assert status["state"] == "done", status.get("error")
    return status["result"]


def _column(table: dict, name: str) -> np.ndarray:
    k = table["columns"].index(name)
    return np.array([row[k] for row in table["rows"]], dtype=float)


def test_catalog_and_forms(tmp_path):
    catalog = tools.catalog()
    tab = next(t for t in catalog["tabs"] if t["name"] == "couplings")
    assert tab["title"] == "Couplings" and tab["order"] == 65
    assert [t["name"] for t in catalog["tools"] if t["tab"] == "couplings"] == [
        "diffusion",
        "oxidation",
        "oxidation_morphology",
        "hygromorph",
    ]
    hygro = next(t for t in catalog["tools"] if t["name"] == "hygromorph")
    fields = {f["name"]: f for f in hygro["fields"]}
    assert fields["bottom_morphology"]["kind"] == "item"
    assert fields["bottom_morphology"]["when"] == {"bottom": ["morphology"]}
    assert fields["middle"]["default"] == "none" and fields["middle_thickness"]["when"] == {
        "middle": ["constants", "morphology"]
    }
    session, (key,) = _session(tmp_path, SPHERES)
    parsed = tools.TOOLS["hygromorph"].parse({"top": "morphology", "top_morphology": key}, session)
    assert parsed["top_morphology"]["key"] == key and parsed["middle_thickness"] is None
    assert parsed["top_properties"] is None and parsed["bottom_properties"]["young"] == 2000.0
    with pytest.raises(ValueError, match="required"):
        tools.TOOLS["diffusion"].parse({}, session)
    job = ToolJob(
        {"tool": "diffusion", "params": {"source": key, "phases": {"0": {"D": 1}}}}, session
    )
    job._run()
    assert job.status()["state"] == "error" and "unknown ['D']" in job.status()["error"]


@pytest.mark.slow
def test_transient_diffusion(tmp_path):
    session, (key,) = _session(tmp_path, SPHERES)
    # one diffusivity and solubility everywhere: a slab exposed on both faces (Crank)
    same = {"0": {"diffusivity": 1.0}, "1": {"diffusivity": 1.0}}
    out = _run(
        session, "diffusion", {"source": key, "resolution": 12, "phases": same, "video": False}
    )
    v = out["values"]
    assert v["early slope of log M against log t (Fick: 1/2)"] == pytest.approx(0.5, abs=0.07)
    assert v["effective diffusivity of the uptake (Crank)"] == pytest.approx(1.0, rel=0.05)
    assert v["steady effective diffusivity along x"] == pytest.approx(1.0, rel=1e-6)
    assert v["final fraction (M - M0) / (M∞ - M0)"] > 0.99
    # exposed on one face: half of a slab twice as thick
    out = _run(
        session,
        "diffusion",
        {
            "source": key,
            "resolution": 8,
            "phases": same,
            "exposed": ["x-"],
            "homogenize": False,
            "video": False,
        },
    )
    assert out["values"]["effective diffusivity of the uptake (Crank)"] == pytest.approx(
        1.0, rel=0.05
    )
    # a first-order consumption: the steady uptake of Thiele's slab, tanh(φ)/φ, φ = (L/2) √(k/D)
    react = {k: {"diffusivity": 1.0, "consumption": 1.0} for k in ("0", "1")}
    out = _run(
        session,
        "diffusion",
        {"source": key, "resolution": 8, "phases": react, "homogenize": False, "video": False},
    )
    v = out["values"]
    assert "effective diffusivity of the uptake (Crank)" not in v
    assert v["final fraction (M - M0) / (M∞ - M0)"] == pytest.approx(math.tanh(2.0) / 2.0, rel=0.05)
    # slower, more soluble spheres: the effective diffusivities within the bounds, and at
    # equilibrium the concentration of each phase its solubility (the activity is 1)
    out = _run(session, "diffusion", {"source": key, "resolution": 8})
    v = out["values"]
    low, high = v["diffusivity, Reuss bound"], v["diffusivity, Voigt bound <D S> / <S>"]
    assert low < v["steady effective diffusivity along x"] < high
    assert low < v["effective diffusivity of the uptake (Crank)"] < high
    uptake = out["tables"][0]
    assert _column(uptake, "c_0")[-1] == pytest.approx(1.0, rel=0.01)
    assert _column(uptake, "c_1")[-1] == pytest.approx(3.0, rel=0.01)
    assert np.all(np.diff(_column(uptake, "fraction")) > 0)
    assert [f["title"] for f in out["figures"]] == [
        "uptake curve",
        "concentration in the middle section (x-y) at the times",
    ]
    # no video unless asked: the fields of the times, in 3D step by step and in the export
    assert out["videos"] == [] and out["fields3d"]["field"] == "concentration"
    names = {f["name"] for f in out["files"]}
    assert {"uptake.csv", "uptake.png", "fields.png", "diffusion_steps.json",
            "diffusion_step001.vti", "sample_0000.vti"} <= names  # fmt: skip
    shown = api.surface({"key": out["key"], "step": 2, "field": "activity"}, session)
    assert shown["step"] == 2 and len(shown["steps"]) == len(_column(uptake, "time"))
    export = api.export({"source": out["key"], "outputs": [], "video": "run", "fps": 4}, session)
    assert len(export["videos"]) == 1
    asked = _run(session, "diffusion", {"source": key, "resolution": 8, "video": True,
                                        "homogenize": False})  # fmt: skip
    assert asked["videos"][0]["title"] == "concentration along the uptake"


def test_thermal_oxidation(tmp_path):
    session, _ = _session(tmp_path)
    law = {"tracer": [0.0, 1.0], "young": [2400.0, 3200.0], "expansion": [0.0, -0.01]}
    out = _run(session, "oxidation", {"times": [111.0, 240.0, 800.0], "nodes": 41, "law": law})
    layers = out["tables"][0]
    hours = _column(layers, "time_h")
    layer = _column(layers, "oxidized_layer_um")
    surface, core = _column(layers, "amide_surface"), _column(layers, "amide_core")
    assert hours.tolist() == [111.0, 240.0, 800.0]
    # Ernault et al. (2019): an oxidized layer of about 200 µm, the core hardly oxidized
    assert np.all((layer[1:] > 50.0) & (layer[1:] < 300.0))
    assert np.all(core < 0.05 * surface) and np.all(np.diff(surface) > 0)
    assert np.all(surface <= _column(layers, "amide_thin_film"))  # oxygen is scarcer inside
    v = out["values"]
    assert v["final young at the surface"] > v["final young in the core"] > 2400.0 - 1e-9
    assert v["final expansion at the surface"] < v["final expansion in the core"] <= 0.0
    assert [f["title"] for f in out["figures"]] == [
        "profiles through the thickness",
        "kinetics and the oxidized layer",
        "properties of the ageing law through the thickness",
    ]
    assert out["tables"][1]["count"] == 40  # the nodes below the surface


def test_thermal_oxidation_of_a_morphology(tmp_path):
    """The oxidation of a matrix around impermeable spheres, in its volume: the oxygen
    goes round them and the oxidized layer is about that of a plate of the polymer (two
    voxels of this coarse image, which the form reports); with the spheres inert and
    permeable, the matrix oxidizes more."""
    session, (key,) = _session(tmp_path, SPHERES)
    params = {"source": key, "resolution": 12, "edge": 0.6e-3, "times": [111.0, 400.0]}
    out = _run(session, "oxidation_morphology", params)
    table = out["tables"][0]
    mean, layer = _column(table, "amide_mean"), _column(table, "oxidized_layer_um")
    plate = _column(table, "oxidized_layer_plate_um")
    assert np.all(np.diff(mean) > 0) and np.all(mean < _column(table, "amide_max"))
    assert np.all((layer > 0.5 * plate) & (layer < 1.5 * plate))
    v = out["values"]
    assert v["voxels"] == 12**3 and 0.5 < v["fraction of the phases that oxidize"] < 0.9
    assert v["final oxidized layer (µm)"] == pytest.approx(layer[-1])
    across = v["voxels across the final oxidized layer (four at least resolve it)"]
    assert across == pytest.approx(layer[-1] / 50.0)
    assert [f["title"] for f in out["figures"]] == [
        "mean oxidation profiles from the exposed face",
        "amides in the middle section (x-y) at the times",
    ]
    assert out["tables"][1]["columns"] == ["depth_um", "amide_111h", "amide_400h"]
    through = _run(session, "oxidation_morphology", {**params, "permeable": [1]})
    assert _column(through["tables"][0], "amide_mean")[-1] > mean[-1]
    for bad, message in (
        ({"oxidizing": [5]}, "one at least among the phases"),
        ({"permeable": [0]}, "not both"),
        ({"constants": {"k9": 1.0}}, "unknown"),
        ({"resolution": 100}, "from 4 to 64"),
    ):
        job = ToolJob({"tool": "oxidation_morphology", "params": {**params, **bad}}, session)
        job._run()
        assert job.status()["state"] == "error" and message in job.status()["error"]


def test_hygromorph_multilayers(tmp_path):
    session, (key,) = _session(tmp_path, FIBRES, size=8.0)
    # the default: an active layer swelling more along y under a passive one bends
    # towards the passive one (centre of curvature above: negative), more along y
    out = _run(session, "hygromorph", {})
    v = out["values"]
    assert v["final κy"] < v["final κx"] < 0.0 and v["final κxy (twist)"] == 0.0
    history = next(t for t in out["tables"] if t["title"].startswith("curvatures"))
    assert np.all(np.diff(_column(history, "kappa_y")) < 0)  # bending as the moisture enters
    assert v["final κx"] == pytest.approx(v["equilibrium κx"], rel=1e-3)
    # an isotropic bilayer whose top layer swells: Timoshenko's curvature, positive (the
    # plate bends towards its lower face, the layer that swells less)
    passive = {"young": 3000.0, "poisson": 0.3, "diffusivity": 1e-12, "solubility": 1.0}
    active = {**passive, "young": 1000.0, "beta_x": 0.01, "beta_y": 0.01}
    out = _run(
        session,
        "hygromorph",
        {"bottom_properties": passive, "top_properties": active, "exposed": "both"},
    )
    v = out["values"]
    assert v["equilibrium κx"] > 0.0
    assert v["equilibrium κx"] == pytest.approx(
        v["Timoshenko's curvature of the bilayer"], rel=1e-9
    )
    assert v["equilibrium κy"] == pytest.approx(v["equilibrium κx"], rel=1e-9)
    # a bottom layer that shrinks as it takes moisture: the plate bends towards it
    shrinking = {**passive, "beta_x": -0.01, "beta_y": -0.01}
    out = _run(session, "hygromorph", {"bottom_properties": shrinking, "top_properties": passive})
    assert out["values"]["final κx"] > 0.0 and out["values"]["final κy"] > 0.0
    # a swelling layer of fibres aligned along x, homogenized from the morphology: the
    # fibres restrain its swelling along them, and it bends across them
    out = _run(
        session, "hygromorph", {"bottom": "morphology", "bottom_morphology": key, "resolution": 12}
    )
    v = out["values"]
    assert abs(v["final κy"]) > 2.0 * abs(v["final κx"])
    layers = out["tables"][0]
    assert layers["rows"][0][layers["columns"].index("origin")].startswith("homogenized from")
    # its plane-stress stiffness Q and compliance S = Q⁻¹ (xx, yy, xy = s)
    pairs = (("xx", "xy", "xs"), ("xy", "yy", "ys"), ("xs", "ys", "ss"))
    q, s = (np.array([[_column(layers, f"{x}_{ab}")[0] for ab in row] for row in pairs])
            for x in "QS")  # fmt: skip
    assert np.allclose(s @ q, np.eye(3), atol=1e-4) and "S_ij = Q⁻¹" in layers["title"]
    beta_x, beta_y = _column(layers, "beta_x")[0], _column(layers, "beta_y")[0]
    assert 0.0 < beta_x < 0.5 * beta_y
    assert [f["title"] for f in out["figures"]] == [
        "curvatures and moisture profiles",
        "a strip along x bending",
    ]
