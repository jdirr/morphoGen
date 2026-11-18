"""The tools of the tab Physics of the interface, run on small voxel images: the transport
properties of a morphology between their bounds (and exactly for layers), the flow
through its pores with the acoustic parameters and the absorption of a layer, and the
dynamic moduli of a viscoelastic composite between the moduli of its limits."""

import numpy as np
import pytest

from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.tools import ToolJob
from morphogen.models.base import model_from_config
from morphogen.seeding import derive_seed

SPHERES = {"type": "boolean_spheres", "radius": 0.6, "volume_fraction": 0.3}


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


def _fails(session, name, params, match):
    job = ToolJob({"tool": name, "params": params}, session)
    job._run()
    status = job.status()
    assert status["state"] == "error" and match in status["error"], status


def _table_of(result: dict, title: str) -> list[dict]:
    table = next(t for t in result["tables"] if t["title"].startswith(title))
    return [dict(zip(table["columns"], row, strict=True)) for row in table["rows"]]


def _column(table: dict, name: str) -> np.ndarray:
    k = table["columns"].index(name)
    return np.array([row[k] for row in table["rows"]], dtype=float)


def test_catalog_and_forms():
    catalog = tools.catalog()
    tab = next(t for t in catalog["tabs"] if t["name"] == "physics")
    assert tab["title"] == "Physics" and tab["order"] == 62 and tab["section"] == "simulate"
    assert [t["name"] for t in catalog["tools"] if t["tab"] == "physics"] == [
        "transport",
        "sound",
        "dynamic",
        "waves",
    ]
    sound = next(t for t in catalog["tools"] if t["name"] == "sound")
    assert {f["name"] for f in sound["fields"]} >= {"source", "pores", "unit", "thickness"}


def test_transport_properties_between_their_bounds(tmp_path):
    session, (key,) = _session(tmp_path, SPHERES)
    res = _run(session, "transport", {"source": key, "physics": "electrical",
                                      "coefficients": [1.0, 10.0], "voxels": 16})  # fmt: skip
    values = res["values"]
    apparent = values["apparent electrical conductivity"]
    assert (
        values["Hashin-Shtrikman lower bound"] < apparent < values["Hashin-Shtrikman upper bound"]
    )
    assert values["Wiener lower bound (series)"] < values["Hashin-Shtrikman lower bound"]
    # (the scalar is that of one loading: close to the mean of the diagonal of the tensor)
    assert apparent == pytest.approx(np.mean([values[f"along {a}"] for a in "xyz"]), rel=0.05)
    assert values["fraction of phase 0"] + values["fraction of phase 1"] == pytest.approx(1.0)
    (table,) = res["tables"]
    assert table["title"].startswith("electrical conductivity tensor") and table["count"] == 3
    assert _column(table, "1")[0] == pytest.approx(values["along x"])
    # an inert phase: the diffusivity of the matrix over its tortuosity
    res = _run(session, "transport", {"source": key, "physics": "diffusion",
                                      "coefficients": [2.0, 0.0], "voxels": 16})  # fmt: skip
    values = res["values"]
    assert 0.0 < values["apparent diffusivity"] < 2.0 * values["fraction of phase 0"]
    _fails(session, "transport", {"source": key, "coefficients": [1.0], "voxels": 16},
           "one per phase")  # fmt: skip
    _fails(session, "transport", {"source": key, "coefficients": [0.0, 0.0], "voxels": 16},
           "positive")  # fmt: skip
    _fails(session, "transport", {"source": key, "coefficients": [1.0, 2.0], "voxels": 2},
           "voxels per edge")  # fmt: skip


def test_flow_and_sound_absorption(tmp_path):
    """Spheres of 0.6 mm in air: a permeability of the order of the squared radius, a
    tortuosity near Maxwell's 1 + c / 2, the viscous length below the thermal one, and a
    layer that absorbs more at high frequencies."""
    session, (key,) = _session(tmp_path, SPHERES)
    res = _run(session, "sound", {"source": key, "pores": 0, "voxels": 24, "unit": 1e-3,
                                  "thickness": 0.05, "frequencies": [200.0, 5000.0]})  # fmt: skip
    values = res["values"]
    fraction = 1.0 - values["porosity"]
    assert 0.2 < fraction < 0.4
    assert 1e-9 < values["permeability (m²)"] < (0.6e-3) ** 2
    assert values["resistivity (Pa s/m²)"] == pytest.approx(1.81e-5 / values["permeability (m²)"])
    assert values["tortuosity"] == pytest.approx(1 + fraction / 2, rel=0.08)
    assert 0.0 < values["viscous length (m)"] < values["thermal length (m)"] < 2e-3
    assert values["thermal permeability (m²)"] > values["permeability (m²)"]
    parameters, curve = res["tables"]
    assert parameters["count"] == 3 and curve["count"] == 60
    alpha = _column(curve, "absorption")
    assert np.all((alpha > 0.0) & (alpha < 1.0)) and alpha[:5].mean() < alpha[-5:].mean()
    assert values["highest absorption"] == pytest.approx(
        max(alpha.max(), values["highest absorption"])
    )
    assert [f["title"] for f in res["figures"]] == ["sound absorption of the layer"]
    # at an angle and in a diffuse field (along x: the frame along the layer that of y, z)
    common = {"source": key, "pores": 0, "voxels": 16, "unit": 1e-3, "thickness": 0.05,
              "frequencies": [200.0, 5000.0], "direction": "x"}  # fmt: skip
    curves = {}
    for kind in ("normal", "oblique", "diffuse"):
        out = _run(session, "sound", {**common, "incidence": kind, "angle": 60.0})
        curve = out["tables"][1]
        curves[kind] = _column(curve, "absorption")
        assert _column(curve, "absorption at normal incidence") == pytest.approx(curves["normal"])
        assert np.all((curves[kind] > 0.0) & (curves[kind] < 1.0))
    assert "incidence at 60°" in out["tables"][1]["title"] or "diffuse field" in curve["title"]
    assert not np.allclose(curves["oblique"], curves["normal"], rtol=0.01)
    assert not np.allclose(curves["diffuse"], curves["oblique"], rtol=0.01)
    _fails(session, "sound", {**common, "incidence": "oblique", "angle": 95.0}, "from 0 to 90")
    _fails(session, "sound", {"source": key, "pores": 7, "voxels": 16}, "no voxel of the phase")
    _fails(session, "sound", {"source": key, "voxels": 16, "frequencies": [100.0]}, "range")
    _fails(session, "sound", {"source": key, "voxels": 16, "thickness": 0.0}, "thickness")


def test_dynamic_moduli_of_a_viscoelastic_composite(tmp_path):
    """A viscoelastic matrix with stiff spheres: the storage modulus rises with the
    frequency between its relaxed and instantaneous values, the loss peaks between the
    two times of the Prony series."""
    session, (key,) = _session(tmp_path, SPHERES)
    materials = {"0": {"young": 3.0, "poisson": 0.35}, "1": {"young": 70.0, "poisson": 0.22}}
    res = _run(session, "dynamic", {"source": key, "materials": materials, "voxels": 10,
                                    "shear": [0.5, 0.01, 0.3, 1.0], "frequencies": [1e-3, 1e4],
                                    "count": 6})  # fmt: skip
    (table,) = res["tables"]
    storage, loss = _column(table, "shear storage"), _column(table, "shear loss")
    tan = _column(table, "tan δ (shear)")
    assert table["count"] == 6 and np.all(np.diff(storage) > 0.0) and np.all(loss > 0.0)
    assert np.allclose(tan, loss / storage, rtol=1e-4)  # (the cells of a table are rounded)
    assert tan[0] < tan.max() > tan[-1]  # a peak inside the range
    values = res["values"]
    assert values["highest loss factor (shear)"] == pytest.approx(tan.max(), rel=1e-4)
    assert values["shear storage at the lowest frequency"] == pytest.approx(storage[0], rel=1e-4)
    # above the relaxed matrix (20 % of its modulus left), below the bound of Voigt of the
    # instantaneous moduli
    matrix, spheres = 3.0 / (2 * 1.35), 70.0 / (2 * 1.22)
    assert 0.2 * matrix < storage[0] < storage[-1] < 0.7 * matrix + 0.3 * spheres * 1.5
    assert [f["title"] for f in res["figures"]] == ["shear moduli and loss factor of the composite"]
    for change, match in (
        ({"shear": [0.5, 0.01, 0.3]}, "pairs"),
        ({"shear": [], "bulk": []}, "at least one term"),
        ({"phase": 5}, "viscoelastic phase"),
        ({"count": 1}, "number of frequencies"),
        ({"frequencies": [10.0, 1.0, 5.0]}, "range of angular frequencies"),
    ):
        params = {"source": key, "materials": materials, "voxels": 8, **change}
        _fails(session, "dynamic", params, match)


def test_band_structure_of_light_and_sound(tmp_path):
    """Light in a cubic lattice of dielectric spheres: at long wavelengths the first two
    bands (its two polarizations) travel at the speed of the homogenized permittivity,
    between the bounds of Wiener; sound among rigid spheres: slowed by their tortuosity."""
    lattice = {"type": "strut_lattice", "cell": "cubic", "cell_size": 4.0, "volume_fraction": 0.2}
    session, (key,) = _session(tmp_path, lattice)
    light = {"source": key, "permittivity": [1.0, 9.0], "resolution": 8, "coarsen": 2,
             "path": "G X", "points": 4, "bands": 4}  # fmt: skip
    out = _run(session, "waves", light)
    values = out["values"]
    assert values["voxels"] == 16**3 and values["wave vectors"] == 5
    bands = _table_of(out, "the bands at the wave vectors")
    first = bands[1]  # a quarter of the way to X: k L / 2π = 1/8
    assert first["band 1"] == pytest.approx(first["band 2"], rel=0.02)  # two polarizations
    index = 0.125 / first["band 1"]  # the effective index: c k / ω
    fraction = 0.2
    wiener = (1.0 / np.sqrt((1 - fraction) + fraction / 9.0), np.sqrt(1 - fraction + 9 * fraction))
    assert wiener[0] ** -1 > 0 and 1.0 / wiener[0] < index < wiener[1]
    assert bands[0]["band 1"] == pytest.approx(0.0, abs=1e-3)  # the centre of the zone
    assert out["title"].startswith("Light in") and len(out["figures"]) == 1
    sound = _run(session, "waves", {"source": key, "waves": "acoustic", "density": [1.2, 0.0],
                                    "bulk": [1.42e5, 1.0], "resolution": 6, "path": "G X",
                                    "points": 4, "bands": 3})  # fmt: skip
    rows = _table_of(sound, "the bands at the wave vectors")
    slowness = 0.125 / rows[1]["band 1"]  # c0 k / ω: above 1 among rigid struts
    assert 1.0 < slowness < 1.3
    _fails(session, "waves", {"source": key, "permittivity": [1.0]}, "one per phase")
    _fails(session, "waves", {"source": key, "permittivity": [1.0, -2.0]}, "positive")
    _fails(session, "waves", {"source": key, "resolution": 64}, "from 4 to 32")
    _fails(session, "waves", {"source": key, "waves": "acoustic", "density": [0.0, 0.0],
                              "bulk": [1.0, 1.0]}, "one fluid")  # fmt: skip
