"""The tools of the tab Mechanics: the loading path of a morphology (FFT on its voxels,
finite elements on its conforming mesh), the reduced model of the NTFA against the full
field, the material point, the linear tensors, and the stiffness of the frame of beams of
the networks (with the frame of the summaries of Generate)."""

import csv
import importlib.util
import json

import numpy as np
import pytest
import yaml

from morphogen.config import MeshSpec
from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.tools import ToolJob, mechanics
from morphogen.homogenization import Material, NTFAModel, homogenize
from morphogen.mesh.conforming import conforming_mesh, has_mmg
from morphogen.models.base import model_from_config
from morphogen.seeding import derive_seed
from morphogen.viz.video import vtk_can_render

E, NU = 1.0, 0.3
R0 = 0.002


def _workspace(tmp_path):
    """A session whose workspace holds a Boolean model of spheres (without the views)."""
    session = api.Session(tmp_path)
    model = model_from_config({"type": "boolean_spheres", "radius": 0.6, "volume_fraction": 0.2})
    micro = model.generate(Domain(size=4.0, periodic=True), derive_seed(7, 0))
    key, _ = session.folder()
    session.add(
        key,
        "morphology",
        "boolean_spheres (seed 7)",
        micro=micro,
        seed=7,
        model=model.to_config(),
        domain={"size": 4.0, "periodic": True},
        phases={"0": "matrix", "1": "spheres"},
    )
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


def _young(stiffness) -> float:
    """The apparent Young modulus along 1 of a stiffness (the lateral stresses free)."""
    return 1.0 / np.linalg.inv(stiffness)[0, 0]


ELASTIC = {"0": {"young": E, "poisson": NU}, "1": {"young": 10 * E, "poisson": NU}}
POROUS = {"0": {"young": E, "poisson": NU}, "1": "void"}
PERFECT = {"behaviour": "mises", "yield_stress": R0, "hardening": 0.0, "isotropic": "0, 1"}


def test_catalog():
    catalog = tools.catalog()
    tab = next(t for t in catalog["tabs"] if t["name"] == "mechanics")
    assert tab["order"] == 40
    names = [t["name"] for t in catalog["tools"] if t["tab"] == "mechanics"]
    assert names == ["path", "macroscopic_law", "ntfa", "point", "identify", "tensors", "frame"]
    path = next(t for t in catalog["tools"] if t["name"] == "path")
    fields = {f["name"]: f for f in path["fields"]}
    assert fields["materials"]["of"] == "source" and fields["source"]["required"]
    assert fields["kinematic"]["when"] == {"behaviour": ["mises", "porous"]}
    assert fields["tau0"]["when"] == {"behaviour": ["crystal"]}
    assert fields["gurson"]["when"] == {"behaviour": ["porous"], "criterion": ["gurson"]}
    assert fields["voxels"]["when"] == {"method": ["fft"]}
    assert fields["path"]["when"] == {"loading": ["custom"]}
    behaviours = [c[0] for c in fields["behaviour"]["choices"]]
    assert behaviours[:5] == ["elastic", "mises", "crystal", "porous", "zmat"]
    assert behaviours[5:9] == ["viscoelastic", "hyperelastic", "damage", "mazars"]
    assert fields["lemaitre"]["when"] == {"behaviour": ["mises"], "coupled_damage": ["lemaitre"]}
    assert fields["kachanov"]["when"] == {"behaviour": ["mises"], "coupled_damage": ["kachanov"]}
    assert fields["mazars_tension"]["when"] == {"behaviour": ["mazars"]}
    assert [c[0] for c in fields["damage_split"]["choices"]] == ["none", "volumetric", "spectral"]
    assert fields["prony_shear"]["when"] == {"behaviour": ["viscoelastic"]}
    assert fields["hyper_model"]["when"] == {"behaviour": ["hyperelastic"]}
    assert fields["fracture_energy"]["when"] == {"behaviour": ["damage"]}
    assert ("mfront" in behaviours) == (importlib.util.find_spec("mgis") is not None)
    # hexagonal crystals, twins and cleavage planes
    families = [c[0] for c in fields["families"]["choices"]]
    assert families[:4] == ["octahedral", "cube", "bcc110", "bcc112"] and "basal" in families
    assert [c[0] for c in fields["twins"]["choices"]][:3] == ["twin111", "twin112", "twin1012"]
    assert fields["twins"]["when"] == fields["c_over_a"]["when"] == {"behaviour": ["crystal"]}
    planes = ["cube", "dodecahedral", "octahedral", "basal", "prismatic"]
    assert [c[0] for c in fields["cleavage"]["choices"]] == ["none", *planes]
    assert fields["cleavage_energy"]["when"] == {"behaviour": ["crystal"], "cleavage": planes}
    assert "twin_fraction" in [c[0] for c in fields["field"]["choices"]]
    # the criterion of cleavage of Beremin, in its box
    assert fields["beremin"]["group"] == "Cleavage" and fields["beremin"]["fold"]
    assert fields["beremin_scale"]["when"] == {"beremin": [True]}
    assert [c[0] for c in fields["beremin_stress"]["choices"]] == ["principal", *planes]
    # the interfaces that debond (finite elements) and the fatigue of the cycle, in boxes
    assert fields["interfaces"]["when"] == {"method": ["fe"]}
    assert fields["interface_energy"]["when"] == {"method": ["fe"], "interfaces": [True]}
    assert fields["interfaces"]["group"] == "Interfaces" and fields["interfaces"]["fold"]
    criteria = [c[0] for c in fields["fatigue"]["choices"]]
    assert criteria == ["none", "crossland", "sines", "dang_van", "findley", "matake"]
    assert fields["fatigue_repeated"]["when"] == {"fatigue": ["sines"]}
    assert fields["fatigue_cycle"]["group"] == "Fatigue"
    assert "interface_damage" in [c[0] for c in fields["field"]["choices"]]
    ntfa = next(t for t in catalog["tools"] if t["name"] == "ntfa")
    assert {"training", "modes", "test_loading", "compare"} <= {f["name"] for f in ntfa["fields"]}


def test_material_point(tmp_path):
    session = api.Session(tmp_path)
    tension = {"young": E, "poisson": NU, "loading": "tension", "amplitude": 0.01, "increments": 20}
    # elastic: the uniaxial stress E eps, the lateral contraction -nu eps
    out = _run(session, "point", {**tension, "behaviour": "elastic"})["values"]
    assert out["S11"] == pytest.approx(0.01 * E, rel=1e-10)
    assert out["E22"] == pytest.approx(-NU * 0.01, rel=1e-10) and abs(out["S22"]) < 1e-12
    # perfect plasticity: the stress saturates at the yield stress; Voce's at R0 + Q
    out = _run(session, "point", {**tension, **PERFECT})
    assert out["values"]["S11"] == pytest.approx(R0, rel=1e-8)
    assert out["values"]["p"] == pytest.approx(0.01 - R0 / E, rel=1e-6)
    curve = next(t for t in out["tables"] if t["title"].startswith("curves"))
    assert curve["count"] == 21 and {"E11", "S11", "p"} <= set(curve["columns"])
    voce = {
        **tension,
        "behaviour": "mises",
        "yield_stress": R0,
        "isotropic": "0.001, 1000",
        "amplitude": 0.05,
    }
    out = _run(session, "point", voce)["values"]
    assert out["S11"] == pytest.approx(R0 + 0.001, rel=1e-6)
    # a test along the same law: no gap; drawn on the figure
    s = np.linspace(0, 0.01, 6)
    points = "strain;stress\n" + "\n".join(
        f"{e:.6f}".replace(".", ",") + ";" + f"{min(E * e, R0):.8f}".replace(".", ",") for e in s
    )
    out = _run(session, "point", {**tension, **PERFECT, "reference": points})
    assert out["values"]["reference_rms_gap"] < 1e-12 + 1e-6 * R0
    assert [f["title"] for f in out["figures"]] == ["curves"]
    # a Z-set file of the same law as Voce's above
    zmat = mechanics.ZMAT_EXAMPLE.replace("b 100.", "b 1000.")
    out = _run(session, "point", {**tension, "behaviour": "zmat", "zmat": zmat, "amplitude": 0.05})[
        "values"
    ]
    assert out["S11"] == pytest.approx(R0 + 0.001, rel=1e-6)
    # a cycle with kinematic hardening (Prager): the Bauschinger effect
    cycle = {
        **tension,
        "loading": "cycle",
        "behaviour": "mises",
        "yield_stress": R0,
        "isotropic": "0, 1",
        "kinematic": "0.1, 0",
        "increments": 10,
    }
    out = _run(session, "point", cycle)
    rows = next(t for t in out["tables"] if t["title"].startswith("curves"))
    s11 = np.array([r[rows["columns"].index("S11")] for r in rows["rows"]], dtype=float)
    assert s11.max() > R0 and s11.min() < -R0  # hardened both ways


def _column(session, out, name: str) -> np.ndarray:
    """A column of the first table of a result (the curves), all its rows (its file)."""
    _, _, folder, file = out["tables"][0]["url"].split("/")
    with open(session.root / folder / file, encoding="utf-8") as f:
        return np.array([float(row[name]) for row in csv.DictReader(f)])


def test_viscoelastic_hyperelastic_and_damaged_points(tmp_path):
    """The behaviours in closed form of the forms: a Prony series relaxes to the modulus of
    its long times, a neo-Hookean bar follows its closed form when it is nearly
    incompressible, a damaged bar peaks at its strength."""
    session = api.Session(tmp_path)
    tension = {"young": E, "poisson": NU, "loading": "tension", "amplitude": 0.01, "increments": 20}
    # viscoelasticity: E_inf = 9 K G_inf / (3 K + G_inf) once the branch has relaxed
    visco = {**tension, "behaviour": "viscoelastic", "prony_shear": "0.5, 0.1",
             "loading": "relaxation", "hold": 10}  # fmt: skip
    out = _run(session, "point", visco)
    time, s11 = _column(session, out, "time"), _column(session, out, "S11")
    bulk, shear = E / (3 * (1 - 2 * NU)), E / (2 * (1 + NU))
    relaxed = 9 * bulk * 0.5 * shear / (3 * bulk + 0.5 * shear)
    assert s11[-1] == pytest.approx(relaxed * 0.01, rel=1e-6)
    assert s11[np.flatnonzero(np.isclose(time, 1.0))[0]] > 1.05 * s11[-1]
    assert "pair" in _error(session, "point", {**visco, "prony_shear": "0.5"})
    # hyperelasticity, nearly incompressible: the nominal stress mu (lambda - 1 / lambda^2)
    stretch = {**tension, "behaviour": "hyperelastic", "kinematics": "finite", "amplitude": 0.5}
    mu = 0.4
    rubber = {**stretch, "hyper_parameters": str(mu), "hyper_bulk": 1e4 * mu}
    out = _run(session, "point", rubber)["values"]
    assert out["S11"] == pytest.approx(mu * (1.5 - 1 / 1.5**2), rel=2e-3)
    assert out["LE22"] == pytest.approx(-0.5 * out["LE11"], rel=1e-3)
    # without parameters: the neo-Hookean energy of the elastic constants of the form
    soft = _run(session, "point", stretch)["values"]
    assert 0.0 < soft["S11"] < E * 0.5
    two = {**stretch, "hyper_model": "mooney_rivlin", "hyper_parameters": "0.15, 0.04"}
    assert _run(session, "point", two)["values"]["S11"] > 0.0
    assert "c10, c01" in _error(session, "point", {**two, "hyper_parameters": "0.15"})
    assert "mooney rivlin" in _error(session, "point", {**two, "hyper_parameters": ""})
    ogden = {**stretch, "hyper_model": "ogden", "hyper_parameters": "0.6, 1.3, 0.001, 5.0"}
    assert _run(session, "point", ogden)["values"]["S11"] > 0.0
    # the forms of five parameters and more of Mooney and Rivlin; the eight chains
    five = {**two, "hyper_parameters": "0.15, 0.04, 0.01, 0.002, 0.001"}
    stiffer = _run(session, "point", five)["values"]["S11"]
    assert stiffer > _run(session, "point", two)["values"]["S11"]
    chains = {**rubber, "hyper_model": "eight_chain", "hyper_parameters": f"{mu}, 3.0"}
    u = (1.5**2 + 2 / 1.5) / (3 * 3.0**2)  # the square of the stretch of the chains / lock^2
    locking = 2 * (1.5 - 1 / 1.5**2) * mu / 6 * (3 - u) / (1 - u)
    assert _run(session, "point", chains)["values"]["S11"] == pytest.approx(locking, rel=2e-3)
    # fibres along the tension (Holzapfel, Gasser and Ogden): the stress of the matrix
    # and 2 k1 l (l^2 - 1) exp(k2 (l^2 - 1)^2)
    fibres = {**stretch, "hyper_model": "holzapfel", "hyper_parameters": "0.4, 1.0, 0.5",
              "hyper_fibres": "1, 0, 0", "hyper_bulk": 1e4}  # fmt: skip
    expected = 0.4 * (1.5 - 1 / 1.5**2) + 2 * 1.5 * 1.25 * np.exp(0.5 * 1.25**2)
    assert _run(session, "point", fibres)["values"]["S11"] == pytest.approx(expected, rel=2e-3)
    assert "fibres" in _error(session, "point", {**fibres, "hyper_fibres": "1, 0"})
    # a compressible foam without lateral contraction (the bulk modulus 2/3 of the shear's)
    foam = {**stretch, "hyper_model": "hyperfoam", "hyper_parameters": "0.4, 2.0",
            "hyper_bulk": 0.4 * 2 / 3}  # fmt: skip
    crushed = _run(session, "point", foam)["values"]
    assert crushed["S11"] == pytest.approx(0.4 * (1.5 - 1 / 1.5), rel=1e-6)
    assert abs(crushed["LE22"]) < 1e-9
    # the softening of Mullins needs its three parameters
    assert "Mullins" in _error(session, "point", {**rubber, "hyper_mullins": "2, 0.3"})
    assert _run(session, "point", {**rubber, "hyper_mullins": "2, 0.3, 0.1"})["values"][
        "S11"
    ] == pytest.approx(out["S11"], rel=1e-9)  # no softening on the first loading
    # damage: elastic up to the strength, then the stress falls
    damage = {**tension, "behaviour": "damage", "strength": 0.002, "ultimate": 0.02}
    out = _run(session, "point", damage)
    s11, damaged = _column(session, out, "S11"), _column(session, out, "p")
    assert s11.max() == pytest.approx(0.002, rel=1e-6) and s11[-1] < 0.7 * s11.max()
    assert damaged[2] == 0.0 and 0.5 < damaged[-1] < 1.0
    band = {**damage, "ultimate": "", "fracture_energy": 1e-5, "damage_length": 0.5}
    assert _run(session, "point", band)["values"]["S11"] < 0.002
    assert "size of the elements" in _error(session, "point", {**band, "damage_length": ""})


def test_points_damaged_by_the_laws_of_lemaitre_kachanov_and_mazars(tmp_path):
    """The damage laws of the forms at a material point: Lemaitre's damage of a perfectly
    plastic bar, (R0² / (2 E S)) (p - p_D); Kachanov's under a held stress, 1 - (1 - t /
    t_R)^(1 / (k + 1)); Mazars' strength in tension, E κ0; the unilateral effect of the
    spectral split."""
    session = api.Session(tmp_path)
    tension = {"young": E, "poisson": NU, "loading": "tension", "amplitude": 0.05, "increments": 50}
    ductile = {**tension, "behaviour": "mises", "yield_stress": R0, "isotropic": "0, 1",
               "coupled_damage": "lemaitre", "lemaitre": "5e-7, 1",
               "damage_threshold": 0.01}  # fmt: skip
    out = _run(session, "point", ductile)
    damaged = _column(session, out, "p")
    plastic = 0.05 - R0 / E  # of the effective stress, perfectly plastic
    exact = R0**2 / (2 * E * 5e-7) * (plastic - 0.01)
    assert damaged[-1] == pytest.approx(exact, rel=1e-6) and 0.05 < exact < 0.9
    assert _column(session, out, "S11")[-1] == pytest.approx((1 - exact) * R0, rel=1e-6)
    broken = _run(session, "point", {**ductile, "critical_damage": 0.5 * exact})
    assert _column(session, broken, "p")[-1] == pytest.approx(1 - 1e-6)
    assert "strength S" in _error(session, "point", {**ductile, "lemaitre": "1, 2, 3"})
    # creep: the damage of Kachanov under a held stress
    stress, a, r = 0.4 * R0, R0, 4.0
    t_r = 1.0 / ((r + 1) * (stress / a) ** r)
    creep = {**tension, "behaviour": "mises", "yield_stress": 0.0, "isotropic": "0, 1",
             "norton": f"{2 * R0}, 5", "coupled_damage": "kachanov", "kachanov": f"{a}, {r}",
             "loading": "creep", "stress": stress, "hold": 0.5 * t_r,
             "increments": 200}  # fmt: skip
    out = _run(session, "point", creep)
    time, damaged = _column(session, out, "time"), _column(session, out, "p")
    # the stress rises to its value in a time 1, which counts for 1 / (r + 1) of it held
    held = time[-1] - 1.0 + 1.0 / (r + 1)
    assert damaged[-1] == pytest.approx(1 - (1 - held / t_r) ** (1 / (r + 1)), rel=0.02)
    assert "viscoplastic" in _error(session, "point", {**creep, "norton": ""})
    assert "α and β" in _error(session, "point", {**creep, "hayhurst": "0.2"})
    # concrete: the strength of Mazars' law in tension, E κ0 (A_t = 1)
    concrete = {**tension, "behaviour": "mazars", "mazars_threshold": 1e-4, "amplitude": 5e-4,
                "increments": 100}  # fmt: skip
    out = _run(session, "point", concrete)
    assert _column(session, out, "S11").max() == pytest.approx(E * 1e-4, rel=0.02)
    assert 0.9 < _column(session, out, "p")[-1] < 1.0
    assert "A_t, B_t" in _error(session, "point", {**concrete, "mazars_tension": "1"})
    # the split of the quasi-brittle damage is that of the form
    material = tools.TOOLS["point"].parse(
        {**tension, "behaviour": "damage", "strength": 0.002, "ultimate": 0.02,
         "damage_split": "spectral"}, session)  # fmt: skip
    from morphogen.gui.tools.mechanics import _behaviour

    assert _behaviour(material, {"young": E, "poisson": NU}, tmp_path)["damage"]["split"] == (
        "spectral"
    )


def test_material_point_twins_and_cleavage(tmp_path):
    """The crystals of the form twin and cleave: magnesium pulled along its axis c twins
    (the strain of its twins is their fraction times their shear and their Schmid
    factor) and does not when compressed; iron pulled along [001] cleaves on (001)."""
    from morphogen.homogenization.crystal import twin_systems

    session = api.Session(tmp_path)
    magnesium = {
        "young": 45000.0,
        "poisson": 0.3,
        "behaviour": "crystal",
        "hexagonal": "59400, 25600, 21400, 61600, 16400",
        "c_over_a": 1.624,
        "families": ["basal"],
        "tau0": 10.0,
        "slip_isotropic": "0, 1",
        "slip_viscosity": "2, 4",
        "twins": ["twin1012"],
        "twin_tau0": 30.0,
        "twin_saturation": 0.9,
        "loading": "tension",
        "component": "33",
        "amplitude": 0.04,
        "increments": 40,
    }
    out = _run(session, "point", magnesium)["values"]
    normals, directions, shear = twin_systems("twin1012", 1.624)
    schmid = normals[0, 2] * directions[0, 2]
    young = (
        1.0
        / np.linalg.inv(Material.hexagonal(59400, 25600, 21400, 61600, 16400).matrix("elastic"))[
            2, 2
        ]
    )
    assert 0.5 < out["twin_fraction"] < 0.9
    assert out["E33"] - out["S33"] / young == pytest.approx(
        schmid * shear * out["twin_fraction"], rel=1e-6
    )
    compressed = {**magnesium, "loading": "custom", "path": "t=0,1 n=10 E33=0,-0.01"}
    out = _run(session, "point", compressed)["values"]
    assert out["twin_fraction"] == 0.0 and out["S33"] == pytest.approx(-0.01 * young, rel=1e-8)
    iron = {
        "young": 200000.0,
        "poisson": 0.3,
        "behaviour": "crystal",
        "cubic": "230000, 135000, 117000",
        "families": ["bcc110"],
        "tau0": 300.0,
        "slip_isotropic": "0, 1",
        "slip_viscosity": "10, 10",
        "cleavage": "cube",
        "cleavage_strength": 500.0,
        "cleavage_energy": 5.0,
        "cleavage_length": 1.0,
        "cleavage_viscosity": "0.5, 1",
        "loading": "tension",
        "component": "33",
        "amplitude": 0.06,
        "increments": 100,
    }
    out = _run(session, "point", iron)
    values = out["values"]
    assert values["cleavage_damage"] > 0.99 and values["S33"] < 0.01 * 500.0 and values["p"] == 0
    curve = next(t for t in out["tables"] if t["title"].startswith("curves"))
    assert {"cleavage_opening", "cleavage_damage"} <= set(curve["columns"])
    # errors of the form
    assert "one lattice" in _error(session, "point", {**magnesium, "twins": ["twin111"]})
    assert "c_over_a" in _error(session, "point", {**magnesium, "c_over_a": None})
    assert "not both" in _error(session, "point", {**magnesium, "cubic": "1, 1, 1"})
    assert "C11, C12, C13, C33, C44" in _error(session, "point", {**magnesium, "hexagonal": "1"})
    assert "cleavage strength" in _error(session, "point", {**iron, "cleavage_strength": None})
    assert "fracture energy of the planes" in _error(
        session, "point", {**iron, "cleavage_length": None}
    )
    assert "softens faster" in _error(session, "point", {**iron, "cleavage_energy": 1.0})
    # planes with a damage band (regularized by their microdamage): a point alone has no
    # gradient, and softens linearly with its damage, which is its microdamage
    banded = {**iron, "cleavage_length": None, "cleavage_band": 2.0}
    out = _run(session, "point", banded)
    assert out["values"]["cleavage_damage"] == 1.0 and out["values"]["S33"] < 0.01 * 500.0
    assert out["values"]["cleavage_microdamage"] == out["values"]["cleavage_opening"]
    assert "width of the damage band" in _error(session, "point", {**iron, "cleavage_length": None})


def test_material_point_twins_turn_and_cracks_close(tmp_path):
    """The crystals of the form under a stress that is reversed: the twins of magnesium
    shear back to nothing (or stay, without detwinning; at a share of their resistance)
    and turn its lattice at a threshold of their fraction (grain by grain: the same for
    a point); the cleavage plane of iron closes under pressure and carries it (or stays
    open, without closure; at a share of its strength)."""
    from morphogen.gui.tools.mechanics import _behaviour

    session = api.Session(tmp_path)
    magnesium = {
        "young": 45000.0,
        "poisson": 0.3,
        "behaviour": "crystal",
        "hexagonal": "59400, 25600, 21400, 61600, 16400",
        "c_over_a": 1.624,
        "families": ["basal"],
        "tau0": 10.0,
        "slip_isotropic": "0, 1",
        "slip_viscosity": "2, 4",
        "twins": ["twin1012"],
        "twin_tau0": 30.0,
        "loading": "custom",
        "path": "t=0,1,2 n=30,40 E33=0,0.03,-0.003",
    }
    young = (
        1.0
        / np.linalg.inv(Material.hexagonal(59400, 25600, 21400, 61600, 16400).matrix("elastic"))[
            2, 2
        ]
    )
    out = _run(session, "point", magnesium)
    values = out["values"]
    curve = next(t for t in out["tables"] if t["title"].startswith("curves"))
    column = curve["columns"].index("twin_fraction")
    twinned = max(row[column] for row in curve["rows"])
    assert 0.4 < twinned < 0.5 and "twin_misorientation" not in curve["columns"]
    # pushed back: no twin left, the crystal as it was
    assert values["twin_fraction"] == 0.0
    assert values["S33"] == pytest.approx(-0.003 * young, rel=1e-6)
    kept = _run(session, "point", {**magnesium, "twin_detwinning": False})["values"]
    assert kept["twin_fraction"] == pytest.approx(twinned, abs=1e-6)
    assert kept["S33"] < 3 * values["S33"]
    eased = tools.TOOLS["point"].parse({**magnesium, "twin_detwinning_share": 0.5}, session)
    spec = _behaviour(eased, {"young": 45000.0, "poisson": 0.3}, tmp_path)
    assert spec["crystal_plasticity"]["twin_detwinning"] == 0.5
    turning = {**magnesium, "twin_reorientation": 0.3, "path": "t=0,1 n=30 E33=0,0.03"}
    out = _run(session, "point", turning)
    # the lattice of the twin; the twinned fraction is that of the shears of the twins
    assert out["values"]["twin_reoriented"] == 1.0
    assert 0.3 <= out["values"]["twin_fraction"] < 0.33
    assert out["values"]["twin_misorientation"] == pytest.approx(86.31, abs=0.01)
    form = tools.TOOLS["point"].parse(turning, session)
    spec = _behaviour(form, {"young": 45000.0, "poisson": 0.3}, tmp_path)
    assert spec["crystal_plasticity"]["twin_reorientation"] == 0.3
    assert "twin_detwinning" not in spec["crystal_plasticity"]
    assert "]0, twin_saturation" in _error(session, "point", {**turning, "twin_reorientation": 1.5})
    # grain by grain: a point is a grain of one point
    by_grain = {**turning, "twin_reorientation_growth": 0.25}
    same = _run(session, "point", by_grain)["values"]
    assert same["S33"] == pytest.approx(out["values"]["S33"], rel=1e-6)
    form = tools.TOOLS["point"].parse(by_grain, session)
    spec = _behaviour(form, {"young": 45000.0, "poisson": 0.3}, tmp_path)
    assert spec["crystal_plasticity"]["twin_reorientation"] == {"threshold": 0.3, "growth": 0.25}
    assert "give that fraction" in _error(
        session, "point", {**magnesium, "twin_reorientation_growth": 0.25}
    )
    iron = {
        "young": 200000.0,
        "poisson": 0.3,
        "behaviour": "crystal",
        "cubic": "230000, 135000, 117000",
        "families": ["bcc110"],
        "tau0": 300.0,
        "slip_isotropic": "0, 1",
        "slip_viscosity": "10, 10",
        "cleavage": "cube",
        "cleavage_strength": 500.0,
        "cleavage_energy": 5.0,
        "cleavage_length": 1.0,
        "cleavage_viscosity": "0.5, 1",
        "loading": "custom",
        "path": "t=0,1,2 n=100,40 E33=0,0.06,-0.002",
    }
    stiff = 1.0 / np.linalg.inv(Material.cubic(230000, 135000, 117000).matrix("elastic"))[2, 2]
    values = _run(session, "point", iron)["values"]
    assert values["cleavage_damage"] > 0.99 and values["cleavage_opening"] == 0.0
    assert values["S33"] == pytest.approx(-0.002 * stiff, rel=1e-6)
    open_ = _run(session, "point", {**iron, "cleavage_closure": False})["values"]
    # (the lattice alone is pressed, and slips)
    assert open_["cleavage_opening"] > 0.05 and open_["S33"] < 2.5 * values["S33"]
    rubbing = tools.TOOLS["point"].parse({**iron, "cleavage_friction": 0.4}, session)
    spec = _behaviour(rubbing, {"young": 200000.0, "poisson": 0.3}, tmp_path)
    assert spec["crystal_plasticity"]["cleavage"]["friction"] == 0.4
    assert "closure" not in spec["crystal_plasticity"]["cleavage"]
    eased = tools.TOOLS["point"].parse({**iron, "cleavage_closure_share": 0.1}, session)
    spec = _behaviour(eased, {"young": 200000.0, "poisson": 0.3}, tmp_path)
    assert spec["crystal_plasticity"]["cleavage"]["closure"] == 0.1
    # the slip nucleates the cleavage: a softer slip, planes that it weakens as it goes
    soft = {**iron, "tau0": 100.0, "cleavage_strength": 600.0, "cleavage_energy": 4.0,
            "path": "t=0,1 n=120 E33=0,0.08"}  # fmt: skip
    shielded = _run(session, "point", soft)["values"]
    assert shielded["cleavage_damage"] == 0.0 and shielded["S33"] > 200.0
    nucleated = _run(session, "point", {**soft, "cleavage_coupling": 8.0})["values"]
    assert nucleated["cleavage_damage"] > 0.9 and nucleated["S33"] < 0.2 * shielded["S33"]
    assert "a coupling below" in _error(session, "point", {**soft, "cleavage_coupling": 40.0})
    assert "planes that close" in _error(
        session, "point", {**iron, "cleavage_closure": False, "cleavage_friction": 0.4}
    )


def test_material_point_paths(tmp_path):
    session = api.Session(tmp_path)
    tension = {"young": E, "poisson": NU, "loading": "tension", "amplitude": 0.01, "increments": 20}
    # Norton's creep under a held stress: the plastic strain rate (S / K)^n
    norton = {**tension, "behaviour": "mises", "yield_stress": 0.0, "isotropic": "0, 1"}
    creep = {**norton, "norton": "0.01, 4", "loading": "creep", "stress": 0.002, "hold": 10}
    out = _run(session, "point", creep)
    time, p = _column(session, out, "time"), _column(session, out, "p")
    assert out["tables"][0]["count"] == len(time) == 41
    held = p[-1] - p[np.flatnonzero(np.isclose(time, 1.0))[0]]
    assert held == pytest.approx(10 * (0.002 / 0.01) ** 4, rel=1e-6)
    # relaxation under a held strain: the stress relaxes towards the yield stress
    relaxation = {**norton, "yield_stress": R0, "norton": "0.001, 3", "loading": "relaxation"}
    out = _run(session, "point", relaxation)
    time, s11 = _column(session, out, "time"), _column(session, out, "S11")
    assert R0 < s11[-1] < s11[np.flatnonzero(np.isclose(time, 1.0))[0]]
    # finite strains: the logarithmic strain of the stretch; a JSON path of a shear
    out = _run(session, "point", {**tension, **PERFECT, "kinematics": "finite"})["values"]
    assert out["LE11"] == pytest.approx(np.log(1.01), rel=1e-9)
    # the yield of the Kirchhoff stress J sigma (conjugate to the logarithmic strain)
    jacobian = np.prod([1 + out[f"E{c}"] for c in ("11", "22", "33")])
    assert jacobian * out["TS11"] == pytest.approx(R0, rel=1e-6) and jacobian > 1
    shear = {"times": [0, 1], "increments": 4, "strain": {"12": 0.001}}
    custom = {**tension, "behaviour": "elastic", "loading": "custom", "path": json.dumps(shear)}
    out = _run(session, "point", custom)["values"]
    assert out["S12"] == pytest.approx(E / (2 * (1 + NU)) * 0.001, rel=1e-10)
    # porous plasticity (GTN) under a triaxial strain: the porosity grows
    gtn = {
        "behaviour": "porous",
        "criterion": "gurson",
        "porosity": 0.01,
        "yield_stress": R0,
        "isotropic": "0, 1",
        "loading": "custom",
        "path": "t=0,1 n=20 E11=0,0.01 E22=0,0.005 E33=0,0.005",
    }
    out = _run(session, "point", {"young": E, "poisson": NU, **gtn})["values"]
    assert out["f"] > 0.012 and out["p"] > 0
    # a crystal along <001>: twelve octahedral systems of Schmid factor 1 / sqrt(6) yield
    crystal = {
        **tension,
        "behaviour": "crystal",
        "tau0": 0.0008,
        "slip_isotropic": "0, 1",
        "slip_viscosity": "1e-6, 10",
        "increments": 10,
    }
    out = _run(session, "point", crystal)["values"]
    assert out["S11"] == pytest.approx(0.0008 * np.sqrt(6), rel=0.01) and out["p"] > 0
    # errors of the form
    assert "Norton viscosity" in _error(session, "point", {**tension, **PERFECT, "norton": "1"})
    assert "pairs" in _error(session, "point", {**tension, **PERFECT, "kinematic": "1, 2, 3"})
    assert "JSON" in _error(session, "point", {**tension, "material": "{young: 1}"})
    assert "path" in _error(session, "point", {**tension, "loading": "custom", "path": "x"})
    if importlib.util.find_spec("mgis") is None:  # a hyperbolic flow: through MFront only
        assert "MFront and MGIS" in _error(
            session, "point", {**tension, "behaviour": "zmat", "zmat": HYPERBOLIC}
        )
    # Hill's criterion: on morphoGen's kernels (h12 = 1.5: von Mises in tension along 1)
    out = _run(session, "point", {**tension, "behaviour": "zmat", "zmat": HILL})
    assert out["values"]["S11"] == pytest.approx(0.002 + 0.01 * out["values"]["p"], rel=1e-6)
    # an MFront behaviour in the materials of the configurations
    form = {"behaviour": "mfront", "mfront": "Norton.mfront", "mfront_parameters": '{"A": 2}'}
    spec = mechanics._behaviour(form, {"young": E, "poisson": NU}, tmp_path)
    assert spec["mfront"] == {"source": "Norton.mfront", "parameters": {"A": 2}}
    assert Material.from_config(spec).mfront.parameters == (("A", 2.0),)


HYPERBOLIC = """***behavior gen_evp
 **elasticity isotropic
   young 1. poisson 0.3
 **potential gen_evp ev
   *criterion mises
   *flow hyperbolic
     K 0.01
     m 1.44
     eps0 1e-8
   *isotropic constant
     R0 0.002
***return"""

HILL = """***behavior gen_evp
 **elasticity isotropic
   young 1. poisson 0.3
 **potential gen_evp ep
   *flow plasticity
   *criterion hill
    hill1 1. hill2 1. hill3 1. hill4 1.5 hill5 1. hill6 1.
   *isotropic linear
    R0 0.002 H 0.01
***return"""


def test_loading_path_fft(tmp_path):
    session, key = _workspace(tmp_path)
    micro = session.item(key)["micro"]
    common = {
        "source": key,
        "method": "fft",
        "voxels": 8,
        "integration": "full",
        "increments": 4,
        "amplitude": 0.01,
        "video": False,
    }
    # elastic phases: the linear response of the stiffness of the same voxels
    out = _run(session, "path", {**common, "materials": ELASTIC, "behaviour": "elastic"})
    image = micro.voxelize(shape=8)
    mats = {0: Material(young=E, poisson=NU), 1: Material(young=10 * E, poisson=NU)}
    stiffness = homogenize(image, mats, properties=["stiffness"])["stiffness.periodic"]["tensor"]
    values = out["values"]
    assert values["S11"] / values["E11"] == pytest.approx(_young(stiffness), rel=1e-5)
    assert values["p"] == 0 and values["steps"] == 4
    (item,) = out["items"]
    assert item["kind"] == "simulation" and item["source"] == key
    folder = session.file(item["key"], "results.csv").parent
    names = {f["name"] for f in out["files"]}
    assert {
        "results.csv",
        "homogenization.csv",
        "configuration.yaml",
        "sample_0000_tension_periodic.csv",
        "sample_0000_tension_periodic.vti",
        "tension_step001.vti",
        "tension_step004.vti",
    } <= names
    assert (folder / "sample_0000_tension_periodic.vti").exists()
    # a perfectly plastic matrix around voids: the stress saturates below the yield stress
    out = _run(
        session,
        "path",
        {
            **common,
            "materials": POROUS,
            **PERFECT,
            "amplitude": 0.02,
            "increments": 8,
            "video": True,
            "field": "p",
            "fields": "end",
        },
    )
    curve = next(t for t in out["tables"] if t["title"].startswith("curves"))
    s11 = np.array([r[curve["columns"].index("S11")] for r in curve["rows"]], dtype=float)
    assert s11[-1] < R0 and abs(s11[-1] - s11[-2]) < 0.02 * s11[-1]
    assert out["values"]["p"] > 0 and len(out["videos"]) == 1
    assert {f["title"] for f in out["figures"]} == {"curves", "mean of each phase"}
    assert not any(f["name"].startswith("tension_step") for f in out["files"])
    # the result of the path is exported as a simulation
    request = {
        "source": out["items"][0]["key"],
        "outputs": ["figure", "tables", "fields"],
        "field": "von_mises",
    }
    exported = api.export(request, session)
    assert {"results.csv", "sample_0000_tension_periodic.vti", "view.png"} <= {
        f["name"] for f in exported["files"]
    }
    # errors of the form
    assert "phase 1 is a void" in _error(
        session, "path", {**common, "materials": POROUS, "phase": 1}
    )
    assert "voxels per edge" in _error(session, "path", {**common, "voxels": 2})


def test_loading_path_of_a_damaged_matrix(tmp_path):
    """A loading path with the damage of the form in the matrix, delayed by its viscosity:
    the response softens and the damage is that of the matrix."""
    session, key = _workspace(tmp_path)
    params = {
        "source": key,
        "method": "fft",
        "voxels": 8,
        "increments": 8,
        "amplitude": 0.006,
        "video": False,
        "materials": ELASTIC,
        "behaviour": "damage",
        "strength": 0.002,
        "fracture_energy": 2e-5,
        "damage_length": 0.5,
        "damage_viscosity": 0.3,
    }
    out = _run(session, "path", params)
    s11, damaged = _column(session, out, "S11"), _column(session, out, "p")
    assert 0 < s11.argmax() < len(s11) - 1 and 0.0 < s11[-1] < s11.max()  # a peak inside
    assert damaged[0] == 0.0 and 0.05 < damaged[-1] < 1.0


def test_loading_paths_on_the_gpu(tmp_path):
    """The device of the loading paths: on the GPU the curve of the CPU (a perfectly plastic
    matrix around voids), the device kept in the configuration of the run; without a GPU,
    "gpu" is an error of the run."""
    from morphogen import gpu

    session, key = _workspace(tmp_path)
    common = {"source": key, "method": "fft", "voxels": 8, "integration": "full",
              "increments": 4, "amplitude": 0.02, "video": False, "materials": POROUS,
              **PERFECT}  # fmt: skip
    if not gpu.available():
        assert "gpu" in _error(session, "path", {**common, "device": "gpu"})
        return
    cpu = _run(session, "path", common)
    on_gpu = _run(session, "path", {**common, "device": "gpu"})
    assert on_gpu["values"]["S11"] == pytest.approx(cpu["values"]["S11"], rel=1e-8)
    config = session.file(on_gpu["items"][0]["key"], "configuration.yaml")
    assert "device: gpu" in config.read_text(encoding="utf-8")


FE = {
    "method": "fe",
    "element_size": 2.0,
    "order": "1",
    "increments": 3,
    "amplitude": 0.01,
    "video": False,
}


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="conforming meshes need MMG (pip install pymmg)")
def test_loading_path_fe(tmp_path):
    session, key = _workspace(tmp_path)
    micro = session.item(key)["micro"]
    mesh = conforming_mesh(micro, **MeshSpec(size=2.0, order=1).kwargs(porous=False))
    mats = {0: Material(young=E, poisson=NU), 1: Material(young=10 * E, poisson=NU)}
    # elastic phases: the linear response of the stiffness of the same mesh, both conditions
    for bc in ("periodic", "kubc"):
        params = {"source": key, **FE, "materials": ELASTIC, "behaviour": "elastic", "bc": bc}
        out = _run(session, "path", params)
        tensor = homogenize(mesh, mats, bcs=[bc], properties=["stiffness"])[f"stiffness.{bc}"]
        values = out["values"]
        assert values["mesh_elements"] == mesh.n_elements
        assert values["S11"] / values["E11"] == pytest.approx(_young(tensor["tensor"]), rel=1e-5)
    names = {f["name"] for f in out["files"]}
    assert {"sample_0000_tet.vtu", "sample_0000_tension_kubc.vtu", "tension_step003.vtu"} <= names


def test_loading_path_interfaces_of_the_form(tmp_path):
    """The errors of the form of the interfaces that debond, before any mesh."""
    session, key = _workspace(tmp_path)
    params = {"source": key, **FE, "materials": ELASTIC, "behaviour": "elastic"}
    weak = {"interfaces": True, "interface_strength": 0.004, "interface_energy": 4e-4}
    assert "strength of the interfaces" in _error(
        session, "path", {**params, **weak, "interface_strength": 0.0}
    )
    assert "friction of the interfaces" in _error(
        session, "path", {**params, **weak, "interface_friction": -0.1}
    )
    video = {"video": True, "field": "interface_damage"}
    assert "field of the interfaces" in _error(session, "path", {**params, **video})
    voxels = {**params, "method": "fft", "voxels": 8, **weak, **video}
    assert "field of the interfaces" in _error(session, "path", voxels)


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="conforming meshes need MMG (pip install pymmg)")
def test_loading_path_with_interfaces_that_debond(tmp_path):
    """The interfaces of the spheres debond along the path (cohesive zones on the mesh):
    strong ones give the response of the bonded phases, weak ones soften it."""
    session, key = _workspace(tmp_path)
    params = {"source": key, **FE, "materials": ELASTIC, "behaviour": "elastic", "increments": 1}
    bonded = _run(session, "path", params)["values"]
    assert "debonded_area" not in bonded
    strong = {"interfaces": True, "interface_strength": 1.0, "interface_energy": 1.0}
    values = _run(session, "path", {**params, **strong})["values"]
    # the penalty stiffness of the interfaces: 100 E / h
    assert values["S11"] == pytest.approx(bonded["S11"], rel=5e-3)
    assert values["interface_damage"] == 0.0 and values["debonded_area"] == 0.0
    weak = {
        "interfaces": True,
        "interface_strength": 0.004,
        "interface_energy": 4e-4,
        "interface_friction": 0.2,
        "interface_viscosity": 0.05,
    }
    out = _run(session, "path", {**params, **weak, "increments": 4})
    values = out["values"]
    assert values["interface_damage"] > 0.5 and values["interface_opening"] > 0.0
    assert values["S11"] < 0.985 * bonded["S11"]  # softened (its viscosity delays it)
    (item,) = out["items"]
    assert item["info"]["interfaces"] == "debonding"
    # the fields of the interfaces are those of the simulation; its configuration has them
    listed = api.simulation_fields({"key": item["key"]}, session)["fields"]
    assert {"interface_damage", "interface_opening", "von_mises"} <= set(listed)
    folder = session.file(item["key"], "results.csv").parent
    saved = yaml.safe_load((folder / "configuration.yaml").read_text(encoding="utf-8"))
    assert saved["homogenization"]["nonlinear"]["interfaces"] == {
        "normal_strength": 0.004,
        "mode_I": 4e-4,
        "friction": 0.2,
        "viscosity": 0.05,
    }
    # the grains are those of tessellations
    assert "grain of each element" in _error(
        session, "path", {**params, **weak, "interface_faces": "grains"}
    )


def test_loading_path_of_a_matrix_that_cleaves(tmp_path):
    """A loading path whose matrix is a crystal with cleavage planes: the damage of its
    planes joins the values, the curves and the fields of the simulation."""
    session, key = _workspace(tmp_path)
    params = {
        "source": key,
        "method": "fft",
        "voxels": 8,
        "increments": 10,
        "amplitude": 0.01,
        "video": False,
        "materials": ELASTIC,
        "behaviour": "crystal",
        "cubic": "1.7, 1.2, 0.75",
        "tau0": 0.01,
        "slip_isotropic": "0, 1",
        "slip_viscosity": "0.0002, 10",
        "cleavage": "cube",
        "cleavage_strength": 0.003,
        "cleavage_energy": 2e-4,
        "cleavage_length": 0.5,  # the voxels
        "cleavage_viscosity": "0.002, 1",
    }
    out = _run(session, "path", params)
    values = out["values"]
    assert 0.0 < values["cleavage_damage"] < 1.0 and values["cleavage_opening"] > 0.0
    (item,) = out["items"]
    listed = api.simulation_fields({"key": item["key"]}, session)["fields"]
    assert {"cleavage_damage", "cleavage_opening"} <= set(listed)
    folder = session.file(item["key"], "results.csv").parent
    saved = yaml.safe_load((folder / "configuration.yaml").read_text(encoding="utf-8"))
    assert saved["homogenization"]["materials"]["0"]["crystal_plasticity"]["cleavage"] == {
        "planes": "cube",
        "strength": 0.003,
        "energy": 2e-4,
        "length": 0.5,
        "K": 0.002,
        "n": 1.0,
    }
    video = {"video": True, "field": "twin_fraction"}
    assert "crystals with twins" in _error(session, "path", {**params, **video})
    video = {"video": True, "field": "cleavage_microdamage"}
    assert "planes with a damage band" in _error(session, "path", {**params, **video})
    # planes with a damage band: regularized by their microdamage, a field of the simulation
    banded = {**params, "cleavage_length": None, "cleavage_band": 1.5}
    out = _run(session, "path", banded)
    assert out["values"]["cleavage_microdamage"] > 0.0
    (item,) = out["items"]
    listed = api.simulation_fields({"key": item["key"]}, session)["fields"]
    assert {"cleavage_damage", "cleavage_opening", "cleavage_microdamage"} <= set(listed)
    folder = session.file(item["key"], "results.csv").parent
    saved = yaml.safe_load((folder / "configuration.yaml").read_text(encoding="utf-8"))
    planes = saved["homogenization"]["materials"]["0"]["crystal_plasticity"]["cleavage"]
    assert planes["regularization"] == {"width": 1.5} and "length" not in planes


def test_loading_path_and_the_probability_of_cleavage(tmp_path):
    """The criterion of Beremin along a loading path of the form: the Weibull stress and
    the probability of failure join the curves and the values, the stress that drives the
    cleavage the fields; they are those of the criterion on the fields of the steps."""
    from morphogen.homogenization.beremin import weibull_stress
    from morphogen.io import read_vti

    session, key = _workspace(tmp_path)
    params = {
        "source": key,
        "method": "fft",
        "voxels": 8,
        "materials": ELASTIC,
        **PERFECT,
        "increments": 6,
        "amplitude": 0.012,
        "video": False,
        "beremin": True,
        "beremin_modulus": 20.0,
        "beremin_scale": 0.02,
        "beremin_volume": 2.0,
        "beremin_plastic": 1e-5,
    }
    out = _run(session, "path", params)
    values = out["values"]
    assert 0.0 < values["beremin_probability"] < 1.0 and values["beremin_weibull"] > 0.0
    assert values["beremin_probability"] == pytest.approx(
        -np.expm1(-((values["beremin_weibull"] / 0.02) ** 20.0)), rel=1e-6
    )
    assert "probability of cleavage" in {f["title"] for f in out["figures"]}
    curve = next(t for t in out["tables"] if t["title"].startswith("curves"))
    assert {"beremin_weibull", "beremin_probability"} <= set(curve["columns"])
    (item,) = out["items"]
    assert item["info"]["cleavage (Beremin)"].startswith("probability")
    folder = session.file(item["key"], "results.csv").parent
    end = {
        image.name: image.data for image in read_vti(folder / "sample_0000_tension_periodic.vti")
    }
    driving = end["beremin_stress"]
    assert np.isfinite(driving).any()  # the voxels that have yielded
    volume = (4.0 / 8) ** 3
    assert weibull_stress(driving, volume, 20.0, 2.0) == pytest.approx(
        values["beremin_weibull"], rel=1e-5
    )
    saved = yaml.safe_load((folder / "configuration.yaml").read_text(encoding="utf-8"))
    assert saved["homogenization"]["beremin"] == {
        "modulus": 20.0,
        "scale": 0.02,
        "volume": 2.0,
        "plastic": 1e-5,
    }
    # the planes of the crystals: here in the axes of the sample, below the principal stress
    planes = _run(session, "path", {**params, "beremin_stress": "cube"})["values"]
    assert 0.0 < planes["beremin_weibull"] <= values["beremin_weibull"] * (1 + 1e-9)
    # errors of the form
    assert "scale of the stresses" in _error(session, "path", {**params, "beremin_scale": None})
    assert "modulus of Weibull" in _error(session, "path", {**params, "beremin_modulus": 0.5})


def test_loading_path_judged_in_fatigue(tmp_path):
    """A fatigue criterion judges the points of the solid phases along the cycle of the
    path: the fields, the figure and the values of the tool are those of the criterion on
    the stresses of the steps."""
    from morphogen.homogenization import DangVan
    from morphogen.io import read_vti

    session, key = _workspace(tmp_path)
    params = {
        "source": key,
        "method": "fft",
        "voxels": 8,
        "materials": ELASTIC,
        "behaviour": "elastic",
        "loading": "cycle",
        "increments": 2,
        "amplitude": 0.002,
        "video": False,
        "fatigue": "dang_van",
        "fatigue_bending": 0.011,
        "fatigue_torsion": 0.008,
    }
    out = _run(session, "path", params)
    values = out["values"]
    assert values["fatigue_safety"] == pytest.approx(1.0 / values["fatigue_utilization"])
    assert 0.0 < values["fatigue_beyond"] < 0.5 and values["fatigue_utilization"] > 1.0
    assert "fatigue of the cycle" in {f["title"] for f in out["figures"]}
    (item,) = out["items"]
    assert item["info"]["fatigue"].startswith("dang_van, utilization")
    folder = session.file(item["key"], "results.csv").parent
    # the criterion on the stresses of the steps of the cycle (the files of the fields)
    names = [f"stress_{c}" for c in ("11", "22", "33", "23", "31", "12")]
    history = []
    for step in range(1, 9):
        images = {
            image.name: image.data for image in read_vti(folder / f"cycle_step{step:03d}.vti")
        }
        history.append(np.stack([images[name] for name in names], axis=-1))
    history = np.array([np.zeros_like(history[0]), *history])
    equivalent = DangVan.from_limits(0.011, 0.008).equivalent(history.reshape(9, -1, 6))
    end = {image.name: image.data for image in read_vti(folder / "sample_0000_cycle_periodic.vti")}
    assert set(end) >= {"fatigue_utilization", "fatigue_equivalent", "fatigue_amplitude"}
    assert end["fatigue_utilization"].max() == pytest.approx(values["fatigue_utilization"])
    assert np.allclose(end["fatigue_equivalent"].ravel(), equivalent, rtol=1e-5, atol=1e-9)
    saved = yaml.safe_load((folder / "configuration.yaml").read_text(encoding="utf-8"))
    assert saved["homogenization"]["fatigue"] == {
        "criterion": "dang_van",
        "bending": 0.011,
        "torsion": 0.008,
    }
    # the last half of the cycle only; Sines takes the limit of repeated bending
    sines = {"fatigue": "sines", "fatigue_repeated": 0.009, "fatigue_cycle": [0.002, 0.008]}
    half = _run(session, "path", {**params, **sines, "duration": 0.002})["values"]
    assert 0.0 < half["fatigue_utilization"] < values["fatigue_utilization"]
    # errors of the form
    assert "fatigue limit (bending)" in _error(session, "path", {**params, "fatigue_bending": None})
    assert "fatigue limit (repeated)" in _error(session, "path", {**params, "fatigue": "sines"})
    assert "two times of the path" in _error(session, "path", {**params, "fatigue_cycle": [1, 9]})
    assert "torsion limit between" in _error(session, "path", {**params, "fatigue_torsion": 0.004})


@pytest.mark.skipif(not has_mmg(), reason="conforming meshes need MMG (pip install pymmg)")
def test_loading_path_fe_finite(tmp_path):
    session, key = _workspace(tmp_path)
    # plasticity in finite strains, with the video of the deformed mesh
    params = {"source": key, **FE, "materials": ELASTIC, **PERFECT, "kinematics": "finite"}
    out = _run(session, "path", {**params, "video": True})
    assert out["values"]["p"] > 0 and "LE11" in out["values"] and "TS11" in out["values"]
    # VTK draws the deformed mesh where it renders; elsewhere the tool warns, without a video
    assert len(out["videos"]) == (1 if vtk_can_render() else 0)
    # the view of the fields of the Export tab, on the deformed mesh (VTK draws it)
    outputs = ["figure", "mesh"] if vtk_can_render() else ["mesh"]
    exported = api.export({"source": out["items"][0]["key"], "outputs": outputs}, session)
    names = {f["name"] for f in exported["files"]}
    assert "sample_0000_tet.vtu" in names and ("view.png" in names) == vtk_can_render()


def test_ntfa(tmp_path):
    session, key = _workspace(tmp_path)
    params = {
        "source": key,
        "materials": ELASTIC,
        "voxels": 8,
        "modes": 3,
        "yield_stress": R0,
        "isotropic": "0.001, 100",
        "training": ["tension_11"],
        "training_amplitude": 0.004,
        "training_increments": 8,
        "test_loading": "cycle",
        "test_amplitude": 0.003,
        "test_increments": 4,
    }
    out = _run(session, "ntfa", params)
    values = out["values"]
    assert values["modes"] == 3 and values["energy_0"] > 0.999
    # on its training path: exact while elastic, then a few percent (the original NTFA
    # stiffens as the plastic flow spreads); a few percent along a cycle too
    assert values["gap_tension_11"] < 0.04 and values["gap_cycle"] < 0.06
    training = next(t for t in out["tables"] if t["title"].startswith("path tension_11"))
    full, reduced = (training["columns"].index(c) for c in ("S11 full", "S11 NTFA"))
    first = training["rows"][1]  # elastic (the cells of the page: six digits)
    assert first[reduced] == pytest.approx(first[full], rel=2e-5)
    assert values["ntfa_ms"] < 1e3 * values["full_field_s"]
    assert "ntfa_model.zip" in {f["name"] for f in out["files"]}  # the .npz of NTFAModel
    _, _, folder, _ = out["files"][0]["url"].split("/")
    again = NTFAModel.load(session.root / folder / "ntfa_model.npz")
    assert again.n_modes == 3
    assert len(out["tables"]) == 2 and len(out["figures"]) == 1
    assert "at least one" in _error(session, "ntfa", {**params, "training": []})
    heated = {**params, "test_loading": "custom", "test_path": "t=0,1 E11=0.001 dT=1"}
    assert "temperature" in _error(session, "ntfa", heated)


def test_linear_tensors(tmp_path):
    session = api.Session(tmp_path)
    # a polycrystal of copper: cubic crystals turned by the orientations of their grains
    model = model_from_config({"type": "voronoi", "n_cells": 12, "crystal_orientations": True})
    micro = model.generate(Domain(size=1.0, periodic=True), derive_seed(3, 0))
    key, _ = session.folder()
    session.add(
        key,
        "morphology",
        "voronoi (seed 3)",
        micro=micro,
        seed=3,
        model=model.to_config(),
        domain={"size": 1.0, "periodic": True},
        phases={"1": "cells"},
    )
    c11, c12, c44 = 1.684, 1.214, 0.754
    params = {
        "source": key,
        "materials": {"1": {"young": 1, "poisson": 0.3}},
        "crystal": "cubic",
        "constants": [c11, c12, c44],
        "voxels": 12,
    }
    out = _run(session, "tensors", params)
    values = out["values"]
    # the bulk modulus of a cubic polycrystal is (C11 + 2 C12) / 3 whatever its texture
    assert values["stiffness.bulk_modulus"] == pytest.approx((c11 + 2 * c12) / 3, rel=1e-6)
    shear = _bounds(out)["shear_modulus"]
    assert shear["reuss"] < values["stiffness.shear_modulus"] < shear["voigt"]
    assert values["stiffness.universal_anisotropy"] > 0  # a few grains: not isotropic
    stiffness = next(t for t in out["tables"] if t["title"].startswith("stiffness"))
    assert stiffness["columns"] == ["", *mechanics.COMPONENTS] and stiffness["count"] == 6
    # its compliance S = C⁻¹ beside it, in the same notation (stated, against TFEL's order)
    compliance = next(t for t in out["tables"] if t["title"].startswith("compliance"))
    assert compliance["columns"] == ["", *mechanics.COMPONENTS] and "C⁻¹" in compliance["title"]
    assert "23, 31, 12" in stiffness["title"] and "12, 13, 23" in stiffness["title"]
    c, s = (np.array([r[1:] for r in t["rows"]]) for t in (stiffness, compliance))
    assert np.allclose(s @ c, np.eye(6), atol=1e-4)
    assert s[3, 3] == pytest.approx(1 / values["stiffness.G23"], rel=1e-5)
    assert s[0, 1] == pytest.approx(-values["stiffness.nu12"] / values["stiffness.E1"], rel=1e-5)
    assert [f["title"] for f in out["figures"]] == ["directional Young's modulus"]
    # a composite: its thermal expansion follows from its stiffness (Levin, Rosen-Hashin)
    session, key = _workspace(tmp_path / "composite")
    alphas, conductivities = (1e-5, 2e-6), (1.0, 10.0)
    params = {
        "source": key,
        "materials": ELASTIC,
        "voxels": 12,
        "tensors": ["stiffness", "thermal_expansion", "conductivity_tensor"],
        "expansions": list(alphas),
        "conductivities": list(conductivities),
    }
    out = _run(session, "tensors", params)
    values = out["values"]
    f = np.array([values["fraction_0"], values["fraction_1"]])
    k = np.array([E, 10 * E]) / (3 * (1 - 2 * NU))
    table = next(t for t in out["tables"] if t["title"].startswith("stiffness"))
    _, _, folder, name = table["url"].split("/")
    with open(session.root / folder / name, encoding="utf-8") as file:
        c = np.array([row[1:] for row in csv.reader(file)][1:], dtype=float)  # all digits
    hydro = np.array([1.0, 1, 1, 0, 0, 0])
    expected = f @ alphas * hydro + (alphas[0] - alphas[1]) / (1 / k[0] - 1 / k[1]) * (
        3 * np.linalg.solve(c, hydro) - (f @ (1 / k)) * hydro
    )
    measured = [values[f"thermal_expansion.A{n}"] for n in ("11", "22", "33")]
    assert measured == pytest.approx(expected[:3], rel=1e-6)
    conductivity = _bounds(out)["conductivity"]
    conduction = next(t for t in out["tables"] if t["title"].startswith("conductivity"))
    for i, row in enumerate(conduction["rows"]):  # between the harmonic and arithmetic means
        assert conductivity["reuss"] < row[1 + i] < conductivity["voigt"]
    assert "one per phase" in _error(session, "tensors", {**params, "expansions": [1e-5]})


def test_linear_tensors_by_extended_finite_elements(tmp_path):
    """The linear tensors on a regular grid cut by the level set (XFEM): no mesh, the
    interfaces where the model puts them. The moduli stand between the bounds of the
    phase fractions and close to those of the FFT solver on voxels twice as fine (12
    cells for spheres of 1.8 cells of radius: a few percent), and the thermal expansion
    follows from the stiffness and the fractions (Levin), as for any solver."""
    session, key = _workspace(tmp_path)
    alphas = (1e-5, 2e-6)
    params = {
        "source": key,
        "materials": ELASTIC,
        "method": "xfem",
        "cells": 12,
        "tensors": ["stiffness", "thermal_expansion"],
        "expansions": list(alphas),
    }
    out = _run(session, "tensors", params)
    values = out["values"]
    assert values["cells"] == 12**3 and "voxels" not in values
    assert values["fraction_0"] + values["fraction_1"] == pytest.approx(1.0, abs=1e-9)
    bounds = _bounds(out)
    for name in ("bulk_modulus", "shear_modulus"):
        assert bounds[name]["reuss"] < values[f"stiffness.{name}"] < bounds[name]["voigt"]
    fft = _run(session, "tensors", {"source": key, "materials": ELASTIC, "voxels": 24})["values"]
    assert values["fraction_1"] == pytest.approx(fft["fraction_1"], abs=0.02)
    for name in ("bulk_modulus", "shear_modulus"):
        assert values[f"stiffness.{name}"] == pytest.approx(fft[f"stiffness.{name}"], rel=0.08)
    f = np.array([values["fraction_0"], values["fraction_1"]])
    k = np.array([E, 10 * E]) / (3 * (1 - 2 * NU))
    table = next(t for t in out["tables"] if t["title"].startswith("stiffness"))
    _, _, folder, name = table["url"].split("/")
    with open(session.root / folder / name, encoding="utf-8") as file:
        c = np.array([row[1:] for row in csv.reader(file)][1:], dtype=float)
    hydro = np.array([1.0, 1, 1, 0, 0, 0])
    expected = f @ alphas * hydro + (alphas[0] - alphas[1]) / (1 / k[0] - 1 / k[1]) * (
        3 * np.linalg.solve(c, hydro) - (f @ (1 / k)) * hydro
    )
    measured = [values[f"thermal_expansion.A{n}"] for n in ("11", "22", "33")]
    assert measured == pytest.approx(expected[:3], rel=1e-4)
    # what the extended finite elements do not take
    assert "from 4 to 48" in _error(session, "tensors", {**params, "cells": 64})
    crystal = {**params, "crystal": "cubic", "constants": [1.684, 1.214, 0.754]}
    assert "isotropic phases" in _error(session, "tensors", crystal)
    model = model_from_config(
        {"type": "coating", "thickness": 0.2,
         "model": {"type": "hardcore_spheres", "radius": 0.8, "volume_fraction": 0.2}}
    )  # fmt: skip
    coated = model.generate(Domain(size=4.0, periodic=True), 3)
    other, _ = session.folder()
    session.add(other, "morphology", "coated", micro=coated, seed=3, model=model.to_config(),
                phases={"0": "matrix", "1": "spheres", "2": "coating"})  # fmt: skip
    three = {**ELASTIC, "2": {"young": 2 * E, "poisson": NU}}
    several = {"source": other, "materials": three, "method": "xfem", "cells": 8}
    assert "two phases" in _error(session, "tensors", several)
    spheres = model_from_config({"type": "boolean_spheres", "radius": 0.6, "volume_fraction": 0.2})
    window = spheres.generate(Domain(size=4.0, periodic=False), 3)
    box, _ = session.folder()
    session.add(box, "morphology", "window", micro=window, seed=3, model=spheres.to_config(),
                phases={"0": "matrix", "1": "spheres"})  # fmt: skip
    assert "periodic" in _error(session, "tensors", {**params, "source": box})


def _table(out, start: str) -> dict:
    """The table of a result whose title starts with ``start``."""
    return next(t for t in out["tables"] if t["title"].startswith(start))


def _bounds(out) -> dict:
    """The bounds of the phase fractions of a result: property -> bound -> value."""
    table = next(t for t in out["tables"] if t["title"].startswith("bounds"))
    return {r[0]: dict(zip(table["columns"], r, strict=True)) for r in table["rows"]}


def test_frame_of_beams(tmp_path):
    """The stiffness of the frame of beams of a network of the workspace: offered the
    realizations that keep a framework (an open foam, a planar Delaunay network; not the
    spheres, plainly refused), the values, tables and figure of its stiffness against
    network_stiffness, Timoshenko beams of Cowper's coefficient with rigid joints and the
    modulus and Poisson's ratio of the fibres, Ban et al.'s softening along an axis, the
    plane frame per unit area, the run kept in the workspace, the errors."""
    from morphogen.homogenization.frames import SpaceFrame, cowper_coefficient
    from morphogen.homogenization.networks import network_stiffness, softening_coefficients

    session, spheres = _workspace(tmp_path)
    keys = {}
    for name, cfg, domain in (
        ("foam", {"type": "open_foam", "n_cells": 20, "strut_radius": 0.01}, Domain(size=1.0)),
        ("planar", {"type": "delaunay_network", "n_points": 60, "radius": 0.003, "dimension": 2},
         Domain(size=(1.0, 1.0, 0.1))),
    ):  # fmt: skip
        model = model_from_config(cfg)
        key, _ = session.folder()
        session.add(key, "morphology", name, micro=model.generate(domain, derive_seed(4, 0)),
                    model=model.to_config(), phases={"0": "pores", "1": "fibres"})  # fmt: skip
        keys[name] = key
    traits = {i["title"]: i["traits"] for i in session.workspace()}
    assert "framework" in traits["#2 foam"] and "framework" in traits["#3 planar"]
    assert "framework" not in traits["#1 boolean_spheres (seed 7)"]
    fields = {f["name"]: f for f in tools.TOOLS["frame"].to_dict()["fields"]}
    assert fields["source"]["needs"] == ["framework"]
    assert fields["shear_coefficient"]["when"] == {"beams": ["timoshenko"]}
    assert fields["softening"]["fold"] is True and fields["young"]["group"] == "Beams"
    framework = session.item(keys["foam"])["micro"].metadata["framework"]
    out = _run(session, "frame", {"source": keys["foam"]})
    plain = network_stiffness(framework, 0.01)
    values = out["values"]
    assert values["E / E_s"] == pytest.approx(plain.summary()["frame_young"], rel=1e-12)
    assert values["E2"] == pytest.approx(plain.young(1), rel=1e-12)
    assert values["fibre volume fraction"] == pytest.approx(plain.density, rel=1e-12)
    assert values["beams"] == len(framework["bars"]) and "shear coefficient κ" not in values
    stiffness = next(t for t in out["tables"] if t["title"].startswith("stiffness"))
    assert stiffness["columns"] == ["", *mechanics.COMPONENTS] and stiffness["count"] == 6
    assert stiffness["rows"][1][2] == pytest.approx(plain.stiffness[1, 1], rel=1e-5)
    compliance = out["tables"][1]  # S = C⁻¹ beside C, in the same notation
    assert compliance["title"].startswith("compliance S_ij = C⁻¹ of the frame")
    assert compliance["rows"][3][4] == pytest.approx(plain.compliance()[3, 3], rel=1e-5)
    assert compliance["rows"][3][4] == pytest.approx(1 / values["G23"], rel=1e-5)
    axes = _table(out, "uniaxial stresses")
    assert [r[0] for r in axes["rows"]] == ["x", "y", "z"] and "torsion energy" in axes["columns"]
    assert [f["title"] for f in out["figures"]] == [
        "directional Young's modulus",
        "the frame: axial force (the force chains) under a stress along x",
    ]
    assert "bending-dominated" in out["text"]
    files = {f["name"] for f in out["files"]}
    assert {"stiffness.csv", "compliance.csv", "axes.csv", "frame_stiffness.json",
            "young_modulus.png", "frame.png", "frame.svg"} <= files  # fmt: skip
    with open(session.root / out["key"] / "frame_stiffness.json", encoding="utf-8") as file:
        kept = json.load(file)
    s, c = np.array(kept["compliance"]), np.array(kept["stiffness"])
    assert np.allclose(s @ c, np.eye(6), atol=1e-9) and "engineering shears" in kept["notation"]
    (run,) = [i for i in session.workspace() if i["kind"] == "result"]
    assert run["title"] == "#4 Stiffness of the frame of beams (#2)"
    assert list(run["info"])[:2] == ["E / E_s", "Poisson's ratio"]
    params = {"source": keys["foam"], "beams": "timoshenko", "rigid_joints": "0.33",
              "young": "2", "poisson": "0.25", "softening": ["modulus"], "axis": "y"}  # fmt: skip
    out = _run(session, "frame", params)
    exact = network_stiffness(framework, 0.01, young=2.0, poisson=0.25, shear_coefficient="cowper",
                              rigid_joints=0.33)  # fmt: skip
    assert out["values"]["E / E_s"] == pytest.approx(exact.summary(2.0)["frame_young"], rel=1e-12)
    assert out["values"]["shear coefficient κ"] == pytest.approx(cowper_coefficient(0.25))
    assert "shear energy" in _table(out, "uniaxial stresses")["columns"]
    frame = SpaceFrame.from_framework(framework, 0.01, young=2.0, poisson=0.25,
                                      shear_coefficient="cowper", rigid_joints=0.33)  # fmt: skip
    expected = softening_coefficients(frame, "modulus", axis=1)
    softened = _table(out, "softening")
    assert softened["columns"] == ["mode", "s", "v", "E₀"] and softened["rows"][0][0] == "modulus"
    assert softened["rows"][0][1] == pytest.approx(expected["mean"], rel=1e-5)
    assert "softening along y" in out["text"]
    with open(session.root / out["key"] / "frame_stiffness.json", encoding="utf-8") as file:
        kept = json.load(file)
    assert kept["softening"]["modes"][0]["s"] == pytest.approx(expected["mean"], rel=1e-12)
    # a planar network: a plane frame, per unit area
    out = _run(session, "frame", {"source": keys["planar"]})
    planar = session.item(keys["planar"])["micro"].metadata["framework"]
    expected = network_stiffness(planar, 0.003, dimension=2)
    assert out["tables"][0]["columns"] == ["", *mechanics.PLANE_COMPONENTS]
    assert out["tables"][1]["columns"] == ["", *mechanics.PLANE_COMPONENTS]  # its compliance
    assert "per unit area" in out["tables"][0]["title"]
    assert out["values"]["fibre volume per unit area"] == pytest.approx(expected.density)
    assert out["values"]["G12"] == pytest.approx(expected.engineering()["G12"], rel=1e-12)
    assert "per unit area" in out["text"]
    # the errors: not a network, a plane network along z
    assert "is not a network of fibres or struts" in _error(session, "frame", {"source": spheres})
    planar_z = {"source": keys["planar"], "softening": ["axial"], "axis": "z"}
    assert "x or y" in _error(session, "frame", planar_z)


def test_frame_replicas_and_drawing(tmp_path):
    """The tool Stiffness of the frame of beams: Ban et al.'s replicas of the modes of the
    softening (the distribution, coefficients of variation, number and seed of the form)
    against softening of the same frame, in a table, a figure and frame_stiffness.json,
    their progress followed replica by replica; the drawing of the frame coloured by the
    axial forces or the energies of its beams under a uniaxial stress along an axis (the
    figure produced, the share of the beams storing half the energy against beam_forces),
    in its plane for a planar network, the beams alone; the errors."""
    from morphogen.homogenization.networks import beam_forces, network_frame, softening

    session, _ = _workspace(tmp_path)
    keys = {}
    for name, cfg, domain in (
        ("foam", {"type": "open_foam", "n_cells": 20, "strut_radius": 0.01}, Domain(size=1.0)),
        ("planar", {"type": "delaunay_network", "n_points": 60, "radius": 0.003, "dimension": 2},
         Domain(size=(1.0, 1.0, 0.1))),
    ):  # fmt: skip
        model = model_from_config(cfg)
        key, _ = session.folder()
        session.add(key, "morphology", name, micro=model.generate(domain, derive_seed(4, 0)),
                    model=model.to_config(), phases={"0": "pores", "1": "fibres"})  # fmt: skip
        keys[name] = key
    fields = {f["name"]: f for f in tools.TOOLS["frame"].to_dict()["fields"]}
    assert fields["draw"]["group"] == "Drawing of the frame" and not fields["draw"]["fold"]
    assert fields["draw_axis"]["group"] == "Drawing of the frame"
    softening_box = "Softening of non-identical beams"
    assert fields["replicas"]["group"] == fields["seed"]["group"] == softening_box
    assert fields["cvs"]["default"] == [0.1, 0.2, 0.3] and fields["replicas"]["default"] == 0
    params = {"source": keys["foam"], "softening": ["modulus", "axial"], "replicas": "4",
              "cvs": "0.1, 0.3", "distribution": "bimodal", "seed": "2", "draw": "stretching",
              "draw_axis": "z"}  # fmt: skip
    job = ToolJob({"tool": "frame", "params": params}, session)
    messages, report = [], job.report
    job.report = lambda fraction, message="": (messages.append(message), report(fraction, message))
    job._run()
    status = job.status()
    assert status["state"] == "done", status.get("error")
    out = status["result"]
    assert {"replicas of the modulus: 1 / 8", "replicas of the axial: 8 / 8"} <= set(messages)
    frame = network_frame(session.item(keys["foam"])["micro"])
    with open(session.root / out["key"] / "frame_stiffness.json", encoding="utf-8") as file:
        kept = json.load(file)
    assert kept["replicas"]["distribution"] == "bimodal" and kept["replicas"]["cv"] == [0.1, 0.3]
    rows = kept["replicas"]["rows"]
    for k, mode in enumerate(("modulus", "axial")):
        expected = softening(frame, mode, (0.1, 0.3), replicas=4, law="bimodal", seed=2)
        for got, row in zip(rows[2 * k : 2 * k + 2], expected.rows(), strict=True):
            assert got.keys() == row.keys()
            for key, value in row.items():  # the mode is a name, the rest numbers
                wanted = value if isinstance(value, str) else pytest.approx(value, rel=1e-9)
                assert got[key] == wanted, key
    table = next(t for t in out["tables"] if t["title"].startswith("replicas"))
    assert table["columns"] == list(mechanics.REPLICA_COLUMNS.values()) and table["count"] == 4
    assert table["rows"][3][0] == "axial" and table["rows"][3][1] == 0.3
    assert table["rows"][3][5] == pytest.approx(rows[3]["corrected"], rel=1e-5)
    assert "bimodal law, 4 per CV, seed 2" in table["title"]
    titles = [f["title"] for f in out["figures"]]
    assert titles[1:] == [
        "the frame: stretching energy of each beam under a stress along z",
        "replicas: the mean and scatter of E against the second order",
    ]
    files = {f["name"] for f in out["files"]}
    assert {"frame.png", "replicas.png", "replicas.svg", "replicas.csv"} <= files
    assert "Replicas (bimodal law, 4 per CV, corrected by the control variate)" in out["text"]
    loads = beam_forces(frame, 2)
    energy = sum(loads[k] for k in ("stretching", "bending", "torsion"))
    stored = np.cumsum(np.sort(energy)[::-1]) / energy.sum()
    half = (np.searchsorted(stored, 0.5) + 1) / len(energy)
    assert out["values"]["beams storing half the energy"] == pytest.approx(half)
    assert kept["drawing"]["axis"] == "z" and kept["drawing"]["colour"] == "stretching"
    assert kept["drawing"]["tension"] == pytest.approx(loads["force"].max())
    assert "of the beams store half the energy" in out["text"]
    # a planar network, in its plane; the beams alone
    out = _run(session, "frame", {"source": keys["planar"], "draw": "bending", "draw_axis": "y"})
    assert out["figures"][1]["title"].startswith("the frame: bending energy of each beam")
    assert (session.root / out["key"] / "frame.png").stat().st_size > 1000
    out = _run(session, "frame", {"source": keys["planar"], "draw": "beams"})
    assert out["figures"][1]["title"] == "the frame: its beams"
    # the errors
    for params, message in (
        ({"source": keys["foam"], "replicas": "4"}, "of the modes of the softening"),
        ({"source": keys["foam"], "softening": ["axial"], "replicas": "1"}, "at least 2"),
        ({"source": keys["foam"], "softening": ["axial"], "replicas": "3",
          "distribution": "bimodal", "cvs": "0.5, 1.5"}, "below 1"),
        ({"source": keys["planar"], "draw_axis": "z"}, "x or y"),
    ):  # fmt: skip
        assert message in _error(session, "frame", params)


def test_frame_in_the_summaries_of_networks(tmp_path):
    """Generate: the stiffness of the frame of beams of a network model (frame_stiffness:
    true) in a table of its own, labelled, and in the card of its morphology."""
    session = api.Session(tmp_path)
    request = {"model": "open_foam", "domain": {"size": 1.0, "periodic": True}, "seed": 2,
               "resolution": 24, "params": {"n_cells": 12, "strut_radius": 0.02}}  # fmt: skip
    out = api.generate(request, session)
    assert "frame" not in out and "frame E / E_s" not in out["item"]["info"]
    request["params"]["frame_stiffness"] = True
    out = api.generate(request, session)
    frame = out["frame"]["values"]
    assert frame["E / E_s"] == out["summary"]["frame_young"]
    assert frame["E / (E_s φ)"] == out["summary"]["frame_young_per_density"]
    assert list(frame) == list(api.FRAME_SUMMARY.values()) and "uniaxial" in out["frame"]["note"]
    # its stiffness and compliance, in the summary and in tables of their own
    stiffness, compliance = out["frame"]["tables"]
    assert stiffness["title"].startswith("stiffness C_ij of the frame")
    assert compliance["url"].endswith("/frame_compliance.csv")
    c, s = api.frame_matrices(out["summary"])
    assert np.allclose(s @ c, np.eye(6), atol=1e-6)
    assert stiffness["rows"][0][1] == pytest.approx(c[0, 0], rel=1e-5)
    young = np.mean(1 / np.diag(s)[:3])  # E / E_s, the mean over the axes
    assert young == pytest.approx(out["summary"]["frame_young"], rel=1e-6)
    assert out["item"]["info"]["frame E / E_s"] == frame["E / E_s"]
    assert "framework" in out["item"]["traits"]
