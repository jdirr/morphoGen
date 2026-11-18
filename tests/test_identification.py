"""Identification of the parameters of a behaviour on reference responses
(:mod:`morphogen.homogenization.identification`): the parameters of a response computed
with known ones recovered from other starting values, on one reference and on several
(each by its weight), the stop of a long identification, and the errors of the parameters
and references given."""

import copy
import threading

import numpy as np
import pytest

from morphogen.homogenization import Material, simulate_point
from morphogen.homogenization.identification import (
    Reference,
    driven,
    get_parameter,
    identify,
    identify_references,
    set_parameter,
)
from morphogen.stopping import Stopped, stoppable

TRUE = {
    "young": 1.0,
    "poisson": 0.3,
    "plasticity": {"yield": 0.002, "hardening": 0.05, "isotropic": [{"Q": 0.001, "b": 100.0}]},
}
CYCLE = {"times": [0.0, 1.0, 3.0, 4.0], "increments": [10, 20, 10],
         "strain": {"11": [0.0, 0.01, -0.01, 0.0]}}  # fmt: skip


def _reference():
    res = simulate_point(Material.from_config(TRUE), CYCLE)
    return res.times, res.stress


def test_parameters_recovered():
    """The plastic parameters of a cycle recovered from starting values 50 % off, to
    1e-5: the stresses of the reference at its times, those of the driven component."""
    times, stresses = _reference()
    start = {**TRUE, "plasticity": {"yield": 0.003, "hardening": 0.03,
                                    "isotropic": [{"Q": 0.0015, "b": 60.0}]}}  # fmt: skip
    seen = []
    fit = identify(start, CYCLE, times, stresses, ["yield", "hardening", "Q", "b"],
                   callback=lambda k, e: seen.append(e))  # fmt: skip
    assert fit.components == [0] and fit.initial_error > 0.05 and fit.error < 1e-7
    for name, value in fit.values.items():
        assert value == pytest.approx(get_parameter(TRUE, name), rel=1e-5), name
    assert fit.initial["yield"] == 0.003 and len(seen) == fit.evaluations
    assert Material.from_config(fit.material).plasticity.yield_stress == pytest.approx(0.002)
    assert driven(fit.response) == 0
    # one component given as a curve: its index
    young = identify({**TRUE, "young": 2.0}, CYCLE, times, stresses[:, 0], ["young"],
                     components=[0])  # fmt: skip
    assert young.values["young"] == pytest.approx(1.0, rel=1e-6)


def _hardening(R0, Q, b, C, D):
    return {"young": 1.0, "poisson": 0.3, "plasticity": {
        "yield": R0, "isotropic": [{"Q": Q, "b": b}], "kinematic": [{"C": C, "D": D}]}}  # fmt: skip


TENSION = {"times": [0.0, 1.0], "increments": [20], "strain": {"11": [0.0, 0.01]}}
SHEAR = {"times": [0.0, 1.0, 3.0], "increments": [10, 20],
         "strain": {"12": [0.0, 0.015, -0.015]}}  # fmt: skip


def _of(material, path):
    res = simulate_point(Material.from_config(material), path)
    return Reference(path, res.times, res.stress)


def test_several_references():
    """The isotropic and kinematic hardenings of a tension and a shear cycle followed
    together, recovered from values 40 % off; references weighted, each by its weight
    whatever its number of points."""
    true = _hardening(0.002, 0.001, 100.0, 0.2, 200.0)
    tension, shear = _of(true, TENSION), _of(true, SHEAR)
    names = ["yield", "Q", "b", "C", "D"]
    fit = identify_references(_hardening(0.003, 0.0014, 60.0, 0.12, 280.0), [tension, shear],
                              names, max_evaluations=400)  # fmt: skip
    assert [r.components for r in fit.references] == [[0], [5]] and fit.components == [0]
    assert len(fit.responses) == len(fit.initial_responses) == 2 and fit.response.name
    assert max(fit.gaps) < 1e-8 < 1e-2 < min(fit.initial_gaps)
    for name in names:
        assert fit.values[name] == pytest.approx(get_parameter(true, name), rel=1e-5), name
    # two tensions of different yield stresses: the yield stress found between them,
    # nearer to the one weighted three times
    soft = _of(_hardening(0.002, 0.001, 100.0, 0.1, 200.0), TENSION)
    hard = _of(_hardening(0.003, 0.001, 100.0, 0.1, 200.0), TENSION)
    middle = _hardening(0.0025, 0.001, 100.0, 0.1, 200.0)
    equal = identify_references(middle, [soft, hard], ["yield"])
    heavy = copy.copy(hard)
    heavy.weight = 3.0
    weighted = identify_references(middle, [soft, heavy], ["yield"])
    assert 0.002 < equal.values["yield"] < weighted.values["yield"] < 0.003
    gaps = np.array(weighted.gaps)
    assert weighted.error == pytest.approx(np.sqrt((gaps**2 @ [1.0, 3.0]) / 4.0), rel=1e-12)
    # each point of a reference twice: the same reference
    twice = Reference(TENSION, np.repeat(hard.times, 2), np.repeat(hard.stresses, 2, axis=0))
    again = identify_references(middle, [soft, twice], ["yield"])
    assert again.values["yield"] == pytest.approx(equal.values["yield"], rel=1e-8)


def test_porous_parameters():
    """Gurson's q1, q2 and the initial porosity of a tension and a uniaxial strain (two
    triaxialities) recovered together from values 30 to 50 % off; the parameters of porous
    plasticity named (its matrix, criteria, nucleation), tables refused."""

    def gtn(q1, q2, f0):
        return {"young": 1.0, "poisson": 0.3, "porous_plasticity": {
            "criterion": "gurson", "yield": 0.002, "hardening": 0.02, "porosity": f0,
            "q1": q1, "q2": q2}}  # fmt: skip

    tension = {"times": [0.0, 1.0], "increments": 20, "strain": {"11": [0.0, 0.05]}}
    strained = {**tension, "strain": {"11": [0.0, 0.05], "22": [0.0, 0.0], "33": [0.0, 0.0]}}
    true = gtn(1.5, 1.0, 0.01)
    fit = identify_references(gtn(1.0, 1.3, 0.015), [_of(true, tension), _of(true, strained)],
                              ["q1", "q2", "porosity"])  # fmt: skip
    for name, value in (("q1", 1.5), ("q2", 1.0), ("porosity", 0.01)):
        assert fit.values[name] == pytest.approx(value, rel=1e-4), name
    assert fit.responses[1].porosity[-1] > 0.05  # the voids grow under the triaxiality
    # q3 = q1² given (a behaviour written back by morphoGen): it follows q1
    given = gtn(1.0, 1.0, 0.01)
    given["porous_plasticity"]["q3"] = 1.0
    alone = identify_references(given, [_of(true, strained)], ["q1"])
    assert alone.values["q1"] == pytest.approx(1.5, rel=1e-4) and alone.gaps[0] < 1e-6
    assert "q3" not in alone.material["porous_plasticity"]
    config = {"young": 1.0, "poisson": 0.3, "porous_plasticity": {
        "criterion": "rousselier", "R0": 1.0, "H": 0.1, "sigma1": 0.5, "D": 2.0,
        "isotropic": [{"Q": 1.0, "b": 2.0}], "norton": {"K": 1.0, "n": 3.0},
        "nucleation": {"type": "gaussian", "fN": 0.04, "eN": 0.3, "sN": 0.1}}}  # fmt: skip
    values = {"yield": 1.0, "hardening": 0.1, "D": 2.0, "b": 2.0, "n": 3.0, "eN": 0.3}
    assert {k: get_parameter(config, k) for k in values} == values
    set_parameter(config, "yield", 2.0)
    assert config["porous_plasticity"]["R0"] == 2.0
    elliptic = {"criterion": "elliptic", "yield": 1.0, "C": [[0.0, 1.2], [0.5, 3.0]], "F": 0.05}
    table = {"young": 1.0, "poisson": 0.3, "porous_plasticity": elliptic}
    assert get_parameter(table, "F") == 0.05
    for material, name, message in (
        (table, "C", "a table of the porosity"),
        (table, "fN", "no nucleation"),
        (config, "Q2", "1 isotropic terms"),
        (table, "K", "no Norton flow"),
        (config, "tau0", "of porous plasticity"),
    ):
        with pytest.raises(ValueError, match=message):
            get_parameter(material, name)


def test_hill_coefficients():
    """The coefficients of Hill's criterion recovered from von Mises's (all 1) on tensions
    along the three axes and a shear: each tension gives a combination 4 h11 + h22 + h33 of
    the normal coefficients, the shear its own; those not given are 1."""
    true = _hardening(0.002, 0.0, 1.0, 0.0, 0.0)
    true["plasticity"] = {"yield": 0.002, "hardening": 0.02,
                          "hill": {"11": 1.3, "22": 0.8, "33": 1.1, "12": 0.7}}  # fmt: skip
    paths = [{"times": [0, 1], "increments": 15, "strain": {c: [0, 0.01]}}
             for c in ("11", "22", "33", "12")]  # fmt: skip
    start = {**true, "plasticity": {"yield": 0.002, "hardening": 0.02}}
    names = ["hill11", "hill22", "hill33", "hill12"]
    fit = identify_references(start, [_of(true, p) for p in paths], names)
    for name, value in zip(names, (1.3, 0.8, 1.1, 0.7), strict=True):
        assert fit.values[name] == pytest.approx(value, rel=1e-6), name
    assert max(fit.gaps) < 1e-8 and fit.initial["hill11"] == 1.0
    assert fit.material["plasticity"]["hill"]["23"] == 1.0
    assert get_parameter(true, "hill31") == 1.0 and get_parameter(true, "hill11") == 1.3


def test_macroscopic_law():
    """The macroscopic law of a 'morphology' that is a Hill material (its response the
    material point's): the six loadings run, each response compared on its loaded
    component; from the law estimated on the responses (E and ν of the first increment of
    the first tension, h11 held at 1), the nine parameters of that material recovered; a
    von Mises law of two loadings; the errors."""
    from morphogen.homogenization.identification import LOADINGS, law_start, macroscopic_law

    hill = {"11": 1.0, "22": 0.8, "33": 1.3, "12": 0.7, "23": 1.25, "31": 0.9}
    true = {"young": 1.0, "poisson": 0.3,
            "plasticity": {"yield": 0.002, "hardening": 0.05, "hill": hill}}  # fmt: skip
    seen = []

    def respond(path, material=true):
        seen.append(next(iter(path["strain"])))
        return simulate_point(Material.from_config(material), path)

    law = macroscopic_law(respond, strain=0.01, increments=15)
    assert seen == list(LOADINGS) and law.loadings == LOADINGS
    assert [r.components for r in law.identification.references] == [[0], [1], [2], [5], [3], [4]]
    fit = law.identification
    assert fit.initial["young"] == pytest.approx(1.0)
    assert fit.initial["poisson"] == pytest.approx(0.3)
    assert "hill11" not in fit.values and max(fit.gaps) < 1e-8
    for name, value in (
        ("yield", 0.002),
        ("hardening", 0.05),
        *((f"hill{c}", h) for c, h in hill.items() if c != "11"),
    ):
        assert fit.values[name] == pytest.approx(value, rel=1e-6), name
    assert Material.from_config(fit.material).plasticity.hill[0] == 1.0  # h11, not fitted
    # a von Mises law, a tension and a shear
    mises = {"young": 1.0, "poisson": 0.3, "plasticity": {"yield": 0.002, "hardening": 0.05}}
    law = macroscopic_law(lambda path: respond(path, mises), law="mises", loadings=["11", "12"],
                          strain=0.01, increments=15)  # fmt: skip
    assert law.identification.values["yield"] == pytest.approx(0.002, rel=1e-6)
    assert "hill" not in law.identification.material["plasticity"]
    start, fitted = law_start(law.responses[1:], ["12"], "hill")  # a shear alone
    assert fitted == ["young", "poisson", "yield", "hardening"]
    assert start["young"] == pytest.approx(1.0, rel=1e-6)  # E = 2 G (1 + 0.3)
    with pytest.raises(ValueError, match="loadings"):
        macroscopic_law(respond, loadings=["13"])
    with pytest.raises(ValueError, match="'hill', 'mises' or 'gurson'"):
        law_start(law.responses, ["11", "12"], "tresca")


def test_macroscopic_law_of_cycles():
    """Cycles (to the strain, then back to its opposite) show the Bauschinger effect: the
    law estimated takes a backstress, and the kinematic hardening of a von Mises material
    taken as the morphology is recovered with its other parameters."""
    from morphogen.homogenization.identification import macroscopic_law

    true = {"young": 1.0, "poisson": 0.3, "plasticity": {
        "yield": 0.002, "hardening": 0.02, "kinematic": [{"C": 0.3, "D": 150.0}]}}  # fmt: skip
    law = macroscopic_law(lambda path: simulate_point(Material.from_config(true), path),
                          law="mises", loadings=["11", "12"], strain=0.01, increments=10,
                          cycle=True, max_evaluations=300)  # fmt: skip
    assert law.paths[0]["strain"] == {"11": [0.0, 0.01, -0.01]}
    fit = law.identification
    assert set(fit.values) == {"young", "poisson", "yield", "hardening", "C", "D"}
    for name, value in (("yield", 0.002), ("hardening", 0.02), ("C", 0.3), ("D", 150.0)):
        assert fit.values[name] == pytest.approx(value, rel=1e-5), name
    assert max(fit.gaps) < 1e-8 < min(fit.initial_gaps)


def test_macroscopic_law_of_a_porous_morphology():
    """Gurson's law of a porous 'morphology' that is a GTN material: its matrix and its
    porosity given, q1 and q2 recovered from Tvergaard's 1.5 and 1 on a tension and a
    tension whose lateral strains are held (two stress triaxialities, compared on the
    three normal stresses), with the elasticity of the first increment."""
    from morphogen.homogenization.identification import law_start, macroscopic_law

    matrix = {"young": 1.0, "poisson": 0.3, "plasticity": {"yield": 0.002, "hardening": 0.02}}
    true = {"young": 1.0, "poisson": 0.3, "porous_plasticity": {
        "criterion": "gurson", "yield": 0.002, "hardening": 0.02, "porosity": 0.05, "q1": 1.6,
        "q2": 0.9}}  # fmt: skip
    law = macroscopic_law(lambda path: simulate_point(Material.from_config(true), path),
                          law="gurson", loadings=["11", "confined11"], matrix=matrix,
                          porosity=0.05, strain=0.02, increments=15)  # fmt: skip
    assert law.paths[1]["strain"] == {"11": [0.0, 0.02], "22": [0.0, 0.0], "33": [0.0, 0.0]}
    fit = law.identification
    assert [r.components for r in fit.references] == [[0], [0, 1, 2]]
    assert fit.initial == {"young": 1.0, "poisson": pytest.approx(0.3), "q1": 1.5, "q2": 1.0}
    assert fit.values["q1"] == pytest.approx(1.6, rel=1e-5)
    assert fit.values["q2"] == pytest.approx(0.9, rel=1e-5) and max(fit.gaps) < 1e-8
    assert fit.material["porous_plasticity"]["porosity"] == 0.05
    for kwargs, message in (
        ({"matrix": None, "porosity": 0.05}, "the matrix"),
        ({"matrix": matrix, "porosity": 1.2}, "between 0 and 1"),
        ({"matrix": {**matrix, "plasticity": {"yield": 1.0, "kinematic": [{"C": 1, "D": 1}]}},
          "porosity": 0.05}, "without kinematic"),
    ):  # fmt: skip
        with pytest.raises(ValueError, match=message):
            law_start(law.responses, ["11", "confined11"], "gurson", **kwargs)
    with pytest.raises(ValueError, match="a tension or a shear"):
        law_start(law.responses[1:], ["confined11"], "mises")


def test_identification_stopped():
    times, stresses = _reference()
    event = threading.Event()
    event.set()
    with stoppable(event), pytest.raises(Stopped):
        from morphogen.stopping import checkpoint

        identify(TRUE, CYCLE, times, stresses, ["yield"], callback=lambda k, e: checkpoint())


def test_parameters_named():
    config = {"young": 1.0, "poisson": 0.3, "plasticity": {
        "yield": 1.0, "kinematic": [{"C": 10.0, "D": 1.0}, {"C": 5.0, "D": 0.0}],
        "norton": {"K": 2.0, "n": 5.0}}}  # fmt: skip
    assert get_parameter(config, "C2") == 5.0 and get_parameter(config, "n") == 5.0
    set_parameter(config, "D", 3.0)
    assert config["plasticity"]["kinematic"][0]["D"] == 3.0
    times, stresses = _reference()
    with pytest.raises(ValueError, match="2 kinematic terms"):
        get_parameter(config, "C3")
    with pytest.raises(ValueError, match="one of young"):
        get_parameter(config, "tau0")
    with pytest.raises(ValueError, match="no Norton flow"):
        get_parameter(TRUE, "K")
    with pytest.raises(ValueError, match="no von Mises plasticity"):
        get_parameter({"young": 1.0, "poisson": 0.3}, "yield")
    with pytest.raises(ValueError, match="positive starting value"):
        identify({**TRUE, "plasticity": {"yield": 0.002}}, CYCLE, times, stresses, ["hardening"])
    with pytest.raises(ValueError, match="at least one"):
        identify(TRUE, CYCLE, times, stresses, [])
    with pytest.raises(ValueError, match="its index"):
        identify(TRUE, CYCLE, times, stresses[:, 0], ["yield"])
    with pytest.raises(ValueError, match="at least two times"):
        identify(TRUE, CYCLE, times[:1], stresses[:1], ["yield"])
    with pytest.raises(ValueError, match="at least one response"):
        identify_references(TRUE, [], ["yield"])
    with pytest.raises(ValueError, match="weight of a reference"):
        identify_references(TRUE, [Reference(CYCLE, times, stresses, weight=0.0)], ["yield"])
    with pytest.raises(ValueError, match="indices from 0 to 5"):
        identify(TRUE, CYCLE, times, stresses, ["yield"], components=[6])
    with pytest.raises(ValueError, match="its 6 components"):
        identify(TRUE, CYCLE, times, stresses[:, :3], ["yield"], components=[0, 1])
    # the columns of the components compared, in their order
    two = identify(TRUE, CYCLE, times, stresses[:, [1, 0]], ["yield"], components=[1, 0])
    assert two.components == [1, 0] and two.error < 1e-7
    assert np.isfinite(get_parameter(TRUE, "b"))
