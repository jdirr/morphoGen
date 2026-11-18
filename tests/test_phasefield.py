"""Brittle fracture of voxel images by the phase-field method.

Closed forms: the homogeneous response of a cell (the damage of ``AT2`` from the first
load, the elastic limit of ``AT1``), the optimal profiles of a crack (the exponential of
``AT2``, the parabola of finite width of ``AT1``) and its regularized area, whose known
excess on linear elements is ``1 + h / (2 l)`` and ``1 + 3 h / (8 l)``; then cells that
break: a bar whose fracture energy does not depend on its voxels, a slit that crosses its
cell near the stress of Griffith, compression that does not break, an inclusion that
does not.
"""

import numpy as np
import pytest

import morphogen as mg
from morphogen.fracture import PhaseField, PhaseFieldProblem, PhaseFieldResult, phase_field_fracture
from morphogen.homogenization import VOID, Material, homogenize

E, NU, GC = 100.0, 0.25, 0.01
STEEL = Material(young=E, poisson=NU)
LAM, MU = STEEL.lame
#: The modulus of a uniaxial strain
OEDOMETRIC = LAM + 2.0 * MU
FULL = {c: [0.0, 0.0] for c in ("11", "22", "33", "23", "31", "12")}


def bar(n, weak=None):
    data = np.zeros((1, 1, n), dtype=np.uint8)
    if weak is not None:
        data[0, 0, weak] = 1
    return mg.VoxelImage(data, 1.0 / n, periodic=True)


@pytest.mark.parametrize("split", ["none", "spectral", "volumetric"])
@pytest.mark.parametrize("model", ["AT2", "AT1"])
def test_homogeneous_cell_follows_the_closed_forms(model, split):
    """A uniaxial extension (all its energy tensile): ``d = 2 psi / (G_c / l + 2 psi)``
    (``AT2``); no damage below ``psi = 3 G_c / (16 l)`` then ``d = 1 - 3 G_c / (16 l
    psi)`` (``AT1``); the stress ``(1 - d)^2 M eps``."""
    length = 0.1
    pf = PhaseField(GC, length, model=model, split=split)
    image = mg.VoxelImage(np.zeros((3, 3, 3), dtype=np.uint8), 0.05, periodic=True)
    limit = np.sqrt(3.0 * GC / (8.0 * length * OEDOMETRIC))  # the elastic limit of AT1
    path = {"times": [0, 1.0, 2.0], "increments": [1, 12],
            "strain": {**{c: [0.0] * 3 for c in FULL}, "11": [0, limit, 2.5 * limit]}}  # fmt: skip
    r = PhaseFieldProblem(image, {0: STEEL}, pf, staggered_tol=1e-10, max_staggered=400).run(path)
    assert isinstance(r, PhaseFieldResult)
    e = r.strain[:, 0]
    psi = 0.5 * OEDOMETRIC * e**2
    if model == "AT2":
        d = 2.0 * psi / (GC / length + 2.0 * psi)
    else:
        threshold = 3.0 * GC / (16.0 * length)
        d = np.where(psi > threshold, 1.0 - threshold / np.maximum(psi, 1e-300), 0.0)
    assert np.allclose(r.damage, d, rtol=1e-8, atol=1e-10)
    assert np.allclose(r.stress[:, 0], ((1 - d) ** 2 + pf.residual) * OEDOMETRIC * e, rtol=1e-8)
    assert np.allclose(r.plastic, r.damage) and np.allclose(r.phases[0]["plastic"], d, atol=1e-10)
    # the energies: the stored one degraded, the fracture one G_c gamma(d)
    gamma = d**2 / (2 * length) if model == "AT2" else 0.375 * d / length
    assert np.allclose(r.crack_surface, gamma, rtol=1e-8, atol=1e-12)
    assert np.allclose(r.fracture_energy, GC * gamma, rtol=1e-8, atol=1e-14)
    assert np.allclose(r.elastic_energy, ((1 - d) ** 2 + pf.residual) * psi, rtol=1e-8)
    peak = pf.critical_stress(OEDOMETRIC) * (1.0 + pf.residual)
    if model == "AT1":  # the stress at the elastic limit is the strength of the cell
        assert r.strength() == pytest.approx(peak, rel=1e-9)
        assert r.stress[1, 0] == pytest.approx(r.strength())
    else:
        assert r.strength() < peak * (1 + 1e-12)
        assert pf.critical_stress(OEDOMETRIC) == pytest.approx(
            9.0 / 16.0 * np.sqrt(OEDOMETRIC * GC / (3 * length))
        )


def test_uniaxial_stress_is_condensed():
    """A tension whose lateral stresses are free: the cell of the split ``none`` keeps
    its Poisson contraction and follows the closed form of its Young's modulus."""
    length = 0.1
    pf = PhaseField(GC, length, split="none")
    image = mg.VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8), 0.05, periodic=True)
    peak = np.sqrt(GC / (3.0 * length * E))
    r = phase_field_fracture(image, {0: STEEL}, pf, {"times": [0, 1.0], "increments": 8,
                                                     "strain": {"33": [0, 2.0 * peak]}},
                             staggered_tol=1e-10, max_staggered=400)  # fmt: skip
    e = r.strain[:, 2]
    d = E * e**2 / (GC / length + E * e**2)
    assert np.allclose(r.stress[:, 2], ((1 - d) ** 2 + pf.residual) * E * e, rtol=1e-7)
    assert np.abs(r.stress[:, [0, 1, 3, 4, 5]]).max() < 1e-9
    assert np.allclose(r.strain[:, 0], -NU * e, rtol=1e-7)
    assert r.strength() == pytest.approx(pf.critical_stress(E), rel=2e-2)
    assert r.driven() == 2 and r.value("strength") == r.strength()
    assert r.value("work") == pytest.approx(np.trapezoid(r.stress[:, 2], e))


@pytest.mark.parametrize("model", ["AT2", "AT1"])
def test_optimal_profile_and_area_of_a_crack(model):
    """A plane of broken voxels across a bar, without load: the damage decays as ``exp(-x
    / l)`` (``AT2``) or ``(1 - x / (2 l))^2`` up to ``2 l`` (``AT1``: an obstacle
    problem), and its regularized area per unit section is 1 for the two half profiles,
    and ``h / (2 l)`` (``AT2``) or ``3 h / (8 l)`` (``AT1``) for the broken voxels between
    them: the known excess of a crack one element wide."""
    n, length = 80, 0.1
    h = 1.0 / n
    broken = np.zeros((1, 1, n), dtype=bool)
    broken[0, 0, n // 2] = True
    problem = PhaseFieldProblem(bar(n), {0: STEEL}, PhaseField(GC, length, model=model),
                                initial=broken, tol=1e-11)  # fmt: skip
    x = np.arange(0, 12) * h
    d = problem.d[0, 0, n // 2 + 1 : n // 2 + 13]
    assert problem.d[0, 0, n // 2] == 1.0 and problem.d[0, 0, n // 2 + 1] == 1.0
    exact = np.exp(-x / length) if model == "AT2" else np.maximum(1 - x / (2 * length), 0) ** 2
    assert np.allclose(d, exact, atol=2e-3)
    if model == "AT1":  # a finite width: sound beyond 2 l
        assert problem.d[0, 0, 0] == 0.0 and problem.d[0, 0, n // 2 + 20] == 0.0
    area = problem.crack_surface() / (h * h)
    # the two half profiles (linear elements), and the broken voxels of width h between
    band = h / (2 * length) if model == "AT2" else 3 * h / (8 * length)
    assert area == pytest.approx(1.0 + band, rel=5e-3)


def test_the_fracture_energy_of_a_bar_does_not_depend_on_its_voxels():
    """A bar pulled until its weakest voxel breaks (``AT1``: elastic until then): the
    energy of its crack is ``G_c (1 + 3 h / (8 l))`` per unit section on 40 voxels as on
    80; the history field of Miehe et al. widens the crack and over-estimates it."""
    length = 0.05
    soft = Material(young=0.9 * E, poisson=0.0)
    mats = {0: Material(young=E, poisson=0.0), 1: soft}
    pf = PhaseField(GC, length, model="AT1", split="none")
    peak = pf.critical_stress(0.9 * E)
    path = {"times": [0, 1.0], "increments": 30, "strain": {"33": [0, 2.0 * peak / E]}}
    energies = {}
    for n in (40, 80):
        r = phase_field_fracture(bar(n, weak=n // 2), mats, pf, path, staggered_tol=1e-4,
                                 max_staggered=600, fields=True)  # fmt: skip
        assert 0.95 * peak < r.strength() < 1.01 * peak
        assert abs(r.stress[-1, 2]) < 1e-3 * peak
        energies[n] = r.fracture_energy[-1] / GC  # the bar has a unit length
        assert energies[n] == pytest.approx(1.0 + 3.0 / (8.0 * length * n), rel=1e-2)
        damage = r.fields[1.0]["damage"][0, 0]
        assert damage.max() > 0.99 and damage[2] == 0.0  # one crack, sound far from it
        # broken at once: the work done includes the energy stored at the peak, released
        assert r.work() > r.fracture_energy[-1]
    assert energies[80] < energies[40] < 1.2
    history = PhaseField(GC, length, model="AT1", split="none", irreversibility="history")
    wide = phase_field_fracture(bar(40, weak=20), mats, history, path, staggered_tol=1e-4,
                                max_staggered=600)  # fmt: skip
    assert wide.fracture_energy[-1] / GC > energies[40] * 1.05


def test_compression_breaks_only_without_a_split():
    image = mg.VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8), 0.05, periodic=True)
    normal = ("11", "22", "33")
    squeeze = {"times": [0, 1.0], "increments": 4,
               "strain": {c: [0, -0.02 if c in normal else 0.0] for c in FULL}}  # fmt: skip
    for split, broken in (("none", True), ("spectral", False), ("volumetric", False)):
        r = phase_field_fracture(image, {0: STEEL}, PhaseField(GC, 0.1, split=split), squeeze)
        assert (r.damage[-1] > 0.5) == broken
        if not broken:
            bulk = LAM + 2.0 * MU / 3.0
            assert r.stress[-1, 0] == pytest.approx(-0.06 * bulk * (1 + 1e-6), rel=1e-9)
    # a shear breaks with every split
    shear = {"times": [0, 1.0], "increments": 4, "strain": {**FULL, "12": [0, 0.08]}}
    for split in ("spectral", "volumetric"):
        r = phase_field_fracture(image, {0: STEEL}, PhaseField(GC, 0.1, split=split), shear)
        assert r.damage[-1] > 0.5


@pytest.mark.slow
def test_a_slit_crosses_its_cell_near_the_stress_of_griffith():
    """A periodic row of collinear slits of half the period, pulled across (plane strain):
    the cell breaks along the row, its regularized crack length close to the period, at a
    stress below that of Griffith for the row, ``sqrt(E' G_c / (W tan(pi a / W)))``, as
    the cracks regularized over a length of a sixth of the slit do."""
    n = 32
    h = 1.0 / n
    length = 2.0 * h
    image = mg.VoxelImage(np.zeros((n, n, 1), dtype=np.uint8), h, periodic=True)
    broken = np.zeros((n, n, 1), dtype=bool)
    broken[n // 4 : 3 * n // 4, n // 2, 0] = True
    material = Material(young=E, poisson=0.3)
    pf = PhaseField(GC, length, model="AT1")
    problem = PhaseFieldProblem(image, {0: material}, pf, initial=broken, staggered_tol=2e-3,
                                max_staggered=600)  # fmt: skip
    path = {"times": [0, 1.0], "increments": 30, "strain": {"22": [0, 0.03]}}
    r = problem.run(path, fields=True)
    griffith = np.sqrt(E / (1 - 0.3**2) * GC / np.tan(np.pi * 0.25))
    assert 0.7 * griffith < r.strength() < griffith
    assert abs(r.stress[-1, 1]) < 0.02 * r.strength()
    damage = r.fields[1.0]["damage"][:, :, 0]
    assert np.all(damage[:, n // 2] > 0.9)  # the row of the slits, across the cell
    assert damage[:, : n // 2 - 6].max() < 1e-6  # and nothing away from it
    # the regularized length of the crack: the period, with the excess of linear elements
    assert r.crack_surface[-1] == pytest.approx(1.0 + 3 * h / (8 * length), rel=0.08)


def test_tough_inclusions_and_voids():
    """A phase without a toughness does not break and is not degraded; a pore neither
    carries nor cracks; the mean damage is that of the phases that break."""
    n = 12
    data = np.zeros((n, n, 1), dtype=np.uint8)
    data[4:8, 4:8] = 1
    image = mg.VoxelImage(data, 1.0 / n, periodic=True)
    pf = PhaseField({0: GC}, 2.0 / n, split="none")
    assert pf.of(0) == GC and pf.of(1) is None
    stiff = Material(young=5 * E, poisson=NU)
    problem = PhaseFieldProblem(image, {0: STEEL, 1: stiff}, pf, staggered_tol=1e-3)
    path = {"times": [0, 1.0], "increments": 10, "strain": {"11": [0, 0.03]}}
    r = problem.run(path, fields=True)
    g = problem.degradation()
    assert np.all(g[data == 1] == 1.0) and g[data == 0].min() < 0.05
    assert r.phases[0]["plastic"][-1] > 0.2
    assert r.damage[-1] == pytest.approx(r.phases[0]["plastic"][-1])
    assert np.abs(r.stress[-1]).max() < 0.2 * r.strength()
    porous = PhaseFieldProblem(image, {0: STEEL, 1: VOID}, PhaseField(GC, 2.0 / n))
    out = porous.run({"times": [0, 1.0], "increments": 3, "strain": {"11": [0, 1e-3]}})
    assert 0.0 < out.stress[-1, 0] < OEDOMETRIC * 1e-3 and set(out.phases) == {0}
    with pytest.raises(ValueError, match="no phase with a toughness"):
        PhaseFieldProblem(image, {0: STEEL, 1: VOID}, PhaseField({1: GC}, 2.0 / n))


def test_phase_fields_are_checked():
    pf = PhaseField.from_config({"toughness": {"0": 1.0, 1: 2.0}, "length": 0.1, "model": "AT1"})
    assert pf.of(1) == 2.0 and PhaseField.from_config(pf) is pf
    for bad, match in (
        ({"toughness": -1.0, "length": 0.1}, "toughness"),
        ({"toughness": {}, "length": 0.1}, "toughness"),
        ({"toughness": 1.0, "length": 0.0}, "length"),
        ({"toughness": 1.0, "length": 0.1, "model": "AT3"}, "model"),
        ({"toughness": 1.0, "length": 0.1, "split": "hybrid"}, "split"),
        ({"toughness": 1.0, "length": 0.1, "residual": 0.5}, "residual"),
        ({"toughness": 1.0, "length": 0.1, "irreversibility": "none"}, "irreversibility"),
        ({"toughness": 1.0, "length": 0.1, "width": 2}, "unknown"),
    ):
        with pytest.raises(ValueError, match=match):
            PhaseField.from_config(bad)
    with pytest.raises(ValueError, match="one toughness"):
        pf.critical_stress(E)
    image = mg.VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8), 0.05, periodic=True)
    crystal = Material.cubic(168.0, 121.0, 75.0)
    with pytest.raises(ValueError, match="isotropic"):
        PhaseFieldProblem(image, {0: crystal}, PhaseField(GC, 0.1))
    PhaseFieldProblem(image, {0: crystal}, PhaseField(GC, 0.1, split="none"))  # crystals: no split
    with pytest.raises(ValueError, match="shape of the image"):
        PhaseFieldProblem(image, {0: STEEL}, PhaseField(GC, 0.1), initial=np.zeros((3, 3, 3)))
    with pytest.raises(ValueError, match="fields"):
        PhaseFieldProblem(image, {0: STEEL}, PhaseField(GC, 0.1)).run(
            {"times": [0, 1], "increments": 1, "strain": {"11": [0, 1e-3]}}, fields="all"
        )


def test_fracture_in_homogenize_and_configurations(tmp_path):
    import json

    from morphogen import pipeline
    from morphogen.config import ConfigError, load_config

    length = 0.05
    mats = {0: Material(young=E, poisson=0.0), 1: Material(young=0.9 * E, poisson=0.0)}
    path = {"times": [0, 1.0], "increments": 20, "strain": {"33": [0, 0.03]}}
    fracture = {"toughness": GC, "length": length, "model": "AT1", "split": "none",
                "staggered_tol": 1e-4, "max_staggered": 600}  # fmt: skip
    out = homogenize(bar(40, weak=20), mats, properties=[], paths={"pull": path},
                     fracture=fracture, fields=True)  # fmt: skip
    r = out["pull.periodic"]
    assert isinstance(r, PhaseFieldResult) and r.name == "pull" and r.bc == "periodic"
    assert r.value("fracture_energy") == pytest.approx(GC * (1 + 3 / (8 * length * 40)), rel=1e-2)
    assert r.fields[1.0]["damage"].shape == (1, 1, 40)
    from helpers import box_mesh

    with pytest.raises(ValueError, match="voxel images"):  # the phase field: on images
        homogenize(box_mesh(), {0: STEEL}, properties=[], paths={"pull": path},
                   fracture=fracture, bcs=["periodic"])  # fmt: skip
    # a study of the strength and of the work of fracture of random laminates
    data = {
        "model": {"type": "random_laminate", "thickness": 0.4, "spacing": 0.6},
        "domain": {"periodic": True},
        "seed": 4,
        "voxels": {"voxel_size": 0.125},
        "homogenization": {
            "method": "fft",
            "materials": {0: {"young": E, "poisson": 0.0}, 1: {"young": 3 * E, "poisson": 0.0}},
            "properties": [],
            "paths": {"pull": {"times": [0, 1.0], "increments": 10, "strain": {"33": [0, 0.04]}}},
            "fracture": {"toughness": {0: GC, 1: 4 * GC}, "length": 0.25, "model": "AT1",
                         "split": "none"},
        },
        "rve": {"sizes": [1.0, 1.5], "realizations": [3, 2], "bootstrap": 0,
                "properties": ["pull.strength", "pull.work", "pull.fracture_energy"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "stack"},
    }  # fmt: skip
    cfg = load_config(data)
    assert cfg.homogenization.fracture["model"] == "AT1"
    outdir = pipeline.run(cfg)
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    for name in ("pull.strength.periodic", "pull.work.periodic", "pull.fracture_energy.periodic"):
        stats = report["properties"][name]["statistics"]
        assert [s["n"] for s in stats] == [3, 2] and all(s["mean"] > 0 for s in stats)
    curves = pipeline.read_rows(next(outdir.glob("stack_L1_0000_pull_periodic.csv")))
    assert {"damage", "crack_surface", "fracture_energy", "S33", "p_0"} <= set(curves[0])
    # the layers in series break in the weakest: one crack across the cell, whose energy
    # is that of the fracture and nearly all the work done
    assert curves[-1]["S33"] < 0.02 * max(c["S33"] for c in curves)
    assert 1.0 < curves[-1]["crack_surface"] < 1.4
    work = report["properties"]["pull.work.periodic"]["statistics"][0]["mean"]
    spent = report["properties"]["pull.fracture_energy.periodic"]["statistics"][0]["mean"]
    assert spent < work < 1.1 * spent
    for change, match in (
        ({"paths": {}}, "paths"),
        ({"fracture": {"toughness": GC}}, "fracture"),
        ({"fracture": {"toughness": GC, "length": 0.1, "model": "AT9"}}, "model"),
    ):
        bad = {**data, "homogenization": {**data["homogenization"], **change}}
        with pytest.raises(ConfigError, match=match):
            load_config(bad)
    # in finite strains too (tests/test_phasefield_finite.py)
    large = {**data["homogenization"], "nonlinear": {"kinematics": "finite"}}
    cfg = load_config({**data, "homogenization": large})
    assert cfg.homogenization.nonlinear == {"kinematics": "finite"}
    with pytest.raises(ConfigError, match="unknown quantity 'toughness'"):
        load_config({**data, "rve": {**data["rve"], "properties": ["pull.toughness"]}})
    # the columns of a fracture belong to the paths of a fracture only
    plain = {k: v for k, v in data["homogenization"].items() if k != "fracture"}
    with pytest.raises(ConfigError, match="unknown quantity 'crack_surface'"):
        load_config({**data, "homogenization": plain,
                     "rve": {**data["rve"], "properties": ["pull.crack_surface"]}})  # fmt: skip
    names = ["pull.strength@1", "pull.work"]
    cfg = load_config({**data, "homogenization": plain,
                       "rve": {**data["rve"], "properties": names}})  # fmt: skip
    assert cfg.rve.properties == names
