"""Brittle fracture of voxel images by the phase-field method in finite strains.

Closed forms: the homogeneous response of a cell of rubber or of an elastic solid
stretched to 1.5 (the damage of ``AT2`` from its strain energy, the nominal stress of
the sound solid times the degradation), in uniaxial strain and in uniaxial stress; a
dilatation that breaks and a compression that does not. Then cells that break: a cracked
plate at small strains against the solver of the small strains, and a cracked strip of
rubber that breaks at the stretch of Rivlin and Thomas, where the energy of the stretched
strip per unit length of crack is the toughness.
"""

import numpy as np
import pytest

import morphogen as mg
from morphogen.fracture import (
    FinitePhaseFieldProblem,
    PhaseFieldProblem,
    PhaseFieldResult,
    phase_field_fracture,
)
from morphogen.homogenization import Material, homogenize
from morphogen.homogenization.materials import materials_from_config
from morphogen.homogenization.nonlinear import simulate_point

E, NU = 1.0, 0.3
GC, LENGTH = 0.2, 0.5
ELASTIC = Material(young=E, poisson=NU)
LAM, MU = ELASTIC.lame
BULK = LAM + 2.0 * MU / 3.0
RUBBER = materials_from_config(
    {0: {"young": E, "poisson": NU, "hyperelasticity": {"model": "neo_hookean"}}}
)[0]
SHEARS = {c: [0.0, 0.0] for c in ("23", "13", "12")}


def cube(n=3):
    return mg.VoxelImage(np.zeros((n, n, n), dtype=np.uint8), 1.0 / n, periodic=True)


def neo_hookean(stretch):
    """The isochoric and volumetric energies of the rubber in a uniaxial strain: ``mu / 2
    (J^-2/3 I1 - 3)`` and ``K / 2 (J - 1)^2`` with ``J`` the stretch."""
    lam = np.asarray(stretch, dtype=float)
    return 0.5 * MU * (lam ** (-2.0 / 3.0) * (lam**2 + 2.0) - 3.0), 0.5 * BULK * (lam - 1.0) ** 2


@pytest.mark.parametrize(
    ("material", "split"),
    [
        (RUBBER, "none"),
        (RUBBER, "volumetric"),
        (ELASTIC, "none"),
        (ELASTIC, "volumetric"),
        (ELASTIC, "spectral"),
    ],
)
def test_homogeneous_cell_follows_the_closed_forms(material, split):
    """A cell stretched to 1.5 in uniaxial strain: the damage ``2 psi / (G_c / l + 2 psi)``
    of its strain energy (AT2: all of it drives the damage in tension, whatever the split)
    and the nominal stress of the sound solid times ``(1 - d)^2``."""
    fracture = {"toughness": GC, "length": LENGTH, "model": "AT2", "split": split,
                "residual": 1e-9}  # fmt: skip
    path = {"times": [0, 1], "increments": 10,
            "strain": {"11": [0, 0.5], "22": [0, 0], "33": [0, 0], **SHEARS}}  # fmt: skip
    problem = FinitePhaseFieldProblem(cube(), {0: material}, fracture, staggered_tol=1e-9)
    res = problem.run(path, fields=True)
    assert isinstance(res, PhaseFieldResult) and res.finite and res.n_steps == 10
    stretch = 1.0 + res.strain[:, 0]
    if material is RUBBER:
        psi = sum(neo_hookean(stretch))
    else:  # Hencky: (lambda / 2 + mu) of the square of the logarithmic strain
        psi = (0.5 * LAM + MU) * np.log(stretch) ** 2
    damage = 2.0 * psi / (GC / LENGTH + 2.0 * psi)
    np.testing.assert_allclose(res.damage, damage, atol=1e-8)
    assert 0.3 < damage[-1] < 0.6
    sound = simulate_point(material, path, kinematics="finite")
    np.testing.assert_allclose(res.stress, (1.0 - damage[:, None]) ** 2 * sound.stress, atol=1e-7)
    # the energies of the cell: the degraded strain energy, and G_c d^2 / (2 l)
    np.testing.assert_allclose(res.elastic_energy, (1.0 - damage) ** 2 * psi, atol=1e-7)
    np.testing.assert_allclose(res.fracture_energy, GC * damage**2 / (2 * LENGTH), atol=1e-8)
    np.testing.assert_allclose(res.crack_surface, damage**2 / (2 * LENGTH), atol=1e-8)
    # the true measures of the path, and the damage with the fields of the voxels
    np.testing.assert_allclose(res.log_strain[:, 0], np.log(stretch), atol=1e-9)
    np.testing.assert_allclose(res.cauchy[:, 0], res.stress[:, 0], atol=1e-7)  # J = stretch
    last = res.fields[1.0]
    np.testing.assert_allclose(last["damage"], damage[-1], atol=1e-8)
    assert last["damage"].shape == (3, 3, 3) and last["stress"].shape == (3, 3, 3, 6)
    assert res.phases[0]["plastic"] == pytest.approx(damage, abs=1e-8)
    # the strain is prescribed: the damage of a step is found at once, and confirmed
    assert (res.iterations == 2).all() and "newton_iterations" in res.info


def test_uniaxial_stress():
    """A bar of rubber pulled with free sides: its sides contract as those of the sound
    rubber (the degradation scales all the stresses), its energy is the work of the sound
    rubber, and its nominal stress that of the sound rubber times ``(1 - d)^2``."""
    fracture = {"toughness": GC, "length": LENGTH, "model": "AT2", "split": "volumetric",
                "residual": 1e-9}  # fmt: skip
    path = {"times": [0, 1], "increments": 8, "strain": {"11": [0, 0.6]}}
    res = FinitePhaseFieldProblem(cube(2), {0: RUBBER}, fracture, staggered_tol=1e-9).run(path)
    fine = simulate_point(RUBBER, {**path, "increments": 400}, kinematics="finite")
    work = np.concatenate(
        [
            [0.0],
            np.cumsum(
                0.5 * (fine.stress[1:, 0] + fine.stress[:-1, 0]) * np.diff(fine.strain[:, 0])
            ),
        ]
    )
    psi = np.interp(res.strain[:, 0], fine.strain[:, 0], work)
    damage = 2.0 * psi / (GC / LENGTH + 2.0 * psi)
    np.testing.assert_allclose(res.damage, damage, atol=2e-6)
    sound = np.interp(res.strain[:, 0], fine.strain[:, 0], fine.stress[:, 0])
    np.testing.assert_allclose(res.stress[:, 0], (1.0 - damage) ** 2 * sound, atol=2e-6)
    np.testing.assert_allclose(res.stress[:, 1:], 0.0, atol=1e-6)  # the free components
    lateral = np.interp(res.strain[:, 0], fine.strain[:, 0], fine.strain[:, 1])
    np.testing.assert_allclose(res.strain[:, 1], lateral, atol=1e-6)
    assert res.strain[-1, 1] < -0.1 and damage[-1] > 0.25


@pytest.mark.parametrize("material", [RUBBER, ELASTIC])
def test_a_compression_breaks_only_without_a_split(material):
    """A hydrostatic compression to J = 0.51 stores a volumetric energy only: it drives
    the damage when the energy is not split, and none of it with the volumetric split (nor
    with the spectral one of the elastic solid); a dilatation breaks in both cases."""
    for sign in (-1.0, 1.0):
        strain = {c: [0, 0.2 * sign] for c in ("11", "22", "33")}
        path = {"times": [0, 1], "increments": 4, "strain": {**strain, **SHEARS}}
        j = (1.0 + 0.2 * sign) ** 3
        if material is RUBBER:
            energy = 0.5 * BULK * (j - 1.0) ** 2
        else:
            energy = 0.5 * BULK * np.log(j) ** 2
        whole = 2.0 * energy / (GC / LENGTH + 2.0 * energy)
        splits = (
            ("none", "volumetric") if material is RUBBER else ("none", "volumetric", "spectral")
        )
        for split in splits:
            fracture = {"toughness": GC, "length": LENGTH, "model": "AT2", "split": split}
            res = FinitePhaseFieldProblem(cube(2), {0: material}, fracture).run(path)
            expected = whole if (split == "none" or sign > 0) else 0.0
            assert res.damage[-1] == pytest.approx(expected, abs=1e-6), (split, sign)
    assert whole > 0.05


def cracked_plate(n):
    image = mg.VoxelImage(np.zeros((n, n, 1), dtype=np.uint8), 1.0 / n, periodic=True)
    initial = np.zeros((n, n, 1), dtype=bool)
    initial[: n // 4, n // 2, 0] = True
    return image, initial


def test_small_strains_are_those_of_the_small_strain_solver():
    """A cracked plate of a brittle solid that breaks at a strain of 0.1 %: the strength,
    the strain at which it breaks and the area of its crack are those of the phase field
    of the small strains (within the strains themselves, and the F-bar voxels of the
    nonlinear solver: softer at the tip of the crack than the fully integrated ones)."""
    n = 16
    image, initial = cracked_plate(n)
    length = 2.0 / n
    toughness = 0.002**2 * 8 * length / 3  # a strength of 0.002 E without the crack
    fracture = {"toughness": toughness, "length": length, "model": "AT1", "split": "spectral"}
    path = {"times": [0, 1], "increments": 32,
            "strain": {"22": [0, 0.004], "11": [0, 0], "33": [0, 0], **SHEARS}}  # fmt: skip
    small = PhaseFieldProblem(image, {0: ELASTIC}, fracture, initial=initial).run(path)
    finite = phase_field_fracture(image, {0: ELASTIC}, fracture, path, initial=initial,
                                  kinematics="finite")  # fmt: skip
    assert finite.finite and not small.finite
    s_small, s_finite = small.stress[:, 1], finite.stress[:, 1]
    assert s_finite.max() == pytest.approx(s_small.max(), rel=0.04)
    assert abs(int(np.argmax(s_finite)) - int(np.argmax(s_small))) <= 1
    # both break across the cell: no stress left, one crack of the area of the section
    assert s_small[-1] < 0.01 * s_small.max() and s_finite[-1] < 0.01 * s_finite.max()
    through = 1.0 + 3.0 / (8.0 * 2.0)  # 1 + 3 h / (8 l)
    assert small.crack_surface[-1] == pytest.approx(through, rel=0.02)
    assert finite.crack_surface[-1] == pytest.approx(small.crack_surface[-1], rel=0.01)
    assert finite.damage[-1] == pytest.approx(small.damage[-1], rel=0.01)
    np.testing.assert_allclose(finite.stress[:5, 1], small.stress[:5, 1], rtol=0.01)


@pytest.mark.slow
def test_a_cracked_strip_of_rubber_breaks_at_the_stretch_of_rivlin_and_thomas():
    """The pure shear test of a rubber (Rivlin & Thomas 1953): a strip of height H with a
    long crack, stretched across it with its length held. The energy released per unit
    area of crack is ``W(lambda) H``, ``W`` the energy of the strip stretched far from the
    crack: with the toughness set for a stretch of 1.5 (that of the voxels, ``G_c (1 + 3 h
    / (8 l))``), the crack starts shortly before and has crossed the cell shortly after.
    The voxels are fully integrated (a rubber of Poisson's ratio 0.4 does not lock): where
    the crack runs at a fixed stretch, the F-bar voxels of its band, sheared far, turn
    inside out under Newton's corrections, and the step is divided to no avail."""
    nx, ny, critical = 48, 16, 1.5
    h = 1.0 / ny
    length = 2.0 * h
    rubber = materials_from_config(
        {0: {"young": 1.0, "poisson": 0.4, "hyperelasticity": {"model": "neo_hookean"}}}
    )
    held = {"11": [0, 0], **SHEARS}
    far = {"times": [0, 1], "increments": 200, "strain": {"22": [0, 1.0], **held},
           "stress": {"33": [0, 0]}}  # fmt: skip
    point = simulate_point(rubber[0], far, kinematics="finite")
    stretch = 1.0 + point.strain[:, 1]
    work = np.concatenate(
        [[0.0], np.cumsum(0.5 * (point.stress[1:, 1] + point.stress[:-1, 1]) * np.diff(stretch))]
    )
    factor = 1.0 + 3.0 * h / (8.0 * length)
    toughness = float(np.interp(critical, stretch, work)) / factor  # W(1.5) H = G_c factor
    image = mg.VoxelImage(np.zeros((nx, ny, 1), dtype=np.uint8), h, periodic=True)
    initial = np.zeros((nx, ny, 1), dtype=bool)
    initial[: nx // 2, ny // 2, 0] = True
    fracture = {"toughness": toughness, "length": length, "model": "AT1", "split": "volumetric"}
    path = {"times": [0, 1], "increments": 30, "strain": {"22": [0, 0.7], **held},
            "stress": {"33": [0, 0]}}  # fmt: skip
    problem = FinitePhaseFieldProblem(image, rubber, fracture, initial=initial, integration="full")
    res = problem.run(path)
    stress = res.stress[:, 1]
    peak = int(np.argmax(stress))
    start = 1.0 + res.strain[peak, 1]
    broken = 1.0 + res.strain[np.flatnonzero(stress[peak:] < 0.1 * stress[peak])[0] + peak, 1]
    assert 0.94 * critical < start < critical < broken < 1.05 * critical
    assert res.crack_surface[-1] == pytest.approx(factor, rel=0.04)  # across the cell
    assert res.crack_surface[0] == pytest.approx(0.5 * factor, rel=0.2)
    # the sides contract while the strip stretches, and come back when it is broken
    assert res.strain[peak, 2] < -0.1 and abs(res.strain[-1, 2]) < 0.05


def test_finite_phase_fields_are_checked():
    fracture = {"toughness": GC, "length": LENGTH, "model": "AT2"}
    with pytest.raises(ValueError, match="spectral split"):  # the default split
        FinitePhaseFieldProblem(cube(2), {0: RUBBER}, fracture)
    plastic = Material(young=E, poisson=NU, plasticity={"yield": 0.1})
    with pytest.raises(ValueError, match=r"elastic \(isotropic\) or hyperelastic"):
        FinitePhaseFieldProblem(cube(2), {0: plastic}, {**fracture, "split": "none"})
    for option in ("device", "kinematics", "grains"):
        with pytest.raises(ValueError, match="no option"):
            FinitePhaseFieldProblem(cube(2), {0: ELASTIC}, fracture, **{option: "cpu"})
    problem = FinitePhaseFieldProblem(cube(2), {0: ELASTIC}, fracture)
    heated = {"times": [0, 1], "strain": {"11": [0, 0.1]}, "temperature": [0, 10.0]}
    with pytest.raises(ValueError, match="temperature"):
        problem.run(heated)
    with pytest.raises(ValueError, match="kinematics"):
        phase_field_fracture(cube(2), {0: ELASTIC}, fracture, heated, kinematics="large")
    # a phase that does not break may have any behaviour: a plastic matrix around it
    data = np.zeros((2, 2, 2), dtype=np.uint8)
    data[0] = 1
    image = mg.VoxelImage(data, 0.5, periodic=True)
    both = FinitePhaseFieldProblem(
        image, {0: plastic, 1: ELASTIC}, {**fracture, "toughness": {1: GC}, "split": "none"}
    )
    res = both.run({"times": [0, 1], "increments": 3, "strain": {"11": [0, 0.3]}})
    assert res.damage[-1] > 0.05 and res.phases[1]["plastic"][-1] > 0.05
    assert both.damage.breakable.sum() == 4  # the voxels of the elastic phase alone


def test_finite_strains_in_homogenize_and_configurations():
    """``nonlinear: {kinematics: finite}`` with a ``fracture``: the phase field in finite
    strains, in ``homogenize`` and in the configurations (which refused it)."""
    from morphogen.config import load_config

    fracture = {"toughness": GC, "length": LENGTH, "model": "AT2", "split": "none",
                "staggered_tol": 1e-8}  # fmt: skip
    path = {"times": [0, 1], "increments": 4,
            "strain": {"11": [0, 0.4], "22": [0, 0], "33": [0, 0], **SHEARS}}  # fmt: skip
    out = homogenize(cube(2), {0: RUBBER}, properties=[], paths={"pull": path},
                     fracture=fracture, nonlinear={"kinematics": "finite"})  # fmt: skip
    res = out["pull.periodic"]
    assert isinstance(res, PhaseFieldResult) and res.finite
    psi = sum(neo_hookean(1.4))
    assert res.damage[-1] == pytest.approx(2 * psi / (GC / LENGTH + 2 * psi), abs=1e-7)
    small = homogenize(cube(2), {0: RUBBER}, properties=[], paths={"pull": path},
                       fracture=fracture)["pull.periodic"]  # fmt: skip
    assert not small.finite and small.damage[-1] != pytest.approx(res.damage[-1], abs=1e-3)
    cfg = load_config(
        {
            "model": {"type": "boolean_spheres", "radius": 0.3, "volume_fraction": 0.2},
            "domain": {"size": 1.0, "periodic": True},
            "voxels": {"shape": 4},
            "homogenization": {
                "method": "fft",
                "materials": {
                    0: {"young": 1.0, "poisson": 0.3, "hyperelasticity": {"model": "neo_hookean"}},
                    1: {"young": 10.0, "poisson": 0.3},
                },
                "properties": [],
                "paths": {"pull": path},
                "fracture": {"toughness": {0: GC}, "length": LENGTH, "split": "volumetric"},
                "nonlinear": {"kinematics": "finite"},
            },
        }
    )
    assert cfg.homogenization.nonlinear["kinematics"] == "finite"
    assert cfg.homogenization.fracture["split"] == "volumetric"
