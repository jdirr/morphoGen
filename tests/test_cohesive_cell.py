"""Cells whose interfaces debond along loading paths (:mod:`morphogen.fracture.periodic`).

The interfaces opened on the torus of a periodic cell: inside the cell, on its faces (the
two sides a period apart), in a cell one element thick, with 10-node elements, between
grains; the periodic links of the two lips. A bar with one interface against its closed
forms: tension and shear along the bilinear law, the work that breaks it, the contact of
its lips in compression, Coulomb's friction and the energy that it dissipates, lips
without adhesion, the exponential law, uniaxial stress, finite strains; the viscous
regularization (its recurrence, and the interface that breaks at once). Kinematic
conditions against the solver of bodies with interfaces. Beads that debond from their
matrix: between the bonded composite and the matrix with holes. The option of
``homogenize``, of the configurations and of the command line.
"""

import json

import numpy as np
import pytest
from scipy.optimize import brentq

from morphogen import cli, io
from morphogen.config import ConfigError, load_config
from morphogen.fracture import (
    Bilinear,
    CohesiveCell,
    CohesiveProblem,
    Contact,
    Exponential,
    insert_interfaces,
    interface_law,
    open_cell,
)
from morphogen.fracture.periodic import SERIES
from morphogen.homogenization import Loading, Material, homogenize
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.nonlinear import NonlinearFEProblem
from morphogen.image import VoxelImage
from morphogen.mesh.conforming import has_mmg
from morphogen.mesh.tetra import TetMesh
from morphogen.viz.workflow import path_fields

mmg = pytest.mark.skipif(not has_mmg(), reason="MMG")

E, NU = 1000.0, 0.25
M = E * (1 - NU) / ((1 + NU) * (1 - 2 * NU))  # the modulus of a uniaxial strain
G = E / (2 * (1 + NU))
SOLID = Material(young=E, poisson=NU)
MATERIALS = {0: SOLID, 1: SOLID, 2: SOLID}
LENGTH = 1.0
ZERO = {"11": 0.0, "22": 0.0, "33": 0.0, "23": 0.0, "13": 0.0, "12": 0.0}
#: The interface inside the cell, and the one on its faces (its sides a period apart)
INSIDE, ACROSS = [(0, 1)], [(0, 2)]


def _bar(shape=(2, 2, 10), order: int = 1) -> TetMesh:
    """A periodic bar of three layers along its third axis (phases 0, 1 and 2), of length
    1: two interfaces inside the cell, one on its faces."""
    data = np.zeros(shape, np.uint8)
    n = shape[2]
    first, second = max(1, 3 * n // 10), max(2, 7 * n // 10)
    data[:, :, first:second] = 1
    data[:, :, second:] = 2
    mesh = voxel_tet_mesh(VoxelImage(data, spacing=(0.1, 0.1, LENGTH / n), periodic=True))
    return mesh if order == 1 else mesh.quadratic()


def _softening(strain, modulus, stiffness, strength, energy):
    """The stress of a bar of unit length with one bilinear interface, for its mean
    strain: elastic, softening (stable: its length over its modulus below the slope of
    the law), broken."""
    onset, failure = strength / stiffness, 2 * energy / strength
    elastic = strain * LENGTH / (LENGTH / modulus + 1 / stiffness)
    soft = (strain * LENGTH - failure) / (LENGTH / modulus - (failure - onset) / strength)
    return np.where(elastic <= strength, elastic, np.maximum(soft, 0.0))


# -- the interfaces of a cell, opened on its torus ------------------------------------------


def _sides(mesh: TetMesh) -> list:
    """The phases of the elements around each node."""
    out = [set() for _ in range(mesh.n_nodes)]
    for element, phase in zip(mesh.elements, mesh.phase, strict=True):
        for node in element:
            out[node].add(int(phase))
    return out


def _check_links(opened: TetMesh, one_side: bool = False) -> int:
    """The periodic links join translates, each to the first node of its class; with
    every interface opened (``one_side``), nodes of the same phase; their number."""
    slave, master, shift = opened.periodic_links()
    np.testing.assert_allclose(
        opened.nodes[slave], opened.nodes[master] + shift * opened.size, atol=1e-12
    )
    assert np.all(master < slave) and not np.isin(master, slave).any()
    assert len(np.unique(slave)) == len(slave)
    if one_side:
        sides = _sides(opened)
        assert all(len(side) == 1 for side in sides)
        assert all(sides[s] == sides[m] for s, m in zip(slave, master, strict=True))
    return len(slave)


def test_interfaces_opened_inside_a_cell_and_on_its_faces():
    mesh = _bar()
    # 99 nodes for 40 points of the torus (2 x 2 x 10): 59 are linked to another
    assert (mesh.n_nodes, mesh.n_elements) == (99, 240) and len(mesh.periodic_links()[0]) == 59
    # inside: the nine nodes of the interface doubled (four points of the torus), its lips
    # linked each to their own
    opened, interfaces, periods = open_cell(mesh, INSIDE)
    assert opened.n_nodes == 108 and interfaces.n_elements == 8 and not periods.any()
    assert interfaces.areas(opened).sum() == pytest.approx(0.04)
    np.testing.assert_allclose(opened.nodes[interfaces.minus], opened.nodes[interfaces.plus])
    assert not np.intersect1d(interfaces.minus, interfaces.plus).size
    assert _check_links(opened) == 59 + 9 - 4
    x = opened.nodes[interfaces.minus]
    normal = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    assert np.all(normal[:, 2] > 0) and np.all(interfaces.phases == [0, 1])  # from 0 to 1
    # on the faces of the cell: no new node, the two faces no longer linked, a period apart
    opened, interfaces, periods = open_cell(mesh, ACROSS)
    assert opened.n_nodes == 99 and interfaces.n_elements == 8
    assert np.all(periods == [0, 0, 1]) and np.all(interfaces.phases == [0, 2])
    np.testing.assert_allclose(
        opened.nodes[interfaces.plus], opened.nodes[interfaces.minus] + [0, 0, LENGTH]
    )
    assert _check_links(opened) == 59 - 4  # four points of the torus are now two
    x = opened.nodes[interfaces.minus]
    assert np.all(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])[:, 2] < 0)  # through the face
    # every interface: each node on one side
    opened, interfaces, periods = open_cell(mesh)
    assert opened.n_nodes == 117 and interfaces.n_elements == 24
    assert sorted(map(tuple, np.unique(periods, axis=0))) == [(0, 0, 0), (0, 0, 1)]
    assert _check_links(opened, one_side=True) == 59 - 4 + 10
    # without periodicity: the interfaces of insert_interfaces
    opened, interfaces, periods = open_cell(mesh, periodic=False)
    reference, faces = insert_interfaces(mesh, "phases")
    assert opened.n_nodes == reference.n_nodes == 117 and not periods.any()
    assert interfaces.n_elements == faces.n_elements == 16 and not len(opened.periodic_links()[0])


@pytest.mark.parametrize("shape", [(1, 2, 10), (1, 1, 10), (2, 2, 3)])
def test_interfaces_of_cells_one_element_thick(shape):
    """Elements that hold a node and its translate: the corners of the elements are
    grouped, not the elements."""
    mesh = _bar(shape)
    opened, interfaces, periods = open_cell(mesh)
    area = 0.01 * shape[0] * shape[1]
    assert interfaces.areas(opened).sum() == pytest.approx(3 * area)
    _check_links(opened, one_side=True)
    crossing = np.any(periods != 0, axis=1)
    assert interfaces.areas(opened)[crossing].sum() == pytest.approx(area)


def test_interfaces_of_ten_node_elements_and_of_grains():
    mesh = _bar(order=2)
    opened, interfaces, periods = open_cell(mesh)
    assert opened.order == 2 and interfaces.order == 2 and interfaces.minus.shape == (24, 6)
    assert interfaces.areas(opened).sum() == pytest.approx(0.12)
    _check_links(opened, one_side=True)
    np.testing.assert_allclose(
        opened.nodes[interfaces.plus], opened.nodes[interfaces.minus] + periods[:, None] * LENGTH
    )
    assert opened.n_elements == mesh.n_elements and np.all(opened.volumes() > 0)
    # the mid-edge nodes of the opened faces are doubled as their corners
    inside, faces, _ = open_cell(mesh, INSIDE)
    assert inside.n_nodes == mesh.n_nodes + 25 and faces.n_elements == 8  # a grid of 5 x 5
    # grains of one phase
    linear = _bar()
    grains = TetMesh(linear.nodes, linear.elements, np.zeros(linear.n_elements, np.int64),
                     linear.lower, linear.upper, periodic=True, cell=linear.phase)  # fmt: skip
    opened, interfaces, _ = open_cell(grains, "grains")
    assert opened.n_nodes == 117 and interfaces.n_elements == 24
    with pytest.raises(ValueError, match="grain of each element"):
        open_cell(linear, "grains")
    for bad in ("sides", [(0, 0)], [(0, 1, 2)], 3):
        with pytest.raises(ValueError, match="pairs of"):
            open_cell(linear, bad)
    with pytest.raises(ValueError, match="one value per shared face"):
        open_cell(linear, lambda c: np.ones(3, dtype=bool))
    # a function of the centroids: one plane of faces
    opened, interfaces, _ = open_cell(linear, lambda c: np.abs(c[:, 2] - 0.5) < 1e-9)
    assert interfaces.n_elements == 8 and np.all(interfaces.phases == 1)
    # a cell whose opposite faces differ
    data = np.zeros((2, 2, 4), np.uint8)
    data[0, 0, 3] = 1
    cut = voxel_tet_mesh(VoxelImage(data, periodic=True)).subset([0])
    with pytest.raises(ValueError, match="opposite faces"):
        open_cell(cut)


# -- a bar with one interface ------------------------------------------------------------------

LAW = Bilinear(1e5, 10.0, 8.0, 0.1, 0.12)


@pytest.mark.parametrize("faces", [INSIDE, ACROSS])
@pytest.mark.parametrize("order", [1, 2])
def test_bar_pulled_until_its_interface_breaks(faces, order):
    cell = CohesiveCell(_bar(order=order), MATERIALS, LAW, faces=faces)
    path = {"times": [0, 1, 2], "increments": [50, 10],
            "strain": {**ZERO, "33": [0, 0.025, -0.01]}}  # fmt: skip
    res = cell.run(path)
    strain, stress = res.strain[:51, 2], res.stress[:51, 2]
    exact = _softening(strain, M, 1e5, 10.0, 0.1)
    np.testing.assert_allclose(stress, exact, atol=2e-5)
    # broken at a strain of 0.02
    assert exact.max() > 9.9 and exact[40] == 0 and np.abs(stress[40:]).max() < 1e-5
    np.testing.assert_allclose(res.stress[:51, 0], NU / (1 - NU) * stress, atol=1e-6)
    # the work of the stress: the fracture energy of the interface over the length
    assert np.trapezoid(stress, strain) == pytest.approx(0.1 / LENGTH, rel=1e-3)
    series = res.series
    assert set(series) == set(SERIES) and series["debonded_area"][30] == 0
    assert series["debonded_area"][41] == pytest.approx(0.04)
    assert series["interface_damage"][0] == 0
    assert series["interface_damage"][-1] == pytest.approx(1.0)
    assert np.all(np.diff(series["interface_damage"]) >= 0)
    assert series["interface_opening"][50] == pytest.approx(0.025, rel=1e-6)
    assert not series["interface_dissipation"].any()
    # pushed back: the lips touch, the bar is whole again but for the penalty
    assert res.stress[-1, 2] == pytest.approx(-0.01 * LENGTH / (LENGTH / M + 1e-5), rel=1e-7)
    assert series["interface_opening"][-1] == 0 and series["debonded_area"][-1] > 0
    points = cell.interface_points()
    assert points["weight"].sum() == pytest.approx(0.04) and np.all(points["damage"] == 1)
    assert np.all(points["opening"] < 0) and points["position"].shape == (len(points["weight"]), 3)
    assert np.ptp(points["position"][:, 2]) < 1e-12
    assert res.value("debonded_area", 2.0) == pytest.approx(0.04)
    assert "interface_damage" in res.rows()[-1] and set(SERIES) <= set(res.components())


@pytest.mark.parametrize("faces", [INSIDE, ACROSS])
def test_bar_sheared_then_rubbed(faces):
    """Mode II along the bilinear law; then the broken interface pressed and sheared:
    Coulomb's friction, reached over the elastic slip of the lips, and the energy that it
    dissipates."""
    mu, elastic = 0.4, 1e-4
    law = Bilinear(1e5, 10.0, 8.0, 0.1, 0.12, friction=mu)
    cell = CohesiveCell(_bar(), MATERIALS, law, faces=faces, slip=elastic)
    res = cell.run({"times": [0, 1], "increments": 40, "strain": {**ZERO, "13": [0, 0.04]}})
    exact = _softening(res.strain[:, 4], G, 1e5, 8.0, 0.12)
    np.testing.assert_allclose(res.stress[:, 4], exact, atol=2e-5)
    assert exact.max() > 7.9 and res.series["debonded_area"][-1] == pytest.approx(0.04)
    assert res.series["interface_dissipation"].max() < 1e-12  # lips that slide without pressure
    path = {"times": [0, 1, 2, 3, 4], "increments": [25, 10, 20, 20],
            "strain": {**ZERO, "33": [0, 0.025, -0.01, -0.01, -0.01],
                       "13": [0, 0, 0, 0.02, 0.0]}}  # fmt: skip
    res = cell.run(path)
    pressure = 0.01 * LENGTH / (LENGTH / M + 1e-5)
    assert -res.stress[35, 2] == pytest.approx(pressure, rel=1e-8)
    shear, strain = res.stress[35:56, 4], res.strain[35:56, 4]
    # lips that stick: the stiffness that reaches Coulomb's bound at the elastic slip
    compliance = LENGTH / G + elastic / (mu * pressure)
    stick = strain * LENGTH / compliance
    np.testing.assert_allclose(shear, np.minimum(stick, mu * pressure), atol=1e-7)
    assert shear.max() == pytest.approx(mu * pressure, rel=1e-8)
    # the slip times the friction, over the area of the interface
    slip = 0.02 * LENGTH - mu * pressure * compliance
    rubbed = res.series["interface_dissipation"]
    assert rubbed[35] == 0 and rubbed[55] == pytest.approx(0.04 * mu * pressure * slip, rel=1e-6)
    # sheared back: the lips stick again (inside Coulomb's cone), nothing more dissipated
    assert res.stress[-1, 4] == pytest.approx(mu * pressure - 0.02 / compliance, rel=1e-7)
    assert rubbed[-1] == rubbed[55]
    assert np.abs(cell.interface_points()["slip"]).max() == pytest.approx(slip, rel=1e-6)
    # by default, 0.005 times the size of the interface elements
    default = CohesiveCell(_bar(), MATERIALS, law, faces=faces)
    assert default.slip == pytest.approx(0.005 * np.sqrt(0.005))
    with pytest.raises(ValueError, match="elastic slip"):
        CohesiveCell(_bar(), MATERIALS, law, slip=0.0)


def test_lips_without_adhesion():
    mu = 0.4
    cell = CohesiveCell(_bar(), MATERIALS, Contact(1e5, friction=mu), faces=INSIDE)
    path = {"times": [0, 1, 2], "increments": [5, 20],
            "strain": {**ZERO, "33": [0, -0.01, -0.01], "13": [0, 0, 0.02]}}  # fmt: skip
    res = cell.run(path)
    pressure = 0.01 * LENGTH / (LENGTH / M + 1e-5)
    assert -res.stress[5, 2] == pytest.approx(pressure, rel=1e-8)
    assert res.stress[-1, 4] == pytest.approx(mu * pressure, rel=1e-7)
    assert res.series["debonded_area"][-1] == pytest.approx(0.04)
    # pulled: nothing holds the lips
    res = cell.run({"times": [0, 1], "increments": 4, "strain": {**ZERO, "33": [0, 0.01]}})
    assert np.abs(res.stress).max() < 1e-9
    assert res.series["interface_opening"][-1] == pytest.approx(0.01)


def test_exponential_law_and_uniaxial_stress():
    law = Exponential(strength=10.0, energy=0.1)
    cell = CohesiveCell(_bar(), MATERIALS, law, faces=INSIDE)
    res = cell.run({"times": [0, 1], "increments": 60, "strain": {**ZERO, "33": [0, 0.03]}})
    # the strain of each opening: that of the bar under the traction, and the opening
    opening = np.linspace(0, 0.03, 30001)
    traction = (
        np.e * 10.0 * opening / law.critical_opening * np.exp(-opening / law.critical_opening)
    )
    strain = traction / M + opening / LENGTH
    assert np.all(np.diff(strain) > 0)
    np.testing.assert_allclose(res.stress[:, 2], np.interp(res.strain[:, 2], strain, traction),
                               atol=1e-4)  # fmt: skip
    assert res.stress[:, 2].max() == pytest.approx(10.0, rel=1e-4)
    assert 0.9 < res.series["interface_damage"][-1] < 1
    # free lateral faces: Young's modulus, and Poisson's contraction of the bar alone
    for faces in (INSIDE, ACROSS):
        cell = CohesiveCell(_bar(), MATERIALS, LAW, faces=faces)
        res = cell.run({"times": [0, 1], "increments": 50, "strain": {"33": [0, 0.025]}})
        exact = _softening(res.strain[:, 2], E, 1e5, 10.0, 0.1)
        np.testing.assert_allclose(res.stress[:, 2], exact, atol=2e-5)
        assert np.abs(res.stress[:, [0, 1, 3, 4, 5]]).max() < 1e-9
        np.testing.assert_allclose(res.strain[:, 0], -NU * exact / E, atol=1e-9)


@pytest.mark.parametrize("faces", [INSIDE, ACROSS])
def test_bar_in_finite_strains(faces):
    """Hencky's elasticity of the bar, ``P = M ln(λ) / λ`` under a uniaxial stretch, and
    the traction of the interface per unit reference area."""
    law = Bilinear(1e5, 10.0, 10.0, 0.1, 0.1)
    onset, failure = 1e-4, 0.02

    def traction(opening):
        if opening <= onset:
            return 1e5 * opening
        return max(10.0 * (failure - opening) / (failure - onset), 0.0)

    def exact(stretch):
        def residual(opening):
            bulk = 1.0 + stretch - opening / LENGTH
            return M * np.log(bulk) / bulk - traction(opening)

        return traction(brentq(residual, 0.0, stretch * LENGTH, xtol=1e-14))

    cell = CohesiveCell(_bar(), MATERIALS, law, faces=faces, kinematics="finite")
    res = cell.run({"times": [0, 1], "increments": 50, "strain": {**ZERO, "33": [0, 0.025]}})
    reference = np.array([exact(h) if h > 0 else 0.0 for h in res.strain[:, 2]])
    np.testing.assert_allclose(res.stress[:, 2], reference, atol=5e-5)
    assert res.kinematics == "finite" and res.series["debonded_area"][-1] == pytest.approx(0.04)
    small = _softening(res.strain[:, 2], M, 1e5, 10.0, 0.1)
    assert np.abs(reference - small).max() > 0.05  # not the small strains


def test_viscosity_slows_the_damage_and_passes_the_snap_back():
    """The regularization of Gao and Bower: ``d = (d_n + r d_∞) / (1 + r)``, ``r`` the time
    step over the viscosity; a brittle interface, which breaks at once, passes with it."""
    stiffness, strength, energy = 1e5, 10.0, 0.1
    onset, failure = strength / stiffness, 2 * energy / strength
    steps, viscosity = 40, 0.05
    rate = (1.0 / steps) / viscosity

    def inviscid(opening):
        if opening <= onset:
            return 0.0
        return min(1.0, failure * (opening - onset) / (opening * (failure - onset)))

    cell = CohesiveCell(_bar(), MATERIALS, LAW, faces=INSIDE, viscosity=viscosity)
    res = cell.run({"times": [0, 1], "increments": steps, "strain": {**ZERO, "33": [0, 0.03]}})
    damage, reference = 0.0, [0.0]
    for strain in res.strain[1:, 2]:

        def lagged(opening, damage=damage):
            return (damage + rate * max(inviscid(opening), damage)) / (1.0 + rate)

        def residual(opening, strain=strain, lagged=lagged):
            stress = (strain * LENGTH - opening) * M / LENGTH
            return stress - (1.0 - lagged(opening)) * stiffness * opening

        opening = brentq(residual, 0.0, strain * LENGTH, xtol=1e-15)
        damage = lagged(opening)
        reference.append((1.0 - damage) * stiffness * opening)
    np.testing.assert_allclose(res.stress[:, 2], reference, atol=1e-6)
    assert res.stress[:, 2].max() > 10.5  # above the strength: the damage lags
    quick = CohesiveCell(_bar(), MATERIALS, LAW, faces=INSIDE, viscosity=1e-7).run(
        {"times": [0, 1], "increments": steps, "strain": {**ZERO, "33": [0, 0.03]}}
    )
    np.testing.assert_allclose(
        quick.stress[:, 2],
        _softening(quick.strain[:, 2], M, stiffness, strength, energy),
        atol=1e-3,
    )
    # a brittle interface: the bar gives back more than the interface takes
    brittle = Bilinear(1e5, 10.0, 10.0, 0.004, 0.004)
    path = {"times": [0, 1], "increments": 20, "strain": {**ZERO, "33": [0, 0.02]}}
    with pytest.raises(RuntimeError, match="give the cell a viscosity"):
        CohesiveCell(_bar(), MATERIALS, brittle, faces=INSIDE, max_cuts=4).run(path)
    res = CohesiveCell(_bar(), MATERIALS, brittle, faces=INSIDE, viscosity=0.02).run(path)
    assert res.stress[-1, 2] < 1e-3 * res.stress[:, 2].max()
    assert res.series["debonded_area"][-1] == pytest.approx(0.04)


def test_grain_boundaries_and_several_interfaces():
    """Every interface of the bar in series: three interfaces of one law carry the stress
    of the bar, their openings add up."""
    linear = _bar()
    grains = TetMesh(linear.nodes, linear.elements, np.zeros(linear.n_elements, np.int64),
                     linear.lower, linear.upper, periodic=True, cell=linear.phase)  # fmt: skip
    cell = CohesiveCell(grains, {0: SOLID}, LAW, faces="grains")
    assert cell.interfaces.n_elements == 24
    res = cell.run({"times": [0, 1], "increments": 8, "strain": {**ZERO, "33": [0, 0.008]}})
    np.testing.assert_allclose(
        res.stress[:, 2], res.strain[:, 2] * LENGTH / (LENGTH / M + 3e-5), rtol=1e-9
    )
    assert res.series["interface_opening"][-1] == pytest.approx(res.stress[-1, 2] / 1e5)


def test_a_law_for_each_pair_of_phases():
    """Three interfaces in series, each with its law: the weakest breaks while the two
    others unload along their penalty; the fields of the elements show which."""
    weak = Bilinear(1e5, 5.0, 5.0, 0.05, 0.05)
    tough = {"normal_strength": 20.0, "mode_I": 0.2, "stiffness": 1e5}
    cell = CohesiveCell(_bar(), MATERIALS, {(0, 1): LAW, (2, 0): weak, (1, 2): tough})
    assert sorted(cell.laws) == [(0, 1), (0, 2), (1, 2)] and cell.interfaces.n_elements == 24
    assert cell.laws[(0, 2)] is weak and cell.laws[(1, 2)] == Bilinear(1e5, 20.0, 20.0, 0.2, 0.2)
    path = {"times": [0, 1], "increments": 40, "strain": {**ZERO, "33": [0, 0.02]}}
    res = cell.run(path, fields=True)
    strain = res.strain[:, 2]
    onset, failure = 5.0 / 1e5, 2 * 0.05 / 5.0
    elastic = strain * LENGTH / (LENGTH / M + 3e-5)
    soft = (strain * LENGTH - failure) / (LENGTH / M + 2e-5 - (failure - onset) / 5.0)
    exact = np.where(elastic <= 5.0, elastic, np.maximum(soft, 0.0))
    np.testing.assert_allclose(res.stress[:, 2], exact, atol=2e-5)
    assert res.series["debonded_area"][-1] == pytest.approx(0.04)
    pairs = np.sort(cell.interfaces.phases, axis=1)
    broken = cell.damage().mean(axis=1)
    np.testing.assert_allclose(broken[(pairs == [0, 2]).all(axis=1)], 1.0)
    assert not broken[~(pairs == [0, 2]).all(axis=1)].any()
    # the fields of the elements: those on the two sides of the broken interface
    fields = res.fields[1.0]
    damaged = fields["interface_damage"]
    assert damaged.shape == (cell.mesh.n_elements,) and set(np.unique(damaged)) == {0.0, 1.0}
    assert int(damaged.sum()) == 16  # eight interface elements, an element on each side
    centres = cell.mesh.nodes[cell.mesh.elements[damaged > 0]].mean(axis=1)[:, 2]
    assert np.all((centres < 0.1) | (centres > 0.9))  # along the faces of the cell
    assert "interface_damage" in path_fields(fields)  # a colour of the figures of the path
    opening = fields["interface_opening"]
    assert opening.max() == pytest.approx(0.02, rel=1e-6) and np.all(opening[damaged == 0] < 1e-4)
    # the pairs given are the interfaces; with other faces, every pair needs its law
    one = CohesiveCell(_bar(), MATERIALS, {(1, 0): LAW})
    assert one.interfaces.n_elements == 8 and list(one.laws) == [(0, 1)] and one.law is LAW
    with pytest.raises(ValueError, match="no law for the interfaces between the phases"):
        CohesiveCell(_bar(), MATERIALS, {(0, 1): LAW}, faces=[(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="pairs of two phases"):
        CohesiveCell(_bar(), MATERIALS, {(0, 1, 2): LAW})
    with pytest.raises(TypeError, match="law of the pair"):
        CohesiveCell(_bar(), MATERIALS, {(0, 1): 3.0})
    # grains: the law of the two phases that a boundary parts, two grains of one phase too
    linear = _bar()
    grains = TetMesh(linear.nodes, linear.elements, np.zeros(linear.n_elements, np.int64),
                     linear.lower, linear.upper, periodic=True, cell=linear.phase)  # fmt: skip
    same = CohesiveCell(grains, {0: SOLID}, {(0, 0): weak}, faces="grains")
    assert same.interfaces.n_elements == 24 and list(same.laws) == [(0, 0)]
    # through homogenize: the pairs as "0-1"
    out = homogenize(_bar(), MATERIALS, properties=[], paths={"pull": path}, nonlinear={
        "interfaces": {"laws": {"0-1": {"stiffness": 1e5, "normal_strength": 10.0,
                                        "shear_strength": 8.0, "mode_I": 0.1, "mode_II": 0.12},
                                "0-2": {"stiffness": 1e5, "normal_strength": 5.0, "mode_I": 0.05},
                                "1-2": tough}}})  # fmt: skip
    np.testing.assert_allclose(out["pull.periodic"].stress, res.stress, atol=1e-9)
    for bad, match in [
        ({"laws": {"0-1": tough}, "friction": 0.1}, "takes no law of its own"),
        ({"laws": {}}, "takes no law of its own"),
        ({"laws": {"a-b": tough}}, "pairs of phases"),
    ]:
        with pytest.raises(ValueError, match=match):
            homogenize(_bar(), MATERIALS, properties=[], paths={"pull": path},
                       nonlinear={"interfaces": bad})  # fmt: skip


# -- kinematic conditions -----------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
def test_kinematic_conditions_against_the_solver_of_bodies_with_interfaces(order):
    data = np.zeros((4, 4, 6), np.uint8)
    data[:, :, 3:] = 1
    block = voxel_tet_mesh(VoxelImage(data, spacing=(0.1, 0.1, 0.1), periodic=False))
    if order == 2:
        block = block.quadratic()
    materials = {0: SOLID, 1: Material(young=3 * E, poisson=0.2)}
    law = Bilinear(1e5, 10.0, 10.0, 0.02, 0.02)
    cell = CohesiveCell(block, materials, law, "kubc")
    path = {"times": [0, 1], "increments": 20, "strain": {**ZERO, "33": [0, 0.03], "13": [0, 0.01]}}
    res = cell.run(path, fields=True)
    opened, interfaces = insert_interfaces(block, "phases")
    x = opened.nodes
    on = np.flatnonzero(
        (np.abs(x - block.lower) <= 1e-9).any(axis=1)
        | (np.abs(x - block.upper) <= 1e-9).any(axis=1)
    )
    held = np.zeros((len(on), 3))
    held[:, 2] = 0.03 * x[on, 2] + 0.005 * x[on, 0]
    held[:, 0] = 0.005 * x[on, 2]
    out = CohesiveProblem(opened, interfaces, materials, law).run([(on, [0, 1, 2], held)], steps=20)

    def table(nodes, u):
        rows = np.round(np.column_stack([nodes, u]), 8)
        return rows[np.lexsort(rows.T[::-1])]

    mine = table(cell.mesh.nodes, res.fields[1.0]["displacement"])
    np.testing.assert_allclose(mine, table(x, out.displacement.reshape(-1, 3)), atol=2e-8)
    assert res.series["debonded_area"][-1] == pytest.approx(out.cracked[-1], rel=1e-9)
    assert 0 < out.cracked[-1] < 0.16  # the rim of the interface is held


# -- beads in a matrix ---------------------------------------------------------------------------


@mmg
@pytest.mark.slow
def test_beads_that_debond_from_their_matrix():
    """Between the composite with bonded beads and the matrix with holes: as stiff as the
    first before the interfaces break, near the second once they have, and the first again
    in compression."""
    import morphogen as mg
    from morphogen.mesh.conforming import conforming_mesh

    micro = mg.models.HardcoreSpheres(radius=0.45, number=1).generate(
        mg.Domain.cube(2.0, periodic=True), seed=3
    )
    mesh = conforming_mesh(micro, size=0.4, periodic=True, order=1, method="grid")
    matrix, glass = Material(young=3000.0, poisson=0.35), Material(young=70000.0, poisson=0.22)
    materials = {0: matrix, 1: glass}
    small = {"times": [0, 1], "increments": 1, "strain": {"11": [0, 1e-4]}}
    bonded = NonlinearFEProblem(mesh, materials, solver="direct").run(small)
    porous = NonlinearFEProblem(mesh, {0: matrix, 1: Material(void=True)}, solver="direct").run(
        small
    )
    stiff, soft = bonded.stress[1, 0] / 1e-4, porous.stress[1, 0] / 1e-4
    assert soft < 0.9 * stiff
    law = {"normal_strength": 30.0, "mode_I": 0.05, "friction": 0.3}
    cell = CohesiveCell(mesh, materials, law, viscosity=0.01)
    assert isinstance(cell.law, Bilinear) and cell.law.mode_II == 0.05
    area = float(cell.interfaces.areas(cell.mesh).sum())
    assert area == pytest.approx(4 * np.pi * 0.45**2, rel=0.05) and not cell.periods.any()
    path = {"times": [0, 1, 2, 3], "increments": [20, 5, 5],
            "strain": {"11": [0, 0.03, 0.0, -0.01]}}  # fmt: skip
    res = cell.run(path)
    stress, strain = res.stress[:, 0], res.strain[:, 0]
    assert stress[1] / strain[1] == pytest.approx(stiff, rel=0.01)  # the penalty of the interface
    tangent = (stress[20] - stress[18]) / (strain[20] - strain[18])
    assert soft < tangent < soft + 0.35 * (stiff - soft)
    assert np.abs(res.stress[:, 1:]).max() < 1e-4 * stress.max()
    series = res.series
    assert 0.5 * area < series["debonded_area"][20] < area
    assert np.all(np.diff(series["debonded_area"]) >= 0) and series["interface_opening"][20] > 0.01
    # pushed: the beads carry the load again
    pushed = (stress[-1] - stress[-3]) / (strain[-1] - strain[-3])
    assert pushed == pytest.approx(stiff, rel=0.05)
    assert series["interface_dissipation"][-1] > 0


@mmg
@pytest.mark.slow
def test_grain_boundaries_of_a_polycrystal_break():
    """A polycrystal of eight grains pulled until a crack has crossed it along its grain
    boundaries: the work of the stress is the fracture energy times the area of the
    crack; whole grains (their boundaries on the faces of the cell, with their images)
    and grains cut by the box (boundaries inside, periodic links) give the same."""
    import morphogen as mg
    from morphogen.mesh.conforming import conforming_mesh

    micro = mg.models.Voronoi(n_cells=8).generate(mg.Domain.cube(1.0, periodic=True), seed=2)
    steel = {1: Material(young=200e3, poisson=0.3)}
    law = {"normal_strength": 300.0, "mode_I": 0.5}
    path = {"times": [0, 1], "increments": 80, "strain": {"11": [0, 0.008]}}
    peaks, areas = [], []
    for cells in ("whole", "box"):
        mesh = conforming_mesh(micro, size=0.2, cells=cells, order=1)
        cell = CohesiveCell(mesh, steel, law, faces="grains")
        crossing = np.any(cell.periods != 0, axis=1)
        assert (crossing.mean() > 0.4) == (cells == "whole")
        assert float(cell._areas.sum()) == pytest.approx(5.72, rel=0.01)
        res = cell.run(path)
        stress, strain = res.stress[:, 0], res.strain[:, 0]
        assert stress[1] / strain[1] == pytest.approx(200e3, rel=0.005)
        # the boundaries are not across the load: stronger than they are, and a crack
        # that is not flat
        assert 1.1 * 300.0 < stress.max() < 1.6 * 300.0
        crack = res.series["debonded_area"][-1]
        assert 1.1 < crack < 1.6 and abs(stress[-1]) < 1e-6 * stress.max()
        assert np.trapezoid(stress, strain) == pytest.approx(0.5 * crack, rel=0.02)
        assert np.abs(res.stress[:, 1:]).max() < 1e-6 * stress.max()
        peaks.append(stress.max())
        areas.append(crack)
    assert peaks[0] == pytest.approx(peaks[1], rel=0.01)
    assert areas[0] == pytest.approx(areas[1], rel=0.01)


# -- homogenize, the configurations and the command line ---------------------------------------


def test_homogenize_follows_the_paths_with_the_interfaces(tmp_path):
    mesh = _bar()
    path = {"times": [0, 1], "increments": 10, "strain": {**ZERO, "33": [0, 0.025]}}
    spec = {"stiffness": 1e5, "normal_strength": 10.0, "shear_strength": 8.0, "mode_I": 0.1,
            "mode_II": 0.12, "faces": [[0, 1]]}  # fmt: skip
    out = homogenize(mesh, MATERIALS, properties=[], paths={"pull": path},
                     nonlinear={"interfaces": spec})  # fmt: skip
    res = out["pull.periodic"]
    direct = CohesiveCell(mesh, MATERIALS, LAW, faces=INSIDE).run(path)
    np.testing.assert_allclose(res.stress, direct.stress, rtol=1e-12, atol=1e-12)
    assert res.series["debonded_area"][-1] == pytest.approx(0.04)
    # the penalty left out: the stiffest modulus over the size of the interface elements
    law = interface_law({"normal_strength": 10.0, "mode_I": 0.1}, 1000.0, 0.1)
    assert law == Bilinear(1e6, 10.0, 10.0, 0.1, 0.1)
    assert interface_law({"law": "contact", "friction": 0.2}, 2.0, 0.5) == Contact(400.0, 0.2)
    assert interface_law({"law": "exponential", "strength": 2.0, "energy": 0.5, "beta": 2.0},
                         1.0, 1.0) == Exponential(2.0, 0.5, 2.0)  # fmt: skip
    default = CohesiveCell(mesh, MATERIALS, {"normal_strength": 10.0, "mode_I": 0.1})
    assert default.law.stiffness == pytest.approx(100 * E / np.sqrt(0.005))
    for bad, match in [
        ({"interfaces": {"mode_I": 0.1}}, "needs 'normal_strength'"),
        ({"interfaces": {"law": "linear"}}, "the law is one of"),
        ({"interfaces": {**spec, "beta": 2.0}}, "unknown parameters"),
        ({"interfaces": {**spec, "area": 2.0}}, "unknown options"),
        ({"interfaces": 0.3}, "a mapping"),
        ({"interfaces": {**spec, "viscosity": -1.0}}, "viscosity"),
        ({"interfaces": {**spec, "slip": -1.0}}, "elastic slip"),
        ({"interfaces": {**spec, "faces": [[3, 4]]}}, "no interface to open"),
        ({"interfaces": spec, "contact": True}, "not both"),
    ]:
        with pytest.raises(ValueError, match=match):
            homogenize(mesh, MATERIALS, properties=[], paths={"pull": path}, nonlinear=bad)
    with pytest.raises(ValueError, match="factorized"):
        homogenize(mesh, MATERIALS, properties=[], paths={"pull": path}, solver="cg",
                   nonlinear={"interfaces": spec})  # fmt: skip
    image = VoxelImage(np.ones((4, 4, 4), np.uint8), periodic=True)
    with pytest.raises(ValueError, match="phases of a mesh"):
        homogenize(image, {1: SOLID}, properties=[], paths={"pull": path},
                   nonlinear={"interfaces": spec})  # fmt: skip
    with pytest.raises(TypeError, match="traction-separation law"):
        CohesiveCell(mesh, MATERIALS, 3.0)
    with pytest.raises(NotImplementedError, match="loading paths"):
        CohesiveCell(mesh, MATERIALS, LAW).solve(Loading.parse("E11=0.01"))
    # the command line, on the file of the mesh
    mesh_file = tmp_path / "bar.vtu"
    io.save_mesh(mesh, mesh_file)
    (tmp_path / "mats.yaml").write_text(
        "".join(f"{p}: {{young: {E}, poisson: {NU}}}\n" for p in range(3)), encoding="utf-8"
    )
    report = tmp_path / "report.json"
    code = cli.main([
        "homogenize", str(mesh_file), "--materials", str(tmp_path / "mats.yaml"),
        "--property", "bulk_modulus",
        "--path", "pull: t=0,1 n=10 E33=0.025 E11=0 E22=0 E23=0 E13=0 E12=0",
        "--interfaces",
        "stiffness:1e5,normal_strength:10,shear_strength:8,mode_I:0.1,mode_II:0.12,faces:0-1",
        "--curves", str(tmp_path / "curves"), "--periodic", "-o", str(report),
    ])  # fmt: skip
    assert code == 0
    final = json.loads(report.read_text(encoding="utf-8"))["paths"]["pull.periodic"]
    assert final["debonded_area"] == pytest.approx(0.04)
    assert final["S33"] == pytest.approx(direct.stress[-1, 2], abs=1e-9)
    curve = (tmp_path / "curves" / "bar_pull_periodic.csv").read_text(encoding="utf-8")
    assert "debonded_area" in curve.splitlines()[0] and "interface_dissipation" in curve
    for text, match in [("mode_I", "KEY:VALUE"), ("mode_I:x", "is a number"),
                        ("faces:0+1", "pairs of phases")]:  # fmt: skip
        with pytest.raises(ValueError, match=match):
            cli._interface_options(text)
    assert cli._interface_options("law:contact,faces:0-1+0-2,friction:0.2") == {
        "law": "contact", "faces": [[0, 1], [0, 2]], "friction": 0.2}  # fmt: skip


def test_configuration_of_the_interfaces():
    base = {
        "model": {"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.1},
        "domain": {"size": 6.0, "periodic": True},
        "mesh": {"size": 1.0},
        "homogenization": {
            "materials": {0: {"young": 1000.0, "poisson": 0.3}, 1: {"young": 5e3, "poisson": 0.2}},
            "properties": [],
            "paths": {"pull": {"times": [0, 1], "strain": {"33": 0.02}}},
            "nonlinear": {
                "interfaces": {
                    "normal_strength": 30,
                    "mode_I": 0.05,
                    "friction": 0.3,
                    "viscosity": 0.01,
                }
            },
        },
    }
    cfg = load_config(base)
    options = cfg.homogenization.options()["nonlinear"]
    law = {"law": "bilinear", "normal_strength": 30.0, "mode_I": 0.05, "friction": 0.3}
    assert options == {"interfaces": {**law, "viscosity": 0.01, "faces": "phases"}}
    assert cfg.homogenization.method == "fe" and cfg.mesh.kwargs(set())["phases"] == "all"

    def changed(**homog):
        data = json.loads(json.dumps(base))
        data["homogenization"]["materials"] = base["homogenization"]["materials"]
        data["homogenization"].update(homog)
        return data

    other = load_config(changed(nonlinear={"interfaces": {
        "law": "exponential", "strength": 30, "energy": 0.05, "faces": [[0, 1]]}}))  # fmt: skip
    assert other.homogenization.options()["nonlinear"]["interfaces"] == {
        "law": "exponential", "strength": 30.0, "energy": 0.05, "faces": [[0, 1]]}  # fmt: skip
    off = load_config(changed(nonlinear={"interfaces": False}))
    assert off.homogenization.options()["nonlinear"] == {}
    # a law for each pair of phases
    pairs = load_config(changed(nonlinear={"interfaces": {
        "laws": {"1-0": {"normal_strength": 30, "mode_I": 0.05, "friction": 0.2}},
        "viscosity": 0.02, "faces": "grains"}}))  # fmt: skip
    assert pairs.homogenization.options()["nonlinear"]["interfaces"] == {
        "laws": {"0-1": {"law": "bilinear", "normal_strength": 30.0, "mode_I": 0.05,
                         "friction": 0.2}},
        "viscosity": 0.02, "faces": "grains"}  # fmt: skip
    # the columns of the interfaces are quantities of the statistics of volume elements
    data = changed()
    data["domain"].pop("size")
    data["rve"] = {"sizes": [6.0], "realizations": 2, "properties": ["pull.debonded_area"]}
    assert load_config(data).rve.properties == ["pull.debonded_area"]
    data["homogenization"]["nonlinear"] = {}
    with pytest.raises(ConfigError, match="unknown quantity 'debonded_area'"):
        load_config(data)
    law = {"normal_strength": 30, "mode_I": 0.05}
    bad = [
        (changed(nonlinear={"interfaces": {"mode_I": 0.05}}), "needs \\['normal_strength'\\]"),
        (changed(nonlinear={"interfaces": {**law, "strength": 3}}), "not parameters of the law"),
        (changed(nonlinear={"interfaces": {**law, "mu": 0.3}}), "unknown keys"),
        (changed(nonlinear={"interfaces": {**law, "law": "linear"}}), "law: one of"),
        (changed(nonlinear={"interfaces": {**law, "friction": -0.1}}), "0 or more"),
        (changed(nonlinear={"interfaces": {**law, "mode_I": 0}}), "positive"),
        (changed(nonlinear={"interfaces": {**law, "slip": 0}}), "positive"),
        (changed(nonlinear={"interfaces": {**law, "eta": "two"}}), "a number"),
        (changed(nonlinear={"interfaces": {**law, "faces": "all"}}), "'phases', 'grains'"),
        (changed(nonlinear={"interfaces": {**law, "faces": [[0, 0]]}}), "two different phases"),
        (changed(nonlinear={"interfaces": True}), "a mapping"),
        (changed(nonlinear={"interfaces": {"laws": {"0-1": law}, "friction": 0.1}}), "'laws'"),
        (changed(nonlinear={"interfaces": {"laws": {"0": law}}}), "pairs of two phases"),
        (changed(nonlinear={"interfaces": {"laws": {"0-1": {"mode_I": 1.0}}}}), "needs"),
        (
            changed(nonlinear={"interfaces": {"laws": {"0-1": {**law, "slip": 1.0}}}}),
            "not parameters of the law",
        ),
        (changed(nonlinear={"interfaces": law, "contact": True}), "not both"),
        (changed(paths={}, properties=["bulk_modulus"]), "along the loading paths"),
        (changed(solver="cg"), "factorized"),
    ]
    voxels = changed(method="fft")
    voxels.pop("mesh")
    voxels["voxels"] = {"shape": 8}
    bad.append((voxels, "method: fe"))
    for data, match in bad:
        with pytest.raises(ConfigError, match=match):
            load_config(data)


def test_elastic_cell_brought_back_to_zero_strain():
    """A state without stress has no force with which to compare its residuals: the scales
    keep a share of the largest reached along the path (finite elements and FFT)."""
    data = np.zeros((3, 3, 3), np.uint8)
    data[1, 1, 1] = 1
    image = VoxelImage(data, periodic=True)
    materials = {0: Material(young=3000.0, poisson=0.35), 1: Material(young=70000.0, poisson=0.22)}
    paths = {
        "stress": {"times": [0, 1, 2], "increments": [4, 4], "strain": {"11": [0, 0.03, 0.0]}},
        "strain": {
            "times": [0, 1, 2],
            "increments": [4, 4],
            "strain": {**ZERO, "11": [0, 0.03, 0.0]},
        },
    }
    for sample in (voxel_tet_mesh(image), image):
        out = homogenize(sample, materials, properties=[], paths=paths)
        for name in paths:
            res = out[f"{name}.periodic"]
            assert len(res.times) == 9 and res.info["cuts"] == 0
            assert np.abs(res.stress[-1]).max() < 1e-9 * np.abs(res.stress[4]).max()
            assert np.abs(res.strain[-1]).max() < 1e-12
