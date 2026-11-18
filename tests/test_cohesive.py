"""Cohesive zone elements: the laws (fracture energies of the modes and of their mixture,
irreversibility, compression), the insertion of interfaces (between phases, along a
crack whose front stays closed), the debonding of a bar in small and finite strains, the
frame indifference of the finite strain elements, the calibration formulas, the
debonding of the particles of a composite, and the double cantilever beam against the
corrected beam theory (Linke & Lammering 2023); the elements of six nodes a side between
10-node tetrahedra: the same bar, the area and the forces of the curved interface of a
sphere, and the beam with as many nodes."""

import math

import numpy as np
import pytest

from morphogen.fracture import (
    Bilinear,
    CohesiveProblem,
    Exponential,
    cohesive_zone_length,
    dcb_beam_theory,
    dcb_specimen,
    insert_interfaces,
    traction,
    turon_strength,
)
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage
from morphogen.mesh.conforming import has_mmg

needs_mmg = pytest.mark.skipif(not has_mmg(), reason="MMG (pymmg) is not installed")
LAW = Bilinear(1e4, 5.0, 8.0, 0.5, 1.0, eta=2.0)
STEEL = Material(young=1000.0, poisson=0.3)


def _energy(law, direction, reach):
    """The work of the traction along a monotonic separation in ``direction``."""
    direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    h, prev, energy, last = 0.0, 0.0, 0.0, 0.0
    for s in np.linspace(0.0, reach, 100001)[1:]:
        t, h = traction(law, s * direction, h)
        value = t @ direction
        energy += 0.5 * (value + prev) * (s - last)
        prev, last = value, s
    return energy


def test_traction_separation_laws():
    for b in (0.0, 0.3, 1.0):  # the energy at a fixed mixity: G_Ic + (G_IIc - G_Ic) B^eta
        direction = [math.sqrt(1 - b), math.sqrt(b), 0.0]
        assert _energy(LAW, direction, 0.5) == pytest.approx(0.5 + 0.5 * b**2, rel=1e-4)
    ex = Exponential(5.0, 0.5, beta=1.0)
    assert _energy(ex, [1.0, 0.0, 0.0], 40 * ex.critical_opening) == pytest.approx(0.5, rel=1e-4)
    peak = max(traction(ex, [s, 0, 0])[0][0] for s in np.linspace(0, 3 * ex.critical_opening, 3001))
    assert peak == pytest.approx(5.0, rel=1e-5)  # sigma_c at delta_c
    # irreversible: unloading along the secant to the origin, reloading along it
    t, h = traction(LAW, [0.05, 0.0, 0.0])
    assert 0 < h < 1
    back, h2 = traction(LAW, [0.02, 0.0, 0.0], h)
    assert h2 == h and back[0] == pytest.approx(t[0] * 0.02 / 0.05)
    t, h = traction(ex, [3 * ex.critical_opening, 0, 0])
    back, _ = traction(ex, [ex.critical_opening, 0, 0], h)
    assert back[0] == pytest.approx(t[0] / 3)
    # compression: no damage, the penalty
    t, h = traction(LAW, [-0.1, 0.0, 0.0])
    assert h == 0 and t[0] == pytest.approx(-0.1 * 1e4)
    # shear in any tangential direction alike
    t1, _ = traction(LAW, [0.0, 0.01, 0.0])
    t2, _ = traction(LAW, [0.0, 0.0, 0.01])
    assert t1[1] == pytest.approx(t2[2]) and t1[2] == t2[1] == 0
    with pytest.raises(ValueError, match="failure separation"):
        Bilinear(10.0, 5.0, 5.0, 0.1, 0.1)
    with pytest.raises(ValueError):
        Exponential(-1.0, 1.0)


def test_insertion_of_interfaces():
    # a cubic inclusion of one voxel: its six faces opened, its eight corners duplicated
    data = np.zeros((3, 3, 3), np.uint8)
    data[1, 1, 1] = 1
    mesh = voxel_tet_mesh(VoxelImage(data, periodic=False))
    opened, inter = insert_interfaces(mesh, "phases")
    assert opened.n_nodes == mesh.n_nodes + 8 and inter.n_elements == 12
    assert inter.areas(opened).sum() == pytest.approx(6.0)
    assert (inter.phases == [0, 1]).all()  # side -: the matrix, side +: the inclusion
    inside = opened.elements[opened.phase == 1]
    assert np.isin(inter.plus, inside).all() and not np.isin(inter.minus, inside).any()
    # normals from side - to side +: towards the centre of the inclusion
    x = opened.nodes
    n = np.cross(
        x[inter.minus[:, 1]] - x[inter.minus[:, 0]], x[inter.minus[:, 2]] - x[inter.minus[:, 0]]
    )
    assert np.all(np.einsum("ij,ij->i", n, 1.5 - x[inter.minus].mean(axis=1)) > 0)
    # a crack through half of a block: the nodes of its front are not split
    block = voxel_tet_mesh(VoxelImage(np.zeros((4, 2, 2), np.uint8), periodic=False))
    opened, inter = insert_interfaces(
        block, lambda c: (np.abs(c[:, 2] - 1.0) < 1e-9) & (c[:, 0] < 2.0)
    )
    front = np.flatnonzero(
        (np.abs(block.nodes[:, 0] - 2.0) < 1e-9) & (np.abs(block.nodes[:, 2] - 1.0) < 1e-9)
    )
    behind = np.flatnonzero(
        (block.nodes[:, 0] < 2.0 - 1e-9) & (np.abs(block.nodes[:, 2] - 1.0) < 1e-9)
    )
    assert opened.n_nodes == block.n_nodes + len(behind)
    assert inter.n_elements == 8 and len(front) == 3
    with pytest.raises(ValueError, match="interior face"):
        insert_interfaces(block, np.ones(3, dtype=bool))


def _bar():
    data = np.zeros((2, 2, 4), np.uint8)
    data[:, :, 2:] = 1
    mesh = voxel_tet_mesh(VoxelImage(data, periodic=False))
    opened, inter = insert_interfaces(mesh, "phases")
    x = opened.nodes
    bottom, top = np.flatnonzero(x[:, 2] < 1e-9), np.flatnonzero(x[:, 2] > 4 - 1e-9)
    return opened, inter, [(bottom, [0, 1, 2], 0.0), (top, 2, 1.0), (top, [0, 1], 0.0)]


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_debonding_of_a_bar(kinematics, tmp_path):
    """A bar of two blocks pulled apart: its peak load N A, its work G_Ic A (the elastic
    energy is given back), a crack of the area of the interface."""
    mesh, inter, prescribed = _bar()
    law = Bilinear(1e5, 5.0, 5.0, 0.5, 0.5)
    problem = CohesiveProblem(mesh, inter, {0: STEEL, 1: STEEL}, law, kinematics=kinematics)
    res = problem.run(prescribed, load=0.25, steps=250, fields=True)
    load = res.reactions[:, 1, 2]
    assert load.max() == pytest.approx(5.0 * 4.0, rel=0.02)
    work = np.trapezoid(load, res.loads)
    assert work == pytest.approx(0.5 * 4.0, rel=0.03)
    assert res.cracked[-1] == pytest.approx(4.0) and res.damage.min() == pytest.approx(1.0)
    assert len(res.fields) == len(res.loads) and abs(load[-1]) < 1e-6
    rows = res.rows()
    assert (
        rows[0]["load"] == 0
        and "R2z" in rows[-1]
        and rows[-1]["cracked_area"] == pytest.approx(4.0)
    )
    bulk, wedges = problem.write(tmp_path / "bar", res)
    assert bulk.stat().st_size > 1000 and b"damage" in wedges.read_bytes()


def test_frame_indifference_in_finite_strains():
    """An opened interface turned rigidly: in finite strains its forces turn with it (the
    current normal), the neo-Hookean bulk carries no stress; small strains do not."""
    mesh, inter, _ = _bar()
    law = Exponential(5.0, 0.5)
    x = mesh.nodes
    top = np.zeros(mesh.n_nodes, dtype=bool)
    top[np.unique(mesh.elements[mesh.phase == 1])] = True
    opening = np.where(top[:, None], [0.0, 0.0, 0.05], 0.0)
    angle = 0.7
    rot = np.array(
        [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]]
    )
    centre = x.mean(axis=0)
    turned = (x + opening - centre) @ rot.T + centre - x
    finite = CohesiveProblem(mesh, inter, {0: STEEL, 1: STEEL}, law, kinematics="finite")
    hist = finite._new_state()
    rigid = (x - centre) @ rot.T + centre - x
    f_rigid, _, _ = finite._forces(rigid.ravel(), hist, False)
    assert np.abs(f_rigid).max() < 1e-9
    nodes = np.unique(inter.plus)
    f0, _, _ = finite._forces(opening.ravel(), hist, False)
    f1, _, _ = finite._forces(turned.ravel(), hist, False)
    np.testing.assert_allclose(
        f1.reshape(-1, 3)[nodes], f0.reshape(-1, 3)[nodes] @ rot.T, atol=1e-9
    )
    small = CohesiveProblem(mesh, inter, {0: STEEL, 1: STEEL}, law)
    g0, _, _ = small._forces(opening.ravel(), hist, False)
    g1, _, _ = small._forces(turned.ravel(), hist, False)
    gap = np.abs(g1.reshape(-1, 3)[nodes] - g0.reshape(-1, 3)[nodes] @ rot.T).max()
    assert gap > 1e-3 * np.abs(g0).max()


def test_calibration_formulas():
    assert cohesive_zone_length(1000.0, 0.5, 5.0) == pytest.approx(20.0)
    t = turon_strength(1000.0, 0.5, 2.0, elements=3)
    assert cohesive_zone_length(1000.0, 0.5, t, factor=9 * math.pi / 32) == pytest.approx(6.0)
    d = np.linspace(0.0, 8.0, 801)
    load, crack = dcb_beam_theory(1000.0, 0.3, 2.0, 4.0, 20.0, 0.5, d)
    shear = 1000.0 / 2.6
    gamma = 1.18 * 1000.0 / shear
    chi = math.sqrt(1000.0 / (11 * shear) * (3 - 2 * (gamma / (1 + gamma)) ** 2))
    inertia = 2.0 * 4.0**3 / 12
    elastic = 3 * 1000.0 * inertia * d / (2 * (20.0 + chi * 4.0) ** 3)
    onset = np.argmax(crack > 20.0)
    np.testing.assert_allclose(load[:onset], elastic[:onset])
    assert np.all(np.diff(crack) >= 0) and np.all(np.diff(load[onset:]) < 0)
    # along the propagation, G = P^2 (a + chi h)^2 / (w E I) = G_Ic
    a = crack[onset + 10 :]
    g = load[onset + 10 :] ** 2 * (a + chi * 4.0) ** 2 / (2.0 * 1000.0 * inertia)
    np.testing.assert_allclose(g, 0.5, rtol=1e-10)


def test_debonding_of_particles_in_a_composite():
    """Stiff particles in a matrix under tension: their interfaces open and soften the
    composite (its stress falls below the bonded one)."""
    data = np.zeros((6, 6, 6), np.uint8)
    data[1:3, 1:3, 1:3] = 1
    data[3:5, 3:5, 3:5] = 1
    mesh = voxel_tet_mesh(VoxelImage(data, periodic=False))
    opened, inter = insert_interfaces(mesh, "phases")
    assert inter.n_elements == 2 * 6 * 4 * 2
    x = opened.nodes
    bottom, top = np.flatnonzero(x[:, 2] < 1e-9), np.flatnonzero(x[:, 2] > 6 - 1e-9)
    materials = {0: Material(young=1000.0, poisson=0.3), 1: Material(young=20000.0, poisson=0.2)}
    law = Bilinear(1e5, 5.0, 5.0, 0.05, 0.05)
    res = CohesiveProblem(opened, inter, materials, law).run(
        [(bottom, [0, 1, 2], 0.0), (top, 2, 1.0)], load=0.2, steps=20, viscosity=1e-3
    )
    stress = res.reactions[:, 1, 2] / 36.0
    secant = stress[1:] / (res.loads[1:] / 6.0)
    assert res.cracked[-1] > 0 and secant[-1] < 0.9 * secant[0]


@pytest.mark.slow
def test_double_cantilever_beam_against_beam_theory(tmp_path):
    """The double cantilever beam (60 x 2, arms of 4, precrack 20, E = 1000, G_Ic = 0.5,
    T = 5: a cohesive zone of 20 elements of 1): the peak load of the corrected beam theory
    (within 2 %), the loads (12 %) and crack lengths (2.5) of the propagation."""
    mesh, inter, sets = dcb_specimen(60.0, 2.0, 4.0, 20.0, 1.0)
    problem = CohesiveProblem(mesh, inter, {0: STEEL}, Bilinear(1e5, 5.0, 5.0, 0.5, 0.5))
    res = problem.run(
        [
            (sets["top"], 2, 1.0),
            (sets["bottom"], 2, -1.0),
            (sets["top"], [0, 1], 0.0),
            (sets["bottom"], [0, 1], 0.0),
        ],
        load=3.0,
        steps=60,
        viscosity=1e-3,
    )
    d = 2 * res.loads
    load = res.reactions[:, 0, 2]
    beam, crack = dcb_beam_theory(1000.0, 0.3, 2.0, 4.0, 20.0, 0.5, d)
    assert load.max() == pytest.approx(beam.max(), rel=0.02)
    growing = d > 4.4
    np.testing.assert_allclose(load[growing], beam[growing], rtol=0.12)
    np.testing.assert_allclose(20.0 + res.cracked[growing] / 2.0, crack[growing], atol=2.5)


# -- interface elements between 10-node tetrahedra -----------------------------------------------


def test_insertion_of_interfaces_in_a_quadratic_mesh():
    """A mesh of 10-node tetrahedra is opened by its corners; each lip has its own mid-edge
    nodes, at the places of those of the mesh, and the front keeps its own."""
    block = voxel_tet_mesh(VoxelImage(np.zeros((4, 2, 2), np.uint8), periodic=False))
    quad = block.quadratic()

    def behind(c):
        return (np.abs(c[:, 2] - 1.0) < 1e-9) & (c[:, 0] < 2.0)

    linear, three = insert_interfaces(block, behind)
    opened, six = insert_interfaces(quad, behind)
    assert opened.order == 2 and six.order == 2 and three.order == 1
    assert six.minus.shape == (8, 6) and six.n_elements == three.n_elements
    assert np.array_equal(opened.linear().elements, linear.elements)
    assert six.areas(opened).sum() == pytest.approx(three.areas(linear).sum()) == 4.0
    np.testing.assert_allclose(six.weights.sum(), 1.0)
    x = opened.nodes
    np.testing.assert_allclose(x[six.minus], x[six.plus], atol=1e-12)  # the lips: one place
    # the mid-edge nodes: the middles of the edges of their faces
    for side in (six.minus, six.plus):
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
            np.testing.assert_allclose(x[side[:, 3 + k]], 0.5 * (x[side[:, i]] + x[side[:, j]]))
    # the nodes of the front belong to both lips; the others to one
    shared = np.intersect1d(six.minus, six.plus)
    assert len(shared) and np.allclose(x[shared, 0], 2.0) and np.allclose(x[shared, 2], 1.0)
    assert len(shared) == 5  # three corners of the front and the two mid-edge nodes between
    lips = np.setdiff1d(np.union1d(six.minus, six.plus), shared)
    assert opened.n_nodes == quad.n_nodes + len(lips) // 2
    assert opened.volumes().sum() == pytest.approx(16.0)
    with pytest.raises(ValueError, match="six for 10-node"):
        CohesiveProblem(opened, three, {0: STEEL}, LAW)
    with pytest.raises(ValueError, match="4-node tetrahedra"):
        CohesiveProblem(opened, six, {0: STEEL}, LAW, bulk="points")


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_debonding_of_a_bar_of_quadratic_elements(kinematics, tmp_path):
    """The bar pulled apart, with 10-node tetrahedra and interface elements of six nodes a
    side. Without Poisson's effect its fields are homogeneous, in both spaces of elements:
    the loads are those of the 4-node elements at every step, the peak load N A and the
    work of the interfaces G_Ic A."""
    solid = Material(young=1000.0, poisson=0.0)
    results = []
    for order in (1, 2):
        data = np.zeros((2, 2, 4), np.uint8)
        data[:, :, 2:] = 1
        mesh = voxel_tet_mesh(VoxelImage(data, periodic=False))
        opened, inter = insert_interfaces(mesh.quadratic() if order == 2 else mesh, "phases")
        x = opened.nodes
        bottom, top = np.flatnonzero(x[:, 2] < 1e-9), np.flatnonzero(x[:, 2] > 4 - 1e-9)
        law = Bilinear(1e5, 5.0, 5.0, 0.5, 0.5)
        problem = CohesiveProblem(opened, inter, {0: solid, 1: solid}, law, kinematics=kinematics)
        res = problem.run(
            [(bottom, [0, 1, 2], 0.0), (top, 2, 1.0), (top, [0, 1], 0.0)], load=0.25, steps=60
        )
        results.append(res)
        assert res.damage.shape == (inter.n_elements, 3 if order == 1 else 6)
        assert res.cracked[-1] == pytest.approx(4.0) and res.damage.min() == pytest.approx(1.0)
    linear, quadratic = results
    assert quadratic.reactions[:, 1, 2].max() == pytest.approx(5.0 * 4.0, rel=0.03)
    assert quadratic.work["interfaces"][-1] == pytest.approx(0.5 * 4.0, rel=0.03)
    np.testing.assert_allclose(quadratic.loads, linear.loads, rtol=1e-12)
    np.testing.assert_allclose(
        quadratic.reactions[:, 1, 2], linear.reactions[:, 1, 2], rtol=1e-5, atol=1e-4
    )
    bulk, wedges = problem.write(tmp_path / "bar", res)
    assert bulk.stat().st_size > 1000 and b"damage" in wedges.read_bytes()


@needs_mmg
def test_curved_interface_of_a_sphere():
    """The interface elements of a sphere, on the curved faces of its mesh: their area is
    that of the sphere within 0.05 % with elements of its radius (6 % less for the flat
    faces of the same mesh, on the background grid), and the force that a rigid
    translation δ of the sphere in its matrix takes through an interface of stiffness K
    is K δ times that area."""
    import morphogen as mg
    from morphogen.models.union import PrimitiveUnion
    from morphogen.primitives import Spheres

    radius, stiffness, delta = 1.2, 50.0, 0.01
    domain = mg.Domain.cube(4.0, periodic=True)
    centre = np.array([[1.9, 2.15, 2.05]])
    micro = PrimitiveUnion(domain, Spheres(centre, np.array([radius])), None)
    sphere = 4.0 * math.pi * radius**2
    law = Bilinear(stiffness, 10.0, 10.0, 5.0, 5.0)
    materials = {0: Material(young=3.0, poisson=0.35), 1: Material(young=70.0, poisson=0.22)}
    flat = mg.mesh.conforming_mesh(micro, size=1.2, method="grid")
    off = {}
    for name, mesh in (("flat", flat.quadratic()), ("curved", mg.mesh.curved_mesh(flat, micro))):
        opened, inter = insert_interfaces(mesh, "phases")
        assert opened.curved == (name == "curved") and inter.order == 2
        assert opened.volumes().sum() == pytest.approx(64.0, rel=1e-12)
        area = inter.areas(opened).sum()
        problem = CohesiveProblem(opened, inter, materials, law)
        inside = np.unique(opened.elements[opened.phase == 1])
        u = np.zeros((opened.n_nodes, 3))
        u[inside, 0] = delta  # the sphere moved as a whole: no strain, an opening δ
        problem._forces(u.ravel(), problem._new_state(), False)
        interfaces, bulk = problem._parts
        force = interfaces.reshape(-1, 3)[inside].sum(axis=0)
        assert np.abs(bulk).max() < 1e-12
        np.testing.assert_allclose(force, [stiffness * delta * area, 0.0, 0.0], atol=1e-12)
        off[name] = area / sphere - 1.0
    assert -0.07 < off["flat"] < -0.05 and abs(off["curved"]) < 5e-4


@pytest.mark.slow
def test_double_cantilever_beam_of_quadratic_elements():
    """The double cantilever beam with elements of 2 (two across each arm): the 10-node
    tetrahedra follow the corrected beam theory along the propagation (loads within 5 %,
    crack lengths within 1), where the 4-node ones, which do not bend, are 16 % off at the
    peak and 20 % along the propagation; with as many nodes (elements of 1), the 4-node
    elements are within 12 %."""
    off = {}
    for order in (1, 2):
        mesh, inter, sets = dcb_specimen(60.0, 2.0, 4.0, 20.0, 2.0, order=order)
        assert mesh.order == order and inter.order == order
        problem = CohesiveProblem(mesh, inter, {0: STEEL}, Bilinear(1e5, 5.0, 5.0, 0.5, 0.5))
        res = problem.run(
            [
                (sets["top"], 2, 1.0),
                (sets["bottom"], 2, -1.0),
                (sets["top"], [0, 1], 0.0),
                (sets["bottom"], [0, 1], 0.0),
            ],
            load=3.0,
            steps=60,
            viscosity=1e-3,
        )
        d = 2 * res.loads
        load = res.reactions[:, 0, 2]
        beam, crack = dcb_beam_theory(1000.0, 0.3, 2.0, 4.0, 20.0, 0.5, d)
        growing = d > 4.4
        off[order] = (
            load.max() / beam.max() - 1.0,
            np.abs(load[growing] / beam[growing] - 1.0).max(),
            np.abs(20.0 + res.cracked[growing] / 2.0 - crack[growing]).max(),
        )
    assert off[1][0] > 0.12 and off[1][1] > 0.15  # too stiff: the 4-node elements lock
    assert abs(off[2][0]) < 0.1 and off[2][1] < 0.05 and off[2][2] < 1.0
