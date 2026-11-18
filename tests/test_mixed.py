"""Mixed-mode cracks in 3D (modes I, II and III) in small and finite strains: the geometry
of kinked cracks, the direction of kinking of Erdogan & Sih, the G-θ method and the
interaction integrals on an edge-cracked plate (against the handbook, the exact antiplane
solution and the energy of the body; the modes of superposed loadings; the contact of
closed lips), the neo-Hookean plate (its energy, the small-strain limit, the objectivity
under a superposed rotation), a penny crack in shear (the three modes along a curved
front, against Kassir & Sih), and kinked cracks remeshed by MMG, which grow towards the
local symmetry (K_II → 0)."""

import math

import numpy as np
import pytest

from morphogen.fracture import lefm, sent_factor
from morphogen.fracture.lefm import Crack
from morphogen.fracture.mixed import (
    KinkedCrack,
    energy_release_rate,
    kink_angle,
    propagate_kinked,
    solve_crack,
)
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage
from morphogen.mesh.conforming import has_mmg

W, H, A = 16.0, 32.0, 4.5
PLATE = Material(young=1000.0, poisson=0.0)  # nu = 0: plane stress and strain alike, no 3D effect
RUBBER = Material(young=1000.0, poisson=0.3)
CRACK = Crack("straight", (0.0, 16.0, 0.0), (0, 1, 0), A, direction=(1, 0, 0))
K_I = math.sqrt(math.pi * A) * sent_factor(A / W)  # unit tension
K_III = math.sqrt(math.pi * A) * math.sqrt(2 * W / (math.pi * A) * math.tan(math.pi * A / (2 * W)))


def _plate(h, layers=2):
    shape = (round(W / h), round(H / h), layers)
    image = VoxelImage(np.zeros(shape, np.uint8), spacing=(h, h, h), periodic=False)
    return voxel_tet_mesh(image)


def _faces(m, traction_top, traction_right=None):
    """Nodal forces of tractions on the faces y = H (and -traction on y = 0), x = W (and
    -traction on x = 0)."""
    f = lefm.face_forces(m, lambda c: c[:, 1] > H - 1e-9, traction_top)
    f += lefm.face_forces(m, lambda c: c[:, 1] < 1e-9, -np.asarray(traction_top, dtype=float))
    if traction_right is not None:
        f += lefm.face_forces(m, lambda c: c[:, 0] > W - 1e-9, traction_right)
        f += lefm.face_forces(m, lambda c: c[:, 0] < 1e-9, -np.asarray(traction_right, dtype=float))
    return f


def _tension(m):
    return _faces(m, [0, 1.0, 0])


def _shear(m):  # uniform in-plane shear: antisymmetric about the crack, pure mode II
    return _faces(m, [1.0, 0, 0], [0, 1.0, 0])


def _nearest(m, point):
    return int(np.argmin(np.linalg.norm(m.nodes - np.asarray(point), axis=1)))


def _supports(m):  # the rigid motions only (self-equilibrated forces)
    a, b = _nearest(m, (W, H / 2, 0.0)), _nearest(m, (W, H, 0.0))
    return [(a, [0, 1], 0.0), (b, 0, 0.0), (np.flatnonzero(m.nodes[:, 2] < 1e-9), 2, 0.0)]


def _grips(u, rotation=None):
    """The faces y = 0 and H clamped, pulled apart by 2u, then turned by ``rotation``."""

    def entries(m):
        out = []
        for nodes, sign in ((np.flatnonzero(m.nodes[:, 1] > H - 1e-9), 1.0),
                            (np.flatnonzero(m.nodes[:, 1] < 1e-9), -1.0)):  # fmt: skip
            disp = np.zeros((len(nodes), 3))
            disp[:, 1] = sign * u
            if rotation is not None:
                centre = np.array([W / 2, H / 2, 0.5])
                disp = (m.nodes[nodes] + disp - centre) @ rotation.T + centre - m.nodes[nodes]
            out.append((nodes, [0, 1, 2], disp))
        return out

    return entries


def _rotation(degrees, axis):
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    a = math.radians(degrees)
    return np.eye(3) + math.sin(a) * cross + (1 - math.cos(a)) * cross @ cross


def test_kinked_crack_geometry():
    crack = KinkedCrack(((0.0, 1.0, 0.3), (2.0, 1.0, 0.7)))  # in the plane of the first point
    assert crack.points[1] == (2.0, 1.0, 0.3) and crack.size == crack.last == 2.0
    x = np.array([[1.0, 1.5, 5.0], [3.0, 0.0, -2.0]])
    np.testing.assert_allclose(crack.phi(x), [0.5, -1.0])  # the normal z × x = y
    np.testing.assert_allclose(crack.psi(x), [-1.0, 1.0])
    r, n, d, t, s = crack.frame(x)
    np.testing.assert_allclose(r, [math.hypot(1.0, 0.5), math.sqrt(2.0)])
    assert np.allclose(n, [0, 1, 0]) and np.allclose(d, [1, 0, 0]) and np.allclose(t, [0, 0, 1])
    np.testing.assert_allclose(s, [4.7, -2.3])
    kinked = crack.grown(1.0, math.radians(30.0))
    np.testing.assert_allclose(kinked.tip, [2.0 + math.sqrt(3) / 2, 1.5, 0.3])
    assert kinked.last == pytest.approx(1.0) and kinked.size == pytest.approx(3.0)
    # across the kink: below the new segment, above the first, in the wedge of the corner
    y = np.array([[2.5, 1.0, 0.0], [1.9, 1.1, 0.0], [2.05, 0.9, 0.0]])
    np.testing.assert_allclose(kinked.phi(y), [-0.25, 0.1, -math.hypot(0.05, 0.1)], atol=1e-12)
    np.testing.assert_allclose(kinked.frame(y)[1], [[-0.5, math.sqrt(3) / 2, 0.0]] * 3)
    assert crack.grown(0.5).points == ((0.0, 1.0, 0.3), (2.5, 1.0, 0.3))  # straight: longer
    shorter = kinked.grown(-0.25)
    assert len(shorter.points) == 3 and shorter.last == pytest.approx(0.75)
    for bad in ((-1.0, 0.0), (-0.1, 0.2)):
        with pytest.raises(ValueError, match="shortened"):
            kinked.grown(*bad)
    with pytest.raises(ValueError, match="distinct"):
        KinkedCrack(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))  # along the front: one point
    planar = KinkedCrack.of(CRACK)
    assert planar.axis == (0.0, 0.0, 1.0)
    np.testing.assert_allclose(planar.phi(x), CRACK.phi(x))
    np.testing.assert_allclose(planar.psi(x), CRACK.psi(x))
    points, lengths = planar.front((0, 0, 0), (W, H, 2.0))
    assert lengths.sum() == pytest.approx(2.0) and np.allclose(points[:, :2], [A, 16.0])


def test_auxiliary_fields_keep_the_side_of_the_crack():
    """Behind the front the auxiliary fields are differenced on the side of each point,
    even within a step of the plane of the crack (an element flat along the lips), and
    continued across it on the side given (lips moved by the mesh)."""
    from morphogen.fracture.mixed import _auxiliary_gradients

    x1 = np.array([-0.5])
    above = _auxiliary_gradients(x1, np.array([1e-3]), 500.0, 0.3)
    np.testing.assert_allclose(_auxiliary_gradients(x1, np.array([1e-9]), 500.0, 0.3), above,
                               rtol=1e-2, atol=1e-2 * np.abs(above).max())  # fmt: skip
    continued = _auxiliary_gradients(x1, np.array([-1e-4]), 500.0, 0.3, np.array([np.pi]))
    np.testing.assert_allclose(continued, above, rtol=1e-2, atol=1e-2 * np.abs(above).max())


def test_kink_angle_of_the_largest_hoop_stress():
    assert kink_angle(1.0, 0.0) == 0.0
    assert math.degrees(kink_angle(0.0, 1.0)) == pytest.approx(-70.5288, abs=1e-3)
    assert kink_angle(1.0, -0.5) == pytest.approx(-kink_angle(1.0, 0.5))
    theta = np.linspace(-math.pi + 1e-3, math.pi - 1e-3, 200001)
    for k1, k2 in ((1.0, 0.3), (1.0, 1.0), (0.2, -1.0)):
        hoop = np.cos(theta / 2) * (k1 * np.cos(theta / 2) ** 2 - 1.5 * k2 * np.sin(theta))
        assert kink_angle(k1, k2) == pytest.approx(theta[np.argmax(hoop)], abs=1e-4)


def test_mode_one_against_the_handbook_and_the_energy():
    """The edge crack in tension: K_I converging to the handbook (Tada, Paris & Irwin), G
    of the G-θ method independent of its domain, close to the energy of the body and to
    K_I² / E'."""
    ratios = []
    for h in (1.0, 0.5):
        mesh = _plate(h)
        solution = solve_crack(mesh, CRACK, PLATE, _supports, forces=_tension)
        modes = solution.modes()
        ratios.append(modes["K_I"] / K_I)
        assert abs(modes["K_II"]) + abs(modes["K_III"]) < 1e-6 * modes["K_I"]
        assert modes["G_I"] == pytest.approx(modes["G"]) and abs(modes["angle"]) < 1e-6
    assert ratios[0] < ratios[1] and 0.93 < ratios[1] < 1.0  # linear tetrahedra: from below
    g = solution.release_rate()
    for r_out in (2.0, 3.0):
        assert solution.release_rate(r_in=0.5, r_out=r_out) == pytest.approx(g, rel=0.01)
    assert g == pytest.approx(modes["K_I"] ** 2 / PLATE.young, rel=0.02)

    def solve(crack):
        return solve_crack(mesh, crack, PLATE, _supports, forces=_tension)

    assert g == pytest.approx(energy_release_rate(solve, CRACK, 0.5), rel=0.05)


def test_mode_three_of_antiplane_shear():
    """Antiplane shear of the plate (only the displacements along the front free): K_III
    of the interaction integral within 1.5 % of the exact solution of the strip, G of the
    G-θ method converging to it, uniform along the front (stations)."""
    mesh = _plate(0.5, layers=6)
    x = mesh.nodes

    def supports(m):
        corner = _nearest(m, (W, 0.0, 0.0))
        return [(np.arange(m.n_nodes), [0, 1], 0.0), (corner, 2, 0.0)]

    solution = solve_crack(mesh, CRACK, PLATE, supports, forces=lambda m: _faces(m, [0, 0, 1.0]))
    assert x.shape[0] < solution.mesh.n_nodes  # the lips opened
    modes = solution.modes()
    mu = PLATE.young / 2
    assert modes["K_III"] == pytest.approx(K_III, rel=0.015)
    assert modes["G"] == pytest.approx(K_III**2 / (2 * mu), rel=0.06)
    assert modes["G_III"] == pytest.approx(modes["G"])
    assert abs(modes["K_I"]) + abs(modes["K_II"]) < 1e-6 * modes["K_III"]
    s, g = solution.release_rate(stations=3)
    np.testing.assert_allclose(s, [0.0, 1.5, 3.0], atol=1e-9)
    np.testing.assert_allclose(g, modes["G"], rtol=0.02)
    profile = solution.modes(stations=3)
    np.testing.assert_allclose(profile["K_III"], modes["K_III"], rtol=0.02)
    with pytest.raises(ValueError, match="apart"):
        solution.release_rate(stations=8)


@pytest.mark.slow
def test_modes_one_and_two_superposed():
    """In-plane shear (pure mode II, kinking at -70.5°) and tension: the interaction
    integrals superpose, G adds the energies of the two modes, shared between them within
    a few percent; frictionless contact closes the lips under compression, leaving the
    factors of the shear alone."""
    mesh = _plate(0.5)

    def solve(forces, crack=CRACK, **options):
        return solve_crack(mesh, crack, PLATE, _supports, forces=forces, **options)

    one, two = solve(_tension).modes(), solve(_shear).modes()
    assert abs(two["K_I"]) < 1e-6 * two["K_II"] and two["G_II"] == pytest.approx(two["G"])
    assert math.degrees(two["angle"]) == pytest.approx(-70.5288, abs=1e-3)
    assert two["G"] == pytest.approx(
        energy_release_rate(lambda c: solve(_shear, c), CRACK, 0.5), rel=0.03
    )
    both = solve(lambda m: _tension(m) + 0.7 * _shear(m)).modes()
    assert both["K_I"] == pytest.approx(one["K_I"] + 0.7 * two["K_I"], rel=1e-9)
    assert both["K_II"] == pytest.approx(one["K_II"] + 0.7 * two["K_II"], rel=1e-9)
    assert both["G"] == pytest.approx(one["G"] + 0.49 * two["G"], rel=1e-3)
    assert both["G_I"] == pytest.approx(one["G"], rel=0.02)
    assert both["G_II"] == pytest.approx(0.49 * two["G"], rel=0.04)
    assert both["angle"] == pytest.approx(kink_angle(both["K_I"], both["K_II"]))
    # compression and shear: the lips interpenetrate, unless in contact
    closing = lambda m: -_tension(m) + _shear(m)  # noqa: E731
    free = solve(closing)
    jump = free.displacement[free.faces.plus] - free.displacement[free.faces.minus]
    assert jump[..., 1].min() < -1e-3 and free.modes()["K_I"] < -0.5 * one["K_I"]
    closed = solve(closing, contact=True, steps=1)
    jump = closed.displacement[closed.faces.plus] - closed.displacement[closed.faces.minus]
    assert jump[..., 1].min() > -1e-4  # the penalty
    modes = closed.modes()
    assert abs(modes["K_I"]) < 0.02 * two["K_II"]
    assert modes["K_II"] == pytest.approx(two["K_II"], rel=0.01)
    assert modes["G"] == pytest.approx(two["G"], rel=0.02)


@pytest.mark.slow
def test_finite_strains_energy_and_objectivity():
    """The neo-Hookean plate pulled by its grips: G of the G-θ method that of small strains
    at small loads, softer at large ones and close to the energy of the body there; under
    a superposed rotation of the grips, the same deformation turned, the same G and
    factors (small strains are not objective)."""
    mesh = _plate(1.0)

    def finite(u, rotation=None, crack=CRACK, steps=4):
        return solve_crack(
            mesh, crack, RUBBER, _grips(u, rotation), kinematics="finite", steps=steps
        )

    def small(u, rotation=None):
        return solve_crack(mesh, CRACK, RUBBER, _grips(u, rotation)).release_rate()

    assert finite(0.01, steps=1).release_rate() == pytest.approx(small(0.01), rel=0.003)
    big = finite(0.8)
    g = big.release_rate()
    assert g < 0.95 * small(0.8)
    energy = energy_release_rate(lambda c: finite(0.8, crack=c), CRACK, 0.5)
    assert g == pytest.approx(energy, rel=0.04)
    rotation = _rotation(20.0, (1, 2, 3))
    turned = finite(0.8, rotation)
    centre = np.array([W / 2, H / 2, 0.5])
    expected = (big.mesh.nodes + big.displacement - centre) @ rotation.T + centre
    np.testing.assert_allclose(turned.mesh.nodes + turned.displacement, expected, atol=1e-5)
    assert turned.release_rate() == pytest.approx(g, rel=1e-5)
    modes, turned_modes = big.modes(), turned.modes()
    assert turned_modes["K_I"] == pytest.approx(modes["K_I"], rel=1e-5)
    assert abs(turned_modes["K_II"]) + abs(turned_modes["K_III"]) < 1e-4 * modes["K_I"]
    assert abs(small(0.8, rotation) / small(0.8) - 1.0) > 0.5


@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_penny_crack_in_tension_and_shear():
    """A penny crack of radius 1 in a cube of side 4 in tension and shear across its plane:
    along the curved front K_I uniform, K_II and K_III as cos ω and sin ω, their amplitudes
    in the ratio 1 / (1 - ν) (Kassir & Sih, infinite body: within 10 % in the cube on this
    coarse mesh), independent of the domain (the curvature terms of the interaction
    integrals: without them K_II was 30 % low and changed with the domain)."""
    half, nu = 2.0, 0.3
    penny = Crack("penny", (0.0, 0.0, 0.0), (0, 0, 1), 1.0)
    mesh = lefm.crack_mesh((-half,) * 3, (half,) * 3, penny, size=0.35)

    def loads(m):  # sigma_zz = sigma_xz = 1
        f = lefm.face_forces(m, lambda c: c[:, 2] > half - 1e-9, [1.0, 0.0, 1.0])
        f += lefm.face_forces(m, lambda c: c[:, 2] < 1e-9 - half, [-1.0, 0.0, -1.0])
        f += lefm.face_forces(m, lambda c: c[:, 0] > half - 1e-9, [0.0, 0.0, 1.0])
        return f + lefm.face_forces(m, lambda c: c[:, 0] < 1e-9 - half, [0.0, 0.0, -1.0])

    def supports(m):
        a, b, c = (_nearest(m, p) for p in ((-half, -half, -half), (half, -half, -half),
                                             (-half, half, -half)))  # fmt: skip
        return [(a, [0, 1, 2], 0.0), (b, [1, 2], 0.0), (c, 2, 0.0)]

    solution = solve_crack(mesh, penny, RUBBER, supports, forces=loads)
    unit = math.sqrt(1.0 / math.pi)
    k1, k2, k3 = 2 * unit, 4 * unit / (2 - nu), 4 * (1 - nu) * unit / (2 - nu)
    amplitudes = []
    for r_in, r_out in ((None, None), (0.25, 0.5)):
        modes = solution.modes(r_in=r_in, r_out=r_out, stations=8)
        omega = modes["s"] + math.pi / 2  # s from the axis n × x = y, omega from x
        cos, sin = np.cos(omega), np.sin(omega)
        a2, a3 = modes["K_II"] @ cos / (cos @ cos), modes["K_III"] @ sin / (sin @ sin)
        assert np.mean(modes["K_I"]) == pytest.approx(k1, rel=0.1)
        assert a2 == pytest.approx(k2, rel=0.1) and a3 == pytest.approx(k3, rel=0.1)
        assert a2 / a3 == pytest.approx(1 / (1 - nu), rel=0.06)
        assert np.std(modes["K_II"] - a2 * cos) < 0.1 * k2
        amplitudes.append(a2)
    assert amplitudes[1] == pytest.approx(amplitudes[0], rel=0.03)


def _mixed(m):
    return _tension(m) + 0.7 * _shear(m)


def _mesh_of(size):
    return lambda crack: lefm.crack_mesh((0, 0, 0), (W, H, 1.0), crack, size=size)


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_kinked_crack_remeshed_towards_local_symmetry():
    """The edge crack in tension and shear kinks at the angle of Erdogan & Sih: on the
    mesh rebuilt by MMG for the kinked crack (its front along edges), K_II nearly vanishes
    (it stays without the kink); G of the G-θ method independent of its domain, close to
    the energy of its factors and to that of the body (the crack shorter and longer by an
    element, each meshed for itself)."""
    crack = KinkedCrack.of(CRACK)
    mesh_of = _mesh_of(0.75)

    def solve(c):
        return solve_crack(mesh_of(c), c, PLATE, _supports, forces=_mixed)

    start = solve(crack).modes()
    assert start["K_II"] / start["K_I"] == pytest.approx(0.52, abs=0.03)
    assert math.degrees(start["angle"]) == pytest.approx(-41.2, abs=1.5)
    ahead = solve(crack.grown(2.5)).modes()
    assert ahead["K_II"] / ahead["K_I"] > 0.3
    kinked = crack.grown(2.5, start["angle"])
    solution = solve(kinked)
    assert solution.mesh.metadata["front_nodes_moved"] > 0
    assert solution.area() == pytest.approx(kinked.size * 1.0, rel=0.02)
    modes = solution.modes()
    assert abs(modes["K_II"] / modes["K_I"]) < 0.15
    g = solution.release_rate()
    assert solution.release_rate(r_in=0.75, r_out=1.5) == pytest.approx(g, rel=0.03)
    assert g == pytest.approx((modes["K_I"] ** 2 + modes["K_II"] ** 2) / PLATE.young, rel=0.03)
    assert g == pytest.approx(energy_release_rate(solve, kinked, 0.75), rel=0.05)


@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_propagation_of_a_kinked_crack():
    """Fatigue growth under tension and shear, remeshed at each step: after the first kink
    the path turns little, towards the normal of the largest principal stress (-27°), and
    Paris' law counts the cycles."""
    growth = propagate_kinked(_mesh_of(1.0), KinkedCrack.of(CRACK), PLATE, _supports,
                              forces=_mixed, increment=1.5, steps=2, paris=(1e-3, 3.0))  # fmt: skip
    rows = growth.rows()
    assert (
        len(rows) == 3 and rows[0]["angle_deg"] < -35.0 and rows[2]["length"] == pytest.approx(7.5)
    )
    assert abs(growth.factors[1, 1] / growth.factors[1, 0]) < 0.2
    assert abs(growth.angles[2]) < abs(growth.angles[0]) / 3
    last = np.subtract(growth.cracks[-1].points[-1], growth.cracks[-1].points[-2])
    assert -40.0 < math.degrees(math.atan2(last[1], last[0])) < -20.0
    assert np.all(np.diff(growth.rates) > 0) and np.all(np.diff(growth.cycles) > 0)
