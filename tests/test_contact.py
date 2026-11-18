"""Contact, self-contact and friction of the surfaces of meshes (morphogen.contact): the
nearest point of a face (flat and curved, its edges, the rim of a surface); the distances
to the rigid obstacles; the surfaces of a mesh, their points and rims, bodies assembled;
two blocks of meshes that do not match pressed together (a uniform pressure, the balance of
the works; 4- and 10-node elements, one pass and two); a block slid far along another
(stick, then T = μ N, the work of friction); Hertz's line contact of a rigid cylinder on a
slab; a platen that rubs; the augmented Lagrangian; a slot that closes on itself (the
stiffness of the solid block); the results and the errors; periodic cells: layers that
close on their images through the faces of the cell (the stress that the contact carries,
strain- and stress-controlled paths), a cell cut through its slot or beside it (the same
response); and, slow, Hertz's contact of a sphere and a clip whose arms come to rest on
each other in finite strains."""

import dataclasses
import json
import math

import numpy as np
import pytest

from morphogen import cli, io
from morphogen.config import ConfigError, load_config
from morphogen.contact import (
    ContactCell,
    ContactProblem,
    Obstacle,
    Pair,
    Surface,
    assemble,
    effective_modulus,
    flat_punch,
    hertz_cylinder,
    hertz_sphere,
    kernels,
    mindlin_stick_radius,
    sneddon_cone,
    walls,
)
from morphogen.contact.periodic import SERIES
from morphogen.homogenization import homogenize
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage

STEEL = Material(young=1000.0, poisson=0.3)
NO_RIM = np.zeros(3, dtype=np.bool_)


def _block(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), order=1):
    image = VoxelImage(np.zeros(shape, np.uint8), spacing=spacing, origin=origin, periodic=False)
    mesh = voxel_tet_mesh(image)
    return mesh.quadratic() if order == 2 else mesh


def _two_blocks(order=1, long=False):
    """A block of 2 x 2 x 2 (or 6 x 2 x 2: a track) and, on it, one of the same base
    meshed finer: the mesh of both, the faces of the lower one and of the upper one that
    touch, the nodes of the base and of the top."""
    lower = _block((6 if long else 2, 2, 2), order=order)
    upper = _block((3, 3, 2), (2 / 3, 2 / 3, 1.0), (0.5 if long else 0.0, 0.0, 2.0), order)
    mesh = assemble([lower, upper], phases=[0, 1])
    at = lambda c: np.abs(c[:, 2] - 2.0) < 1e-9  # noqa: E731
    x = mesh.nodes
    base, top = np.flatnonzero(x[:, 2] < 1e-9), np.flatnonzero(x[:, 2] > 4 - 1e-9)
    return mesh, Surface.of(mesh, at, phases=0), Surface.of(mesh, at, phases=1), base, top


def test_nearest_point_of_a_face():
    rng = np.random.default_rng(0)
    a, b, c = np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.2, 0.9, 0.1])
    x3 = np.stack([a, b, c])
    grid = np.array(
        [[u, v] for u in np.linspace(0, 1, 101) for v in np.linspace(0, 1, 101) if u + v <= 1]
    )
    cloud = (1 - grid.sum(axis=1))[:, None] * a + grid[:, :1] * b + grid[:, 1:] * c
    for _ in range(100):  # against a search over the triangle
        p = rng.uniform(-1, 2, 3)
        xi, eta, d2, how = kernels.closest(3, x3, p, 0.0, NO_RIM)
        brute = ((cloud - p) ** 2).sum(axis=1).min()
        assert d2 <= brute + 1e-12 and math.sqrt(brute) - math.sqrt(d2) < 2e-3
        assert xi >= 0 and eta >= 0 and xi + eta <= 1 + 1e-12 and how in (0, 1)
    # the foot of a perpendicular, a point over an edge, a point beyond the rim
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    foot = 0.5 * a + 0.3 * b + 0.2 * c
    xi, eta, d2, how = kernels.closest(3, x3, foot + 0.4 * normal, 0.0, NO_RIM)
    assert (xi, eta, how) == pytest.approx((0.3, 0.2, 1)) and d2 == pytest.approx(0.16)
    over = 0.5 * (a + b) - 0.01 * np.cross(normal, b - a) + 0.1 * normal  # beside the edge 0-1
    xi, eta, _, how = kernels.closest(3, x3, over, 0.0, NO_RIM)
    assert how == 0 and eta == 0 and xi == pytest.approx(0.5)
    assert kernels.closest(3, x3, over, 0.0, np.array([True, False, False]))[3] == -1
    xi, eta, _, how = kernels.closest(3, x3, over, 0.05, NO_RIM)  # within the slack: the plane
    assert how == 1 and eta < 0
    assert kernels.closest(3, x3, over, 0.05, np.array([True, False, False]))[3] == -1
    # a curved face of six nodes on a sphere: the foot of the normal at a point is that point
    on = lambda q: 2.0 * np.asarray(q, dtype=float) / np.linalg.norm(q)  # noqa: E731
    corners = [on([1, 0, 1.2]), on([0, 1, 1.2]), on([-0.3, -0.2, 1.5])]
    mids = [on(corners[i] + corners[(i + 1) % 3]) for i in range(3)]
    x6 = np.stack(corners + mids)
    point, t1, t2, n = np.empty(3), np.empty(3), np.empty(3), np.empty(3)
    for xi, eta in ((0.2, 0.3), (0.6, 0.1), (0.05, 0.9), (1 / 3, 1 / 3)):
        kernels._at(6, x6, xi, eta, point, t1, t2)
        kernels._normal(t1, t2, n)
        for height in (0.2, -0.1):
            u, v, d2, how = kernels.closest(6, x6, point + height * n, 0.0, NO_RIM)
            assert (u, v) == pytest.approx((xi, eta), abs=1e-9) and how == 1
            assert d2 == pytest.approx(height**2)
    kernels._at(6, x6, 0.5, 0.0, point, t1, t2)  # beside its edge 0-1: the edge
    kernels._normal(t1, t2, n)
    u, v, _, how = kernels.closest(6, x6, point - 0.3 * np.cross(n, t1) + 0.05 * n, 0.0, NO_RIM)
    assert how == 0 and v == 0 and 0 < u < 1


def test_obstacles():
    """The signed distances of the rigid obstacles: their gradients are their normals."""
    rng = np.random.default_rng(1)
    skin = Surface.of(_block((1, 1, 1)))
    obstacles = [
        Obstacle.plane(skin, (0.1, 0.2, 0.3), (0, 3, 4)),
        Obstacle.sphere(skin, (0.1, 0.2, 0.3), 0.7),
        Obstacle.sphere(skin, (0.1, 0.2, 0.3), 0.7, inside=True),
        Obstacle.cylinder(skin, (0.1, 0.2, 0.3), (0, 3, 4), 0.5),
        Obstacle.cone(skin, (0.1, 0.2, 0.3), (0, 3, 4), 60.0),
    ]
    normal, other = np.empty(3), np.empty(3)
    for obstacle in obstacles:
        geometry = obstacle.geometry()
        for _ in range(50):
            p = rng.uniform(-1, 1, 3)
            kernels.obstacle_distance(obstacle.kind, geometry, p, normal)
            gradient = np.empty(3)
            for c in range(3):
                e = np.zeros(3)
                e[c] = 1e-6
                above = kernels.obstacle_distance(obstacle.kind, geometry, p + e, other)
                below = kernels.obstacle_distance(obstacle.kind, geometry, p - e, other)
                gradient[c] = (above - below) / 2e-6
            np.testing.assert_allclose(gradient, normal, atol=1e-6)
            assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert obstacles[1].distance([[0.1, 0.2, 2.3], [0.1, 0.2, 0.3]]) == pytest.approx([1.3, -0.7])
    assert obstacles[2].distance([0.1, 0.2, 0.3]) == pytest.approx([0.7])  # held inside
    cone = Obstacle.cone(skin, (0, 0, 0), (0, 0, 1), 45.0)
    side, within, behind = cone.distance([[1.0, 0, 1.0], [0, 0, 1.0], [0, 0, -2.0]])
    assert side == pytest.approx(0, abs=1e-12) and within == pytest.approx(-math.sqrt(0.5))
    assert behind == pytest.approx(2.0)  # the apex
    # motions: a vector grown with the load factor, or a function of it
    moving = Obstacle.plane(skin, (0, 0, 1), (0, 0, -1), motion=(0, 0, -0.5), name="platen")
    assert moving.distance([0, 0, 0.4], 1.0) == pytest.approx([0.1]) and moving.name == "platen"
    staged = Obstacle.plane(skin, (0, 0, 1), (0, 0, -1), motion=lambda t: [0.1 * t, 0, 0])
    np.testing.assert_allclose(staged.geometry(2.0)[:3], [0.2, 0, 1])
    for make, match in (
        (lambda: Obstacle.sphere(skin, (0, 0, 0), -1.0), "radius"),
        (lambda: Obstacle.cone(skin, (0, 0, 0), (0, 0, 1), 95.0), "half-angle"),
        (lambda: Obstacle.plane(skin, (0, 0, 0), (0, 0, 1), inside=True), "inside"),
        (lambda: Obstacle.plane(skin, (0, 0, 0), (0, 0, 1), friction=-0.1), "friction"),
        (lambda: Obstacle.plane(skin, (0, 0), (0, 0, 1)), "three components"),
    ):
        with pytest.raises(ValueError, match=match):
            make()


def test_surfaces_of_a_mesh():
    for order in (1, 2):
        mesh, lower, upper, _, _ = _two_blocks(order)
        x = mesh.nodes
        skin = Surface.of(mesh)
        assert skin.order == order and skin.faces.shape[1] == (3, 6)[order - 1]
        assert skin.areas(x).sum() == pytest.approx(2 * 24.0)  # two cubes of edge 2
        assert not skin.rims().any()  # closed surfaces
        assert lower.n_faces == 8 and upper.n_faces == 18
        assert lower.areas(x).sum() == pytest.approx(4.0)
        assert lower.rims().sum() == 8 and upper.rims().sum() == 12  # their perimeters
        np.testing.assert_allclose(lower.normals(x), [[0, 0, 1.0]] * 8, atol=1e-12)
        np.testing.assert_allclose(upper.normals(x), [[0, 0, -1.0]] * 18, atol=1e-12)
        face, xi, eta, weight = upper.points(x)
        assert len(face) == 18 * (3, 6)[order - 1] and weight.sum() == pytest.approx(4.0)
        positions = upper.positions(x, face, xi, eta)
        assert np.allclose(positions[:, 2], 2.0) and positions[:, :2].min() > 0
        assert upper.size(x) == pytest.approx((2 * 2 / 3 + 2 / 3 * math.sqrt(2)) / 3)
        assert upper.subset(np.arange(4)).n_faces == 4
        nodes_of_upper = len(np.unique(mesh.elements[mesh.phase == 1]))
        assert mesh.metadata["bodies"] == [mesh.n_nodes - nodes_of_upper, nodes_of_upper]
    with pytest.raises(ValueError, match="three or six"):
        Surface(np.zeros((2, 4), dtype=int))
    with pytest.raises(ValueError, match="one value per face"):
        Surface.of(mesh, lambda c: np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="same order"):
        assemble([_block((1, 1, 1)), _block((1, 1, 1), order=2)])
    with pytest.raises(ValueError, match="same order"):
        Pair(Surface.of(_block((1, 1, 1))), Surface.of(_block((1, 1, 1), order=2)))


def _press(mesh, base, top, depth):
    """The entries of a pressing of the upper block: its top moved down, the base held,
    the rigid motions of each block along the interface taken off."""
    x = mesh.nodes
    pin = lambda *at: int(np.argmin(np.linalg.norm(x - np.array(at), axis=1)))  # noqa: E731
    return [
        (base, 2, 0.0),
        (top, 2, -depth),
        (pin(0, 0, 0), [0, 1], 0.0),
        (pin(2, 0, 0), 1, 0.0),
        (pin(0, 0, 4), [0, 1], 0.0),
        (pin(2, 0, 4), 1, 0.0),
    ]


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
def test_blocks_of_other_meshes_transmit_a_pressure(order, symmetric):
    """Two blocks whose meshes do not match, pressed together without friction: a uniaxial
    stress E δ / (H + E / ε) through both, the penalty ε in series with them; they slide
    freely on each other as they widen."""
    mesh, lower, upper, base, top = _two_blocks(order)
    pair = Pair(upper, lower, symmetric=symmetric)
    problem = ContactProblem(mesh, {0: STEEL, 1: STEEL}, [pair], kinematics="small")
    res = problem.run(_press(mesh, base, top, 0.02), steps=2)
    stiffness = 10 * 1000.0 / lower.size(mesh.nodes)  # the default penalty, of the coarser
    stress = 0.02 / (4 / 1000.0 + 1 / stiffness)
    assert -res.reactions[-1, 1, 2] == pytest.approx(4 * stress, rel=3e-3)
    np.testing.assert_allclose(res.forces[-1, 0], [0, 0, -4 * stress], rtol=3e-3, atol=2e-3)
    assert res.area[-1, 0] == pytest.approx(4.0) and res.area[0, 0] == 0
    points = res.points[0]
    mean = points["pressure"] @ points["weight"] / points["weight"].sum()
    assert mean == pytest.approx(stress, rel=3e-3)
    assert res.pressure[-1, 0] < 1.3 * stress and points["pressure"].min() > 0.3 * stress
    assert res.penetration[-1] == pytest.approx(stress / stiffness, rel=0.4)
    assert res.iterations.max() <= 4
    x, u = mesh.nodes, res.displacement  # they widen freely: the lateral strain ν σ / E
    edge = np.abs(x[:, 0] - 2.0) < 1e-9
    assert u[edge, 0].mean() == pytest.approx(0.3 * stress / 1000.0 * 2, rel=0.02)
    work = res.work
    assert work["external"][-1] == pytest.approx(work["bulk"][-1] + work["contact"][-1], rel=1e-6)
    assert 0 < work["contact"][-1] < 0.05 * work["bulk"][-1] and work["friction"][-1] == 0
    assert "interfaces" not in work and res.cohesive is None and res.names == ["pair1"]


def _on_its_corner(mesh, at=(0.0, 0.0, 0.0)):
    """A mesh turned so that the diagonal of its box is along the axis 3, its corner at
    the origin moved to ``at``: the lowest point of a cube that stands on it."""
    d = np.ones(3) / math.sqrt(3.0)
    a = np.cross(d, [0.0, 0.0, 1.0])
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    turn = np.eye(3) + k + k @ k * ((1.0 - d[2]) / (a @ a))  # takes the diagonal to the axis 3
    return dataclasses.replace(mesh, nodes=mesh.nodes @ turn.T + np.asarray(at, dtype=float))


@pytest.mark.parametrize("order", [1, 2])
def test_contact_points_at_the_nodes(order):
    """The contact points at the nodes of the faces: their weights are shares of the
    areas (a third per vertex of a face of three nodes; 1/20 per vertex, 2/15 per middle
    of an edge and 9/20 at the centroid of a face of six) that integrate polynomials of
    degree 1 (3 for six nodes); two blocks of the same section pressed together transmit
    their uniaxial stress, the nodes of the rim of one on the rim of the other."""
    mesh, lower, upper, base, top = _two_blocks(order)
    x = mesh.nodes
    face, xi, eta, weight = upper.points(x, "nodes")
    assert len(face) == 18 * (3, 7)[order - 1] and weight.sum() == pytest.approx(4.0)
    shares = weight.reshape(18, -1) / upper.areas(x)[:, None]
    expected = [1 / 3] * 3 if order == 1 else [1 / 20] * 3 + [2 / 15] * 3 + [9 / 20]
    np.testing.assert_allclose(shares, np.tile(expected, (18, 1)), rtol=1e-12)
    at = upper.positions(x, face, xi, eta)
    corners = x[upper.faces[:, :3]].reshape(-1, 3)
    assert all(np.isclose(at, c).all(axis=1).any() for c in corners)  # the vertices are points
    # over the square of side 2: the integrals of x, and of x^3 and x^2 y for six nodes
    assert weight @ at[:, 0] == pytest.approx(4.0)
    if order == 2:
        assert weight @ at[:, 0] ** 3 == pytest.approx(8.0)
        assert weight @ (at[:, 0] ** 2 * at[:, 1]) == pytest.approx(16.0 / 3.0)
    with pytest.raises(ValueError, match="contact points"):
        upper.points(x, "vertices")
    with pytest.raises(ValueError, match="contact points among"):
        Pair(upper, lower, points="edges")
    with pytest.raises(ValueError, match="contact points among"):
        Obstacle.plane(upper, (0, 0, 2), (0, 0, 1), points="edges")
    for symmetric in (False, True):
        pair = Pair(upper, lower, points="nodes", symmetric=symmetric)
        problem = ContactProblem(mesh, {0: STEEL, 1: STEEL}, [pair], kinematics="small")
        res = problem.run(_press(mesh, base, top, 0.02), steps=2)
        stiffness = 10 * 1000.0 / lower.size(mesh.nodes)
        stress = 0.02 / (4 / 1000.0 + 1 / stiffness)
        assert -res.reactions[-1, 1, 2] == pytest.approx(4 * stress, rel=5e-3)
        assert res.area[-1, 0] == pytest.approx(4.0) and res.iterations.max() <= 4


@pytest.mark.parametrize("order", [1, 2])
def test_sides_of_a_block_that_stands_on_a_plane_carry_nothing(order):
    """A block that stands on a rigid plane, its whole skin a contact surface: the nodes
    of the edges of its base belong to its sides as well, which the plane sees edge-on.
    The weight of a point at a node is the share of its face that the other surface sees:
    the sides carry nothing, the stress is the uniaxial one (penalty and block in series),
    the area that of the base, as with the points inside the faces."""
    block = _block((2, 2, 2), order=order)
    x = block.nodes
    top = np.flatnonzero(x[:, 2] > 2 - 1e-9)
    pin = lambda *at: int(np.argmin(np.linalg.norm(x - np.array(at), axis=1)))  # noqa: E731
    prescribed = [(top, 2, 0.0), (pin(0, 0, 2), [0, 1], 0.0), (pin(2, 0, 2), 1, 0.0)]
    skin = Surface.of(block)
    stress = 0.02 / (2 / 1000.0 + 1 / 2e4)
    for rule in ("gauss", "nodes"):
        plane = Obstacle.plane(skin, (0, 0, 0), (0, 0, 1), motion=(0, 0, 0.02), stiffness=2e4,
                               points=rule)  # fmt: skip
        problem = ContactProblem(block, {0: STEEL}, obstacles=[plane], kinematics="small")
        res = problem.run(prescribed, load=1.0, steps=2)
        assert res.forces[-1, 0, 2] == pytest.approx(-4 * stress, rel=1e-6)
        assert res.area[-1, 0] == pytest.approx(4.0, rel=1e-9)
        assert res.penetration[-1] == pytest.approx(stress / 2e4, rel=1e-6)
        points = res.points[0]
        pressed = points["pressure"] > 0
        assert points["weight"][pressed].sum() == pytest.approx(4.0, rel=1e-9)
        np.testing.assert_allclose(points["pressure"][pressed], stress, rtol=1e-6)


@pytest.mark.parametrize("order", [1, 2])
def test_corner_of_a_cube_meets_a_plane(order):
    """A cube that stands on a corner, a rigid plane brought up against it: the points
    inside its faces let the corner through (no force until the plane reaches them);
    those at its nodes stop it at once, and with updates of the multipliers the corner
    does not go into the plane."""
    cube = _on_its_corner(_block((2, 2, 2), (0.5, 0.5, 0.5), order=order))
    x = cube.nodes
    corner = int(np.argmin(x[:, 2]))
    assert x[corner, 2] == pytest.approx(0.0, abs=1e-12)
    held = np.flatnonzero(x[:, 2] > 0.5 * x[:, 2].max())
    skin = Surface.of(cube)
    rise = 0.05
    # the lowest points inside the faces are above the plane at the end: nothing touches
    inside = skin.positions(x, *skin.points(x)[:3])[:, 2].min()
    assert inside > rise
    through, stopped = {}, {}
    for rule, augmentations in (("gauss", 0), ("nodes", 0), ("nodes", 4)):
        plane = Obstacle.plane(skin, (0, 0, 0), (0, 0, 1), motion=(0, 0, rise), points=rule)
        problem = ContactProblem(cube, {0: STEEL}, obstacles=[plane], kinematics="small",
                                 augmentations=augmentations, tolerance=1e-6)  # fmt: skip
        res = problem.run([(held, [0, 1, 2], 0.0)], load=1.0, steps=5)
        below = rise - (x[corner, 2] + res.displacement[corner, 2])  # of the plane
        (through if rule == "gauss" else stopped)[augmentations] = (res.forces[-1, 0, 2], below)
    force, below = through[0]
    assert force == 0.0 and below == pytest.approx(rise)  # the corner went through the plane
    force, below = stopped[0]
    assert force < -1.0 and 0.0 < below < 0.2 * rise  # the penalty alone: a small penetration
    tight, closer = stopped[4]
    assert tight < force and closer < 0.2 * below  # multipliers: stiffer, hardly any


@pytest.mark.parametrize("order", [1, 2])
def test_corner_of_a_cube_meets_a_block(order):
    """The same corner pressed into the face of a larger block, two meshed bodies: with
    the points inside its faces the cube sinks into the block without a force; with those
    at its nodes the block carries it, and the two bodies are in equilibrium."""
    block = _block((4, 4, 2), (0.5, 0.5, 0.5), order=order)  # 2 x 2 x 1: its top at 1
    cube = _on_its_corner(_block((2, 2, 2), (0.5, 0.5, 0.5), order=order), at=(0.9, 1.1, 1.0))
    mesh = assemble([block, cube], phases=[0, 1])
    x = mesh.nodes
    face = Surface.of(mesh, lambda c: np.abs(c[:, 2] - 1.0) < 1e-9, phases=0)
    skin = Surface.of(mesh, phases=1)
    base = np.flatnonzero(x[:, 2] < 1e-9)
    upper = np.flatnonzero(x[:, 2] > 1.0 + 0.5 * (x[:, 2].max() - 1.0))
    prescribed = [(base, [0, 1, 2], 0.0), (upper, [0, 1], 0.0), (upper, 2, -0.03)]
    forces = {}
    for rule in ("gauss", "nodes"):
        pair = Pair(skin, face, points=rule)
        problem = ContactProblem(mesh, {0: STEEL, 1: STEEL}, [pair], kinematics="small")
        res = problem.run(prescribed, steps=3)
        forces[rule] = res.forces[-1, 0]
        if rule == "nodes":  # what the cube gives the block, the base of the block takes
            assert res.iterations.max() <= 5 and res.area[-1, 0] > 0
            carried = np.abs(res.forces[-1, 0]).max()
            np.testing.assert_allclose(
                res.reactions[-1, 0], -res.forces[-1, 0], atol=1e-4 * carried
            )
    assert not forces["gauss"].any()
    assert (
        forces["nodes"][2] < -1.0 and np.abs(forces["nodes"][:2]).max() < 0.1 * -forces["nodes"][2]
    )


def test_augmented_lagrangian_takes_the_penetration_off():
    mesh, lower, upper, base, top = _two_blocks()
    pair = Pair(upper, lower, stiffness=2e3)  # a soft penalty: two elements of the blocks
    soft = ContactProblem(mesh, {0: STEEL, 1: STEEL}, [pair], kinematics="small")
    res = soft.run(_press(mesh, base, top, 0.02), steps=2)
    stress = 0.02 / (4 / 1000.0 + 1 / 2e3)
    assert -res.reactions[-1, 1, 2] == pytest.approx(4 * stress, rel=3e-3)
    assert res.penetration[-1] == pytest.approx(stress / 2e3, rel=0.2)
    tight = ContactProblem(mesh, {0: STEEL, 1: STEEL}, [pair], kinematics="small",
                           augmentations=30, tolerance=2e-5)  # fmt: skip
    res = tight.run(_press(mesh, base, top, 0.02), steps=2)
    assert res.penetration[-1] <= 2e-5
    assert -res.reactions[-1, 1, 2] == pytest.approx(4 * 0.02 * 1000.0 / 4, rel=5e-3)


@pytest.mark.parametrize("passes", ["slider", "track", "both"])
def test_block_slides_far_at_coulomb(passes):
    """A block pressed on a track, then slid along it by more than its own length: the
    tangential force is μ N whichever surface carries the points; the work of the contact
    is that of friction."""
    mesh, track, slider, base, top = _two_blocks(long=True)
    pair = {
        "slider": Pair(slider, track, friction=0.3),
        "track": Pair(track, slider, friction=0.3),
        "both": Pair(slider, track, friction=0.3, symmetric=True),
    }[passes]
    problem = ContactProblem(mesh, {0: STEEL, 1: STEEL}, [pair], kinematics="small")
    prescribed = [
        (base, [0, 1, 2], 0.0),
        (top, 2, lambda t: -0.02 * min(t, 1.0)),
        (top, 0, lambda t: 2.5 * max(t - 1.0, 0.0)),
        (top, 1, 0.0),
    ]
    res = problem.run(prescribed, load=2.0, steps=40)
    normal, tangential = -res.reactions[:, 1, 2], res.reactions[:, 2, 0]
    pressed = np.searchsorted(res.loads, 1.0 - 1e-9)
    assert abs(tangential[pressed]) < 0.05 * normal[pressed]  # their unequal widening only
    sliding = res.loads > 1.1
    ratio = tangential[sliding] / normal[sliding]
    assert np.all(np.abs(ratio - 0.3) < 0.02) and ratio.mean() == pytest.approx(0.3, rel=0.01)
    assert res.area[sliding, 0] == pytest.approx(4.0, rel=0.1)
    sign = -1.0 if passes == "track" else 1.0  # of the slave surface on its master
    np.testing.assert_allclose(sign * res.forces[sliding, 0, 0], tangential[sliding], rtol=1e-4)
    work = res.work
    assert work["external"][-1] == pytest.approx(work["bulk"][-1] + work["contact"][-1], rel=1e-6)
    assert work["friction"][-1] == pytest.approx(work["contact"][-1], rel=0.04)
    assert np.all(np.diff(work["friction"]) >= 0)
    assert work["friction"][-1] == pytest.approx(0.3 * normal[sliding].mean() * 2.5, rel=0.03)
    dissipated = res.points[0]["dissipated"]
    assert dissipated.min() >= 0 and dissipated.max() > 0


def test_hertz_line_contact():
    """A rigid cylinder pressed on a slab in plane strain (half of it, by symmetry): the
    largest pressure and the half-width of Hertz for the force that it takes, and his
    pressure sqrt(1 - x² / b²) along the contact."""
    size, width, radius = 0.5, 8.0, 20.0
    mesh = _block((16, 1, 16), (size,) * 3, order=2)
    x = mesh.nodes
    top = Surface.of(mesh, lambda c: np.abs(c[:, 2] - width) < 1e-9)
    cylinder = Obstacle.cylinder(top, (0, 0, width + radius), (0, 1, 0), radius,
                                 motion=(0, 0, -0.06))  # fmt: skip
    problem = ContactProblem(mesh, {0: STEEL}, obstacles=[cylinder], kinematics="small")
    prescribed = [
        (np.arange(len(x)), 1, 0.0),
        (np.flatnonzero(x[:, 0] < 1e-9), 0, 0.0),
        (np.flatnonzero(x[:, 2] < 1e-9), 2, 0.0),
    ]
    res = problem.run(prescribed, steps=4)
    assert res.names == ["cylinder"] and np.all(np.diff(res.forces[:, 0, 2]) > 0)
    force = 2 * res.forces[-1, 0, 2] / size  # per unit length, of the two halves
    exact = hertz_cylinder(radius, effective_modulus(1000.0, 0.3), force)
    assert res.pressure[-1, 0] == pytest.approx(exact["pressure"], rel=0.05)
    assert res.area[-1, 0] / size == pytest.approx(exact["half_width"], abs=0.1)
    points = res.points[0]
    on = points["pressure"] > 0
    profile = exact["pressure"] * np.sqrt(
        np.maximum(1 - (points["position"][on, 0] / exact["half_width"]) ** 2, 0)
    )
    assert np.sqrt(np.mean((points["pressure"][on] - profile) ** 2)) < 0.06 * exact["pressure"]
    # the cylinder does the work that the slab stores, and a little in the penalty
    work = res.work
    assert work["external"][-1] == pytest.approx(0, abs=1e-12)
    assert work["obstacles"][-1] == pytest.approx(work["bulk"][-1], rel=0.03)
    assert work["obstacles"][-1] > work["bulk"][-1] and work["friction"][-1] == 0
    assert np.all(np.abs(points["gap"][on]) < 5e-3) and res.penetration[-1] < 5e-3


def test_platen_that_rubs():
    """A block pressed by a rigid platen, which then slides: it takes μ N along its way,
    and the base takes the opposite of what the platen gives."""
    mesh = _block((2, 2, 2))
    x = mesh.nodes
    top = Surface.of(mesh, lambda c: np.abs(c[:, 2] - 2.0) < 1e-9)
    platen = Obstacle.plane(
        top, (0, 0, 2), (0, 0, -1), friction=0.25, name="platen",
        motion=lambda t: [0.05 * max(t - 1.0, 0.0), 0.0, -0.02 * min(t, 1.0)],
    )  # fmt: skip
    problem = ContactProblem(mesh, {0: STEEL}, obstacles=[platen], kinematics="small")
    base = np.flatnonzero(x[:, 2] < 1e-9)
    res = problem.run([(base, [0, 1, 2], 0.0)], load=2.0, steps=16)
    on_platen = res.forces[:, 0]
    np.testing.assert_allclose(on_platen, res.reactions[:, 0], atol=1e-6 * abs(on_platen).max())
    assert on_platen[-1, 2] > 0 and on_platen[-1, 0] == pytest.approx(
        -0.25 * on_platen[-1, 2], rel=1e-3
    )
    pressed = np.searchsorted(res.loads, 1.0 - 1e-9)
    assert abs(on_platen[pressed, 0]) < 1e-6 * on_platen[pressed, 2]
    work = res.work
    assert work["friction"][-1] > 0 and work["obstacles"][-1] > work["bulk"][-1]
    balance = work["bulk"][-1] + work["friction"][-1]
    assert work["obstacles"][-1] == pytest.approx(balance, rel=0.05)
    assert "Fx_platen" in res.rows()[-1] and res.rows()[-1]["area_platen"] == pytest.approx(4.0)


def _slotted(order=1):
    """A block of 4 x 2.5 (one voxel of 0.25 thick) with a slot of 2 x 0.25 across it."""
    data = np.ones((16, 1, 10), np.uint8)
    data[4:12, :, 4:5] = 0
    image = VoxelImage(data, spacing=(0.25,) * 3, periodic=False)
    mesh = voxel_tet_mesh(image).subset([1])
    return mesh.quadratic() if order == 2 else mesh


@pytest.mark.parametrize("friction", [0.0, 0.3])
def test_slot_closes_on_itself(friction, tmp_path):
    """A block with a slot, compressed: nothing touches until the slot closes; then its
    walls press on each other and the block gets the stiffness of a solid one."""
    mesh = _slotted()
    x = mesh.nodes
    pair = Pair(Surface.of(mesh), friction=friction)  # the whole surface, against itself
    problem = ContactProblem(mesh, {1: STEEL}, [pair], kinematics="small")
    prescribed = [
        (np.arange(len(x)), 1, 0.0),
        (np.flatnonzero(x[:, 2] < 1e-9), 2, 0.0),
        (np.flatnonzero((x[:, 2] < 1e-9) & (x[:, 0] < 1e-9)), 0, 0.0),
        (np.flatnonzero(x[:, 2] > 2.5 - 1e-9), 2, -0.6),
    ]
    res = problem.run(prescribed, steps=24, fields=True)
    force, moved = -res.reactions[:, 3, 2], 0.6 * res.loads
    open_ = moved < 0.24
    assert np.all(res.area[open_, 0] == 0) and np.all(res.pressure[open_, 0] == 0)
    before = np.polyfit(moved[open_], force[open_], 1)[0]
    np.testing.assert_allclose(force[open_], before * moved[open_], rtol=1e-6, atol=1e-9)
    after = np.polyfit(moved[-5:], force[-5:], 1)[0]
    solid = 1000.0 / (1 - 0.3**2) * 4.0 * 0.25 / (2.5 - 0.25)  # plane strain, the slot closed
    assert after > 1.6 * before and 0.88 * solid < after < solid
    assert 0.3 < res.area[-1, 0] <= 0.5 + 1e-9  # most of the slot, of 2 x 0.25
    assert res.penetration.max() < 0.02 * 0.25 * 2  # of the size of its faces
    work = res.work
    assert work["external"][-1] == pytest.approx(work["bulk"][-1] + work["contact"][-1], rel=1e-6)
    assert (work["friction"][-1] > 0) == (friction > 0)
    # the results: rows, the pressure at the nodes, the file
    pressure = problem.nodal_pressure()
    walls = (np.abs(x[:, 2] - 1.0) < 1e-9) | (np.abs(x[:, 2] - 1.25) < 1e-9)
    middle = walls & (np.abs(x[:, 0] - 2.0) < 0.6)
    assert pressure[middle].min() > 0 and not pressure[~walls].any()
    path = problem.write(tmp_path / "slot", res)
    assert b"contact_pressure" in path.read_bytes() and len(res.fields) == len(res.loads)
    row = res.rows()[-1]
    assert row["area_pair1"] == res.area[-1, 0] and "work_friction" in row and "R4z" in row


_SHEARS = {"23": 0.0, "13": 0.0, "12": 0.0}


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("lateral", ["held", "free"])
def test_layers_close_on_their_images_through_the_faces_of_a_cell(order, lateral):
    """A periodic stack of layers of 0.9 parted by gaps of 0.1, compressed across them: no
    stress until a layer meets the image of the next through the faces of the cell, then
    the stack of a solid with the penalty in series: uniaxial strain (the layers held) or
    uniaxial stress (the layers free to widen: the macroscopic strains that follow)."""
    data = np.ones((2, 2, 20), np.uint8)
    data[:, :, 0] = data[:, :, 19] = 0  # half a gap at each end of the cell
    mesh = voxel_tet_mesh(VoxelImage(data, spacing=(1.0, 1.0, 0.05), periodic=True)).subset([1])
    if order == 2:
        mesh = mesh.quadratic()
    assert walls(mesh).areas(mesh.nodes).sum() == pytest.approx(8.0)  # the two faces of a layer
    cell = ContactCell(mesh, {1: STEEL}, kinematics="small")
    strain = {"33": -0.15, **_SHEARS}
    if lateral == "held":
        strain.update({"11": 0.0, "22": 0.0})
    res = cell.run({"times": [0, 1], "increments": 6, "strain": strain})
    lam, mu = 1000.0 * 0.3 / (1.3 * 0.4), 1000.0 / 2.6
    modulus = lam + 2 * mu if lateral == "held" else 1000.0
    stiffness = 10 * 1000.0 / walls(mesh).size(mesh.nodes)  # the penalty of the two sides
    contact = res.series
    for k, e33 in enumerate(res.strain[:, 2]):
        closing = -(e33 + 0.1)  # beyond the closure of the gap, in a cell of height 1
        if closing <= 1e-9:
            assert np.abs(res.stress[k]).max() < 1e-6 and contact["contact_pressure"][k] < 1e-6
            continue
        stress = closing / (0.9 / modulus + 1 / stiffness)
        assert res.stress[k, 2] == pytest.approx(-stress, rel=1e-6)
        assert contact["contact_area"][k] == pytest.approx(4.0)  # the whole face of the cell
        assert contact["contact_pressure"][k] == pytest.approx(stress, rel=1e-4)
        assert contact["contact_penetration"][k] == pytest.approx(stress / stiffness, rel=1e-4)
        if lateral == "held":  # the layers alone carry the lateral stress: 0.9 of the cell
            lateral_stress = -0.9 * lam / (lam + 2 * mu) * stress
            np.testing.assert_allclose(res.stress[k, :2], lateral_stress, rtol=1e-6)
        else:
            np.testing.assert_allclose(res.stress[k, :2], 0.0, atol=1e-6)
            np.testing.assert_allclose(res.strain[k, :2], 0.3 * stress / 1000.0, rtol=1e-5)
    assert contact["contact_area"][-1] > 0 and contact["contact_dissipation"][-1] == 0
    points = cell.slave_points()
    assert points["pressure"].min() > 0 and points["position"].shape == (len(points["gap"]), 3)


@pytest.mark.parametrize("kinematics, friction", [("small", 0.0), ("finite", 0.3)])
def test_periodic_cell_cut_through_its_slot(kinematics, friction):
    """A periodic cell with a slot, compressed until the slot closes; the same medium cut
    through the slot (the walls of the slot then go through the faces of the cell) has
    the same response."""
    data = np.ones((16, 1, 10), np.uint8)
    data[4:12, :, 4:5] = 0
    path = {"times": [0, 1], "increments": 12,
            "strain": {"11": 0.0, "22": 0.0, "33": -0.2, **_SHEARS}}  # fmt: skip
    results = []
    for roll in (0, 8):
        image = VoxelImage(np.roll(data, roll, axis=0), spacing=(0.25,) * 3, periodic=True)
        mesh = voxel_tet_mesh(image).subset([1])
        assert walls(mesh).n_faces == 36
        cell = ContactCell(mesh, {1: STEEL}, kinematics=kinematics, friction=friction)
        assert len(cell._images.node) > mesh.n_nodes  # with images of its walls
        results.append(cell.run(path))
    whole, cut = results
    np.testing.assert_allclose(cut.stress, whole.stress, atol=1e-7 * np.abs(whole.stress).max())
    for name in SERIES:
        np.testing.assert_allclose(cut.series[name], whole.series[name],
                                   rtol=1e-6, atol=1e-9)  # fmt: skip
    area, stress = whole.series["contact_area"], -whole.stress[:, 2]
    open_ = area == 0
    assert open_[:4].all() and not open_[-4:].any() and area[-1] > 0.3
    before = stress[3] / 0.05
    after = (stress[-1] - stress[-3]) / (0.2 * 2 / 12)
    assert after > 1.4 * before  # the slot closed: stiffer
    assert (whole.series["contact_dissipation"][-1] > 0) == (friction > 0)
    assert whole.series["contact_penetration"].max() < 0.02


def test_walls_cut_by_the_faces_of_a_cell_go_on_in_its_images():
    """The walls of a slot that the faces of the cell cut have no rim there: a face and the
    image of its neighbour across the cell share their nodes, a period apart."""
    data = np.ones((16, 1, 10), np.uint8)
    data[4:12, :, 4:5] = 0
    image = VoxelImage(np.roll(data, 8, axis=0), spacing=(0.25,) * 3, periodic=True)
    mesh = voxel_tet_mesh(image).subset([1])
    cell = ContactCell(mesh, {1: STEEL}, kinematics="small")
    contact, own = cell._images.contact, walls(mesh).n_faces
    assert Surface(contact.master).rims()[:own].any()  # by their own nodes, the walls end
    assert not contact.rims[:own].any()  # they go on in the images of the cell
    # the images of nodes of a face of the cell are the nodes of the opposite face
    own_nodes, images = contact.ident[: mesh.n_nodes], contact.ident[mesh.n_nodes :]
    assert len(np.unique(own_nodes)) == mesh.n_nodes and np.isin(images, own_nodes).any()


def test_ligament_between_two_slots_is_not_a_surface_gone_through():
    """Two slots parted by a ligament thinner than the faces of its walls: its two sides,
    back to back, are not two surfaces that went through each other. Nothing presses
    until the slots close; then their walls do, and the block stiffens."""
    data = np.ones((8, 1, 15), np.uint8)
    data[2:6, :, 5] = data[2:6, :, 7] = 0  # two slots of 2 x 0.1, a ligament of 0.1 between
    image = VoxelImage(data, spacing=(0.5, 0.5, 0.1), periodic=False)
    mesh = voxel_tet_mesh(image).subset([1])
    x = mesh.nodes
    surface = Surface.of(mesh)
    pair = Pair(surface)
    problem = ContactProblem(mesh, {1: STEEL}, [pair], kinematics="small")
    part = problem._passes[0]
    assert part.par[kernels.P_DEPTH] > 0.1  # the ligament is thinner than the depth looked for
    prescribed = [
        (np.arange(len(x)), 1, 0.0),
        (np.flatnonzero(x[:, 2] < 1e-9), 2, 0.0),
        (np.flatnonzero((x[:, 2] < 1e-9) & (x[:, 0] < 1e-9)), 0, 0.0),
        (np.flatnonzero(x[:, 2] > 1.5 - 1e-9), 2, -0.3),
    ]
    res = problem.run(prescribed, steps=15)
    force, moved = -res.reactions[:, 3, 2], 0.3 * res.loads
    open_ = moved < 0.09
    assert open_.sum() >= 4 and np.all(res.area[open_, 0] == 0)
    slope = np.polyfit(moved[open_], force[open_], 1)[0]
    np.testing.assert_allclose(force[open_], slope * moved[open_], rtol=1e-6, atol=1e-9)
    assert res.area[-1, 0] > 0.5 and res.penetration.max() < 0.02
    after = (force[-1] - force[-3]) / (moved[-1] - moved[-3])
    assert after > 1.5 * slope
    work = res.work
    assert work["external"][-1] == pytest.approx(work["bulk"][-1] + work["contact"][-1], rel=1e-6)


def _stack():
    """A periodic stack of layers of 0.9 parted by gaps of 0.1 (the cell of the layers
    that close on their images)."""
    data = np.ones((2, 2, 20), np.uint8)
    data[:, :, 0] = data[:, :, 19] = 0
    return voxel_tet_mesh(VoxelImage(data, spacing=(1.0, 1.0, 0.05), periodic=True)).subset([1])


def test_homogenize_follows_the_paths_with_the_contact_of_the_walls(tmp_path):
    """``nonlinear={"contact": ...}``: the paths of a mesh are those of a ContactCell, and
    their curves have the columns of the contact; the same from the command line."""
    mesh = _stack()
    path = {"times": [0, 1], "increments": 6,
            "strain": {"33": -0.15, "11": 0.0, "22": 0.0, **_SHEARS}}  # fmt: skip
    out = homogenize(mesh, {1: STEEL}, properties=[], paths={"press": path},
                     nonlinear={"contact": {"friction": 0.2}})  # fmt: skip
    res = out["press.periodic"]
    direct = ContactCell(mesh, {1: STEEL}, friction=0.2).run(path)
    np.testing.assert_allclose(res.stress, direct.stress, rtol=1e-9, atol=1e-9)
    assert res.kinematics == "finite" and res.series["contact_area"][-1] == pytest.approx(4.0)
    rows = res.rows()
    assert rows[-1]["contact_area"] == pytest.approx(4.0) and rows[0]["contact_pressure"] == 0
    assert set(SERIES) <= set(res.components()) and "contact_area" in res.row()
    assert res.value("contact_area", 1.0) == pytest.approx(4.0)
    small = homogenize(mesh, {1: STEEL}, properties=[], paths={"press": path},
                       nonlinear={"contact": True, "kinematics": "small"})  # fmt: skip
    assert small["press.periodic"].kinematics == "small"
    # the contact points at the nodes of the faces: flat layers, the same response
    nodes = homogenize(mesh, {1: STEEL}, properties=[], paths={"press": path},
                       nonlinear={"contact": {"friction": 0.2, "points": "nodes"}})  # fmt: skip
    np.testing.assert_allclose(nodes["press.periodic"].stress, res.stress, rtol=1e-6, atol=1e-6)
    assert nodes["press.periodic"].series["contact_area"][-1] == pytest.approx(4.0)
    for bad, match in [
        ({"contact": {"points": "edges"}}, "contact points among"),
        ({"contact": {"mu": 0.1}}, "unknown options"),
        ({"contact": 0.3}, "True or a mapping"),
        ({"contact": {"friction": -1.0}}, "friction"),
    ]:
        with pytest.raises(ValueError, match=match):
            homogenize(mesh, {1: STEEL}, properties=[], paths={"press": path}, nonlinear=bad)
    with pytest.raises(ValueError, match="factorized"):
        homogenize(mesh, {1: STEEL}, properties=[], paths={"press": path}, solver="cg",
                   nonlinear={"contact": True})  # fmt: skip
    image = VoxelImage(np.ones((4, 4, 4), np.uint8), periodic=True)
    with pytest.raises(ValueError, match="walls of the pores of a mesh"):
        homogenize(image, {1: STEEL}, properties=[], paths={"press": path},
                   nonlinear={"contact": True})  # fmt: skip
    # the command line, on the file of the mesh
    mesh_file = tmp_path / "stack.vtu"
    io.save_mesh(mesh, mesh_file)
    (tmp_path / "mats.yaml").write_text("1: {young: 1000.0, poisson: 0.3}\n", encoding="utf-8")
    report = tmp_path / "report.json"
    code = cli.main([
        "homogenize", str(mesh_file), "--materials", str(tmp_path / "mats.yaml"),
        "--property", "bulk_modulus",
        "--path", "press: t=0,1 n=6 E33=-0.15 E11=0 E22=0 E23=0 E13=0 E12=0",
        "--contact", "0.2", "--curves", str(tmp_path / "curves"), "--periodic", "-o", str(report),
    ])  # fmt: skip
    assert code == 0
    final = json.loads(report.read_text(encoding="utf-8"))["paths"]["press.periodic"]
    assert final["contact_area"] == pytest.approx(4.0)
    assert final["S33"] == pytest.approx(res.stress[-1, 2], rel=1e-6)
    curve = (tmp_path / "curves" / "stack_press_periodic.csv").read_text(encoding="utf-8")
    assert "contact_area" in curve.splitlines()[0] and "contact_dissipation" in curve


def test_configuration_of_the_contact_of_the_walls():
    base = {
        "model": {"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.1},
        "domain": {"size": 6.0, "periodic": True},
        "mesh": {"size": 1.0},
        "homogenization": {
            "materials": {0: {"young": 1000.0, "poisson": 0.3}, 1: "void"},
            "properties": [],
            "paths": {"press": {"times": [0, 1], "strain": {"33": -0.1}}},
            "nonlinear": {"contact": {"friction": 0.3, "augmentations": 2}},
        },
    }
    cfg = load_config(base)
    options = cfg.homogenization.options()["nonlinear"]
    assert options == {"contact": {"friction": 0.3, "augmentations": 2}, "kinematics": "finite"}
    assert cfg.homogenization.method == "fe" and cfg.mesh.kwargs({1})["phases"] == "matrix"

    def changed(**homog):
        data = json.loads(json.dumps(base))
        data["homogenization"]["materials"] = {0: {"young": 1000.0, "poisson": 0.3}, 1: "void"}
        data["homogenization"].update(homog)
        return data

    true = load_config(changed(nonlinear={"contact": True, "kinematics": "small"}))
    assert true.homogenization.options()["nonlinear"] == {"contact": {}, "kinematics": "small"}
    off = load_config(changed(nonlinear={"contact": False}))
    assert off.homogenization.options()["nonlinear"] == {}
    nodes = load_config(changed(nonlinear={"contact": {"points": "nodes"}}))
    assert nodes.homogenization.options()["nonlinear"]["contact"] == {"points": "nodes"}
    # the columns of the contact are quantities of the statistics of volume elements
    data = changed()
    data["domain"].pop("size")
    data["rve"] = {"sizes": [6.0], "realizations": 2, "properties": ["press.contact_area"]}
    assert load_config(data).rve.properties == ["press.contact_area"]
    data["homogenization"]["nonlinear"] = {}
    with pytest.raises(ConfigError, match="unknown quantity 'contact_area'"):
        load_config(data)
    bad = [
        (changed(nonlinear={"contact": {"mu": 0.3}}), "unknown keys"),
        (changed(nonlinear={"contact": {"friction": -0.1}}), "0 or more"),
        (changed(nonlinear={"contact": {"stiffness": 0}}), "positive"),
        (changed(nonlinear={"contact": {"augmentations": 1.5}}), "augmentations"),
        (changed(nonlinear={"contact": {"points": "edges"}}), "'gauss' .* or 'nodes'"),
        (changed(nonlinear={"contact": "yes"}), "true, or a mapping"),
        (changed(paths={}, properties=["bulk_modulus"]), "along the\\s+loading paths"),
        (changed(solver="cg"), "factorized"),
    ]
    solid = changed()
    solid["homogenization"]["materials"] = {0: {"young": 1.0, "poisson": 0.3},
                                            1: {"young": 2.0, "poisson": 0.3}}  # fmt: skip
    bad.append((solid, "a void material"))
    meshed = changed()
    meshed["mesh"]["phases"] = "all"
    bad.append((meshed, "mesh of the solid alone"))
    voxels = changed(method="fft")
    voxels.pop("mesh")
    voxels["voxels"] = {"shape": 8}
    bad.append((voxels, "method: fe"))
    for data, match in bad:
        with pytest.raises(ConfigError, match=match):
            load_config(data)


def test_contact_cell_checks_its_inputs():
    solid = voxel_tet_mesh(VoxelImage(np.ones((2, 2, 2), np.uint8), periodic=True))
    with pytest.raises(ValueError, match="no free surface"):
        ContactCell(solid, {1: STEEL})
    data = np.ones((4, 4, 4), np.uint8)
    data[1:3, 1:3, 1:3] = 0
    mesh = voxel_tet_mesh(VoxelImage(data, periodic=True)).subset([1])
    assert walls(mesh).areas(mesh.nodes).sum() == pytest.approx(24.0)  # of the cubic pore
    with pytest.raises(ValueError, match="tolerance"):
        ContactCell(mesh, {1: STEEL}, tolerance=-1.0)
    with pytest.raises(ValueError, match="friction"):
        ContactCell(mesh, {1: STEEL}, friction=-0.5)
    with pytest.raises(ValueError, match="surface of the faces"):
        ContactCell(mesh.quadratic(), {1: STEEL}, surface=walls(mesh))
    with pytest.raises(ValueError, match="factorized"):
        ContactCell(mesh, {1: STEEL}, solver="amg")
    cell = ContactCell(mesh, {1: STEEL}, bc="kubc", kinematics="small")
    assert len(cell._images.node) == mesh.n_nodes  # no image without periodicity
    res = cell.run({"times": [0, 1], "increments": 1, "strain": {"33": -0.01, **_SHEARS,
                                                                "11": 0.0, "22": 0.0}})  # fmt: skip
    assert res.stress[-1, 2] < 0 and res.series["contact_area"][-1] == 0


def test_contact_problem_checks_its_inputs():
    mesh, lower, upper, base, top = _two_blocks()
    materials = {0: STEEL, 1: STEEL}
    with pytest.raises(ValueError, match="at least one pair"):
        ContactProblem(mesh, materials)
    with pytest.raises(TypeError, match="Pair"):
        ContactProblem(mesh, materials, [upper])
    with pytest.raises(TypeError, match="Obstacle"):
        ContactProblem(mesh, materials, obstacles=[upper])
    with pytest.raises(ValueError, match="six for"):
        ContactProblem(mesh.quadratic(), materials, [Pair(upper, lower)])
    with pytest.raises(ValueError, match="without faces"):
        ContactProblem(mesh, materials, [Pair(upper.subset(np.zeros(0, dtype=int)), lower)])
    with pytest.raises(ValueError, match="augmentations"):
        ContactProblem(mesh, materials, [Pair(upper, lower)], augmentations=-1)
    with pytest.raises(ValueError, match="tolerance"):
        ContactProblem(mesh, materials, [Pair(upper, lower)], tolerance=0.0)
    with pytest.raises(ValueError, match="together"):
        ContactProblem(mesh, materials, [Pair(upper, lower)], law=object())
    for options, match in (({"friction": -1.0}, "friction"), ({"stiffness": 0.0}, "stiffness"),
                           ({"depth": -1.0}, "depth")):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            Pair(upper, lower, **options)
    problem = ContactProblem(mesh, materials, [Pair(upper, lower)], kinematics="small")
    with pytest.raises(ValueError, match="initial"):
        problem.run(_press(mesh, base, top, 0.01), initial=np.zeros((mesh.n_nodes, 3)))


def test_closed_forms_of_contact():
    modulus = effective_modulus(1000.0, 0.3)
    assert modulus == pytest.approx(1000.0 / 0.91)
    assert effective_modulus(1000.0, 0.3, 1000.0, 0.3) == pytest.approx(modulus / 2)
    sphere = hertz_sphere(10.0, modulus, approach=0.1)
    assert sphere["radius"] == pytest.approx(1.0)
    assert sphere["pressure"] == pytest.approx(2 * modulus * sphere["radius"] / (math.pi * 10))
    assert hertz_sphere(10.0, modulus, force=sphere["force"])["approach"] == pytest.approx(0.1)
    line = hertz_cylinder(20.0, modulus, 30.0)
    assert line["pressure"] == pytest.approx(modulus * line["half_width"] / (2 * 20.0))
    cone = sneddon_cone(70.3, modulus, 0.5)
    assert cone["force"] == pytest.approx(
        math.pi / 2 * modulus * cone["radius"] ** 2 / math.tan(math.radians(70.3))
    )
    assert flat_punch(0.5, modulus, 0.01)["force"] == pytest.approx(0.01 * modulus)
    assert (
        mindlin_stick_radius(1.0, 0.3, 10.0, 0.0) == 1
        and mindlin_stick_radius(1.0, 0.3, 10.0, 3.0) == 0
    )
    with pytest.raises(ValueError):
        hertz_sphere(10.0, modulus)
    with pytest.raises(ValueError):
        mindlin_stick_radius(1.0, 0.3, 10.0, 4.0)


@pytest.mark.slow
def test_hertz_contact_of_a_sphere():
    """A rigid sphere pressed on a block (a quarter of it): Hertz's force, radius and
    pressure while the contact is small against the block."""
    size, width, radius = 0.5, 4.0, 10.0
    mesh = _block((8, 8, 8), (size,) * 3, order=2)
    x = mesh.nodes
    top = Surface.of(mesh, lambda c: np.abs(c[:, 2] - width) < 1e-9)
    sphere = Obstacle.sphere(top, (0, 0, width + radius), radius, motion=(0, 0, -0.03))
    problem = ContactProblem(mesh, {0: STEEL}, obstacles=[sphere], kinematics="small")
    prescribed = [(np.flatnonzero(x[:, c] < 1e-9), c, 0.0) for c in range(3)]
    res = problem.run(prescribed, steps=3)
    exact = hertz_sphere(radius, effective_modulus(1000.0, 0.3), approach=0.03)
    assert 4 * res.forces[-1, 0, 2] == pytest.approx(exact["force"], rel=0.1)
    assert math.sqrt(4 * res.area[-1, 0] / math.pi) == pytest.approx(exact["radius"], rel=0.12)
    assert res.pressure[-1, 0] == pytest.approx(exact["pressure"], rel=0.1)
    forces = 4 * res.forces[1:, 0, 2]  # F in δ^(3/2), a little stiffer as the contact grows
    slope = np.polyfit(np.log(res.loads[1:]), np.log(forces), 1)[0]
    assert slope == pytest.approx(1.5, abs=0.15)


@pytest.mark.slow
def test_clip_whose_arms_come_to_rest_on_each_other():
    """Two arms that a slot parts, in finite strains: the upper one, pushed down by its
    tip, comes to rest on the lower one, and the two bend together, twice as stiff."""
    data = np.ones((16, 1, 10), np.uint8)
    data[4:, :, 4:6] = 0  # the slot between the arms
    data[15:, :, 6:] = 0  # the upper arm a little shorter: its corner lands on the lower
    mesh = voxel_tet_mesh(VoxelImage(data, spacing=(0.25,) * 3, periodic=False)).subset([1])
    x = mesh.nodes
    problem = ContactProblem(mesh, {1: STEEL}, [Pair(Surface.of(mesh))])
    tip = np.flatnonzero((np.abs(x[:, 0] - 3.75) < 1e-9) & (np.abs(x[:, 2] - 2.5) < 1e-9))
    prescribed = [
        (np.arange(len(x)), 1, 0.0),
        (np.flatnonzero(x[:, 0] < 1e-9), [0, 2], 0.0),
        (tip, 2, -1.0),
    ]
    res = problem.run(prescribed, steps=20)
    force = -res.reactions[:, 2, 2]
    apart = res.area[:, 0] == 0
    assert apart[:9].all() and not apart[-8:].any()  # they touch near a deflection of 0.5
    before = np.polyfit(res.loads[1:8], force[1:8], 1)[0]
    # on average: the corner of the upper arm carries no point, and the force jumps as it
    # passes from a point of the lower arm to the next
    first = np.flatnonzero(~apart)[0]
    after = (force[-1] - force[first]) / (res.loads[-1] - res.loads[first])
    assert after / before == pytest.approx(2.0, rel=0.15)
    assert res.penetration.max() < 0.02
