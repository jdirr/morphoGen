"""Cracks of any surface and front: the level sets and axes of a surface crack against
those of the penny, straight and kinked cracks; its growth by rows; the mesh refined
around its front; an inclined penny crack against Kassir & Sih (the three modes along
the front, the angles of kinking of Erdogan & Sih); an elliptical crack against Irwin;
the growth of each point of the front: an ellipse towards a circle, an inclined crack
towards local symmetry; a straight front as an open surface crack."""

import math

import numpy as np
import pytest
from scipy.special import ellipe

from morphogen.fracture import lefm
from morphogen.fracture.lefm import Crack
from morphogen.fracture.mixed import KinkedCrack, energy_release_rate, kink_angle, solve_crack
from morphogen.fracture.surface import FrontGrowth, SurfaceCrack, at_stations, propagate_front
from morphogen.homogenization.materials import Material
from morphogen.mesh.conforming import has_mmg

SOLID = Material(young=1.0, poisson=0.3)
NU = 0.3
HALF = 2.5  # the body: a cube of side 5 around a crack of size 1


def _nearest(mesh, point):
    return int(np.argmin(np.linalg.norm(mesh.nodes - np.asarray(point), axis=1)))


def _tension(m):
    """A unit tension along z on the faces z = ±HALF."""
    f = lefm.face_forces(m, lambda c: c[:, 2] > HALF - 1e-9, [0.0, 0.0, 1.0])
    return f + lefm.face_forces(m, lambda c: c[:, 2] < 1e-9 - HALF, [0.0, 0.0, -1.0])


def _supports(m):
    a, b, c = (
        _nearest(m, p) for p in ((-HALF, -HALF, -HALF), (HALF, -HALF, -HALF), (-HALF, HALF, -HALF))
    )
    return [(a, [0, 1, 2], 0.0), (b, [1, 2], 0.0), (c, 2, 0.0)]


def _mesh(size, front, radius):
    return lambda crack: lefm.crack_mesh(
        (-HALF,) * 3, (HALF,) * 3, crack, size=size, front_size=front, front_radius=radius
    )


def test_surface_of_a_penny_crack():
    """A planar surface crack with a circular front: the level sets, the axes and the
    front of the penny crack (the front a polygon of 64 stations), its area and reach."""
    penny = Crack("penny", (0.2, -0.1, 0.3), (0, 0, 1), 1.0)
    crack = SurfaceCrack.ellipse(penny.point, penny.normal, 1.0, stations=64)
    assert crack.closed and crack.curved and crack.points.shape == (4, 64, 3)
    x = np.random.default_rng(0).uniform(-2, 2, (4000, 3))
    sagitta = 1.0 - math.cos(math.pi / 64)
    assert crack.phi(x) == pytest.approx(penny.phi(x), abs=1e-12)
    assert crack.psi(x) == pytest.approx(penny.psi(x), abs=1.5 * sagitta)
    r, n, d, t, s = crack.frame(x)
    r0, n0, d0, t0, _ = penny.frame(x)
    assert r == pytest.approx(r0, abs=2 * sagitta) and n == pytest.approx(n0, abs=1e-12)
    away = np.hypot(x[:, 0] - 0.2, x[:, 1] + 0.1) > 0.3  # (the axis: any direction)
    assert np.abs(d - d0)[away].max() < 0.05 and np.abs(t - t0)[away].max() < 0.05
    assert np.all((s >= -math.pi) & (s < math.pi))
    x1, x2 = crack.local(x)
    assert x1 == pytest.approx(crack.psi(x)) and r == pytest.approx(np.hypot(x1, x2))
    assert crack.size == pytest.approx(1.0) and crack.reach == pytest.approx(1.0)
    assert crack.area() == pytest.approx(math.pi, rel=2e-3)
    assert crack.length == pytest.approx(2 * math.pi, rel=1e-3)
    points, lengths = crack.front((-2,) * 3, (2,) * 3, 100)
    assert len(points) == 100 and lengths.sum() == pytest.approx(crack.length)
    assert np.hypot(points[:, 0] - 0.2, points[:, 1] + 0.1) == pytest.approx(1.0, abs=sagitta)
    assert len(crack.front((0.2, -0.1, 0.0), (2, 2, 1), 100)[0]) == 25  # a quarter in the box
    # an ellipse: its plane, its area, its reach (the least distance of its tangents)
    ellipse = SurfaceCrack.ellipse((0, 0, 0), (0, 0, 2), 2.0, 1.0, major=(1, 0, 0), stations=96)
    assert ellipse.phi(x) == pytest.approx(x[:, 2], abs=1e-12)
    assert ellipse.area() == pytest.approx(2 * math.pi, rel=2e-3)
    assert ellipse.reach == pytest.approx(1.0, rel=1e-3)
    assert np.abs(ellipse.points[-1][:, 0]).max() == pytest.approx(2.0)
    assert SurfaceCrack.of(penny, stations=32).points.shape == (4, 32, 3)
    for bad in (np.zeros((1, 8, 3)), np.zeros((2, 8, 2)), np.zeros((3, 8, 3))):
        with pytest.raises(ValueError, match="SurfaceCrack"):
            SurfaceCrack(bad)
    with pytest.raises(ValueError, match="ellipse"):
        SurfaceCrack.ellipse((0, 0, 0), (0, 0, 1), -1.0)


def test_open_front_of_straight_and_kinked_cracks():
    """A straight front as an open surface crack: the level sets and the axes of the
    straight crack exactly, and of the kinked crack; grown by an angle, the kinked one."""
    lower, upper = (0.0, 0.0, 0.0), (4.0, 6.0, 1.0)
    straight = Crack("straight", (0.0, 3.0, 0.0), (0.0, 1.0, 0.0), 1.5, (1.0, 0.0, 0.0))
    crack = SurfaceCrack.of(straight, lower, upper, stations=9)
    assert not crack.closed and crack.points.shape == (2, 9, 3)
    x = np.random.default_rng(1).uniform(lower, upper, (3000, 3))
    assert crack.phi(x) == pytest.approx(straight.phi(x), abs=1e-12)
    assert crack.psi(x) == pytest.approx(straight.psi(x), abs=1e-12)
    for mine, theirs in zip(crack.frame(x)[:4], straight.frame(x)[:4], strict=True):
        assert mine == pytest.approx(theirs, abs=1e-12)
    assert crack.reach == pytest.approx(1.5) and crack.size == pytest.approx(1.5)
    points, lengths = crack.front(lower, upper)
    assert lengths.sum() == pytest.approx(1.0, rel=0.01)  # the thickness of the plate
    assert points[:, :2] == pytest.approx(np.tile([1.5, 3.0], (len(points), 1)))
    kinked = KinkedCrack.of(straight).grown(0.5, 0.4)
    for surface in (SurfaceCrack.of(kinked, lower, upper, stations=9), crack.grown(0.5, 0.4)):
        assert surface.psi(x) == pytest.approx(kinked.psi(x), abs=1e-9)
        differ = np.abs(surface.phi(x) - kinked.phi(x)) > 1e-9  # (the side nearest the kink)
        assert np.count_nonzero(differ) < 0.01 * len(x)
        assert surface.reach == pytest.approx(kinked.last, rel=1e-9)
    with pytest.raises(ValueError, match="the box"):
        SurfaceCrack.of(straight)


def test_growth_by_rows():
    """Each station advances by its own amount at its own angle: a new row; a uniform
    advance keeps a penny a penny, an advance at an angle makes a cone; the front moves
    back within its last row."""
    crack = SurfaceCrack.ellipse((0, 0, 0), (0, 0, 1), 1.0, stations=48)
    grown = crack.grown(0.25)
    assert grown.points.shape == (5, 48, 3) and grown.reach == pytest.approx(1.25)
    assert np.linalg.norm(grown.points[-1], axis=1) == pytest.approx(1.25)
    assert grown.area() == pytest.approx(math.pi * 1.25**2, rel=3e-3)
    cone = crack.grown(0.2, math.radians(30.0))
    front = cone.points[-1]
    assert front[:, 2] == pytest.approx(0.1) and cone.reach == pytest.approx(0.2)
    assert np.hypot(front[:, 0], front[:, 1]) == pytest.approx(1.0 + 0.2 * math.cos(math.pi / 6))
    # the level set behind the front: the cone; ahead of it: its continuation
    slope, edge = math.tan(math.pi / 6), 1.0 + 0.2 * math.cos(math.pi / 6)
    on_cone = np.array([[1.05, 0.0, 0.05 * slope], [1.5, 0.0, 0.1 + (1.5 - edge) * slope]])
    assert cone.phi(on_cone) == pytest.approx(0.0, abs=2e-3)
    assert cone.phi(np.array([[0.5, 0.2, 0.3]])) == pytest.approx(0.3)
    # advances and angles of each station
    angle = np.radians(20.0) * np.cos(crack.s)
    uneven = crack.grown(0.1 + 0.05 * np.cos(crack.s) ** 2, angle)
    step = uneven.points[-1] - crack.points[-1]
    assert np.linalg.norm(step, axis=1) == pytest.approx(0.1 + 0.05 * np.cos(crack.s) ** 2)
    assert step[:, 2] == pytest.approx(np.linalg.norm(step, axis=1) * np.sin(angle), abs=1e-9)
    back = crack.grown(-0.1)
    assert back.points.shape == crack.points.shape and back.size == pytest.approx(0.9)
    for da, turn in ((-1.5, 0.0), (-0.1, 0.2), (np.r_[0.1, -0.1 * np.ones(47)], 0.0)):
        with pytest.raises(ValueError, match="grown"):
            crack.grown(da, turn)
    # values at stations of the integrals, interpolated around the front
    at = np.linspace(-math.pi, math.pi, 12, endpoint=False)
    assert at_stations(crack, at, np.cos(at)) == pytest.approx(np.cos(crack.s), abs=0.04)
    values = 1.0 + 0.5 * np.cos(2 * at) + 0.1 * (-1.0) ** np.arange(12)  # with noise
    smooth = at_stations(crack, at, values, harmonics=4)
    assert smooth == pytest.approx(1.0 + 0.5 * np.cos(2 * crack.s), abs=1e-12)


@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_mesh_refined_around_the_front():
    """The mesh of a crack remeshed with small elements around its front alone: the two
    sides of the crack and the faces of the box kept, the lips opened behind the front."""
    crack = SurfaceCrack.ellipse((0.0, 0.0, 0.0), (0, 0, 1), 1.0, 0.6, major=(1, 0, 0))
    coarse = lefm.crack_mesh((-HALF,) * 3, (HALF,) * 3, crack, size=0.7)
    fine = lefm.crack_mesh((-HALF,) * 3, (HALF,) * 3, crack, size=0.7, front_size=0.12)
    assert fine.metadata["front_size"] == 0.12 and fine.metadata["front_radius"] == 0.48
    assert 1.5 * coarse.n_nodes < fine.n_nodes < 6 * coarse.n_nodes

    def edges(mesh, near):
        x = mesh.nodes[mesh.elements]
        length = np.linalg.norm(x[:, 1] - x[:, 0], axis=1)
        return length[(crack.frame(x.mean(axis=1))[0] < 0.25) == near].mean()

    assert edges(fine, True) < 0.2 and edges(fine, False) > 0.3
    assert edges(coarse, True) > 0.4
    assert set(np.unique(fine.phase)) == {0, 1} and fine.quality().min() > 0.02
    assert fine.volumes().sum() == pytest.approx(125.0, rel=1e-9)
    assert fine.nodes.min(axis=0) == pytest.approx((-HALF,) * 3)
    opened, area = lefm.open_crack(fine, crack)
    assert area == pytest.approx(crack.area(), rel=0.03)
    assert opened.n_nodes > fine.n_nodes


@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_inclined_penny_crack_against_kassir_and_sih():
    """A penny crack whose normal is at 30° from the tension: K_I uniform along its
    front, K_II and K_III as the cosine and the sine of the angle from the shear on its
    plane, at the amplitudes of Kassir & Sih (1966; within 10 % in a cube of five radii),
    and the angle of kinking of each point that of Erdogan & Sih for its own factors:
    from -46° to 46° around the front."""
    alpha = math.radians(30.0)
    normal = (math.sin(alpha), 0.0, math.cos(alpha))
    crack = SurfaceCrack.ellipse((0.0, 0.0, 0.0), normal, 1.0, major=(0, 1, 0), stations=48)
    solution = solve_crack(_mesh(0.7, 0.12, 0.4)(crack), crack, SOLID, _supports, forces=_tension)
    modes = solution.modes(stations=8, r_in=0.12, r_out=0.36)
    shear = np.array([-math.cos(alpha), 0.0, math.sin(alpha)])  # of the traction on the plane
    cos = at_stations(crack, crack.s, crack._d @ shear)
    cos = np.interp(modes["s"], crack.s, cos, period=2 * math.pi)
    sin = np.interp(modes["s"], crack.s, crack._t @ shear, period=2 * math.pi)
    unit = math.sqrt(1.0 / math.pi)
    k1 = 2 * unit * math.cos(alpha) ** 2
    tau = math.sin(alpha) * math.cos(alpha)
    k2, k3 = 4 * unit * tau / (2 - NU), 4 * (1 - NU) * unit * tau / (2 - NU)
    assert modes["K_I"] == pytest.approx(k1, rel=0.1)
    assert modes["K_II"] @ cos / (cos @ cos) == pytest.approx(k2, rel=0.1)
    assert modes["K_III"] @ sin / (sin @ sin) == pytest.approx(k3, rel=0.12)
    assert np.std(modes["K_II"] - k2 * cos) < 0.1 * k2
    exact = np.array([kink_angle(k1, k2 * c) for c in cos])
    assert np.degrees(exact).max() == pytest.approx(46.4, abs=0.1)
    assert np.degrees(modes["angle"]) == pytest.approx(np.degrees(exact), abs=4.0)
    assert solution.element_size() < 0.2  # that of the front, not of the body


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_elliptical_crack_against_irwin():
    """An elliptical crack of semi-axes 1 and 0.5 in tension: ``K_I = σ sqrt(π b) / E(k)
    (sin²φ + (b/a)² cos²φ)^(1/4)`` along its front (Irwin 1962), largest at the ends of
    its minor axis, in the ratio sqrt(a / b) with the ends of its major axis; no other
    mode."""
    a, b = 1.0, 0.5
    crack = SurfaceCrack.ellipse((0.0, 0.0, 0.0), (0, 0, 1), a, b, major=(1, 0, 0), stations=64)
    solution = solve_crack(_mesh(0.6, 0.06, 0.3)(crack), crack, SOLID, _supports, forces=_tension)
    modes = solution.modes(stations=16, r_in=0.08, r_out=0.24)
    stations = np.append(crack.s, math.pi)
    angle = np.interp(modes["s"], stations, 2 * math.pi * np.arange(65) / 64)
    exact = math.sqrt(math.pi * b) / ellipe(1 - (b / a) ** 2)
    exact = exact * (np.sin(angle) ** 2 + (b / a) ** 2 * np.cos(angle) ** 2) ** 0.25
    ratio = modes["K_I"] / exact
    assert ratio.mean() == pytest.approx(1.0, abs=0.05) and ratio.std() < 0.1
    minor = modes["K_I"][np.abs(np.sin(angle)) > 0.9].mean()
    major = modes["K_I"][np.abs(np.cos(angle)) > 0.9].mean()
    # (the ends of the major axis, where the front turns within four elements: 13 % low)
    assert minor / major == pytest.approx(math.sqrt(a / b), rel=0.2)
    assert np.abs(modes["K_II"]).max() < 0.12 and np.abs(modes["K_III"]).max() < 0.12


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_elliptical_crack_grows_towards_a_circle():
    """In tension the points of the minor axis of an elliptical crack advance fastest: its
    aspect ratio rises towards 1, in its plane."""
    crack = SurfaceCrack.ellipse((0.0, 0.0, 0.0), (0, 0, 1), 1.0, 0.6, major=(1, 0, 0))
    growth = propagate_front(
        _mesh(0.6, 0.08, 0.3), crack, SOLID, _supports, forces=_tension, increment=0.25,
        steps=2, stations=12, paris=(1e-3, 3.0), domain={"r_in": 0.1, "r_out": 0.25},
    )  # fmt: skip
    assert isinstance(growth, FrontGrowth) and len(growth.cracks) == 3
    ratios = []
    for grown in growth.cracks:
        front = grown.points[-1]
        ratios.append(np.abs(front[:, 1]).max() / np.abs(front[:, 0]).max())
        assert np.abs(front[:, 2]).max() < 0.05  # in its plane
    assert ratios[0] == pytest.approx(0.6) and ratios[0] < ratios[1] < ratios[2] < 1.0
    assert ratios[2] > 0.75
    rows = growth.rows()
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert rows[0]["area"] < rows[1]["area"] < rows[2]["area"]
    assert rows[0]["N"] == 0.0 and rows[1]["N"] > 0.0 and rows[2]["N"] > rows[1]["N"]
    assert growth.advances[:2] == pytest.approx(0.25) and growth.advances[2] == 0.0
    assert growth.factors.shape == (3, 12, 3)
    # the spread of K_I along the front narrows as the crack rounds
    spread = [f[:, 0].max() / f[:, 0].min() for f in growth.factors]
    assert spread[2] < spread[0]


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_inclined_crack_turns_towards_local_symmetry():
    """The inclined penny crack grown: each point of its front kinks by its own angle,
    which turns the surface towards the plane normal to the tension; the mode II of the
    new fronts falls (the criterion of local symmetry), the mode I rises, and the mode
    III, which the kinks do not relieve, stays."""
    alpha = math.radians(30.0)
    normal = (math.sin(alpha), 0.0, math.cos(alpha))
    crack = SurfaceCrack.ellipse((0.0, 0.0, 0.0), normal, 1.0, major=(0, 1, 0), stations=48)
    growth = propagate_front(
        _mesh(0.6, 0.08, 0.3), crack, SOLID, _supports, forces=_tension, increment=0.3,
        steps=2, stations=12, domain={"r_in": 0.1, "r_out": 0.25},
    )  # fmt: skip
    k2 = [np.abs(f[:, 1]).max() for f in growth.factors]
    k1 = [f[:, 0].mean() for f in growth.factors]
    k3 = [np.abs(f[:, 2]).max() for f in growth.factors]
    assert k2[1] < 0.5 * k2[0] and k2[2] < 0.4 * k2[0]
    assert k1[0] < k1[1] < k1[2]
    assert k3[2] == pytest.approx(k3[0], rel=0.25)
    assert np.degrees(np.abs(growth.angles[0]).max()) == pytest.approx(46.4, abs=4.0)
    # the first kinks bring the normal of the surface nearer the axis of the tension
    first, second = growth.cracks[0], growth.cracks[1]
    most = np.argmax(np.abs(first._d @ np.array([-math.cos(alpha), 0.0, math.sin(alpha)])))
    # (by 46°, from 30° on one side of it to 16° on the other)
    assert abs(first._n[most, 2]) == pytest.approx(math.cos(alpha))
    assert abs(second._n[most, 2]) == pytest.approx(math.cos(math.radians(16.4)), abs=0.03)
    assert 0.2 < second.reach <= 0.3  # the least advance of the stations


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_straight_front_as_a_surface_crack():
    """An edge crack through a plate in tension, as a planar crack and as a surface crack
    with an open front: the same energy release rate and factors (the same level sets,
    the meshes of MMG alike), and that of the energies of two cracks."""
    lower, upper = (0.0, 0.0, 0.0), (4.0, 8.0, 1.0)
    straight = Crack("straight", (0.0, 4.0, 0.0), (0.0, 1.0, 0.0), 1.2, (1.0, 0.0, 0.0))
    surface = SurfaceCrack.of(straight, lower, upper, stations=9)

    def forces(m):
        f = lefm.face_forces(m, lambda c: c[:, 1] > 8.0 - 1e-9, [0.0, 1.0, 0.0])
        return f + lefm.face_forces(m, lambda c: c[:, 1] < 1e-9, [0.0, -1.0, 0.0])

    def supports(m):
        a, b, c = (_nearest(m, p) for p in ((4.0, 0.0, 0.0), (4.0, 8.0, 0.0), (4.0, 0.0, 1.0)))
        return [(a, [0, 1, 2], 0.0), (b, [0, 2], 0.0), (c, 0, 0.0)]

    def solve(crack):
        mesh = lefm.crack_mesh(lower, upper, crack, size=0.35)
        return solve_crack(mesh, crack, SOLID, supports, forces=forces)

    planar, open_front = solve(straight), solve(surface)
    g = planar.release_rate()
    assert open_front.release_rate() == pytest.approx(g, rel=0.02)
    for key in ("K_I", "G"):
        assert open_front.modes()[key] == pytest.approx(planar.modes()[key], rel=0.02)
    assert abs(open_front.modes()["K_II"]) < 0.05 * planar.modes()["K_I"]
    assert energy_release_rate(solve, surface, 0.15) == pytest.approx(g, rel=0.1)
