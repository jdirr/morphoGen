"""Cracks that feel the toughness of the phases of a morphology and of the interfaces between
them: the first-order kinked factors (Cotterell & Rice), the toughness of phases and of pairs
of phases, the competition of deflection and penetration at the interface of two alike phases
(He & Hutchinson: deflected below Γ_interface / Γ_bulk = 1/4, through it above), on the
phases of a morphology and on those of a crack mesh conforming to them; a tough inclusion
raising the critical loads of the cracks reaching it, at a fixed load and by load steps;
weak interfaces of layers deflecting the crack along them, a longer path than through strong
ones; the staircase path of a brick-and-mortar along its weak interfaces and the ductile
growth through a tough inclusion (slow)."""

import math

import numpy as np
import pytest

from morphogen.domain import Domain
from morphogen.fracture import lefm
from morphogen.fracture.ductile import propagate_ductile
from morphogen.fracture.mixed import KinkedCrack, kink_angle, propagate_kinked, solve_crack
from morphogen.fracture.toughness import (
    Advance,
    Toughness,
    advance_candidates,
    choose_advance,
    interface_paths,
    kinked_factors,
    kinked_release_rate,
)
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage
from morphogen.mesh.conforming import has_mmg

mmg = pytest.mark.skipif(not has_mmg(), reason="MMG")

W = H = 16.0
T = 1.0
BOX = ((0.0, 0.0, 0.0), (W, H, T))
CRACK = KinkedCrack(((0.0, 8.0, 0.0), (4.5, 8.0, 0.0)))
MATRIX = Material(young=1000.0, poisson=0.3)
ALIKE = {0: MATRIX, 1: MATRIX}


def _tension(m):
    f = lefm.face_forces(m, lambda c: c[:, 1] > H - 1e-9, [0.0, 1.0, 0.0])
    return f + lefm.face_forces(m, lambda c: c[:, 1] < 1e-9, [0.0, -1.0, 0.0])


def _rigid(m):
    def nearest(p):
        return int(np.argmin(np.linalg.norm(m.nodes - np.asarray(p), axis=1)))

    a, b = nearest((W, H / 2, 0.0)), nearest((W, H, 0.0))
    return [(a, [0, 1], 0.0), (b, 0, 0.0), (np.flatnonzero(m.nodes[:, 2] < 1e-9), 2, 0.0)]


class _Halves:
    """The two halves of the plate, ``x < x0`` (phase 0) and beyond (phase 1): a plane
    interface normal to the crack, by point queries and by a level set (conforming meshes)."""

    def __init__(self, x0: float):
        self.domain = Domain(size=(W, H, T), periodic=False)
        self.x0 = float(x0)
        self.phase_names = {0: "near", 1: "far"}

    def contains(self, points):
        return np.asarray(points, dtype=float)[:, 0] > self.x0

    def level_set(self, shape=None, *, voxel_size=None, grid=None):
        values = (self.x0 - grid.center_points()[:, 0]).reshape(grid.shape)
        return VoxelImage(values, spacing=tuple(grid.spacing), origin=tuple(grid.origin))


def _layers(points):
    """Layers of 2 along x, alternately of the phases 0 and 1 (interfaces at x = 6, 8...)."""
    return np.floor(np.asarray(points)[:, 0] / 2.0).astype(np.int64) % 2


def _inclusion(points):
    """A disc of radius 1 centred 2.3 ahead of the tip (from x = 5.8 to 7.8), phase 1."""
    p = np.asarray(points)
    return (np.hypot(p[:, 0] - 6.8, p[:, 1] - 8.0) < 1.0).astype(np.int64)


def _bricks(points, thickness: float = 2.0, width: float = 3.0):
    """A brick-and-mortar of interfaces without thickness: layers of ``thickness`` along x,
    of bricks of ``width`` along y shifted by half a brick from one layer to the next, in
    three phases so that neighbouring bricks differ (every boundary an interface)."""
    p = np.asarray(points)
    odd = np.floor(p[:, 0] / thickness).astype(np.int64) % 2
    j = np.floor((p[:, 1] - 0.5 * width * odd) / width).astype(np.int64)
    return (j + 2 * odd) % 3


_MESHES: dict = {}


def _meshes(morphology, size: float):
    """The crack meshes of the plate with the phases of a morphology (staircases), each crack
    meshed once (the growths of the tests share their first cracks)."""

    def mesh_of(crack):
        key = (morphology, size, crack)
        if key not in _MESHES:
            _MESHES[key] = lefm.crack_mesh(*BOX, crack, size, morphology=morphology)
        return _MESHES[key]

    return mesh_of


def test_kinked_factors_of_cotterell_and_rice():
    """The factors of a kink: those of the front straight ahead; in the direction of the
    largest hoop stress (Erdogan & Sih), k_II vanishes and k_I is the largest, the hoop
    stress of the asymptotic field; in pure mode I a kink at ±90° releases a quarter of G
    (He & Hutchinson's 1/4), in pure mode II the kink of -70.5° releases 4/3 of G; a closed
    kink releases nothing of mode I; the tearing mode adds K_III² / 2μ."""
    np.testing.assert_allclose(kinked_factors(2.0, -0.7, 0.0), (2.0, -0.7))
    theta = np.linspace(-np.pi + 1e-3, np.pi - 1e-3, 200001)
    for k1, k2 in ((1.0, 0.3), (1.0, -1.0), (0.2, 1.0)):
        angle = kink_angle(k1, k2)
        a, b = kinked_factors(k1, k2, angle)
        assert abs(b) < 1e-12 and a == pytest.approx(kinked_factors(k1, k2, theta)[0].max())
        hoop = math.cos(angle / 2) * (k1 * math.cos(angle / 2) ** 2 - 1.5 * k2 * math.sin(angle))
        assert a == pytest.approx(hoop)
    e_prime, mu = 1000.0 / (1 - 0.3**2), 1000.0 / 2.6
    straight = kinked_release_rate(1.0, 0.0, 0.0, 0.0, e_prime, mu)
    assert straight == pytest.approx(1.0 / e_prime)
    for angle in (-math.pi / 2, math.pi / 2):
        assert kinked_release_rate(1.0, 0.0, 0.0, angle, e_prime, mu) == pytest.approx(
            straight / 4.0, rel=1e-12
        )
    mode_two = kinked_release_rate(0.0, 1.0, 0.0, kink_angle(0.0, 1.0), e_prime, mu)
    assert mode_two == pytest.approx(4.0 / 3.0 / e_prime, rel=1e-9)
    mirror = -kink_angle(0.0, 1.0)  # the mirror of the kink: closed
    closed = kinked_factors(0.0, 1.0, mirror)
    assert closed[0] < 0
    assert kinked_release_rate(0.0, 1.0, 0.0, mirror, e_prime, mu) == pytest.approx(
        closed[1] ** 2 / e_prime
    )
    assert kinked_release_rate(1.0, 0.0, 0.5, 0.0, e_prime, mu) == pytest.approx(
        straight + 0.25 / (2 * mu)
    )


def test_toughness_of_phases_and_interfaces():
    """One value or a value per phase (a default for the others), interfaces of one value or
    per pair in either order (none: no criterion); one value for all the phases without
    interfaces is uniform, the growth of a homogeneous body; positive values."""
    assert Toughness.of(None) is None and Toughness.of(2).value == 2.0
    rule = Toughness({0: 1.0, 1: 4.0}, {(1, 0): 0.2}, default=2.0)
    assert not rule.uniform and rule.of_phase(1) == 4.0 and rule.of_phase(7) == 2.0
    assert rule.of_interface(0, 1) == rule.of_interface(1, 0) == 0.2
    assert rule.of_interface(0, 2) is None and rule.of_interface(1, 1) is None
    assert Toughness({0: 1.5, 1: 1.5}).uniform and Toughness({0: 1.5, 1: 1.5}).value == 1.5
    assert not Toughness(1.0, 0.3).uniform and Toughness(1.0, 0.3).of_interface(4, 9) == 0.3
    assert Toughness.of(rule) is rule and Toughness.of({0: 1, 1: 2}).of_phase(1) == 2.0
    with pytest.raises(KeyError):
        Toughness({0: 1.0}).of_phase(1)
    with pytest.raises(ValueError, match="positive"):
        Toughness({0: 1.0, 1: 0.0})
    with pytest.raises(ValueError, match="two phases"):
        Toughness(1.0, {(1, 1): 0.5})
    with pytest.raises(ValueError, match="not one value"):
        _ = rule.value
    advance = Advance("interface", 0.5, CRACK, 0.02, 0.1, (0, 1))
    assert advance.ratio == pytest.approx(0.2) and advance.label == "0/1"


@pytest.mark.slow
@mmg
def test_deflection_or_penetration_of_he_and_hutchinson():
    """A crack normal to the interface of two alike phases, 0.75 before it (within the
    increment of 1): the directions along the interface are at ±90°, their G a quarter of
    that of the bulk (0.25 ∓ K_II / K_I, Cotterell & Rice); with Γ_interface / Γ_bulk =
    0.15 the crack goes on to the interface and turns along it, at a lower critical load,
    with 0.4 it goes through, straight. The same on the phases of a crack mesh conforming to
    the interface (no morphology given)."""
    halves = _Halves(5.25)
    growths = {}
    for ratio in (0.15, 0.4):
        rule = Toughness(1.0, ratio, morphology=halves)
        growths[ratio] = propagate_kinked(_meshes(halves, 0.6), CRACK, ALIKE, _rigid,
                                          forces=_tension, increment=1.0, steps=1,
                                          toughness=rule, keep=True)  # fmt: skip
    first = growths[0.15].solutions[0]
    modes = first.modes()
    candidates = advance_candidates(first, Toughness(1.0, 0.15, morphology=halves), 1.0)
    bulk, along = candidates[0], candidates[1:]
    assert bulk.kind == "bulk" and bulk.release_rate == pytest.approx(modes["G"])
    assert sorted(round(math.degrees(c.angle)) for c in along) == [-90, 90]
    mixity = abs(modes["K_II"] / modes["K_I"])
    for c in along:
        assert c.kind == "interface" and c.met == (0, 1) and c.toughness == 0.15
        assert c.release_rate / bulk.release_rate == pytest.approx(0.25, abs=1.2 * mixity + 1e-3)
        assert c.crack.points[-2] == pytest.approx((5.25, 8.0, 0.0), abs=1e-5)
    deflected, through = growths[0.15], growths[0.4]
    assert deflected.advances[0].kind == "interface" and through.advances[0].kind == "bulk"
    tip = deflected.cracks[1].tip
    assert tip[0] == pytest.approx(5.25, abs=1e-4)
    assert abs(tip[1] - 8.0) == pytest.approx(1.0, abs=1e-4)
    assert through.cracks[1].tip[0] == pytest.approx(5.5, abs=1e-3)
    assert abs(through.cracks[1].tip[1] - 8.0) < 0.05
    assert deflected.loads[0] < through.loads[0] == pytest.approx(1.0 / math.sqrt(modes["G"]))
    assert deflected.rows()[0]["path"] == "interface" and through.rows()[0]["met"] == "0"
    # the interfaces of a crack mesh conforming to the phases
    mesh = lefm.crack_mesh(*BOX, CRACK, 0.6, morphology=lefm.phase_mesh(halves, *BOX, 0.6))
    solution = solve_crack(mesh, CRACK, ALIKE, _rigid, forces=_tension)
    paths = interface_paths(solution, Toughness(1.0, 0.15), 1.0)
    assert sorted(round(math.degrees(a)) for a, _, _ in paths) == [-90, 90]
    for ratio, kind in ((0.15, "interface"), (0.4, "bulk")):
        choice = choose_advance(advance_candidates(solution, Toughness(1.0, ratio), 1.0))
        assert choice.kind == kind


@pytest.mark.slow
@mmg
def test_a_tough_inclusion_holds_the_crack_back():
    """A disc four times tougher than the matrix ahead of the crack (the same elasticity):
    at a fixed load the critical load sqrt(Γ / G) of the advance into it doubles and rises
    along the growth (it falls without the tough disc: an unstable growth), the advances
    meeting the matrix, then the disc; by load steps, the advance into the disc waits for a
    load nearly twice as large, after which the crack runs through it at once."""
    rule = Toughness({0: 1.0, 1: 4.0}, morphology=_inclusion)
    fixed = propagate_kinked(_meshes(_inclusion, 0.75), CRACK, ALIKE, _rigid, forces=_tension,
                             increment=0.75, steps=2, toughness=rule)  # fmt: skip
    plain = 1.0 / np.sqrt(fixed.rates)  # one toughness: the same cracks, straight
    assert [a.label for a in fixed.advances] == ["0", "1", "1"]
    np.testing.assert_allclose(fixed.loads, plain * [1.0, 2.0, 2.0], rtol=1e-12)
    assert fixed.loads[1] > fixed.loads[0] and plain[1] < plain[0]
    assert all(abs(y) < 0.05 for y in (c.tip[1] - 8.0 for c in fixed.cracks))
    growths = {}
    for tough in (0.05, 0.2):
        rule = Toughness({0: 0.05, 1: tough}, morphology=_inclusion)
        growths[tough] = propagate_ductile(_meshes(_inclusion, 0.75), CRACK, ALIKE, _rigid,
                                           forces=_tension, load_step=0.25, increment=0.75,
                                           steps=2, toughness=rule)  # fmt: skip
    plain, tough = growths[0.05], growths[0.2]
    np.testing.assert_allclose(plain.loads, [1.25, 1.25, 1.25])  # 1.25² G(1) > 0.05 > G(1)
    assert not plain.advances  # one toughness: the growth of a homogeneous plate
    assert tough.loads[0] == 1.25 and tough.loads[1] > 1.5 and tough.loads[2] == tough.loads[1]
    g = tough.tip[1] / tough.loads[1] ** 2  # G of the second crack at the unit load
    assert tough.loads[1] == pytest.approx(0.25 * math.ceil(math.sqrt(0.2 / g) / 0.25))
    rows = tough.rows()
    assert [r["met"] for r in rows] == ["0", "1", "1"] and all(r["ratio"] >= 1.0 for r in rows)
    waits = [h["ratio"] for h in tough.history if h["advance"] == 1]
    assert len(waits) >= 3 and np.all(np.diff(waits) > 0) and waits[-2] < 1.0 <= waits[-1]


@pytest.mark.slow
@mmg
def test_weak_interfaces_deflect_a_layered_crack():
    """Layers of 2 normal to the crack, alike but for their weak interfaces (Γ_interface /
    Γ_bulk = 0.1): the crack reaching an interface turns along it and follows it (the bulk
    would kink back towards the load in mode II, less favourable), a path longer than its
    extent along the first direction and deflected; through strong interfaces (1.0) the path
    stays straight."""
    paths = {}
    for ratio in (0.1, 1.0):
        rule = Toughness(1.0, ratio, morphology=_layers)
        paths[ratio] = propagate_kinked(_meshes(_layers, 0.75), CRACK, ALIKE, _rigid,
                                        forces=_tension, increment=0.75, steps=3,
                                        toughness=rule)  # fmt: skip
    weak, strong = paths[0.1], paths[1.0]
    kinds = [a.kind for a in weak.advances]
    turn = kinds.index("interface")  # within an increment of the first interface
    assert kinds[0] == "bulk" and set(kinds[turn:]) == {"interface"} and turn < len(kinds) - 1
    assert weak.advances[turn].met == (0, 1) and abs(math.degrees(weak.advances[turn].angle)) > 80
    assert all(abs(a.angle) < 0.05 for a in weak.advances[turn + 1 :])  # along it
    assert all(a.kind == "bulk" for a in strong.advances)
    for growth, deflected in ((weak, True), (strong, False)):
        tip = growth.cracks[-1].tip
        length = growth.cracks[-1].size - CRACK.size
        extent = tip[0] - CRACK.tip[0]
        assert (abs(tip[1] - 8.0) > 0.7) == deflected and (abs(tip[1] - 8.0) < 0.05) != deflected
        assert (length > 1.4 * extent) == deflected
    assert weak.cracks[-1].tip[0] == pytest.approx(6.0, abs=0.01)


@pytest.mark.slow
@mmg
def test_a_brick_and_mortar_deflects_the_crack_into_a_staircase():
    """A brick-and-mortar of alike bricks (layers of 2 along x, bricks of 3 along y shifted
    by half a brick), its interfaces weak (Γ_interface / Γ_bulk = 0.1): the crack turns along
    the first interface between layers, stops at the junction of the next head joint and
    turns into it, then along the next interface between layers: a staircase, deflected and
    longer than its extent along x (nacre's toughening path); its interfaces strong (1.0),
    the crack runs straight through the bricks."""
    paths = {}
    for ratio in (0.1, 1.0):
        rule = Toughness(1.0, ratio, morphology=_bricks)
        paths[ratio] = propagate_kinked(_meshes(_bricks, 0.6), CRACK,
                                        {0: MATRIX, 1: MATRIX, 2: MATRIX}, _rigid,
                                        forces=_tension, increment=0.6, steps=10,
                                        toughness=rule)  # fmt: skip
    weak, strong = paths[0.1], paths[1.0]
    kinds = [a.kind for a in weak.advances[:-1]]
    assert kinds[:2] == ["bulk", "bulk"] and set(kinds[2:]) == {"interface"}
    turns = [round(math.degrees(a.angle)) for a in weak.advances[2:-1] if abs(a.angle) > 0.35]
    assert len(turns) >= 3 and all(abs(abs(t) - 90) <= 2 for t in turns)
    points = np.asarray(weak.cracks[-1].points)[:, :2]
    xs = points[:, 0]
    assert np.all(np.diff(xs) > -1e-4) and xs[-1] > 7.9  # through two interfaces between layers
    assert abs(points[-1, 1] - 8.0) > 2.0  # deflected
    assert weak.cracks[-1].size - CRACK.size > 1.5 * (xs[-1] - CRACK.tip[0])
    assert all(a.kind == "bulk" for a in strong.advances)
    assert np.ptp(np.asarray(strong.cracks[-1].points)[:, 1]) < 0.1


@pytest.mark.slow
@mmg
def test_ductile_growth_held_back_by_a_tough_inclusion():
    """A viscoplastic plate pulled by its grips, a disc of the same metal but four times
    tougher ahead of its crack: J_tip against the toughness of the phase at the front, the
    advance into the disc at a higher load than through the plain plate."""
    viscous = Material(young=1000.0, poisson=0.3, plasticity={
        "yield": 1.0, "hardening": 50.0, "norton": {"K": 1.0, "n": 5.0}})  # fmt: skip

    def grips(m, u=0.03):
        top, bottom = np.flatnonzero(m.nodes[:, 1] > H - 1e-9), np.flatnonzero(m.nodes[:, 1] < 1e-9)
        return [(top, [0, 1, 2], [0.0, u, 0.0]), (bottom, [0, 1, 2], [0.0, -u, 0.0])]

    growths = {}
    for tough in (0.03, 0.12):  # 0.03: one toughness, the growth of a homogeneous plate
        rule = Toughness({0: 0.03, 1: tough}, morphology=_inclusion)
        growths[tough] = propagate_ductile(_meshes(_inclusion, 0.75), CRACK,
                                           {0: viscous, 1: viscous}, grips, load_step=0.5,
                                           increment=0.75, steps=1, toughness=rule,
                                           transfer="nearest", kink=False)  # fmt: skip
    plain, tough = growths[0.03], growths[0.12]
    assert not plain.advances and [a.label for a in tough.advances] == ["0", "1"]
    assert np.all(plain.tip >= 0.03) and tough.tip[1] >= 0.12 > tough.tip[0] >= 0.03
    assert tough.loads[0] == plain.loads[0] and tough.loads[1] > plain.loads[1]
