"""Cracks through a morphology: crack meshes that carry its phases (the materials) and the
sides of the crack apart, as staircases or conforming to the interfaces of the phases (the
mesh of its phases cut by each crack by MMG), its pores left out; equal materials give the
homogeneous plate; near the front, a stiff inclusion deflects the crack, a soft one or a
pore attracts it (the interaction integrals on the material at the front), and the crack
runs through a pore; the state of an inelastic morphology moves material by material between
the meshes of two cracks, and a ductile crack grows through the phases."""

import math

import numpy as np
import pytest

from morphogen.domain import Domain
from morphogen.fracture import lefm
from morphogen.fracture.ductile import _problem, propagate_ductile, transfer_state
from morphogen.fracture.mixed import KinkedCrack, propagate_kinked, solve_crack, through_voids
from morphogen.homogenization.materials import Material
from morphogen.mesh.conforming import has_mmg
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments

pytestmark = pytest.mark.skipif(not has_mmg(), reason="MMG")

W = H = 16.0
T = 1.0
BOX = ((0.0, 0.0, 0.0), (W, H, T))
CRACK = KinkedCrack(((0.0, 8.0, 0.0), (4.5, 8.0, 0.0)))
MATRIX = Material(young=1000.0, poisson=0.3)
STIFF = Material(young=10000.0, poisson=0.3)
SOFT = Material(young=100.0, poisson=0.3)
VOID = Material(void=True)


def _cylinder(x: float, y: float, radius: float) -> PrimitiveUnion:
    """A plate of 16 x 16 x 1 whose phase 1 is a cylinder through its thickness."""
    segments = Segments(np.array([[x, y, -1.0]]), np.array([[x, y, T + 1.0]]),
                        np.array([radius]))  # fmt: skip
    return PrimitiveUnion(Domain(size=(W, H, T), periodic=False), segments=segments)


def _tension(m):
    f = lefm.face_forces(m, lambda c: c[:, 1] > H - 1e-9, [0.0, 1.0, 0.0])
    return f + lefm.face_forces(m, lambda c: c[:, 1] < 1e-9, [0.0, -1.0, 0.0])


def _rigid(m):
    def nearest(p):
        return int(np.argmin(np.linalg.norm(m.nodes - np.asarray(p), axis=1)))

    a, b = nearest((W, H / 2, 0.0)), nearest((W, H, 0.0))
    return [(a, [0, 1], 0.0), (b, 0, 0.0), (np.flatnonzero(m.nodes[:, 2] < 1e-9), 2, 0.0)]


def _grips(m, u=0.03):
    top, bottom = np.flatnonzero(m.nodes[:, 1] > H - 1e-9), np.flatnonzero(m.nodes[:, 1] < 1e-9)
    return [(top, [0, 1, 2], [0.0, u, 0.0]), (bottom, [0, 1, 2], [0.0, -u, 0.0])]


def _volume(mesh, phase=None) -> float:
    v = np.abs(mesh.volumes())
    return float(v.sum() if phase is None else v[mesh.phase == phase].sum())


@pytest.mark.slow
def test_crack_meshes_of_a_morphology():
    """The staircase crack mesh is the crack mesh of the box, its phases those of the
    morphology at the centroids and its sides the former phases; the conforming one follows
    the cylinder (its phases those of the centroids, its volume within 3 %: polygons inscribed
    in the level set of the grid of h / 2, remeshed within h / 20) and the plane of
    the crack (within a tenth of an element, the area of the opened crack that of the plane),
    the faces of the box kept; the pore is left out, its walls free faces on the cylinder;
    the matrix void, two cylinders fall apart."""
    micro = _cylinder(5.5, 10.5, 1.5)
    size = 0.5
    plain = lefm.crack_mesh(*BOX, CRACK, size)
    stairs = lefm.crack_mesh(*BOX, CRACK, size, morphology=micro)
    np.testing.assert_array_equal(stairs.elements, plain.elements)
    np.testing.assert_array_equal(stairs.side, plain.phase)
    centroids = stairs.nodes[stairs.elements].mean(axis=1)
    np.testing.assert_array_equal(stairs.phase, micro.contains(centroids))
    assert stairs.metadata["crack_sides"] == "side" and plain.metadata["crack_sides"] == "phase"
    phases = lefm.phase_mesh(micro, *BOX, size)
    mesh = lefm.crack_mesh(*BOX, CRACK, size, morphology=phases)
    assert mesh.side is not None and set(np.unique(mesh.phase)) == {0, 1}
    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    assert np.mean(micro.contains(centroids) != (mesh.phase == 1)) < 0.005
    assert np.mean((CRACK.phi(centroids) < 0) != (mesh.side == 1)) < 0.005
    assert _volume(mesh) == pytest.approx(W * H * T, rel=1e-9)
    assert _volume(mesh, 1) == pytest.approx(math.pi * 1.5**2 * T, rel=0.03)
    tets = mesh.elements
    on = np.zeros(mesh.n_nodes, dtype=bool)
    on[np.intersect1d(tets[mesh.side == 0], tets[mesh.side == 1])] = True
    assert np.abs(CRACK.phi(mesh.nodes[on])).max() < 0.1 * size
    opened, area = lefm.open_crack(mesh, CRACK)
    assert area == pytest.approx(4.5 * T, rel=1e-6) and opened.side is not None
    pores = lefm.crack_mesh(*BOX, CRACK, size, morphology=phases, voids=[1])
    assert set(np.unique(pores.phase)) == {0}
    assert _volume(pores) == pytest.approx(W * H * T - _volume(mesh, 1), rel=1e-3)
    walls = lefm.interface_nodes(pores)
    radius = np.hypot(pores.nodes[walls, 0] - 5.5, pores.nodes[walls, 1] - 10.5)
    assert walls.any() and np.abs(radius - 1.5).max() < 0.05
    discs = PrimitiveUnion(Domain(size=(W, H, T), periodic=False), segments=Segments(
        np.array([[4.0, 4.0, -1.0], [12.0, 12.0, -1.0]]),
        np.array([[4.0, 4.0, 2.0], [12.0, 12.0, 2.0]]), np.array([1.5, 1.5])))  # fmt: skip
    assert lefm.without_voids(lefm.with_phases(plain, discs), [1]).n_elements < plain.n_elements
    with pytest.raises(ValueError, match="falls apart"):  # the matrix void: two discs apart
        lefm.without_voids(lefm.with_phases(plain, discs), [0])


@pytest.mark.slow
def test_equal_materials_give_the_homogeneous_plate():
    """The same material in both phases: on the staircase mesh, the factors and G of the
    plate exactly; on the conforming mesh within 1 % (another mesh); and the G at the tip,
    restricted to the material of the front, that of the domain."""
    micro = _cylinder(5.5, 10.5, 1.5)
    size = 0.5
    plain = solve_crack(lefm.crack_mesh(*BOX, CRACK, size), CRACK, MATRIX, _rigid,
                        forces=_tension).modes()  # fmt: skip
    stairs = lefm.crack_mesh(*BOX, CRACK, size, morphology=micro)
    same = solve_crack(stairs, CRACK, {0: MATRIX, 1: Material(young=1000.0, poisson=0.3)},
                       _rigid, forces=_tension)  # fmt: skip
    for key in ("K_I", "K_II", "G"):
        assert same.modes()[key] == pytest.approx(plain[key], rel=1e-9, abs=1e-12)
    phases = lefm.phase_mesh(micro, *BOX, size)
    mesh = lefm.crack_mesh(*BOX, CRACK, size, morphology=phases)
    other = solve_crack(mesh, CRACK, {0: MATRIX, 1: MATRIX}, _rigid, forces=_tension)
    modes = other.modes()
    assert modes["K_I"] == pytest.approx(plain["K_I"], rel=0.01)
    assert modes["G"] == pytest.approx(plain["G"], rel=0.02)
    assert abs(modes["K_II"]) < 0.01 * modes["K_I"]
    assert other.release_rate(energy="tip") == pytest.approx(other.release_rate(), rel=1e-12)


@pytest.mark.slow
def test_inclusions_deflect_and_pores_attract():
    """A cylinder of radius 1.5 beside the path, its centre 1 ahead of the tip and 2.5 above
    the crack (the domain of the integrals reaches it): a stiff one (10 E) turns the crack
    away (K_II > 0, a kink below -3°), a soft one (E / 10) and a pore turn it towards
    themselves (above +5°: Kirsch's hole gives +16° by the weight of the crack faces), the
    same material leaves it straight; at the tip, G rises by the pore (anti-shielding) and
    falls by the stiff inclusion (shielding) against the G of the domain."""
    micro = _cylinder(5.5, 10.5, 1.5)
    phases = lefm.phase_mesh(micro, *BOX, 0.5)
    mesh = lefm.crack_mesh(*BOX, CRACK, 0.5, morphology=phases)
    pores = lefm.crack_mesh(*BOX, CRACK, 0.5, morphology=phases, voids=[1])
    angles, tip, far = {}, {}, {}
    for name, m, inclusion in (("stiff", mesh, STIFF), ("equal", mesh, MATRIX),
                               ("soft", mesh, SOFT), ("pore", pores, VOID)):  # fmt: skip
        solution = solve_crack(m, CRACK, {0: MATRIX, 1: inclusion}, _rigid, forces=_tension)
        angles[name] = math.degrees(solution.modes()["angle"])
        tip[name], far[name] = solution.release_rate(energy="tip"), solution.release_rate()
    assert angles["stiff"] < -3.0 and abs(angles["equal"]) < 1.5
    assert angles["soft"] > 5.0 and angles["pore"] > 5.0
    assert tip["pore"] > far["pore"] and tip["stiff"] < far["stiff"]


@pytest.mark.slow
def test_the_crack_runs_through_a_pore():
    """A pore of radius 1 on the path, 1 ahead of the tip: a front that would end in it goes
    through it, the crack emanating from its far wall (its front beyond the wall by a quarter
    to three quarters of the increment); the growth at a fixed load crosses it and goes on
    (finite factors beyond)."""
    micro = _cylinder(6.5, 8.0, 1.0)
    phases = lefm.phase_mesh(micro, *BOX, 0.5)

    def mesh_of(crack):
        return lefm.crack_mesh(*BOX, crack, 0.5, morphology=phases, voids=[1])

    mesh = mesh_of(CRACK)
    crossed = through_voids(CRACK.grown(1.5), mesh, 1.5)
    assert crossed.tip[0] > 7.5 and crossed.size > CRACK.size + 3.0
    assert through_voids(CRACK.grown(0.5), mesh, 1.5) == CRACK.grown(0.5)  # in the matrix
    growth = propagate_kinked(mesh_of, CRACK, {0: MATRIX, 1: VOID}, _rigid, forces=_tension,
                              increment=1.5, steps=1)  # fmt: skip
    assert growth.cracks[1].tip[0] > 7.5
    assert np.all(np.isfinite(growth.factors)) and growth.rates[1] > 0


def test_state_moves_material_by_material():
    """An inelastic morphology (a hardening matrix, a stiffer harder cylinder): its state
    carried onto the same mesh by the nearest element is the identity; onto the mesh of
    another crack a state set to the phase of each point comes back as the phase of each new
    point, by both methods (the nodes of the interfaces have a value per material); with the
    same material in both phases, the driving forces of the plate exactly."""
    micro = _cylinder(5.5, 10.5, 1.5)
    hard = Material(young=3000.0, poisson=0.3, plasticity={"yield": 3.0, "hardening": 150.0})
    soft = Material(young=1000.0, poisson=0.3, plasticity={"yield": 1.0, "hardening": 50.0})
    materials = {0: soft, 1: hard}
    phases = lefm.phase_mesh(micro, *BOX, 1.0)
    mesh = lefm.crack_mesh(*BOX, CRACK, 1.0, morphology=phases)
    plastic = solve_crack(mesh, CRACK, materials, _grips, steps=3)
    assert plastic.bulk.old[:, 6].max() > 1e-3  # the matrix yields
    opened, _, _, problem = _problem(mesh, CRACK, materials, "small", None, "auto")
    transfer_state(plastic, opened, problem.bulk, CRACK, "nearest")
    np.testing.assert_array_equal(problem.bulk.old, plastic.bulk.old)
    grown = CRACK.grown(1.0, math.radians(15.0))
    other = lefm.crack_mesh(*BOX, grown, 1.0, morphology=phases)
    target_mesh, _, _, target = _problem(other, grown, materials, "small", None, "auto")
    plastic.bulk.old[...] = plastic.mesh.phase[:, None]  # the state: the phase of the point
    for method in ("nearest", "projection"):
        transfer_state(plastic, target_mesh, target.bulk, grown, method)
        assert np.abs(target.bulk.old - target_mesh.phase[:, None]).max() < 1e-12
    stairs = lefm.crack_mesh(*BOX, CRACK, 1.0, morphology=micro)
    plate = solve_crack(lefm.crack_mesh(*BOX, CRACK, 1.0), CRACK, soft, _grips, steps=3)
    alike = solve_crack(stairs, CRACK, {0: soft, 1: soft}, _grips, steps=3)
    for energy in ("work", "free", "tip"):
        assert alike.release_rate(energy=energy) == pytest.approx(
            plate.release_rate(energy=energy), rel=1e-9
        )


@pytest.mark.slow
def test_ductile_growth_through_a_morphology():
    """A viscoplastic matrix with a stiff elastic cylinder beside the path, its crack growing
    once J_tip reaches 0.03, remeshed at each advance (the state moved material by material):
    the advances at rising loads, deflected away from the stiff inclusion once beside it (a
    kink below -3°: -9.7°; the same material in both phases leaves it within 0.4° of
    straight)."""
    micro = _cylinder(6.5, 10.5, 1.5)
    phases = lefm.phase_mesh(micro, *BOX, 0.75)
    viscous = Material(young=1000.0, poisson=0.3, plasticity={
        "yield": 1.0, "hardening": 50.0, "norton": {"K": 1.0, "n": 5.0}})  # fmt: skip

    def mesh_of(crack):
        return lefm.crack_mesh(*BOX, crack, 0.75, morphology=phases)

    growth = propagate_ductile(mesh_of, CRACK, {0: viscous, 1: STIFF}, _grips,
                               load_step=0.5, increment=1.0, steps=2, toughness=0.03,
                               transfer="nearest")  # fmt: skip
    assert len(growth.cracks) == 3 and np.all(growth.tip >= 0.03)
    assert np.all(np.diff(growth.loads) >= 0) and np.all(np.isfinite(growth.angles))
    assert math.degrees(growth.angles[1]) < -3.0  # away from the stiff inclusion above
    assert growth.cracks[-1].tip[1] < 8.0 - 0.05
