"""Coulomb's friction of interfaces whose nodes match (cohesive zones, the lips of cracks):
the law of a point (stick, slip by a radial return, the energy dissipated by a cycle, the
strength left to a broken cohesive point); two blocks pressed, then sheared, which stick
and then slide at T = μ N, in small and finite strains, with the balance of the works; a
cohesive interface that keeps Coulomb's strength once broken; a closed crack under
compression and shear, whose lips slide by the shear less Coulomb's and stick below it;
and the stress intensity factor and energy release rate of an edge crack whose lips rub
(slow)."""

import math

import numpy as np
import pytest

from morphogen.fracture import lefm
from morphogen.fracture.cohesive import (
    Bilinear,
    CohesiveProblem,
    Contact,
    Exponential,
    insert_interfaces,
    interior_faces,
    lip_tractions,
    traction,
)
from morphogen.fracture.lefm import Crack
from morphogen.fracture.mixed import solve_crack
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage

STEEL = Material(young=1000.0, poisson=0.3)


def test_friction_of_a_point():
    """Lips pressed by p = K <-δ_n>: they stick below μ p, slide at it; a cycle of
    amplitude D at a constant pressure dissipates 4 μ p (D - μ p / K_t)."""
    law = Contact(1e4, friction=0.3, tangential=2e3)
    p = 1e4 * 0.01
    t, _, slip, work = traction(law, [-0.01, 0.005, 0.0], slip=[0.0, 0.0])
    np.testing.assert_allclose(t, [-p, 2e3 * 0.005, 0.0])  # stick: elastic
    assert not slip.any() and work == 0
    t, _, slip, work = traction(law, [-0.01, 0.03, 0.04], slip=[0.0, 0.0])
    np.testing.assert_allclose(t, [-p, 0.3 * p * 0.6, 0.3 * p * 0.8])  # slip along the trial
    np.testing.assert_allclose(slip, np.array([0.03, 0.04]) - t[1:] / 2e3)
    assert work == pytest.approx(0.3 * p * (0.05 - 0.3 * p / 2e3))
    # a cycle between -D and D after a first loading to D
    amplitude, state, total = 0.05, np.zeros(2), 0.0
    for s in np.concatenate([np.linspace(0, amplitude, 50), np.linspace(amplitude, -amplitude, 99),
                             np.linspace(-amplitude, amplitude, 99)]):  # fmt: skip
        _, _, state, work = traction(law, [-0.01, s, 0.0], slip=state)
        total += work
    quarter = 0.3 * p * (amplitude - 0.3 * p / 2e3)
    assert total == pytest.approx(5 * quarter, rel=1e-12)
    # lips apart carry nothing and forget their slip; without friction, as before
    t, _, slip, work = traction(law, [0.01, 0.05, 0.0], slip=[0.02, 0.0])
    assert not t.any() and work == 0
    np.testing.assert_allclose(slip, [0.05, 0.0])
    t, history = traction(Contact(1e4), [-0.01, 0.05, 0.0])
    np.testing.assert_allclose(t, [-100.0, 0.0, 0.0])
    # a cohesive point under compression: (1 - d) K δ_s + d min(K δ_s, μ p), μ p once broken
    bilinear = Bilinear(1e4, 5.0, 8.0, 0.5, 1.0, friction=0.5)
    pressure, state, history = 1e4 * 0.002, np.zeros(2), 0.0
    for s in (0.0005, 0.001, 0.05, 0.2, 0.3):
        t, history, state, _ = traction(bilinear, [-0.002, s, 0.0], history, state)
        expected = (1 - history) * 1e4 * s + history * min(1e4 * s, 0.5 * pressure)
        assert t[1] == pytest.approx(expected) and t[0] == pytest.approx(-pressure)
    assert history == 1.0 and t[1] == pytest.approx(0.5 * pressure)
    exponential = Exponential(5.0, 0.5, friction=0.5)
    t, history, state, _ = traction(exponential, [-0.002, 1.0, 0.0], 0.0, np.zeros(2))
    broken = 1 - math.exp(-history / exponential.critical_opening)
    cohesive, _ = traction(Exponential(5.0, 0.5), [-0.002, 1.0, 0.0])
    assert t[1] - cohesive[1] == pytest.approx(broken * 0.5 * -t[0])
    for law_of in (
        lambda: Contact(1.0, friction=-0.1),
        lambda: Bilinear(1e4, 5, 8, 0.5, 1, friction=-1),
        lambda: Exponential(5.0, 0.5, friction=math.nan),
    ):
        with pytest.raises(ValueError, match="friction"):
            law_of()
    with pytest.raises(ValueError, match="tangential"):
        Contact(1.0, friction=0.1, tangential=0.0)


def _blocks(order=1):
    """Two blocks, one on the other; the entries of a loading in two stages: the upper
    face pressed down by ``press`` (load factors 0 to 1), then moved by ``shear``."""
    data = np.zeros((2, 2, 4), np.uint8)
    data[:, :, 2:] = 1
    mesh = voxel_tet_mesh(VoxelImage(data, periodic=False))
    if order == 2:
        mesh = mesh.quadratic()
    opened, inter = insert_interfaces(mesh, "phases")
    x = opened.nodes
    bottom, top = np.flatnonzero(x[:, 2] < 1e-9), np.flatnonzero(x[:, 2] > 4 - 1e-9)

    def prescribed(press, shear):
        return [
            (bottom, [0, 1, 2], 0.0),
            (top, 2, lambda t: -press * min(t, 1.0)),
            (top, 0, lambda t: shear * max(t - 1.0, 0.0)),
            (top, 1, 0.0),
        ]

    return opened, inter, prescribed


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_blocks_stick_then_slide(kinematics, tmp_path):
    """Pressed, then sheared: the blocks stick while T < μ N, then slide at T = μ N; the
    work of friction is T times the sliding, and the works add up."""
    mesh, inter, prescribed = _blocks()
    law = Contact(1e5, friction=0.4)
    problem = CohesiveProblem(mesh, inter, {0: STEEL, 1: STEEL}, law, kinematics=kinematics)
    res = problem.run(prescribed(0.02, 0.08), load=2.0, steps=24)
    normal, tangential = -res.reactions[:, 1, 2], res.reactions[:, 2, 0]
    pressed = np.searchsorted(res.loads, 1.0 - 1e-9)
    assert abs(tangential[pressed]) < 1e-9 * normal[pressed] and res.work["friction"][pressed] == 0
    ratio = tangential / np.maximum(normal, 1e-300)
    assert np.all(ratio < 0.4 * (1 + 1e-5)) and np.all(np.diff(ratio[pressed:]) > -1e-5)
    if kinematics == "small":  # the frame of the interface does not turn
        assert ratio[-1] == pytest.approx(0.4, rel=1e-4)
    else:  # it turns with the blocks: the reactions are not along it
        assert ratio[-1] == pytest.approx(0.4, rel=0.03)
    if kinematics == "small":  # steady sliding: the work of T is all dissipated
        sliding = np.flatnonzero(ratio > 0.4 * (1 - 1e-3))
        first, last = sliding[0], sliding[-1]
        moved = 0.08 * (res.loads[last] - res.loads[first])
        rubbed = res.work["friction"][last] - res.work["friction"][first]
        assert rubbed == pytest.approx(tangential[last] * moved, rel=1e-3)
    total = res.work["bulk"][-1] + res.work["interfaces"][-1]
    assert res.work["external"][-1] == pytest.approx(total, rel=1e-6)
    # the work of the interfaces: friction, and the little energy of the penalty
    assert res.work["friction"][-1] == pytest.approx(res.work["interfaces"][-1], rel=0.02)
    assert res.slip.shape == (inter.n_elements, 3, 2) and np.abs(res.slip).max() > 0.02
    assert res.friction.shape == (inter.n_elements, 3) and res.friction.min() > 0
    assert "work_friction" in res.rows()[-1]
    _, wedges = problem.write(tmp_path / "blocks", res)
    assert b"slip" in wedges.read_bytes()
    # without friction the blocks slide freely
    free = CohesiveProblem(mesh, inter, {0: STEEL, 1: STEEL}, Contact(1e5), kinematics=kinematics)
    res = free.run(prescribed(0.02, 0.08), load=2.0, steps=8)
    assert abs(res.reactions[-1, 2, 0]) < 1e-6 * -res.reactions[-1, 1, 2]
    assert "friction" not in res.work and not res.slip.any()


def test_blocks_of_quadratic_elements_slide_at_coulomb():
    mesh, inter, prescribed = _blocks(order=2)
    problem = CohesiveProblem(mesh, inter, {0: STEEL, 1: STEEL}, Contact(1e5, friction=0.25))
    res = problem.run(prescribed(0.02, 0.06), load=2.0, steps=12)
    assert res.reactions[-1, 2, 0] == pytest.approx(-0.25 * res.reactions[-1, 1, 2], rel=1e-4)
    assert res.slip.shape == (inter.n_elements, 6, 2)


def test_broken_interface_keeps_coulomb_strength():
    """A cohesive interface sheared under a pressure: its strength, then Coulomb's μ N
    once it is broken; friction dissipates the rest of the work of the interfaces."""
    mesh, inter, prescribed = _blocks()
    law = Bilinear(1e5, 5.0, 8.0, 0.3, 0.6, friction=0.5)
    problem = CohesiveProblem(mesh, inter, {0: STEEL, 1: STEEL}, law)
    res = problem.run(prescribed(0.02, 0.3), load=2.0, steps=60)
    normal, tangential = -res.reactions[:, 1, 2], res.reactions[:, 2, 0]
    area = inter.areas(mesh).sum()
    assert res.cracked[-1] == pytest.approx(area) and res.damage.min() == pytest.approx(1.0)
    assert tangential.max() > 8.0 * area  # the shear strength, and friction on the broken share
    assert tangential[-1] == pytest.approx(0.5 * normal[-1], rel=1e-4)
    cohesive = res.work["interfaces"][-1] - res.work["friction"][-1]
    assert cohesive == pytest.approx(0.6 * area, rel=0.05)  # G_IIc, and the penalty's energy
    # the part already broken at the start: Coulomb's strength at once
    res = problem.run(prescribed(0.02, 0.1), load=2.0, steps=12, history=np.ones(inter.n_elements))
    assert res.reactions[-1, 2, 0] == pytest.approx(-0.5 * res.reactions[-1, 1, 2], rel=1e-4)
    assert res.reactions[:, 2, 0].max() < 0.5 * normal.max() * (1 + 1e-5)
    with pytest.raises(ValueError, match="history"):
        problem.run(prescribed(0.02, 0.02), history=np.ones(3))


def _closed_crack(friction, size=0.25, width=12.0):
    """A plate in plane strain with a crack of half-length 1 at its centre, under a
    compression 10 across the crack and a shear 8: the jump of the displacement along the
    lips at their nodes, with their abscissae."""
    young, nu, half = 1000.0, 0.3, 0.5 * width
    n = round(width / size)
    image = VoxelImage(np.zeros((n, 1, n), np.uint8), spacing=(size,) * 3, periodic=False)
    opened, inter = insert_interfaces(
        voxel_tet_mesh(image),
        lambda c: (np.abs(c[:, 2] - half) < 1e-6) & (np.abs(c[:, 0] - half) < 1.0),
    )
    x = opened.nodes
    forces = lefm.face_forces(opened, lambda c: c[:, 2] > width - 1e-9, [8.0, 0.0, -10.0])
    forces += lefm.face_forces(opened, lambda c: c[:, 2] < 1e-9, [-8.0, 0.0, 10.0])
    forces += lefm.face_forces(opened, lambda c: c[:, 0] > width - 1e-9, [0.0, 0.0, 8.0])
    forces += lefm.face_forces(opened, lambda c: c[:, 0] < 1e-9, [0.0, 0.0, -8.0])
    corner = np.flatnonzero((x[:, 0] < 1e-9) & (x[:, 2] < 1e-9))
    other = np.flatnonzero((x[:, 0] > width - 1e-9) & (x[:, 2] < 1e-9))
    prescribed = [(np.arange(len(x)), 1, 0.0), (corner, [0, 2], 0.0), (other, 2, 0.0)]
    law = Contact(50 * young / size, friction=friction)
    problem = CohesiveProblem(opened, inter, {0: Material(young=young, poisson=nu)}, law)
    res = problem.run(prescribed, steps=2, forces=lambda t: t * forces)
    jump = res.displacement[inter.plus] - res.displacement[inter.minus]
    return x[inter.minus][..., 0] - half, jump, res, (problem, opened, inter, law)


def test_closed_crack_slides_by_the_shear_less_friction():
    """The lips of a closed crack under a shear τ and a pressure σ slide as those of a
    crack under τ - μ σ (Coulomb's traction is uniform along them): the ratio of the
    slips with and without friction; lips that μ σ > τ holds do not slide."""
    where, free, _, _ = _closed_crack(0.0)
    _, rubbing, res, (_, opened, inter, law) = _closed_crack(0.3)
    centre = np.abs(where) < 1e-9
    slip0, slip = np.abs(free[centre][:, 0]).mean(), np.abs(rubbing[centre][:, 0]).mean()
    assert slip / slip0 == pytest.approx((8.0 - 0.3 * 10.0) / 8.0, rel=5e-3)
    exact = 4 * (1 - 0.3**2) / 1000.0 * 8.0  # of the crack in the infinite plane
    assert 0.8 * exact < slip0 < exact  # four linear elements along its half-length
    assert rubbing[..., 2].max() <= 1e-12 and rubbing[..., 2].min() > -1e-3  # closed
    assert res.work["friction"][-1] > 0
    # the tractions of the lips at the end: the pressure, and Coulomb's shear
    tractions, jumps = lip_tractions(opened.nodes, inter, res.displacement, law, res)
    inside = np.abs(where).max(axis=1) < 0.7  # away from the fronts
    np.testing.assert_allclose(tractions[inside][..., 2].mean(), -10.0, rtol=0.02)
    np.testing.assert_allclose(np.abs(tractions[inside][..., 0]).mean(), 3.0, rtol=0.02)
    assert jumps.shape == tractions.shape == (inter.n_elements, 3, 3)
    _, stuck, res, _ = _closed_crack(0.9)
    stick = 8.0 / (50 * 1000.0 / 0.25)  # the elastic stick of the penalty: τ / K_t
    assert np.abs(stuck[..., 0]).max() < 2 * stick and res.work["friction"][-1] == 0
    corners, pairs = interior_faces(opened)
    assert corners.shape[1] == 3 and pairs.shape == (len(corners), 2)


W, H, A = 16.0, 32.0, 4.5
PLATE = Material(young=1000.0, poisson=0.0)
CRACK = Crack("straight", (0.0, 16.0, 0.0), (0, 1, 0), A, direction=(1, 0, 0))


def _faces(m, top, right=None):
    f = lefm.face_forces(m, lambda c: c[:, 1] > H - 1e-9, top)
    f += lefm.face_forces(m, lambda c: c[:, 1] < 1e-9, -np.asarray(top, dtype=float))
    if right is not None:
        f += lefm.face_forces(m, lambda c: c[:, 0] > W - 1e-9, right)
        f += lefm.face_forces(m, lambda c: c[:, 0] < 1e-9, -np.asarray(right, dtype=float))
    return f


def _supports(m):
    nearest = lambda point: int(np.argmin(np.linalg.norm(m.nodes - point, axis=1)))  # noqa: E731
    a, b = nearest(np.array([W, H / 2, 0.0])), nearest(np.array([W, H, 0.0]))
    return [(a, [0, 1], 0.0), (b, 0, 0.0), (np.flatnonzero(m.nodes[:, 2] < 1e-9), 2, 0.0)]


@pytest.mark.slow
def test_factors_of_an_edge_crack_whose_lips_rub():
    """An edge crack closed by a compression σ, under a shear τ: K_II = (1 - μ σ / τ) K_II
    of the shear and G as its square (the uniform friction of the lips is a shear μ σ
    taken off); no K_I; lips that μ σ > τ holds have no factor at all."""
    image = VoxelImage(np.zeros((32, 64, 2), np.uint8), spacing=(0.5,) * 3, periodic=False)
    mesh = voxel_tet_mesh(image)
    shear = lambda m: _faces(m, [1.0, 0, 0], [0, 1.0, 0])  # noqa: E731
    closing = lambda m: _faces(m, [1.0, -1.0, 0], [0, 1.0, 0])  # noqa: E731
    two = solve_crack(mesh, CRACK, PLATE, _supports, forces=shear).modes()
    for friction in (0.4, 0.8):
        sol = solve_crack(mesh, CRACK, PLATE, _supports, forces=closing, contact=True,
                          friction=friction, steps=2)  # fmt: skip
        modes = sol.modes()
        assert modes["K_II"] == pytest.approx((1 - friction) * two["K_II"], abs=0.04 * two["K_II"])
        assert modes["G"] == pytest.approx((1 - friction) ** 2 * two["G"], rel=0.04)
        assert modes["K_I"] == 0 and modes["G_II"] == pytest.approx(modes["G"], rel=1e-4)
        assert sol.friction == friction and sol.steps.work["friction"][-1] > 0
        assert sol.steps.slip.shape == (sol.faces.n_elements, 3, 2)  # of the lips alone
    stuck = solve_crack(mesh, CRACK, PLATE, _supports, forces=closing, contact=True,
                        friction=1.5, steps=2)  # fmt: skip
    assert stuck.steps.iterations.max() <= 3  # the predictor: lips that stick
    modes = stuck.modes()
    assert abs(modes["K_II"]) < 0.04 * two["K_II"] and modes["G"] < 0.005 * two["G"]
    with pytest.raises(ValueError, match="friction"):
        solve_crack(mesh, CRACK, PLATE, _supports, forces=closing, friction=0.4)
