"""Cleavage cracks that close and twins that turn the lattice: the opening of a cleavage
plane goes back under pressure, the closed plane carries the pressure as the sound
crystal, its lips rub; the lattice of a point becomes that of its predominant twin at a
threshold of its twinned fraction (the disorientations of the twins, the Schmid law of
the turned lattice, detwinning); the tangents; the same in cells, on voxels and on meshes."""

import dataclasses
import math

import numpy as np
import pytest

from helpers import box_mesh
from morphogen.homogenization import Material
from morphogen.homogenization import crystal as cr
from morphogen.homogenization.crystal import CrystalPlasticity, disorientation, twin_systems
from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem
from morphogen.homogenization.nonlinear import NonlinearFEProblem, simulate_point
from morphogen.image import VoxelImage
from morphogen.orientations import euler_to_matrix
from test_twinning_cleavage import (
    IRON,
    MAGNESIUM,
    RATIO_MG,
    RATIO_ZN,
    ZINC,
    _kernel,
    _point,
    cubic,
    hexagonal,
    modulus,
    textured_magnesium,
)

# -- cracks that close ---------------------------------------------------------------------------


def test_cracks_close_under_pressure():
    """A cubic crystal pulled along [001] until its (001) plane has opened, then pushed:
    the plane keeps its opening down to the pressure of its strength, closes by the flow
    rule of its opening in the other sense, ``sigma = -(Y_0 exp(-w d) + K |d delta / dt|)``
    with ``d`` the opening cumulated in both senses, and once closed carries the pressure
    as the sound crystal; pulled again, it opens at the strength that its damage left.
    The work of the cycle is the energy that the damage dissipated."""
    strength, softening, k = 500.0, 100.0, 0.5
    planes = {"planes": "cube", "strength": strength, "softening": softening, "K": k, "n": 1.0}
    m = cubic(cleavage=planes)
    assert m.crystal_plasticity.cleavage_closure
    young = modulus(m)
    steps = 80
    path = {"times": [0, 1, 2, 3], "increments": steps, "strain": {"33": [0.008, -0.004, 0.008]}}
    res = simulate_point(m, path)
    stress, strain = res.stress[:, 2], res.strain[:, 2]
    opening, damage = res.series["cleavage_opening"], res.series["cleavage_damage"]
    assert res.info["cuts"] == 0
    # the strain is that of the lattice and of the opening, all along; the damage that of
    # the opening cumulated in both senses
    assert np.allclose(strain - stress / young, opening, rtol=0, atol=1e-10)
    moved = np.diff(opening)
    cumulated = np.concatenate([[0.0], np.cumsum(np.abs(moved))])
    assert np.allclose(damage, -np.expm1(-softening * cumulated), rtol=0, atol=1e-12)
    rate = np.abs(moved) * steps
    flow = strength * (1.0 - damage[1:]) + k * rate  # the flow rule at the end of each step
    top = steps
    closing = np.flatnonzero(moved[top:] < 0) + top  # the steps that close the plane
    closed = np.flatnonzero(opening[top:] == 0.0)[0] + top  # the first step that ends closed
    # unloaded, the plane keeps its opening down to the pressure of its softened strength
    softened = strength * (1.0 - damage[top])
    assert np.all(opening[top : closing[0] + 1] == opening[top]) and opening[top] > 0.005
    one = young * (strain[top] - strain[top + 1])  # an elastic step of the unloading
    assert -softened < stress[closing[0]] < -softened + one
    # it closes by its flow rule (but in the step that ends closed, held at no opening)
    assert closed == closing[-1] + 1 and len(closing) > 20
    inside = closing[:-1]
    assert np.allclose(stress[inside + 1], -flow[inside], rtol=1e-7, atol=1e-7 * strength)
    assert -stress[closed] > flow[closed - 1]  # a pressure that would have closed it more
    # closed: the stress of the sound crystal, the damage kept
    again = np.flatnonzero(opening[closed:] > 0)[0] + closed  # the first step that reopens
    shut = np.arange(closed, again)
    assert np.all(opening[shut] == 0.0) and np.all(damage[shut] == damage[closed])
    assert np.allclose(stress[shut], young * strain[shut], rtol=1e-9, atol=1e-9 * strength)
    assert stress[2 * steps] == pytest.approx(-0.004 * young, rel=1e-9)
    # pulled again: it opens at the strength that the damage left, and softens further
    left = strength * (1.0 - damage[closed])
    assert 0.25 * strength < left < softened and stress[again - 1] < left
    after = np.arange(again - 1, 3 * steps)
    assert np.allclose(stress[after + 1], flow[after], rtol=1e-7, atol=1e-7 * strength)
    assert damage[-1] > damage[closed] > damage[top]
    # the work of the cycle: the energy that the damage dissipated, the elastic energy left
    # and what the viscosity took (K rate^2 dt)
    work = np.trapezoid(stress, strain)
    dissipated = strength / softening * damage[-1] + k * np.sum(rate**2) / steps
    assert work - 0.5 * stress[-1] ** 2 / young == pytest.approx(dissipated, rel=0.01)
    # without the closure the plane keeps its opening: the lattice alone carries the pressure
    kept = simulate_point(cubic(cleavage={**planes, "closure": False}), path)
    was = kept.series["cleavage_opening"]
    assert np.allclose(kept.stress[: top + 1, 2], stress[: top + 1], rtol=0, atol=1e-9)
    assert np.all(was[top : 2 * steps + 1] == was[top])
    assert was[top] == pytest.approx(opening[top], abs=1e-12)
    assert kept.stress[2 * steps, 2] == pytest.approx(young * (-0.004 - was[top]), rel=1e-9)


@pytest.mark.parametrize("friction", [0.0, 0.4])
def test_lips_of_a_closed_crack_rub(friction):
    """A (001) plane broken in tension, closed again, then sheared under pressure: the
    lips slide when the shear stress comes to the friction of the pressure that holds
    them, ``|tau| = mu |sigma_n|`` (freely without friction), and no longer rub once the
    plane is pulled open. A sound plane under pressure is as much stronger in shear (the
    basal plane of zinc; in the cubic crystal the shear of a pressed {100} plane is
    released by the plane across it, which nothing presses)."""
    strength, shear = 500.0, 0.4
    planes = {"planes": "cube", "strength": strength, "softening": 100.0, "shear": shear,
              "K": 0.05, "n": 1.0, "friction": friction}  # fmt: skip
    m = cubic(cleavage=planes)
    assert m.crystal_plasticity.cleavage_friction == friction
    young, c44 = modulus(m), IRON[2]
    steps = 60
    path = {
        "times": [0, 1, 2, 3, 4],
        "increments": steps,
        "strain": {"33": [0.08, -0.002, -0.002, 0.004], "31": [0.0, 0.0, 0.01, 0.01]},
    }
    res = simulate_point(m, path)
    normal, tangential = res.stress[:, 2], res.stress[:, 4]
    opening, damage = res.series["cleavage_opening"], res.series["cleavage_damage"]
    assert damage[steps] > 0.999 and opening[2 * steps] == 0.0  # broken, then closed
    pressure = -normal[2 * steps]
    assert pressure == pytest.approx(0.002 * young, rel=1e-6)
    # sheared: elastic up to the friction of the pressure, then the lips slide at it
    slides = friction * pressure
    sheared = np.arange(2 * steps + 1, 3 * steps + 1)
    elastic = c44 * res.strain[sheared, 4]
    expected = np.minimum(elastic, slides)
    assert np.allclose(tangential[sheared], expected, rtol=0, atol=2e-3 * strength)
    assert np.allclose(normal[sheared], -pressure, rtol=1e-6)  # the pressure is that of the lattice
    if friction:
        assert tangential[3 * steps] == pytest.approx(slides, rel=5e-3) and slides < elastic[-1]
    # pulled open: nothing holds the lips, the shear stress is released
    assert opening[-1] > 0.003 and abs(tangential[-1]) < 2e-3 * strength
    # a sound (001) plane sheared under pressure: the (100) plane, which nothing presses,
    # releases the shear at its own strength
    sound = {"times": [0, 1, 2], "increments": 40,
             "strain": {"33": [-0.002, -0.002], "31": [0.0, 0.004]}}  # fmt: skip
    res = simulate_point(m, sound)
    slip = res.strain[-1, 4] - res.stress[-1, 4] / c44
    assert slip > 1e-3 and np.all(res.stress[:, 0] > -1e-6)
    # (without friction the two planes slide alike, each by half of the slip)
    each = slip if friction else 0.5 * slip
    assert res.stress[-1, 4] == pytest.approx(shear * strength * math.exp(-100.0 * each), rel=2e-3)
    # the basal plane of zinc, alone of its kind: its shear strength and the friction of
    # the pressure; pulled instead (below its strength), the shear strength alone
    basal = {"planes": "basal", "strength": 80.0, "softening": 50.0, "shear": 0.5, "K": 0.01,
             "n": 1.0, "friction": friction}  # fmt: skip
    zinc = hexagonal(ZINC, c_over_a=RATIO_ZN, cleavage=basal)
    c33, c44 = ZINC[3], ZINC[4]
    for strain, pressed in ((-1e-3, True), (5e-4, False)):
        sound = {"times": [0, 1, 2], "increments": 40,
                 "strain": {"11": [0.0, 0.0], "22": [0.0, 0.0], "33": [strain, strain],
                            "23": [0.0, 0.0], "31": [0.0, 0.004], "12": [0.0, 0.0]}}  # fmt: skip
        res = simulate_point(zinc, sound)
        assert res.stress[40, 2] == pytest.approx(c33 * strain, rel=1e-9)
        slip = res.strain[-1, 4] - res.stress[-1, 4] / c44
        held = 0.5 * 80.0 * math.exp(-50.0 * slip) + (friction * c33 * 1e-3 if pressed else 0.0)
        assert slip > 5e-4 and res.stress[-1, 4] == pytest.approx(held, rel=2e-3)


def test_closure_in_a_configuration():
    spec = {"planes": "cube", "strength": 500.0, "softening": 100.0, "K": 1.0, "n": 2.0}
    cp = CrystalPlasticity.from_config({"cleavage": spec})
    assert cp.cleavage_closure and cp.cleavage_friction == 0.0 and not cp.reorients
    assert "closure" not in cp.to_config()["cleavage"]
    kinds = cp.kinds_array()
    assert kinds.shape == (9, cr.CP_WIDTH - cr.CP_KIND)
    opening = kinds[:, 0] == cr.OPENING
    assert np.all(kinds[opening, 4] == 1.0) and np.all(kinds[~opening, 4] == 0.0)
    # the opening system of the plane of each accommodation system
    assert kinds[:, 6].tolist() == [-1, 0, 0, -1, 3, 3, -1, 6, 6]
    rubbing = CrystalPlasticity.from_config({"cleavage": {**spec, "friction": 0.3}})
    assert np.all(rubbing.kinds_array()[~opening, 4] == 0.3)
    assert rubbing.to_config()["cleavage"]["friction"] == 0.3
    assert CrystalPlasticity.from_config(rubbing.to_config()) == rubbing
    open_ = CrystalPlasticity.from_config({"cleavage": {**spec, "closure": False}})
    assert not open_.cleavage_closure and np.all(open_.kinds_array()[:, 4] == 0.0)
    assert open_.to_config()["cleavage"]["closure"] is False
    assert CrystalPlasticity.from_config(open_.to_config()) == open_
    for bad, match in [
        ({**spec, "closure": "yes"}, "closure: true"),
        ({**spec, "friction": -0.1}, "friction of 0 or more"),
        ({**spec, "friction": "rough"}, "friction: a number"),
        ({**spec, "closure": False, "friction": 0.3}, "planes that close"),
    ]:
        with pytest.raises(ValueError, match=match):
            CrystalPlasticity.from_config({"cleavage": bad})


# -- twins that turn the lattice -----------------------------------------------------------------


def _twin_rotations(family, ratio=None):
    normals, _, _ = twin_systems(family, ratio)
    return 2.0 * np.einsum("ti,tj->tij", normals, normals) - np.eye(3)


def test_disorientations_of_the_twins():
    """The lattice of a twin is that of its parent turned by half a turn about the normal
    of the twin plane; its disorientation, the smallest angle over the symmetries of the
    lattice: 60 degrees for the twins of the cubic crystals (the boundary Sigma 3), and
    for the hexagonal ones twice the angle of the twin plane and the basal plane, or its
    supplement (86.3 degrees for the {10-12} twin of magnesium, 56.2 for {10-11})."""
    for family in ("twin111", "twin112"):
        assert np.allclose(disorientation(_twin_rotations(family), "cubic"), 60.0, atol=1e-9)
    for ratio in (RATIO_MG, 1.587, RATIO_ZN):
        # the tangent of the angle of the twin plane and the basal plane
        slopes = {"twin1012": ratio / math.sqrt(3.0), "twin1011": 2.0 * ratio / math.sqrt(3.0),
                  "twin1122": ratio, "twin1121": 2.0 * ratio}  # fmt: skip
        for family, slope in slopes.items():
            twice = 2.0 * math.degrees(math.atan(slope))
            expected = min(twice, 180.0 - twice)
            angles = disorientation(_twin_rotations(family, ratio), "hexagonal")
            assert np.allclose(angles, expected, atol=1e-9), (family, ratio)
    magnesium = disorientation(_twin_rotations("twin1012", RATIO_MG), "hexagonal")
    assert magnesium[0] == pytest.approx(86.3, abs=0.05)
    assert disorientation(_twin_rotations("twin1011", RATIO_MG), "hexagonal")[0] == pytest.approx(
        56.2, abs=0.1
    )
    titanium = disorientation(_twin_rotations("twin1121", 1.587), "hexagonal")
    assert titanium[0] == pytest.approx(35.0, abs=0.1)
    # no rotation, the symmetries of the lattices, a small rotation
    assert disorientation(np.eye(3), "cubic") == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(disorientation(np.array(cr._rotations("hexagonal")), "hexagonal"), 0.0,
                       atol=1e-6)  # fmt: skip
    small = euler_to_matrix((7.0, 0.0, 0.0))[0]
    assert disorientation(small, "cubic") == pytest.approx(7.0, abs=1e-9)
    # the stiffness of a turned lattice: that of the fourth-order tensor
    rng = np.random.default_rng(3)
    c = rng.normal(size=(6, 6))
    c = c + c.T
    q = euler_to_matrix((25.0, 40.0, 70.0))[0]
    out, bond, tmp = np.zeros((6, 6)), np.zeros((6, 6)), np.zeros((6, 6))
    cr.rotate_stiffness(c, q, out, bond, tmp)
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1)]
    full = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(pairs):
        for b, (k, ll) in enumerate(pairs):
            full[i, j, k, ll] = full[j, i, k, ll] = full[i, j, ll, k] = full[j, i, ll, k] = c[a, b]
    turned = np.einsum("ia,jb,kc,ld,abcd->ijkl", q, q, q, q, full)
    expected = np.array([[turned[i, j, k, ll] for k, ll in pairs] for i, j in pairs])
    assert np.allclose(out, expected, rtol=0, atol=1e-12 * np.abs(c).max())


def _magnesium(k=0.1, n=20.0, hardening=0.0, **more):
    """Magnesium that twins easily and slips with difficulty (a twin that stays elastic);
    ``hardening``: of its twins (two twins that flow alike share their shear in one way
    only if they harden)."""
    return hexagonal(
        MAGNESIUM,
        c_over_a=RATIO_MG,
        K=k,
        n=n,
        families={
            "basal": {"tau0": 600.0},
            "prismatic": {"tau0": 600.0},
            "pyramidal2_ca": {"tau0": 600.0},
        },
        twins={"twin1012": {"tau0": 15.0, "Q": hardening, "b": 30.0}},
        **more,
    )


def _schmid(cp, axis, rotation=None):
    """The Schmid factors of the systems of a crystal pulled along ``axis`` (in its axes),
    its lattice turned by ``rotation``."""
    normals, directions = cp.systems()
    if rotation is not None:
        normals, directions = normals @ rotation.T, directions @ rotation.T
    return (normals @ axis) * (directions @ axis)


def test_twins_untwin():
    """A crystal of magnesium pulled along its axis c, then pushed: its twins shear back
    at the resolved stress at which they formed, the twinned fraction going down with the
    strain as it went up, ``F = eps_p / (gamma_tw m)``; once it has untwinned the crystal
    is as it was (no strain left) and carries the pressure by its hard slip; pulled again,
    it twins as at first. Without detwinning it keeps its twins, and slips at once."""
    spec = {"twins": {"twin1012": {"tau0": 15.0}}, "c_over_a": RATIO_MG, "K": 0.05, "n": 20.0,
            "families": {"pyramidal2_ca": {"tau0": 90.0}}}  # fmt: skip
    m = hexagonal(MAGNESIUM, **spec)
    cp = m.crystal_plasticity
    assert cp.twin_detwinning and np.all(cp.kinds_array()[cp.n_slip :, 4] == 1.0)
    axis = np.array([0.0, 0.0, 1.0])
    factors = _schmid(cp, axis)
    twin, slip = factors[cp.n_slip], np.abs(factors[: cp.n_slip]).max()
    assert np.allclose(factors[cp.n_slip :], twin) and twin == pytest.approx(0.4989, abs=1e-4)
    shear = twin_systems("twin1012", RATIO_MG)[2]
    steps = 100
    path = {"times": [0, 1, 2, 3], "increments": steps, "strain": {"33": [0.03, -0.01, 0.02]}}
    res = simulate_point(m, path)
    assert res.info["cuts"] == 0
    stress, strain = res.stress[:, 2], res.strain[:, 2]
    fraction = res.series["twin_fraction"]
    young = modulus(m)
    # the twinned fraction is the strain of the twins over their shear, there and back
    plastic = strain - stress / young
    undone = int(np.flatnonzero((fraction == 0.0) & (np.arange(len(fraction)) > steps))[0])
    assert np.allclose(fraction[:undone], plastic[:undone] / (shear * twin), rtol=0, atol=1e-9)
    assert fraction[steps] == pytest.approx(0.03 / (shear * twin), rel=0.02)
    # twinning and detwinning at Schmid's stress
    flow = 15.0 / twin
    assert np.all(stress[steps // 2 : steps + 1] > flow)
    assert np.all(stress[steps // 2 : steps + 1] < 1.01 * flow)
    back = np.flatnonzero(np.diff(fraction[: undone + 1]) < 0)[1:-1] + 1  # steps that untwin
    assert len(back) > 60 and np.all(stress[back] < -flow) and np.all(stress[back] > -1.01 * flow)
    # untwinned: as it was, then the slip of the pressed crystal
    held = np.arange(undone, 2 * steps + 1)
    assert np.all(fraction[held] == 0.0)
    assert np.all(np.abs(plastic[undone : undone + 3]) < 1e-9)
    assert stress[2 * steps] == pytest.approx(-90.0 / slip, rel=0.01)
    # pulled again: it twins as at first
    assert fraction[-1] > 0.1 and stress[-1] == pytest.approx(flow, rel=0.01)
    # without detwinning: the twins stay, the crystal slips as soon as it is pushed
    kept = simulate_point(hexagonal(MAGNESIUM, **spec, twin_detwinning=False), path)
    was = kept.series["twin_fraction"]
    assert np.allclose(kept.stress[: steps + 1, 2], stress[: steps + 1], rtol=0, atol=1e-9)
    assert np.all(was[steps : 2 * steps + 1] == was[steps])
    slipping = np.flatnonzero(kept.stress[steps : 2 * steps, 2] < -0.99 * 90.0 / slip)
    assert slipping[0] < 20 and stress[steps + slipping[0]] > -1.01 * flow
    cfg = kept.info.get("cuts", 0)
    assert cfg == 0
    plain = hexagonal(MAGNESIUM, **spec, twin_detwinning=False).crystal_plasticity
    assert not plain.twin_detwinning and np.all(plain.kinds_array()[:, 4] == 0.0)
    assert plain.to_config()["twin_detwinning"] is False and "twin_detwinning" not in cp.to_config()
    assert CrystalPlasticity.from_config(plain.to_config()) == plain
    with pytest.raises(ValueError, match="twin_detwinning: true"):
        CrystalPlasticity.from_config({**spec, "twin_detwinning": "no"})


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_twins_turn_the_lattice(kinematics):
    """A crystal of magnesium pulled 8 degrees off its axis c: it twins on its best
    {10-12} system at the stress of Schmid's law; at the threshold of its twinned
    fraction its lattice becomes that of the twin (86.3 degrees off) and its twins start
    again; pushed back, the twin shears the other way at the same resolved stress
    (detwinning), and at the same threshold the lattice is that of the grain again, the
    strain of the twin taken back. The twinned fraction is that of the shears of the
    twins, there and back: a turn does not change it."""
    threshold = 0.6
    plain, turning = _magnesium(), _magnesium(twin_reorientation=threshold)
    cp = turning.crystal_plasticity
    assert cp.reorients and not cp.by_grain and not plain.crystal_plasticity.reorients
    assert cp.field_names == ("twin_fraction", "twin_reoriented", "twin_misorientation")
    euler = (0.0, 8.0, 0.0)
    axis = euler_to_matrix(euler)[0] @ np.array([0.0, 0.0, 1.0])  # of the tension, in the crystal
    factors = _schmid(cp, axis)
    twins = slice(cp.n_slip, cp.n_slip + cp.n_twins)
    # two twins share the largest factor, mirror images in the plane of the tilt: either
    # is the predominant one, with the same response along the axis
    best = cp.n_slip + int(np.argmax(factors[twins]))
    assert np.sort(factors[twins])[-2] == pytest.approx(factors[best], rel=1e-9)
    steps = 200
    path = {"times": [0, 1, 2], "increments": steps, "strain": {"33": [0.055, -0.012]}}
    res = simulate_point(turning, path, orientation=euler, kinematics=kinematics)
    ref = simulate_point(plain, path, orientation=euler, kinematics=kinematics)
    stress, strain = res.stress[:, 2], res.strain[:, 2]
    fraction, angle = res.series["twin_fraction"], res.series["twin_misorientation"]
    assert np.array_equal(res.series["twin_reoriented"], (angle > 0).astype(float))
    turned = int(np.flatnonzero(angle > 0)[0])  # the first step that ends with the lattice turned
    steps = int(np.searchsorted(res.times, 1.0 - 1e-9))  # the end of the tension
    # up to there, the crystal whose lattice does not turn; twinning at Schmid's stress
    assert np.array_equal(res.times[:turned], ref.times[:turned])
    assert np.allclose(stress[:turned], ref.stress[:turned, 2], rtol=0, atol=1e-9)
    assert np.allclose(fraction[:turned], ref.series["twin_fraction"][:turned], atol=1e-12)
    flow = 15.0 / factors[best]
    plateau = stress[turned // 2 : turned]
    # (in finite strains the stress is nominal and the lattice turns a little with the shear)
    slack, low = (0.02, 1.0) if kinematics == "small" else (0.06, 0.94)
    assert np.all(plateau > low * flow) and np.all(plateau < (1 + slack) * flow + 0.5)
    assert fraction[turned - 1] < threshold <= ref.series["twin_fraction"][turned]
    # turned: the lattice of the point is the twin, at the disorientation of the {10-12}
    # twin; its twinned fraction is the one that its twins sheared
    assert fraction[turned] == pytest.approx(ref.series["twin_fraction"][turned], abs=1e-9)
    assert angle[turned] == pytest.approx(86.31, abs=0.01)
    assert np.all(angle[turned : steps + 1] == angle[turned])
    # the twin is hard to slip and does not twin again: it is elastic up to the end
    assert np.all(fraction[turned : steps + 1] == fraction[turned])
    assert res.plastic[steps] == pytest.approx(res.plastic[turned], abs=1e-12)
    assert stress[steps] > 4 * flow
    if kinematics == "small":
        # the Schmid law of the turned lattice: no system of the twin flows below this
        normals, _ = cp.systems()
        q = 2.0 * np.outer(normals[best], normals[best]) - np.eye(3)
        after = _schmid(cp, axis, q)
        assert after[best] == pytest.approx(-factors[best], abs=1e-12)  # the same twin, reversed
        tau0 = cp.params_array()[:, 0]
        limit = (tau0[: cp.n_slip] / np.abs(after[: cp.n_slip])).min()
        assert stress[steps] < limit
        assert np.all(after[twins] < 1e-9)  # and no twin of the twin is pulled the right way
    # pushed back: detwinning at the stress of the twinning, reversed; the other twins of
    # the family are held (no twin within the twin: the plastic strain is that of the
    # twin taken back)
    back = np.flatnonzero((fraction < fraction[turned]) & (np.arange(len(fraction)) > steps))
    undone = int(np.flatnonzero((angle == 0) & (np.arange(len(angle)) > steps))[0])
    during = back[back < undone]
    assert len(during) > 20 and np.all(np.diff(fraction[during]) < 0)
    inside = during[len(during) // 4 :]
    assert np.all(stress[inside] < -low * flow)
    assert np.all(stress[inside] > -(1 + slack) * flow - 0.5)
    assert fraction[undone - 1] == pytest.approx(fraction[turned] - threshold, abs=0.02)
    assert np.all(angle[steps:undone] == angle[turned])
    # the lattice of the grain again: no rotation, what is left of the twin (what the
    # first threshold overshot, a step of twinning) shearing back in it, and the strain
    # of the twin taken back
    assert fraction[undone] < 0.01 and np.all(angle[undone:] == 0.0)
    assert np.all(fraction[undone + 2 :] == 0.0)
    young = stress[1] / strain[1]
    residual = strain[undone] - stress[undone] / young
    per_step = (0.055 + 0.012) / 200
    assert abs(residual) < (2.5 if kinematics == "small" else 8.0) * per_step
    # and it twins again when pulled, as at first
    again = {"times": [0, 1, 2, 3], "increments": steps, "strain": {"33": [0.055, -0.012, 0.03]}}
    res = simulate_point(turning, again, orientation=euler, kinematics=kinematics)
    assert res.series["twin_misorientation"][-1] == 0.0
    assert 0.3 < res.series["twin_fraction"][-1] < threshold
    assert res.stress[-1, 2] == pytest.approx(flow, rel=slack + 0.02)


def test_twin_of_a_cubic_crystal_and_back():
    """A face-centred cubic crystal sheared along one {111}<11-2> twin system, its whole
    strain prescribed: the system twins alone at its resolved stress; at the threshold the
    lattice is that of the twin, 60 degrees off, and is elastic; sheared back, the twin is
    undone at the same resolved stress, and the crystal is as it was: the lattice of the
    grain, no plastic strain left but what the two thresholds overshot."""
    tau0, threshold = 60.0, 0.2
    m = cubic(K=0.05, n=20.0, twins={"twin111": {"tau0": tau0}}, twin_reorientation=threshold)
    cp = m.crystal_plasticity
    normals, directions = cp.systems()
    twin = 3
    schmid = 0.5 * (np.outer(directions[twin], normals[twin])
                    + np.outer(normals[twin], directions[twin]))  # fmt: skip
    voigt = np.array([schmid[0, 0], schmid[1, 1], schmid[2, 2], 2 * schmid[1, 2],
                      2 * schmid[2, 0], 2 * schmid[0, 1]])  # fmt: skip
    shear = twin_systems("twin111")[2]
    assert shear == pytest.approx(1 / math.sqrt(2))
    step = 5e-4
    there, back = threshold * shear + 3 * step, -8 * step
    counts = [round(there / step), round((there - back) / step)]
    names = ("11", "22", "33", "23", "31", "12")
    strains = {c: [there * v, back * v] for c, v in zip(names, voigt, strict=True)}
    path = {"times": [0, 1, 2], "increments": counts, "strain": strains}
    res = simulate_point(m, path)
    assert res.info["cuts"] == 0
    tau = res.stress @ voigt  # the resolved shear stress of the twin
    fraction, angle = res.series["twin_fraction"], res.series["twin_misorientation"]
    turned = int(np.flatnonzero(angle > 0)[0])
    assert turned < counts[0] and angle[turned] == pytest.approx(60.0, abs=1e-6)
    twinning = np.arange(turned // 2, turned)
    assert np.all(tau[twinning] > tau0) and np.all(tau[twinning] < 1.02 * tau0)
    # the fraction is the shear of the twin over its shear: the others do not twin
    gamma = res.strain @ voigt / (voigt @ voigt)  # the shear prescribed along the system
    elastic = tau[1] / gamma[1]
    assert np.allclose(fraction[:turned], (gamma[:turned] - tau[:turned] / elastic) / shear,
                       rtol=0, atol=2e-4)  # fmt: skip
    assert fraction[turned - 1] < threshold <= fraction[turned] < threshold + 0.002
    assert res.series["twin_reoriented"][turned] == 1.0
    # the twin: elastic, stiffer steps
    assert np.all(np.diff(tau[turned : counts[0] + 1]) > 0.5 * elastic * step)
    undone = int(np.flatnonzero((angle == 0) & (np.arange(len(angle)) > counts[0]))[0])
    untwinning = np.flatnonzero((fraction < fraction[turned]) & (np.arange(len(fraction)) < undone))
    untwinning = untwinning[untwinning > turned][5:]
    assert len(untwinning) > 100
    assert np.all(tau[untwinning] < -tau0) and np.all(tau[untwinning] > -1.02 * tau0)
    assert fraction[undone - 1] == pytest.approx(fraction[turned] - threshold, abs=0.01)
    assert fraction[undone] < 0.002 and np.all(angle[undone:] == 0.0)
    # as it was: the stress of the strain, but the overshoot of the two thresholds
    stiffness = m.matrix("elastic")
    left = res.strain[undone] - np.linalg.solve(stiffness, res.stress[undone])
    assert np.abs(left).max() < 2.5 * step * np.abs(voigt).max()
    assert res.plastic[undone] > 0.1  # though it has sheared there and back


def test_reorientation_in_a_configuration():
    spec = {"twins": {"twin1012": {"tau0": 15.0}}, "c_over_a": RATIO_MG, "K": 1.0, "n": 5.0}
    cp = CrystalPlasticity.from_config({**spec, "twin_reorientation": 0.8})
    assert cp.reorients and cp.twin_reorientation == 0.8 and cp.lattice == "hexagonal"
    assert cp.to_config()["twin_reorientation"] == 0.8
    assert CrystalPlasticity.from_config(cp.to_config()) == cp
    shear = twin_systems("twin1012", RATIO_MG)[2]
    kinds = cp.kinds_array()
    assert np.allclose(kinds[:, 5], 1.0 / (0.8 * shear)) and np.allclose(kinds[:, 2], 1.0 / shear)
    plain = CrystalPlasticity.from_config(spec)
    assert not plain.reorients and "twin_reorientation" not in plain.to_config()
    assert np.all(plain.kinds_array()[:, 5] == 0.0)
    assert plain.field_names == ("twin_fraction",)
    # the internal variables of the turned lattices: after those of the kernels
    n = cp.n_systems
    assert cr.state_width(n) == cr.CP_ALPHA + 2 * n + 6
    assert cr.state_width(n, turns=True) == cr.state_width(n) + cr.CP_TURNED
    assert cr.state_width(n, True, True) == cr.state_width(n, True) + cr.CP_TURNED
    m = hexagonal(MAGNESIUM, **spec, twin_reorientation=0.8)
    for finite in (False, True):
        assert _kernel(m, (0.0, 0.0, 0.0), finite)[1].width == cr.state_width(n, finite, True)
    for bad, match in [
        ({**spec, "twin_reorientation": 1.0}, "in \\]0, twin_saturation = 1\\["),
        ({**spec, "twin_reorientation": 0.0}, "in \\]0, twin_saturation"),
        ({**spec, "twin_saturation": 0.5, "twin_reorientation": 0.6}, "twin_saturation = 0.5"),
        ({**spec, "twin_reorientation": "soon"}, "are numbers"),
        (
            {
                "families": ["basal"],
                "tau0": 1.0,
                "K": 1.0,
                "n": 2.0,
                "c_over_a": 1.6,
                "twin_reorientation": 0.5,
            },
            "that of twins",
        ),
    ]:
        with pytest.raises(ValueError, match=match):
            CrystalPlasticity.from_config(bad)


# -- the tangents ---------------------------------------------------------------------------------


def _differences(table, beh, grad, old, dt, finite, h=1e-8):
    fd = None
    for j in range(grad.size):
        d = np.zeros(grad.size)
        d[j] = h
        up = _point(table, beh, grad + d, old, dt, finite)[0]
        down = _point(table, beh, grad - d, old, dt, finite)[0]
        fd = np.zeros((up.size, grad.size)) if fd is None else fd
        fd[:, j] = (up - down) / (2 * h)
    return fd


def _gradient(crystal, euler, finite):
    """The strain (or the displacement gradient, with a small rotation) of a symmetric
    tensor given in the axes of the crystal."""
    g = euler_to_matrix(euler)[0]
    sample = g.T @ np.asarray(crystal) @ g
    if finite:
        grad = sample.ravel().copy()
        grad[[1, 3]] += [3e-4, -3e-4]
        return grad
    return np.array([sample[0, 0], sample[1, 1], sample[2, 2], 2 * sample[1, 2],
                     2 * sample[2, 0], 2 * sample[0, 1]])  # fmt: skip


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("case", ["closing", "closed", "rubbing"])
def test_tangent_of_a_plane_that_closes(finite, case):
    """The tangent of the implicit update against finite differences where the basal
    plane of a zinc crystal closes under pressure: while it closes, once it is held
    closed (the tangent is then that of the lattice and of the slips), and with its lips
    that rub on each other under the pressure; the same state from a warm start."""
    # (basal slip would release the shear of the basal plane before its lips rub)
    families = {"pyramidal2_ca": {"tau0": 60.0}}
    if case != "rubbing":
        families["basal"] = {"tau0": 10.0, "Q": 8.0, "b": 20.0}
    planes = {"planes": "basal", "strength": 90.0, "softening": 150.0, "shear": 0.15, "K": 30.0,
              "n": 2.0}  # fmt: skip
    friction = 0.1 if case == "rubbing" else 0.0
    m = hexagonal(ZINC, c_over_a=RATIO_ZN, K=20.0, n=4.0, families=families,
                  cleavage={**planes, "friction": friction})  # fmt: skip
    cp = m.crystal_plasticity
    euler = (20.0, 35.0, 50.0)
    table, beh = _kernel(m, euler, finite)
    ns = cp.n_systems
    first = cp.n_slip  # the opening of the basal plane, then its three accommodations
    old = np.zeros((1, beh.width))
    alpha = old[0, cr.CP_ALPHA : cr.CP_ALPHA + ns]
    v = old[0, cr.CP_ALPHA + ns : cr.CP_ALPHA + 2 * ns]
    alpha[first] = v[first] = 0.004  # the basal plane has opened
    # in the axes of the crystal: a pressure along c, shears of the basal plane
    push = {"closing": -2.5e-3, "closed": -9e-3, "rubbing": -9e-3}[case]
    crystal = [[1e-3, 5e-4, 8e-4], [5e-4, -5e-4, 6e-4], [8e-4, 6e-4, push]]
    grad = _gradient(crystal, euler, finite)
    s, a, new = _point(table, beh, grad, old, 0.5, finite)
    opening = new[0, cr.CP_ALPHA + first]
    slid = np.abs(new[0, cr.CP_ALPHA + ns + first + 1 : cr.CP_ALPHA + 2 * ns]).max()
    if case == "closing":
        assert 1e-3 < opening < 3.5e-3 and slid > 1e-5
    else:
        assert opening == 0.0 and new[0, cr.CP_ALPHA + ns + first] == pytest.approx(0.008)
        assert slid > 1e-5  # the lips slide, with or without friction
    fd = _differences(table, beh, grad, old, 0.5, finite)
    assert np.allclose(a, fd, rtol=0, atol=2e-6 * np.abs(fd).max())
    s1, _, new1 = _point(table, beh, grad, old, 0.5, finite, guess=1, new=new)
    assert np.allclose(new1, new, rtol=0, atol=1e-11) and np.allclose(s1, s, atol=1e-7)
    if case == "rubbing":
        # the friction holds the lips: they slide less than without it
        free = hexagonal(ZINC, c_over_a=RATIO_ZN, K=20.0, n=4.0, families=families,
                         cleavage=planes)  # fmt: skip
        table0, beh0 = _kernel(free, euler, finite)
        new0 = _point(table0, beh0, grad, old, 0.5, finite)[2]
        assert (
            slid < 0.8 * np.abs(new0[0, cr.CP_ALPHA + ns + first + 1 : cr.CP_ALPHA + 2 * ns]).max()
        )


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("case", ["untwinning", "undone"])
def test_tangent_of_twins_that_untwin(finite, case):
    """The tangent of the implicit update against finite differences for a crystal of
    magnesium that has nearly twinned as far as it may and is pressed along its axis c:
    while its twins shear back (the barrier of their saturation does not hold them: it
    is that of the twinning), and in the step in which they are undone (held at no twin);
    its twins harden with their net shear. The same state from a warm start."""
    m = hexagonal(
        MAGNESIUM,
        c_over_a=RATIO_MG,
        K=5.0,
        n=4.0,
        families={"basal": {"tau0": 4.0, "Q": 3.0, "b": 20.0}, "pyramidal2_ca": {"tau0": 40.0}},
        twins={"twin1012": {"tau0": 12.0, "Q": 6.0, "b": 15.0, "C": 200.0, "D": 10.0}},
    )
    cp = m.crystal_plasticity
    euler = (15.0, 25.0, 40.0)
    table, beh = _kernel(m, euler, finite)
    ns = cp.n_systems
    shear = twin_systems("twin1012", RATIO_MG)[2]
    twins = slice(cr.CP_ALPHA + ns + cp.n_slip, cr.CP_ALPHA + ns + cp.n_slip + cp.n_twins)
    old = np.zeros((1, beh.width))
    share = 0.97 if case == "untwinning" else 0.012
    old[0, twins] = share * shear / 6  # the net shear of its six twins
    crystal = [[1.2e-3, 5e-4, 8e-4], [5e-4, 0.8e-3, 6e-4], [8e-4, 6e-4, -4e-3]]
    grad = _gradient(crystal, euler, finite)
    s, a, new = _point(table, beh, grad, old, 0.5, finite)
    left = new[0, twins]
    if case == "untwinning":
        assert np.all(left < old[0, twins]) and left.min() > 0.5 * shear / 6
    else:
        assert np.all(left == 0.0)  # undone, and held there
    assert new[0, cr.CP_ALPHA + ns : cr.CP_ALPHA + ns + cp.n_slip].max() > 1e-4  # and it slips
    fd = _differences(table, beh, grad, old, 0.5, finite)
    assert np.allclose(a, fd, rtol=0, atol=2e-6 * np.abs(fd).max())
    s1, _, new1 = _point(table, beh, grad, old, 0.5, finite, guess=1, new=new)
    assert np.allclose(new1, new, rtol=0, atol=1e-11) and np.allclose(s1, s, atol=1e-7)
    # the step after, from the slips of this one
    s2 = _point(table, beh, 1.3 * grad, new, 0.5, finite, guess=2, new=old)[0]
    s3 = _point(table, beh, 1.3 * grad, new, 0.5, finite)[0]
    assert np.allclose(s2, s3, rtol=0, atol=1e-7 * np.abs(s3).max())


@pytest.mark.parametrize("finite", [False, True])
def test_tangent_of_a_turned_lattice(finite):
    """The tangent of the implicit update against finite differences for a point whose
    lattice a twin has turned, and in the step at the end of which another twin turns it
    again (the stress of the step is that of the lattice of its start)."""
    m = hexagonal(
        MAGNESIUM,
        c_over_a=RATIO_MG,
        K=5.0,
        n=4.0,
        families={
            "basal": {"tau0": 4.0, "Q": 3.0, "b": 20.0},
            "prismatic": {"tau0": 25.0},
            "pyramidal2_ca": {"tau0": 40.0},
        },
        twins={"twin1012": {"tau0": 12.0}},
        twin_reorientation=0.5,
    )
    cp = m.crystal_plasticity
    euler = (15.0, 25.0, 40.0)
    table, beh = _kernel(m, euler, finite)
    ns = cp.n_systems
    width = cr.state_width(ns, finite)
    assert beh.width == width + cr.CP_TURNED
    g = euler_to_matrix(euler)[0]
    normals, _ = cp.systems()
    twin = cp.n_slip + 2
    n = g.T @ normals[twin]  # the normal of the twin plane in the axes of the sample
    old = np.zeros((1, beh.width))
    old[0, width : width + 9] = (2.0 * np.outer(n, n) - np.eye(3)).ravel() - np.eye(3).ravel()
    old[0, width + 9] = twin + 1
    # in the axes of the twin: an extension along its axis c (it twins), shears (it slips)
    q = 2.0 * np.outer(normals[twin], normals[twin]) - np.eye(3)
    crystal = q @ np.array([[-1e-3, 6e-4, 9e-4], [6e-4, -1.5e-3, 5e-4], [9e-4, 5e-4, 3e-3]]) @ q.T
    grad = _gradient(crystal, euler, finite)
    s, a, new = _point(table, beh, grad, old, 0.5, finite)
    moved = new[0, cr.CP_ALPHA + ns : cr.CP_ALPHA + 2 * ns]
    assert moved[: cp.n_slip].max() > 1e-4 and moved[cp.n_slip :].max() > 1e-5
    assert np.array_equal(new[0, width:], old[0, width:])  # the lattice stays that of the twin
    fd = _differences(table, beh, grad, old, 0.5, finite)
    assert np.allclose(a, fd, rtol=0, atol=2e-6 * np.abs(fd).max())
    s1, _, new1 = _point(table, beh, grad, old, 0.5, finite, guess=1, new=new)
    assert np.allclose(new1, new, rtol=0, atol=1e-11) and np.allclose(s1, s, atol=1e-7)
    # not the stress of the lattice of the grain
    plain = np.zeros((1, beh.width))
    assert (
        np.abs(_point(table, beh, grad, plain, 0.5, finite)[0] - s).max() > 0.05 * np.abs(s).max()
    )
    # a point about to turn: the twins of the step bring its fraction over the threshold
    shear = twin_systems("twin1012", RATIO_MG)[2]
    near = old.copy()
    near[0, cr.CP_ALPHA + ns + cp.n_slip : cr.CP_ALPHA + ns + cp.n_slip + cp.n_twins] = (
        0.499 * shear / 6
    )
    s2, a2, new2 = _point(table, beh, grad, near, 0.5, finite)
    assert not np.array_equal(new2[0, width : width + 9], near[0, width : width + 9])
    twins = slice(cr.CP_ALPHA + ns + cp.n_slip, cr.CP_ALPHA + ns + cp.n_slip + cp.n_twins)
    assert np.all(new2[0, twins] == 0.0)  # the twins of the new lattice start from nothing
    fd = _differences(table, beh, grad, near, 0.5, finite)
    assert np.allclose(a2, fd, rtol=0, atol=2e-6 * np.abs(fd).max())
    # warm starts across the turn: from the state of the step, and from the one before
    s3, _, new3 = _point(table, beh, grad, near, 0.5, finite, guess=1, new=new2)
    assert np.allclose(new3, new2, rtol=0, atol=1e-11) and np.allclose(s3, s2, atol=1e-7)
    s4 = _point(table, beh, grad, new2, 0.5, finite, guess=2, new=near)[0]
    s5 = _point(table, beh, grad, new2, 0.5, finite)[0]
    assert np.allclose(s4, s5, rtol=0, atol=1e-7 * np.abs(s5).max())


# -- cells ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_homogeneous_cells_close_and_turn(kinematics):
    """Homogeneous crystals follow their material point, on voxels and on a mesh: iron
    whose cleavage plane opens, closes and rubs; magnesium whose twin turns the lattice
    and turns it back. Their fields are those of the point; in finite strains the
    orientation of the elements is that of the twin."""
    iron = cubic(
        K=10.0,
        n=5.0,
        families={"bcc110": {"tau0": 400.0}},
        cleavage={
            "planes": "cube",
            "strength": 500.0,
            "softening": 100.0,
            "K": 5.0,
            "n": 1.0,
            "friction": 0.3,
        },
    )
    magnesium = _magnesium(k=2.0, n=5.0, hardening=10.0, twin_reorientation=0.6)
    cases = [
        (iron, np.array([5.0, 10.0, 20.0]),
         {"times": [0, 1, 2, 3], "increments": 24, "strain": {"33": [0.012, -0.004, 0.006]}}),
        (magnesium, np.array([0.0, 8.0, 0.0]),
         {"times": [0, 1, 2], "increments": 40, "strain": {"33": [0.055, -0.012]}}),
    ]  # fmt: skip
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    grains = np.zeros(image.shape, int)
    for m, euler, path in cases:
        cp = m.crystal_plasticity
        ref = simulate_point(m, path, orientation=euler, kinematics=kinematics)
        assert ref.info["cuts"] == 0
        scale = np.abs(ref.stress).max()
        if cp.cleavage:
            opening = ref.series["cleavage_opening"]
            assert opening.max() > 0.005 and opening[48] == 0.0 and opening[-1] > 0.003
        else:
            angle = ref.series["twin_misorientation"]
            assert angle.max() == pytest.approx(86.31, abs=0.01) and angle[-1] == 0.0
        fft = NonlinearFFTProblem(image, {0: m}, grains=grains, orientations=euler[None],
                                  kinematics=kinematics)  # fmt: skip
        mesh = box_mesh((1, 1, 1), order=1, perturb=0.2)
        mesh = dataclasses.replace(
            mesh, cell=np.zeros(mesh.n_elements, int), orientations=euler[None]
        )
        fe = NonlinearFEProblem(mesh, {0: m}, "periodic", kinematics=kinematics)
        for problem in (fft, fe):
            res = problem.run(path, fields="steps")
            assert res.info["cuts"] == 0
            assert np.allclose(res.stress, ref.stress, rtol=0, atol=2e-5 * scale)
            assert set(res.series) == set(cp.field_names)
            for name in cp.field_names:
                assert np.allclose(res.series[name], ref.series[name], rtol=0, atol=1e-6)
                for k, time in enumerate(sorted(res.fields)):
                    values = res.fields[time][name]
                    index = round(time * len(ref.times[1:]) / ref.times[-1])
                    assert np.allclose(values, ref.series[name][index], rtol=0, atol=1e-6), (
                        name,
                        k,
                    )
            if cp.reorients and kinematics == "finite" and problem is fe:
                # the orientation of the elements at the end of the tension: the twin's
                turned = euler_to_matrix(res.fields[1.0]["orientation"])
                first = euler_to_matrix(euler)[0]
                off = disorientation(turned @ first.T, "hexagonal")
                assert np.allclose(off, 86.31, atol=2.0)  # with the rotation of the lattice


def _magnesium_cycle(**more):
    """The textured polycrystal of magnesium pulled by 4 % along the axes c of its grains,
    then brought back: its response, with ``more`` keys of its crystal plasticity."""
    image, grains, orientations, magnesium = textured_magnesium()
    spec = magnesium.to_config()
    spec["crystal_plasticity"] = {**spec["crystal_plasticity"], **more}
    problem = NonlinearFFTProblem(image, {1: Material.from_config(spec)}, grains=grains,
                                  orientations=orientations)  # fmt: skip
    path = {"times": [0, 1, 2], "increments": 16, "strain": {"33": [0.04, 0.0]}}
    return problem.run(path, fields=True)


def test_textured_magnesium_untwins():
    """A polycrystal of magnesium pulled along the axes c of its grains, then brought back:
    the twins that the tension made shear back under a pressure about that of the
    tension, and are nearly gone when the strain is."""
    res = _magnesium_cycle()
    stress, fraction = res.stress[:, 2], res.series["twin_fraction"]
    assert res.info["cuts"] == 0 and len(stress) == 33
    assert 0.3 < fraction[16] < 0.5 and fraction[-1] < 0.15 * fraction[16]
    assert np.all(np.diff(fraction[18:]) < 0)
    assert 0.8 * stress[16] < -stress[-1] < 1.05 * stress[16]
    field = res.fields[2.0]["twin_fraction"]
    assert np.nanmean(field) == pytest.approx(fraction[-1], rel=1e-6, abs=1e-12)


@pytest.mark.slow
def test_textured_magnesium_keeps_or_turns_its_twins():
    """The same polycrystal without detwinning: brought back, it keeps its twins and must
    slip on its hard systems, at nearly twice the pressure. With lattices that turn at
    half of their twinned fraction: nearly half of its points are twins at the end of the
    tension, 86 degrees off their grains, and harder to pull (their axes c lie across the
    tension); brought back, they twin back, and some are the grain again."""
    plain = _magnesium_cycle()
    kept = _magnesium_cycle(twin_detwinning=False)
    assert np.allclose(kept.stress[:17], plain.stress[:17], rtol=0, atol=1e-6 * 100.0)
    was = kept.series["twin_fraction"]
    assert np.all(np.abs(was[16:] - was[16]) < 0.01)
    assert -kept.stress[-1, 2] > 1.6 * -plain.stress[-1, 2]
    turning = _magnesium_cycle(twin_reorientation=0.5)
    assert set(turning.series) == {"twin_fraction", "twin_reoriented", "twin_misorientation"}
    angle = np.asarray(turning.fields[1.0]["twin_misorientation"])
    share = np.asarray(turning.fields[1.0]["twin_reoriented"])  # of the points of each voxel
    turned = np.nanmean(share)
    assert 0.3 < turned < 0.6 and np.nanmax(angle) == pytest.approx(86.31, abs=0.01)
    assert np.allclose(angle, 86.3119 * share, atol=1e-3)
    assert turning.series["twin_misorientation"][16] == pytest.approx(np.nanmean(angle), rel=1e-6)
    assert turning.series["twin_reoriented"][16] == pytest.approx(turned, rel=1e-6)
    # the turned volume is not the twinned volume of the grains, at one threshold for all
    grains = np.asarray(textured_magnesium()[1].data).ravel()
    twinned = np.asarray(turning.fields[1.0]["twin_fraction"]).ravel()
    excess = [share.ravel()[grains == g].mean() - twinned[grains == g].mean()
              for g in np.unique(grains)]  # fmt: skip
    assert min(excess) < -0.2 and max(excess) > 0.2
    assert turning.stress[16, 2] > 1.1 * plain.stress[16, 2]
    # brought back: detwinning, in the grains and in the twins
    assert turning.series["twin_fraction"][-1] < 0.6 * turning.series["twin_fraction"][16]
    assert turning.series["twin_misorientation"][-1] < turning.series["twin_misorientation"][16]
    assert -turning.stress[-1, 2] < 0.8 * -kept.stress[-1, 2]


@pytest.mark.slow
def test_cracked_polycrystal_closes():
    """A polycrystal of iron broken by cleavage through its grains, then pushed: its
    crack closes under a low pressure, and once closed the polycrystal carries the
    pressure nearly as the sound one. Without the closure the crack stays open: the
    lattice alone is pressed, as soon as the strain goes down, and breaks further."""
    from morphogen.domain import Domain
    from morphogen.models.base import model_from_config

    size, shape, energy, strength = 0.1, 8, 0.05, 500.0
    model = model_from_config({"type": "voronoi", "n_cells": 6, "crystal_orientations": True})
    micro = model.generate(Domain(size=size, periodic=True), 2)
    image = micro.voxelize(shape=shape)
    results = {}
    for closure, low, steps in ((True, -0.002, 24), (False, 0.004, 12)):
        iron = cubic(
            K=10.0,
            n=10.0,
            families={"bcc110": {"tau0": 300.0, "Q": 100.0, "b": 20.0}},
            cleavage={
                "planes": "cube",
                "strength": strength,
                "energy": energy,
                "length": size / shape,
                "K": 50.0,
                "n": 1.0,
                "closure": closure,
            },
        )
        problem = NonlinearFFTProblem(image, {1: iron}, grains=micro.labels(grid=image.grid()),
                                      orientations=micro.orientations)  # fmt: skip
        path = {"times": [0, 1, 2], "increments": [40, steps], "strain": {"33": [0.01, low]}}
        results[closure] = problem.run(path, fields=True)
    closing, open_ = results[True], results[False]
    at = lambda res, time: int(np.searchsorted(res.times, time - 1e-9))  # noqa: E731
    top = at(closing, 1.0)
    stress, strain = closing.stress[:, 2], closing.strain[:, 2]
    opening = closing.series["cleavage_opening"]
    young = stress[1] / strain[1]
    # broken: little stress left at the end of the tension, the same for both
    assert stress.max() > strength and stress[top] < 0.15 * stress.max()
    assert open_.stress[: at(open_, 1.0) + 1, 2].max() == pytest.approx(stress.max(), rel=0.01)
    # the crack closes under a low pressure: still open at 0.4 % of strain, closed at 0
    half = at(closing, 1.5)
    assert strain[half] == pytest.approx(0.004) and abs(stress[half]) < 0.1 * strength
    assert 0.3 * opening[top] < opening[half] < 0.6 * opening[top]
    assert opening[-1] < 0.02 * opening[top]
    # closed: the stress of the sound polycrystal, nearly (some grains slip)
    low = at(closing, 1.0 + 11.0 / 12.0)
    assert strain[low] == pytest.approx(-0.001)
    assert 0.85 * young * strain[low] > stress[low] > young * strain[low]
    assert stress[-1] < 0.8 * young * strain[-1]
    # without the closure: the opening stays, the lattice is pressed at once and yields
    kept = open_.series["cleavage_opening"]
    assert np.all(kept[at(open_, 1.0) :] > 0.99 * kept[at(open_, 1.0)])
    assert open_.strain[-1, 2] == pytest.approx(0.004) and open_.stress[-1, 2] < -strength
    assert open_.series["cleavage_damage"][-1] > 1.1 * closing.series["cleavage_damage"][half]


# -- a resistance of their own on the way back -------------------------------------------------


def test_twins_untwin_at_a_share_of_their_resistance():
    """Twins that shear back at a share of their resistance (no nucleation on the way
    back): a crystal of magnesium whose twins harden, pulled along its axis c then
    pushed, twins by its flow rule, ``tau = r + wall + K rate^(1/n)``, and untwins by
    ``-tau = share r + K rate^(1/n)``, with ``r = tau_0 + Q sum_t (1 - exp(-b v_t))`` the
    resistance hardened by the net shears of its six twins; its twinned fraction is the
    strain of the twins over their shear, there and back."""
    share, tau0, q, b, k, n = 0.4, 15.0, 10.0, 40.0, 0.05, 20.0
    spec = {"twins": {"twin1012": {"tau0": tau0, "Q": q, "b": b}}, "c_over_a": RATIO_MG, "K": k,
            "n": n, "families": {"pyramidal2_ca": {"tau0": 90.0}}}  # fmt: skip
    m = hexagonal(MAGNESIUM, **spec, twin_detwinning=share)
    cp = m.crystal_plasticity
    assert cp.twin_detwinning == share and np.all(cp.kinds_array()[cp.n_slip :, 4] == share)
    assert cp.to_config()["twin_detwinning"] == share
    assert CrystalPlasticity.from_config(cp.to_config()) == cp
    twin = _schmid(cp, np.array([0.0, 0.0, 1.0]))[cp.n_slip]
    shear = twin_systems("twin1012", RATIO_MG)[2]
    steps = 100
    path = {"times": [0, 1, 2], "increments": steps, "strain": {"33": [0.03, -0.01]}}
    res = simulate_point(m, path)
    assert res.info["cuts"] == 0
    stress, strain, fraction = res.stress[:, 2], res.strain[:, 2], res.series["twin_fraction"]
    undone = int(np.flatnonzero((fraction == 0.0) & (np.arange(len(fraction)) > steps))[0])
    plastic = strain - stress / modulus(m)
    assert np.allclose(fraction[:undone], plastic[:undone] / (shear * twin), rtol=0, atol=1e-9)
    assert fraction[steps] > 0.4
    # the six twins shear alike (they harden): the net shear of each, its rate over a step
    v = fraction * shear / 6
    resistance = tau0 + q * 6 * (1.0 - np.exp(-b * v))
    viscous = k * (np.abs(np.diff(v)) * steps) ** (1.0 / n)
    wall = cr._BARRIER * tau0 * fraction / (1.0 - fraction)
    tau = stress * twin
    there = np.arange(20, steps)
    assert np.allclose(tau[there + 1], (resistance + wall)[there + 1] + viscous[there], rtol=1e-9)
    back = np.flatnonzero(np.diff(fraction[: undone + 1]) < 0)[1:-1]
    assert len(back) > 60
    assert np.allclose(-tau[back + 1], share * resistance[back + 1] + viscous[back], rtol=1e-9)
    # at less than half of the stress of the twinning: its share, and the hardening undone
    assert -stress[back + 1].min() < 0.45 * stress[steps]
    # untwinned: as it was
    assert np.all(fraction[undone:] == 0.0) and abs(plastic[undone]) < 1e-9
    # a share of nothing: the twins that stay
    kept = hexagonal(MAGNESIUM, **spec, twin_detwinning=0.0).crystal_plasticity
    assert kept == hexagonal(MAGNESIUM, **spec, twin_detwinning=False).crystal_plasticity
    for bad in (-0.5, "half", math.inf):
        with pytest.raises(ValueError, match="twin_detwinning: true, false, or the share"):
            CrystalPlasticity.from_config({**spec, "twin_detwinning": bad})


def test_cracks_close_at_a_share_of_their_strength():
    """A cleavage plane that closes at a share of its strength: unloaded, it keeps its
    opening down to that share of the pressure, closes by ``sigma = -(share Y_0 exp(-w d)
    + K |d delta / dt|)``, carries the pressure as the sound crystal once closed, and
    opens again at the whole of the strength that its damage left."""
    strength, softening, k, share = 500.0, 100.0, 0.5, 0.2
    planes = {"planes": "cube", "strength": strength, "softening": softening, "K": k, "n": 1.0,
              "closure": share}  # fmt: skip
    m = cubic(cleavage=planes)
    cp = m.crystal_plasticity
    kinds = cp.kinds_array()
    assert cp.cleavage_closure == share and np.all(kinds[kinds[:, 0] == cr.OPENING, 4] == share)
    assert cp.to_config()["cleavage"]["closure"] == share
    assert CrystalPlasticity.from_config(cp.to_config()) == cp
    young = modulus(m)
    steps = 80
    path = {"times": [0, 1, 2, 3], "increments": steps, "strain": {"33": [0.008, -0.004, 0.008]}}
    res = simulate_point(m, path)
    assert res.info["cuts"] == 0
    stress, strain = res.stress[:, 2], res.strain[:, 2]
    opening, damage = res.series["cleavage_opening"], res.series["cleavage_damage"]
    moved = np.diff(opening)
    rate = np.abs(moved) * steps
    closing = np.flatnonzero(moved[steps:] < 0) + steps
    closed = np.flatnonzero(opening[steps:] == 0.0)[0] + steps
    # unloaded, the plane keeps its opening down to the share of its softened strength
    softened = strength * (1.0 - damage[steps])
    one = young * (strain[steps] - strain[steps + 1])
    assert -share * softened < stress[closing[0]] < -share * softened + one
    inside = closing[:-1]
    flow = strength * (1.0 - damage[1:])
    assert len(inside) > 20
    assert np.allclose(stress[inside + 1], -(share * flow + k * rate)[inside], rtol=1e-7,
                       atol=1e-7 * strength)  # fmt: skip
    # the same plane closing at its whole strength: later, under five times the pressure
    whole = simulate_point(cubic(cleavage={**planes, "closure": True}), path)
    first = np.flatnonzero(np.diff(whole.series["cleavage_opening"])[steps:] < 0)[0] + steps
    assert first > closing[0] and whole.stress[first, 2] < 4.0 * stress[closing[0]]
    # closed: the sound crystal; pulled again: the whole strength that the damage left
    assert stress[2 * steps] == pytest.approx(-0.004 * young, rel=1e-9)
    again = np.flatnonzero(opening[closed:] > 0)[0] + closed
    after = np.arange(again - 1, 3 * steps)
    assert np.allclose(stress[after + 1], (flow + k * rate)[after], rtol=1e-7, atol=1e-6 * strength)
    with pytest.raises(ValueError, match=r"cleavage\.closure: true, false, or the share"):
        CrystalPlasticity.from_config({"cleavage": {**planes, "closure": -1.0}})


@pytest.mark.parametrize("finite", [False, True])
def test_tangent_at_a_share_of_the_resistance(finite):
    """The tangent of the implicit update against finite differences for systems that go
    back at a share of their resistance: twins that harden with their net shear, a
    cleavage plane that softens as it closes."""
    magnesium = hexagonal(
        MAGNESIUM,
        c_over_a=RATIO_MG,
        K=5.0,
        n=4.0,
        families={"basal": {"tau0": 4.0, "Q": 3.0, "b": 20.0}, "pyramidal2_ca": {"tau0": 40.0}},
        twins={"twin1012": {"tau0": 12.0, "Q": 6.0, "b": 15.0, "C": 200.0, "D": 10.0}},
        twin_detwinning=0.4,
    )
    cp = magnesium.crystal_plasticity
    euler = (15.0, 25.0, 40.0)
    table, beh = _kernel(magnesium, euler, finite)
    ns = cp.n_systems
    shear = twin_systems("twin1012", RATIO_MG)[2]
    twins = slice(cr.CP_ALPHA + ns + cp.n_slip, cr.CP_ALPHA + ns + cp.n_slip + cp.n_twins)
    old = np.zeros((1, beh.width))
    old[0, twins] = 0.97 * shear / 6
    grad = _gradient([[1.2e-3, 5e-4, 8e-4], [5e-4, 0.8e-3, 6e-4], [8e-4, 6e-4, -2.5e-3]], euler,
                     finite)  # fmt: skip
    s, a, new = _point(table, beh, grad, old, 0.5, finite)
    assert new[0, twins].min() < 0.9 * old[0, twins].min()  # it untwins
    fd = _differences(table, beh, grad, old, 0.5, finite)
    assert np.allclose(a, fd, rtol=0, atol=2e-6 * np.abs(fd).max())
    # more than at the whole of its resistance
    table1, beh1 = _kernel(
        dataclasses.replace(
            magnesium, crystal_plasticity=dataclasses.replace(cp, twin_detwinning=True)
        ),
        euler,
        finite,
    )
    assert new[0, twins].sum() < _point(table1, beh1, grad, old, 0.5, finite)[2][0, twins].sum()
    families = {"pyramidal2_ca": {"tau0": 60.0}, "basal": {"tau0": 10.0, "Q": 8.0, "b": 20.0}}
    planes = {"planes": "basal", "strength": 90.0, "softening": 150.0, "shear": 0.15, "K": 30.0,
              "n": 2.0, "closure": 0.3}  # fmt: skip
    zinc = hexagonal(ZINC, c_over_a=RATIO_ZN, K=20.0, n=4.0, families=families, cleavage=planes)
    cp = zinc.crystal_plasticity
    euler = (20.0, 35.0, 50.0)
    table, beh = _kernel(zinc, euler, finite)
    ns, first = cp.n_systems, cp.n_slip
    old = np.zeros((1, beh.width))
    old[0, cr.CP_ALPHA + first] = old[0, cr.CP_ALPHA + ns + first] = 0.004
    grad = _gradient([[1e-3, 5e-4, 8e-4], [5e-4, -5e-4, 6e-4], [8e-4, 6e-4, -1.2e-3]], euler,
                     finite)  # fmt: skip
    s, a, new = _point(table, beh, grad, old, 0.5, finite)
    assert 3e-3 < new[0, cr.CP_ALPHA + first] < 3.6e-3  # it closes, under a low pressure
    fd = _differences(table, beh, grad, old, 0.5, finite)
    assert np.allclose(a, fd, rtol=0, atol=2e-6 * np.abs(fd).max())
    s1, _, new1 = _point(table, beh, grad, old, 0.5, finite, guess=1, new=new)
    assert np.allclose(new1, new, rtol=0, atol=1e-11) and np.allclose(s1, s, atol=1e-7)


# -- lattices turned grain by grain ------------------------------------------------------------


def test_grain_by_grain_in_a_configuration():
    spec = {"twins": {"twin1012": {"tau0": 15.0}}, "c_over_a": RATIO_MG, "K": 1.0, "n": 5.0}
    turn = {"threshold": 0.2, "growth": 0.3}
    cp = CrystalPlasticity.from_config({**spec, "twin_reorientation": turn})
    assert cp.reorients and cp.by_grain
    assert cp.twin_reorientation == 0.2 and cp.twin_reorientation_growth == 0.3
    assert cp.to_config()["twin_reorientation"] == turn
    assert CrystalPlasticity.from_config(cp.to_config()) == cp
    shear = twin_systems("twin1012", RATIO_MG)[2]
    kinds = cp.kinds_array()
    assert kinds.shape == (6, cr.CP_WIDTH - cr.CP_KIND)
    # no threshold for the kernels (the lattices are turned after the steps); the twinned
    # fraction per unit of shear
    assert np.all(kinds[:, 5] == -1.0) and np.allclose(kinds[:, 8], 1.0 / shear)
    alone = CrystalPlasticity.from_config({**spec, "twin_reorientation": {"threshold": 0.2}})
    assert alone.by_grain and alone.twin_reorientation_growth == 0.0
    point = CrystalPlasticity.from_config({**spec, "twin_reorientation": 0.2})
    assert not point.by_grain and point.twin_reorientation_growth is None
    assert np.allclose(point.kinds_array()[:, 5], 1.0 / (0.2 * shear))
    assert cp.field_names == point.field_names == cr.FIELDS[:3]
    for bad, match in [
        ({"growth": 0.3}, "a twinned fraction, or"),
        ({"threshold": 0.2, "slope": 0.3}, "a twinned fraction, or"),
        ({"threshold": 0.2, "growth": -0.1}, "a growth of 0 or more"),
        ({"threshold": 1.2, "growth": 0.1}, "in \\]0, twin_saturation"),
        ({"threshold": 0.2, "growth": "fast"}, "are numbers"),
    ]:
        with pytest.raises(ValueError, match=match):
            CrystalPlasticity.from_config({**spec, "twin_reorientation": bad})
    with pytest.raises(ValueError, match="that of a twin_reorientation threshold"):
        CrystalPlasticity((), twins=cp.twins, c_over_a=RATIO_MG, twin_reorientation_growth=0.2)


def _twinned_points(cp, fractions, predominant):
    """Internal variables of points of a crystal of twins alone whose twinned fractions
    are ``fractions``, 60 % of each on its ``predominant`` twin and the rest on the next;
    and those of the start of a step over which they grew by a tenth."""
    n = cp.n_systems
    shear = 1.0 / cp.kinds_array()[0, 8]
    state = np.zeros((len(fractions), cr.state_width(n, False, True)))
    a = cr.CP_ALPHA + n
    for i, (fraction, twin) in enumerate(zip(fractions, predominant, strict=True)):
        state[i, a + twin] = 0.6 * fraction * shear
        state[i, a + (twin + 1) % n] = 0.4 * fraction * shear
        state[i, 6] = 0.01 * (i + 1)  # their cumulated plastic strain: left as it is
    return state, 0.9 * state


def test_predominant_twin_reorientation_of_a_grain():
    """The rule that turns the lattices of a grain (Tomé, Lebensohn & Kocks 1991) on ten
    points of given twinned fractions: the most twinned points take the lattice of their
    predominant twin, past the threshold, while the turned volume is below the twinned
    volume of the grain; the threshold grows with their ratio; the lattices come back,
    the most undone first, while the turned volume is above the twinned one, and where
    the twin is undone altogether."""
    spec = {"twins": {"twin1012": {"tau0": 15.0}}, "c_over_a": RATIO_MG, "K": 1.0, "n": 5.0}
    cp = CrystalPlasticity.from_config({**spec, "twin_reorientation": {"threshold": 0.2}})
    n = cp.n_systems
    normals, directions = cp.systems()
    systems = np.concatenate([directions, normals], axis=1)[None]  # one grain, in its axes
    shear = twin_systems("twin1012", RATIO_MG)[2]
    fractions = 0.06 + 0.05 * np.arange(10)  # 0.06 ... 0.51: a mean of 0.285
    predominant = np.arange(10) % n
    grains = np.zeros(10, dtype=int)
    rot = cr.state_width(n)
    a = cr.CP_ALPHA + n
    state, before = _twinned_points(cp, fractions, predominant)
    first = state.copy()
    # the three most twinned points: 0.1, 0.2, then 0.3 of the grain, past its 0.285
    assert cr.turn_twins(cp, state, before, grains, systems) == 3
    assert np.array_equal(state[:7], first[:7])
    for i in (7, 8, 9):
        normal = normals[predominant[i]]
        q = 2.0 * np.outer(normal, normal) - np.eye(3)
        assert np.allclose(state[i, rot : rot + 9].reshape(3, 3) + np.eye(3), q, atol=1e-15)
        assert state[i, rot + 9] == predominant[i] + 1  # the twin that brings it back
        assert state[i, rot + 10] == pytest.approx(fractions[i], rel=1e-12)
        assert np.all(state[i, cr.CP_ALPHA : a + n] == 0.0)  # its twins start again
        assert state[i, 6] == first[i, 6]
    fields = cp.state_fields(state)
    assert np.allclose(fields["twin_fraction"], fractions, rtol=1e-12)  # a turn keeps it
    assert fields["twin_reoriented"].tolist() == [0.0] * 7 + [1.0] * 3
    assert np.allclose(fields["twin_misorientation"][7:], 86.31, atol=0.01)
    # nothing more while the turned volume is not below the twinned one, nor of twins
    # that did not grow over the step
    turned = state.copy()
    assert cr.turn_twins(cp, state, before, grains, systems) == 0
    again, _ = _twinned_points(cp, fractions, predominant)
    assert cr.turn_twins(cp, again, again, grains, systems) == 0
    # by their volumes: the most twinned point is a quarter of the grain
    heavy, start = _twinned_points(cp, fractions, predominant)
    weights = np.array([1.0] * 9 + [3.0])
    assert cr.turn_twins(cp, heavy, start, grains, systems, weights=weights) == 2
    assert np.flatnonzero(cp.state_fields(heavy)["twin_reoriented"]).tolist() == [8, 9]
    # the points of another crystal, or of another grain, are not those of this one
    rows = np.array([1, 3, 5, 7, 9])
    part, start = _twinned_points(cp, fractions, predominant)
    assert cr.turn_twins(cp, part, start, grains[rows], systems, points=rows) == 2  # 0.31
    assert np.flatnonzero(cp.state_fields(part)["twin_reoriented"]).tolist() == [7, 9]
    two, start = _twinned_points(cp, fractions, predominant)
    halves = (np.arange(10) >= 5).astype(int)
    # (0.2 of the first half for the 0.16 that it has twinned; 0.6 of the second for 0.41)
    assert cr.turn_twins(cp, two, start, halves, np.repeat(systems, 2, axis=0)) == 4
    assert np.flatnonzero(cp.state_fields(two)["twin_reoriented"]).tolist() == [4, 7, 8, 9]
    # the twins of the grain grow by half: its twinned volume is 0.3585 for 0.3 turned;
    # one more lattice is turned at a constant threshold, none at one that has grown to
    # 0.2 + 0.5 x 0.3 / 0.3585 = 0.62
    more = turned.copy()
    more[:7, a : a + n] *= 1.5
    growing = dataclasses.replace(cp, twin_reorientation_growth=0.5)
    assert cr.turn_twins(growing, more.copy(), turned, grains, systems) == 0
    assert cr.turn_twins(cp, more, turned, grains, systems) == 1
    assert np.flatnonzero(cp.state_fields(more)["twin_reoriented"]).tolist() == [6, 7, 8, 9]
    # the twins shear back: by half in the grain; in the turned lattices, the twin that
    # brings the first one back has undone 0.45 (more than the 0.41 that the point had
    # twinned), 0.25 and 0.1 of them
    back = turned.copy()
    back[:7, a : a + n] *= 0.5
    for i, undone in ((7, 0.45), (8, 0.25), (9, 0.1)):
        back[i, a + predominant[i]] = undone * shear
    start = back.copy()
    start[7:, a : a + n] *= 0.9
    fields = cp.state_fields(back)
    assert np.allclose(fields["twin_fraction"][7:], [0.0, 0.21, 0.41], atol=1e-12)
    assert fields["twin_fraction"].mean() == pytest.approx(0.1355, rel=1e-9)
    # the first comes back (its twin is undone), the second too (0.2 turned for 0.1355
    # twinned, its twin undone by more than the threshold), with what is left of its twin
    # on the system that had turned it; the third stays (0.1 turned)
    was = back.copy()
    assert cr.turn_twins(cp, back, start, grains, systems) == 2
    assert np.all(back[7:9, rot:] == 0.0) and np.array_equal(back[9], was[9])
    assert np.array_equal(back[:7], was[:7])
    assert np.all(back[7, cr.CP_ALPHA : a + n] == 0.0)
    left = np.zeros(n)
    left[predominant[8]] = 0.21 * shear
    assert np.allclose(back[8, a : a + n], left, rtol=1e-12, atol=1e-15)
    after = cp.state_fields(back)
    assert np.allclose(after["twin_fraction"], fields["twin_fraction"], atol=1e-12)
    assert after["twin_reoriented"].tolist() == [0.0] * 9 + [1.0]
    # a crystal whose lattices turn point by point is left to its kernels
    point = CrystalPlasticity.from_config({**spec, "twin_reorientation": 0.2})
    state, before = _twinned_points(point, fractions, predominant)
    assert cr.turn_twins(point, state, before, grains, systems) == 0


def test_lattices_turned_grain_by_grain():
    """A material point is a grain of one point: its lattice turns, and comes back, as
    at a constant threshold. In a grain of 64 voxels pulled 8 degrees off its axis c the
    lattices of its integration points are turned one by one as the grain twins: the
    turned volume is the twinned volume of the grain, to a point, where a constant
    threshold turns them all at once; brought back, the lattices come back as the twins
    are undone."""
    euler = np.array([0.0, 8.0, 0.0])
    path = {"times": [0, 1, 2], "increments": 40, "strain": {"33": [0.055, -0.012]}}

    def magnesium(turn):
        return _magnesium(k=2.0, n=5.0, hardening=10.0, twin_reorientation=turn)

    point = simulate_point(magnesium(0.3), path, orientation=euler)
    grain = simulate_point(magnesium({"threshold": 0.3, "growth": 0.5}), path, orientation=euler)
    assert point.info["cuts"] == 0 and grain.info["cuts"] == 0
    assert np.allclose(grain.stress, point.stress, rtol=0, atol=1e-6 * np.abs(point.stress).max())
    for name in ("twin_fraction", "twin_reoriented", "twin_misorientation"):
        assert np.allclose(grain.series[name], point.series[name], rtol=0, atol=1e-9)
    assert point.series["twin_reoriented"].max() == 1.0 and point.series["twin_reoriented"][-1] == 0
    image = VoxelImage(np.zeros((4, 4, 4), dtype=np.uint8))
    grains = np.zeros(image.shape, int)
    threshold = 0.1
    runs = {}
    for name, turn in (("grain", {"threshold": threshold}), ("point", threshold)):
        problem = NonlinearFFTProblem(image, {0: magnesium(turn)}, grains=grains,
                                      orientations=euler[None])  # fmt: skip
        runs[name] = problem.run(path, fields=True)
        assert runs[name].info["cuts"] == 0
    twinned, turned = (runs["grain"].series[k] for k in ("twin_fraction", "twin_reoriented"))
    pulled = np.arange(41)
    past = pulled[twinned[pulled] >= threshold]
    assert past[0] < 8 and np.all(turned[: past[0]] == 0.0)
    excess = (turned - twinned)[past]
    points = 8 * 64  # the integration points of the voxels
    assert np.all(excess >= -1e-12) and np.all(excess < 1.0 / points + 1e-12)
    assert 0.5 < twinned[40] < 0.6
    # the fields of the voxels: the share of their points whose lattice is turned, each
    # 86.3 degrees off
    field = runs["grain"].fields[1.0]
    share = np.asarray(field["twin_reoriented"])
    assert np.allclose(8 * share, np.round(8 * share), atol=1e-12) and 0.0 < share.min() < 1.0
    assert share.mean() == pytest.approx(turned[40], abs=1e-12)
    assert np.allclose(field["twin_misorientation"], 86.3119 * share, atol=1e-3)
    # the orientation of a voxel is that of most of its points: of the grain, or of the twin
    now = euler_to_matrix(np.asarray(field["orientation"]).reshape(-1, 3))
    off = disorientation(now @ euler_to_matrix(euler)[0].T, "hexagonal")
    assert np.allclose(off[share.ravel() > 0.5], 86.31, atol=0.01)
    assert np.allclose(off[share.ravel() <= 0.5], 0.0, atol=1e-6)
    # point by point: every voxel at once, six times the twinned volume and more
    assert runs["point"].series["twin_reoriented"][40] == 1.0
    assert runs["point"].series["twin_fraction"][40] < 0.15
    # brought back: the lattices come back with the twins
    assert np.abs(turned - twinned)[41:].max() < 0.08
    assert turned[-1] == 0.0 and twinned[-1] < 0.01
    assert np.all(np.diff(twinned[42:]) < 1e-12)


def test_turned_volume_follows_the_twinned_volume_of_the_grains():
    """A textured polycrystal of magnesium pulled along the axes c of its grains, its
    lattices turned grain by grain: at the end of the tension the turned volume of each
    grain is its twinned volume, to an integration point (an eighth of a voxel); it
    hardens as its lattices turn; brought back, its twins shear back and its lattices
    come back."""
    res = _magnesium_cycle(twin_reorientation={"threshold": 0.2})
    assert res.info["cuts"] == 0
    assert set(res.series) == {"twin_fraction", "twin_reoriented", "twin_misorientation"}
    grains = np.asarray(textured_magnesium()[1].data).ravel()
    twinned, turned = res.series["twin_fraction"], res.series["twin_reoriented"]
    assert 0.3 < twinned[16] < 0.42 and turned[16] == pytest.approx(twinned[16], abs=0.01)
    pulled, back = res.fields[1.0], res.fields[2.0]
    for g in np.unique(grains):
        own = grains == g
        excess = np.mean(np.ravel(pulled["twin_reoriented"])[own]) - np.mean(
            np.ravel(pulled["twin_fraction"])[own]
        )
        assert -1e-12 < excess * 8 * own.sum() < 1.0 + 1e-9, g  # a point above at most
        left = np.mean(np.ravel(back["twin_reoriented"])[own]) - np.mean(
            np.ravel(back["twin_fraction"])[own]
        )
        assert abs(left) < 0.1, g
    plain = _magnesium_cycle()
    assert np.allclose(res.stress[:4], plain.stress[:4], rtol=0, atol=1e-6 * 100.0)  # no twin yet
    assert res.stress[16, 2] > 1.25 * plain.stress[16, 2]  # the twins are hard along the load
    assert twinned[-1] < 0.25 * twinned[16] and turned[-1] < 0.3 * turned[16]


def test_cleavage_planes_of_a_turned_lattice():
    """The criterion of Beremin on the basal plane of a crystal of magnesium pulled 8
    degrees off its axis c, whose twins turn its lattice: the normal stress of the plane
    is ``sigma cos^2`` of the angle of the tension to the axis c of the lattice as it is,
    that of the grain then that of its twin, nearly across the tension. The fields of the
    path have the orientation of the lattices, in small strains too."""
    from morphogen.homogenization.beremin import path_beremin

    euler = np.array([0.0, 8.0, 0.0])
    m = _magnesium(k=2.0, n=5.0, hardening=10.0, twin_reorientation=0.3)
    cp = m.crystal_plasticity
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    problem = NonlinearFFTProblem(image, {0: m}, grains=np.zeros(image.shape, int),
                                  orientations=euler[None])  # fmt: skip
    path = {"times": [0, 1], "increments": 20, "strain": {"33": [0.03]}}
    res = problem.run(path, fields="steps")
    angle = res.series["twin_misorientation"]
    turned = int(np.flatnonzero(angle > 0)[0])
    assert 5 < turned < 18
    g = euler_to_matrix(euler)[0]
    axis = g.T @ np.array([0.0, 0.0, 1.0])  # the axis c of the grain, in the sample
    normals, _ = cp.systems()
    shares = res.fields[res.times[turned - 1]]["twin_fraction"]  # (the twins of the grain)
    assert np.all(np.asarray(shares) > 0.2)
    law = {"modulus": 20.0, "scale": 100.0, "volume": 8.0, "planes": "basal",
           "c_over_a": RATIO_MG}  # fmt: skip
    found = path_beremin(res, law, orientations=np.broadcast_to(euler, (8, 3)))
    stress = res.stress[1:, 2]
    # before: the plane of the grain, 8 degrees off the tension
    assert np.allclose(found.weibull[: turned - 1], axis[2] ** 2 * stress[: turned - 1], rtol=1e-6)
    # after: that of the twin, turned by half a turn about the normal of the twin plane
    now = euler_to_matrix(np.asarray(res.fields[1.0]["orientation"]).reshape(-1, 3))
    after = np.einsum("pji,j->pi", now, [0.0, 0.0, 1.0])  # the axes c of the voxels
    assert np.allclose(after, after[0], atol=1e-9)
    candidates = [(2.0 * np.outer(n, n) - np.eye(3)) @ axis for n in normals[cp.n_slip :] @ g]
    assert min(np.abs(c - after[0]).max() for c in candidates) < 1e-9
    assert math.degrees(math.acos(abs(after[0] @ axis))) == pytest.approx(86.31, abs=0.01)
    # (from the end of the step that turned it: the fields are those of the ends of the steps)
    late = slice(turned - 1, None)
    # (a plane nearly along the tension: its normal stress has the round-off of the others)
    assert np.allclose(found.weibull[late], after[0, 2] ** 2 * stress[late], rtol=1e-6, atol=1e-5)
    assert after[0, 2] ** 2 < 1e-3 and found.weibull[-1] < 1e-3 * stress[-1]
    # the largest Weibull stress of the path: that of the grain before its lattice turned
    assert found.largest == pytest.approx(axis[2] ** 2 * stress[turned - 2], rel=1e-6)
    # without the orientations of the lattices at each time: the planes of the grains
    for fields in res.fields.values():
        del fields["orientation"]
    fixed = path_beremin(res, law, orientations=np.broadcast_to(euler, (8, 3)))
    assert np.allclose(fixed.weibull, axis[2] ** 2 * stress, rtol=1e-6)
