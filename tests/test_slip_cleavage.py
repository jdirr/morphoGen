"""Cleavage that the slip nucleates: the strength of the cleavage planes of a crystal falls
with its cumulated slip, and their damage eases its slip (the coupling of plasticity and
damage of Aslan, Cordero, Gaubert & Forest 2011). The flow rules of the two along a
tension and a compression, the tangents, cells against the point, a polycrystal."""

import dataclasses
import math

import numpy as np
import pytest

from helpers import box_mesh
from morphogen.homogenization import crystal as cr
from morphogen.homogenization.crystal import CrystalPlasticity
from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem
from morphogen.homogenization.nonlinear import NonlinearFEProblem, simulate_point
from morphogen.image import VoxelImage
from test_closure_reorientation import _differences, _gradient
from test_twinning_cleavage import _kernel, _point, cubic, modulus

TAU0, K, N = 100.0, 0.5, 10.0
STRENGTH, SOFTENING, K_PLANES, COUPLING = 600.0, 150.0, 0.5, 8.0
SCHMID = 1.0 / math.sqrt(6.0)  # of the eight {110}<111> systems that a tension along [001] loads


def iron(coupling=COUPLING, **more):
    """A cubic crystal that slips at 100 MPa and cleaves on its {100} planes at 600 MPa:
    pulled along [001] it yields at 245 MPa, far below the strength of its planes."""
    planes = {"planes": "cube", "strength": STRENGTH, "softening": SOFTENING, "K": K_PLANES,
              "n": 1.0, "coupling": coupling, **more}  # fmt: skip
    return cubic(K=K, n=N, families={"bcc110": {"tau0": TAU0}}, cleavage=planes)


def _history(res, material):
    """The cumulated slip of the crystal and the cumulated opening of its (001) plane
    along a path along [001], from its strain: that of the lattice, of the opening and
    of the eight slip systems, which slip alike."""
    stress, strain = res.stress[:, 2], res.strain[:, 2]
    opening = res.series["cleavage_opening"]
    slip = strain - stress / modulus(material) - opening
    gamma = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(slip)))]) / SCHMID
    damage = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(opening)))])
    return gamma, damage


def test_slip_lowers_the_cleavage_strength():
    """A crystal pulled along [001] slips at its yield stress, far below the strength of
    its (001) plane, and that strength falls with its cumulated slip, ``Y_0 exp(-Theta
    gamma_cum)``: the plane opens when it has come down to the stress, at ``gamma_cum =
    ln(Y_0 / sigma) / Theta``, and the crystal breaks. The slips follow their flow rule
    up to there, the opening follows its own, ``sigma = Y_0 exp(-w d - Theta gamma_cum) + K
    d delta / dt``, from there. Without the coupling the slip shields the cleavage."""
    m = iron()
    cp = m.crystal_plasticity
    assert cp.cleavage_coupling == COUPLING
    steps = 200
    path = {"times": [0, 1], "increments": steps, "strain": {"33": [0.08]}}
    res = simulate_point(m, path)
    stress = res.stress[:, 2]
    opening = res.series["cleavage_opening"]
    gamma, damage = _history(res, m)
    dt = np.diff(res.times)
    slip_rate = np.diff(gamma) / 8 / dt  # of each of the eight systems
    opening_rate = np.diff(opening) / dt
    # it yields, and slips at its yield stress
    flow = TAU0 / SCHMID
    slipping = np.flatnonzero(slip_rate > 1e-9)
    assert len(slipping) > 100
    viscous = K * slip_rate[slipping] ** (1.0 / N)
    assert np.allclose(stress[slipping + 1] * SCHMID, TAU0 + viscous, rtol=1e-6)
    assert np.all(stress[slipping[5:] + 1] < 1.01 * flow) and stress.max() < 1.01 * flow
    # its plane opens when its strength has fallen to the stress
    first = int(np.flatnonzero(opening > 0)[0])
    assert np.all(opening[:first] == 0.0)
    onset = math.log(STRENGTH / stress[first - 1]) / COUPLING
    assert gamma[first - 1] == pytest.approx(onset, abs=2.0 * np.diff(gamma)[first - 3])
    assert gamma[first - 1] == pytest.approx(0.111, abs=0.002)
    # and from there by its flow rule, the slip at rest
    opened = np.flatnonzero(opening_rate > 1e-9)
    assert len(opened) > 50 and opened[0] == first - 1
    strength = STRENGTH * np.exp(-SOFTENING * damage - COUPLING * gamma)
    assert np.allclose(stress[opened + 1], strength[opened + 1] + K_PLANES * opening_rate[opened],
                       rtol=1e-6, atol=1e-6 * STRENGTH)  # fmt: skip
    assert np.all(slip_rate[opened[2:]] < 1e-9)
    assert res.series["cleavage_damage"][-1] > 0.99 and stress[-1] < 0.01 * flow
    # the damage of the field is that of the opening: 1 - exp(-w d)
    assert np.allclose(res.series["cleavage_damage"], -np.expm1(-SOFTENING * damage), atol=1e-9)
    # without the coupling: the crystal slips on, its plane does not open
    plain = simulate_point(iron(0.0), path)
    assert plain.series["cleavage_opening"].max() == 0.0
    assert plain.stress[-1, 2] == pytest.approx(stress[first - 1], rel=1e-3)
    assert np.allclose(plain.stress[: first - 2], res.stress[: first - 2], rtol=0, atol=1e-6)


def test_damage_eases_the_slip():
    """The same crystal pulled until its plane is half broken, then pushed: the plane
    closes, and the crystal slips in compression at a resistance that the damage of the
    plane has lowered, ``-sigma m = tau_0 - Theta exp(-Theta gamma_cum) (Y_0 / w) (1 -
    exp(-w d)) + K rate^(1/n)`` — the energy that the planes have dissipated and the
    strength that the slip takes off them are those of one potential."""
    m = iron()
    path = {"times": [0, 1, 2], "increments": [200, 100], "strain": {"33": [0.0505, 0.035]}}
    res = simulate_point(m, path)
    stress = res.stress[:, 2]
    opening = res.series["cleavage_opening"]
    gamma, damage = _history(res, m)
    top = int(np.searchsorted(res.times, 1.0 - 1e-9))
    assert 0.3 < res.series["cleavage_damage"][top] < 0.7 and opening[top] > 0.002
    released = (
        COUPLING * np.exp(-COUPLING * gamma) * STRENGTH * -np.expm1(-SOFTENING * damage) / SOFTENING
    )
    rate = np.diff(gamma) / 8 / np.diff(res.times)
    pushed = np.flatnonzero((rate > 1e-7) & (np.arange(len(rate)) >= top))
    assert len(pushed) > 30 and np.all(opening[pushed + 1] == 0.0)  # the plane is closed
    resistance = TAU0 - released[pushed + 1] + K * rate[pushed] ** (1.0 / N)
    assert np.allclose(-stress[pushed + 1] * SCHMID, resistance, rtol=1e-7)
    # 9 MPa off the 100 MPa of the sound crystal, then 8 as it slips on (the strength
    # that the slip takes off the planes is as much less to release): 226 MPa for 245
    assert 7.0 < released[-1] < released[pushed[0] + 1] < 10.0
    assert -stress[-1] == pytest.approx((TAU0 - released[-1]) / SCHMID, rel=0.01)
    assert -stress[-1] < 0.93 * TAU0 / SCHMID
    # without the coupling the plane never opened, and the crystal yields as it did
    plain = simulate_point(iron(0.0), path)
    assert -plain.stress[-1, 2] == pytest.approx(TAU0 / SCHMID, rel=0.01)


def test_coupling_in_a_configuration():
    spec = {"planes": "cube", "strength": 600.0, "softening": 150.0, "K": 1.0, "n": 2.0}
    slip = {"families": {"bcc110": {"tau0": 100.0}}, "K": 1.0, "n": 5.0}
    cp = CrystalPlasticity.from_config({**slip, "cleavage": {**spec, "coupling": 8.0}})
    assert cp.cleavage_coupling == 8.0 and cp.to_config()["cleavage"]["coupling"] == 8.0
    assert CrystalPlasticity.from_config(cp.to_config()) == cp
    kinds = cp.kinds_array()
    assert kinds.shape == (cp.n_systems, cr.CP_WIDTH - cr.CP_KIND)
    column = cr.CP_COUPLE - cr.CP_KIND
    assert np.all(kinds[: cp.n_slip, column] == 0.0) and np.all(kinds[cp.n_slip :, column] == 8.0)
    plain = CrystalPlasticity.from_config({**slip, "cleavage": spec})
    assert plain.cleavage_coupling == 0.0 and "coupling" not in plain.to_config()["cleavage"]
    assert np.all(plain.kinds_array()[:, column] == 0.0)
    # the damage of the three planes takes 3 x 8 x 600 / 150 = 96 MPa off the slip at most
    for bad, match in [
        ({**spec, "coupling": -1.0}, "a coupling of 0 or more"),
        ({**spec, "coupling": "strong"}, "coupling: a number"),
        ({**spec, "coupling": 8.5}, "no resistance to the slip systems of 100 .* below 8.333"),
    ]:
        with pytest.raises(ValueError, match=match):
            CrystalPlasticity.from_config({**slip, "cleavage": bad})
    with pytest.raises(ValueError, match="that of cleavage planes"):
        CrystalPlasticity(cp.families, cleavage_coupling=1.0)
    # planes alone: nothing to couple them with, and no bound
    alone = CrystalPlasticity.from_config({"cleavage": {**spec, "coupling": 50.0}})
    assert alone.cleavage_coupling == 50.0


@pytest.mark.parametrize("finite", [False, True])
def test_tangent_of_coupled_slip_and_cleavage(finite):
    """The tangent of the implicit update against finite differences for a crystal that
    has slipped, whose planes have opened, and that slips and cleaves together over a
    step: the slip lowers the strength of the planes, their damage that of the slip. And
    the Jacobian of the local residual against its finite differences."""
    planes = {"planes": "cube", "strength": 600.0, "softening": 150.0, "K": 30.0, "n": 2.0,
              "shear": 0.6, "coupling": 6.0}  # fmt: skip
    m = cubic(K=20.0, n=4.0, families={"bcc110": {"tau0": 100.0, "Q": 30.0, "b": 10.0}},
              cleavage=planes)  # fmt: skip
    cp = m.crystal_plasticity
    euler = (15.0, 25.0, 40.0)
    table, beh = _kernel(m, euler, finite)
    ns, first = cp.n_systems, cp.n_slip
    old = np.zeros((1, beh.width))
    v = old[0, cr.CP_ALPHA + ns : cr.CP_ALPHA + 2 * ns]
    v[:first] = 0.008  # a cumulated slip of 0.096: the planes are at 56 % of their strength
    for plane in range(3):
        old[0, cr.CP_ALPHA + first + 3 * plane] = v[first + 3 * plane] = 0.001 * (plane + 1)
    shape = np.array([[1.0e-3, 8e-4, 1.1e-3], [8e-4, 0.2e-3, 7e-4], [1.1e-3, 7e-4, 1.6e-3]])
    for start, end in ((0.85, 0.9), (1.2, 1.26)):
        # a small step from a state that a first step has brought to its flow rules (a
        # large one would be divided, and its tangent that of its last part)
        state = _point(table, beh, _gradient(start * shape, euler, finite), old, 0.5, finite)[2]
        grad = _gradient(end * shape, euler, finite)
        s, a, new = _point(table, beh, grad, state, 0.05, finite)
        moved = (new - state)[0, cr.CP_ALPHA + ns : cr.CP_ALPHA + 2 * ns]
        assert (moved[:first] > 1e-8).sum() >= 1 and (moved[first:] > 1e-8).sum() >= 3
        fd = _differences(table, beh, grad, state, 0.05, finite)
        assert np.allclose(a, fd, rtol=0, atol=2e-6 * np.abs(fd).max())
        s1, _, new1 = _point(table, beh, grad, state, 0.05, finite, guess=1, new=new)
        assert np.allclose(new1, new, rtol=0, atol=1e-11) and np.allclose(s1, s, atol=1e-7)
    if finite:
        return
    # the Jacobian of the residual of an active set of slips and of systems of the planes
    par = np.zeros((ns, cr.CP_WIDTH))
    par[:, : cr.CP_KIND], par[:, cr.CP_KIND :] = cp.params_array(), cp.kinds_array()
    normals, directions = cp.systems()
    systems = np.concatenate([directions, normals], axis=1)
    stiffness = np.ascontiguousarray(m.matrix("elastic"))
    work = cr.work_arrays(ns)
    schmid, stiff = work[5][0], work[5][1]
    cr._schmid_small(systems, ns, stiffness, schmid, stiff, work[6][3], work[1][3], work[1][2])
    rng = np.random.default_rng(3)
    stress = 150.0 * rng.normal(size=6) + [250.0, 250.0, 250.0, 0.0, 0.0, 0.0]
    weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    work[0][4][:ns] = schmid[:ns] @ (weights * stress)
    # (the interaction of the systems through the elasticity: m_s : C : m_r)
    interaction = (schmid[:ns] * weights) @ stiff[:ns].T
    assert np.allclose(work[1][3][:ns, :ns], interaction, rtol=1e-13, atol=1e-9)
    alpha0, v0 = old[0, cr.CP_ALPHA : cr.CP_ALPHA + ns].copy(), v.copy()
    active = [0, 3, 7, first, first + 1, first + 3, first + 5, first + 6]
    dg, act, sgn = np.zeros(ns), np.zeros(ns), np.ones(ns)
    act[active] = 1.0
    dg[active] = 2e-4 + 3e-4 * rng.random(len(active))
    fv, none = np.zeros(ns), work[3][0]

    def residual(slips, jacobian=False):
        out = np.zeros(ns)
        cr._cp_eval(False, par, cp.interaction_matrix(), ns, 0.5, alpha0, v0, slips, act, sgn,
                    systems, stiffness, none, none, work, out, fv, jacobian)  # fmt: skip
        return out

    residual(dg, True)
    jacobian = work[1][0][np.ix_(active, active)].copy()
    numeric = np.zeros_like(jacobian)
    for j, r in enumerate(active):
        step = np.zeros(ns)
        step[r] = 1e-8
        numeric[:, j] = (residual(dg + step) - residual(dg - step))[active] / 2e-8
    assert np.allclose(jacobian, numeric, rtol=0, atol=1e-6 * np.abs(jacobian).max())
    # with the terms of the coupling: of the planes on the slips, of the slips on the planes
    slips, cleaving = np.array(active) < first, np.array(active) >= first
    assert np.abs(jacobian[np.ix_(slips, cleaving)]).min() > 1.0
    assert np.abs(jacobian[np.ix_(cleaving, slips)]).min() > 1.0


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_homogeneous_cells_follow_the_point(kinematics):
    """A crystal strained along [001] and sheared, its whole strain prescribed: it slips,
    its planes open below their strength and break; unloaded, they close. On voxels and
    on a mesh, a homogeneous crystal is its material point."""
    m = iron(K=20.0)
    zero = [0.0, 0.0]
    strain = {"11": zero, "22": zero, "33": [0.02, 0.012], "23": zero, "31": [0.004, 0.004],
              "12": zero}  # fmt: skip
    path = {"times": [0, 1, 2], "increments": [40, 20], "strain": strain}
    euler = np.array([5.0, 10.0, 20.0])
    ref = simulate_point(m, path, orientation=euler, kinematics=kinematics)
    assert ref.info["cuts"] == 0
    damage = ref.series["cleavage_damage"]
    assert damage[-1] > 0.99 and ref.plastic[-1] > 2e-3
    assert 0.9 * STRENGTH < ref.stress[:, 2].max() < 0.995 * STRENGTH  # below their strength
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    fft = NonlinearFFTProblem(image, {0: m}, grains=np.zeros(image.shape, int),
                              orientations=euler[None], kinematics=kinematics)  # fmt: skip
    mesh = box_mesh((1, 1, 1), order=1, perturb=0.2)
    mesh = dataclasses.replace(mesh, cell=np.zeros(mesh.n_elements, int), orientations=euler[None])
    fe = NonlinearFEProblem(mesh, {0: m}, "periodic", kinematics=kinematics)
    for problem in (fft, fe):
        res = problem.run(path)
        assert res.info["cuts"] == 0
        assert np.allclose(res.stress, ref.stress, rtol=0, atol=1e-9 * np.abs(ref.stress).max())
        for name in ("cleavage_damage", "cleavage_opening"):
            assert np.allclose(res.series[name], ref.series[name], rtol=0, atol=1e-10)


@pytest.mark.slow
def test_slip_nucleates_the_cleavage_of_a_polycrystal():
    """A polycrystal of iron that yields below the strength of its {100} planes: it
    hardens until their strength is reached somewhere, and breaks across a section. With
    the coupling the slip of its grains lowers the strength of their planes: it breaks
    sooner, under a lower stress, after less plastic strain."""
    from morphogen.domain import Domain
    from morphogen.models.base import model_from_config

    size, shape, energy, strength = 0.1, 8, 0.1, 800.0
    model = model_from_config({"type": "voronoi", "n_cells": 8, "crystal_orientations": True})
    micro = model.generate(Domain(size=size, periodic=True), 2)
    image = micro.voxelize(shape=shape)
    runs = {}
    for coupling in (0.0, 5.0):
        material = cubic(
            K=10.0,
            n=10.0,
            families={"bcc110": {"tau0": 150.0, "Q": 100.0, "b": 20.0}},
            cleavage={
                "planes": "cube",
                "strength": strength,
                "energy": energy,
                "length": size / shape,
                "K": 50.0,
                "n": 1.0,
                "coupling": coupling,
            },
        )
        problem = NonlinearFFTProblem(image, {1: material}, grains=micro.labels(grid=image.grid()),
                                      orientations=micro.orientations)  # fmt: skip
        path = {"times": [0, 1], "increments": 60, "strain": {"33": 0.06}}
        runs[coupling] = problem.run(path, fields=True)
    plain, coupled = runs[0.0], runs[5.0]
    for res in (plain, coupled):
        stress = res.stress[:, 2]
        assert stress[-1] < 0.15 * stress.max()
        broken = int((np.asarray(res.fields[1.0]["cleavage_damage"]) > 0.9).sum())
        assert shape**2 <= broken < 2 * shape**2  # one section of the cell
    peaks = [int(np.argmax(res.stress[:, 2])) for res in (plain, coupled)]
    # the same polycrystal up to its yield; then 576 MPa at 1.6 % for 648 MPa at 2.4 %
    assert np.allclose(coupled.stress[:8], plain.stress[:8], rtol=0, atol=1e-6 * strength)
    assert coupled.stress[peaks[1], 2] < 0.92 * plain.stress[peaks[0], 2]
    assert coupled.strain[peaks[1], 2] < 0.75 * plain.strain[peaks[0], 2]
    assert coupled.plastic[peaks[1]] < 0.75 * plain.plastic[peaks[0]]
    assert 0.5 * strength < coupled.stress[peaks[1], 2] < plain.stress[peaks[0], 2] < strength
