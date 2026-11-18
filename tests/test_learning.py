"""Behaviours learned from data (morphogen.learning): the random paths, the gradients of
the models against central differences, the convexity of the potential, elasticity,
Hencky's hyperelasticity and von Mises plasticity learned, the models of the logarithmic
and of the Green-Lagrange strain in finite strains, the learned laws at a material point
and on voxels, reproducibility, files and the command line."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from helpers import box_mesh
from morphogen import cli
from morphogen.homogenization.behaviours import MFRONT
from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem
from morphogen.homogenization.finite import gradient_of, log_strain
from morphogen.homogenization.materials import Material
from morphogen.homogenization.nonlinear import NonlinearFEProblem, simulate_point
from morphogen.homogenization.paths import LoadingPath
from morphogen.image import VoxelImage
from morphogen.learning import (
    Dataset,
    FeedForward,
    LearnedBehaviour,
    Potential,
    Recurrent,
    Scales,
    Standard,
    errors,
    generate,
    isotropic_constants,
    learn,
    learned_material,
    load_model,
    path_errors,
    random_paths,
    train,
)
from morphogen.learning.behaviour import LearnedPoints, finite_stress, learned_law
from morphogen.learning.data import _tensor, _voigt, gradient_tensor, green_lagrange
from morphogen.learning.logarithmic import _measures, _pullback
from morphogen.learning.networks import STRAINS, Model
from morphogen.learning.training import rotations, voigt_rotations
from morphogen.orientations import euler_to_matrix

ELASTIC = {"young": 200000.0, "poisson": 0.3}
PLASTIC = {**ELASTIC, "plasticity": {"yield": 200.0, "hardening": 10000.0}}
#: An orthotropic stiffness in morphoGen's Voigt order (11, 22, 33, 23, 31, 12): its three
#: shear moduli differ, so that an exchange of shear components shows
ORTHOTROPIC = np.array([[250.0, 100.0, 80.0, 0, 0, 0], [100.0, 200.0, 90.0, 0, 0, 0],
                        [80.0, 90.0, 150.0, 0, 0, 0], [0, 0, 0, 60.0, 0, 0],
                        [0, 0, 0, 0, 40.0, 0], [0, 0, 0, 0, 0, 20.0]]) * 1000.0  # fmt: skip
#: A path of the six strain components
SIX = ("t=0,1,2 n=2 E11=0,0.001,0 E22=0,-0.0005,0.0008 E33=0,0.0003,0 E23=0,0.002,0.001 "
       "E31=0,-0.001,0.0015 E12=0,0.0007,-0.002")  # fmt: skip


@pytest.fixture(scope="module")
def plastic_data():
    return generate(PLASTIC, random_paths(12, segments=3, increments=5, seed=3))


@pytest.fixture(scope="module")
def finite_data():
    """Von Mises plasticity in finite strains (the Hencky points) along random walks of the
    displacement gradient to 10 %."""
    return generate(PLASTIC, random_paths(8, segments=3, increments=5, amplitude=0.1, seed=3),
                    kinematics="finite")  # fmt: skip


def _perturbed(model, data, seed=2):
    """A model set up on ``data``, its parameters moved off their start."""
    model.setup(data)
    rng = np.random.default_rng(seed)
    for k, v in model.params.items():
        model.params[k] = v + 0.1 * rng.normal(size=v.shape)
    return model


def _differences(model, arrays, h=1e-6) -> float:
    """The largest gap of the gradients of the objective to central differences, relative
    to the largest gradient of each parameter (three entries of each)."""
    _, grads = model.objective(arrays)
    rng = np.random.default_rng(1)
    worst = 0.0
    for name, value in model.params.items():
        scale = max(float(np.abs(grads[name]).max()), 1e-12)
        for _ in range(3):
            i = tuple(int(rng.integers(s)) for s in value.shape)
            old = value[i]
            value[i] = old + h
            up = model.objective(arrays, grad=False)[0]
            value[i] = old - h
            down = model.objective(arrays, grad=False)[0]
            value[i] = old
            worst = max(worst, abs((up - down) / (2 * h) - grads[name][i]) / scale)
    return worst


def _jacobian(f, x, h=1e-7):
    """The central differences of ``f`` with respect to the last axis of ``x``."""
    cols = []
    for k in range(x.shape[-1]):
        d = np.zeros(x.shape[-1])
        d[k] = h
        cols.append((f(x + d) - f(x - d)) / (2 * h))
    return np.stack(cols, axis=-1)


def test_random_paths():
    paths = random_paths(5, segments=3, increments=4, amplitude=0.01, volumetric=0.0, seed=1)
    values = np.array([p.values for p in paths])
    assert values.shape == (5, 4, 6) and all(p.increments == (4, 4, 4) for p in paths)
    assert np.allclose(values[..., :3].sum(axis=-1), 0.0, atol=1e-15)  # isochoric
    assert np.abs(values[..., 3:]).max() <= 0.02  # engineering shears
    again = random_paths(5, segments=3, increments=4, amplitude=0.01, volumetric=0.0, seed=1)
    assert np.array_equal(values, np.array([p.values for p in again]))
    linear = random_paths(5, kind="linear", seed=1)
    assert not np.array_equal(np.array([p.values for p in linear]), values)
    tension = random_paths(1, components=["11", "12"], free=["22", "33"], rate=0.01, seed=0)[0]
    assert tension.controls == (True, False, False, True, True, True)
    v = np.asarray(tension.values)
    assert np.all(v[:, 1:5] == 0.0)
    assert np.allclose(np.diff(tension.times), np.abs(np.diff(v, axis=0)).max(axis=1) / 0.01)
    with pytest.raises(ValueError, match="kind"):
        random_paths(2, kind="spiral")
    with pytest.raises(ValueError, match="free"):
        random_paths(2, components=["11"], free=["11"])


def test_rotations():
    rot = rotations(4, np.random.default_rng(0))
    assert np.allclose(rot @ np.swapaxes(rot, 1, 2), np.eye(3)) and np.allclose(
        np.linalg.det(rot), 1
    )
    te, ts = voigt_rotations(rot)
    e, s = np.random.default_rng(1).normal(size=(2, 6))
    # the work is invariant, an isotropic stiffness commutes with the rotations
    work = np.einsum("ni,ni->n", np.einsum("nij,j->ni", te, e), np.einsum("nij,j->ni", ts, s))
    assert np.allclose(work, e @ s)
    lam, mu = 1.0, 0.7
    c = lam * np.outer([1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]) + mu * np.diag([2, 2, 2, 1, 1, 1])
    assert np.allclose(ts @ c, c @ te)
    assert isotropic_constants(c) == pytest.approx((mu * (3 * lam + 2 * mu) / (lam + mu),
                                                    lam / (2 * (lam + mu))))  # fmt: skip


@pytest.mark.parametrize("kind", ["feedforward", "potential", "recurrent", "standard",
                                  "viscoplastic", "viscoelastic", "logarithmic",
                                  "green-lagrange", "feedforward-ln", "potential-ln",
                                  "recurrent-ln"])  # fmt: skip
def test_gradients_against_differences(plastic_data, finite_data, kind):
    small = {"hidden": 5, "internal": 3}
    model = {"feedforward": FeedForward(hidden=(5, 4)), "potential": Potential(hidden=(5, 4)),
             "recurrent": Recurrent(hidden=5, rate=True), "standard": Standard(**small),
             "viscoplastic": Standard(**small, rate=True),
             "viscoelastic": Standard(**small, rate=True, plastic=False),
             "logarithmic": Standard(**small),  # finite strains: the free energy of 1/2 ln C
             "green-lagrange": Standard(**small, strain="green-lagrange"),
             # finite strains: the networks of 1/2 ln C, their outputs T, S = J^T T
             "feedforward-ln": FeedForward(hidden=(5, 4)), "potential-ln": Potential(hidden=(5, 4)),
             "recurrent-ln": Recurrent(hidden=5, rate=True)}[kind]  # fmt: skip
    finite = kind in ("logarithmic", "green-lagrange") or kind.endswith("-ln")
    data = finite_data if finite else plastic_data
    model = _perturbed(model, data)
    arrays = model.prepare(data)
    count = 4 if model.recurrent else 40
    # the losses of a standard model hold the round-off of Newton's method: larger steps
    h = 1e-5 if model.kind == "standard" else 1e-6
    assert _differences(model, tuple(a[:count] for a in arrays), h) < 1e-6


def test_tangents_against_differences(plastic_data):
    e = plastic_data.strain[0, 5:8]
    for model in (FeedForward(hidden=(5, 4)), Potential(hidden=(5, 4))):
        model = _perturbed(model, plastic_data)
        fd = _jacobian(model.predict, e)
        assert np.abs(model.tangent(e) - fd).max() < 1e-6 * np.abs(fd).max()
    model = _perturbed(Recurrent(hidden=5, rate=True), plastic_data)
    e0, e1 = plastic_data.strain[0, 3:5], plastic_data.strain[0, 4:6]
    h0 = np.random.default_rng(0).normal(size=(2, 5))
    _, c, _ = model.step(e1, e0, h0, 0.1)
    fd = _jacobian(lambda x: model.step(x, e0, h0, 0.1)[0], e1)
    assert np.abs(c - fd).max() < 1e-6 * np.abs(fd).max()
    # the steps of points are the recurrence of the paths
    out = model.predict(plastic_data.strain[:2], plastic_data.times[:2])
    state, steps = np.zeros((2, 5)), []
    for t in range(1, plastic_data.n_points):
        dt = plastic_data.times[0, t] - plastic_data.times[0, t - 1]
        s, _, state = model.step(plastic_data.strain[:2, t], plastic_data.strain[:2, t - 1],
                                 state, dt)  # fmt: skip
        steps.append(s)
    assert np.allclose(np.stack(steps, axis=1), out[:, 1:], rtol=0, atol=1e-9 * np.abs(out).max())
    # P = F S and dP/dF of a law of the Green-Lagrange strain
    model = _perturbed(Potential(hidden=(5, 4)), plastic_data)
    f = np.eye(3) + 0.05 * np.random.default_rng(3).normal(size=(2, 3, 3))

    def pk1(h):
        g = np.eye(3) + h.reshape(-1, 3, 3)
        e = green_lagrange(g)
        return finite_stress(g, model.predict(e), model.tangent(e))

    a = pk1((f - np.eye(3)).reshape(2, 9))[1]
    fd = _jacobian(lambda h: pk1(h)[0], (f - np.eye(3)).reshape(2, 9))
    assert np.abs(a - fd).max() < 1e-6 * np.abs(fd).max()


def test_potential_convex(plastic_data):
    model = _perturbed(Potential(hidden=(6, 5)), plastic_data, seed=4)
    e = 0.02 * np.random.default_rng(5).normal(size=(50, 6))
    c = model.tangent(e)
    assert np.array_equal(c, np.swapaxes(c, 1, 2))
    assert np.linalg.eigvalsh(c).min() >= -1e-10 * np.abs(c).max()
    psi = model.energy
    a, b = e[:25], e[25:]
    assert np.all(psi(0.5 * (a + b)) <= 0.5 * (psi(a) + psi(b)))
    assert np.all(psi(e) >= 0.0) and psi(np.zeros(6)) == 0.0
    assert np.abs(model.predict(np.zeros(6))).max() < 1e-12 * np.abs(model.predict(e)).max()
    # the stress is the gradient of the energy
    fd = _jacobian(psi, e)
    assert np.abs(fd - model.predict(e)).max() < 1e-6 * np.abs(fd).max()


def test_learns_elasticity():
    data = generate(ELASTIC, random_paths(10, segments=3, increments=4, seed=0))
    train_set, valid = data.split(0.3)
    for cls in (FeedForward, Potential):
        model = cls(hidden=(8, 8))
        train(model, train_set, valid, epochs=50, batch=32)
        assert errors(model, valid)["relative"] < 0.01  # 0.002 and 0.0008 measured
    young, poisson = isotropic_constants(model.tangent(np.zeros(6)))
    assert young == pytest.approx(200000.0, rel=0.01) and poisson == pytest.approx(0.3, abs=0.005)
    # the learned law at a material point, the lateral stresses solved for
    path = LoadingPath.parse("t=0,1 n=4 E11=0.002")
    got = simulate_point(learned_material(model), path)
    ref = simulate_point(Material.from_config(ELASTIC), path)
    assert np.abs(got.stress - ref.stress).max() < 0.01 * np.abs(ref.stress).max()
    assert np.abs(got.strain - ref.strain).max() < 0.01 * np.abs(ref.strain).max()


def test_learns_hyperelasticity():
    """Hencky's elasticity in finite strains: ``S(E)`` far from linear (the linear fit misses
    25 %), learned by a potential of ``E``; ``T(1/2 ln C)`` linear, the quadratic term of a
    potential of ``ln C``, its start (the linear fit of ``T`` to ``H``) Hencky's law to
    round-off, at a material point too; the learned laws along a uniaxial tension."""
    paths = random_paths(24, segments=3, increments=5, amplitude=0.04, volumetric=0.3, seed=0)
    data = generate(ELASTIC, paths, kinematics="finite")
    assert data.measure == "green-lagrange" and data.n_points == 16
    train_set, valid = data.split(0.25)
    model = Potential(hidden=(16, 16), strain="green-lagrange")
    train(model, train_set, valid, epochs=300, batch=32)
    linear = Potential(hidden=(1,), strain="green-lagrange")
    linear.setup(train_set)
    assert errors(model, valid)["relative"] < 0.07 < 0.2 < errors(linear, valid)["relative"]
    hencky = Potential(hidden=(1,))  # of ln C
    hencky.setup(train_set)
    hencky.params["v"][:] = -60.0  # the network silenced: the quadratic term of the start
    assert errors(hencky, valid)["relative"] < 1e-12  # 3e-15 measured
    logarithmic = Potential(hidden=(16, 16))
    train(logarithmic, train_set, valid, epochs=300, batch=32)
    assert errors(logarithmic, valid)["relative"] < 0.005  # 0.017 % measured (4.2 % in E)
    # the learned laws in finite strains at a material point: a uniaxial tension
    path = LoadingPath.parse("t=0,1 n=5 E11=0.03")
    ref = simulate_point(Material.from_config(ELASTIC), path, kinematics="finite")
    for law, most in ((model, 0.05), (logarithmic, 0.01), (hencky, 1e-10)):
        got = simulate_point(learned_material(law, **ELASTIC), path, kinematics="finite")
        assert np.abs(got.stress - ref.stress).max() < most * np.abs(ref.stress).max()
    # Hencky's law: the lateral strains of the reference
    assert np.abs(got.strain - ref.strain).max() < 1e-10 * np.abs(ref.strain).max()


@pytest.mark.slow
def test_learns_plasticity():
    # random walks of the strains 11, 22, 33 and 12, turned by random rotations
    paths = random_paths(128, segments=4, increments=8, components=["11", "22", "33", "12"])
    data = generate(PLASTIC, paths)
    train_set, valid = data.split(0.25)
    model = Recurrent(hidden=32)
    train(model, train_set, valid, epochs=300, batch=16, rotate=True)
    baseline = FeedForward(hidden=(32, 32))
    train(baseline, train_set, valid, epochs=20, batch=64)
    assert errors(model, valid)["relative"] < 0.07 < 0.25 < errors(baseline, valid)["relative"]
    tests = {
        "cycle": "t=0,1,3,4 n=8,16,8 E11=0,0.01,-0.01,0 E12=0 E23=0 E31=0",  # uniaxial stress
        "shear": "t=0,1,3 n=8,16 E23=0,0.015,-0.015 E12=0 E31=0",  # not driven by the data
        "turn": "t=0,1,2 n=8,8 E11=0,0.006,0.006 E12=0,0,0.012 E23=0 E31=0",
    }
    learned = path_errors(model, PLASTIC, tests)
    base = path_errors(baseline, PLASTIC, tests)
    for name in tests:
        assert learned[name]["stress"] < 0.2 and learned[name]["strain"] < 0.2, name
    # a function of the strain has no hysteresis
    assert min(base["cycle"]["stress"], base["shear"]["stress"]) > 0.4


def _isotropic(young, poisson):
    """The isotropic stiffness of Young's modulus and Poisson's ratio (engineering shears)."""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = 0.5 * young / (1.0 + poisson)
    trace = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    return lam * np.outer(trace, trace) + mu * np.diag([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])


#: A standard linear solid: its relaxed stiffness, that of its Maxwell branch and the
#: relaxation time of the branch
RELAXED, BRANCH, RELAXATION = _isotropic(100000.0, 0.3), _isotropic(100000.0, 0.25), 0.5


def _linear_solid(paths) -> Dataset:
    """The responses of the standard linear solid along strain paths: ``psi = eps C_r eps /
    2 + (eps - a) C_b (eps - a) / 2``, ``phi = t_r a' C_b a' / 2``, the viscous strain ``a``
    of a step the minimizer of the incremental potential (implicit Euler)."""
    times, strains, stresses = [], [], []
    for path in paths:
        t = np.concatenate([[path.times[0]], path.schedule()])
        e = np.array([path.at(s)[0] for s in t])
        a, out = np.zeros(6), [np.zeros(6)]
        for k in range(1, len(t)):
            dt = t[k] - t[k - 1]
            a = (dt * e[k] + RELAXATION * a) / (dt + RELAXATION)
            out.append(RELAXED @ e[k] + BRANCH @ (e[k] - a))
        times.append(t)
        strains.append(e)
        stresses.append(np.array(out))
    return Dataset(np.array(times), np.array(strains), np.array(stresses))


def test_standard_thermodynamics(plastic_data):
    """A standard model whatever its parameters (moved off their start), along paths beyond
    its data: the dissipation of every step nonnegative, the tangent symmetric, positive
    definite and the derivative of the stress of a step (its starting state held); the
    stress and the thermodynamic forces the derivatives of the free energy, the
    dissipation their work along the internal variables; the steps of points the
    recurrence of the paths."""
    model = _perturbed(Standard(hidden=6, internal=4, rate=True), plastic_data, seed=5)
    strain, times = 1.5 * plastic_data.strain[:6], plastic_data.times[:6]
    h = model.history(strain, times)
    spent = h["dissipation"][:, 1:]
    work = np.abs(h["stress"] * strain).max()
    assert spent.min() >= -1e-12 * work and spent.max() > 1e-3 * work
    assert np.all(h["iterations"][:, 1:] >= 0)  # converged (negative otherwise)
    c = h["tangent"][:, 1:]
    assert np.array_equal(c, np.swapaxes(c, -1, -2)) and np.linalg.eigvalsh(c).min() > 0
    alpha = h["internal"]
    fd = _jacobian(lambda e: model.energy(e, alpha), strain)
    assert np.abs(fd - h["stress"]).max() < 1e-6 * np.abs(h["stress"]).max()
    force = -_jacobian(lambda a: model.energy(strain, a), alpha, h=1e-6)
    done = np.sum(force[:, 1:] * np.diff(alpha, axis=1), axis=-1)
    assert np.abs(done - spent).max() < 1e-6 * spent.max()
    # a step of points from the states of the paths: their stresses, states and tangents
    e0, e1, a0 = strain[:3, 4], strain[:3, 5], alpha[:3, 4]
    dt = times[0, 5] - times[0, 4]
    s, tangent, a1 = model.step(e1, e0, a0, dt)
    assert np.allclose(s, h["stress"][:3, 5], rtol=0, atol=1e-12 * np.abs(s).max())
    assert np.allclose(a1, alpha[:3, 5], rtol=0, atol=1e-12 * np.abs(a1).max())
    fd = _jacobian(lambda x: model.step(x, e0, a0, dt)[0], e1)
    assert np.abs(tangent - fd).max() < 1e-6 * np.abs(fd).max()
    with pytest.raises(ValueError, match="times"):
        model.predict(strain)
    with pytest.raises(ValueError, match="dissipation"):
        Standard(plastic=False)
    with pytest.raises(ValueError, match="strain"):
        Standard(strain="hencky")


@pytest.mark.parametrize("strain", ["logarithmic", "green-lagrange"])
def test_standard_finite_strains(plastic_data, finite_data, strain):
    """A standard model in finite strains whatever its parameters, along paths of gradients
    to 15 %: thermodynamically consistent (the dissipation of every step nonnegative, ``S =
    dpsi/dE`` and the dissipation the work of the forces of its free energy), its tangent
    ``dS/dE`` symmetric and the derivative of the stress of a step, ``dP/dF`` that of ``P =
    F S`` at its points; objective (a rotation superposed on the gradients turns ``P`` and
    its tangent, the state unchanged along two steps); the small strain model of the same
    parameters recovered at small deformations, to the first order of the gradient."""
    model = _perturbed(Standard(hidden=6, internal=4, strain=strain), finite_data, seed=5)
    assert model.logarithmic == (strain == "logarithmic")
    e = 1.5 * finite_data.strain[:6]
    h = model.history(e)
    spent = h["dissipation"][:, 1:]
    work = np.abs(h["stress"] * e).max()
    assert spent.min() >= -1e-12 * work and spent.max() > 1e-3 * work
    assert np.all(h["iterations"][:, 1:] >= 0)
    c = h["tangent"][:, 1:]
    assert np.array_equal(c, np.swapaxes(c, -1, -2))
    if strain == "green-lagrange":  # convex in E; in H, T : d2H/dE2 may make it indefinite
        assert np.linalg.eigvalsh(c).min() > 0
    alpha = h["internal"]
    fd = _jacobian(lambda x: model.energy(x, alpha), e)
    assert np.abs(fd - h["stress"]).max() < 1e-6 * np.abs(h["stress"]).max()
    force = -_jacobian(lambda a: model.energy(e, a), alpha, h=1e-6)
    done = np.sum(force[:, 1:] * np.diff(alpha, axis=1), axis=-1)
    assert np.abs(done - spent).max() < 1e-6 * spent.max()
    e0, e1, a0 = e[:3, 4], e[:3, 5], alpha[:3, 4]
    s, tangent, a1 = model.step(e1, e0, a0, 1.0)
    assert np.allclose(s, h["stress"][:3, 5], rtol=0, atol=1e-12 * np.abs(s).max())
    assert np.allclose(a1, alpha[:3, 5], rtol=0, atol=1e-12 * np.abs(a1).max())
    fd = _jacobian(lambda x: model.step(x, e0, a0, 1.0)[0], e1)
    assert np.abs(tangent - fd).max() < 1e-6 * np.abs(fd).max()
    # at its points: P = F S, dP/dF; a superposed rotation R F turns P and dP/dF
    rng = np.random.default_rng(6)
    rot = rotations(4, rng)
    points, law = _points(model, np.repeat(np.eye(3)[None], 4, axis=0), finite=True)
    grads = [0.12 * rng.normal(size=(4, 9)) for _ in range(2)]
    a = _steps(points, law, grads)

    def spun(g):  # R F - I
        return (np.einsum("nij,njk->nik", rot, np.eye(3) + g.reshape(4, 3, 3))
                - np.eye(3)).reshape(4, 9)  # fmt: skip

    for (p, t), (p1, t1) in zip(a, _steps(points, law, [spun(g) for g in grads]), strict=True):
        turned = np.einsum("nij,njk->nik", rot, p.reshape(4, 3, 3)).reshape(4, 9)
        assert np.abs(p1 - turned).max() < 1e-12 * np.abs(p).max()
        t4 = np.einsum("nia,nkc,naJcL->niJkL", rot, rot, t.reshape(4, 3, 3, 3, 3))
        assert np.abs(t1 - t4.reshape(4, 9, 9)).max() < 1e-12 * np.abs(t).max()
    fd = _jacobian(lambda g: _steps(points, law, [grads[0], g])[1][0], grads[1])
    assert np.abs(a[1][1] - fd).max() < 1e-6 * np.abs(fd).max()
    # the small strain model of the same parameters at gradients lam h (h: its data, to 1 %),
    # its scales lam s: the same normalized response but for the geometric terms, of the
    # order of lam
    small = _perturbed(Standard(hidden=6, internal=4), plastic_data, seed=5)
    gaps = []
    for lam in (1.0, 1e-2):
        s0 = small.scales
        scaled = copy.deepcopy(small)
        scaled.scales = Scales(lam * s0.strain, lam * s0.stress, s0.increment, s0.time)
        finite = copy.deepcopy(scaled)
        finite.measure, finite.options["strain"] = "green-lagrange", strain
        g = lam * plastic_data.strain[:4]
        f = gradient_tensor(g)
        pk2 = finite.history(green_lagrange(f))["stress"]
        cauchy = np.einsum("ptij,ptjk,ptlk->ptil", f, _tensor(pk2, False), f)
        cauchy = _voigt(cauchy / np.linalg.det(f)[..., None, None], False)
        ref = scaled.history(g)["stress"]
        gaps.append(np.abs(cauchy - ref).max() / np.abs(ref).max())
    # measured 3.1 % and 3.1e-4 (12.5 % and 0.13 % in E): the ratio lam
    assert gaps[0] < 0.2 and 0.005 < gaps[1] / gaps[0] < 0.02


def test_learns_a_standard_linear_solid():
    """A standard linear solid (quadratic free energy and dissipation potential) is a
    standard model: its parameters set, the model gives its responses to round-off in one
    Newton iteration a step, and its dissipation; learned from its responses along random
    paths at random rates, it is recovered (the relaxation of an unseen step of strain)."""
    data = _linear_solid(random_paths(24, segments=3, increments=6, rate=0.01, seed=0))
    train_set, valid = data.split(0.25)
    model = Standard(hidden=1, internal=6, rate=True, plastic=False)
    model.setup(train_set)
    s = model.scales
    ratio = s.strain / s.stress  # x = eps / s_eps, alpha = a / s_eps, psi / (s_sigma s_eps)
    model.params.update(L=np.linalg.cholesky(ratio * np.block([[RELAXED + BRANCH, -BRANCH],
                                                               [-BRANCH, BRANCH]])),
                        M=np.linalg.cholesky(ratio * RELAXATION * BRANCH / s.time),
                        w=np.full(1, -60.0))  # fmt: skip
    h = model.history(valid.strain, valid.times)
    size = np.abs(valid.stress).max()
    assert np.abs(h["stress"] - valid.stress).max() < 1e-12 * size
    assert np.all(h["iterations"][:, 1:] == 1)
    rate = np.diff(s.strain * h["internal"], axis=1) / np.diff(valid.times, axis=1)[..., None]
    spent = RELAXATION * np.einsum("pti,ij,ptj->pt", rate, BRANCH, rate)  # D = t_r a' C_b a'
    spent = spent * np.diff(valid.times, axis=1)
    assert np.abs(h["dissipation"][:, 1:] - spent).max() < 1e-10 * spent.max()
    learned = Standard(hidden=4, internal=6, rate=True, plastic=False)
    train(learned, train_set, valid, epochs=600, batch=8, learning_rate=0.03)
    relax = _linear_solid([LoadingPath.parse("t=0,0.05,3 n=2,30 E11=0,0.004,0.004 E22=0 E33=0 "
                                             "E23=0 E31=0 E12=0,0.002,0.002")])  # fmt: skip
    assert errors(learned, valid)["relative"] < 0.005
    assert errors(learned, relax)["relative"] < 0.01


def test_standard_law_in_the_solvers(tmp_path, plastic_data):
    """A standard law integrated as a learned law along the seam of the MFront points, from
    its file: at a material point (every strain prescribed: the steps of the model; a
    tension, the lateral stresses solved for with its tangent), on voxels and by finite
    elements (homogeneous cells: the response of the point), its dissipation along the
    tension nonnegative; its options in its file and in the configurations."""
    model = Standard(hidden=6, internal=4)
    train(model, plastic_data, epochs=20, batch=4)
    path = model.save(tmp_path / "standard")
    loaded = load_model(path)
    assert loaded.kind == "standard" and loaded.options == model.options
    material = Material(**ELASTIC, learned=str(path))
    spec = "t=0,1,2 n=4 E11=0,0.004,-0.002 E22=0 E33=0 E23=0 E31=0 E12=0,0.002,0.003"
    res = simulate_point(material, LoadingPath.parse(spec))
    direct = loaded.predict(res.strain, res.times)
    assert np.allclose(res.stress, direct, rtol=0, atol=1e-10 * np.abs(direct).max())
    tension = LoadingPath.parse("t=0,1,2 n=5,5 E11=0,0.006,-0.002")  # the other stresses free
    point = simulate_point(material, tension)
    size = np.abs(point.stress).max()
    assert np.abs(point.stress[:, 1:]).max() < 1e-6 * size
    h = loaded.history(point.strain, point.times)
    assert np.allclose(h["stress"], point.stress, rtol=0, atol=1e-8 * size)
    assert h["dissipation"].min() >= 0.0 and h["dissipation"].sum() > 0.0
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    voxels = NonlinearFFTProblem(image, {0: material}).run(tension)
    elements = NonlinearFEProblem(box_mesh((2, 2, 2), perturb=0.2), {0: material}).run(tension)
    for got in (voxels, elements):
        assert np.abs(got.stress - point.stress).max() < 1e-6 * size
        assert np.abs(got.strain - point.strain).max() < 1e-6 * np.abs(point.strain).max()
    cfg = {"material": ELASTIC, "data": {"paths": 4, "segments": 1, "increments": 2},
           "model": {"kind": "standard", "hidden": 3, "internal": 2},
           "training": {"epochs": 2}}  # fmt: skip
    out = learn(cfg)[1]
    assert out["model"] == "standard" and out["options"]["internal"] == 2
    with pytest.raises(ValueError, match="internal"):
        learn({**cfg, "model": {"kind": "recurrent", "internal": 2}})


def test_standard_finite_law_in_the_solvers(tmp_path, finite_data):
    """A standard law of the logarithmic strain in finite strains, from its file: at a
    material point (a tension to 8 % and back, the lateral stresses solved for with ``dP/dF``),
    on voxels and by finite elements (homogeneous cells: the response of the point), its
    dissipation nonnegative; turned by an orientation, the law of the turned gradient
    turned back, its state carried; heated free of stress, ``F = exp(alpha dT) I``; its
    option in the configurations."""
    model = Standard(hidden=6, internal=4)
    train(model, finite_data, epochs=20, batch=4)
    path = model.save(tmp_path / "finite")
    loaded = load_model(path)
    assert loaded.options["strain"] == "logarithmic" and loaded.measure == "green-lagrange"
    material = Material(**ELASTIC, learned=str(path))
    tension = LoadingPath.parse("t=0,1,2 n=8,8 E11=0,0.08,0.02")  # the other stresses free
    point = simulate_point(material, tension, kinematics="finite")
    size = np.abs(point.stress).max()
    assert np.abs(point.stress[:, 1:]).max() < 1e-6 * size
    h = loaded.history(green_lagrange(gradient_tensor(point.strain)))
    assert h["dissipation"].min() >= 0.0 and h["dissipation"].sum() > 0.0
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    voxels = NonlinearFFTProblem(image, {0: material}, kinematics="finite").run(tension)
    elements = NonlinearFEProblem(box_mesh((2, 2, 2), perturb=0.2), {0: material},
                                  kinematics="finite").run(tension)  # fmt: skip
    for got in (voxels, elements):
        assert np.abs(got.stress - point.stress).max() < 1e-6 * size
        assert np.abs(got.strain - point.strain).max() < 1e-6 * np.abs(point.strain).max()
    # turned: P(g; F) = g^T P(I; g F g^T) g along two steps
    rng = np.random.default_rng(4)
    rot = rotations(3, rng)
    turned, law = _points(loaded, rot, finite=True)
    plain, _ = _points(loaded, np.repeat(np.eye(3)[None], 3, axis=0), finite=True)
    grads = [0.1 * rng.normal(size=(3, 9)) for _ in range(2)]

    def inside(g):  # g F g^T - I
        f = np.eye(3) + g.reshape(3, 3, 3)
        return (np.einsum("nij,njk,nlk->nil", rot, f, rot) - np.eye(3)).reshape(3, 9)

    plain_steps = _steps(plain, law, [inside(g) for g in grads])
    for (p, _), (p0, _) in zip(_steps(turned, law, grads), plain_steps, strict=True):
        back = np.einsum("nai,naB,nBJ->niJ", rot, p0.reshape(3, 3, 3), rot).reshape(3, 9)
        assert np.abs(p - back).max() < 1e-12 * np.abs(p).max()
    alpha, heat = 1.2e-5, 300.0
    hot = learned_material(loaded, **ELASTIC, expansion=alpha)
    free = simulate_point(hot, LoadingPath.parse(f"t=0,1 n=2 dT=0,{heat}"), kinematics="finite")
    assert np.allclose(free.strain[-1, :3], np.expm1(alpha * heat), rtol=1e-9)
    assert np.abs(free.strain[-1, 3:]).max() < 1e-15
    assert np.abs(free.stress).max() < 1e-9 * ELASTIC["young"] * alpha * heat
    cfg = {"material": PLASTIC, "kinematics": "finite",
           "data": {"paths": 4, "segments": 1, "increments": 2, "amplitude": 0.05},
           "model": {"kind": "standard", "hidden": 3, "internal": 2, "strain": "green-lagrange"},
           "training": {"epochs": 2}}  # fmt: skip
    model, out = learn(cfg)
    assert out["options"]["strain"] == "green-lagrange" and not model.logarithmic
    # the strain of the other models too, logarithmic by default
    recurrent = {"kind": "recurrent", "hidden": 3, "strain": "green-lagrange"}
    model, out = learn({**cfg, "model": recurrent})
    assert out["options"]["strain"] == "green-lagrange" and not model.logarithmic
    assert learn({**cfg, "model": {"kind": "potential", "hidden": [3]}})[0].logarithmic
    with pytest.raises(ValueError, match="strain"):
        learn({**cfg, "model": {"kind": "recurrent", "strain": "hencky"}})


@pytest.mark.slow
def test_standard_plasticity():
    """Von Mises plasticity learned by a standard model on the data of the recurrent one
    (test_learns_plasticity): its errors below those of the recurrent network on the
    validation paths and on the unseen paths, its dissipation nonnegative at every step of
    every path and its tangent symmetric positive definite; along the cycle its work is the
    reference's and its cumulated dissipation lies between the reference's dissipation
    (``R0 p``) and plastic work (``R0 p + H p^2 / 2``)."""
    paths = random_paths(128, segments=4, increments=8, components=["11", "22", "33", "12"])
    train_set, valid = generate(PLASTIC, paths).split(0.25)
    model = Standard(hidden=32, internal=8)
    train(model, train_set, valid, epochs=300, batch=16, rotate=True)
    assert errors(model, valid)["relative"] < 0.035  # 2.3 % measured (the recurrent 4.1 %)
    h = model.history(valid.strain, valid.times)
    assert h["dissipation"].min() >= 0.0
    c = h["tangent"][:, 1:]
    assert np.array_equal(c, np.swapaxes(c, -1, -2)) and np.linalg.eigvalsh(c).min() > 0
    tests = {
        "cycle": "t=0,1,3,4 n=8,16,8 E11=0,0.01,-0.01,0 E12=0 E23=0 E31=0",  # uniaxial stress
        "shear": "t=0,1,3 n=8,16 E23=0,0.015,-0.015 E12=0 E31=0",  # not driven by the data
        "turn": "t=0,1,2 n=8,8 E11=0,0.006,0.006 E12=0,0,0.012 E23=0 E31=0",
    }
    learned = path_errors(model, PLASTIC, tests)
    # measured 7.4 %, 4.8 % and 8.4 % (the recurrent network 7.9 %, 11 % and 12 %), the
    # strains 0.7 %, 0.5 % and 0.3 % (10.5 %, 6.0 % and 1.7 %)
    for name, most in (("cycle", 0.09), ("shear", 0.07), ("turn", 0.1)):
        assert learned[name]["stress"] < most and learned[name]["strain"] < 0.02, name
    path = LoadingPath.parse(tests["cycle"])
    ref = simulate_point(Material.from_config(PLASTIC), path)
    got = simulate_point(learned_material(model, **ELASTIC), path)
    h = model.history(got.strain, got.times)
    assert h["dissipation"].min() >= 0.0

    def work(r):
        return np.sum(0.5 * (r.stress[1:] + r.stress[:-1]) * np.diff(r.strain, axis=0))

    assert work(got) == pytest.approx(work(ref), rel=0.01)  # 11.11 MPa both
    p = ref.plastic[-1]
    yield_, hardening = PLASTIC["plasticity"]["yield"], PLASTIC["plasticity"]["hardening"]
    # measured 8.72 MPa between 6.03 and 10.57 MPa
    assert yield_ * p < h["dissipation"].sum() < yield_ * p + 0.5 * hardening * p * p


@pytest.mark.slow
def test_standard_finite_plasticity():
    """Von Mises plasticity with hardening in finite strains (the Hencky points) along walks
    of the logarithmic strains 11, 22, 33 and the simple shear 12 to 10 % (Green-Lagrange
    strains to 22 %): the standard model and the recurrent network of the logarithmic
    strain learn it as the small strain models learn the same walks in small strains, where
    the recurrent network and the standard model of the Green-Lagrange strain miss their
    own data; the dissipation of the standard model nonnegative at every step and its
    tangent symmetric; along unseen paths (the lateral strains solved for) the laws of
    ``ln C`` within the errors of their validation paths, and the dissipation of the
    standard model along the cycle between the reference's dissipation and its plastic
    work."""
    yield_, hardening = 200.0, 2000.0
    hard = {**ELASTIC, "plasticity": {"yield": yield_, "hardening": hardening}}
    walks = random_paths(128, segments=4, increments=8, amplitude=0.1, volumetric=0.01,
                         components=["11", "22", "33", "12"])  # fmt: skip
    paths = []
    for walk in walks:  # a step a segment, the stretches exp(e): the volume change of e
        t = np.concatenate([[walk.times[0]], walk.schedule()])
        v = np.array([walk.at(s)[0] for s in t])
        v[:, :3] = np.expm1(v[:, :3])
        paths.append(LoadingPath(t, v, walk.controls, increments=(1,)))
    train_set, valid = generate(hard, paths, kinematics="finite").split(0.25)
    model = Standard(hidden=32, internal=8)
    train(model, train_set, valid, epochs=300, batch=16, rotate=True)
    gru = Recurrent(hidden=32, strain="green-lagrange")
    train(gru, train_set, valid, epochs=300, batch=16, rotate=True)
    lagrangian = Standard(hidden=32, internal=8, strain="green-lagrange")
    train(lagrangian, train_set, valid, epochs=100, batch=16, rotate=True)
    logarithmic = Recurrent(hidden=32)
    train(logarithmic, train_set, valid, epochs=300, batch=16, rotate=True)
    error = errors(model, valid)["relative"]
    # 11.4 % on the development machine (Windows), 14.4 % on Linux (the round-off of the
    # training differs between machines; the seeds 0 to 2 spread from 10.7 to 14.4 %); the
    # recurrent network of ln C 11.2 %, both laws of E 64 %
    assert max(error, errors(logarithmic, valid)["relative"]) < 0.2 < 0.4 < min(
        errors(gru, valid)["relative"], errors(lagrangian, valid)["relative"])  # fmt: skip
    h = model.history(valid.strain, valid.times)
    assert h["dissipation"].min() >= 0.0
    c = h["tangent"][:, 1:]
    assert np.array_equal(c, np.swapaxes(c, -1, -2))
    tests = {
        "cycle": "t=0,1,3,4 n=8,16,8 E11=0,0.1,-0.1,0 E12=0 E23=0 E31=0",  # uniaxial stress
        "shear": "t=0,1,3 n=8,16 E23=0,0.15,-0.15 E12=0 E31=0",  # not driven by the data
        "turn": "t=0,1,2 n=8,8 E11=0,0.06,0.06 E12=0,0,0.12 E23=0 E31=0",
    }
    learned = path_errors(model, hard, tests, "finite")
    # measured 19 %, 11 % and 29 %, the strains 0.6 %, 0.8 % and 0.4 % (on Linux 22 %, 15 % and
    # 35 %, the strains 0.9 %, 4.0 % and 0.4 %; over the seeds 0 to 2, the strains up to 2.1,
    # 4.0 and 0.4 %)
    for name, most in (("cycle", 0.3), ("shear", 0.2), ("turn", 0.4)):
        assert learned[name]["stress"] < most and learned[name]["strain"] < 0.05, name
    # measured 11 %, 10 % and 12 %, the strains 2.2 %, 6.8 % and 1.2 % (in E: no solution
    # along the cycle and the shear, 34 % along the turn)
    recurrent = path_errors(logarithmic, hard, tests, "finite")
    for name in tests:
        assert recurrent[name]["stress"] < 0.2 and recurrent[name]["strain"] < 0.15, name
    path = LoadingPath.parse(tests["cycle"])
    ref = simulate_point(Material.from_config(hard), path, kinematics="finite")
    got = simulate_point(learned_material(model, **ELASTIC), path, kinematics="finite")
    h = model.history(green_lagrange(gradient_tensor(got.strain)), got.times)
    p = ref.plastic[-1]
    assert h["dissipation"].min() >= 0.0
    # measured 180 MPa between 77 and 225 MPa
    assert yield_ * p < h["dissipation"].sum() < yield_ * p + 0.5 * hardening * p * p


def test_networks_in_the_logarithmic_strain(plastic_data, finite_data):
    """The perceptron, the potential and the recurrent network of the logarithmic strain in
    finite strains (the default), whatever their parameters, along gradients to 15 %: their
    tangents ``dS/dE`` the derivatives of their stresses (of the steps of the recurrent
    network, whose steps of points are the recurrence of its paths), that of the potential
    symmetric to the bit and its stress the derivative of its energy; at their points
    ``dP/dF`` that of ``P = F S``, and a rotation superposed on the gradients turns ``P``
    and its tangent (the state carried along two steps); the small strain model of the same
    parameters recovered at small deformations, to the first order of the gradient; a
    recurrent law of ``ln C`` along a tension at a material point (the lateral stresses
    solved for with its tangent) and on voxels."""
    e = 1.5 * finite_data.strain[0, 5:8]
    rng = np.random.default_rng(6)
    rot = rotations(4, rng)
    grads = [0.12 * rng.normal(size=(4, 9)) for _ in range(2)]

    def spun(g):  # R F - I
        return (np.einsum("nij,njk->nik", rot, np.eye(3) + g.reshape(4, 3, 3))
                - np.eye(3)).reshape(4, 9)  # fmt: skip

    for model in (FeedForward(hidden=(5, 4)), Potential(hidden=(5, 4)),
                  Recurrent(hidden=5, rate=True)):  # fmt: skip
        model = _perturbed(model, finite_data)
        assert model.options["strain"] == "logarithmic" and model.logarithmic
        if model.recurrent:
            e0, e1 = 1.5 * finite_data.strain[0, 3:5], 1.5 * finite_data.strain[0, 4:6]
            h0 = np.random.default_rng(0).normal(size=(2, 5))
            c = model.step(e1, e0, h0, 0.1)[1]
            fd = _jacobian(lambda x, e0=e0, h0=h0, m=model: m.step(x, e0, h0, 0.1)[0], e1)
            out = model.predict(finite_data.strain[:2], finite_data.times[:2])
            state, steps = np.zeros((2, 5)), []
            for t in range(1, finite_data.n_points):
                dt = finite_data.times[0, t] - finite_data.times[0, t - 1]
                s, _, state = model.step(finite_data.strain[:2, t],
                                         finite_data.strain[:2, t - 1], state, dt)  # fmt: skip
                steps.append(s)
            assert np.abs(np.stack(steps, axis=1) - out[:, 1:]).max() <= 1e-12 * np.abs(out).max()
        else:
            c, fd = model.tangent(e), _jacobian(model.predict, e)
            s, c1, _ = model.step(e, e, np.zeros((3, 0)), 1.0)
            # S = J^T T by the kernel of the step and by NumPy's einsum in predict: equal to
            # round-off (to the bit on x86, not on macOS arm64); the tangents the same kernel's
            assert np.abs(s - model.predict(e)).max() <= 1e-12 * np.abs(s).max()
            assert np.array_equal(c1, c)
        assert np.abs(c - fd).max() < 1e-6 * np.abs(fd).max(), model.kind  # 1.7e-9 measured
        if model.kind == "potential":
            assert np.array_equal(c, np.swapaxes(c, 1, 2))
            fd = _jacobian(model.energy, e)
            assert np.abs(fd - model.predict(e)).max() < 1e-6 * np.abs(fd).max()
        points, law = _points(model, np.repeat(np.eye(3)[None], 4, axis=0), finite=True)
        a = _steps(points, law, grads)
        for (p, t), (p1, t1) in zip(a, _steps(points, law, [spun(g) for g in grads]),
                                    strict=True):  # fmt: skip
            turned = np.einsum("nij,njk->nik", rot, p.reshape(4, 3, 3)).reshape(4, 9)
            assert np.abs(p1 - turned).max() < 1e-12 * np.abs(p).max()
            t4 = np.einsum("nia,nkc,naJcL->niJkL", rot, rot, t.reshape(4, 3, 3, 3, 3))
            assert np.abs(t1 - t4.reshape(4, 9, 9)).max() < 1e-12 * np.abs(t).max()
        fd = _jacobian(lambda g, p=points, w=law: _steps(p, w, [grads[0], g])[1][0], grads[1])
        assert np.abs(a[1][1] - fd).max() < 1e-6 * np.abs(fd).max(), model.kind
    # the small strain model of the same parameters at gradients lam h (h: its data, to 1 %),
    # its scales lam s: the same normalized response but for the geometric terms
    for small in (FeedForward(hidden=(5, 4)), Potential(hidden=(5, 4)), Recurrent(hidden=5)):
        small = _perturbed(small, plastic_data, seed=5)
        times = (plastic_data.times[:4],) if small.recurrent else ()
        gaps = []
        for lam in (1.0, 1e-2):
            s0 = small.scales
            scaled = copy.deepcopy(small)
            scaled.scales = Scales(lam * s0.strain, lam * s0.stress, lam * s0.increment, s0.time)
            finite = copy.deepcopy(scaled)
            finite.measure = "green-lagrange"
            g = lam * plastic_data.strain[:4]
            f = gradient_tensor(g)
            pk2 = finite.predict(green_lagrange(f), *times)
            cauchy = np.einsum("ptij,ptjk,ptlk->ptil", f, _tensor(pk2, False), f)
            cauchy = _voigt(cauchy / np.linalg.det(f)[..., None, None], False)
            ref = scaled.predict(g, *times)
            gaps.append(np.abs(cauchy - ref).max() / np.abs(ref).max())
        # measured 3.5 %, 4.0 % and 2.8 % at lam = 1, a hundred times less at 1e-2
        assert gaps[0] < 0.2 and 0.005 < gaps[1] / gaps[0] < 0.02, small.kind
    # in the solvers: a recurrent law of ln C at a material point (its exact tangent), a
    # potential of ln C there, on voxels and by finite elements (its symmetric tangent)
    tension = LoadingPath.parse("t=0,1,2 n=8,8 E11=0,0.08,0.02")  # the other stresses free
    recurrent = Recurrent(hidden=6)
    train(recurrent, finite_data, epochs=20, batch=4)
    point = simulate_point(learned_material(recurrent, **ELASTIC), tension, kinematics="finite")
    assert np.abs(point.stress[:, 1:]).max() < 1e-6 * np.abs(point.stress).max()
    potential = Potential(hidden=(6,))
    train(potential, finite_data, epochs=20, batch=16)
    material = learned_material(potential, **ELASTIC)
    point = simulate_point(material, tension, kinematics="finite")
    size = np.abs(point.stress).max()
    assert np.abs(point.stress[:, 1:]).max() < 1e-6 * size
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    voxels = NonlinearFFTProblem(image, {0: material}, kinematics="finite").run(tension)
    elements = NonlinearFEProblem(box_mesh((2, 2, 2), perturb=0.2), {0: material},
                                  kinematics="finite").run(tension)  # fmt: skip
    for got in (voxels, elements):
        assert np.abs(got.stress - point.stress).max() < 1e-6 * size
        assert np.abs(got.strain - point.strain).max() < 1e-6 * np.abs(point.strain).max()


def test_files_before_the_option_strain(tmp_path, finite_data):
    """The models see the logarithmic strain in finite strains by default; the files written
    before the option (their options without it) load as the laws of the Green-Lagrange
    strain they were: their stresses, tangents and states to the bit those of the models of
    ``strain="green-lagrange"``, whose networks are those of the small strains (``E`` in,
    ``S`` out)."""
    assert all(cls().options["strain"] == "logarithmic"
               for cls in (FeedForward, Potential, Recurrent, Standard))  # fmt: skip
    e = finite_data.strain[:2]
    for model in (FeedForward(hidden=(4,), strain="green-lagrange"),
                  Potential(hidden=(4,), strain="green-lagrange"),
                  Recurrent(hidden=4, strain="green-lagrange")):  # fmt: skip
        train(model, finite_data, epochs=2, batch=4)
        path = model.save(tmp_path / model.kind)
        with np.load(path) as f:  # the file without the option
            meta = json.loads(str(f["metadata"]))
            arrays = {k: f[k] for k in f.files if k != "metadata"}
        del meta["options"]["strain"]
        np.savez(path, metadata=np.array(json.dumps(meta)), **arrays)
        old = load_model(path)
        assert old.options == model.options and not old.logarithmic
        times = (finite_data.times[:2],) if model.recurrent else ()
        assert np.array_equal(old.predict(e, *times), model.predict(e, *times))
        h0 = np.full((2, model.n_state), 0.1)
        for a, b in zip(old.step(e[:, 3], e[:, 2], h0, 1.0), model.step(e[:, 3], e[:, 2], h0, 1.0),
                        strict=True):  # fmt: skip
            assert np.array_equal(a, b)
        plain = copy.deepcopy(old)
        plain.measure = "small"
        assert np.array_equal(plain.predict(e, *times), old.predict(e, *times))
        log = copy.deepcopy(old)
        log.options["strain"] = "logarithmic"
        assert log.logarithmic
        assert not np.allclose(log.predict(e, *times), old.predict(e, *times), rtol=1e-3)


def test_reproducible_and_saved(tmp_path, plastic_data):
    train_set, valid = plastic_data.split(0.25, seed=1)

    def fitted(seed):
        model = Recurrent(hidden=6, seed=seed)
        train(model, train_set, valid, epochs=5, batch=4, rotate=True, seed=seed)
        return model

    a, b, c = fitted(0), fitted(0), fitted(1)
    assert all(np.array_equal(a.params[k], b.params[k]) for k in a.params)
    assert not np.allclose(a.params["Wi"], c.params["Wi"])
    path = a.save(tmp_path / "model")
    assert path.suffix == ".npz"
    loaded = load_model(path)
    assert loaded.kind == "recurrent" and loaded.options == a.options
    assert loaded.metadata["training"]["epochs"] == 5 and loaded.scales == a.scales
    assert np.array_equal(loaded.predict(valid.strain, valid.times),
                          a.predict(valid.strain, valid.times))  # fmt: skip
    # a material of the file, its configuration; its points are the steps of the model
    material = Material.from_config({**ELASTIC, "learned": str(path)})
    assert material.nonlinear
    assert Material.from_config(material.to_config()).learned.model == str(path)
    spec = "t=0,1,2 n=4 E11=0,0.004,-0.002 E22=0 E33=0 E23=0 E31=0 E12=0,0.002,0.003"
    res = simulate_point(material, LoadingPath.parse(spec))
    direct = loaded.predict(res.strain, res.times)
    assert np.allclose(res.stress, direct, rtol=0, atol=1e-10 * np.abs(direct).max())
    again = Dataset.load(plastic_data.save(tmp_path / "data.npz"))
    assert np.array_equal(again.stress, plastic_data.stress) and again.measure == "small"


def test_learned_law_on_voxels():
    data = generate(ELASTIC, random_paths(8, segments=2, increments=3, seed=0))
    model = FeedForward(hidden=(4,))
    train(model, data, epochs=30, batch=16)
    image = VoxelImage(np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=np.uint8))
    stiff = Material(young=400000.0, poisson=0.25)
    path = LoadingPath.parse("t=0,1 n=2 E11=0.001")
    ref = NonlinearFFTProblem(image, {0: Material.from_config(ELASTIC), 1: stiff}).run(path)
    problem = NonlinearFFTProblem(image, {0: learned_material(model, **ELASTIC), 1: stiff})
    got = problem.run(path)
    assert np.abs(got.stress - ref.stress).max() < 0.01 * np.abs(ref.stress).max()
    # the homogenized responses of a microstructure as data
    micro = generate(problem.run, random_paths(2, segments=1, increments=2, seed=0))
    assert micro.stress.shape == (2, 3, 6) and np.abs(micro.stress).max() > 0


def test_learned_errors(plastic_data):
    model = FeedForward(hidden=(4,))
    train(model, plastic_data, epochs=1)
    with pytest.raises(ValueError, match="one of them"):
        Material.from_config({**PLASTIC, "learned": LearnedBehaviour(model)})
    with pytest.raises(ValueError, match="elastic parameters"):
        Material(conductivity=1.0, learned=model)
    with pytest.raises(ValueError, match="no learned law"):
        Material(void=True, learned=model)
    with pytest.raises(ValueError, match="not trained"):
        LearnedBehaviour(FeedForward())
    material = learned_material(model, **ELASTIC)
    with pytest.raises(ValueError, match="green-lagrange"):
        simulate_point(material, LoadingPath.parse("t=0,1 n=2 E11=0.001"), kinematics="finite")
    # without an expansion, a change of temperature leaves the law as it is
    path = LoadingPath.parse("t=0,1 n=2 E11=0.001")
    heated = LoadingPath.parse("t=0,1 n=2 E11=0.001 dT=0,10")
    assert np.array_equal(simulate_point(material, heated).stress,
                          simulate_point(material, path).stress)  # fmt: skip
    with pytest.raises(FileNotFoundError):
        Material(**ELASTIC, learned="missing.npz").learned.load()
    with pytest.raises(ValueError, match="rotate"):
        train(model, plastic_data, epochs=1, rotate="hexagonal")
    with pytest.raises(ValueError, match="hidden"):
        Recurrent(hidden=[8, 8])
    with pytest.raises(ValueError, match="hidden, internal"):  # from a list of the command
        Standard(hidden=[4, 4])
    with pytest.raises(ValueError, match="smoothing"):
        Standard(smoothing=float("inf"))
    for cls in (FeedForward, Potential, Recurrent):
        with pytest.raises(ValueError, match="strain"):
            cls(strain="hencky")


def _linear(data):
    """The least-squares fit of a perceptron set up on ``data``, its network silenced: the
    linear law of the data, exact for elastic data."""
    model = FeedForward(hidden=(2,))
    model.setup(data)
    model.params["Wout"][:] = 0.0
    return model


def _points(model, rot, finite=False, expansion=None):
    """The points of a learned law, each turned by its rotation (n, 3, 3), of the thermal
    ``expansion`` (3 x 3, in the axes of the material); their exact tangents."""
    n = len(rot)
    law = learned_law(LearnedBehaviour(model), finite=finite, expansion=expansion)
    points = LearnedPoints([law], np.full(n, MFRONT), np.zeros(n, dtype=int), np.arange(n), rot,
                           finite, symmetric=False)  # fmt: skip
    return points, law


def _steps(points, law, grads, dt=1.0, dtemp=0.0):
    """The stresses and tangents of the points along steps of their strains (or
    gradients), the state of each step carried to the next, at the change of temperature
    ``dtemp``."""
    n, nc = grads[0].shape[0], 9 if points.finite else 6
    old, out = np.zeros((n, law.width)), []
    for g in grads:
        new, s, c = np.zeros_like(old), np.zeros((n, nc)), np.zeros((n, nc, nc))
        assert points.integrate(g, dtemp, dt, old, new, s, c) == 0
        out.append((s, c))
        old = new
    return out


def test_learned_laws_turned_with_their_material():
    """A law learned in the axes of an orthotropic material whose three shear moduli differ,
    turned by an orientation, is the material turned by it: at a material point along a
    path of the six components and in its tangent, to round-off (the shears in morphoGen's
    order 23, 31, 12 at every interface: the data, the model, the rotations, the points);
    not oriented, it stays in the axes of the sample."""
    crystal = Material.from_config({"stiffness": ORTHOTROPIC.tolist()})
    data = generate(crystal, random_paths(6, segments=2, increments=3, seed=0))
    model = _linear(data)
    scale = ORTHOTROPIC.max()
    assert np.abs(model.tangent(np.zeros(6)) - ORTHOTROPIC).max() < 1e-9 * scale
    euler = [30.0, 50.0, 70.0]
    path = LoadingPath.parse(SIX)
    ref = simulate_point(crystal, path, orientation=euler)
    turned = crystal.matrix("elastic", euler)
    size = np.abs(ref.stress).max()
    assert np.abs(ref.stress - ref.strain @ turned.T).max() < 1e-9 * size
    got = simulate_point(learned_material(model), path, orientation=euler)
    assert np.abs(got.stress - ref.stress).max() < 1e-9 * size
    # its tangent at a point: the stiffness turned, T^T C T
    points, law = _points(model, euler_to_matrix(np.array([euler])))
    ((_, tangent),) = _steps(points, law, [ref.strain[[2]]])
    assert np.abs(tangent[0] - turned).max() < 1e-9 * scale
    # not oriented: the law in the axes of the sample, whatever the orientation
    fixed = Material(**ELASTIC, learned=LearnedBehaviour(model, oriented=False))
    assert not fixed.oriented("elastic") and learned_material(model).oriented("elastic")
    spec = {"model": "law.npz", "oriented": False}
    assert LearnedBehaviour.from_config(spec).to_config() == spec
    assert LearnedBehaviour.from_config("law.npz").to_config() == {"model": "law.npz"}
    plain = simulate_point(fixed, path, orientation=euler)
    assert np.abs(plain.stress - plain.strain @ ORTHOTROPIC.T).max() < 1e-9 * size


def test_turned_laws_equivariant(plastic_data):
    """A learned law turned by ``g`` is the law of the turned strain, turned back, to
    round-off (the rotations as 3 x 3 tensors): a recurrent law along steps, its state
    carried, and a potential of the Green-Lagrange strain and of the logarithmic strain in
    finite strains (``P = F S``, ``dP/dF``); the tangents of the turned laws are the
    derivatives of their stresses."""
    rng = np.random.default_rng(4)
    rot = rotations(5, rng)
    same = np.repeat(np.eye(3)[None], 5, axis=0)
    model = _perturbed(Recurrent(hidden=5), plastic_data)
    turned, law = _points(model, rot)
    plain, _ = _points(model, same)
    steps = list(0.003 * rng.normal(size=(2, 5, 6)))

    def inside(e):  # g eps g^T, engineering shears
        return _voigt(np.einsum("nij,njk,nlk->nil", rot, _tensor(e, True), rot), True)

    a, b = _steps(turned, law, steps), _steps(plain, law, [inside(e) for e in steps])
    for (s, _), (s0, _) in zip(a, b, strict=True):
        back = _voigt(np.einsum("nji,njk,nkl->nil", rot, _tensor(s0, False), rot), False)
        assert np.abs(s - back).max() < 1e-12 * np.abs(s).max()

    def second(e):  # the stress of the second step at its strain e (the first one held)
        return _steps(turned, law, [steps[0], e])[1][0]

    fd = _jacobian(second, steps[1])
    assert np.abs(a[1][1] - fd).max() < 1e-6 * np.abs(fd).max()
    # finite strains: P(g; F) = g^T P(I; g F g^T) g, its tangent turned as a fourth-order tensor
    paths = random_paths(3, segments=1, increments=2, amplitude=0.02, seed=1)
    elastic = generate(ELASTIC, paths, kinematics="finite")
    f = np.eye(3) + 0.05 * rng.normal(size=(5, 3, 3))
    for strain in STRAINS:
        potential = _perturbed(Potential(hidden=(5, 4), strain=strain), elastic)
        turned, law = _points(potential, rot, finite=True)
        plain, _ = _points(potential, same, finite=True)
        ((p, a),) = _steps(turned, law, [(f - np.eye(3)).reshape(5, 9)])
        ((p0, a0),) = _steps(plain, law, [(np.einsum("nij,njk,nlk->nil", rot, f, rot)
                                           - np.eye(3)).reshape(5, 9)])  # fmt: skip
        back = np.einsum("nai,naB,nBJ->niJ", rot, p0.reshape(5, 3, 3), rot).reshape(5, 9)
        assert np.abs(p - back).max() < 1e-12 * np.abs(p).max(), strain
        a4 = np.einsum("nai,nBJ,nck,nDL,naBcD->niJkL", rot, rot, rot, rot,
                       a0.reshape(5, 3, 3, 3, 3))  # fmt: skip
        assert np.abs(a - a4.reshape(5, 9, 9)).max() < 1e-12 * np.abs(a).max(), strain


def test_learned_crystal_in_a_polycrystal():
    """The law of a cubic crystal learned in its axes (its linear fit), turned grain by
    grain: the homogenized response of a polycrystal on voxels is that of its crystals, to
    the tolerance of the solver; not oriented, every grain in the axes of the sample, it is
    not."""
    from morphogen.domain import Domain
    from morphogen.models.base import model_from_config

    cubic = Material.from_config({"cubic": [168400.0, 121400.0, 75400.0]})  # copper
    model = _linear(generate(cubic, random_paths(6, segments=2, increments=3, seed=0)))
    voronoi = model_from_config({"type": "voronoi", "n_cells": 6, "crystal_orientations": True})
    micro = voronoi.generate(Domain(size=1.0, periodic=True), 3)
    image = micro.voxelize(shape=6)
    grains = {"grains": micro.labels(grid=image.grid()), "orientations": micro.orientations}
    path = LoadingPath.parse("t=0,1 n=2 E11=0.001")  # the other stresses free

    def run(material):
        return NonlinearFFTProblem(image, {1: material}, tol=1e-8, **grains).run(path)

    ref = run(cubic)
    got = run(learned_material(model))
    size = np.abs(ref.stress).max()
    assert np.abs(got.stress - ref.stress).max() < 1e-6 * size
    assert np.abs(got.strain - ref.strain).max() < 1e-6 * np.abs(ref.strain).max()
    fixed = run(Material(**ELASTIC, learned=LearnedBehaviour(model, oriented=False)))
    assert np.abs(fixed.strain - ref.strain).max() > 0.01 * np.abs(ref.strain).max()


@pytest.mark.slow
def test_learned_crystal_plasticity_in_a_polycrystal():
    """Méric-Cailletaud crystal plasticity learned at a material point in the axes of the
    crystal (the paths turned by the rotations of the cube), turned grain by grain: the
    homogenized tension of a polycrystal within the error of the law at a point."""
    from morphogen.domain import Domain
    from morphogen.models.base import model_from_config

    crystal = {"cubic": [250000.0, 160000.0, 120000.0],
               "crystal_plasticity": {"families": ["octahedral"], "tau0": 80.0, "Q": 20.0,
                                      "b": 10.0, "K": 25.0, "n": 10.0}}  # fmt: skip
    paths = random_paths(256, segments=4, increments=8, amplitude=0.01, rate=1e-3, seed=0)
    train_set, valid = generate(crystal, paths).split(0.25)
    model = Recurrent(hidden=32, rate=True)
    train(model, train_set, valid, epochs=300, batch=16, rotate="cubic")
    point = errors(model, valid)["relative"]
    assert point < 0.08  # 5.2 % measured (10.4 % without the rotations)
    voronoi = model_from_config({"type": "voronoi", "n_cells": 8, "crystal_orientations": True})
    micro = voronoi.generate(Domain(size=1.0, periodic=True), 11)
    image = micro.voxelize(shape=8)
    grains = {"grains": micro.labels(grid=image.grid()), "orientations": micro.orientations}
    path = LoadingPath.parse("t=0,10 n=20 E11=0,0.01")  # 1e-3 /s, the other stresses free
    ref = NonlinearFFTProblem(image, {1: Material.from_config(crystal)}, **grains).run(path)
    got = NonlinearFFTProblem(image, {1: learned_material(model)}, **grains).run(path)
    a, b = got.stress[:, 0], ref.stress[:, 0]
    assert np.sqrt(np.sum((a - b) ** 2) / np.sum(b**2)) < 2 * point  # 6.7 % measured


def test_thermal_strains_of_learned_laws():
    """A learned law sees the mechanical strain, the stress-free thermal strain of its
    material taken out: heated free of stress it expands by alpha dT without a stress, in
    finite strains by exp(alpha dT) as morphoGen's elastic points do; constrained, and
    along a heated tension, it gives the stresses of the material (exact for the linear
    law of elastic data; in finite strains that of ln C, Hencky's law, gives the stresses
    of the Hencky points, whose split it takes); a crystal of three expansions turned by
    an orientation expands as the crystal does."""
    alpha, heat = 1.2e-5, 300.0
    model = _linear(generate(ELASTIC, random_paths(6, segments=2, increments=3, seed=0)))
    reference = Material.from_config({**ELASTIC, "expansion": alpha})
    learned = learned_material(model, **ELASTIC, expansion=alpha)
    free = LoadingPath.parse(f"t=0,1 n=2 dT=0,{heat}")  # every stress free
    got = simulate_point(learned, free)
    assert np.allclose(got.strain[-1], [alpha * heat] * 3 + [0.0] * 3, rtol=1e-9, atol=1e-15)
    thermal = ELASTIC["young"] * alpha * heat
    assert np.abs(got.stress).max() < 1e-9 * thermal
    for spec in ("t=0,1 n=2 E11=0 E22=0 E33=0 E23=0 E31=0 E12=0 dT=0,300",
                 "t=0,1,2 n=3 E11=0,0.002,0.001 dT=0,200,300"):  # fmt: skip
        path = LoadingPath.parse(spec)
        a, b = simulate_point(learned, path), simulate_point(reference, path)
        assert np.abs(a.stress - b.stress).max() < 1e-8 * np.abs(b.stress).max()
        assert np.abs(a.strain - b.strain).max() <= 1e-8 * np.abs(b.strain).max()
    # finite strains: F = exp(alpha dT) I, as morphoGen's elastic points (Hencky's)
    finite_data = generate(ELASTIC, random_paths(6, segments=2, increments=3, seed=0),
                           kinematics="finite")  # fmt: skip
    hot = learned_material(_linear(finite_data), **ELASTIC, expansion=alpha)
    a = simulate_point(hot, free, kinematics="finite")
    b = simulate_point(reference, free, kinematics="finite")
    assert np.allclose(a.strain[-1, :3], np.expm1(alpha * heat), rtol=1e-9)
    assert np.allclose(b.strain, a.strain, rtol=1e-8, atol=1e-15)
    # the linear law of ln C is Hencky's law: the stresses of the Hencky points. The learned
    # laws now split the heated strain as they do (1/2 ln C - alpha dT, the free energy per
    # unit reference volume); with the multiplicative split of before (S = J_th A S_m A, the
    # free energy per unit volume of the heated configuration) this law gave J_th = exp(3
    # alpha dT) times their stresses, 1.1 % at 300 K, which this test encoded
    for spec in ("t=0,1 n=2 E11=0 E22=0 E33=0 E23=0 E31=0 E12=0 dT=0,300",
                 "t=0,1,2 n=3 E11=0,0.05,0.02 dT=0,200,300"):  # fmt: skip
        path = LoadingPath.parse(spec)
        a = simulate_point(hot, path, kinematics="finite")
        b = simulate_point(reference, path, kinematics="finite")
        assert np.abs(a.stress - b.stress).max() < 1e-12 * np.abs(b.stress).max()
        assert np.abs(a.strain - b.strain).max() <= 1e-12 * np.abs(b.strain).max()
    # a crystal of three expansions turned by an orientation, free and constrained
    principal, euler = [1e-5, 2e-5, 3e-5], [30.0, 50.0, 70.0]
    crystal = Material.from_config({"stiffness": ORTHOTROPIC.tolist(), "expansion": principal})
    turned = learned_material(_linear(generate({"stiffness": ORTHOTROPIC.tolist()},
                                               random_paths(6, segments=2, increments=3))),
                              expansion=principal)  # fmt: skip
    for spec in (f"t=0,1 n=2 dT=0,{heat}",
                 "t=0,1 n=2 E11=0 E22=0 E33=0 E23=0 E31=0 E12=0 dT=0,300"):  # fmt: skip
        path = LoadingPath.parse(spec)
        a = simulate_point(turned, path, orientation=euler)
        b = simulate_point(crystal, path, orientation=euler)
        assert np.abs(a.strain - b.strain).max() <= 1e-8 * max(np.abs(b.strain).max(), 1e-3)
        assert np.abs(a.stress - b.stress).max() <= 1e-8 * max(np.abs(b.stress).max(), thermal)


class _HenckyOfE(Model):
    """Hencky's elasticity as a law of the Green-Lagrange strain, ``S = J^T C H`` (``H =
    1/2 ln C``) and its exact tangent: a learned law of ``E`` that is Hencky's law."""

    kind = "hencky"

    def __init__(self, stiffness):
        super().__init__(strain="green-lagrange")
        self.params = {"C": np.asarray(stiffness, dtype=float)}
        self.scales = Scales(1.0, 1.0, 1.0, 1.0)
        self.measure = "green-lagrange"

    def step(self, e_new, e_old, h_old, dt):
        x = np.ascontiguousarray(e_new, dtype=float)
        c = self.params["C"]
        k = np.ascontiguousarray(np.broadcast_to(c, (x.shape[0], 6, 6)))
        s, t = _pullback(x, np.ascontiguousarray(_measures(x, 1.0)[0] @ c.T), k, 1.0, True)
        return s, t, np.zeros((x.shape[0], 0))


def test_heated_laws_as_the_hencky_points(finite_data):
    """Heated in finite strains, a learned law splits the strain as morphoGen's Hencky
    points do (the mechanical logarithmic strain ``1/2 ln C - alpha dT``, the stress
    conjugate to ``1/2 ln C``, the free energy per unit reference volume): a law that is
    Hencky's elasticity exactly, of ``ln C`` or of ``E``, gives their stresses to round-off
    with an isotropic expansion and with the three expansions of an orthotropic crystal
    turned by an orientation, under constraint, along a heated gradient of six components
    and along a heated tension, at a material point, on voxels and by finite elements;
    heated free, no stress and the logarithmic strain the thermal strain. Every model in
    both strains, its parameters moved, turned with an anisotropic expansion and heated:
    ``dP/dF`` the derivative of ``P``; the potential and the standard model free of stress
    at the free thermal stretch. Unheated, or without an expansion, the law of before to
    the bit."""
    alpha, heat = 1.2e-5, 300.0
    principal, euler = [1e-5, 2e-5, 3e-5], [30.0, 50.0, 70.0]
    ortho = {"stiffness": ORTHOTROPIC.tolist()}
    walks = random_paths(6, segments=2, increments=3, amplitude=0.03, seed=0)
    paths = [LoadingPath.parse(spec) for spec in (
        "t=0,1 n=2 E11=0 E22=0 E33=0 E23=0 E31=0 E12=0 dT=0,300",
        "t=0,1 n=2 E11=0,0.05 E22=0,-0.02 E33=0,0.03 E23=0,0.04 E31=0,-0.03 E12=0,0.06 dT=0,300",
        "t=0,1,2 n=3 E11=0,0.05,0.02 dT=0,200,300")]  # fmt: skip
    free = LoadingPath.parse(f"t=0,1 n=2 dT=0,{heat}")
    hot = []
    for spec, stiffness, expansion, orientation in (
        (ELASTIC, _isotropic(**ELASTIC), alpha, None),
        (ortho, ORTHOTROPIC, principal, euler),
    ):
        reference = Material.from_config({**spec, "expansion": expansion})
        silenced = Potential(hidden=(1,))  # of ln C: its start, the linear fit of T to H
        silenced.setup(generate(spec, walks, kinematics="finite"))
        silenced.params["v"][:] = -60.0
        for model in (silenced, _HenckyOfE(stiffness)):
            learned = learned_material(model, **ELASTIC, expansion=expansion)
            for path in paths:  # measured 1e-16 to 8e-15
                a = simulate_point(learned, path, kinematics="finite", orientation=orientation)
                b = simulate_point(reference, path, kinematics="finite", orientation=orientation)
                assert np.abs(a.stress - b.stress).max() < 1e-12 * np.abs(b.stress).max()
                assert np.abs(a.strain - b.strain).max() <= 1e-12 * np.abs(b.strain).max()
            a = simulate_point(learned, free, kinematics="finite", orientation=orientation)
            angles = None if orientation is None else np.asarray(orientation)
            thermal = _voigt(heat * reference.expansion_tensor(angles), True)
            assert np.abs(a.stress).max() < 1e-9 * ELASTIC["young"] * alpha * heat
            gap = log_strain(gradient_of(a.strain[-1])) - thermal
            assert np.abs(gap).max() < 1e-10 * np.abs(thermal).max()  # 1.5e-15 measured
        hot.append(learned_material(silenced, **ELASTIC, expansion=expansion))
    # homogeneous cells: on voxels the crystal turned by its grain, by finite elements the
    # isotropic material, the response of the material point (measured 2e-10 and 6e-16)
    tension = paths[2]
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    voxels = NonlinearFFTProblem(image, {0: hot[1]}, grains=np.zeros((2, 2, 2), dtype=int),
                                 orientations=[euler], kinematics="finite",
                                 newton_tol=1e-9).run(tension)  # fmt: skip
    elements = NonlinearFEProblem(box_mesh((2, 2, 2), perturb=0.2), {0: hot[0]},
                                  kinematics="finite", newton_tol=1e-9).run(tension)  # fmt: skip
    for got, material, orientation in ((voxels, hot[1], euler), (elements, hot[0], None)):
        point = simulate_point(material, tension, kinematics="finite", orientation=orientation)
        assert np.abs(got.stress - point.stress).max() < 1e-6 * np.abs(point.stress).max()
        assert np.abs(got.strain - point.strain).max() < 1e-6 * np.abs(point.strain).max()
    # every model, both strains: the tangent under heating, no stress at the free stretch
    # F = exp(dT g^T alpha g) of a law free of stress at the natural state, and the law of
    # before to the bit unheated or without an expansion
    rng = np.random.default_rng(8)
    rot = rotations(3, rng)
    tensor = np.diag(principal)
    grads = [0.1 * rng.normal(size=(3, 9)) for _ in range(2)]
    values, vectors = np.linalg.eigh(heat * np.einsum("nji,jk,nkl->nil", rot, tensor, rot))
    stretch = np.einsum("nij,nj,nkj->nik", vectors, np.exp(values), vectors) - np.eye(3)
    stretch = stretch.reshape(3, 9)
    for strain in STRAINS:
        for model in (FeedForward(hidden=(5, 4), strain=strain),
                      Potential(hidden=(5, 4), strain=strain), Recurrent(hidden=5, strain=strain),
                      Standard(hidden=6, internal=4, strain=strain)):  # fmt: skip
            model = _perturbed(model, finite_data, seed=5)
            points, law = _points(model, rot, finite=True, expansion=tensor)
            a = _steps(points, law, grads, dtemp=heat)
            fd = _jacobian(lambda g, p=points, w=law: _steps(p, w, [grads[0], g], dtemp=heat)[1][0],
                           grads[1])  # fmt: skip
            # measured 3.4e-9 to 7.1e-9
            assert np.abs(a[1][1] - fd).max() < 1e-6 * np.abs(fd).max(), (model.kind, strain)
            if model.kind in ("potential", "standard"):  # 4e-14 of the constrained stress
                ((p, _),) = _steps(points, law, [stretch], dtemp=heat)
                ((p0, _),) = _steps(points, law, [np.zeros((3, 9))], dtemp=heat)
                assert np.abs(p).max() < 1e-12 * np.abs(p0).max(), (model.kind, strain)
            plain, _ = _points(model, rot, finite=True)
            before = _steps(plain, law, grads)
            for other in (_steps(points, law, grads), _steps(plain, law, grads, dtemp=heat)):
                for (s, c), (s0, c0) in zip(other, before, strict=True):
                    assert np.array_equal(s, s0) and np.array_equal(c, c0)


def test_heated_points_fail_where_c_is_singular(finite_data):
    """Heated in finite strains, a learned point whose ``C`` is not positive definite fails
    as a Hencky point does, whatever its model, which is not called there (the standard
    model of ``E`` raised ``ZeroDivisionError``): its stress is not finite and counted, the
    other points as integrated alone; at a material point compressed to ``F11 = 0`` while
    heated, the solver divides its step and stops as for the Hencky points."""
    tensor = 1.2e-5 * np.eye(3)
    rot = np.repeat(np.eye(3)[None], 2, axis=0)
    grads = np.zeros((2, 9))
    grads[0, 0], grads[1, 0] = -1.0, -0.02  # F11 = 0 (C singular), and a compression
    for strain in STRAINS:
        for model in (FeedForward(hidden=(5, 4), strain=strain),
                      Potential(hidden=(5, 4), strain=strain), Recurrent(hidden=5, strain=strain),
                      Standard(hidden=6, internal=4, strain=strain)):  # fmt: skip
            model = _perturbed(model, finite_data, seed=5)
            points, law = _points(model, rot, finite=True, expansion=tensor)
            old = np.zeros((2, law.width))
            new, s, c = np.zeros_like(old), np.zeros((2, 9)), np.zeros((2, 9, 9))
            assert points.integrate(grads, 300.0, 1.0, old, new, s, c) == 1, (model.kind, strain)
            assert np.isnan(s[0]).all() and np.isnan(c[0]).all()
            alone, one = _points(model, rot[1:], finite=True, expansion=tensor)
            ((s1, _),) = _steps(alone, one, [grads[1:]], dtemp=300.0)
            assert np.abs(s[1] - s1[0]).max() <= 1e-12 * np.abs(s1).max(), (model.kind, strain)
    model = _perturbed(Standard(hidden=6, internal=4, strain="green-lagrange"), finite_data)
    path = LoadingPath.parse("t=0,1 n=1 E11=0,-1 E22=0 E33=0 E23=0 E31=0 E12=0 dT=0,300")
    for material in (learned_material(model, **ELASTIC, expansion=1.2e-5),
                     Material.from_config({**ELASTIC, "expansion": 1.2e-5})):  # fmt: skip
        with pytest.raises(RuntimeError, match=r"no convergence at time 0\.875"):
            simulate_point(material, path, kinematics="finite", max_cuts=3)


def test_learned_laws_elsewhere(plastic_data):
    """A learned law where the other inelastic behaviours go: the point bulk of the crack
    solvers, files relative to the configuration, the MFront exports (its source:
    test_learning_mfront.py)."""
    from morphogen.config import _mfront_paths
    from morphogen.fracture.bulk import nonlinear_materials
    from morphogen.io.mfront import mfront_source

    model = FeedForward(hidden=(4,))
    train(model, plastic_data, epochs=1)
    material = learned_material(model, **ELASTIC)
    assert nonlinear_materials({0: Material(**ELASTIC), 1: material}, np.array([0, 1]))
    paths = _mfront_paths({0: {**ELASTIC, "learned": "law.npz"},
                           1: {**ELASTIC, "learned": {"model": "law.npz"}}},
                          Path("conf"))  # fmt: skip
    assert paths[0]["learned"] == str(Path("conf") / "law.npz")
    assert paths[1]["learned"] == {"model": str(Path("conf") / "law.npz")}
    assert "@Behaviour Learned;" in mfront_source(material, "Learned")


def test_cli_learn(tmp_path, capsys):
    config = {"learning": {
        "material": ELASTIC,
        "data": {"paths": 6, "segments": 2, "increments": 3},
        "model": {"kind": "feedforward", "hidden": [4]},
        "training": {"epochs": 20, "batch": 8},
        "paths": {"tension": "t=0,1 n=3 E11=0.002"},
    }}  # fmt: skip
    path = tmp_path / "law.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert cli.main(["learn", str(path), "--json", "--data", str(tmp_path / "data.npz")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"] == "feedforward" and out["paths"] == 6 and out["points"] == 7
    assert out["tests"]["tension"]["stress"] < 0.02 and (tmp_path / "data.npz").is_file()
    assert load_model(tmp_path / "law.npz").metadata["data"]["source"] == ELASTIC
    args = ["learn", str(path), "--model", "potential", "--epochs", "5", "-o", str(tmp_path / "p")]
    assert cli.main(args) == 0
    assert "potential model" in capsys.readouterr().out and (tmp_path / "p.npz").is_file()
    path.write_text(json.dumps({"learning": {"material": ELASTIC, "model": {"size": 3}}}))
    assert cli.main(["learn", str(path)]) != 0
    assert "learning.model: unknown keys ['size']" in capsys.readouterr().err
