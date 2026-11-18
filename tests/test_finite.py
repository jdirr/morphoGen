"""Finite strains: the logarithmic strain space of the elastic and von Mises behaviours,
their tangents and objectivity, the macroscopic measures, and the finite element
solution of the periodic element."""

import numpy as np
import pytest
from scipy.linalg import expm

from helpers import box_mesh
from morphogen.homogenization import Material, Plasticity
from morphogen.homogenization.finite import cauchy, log_strain, update_points_finite
from morphogen.homogenization.materials import material_table
from morphogen.homogenization.nonlinear import (
    NonlinearFEProblem,
    behaviour_table,
    simulate_point,
)
from morphogen.homogenization.tensors import eigh3, log_divided, log_divided2

E, NU = 200e3, 0.3


def test_eigen_and_divided_differences():
    rng = np.random.default_rng(0)
    for scale in (1.0, 1e-9):
        a = rng.normal(size=(3, 3)) * scale
        a = a + a.T
        w, v = np.zeros(3), np.zeros((3, 3))
        eigh3(a.copy(), w, v)
        assert np.allclose(v @ np.diag(w) @ v.T, a, atol=1e-14 * scale)
        assert np.allclose(v.T @ v, np.eye(3), atol=1e-14)
    f = lambda x: 0.5 * np.log1p(x)  # noqa: E731
    for x, y in ((0.3, -0.2), (0.1, 0.1 + 1e-7), (0.25, 0.25)):
        exact = 0.5 / (1 + x) if x == y else (f(x) - f(y)) / (x - y)
        assert log_divided(x, y) == pytest.approx(exact, rel=1e-8)
    for x, y, z in ((0.3, -0.2, 0.05), (0.1, 0.1 + 1e-6, 0.1 - 2e-6), (0.2, 0.2, 0.2)):
        if np.ptp([x, y, z]) > 1e-3:
            exact = ((f(x) - f(y)) / (x - y) - (f(y) - f(z)) / (y - z)) / (x - z)
        else:
            exact = -0.25 / (1 + (x + y + z) / 3) ** 2
        assert log_divided2(x, y, z) == pytest.approx(exact, rel=1e-6)
        assert log_divided2(z, x, y) == pytest.approx(log_divided2(x, y, z), rel=1e-12)


def _point(material, grad, old=None, dt=0.5):
    mats = {0: material}
    table = material_table([0], mats, "elastic")
    beh = behaviour_table([0], mats, table, finite=True)
    old = np.zeros((1, beh.width)) if old is None else old
    new = np.zeros_like(old)
    p, a = np.zeros((1, 9)), np.zeros((1, 9, 9))
    failed = update_points_finite(beh.kind, table.mat, beh.pmat, beh.data, grad[None], 0.0, dt,
                                  old, new, p, a, 0, 1.0)  # fmt: skip
    assert failed == 0
    return p[0], a[0], new


VON_MISES = Material(
    young=E,
    poisson=NU,
    plasticity=Plasticity(
        200.0,
        hardening=500.0,
        isotropic=[(100.0, 20.0)],
        kinematic=[(5e4, 300.0)],
        viscosity=80.0,
        exponent=6.0,
    ),
)


@pytest.mark.parametrize("material", [Material.cubic(168e3, 121e3, 75e3), VON_MISES])
def test_tangent_and_objectivity(material):
    """dP/dF against finite differences, also with equal principal stretches; a rotation
    superposed on the deformation turns the first Piola-Kirchhoff stress."""
    rng = np.random.default_rng(1)
    grads = [rng.normal(0, 0.05, 9) + np.diag([0, 0, 0.1]).ravel()]
    grads += [np.diag([0.01, 0.01, 0.01]).ravel(), np.diag([0.0, 0.0, 0.02]).ravel(), np.zeros(9)]
    for grad in grads:
        p, a, _ = _point(material, grad)
        fd = np.zeros((9, 9))
        h = 1e-7
        for j in range(9):
            d = np.zeros(9)
            d[j] = h
            fd[:, j] = (_point(material, grad + d)[0] - _point(material, grad - d)[0]) / (2 * h)
        assert np.allclose(a, 0.5 * (fd + fd.T), rtol=0, atol=1e-7 * np.abs(fd).max())
    q = expm(np.array([[0, -0.3, 0.2], [0.3, 0, -0.5], [-0.2, 0.5, 0]]))
    f = np.eye(3) + grads[0].reshape(3, 3)
    p_q = _point(material, (q @ f - np.eye(3)).ravel())[0]
    p = _point(material, grads[0])[0]
    assert np.allclose(p_q.reshape(3, 3), q @ p.reshape(3, 3), atol=1e-12 * np.abs(p).max())


def test_hencky_tension():
    """Uniaxial tension of an isotropic Hencky material: the Kirchhoff stress is E times
    the logarithmic strain, the lateral logarithmic strains -nu times it."""
    m = Material(young=E, poisson=NU)
    res = simulate_point(m, {"times": [0, 1], "increments": 4, "strain": {"11": 0.3}},
                         kinematics="finite")  # fmt: skip
    eps = np.log1p(res.strain[:, 0])
    assert np.allclose(res.log_strain[:, 0], eps, atol=1e-12)
    assert np.allclose(res.log_strain[:, 1], -NU * eps, atol=1e-9)
    jac = np.exp((1 - 2 * NU) * eps)
    assert np.allclose(res.cauchy[:, 0] * jac, E * eps, rtol=1e-9, atol=1e-6)
    assert np.allclose(res.stress[:, 0], E * eps / (1 + res.strain[:, 0]), rtol=1e-9, atol=1e-6)


def test_simple_shear():
    """Simple shear of an isotropic Hencky material: the shear of the macroscopic gradient
    is a simple shear, and the Cauchy stress that of the Eulerian logarithmic strain."""
    from scipy.linalg import logm

    m = Material(young=E, poisson=NU)
    gamma = 0.8
    zero = {k: 0.0 for k in ("11", "22", "33", "23", "31")}
    res = simulate_point(m, {"times": [0, 1], "increments": 2,
                             "strain": {**zero, "12": gamma}}, kinematics="finite")  # fmt: skip
    f = np.eye(3)
    f[0, 1] = gamma
    mu = E / (2 * (1 + NU))
    sigma = mu * logm(f @ f.T).real  # det F = 1: the Kirchhoff stress
    expected = [sigma[0, 0], sigma[1, 1], sigma[2, 2], sigma[1, 2], sigma[2, 0], sigma[0, 1]]
    assert np.allclose(res.cauchy[-1], expected, rtol=0, atol=1e-9 * mu)
    assert res.stress[-1, 5] == pytest.approx(expected[5], rel=1e-9)  # P12 = sigma12


def test_logarithmic_plasticity():
    """Tension of a von Mises material in finite strains: in the logarithmic strain
    space, the Kirchhoff stress against the logarithmic strain is the small strain curve
    (rate-independent, nonlinear isotropic and kinematic hardening)."""
    m = Material(young=E, poisson=NU, plasticity=Plasticity(
        200.0, hardening=300.0, isotropic=[(150.0, 10.0)], kinematic=[(2e4, 100.0)]))  # fmt: skip
    times = np.linspace(0, 1, 21)
    nominal = 0.4 * times
    finite = simulate_point(m, {"times": times, "increments": 1, "strain": {"11": nominal}},
                            kinematics="finite")  # fmt: skip
    small = simulate_point(
        m, {"times": times, "increments": 1, "strain": {"11": np.log1p(nominal)}}
    )  # fmt: skip
    kirchhoff = finite.cauchy[:, 0] * np.exp(finite.log_strain[:, :3].sum(axis=1))
    assert np.allclose(kirchhoff, small.stress[:, 0], rtol=1e-8, atol=1e-6)
    assert np.allclose(finite.plastic, small.plastic, rtol=1e-8, atol=1e-12)
    assert finite.plastic[-1] > 0.3


def test_finite_element_paths():
    """A homogeneous von Mises element follows its material point in finite strains; a
    heterogeneous elastic element at small strains follows the small strain solution."""
    m = Material(young=E, poisson=NU, expansion=1e-5, plasticity=Plasticity(
        300.0, isotropic=[(200.0, 30.0)], kinematic=[(5e4, 300.0)], viscosity=150.0,
        exponent=6.0))  # fmt: skip
    path = {"times": [0, 10, 20], "increments": [6, 6], "strain": {"11": [0.2, 0.05]},
            "stress": {"12": [100, -100]}, "temperature": [50, 0]}  # fmt: skip
    ref = simulate_point(m, path, kinematics="finite")
    for order, bc in [(1, "kubc"), (2, "periodic")]:
        mesh = box_mesh((1, 1, 1), order=order, perturb=0.2)
        res = NonlinearFEProblem(mesh, {0: m}, bc, kinematics="finite").run(path)
        scale = np.abs(ref.stress).max()
        assert np.allclose(res.stress, ref.stress, rtol=0, atol=2e-6 * scale)
        assert np.allclose(res.cauchy, ref.cauchy, rtol=0, atol=3e-6 * scale)
        assert np.allclose(res.strain, ref.strain, rtol=0, atol=1e-7)
    soft, hard = Material(young=3e3, poisson=0.4), Material(young=E, poisson=NU)
    rng = np.random.default_rng(2)
    mesh = box_mesh((2, 2, 2), phase=lambda c: (rng.random(len(c)) < 0.4).astype(int), order=2)
    small_path = {"times": [0, 1], "increments": 1, "strain": {"11": 1e-6}, "stress": {"12": 0}}
    small = NonlinearFEProblem(mesh, {0: soft, 1: hard}, "periodic").run(small_path)
    finite = NonlinearFEProblem(mesh, {0: soft, 1: hard}, "periodic", kinematics="finite").run(
        small_path, fields=True
    )
    assert np.allclose(finite.stress, small.stress, rtol=0, atol=1e-5 * np.abs(small.stress).max())
    assert np.allclose(finite.strain, small.strain, rtol=0, atol=1e-11)
    assert finite.phases[1]["cauchy"].shape == (2, 6)
    fields = finite.fields[1.0]
    vol = np.abs(mesh.volumes())
    mean = np.average(fields["stress"], axis=0, weights=vol)
    assert np.allclose(mean, finite.cauchy[-1], rtol=0, atol=1e-5 * np.abs(mean).max())
    assert fields["displacement"].shape == (mesh.n_nodes, 3)


def test_macroscopic_measures():
    """Cauchy stress and logarithmic strain of the macroscopic gradient and stress."""
    rng = np.random.default_rng(4)
    h = rng.normal(0, 0.1, (5, 9))
    f = np.eye(3) + h.reshape(5, 3, 3)
    p = rng.normal(0, 100.0, (5, 9))
    sigma = cauchy(p, h)
    for k in range(5):
        full = p[k].reshape(3, 3) @ f[k].T / np.linalg.det(f[k])
        full = 0.5 * (full + full.T)
        assert np.allclose(sigma[k], [full[0, 0], full[1, 1], full[2, 2], full[1, 2],
                                      full[2, 0], full[0, 1]])  # fmt: skip
        w, v = np.linalg.eigh(f[k].T @ f[k])
        eps = v @ np.diag(0.5 * np.log(w)) @ v.T
        expected = [eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[1, 2], 2 * eps[2, 0], 2 * eps[0, 1]]
        assert np.allclose(log_strain(h[k]), expected, atol=1e-13)
