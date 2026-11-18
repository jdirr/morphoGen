"""Porous elasto-(visco)plasticity (Gurson-Tvergaard-Needleman, Rousselier, the elliptic
criterion of compaction): the hydrostatic yield of GTN and the von Mises limit of the dense
material, the porosity and the plastic volume change, Chu and Needleman's nucleation, the
yield of the compaction of the elliptic criterion along its densification, the consistent
tangents in small and finite strains, the objectivity in finite strains, the rate of
Norton's viscoplasticity, the Jacobian of the update against differences, GTN's matrix
stress from a guess, MINRES on the indefinite tangents of softening points, the volume
elements (finite elements, FFT on the CPU and the GPU), and the kinematic hardening (von
Mises's at f = 0, the Bauschinger effect, the hydrostatic backstress, the tangents and the
Jacobian)."""

import math

import numpy as np
import pytest

from helpers import box_mesh
from morphogen import gpu
from morphogen.homogenization import Material, NonlinearFEProblem, NonlinearFFTProblem
from morphogen.homogenization.behaviours import Plasticity
from morphogen.homogenization.finite import update_points_finite
from morphogen.homogenization.materials import lame, material_table
from morphogen.homogenization.nonlinear import behaviour_table, simulate_point
from morphogen.homogenization.porous import (
    MAX_KINEMATIC,
    P_F0,
    POROUS_WIDTH,
    STATE_BACK,
    PorousPlasticity,
    porous_update,
    state_width,
)
from morphogen.image import VoxelImage

E, NU = 200000.0, 0.3
LAM, MU = lame(E, NU)
GTN = {"criterion": "gurson", "yield": 300.0, "porosity": 0.01, "q1": 1.5, "q2": 1.0}
FOAM = {"criterion": "elliptic", "yield": 1.0, "porosity": 0.5, "C": [[0, 1.0], [0.5, 3.0]],
        "F": [[0, 0.0], [0.5, 0.5]]}  # fmt: skip


def _porous(**spec):
    return Material(young=E, poisson=NU, porous_plasticity=spec)


def _kernel_path(pp, strains, dt=0.1):
    """Stresses, internal variables and tangents of a point of ``pp`` along strains."""
    par = pp.params(LAM, MU)
    old, new = np.zeros(state_width(len(pp.kinematic))), np.zeros(state_width(len(pp.kinematic)))
    out = []
    for eps in strains:
        sig, tan = np.zeros(6), np.zeros((6, 6))
        assert porous_update(par, np.asarray(eps, float), np.zeros(6), dt, old, new, sig, tan) == 0
        old, new = new, old.copy()
        out.append((sig, old.copy(), tan))
    return par, out


def test_gurson_hydrostatic_yield_and_dense_limit():
    """GTN yields under a hydrostatic stress at Σm = 2 σ0 / (3 q2) acosh((1 + q3 f²) / (2 q1
    f)), then softens as the voids grow; at f = 0 every criterion is von Mises."""
    f0, q1, q2, s0 = 0.01, 1.5, 1.0, 300.0
    exact = 2 * s0 / (3 * q2) * math.acosh((1 + q1 * q1 * f0 * f0) / (2 * q1 * f0))
    kb = LAM + 2 * MU / 3
    # the 20th of the 30 steps reaches the yield exactly
    strains = [[e, e, e, 0, 0, 0] for e in np.linspace(0, 1.5 * exact / (3 * kb), 31)[1:]]
    _, out = _kernel_path(PorousPlasticity("gurson", s0, porosity=f0, q1=q1, q2=q2), strains)
    mean = np.array([o[0][:3].mean() for o in out])
    assert mean.argmax() == 19 and mean.max() == pytest.approx(exact, rel=1e-12)
    assert out[19][1][6] == 0.0 and out[20][1][6] > 0.0
    assert np.all(np.diff(mean[mean.argmax() :]) < 0)  # softening by the growth of the voids
    assert np.abs(np.array([o[0][3:] for o in out])).max() < 1e-9 * exact
    path = {"times": [0, 1], "increments": 20, "strain": {"11": [0, 0.02]}}
    dense = Plasticity(300.0, 1000.0, isotropic=[(100.0, 20.0)])
    ref = simulate_point(Material(young=E, poisson=NU, plasticity=dense), path)
    matrix = {"yield": 300.0, "hardening": 1000.0, "isotropic": [{"Q": 100.0, "b": 20.0}]}
    for spec in (
        {"criterion": "gurson"},
        {"criterion": "rousselier", "sigma1": 200.0, "D": 2.0},
        {"criterion": "elliptic", "C": [[0, 1.0], [0.5, 4.0]], "F": [[0, 0.0], [0.5, 1.0]]},
    ):
        res = simulate_point(_porous(**spec, **matrix), path)
        np.testing.assert_allclose(res.stress, ref.stress, rtol=0, atol=1e-7 * 300.0)
        assert res.porosity.max() < 1e-15 and res.plastic[-1] == pytest.approx(ref.plastic[-1])


def test_porosity_volume_change_and_nucleation():
    """The growth porosity follows the plastic volume change, Δfg = (1 - f) Δtr εp, at each
    step; in pure shear GTN does not grow its voids, and Chu and Needleman's nucleation
    gives f = fN / 2 (erf((p - εN) / (sN √2)) + erf(εN / (sN √2)))."""
    pp = PorousPlasticity("gurson", 300.0, porosity=0.02, hardening=500.0)
    strains = [k * 0.001 * np.array([1.0, 0.4, 0.3, 0.2, 0.0, 0.1]) for k in range(1, 31)]
    par, out = _kernel_path(pp, strains)
    previous = np.zeros(POROUS_WIDTH)
    for _, state, _ in out:
        f = par[P_F0] + state[7] + state[8]
        growth = (1 - f) * (state[:3].sum() - previous[:3].sum())
        assert state[7] - previous[7] == pytest.approx(growth, rel=1e-12, abs=1e-15)
        previous = state
    assert previous[7] > 0.01  # the voids grew
    nucleation = {"type": "gaussian", "fN": 0.04, "eN": 0.1, "sN": 0.05}
    shear = PorousPlasticity("gurson", 300.0, porosity=0.0, hardening=200.0, nucleation=nucleation)
    strains = [[0, 0, 0, 0, 0, g] for g in np.linspace(0, 0.4, 801)[1:]]
    par, out = _kernel_path(shear, strains)
    state = out[-1][1]
    p = state[6]
    exact = 0.02 * (
        math.erf((p - 0.1) / (0.05 * math.sqrt(2))) + math.erf(0.1 / (0.05 * math.sqrt(2)))
    )
    assert abs(state[7]) < 1e-12 and state[8] == pytest.approx(exact, rel=0.01)


def test_compaction_with_the_elliptic_criterion():
    """Isostatic compaction of the elliptic criterion of a perfectly plastic matrix: on the
    yield surface σ* = 3 sqrt(F(f)) |Σm| = R0 at each step as the porosity falls (the mean
    Kirchhoff stress in finite strains), until the dense material carries the pressure
    elastically; die compaction densifies too, at a smaller lateral stress."""
    foam = _porous(**FOAM)
    for kinematics in ("small", "finite"):
        path = {"times": [0, 1], "increments": 40, "strain": {k: -0.3 for k in ("11", "22", "33")}}
        res = simulate_point(foam, path, kinematics=kinematics)
        f = res.porosity
        assert np.all(np.diff(f) <= 1e-15) and f[0] == 0.5 and f[-1] < 0.1
        mean = res.stress[:, :3].mean(axis=1)
        if kinematics == "finite":
            jac = np.prod(1 + res.strain[:, :3], axis=1)
            mean = jac * res.cauchy[:, :3].mean(axis=1)
        plastic = np.flatnonzero((np.diff(res.plastic, prepend=0) > 0) & (f > 0.05))
        assert plastic.size > 10
        np.testing.assert_allclose(-mean[plastic], 1.0 / (3 * np.sqrt(f[plastic])), rtol=1e-8)
        strain = {"11": -0.3, "22": 0.0, "33": 0.0}  # the walls of the die
        die = simulate_point(foam, {"times": [0, 1], "increments": 30, "strain": strain},
                             kinematics=kinematics)  # fmt: skip
        assert np.all(np.diff(die.porosity) < 0) and die.porosity[-1] < 0.35
        assert np.all(np.abs(die.stress[1:, 1]) < np.abs(die.stress[1:, 0]))


def test_consistent_tangents_and_objectivity():
    """The tangents (symmetrized) are the derivatives of the stress by central differences,
    in small strains and in finite strains (dP/dF), for the three criteria, Norton's flow and
    the nucleation; a superposed rotation of F turns P and leaves the internal variables."""
    cases = [
        PorousPlasticity("gurson", 300.0, porosity=0.02, hardening=500.0, fc=0.05, fF=0.2,
                         nucleation={"type": "gaussian", "fN": 0.04, "eN": 0.05, "sN": 0.05}),
        PorousPlasticity("gurson", 300.0, porosity=0.02, viscosity=200.0, exponent=4.0),
        PorousPlasticity("rousselier", 300.0, porosity=0.01, hardening=800.0, sigma1=200.0, D=2.0),
        PorousPlasticity("elliptic", 50.0, porosity=0.3, hardening=100.0,
                         C=[(0, 1.0), (0.4, 2.5)], F=[(0, 0.0), (0.4, 0.4)]),
    ]  # fmt: skip
    for pp in cases:
        sign = -1.0 if pp.criterion == "elliptic" else 1.0
        direction = sign * np.array([1.0, 0.3, 0.6, 0.4, -0.25, 0.15])
        strains = [k * 0.0015 * direction for k in range(1, 8)]
        par, out = _kernel_path(pp, strains)
        old = out[-2][1]
        eps = strains[-1]
        num = np.zeros((6, 6))
        for j in range(6):
            pair = []
            for h in (1e-7, -1e-7):
                e = eps.copy()
                e[j] += h
                sig = np.zeros(6)
                porous_update(
                    par, e, np.zeros(6), 0.1, old, np.zeros(POROUS_WIDTH), sig, np.zeros((6, 6))
                )
                pair.append(sig)
            num[:, j] = (pair[0] - pair[1]) / 2e-7
        assert np.abs(out[-1][2] - 0.5 * (num + num.T)).max() < 1e-6 * np.abs(num).max()
        # finite strains
        m = Material(young=E, poisson=NU, porous_plasticity=pp)
        table = material_table([0], {0: m}, "elastic")
        beh = behaviour_table([0], {0: m}, table, finite=True)

        def step(f, state, beh=beh, table=table):
            new, pk1, tan = np.zeros_like(state), np.zeros((1, 9)), np.zeros((1, 9, 9))
            args = (beh.kind, table.mat, beh.pmat, beh.data, (f - np.eye(3)).reshape(1, 9))
            assert update_points_finite(*args, 0.0, 0.1, state, new, pk1, tan, 0, 1.0) == 0
            return pk1[0], tan[0], new

        f = np.eye(3)
        state = np.zeros((1, beh.width))
        d = 0.01 * sign * np.array([[1.0, 0.2, 0.0], [0.1, 0.5, 0.15], [0.0, 0.05, 0.6]])
        for _ in range(6):
            f = f + d
            _, _, state = step(f, state)
        f1 = f + 0.5 * d
        p, a, new = step(f1, state)
        num = np.zeros((9, 9))
        for j in range(9):
            fp, fm = f1.ravel().copy(), f1.ravel().copy()
            fp[j] += 1e-7
            fm[j] -= 1e-7
            num[:, j] = (step(fp.reshape(3, 3), state)[0] - step(fm.reshape(3, 3), state)[0]) / 2e-7
        assert np.abs(a - 0.5 * (num + num.T)).max() < 1e-5 * np.abs(num).max()
        k = np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0)
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        q = np.eye(3) + math.sin(0.7) * kx + (1 - math.cos(0.7)) * kx @ kx
        p_q, _, new_q = step(q @ f1, state)
        np.testing.assert_allclose(
            p_q.reshape(3, 3), q @ p.reshape(3, 3), atol=1e-12 * np.abs(p).max()
        )
        np.testing.assert_allclose(new_q, new, atol=1e-12)


def test_viscoplastic_rate_and_dense_norton():
    """Norton's flow of the matrix: a faster tension needs a higher stress; dense, it is the
    von Mises viscoplasticity of the same K and n."""
    path = {"times": [0, 1], "increments": 20, "strain": {"11": [0, 0.01]}}
    slow = {"times": [0, 100], "increments": 20, "strain": {"11": [0, 0.01]}}
    gtn = _porous(**GTN, norton={"K": 300.0, "n": 5.0})
    fast, creep = simulate_point(gtn, path), simulate_point(gtn, slow)
    assert fast.stress[-1, 0] > creep.stress[-1, 0] > 300.0 and fast.porosity[-1] > 0.01
    dense = _porous(**{**GTN, "porosity": 0.0}, norton={"K": 300.0, "n": 5.0})
    norton = Plasticity(300.0, viscosity=300.0, exponent=5.0)
    mises = simulate_point(Material(young=E, poisson=NU, plasticity=norton), path)
    np.testing.assert_allclose(simulate_point(dense, path).stress, mises.stress, rtol=0,
                               atol=1e-7 * 300.0)  # fmt: skip


def test_jacobian_against_differences():
    """The Jacobian of the four equations of a porous point, and their derivatives by the
    trial invariants, against central differences of the equations: GTN (with its
    coalescence, Chu and Needleman's and an exponential nucleation, Norton's flow),
    Rousselier (at its vertex too) and the elliptic criterion with its tables."""
    from morphogen.homogenization.porous import _jacobian, _residual

    kb = LAM + 2 * MU / 3
    cases = [
        (PorousPlasticity("gurson", 300.0, porosity=0.01, hardening=1000.0,
                          isotropic=((100.0, 10.0),)), 300.0, False),
        (PorousPlasticity("gurson", 300.0, porosity=0.05, fc=0.04, fF=0.2), 300.0, False),
        (PorousPlasticity("gurson", 300.0, porosity=0.01, nucleation={
            "type": "gaussian", "fN": 0.04, "eN": 0.05, "sN": 0.02}), 300.0, False),
        (PorousPlasticity("gurson", 300.0, porosity=0.02, nucleation={
            "type": "exponential", "B": 0.05, "e0": 0.1}), 300.0, False),
        (PorousPlasticity("gurson", 300.0, porosity=0.02, viscosity=300.0, exponent=5.0),
         300.0, False),
        (PorousPlasticity("rousselier", 300.0, porosity=0.01, sigma1=400.0, D=2.0,
                          hardening=800.0), 300.0, False),
        (PorousPlasticity("rousselier", 300.0, porosity=0.01, sigma1=400.0, D=2.0), 300.0, True),
        (PorousPlasticity(**{k: v for k, v in FOAM.items() if k != "yield"},
                          yield_stress=1.0), 1.0, False),
    ]  # fmt: skip
    rng = np.random.default_rng(2)

    def residual(par, x, sm_tr, seq_tr, f0, vertex):
        out = np.zeros(4)
        return _residual(par, x, sm_tr, seq_tr, kb, MU, f0, 0.05, 0.1, vertex, out, 0.0, kin), out

    kin = np.zeros((1 + MAX_KINEMATIC, 7))  # no backstresses

    checked = 0
    for pp, scale, vertex in cases:
        par = pp.params(LAM, MU)
        f0 = par[P_F0]
        for _ in range(10):
            sm_tr, seq_tr = rng.normal(0, 1.5) * scale, abs(rng.normal(1.2, 0.5)) * scale
            z = abs(rng.normal(0.3, 0.1)) if pp.viscous else abs(rng.normal(2e-3, 1e-3))
            x = np.array([rng.normal(0, 2e-3) * scale / 300, abs(rng.normal(1e-3, 5e-4)) *
                          scale / 300, z, min(f0 + abs(rng.normal(0, 0.01)), 0.6)])  # fmt: skip
            s, _ = residual(par, x, sm_tr, seq_tr, f0, vertex)
            if s < 0:
                continue
            jac, dr = np.zeros((4, 4)), np.zeros((4, 2))
            _jacobian(par, x, sm_tr, seq_tr, kb, MU, f0, 0.05, 0.1, vertex, s, np.zeros(8),
                      jac, dr, kin)  # fmt: skip
            fd, fdr = np.zeros((4, 4)), np.zeros((4, 2))
            for j in range(4):
                h = 1e-6 * max(abs(x[j]), 1e-4 if j == 3 else 1e-5)
                step = np.eye(4)[j] * h
                plus = residual(par, x + step, sm_tr, seq_tr, f0, vertex)[1]
                fd[:, j] = (plus - residual(par, x - step, sm_tr, seq_tr, f0, vertex)[1]) / (2 * h)
            for j, (a, b) in enumerate(((1, 0), (0, 1))):
                h = 1e-6 * scale
                fdr[:, j] = (residual(par, x, sm_tr + a * h, seq_tr + b * h, f0, vertex)[1]
                             - residual(par, x, sm_tr - a * h, seq_tr - b * h, f0, vertex)[1]
                             ) / (2 * h)  # fmt: skip
            for mine, theirs in ((jac, fd), (dr, fdr)):
                rows = np.max(np.abs(theirs), axis=1, keepdims=True) + 1e-300
                assert np.max(np.abs(mine - theirs) / rows) < 1e-5
            checked += 1
    assert checked > 60


def test_matrix_stress_from_a_guess():
    """GTN's matrix stress bracketed from a guess, that of a nearby point (the differences
    of the Jacobian, the line search) or one a thousand times off on either side, is the
    root found from the estimate, with its derivatives, within roundoff."""
    from morphogen.homogenization.porous import _gurson_stress

    rng = np.random.default_rng(5)
    q1, q2, q3 = 1.5, 1.0, 2.25
    for _ in range(100):
        fs, seq, sm = rng.uniform(0.0, 0.3), abs(rng.normal(300, 200)), rng.normal(0, 400)
        cold = _gurson_stress(q1, q2, q3, fs, seq, sm, 0.0)
        assert cold[0] > 0.0
        for factor in (1.0, 1.0 + 1e-7, 1.0 - 1e-7, 0.5, 2.0, 1e-3, 1e3):
            warm = _gurson_stress(q1, q2, q3, fs, seq, sm, factor * cold[0])
            np.testing.assert_allclose(warm, cold, rtol=1e-12, atol=1e-14)


def test_minres_on_an_indefinite_system():
    """MINRES solves a symmetric indefinite system with a positive definite preconditioner
    (20 negative eigenvalues of 300), and the conjugate gradients hand it over at their first
    direction of negative curvature; a positive definite system keeps the conjugate
    gradients alone."""
    from morphogen.homogenization.solvers import conjugate_gradient, minres

    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(300, 300)))
    lam = np.concatenate([rng.uniform(0.1, 2.0, 280), -rng.uniform(0.05, 0.5, 20)])
    a = q @ np.diag(lam) @ q.T
    d = np.abs(np.diag(a)) + 0.5
    b = rng.normal(size=300)
    exact = np.linalg.solve(a, b)
    for solver in (minres, conjugate_gradient):
        x, it, res = solver(lambda v: a @ v, lambda r: r / d, b, 1e-10, 2000)
        assert res < 1e-10 and np.linalg.norm(b - a @ x) < 1e-10 * np.linalg.norm(b)
        assert np.linalg.norm(x - exact) < 1e-8 * np.linalg.norm(exact) and it < 300
    spd = q @ np.diag(np.abs(lam) + 0.1) @ q.T
    _, it, res = conjugate_gradient(lambda v: spd @ v, lambda r: r / d, b, 1e-10, 2000)
    assert res < 1e-10 and it < 60


def test_paths_with_softening_points(caplog):
    """Stiff spheres in a GTN matrix with a nucleation whose lateral strains are held: the
    symmetrized tangent of its points (forced: ``tangent="symmetric"``; by default the
    porous points keep their full tangents) loses its positive definiteness at the high
    stress triaxialities (above about 4 with a weak hardening), and so does that of the
    voxels; MINRES takes the linear solves over from the conjugate gradients at the first
    direction of negative curvature, so that they converge (no warning) and no step is
    divided; the GPU follows the CPU."""
    import logging

    from morphogen.domain import Domain
    from morphogen.models import BooleanSpheres

    micro = BooleanSpheres(radius=1.0, volume_fraction=0.2).generate(Domain.cube(10.0), seed=2)
    image = micro.voxelize(shape=8)
    gtn = {**GTN, "yield": 300.0, "hardening": 1000.0, "nucleation": {
        "type": "gaussian", "fN": 0.04, "eN": 0.1, "sN": 0.05}}  # fmt: skip
    mats = {0: _porous(**gtn), 1: Material(young=400e3, poisson=0.2)}
    path = {"times": [0, 10], "increments": 10, "strain": {"11": 0.02, "22": 0.0, "33": 0.0}}
    with caplog.at_level(logging.WARNING, logger="morphogen"):
        res = NonlinearFFTProblem(image, mats, tangent="symmetric").run(path)
    assert not [r for r in caplog.records if "stopped at relative residual" in r.getMessage()]
    assert res.info["cuts"] == 0 and res.phases[0]["porosity"][-1] > 0.03
    if gpu.available():
        on_gpu = NonlinearFFTProblem(image, mats, device="gpu", tangent="symmetric").run(path)
        np.testing.assert_allclose(on_gpu.stress, res.stress, rtol=0, atol=1e-9 * 700.0)


def test_volume_elements_and_configuration():
    """A porous matrix around an elastic inclusion by finite elements and by FFT: the
    porosity of the porous phase grows in tension, element fields included; the FFT path
    on the GPU follows that of the CPU; the configuration of a material round-trips."""
    matrix = _porous(**GTN, hardening=1000.0)
    mats = {0: matrix, 1: Material(young=4e5, poisson=0.2)}
    path = {"times": [0, 1], "increments": 5, "strain": {"11": [0, 0.01]}}
    mesh = box_mesh((3, 3, 3), phase=lambda c: (np.abs(c - 0.5).max(axis=1) < 1 / 6).astype(int),
                    upper=(1.0, 1.0, 1.0))  # fmt: skip
    fe = NonlinearFEProblem(mesh, mats).run(path, fields=True)
    assert fe.porosity[0] == pytest.approx(0.01) and fe.porosity[-1] > 0.01
    assert fe.phases[0]["porosity"][-1] == pytest.approx(fe.porosity[-1])
    assert "porosity" not in fe.phases[1] and "f" in fe.rows()[-1]
    field = fe.fields[1.0]["porosity"]
    assert np.all(np.isnan(field[mesh.phase == 1])) and np.all(field[mesh.phase == 0] > 0.0)
    assert np.nanmax(field) > 0.01
    data = np.zeros((4, 4, 4), np.uint8)
    data[1:3, 1:3, 1:3] = 1
    image = VoxelImage(data, periodic=True)
    fft = NonlinearFFTProblem(image, mats).run(path)
    assert fft.phases[0]["porosity"][-1] > 0.01
    if gpu.available():
        on_gpu = NonlinearFFTProblem(image, mats, device="gpu").run(path)
        np.testing.assert_allclose(on_gpu.stress, fft.stress, rtol=0, atol=1e-9 * 300.0)
        assert on_gpu.phases[0]["porosity"][-1] == pytest.approx(fft.phases[0]["porosity"][-1],
                                                                 rel=1e-9)  # fmt: skip
    back = Material.from_config(matrix.to_config())
    assert back == matrix and back.porous and back.nonlinear
    with pytest.raises(ValueError, match="one of them"):
        Material(young=E, poisson=NU, plasticity={"yield": 1.0}, porous_plasticity=GTN)
    with pytest.raises(ValueError, match="criterion"):
        PorousPlasticity("cam_clay", 1.0)
    with pytest.raises(ValueError, match="fc"):
        PorousPlasticity("gurson", 1.0, fc=0.2, fF=0.1)


#: Armstrong-Frederick and Prager terms of the backstresses (C, D)
KIN = ((40000.0, 300.0), (8000.0, 0.0))


def _deviator(rng, scale):
    """A random deviator (tensor components)."""
    v = rng.normal(size=6) * scale
    v[:3] -= v[:3].mean()
    return v


def test_kinematic_configuration():
    """The kinematic terms of a porous plasticity: their configuration (round trip, the
    rows of the kernels, the internal variables of the points), and the refusals: more than
    four terms, negative coefficients, Rousselier's criterion."""
    from morphogen.homogenization.porous import P_KIN, P_NKIN

    spec = {**GTN, "kinematic": [{"C": 1e4, "D": 100.0}, [5e3, 0.0]]}
    pp = PorousPlasticity.from_config(spec)
    assert pp.kinematic == ((1e4, 100.0), (5e3, 0.0))
    assert PorousPlasticity.from_config(pp.to_config()) == pp
    row = pp.params(LAM, MU)
    assert row[P_NKIN] == 2 and list(row[P_KIN : P_KIN + 4]) == [1e4, 100.0, 5e3, 0.0]
    m = _porous(**spec)
    beh = behaviour_table([0], {0: m}, material_table([0], {0: m}, "elastic"))
    assert beh.width == state_width(2) == STATE_BACK + 12
    with pytest.raises(ValueError, match="at most 4"):
        PorousPlasticity("gurson", 300.0, kinematic=[(1e3, 0.0)] * 5)
    with pytest.raises(ValueError, match="C >= 0 and D >= 0"):
        PorousPlasticity("elliptic", 300.0, kinematic=[(1e3, -1.0)])
    with pytest.raises(ValueError, match="criteria gurson and elliptic"):
        PorousPlasticity("rousselier", 300.0, sigma1=200.0, D=2.0, kinematic=[(1e3, 0.0)])
    # the consumers: the form of the tab Mechanics, the identification, the NTFA (refused)
    from morphogen.gui.tools.mechanics import _porous as form
    from morphogen.homogenization.identification import get_parameter, set_parameter
    from morphogen.homogenization.ntfa import train_ntfa

    fields = {"criterion": "elliptic", "porosity": 0.3, "yield_stress": 50.0, "hardening": 0.0,
              "elliptic": [2.0, 0.5], "kinematic": [3e4, 200.0, 5e3, 0.0]}  # fmt: skip
    assert PorousPlasticity.from_config(form(fields)).kinematic == ((3e4, 200.0), (5e3, 0.0))
    config = m.to_config()
    assert get_parameter(config, "C2") == 5e3 and get_parameter(config, "q1") == 1.5
    set_parameter(config, "D1", 50.0)
    assert Material.from_config(config).porous_plasticity.kinematic[0] == (1e4, 50.0)
    image = VoxelImage(np.zeros((4, 4, 4), dtype=np.uint8), periodic=True)
    with pytest.raises(ValueError, match=r"porous plasticity .* is not reduced"):
        train_ntfa(image, {0: m}, [])


def test_kinematic_hardening_at_full_density_is_von_mises():
    """Dense (f = 0, no nucleation), GTN and the elliptic criterion with kinematic terms are
    the von Mises kernel with the same Armstrong-Frederick terms along a cycle of a general
    direction that turns: the stresses, plastic strains, p and backstresses within 10⁻¹⁰,
    the tangents within 10⁻⁸, no porosity."""
    from morphogen.homogenization.behaviours import STATE_X, Plasticity, _j2_update

    iso = ((80.0, 15.0),)
    vm = Plasticity(300.0, 500.0, isotropic=iso, kinematic=KIN)
    direction = np.array([1.0, -0.4, 0.2, 0.5, -0.3, 0.25])
    turn = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    amp = np.concatenate([np.linspace(0, 1, 15)[1:], np.linspace(1, -1, 26)[1:],
                          np.linspace(-1, 0.6, 21)[1:]])  # fmt: skip
    strains = [0.006 * a * direction + 0.002 * math.sin(5 * a) * turn for a in amp]
    par = vm.params(LAM, MU)
    old, new = np.zeros(STATE_X + 12), np.zeros(STATE_X + 12)
    ref = []
    for eps in strains:
        sig, tan = np.zeros(6), np.zeros((6, 6))
        assert _j2_update(par, 2, eps, np.zeros(6), 0.1, old, new, sig, tan, np.empty((7, 6))) == 0
        old, new = new, old.copy()
        ref.append((sig, old.copy(), tan))
    assert np.abs(ref[-1][1][STATE_X:]).max() > 20.0  # the backstresses
    elliptic = {"criterion": "elliptic", "C": FOAM["C"], "F": FOAM["F"]}
    for spec in ({"criterion": "gurson"}, elliptic):
        pp = PorousPlasticity(**spec, yield_stress=300.0, hardening=500.0, isotropic=iso,
                              kinematic=KIN)  # fmt: skip
        _, out = _kernel_path(pp, strains)
        for (sig, state, tan), (s_ref, v_ref, t_ref) in zip(out, ref, strict=True):
            np.testing.assert_allclose(sig, s_ref, rtol=0, atol=1e-10 * 300.0)
            np.testing.assert_allclose(state[:7], v_ref[:7], rtol=0, atol=1e-12)
            np.testing.assert_allclose(state[STATE_BACK:], v_ref[STATE_X:], rtol=0,
                                       atol=1e-10 * 300.0)  # fmt: skip
            assert abs(state[7]) < 1e-15 and state[8] == 0.0
            np.testing.assert_allclose(tan, t_ref, rtol=0, atol=1e-8 * E)


def test_kinematic_bauschinger_and_hydrostatic_backstress():
    """With voids, the kinematic terms yield in reverse earlier than the isotropic hardening
    of the same saturation along a tension-compression cycle (the Bauschinger effect: below
    the initial yield stress); under a hydrostatic strain Prager's backstress has the
    hydrostatic part 2/9 C tr εp, the mean stress yields on the shifted surface (Σm - tr X /
    3 at GTN's hydrostatic yield of the current porosity), it hardens where the voids alone
    soften the point, the voids grow less, and the growth porosity is (1 - f) Δtr εp at
    every step."""
    path = {"times": [0, 1, 3], "increments": [20, 40], "strain": {"11": [0, 0.02, -0.02]}}
    gtn = {"criterion": "gurson", "yield": 300.0, "porosity": 0.01}
    reverse = []
    for hardening in ({"kinematic": [{"C": 30000.0, "D": 200.0}]},
                      {"isotropic": [{"Q": 150.0, "b": 200.0}]}):  # fmt: skip
        res = simulate_point(_porous(**gtn, **hardening), path)
        top = int(np.argmax(res.strain[:, 0]))
        p = np.asarray(res.plastic)
        k = top + 1 + int(np.argmax(np.diff(p[top:]) > 0))  # the first reverse flow
        assert p[k - 1] == p[top] < p[k]
        reverse.append(-res.stress[k, 0])
        assert res.stress[top, 0] > 400.0 and res.porosity[top] > 0.01
    assert reverse[0] < 0.5 * reverse[1] and reverse[0] < 300.0
    # the hydrostatic backstress
    c, r0, f0, q1 = 150000.0, 300.0, 0.01, 1.5
    strains = [[e, e, e, 0, 0, 0] for e in np.linspace(0, 0.004, 41)[1:]]
    _, out = _kernel_path(PorousPlasticity("gurson", r0, porosity=f0, kinematic=((c, 0.0),)),
                          strains)  # fmt: skip
    _, voids = _kernel_path(PorousPlasticity("gurson", r0, porosity=f0), strains)
    means, previous = [], np.zeros(STATE_BACK + 6)
    for sig, state, _ in out:
        tr_p = state[:3].sum()
        back = state[STATE_BACK : STATE_BACK + 6]
        assert back[:3].mean() == pytest.approx(2 / 9 * c * tr_p, rel=1e-12, abs=1e-300)
        assert np.abs(back[3:]).max() < 1e-9 * abs(back[0]) + 1e-300
        f = f0 + state[7] + state[8]
        growth = (1 - f) * (tr_p - previous[:3].sum())
        assert state[7] - previous[7] == pytest.approx(growth, rel=1e-12, abs=1e-18)
        if state[6] > previous[6]:
            hydrostatic = 2 * r0 / 3 * math.acosh((1 + q1 * q1 * f * f) / (2 * q1 * f))
            assert sig[:3].mean() - back[:3].mean() == pytest.approx(hydrostatic, rel=1e-10)
            means.append(sig[:3].mean())
        previous = state
    assert len(means) > 15 and np.all(np.diff(means) > 0)
    plain = [s[:3].mean() for s, st, _ in voids if st[6] > 0]
    assert np.all(np.diff(plain) < 0)  # the voids alone soften
    assert out[-1][1][7] < voids[-1][1][7]


def test_kinematic_tangents_and_jacobian_against_differences():
    """With backstresses, along paths that turn and grow (GTN) or close (the elliptic
    criterion) the voids: the tangents (symmetrized) are the central differences of the
    stress in small strains and of the first Piola-Kirchhoff stress in finite strains,
    which a superposed rotation turns, leaving the internal variables; the Jacobian of the
    four equations and their derivatives by the trial invariants (m*, J*) are the central
    differences of the equations."""
    from morphogen.homogenization.porous import _jacobian, _relative_deviator, _residual

    cases = [
        PorousPlasticity("gurson", 300.0, porosity=0.02, hardening=500.0, kinematic=KIN,
                         nucleation={"type": "constant", "A": 0.5}),
        PorousPlasticity("gurson", 300.0, porosity=0.02, viscosity=200.0, exponent=4.0,
                         kinematic=((30000.0, 200.0),)),
        PorousPlasticity("elliptic", 50.0, porosity=0.3, hardening=100.0,
                         kinematic=((5000.0, 50.0),), C=[(0, 1.0), (0.4, 2.5)],
                         F=[(0, 0.0), (0.4, 0.4)]),
    ]  # fmt: skip
    for pp in cases:
        width = state_width(len(pp.kinematic))
        sign = -1.0 if pp.criterion == "elliptic" else 1.0
        first = sign * np.array([1.0, 0.3, 0.6, 0.4, -0.25, 0.15])
        second = sign * np.array([0.5, 0.9, 0.4, -0.5, 0.3, 0.6])
        strains = [k * 0.0015 * first for k in range(1, 6)]
        strains += [5 * 0.0015 * first + k * 0.001 * second for k in range(1, 4)]
        par, out = _kernel_path(pp, strains)
        old, eps = out[-2][1], strains[-1]
        f = par[P_F0] + out[-1][1][7] + out[-1][1][8]
        assert (f > par[P_F0]) == (sign > 0) and np.abs(out[-1][1][STATE_BACK:]).max() > 1.0
        num = np.zeros((6, 6))
        for j in range(6):
            pair = []
            for h in (1e-7, -1e-7):
                e, sig = eps.copy(), np.zeros(6)
                e[j] += h
                porous_update(par, e, np.zeros(6), 0.1, old, np.zeros(width), sig, np.zeros((6, 6)))
                pair.append(sig)
            num[:, j] = (pair[0] - pair[1]) / 2e-7
        assert np.abs(out[-1][2] - 0.5 * (num + num.T)).max() < 1e-6 * np.abs(num).max()
    # finite strains: dP/dF and a superposed rotation
    m = Material(young=E, poisson=NU, porous_plasticity=cases[0])
    table = material_table([0], {0: m}, "elastic")
    beh = behaviour_table([0], {0: m}, table, finite=True)

    def step(f, state):
        new, pk1, tan = np.zeros_like(state), np.zeros((1, 9)), np.zeros((1, 9, 9))
        args = (beh.kind, table.mat, beh.pmat, beh.data, (f - np.eye(3)).reshape(1, 9))
        assert update_points_finite(*args, 0.0, 0.1, state, new, pk1, tan, 0, 1.0) == 0
        return pk1[0], tan[0], new

    f, state = np.eye(3), np.zeros((1, beh.width))
    d = 0.01 * np.array([[1.0, 0.2, 0.0], [0.1, 0.5, 0.15], [0.0, 0.05, 0.6]])
    for k in range(6):
        f = f + (d if k < 4 else 0.6 * d.T)
        _, _, state = step(f, state)
    f1 = f + 0.5 * d.T
    p, a, new = step(f1, state)
    assert np.abs(new[0, STATE_BACK:]).max() > 1.0
    num = np.zeros((9, 9))
    for j in range(9):
        fp, fm = f1.ravel().copy(), f1.ravel().copy()
        fp[j] += 1e-7
        fm[j] -= 1e-7
        num[:, j] = (step(fp.reshape(3, 3), state)[0] - step(fm.reshape(3, 3), state)[0]) / 2e-7
    assert np.abs(a - 0.5 * (num + num.T)).max() < 1e-5 * np.abs(num).max()
    k = np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    q = np.eye(3) + math.sin(0.7) * kx + (1 - math.cos(0.7)) * kx @ kx
    p_q, _, new_q = step(q @ f1, state)
    np.testing.assert_allclose(p_q.reshape(3, 3), q @ p.reshape(3, 3), atol=1e-12 * np.abs(p).max())
    np.testing.assert_allclose(new_q, new, atol=1e-12 * np.abs(new).max())
    # the Jacobian and the derivatives by the trial invariants
    kb = LAM + 2 * MU / 3
    rng = np.random.default_rng(5)
    checked = 0
    for pp, scale in zip(cases, (300.0, 300.0, 50.0), strict=True):
        par = pp.params(LAM, MU)
        f0 = par[P_F0]

        def residual(x, sm_tr, kin, par=par, f0=f0):
            out = np.zeros(4)
            return _residual(par, x, sm_tr, 0.0, kb, MU, f0, 0.05, 0.1, False, out, 0.0, kin), out

        for _ in range(10):
            kin = np.zeros((1 + MAX_KINEMATIC, 7))
            kin[0, :6] = _deviator(rng, scale)
            for i in range(len(pp.kinematic)):
                kin[1 + i, :6] = 0.3 * _deviator(rng, scale)
                kin[1 + i, 6] = 0.2 * rng.normal() * scale
            sm_tr = rng.normal(0, 1.5) * scale
            z = abs(rng.normal(0.3, 0.1)) if pp.viscous else abs(rng.normal(2e-3, 1e-3))
            x = np.array([rng.normal(0, 2e-3) * scale / 300, abs(rng.normal(1e-3, 5e-4)) *
                          scale / 300, z, min(f0 + abs(rng.normal(0, 0.01)), 0.6)])  # fmt: skip
            s, _ = residual(x, sm_tr, kin)
            if s < 0:
                continue
            jac, dr = np.zeros((4, 4)), np.zeros((4, 2))
            _jacobian(par, x, sm_tr, 0.0, kb, MU, f0, 0.05, 0.1, False, s, np.zeros(8), jac, dr,
                      kin)  # fmt: skip
            fd, fdr = np.zeros((4, 4)), np.zeros((4, 2))
            for j in range(4):
                h = 1e-6 * max(abs(x[j]), 1e-4 if j == 3 else 1e-5)
                step_x = np.eye(4)[j] * h
                fd[:, j] = (residual(x + step_x, sm_tr, kin)[1]
                            - residual(x - step_x, sm_tr, kin)[1]) / (2 * h)  # fmt: skip
            h = 1e-6 * scale
            fdr[:, 0] = (residual(x, sm_tr + h, kin)[1] - residual(x, sm_tr - h, kin)[1]) / (2 * h)
            # J* along the direction n of the relative deviator: ξ* + 2/3 h n has J* + h
            dp = 0.1 * z**pp.exponent if pp.viscous else z
            xi, w = np.zeros(6), np.zeros(6)
            _relative_deviator(par, kin, dp, xi, w)
            n = 1.5 * xi / math.sqrt(1.5 * (xi[:3] @ xi[:3] + 2 * xi[3:] @ xi[3:]))
            plus, minus = kin.copy(), kin.copy()
            plus[0, :6] += 2 / 3 * h * n
            minus[0, :6] -= 2 / 3 * h * n
            fdr[:, 1] = (residual(x, sm_tr, plus)[1] - residual(x, sm_tr, minus)[1]) / (2 * h)
            for mine, theirs in ((jac, fd), (dr, fdr)):
                rows = np.max(np.abs(theirs), axis=1, keepdims=True) + 1e-300
                assert np.max(np.abs(mine - theirs) / rows) < 1e-5
            checked += 1
    assert checked > 20
