"""Nonlinear behaviours along loading paths: von Mises elasto-(visco)plasticity, the
material point, and the periodic element whose macroscopic strain components are
degrees of freedom (finite elements)."""

import json

import numpy as np
import pytest
import yaml

import morphogen as mg
from helpers import box_mesh, laminate_path
from morphogen import cli
from morphogen.config import ConfigError, load_config
from morphogen.homogenization import (
    Loading,
    LoadingPath,
    Material,
    NonlinearFEProblem,
    Plasticity,
    homogenize,
    make_problem,
    simulate_point,
)
from morphogen.homogenization import nonlinear as nl
from morphogen.homogenization.behaviours import state_width, update_points
from morphogen.homogenization.elements import tensor_to_voigt

E, NU = 200e3, 0.3


def plastic(**kwargs):
    return Material(young=E, poisson=NU, plasticity=Plasticity(**kwargs))


def test_plasticity_parameters():
    spec = {
        "young": 70e3,
        "poisson": 0.33,
        "plasticity": {
            "yield": 100,
            "hardening": 500,
            "isotropic": [{"Q": 80, "b": 20}],
            "kinematic": [{"C": 1e4, "D": 100}, [2e3, 0]],
            "norton": {"K": 50, "n": 5},
        },
    }
    m = Material.from_config(spec)
    assert m.plastic and m.plasticity.viscous
    assert m.plasticity.kinematic == ((1e4, 100.0), (2e3, 0.0))
    assert Material.from_config(m.to_config()) == m
    assert Material.from_config({"bulk": 1e3, "shear": 500, "plasticity": {"R0": 1}}).plastic
    for bad, match in [
        ({"yield": -1}, "yield"),
        ({"R0": 1, "norton": {"K": 1}}, "norton"),
        ({"R0": 1, "norton": {"K": 1, "n": 0.5}}, "n >= 1"),
        ({"R0": 1, "isotropic": [[1, 1]] * 4}, "at most"),
        ({"R0": 1, "kinematic": {"C": 1, "D": -1}}, "kinematic"),
        ({"R0": 1, "unknown": 2}, "unknown"),
        ({"hardening": 3}, "yield stress"),
    ]:
        with pytest.raises(ValueError, match=match):
            Plasticity.from_config(bad)
    with pytest.raises(ValueError, match="isotropic elastic"):
        Material(stiffness=np.diag([3.0, 3, 3, 1, 1, 1]), plasticity={"yield": 1.0})
    with pytest.raises(ValueError, match="void"):
        Material(void=True, plasticity={"yield": 1.0})


def test_loading_paths():
    p = LoadingPath.from_spec(
        {"times": [0, 1, 3], "increments": [2, 4], "strain": {"11": [0.01, -0.01]},
         "stress": {"21": 5.0}, "temperature": 20}
    )  # fmt: skip
    assert p.controls == (True, False, False, False, False, False)
    assert p.values[1][0] == 0.01 and p.values[2][5] == 5.0 and p.temperature == (0, 20, 20)
    assert np.allclose(p.schedule(), [0.5, 1, 1.5, 2, 2.5, 3])
    values, temp = p.at(2.0)
    assert values[0] == pytest.approx(0.0) and values[5] == 5.0 and temp == 20
    assert LoadingPath.from_spec(p.to_dict()) == p
    parsed = LoadingPath.parse("t=0,1,3 n=2,4 E11=0,0.01,-0.01 S12=5 dT=20")
    assert parsed == p
    tension = LoadingPath.uniaxial(2, times=[0, 10], stress=100.0, increments=5)
    assert not any(tension.controls) and tension.values[-1][2] == 100.0
    for bad, match in [
        ({"times": [0, 1, 1]}, "increasing"),
        ({"times": [0, 1], "strain": {"11": [0.1, 0.2]}}, "natural state"),
        ({"times": [0, 1], "strain": {"11": 1}, "stress": {"11": 1}}, "twice"),
        ({"times": [0, 1], "strain": {"14": 1}}, "unknown component"),
        ({"times": [0, 1, 2], "strain": {"11": [0, 1, 2, 3]}}, "one value per time"),
        ({"times": [0, 1], "increments": [1, 2]}, "increments"),
        ({"strain": {"11": 1}}, "times"),
    ]:
        with pytest.raises(ValueError, match=match):
            LoadingPath.from_spec(bad)
    with pytest.raises(ValueError, match="unknown item"):
        LoadingPath.parse("t=0,1 X11=1")


def test_material_point():
    """Uniaxial responses against closed forms: linear isotropic and kinematic
    hardening and Voce's saturation are integrated exactly; Norton's creep; the
    Armstrong-Frederick hardening converges at first order in the step."""
    tension = LoadingPath.uniaxial(0, times=[0, 1], strain=0.01, increments=20)
    res = simulate_point(plastic(yield_stress=300.0, hardening=2000.0, kinematic=[(3000, 0)]),
                         tension)  # fmt: skip
    s = res.stress[:, 0]
    exact = np.where(s <= 300, s / E, s / E + (s - 300) / 5000.0)
    assert np.allclose(res.strain[:, 0], exact, rtol=0, atol=1e-15)
    assert np.abs(res.stress[:, 1:]).max() < 1e-9 and res.iterations.max() <= 2
    voce = plastic(yield_stress=300.0, isotropic=[(200.0, 30.0)])
    res = simulate_point(voce, LoadingPath.uniaxial(0, times=[0, 1], strain=0.05, increments=25))
    s, p = res.stress[:, 0], res.plastic
    assert np.allclose(p, np.maximum(res.strain[:, 0] - s / E, 0.0), atol=1e-12)
    assert np.allclose(s[p > 0], 300 + 200 * (1 - np.exp(-30 * p[p > 0])), rtol=1e-10)
    # creep under 100 MPa (reached in 1 s) for 1000 s: deps_p/dt = (sigma / K)^n
    creep = plastic(yield_stress=0.0, viscosity=500.0, exponent=4.0)
    res = simulate_point(creep, LoadingPath.uniaxial(0, times=[0, 1, 1001], stress=100.0,
                                                     increments=[10, 100]))  # fmt: skip
    rate = (100.0 / 500.0) ** 4
    expected = rate / 5.0 + rate * 1000.0  # the ramp, then the hold
    assert res.plastic[-1] == pytest.approx(expected, rel=1e-4)
    errors = []
    af = plastic(yield_stress=300.0, kinematic=[(1e5, 500.0)])
    for n in (40, 400):
        res = simulate_point(af, LoadingPath.uniaxial(0, times=[0, 1], strain=0.02, increments=n))
        s, ep = res.stress[:, 0], res.plastic
        exact = 300 + 1e5 / 500 * (1 - np.exp(-500 * ep))
        errors.append(np.abs(s - exact)[ep > 0].max())
    assert errors[1] < 0.15 * errors[0]


def test_consistent_tangent():
    """The tangent of the implicit update against finite differences (symmetric part)."""
    m = plastic(yield_stress=200.0, hardening=500.0, isotropic=[(100.0, 20.0)],
                kinematic=[(5e4, 300.0), (1e4, 0.0)], viscosity=80.0, exponent=6.0)  # fmt: skip
    params = m.plasticity.params(*m.lame)[None]
    rng = np.random.default_rng(1)
    width = state_width(2)
    for viscous in (True, False):
        par = params.copy()
        if not viscous:
            par[0, 4:6] = 0.0
        old = np.zeros((1, width))
        old[0, :6] = rng.normal(0, 2e-3, 6)
        old[0, 6] = 0.01
        for k in range(2):  # deviatoric backstresses
            x = rng.normal(0, 30.0, 6)
            x[:3] -= x[:3].mean()
            old[0, 7 + 6 * k : 13 + 6 * k] = x
        strain = rng.normal(0, 4e-3, 6)
        strain[0] += 6e-3

        def update(eps, old=old, par=par):
            new = np.zeros_like(old)
            sig, tan = np.zeros((1, 6)), np.zeros((1, 6, 6))
            data = (
                np.zeros((1, 6, 6)),
                np.zeros((1, 6)),
                par,
                2,
                np.zeros((1, 1, 6)),
                np.zeros((1, 1, 7)),
                np.zeros((1, 1, 1)),
                np.zeros(1, dtype=np.int64),
            )
            failed = update_points(np.array([1]), np.array([0]), np.array([0]), data, eps[None],
                                   0.0, 0.5, old, new, sig, tan, 0, 1.0)  # fmt: skip
            assert failed == 0
            return sig[0], tan[0], new[0]

        _, tan, new = update(strain)
        assert new[6] > old[0, 6]  # a plastic step
        fd = np.zeros((6, 6))
        h = 1e-7
        for j in range(6):
            d = np.zeros(6)
            d[j] = h
            fd[:, j] = (update(strain + d)[0] - update(strain - d)[0]) / (2 * h)
        assert np.allclose(tan, 0.5 * (fd + fd.T), rtol=1e-4, atol=1e-4 * np.abs(fd).max())


def test_homogeneous_element():
    """A homogeneous volume element follows its material point exactly, with linear and
    quadratic elements under periodic and kinematic conditions, along a cyclic path that
    mixes strain and stress components with a change of temperature."""
    m = Material(young=E, poisson=NU, expansion=1e-5, plasticity=Plasticity(
        300.0, isotropic=[(200.0, 30.0)], kinematic=[(5e4, 300.0)], viscosity=150.0,
        exponent=6.0))  # fmt: skip
    path = {"times": [0, 10, 20, 30], "increments": [8, 8, 8],
            "strain": {"11": [0.01, -0.01, 0.0]}, "stress": {"12": [40, -40, 0]},
            "temperature": [50, 50, 0]}  # fmt: skip
    ref = simulate_point(m, path)
    for order, bc in [(1, "periodic"), (1, "kubc"), (2, "periodic")]:
        mesh = box_mesh((2, 2, 2), order=order, perturb=0.2)
        res = NonlinearFEProblem(mesh, {0: m}, bc).run(path)
        assert np.allclose(res.stress, ref.stress, rtol=0, atol=2e-6 * np.abs(ref.stress).max())
        assert np.allclose(res.strain, ref.strain, rtol=0, atol=1e-9)
        assert res.plastic[-1] == pytest.approx(ref.plastic[-1], rel=1e-6)
        assert res.phases[0]["plastic"][-1] == pytest.approx(ref.plastic[-1], rel=1e-6)


def test_elastic_paths_match_condensation():
    """With linear behaviours, a path reaches the solution of the condensed linear
    problem in one Newton iteration."""
    m0 = Material(young=2.8, poisson=0.39, expansion=95e-6)
    m1 = Material(young=89.0, poisson=0.3, expansion=5.5e-6)
    rng = np.random.default_rng(3)
    mesh = box_mesh((3, 3, 3), phase=lambda c: (rng.random(len(c)) < 0.35).astype(int),
                    perturb=0.2)  # fmt: skip
    mats = {0: m0, 1: m1}
    loading = Loading.mixed(strain={"11": 0.01}, stress={"22": 0, "33": 0, "23": 0, "31": 0,
                            "12": 0.002}, temperature=50.0)  # fmt: skip
    for bc in ("periodic", "kubc"):
        sol = make_problem(mesh, mats, "elastic", bc).solve(loading)
        res = NonlinearFEProblem(mesh, mats, bc).run(
            {"times": [0, 1], "increments": 1, "strain": {"11": 0.01}, "stress": {"12": 0.002},
             "temperature": 50.0}
        )  # fmt: skip
        assert np.allclose(res.strain[-1], tensor_to_voigt(sol.strain, strain=True), atol=1e-14)
        assert np.allclose(res.stress[-1], tensor_to_voigt(sol.stress, strain=False), atol=1e-12)
        assert res.iterations.tolist() == [1]


def test_plastic_laminate():
    """A periodic laminate of an elasto-viscoplastic and an elastic layer: the finite
    element solution is uniform in each layer and follows the exact laminate response
    along a cyclic path with a shear stress and a change of temperature."""
    soft = Material(young=70e3, poisson=0.33, plasticity=Plasticity(
        100.0, isotropic=[(80.0, 20.0)], kinematic=[(2e4, 200.0)], viscosity=50.0,
        exponent=5.0))  # fmt: skip
    hard = Material(young=E, poisson=NU, expansion=12e-6)
    path = {"times": [0, 1, 2, 3], "increments": [6, 6, 6],
            "strain": {"11": [0.006, -0.004, 0.003]}, "stress": {"12": [20, -20, 0]},
            "temperature": [30, 30, 0]}  # fmt: skip
    strain, stress = laminate_path([soft, hard], [0.5, 0.5], path)
    mesh = box_mesh((2, 2, 4), upper=(1.0, 1.0, 1.0), phase=lambda c: (c[:, 2] > 0.5).astype(int))
    res = NonlinearFEProblem(mesh, {0: soft, 1: hard}, "periodic").run(path, fields=True)
    assert np.allclose(res.stress, stress, rtol=0, atol=2e-6 * np.abs(stress).max())
    assert np.allclose(res.strain, strain, rtol=0, atol=1e-8)
    assert sorted(res.fields) == [1.0, 2.0, 3.0]
    fields = res.fields[3.0]
    soft_p = fields["p"][res.support.phase == 0]
    assert np.ptp(soft_p) < 1e-6 * soft_p.max() and np.all(fields["p"][res.support.phase == 1] == 0)
    assert res.phases[0]["plastic"][-1] == pytest.approx(soft_p.mean(), rel=1e-9)


def test_steps_are_divided():
    """A step that does not converge is divided; Voce's saturation is integrated
    exactly whatever the steps."""
    m = plastic(yield_stress=300.0, isotropic=[(200.0, 30.0)])
    mesh = box_mesh((1, 1, 1))
    res = NonlinearFEProblem(mesh, {0: m}, "periodic", max_newton=1).run(
        LoadingPath.uniaxial(0, times=[0, 1], strain=0.05, increments=1)
    )
    assert res.info["cuts"] > 0 and res.n_steps > 1
    s = res.stress[-1, 0]
    p = res.plastic[-1]
    assert s == pytest.approx(300 + 200 * (1 - np.exp(-30 * p)), rel=1e-6)
    with pytest.raises(RuntimeError, match="no convergence"):
        NonlinearFEProblem(mesh, {0: m}, "periodic", max_newton=0, max_cuts=2).run(
            LoadingPath.uniaxial(0, times=[0, 1], strain=0.05, increments=1)
        )


def test_prediction_of_a_step():
    """The prediction of a step from the steps before it (``nonlinear._predicted``): at
    the first order it is exact for unknowns that follow a straight line in time, at the
    second for a parabola, whatever the steps. The next step is predicted at the second
    order where it would have predicted the last step clearly better than the first, on
    a response that does not turn (``nonlinear._second_order``)."""
    rng = np.random.default_rng(0)
    b = 1.0 + 0.2 * rng.normal(size=7)  # the fluctuation: q(t) = b t + c t^2
    c = 0.1 * rng.normal(size=7)
    h = 1.0 + 0.2 * rng.normal(size=2)  # the macroscopic unknowns likewise
    k = 0.1 * rng.normal(size=2)
    times = np.array([0.0, 0.3, 0.8, 1.0, 1.7])

    def step(i, curved=1.0):
        """The converged step from ``times[i - 1]`` to ``times[i]`` (segment 0)."""
        t0, t1 = times[i - 1], times[i]
        dq = b * (t1 - t0) + curved * c * (t1**2 - t0**2)
        de = h * (t1 - t0) + curved * k * (t1**2 - t0**2)
        return (0, t1 - t0, dq, de)

    last, target = step(2, 0.0), step(3, 0.0)  # a straight line
    dq, de = nl._predicted(target[1], last)
    assert np.allclose(dq, target[2], rtol=1e-13) and np.allclose(de, target[3], rtol=1e-13)
    before, last, target = step(2), step(3), step(4)  # a parabola, unequal steps
    dq, de = nl._predicted(target[1], last, before)
    assert np.allclose(dq, target[2], rtol=1e-12, atol=1e-13)
    assert np.allclose(de, target[3], rtol=1e-12, atol=1e-13)
    first = nl._predicted(target[1], last)[0]
    assert 0.01 < np.linalg.norm(first - target[2]) / np.linalg.norm(target[2]) < 0.25
    assert nl._second_order(target, last, before)

    def steps(*rates):
        """Steps of one unit of time whose unknowns move at these rates."""
        return [(0, 1.0, rate * np.ones(3), rate * np.ones(1)) for rate in rates]

    # rates 4, 5, then 6: the second order is exact, the first one a sixth off
    one, two, three = steps(4.0, 5.0, 6.0)
    assert nl._second_order(three, two, one)
    # a response that turns: rates 1, 2, then 2 again (the first order predicts 2, the
    # second 3); rates 1, 2, then 3 (the second order is exact, but the first one is a
    # third off for the fluctuation and for the macroscopic unknowns: too far)
    one, two, three = steps(1.0, 2.0, 2.0)
    assert not nl._second_order(three, two, one)
    one, two, three = steps(1.0, 2.0, 3.0)
    assert not nl._second_order(three, two, one)
    # the steps of another segment of the path, or too few steps: the first order
    elsewhere = (1, *before[1:])
    assert not nl._second_order(target, last, elsewhere)
    assert not nl._second_order(target, last, None)
    assert not nl._second_order(target, None, None)
    assert np.array_equal(nl._predicted(target[1], last, elsewhere)[0], first)


def test_secant_of_the_condensed_block():
    """The condensed block of the macroscopic unknowns kept from an earlier linearization,
    updated by the secant of a correction (``nonlinear._secant``): the updated block maps
    the step to the change of the condensed residual. Symmetric, it stays symmetric
    positive definite (BFGS), a secant whose curvature along the step is far from that
    of the block being damped (Powell: a fifth of it at least, five times at most);
    otherwise it changes along the step only (Broyden), by five times its image at
    most."""
    rng = np.random.default_rng(1)
    m, p = rng.normal(size=(2, 5, 5))
    true = m @ m.T + 5.0 * np.eye(5)  # the block of the tangents as they are
    kept = true + 0.2 * p @ p.T  # the one of an earlier linearization
    step, before = rng.normal(size=(2, 5))
    new = nl._secant(kept, step, before, before + true @ step, True)
    assert np.allclose(new @ step, true @ step, rtol=1e-12)
    assert np.allclose(new, new.T, rtol=0, atol=1e-13 * np.abs(new).max())
    assert np.linalg.eigvalsh(new).min() > 0.0
    # the secants of a full set of steps: the block of the tangents, along these steps
    block = kept
    steps = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    for s in steps.T:
        block = nl._secant(block, s, before, before + true @ s, True)
    assert np.allclose(block @ steps[:, -1], true @ steps[:, -1], rtol=1e-12)
    assert np.linalg.norm(block - true) < np.linalg.norm(kept - true)
    # secants far from the block: a curvature of the other sign, or fifty times larger
    energy = step @ kept @ step
    for factor, bound in ((-1.0, 0.2), (50.0, 5.0)):
        damped = nl._secant(kept, step, before, before + factor * (kept @ step), True)
        assert step @ damped @ step == pytest.approx(bound * energy, rel=1e-12)
        assert np.allclose(damped, damped.T, rtol=0, atol=1e-13 * np.abs(damped).max())
        assert np.linalg.eigvalsh(damped).min() > 0.0
    assert nl._secant(kept, np.zeros(5), before, before, True) is kept
    skew = true + rng.normal(size=(5, 5))  # a block that is not symmetric
    new = nl._secant(kept, step, before, before + skew @ step, False)
    assert np.allclose(new @ step, skew @ step, rtol=1e-12)
    other = rng.normal(size=5)
    other -= (other @ step) / (step @ step) * step
    assert np.allclose(new @ other, kept @ other, rtol=1e-12)
    damped = nl._secant(kept, step, before, before + 100.0 * (kept @ step), False)
    moved = np.linalg.norm((damped - kept) @ step)
    assert moved == pytest.approx(5.0 * np.linalg.norm(kept @ step), rel=1e-12)
    assert np.allclose(damped @ other, kept @ other, rtol=1e-12)
    assert nl._secant(kept, np.zeros(5), before, before, False) is kept


def test_homogenize_paths(tmp_path):
    soft = plastic(yield_stress=100.0, hardening=1000.0)
    mats = {0: soft, 1: Material(young=E, poisson=NU)}
    mesh = box_mesh((2, 2, 2), phase=lambda c: (c[:, 0] > 0.5).astype(int))
    paths = {"tension": {"times": [0, 1], "increments": 4, "strain": {"11": 0.004}}}
    out = homogenize(mesh, mats, bcs=["periodic", "kubc", "mixed"], properties=[], paths=paths)
    assert sorted(out) == ["tension.kubc", "tension.periodic"]
    res = out["tension.periodic"]
    assert res.n_steps == 4 and res.components()["E11"] == pytest.approx(0.004)
    assert res.row()["property"] == "tension" and len(res.rows()) == 5
    image = mg.image.VoxelImage(np.zeros((4, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="conforming mesh"):  # images: FFT, periodic only
        homogenize(image, mats, bcs=["kubc"], properties=[], paths=paths)
    with pytest.raises(ValueError, match="periodic or kinematic"):
        NonlinearFEProblem(mesh, mats, "subc")
    # command line, on a mesh file
    mesh_file = tmp_path / "box.vtu"
    mg.io.save_mesh(mesh, mesh_file)
    (tmp_path / "mats.yaml").write_text(
        yaml.safe_dump({0: soft.to_config(), 1: {"young": E, "poisson": NU}}), encoding="utf-8"
    )
    report = tmp_path / "report.json"
    code = cli.main([
        "homogenize", str(mesh_file), "--materials", str(tmp_path / "mats.yaml"),
        "--property", "bulk_modulus", "--path", "tension: t=0,1 n=4 E11=0.004",
        "--curves", str(tmp_path / "curves"), "--periodic", "-o", str(report),
    ])  # fmt: skip
    assert code == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["paths"]["tension.periodic"]["S11"] == pytest.approx(res.stress[-1, 0], rel=1e-6)
    assert (tmp_path / "curves" / "box_tension_periodic.csv").exists()


def test_configuration_paths():
    base = {
        "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.2},
        "domain": {"size": 4.0, "periodic": True},
        "mesh": {"size": 1.0},
        "homogenization": {
            "materials": {0: {"young": 7e4, "poisson": 0.33, "plasticity": {"yield": 100}},
                          1: {"young": 4e5, "poisson": 0.2}},
            "paths": {"tension": {"times": [0, 1], "strain": {"11": 0.01}}},
            "nonlinear": {"tolerance": 1e-7, "max_iterations": 20},
        },
    }  # fmt: skip
    cfg = load_config(base).homogenization
    assert cfg.properties == [] and cfg.method == "fe"
    assert cfg.options()["nonlinear"] == {"newton_tol": 1e-7, "max_newton": 20}
    bad = [
        ({"paths": {"bulk_modulus": {"times": [0, 1]}}}, "not a valid name"),
        ({"paths": {"a": {"times": [1, 0]}}}, "increasing"),
        ({"loadings": {"tension": {"strain": {"11": 0.01}}}}, "both a loading and a path"),
        ({"nonlinear": {"tol": 1}}, "unknown"),
        ({"boundary_conditions": ["subc"], "materials": {0: {"young": 1, "poisson": 0.3}}},
         "periodic or kubc"),
        ({"method": "fft", "boundary_conditions": ["kubc"]}, "the FFT solver is periodic"),
        ({"nonlinear": {"integration": "simpson"}}, "integration"),
    ]  # fmt: skip
    for change, match in bad:
        data = json.loads(json.dumps(base))
        data["homogenization"].update(change)
        if change.get("method") == "fft":
            data.pop("mesh")
            data["voxels"] = {"shape": 8}
        with pytest.raises(ConfigError, match=match):
            load_config(data)
    # the integration of the elements of a mesh: every point, or their mean dilatation
    data = json.loads(json.dumps(base))
    data["homogenization"].update(integration="reduced", nonlinear={"integration": "full"})
    options = load_config(data).homogenization.options()
    assert options["integration"] == "bbar" and options["nonlinear"] == {"integration": "full"}
    # loading paths on voxels (FFT)
    data = json.loads(json.dumps(base))
    data.pop("mesh")
    data["voxels"] = {"shape": 8}
    data["homogenization"].update(method="fft", nonlinear={"integration": "full"})
    cfg = load_config(data).homogenization
    assert cfg.method == "fft" and cfg.options()["nonlinear"] == {"integration": "full"}
