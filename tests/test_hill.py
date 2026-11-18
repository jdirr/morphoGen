"""Hill's anisotropic plasticity (:class:`morphogen.homogenization.Plasticity` with ``hill``,
the implicit update of :mod:`morphogen.homogenization.behaviours`): von Mises when its
coefficients are all 1 (the general update against the radial return, with isotropic and
kinematic hardenings, viscous or not), the yield stresses along the axes and in the shears
of distinct coefficients and the ratio of the lateral plastic strains (closed forms), the
consistent tangent (finite differences), homogeneous voxels and elements following their
material point in small and finite strains, and the configurations."""

import numpy as np
import pytest

from helpers import box_mesh
from morphogen import gpu
from morphogen.homogenization import (
    Material,
    NonlinearFEProblem,
    NonlinearFFTProblem,
    Plasticity,
    simulate_point,
)
from morphogen.homogenization.behaviours import _hill_update, _hill_work, state_width
from morphogen.homogenization.materials import lame
from morphogen.image import VoxelImage

E, NU = 200e3, 0.3
HILL = {"11": 1.3, "22": 0.8, "33": 1.1, "12": 0.7, "23": 1.25, "31": 0.9}
HARDENING = {"yield": 200.0, "hardening": 1000.0, "isotropic": [{"Q": 50.0, "b": 20.0}],
             "kinematic": [{"C": 20000.0, "D": 100.0}]}  # fmt: skip
CYCLE = {"times": [0, 1, 3, 4], "increments": [10, 20, 10],
         "strain": {"11": [0, 0.01, -0.01, 0.0], "12": [0, 0.004, 0.0, 0.002]}}  # fmt: skip
VOIGT = {"11": 0, "22": 1, "33": 2, "23": 3, "31": 4, "12": 5}


def _material(**plasticity):
    return Material.from_config({"young": E, "poisson": NU, "plasticity": plasticity})


@pytest.mark.parametrize("norton", [None, {"K": 50.0, "n": 5.0}])
def test_unit_coefficients_are_von_mises(norton):
    """Hill's update with its coefficients all 1 (Newton's method on the plastic strain
    increment and dp) gives the radial return of von Mises along a cycle of tension and
    shear, with isotropic and kinematic hardenings, rate-independent or viscous."""
    law = {**HARDENING, **({"norton": norton} if norton else {})}
    mises = simulate_point(_material(**law), CYCLE)
    hill = simulate_point(_material(**law, hill={}), CYCLE)
    scale = np.abs(mises.stress).max()
    assert np.abs(hill.stress - mises.stress).max() < 1e-10 * scale
    assert hill.plastic[-1] == pytest.approx(mises.plastic[-1], rel=1e-10)
    assert mises.plastic[-1] > 0.01


def test_yield_stresses_and_flow():
    """Perfect plasticity of distinct coefficients: the stress saturates at the yield stress
    of Hill's criterion in tension along each axis, R0 sqrt(6 / (4 h11 + h22 + h33)), and in
    shear in each plane, R0 / sqrt(3 h12) (the components 12, 23 and 31 told apart); in
    tension along 1 the lateral plastic strains are in the ratio of the deviator of M s."""
    mat = _material(**{"yield": 200.0, "hill": HILL})
    expected = mat.plasticity.yield_stresses()
    assert expected["12"] == pytest.approx(200.0 / np.sqrt(3 * 0.7))
    for c, k in VOIGT.items():
        strain = 0.01 if c[0] == c[1] else 0.02
        res = simulate_point(mat, {"times": [0, 1], "increments": 20, "strain": {c: [0, strain]}})
        assert res.stress[-1, k] == pytest.approx(expected[c], rel=1e-9), c
    res = simulate_point(mat, {"times": [0, 1], "increments": 20, "strain": {"11": [0, 0.02]}})
    d = res.strain[-1] - res.strain[-2]  # plastic only: the stress no longer changes
    h = np.array([HILL["11"], HILL["22"], HILL["33"]])
    ms = h * np.array([2.0, -1.0, -1.0]) / 3.0
    dev = ms - ms.mean()
    assert d[1] / d[2] == pytest.approx(dev[1] / dev[2], rel=1e-9)
    assert d[0] + d[1] + d[2] == pytest.approx(0.0, abs=1e-12)  # incompressible flow
    von_mises = Plasticity(200.0).yield_stresses()
    assert von_mises["11"] == pytest.approx(200.0) and von_mises["12"] == pytest.approx(
        200.0 / np.sqrt(3)
    )


@pytest.mark.parametrize("viscous", [False, True])
def test_consistent_tangent(viscous):
    """The tangent of a plastic step (two kinematic terms, one linear) against the finite
    differences of the stress, symmetrized as the kernel symmetrizes it."""
    law = Plasticity(200.0, hardening=1000.0, isotropic=((50.0, 20.0),),
                     kinematic=((20000.0, 100.0), (5000.0, 0.0)),
                     viscosity=50.0 if viscous else None, exponent=5.0 if viscous else None,
                     hill=tuple(HILL.values()))  # fmt: skip
    par = law.params(*lame(E, NU))
    rng = np.random.default_rng(1)
    old = np.zeros(state_width(2))
    old[:6] = rng.normal(0, 1e-3, 6)
    old[:3] -= old[:3].mean()
    old[6] = 0.01
    old[7:] = rng.normal(0, 20.0, 12)
    for k in range(2):
        old[7 + 6 * k : 10 + 6 * k] -= old[7 + 6 * k : 10 + 6 * k].mean()
    eps, eth, dt = rng.normal(0, 4e-3, 6), np.zeros(6), 0.1

    def update(e):
        new, sig, tan = np.zeros_like(old), np.zeros(6), np.zeros((6, 6))
        assert _hill_update(par, 2, e, eth, dt, old, new, sig, tan, _hill_work()) == 0
        return new, sig, tan

    new, _, tan = update(eps)
    assert new[6] - old[6] > 1e-3  # a plastic step
    fd = np.zeros((6, 6))
    for c in range(6):
        step = np.zeros(6)
        step[c] = 1e-8
        fd[:, c] = (update(eps + step)[1] - update(eps - step)[1]) / 2e-8
    assert np.abs(tan - 0.5 * (fd + fd.T)).max() < 1e-7 * np.abs(fd).max()


def test_solvers_follow_the_material_point():
    """Homogeneous voxels (FFT) and elements (finite elements) of Hill's criterion follow
    their material point along a path mixing a strain and a shear stress, in small and
    finite strains, on the GPU too where there is one."""
    m = _material(**HARDENING, norton={"K": 50.0, "n": 5.0}, hill=HILL)
    path = {"times": [0, 1, 2], "increments": [6, 6], "strain": {"11": [0.01, -0.005]},
            "stress": {"12": [60, -60]}}  # fmt: skip
    image = VoxelImage(np.zeros((2, 2, 3), dtype=np.uint8))
    for kinematics in ("small", "finite"):
        ref = simulate_point(m, path, kinematics=kinematics)
        scale = np.abs(ref.stress).max()
        res = NonlinearFFTProblem(image, {0: m}, kinematics=kinematics).run(path)
        assert np.allclose(res.stress, ref.stress, rtol=0, atol=2e-6 * scale)
        assert res.plastic[-1] == pytest.approx(ref.plastic[-1], rel=1e-6)
        mesh = box_mesh((1, 1, 1), perturb=0.2)
        res = NonlinearFEProblem(mesh, {0: m}, "periodic", kinematics=kinematics).run(path)
        assert np.allclose(res.stress, ref.stress, rtol=0, atol=2e-6 * scale)
        if gpu.available():
            res = NonlinearFFTProblem(image, {0: m}, kinematics=kinematics, device="gpu").run(path)
            assert np.allclose(res.stress, ref.stress, rtol=0, atol=2e-6 * scale)


def test_configurations():
    """Hill's coefficients by their components (1 where not given), written back; the
    errors of the configurations; the NTFA, which reduces von Mises phases, refuses them."""
    law = Plasticity.from_config({"yield": 200.0, "hill": {"11": 1.3, 22: 0.8}})
    assert law.hill == (1.3, 0.8, 1.0, 1.0, 1.0, 1.0)
    assert law.to_config()["hill"] == {"11": 1.3, "22": 0.8, "33": 1.0, "12": 1.0, "23": 1.0,
                                       "31": 1.0}  # fmt: skip
    assert Plasticity.from_config(law.to_config()) == law
    assert "hill" not in Plasticity(200.0).to_config()
    row = law.params(*lame(E, NU))
    assert row[-6:].tolist() == [1.3, 0.8, 1.0, 1.0, 1.0, 1.0]  # Voigt order: 23, 31, 12
    assert Plasticity(200.0, hill=(1, 1, 1, 0.5, 2, 3)).params(1.0, 1.0)[-3:].tolist() == [
        2.0,
        3.0,
        0.5,
    ]
    for spec, message in (
        ({"13": 1.0}, "unknown components"),
        ({"11": 0.0}, "positive coefficients"),
        ((1.0, 1.0), "six coefficients"),
    ):
        with pytest.raises(ValueError, match=message):
            Plasticity.from_config({"yield": 1.0, "hill": spec})
    from morphogen.homogenization.ntfa import train_ntfa
    from morphogen.io.mfront import mfront_source, mfront_symmetry

    # the MFront source: an orthotropic behaviour with TFEL's F ... N (in WSL, the law
    # compiled against the native kernel: tests/test_mfront_export.py)
    material = _material(**{"yield": 1.0, "hill": HILL})
    source = mfront_source(material, "Hill")
    assert "@OrthotropicBehaviour;" in source and 'criterion : "Hill" {F : ' in source
    assert mfront_symmetry(material) == "orthotropic"

    image = VoxelImage(np.zeros((4, 4, 4), dtype=np.uint8), periodic=True)
    with pytest.raises(ValueError, match="not Hill's criterion"):
        train_ntfa(image, {0: _material(**{"yield": 1.0, "hill": HILL})}, [CYCLE])
