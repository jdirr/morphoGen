"""The nonuniform transformation field analysis (NTFA) against full-field simulations:
exact for uniform fields, its reduced operators, a composite along trained and other
paths, viscoplasticity, storage."""

from __future__ import annotations

import numpy as np
import pytest

from morphogen.homogenization.apparent import stiffness_tensor
from morphogen.homogenization.behaviours import Plasticity
from morphogen.homogenization.fft import FFTElasticProblem
from morphogen.homogenization.fft_nonlinear import NonlinearFFTProblem
from morphogen.homogenization.materials import Material
from morphogen.homogenization.ntfa import NTFAModel, train_ntfa
from morphogen.homogenization.paths import LoadingPath
from morphogen.image import VoxelImage

LAW = Plasticity(100.0, hardening=1000.0, isotropic=((30.0, 50.0),))
TENSION = LoadingPath.uniaxial(0, times=[0, 1], strain=[0, 0.01], increments=10)


def _full(image, materials, path):
    return NonlinearFFTProblem(image, materials, "periodic", integration="full").run(path)


def _sphere(n):
    g = (np.indices((n, n, n)).transpose(1, 2, 3, 0) + 0.5) / n - 0.5
    labels = (np.linalg.norm(g, axis=-1) < 0.3).astype(np.uint8)  # a stiff sphere, fraction 0.11
    return VoxelImage(labels, spacing=(1.0 / n,) * 3, periodic=True)


COMPOSITE = {
    0: Material(young=70e3, poisson=0.3, plasticity=LAW),
    1: Material(young=400e3, poisson=0.2),
}


@pytest.mark.parametrize("viscous", [False, True])
def test_uniform_fields_are_exact(viscous):
    law = Plasticity(
        100.0,
        hardening=1000.0,
        viscosity=50.0 if viscous else None,
        exponent=4.0 if viscous else None,
    )
    image = VoxelImage(np.zeros((6, 6, 6), np.uint8), spacing=(1 / 6,) * 3, periodic=True)
    mats = {0: Material(young=70e3, poisson=0.3, plasticity=law)}
    path = LoadingPath.uniaxial(0, times=[0, 1, 2], strain=[0, 0.01, -0.004], increments=8)
    model = train_ntfa(image, mats, [path], modes=2)
    assert model.n_modes == 1  # uniform snapshots: one mode
    ref, red = _full(image, mats, path), model.run(path)
    np.testing.assert_allclose(red.stress, ref.stress, atol=1e-3)  # the full field's tolerance
    np.testing.assert_allclose(red.strain, ref.strain, atol=1e-8)


@pytest.fixture(scope="module")
def composite():
    image = _sphere(20)
    shear = LoadingPath.from_spec({"times": [0, 1], "strain": {"12": [0, 0.02]}, "increments": 10})
    return image, train_ntfa(image, COMPOSITE, [TENSION, shear], modes=6, keep_fields=True)


@pytest.mark.slow
def test_reduced_operators(composite):
    image, model = composite
    c, _ = stiffness_tensor(FFTElasticProblem(image, COMPOSITE, "periodic", tol=1e-11))
    np.testing.assert_allclose(model.stiffness, c, rtol=1e-9, atol=1e-6)  # the elastic limit
    # reciprocity (Hill's lemma): rho_k = -c_r a_k; c_r d symmetric
    frac = np.array([model.fractions[r] for r in model.mode_phase])
    np.testing.assert_allclose(model.rho, -frac[:, None] * model.a, rtol=1e-8, atol=1e-6)
    cd = frac[:, None] * model.d
    np.testing.assert_allclose(cd, cd.T, rtol=1e-8, atol=1e-6)
    assert model.n_modes == 6 and model.energy[0] > 0.999


@pytest.mark.slow
def test_composite_against_full_fields(composite):
    image, model = composite
    cycle = LoadingPath.uniaxial(0, times=[0, 1, 2], strain=[0, 0.01, -0.006], increments=10)
    shear = LoadingPath.from_spec({"times": [0, 1], "strain": {"12": [0, 0.02]}, "increments": 10})
    # largest errors: 2.5 % (tension), 4.3 % (shear), 7.7 % just after the reversal of the
    # cycle (2.7 % at its end): 20^3 voxels, six modes trained on tension and shear
    for path, k, tol in ((TENSION, 0, 0.04), (cycle, 0, 0.09), (shear, 5, 0.06)):
        ref, red = _full(image, COMPOSITE, path), model.run(path)
        err = np.abs(red.stress[:, k] - ref.stress[:, k]).max() / np.abs(ref.stress[:, k]).max()
        assert err < tol
        assert abs(red.stress[-1, k] - ref.stress[-1, k]) < 0.05 * abs(ref.stress[-1, k])
        # the lateral stresses stay free (mixed control)
        free = [i for i in range(6) if path.controls[i] is False]
        assert np.abs(red.stress[:, free]).max() < 1e-6 * np.abs(red.stress).max()
    assert red.info["solver"] == "ntfa" and red.phases[0]["plastic"][-1] > 0
    # the local fields of the reduced state: their mean stress is the macroscopic one
    out = model.run(TENSION)
    fields = model.local_fields(out.strain[-1], np.zeros(model.n_modes))
    np.testing.assert_allclose(
        fields["stress"].mean(axis=0), model.stiffness @ out.strain[-1], rtol=1e-8, atol=1e-6
    )


@pytest.mark.slow
def test_storage(composite, tmp_path):
    _, model = composite
    model.save(tmp_path / "ntfa.npz")
    again = NTFAModel.load(tmp_path / "ntfa.npz")
    np.testing.assert_allclose(again.run(TENSION).stress, model.run(TENSION).stress, rtol=1e-12)
    with pytest.raises(ValueError, match="temperature"):
        model.run({"times": [0, 1], "strain": {"11": [0, 0.01]}, "temperature": [0, 10]})


@pytest.mark.slow
def test_phase_means(composite):
    _, model = composite
    res = model.run(TENSION)
    counts = {0: 0.0, 1: 0.0}
    for r in res.phases:
        counts[r] = model.fractions.get(r, 1.0 - sum(model.fractions.values()))
    mean = sum(counts[r] * res.phases[r]["stress"] for r in res.phases)
    np.testing.assert_allclose(mean, res.stress, rtol=1e-8, atol=1e-6)
    strain = sum(counts[r] * res.phases[r]["strain"] for r in res.phases)
    np.testing.assert_allclose(strain, res.strain, rtol=1e-8, atol=1e-12)
    assert res.rows()[-1]["S11_1"] > res.rows()[-1]["S11_0"]  # the stiff spheres carry more


def test_configuration_and_homogenize(tmp_path):
    from morphogen import config
    from morphogen.homogenization.apparent import homogenize

    paths = {"pull": {"times": [0, 1], "increments": [6], "strain": {"11": [0, 0.008]}}}
    data = {
        "materials": {
            0: {"young": 70e3, "poisson": 0.3, "plasticity": {"yield": 100, "hardening": 1000}},
            1: {"young": 400e3, "poisson": 0.2},
        },
        "method": "fft",
        "paths": paths,
        "ntfa": {"training": paths, "modes": 3, "save": "model.npz"},
    }
    spec = config._homogenization(data, tmp_path)
    assert spec.ntfa["modes"] == 3 and spec.ntfa["save"] == str(tmp_path / "model.npz")
    with pytest.raises(config.ConfigError, match="paths to follow"):
        config._homogenization({**data, "paths": {}})
    with pytest.raises(config.ConfigError, match=r"ntfa.modes"):
        config._homogenization({**data, "ntfa": {"training": paths, "modes": 0}})
    out = homogenize(
        _sphere(10),
        COMPOSITE,
        properties=(),
        **{k: v for k, v in spec.options().items() if k in ("paths", "ntfa")},
    )
    assert set(out) == {"pull.ntfa"} and out["pull.ntfa"].bc == "ntfa"
    assert (tmp_path / "model.npz").exists()


@pytest.mark.slow
@pytest.mark.parametrize("viscous", [False, True])
def test_kinematic_hardening(viscous):
    """Armstrong-Frederick and Prager backstresses: exact for uniform fields; a composite
    along cycles (the Bauschinger effect of the matrix and of the internal stresses)."""
    law = Plasticity(
        100.0,
        hardening=200.0,
        isotropic=((40.0, 20.0),),
        kinematic=((20000.0, 150.0), (5000.0, 0.0)),
        viscosity=50.0 if viscous else None,
        exponent=5.0 if viscous else None,
    )
    image = VoxelImage(np.zeros((6, 6, 6), np.uint8), spacing=(1 / 6,) * 3, periodic=True)
    mats = {0: Material(young=70e3, poisson=0.3, plasticity=law)}
    path = LoadingPath.uniaxial(0, times=[0, 1, 2, 3], strain=[0, 0.01, -0.01, 0.01], increments=10)
    ref, red = _full(image, mats, path), train_ntfa(image, mats, [path], modes=2).run(path)
    np.testing.assert_allclose(red.stress, ref.stress, atol=1e-3)
    if viscous:
        return
    mats = {
        0: Material(young=70e3, poisson=0.33, plasticity=law),
        1: Material(young=400e3, poisson=0.2),
    }
    image = _sphere(16)
    cycle = LoadingPath.uniaxial(0, times=[0, 1, 2], strain=[0, 0.01, -0.006], increments=8)
    model = train_ntfa(image, mats, [cycle], modes=6)
    two = LoadingPath.uniaxial(
        0, times=[0, 1, 2, 3, 4], strain=[0, 0.008, -0.008, 0.008, -0.008], increments=6
    )
    ref, red = _full(image, mats, two), model.run(two)
    err = np.abs(red.stress[:, 0] - ref.stress[:, 0]) / np.abs(ref.stress[:, 0]).max()
    assert err.max() < 0.08 and err[-1] < 0.04
