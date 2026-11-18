"""FFT solver of periodic voxel images (trilinear elements, Fourier preconditioner)."""

import numpy as np
import pytest

import morphogen as mg
from helpers import backus
from morphogen.homogenization import (
    VOID,
    FFTElasticProblem,
    FFTThermalProblem,
    Loading,
    Material,
    apparent_bulk_modulus,
    apparent_conductivity,
    apparent_shear_modulus,
    bounds,
    conductivity_tensor,
    homogenize,
    stiffness_tensor,
)
from morphogen.homogenization.elements import isotropic_stiffness, tensor_to_voigt
from morphogen.homogenization.fft import apply_operator

SOFT = Material(young=1.0, poisson=0.2, conductivity=1.0)
STIFF = Material(young=20.0, poisson=0.35, conductivity=30.0)
MATS = {0: SOFT, 1: STIFF}


def random_image(shape, fraction, seed, spacing=1.0):
    rng = np.random.default_rng(seed)
    data = (rng.random(shape) < fraction).astype(np.uint8)
    return mg.VoxelImage(data, spacing=spacing, periodic=True)


def dense_operator(problem: FFTElasticProblem) -> np.ndarray:
    """Matrix of the periodic voxel operator, column by column."""
    nx, ny, nz = problem.shape
    mats = problem.element_matrices
    n = problem.nd * nx * ny * nz
    k = np.zeros((n, n))
    unit = np.zeros(n)
    out = np.empty(n)
    for j in range(n):
        unit[j] = 1.0
        apply_operator(nx, ny, nz, problem.nd, problem.mat, mats, unit, out)
        k[:, j] = out
        unit[j] = 0.0
    return k


def test_solution_matches_a_direct_solve():
    img = random_image((3, 4, 3), 0.4, seed=1, spacing=(0.5, 0.7, 0.4))
    problem = FFTElasticProblem(img, MATS, tol=1e-12)
    k = dense_operator(problem)
    assert np.allclose(k, k.T, atol=1e-12 * np.abs(k).max())
    strain = np.array([0.2, -0.1, 0.05, 0.03, -0.02, 0.07])
    sol = problem.solve(Loading.strain(strain), fields=True)
    # same averages from the minimum-norm solution of the singular dense system
    el = problem.element
    from morphogen.homogenization import kernels

    u_loc = el.affine_displacement(strain)
    dof = np.arange(k.shape[0], dtype=np.int64)
    loads = np.ascontiguousarray(problem.element_matrices @ u_loc)
    rhs = kernels.element_load(3, 4, 3, True, problem.mat, dof, 3, loads, dof.size)
    w = np.linalg.lstsq(k, rhs, rcond=None)[0]
    sums = kernels.elastic_sums(
        3,
        4,
        3,
        True,
        problem.mat,
        problem.stat,
        len(problem.table.phases),
        problem.table.matrices,
        problem.table.eigen,
        0.0,
        w,
        u_loc,
        el.b_gauss,
        el.weight,
    )
    stress = sums.sum(axis=0)[6:12] / img.volume
    np.testing.assert_allclose(tensor_to_voigt(sol.stress, strain=False), stress, rtol=1e-9)
    assert sol.info.method == "fft" and sol.info.residual < 1e-12
    assert sol.support is img and sol.fields["stress"].shape == (3, 4, 3, 6)


def test_homogeneous_medium_and_laminate_are_exact():
    img = mg.VoxelImage(np.ones((4, 3, 5), dtype=np.uint8), spacing=(0.5, 0.7, 0.3), periodic=True)
    assert apparent_bulk_modulus(FFTElasticProblem(img, MATS)).value == pytest.approx(STIFF.bulk)
    res = apparent_shear_modulus(FFTElasticProblem(img, MATS), (1, 2))
    assert res.value == pytest.approx(STIFF.shear) and res.energy == pytest.approx(STIFF.shear)
    assert apparent_conductivity(FFTThermalProblem(img, MATS), 2).value == pytest.approx(30.0)
    data = np.zeros((3, 4, 10), dtype=np.uint8)
    data[:, :, 3:7] = 1
    lam = mg.VoxelImage(data, spacing=(1.0, 1.3, 0.8), periodic=True)
    exact = backus([isotropic_stiffness(*m.lame) for m in (SOFT, STIFF)], [0.6, 0.4])
    c, _ = stiffness_tensor(FFTElasticProblem(lam, MATS, tol=1e-12))
    np.testing.assert_allclose(c, exact, rtol=1e-9, atol=1e-11 * np.abs(exact).max())
    kappa, _ = conductivity_tensor(FFTThermalProblem(lam, MATS, tol=1e-12))
    parallel = 0.6 * 1.0 + 0.4 * 30.0
    series = 1.0 / (0.6 / 1.0 + 0.4 / 30.0)
    np.testing.assert_allclose(np.diag(kappa), [parallel, parallel, series], rtol=1e-9)


def test_translation_invariance_bounds_and_hill_lemma():
    img = random_image((8, 7, 9), 0.35, seed=2)
    rolled = mg.VoxelImage(np.roll(img.data, (2, -1, 3), axis=(0, 1, 2)), periodic=True)
    a, _ = stiffness_tensor(FFTElasticProblem(img, MATS, tol=1e-11))
    b, _ = stiffness_tensor(FFTElasticProblem(rolled, MATS, tol=1e-11))
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)
    assert np.allclose(a, a.T, rtol=1e-8)
    res = apparent_bulk_modulus(FFTElasticProblem(img, MATS))
    assert res.energy == pytest.approx(res.value, rel=1e-7)
    kb = bounds(mg.analysis.phase_fractions(img), MATS)["bulk_modulus"]
    assert kb["hs_lower"] <= res.value <= kb["hs_upper"]


def test_porous_media():
    from scipy import ndimage

    rng = np.random.default_rng(5)
    noise = ndimage.gaussian_filter(rng.standard_normal((16, 16, 16)), 2.0, mode="wrap")
    img = mg.VoxelImage((noise > np.quantile(noise, 0.5)).astype(np.uint8), periodic=True)
    mats = {0: VOID, 1: STIFF}
    out = homogenize(img, mats, properties=["bulk_modulus", "shear_modulus", "conductivity"])
    b = bounds(mg.analysis.phase_fractions(img), mats)
    for key, res in out.items():
        prop = key.split(".")[0]
        assert 0.0 < res.value < b[prop]["hs_upper"], key
        assert res.energy == pytest.approx(res.value, rel=1e-6), key
        assert res.solution.material_fraction == pytest.approx(0.5, abs=0.01)
        assert res.solution.info.iterations < 400
    empty = mg.VoxelImage(np.zeros((4, 4, 4), np.uint8), periodic=True)
    assert homogenize(empty, mats)["bulk_modulus.periodic"].value == 0.0


def test_errors():
    img = random_image((3, 3, 3), 0.5, seed=0)
    with pytest.raises(ValueError, match="periodic"):
        FFTElasticProblem(img, MATS, "kubc")
    with pytest.raises(ValueError, match="loading is thermal"):
        FFTElasticProblem(img, MATS).solve(Loading.flux([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="loading is elastic"):
        FFTThermalProblem(img, MATS).solve(Loading.stress(np.eye(3)))
    with pytest.raises(ValueError, match="no material"):
        FFTElasticProblem(img, {1: STIFF})
    with pytest.raises(TypeError, match="VoxelImage"):
        FFTElasticProblem(np.zeros((3, 3, 3)), MATS)
    partial = mg.VoxelImage(np.full((3, 3, 3), 0.5), periodic=True)
    with pytest.raises(ValueError, match="integer"):
        FFTElasticProblem(partial, MATS)
