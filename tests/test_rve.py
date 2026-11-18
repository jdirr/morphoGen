"""Statistical RVE size: formulas, fits, published values and the window method."""

import itertools
import math

import numpy as np
import pytest

import morphogen as mg
from morphogen import rve


def test_point_variances():
    assert rve.point_variance(0.16) == pytest.approx(0.16 * 0.84)
    assert rve.point_variance(0.3, 10.0, 2.0) == pytest.approx(0.21 * 64)
    assert rve.point_variance_multiphase([0.3, 0.7], [1.0, 0.0]) == pytest.approx(0.21)
    assert rve.point_variance_multiphase([0.2, 0.3, 0.5], [1, 2, 3]) == pytest.approx(
        0.2 * 1.3**2 + 0.3 * 0.3**2 + 0.5 * 0.7**2
    )


def test_ensemble_statistics_and_relative_error():
    volumes = [1, 1, 1, 8, 8, 8, 8]
    values = [1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 4.0]
    st = rve.ensemble_statistics(volumes, values)
    assert st.volume.tolist() == [1, 8] and st.n.tolist() == [3, 4]
    np.testing.assert_allclose(st.mean, [2.0, 2.5])
    np.testing.assert_allclose(st.variance, [1.0, 1.0])
    np.testing.assert_allclose(st.relative_error(), [2 * math.sqrt(1 / 3) / 2, 2 * 0.5 / 2.5])
    assert len(st.rows()) == 2


def test_fit_recovers_power_laws():
    v = np.array([1e2, 1e3, 1e4, 1e5])
    law = rve.fit_scaling_law(v, 0.3 * v**-0.7)
    assert law.K == pytest.approx(0.3) and law.gamma == pytest.approx(0.7)
    assert law.r2 == pytest.approx(1.0) and not law.gamma_fixed
    fixed = rve.fit_scaling_law(v, 2.0 / v, gamma=1.0)
    assert fixed.K == pytest.approx(2.0) and fixed.gamma_fixed
    assert fixed.integral_range(0.25) == pytest.approx(8.0)  # A3 = K / D^2
    sub = rve.fit_scaling_law(
        np.append(v, 1e6), np.append(0.3 * v**-0.7, 1.0), volume_range=(None, 1e5)
    )
    assert sub.n_points == 4 and sub.gamma == pytest.approx(0.7)
    with pytest.raises(ValueError):
        rve.fit_scaling_law([1.0], [0.1])


def test_rve_formulas_are_consistent():
    law = rve.ScalingLaw(K=0.5, gamma=0.8)
    v = law.rve_volume(0.05, mean=0.2, n=3)
    assert law.relative_error(v, 3, 0.2) == pytest.approx(0.05)
    assert law.realizations(v, 0.05, 0.2) == pytest.approx(3.0)
    # gamma = 1 reduces to Kanit et al. (2003): V = 4 D^2 A3 / (eps^2 Z^2 n)
    classic = rve.rve_volume_classic(0.21, 4.0, 0.05, 0.3, 2)
    assert rve.ScalingLaw(K=0.21 * 4.0, gamma=1.0).rve_volume(0.05, 0.3, 2) == pytest.approx(
        classic
    )
    rows = rve.rve_table(law, 0.2, eps_rel=(0.01, 0.05), n=(1, 10))
    assert len(rows) == 4 and rows[0]["edge"] == pytest.approx(rows[0]["volume"] ** (1 / 3))


def test_published_rve_sizes():
    # Dirrenberger et al. (2014), Table 10 (gamma = 2/3, eps = 5 %, n = 1): V_V -> 164^3,
    # thermal conductivity (UTG) -> 204^3 (K given with two significant digits)
    fibres = rve.ScalingLaw(K=0.44, gamma=2 / 3)
    assert fibres.rve_volume(0.05, 0.16) ** (1 / 3) == pytest.approx(164, rel=0.02)
    utg = rve.ScalingLaw(K=1.3e3, gamma=2 / 3)
    assert utg.rve_volume(0.05, 7.07) ** (1 / 3) == pytest.approx(204, rel=0.01)
    # Yang et al. (2019), Tables 4-5 (volumes in numbers of grains, n = 1)
    mu = rve.ScalingLaw(K=2.82e8, gamma=1.07)
    np.testing.assert_allclose(
        mu.rve_volume([0.01, 0.02, 0.05, 0.10], 49031.18), [2756, 753, 135, 37], rtol=0.03
    )
    k = rve.ScalingLaw(K=2.29e9, gamma=1.17)
    np.testing.assert_allclose(
        k.rve_volume([0.01, 0.02, 0.05, 0.10], 130972.25), [1504, 461, 96, 29], rtol=0.04
    )


def test_window_statistics_on_exact_patterns():
    # checkerboard of 2^3 blocks: windows of 2 voxels aligned on blocks are pure
    i, j, k = np.indices((8, 8, 8))
    board = (((i // 2) + (j // 2) + (k // 2)) % 2).astype(np.uint8)
    st = rve.window_statistics(mg.VoxelImage(board, spacing=0.5), [2, 4, 8])
    np.testing.assert_allclose(st.volume, [1.0, 8.0, 64.0])
    np.testing.assert_allclose(st.mean, 0.5)
    assert st.variance[0] == pytest.approx(0.25 * 64 / 63)
    assert st.variance[1] == pytest.approx(0.0) and np.isnan(st.variance[2])
    # scalar fields: phase=None averages the values themselves
    field = np.arange(64, dtype=float).reshape(4, 4, 4)
    st = rve.window_statistics(field, [2], phase=None)
    assert st.mean[0] == pytest.approx(field.mean())
    with pytest.raises(ValueError):
        rve.window_statistics(field, [5])


def _one_tiling(data, w):
    """The count, mean and variance of the windows of ``w`` voxels tiling an image from its
    origin, computed as before the shifted tilings (the indicator of phase 1)."""
    nx, ny, nz = (s // w for s in data.shape)
    block = (data == 1)[: nx * w, : ny * w, : nz * w].reshape(nx, w, ny, w, nz, w)
    means = block.mean(axis=(1, 3, 5), dtype=np.float64).ravel()
    return means.size, means.mean(), means.var(ddof=1)


def test_shifted_tilings_of_a_periodic_image():
    """Tilings shifted around a periodic image: each a translate of the first, the mean of
    its windows the image mean when its edges are multiples of theirs, and so that of all
    the windows of all the tilings, exactly (dyadic means); the variance pooled within the
    T tilings, the mean of their unbiased variances, that of all the windows about their
    mean (T n - 1 degrees of freedom) lower by T (n - 1) / (T n - 1); n the windows of one
    tiling. One tiling (the default) gives the statistics of before, to the bit; squares
    across a planar image, their edges the roots of their areas."""
    data = (np.random.default_rng(7).random((16, 16, 16)) < 0.3).astype(np.uint8)
    image = mg.VoxelImage(data, spacing=0.5, periodic=True)
    st = rve.window_statistics(image, [1, 2, 4, 8], shifts=2)
    np.testing.assert_array_equal(st.tilings, [1, 8, 8, 8])  # one voxel: a single tiling
    np.testing.assert_array_equal(st.n, [4096, 512, 64, 8])
    assert (st.mean == data.mean()).all() and st.dimension == 3
    assert list(st.rows()[1]) == ["volume", "n", "tilings", "mean", "variance", "relative_error"]
    variances = []
    for offset in itertools.product((0, 2), repeat=3):  # the tilings of the windows of 4
        shifted = np.roll(data, [-o for o in offset], axis=(0, 1, 2)).astype(float)
        means = shifted.reshape(4, 4, 4, 4, 4, 4).mean(axis=(1, 3, 5)).ravel()
        assert means.mean() == data.mean()
        variances.append(means.var(ddof=1))
    assert st.variance[2] == pytest.approx(np.mean(variances), rel=1e-12)
    every = np.concatenate([np.roll(data, [-o for o in offset], axis=(0, 1, 2))
                            .reshape(4, 4, 4, 4, 4, 4).mean(axis=(1, 3, 5)).ravel()
                            for offset in itertools.product((0, 2), repeat=3)])  # fmt: skip
    assert every.var(ddof=1) == pytest.approx(st.variance[2] * 8 * 63 / (8 * 64 - 1), rel=1e-12)
    # a field, the remainder of windows that do not divide the image, distinct offsets only
    field = np.random.default_rng(8).random((16, 16, 16))
    st = rve.window_statistics(field, [2, 4], phase=None, shifts=3, periodic=True)
    np.testing.assert_array_equal(st.tilings, [8, 27])  # 2 // 3 = 0, 1; 4 // 3 = 0, 1, 2
    np.testing.assert_allclose(st.mean, field.mean(), rtol=1e-14)
    st = rve.window_statistics(image, [3, 5], shifts=4)
    np.testing.assert_array_equal(st.tilings, [27, 64])
    np.testing.assert_array_equal(st.n, [125, 27])
    # squares across a planar image: their areas, and edges
    planar = mg.VoxelImage(data[:, :, :1], spacing=0.5, periodic=True)
    st = rve.window_statistics(planar, [2, 4], shifts=2)
    assert st.dimension == 2
    np.testing.assert_array_equal(st.tilings, [4, 4])
    np.testing.assert_array_equal(st.n, [64, 16])
    np.testing.assert_allclose(st.volume, [1.0, 4.0], rtol=1e-15)
    np.testing.assert_allclose(st.edges(), [1.0, 2.0], rtol=1e-15)
    # one tiling: the statistics of before, to the bit, with or without shifts=1
    sizes = [2, 3, 4, 5]
    default = rve.window_statistics(image, sizes)
    explicit = rve.window_statistics(image, sizes, shifts=1)
    assert default.tilings is None and explicit.tilings is None
    for k, w in enumerate(sizes):
        assert (default.n[k], default.mean[k], default.variance[k]) == _one_tiling(data, w)
    for name in ("volume", "n", "mean", "variance"):
        np.testing.assert_array_equal(getattr(default, name), getattr(explicit, name))
    assert list(default.rows()[0]) == ["volume", "n", "mean", "variance", "relative_error"]
    np.testing.assert_allclose(default.edges(), 0.5 * np.array(sizes), rtol=1e-15)
    for image, message in (
        (mg.VoxelImage(data), "periodic"),  # not periodic
        (data, "periodic"),  # an array: periodic=True
    ):
        with pytest.raises(ValueError, match=message):
            rve.window_statistics(image, [2], shifts=2)
    with pytest.raises(ValueError, match="shifts"):
        rve.window_statistics(planar, [2], shifts=0)


def test_shifted_tilings_estimate_the_variance():
    """On random images of independent voxels (D^2(V) = p (1 - p) / w^3 exactly, the windows
    of a tiling uncorrelated: the variance of one tiling unbiased), the variance pooled
    within shifted tilings agrees with that of one tiling within its error (relative
    sqrt(2 / (n - 1))), with D^2(V) on average, and scatters less from image to image."""
    p, w = 0.3, 4
    exact = p * (1.0 - p) / w**3
    one, shifted = [], []
    for seed in range(16):
        data = (np.random.default_rng(seed).random((32, 32, 32)) < p).astype(np.uint8)
        image = mg.VoxelImage(data, periodic=True)
        a = rve.window_statistics(image, [w])
        b = rve.window_statistics(image, [w], shifts=4)
        assert a.n[0] == b.n[0] == 512 and b.tilings[0] == 64
        error = a.variance[0] * math.sqrt(2.0 / (a.n[0] - 1))
        assert abs(b.variance[0] - a.variance[0]) < 3.0 * error
        one.append(a.variance[0])
        shifted.append(b.variance[0])
    spread = exact * math.sqrt(2.0 / 511) / math.sqrt(len(one))
    assert abs(np.mean(shifted) - exact) < 3.0 * spread
    assert abs(np.mean(one) - exact) < 3.0 * spread
    assert np.std(shifted) < 0.8 * np.std(one)


def _window_variance_theory(model, a):
    """Variance of the volume fraction over a cube of edge a (exact, from C(h))."""
    p = model.theory()["volume_fraction"]
    hmax = 2.0 * model.radius.upper()
    n = 48
    h = (np.arange(n) + 0.5) * hmax / n
    hx, hy, hz = np.meshgrid(h, h, h, indexing="ij")
    r = np.sqrt(hx**2 + hy**2 + hz**2)
    weight = (
        np.clip(1 - hx / a, 0, None) * np.clip(1 - hy / a, 0, None) * np.clip(1 - hz / a, 0, None)
    )
    integrand = (model.covariance(r) - p * p) * weight
    return 8.0 * integrand.sum() * (hmax / n) ** 3 / a**3


def test_window_method_matches_boolean_theory():
    model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
    edges = [16, 32]  # 2 and 4 length units with voxels of 1/8
    variances = []
    for seed in range(3):
        img = model.generate(mg.Domain.cube(16.0), seed=seed).voxelize(shape=128)
        variances.append(rve.window_statistics(img, edges).variance)
    measured = np.mean(variances, axis=0)
    for a, v in zip((2.0, 4.0), measured, strict=True):
        expected = _window_variance_theory(model, a)
        n_windows = 3 * (16 / a) ** 3
        assert v == pytest.approx(expected, rel=4 * math.sqrt(2 / n_windows))
    # asymptotically D^2(V) = D^2 A3 / V
    big = _window_variance_theory(model, 40.0)
    assert big == pytest.approx(0.21 * model.integral_range() / 40.0**3, rel=0.1)
