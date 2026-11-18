"""Domains, grids, seeds, distributions, orientations and images."""

import numpy as np
import pytest
from scipy import stats

import morphogen as mg
from morphogen import distributions as dist
from morphogen import orientations as ori
from morphogen.seeding import derive_seed, make_rng, seed_from_description

# -- domains and grids -------------------------------------------------------------------


def test_domain_basics():
    d = mg.Domain(size=(2.0, 3.0, 4.0), origin=(1.0, 0.0, -1.0), periodic=False, unit="um")
    assert d.volume == pytest.approx(24.0)
    assert d.surface_area == pytest.approx(2 * (6 + 12 + 8))
    np.testing.assert_allclose(d.upper, [3.0, 3.0, 3.0])
    np.testing.assert_allclose(d.center, [2.0, 1.5, 1.0])
    big = d.dilated(0.5)
    assert not big.periodic and big.size == (3.0, 4.0, 5.0) and big.origin == (0.5, -0.5, -1.5)
    assert mg.Domain.from_dict(d.to_dict()) == d


@pytest.mark.parametrize("bad", [0.0, -1.0, (1.0, 2.0), "abc", np.inf])
def test_domain_rejects_bad_sizes(bad):
    with pytest.raises(ValueError):
        mg.Domain(size=bad)


def test_domain_wrap_and_distance():
    d = mg.Domain.cube(2.0, origin=(1.0, 1.0, 1.0))
    w = d.wrap([[0.5, 3.5, 1.0], [-1e-17 + 1.0, 2.0, 2.999]])
    assert np.all(w >= d.lower) and np.all(w < d.upper)
    np.testing.assert_allclose(w[0], [2.5, 1.5, 1.0])
    dist_ = d.distance_to_box([[2.0, 2.0, 2.0], [4.0, 2.0, 2.0], [4.0, 4.0, 2.0]])
    np.testing.assert_allclose(dist_, [0.0, 1.0, np.sqrt(2.0)])
    assert d.contains([[1.0, 1.0, 1.0], [3.0, 2.0, 2.0]]).tolist() == [True, False]


def test_grid_from_shape_and_voxel_size():
    d = mg.Domain(size=(4.0, 2.0, 1.0))
    g = d.grid(8)  # 8 voxels along the longest edge, cubic voxels
    assert g.shape == (8, 4, 2)
    np.testing.assert_allclose(g.spacing, 0.5)
    g = d.grid(voxel_size=0.3)  # rounded, spacing adjusted to fit the box exactly
    assert g.shape == (13, 7, 3)
    np.testing.assert_allclose(g.spacing * g.shape, d.size)
    g = d.grid((3, 4, 5))
    assert g.n_voxels == 60
    np.testing.assert_allclose(g.centers(2), (np.arange(5) + 0.5) * 0.2)
    with pytest.raises(ValueError):
        d.grid()
    with pytest.raises(ValueError):
        d.grid(4, voxel_size=0.1)


def test_grid_center_points_order_matches_arrays():
    g = mg.Domain(size=(3.0, 2.0, 4.0)).grid((3, 2, 4))
    pts = g.center_points(1, 3).reshape(3, 2, 2, 3)
    np.testing.assert_allclose(pts[2, 1, 0], [2.5, 1.5, 1.5])
    np.testing.assert_allclose(pts[0, 0, 1], [0.5, 0.5, 2.5])


# -- seeds ---------------------------------------------------------------------------------


def test_seeds_are_reproducible_and_independent():
    r1, info = make_rng(42)
    r2, _ = make_rng(seed_from_description(info))
    assert np.array_equal(r1.random(5), r2.random(5))
    a = np.random.default_rng(derive_seed(7, 0, 1)).random(3)
    b = np.random.default_rng(derive_seed(7, 0, 1)).random(3)
    c = np.random.default_rng(derive_seed(7, 1, 0)).random(3)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    _, fresh = make_rng(None)  # fresh entropy is recorded
    assert fresh["entropy"] > 0
    with pytest.raises(ValueError):
        make_rng(-1)


# -- distributions ---------------------------------------------------------------------------

DISTRIBUTIONS = [
    dist.Constant(1.5),
    dist.Uniform(0.5, 1.5),
    dist.LogNormal(1.0, 0.3),
    dist.TruncatedNormal(20.0, 13.5),
    dist.Discrete((1.0, 2.0, 4.0), (0.5, 0.3, 0.2)),
]


@pytest.mark.parametrize("d", DISTRIBUTIONS, ids=lambda d: d.kind)
def test_distribution_moments_match_samples(d, rng):
    x = d.sample(rng, 200_000)
    assert np.all(x > 0) and x.max() <= d.upper()
    for k in (1, 2, 3):
        m = d.moment(k)
        assert np.mean(x**k) == pytest.approx(m, rel=0.02)
        nodes, weights = d.quadrature(256)
        assert np.sum(weights * nodes**k) == pytest.approx(m, rel=0.01)
    assert dist.distribution_from_config(d.to_config()) == d


def test_distribution_config_parsing():
    assert dist.distribution_from_config(2) == dist.Constant(2.0)
    u = dist.distribution_from_config({"type": "uniform", "low": 1, "high": 2})
    assert u.mean() == pytest.approx(1.5)
    ln = dist.distribution_from_config({"type": "lognormal", "mean": 2.0, "std": 0.5})
    assert ln.mean() == pytest.approx(2.0) and ln.std() == pytest.approx(0.5)
    for bad in [
        {"type": "gamma"},
        {"type": "uniform", "low": 1},
        {"low": 1},
        True,
        "x",
        {"type": "uniform", "low": 2, "high": 1},
        {"type": "constant", "value": -1},
    ]:
        with pytest.raises(ValueError):
            dist.distribution_from_config(bad)


# -- orientations ------------------------------------------------------------------------------


def test_orientation_distributions(rng):
    v = ori.Isotropic().sample(rng, 50_000)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0)
    # uniform on the sphere: each coordinate is uniform on [-1, 1]
    assert stats.kstest(v[:, 2], "uniform", args=(-1, 2)).pvalue > 1e-3
    a = ori.Aligned((0, 2, 0)).sample(rng, 3)
    np.testing.assert_allclose(a, [[0, 1, 0]] * 3)
    p = ori.Planar((1, 1, 0)).sample(rng, 1000)
    np.testing.assert_allclose(p @ np.array([1, 1, 0]) / np.sqrt(2), 0.0, atol=1e-12)
    for cfg in ["isotropic", {"type": "aligned", "direction": [1, 0, 0]}, {"type": "planar"}]:
        o = ori.orientation_from_config(cfg)
        assert ori.orientation_from_config(o.to_config()) == o
    with pytest.raises(ValueError):
        ori.orientation_from_config({"type": "aligned", "direction": [0, 0, 0]})


def test_orthonormal_basis(rng):
    d = ori.uniform_unit_vectors(rng, 100)
    e1, e2 = ori.orthonormal_basis(d)
    for a, b in ((d, e1), (d, e2), (e1, e2)):
        np.testing.assert_allclose(np.einsum("ij,ij->i", a, b), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.einsum("ij,ij->i", np.cross(d, e1), e2), 1.0)


def test_random_orientations_are_uniform_and_invertible(rng):
    e = ori.random_orientations(rng, 20_000)
    assert stats.kstest(np.cos(np.radians(e[:, 1])), "uniform", args=(-1, 2)).pvalue > 1e-3
    g = ori.euler_to_matrix(e)
    np.testing.assert_allclose(
        g @ np.swapaxes(g, 1, 2), np.broadcast_to(np.eye(3), g.shape), atol=1e-12
    )
    np.testing.assert_allclose(ori.euler_to_matrix(ori.matrix_to_euler(g)), g, atol=1e-9)


# -- images ------------------------------------------------------------------------------------


def test_voxel_image_geometry():
    img = mg.VoxelImage(np.zeros((4, 5, 6), np.uint8), spacing=0.5, origin=(1, 2, 3))
    assert img.shape == (4, 5, 6) and img.volume == pytest.approx(2 * 2.5 * 3)
    sub = img.crop((1, 1, 1), (3, 4, 5))
    assert sub.shape == (2, 3, 4) and sub.origin == (1.5, 2.5, 3.5) and not sub.periodic
    assert img.grid().shape == img.shape
    with pytest.raises(ValueError):
        mg.VoxelImage(np.zeros((3, 3)))
