"""Stereology: Saltykov's matrix, the unfolding of exact sections of spheres and of turned
ellipsoids, the shape consistency of Degdeg & Dirrenberger, sections of voxel images (a
watershed, the frame of non-periodic images)."""

import numpy as np
import pytest

import morphogen as mg
from morphogen import analysis as an
from morphogen.distributions import Discrete
from morphogen.image import VoxelImage

#: Table 1 and Eq. (4) of Degdeg & Dirrenberger: mean section radii (um) normal to x, y, z
#: and the axes of their pores (columns a, b, c) in the frame of the scanner.
PAPER_MEANS = {0: 119.9, 1: 146.0, 2: 170.8}
PAPER_AXES = np.array(
    [[-0.0549, -0.0010, -0.9985], [-0.1578, -0.9874, 0.0097], [0.9859, -0.1581, -0.0541]]
)


def _rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _exact_sections(rng, sizes, shape, rotation, axis, planes, box):
    """Sections of aligned ellipsoids of short semi-axes ``sizes`` (uniform centres in a
    periodic box) by ``planes`` random planes normal to ``axis``."""
    centres = rng.uniform(0.0, box, size=(len(sizes), 3))
    l1, l2 = shape
    s = sizes**2 * (rotation @ np.diag([1.0, l1 * l1, l2 * l2]) @ rotation.T)[axis, axis]
    radii = []
    for z in rng.uniform(0.0, box, planes):
        d = (centres[:, axis] - z + box / 2) % box - box / 2
        cut = d**2 < s
        area = np.pi * sizes[cut] ** 3 * l1 * l2 * (s[cut] - d[cut] ** 2) / s[cut] ** 1.5
        radii.append(np.sqrt(area / np.pi))
    r = np.concatenate(radii)
    return an.Sections(axis, r, np.ones_like(r), planes * box * box)


def test_saltykov_matrix_and_section_scales():
    p = an.saltykov_matrix(20)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-14)
    i = np.arange(1, 21)
    np.testing.assert_allclose(np.diag(p), np.sqrt(2 * i - 1) / i)
    assert np.all(np.triu(p, 1) == 0.0)  # no section larger than its particle
    assert an.section_scales() == (1.0, 2.0)
    # an ellipsoid 1 : 1.3 : 2 along x, y, z: the largest sections bc, ac, ab and the heights
    for axis, kappa, height in (
        (0, np.sqrt(2.6), 2.0),
        (1, np.sqrt(2.0), 2.6),
        (2, np.sqrt(1.3), 4.0),
    ):
        np.testing.assert_allclose(an.section_scales((1.3, 2.0), axis=axis), (kappa, height))


def test_unfolding_of_exact_sections():
    rng = np.random.default_rng(1)
    box, n = 40.0, 40000
    sizes = rng.choice([0.6, 0.8, 1.0], size=n, p=[0.5, 0.3, 0.2])
    true = np.array([0.5, 0.3, 0.2]) * n / box**3
    spheres = _exact_sections(rng, sizes, (1.0, 1.0), np.eye(3), 2, 300, box)
    u = an.unfold_sections(spheres, percentile=100.0, classes=9)
    np.testing.assert_allclose(u.radii, np.linspace(0.6, 1.0, 9), rtol=1e-3)
    np.testing.assert_allclose(u.density[[0, 4, 8]], true, rtol=0.03)
    assert np.abs(u.density[[1, 2, 3, 5, 6, 7]]).max() < 0.02 * true.sum()
    assert u.errors[0] < 0.01
    with pytest.raises(ValueError, match="number of classes"):
        an.unfold_sections(spheres)
    # turned ellipsoids 1 : 1.3 : 2 seen in three directions: one population, one scale
    rotation, shape = _rotation(4), (1.3, 2.0)
    sections = [_exact_sections(rng, sizes, shape, rotation, a, 300, box) for a in range(3)]
    u = an.unfold_sections(sections, shape=shape, rotation=rotation, percentile=100.0, classes=9)
    np.testing.assert_allclose(u.scales, 1.0, rtol=2e-3)
    np.testing.assert_allclose(u.directional[:, [0, 4, 8]], np.tile(true, (3, 1)), rtol=0.04)
    m1, m2 = (
        np.average([0.6, 0.8, 1.0], weights=[0.5, 0.3, 0.2]),
        np.average([0.36, 0.64, 1.0], weights=[0.5, 0.3, 0.2]),
    )
    np.testing.assert_allclose(
        an.shape_consistency(sections, shape, rotation), np.pi / 4 * m2 / m1, rtol=0.01
    )
    wrong = an.shape_consistency(sections, (1.0, 1.0), rotation)
    assert wrong.max() / wrong.min() > 1.1
    d = u.distribution()
    assert isinstance(d, Discrete) and min(d.values) > 0.55 and max(d.values) < 1.05


def test_consistent_shape():
    rotation = _rotation(4)
    exact = {a: 0.7 * an.section_scales((1.3, 2.0), rotation, a)[0] for a in range(3)}
    np.testing.assert_allclose(an.consistent_shape(exact, rotation), (1.3, 2.0), rtol=1e-10)
    # the values of Degdeg & Dirrenberger: the ratio varies by 7 % (MIL shape) and 34 %
    # (covariance shape) between the directions, and is constant for about 1 : 1.40 : 2.07
    for shape, spread in (((1.34, 1.80), 0.07), ((1.22, 3.22), 0.34)):
        q = an.shape_consistency(PAPER_MEANS, shape, PAPER_AXES)
        assert q.max() / q.min() - 1.0 == pytest.approx(spread, abs=0.005)
    np.testing.assert_allclose(
        an.consistent_shape(PAPER_MEANS, PAPER_AXES), (1.40, 2.07), atol=0.006
    )
    assert an.consistent_shape({0: 1.0, 1: 1.0, 2: 1.0}) == pytest.approx((1.0, 1.0))


def test_sections_of_voxel_spheres():
    model = mg.models.HardcoreSpheres(
        radius={"type": "discrete", "values": [6.0, 8.0, 10.0], "weights": [0.5, 0.3, 0.2]},
        volume_fraction=0.15,
        min_gap=2.0,
    )
    real = model.generate(mg.Domain.cube(160.0, periodic=True), seed=3)
    img = real.voxelize((160, 160, 160))
    radii = np.asarray(real.spheres.radii)
    sections = [an.section_sizes(img, 1, a) for a in range(3)]
    expected = 2.0 * radii.sum() / 160.0**3  # sections per unit area: 2 r N_V
    for s in sections:
        assert s.weights.sum() / s.area == pytest.approx(expected, rel=0.01)
    target = np.pi / 4 * (radii**2).mean() / radii.mean()
    np.testing.assert_allclose(an.shape_consistency(sections), target, rtol=0.01)
    u = an.unfold_sections(sections, classes=9)
    np.testing.assert_allclose(u.scales, 10.0, rtol=0.01)
    assert u.density.sum() == pytest.approx(len(radii) / 160.0**3, rel=0.05)


def test_watershed_and_frame():
    yy, xx = np.mgrid[0:64, 0:64]
    two = ((xx - 24) ** 2 + (yy - 32) ** 2 <= 100) | ((xx - 40) ** 2 + (yy - 32) ** 2 <= 100)
    stack = VoxelImage(np.repeat(two[:, :, None], 3, axis=2).astype(np.uint8), periodic=True)
    assert len(an.section_sizes(stack, 1, 2).radii) == 3  # one section per slice
    split = an.section_sizes(stack, 1, 2, separate=True)
    assert len(split.radii) == 6
    np.testing.assert_allclose(split.radii, np.sqrt(two.sum() / 2 / np.pi), rtol=0.03)
    framed = VoxelImage(stack.data, periodic=False)  # inside the frame: weights near 1
    split = an.section_sizes(framed, 1, 2, separate=True)
    np.testing.assert_allclose(split.radii, np.sqrt(two.sum() / 2 / np.pi), rtol=0.03)
    # halves of 21 x 18-19 pixels: weights 64^2 / ((63 - 21)(63 - 18.5))
    np.testing.assert_allclose(split.weights, 64 * 64 / (42 * 44.5), rtol=0.02)
    # discs of radius 5 on a lattice of period 16: windows of 100 pixels at every offset,
    # the discs cut by the frame dropped, the others weighted: 1 / 256 on average, exactly
    big = np.zeros((16 * 9, 16 * 9), np.uint8)
    yy, xx = np.mgrid[0:16, 0:16]
    big[:] = np.tile(((xx - 8) ** 2 + (yy - 8) ** 2 <= 25).astype(np.uint8), (9, 9))
    densities = []
    for t1 in range(16):
        for t2 in range(16):
            window = VoxelImage(big[t1 : t1 + 100, t2 : t2 + 100, None].copy(), periodic=False)
            s = an.section_sizes(window, 1, 2)
            densities.append(s.weights.sum() / s.area)
    assert np.mean(densities) == pytest.approx(1.0 / 256.0, rel=1e-12)
