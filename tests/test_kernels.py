"""Rasterization kernels against brute-force references."""

import numpy as np
import pytest

import morphogen as mg
from morphogen.kernels.rsa import random_sequential_addition
from morphogen.kernels.union import UnionAccelerator
from morphogen.primitives import Segments, Spheres, with_periodic_images


def _random_primitives(rng, round_caps):
    spheres = Spheres(rng.uniform(0, 10, (40, 3)), rng.uniform(0.3, 1.5, 40))
    seg = Segments(
        rng.uniform(-1, 11, (15, 3)),
        rng.uniform(-1, 11, (15, 3)),
        rng.uniform(0.2, 0.7, 15),
        round_caps=round_caps,
    )
    return spheres, seg


def _brute_force_distance(points, spheres, seg):
    d_s = np.linalg.norm(points[:, None] - spheres.centers[None], axis=2) - spheres.radii[None]
    ab = seg.end - seg.start
    length = np.linalg.norm(ab, axis=1)
    u = np.einsum("pnk,nk->pn", points[:, None] - seg.start[None], ab) / (length**2)[None]
    if seg.round_caps:
        uc = np.clip(u, 0, 1)
        foot = seg.start[None] + uc[..., None] * ab[None]
        d_g = np.linalg.norm(points[:, None] - foot, axis=2) - seg.radii[None]
    else:
        foot = seg.start[None] + u[..., None] * ab[None]
        radial = np.linalg.norm(points[:, None] - foot, axis=2) - seg.radii[None]
        axial = (np.abs(u - 0.5) - 0.5) * length[None]
        d_g = np.hypot(np.maximum(radial, 0), np.maximum(axial, 0)) + np.minimum(
            np.maximum(radial, axial), 0
        )
    return np.minimum(d_s.min(axis=1), d_g.min(axis=1))


@pytest.mark.parametrize("round_caps", [True, False])
def test_rasterization_matches_brute_force(rng, round_caps):
    spheres, seg = _random_primitives(rng, round_caps)
    lo, hi = np.zeros(3), np.full(3, 10.0)
    shape = (30, 27, 25)
    spacing = (hi - lo) / np.array(shape)
    grid = mg.Domain.cube(10.0, periodic=False).grid(shape)
    ref = _brute_force_distance(grid.center_points(), spheres, seg).reshape(shape)

    acc = UnionAccelerator(spheres, seg, lo, hi, band=0.6)
    counts = acc.count(lo, spacing, shape)
    assert np.array_equal(counts == 1, ref <= 0)
    dist = acc.distance(lo, spacing, shape)
    np.testing.assert_allclose(dist, np.clip(ref, -0.6, 0.6), atol=1e-5)

    pts = rng.uniform(0, 10, (2000, 3))
    ref_pts = _brute_force_distance(pts, spheres, seg)
    assert np.array_equal(acc.inside(pts), ref_pts <= 0)
    np.testing.assert_allclose(acc.signed_distance(pts), np.clip(ref_pts, -0.6, 0.6), atol=1e-9)


def test_supersampling_counts_sub_voxels(rng):
    spheres = Spheres(rng.uniform(0, 4, (5, 3)), rng.uniform(0.5, 1.0, 5))
    lo, hi = np.zeros(3), np.full(3, 4.0)
    acc = UnionAccelerator(spheres, None, lo, hi)
    coarse = acc.count(lo, np.full(3, 0.4), (10, 10, 10), supersampling=3)
    fine = acc.count(lo, np.full(3, 0.4 / 3), (30, 30, 30), supersampling=1)
    assert np.array_equal(coarse, fine.reshape(10, 3, 10, 3, 10, 3).sum(axis=(1, 3, 5)))
    with pytest.raises(ValueError):
        acc.count(lo, np.full(3, 0.4), (10, 10, 10), supersampling=7)


def test_periodic_images_reproduce_the_torus(rng):
    d = mg.Domain.cube(5.0)
    spheres = Spheres(rng.uniform(0, 5, (12, 3)), rng.uniform(0.5, 1.2, 12))
    seg = Segments(
        rng.uniform(0, 5, (6, 3)), rng.uniform(0, 5, (6, 3)) * 0.3 + 2.0, np.full(6, 0.4)
    )
    img_s = with_periodic_images(spheres, d)
    img_g = with_periodic_images(seg, d)
    acc = UnionAccelerator(img_s, img_g, d.lower, d.upper)
    pts = rng.uniform(0, 5, (3000, 3))
    # brute force with the minimum-image convention for spheres and 27 copies for segments
    diff = pts[:, None] - spheres.centers[None]
    diff -= 5.0 * np.round(diff / 5.0)
    inside = np.any(np.linalg.norm(diff, axis=2) <= spheres.radii[None], axis=1)
    for shift in np.array(np.meshgrid([-5, 0, 5], [-5, 0, 5], [-5, 0, 5])).reshape(3, -1).T:
        s = seg.translated(shift)
        ab = s.end - s.start
        u = np.clip(
            np.einsum("pnk,nk->pn", pts[:, None] - s.start[None], ab)
            / np.sum(ab * ab, axis=1)[None],
            0,
            1,
        )
        foot = s.start[None] + u[..., None] * ab[None]
        inside |= np.any(np.linalg.norm(pts[:, None] - foot, axis=2) <= s.radii[None], axis=1)
    assert np.array_equal(acc.inside(pts), inside)


def test_rsa_kernel_places_non_overlapping_spheres(rng):
    radii = rng.uniform(0.3, 0.6, 300)
    pos = random_sequential_addition(radii, np.full(3, 10.0), 0.05, rng, 10**6)
    diff = pos[:, None] - pos[None]
    diff -= 10.0 * np.round(diff / 10.0)
    dist = np.linalg.norm(diff, axis=2) - radii[:, None] - radii[None]
    np.fill_diagonal(dist, np.inf)
    assert dist.min() >= 0.05 - 1e-12
    assert np.all((pos >= 0) & (pos < 10))
    with pytest.raises(RuntimeError, match="jammed"):
        random_sequential_addition(np.full(500, 1.0), np.full(3, 8.0), 0.0, rng, 20_000)
