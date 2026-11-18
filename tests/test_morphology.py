"""Chord lengths, granulometry and local thickness, Euler characteristic and geometric
tortuosity."""

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.analysis import (
    MEASURES,
    chord_lengths,
    euler_characteristic,
    geometric_tortuosity,
    granulometry,
    mean_chord_length,
    measure,
    minkowski,
)


def ball_image(n, centres, radius, periodic=False):
    g = np.arange(n) + 0.5
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    out = np.zeros((n, n, n), dtype=bool)
    for c in centres:
        out |= (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 <= radius**2
    return mg.VoxelImage(out.astype(np.uint8), periodic=periodic)


def test_chords_of_the_boolean_model():
    """Pore chords of a Boolean model of spheres: exponential, of mean 1 / (lambda pi
    r^2); on a periodic image, mean chords of the phases are in the ratio p / (1 - p)."""
    r, p = 0.08, 0.3
    micro = mg.models.BooleanSpheres(radius=r, volume_fraction=p).generate(
        mg.Domain.cube(1.0, periodic=True), seed=1
    )
    img = micro.voxelize((128, 128, 128))
    lam = -math.log(1 - p) / (4 / 3 * math.pi * r**3)
    l0, l1 = mean_chord_length(img, 0), mean_chord_length(img, 1)
    assert l0 == pytest.approx(1 / (lam * math.pi * r * r), rel=0.05)
    # on every line crossing both phases, the runs alternate: as many chords of each
    v0 = v1 = 0
    for axis in range(3):
        lines = np.moveaxis(img.data == 1, axis, -1).reshape(-1, img.shape[axis])
        mixed = lines.any(axis=1) & ~lines.all(axis=1)
        v1 += int(lines[mixed].sum())
        v0 += int((~lines[mixed]).sum())
    assert l1 / l0 == pytest.approx(v1 / v0, rel=1e-12)
    # a slab: chords across it only
    slab = np.zeros((8, 9, 10), dtype=np.uint8)
    slab[:, :, 3:7] = 1
    image = mg.VoxelImage(slab, spacing=(1.0, 1.0, 0.5))
    assert np.all(chord_lengths(image, 1, 2) == 2.0) and len(chord_lengths(image, 1, 2)) == 72
    assert chord_lengths(image, 1, 0).size == 0  # cut by the faces
    periodic = mg.VoxelImage(slab, spacing=(1.0, 1.0, 0.5), periodic=True)
    assert np.all(chord_lengths(periodic, 0, 2) == 3.0)  # the pores wrap around


def test_euler_characteristic():
    n = 30
    x, y, z = np.mgrid[:n, :n, :n] - (n - 1) / 2
    shapes = {
        1: x**2 + y**2 + z**2 < 64,
        0: (np.abs(np.sqrt(x**2 + y**2) - 9) < 3) & (np.abs(z) < 3),  # a ring
        2: (x**2 + y**2 + z**2 < 144) & ~(x**2 + y**2 + z**2 < 36),  # a hollow ball
        # a ring with a bar across: two tunnels
        -1: (np.abs(np.sqrt(x**2 + y**2) - 9) < 3) & (np.abs(z) < 3)
        | (np.abs(x) < 2) & (np.abs(z) < 3) & (np.abs(y) < 10),
    }
    for chi, shape in shapes.items():
        assert euler_characteristic(mg.VoxelImage(shape.astype(np.uint8)), 1) == chi
    # on the torus: a slab, a rod and the whole box have characteristic 0
    slab = np.zeros((12, 12, 12), dtype=np.uint8)
    slab[:, :, 3:6] = 1
    rod = np.zeros_like(slab)
    rod[:, 3:6, 3:6] = 1
    for shape in (slab, rod, np.ones_like(slab)):
        assert euler_characteristic(mg.VoxelImage(shape, periodic=True), 1) == 0
    # balls at least two voxels apart: one per ball
    rng = np.random.default_rng(3)
    centres = []
    while len(centres) < 12:
        c = rng.uniform(5, 43, 3)
        if all(np.linalg.norm(c - d) > 2 * 4 + 3 for d in centres):
            centres.append(c)
    img = ball_image(48, centres, 4.0)
    assert euler_characteristic(img, 1) == 12
    mk = minkowski(img, 1)
    assert mk.euler_density == pytest.approx(12 / 48**3)


def test_granulometry_and_local_thickness():
    for t in (4, 10):
        slab = np.zeros((20, 20, 24), dtype=np.uint8)
        slab[:, :, 5 : 5 + t] = 1
        g = granulometry(mg.VoxelImage(slab, periodic=True), 1)
        assert np.allclose(g.thickness[slab == 1], t)
        assert np.all(np.diff(g.fractions) <= 0)
    for radius in (6, 10):
        g = granulometry(ball_image(32, [(16, 16, 16)], radius), 1)
        assert radius - 1.6 < g.mean_radius <= radius
        assert g.size_distribution.sum() == pytest.approx(1.0)
    pores = granulometry(mg.VoxelImage(np.zeros((6, 6, 6), dtype=np.uint8)), 1)
    assert np.isnan(pores.mean_radius)


def test_geometric_tortuosity():
    layer = np.zeros((20, 12, 12), dtype=np.uint8)
    layer[:, :, 4:8] = 1
    image = mg.VoxelImage(layer, spacing=(0.5, 1.0, 1.0))
    assert geometric_tortuosity(image, 1, 0) == pytest.approx(1.0)
    assert math.isinf(geometric_tortuosity(image, 1, 2))
    # a diagonal line of voxels: steps of sqrt(2) voxels
    n = 15
    line = np.zeros((n, n, 3), dtype=np.uint8)
    line[np.arange(n), np.arange(n), 1] = 1
    expected = ((n - 1) * math.sqrt(2) + 1) / n
    assert geometric_tortuosity(mg.VoxelImage(line), 1, 0) == pytest.approx(expected)


def test_measures_registry():
    micro = mg.models.BooleanSpheres(radius=0.1, volume_fraction=0.3).generate(
        mg.Domain.cube(1.0, periodic=True), seed=2
    )
    img = micro.voxelize((40, 40, 40))
    values = measure(img, ["chords", "thickness", "euler", "tortuosity"])
    for key in (
        "mean_chord_0",
        "mean_chord_1",
        "mean_thickness",
        "mean_pore_size",
        "euler_density_1",
        "tortuosity_x",
    ):
        assert key in values
    assert {"chords", "thickness", "euler", "tortuosity"} <= set(MEASURES)
    assert values["mean_pore_size"] > values["mean_thickness"] > 0
