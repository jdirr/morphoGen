"""The particles of a phase in a volume: two overlapping balls cut by the plane of their
neck (exact volumes of their caps), through the faces of a periodic image; a packing of
hard spheres recovered sphere by sphere; the frame of an image that is not periodic
(Miles and Lantuéjoul); discs in a planar image as the sections of the stereology; the
named measure."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

import morphogen as mg
from morphogen import analysis as an
from morphogen.analysis.particles import Particles, particle_sizes
from morphogen.kernels.watershed import flood_volume, prominent_maxima_volume


def balls(n, centres, radii, periodic=True):
    x = (np.arange(n) + 0.5) / n
    grid = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1)
    inside = np.zeros((n, n, n), bool)
    for centre, radius in zip(centres, radii, strict=True):
        d = grid - np.asarray(centre, dtype=float)
        if periodic:
            d -= np.rint(d)
        inside |= (d**2).sum(axis=-1) < radius**2
    return mg.VoxelImage(inside.astype(np.uint8), 1.0 / n, periodic=periodic)


def test_two_overlapping_balls_across_the_faces():
    """Two balls of radius R whose centres are d apart: the watershed cuts them by the
    plane of their neck, each keeping its ball less a cap of height R - d / 2; one of
    them crosses a face of the periodic box."""
    radius, d = 0.2, 0.3
    image = balls(64, [(0.95, 0.5, 0.5), (0.95 + d, 0.5, 0.5)], [radius, radius])
    found = particle_sizes(image)
    assert isinstance(found, Particles) and found.count == 2
    height = radius - 0.5 * d
    exact = 4.0 / 3.0 * np.pi * radius**3 - np.pi * height**2 * (3.0 * radius - height) / 3.0
    assert found.volumes == pytest.approx([exact, exact], rel=0.015)
    assert found.volumes.sum() == pytest.approx(np.asarray(image.data).mean())
    assert found.radii == pytest.approx(np.cbrt(0.75 * exact / np.pi), rel=0.005)
    # the centres of the two parts: on the axis, away from the neck, in the box
    order = np.argsort((found.centroids[:, 0] - 0.5) % 1.0)  # 0.94 before 0.26
    centres = found.centroids[order]
    assert centres[:, 1:] == pytest.approx(0.5, abs=1e-9)
    assert 0.93 < centres[0, 0] < 0.95 and 0.25 < centres[1, 0] < 0.27
    assert set(np.unique(found.labels)) == {0, 1, 2} and found.weights == pytest.approx(1.0)
    assert found.density() == pytest.approx(2.0)
    # without the watershed, or with a prominence that their neck does not reach: one
    whole = particle_sizes(image, separate=False)
    assert whole.count == 1 and whole.volumes[0] == pytest.approx(2 * exact, rel=0.005)
    assert whole.centroids[0] == pytest.approx([0.10, 0.5, 0.5], abs=0.005)  # 1.10, wrapped
    assert particle_sizes(image, prominence=0.1).count == 1
    # the other phase, and a phase that is absent
    assert particle_sizes(image, phase=0, separate=False).count == 1
    assert particle_sizes(image, phase=3).count == 0 and particle_sizes(image, 3).density() == 0.0


def test_hard_spheres_recovered_one_by_one():
    """A packing of hard spheres of radii between 0.05 and 0.1, which touch in the
    voxels: as many particles as spheres, at their centres, with their radii."""
    model = mg.models.HardcoreSpheres(
        radius=mg.distributions.Uniform(0.05, 0.1), volume_fraction=0.3
    )
    micro = model.generate(mg.Domain.cube(1.0, periodic=True), seed=3)
    spheres = micro.spheres
    image = micro.voxelize(shape=96)
    assert an.connected_components(image, 1).n_components < len(spheres.radii)  # they touch
    found = particle_sizes(image)
    assert found.count == len(spheres.radii) == 155
    assert np.sort(found.radii) == pytest.approx(np.sort(spheres.radii), rel=0.01)
    tree = cKDTree(np.mod(spheres.centers, 1.0), boxsize=1.0)
    distance, index = tree.query(np.mod(found.centroids, 1.0))
    assert distance.max() < 0.25 / 96 and len(set(index)) == 155
    assert found.radii == pytest.approx(spheres.radii[index], rel=0.01)
    assert found.mean_radius() == pytest.approx(spheres.radii.mean(), rel=1e-3)
    assert found.radius_cv() == pytest.approx(spheres.radii.std() / spheres.radii.mean(), rel=0.01)
    edges = np.linspace(0.05, 0.1, 6)
    classes = found.density(edges)
    assert classes.sum() == pytest.approx(155.0) and np.all(classes > 0)
    # the frame of windows that are not periodic: the spheres they cut are left out, and
    # the others weighted; counting the whole spheres alone misses two thirds of them
    data = np.asarray(image.data)
    weighted, whole, radius = [], [], []
    for corner in np.ndindex(2, 2, 2):
        i, j, k = (48 * c for c in corner)
        window = mg.VoxelImage(data[i : i + 48, j : j + 48, k : k + 48], 1.0 / 96)
        part = particle_sizes(window)
        assert np.all((part.weights == 0.0) | (part.weights > 1.0))
        weighted.append(part.density())
        whole.append(np.count_nonzero(part.weights) / part.volume)
        radius.append(part.mean_radius())
    assert np.mean(weighted) == pytest.approx(155.0, rel=0.03)
    assert np.mean(whole) < 0.4 * 155.0
    assert np.mean(radius) == pytest.approx(spheres.radii.mean(), rel=0.01)
    # small particles left out, the others renumbered
    large = particle_sizes(image, min_radius=0.08)
    assert large.count == np.count_nonzero(found.radii >= 0.08)
    assert set(np.unique(large.labels)) == set(range(large.count + 1))


def test_discs_of_a_planar_image():
    """A planar image (one voxel thick): its particles are the sections of the
    stereology, split by the same watershed."""
    n = 96
    x = (np.arange(n) + 0.5) / n
    grid = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1)
    inside = np.zeros((n, n), bool)
    for centre, radius in (((0.3, 0.3), 0.15), ((0.52, 0.3), 0.12), ((0.7, 0.75), 0.1)):
        inside |= ((grid - centre) ** 2).sum(axis=-1) < radius**2
    image = mg.VoxelImage(inside[:, :, None].astype(np.uint8), 1.0 / n, periodic=True)
    found = particle_sizes(image)
    sections = an.section_sizes(image, 1, 2, separate=True)
    assert found.count == len(sections.radii) == 3
    areas = np.sort(found.volumes) * n  # one voxel thick
    assert np.sqrt(areas / np.pi) == pytest.approx(np.sort(sections.radii), rel=1e-12)
    framed = particle_sizes(mg.VoxelImage(image.data, 1.0 / n, periodic=False))
    assert framed.count == 3 and np.all(framed.weights > 1.0)  # no frame across the plane


def test_watershed_of_a_volume():
    """The kernels: the maxima of a prominence, the markers that share a plateau, the
    voxels outside the mask, the faces of a periodic volume."""
    height = np.zeros((1, 1, 9))
    height[0, 0] = [1, 3, 1, 2, 1, 5, 1, 0, 1]
    mask = np.ones(height.shape, bool)
    peaks = prominent_maxima_volume(height, mask, 1.5, False)
    assert list(np.flatnonzero(peaks)) == [1, 5]  # 2 has a prominence of 1 only
    assert list(np.flatnonzero(prominent_maxima_volume(height, mask, 0.5, False))) == [1, 3, 5, 8]
    # periodic: the last voxel joins the first through the face, no longer a peak
    assert list(np.flatnonzero(prominent_maxima_volume(height, mask, 0.5, True))) == [1, 3, 5]
    labels = np.zeros((1, 1, 9), np.int32)
    labels[0, 0, 1], labels[0, 0, 5] = 1, 2
    labels[0, 0, 7] = -1  # outside the mask
    flooded = labels.copy()
    flood_volume(-height, flooded, False)
    # the higher marker floods first: the saddle 3 is its own; 8 is cut off by 7
    assert list(flooded[0, 0]) == [1, 1, 1, 2, 2, 2, 2, -1, 0]
    flooded = labels.copy()
    flood_volume(-height, flooded, True)
    assert flooded[0, 0, 8] == 1  # reached through the face


def test_particles_measure():
    image = balls(32, [(0.3, 0.3, 0.3), (0.7, 0.7, 0.7)], [0.15, 0.2])
    out = an.measure(image, ["particles"])
    assert out["particle_density"] == pytest.approx(2.0)
    assert out["particle_mean_radius"] == pytest.approx(0.175, rel=0.01)
    assert out["particle_radius_cv"] == pytest.approx(0.025 / 0.175, rel=0.03)
    empty = mg.VoxelImage(np.zeros((4, 4, 4), np.uint8), 0.25, periodic=True)
    assert an.measure(empty, ["particles"]) == {"particle_density": 0.0}
