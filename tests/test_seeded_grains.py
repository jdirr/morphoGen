"""Grains grown from seeds (models.seeded, kernels.seeds): the breakdown of an ellipsoid into
the spheres of its medial surface (inscribed, touching it, filling it), the overlap that
the placement tolerates, the placement itself (the tolerance obeyed, the crowded seeds
placed or left out), the grains (unions of the Laguerre cells of the spheres of their
seeds: their volumes fill the box, their shapes and axes those of the seeds), their
configuration, labels and table, and their meshes, surfaces and crack meshes by grain."""

from __future__ import annotations

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.distributions import Constant
from morphogen.kernels.seeds import place_seeds
from morphogen.mesh.conforming import has_mmg
from morphogen.models.seeded import (
    MOST_SPHERES,
    SeededLaguerre,
    SeededTessellation,
    ellipsoid_spheres,
    overlap_tolerance,
)
from morphogen.orientations import euler_to_matrix

SHAPES = [(2.0, 1.0, 1.0), (4.0, 1.0, 1.0), (2.0, 2.0, 1.0), (3.0, 2.0, 1.0), (4.0, 2.0, 0.5)]


def inside(points, a, b, c):
    return (points[:, 0] / a) ** 2 + (points[:, 1] / b) ** 2 + (points[:, 2] / c) ** 2


@pytest.mark.parametrize("abc", SHAPES)
def test_breakdown_of_an_ellipsoid(abc):
    """Each sphere touching the ellipsoid at the nearest points of its surface in closed
    form (on the surface, at the distance of its radius, the normal pointing back to its
    centre) and inside it (no point of a fine sampling of the surface nearer); their union
    93 % of the ellipsoid at the spacing 0.5 and 97 % at 0.25; a sphere is its own
    breakdown."""
    a, b, c = abc
    rng = np.random.default_rng(0)
    theta, phi = np.meshgrid(np.linspace(0.0, np.pi, 301), np.linspace(0.0, 2 * np.pi, 601))
    surface = np.column_stack([a * (np.sin(theta) * np.cos(phi)).ravel(),
                               b * (np.sin(theta) * np.sin(phi)).ravel(),
                               c * np.cos(theta).ravel()])  # fmt: skip
    trial = rng.uniform(-1.0, 1.0, (60000, 3)) * abc
    trial = trial[inside(trial, a, b, c) <= 1.0]
    for spacing, least in ((0.5, 0.93), (0.25, 0.97)):
        centres, radii = ellipsoid_spheres(a, b, c, spacing=spacing, most=10**6)
        assert np.all(centres[:, 2] == 0.0) and np.all(radii > 0) and radii.max() <= c + 1e-12
        x, y = centres[:, 0], centres[:, 1]
        wide = b * b * y / (b * b - c * c) if b > c else np.zeros_like(y)  # prolate: y = 0
        near = np.column_stack([a * a * x / (a * a - c * c), wide, np.zeros_like(x)])
        near[:, 2] = c * np.sqrt(np.clip(1.0 - (near[:, 0] / a) ** 2 - (near[:, 1] / b) ** 2,
                                         0.0, None))  # fmt: skip
        np.testing.assert_allclose(inside(near, a, b, c), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(near - centres, axis=1), radii, rtol=1e-12)
        normal = near / np.array([a * a, b * b, c * c])
        cross = np.cross(normal, near - centres)
        assert np.abs(cross).max() <= 1e-9 * np.abs(normal).max() * radii.max()
        for k in range(0, len(radii), 5):
            gap = np.linalg.norm(surface - centres[k], axis=1).min()
            assert gap >= radii[k] * (1.0 - 1e-12)  # no point of the surface nearer
        covered = np.zeros(len(trial), dtype=bool)
        for x, r in zip(centres, radii, strict=True):
            covered |= np.sum((trial - x) ** 2, axis=1) <= r * r
        assert covered.mean() >= least
    assert len(ellipsoid_spheres(8.0, 1.0, 0.5)[1]) <= MOST_SPHERES
    centres, radii = ellipsoid_spheres(1.5, 1.5, 1.5)
    assert centres.tolist() == [[0.0, 0.0, 0.0]] and radii.tolist() == [1.5]
    with pytest.raises(ValueError, match="semi-axes"):
        ellipsoid_spheres(1.0, 2.0, 1.0)


def test_overlap_tolerance():
    """The rational fit of Hart and Rimoli to the coefficient of variation of the volumes
    of the seeds: 0.695 for seeds of one size, 0.82 for a variation of 0.5, 0.48 for a
    variation of 1."""
    assert overlap_tolerance(np.ones(10)) == pytest.approx(0.297562 / 0.428216)
    for sigma, expected in ((0.48, 0.82), (0.83, 0.48)):
        v = np.random.default_rng(1).lognormal(0.0, sigma, 200000)
        cv = v.std() / v.mean()
        fit = (0.471115 * cv**2 - 0.602324 * cv + 0.297562) / (cv**2 - 1.08469 * cv + 0.428216)
        assert overlap_tolerance(v) == pytest.approx(fit)
        assert fit == pytest.approx(expected, abs=0.02)


def separations(positions, radii, box):
    """The overlap of every pair of spheres of different seeds relative to the smaller
    radius (periodic box)."""
    d = positions[:, None, :] - positions[None, :, :]
    d -= box * np.round(d / box)
    gap = np.linalg.norm(d, axis=2)
    small = np.minimum(radii[:, None], radii[None, :])
    return (radii[:, None] + radii[None, :] - gap) / small


def test_placement_of_seeds():
    """Spheres (one per seed) and pairs of spheres placed in a periodic box: the seeds that
    clear overlap the others by at most the tolerance; in a box too small, the seeds that
    cannot clear are left out, or placed (crowded) where they overlap least."""
    rng = np.random.default_rng(2)
    box = np.array([4.0, 4.0, 4.0])
    for rtol in (0.0, 0.5):
        first = np.arange(31)
        radii = np.full(30, 0.5)
        centres, placed, crowd, pos = place_seeds(np.zeros((30, 3)), radii, first, box, rng,
                                                  periodic=True, rtol=rtol)  # fmt: skip
        assert placed.all() and not crowd.any()
        excess = separations(pos, radii, box)
        np.fill_diagonal(excess, -np.inf)
        assert excess.max() <= rtol + 1e-12
        np.testing.assert_allclose(pos, centres % box)
    # seeds of two spheres, too many for the box
    n = 120
    offsets = np.tile([[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]], (n, 1))
    radii = np.full(2 * n, 0.5)
    first = np.arange(0, 2 * n + 1, 2)
    seed_of = np.repeat(np.arange(n), 2)
    for crowded in (False, True):
        rng = np.random.default_rng(3)
        centres, placed, crowd, pos = place_seeds(offsets, radii, first, box, rng,
                                                  periodic=True, rtol=0.0, attempts=200,
                                                  crowded=crowded)  # fmt: skip
        if not crowded:
            assert 0 < placed.sum() < n and not crowd.any()
        else:
            assert placed.all() and 0 < crowd.sum() < n
        clear = placed & ~crowd
        kept = placed[seed_of]
        excess = separations(pos[kept], radii[kept], box)
        same = seed_of[kept][:, None] == seed_of[kept][None, :]
        excess[same] = -np.inf
        cleared = clear[seed_of[kept]]
        assert excess[np.ix_(cleared, cleared)].max() <= 1e-12


def test_grains_of_spheres():
    """Seeds of one size, each a sphere: one cell per grain, the grains filling the box
    (their volumes summing to it), of the volume of the seeds on average; the overlap of
    Hart and Rimoli; a box that is not periodic cut from seeds about it."""
    model = SeededLaguerre(diameter=1.0)
    micro = model.generate(mg.Domain.cube(4.0, periodic=True), seed=4)
    assert isinstance(micro, SeededTessellation)
    seed_volume = math.pi / 6.0
    assert micro.n_grains == micro.n_cells == micro.metadata["seeds_drawn"]
    assert micro.n_grains == math.ceil(64.0 / seed_volume)
    assert micro.metadata["overlap"] == pytest.approx(0.6949, abs=1e-4)
    assert micro.metadata["seeds_left_out"] == micro.metadata["seeds_crowded"] == 0
    volumes = micro.grain_volumes()
    assert volumes.sum() == pytest.approx(64.0, rel=1e-9) and np.all(volumes > 0)
    assert volumes.mean() == pytest.approx(seed_volume, rel=0.02)
    assert model.theory(mg.Domain.cube(4.0))["mean_grain_volume"] == pytest.approx(seed_volume)
    box = mg.Domain.cube(4.0, periodic=False)
    cut = model.generate(box, seed=5)
    assert cut.metadata["germ_margin"] == 1.0
    assert np.any(np.any((cut.germs < 0.0) | (cut.germs > 4.0), axis=1))
    assert cut.grain_volumes().sum() == pytest.approx(64.0, rel=1e-9)


def grain_shapes(micro, n):
    """The ratios of the semi-axes of the grains and the cosine of their long axes with
    those of their seeds, from the second moments of a label image of ``n³`` voxels."""
    image = micro.labels(n)
    labels = image.data.ravel()
    lengths = micro.domain.lengths
    x = micro.domain.lower + (np.indices((n, n, n)).reshape(3, -1).T + 0.5) * lengths / n
    d = x - micro.seeds["centres"][labels]
    d -= lengths * np.round(d / lengths)
    out = []
    for g in range(micro.n_grains):
        mine = labels == g
        if mine.sum() < 100:
            continue
        dd = d[mine] - d[mine].mean(axis=0)
        w, v = np.linalg.eigh(dd.T @ dd)
        w = np.sqrt(w[::-1])
        axis = euler_to_matrix(micro.seeds["angles"][g])[0][0]  # a, in the sample
        out.append((w[0] / w[1], w[1] / w[2], abs(float(v[:, -1] @ axis))))
    return np.asarray(out)


def test_grains_follow_their_seeds():
    """Seeds of aspect ratio 2.5 at random: the grains elongated as their seeds, along the
    long axes of their seeds; flat seeds aligned with the box give flat grains."""
    long_ = SeededLaguerre(diameter=1.0, aspect_ratio=Constant(2.5))
    micro = long_.generate(mg.Domain.cube(4.0, periodic=True), seed=6)
    shapes = grain_shapes(micro, 64)
    assert len(shapes) > 0.8 * micro.n_grains
    assert 2.0 < np.median(shapes[:, 0]) < 3.0 and np.median(shapes[:, 1]) < 1.35
    assert np.median(shapes[:, 2]) > 0.98
    flat = SeededLaguerre(diameter=1.0, flatness=Constant(2.0), orientation="aligned")
    shapes = grain_shapes(flat.generate(mg.Domain.cube(4.0, periodic=True), seed=7), 64)
    assert np.median(shapes[:, 0]) < 1.35 and 1.6 < np.median(shapes[:, 1]) < 2.4


def test_crowded_seeds():
    """Elongated seeds jam the sequential addition: placed crowded, every seed has its
    grain and the grains the volume of the seeds; left out, as MicroStructPy, their
    volume goes to the grains of the others."""
    domain = mg.Domain.cube(4.0, periodic=True)
    common = {"diameter": 1.0, "aspect_ratio": Constant(3.0), "attempts": 2000}
    placed = SeededLaguerre(**common).generate(domain, seed=8)
    m = placed.metadata
    assert m["seeds_left_out"] == 0 and m["seeds_crowded"] > 0
    assert placed.n_grains == m["seeds_drawn"]
    left = SeededLaguerre(**common, crowded="leave").generate(domain, seed=8)
    m = left.metadata
    assert m["seeds_crowded"] == 0 and m["seeds_left_out"] > 0.1 * m["seeds_drawn"]
    assert left.n_grains == m["seeds_drawn"] - m["seeds_left_out"]


def test_configuration_labels_and_table(tmp_path):
    """The parameters checked; the grains of two phases with their crystal orientations;
    the labels those of the grains, the grain of a point that of its sphere; the
    realization saved and read back; the table of the grains."""
    for wrong in ({"aspect_ratio": 0.5}, {"flatness": mg.distributions.Uniform(0.5, 2.0)},
                  {"orientation": "fibre"}, {"crowded": "keep"}, {"overlap": 1.5},
                  {"attempts": 0}, {"phase_fractions": (0.5, 0.6)}):  # fmt: skip
        with pytest.raises(ValueError):
            SeededLaguerre(**wrong)
    model = SeededLaguerre(diameter=mg.distributions.Uniform(0.8, 1.2), aspect_ratio=1.5,
                           phase_fractions=(0.3, 0.7), crystal_orientations=True)  # fmt: skip
    assert mg.models.model_from_config(model.to_config()) == model
    micro = model.generate(mg.Domain.cube(3.0, periodic=True), seed=9)
    assert set(np.unique(micro.grain_phases)) == {0, 1}
    assert micro.orientations.shape == (micro.n_grains, 3)
    labels = micro.labels(24)
    assert labels.data.max() == micro.n_grains - 1
    points = np.random.default_rng(10).uniform(0.0, 3.0, (500, 3))
    np.testing.assert_array_equal(micro.grain_of(points), micro.grains[micro.cell_of(points)])
    phases = micro.voxelize(24).data
    np.testing.assert_array_equal(phases, micro.grain_phases[labels.data])
    path = tmp_path / "grains.json"
    mg.io.save_microstructure(micro, path)
    again = mg.io.load_microstructure(path)
    assert isinstance(again, SeededTessellation)
    np.testing.assert_array_equal(again.labels(24).data, labels.data)
    np.testing.assert_allclose(again.orientations, micro.orientations)
    with pytest.raises(ValueError, match="Neper"):
        mg.io.write_neper(tmp_path / "grains", micro)
    table = micro.cell_table()
    assert len(table["id"]) == micro.n_grains and table["spheres"].sum() == micro.n_cells
    assert np.all(table["a"] >= table["b"]) and np.all(table["b"] >= table["c"])
    np.testing.assert_allclose(table["a"] / table["b"], 1.5)
    assert micro.summary() == {"n_grains": micro.n_grains, "n_spheres": micro.n_cells}


def test_surface_of_the_grains():
    """The exact surfaces of the cells, by grain: no facet between cells of a grain, the
    regions the grains with their phases."""
    from morphogen.viz.surfaces import _cells_surface

    micro = SeededLaguerre(diameter=0.5, aspect_ratio=2.0, phase_fractions=(0.5, 0.5)).generate(
        mg.Domain.cube(1.0, periodic=True), seed=11
    )
    assert micro.n_cells > micro.n_grains
    surface = _cells_surface(micro, 32)
    pairs = [(int(g[0]), int(g[1])) for g in surface.groups]
    assert pairs and all(a != b for a, b in pairs)
    assert max(max(pair) for pair in pairs) < micro.n_grains
    np.testing.assert_array_equal(surface.region_phase, micro.grain_phases)
    np.testing.assert_array_equal(surface.region_label, np.arange(micro.n_grains))


def test_grains_of_the_elements_of_a_crack_mesh():
    """The elements of a mesh in the grains of their centroids (the crack meshes of
    polycrystals): grains, not the cells of their spheres, with the orientations of the
    grains."""
    from morphogen.fracture.lefm import with_grains
    from morphogen.mesh.conforming import background_mesh
    from morphogen.mesh.tetra import TetMesh

    micro = SeededLaguerre(diameter=0.5, aspect_ratio=2.0, crystal_orientations=True).generate(
        mg.Domain.cube(1.0, periodic=True), seed=12
    )
    nodes, tets, _, _ = background_mesh((6, 6, 6), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    mesh = TetMesh(nodes, tets, np.ones(len(tets), dtype=np.int64), (0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0))  # fmt: skip
    grained = with_grains(mesh, micro)
    centroids = nodes[tets].mean(axis=1)
    np.testing.assert_array_equal(grained.cell, micro.grains[micro.cell_of(centroids)])
    assert grained.cell.max() < micro.n_grains < micro.n_cells
    assert grained.orientations.shape == (micro.n_grains, 3)
    assert mg.models.Voronoi(n_cells=5).generate(mg.Domain.cube(1.0), seed=1).n_grains == 5


@pytest.mark.skipif(not has_mmg(), reason="MMG (pymmg) is not installed")
def test_mesh_of_the_grains():
    """The conforming mesh of the grains: one subdomain per grain (the cells of its spheres
    merged), their volumes those of the grains, their phases and orientations."""
    from morphogen.mesh.polycrystal import polycrystal_mesh
    from morphogen.mesh.tetra import tet_volumes

    micro = SeededLaguerre(diameter=0.5, aspect_ratio=2.0, crystal_orientations=True,
                           phase_fractions=(0.3, 0.7)).generate(
        mg.Domain.cube(1.0, periodic=True), seed=3)  # fmt: skip
    mesh = polycrystal_mesh(micro, size=0.1)
    assert set(np.unique(mesh.cell)) == set(range(micro.n_grains))
    per_grain = np.bincount(mesh.cell, tet_volumes(mesh.nodes, mesh.elements))
    np.testing.assert_allclose(per_grain, micro.grain_volumes(), rtol=0.02)
    np.testing.assert_array_equal(mesh.phase, micro.grain_phases[mesh.cell])
    assert mesh.orientations.shape == (micro.n_grains, 3)
    assert mesh.metadata["n_grains"] == micro.n_grains < mesh.metadata["n_cells"]
