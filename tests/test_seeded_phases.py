"""Seeds of several phases and the verification of their grains (models.seeded,
mesh.polyhedra): the exact moments of the cells, the shapes of the grains (the ellipsoids of
their second moments), the grains against their seeds, the seeds of each phase of their own
laws and in their fractions, the amorphous phases merged into one region, the placement of
seeds of very different sizes, the exact measure ``exact_grains``; the seeds of the plane
(ellipses, columnar grains); the figure of the grains against their seeds."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

import morphogen as mg
from morphogen.distributions import Constant, LogNormal, Uniform
from morphogen.kernels.seeds import place_seeds
from morphogen.mesh.conforming import has_mmg
from morphogen.models.seeded import SeededLaguerre, SeededTessellation
from morphogen.models.tessellation import Tessellation
from morphogen.morphology.measures import measure_exact


def test_moments_of_the_pieces():
    """The volumes, centroids and second moments of the cells of a tessellation of a box
    that is not periodic sum to those of the box, exactly; their volumes those of the cell
    complex."""
    lengths = np.array([1.0, 2.0, 1.5])
    micro = mg.models.Laguerre(n_cells=30, radius=Uniform(0.05, 0.2)).generate(
        mg.Domain(size=tuple(lengths), periodic=False), seed=3
    )
    cx = micro.cell_complex()
    volume, centroid, second = cx.piece_moments()
    np.testing.assert_allclose(volume, cx.piece_volumes(), atol=1e-14)
    box = float(np.prod(lengths))
    centre = micro.domain.lower + lengths / 2.0
    assert volume.sum() == pytest.approx(box, rel=1e-13)
    np.testing.assert_allclose((volume[:, None] * centroid).sum(axis=0), box * centre,
                               rtol=1e-12)  # fmt: skip
    d = centroid - centre
    total = second.sum(axis=0) + np.einsum("n,ni,nj->ij", volume, d, d)
    np.testing.assert_allclose(total, np.diag(box * lengths**2 / 12.0), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(second) > 0)


def test_shapes_of_the_grains():
    """The cell of a single germ in a periodic box is the box: its ellipsoid of the same
    second moments has the semi-axes ``sqrt(5/12) L``; the grains of a periodic tessellation
    fill the box, gathered across its faces; in a box that is not periodic, the grains cut
    by its faces are told apart."""
    one = mg.models.Voronoi(n_cells=1).generate(mg.Domain(size=(2.0, 1.0, 1.0), periodic=True),
                                                seed=1)  # fmt: skip
    shapes = one.grain_shapes()
    assert shapes["volume"].tolist() == pytest.approx([2.0])
    np.testing.assert_allclose(shapes["semi_axes"][0], np.sqrt(5.0 / 12.0) * np.array([2, 1, 1]))
    assert abs(shapes["axes"][0][:, 0] @ [1.0, 0.0, 0.0]) == pytest.approx(1.0)
    cells = mg.models.Voronoi(n_cells=40).generate(mg.Domain.cube(1.0, periodic=True), seed=2)
    shapes = cells.grain_shapes()
    assert shapes["volume"].sum() == pytest.approx(1.0, rel=1e-12)
    assert shapes["inside"].all() and np.all((shapes["centroid"] >= 0) & (shapes["centroid"] < 1))
    np.testing.assert_allclose(shapes["diameter"], np.cbrt(6.0 * shapes["volume"] / math.pi))
    cut = mg.models.Voronoi(n_cells=40).generate(mg.Domain.cube(1.0, periodic=False), seed=2)
    shapes = cut.grain_shapes()
    assert 0 < shapes["inside"].sum() < cut.n_grains
    lows = cut.domain.lower
    touching = [g for g in range(cut.n_grains) if shapes["inside"][g]]
    assert all(np.all(shapes["centroid"][g] > lows) for g in touching)


def test_grains_against_their_seeds():
    """Seeds of lognormal diameters and of ratios a/b uniform in [1.5, 2.5] at random: the
    diameters of the grains within 5 % of their seeds (RMS), their ratios within 0.3, their
    long axes within 10° of those of their seeds on average, the diameters and the ratios
    of the grains drawn from the laws of the seeds (Kolmogorov–Smirnov below 0.15)."""
    model = SeededLaguerre(diameter=LogNormal(1.0, 0.2), aspect_ratio=Uniform(1.5, 2.5))
    micro = model.generate(mg.Domain.cube(4.0, periodic=True), seed=1)
    cmp = micro.seed_comparison()
    rel = (cmp["grain_diameter"] - cmp["seed_diameter"]) / cmp["seed_diameter"]
    assert np.sqrt(np.mean(rel**2)) < 0.05
    errors = cmp["errors"]
    assert errors["aspect_ratio"]["rmse"] < 0.3 and errors["diameter"]["r2"] > 0.9
    assert errors["axis_angle"]["mean"] < 10.0
    assert errors["diameter"]["ks"][1] < 0.15 and errors["aspect_ratio"]["ks"][1] < 0.15
    assert math.isnan(errors["flatness"]["ks"][1])  # a constant law
    assert cmp["fractions"] == {1: (pytest.approx(1.0), 1.0)}  # every grain of phase 1
    columns = measure_exact(micro, ["exact_grains"])
    assert columns["grains"] == micro.n_grains
    assert columns["seed_diameter_error"] == pytest.approx(np.sqrt(np.mean(rel**2)))
    assert columns["seed_axis_angle"] == pytest.approx(errors["axis_angle"]["mean"])
    with pytest.raises(ValueError, match="tessellation"):
        measure_exact(mg.models.BooleanSpheres(radius=0.3, volume_fraction=0.2).generate(
            mg.Domain.cube(2.0), seed=1), ["exact_grains"])  # fmt: skip


def test_seeds_of_each_phase():
    """Two phases of their own diameters (1 and 2): the shares of their seeds ``q / E[V]``,
    the fractions of their volumes those asked (within 0.08), the diameters of the grains of
    each phase those of its seeds; the parameters checked."""
    model = SeededLaguerre(phase_fractions=(0.5, 0.5), phase_diameter=(1.0, 2.0),
                           phase_orientation=("random", "aligned"))  # fmt: skip
    shares = [f.share for f in model.families()]
    assert shares == pytest.approx([8.0 / 9.0, 1.0 / 9.0])
    assert model.mean_seed_volume() == pytest.approx(
        1.0 / (0.5 / (math.pi / 6) + 0.5 / (8 * math.pi / 6))
    )
    micro = model.generate(mg.Domain.cube(8.0, periodic=True), seed=2)
    cmp = micro.seed_comparison()
    for phase, size in ((0, 1.0), (1, 2.0)):
        assert cmp["fractions"][phase][0] == pytest.approx(0.5, abs=0.08)
        mine = cmp["phase"] == phase
        assert np.median(cmp["grain_diameter"][mine]) == pytest.approx(size, rel=0.08)
    assert np.all(micro.seeds["angles"][micro.grain_phases == 1] == 0.0)  # aligned seeds
    theory = model.theory(mg.Domain.cube(8.0))
    assert theory["volume_fraction_1"] == 0.5 and theory["expected_grains"] > 0
    assert mg.models.model_from_config(model.to_config()) == model
    for wrong in ({"phase_diameter": (1.0, 2.0)},
                  {"phase_fractions": (0.5, 0.5), "phase_diameter": (1.0,)},
                  {"phase_fractions": (0.5, 0.5), "phase_aspect_ratio": (1.0, 0.5)},
                  {"phase_fractions": (0.5, 0.5), "phase_orientation": ("random", "fibre")},
                  {"phase_fractions": (0.5, 0.5), "amorphous_phases": (2,)}):  # fmt: skip
        with pytest.raises(ValueError):
            SeededLaguerre(**wrong)


def test_amorphous_phases(tmp_path):
    """Crystals in an amorphous matrix: the seeds of the matrix merged into one region (one
    grain, one label, no seed, the identity orientation), the distance to the grain
    boundaries in it that to the crystals (not to the faces of its cells; a lower bound);
    saved without NaN and read back; meshed with the matrix as one subdomain, or left out
    (a void)."""
    model = SeededLaguerre(phase_fractions=(0.3, 0.7), phase_diameter=(Constant(0.25),
                           Constant(0.5)), phase_aspect_ratio=(1.0, 2.0), amorphous_phases=(0,),
                           crystal_orientations=True)  # fmt: skip
    micro = model.generate(mg.Domain.cube(1.0, periodic=True), seed=4)
    region = np.flatnonzero(micro.amorphous)
    assert len(region) == 1 and micro.grain_phases[region[0]] == 0
    assert micro.metadata["amorphous_phases"] == [0]
    assert np.all(np.isnan(micro.seeds["axes"][region]))
    np.testing.assert_array_equal(micro.orientations[region], 0.0)
    labels = micro.labels(32).data
    phases = micro.voxelize(32).data
    assert set(np.unique(labels[phases == 0])) == {region[0]}
    points = np.random.default_rng(5).uniform(0.0, 1.0, (400, 3))
    inside = points[micro.grain_of(points) == region[0]]
    distance = micro.boundary_distance(inside)  # to the crystals: not the faces of the region
    cells = Tessellation.boundary_distance(micro, inside)  # to the faces of every cell
    assert np.all(distance >= cells - 1e-12) and np.mean(distance > cells + 1e-9) > 0.2
    probe = np.random.default_rng(6).uniform(0.0, 1.0, (20000, 3))
    crystals = cKDTree(probe[micro.grain_of(probe) != region[0]], boxsize=1.0)
    assert np.all(distance <= crystals.query(inside)[0] + 1e-12)  # a lower bound
    path = tmp_path / "rock.json"
    mg.io.save_microstructure(micro, path)
    assert "NaN" not in path.read_text()
    json.loads(path.read_text())
    again = mg.io.load_microstructure(path)
    assert (
        isinstance(again, SeededTessellation)
        and again.amorphous.tolist() == micro.amorphous.tolist()
    )
    assert again.cell_table()["a"][region[0]] != again.cell_table()["a"][region[0]]  # NaN
    if not has_mmg():
        return
    from morphogen.mesh.polycrystal import polycrystal_mesh

    mesh = polycrystal_mesh(micro, size=0.15)
    assert set(np.unique(mesh.cell)) == set(range(micro.n_grains))
    solid = polycrystal_mesh(micro, size=0.15, phases=(1,))
    assert set(np.unique(solid.phase)) == {1} and region[0] not in set(np.unique(solid.cell))
    with pytest.raises(ValueError, match="phases"):
        polycrystal_mesh(micro, size=0.15, phases="crystals")


def test_placement_of_seeds_of_very_different_sizes():
    """Spheres of radii 0.02 to 0.6 placed in boxes periodic or not (the grids by levels of
    sizes): every seed that clears overlaps those placed before it within the tolerance, by
    brute force."""
    rng = np.random.default_rng(5)
    for periodic in (True, False):
        radii = np.concatenate([rng.uniform(0.3, 0.6, 15), rng.uniform(0.02, 0.06, 400)])
        radii = radii[np.argsort(-radii)]
        n = len(radii)
        box = np.array([4.0, 3.0, 5.0])
        _, placed, crowd, pos = place_seeds(np.zeros((n, 3)), radii, np.arange(n + 1), box, rng,
                                            periodic=periodic, rtol=0.2, attempts=300)  # fmt: skip
        d = pos[:, None] - pos[None]
        if periodic:
            d -= box * np.round(d / box)
        small = np.minimum(radii[:, None], radii[None])
        excess = (radii[:, None] + radii[None] - np.linalg.norm(d, axis=2)) / small - 0.2
        for i in np.flatnonzero(placed & ~crowd):
            earlier = np.flatnonzero(placed[:i])
            assert excess[i, earlier].max(initial=-np.inf) <= 1e-12
        assert placed.sum() > 400


def test_seeds_of_the_plane():
    """Elliptical seeds in the middle plane of a thin box: germs in that plane, columnar
    grains (prisms through the thickness, the same labels on every layer), the areas of the
    grains summing to that of the box, their diameters (of the discs of their areas) within
    8 % of their seeds, their long axes along those of their seeds, the tolerance of the
    plane; in a box that is not periodic, the seeds about it in the plane only; the
    parameters checked."""
    from morphogen.models.seeded import overlap_tolerance

    model = SeededLaguerre(
        dimension=2, diameter=LogNormal(1.0, 0.2), aspect_ratio=Uniform(1.5, 3.0)
    )
    box = mg.Domain(size=(8.0, 8.0, 0.2), periodic=True)
    micro = model.generate(box, seed=1)
    assert micro.metadata["dimension"] == 2 and np.ptp(micro.germs[:, 2]) == 0.0
    assert micro.germs[0, 2] == pytest.approx(0.1)
    assert micro.metadata["overlap"] == pytest.approx(
        overlap_tolerance(math.pi * micro.seeds["axes"][:, 0] * micro.seeds["axes"][:, 1], 2),
        rel=0.05,
    )
    assert overlap_tolerance(np.ones(5), 2) == pytest.approx(0.184959 / 0.365096)
    layers = micro.labels((64, 64, 4)).data
    assert all(np.array_equal(layers[:, :, 0], layers[:, :, k]) for k in range(1, 4))
    shapes = micro.grain_shapes()
    assert shapes["area"].sum() == pytest.approx(64.0, rel=1e-9)
    assert shapes["semi_axes"].shape == (micro.n_grains, 2)
    cmp = micro.seed_comparison()
    assert "flatness" not in cmp["errors"] and "seed_flatness" not in cmp
    rel = (cmp["grain_diameter"] - cmp["seed_diameter"]) / cmp["seed_diameter"]
    assert np.sqrt(np.mean(rel**2)) < 0.08 and cmp["errors"]["axis_angle"]["mean"] < 8.0
    columns = measure_exact(micro, ["exact_grains"])
    assert "grain_flatness_median" not in columns and columns["grains"] == micro.n_grains
    theory = model.theory(box)
    assert theory["expected_grains"] == pytest.approx(64.0 / theory["mean_seed_area"])
    assert micro.n_grains == pytest.approx(theory["expected_grains"], rel=0.15)
    cut = model.generate(mg.Domain(size=(6.0, 6.0, 0.2), periodic=False), seed=2)
    assert np.all(cut.germs[:, 2] == pytest.approx(0.1))
    assert cut.germs[:, :2].min() < 0.0 and cut.germs[:, :2].max() > 6.0
    shapes = cut.grain_shapes()
    assert 0 < shapes["inside"].sum() < cut.n_grains  # the faces z do not cut the columns
    flat = {"dimension": 2, "phase_fractions": (0.5, 0.5), "phase_flatness": (1.0, 1.5)}
    for wrong in ({"dimension": 4}, {"dimension": 2, "flatness": 2.0}, flat):
        with pytest.raises(ValueError):
            SeededLaguerre(**wrong)


def test_figure_of_the_grains_against_their_seeds():
    """The figure of the verification: a panel per quantity (b/c but in the plane), the
    angles and the fractions; drawn for two phases in space and for the plane."""
    pytest.importorskip("matplotlib")
    from morphogen.viz.figures import seed_figure

    model = SeededLaguerre(phase_fractions=(0.5, 0.5), phase_diameter=(0.6, 1.0), aspect_ratio=2.0)
    two = model.generate(mg.Domain.cube(2.0, periodic=True), seed=3)
    fig = seed_figure(two, title="two phases")
    assert len(fig.axes) == 5 and fig.axes[0].get_xlabel() == "equivalent diameter"
    plane = SeededLaguerre(dimension=2, aspect_ratio=2.0).generate(
        mg.Domain(size=(4.0, 4.0, 0.1), periodic=True), seed=4
    )
    assert len(seed_figure(plane).axes) == 4
