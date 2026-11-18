"""Lattice growth: nucleation and growth (JMAK), Potts grain growth, aggregation."""

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

import morphogen as mg
from morphogen import io
from morphogen.automata.growth import (
    aggregate,
    grow_grains,
    potts_coarsening,
    shape_volume_coefficient,
)
from morphogen.models import AggregationModel, LatticeField, NucleationGrowthModel

# -- nucleation and growth -----------------------------------------------------------------


def test_site_saturation_gives_the_voronoi_tessellation_of_the_nuclei(rng):
    n = 24
    cells = rng.integers(0, n, (40, 3))
    labels, arrival, real = grow_grains((n, n, n), 0.5, cells, np.zeros(40))
    assert real.sum() == len(np.unique(cells, axis=0))
    centres = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) * 0.5
    dist, _ = cKDTree((cells + 0.5) * 0.5, boxsize=n * 0.5).query(centres)
    assert np.allclose(arrival.ravel(), dist)  # exact Euclidean distances
    owner = (cells[labels.ravel()] + 0.5) * 0.5
    d = owner - centres
    d -= n * 0.5 * np.round(d / (n * 0.5))
    assert np.allclose(np.linalg.norm(d, axis=1), dist)  # the nearest nucleus (ties aside)
    flat, _, _ = grow_grains((n, n, 1), 1.0, cells * [1, 1, 0], np.zeros(40), dims=2)
    assert flat.shape == (n, n, 1)
    _, warr, _ = grow_grains((n, n, n), 1.0, cells, np.zeros(40), periodic=False)
    dist, _ = cKDTree(cells + 0.5).query(centres / 0.5)
    assert np.allclose(warr.ravel(), dist)


@pytest.mark.parametrize(("shape", "norm"), [("cube", np.inf), ("octahedron", 1)])
def test_faceted_growth_metrics(rng, shape, norm):
    n = 16
    cells = rng.integers(0, n, (6, 3))
    _, arrival, _ = grow_grains(
        (n, n, n), 1.0, cells, np.zeros(6), growth_shape=shape, periodic=False, growth_rate=2.0
    )
    centres = np.indices((n, n, n)).reshape(3, -1).T
    d = np.min([np.linalg.norm(centres - c, ord=norm, axis=1) for c in cells], axis=0)
    assert np.allclose(arrival.ravel(), d / 2.0)
    # a rotation by 45 degrees about z turns cubes into diamonds in the xy plane
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    rot = np.array([[[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]])
    _, arr, _ = grow_grains(
        (9, 9, 1),
        1.0,
        [[4, 4, 0]],
        [0.0],
        growth_shape="cube",
        rotations=rot,
        periodic=False,
        dims=2,
    )
    assert arr[4, 8, 0] == pytest.approx(4 / math.sqrt(2))  # along a diagonal of the square
    assert arr[8, 8, 0] == pytest.approx(4 * math.sqrt(2))  # towards a vertex
    assert shape_volume_coefficient("cube", 2) == 4.0


def test_phantom_nuclei():
    cells = np.array([[0, 0, 0], [3, 0, 0], [8, 8, 8]])
    labels, arrival, real = grow_grains((10, 10, 10), 1.0, cells, [0.0, 5.0, 2.0])
    assert real.tolist() == [True, False, True]  # (3, 0, 0) is transformed at t = 3
    assert labels[3, 0, 0] == 0 and arrival[3, 0, 0] == 3.0
    assert arrival[8, 8, 8] == 2.0 and labels[8, 8, 8] == 2
    with pytest.raises(ValueError, match="growth_shape"):
        grow_grains((4, 4, 4), 1.0, cells, np.zeros(3), growth_shape="prism")


def _mean_over(model, domain, seeds, value):
    return np.mean([value(model.generate(domain, seed=s)) for s in range(seeds)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nucleus_density": 2e-3, "time": 4.0},
        {"nucleation": "continuous", "nucleation_rate": 2e-4, "time": 7.0},
        {
            "nucleus_density": 2e-3,
            "time": 3.0,
            "growth_shape": "cube",
            "crystal_orientations": True,
        },
    ],
)
def test_avrami_kinetics(kwargs):
    """Transformed fractions follow X = 1 - exp(-X_ext) (Kolmogorov, Johnson-Mehl, Avrami)."""
    model = NucleationGrowthModel(**kwargs)
    domain = mg.Domain.cube(40.0)
    x = _mean_over(model, domain, 6, lambda m: m.phases.mean())
    expected = model.theory(domain)["transformed_fraction"]
    assert 0.2 < expected < 0.8
    assert x == pytest.approx(expected, abs=0.03)
    micro = model.generate(domain, seed=0)
    k = micro.attributes["kinetics"]
    assert k["transformed"][-1] == pytest.approx(micro.phases.mean())
    assert np.all(np.diff(k["transformed"]) >= 0)


def test_johnson_mehl_grain_density():
    model = NucleationGrowthModel(nucleation="continuous", nucleation_rate=2e-4)
    domain = mg.Domain.cube(40.0)
    grains = _mean_over(model, domain, 6, lambda m: m.summary()["n_grains"])
    assert grains == pytest.approx(model.expected_count(domain), rel=0.08)
    th = model.theory(domain)
    assert th["avrami_exponent"] == 4.0
    assert th["mean_grain_volume"] == pytest.approx(1.0 / th["grain_density"])
    # X(t_1/2) = 1/2
    assert model.transformed_fraction(th["half_transformation_time"]) == pytest.approx(0.5)


def test_planar_kinetics():
    plate = mg.Domain(size=(96.0, 96.0, 2.0))
    model = NucleationGrowthModel(nucleus_density=5e-4, dimension=2, time=6.0)
    x = _mean_over(model, plate, 10, lambda m: m.phases.mean())
    expected = model.theory(plate)["transformed_fraction"]
    assert expected == pytest.approx(1 - math.exp(-5e-4 * 2.0 * math.pi * 36.0))
    assert x == pytest.approx(expected, abs=0.03)
    assert model.theory() == {}
    assert model.generate(plate, seed=1).shape == (96, 96, 1)


def test_nucleation_growth_structures(tmp_path):
    domain = mg.Domain.cube(24.0)
    model = NucleationGrowthModel(
        nucleus_density=4e-3, time=3.3, crystal_orientations=True, cell_size=0.5
    )
    micro = model.generate(domain, seed=4)
    assert isinstance(micro, LatticeField) and micro.has_labels
    labels = micro.cell_labels
    n = micro.summary()["n_grains"]
    assert np.array_equal(np.unique(labels[labels >= 0]), np.arange(n))
    assert np.array_equal(labels >= 0, micro.phases == 1)
    assert micro.orientations.shape == (n, 3)
    assert len(micro.attributes["nucleation_times"]) == n
    # the level set is the growth distance minus the radius reached
    assert np.array_equal(micro.level_set().data < 0, micro.phases == 1)
    fine = micro.voxelize(shape=72)
    assert abs(fine.data.mean() - micro.phases.mean()) < 0.02
    assert abs(micro.phases.mean() - model.theory(domain)["volume_fraction"]) < 0.08
    io.save_microstructure(micro, tmp_path / "g.json")
    back = io.load_microstructure(tmp_path / "g.json")
    assert np.array_equal(back.cell_labels, labels) and back.model == model
    two = NucleationGrowthModel(nucleus_density=4e-3, phase_fractions=(0.4, 0.6))
    poly = two.generate(domain, seed=1)
    grains = poly.cell_labels
    assert np.all(grains >= 0)
    for g in range(poly.summary()["n_grains"]):  # one phase per grain
        assert np.unique(poly.phases[grains == g]).size == 1
    window = model.generate(mg.Domain.cube(24.0, periodic=False), seed=4)
    assert window.shape == (48, 48, 48) and 0.1 < window.phases.mean() < 0.9


def test_grain_growth_after_transformation():
    domain = mg.Domain.cube(32.0)
    base = NucleationGrowthModel(nucleus_density=0.02)
    coarse = NucleationGrowthModel(nucleus_density=0.02, coarsening_steps=40)
    n0 = base.generate(domain, seed=2).summary()["n_grains"]
    micro = coarse.generate(domain, seed=2)
    hist = micro.attributes["coarsening"]
    assert micro.summary()["n_grains"] < 0.7 * n0
    assert hist["step"][-1] == 40 and hist["n_grains"][-1] <= hist["n_grains"][0]
    assert coarse.expected_count(domain) is None and "grain_density" not in coarse.theory()


def test_potts_coarsening(rng):
    n = 32
    cells = rng.integers(0, n, (300, 3))
    labels, _, _ = grow_grains((n, n, n), 1.0, cells, np.zeros(300))

    def boundary(lab):
        return sum(np.count_nonzero(lab != np.roll(lab, 1, axis=a)) for a in range(3))

    energies = [boundary(labels)]
    out = potts_coarsening(
        labels, 10, temperature=0.0, seed=3, callback=lambda k, lab: energies.append(boundary(lab))
    )
    assert np.all(np.diff(energies) <= 0)  # zero temperature never raises the energy
    assert np.array_equal(out, potts_coarsening(labels, 10, temperature=0.0, seed=3))
    hot = potts_coarsening(labels, 10, temperature=1.0, seed=3)
    assert not np.array_equal(hot, out)
    assert set(np.unique(out)) <= set(np.unique(labels))
    with pytest.raises(ValueError, match="fully labelled"):
        potts_coarsening(np.full((4, 4, 4), -1), 1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "nucleus_density"),
        ({"nucleation": "continuous", "nucleus_density": 1e-3}, "nucleation_rate"),
        ({"nucleation": "instant", "nucleus_density": 1e-3}, "nucleation"),
        ({"nucleus_density": 1e-3, "growth_shape": "prism"}, "growth_shape"),
        ({"nucleus_density": 1e-3, "time": 2.0, "coarsening_steps": 5}, "complete"),
        ({"nucleus_density": 1e-3, "phase_fractions": (0.5, 0.6)}, "phase_fractions"),
        ({"nucleus_density": -1.0}, "nucleus_density"),
    ],
)
def test_nucleation_growth_checks(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NucleationGrowthModel(**kwargs)


# -- aggregation ---------------------------------------------------------------------------


def _mass_dimension(occupied, centre, r_min, dims):
    pts = np.argwhere(occupied)[:, :dims] - centre[:dims]
    r = np.sqrt((pts**2).sum(axis=1))
    rg = math.sqrt(np.mean(r**2))
    radii = np.geomspace(r_min, rg, 8)
    mass = [np.count_nonzero(r <= x) for x in radii]
    return np.polyfit(np.log(radii), np.log(mass), 1)[0]


def test_dla_fractal_dimension():
    """Witten-Sander clusters: D = 1.71 in 2D (Meakin 1983)."""
    shape = (192, 192, 1)
    occ, owner, order = aggregate(shape, np.random.default_rng(1), n_particles=2500, dims=2)
    assert occ.sum() == 2501 and order.max() == 2500
    assert np.all(owner[occ == 1] == 0)
    d = _mass_dimension(occ, np.array([96, 96, 0]), 4.0, 2)
    assert 1.55 < d < 1.85
    eden, _, _ = aggregate(
        shape, np.random.default_rng(1), process="eden", n_particles=2500, dims=2
    )
    assert 1.85 < _mass_dimension(eden, np.array([96, 96, 0]), 4.0, 2) < 2.1
    dense, _, _ = aggregate(
        shape, np.random.default_rng(1), n_particles=2500, dims=2, sticking=0.05
    )
    pts = np.argwhere(dense)[:, :2] - 96
    loose = np.argwhere(occ)[:, :2] - 96
    assert np.mean((pts**2).sum(1)) < np.mean((loose**2).sum(1))  # reaction-limited: denser


def test_dla_in_three_dimensions():
    occ, _, _ = aggregate((64, 64, 64), np.random.default_rng(2), n_particles=6000)
    assert 2.2 < _mass_dimension(occ, np.array([32, 32, 32]), 3.0, 3) < 2.8


def test_deposits():
    for process in ("dla", "eden", "ballistic"):
        occ, owner, _ = aggregate(
            (40, 40, 40), np.random.default_rng(3), process=process, geometry="deposit"
        )
        assert occ[:, :, -1].any() and occ[:, :, 0].any()  # from the substrate to the top
        assert np.all(owner[occ == 1] >= 0)
        if process == "ballistic":
            assert 0.2 < occ.mean() < 0.4
    flat, _, _ = aggregate(
        (128, 64, 1), np.random.default_rng(3), process="ballistic", geometry="deposit", dims=2
    )
    assert flat[:, -1, 0].any() and 0.4 < flat.mean() < 0.52  # 2D ballistic deposition: ~0.46
    with pytest.raises(ValueError, match="process"):
        aggregate((8, 8, 8), np.random.default_rng(0), process="dlca", n_particles=1)
    with pytest.raises(ValueError, match="particles"):
        aggregate((8, 8, 8), np.random.default_rng(0))


def test_aggregation_model(tmp_path):
    domain = mg.Domain.cube(32.0)
    model = AggregationModel(n_seeds=6, volume_fraction=0.08)
    micro = model.generate(domain, seed=1)
    assert micro.phases.mean() == pytest.approx(0.08, abs=1e-4)
    assert micro.summary()["n_clusters"] == 6
    assert np.array_equal(micro.cell_labels >= 0, micro.phases == 1)
    assert model.theory()["fractal_dimension"] == 2.5
    io.save_microstructure(micro, tmp_path / "a.json")
    assert np.array_equal(io.load_microstructure(tmp_path / "a.json").phases, micro.phases)
    window = AggregationModel(n_seeds=60, process="eden").generate(
        mg.Domain.cube(32.0, periodic=False), seed=1
    )
    assert window.shape == (32, 32, 32) and abs(window.phases.mean() - 0.1) < 0.03
    film = AggregationModel(geometry="deposit", process="ballistic")
    deposit = film.generate(mg.Domain(size=(32.0, 32.0, 16.0), periodic=False), seed=2)
    assert deposit.summary()["n_trees"] > 10 and deposit.phases[:, :, -1].any()
    with pytest.raises(ValueError, match="non-periodic"):
        film.generate(domain, seed=1)
    with pytest.raises(ValueError, match="sticking"):
        AggregationModel(sticking=0.0)
    with pytest.raises(ValueError, match="geometry"):
        AggregationModel(geometry="film")
