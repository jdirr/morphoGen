"""Measures of realizations on their implicit geometry (morphology.measures): the probes
answered by the functions of the phases, against the closed forms of Boolean spheres, of a
single ball and of the excursions of a Gaussian field, without a voxel image."""

from __future__ import annotations

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.models.union import PrimitiveUnion
from morphogen.morphology import (
    capacity,
    chords,
    compare_with_theory,
    contact_distribution,
    covariance,
    crossings,
    dilation,
    first_hits,
    minkowski_functionals,
    phases_at,
    sample_points,
    specific_surface,
    volume_fraction,
)
from morphogen.primitives import Spheres


@pytest.fixture(scope="module")
def boolean():
    model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
    return model, model.generate(mg.Domain.cube(12.0), seed=1)


def _ball():
    domain = mg.Domain.cube(4.0)
    return PrimitiveUnion(domain, spheres=Spheres(np.array([[2.0, 2.0, 2.0]]), np.array([1.0])))


def test_rays_and_crossings_of_one_ball():
    u = _ball()
    # a ray from outside along x: the sphere at 1, crossed at 1 and 3
    hit = first_hits(u, [[0.0, 2.0, 2.0]], [[1.0, 0.0, 0.0]], 4.0)
    assert hit[0] == pytest.approx(1.0, abs=1e-6)
    inside, found = crossings(u, [[0.0, 2.0, 2.0]], [[1.0, 0.0, 0.0]], 3.9)
    assert not inside[0]
    assert found[0] == pytest.approx([1.0, 3.0], abs=1e-8)
    # a grazing chord: at the height 0.99 the chord is 2 sqrt(1 - 0.99²)
    _, found = crossings(u, [[0.0, 2.99, 2.0]], [[1.0, 0.0, 0.0]], 3.9)
    assert found[0][1] - found[0][0] == pytest.approx(2 * math.sqrt(1 - 0.99**2), abs=1e-8)
    assert first_hits(u, [[0.0, 3.5, 2.0]], [[1.0, 0.0, 0.0]], 3.9)[0] == math.inf


def test_crofton_and_chords_of_one_ball():
    u = _ball()
    est = specific_surface(u, lines=4096, seed=0)
    exact = 4 * math.pi / 64.0
    assert abs(est.value - exact) < 4 * est.error
    ch = chords(u, lines=4096, seed=1)
    assert ch.mean == pytest.approx(4.0 / 3.0, rel=0.03)  # 4 V / S of a ball


def test_boolean_spheres_probes_against_theory(boolean):
    model, micro = boolean
    exact = minkowski_functionals(micro)  # the realization itself
    vf = volume_fraction(micro, points=2**16)
    assert abs(vf.value - exact.volume_fraction) < 4 * vf.error
    lags = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    c = covariance(micro, lags, pairs=2**15)
    assert np.all(np.abs(c.values - model.covariance(lags)) < 4 * c.errors + 0.012)
    sizes = np.array([0.0, 0.25, 0.5, 1.0])
    t = capacity(micro, "ball", sizes)
    assert np.all(np.abs(t.values - model.choquet_capacity("ball", sizes)) < 4 * t.errors + 0.012)
    lengths = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    t = capacity(micro, "segment", lengths)
    expected = model.choquet_capacity("segment", lengths)
    assert np.all(np.abs(t.values - expected) < 4 * t.errors + 0.012)
    # the pores' chords: exponential of mean 4 q / (θ q S̄) = 1 / (θ π R²)
    ch = chords(micro, phase=0, lines=256)
    assert ch.mean == pytest.approx(1.0 / (model.theta * math.pi), rel=0.08)
    h = contact_distribution(micro, [0.5, 1.0], "spherical")
    q = 1.0 - model.theory()["volume_fraction"]
    expected = 1.0 - (1.0 - model.choquet_capacity("ball", [0.5, 1.0])) / q
    assert np.all(np.abs(h.values - expected) < 4 * h.errors + 0.02)


def test_comparison_with_theory_is_exact_for_spheres(boolean):
    _, micro = boolean
    cmp = compare_with_theory(micro)
    assert cmp.exact
    rows = cmp.as_dict()
    assert set(rows) == {
        "volume_fraction",
        "specific_surface",
        "mean_curvature_density",
        "euler_density",
    }
    assert rows["volume_fraction"]["measured"] == pytest.approx(
        minkowski_functionals(micro).volume_fraction
    )
    assert "covariance" in cmp.curves and "capacity_ball" in cmp.curves
    assert "measure" in cmp.table()


def test_gaussian_excursion_by_probes():
    model = mg.models.GaussianExcursion(volume_fraction=0.4, correlation_length=1.0)
    micro = model.generate(mg.Domain.cube(16.0), seed=2)
    lags = np.array([0.0, 0.5, 1.0, 2.0])
    c = covariance(micro, lags, pairs=2**14)
    assert np.all(np.abs(c.values - model.covariance(lags)) < 4 * c.errors + 0.01)
    sv = specific_surface(micro, lines=192, seed=3)
    assert abs(sv.value - model.theory()["specific_surface"]) < 4 * sv.error + 0.02


def test_dilation_is_exact_for_unions(boolean):
    model, micro = boolean
    grown = dilation(micro, 0.5)
    x = sample_points(micro.domain, 4096, seed=5)
    d = micro.level_set_at(x, 2.0)
    assert np.array_equal(grown.contains(x), d <= 0.5)
    # the exact volume fraction of the dilation is the capacity of balls of the realization
    expected = model.choquet_capacity("ball", [0.5])[0]
    assert minkowski_functionals(grown).volume_fraction == pytest.approx(expected, abs=0.03)


def test_phases_of_points_and_minus_sampling():
    model = mg.models.Voronoi(n_cells=20, phase_fractions=(0.5, 0.5))
    micro = model.generate(mg.Domain.cube(8.0, periodic=False), seed=4)
    x = sample_points(micro.domain, 2000, seed=0)
    image = micro.voxelize(32)
    assert set(np.unique(phases_at(micro, x))) <= {0, 1}
    vf = volume_fraction(micro, points=2**14)
    assert vf.value == pytest.approx(float(image.data.mean()), abs=0.03)
    with pytest.raises(ValueError, match="fit in the box"):
        covariance(micro, [0.0, 9.0], direction=(1, 0, 0))
    with pytest.raises(ValueError, match="union of spheres"):
        capacity(micro, "ball", [0.1])


def test_named_exact_measures(tmp_path, capsys):
    """The named measures of the implicit geometry: their columns (the Minkowski densities of
    a union of spheres exactly, against the estimates of the probes), in a run without voxels
    (``analysis``), from the command line on a geometry, and their errors."""
    import csv
    import json

    from morphogen import cli, pipeline
    from morphogen.config import ConfigError, load_config
    from morphogen.morphology.measures import EXACT_MEASURES, measure_exact

    model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
    micro = model.generate(mg.Domain.cube(8.0, periodic=True), seed=1)
    out = measure_exact(micro, sorted(set(EXACT_MEASURES) - {"exact_grains"}))  # of cells
    assert out["exact_V_V"] == pytest.approx(
        out["exact_fraction"], abs=4 * out["exact_fraction_std"]
    )
    assert out["exact_S_V"] == pytest.approx(out["exact_specific_surface"],
                                             abs=4 * out["exact_specific_surface_std"])  # fmt: skip
    assert out["exact_mean_chord_1"] > 0 and out["exact_mean_chord_0"] > 0
    with pytest.raises(ValueError, match="exact measure"):
        measure_exact(micro, ["volume_fraction"])
    voronoi = mg.models.Voronoi(n_cells=6).generate(mg.Domain.cube(2.0), seed=1)
    with pytest.raises(ValueError, match="union of spheres"):
        measure_exact(voronoi, ["exact_minkowski"])
    with pytest.raises(ValueError, match="tessellation"):
        measure_exact(micro, ["exact_grains"])
    assert measure_exact(voronoi, ["exact_grains"])["grain_fraction_1"] == pytest.approx(1.0)
    cfg = {
        "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
        "domain": {"size": 8.0, "periodic": True},
        "seed": 1,
        "analysis": ["exact_fraction", "exact_minkowski"],
        "outputs": {"directory": str(tmp_path / "run"), "name": "spheres", "geometry": ["json"]},
    }  # no voxels: the measures of the implicit geometry need none
    outdir = pipeline.run(load_config(cfg))
    with open(outdir / "results.csv", encoding="utf-8") as f:
        (row,) = list(csv.DictReader(f))
    assert float(row["exact_V_V"]) == pytest.approx(float(row["exact_fraction"]), abs=0.01)
    with pytest.raises(ConfigError, match="voxels"):  # a measure of the voxels still needs them
        load_config({**cfg, "analysis": ["exact_fraction", "euler"]})
    geometry = str(outdir / "spheres_0000.json")
    assert cli.main(["analyze", geometry, "--measure", "exact_minkowski", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["exact_V_V"] == pytest.approx(float(row["exact_V_V"]))
    assert shown["theory_volume_fraction"] == pytest.approx(0.3)
    with pytest.raises(SystemExit, match="measures of an image"):
        cli.main(["analyze", geometry, "--measure", "euler"])
