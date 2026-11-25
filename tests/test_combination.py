"""Combinations of models (set operations, inclusions in a polycrystal) and the
grain-boundary phase of tessellations."""

import json

import numpy as np
import pytest

import morphogen as mg
from morphogen import pipeline
from morphogen.config import ConfigError, load_config
from morphogen.homogenization import Material, homogenize
from morphogen.models.base import Microstructure, model_from_config
from morphogen.seeding import derive_seed

SPHERES = {"type": "boolean_spheres", "radius": 0.06, "volume_fraction": 0.2}
FIBRES = {"type": "boolean_cylinders", "radius": 0.03, "length": 0.3, "volume_fraction": 0.15}
FIELD = {"type": "gaussian_excursion", "correlation_length": 0.1, "volume_fraction": 0.5}
CUBE = mg.Domain.cube(1.0, periodic=True)


def combination(operation, *models, **extra):
    return model_from_config(
        {"type": "combination", "operation": operation, "models": list(models), **extra}
    )


def test_set_operations():
    expected = {
        "union": lambda a, b: a | b,
        "intersection": lambda a, b: a & b,
        "difference": lambda a, b: a & ~b,
        "complement": lambda a, b: ~a,
    }
    for operation, rule in expected.items():
        models = [SPHERES] if operation == "complement" else [SPHERES, FIELD]
        micro = combination(operation, *models).generate(CUBE, seed=2)
        img = micro.voxelize((32, 32, 32))
        a = micro.parts[0].voxelize(grid=img.grid()).data == 1
        b = micro.parts[-1].voxelize(grid=img.grid()).data == 1
        assert np.array_equal(img.data == 1, rule(a, b)), operation
        # the level sets combine: same phase as the image at the voxel centres
        assert np.array_equal(micro.level_set(grid=img.grid()).data < 0, img.data == 1)
        fine = micro.voxelize((16, 16, 16), supersampling=2)
        assert fine.data.dtype == np.float32 and 0 < fine.data.mean() < 1
    # independent components: the seeds differ, and the realization is reproducible
    twice = combination("union", SPHERES, SPHERES)
    one, again = twice.generate(CUBE, seed=5), twice.generate(CUBE, seed=5)
    assert not np.array_equal(one.parts[0].spheres.centers, one.parts[1].spheres.centers)
    assert np.array_equal(one.parts[1].spheres.centers, again.parts[1].spheres.centers)
    # point queries and round trips
    micro = combination("difference", SPHERES, FIBRES).generate(CUBE, seed=1)
    img = micro.voxelize((24, 24, 24))
    inside = micro.contains(img.grid().center_points(0, 24)).reshape(img.shape)
    assert np.array_equal(inside, img.data == 1)
    back = Microstructure.from_dict(json.loads(json.dumps(micro.to_dict())))
    assert np.array_equal(back.voxelize((24, 24, 24)).data, img.data)
    assert back.model.to_config()["models"][1]["type"] == "boolean_cylinders"
    assert set(micro.summary()) == {"c0_n_spheres", "c1_n_segments"}


def test_theory_of_set_operations():
    """Union of two Boolean models of spheres: a Boolean model of both; its fraction
    and specific surface follow from those of the components."""
    model = combination("union", SPHERES, {**SPHERES, "radius": 0.04, "volume_fraction": 0.1})
    t = model.theory(CUBE)
    assert t["volume_fraction"] == pytest.approx(1 - 0.8 * 0.9)
    s1 = -0.8 * np.log(0.8) * 3 / 0.06  # Boolean spheres: S = -(1 - p) ln(1 - p) 3 / r
    s2 = -0.9 * np.log(0.9) * 3 / 0.04
    assert t["specific_surface"] == pytest.approx(s1 * 0.9 + s2 * 0.8)
    fractions = [
        (model.generate(CUBE, seed=s).voxelize((64, 64, 64)).data == 1).mean() for s in range(3)
    ]
    assert np.mean(fractions) == pytest.approx(t["volume_fraction"], abs=0.02)
    inter = combination("intersection", SPHERES, FIELD).theory(CUBE)
    assert inter["volume_fraction"] == pytest.approx(0.1)
    assert combination("complement", SPHERES).theory(CUBE)["volume_fraction"] == pytest.approx(0.8)
    assert combination("inclusions", SPHERES, FIBRES).theory(CUBE) == {}


def test_model_errors():
    cases = [
        ({"operation": "xor", "models": [SPHERES, FIELD]}, "operation"),
        ({"operation": "complement", "models": [SPHERES, FIELD]}, "exactly one"),
        ({"operation": "union", "models": [SPHERES]}, "at least two"),
        ({"operation": "union", "models": SPHERES}, "list of models"),
        (
            {"operation": "inclusions", "models": [SPHERES, FIELD], "inclusion_phases": [1, 2]},
            "one",
        ),
        ({"operation": "union", "models": [SPHERES, FIELD], "inclusion_phases": [2]}, "only"),
        ({"operation": "union", "models": [SPHERES, {"type": "boolean_spheres"}]}, "boolean"),
    ]
    for params, message in cases:
        with pytest.raises(ValueError, match=message):
            model_from_config({"type": "combination", **params})
    fibres = combination("union", SPHERES, {"type": "poisson_fibers", "radius": 0.02,
                                             "volume_fraction": 0.1})  # fmt: skip
    assert not fibres.supports_periodic
    with pytest.raises(ValueError, match="periodic"):
        fibres.generate(CUBE, seed=1)


def brute_boundary_distance(tess, points):
    """Distances to the planes of every pair of cells (images of the germs included)."""
    shifts = np.array(list(np.ndindex(3, 3, 3)), dtype=float) - 1.0
    g = np.concatenate([tess.germs + s for s in shifts])
    w = np.tile(tess.weights if tess.weights is not None else np.zeros(tess.n_cells), 27)
    germ = np.tile(np.arange(tess.n_cells), 27)
    power = ((points[:, None, :] - g[None]) ** 2).sum(-1) - w[None]
    own = np.argmin(power, axis=1)
    gap = np.linalg.norm(g[None] - g[own][:, None], axis=2)
    other = germ[None, :] != germ[own][:, None]
    d = (power - power[np.arange(len(points)), own][:, None]) / np.where(other, 2 * gap, 1.0)
    return np.where(other, d, np.inf).min(axis=1)


def test_grain_boundary_phase():
    rng = np.random.default_rng(0)
    points = rng.random((500, 3))
    for model in (
        mg.models.Voronoi(n_cells=30, boundary_thickness=0.03),
        mg.models.Laguerre(
            n_cells=30, radius=mg.distributions.Uniform(0.05, 0.15), boundary_thickness=0.03
        ),
    ):
        tess = model.generate(CUBE, seed=3)
        exact = brute_boundary_distance(tess, points)
        assert np.allclose(tess.boundary_distance(points), exact, rtol=0, atol=1e-12)
        near = tess.boundary_distance(points, reach=0.01)
        assert np.allclose(near[exact < 0.01], exact[exact < 0.01], rtol=0, atol=1e-12)
        assert np.all(near[exact >= 0.01] >= 0.01 - 1e-12)
        assert model.active_phases() == {1: "cells", 0: "grain boundaries"}
        assert model.has_level_set and not model.meshed_by_grains
        img = tess.voxelize((40, 40, 40))
        centres = img.grid().center_points(0, 40)
        d = tess.boundary_distance(centres).reshape(img.shape)
        assert np.array_equal(img.data == 0, d < 0.015)
        assert np.array_equal(tess.level_set(grid=img.grid()).data < 0, img.data == 1)
        assert np.array_equal(tess.contains(centres).reshape(img.shape), img.data == 1)
        labels = tess.labels(grid=img.grid()).data
        assert np.unique(labels[img.data == 0]).size > 1  # the boundaries keep their cells
    # the fraction of the boundaries is S_V t to first order in t
    voronoi = mg.models.Voronoi(n_cells=50, boundary_thickness=0.01)
    first = voronoi.theory(CUBE)["boundary_fraction_first_order"]
    measured = (voronoi.generate(CUBE, seed=1).voxelize((96, 96, 96)).data == 0).mean()
    assert 0.85 * first < measured < first
    # multiphase cells: a new phase for the boundaries, no level set
    mosaic = mg.models.Voronoi(n_cells=10, phase_fractions=(0.5, 0.5), boundary_thickness=0.05)
    assert mosaic.grain_boundary_phase == 2 and not mosaic.has_level_set
    assert set(np.unique(mosaic.generate(CUBE, seed=2).voxelize((24, 24, 24)).data)) == {0, 1, 2}
    back = Microstructure.from_dict(json.loads(json.dumps(tess.to_dict())))
    assert back.boundary_thickness == 0.03 and np.array_equal(
        back.voxelize((40, 40, 40)).data, img.data
    )
    with pytest.raises(ValueError, match="grain-boundary"):
        mg.mesh.conforming.conforming_mesh(tess, size=0.2, grains=True)


COPPER = {"cubic": [168.4, 121.4, 75.4]}
K_COPPER = (168.4 + 2 * 121.4) / 3


def test_inclusions_in_a_polycrystal(tmp_path):
    """Inclusions and grain boundaries of the bulk modulus of the copper crystals: the
    bulk modulus of the aggregate is exactly theirs; the grains keep their labels and
    orientations, so that the run gives the same moduli as the solver on the image."""
    poly = {"type": "voronoi", "n_cells": 12, "crystal_orientations": True,
            "boundary_thickness": 0.06}  # fmt: skip
    model = combination("inclusions", poly, {**SPHERES, "radius": 0.08, "volume_fraction": 0.1})
    phases = {1: "cells", 0: "grain boundaries", 2: "inclusions (boolean_spheres)"}
    assert model.active_phases() == phases
    assert model.phases_of_inclusions == [2] and not model.has_level_set
    micro = model.generate(CUBE, seed=3)
    img = micro.voxelize((16, 16, 16))
    assert set(np.unique(img.data)) == {0, 1, 2} and micro.has_labels
    assert micro.orientations is micro.matrix.orientations
    grain_boundary = {"young": 9 * K_COPPER * 40 / (3 * K_COPPER + 40), "poisson": 0.0}
    grain_boundary["poisson"] = (3 * K_COPPER - 2 * 40) / (2 * (3 * K_COPPER + 40))
    data = {
        "model": model.to_config(),
        "domain": {"size": 1.0, "periodic": True},
        "seed": 3,
        "voxels": {"shape": 16},
        "homogenization": {  # the FFT solver, compared with its run on the image below
            "method": "fft",
            "materials": {0: grain_boundary, 1: COPPER, 2: grain_boundary},
            "properties": ["bulk_modulus", "shear_modulus"],
        },
        "outputs": {"directory": str(tmp_path / "out"), "name": "c", "labels": ["vti"]},
    }
    rows = pipeline.read_rows(pipeline.run(load_config(data)) / "results.csv")
    assert rows[0]["bulk_modulus.periodic"] == pytest.approx(K_COPPER, rel=1e-8)
    micro = load_config(data).model.generate(CUBE, derive_seed(3, 0))
    img = micro.voxelize((16, 16, 16))
    labels = micro.labels(grid=img.grid())
    materials = {p: Material.from_config(m) for p, m in data["homogenization"]["materials"].items()}
    direct = homogenize(img, materials, grains=labels, orientations=micro.orientations)
    shear = direct["shear_modulus.periodic"].value
    assert rows[0]["shear_modulus.periodic"] == pytest.approx(shear, rel=1e-6)
    assert 40 < shear < 54.7
    missing = {**data, "homogenization": {**data["homogenization"], "materials": {1: COPPER}}}
    with pytest.raises(ConfigError, match=r"phase\(s\) \[0, 2\]"):
        load_config(missing)
    for key in ("damask", "neper"):
        bad = {**data, "outputs": {**data["outputs"], key: True}}
        with pytest.raises(ConfigError, match="tessellations"):
            load_config(bad)
