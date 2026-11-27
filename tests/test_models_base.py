"""Model registry, configuration round trips and serialization of realizations."""

import json

import numpy as np
import pytest

import morphogen as mg
from morphogen.models import Microstructure, get_model, list_models, model_from_config

CONFIGS = {
    "boolean_spheres": (
        {"radius": {"type": "lognormal", "mean": 1.0, "std": 0.2}, "volume_fraction": 0.2},
        True,
    ),
    "boolean_cylinders": (
        {"radius": 0.3, "length": 3.0, "orientation": "isotropic", "volume_fraction": 0.1},
        True,
    ),
    "poisson_fibers": ({"radius": 0.5, "volume_fraction": 0.1}, False),
    "hardcore_spheres": ({"radius": 0.8, "volume_fraction": 0.2}, True),
    "hardcore_fibers": ({"radius": 0.3, "length": 2.0, "volume_fraction": 0.1}, True),
    "curved_fibers": (
        {"radius": 0.3, "length": 4.0, "persistence": 3.0, "segments": 6, "volume_fraction": 0.1},
        True,
    ),
    "random_walk_fibers": (
        {
            "radius": 0.3,
            "length": 2.0,
            "orientation": {"type": "beta", "beta": 0.5},
            "volume_fraction": 0.2,
        },
        True,
    ),
    "cross_linked_fibers": (
        {"radius": 0.3, "length": 3.0, "volume_fraction": 0.1, "link_radius": 0.45},
        True,
    ),
    "long_fibers": (
        {
            "radius": {"type": "uniform", "low": 0.3, "high": 0.4},
            "volume_fraction": 0.5,
            "misalignment": 3.0,
        },
        False,
    ),
    "voronoi": ({"n_cells": 30, "phase_fractions": [0.5, 0.5], "crystal_orientations": True}, True),
    "laguerre": (
        {"intensity": 0.05, "radius": {"type": "uniform", "low": 0.5, "high": 1.0}},
        False,
    ),
    "gaussian_excursion": ({"correlation_length": 1.0, "volume_fraction": 0.4}, True),
    "tpms": (
        {"surface": "gyroid", "cell_size": 4.0, "variant": "sheet", "volume_fraction": 0.2},
        True,
    ),
    "strut_lattice": ({"cell": "bcc", "cell_size": 4.0, "strut_radius": 0.4}, True),
    "cellular_automaton": ({"rule": "conserved", "volume_fraction": 0.4, "steps": 5}, True),
    "activator_inhibitor": (
        {"activation_radius": 1.5, "inhibition_radius": 3.0, "steps": 4},
        False,
    ),
    "reaction_diffusion": ({"pattern": "labyrinth", "steps": 1500, "cell_size": 0.5}, False),
    "nucleation_growth": (
        {"nucleation": "continuous", "nucleation_rate": 0.01, "time": 2.0, "cell_size": 0.5},
        True,
    ),
    "aggregation": ({"n_seeds": 4, "volume_fraction": 0.1}, True),
    "tile_assembly": ({"tiles": "pipes", "tile_length": 2.0, "degrees": [0, 2, 3]}, True),
    "random_lattice": (
        {
            "cell": "bcc",
            "cell_size": 4.0,
            "strut_radius": 0.4,
            "removal_fraction": 0.2,
            "perturbation": 0.1,
        },
        True,
    ),
    "random_laminate": (
        {
            "thickness": {"type": "lognormal", "mean": 0.6, "std": 0.2},
            "spacing": 1.0,
            "hole_fraction": 0.2,
            "hole_radius": 0.5,
        },
        True,
    ),
    "penetrable_spheres": (
        {"radius": 0.8, "impenetrability": 0.5, "reduced_density": 0.3},
        True,
    ),
    "reconstruction": (
        {"correlation_length": 0.6, "volume_fraction": 0.3, "cell_size": 0.5, "sweeps": 5},
        True,
    ),
    "transparent_dead_leaves": (
        {"radius": 1.2, "transparency": 0.6, "volume_fraction": 0.4},
        True,
    ),
    "dead_leaves": (
        {"radius": {"type": "uniform", "low": 0.8, "high": 1.5}, "phase_fractions": [0.3, 0.7]},
        True,
    ),
    "combination": (
        {
            "operation": "inclusions",
            "models": [
                {"type": "voronoi", "n_cells": 20, "boundary_thickness": 0.3},
                {"type": "boolean_spheres", "radius": 0.8, "volume_fraction": 0.1},
            ],
        },
        True,
    ),
    "nested_grains": (
        {
            "grains": {"type": "voronoi", "n_cells": 12, "crystal_orientations": True},
            "inner": {"type": "voronoi", "n_cells": 120},
            "misorientation": {"type": "uniform", "low": 1.0, "high": 3.0},
        },
        True,
    ),
    "cox_spheres": (
        {
            "aggregates": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
            "radius": 0.2,
            "local_fraction": 0.3,
            "aggregate_phase": 2,
        },
        True,
    ),
    "coating": (
        {
            "model": {"type": "hardcore_spheres", "radius": 0.8, "volume_fraction": 0.15},
            "thickness": 0.2,
            "side": "inside",
        },
        True,
    ),
    "hierarchical_lattice": (
        {"cell_size": 4.0, "generations": 1, "cells": 2, "tube_radius": 0.06},
        True,
    ),
    "fractal_sponge": ({"cell_size": 4.0, "level": 2}, True),
    "nacre": ({"tablet_size": 2.0, "tablet_thickness": 0.6, "mortar_thickness": 0.15}, True),
    "bouligand": (
        {"fiber_radius": 0.3, "layer_thickness": 2 / 3, "pitch_angle": 30.0},
        True,
    ),
    "honeycomb": ({"cell_size": 2.0, "wall_thickness": 0.3, "arrangement": "voronoi"}, True),
    "osteons": (
        {"radius": {"type": "uniform", "low": 0.8, "high": 1.2}, "cement_thickness": 0.1},
        False,
    ),
    "open_foam": ({"n_cells": 20, "strut_radius": 0.25, "relaxation": 2}, True),
    "delaunay_network": ({"n_points": 40, "radius": 0.15, "regularity": 0.5}, True),
    "level_set_foam": (
        {"seed_radius": 1.0, "thickness": 0.3, "curvature": -0.5, "c1": 20.0, "c2": 1.0},
        True,
    ),
    "branching": ({"n_attractors": 300, "step": 0.4, "tip_radius": 0.1}, False),
    # (the demonstration map that comes with morphoGen)
    "ebsd": ({"file": "ebsd_map.ctf", "window": "random"}, False),
    # (the volume that comes with morphoGen)
    "image": ({"file": "example_image.npz", "window": "random"}, False),
    "scale_rich_network": ({"n_ligaments": 80, "decay": 0.5, "volume_fraction": 0.3}, False),
    "woven_textile": (
        {
            "weave": "twill",
            "pattern": [2, 2],
            "layers": 2,
            "yarn_thickness": {"type": "uniform", "low": 0.3, "high": 0.4},
            "random_shift": True,
        },
        True,
    ),
    # a shape of Microgen (3MAH), when the package is installed (OPTIONAL)
    "microgen": (
        {
            "shape": "Tpms",
            "arguments": {"surface_function": "gyroid", "offset": 0.6, "cell_size": 4.0},
            "part": "sheet",
        },
        True,
    ),
}
#: The models that need a package that morphoGen does not require, by the package
OPTIONAL = {"microgen": "microgen"}


def needs(name: str) -> None:
    """Skip the test of a model whose optional package is missing."""
    if name in OPTIONAL:
        pytest.importorskip(OPTIONAL[name])


def test_registry_lists_all_models():
    assert set(CONFIGS) == set(list_models())
    with pytest.raises(ValueError, match="available models"):
        get_model("unknown")


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_config_and_json_round_trips(name):
    needs(name)
    params, periodic = CONFIGS[name]
    model = model_from_config({"type": name, **params})
    assert model_from_config(model.to_config()) == model
    assert json.loads(json.dumps(model.to_config())) == model.to_config()
    for p in type(model).parameters():
        assert {"name", "kind", "default", "required", "unit", "doc"} <= set(p) and p["doc"]
    micro = model.generate(mg.Domain.cube(8.0, periodic=periodic), seed=3)
    data = json.loads(json.dumps(micro.to_dict()))
    back = Microstructure.from_dict(data)
    assert back.model == model and back.seed == micro.seed
    a, b = micro.voxelize(shape=24), back.voxelize(shape=24)
    assert np.array_equal(a.data, b.data)
    assert a.metadata["model"]["type"] == name and a.metadata["phases"]
    # same seed, same realization; another seed, another one
    again = model.generate(mg.Domain.cube(8.0, periodic=periodic), seed=3).voxelize(shape=24)
    assert np.array_equal(again.data, a.data)


def test_configuration_errors():
    with pytest.raises(ValueError, match="unknown parameters"):
        model_from_config(
            {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.2, "colour": "red"}
        )
    with pytest.raises(ValueError, match=r"model\.radius"):
        model_from_config({"type": "boolean_spheres", "radius": "big", "volume_fraction": 0.2})
    with pytest.raises(ValueError, match=r"model\.n_cells"):
        model_from_config({"type": "voronoi", "n_cells": 2.5})
    with pytest.raises(ValueError, match="'type'"):
        model_from_config({"radius": 1.0})


def test_point_queries_agree_with_voxels():
    for model in (
        mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3),
        mg.models.Voronoi(n_cells=20, phase_fractions=(0.5, 0.5)),
    ):
        micro = model.generate(mg.Domain.cube(6.0), seed=1)
        img = micro.voxelize(shape=20)
        inside = micro.contains(img.grid().center_points()).reshape(img.shape)
        assert np.array_equal(inside, img.data == 1)
    union = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3).generate(
        mg.Domain.cube(6.0), seed=1
    )
    ls = union.level_set(shape=30, band=0.5)
    pts = ls.grid().center_points()
    np.testing.assert_allclose(
        union.signed_distance(pts, 0.5).reshape(ls.shape), ls.data, atol=1e-6
    )
    with pytest.raises(NotImplementedError):
        mg.models.Voronoi(n_cells=5).generate(mg.Domain.cube(1.0)).level_set(shape=4)
