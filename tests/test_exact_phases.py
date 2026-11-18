"""The implicit functions of the phases at any point and the exact meshes of partitions:
one interface for the functions on grids and at points (``phase_level_sets_at``, the same
values at the centres of the voxels), the routes of the structures of several phases,
and the exact meshes that took the background grid before: several phases, the phases
kept by a list, the sharp edges of smooth pieces, the 10-node elements curved by
``curved_mesh`` (docs/theory.md §75, §76, §80, §94)."""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

import morphogen as mg
from morphogen.domain import Grid
from morphogen.mesh.conforming import ImplicitBody, conforming_mesh, has_mmg
from morphogen.mesh.exact import adaptive_mesh, exact_mesh, exact_route
from morphogen.models import model_from_config
from morphogen.models.implicit import band_of, functions_at, point_functions
from morphogen.models.nested import CoatedStructure
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Spheres

needs_mmg = pytest.mark.skipif(not has_mmg(), reason="MMG (pymmg) is not installed")

CUBE = mg.Domain.cube(1.0, periodic=True)


@functools.cache
def _partitions() -> dict:
    """Structures of several phases (or of other phases than 0 and 1)."""
    out = {
        "leaves": mg.models.DeadLeavesModel(radius=0.15, phase_fractions=[0.3, 0.3, 0.4]),
        "cells_layers": {"type": "voronoi", "n_cells": 12, "phase_fractions": [0.3, 0.3, 0.4],
                         "boundary_thickness": 0.04},
        "laguerre_layers": {"type": "laguerre", "n_cells": 12, "boundary_thickness": 0.04},
        "cells": {"type": "voronoi", "n_cells": 12, "phase_fractions": [0.5, 0.5]},
        "coating": {"type": "coating", "thickness": 0.04,
                    "model": {"type": "boolean_spheres", "radius": 0.12, "volume_fraction": 0.15}},
        "shell_core": {"type": "coating", "thickness": 0.04, "side": "inside", "core_phase": 3,
                       "model": {"type": "boolean_spheres", "radius": 0.12,
                                 "volume_fraction": 0.15}},
        "hollow": {"type": "coating", "thickness": 0.03, "side": "inside",
                   "model": {"type": "strut_lattice", "cell": "bcc", "cell_size": 0.5,
                             "strut_radius": 0.06}},
        "inclusions": {"type": "combination", "operation": "inclusions", "models": [
            {"type": "dead_leaves", "radius": 0.2, "phase_fractions": [0.5, 0.5]},
            {"type": "boolean_spheres", "radius": 0.08, "volume_fraction": 0.1},
            {"type": "boolean_spheres", "radius": 0.06, "volume_fraction": 0.05}]},
        "nested": {"type": "combination", "operation": "nested", "models": [
            {"type": "tpms", "surface": "gyroid", "cell_size": 0.5, "volume_fraction": 0.5},
            {"type": "dead_leaves", "radius": 0.1, "phase_fractions": [0.3, 0.3, 0.4]}]},
        "aggregates": {"type": "cox_spheres", "radius": 0.04, "local_fraction": 0.2,
                       "aggregate_phase": 2,
                       "aggregates": {"type": "boolean_spheres", "radius": 0.2,
                                      "volume_fraction": 0.3}},
        "mapped": {"type": "voronoi", "n_cells": 12, "boundary_thickness": 0.04,
                   "anisotropy": {"stretch": [1.3, 1.0, 1.0]}},
    }  # fmt: skip
    models = {k: (v if not isinstance(v, dict) else model_from_config(v)) for k, v in out.items()}
    structures = {k: m.generate(CUBE, seed=i + 1) for i, (k, m) in enumerate(models.items())}
    structures["osteons"] = mg.models.Osteons().generate(
        mg.Domain(size=(12.0, 12.0, 1.0), periodic=True), 1
    )
    structures["tiling"] = model_from_config(
        {"type": "cellular_automaton", "lattice": "hexagonal", "cell_size": 0.1,
         "rule": "conserved", "volume_fraction": 0.5}
    ).generate(mg.Domain(size=(1.0, 1.0, 0.25), periodic=True), seed=3)  # fmt: skip
    return structures


NAMES = ["aggregates", "cells", "cells_layers", "coating", "hollow", "inclusions",
         "laguerre_layers", "leaves", "mapped", "nested", "osteons", "shell_core",
         "tiling"]  # fmt: skip


@pytest.mark.parametrize("name", NAMES)
def test_functions_at_points_are_those_of_the_grids(name):
    """Every phase of the structures of several phases has its function at any point
    (``phase_level_sets_at``): at the centres of the voxels of a grid, within its band,
    those of ``phase_level_set`` (computed in single precision: 10⁻⁶ of the band), of the
    signs of the phases of the image; ``implicit_at`` and ``level_set_at`` are one phase
    of it."""
    micro = _partitions()[name]
    assert sorted(_partitions()) == NAMES and point_functions(micro)
    shape = (40, 40, 2) if name == "osteons" else (20, 20, 20)
    grid = Grid(micro.domain, shape)
    points = grid.center_points()
    band = band_of(grid)
    values = micro.phase_level_sets_at(points, band)
    phases = micro.implicit_phases()
    assert sorted(values) == phases and len(phases) >= 2
    image = np.asarray(micro.voxelize(grid=grid).data).ravel()
    for k in phases:
        sampled = np.asarray(micro.phase_level_set(k, grid=grid).data, dtype=float).ravel()
        at = np.clip(values[k], -band, band)
        near = np.abs(sampled) < 0.5 * band
        assert np.allclose(at[near], sampled[near], atol=1e-6 * band), k
        assert np.array_equal(at < 0, sampled < 0), k
        assert np.array_equal(at < 0, image == k), k
        assert np.array_equal(micro.implicit_at(k, points, band), at)
    if 1 in phases:  # the function of phase 1, within the band
        at = np.clip(micro.level_set_at(points, band), -band, band)
        near = np.abs(values[1]) < 0.5 * band
        assert np.allclose(at[near], values[1][near], atol=1e-12)
        assert np.array_equal(at < 0, values[1] < 0)


def test_structures_without_functions_at_points():
    """The structures whose functions are those of grids only, and the phases that a
    structure does not have: no point-wise function (``implicit_at`` is None), no exact
    route."""
    transparent = mg.models.TransparentDeadLeavesModel(radius=0.2, volume_fraction=0.3)
    transparent = transparent.generate(CUBE, 1)
    gyroid = {"type": "tpms", "surface": "gyroid", "cell_size": 0.5, "volume_fraction": 0.3}
    coated = model_from_config({"type": "coating", "model": gyroid, "thickness": 0.05})
    coated = coated.generate(CUBE, 1)  # the distance to a gyroid: that of its voxels
    one = mg.models.Voronoi(n_cells=8).generate(CUBE, 1)  # one phase: no interface
    body = ImplicitBody(CUBE, lambda p: np.linalg.norm(p - 0.5, axis=1) - 0.3)
    points = np.random.default_rng(0).random((10, 3))
    for micro in (transparent, coated, one):
        assert not point_functions(micro) and micro.implicit_at(1, points, 0.1) is None
        with pytest.raises(NotImplementedError):
            micro.phase_level_sets_at(points)
    assert exact_route(transparent) == exact_route(coated) == exact_route(body) == ""
    assert exact_route(one) == "cells" and exact_route(one, grains=False) == ""
    # a coated gyroid keeps the function of its phase 1, the gyroid's
    assert np.allclose(coated.level_set_at(points, 0.1), coated.inner.level_set_at(points, 0.1))
    sphere = PrimitiveUnion(CUBE, Spheres(np.full((1, 3), 0.5), np.array([0.3])))
    assert sphere.implicit_at(2, points, 0.1) is None  # not one of its phases

    # an object with a level set at points only: its two phases
    class Ball:
        domain = CUBE

        def level_set_at(self, p, reach=np.inf):
            return np.linalg.norm(np.asarray(p) - 0.5, axis=1) - 0.3

    assert point_functions(Ball()) and exact_route(Ball()) == "level_set"
    found = functions_at(Ball(), points, 1.0)
    assert np.array_equal(found[0], -found[1])


def test_routes_of_partitions():
    """The structures of several phases take the level-set route: their functions at any
    point; the tessellations by their grains keep their cells."""
    structures = _partitions()
    for name, micro in structures.items():
        expected = "cells" if name == "cells" else "level_set"
        assert exact_route(micro) == expected, name
    assert exact_route(structures["cells"], grains=False) == "level_set"


# -- the exact meshes -----------------------------------------------------------------------


def _coated_sphere(size: float = 3.4):
    """A sphere of radius 1 with a coating of 0.3 in a periodic cube: three phases."""
    domain = mg.Domain.cube(size, periodic=True)
    inner = PrimitiveUnion(domain, Spheres(np.array([[1.6, 1.75, 1.7]]), np.array([1.0])), None)
    micro = CoatedStructure(domain, inner, 0.3, "outside", 2, 0)
    volumes = {1: 4.0 / 3.0 * math.pi, 2: 4.0 / 3.0 * math.pi * (1.3**3 - 1.0)}
    return micro, volumes


@needs_mmg
def test_three_phases_on_the_exact_route():
    """A sphere and its coating: by default from the exact functions of the three phases,
    the faces matching; the 10-node elements, curved by ``curved_mesh`` with the
    functions of the structure, keep the volumes of the sphere and of the coating within
    10⁻³; the phases kept by a list are those of the whole mesh."""
    micro, exact = _coated_sphere()
    mesh = conforming_mesh(micro, size=0.6, order=2)
    assert mesh.metadata["mesher"].startswith("adaptive level set")
    assert mesh.metadata["level_sets"] == [1, 2] and set(np.unique(mesh.phase)) == {0, 1, 2}
    assert mesh.periodic and mesh.matching_faces() and mesh.min_jacobian() >= 0.3
    curved = mesh.metadata["curved"]
    assert curved["sampling"] == 0.0 and curved["nodes"] > 0.95 * curved["interface_nodes"]
    volumes = mesh.volumes()
    for k in (1, 2):
        assert volumes[mesh.phase == k].sum() == pytest.approx(exact[k], rel=1e-3)
    assert volumes.sum() == pytest.approx(3.4**3, rel=1e-12)
    kept = exact_mesh(micro, size=0.6, phases=[1, 2], order=2)
    assert set(np.unique(kept.phase)) == {1, 2} and kept.metadata["phases"] == [1, 2]
    assert kept.volumes()[kept.phase == 2].sum() == pytest.approx(exact[2], rel=1e-3)
    with pytest.raises(ValueError, match="phases"):
        conforming_mesh(micro, size=0.6, phases="solid")
    with pytest.raises(ValueError, match="not phases"):
        conforming_mesh(micro, size=0.6, phases=[1, 5], method="exact")


def _two_spheres(periodic: bool):
    """Two spheres (radii 1 and 0.9, centres 1.245 apart) that overlap: their union, its
    volume, and the distances of points to the circle where they meet."""
    domain = mg.Domain.cube(3.0, periodic=periodic)
    step = 1.245 * np.array([0.6, 0.8, 0.0])
    first = np.array([0.2, 1.4, 1.45]) if periodic else np.full(3, 1.5) - 0.5 * step
    centres = np.array([first, first + step])
    micro = PrimitiveUnion(domain, Spheres(centres, np.array([1.0, 0.9])), None)
    r1, r2, d = 1.0, 0.9, 1.245
    lens = math.pi * (r1 + r2 - d) ** 2 * (d * d + 2 * d * (r1 + r2) - 3 * (r1 - r2) ** 2)
    volume = 4 * math.pi / 3 * (r1**3 + r2**3) - lens / (12 * d)
    return micro, centres, volume


@needs_mmg
def test_sharp_edges_on_the_exact_route():
    """Two overlapping spheres: by default the exact mesh follows the groove where they
    meet, its sharp edges between the two pieces, their nodes on both spheres to
    round-off even with 4-node elements; the union loses the volume of the chords of
    4-node elements (less than 2·10⁻²) and within 5·10⁻⁴ with curved 10-node ones."""
    micro, centres, volume = _two_spheres(False)
    mesh = conforming_mesh(micro, size=0.45)
    info = mesh.metadata["sharp_edges"]
    assert mesh.metadata["mesher"].startswith("adaptive level set")
    assert info["pieces"] == 2 and info["edges"] == len(mesh.sharp) > 30
    assert info["depth"] == pytest.approx(0.45 / 20)
    assert np.all(mesh.sharp_pieces == [0, 1])
    corners = np.unique(mesh.sharp)
    off = np.linalg.norm(mesh.nodes[corners][:, None, :] - centres[None], axis=2) - [1.0, 0.9]
    assert np.abs(off).max() < 1e-12
    assert np.all(np.bincount(mesh.sharp.ravel())[corners] == 2)  # the circle, once
    assert -2e-2 < mesh.volumes()[mesh.phase == 1].sum() / volume - 1.0 < 0.0
    from morphogen.mesh.curved import curved_mesh

    curved = curved_mesh(mesh, micro)
    assert curved.metadata["curved"]["sharp_nodes"] > 2 * len(corners) - 2
    assert curved.volumes()[curved.phase == 1].sum() == pytest.approx(volume, rel=5e-4)
    # rounded on request
    rounded = conforming_mesh(micro, size=0.45, sharp_edges=False)
    assert rounded.sharp is None and "sharp_edges" not in rounded.metadata
    with pytest.raises(ValueError, match="smooth pieces"):
        adaptive_mesh(mg.models.TPMS(surface="gyroid", cell_size=1.0, volume_fraction=0.3)
                      .generate(CUBE), size=0.5, sharp_edges=True)  # fmt: skip


@pytest.mark.slow
@needs_mmg
def test_sharp_edges_through_the_faces_of_a_periodic_box_on_the_exact_route():
    """Two spheres that overlap through a face of a periodic box: the faces match, the
    nodes of the sharp edges lie on both spheres (their images too), and the curved
    elements keep the volume of the union within 3·10⁻⁴."""
    micro, centres, volume = _two_spheres(True)
    mesh = conforming_mesh(micro, size=0.3, order=2)
    assert mesh.periodic and mesh.matching_faces() and len(mesh.sharp) > 40
    assert mesh.volumes()[mesh.phase == 1].sum() == pytest.approx(volume, rel=3e-4)
    corners = mesh.nodes[np.unique(mesh.sharp)]
    gaps = corners[:, None, :] - centres[None]
    gaps -= 3.0 * np.round(gaps / 3.0)
    off = np.linalg.norm(gaps, axis=2) - [1.0, 0.9]
    assert np.abs(off).max() < 1e-9


@pytest.mark.slow
@needs_mmg
def test_partitions_on_the_exact_route():
    """Dead leaves of three phases, cells of two phases with their grain boundaries, a
    tiling of hexagons, in periodic boxes: every phase cut from its exact function (more
    than two phases: cut at once and remeshed once), the faces matching, at the fractions of fine
    images (the leaves within 0.005, the layers of the grain boundaries, a share of the
    elements thick, within 0.02, the prisms of the tiling within 0.002: its faces, cut
    exactly, do not match, and its two phases are cut at once as several are)."""
    cube = mg.Domain.cube(1.0, periodic=True)
    leaves = mg.models.DeadLeavesModel(radius=0.3, phase_fractions=[0.3, 0.3, 0.4])
    layers = model_from_config({"type": "voronoi", "n_cells": 8, "phase_fractions": [0.5, 0.5],
                                "boundary_thickness": 0.1})  # fmt: skip
    tiling = model_from_config(
        {"type": "cellular_automaton", "lattice": "hexagonal", "cell_size": 0.2,
         "rule": "conserved", "volume_fraction": 0.5}
    )  # fmt: skip
    cases = (
        (leaves.generate(cube, 2), dict(size=0.35, hausdorff=0.035), 0.005),
        (layers.generate(cube, 1), dict(size=0.25, hausdorff=0.025), 0.02),
        (tiling.generate(mg.Domain(size=(1.0, 1.0, 0.25), periodic=True), 3), dict(size=0.1),
         2e-3),
    )  # fmt: skip
    for micro, options, within in cases:
        name = type(micro).__name__
        mesh = conforming_mesh(micro, **options)
        assert mesh.metadata["mesher"].startswith("adaptive level set"), name
        phases = micro.implicit_phases()
        assert mesh.matching_faces() and (mesh.metadata.get("cut_once") or len(phases) == 2)
        assert set(np.unique(mesh.phase)) == set(phases), name
        shape = tuple(max(2, round(128 * length)) for length in micro.domain.lengths)
        image = np.asarray(micro.voxelize(shape=shape).data)
        fractions = mesh.phase_fractions()
        for k in phases:
            assert fractions[k] == pytest.approx(np.mean(image == k), abs=within), name
