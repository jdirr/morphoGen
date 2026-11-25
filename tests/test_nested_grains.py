"""Grains with subgrains, or filled with another structure (models.nested_grains): the
subgrains refine the grains and hold their germs, their misorientations and the angles
between two subgrains of one grain follow their laws, lamellae turn with the crystals of
their grains, inclusions fill the grains at the fraction of their model, and the
structures are periodic, serializable and homogenized with the crystals of their
subgrains."""

import math

import numpy as np
import pytest
from scipy import stats

import morphogen as mg
from morphogen import texture
from morphogen.domain import Grid
from morphogen.homogenization import Material, homogenize
from morphogen.mesh import conforming_mesh, exact_route
from morphogen.models import NestedGrains, NestedGrainStructure, model_from_config
from morphogen.models.base import Microstructure
from morphogen.models.nested_grains import _turns
from morphogen.models.scale_rich_3d import polyhedron_volume
from morphogen.orientations import euler_to_matrix, uniform_unit_vectors

PERIODIC = mg.Domain.cube(1.0, periodic=True)
GRAINS = {"type": "voronoi", "n_cells": 12, "crystal_orientations": True}
# copper (GPa)
CU = Material.cubic(168.4, 121.4, 75.4)


def subgrains(**kwargs) -> NestedGrains:
    params = {"grains": GRAINS, "inner": {"type": "voronoi", "n_cells": 240}}
    params.update(kwargs)
    return NestedGrains(**params)


def _brute_labels(micro: NestedGrainStructure, points) -> np.ndarray:
    """The subgrain of each point from the germs of the subgrains in the axes of the
    sample: the nearest germ of its grain (an isotropic Voronoi inner tessellation, turned
    by rotations, which keep the distances)."""
    grains = micro.grains.cell_of(points)
    germs = micro.germs
    out = np.empty(len(points), dtype=np.int64)
    lengths = micro.domain.lengths
    for i, (p, g) in enumerate(zip(points, grains, strict=True)):
        k0, k1 = micro.first[g], micro.first[g + 1]
        d = germs[k0:k1] - p
        d -= lengths * np.round(d / lengths)
        out[i] = k0 + int(np.argmin(np.einsum("ij,ij->i", d, d)))
    return out


@pytest.mark.parametrize("turned", [True, False])
def test_subgrains_refine_the_grains_and_hold_their_germs(turned):
    model = subgrains(turned=turned)
    micro = model.generate(PERIODIC, seed=3)
    labels = micro.labels(40)
    grains = micro.grains.labels(grid=labels.grid()).data
    assert labels.data.dtype == np.int32
    assert np.array_equal(micro.parents[labels.data], grains)  # nested in their grains
    assert np.array_equal(micro.grain_labels(grid=labels.grid()).data, grains)
    assert np.array_equal(micro.cell_of(micro.germs), np.arange(micro.n_cells))
    # each point in the subgrain of the nearest germ of its grain
    points = np.random.default_rng(1).random((400, 3))
    assert np.array_equal(micro.cell_of(points), _brute_labels(micro, points))
    # the germs: those of the inner tessellation in each grain (all of them without turns,
    # each once; the places the turned grains see hold as many on average), the grains
    # without one a subgrain of their own
    alone = int(np.sum(np.diff(micro.first) == 1))
    if turned:
        assert abs(micro.n_cells - alone - 240) < 4.0 * math.sqrt(240)
    else:
        assert micro.n_cells == 240 + alone
    assert model.theory(PERIODIC)["subgrain_germs_per_grain"] == pytest.approx(20.0)
    summary = micro.summary()
    assert summary["n_grains"] == 12 and summary["n_subgrains"] == micro.n_cells
    # the phases are those of the grains
    assert np.array_equal(micro.voxelize(grid=labels.grid()).data,
                          micro.grains.voxelize(grid=labels.grid()).data)  # fmt: skip
    assert micro.implicit_phases() == micro.grains.implicit_phases()


def test_subgrains_of_anisotropic_laths_and_of_windows():
    """Laths: an inner tessellation stretched along the axis c of the crystal of each
    grain; anisotropic grains of a window; the labels still refine the grains."""
    laths = subgrains(inner={"type": "voronoi", "n_cells": 200, "anisotropy": [1, 1, 3]})
    micro = laths.generate(PERIODIC, seed=2)
    lab = micro.labels(32)
    assert np.array_equal(micro.parents[lab.data], micro.grains.labels(grid=lab.grid()).data)
    window = mg.Domain.cube(1.0, periodic=False)
    pancakes = subgrains(grains={**GRAINS, "anisotropy": [2, 2, 0.5]}, turned=False)
    micro = pancakes.generate(window, seed=4)
    lab = micro.labels(32)
    assert np.array_equal(micro.parents[lab.data], micro.grains.labels(grid=lab.grid()).data)
    present = np.unique(lab.data)
    assert len(present) > len(np.unique(micro.parents[present]))  # grains with subgrains


def test_misorientations_follow_their_law():
    law = {"type": "uniform", "low": 1.0, "high": 4.0}
    micro = subgrains(misorientation=law).generate(PERIODIC, seed=5)
    parent = micro.grains.orientations[micro.parents]
    angles = texture.disorientation(micro.orientations, parent, "cubic")
    np.testing.assert_allclose(angles, micro.misorientations, atol=1e-6)
    assert stats.kstest(micro.misorientations, stats.uniform(1.0, 3.0).cdf).pvalue > 1e-3
    # a constant angle, or none
    micro = subgrains(misorientation=2.5).generate(PERIODIC, seed=5)
    parent = micro.grains.orientations[micro.parents]
    np.testing.assert_allclose(texture.disorientation(micro.orientations, parent), 2.5,
                               atol=1e-6)  # fmt: skip
    micro = subgrains(misorientation=None).generate(PERIODIC, seed=5)
    np.testing.assert_allclose(micro.orientations, micro.grains.orientations[micro.parents])


def test_the_angle_between_two_subgrains_of_a_grain():
    """cos(psi / 2) uniform between cos((ta + tb) / 2) and cos((ta - tb) / 2): the law
    against the products of random rotations, and the boundaries between the subgrains of
    the grains of an image."""
    model = subgrains(misorientation={"type": "uniform", "low": 1.0, "high": 5.0})
    rng = np.random.default_rng(0)
    n = 4000
    turns = [
        _turns(uniform_unit_vectors(rng, n), np.radians(model.misorientation.sample(rng, n)))
        for _ in range(2)
    ]
    r = turns[1] @ np.swapaxes(turns[0], 1, 2)
    psi = np.degrees(np.arccos(np.clip((np.trace(r, axis1=1, axis2=2) - 1.0) / 2.0, -1, 1)))
    assert stats.kstest(psi, model.subgrain_disorientation_cdf).pvalue > 1e-3
    mean, square = model.subgrain_disorientation_moments()
    sigma = math.sqrt(square - mean**2)
    assert abs(psi.mean() - mean) < 4.0 * sigma / math.sqrt(n)
    # a constant angle t: psi / 2t of density 2x (small angles), mean 4t / 3
    assert subgrains(misorientation=3.0).subgrain_disorientation_moments()[0] == pytest.approx(
        4.0, rel=1e-3
    )
    # neighbouring subgrains of a grain in an image (pairs that share subgrains: a tenth
    # of them taken as independent)
    micro = model.generate(PERIODIC, seed=7)
    pairs, _, angles = texture.boundary_disorientations(micro.labels(40).data,
                                                        micro.orientations)  # fmt: skip
    same = micro.parents[pairs[:, 0]] == micro.parents[pairs[:, 1]]
    within = angles[same]
    assert within.size > 300
    assert abs(within.mean() - mean) < 4.0 * sigma / math.sqrt(within.size / 10)
    assert within.max() < 10.0 + 1e-9  # at most the sum of two misorientations
    assert np.median(angles[~same]) > 20.0  # the boundaries of the grains


def test_turned_lamellae_lie_on_planes_of_the_crystals():
    """The layers of a laminate normal to the axis z of the inner model lie, in each
    grain, normal to the axis c of its crystal (the third row of g): the principal axis of
    the gradients of the image, inside the grains (wrapped ones included)."""
    model = NestedGrains(
        grains={"type": "voronoi", "n_cells": 10, "crystal_orientations": True},
        inner={"type": "random_laminate", "thickness": 0.04, "spacing": 0.04},
    )
    micro = model.generate(PERIODIC, seed=4)
    assert model.active_phases() == {1: "cells", 2: "random_laminate in the grains"}
    n = 72
    phases, labels = micro.voxelize(n).data, micro.labels(n).data
    assert set(np.unique(phases)) == {1, 2}
    f = (phases == 2).astype(float)
    gradient = np.stack([np.roll(f, -1, a) - np.roll(f, 1, a) for a in range(3)], axis=-1)
    checked = 0
    for k in range(micro.n_grains):
        inside = labels == k
        core = inside.copy()  # away from the boundaries of the grain (periodic erosion)
        for a in range(3):
            for s in (-2, -1, 1, 2):
                core &= np.roll(inside, s, axis=a)
        if core.sum() < 2000:
            continue
        v = gradient[core]
        normal = np.linalg.eigh(v.T @ v)[1][:, -1]
        axis_c = euler_to_matrix(micro.grains.orientations[k])[0][2]
        assert abs(normal @ axis_c) > 0.97, k
        checked += 1
    assert checked >= 6


def test_inclusions_fill_the_grains_of_their_phases():
    model = NestedGrains(
        grains={"type": "voronoi", "n_cells": 30, "phase_fractions": [0.5, 0.5]},
        inner={"type": "boolean_spheres", "radius": 0.03, "volume_fraction": 0.2},
        filled_phases=[1],
    )
    theory = model.theory(PERIODIC)
    assert theory["volume_fraction_2"] == pytest.approx(0.1)
    assert theory["volume_fraction_1"] == pytest.approx(0.4)
    assert theory["volume_fraction_0"] == pytest.approx(0.5)
    fractions = []
    grid = PERIODIC.grid(80)
    for seed in range(3):
        micro = model.generate(PERIODIC, seed=seed)
        phases = micro.voxelize(grid=grid).data
        grains = micro.grains.voxelize(grid=grid).data
        assert np.array_equal(phases == 0, grains == 0)  # only the grains of phase 1
        assert np.all(grains[phases == 2] == 1)
        fractions.append(float(np.mean(phases[grains == 1] == 2)))
    # each point of a filled grain sees a point of the realization uniform in its box
    spheres = model.inner.theory(PERIODIC)
    sigma = math.sqrt(0.2 * 0.8 * spheres["integral_range"] / 0.5) / math.sqrt(3)
    assert abs(np.mean(fractions) - 0.2) < 4.0 * sigma


def test_unturned_grains_cut_one_realization():
    model = NestedGrains(
        grains={"type": "voronoi", "n_cells": 20, "phase_fractions": [0.4, 0.6]},
        inner={"type": "tpms", "surface": "gyroid", "cell_size": 0.25, "volume_fraction": 0.3},
        turned=False,
        filled_phases=[1],
        inner_phase=3,
    )
    micro = model.generate(PERIODIC, seed=2)
    grid = PERIODIC.grid(48)
    phases = micro.voxelize(grid=grid).data
    grains = micro.grains.voxelize(grid=grid).data
    inner = micro.inner.voxelize(grid=grid).data
    assert np.array_equal(phases, np.where((grains == 1) & (inner == 1), 3, grains))
    assert micro.implicit_phases() == [0, 1, 3]
    # the functions of the phases: their signs those of the image
    for k in (0, 1, 3):
        field = micro.phase_level_set(k, grid=grid).data
        assert np.array_equal(field < 0, phases == k)


def test_periodic_and_serialized():
    model = NestedGrains(
        grains={"type": "laguerre", "n_cells": 14, "crystal_orientations": True,
                "radius": {"type": "uniform", "low": 0.5, "high": 1.0}},
        inner={"type": "voronoi", "n_cells": 150},
        misorientation={"type": "lognormal", "mean": 3.0, "std": 1.0},
    )  # fmt: skip
    micro = model.generate(PERIODIC, seed=6)
    n = 36
    h = 1.0 / n
    labels = micro.labels(n).data
    shifted = Grid(mg.Domain(1.0, origin=(3 * h, 0.0, -5 * h), periodic=True), (n, n, n))
    assert np.array_equal(micro.labels(grid=shifted).data,
                          np.roll(np.roll(labels, -3, axis=0), 5, axis=2))  # fmt: skip
    # the configuration and the geometry, read back
    again = model_from_config(model.to_config())
    assert again == model
    assert np.array_equal(again.generate(PERIODIC, seed=6).labels(n).data, labels)
    copy = Microstructure.from_dict(micro.to_dict())
    assert isinstance(copy, NestedGrainStructure)
    assert np.array_equal(copy.labels(n).data, labels)
    np.testing.assert_allclose(copy.orientations, micro.orientations)
    filled = NestedGrains(
        grains=GRAINS, inner={"type": "hardcore_spheres", "radius": 0.05, "volume_fraction": 0.2}
    ).generate(PERIODIC, seed=1)
    image = filled.voxelize(n).data
    assert np.array_equal(Microstructure.from_dict(filled.to_dict()).voxelize(n).data, image)
    fraction = filled.voxelize(n, supersampling=2).data
    assert fraction.dtype == np.float32 and 0.0 <= fraction.min() and fraction.max() <= 1.0
    assert abs(float(fraction.mean()) - float(np.mean(image == 1))) < 0.02


def test_invalid_parameters():
    with pytest.raises(ValueError, match="Voronoi or Laguerre"):
        NestedGrains(grains={"type": "boolean_spheres", "volume_fraction": 0.2}, inner=GRAINS)
    with pytest.raises(ValueError, match="crystal_orientations"):
        subgrains(grains={"type": "voronoi", "n_cells": 12})
    with pytest.raises(ValueError, match="phase_fractions"):
        subgrains(inner={"type": "voronoi", "n_cells": 40, "phase_fractions": [0.5, 0.5]})
    with pytest.raises(ValueError, match="inner_phase"):
        subgrains(inner_phase=3)
    with pytest.raises(ValueError, match="90 degrees"):
        subgrains(misorientation=120.0)
    with pytest.raises(ValueError, match="filled_phases"):
        subgrains(filled_phases=[0])
    with pytest.raises(ValueError, match="periodic"):
        NestedGrains(grains=GRAINS, inner={"type": "poisson_fibers", "volume_fraction": 0.1})
    with pytest.raises(ValueError, match="missing"):
        model_from_config({"type": "nested_grains", "grains": GRAINS})


def test_fft_polycrystal_with_subgrains():
    """Copper: the bulk modulus of any polycrystal of cubic crystals is that of the
    crystal; subgrains not turned are their grains, and turned by a hundredth of a degree
    nearly so."""
    shape = 24
    res = {}
    for angle in (None, 0.01):
        micro = subgrains(inner={"type": "voronoi", "n_cells": 150},
                          misorientation=angle).generate(PERIODIC, seed=5)  # fmt: skip
        image = micro.voxelize(shape)
        found = homogenize(image, {1: CU}, grains=micro.labels(shape),
                           orientations=micro.orientations)  # fmt: skip
        assert found["bulk_modulus.periodic"].value == pytest.approx(137.0667, rel=1e-6)
        res[angle] = found["shear_modulus.periodic"].value
    grains = homogenize(image, {1: CU}, grains=micro.grain_labels(shape),
                        orientations=micro.grains.orientations)  # fmt: skip
    assert res[None] == pytest.approx(grains["shear_modulus.periodic"].value, rel=1e-9)
    assert res[0.01] == pytest.approx(res[None], rel=1e-4)
    assert res[0.01] != res[None]


def _pieces_holding(pieces, points) -> np.ndarray:
    """The first convex piece that holds each point (-1: none)."""
    owner = np.full(len(points), -1)
    for k, (_, _, _, normals, planes) in enumerate(pieces):
        inside = np.all(points @ normals.T <= planes + 1e-12, axis=1)
        owner[inside & (owner < 0)] = k
    return owner


WINDOW = mg.Domain.cube(1.0, periodic=False)
SUBGRAIN_CASES = {
    "turned": (subgrains(), PERIODIC),
    "unturned": (subgrains(turned=False), PERIODIC),
    "boundary phase": (
        subgrains(grains={**GRAINS, "boundary_thickness": 0.03, "phase_fractions": [0.0, 1.0]}),
        PERIODIC,
    ),
    "laths of a window": (
        subgrains(inner={"type": "voronoi", "n_cells": 200, "anisotropy": [1, 1, 3]},
                  turned=False),
        WINDOW,
    ),
    "turned anisotropic grains of a window": (
        subgrains(grains={**GRAINS, "anisotropy": {"stretch": [2, 2, 0.5], "euler": [30, 40, 0]}},
                  inner={"type": "laguerre", "n_cells": 200, "radius": 0.03}),
        WINDOW,
    ),
}  # fmt: skip


@pytest.mark.parametrize("case", list(SUBGRAIN_CASES))
def test_subgrains_are_exact_convex_pieces(case):
    """The exact cells of the subgrains: the pieces of the cells of the grains cut by the
    planes of equal power distance to two germs of the inner tessellation in the frame of
    each grain (and by the planes of the grain-boundary phase), which partition the box:
    each point in a piece of its subgrain and of its phase, the volume of each subgrain the
    fraction of the points in it (Monte Carlo)."""
    model, domain = SUBGRAIN_CASES[case]
    micro = model.generate(domain, seed=7)
    assert exact_route(micro) == "pieces"
    pieces, phases, labels = micro.convex_pieces()
    volumes = np.array([polyhedron_volume(p[0], p[1], p[3], p[4]) for p in pieces])
    box = float(np.prod(domain.lengths))
    assert volumes.sum() == pytest.approx(box, rel=1e-9) and volumes.min() > 0
    rng = np.random.default_rng(1)
    points = domain.lower + rng.random((6000, 3)) * domain.lengths
    owner = _pieces_holding(pieces, points)
    assert np.all(owner >= 0)
    assert np.array_equal(labels[owner], micro.cell_of(points))
    assert np.array_equal(phases[owner], micro.phases_at(points))
    expected = np.bincount(labels, weights=volumes, minlength=micro.n_cells) / box * len(points)
    counts = np.bincount(micro.cell_of(points), minlength=micro.n_cells)
    assert np.all(np.abs(counts - expected) <= 4.5 * np.sqrt(expected) + 2.0)  # Poisson
    if case == "boundary phase":  # the cells of phase 1 and the boundary phase
        assert set(np.unique(phases)) == {1, micro.cells.tessellation.boundary_phase}


def test_exact_mesh_of_subgrains_and_its_crystals():
    """The exact mesh of the subgrains, grain by grain: every element in the subgrain and
    the phase of its centre, the volume of each subgrain that of its pieces, the crystals
    of the subgrains, the opposite faces matching; the bulk modulus of its copper crystals,
    by finite elements, that of the crystal."""
    model = subgrains(inner={"type": "voronoi", "n_cells": 60})
    micro = model.generate(PERIODIC, seed=1)
    mesh = conforming_mesh(micro, size=0.2, method="exact")
    centres = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    assert np.array_equal(mesh.cell, micro.cell_of(centres))
    assert np.array_equal(mesh.phase, micro.phases_at(centres))
    assert np.array_equal(mesh.orientations, micro.orientations)
    pieces, _, labels = micro.convex_pieces()
    exact = np.zeros(micro.n_cells)
    np.add.at(exact, labels, [polyhedron_volume(p[0], p[1], p[3], p[4]) for p in pieces])
    p = mesh.nodes[mesh.elements[:, :4]]
    six = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    meshed = np.bincount(mesh.cell, weights=six / 6.0, minlength=micro.n_cells)
    assert meshed == pytest.approx(exact, abs=1e-12)
    assert mesh.matching_faces()
    found = homogenize(mesh, {1: CU}, properties=["bulk_modulus"])
    assert found["bulk_modulus.periodic"].value == pytest.approx(137.0667, rel=1e-6)


FILLED = NestedGrains(
    grains={"type": "voronoi", "n_cells": 10, "crystal_orientations": True,
            "phase_fractions": [0.5, 0.5]},
    inner={"type": "boolean_spheres", "radius": 0.1, "volume_fraction": 0.2},
    filled_phases=[1],
)  # fmt: skip


def test_filled_grains_have_functions_at_any_point():
    """Filled grains: the functions of their phases at any point (the union over the
    grains of the inner structure each sees, cut by its cell), whose signs are the phases
    and which are continuous across the faces of the grains, where the patterns change
    (their slopes within their bound); those of the grids are the same at the centres of
    the voxels; the exact route of the level sets."""
    from morphogen.models.implicit import point_functions

    micro = FILLED.generate(PERIODIC, seed=6)
    assert point_functions(micro) and exact_route(micro) == "level_set"
    rng = np.random.default_rng(2)
    points = rng.random((20000, 3))
    functions = micro.phase_level_sets_at(points, 0.2)
    assert sorted(functions) == micro.implicit_phases() == [0, 1, 2]
    phases = micro.phases_at(points)
    for k, f in functions.items():
        assert np.all((f < 0) == (phases == k))
    step = 1e-6 * uniform_unit_vectors(rng, len(points))
    moved = micro.phase_level_sets_at(points + step, 0.2)
    # no jump, across the faces of the grains as well: the inner phase within the bound of
    # its slopes; the others within those of the functions of the cells of the mosaic
    assert np.max(np.abs(moved[2] - functions[2])) <= 1.0001e-6 * micro.lipschitz
    for k in (0, 1):
        assert np.max(np.abs(moved[k] - functions[k])) <= 2.0001e-6
    grid = PERIODIC.grid(16)
    level = micro.phase_level_set(2, grid=grid).data
    at = micro.phase_level_sets_at(grid.center_points(), 4.0 / 16)[2].reshape(grid.shape)
    assert np.allclose(level, at, atol=1e-6)


@pytest.mark.slow
def test_exact_mesh_of_filled_grains():
    """Lamellar colonies (lamellae turned with the crystals of their grains) meshed by the
    functions of their phases: the phase of each element that of its centre (but a few
    along the interfaces and the edges where the lamellae meet the faces of the grains),
    the volumes of the phases those of a fine voxel image, each element with the grain of
    its centre and its crystal."""
    colonies = NestedGrains(
        grains={"type": "voronoi", "n_cells": 4, "crystal_orientations": True},
        inner={"type": "random_laminate", "thickness": 0.15, "spacing": 0.3},
    )
    micro = colonies.generate(PERIODIC, seed=6)
    mesh = conforming_mesh(micro, size=0.3, method="exact")
    centres = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    p = mesh.nodes[mesh.elements[:, :4]]
    volume = np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                       np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6.0  # fmt: skip
    same = micro.phases_at(centres) == mesh.phase
    assert volume[same].sum() > 0.99 * volume.sum()
    image = micro.voxelize(64).data
    assert micro.implicit_phases() == [1, 2]  # the grains, the lamellae
    for k in (1, 2):
        assert volume[mesh.phase == k].sum() == pytest.approx(np.mean(image == k), abs=0.01)
    assert np.array_equal(mesh.cell, micro.cell_of(centres))
    assert np.array_equal(mesh.orientations, micro.orientations)
