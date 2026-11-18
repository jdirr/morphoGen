"""The implicit geometry of every model: an implicit function for each of its phases whose
sign is that of the voxel images at any resolution and whose zero set is the interface of
the model (spheres, planes of cells, cubes of a sponge: against their exact positions), and
the conforming meshes of structures of several phases cut from them."""

import numpy as np
import pytest

import morphogen as mg
from morphogen.homogenization import Material, homogenize
from morphogen.mesh.conforming import (
    background_interpolation,
    background_mesh,
    conforming_mesh,
    has_mmg,
)
from morphogen.mesh.tetra import tet_volumes
from morphogen.models import model_from_config
from morphogen.models.deadleaves import DeadLeaves
from morphogen.models.nested import CoatedStructure, FractalStructure
from morphogen.models.tessellation import Tessellation
from morphogen.models.transparent import TransparentDeadLeaves
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Spheres
from test_models_base import CONFIGS

#: Configurations of more than two phases, or of other phases than 0 and 1, beside those
#: of the registry (name, parameters, periodic)
SEVERAL = [
    ("voronoi", {"n_cells": 30, "phase_fractions": [0.4, 0.3, 0.3]}, True),
    ("voronoi", {"n_cells": 30, "phase_fractions": [0.5, 0.5], "boundary_thickness": 0.4}, True),
    ("laguerre", {"intensity": 0.05, "radius": 0.8, "phase_fractions": [0.5, 0.5]}, False),
    ("dead_leaves", {"radius": 1.0, "phase_fractions": [0.3, 0.3, 0.4]}, True),
    ("dead_leaves", {"radius": 1.0, "shape": "spheroid", "aspect_ratio": 2.5}, True),
    ("dead_leaves", {"radius": 1.0, "coverage": 1.0}, False),
    (
        "dead_leaves",  # mapped by an anisotropy along the axes of the box
        {"radius": 1.0, "phase_fractions": [0.3, 0.3, 0.4], "anisotropy": [2.0, 1.0, 0.5]},
        True,
    ),
    ("voronoi", {"n_cells": 30, "phase_fractions": [0.5, 0.5], "anisotropy": 2.0}, True),
    ("transparent_dead_leaves", {"radius": 1.0, "volume_fraction": 0.5, "transparency": 1.0}, True),
    ("osteons", {"radius": 1.2, "canal_ratio": 0.25, "cement_thickness": 0.08, "axis": 0}, False),
    ("fractal_sponge", {"cell_size": 8.0, "level": 3, "pattern": "vicsek"}, False),
    (
        "coating",
        {"model": {"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.2},
         "thickness": 0.3},
        True,
    ),
    (
        "coating",
        {"model": {"type": "strut_lattice", "cell": "bcc", "cell_size": 4.0, "strut_radius": 0.5},
         "thickness": 0.2, "side": "inside", "core_phase": 2},
        True,
    ),
    (
        "coating",
        {"model": {"type": "tpms", "surface": "gyroid", "cell_size": 4.0, "volume_fraction": 0.3},
         "thickness": 0.3},
        True,
    ),
    (
        "cox_spheres",
        {"radius": 0.3, "local_fraction": 0.3, "aggregate_phase": 2,
         "aggregates": {"type": "boolean_spheres", "radius": 2.0, "volume_fraction": 0.4}},
        True,
    ),
    (
        "combination",
        {"operation": "inclusions",
         "models": [{"type": "voronoi", "n_cells": 20, "phase_fractions": [0.5, 0.5]},
                    {"type": "hardcore_spheres", "radius": 0.8, "volume_fraction": 0.15},
                    {"type": "boolean_cylinders", "radius": 0.3, "length": 3.0,
                     "volume_fraction": 0.05}]},
        True,
    ),
    (
        "combination",
        {"operation": "nested",
         "models": [{"type": "tpms", "surface": "gyroid", "cell_size": 8.0,
                     "volume_fraction": 0.5},
                    {"type": "strut_lattice", "cell": "bcc", "cell_size": 2.0,
                     "strut_radius": 0.3}]},
        True,
    ),
    (
        "cellular_automaton",
        {"rule": "conserved", "volume_fraction": 0.4, "steps": 3, "lattice": "fcc"},
        True,
    ),
    (
        "cellular_automaton",
        {"rule": "conserved", "volume_fraction": 0.4, "steps": 3, "lattice": "icosahedral"},
        False,
    ),
]  # fmt: skip
CASES = [(name, params, periodic) for name, (params, periodic) in CONFIGS.items()] + SEVERAL


@pytest.mark.parametrize(
    "name, params, periodic", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)]
)
def test_every_phase_of_every_model_has_an_implicit_function(name, params, periodic):
    """Its sign is that of the voxel image, on the grid of the model and on another (but
    at the points exactly on an interface), and the phases partition the points."""
    model = model_from_config({"type": name, **params})
    micro = model.generate(mg.Domain((8.0, 8.0, 8.0), periodic=periodic), seed=1)
    phases = micro.implicit_phases()
    assert phases == sorted(phases) and len(phases) >= 1
    for shape in ((37, 37, 37), (29, 29, 29)):
        image = np.asarray(micro.voxelize(shape).data)
        assert set(np.unique(image)) <= set(phases)
        inside = np.zeros(image.shape, dtype=int)
        ties = np.zeros(image.shape, dtype=bool)
        for k in phases:
            field = micro.phase_level_set(k, shape)
            assert field.shape == image.shape and np.all(np.isfinite(field.data))
            level = np.asarray(field.data, dtype=float)
            tie = np.abs(level) < 1e-6
            assert np.all(((level < 0) == (image == k)) | tie)
            inside += level < 0
            ties |= tie
        assert np.all((inside == 1) | ties) and ties.mean() < 0.05
    if phases == [0, 1]:  # one level set: that of the phase 1
        assert np.array_equal(micro.level_set(25).data, micro.phase_level_set(1, 25).data)
    with pytest.raises(ValueError, match="phase"):
        micro.phase_level_set(max(phases) + 7, 25)


def test_fractions_and_distances_of_a_sponge():
    """The Menger sponge on the grid of its smallest cubes: (20/27)^n exactly; its
    function is the distance in the maximum norm to the nearest face between a cube kept
    and a cube removed, within the edge of the smallest cubes."""
    model = model_from_config({"type": "fractal_sponge", "cell_size": 3.0, "level": 2})
    micro = model.generate(mg.Domain.cube(3.0), seed=0)
    assert isinstance(micro, FractalStructure) and model.has_level_set
    level = np.asarray(micro.level_set((27, 27, 27)).data, dtype=float)  # 3 voxels per cube
    assert (level < 0).mean() == pytest.approx((20 / 27) ** 2, abs=1e-12)
    # along x through the middle of a cube kept beside the hole of the first level (the
    # square 1 < y, z < 2): at 1/18, 1/6 and 5/18 from the face x = 1 of the hole
    h = 1.0 / 3.0  # the edge of the smallest cubes
    x = (np.arange(27) + 0.5) / 9.0
    line = level[:, 13, 13]  # y = z = 1.5: in the hole
    assert np.all(line > 0)
    assert line[13] == pytest.approx(h)  # the middle of the hole: farther than a cube
    row = level[:, 13, 7]  # z = 5/6: the first of the three cubes below the hole, y = 1.5
    assert x[9] == pytest.approx(1.0 + 1.0 / 18.0)
    # that cube is kept at the first level, its middle third removed at the second
    assert np.sign(row[9]) == 1.0 and abs(row[9]) == pytest.approx(1.0 / 18.0)


def test_cells_of_several_phases_against_their_planes():
    """Two cells of two phases: the interface is the plane halfway between their germs;
    with a boundary layer, the planes at half its thickness on each side."""
    domain = mg.Domain.cube(4.0, periodic=False)
    germs = [[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]]
    cells = Tessellation(domain, germs, cell_phases=[0, 1])
    assert cells.implicit_phases() == [0, 1]
    grid = cells.grid((40, 40, 40))  # a band of 0.4
    x = grid.centers(0)
    for phase, sign in ((1, -1.0), (0, 1.0)):
        level = np.asarray(cells.phase_level_set(phase, grid=grid).data, dtype=float)
        scale = 2.0 * (domain.volume / 2) ** (1 / 3)  # twice the mean spacing of the germs
        expected = sign * (x - 2.0) * 2.0 * 2.0 / scale  # the distance times the gap / spacing
        assert level[:, 20, 20] == pytest.approx(np.clip(expected, -0.4, 0.4), abs=1e-6)
    layered = Tessellation(domain, germs, cell_phases=[0, 1], boundary_thickness=0.6,
                           boundary_phase=2)  # fmt: skip
    assert layered.implicit_phases() == [0, 1, 2]
    layer = np.asarray(layered.phase_level_set(2, grid=grid).data, dtype=float)[:, 20, 20]
    assert layer == pytest.approx(np.clip(np.abs(x - 2.0) - 0.3, -0.4, 0.4), abs=1e-6)
    one = np.asarray(layered.phase_level_set(1, grid=grid).data, dtype=float)[:, 20, 20]
    assert np.all((one < 0) == (x > 2.3))
    single = Tessellation(domain, germs)  # cells of one phase: no interface
    assert single.implicit_phases() == [1]
    assert np.all(single.phase_level_set(1, grid=grid).data < 0)


def test_leaves_against_their_spheres():
    """Dead leaves: a leaf on top of another, the distances to their spheres (the hidden
    part of the lower one is not of its phase); a transparent leaf: the distance to its
    sphere, on the side that its value decides."""
    domain = mg.Domain.cube(8.0, periodic=False)
    leaves = DeadLeaves(domain, [[3.0, 4.0, 4.0], [5.0, 4.0, 4.0]], [1.5, 1.5], [1, 2],
                        background=0)  # fmt: skip
    assert leaves.implicit_phases() == [0, 1, 2]
    grid = leaves.grid((80, 80, 80))  # a band of 0.4
    x = grid.centers(0)
    y, z = grid.centers(1)[40] - 4.0, grid.centers(2)[40] - 4.0
    d1 = np.sqrt((x - 3.0) ** 2 + y * y + z * z) - 1.5
    d2 = np.sqrt((x - 5.0) ** 2 + y * y + z * z) - 1.5
    band = 0.4
    fields = {k: np.asarray(leaves.phase_level_set(k, grid=grid).data)[:, 40, 40] for k in range(3)}
    assert fields[1] == pytest.approx(np.clip(d1, -band, band), abs=1e-6)  # the one on top
    assert fields[2] == pytest.approx(np.clip(np.maximum(d2, -d1), -band, band), abs=1e-6)
    assert fields[0] == pytest.approx(np.clip(-np.minimum(d1, d2), -band, band), abs=1e-6)
    glass = TransparentDeadLeaves(domain, [[4.0, 4.0, 4.0]], [2.0], [1.0], alpha=0.5,
                                  threshold=0.6, mean_value=0.5)  # fmt: skip
    # in the leaf: 0.5 * 1 + 0.5 * 0.5 = 0.75 > 0.6; outside: the mean, 0.5
    level = np.asarray(glass.level_set(grid=grid).data)[:, 40, 40]
    d = np.sqrt((x - 4.0) ** 2 + y * y + z * z) - 2.0
    assert level == pytest.approx(np.clip(d, -band, band), abs=1e-6)
    dim = TransparentDeadLeaves(domain, [[4.0, 4.0, 4.0]], [2.0], [0.6], alpha=0.5,
                                threshold=0.6, mean_value=0.5)  # fmt: skip
    assert np.all(dim.level_set(grid=grid).data > 0)  # 0.55: below the threshold everywhere


def test_interpolation_on_the_background_mesh():
    """The function linear in each tetrahedron of the background mesh: the values of its
    nodes at the nodes, their barycentric means inside the tetrahedra."""
    lower, upper = np.zeros(3), np.array([2.0, 1.5, 2.5])
    nodes, tets, _, _ = background_mesh((4, 3, 5), lower, upper)
    rng = np.random.default_rng(0)
    values = rng.normal(size=(5, 4, 6))
    assert background_interpolation(values, lower, upper, nodes) == pytest.approx(values.ravel())
    weights = rng.dirichlet(np.ones(4), size=len(tets))
    points = np.einsum("tk,tkj->tj", weights, nodes[tets])
    expected = np.einsum("tk,tk->t", weights, values.ravel()[tets])
    found = background_interpolation(values, lower, upper, points)
    assert found == pytest.approx(expected, abs=1e-12)


@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_conforming_mesh_of_a_coated_sphere():
    """A sphere, its coating and the matrix: three phases cut one after the other from
    their implicit functions, a mesh conforming to both spheres, their volumes within
    the chords of its elements, its opposite faces matching; the phases kept."""
    domain = mg.Domain.cube(5.0)
    centre, radius, thickness = np.full((1, 3), 2.5), 1.2, 0.6
    inner = PrimitiveUnion(domain, spheres=Spheres(centre, np.array([radius])))
    micro = CoatedStructure(domain, inner, thickness, "outside", 2, 0)
    assert micro.implicit_phases() == [0, 1, 2]
    mesh = conforming_mesh(micro, size=0.6, method="grid")
    assert set(np.unique(mesh.phase)) == {0, 1, 2}
    assert mesh.periodic and mesh.matching_faces() and mesh.quality().min() > 0.1
    volumes = tet_volumes(mesh.nodes, mesh.elements)
    assert volumes.sum() == pytest.approx(125.0, rel=1e-10)
    sphere = 4.0 / 3.0 * np.pi * radius**3
    shell = 4.0 / 3.0 * np.pi * (radius + thickness) ** 3 - sphere
    assert volumes[mesh.phase == 1].sum() == pytest.approx(sphere, rel=0.07)
    assert volumes[mesh.phase == 2].sum() == pytest.approx(shell, rel=0.03)
    # each phase is connected to itself only through its own elements: the nodes of the
    # sphere and those of the matrix are apart, the coating between them
    of = [set(mesh.elements[mesh.phase == k].ravel()) for k in range(3)]
    assert not of[0] & of[1] and of[0] & of[2] and of[1] & of[2]
    # finite elements on the three phases: one material, its moduli; three, those of the
    # FFT solver on the voxels
    same = {k: Material(young=10.0, poisson=0.3) for k in range(3)}
    moduli = homogenize(mesh, same, properties=["bulk_modulus", "shear_modulus"])
    assert moduli["bulk_modulus.periodic"].value == pytest.approx(10.0 / 1.2, rel=1e-8)
    assert moduli["shear_modulus.periodic"].value == pytest.approx(10.0 / 2.6, rel=1e-8)
    three = {0: Material(young=1.0, poisson=0.3), 1: Material(young=50.0, poisson=0.25),
             2: Material(young=5.0, poisson=0.35)}  # fmt: skip
    fe = homogenize(mesh, three, properties=["bulk_modulus", "shear_modulus"])
    fft = homogenize(micro.voxelize((48, 48, 48)), three,
                     properties=["bulk_modulus", "shear_modulus"])  # fmt: skip
    for name in ("bulk_modulus.periodic", "shear_modulus.periodic"):
        assert fe[name].value == pytest.approx(fft[name].value, rel=0.02)
    kept = conforming_mesh(micro, size=0.6, phases=[1, 2], method="grid")
    assert set(np.unique(kept.phase)) == {1, 2}
    assert tet_volumes(kept.nodes, kept.elements).sum() == pytest.approx(
        volumes[mesh.phase != 0].sum(), rel=1e-9
    )
    with pytest.raises(ValueError, match="phases"):
        conforming_mesh(micro, size=0.6, phases="solid", method="grid")
    with pytest.raises(ValueError, match="not phases"):
        conforming_mesh(micro, size=0.6, phases=[1, 5], method="grid")


@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_cells_of_three_phases_are_meshed_by_their_grains():
    """Cells of several phases: their exact polyhedra, each element with the phase of its
    cell, at the fractions of the cells."""
    model = model_from_config(
        {"type": "voronoi", "n_cells": 12, "phase_fractions": [0.4, 0.3, 0.3]}
    )
    micro = model.generate(mg.Domain.cube(6.0), seed=1)
    mesh = conforming_mesh(micro, size=0.8)
    assert set(np.unique(mesh.phase)) == {0, 1, 2} and mesh.matching_faces()
    assert np.array_equal(mesh.phase, micro.cell_phases[mesh.cell])
    volumes = tet_volumes(mesh.nodes, mesh.elements)
    assert volumes.sum() == pytest.approx(216.0, rel=1e-10)
    image = np.asarray(micro.voxelize((96, 96, 96)).data)
    for k in range(3):
        fraction = volumes[mesh.phase == k].sum() / 216.0
        assert fraction == pytest.approx((image == k).mean(), abs=5e-3)


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_conforming_mesh_of_coated_spheres():
    """Spheres, their coating and the matrix: three phases meshed conformingly, at the
    fractions of the spheres and of their coatings within the accuracy of the chords of
    4-node elements as large as 0.8 of the radius: the spheres, polyhedra inscribed within
    the Hausdorff distance of the mesher, lose 7.6 % of their volume (3.5 % with elements
    of 0.5), which the coatings around them gain in part (+1.3 %; -0.4 %). The coatings of
    spheres closer than twice their thickness overlap: their volume is the sum of the
    shells less the lenses that two coated spheres share (4 % of it here); the mesh
    follows the grooves where they meet. With these grooves rounded
    (``sharp_edges=False``) the coatings, thinner than the elements, lose 5 %."""
    model = model_from_config({
        "type": "coating",
        "model": {"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.2},
        "thickness": 0.3,
    })  # fmt: skip
    micro = model.generate(mg.Domain.cube(6.0), seed=1)
    count = micro.inner.spheres.radii.size
    mesh = conforming_mesh(micro, size=0.8, method="grid")
    assert set(np.unique(mesh.phase)) == {0, 1, 2} and mesh.matching_faces()
    volumes = tet_volumes(mesh.nodes, mesh.elements)
    assert volumes.sum() == pytest.approx(216.0, rel=1e-10)
    spheres = count * 4.0 / 3.0 * np.pi / 216.0
    centres = micro.inner.spheres.centers
    apart = centres[:, None, :] - centres[None, :, :]
    apart -= 6.0 * np.round(apart / 6.0)
    apart = np.linalg.norm(apart, axis=2)[np.triu_indices(count, 1)]
    apart = apart[apart < 2.6]  # the coated spheres that overlap: 11 pairs
    lenses = np.sum(np.pi * (2.6 - apart) ** 2 * (apart + 5.2) / 12.0)
    coatings = (count * 4.0 / 3.0 * np.pi * (1.3**3 - 1.0) - lenses) / 216.0
    assert len(apart) == 11 and coatings == pytest.approx(0.2226, abs=1e-4)
    assert 0.90 * spheres < volumes[mesh.phase == 1].sum() / 216.0 < 0.95 * spheres
    assert volumes[mesh.phase == 2].sum() / 216.0 == pytest.approx(coatings, rel=0.03)
    assert (mesh.quality() < 0.01).mean() < 1e-3
    assert mesh.metadata["sharp_edges"]["pieces"] == 3 and len(mesh.sharp) > 100
    rounded = conforming_mesh(micro, size=0.8, sharp_edges=False, method="grid")
    volumes = tet_volumes(rounded.nodes, rounded.elements)
    assert volumes[rounded.phase == 1].sum() / 216.0 == pytest.approx(spheres, rel=0.05)
    assert 0.93 * coatings < volumes[rounded.phase == 2].sum() / 216.0 < coatings
