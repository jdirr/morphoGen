"""Triangular meshes of the plane (mesh.triangles) and their finite elements
(homogenization.plane): the exact polygons of the planar tessellations, graded from their
small features, the sections of the implicit structures, the matching sides of the periodic
ones, the layers of tetrahedra extruded from them, their files; the problems of the plane
checked against the uniform fields, the laminates, the checkerboard of Dykhne and the same
problems in space on the extruded layers; the curved six-node triangles of the sections
(mesh.curved) against the areas of discs, the conductivity of square arrays of cylinders
(Perrins et al., Keller) and the neutral coated cylinder of Hashin."""

from __future__ import annotations

import numpy as np
import pytest

import morphogen as mg
from morphogen.distributions import Uniform
from morphogen.homogenization import Material, homogenize
from morphogen.homogenization.apparent import conductivity_tensor, stiffness_tensor
from morphogen.homogenization.fe import FEElasticProblem, FEThermalProblem
from morphogen.homogenization.materials import isotropic_matrix, rotate_stiffness
from morphogen.homogenization.plane import PlaneElasticProblem, PlaneThermalProblem
from morphogen.io import read_tri_mesh, save_mesh, write_periodic_pairs
from morphogen.mesh.conforming import mmg_executable
from morphogen.mesh.curved import curved_triangles
from morphogen.mesh.triangles import (
    TriMesh,
    background_triangulation,
    planar_pieces,
    planar_tessellation,
    plane_level_set,
    plane_mesh,
    section_mesh,
    tessellation_mesh,
)
from morphogen.models.long_fibers import LongFibers
from morphogen.models.nested import CoatedStructure
from morphogen.models.seeded import SeededLaguerre
from morphogen.models.tessellation import Tessellation
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments


def _has_mmg2d() -> bool:
    try:
        mmg_executable(None, "mmg2d")
    except (ImportError, FileNotFoundError):
        return False
    return True


needs_mmg2d = pytest.mark.skipif(not _has_mmg2d(), reason="mmg2d (pymmg) is not installed")


def _shoelace(p: np.ndarray) -> float:
    return 0.5 * float(
        np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))
    )


def _seeds(periodic: bool, size=3.0, seed=3):
    return SeededLaguerre(dimension=2, aspect_ratio=2.0, crystal_orientations=True).generate(
        mg.Domain(size=(size, size, 0.1), periodic=periodic), seed=seed
    )


def _square(n: int, length=1.0, phase=None, **options) -> TriMesh:
    nodes, tri, _, _ = background_triangulation((n, n), (0.0, 0.0), (length, length))
    phase = np.zeros(len(tri), dtype=np.int64) if phase is None else phase(nodes[tri].mean(axis=1))
    return TriMesh(nodes, tri, phase, (0.0, 0.0), (length, length), periodic=True, **options)


# -- the meshes ----------------------------------------------------------------------------


def test_triangles_and_their_layers():
    """A mesh of the plane orients its triangles counterclockwise, gives the edges and the
    facing nodes of its sides, its six-node elements and its layer of tetrahedra: three per
    triangle, of the volume of the area times the thickness, matching across the layer."""
    mesh = _square(4)
    assert mesh.areas().min() > 0 and mesh.areas().sum() == pytest.approx(1.0)
    flipped = TriMesh(mesh.nodes, mesh.elements[:, [0, 2, 1]], mesh.phase, mesh.lower,
                      mesh.upper)  # fmt: skip
    assert flipped.areas().min() > 0
    assert mesh.matching_sides() and len(mesh.periodic_pairs(0)) == 5
    assert len(mesh.side_edges(1, 1)) == 4
    quad = mesh.quadratic()
    assert quad.order == 2 and quad.n_nodes == 25 + 56 and quad.side_edges(0, 0).shape == (4, 3)
    assert quad.matching_sides() and quad.linear().n_nodes == 25
    layer = TriMesh(mesh.nodes, mesh.elements, mesh.phase, mesh.lower, mesh.upper, True,
                    thickness=0.3, z=1.0).extruded()  # fmt: skip
    assert layer.n_elements == 3 * mesh.n_elements
    assert layer.volumes().min() > 0 and layer.volumes().sum() == pytest.approx(0.3)
    np.testing.assert_allclose(layer.lower, [0.0, 0.0, 0.85])
    with pytest.raises(ValueError, match=r"3 \(linear\) or 6"):
        TriMesh(mesh.nodes, mesh.elements[:, :2], mesh.phase[:1], mesh.lower, mesh.upper)


def test_power_diagram_of_the_plane():
    """The pieces of the power diagram of coplanar germs fill the box and are those of the
    exact cells in space: their areas are the volumes of the cells over the thickness."""
    rng = np.random.default_rng(1)
    germs = np.column_stack([rng.random((40, 2)) * 2.0, np.full(40, 0.05)])
    for periodic in (True, False):
        tess = Tessellation(mg.Domain(size=(2.0, 2.0, 0.1), periodic=periodic), germs,
                            rng.uniform(0.0, 0.02, 40), np.ones(40, dtype=np.uint8))  # fmt: skip
        assert planar_tessellation(tess)
        vertices, loops, cell = planar_pieces(tess)
        areas = np.array([_shoelace(vertices[loop]) for loop in loops])
        assert areas.min() > 0 and areas.sum() == pytest.approx(4.0, rel=1e-12)
        per_cell = np.bincount(cell, areas, minlength=40)
        exact = tess.grain_shapes()["volume"] / 0.1
        np.testing.assert_allclose(per_cell, exact, rtol=1e-9, atol=1e-12)
    assert not planar_tessellation(mg.models.Voronoi(n_cells=5).generate(mg.Domain.cube(1.0),
                                                                         seed=1))  # fmt: skip


@needs_mmg2d
@pytest.mark.parametrize("periodic", [True, False])
def test_exact_grains_of_the_plane(periodic):
    """The mesh of the columnar grains of seeds of the plane: each element in its grain,
    their areas those of the exact grains to round-off, triangles of angles above a few
    degrees (the edges graded from the small features of the polygons), the opposite sides
    of a periodic structure matching."""
    seeds = _seeds(periodic)
    mesh = tessellation_mesh(seeds, size=0.2, min_feature=0)
    exact = seeds.grain_shapes()["area"]
    per = np.bincount(mesh.cell, mesh.areas(), minlength=seeds.n_grains)
    np.testing.assert_allclose(per, exact, rtol=1e-9, atol=1e-12)
    assert mesh.areas().sum() == pytest.approx(9.0, rel=1e-12)
    assert mesh.min_angles().min() > 3.0
    assert mesh.periodic == periodic
    if periodic:
        assert mesh.matching_sides()
    assert mesh.orientations is not None and mesh.thickness == pytest.approx(0.1)
    quad = plane_mesh(seeds, size=0.2, order=2, min_feature=0)
    assert quad.order == 2 and quad.n_elements == mesh.n_elements
    middle = 0.5 * (quad.nodes[quad.elements[:, 0]] + quad.nodes[quad.elements[:, 1]])
    np.testing.assert_allclose(quad.nodes[quad.elements[:, 3]], middle, atol=1e-12)


@needs_mmg2d
@pytest.mark.parametrize("periodic", [True, False])
def test_small_features_removed(periodic):
    """By default, as the cells in space, the features of the polygons smaller than a
    quarter of the elements are removed (short edges collapsed, the vertices near the
    sides moved onto them): far fewer elements, none flat, the areas of the grains within
    a few percent of the exact ones and their sum that of the box, the sides matching."""
    seeds = SeededLaguerre(dimension=2, diameter=Uniform(0.08, 0.16), aspect_ratio=2.5,
                           orientation="aligned").generate(
        mg.Domain(size=(1.0, 1.0, 0.05), periodic=periodic), seed=4)  # fmt: skip
    exact = tessellation_mesh(seeds, size=0.025, min_feature=0)
    mesh = tessellation_mesh(seeds, size=0.025)
    assert mesh.n_elements < 0.5 * exact.n_elements
    assert mesh.min_angles().min() > 5.0
    assert mesh.areas().sum() == pytest.approx(1.0, rel=1e-12)
    areas = seeds.grain_shapes()["area"]
    per = np.bincount(mesh.cell, mesh.areas(), minlength=seeds.n_grains)
    used = areas > 1e-3
    error = np.abs(per[used] - areas[used]) / areas[used]
    assert error.max() < 0.08 and error.mean() < 0.01
    if periodic:
        assert mesh.matching_sides()


@needs_mmg2d
def test_sections_of_implicit_structures():
    """The section of spheres in the plane z: the fraction of the discs that of their level
    set on a fine grid of the plane, matching sides; the phases kept."""
    micro = mg.models.HardcoreSpheres(volume_fraction=0.3, radius=0.3).generate(
        mg.Domain.cube(2.0, periodic=True), seed=4
    )
    mesh = section_mesh(micro, size=0.08, z=1.0)
    assert mesh.matching_sides() and set(np.unique(mesh.phase)) == {0, 1}
    values, _ = plane_level_set(micro, 0.005, 1.0)  # the discs on a fine grid of the plane
    voxels = float(np.mean(values[:-1, :-1] < 0.0))
    assert mesh.phase_fractions()[1] == pytest.approx(voxels, abs=0.01)
    solid = section_mesh(micro, size=0.08, z=1.0, phases="solid")
    assert set(np.unique(solid.phase)) == {1}
    assert solid.areas().sum() == pytest.approx(4.0 * mesh.phase_fractions()[1], rel=1e-9)


@needs_mmg2d
def test_files_of_the_plane(tmp_path):
    """The meshes of the plane in .vtu, .mesh, .msh and .inp, read back with their grains,
    orientations, rectangle, periodicity, plane and thickness; the plane elements of
    Abaqus, their sides and the crystals of each grain turned into the axes of the box."""
    seeds = _seeds(True)
    mesh = tessellation_mesh(seeds, size=0.3)
    for order in (1, 2):
        m = mesh if order == 1 else mesh.quadratic()
        for ext in ("vtu", "mesh"):
            back = read_tri_mesh(save_mesh(m, tmp_path / f"m{order}.{ext}"))
            assert back.order == (order if ext == "vtu" else 1)
            assert back.n_elements == m.n_elements and back.periodic
            assert back.z == pytest.approx(m.z) and back.thickness == pytest.approx(0.1)
            np.testing.assert_array_equal(back.cell, m.cell)
            np.testing.assert_allclose(np.sort(back.areas()), np.sort(m.linear().areas()))
        assert read_tri_mesh(tmp_path / f"m{order}.vtu").orientations is not None
        text = save_mesh(m, tmp_path / f"m{order}.inp").read_text()
        assert ("CPE3" if order == 1 else "CPE6") in text and "*NSET, NSET=Y1" in text
        msh = save_mesh(m, tmp_path / f"m{order}.msh").read_text()
        assert '1 104 "y1"' in msh and '2 2 "phase_1"' in msh
    cubic = Material.cubic(c11=168.0, c12=121.0, c44=75.0)
    text = save_mesh(mesh, tmp_path / "grains.inp", materials={1: cubic}).read_text()
    grains = len(np.unique(mesh.cell))
    assert text.count("*ELASTIC, TYPE=ANISOTROPIC") == grains
    assert text.count("*SOLID SECTION, ELSET=CELL_") == grains
    with pytest.raises(ValueError, match="plane stress"):
        save_mesh(mesh, tmp_path / "stress.inp", materials={1: cubic}, plane="stress")
    pairs = write_periodic_pairs(tmp_path / "pairs.csv", mesh).read_text().splitlines()
    assert pairs[0] == "axis,node_lower,node_upper" and {r[0] for r in pairs[1:]} == {"x", "y"}


# -- the finite elements of the plane ------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("bc", ["periodic", "kubc", "subc"])
def test_uniform_fields(order, bc):
    """A homogeneous square: the stiffness and the conductivity of its material under every
    condition, in generalized plane strain and in plane strain, to round-off."""
    mesh = _square(5)
    mesh = mesh if order == 1 else mesh.quadratic()
    mat = {0: Material(young=200.0, poisson=0.3, conductivity=5.0)}
    lam, mu = 200.0 * 0.3 / (1.3 * 0.4), 200.0 / 2.6
    for plane in ("generalized", "strain"):
        c, _ = stiffness_tensor(PlaneElasticProblem(mesh, mat, bc, plane=plane))
        np.testing.assert_allclose(c, isotropic_matrix(lam, mu), atol=1e-9)
    k, _ = conductivity_tensor(PlaneThermalProblem(mesh, mat, bc))
    np.testing.assert_allclose(k, 5.0 * np.eye(3), atol=1e-11)


@pytest.mark.parametrize("order", [1, 2])
def test_laminates_of_crystals(order):
    """Three layers of turned crystals and of an isotropic phase across x: the periodic
    stiffness of the plane is that of the laminate, exactly (the strains uniform in each
    layer), the anti-plane shears coupled to the plane."""
    nodes, tri, _, _ = background_triangulation((12, 4), (0.0, 0.0), (1.2, 1.0))
    x = nodes[tri].mean(axis=1)[:, 0]
    phase = np.where(x < 0.4, 0, np.where(x < 0.9, 1, 2))
    euler = np.array([[30.0, 40.0, 10.0], [0.0, 0.0, 0.0], [80.0, 20.0, 60.0]])
    mesh = TriMesh(nodes, tri, phase, (0, 0), (1.2, 1.0), True, cell=phase, orientations=euler)
    mesh = mesh if order == 1 else mesh.quadratic()
    mats = {0: Material.cubic(c11=168.0, c12=121.0, c44=75.0),
            1: Material(young=70.0, poisson=0.33),
            2: Material.cubic(c11=250.0, c12=150.0, c44=120.0)}  # fmt: skip
    layers = [rotate_stiffness(mats[0].stiffness, euler[0]), mats[1].matrix("elastic"),
              rotate_stiffness(mats[2].stiffness, euler[2])]  # fmt: skip
    fractions = [0.4 / 1.2, 0.5 / 1.2, 0.3 / 1.2]
    n, t = [0, 5, 4], [1, 2, 3]  # the components across the layers, and along them
    inv = [np.linalg.inv(c[np.ix_(n, n)]) for c in layers]
    cnn = np.linalg.inv(sum(f * i for f, i in zip(fractions, inv, strict=True)))
    a = sum(f * i @ c[np.ix_(n, t)] for f, i, c in zip(fractions, inv, layers, strict=True))
    ctt = sum(f * (c[np.ix_(t, t)] - c[np.ix_(t, n)] @ i @ c[np.ix_(n, t)])
              for f, i, c in zip(fractions, inv, layers, strict=True)) + a.T @ cnn @ a  # fmt: skip
    ref = np.zeros((6, 6))
    ref[np.ix_(n, n)], ref[np.ix_(n, t)], ref[np.ix_(t, t)] = cnn, cnn @ a, ctt
    ref[np.ix_(t, n)] = (cnn @ a).T
    c, _ = stiffness_tensor(PlaneElasticProblem(mesh, mats, "periodic"))
    np.testing.assert_allclose(c, ref, atol=1e-10 * np.abs(ref).max())


def test_checkerboard_of_dykhne():
    """The conductivity of a checkerboard of two phases tends to the geometric mean of
    theirs (Dykhne), from above, with the refinement and the order of the elements; across
    the plane, it is their arithmetic mean."""
    mats = {0: Material(conductivity=1.0), 1: Material(conductivity=10.0)}

    def board(xy):
        return ((np.floor(2.0 * xy[:, 0]) + np.floor(2.0 * xy[:, 1])) % 2).astype(np.int64)

    errors = []
    for n in (16, 32):
        for order in (1, 2):
            mesh = _square(n, phase=board)
            k, _ = conductivity_tensor(
                PlaneThermalProblem(mesh if order == 1 else mesh.quadratic(), mats, "periodic")
            )
            errors.append(k[0, 0] / np.sqrt(10.0) - 1.0)
            assert k[2, 2] == pytest.approx(5.5, rel=1e-12)
            assert k[0, 0] == pytest.approx(k[1, 1], rel=1e-9) and abs(k[0, 1]) < 1e-9
    assert all(e > 0 for e in errors)
    assert errors[1] < errors[0] and errors[3] < errors[2] and errors[2] < errors[0]
    assert errors[3] < 0.04


def test_the_layer_in_space():
    """The problems of the plane and those of the layer of tetrahedra extruded from its mesh
    (periodic across it) are the same: the stiffness and the conductivity of grains of turned
    crystals, to round-off; kinematic conditions stiffer than periodic ones, static ones
    softer."""
    rng = np.random.default_rng(5)

    def grains(xy):
        return np.floor(4 * xy[:, 0]).astype(np.int64) + 4 * np.floor(4 * xy[:, 1]).astype(np.int64)

    base = _square(12, phase=grains)
    cell = base.phase.copy()
    mesh = TriMesh(base.nodes, base.elements, np.zeros_like(cell), base.lower, base.upper, True,
                   cell=cell, orientations=rng.uniform(0, 90, (16, 3)), thickness=0.2)  # fmt: skip
    mats = {0: Material.cubic(c11=168.0, c12=121.0, c44=75.0, conductivity=3.0)}
    c2, _ = stiffness_tensor(PlaneElasticProblem(mesh, mats, "periodic"))
    c3, _ = stiffness_tensor(FEElasticProblem(mesh.extruded(), mats, "periodic", solver="direct"))
    np.testing.assert_allclose(c2, c3, atol=1e-9 * np.abs(c3).max())
    anisotropic = Material(conductivities=np.diag([1.0, 2.0, 5.0]))
    k2, _ = conductivity_tensor(PlaneThermalProblem(mesh, {0: anisotropic}, "periodic"))
    k3, _ = conductivity_tensor(FEThermalProblem(mesh.extruded(), {0: anisotropic}, "periodic",
                                                 solver="direct"))  # fmt: skip
    np.testing.assert_allclose(k2, k3, atol=1e-10)
    kubc, _ = stiffness_tensor(PlaneElasticProblem(mesh, mats, "kubc"))
    subc, _ = stiffness_tensor(PlaneElasticProblem(mesh, mats, "subc"))
    assert np.all(np.linalg.eigvalsh(kubc - c2) > -1e-9)
    assert np.all(np.linalg.eigvalsh(c2 - subc) > -1e-9)
    strain, _ = stiffness_tensor(PlaneElasticProblem(mesh, mats, "periodic", plane="strain"))
    assert np.all(np.linalg.eigvalsh(strain - c2) > -1e-9)  # the anti-plane constrained


def test_homogenize_the_plane():
    """``homogenize`` of a mesh of the plane: the moduli, the tensors, the thermal expansion
    and the named loadings under its conditions, the fields of its solutions; what it does
    not solve refused."""
    mesh = _square(6, phase=lambda xy: (xy[:, 0] > 0.5).astype(np.int64))
    mats = {0: Material(young=100.0, poisson=0.25, conductivity=1.0, expansion=1e-5),
            1: Material(young=300.0, poisson=0.25, conductivity=4.0, expansion=1e-5)}  # fmt: skip
    out = homogenize(mesh, mats, bcs=("periodic", "kubc"), fields=True,
                     properties=("bulk_modulus", "stiffness", "thermal_expansion",
                                 "conductivity_tensor"),
                     loadings={"pull": {"strain": {"11": 0.01}}})  # fmt: skip
    np.testing.assert_allclose(out["thermal_expansion.periodic"]["tensor"], 1e-5 * np.eye(3),
                               atol=1e-15)  # fmt: skip
    k = out["conductivity_tensor.periodic"]["tensor"]
    assert k[0, 0] == pytest.approx(1.0 / (0.5 / 1.0 + 0.5 / 4.0)) and k[1, 1] == pytest.approx(2.5)
    pull = out["pull.periodic"].components()
    assert pull["S22"] == pytest.approx(0.0, abs=1e-9) and pull["E22"] < 0
    sol = out["bulk_modulus.periodic"].solution
    assert sol.fields["displacement"].shape == (mesh.n_nodes, 3) and sol.support is mesh
    assert {p.phase for p in sol.phases} == {0, 1}
    with pytest.raises(ValueError, match="not by the finite elements of the plane"):
        homogenize(mesh, mats, properties=("swelling",))
    with pytest.raises(ValueError, match="conditions kubc, periodic and subc"):
        homogenize(mesh, mats, bcs=("mixed",))
    with pytest.raises(ValueError, match="plane: the hypothesis"):
        homogenize(
            mg.models.Voronoi(n_cells=3)
            .generate(mg.Domain.cube(1.0), seed=1)
            .voxelize(shape=(4, 4, 4)),
            mats,
            plane="strain",
        )
    porous = _square(6, phase=lambda xy: (np.hypot(xy[:, 0] - 0.5, xy[:, 1] - 0.5) < 0.3)
                     .astype(np.int64))  # fmt: skip
    voids = {0: Material(young=100.0, poisson=0.25), 1: Material(void=True)}
    soft = homogenize(porous, voids, bcs=("periodic", "kubc"))
    assert 0 < soft["bulk_modulus.periodic"].value < soft["bulk_modulus.kubc"].value < 100.0
    with pytest.raises(ValueError, match="porous"):
        PlaneElasticProblem(porous, voids, "subc")


# -- the runs ------------------------------------------------------------------------------


@needs_mmg2d
def test_columnar_grains_in_space():
    """The mesher in space takes the columnar grains of seeds of the plane: their exact
    polygons extruded through the box in layers, the volume of each grain its area times
    the thickness, the faces matching, ten-node elements on demand."""
    seeds = _seeds(True)
    mesh = mg.mesh.conforming_mesh(seeds, size=0.05, min_feature=0)
    assert mesh.metadata["layers"] == 2 and mesh.periodic and mesh.matching_faces()
    per = np.bincount(mesh.cell, np.abs(mesh.volumes()), minlength=seeds.n_grains)
    np.testing.assert_allclose(per, 0.1 * seeds.grain_shapes()["area"], rtol=1e-9, atol=1e-13)
    quad = mg.mesh.conforming_mesh(seeds, size=0.3, order=2)
    assert quad.order == 2 and quad.volumes().min() > 0
    with pytest.raises(ValueError, match="x and y together"):
        mg.mesh.conforming_mesh(seeds, size=0.3, periodic=(True, False, True))


def _plane_config(tmp_path, **changes) -> dict:
    data = {
        "model": {"type": "seeded_laguerre", "dimension": 2, "diameter": 0.6,
                  "crystal_orientations": True},
        "domain": {"size": [3.0, 3.0, 0.1], "periodic": True},
        "seed": 3,
        "mesh": {"size": 0.25},
        "homogenization": {"method": "fe", "materials": {1: {"cubic": [168.0, 121.0, 75.0],
                                                             "conductivity": 3.0}},
                           "boundary_conditions": ["periodic", "kubc"],
                           "properties": ["stiffness", "conductivity"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "sheet",
                    "conforming_mesh": ["vtu", "inp"], "periodic_pairs": True},
    }  # fmt: skip
    for key, value in changes.items():
        section, _, name = key.partition("__")
        if name:
            data[section][name] = value
        else:
            data[section] = value
    return data


@needs_mmg2d
def test_runs_of_the_plane(tmp_path):
    """A configuration of seeds of the plane meshes its grains in their plane by default and
    solves them by the finite elements of the plane: the files of the triangles, the plane
    elements of Abaqus with their crystals, the stiffness of the grains (their bulk modulus
    that of the crystals) under periodic and kinematic conditions; plane strain asked."""
    from morphogen.config import load_config
    from morphogen.pipeline import run

    cfg = load_config(_plane_config(tmp_path), base_dir=tmp_path)
    assert cfg.mesh.plane is None and cfg.mesh.order == 2
    run(cfg)
    out = tmp_path / "out"
    names = {p.name for p in out.iterdir()}
    assert {"sheet_0000_tri.vtu", "sheet_0000_tri.inp", "sheet_0000_tri_periodic.csv"} <= names
    mesh = read_tri_mesh(out / "sheet_0000_tri.vtu")
    assert mesh.order == 2 and mesh.periodic and mesh.orientations is not None
    assert "CPE6" in (out / "sheet_0000_tri.inp").read_text()
    rows = (out / "results.csv").read_text().splitlines()
    header = rows[0].split(",")
    values = dict(zip(header, rows[1].split(","), strict=True))
    bulk = float(values["stiffness.periodic.bulk_modulus"])
    assert bulk == pytest.approx((168.0 + 2 * 121.0) / 3.0, rel=1e-9)
    assert float(values["conductivity.kubc"]) == pytest.approx(3.0)
    assert float(values["stiffness.kubc.C11"]) >= float(values["stiffness.periodic.C11"])
    strain = load_config(_plane_config(tmp_path, homogenization__plane="strain"),
                         base_dir=tmp_path)  # fmt: skip
    assert strain.homogenization.plane == "strain"


def test_configurations_of_the_plane(tmp_path):
    """The keys of the meshes of the plane and what they do not take."""
    from morphogen.config import ConfigError, load_config

    base = _plane_config(tmp_path)
    cfg = load_config({**base, "mesh": {"size": 0.25, "plane": True, "z": 0.05}},
                      base_dir=tmp_path)  # fmt: skip
    assert cfg.mesh.plane is True and cfg.mesh.z == 0.05
    assert cfg.mesh.plane_kwargs()["z"] == 0.05
    straight = {"size": 0.25, "plane": True, "order": 2, "curved": False}
    cfg = load_config({**base, "mesh": straight}, base_dir=tmp_path)
    assert cfg.mesh.plane_kwargs()["curved"] is False
    with pytest.raises(ConfigError, match=r"mesh\.plane: true"):
        load_config({**base, "mesh": {"size": 0.25, "plane": "yes"}}, base_dir=tmp_path)
    with pytest.raises(ConfigError, match=r"mesh\.z"):
        load_config({**base, "mesh": {"size": 0.25, "plane": False, "z": 0.05}},
                    base_dir=tmp_path)  # fmt: skip
    with pytest.raises(ConfigError, match=r"homogenization\.plane"):
        load_config(_plane_config(tmp_path, homogenization__plane="stress"), base_dir=tmp_path)
    with pytest.raises(ConfigError, match=r"homogenization\.plane: the elastic problems"):
        load_config({**_plane_config(tmp_path, homogenization__plane="strain"),
                     "mesh": {"size": 0.25, "plane": False}}, base_dir=tmp_path)  # fmt: skip
    with pytest.raises(ConfigError, match=r"do not take outputs\.code_aster"):
        outputs = {**base["outputs"], "code_aster": True}
        load_config({**base, "mesh": {"size": 0.25, "plane": True}, "outputs": outputs},
                    base_dir=tmp_path)  # fmt: skip


@needs_mmg2d
def test_the_command_line_of_the_plane(tmp_path, capsys):
    """``morphogen homogenize`` of a geometry of seeds of the plane: its triangles and the
    finite elements of the plane, in plane strain on demand; a mesh of the plane read back."""
    import json

    from morphogen import cli

    seeds = _seeds(True)
    geometry = tmp_path / "seeds.json"
    mg.io.save_microstructure(seeds, geometry)
    spec = "1=cubic:168/121/75"
    code = cli.main(["homogenize", str(geometry), "--size", "0.3", "--order", "1",
                     "--material", spec, "--property", "stiffness", "--plane", "strain",
                     "--save-mesh", str(tmp_path / "plane.vtu"), "--json"])  # fmt: skip
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["plane"] == "strain" and out["order"] == 1
    assert cli.main(["homogenize", str(tmp_path / "plane.vtu"), "--material", spec,
                     "--property", "bulk_modulus", "--periodic"]) == 0  # fmt: skip
    assert "triangles" in capsys.readouterr().out


@needs_mmg2d
def test_figures_of_the_plane(tmp_path):
    """The figures of a mesh of the plane and of a field of its solution, on its triangles
    deformed by the displacement."""
    pytest.importorskip("matplotlib")
    from morphogen.viz.figures import mesh_figure, mesh_section

    seeds = _seeds(True)
    mesh = tessellation_mesh(seeds, size=0.3)
    polygons, elements = mesh_section(mesh)
    assert len(polygons) == mesh.n_elements and elements.tolist() == list(range(mesh.n_elements))
    fig = mesh_figure(mesh, values=np.arange(mesh.n_elements), label="element")
    fig.savefig(tmp_path / "mesh.png")
    assert (tmp_path / "mesh.png").stat().st_size > 1000


@needs_mmg2d
@pytest.mark.parametrize("periodic", [True, False])
def test_exact_sections_of_tessellations(periodic):
    """The section of a Laguerre tessellation in space by a plane is the power diagram of the
    plane of the weights w - (z - z_k)²: the areas of its grains those of the points of the
    plane in each cell (Monte Carlo), its mesh their exact polygons."""
    micro = mg.models.Laguerre(n_cells=60, radius=Uniform(0.05, 0.15), crystal_orientations=True)
    micro = micro.generate(mg.Domain.cube(1.0, periodic=periodic), seed=2)
    vertices, loops, cell = planar_pieces(micro, 0.37)
    areas = np.bincount(cell, [_shoelace(vertices[loop]) for loop in loops], minlength=60)
    assert areas.sum() == pytest.approx(1.0, rel=1e-12)
    rng = np.random.default_rng(0)
    points = np.column_stack([rng.random((200_000, 2)), np.full(200_000, 0.37)])
    sampled = np.bincount(micro.cell_of(points), minlength=60) / 200_000
    sigma = np.sqrt(np.maximum(areas * (1 - areas), 1e-6) / 200_000)
    assert np.all(np.abs(areas - sampled) < 5 * sigma + 1e-4)
    mesh = plane_mesh(micro, size=0.05, z=0.37, min_feature=0)
    per = np.bincount(mesh.cell, mesh.areas(), minlength=60)
    np.testing.assert_allclose(per, areas, rtol=1e-9, atol=1e-12)
    assert mesh.z == pytest.approx(0.37) and mesh.periodic == periodic
    if periodic:
        assert mesh.matching_sides()


def test_paths_of_the_plane():
    """The loading paths of a mesh of the plane: on its layer of tetrahedra in space, periodic
    across it, the same as on that layer itself; under periodic conditions only; not in plane
    strain."""
    from morphogen.homogenization import LoadingPath
    from morphogen.homogenization.behaviours import Plasticity

    mesh = _square(6, phase=lambda xy: (np.hypot(xy[:, 0] - 0.5, xy[:, 1] - 0.5) < 0.3)
                   .astype(np.int64)).quadratic()  # fmt: skip
    mats = {0: Material(young=70e3, poisson=0.3, plasticity=Plasticity(yield_stress=100.0,
                                                                       hardening=1000.0)),
            1: Material(young=400e3, poisson=0.25)}  # fmt: skip
    path = LoadingPath.from_spec({"times": [0, 1], "increments": 4, "strain": {"11": 0.01}})
    plane = homogenize(mesh, mats, bcs=("periodic", "kubc"), properties=(), paths={"pull": path})
    assert "pull.kubc" not in plane
    layer = homogenize(mesh.layer(), mats, properties=(), paths={"pull": path})
    a, b = plane["pull.periodic"].components(), layer["pull.periodic"].components()
    assert a["S11"] == pytest.approx(b["S11"], rel=1e-9) and abs(a["S22"]) < 1e-6
    assert 100.0 < a["S11"] < 400e3 * 0.01
    assert mesh.layer().order == 2 and mesh.layer().periodic
    with pytest.raises(ValueError, match="generalized plane strain"):
        homogenize(mesh, mats, properties=(), paths={"pull": path}, plane="strain")


def test_assembled_by_chunks(monkeypatch):
    """The stiffness of the problems of the plane assembled by chunks of elements is the
    stiffness assembled at once (the global unknowns of static conditions among them)."""
    import morphogen.homogenization.plane as plane

    mesh = _square(10, phase=lambda xy: (xy[:, 0] > 0.5).astype(np.int64)).quadratic()
    mats = {0: Material.cubic(c11=168.0, c12=121.0, c44=75.0), 1: Material(young=70.0, poisson=0.3)}
    whole = PlaneElasticProblem(mesh, mats, "subc")._stiffness()
    monkeypatch.setattr(plane, "CHUNK", 37)
    parts = PlaneElasticProblem(mesh, mats, "subc")._stiffness()
    assert abs(whole - parts).max() <= 1e-12 * abs(whole).max()


def test_periodic_equations_of_the_plane(tmp_path):
    """The periodic equations of Abaqus and CalculiX of a mesh of the plane: the reference
    nodes of its two axes, the two degrees of freedom of the plane elements of each node of
    its upper sides tied to its image, the areas of its sides times its thickness."""
    from morphogen.io import write_periodic_equations

    mesh = _square(3, length=2.0)
    mesh.thickness = 0.5
    text = write_periodic_equations(tmp_path / "equations.inp", mesh).read_text()
    assert "*NSET, NSET=REF_Y" in text and "REF_Z" not in text
    upper = np.unique(np.concatenate([mesh.side_nodes(0, 1), mesh.side_nodes(1, 1)]))
    assert text.count("*EQUATION") == 2 * len(upper)
    assert "A_x = 1, A_y = 1" in text
    thermal = write_periodic_equations(tmp_path / "heat.inp", mesh, physics="thermal")
    assert thermal.read_text().count("*EQUATION") == len(upper)


def test_plane_hypothesis_of_tetrahedra(tmp_path):
    """The hypothesis of the plane asked of a run whose meshes are tetrahedra is refused."""
    from morphogen.config import ConfigError, load_config

    data = {"model": {"type": "voronoi", "n_cells": 5}, "domain": {"size": 1.0, "periodic": True},
            "mesh": {"size": 0.3},
            "homogenization": {"method": "fe", "plane": "strain",
                               "materials": {1: {"young": 1.0, "poisson": 0.3}}},
            "outputs": {"directory": str(tmp_path)}}  # fmt: skip
    with pytest.raises(ConfigError, match=r"homogenization\.plane"):
        load_config(data, base_dir=tmp_path)


def test_maps_of_grains(tmp_path):
    """An orientation map (EBSD) meshed in its plane by its pixels: two triangles per pixel,
    each with the grain and the orientation of its pixel; the bulk modulus of its crystals
    under every condition, kinematic conditions stiffer than static ones; in space, the
    pixels extruded with their grains; the pixels of no grain one more grain; a run of the
    map by the finite elements of the plane."""
    from morphogen.config import load_config
    from morphogen.mesh.triangles import lattice_map, lattice_mesh
    from morphogen.models.ebsd import EBSD
    from morphogen.models.lattice_field import LatticeField
    from morphogen.pipeline import run

    micro = EBSD(file="ebsd_map.ctf").generate(mg.Domain(size=(16.0, 12.0, 0.5), periodic=False),
                                               seed=1)  # fmt: skip
    assert lattice_map(micro)
    mesh = plane_mesh(micro, size=0.5)
    assert mesh.n_elements == 2 * 32 * 24 and mesh.areas() == pytest.approx(np.full(1536, 0.125))
    labels = micro.cell_labels[:, :, 0].ravel()
    np.testing.assert_array_equal(mesh.cell, np.repeat(labels, 2))
    np.testing.assert_allclose(mesh.orientations, micro.orientations)
    nickel = {1: Material.cubic(c11=246.5, c12=147.3, c44=124.7)}
    out = homogenize(mesh, nickel, bcs=("kubc", "subc"), properties=("bulk_modulus", "stiffness"))
    for bc in ("kubc", "subc"):
        assert out[f"bulk_modulus.{bc}"].value == pytest.approx((246.5 + 2 * 147.3) / 3)
    assert out["stiffness.kubc"]["E1"] > out["stiffness.subc"]["E1"]
    solid = mg.mesh.conforming_mesh(micro, size=0.5)  # in space: the pixels extruded
    assert len(np.unique(solid.cell)) == len(np.unique(labels)) and solid.orientations is not None
    assert np.abs(solid.volumes()).sum() == pytest.approx(16.0 * 12.0 * 0.5)
    phases = np.ones((4, 3, 1), dtype=np.uint8)
    holes = np.zeros((4, 3, 1), dtype=np.int32)
    holes[0, 0, 0] = -1
    field = LatticeField(mg.Domain(size=(4.0, 3.0, 1.0), periodic=False), phases, labels=holes,
                         orientations=[[10.0, 20.0, 30.0]])  # fmt: skip
    grains = lattice_mesh(field)
    assert set(np.unique(grains.cell)) == {0, 1} and len(grains.orientations) == 2
    data = {"model": {"type": "ebsd", "file": "ebsd_map.ctf"},
            "domain": {"size": [16.0, 12.0, 0.5], "periodic": False}, "seed": 1,
            "mesh": {"size": 0.5},
            "homogenization": {"method": "fe", "boundary_conditions": ["kubc"],
                               "materials": {1: {"cubic": [246.5, 147.3, 124.7]}},
                               "properties": ["bulk_modulus"]},
            "outputs": {"directory": str(tmp_path / "out"), "name": "map",
                        "conforming_mesh": ["vtu"]}}  # fmt: skip
    cfg = load_config(data, base_dir=tmp_path)
    assert cfg.mesh.on_plane(micro)
    run(cfg)
    assert (tmp_path / "out" / "map_0000_tri.vtu").exists()


@needs_mmg2d
def test_small_features_halved_where_they_fold(caplog):
    """Where the polygons would fold when their features below a quarter of the size are
    removed, those below half of it are: no flat element, the sides matching."""
    import logging

    from morphogen.distributions import LogNormal

    sheet = SeededLaguerre(dimension=2, diameter=LogNormal(0.12, 0.03), aspect_ratio=2.5,
                           orientation="aligned", crystal_orientations=True).generate(
        mg.Domain(size=(1.0, 1.0, 0.05), periodic=True), seed=1)  # fmt: skip
    with caplog.at_level(logging.INFO, logger="morphogen"):
        mesh = tessellation_mesh(sheet, size=0.05)
    assert "below half of it instead" in caplog.text
    assert mesh.matching_sides() and mesh.min_angles().min() > 3.0
    assert mesh.n_elements < 0.5 * tessellation_mesh(sheet, size=0.05, min_feature=0).n_elements


# -- curved six-node triangles ---------------------------------------------------------------


def _fibres(fraction: float):
    """Long straight fibres in a square array, four in a periodic box of edge 2: the section
    of a square array of cylinders, one per unit cell."""
    radius = float(np.sqrt(fraction / np.pi))
    return LongFibers(radius=radius, volume_fraction=fraction, arrangement="square").generate(
        mg.Domain(size=(2.0, 2.0, 0.5), periodic=True), seed=0
    )


def _coated(core: float = 0.2, coat: float = 0.3):
    """A coated cylinder in a periodic unit cell: its core (phase 1) of radius ``core``, its
    coating (phase 2) to the radius ``coat``, the matrix about them (phase 0)."""
    domain = mg.Domain(size=(1.0, 1.0, 1.0), periodic=True)
    axis = Segments(np.array([[0.47, 0.53, 0.0]]), np.array([[0.47, 0.53, 1.0]]), np.array([core]))
    return CoatedStructure(domain, PrimitiveUnion(domain, None, axis), coat - core, "outside", 2, 0)


class _Sampled:
    """A structure that gives its implicit functions on grids only."""

    def __init__(self, micro):
        self._micro = micro

    def __getattr__(self, name):
        if name in ("implicit_at", "phase_level_sets_at", "level_set_at"):
            raise AttributeError(name)
        return getattr(self._micro, name)


def _perrins(fraction: float, ratio: float) -> float:
    """The conductivity of a square array of cylinders of conductivity ``ratio`` in a matrix
    of unit conductivity (Perrins, McKenzie & McPhedran 1979, Proc. R. Soc. A 369, 207)."""
    f, t = fraction, (1.0 + ratio) / (1.0 - ratio)
    return 1.0 - 2.0 * f / (t + f - 0.305827 * f**4 * t / (t**2 - 1.402958 * f**8)
                            - 0.013362 * f**8)  # fmt: skip


def _conducting(ratio: float) -> dict:
    """A matrix of unit conductivity and cylinders of conductivity ``ratio`` (pores for 0)."""
    cylinders = (Material(void=True) if ratio == 0.0
                 else Material(young=1.0, poisson=0.3, conductivity=ratio))  # fmt: skip
    return {0: Material(young=1.0, poisson=0.3, conductivity=1.0), 1: cylinders}


@needs_mmg2d
def test_curved_triangles_of_discs():
    """The six-node triangles of the section of a square array of fibres (0.4 of the area),
    their nodes of the interfaces on the circles: the area of the fibres within 1e-6 at the
    size 0.1 and 1e-7 at 0.05, falling as the fourth power of the size (the straight edges
    of the mesher 2e-3 off); the box keeps its area, the sides their matching nodes, the
    elements more than 0.3 of the area elements of their straight triangles. The functions
    of a structure sampled on a grid of the plane put the nodes on their splines: 1e-5."""
    micro = _fibres(0.4)
    errors = []
    for size in (0.1, 0.05):
        straight = plane_mesh(micro, size=size, order=2, curved=False)
        mesh = plane_mesh(micro, size=size, order=2)
        assert mesh.curved and not straight.curved and not mesh.straight().curved
        info = mesh.metadata["curved"]
        assert info["nodes"] == info["interface_nodes"] > 0 and info["sampling"] == 0.0
        assert mesh.areas().sum() == pytest.approx(4.0, rel=1e-13)
        assert mesh.matching_sides() and mesh.min_jacobian() > 0.3
        errors.append(abs(mesh.phase_fractions()[1] / 0.4 - 1.0))
        assert abs(straight.phase_fractions()[1] / 0.4 - 1.0) > 1000 * errors[-1]
    assert errors[0] < 1e-6 and errors[1] < 1e-7 and errors[0] > 8 * errors[1]
    sampled = curved_triangles(plane_mesh(micro, size=0.1), _Sampled(micro))
    assert sampled.metadata["curved"]["sampling"] > 0.0
    assert sampled.phase_fractions()[1] == pytest.approx(0.4, rel=1e-5)
    with pytest.raises(ValueError, match="six-node"):
        plane_mesh(micro, size=0.1, curved=True)


@needs_mmg2d
@pytest.mark.parametrize("ratio", [10.0, 0.0])
def test_conductivity_of_a_square_array_of_cylinders(ratio):
    """Cylinders filling 0.4 of the plane, ten times as conducting as their matrix or
    insulating (pores left out of the mesh, their walls curved): the curved six-node
    triangles of size 0.1 give the conductivity of Perrins, McKenzie & McPhedran (1979)
    within 6e-5, where the straight ones are 1e-3 off."""
    micro = _fibres(0.4)
    reference = _perrins(0.4, ratio)
    assert reference == pytest.approx({10.0: 1.980607, 0.0: 0.425351}[ratio], abs=1e-6)
    phases = "all" if ratio else "matrix"
    errors = []
    for curved in (True, False):
        mesh = plane_mesh(micro, size=0.1, order=2, curved=curved, phases=phases)
        out = homogenize(mesh, _conducting(ratio), properties=["conductivity_tensor"])
        k = out["conductivity_tensor.periodic"]["tensor"]
        errors.append(abs(k[0, 0] / reference - 1.0))
        if curved:  # the square symmetry of the array, to the error of the elements
            assert abs(k[0, 1]) < 1e-5 * k[0, 0] and k[1, 1] == pytest.approx(k[0, 0], rel=1e-4)
    assert errors[0] < 6e-5 and errors[1] > 5e-4


@needs_mmg2d
def test_reciprocal_theorem_of_keller():
    """Cylinders filling 0.6 of the plane, where the series of Perrins et al. is 5e-4 off:
    the conductivities of the ratios 10 and 1/10 obey the reciprocal theorem of Keller
    (1964), their product 1, within 1e-4 on curved six-node triangles of size 0.1."""
    micro = _fibres(0.6)
    mesh = plane_mesh(micro, size=0.1, order=2)
    k = [homogenize(mesh, _conducting(t), properties=["conductivity_tensor"])
         ["conductivity_tensor.periodic"]["tensor"][0, 0] for t in (10.0, 0.1)]  # fmt: skip
    assert k[0] * k[1] == pytest.approx(1.0, abs=1e-4)
    assert _perrins(0.6, 10.0) * _perrins(0.6, 0.1) == pytest.approx(1.0, abs=1e-3)


def _neutral(core: float, coat: float) -> tuple[dict, float, float]:
    """The materials of a neutral coated cylinder (Hashin & Rosen 1964): its core (phase 1),
    its coating (phase 2), the matrix (phase 0) of the conductivity and of the plane bulk
    modulus ``k = lambda + mu`` of the assemblage of coated cylinders, which the coated
    cylinder then does not disturb; and these two."""
    c = (core / coat) ** 2
    sc, ss = 10.0, 1.0
    sm = ss * ((1 + c) * sc + (1 - c) * ss) / ((1 - c) * sc + (1 + c) * ss)
    kc, muc, ks, mus = 20.0, 8.0, 3.0, 1.5
    km = ks + c / (1.0 / (kc - ks) + (1.0 - c) / (ks + mus))
    out = {}
    for phase, (k, mu, s) in {1: (kc, muc, sc), 2: (ks, mus, ss), 0: (km, 2.0, sm)}.items():
        lam = k - mu  # in plane strain, k = lambda + mu
        young, poisson = mu * (3 * lam + 2 * mu) / (lam + mu), lam / (2 * (lam + mu))
        out[phase] = Material(young=young, poisson=poisson, conductivity=s)
    return out, sm, km


@needs_mmg2d
def test_neutral_coated_cylinder():
    """A coated cylinder whose coating makes it neutral (Hashin & Rosen 1964; Milton 2002): the
    matrix about it keeps its uniform field, and the cell the conductivity and the plane bulk
    modulus of the matrix whatever its conditions. The curved six-node triangles of size 0.1
    give them within 3e-5 under the three conditions, the straight ones 1e-3 off."""
    materials, sm, km = _neutral(0.2, 0.3)
    micro = _coated(0.2, 0.3)
    bcs = ["kubc", "periodic", "subc"]
    for curved, low, high in ((True, 0.0, 3e-5), (False, 5e-4, 5e-3)):
        mesh = plane_mesh(micro, size=0.1, order=2, curved=curved)
        assert set(np.unique(mesh.phase)) == {0, 1, 2}
        out = homogenize(mesh, materials, bcs=bcs, properties=["conductivity_tensor", "stiffness"])
        for bc in bcs:
            k = out[f"conductivity_tensor.{bc}"]["tensor"]
            c = out[f"stiffness.{bc}"]["tensor"]
            bulk = (c[0, 0] + c[0, 1]) / (2 * km) - 1.0
            for error in (k[0, 0] / sm - 1.0, k[1, 1] / sm - 1.0, bulk):
                assert low <= abs(error) < high


@needs_mmg2d
def test_curved_layer_in_space(tmp_path):
    """The layer in space of curved six-node triangles: ten-node tetrahedra whose edges over
    the curved edges are curved as they are, the areas of the phases the volumes of the
    layer over its thickness exactly; its periodic tensors those of the plane within 1e-5
    (the integration points of the curved triangles and tetrahedra differ). The files of
    VTK keep the curved edges."""
    materials, _, _ = _neutral(0.2, 0.3)
    mesh = plane_mesh(_coated(0.2, 0.3), size=0.15, order=2)
    layer = mesh.layer()
    assert layer.curved and layer.min_jacobian() == pytest.approx(mesh.min_jacobian())
    thickness = float(layer.upper[2] - layer.lower[2])
    for p in (0, 1, 2):
        area = mesh.areas()[mesh.phase == p].sum()
        assert layer.volumes()[layer.phase == p].sum() / thickness == pytest.approx(area, rel=1e-12)
    properties = ["stiffness", "conductivity_tensor"]
    plane = homogenize(mesh, materials, properties=properties)
    space = homogenize(layer, materials, properties=properties)
    for key in ("stiffness.periodic", "conductivity_tensor.periodic"):
        a, b = plane[key]["tensor"], space[key]["tensor"]
        assert np.abs(a - b).max() < 1e-5 * np.abs(a).max()
    back = read_tri_mesh(save_mesh(mesh, tmp_path / "curved.vtu"))
    assert back.curved and back.areas().sum() == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose(np.sort(back.areas()), np.sort(mesh.areas()), rtol=1e-9)
