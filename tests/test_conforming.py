"""Conforming tetrahedral meshes of implicit morphologies (MMG) and their exports."""

import numpy as np
import pytest

import morphogen as mg
from morphogen.homogenization import (
    VOID,
    Material,
    apparent_bulk_modulus,
    apparent_conductivity,
    bounds,
    homogenize,
)
from morphogen.homogenization.fe import FEElasticProblem, FEThermalProblem, periodic_map
from morphogen.mesh.conforming import background_mesh, conforming_mesh, mmg_executable
from morphogen.mesh.tetra import TetMesh, tet_volumes

try:
    mmg_executable()
except ImportError:  # pragma: no cover - depends on the environment
    pytest.skip("MMG (pymmg) is not installed", allow_module_level=True)

TI = Material(young=110e3, poisson=0.34, conductivity=7.0)


@pytest.fixture(scope="module")
def spheres():
    model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3)
    micro = model.generate(mg.Domain.cube(4.0, periodic=False), seed=3)
    return micro, conforming_mesh(micro, size=0.5)


@pytest.fixture(scope="module")
def gyroid():
    """Periodic mesh of both phases of a sheet gyroid."""
    model = mg.models.TPMS(surface="gyroid", variant="sheet", cell_size=1.0, volume_fraction=0.3)
    micro = model.generate(mg.Domain.cube(1.0, periodic=True))
    return conforming_mesh(micro, size=0.25, background=0.0625)


def test_background_mesh_is_a_tiling_of_the_box():
    nodes, tets, tri, refs = background_mesh((2, 3, 4), (0.0, -1.0, 0.5), (1.0, 2.0, 2.5))
    mesh = TetMesh(nodes, tets, np.zeros(len(tets)), (0.0, -1.0, 0.5), (1.0, 2.0, 2.5))
    assert len(tets) == 6 * 24 and np.all(mesh.volumes() > 0)
    assert mesh.volumes().sum() == pytest.approx(mesh.box_volume)
    assert sorted(set(refs.tolist())) == [1, 2, 3, 4, 5, 6] and len(tri) == 2 * 2 * (6 + 8 + 12)
    assert mesh.matching_faces()
    # with an even number of cubes per axis, the triangulation is symmetric with
    # respect to the mid-planes of the box (no preferred diagonal)
    nodes, tets, _, _ = background_mesh((4, 2, 2), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    def tet_set(points):
        return {tuple(sorted(map(tuple, np.round(points[t], 9)))) for t in tets}

    reference = tet_set(nodes)
    for axis in range(3):
        mirrored = nodes.copy()
        mirrored[:, axis] = 1.0 - mirrored[:, axis]
        assert tet_set(mirrored) == reference


@pytest.mark.slow
def test_mesh_of_spheres(spheres):
    micro, mesh = spheres
    assert mesh.order == 1 and set(np.unique(mesh.phase)) == {0, 1}
    vol = mesh.volumes()
    assert np.all(vol > 0) and vol.sum() == pytest.approx(mesh.box_volume, rel=1e-9)
    assert mesh.quality().min() > 0
    # the meshed volume fraction is that of the realization (fine voxel estimate)
    fine = mg.analysis.basic.volume_fraction(micro.voxelize(shape=96, supersampling=2), 1)
    assert mesh.phase_fractions()[1] == pytest.approx(fine, abs=0.01)
    # nodes of the faces lie exactly on them
    for axis in range(3):
        for side in (0, 1):
            nodes = mesh.face_nodes(axis, side)
            target = mesh.lower[axis] if side == 0 else mesh.upper[axis]
            assert nodes.size and np.all(mesh.nodes[nodes, axis] == target)
    # by default from the exact functions of the spheres, the sharp edges where they meet
    # followed (docs/theory.md §94)
    assert not mesh.periodic and mesh.metadata["mesher"].startswith("adaptive level set")
    assert len(mesh.sharp) > 0 and mesh.metadata["sharp_edges"]["pieces"] > 1
    solid = conforming_mesh(micro, size=0.5, phases="solid", order=2)
    assert solid.order == 2 and set(np.unique(solid.phase)) == {1}
    assert solid.phase_fractions()[1] == pytest.approx(fine, abs=0.01)
    # on the background grid
    grid = conforming_mesh(micro, size=0.5, method="grid")
    assert grid.metadata["mesher"] == "mmg3d -ls" and grid.quality().min() > 0
    assert grid.phase_fractions()[1] == pytest.approx(fine, abs=0.01)
    assert grid.volumes().sum() == pytest.approx(grid.box_volume, rel=1e-9)


@pytest.mark.slow
def test_periodic_mesh_has_matching_faces_and_exact_conditions(gyroid):
    mesh = gyroid
    assert mesh.periodic and mesh.matching_faces() and mesh.metadata["matching_faces"]
    _, _, counts = periodic_map(mesh)
    assert counts["interpolated"] == 0 and counts["matched"] > 0
    same = {0: TI, 1: TI}
    for order in (1, 2):
        sample = mesh if order == 1 else mesh.quadratic()
        k = apparent_bulk_modulus(FEElasticProblem(sample, same, "periodic"))
        assert k.value == pytest.approx(TI.bulk, rel=1e-9)
        c = apparent_conductivity(FEThermalProblem(sample, same, "periodic"))
        assert c.value == pytest.approx(7.0, rel=1e-9)


@pytest.mark.slow
def test_porous_lattice(gyroid):
    mesh = gyroid.subset([1])  # the solid sheet
    assert set(np.unique(mesh.phase)) == {1} and mesh.matching_faces()
    mats = {0: VOID, 1: TI}
    fraction = mesh.phase_fractions()[1]
    assert fraction == pytest.approx(0.3, abs=0.02)
    out = homogenize(mesh, mats, properties=["bulk_modulus", "conductivity"])
    b = bounds({0: 1 - fraction, 1: fraction}, mats)
    for key, res in out.items():
        prop = key.split(".")[0]
        assert 0 < res.value < b[prop]["hs_upper"], key
        assert res.energy == pytest.approx(res.value, rel=1e-7), key


@pytest.mark.slow
def test_pores_of_phase_1_mesh_the_matrix():
    """Spherical pores (phase 1) in a matrix: ``phases="matrix"`` meshes phase 0 alone, as
    the configurations do when the void is phase 1 (``phases: auto``)."""
    from morphogen.config import MeshSpec

    model = mg.models.BooleanSpheres(radius=0.3, volume_fraction=0.2)
    micro = model.generate(mg.Domain.cube(1.0, periodic=True), seed=1)
    pores = (micro.voxelize(shape=64).data == 1).mean()
    mesh = conforming_mesh(micro, size=0.1, phases="matrix")
    assert set(np.unique(mesh.phase)) == {0} and mesh.matching_faces()
    assert mesh.volumes().sum() == pytest.approx(1.0 - pores, abs=0.02)
    out = homogenize(mesh, {0: TI, 1: VOID}, properties=["bulk_modulus"])
    assert 0 < out["bulk_modulus.periodic"].value < TI.bulk
    assert [MeshSpec(size=1.0).kwargs(v)["phases"] for v in ({1}, {0}, True, set(), False)] == [
        "matrix", "solid", "solid", "all", "all",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="phases"):
        conforming_mesh(micro, size=0.1, phases="pores")


def _holes(radius):
    """The solid of a square array of circular holes centred in unit cells along x and y:
    negative in the solid."""

    def level_set(points):
        d = points[:, :2] % 1.0 - 0.5
        return radius - np.hypot(d[:, 0], d[:, 1])

    return level_set


@pytest.mark.slow
def test_mirrored_meshes_are_exactly_symmetric_and_periodic():
    """A square array of circular holes (an implicit body, porosity 0.5) meshed in an
    eighth of its cell and mirrored: exactly symmetric about the mid-planes, the opposite
    faces matching (exact periodic conditions), copies side by side whose translations
    are exact too: the group of a 2 x 2 cell found on the mesh (mmm and four translations),
    which a mesh of the whole box does not have; mirrored along x and y only, the faces
    along z still match; an asymmetric level set is refused."""
    from scipy.spatial import cKDTree

    from morphogen.homogenization.bifurcation import mesh_symmetries
    from morphogen.mesh import ImplicitBody, mirrored_mesh

    holes = _holes(np.sqrt(0.5 / np.pi))
    body = ImplicitBody(mg.Domain(size=(1.0, 1.0, 0.2), periodic=True), holes)
    mesh = mirrored_mesh(body, size=0.1, cells=(2, 2, 1), phases="solid")
    assert mesh.periodic and mesh.matching_faces() and set(np.unique(mesh.phase)) == {1}
    _, _, counts = periodic_map(mesh)
    assert counts["interpolated"] == 0 and counts["matched"] > 0
    assert np.all(mesh.volumes() > 0) and mesh.quality().min() > 0.1
    assert mesh.volumes().sum() / mesh.box_volume == pytest.approx(0.5, abs=0.01)
    tree, centre = cKDTree(mesh.nodes), 0.5 * (mesh.lower + mesh.upper)
    for axis in range(3):
        image = mesh.nodes.copy()
        image[:, axis] = 2 * centre[axis] - image[:, axis]
        assert tree.query(image)[0].max() < 1e-12
    assert mesh_symmetries(mesh, cells=(2, 2, 1)).order == 32
    whole = ImplicitBody(mg.Domain(size=(2.0, 2.0, 0.2), periodic=True), holes)
    assert mesh_symmetries(conforming_mesh(whole, size=0.1, phases="solid"), (2, 2, 1)).order < 32
    flat = mirrored_mesh(body, size=0.1, mirrors=(0, 1), phases="solid")
    assert flat.matching_faces() and flat.metadata["mirrors"] == [0, 1]
    shifted = ImplicitBody(body.domain, lambda p: holes(p + np.array([0.1, 0.0, 0.0])))
    with pytest.raises(ValueError, match="not symmetric along axis 0"):
        mirrored_mesh(shifted, size=0.1)


def test_mirrored_meshes_with_diagonal_mirrors():
    """The square array of holes with its diagonal mirrors (4/mmm) and a simple cubic array
    of spherical holes (m-3m): the chamber of the group (a union of background tetrahedra)
    meshed and mapped by its 16 and 48 operations, the mesh invariant under each of them,
    positive, its faces matching, its whole group detected (64 operations for a 2 x 2 cell
    with its translations, 48 for the cube); a level set without the diagonal symmetry, or
    unequal edges, refused."""
    from scipy.spatial import cKDTree

    from morphogen.homogenization.bifurcation import mesh_symmetries
    from morphogen.mesh import ImplicitBody, mirrored_mesh
    from morphogen.mesh.conforming import point_group

    square, cube = ((0, 1),), ((0, 1), (1, 2))
    assert len(point_group((0, 1, 2), square)) == 16 and len(point_group((0, 1, 2), cube)) == 48
    holes = _holes(np.sqrt(0.5 / np.pi))
    radius = (0.3 * 3 / (4 * np.pi)) ** (1 / 3)

    def spheres(points):
        d = points % 1.0 - 0.5
        return radius - np.linalg.norm(d, axis=1)

    cases = (
        (ImplicitBody(mg.Domain(size=(1.0, 1.0, 0.3), periodic=True), holes), square, (2, 2, 1),
         64, 0.5),
        (ImplicitBody(mg.Domain.cube(1.0, periodic=True), spheres), cube, (1, 1, 1), 48, 0.7),
    )  # fmt: skip
    for body, diagonals, cells, order, solid in cases:
        mesh = mirrored_mesh(body, size=0.15, diagonals=diagonals, cells=cells, phases="solid")
        group = point_group((0, 1, 2), diagonals)
        assert mesh.metadata["group_order"] == len(group) and mesh.matching_faces()
        assert np.all(mesh.volumes() > 0) and mesh.quality().min() > 0.1
        _, _, counts = periodic_map(mesh)
        assert counts["interpolated"] == 0 and counts["matched"] > 0
        assert mesh.volumes().sum() / mesh.box_volume == pytest.approx(solid, abs=0.015)
        tree, centre = cKDTree(mesh.nodes), 0.5 * (mesh.lower + mesh.upper)
        for m in group:
            assert tree.query(centre + (mesh.nodes - centre) @ m.T)[0].max() < 1e-12
        assert mesh_symmetries(mesh, cells=cells).order == order
    domain = mg.Domain(size=(1.0, 1.0, 0.3), periodic=True)
    ellipses = ImplicitBody(
        domain, lambda p: 0.3 - np.hypot((p[:, 0] % 1.0 - 0.5) / 1.2, p[:, 1] % 1.0 - 0.5)
    )
    with pytest.raises(ValueError, match="diagonal"):
        mirrored_mesh(ellipses, size=0.15, diagonals=square)
    oblong = ImplicitBody(mg.Domain(size=(1.0, 2.0, 0.3), periodic=True), holes)
    with pytest.raises(ValueError, match="equal edges"):
        mirrored_mesh(oblong, size=0.15, diagonals=square)


def test_meshes_without_grains_and_errors():
    micro = mg.models.Voronoi(n_cells=5).generate(mg.Domain.cube(1.0), seed=0)
    # cells of one phase, not meshed by grains: no interface, a mesh of the box
    plain = conforming_mesh(micro, size=0.3, grains=False)
    assert set(np.unique(plain.phase)) == {1} and plain.matching_faces()
    assert tet_volumes(plain.nodes, plain.elements).sum() == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(TypeError, match="tessellation"):
        spheres = mg.models.BooleanSpheres(radius=0.2, volume_fraction=0.3)
        conforming_mesh(spheres.generate(mg.Domain.cube(1.0), seed=0), size=0.3, grains=True)
    with pytest.raises(ValueError, match="phases"):
        conforming_mesh(micro, size=0.3, phases="pores")
    with pytest.raises(FileNotFoundError):
        mmg_executable("/no/such/mmg3d")


@pytest.mark.parametrize("order", [1, 2])
def test_exports(spheres, tmp_path, order):
    mesh = spheres[1] if order == 1 else spheres[1].quadratic()
    for ext in ("vtu", "inp", "msh", "mesh"):
        mg.io.save_mesh(mesh, tmp_path / f"m.{ext}")
    back = mg.io.read_tet_mesh(tmp_path / "m.vtu")
    np.testing.assert_allclose(back.nodes, mesh.nodes)
    np.testing.assert_array_equal(back.elements, mesh.elements)
    np.testing.assert_array_equal(back.phase, mesh.phase)
    medit = mg.io.read_tet_mesh(tmp_path / "m.mesh")
    assert medit.n_elements == mesh.n_elements and medit.order == 1
    np.testing.assert_array_equal(medit.phase, mesh.phase)
    inp = (tmp_path / "m.inp").read_text(encoding="utf-8")
    assert ("C3D4" if order == 1 else "C3D10") in inp and "*NSET, NSET=Z1" in inp
    assert "ELSET=PHASE_1" in inp
    msh = mg.io.read_msh(tmp_path / "m.msh")
    tet_type = 4 if order == 1 else 11
    assert len(msh["elements"][tet_type]) == mesh.n_elements
    if order == 2:  # Gmsh orders the last two mid-edge nodes the other way
        gmsh = msh["elements"][11][:, 2:] - 1
        np.testing.assert_array_equal(gmsh[:, [8, 9]], mesh.elements[:, [9, 8]])
