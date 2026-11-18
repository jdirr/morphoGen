"""File formats: round trips, independent readers when available, meshes and geometry."""

import json

import numpy as np
import pytest

import morphogen as mg
from helpers import edge_face_counts
from morphogen import io
from morphogen.mesh import HEX_CORNERS, surface_mesh, voxel_mesh


@pytest.fixture(scope="module")
def spheres():
    return mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.25).generate(
        mg.Domain.cube(6.0, origin=(1.0, -2.0, 0.5)), seed=2
    )


@pytest.fixture(scope="module")
def phase_image(spheres):
    return spheres.voxelize(shape=(24, 20, 16))


# -- images -------------------------------------------------------------------------------


@pytest.mark.parametrize("ext", ["vti", "vtk", "mhd", "npz", "npy", "tif"])
@pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.float32])
def test_image_round_trips(tmp_path, phase_image, ext, dtype):
    if ext == "tif":
        pytest.importorskip("tifffile")
    img = phase_image.with_data((phase_image.data * 3).astype(dtype))
    path = io.save_image(img, tmp_path / f"a.{ext}")
    back = io.load_image(path)
    assert back.data.dtype == img.data.dtype and np.array_equal(back.data, img.data)
    if ext != "npy":
        np.testing.assert_allclose(back.spacing, img.spacing)
    if ext not in ("npy", "tif"):
        np.testing.assert_allclose(back.origin, img.origin)
    if ext == "npz":
        assert back.metadata == img.metadata and back.periodic == img.periodic


@pytest.mark.parametrize("options", [{"compress": True}, {"compress": False}])
def test_multi_array_vti(tmp_path, phase_image, options):
    labels = phase_image.with_data(
        np.arange(phase_image.n_voxels, dtype=np.int64).reshape(phase_image.shape), name="label"
    )
    path = io.write_vti(tmp_path / "m.vti", [phase_image, labels], **options)
    assert np.array_equal(io.load_image(path, name="label").data, labels.data)
    assert [img.name for img in io.read_vti(path)] == ["phase", "label"]
    with pytest.raises(ValueError, match="no array"):
        io.load_image(path, name="other")


def test_legacy_vtk_ascii_and_mhd_compressed(tmp_path, phase_image):
    io.write_vtk(tmp_path / "a.vtk", phase_image, binary=False)
    assert np.array_equal(io.load_image(tmp_path / "a.vtk").data, phase_image.data)
    io.write_mhd(tmp_path / "c.mhd", phase_image, compress=True)
    assert np.array_equal(io.load_image(tmp_path / "c.mhd").data, phase_image.data)
    header = (tmp_path / "c.mhd").read_text(encoding="utf-8")
    centre = np.array(phase_image.origin) + 0.5 * np.array(phase_image.spacing)
    assert "Offset = " + " ".join(repr(float(v)) for v in centre) in header


def test_image_format_errors(tmp_path, phase_image):
    with pytest.raises(ValueError, match="unknown image format"):
        io.save_image(phase_image, tmp_path / "a.png")
    with pytest.raises(ValueError, match="single array"):
        io.save_image(
            [phase_image, phase_image.with_data(phase_image.data, name="b")], tmp_path / "a.mhd"
        )


def test_vtk_library_reads_our_files(tmp_path, phase_image):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    io.write_vti(tmp_path / "a.vti", phase_image)
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(tmp_path / "a.vti"))
    reader.Update()
    out = reader.GetOutput()
    values = vtk_to_numpy(out.GetCellData().GetArray("phase"))
    assert np.array_equal(values.reshape(phase_image.shape[::-1]).T, phase_image.data)
    np.testing.assert_allclose(out.GetOrigin(), phase_image.origin)


# -- voxel meshes ----------------------------------------------------------------------------


def _hex_volumes(mesh):
    """Signed volume of each hexahedron (six tetrahedra through the 0-6 diagonal)."""
    x = mesh.nodes[mesh.elements]
    tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    vol = 0.0
    for a, b, c, d in tets:
        vol = (
            vol
            + np.einsum(
                "ij,ij->i", x[:, b] - x[:, a], np.cross(x[:, c] - x[:, a], x[:, d] - x[:, a])
            )
            / 6
        )
    return vol


def test_voxel_mesh_structure(phase_image):
    full = voxel_mesh(phase_image)
    nx, ny, nz = phase_image.shape
    assert full.n_elements == nx * ny * nz and full.n_nodes == (nx + 1) * (ny + 1) * (nz + 1)
    np.testing.assert_allclose(_hex_volumes(full), np.prod(phase_image.spacing))
    assert len(full.node_sets["x0"]) == (ny + 1) * (nz + 1)
    pairs = full.periodic_pairs(0)
    d = full.nodes[pairs[:, 1]] - full.nodes[pairs[:, 0]]
    np.testing.assert_allclose(d, [[nx * phase_image.spacing[0], 0, 0]] * len(pairs))
    assert len(full.boundary_faces("z1")) == nx * ny
    solid = voxel_mesh(phase_image, phases=[1])
    assert solid.n_elements == int(phase_image.data.sum())
    assert solid.volume() == pytest.approx(phase_image.data.sum() * phase_image.voxel_volume)
    assert np.all(solid.phase == 1) and set(solid.element_sets()) == {"phase_1"}
    assert HEX_CORNERS.shape == (8, 3)
    with pytest.raises(ValueError):
        voxel_mesh(phase_image, phases=[7])


def test_mesh_files(tmp_path, phase_image):
    labels = phase_image.with_data(
        np.arange(phase_image.n_voxels).reshape(phase_image.shape) % 5, name="cell"
    )
    mesh = voxel_mesh(phase_image, labels=labels)
    io.save_mesh(mesh, tmp_path / "m.inp", label_sets=True)
    text = (tmp_path / "m.inp").read_text(encoding="utf-8")
    assert (
        text.count("*ELEMENT") == 2 and "*ELSET, ELSET=CELL_4" in text and "*NSET, NSET=Z1" in text
    )
    first_node = text.split("*NODE\n")[1].split("\n")[0]
    np.testing.assert_allclose([float(v) for v in first_node.split(",")[1:]], mesh.nodes[0])
    io.save_mesh(mesh, tmp_path / "m.msh")
    msh = io.read_msh(tmp_path / "m.msh")
    np.testing.assert_allclose(msh["nodes"], mesh.nodes)
    hexes = msh["elements"][5]
    assert np.array_equal(hexes[:, 2:] - 1, mesh.elements)
    assert np.array_equal(hexes[:, 0] - 1, mesh.phase)
    assert len(msh["elements"][3]) == sum(len(mesh.boundary_faces(f)) for f in mesh.node_sets)
    io.save_mesh(mesh, tmp_path / "m.vtu")
    vtu = io.read_vtu(tmp_path / "m.vtu")
    assert np.array_equal(vtu["cells"], mesh.elements)
    assert np.array_equal(vtu["cell_data"]["cell"], mesh.label)
    io.write_periodic_pairs(tmp_path / "p.csv", mesh)
    lines = (tmp_path / "p.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "axis,node_lower,node_upper" and len(lines) > 1


def test_gmsh_reads_our_meshes(tmp_path, phase_image):
    gmsh = pytest.importorskip("gmsh")
    mesh = voxel_mesh(phase_image, phases=[1])
    io.save_mesh(mesh, tmp_path / "m.msh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(tmp_path / "m.msh"))
        _, tags, _ = gmsh.model.mesh.getElements(3)
        assert len(tags[0]) == mesh.n_elements
    finally:
        gmsh.finalize()


# -- surfaces ------------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["level_set", "phase", "fraction"])
def test_surface_meshes_are_closed(tmp_path, spheres, source):
    pytest.importorskip("skimage")
    if source == "level_set":
        field = spheres.level_set(shape=40)
    elif source == "phase":
        field = spheres.voxelize(shape=40)
    else:
        field = spheres.voxelize(shape=40, supersampling=2)
    surf = surface_mesh(field)
    assert np.all(edge_face_counts(surf.faces, surf.vertices) == 2)
    solid = spheres.voxelize(shape=80, supersampling=2).data.mean() * spheres.domain.volume
    assert surf.volume() == pytest.approx(solid, rel=0.03)
    lo, hi = spheres.domain.lower, spheres.domain.upper
    assert np.all(surf.vertices >= lo - 1e-9) and np.all(surf.vertices <= hi + 1e-9)
    io.save_mesh(surf, tmp_path / "s.stl")
    back = io.read_stl(tmp_path / "s.stl")
    assert len(back.faces) == len(surf.faces)
    assert back.volume() == pytest.approx(surf.volume(), rel=1e-5)


# -- geometry ----------------------------------------------------------------------------------


def test_geometry_files(tmp_path, spheres):
    io.save_microstructure(spheres, tmp_path / "g.json")
    back = io.load_microstructure(tmp_path / "g.json")
    assert np.array_equal(back.spheres.centers, spheres.spheres.centers)
    assert (
        json.loads((tmp_path / "g.json").read_text(encoding="utf-8"))["kind"] == "primitive_union"
    )
    (csv_path,) = io.save_microstructure(spheres, tmp_path / "g.csv")
    table = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(table[:, 3], spheres.spheres.radii)
    io.save_microstructure(spheres, tmp_path / "g.geo", phase=0)
    geo = (tmp_path / "g.geo").read_text(encoding="utf-8")
    assert 'SetFactory("OpenCASCADE");' in geo and "BooleanDifference" in geo
    assert geo.count("Sphere(") >= len(spheres.spheres)
    cells = mg.models.Voronoi(n_cells=10, crystal_orientations=True).generate(mg.Domain.cube(1.0))
    (cells_csv,) = io.save_microstructure(cells, tmp_path / "c.csv")
    assert "phi1" in cells_csv.read_text(encoding="utf-8").splitlines()[0]
    with pytest.raises(ValueError, match="Gmsh scripts"):
        io.save_microstructure(cells, tmp_path / "c.geo")


def test_gmsh_meshes_geo_scripts(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    micro = mg.models.BooleanCylinders(
        radius=0.25, length=2.0, volume_fraction=0.05, caps="round"
    ).generate(mg.Domain.cube(3.0), seed=1)
    io.save_microstructure(micro, tmp_path / "f.geo", mesh_size=0.3)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(tmp_path / "f.geo"))
        gmsh.model.mesh.generate(3)
        volume = sum(gmsh.model.occ.getMass(d, t) for d, t in gmsh.model.getEntities(3))
    finally:
        gmsh.finalize()
    expected = micro.voxelize(voxel_size=0.02).data.mean() * 27.0
    assert volume == pytest.approx(expected, rel=0.02)  # caps are enlarged by 1 %
