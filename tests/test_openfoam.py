"""The meshes and cases of OpenFOAM (morphogen.io.openfoam): the polyMesh of voxels and of
conforming tetrahedra checked on its own (closed cells of the right volumes, the order and
orientation of the faces, the patches, the cyclic pairs), read back, the diffusion case run
by a stand-in of laplacianFoam (tests/fake_openfoam.py) against finite volumes on the
voxels of the image, the flow case and the reading of the results."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as splinalg

import fake_openfoam as fof
import morphogen as mg
from helpers import box_mesh
from morphogen import cli, io, pipeline
from morphogen.config import ConfigError, load_config
from morphogen.io.openfoam import (
    FACES,
    WALLS,
    polymesh,
    read_openfoam,
    read_polymesh,
    write_openfoam,
    write_polymesh,
)


def _image(shape=(7, 6, 5), fraction=0.35, seed=0, spacing=(0.5, 0.4, 0.3)):
    rng = np.random.default_rng(seed)
    data = (rng.random(shape) < fraction).astype(np.uint8)  # phase 0: the pores
    return mg.VoxelImage(data, spacing=spacing, origin=(1.0, 2.0, 3.0), periodic=True)


def _check(mesh):
    """The polyMesh on its own: closed cells, owner < neighbour in the upper-triangular
    order, normals from the owner to the neighbour and outward, contiguous patches."""
    n_int = mesh.n_internal
    s = mesh.face_areas()
    closed = np.zeros((mesh.n_cells, 3))
    np.add.at(closed, mesh.owner, s)
    np.add.at(closed, mesh.neighbour, -s[:n_int])
    assert np.abs(closed).max() <= 1e-12 * np.abs(s).max()
    o, n = mesh.owner[:n_int], mesh.neighbour
    assert np.all(o < n)
    assert np.all((np.diff(o) > 0) | ((np.diff(o) == 0) & (np.diff(n) > 0)))
    vol = mesh.cell_volumes()
    assert np.all(vol > 0)
    centres = np.zeros((mesh.n_cells, 3))
    count = np.zeros(mesh.n_cells)
    fc = mesh.face_centres()
    np.add.at(centres, mesh.owner, fc)
    np.add.at(count, mesh.owner, 1)
    np.add.at(centres, n, fc[:n_int])
    np.add.at(count, n, 1)
    centres /= count[:, None]
    assert np.all(np.einsum("ij,ij->i", s[:n_int], centres[n] - centres[o]) > 0)
    bo = mesh.owner[n_int:]
    assert np.all(np.einsum("ij,ij->i", s[n_int:], fc[n_int:] - centres[bo]) > 0)
    start = n_int
    for p in mesh.patches:
        assert p.start == start
        start += p.size
    assert start == len(mesh.faces)
    return vol


def _check_cyclic(mesh, axis, length):
    lo, hi = mesh.patch(FACES[2 * axis]), mesh.patch(FACES[2 * axis + 1])
    assert lo.type == hi.type == "cyclic" and lo.size == hi.size
    assert lo.entries["neighbourPatch"] == hi.name and hi.entries["neighbourPatch"] == lo.name
    a = mesh.points[mesh.faces[lo.start : lo.start + lo.size]]
    b = mesh.points[mesh.faces[hi.start : hi.start + hi.size]]
    shift = np.zeros(3)
    shift[axis] = length
    m = mesh.faces.shape[1]
    reverse = [0, *range(m - 1, 0, -1)]  # the reverse order from the same first point
    np.testing.assert_allclose(b, (a + shift)[:, reverse], atol=1e-12)
    sep = np.array(lo.entries["separationVector"].strip("()").split(), dtype=float)
    np.testing.assert_allclose(sep, a.mean(axis=(0, 1)) - b.mean(axis=(0, 1)), atol=1e-12)


def test_voxel_polymesh():
    image = _image()
    pores = np.asarray(image.data) == 0
    for cyclic in ((), (1, 2), (0, 1, 2)):
        mesh = polymesh(image, (0,), cyclic=cyclic)
        vol = _check(mesh)
        np.testing.assert_allclose(vol, np.prod(image.spacing), rtol=1e-12)
        assert mesh.n_cells == pores.sum()
        np.testing.assert_array_equal(mesh.zones["phase0"], np.arange(mesh.n_cells))
        # the walls: the faces between a pore and a solid voxel, through the cyclic faces
        walls = sum(np.sum(pores & ~np.roll(pores, d, axis=a)) for a in range(3) for d in (1, -1))
        across = sum(
            np.sum(np.take(pores, 0, a) & ~np.take(pores, -1, a))
            + np.sum(np.take(pores, -1, a) & ~np.take(pores, 0, a))
            for a in range(3)
            if a not in cyclic
        )
        assert mesh.patch(WALLS).size == walls - across
        for a in range(3):
            if a in cyclic:
                _check_cyclic(mesh, a, image.spacing[a] * image.shape[a])
            else:
                assert mesh.patch(FACES[2 * a]).size == np.take(pores, 0, a).sum()
    with pytest.raises(ValueError, match="either cyclic or a symmetry"):
        polymesh(image, (0,), cyclic=(1,), symmetry=(1,))


def test_tetrahedral_polymesh():
    def phase(c):
        return (np.linalg.norm(c - [0.5, 0.6, 0.45], axis=1) < 0.3).astype(int)

    mesh = box_mesh((4, 3, 3), upper=(1.0, 1.2, 0.9), phase=phase, perturb=0.2)
    for cyclic in ((), (1, 2), (0, 1, 2)):
        poly = polymesh(mesh, (0,), cyclic=cyclic)
        vol = _check(poly)
        np.testing.assert_allclose(
            np.sort(vol), np.sort(np.abs(mesh.volumes()[mesh.phase == 0])), rtol=1e-12
        )
        for a in cyclic:
            _check_cyclic(poly, a, mesh.size[a])
        assert poly.patch(WALLS).type == "wall" and poly.patch(WALLS).size > 0
    both = polymesh(mesh.quadratic(), (0, 1))  # the corners of 10-node elements
    _check(both)
    assert both.patch(WALLS).size == 0
    assert sorted(both.zones) == ["phase0", "phase1"]
    assert len(both.zones["phase1"]) == int((mesh.phase == 1).sum())
    # whole periodic cells: their boundary is a patch, OUTER
    micro = mg.models.Voronoi(n_cells=5).generate(mg.Domain.cube(1.0, periodic=True), seed=2)
    from morphogen.mesh.conforming import conforming_mesh

    cells = conforming_mesh(micro, size=0.35, cells="whole")
    poly = polymesh(cells, np.unique(cells.phase))
    _check(poly)
    assert [p.name for p in poly.patches] == ["OUTER", WALLS]
    assert poly.patch("OUTER").size == len(cells.outer_faces())
    with pytest.raises(ValueError, match="whole cells"):
        polymesh(cells, np.unique(cells.phase), cyclic=(0,))


def test_files_read_back(tmp_path):
    mesh = polymesh(_image(), (0,), cyclic=(1, 2))
    files = write_polymesh(tmp_path, mesh)
    assert [f.name for f in files] == [
        "points",
        "faces",
        "owner",
        "neighbour",
        "boundary",
        "cellZones",
    ]
    owner = (tmp_path / "constant" / "polyMesh" / "owner").read_text()
    assert "class       labelList;" in owner and 'location    "constant/polyMesh";' in owner
    assert (
        f'note        "nPoints:{len(mesh.points)}  nCells:{mesh.n_cells}  '
        f'nFaces:{len(mesh.faces)}  nInternalFaces:{mesh.n_internal}";'
    ) in owner
    back = read_polymesh(tmp_path)
    np.testing.assert_allclose(back.points, mesh.points, rtol=1e-11)
    for name in ("faces", "owner", "neighbour"):
        np.testing.assert_array_equal(getattr(back, name), getattr(mesh, name))
    assert [(p.name, p.type, p.start, p.size) for p in back.patches] == [
        (p.name, p.type, p.start, p.size) for p in mesh.patches
    ]
    assert back.patch("YMIN").entries["transform"] == "translational"
    np.testing.assert_array_equal(back.zones["phase0"], mesh.zones["phase0"])


def _direct_diffusivity(image, cyclic):
    """The effective diffusivity along x of the pores of an image by finite volumes on its
    voxels (the flux of two points between neighbours, half a voxel to the inlet and the
    outlet, no flux through the solid; the lateral faces periodic or closed)."""
    pores = np.asarray(image.data) == 0
    h = np.asarray(image.spacing, dtype=float)
    shape = pores.shape
    index = np.arange(pores.size).reshape(shape)
    rows, cols, vals = [], [], []
    rhs = np.zeros(pores.size)
    for a in range(3):
        area = np.prod(h) / h[a]
        coef = area / h[a]
        nb = np.roll(index, -1, axis=a)
        both = pores & np.roll(pores, -1, axis=a)
        if a not in cyclic:  # no coupling across the last layer
            last = [slice(None)] * 3
            last[a] = -1
            both[tuple(last)] = False
        i, j = index[both], nb[both]
        rows += [i, j, i, j]
        cols += [i, j, j, i]
        vals += [np.full(i.size, coef)] * 2 + [np.full(i.size, -coef)] * 2
    area = h[1] * h[2]
    for layer, value in ((0, 1.0), (-1, 0.0)):  # T = 1 at the inlet, 0 at the outlet
        i = index[layer][pores[layer]]
        rows.append(i)
        cols.append(i)
        vals.append(np.full(i.size, area / (0.5 * h[0])))
        rhs[i] += area / (0.5 * h[0]) * value
    k = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(pores.size, pores.size),
    )
    k = k + sparse.identity(pores.size) * 1e-14  # the solid voxels and the closed pores
    t = splinalg.spsolve(k.tocsc(), rhs)
    outlet = index[-1][pores[-1]]
    flux = np.sum(area / (0.5 * h[0]) * t[outlet])
    length, section = shape[0] * h[0], shape[1] * h[1] * shape[2] * h[2]
    return flux * length / section


def test_diffusion_case(tmp_path):
    """The case of the diffusion run by the stand-in of laplacianFoam: straight channels
    give their fraction exactly; a random medium the effective diffusivity of finite
    volumes on its voxels, with closed or periodic lateral faces."""
    data = np.ones((6, 5, 4), np.uint8)
    data[:, 1, 1] = data[:, 3, 2] = data[:, 0, 3] = 0
    channels = mg.VoxelImage(data, spacing=(0.5, 0.4, 0.3), periodic=True)
    for lateral in ("symmetryPlane", "cyclic"):
        case = tmp_path / f"channels_{lateral}"
        write_openfoam(
            case, channels, kind="diffusion", lateral=lateral, diffusivity=2.0, difference=3.0
        )
        result = fof.solve(case)
        run = read_openfoam(case)
        assert run.properties["relative_diffusivity"] == pytest.approx(0.15, rel=1e-12)
        assert run.properties["effective_diffusivity"] == pytest.approx(0.3, rel=1e-12)
        assert result["flux"]["XMIN"] == pytest.approx(-result["flux"]["XMAX"], rel=1e-12)
    image = _image((10, 8, 8), fraction=0.3, seed=1, spacing=(1.0, 1.0, 1.0))
    for lateral, cyclic in (("symmetryPlane", ()), ("cyclic", (1, 2))):
        case = tmp_path / f"random_{lateral}"
        export = write_openfoam(case, image, kind="diffusion", lateral=lateral)
        result = fof.solve(case)
        run = read_openfoam(case)
        direct = _direct_diffusivity(image, cyclic)
        assert run.properties["relative_diffusivity"] == pytest.approx(direct, rel=1e-10)
        assert result["flux"]["XMIN"] == pytest.approx(-result["flux"]["XMAX"], rel=1e-10)
        assert export.manifest["connected_fraction"] <= export.manifest["fraction"]
        text = (case / "system" / "controlDict").read_text()
        assert "application     laplacianFoam;" in text and "volIntegrate" in text


def test_closed_pores_left_out(tmp_path):
    data = np.ones((6, 5, 5), np.uint8)
    data[:, 2, 2] = 0  # a channel from the inlet to the outlet
    data[2, 2, 3] = 0  # a dead end on it: kept
    data[3, 0, 0] = data[3, 0, 1] = 0  # a closed pore: left out
    image = mg.VoxelImage(data, spacing=(1.0, 1.0, 1.0))
    export = write_openfoam(tmp_path, image, kind="flow", lateral="symmetryPlane")
    assert export.mesh.n_cells == 7
    assert export.manifest["fraction"] == pytest.approx(9 / data.size)
    whole = write_openfoam(
        tmp_path / "all", image, kind="flow", lateral="symmetryPlane", connected=False
    )
    assert whole.mesh.n_cells == 9
    with pytest.raises(ValueError, match="joins the inlet"):
        write_openfoam(tmp_path / "none", image, kind="flow", axis=1, lateral="symmetryPlane")


def test_flow_case(tmp_path):
    mesh = box_mesh(
        (4, 3, 3),
        upper=(1.0, 1.2, 0.9),
        phase=lambda c: (np.linalg.norm(c - 0.5, axis=1) < 0.3).astype(int),
    )
    export = write_openfoam(tmp_path, mesh, kind="flow", viscosity=1e-3, pressure_drop=2.0)
    assert export.manifest["lateral"] == "cyclic"  # a periodic mesh with matching faces
    names = sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in export.files)
    for name in (
        "0/U",
        "0/p",
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "constant/transportProperties",
        "constant/physicalProperties",
        "constant/momentumTransport",
        "constant/turbulenceProperties",
        "Allrun",
        "morphogen_openfoam.json",
    ):
        assert name in names
    u = (tmp_path / "0" / "U").read_text()
    p = (tmp_path / "0" / "p").read_text()
    for patch in export.mesh.patches:  # every patch has its condition
        assert f"    {patch.name}\n" in u and f"    {patch.name}\n" in p
    assert "type            noSlip;" in u
    assert "uniform 2;" in p and "uniform 0;" in p
    control = (tmp_path / "system" / "controlDict").read_text()
    assert (
        "application     simpleFoam;" in control
        and "solver          incompressibleFluid;" in control
    )
    assert "name            XMAX;" in control and "patch           XMAX;" in control
    assert "nNonOrthogonalCorrectors 2;" in (tmp_path / "system" / "fvSolution").read_text()
    assert (
        "nu              [0 2 -1 0 0 0 0] 0.001;"
        in (tmp_path / "constant" / "transportProperties").read_text()
    )
    manifest = json.loads((tmp_path / "morphogen_openfoam.json").read_text())
    assert manifest["area"] == pytest.approx(1.08) and manifest["length"] == 1.0
    # the outputs of the function object: the last flow rate gives the permeability
    folder = tmp_path / "postProcessing" / "flowRate" / "0"
    folder.mkdir(parents=True)
    (folder / "surfaceFieldValue.dat").write_text(
        "# Region type : patch XMAX\n# Faces  : 18\n# Time\tsum(phi)\n1\t1.0e-4\n57\t-2.16e-3\n"
    )
    run = read_openfoam(tmp_path)
    assert run.properties["flow_rate"] == pytest.approx(2.16e-3)
    assert run.properties["permeability"] == pytest.approx(1e-3 * 2.16e-3 * 1.0 / (1.08 * 2.0))
    assert run.properties["mean_velocity"] == pytest.approx(2.16e-3 / 1.08)


@pytest.mark.skipif(importlib.util.find_spec("vtk") is None, reason="VTK reads the polyMesh")
def test_read_by_vtk(tmp_path):
    """VTK's reader of OpenFOAM (vtkOpenFOAMReader): the cells, points, patches and volumes
    of a case of voxels and of one of tetrahedra, with cyclic patches."""
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy

    tets = box_mesh(
        (4, 3, 3),
        upper=(1.0, 1.2, 0.9),
        phase=lambda c: (np.linalg.norm(c - 0.5, axis=1) < 0.3).astype(int),
    )
    for name, sample in (("voxels", _image()), ("tetrahedra", tets)):
        export = write_openfoam(tmp_path / name, sample, kind="flow", lateral="cyclic")
        reader = vtk.vtkOpenFOAMReader()
        reader.SetFileName(str(tmp_path / name / f"{name}.foam"))
        reader.Update()
        reader.EnableAllPatchArrays()
        reader.Update()
        blocks = reader.GetOutput()
        internal = blocks.GetBlock(0)
        assert internal.GetNumberOfCells() == export.mesh.n_cells
        assert internal.GetNumberOfPoints() == len(export.mesh.points)
        patches = blocks.GetBlock(1)
        read = {
            patches.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()): patches.GetBlock(
                i
            ).GetNumberOfCells()
            for i in range(patches.GetNumberOfBlocks())
        }
        assert read == {p.name: p.size for p in export.mesh.patches if p.size}
        size = vtk.vtkCellSizeFilter()
        size.SetInputData(internal)
        size.SetComputeVolume(True)
        size.Update()
        volume = vtk_to_numpy(size.GetOutput().GetCellData().GetArray("Volume"))
        assert volume.sum() == pytest.approx(export.mesh.cell_volumes().sum(), rel=1e-6)


def test_runs_and_commands(tmp_path, capsys):
    """outputs.openfoam in a run (the pores of the image of each realization, or the
    conforming mesh), its errors; the commands morphogen openfoam export and read."""
    cfg = {
        "model": {"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.3},
        "domain": {"size": 2.0, "periodic": True},
        "seed": 1,
        "voxels": {"shape": 10},
        "outputs": {"directory": str(tmp_path / "out"), "name": "sph",
                    "openfoam": {"kind": "diffusion", "diffusivity": 2.0}},
    }  # fmt: skip
    outdir = pipeline.run(load_config(cfg))
    case = outdir / "sph_0000_openfoam"
    manifest = json.loads((case / "morphogen_openfoam.json").read_text())
    assert manifest["kind"] == "diffusion" and manifest["lateral"] == "cyclic"
    fof.solve(case)
    run = read_openfoam(case)
    assert 0.0 < run.properties["relative_diffusivity"] < manifest["fraction"]
    for change, match in [
        ({"outputs": {**cfg["outputs"], "openfoam": {"kind": "boiling"}}}, "kind"),
        ({"outputs": {**cfg["outputs"], "openfoam": {"sample": "mesh"}}}, "mesh"),
        ({"outputs": {**cfg["outputs"], "openfoam": {"axis": 3}}}, "axis"),
    ]:
        with pytest.raises(ConfigError, match=match):
            load_config({**cfg, **change})
    # the commands: an image exported (a flow), its outputs read back
    image = _image()
    io.save_image(image, tmp_path / "pores.vti")
    assert cli.main(["openfoam", "export", str(tmp_path / "pores.vti"), "--periodic",
                     "--viscosity", "0.01", "-o", str(tmp_path / "flow")]) == 0  # fmt: skip
    assert "./Allrun" in capsys.readouterr().out
    assert cli.main(["openfoam", "read", str(tmp_path / "flow")]) == 0
    assert "no outputs" in capsys.readouterr().out
    folder = tmp_path / "flow" / "postProcessing" / "flowRate" / "0"
    folder.mkdir(parents=True)
    (folder / "surfaceFieldValue.dat").write_text("# Time sum(phi)\n10 0.25\n")
    assert cli.main(["openfoam", "read", str(tmp_path / "flow"), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    length, area = 7 * 0.5, 6 * 0.4 * 5 * 0.3
    assert out["properties"]["permeability"] == pytest.approx(0.01 * 0.25 * length / area)
    assert out["lateral"] == "cyclic"
