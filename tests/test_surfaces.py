"""The 3D view of the interface: the surfaces of morphologies and simulations
(:mod:`morphogen.viz.surfaces`), the endpoint ``/api/surface`` and the functions of its
script ``viewer.js`` (run by Node)."""

import base64
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from importlib import resources

import numpy as np
import pytest

import morphogen as mg
from morphogen import io
from morphogen.gui import api
from morphogen.gui.server import GuiServer
from morphogen.mesh.conforming import background_mesh
from morphogen.mesh.tetra import TetMesh
from morphogen.viz import surfaces as sf
from morphogen.viz.sections import level_set_measures

NODE = shutil.which("node")
VIEWER = resources.files("morphogen") / "gui" / "viewer.js"


def _array(text, dtype, columns):
    return np.frombuffer(base64.b64decode(text), dtype=dtype).reshape(-1, columns)


def _edges(triangles):
    """The edges of triangles (directed) and their counts as undirected edges."""
    directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    undirected, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    return directed, undirected, counts


@pytest.fixture(scope="module")
def spheres():
    return mg.models.HardcoreSpheres(radius=0.8, volume_fraction=0.2).generate(
        mg.Domain.cube(5.0), 3
    )


# -- morphologies ---------------------------------------------------------------------------------


def test_spheres_closed_outward_and_measured(spheres):
    """Marching cubes of the level set of hard-core spheres: closed surfaces (open only on
    the faces of the box, where the volume draws the phases), unit normals out of the
    spheres, the area of the interface that the measures of the level set find, the
    phases of the volume those of the voxels."""
    pytest.importorskip("skimage")
    surface = sf.morphology_surface(spheres, 96)
    assert surface.method == "marching cubes" and surface.resolution == 96
    assert not surface.reduced and surface.n_triangles <= sf.MAX_TRIANGLES
    assert surface.groups[:, :2].tolist() == [[1, 0]]
    data = sf.encode(surface)
    positions = _array(data["positions"], "<f4", 3)
    normals = _array(data["normals"], "<f4", 3)
    triangles = _array(data["indices"], "<u4", 3).astype(np.int64)
    assert len(triangles) == data["triangles"] == surface.n_triangles
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    # closed: every edge in two triangles, once each way, but on the faces of the box
    directed, undirected, counts = _edges(triangles)
    assert counts.max() == 2
    ends = positions[undirected[counts == 1]].astype(float)
    on_face = np.zeros(len(ends), dtype=bool)
    for a in range(3):
        for value in (0.0, 5.0):
            on_face |= np.all(np.abs(ends[:, :, a] - value) < 1e-5, axis=1)
    assert on_face.all() and len(ends) > 0  # spheres cut by the faces of the periodic box
    assert len(np.unique(directed, axis=0)) == len(directed)  # consistently oriented
    # the normals point out of the spheres (phase 1), towards the matrix
    shift = positions[:, None, :] - spheres.spheres.centers[None]
    shift -= 5.0 * np.rint(shift / 5.0)
    nearest = shift[np.arange(len(shift)), np.argmin(np.linalg.norm(shift, axis=2), axis=1)]
    radial = nearest / np.linalg.norm(nearest, axis=1, keepdims=True)
    assert np.mean(np.einsum("ij,ij->i", radial, normals) > 0.9) > 0.97
    # the area: that measured on the level set at the same resolution, close to the spheres'
    measured = level_set_measures(spheres, 96)["specific_surface"] * spheres.domain.volume
    assert surface.area(1, 0) == pytest.approx(measured, rel=2e-3)
    theory = len(spheres.spheres.radii) * 4.0 * np.pi * 0.8**2
    assert surface.area(1, 0) == pytest.approx(theory, rel=0.01)
    # the volume: the level set at the cell centres, negative in the voxels of phase 1
    volume = data["volume"]
    assert volume["kind"] == "level_set" and volume["shape"] == [96, 96, 96]
    assert volume["box"] and volume["periodic"]
    q = np.frombuffer(base64.b64decode(volume["data"]), np.uint8).reshape((96, 96, 96), order="F")
    phases = spheres.voxelize(shape=96).data
    assert np.mean((q < 128) == (phases == 1)) > 0.9999
    assert data["regions"]["phase"] == [0, 1] and data["regions"]["label"] is None
    assert data["regions"]["color"][1] == "#1f4e79"  # the colours of the sections


def test_budget_and_voxel_faces(spheres):
    """A budget of triangles makes the grid coarser, the finest within it (thin struts
    make more triangles per cube); the faces of the voxels (without scikit-image, or
    asked for) lie on the grid, with the normals of the axes."""
    pytest.importorskip("skimage")
    small = sf.morphology_surface(spheres, 96, max_triangles=20_000)
    assert small.n_triangles <= 20_000 and small.reduced and small.resolution < 96
    assert small.volume.data.shape == (small.resolution,) * 3
    octet = mg.models.StrutLattice(cell="octet", cell_size=1.0, volume_fraction=0.1)
    lattice = sf.morphology_surface(
        octet.generate(mg.Domain.cube(2.0), 1), 128, max_triangles=60_000
    )
    assert lattice.method == "marching cubes" and lattice.reduced
    assert 0.75 * 60_000 < lattice.n_triangles <= 60_000  # a fine grid, not simplified
    _, _, counts = _edges(lattice.triangles)
    assert counts.max() == 2
    voxels = sf.morphology_surface(spheres, 32, method="voxels")
    assert voxels.method == "voxels" and voxels.groups[:, :2].tolist() == [[1, 0]]
    h = 5.0 / 32
    np.testing.assert_allclose(voxels.positions / h, np.rint(voxels.positions / h), atol=1e-9)
    assert set(map(tuple, np.abs(voxels.normals).astype(int))) == {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
    phase = spheres.voxelize(shape=32).data
    faces = sum(int(np.count_nonzero(np.diff(phase, axis=a))) for a in range(3))
    assert voxels.n_triangles == 2 * faces  # two triangles per face between the phases
    # the normals point from phase 1 to phase 0: the voxel behind a face is of phase 1
    centre = voxels.positions[voxels.triangles].mean(axis=1)
    normal = voxels.normals[voxels.triangles[:, 0]]
    behind = np.floor((centre - 0.5 * h * normal) / h).astype(int)
    assert np.all(phase[tuple(behind.T)] == 1)
    data = sf.encode(voxels)  # the compact form of voxel faces: nodes and directions
    assert data["positions"] is None and data["lattice"]["nodes"] == [33, 33, 33]
    assert data["volume"]["kind"] == "regions"
    with pytest.raises(ValueError, match="resolution"):
        sf.morphology_surface(spheres, 4)


def test_polycrystal_and_labels():
    """Polycrystals: the exact facets of their cells, between cells (the larger inside)
    and on the faces of the box, with the phase and label of each cell; with grain
    boundaries, the faces between voxels of different (cell, phase) regions."""
    domain = mg.Domain.cube(1.0)
    grains = mg.models.Voronoi(n_cells=30, phase_fractions=[0.5, 0.5]).generate(domain, 2)
    surface = sf.morphology_surface(grains, 32)
    assert surface.method == "cells"
    inside, outside = surface.groups[:, 0], surface.groups[:, 1]
    assert np.all((inside > outside) | (outside == sf.EXTERIOR))
    np.testing.assert_array_equal(surface.region_phase, grains.cell_phases)
    np.testing.assert_array_equal(surface.region_label, np.arange(30))
    box = sum(surface.area(int(g[0]), sf.EXTERIOR) for g in surface.groups if g[1] < 0)
    assert box == pytest.approx(6.0, rel=1e-9)  # the faces of the unit cube
    complex_ = grains.cell_complex()
    assert surface.area() - box == pytest.approx(complex_.grain_boundary_area(), rel=1e-9)
    # each triangle between its two cells: a point behind it in the inside cell
    p = surface.positions[surface.triangles]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    owner = np.repeat(np.arange(len(surface.groups)), surface.groups[:, 5])
    behind = grains.cell_of(p.mean(axis=1) - 1e-6 * normal)
    assert np.mean(behind == surface.groups[owner, 0]) > 0.999
    assert surface.volume.kind == "regions" and not surface.volume.box  # sections only
    data = sf.encode(surface)
    assert data["regions"]["label"] == list(range(30))
    assert len(data["regions"]["label_color"]) == 30
    # grain boundaries of a phase of their own: voxel faces between (cell, phase) regions
    walls = mg.models.Voronoi(n_cells=12, boundary_thickness=0.06, boundary_phase=0).generate(
        domain, 2
    )
    voxels = sf.morphology_surface(walls, 24)
    assert voxels.method == "voxels" and voxels.region_label is not None
    assert set(voxels.region_phase.tolist()) == {0, 1}
    assert np.all(voxels.groups[:, 0] > voxels.groups[:, 1])
    regions = voxels.volume.data
    assert regions.shape == (24, 24, 24)
    np.testing.assert_array_equal(
        voxels.region_phase[regions], walls.voxelize(shape=24).data.astype(np.int64)
    )


# -- simulations ----------------------------------------------------------------------------------


def _fe_result(folder):
    """A finite element result: a cube of edge 2 in 4 x 4 x 4 cubes of six tetrahedra, an
    inclusion (phase 1) around its centre, an affine displacement, element fields and a
    linear nodal field."""
    nodes, tets, _, _ = background_mesh((4, 4, 4), (0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    centre = nodes[tets].mean(axis=1)
    phase = (np.linalg.norm(centre - 1.0, axis=1) < 0.55).astype(np.int64)
    mesh = TetMesh(nodes, tets, phase, (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), periodic=True)
    u = np.stack([0.01 * nodes[:, 1], 0.02 * nodes[:, 2], np.zeros(len(nodes))], axis=1)
    cells = {
        "von_mises": np.where(phase == 1, 5.0, 1.0).astype(np.float32),
        "stress_12": centre[:, 0].astype(np.float32),
    }
    path = folder / "sample_0000_shear_modulus_periodic.vtu"
    nodal = {"displacement": u, "temperature": nodes[:, 0] + 2.0 * nodes[:, 1]}
    io.write_vtu_tet(path, mesh, cells, nodal)
    (folder / "results.csv").write_text("realization,shear_modulus.periodic\n0,1.0\n")
    return mesh, u


def test_mesh_surface_fields_and_displacement(tmp_path):
    """The faces of a mesh on the box and between its phases, the element field averaged
    at the nodes on each side, the nodal displacement, the usual fields first; the regions
    and the field of the elements containing the cells of a grid, for the sections."""
    session = api.Session(tmp_path)
    key, folder = session.folder()
    mesh, u = _fe_result(folder)
    session.add(key, "simulation", "FE", phases={"0": "matrix", "1": "inclusion"})
    data = api.surface({"key": key, "resolution": 16}, session)
    assert data["method"] == "mesh" and data["kind"] == "simulation"
    assert data["field"]["name"] == "von_mises"
    assert data["field"]["names"][:2] == ["von_mises", "displacement"]
    assert data["field"]["range"] == [1.0, 5.0] and not data["field"]["point"]
    groups = np.array(data["groups"])
    assert sorted(map(tuple, groups[:, :2].tolist())) == [(0, -1), (1, 0)]
    positions = _array(data["positions"], "<f4", 3).astype(float)
    triangles = _array(data["indices"], "<u4", 3).astype(np.int64)
    p = positions[triangles]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    owner = np.repeat(np.arange(len(groups)), groups[:, 5])
    assert area[groups[owner, 1] == -1].sum() == pytest.approx(24.0)  # the faces of the box
    # the field on each side: that of the inclusion inside, of the matrix outside
    values = _array(data["field"]["values"], "<f4", 2)
    vertex_group = np.repeat(np.arange(len(groups)), groups[:, 3])
    interface = groups[vertex_group, 1] == 0
    np.testing.assert_allclose(values[interface], [[5.0, 1.0]] * int(interface.sum()))
    np.testing.assert_allclose(values[~interface], 1.0)
    # the displacement of the nodes, and its magnification (a tenth of the box at most)
    shown = _array(data["displacement"], "<f4", 3)
    expected = np.stack([0.01 * positions[:, 1], 0.02 * positions[:, 2], 0 * positions[:, 0]], 1)
    np.testing.assert_allclose(shown, expected, atol=1e-6)
    assert data["magnification"] == pytest.approx(0.1 * 2.0 / np.linalg.norm(u, axis=1).max())
    nodal = api.surface({"key": key, "field": "displacement"}, session)["field"]
    assert nodal["point"] and nodal["range"][1] == pytest.approx(np.linalg.norm(u, axis=1).max())
    # the sections: the phase of the element holding the centre of each cell, its field
    volume = data["volume"]
    assert volume["kind"] == "regions" and not volume["box"] and volume["shape"] == [16] * 3
    cells = np.frombuffer(base64.b64decode(volume["data"]), np.uint8).reshape((16,) * 3, order="F")
    assert cells[8, 8, 8] == 1 and cells[0, 0, 0] == 0 and set(np.unique(cells)) == {0, 1}
    fraction = mesh.phase_fractions()[1]
    assert cells.mean() == pytest.approx(fraction, abs=0.03)
    field = data["field"]["volume"]
    q = np.frombuffer(base64.b64decode(field["data"]), np.uint16).reshape((16,) * 3, order="F")
    np.testing.assert_allclose(1.0 + 4.0 * q / 65535, np.where(cells == 1, 5.0, 1.0), atol=1e-4)
    linear = api.surface({"key": key, "resolution": 16, "field": "temperature"}, session)
    lo, hi = linear["field"]["volume"]["range"]
    q = np.frombuffer(base64.b64decode(linear["field"]["volume"]["data"]), np.uint16)
    i, j, _ = (np.indices((16,) * 3) + 0.5) / 8.0  # the centres of the cells
    expected = i + 2.0 * j  # interpolated exactly
    shown = lo + q.reshape((16,) * 3, order="F") / 65535 * (hi - lo)
    np.testing.assert_allclose(shown, expected, atol=1e-3)
    with pytest.raises(ValueError, match="field"):
        api.surface({"key": key, "field": "porosity"}, session)
    # beyond the budget, the mesh is simplified by clustering its vertices
    corners = mesh.elements
    small = sf.mesh_surface(mesh.nodes, corners, mesh.phase, [0, 1], max_triangles=150)
    assert small.reduced and 0 < small.n_triangles <= 150
    assert small.groups[:, 5].sum() == small.n_triangles


def test_voxel_simulation_surface(tmp_path):
    """A simulation on voxels: the faces between the phases of its image, the field of
    its voxels as a volume quantized over its range."""
    session = api.Session(tmp_path)
    key, folder = session.folder()
    grid = np.indices((8, 8, 8)).sum(axis=0)
    phase = ((np.indices((8, 8, 8)) - 3.5) ** 2).sum(axis=0) < 6
    field = (grid / 21.0).astype(np.float32)
    image = mg.image.VoxelImage(phase.astype(np.uint8), spacing=(0.5,) * 3, name="phase")
    io.save_image(image, folder / "sample_0000.vti")
    io.save_image(
        [image.with_data(field, name="von_mises"), image.with_data(-field, name="stress_12")],
        folder / "sample_0000_shear_modulus_periodic.vti",
    )
    (folder / "results.csv").write_text("realization\n0\n")
    session.add(key, "simulation", "FFT")
    data = api.surface({"key": key, "field": "stress_12"}, session)
    assert data["method"] == "voxels" and data["upper"] == [4.0, 4.0, 4.0]
    assert data["field"]["names"] == ["von_mises", "stress_12"]
    assert data["field"]["values"] is None and data["displacement"] is None
    volume = data["field"]["volume"]
    lo, hi = volume["range"]
    assert (lo, hi) == pytest.approx((-1.0, 0.0))
    q = np.frombuffer(base64.b64decode(volume["data"]), np.uint16).reshape((8, 8, 8), order="F")
    np.testing.assert_allclose(lo + q / 65535 * (hi - lo), -field, atol=1e-4)
    assert [g[:2] for g in data["groups"]] == [[1, 0]]
    regions = np.frombuffer(base64.b64decode(data["volume"]["data"]), np.uint8)
    np.testing.assert_array_equal(regions.reshape((8, 8, 8), order="F"), phase)


# -- the endpoint ----------------------------------------------------------------------------------


def test_surface_endpoint_and_errors(tmp_path):
    """GET /api/surface of the items of the workspace, /viewer.js, and clear errors."""
    server = GuiServer(("127.0.0.1", 0), tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=120) as r:
            return r.status, r.headers.get("Content-Type"), r.read()

    def error(path):
        with pytest.raises(urllib.error.HTTPError) as err:
            get(path)
        return err.value.code, json.loads(err.value.read())["error"]

    try:
        status, kind, script = get("/viewer.js")
        assert status == 200 and kind.startswith("text/javascript")
        assert b"MorphoViewer" in script and b"webgl2" in script
        assert b'src="/viewer.js"' in get("/")[2]
        session = server.session
        item = api.generate(
            {"model": "tpms", "params": {"volume_fraction": 0.3}, "domain": {"size": 1.0},
             "resolution": 24},
            session,
        )["item"]  # fmt: skip
        status, kind, body = get(f"/api/surface?key={item['key']}&resolution=24")
        data = json.loads(body)
        assert status == 200 and kind == "application/json"
        assert data["key"] == item["key"] and data["requested"] == 24
        assert data["triangles"] > 0 and data["budget"] == sf.MAX_TRIANGLES
        assert data["phases"] == item["phases"]
        assert error("/api/surface?key=nothing")[0] == 400
        code, message = error(f"/api/surface?key={item['key']}&resolution=4")
        assert code == 400 and "resolution" in message
        code, message = error(f"/api/surface?key={item['key']}&budget=10")
        assert code == 400 and "budget" in message
        key, _ = session.folder()  # a simulation without fields
        session.add(key, "simulation", "empty")
        code, message = error(f"/api/surface?key={key}")
        assert code == 400 and "nothing to draw" in message
    finally:
        server.shutdown()
        server.server_close()


# -- the script of the view (Node) ----------------------------------------------------------------


@pytest.mark.skipif(NODE is None, reason="Node.js runs the functions of the 3D view")
def test_viewer_functions(spheres):
    """The pure functions of viewer.js: matrices, the section plane, colour maps and
    scales, the decoding of the surfaces (voxel faces from their nodes)."""
    pytest.importorskip("skimage")
    voxels = sf.encode(sf.morphology_surface(spheres, 12, method="voxels"))
    smooth = sf.encode(sf.morphology_surface(spheres, 12))
    script = f"""
const v = require({json.dumps(str(VIEWER))});
const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = {{}};
const m = v.multiply(v.perspective(0.6, 1.5, 0.1, 100), v.lookAt([4, -3, 2], [1, 1, 1], [0, 0, 1]));
out.identity = Array.from(v.multiply(v.identity(), m)).map((x, i) => x - m[i]);
out.centre = v.project(m, [1, 1, 1], 300, 200);
out.right = v.project(m, [1, 1, 1].map((c, i) => c + 0.1 * [0.6, 0.8, 0][i]), 300, 200);
out.behind = v.project(m, [8, -7, 3], 300, 200);
out.eye = v.orbitEye([1, 2, 3], 5, 0.3, 0.4);
out.fits = [1.5, 1, 0.5].map((aspect) => v.fit([0, 0, 0], [6, 6, 1], Math.PI / 4, aspect).distance);
out.planes = [v.clipPlane("y", 0.25, [0, 0, 0], [4, 4, 4], false),
  v.clipPlane("y", 0.25, [0, 0, 0], [4, 4, 4], true), v.clipPlane(2, 1.5, [0, 0, 0], [2, 2, 2]),
  v.clipPlane("none", 0.5, [0, 0, 0], [1, 1, 1])];
out.quad = v.sectionQuad("x", 0.5, [0, 0, 0], [2, 4, 6]);
out.faces = v.boxFaces([0, 0, 0], [1, 2, 3]);
out.colours = ["viridis", "gray", "coolwarm", "nope"].map((n) =>
  [0, 0.5, 1, -1, 2].map((t) => v.colormap(n, t)));
out.table = Array.from(v.colormapTable("magma", 4));
out.ranges = [v.defaultRange(-1, 3, "coolwarm"), v.defaultRange(-1, 3, "viridis"),
  v.defaultRange(1e-12, 2, "viridis"), v.defaultRange(2, 2, "viridis"),
  v.defaultRange(NaN, 1, "jet")];
out.arrays = input.map((d) => {{
  const a = v.surfaceArrays(d);
  return {{positions: Array.from(a.positions), normals: Array.from(a.normals),
    regions: Array.from(a.regions), indices: a.indices.length}};
}});
process.stdout.write(JSON.stringify(out));
"""
    run = subprocess.run(
        [NODE, "-e", script],
        input=json.dumps([voxels, smooth]),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    out = json.loads(run.stdout)
    assert max(abs(x) for x in out["identity"]) < 1e-6
    assert out["centre"][:2] == pytest.approx([150, 100], abs=1e-3)  # the target, centred
    assert -1 < out["centre"][2] < 1
    assert out["right"][0] > 150  # a point to the right of the view
    assert out["behind"] is None  # behind the eye
    eye = np.array(out["eye"])
    assert np.linalg.norm(eye - [1, 2, 3]) == pytest.approx(5)
    assert eye[2] - 3 == pytest.approx(5 * np.sin(0.4))  # z up
    wide, square, tall = out["fits"]  # a view taller than wide frames the box by its width
    radius = np.hypot(np.hypot(6, 6), 1) / 2
    assert wide == pytest.approx(square) == pytest.approx(1.08 * radius / np.sin(np.pi / 8))
    assert tall == pytest.approx(1.08 * radius / np.sin(np.arctan(0.5 * np.tan(np.pi / 8))))
    assert out["planes"] == [[0, 1, 0, -1], [0, -1, 0, 1], [0, 0, 1, -2], [0, 0, 0, 1]]
    quad = np.array(out["quad"])
    assert np.all(quad[:, 0] == 1)
    assert {tuple(q[1:]) for q in quad} == {(0, 0), (4, 0), (4, 6), (0, 6)}
    for face in out["faces"]:  # counter-clockwise from outside
        c = np.array(face["corners"], dtype=float)
        assert np.dot(np.cross(c[1] - c[0], c[2] - c[0]), face["normal"]) > 0
    viridis, gray, coolwarm, unknown = out["colours"]
    assert viridis[0] == [68, 1, 84] and viridis[2] == [253, 231, 37]
    assert viridis[3] == viridis[0] and viridis[4] == viridis[2]  # clamped
    assert gray[1] == [128, 128, 128] and unknown == viridis
    assert coolwarm[1][0] == pytest.approx(221, abs=3)  # the light middle of a diverging map
    assert out["table"][:4] == [0, 0, 4, 255] and out["table"][-4:] == [252, 253, 191, 255]
    assert out["ranges"] == [[-3, 3], [-1, 3], [0, 2], [1.9, 2.1], [0, 1]]
    for payload, decoded in zip((voxels, smooth), out["arrays"], strict=True):
        surface_positions = np.array(decoded["positions"]).reshape(-1, 3)
        assert len(surface_positions) == payload["vertices"]
        assert decoded["indices"] == 3 * payload["triangles"]
        normals = np.reshape(decoded["normals"], (-1, 3))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5)
        regions = np.reshape(decoded["regions"], (-1, 2))
        for inside, outside, first, count, _, _ in payload["groups"]:
            assert np.all(regions[first : first + count] == [inside, outside])
    expected = sf.morphology_surface(spheres, 12, method="voxels").positions
    decoded = np.reshape(out["arrays"][0]["positions"], (-1, 3))
    np.testing.assert_allclose(decoded, expected, atol=1e-6)


@pytest.mark.skipif(NODE is None, reason="Node.js checks the syntax of the script")
def test_viewer_script_and_colormaps():
    """viewer.js is valid JavaScript, served with the page, and its colour maps are those
    of matplotlib (the views of the server) within a few levels."""
    subprocess.run([NODE, "--check", str(VIEWER)], check=True, capture_output=True)
    matplotlib = pytest.importorskip("matplotlib")
    names = list(api.CMAPS)
    run = subprocess.run(
        [NODE, "-e", f"const v = require({json.dumps(str(VIEWER))});"
         f"process.stdout.write(JSON.stringify({json.dumps(names)}.map((n) =>"
         "Array.from({length: 101}, (_, k) => v.colormap(n, k / 100)))));"],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )  # fmt: skip
    for name, table in zip(names, json.loads(run.stdout), strict=True):
        reference = matplotlib.colormaps[name](np.linspace(0, 1, 101))[:, :3] * 255
        assert np.abs(np.array(table) - reference).max() < 12, name


def test_colour_scale_of_singular_fields():
    """The default colour scale of a field on the vertices leaves out its first and last
    percentiles: the peak of a crack tip does not darken the rest; a field of few values
    and the volumes keep their whole range."""
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.uniform(0.0, 1.0, 5000), [80.0, 120.0]])  # a singular tip
    lo, hi = sf._shown_range(values, 0.0, 120.0)
    assert 0.0 <= lo < 0.02 and 0.97 < hi < 1.0
    phases = np.r_[np.zeros(995), np.ones(5)]  # one phase on 0.5 % of the vertices
    assert sf._shown_range(phases, 0.0, 1.0) == [0.0, 1.0]
    assert sf._shown_range(None, -2.0, 3.0) == [-2.0, 3.0]
    assert sf._shown_range(np.full(4, np.nan), 0.0, 1.0) == [0.0, 1.0]
