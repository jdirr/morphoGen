"""The tools of the tab Stability: their catalog and forms; the post-bifurcation of a
honeycomb frame (its triple point at N L²/EI = 3.65 and the 2 x 2 cell that its Bloch
waves ask for); of morphologies of the workspace, a plane square array of holes on a
mirrored conforming mesh (a simple point that doubles the period, the holes turning into
alternating ellipses) and a gyroid on voxels (whose body-centring translation doubles no
period); the pattern transformation of the square array of holes along a nonlinear path
in finite strains and its bifurcated branch; and the complete band gap of stiff heavy
cubes in a soft matrix."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from morphogen import io
from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.tools import ToolJob
from morphogen.gui.tools import stability as st
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.image import VoxelImage
from morphogen.io.vtk import read_vtu
from morphogen.mesh.conforming import has_mmg
from morphogen.models.base import model_from_config
from morphogen.models.lattice_field import LatticeField

POROSITY = 0.5


def _holes(n: int = 48) -> LatticeField:
    """A plane square array of circular holes (a cell of edge 1, one voxel thick) as a
    design: its level set the distance to the edge of the hole, negative in the solid."""
    c = (np.arange(n) + 0.5) / n - 0.5
    d = np.hypot(c[:, None], c[None, :]) - np.sqrt(POROSITY / np.pi)
    field = (0.5 + d * n / 4)[:, :, None].astype(np.float32)
    domain = Domain(size=(1.0, 1.0, 1.0 / n), periodic=True)
    return LatticeField(domain, (field > 0.5).astype(np.uint8), field=field, threshold=0.5)


def _cubes(n: int = 4, cube: int = 2) -> LatticeField:
    """Cubes of ``cube`` voxels (phase 1) centred in a cell of ``n`` voxels of matrix."""
    data = np.zeros((n, n, n), np.uint8)
    lo = (n - cube) // 2
    data[lo : lo + cube, lo : lo + cube, lo : lo + cube] = 1
    return LatticeField(Domain(size=1.0, periodic=True), data)


@pytest.fixture
def workspace(tmp_path):
    """A session whose workspace holds the square array of holes, a gyroid of 2 x 2 x 2
    unit cells and the cubes in a matrix."""
    session = api.Session(tmp_path)
    keys = {}

    def add(name, micro, model=None, phases=None):
        key, _ = session.folder()
        session.add(key, "morphology", name, micro=micro, model=model,
                    phases=phases or {"0": "void", "1": "solid"})  # fmt: skip
        keys[name] = key

    add("holes", _holes())
    cfg = {"type": "tpms", "surface": "gyroid", "cell_size": 1.0, "volume_fraction": 0.3}
    add("gyroid", model_from_config(cfg).generate(Domain(size=2.0, periodic=True), 1), cfg)
    add("cubes", _cubes(), phases={"0": "matrix", "1": "cubes"})
    return session, keys


def _job(session, name, params):
    job = ToolJob({"tool": name, "params": params}, session)
    job._run()  # synchronously
    return job.status()


def _run(session, name, params):
    status = _job(session, name, params)
    assert status["state"] == "done", status.get("error")
    json.dumps(status)  # sent to the page as it is
    return status["result"]


def _table(result, start: str) -> list[dict]:
    """The shown rows of the table whose title starts so, as mappings."""
    table = next(t for t in result["tables"] if t["title"].startswith(start))
    return [dict(zip(table["columns"], row, strict=True)) for row in table["rows"]]


def test_catalog_forms_and_cells(workspace):
    session, keys = workspace
    catalog = tools.catalog()
    assert next(t for t in catalog["tabs"] if t["name"] == "stability")["order"] == 60
    mine = [t for t in catalog["tools"] if t["tab"] == "stability"]
    names = ["frame_bifurcation", "morphology_bifurcation", "path_bifurcation", "band_structure"]
    assert [t["name"] for t in mine] == names
    fields = {f["name"]: f for f in mine[2]["fields"]}
    assert fields["source"]["when"] == {"cell": ["morphology"]} and fields["source"]["required"]
    assert fields["materials"]["of"] == "source" and fields["young"]["when"] == {"cell": ["holes"]}
    # the forms: the hidden fields left out, the required ones checked when shown
    path = tools.TOOLS["path_bifurcation"]
    parsed = path.parse({}, session)
    assert parsed["cell"] == "holes" and parsed["source"] is None and parsed["porosity"] == 0.5
    with pytest.raises(ValueError, match="required"):
        path.parse({"cell": "morphology"}, session)
    # the loadings: unit strains, the largest component -1 in compression
    np.testing.assert_allclose(st._strain({"loading": "biaxial", "ratio": 0.5}, 2)[0],
                               np.diag([-1.0, -0.5]))  # fmt: skip
    e, _ = st._strain({"loading": "custom", "strain": [0, 0, -2, 0, 0, 1]}, 3)
    assert e[2, 2] == -1.0 and e[0, 1] == e[1, 0] == 0.5
    with pytest.raises(ValueError, match="6 values"):
        st._strain({"loading": "custom", "strain": [1, 2]}, 3)
    with pytest.raises(ValueError, match="axis"):
        st._strain({"loading": "uniaxial", "axis": "z"}, 2)
    # the cells: the holes symmetric about their mid-planes and the diagonal of the square;
    # one unit cell of the gyroid (its box holds 2 x 2 x 2), about none
    holes = session.item(keys["holes"])["micro"]
    body = st._CellBody(holes, holes.domain.lower, holes.domain.lengths, thickness=0.3)
    assert body.domain.size == (1.0, 1.0, 0.3) and st._mirrors(body) == ([0, 1, 2], [(0, 1)])
    gyroid = session.item(keys["gyroid"])
    lower, size, _ = st._primitive(gyroid, "unit")
    np.testing.assert_allclose(size, 1.0)
    assert st._primitive(gyroid, "box")[1].tolist() == [2.0, 2.0, 2.0]
    assert st._mirrors(st._CellBody(gyroid["micro"], lower, size)) == ([], [])
    # errors of the forms are reported by the jobs, with what to change
    assert "cells" in _job(session, "frame_bifurcation", {"cells": 9})["error"]
    status = _job(session, "morphology_bifurcation", {"source": keys["gyroid"], "mesh": "mirrored"})
    assert "symmetric about none" in status["error"]
    empty = ValueError("post_bifurcation: no load-carrying material in the cell")
    assert "more voxels per edge" in str(st._unresolved(empty, "voxels"))
    other = ValueError("something else")
    assert st._unresolved(other, "voxels") is other
    if api.available_memory() is not None:  # cells too large for the dense bases: refused
        huge = SimpleNamespace(n_nodes=10**6, order=1)
        with pytest.raises(ValueError, match="free memory"):
            st._check_size(huge, (1, 1, 1), 1, None, split=False)


@pytest.mark.slow
def test_post_bifurcation_of_a_honeycomb_frame(workspace):
    """The honeycomb in a cell of 2 x 2 under equibiaxial compression: the group of 48
    operations, a triple first point in an irrep of dimension 3 whose three axial classes
    are the modes I, II and III of Azulay & Combescure, at N L²/EI = 3.6508 in the walls;
    over the Brillouin zone of the primitive cell the lowest Bloch-wave load is at the M
    points, held by the 2 x 2 cell. Uniaxially, a double point; in shear, the square
    lattice carries no axial force and does not buckle."""
    session, _ = workspace
    r = _run(session, "frame_bifurcation", {})
    v = r["values"]
    assert v["group order"] == 48 and v["irreps"] == 10
    assert v["its dimension"] == 3 and v["multiplicity"] == 3
    assert v["axial force N L²/EI"] == pytest.approx(3.6508, rel=1e-4)
    assert v["period doubled along"] == "x, y"
    rows = _table(r, "patterns")
    first = v["irrep of the first bifurcation"]
    classes = sorted((p["point group"], p["order |H|"], p["conjugates"]) for p in rows
                     if p["irrep"] == first)  # fmt: skip
    assert classes == [("C6", 6, 4), ("D2", 4, 6), ("D2", 8, 3)]
    assert len(rows) == 18  # a pattern for each axial class
    assert v["Bloch: cell needed"] == "2 × 2" and v["Bloch: held by the cell"]
    assert v["Bloch: lowest load"] == pytest.approx(v["first critical load λ"], rel=1e-8)
    assert [f["title"] for f in r["figures"]] == [
        "patterns of the first bifurcation",
        "patterns of every irrep",
        "Bloch-wave buckling of the primitive cell",
    ]
    assert len(r["videos"]) == 3 and "triple point" in r["text"]
    r = _run(session, "frame_bifurcation", {"loading": "uniaxial", "axis": "x", "video": False,
                                            "patterns": "critical", "bloch_grid": 0})  # fmt: skip
    v = r["values"]
    assert v["group order"] == 16 and v["multiplicity"] == 2 and v["its dimension"] == 2
    square = {"lattice": "square", "loading": "shear", "video": False}
    r = _run(session, "frame_bifurcation", square)
    assert r["values"]["bifurcation"] == "none" and "no buckling" in r["text"]


@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_post_bifurcation_of_a_square_array_of_holes(workspace):
    """The plane square array of holes in uniaxial compression, on its conforming mesh
    mirrored with the diagonal of the square in a cell of 2 x 2: a simple first point in a
    one-dimensional irrep that doubles the period along both axes, whose pattern (read
    from the file of the patterns) turns the holes into alternating ellipses."""
    session, keys = workspace
    params = {"source": keys["holes"], "loading": "uniaxial", "axis": "y", "element_size": 0.15}
    r = _run(session, "morphology_bifurcation", {**params, "video": False})
    v = r["values"]
    assert v["mesh"] == "mirrored" and v["cells"] == "2 × 2 × 1" and v["group order"] == 32
    assert v["its dimension"] == 1 and v["multiplicity"] == 1
    assert v["period doubled along"] == "x, y" and 0.03 < v["first critical load λ"] < 0.1
    url = next(f["url"] for f in r["files"] if f["name"] == "patterns.vtu")
    path = session.file(url.split("/")[2], "patterns.vtu")
    mesh, data = io.read_tet_mesh(path), read_vtu(path)["point_data"]
    u = data[f"pattern_{v['irrep of the first bifurcation']}_1"]
    e = st._ellipticities(mesh.nodes, u, np.sqrt(POROSITY / np.pi), band=0.075)
    assert e[0] * e[1] < 0 and e[0] * e[2] < 0 and e[0] * e[3] > 0
    assert [f["title"] for f in r["figures"]][:1] == ["the analysed mesh (phases)"]


def test_post_bifurcation_of_a_gyroid_on_voxels(workspace):
    """A unit cell of a gyroid on six tetrahedra per voxel (it has no mirror): the
    operations of the cube combined with half translations that the voxels keep, a first
    point and its pattern; the body-centring translation of the gyroid is no doubling of
    the period of a cell of one primitive cell."""
    session, keys = workspace
    r = _run(session, "morphology_bifurcation", {"source": keys["gyroid"], "video": False})
    v = r["values"]
    assert v["mesh"] == "voxels" and v["cells"] == "1 × 1 × 1" and v["group order"] == 24
    assert v["first critical load λ"] > 0 and "period doubled along" not in v
    assert all(row["wave vectors (π/a)"] == "" for row in _table(r, "irreducible"))
    assert {f["name"] for f in r["files"]} >= {"irreps.csv", "patterns.vtu", "pattern_1.png"}


def test_pattern_transformation_along_a_path(workspace):
    """The square array of holes (porosity 0.5, voxels) in plane-strain compression along
    y, the lateral stress free, in finite strains: a simple first bifurcation between 3 and
    9 % in a one-dimensional irrep that doubles the period, the holes turning into
    alternating ellipses (Mullin et al. 2007; Bertoldi et al. 2008); on the bifurcated
    branch the pattern grows, the stress levels off and the lateral strain turns negative."""
    session, _ = workspace
    params = {"branch": True, "strain": 0.15, "increments": 24, "video": False}
    r = _run(session, "path_bifurcation", params)
    v = r["values"]
    assert v["mesh"] == "voxels" and v["group order"] == 16
    assert 0.03 < v["critical nominal strain"] < 0.09 and v["critical strain 11"] > 0
    assert v["its dimension"] == 1 and v["multiplicity"] == 1
    assert v["period doubled along"] == "x, y" and v["holes"] == "alternating ellipses"
    assert v["branch: growth of the pattern"] > 50 and v["branch: stress spread"] < 0.05
    assert v["branch: final strain 11"] < 0  # auxetic once transformed
    spectra = _table(r, "lowest eigenvalues")
    assert spectra[0]["nominal strain"] == 0 and all(
        value > 0 for key, value in spectra[0].items() if key.startswith("irrep")
    )
    titles = {f["title"] for f in r["figures"]}
    assert {"response along the path", "the bifurcated branch"} <= titles


def test_band_gap_of_stiff_heavy_cubes(workspace):
    """Stiff heavy cubes in a soft matrix (Young's modulus and density 100 and 10 times
    those of the matrix): the three translations at Γ and a complete band gap between the
    bands 6 and 7 along Γ–X–M–Γ–R."""
    session, keys = workspace
    materials = {"0": {"young": 1.0, "poisson": 0.35}, "1": {"young": 100.0, "poisson": 0.3}}
    params = {"source": keys["cubes"], "materials": materials, "densities": [1, 10],
              "resolution": 4, "order": "1", "points": 2}  # fmt: skip
    r = _run(session, "band_structure", params)
    v = r["values"]
    assert v["wave vectors"] == 9 and v["widest gap: bands"] == "6 – 7"
    assert v["c_T"] == pytest.approx(np.sqrt(1.0 / 2.7))
    gap = next(g for g in _table(r, "band gaps") if g["bands"] == "6 – 7")
    assert 0.5 < gap["low"] < gap["high"] < 1.6
    gamma = _table(r, "frequencies")[0]
    assert gamma["corner"] == "Γ" and max(gamma[f"band {k}"] for k in (1, 2, 3)) < 1e-3
    assert r["figures"][0]["title"] == "band structure"


def test_views_of_patterns_without_vtk():
    """The views drawn by matplotlib when VTK cannot render: a plane mesh seen from above
    and a mesh in space, a mode oscillating."""
    plane = voxel_tet_mesh(VoxelImage(np.ones((3, 3, 1), np.uint8), periodic=True))
    u = np.random.default_rng(0).normal(size=(plane.n_nodes, 3))
    frames = st._matplotlib_frames(plane, u, planar=True, frames=3)
    assert len(frames) == 3 and frames[0].shape[2] == 3
    assert not np.array_equal(frames[1], frames[2])
    space = voxel_tet_mesh(VoxelImage(np.ones((2, 2, 2), np.uint8), periodic=True))
    (view,) = st._matplotlib_frames(space, None, planar=False)
    assert view.ndim == 3 and view.std() > 0
