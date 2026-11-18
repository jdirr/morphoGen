"""The tools of the tab Solvers & formats of the interface: the inputs of AMITEX_FFTP (the
image, materials, loadings and MFront laws, read back) and its results read back, the
Abaqus input (voxel and conforming meshes, materials, crystal orientations, periodic
equations or KUBC, the steps, counted back), the meshes and images in other formats
(parsed back), the MFront sources (compiled when MFront is installed) and the material
files of Z-set; a tool run through the server, its files downloaded."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO

import numpy as np
import pytest
import yaml

from morphogen import io
from morphogen.domain import Domain
from morphogen.gui import api, tools
from morphogen.gui.server import GuiServer
from morphogen.gui.tools import ToolJob
from morphogen.homogenization import zmat
from morphogen.homogenization.materials import materials_from_config
from morphogen.mesh.conforming import has_mmg
from morphogen.models.base import model_from_config
from morphogen.seeding import derive_seed

needs_mmg = pytest.mark.skipif(not has_mmg(), reason="conforming meshes need MMG (pymmg)")
#: AMITEX's order (xx, yy, zz, xy, xz, yz) of morphoGen's Voigt components
TO6 = [0, 1, 2, 5, 4, 3]
ELASTIC = {"0": {"young": 70e3, "poisson": 0.3}, "1": {"young": 400e3, "poisson": 0.2}}
CHABOCHE = """
***behavior gen_evp
 **elasticity isotropic
      young 200000.
      poisson 0.30
 **potential gen_evp ev
   *criterion mises
   *flow norton
           n 7.0
           K 400.
   *kinematic nonlinear
         C 15000.0
         D 300.0
   *isotropic constant
           R0 130.0
***return
"""
TWO_POTENTIALS = """
***behavior gen_evp
 **elasticity isotropic
     young 200000.
     poisson 0.3
 **potential gen_evp ev
   *criterion mises
   *flow norton
     n 5. K 100.
   *isotropic constant
     R0 20.0
 **potential gen_evp ep
   *kinematic nonlinear
     C 25000.0 D 10.0
   *isotropic constant
     R0 850.0
***return
"""


def _session(tmp_path):
    """A session whose workspace holds a periodic Boolean model of spheres, its window
    (not periodic) and a periodic Voronoi polycrystal of six grains."""
    session = api.Session(tmp_path / "session")
    keys = {}
    spheres = model_from_config({"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.3})
    voronoi = model_from_config({"type": "voronoi", "n_cells": 6, "crystal_orientations": True})
    for key, model, periodic, phases, size in [
        ("spheres", spheres, True, {"0": "matrix", "1": "spheres"}, 2.0),
        ("window", spheres, False, {"0": "matrix", "1": "spheres"}, 2.0),
        ("grains", voronoi, True, {"1": "cells"}, 1.0),
    ]:
        domain = Domain(size=size, periodic=periodic)
        micro = model.generate(domain, derive_seed(7, 0))
        keys[key], _ = session.folder()
        session.add(keys[key], "morphology", key, micro=micro, seed=7, model=model.to_config(),
                    domain=domain.to_dict(), phases=phases)  # fmt: skip
    return session, keys


def _run(session, name, params):
    job = ToolJob({"tool": name, "params": params}, session)
    job._run()  # synchronously
    status = job.status()
    assert status["state"] == "done", status.get("error")
    return status["result"], job.folder


def _error(session, name, params) -> str:
    job = ToolJob({"tool": name, "params": params}, session)
    job._run()
    status = job.status()
    assert status["state"] == "error"
    return status["error"]


def _files(result) -> dict:
    return {f["name"]: f["kind"] for f in result["files"]}


def _inp(path) -> list[tuple[str, list[str]]]:
    """The keyword lines of an Abaqus input with the data lines of each (comments left
    out)."""
    blocks: list[tuple[str, list[str]]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("**"):
            continue
        if line.startswith("*"):
            blocks.append((line, []))
        elif blocks:
            blocks[-1][1].append(line)
    return blocks


def _count(blocks, keyword: str) -> int:
    """The data lines of the blocks of a keyword (``*NODE``, ``*ELEMENT``)."""
    return sum(len(data) for line, data in blocks if line.split(",")[0] == keyword)


def _std(path, rows):
    """An output file of AMITEX: a comment line, then rows of numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# time stress strain\n" + "".join(" ".join(f"{v:.15e}" for v in r) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


def test_catalog():
    catalog = tools.catalog()
    tab = next(t for t in catalog["tabs"] if t["name"] == "interop")
    assert tab["title"] == "Solvers & formats" and tab["order"] == 85
    names = [t["name"] for t in catalog["tools"] if t["tab"] == "interop"]
    assert names == ["amitex_inputs", "amitex_results", "abaqus_input", "mesh_formats",
                     "mfront_sources", "zmat_import", "zmat_export"]  # fmt: skip
    amitex = {f["name"]: f for f in tools.TOOLS["amitex_inputs"].to_dict()["fields"]}
    assert amitex["materials"]["of"] == "source" and amitex["behaviours"]["kind"] == "json"
    assert amitex["components"]["when"] == {"loading": ["path", "both"]}
    abaqus = {f["name"]: f for f in tools.TOOLS["abaqus_input"].to_dict()["fields"]}
    assert abaqus["element_size"]["when"] == {"mesh": ["conforming"]}
    assert abaqus["resolution"]["when"] == {"mesh": ["voxels"]}
    formats = {f["name"]: f for f in tools.TOOLS["mesh_formats"].to_dict()["fields"]}
    assert {"msh", "medit", "xdmf", "neper", "damask", "tif", "mhd"} <= {
        c[0] for c in formats["formats"]["choices"]
    }
    parsed = tools.TOOLS["mfront_sources"].parse({"behaviour": "crystal"}, None)
    assert parsed["yield_stress"] is None and parsed["tau0"] == 40.0 and parsed["young"] == 2e5


def test_amitex_inputs_and_results(tmp_path):
    """The inputs of AMITEX (the image, its materials, the unit strains and a path),
    parsed back; a plastic phase as an MFront law; the results of the runs read back."""
    session, keys = _session(tmp_path)
    params = {"source": keys["spheres"], "resolution": 8, "materials": ELASTIC,
              "loading": "both", "components": "E11=0.01, S22=5", "increments": [4]}  # fmt: skip
    result, folder = _run(session, "amitex_inputs", params)
    files = _files(result)
    stem = "boolean_spheres"
    for name, kind in [("README.md", "report"), (f"{stem}_material.xml", "configuration"),
                       (f"{stem}_materials.vtk", "data"), (f"{stem}_run.sh", "configuration"),
                       (f"{stem}_amitex.zip", "archive"), (f"{stem}_amitex.json", "configuration"),
                       ("materials.csv", "table"), ("image.png", "figure")]:  # fmt: skip
        assert files.get(name) == kind, name
    materials = ET.parse(folder / f"{stem}_material.xml").getroot().findall("Material")
    assert [m.get("Law") for m in materials] == ["elasiso", "elasiso"]
    lam, mu = materials_from_config(ELASTIC)[1].lame
    assert [float(c.get("Value")) for c in materials[1].findall("Coeff")] == pytest.approx(
        [lam, mu]
    )
    assert len(ET.parse(folder / f"{stem}_stiffness.xml").getroot().findall("Loading")) == 6
    (path,) = ET.parse(folder / f"{stem}_path.xml").getroot().findall("Loading")
    assert path.find("Time_Discretization").get("Nincr") == "4"
    drive = {
        el.tag: (el.get("Driving"), float(el.get("Value"))) for el in path if el.get("Driving")
    }
    assert drive["xx"] == ("Strain", 0.01) and drive["yy"] == ("Stress", 5.0)
    ids = io.read_vtk(folder / f"{stem}_materials.vtk")[0].data
    assert ids.shape == (8, 8, 8) and set(np.unique(ids)) == {1, 2}
    script = (folder / f"{stem}_run.sh").read_text(encoding="utf-8")
    assert f"-c {stem}_path.xml" in script and f"-s results/{stem}_stiffness" in script
    assert f"sh {stem}_run.sh" in result["text"] and result["values"]["runs"] == "stiffness, path"
    with zipfile.ZipFile(folder / f"{stem}_amitex.zip") as archive:
        assert {f"{stem}_material.xml", "README.md", f"{stem}_zones.vtk"} <= set(archive.namelist())
    # the results of the runs, as AMITEX writes them (one row per increment)
    a = np.random.default_rng(3).normal(size=(6, 6))
    c = a @ a.T + 6 * np.eye(6)
    rows = [[k + 1.0, *c[:, k][TO6], *np.eye(6)[k][TO6], *np.zeros(12), 5] for k in range(6)]
    _std(folder / "results" / f"{stem}_stiffness.std", rows)
    stress, strain = np.array([7e2, 5, 0, 0, 0, 0]), np.array([0.01, -2e-3, -2e-3, 0, 0, 0])
    rows = [[t, *(t * stress), *(t * strain), *np.zeros(12), 3] for t in (0.25, 0.5, 0.75, 1.0)]
    _std(folder / "results" / f"{stem}_path.std", rows)
    _std(folder / "results" / f"{stem}_path.mstd",
         [[t, *np.full(24, m * t)] for t in (0.25, 0.5, 0.75, 1.0) for m in (1, 2)])  # fmt: skip
    read, _ = _run(session, "amitex_results", {"directory": str(folder)})
    assert read["values"]["bulk_modulus"] == pytest.approx(c[:3, :3].sum() / 9)
    assert read["values"]["shear_modulus"] == pytest.approx(c[5, 5]) and "E1" in read["values"]
    stiffness = next(t for t in read["tables"] if t["title"].startswith("apparent stiffness"))
    assert np.allclose([r[1:] for r in stiffness["rows"]], c)
    compliance = next(t for t in read["tables"] if t["title"].startswith("apparent compliance"))
    assert np.allclose(np.array([r[1:] for r in compliance["rows"]]) @ c, np.eye(6), atol=1e-4)
    run = next(t for t in read["tables"] if t["title"] == "averages of the run path")
    assert run["count"] == 4 and run["columns"][:3] == ["time", "E11", "E22"]
    assert run["rows"][-1][1] == pytest.approx(0.01)
    assert any(t["title"].startswith("averages of the materials") for t in read["tables"])
    if importlib.util.find_spec("matplotlib") is not None:
        assert [f["title"] for f in read["figures"]] == ["stress-strain curves of path"]
    else:  # the tables only
        assert not read["figures"] and any("matplotlib" in w for w in read["warnings"])
    single, _ = _run(session, "amitex_results",
                     {"directory": str(folder / "results" / f"{stem}_path.std")})  # fmt: skip
    assert single["values"]["runs"] == f"{stem}_path"
    # a plastic phase: its MFront law of the umat interface, along the path only
    plastic = {
        **params,
        "loading": "path",
        "behaviours": json.dumps({"1": {"plasticity": {"yield": 300, "hardening": 1e3}}}),
    }
    result, folder = _run(session, "amitex_inputs", plastic)
    assert _files(result)["mfront/booleanspheresPhase1.mfront"] == "configuration"
    manifest = json.loads((folder / f"{stem}_amitex.json").read_text(encoding="utf-8"))
    assert manifest["materials"][1]["law"] == "umatbooleanspheresphase1"
    assert result["values"]["MFront laws"] == 1
    assert "mfront --obuild --interface=umat" in result["text"]
    assert "unit strains" in _error(session, "amitex_inputs", {**plastic, "loading": "stiffness"})
    assert "no results yet" in _error(session, "amitex_results", {"directory": str(folder)})
    assert "no folder" in _error(session, "amitex_results", {"directory": str(tmp_path / "no")})


def test_abaqus_voxels(tmp_path):
    """Voxel meshes: the nodes, elements, materials, periodic equations and steps of the
    input counted back; KUBC for a morphology that is not periodic."""
    session, keys = _session(tmp_path)
    params = {"source": keys["spheres"], "mesh": "voxels", "resolution": 6, "materials": ELASTIC,
              "components": "E11=0.01 S12=5", "times": [0, 1, 3], "increments": [2, 4]}  # fmt: skip
    result, folder = _run(session, "abaqus_input", params)
    blocks = _inp(folder / "boolean_spheres.inp")
    assert _count(blocks, "*NODE") == 7**3 + 3  # and the reference nodes
    assert _count(blocks, "*ELEMENT") == 6**3
    upper = 7**3 - 6**3  # the nodes of the upper faces, tied to their images
    assert sum(line == "*EQUATION" for line, _ in blocks) == 3 * upper
    assert result["values"]["equations"] == 3 * upper and result["values"]["steps"] == 2
    keywords = [line for line, _ in blocks]
    assert keywords.count("*ELASTIC") == 2 and "*MATERIAL, NAME=PHASE_1" in keywords
    steps = [k for k, (line, _) in enumerate(blocks) if line.startswith("*STEP")]
    assert len(steps) == 2 and "NLGEOM=NO" in blocks[steps[0]][0]
    second = dict(blocks[steps[1] : steps[1] + 4])
    assert second["*STATIC"] == ["0.5, 2, 5e-06, 0.5"]
    # E11 a displacement of REF_X (times the edge, 2), the stresses forces (times the
    # areas of the faces, 4): S12 on REF_Y along 1, the free components zero
    assert second["*BOUNDARY"] == ["CORNER, 1, 3, 0.", "REF_X, 2, 3, 0.", "REF_Y, 3, 3, 0.",
                                   "REF_X, 1, 1, 0.02"]  # fmt: skip
    assert second["*CLOAD"] == ["REF_Y, 2, 0", "REF_Z, 3, 0", "REF_Z, 2, 0", "REF_Z, 1, 0",
                                "REF_Y, 1, 20"]  # fmt: skip
    files = _files(result)
    assert files["boolean_spheres.inp"] == "configuration" and "README.md" in files
    assert "boolean_spheres_equations.inp" not in files  # in the input
    assert "the node sets of the faces X0, X1" in result["text"] and "2 steps" in result["text"]
    assert "periodic equations" in result["text"] and "abaqus job=boolean_spheres" in result["text"]
    # the window of the model: the faces under uniform strains, strains only
    window = {**params, "source": keys["window"], "components": "E11=0.01 E22=0 E33=0 E23=0 "
              "E31=0 E12=0.004", "times": [0, 1], "increments": [1]}  # fmt: skip
    result, folder = _run(session, "abaqus_input", window)
    blocks = _inp(folder / "boolean_spheres.inp")
    assert _count(blocks, "*NODE") == 7**3  # no reference nodes
    assert result["values"]["boundary conditions"] == "KUBC (the faces)"
    (step,) = [k for k, (line, _) in enumerate(blocks) if line.startswith("*STEP")]
    boundary = blocks[step + 2][1]
    assert len(boundary) == 3 * (7**3 - 5**3)  # the nodes of the six faces, three each
    nodes = np.array([[float(v) for v in d.split(",")[1:]] for d in _inp_nodes(folder)])
    for line in boundary[::3]:  # u1 = H11 x + H12 y (H upper triangular, x0 = 0)
        node, dof, _, value = line.split(",")
        x = nodes[int(node) - 1]
        assert int(dof) == 1 and float(value) == pytest.approx(0.01 * x[0] + 0.004 * x[1])
    keywords = [line for line, _ in blocks]
    assert "*NODE PRINT, NSET=X1, TOTALS=YES" in keywords and "*NSET, NSET=X1" in keywords
    assert "stresses" in _error(session, "abaqus_input", {**window, "components": "E11=0.01"})
    # no step: the model only
    model, folder = _run(session, "abaqus_input", {**params, "loading": "none"})
    assert not any(line.startswith("*STEP") for line, _ in _inp(folder / "boolean_spheres.inp"))
    assert model["values"]["steps"] == 0 and model["values"]["equations"] == 3 * upper
    assert "*TEMPERATURE" not in [line for line, _ in blocks]  # no material expands
    # expanding phases: the nodes start at 293.15 K, and each step keeps them there
    hot = json.dumps({"0": {"expansion": 1e-5}, "1": {"expansion": 2e-5}})
    _, folder = _run(session, "abaqus_input", {**params, "behaviours": hot})
    blocks = _inp(folder / "boolean_spheres.inp")
    assert dict(blocks)["*INITIAL CONDITIONS, TYPE=TEMPERATURE"] == ["ALL_NODES, 293.15"]
    assert [data for line, data in blocks if line == "*TEMPERATURE"] == [["ALL_NODES, 293.15"]] * 2


def _inp_nodes(folder):
    """The data lines of the first *NODE block of the input in ``folder``."""
    blocks = _inp(next(folder.glob("*.inp")))
    return next(data for line, data in blocks if line == "*NODE")


def _dat(text: str) -> dict:
    """The blocks of a .dat file of CalculiX: (quantity, node set, time) -> rows (the rows
    start with a node number; the headers of the steps and increments end a block)."""
    out: dict = {}
    key = None
    for line in text.splitlines():
        title = re.match(r"\s*(.+?) \(.*\) for set (\S+) and time\s+(\S+)", line)
        if title:
            key = (title.group(1), title.group(2), float(title.group(3)))
            out[key] = []
        elif key is not None and re.match(r"\s*\d", line):
            out[key].append([float(v) for v in line.split()])
        elif line.strip():
            key = None
    return out


@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX (ccx)")
def test_abaqus_steps_with_calculix(tmp_path):
    """The inputs solved by CalculiX for a homogeneous material: the periodic equations
    under a strain and a shear stress (the displacements of the reference nodes), a cycle
    of stresses along three steps, and the faces of a window under a uniform strain (their
    total reaction forces), against the closed forms."""
    session, keys = _session(tmp_path)
    e, nu, edge = 2e5, 0.3, 2.0
    lam, mu = e * nu / ((1 + nu) * (1 - 2 * nu)), e / (2 * (1 + nu))
    same = {"0": {"young": e, "poisson": nu}, "1": {"young": e, "poisson": nu}}

    def solve(name, source, components, times=(0, 1), behaviours=None):
        params = {"source": keys[source], "mesh": "voxels", "resolution": 3, "materials": same,
                  "components": components, "times": list(times), "increments": [1],
                  "name": name, "behaviours": json.dumps(behaviours or {})}  # fmt: skip
        _, folder = _run(session, "abaqus_input", params)
        subprocess.run(["ccx", "-i", name], cwd=folder, check=True, capture_output=True)
        return _dat((folder / f"{name}.dat").read_text(encoding="ascii"))

    out = solve("mixed", "spheres", "E11=0.001 S12=50")
    (ref_x,) = out[("displacements", "REF_X", 1.0)]
    (ref_y,) = out[("displacements", "REF_Y", 1.0)]
    assert ref_x[1] == pytest.approx(0.001 * edge)  # u1(REF_X) = H11 L
    assert ref_y[1] == pytest.approx(50 / mu * edge)  # the shear stress S12: H12 = S12 / G
    assert ref_y[2] == pytest.approx(-nu * 0.001 * edge)  # free lateral contraction
    # expanding: the steps keep the nodes at their initial 293.15 K (CalculiX fails on a
    # step without a temperature), so that nothing expands along the path
    expanding = {"0": {"expansion": 1e-5}, "1": {"expansion": 1e-5}}
    hot = solve("hot", "spheres", "E11=0.001 S12=50", behaviours=expanding)
    for ref in ("REF_X", "REF_Y", "REF_Z"):
        key = ("displacements", ref, 1.0)
        assert np.allclose(hot[key], out[key], rtol=1e-12, atol=1e-15)
    out = solve("cycle", "spheres", "S11=100,0,-100", times=(0, 1, 2, 3))
    cycle = [out[("displacements", "REF_X", t)][0][1] for t in (1.0, 2.0, 3.0)]
    assert cycle == pytest.approx([100 / e * edge, 0.0, -100 / e * edge], abs=1e-12)
    out = solve("kubc", "window", "E11=0.001 E22=0 E33=0 E23=0 E31=0 E12=0")
    area = edge**2
    assert out[("total force", "X1", 1.0)][0][0] == pytest.approx((lam + 2 * mu) * 0.001 * area)
    assert out[("total force", "Y1", 1.0)][0][1] == pytest.approx(lam * 0.001 * area)


@needs_mmg
def test_abaqus_conforming_polycrystal_and_formats(tmp_path):
    """The conforming mesh of a polycrystal (whole periodic grains): a crystal turned in
    each grain, its crystal plasticity an MFront law, the periodic equations; the mesh in
    the formats of Gmsh, Medit, FEniCSx, its node pairs, Neper, DAMASK, TIFF and
    MetaImage, read back."""
    session, keys = _session(tmp_path)
    crystal = {"cubic": [168e3, 121e3, 75e3],
               "crystal_plasticity": {"tau0": 40, "K": 20, "n": 12}}  # fmt: skip
    params = {"source": keys["grains"], "element_size": 0.5, "materials": {"1": ELASTIC["1"]},
              "behaviours": {"1": crystal}, "components": "E11=0.01"}  # fmt: skip
    result, folder = _run(session, "abaqus_input", params)
    blocks = _inp(folder / "voronoi.inp")
    values = result["values"]
    assert _count(blocks, "*NODE") == values["nodes"] + 3
    assert _count(blocks, "*ELEMENT") == values["elements"] and values["element type"] == "C3D4"
    assert values["grains"] == 6 and values["MFront laws"] == 1 and values["equations"] > 0
    keywords = [line for line, _ in blocks]
    assert sum(line.startswith("*ORIENTATION, NAME=ORI_") for line in keywords) == 6
    sections = [line for line in keywords if line.startswith("*SOLID SECTION")]
    assert len(sections) == 6 and all("ORIENTATION=ORI_" in line for line in sections)
    assert all("MATERIAL=ABAQUSBEHAVIOUR_VORONOIPHASE1_3D" in line for line in sections)
    assert dict(blocks)["*DEPVAR"] == ["43,"]  # 7 + 3 x 12 slip systems
    source = folder / "voronoi_mfront" / "voronoiPhase1.mfront"
    assert "@OrthotropicBehaviour;" in source.read_text(encoding="utf-8")
    assert _files(result)["voronoi_mfront/voronoiPhase1.mfront"] == "configuration"
    assert "mfront --obuild --interface=abaqus voronoiPhase1.mfront" in result["text"]
    assert "CELL_<c> (the grains, their orientations ORI_<c>)" in result["text"]
    assert "the node sets OUTER" in result["text"]  # the boundary of the whole cells
    # the other formats of the mesh, the tessellation and the image (TIFF: with tifffile)
    tiff = importlib.util.find_spec("tifffile") is not None
    chosen = ["msh", "medit", "xdmf", "pairs", "neper", "damask", "mhd", *(["tif"] * tiff)]
    formats = {"source": keys["grains"], "formats": chosen, "element_size": 0.5,
               "resolution": 10, "image": "labels"}  # fmt: skip
    result, folder = _run(session, "mesh_formats", formats)
    values = result["values"]
    assert len(io.read_msh(folder / "voronoi.msh")["nodes"]) == values["mesh nodes"]
    medit = io.read_tet_mesh(folder / "voronoi.mesh")
    assert medit.n_elements == values["mesh elements"] and medit.whole
    assert len(np.unique(medit.cell)) == 6
    grids = ET.parse(folder / "voronoi.xdmf").getroot().find("Domain").findall("Grid")
    assert [g.get("Name") for g in grids] == ["mesh", "phase", "grain", "faces"]
    topology = int(grids[0].find("Topology").get("NumberOfElements"))
    assert topology == values["mesh elements"]
    header = (folder / "voronoi_periodic.csv").read_text(encoding="ascii").splitlines()[0]
    assert header == "node,image,shift_x,shift_y,shift_z"
    assert np.loadtxt(folder / "voronoi_seeds.txt").shape == (6, 3)
    assert "neper -T" in (folder / "voronoi_neper.sh").read_text(encoding="utf-8")
    damask = yaml.safe_load((folder / "voronoi_damask_material.yaml").read_text(encoding="utf-8"))
    assert len(damask["material"]) == 6 and (folder / "voronoi_damask.vti").exists()
    mhd = io.read_mhd(folder / "voronoi.mhd")
    assert mhd.shape == (10, 10, 10) and mhd.data.dtype == np.uint16
    assert len(np.unique(mhd.data)) == 6  # the cells
    if tiff:
        tif = io.read_tiff(folder / "voronoi.tif")
        assert tif.data.dtype == np.uint16 and np.array_equal(tif.data, mhd.data)
    files = _files(result)
    assert files["voronoi.mesh"] == "data" and files["voronoi_formats.zip"] == "archive"
    assert "FEniCSx" in result["text"]


def test_formats_of_voxels(tmp_path):
    """The voxel mesh (Gmsh, periodic pairs) and images (the level set as a MetaImage);
    what needs a tessellation."""
    session, keys = _session(tmp_path)
    params = {"source": keys["spheres"], "formats": ["voxel_msh", "voxel_pairs", "mhd"],
              "resolution": 6, "image": "level_set"}  # fmt: skip
    result, folder = _run(session, "mesh_formats", params)
    msh = io.read_msh(folder / "boolean_spheres_voxels.msh")
    assert len(msh["nodes"]) == 7**3 and len(msh["elements"][5]) == 6**3
    pairs = np.loadtxt(folder / "boolean_spheres_voxels_periodic.csv", delimiter=",",
                       skiprows=1, usecols=(1, 2))  # fmt: skip
    assert pairs.shape == (3 * 7**2, 2)
    level = io.read_mhd(folder / "boolean_spheres.mhd")
    assert level.shape == (6, 6, 6) and level.data.dtype == np.float32
    assert {"boolean_spheres.raw", "image.png"} <= set(_files(result))
    for formats, match in [(["neper"], "Neper"), (["damask"], "DAMASK"), ([], "at least one")]:
        assert match in _error(session, "mesh_formats", {**params, "formats": formats})
    assert "cells" in _error(session, "mesh_formats", {**params, "image": "labels"})
    # a morphology of several phases: the implicit function of each phase, one file each
    model = model_from_config(
        {"type": "coating", "thickness": 0.1,
         "model": {"type": "hardcore_spheres", "radius": 0.4, "volume_fraction": 0.2}}
    )  # fmt: skip
    domain = Domain(size=2.0, periodic=True)
    coated = model.generate(domain, 3)
    key, _ = session.folder()
    names = {"0": "matrix", "1": "spheres", "2": "coating"}
    session.add(key, "morphology", "coated", micro=coated, seed=3, model=model.to_config(),
                domain=domain.to_dict(), phases=names)  # fmt: skip
    several = {"source": key, "formats": ["mhd", "tif"], "resolution": 12, "image": "level_set"}
    result, folder = _run(session, "mesh_formats", several)
    phases = coated.voxelize(shape=12).data
    for k in range(3):
        level = io.read_mhd(folder / f"coating_phase{k}.mhd")
        assert level.data.dtype == np.float32 and ((level.data < 0) == (phases == k)).all()
        np.testing.assert_array_equal(
            io.read_tiff(folder / f"coating_phase{k}.tif").data, level.data
        )
    assert {f"image_phase{k}.png" for k in range(3)} <= set(_files(result))
    assert "coating_phase2.mhd" in result["text"]


def test_mfront_sources(tmp_path):
    """The sources of von Mises plasticity, crystal plasticity (small and finite
    strains), Hencky's elasticity and a material of the configurations; their materials
    read back; compiled by MFront when it is installed."""
    session = api.Session(tmp_path / "session")
    mises = {"behaviour": "vonmises", "young": 70e3, "poisson": 0.33, "yield_stress": 100,
             "voce": [50, 20], "kinematic": [20000, 200, 5000, 0], "norton": [50, 5]}  # fmt: skip
    result, folder = _run(session, "mfront_sources", mises)
    text = (folder / "MorphogenVonMises.mfront").read_text(encoding="utf-8")
    for line in ('inelastic_flow : "Norton"', 'isotropic_hardening : "Voce" {R0 : 0, Rinf : 50.0',
                 'kinematic_hardening : "Armstrong-Frederick" {C : 20000.0, D : 200.0}',
                 'kinematic_hardening : "Prager" {C : 5000.0}', "K : 50.0"):  # fmt: skip
        assert line in text
    assert result["values"]["internal state variables"] == 7 + 6 * 2
    assert result["values"]["umat function"] == "umatmorphogenvonmises"
    material = yaml.safe_load((folder / "material.yaml").read_text(encoding="utf-8"))
    read = materials_from_config(material["materials"])[1]
    assert read.plasticity.viscosity == 50 and read.plasticity.kinematic[1] == (5000.0, 0.0)
    assert {"MorphogenVonMises.mfront", "README.md", "MorphogenVonMises.zip"} <= set(_files(result))
    crystal = {"behaviour": "crystal", "elasticity": "cubic", "families": ["octahedral"],
               "interaction": [1, 1, 0.6, 12.3, 1.6, 1.3], "name": "Grain_1"}  # fmt: skip
    result, folder = _run(session, "mfront_sources", crystal)
    text = (folder / "Grain1.mfront").read_text(encoding="utf-8")  # an identifier for Abaqus
    assert "@IntegerConstant Nss = 12;" in text and "@OrthotropicBehaviour;" in text
    assert result["values"]["internal state variables"] == 43
    assert result["values"]["Abaqus material"] == "ABAQUSBEHAVIOUR_GRAIN1_3D"
    result, folder = _run(session, "mfront_sources", {**crystal, "kinematics": "finite"})
    assert "@DSL DefaultFiniteStrain;" in (folder / "Grain1.mfront").read_text(encoding="utf-8")
    assert "more lines in Grain1.mfront" in result["text"]  # morphoGen's kernel in the file
    hencky = {"behaviour": "elastic", "kinematics": "finite"}
    result, folder = _run(session, "mfront_sources", hencky)
    assert "@StrainMeasure Hencky;" in (folder / "MorphogenHencky.mfront").read_text("utf-8")
    assert result["values"]["umat function"] == "umatmorphogenhencky_malls"
    assert "finite strains" in _error(session, "mfront_sources", {"behaviour": "elastic"})
    material = {"young": 2e5, "poisson": 0.3, "plasticity": {"yield": 250, "hardening": 800}}
    own = {"behaviour": "material", "material": json.dumps(material)}
    result, folder = _run(session, "mfront_sources", own)
    text = (folder / "MorphogenMaterial.mfront").read_text(encoding="utf-8")
    assert 'isotropic_hardening : "Linear" {R0 : 250.0, H : 800.0}' in text
    assert "pairs" in _error(session, "mfront_sources", {**mises, "voce": [50]})
    if shutil.which("mfront") is None:
        result, _ = _run(session, "mfront_sources", {**mises, "compile": True})
        assert any("not on the PATH" in w for w in result["warnings"])


@pytest.mark.skipif(shutil.which("mfront") is None, reason="MFront (TFEL) on the PATH")
def test_mfront_sources_compile(tmp_path, monkeypatch):
    """The sources of von Mises and crystal plasticity (small and finite strains) compiled
    by MFront with its generic interface."""
    monkeypatch.setenv("MORPHOGEN_MFRONT_CACHE", str(tmp_path / "cache"))
    session = api.Session(tmp_path / "session")
    crystal = {"behaviour": "crystal", "elasticity": "cubic"}
    for params in (
        {"behaviour": "vonmises", "norton": [50, 5]},
        crystal,
        {**crystal, "kinematics": "finite"},
    ):
        result, _ = _run(session, "mfront_sources", {**params, "compile": True})
        assert result["values"]["library"] and not result["warnings"]


def test_zmat_import(tmp_path):
    """A gen_evp behaviour on morphoGen's kernels (its brick for the other codes too), the
    same through MFront, a behaviour of two potentials through MFront; the bricks read,
    the coverage of Z-mat."""
    session = api.Session(tmp_path / "session")
    result, folder = _run(session, "zmat_import", {"text": CHABOCHE})
    values = result["values"]
    assert values["translation"] == "native" and values["kind"] == "plastic"
    material = yaml.safe_load((folder / "material.yaml").read_text(encoding="utf-8"))
    read = materials_from_config(material["materials"])[1]
    assert read.plasticity.yield_stress == 130 and read.plasticity.viscosity == 400
    brick = (folder / "ZmatBehaviour.mfront").read_text(encoding="utf-8")
    assert "@Brick StandardElastoViscoPlasticity" in brick
    bricks = next(t for t in result["tables"] if t["title"].startswith("the behaviour"))
    assert ["potential ev: flow", "norton", "n 7, K 400"] in bricks["rows"]
    coverage = next(t for t in result["tables"] if t["title"].startswith("coverage"))
    assert coverage["count"] == len(zmat.COVERAGE) == len(coverage["rows"])
    assert {"zmat.mat", "material.yaml", "zmat.zip", "coverage.csv"} <= set(_files(result))
    # MFront for the other codes; two potentials: MFront only
    result, folder = _run(session, "zmat_import", {"text": CHABOCHE, "native": False,
                                                   "name": "alloy"})  # fmt: skip
    material = yaml.safe_load((folder / "material.yaml").read_text(encoding="utf-8"))
    assert material["materials"][1]["mfront"]["source"] == "ZmatAlloy.mfront"
    assert (folder / "ZmatAlloy.mfront").exists() and result["values"]["translation"] == "mfront"
    result, folder = _run(session, "zmat_import", {"text": TWO_POTENTIALS})
    assert result["values"]["translation"] == "mfront" and result["values"]["potentials"] == 2
    assert "several potentials" in result["text"]
    assert "the file has 1" in _error(session, "zmat_import", {"text": CHABOCHE, "behaviour": 1})
    with pytest.raises(ValueError, match="required"):
        tools.TOOLS["zmat_import"].parse({}, None)


def test_zmat_export(tmp_path):
    """The behaviours of the workspace written as material files of Z-mat: imported, then
    identified-like (two backstresses, two saturating terms, Norton's flow), read back as
    the same materials; the behaviours a file of Z-mat does not hold refused."""
    session = api.Session(tmp_path / "session")
    imported, _ = _run(session, "zmat_import", {"text": CHABOCHE})
    (item,) = imported["items"]
    result, folder = _run(session, "zmat_export", {"behaviour": item["key"], "name": "alloy"})
    assert result["values"] == {"behaviour": "gen_evp", "potentials": 1}
    text = (folder / "alloy.mat").read_text(encoding="utf-8")
    assert text.startswith(f"% {item['title']}") and "*flow norton" in text
    original = zmat.zmat_material(CHABOCHE).to_config()
    assert zmat.zmat_material(text).to_config() == original
    assert {"alloy.mat", "README.md", "alloy.zip"} <= set(_files(result))
    assert "    ***behavior gen_evp" in result["text"]
    chaboche = {"young": 1.0, "poisson": 0.3, "expansion": 1e-5, "plasticity": {
        "yield": 0.0019999999999924245, "isotropic": [{"Q": 0.001, "b": 100.0},
                                                    {"Q": 0.0005, "b": 7.0}],
        "kinematic": [{"C": 0.2, "D": 200.0}, {"C": 0.01, "D": 0.0}],
        "norton": {"K": 0.003, "n": 6.0}}}  # fmt: skip
    key, _ = session.folder()
    session.add(key, "behaviour", "identified", material=chaboche)
    result, folder = _run(session, "zmat_export", {"behaviour": key})
    text = (folder / "behaviour.mat").read_text(encoding="utf-8")
    assert "*kinematic nonlinear X1" in text and "*kinematic linear X2" in text
    assert "*isotropic nonlinear_sum" in text and "R0 0.0019999999999924245" in text
    assert zmat.zmat_material(text).to_config() == chaboche
    result, folder = _run(session, "zmat_export", {"behaviour": key, "digits": 6})
    assert "R0 0.002\n" in (folder / "behaviour.mat").read_text(encoding="utf-8")
    assert "to 6 significant digits" in result["text"]
    linear = {**chaboche, "plasticity": {**chaboche["plasticity"], "hardening": 0.01}}
    model = {"criterion": "gurson", "yield": 0.002, "porosity": 0.0717, "q1": 1.3963, "q2": 0.7474}
    gurson = {"young": 1.0, "poisson": 0.3, "porous_plasticity": model}
    porous, _ = session.folder()
    session.add(porous, "behaviour", "Gurson model of a porous morphology", material=gurson)
    result, folder = _run(session, "zmat_export", {"behaviour": porous})
    assert result["values"] == {"behaviour": "porous_plastic", "potentials": 1}
    text = (folder / "behaviour.mat").read_text(encoding="utf-8")
    back = zmat.zmat_material(text).to_config()
    assert (
        back["porous_plasticity"]["q1"] == 1.3963
        and back["porous_plasticity"]["porosity"] == 0.0717
    )
    other, _ = session.folder()
    session.add(other, "behaviour", "with a linear hardening", material=linear)
    message = _error(session, "zmat_export", {"behaviour": other})
    assert "no isotropic brick of Z-mat" in message


def test_through_the_server(tmp_path):
    """The tab in the catalog of the server, a tool run as a job of the page on a
    morphology of the workspace, and its files downloaded: the input, the MFront law in its
    folder, the archive."""
    server = GuiServer(("127.0.0.1", 0), tmp_path / "session")
    model = model_from_config({"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.3})
    micro = model.generate(Domain(size=2.0, periodic=True), derive_seed(7, 0))
    key, _ = server.session.folder()
    server.session.add(key, "morphology", "spheres", micro=micro, seed=7, model=model.to_config(),
                       phases={"0": "matrix", "1": "spheres"})  # fmt: skip
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=60) as r:
            return r.read()

    def post(path, data):
        body = json.dumps(data).encode()
        req = urllib.request.Request(base + path, data=body,
                                     headers={"Content-Type": "application/json"})  # fmt: skip
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    try:
        catalog = json.loads(get("/api/tools"))
        assert "interop" in [t["name"] for t in catalog["tabs"]]
        params = {"source": key, "mesh": "voxels", "resolution": 4, "materials": ELASTIC,
                  "behaviours": {"0": {"plasticity": {"yield": 100}}}}  # fmt: skip
        job = post("/api/tool", {"tool": "abaqus_input", "params": params})["job"]
        for _ in range(600):
            status = json.loads(get(f"/api/job/{job}"))
            if status["state"] != "running":
                break
            time.sleep(0.1)
        assert status["state"] == "done", status.get("error")
        result = status["result"]
        urls = {f["name"]: f["url"] for f in result["files"]}
        law = "boolean_spheres_mfront/booleanspheresPhase0.mfront"  # in a folder of the result
        assert b"@Behaviour booleanspheresPhase0;" in get(urls[law])
        assert get(urls["boolean_spheres.inp"]).startswith(b"*HEADING")
        archive = zipfile.ZipFile(BytesIO(get(result["archive"])))
        assert {"boolean_spheres.inp", law} <= set(archive.namelist())
    finally:
        server.shutdown()
