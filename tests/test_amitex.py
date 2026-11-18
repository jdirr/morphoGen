"""AMITEX_FFTP runs: the inputs written for an image (ten-line VTK headers, big-endian
integers and doubles, the native laws, the orientations of the grains, the loadings)
and the results read back (.std, .mstd, .zstd, the apparent stiffness)."""

import csv
import dataclasses
import importlib.util
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from morphogen import cli, io, pipeline
from morphogen.config import ConfigError, load_config
from morphogen.homogenization import (
    CrystalPlasticity,
    Loading,
    LoadingPath,
    Material,
    Plasticity,
)
from morphogen.homogenization.materials import material_table, tensor4_to_voigt, voigt_to_tensor4
from morphogen.image import VoxelImage
from morphogen.io.amitex import COMPONENTS
from morphogen.orientations import random_orientations

MATS = {
    0: Material(void=True),
    1: Material(young=70e3, poisson=0.3),
    2: Material.cubic(168e3, 121e3, 75e3),
}
#: AMITEX's components (xx, yy, zz, xy, xz, yz) of morphoGen's Voigt components, and of
#: gradients stored row by row
TO6 = [0, 1, 2, 5, 4, 3]
TO9 = [0, 4, 8, 1, 2, 5, 3, 6, 7]


def polycrystal(shape=(6, 5, 4), seed=0):
    """Phases 0 (void), 1 (isotropic) and 2 (cubic crystal), and grains 0..6."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 3, size=shape).astype(np.uint8)
    grains = rng.integers(0, 7, size=shape)
    return VoxelImage(data, spacing=(0.5, 0.25, 0.4), origin=(1.0, 0.0, -2.0)), grains


def _header(path):
    lines = path.read_bytes().split(b"\n", 10)
    return [line.decode("ascii") for line in lines[:10]], lines[10]


def _bin(path):
    count, kind, rest = path.read_bytes().split(b"\n", 2)
    assert kind == b"double"
    values = np.frombuffer(rest, dtype=">f8")
    assert values.size == int(count)
    return values


def _coeffs(material):
    return {int(c.get("Index")): c for c in material.findall("Coeff")}


def _drivings(loading):
    return {el.tag: (el.get("Driving"), float(el.get("Value"))) for el in loading
            if el.get("Driving")}  # fmt: skip


def test_amitex_inputs(tmp_path):
    """The fields, materials, loadings, algorithm and script of an export; the stiffness of
    each voxel rebuilt from the material file as AMITEX turns it equals morphoGen's."""
    image, grains = polycrystal()
    orientations = random_orientations(np.random.default_rng(1), 7)
    export = io.write_amitex(tmp_path, image, MATS, grains=grains, orientations=orientations,
                             name="poly")  # fmt: skip
    lines, data = _header(tmp_path / "poly_materials.vtk")
    assert lines[0] == "# vtk DataFile Version 4.5" and lines[2] == "BINARY"
    assert lines[3] == "DATASET STRUCTURED_POINTS" and lines[4] == "DIMENSIONS 7 6 5"
    assert lines[7] == "CELL_DATA 120" and lines[8] == "SCALARS MaterialId int"
    assert lines[9] == "LOOKUP_TABLE default" and len(data) == 4 * 120
    mat = io.read_vtk(tmp_path / "poly_materials.vtk")[0].data
    zone = io.read_vtk(tmp_path / "poly_zones.vtk")[0].data
    assert np.array_equal(mat, image.data.astype(int) + 1)
    assert np.all(zone[mat != 3] == 1)
    present = np.unique(grains[mat == 3])
    assert np.array_equal(np.unique(zone[mat == 3]), np.arange(1, present.size + 1))
    if importlib.util.find_spec("vtk") is not None:  # an independent reader
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(str(tmp_path / "poly_zones.vtk"))
        reader.Update()
        out = reader.GetOutput()
        assert out.GetDimensions() == (7, 6, 5)
        values = vtk_to_numpy(out.GetCellData().GetArray(0))
        assert np.array_equal(values.reshape(4, 5, 6).transpose(2, 1, 0), zone)
    # the materials: elasiso for the void and the isotropic phase, elasaniso for the crystal
    root = ET.parse(tmp_path / "poly_material.xml").getroot()
    void, iso, crystal = root.findall("Material")
    assert [m.get("Law") for m in (void, iso, crystal)] == ["elasiso", "elasiso", "elasaniso"]
    assert [m.get("numM") for m in (void, iso, crystal)] == ["1", "2", "3"]
    assert [float(c.get("Value")) for c in void.findall("Coeff")] == [0.0, 0.0]
    lam, mu = MATS[1].lame
    assert [float(c.get("Value")) for c in iso.findall("Coeff")] == pytest.approx([lam, mu])
    ref = root.find("Reference_Material")
    assert float(ref.get("Lambda0")) > 0 and float(ref.get("Mu0")) > 0
    coeffs = _coeffs(crystal)
    constants = [float(coeffs[k].get("Value")) for k in range(1, 10)]
    assert constants == [168e3, 121e3, 121e3, 168e3, 121e3, 168e3, 75e3, 75e3, 75e3]
    axes = np.column_stack([_bin(tmp_path / coeffs[k].get("File")) for k in range(10, 16)])
    assert all(coeffs[k].get("Type") == "Constant_Zone" for k in range(10, 16))
    assert axes.shape == (present.size, 6) and len(crystal.findall("IntVar")) == 6
    # the stiffness of each crystal voxel: the orthotropic constants turned to the axes N1,
    # N2, N3 = N1 x N2 of its zone (columns of the rotation), against morphoGen's table
    c11, c12, c13, c22, c23, c33, c66, c55, c44 = constants
    local = np.diag([0, 0, 0, c44, c55, c66])
    local[:3, :3] = [[c11, c12, c13], [c12, c22, c23], [c13, c23, c33]]
    table = material_table(image.data, MATS, "elastic", grains=grains, orientations=orientations)
    for v in np.flatnonzero(mat.ravel() == 3):
        n1, n2 = axes[zone.ravel()[v] - 1, :3], axes[zone.ravel()[v] - 1, 3:]
        q = np.column_stack([n1, n2, np.cross(n1, n2)])
        turned = np.einsum("ia,jb,kc,ld,abcd->ijkl", q, q, q, q, voigt_to_tensor4(local))
        expected = table.matrices[table.mat[v]]
        assert np.allclose(tensor4_to_voigt(turned), expected, atol=1e-9 * 168e3)
    # the loadings: the six unit strains, one partial loading each
    tags = ET.parse(tmp_path / "poly_stiffness.xml").getroot().findall("Loading")
    assert [t.get("Tag") for t in tags] == ["1", "2", "3", "4", "5", "6"]
    for k, tag in enumerate(tags):
        drive = _drivings(tag)
        assert set(drive) == set(COMPONENTS)
        assert all(d == "Strain" for d, _ in drive.values())
        assert [drive[c][1] for c in COMPONENTS] == list(np.eye(6)[k])
        assert tag.find("Time_Discretization").get("Tfinal") == f"{k + 1.0!r}"
    algo = ET.parse(tmp_path / "poly_algorithm.xml").getroot()
    assert algo.find("Mechanics/Small_Perturbations").get("Value") == "true"
    assert algo.find("Algorithm/Convergence_Criterion").get("Value") == "Default"
    script = (tmp_path / "poly_run.sh").read_text(encoding="utf-8")
    assert (
        "amitex_fftp -nm poly_materials.vtk -nz poly_zones.vtk -m poly_material.xml -c "
        "poly_stiffness.xml -a poly_algorithm.xml -s results/poly_stiffness"
    ) in script
    manifest = json.loads((tmp_path / "poly_amitex.json").read_text(encoding="utf-8"))
    assert [m["phase"] for m in manifest["materials"]] == [0, 1, 2]
    assert manifest["materials"][2]["grains"] == [int(g) for g in present]
    assert set(export.runs) == {"stiffness"} and all(p.exists() for p in export.files)


def test_amitex_loadings_and_errors(tmp_path):
    """Named loadings and paths, finite strains; what AMITEX's native laws cannot do."""
    image, _ = polycrystal()
    tension = Loading.mixed(strain={"11": 0.01}, stress={"22": 0, "33": 0, "23": 0, "31": 0,
                            "12": 5.0})  # fmt: skip
    path = LoadingPath.from_spec({"times": [0, 10, 30], "increments": [5, 2],
                                  "strain": {"12": [0, 0.02, 0.0]},
                                  "stress": {"11": [0, 100, 100]}})  # fmt: skip
    export = io.write_amitex(tmp_path, image, MATS, loadings={"tension": tension},
                             paths={"cycle": path}, name="b", tolerance=1e-5)  # fmt: skip
    assert set(export.runs) == {"tension", "cycle"}  # no stiffness without asking
    drive = _drivings(ET.parse(tmp_path / "b_tension.xml").getroot().find("Loading"))
    assert drive["xx"] == ("Strain", 0.01) and drive["xy"] == ("Stress", 5.0)
    assert drive["yy"] == drive["zz"] == drive["yz"] == drive["xz"] == ("Stress", 0.0)
    tags = ET.parse(tmp_path / "b_cycle.xml").getroot().findall("Loading")
    steps = [(t.find("Time_Discretization").get("Nincr"),
              t.find("Time_Discretization").get("Tfinal")) for t in tags]  # fmt: skip
    assert steps == [("5", "10.0"), ("2", "30.0")]
    assert [_drivings(t)["xy"] for t in tags] == [("Strain", 0.02), ("Strain", 0.0)]
    assert [_drivings(t)["xx"] for t in tags] == [("Stress", 100.0), ("Stress", 100.0)]
    algo = ET.parse(tmp_path / "b_algorithm.xml").getroot()
    assert algo.find("Algorithm/Convergence_Criterion").get("Value") == "1e-05"
    # finite strains: the gradient upper triangular, as morphoGen's
    tension = (1.0, 10, (0, 0, 0.05, 0, 0, 0), (False, False, True, False, False, False))
    io.write_amitex_loading(tmp_path / "gd.xml", [tension], kinematics="finite")
    drive = _drivings(ET.parse(tmp_path / "gd.xml").getroot().find("Loading"))
    assert drive["zz"] == ("Strain", 0.05) and drive["xx"] == ("Stress", 0.0)
    assert drive["yx"] == drive["zx"] == drive["zy"] == ("Strain", 0.0) and len(drive) == 9
    plastic = dict(MATS)
    plastic[1] = Material(young=70e3, poisson=0.3, plasticity=Plasticity(yield_stress=100))
    with pytest.raises(ValueError, match="unit strains"):  # MFront laws: along paths
        io.write_amitex(tmp_path / "p", image, plastic)
    io.write_amitex(tmp_path / "p", image, plastic, elastic_only=True)
    law = {"library": "libBehaviour.so", "behaviour": "Law"}  # no source to compile for umat
    own = {**MATS, 1: Material(young=70e3, poisson=0.3, mfront=law)}
    with pytest.raises(ValueError, match="given by its library"):
        io.write_amitex(tmp_path / "m", image, own, paths={"cycle": path})
    with pytest.raises(ValueError, match="change of temperature"):
        hot = Loading.strain(np.zeros(6), temperature=10.0)
        io.write_amitex(tmp_path / "t", image, MATS, loadings={"hot": hot})
    d = np.array(MATS[2].stiffness, dtype=float)
    d[0, 3] = d[3, 0] = 1e3
    with pytest.raises(ValueError, match="orthotropic"):
        io.write_amitex(tmp_path / "o", image, {**MATS, 2: Material.anisotropic(d)})


def test_amitex_mfront_laws(tmp_path):
    """Nonlinear materials as MFront laws of the umat interface: their sources, library,
    coefficients (those that the interface imposes, the axes of each zone for crystals)
    and internal variables, their compilation by the script, along paths only."""
    from morphogen.io.amitex import UMAT_LIBRARY
    from morphogen.io.mfront import mfront_source, orthotropic_constants

    image, grains = polycrystal()
    orientations = random_orientations(np.random.default_rng(2), 7)
    plast = Plasticity(yield_stress=100, hardening=500, kinematic=((2e4, 200), (5e3, 0)),
                       viscosity=50, exponent=5)  # fmt: skip
    cp = CrystalPlasticity.from_config({"families": ["octahedral"], "tau0": 40, "K": 20, "n": 12})
    mats = {0: Material(void=True),
            1: Material(young=70e3, poisson=0.3, plasticity=plast, density=2.7e-9),
            2: dataclasses.replace(MATS[2], crystal_plasticity=cp)}  # fmt: skip
    path = LoadingPath.from_spec({"times": [0, 10], "increments": 5, "strain": {"11": 0.01}})
    export = io.write_amitex(tmp_path, image, mats, grains=grains, orientations=orientations,
                             paths={"tension": path}, name="nl-1")  # fmt: skip
    assert set(export.runs) == {"tension"}
    root = ET.parse(tmp_path / "nl-1_material.xml").getroot()
    void, mises, crystal = root.findall("Material")
    assert void.get("Law") == "elasiso"
    for el, phase in ((mises, 1), (crystal, 2)):
        behaviour = f"nl1Phase{phase}"
        assert el.get("Lib") == UMAT_LIBRARY and el.get("Law") == f"umat{behaviour.lower()}"
        text = (tmp_path / "mfront" / f"{behaviour}.mfront").read_text(encoding="utf-8")
        assert text == mfront_source(mats[phase], behaviour)
    coeffs = _coeffs(mises)  # YOUN, NU, RHO, ALPH
    assert [float(coeffs[k].get("Value")) for k in range(1, 5)] == [70e3, 0.3, 2.7e-9, 0.0]
    assert len(mises.findall("IntVar")) == 7 + 6 * 2
    coeffs = _coeffs(crystal)
    assert sorted(coeffs) == list(range(1, 20))
    constants = [float(coeffs[k].get("Value")) for k in range(1, 10)]
    assert constants == pytest.approx(orthotropic_constants(MATS[2].stiffness))
    axes = np.column_stack([_bin(tmp_path / coeffs[k].get("File")) for k in range(10, 16)])
    table = material_table(image.data, mats, "elastic", grains=grains, orientations=orientations)
    rows = table.mat[image.data.ravel() == 2]
    g = table.rotations[rows.min() : rows.max() + 1]
    assert np.allclose(axes, np.hstack([g[:, 0], g[:, 1]]))  # V1, V2: rows of g, per zone
    assert [float(coeffs[k].get("Value")) for k in range(16, 20)] == [0.0] * 4
    assert len(crystal.findall("IntVar")) == 7 + 3 * 12
    script = (tmp_path / "nl-1_run.sh").read_text(encoding="utf-8")
    build = "(cd mfront && mfront --obuild --interface=umat nl1Phase1.mfront nl1Phase2.mfront)"
    assert build in script and script.index(build) < script.index("amitex_fftp")
    manifest = json.loads((tmp_path / "nl-1_amitex.json").read_text(encoding="utf-8"))
    entry = manifest["materials"][2]
    assert entry["mfront"] == "mfront/nl1Phase2.mfront" and entry["internal_variables"] == 43
    with pytest.raises(ValueError, match="unit strains"):
        io.write_amitex(tmp_path / "s", image, mats, grains=grains, orientations=orientations,
                        paths={"tension": path}, stiffness=True)  # fmt: skip


def test_amitex_finite_strains(tmp_path):
    """Runs in finite strains: Small_Perturbations false, the nine components of the paths,
    every phase but the voids an MFront law of finite strains (Hencky's elasticity for the
    elastic ones; the functions _malls of Hencky's strain measure, the kernel of crystals
    on the deformation gradient), their coefficients and internal variables, the elasticity
    of the nonlinear phases, the errors, and the results read in finite strains."""
    from morphogen.io.mfront import mfront_source

    image, grains = polycrystal()
    data = image.data.copy()
    data[0, 0, :2] = 3, 4
    image = VoxelImage(data, spacing=image.spacing, origin=image.origin)
    orientations = random_orientations(np.random.default_rng(2), 7)
    cp = CrystalPlasticity.from_config({"families": ["octahedral"], "tau0": 40, "K": 20, "n": 12})
    mats = {0: Material(void=True), 1: Material(young=70e3, poisson=0.3, density=2.7e-9),
            2: MATS[2],
            3: Material(young=70e3, poisson=0.33, plasticity=Plasticity(yield_stress=100)),
            4: dataclasses.replace(MATS[2], crystal_plasticity=cp)}  # fmt: skip
    path = LoadingPath.from_spec({"times": [0, 10], "increments": 5,
                                  "strain": {"11": 0.05, "12": 0.02}})  # fmt: skip
    export = io.write_amitex(tmp_path, image, mats, grains=grains, orientations=orientations,
                             paths={"pull": path}, name="gd", kinematics="finite")  # fmt: skip
    assert set(export.runs) == {"pull"}
    algo = ET.parse(tmp_path / "gd_algorithm.xml").getroot()
    assert algo.find("Mechanics/Small_Perturbations").get("Value") == "false"
    drive = _drivings(ET.parse(tmp_path / "gd_pull.xml").getroot().find("Loading"))
    assert len(drive) == 9 and drive["xx"] == ("Strain", 0.05) and drive["xy"] == ("Strain", 0.02)
    assert drive["yx"] == drive["zx"] == drive["zy"] == ("Strain", 0.0)
    root = ET.parse(tmp_path / "gd_material.xml").getroot()
    void, iso, cubic, mises, crystal = root.findall("Material")
    assert void.get("Law") == "elasiso"
    for el, phase, function in ((iso, 1, "_malls"), (cubic, 2, "_malls"), (mises, 3, "_malls"),
                                (crystal, 4, "")):  # fmt: skip
        behaviour = f"gdPhase{phase}"
        assert el.get("Law") == f"umat{behaviour.lower()}{function}"
        text = (tmp_path / "mfront" / f"{behaviour}.mfront").read_text(encoding="utf-8")
        assert text == mfront_source(mats[phase], behaviour, kinematics="finite")
    assert [float(c.get("Value")) for c in iso.findall("Coeff")] == [70e3, 0.3, 2.7e-9, 0.0]
    assert len(iso.findall("IntVar")) == 1  # none, one at least for Cast3M
    assert sorted(_coeffs(cubic)) == list(range(1, 20)) and len(cubic.findall("IntVar")) == 1
    assert len(mises.findall("IntVar")) == 7 and len(crystal.findall("IntVar")) == 49 + 1
    script = (tmp_path / "gd_run.sh").read_text(encoding="utf-8")
    assert "mfront --obuild --interface=umat gdPhase1.mfront gdPhase2.mfront gdPhase3.mfront " \
           "gdPhase4.mfront" in script  # fmt: skip
    manifest = json.loads((tmp_path / "gd_amitex.json").read_text(encoding="utf-8"))
    assert manifest["kinematics"] == "finite"
    # the elasticity of the nonlinear phases: Hencky's
    io.write_amitex(
        tmp_path / "el",
        image,
        mats,
        grains=grains,
        orientations=orientations,
        paths={"pull": path},
        name="el",
        kinematics="finite",
        elastic_only=True,
    )
    root = ET.parse(tmp_path / "el" / "el_material.xml").getroot()
    mises = root.findall("Material")[3]
    assert mises.get("Law") == "umatelphase3_malls" and len(mises.findall("IntVar")) == 1
    text = (tmp_path / "el" / "mfront" / "elPhase3.mfront").read_text(encoding="utf-8")
    assert "Hencky elasticity" in text and "Brick" not in text
    for kwargs in ({"stiffness": True}, {"loadings": {"t": Loading.strain(np.eye(6)[0])}},
                   {"paths": None}):  # fmt: skip
        options = {"paths": {"pull": path}, **kwargs}
        with pytest.raises(ValueError, match="loading paths only"):
            io.write_amitex(tmp_path / "bad", image, mats, name="b", kinematics="finite",
                            **options)  # fmt: skip
    with pytest.raises(ValueError, match="kinematics"):
        io.write_amitex(tmp_path / "bad", image, MATS, kinematics="large")
    # the results of the run, in finite strains as the export says
    (tmp_path / "results").mkdir()
    line = ["10.0", *_fmt(np.zeros(60)), "5"]
    _std(tmp_path / "results" / "gd_pull.std", [line], comments=["# time"])
    run = io.read_amitex_run(tmp_path)
    assert run.results["pull"].kinematics == "finite" and run.stiffness is None


def _std(path, rows, comments=("# time stress strain",)):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(comments) + "\n")
        for row in rows:
            f.write(" ".join(row) + "\n")


def _fmt(values):
    return [f"{v:.15e}" for v in values]


def test_amitex_results(tmp_path):
    """The averages of the unit cell, the materials and the zones in morphoGen's orders,
    the apparent stiffness of an export; finite strains; numbers written by Fortran."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 6))
    c = a @ a.T + 6 * np.eye(6)
    image, grains = polycrystal()
    io.write_amitex(tmp_path, image, MATS, grains=grains,
                    orientations=random_orientations(rng, 7), name="s")  # fmt: skip
    (tmp_path / "results").mkdir()
    rows = [["0.0", *_fmt(np.zeros(24)), "0"]]
    for k in range(6):
        stress, strain = c[:, k], np.eye(6)[k]
        std = rng.random(12)
        rows.append([f"{k + 1.0}", *_fmt(stress[TO6]), *_fmt(strain[TO6]), *_fmt(std), "7"])
    rows[1][13] = "0.49801817-310"  # the exponent of Fortran without its letter
    _std(tmp_path / "results" / "s_stiffness.std", rows)
    res = io.read_amitex(tmp_path / "results" / "s_stiffness")
    assert res.kinematics == "small" and res.times.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert np.allclose(res.stress[1:].T, c) and np.allclose(res.strain[1:], np.eye(6))
    assert res.stress_std[1, 0] == pytest.approx(4.9801817e-311, rel=1e-6)
    assert res.iterations.tolist() == [0, 7, 7, 7, 7, 7, 7]
    run = io.read_amitex_run(tmp_path)
    assert np.allclose(run.stiffness, c) and set(run.results) == {"stiffness"}
    props = run.properties
    assert props["bulk_modulus"] == pytest.approx(c[:3, :3].sum() / 9)
    assert props["shear_modulus"] == pytest.approx(c[5, 5]) and "E1" in props
    # materials (.mstd) and zones (.zstd): one row each per increment
    rows = []
    for t in (1.0, 2.0):
        for m in (1, 2, 3):
            rows.append([f"{t}", *_fmt(m * t * np.arange(1, 25)[:24])])
    _std(tmp_path / "results" / "s_stiffness.mstd", rows)
    per = io.read_amitex_materials(tmp_path / "results" / "s_stiffness")
    assert sorted(per) == [1, 2, 3]
    assert np.allclose(per[2].stress[:, 0], [2.0, 4.0]) and per[2].iterations is None
    assert np.allclose(per[3].stress[1], 6.0 * np.arange(1, 7)[TO6])
    rows = [[f"{t}", *_fmt(np.full(24, z)), f"{10 * z}", f"{t}"] for t in (1.0, 2.0)
            for z in (1, 2)]  # fmt: skip
    _std(tmp_path / "results" / "s_stiffness_3.zstd", rows)
    zones = io.read_amitex_zones(tmp_path / "results" / "s_stiffness", 3)
    assert sorted(zones) == [1, 2] and zones[2].extra.tolist() == [[20.0, 1.0], [20.0, 2.0]]
    # finite strains: Cauchy, first Piola-Kirchhoff, Green-Lagrange, gradient and their
    # standard deviations
    pk1, grad = rng.normal(size=9), rng.normal(size=9)
    cauchy, gl = rng.normal(size=6), rng.normal(size=6)
    line = ["1.0", *_fmt(cauchy[TO6]), *_fmt(pk1[TO9]), *_fmt(gl[TO6]), *_fmt(grad[TO9]),
            *_fmt(np.zeros(30)), "4"]  # fmt: skip
    _std(tmp_path / "gd.std", [line], comments=["# xx,yy,zz,xy,xz,yz,yx,zx,zy"])
    res = io.read_amitex(tmp_path / "gd")
    assert res.kinematics == "finite" and np.allclose(res.pk1[0], pk1)
    assert np.allclose(res.gradient[0], grad) and np.allclose(res.cauchy[0], cauchy)
    assert np.allclose(res.stress[0], pk1[[0, 4, 8, 5, 2, 1]]) and res.iterations.tolist() == [4]
    with pytest.raises(ValueError, match="columns"):
        _std(tmp_path / "short.std", [["1.0", "2.0"]])
        io.read_amitex(tmp_path / "short")


def test_amitex_configuration_and_command_line(tmp_path, capsys):
    """outputs.amitex in a run, its errors; morphogen amitex export and read."""
    cfg = {
        "model": {"type": "boolean_spheres", "radius": 0.6, "volume_fraction": 0.2},
        "domain": {"size": 3.0, "periodic": True},
        "seed": 1,
        "voxels": {"shape": 8},
        "outputs": {"directory": str(tmp_path / "out"), "name": "sph",
                    "amitex": {"processes": 2}},
        "homogenization": {"method": "fft", "boundary_conditions": ["periodic"],
                           "materials": {0: {"young": 7e4, "poisson": 0.33},
                                         1: {"young": 4e5, "poisson": 0.2}},
                           "properties": ["bulk_modulus"]},
    }  # fmt: skip
    outdir = pipeline.run(load_config(cfg))
    target = outdir / "sph_0000_amitex"
    assert (target / "sph_0000_material.xml").exists()
    assert "mpirun -np ${NP:-2}" in (target / "sph_0000_run.sh").read_text(encoding="utf-8")
    # a plastic phase: its MFront law along the paths, without the unit strains
    homog = {**cfg["homogenization"], "paths": {"pull": {"times": [0, 1], "strain": {"11": 0.002}}},
             "materials": {0: {"young": 7e4, "poisson": 0.33, "plasticity": {"yield": 100}},
                           1: {"young": 4e5, "poisson": 0.2}}}  # fmt: skip
    plastic = {**cfg, "outputs": {**cfg["outputs"], "name": "pl"}, "homogenization": homog}
    target = pipeline.run(load_config(plastic)) / "pl_0000_amitex"
    assert (target / "mfront" / "pl0000Phase0.mfront").exists()
    assert (target / "pl_0000_pull.xml").exists()
    assert not (target / "pl_0000_stiffness.xml").exists()
    # finite strains: the paths, every phase an MFront law (the elastic one, Hencky's)
    finite = {**cfg, "outputs": {**cfg["outputs"], "name": "gd"},
              "homogenization": {**homog, "nonlinear": {"kinematics": "finite"}}}  # fmt: skip
    target = pipeline.run(load_config(finite)) / "gd_0000_amitex"
    algo = ET.parse(target / "gd_0000_algorithm.xml").getroot()
    assert algo.find("Mechanics/Small_Perturbations").get("Value") == "false"
    assert "Hencky elasticity" in (target / "mfront" / "gd0000Phase1.mfront").read_text()
    no_paths = {**cfg["homogenization"], "nonlinear": {"kinematics": "finite"}}
    for change, match in [
        ({"homogenization": None}, "materials"),
        ({"outputs": {**cfg["outputs"], "amitex": {"bad": 1}}}, "bad"),
        ({"outputs": {**cfg["outputs"], "amitex": {"tolerance": 0.1}}}, "1e-3"),
        ({"homogenization": no_paths}, "runs of AMITEX"),
    ]:
        data = {**cfg, **change}
        if data["homogenization"] is None:
            del data["homogenization"]
        with pytest.raises(ConfigError, match=match):
            load_config(data)
    image, _ = polycrystal()
    io.save_image(image, tmp_path / "poly.vti")
    mats = tmp_path / "mats.json"
    mats.write_text(json.dumps({str(p): m.to_config() for p, m in MATS.items()}), encoding="utf-8")
    assert cli.main(["amitex", "export", str(tmp_path / "poly.vti"), "--materials", str(mats),
                     "-o", str(tmp_path / "cmd"), "--loading", "shear: E12=0.01"]) == 0  # fmt: skip
    names = {p.name for p in (tmp_path / "cmd").iterdir()}
    assert {"poly_shear.xml", "poly_material.xml", "poly_amitex.json"} <= names
    assert cli.main(["amitex", "read", str(tmp_path / "cmd"), "--json"]) == 0
    assert cli.main(["amitex", "read", str(tmp_path / "cmd"), "-o", str(tmp_path / "t.csv")]) != 0
    assert "stiffness" in capsys.readouterr().err
    assert cli.main(["amitex", "export", str(tmp_path / "poly.vti"), "--materials", str(mats),
                     "-o", str(tmp_path / "gd"), "--path", "pull: t=0,1 n=2 E11=0.01",
                     "--kinematics", "finite"]) == 0  # fmt: skip
    algo = ET.parse(tmp_path / "gd" / "poly_algorithm.xml").getroot()
    assert algo.find("Mechanics/Small_Perturbations").get("Value") == "false"
    assert cli.main(["amitex", "export", str(tmp_path / "poly.vti"), "--materials", str(mats),
                     "-o", str(tmp_path / "cmd2"), "--name", "q"]) == 0  # fmt: skip
    (tmp_path / "cmd2" / "results").mkdir()
    rows = [[f"{k + 1.0}", *_fmt(np.eye(6)[k][TO6] * 3.0), *_fmt(np.eye(6)[k][TO6]),
             *_fmt(np.zeros(12)), "3"] for k in range(6)]  # fmt: skip
    _std(tmp_path / "cmd2" / "results" / "q_stiffness.std", rows)
    assert cli.main(["amitex", "read", str(tmp_path / "cmd2"), "-o", str(tmp_path / "t.csv")]) == 0
    with open(tmp_path / "t.csv", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert float(row["C11"]) == 3.0 and float(row["bulk_modulus"]) == pytest.approx(1.0)
    assert float(row["S11"]) == pytest.approx(1 / 3.0) and float(row["S66"]) == pytest.approx(1 / 3)
    assert float(row["volume"]) == pytest.approx(6 * 0.5 * 5 * 0.25 * 4 * 0.4)
