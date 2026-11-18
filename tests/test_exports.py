"""Inputs of other programs (periodic equations, DAMASK, Neper, FEniCSx) and images
written as XDMF, at once or slab by slab."""

import dataclasses
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

import morphogen as mg
from morphogen import io, pipeline
from morphogen.config import ConfigError, load_config
from morphogen.homogenization import Material
from morphogen.homogenization.porous import PorousPlasticity
from morphogen.io.solvers import euler_to_quaternion, periodic_images
from morphogen.mesh.conforming import conforming_mesh, has_mmg
from morphogen.mesh.voxel import voxel_mesh
from morphogen.orientations import euler_to_matrix, random_orientations


def parse_equations(text):
    """Reference nodes and equations (list of (node, dof, coefficient) terms)."""
    lines = [line.strip() for line in text.splitlines() if not line.startswith("**")]
    refs, equations, i = {}, [], 0
    while i < len(lines):
        if lines[i].startswith("*NSET, NSET=REF_"):
            refs[lines[i][-1]] = int(lines[i + 1])
        if lines[i] == "*EQUATION":
            n = int(lines[i + 1])
            values, j = [], i + 2
            while len(values) < 3 * n:
                values += [float(v) for v in lines[j].split(",")]
                j += 1
            equations.append(
                [(int(values[k]), int(values[k + 1]), values[k + 2]) for k in range(0, 3 * n, 3)]
            )
            i = j
            continue
        i += 1
    return refs, equations


def check_equations(mesh, length, tmp_path):
    text = io.write_periodic_equations(tmp_path / "eq.inp", mesh).read_text(encoding="utf-8")
    refs, equations = parse_equations(text)
    slaves, masters, _ = periodic_images(mesh)
    assert len(equations) == 3 * len(slaves) and set(refs) == {"X", "Y", "Z"}
    dependent = [eq[0][:2] for eq in equations]
    assert len(set(dependent)) == len(dependent)  # each dependent dof once
    assert not set(masters) & set(slaves)
    # an affine field u = E x with u(REF_d) = E L_d e_d satisfies every equation
    e = np.array([[0.1, 0.02, -0.03], [0.04, -0.05, 0.06], [0.07, 0.01, 0.08]])
    u = {k + 1: e @ x for k, x in enumerate(mesh.nodes)}
    for d, axis in enumerate("XYZ"):
        u[refs[axis]] = e @ (length * np.eye(3)[d])
    for eq in equations:
        assert abs(sum(c * u[n][dof - 1] for n, dof, c in eq)) < 1e-12
    assert "*NSET, NSET=CORNER" in text


def test_periodic_equations_of_voxel_meshes(tmp_path):
    img = mg.VoxelImage(np.random.default_rng(0).integers(0, 2, (4, 3, 5)).astype(np.uint8),
                        spacing=(0.5, 1.0, 0.25), periodic=True)  # fmt: skip
    mesh = voxel_mesh(img)
    check_equations(mesh, np.array([2.0, 3.0, 1.25]), tmp_path)
    with pytest.raises(ValueError, match="matching"):
        io.write_periodic_equations(tmp_path / "x.inp", voxel_mesh(img, phases=[1]))
    thermal = io.write_periodic_equations(tmp_path / "t.inp", mesh, physics="thermal").read_text(
        encoding="utf-8"
    )
    _, equations = parse_equations(thermal)
    assert len(equations) == len(periodic_images(mesh)[0])
    assert {dof for eq in equations for _, dof, _ in eq} == {11}


@pytest.mark.skipif(not has_mmg(), reason="MMG (pymmg) is not installed")
def test_periodic_equations_and_fenicsx_of_polycrystals(tmp_path):
    tess = mg.models.Voronoi(n_cells=8, crystal_orientations=True).generate(
        mg.Domain.cube(1.0, periodic=True), seed=2
    )
    mesh = conforming_mesh(tess, size=0.3)
    check_equations(mesh, np.ones(3), tmp_path)
    path = io.save_mesh(mesh, tmp_path / "m.xdmf")
    root = ET.parse(path).getroot()
    grids = {g.get("Name"): g for g in root.iter("Grid")}
    assert list(grids) == ["mesh", "phase", "grain", "faces"]
    topo = grids["mesh"].find("Topology")
    assert int(topo.get("NumberOfElements")) == mesh.n_elements
    faces = int(grids["faces"].find("Topology").get("NumberOfElements"))
    assert faces == len(mesh.outer_faces())
    grain = grids["grain"].find("Attribute/DataItem").text.split()
    assert np.array_equal(np.array(grain, dtype=int), mesh.cell)


def test_damask_and_neper(tmp_path):
    rng = np.random.default_rng(4)
    euler = random_orientations(rng, 6)
    q = euler_to_quaternion(euler)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0) and np.all(q[:, 0] >= 0)
    for qi, g in zip(q, euler_to_matrix(euler), strict=True):  # DAMASK qu2om, P = -1
        q0, q1, q2, q3 = qi
        qq = q0**2 - (q1**2 + q2**2 + q3**2)
        om = np.array(
            [
                [qq + 2 * q1 * q1, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
                [2 * (q2 * q1 - q0 * q3), qq + 2 * q2 * q2, 2 * (q2 * q3 + q0 * q1)],
                [2 * (q3 * q1 + q0 * q2), 2 * (q3 * q2 - q0 * q1), qq + 2 * q3 * q3],
            ]
        )
        assert np.allclose(om, g)
    tess = mg.models.Laguerre(
        n_cells=7, radius=mg.distributions.Uniform(0.05, 0.1), crystal_orientations=True
    ).generate(mg.Domain.cube(1.0, periodic=True), seed=1)
    labels = tess.labels((10, 10, 10))
    cu = Material.cubic(168.4e9, 121.4e9, 75.4e9)
    vti, mat = io.write_damask(tmp_path / "g", labels, tess.orientations, materials={1: cu})
    grid = io.load_image(vti, "material")
    used = np.unique(labels.data)
    assert np.array_equal(used[grid.data], labels.data)
    config = yaml.safe_load(mat.read_text(encoding="utf-8"))
    assert len(config["material"]) == len(used)
    assert config["phase"]["phase_1"]["mechanical"]["elastic"]["C_44"] == pytest.approx(75.4e9)
    first = config["material"][0]["constituents"][0]
    assert np.allclose(first["O"], euler_to_quaternion(tess.orientations[used[0]])[0])
    files = io.write_neper(tmp_path / "t", tess)
    names = {f.name for f in files}
    assert names == {"t_seeds.txt", "t_weights.txt", "t_ori.txt", "t_neper.sh"}
    assert np.allclose(np.loadtxt(tmp_path / "t_weights.txt") ** 2, tess.weights)
    script = (tmp_path / "t_neper.sh").read_text(encoding="utf-8")
    assert "-n 7" in script and "-periodicity all" in script and "euler-bunge" in script


def test_xdmf_images_and_chunked_volumes(tmp_path):
    rng = np.random.default_rng(1)
    phase = mg.VoxelImage(rng.integers(0, 3, (5, 6, 7)).astype(np.uint8), spacing=(0.5, 1, 2),
                          origin=(1, 2, 3), name="phase")  # fmt: skip
    field = phase.with_data(rng.random((5, 6, 7)).astype(np.float32), name="field")
    io.save_image([phase, field], tmp_path / "a.xdmf")
    for img in (phase, field):
        back = io.load_image(tmp_path / "a.xdmf", img.name)
        assert np.array_equal(back.data, img.data) and back.data.dtype == img.data.dtype
        assert back.spacing == img.spacing and back.origin == img.origin
    models = [
        mg.models.BooleanSpheres(radius=0.15, volume_fraction=0.3),
        mg.models.Voronoi(n_cells=20),
        mg.models.GaussianExcursion(correlation_length=0.1, volume_fraction=0.4),
    ]
    for model in models:
        micro = model.generate(mg.Domain.cube(1.0, periodic=True), seed=3)
        full = micro.voxelize((12, 10, 11))
        out = io.voxelize_to_file(micro, tmp_path / f"{model.name}.xdmf", (12, 10, 11), slab=3)
        assert np.array_equal(io.load_image(out).data, full.data)
        mhd = io.voxelize_to_file(micro, tmp_path / f"{model.name}.mhd", (12, 10, 11), slab=4)
        assert np.array_equal(io.load_image(mhd).data, full.data)
    out = io.voxelize_to_file(micro, tmp_path / "labels.xdmf", (12, 10, 11), labels=False)
    voronoi = models[1].generate(mg.Domain.cube(1.0, periodic=True), seed=3)
    lab = io.voxelize_to_file(voronoi, tmp_path / "cells.xdmf", (8, 8, 8), labels=True, slab=3)
    assert np.array_equal(io.load_image(lab).data, voronoi.labels((8, 8, 8)).data)
    with pytest.raises(ValueError, match="xdmf"):
        io.voxelize_to_file(voronoi, tmp_path / "x.vti", (4, 4, 4))


def test_export_outputs_of_runs(tmp_path):
    data = {
        "model": {"type": "voronoi", "n_cells": 6, "crystal_orientations": True},
        "domain": {"size": 1.0, "periodic": True},
        "seed": 3,
        "voxels": {"shape": 8},
        "outputs": {
            "directory": str(tmp_path / "out"),
            "name": "g",
            "mesh": ["inp"],
            "periodic_equations": True,
            "damask": True,
            "neper": True,
        },
    }
    outdir = pipeline.run(load_config(data))
    for name in ("g_0000_equations.inp", "g_0000_damask.vti", "g_0000_damask_material.yaml",
                 "g_0000_seeds.txt", "g_0000_neper.sh"):  # fmt: skip
        assert (outdir / name).exists(), name
    for change, message in (
        (
            {"model": {"type": "boolean_spheres", "radius": 0.1, "volume_fraction": 0.2}},
            "tessellations",
        ),
        ({"model": {"type": "voronoi", "n_cells": 6}}, "orientations"),
        ({"domain": {"size": 1.0, "periodic": False}}, "periodic"),
    ):
        bad = {**data, **change}
        with pytest.raises(ConfigError, match=message):
            load_config(bad)
    bad = {**data, "outputs": {"periodic_equations": True}}
    with pytest.raises(ConfigError, match="nodes of a mesh"):
        load_config(bad)


def _two_grains():
    """A box of six tetrahedra in two grains (phases 0 and 1), the second one turned."""
    from helpers import box_mesh

    mesh = box_mesh((1, 1, 1), phase=lambda c: (c[:, 0] + c[:, 1] > 1.15).astype(int))
    turned = np.array([[0.0, 0.0, 0.0], [30.0, 40.0, 50.0]])
    return dataclasses.replace(mesh, cell=mesh.phase.copy(), orientations=turned)


def _anisotropic(seed=0):
    """A stiffness with 21 distinct constants (positive definite)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(6, 6))
    return 2e4 * (a @ a.T) + 1e5 * np.eye(6)


def test_abaqus_materials(tmp_path):
    """The materials of Abaqus inputs: isotropic and anisotropic elasticity (the 21
    constants in Abaqus's order, components 11, 22, 33, 12, 13, 23), expansion, density,
    voids, the sections of the grains with their orientations; the nonlinear materials as
    MFront laws of the abaqus interface (their names, internal variables, sources)."""
    from morphogen.homogenization import CrystalPlasticity, Plasticity
    from morphogen.io.meshes import abaqus_anisotropic

    c = _anisotropic()
    expected = [c[0, 0], c[0, 1], c[1, 1], c[0, 2], c[1, 2], c[2, 2],  # D1111 ... D3333
                c[0, 5], c[1, 5], c[2, 5], c[5, 5],                    # D1112 ... D1212
                c[0, 4], c[1, 4], c[2, 4], c[5, 4], c[4, 4],           # D1113 ... D1313
                c[0, 3], c[1, 3], c[2, 3], c[5, 3], c[4, 3], c[3, 3]]  # fmt: skip
    assert abaqus_anisotropic(c) == pytest.approx(expected, rel=1e-15)
    mesh = _two_grains()
    mats = {0: Material(young=70e3, poisson=0.3, expansion=2e-5, density=2.7e-9),
            1: Material.anisotropic(c, expansion=(1e-5, 2e-5, 3e-5))}  # fmt: skip
    text = io.save_mesh(mesh, tmp_path / "el.inp", materials=mats).read_text(encoding="utf-8")
    block = text.split("*MATERIAL, NAME=PHASE_0\n")[1].split("*MATERIAL")[0]
    assert block == "*ELASTIC\n70000, 0.3\n*EXPANSION\n2e-05\n*DENSITY\n2.7e-09\n"
    block = text.split("*MATERIAL, NAME=PHASE_1\n*ELASTIC, TYPE=ANISOTROPIC\n")[1]
    rows = block.split("*")[0].splitlines()
    assert [len(r.split(",")) for r in rows] == [8, 8, 5]
    assert [float(v) for r in rows for v in r.split(",")] == pytest.approx(expected, rel=1e-14)
    assert "*EXPANSION, TYPE=ORTHO\n1e-05, 2e-05, 3e-05\n" in text
    # the expanding materials count their expansion from the nodes' initial 293.15 K
    assert text.endswith(
        f"*NSET, NSET=ALL_NODES, GENERATE\n1, {mesh.n_nodes}, 1\n"
        "*INITIAL CONDITIONS, TYPE=TEMPERATURE\nALL_NODES, 293.15\n"
    )
    assert "*SOLID SECTION, ELSET=CELL_1, MATERIAL=PHASE_1, ORIENTATION=ORI_1\n" in text
    voids = {0: Material(void=True), 1: mats[1]}
    text = io.save_mesh(mesh, tmp_path / "v.inp", materials=voids).read_text(encoding="utf-8")
    assert "phase 0: void" in text and "ELSET=CELL_0," not in text
    # the nonlinear materials: MFront laws, without '_' in the names of their behaviours
    cp = CrystalPlasticity.from_config({"families": ["octahedral"], "tau0": 40, "K": 20, "n": 12})
    crystal = dataclasses.replace(Material.cubic(168e3, 121e3, 75e3), crystal_plasticity=cp)
    plastic = {0: Material(young=70e3, poisson=0.3, plasticity=Plasticity(yield_stress=100)),
               1: crystal}  # fmt: skip
    text = io.save_mesh(mesh, tmp_path / "nl-1.inp", materials=plastic).read_text(encoding="utf-8")
    for p, depvar in ((0, 7), (1, 43)):
        name = f"ABAQUSBEHAVIOUR_NL1PHASE{p}_3D"
        assert (f"*MATERIAL, NAME={name}\n*DEPVAR\n{depvar},\n*USER MATERIAL, CONSTANTS=1, "
                "UNSYMM\n0.\n") in text  # fmt: skip
        assert (tmp_path / "nl-1_mfront" / f"nl1Phase{p}.mfront").exists()
        assert f"ELSET=CELL_{p}, MATERIAL={name}, ORIENTATION=ORI_{p}" in text
    assert "mfront --obuild --interface=abaqus nl1Phase0.mfront nl1Phase1.mfront" in text
    law = {"library": "libBehaviour.so", "behaviour": "Law"}  # no source to compile for Abaqus
    own = {0: Material(young=70e3, poisson=0.3, mfront=law)}
    with pytest.raises(ValueError, match="given by its library"):
        io.save_mesh(mesh, tmp_path / "m.inp", materials=own)


def _calculix_stresses(dat, elset):
    """The stresses (sxx, syy, szz, sxy, sxz, syz) of the elements of a set in a .dat file of
    CalculiX: in the axes of their orientation, whose name ends each row."""
    lines = dat.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if "stresses" in line and elset in line)
    rows = []
    for line in lines[start + 1 :]:
        if not line.strip():
            if rows:
                break
            continue
        rows.append([float(v) for v in line.split()[2:8]])
    return np.array(rows)


def _calculix_check(path, nodes, cell, orientations, materials, points, heat=0.0):
    """CalculiX on an Abaqus input of morphoGen under a homogeneous strain (the
    displacements of every node) and, with ``heat``, a change of temperature from the
    initial 293.15 K of the input (the step at the absolute temperature): the stresses of
    the integration points (``points`` per element) of each cell, turned back from the
    axes of its orientation, against morphoGen's stiffness of the material of the cell
    turned by the same orientation, and its expansion (in the axes of the crystal)."""
    from morphogen.homogenization.materials import rotate_stiffness

    strain = np.array([[1e-3, 2e-4, -3e-4], [2e-4, -5e-4, 4e-4], [-3e-4, 4e-4, 6e-4]])
    lines = ["*STEP", "*STATIC", "*BOUNDARY"]
    for k, x in enumerate(nodes, start=1):
        u = strain @ x
        lines += [f"{k}, {d + 1}, {d + 1}, {u[d]:.12e}" for d in range(3)]
    if heat:
        lines += ["*TEMPERATURE", f"ALL_NODES, {293.15 + heat:.12g}"]
    cells = np.unique(cell)
    lines += [*[f"*EL PRINT, ELSET=CELL_{g}\nS" for g in cells], "*END STEP"]
    with open(path, "a", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    subprocess.run(["ccx", "-i", path.stem], cwd=path.parent, check=True, capture_output=True)
    for g in cells:
        material = materials[g]  # one phase per cell, of the same number
        rot = euler_to_matrix(orientations[g]).reshape(3, 3)  # the axes of ORI_g: rows
        e = strain.copy()
        if heat and material.expands:
            alpha = np.asarray(material.expansion, dtype=float)  # in the axes of the crystal
            e -= heat * rot.T @ (alpha * np.eye(3) if alpha.ndim == 0 else alpha) @ rot
        voigt = np.array([e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[2, 0], 2 * e[0, 1]])
        if material.stiffness is None:
            stiffness = material.matrix("elastic")
        else:
            stiffness = rotate_stiffness(np.asarray(material.stiffness), orientations[g])
        expected = stiffness @ voigt  # morphoGen's order 11, 22, 33, 23, 31, 12
        out = _calculix_stresses(path.with_suffix(".dat"), f"CELL_{g}")
        assert out.shape == (points * np.count_nonzero(cell == g), 6)
        for sxx, syy, szz, sxy, sxz, syz in out:
            local = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
            s = rot.T @ local @ rot  # back to the axes of the sample
            got = [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[2, 0], s[0, 1]]
            assert np.allclose(got, expected, rtol=0, atol=1e-5 * np.abs(expected).max())


def _voxel_grains():
    """A voxel mesh of 3 x 2 x 2 voxels in two grains (labels 0 and 1, phases 0 and 1)."""
    from morphogen.image import VoxelImage

    data = np.zeros((3, 2, 2), dtype=np.uint8)
    data[1:, 1:] = 1
    image = VoxelImage(data, spacing=(0.5, 0.4, 0.3))
    return voxel_mesh(image, labels=VoxelImage(data.astype(np.int32))), np.array(
        [[0.0, 0.0, 0.0], [30.0, 40.0, 50.0]]
    )


def test_abaqus_voxel_materials(tmp_path):
    """The materials of the Abaqus inputs of voxel meshes: with the orientations of the
    labels, the sets of the cells, their orientations and one section per cell; without,
    one section per phase; the errors."""
    mesh, euler = _voxel_grains()
    mats = {0: Material(young=70e3, poisson=0.3), 1: Material.anisotropic(_anisotropic())}
    text = io.save_mesh(mesh, tmp_path / "v.inp", materials=mats,
                        orientations=euler).read_text(encoding="utf-8")  # fmt: skip
    assert "*ELSET, ELSET=CELL_0\n" in text and "*ORIENTATION, NAME=ORI_1" in text
    assert "*MATERIAL, NAME=PHASE_1\n*ELASTIC, TYPE=ANISOTROPIC\n" in text
    for g in (0, 1):
        assert f"*SOLID SECTION, ELSET=CELL_{g}, MATERIAL=PHASE_{g}, ORIENTATION=ORI_{g}" in text
    text = io.save_mesh(mesh, tmp_path / "p.inp", materials=mats).read_text(encoding="utf-8")
    assert "*SOLID SECTION, ELSET=PHASE_1, MATERIAL=PHASE_1\n" in text and "ORI_" not in text
    bare = dataclasses.replace(mesh, label=None)
    with pytest.raises(ValueError, match="labels"):
        io.save_mesh(bare, tmp_path / "b.inp", materials=mats, orientations=euler)
    mixed = dataclasses.replace(mesh, label=np.zeros_like(mesh.label))
    with pytest.raises(ValueError, match="spans the phases"):
        io.save_mesh(mixed, tmp_path / "m.inp", materials=mats, orientations=euler[:1])


def test_abaqus_finite_strains(tmp_path):
    """Abaqus inputs in finite strains: every material an MFront law of finite strains
    (Hencky's elasticity, von Mises in the logarithmic strain space, crystals by
    morphoGen's kernel), the axes of the orthotropic ones in their first six state
    variables (``*INITIAL CONDITIONS, TYPE=SOLUTION``: the rows of the orientation, then
    zeros, as many values for every set, seven on its line and eight on the next ones),
    no orientation in their sections; voxel meshes, whose phases without orientations
    have the axes of the sample; the voxel inputs of a run in finite strains; errors."""
    from helpers import abaqus_initial_conditions as _initial_conditions
    from morphogen.homogenization import CrystalPlasticity, Plasticity
    from morphogen.io.mfront import mfront_source

    mesh = _two_grains()
    cp = CrystalPlasticity.from_config({"families": ["octahedral"], "tau0": 40, "K": 20, "n": 12})
    crystal = dataclasses.replace(Material.cubic(168e3, 121e3, 75e3), crystal_plasticity=cp)
    mats = {0: Material(young=70e3, poisson=0.3, plasticity=Plasticity(yield_stress=100)),
            1: crystal}  # fmt: skip
    text = io.save_mesh(mesh, tmp_path / "gd.inp", materials=mats,
                        kinematics="finite").read_text(encoding="utf-8")  # fmt: skip
    assert "their steps need NLGEOM=YES" in text and "*ORIENTATION," not in text
    for p, depvar in ((0, 7), (1, 49 + 1 + 6)):
        name = f"ABAQUSBEHAVIOUR_GDPHASE{p}_3D"
        assert (f"*MATERIAL, NAME={name}\n*DEPVAR\n{depvar},\n*USER MATERIAL, CONSTANTS=1, "
                "UNSYMM\n0.\n") in text  # fmt: skip
        assert f"*SOLID SECTION, ELSET=CELL_{p}, MATERIAL={name}\n" in text
        source = (tmp_path / "gd_mfront" / f"gdPhase{p}.mfront").read_text(encoding="utf-8")
        assert source == mfront_source(mats[p], f"gdPhase{p}", kinematics="finite")
    initial = _initial_conditions(text)
    g = euler_to_matrix(mesh.orientations[1]).reshape(3, 3)
    values, per_line = initial["CELL_1"]
    assert list(initial) == ["CELL_1"] and per_line == [7] + [8] * 6 + [1]
    assert np.allclose(values[:6], [*g[0], *g[1]]) and not any(values[6:])
    # elasticity: isotropic without state variables, cubic with its axes only
    elastic = {0: Material(young=70e3, poisson=0.3), 1: Material.cubic(168e3, 121e3, 75e3)}
    text = io.save_mesh(mesh, tmp_path / "el.inp", materials=elastic,
                        kinematics="finite").read_text(encoding="utf-8")  # fmt: skip
    assert "NAME=ABAQUSBEHAVIOUR_ELPHASE0_3D\n*USER MATERIAL" in text
    assert "NAME=ABAQUSBEHAVIOUR_ELPHASE1_3D\n*DEPVAR\n6,\n" in text and "*ELASTIC" not in text
    assert "Hencky" in (tmp_path / "el_mfront" / "elPhase1.mfront").read_text(encoding="utf-8")
    values, per_line = _initial_conditions(text)["CELL_1"]
    assert per_line == [6] and np.allclose(values, [*g[0], *g[1]])
    voxels, _ = _voxel_grains()  # the phases of a voxel mesh, in the axes of the sample
    text = io.save_mesh(voxels, tmp_path / "vx.inp", materials=elastic,
                        kinematics="finite").read_text(encoding="utf-8")  # fmt: skip
    assert _initial_conditions(text) == {"PHASE_1": ([1, 0, 0, 0, 1, 0], [6])}
    assert "TYPE=TEMPERATURE" not in text  # no material expands
    hot = {0: Material(young=7e4, poisson=0.3, expansion=1e-5)}  # its law, from 293.15 K
    text = io.save_mesh(mesh, tmp_path / "t.inp", materials=hot,
                        kinematics="finite").read_text(encoding="utf-8")  # fmt: skip
    assert text.endswith("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nALL_NODES, 293.15\n")
    with pytest.raises(ValueError, match="no stress-free expansion in finite strains"):
        io.save_mesh(mesh, tmp_path / "b.inp", kinematics="finite",
                     materials={0: dataclasses.replace(mats[0], expansion=1e-5)})  # fmt: skip
    # porous plasticity: GTN of the brick refused so, morphoGen's kernel (Rousselier's
    # criterion) computes its thermal strain (its internal variables, then p and f)
    gtn = Material(young=2e5, poisson=0.3, expansion=1e-5, porous_plasticity=PorousPlasticity(
        criterion="gurson", yield_stress=200.0, porosity=0.01))  # fmt: skip
    with pytest.raises(ValueError, match="no stress-free expansion in finite strains"):
        io.save_mesh(mesh, tmp_path / "g.inp", kinematics="finite", materials={0: gtn})
    rousselier = PorousPlasticity(criterion="rousselier", yield_stress=200.0, porosity=0.01,
                                  sigma1=300.0, D=2.0)  # fmt: skip
    own = dataclasses.replace(gtn, porous_plasticity=rousselier)
    text = io.save_mesh(mesh, tmp_path / "r.inp", materials={0: own},
                        kinematics="finite").read_text(encoding="utf-8")  # fmt: skip
    assert "*DEPVAR\n12,\n" in text and text.endswith("ALL_NODES, 293.15\n")
    assert "porous_update" in (tmp_path / "r_mfront" / "rPhase0.mfront").read_text("utf-8")
    with pytest.raises(ValueError, match="kinematics"):
        io.save_mesh(mesh, tmp_path / "k.inp", materials=elastic, kinematics="large")
    # the voxel inputs of a run in finite strains: the grains by their state variables
    cfg = {
        "model": {"type": "voronoi", "n_cells": 4, "crystal_orientations": True},
        "domain": {"size": 1.0, "periodic": True},
        "seed": 3,
        "voxels": {"shape": 6},
        "homogenization": {"method": "fft", "properties": ["bulk_modulus"],
                           "nonlinear": {"kinematics": "finite"},
                           "materials": {1: {"cubic": [168e3, 121e3, 75e3]}}},
        "outputs": {"directory": str(tmp_path / "out"), "name": "grains",
                    "mesh": {"formats": ["inp"]}},
    }  # fmt: skip
    text = (pipeline.run(load_config(cfg)) / "grains_0000.inp").read_text(encoding="utf-8")
    grains = text.count("*ELSET, ELSET=CELL_")
    assert grains >= 2 and len(_initial_conditions(text)) == grains
    assert "*ORIENTATION," not in text and "grains_0000_mfront" in text


@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX (ccx)")
def test_calculix_runs_the_abaqus_materials(tmp_path):
    """CalculiX, which reads Abaqus inputs, under a homogeneous strain: the stresses of an
    isotropic grain and of an anisotropic one (21 distinct constants, turned by its
    orientation), turned back from the axes of their orientations, equal morphoGen's, on
    tetrahedra (C3D4) and on voxels (C3D8, eight integration points); the same grains
    expanding (isotropic, and three principal values in the axes of the crystal), heated
    by 100 K from the initial 293.15 K of the inputs."""
    mesh = _two_grains()
    mats = {0: Material(young=70e3, poisson=0.3), 1: Material.anisotropic(_anisotropic(1))}
    path = io.save_mesh(mesh, tmp_path / "tet.inp", materials=mats)
    _calculix_check(path, mesh.nodes, mesh.cell, mesh.orientations, mats, points=1)
    voxels, euler = _voxel_grains()
    path = io.save_mesh(voxels, tmp_path / "hex.inp", materials=mats, orientations=euler)
    _calculix_check(path, voxels.nodes, voxels.label, euler, mats, points=8)
    hot = {0: Material(young=70e3, poisson=0.3, expansion=2e-5),
           1: Material.anisotropic(_anisotropic(1), expansion=(1e-5, 2e-5, 3e-5))}  # fmt: skip
    path = io.save_mesh(mesh, tmp_path / "hot_tet.inp", materials=hot)
    _calculix_check(path, mesh.nodes, mesh.cell, mesh.orientations, hot, points=1, heat=100.0)
    path = io.save_mesh(voxels, tmp_path / "hot_hex.inp", materials=hot, orientations=euler)
    _calculix_check(path, voxels.nodes, voxels.label, euler, hot, points=8, heat=100.0)


def test_voxel_inp_of_runs(tmp_path):
    """The Abaqus inputs of the voxel meshes of a run with a homogenization section: the
    materials, and the orientations and sections of the grains of a polycrystal."""
    cfg = {
        "model": {"type": "voronoi", "n_cells": 4, "crystal_orientations": True},
        "domain": {"size": 1.0, "periodic": True},
        "seed": 3,
        "voxels": {"shape": 6},
        "homogenization": {"method": "fft", "properties": ["bulk_modulus"],
                           "materials": {1: {"cubic": [168e3, 121e3, 75e3]}}},
        "outputs": {"directory": str(tmp_path / "out"), "name": "grains",
                    "mesh": {"formats": ["inp"]}},
    }  # fmt: skip
    text = (pipeline.run(load_config(cfg)) / "grains_0000.inp").read_text(encoding="utf-8")
    grains = text.count("*ELSET, ELSET=CELL_")
    assert grains >= 2 and text.count("*ORIENTATION, NAME=ORI_") == grains
    assert "*MATERIAL, NAME=PHASE_1\n*ELASTIC, TYPE=ANISOTROPIC\n" in text
    assert text.count(", MATERIAL=PHASE_1, ORIENTATION=ORI_") == grains
