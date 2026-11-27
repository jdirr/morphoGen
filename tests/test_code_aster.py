"""The studies of Code_Aster (morphogen.io.code_aster): the mesh in Code_Aster's format and
its groups, the periodic classes of the nodes, the command file run by a stand-in of
Code_Aster (tests/fake_code_aster.py: its own reading of the mesh, its own linear
tetrahedra, the pairing of LIAISON_GROUP and the Euler angles of Code_Aster's routines)
against morphoGen's finite elements on the same meshes, and the reading of the tables."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

import fake_code_aster as fake
from helpers import box_mesh
from morphogen import cli, io, pipeline
from morphogen.config import ConfigError, load_config
from morphogen.homogenization import Material
from morphogen.homogenization.apparent import conductivity_tensor, make_problem, stiffness_tensor
from morphogen.io.code_aster import (
    STRAIN_CASES,
    aster_groups,
    read_aster_mesh,
    read_aster_table,
    read_code_aster,
    write_aster_mesh,
    write_code_aster,
)
from morphogen.mesh.tetra import TetMesh
from morphogen.orientations import euler_to_matrix

MATERIALS = {
    0: Material(young=3.0, poisson=0.35, conductivity=0.2),
    1: Material(young=70.0, poisson=0.22, conductivity=1.0),
}


def _inclusion(c):
    return (np.linalg.norm(c - [0.5, 0.6, 0.45], axis=1) < 0.35).astype(int)


@pytest.fixture
def mesh():
    return box_mesh((3, 3, 3), upper=(1.0, 1.2, 0.9), phase=_inclusion, perturb=0.2)


def _relative(a, b):
    return float(np.abs(np.asarray(a) - b).max() / np.abs(b).max())


def test_mesh_in_code_aster_format(tmp_path, mesh):
    for m in (mesh, mesh.quadratic()):
        path = write_aster_mesh(tmp_path / "m.mail", m)
        lines = path.read_text().splitlines()
        assert max(len(line) for line in lines) <= 80
        assert lines[-1] == "FIN" and lines[0] == "TITRE"
        back = read_aster_mesh(path)
        np.testing.assert_allclose(back["nodes"], m.nodes, rtol=1e-11, atol=1e-12)
        kind = "TETRA10" if m.order == 2 else "TETRA4"
        names, cells = back["elements"][kind]
        np.testing.assert_array_equal(cells, m.elements)
        assert names[0] == "M1" and len(names) == m.n_elements
        # the phases, the triangles of the faces and the nodes of the faces
        for p in (0, 1):
            ids = [int(n[1:]) - 1 for n in back["groups_ma"][f"PHASE_{p}"]]
            np.testing.assert_array_equal(ids, np.flatnonzero(m.phase == p))
        tri_kind = "TRIA6" if m.order == 2 else "TRIA3"
        tri_names, tri = back["elements"][tri_kind]
        number = {n: k for k, n in enumerate(tri_names)}
        for axis, face in enumerate(("X", "Y", "Z")):
            for side, end in ((0, "MIN"), (1, "MAX")):
                expected = m.boundary_faces(axis, side)
                got = tri[[number[n] for n in back["groups_ma"][face + end]]]
                np.testing.assert_array_equal(got, expected)
                np.testing.assert_array_equal(
                    back["groups_no"][f"N_{face}{end}"], m.face_nodes(axis, side)
                )
        # the parser of the stand-in, independent: the same counts, words of 8 characters
        parsed = fake.read_mail(path)
        assert len(parsed["cells"]) == m.n_elements + len(tri)
        assert sorted(parsed["groups_no"]) == sorted(back["groups_no"])


def test_periodic_classes(mesh):
    groups = aster_groups(mesh, periodic=True)
    # seven classes: three faces, three edges, a corner
    assert sorted(groups.links) == ["001", "010", "011", "100", "101", "110", "111"]
    slaves = np.concatenate([groups.nodes[f"PS_{c}"] for c in groups.links])
    masters = np.concatenate([groups.nodes[f"PM_{c}"] for c in groups.links])
    upper = np.unique(np.concatenate([mesh.face_nodes(a, 1) for a in range(3)]))
    np.testing.assert_array_equal(np.sort(slaves), upper)  # each node of an upper face once
    assert not np.isin(masters, slaves).any()  # images tied to no other node
    for cls, (shift, s, m) in groups.links.items():
        np.testing.assert_allclose(
            mesh.nodes[s] - mesh.nodes[m], np.tile(shift * mesh.size, (len(s), 1)), atol=1e-12
        )
        assert len(cls) == 3
    assert groups.origin == int(np.argmin(np.linalg.norm(mesh.nodes - mesh.lower, axis=1)))
    assert groups.nodes["ORIGIN"].tolist() == [groups.origin]


def test_periodic_and_kinematic_studies_against_morphogen(tmp_path, mesh):
    """The six unit strains and three unit gradients of the command file, run by the
    stand-in: the apparent stiffness and conductivity of morphoGen's finite elements on
    the same mesh (periodic and kinematic conditions)."""
    for conditions, bc in (("periodic", "periodic"), ("kinematic", "kubc")):
        study = write_code_aster(tmp_path / conditions, mesh, MATERIALS, conditions=conditions)
        assert sorted(study.cases) == sorted([*STRAIN_CASES, "G1", "G2", "G3"])
        run = fake.run(study.directory / "aster.export")
        result = read_code_aster(study.directory)
        c_ref, _ = stiffness_tensor(make_problem(mesh, MATERIALS, "elastic", bc))
        k_ref, _ = conductivity_tensor(make_problem(mesh, MATERIALS, "thermal", bc))
        assert _relative(result.stiffness, c_ref) < 2e-6
        assert _relative(result.conductivity, k_ref) < 2e-6
        assert result.properties["bulk_modulus"] == pytest.approx(c_ref[:3, :3].sum() / 9, 1e-5)
        names = [n for n, _ in run.calls]
        assert names[0] == "DEBUT" and names[-1] == "FIN"
        assert names.count("MECA_STATIQUE") == 6 and names.count("THER_LINEAIRE") == 3
        if conditions == "periodic":
            loads = [c for n, c in run.calls if n == "AFFE_CHAR_MECA"]
            assert len(loads) == 6 and all(len(c["LIAISON_GROUP"]) == 21 for c in loads)
            # the jumps of the unit shear E12: (E s L)_i, E12 = E21 = 1/2
            jumps = {
                (g["GROUP_NO_1"], g["DDL_1"]): g["COEF_IMPO"] for g in loads[5]["LIAISON_GROUP"]
            }
            assert jumps[("PS_010", "DX")] == pytest.approx(0.5 * 1.2)
            assert jumps[("PS_100", "DY")] == pytest.approx(0.5 * 1.0)
            assert jumps[("PS_001", "DX")] == 0.0
        else:
            assert names.count("FORMULE") == 18 + 3


def test_crystals_turned_by_their_grains(tmp_path):
    """Cubic crystals (ELAS_ORTH) and orthotropic conductors (THER_ORTH) turned by the
    Bunge angles of their grains (MASSIF, ANGL_EULER): Code_Aster's frame of these angles
    is the frame of the crystal, the stiffness and conductivity those of morphoGen."""
    base = box_mesh((3, 3, 3), upper=(1.0, 1.2, 0.9), perturb=0.2)
    centres = base.nodes[base.elements].mean(axis=1)
    cell = (centres[:, 0] > 0.5).astype(int) + 2 * (centres[:, 1] > 0.6).astype(int)
    orientations = np.array(
        [[10.0, 20.0, 30.0], [45.0, 60.0, 15.0], [100.0, 35.0, 200.0], [0.0, 0.0, 0.0]]
    )
    mesh = TetMesh(
        base.nodes,
        base.elements,
        np.where(cell == 3, 1, 0),
        base.lower,
        base.upper,
        periodic=True,
        cell=cell,
        orientations=orientations,
    )
    cubic = Material.cubic(108.0, 62.0, 28.0)
    crystal = Material(
        stiffness=cubic.stiffness,
        symmetry=cubic.symmetry,
        conductivities=((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 4.0)),
    )
    materials = {0: crystal, 1: MATERIALS[1]}
    study = write_code_aster(tmp_path, mesh, materials)
    run = fake.run(study.directory / "aster.export")
    cara = next(c for n, c in run.calls if n == "AFFE_CARA_ELEM")
    assert [m["GROUP_MA"] for m in cara["MASSIF"]] == ["GRAIN_0", "GRAIN_1", "GRAIN_2"]
    for c, item in enumerate(cara["MASSIF"]):  # the local frame: the axes of the crystal
        np.testing.assert_allclose(
            fake.euler_frame(item["ANGL_EULER"]), euler_to_matrix(orientations[c])[0].T, atol=1e-12
        )
    result = read_code_aster(study.directory)
    c_ref, _ = stiffness_tensor(make_problem(mesh, materials, "elastic", "periodic"))
    k_ref, _ = conductivity_tensor(make_problem(mesh, materials, "thermal", "periodic"))
    assert _relative(result.stiffness, c_ref) < 2e-6
    assert _relative(result.conductivity, k_ref) < 2e-6
    # without grains, the axes of the crystals are those of the sample (ANGL_REP 0 0 0)
    plain = TetMesh(base.nodes, base.elements, mesh.phase, base.lower, base.upper, periodic=True)
    study = write_code_aster(tmp_path / "plain", plain, materials, physics="elastic")
    run = fake.run(study.directory / "aster.export")
    cara = next(c for n, c in run.calls if n == "AFFE_CARA_ELEM")
    assert cara["MASSIF"]["ANGL_REP"] == (0.0, 0.0, 0.0)
    c_ref, _ = stiffness_tensor(make_problem(plain, materials, "elastic", "periodic"))
    assert _relative(read_code_aster(study.directory).stiffness, c_ref) < 2e-6
    assert read_code_aster(study.directory).conductivity is None


def test_voids_and_whole_periodic_cells(tmp_path):
    porous = box_mesh(
        (4, 4, 4),
        phase=lambda c: (np.linalg.norm(c - 0.5, axis=1) < 0.3).astype(int),
        upper=(1.0, 1.0, 1.0),
        perturb=0.1,
    )
    materials = {0: Material(young=10.0, poisson=0.3), 1: Material(void=True)}
    study = write_code_aster(tmp_path / "void", porous, materials)
    mail = read_aster_mesh(study.directory / "aster.mail")
    assert "PHASE_1" not in mail["groups_ma"]
    assert len(mail["elements"]["TETRA4"][0]) == int((porous.phase == 0).sum())
    fake.run(study.directory / "aster.export")
    c_ref, _ = stiffness_tensor(make_problem(porous, materials, "elastic", "periodic"))
    assert _relative(read_code_aster(study.directory).stiffness, c_ref) < 2e-6
    # whole cells: translations of -1, 0 or 1 period along each axis
    import morphogen as mg
    from morphogen.mesh.conforming import conforming_mesh

    micro = mg.models.Voronoi(n_cells=6).generate(mg.Domain.cube(1.0, periodic=True), seed=3)
    cells = conforming_mesh(micro, size=0.3, cells="whole")
    assert cells.whole
    crystal = {p: Material.cubic(108.0, 62.0, 28.0) for p in np.unique(cells.phase)}
    study = write_code_aster(tmp_path / "whole", cells, crystal, physics="elastic")
    groups = read_aster_mesh(study.directory / "aster.mail")["groups_no"]
    assert any("M" in g[3:] for g in groups if g.startswith("PS_"))
    fake.run(study.directory / "aster.export")
    c_ref, _ = stiffness_tensor(make_problem(cells, crystal, "elastic", "periodic"))
    assert _relative(read_code_aster(study.directory).stiffness, c_ref) < 2e-6


def test_files_of_a_study(tmp_path, mesh):
    study = write_code_aster(tmp_path, mesh.quadratic(), MATERIALS, name="cell", fields=True)
    export = (tmp_path / "cell.export").read_text().splitlines()
    assert "F comm cell.comm D 1" in export and "F mail cell.mail D 20" in export
    assert "F rmed cell_fields.rmed R 80" in export
    units = {int(line.split()[4]): line.split()[2] for line in export if line.startswith("F")}
    for case, spec in study.cases.items():
        assert units[spec["unit"]] == f"cell_{case}.table"
    comm = (tmp_path / "cell.comm").read_text()
    compile(comm, "cell.comm", "exec")  # Python, as Code_Aster reads it
    assert 'LIRE_MAILLAGE(\n    FORMAT="ASTER",\n    UNITE=20,' in comm
    assert "IMPR_RESU(" in comm and 'TYPE_CALCUL="STAT"' in comm
    manifest = json.loads((tmp_path / "cell_aster.json").read_text())
    assert manifest["conditions"] == "periodic" and manifest["order"] == 2
    assert manifest["volume"] == pytest.approx(1.08)
    # conditions by default: kinematic for a mesh that is not periodic
    open_mesh = box_mesh((2, 2, 2), periodic=False)
    study = write_code_aster(tmp_path / "open", open_mesh, MATERIALS, physics="thermal")
    assert json.loads((tmp_path / "open" / "aster_aster.json").read_text())["conditions"] == (
        "kinematic"
    )
    assert sorted(study.cases) == ["G1", "G2", "G3"]


def test_errors(tmp_path, mesh):
    with pytest.raises(ValueError, match="no material for phase"):
        write_code_aster(tmp_path, mesh, {0: MATERIALS[0]})
    triclinic = np.diag([100.0, 90.0, 80.0, 30.0, 31.0, 32.0])
    triclinic[0, 3] = triclinic[3, 0] = 5.0
    with pytest.raises(ValueError, match="orthotropic"):
        write_code_aster(
            tmp_path, mesh, {0: Material(stiffness=triclinic), 1: MATERIALS[1]}, physics="elastic"
        )
    with pytest.raises(ValueError, match="name"):
        write_code_aster(tmp_path, mesh, MATERIALS, name="a b")
    with pytest.raises(ValueError, match="conditions"):
        write_code_aster(tmp_path, mesh, MATERIALS, conditions="mixed")
    with pytest.raises(ValueError, match="no conductivity"):
        write_code_aster(
            tmp_path,
            mesh,
            {0: Material(young=1.0, poisson=0.3), 1: MATERIALS[1]},
            physics="thermal",
        )


def test_tables(tmp_path):
    """The format TABLEAU of IMPR_TABLE: comments, a line of dashes, the names, the rows."""
    path = tmp_path / "t.table"
    path.write_text(
        "#\n#ASTER 17.4.0 CONCEPT INTE_E11 CALCULE LE 27/11/2026\n#\n" + "-" * 70 + "\n"
        "NOM_CHAM         NUME_ORDRE INST         VOL          GROUP_MA         "
        "INTE_SIXX    INTE_SIYY    MOYE_SIXX    MOYE_SIYY\n"
        "SIEF_ELGA                 1  0.00000E+00  6.00000E-01 PHASE_0           "
        "1.20000E+00  2.40000E-01  2.00000E+00  4.00000E-01\n"
        "SIEF_ELGA                 1  0.00000E+00  4.00000E-01 PHASE_1           "
        "3.00000E+00 -1.00000E-01  7.50000E+00 -2.50000E-01\n"
    )
    table = read_aster_table(path)
    assert table["GROUP_MA"] == ["PHASE_0", "PHASE_1"]
    assert sum(table["INTE_SIXX"]) == pytest.approx(4.2)
    assert table["VOL"] == [0.6, 0.4]
    path.write_text("# nothing\n")
    with pytest.raises(ValueError, match="no table"):
        read_aster_table(path)


def test_mail_through_save_mesh(tmp_path, mesh):
    path = io.save_mesh(mesh, tmp_path / "cell.mail")
    back = read_aster_mesh(path)
    assert "BOUNDARY" in back["groups_no"] and "ORIGIN" in back["groups_no"]
    assert not any(g.startswith("PS_") for g in back["groups_no"])


@pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="MED files need h5py (meshio's writer)"
)
def test_med_mesh(tmp_path, mesh):
    import meshio

    study = write_code_aster(tmp_path, mesh, MATERIALS, mesh_format="med")
    assert "F mmed aster.med D 20" in (tmp_path / "aster.export").read_text()
    assert 'NOM_MED="mesh"' in (tmp_path / "aster.comm").read_text()
    back = meshio.read(study.directory / "aster.med")
    np.testing.assert_allclose(back.points, mesh.nodes)
    # MED's order of the corners (the second and the third swapped): positive volumes there
    tets = back.cells_dict["tetra"]
    np.testing.assert_array_equal(tets, mesh.elements[:, [0, 2, 1, 3]])
    x = back.points[tets]
    a, b, c = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]
    assert np.all(np.einsum("ij,ij->i", a, np.cross(b, c)) < 0)  # VTK's volume, mirrored
    groups = aster_groups(mesh, periodic=True)
    tags = back.cell_data["cell_tags"][0]
    for p in (0, 1):  # the elements of each phase: the families holding its group
        fam = [f for f, names in back.cell_tags.items() if f"PHASE_{p}" in names]
        np.testing.assert_array_equal(
            np.flatnonzero(np.isin(tags, fam)), np.flatnonzero(mesh.phase == p)
        )
    node_tags = back.point_data["point_tags"]
    for name, ids in groups.nodes.items():
        fam = [f for f, names in back.point_tags.items() if name in names]
        np.testing.assert_array_equal(np.flatnonzero(np.isin(node_tags, fam)), np.sort(ids))


def test_runs_and_commands(tmp_path, capsys):
    """outputs.code_aster in a run (the conforming mesh of each realization), its errors;
    the commands morphogen code-aster export and read."""
    cfg = {
        "model": {"type": "voronoi", "n_cells": 6},
        "domain": {"size": 1.0, "periodic": True},
        "seed": 1,
        "mesh": {"size": 0.3, "order": 1},
        "outputs": {"directory": str(tmp_path / "out"), "name": "poly",
                    "code_aster": {"physics": "elastic"}},
        "homogenization": {"method": "fe", "boundary_conditions": ["periodic"],
                           "materials": {1: {"young": 70.0, "poisson": 0.22}},
                           "properties": ["bulk_modulus"]},
    }  # fmt: skip
    outdir = pipeline.run(load_config(cfg))
    study = outdir / "poly_0000_aster"
    assert (study / "poly_0000.comm").exists() and (study / "poly_0000.mail").exists()
    manifest = json.loads((study / "poly_0000_aster.json").read_text())
    assert manifest["conditions"] == "periodic"
    assert sorted(manifest["cases"]) == sorted(STRAIN_CASES)
    fake.run(study / "poly_0000.export")
    run = read_code_aster(study)  # the homogeneous material of the grains
    assert run.properties["bulk_modulus"] == pytest.approx(70.0 / (3 * (1 - 2 * 0.22)), 1e-6)
    for change, match in [
        ({"outputs": {**cfg["outputs"], "code_aster": {"bad": 1}}}, "bad"),
        ({"outputs": {**cfg["outputs"], "code_aster": {"conditions": "mixed"}}}, "conditions"),
        ({"mesh": None}, "mesh"),
    ]:
        data = {**cfg, **change}
        if data.get("mesh") is None:
            data.pop("mesh")
            data["homogenization"] = {**cfg["homogenization"], "method": "fft"}
            data["voxels"] = {"shape": 8}
        with pytest.raises(ConfigError, match=match):
            load_config(data)
    # the commands: a mesh of the box (.vtu) exported, run, read back
    mesh = box_mesh((3, 3, 3), upper=(1.0, 1.2, 0.9), phase=_inclusion, perturb=0.2)
    io.save_mesh(mesh, tmp_path / "cell.vtu")
    materials = [
        "0=young:3,poisson:0.35,conductivity:0.2",
        "1=young:70,poisson:0.22,conductivity:1",
    ]
    assert cli.main(["code-aster", "export", str(tmp_path / "cell.vtu"), "--material",
                     *materials, "--periodic", "-o", str(tmp_path / "cmd")]) == 0  # fmt: skip
    assert "run_aster aster.export" in capsys.readouterr().out
    table = ["-o", str(tmp_path / "t.csv")]
    assert cli.main(["code-aster", "read", str(tmp_path / "cmd"), *table]) != 0
    assert "needs their tables" in capsys.readouterr().err
    fake.run(tmp_path / "cmd" / "aster.export")
    assert cli.main(["code-aster", "read", str(tmp_path / "cmd"), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    c_ref, _ = stiffness_tensor(make_problem(mesh, MATERIALS, "elastic", "periodic"))
    assert _relative(out["stiffness"], c_ref) < 2e-6 and len(out["conductivity"]) == 3
    assert cli.main(["code-aster", "read", str(tmp_path / "cmd"), *table]) == 0
    assert "C11" in (tmp_path / "t.csv").read_text().splitlines()[0]
