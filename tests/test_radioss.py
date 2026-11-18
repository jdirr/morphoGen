"""The inputs of OpenRadioss (morphogen.io.radioss) read back card by card against the
fields of the Radioss Reference Guide (integers of 10 characters, reals of 20, the
columns of each keyword), the orientation of the elements, the materials, loadings,
contacts and time histories; the model of morphogen.dynamics (units, materials,
loadings, time steps); the readers of the listings, time histories and legacy VTK
files of unstructured grids."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from morphogen import io
from morphogen.domain import Domain
from morphogen.dynamics import (
    ExplicitLoading,
    ExplicitMaterial,
    JohnsonCook,
    UnitSystem,
    critical_time_step,
    explicit_materials,
)
from morphogen.dynamics.model import CURVE_STRAINS, NORTON_RATES, explicit_mesh
from morphogen.image import VoxelImage
from morphogen.io import radioss
from morphogen.io.vtk import cells_of_type, read_vtk_grid
from morphogen.mesh.tetra import TetMesh
from morphogen.mesh.voxel import voxel_mesh
from morphogen.models.base import model_from_config

STEEL = {"young": 210e3, "poisson": 0.3, "density": 7.85e-9}
ALUMINIUM = {
    "young": 70e3,
    "poisson": 0.33,
    "density": 2.7e-9,
    "plasticity": {"yield": 200.0, "hardening": 500.0},
}


def _cards(path) -> list[tuple[str, list[str]]]:
    """The keywords of an input with their data lines (comments left out)."""
    blocks: list[tuple[str, list[str]]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("/"):
            blocks.append((line, []))
        elif blocks:
            blocks[-1][1].append(line)
    return blocks


def _get(blocks, prefix: str) -> list[tuple[str, list[str]]]:
    return [(k, d) for k, d in blocks if k.startswith(prefix)]


def _ints(line: str) -> list[int]:
    """The integer fields of 10 characters of a line."""
    return [int(line[k : k + 10]) for k in range(0, len(line), 10) if line[k : k + 10].strip()]


def _reals(line: str) -> list[float]:
    """The real fields of 20 characters of a line."""
    return [float(line[k : k + 20]) for k in range(0, len(line), 20) if line[k : k + 20].strip()]


def _cube(n=4, phases=None):
    """The voxel mesh of a cube of ``n`` voxels per edge (edge 1), all solid or with the
    phase image given."""
    data = np.ones((n, n, n), dtype=np.uint8) if phases is None else phases
    image = VoxelImage(data=data, spacing=(1.0 / n,) * 3, origin=(0.0, 0.0, 0.0))
    return voxel_mesh(image, sorted({int(q) for q in np.unique(data)} - {0}) or None)


def _materials(**overrides):
    spec = {1: dict(STEEL)}
    spec.update(overrides)
    return {q: ExplicitMaterial.from_config(v) for q, v in spec.items()}


# -- the model -------------------------------------------------------------------------------


def test_units():
    u = UnitSystem.parse("Mg mm s")
    assert u.unit("stress") == "MPa" and u.unit("force") == "N" and u.unit("energy") == "mJ"
    assert u.from_si(7850.0, "density") == pytest.approx(7.85e-9)
    assert u.from_si(10.0, "velocity") == pytest.approx(1e4)
    assert u.to_si(1.0, "stress") == pytest.approx(1e6)
    v = UnitSystem("kg", "mm", "ms")
    assert v.unit("stress") == "GPa" and v.unit("velocity") == "m/s" and v.unit("force") == "kN"
    assert UnitSystem.parse(("g", "mm", "ms")).unit("stress") == "MPa"
    assert np.allclose(u.from_si([1.0, 2.0], "length"), [1e3, 2e3])
    with pytest.raises(ValueError, match="mass"):
        UnitSystem("tonne", "mm", "s")
    with pytest.raises(ValueError, match="mass, length and time"):
        UnitSystem.parse("Mg mm")


def test_materials():
    elastic = ExplicitMaterial.from_config(STEEL)
    assert elastic.law == "LAW1"
    modulus = 210e3 * 0.7 / (1.3 * 0.4)
    assert elastic.wave_speed == pytest.approx(np.sqrt(modulus / 7.85e-9))
    plastic = ExplicitMaterial.from_config({**ALUMINIUM, "failure_strain": 0.3})
    assert plastic.law == "LAW36" and plastic.failure_strain == 0.3 and plastic.rates() == (0.0,)
    assert np.allclose(plastic.yield_curve([0.0, 0.1]), [200.0, 250.0])
    voce = ExplicitMaterial.from_config(
        {
            **STEEL,
            "plasticity": {
                "yield": 100,
                "isotropic": [{"Q": 50, "b": 10}],
                "norton": {"K": 20, "n": 4},
            },
        }
    )
    p = np.array([0.0, 0.2])
    assert voce.rates() == NORTON_RATES
    assert np.allclose(voce.yield_curve(p), 100 + 50 * (1 - np.exp(-10 * p)))
    assert np.allclose(voce.yield_curve(p, 16.0), 100 + 50 * (1 - np.exp(-10 * p)) + 20 * 2.0)
    jc = ExplicitMaterial.from_config(
        {**STEEL, "johnson_cook": {"A": 300, "B": 450, "n": 0.6, "C": 0.02, "rate": 1.0}}
    )
    assert jc.law == "LAW2"
    assert jc.yield_curve([0.0], np.e)[0] == pytest.approx(300 * 1.02)
    assert ExplicitMaterial.from_config("void") is None
    spec = explicit_materials({"0": "void", "1": {"young": 70e3, "poisson": 0.3}}, density=2.7e-9)
    assert spec[0] is None and spec[1].density == 2.7e-9
    with pytest.raises(ValueError, match="density"):
        ExplicitMaterial.from_config({"young": 70e3, "poisson": 0.3})
    with pytest.raises(ValueError, match="plastic law"):
        ExplicitMaterial.from_config({**STEEL, "failure_strain": 0.2})
    with pytest.raises(ValueError, match="kinematic"):
        ExplicitMaterial.from_config(
            {**STEEL, "plasticity": {"yield": 1, "kinematic": [{"C": 1000, "D": 10}]}}
        )
    with pytest.raises(ValueError, match="not both"):
        ExplicitMaterial(210e3, 0.3, 7.85e-9, plasticity={"yield": 1}, johnson_cook={"a": 1})
    with pytest.raises(ValueError, match="crystal_plasticity"):
        ExplicitMaterial.from_config(
            {
                "young": 1,
                "poisson": 0.3,
                "density": 1,
                "crystal_plasticity": {"tau0": 40, "K": 20, "n": 12},
            }
        )
    with pytest.raises(ValueError, match="johnson_cook"):
        JohnsonCook.from_config({"b": 1})
    assert ExplicitMaterial.from_config(jc.to_config()) == jc


def test_loadings():
    lengths = np.array([2.0, 2.0, 4.0])
    c = ExplicitLoading(kind="compression", strain=0.3, strain_rate=100.0, ramp=0.1)
    end = c.end_time(lengths)
    assert end == pytest.approx(0.3 / (100.0 * 0.95))
    assert c.speed(lengths) == pytest.approx(400.0) and c.sign() == -1
    assert c.displacement([end], lengths)[0] == pytest.approx(-0.3 * 4.0)
    table = c.velocity_function(end)
    assert table[0].tolist() == [0.0, 0.0] and table[-1, 1] == 1.0 and table[-1, 0] > end
    assert np.all(np.diff(table[:, 1]) >= 0)
    s = ExplicitLoading.from_config({"kind": "shear", "axis": "z"})
    assert s.shear_axis == 0 and s.driven == 0 and s.sign() == 1
    i = ExplicitLoading(kind="impact", velocity=5e3, strain=0.2)
    assert i.end_time(lengths) == pytest.approx(0.2 * 4.0 / 5e3)
    assert ExplicitLoading.from_config(c.to_config()) == c
    for bad, match in (
        ({"kind": "bend"}, "loading"),
        ({"kind": "compression", "strain": 1.2}, "below 1"),
        ({"kind": "impact"}, "velocity"),
        ({"kind": "shear", "axis": "x", "shear_axis": "x"}, "another axis"),
        ({"axis": "w"}, "axis"),
        ({"lateral": "glued"}, "lateral"),
        ({"speed": 3}, "unknown keys"),
    ):
        with pytest.raises(ValueError, match=match):
            ExplicitLoading.from_config(bad)


def test_time_step():
    mesh = _cube(4)
    material = ExplicitMaterial.from_config(STEEL)
    dt = critical_time_step(mesh, {1: material})
    assert dt == pytest.approx(0.9 * 0.25 / material.wave_speed)
    tets = TetMesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]),
        np.array([[0, 1, 2, 3]]),
        np.array([1]),
        [0, 0, 0],
        [1, 1, 1],
    )
    altitude = 3 * (1 / 6) / (0.5 * np.sqrt(3))  # the corner opposite the slanted face
    assert critical_time_step(tets, {1: material}, scale=1.0) == pytest.approx(
        altitude / material.wave_speed
    )


# -- the inputs ------------------------------------------------------------------------------


def test_cards_of_a_compression(tmp_path):
    mesh = _cube(4)
    loading = ExplicitLoading(kind="compression", strain=0.2, strain_rate=1000.0)
    inputs = io.write_radioss(
        tmp_path, mesh, _materials(), loading, name="cube", frames=8, history=100, title="a cube"
    )
    text = inputs.starter.read_text(encoding="ascii")
    assert text.startswith("#RADIOSS STARTER\n") and text.rstrip().endswith("/END")
    assert max(len(line) for line in text.splitlines()) <= 100
    blocks = _cards(inputs.starter)
    ((_, begin),) = _get(blocks, "/BEGIN")
    assert begin[0] == "cube" and _ints(begin[1])[0] == 2022
    assert begin[2].split() == ["Mg", "mm", "s"] == begin[3].split()
    assert len(begin[2]) == 60 and begin[2][:20].strip() == "Mg"
    ((_, nodes),) = _get(blocks, "/NODE")
    assert len(nodes) == 125
    xyz = np.array([[float(line[10 + 20 * k : 30 + 20 * k]) for k in range(3)] for line in nodes])
    assert np.allclose(xyz, mesh.nodes) and [int(line[:10]) for line in nodes] == list(
        range(1, 126)
    )
    ((_, bricks),) = _get(blocks, "/BRICK/1")
    connectivity = np.array([_ints(line) for line in bricks])
    assert connectivity.shape == (64, 9) and connectivity[:, 0].tolist() == list(range(1, 65))
    corners = xyz[connectivity[:, 1:] - 1]  # positive: (x4 - x1, x5 - x1, x2 - x1) right-handed
    det = np.einsum(
        "ij,ij->i",
        corners[:, 3] - corners[:, 0],
        np.cross(corners[:, 4] - corners[:, 0], corners[:, 1] - corners[:, 0]),
    )
    assert np.all(det > 0)
    ((_, part),) = _get(blocks, "/PART/1")
    assert part[0] == "phase 1" and _ints(part[1]) == [1, 1, 0]
    ((_, prop),) = _get(blocks, "/PROP/SOLID/1")
    first = prop[1]
    assert int(first[:10]) == 24 and int(first[10:20]) == -1 and first[20:30] == " " * 10
    assert int(first[30:40]) == -1 and int(first[60:70]) == 3
    ((_, mat),) = _get(blocks, "/MAT/LAW1/1")
    assert _reals(mat[1]) == [7.85e-9] and _reals(mat[2]) == [210e3, 0.3]
    groups = {d[0]: d[1:] for k, d in _get(blocks, "/GRNOD/NODE/")}
    lower = [n for line in groups["lower face"] for n in _ints(line)]
    upper = [n for line in groups["upper face"] for n in _ints(line)]
    assert (
        sorted(lower) == sorted((radioss.face_nodes(mesh, 2, 0) + 1).tolist()) and len(lower) == 25
    )
    assert np.allclose(xyz[np.array(upper) - 1, 2], 1.0)
    bcs = {d[0]: d[1] for k, d in _get(blocks, "/BCS/")}
    assert bcs["lower face held"][:10] == "   111 000" and int(bcs["lower face held"][20:30]) == 1
    assert bcs["upper face held across the drive"][:10] == "   110 000"
    ((_, impvel),) = _get(blocks, "/IMPVEL/1")
    head = impvel[1]
    funct, direction, group = int(head[:10]), head[10:20].strip(), int(head[40:50])
    assert direction == "Z" and group == 2
    scale = _reals(impvel[2])
    assert scale[0] == 1.0 and scale[1] == pytest.approx(-1000.0) and scale[3] > inputs.end_time
    ((_, function),) = _get(blocks, f"/FUNCT/{funct}")
    table = np.array([_reals(line) for line in function[1:]])
    assert table[0].tolist() == [0.0, 0.0] and table[-1, 1] == 1.0
    ((_, surf),) = _get(blocks, "/SURF/PART/EXT/1")
    assert _ints(surf[1]) == [1]
    ((_, contact),) = _get(blocks, "/INTER/TYPE24/1")
    assert int(contact[1][:10]) == 1 and int(contact[1][10:20]) == 0
    assert float(contact[4][20:40]) == 0.0 and float(contact[4][80:100]) > inputs.end_time
    histories = _get(blocks, "/TH/NODE/")
    assert [d[0] for _, d in histories] == [
        "reactions of the lower face",
        "reactions of the upper face",
    ]
    lines = histories[0][1]
    assert lines[1].split() == ["REACZ"] and lines[2][20:].startswith("L")
    assert len(lines) - 2 == 25 and int(lines[2][:10]) == int(lines[2][21:])
    ((_, parts),) = _get(blocks, "/TH/PART/")
    assert parts[1].strip() == "DEF" and _ints(parts[2]) == [1]
    engine = _cards(inputs.engine)
    run = dict(engine)
    assert _reals(run["/RUN/cube/1"][0])[0] == pytest.approx(inputs.end_time)
    assert _reals(run["/TFILE"][0])[0] == pytest.approx(inputs.end_time / 100)
    assert _reals(run["/ANIM/DT"][0]) == pytest.approx([0.0, inputs.end_time / 8])
    for name in ("/ANIM/VECT/DISP", "/ANIM/VECT/FREAC", "/ANIM/ELEM/VONM", "/ANIM/ELEM/OFF"):
        assert name in run
    assert not any(k.startswith("/DT/NODA") for k, _ in engine)
    m = json.loads((tmp_path / radioss.MANIFEST).read_text())
    assert m["n_elements"] == 64 and m["area"] == 1.0 and m["mass"] == pytest.approx(7.85e-9)
    assert m["faces"] == {"lower": 25, "upper": 25} and m["laws"] == {"1": "LAW1"}


def test_free_faces_shear_impact_and_scaling(tmp_path):
    mesh = _cube(3)
    free = io.write_radioss(
        tmp_path / "free",
        mesh,
        _materials(),
        {"kind": "tension", "lateral": "free", "contact": False, "strain": 0.01},
        time_step=1e-7,
    )
    blocks = _cards(free.starter)
    bcs = {d[0]: d[1] for _, d in _get(blocks, "/BCS/")}
    assert bcs["lower face held"][:10] == "   001 000"
    assert bcs["a node of the lower face held"][:10] == "   111 000"
    assert "upper face held across the drive" not in bcs
    groups = {
        d[0]: [n for line in d[1:] for n in _ints(line)] for _, d in _get(blocks, "/GRNOD/NODE/")
    }
    assert len(groups["anchor"]) == 1 and len(groups["lower face"]) == 15
    assert not _get(blocks, "/INTER/") and not _get(blocks, "/SURF/")
    ((_, impvel),) = _get(blocks, "/IMPVEL/1")
    assert _reals(impvel[2])[1] > 0  # pulled
    engine = dict(_cards(free.engine))
    assert _reals(engine["/DT/NODA/CST/0"][0]) == [0.9, 1e-7]
    shear = io.write_radioss(
        tmp_path / "shear", mesh, _materials(), {"kind": "shear", "axis": "y", "shear_axis": "z"}
    )
    blocks = _cards(shear.starter)
    bcs = {d[0]: d[1] for _, d in _get(blocks, "/BCS/")}
    assert bcs["lower face held"][:10] == "   111 000"
    assert bcs["upper face held across the drive"][:10] == "   110 000"
    assert _get(blocks, "/IMPVEL/1")[0][1][1][10:20].strip() == "Z"
    reactions = _get(blocks, "/TH/NODE/")[0][1][1].split()
    assert reactions == ["REACY", "REACZ"]
    impact = io.write_radioss(
        tmp_path / "impact",
        mesh,
        _materials(),
        {"kind": "impact", "velocity": 2e4, "friction": 0.2},
        length_scale=10.0,
    )
    blocks = _cards(impact.starter)
    ((_, inivel),) = _get(blocks, "/INIVEL/TRA/1")
    assert _reals(inivel[1][:60]) == [0.0, 0.0, -2e4] and int(inivel[1][60:70]) == 1
    ((_, group),) = _get(blocks, "/GRNOD/PART/1")
    assert _ints(group[1]) == [1]
    ((_, wall),) = _get(blocks, "/RWALL/PLANE/1")
    assert _ints(wall[1]) == [0, 2, 0, 0]  # sliding with friction
    assert _reals(wall[2])[:2] == [15.0, 0.2]  # the search spans the morphology (edge 10)
    assert _reals(wall[3]) == [5.0, 5.0, 0.0] and _reals(wall[4]) == [5.0, 5.0, 1.0]
    assert not _get(blocks, "/IMPVEL") and not _get(blocks, "/BCS")
    assert [k for k, _ in _get(blocks, "/TH/")] == ["/TH/RWALL/1", "/TH/PART/2"]
    m = impact.manifest
    assert m["lengths"] == [10.0, 10.0, 10.0] and m["mass"] == pytest.approx(7.85e-9 * 1000)
    with pytest.raises(ValueError, match="length_scale"):
        io.write_radioss(tmp_path / "bad", mesh, _materials(), "compression", length_scale=0)


def test_tetrahedra_oriented_for_radioss(tmp_path):
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1.0]])
    corners = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])  # VTK's orientation, then Radioss's
    x = nodes[corners]
    vtk = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])
    corners[vtk < 0] = corners[vtk < 0][:, [1, 0, 2, 3]]  # both positive for VTK
    mesh = TetMesh(nodes, corners, np.array([1, 2]), [0, 0, 0], [1, 1, 1])
    out = radioss.oriented(mesh)
    x = nodes[out]
    det = np.einsum("ij,ij->i", x[:, 0] - x[:, 3], np.cross(x[:, 1] - x[:, 3], x[:, 2] - x[:, 3]))
    assert np.all(det > 0) and np.array_equal(np.sort(out, axis=1), np.sort(corners, axis=1))
    # quadratic: the mid-side nodes follow their edges (5: 1-2, 6: 2-3, 7: 3-1, 8: 1-4...)
    edges = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    mids, extra, rows = {}, [], []
    for tet in corners:
        row = list(tet)
        for a, b in edges:
            key = tuple(sorted((tet[a], tet[b])))
            if key not in mids:
                mids[key] = len(nodes) + len(extra)
                extra.append(0.5 * (nodes[tet[a]] + nodes[tet[b]]))
            row.append(mids[key])
        rows.append(row)
    nodes10 = np.vstack([nodes, extra])
    mesh10 = TetMesh(nodes10, np.array(rows), np.array([1, 1]), [0, 0, 0], [1, 1, 1])
    out10 = radioss.oriented(mesh10)
    for tet in out10:
        for k, (a, b) in enumerate(edges):
            assert np.allclose(nodes10[tet[4 + k]], 0.5 * (nodes10[tet[a]] + nodes10[tet[b]]))
    x = nodes10[out10[:, :4]]
    det = np.einsum("ij,ij->i", x[:, 0] - x[:, 3], np.cross(x[:, 1] - x[:, 3], x[:, 2] - x[:, 3]))
    assert np.all(det > 0)
    inputs = io.write_radioss(
        tmp_path,
        mesh10,
        {1: ExplicitMaterial.from_config(STEEL)},
        {"kind": "impact", "velocity": 1e3},
    )
    blocks = _cards(inputs.starter)
    ((_, tetra),) = _get(blocks, "/TETRA10/1")
    assert len(tetra) == 4 and len(_ints(tetra[0])) == 1 and len(_ints(tetra[1])) == 10
    assert _ints(tetra[1]) == (out10[0] + 1).tolist()


def test_parts_of_phases_and_tabulated_laws(tmp_path):
    data = np.ones((4, 4, 4), dtype=np.uint8)
    data[:, :, :2] = 2
    mesh = _cube(4, data)
    norton = {
        **STEEL,
        "plasticity": {"yield": 100, "hardening": 50, "norton": {"K": 20, "n": 4}},
        "failure_strain": 0.4,
    }
    jc = {**STEEL, "johnson_cook": {"a": 300, "b": 450, "n": 0.6, "c": 0.02, "rate": 1.0}}
    materials = {1: ExplicitMaterial.from_config(norton), 2: ExplicitMaterial.from_config(jc)}
    inputs = io.write_radioss(
        tmp_path, mesh, materials, "compression", phase_names={1: "fibre", 2: "matrix"}
    )
    blocks = _cards(inputs.starter)
    assert [d[0] for _, d in _get(blocks, "/PART/")] == ["phase 1 (fibre)", "phase 2 (matrix)"]
    ids = [int(line[:10]) for k, d in _get(blocks, "/BRICK/") for line in d]
    assert sorted(ids) == list(range(1, 65))
    ((_, law36),) = _get(blocks, "/MAT/LAW36/1")
    assert _reals(law36[2]) == [210e3, 0.3, 0.4, 0.0, 0.0]
    line = law36[3]
    assert int(line[:10]) == len(NORTON_RATES) and int(line[90:100]) == 1  # VP: the plastic rate
    ids = [n for row in law36[5 : 5 + 3] for n in _ints(row)]
    assert len(ids) == len(NORTON_RATES)
    rates = [r for row in law36[-3:] for r in _reals(row)]
    assert rates == pytest.approx(list(NORTON_RATES))
    curve = np.array([_reals(r) for r in dict(blocks)[f"/FUNCT/{ids[3]}"][1:]])
    assert curve[:, 0] == pytest.approx(list(CURVE_STRAINS))
    expected = materials[1].yield_curve(curve[:, 0], NORTON_RATES[3])
    assert curve[:, 1] == pytest.approx(expected, rel=1e-10)
    ((_, law2),) = _get(blocks, "/MAT/LAW2/2")
    assert _reals(law2[2][:40]) == [210e3, 0.3] and int(law2[2][40:50]) == 0
    assert _reals(law2[3]) == [300, 450, 0.6, 0, 0]
    assert _reals(law2[4][:40]) == [0.02, 1.0]


def test_history_samples_of_large_faces(tmp_path, monkeypatch):
    monkeypatch.setattr(radioss, "MAX_HISTORY_VALUES", 50_000)
    inputs = io.write_radioss(tmp_path, _cube(8), _materials(), "compression", history=1000)
    assert inputs.manifest["history"] == 50_000 // (
        2 * 81
    )  # the reactions of two faces of 81 nodes
    engine = dict(_cards(inputs.engine))
    assert _reals(engine["/TFILE"][0])[0] == pytest.approx(inputs.end_time / (50_000 // 162))
    monkeypatch.setattr(radioss, "MAX_HISTORY_VALUES", 5_000)
    inputs = io.write_radioss(tmp_path, _cube(8), _materials(), "compression", history=1000)
    assert inputs.manifest["history"] == 100  # at least


def test_errors(tmp_path):
    data = np.zeros((3, 3, 3), dtype=np.uint8)
    data[1, 1, :] = 1  # a column along z, touching neither x face
    mesh = _cube(3, data)
    with pytest.raises(ValueError, match="no solid on the lower face normal to x"):
        io.write_radioss(tmp_path, mesh, _materials(), {"kind": "compression", "axis": "x"})
    with pytest.raises(ValueError, match=r"phases \[1\]"):
        io.write_radioss(tmp_path, mesh, {}, "compression")
    assert radioss.run_name("ab") == "ab_run" and radioss.run_name("my model #1") == "my_model_1"
    assert radioss._f(1.0 / 3.0).strip() == "0.333333333333" and len(radioss._f(-1.23e-123)) == 20
    with pytest.raises(ValueError, match="10 digits"):
        radioss._i(12345678901)


def test_mesh_of_a_morphology(tmp_path):
    micro = model_from_config(
        {"type": "tpms", "surface": "gyroid", "volume_fraction": 0.3, "cell_size": 5.0}
    ).generate(Domain(size=10.0), 1)
    spec = explicit_materials({0: "void", 1: ALUMINIUM})
    mesh = explicit_mesh(micro, spec, resolution=10)
    assert set(np.unique(mesh.phase)) == {1} and mesh.n_elements < 1000
    with pytest.raises(ValueError, match="no material"):
        explicit_mesh(micro, {1: spec[1]}, resolution=10)
    with pytest.raises(ValueError, match="voxels or conforming"):
        explicit_mesh(micro, spec, mesh="hexes")


# -- the outputs -----------------------------------------------------------------------------


LISTING = "\n".join(
    [
        "     CYCLE    TIME      TIME-STEP  ELEMENT          ERROR  I-ENERGY    K-ENERGY T"
        "  K-ENERGY R  EXT-WORK     MAS.ERR",
        "         0   0.000      0.1234E-06 SOLID         69   0.0%   0.000       0.000"
        "       0.000       0.000       0.000",
        "      1000  0.1234E-03  0.1234E-06 SOLID         69   0.1%  0.4567E+02  0.1E+01"
        "     0.000       0.4E+02     0.000",
        "      2000  0.2468D-03  0.1234E-06 SOLID         69   0.1%  0.4567E+02  0.1E+01"
        "     0.000       0.4E+02     0.000",
        " ** NORMAL TERMINATION",
    ]
)


def test_listings(tmp_path):
    path = tmp_path / "run_0001.out"
    path.write_text(LISTING)
    listing = io.read_radioss_listing(path)
    assert listing.cycles.tolist() == [0, 1000, 2000] and listing.normal is True
    assert listing.time == pytest.approx(2.468e-4) and listing.steps[0] == pytest.approx(1.234e-7)
    path.write_text(
        " ERROR ID :         55\n ** ERROR IN MATERIAL DEFINITION\n   MATERIAL 1\n"
        " ** ERROR TERMINATION\n"
    )
    listing = io.read_radioss_listing(path)
    assert listing.normal is False and "MATERIAL DEFINITION" in listing.errors[0]
    path.write_text("starting\n")
    assert io.read_radioss_listing(path).normal is None


def test_time_histories(tmp_path):
    path = tmp_path / "runT01.csv"
    path.write_text('"time","NODE L1 REACZ","NODE L1 REACZ","PART 1 IE"\n0,1,2,3\n0.5,4,5,6\n')
    th = io.read_radioss_th(path)
    assert list(th) == ["time", "NODE L1 REACZ", "NODE L1 REACZ#2", "PART 1 IE"]
    assert th["PART 1 IE"].tolist() == [3.0, 6.0]
    path.write_text("0 1 2\n1 3 4\n")
    th = io.read_radioss_th(path)
    assert th["time"].tolist() == [0.0, 1.0] and th["column 2"].tolist() == [2.0, 4.0]
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        io.read_radioss_th(path)


def _legacy(path, *, binary=False, modern=False):
    """A legacy VTK file of two cells (a tetrahedron and a hexahedron) with data."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.0]]
    )
    cells = [[0, 1, 2, 4], [0, 1, 2, 3, 4, 5, 6, 7]]
    types = [10, 12]
    out = [
        b"# vtk DataFile Version 3.0",
        b"frame TIME = 0.5",
        b"BINARY" if binary else b"ASCII",
        b"DATASET UNSTRUCTURED_GRID",
        b"POINTS 8 float",
    ]

    def block(values, fmt, ctype):
        if binary:
            return [struct.pack(f">{len(values)}{fmt}", *values)]
        return [" ".join(str(v) for v in values).encode()]

    out += block(points.ravel().tolist(), "f", "float")
    if modern:
        out += [b"CELLS 3 12", b"OFFSETS vtktypeint64", *block([0, 4, 12], "q", "q")]
        out += [b"CONNECTIVITY vtktypeint64", *block([*cells[0], *cells[1]], "q", "q")]
    else:
        flat = [4, *cells[0], 8, *cells[1]]
        out += [f"CELLS 2 {len(flat)}".encode(), *block(flat, "i", "int")]
    out += [b"CELL_TYPES 2", *block(types, "i", "int")]
    out += [b"CELL_DATA 2", b"SCALARS Von_Mises float 1", b"LOOKUP_TABLE default"]
    out += block([1.5, 2.5], "f", "float")
    out += [b"FIELD FieldData 1", b"ELEMENT_ID 1 2 int", *block([7, 9], "i", "int")]
    out += [b"POINT_DATA 8", b"VECTORS Displacement float"]
    out += block(np.arange(24, dtype=float).tolist(), "f", "float")
    path.write_bytes(b"\n".join(out) + b"\n")
    return points


@pytest.mark.parametrize(
    "binary,modern", [(False, False), (True, False), (False, True), (True, True)]
)
def test_legacy_unstructured_grids(tmp_path, binary, modern):
    path = tmp_path / "frame.vtk"
    points = _legacy(path, binary=binary, modern=modern)
    grid = read_vtk_grid(path)
    assert np.allclose(grid["points"], points) and grid["types"].tolist() == [10, 12]
    assert grid["offsets"].tolist() == [0, 4, 12]
    hexes, picked = cells_of_type(grid, 12)
    assert hexes.tolist() == [list(range(8))] and picked.tolist() == [1]
    tets, picked = cells_of_type(grid, 10)
    assert tets.tolist() == [[0, 1, 2, 4]]
    assert grid["cell_data"]["Von_Mises"].tolist() == [1.5, 2.5]
    assert grid["cell_data"]["ELEMENT_ID"].tolist() == [7, 9]
    assert grid["point_data"]["Displacement"].shape == (8, 3)
    assert cells_of_type(grid, 24)[0].shape == (0, 10)


def test_legacy_errors(tmp_path):
    path = tmp_path / "bad.vtk"
    path.write_text("not vtk\n")
    with pytest.raises(ValueError, match="legacy VTK"):
        read_vtk_grid(path)
    path.write_text("# vtk DataFile Version 3.0\nt\nASCII\nDATASET STRUCTURED_POINTS\n")
    with pytest.raises(ValueError, match="UNSTRUCTURED_GRID"):
        read_vtk_grid(path)
