"""Particles for the discrete element method: one phase without its matrix, as closed
surfaces whole across the periodic faces, or as the exact spheres of the structures made of
spheres and segments, in the files that the codes of discrete elements read."""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from morphogen.domain import Domain
from morphogen.io import export
from morphogen.io.dem import (
    particle_spheres,
    particle_table,
    particle_tetrahedra,
    particles,
    remesh_surfaces,
    sphere_surfaces,
    write_particles,
)
from morphogen.io.vtk import read_vtu
from morphogen.models.base import model_from_config
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments, Spheres

pytest.importorskip("skimage")

CENTRES = np.array([[1.0, 1.0, 1.0], [3.7, 2.0, 2.0]])
RADII = np.array([0.6, 0.8])


def _ball(r):
    return 4.0 / 3.0 * math.pi * r**3


def _two_spheres(periodic=True):
    domain = Domain(size=4.0, periodic=periodic)
    return PrimitiveUnion(domain, spheres=Spheres(CENTRES.copy(), RADII.copy()))


def test_whole_particles_across_the_periodic_faces():
    """A sphere that crosses a face of the periodic box is one particle, whole: its volume
    and centroid those of the sphere (beyond the face), within the discretization."""
    found = particles(_two_spheres(), shape=80)
    assert len(found) == 2 and all(p.whole for p in found)
    for p, c, r in zip(found, CENTRES, RADII, strict=True):
        assert p.volume == pytest.approx(_ball(r), rel=0.01)
        assert p.area == pytest.approx(4 * math.pi * r * r, rel=0.02)
        assert p.centroid == pytest.approx(c, abs=0.01)
        assert p.equivalent_radius == pytest.approx(r, rel=0.005)
    assert found[1].surface.vertices[:, 0].max() == pytest.approx(3.7 + 0.8, abs=0.03)


def test_particles_cut_by_a_window():
    """In a non-periodic window, the sphere cut by the face x = 4 is a cap closed by the face:
    the volume of the spherical cap, not whole."""
    found = particles(_two_spheres(periodic=False), shape=80)
    assert [p.whole for p in found] == [True, False]
    h = 4.0 - (3.7 - 0.8)
    assert found[1].volume == pytest.approx(math.pi * h * h * (3 * 0.8 - h) / 3, rel=0.01)
    assert found[1].surface.vertices[:, 0].max() == pytest.approx(4.0)


def test_clusters_that_wind_around_the_box():
    """A fibre through the periodic faces is no particle but an unbounded cluster, cut by the
    box; the matrix around spheres too; a voxel image gives the same particles."""
    domain = Domain(size=4.0, periodic=True)
    start = np.array([[2.0, 2.0, 0.0], [2.0, 2.0, 2.0]])
    end = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 4.0]])
    fibre = PrimitiveUnion(domain, segments=Segments(start, end, np.array([0.5, 0.5])))
    (found,) = particles(fibre, shape=48)
    assert not found.whole and found.volume == pytest.approx(math.pi * 0.25 * 4.0, rel=0.02)
    (matrix,) = particles(_two_spheres(), phase=0, shape=40)
    assert not matrix.whole
    assert matrix.volume == pytest.approx(64.0 - _ball(0.6) - _ball(0.8), rel=0.01)
    image = _two_spheres().voxelize(64)
    from_image = particles(image)
    assert len(from_image) == 2 and all(p.whole for p in from_image)
    assert from_image[1].volume == pytest.approx(_ball(0.8), rel=0.05)
    small = particles(image, min_voxels=5000)
    assert len(small) == 1  # the smaller sphere (about 3700 voxels) left out


def test_exact_spheres_and_clumps():
    """The spheres of a union of spheres and segments: each sphere once, its centre in the
    periodic box; a segment a clump of spheres along its axis, its caps at its ends."""
    domain = Domain(size=4.0, periodic=True)
    seg = Segments(np.array([[3.5, 1.0, 1.0]]), np.array([[5.5, 1.0, 1.0]]), np.array([0.5]))
    micro = PrimitiveUnion(domain, spheres=Spheres(np.array([[4.2, 1, 1.0]]), np.array([0.3])),
                           segments=seg)  # fmt: skip
    s = particle_spheres(micro)
    assert s["centre"][0] == pytest.approx([0.2, 1.0, 1.0]) and s["radius"][0] == 0.3
    beads = s["centre"][s["clump"] == 1]
    assert len(beads) == math.ceil(2.0 / (0.5 * 0.5)) + 1  # 0.5 radii apart at most
    # the middle of the axis (4.5) brought into the box: the ends 3.5 - 4 and 5.5 - 4
    assert beads[0] == pytest.approx([-0.5, 1, 1]) and beads[-1] == pytest.approx([1.5, 1, 1])
    assert np.all(s["radius"][1:] == 0.5)
    with pytest.raises(ValueError, match="union of spheres and segments"):
        particle_spheres(micro, phase=0)
    voronoi = model_from_config({"type": "voronoi", "n_cells": 4}).generate(domain, 1)
    with pytest.raises(ValueError, match="union of spheres"):
        particle_spheres(voronoi)
    surfaces = sphere_surfaces(s)
    assert len(surfaces) == 2 and surfaces[0].volume() == pytest.approx(_ball(0.3), rel=1e-12)


def test_files_of_the_particles(tmp_path):
    """The files: one STL solid, OBJ object or VTU number per particle; the exact spheres in
    tables, LAMMPS data files (box and atoms) and YADE columns; export() takes them."""
    micro = _two_spheres()
    stl = write_particles(tmp_path / "p.stl", micro, shape=48)
    text = stl.read_text()
    assert re.findall(r"^solid (\w+)", text, re.M) == ["particle_1", "particle_2"]
    assert text.count("endsolid") == 2
    obj = write_particles(tmp_path / "p.obj", micro, shape=48, exact=False)
    assert re.findall(r"^o (\w+)", obj.read_text(), re.M) == ["particle_1", "particle_2"]
    grid = read_vtu(write_particles(tmp_path / "p.vtu", micro, shape=48))
    assert set(np.unique(grid["cell_data"]["particle"])) == {1, 2}
    files = write_particles(tmp_path / "each.stl", micro, shape=48, separate=True)
    assert [f.name for f in files] == ["particle_00001.stl", "particle_00002.stl"]
    # the exact spheres
    lammps = write_particles(tmp_path / "p.data", micro).read_text()
    assert "2 atoms" in lammps and "0 4 xlo xhi" in lammps and "Atoms # sphere" in lammps
    rows = lammps.split("Atoms # sphere")[1].split()
    assert [float(v) for v in rows[:7]] == pytest.approx([1, 1, 1.2, 1.0, 1.0, 1.0, 1.0])
    yade = write_particles(tmp_path / "p.txt", micro, format="yade").read_text().splitlines()
    assert yade[0].startswith("# x y z r") and yade[2].split() == ["3.7", "2", "2", "0.8"]
    table = write_particles(tmp_path / "p.csv", micro).read_text().splitlines()
    assert table[0].split(",")[:5] == ["sphere", "x", "y", "z", "radius"] and len(table) == 3
    # clumps: the molecules of LAMMPS's hybrid style, the clumps of YADE
    seg = Segments(np.array([[1.0, 1.0, 1.0]]), np.array([[2.0, 1.0, 1.0]]), np.array([0.5]))
    rod = PrimitiveUnion(Domain(size=4.0, periodic=False), segments=seg)
    hybrid = write_particles(tmp_path / "rod.data", rod).read_text()
    assert "Atoms # hybrid" in hybrid and hybrid.strip().splitlines()[-1].split()[-1] == "1"
    assert (
        write_particles(tmp_path / "rod.txt", rod, format="yade")
        .read_text()
        .startswith("# x y z r clumpId")
    )
    # the particles of a structure without spheres: their table; equivalent spheres
    voronoi = model_from_config({"type": "voronoi", "n_cells": 6, "phase_fractions": [0.5, 0.5]})
    cells = voronoi.generate(Domain(size=1.0, periodic=True), 3)
    found = particles(cells, shape=40)
    table = particle_table(found)
    assert table["volume"].sum() == pytest.approx(
        float(np.mean(cells.voxelize(40).data == 1)), rel=0.05
    )
    out = export(cells, tmp_path / "cells.csv", particles=1, shape=40)
    assert out.read_text().splitlines()[0].startswith("particle,volume,area")
    assert export(micro, tmp_path / "e.stl", particles=1, shape=40).read_text().count("solid ") == 4
    with pytest.raises(ValueError, match="exact"):
        write_particles(tmp_path / "x.data", cells, exact=True)
    with pytest.raises(ValueError, match="suffixes"):
        write_particles(tmp_path / "x.png", micro)


def test_particles_on_the_command_line(tmp_path, capsys):
    """``morphogen export geometry.json particles.stl --particles 1`` writes the particles of
    a geometry; ``--format yade`` the spheres as YADE reads them."""
    from morphogen.cli import main
    from morphogen.io import save_microstructure

    geometry = save_microstructure(_two_spheres(), tmp_path / "micro.json")
    assert main(["export", str(geometry), str(tmp_path / "p.stl"), "--particles", "1",
                 "--shape", "40"]) == 0  # fmt: skip
    assert (tmp_path / "p.stl").read_text().count("endsolid") == 2
    assert main(["export", str(geometry), str(tmp_path / "p.txt"), "--particles", "1",
                 "--format", "yade"]) == 0  # fmt: skip
    assert (tmp_path / "p.txt").read_text().startswith("# x y z r")
    assert main(["export", str(geometry), str(tmp_path / "t.msh"), "--particles", "1",
                 "--tetrahedra"]) == 0  # fmt: skip
    assert {kind for kind, *_ in _lmgc90_read(tmp_path / "t.msh")[1]} == {"4"}
    assert main(["export", str(geometry), str(tmp_path / "q.txt"), "--format", "yade"]) != 0
    assert "--particles" in capsys.readouterr().err
    assert main(["export", str(geometry), str(tmp_path / "q.vti"), "--tetrahedra"]) != 0
    assert "--particles" in capsys.readouterr().err


def test_particles_in_the_export_of_the_interface(tmp_path):
    """The tab Export writes the particles of a phase of a morphology in the format chosen:
    one STL per particle in an archive of their own, the spheres for LAMMPS."""
    from morphogen.gui import api

    session = api.Session(tmp_path)
    made = api.generate(
        {"model": "hardcore_spheres", "params": {"radius": 0.4, "volume_fraction": 0.1},
         "domain": {"size": 3.0, "periodic": True}, "seed": 2, "resolution": 32},
        session,
    )  # fmt: skip
    key = made["item"]["key"]
    n = len(session.item(key)["micro"].spheres)
    request = {"source": key, "outputs": ["particles"], "shape": 48, "particles_phase": "1"}
    out = api.export({**request, "particles_format": "stl_each"}, session)
    names = {f["name"] for f in out["files"]}
    assert {"particles_stl.zip", "export.zip"} <= names
    import zipfile

    folder = tmp_path / out["archive"].split("/")[2]
    with zipfile.ZipFile(folder / "particles_stl.zip") as archive:
        assert len(archive.namelist()) == n  # every sphere whole, none cut by the faces
    for fmt, kind in (("msh", "2"), ("msh_tetrahedra", "4")):
        out = api.export({**request, "particles_format": fmt}, session)
        name = api.PARTICLE_FILES[fmt]
        assert name in {f["name"] for f in out["files"]}
        folder = tmp_path / out["archive"].split("/")[2]
        assert {k for k, *_ in _lmgc90_read(folder / name)[1]} == {kind}
        assert len(_separated(folder / name)) == n
    with pytest.raises(ValueError, match="triangle size"):
        api.export({**request, "particles_format": "msh", "particles_size": "fine"}, session)
    out = api.export({**request, "particles_format": "blender"}, session)
    folder = tmp_path / out["archive"].split("/")[2]
    with zipfile.ZipFile(folder / "blender_scene.zip") as archive:
        assert {"blender/scene.json", "blender/morphogen_blender.py"} <= set(archive.namelist())
    out = api.export({**request, "particles_format": "data"}, session)
    data = (tmp_path / out["archive"].split("/")[2] / "particles.data").read_text()
    assert f"{n} atoms" in data
    with pytest.raises(ValueError, match="phase of the particles"):
        api.export({**request, "particles_phase": "7"}, session)
    with pytest.raises(ValueError, match="format of the particles"):
        api.export({**request, "particles_format": "step"}, session)


def _lmgc90_read(path):
    """The file as LMGC90's pre-processor reads it (``pylmgc90.pre.readMesh``: its
    ``readGmshv2``, version 2026.rc1): the version on the second line, ASCII; every line
    split, its first word a keyword or the data of the section (a blank line breaks it);
    the nodes, the elements of the types it knows with their physical and geometrical
    entities (the first two tags), the physical names (their quotes stripped) in place of
    their numbers."""
    lines = Path(path).read_text(encoding="ascii").split("\n")
    assert lines[0] == "$MeshFormat" and lines[-1] == ""
    version, binary, _ = lines[1].split()
    assert version.startswith("2") and binary == "0"
    keywords = ("$MeshFormat", "$Nodes", "$EndNodes", "$Elements", "$EndElements",
                "$PhysicalNames", "$EndPhysicalNames", "$EndMeshFormat")  # fmt: skip
    nodes, elements, names, keyword = {}, [], {}, None
    for line in lines[2:-1]:
        words = line.split()
        assert words, "a blank line: LMGC90's reader fails on it"
        keyword = words[0] if words[0] in keywords else keyword
        if keyword == "$Nodes" and words[0] != "$Nodes" and len(words) > 1:
            nodes[int(words[0])] = np.array([float(w) for w in words[1:4]])
        if keyword == "$Elements" and words[0] != "$Elements" and len(words) > 1:
            assert words[1] in ("2", "4")  # T3xxx and TE4xx among its surfaces and volumes
            tags = int(words[2])
            elements.append((words[1], words[3], words[4], [int(w) for w in words[3 + tags :]]))
        if keyword == "$PhysicalNames" and words[0] != "$PhysicalNames" and len(words) > 1:
            names[words[1]] = " ".join(words[2:])[1:-1]
    return nodes, [(kind, names.get(pe, pe), ge, conn) for kind, pe, ge, conn in elements]


def _separated(path):
    """The meshes of the geometrical entities, as LMGC90's ``separateMeshes(dim,
    entity_type="geometricalEntity")`` gives them: ``{entity: (physical name, nodes,
    elements)}``, the elements numbered in their nodes."""
    nodes, elements = _lmgc90_read(path)
    out = {}
    for _, physical, entity, conn in elements:
        out.setdefault(entity, (physical, []))[1].append(conn)
    meshes = {}
    for entity, (physical, cells) in out.items():
        cells = np.array(cells)
        numbers, local = np.unique(cells, return_inverse=True)
        xyz = np.array([nodes[int(n)] for n in numbers])
        meshes[entity] = (physical, xyz, local.reshape(cells.shape))
    return meshes


def _closed_volume(xyz, triangles):
    """The volume a closed, consistently oriented triangle surface encloses (positive for
    outward normals), after checking that every edge joins two triangles that run along
    it in opposite directions."""
    directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    assert len(np.unique(directed, axis=0)) == len(directed)
    _, count = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    assert np.all(count == 2)
    v = xyz[triangles]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum()) / 6.0


def _tet_volumes(xyz, tets):
    p = xyz[tets]
    return np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0],
                                                             p[:, 3] - p[:, 0])) / 6.0  # fmt: skip


def _skin(tets):
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]],
                            tets[:, [1, 2, 3]]])  # fmt: skip
    _, count = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    return int((count == 1).sum())


def test_gmsh_files_for_lmgc90(tmp_path):
    """The Gmsh files read as LMGC90 reads them: one geometrical entity and one physical
    group per particle; its triangles a closed surface of outward normals that holds the
    volume of the particle, or its tetrahedra, positive, that fill that surface (their skin
    its triangles); the particles cut by the faces of a window named so; gmsh and meshio
    read the files too."""
    micro = _two_spheres()
    exact = _separated(write_particles(tmp_path / "exact.msh", micro))
    assert sorted(exact) == ["1", "2"] and exact["1"][0] == "particle_1"
    for (_, xyz, cells), r in zip([exact["1"], exact["2"]], RADII, strict=True):
        assert _closed_volume(xyz, cells) == pytest.approx(_ball(r), rel=1e-12)
    found = particles(micro, shape=48)
    surfaces = _separated(write_particles(tmp_path / "cut.msh", micro, shape=48, exact=False))
    tets = _separated(write_particles(tmp_path / "tets.msh", micro, shape=48, exact=False,
                                      elements="tetrahedra"))  # fmt: skip
    for k, p in enumerate(found, start=1):
        _, xyz, triangles = surfaces[str(k)]
        assert _closed_volume(xyz, triangles) == pytest.approx(p.volume, rel=1e-9)
        _, nodes, cells = tets[str(k)]
        volumes = _tet_volumes(nodes, cells)
        assert volumes.min() > 0 and volumes.sum() == pytest.approx(p.volume, rel=1e-9)
        assert _skin(cells) == len(triangles)
    window = _separated(write_particles(tmp_path / "w.msh", _two_spheres(periodic=False),
                                        shape=48, exact=False))  # fmt: skip
    assert sorted(m[0] for m in window.values()) == ["particle_1", "particle_2_cut"]
    import meshio

    read = meshio.read(tmp_path / "tets.msh")
    assert [b.type for b in read.cells] == ["tetra"] * len(read.cells)
    assert set(np.concatenate(read.cell_data["gmsh:geometrical"]).tolist()) == {1, 2}
    if importlib.util.find_spec("gmsh"):
        import gmsh

        gmsh.initialize(readConfigFiles=False, interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(str(tmp_path / "tets.msh"))
            assert len(gmsh.model.getEntities(3)) == 2
            groups = gmsh.model.getPhysicalGroups(3)
            assert sorted(gmsh.model.getPhysicalName(d, g) for d, g in groups) == [
                "particle_1", "particle_2"]  # fmt: skip
        finally:
            gmsh.finalize()
    with pytest.raises(ValueError, match="Gmsh file"):
        write_particles(tmp_path / "x.stl", micro, elements="tetrahedra")
    with pytest.raises(ValueError, match="elements"):
        write_particles(tmp_path / "x.msh", micro, elements="hexahedra")
    with pytest.raises(ValueError, match="size"):
        write_particles(tmp_path / "x.msh", micro, size=-1.0)


def test_capsules_and_fans():
    """A segment is the closed surface of its spherocylinder (a capsule of outward normals
    and of its exact volume), a sphere its icosphere; without gmsh, the fan of their
    triangles to their centres fills them (they are star-shaped about them), and a surface
    that is not raises an error that names gmsh."""
    seg = Segments(np.array([[1.0, 1.0, 1.0]]), np.array([[2.5, 1.6, 1.2]]), np.array([0.3]))
    rod = PrimitiveUnion(Domain(size=4.0, periodic=False), segments=seg,
                         spheres=Spheres(np.array([[3.0, 3.0, 3.0]]), np.array([0.4])))  # fmt: skip
    capsule, ball = sphere_surfaces(particle_spheres(rod))[::-1]
    length = float(np.linalg.norm(seg.end[0] - seg.start[0]))
    expected = math.pi * 0.09 * length + _ball(0.3)
    assert _closed_volume(capsule.vertices, capsule.faces) == pytest.approx(expected, rel=1e-12)
    assert ball.volume() == pytest.approx(_ball(0.4), rel=1e-12)
    radial = np.linalg.norm(np.cross(capsule.vertices - seg.start[0],
                                     (seg.end[0] - seg.start[0]) / length), axis=1)  # fmt: skip
    assert radial.max() == pytest.approx(0.3, rel=0.01)  # scaled a little to the volume
    saved = sys.modules.get("gmsh")
    sys.modules["gmsh"] = None  # import gmsh fails: the fans
    try:
        for (nodes, tets), s in zip(particle_tetrahedra([capsule, ball]), [capsule, ball],
                                    strict=True):  # fmt: skip
            volumes = _tet_volumes(nodes, tets)
            assert volumes.min() > 0 and volumes.sum() == pytest.approx(s.volume(), rel=1e-12)
            assert _skin(tets) == len(s.faces)
        apart = sphere_surfaces({"centre": np.array([[0.0, 0, 0], [3.0, 0, 0]]),
                                 "radius": np.array([0.5, 0.5]),
                                 "clump": np.array([0, 0])})  # fmt: skip
        with pytest.raises(ImportError, match="gmsh"):
            particle_tetrahedra(apart)  # two balls apart: their centre outside both
    finally:
        if saved is None:
            del sys.modules["gmsh"]
        else:
            sys.modules["gmsh"] = saved


def test_surfaces_remeshed_by_mmg(tmp_path):
    """MMG remeshes the surfaces cut from the voxels at regular triangles of the size asked
    (its Hausdorff distance a tenth of it), closed and outward, their volumes within 2 %."""
    from morphogen.mesh.conforming import mmg_executable

    try:
        mmg_executable(program="mmgs")
    except (ImportError, FileNotFoundError):
        pytest.skip("MMG's mmgs (pymmg) is not installed")
    found = particles(_two_spheres(), shape=48)
    size = 0.2
    remeshed = remesh_surfaces([p.surface for p in found], size)
    for s, p in zip(remeshed, found, strict=True):
        assert _closed_volume(s.vertices, s.faces) == pytest.approx(p.volume, rel=0.02)
        v = s.vertices[s.faces]
        edges = np.linalg.norm(v - np.roll(v, 1, axis=1), axis=2)
        assert 0.4 * size < np.median(edges) < 1.2 * size
        area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
        quality = 4 * math.sqrt(3) * area / (edges**2).sum(axis=1)
        assert quality.min() > 0.3 and len(s.faces) < len(p.surface.faces)
    path = write_particles(tmp_path / "coarse.msh", _two_spheres(), shape=48, exact=False,
                           size=size)  # fmt: skip
    assert len(_separated(path)) == 2
