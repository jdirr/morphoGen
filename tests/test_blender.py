"""morphoGen and Blender: the scene of the rigid bodies written for Blender, the particles
moved by its transforms read back (spheres, capsules, surfaces), the particles given by
their surfaces voxelized; and, where Blender or the module bpy is found, the particles
settled under gravity without overlaps."""

from __future__ import annotations

import json
import math
import os
import shutil

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

import morphogen as mg
from morphogen.interop import blender as bl
from morphogen.io.dem import particle_spheres, sphere_surfaces
from morphogen.models import Microstructure
from morphogen.models.mesh_particles import MeshParticles
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments, Spheres

CUBE = mg.Domain.cube(1.0, periodic=True)


def _spheres():
    centres = np.array([[0.05, 0.5, 0.5], [0.5, 0.5, 0.2], [0.8, 0.3, 0.7]])
    return PrimitiveUnion(CUBE, spheres=Spheres(centres, np.array([0.1, 0.15, 0.12])))


def _moved(scene, move):
    """transforms.json as Blender writes it: ``move(k, location)`` gives the location and
    quaternion of particle k in the scaled scene."""
    spec = json.loads((scene.folder / bl.SCENE).read_text())
    s = spec["scale"]
    out = []
    for k, p in enumerate(spec["particles"]):
        location, quaternion = move(k, np.asarray(p["centre"]) * s)
        out.append({"name": f"particle_{k}", "location": list(location),
                    "quaternion": list(quaternion)})  # fmt: skip
    (scene.folder / bl.TRANSFORMS).write_text(json.dumps({"particles": out}))


def test_scene_for_blender(tmp_path):
    """The scene: the exact spheres inside the walls (a sphere through a periodic face
    moved in), the box of the domain and twice its height, the script for Blender."""
    scene = bl.write_blender(tmp_path / "s", _spheres(), frames=10)
    assert scene.particles == 3 and scene.kinds == {"sphere": 3}
    spec = json.loads((scene.folder / bl.SCENE).read_text())
    assert spec["upper"] == [1.0, 1.0, 2.0] and spec["scale"] == pytest.approx(10.0)
    first = spec["particles"][0]
    assert first["centre"] == pytest.approx([0.1, 0.5, 0.5]) and first["radius"] == 0.1
    compile(scene.script.read_text(), str(scene.script), "exec")  # Python for Blender
    with pytest.raises(ValueError, match="collision"):
        bl.write_blender(tmp_path / "x", _spheres(), collision="BOX")
    seg = Segments(np.array([[0.2, 0.5, 0.5]]), np.array([[0.6, 0.5, 0.5]]), np.array([0.05]))
    fibre = PrimitiveUnion(CUBE, segments=seg)
    spec = json.loads((bl.write_blender(tmp_path / "f", fibre).folder / bl.SCENE).read_text())
    (capsule,) = spec["particles"]
    assert capsule["kind"] == "capsule" and capsule["axis"] == pytest.approx([0.4, 0, 0])


def test_particles_moved_by_the_transforms(tmp_path):
    """The transforms of Blender read back: spheres dropped by a translation, a capsule
    turned a quarter turn about y (its axis from x to z), surfaces moved; the window up to
    the top of the highest particle."""
    scene = bl.write_blender(tmp_path / "s", _spheres())
    _moved(scene, lambda k, c: (c - np.array([0, 0, 1.0]), [1, 0, 0, 0]))  # 0.1 down
    moved = bl.read_blender(scene)
    assert isinstance(moved, PrimitiveUnion) and not moved.domain.periodic
    assert moved.spheres.centers[1] == pytest.approx([0.5, 0.5, 0.1])
    assert moved.domain.size[2] == pytest.approx(0.7 - 0.1 + 0.12)  # the top of the third
    seg = Segments(np.array([[0.3, 0.5, 0.5]]), np.array([[0.7, 0.5, 0.5]]), np.array([0.05]))
    scene = bl.write_blender(tmp_path / "f", PrimitiveUnion(CUBE, segments=seg))
    # the orientation of the body in the world: its axis (local z) along x, as it was made
    half = math.sqrt(0.5)
    _moved(scene, lambda k, c: (c, [half, 0, half, 0]))  # a quarter turn about y: z to x
    same = bl.read_blender(scene).segments
    assert same.start[0] == pytest.approx([0.3, 0.5, 0.5]) and same.radii[0] == 0.05
    assert same.end[0] == pytest.approx([0.7, 0.5, 0.5])
    _moved(scene, lambda k, c: (c, [1, 0, 0, 0]))  # no turn in the world: the axis along z
    upright = bl.read_blender(scene).segments
    assert upright.start[0] == pytest.approx([0.5, 0.5, 0.3])
    assert upright.end[0] == pytest.approx([0.5, 0.5, 0.7])
    surfaces = PrimitiveUnion(CUBE, spheres=Spheres(np.array([[0.5, 0.5, 0.5]]), np.array([0.2])))
    scene = bl.write_blender(tmp_path / "m", surfaces, exact=False, shape=32)
    assert scene.kinds == {"mesh": 1}
    _moved(scene, lambda k, c: (c + np.array([0, 0, 1.0]), [1, 0, 0, 0]))
    up = bl.read_blender(scene, height=1.0)
    assert isinstance(up, MeshParticles) and up.domain.size[2] == pytest.approx(1.0)
    assert up.surfaces[0].vertices.mean(axis=0)[2] == pytest.approx(0.6, abs=0.01)
    with pytest.raises(ValueError, match="transforms"):
        (scene.folder / bl.TRANSFORMS).write_text(json.dumps({"particles": []}))
        bl.read_blender(scene)


def test_particles_of_surfaces():
    """Particles given by closed surfaces: their voxels are those of the spheres they
    approximate (icospheres), through the periodic faces too; their implicit function has
    the sign of the voxels; saved and read back."""
    exact = PrimitiveUnion(CUBE, spheres=Spheres(np.array([[0.3, 0.3, 0.3], [0.9, 0.5, 0.95]]),
                                                 np.array([0.2, 0.15])))  # fmt: skip
    meshed = MeshParticles(CUBE, sphere_surfaces(particle_spheres(exact), level=4))
    image = meshed.voxelize(48)
    assert np.mean(image.data == exact.voxelize(48).data) > 0.9999
    level = meshed.level_set(48)
    assert np.array_equal(level.data < 0, image.data == 1)
    window = MeshParticles(mg.Domain.cube(1.0, periodic=False), meshed.surfaces)
    assert window.voxelize(48).data.sum() < image.data.sum()  # the parts beyond the faces
    back = Microstructure.from_dict(json.loads(json.dumps(meshed.to_dict())))
    assert np.array_equal(back.voxelize(24).data, meshed.voxelize(24).data)
    fraction = meshed.voxelize(16, supersampling=2)
    assert fraction.name == "fraction" and 0.0 < float(fraction.data.mean()) < 1.0


def _runner() -> dict:
    """How to run Blender here: its program, or a Python with the module bpy
    (MORPHOGEN_BPY_PYTHON); empty when neither is found."""
    python = os.environ.get("MORPHOGEN_BPY_PYTHON")
    if python:
        return {"python": python}
    blender = bl.blender_executable()
    return {"blender": blender} if blender and shutil.which(blender) else {}


@pytest.mark.skipif(not _runner(), reason="Blender (or a Python with bpy) is not found")
def test_spheres_and_fibres_settled_by_blender(tmp_path):
    """Hard spheres dropped in Blender: they fall, inside the walls, onto the floor, and
    overlap by less than 1 % of their radius (Bullet's spheres); fibres keep their lengths
    and radii and lie lower than they started."""
    micro = mg.models.HardcoreSpheres(radius=0.1, volume_fraction=0.15).generate(CUBE, seed=1)
    settled = bl.settle(micro, folder=tmp_path / "s", frames=150, **_runner())
    c, r = settled.spheres.centers, settled.spheres.radii
    gaps = squareform(pdist(c)) - (r[:, None] + r[None, :])
    np.fill_diagonal(gaps, np.inf)
    assert -gaps.min() < 0.01 * r.min()
    assert np.all(c[:, :2] >= r[:, None] - 1e-3) and np.all(c[:, :2] <= 1.0 - r[:, None] + 1e-3)
    assert (c[:, 2] - r).min() > -1e-3 and c[:, 2].mean() < micro.spheres.centers[:, 2].mean()
    assert (tmp_path / "s" / "scene.blend").is_file()
    fibres = mg.models.HardcoreFibers(radius=0.04, length=0.3, volume_fraction=0.08)
    loose = fibres.generate(CUBE, seed=2)
    packed = bl.settle(loose, folder=tmp_path / "f", frames=150, **_runner())
    assert np.allclose(packed.segments.lengths, loose.segments.lengths)
    assert packed.segments.midpoints[:, 2].mean() < loose.segments.midpoints[:, 2].mean()


def _two_balls(periodic=True):
    domain = mg.Domain.cube(1.0, periodic=periodic)
    centres = np.array([[0.3, 0.5, 0.5], [0.7, 0.4, 0.6]])
    return PrimitiveUnion(domain, spheres=Spheres(centres, np.array([0.2, 0.15])))


def _merged(path):
    """The surface of a binary STL file on merged vertices."""
    from morphogen.io.meshes import read_stl
    from morphogen.mesh.surface import SurfaceMesh

    raw = read_stl(path)
    vertices, index = np.unique(raw.vertices, axis=0, return_inverse=True)
    return SurfaceMesh(vertices, index.reshape(-1)[raw.faces])


def _closed(surface) -> bool:
    """Every edge in two triangles that run along it in opposite directions."""
    f = surface.faces
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, count = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    return bool(np.all(count == 2)) and len(np.unique(directed, axis=0)) == len(directed)


def test_scene_of_a_microstructure_for_blender(tmp_path, monkeypatch):
    """The scene of a microstructure: the closed surface of each phase from the mesh
    conforming to its interfaces (the exact route of the spheres), of the volume of the
    phase; the fluid domain extended upstream and downstream along the axis, its inflow and
    outflow in the extensions, the phases its obstacles; the marching cubes of the phases
    of a mosaic, and without MMG; the errors."""
    micro = _two_balls()
    scene = bl.write_blender_scene(tmp_path / "s", micro, fluid="gas", size=0.12)
    assert scene.objects == ["phase_1"] and scene.route == "level_set" and scene.fluid == "gas"
    surface = _merged(scene.folder / "phase_1.stl")
    balls = 4.0 / 3.0 * math.pi * (0.2**3 + 0.15**3)
    assert _closed(surface) and surface.volume() == pytest.approx(balls, rel=0.03)
    spec = json.loads((scene.folder / bl.SCENE).read_text())
    assert spec["kind"] == "microstructure" and spec["route"] == "level_set"
    assert spec["objects"][0]["volume"] == pytest.approx(surface.volume(), rel=1e-5)
    fluid = spec["fluid"]
    assert fluid["lower"] == pytest.approx([-0.25, 0, 0])
    assert fluid["upper"] == pytest.approx([1.25, 1, 1])
    (in_lo, in_hi), (out_lo, out_hi) = fluid["inflow"], fluid["outflow"]
    assert -0.25 < in_lo[0] < in_hi[0] < 0.0 and 1.0 < out_lo[0] < out_hi[0] < 1.25
    assert in_lo[1] > 0.0 and in_hi[2] < 1.0  # inside the walls of the domain
    assert fluid["velocity"] == [1.0, 0.0, 0.0] and fluid["effectors"] == ["phase_1"]
    assert fluid["type"] == "GAS" and spec["scale"] == pytest.approx(2.0 / 1.5)
    compile(scene.script.read_text(), str(scene.script), "exec")  # Python for Blender
    along_z = bl.write_blender_scene(tmp_path / "z", micro, fluid="liquid", axis=2,
                                     surfaces="implicit", shape=32, viscosity=1e-6)  # fmt: skip
    spec = json.loads((along_z.folder / bl.SCENE).read_text())
    assert spec["fluid"]["upper"] == pytest.approx([1, 1, 1.25])
    assert spec["fluid"]["velocity"] == [0.0, 0.0, 1.0] and spec["fluid"]["type"] == "LIQUID"
    assert spec["fluid"]["viscosity"] == 1e-6
    # the cells of a mosaic of three phases: a surface each, closed, of the volume of its phase
    mosaic = mg.models.model_from_config(
        {"type": "voronoi", "n_cells": 8, "phase_fractions": [0.3, 0.3, 0.4]}
    ).generate(mg.Domain.cube(1.0, periodic=False), 3)
    cells = bl.write_blender_scene(tmp_path / "m", mosaic, surfaces="implicit", shape=40)
    assert cells.objects == ["phase_1", "phase_2"] and cells.route == "marching cubes"
    image = mosaic.voxelize(shape=40)
    for k in (1, 2):
        s = _merged(cells.folder / f"phase_{k}.stl")
        assert _closed(s) and s.volume() == pytest.approx(np.mean(image.data == k), abs=0.02)
    assert not (cells.folder / "phase_0.stl").exists()
    import morphogen.mesh

    def no_mmg(*args, **kwargs):
        raise ImportError("no MMG")

    monkeypatch.setattr(morphogen.mesh, "conforming_mesh", no_mmg)
    plain = bl.write_blender_scene(tmp_path / "p", micro, shape=32)
    assert plain.route == "marching cubes" and _closed(_merged(plain.folder / "phase_1.stl"))
    assert json.loads((plain.folder / bl.SCENE).read_text())["fluid"] is None
    for bad, match in (({"fluid": "water"}, "fluid"), ({"axis": 3}, "axis"),
                       ({"phases": [7]}, "phases"), ({"surfaces": "voxels"}, "surfaces"),
                       ({"velocity": -1.0}, "velocity")):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            bl.write_blender_scene(tmp_path / "x", micro, **bad)


def _uni(path, values):
    """A grid written as Mantaflow writes it: gzip, the header MNT3 (sizes, grid type,
    element type and size, 252 bytes of information, a fourth dimension, a time stamp),
    the values x fastest."""
    import gzip
    import struct

    nx, ny, nz = values.shape[:3]
    components = 1 if values.ndim == 3 else values.shape[3]
    head = struct.pack("<6i", nx, ny, nz, 12 if components == 3 else 1,
                       2 if components == 3 else 1, 4 * components)  # fmt: skip
    head += b"mantaflow".ljust(252, b"\0") + struct.pack("<iQ", 0, 1)
    data = np.ascontiguousarray(
        values.reshape(nx, ny, nz, components).transpose(2, 1, 0, 3), dtype="<f4"
    )
    path.write_bytes(gzip.compress(b"MNT3" + head + data.tobytes()))


def test_flow_read_back(tmp_path):
    """The grids that Mantaflow caches read back (gzip, the header MNT3, x fastest): the
    velocity and density of a frame on the cells of the fluid domain, located in the units
    of the microstructure; its voxel images; the errors."""
    scene = bl.write_blender_scene(tmp_path / "s", _two_balls(), fluid="gas", shape=24,
                                   surfaces="implicit")  # fmt: skip
    data = scene.folder / "cache" / "data"
    data.mkdir(parents=True)
    rng = np.random.default_rng(1)
    velocity = rng.normal(size=(12, 8, 8, 3)).astype(np.float32)
    density = rng.random((12, 8, 8)).astype(np.float32)
    for frame in (1, 2):
        _uni(data / f"vel_{frame:04d}.uni", velocity * frame)
    _uni(data / "density_0002.uni", density)
    flow = bl.read_blender_flow(scene)
    assert flow.frame == 2 and np.array_equal(flow.velocity, 2 * velocity)
    assert np.array_equal(flow.density, density)
    assert flow.origin == pytest.approx([-0.25, 0, 0])
    assert flow.spacing == pytest.approx([1.5 / 12, 1 / 8, 1 / 8])
    first = bl.read_blender_flow(scene.folder, frame=1)
    assert first.density is None and np.array_equal(first.velocity, velocity)
    images = flow.images()
    assert [i.name for i in images] == ["velocity_x", "velocity_y", "velocity_z", "speed",
                                        "density"]  # fmt: skip
    assert images[3].data == pytest.approx(np.linalg.norm(2.0 * velocity, axis=-1), rel=1e-6)
    assert images[0].origin == pytest.approx([-0.25, 0, 0])
    with pytest.raises(ValueError, match="frame"):
        bl.read_blender_flow(scene, frame=3)
    dry = bl.write_blender_scene(tmp_path / "d", _two_balls(), shape=24, surfaces="implicit")
    with pytest.raises(ValueError, match="fluid"):
        bl.read_blender_flow(dry)
    empty = bl.write_blender_scene(tmp_path / "e", _two_balls(), fluid="gas", shape=24,
                                   surfaces="implicit")  # fmt: skip
    with pytest.raises(FileNotFoundError, match="bake"):
        bl.read_blender_flow(empty)


def test_blender_scene_in_the_export_of_the_interface(tmp_path):
    """The tab Export writes the scene of a morphology for Blender: the surfaces of its
    phases, scene.json with the fluid and axis chosen, and the script, in an archive."""
    import zipfile

    from morphogen.gui import api

    session = api.Session(tmp_path)
    made = api.generate(
        {"model": "hardcore_spheres", "params": {"radius": 0.4, "volume_fraction": 0.1},
         "domain": {"size": 3.0, "periodic": True}, "seed": 2, "resolution": 32},
        session,
    )  # fmt: skip
    request = {"source": made["item"]["key"], "outputs": ["blender"], "element_size": "0.75",
               "blender_fluid": "liquid", "blender_axis": "z"}  # fmt: skip
    out = api.export(request, session)
    assert "blender_scene.zip" in {f["name"] for f in out["files"]}
    folder = tmp_path / out["archive"].split("/")[2]
    with zipfile.ZipFile(folder / "blender_scene.zip") as archive:
        names = set(archive.namelist())
        spec = json.loads(archive.read("blender_scene/scene.json"))
    assert {"blender_scene/phase_1.stl", "blender_scene/" + bl.SCENE_SCRIPT} <= names
    assert spec["fluid"]["type"] == "LIQUID" and spec["fluid"]["axis"] == 2
    with pytest.raises(ValueError, match="fluid"):
        api.export({**request, "blender_fluid": "water"}, session)


@pytest.mark.skipif(not _runner(), reason="Blender (or a Python with bpy) is not found")
def test_scene_built_by_blender(tmp_path):
    """Blender builds the scene: the object of the phase with the volume of its surface,
    the fluid domain, the phase its obstacle, the inflow and the outflow, the walls along
    the flow and its two ends open; with the program Blender, Mantaflow bakes the flow,
    nil in the obstacles and running along the axis in the pores (the module bpy of some
    versions bakes nothing: its Mantaflow lacks its bindings)."""
    program = bl.blender_executable()
    baker = {"blender": program} if program and shutil.which(program) else None
    scene = bl.write_blender_scene(tmp_path / "s", _two_balls(), fluid="gas", size=0.12,
                                   resolution=24, frames=6, bake=baker is not None)  # fmt: skip
    report = json.loads(bl.run_blender(scene, **(baker or _runner())).read_text())
    spec = json.loads((scene.folder / bl.SCENE).read_text())
    objects = report["objects"]
    phase = objects["phase_1"]
    assert phase["fluid"] == "EFFECTOR" and phase["effector"] == "COLLISION"
    assert phase["volume"] == pytest.approx(spec["objects"][0]["volume"], rel=1e-4)
    assert objects["fluid_domain"]["fluid"] == "DOMAIN"
    assert objects["fluid_domain"]["volume"] == pytest.approx(1.5, rel=1e-5)
    assert objects["inflow"]["flow"] == "INFLOW" and objects["outflow"]["flow"] == "OUTFLOW"
    assert objects["inflow"]["velocity"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    fluid = report["fluid"]
    assert fluid["domain_type"] == "GAS" and fluid["resolution_max"] == 24
    assert fluid["borders"] == {"left": False, "right": False, "front": True, "back": True,
                                "bottom": True, "top": True}  # fmt: skip
    assert (scene.folder / "scene.blend").is_file()
    if baker is None:
        return
    assert report["bake"]["velocity_frames"] == 6
    flow = bl.read_blender_flow(scene)
    nx, ny, nz = flow.velocity.shape[:3]
    assert (nx, ny, nz) == (24, 16, 16)
    centres = [flow.origin[k] + (np.arange(n) + 0.5) * flow.spacing[k]
               for k, n in enumerate((nx, ny, nz))]  # fmt: skip
    grid = np.stack(np.meshgrid(*centres, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = np.zeros(len(grid), dtype=bool)
    sample = np.all((grid >= 0.0) & (grid <= 1.0), axis=1)
    inside[sample] = _two_balls().level_set_at(grid[sample]) < -0.05
    speed = np.linalg.norm(flow.velocity.reshape(-1, 3), axis=1)
    pores = sample & ~inside
    assert inside.sum() > 20 and speed[inside].mean() < 0.05 * speed[pores].mean()
    assert flow.velocity.reshape(-1, 3)[pores, 0].mean() > 0.0
