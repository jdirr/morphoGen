"""morphoGen and Blender: a microstructure as the objects of a scene of Blender, with a fluid
domain around them for its fluid solver; the particles of a phase as the rigid bodies of
Blender's physics (Bullet), settled under gravity in a box, and the packing read back.

:func:`write_blender_scene` writes a folder for Blender: the closed surface of each phase
of a realization (:func:`phase_surfaces`: the boundary of its tetrahedra in the mesh
conforming to the interfaces, from the exact geometry when the structure has it; or the
marching cubes of its implicit function), as binary STL files that other codes read as
well (snappyHexMesh, the mesher of OpenFOAM), ``scene.json`` and the script
``morphogen_blender_scene.py``, which makes a mesh object of each phase with its colour
and, for a fluid, sets up Mantaflow (the fluid solver of Blender): its domain (the box of
the sample, extended upstream and downstream along an axis), the phases as obstacles
(effectors of collision), an inflow at one end and an outflow at the other; it saves
``scene.blend`` and bakes the flow if asked. Mantaflow is a solver for visual effects
(smoke and liquids, its own units, no viscosity for gases): the flow it bakes shows the
streams through the pores, not quantitative Stokes or Darcy flows; the scene is the
geometry of the sample for Blender, its fluid tools and the add-ons that drive OpenFOAM
from it. :func:`read_blender_flow` reads the velocity grid it caches (``.uni`` files of
Mantaflow) back as arrays on the grid of the domain.

:func:`write_blender` writes a folder for Blender: the particles of a phase (the exact
spheres and spherocylinders of a union of spheres and segments, Bullet's own sphere and
capsule shapes; else the closed surfaces of :func:`morphogen.io.dem.particles`, their
convex hulls), an open box of passive walls around them, the world of the rigid bodies
(gravity, substeps, iterations, friction, restitution) in ``scene.json``, and the script
``morphogen_blender.py``, which builds the scene, runs the simulation and writes the
final place of each particle (``transforms.json``) and the scene (``scene.blend``, to look
at it, render it or go on in Blender). :func:`run_blender` runs the script with
Blender (``blender --background --python``; the program found on the PATH or by
``MORPHOGEN_BLENDER``) or with a Python that has the module ``bpy`` (``pip install bpy``).
:func:`read_blender` turns the particles moved into a microstructure: the spheres and
segments moved (:class:`~morphogen.models.union.PrimitiveUnion`), the surfaces moved
(:class:`~morphogen.models.mesh_particles.MeshParticles`), in the box of the walls up to
the top of the packing. :func:`settle` does the three.

The scene is scaled so that the box is ``scene_size`` units of Blender wide (rigid bodies
behave best between 0.1 and 10 units), and the transforms are scaled back.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import struct
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from morphogen.domain import Domain

SCRIPT = "morphogen_blender.py"
SCENE = "scene.json"
TRANSFORMS = "transforms.json"
#: The script of a scene of a microstructure, and what it reports of the scene built
SCENE_SCRIPT = "morphogen_blender_scene.py"
REPORT = "report.json"
#: The fluids of Mantaflow's domains
FLUIDS = ("gas", "liquid")
#: The colours of the phases in Blender (RGBA), the phase k taking the (k - 1)-th
PHASE_COLOURS = (
    (0.72, 0.74, 0.78, 1.0),
    (0.85, 0.45, 0.20, 1.0),
    (0.25, 0.55, 0.85, 1.0),
    (0.35, 0.70, 0.35, 1.0),
    (0.80, 0.30, 0.55, 1.0),
    (0.90, 0.80, 0.25, 1.0),
)


@dataclass
class BlenderScene:
    """The folder of a scene for Blender and what it holds."""

    folder: Path
    particles: int
    kinds: dict

    @property
    def script(self) -> Path:
        return self.folder / SCRIPT


def _capsule(start, end, radius) -> dict:
    axis = np.asarray(end, float) - np.asarray(start, float)
    return {"kind": "capsule", "centre": (0.5 * (np.asarray(start) + np.asarray(end))).tolist(),
            "axis": axis.tolist(), "radius": float(radius)}  # fmt: skip


def _particles(micro, phase: int, shape, exact: bool | None) -> list[dict]:
    """The particles of the scene: spheres, capsules or surfaces (vertices about their
    centroid, which is the place of the particle)."""
    from morphogen.io.dem import _has_spheres, particles

    exact = _has_spheres(micro, phase) if exact is None else bool(exact)
    out = []
    if exact:
        if not _has_spheres(micro, phase):
            raise ValueError("exact: the spheres and segments of a PrimitiveUnion (phase 1)")
        domain = micro.domain
        for c, r in zip(micro.spheres.centers, micro.spheres.radii, strict=True):
            c = domain.wrap(c) if domain.periodic else np.asarray(c)
            out.append(
                {"kind": "sphere", "centre": np.asarray(c, float).tolist(), "radius": float(r)}
            )
        seg = micro.segments
        for a, b, r in zip(seg.start, seg.end, seg.radii, strict=True):
            middle = 0.5 * (a + b)
            shift = domain.wrap(middle) - middle if domain.periodic else 0.0
            out.append(_capsule(a + shift, b + shift, r))
        return out
    for p in particles(micro, phase, shape=shape):
        centre = p.centroid
        out.append({"kind": "mesh", "centre": centre.tolist(),
                    "vertices": (p.surface.vertices - centre).tolist(),
                    "faces": p.surface.faces.tolist(), "volume": float(p.volume)})  # fmt: skip
    return out


def _inside_box(particles: list[dict], lower, upper) -> None:
    """Each particle moved, if needed, so that its bounding box is inside the walls
    across x and y (the particles of a periodic box that cross its faces)."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    for p in particles:
        c = np.asarray(p["centre"], float)
        if p["kind"] == "sphere":
            reach = np.full(3, p["radius"])
        elif p["kind"] == "capsule":
            reach = 0.5 * np.abs(np.asarray(p["axis"])) + p["radius"]
        else:
            v = np.asarray(p["vertices"])
            reach = np.maximum(-v.min(axis=0), v.max(axis=0))
        lo, hi = lower[:2] + reach[:2], upper[:2] - reach[:2]
        c[:2] = np.where(hi >= lo, np.clip(c[:2], lo, hi), 0.5 * (lower[:2] + upper[:2]))
        c[2] = max(c[2], lower[2] + reach[2])
        p["centre"] = c.tolist()


def write_blender(
    folder,
    micro,
    phase: int = 1,
    *,
    shape=64,
    exact: bool | None = None,
    frames: int = 250,
    height: float = 2.0,
    gravity: float = 9.81,
    friction: float = 0.5,
    restitution: float = 0.1,
    density: float = 1.0,
    substeps: int = 20,
    iterations: int = 40,
    collision: str = "CONVEX_HULL",
    scene_size: float = 10.0,
    save_blend: bool = True,
) -> BlenderScene:
    """A folder for Blender: the particles of a phase of a realization as rigid bodies
    (see the module), in an open box of the width of its domain and ``height`` times its
    height, dropped under ``gravity`` (units of the domain per second squared, before the
    scaling of the scene; Bullet's world of ``substeps`` steps per frame and ``iterations``
    of its solver) for ``frames`` frames at 24 per second; ``collision`` the shape of the
    surfaces (``CONVEX_HULL``, or ``MESH``: concave, slower); ``density`` gives the mass of
    each particle from its volume."""
    if collision not in ("CONVEX_HULL", "MESH"):
        raise ValueError("collision: CONVEX_HULL or MESH")
    if not frames >= 1 or not height >= 1.0:
        raise ValueError("frames: one at least; height: at least that of the domain (1)")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    domain: Domain = micro.domain
    found = _particles(micro, int(phase), shape, exact)
    lower = domain.lower
    upper = domain.upper.copy()
    upper[2] = lower[2] + height * domain.lengths[2]
    _inside_box(found, lower, upper)
    scale = float(scene_size) / float(max(domain.lengths[0], domain.lengths[1]))
    spec = {
        "morphogen": _version(),
        "scale": scale,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "frames": int(frames),
        "gravity": float(gravity),
        "friction": float(friction),
        "restitution": float(restitution),
        "density": float(density),
        "substeps": int(substeps),
        "iterations": int(iterations),
        "collision": collision,
        "save_blend": bool(save_blend),
        "particles": found,
    }
    (folder / SCENE).write_text(json.dumps(spec), encoding="utf-8")
    (folder / SCRIPT).write_text(BLENDER_SCRIPT, encoding="utf-8")
    kinds: dict = {}
    for p in found:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    return BlenderScene(folder, len(found), kinds)


def _version() -> str:
    from morphogen import __version__

    return __version__


def blender_executable() -> str | None:
    """The program Blender: ``MORPHOGEN_BLENDER``, or ``blender`` on the PATH (None if
    neither)."""
    given = os.environ.get("MORPHOGEN_BLENDER")
    if given:
        return given
    return shutil.which("blender")


def _script_of(scene) -> tuple[Path, Path, Path]:
    """The folder of a scene, its script and the file the script writes last: the rigid
    bodies (:class:`BlenderScene`, ``transforms.json``) or a microstructure
    (:class:`MicrostructureScene`, ``report.json``), a folder by its ``scene.json``."""
    folder = Path(scene.folder if hasattr(scene, "folder") else scene)
    kind = "rigid_bodies"
    if isinstance(scene, MicrostructureScene):
        kind = "microstructure"
    elif not isinstance(scene, BlenderScene) and (folder / SCENE).is_file():
        kind = json.loads((folder / SCENE).read_text(encoding="utf-8")).get("kind", kind)
    if kind == "microstructure":
        return folder, folder / SCENE_SCRIPT, folder / REPORT
    return folder, folder / SCRIPT, folder / TRANSFORMS


def run_blender(scene, *, blender: str | None = None, python: str | None = None,
                timeout: float | None = None) -> Path:  # fmt: skip
    """Run the script of a scene (a :class:`BlenderScene`, a :class:`MicrostructureScene`
    or its folder): with ``python``, a Python that has the module ``bpy``; else with
    Blender in the background (``blender``, by default :func:`blender_executable`).
    Returns the file the script writes last (the transforms of the rigid bodies, the
    report of the scene of a microstructure); raises RuntimeError with the end of
    Blender's output when it fails. A script that wrote it before Blender failed on its
    way out went through, and its file is taken with a warning: the module bpy of PyPI
    (5.3) fails so on a scene with a fluid, whose Mantaflow lacks its Python bindings
    (bake with the program Blender)."""
    folder, script, out = _script_of(scene)
    if python is not None:
        command = [python, str(script), str(folder)]
    else:
        program = blender or blender_executable()
        if program is None:
            raise RuntimeError(
                "Blender not found: install it (blender on the PATH, or MORPHOGEN_BLENDER), "
                "or give a Python that has the module bpy (pip install bpy); the folder "
                f"{folder} holds the scene and its script to run by hand"
            )
        command = [program, "--background", "--python-exit-code", "1", "--python",
                   str(script), "--", str(folder)]  # fmt: skip
    if out.is_file():
        out.unlink()  # that of an earlier run
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0 and out.is_file():  # written last: the script went through
        warnings.warn(
            f"Blender exited with code {done.returncode} after its script wrote {out.name} "
            f"(the end of its output: {(done.stdout + done.stderr)[-300:]!r})",
            stacklevel=2,
        )
        return out
    if done.returncode != 0 or not out.is_file():
        tail = (done.stdout + done.stderr)[-2000:]
        raise RuntimeError(f"Blender failed ({done.returncode}): {tail}")
    return out


def _rotation(q) -> np.ndarray:
    """The rotation matrix of a unit quaternion (w, x, y, z)."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def read_blender(scene, *, height: float | None = None):
    """The particles of a scene moved by Blender (its ``transforms.json``) as a
    microstructure in a window (non-periodic) of the box of the walls: up to ``height``
    (a length of the domain), by default the top of the highest particle. Spheres and
    capsules give a :class:`~morphogen.models.union.PrimitiveUnion` (spheres and round
    segments), surfaces a :class:`~morphogen.models.mesh_particles.MeshParticles`."""
    from morphogen.mesh.surface import SurfaceMesh
    from morphogen.models.mesh_particles import MeshParticles
    from morphogen.models.union import PrimitiveUnion
    from morphogen.primitives import Segments, Spheres

    folder = Path(scene.folder if isinstance(scene, BlenderScene) else scene)
    spec = json.loads((folder / SCENE).read_text(encoding="utf-8"))
    moved = json.loads((folder / TRANSFORMS).read_text(encoding="utf-8"))["particles"]
    if len(moved) != len(spec["particles"]):
        raise ValueError("read_blender: the transforms are not those of the scene's particles")
    s = float(spec["scale"])
    spheres_c, spheres_r, seg_a, seg_b, seg_r, surfaces = [], [], [], [], [], []
    top = -math.inf
    for p, t in zip(spec["particles"], moved, strict=True):
        place = np.asarray(t["location"], float) / s
        turn = _rotation(t["quaternion"])
        if p["kind"] == "sphere":
            spheres_c.append(place)
            spheres_r.append(p["radius"])
            top = max(top, place[2] + p["radius"])
        elif p["kind"] == "capsule":
            half = 0.5 * float(np.linalg.norm(p["axis"]))
            axis = turn @ np.array([0.0, 0.0, half])
            seg_a.append(place - axis)
            seg_b.append(place + axis)
            seg_r.append(p["radius"])
            top = max(top, max(place[2] - axis[2], place[2] + axis[2]) + p["radius"])
        else:
            v = np.asarray(p["vertices"], float) @ turn.T + place
            surfaces.append(SurfaceMesh(v, np.asarray(p["faces"], np.int64)))
            top = max(top, float(v[:, 2].max()))
    lower, upper = np.asarray(spec["lower"], float), np.asarray(spec["upper"], float)
    z_top = lower[2] + float(height) if height is not None else min(top, upper[2])
    if not z_top > lower[2]:
        raise ValueError("read_blender: no particle above the floor")
    window = Domain(size=(upper[0] - lower[0], upper[1] - lower[1], z_top - lower[2]),
                    origin=tuple(lower), periodic=False)  # fmt: skip
    meta = {"blender": {"frames": spec["frames"], "particles": len(moved)}}
    if surfaces:
        return MeshParticles(window, surfaces, metadata=meta)
    spheres = Spheres(np.array(spheres_c).reshape(-1, 3), np.array(spheres_r))
    segments = Segments(np.array(seg_a).reshape(-1, 3), np.array(seg_b).reshape(-1, 3),
                        np.array(seg_r), round_caps=True)  # fmt: skip
    return PrimitiveUnion(window, spheres=spheres, segments=segments, metadata=meta)


def settle(micro, phase: int = 1, *, folder, blender: str | None = None,
           python: str | None = None, height: float | None = None, **options):  # fmt: skip
    """The particles of a phase settled under gravity by Blender: :func:`write_blender`
    (``options``), :func:`run_blender`, :func:`read_blender`."""
    scene = write_blender(folder, micro, phase, **options)
    run_blender(scene, blender=blender, python=python)
    return read_blender(scene, height=height)


@dataclass
class MicrostructureScene:
    """The folder of the scene of a microstructure for Blender: the names of its objects
    (one per phase), where its surfaces come from (``route``: the exact route of the
    conforming mesh, ``grid``, or ``marching cubes``) and its fluid (None, gas, liquid)."""

    folder: Path
    objects: list
    route: str
    fluid: str | None

    @property
    def script(self) -> Path:
        return self.folder / SCENE_SCRIPT


def _boundary(nodes: np.ndarray, tets: np.ndarray):
    """The boundary of a set of tetrahedra: the faces that one of them has alone, turned
    away from the opposite corner (outward normals), on the nodes they use."""
    from morphogen.mesh.surface import SurfaceMesh

    tets = np.asarray(tets, dtype=np.int64)[:, :4]
    faces = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 3, 2]], tets[:, [0, 1, 3]],
                            tets[:, [0, 2, 1]]])  # fmt: skip
    opposite = np.concatenate([tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]])
    p = nodes[faces]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = np.einsum("ij,ij->i", normal, nodes[opposite] - p.mean(axis=1)) > 0.0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    _, inverse, count = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True,
                                  return_counts=True)  # fmt: skip
    faces = faces[count[inverse.ravel()] == 1]
    used, local = np.unique(faces, return_inverse=True)
    return SurfaceMesh(np.asarray(nodes[used], dtype=float), local.reshape(-1, 3))


def phase_surfaces(micro, phases=None, *, size: float | None = None, shape=64,
                   surfaces: str = "mesh") -> tuple[dict, str]:  # fmt: skip
    """The closed surface of each phase of a realization (outward normals, closed on the
    faces of its box), ``{phase: SurfaceMesh}``, and where they come from. ``surfaces``
    "mesh": the boundary of the tetrahedra of the phase in the mesh conforming to the
    interfaces (:func:`~morphogen.mesh.conforming_mesh`, elements of ``size``, by default
    a tenth of the shortest edge of the box): from the exact geometry when the structure
    has an exact route (the route, ``level_set``, ``cells`` or ``pieces``), else from its
    implicit functions on a grid (``grid``); without MMG, or with "implicit", the marching
    cubes of the implicit function of each phase on ``shape`` voxels (``marching cubes``).
    The phases are by default all those of the realization but 0 (the pores or the
    matrix). The conforming mesh follows the sharp edges of the exact geometry: minutes
    for many overlapping spheres, where the marching cubes take a fraction of a
    second."""
    from morphogen.mesh import conforming_mesh, exact_route
    from morphogen.mesh.surface import surface_mesh

    if surfaces not in ("mesh", "implicit"):
        raise ValueError("surfaces: 'mesh' (a conforming mesh) or 'implicit' (marching cubes)")
    present = [int(p) for p in micro.implicit_phases()]
    chosen = [p for p in present if p != 0] if phases is None else [int(p) for p in phases]
    if not chosen or any(p not in present for p in chosen):
        raise ValueError(f"phases: some of the phases {present} of the realization")
    if surfaces == "mesh":
        lengths = np.asarray(micro.domain.lengths, dtype=float)
        size = float(lengths.min()) / 10.0 if size is None else float(size)
        if not size > 0:
            raise ValueError("size: a positive length (the elements of the conforming mesh)")
        keep = "all"
        if present == [0, 1] and chosen == [1]:
            keep = "solid"
        elif present == [0, 1] and chosen == [0]:
            keep = "matrix"
        try:
            mesh = conforming_mesh(micro, size=size, order=1, phases=keep, periodic=False)
        except ImportError:  # no MMG: the marching cubes of the implicit functions
            mesh = None
        if mesh is not None:
            out = {}
            for p in chosen:
                inside = np.asarray(mesh.phase) == p
                if not inside.any():
                    raise ValueError(f"phases: the phase {p} is not in the conforming mesh")
                out[p] = _boundary(np.asarray(mesh.nodes, float), mesh.elements[inside])
            return out, exact_route(micro) or "grid"
    out = {p: surface_mesh(micro.phase_level_set(p, shape), closed=True) for p in chosen}
    return out, "marching cubes"


def _fluid_boxes(lower, upper, axis: int, buffer: float) -> dict:
    """The boxes of the fluid around a sample: the domain (extended upstream and downstream
    along ``axis`` by ``buffer`` times its length), the inflow and the outflow (slabs in
    those extensions, inset from the faces of the domain)."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    extra = float(buffer) * float(upper[axis] - lower[axis])
    dom_lo, dom_hi = lower.copy(), upper.copy()
    dom_lo[axis] -= extra
    dom_hi[axis] += extra
    inset = 0.02 * (dom_hi - dom_lo)
    in_lo, in_hi = dom_lo + inset, dom_hi - inset
    out_lo, out_hi = in_lo.copy(), in_hi.copy()
    in_hi[axis] = dom_lo[axis] + 0.5 * extra
    in_lo[axis] = dom_lo[axis] + 0.1 * extra
    out_lo[axis] = dom_hi[axis] - 0.5 * extra
    out_hi[axis] = dom_hi[axis] - 0.1 * extra
    return {"lower": dom_lo.tolist(), "upper": dom_hi.tolist(),
            "inflow": [in_lo.tolist(), in_hi.tolist()],
            "outflow": [out_lo.tolist(), out_hi.tolist()]}  # fmt: skip


def write_blender_scene(
    folder,
    micro,
    phases=None,
    *,
    size: float | None = None,
    shape=64,
    surfaces: str = "mesh",
    fluid: str | None = None,
    axis: int = 0,
    resolution: int = 64,
    velocity: float = 1.0,
    buffer: float = 0.25,
    frames: int = 100,
    viscosity: float | None = None,
    bake: bool = False,
    scene_size: float = 2.0,
    save_blend: bool = True,
) -> MicrostructureScene:
    """A folder for Blender (see the module): the closed surface of each phase of a
    realization (:func:`phase_surfaces`: ``phases``, ``size``, ``shape``, ``surfaces``) as
    ``phase_<k>.stl`` (binary, in the units of the domain), ``scene.json`` and the script
    that builds the scene. ``fluid`` "gas" (smoke carried by the flow) or "liquid" sets up
    Mantaflow: its domain is the box of the sample extended by ``buffer`` times its length
    upstream and downstream along ``axis`` (0, 1, 2: x, y, z), its walls the four faces
    along the axis and its two ends open, the phases its obstacles, an inflow of
    ``velocity`` (units of the domain per second) along the axis in the upstream
    extension and an outflow in the downstream one, ``resolution`` cells along its
    longest edge, ``frames`` frames, no gravity (the flow driven by the inflow), the
    kinematic ``viscosity`` of a liquid (m2/s, as Blender takes it), its cache in
    ``cache`` (``.uni`` files, which :func:`read_blender_flow` reads); ``bake`` bakes it
    when the script runs. The scene is scaled so that its longest edge is ``scene_size``
    units of Blender."""
    from morphogen.io.meshes import write_stl

    if fluid is not None and fluid not in FLUIDS:
        raise ValueError(f"fluid: None or one of {', '.join(FLUIDS)}")
    if axis not in (0, 1, 2):
        raise ValueError("axis: 0, 1 or 2 (x, y or z), that of the flow")
    if not int(resolution) >= 8 or not int(frames) >= 1:
        raise ValueError("resolution: 8 cells at least; frames: one at least")
    if not float(velocity) > 0 or not float(buffer) > 0:
        raise ValueError("velocity and buffer: positive")
    if viscosity is not None and not float(viscosity) > 0:
        raise ValueError("viscosity: positive (m2/s), that of a liquid")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    domain: Domain = micro.domain
    found, route = phase_surfaces(micro, phases, size=size, shape=shape, surfaces=surfaces)
    objects = []
    for k, (p, surface) in enumerate(sorted(found.items())):
        name = f"phase_{p}"
        write_stl(folder / f"{name}.stl", surface, name=name)
        colour = PHASE_COLOURS[(p - 1) % len(PHASE_COLOURS)] if p > 0 else PHASE_COLOURS[k]
        objects.append({"name": name, "phase": p, "file": f"{name}.stl",
                        "colour": list(colour), "volume": float(surface.volume())})  # fmt: skip
    lower, upper = np.asarray(domain.lower, float), np.asarray(domain.upper, float)
    spec_fluid = None
    extent = upper - lower
    if fluid is not None:
        spec_fluid = _fluid_boxes(lower, upper, axis, buffer)
        extent = np.asarray(spec_fluid["upper"]) - np.asarray(spec_fluid["lower"])
        direction = [0.0, 0.0, 0.0]
        direction[axis] = float(velocity)
        spec_fluid.update({
            "type": "GAS" if fluid == "gas" else "LIQUID", "axis": int(axis),
            "velocity": direction, "resolution": int(resolution), "frames": int(frames),
            "viscosity": None if viscosity is None else float(viscosity),
            "effectors": [o["name"] for o in objects], "cache": "cache", "bake": bool(bake),
        })  # fmt: skip
    spec = {
        "kind": "microstructure",
        "morphogen": _version(),
        "scale": float(scene_size) / float(extent.max()),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "route": route,
        "objects": objects,
        "fluid": spec_fluid,
        "save_blend": bool(save_blend) or bool(bake),
    }
    (folder / SCENE).write_text(json.dumps(spec, indent=1), encoding="utf-8")
    (folder / SCENE_SCRIPT).write_text(BLENDER_SCENE_SCRIPT, encoding="utf-8")
    return MicrostructureScene(folder, [o["name"] for o in objects], route, fluid)


def read_uni(path) -> np.ndarray:
    """A grid of Mantaflow's ``.uni`` files (gzip): the header ``MNT3`` (its sizes, the
    type and the size of its elements, 252 bytes of build information, a fourth dimension
    and a time stamp), then the values, x fastest; ``(nx, ny, nz)`` for scalars, ``(nx,
    ny, nz, 3)`` for vectors (float32)."""
    raw = Path(path).read_bytes()
    data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    if data[:4] not in (b"MNT3", b"MNT2"):
        raise ValueError(f"{path}: not a grid of Mantaflow (header {data[:4]!r})")
    nx, ny, nz, _, _, nbytes = struct.unpack("<6i", data[4:28])
    components = nbytes // 4
    count = nx * ny * nz * components
    offset = len(data) - 4 * count
    if offset < 28 or components not in (1, 3):
        raise ValueError(f"{path}: a grid of {nx}x{ny}x{nz} values of {nbytes} bytes expected")
    values = np.frombuffer(data, dtype="<f4", count=count, offset=offset)
    values = values.reshape(nz, ny, nx, components).transpose(2, 1, 0, 3)
    return values[..., 0].copy() if components == 1 else values.copy()


@dataclass
class BlenderFlow:
    """A frame of a flow baked by Blender's Mantaflow: its ``velocity`` (``(nx, ny, nz,
    3)``, in the units of Mantaflow's grid, along the axes of the domain) and the
    ``density`` of the smoke of a gas, on the cells of the fluid domain (``origin``, its
    lower corner, and ``spacing``, in the units of the microstructure)."""

    velocity: np.ndarray
    density: np.ndarray | None
    origin: np.ndarray
    spacing: np.ndarray
    frame: int

    def images(self) -> list:
        """Voxel images of the components of the velocity, its norm and the density (on
        the grid of the domain), for :func:`morphogen.io.vtk.write_vti`."""
        from morphogen.image import VoxelImage

        grid = {"spacing": tuple(self.spacing), "origin": tuple(self.origin)}
        out = [VoxelImage(self.velocity[..., k], name=f"velocity_{'xyz'[k]}", **grid)
               for k in range(3)]  # fmt: skip
        speed = np.sqrt((self.velocity.astype(float) ** 2).sum(axis=-1))
        out.append(VoxelImage(speed, name="speed", **grid))
        if self.density is not None:
            out.append(VoxelImage(self.density, name="density", **grid))
        return out


def read_blender_flow(scene, frame: int | None = None) -> BlenderFlow:
    """The flow of a scene baked by Blender (:func:`write_blender_scene` with a fluid):
    the velocity (and density) that Mantaflow caches at ``frame`` (by default the last
    one), on the cells of its domain, located in the units of the microstructure."""
    folder = Path(scene.folder if hasattr(scene, "folder") else scene)
    spec = json.loads((folder / SCENE).read_text(encoding="utf-8"))
    fluid = spec.get("fluid")
    if not fluid:
        raise ValueError("read_blender_flow: a scene with a fluid (write_blender_scene)")
    data = folder / fluid["cache"] / "data"
    frames = sorted(int(p.stem.rsplit("_", 1)[1]) for p in data.glob("vel_*.uni"))
    if not frames:
        raise FileNotFoundError(f"no velocity baked in {data}: bake the scene in Blender")
    frame = frames[-1] if frame is None else int(frame)
    if frame not in frames:
        raise ValueError(f"frame: one of those baked, {frames[0]} to {frames[-1]}")
    velocity = read_uni(data / f"vel_{frame:04d}.uni")
    density_file = data / f"density_{frame:04d}.uni"
    density = read_uni(density_file) if density_file.is_file() else None
    lower, upper = np.asarray(fluid["lower"], float), np.asarray(fluid["upper"], float)
    spacing = (upper - lower) / np.asarray(velocity.shape[:3], dtype=float)
    return BlenderFlow(velocity, density, lower, spacing, frame)


#: The script that Blender runs (or a Python with the module bpy): it reads scene.json
#: beside it, builds the scene, runs the rigid bodies, writes transforms.json and the scene
BLENDER_SCRIPT = r'''"""The scene of morphoGen in Blender: rigid bodies settled under gravity.

Run: blender --background --python morphogen_blender.py -- <folder>
or:  python morphogen_blender.py <folder>   (a Python with the module bpy)
It reads scene.json in <folder> (by default the folder of this script), builds the open
box of passive walls and the particles (spheres, capsules, convex hulls of surfaces) as
active rigid bodies, runs the frames, and writes transforms.json (the place of each
particle: location and quaternion w, x, y, z) and scene.blend.
"""

import json
import math
import sys
from pathlib import Path

import bpy


def ico(radius, level=2):
    t = (1.0 + 5.0**0.5) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t),
         (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    v = [tuple(c / math.sqrt(sum(x * x for x in p)) for c in p) for p in v]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(level):
        cache, split = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = [x + y for x, y in zip(v[a], v[b])]
                n = math.sqrt(sum(x * x for x in m))
                v.append(tuple(x / n for x in m))
                cache[key] = len(v) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            split += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = split
    return [tuple(radius * x for x in p) for p in v], f


def cylinder(radius, length, n=16):
    v, f = [], []
    for z in (-0.5 * length, 0.5 * length):
        for k in range(n):
            a = 2 * math.pi * k / n
            v.append((radius * math.cos(a), radius * math.sin(a), z))
    for k in range(n):
        f.append((k, (k + 1) % n, n + (k + 1) % n, n + k))
    f.append(tuple(reversed(range(n))))
    f.append(tuple(n + k for k in range(n)))
    return v, f


def turn_z_to(axis):
    """The quaternion (w, x, y, z) turning z onto the unit vector of axis."""
    n = math.sqrt(sum(a * a for a in axis))
    x, y, z = (a / n for a in axis)
    c = z  # cos of the angle between z and the axis
    if c < -1.0 + 1e-12:
        return (0.0, 1.0, 0.0, 0.0)
    w = math.sqrt(0.5 * (1.0 + c))
    s = 0.5 / w
    return (w, -y * s, x * s, 0.0)


def body(name, vertices, faces, location, quaternion=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(p) for p in vertices], [], [tuple(q) for q in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    if quaternion is not None:
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = quaternion
    bpy.context.scene.collection.objects.link(obj)
    return obj


def rigid(objects, kind):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.rigidbody.objects_add(type=kind)


def main(folder):
    spec = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    s = float(spec["scale"])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    frames = int(spec["frames"])
    scene.frame_start, scene.frame_end = 1, frames
    scene.use_gravity = True
    scene.gravity = (0.0, 0.0, -float(spec["gravity"]) * s)
    bpy.ops.rigidbody.world_add()
    world = scene.rigidbody_world
    world.substeps_per_frame = int(spec["substeps"])
    world.solver_iterations = int(spec["iterations"])
    world.point_cache.frame_start, world.point_cache.frame_end = 1, frames
    lo = [s * x for x in spec["lower"]]
    hi = [s * x for x in spec["upper"]]
    size = [b - a for a, b in zip(lo, hi)]
    t = 0.05 * max(size[0], size[1])  # the thickness of the walls, outside the box
    walls = []
    slabs = [((lo[0] - t, lo[1] - t, lo[2] - t), (hi[0] + t, hi[1] + t, lo[2])),
             ((lo[0] - t, lo[1] - t, lo[2]), (lo[0], hi[1] + t, hi[2])),
             ((hi[0], lo[1] - t, lo[2]), (hi[0] + t, hi[1] + t, hi[2])),
             ((lo[0], lo[1] - t, lo[2]), (hi[0], lo[1], hi[2])),
             ((lo[0], hi[1], lo[2]), (hi[0], hi[1] + t, hi[2]))]
    for k, (a, b) in enumerate(slabs):
        c = [(x + y) / 2 for x, y in zip(a, b)]
        h = [(y - x) / 2 for x, y in zip(a, b)]
        signs = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        v = [(sx * h[0], sy * h[1], sz * h[2]) for sx, sy, sz in signs]
        f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        walls.append(body(f"wall_{k}", v, f, c))
    rigid(walls, "PASSIVE")
    for w in walls:
        w.rigid_body.collision_shape = "BOX"
        w.rigid_body.friction = float(spec["friction"])
        w.rigid_body.restitution = float(spec["restitution"])
    particles, shapes, volumes = [], [], []
    for k, p in enumerate(spec["particles"]):
        centre = [s * x for x in p["centre"]]
        if p["kind"] == "sphere":
            r = s * p["radius"]
            v, f = ico(r)
            particles.append(body(f"particle_{k}", v, f, centre))
            shapes.append("SPHERE")
            volumes.append(4.0 / 3.0 * math.pi * r**3)
        elif p["kind"] == "capsule":
            r = s * p["radius"]
            length = s * math.sqrt(sum(a * a for a in p["axis"]))
            v, f = cylinder(r, length + 2.0 * r)
            q = turn_z_to(p["axis"]) if length > 0 else (1.0, 0.0, 0.0, 0.0)
            particles.append(body(f"particle_{k}", v, f, centre, q))
            shapes.append("CAPSULE")
            volumes.append(math.pi * r * r * length + 4.0 / 3.0 * math.pi * r**3)
        else:
            v = [[s * x for x in q] for q in p["vertices"]]
            particles.append(body(f"particle_{k}", v, p["faces"], centre))
            shapes.append(spec["collision"])
            volumes.append(abs(p["volume"]) * s**3)
    if particles:
        rigid(particles, "ACTIVE")
    density = float(spec["density"]) / s**3
    for obj, shape, volume in zip(particles, shapes, volumes):
        rb = obj.rigid_body
        rb.collision_shape = shape
        rb.mass = max(density * volume, 1e-9)
        rb.friction = float(spec["friction"])
        rb.restitution = float(spec["restitution"])
        rb.use_margin = True
        rb.collision_margin = 0.0 if shape in ("SPHERE", "CAPSULE") else 0.001 * max(size)
    for frame in range(1, frames + 1):
        scene.frame_set(frame)
    out = []
    for obj in particles:
        loc, rot, _ = obj.matrix_world.decompose()
        out.append({"name": obj.name, "location": list(loc), "quaternion": list(rot)})
    (folder / "transforms.json").write_text(json.dumps({"particles": out}), encoding="utf-8")
    if spec.get("save_blend", True):
        bpy.ops.wm.save_as_mainfile(filepath=str(folder / "scene.blend"))


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(Path(argv[0]) if argv else Path(__file__).resolve().parent)
'''


#: The script of the scene of a microstructure that Blender runs (or a Python with the
#: module bpy): it reads scene.json beside it, makes the objects of the phases and the fluid
#: of Mantaflow, saves scene.blend, bakes if asked and writes report.json
BLENDER_SCENE_SCRIPT = r'''"""A microstructure of morphoGen in Blender, and a fluid around it.

Run: blender --background --python morphogen_blender_scene.py -- <folder>
or:  python morphogen_blender_scene.py <folder>   (a Python with the module bpy)
It reads scene.json in <folder> (by default the folder of this script): each phase
(phase_<k>.stl, binary, in the units of the microstructure) becomes a mesh object with its
colour; with a fluid, Mantaflow's domain (the box extended along the axis of the flow),
the phases as obstacles (effectors of collision), an inflow upstream and an outflow
downstream. It saves scene.blend, bakes the fluid if asked (its cache in <folder>/cache)
and writes report.json (the objects, their volumes and the settings of the fluid).
"""

import json
import struct
import sys
import time
from pathlib import Path

import bpy


def read_stl(path):
    """The vertices (merged) and triangles of a binary STL file."""
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    index, vertices, faces = {}, [], []
    for k in range(count):
        values = struct.unpack_from("<12f", data, 84 + 50 * k)
        face = []
        for j in range(3):
            p = values[3 + 3 * j:6 + 3 * j]
            if p not in index:
                index[p] = len(vertices)
                vertices.append(p)
            face.append(index[p])
        faces.append(face)
    return vertices, faces


def mesh_object(name, vertices, faces, colour=None, wire=False):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(p) for p in vertices], [], [tuple(f) for f in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if colour is not None:
        material = bpy.data.materials.new(name)
        material.diffuse_color = colour
        tree = getattr(material, "node_tree", None)
        shader = tree.nodes.get("Principled BSDF") if tree is not None else None
        if shader is not None:
            shader.inputs["Base Color"].default_value = colour
        mesh.materials.append(material)
    if wire:
        obj.display_type = "WIRE"
    return obj


def box(name, lower, upper):
    v = [(x, y, z) for x in (lower[0], upper[0]) for y in (lower[1], upper[1])
         for z in (lower[2], upper[2])]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return mesh_object(name, v, f, wire=True)


def volume(obj):
    """The volume a closed mesh encloses (its triangulated faces and the origin)."""
    mesh, total = obj.data, 0.0
    co = [obj.matrix_world @ v.co for v in mesh.vertices]
    for poly in mesh.polygons:
        ids = list(poly.vertices)
        for k in range(1, len(ids) - 1):
            a, b, c = co[ids[0]], co[ids[k]], co[ids[k + 1]]
            total += a.dot(b.cross(c)) / 6.0
    return total


def fluid_modifier(obj, kind):
    mod = obj.modifiers.new("Fluid", "FLUID")
    mod.fluid_type = kind
    return mod


def main(folder):
    spec = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    s = float(spec["scale"])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    objects = {}
    for item in spec["objects"]:
        vertices, faces = read_stl(folder / item["file"])
        vertices = [tuple(s * x for x in p) for p in vertices]
        objects[item["name"]] = mesh_object(item["name"], vertices, faces, item["colour"])
    report = {"objects": {}, "fluid": None, "bake": None}
    fluid = spec.get("fluid")
    if fluid:
        frames = int(fluid["frames"])
        scene.frame_start, scene.frame_end = 1, frames
        scene.gravity = (0.0, 0.0, 0.0)  # the flow driven by the inflow alone
        lower = [s * x for x in fluid["lower"]]
        upper = [s * x for x in fluid["upper"]]
        domain = box("fluid_domain", lower, upper)
        settings = fluid_modifier(domain, "DOMAIN").domain_settings
        settings.domain_type = fluid["type"]
        settings.resolution_max = int(fluid["resolution"])
        settings.cache_directory = "//" + fluid["cache"]
        settings.cache_type = "ALL"
        settings.cache_data_format = "UNI"
        settings.cache_frame_start, settings.cache_frame_end = 1, frames
        if hasattr(settings, "gravity"):
            try:
                settings.gravity = (0.0, 0.0, 0.0)
            except (AttributeError, TypeError):
                pass
        sides = {0: ("left", "right"), 1: ("front", "back"), 2: ("bottom", "top")}
        axis = int(fluid["axis"])
        for k, pair in sides.items():
            for side in pair:  # walls along the flow, its two ends open
                setattr(settings, f"use_collision_border_{side}", k != axis)
        if fluid["type"] == "GAS":
            settings.alpha = 0.0  # no buoyancy: the smoke goes with the flow
            settings.beta = 0.0
            settings.vorticity = 0.0
        elif fluid.get("viscosity"):
            nu = float(fluid["viscosity"])
            exponent = 0
            while nu * 10.0**exponent < 1.0:
                exponent += 1
            settings.viscosity_base = nu * 10.0**exponent
            settings.viscosity_exponent = exponent
        for name in fluid["effectors"]:
            effector = fluid_modifier(objects[name], "EFFECTOR").effector_settings
            effector.effector_type = "COLLISION"
        velocity = tuple(s * v for v in fluid["velocity"])
        source = "SMOKE" if fluid["type"] == "GAS" else "LIQUID"
        for name, behaviour in (("inflow", "INFLOW"), ("outflow", "OUTFLOW")):
            lo, hi = ([s * x for x in corner] for corner in fluid[name])
            flow = fluid_modifier(box(name, lo, hi), "FLOW").flow_settings
            flow.flow_type = source
            flow.flow_behavior = behaviour
            flow.flow_source = "MESH"
            if behaviour == "INFLOW":
                flow.use_initial_velocity = True
                flow.velocity_coord = velocity
        report["fluid"] = {
            "domain_type": settings.domain_type,
            "resolution_max": settings.resolution_max,
            "borders": {side: getattr(settings, f"use_collision_border_{side}")
                        for pair in sides.values() for side in pair},
            "cache_directory": settings.cache_directory,
        }
    for name, obj in bpy.data.objects.items():
        mod = obj.modifiers.get("Fluid")
        entry = {"vertices": len(obj.data.vertices), "faces": len(obj.data.polygons),
                 "volume": volume(obj) / s**3, "fluid": mod.fluid_type if mod else None}
        if mod is not None and mod.fluid_type == "EFFECTOR":
            entry["effector"] = mod.effector_settings.effector_type
        if mod is not None and mod.fluid_type == "FLOW":
            entry["flow"] = mod.flow_settings.flow_behavior
            entry["velocity"] = [v / s for v in mod.flow_settings.velocity_coord]
        report["objects"][name] = entry
    if spec.get("save_blend", True):
        bpy.ops.wm.save_as_mainfile(filepath=str(folder / "scene.blend"))
    if fluid and fluid.get("bake"):
        start = time.time()
        with bpy.context.temp_override(scene=scene, active_object=domain, object=domain):
            result = bpy.ops.fluid.bake_all()
        cache = folder / fluid["cache"] / "data"
        report["bake"] = {"result": sorted(result), "seconds": time.time() - start,
                          "velocity_frames": len(list(cache.glob("vel_*.uni")))}
        bpy.ops.wm.save_mainfile()
    (folder / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(Path(argv[0]).resolve() if argv else Path(__file__).resolve().parent)
'''
