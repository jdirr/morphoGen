"""One entry point for every export: :func:`export` writes an object to the format of the
suffix of its path, choosing the writer from the object and the suffix.

=====================  ====================================================================
object                 suffixes
=====================  ====================================================================
voxel image            .vti .vtk .xdmf .mhd .npz .npy .tif (data); .png .svg .pdf .tex
                       (sections); .csv .txt .xlsx (voxels); .mp4 .gif (``kind``:
                       "sections" along ``axis``, "orthoslices", "turntable")
mesh (tetrahedra,      .vtu .inp .msh .mesh .xdmf .stl and meshio's (data); .png (a view);
voxels, surface)       .csv .txt .xlsx (nodes and elements); .mp4 .gif (``kind``:
                       "turntable", "sweep" along ``axis``, "field" with ``displacement``;
                       ``field`` colours them)
microstructure         .json .csv .geo; the others voxelized (``shape``, by default 64,
                       or ``voxel_size``)
particles of a phase   with ``particles=k`` (a microstructure or a voxel image): the phase
(DEM)                  ``k`` without its matrix (:mod:`morphogen.io.dem`), .stl .obj .vtu
                       .msh (a closed surface per particle, whole across periodic faces;
                       Gmsh 2.2 for LMGC90, its tetrahedra with ``elements="tetrahedra"``),
                       .csv .txt .xlsx (the particles, or the exact spheres), .data
                       (LAMMPS, LIGGGHTS), ``format="yade"``
band structure         .csv .txt .xlsx (the bands); figure suffixes (the plot)
bifurcation analysis   .csv .txt .xlsx (irreps and patterns); .vtu (patterns of a mesh);
                       figure suffixes with ``frame`` (patterns of a frame)
loading path result    .csv .txt .xlsx (its curve)
frame of beams         .mp4 .gif with ``pattern`` (its dofs: a pattern oscillating), in
                       the plane or in space
matplotlib figure      .png .tif .jpg .svg .pdf .eps .tex
frames (RGB arrays)    .mp4 .gif or a folder
table                  .csv .txt .tsv .xlsx (records, columns, an array)
=====================  ====================================================================

Videos take ``fps``, ``mode`` ("once", "loop", "bounce") and ``repeat``; 3D views ``cmap``,
``clim``, ``size``; field animations ``magnification``, ``wave`` and ``section``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from morphogen.image import VoxelImage
from morphogen.io.tables import FORMATS as TABLES
from morphogen.io.tables import write_table, write_tables
from morphogen.viz.export import BITMAP, TIKZ, VECTOR, save_figure

FIGURES = BITMAP + VECTOR + TIKZ
VIDEOS = (".mp4", ".gif")
_VIDEO_OPTIONS = ("fps", "mode", "repeat", "crf")


def _split(options: dict) -> tuple[dict, dict]:
    video = {k: options.pop(k) for k in _VIDEO_OPTIONS if k in options}
    return video, options


def _video(obj, path: Path, options: dict) -> Path:
    from morphogen.homogenization.frames import Frame, SpaceFrame
    from morphogen.viz import video as vv

    video, options = _split(dict(options))
    kind = options.pop("kind", None)
    if isinstance(obj, VoxelImage):
        kind = kind or "sections"
        if kind == "sections":
            frames = vv.section_frames(obj, **options)
        elif kind == "orthoslices":
            frames = vv.orthoslice_frames(obj, **options)
        elif kind == "turntable":
            frames = vv.turntable_frames(obj, **options)
        else:
            raise ValueError(
                "export: kind of a voxel image video: sections, orthoslices, turntable"
            )
    elif isinstance(obj, (Frame, SpaceFrame)):
        frames = vv.frame_pattern_frames(obj, options.pop("pattern"), **options)
    elif hasattr(obj, "elements") and hasattr(obj, "nodes"):
        kind = kind or ("field" if "displacement" in options else "turntable")
        if kind == "turntable":
            frames = vv.turntable_frames(obj, **options)
        elif kind == "sweep":
            frames = vv.section_sweep_frames(obj, options.pop("field", None), **options)
        elif kind == "field":
            frames = vv.field_frames(
                obj, options.pop("displacement"), options.pop("field", None), **options
            )
        else:
            raise ValueError("export: kind of a mesh video: turntable, sweep, field")
    else:
        frames = list(obj)
    return vv.write_video(frames, path, **video)


def _mesh_tables(mesh) -> dict:
    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.elements if hasattr(mesh, "elements") else mesh.faces)
    node_table = {
        "node": np.arange(len(nodes)),
        "x": nodes[:, 0],
        "y": nodes[:, 1],
        "z": nodes[:, 2],
    }
    element_table = {"element": np.arange(len(elements))}
    element_table.update({f"n{k + 1}": elements[:, k] for k in range(elements.shape[1])})
    phase = getattr(mesh, "phase", None)
    if phase is not None:
        element_table["phase"] = np.asarray(phase)
    return {"nodes": node_table, "elements": element_table}


def _voxel_table(image: VoxelImage) -> dict:
    data = np.asarray(image.data)
    idx = np.indices(data.shape).reshape(3, -1)
    x = np.asarray(image.origin)[:, None] + (idx + 0.5) * np.asarray(image.spacing)[:, None]
    return {
        "i": idx[0],
        "j": idx[1],
        "k": idx[2],
        "x": x[0],
        "y": x[1],
        "z": x[2],
        "value": data.ravel(),
    }


def export(obj, path, **options) -> Path | list[Path]:
    """Write ``obj`` to ``path`` in the format of its suffix (see the module); the options
    go to the writer."""
    from morphogen.homogenization.bifurcation import Bifurcation
    from morphogen.homogenization.bloch import BandStructure
    from morphogen.homogenization.paths import PathResult
    from morphogen.io import save_image, save_mesh, save_microstructure
    from morphogen.models.base import Microstructure

    path = Path(path)
    suffix = path.suffix.lower()
    if options.get("particles") not in (None, False):  # one phase without its matrix (DEM)
        from morphogen.io.dem import write_particles

        phase = options.pop("particles")
        return write_particles(path, obj, 1 if phase is True else int(phase), **options)
    options.pop("particles", None)
    if isinstance(obj, Microstructure) and suffix not in (".json", ".csv", ".geo"):
        shape, size = options.pop("shape", None), options.pop("voxel_size", None)
        shape = 64 if shape is None and size is None else shape
        return export(obj.voxelize(shape, voxel_size=size), path, **options)
    if suffix in VIDEOS or (suffix == "" and options.get("kind")):
        return _video(obj, path, options)
    if hasattr(obj, "savefig"):
        return save_figure(obj, path, **options)
    if isinstance(obj, PathResult):
        return write_table(path, obj.rows(), **options)
    if isinstance(obj, BandStructure):
        if suffix in TABLES:
            table = {
                "distance": obj.distance,
                "kx": obj.wavevectors[:, 0],
                "ky": obj.wavevectors[:, 1],
            }
            table["kz"] = obj.wavevectors[:, 2]
            table.update(
                {f"band_{n + 1}": obj.frequencies[:, n] for n in range(obj.frequencies.shape[1])}
            )
            return write_table(path, table, **options)
        return save_figure(obj.plot().figure, path, **options)
    if isinstance(obj, Bifurcation):
        if suffix in TABLES:
            rows = []
            for row in obj.summary():
                base = {k: row[k] for k in ("irrep", "dim", "multiplicity", "load")}
                if row["patterns"]:
                    rows.extend({**base, **p} for p in row["patterns"])
                else:
                    rows.append(base)
            return write_table(path, rows, **options)
        if suffix == ".vtu":
            return obj.write(path, **options)
        return save_figure(obj.figure(options.pop("frame"), **options), path)
    if isinstance(obj, Microstructure):
        return save_microstructure(obj, path, **options)
    if isinstance(obj, VoxelImage) or (
        isinstance(obj, (list, tuple)) and obj and all(isinstance(o, VoxelImage) for o in obj)
    ):
        image = obj if isinstance(obj, VoxelImage) else obj[0]
        if suffix in TABLES:
            return write_table(path, _voxel_table(image), **options)
        if suffix in FIGURES and suffix not in (".tif", ".tiff"):
            from morphogen.viz.figures import slices_figure

            return save_figure(slices_figure(image, **options), path)
        return save_image(obj, path, **options)
    if hasattr(obj, "nodes") and (hasattr(obj, "elements") or hasattr(obj, "faces")):
        if suffix in TABLES:
            return write_tables(path, _mesh_tables(obj))
        if suffix in BITMAP and hasattr(obj, "elements"):
            from PIL import Image

            from morphogen.viz.video import turntable_frames

            frame = turntable_frames(obj, frames=1, **options)[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(path)
            return path
        return save_mesh(obj, path, **options)
    if isinstance(obj, (dict, list, tuple, np.ndarray)):
        return write_table(path, obj, **options)
    raise TypeError(f"export: no writer for {type(obj).__name__} to {suffix or 'a folder'}")
