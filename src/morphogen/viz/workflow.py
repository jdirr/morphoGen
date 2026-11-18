"""Figures and videos of the realizations of a run (the ``visualization`` section of the
configurations): the chain from the morphological model to its voxel image, its
conforming mesh and the fields of its homogenization.

A view shows one object of each realization (``of``):

* ``image``: the voxel image; figures: its three middle sections (``kind: sections``) or
  the surface of a ``phase`` (``surface``, VTK); videos: the three sections moving
  together (``orthoslices``, the default, as in Avizo), a section sweeping along ``axis``
  (``sections``) or the surface of a phase turning (``surface``);
* ``labels``: the cell labels of tessellations and grain structures (sections,
  orthoslices);
* ``mesh``: the conforming mesh coloured by phase (``color: none``: uniform); figures: a
  view; videos: a turntable (``turntable``) or a cross-section sweeping along ``axis``
  (``sweep``);
* a result of the homogenization, ``<property>.<bc>`` (bulk_modulus, shear_modulus,
  conductivity), ``<loading>.<bc>`` or ``<path>.<bc>``: the field ``color`` (by default the
  von Mises stress, or the norm of the heat flux) on the mesh deformed by its displacement
  magnified ``magnification`` times (by default the largest displacement drawn as a tenth
  of the mesh, and the true deformation along loading paths in finite strains; ``part:
  fluctuation`` removes the macroscopic part ``E x``), cut by the plane ``section: (axis,
  fraction)``; on the voxels of the FFT solver, the sections of the field. Videos of
  properties and loadings grow the load (``wave: ramp``) or make it oscillate (``wave:
  sine``); those of loading paths follow their steps from the natural state (``frames``:
  interpolated between the steps). Figures of loading paths draw their curves (``kind:
  curves``, the default) or their field at the end (``kind: field``).

The fields are named as in the files of the fields: ``von_mises``, ``energy_density``,
``stress_11`` ... ``stress_12`` and ``strain_11`` ... (Voigt order 11, 22, 33, 23, 31, 12),
``displacement`` (its norm); ``flux``, ``gradient`` (norms), ``flux_1`` ...,
``dissipation``, ``temperature``; along paths ``p`` (cumulated plastic strain) and
``plastic_strain_11`` ...

Files: ``<stem>_<name>.<format>``, with the name given (``name``) or made of the object,
the kind and the field: ``spheres_0000_image_orthoslices.mp4``,
``spheres_0000_shear_modulus_periodic_von_mises.png``.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("morphogen")

COMPONENTS = ("11", "22", "33", "23", "31", "12")
_TENSORS = tuple(f"{n}_{c}" for n in ("stress", "strain") for c in COMPONENTS)
#: The fields colouring the views of the results, by kind of result
FIELDS = {
    "elastic": ("von_mises", "energy_density", "displacement", *_TENSORS),
    "thermal": (
        "flux",
        "gradient",
        "dissipation",
        "temperature",
        *(f"flux_{d}" for d in "123"),
        *(f"gradient_{d}" for d in "123"),
    ),
    "path": (
        "von_mises",
        "p",
        "displacement",
        *_TENSORS,
        *(f"plastic_strain_{c}" for c in COMPONENTS),
        # of the crystals that twin or cleave
        "twin_fraction",
        "twin_reoriented",
        "twin_misorientation",
        "cleavage_opening",
        "cleavage_damage",
        "cleavage_microdamage",
        # of the cells whose interfaces debond, and of the fatigue of a cycle
        "interface_damage",
        "interface_opening",
        "fatigue_utilization",
        "fatigue_equivalent",
        "fatigue_amplitude",
        "fatigue_hydrostatic_max",
        # of the criterion of cleavage of Beremin
        "beremin_stress",
    ),
}
DEFAULT_FIELD = {"elastic": "von_mises", "thermal": "flux", "path": "von_mises"}
#: The fields of the nodes of finite element meshes (the displacement of the loading
#: paths of the FFT solver is at the centres of the voxels)
NODAL = ("displacement", "temperature")
#: The kinds of the figures and of the videos of each object, the first ones the defaults
KINDS = {
    "image": (("sections", "surface"), ("orthoslices", "sections", "surface")),
    "labels": (("sections",), ("orthoslices", "sections")),
    "mesh": (("view",), ("turntable", "sweep")),
    "result": (("field",), ("field",)),
    "path": (("curves", "field"), ("field",)),
}
_VECTORS = ("displacement", "flux", "gradient")


def von_mises(stress) -> np.ndarray:
    """Von Mises stress of stresses in the Voigt order (last axis; tensor shears)."""
    s = np.asarray(stress, dtype=float)
    mean = s[..., :3].sum(axis=-1) / 3.0
    dev2 = np.sum((s[..., :3] - mean[..., None]) ** 2, axis=-1)
    dev2 += 2.0 * np.sum(s[..., 3:] ** 2, axis=-1)
    return np.sqrt(1.5 * dev2)


def element_arrays(fields: dict, physics: str) -> dict[str, np.ndarray]:
    """The element fields of a solution (``Solution.fields``) by name: energy density, von
    Mises stress, stress and strain components; or dissipation, flux and gradient
    components."""
    out: dict[str, np.ndarray] = {}
    if physics == "elastic":
        eps, sig = fields["strain"], fields["stress"]
        out["energy_density"] = fields["energy"]
        out["von_mises"] = von_mises(sig)
        for i, name in enumerate(COMPONENTS):
            out[f"stress_{name}"] = sig[..., i]
        for i, name in enumerate(COMPONENTS):
            out[f"strain_{name}"] = eps[..., i]
    else:
        grad, flux = fields["gradient"], fields["flux"]
        out["dissipation"] = fields["dissipation"]
        for d in range(3):
            out[f"flux_{d + 1}"] = flux[..., d]
        for d in range(3):
            out[f"gradient_{d + 1}"] = grad[..., d]
    return out


def solution_fields(solution) -> dict[str, np.ndarray]:
    """All the named fields of a solution kept with ``fields=True``: those of
    :func:`element_arrays`, the nodal displacement or temperature of finite elements, and
    the flux and gradient vectors."""
    fields = solution.fields
    out = element_arrays(fields, solution.physics)
    if solution.physics == "elastic":
        if "displacement" in fields:
            out["displacement"] = np.asarray(fields["displacement"])
    else:
        out["flux"] = np.asarray(fields["flux"])
        out["gradient"] = np.asarray(fields["gradient"])
        if "temperature" in fields:
            out["temperature"] = np.asarray(fields["temperature"])
    return out


def path_fields(arrays: dict) -> dict[str, np.ndarray]:
    """The named fields of a loading path at one time (``PathResult.fields[t]``)."""
    out = {"von_mises": np.asarray(arrays["mises"]), "p": np.asarray(arrays["p"])}
    for name in ("stress", "strain", "plastic_strain"):
        values = np.asarray(arrays[name])
        for i, c in enumerate(COMPONENTS):
            out[f"{name}_{c}"] = values[..., i]
    if "displacement" in arrays:
        out["displacement"] = np.asarray(arrays["displacement"])
    for name in FIELDS["path"]:  # the other scalar fields that a solver or a study adds
        if name in arrays and name not in out:
            out[name] = np.asarray(arrays[name])
    return out


def field_values(named: dict, name: str) -> np.ndarray:
    """The values of the field ``name`` among the named fields (vectors by their norm)."""
    if name not in named:
        raise ValueError(f"visualization: no field {name!r} here (fields: {', '.join(named)})")
    values = np.asarray(named[name], dtype=float)
    if name in _VECTORS and values.shape[-1] == 3:
        values = np.linalg.norm(values, axis=-1)
    return values


def requirements(spec) -> set[str]:
    """What the views of a :class:`~morphogen.config.VisualizationSpec` need: ``"vtk"``
    (3D views), ``"ffmpeg"`` (.mp4), ``"matplotlib"`` (figures)."""
    out = {"matplotlib"} if spec.views else set()  # the colormaps
    if any(view.vtk for view in spec.views):
        out.add("vtk")
    if any("mp4" in v.formats for v in spec.videos):
        out.add("ffmpeg")
    return out


def check(spec) -> None:
    """Raise an error naming what the views need and is missing (before a run)."""
    from morphogen.viz.video import ffmpeg_path, vtk_can_render

    missing = []
    for need in sorted(requirements(spec)):
        if need == "ffmpeg":
            if ffmpeg_path() is None:
                missing.append(
                    "ffmpeg for .mp4 videos (on the PATH, MORPHOGEN_FFMPEG, or pip install "
                    "imageio-ffmpeg; or the format gif)"
                )
        elif importlib.util.find_spec(need) is None:
            missing.append(
                "VTK for the 3D views (pip install 'morphogen[video]')"
                if need == "vtk"
                else "matplotlib for the figures (pip install 'morphogen[plot]')"
            )
        elif need == "vtk" and not vtk_can_render():
            missing.append(
                "a display for the 3D views of VTK (X, EGL or OSMesa; on a headless machine, "
                "run under xvfb-run)"
            )
    if missing:
        raise RuntimeError("visualization: missing " + "; ".join(missing))


# -- views -------------------------------------------------------------------------------


def _camera(spec) -> dict:
    """The camera of the 3D views of a visualization section."""
    return {"elevation": spec.elevation, "azimuth": spec.azimuth}


def _save_rgb(rgb: np.ndarray, path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"compression": "tiff_lzw"} if path.suffix.lower() in (".tif", ".tiff") else {}
    Image.fromarray(np.ascontiguousarray(rgb[..., :3])).save(path, **options)
    return path


def _source(view, image, labels):
    source = labels if view.of == "labels" else image
    if source is None:
        what = "cell labels (tessellations, grain structures)" if view.of == "labels" else "image"
        raise ValueError(f"visualization: this realization has no {what}")
    return source


def _result(results, key: str):
    if not results or key not in results:
        raise ValueError(
            f"visualization: no result {key!r} (results: {', '.join(sorted(results or {}))})"
        )
    return results[key]


def _points(support) -> np.ndarray:
    """The nodes of a mesh, or the centres of the voxels of an image (nx, ny, nz, 3)."""
    from morphogen.mesh.tetra import TetMesh

    if isinstance(support, TetMesh):
        return support.nodes
    axes = [
        support.origin[d] + (np.arange(n) + 0.5) * support.spacing[d]
        for d, n in enumerate(support.data.shape)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _macro(res, time) -> np.ndarray:
    """The macroscopic strain (displacement gradient in finite strains) of a result."""
    from morphogen.homogenization.paths import PathResult

    if not isinstance(res, PathResult):
        return np.asarray(res.solution.strain, dtype=float)
    k = int(np.argmin(np.abs(np.asarray(res.times) - time)))
    if res.finite:
        from morphogen.homogenization.finite import gradient_of

        return np.asarray(gradient_of(res.strain[k]), dtype=float).reshape(3, 3)
    from morphogen.homogenization.elements import voigt_to_tensor

    return voigt_to_tensor(res.strain[k], strain=True)


def _snapshot(res, view, time=None):
    """The support, the named fields and the displacement (``None`` without one) of a
    result (of a loading path: at ``time``)."""
    from morphogen.homogenization.paths import PathResult

    if isinstance(res, PathResult):
        if time not in res.fields:
            raise ValueError(f"visualization: {view.of} kept no fields at time {time}")
        support, named = res.support, path_fields(res.fields[time])
    else:
        sol = res.solution
        if sol.fields is None:
            raise ValueError(f"visualization: {view.of} kept no fields")
        support, named = sol.support, solution_fields(sol)
    u = named.get("displacement")
    if u is not None and view.part == "fluctuation":
        u = u - _points(support) @ _macro(res, time).T
    return support, named, u


def _magnification(view, res):
    from morphogen.homogenization.paths import PathResult

    if view.magnification is None and isinstance(res, PathResult) and res.finite:
        return 1.0 if view.part == "total" else None  # the true deformation
    return view.magnification


def _figure(spec, view, path: Path, image, labels, mesh, results) -> Path:
    from morphogen.homogenization.paths import PathResult
    from morphogen.mesh.tetra import TetMesh
    from morphogen.viz import video as vv
    from morphogen.viz.export import save_figure
    from morphogen.viz.figures import field_figure, path_figure, slices_figure

    cmap = view.cmap or spec.cmap
    if view.of in ("image", "labels"):
        source = _source(view, image, labels)
        if view.kind == "surface":
            frame = vv.turntable_frames(
                source, phase=view.phase, frames=1, size=spec.size, **_camera(spec)
            )[0]
            return _save_rgb(frame, path)
        return save_figure(slices_figure(source, labels=view.of == "labels"), path, dpi=spec.dpi)
    if view.of == "mesh":
        if mesh is None:
            raise ValueError("visualization: this realization has no conforming mesh")
        field = None if view.color == "none" else "phase"
        return _save_rgb(vv.deformed_view(mesh, None, field, size=spec.size, **_camera(spec)), path)
    res = _result(results, view.of)
    title = f"{view.of}: {view.color}" if view.kind == "field" else view.of
    if isinstance(res, PathResult) and view.kind == "curves":
        return save_figure(path_figure(res, title=title), path, dpi=spec.dpi)
    time = max(res.fields) if isinstance(res, PathResult) and res.fields else None
    support, named, u = _snapshot(res, view, time)
    values = field_values(named, view.color)
    if isinstance(support, TetMesh):
        rgb = vv.deformed_view(
            support,
            u,
            values,
            magnification=_magnification(view, res),
            section=view.section,
            size=spec.size,
            **_camera(spec),
            cmap=cmap,
            clim=view.clim,
            title=view.color,
        )
        return _save_rgb(rgb, path)
    fig = field_figure(values, cmap=cmap, clim=view.clim, title=title, label=view.color)
    return save_figure(fig, path, dpi=spec.dpi)


def _natural(values: np.ndarray) -> np.ndarray:
    """The natural state of a field: zero, NaN kept (the pores of images)."""
    return np.where(np.isnan(values), np.nan, 0.0)


def _frames(spec, view, image, labels, mesh, results) -> list[np.ndarray]:
    from morphogen.homogenization.paths import PathResult
    from morphogen.mesh.tetra import TetMesh
    from morphogen.viz import video as vv

    cmap = view.cmap or spec.cmap
    if view.of in ("image", "labels"):
        source = _source(view, image, labels)
        flags = {"labels": True} if view.of == "labels" else {}
        if view.kind == "orthoslices":
            size = max(spec.size) // 2
            return vv.orthoslice_frames(
                source, frames=view.frames or 48, size=size, cmap=cmap, **flags
            )
        if view.kind == "sections":
            return vv.section_frames(source, view.axis, size=max(spec.size), cmap=cmap, **flags)
        return vv.turntable_frames(
            source, phase=view.phase, frames=view.frames or 72, size=spec.size, **_camera(spec)
        )
    if view.of == "mesh":
        if mesh is None:
            raise ValueError("visualization: this realization has no conforming mesh")
        field = None if view.color == "none" else "phase"
        if view.kind == "turntable":
            return vv.turntable_frames(
                mesh, field=field, frames=view.frames or 72, size=spec.size, **_camera(spec)
            )
        return vv.section_sweep_frames(
            mesh, field, axis=view.axis, frames=view.frames or 48, size=spec.size, **_camera(spec)
        )
    res = _result(results, view.of)
    clim = view.clim
    if isinstance(res, PathResult):
        times = sorted(res.fields)
        if not times:
            raise ValueError(f"visualization: {view.of} kept no fields")
        snaps = [_snapshot(res, view, t) for t in times]
        support = snaps[0][0]
        values = [field_values(named, view.color) for _, named, _ in snaps]
        values.insert(0, _natural(values[0]))
        captions = [f"t = {t:g}" for t in (float(res.times[0]), *times)]
        if isinstance(support, TetMesh):
            us = [np.zeros_like(support.nodes) if u is None else u for _, _, u in snaps]
            us.insert(0, np.zeros_like(support.nodes))
            return vv.history_frames(
                support,
                us,
                values,
                frames=view.frames,
                magnification=_magnification(view, res),
                section=view.section,
                size=spec.size,
                **_camera(spec),
                cmap=cmap,
                clim=clim,
                title=view.color,
                captions=captions,
            )
        lo, hi = clim or (None, None)
        return vv.time_section_frames(
            values,
            view.axis,
            size=max(spec.size),
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            captions=captions,
            label=view.color,
        )
    support, named, u = _snapshot(res, view)
    values = field_values(named, view.color)
    if isinstance(support, TetMesh):
        if u is None:  # conduction: no deformation, a cross-section sweeping through
            return vv.section_sweep_frames(
                support,
                values,
                axis=view.axis,
                frames=view.frames or 48,
                size=spec.size,
                **_camera(spec),
                cmap=cmap,
                clim=clim,
                title=view.color,
            )
        return vv.field_frames(
            support,
            u,
            values,
            magnification=view.magnification,
            frames=view.frames or 48,
            wave=view.wave,
            section=view.section,
            size=spec.size,
            **_camera(spec),
            cmap=cmap,
            clim=clim,
            title=view.color,
            scaling=vv.field_scaling(view.color),
        )
    lo, hi = clim or (None, None)
    return vv.section_frames(
        values,
        view.axis,
        size=max(spec.size),
        cmap=cmap,
        vmin=lo,
        vmax=hi,
        labels=False,
        bar=True,
        label=view.color,
    )


def visualize(spec, stem: str, outdir, *, image=None, labels=None, mesh=None, results=None):
    """Write the figures and videos of ``spec`` (a
    :class:`~morphogen.config.VisualizationSpec`) for one realization: its voxel ``image``,
    cell ``labels``, conforming ``mesh`` and the ``results`` of its homogenization (kept
    with their fields); returns the paths written."""
    from morphogen.viz.video import write_video

    outdir = Path(outdir)
    written = []
    for view in spec.figures:
        for fmt in view.formats:
            path = outdir / f"{stem}_{view.label}.{fmt}"
            written.append(_figure(spec, view, path, image, labels, mesh, results))
            log.info("figure %s", path.name)
    for view in spec.videos:
        frames = _frames(spec, view, image, labels, mesh, results)
        for fmt in view.formats:
            path = outdir / (f"{stem}_{view.label}" + ("_frames" if fmt == "frames" else f".{fmt}"))
            fps = view.fps or spec.fps
            written.append(write_video(frames, path, fps=fps, mode=view.mode, repeat=view.repeat))
            log.info("video %s (%d frames)", path.name, len(frames))
    return written
