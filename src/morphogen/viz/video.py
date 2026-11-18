"""Videos of morphologies and fields: ``.mp4`` through ffmpeg (``ffmpeg`` on the path, the
``MORPHOGEN_FFMPEG`` variable or the imageio-ffmpeg package), ``.gif`` through Pillow, or a
folder of PNG frames.

Morphologies: a section sweeping through a voxel image along an axis, the three orthogonal
sections moving together (as the orthoslices of Avizo), a turntable of the surface of a
phase or of a mesh. Fields on tetrahedral meshes: a turntable coloured by a field, a
cross-section sweeping through the mesh, and the deformed mesh with a magnified
displacement (a mode oscillating, or a load growing), optionally cut by a plane, and its
history along the steps of a loading path; fields on voxel images, the same section at
successive times. The 3D views are rendered off screen by VTK; colours come from
matplotlib colormaps, on a fixed range over the frames, with a colour bar beside them.
Every video plays once, loops (``repeat`` times) or goes back and forth
(``mode="bounce"``).
"""

from __future__ import annotations

import functools
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from morphogen.image import VoxelImage
from morphogen.viz.png import label_colors

MODES = ("once", "loop", "bounce")


def ffmpeg_path() -> str | None:
    """The ffmpeg executable: ``MORPHOGEN_FFMPEG``, the path, or imageio-ffmpeg's."""
    env = os.environ.get("MORPHOGEN_FFMPEG")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # optional
        return None


def sequence(frames, mode: str = "once", repeat: int = 1) -> list[np.ndarray]:
    """The frames played once, ``repeat`` times in a loop, or forth and back."""
    frames = [np.asarray(f) for f in frames]
    if mode not in MODES:
        raise ValueError(f"mode: one of {MODES}")
    if mode == "bounce":
        frames = frames + frames[-2:0:-1]
    return frames * (max(int(repeat), 1) if mode != "once" else 1)


def _even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return np.pad(frame, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")


def write_video(
    frames, path, *, fps: float = 24.0, mode: str = "once", repeat: int = 1, crf: int = 18
) -> Path:
    """Write RGB frames (uint8, all of one size) to ``.mp4`` (H.264 by ffmpeg, quality
    ``crf``), ``.gif`` or a folder of PNG files (a path without suffix)."""
    path = Path(path)
    frames = [
        np.ascontiguousarray(np.asarray(f, dtype=np.uint8)[..., :3])
        for f in sequence(frames, mode, repeat)
    ]
    if not frames:
        raise ValueError("write_video: no frame")
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".mp4":
        exe = ffmpeg_path()
        if exe is None:
            raise RuntimeError(
                "write_video: ffmpeg is needed for .mp4 (or pip install imageio-ffmpeg)"
            )
        frames = [_even(f) for f in frames]
        h, w = frames[0].shape[:2]
        cmd = [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
        ]
        cmd += [
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(int(crf)),
        ]
        cmd += ["-movflags", "+faststart", str(path)]
        done = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), capture_output=True)
        if done.returncode:
            raise RuntimeError(f"ffmpeg failed: {done.stderr.decode(errors='replace')[:500]}")
    elif suffix == ".gif":
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        loop = 0 if mode != "once" else 1
        images[0].save(
            path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=loop
        )
    elif suffix == "":
        from PIL import Image

        path.mkdir(parents=True, exist_ok=True)
        for k, f in enumerate(frames):
            Image.fromarray(f).save(path / f"frame_{k:05d}.png")
    else:
        raise ValueError("write_video: .mp4, .gif or a folder")
    return path


# -- colours ---------------------------------------------------------------------------


def _cmap(name):
    import matplotlib

    return matplotlib.colormaps[name] if isinstance(name, str) else name


def _rgb(plane: np.ndarray, labels: bool, cmap, vmin, vmax) -> np.ndarray:
    if labels:
        return label_colors(plane)
    span = (vmax - vmin) or 1.0
    return (_cmap(cmap)(np.clip((plane - vmin) / span, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)


def _caption(rgb: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 8 + 6 * len(text), 14), fill=(255, 255, 255))
    draw.text((4, 2), text, fill=(0, 0, 0))
    return np.asarray(img)


def _resize(rgb: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    return np.asarray(
        Image.fromarray(rgb).resize((max(int(w * scale), 1), max(int(h * scale), 1)), Image.NEAREST)
    )


#: Width in pixels of the colour bars beside the frames
BAR = 88


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow < 10.1: its bitmap font
        return ImageFont.load_default()


def _with_bar(rgb: np.ndarray, colors, lo: float, hi: float, label: str = "", categories=None):
    """The frame with a colour bar on its right (``BAR`` pixels wide), titled ``label``: a
    ramp of ``colors`` (a colormap, or RGB rows in [0, 1] from ``lo`` to ``hi``) with five
    values, or a box per category (``categories``: their names, one row of ``colors``
    each)."""
    from PIL import Image, ImageDraw

    h, w = rgb.shape[:2]
    canvas = np.full((h, w + BAR, 3), 255, np.uint8)
    canvas[:, :w] = rgb
    top, bottom = int(0.16 * h), max(int(0.86 * h), int(0.16 * h) + 10)
    x0 = w + 8
    if isinstance(colors, np.ndarray):
        table = np.asarray(colors, dtype=float)[:, :3]
    else:
        table = _cmap(colors)(np.linspace(0.0, 1.0, 256))[:, :3]
    table = (np.clip(table, 0.0, 1.0) * 255).astype(np.uint8)
    boxes = []
    if categories is None:
        rows = np.linspace(len(table) - 1, 0, bottom - top).round().astype(int)
        canvas[top:bottom, x0 : x0 + 14] = table[rows][:, None, :]
    else:
        step = max(min(20, (bottom - top) // max(len(categories), 1)), 6)
        for k in range(len(categories)):
            y = top + k * step
            canvas[y : y + step - 3, x0 : x0 + 14] = table[k]
            boxes.append(y + (step - 3) // 2)
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = _font(11 if h < 320 else 13)
    if label:
        draw.text((x0 - 2, max(top - 22, 1)), str(label)[:13], fill=(0, 0, 0), font=font)
    if categories is None:
        for k in range(5):
            y = bottom - k * (bottom - top) / 4.0
            draw.line([(x0 + 14, y), (x0 + 18, y)], fill=(0, 0, 0))
            draw.text(
                (x0 + 21, y - 6), f"{lo + k * (hi - lo) / 4.0:.3g}", fill=(0, 0, 0), font=font
            )
    else:
        for name, y in zip(categories, boxes, strict=True):
            draw.text((x0 + 21, y - 6), str(name)[:10], fill=(0, 0, 0), font=font)
    return np.asarray(img)


#: Diverging colormaps: their range is centred on zero for fields of both signs
DIVERGING = ("coolwarm", "bwr", "seismic", "RdBu", "RdBu_r", "PiYG", "PRGn", "BrBG", "PuOr",
             "RdGy", "RdYlBu", "RdYlGn", "Spectral", "berlin", "managua", "vanimo")  # fmt: skip


def _range(data: np.ndarray, vmin, vmax, cmap=None):
    """The colour range of data (``vmin``, ``vmax`` when given); centred on zero for a
    diverging colormap when the data change sign."""
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo = float(finite.min()) if vmin is None and finite.size else (vmin or 0.0)
    hi = float(finite.max()) if vmax is None and finite.size else (vmax or 1.0)
    if vmin is None and abs(lo) < 1e-9 * abs(hi):
        lo = 0.0
    if vmin is None and vmax is None and isinstance(cmap, str) and cmap in DIVERGING:
        if lo < 0.0 < hi:
            lo, hi = -max(-lo, hi), max(-lo, hi)
    return lo, hi


# -- morphologies in sections ----------------------------------------------------------


def section_frames(
    image,
    axis: int = 2,
    *,
    size: int = 480,
    cmap="viridis",
    vmin=None,
    vmax=None,
    labels=None,
    caption=True,
    bar: bool = False,
    label: str = "",
):
    """A section sweeping through a voxel image along ``axis``: label colours for phase
    images, a colormap (fixed range) for continuous ones, with a colour bar (``bar``)
    titled ``label``."""
    data = np.asarray(image.data if isinstance(image, VoxelImage) else image)
    labels = data.dtype.kind in "iub" if labels is None else labels
    lo, hi = _range(data, vmin, vmax, cmap) if not labels else (0.0, 1.0)
    frames = []
    for k in range(data.shape[axis]):
        plane = np.take(data, k, axis=axis).T[::-1]  # first in-plane axis horizontal
        rgb = _resize(_rgb(plane, labels, cmap, lo, hi), size)
        rgb = _caption(rgb, f"{'xyz'[axis]} = {k}") if caption else rgb
        frames.append(_with_bar(rgb, cmap, lo, hi, label) if bar and not labels else rgb)
    return frames


def time_section_frames(
    images,
    axis: int = 2,
    index: int | None = None,
    *,
    size: int = 480,
    cmap="viridis",
    vmin=None,
    vmax=None,
    captions=None,
    bar: bool = True,
    label: str = "",
):
    """The same section (``index`` along ``axis``, by default the middle one) of successive
    fields on a voxel grid (a field at the times of a loading path), on one colour range,
    with a colour bar and a caption per time (e.g. ``"t = 10"``)."""
    datas = [
        np.asarray(im.data if isinstance(im, VoxelImage) else im, dtype=float) for im in images
    ]
    if not datas:
        raise ValueError("time_section_frames: at least one field")
    lo, hi = _range(np.concatenate([d.ravel() for d in datas]), vmin, vmax, cmap)
    k = datas[0].shape[axis] // 2 if index is None else int(index)
    out = []
    for n, d in enumerate(datas):
        plane = np.take(d, k, axis=axis).T[::-1]
        rgb = _resize(_rgb(plane, False, cmap, lo, hi), size)
        if captions is not None:
            rgb = _caption(rgb, str(captions[n]))
        out.append(_with_bar(rgb, cmap, lo, hi, label) if bar else rgb)
    return out


def orthoslice_frames(
    image, *, frames: int = 48, size: int = 300, cmap="viridis", vmin=None, vmax=None, labels=None
):
    """The three orthogonal sections moving together through a voxel image, side by side
    (normal to x, y and z)."""
    data = np.asarray(image.data if isinstance(image, VoxelImage) else image)
    labels = data.dtype.kind in "iub" if labels is None else labels
    lo, hi = _range(data, vmin, vmax, cmap) if not labels else (0.0, 1.0)
    out = []
    for t in np.linspace(0.0, 1.0, int(frames)):
        tiles = []
        for axis in range(3):
            k = round(t * (data.shape[axis] - 1))
            plane = np.take(data, k, axis=axis).T[::-1]
            rgb = _resize(_rgb(plane, labels, cmap, lo, hi), size)
            canvas = np.full((size, size, 3), 255, np.uint8)
            canvas[: rgb.shape[0], : rgb.shape[1]] = rgb
            tiles.append(_caption(canvas, f"{'xyz'[axis]} = {k}"))
        out.append(np.concatenate(tiles, axis=1))
    return out


# -- 3D views (VTK) ----------------------------------------------------------------------


def _vtk():
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("3D views need VTK: pip install vtk") from exc
    return vtk, numpy_support


NO_RENDERING = (
    "VTK cannot render here: it finds no display, EGL or OSMesa (on a headless machine, run "
    "under a virtual display, xvfb-run, or install the OSMesa or EGL library)"
)

_PROBE = """\
import vtk
window = vtk.vtkRenderWindow()
window.SetOffScreenRendering(1)
window.AddRenderer(vtk.vtkRenderer())
window.SetSize(8, 8)
window.Render()
"""


@functools.cache
def vtk_can_render() -> bool:
    """Whether VTK can render off screen here. Without a display, EGL or OSMesa, VTK aborts
    the process at its first render, so a subprocess renders first; the answer is kept."""
    if importlib.util.find_spec("vtk") is None:
        return False
    try:
        probe = subprocess.run(
            [sys.executable, "-c", _PROBE], capture_output=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _lut(vtk, cmap, lo, hi):
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(lo, hi)
    colors = _cmap(cmap)(np.linspace(0.0, 1.0, 256))
    for k, c in enumerate(colors):
        lut.SetTableValue(k, *c)
    lut.Build()
    return lut


def _phase_lut(vtk, n: int):
    """A lookup table of the phases ``0 .. n - 1`` in the colours of the sections."""
    n = max(int(n), 2)
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.SetRange(0.0, n - 1.0)
    for k, c in enumerate(label_colors(np.arange(n))):
        lut.SetTableValue(k, *(np.asarray(c, dtype=float) / 255.0), 1.0)
    lut.Build()
    return lut


def _colours(vtk, mesh, field, cmap, clim):
    """Point or cell values of ``field`` on a mesh, their lookup table and the number of
    labels of its colour bar; ``field="phase"``: the phases of the elements, in the
    colours of the sections."""
    if isinstance(field, str):
        if field != "phase":
            raise ValueError("field: 'phase', or values per node or per element")
        phase = np.asarray(mesh.phase, dtype=float)
        n = int(phase.max()) + 1 if phase.size else 1
        return None, phase, _phase_lut(vtk, n), max(n, 2)
    point, cell = _field_values(mesh, field)
    values = point if point is not None else cell
    if values is None:
        return None, None, None, 5
    return point, cell, _lut(vtk, cmap, *(clim or _range(values, None, None, cmap))), 5


def _grid(mesh, point=None, cell=None):
    """A VTK unstructured grid of a tetrahedral mesh with point and cell scalars."""
    vtk, ns = _vtk()
    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    points.SetData(ns.numpy_to_vtk(np.ascontiguousarray(mesh.nodes, dtype=float), deep=True))
    grid.SetPoints(points)
    cell_type = vtk.VTK_TETRA if mesh.order == 1 else vtk.VTK_QUADRATIC_TETRA
    nn = mesh.elements.shape[1]
    offsets = np.arange(0, nn * (mesh.n_elements + 1), nn, dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(
        ns.numpy_to_vtkIdTypeArray(offsets, deep=True),
        ns.numpy_to_vtkIdTypeArray(
            np.ascontiguousarray(mesh.elements, dtype=np.int64).ravel(), deep=True
        ),
    )
    grid.SetCells(cell_type, cells)
    if point is not None:
        arr = ns.numpy_to_vtk(np.ascontiguousarray(point, dtype=float), deep=True)
        arr.SetName("field")
        grid.GetPointData().SetScalars(arr)
    if cell is not None:
        arr = ns.numpy_to_vtk(np.ascontiguousarray(cell, dtype=float), deep=True)
        arr.SetName("field")
        grid.GetCellData().SetScalars(arr)
    return grid


def _phase_surface(image: VoxelImage, phase: int):
    """The surface of a phase of a voxel image (flying edges on its indicator, padded so
    that the faces of the box close it)."""
    vtk, ns = _vtk()
    data = (np.asarray(image.data) == phase).astype(np.float32)
    data = np.pad(data, 1)
    img = vtk.vtkImageData()
    img.SetDimensions(*data.shape)
    img.SetSpacing(*image.spacing)
    img.SetOrigin(*(np.asarray(image.origin) - 0.5 * np.asarray(image.spacing)))
    img.GetPointData().SetScalars(ns.numpy_to_vtk(data.ravel(order="F"), deep=True))
    edges = vtk.vtkFlyingEdges3D()
    edges.SetInputData(img)
    edges.SetValue(0, 0.5)
    edges.ComputeNormalsOn()
    edges.Update()
    return edges.GetOutput()


class Scene:
    """An off-screen VTK scene: actors, a camera turning about the centre of their
    bounds, and the rendered frames as RGB arrays."""

    def __init__(self, size=(640, 480), background=(1.0, 1.0, 1.0)):
        vtk, ns = _vtk()
        if not vtk_can_render():
            raise RuntimeError(NO_RENDERING)
        self._vtk, self._ns = vtk, ns
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*background)
        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.AddRenderer(self.renderer)
        self.size = (int(size[0]), int(size[1]))
        self.window.SetSize(*self.size)
        self._legend = None

    def add(self, data, *, lut=None, color=(0.8, 0.8, 0.85), edges=False, opacity=1.0):
        vtk = self._vtk
        if data.IsA("vtkUnstructuredGrid"):
            surface = vtk.vtkDataSetSurfaceFilter()
            surface.SetInputData(data)
            surface.Update()
            data = surface.GetOutput()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        if lut is not None:
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(lut.GetRange())
            mapper.ScalarVisibilityOn()
        else:
            mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*color)
        prop.SetOpacity(opacity)
        if edges:
            prop.EdgeVisibilityOn()
            prop.SetEdgeColor(0.2, 0.2, 0.2)
        self.renderer.AddActor(actor)
        return actor

    def bar(self, lut, title: str = "", labels: int = 5):
        """A colour bar of the lookup table on the right of the frames, which keep their
        size; ``labels`` equal to the number of colours of the table: one box per value
        (phases)."""
        n = lut.GetNumberOfTableValues()
        colors = np.array([lut.GetTableValue(k)[:3] for k in range(n)])
        lo, hi = lut.GetRange()
        categories = [f"{lo + k:g}" for k in range(n)] if int(labels) == n else None
        self._legend = (colors, lo, hi, title, categories)
        self.window.SetSize(max(self.size[0] - BAR, 16), self.size[1])

    def view(self, elevation: float = 25.0, azimuth: float = 30.0, zoom: float = 1.0):
        camera = self.renderer.GetActiveCamera()
        self.renderer.ResetCamera()
        camera.Elevation(elevation)
        camera.Azimuth(azimuth)
        camera.OrthogonalizeViewUp()
        camera.Zoom(zoom)
        self.renderer.ResetCameraClippingRange()

    def frame(self) -> np.ndarray:
        self.window.Render()
        grab = self._vtk.vtkWindowToImageFilter()
        grab.SetInput(self.window)
        grab.ReadFrontBufferOff()
        grab.Update()
        img = grab.GetOutput()
        w, h, _ = img.GetDimensions()
        arr = self._ns.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)
        rgb = np.ascontiguousarray(arr[::-1, :, :3])
        return rgb if self._legend is None else _with_bar(rgb, *self._legend)


def _field_values(mesh, field):
    """Point or cell scalars of a field (vectors by their norm)."""
    if field is None:
        return None, None
    values = np.asarray(field, dtype=float)
    if values.ndim == 2:
        values = np.linalg.norm(values, axis=1)
    if len(values) == mesh.n_nodes:
        return values, None
    if len(values) == mesh.n_elements:
        return None, values
    raise ValueError("field: one value (or vector) per node or per element")


def turntable_frames(
    obj,
    *,
    field=None,
    phase: int = 1,
    frames: int = 72,
    size=(640, 480),
    elevation: float = 25.0,
    azimuth: float = 30.0,
    cmap="viridis",
    clim=None,
    title: str = "",
    edges: bool = False,
):
    """A turntable of the surface of a phase of a voxel image, or of a tetrahedral mesh
    coloured by ``field`` (per node or element, vectors by their norm; ``"phase"``: the
    phases of the elements)."""
    scene = Scene(size)
    vtk, _ = _vtk()
    if isinstance(obj, VoxelImage):
        scene.add(_phase_surface(obj, phase), color=(0.75, 0.78, 0.86))
    else:
        point, cell, lut, labels = _colours(vtk, obj, field, cmap, clim)
        if lut is not None:
            scene.bar(lut, title or ("phase" if isinstance(field, str) else ""), labels)
        scene.add(_grid(obj, point, cell), lut=lut, edges=edges)
    scene.view(elevation, azimuth)
    out = []
    for _ in range(int(frames)):
        out.append(scene.frame())
        scene.renderer.GetActiveCamera().Azimuth(360.0 / int(frames))
    return out


def _clipped(grid, origin, normal):
    vtk, _ = _vtk()
    plane = vtk.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)
    clip = vtk.vtkTableBasedClipDataSet()
    clip.SetInputData(grid)
    clip.SetClipFunction(plane)
    clip.Update()
    return clip.GetOutput()


def section_sweep_frames(
    mesh,
    field=None,
    *,
    axis: int = 2,
    frames: int = 48,
    size=(640, 480),
    elevation: float = 25.0,
    azimuth: float = 30.0,
    cmap="viridis",
    clim=None,
    title: str = "",
):
    """A cross-section sweeping through a tetrahedral mesh along ``axis``: the part beyond
    the plane, coloured by ``field`` on its outer faces and on the cut (``"phase"``: the
    phases of the elements)."""
    vtk, _ = _vtk()
    point, cell, lut, labels = _colours(vtk, mesh, field, cmap, clim)
    grid = _grid(mesh, point, cell)
    lower, upper = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    normal = np.eye(3)[axis]
    scene = Scene(size)
    if lut is not None:
        scene.bar(lut, title or ("phase" if isinstance(field, str) else ""), labels)
    actor = scene.add(grid, lut=lut)
    scene.view(elevation, azimuth)
    out = []
    for t in np.linspace(0.02, 0.98, int(frames)):
        origin = lower + t * (upper - lower)
        cut = _clipped(grid, origin, normal)
        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInputData(cut)
        surface.Update()
        actor.GetMapper().SetInputData(surface.GetOutput())
        out.append(scene.frame())
    return out


def field_frames(
    mesh,
    displacement,
    field=None,
    *,
    magnification: float | None = None,
    frames: int = 48,
    wave: str = "sine",
    section=None,
    size=(640, 480),
    elevation: float = 25.0,
    azimuth: float = 30.0,
    cmap="viridis",
    clim=None,
    title: str = "",
    scaling: str | None = None,
):
    """The deformed mesh ``x + s(t) m u``: ``wave`` "sine" (a mode oscillating, ``s =
    sin 2πt``) or "ramp" (a load growing, ``s = t``); ``m`` the magnification (by default
    the largest displacement drawn as a tenth of the mesh); coloured by ``field`` (by
    default the norm of the displacement), optionally cut by a plane ``section = (axis,
    fraction)``. The field is that of the full load, which a linear solution scales by
    ``s`` (``scaling`` of :func:`field_scaling`: by default ``"magnitude"`` for the norm of
    the displacement, ``"linear"`` for other values, ``"none"`` for the phases), so that
    the colours follow the load in a range fixed over the frames (``clim``, else that of
    the full load from zero, symmetric when an oscillating load changes the sign)."""
    vtk, ns = _vtk()
    u = np.asarray(displacement, dtype=float).reshape(-1, 3)
    m = _magnification(mesh, [u], magnification)
    values = np.linalg.norm(u, axis=1) if field is None else field
    if scaling is None:
        scaling = "magnitude" if field is None else "linear"
    if isinstance(values, str):  # the phases do not follow the load
        scaling = "none"
    if scaling not in SCALINGS:
        raise ValueError(f"scaling: one of {', '.join(SCALINGS)}")
    if clim is None and scaling != "none":
        point, cell = _field_values(mesh, values)
        lo, hi = _range(point if point is not None else cell, None, None, cmap)
        lo, hi = min(lo, 0.0), max(hi, 0.0)  # the load grows from zero
        if wave == "sine" and scaling == "linear":
            lo, hi = -max(-lo, hi), max(-lo, hi)
        clim = (lo, hi)
    point, cell, lut, labels = _colours(vtk, mesh, values, cmap, clim)
    grid = _grid(mesh, point, cell)
    scene = Scene(size)
    scene.bar(lut, title or ("u" if field is None else ""), labels)
    actor = scene.add(grid, lut=lut)
    times = np.linspace(0.0, 1.0, int(frames), endpoint=wave == "ramp")
    scales = np.sin(2 * np.pi * times) if wave == "sine" else times
    lower, upper = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    first = True
    out = []
    for s in scales:
        moved = mesh.nodes + s * m * u
        grid.GetPoints().SetData(ns.numpy_to_vtk(np.ascontiguousarray(moved), deep=True))
        if scaling != "none":
            factor = {"linear": s, "magnitude": abs(s), "quadratic": s * s}[scaling]
            for data, base in ((grid.GetPointData(), point), (grid.GetCellData(), cell)):
                if base is not None:
                    arr = ns.numpy_to_vtk(np.ascontiguousarray(factor * base), deep=True)
                    arr.SetName("field")
                    data.SetScalars(arr)
        grid.Modified()
        actor.GetMapper().SetInputData(_surface(vtk, grid, section, lower, upper))
        if first:
            scene.view(elevation, azimuth)
            first = False
        out.append(scene.frame())
    return out


#: How the fields of a linear solution follow its load factor ``s`` (:func:`field_scaling`)
SCALINGS = ("linear", "magnitude", "quadratic", "none")


def field_scaling(name: str | None) -> str:
    """How a field of a linear solution follows its load factor ``s``: ``"linear"`` (``s
    v``: the components of stresses, strains and displacements), ``"magnitude"`` (``|s|
    v``: norms and equivalent values, the von Mises stress, the norm of the displacement,
    ``None``), ``"quadratic"`` (``s² v``: energy densities) or ``"none"`` (the phases)."""
    if name is None:
        return "magnitude"
    name = str(name).lower()
    if name == "phase":
        return "none"
    if "energy" in name:
        return "quadratic"
    if name in ("von_mises", "displacement", "equivalent_strain", "heat_flux") or name.endswith(
        ("_norm", "_magnitude")
    ):
        return "magnitude"
    return "linear"


def _magnification(mesh, displacements, magnification) -> float:
    """The magnification of displacements: by default, the largest one drawn as a tenth
    of the size of the mesh."""
    if magnification is not None:
        return float(magnification)
    size = float(np.ptp(mesh.nodes, axis=0).max())
    peak = max((float(np.linalg.norm(u, axis=1).max()) for u in displacements), default=0.0)
    return 0.1 * size / max(peak, 1e-300)


def _surface(vtk, grid, section, lower, upper):
    """The outer faces of a grid, cut by the plane ``section = (axis, fraction)`` of the
    box ``[lower, upper]`` (the part beyond it)."""
    shown = grid
    if section is not None:
        axis, fraction = section
        origin = lower + float(fraction) * (upper - lower)
        shown = _clipped(grid, origin, np.eye(3)[int(axis)])
    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputData(shown)
    surface.Update()
    return surface.GetOutput()


def deformed_view(
    mesh,
    displacement=None,
    field=None,
    *,
    magnification: float | None = None,
    section=None,
    size=(640, 480),
    elevation: float = 25.0,
    azimuth: float = 30.0,
    cmap="viridis",
    clim=None,
    title: str = "",
    edges: bool = False,
) -> np.ndarray:
    """One view (an RGB array) of a tetrahedral mesh coloured by ``field`` (per node or
    element, vectors by their norm; ``"phase"``: the phases of the elements), deformed by
    ``m u`` when a ``displacement`` is given (``m`` as in :func:`field_frames`), cut by
    the plane ``section = (axis, fraction)``."""
    vtk, ns = _vtk()
    point, cell, lut, labels = _colours(vtk, mesh, field, cmap, clim)
    grid = _grid(mesh, point, cell)
    if displacement is not None:
        u = np.asarray(displacement, dtype=float).reshape(-1, 3)
        moved = mesh.nodes + _magnification(mesh, [u], magnification) * u
        grid.GetPoints().SetData(ns.numpy_to_vtk(np.ascontiguousarray(moved), deep=True))
    scene = Scene(size)
    if lut is not None:
        scene.bar(lut, title or ("phase" if isinstance(field, str) else ""), labels)
    lower, upper = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    scene.add(_surface(vtk, grid, section, lower, upper), lut=lut, edges=edges)
    scene.view(elevation, azimuth)
    return scene.frame()


def history_frames(
    mesh,
    displacements,
    fields=None,
    *,
    frames: int | None = None,
    magnification: float | None = None,
    section=None,
    size=(640, 480),
    elevation: float = 25.0,
    azimuth: float = 30.0,
    cmap="viridis",
    clim=None,
    title: str = "",
    captions=None,
):
    """The mesh deformed by the displacements of successive times (the steps of a loading
    path), ``x + m u_k`` (``m`` as in :func:`field_frames`), coloured by the ``fields`` of
    these times (per node or element; by default the norm of the displacement) on one
    range, cut by the plane ``section = (axis, fraction)``. ``frames``: that many frames
    interpolated linearly between the times (by default one per time); ``captions``: a
    text per time (e.g. ``"t = 10"``), shown on the frames nearest to it."""
    vtk, ns = _vtk()
    us = [np.asarray(u, dtype=float).reshape(-1, 3) for u in displacements]
    if not us:
        raise ValueError("history_frames: at least one time")
    raw = [np.linalg.norm(u, axis=1) for u in us] if fields is None else list(fields)
    if len(raw) != len(us):
        raise ValueError("history_frames: one field per displacement")
    pairs = [_field_values(mesh, f) for f in raw]
    on_points = pairs[0][0] is not None
    values = [p if on_points else c for p, c in pairs]
    lut = _lut(vtk, cmap, *(clim or _range(np.concatenate(values), None, None, cmap)))
    m = _magnification(mesh, us, magnification)
    grid = _grid(mesh, *((values[0], None) if on_points else (None, values[0])))
    scene = Scene(size)
    scene.bar(lut, title or ("u" if fields is None else ""))
    actor = scene.add(grid, lut=lut)
    n = len(us)
    positions = (
        np.arange(n, dtype=float) if frames is None else np.linspace(0.0, n - 1.0, int(frames))
    )
    lower, upper = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)

    def show(s: float) -> int:
        i = min(int(np.floor(s)), max(n - 2, 0))
        j = min(i + 1, n - 1)
        a = s - i if n > 1 else 0.0
        moved = mesh.nodes + m * ((1.0 - a) * us[i] + a * us[j])
        grid.GetPoints().SetData(ns.numpy_to_vtk(np.ascontiguousarray(moved), deep=True))
        arr = ns.numpy_to_vtk(
            np.ascontiguousarray((1.0 - a) * values[i] + a * values[j]), deep=True
        )
        arr.SetName("field")
        (grid.GetPointData() if on_points else grid.GetCellData()).SetScalars(arr)
        grid.Modified()
        actor.GetMapper().SetInputData(_surface(vtk, grid, section, lower, upper))
        return i if a < 0.5 else j

    show(positions[-1])  # the camera framing the last (largest) deformation
    scene.view(elevation, azimuth)
    out = []
    for s in positions:
        k = show(float(s))
        rgb = scene.frame()
        out.append(_caption(rgb, str(captions[k])) if captions is not None else rgb)
    return out


def frame_pattern_frames(
    frame,
    mode,
    *,
    magnification: float = 1.0,
    frames: int = 48,
    wave: str = "sine",
    tiles=None,
    size: int = 480,
):
    """A pattern of a frame of beams oscillating (``wave="sine"``) or growing (``"ramp"``)
    on ``tiles`` copies of its cell: a plane frame
    (:class:`~morphogen.homogenization.frames.Frame`, 3 x 3 cells by default) or a frame
    of struts in space (:class:`~morphogen.homogenization.frames.SpaceFrame`, one cell
    seen in perspective)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from morphogen.viz.figures import _figure

    space = frame.lattice.shape[0] == 3
    tiles = tiles or ((1, 1, 1) if space else (3, 3))
    times = np.linspace(0.0, 1.0, int(frames), endpoint=wave == "ramp")
    scales = np.sin(2 * np.pi * times) if wave == "sine" else times
    mode = np.asarray(mode, dtype=float)
    fig = _figure(size / 120, size / 120)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(projection="3d") if space else fig.subplots()
    frame.plot(ax, mode, scale=magnification, tiles=tiles)
    limits = [ax.get_xlim(), ax.get_ylim()] + ([ax.get_zlim()] if space else [])
    setters = (ax.set_xlim, ax.set_ylim) + ((ax.set_zlim,) if space else ())
    out = []
    for s in scales:
        ax.clear()
        moved = s * mode if abs(s) > 1e-12 else None
        frame.plot(ax, moved, scale=magnification * abs(s), tiles=tiles, lw=0.8)
        for setter, lim in zip(setters, limits, strict=True):  # those of the largest motion
            setter(*lim)
        if space:
            ax.set_box_aspect(tuple(max(b - a, 1e-9) for a, b in limits))
        canvas.draw()
        out.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
    return out
