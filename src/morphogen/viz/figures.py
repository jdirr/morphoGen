"""Figures of microstructures, automata, optimizations, frames of beams, textures and of the
grains of seeds against their seeds (matplotlib, optional extra ``[plot]``). Figures are
built without pyplot
(``matplotlib.figure.Figure``): they work headless and can be saved with
``fig.savefig(path)``."""

from __future__ import annotations

import numpy as np

from morphogen.image import VoxelImage
from morphogen.models.base import Microstructure
from morphogen.viz.png import PALETTE, label_colors, slice_of


def _figure(width: float = 6.0, height: float = 4.0):
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("figures need matplotlib: pip install 'morphogen[plot]'") from exc
    return Figure(figsize=(width, height), dpi=120, layout="constrained")


def as_array(source, shape=None, *, labels: bool = False) -> np.ndarray:
    """3D array of a microstructure (voxelized on ``shape`` or its own lattice), of a
    voxel image or of an array; ``labels`` takes the cell labels of the structure."""
    if isinstance(source, Microstructure):
        if labels:
            if not source.has_labels:
                raise ValueError("this microstructure has no cell labels")
            img = (
                source.labels()
                if shape is None and hasattr(source, "shape")
                else source.labels(shape=shape or 64)
            )
            return img.data
        if shape is None and hasattr(source, "lattice"):
            return source.voxelize().data
        return source.voxelize(shape=shape or 64).data
    if isinstance(source, VoxelImage):
        return source.data
    a = np.asarray(source)
    return a[:, :, None] if a.ndim == 2 else a


def _show(ax, plane, kind: str, title: str = ""):
    from morphogen.viz.png import to_rgb

    ax.imshow(to_rgb(plane, kind), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def _kind(data: np.ndarray, labels: bool) -> str:
    if labels:
        return "labels"
    return "phase" if data.dtype.kind in "biu" and data.max(initial=0) < 8 else "field"


def slices_figure(source, *, shape=None, labels: bool = False, title: str | None = None):
    """Middle sections normal to x, y and z (one image for planar structures)."""
    data = as_array(source, shape, labels=labels)
    kind = _kind(data, labels)
    planar = data.shape[2] == 1
    fig = _figure(3.2 if planar else 9.0, 3.4)
    if planar:
        ax = fig.add_subplot(1, 1, 1)
        _show(ax, slice_of(data, 2), kind, "x-y")
    else:
        for i, (axis, name) in enumerate(((0, "y-z"), (1, "x-z"), (2, "x-y"))):
            _show(fig.add_subplot(1, 3, i + 1), slice_of(data, axis), kind, f"{name} section")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def field_figure(data, *, cmap="viridis", clim=None, title: str | None = None, label: str = ""):
    """Middle sections normal to x, y and z of a field on a voxel grid (one for planar
    grids), on one colour range with a colour bar titled ``label``."""
    values = np.asarray(data.data if isinstance(data, VoxelImage) else data, dtype=float)
    values = values[:, :, None] if values.ndim == 2 else values
    from morphogen.viz.video import _range

    lo, hi = clim or _range(values, None, None, cmap)
    planar = values.shape[2] == 1
    sections = ((2, "x-y"),) if planar else ((0, "y-z"), (1, "x-z"), (2, "x-y"))
    fig = _figure(4.0 if planar else 10.0, 3.4)
    axes = []
    for i, (axis, name) in enumerate(sections):
        ax = fig.add_subplot(1, len(sections), i + 1)
        shown = ax.imshow(
            slice_of(values, axis), cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name} section", fontsize=9)
        axes.append(ax)
    bar = fig.colorbar(shown, ax=axes, shrink=0.85)
    if label:
        bar.set_label(label, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def mesh_figure(mesh, *, axis: int = 2, position: float | None = None, values=None,
                cmap="viridis", label: str = "", title: str | None = None,
                grains: bool = False):  # fmt: skip
    """The cut of a tetrahedral mesh by a plane normal to ``axis`` (through the middle of
    its box, or at the coordinate ``position``): the polygons of the elements that the
    plane crosses, in the colours of their phases or coloured by ``values`` (one per
    element, with a colour bar titled ``label``), their edges drawn — the elements of a
    conforming mesh following the interfaces. See :func:`mesh_section` for the polygons. A
    mesh of the plane (:class:`~morphogen.mesh.triangles.TriMesh`) is drawn whole, its
    triangles. ``grains``: the elements of polycrystals in a colour per grain."""
    from matplotlib.collections import PolyCollection

    polygons, elements = mesh_section(mesh, axis, position)
    nodes = np.asarray(mesh.nodes)
    if nodes.shape[1] == 2:  # a mesh of the plane
        nodes = np.column_stack([nodes, np.zeros(len(nodes))])
    lower, upper = nodes.min(axis=0), nodes.max(axis=0)
    u, v = [a for a in range(3) if a != axis]
    width, height = upper[u] - lower[u], upper[v] - lower[v]
    fig = _figure(4.6 * width / max(width, height) + (0.9 if values is not None else 0.0),
                  4.6 * height / max(width, height) + 0.4)  # fmt: skip
    ax = fig.add_subplot(1, 1, 1)
    if values is None:
        if grains and getattr(mesh, "cell", None) is not None:
            colours = label_colors(np.asarray(mesh.cell)[elements]) / 255.0
        else:
            phase = np.asarray(mesh.phase)[elements]
            colours = PALETTE[np.minimum(phase, len(PALETTE) - 1)] / 255.0
        cut = PolyCollection(polygons, facecolors=colours, edgecolors="0.15", linewidths=0.25)
    else:
        cut = PolyCollection(polygons, array=np.asarray(values, dtype=float)[elements], cmap=cmap,
                             edgecolors="0.15", linewidths=0.25)  # fmt: skip
        bar = fig.colorbar(cut, ax=ax, shrink=0.85)
        if label:
            bar.set_label(label, fontsize=9)
    ax.add_collection(cut)
    ax.set_xlim(lower[u], upper[u])
    ax.set_ylim(lower[v], upper[v])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    names = "xyz"
    ax.set_title(title or f"{names[u]}-{names[v]} section of the mesh", fontsize=9)
    return fig


def mesh_section(mesh, axis: int = 2, position: float | None = None):
    """The cut of the elements of a tetrahedral mesh by the plane normal to ``axis`` at
    the coordinate ``position`` (the middle of its box by default): the polygons (3 or 4
    points each, in the two other coordinates, in order around them) and the indices of
    their elements. Their areas sum to the section of the mesh: a face that lies in the
    plane is the cut of the element below it."""
    if axis not in (0, 1, 2):
        raise ValueError(f"mesh_section: axis 0, 1 or 2, got {axis!r}")
    from morphogen.mesh.triangles import TriMesh

    if isinstance(mesh, TriMesh):  # a mesh of the plane: its triangles
        if axis != 2:
            raise ValueError("mesh_section: a mesh of the plane is its section normal to z")
        corners = np.asarray(mesh.nodes, dtype=float)[np.asarray(mesh.elements)[:, :3]]
        return list(corners), np.arange(mesh.n_elements, dtype=np.int64)
    nodes = np.asarray(mesh.nodes, dtype=float)
    corners = np.asarray(mesh.elements)[:, :4]
    along = nodes[:, axis]
    z = 0.5 * (along.min() + along.max()) if position is None else float(position)
    d = along[corners] - z
    # the nodes on the plane count as above it: a face in the plane is cut once
    crossed = np.flatnonzero((d.min(axis=1) < 0.0) & (d.max(axis=1) >= 0.0))
    plane = [a for a in range(3) if a != axis]
    tiny = 1e-12 * float(np.prod(np.ptp(nodes[:, plane], axis=0)))
    polygons, kept = [], []
    for e in crossed:
        points = []
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            da, db = d[e, a], d[e, b]
            if (da < 0.0) != (db < 0.0):  # the edge crosses the plane
                t = da / (da - db)
                pa, pb = nodes[corners[e, a]], nodes[corners[e, b]]
                points.append((pa + t * (pb - pa))[plane])
        points = np.array(points)
        centre = points.mean(axis=0)
        order = np.argsort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))
        points = points[order]
        x, y = points[:, 0], points[:, 1]
        if 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > tiny:
            polygons.append(points)  # (an element that touches the plane has no cut)
            kept.append(e)
    return polygons, np.array(kept, dtype=np.int64)


def path_figure(result, *, title: str | None = None):
    """Curves of a loading path (:class:`~morphogen.homogenization.paths.PathResult`): the
    macroscopic stresses that vary against the strain component that varies most (in
    finite strains, the true stresses against the logarithmic strains), and the mean
    cumulated plastic strain against time when the materials yield."""
    from morphogen.homogenization.elements import VOIGT_NAMES

    true = result.finite and result.log_strain is not None and result.cauchy is not None
    strain = np.asarray(result.log_strain if true else result.strain, dtype=float)
    stress = np.asarray(result.cauchy if true else result.stress, dtype=float)
    e_name, s_name = ("LE", "TS") if true else ("E", "S")
    i = int(np.argmax(np.ptp(strain, axis=0)))
    spans = np.ptp(stress, axis=0)
    shown = [j for j in range(6) if spans[j] > 0.01 * spans.max()] or [i]
    plastic = np.asarray(result.plastic, dtype=float)
    yielding = bool(plastic.size and plastic.max() > 0.0)
    fig = _figure(9.0 if yielding else 5.0, 3.6)
    ax = fig.add_subplot(1, 2 if yielding else 1, 1)
    for j in shown:
        ax.plot(strain[:, i], stress[:, j], marker=".", ms=4, lw=1.4, label=s_name + VOIGT_NAMES[j])
    ax.set_xlabel(e_name + VOIGT_NAMES[i])
    ax.set_ylabel("Cauchy stress" if true else "stress")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if yielding:
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(result.times, plastic, marker=".", ms=4, lw=1.4, color="#d6604d")
        ax2.set_xlabel("time")
        ax2.set_ylabel("mean cumulated plastic strain p")
        ax2.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def tiled_figure(source, *, tiles: int = 3, shape=None, title: str | None = None, axis=None):
    """Periodic tiling (``tiles`` x ``tiles``) of the middle sections of a cell: those
    normal to x, y and z (the x-y section alone for a planar structure), or the one
    normal to ``axis`` (0, 1 or 2)."""
    data = as_array(source, shape)
    kind = _kind(data, False)
    if axis is None:
        axes = (2,) if data.shape[2] == 1 else (0, 1, 2)
    elif axis in (0, 1, 2):
        axes = (int(axis),)
    else:
        raise ValueError("axis: 0, 1 or 2 (the normal of the section)")
    fig = _figure(4.0 * len(axes), 4.0)
    names = ("y-z", "x-z", "x-y")
    for i, a in enumerate(axes):
        plane = np.tile(slice_of(data, a), (tiles, tiles))
        _show(fig.add_subplot(1, len(axes), i + 1), plane, kind,
              f"{names[a]} section" if len(axes) > 1 else "")  # fmt: skip
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def surface_figure(source, *, shape=None, phase: int = 1, title: str | None = None):
    """3D view of the boundary of a phase (marching cubes with scikit-image, voxel faces
    otherwise)."""
    data = as_array(source, shape)
    if data.shape[2] == 1:
        raise ValueError("planar structures: use slices_figure")
    fig = _figure(5.0, 5.0)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    mask = (data == phase).astype(np.float32)
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage.measure import marching_cubes

        padded = np.pad(mask, 1)
        verts, faces, _, _ = marching_cubes(padded, 0.5)
        mesh = Poly3DCollection(verts[faces] - 1, linewidths=0.0, alpha=1.0)
        shade = 0.45 + 0.55 * np.abs(_normals(verts, faces)[:, 2])
        mesh.set_facecolor(np.column_stack([0.12 * shade, 0.31 * shade, 0.47 * shade]))
        ax.add_collection3d(mesh)
        for setter, n in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), data.shape, strict=True):
            setter(0, n)
    except ImportError:
        step = max(1, max(data.shape) // 48)
        ax.voxels(mask[::step, ::step, ::step] > 0, facecolors="#1f4e79", edgecolor="none")
    ax.set_box_aspect(data.shape)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    return fig


def _normals(verts, faces) -> np.ndarray:
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(b - a, c - a)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def history_figure(history: dict, keys=None, *, title: str | None = None):
    """Curves of a history (automaton, optimization...) against its first column."""
    columns = list(history)
    x_key = columns[0]
    keys = [k for k in (keys or columns[1:]) if k in history and k != x_key]
    fig = _figure(6.0, 1.8 * max(1, len(keys)))
    x = np.asarray(history[x_key], dtype=float)
    for i, key in enumerate(keys):
        ax = fig.add_subplot(len(keys), 1, i + 1)
        ax.plot(x, np.asarray(history[key], dtype=float), color="#1f4e79", lw=1.5)
        ax.set_ylabel(key, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if i == len(keys) - 1:
            ax.set_xlabel(x_key, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def kinetics_figure(micro, domain=None):
    """Transformed fraction against time of a nucleation-growth structure, with the
    Avrami law."""
    k = micro.attributes.get("kinetics")
    if k is None:
        raise ValueError("this structure has no transformation kinetics")
    fig = _figure(5.0, 3.2)
    ax = fig.add_subplot(1, 1, 1)
    t = np.asarray(k["time"])
    ax.plot(t, k["transformed"], "o", ms=3, color="#1f4e79", label="automaton")
    model = micro.model
    if model is not None and hasattr(model, "transformed_fraction"):
        fine = np.linspace(0, t.max(), 200)
        ax.plot(
            fine, model.transformed_fraction(fine, micro.domain), color="#d6604d", label="Avrami"
        )
    ax.set_xlabel("time")
    ax.set_ylabel("transformed fraction")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def softening_figure(studies, *, title: str | None = None):
    """Replicas of networks of non-identical beams
    (:class:`~morphogen.homogenization.networks.Softening`, one or a list, a mode each):
    the mean ``E / E_0`` of the replicas against the coefficient of variation with its
    standard errors (filled: corrected by the control variate; open: the plain mean) and
    the second-order mean ``1 - s CV²``; the scatter ``σ_E² / E_0²`` of the replicas
    against ``v CV²`` (its standard error for normal draws) and the diagonal."""
    from morphogen.homogenization.networks import Softening

    studies = [studies] if isinstance(studies, Softening) else list(studies)
    fig = _figure(9.6, 3.9)
    left, right = fig.subplots(1, 2)
    top = 0.0
    for k, study in enumerate(studies):
        colour = f"C{k % 10}"
        rows = study.rows()
        s, v = study.coefficients["mean"], study.coefficients["variance"]
        cv = np.linspace(0.0, 1.05 * float(study.cv.max()), 100)
        left.plot(cv, 1.0 - s * cv**2, color=colour, lw=1.4,
                  label=f"{study.mode}: 1 − s CV², s = {s:.3g}")  # fmt: skip
        left.errorbar(study.cv, study.corrected, yerr=study.error, fmt="o", ms=5, color=colour,
                      capsize=3)  # fmt: skip
        left.errorbar(study.cv, study.mean, yerr=[r["mean_error"] for r in rows], fmt="o", ms=7,
                      mfc="none", color=colour, capsize=3, alpha=0.7)  # fmt: skip
        x, y = study.predicted_scatter(), study.scatter()
        errors = np.array([r["scatter_error"] for r in rows])
        right.errorbar(x, y, yerr=errors, fmt="o", ms=5, color=colour, capsize=3,
                       label=f"{study.mode}: v = {v:.3g}")  # fmt: skip
        top = max(top, float(x.max()), float((y + errors).max()))
    for fill, name in (("0.35", "replicas, control variate"), ("none", "replicas, plain mean")):
        left.plot([], [], "o", color="0.35", mfc=fill, label=name)
    left.set_xlabel("coefficient of variation CV")
    left.set_ylabel("⟨E⟩ / E₀")
    left.set_xlim(left=0.0)
    left.set_title("mean Young's modulus", fontsize=9)
    top = 1.08 * top if top > 0 else 1.0
    right.plot([0.0, top], [0.0, top], color="0.4", lw=1.0, ls="--", label="σ_E²/E₀² = v CV²")
    right.set_xlim(0.0, top)
    right.set_ylim(0.0, top)
    right.set_xlabel("v CV² (second order)")
    right.set_ylabel("σ_E² / E₀² (replicas)")
    right.set_title("scatter of the Young's modulus", fontsize=9)
    for ax in (left, right):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


#: The beams of a frame drawn as vectors at most in its vector files (.svg, .pdf); those
#: of larger frames are rasterized there (the axes, the colour bar and the texts not)
VECTOR_BEAMS = 2000
#: The scales of the values of the beams of :func:`frame_figure`
FRAME_SCALES = ("linear", "symmetric", "log")


def _cell_segments(frame) -> tuple[np.ndarray, np.ndarray]:
    """The beams of a periodic frame as segments in its cell: each from its first node
    (brought into the cell), cut where it leaves the cell and its pieces brought back by
    the periods (pieces x 2 x dimension), with the beam of each piece."""
    lattice = np.asarray(frame.lattice, dtype=float)
    inverse = np.linalg.inv(lattice)
    reduced = np.asarray(frame.nodes, dtype=float)[frame.beams[:, 0]] @ inverse.T
    reduced -= np.floor(reduced)
    step = frame.vectors() @ inverse.T
    end = reduced + step
    inside = np.all((end >= 0.0) & (end <= 1.0), axis=1)
    pieces = [np.stack([reduced[inside], end[inside]], axis=1)]
    owners = [np.flatnonzero(inside)]
    for k in np.flatnonzero(~inside):
        a, b = reduced[k], end[k]
        cuts = [0.0, 1.0]
        for c in range(len(a)):
            lo, hi = sorted((a[c], b[c]))
            cuts += [(n - a[c]) / (b[c] - a[c]) for n in range(int(np.floor(lo)) + 1,
                                                               int(np.ceil(hi)))]  # fmt: skip
        points = a + np.unique(cuts)[:, None] * (b - a)
        shift = np.floor(0.5 * (points[1:] + points[:-1]))
        pieces.append(np.stack([points[:-1] - shift, points[1:] - shift], axis=1))
        owners.append(np.full(len(shift), k))
    return np.concatenate(pieces) @ lattice.T, np.concatenate(owners)


def frame_figure(frame, values=None, *, scale: str = "linear", label: str = "",
                 title: str | None = None):  # fmt: skip
    """The beams of a periodic frame as segments in its cell (a
    :class:`~morphogen.homogenization.frames.Frame` in its plane, a
    :class:`~morphogen.homogenization.frames.SpaceFrame` in an orthographic 3D view),
    each cut where it leaves the cell and its pieces brought back by the periods, with the
    outline of the cell; coloured by one value per beam (``values``, the colour bar
    titled ``label``; none: the beams alone), the strongest drawn widest and on top, on a
    ``scale``: ``linear`` (from zero), ``symmetric`` (about zero, as the axial forces of
    :func:`~morphogen.homogenization.networks.beam_forces`: tension red, compression blue)
    up to the 99th percentile of their magnitudes, or ``log`` (three decades below the
    largest, as the energies of the beams). The beams of frames of more than
    :data:`VECTOR_BEAMS` pieces are rasterized in the vector files."""
    from matplotlib import colormaps
    from matplotlib.colors import LogNorm, Normalize

    if scale not in FRAME_SCALES:
        raise ValueError(f"scale: one of {FRAME_SCALES}")
    segments, owners = _cell_segments(frame)
    dim = segments.shape[2]
    if values is None:
        colours, widths, norm, cmap = "0.2", np.full(len(owners), 0.7), None, None
        order = np.arange(len(owners))
    else:
        v = np.asarray(values, dtype=float).ravel()
        if v.shape != (len(frame.beams),) or not np.all(np.isfinite(v)):
            raise ValueError("values: one finite value per beam")
        if scale == "log":
            top = float(v.max()) if np.any(v > 0) else 1.0
            norm, cmap = LogNorm(1e-3 * top, top), colormaps["magma_r"]
            v = np.clip(v, 1e-3 * top, top)
            t = np.log10(v / top) / 3.0 + 1.0
        else:
            top = float(np.percentile(np.abs(v), 99)) or float(np.abs(v).max()) or 1.0
            symmetric = scale == "symmetric"
            norm = Normalize(-top if symmetric else min(0.0, float(v.min())), top)
            cmap = colormaps["RdBu_r" if symmetric else "magma_r"]
            t = np.minimum(np.abs(v) / top, 1.0)
        t, values = t[owners], v[owners]
        order = np.argsort(t, kind="stable")
        colours, widths = None, 0.3 + 1.9 * t
    fig = _figure(5.8 if values is not None else 5.0, 5.0)
    corners = np.array(list(np.ndindex(*(2,) * dim)), dtype=float)
    edges = [(i, j) for i in range(len(corners)) for j in range(i + 1, len(corners))
             if np.abs(corners[i] - corners[j]).sum() == 1]  # fmt: skip
    outline = (corners @ np.asarray(frame.lattice, dtype=float).T)[np.array(edges)]
    if dim == 3:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection as Lines

        ax = fig.add_subplot(projection="3d")
        ax.set_proj_type("ortho")
        add = ax.add_collection3d
    else:
        from matplotlib.collections import LineCollection as Lines

        ax = fig.add_subplot()
        add = ax.add_collection
    add(Lines(outline, colors="0.6", linewidths=0.6))
    beams = Lines(segments[order], linewidths=widths[order])
    if values is None:
        beams.set_color(colours)
    else:
        beams.set_array(values[order])
        beams.set_cmap(cmap)
        beams.set_norm(norm)
    beams.set_rasterized(len(owners) > VECTOR_BEAMS)
    add(beams)
    lo = np.minimum(segments.reshape(-1, dim).min(axis=0), outline.reshape(-1, dim).min(axis=0))
    hi = np.maximum(segments.reshape(-1, dim).max(axis=0), outline.reshape(-1, dim).max(axis=0))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    if dim == 3:
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)))
    else:
        ax.set_aspect("equal")
    ax.set_axis_off()
    if values is not None:
        fig.colorbar(beams, ax=ax, shrink=0.75, label=label)
    if title:
        ax.set_title(title, fontsize=10)
    return fig


# -- textures ------------------------------------------------------------------------------


def _indices_name(indices, plane: bool) -> str:
    """The name of a family of planes or directions: ``{111}``, ``<110>``, a bar over
    the negative indices."""
    text = "".join(f"{-v}̅" if v < 0 else str(v) for v in indices)
    return f"{{{text}}}" if plane else f"⟨{text}⟩"


def pole_figures_figure(figures, *, axes=("x", "y"), cmap: str = "viridis",
                        vmax: float | None = None, title: str | None = None):  # fmt: skip
    """Pole figures side by side (:func:`morphogen.texture.pole_figure`, one or several)
    on one colour range, in multiples of a random texture: the disc of each projection
    with the names ``axes`` of the first and second axes of the sample on its rim, the
    family and the largest density above it."""
    from morphogen.texture import PoleFigure

    shown_figures = [figures] if isinstance(figures, PoleFigure) else list(figures)
    if not shown_figures:
        raise ValueError("no pole figure to draw")
    top = float(vmax) if vmax else max(f.maximum for f in shown_figures)
    fig = _figure(3.0 * len(shown_figures) + 0.9, 3.3)
    circle = np.linspace(0.0, 2.0 * np.pi, 361)
    drawn, shown = [], None
    for k, f in enumerate(shown_figures):
        ax = fig.add_subplot(1, len(shown_figures), k + 1)
        shown = ax.pcolormesh(f.x, f.x, f.density.T, cmap=cmap, vmin=0.0, vmax=top,
                              shading="nearest", rasterized=True)  # fmt: skip
        ax.plot(np.cos(circle), np.sin(circle), color="black", linewidth=0.8)
        ax.set_aspect("equal")
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        ax.set_axis_off()
        ax.text(1.04, 0.0, axes[0], ha="left", va="center", fontsize=8)
        ax.text(0.0, 1.04, axes[1], ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{_indices_name(f.indices, f.plane)}   max {f.maximum:.1f}", fontsize=9)
        drawn.append(ax)
    bar = fig.colorbar(shown, ax=drawn, shrink=0.8)
    bar.set_label("multiples of a random texture", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def _sector_outline(lattice: str, n: int = 60) -> np.ndarray:
    """The outline of the fundamental sector of a lattice in the plane of an inverse pole
    figure (:func:`morphogen.texture.sector_points`)."""
    from morphogen.texture import SECTOR_CORNERS, sector_points

    corners = [np.array(c, dtype=float) for c, _ in SECTOR_CORNERS[lattice]]
    points = []
    for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
        t = np.linspace(0.0, 1.0, n)[:, None]
        points.append(sector_points((1.0 - t) * a / np.linalg.norm(a) + t * b / np.linalg.norm(b),
                                    lattice))  # fmt: skip
    return np.concatenate(points)


def _sector_names(ax, lattice: str, fontsize: float = 8) -> None:
    from morphogen.texture import SECTOR_CORNERS, sector_points

    places = (("right", "top"), ("left", "top"), ("left", "bottom"))
    for (corner, name), (ha, va) in zip(SECTOR_CORNERS[lattice], places, strict=True):
        x, y = sector_points(np.array(corner), lattice)[0]
        text = "".join(f"{c}̅" if minus else c for minus, c in _signed(name))
        ax.text(x, y, f" {text} ", ha=ha, va=va, fontsize=fontsize)


def _signed(name: str):
    """The digits of indices written with minus signs, each with whether it is negative."""
    minus = False
    for c in name:
        if c == "-":
            minus = True
            continue
        yield minus, c
        minus = False


def ipf_key(ax, lattice: str = "cubic", *, resolution: int = 240) -> None:
    """Draws the colour key of the inverse pole figure maps of a lattice on ``ax``: the
    fundamental sector with the colour of each direction of the crystal
    (:func:`morphogen.texture.direction_colours`) and the names of its corners."""
    from morphogen.texture import _sector_directions, direction_colours

    outline = _sector_outline(lattice)
    width, height = outline[:, 0].max(), outline[:, 1].max()
    x = np.linspace(0.0, width, resolution)
    y = np.linspace(0.0, height, max(3, round(resolution * height / width)))
    grid = np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1).reshape(-1, 2)
    directions, inside = _sector_directions(grid, lattice)
    rgba = np.zeros((len(grid), 4))
    rgba[inside, :3] = direction_colours(directions[inside], lattice)
    rgba[inside, 3] = 1.0
    ax.imshow(rgba.reshape(len(y), len(x), 4), origin="lower", extent=(0.0, width, 0.0, height),
              interpolation="bilinear")  # fmt: skip
    ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(-0.08 * width, 1.12 * width)
    ax.set_ylim(-0.12 * height, 1.15 * height)
    ax.set_axis_off()
    _sector_names(ax, lattice, 7)


def inverse_pole_figure_figure(figure, *, points=None, cmap: str = "viridis",
                               vmax: float | None = None, title: str | None = None):  # fmt: skip
    """An inverse pole figure (:func:`morphogen.texture.inverse_pole_figure`): its
    density over the fundamental sector, in multiples of a random texture, with the
    names of the corners; ``points``: directions of crystals ``(n, 3)`` drawn over it
    (:func:`morphogen.texture.inverse_poles`: the crystals one by one, or the way of one
    of them along a loading path)."""
    from morphogen.texture import sector_points

    outline = _sector_outline(figure.lattice)
    fig = _figure(4.6, 3.6 if figure.lattice == "cubic" else 2.9)
    ax = fig.add_subplot(1, 1, 1)
    shown = ax.pcolormesh(figure.x, figure.y, figure.density.T, cmap=cmap, vmin=0.0,
                          vmax=float(vmax) if vmax else figure.maximum, shading="nearest",
                          rasterized=True)  # fmt: skip
    ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=0.8)
    if points is not None:
        p = sector_points(points, figure.lattice)
        ax.plot(p[:, 0], p[:, 1], ".", color="white", markersize=3, markeredgecolor="black",
                markeredgewidth=0.3)  # fmt: skip
    ax.set_aspect("equal")
    width, height = outline[:, 0].max(), outline[:, 1].max()
    ax.set_xlim(-0.1 * width, 1.15 * width)
    ax.set_ylim(-0.12 * height, 1.15 * height)
    ax.set_axis_off()
    _sector_names(ax, figure.lattice)
    bar = fig.colorbar(shown, ax=ax, shrink=0.8)
    bar.set_label("multiples of a random texture", fontsize=8)
    d = figure.direction
    name = {(1.0, 0.0, 0.0): "x", (0.0, 1.0, 0.0): "y", (0.0, 0.0, 1.0): "z"}.get(
        tuple(round(v, 9) + 0.0 for v in d), f"({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f})"
    )
    ax.set_title(title or f"crystal directions along {name}   max {figure.maximum:.1f}", fontsize=9)
    return fig


def _boundary_lines(labels, step, height: float):
    """The sides between points of different grains of a map, as one broken line (NaN
    between its segments): the grain boundaries, in the axes of the image."""
    lab = np.asarray(labels)
    dx, dy = step
    rows, cols = np.nonzero(lab[:, 1:] != lab[:, :-1])  # vertical sides
    xs = np.stack([(cols + 1) * dx, (cols + 1) * dx, np.full(len(cols), np.nan)], axis=1)
    ys = np.stack(
        [height - rows * dy, height - (rows + 1) * dy, np.full(len(rows), np.nan)], axis=1
    )
    rows2, cols2 = np.nonzero(lab[1:, :] != lab[:-1, :])  # horizontal sides
    xs2 = np.stack([cols2 * dx, (cols2 + 1) * dx, np.full(len(cols2), np.nan)], axis=1)
    y = height - (rows2 + 1) * dy
    ys2 = np.stack([y, y, np.full(len(rows2), np.nan)], axis=1)
    return np.concatenate([xs.ravel(), xs2.ravel()]), np.concatenate([ys.ravel(), ys2.ravel()])


def ipf_map_figure(ebsd, direction=(0.0, 0.0, 1.0), *, grains=None, title: str | None = None):
    """An orientation map (:class:`morphogen.io.ebsd.EBSDMap`) in the colours of the
    direction of each crystal along a direction of the sample (the third axis by
    default: the normal of the map), black where it was not indexed, in the axes of its
    image; with the boundaries of its ``grains`` (:meth:`~morphogen.io.ebsd.EBSDMap.grains`)
    and the colour key of its lattice."""
    width, height = ebsd.size
    fig = _figure(6.6, max(2.4, 5.0 * height / width + 0.6))
    ax, key = fig.subplots(1, 2, width_ratios=(5.0, 1.0))
    ax.imshow(ebsd.colours(direction), extent=(0.0, width, 0.0, height), interpolation="nearest")
    if grains is not None:
        ax.plot(*_boundary_lines(grains.labels, ebsd.step, height), color="black", linewidth=0.5)
    ax.set_xlabel(f"x ({ebsd.unit})", fontsize=8)
    ax.set_ylabel(f"y ({ebsd.unit})", fontsize=8)
    ax.tick_params(labelsize=7)
    present = [p.lattice for k, p in enumerate(ebsd.phases, start=1) if np.any(ebsd.phase == k)]
    hexagonal = present and all(name in ("hexagonal", "6/m") for name in present)
    ipf_key(key, "hexagonal" if hexagonal else "cubic")
    if title:
        ax.set_title(title, fontsize=10)
    return fig


def disorientation_figure(angles, weights=None, *, lattice: str = "cubic", bins: int = 21,
                          title: str | None = None):  # fmt: skip
    """The distribution of disorientations (degrees; ``weights``: the lengths of the
    boundaries, :func:`morphogen.texture.boundary_disorientations`) as a histogram, with
    for cubic crystals the density of crystals at random (Mackenzie)."""
    from morphogen.texture import MACKENZIE_ANGLES, mackenzie

    top = {"cubic": 63.0, "m-3": 63.0, "hexagonal": 94.0}.get(lattice, 180.0)
    fig = _figure(4.6, 3.2)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(np.asarray(angles, dtype=float), bins=bins, range=(0.0, top), weights=weights,
            density=True, color="#7a9cc6", edgecolor="white", linewidth=0.5,
            label="boundaries")  # fmt: skip
    if lattice == "cubic":
        t = np.linspace(0.0, MACKENZIE_ANGLES[3], 400)
        ax.plot(t, mackenzie(t), color="black", linewidth=1.0, label="at random (Mackenzie)")
    ax.set_xlabel("disorientation (degrees)", fontsize=9)
    ax.set_ylabel("density (1/degree)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=10)
    return fig


#: The quantities of the seeds and of their grains, and their labels
SEED_QUANTITIES = {"diameter": "equivalent diameter", "aspect_ratio": "ratio a/b",
                   "flatness": "ratio b/c"}  # fmt: skip


def seed_figure(micro, comparison: dict | None = None, *, title: str | None = None):
    """The grains of seeds against their seeds (the figures of the verification of
    MicroStructPy, :meth:`~morphogen.models.seeded.SeededTessellation.seed_comparison`): for
    each quantity (the equivalent diameter, the ratios a/b and b/c — a/b alone in the
    plane), the distribution functions of the grains of each phase (solid), of their seeds
    (dotted) and the law of the seeds of the phase (thin, black); the angles between the
    distinguished axes of the grains and of their seeds; the fractions of the volume of the
    phases, asked and realized. The crystals that the faces of the box do not cut."""
    cmp = micro.seed_comparison() if comparison is None else comparison
    names = [q for q in SEED_QUANTITIES if f"seed_{q}" in cmp]
    model = getattr(micro, "model", None)
    laws = {}
    if model is not None and hasattr(model, "families"):
        laws = {f.phase: f for f in model.families()}
    fig = _figure(3.0 * (len(names) + 2), 3.1)
    axes = fig.subplots(1, len(names) + 2)
    use = np.asarray(cmp["inside"], dtype=bool)
    phases = np.asarray(cmp["phase"])
    shown = [int(k) for k in np.unique(phases[use])]
    for ax, name in zip(axes, names, strict=False):
        grain = np.asarray(cmp[f"grain_{name}"])
        seed = np.asarray(cmp[f"seed_{name}"])
        both = np.concatenate([grain[use], seed[use]])
        both = both[np.isfinite(both)]
        if not len(both):
            continue
        # the bulk of the values: a few slivers of grains do not stretch the axis
        low = 0.9 * float(both.min())
        high = 1.1 * max(float(np.quantile(grain[use & np.isfinite(grain)], 0.98)),
                         float(np.nanmax(seed[use])))  # fmt: skip
        t = np.linspace(low, high, 300)
        for k in shown:
            colour = PALETTE[(k + 1) % len(PALETTE)] / 255.0
            mine = use & (phases == k)
            for values, style in ((grain[mine], "-"), (seed[mine], ":")):
                x = np.sort(values[np.isfinite(values)])
                if len(x):
                    x = np.concatenate([[low], x, [high]])
                    y = np.concatenate([[0.0], np.arange(1, len(x) - 1) / (len(x) - 2), [1.0]])
                    ax.step(x, y, where="post", color=colour, linestyle=style, linewidth=1.4)
            law = getattr(laws.get(k), name, None)
            if law is not None:
                ax.plot(t, law.cdf(t), color="black", linewidth=0.8)
        ax.set_xlim(low, high)
        ax.set_xlabel(SEED_QUANTITIES[name], fontsize=9)
        ax.set_ylabel("distribution function", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_ylim(0.0, 1.02)
    angle = np.asarray(cmp["axis_angle"])[use]
    angle = angle[np.isfinite(angle)]
    ax = axes[len(names)]
    if len(angle):
        ax.hist(angle, bins=18, range=(0.0, max(10.0, float(angle.max()))), color="#7a9cc6",
                edgecolor="white", linewidth=0.5)  # fmt: skip
    ax.set_xlabel("angle of the axes (degrees)", fontsize=9)
    ax.set_ylabel("grains", fontsize=9)
    ax.tick_params(labelsize=8)
    ax = axes[-1]
    rows = sorted(cmp["fractions"].items())
    x = np.arange(len(rows))
    asked = [a for _, (_, a) in rows]
    ax.bar(x - 0.2, np.nan_to_num(asked, nan=0.0), width=0.4, color="#c8c8c8", label="asked")
    ax.bar(x + 0.2, [r for _, (r, _) in rows], width=0.4, color="#4c72b0", label="realized")
    ax.set_xticks(x, [f"phase {k}" for k, _ in rows], fontsize=8)
    ax.set_ylabel("fraction of the volume", fontsize=9)
    ax.set_ylim(0.0, 1.3 * max([r for _, (r, _) in rows] + list(np.nan_to_num(asked)) + [0.1]))
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def save_gif(frames, path, *, fps: int = 8, kind: str = "auto", scale: int = 4) -> None:
    """Animated GIF of 2D frames (Pillow, installed with matplotlib)."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("animations need Pillow: pip install 'morphogen[plot]'") from exc
    from morphogen.viz.png import to_rgb, upscale

    frames = [np.asarray(f) for f in frames]
    if not frames:
        raise ValueError("no frame to animate")
    lo = min(float(f.min()) for f in frames)
    hi = max(float(f.max()) for f in frames)
    images = [Image.fromarray(upscale(to_rgb(f, kind, lo, hi), scale)) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
