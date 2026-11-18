"""Figures as files: bitmaps (``.png``, ``.tif``/``.tiff``), vector graphics (``.svg``,
``.pdf``, ``.eps``) and TikZ (``.tex``, ``.tikz``): the pgfplots code of a matplotlib
figure, for LaTeX documents.

The TikZ code has one ``axis`` per axes of the figure, placed as in the figure, with its
labels, title, limits, scales, ticks and legend: the lines (``plot``) as coordinates, the
markers of scatter plots, line collections (frames, contours) as coordinate lists cut by
``nan`` (``unbounded coords=jump``), filled rectangles (``axhspan``, bars) and polygons as
filled paths, texts as nodes, and images (slices, maps) as PNG files beside the ``.tex``
(``\\addplot graphics``). Use it with ``\\usepackage{pgfplots}``,
``\\pgfplotsset{compat=1.18}``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BITMAP = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
VECTOR = (".svg", ".pdf", ".eps", ".ps")
TIKZ = (".tex", ".tikz", ".pgf")


def save_figure(fig, path, *, dpi: int = 200, transparent: bool = False) -> Path:
    """Save a matplotlib figure by the suffix of ``path``: bitmaps at ``dpi``, vector
    graphics, or the TikZ code of :func:`tikz` (its images beside it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in TIKZ:
        path.write_text(tikz(fig, path), encoding="utf-8")
    elif suffix in BITMAP or suffix in VECTOR:
        options = {"dpi": dpi, "transparent": transparent, "bbox_inches": "tight"}
        if suffix in (".tif", ".tiff"):
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **options)
    else:
        raise ValueError(f"save_figure: a suffix among {sorted(BITMAP + VECTOR + TIKZ)}")
    return path


def _number(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.6g}"


def _color(c) -> str:
    from matplotlib.colors import to_rgba

    r, g, b, _ = to_rgba(c)
    return f"{{rgb,1:red,{r:.4f};green,{g:.4f};blue,{b:.4f}}}"


def _alpha(c, alpha=None) -> float:
    from matplotlib.colors import to_rgba

    a = to_rgba(c)[3]
    return float(a if alpha is None else alpha * a)


def _tex(text: str) -> str:
    """Matplotlib text (mathtext between $) to LaTeX: special characters escaped outside
    math."""
    parts = str(text).replace(r"\mathdefault", "").split("$")
    out = []
    for k, part in enumerate(parts):
        if k % 2:
            out.append("$" + part + "$")
        else:
            for a, b in (
                ("\\", r"\textbackslash{}"),
                ("&", r"\&"),
                ("%", r"\%"),
                ("#", r"\#"),
                ("_", r"\_"),
            ):
                part = part.replace(a, b)
            out.append(part.replace("{", r"\{").replace("}", r"\}"))
    return "".join(out)


_DASHES = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dashdotted"}
_MARKS = {
    "o": "*",
    ".": "*",
    "s": "square*",
    "^": "triangle*",
    "v": "triangle*",
    "d": "diamond*",
    "D": "diamond*",
    "x": "x",
    "+": "+",
    "*": "star",
}


def _coordinates(x, y) -> str:
    return " ".join(
        f"({_number(a)},{_number(b)})" for a, b in zip(np.ravel(x), np.ravel(y), strict=True)
    )


def _axis(ax, path: Path, index: int, fig_size, origin) -> list[str]:
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection
    from matplotlib.image import AxesImage
    from matplotlib.patches import Polygon, Rectangle

    box = ax.get_position()
    width, height = box.width * fig_size[0], box.height * fig_size[1]
    x0, y0 = box.x0 * fig_size[0] - origin[0], box.y0 * fig_size[1] - origin[1]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    opts = [
        f"at={{({x0:.3f}cm,{y0:.3f}cm)}}",
        f"width={width:.3f}cm",
        f"height={height:.3f}cm",
        "scale only axis",
        f"xmin={_number(xmin)}",
        f"xmax={_number(xmax)}",
        f"ymin={_number(ymin)}",
        f"ymax={_number(ymax)}",
        "unbounded coords=jump",
    ]
    if ax.get_xscale() == "log":
        opts.append("xmode=log")
    if ax.get_yscale() == "log":
        opts.append("ymode=log")
    if ax.get_xlabel():
        opts.append(f"xlabel={{{_tex(ax.get_xlabel())}}}")
    if ax.get_ylabel():
        opts.append(f"ylabel={{{_tex(ax.get_ylabel())}}}")
    if ax.get_title():
        opts.append(f"title={{{_tex(ax.get_title())}}}")
    if not ax.axison:
        opts.append("hide axis")
    from matplotlib.ticker import FixedFormatter, FuncFormatter

    labels = [t.get_text() for t in ax.get_xticklabels()]
    explicit = isinstance(ax.xaxis.get_major_formatter(), (FixedFormatter, FuncFormatter))
    if explicit and any(labels):
        ticks = ax.get_xticks()
        opts.append("xtick={" + ",".join(_number(t) for t in ticks) + "}")
        opts.append("xticklabels={" + ",".join("{" + _tex(t) + "}" for t in labels) + "}")
    if ax.get_aspect() in ("equal", 1.0):
        opts.append("axis equal image")
    body = []
    for k, image in enumerate(im for im in ax.get_images() if isinstance(im, AxesImage)):
        name = f"{path.stem}-{index}-{k}.png"
        rgba = np.asarray(image.to_rgba(image.get_array(), bytes=True))
        if image.origin == "lower":  # the first row of the PNG is the top of the axes
            rgba = rgba[::-1]
        from PIL import Image

        Image.fromarray(rgba).save(path.with_name(name))
        e = image.get_extent()
        body.append(
            f"\\addplot graphics[xmin={_number(e[0])},xmax={_number(e[1])},"
            f"ymin={_number(min(e[2:]))},ymax={_number(max(e[2:]))}] {{{name}}};"
        )
    for patch in ax.patches:
        if isinstance(patch, (Rectangle, Polygon)):
            xy = patch.get_transform().transform(patch.get_path().vertices)
            xy = ax.transData.inverted().transform(xy)
            fill = patch.get_facecolor()
            body.append(
                f"\\fill[color={_color(fill)}, opacity={_alpha(fill):.3f}] "
                + " -- ".join(f"(axis cs:{_number(a)},{_number(b)})" for a, b in xy)
                + " -- cycle;"
            )
    for coll in ax.collections:
        if isinstance(coll, LineCollection):
            segs = coll.get_segments()
            colors = coll.get_colors() if len(coll.get_colors()) else ["k"]
            widths = coll.get_linewidths() if len(coll.get_linewidths()) else [1.0]
            xs, ys = [], []
            for s in segs:
                xs.extend([*s[:, 0], np.nan])
                ys.extend([*s[:, 1], np.nan])
            width = 0.0353 * widths[0]
            body.append(
                f"\\addplot[color={_color(colors[0])}, line width={width:.3f}cm, forget plot] "
                f"coordinates {{{_coordinates(xs, ys)}}};"
            )
        elif isinstance(coll, PathCollection):
            off = coll.get_offsets()
            colors = coll.get_facecolors() if len(coll.get_facecolors()) else ["C0"]
            body.append(
                f"\\addplot[only marks, mark=*, mark size=1pt, color={_color(colors[0])}] "
                f"coordinates {{{_coordinates(off[:, 0], off[:, 1])}}};"
            )
        elif isinstance(coll, PolyCollection):
            for poly, fc in zip(coll.get_paths(), coll.get_facecolors(), strict=False):
                body.append(
                    f"\\fill[color={_color(fc)}, opacity={_alpha(fc):.3f}] "
                    + " -- ".join(f"(axis cs:{_number(a)},{_number(b)})" for a, b in poly.vertices)
                    + " -- cycle;"
                )
    legend = ax.get_legend()
    labelled = {t.get_text() for t in legend.get_texts()} if legend else set()
    for line in ax.get_lines():
        x, y = line.get_data()
        if np.size(x) == 0:
            continue
        style = [
            f"color={_color(line.get_color())}",
            f"line width={0.0353 * line.get_linewidth():.3f}cm",
            _DASHES.get(line.get_linestyle(), "solid"),
        ]
        marker = line.get_marker()
        style.append(f"mark={_MARKS[marker]}" if marker in _MARKS else "mark=none")
        if line.get_linestyle() in ("None", "none", " ", ""):
            style.append("only marks")
        label = line.get_label()
        if label not in labelled:
            style.append("forget plot")
        body.append(f"\\addplot[{', '.join(style)}] coordinates {{{_coordinates(x, y)}}};")
        if label in labelled:
            body.append(f"\\addlegendentry{{{_tex(label)}}}")
    for text in ax.texts:
        x, y = text.get_position()
        body.append(f"\\node at (axis cs:{_number(x)},{_number(y)}) {{{_tex(text.get_text())}}};")
    return ["\\begin{axis}[" + ", ".join(opts) + "]", *body, "\\end{axis}"]


def tikz(fig, path=None) -> str:
    """The pgfplots code of a matplotlib figure (see the module); its images are written
    as PNG files beside ``path`` (by default ``figure.tex`` in the working directory)."""
    path = Path("figure.tex") if path is None else Path(path)
    size = fig.get_size_inches() * 2.54
    axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    origin = (
        min(ax.get_position().x0 for ax in axes) * size[0] if axes else 0.0,
        min(ax.get_position().y0 for ax in axes) * size[1] if axes else 0.0,
    )
    lines = [
        "% pgfplots code written by morphoGen; \\usepackage{pgfplots} \\pgfplotsset{compat=1.18}",
        "\\begin{tikzpicture}",
    ]
    for k, ax in enumerate(axes):
        lines.extend(_axis(ax, path, k, size, origin))
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"
