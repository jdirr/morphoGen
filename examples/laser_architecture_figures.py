"""The figures of docs/laser.md §14 from the results of ``laser_morphologies.py``,
``laser_crack_paths.py`` (``--map``, and the catalogue saved as
``results/laser_crack_paths_catalogue.npz``) and ``laser_stiffness.py``.

Usage: python examples/laser_architecture_figures.py [results folder] [figures folder]
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np

INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
HARD, SOFT = "#3d3d3a", "#f0efec"
CRACK = "#e34948"
#: the classes of morphologies, a categorical colour and a marker each
CLASSES = {
    "tracks": ("#2a78d6", "o"),
    "networks": ("#eb6834", "s"),
    "inclusions": ("#1baf7a", "^"),
    "soft walls": ("#6250d6", "D"),
    "grains of tracks": ("#e87ba4", "v"),
}
CLASS_OF = {
    "straight tracks": "tracks",
    "corrugated tracks": "tracks",
    "random laminate": "tracks",
    "crossed tracks": "networks",
    "hexagonal honeycomb": "networks",
    "Voronoi walls": "networks",
    "Wang tiles (pipes)": "networks",
    "Delaunay fibre network": "networks",
    "Gaussian excursion": "inclusions",
    "Boolean discs": "inclusions",
    "Turing pattern": "inclusions",
    "coarsened automaton": "inclusions",
    "Johnson-Mehl grains": "inclusions",
    "Voronoi cells, soft walls": "soft walls",
    "nacre (bricks)": "soft walls",
    "meta-polycrystal": "grains of tracks",
}
BLUES = ("#86b6ef", "#3987e5", "#184f95")


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    names = {f.name for f in font_manager.fontManager.ttflist}
    sans = [n for n in ("Inter", "Segoe UI", "DejaVu Sans") if n in names]
    plt.rcParams["font.family"] = sans[:1] or ["sans-serif"]
    return plt


def _read(path: Path) -> list:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def morphologies(res: Path, out: Path) -> None:
    """The mosaic of the designs and the maps of the sheets they make."""
    from matplotlib.colors import ListedColormap

    plt = _plt()
    designs = dict(np.load(res / "laser_morphologies.npz"))
    rows = _read(res / "laser_morphologies.csv")
    fig = plt.figure(figsize=(12.0, 6.6), layout="constrained")
    sub = fig.subfigures(1, 2, width_ratios=[1.05, 1.0])
    axes = sub[0].subplots(4, 4)
    cmap = ListedColormap([SOFT, HARD])
    for ax, (name, d) in zip(axes.ravel(), designs.items(), strict=False):
        ax.imshow(d, cmap=cmap, origin="lower", interpolation="nearest", vmin=0, vmax=1)
        colour = CLASSES[CLASS_OF[name]][0]
        ax.set_title(f"{name}\n{100 * d.mean():.0f} % treated", fontsize=7.5, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(colour)
            s.set_linewidth(2.0)
    sub[0].suptitle(
        "(a) the 2D models of morphoGen on a cell of 24 mm, opened by a spot of 1 mm",
        fontsize=9.5,
        color=INK,
        x=0.02,
        ha="left",
    )
    b, c = sub[1].subplots(2, 1)
    untreated = next(r for r in rows if r["design"] == "untreated")
    for hard, filled in (("martensite", True), ("tempered", False)):
        for cls, (colour, marker) in CLASSES.items():
            pts = [r for r in rows if r["hard"] == hard and CLASS_OF.get(r["design"]) == cls]
            kw = {"color": colour, "marker": marker, "s": 34, "zorder": 3}
            if not filled:
                kw = {**kw, "facecolors": "white", "edgecolors": colour, "linewidths": 1.4}
                kw.pop("color")
            lab = cls if hard == "martensite" else None
            b.scatter(
                [100 * r["failure_min"] for r in pts], [r["rm_mean"] for r in pts], label=lab, **kw
            )
            c.scatter([r["rp_spread"] for r in pts], [r["energy_mean"] for r in pts], **kw)
    b.scatter([100 * untreated["failure_min"]], [untreated["rm_mean"]], marker="x", color=INK, s=40)
    b.annotate(
        "untreated",
        (100 * untreated["failure_min"], untreated["rm_mean"]),
        xytext=(-12, 6),
        textcoords="offset points",
        fontsize=7.5,
        color=MUTED,
    )
    b.set_xlabel("failure strain, the least over 0°, 45° and 90° (%)")
    b.set_ylabel("tensile strength, mean over the directions (MPa)")
    _style(b, "(b) the sheets: strength against ductility (filled: martensite; open: tempered)")
    b.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK)
    c.axhline(untreated["energy_mean"], color=MUTED, lw=1, ls=":")
    c.annotate(
        "untreated",
        (0.0, untreated["energy_mean"]),
        xytext=(2, 3),
        textcoords="offset points",
        fontsize=7.5,
        color=MUTED,
    )
    c.set_xlabel("spread of the yield strength over the directions (max − min over the mean)")
    c.set_ylabel("energy to failure (MJ/m³)")
    _style(c, "(c) anisotropy against the energy absorbed")
    fig.savefig(out / "laser_morphologies.png", dpi=150, facecolor="white")
    plt.close(fig)


def _paths(npz: Path) -> dict:
    z = np.load(npz)
    names = list(z["names"])
    return {
        str(n): (z[f"cells_{k}"], z[f"path_{k}"], z[f"disp_{k}"], z[f"force_{k}"])
        for k, n in enumerate(names)
    }


def cracks(res: Path, out: Path) -> None:
    """Crack paths over their designs and the map of steering."""
    from matplotlib.colors import ListedColormap

    plt = _plt()
    box = (13.0, 30.0, 12.0)
    steer = _paths(res / "laser_crack_steering.npz")
    cat = _paths(res / "laser_crack_paths_catalogue.npz")
    shown = [
        ("untreated", steer, "untreated"),
        ("soft corridor at 30°, c = 3", steer, "corridor 30 deg, 2 px, c 3"),
        ("soft corridor at 60°, c = 3", steer, "corridor 60 deg, 2 px, c 3"),
        ("arc turning by 90°, c = 4", steer, "arc corridor turning by 90 deg, c 4"),
        ("corridor ending after 6 mm, c = 3", steer, "corridor at 60 deg ending after 6 mm, c 3"),
        ("hard walls along a corridor", cat, "tough hard walls along a corridor at 30 deg"),
        ("staggered bricks (nacre)", cat, "staggered tough bricks (nacre)"),
        ("tough Voronoi walls", cat, "tough Voronoi walls"),
    ]
    fig = plt.figure(figsize=(12.0, 6.4), layout="constrained")
    sub = fig.subfigures(1, 2, width_ratios=[1.45, 1.0])
    axes = sub[0].subplots(2, 4)
    cmap = ListedColormap([SOFT, HARD])
    for ax, (title, src, key) in zip(axes.ravel(), shown, strict=True):
        if key not in src:
            ax.set_axis_off()
            continue
        cells, path, _, _ = src[key]
        ax.imshow(
            cells,
            cmap=cmap,
            origin="lower",
            extent=(box[0], box[1], -box[2], box[2]),
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.plot([box[0], 15.0], [0.0, 0.0], color=INK, lw=2.0)  # the initial crack
        if len(path):
            ax.scatter(path[:, 0], path[:, 1], s=4, color=CRACK, zorder=3)
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(-box[2], box[2])
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    sub[0].suptitle(
        "(a) the crack grown through the ligament of a CT specimen (W = 30 mm; dark: treated)",
        fontsize=9.5,
        color=INK,
        x=0.02,
        ha="left",
    )
    m, w = sub[1].subplots(2, 1)
    for k, c in enumerate((2.0, 3.0, 4.0)):
        xs, ys = [], []
        for angle in (15.0, 30.0, 45.0, 60.0):
            key = f"corridor {angle:.0f} deg, 2 px, c {c:g}"
            if key in steer and len(steer[key][1]) > 2:
                path = steer[key][1]
                ahead = path[path[:, 0] > 15.0]
                slope = math.degrees(math.atan(np.polyfit(ahead[:, 0], ahead[:, 1], 1)[0]))
                xs.append(angle)
                ys.append(slope)
        m.plot(xs, ys, color=BLUES[k], lw=2, marker="o", ms=5, label=f"c = {c:g}")
    m.plot([0, 65], [0, 65], color=MUTED, lw=1, ls=":")
    m.annotate("the crack follows the corridor", (40, 41), rotation=33, fontsize=7.5, color=MUTED)
    m.set_xlabel("angle of the soft corridor to the initial crack (°)")
    m.set_ylabel("direction of the crack (°)")
    _style(m, "(b) steering: soft corridors 2 mm wide in a hard sheet")
    m.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    palette = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
    for (title, src, key), colour in zip(shown[:5], palette, strict=True):
        if key in src:
            _, _, d, f = src[key]
            w.plot(d, f / 1000.0, lw=1.6, color=colour, label=title)
    w.set_xlabel("load-line displacement (mm)")
    w.set_ylabel("force (kN)")
    _style(w, "(c) the force against the opening")
    w.legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.savefig(out / "laser_crack_paths.png", dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/_static")
    out.mkdir(parents=True, exist_ok=True)
    morphologies(res, out)
    if (res / "laser_crack_steering.npz").exists():
        cracks(res, out)


if __name__ == "__main__":
    main()
