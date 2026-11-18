"""Figures of laser-architectured sheets (matplotlib, headless): the hardness map of the
parallel part with the tracks, the engineering curves of several designs, the plastic
strain on the deformed specimen, the history of an optimization."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _figure(width: float = 6.0, height: float = 4.0):
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("figures need matplotlib: pip install 'morphogen[plot]'") from exc
    return Figure(figsize=(width, height), dpi=130, layout="constrained")


def hardness_figure(maps: dict, path=None, *, title: str = ""):
    """The hardness through the thickness (mean, and the upper and lower layers) of the
    parallel part, the axes of the tracks drawn over it."""
    h = maps["hardness"]
    x, y = maps["x"], maps["y"]
    views = [("mean", h.mean(axis=2))]
    if h.shape[2] > 1:
        views += [("upper layer", h[:, :, 0]), ("lower layer", h[:, :, -1])]
    span = (x[-1] - x[0]) / max(y[-1] - y[0], 1e-9)
    fig = _figure(7.0, 1.0 + len(views) * 7.0 / max(span, 1.0) + 0.4 * len(views))
    axes = fig.subplots(len(views), 1, squeeze=False)[:, 0]
    lo, hi = float(np.min(h)), float(np.max(h))
    extent = (
        x[0] - 0.5 * (x[1] - x[0]),
        x[-1] + 0.5 * (x[1] - x[0]),
        y[0] - 0.5 * (y[1] - y[0]),
        y[-1] + 0.5 * (y[1] - y[0]),
    )
    im = None
    for ax, (name, img) in zip(axes, views, strict=True):
        im = ax.imshow(
            img.T,
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=lo,
            vmax=hi,
            aspect="equal",
            interpolation="nearest",
        )
        if path is not None:
            for t in path:
                ax.plot(t.points[:, 0], t.points[:, 1], color="cyan", lw=0.6, alpha=0.7)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_ylabel("y (mm)")
        ax.set_title(f"{title} hardness, {name}".strip(), fontsize=9)
    axes[-1].set_xlabel("x (mm)")
    fig.colorbar(im, ax=list(axes), label="HV", shrink=0.9)
    return fig


def curves_figure(designs: dict):
    """Engineering curves of designs (``name -> SheetResult``; in tension their maxima
    marked, in bending the load against the deflection), each direction of a design
    with several."""
    fig = _figure(6.0, 4.0)
    ax = fig.add_subplot()
    bending = False
    for name, res in designs.items():
        for d, r in (res.directions or {None: res}).items():
            label = name if d is None else f"{name} {d:g}°"
            if r.tensile is not None:
                t = r.tensile
                (line,) = ax.plot(100.0 * t.strain, t.stress, lw=1.4, label=label)
                ax.plot(
                    100.0 * t.uniform_elongation,
                    t.tensile_strength,
                    "o",
                    color=line.get_color(),
                    ms=4,
                )
            elif r.bending is not None:
                bending = True
                ax.plot(r.bending.deflection, r.bending.load, lw=1.4, label=label)
    if bending:
        ax.set_xlabel("deflection (mm)")
        ax.set_ylabel("load (N)")
    else:
        ax.set_xlabel("engineering strain (%)")
        ax.set_ylabel("engineering stress (MPa)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0.0)
    return fig


def strain_figure(res, *, magnify: float = 1.0, title: str = ""):
    """The equivalent plastic strain (mean over the Gauss points and layers of each
    element) on the deformed mesh at the end of the test."""
    from matplotlib.collections import PolyCollection

    fields = res.tensile.fields
    if not fields:
        raise ValueError("the fields were not kept (evaluate(keep_fields=True))")
    mesh = res.tensile.fields.get("mesh")
    nodes, quads = mesh
    disp = fields["displacement"]
    xy = nodes + magnify * disp
    values = fields["plastic_strain"].mean(axis=1)
    fig = _figure(7.0, 2.4)
    ax = fig.add_subplot()
    coll = PolyCollection(xy[quads], array=values, cmap="viridis", edgecolors="none")
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{title} equivalent plastic strain".strip(), fontsize=9)
    fig.colorbar(coll, ax=ax, label="p", shrink=0.9)
    return fig


def history_figure(opt_out):
    """The objective of each evaluation and the best feasible one so far; with two
    parameters, the evaluations in their plane, coloured by objective."""
    hist = opt_out.history
    names = list(opt_out.values)
    two = len(names) == 2
    fig = _figure(9.0 if two else 6.0, 3.6)
    axes = fig.subplots(1, 2 if two else 1, squeeze=False)[0]
    ev = np.array([h["evaluation"] for h in hist])
    obj = np.array([h["objective"] for h in hist], dtype=float)
    feas = np.array([h["feasible"] for h in hist])
    best = np.maximum.accumulate(np.where(feas & np.isfinite(obj), obj, -np.inf))
    ax = axes[0]
    ax.plot(ev[feas], obj[feas], "o", ms=4, label="feasible")
    ax.plot(ev[~feas], obj[~feas], "x", ms=4, label="infeasible")
    ok = np.isfinite(best)
    ax.step(ev[ok], best[ok], where="post", color="k", lw=1.0, label="best feasible")
    ax.set_xlabel("evaluation")
    ax.set_ylabel(opt_out.settings.get("objective", "objective"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if two:
        ax = axes[1]
        a = np.array([h[names[0]] for h in hist], dtype=float)
        b = np.array([h[names[1]] for h in hist], dtype=float)
        sc = ax.scatter(
            a, b, c=obj, cmap="viridis", s=24, marker="o", edgecolors=np.where(feas, "k", "r")
        )
        ax.plot(opt_out.values[names[0]], opt_out.values[names[1]], "*", color="r", ms=12)
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        fig.colorbar(sc, ax=ax, label=opt_out.settings.get("objective", "objective"))
    return fig


def write_figures(case, designs: dict, opt_out, out: Path, *, resolution: float = 0.05) -> list:
    """Write ``hardness.png`` (the best design), ``curves.png``, ``strain.png`` and, after
    an optimization, ``history.png``; returns the paths."""
    from morphogen.laser.run import hardness_map

    out = Path(out)
    best_name = "optimum" if "optimum" in designs else "pattern"
    best = designs[best_name]
    written = []
    maps = hardness_map(case, best.path, resolution)
    fig = hardness_figure(maps, best.path, title=best_name)
    fig.savefig(out / "hardness.png")
    written.append(out / "hardness.png")
    fig = curves_figure(designs)
    fig.savefig(out / "curves.png")
    written.append(out / "curves.png")
    if best.tensile is not None and best.tensile.fields.get("mesh") is not None:
        fig = strain_figure(best, title=best_name)
        fig.savefig(out / "strain.png")
        written.append(out / "strain.png")
    if opt_out is not None:
        fig = history_figure(opt_out)
        fig.savefig(out / "history.png")
        written.append(out / "history.png")
    return written


def cells_figure(grid, designs: dict, curves: dict, path=None, *, title: str = ""):
    """Cellular designs (``{name: design}`` on ``grid``, the treated cells dark) above
    their engineering curves (``{name: (strain, stress, failure strain)}``, the failure
    marked)."""
    n = len(designs)
    fig = _figure(7.0, 1.0 + 0.62 * n * 7.0 * grid.width / grid.length + 2.6)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[1.0] * n + [3.2])
    colors = ["#1f5f99", "#c0392b", "#2e8b57", "#8e44ad", "#d35400", "#7f8c8d"]
    for k, (name, design) in enumerate(designs.items()):
        ax = fig.add_subplot(gs[k])
        ax.imshow(
            np.asarray(design, dtype=float),
            origin="lower",
            cmap="Greys",
            vmin=0.0,
            vmax=1.6,
            extent=(grid.x0, grid.x0 + grid.length, grid.y0, grid.y0 + grid.width),
            interpolation="nearest",
        )
        ax.set_yticks([])
        if k < n - 1:
            ax.set_xticks([])
        ax.set_ylabel(
            name, rotation=0, ha="right", va="center", fontsize=8, color=colors[k % len(colors)]
        )
    ax = fig.add_subplot(gs[n])
    for k, (name, (e, s, ef)) in enumerate(curves.items()):
        c = colors[k % len(colors)]
        e, s = np.asarray(e), np.asarray(s)
        keep = e <= ef + 1e-12
        ax.plot(e[keep], s[keep], color=c, lw=1.6, label=name)
        ax.plot(e[~keep], s[~keep], color=c, lw=0.8, ls=":")
        sf = np.interp(ef, e, s)
        ax.plot([ef], [sf], "x", color=c, ms=7, mew=2)
    ax.set_xlabel("engineering strain")
    ax.set_ylabel("engineering stress (MPa)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=9)
    if path is not None:
        fig.savefig(Path(path))
    return fig
