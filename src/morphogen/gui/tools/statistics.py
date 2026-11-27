"""The tools of the tab Statistics: the size of the representative volume element of a
morphology (Kanit et al. 2003, Dirrenberger et al. 2014), its morphological measures
against the theory of its model, and its correlation functions and distributions
(covariance, lineal path, chords, granulometry, pore sizes, Choquet capacities) with
the closed forms of its model when it has them."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from morphogen.gui.tools import Field, Watcher, tab, tool

tab(
    "statistics",
    "Statistics",
    "Statistics of a morphology of the workspace: the size of its representative volume "
    "element (RVE) by the ensemble or the window method, its morphological measures "
    "against the theory of its model, its correlation functions and distributions.",
    order=20,
)

#: The phases of an RVE study of moduli when the form gives none: a contrast of 10
DEFAULT_MATERIALS = {0: {"young": 1.0, "poisson": 0.3}, 1: {"young": 10.0, "poisson": 0.3}}
PROPERTIES = (
    ("volume_fraction", "volume fraction"),
    ("bulk_modulus", "apparent bulk modulus"),
    ("shear_modulus", "apparent shear modulus"),
    ("stiffness", "apparent stiffness C_ij and compliance S_ij (six loadings)"),
)
#: The properties of the form computed by homogenization
ELASTIC = ["bulk_modulus", "shear_modulus", "stiffness"]


def _model_of(item: dict, what: str = "an RVE study draws realizations") -> dict:
    if not item.get("model"):
        raise ValueError(
            f"{item['title']} is not the realization of a model: {what} of the model of a "
            "morphology generated in Generate"
        )
    return item["model"]


def _edge(item: dict) -> float:
    return float(np.max(item["micro"].domain.lengths))


def _markdown_tables(text: str) -> list[dict]:
    """The pipe tables of a markdown report, with the heading or bold line before each."""
    out, title, lines = [], "", text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#") or (line.startswith("**") and line.endswith("**")):
            title = line.strip("#* ").strip()
        rule = i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip())
        if line.startswith("|") and rule:
            columns = [c.strip() for c in line.strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append({"title": title, "columns": columns, "rows": rows, "count": len(rows)})
            continue
        i += 1
    return out


@tool(
    "rve",
    "statistics",
    "RVE size",
    "The size of the representative volume element of the model of a morphology for a "
    "property: by the ensemble method (independent realizations of several sizes) or by "
    "the window method (sub-volumes of one large realization), the variance D²(V) of the "
    "property fitted as K V^-γ (γ = 1: the integral range A₃ = K / D²_Z), and the volumes "
    "for the relative errors and numbers of realizations chosen (Kanit et al. 2003; "
    "Dirrenberger, Forest & Jeulin 2014). For the stiffness, the mean C̄ of each size, its "
    "compliance S = C̄⁻¹, the standard deviation of each component and the RVE size of "
    "each; by the window method (FFT), the tensor of a window is its mean mutual energy "
    "⟨ε⁽ᵏ⁾ : C : ε⁽ˡ⁾⟩ under the six unit strains of the realization.",
    [
        Field(
            "source",
            "item",
            label="morphology",
            required=True,
            needs=("model",),
            doc="a morphology generated from a model (its model, seed and periodicity)",
        ),
        Field(
            "method",
            "choice",
            "ensemble",
            choices=(
                ("ensemble", "ensemble: independent realizations of each size"),
                ("windows", "windows: sub-volumes of one large realization"),
            ),
        ),
        Field(
            "sizes",
            "floats",
            label="sizes",
            unit="length",
            when={"method": "ensemble"},
            doc="edges of the cubes (blank: 0.5, 1 and 1.5 times the edge of the morphology)",
        ),
        Field(
            "realizations",
            "ints",
            [20, 10, 5],
            when={"method": "ensemble"},
            doc="realizations of each size (one number: the same for all)",
        ),
        Field(
            "size",
            "float",
            label="size of the realization",
            unit="length",
            when={"method": "windows"},
            doc="blank: twice the edge of the morphology",
        ),
        Field(
            "window_sizes",
            "floats",
            when={"method": "windows"},
            unit="length",
            doc="edges of the windows (blank: 1/8, 1/4 and 1/2 of the realization)",
        ),
        Field(
            "shifts",
            "int",
            1,
            label="shifts per axis",
            when={"method": "windows"},
            doc="1: the windows tiling the realization from its origin; more: also the "
            "tilings shifted by k w / shifts along each axis, wrapping around a periodic "
            "realization (shifts³ tilings; the stiffness from the same solutions)",
        ),
        Field("property", "choice", "volume_fraction", choices=PROPERTIES),
        Field(
            "solver",
            "choice",
            "fft",
            choices=(("fft", "FFT (voxels)"), ("fe", "finite elements (conforming meshes)")),
            when={"property": ELASTIC},
        ),
        Field("materials", "materials", of="source", when={"property": ELASTIC}),
        Field(
            "voxel_size",
            "float",
            unit="length",
            doc="the same for every size (blank: 1/48 of the edge of the morphology)",
        ),
        Field("relative_errors", "floats", [0.01, 0.05]),
        Field("n_realizations", "ints", [1, 10], label="numbers of realizations"),
    ],
    order=5,
    menu="Representative volume",
)
def rve_size(p: dict, ctx) -> dict:
    from morphogen import pipeline
    from morphogen.config import load_config
    from morphogen.rve.scaling import _shift_count

    item = p["source"]
    model = _model_of(item)
    edge = _edge(item)
    periodic = bool(item["micro"].domain.periodic)  # that of the realizations
    voxel = p["voxel_size"] or edge / 48
    spec: dict = {
        "relative_errors": p["relative_errors"],
        "n_realizations": p["n_realizations"],
    }
    if p["method"] == "ensemble":
        sizes = p["sizes"] or [0.5 * edge, edge, 1.5 * edge]
        reps = p["realizations"] or [10]
        reps = reps * len(sizes) if len(reps) == 1 else reps
        if len(reps) != len(sizes):
            raise ValueError("realizations: one number, or one per size")
        spec.update(method="ensemble", sizes=sizes, realizations=reps)
        total = int(sum(reps))
    else:
        size = p["size"] or 2 * edge
        windows = p["window_sizes"] or [size / 8, size / 4, size / 2]
        spec.update(method="windows", size=size, window_sizes=windows)
        # the refusals of the API: a positive count, the shifts on a periodic realization
        shifts = _shift_count(p["shifts"], periodic)
        if shifts > 1:  # 1, the tiling from the origin: the configuration of before
            spec["shifts"] = shifts
        total = 1
    cfg: dict = {
        "model": model,
        "domain": {"periodic": periodic},
        "seed": item.get("seed") or 1,
        "voxels": {"voxel_size": voxel},
        "rve": spec,
        "outputs": {"directory": str(ctx.folder), "name": "rve", "tables": ["xlsx"]},
    }
    prop = p["property"]
    if prop != "volume_fraction":
        windows = p["method"] == "windows"
        if windows and (prop != "stiffness" or p["solver"] != "fft"):
            raise ValueError(
                "the window method studies the volume fraction (voxels) and the stiffness of "
                "its windows (FFT solver)"
            )
        materials = p["materials"] or DEFAULT_MATERIALS
        bc = "periodic" if periodic or windows else "kubc"
        cfg["homogenization"] = {"materials": materials, "method": p["solver"],
                                 "boundary_conditions": [bc], "properties": [prop]}  # fmt: skip
        if p["solver"] == "fe":
            cfg["mesh"] = {"size": 2 * voxel}
    config = load_config(cfg)
    with Watcher(ctx, ctx.folder / "records", "*.json", total, "realizations"):
        pipeline.run(config)
    if prop == "stiffness":
        _mean_tensors(ctx)
    report = (ctx.folder / "rve_report.md").read_text(encoding="utf-8")
    tables = _markdown_tables(report)
    for png in sorted(ctx.folder.glob("*.png")):
        ctx.figures.append({"title": png.stem.replace("_", " "), "url": ctx.url(png.name)})
    return {"tables": tables, "text": report if not tables else ""}


def _mean_tensors(ctx) -> None:
    """The tables of the mean stiffness ``Cbar`` of the largest volume of the study and
    of its compliance ``S = Cbar^-1`` (each boundary condition); the report gives them at
    every volume, with the standard deviation of each component (by the window method, of
    the largest windows; with shifts, C the mean of the windows of all their tilings)."""
    import json

    from morphogen.gui.tools import elastic_tables
    from morphogen.rve.report import BC_LABELS

    report = json.loads((ctx.folder / "rve_report.json").read_text(encoding="utf-8"))
    samples = "windows" if report.get("method") == "windows" else "realizations"
    for t in report.get("tensors", {}).values():
        if t["property"] != "stiffness":
            continue
        last = t["volumes"][-1]
        power = "²" if t.get("dimension", 3) == 2 else "³"  # a square (its area), a cube
        tilings = last.get("tilings") or 1  # n the windows of each tiling
        count = f"{last['n']} {samples}" + (f" x {tilings} tilings" if tilings > 1 else "")
        of = (f"at {last['edge']:.4g}{power}, C the mean of {count} "
              f"({BC_LABELS.get(t['bc'], t['bc'])})")  # fmt: skip
        elastic_tables(ctx, last["stiffness"], of, names=(f"stiffness_{t['bc']}",
                       f"compliance_{t['bc']}"), compliance=last["compliance"])  # fmt: skip


#: The measures offered (those of :data:`morphogen.analysis.measures.MEASURES`)
def _measure_choices():
    from morphogen.analysis.measures import MEASURES
    from morphogen.morphology.measures import EXACT_MEASURES

    return (*MEASURES, *EXACT_MEASURES)


#: The theory of the models beside the exact measures of their realizations
_EXACT_OF_THEORY = {
    "volume_fraction": ("exact_V_V", "exact_fraction"),
    "specific_surface": ("exact_S_V", "exact_specific_surface"),
    "mean_curvature_density": ("exact_M_V",),
    "euler_density": ("exact_chi_V",),
}


@tool(
    "measures",
    "statistics",
    "Morphological measures",
    "Measures of a morphology on a voxel image of it: phase fractions, specific surface "
    "(Crofton), connectivity, Minkowski functionals and Betti numbers, chords, "
    "thicknesses, tortuosity, fractal dimension, anisotropy (mean intercept lengths), "
    "minima of the covariance; or on its implicit geometry itself, without voxels (exact_*: "
    "the fraction by points, the specific surface by lines, the mean chords, the four "
    "Minkowski densities of a union of spheres exactly); with the closed forms of its model "
    "when known.",
    [
        Field("source", "item", label="morphology", required=True),
        Field("resolution", "int", 128, label="voxels per edge", doc="of the image measured"),
        Field(
            "measures",
            "choices",
            ["phase_fractions", "specific_surface", "connectivity"],
            choices=_measure_choices(),
        ),
    ],
    order=1,
    menu="Morphology",
)
def measures(p: dict, ctx) -> dict:
    from morphogen.analysis import measure
    from morphogen.gui.api import _voxels
    from morphogen.morphology.measures import EXACT_MEASURES, measure_exact

    micro = p["source"]["micro"]
    names = p["measures"] or []
    exact = [n for n in names if n in EXACT_MEASURES]
    voxel = [n for n in names if n not in EXACT_MEASURES]
    values = {}
    if voxel:
        _voxels(micro.domain, p["resolution"], 6.0, "image")
        ctx.progress(0.1, "voxel image")
        image = micro.voxelize(shape=p["resolution"])
        labels = micro.labels(grid=image.grid()) if getattr(micro, "has_labels", False) else None
        for k, name in enumerate(voxel):
            ctx.progress(0.1 + 0.6 * k / len(voxel), name)
            values.update(measure(image, [name], labels=labels))
    for k, name in enumerate(exact):  # on the implicit geometry, without voxels
        ctx.progress(0.7 + 0.3 * k / len(exact), name)
        values.update(measure_exact(micro, [name]))
    out = {"values": values}
    model = micro.model
    if model is not None:
        theory = {k: v for k, v in model.theory(micro.domain).items() if isinstance(v, float)}
        if theory:
            rows = []
            for k, v in theory.items():
                row = {"quantity": k, "theory": v, "measured": values.get(k)}
                known = [values[e] for e in _EXACT_OF_THEORY.get(k, ()) if e in values]
                if exact:
                    row["exact"] = known[0] if known else None
                rows.append(row)
            ctx.table(rows, "theory", "theory of the model")
    return out


CURVES = (
    ("covariance", "covariance C(h)"),
    ("lineal_path", "lineal path L(h)"),
    ("chords", "chord length distributions"),
    ("granulometry", "granulometry (openings by balls)"),
    ("pore_sizes", "pore-size function"),
    ("capacity_ball", "Choquet capacity of balls T(B_r)"),
    ("capacity_segment", "Choquet capacity of segments T(l)"),
)


def _axes_figure(curve, title: str, ylabel: str, theory=None):
    """The curve along the axes, with the closed form of the model: one curve ``(x, y)``,
    or one per axis ``{axis: (x, y)}`` dashed in the colour of that axis."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    colours = {}
    for axis in "xyz":
        if axis in curve.lags and len(curve.lags[axis]) > 1:
            (line,) = ax.plot(curve.lags[axis], curve.values[axis], label=axis, lw=1.6)
            colours[axis] = line.get_color()
    if isinstance(theory, dict):
        for axis, xy in theory.items():
            ax.plot(*xy, "--", color=colours.get(axis, "k"), lw=1.2, label=f"theory {axis}")
    elif theory is not None:
        ax.plot(*theory, "k--", lw=1.2, label="theory")
    ax.set_xlabel("h")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    return fig


def _plot(x, ys: dict, title: str, xlabel: str, ylabel: str, *, theory=None, log=False):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    for label, y in ys.items():
        ax.plot(x if not isinstance(y, tuple) else y[0], y if not isinstance(y, tuple) else y[1],
                lw=1.6, label=label)  # fmt: skip
    if theory is not None:
        ax.plot(*theory, "k--", lw=1.2, label="theory")
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if len(ys) > 1 or theory is not None:
        ax.legend(fontsize=8, frameon=False)
    return fig


@tool(
    "curves",
    "statistics",
    "Correlation functions and distributions",
    "Functions of a morphology on a voxel image of it, along the three axes: covariance, "
    "lineal path, chord length distributions, granulometry by openings, pore-size "
    "function, Choquet capacities of balls and segments; the closed forms of its model "
    "(Boolean and Poisson models, Gaussian excursions) drawn with them.",
    [
        Field("source", "item", label="morphology", required=True),
        Field("resolution", "int", 96, label="voxels per edge"),
        Field("phase", "int", 1),
        Field("curves", "choices", ["covariance", "lineal_path", "chords"], choices=CURVES),
    ],
    order=2,
    menu="Morphology",
)
def curves(p: dict, ctx) -> dict:
    from morphogen.analysis.correlation import lineal_path, pore_size_function
    from morphogen.analysis.covariance import covariance
    from morphogen.analysis.morphology import chord_lengths, granulometry
    from morphogen.analysis.random_sets import choquet_capacity
    from morphogen.gui.api import _voxels

    micro = p["source"]["micro"]
    _voxels(micro.domain, p["resolution"], 12.0, "image")
    image = micro.voxelize(shape=p["resolution"])
    model, phase = micro.model, p["phase"]
    chosen = p["curves"] or []
    for k, name in enumerate(chosen):
        ctx.progress(k / max(len(chosen), 1), name)
        if name == "covariance":
            c = covariance(image, phase)
            h = c.lags["x"]
            theory = _axis_covariances(model, c) if phase == 1 else None
            ctx.figure(_axes_figure(c, "covariance", "C(h)", theory), "covariance")
            rows = []
            for i, lag in enumerate(h):
                row = {"h": float(lag)}
                row.update({a: float(v[i]) for a, v in c.values.items() if i < len(v)})
                rows.append(row)
            ctx.table(rows, "covariance")
        elif name == "lineal_path":
            lp = lineal_path(image, phase)
            ctx.figure(_axes_figure(lp, "lineal path", "L(h)"), "lineal_path")
        elif name == "chords":
            ys, rows = {}, []
            for axis in range(3):
                if image.shape[axis] < 2:
                    continue
                lengths = chord_lengths(image, phase, axis)
                if lengths.size:
                    counts, edges = np.histogram(lengths, bins=24, density=True)
                    mid = 0.5 * (edges[1:] + edges[:-1])
                    ys["xyz"[axis]] = (mid, counts)
                    rows.append({"axis": "xyz"[axis], "chords": int(lengths.size),
                                 "mean": float(lengths.mean())})  # fmt: skip
            if ys:
                ctx.figure(_plot(None, ys, "chord lengths", "length", "density"), "chords")
                ctx.table(rows, "chords", "mean chord lengths")
        elif name == "granulometry":
            g = granulometry(image, phase)
            ctx.figure(_plot(g.radii, {"kept": g.fractions}, "granulometry", "radius",
                             "volume fraction kept"), "granulometry")  # fmt: skip
        elif name == "pore_sizes":
            ps = pore_size_function(image, 1 - phase if phase in (0, 1) else 0)
            delta = np.asarray(ps.delta, dtype=float)
            if len(delta) == len(ps.density) + 1:  # the edges of the bins: their centres
                delta = 0.5 * (delta[1:] + delta[:-1])
            ctx.figure(_plot(delta, {"density": ps.density}, "pore-size function",
                             "distance δ", "P(δ)"), "pore_sizes")  # fmt: skip
        elif name in ("capacity_ball", "capacity_segment"):
            element = name.split("_")[1]
            cap = choquet_capacity(image, element, phase=phase)  # segments along x
            theory = _capacities(model, element, cap.sizes) if phase == 1 else None
            ctx.figure(_plot(cap.sizes, {"measured": cap.values}, f"Choquet capacity ({element})",
                             "size", "T(K)", theory=theory), name)  # fmt: skip
    return {}


def _axis_covariances(model, curve):
    """The closed-form covariance of a model along each axis of the measured ``curve`` when
    it depends on the direction (anisotropic fibres: ``covariance(h, direction)``), else the
    single one of an isotropic model (or None)."""
    import inspect

    method = getattr(model, "covariance", None)
    orientation = getattr(model, "orientation", None)
    if method is None or "direction" not in inspect.signature(method).parameters:
        return _closed(model, "covariance", curve.lags["x"])
    if orientation is None or type(orientation).__name__ == "Isotropic":
        return _closed(model, "covariance", curve.lags["x"])
    out = {}
    for k, axis in enumerate("xyz"):
        if axis in curve.lags and len(curve.lags[axis]) > 1:
            x = curve.lags[axis]
            try:
                y = np.asarray(method(x, direction=np.eye(3)[k]), dtype=float)
            except (TypeError, ValueError, NotImplementedError, AttributeError):
                continue
            if y.shape == np.shape(x) and np.all(np.isfinite(y)):
                out[axis] = (x, y)
    return out or None


def _capacities(model, element: str, sizes):
    """The closed-form capacity of a model for the elements measured: segments along x, as
    :func:`morphogen.analysis.random_sets.choquet_capacity` takes them, when the capacity
    depends on the direction (fibres of anisotropic axes); else None."""
    import inspect

    method = getattr(model, "choquet_capacity", None)
    if method is not None and "direction" in inspect.signature(method).parameters:
        return _closed(model, "choquet_capacity", element, sizes, direction=(1.0, 0.0, 0.0))
    return _closed(model, "choquet_capacity", element, sizes)


def _closed(model, name: str, *args, **kwargs):
    """The closed form ``name`` of a model at the abscissae (its last positional argument),
    when it has one that evaluates to finite values; else None."""
    if model is None or not hasattr(model, name):
        return None
    x = args[-1]
    try:
        y = np.asarray(getattr(model, name)(*args, **kwargs), dtype=float)
    except (TypeError, ValueError, NotImplementedError, AttributeError):
        return None
    return (x, y) if y.shape == np.shape(x) and np.all(np.isfinite(y)) else None


@tool(
    "stereology",
    "statistics",
    "Stereology",
    "The sections of a phase by the planes of a voxel image of a morphology normal to the "
    "three axes (their equivalent radii; touching sections split by a watershed), and the "
    "size distribution of its particles recovered from them by Saltykov's unfolding, for "
    "spheres or aligned ellipsoids of a common shape (Degdeg & Dirrenberger); the shape "
    "consistent with the mean sections of the three directions.",
    [
        Field("source", "item", label="morphology", required=True),
        Field("resolution", "int", 96, label="voxels per edge"),
        Field("phase", "int", 1, doc="the phase of the particles"),
        Field("separate", "bool", True, label="split touching sections (watershed)"),
        Field("shape", "floats", [1.0, 1.0], label="shape factors b/a, c/a", doc="1, 1: spheres"),
        Field("bins", "int", 16, label="classes of radii"),
        Field(
            "classes",
            "int",
            label="3D classes unfolded",
            doc="blank: the number that best refolds the three directions",
        ),
    ],
    order=3,
    menu="Morphology",
)
def stereology(p: dict, ctx) -> dict:
    from morphogen.analysis.stereology import consistent_shape, section_sizes, unfold_sections
    from morphogen.gui.api import _voxels

    micro = p["source"]["micro"]
    _voxels(micro.domain, p["resolution"], 6.0, "image")
    image = micro.voxelize(shape=p["resolution"])
    axes = [a for a in range(3) if image.shape[a] > 1 and min(np.delete(image.shape, a)) > 1]
    sections = []
    for k, axis in enumerate(axes):
        ctx.progress(0.8 * k / max(len(axes), 1), f"sections normal to {'xyz'[axis]}")
        sec = section_sizes(image, p["phase"], axis, separate=p["separate"])
        if len(sec.radii):
            sections.append(sec)
    if not sections:
        raise ValueError(f"no section of phase {p['phase']} in the planes of the image")
    rows = [
        {"axis": "xyz"[s.axis], "sections": len(s.radii), "mean radius": s.mean_radius(),
         "per unit area": float(s.weights.sum() / s.area)}
        for s in sections
    ]  # fmt: skip
    ctx.table(rows, "sections", "sections of the three directions")
    shape = tuple((p["shape"] or [1.0, 1.0])[:2])
    if len(shape) != 2:
        raise ValueError("shape factors: two numbers, b/a and c/a")
    ctx.progress(0.85, "unfolding")
    classes = p["classes"] if p["classes"] or len(sections) > 1 else max(2, p["bins"] // 2)
    unfolded = unfold_sections(sections, shape=shape, bins=p["bins"], classes=classes)
    fractions = unfolded.fractions()
    ctx.table(
        [{"short semi-axis": float(r), "per unit volume": float(n), "fraction": float(f)}
         for r, n, f in zip(unfolded.radii, unfolded.density, fractions, strict=True)],
        "unfolded", "size distribution of the particles (Saltykov)",
    )  # fmt: skip
    ctx.figure(
        _plot(unfolded.radii, {"unfolded": unfolded.density}, "particles per unit volume",
              "short semi-axis a", "N_V"),
        "unfolded",
    )  # fmt: skip
    values = {
        f"refolding error {'xyz'[a]}": float(e)
        for a, e in zip(unfolded.axes, unfolded.errors, strict=True)
    }
    values["mean short semi-axis"] = float(np.sum(unfolded.radii * fractions))
    if len(sections) == 3:
        try:
            b, c = consistent_shape(sections)
            values.update({"consistent b/a": b, "consistent c/a": c})
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            pass
    return {"values": values}


@tool(
    "fit_table",
    "statistics",
    "RVE fit of a table of results",
    "The RVE scaling law fitted to a table of results of any origin (apparent moduli of "
    "other codes, image measures): one row per realization, a column of volumes and "
    "columns of properties; the variance D²(V) fitted as K V^-γ (γ free or imposed) and "
    "the RVE volumes for the errors and numbers of realizations chosen, as `morphogen fit`.",
    [
        Field("table", "text", label="table (CSV)", required=True,
              doc="a header line, then one line per realization: e.g. volume,E1 then 8,1.23"),
        Field("volume_column", "str", "volume", label="column of the volumes"),
        Field("properties", "str", label="columns of the properties",
              doc="separated by commas (blank: every numeric column but the volume)"),
        Field("gamma", "float", label="γ imposed", doc="blank: fitted"),
        Field("relative_errors", "floats", [0.01, 0.05]),
        Field("n_realizations", "ints", [1, 10], label="numbers of realizations"),
    ],
    order=6,
    menu="Representative volume",
)  # fmt: skip
def fit_table(p: dict, ctx) -> dict:
    import csv
    import io as textio

    from morphogen import pipeline
    from morphogen.config import RVESpec
    from morphogen.rve import report as rve_report

    text = p["table"].strip()
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t ")
    rows = []
    for raw in csv.DictReader(textio.StringIO(text), dialect=dialect):
        row = {}
        for k, v in raw.items():
            if k is None:
                continue
            try:
                row[k.strip()] = float(str(v).replace(",", ".")) if v not in (None, "") else None
            except ValueError:
                row[k.strip()] = v
        rows.append(row)
    volume = p["volume_column"]
    if not rows or volume not in rows[0]:
        raise ValueError(
            f"table: no column {volume!r} (the columns: {', '.join(rows[0]) if rows else 'none'})"
        )
    for row in rows:
        row["volume"] = row[volume]
    if p["properties"]:
        properties = [s.strip() for s in p["properties"].split(",") if s.strip()]
    else:
        properties = [
            k for k, v in rows[0].items() if isinstance(v, float) and k not in ("volume", volume)
        ]
    if not properties:
        raise ValueError("properties: no numeric column besides the volumes")
    spec = RVESpec(relative_errors=p["relative_errors"], n_realizations=p["n_realizations"],
                   gamma=p["gamma"], bootstrap=200)  # fmt: skip
    ctx.progress(0.3, "fit")
    report = pipeline.build_report(rows, spec, properties)
    rve_report.write(report, ctx.folder)
    for png in sorted(ctx.folder.glob("*.png")):
        ctx.figures.append({"title": png.stem.replace("_", " "), "url": ctx.url(png.name)})
    return {"tables": _markdown_tables((ctx.folder / "rve_report.md").read_text(encoding="utf-8"))}


@tool(
    "reconstruction",
    "statistics",
    "Reconstruction from the statistics",
    "A new morphology whose two-point correlation function (and lineal path) match those "
    "of a morphology of the workspace, by the simulated annealing of Yeong & Torquato "
    "(1998) on a lattice of cells, or by gradient descent on its relaxed cells (the "
    "two-point function at every lag, as MCRpy; the exact covariance of the model when it "
    "has one); it joins the workspace, to compare its properties with those of the "
    "original.",
    [
        Field("source", "item", label="morphology", required=True),
        Field("resolution", "int", 48, label="cells per edge"),
        Field("lineal_path", "float", 0.0, label="weight of the lineal path",
              doc="0: the two-point function alone"),
        Field("method", "choice", "annealing",
              choices=(("annealing", "annealing: swaps of cells, S2 and L along the axes"),
                       ("gradient", "gradient: the relaxed cells, S2 at every lag (MCRpy)"))),
        Field("sweeps", "int", 60, doc="trial swaps per cell", when={"method": "annealing"}),
        Field("iterations", "int", 2000, doc="of L-BFGS-B", when={"method": "gradient"}),
        Field("seed", "int", 1),
    ],
    order=7,
    menu="Reconstruction",
)  # fmt: skip
def reconstruction(p: dict, ctx) -> dict:
    from morphogen import io
    from morphogen.analysis.covariance import covariance
    from morphogen.domain import Domain
    from morphogen.gui import api
    from morphogen.models.base import model_from_config
    from morphogen.seeding import derive_seed

    item = p["source"]
    micro = item["micro"]
    lengths = np.asarray(micro.domain.lengths, dtype=float)
    cell = float(lengths.max()) / p["resolution"]
    method = p.get("method") or "annealing"
    spec = {"type": "reconstruction", "cell_size": cell, "lineal_path": p["lineal_path"],
            "method": method}  # fmt: skip
    if method == "annealing":
        spec["sweeps"] = p.get("sweeps") or 60
    else:
        spec["iterations"] = p.get("iterations") or 2000
    planar = bool(item.get("planar"))
    if planar:
        spec["dimension"] = 2
    if item.get("model") and item["model"].get("type") != "reconstruction":
        spec["reference"] = item["model"]
    else:  # the target: the phases of the morphology on the lattice
        target = micro.voxelize(voxel_size=cell)
        path = ctx.path("target.vti")
        io.save_image(target, path)
        spec["image"] = str(path)
    model = model_from_config(spec)
    domain = Domain(size=tuple(float(v) for v in lengths), periodic=True)
    ctx.progress(0.1, spec["method"])
    new = model.generate(domain, derive_seed(p["seed"], 0))
    ctx.progress(0.85, "views")
    views = api.morphology_views(new, 384)
    a = micro.voxelize(voxel_size=cell)
    b = new.voxelize(voxel_size=cell)
    ca, cb = covariance(a, 1), covariance(b, 1)
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    ax.plot(ca.lags["x"], np.mean([ca.values[k] for k in "xy"], axis=0), lw=1.8, label="original")
    ax.plot(cb.lags["x"], np.mean([cb.values[k] for k in "xy"], axis=0), "--", lw=1.8,
            label="reconstruction")  # fmt: skip
    ax.set_xlabel("h")
    ax.set_ylabel("C(h)")
    ax.set_title("two-point correlation function", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ctx.figure(fig, "covariance")
    for s in views["sections"]:
        ctx.figures.append({"title": f"reconstruction, {s['title']}", "url": s["png"]})
    attrs = getattr(new, "attributes", {}) or {}
    added = ctx.add(
        "morphology",
        f"reconstruction of #{item['number']}",
        micro=new,
        origin="statistics",
        source=item["key"],
        model=model.to_config(),
        domain={"size": [float(v) for v in lengths], "periodic": True},
        seed=p["seed"],
        phases={"0": "phase 0", "1": "phase 1"},
        thumbnail=views["thumbnail"],
        fraction=views["fraction"],
        planar=planar,
        info={"model": "reconstruction", "source": f"#{item['number']}"},
    )
    return {
        "values": {
            "fraction (original)": float((a.data == 1).mean()),
            "fraction (reconstruction)": float((b.data == 1).mean()),
            "initial energy": attrs.get("initial_energy"),
            "final energy": attrs.get("energy"),
            "error of the two-point function": attrs.get("error_two_point"),
        },
        "items": [] if added in ctx.items else [added],
    }


#: The settings of the axes of the orientations of an .ang file, as the form names them
ANG_AXES = (
    ("1", "1: its first axis down the image (usual)"),
    ("2", "2: its first axis up the image"),
    ("3", "3: its first axis to the right"),
    ("4", "4: its first axis to the left"),
)


@tool(
    "texture",
    "statistics",
    "Orientation map (EBSD)",
    "The grains and the texture of an orientation map (an .ang or a .ctf file): the grains "
    "found from the disorientations of its points and their numbers phase by phase, the "
    "map in the colours of its orientations, its pole figures, its inverse pole figure and "
    "the disorientations of its boundaries. Its polycrystal — the grains as columns "
    "through one layer of cells, each with its orientation — joins the workspace, for the "
    "tools of the mechanics.",
    [
        Field("file", "str", "ebsd_map.ctf", label="map", required=True,
              doc="an .ang or a .ctf file (ebsd_map.ctf: the map that comes with morphoGen)"),
        Field("threshold", "float", 5.0, unit="degrees",
              doc="two neighbouring points less than this apart are in one grain"),
        Field("min_size", "int", 5, label="smallest grain", unit="points",
              doc="smaller grains are given to their neighbours"),
        Field("setting", "choice", "1", choices=ANG_AXES, label="axes of an .ang file",
              doc="of its orientations, in the image of the map (a .ctf file gives them in "
              "the axes of the image)"),
        Field("poles", "str", "", label="pole figures",
              doc="families of planes, e.g. 100 110 111 (blank: those of the lattice)"),
        Field("direction", "choice", "z", choices=("x", "y", "z"),
              label="direction of the sample",
              doc="of the inverse pole figure and of the colours of the map"),
        Field("halfwidth", "float", 10.0, unit="degrees",
              doc="of the kernel that spreads each pole"),
        Field("add", "bool", True, label="add its polycrystal to the workspace"),
    ],
    order=8,
    menu="Orientation maps",
)  # fmt: skip
def texture_map(p: dict, ctx) -> dict:
    from morphogen import texture
    from morphogen.domain import Domain
    from morphogen.gui import api
    from morphogen.io import ebsd as maps
    from morphogen.models.ebsd import EBSD
    from morphogen.viz import figures

    model = EBSD(file=p["file"], threshold=p["threshold"], min_size=p["min_size"],
                 setting=int(p["setting"]))  # fmt: skip
    poles = []
    for word in re.split(r"[\s,;]+", (p["poles"] or "").strip()):
        digits = re.findall(r"-?\d", word)
        if word and "".join(digits) != word:
            raise ValueError(f"pole figures: indices such as 111 or 10-10, got {word!r}")
        if word:
            poles.append(tuple(int(v) for v in digits))
    direction = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[p["direction"]]
    ctx.progress(0.05, "reading the map")
    measured = maps.read_ebsd(model.path(), setting=model.setting)
    ctx.progress(0.2, "grains")
    grains = model.grains()
    summary = grains.summary(poles=poles or None, halfwidth=p["halfwidth"])
    ny, nx = measured.shape
    values = {
        "points": f"{nx} x {ny}",
        f"step ({measured.unit})": f"{measured.step[0]:g} x {measured.step[1]:g}",
        "indexed": float(measured.indexed.mean()),
        "grains": grains.count,
    }
    several = len(summary["phases"]) > 1
    pairs, lengths, angles = grains.boundaries()
    ctx.figure(figures.ipf_map_figure(measured, direction, grains=grains), "map",
               f"the map in the colours of its crystals along {p['direction']}")  # fmt: skip
    defaults = {"cubic": ((1, 0, 0), (1, 1, 0), (1, 1, 1)),
                "hexagonal": ((0, 0, 0, 1), (1, 0, -1, 0), (1, 0, -1, 1))}  # fmt: skip
    for step, entry in enumerate(summary["phases"]):
        ctx.progress(0.3 + 0.5 * step / len(summary["phases"]), f"texture of {entry['name']}")
        prefix = f"{entry['name']}: " if several else ""
        values[f"{prefix}lattice"] = entry["lattice"]
        values[f"{prefix}grains"] = entry["grains"]
        values[f"{prefix}mean diameter ({measured.unit})"] = entry["mean_diameter"]
        values[f"{prefix}mean diameter by area ({measured.unit})"] = entry["mean_diameter_by_area"]
        values[f"{prefix}spread inside the grains (degrees)"] = entry["mean_spread"]
        if "mean_disorientation" in entry:
            values[f"{prefix}mean disorientation of the boundaries (degrees)"] = entry[
                "mean_disorientation"
            ]
        for name, largest in entry.get("pole_figures", {}).items():
            values[f"{prefix}pole figure {{{name}}}, largest density"] = largest
        k = entry["phase"]
        of = grains.phase == k
        suffix = f"_{k}" if several else ""
        family = {"m-3": "cubic", "6/m": "hexagonal"}.get(entry["lattice"], entry["lattice"])
        if family in defaults:
            ratio = grains.phases[k - 1].c_over_a if family == "hexagonal" else None
            shown = [texture.pole_figure(grains.orientations[of], indices, family,
                                         weights=grains.areas[of], halfwidth=p["halfwidth"],
                                         c_over_a=ratio)
                     for indices in poles or defaults[family]]  # fmt: skip
            ctx.figure(figures.pole_figures_figure(shown), f"pole_figures{suffix}",
                       f"{prefix}pole figures")  # fmt: skip
            inverse = texture.inverse_pole_figure(grains.orientations[of], direction, family,
                                                  weights=grains.areas[of],
                                                  halfwidth=p["halfwidth"])  # fmt: skip
            points = texture.inverse_poles(grains.orientations[of], direction, family)
            ctx.figure(figures.inverse_pole_figure_figure(inverse, points=points),
                       f"inverse_pole_figure{suffix}",
                       f"{prefix}the crystals along {p['direction']}")  # fmt: skip
        between = of[pairs[:, 0]] if len(pairs) else np.zeros(0, dtype=bool)
        if between.any():
            ctx.figure(figures.disorientation_figure(angles[between], lengths[between],
                                                     lattice=entry["lattice"]),
                       f"disorientations{suffix}",
                       f"{prefix}disorientations of the boundaries")  # fmt: skip
    table = grains.table()
    ctx.table([{name: column[i] for name, column in table.items()} for i in range(grains.count)],
              "grains", "the grains (angles in degrees)")  # fmt: skip
    ctx.table([{"grain a": int(a), "grain b": int(b), "length": float(length),
                "disorientation": float(angle)}
               for (a, b), length, angle in zip(pairs.tolist(), lengths, angles, strict=True)],
              "boundaries", "the boundaries between the grains of one phase")  # fmt: skip
    items = []
    if p["add"]:
        ctx.progress(0.85, "polycrystal")
        width, height = measured.size
        size = [float(width), float(height), float(min(measured.step))]
        new = model.generate(Domain(size=tuple(size), periodic=False, unit=measured.unit), 0)
        views = api.morphology_views(new, 384)
        added = ctx.add(
            "morphology",
            f"polycrystal of {Path(p['file']).name}",
            micro=new,
            origin="statistics",
            model=model.to_config(),
            domain={"size": size, "periodic": False},
            seed=0,
            phases={str(k): v for k, v in model.active_phases().items()},
            thumbnail=views["thumbnail"],
            fraction=views["fraction"],
            planar=True,
            info={"model": "ebsd", "grains": grains.count},
        )
        items = [] if added in ctx.items else [added]
    return {"values": values, "items": items}


#: The criteria of percolation: along an axis, any of them or all three
AXES = (
    ("x", "along x"),
    ("y", "along y"),
    ("z", "along z"),
    ("any", "along any axis"),
    ("all", "along the three axes"),
)


@tool(
    "percolation",
    "statistics",
    "Percolation threshold",
    "The fraction at which a phase of the model of a morphology percolates: the share of "
    "independent realizations whose phase joins the faces normal to an axis (or, on a "
    "periodic morphology, wraps around the box) as a parameter of the model varies (the "
    "volume fraction), its crossing of 1/2 bracketed by bisection down to the tolerance, "
    "the realizations seeded alike at every fraction (common random numbers); optionally, "
    "the rigidity threshold, where the elastic moduli of the phase by the FFT solver vanish, "
    "and the critical exponent of the conductivity or of the elastic moduli of the phase at "
    "the threshold, by the finite-size scaling of its FFT solutions over box sizes.",
    [
        Field(
            "source",
            "item",
            label="morphology",
            required=True,
            needs=("model",),
            doc="a morphology generated from a model (its model, domain and periodicity)",
        ),
        Field("parameter", "str", "volume_fraction", doc="the parameter of the model varied"),
        Field("lower", "float", 0.05, label="lower bound"),
        Field("upper", "float", 0.95, label="upper bound"),
        Field("resolution", "int", 64, label="voxels per edge", doc="of the images"),
        Field("phase", "int", 1),
        Field("axis", "choice", "x", label="percolating", choices=AXES),
        Field(
            "connectivity",
            "choice",
            "6",
            label="neighbours",
            choices=(
                ("6", "6: faces"),
                ("18", "18: faces and edges"),
                ("26", "26: faces, edges and corners"),
                ("4", "4: edges (2-D)"),
                ("8", "8: edges and corners (2-D)"),
            ),
            doc="a planar morphology (one voxel along z) percolates in its plane: 6 is 4 "
            "there, 18 and 26 are 8",
        ),
        Field(
            "boundaries",
            "choice",
            "morphology",
            choices=(
                ("morphology", "those of the morphology"),
                ("periodic", "periodic: wrapping around the box"),
                ("open", "open: joining two faces"),
            ),
        ),
        Field(
            "method",
            "choice",
            "labels",
            label="algorithm",
            choices=(
                ("labels", "labels: the components of the image"),
                ("fire", "forest fire: lit on a face, reaching the other"),
            ),
            doc="the fire crosses the image face to face along the axis, through the "
            "periodic faces across it; the labels wrap around a periodic box",
        ),
        Field("realizations", "int", 9, doc="images at each fraction"),
        Field("tol", "float", 0.005, label="tolerance", doc="width of the final bracket"),
        Field("seed", "int", 1),
        Field(
            "sizes",
            "ints",
            None,
            label="box sizes",
            unit="voxels",
            doc="finite-size scaling: the thresholds of cubes of these sizes (at least 3), at "
            "the voxel size of the resolution, extrapolated to the infinite medium as "
            "p_c + a L^(-1/ν); none: the threshold at the resolution",
        ),
        Field(
            "nu",
            "float",
            None,
            label="ν",
            doc="none: that of random percolation, 0.8765 in 3-D and 4/3 in 2-D",
        ),
        Field("fit_nu", "bool", False, label="fit ν", doc="at least 4 sizes"),
        Field(
            "rigidity",
            "bool",
            False,
            label="rigidity threshold",
            doc="the threshold where the elastic moduli of the phase vanish, not where it "
            "connects (its voxels joined at corners and edges turn about them, so that no "
            "neighbours give it): where half the realizations bear a uniaxial strain along the "
            "axis with a modulus above the cut, the other phases void, by the solver below; "
            "that of the exponent f when its threshold is not given",
        ),
        Field(
            "cut",
            "float",
            None,
            doc="rigidity: the modulus above which a realization bears, over the Young's "
            "modulus of the phase; none: 1e-8 by the FFT solver, 1e-12 by the voxels "
            "factorized, whose energies resolve the moduli down to the round-off",
        ),
        Field(
            "directions",
            "choice",
            "none",
            label="anisotropy",
            choices=(
                ("none", "none: the criterion above"),
                ("axes", "along the axes"),
                ("lattice", "along the axes and the diagonals"),
                ("sampled", "along directions sampled"),
            ),
            doc="the thresholds across the cores of the box along directions (the cylinders "
            "of its inscribed ball, by the forest fire, the realizations common to the "
            "directions), their ratio and their threshold tensor",
        ),
        Field(
            "count",
            "int",
            12,
            label="directions sampled",
            when={"directions": "sampled"},
            doc="over half a circle in 2-D, on a Fibonacci spiral over the half sphere in 3-D",
        ),
        Field(
            "exponent",
            "choice",
            "none",
            label="critical exponent",
            choices=(
                ("none", "none"),
                ("thermal", "t: the conductivity"),
                ("elastic", "f: the elastic moduli"),
            ),
            doc="the exponent of the property of the phase by the solver below: the finite-size "
            "scaling of its mean over the box sizes at the threshold, L^(-t/ν); the voxels of "
            "the solver joining through their nodes, take 26 neighbours (8 in 2-D) and the "
            "periodic boundaries for the threshold of the conductivity",
        ),
        Field(
            "threshold",
            "float",
            None,
            doc="none: extrapolated from the box sizes with the criterion above",
        ),
        Field(
            "threshold_stderr",
            "float",
            None,
            label="its standard error",
            doc="whose effect on the exponent is computed (the realizations at the threshold "
            "less and plus it); none: that of the extrapolation, 0 for a threshold given",
        ),
        Field(
            "contrast",
            "float",
            1e-8,
            doc="the properties of the other phases over those of the phase (0: void)",
        ),
        Field(
            "device",
            "choice",
            "auto",
            choices=(("auto", "auto: the GPU when there is one"), ("cpu", "CPU"), ("gpu", "GPU")),
            doc="of the FFT solver",
        ),
        Field(
            "solver",
            "choice",
            "fft",
            choices=(
                ("fft", "FFT: conjugate gradients, on the device"),
                ("fe", "FE: the voxels assembled and factorized, on the CPU"),
            ),
            doc="of the trilinear voxels, for the rigidity and the exponents: the FFT solver, or "
            "their matrix assembled on the voxels of the materials and solved by conjugate "
            "gradients preconditioned by its sparse factorization, the same moduli in a few "
            "iterations where the FFT solver takes thousands, near the rigidity threshold with "
            "void around (contrast 0)",
        ),
    ],
    order=4,
    groups=(
        ("Family", "parameter"),
        ("Criterion", "phase"),
        ("Bisection", "realizations"),
        ("Finite-size scaling", "sizes", True),
        ("Anisotropy", "directions", True),
        ("Critical exponent", "exponent", True),
    ),
    menu="Morphology",
)
def percolation(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure

    from morphogen.analysis.percolation import (
        family_from_config,
        percolation_scaling,
        percolation_threshold,
    )
    from morphogen.domain import Domain
    from morphogen.gui.api import _voxels

    item = p["source"]
    model = _model_of(item, "the threshold is that")
    domain, name = item["micro"].domain, p["parameter"] or "volume_fraction"
    _voxels(domain, p["resolution"], 12.0, "image")
    config = {
        "model": model,
        "domain": {"size": [float(v) for v in domain.lengths], "periodic": bool(domain.periodic)},
        "voxels": {"shape": p["resolution"]},
    }
    lower, upper, tol, n = p["lower"], p["upper"], p["tol"], p["realizations"]
    sizes = sorted(set(p["sizes"] or ()))
    if sizes and sizes[-1] > 0:  # the largest box (sizes not positive refused by the scaling)
        _voxels(Domain(size=1.0), sizes[-1], 12.0, "image")
    ratio = (upper - lower) / tol if None not in (lower, upper) and tol else 0.0
    total = max(int(n or 1), 1) * (2 + (int(np.ceil(np.log2(ratio))) if ratio > 1 else 0))
    total *= max(len(sizes), 1)
    done = [0]
    criterion = {
        "phase": p["phase"],
        "axis": p["axis"],
        "connectivity": int(p["connectivity"]),
        "periodic": {"morphology": None, "periodic": True, "open": False}[p["boundaries"]],
        "method": p["method"],
        "realizations": n,
        "tol": tol,
        "seed": p["seed"],
    }

    def family(fraction, rng, *size):  # the progress, image by image (and the stops there)
        box = f" ({size[0]} voxels)" if size else ""
        ctx.progress(min(done[0] / total, 0.99), f"{name} = {fraction:.4g}{box}")
        done[0] += 1
        return generate(fraction, rng, *size)

    if p["directions"] != "none":
        count = {"axes": 3, "lattice": 13}.get(p["directions"], p["count"] or 1)
        total *= max(int(count), 1)
        generate = family_from_config(config, name)
        return _anisotropy_outputs(family, p, name, ctx)
    if p["exponent"] != "none":
        if p["threshold"] is not None:
            total = 0
        total += max(len(sizes), 1) * max(int(n or 1), 1) * 3
        generate = family_from_config(config, name, sized=True)
        return _exponent_outputs(family, sizes, criterion, p, name, ctx)
    if p["rigidity"]:
        generate = family_from_config(config, name, sized=True)
        return _rigidity_outputs(_rigidity_of(family, sizes, criterion, p), p, name, ctx)
    if sizes:
        generate = family_from_config(config, name, sized=True)
        scaling = percolation_scaling(
            family, sizes, lower, upper, nu=p["nu"], fit_nu=p["fit_nu"], **criterion
        )
        results, threshold = scaling.results, scaling.threshold
        labels, marked = [f"{s} voxels" for s in scaling.sizes], "extrapolated"
    else:
        generate = family_from_config(config, name)
        result = percolation_threshold(family, lower, upper, **criterion)
        results, threshold, labels, marked = (result,), result.threshold, ["tried"], "threshold"
    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    for one, label in zip(results, labels, strict=True):
        ax.plot(one.fractions, one.probabilities, "o-", lw=1.6, ms=4, label=label)
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.axvline(threshold, color="k", ls="--", lw=1.2, label=f"{marked} {threshold:.4g}")
    ax.set_xlabel(name)
    ax.set_ylabel("percolation probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"percolation of phase {p['phase']} ({p['axis']})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ctx.figure(fig, "percolation", "percolation probability")
    if sizes:
        return _scaling_outputs(scaling, p, name, ctx)
    rows = [{name: float(f), "probability": float(q)}
            for f, q in zip(result.fractions, result.probabilities, strict=True)]  # fmt: skip
    ctx.table(rows, "percolation", "probabilities tried")
    lo, hi = result.bracket
    return {"values": {"threshold": result.threshold, "bracket from": lo, "bracket to": hi,
                       "realizations": result.realizations}}  # fmt: skip


def _scaling_outputs(scaling, p: dict, name: str, ctx) -> dict:
    """The figure, the table and the values of a finite-size scaling of the threshold."""
    from matplotlib.figure import Figure

    x = scaling.sizes.astype(float) ** (-1.0 / scaling.nu)
    lo, hi = scaling.brackets.T
    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    ax.errorbar(x, scaling.thresholds, yerr=[scaling.thresholds - lo, hi - scaling.thresholds],
                fmt="o", ms=4, capsize=2, label="boxes (brackets)")  # fmt: skip
    for size, u, v in zip(scaling.sizes, x, scaling.thresholds, strict=True):
        ax.annotate(f"{size}", (u, v), textcoords="offset points", xytext=(4, 4), fontsize=7)
    line = np.linspace(0.0, 1.05 * x.max(), 50)
    ax.plot(line, scaling.threshold + scaling.amplitude * line, "-", lw=1.4,
            label=f"fit, ν = {scaling.nu:.4g}")  # fmt: skip
    ax.errorbar([0.0], [scaling.threshold], yerr=[scaling.stderr], fmt="s", color="k", ms=5,
                capsize=3, label=f"p_c {scaling.threshold:.4g} ± {scaling.stderr:.2g}")  # fmt: skip
    ax.set_xlim(-0.04 * x.max(), 1.08 * x.max())
    ax.set_xlabel(r"$L^{-1/\nu}$ ($L$: voxels per edge)")
    ax.set_ylabel(f"{name} at the crossing of 1/2")
    ax.set_title(f"finite-size scaling of phase {p['phase']} ({p['axis']})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ctx.figure(fig, "scaling", "finite-size scaling")
    columns = (scaling.sizes, scaling.thresholds, lo, hi, scaling.residuals)
    rows = [{"size": int(s), name: float(t), "bracket from": float(a), "bracket to": float(b),
             "residual": float(r)} for s, t, a, b, r in zip(*columns, strict=True)]  # fmt: skip
    ctx.table(rows, "scaling", "thresholds of the boxes")
    values = {"threshold": scaling.threshold, "standard error": scaling.stderr,
              "amplitude": scaling.amplitude, "nu": scaling.nu}  # fmt: skip
    if scaling.nu_stderr is not None:
        values["nu standard error"] = scaling.nu_stderr
    values["realizations"] = scaling.results[0].realizations
    return {"values": values}


def _exponent_outputs(family, sizes, criterion: dict, p: dict, name: str, ctx) -> dict:
    """The critical exponent of the conductivity or of the elastic moduli of the phase at
    its threshold (given, or extrapolated from the box sizes with the criterion): the means
    of the boxes against their size, drawn on logarithmic axes with their fit, tabulated,
    and the exponent with its errors."""
    from matplotlib.figure import Figure

    from morphogen.analysis.percolation import (
        _extrapolated_threshold,
        percolation_exponent,
        percolation_scaling,
    )

    threshold, stderr = p["threshold"], p["threshold_stderr"]
    if threshold is None:  # that of the connectivity for t, of the moduli for f
        if p["exponent"] == "elastic":
            scaling = _rigidity_of(family, sizes, criterion, {**p, "fit_nu": False})
        else:
            scaling = percolation_scaling(family, sizes, p["lower"], p["upper"], nu=p["nu"],
                                          **criterion)  # fmt: skip
        threshold, error = _extrapolated_threshold(scaling, p["lower"], p["upper"])
        stderr = error if stderr is None else stderr
    seed = p["seed"]
    r = percolation_exponent(
        family,
        sizes,
        threshold,
        threshold_stderr=stderr or 0.0,
        physics=p["exponent"],
        phase=p["phase"],
        axis=p["axis"],
        contrast=p["contrast"],
        nu=p["nu"],
        realizations=p["realizations"],
        device=p["device"],
        solver=p["solver"],
        seed=None if seed is None else seed + 1,
    )
    symbol = "t" if r.physics == "thermal" else "f"
    prop = "conductivity" if r.physics == "thermal" else "uniaxial-strain modulus"
    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    ax.errorbar(r.sizes, r.means, yerr=r.mean_stderr, fmt="o", ms=4, capsize=2,
                label=f"means at {threshold:.4g}")  # fmt: skip
    if np.isfinite(r.exponent):
        line = np.geomspace(r.sizes[0], r.sizes[-1], 50)
        ax.plot(line, r.amplitude * line ** (-r.ratio), "-", lw=1.4,
                label=f"{symbol} = {r.exponent:.3f} ± {r.stderr:.2g} (ν = {r.nu:.4g})")  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L$ (voxels per edge)")
    ax.set_ylabel(f"{prop} (phase 1)")
    ax.set_title(f"{prop} of phase {p['phase']} at the threshold ({p['axis']})", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, frameon=False)
    ctx.figure(fig, "exponent", "finite-size scaling of the property")
    columns = (r.sizes, r.means, r.mean_stderr, r.residuals)
    rows = [{"size": int(s), "mean": float(m), "standard error": float(e),
             "residual": _finite(d)} for s, m, e, d in zip(*columns, strict=True)]  # fmt: skip
    ctx.table(rows, "exponent", "means of the boxes at the threshold")
    values = {"exponent": r.exponent, "standard error": r.stderr,
              "error from the threshold": r.threshold_error, "exponent over nu": r.ratio,
              "nu": r.nu, "threshold": r.threshold, "threshold standard error": stderr or 0.0,
              "contrast": r.contrast}  # fmt: skip
    values = {k: float(v) for k, v in values.items() if np.isfinite(v)}
    values["realizations"], values["iterations"] = r.realizations, r.iterations
    return {"values": values}


def _rigidity_of(family, sizes, criterion: dict, p: dict):
    """The rigidity threshold of the box sizes, where the elastic moduli of the phase vanish
    (:func:`~morphogen.analysis.percolation.rigidity_threshold`), with the bisection of the
    criterion (its neighbours, boundaries and algorithm unused: the solver's)."""
    from morphogen.analysis.percolation import rigidity_threshold

    return rigidity_threshold(family, sizes, p["lower"], p["upper"], nu=p["nu"],
                              fit_nu=p["fit_nu"], phase=criterion["phase"],
                              axis=criterion["axis"], cut=p["cut"],
                              realizations=criterion["realizations"], tol=criterion["tol"],
                              device=p["device"], solver=p["solver"],
                              seed=criterion["seed"])  # fmt: skip


def _rigidity_outputs(result, p: dict, name: str, ctx) -> dict:
    """The figures, the tables and the values of a rigidity threshold: the shares of the
    realizations that bear and their moduli against the fraction, the finite-size scaling."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    for size, one in zip(result.sizes, result.results, strict=True):
        ax.plot(one.fractions, one.probabilities, "o-", lw=1.6, ms=4, label=f"{size} voxels")
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.axvline(result.threshold, color="k", ls="--", lw=1.2,
               label=f"extrapolated {result.threshold:.4g}")  # fmt: skip
    ax.set_xlabel(name)
    ax.set_ylabel("share of the realizations that bear")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"rigidity of phase {p['phase']} ({p['axis']})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ctx.figure(fig, "rigidity", "bearing probability")
    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    solved = [m[np.isfinite(m)] for m in result.moduli]
    positive = np.concatenate([m[m > 0] for m in solved]) if solved else np.empty(0)
    floor = 1e-3 * positive.min() if positive.size else 1e-3 * result.cut
    for size, one, moduli in zip(result.sizes, result.results, result.moduli, strict=True):
        fractions = np.broadcast_to(one.fractions[:, None], moduli.shape)
        keep = np.isfinite(moduli)
        ax.plot(fractions[keep], np.maximum(moduli[keep], floor), "o", ms=3, alpha=0.6,
                label=f"{size} voxels")  # fmt: skip
    ax.axhline(result.cut, color="k", ls="--", lw=1.0, label=f"cut {result.cut:.2g}")
    ax.set_yscale("log")
    ax.set_xlabel(name)
    ax.set_ylabel("modulus over that of the phase")
    ax.set_title("moduli of the realizations solved (the others void)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, frameon=False)
    ctx.figure(fig, "moduli", "moduli of the realizations")
    out = _scaling_outputs(result, p, name, ctx)
    values = {"cut": result.cut, "largest floppy modulus": result.floppy,
              "smallest rigid modulus": result.rigid}  # fmt: skip
    out["values"].update({k: float(v) for k, v in values.items() if np.isfinite(v)})
    out["values"].update(solutions=result.solutions, decided=result.decided,
                         unresolved=result.unresolved, iterations=result.iterations)  # fmt: skip
    return out


def _finite(value):
    """A number of a table or of the values: None when not finite (beyond the bounds)."""
    return float(value) if np.isfinite(value) else None


def _anisotropy_outputs(family, p: dict, name: str, ctx) -> dict:
    """The thresholds of a family across the cores of the box along directions: drawn (a
    rose in 2-D, bars in 3-D, with the quadratic form of the threshold tensor), tabulated,
    and their anisotropy (the ratio, the eigenvalues of the tensor)."""
    from matplotlib.figure import Figure

    from morphogen.analysis.percolation import percolation_anisotropy

    r = percolation_anisotropy(
        family,
        p["lower"],
        p["upper"],
        directions=p["count"] if p["directions"] == "sampled" else p["directions"],
        phase=p["phase"],
        connectivity=int(p["connectivity"]),
        realizations=p["realizations"],
        tol=p["tol"],
        seed=p["seed"],
    )
    n, finite = r.directions, np.isfinite(r.thresholds)
    planar, tensor = bool(np.all(n[:, 2] == 0)), bool(np.all(np.isfinite(r.tensor)))
    fig = Figure(figsize=(5.4, 4.2), layout="constrained")
    if planar:  # against the angle, a direction and its opposite alike
        ax = fig.subplots(subplot_kw={"projection": "polar"})
        angle = np.arctan2(n[:, 1], n[:, 0]) % np.pi
        for turn, label in ((0.0, "thresholds"), (np.pi, None)):
            ax.errorbar((angle + turn)[finite], r.thresholds[finite], yerr=r.stderr[finite],
                        fmt="o", ms=4, capsize=2, color="C0", label=label)  # fmt: skip
        if tensor:
            theta = np.linspace(0.0, 2.0 * np.pi, 361)
            u = np.column_stack([np.cos(theta), np.sin(theta)])
            ax.plot(theta, np.einsum("ia,ab,ib->i", u, r.tensor[:2, :2], u), lw=1.2,
                    color="C1", label="threshold tensor")  # fmt: skip
    else:
        ax = fig.subplots()
        x = np.arange(len(n))
        ax.bar(x[finite], r.thresholds[finite], yerr=r.stderr[finite], capsize=2, color="C0",
               label="thresholds")  # fmt: skip
        if tensor:
            fitted = np.einsum("ia,ab,ib->i", n, r.tensor, n)
            ax.plot(x, fitted, "D", ms=4, color="C1", label="threshold tensor")
        ticks = ["(" + ", ".join(f"{c:.2f}" for c in v) + ")" for v in n]
        ax.set_xticks(x, ticks, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(f"{name} at the crossing of 1/2")
        ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"percolation of phase {p['phase']} along {len(n)} directions", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ctx.figure(fig, "anisotropy", "thresholds along directions")
    columns = (n, r.thresholds, r.stderr, r.deviations, r.deviation_stderr)
    rows = [{"x": float(v[0]), "y": float(v[1]), "z": float(v[2]), name: _finite(t),
             "standard error": _finite(e), "deviation": _finite(d),
             "paired standard error": _finite(s)}
            for v, t, e, d, s in zip(*columns, strict=True)]  # fmt: skip
    ctx.table(rows, "anisotropy", "thresholds along directions")
    values = {"largest over smallest": r.ratio, "ratio standard error": r.ratio_stderr}
    for k, (v, e) in enumerate(zip(r.eigenvalues, r.eigenvalue_stderr, strict=True)):
        values[f"eigenvalue {k + 1}"], values[f"eigenvalue {k + 1} standard error"] = v, e
    values = {k: float(v) for k, v in values.items() if np.isfinite(v)}
    values["realizations"] = r.realizations
    return {"values": values}
