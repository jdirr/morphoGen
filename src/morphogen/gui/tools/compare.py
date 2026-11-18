"""The tools of the tab Comparisons: the curves of several runs of the workspace on one
figure (the loading paths of morphologies and the material points), with the points of a
test, and their final and extreme values side by side."""

from __future__ import annotations

import numpy as np

from morphogen.gui.tools import Field, tab, tool

tab(
    "compare",
    "Comparisons",
    "The curves of several runs of the workspace on one figure: the responses of "
    "morphologies along their loading paths and of behaviours at material points, with "
    "the points of a test; their final and extreme values side by side.",
    order=95,
    section="share",
)

COMPONENTS = ("11", "22", "33", "23", "31", "12")
STRAINS = (
    ("auto", "the strain component that varies most"),
    *[(f"E{c}", f"E{c}") for c in COMPONENTS],
    ("time", "the time"),
)
QUANTITIES = (
    ("auto", "the stress of the strain shown (else the one that varies most)"),
    *[(f"S{c}", f"S{c}") for c in COMPONENTS],
    ("p", "the mean cumulated plastic strain p"),
    ("f", "the porosity f"),
)
#: The colours of the runs (the reference in black)
COLOURS = ("#2166ac", "#d6604d", "#1b7837", "#762a83", "#e08214", "#35978f", "#8c510a")


def _true(rows: list[dict], measures: str) -> bool:
    """Whether the true measures are shown: asked for, and in the table (finite strains)."""
    return measures == "true" and bool(rows) and "LE11" in rows[0] and "TS11" in rows[0]


def _column(rows: list[dict], name: str) -> np.ndarray:
    return np.array([float(r.get(name) or 0.0) for r in rows], dtype=float)


def _axes(rows: list[dict], x: str, y: str, true: bool) -> tuple[str, str]:
    """The columns shown of a table of curves: ``x`` and ``y`` of the form, the automatic
    ones chosen in it (true measures: ``LE``, ``TS``)."""
    e_name, s_name = ("LE", "TS") if true else ("E", "S")
    if x == "auto":
        spans = [np.ptp(_column(rows, e_name + c)) for c in COMPONENTS]
        x = e_name + COMPONENTS[int(np.argmax(spans))]
    elif x.startswith("E") and true:
        x = "L" + x
    if y == "auto":
        if x == "time":
            spans = [np.ptp(_column(rows, s_name + c)) for c in COMPONENTS]
            y = s_name + COMPONENTS[int(np.argmax(spans))]
        else:
            y = s_name + x[-2:]
    elif y.startswith("S") and true:
        y = "T" + y
    return x, y


@tool(
    "compare_curves",
    "compare",
    "Curves of runs",
    "The curves of loading paths of morphologies and of material points of the workspace "
    "on one figure (a strain component, or the time, against a stress, the plastic strain "
    "or the porosity; the true measures of finite strains), with the points of a test and "
    "the gap of each run to them; the final, largest and smallest values of each.",
    [
        Field(
            "runs",
            "items",
            required=True,
            items=("simulation", "behaviour", "result"),
            needs=("curves",),
            doc="loading paths of morphologies (S), material points and identifications (B, R)",
        ),
        Field("x", "choice", "auto", label="along", choices=STRAINS),
        Field("y", "choice", "auto", label="shown", choices=QUANTITIES),
        Field(
            "measures",
            "choice",
            "nominal",
            choices=(
                ("nominal", "nominal (small strains, or E and S in finite strains)"),
                ("true", "true in finite strains (logarithmic strains, Cauchy stresses)"),
            ),
        ),
        Field(
            "reference",
            "text",
            label="points of a test",
            doc="x, y of the quantities shown, one pair per line (comma, space or tab; or "
            "semicolons with decimal commas): drawn with the curves, and the gap of each run",
        ),
    ],
)
def compare_curves(p: dict, ctx) -> dict:
    from matplotlib.figure import Figure

    from morphogen.gui.api import curves_of
    from morphogen.gui.tools.mechanics import _reference
    from morphogen.pipeline import read_rows

    reference = _reference(p.get("reference"))
    fig = Figure(figsize=(6.4, 4.4), layout="constrained")
    ax = fig.add_subplot()
    table, values, names = [], {}, set()
    for k, item in enumerate(p["runs"]):
        rows = read_rows(curves_of(item, ctx.session.root))
        if not rows:
            raise ValueError(f"{item['title']}: its table of curves is empty")
        x_name, y_name = _axes(rows, p["x"], p["y"], _true(rows, p["measures"]))
        missing = [n for n in (x_name, y_name) if n not in rows[0]]
        if missing:
            raise ValueError(f"{item['title']}: no {' nor '.join(missing)} in its curves")
        x, y = _column(rows, x_name), _column(rows, y_name)
        names.add((x_name, y_name))
        label = item["title"]
        ax.plot(x, y, "-", color=COLOURS[k % len(COLOURS)], lw=1.8, marker=".", ms=3,
                label=label)  # fmt: skip
        row = {"run": label, "x": x_name, "y": y_name, "x at the end": x[-1],
               "y at the end": y[-1], "largest y": y.max(), "smallest y": y.min()}  # fmt: skip
        if reference is not None:
            order = np.argsort(x, kind="stable")
            inside = (reference[:, 0] >= x.min()) & (reference[:, 0] <= x.max())
            if np.all(np.diff(x) >= 0) or np.all(np.diff(x) <= 0):
                if inside.any():
                    gap = np.interp(reference[inside, 0], x[order], y[order]) - reference[inside, 1]
                    scale = float(np.abs(reference[:, 1]).max()) or 1.0
                    row["relative gap to the test"] = float(np.sqrt(np.mean(gap**2))) / scale
        table.append(row)
        values[f"#{item['number']} {y_name} at the end"] = float(y[-1])
        ctx.progress((k + 1) / (len(p["runs"]) + 1), item["title"])
    if reference is not None:
        ax.plot(reference[:, 0], reference[:, 1], "ko", ms=4, mfc="none", label="test")
    (x_name, y_name), *others = sorted(names)
    ax.set_xlabel(x_name if not others else "x (see the table)")
    ax.set_ylabel(y_name if not others else "y (see the table)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ctx.figure(fig, "curves", "curves of the runs")
    ctx.table(table, "runs", "the runs compared")
    return {"title": f"Curves of {len(table)} runs", "values": values}
