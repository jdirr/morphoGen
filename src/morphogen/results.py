"""Database of the configurations tested and of their results, and Ashby charts.

A :class:`ResultsDatabase` is one SQLite file (Python's standard library). Every run
directory added (:meth:`ResultsDatabase.add_run`: the outputs of ``morphogen run``, or of
a run with ``outputs.database``) contributes one record per realization: its
configuration flattened into dotted keys (``model.type``, ``model.radius``,
``domain.size``, ``homogenization.materials.1.young``...) and its row of results
(measures, apparent properties, solid fraction, relative, normalized and specific values:
:mod:`morphogen.homogenization.postprocess`). Records are selected by conditions on any
key (:func:`parse_condition`): ``model.type == gyroid``, ``solid_fraction < 0.3``,
``model.type in gyroid,octet``, ``run_name ~ lattice*``.

Ashby charts (:func:`ashby_chart`) place records in a space of two properties, by default
normalized: a relative property (``E / E_s``) against the relative density (the solid
fraction), on logarithmic axes, one family per value of a key (the model by default),
with the scaling laws of cellular solids ``P / P_s = phi^n`` (Gibson & Ashby 1997:
``n = 1`` for stretching-dominated architectures, the rule of mixtures; ``n = 2`` for
bending-dominated open-cell foams).
"""

from __future__ import annotations

import csv
import fnmatch
import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np

#: Runs and their realizations; a record is the JSON of the flattened configuration
#: merged with the row of results of one realization.
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    name TEXT,
    added TEXT,
    config TEXT,
    version TEXT
);
CREATE TABLE IF NOT EXISTS samples (
    run INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    realization INTEGER,
    data TEXT,
    PRIMARY KEY (run, realization)
);
"""

#: Parts of a configuration that do not describe the microstructure or its computation.
_SKIPPED = ("outputs", "rve")

OPERATORS = ("==", "!=", "<=", ">=", "<", ">", " not in ", " in ", "~")


def flatten(data: dict, prefix: str = "") -> dict:
    """Nested mappings as one level of dotted keys; lists become JSON text."""
    out: dict = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        elif isinstance(value, (list, tuple)):
            out[name] = json.dumps(value)
        else:
            out[name] = value
    return out


def _value(text: str):
    """A condition's value: a number, a boolean, or a string."""
    t = text.strip().strip("'\"")
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return float(t)
    except ValueError:
        return t


def parse_condition(text: str) -> tuple[str, str, object]:
    """``"key OP value"`` with OP one of ``== != < <= > >= in, not in`` (comma-separated
    values) or ``~`` (shell pattern, e.g. ``lattice*``)."""
    for op in OPERATORS:
        key, sep, value = text.partition(op)
        if sep and key.strip():
            if op.strip() in ("in", "not in"):
                return key.strip(), op.strip(), [_value(v) for v in value.split(",")]
            return key.strip(), op, _value(value)
    raise ValueError(f"condition {text!r}: expected 'key OP value' with OP in == != < <= > >= in ~")


def _equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b) and isinstance(a, bool) == isinstance(b, bool)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return str(a) == str(b)


def matches(record: dict, conditions) -> bool:
    """Whether a record satisfies every condition (strings or parsed tuples); a missing
    key fails every condition but ``!=`` and ``not in``."""
    for cond in conditions:
        key, op, value = parse_condition(cond) if isinstance(cond, str) else cond
        if key not in record:
            if op in ("!=", "not in"):
                continue
            return False
        v = record[key]
        if op == "==":
            ok = _equal(v, value)
        elif op == "!=":
            ok = not _equal(v, value)
        elif op == "in":
            ok = any(_equal(v, w) for w in value)
        elif op == "not in":
            ok = not any(_equal(v, w) for w in value)
        elif op == "~":
            ok = fnmatch.fnmatchcase(str(v), str(value))
        else:
            try:
                a, b = float(v), float(value)
            except (TypeError, ValueError):
                return False
            ok = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
        if not ok:
            return False
    return True


class ResultsDatabase:
    """SQLite database of runs and of the results of their realizations."""

    def __init__(self, path):
        self.path = Path(path)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)

    def __enter__(self) -> ResultsDatabase:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def add_run(self, directory, *, name: str | None = None) -> int:
        """Add (or replace) the realizations of a run directory: its configuration
        (``run.json``) and results (``results.csv``, or the files of split runs). Returns
        the number of records."""
        from morphogen import pipeline

        directory = Path(directory).resolve()
        manifests = [directory / "run.json", *sorted(directory.glob("run_task*of*.json"))]
        manifest = next((m for m in manifests if m.exists()), None)
        if manifest is None:
            raise ValueError(f"{directory}: not the output directory of a run (no run.json)")
        info = json.loads(manifest.read_text(encoding="utf-8"))
        config = info.get("config", {})
        rows = pipeline.collect_results(directory)
        name = name or config.get("outputs", {}).get("name") or directory.name
        version = info.get("versions", {}).get("morphogen", "")
        return self.add_records(
            rows, name=name, config=config, path=str(directory), version=version
        )

    def add_records(
        self,
        rows: list[dict],
        *,
        name: str,
        config: dict | None = None,
        path: str | None = None,
        version: str = "",
    ) -> int:
        """Add the rows of a run (one per realization) with its configuration; a run of
        the same ``path`` (default: the name) is replaced."""
        config = config or {}
        base = flatten({k: v for k, v in config.items() if k not in _SKIPPED})
        key = path or f"name:{name}"
        with self.connection:
            self.connection.execute("DELETE FROM runs WHERE path = ?", (key,))
            cur = self.connection.execute(
                "INSERT INTO runs (path, name, added, config, version) VALUES (?, ?, ?, ?, ?)",
                (key, name, time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(config), version),
            )
            run = cur.lastrowid
            data = []
            for i, row in enumerate(rows):
                record = {"run": run, "run_name": name, **base, **row}
                realization = int(row.get("realization", i))
                data.append((run, realization, json.dumps(record)))
            self.connection.executemany(
                "INSERT OR REPLACE INTO samples (run, realization, data) VALUES (?, ?, ?)", data
            )
        return len(rows)

    def remove(self, run) -> int:
        """Remove a run (its id, path or name); returns the number of runs removed."""
        with self.connection:
            if isinstance(run, int) or str(run).isdigit():
                cur = self.connection.execute("DELETE FROM runs WHERE id = ?", (int(run),))
            else:
                text = str(run)
                resolved = str(Path(text).resolve())
                cur = self.connection.execute(
                    "DELETE FROM runs WHERE path IN (?, ?) OR name = ?", (text, resolved, text)
                )
        return cur.rowcount

    def runs(self) -> list[dict]:
        """Runs: id, name, path, date added, version and number of records."""
        rows = self.connection.execute(
            "SELECT r.id, r.name, r.path, r.added, r.version, COUNT(s.run) FROM runs r "
            "LEFT JOIN samples s ON s.run = r.id GROUP BY r.id ORDER BY r.id"
        ).fetchall()
        keys = ("id", "name", "path", "added", "version", "records")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def records(self, where=()) -> list[dict]:
        """Records satisfying every condition of ``where`` (see :func:`matches`)."""
        conditions = [parse_condition(c) if isinstance(c, str) else c for c in where]
        out = []
        for (data,) in self.connection.execute(
            "SELECT data FROM samples ORDER BY run, realization"
        ):
            record = json.loads(data)
            if matches(record, conditions):
                out.append(record)
        return out

    def keys(self, where=()) -> dict[str, dict]:
        """Keys of the records: how many records hold each, and whether its values are all
        numbers."""
        out: dict[str, dict] = {}
        for record in self.records(where):
            for k, v in record.items():
                entry = out.setdefault(k, {"count": 0, "numeric": True})
                entry["count"] += 1
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    entry["numeric"] = False
        return dict(sorted(out.items()))


def record_run(directory, database, *, name: str | None = None) -> int:
    """Add a run directory to a database file (created if needed)."""
    with ResultsDatabase(database) as db:
        return db.add_run(directory, name=name)


def write_records(records: list[dict], path, keys=None) -> Path:
    """CSV of records (all their keys, or ``keys``): the table view of a chart."""
    if keys is None:
        keys = []
        for r in records:
            keys.extend(k for k in r if k not in keys)
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(keys), lineterminator="\n")
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in keys})
    return path


# -- Ashby charts ----------------------------------------------------------------------------

#: Categorical slots of scatter charts (any two families can be neighbours: three hues pass
#: the colour-vision checks together; more families fold into "Other"), light and dark.
SERIES = {"light": ("#2a78d6", "#eb6834", "#1baf7a"), "dark": ("#3987e5", "#d95926", "#199e70")}
MARKERS = ("o", "s", "^")
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}
OTHER = "Other"


def default_axes(records: list[dict]) -> tuple[str, str]:
    """The relative density and the first relative property found (Young's modulus first,
    then the bulk and shear moduli, then any)."""
    keys = {k for r in records for k in r}
    relative = sorted(k for k in keys if k.endswith(".relative"))
    for prop in ("E1", "bulk_modulus", "shear_modulus", "conductivity"):
        hits = [k for k in relative if prop in k.split(".")]
        if hits:
            return "solid_fraction", hits[0]
    if relative:
        return "solid_fraction", relative[0]
    raise ValueError("no relative property in the records: give the keys of the axes")


def ashby_points(
    records: list[dict],
    x: str | None = None,
    y: str | None = None,
    by: str | None = "model.type",
    *,
    log: bool = True,
    groups: list | None = None,
    max_groups: int = 3,
) -> dict:
    """Points of an Ashby chart: the families (values of ``by``) with their points, the
    first ``max_groups`` in their own colour (by size, or the order of ``groups``), the
    others folded into "Other". Records without finite values (positive ones on
    logarithmic axes) are left out."""
    if x is None or y is None:
        dx, dy = default_axes(records)
        x, y = x or dx, y or dy
    families: dict[str, list] = {}
    for r in records:
        try:
            vx, vy = float(r[x]), float(r[y])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(vx) and math.isfinite(vy)) or (log and (vx <= 0 or vy <= 0)):
            continue
        name = str(r.get(by, "")) if by else ""
        families.setdefault(name, []).append(
            {"x": vx, "y": vy, "run": r.get("run_name", ""), "realization": r.get("realization")}
        )
    order = [str(g) for g in groups if str(g) in families] if groups else []
    order += sorted((g for g in families if g not in order), key=lambda g: -len(families[g]))
    shown, folded = order[:max_groups], order[max_groups:]
    out = [{"name": g, "slot": i, "points": families[g]} for i, g in enumerate(shown)]
    if folded:
        pts = [p for g in folded for p in families[g]]
        out.append({"name": OTHER, "slot": None, "points": pts, "members": folded})
    return {"x": x, "y": y, "by": by, "log": log, "groups": out}


#: Names and symbols of the properties in the keys of the records.
_NAMES = {
    "bulk_modulus": ("bulk modulus", "K"),
    "shear_modulus": ("shear modulus", "G"),
    "conductivity": ("conductivity", "k"),
    "mean": ("conductivity", "k"),
    "E1": ("Young's modulus E1", "E"),
    "E2": ("Young's modulus E2", "E"),
    "E3": ("Young's modulus E3", "E"),
    "G12": ("shear modulus G12", "G"),
    "G23": ("shear modulus G23", "G"),
    "G31": ("shear modulus G31", "G"),
}


def axis_label(key: str) -> str:
    """Readable name of a key: the relative density, a relative property..."""
    if key == "solid_fraction":
        return "Relative density ρ / ρs (solid fraction)"
    if key == "density":
        return "Density ρ"
    parts = key.split(".")
    kind = parts[-1] if parts[-1] in ("relative", "normalized", "specific") else None
    core = parts[:-1] if kind else parts
    bcs = ("kubc", "periodic", "subc", "mixed")
    bc = next((p for p in core if p in bcs), None)
    core = [p for p in core if p not in bcs] or [key]
    name, symbol = _NAMES.get(core[-1], (core[-1].replace("_", " "), "P"))
    entry, notes = core[-1], [bc] if bc else []
    if core[0] == "stiffness" and len(entry) == 3 and entry[0] in "CS" and entry[1:].isdigit():
        # an entry of the stiffness or of its compliance, in morphoGen's Voigt notation
        name = f"{'stiffness' if entry[0] == 'C' else 'compliance'} {entry}"
        symbol, notes = entry, [*notes, "Voigt 11 22 33 23 31 12, engineering shears"]
    suffix = f", {bc}" if bc else ""
    if kind == "relative":
        return f"Relative {name} ({symbol} / {symbol}s{suffix})"
    if kind == "normalized":
        return f"Normalized {name} ({symbol} / (ρ/ρs {symbol}s){suffix})"
    if kind == "specific":
        return f"Specific {name} ({symbol} / ρ{suffix})"
    return f"{name[:1].upper()}{name[1:]}" + (f" ({'; '.join(notes)})" if notes else "")


def _hull(points: np.ndarray) -> np.ndarray | None:
    """Convex hull (in order) of 2D points, or None for fewer than three points or a
    degenerate set."""
    if len(points) < 3:
        return None
    from scipy.spatial import ConvexHull, QhullError

    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        return None


def ashby_chart(
    records: list[dict],
    path,
    x: str | None = None,
    y: str | None = None,
    by: str | None = "model.type",
    *,
    log: bool = True,
    guides=None,
    groups: list | None = None,
    theme: str = "light",
    title: str | None = None,
) -> Path:
    """Ashby chart of records as a PNG, SVG or PDF file (needs matplotlib): one family per
    value of ``by`` (three in colour, with their markers, envelopes and direct labels;
    others in "Other"), on logarithmic axes by default, with the lines ``y = x^n`` for
    ``guides`` (default: n = 1 and 2 when the axes are relative)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        raise ImportError("Ashby charts need matplotlib: pip install 'morphogen[plot]'") from None
    if theme not in THEMES:
        raise ValueError(f"theme: one of {list(THEMES)}")
    data = ashby_points(records, x, y, by, log=log, groups=groups)
    if not any(g["points"] for g in data["groups"]):
        raise ValueError(f"no record has values of {data['x']!r} and {data['y']!r}")
    colours, ink = SERIES[theme], THEMES[theme]
    relative = data["x"] == "solid_fraction" and data["y"].endswith(".relative")
    if guides is None:
        guides = (1, 2) if relative and log else ()
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    fig.patch.set_facecolor(ink["surface"])
    ax.set_facecolor(ink["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=ink["axis"], labelcolor=ink["secondary"], labelsize=8, width=0.8)
    ax.grid(True, which="major", color=ink["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    count = sum(len(g["points"]) for g in data["groups"])
    placed: list = []  # boxes of the labels drawn, in display coordinates
    for g in data["groups"]:
        pts = np.array([[p["x"], p["y"]] for p in g["points"]])
        colour = ink["muted"] if g["slot"] is None else colours[g["slot"]]
        marker = "o" if g["slot"] is None else MARKERS[g["slot"]]
        hull = None if g["slot"] is None else _hull(np.log10(pts) if log else pts)
        if hull is not None:  # the envelope of a family (not of the others folded)
            ax.fill(*(10.0**hull.T if log else hull.T), color=colour, alpha=0.1, linewidth=0)
        if g["slot"] is not None:
            label = g["name"]
        elif len(g["members"]) == 1:
            label = f"{OTHER}: {g['members'][0]}"
        else:
            label = f"{OTHER} ({len(g['members'])} families)"
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=40,
            marker=marker,
            color=colour,
            edgecolors=ink["surface"],
            linewidths=1.2,
            zorder=3,
            label=label or "all",
        )
    lo, hi = ax.get_xlim(), ax.get_ylim()
    guide_marks = []
    for n in guides:
        xs = np.geomspace(max(lo[0], 1e-12), lo[1], 64) if log else np.linspace(*lo, 64)
        ys = xs ** float(n)
        keep = (ys >= hi[0]) & (ys <= hi[1])
        if keep.sum() < 2:
            continue
        ax.plot(xs[keep], ys[keep], color=ink["muted"], linewidth=0.8, zorder=1)
        guide_marks.append((f"n = {n:g}", np.column_stack([xs[keep], ys[keep]])))
    ax.set_xlim(lo)
    ax.set_ylim(hi)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox

    half = 4.0 * fig.dpi / 72.0  # the markers (s = 40 points^2), that labels must not hide
    for g in data["groups"]:
        for p in g["points"]:
            cx, cy = ax.transData.transform((p["x"], p["y"]))
            placed.append(Bbox.from_extents(cx - half, cy - half, cx + half, cy + half))

    def place(text, anchors, offsets, **kw) -> None:
        """Draw a label at the first anchor and offset where it overlaps no other label
        and no marker (``anchors``: one point or a list of them)."""
        points = [anchors] if np.ndim(anchors) == 1 else list(anchors)
        frame = ax.get_window_extent(renderer)
        for xy in points:
            for dx, dy, ha in offsets:
                t = ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, **kw)
                box = t.get_window_extent(renderer).expanded(1.05, 1.15)
                inside = frame.contains(box.x0, box.y0) and frame.contains(box.x1, box.y1)
                if inside and not any(box.overlaps(b) for b in placed):
                    placed.append(box)
                    return
                t.remove()

    if len(data["groups"]) > 1:  # beside the plot: it hides no point
        legend = ax.legend(
            frameon=False,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            title=by,
            scatterpoints=1,
        )
        plt.setp(legend.get_texts(), color=ink["secondary"])
        legend.get_title().set_color(ink["secondary"])
        legend.get_title().set_fontsize(8)
    for text, line in guide_marks:  # the scaling laws, from their left end along them,
        # under or over the rising line rather than across it
        anchors = [line[int(f * (len(line) - 1))] for f in (0.0, 0.2, 0.4, 0.6)]
        offsets = [(6, -12, "left"), (-6, 4, "right")]
        place(text, anchors, offsets, fontsize=7, color=ink["muted"])
    if len(data["groups"]) > 1:  # direct labels of the families, where they fit
        for g in data["groups"]:
            if g["slot"] is None:
                continue
            pts = np.array([[p["x"], p["y"]] for p in g["points"]])
            k = int(np.argmin(pts[:, 0]))
            place(
                g["name"],
                pts[k],
                [(-7, -3, "right"), (0, -13, "center"), (0, 7, "center"), (7, -3, "left")],
                fontsize=8,
                color=ink["secondary"],
            )
    if log:
        from matplotlib.ticker import FuncFormatter, LogLocator

        plain = FuncFormatter(lambda v, _: f"{v:g}")
        for axis, (a, b) in ((ax.xaxis, lo), (ax.yaxis, hi)):
            if np.log10(b / a) < 2.0:  # less than two decades: ticks at 1, 2, 5
                axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(plain)
            axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel(axis_label(data["x"]), color=ink["secondary"], fontsize=9)
    ax.set_ylabel(axis_label(data["y"]), color=ink["secondary"], fontsize=9)
    head = title or f"{axis_label(data['y'])} — {count} realization{'s' * (count != 1)}"
    ax.set_title(head, color=ink["ink"], fontsize=10, loc="left")
    path = Path(path)
    fig.savefig(path, facecolor=ink["surface"], bbox_inches="tight")
    plt.close(fig)
    return path
