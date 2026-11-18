"""The tools of the interface: each a form of fields and a function run on items of the
workspace of the session, grouped in tabs (statistics, mechanics, fracture, stability,
...) and, in a tab, under the headings of its menu (:func:`menus`). The page builds the
tabs, their menus and the forms from :func:`catalog` and runs a tool as a job
(:class:`ToolJob`) whose progress it follows; a tool returns values, tables, figures,
videos and files, and may add morphologies or results to the workspace, from which any
other tab takes them in turn.

A tool is a function ``run(params, ctx)`` registered by :func:`tool` with its fields
(:class:`Field`): ``params`` are the values of the form, converted and checked (an item
field gives the item of the workspace, a dictionary with its microstructure under
``"micro"``), and ``ctx`` (:class:`Context`) writes the files of the result in a folder of
its own and reports the progress."""

from __future__ import annotations

import csv
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from morphogen.stopping import Stopped, checkpoint, stoppable

#: The tabs of the tools: id -> title, documentation and rank among the tabs of the page
TABS: dict[str, dict] = {}
#: The tools by name
TOOLS: dict[str, Tool] = {}
#: The kinds of fields of the forms
KINDS = (
    "float",
    "int",
    "str",
    "bool",
    "choice",
    "choices",
    "floats",
    "ints",
    "json",
    "text",
    "item",
    "items",
    "materials",
)


#: The sections of the menus of the page: create a morphology, characterize it, simulate
#: it, export and compare the results
SECTIONS = ("create", "characterize", "simulate", "share")
#: The section of the tabs that do not give theirs (the others: simulate)
SECTION_OF = {"statistics": "characterize", "design": "create", "interop": "share"}


def tab(name: str, title: str, doc: str, order: int, section: str | None = None) -> None:
    """Declare a tab of tools: its ``order`` places it among the tabs of its ``section``
    of the menus (:data:`SECTIONS`; by default :data:`SECTION_OF`, else ``simulate``)."""
    section = section or SECTION_OF.get(name, "simulate")
    if section not in SECTIONS:
        raise ValueError(f"tab {name}: section among {', '.join(SECTIONS)}")
    TABS[name] = {"name": name, "title": title, "doc": doc, "order": int(order), "section": section}


@dataclass(frozen=True)
class Field:
    """A field of the form of a tool: ``kind`` among :data:`KINDS` (``choice`` and
    ``choices`` among ``choices``; ``item`` an item of the workspace of the ``items``
    kinds, ``none`` the choice of none when it is optional, ``items`` several of them;
    ``materials`` the elastic constants of the phases of the item of the field ``of``, its
    ``default`` the ``young`` and ``poisson`` that the editor gives the phases that are not
    voids;
    ``needs``: the traits the item must have, see :data:`morphogen.gui.api.TRAITS`), shown
    only when the fields of ``when`` have one of the values given; ``group`` the box of the
    form that holds it (folded until opened if ``fold``), set by the ``groups`` of
    :func:`tool`; ``browse``: ``folder`` or ``file`` for a path, which the desktop
    application (:mod:`morphogen.gui.desktop`) lets choose in a dialog of the system."""

    name: str
    kind: str = "float"
    default: Any = None
    label: str = ""
    doc: str = ""
    unit: str = ""
    choices: tuple = ()
    items: tuple = ("morphology",)
    of: str = ""
    when: dict | None = None
    required: bool = False
    needs: tuple = ()
    group: str = ""
    fold: bool = False
    none: str = ""
    browse: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"field {self.name}: kind among {', '.join(KINDS)}")
        if self.browse not in ("", "folder", "file"):
            raise ValueError(f"field {self.name}: browse is folder or file")

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "label": self.label or self.name.replace("_", " "),
            "doc": self.doc,
            "unit": self.unit,
            "required": self.required,
        }
        if self.choices:
            out["choices"] = [list(c) if isinstance(c, tuple) else [c, c] for c in self.choices]
        if self.kind in ("item", "items"):
            out["items"] = list(self.items)
            out["needs"] = list(self.needs)
            if self.none:
                out["none"] = self.none
        if self.of:
            out["of"] = self.of
        if self.when:
            out["when"] = {k: v if isinstance(v, list) else [v] for k, v in self.when.items()}
        if self.group:
            out["group"] = self.group
            out["fold"] = self.fold
        if self.browse:
            out["browse"] = self.browse
        return out

    def _choice_values(self) -> list:
        return [c[0] if isinstance(c, tuple) else c for c in self.choices]

    def parse(self, value, session):
        """The value of the form, converted and checked."""
        if value in (None, "") or (isinstance(value, list) and not value and self.required):
            if self.required:
                raise ValueError(f"{self.label or self.name}: required")
            # the default of a field of materials is the material of the editor, not a value
            value = self.default if self.kind != "materials" else None
            if value is None:
                return None
        kind = self.kind
        name = self.label or self.name
        try:
            if kind == "float":
                return float(value)
            if kind == "int":
                return int(value)
            if kind == "bool":
                return bool(value) if not isinstance(value, str) else value.lower() == "true"
            if kind in ("floats", "ints"):
                cast = float if kind == "floats" else int
                if isinstance(value, str):
                    value = [v for v in value.replace(";", ",").replace(" ", ",").split(",") if v]
                values = [cast(v) for v in np.atleast_1d(value)]
                return values
        except (TypeError, ValueError):
            raise ValueError(f"{name}: expected {kind}, got {value!r}") from None
        if kind == "choice":
            if value not in self._choice_values():
                raise ValueError(f"{name}: one of {', '.join(map(str, self._choice_values()))}")
            return value
        if kind == "choices":
            values = [value] if isinstance(value, str) else list(value)
            unknown = [v for v in values if v not in self._choice_values()]
            if unknown:
                raise ValueError(f"{name}: among {', '.join(map(str, self._choice_values()))}")
            return values
        if kind == "items":  # several items: their keys, as a list or separated by commas
            keys = [value] if isinstance(value, str) else list(value)
            keys = [k.strip() for v in keys for k in str(v).split(",") if k.strip()]
            if not keys and self.required:
                raise ValueError(f"{name}: required")
            item = Field(self.name, "item", items=self.items, needs=self.needs, label=name)
            return [item.parse(k, session) for k in dict.fromkeys(keys)]
        if kind == "item":
            if session is None:
                raise ValueError(f"{name}: the workspace of a session is needed")
            from morphogen.gui.api import TRAITS, item_traits

            item = session.item(value, self.items)
            missing = [TRAITS[n] for n in self.needs if n not in item_traits(item)]
            if missing:
                raise ValueError(f"{name}: {item['title']} is not {' and '.join(missing)}")
            if item["kind"] == "behaviour":  # its material, the files it names by their paths
                from morphogen.gui.api import behaviour_material

                item = {**item, "material": behaviour_material(item, session.root)}
            return item
        if kind == "json":
            import json

            return json.loads(value) if isinstance(value, str) else value
        if kind == "materials":
            from morphogen.gui.api import _materials

            return _materials({"materials": value})
        return str(value)


@dataclass
class Tool:
    """A tool of the interface (see the module)."""

    name: str
    tab: str
    title: str
    doc: str
    fields: tuple[Field, ...]
    run: Callable
    order: int = 0
    menu: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tab": self.tab,
            "title": self.title,
            "doc": self.doc,
            "menu": self.menu,
            "fields": [f.to_dict() for f in self.fields],
        }

    def parse(self, params: dict, session) -> dict:
        """The values of the form, converted and checked (hidden fields left out)."""
        params = dict(params or {})
        out: dict = {}
        for f in self.fields:  # in order: the conditions refer to earlier fields
            if f.when and not all(
                out.get(k) in (v if isinstance(v, list) else [v]) for k, v in f.when.items()
            ):
                out[f.name] = None
                continue
            out[f.name] = f.parse(params.get(f.name), session)
        return out


def tool(
    name: str,
    tab: str,
    title: str,
    doc: str,
    fields=(),
    order: int = 0,
    groups=(),
    menu: str = "",
):
    """Register the function ``run(params, ctx)`` of a tool (see the module). ``menu``: the
    heading under which the menu of its tab lists it (:func:`menus`), with the tools that
    do the same kind of work; ``order`` ranks it among them. ``groups``: the boxes of its
    form, ``(title, first field)`` or ``(title, first field, True)`` for a box folded until
    opened (details seldom changed), each from its first field to the first field of the
    next; the fields before the first box stand above them."""

    def wrap(fn: Callable) -> Callable:
        known = TOOLS.get(name)
        if known is not None and known.run.__module__ != fn.__module__:
            raise ValueError(f"tool {name}: already the name of a tool of {known.run.__module__}")
        boxed = _grouped(name, tuple(fields), groups)
        TOOLS[name] = Tool(name, tab, title, doc, boxed, fn, order, str(menu))
        return fn

    return wrap


def _grouped(name: str, fields: tuple[Field, ...], groups) -> tuple[Field, ...]:
    """The ``fields`` in the boxes ``groups`` of :func:`tool`."""
    from dataclasses import replace

    starts = {g[1]: (g[0], bool(g[2]) if len(g) > 2 else False) for g in groups}
    unknown = set(starts) - {f.name for f in fields}
    if unknown:
        raise ValueError(f"tool {name}: no field {', '.join(sorted(unknown))} to start a group")
    out, box = [], ("", False)
    for f in fields:
        box = starts.get(f.name, box)
        out.append(replace(f, group=box[0], fold=box[1]) if box[0] else f)
    return tuple(out)


def _load() -> None:
    """Import the modules of the tools, every module of this package (they register
    themselves); a module that cannot be imported (a missing optional dependency) is
    left out with a warning."""
    import importlib
    import pkgutil

    for info in pkgutil.iter_modules(__path__):
        name = f"{__name__}.{info.name}"
        try:
            importlib.import_module(name)
        except ImportError as exc:  # pragma: no cover - optional dependencies
            warnings.warn(f"tools of {info.name} unavailable: {exc}", stacklevel=2)


def menus(tab: str) -> list[tuple[str, list[Tool]]]:
    """The menu of the tools of a tab: its headings (the ``menu`` of :func:`tool`; none for
    the tools that give none) in the order of their first tool, each with its tools in
    their order, so that the tools that do the same kind of work stand together."""
    _load()
    tools = sorted((t for t in TOOLS.values() if t.tab == tab), key=lambda t: t.order)
    return [(h, [t for t in tools if t.menu == h]) for h in dict.fromkeys(t.menu for t in tools)]


def catalog() -> dict:
    """The tabs (by their order) and the tools of the interface, those of a tab as its
    menu lists them (:func:`menus`: each with the heading ``menu`` it stands under)."""
    _load()
    tabs = sorted(TABS.values(), key=lambda t: t["order"])
    tools = [t for tab in tabs for _, listed in menus(tab["name"]) for t in listed]
    tools += sorted((t for t in TOOLS.values() if t.tab not in TABS), key=lambda t: t.order)
    return {"tabs": tabs, "tools": [t.to_dict() for t in tools]}


class Context:
    """What the run of a tool sees: the ``session`` (its workspace), the folder and key of
    its result, and the progress it reports (:meth:`progress`)."""

    def __init__(self, session, key: str, folder: Path, job: ToolJob | None = None):
        self.session = session
        self.key = key
        self.folder = Path(folder)
        self.job = job
        self.figures: list[dict] = []
        self.videos: list[dict] = []
        self.tables: list[dict] = []
        self.items: list[dict] = []

    def progress(self, fraction: float, message: str = "") -> None:
        """Report the progress; a checkpoint of :mod:`morphogen.stopping` (in the thread
        of the job: the tool stops there when asked to)."""
        if self.job is not None:
            self.job.report(fraction, message)
        checkpoint()

    def path(self, name: str) -> Path:
        return self.folder / name

    def url(self, name: str) -> str:
        return f"/files/{self.key}/{name}"

    def figure(self, figure, name: str, title: str = "") -> dict:
        """A figure of the result: a matplotlib figure (also as .svg) or an RGB array."""
        path = self.folder / f"{name}.png"
        if isinstance(figure, np.ndarray):
            from morphogen.viz.png import encode_png

            path.write_bytes(encode_png(figure))
        else:
            from morphogen.viz.export import save_figure

            save_figure(figure, path, dpi=130)
            save_figure(figure, self.folder / f"{name}.svg")
        out = {"title": title or name.replace("_", " "), "url": self.url(path.name)}
        self.figures.append(out)
        return out

    def video(self, frames, name: str, title: str = "", fps: float = 18, mode: str = "bounce"):
        from morphogen.gui.api import _video_format
        from morphogen.viz.video import write_video

        fmt = _video_format()
        path = self.folder / f"{name}.{fmt}"
        write_video(frames, path, fps=fps, mode=mode)
        out = {"title": title or name.replace("_", " "), "url": self.url(path.name), "format": fmt}
        self.videos.append(out)
        return out

    def table(self, rows: list[dict], name: str, title: str = "", shown: int = 40) -> dict:
        """A table of the result, written as .csv (its first ``shown`` rows in the page)."""
        columns: list[str] = []
        for row in rows:
            columns += [k for k in row if k not in columns]
        path = self.folder / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, columns)
            writer.writeheader()
            writer.writerows(rows)
        out = {
            "title": title or name.replace("_", " "),
            "columns": columns,
            "rows": [[_cell(row.get(c)) for c in columns] for row in rows[:shown]],
            "count": len(rows),
            "url": self.url(path.name),
        }
        self.tables.append(out)
        return out

    def add(self, kind: str, title: str, **data) -> dict:
        """Add a morphology or a result to the workspace (in a folder of its own)."""
        key, _ = self.session.folder()
        item = self.session.add(key, kind, title, **data)
        public = self.session.public(item)
        self.items.append(public)
        return public


#: The notation of the tables of stiffnesses and compliances, in space and in the plane
VOIGT_NOTE = "Voigt 11, 22, 33, 23, 31, 12, engineering shears (TFEL, Abaqus: 12, 13, 23)"
PLANE_NOTE = "Voigt 11, 22, 12, engineering shear"


def matrix_rows(matrix) -> list[dict]:
    """The rows of the table of a 6 x 6 matrix (3 x 3 in the plane) in morphoGen's Voigt
    order: the component of each row, then a column per component."""
    from morphogen.homogenization.postprocess import PLANE_ORDER, VOIGT_ORDER

    m = np.asarray(matrix, dtype=float)
    labels = PLANE_ORDER if m.shape == (3, 3) else VOIGT_ORDER
    return [
        {"": a, **{b: float(m[i, j]) for j, b in enumerate(labels)}} for i, a in enumerate(labels)
    ]


def elastic_tables(
    ctx,
    stiffness,
    of: str = "",
    *,
    names=("stiffness", "compliance"),
    units=("", ""),
    kind: str = "",
    compliance=None,
) -> np.ndarray:
    """The tables of a stiffness ``C`` and of its ``compliance`` ``S = C^-1`` (computed when
    not given) as .csv, named ``names``, titled by their ``kind`` (e.g. "apparent"), what
    they are ``of``, their notation and their ``units`` (those of the moduli and their
    inverse by default); returns ``S``."""
    from morphogen.homogenization.postprocess import compliance as inverse

    c = np.asarray(stiffness, dtype=float)
    s = inverse(c) if compliance is None else np.asarray(compliance, dtype=float)
    note = PLANE_NOTE if c.shape == (3, 3) else VOIGT_NOTE
    of, kind = (f" {of}" if of else ""), (f"{kind} " if kind else "")
    unit_c = units[0] or "units of the moduli"
    unit_s = units[1] or "their inverse"
    ctx.table(matrix_rows(c), names[0], f"{kind}stiffness C_ij{of}: {note}; {unit_c}")
    ctx.table(
        matrix_rows(s), names[1], f"{kind}compliance S_ij = C⁻¹{of}: the same notation; {unit_s}"
    )
    return s


def _cell(value):
    if isinstance(value, (float, np.floating)):
        return float(f"{float(value):.6g}") if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value if value is None or isinstance(value, (str, bool)) else str(value)


class ToolJob:
    """A tool running in a thread, with its progress and result (see the module); stopped
    (:meth:`stop`) at the next progress it reports or iteration of its solvers, it keeps
    nothing in the workspace."""

    def __init__(self, request: dict, session):
        _load()
        name = request.get("tool")
        if name not in TOOLS:
            raise ValueError(f"tool: one of {', '.join(sorted(TOOLS))}")
        self.tool = TOOLS[name]
        self.raw_params = dict(request.get("params") or {})  # as the form sent them
        self.params = self.tool.parse(self.raw_params, session)  # errors at once
        self.session = session
        self.key, self.folder = session.folder()
        self.started = time.perf_counter()
        self.state = "running"
        self.fraction = 0.0
        self.message = ""
        self.result: dict | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> ToolJob:
        self.thread.start()
        return self

    def stop(self) -> None:
        """Stop at the next checkpoint (:mod:`morphogen.stopping`)."""
        self._stop.set()

    def report(self, fraction: float, message: str = "") -> None:
        with self._lock:
            self.fraction = float(min(max(fraction, 0.0), 1.0))
            self.message = str(message)

    def _run(self):
        from morphogen.gui.api import _files, shown_warnings

        ctx = Context(self.session, self.key, self.folder, self)
        try:
            with warnings.catch_warnings(record=True) as caught, stoppable(self._stop):
                warnings.simplefilter("always")
                out = self.tool.run(self.params, ctx) or {}
            if "archive" not in out:
                archive = _archive(self.folder)
                if archive is not None:
                    out["archive"] = ctx.url(archive.name)
            result = {
                "tool": self.tool.name,
                "title": out.pop("title", self.tool.title),
                "values": _finite_values(out.pop("values", {})),
                "tables": ctx.tables + out.pop("tables", []),
                "figures": ctx.figures + out.pop("figures", []),
                "videos": ctx.videos + out.pop("videos", []),
                "items": ctx.items + out.pop("items", []),
                "text": out.pop("text", ""),
                "files": _files(self.key, self.folder),
                "time_s": round(time.perf_counter() - self.started, 2),
                "warnings": shown_warnings(caught),
                **out,
            }
            self._keep(result)
            with self._lock:
                self.result, self.state, self.fraction = result, "done", 1.0
        except Stopped:  # nothing kept: the items it made leave the workspace
            for key in [self.key] + [i["key"] for i in ctx.items]:
                if key in self.session.items:
                    self.session.remove(key)
            with self._lock:
                self.state = "stopped"
        except Exception as exc:  # reported to the page
            with self._lock:
                self.error, self.state = f"{type(exc).__name__}: {exc}", "error"

    def _keep(self, result: dict) -> None:
        """The run kept in the workspace as a result: its tool, the parameters of its form
        (to fill it again) and its result (to show it again), titled by the items it took;
        when the tool made its folder an item itself (the simulation of a loading path), the
        run joins that item, which keeps its kind."""
        took = []
        for f in self.tool.fields:
            if f.kind in ("item", "items"):
                value = self.params.get(f.name)
                values = value if isinstance(value, list) else [value]
                took += [f"#{v['number']}" for v in values if isinstance(v, dict)]
        title = self.tool.title + (f" ({', '.join(took)})" if took else "")
        values = dict(list(result["values"].items())[:6])
        # its mesh of fields, if any (cracks, damage, patterns): drawn by the 3D view
        result.update(key=self.key, view3d=bool(meshes_of_result(self.folder)))
        run = {"tool": self.tool.name, "tab": self.tool.tab, "params": self.raw_params,
               "result": result}  # fmt: skip
        made = self.session.items.get(self.key)
        if made is not None:
            made.update(run)
            return
        self.session.add(self.key, "result", title, info=values, view3d=result["view3d"], **run)

    def status(self) -> dict:
        with self._lock:
            out = {
                "state": self.state,
                "progress": self.fraction,
                "message": self.message,
                "elapsed_s": round(time.perf_counter() - self.started, 1),
            }
            if self.result is not None:
                out["result"] = self.result
            if self.error is not None:
                out["error"] = self.error
        return out


def meshes_of_result(folder: Path) -> list[Path]:
    """The meshes of the fields of the result of a tool (.vtu files of tetrahedra with
    their displacement: cracked bodies, damaged specimens, bifurcated patterns), the last
    (that of the last step) last; the meshes alone (``*_tet.vtu``) and the cohesive
    interfaces (``*_interfaces.vtu``, wedges) left out."""
    return sorted(
        p for p in Path(folder).glob("*.vtu") if not p.stem.endswith(("_tet", "_interfaces"))
    )


def _archive(folder: Path) -> Path | None:
    """An archive of every file of the result of a tool (subfolders kept), if any."""
    import zipfile

    files = [p for p in sorted(Path(folder).rglob("*")) if p.is_file() and p.suffix != ".zip"]
    if not files:
        return None
    path = Path(folder) / "result.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(folder).as_posix())
    return path


def _finite_values(values: dict) -> dict:
    out = {}
    for k, v in (values or {}).items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            if np.isfinite(v):
                out[k] = float(v) if isinstance(v, (float, np.floating)) else int(v)
        elif isinstance(v, (str, bool)):
            out[k] = v
    return out


@dataclass
class Watcher:
    """Reports the progress of a run that writes one file per step (``pattern`` in
    ``folder``, ``total`` of them) from a thread, until stopped."""

    ctx: Context
    folder: Path
    pattern: str
    total: int
    message: str = ""
    _stop: threading.Event = field(default_factory=threading.Event)

    def __enter__(self) -> Watcher:
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _watch(self):
        while not self._stop.wait(0.5):
            done = len(list(Path(self.folder).glob(self.pattern)))
            self.ctx.progress(done / max(self.total, 1), f"{self.message} {done} / {self.total}")
