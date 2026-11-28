"""Actions of the web interface, as plain functions of JSON-like data (testable without a
server). Images are PNG data URLs or files of the session folder.

The interface follows the work on a microstructure: a morphology is generated (or comes
out of an optimization), simulated, then exported. Each result is kept in the workspace
of the session (:class:`Session`) under a key, so that a morphology can be sent to the
simulation, to the export or to the optimization, and a simulation to the export or to
the optimization. The views of a morphology are sections of its level set
(:mod:`morphogen.viz.sections`, anti-aliased, up to 2048 pixels per edge without a voxel
image of the volume); voxel images are made on demand, by the exports and the FFT
simulations.

The simulations run the steps of the configurations of the command line on the
morphology of the workspace; their configuration is downloadable, to rerun them with
``morphogen run`` (a morphology generated with a seed is the realization 0 of a run with
that seed). The figures, videos and files are drawn from the fields of a simulation by the
export, without solving again (:func:`export`)."""

from __future__ import annotations

import base64
import csv
import itertools
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings
import zipfile
from pathlib import Path

import numpy as np
import yaml

from morphogen import analysis, io
from morphogen.domain import Domain
from morphogen.models.base import get_model, list_models, model_from_config
from morphogen.seeding import derive_seed
from morphogen.viz.png import encode_png, slice_of, to_rgb, upscale

#: Limits that keep the interface responsive (it is meant for exploration), besides the
#: free memory (:func:`check_memory`).
MAX_RESOLUTION = 2048  # pixels per edge of the sections of the level sets
MAX_SHAPE = 2048  # voxels per edge of the voxel images of the exports
MAX_FFT_SHAPE = 256  # voxels per edge of the FFT simulations
MAX_FRAMES = 60
MAX_OPT_VOXELS = 64**3
#: Resolution of the measures of the level sets (cells per edge)
MEASURE_RESOLUTION = 128
#: Sections larger than this are files of the session, smaller ones data URLs
INLINE_PIXELS = 512
#: Gaussian smoothing (voxels) of the level sets of the optimized designs
DESIGN_SMOOTHING = 1.0


def png_url(plane: np.ndarray, kind: str = "auto", size: int = 320, vmin=None, vmax=None) -> str:
    rgb = to_rgb(plane, kind, vmin, vmax)
    rgb = upscale(rgb, max(1, size // max(rgb.shape[:2])))
    return _data_url(rgb)


def _data_url(rgb: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(rgb)).decode("ascii")


def sections(volume: np.ndarray, kind: str = "auto", **kw) -> list[dict]:
    """Middle sections of a voxel array (one for planar volumes) as PNG data URLs."""
    if volume.shape[2] == 1:
        return [{"title": "x-y", "png": png_url(slice_of(volume, 2), kind, **kw)}]
    names = ((0, "y-z section"), (1, "x-z section"), (2, "x-y section"))
    return [{"title": t, "png": png_url(slice_of(volume, a), kind, **kw)} for a, t in names]


def available_memory() -> float | None:
    """Bytes of physical memory available (None when unknown)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [  # the layout of MEMORYSTATUSEX
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong),
                    ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("extended", ctypes.c_ulonglong),
                ]

            status = Status()
            status.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return float(status.available)
            return None
        if sys.platform == "darwin":  # no SC_AVPHYS_PAGES: the pages vm_stat counts free
            return _vm_stat_available(_vm_stat())
        return float(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError, subprocess.SubprocessError):
        return None


def _vm_stat() -> str:
    """The output of macOS's ``vm_stat`` (the statistics of the virtual memory)."""
    return subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10,
                          check=True).stdout  # fmt: skip


def _vm_stat_available(text: str) -> float | None:
    """Bytes of memory available by the output of macOS's ``vm_stat``: its free,
    speculative and inactive pages (psutil's measure); None if it does not give them."""
    size = re.search(r"page size of (\d+) bytes", text)
    pages = dict(re.findall(r"^Pages (free|inactive|speculative):\s+(\d+)\.?\s*$", text, re.M))
    if size is None or not {"free", "inactive"} <= pages.keys():
        return None
    return float(int(size.group(1)) * sum(int(n) for n in pages.values()))


def shown_warnings(caught) -> list[str]:
    """The warnings of a computation worth showing in the page: those of morphoGen and its
    models, not the deprecations of the libraries (e.g. raised while Numba compiles)."""
    noise = (DeprecationWarning, PendingDeprecationWarning, FutureWarning, ImportWarning)
    out: list[str] = []
    for w in caught:
        module = getattr(w.category, "__module__", "") or ""
        if issubclass(w.category, noise) or module.startswith(("numba", "llvmlite")):
            continue
        if str(w.message) not in out:
            out.append(str(w.message))
    return out


def check_memory(nbytes: float, what: str) -> None:
    """Refuse a computation that would need more than half the free memory."""
    free = available_memory()
    if free is not None and nbytes > 0.5 * free:
        raise ValueError(
            f"{what}: about {nbytes / 2**30:.1f} GB, more than half of the free memory "
            f"({free / 2**30:.1f} GB); choose a lower resolution"
        )


def examples() -> dict[str, dict]:
    """The example configuration of each model (the template of its name, else the first
    template of that model): its parameters (``params``) and its domain (``domain``,
    whose scale its parameters assume), which fill the forms of the interface, so that a
    model can be run as it opens."""
    from importlib import resources

    folder = resources.files("morphogen") / "templates"
    files = sorted((p for p in folder.iterdir() if p.name.endswith(".yaml")), key=lambda p: p.name)
    out: dict[str, dict] = {}
    for exact in (True, False):
        for path in files:
            try:
                spec = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError:
                continue
            model = spec.get("model") if isinstance(spec, dict) else None
            if not isinstance(model, dict) or not isinstance(model.get("type"), str):
                continue
            name = model["type"]
            if name in out or exact != (path.name == f"{name}.yaml"):
                continue
            domain = spec.get("domain") if isinstance(spec.get("domain"), dict) else {}
            out[name] = {
                "params": {k: v for k, v in model.items() if k != "type"},
                "domain": {k: domain[k] for k in ("size", "periodic") if k in domain},
            }
    return out


#: The families of the models: the groups of the menu of the models, in this order (the
#: models of no family come last, under "Other models")
MODEL_FAMILIES = {
    "Random sets and particles": (
        "boolean_spheres", "penetrable_spheres", "hardcore_spheres", "cox_spheres",
        "dead_leaves", "transparent_dead_leaves", "gaussian_excursion", "reconstruction",
        "image",
    ),
    "Fibres and layers": (
        "boolean_cylinders", "poisson_fibers", "hardcore_fibers", "curved_fibers",
        "random_walk_fibers", "long_fibers", "cross_linked_fibers", "delaunay_network",
        "random_laminate", "woven_textile",
    ),
    "Tessellations and foams": (
        "voronoi", "laguerre", "seeded_laguerre", "nested_grains", "ebsd", "open_foam",
        "level_set_foam", "honeycomb",
    ),
    "Lattices and TPMS": (
        "strut_lattice", "random_lattice", "hierarchical_lattice", "scale_rich_network", "tpms",
        "fractal_sponge", "microgen",
    ),
    "Bio-inspired": ("nacre", "bouligand", "osteons"),
    "Growth and patterns": (
        "cellular_automaton", "activator_inhibitor", "reaction_diffusion", "nucleation_growth",
        "aggregation", "branching", "spinodal",
    ),
    "Assemblies and combinations": ("tile_assembly", "combination", "coating"),
}  # fmt: skip


def _family(name: str) -> tuple[int, str]:
    for k, (family, names) in enumerate(MODEL_FAMILIES.items()):
        if name in names:
            return k, family
    return len(MODEL_FAMILIES), "Other models"


def _summary(doc: str) -> str:
    """The first paragraph of a docstring as one line, without its marks of code."""
    first = doc.strip().split("\n\n")[0]
    return " ".join(line.strip() for line in first.splitlines()).replace("``", "")


def models() -> list[dict]:
    """Every model with its documentation, its summary (the first paragraph of it), family,
    phases, parameters and the example values of its parameters, by family
    (:data:`MODEL_FAMILIES`) and name."""
    out = []
    example = examples()
    for name in sorted(list_models(), key=lambda n: (_family(n)[0], n)):
        cls = get_model(name)
        doc = (cls.__doc__ or "").strip()
        parameters = cls.parameters()
        names = {p["name"] for p in parameters}
        given = example.get(name, {"params": {}, "domain": {}})
        out.append(
            {
                "name": name,
                "family": _family(name)[1],
                "summary": _summary(doc) or name,
                "doc": doc,
                "phases": {str(k): v for k, v in cls.phases.items()},
                "parameters": parameters,
                "example": {k: v for k, v in given["params"].items() if k in names},
                "example_domain": given["domain"],
                "evolution": hasattr(cls, "evolution"),
                # conforming meshes: the implicit functions of the phases of every model
                "meshable": True,
            }
        )
    return out


def _clean(params: dict) -> dict:
    """Parameters from the form: empty values are left to their defaults."""
    return {k: v for k, v in params.items() if v not in (None, "")}


def _domain(spec: dict) -> Domain:
    size = spec.get("size", 10.0)
    return Domain(size=size, periodic=bool(spec.get("periodic", True)))


def _seed(value) -> int:
    """The seed of the form, or a fresh one (recorded, so that the realization can be
    simulated and rerun)."""
    if value in (None, ""):
        return int(np.random.SeedSequence().entropy % 2**31)
    return int(value)


def _resolution(value, default: int = 512) -> int:
    n = default if value in (None, "", 0) else int(value)
    if not 8 <= n <= MAX_RESOLUTION:
        raise ValueError(f"resolution: from 8 to {MAX_RESOLUTION} pixels per edge")
    return n


def _shape(value, limit: int = MAX_SHAPE) -> int | None:
    if value in (None, "", 0):
        return None
    n = int(value)
    if not 2 <= n <= limit:
        raise ValueError(f"voxels per edge: from 2 to {limit}")
    return n


def _voxels(domain: Domain, n: int, bytes_per_voxel: float, what: str) -> None:
    """Check the memory of a voxel image of ``n`` voxels along the longest edge."""
    lengths = np.asarray(domain.lengths, dtype=float)
    count = float(np.prod(np.maximum(1, np.rint(lengths / (lengths.max() / n)))))
    check_memory(count * bytes_per_voxel, what)


class Session:
    """The files written by the interface (one folder per result) and its workspace: the
    morphologies, simulations, behaviours (of materials) and results of the tools of the
    session, under the keys of their folders, with their objects in memory (the
    microstructures)."""

    #: The fields of the items sent to the page
    PUBLIC = (
        "key",
        "kind",
        "number",
        "title",
        "origin",
        "source",
        "thumbnail",
        "info",
        "phases",
        "files",
        "property",
        "planar",
        "fraction",
        "materials",
        "tool",
        "tab",
        "view3d",
    )

    #: The items taken out of the workspace that :meth:`restore` can bring back, the last
    #: ones (the close button of the page offers to undo its last closes)
    CLOSED = 16

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, dict] = {}
        self._closed: dict[str, dict] = {}  # the items removed, the last CLOSED of them
        self._count = 0  # the numbers given, never given again (after a removal)
        self._lock = threading.Lock()

    def folder(self) -> tuple[str, Path]:
        key = uuid.uuid4().hex[:12]
        path = self.root / key
        path.mkdir()
        return key, path

    def file(self, key: str, name: str) -> Path:
        path = (self.root / key / name).resolve()
        if self.root.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(name)
        return path

    def add(self, key: str, kind: str, title: str, **data) -> dict:
        """Keep a result in the workspace (``kind``: "morphology", "simulation",
        "behaviour": the material of a material point or of an import of Z-set, see
        :func:`behaviour_material`, or "result": the run of a tool, see
        :class:`morphogen.gui.tools.ToolJob`)."""
        with self._lock:
            self._count += 1
            number = self._count
            item = {"key": key, "kind": kind, "number": number, "title": f"#{number} {title}"}
            item.update(data)
            self.items[key] = item
        return item

    def item(self, key, kinds=("morphology", "simulation")) -> dict:
        with self._lock:
            item = self.items.get(str(key))
        if item is None or item["kind"] not in kinds:
            raise ValueError(f"no {' or '.join(kinds)} {key!r} in the workspace of this session")
        return item

    def remove(self, key) -> None:
        """Take an item out of the workspace (its files stay in the session folder); the
        last :data:`CLOSED` items removed can be brought back by :meth:`restore`."""
        with self._lock:
            item = self.items.pop(str(key), None)
            if item is None:
                raise ValueError(f"no item {key!r} in the workspace of this session")
            self._closed.pop(str(key), None)
            self._closed[str(key)] = item
            while len(self._closed) > self.CLOSED:
                self._closed.pop(next(iter(self._closed)))

    def restore(self, key) -> dict:
        """Bring back an item that :meth:`remove` took out of the workspace, with its
        number, title and objects (the undo of the close button of the page)."""
        with self._lock:
            item = self._closed.pop(str(key), None)
            if item is None:
                raise ValueError(f"no item {key!r} closed in this session to bring back")
            self.items[str(key)] = item
        return item

    def rename(self, key, title) -> dict:
        """Give an item another title (after its number, which it keeps)."""
        title = " ".join(str(title or "").split())
        if not title:
            raise ValueError("title: some text")
        with self._lock:
            item = self.items.get(str(key))
            if item is None:
                raise ValueError(f"no item {key!r} in the workspace of this session")
            item["title"] = f"#{item['number']} {title[:120]}"
        return item

    def morphology_of(self, key) -> dict:
        """The morphology of an item: itself, or that of a simulation."""
        item = self.item(key)
        return item if item["kind"] == "morphology" else self.item(item["source"], ("morphology",))

    def public(self, item: dict) -> dict:
        out = {k: v for k, v in item.items() if k in self.PUBLIC}
        out["traits"] = item_traits(item)
        return out

    def workspace(self) -> list[dict]:
        """The items of the workspace for the page (without their objects), newest first."""
        with self._lock:
            items = sorted(self.items.values(), key=lambda i: -i["number"])
        return [self.public(i) for i in items]


#: The traits of the morphologies that some tools need (``Field(needs=...)``): the page
#: offers them only the items that have them
TRAITS = {
    "model": "the realization of a model (generated in Generate)",
    "grains": "a polycrystal (a Voronoi or Laguerre tessellation)",
    "seeds": "grains grown from seeds (a realization of seeded_laguerre)",
    "curves": "the run of a loading path, a material point, an identification, a learned "
    "law or a dynamic test (its curves)",
    "framework": "a network of fibres or struts (a realization of delaunay_network, open_foam "
    "or random_lattice, which keep their framework for a frame of beams)",
}
#: The tools whose runs have curves (:func:`curves_of`), and the names of their tables
CURVE_TOOLS = {
    "path": "sample_0000_*_*.csv",
    "point": "point_*.csv",
    "identify": "identified_*.csv",
    "learn": "learned_*.csv",
    "radioss_test": "curves.csv",
    "radioss_results": "curves.csv",
}


def item_traits(item: dict) -> list[str]:
    """The traits of an item of the workspace (:data:`TRAITS`)."""
    micro = item.get("micro")
    out = ["model"] if item.get("model") else []
    if micro is not None and hasattr(micro, "cell_of") and hasattr(micro, "germs"):
        out.append("grains")
    if micro is not None and callable(getattr(micro, "seed_comparison", None)):
        out.append("seeds")
    if micro is not None and (getattr(micro, "metadata", None) or {}).get("framework"):
        out.append("framework")
    if item.get("tool") in CURVE_TOOLS:
        out.append("curves")
    return out


def curves_of(item: dict, root) -> Path:
    """The table of the curves of the run of a loading path, a material point or an
    identification (the first of a path under several boundary conditions)."""
    pattern = CURVE_TOOLS.get(item.get("tool"))
    found = sorted((Path(root) / item["key"]).glob(pattern)) if pattern else []
    if not found:
        raise ValueError(
            f"{item['title']}: no curves (a loading path, a material point, an identification)"
        )
    return found[0]


# -- behaviours ----------------------------------------------------------------------------

#: The names of the behaviours of the materials (their keys in the configurations)
BEHAVIOUR_NAMES = {
    "crystal_plasticity": "crystal plasticity",
    "porous_plasticity": "porous plasticity",
    "plasticity": "von Mises plasticity",
    "mfront": "MFront behaviour",
    "learned": "learned law",
}


def behaviour_summary(config: dict) -> tuple[str, dict]:
    """The name of the behaviour of a material (a mapping of the configurations of
    :meth:`~morphogen.homogenization.materials.Material.from_config`) and its main
    constants, for the behaviours of the workspace."""
    kind = next((k for k in BEHAVIOUR_NAMES if k in config), None)
    name = BEHAVIOUR_NAMES.get(kind, "elasticity")
    hill = (config.get("plasticity") or {}).get("hill") if kind == "plasticity" else None
    if hill is not None:  # the hardenings of von Mises on Hill's criterion
        name = "Hill plasticity"
    info: dict = {"behaviour": name}
    if "young" in config:
        info.update(E=config["young"], ν=config.get("poisson"))
    elif "cubic" in config:
        info["C11, C12, C44"] = ", ".join(f"{c:.4g}" for c in config["cubic"])
    plastic = config.get("plasticity") or config.get("porous_plasticity") or {}
    if "yield" in plastic:
        info["R0"] = plastic["yield"]
    if hill is not None:
        from morphogen.homogenization.behaviours import HILL_COMPONENTS

        if isinstance(hill, dict):  # by component, 1 where not given
            hill = {str(k): v for k, v in hill.items()}
            hill = [hill.get(c, 1.0) for c in HILL_COMPONENTS]
        info[", ".join(f"h{c}" for c in HILL_COMPONENTS)] = ", ".join(
            f"{float(h):.4g}" for h in hill
        )
    if "porosity" in plastic:
        info["f0"] = plastic["porosity"]
    families = (config.get("crystal_plasticity") or {}).get("families") or {}
    if families:
        params = next(iter(families.values()))
        info.update(families=", ".join(families), τ0=params.get("tau0"))
    if kind == "mfront":
        info["MFront"] = config["mfront"].get("behaviour")
    return name, {k: v for k, v in info.items() if v is not None}


def behaviour_config(config: dict, folder) -> dict:
    """A material (a mapping of the configurations) made the behaviour of the workspace of
    ``folder``: its MFront source (or the model of its learned law) copied there and named
    by its name (its library, if any, kept where it is), so that the behaviour is saved
    and opened with the workspace."""
    import copy

    config = copy.deepcopy(config)
    for key, name in (("mfront", "source"), ("learned", "model")):
        spec = config.get(key) if isinstance(config, dict) else None
        if isinstance(spec, dict) and spec.get(name):
            source = Path(spec[name])
            if source.is_absolute() and source.is_file():
                target = Path(folder) / source.name
                if not target.exists():
                    shutil.copyfile(source, target)
                spec[name] = source.name
    return config


def behaviour_material(item: dict, root) -> dict:
    """The material of a behaviour of the workspace (a mapping of the configurations), the
    files it names by their names in its folder (the MFront source of an import of
    Z-set, the model of a learned law) given by their paths under ``root``, the folder of
    the session."""
    import copy

    material = copy.deepcopy(item["material"])
    mfront = material.get("mfront") if isinstance(material, dict) else None
    if isinstance(mfront, dict):
        for key in ("source", "library"):
            if mfront.get(key) and not Path(mfront[key]).is_absolute():
                mfront[key] = str(Path(root) / item["key"] / mfront[key])
    learned = material.get("learned") if isinstance(material, dict) else None
    if isinstance(learned, dict) and learned.get("model"):
        if not Path(learned["model"]).is_absolute():
            learned["model"] = str(Path(root) / item["key"] / learned["model"])
    return material


# -- morphologies --------------------------------------------------------------------------


def _png(rgb: np.ndarray, folder: Path | None, key: str | None, name: str) -> str:
    """A view: a data URL when small (or without a session), else a file of the session."""
    if folder is None or max(rgb.shape[:2]) <= INLINE_PIXELS:
        return _data_url(rgb)
    (folder / name).write_bytes(encode_png(rgb))
    return f"/files/{key}/{name}?inline"


def _thumbnail(rgb: np.ndarray, size: int = 96) -> str:
    step = max(1, max(rgb.shape[:2]) // size)
    return _data_url(np.ascontiguousarray(rgb[::step, ::step]))


def morphology_views(micro, resolution: int, folder: Path | None = None, key=None) -> dict:
    """Sections of the implicit functions of a morphology (anti-aliased boundaries: of its
    level set, or of the functions of its phases when it has several; else the phases of
    the pixel centres) at ``resolution`` pixels along the longest edge, its cells, its
    measures (of the level set, at :data:`MEASURE_RESOLUTION` cells per edge; else of a
    voxel image) and a thumbnail. ``level_set``: whether the sections are those of
    implicit functions; ``implicit_phases``: how many phases have theirs."""
    from morphogen.viz import sections as vs

    cuts = vs.sections(micro, resolution)
    level_set = all(s.level_set is not None for s in cuts)  # one function: two phases
    out: dict = {
        "sections": [],
        "level_set": level_set or all(bool(s.coverages) for s in cuts),
        "implicit_phases": len(micro.implicit_phases()),
    }
    xy = None
    for s in cuts:
        rgb = vs.render(s, "phase")
        name = f"{s.names[0]}-{s.names[1]} section"
        out["sections"].append(
            {"title": name, "png": _png(rgb, folder, key, f"section_{''.join(s.names)}.png")}
        )
        if s.axis == 2:
            xy = rgb
    if any(s.labels is not None for s in cuts):
        out["labels"] = [
            {
                "title": f"{s.names[0]}-{s.names[1]} cells",
                "png": _png(vs.render(s, "labels"), folder, key, f"cells_{''.join(s.names)}.png"),
            }
            for s in cuts
            if s.labels is not None
        ]
    out["thumbnail"] = _thumbnail(xy if xy is not None else vs.render(cuts[-1], "phase"))
    n = min(resolution, MEASURE_RESOLUTION)
    if level_set:
        measures = vs.level_set_measures(micro, n)
        out["measures"] = _finite({f"{k} (level set)": v for k, v in measures.items()})
        out["fraction"] = float(measures.get("volume_fraction", np.nan))
    else:
        image = micro.voxelize(shape=n)
        measures = analysis.measure(image, ["phase_fractions", "specific_surface"])
        out["measures"] = _finite({f"{k} ({n} voxels)": v for k, v in measures.items()})
        out["fraction"] = float(measures.get("fraction_1", np.nan))
    return out


def _title(model, seed: int) -> str:
    return f"{model.name} (seed {seed})"


#: The stiffness of the frame of beams in the summaries of the network models
#: (``frame_stiffness: true``: :meth:`~morphogen.homogenization.networks.FrameStiffness.
#: summary`), labelled as the page shows it, and how they compute it
FRAME_SUMMARY = {
    "frame_young": "E / E_s",
    "frame_poisson": "Poisson's ratio",
    "frame_affinity": "E / E affine",
    "frame_stretching": "stretching energy",
    "frame_nonaffinity": "non-affinity",
    "frame_young_per_density": "E / (E_s φ)",
}
FRAME_NOTE = (
    "Euler–Bernoulli beams along the fibres (E_s = 1, ν = 0.3) under a uniaxial stress, the "
    "means over the axes; Timoshenko beams, rigid joints and the softening of non-identical "
    "beams in Mechanics, Stiffness of the frame of beams"
)


def frame_summary(summary: dict) -> dict:
    """The stiffness of the frame of beams in the summary of a realization, labelled
    (:data:`FRAME_SUMMARY`; empty without it)."""
    return {label: summary[k] for k, label in FRAME_SUMMARY.items() if k in summary}


def frame_matrices(summary: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """The stiffness and compliance of the frame of beams in the summary of a realization
    (its entries ``frame_C11``... and ``frame_S11``..., 6 x 6 or 3 x 3 in the plane), or
    None without them."""
    if "frame_C11" not in summary:
        return None
    n = 6 if "frame_C66" in summary else 3
    out = []
    for prefix in ("frame_C", "frame_S"):
        m = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                m[i, j] = m[j, i] = summary[f"{prefix}{i + 1}{j + 1}"]
        out.append(m)
    return out[0], out[1]


def generate(request: dict, session: Session | None = None) -> dict:
    """One realization (the realization 0 of a run with its seed): sections of its level
    set at ``resolution`` pixels per edge, summary, theory and measures; kept as a
    morphology in the workspace of the ``session``."""
    model = model_from_config({"type": request["model"], **_clean(request.get("params", {}))})
    spec = request.get("domain", {})
    domain = _domain(spec)
    seed = _seed(request.get("seed"))
    resolution = _resolution(request.get("resolution"))
    key = folder = None
    if session is not None:
        key, folder = session.folder()
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:  # shown with the results
        warnings.simplefilter("always")
        micro = model.generate(domain, derive_seed(seed, 0))
        views = morphology_views(micro, resolution, folder, key)
    elapsed = time.perf_counter() - t0
    out = {
        "model": model.to_config(),
        "seed": seed,
        "summary": micro.summary(),
        "theory": _finite(model.theory(domain)),
        "measures": views["measures"],
        "resolution": resolution,
        "level_set": views["level_set"],
        "implicit_phases": views["implicit_phases"],
        "sections": views["sections"],
        "time_s": round(elapsed, 3),
        "warnings": shown_warnings(caught),
    }
    if "labels" in views:
        out["labels"] = views["labels"]
    frame = frame_summary(out["summary"])
    if frame:
        out["frame"] = {"values": frame, "note": FRAME_NOTE}
        matrices = frame_matrices(out["summary"])
        if matrices is not None and folder is not None:  # its C and S, as the tools show them
            from morphogen.gui.tools import Context, elastic_tables

            ctx = Context(session, key, folder)
            units = ("units of the modulus of the fibres", "their inverse")
            if len(matrices[0]) == 3:
                units = ("per unit area: units of the modulus of the fibres times a length",
                         "their inverse")  # fmt: skip
            elastic_tables(ctx, matrices[0], "of the frame", names=("frame_stiffness",
                           "frame_compliance"), units=units, compliance=matrices[1])  # fmt: skip
            out["frame"]["tables"] = ctx.tables
    if session is not None:
        io.save_microstructure(micro, folder / "microstructure.json")
        files = {"geometry": f"/files/{key}/microstructure.json"}
        info = {"model": model.name, "seed": seed}
        if views["level_set"] and views["implicit_phases"] > 2:  # one function per phase
            info["implicit functions"] = f"of its {views['implicit_phases']} phases"
        else:
            info["level set"] = views["level_set"]
        if "E / E_s" in frame:  # the frame of beams of a network model
            info["frame E / E_s"] = frame["E / E_s"]
        item = session.add(
            key,
            "morphology",
            _title(model, seed),
            micro=micro,
            origin="generate",
            model=model.to_config(),
            domain={"size": spec.get("size", 10.0), "periodic": domain.periodic},
            seed=seed,
            phases={str(k): v for k, v in model.active_phases().items()},
            thumbnail=views["thumbnail"],
            fraction=views["fraction"],
            planar=_planar(micro),
            info=info,
            files=files,
        )
        out["item"] = session.public(item)
        out["files"] = files
    return out


def _planar(micro) -> bool:
    """Whether a morphology is a plane structure (one cell through the thickness)."""
    lengths = np.asarray(micro.domain.lengths, dtype=float)
    return bool(lengths[2] < 0.05 * lengths[:2].max())


def formation(request: dict) -> dict:
    """Middle sections of a structure along its formation (automata, reaction-diffusion,
    nucleation and growth, aggregation), from the initial state to the realization."""
    model = model_from_config({"type": request["model"], **_clean(request.get("params", {}))})
    if not hasattr(model, "evolution"):
        raise ValueError(f"model {model.name} has no formation history (automata and growth do)")
    frames = int(request.get("frames", 20))
    if not 2 <= frames <= MAX_FRAMES:
        raise ValueError(f"frames: from 2 to {MAX_FRAMES}")
    domain = _domain(request.get("domain", {}))
    seed = request.get("seed")
    seed = None if seed in (None, "") else derive_seed(int(seed), 0)
    states = model.evolution(domain, seed, frames)
    labels = states[-1].dtype == np.int32
    kind = "labels" if labels else ("phase" if states[-1].dtype == np.uint8 else "field")
    lo = min(float(s.min()) for s in states)
    hi = max(float(s.max()) for s in states)
    images = [png_url(slice_of(s, 2), kind, vmin=lo, vmax=hi) for s in states]
    changes = [float(np.mean(b != a)) for a, b in itertools.pairwise(states)]
    return {"frames": images, "kind": kind, "changes": changes}


evolve = formation  # the former name of the action


def count(request: dict) -> dict:
    """Exact number of compatible assemblies, and one assembly drawn by WFC."""
    from morphogen.design import (
        count_assemblies,
        count_polarized_blocks,
        polarized_blocks,
        wave_function_collapse,
    )
    from morphogen.design.tiles import TILE_SETS

    name = request.get("tiles", "polarized")
    if name not in TILE_SETS:
        raise ValueError(f"tiles: one of {list(TILE_SETS)}")
    shape = tuple(int(n) for n in request.get("shape", (3, 3, 3)))
    periodic = bool(request.get("periodic", False))
    tiles = polarized_blocks(8) if name == "polarized" else TILE_SETS[name](8)
    out: dict = {"tiles": name, "n_tiles": tiles.n_tiles, "shape": list(shape)}
    try:
        if name == "polarized":
            n = count_polarized_blocks(shape, periodic=periodic)
        else:
            n = count_assemblies(tiles, shape, periodic=periodic, max_states=5000)
        out["count"] = str(n)
        out["log2_count"] = float(np.log2(n)) if n > 0 else None
    except (ValueError, OverflowError) as exc:
        out["count_error"] = str(exc)
    try:
        rng = np.random.default_rng(request.get("seed"))
        assembly = wave_function_collapse(tiles, shape, rng, periodic=periodic)
        out["assembly"] = sections(tiles.assemble(assembly), "phase")
    except ValueError as exc:
        out["assembly_error"] = str(exc)
    return out


def _finite(values: dict) -> dict:
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and np.isfinite(v)}


# -- optimization ----------------------------------------------------------------------------


class OptimizationJob:
    """A topology optimization running in a thread, with its progress; it starts from a
    void ball, noise, or a morphology of the workspace (``start``: its key, or that of a
    simulation, whose morphology and solid material it takes), and keeps its design in the
    workspace as a morphology (its level set: the densities about 1/2), also when it is
    stopped (:meth:`stop`: the design of its last iteration)."""

    def __init__(self, request: dict, session: Session | None = None):
        from morphogen.homogenization.materials import Material
        from morphogen.optimization import CellOptimization, initial_density
        from morphogen.optimization.topology import INITIAL

        grid = request.get("grid", [48, 48, 1])
        grid = [int(grid)] * 3 if np.ndim(grid) == 0 else [int(n) for n in grid]
        if int(np.prod(grid)) > MAX_OPT_VOXELS:
            raise ValueError(f"grid: at most {MAX_OPT_VOXELS} voxels in the interface")
        settings = _clean(request.get("settings", {}))
        start = request.get("start") or settings.pop("initial", None) or "centered_void"
        self.source = None
        if start not in INITIAL:
            if session is None:
                raise ValueError("start: a morphology needs the workspace of a session")
            item = session.item(start)
            morphology = session.morphology_of(start)
            phase = int(request.get("solid_phase", 1))
            settings["initial"] = initial_density(morphology["micro"], grid, phase=phase)
            solid = (item.get("materials") or {}).get(str(phase))
            if isinstance(solid, dict) and "material" not in settings:
                settings["material"] = Material(
                    young=float(solid["young"]), poisson=float(solid["poisson"]), conductivity=1.0
                )
            self.source = (item, morphology)
        else:
            settings["initial"] = start
        volume = request.get("volume_fraction")
        if volume in (None, "") and self.source is not None:
            volume = float(np.mean(settings["initial"]))
        self.optimization = CellOptimization(tuple(grid), float(volume), **settings)
        self.seed = request.get("seed")
        self.session = session
        self.state = "running"
        self.progress: dict = {"iteration": 0, "objective": [], "volume": []}
        self.image = None
        self.result: dict | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> OptimizationJob:
        self.thread.start()
        return self

    def stop(self) -> None:
        """Stop at the end of the iteration under way, the design so far kept."""
        self._stop.set()

    def _callback(self, iteration, density, objective):
        with self._lock:
            self.progress["iteration"] = iteration
            self.progress["objective"].append(float(objective))
            self.progress["volume"].append(float(density.mean()))
            if iteration % 3 == 1:
                self.image = density.copy()
        return self._stop.is_set()  # True: the optimizer stops there

    def _domain(self, design) -> Domain | None:
        """The domain of the design: that of its morphology (its plane: one voxel thick)."""
        if self.source is None:
            return None
        domain = self.source[1]["micro"].domain
        lengths = np.asarray(domain.lengths, dtype=float)
        shape = design.density.shape
        if shape[2] == 1:
            lengths[2] = min(lengths[0] / shape[0], lengths[1] / shape[1])
        return Domain(size=tuple(float(v) for v in lengths), periodic=True)

    def _run(self):
        try:
            design = self.optimization.run(np.random.default_rng(self.seed), self._callback)
            result = {
                "iterations": design.iterations,
                "converged": design.converged,
                "stopped": self._stop.is_set() and not design.converged,
                "volume_fraction": design.volume_fraction,
                "binary_fraction": design.binary_fraction,
                "properties": _finite(design.properties()),
                "refined_properties": _finite(design.refined_properties(2)),
                "hinges": design.hinges,
                "time_s": round(design.time_s, 2),
            }
            if self.session is not None:
                key, folder = self.session.folder()
                micro = design.to_microstructure(self._domain(design), smoothing=DESIGN_SMOOTHING)
                io.save_microstructure(micro, folder / "design.json")
                files = {"geometry": f"/files/{key}/design.json"}
                tensor = design.binary_tensor
                if np.shape(tensor) == (6, 6) and np.all(np.isfinite(tensor)):  # C, S as .csv
                    from morphogen.gui.tools import Context, elastic_tables

                    ctx = Context(self.session, key, folder)
                    units = ("units of the material", "their inverse")
                    try:
                        elastic_tables(ctx, design.binary_tensor, "of the thresholded design",
                                       units=units)  # fmt: skip
                    except np.linalg.LinAlgError:  # a design that falls apart: no compliance
                        pass
                    result["tables"] = ctx.tables
                plane = slice_of(design.density, 2)
                origin = "" if self.source is None else f" from #{self.source[0]['number']}"
                objective = self.optimization.objective.replace("_", " ")
                item = self.session.add(
                    key,
                    "morphology",
                    f"design ({objective}){origin}",
                    micro=micro,
                    origin="optimize",
                    source=None if self.source is None else self.source[0]["key"],
                    model=None,
                    phases={"0": "void", "1": "solid"},
                    thumbnail=png_url(plane, "field", vmin=0.0, vmax=1.0, size=96),
                    fraction=design.binary_fraction,
                    planar=bool(design.planar),
                    info={"objective": self.optimization.objective, **result["properties"]},
                    files=files,
                )
                result["files"] = files
                result["item"] = self.session.public(item)
            with self._lock:
                self.image = design.density
                self.result = result
                self.state = "done"
        except Exception as exc:  # reported to the page
            with self._lock:
                self.error = str(exc)
                self.state = "error"

    def status(self, tiles: int = 3) -> dict:
        with self._lock:
            out = {
                "state": self.state,
                "iteration": self.progress["iteration"],
                "objective": list(self.progress["objective"]),
                "volume": list(self.progress["volume"]),
            }
            if self.image is not None:
                plane = slice_of(self.image, 2)
                if self.image.shape[2] == 1:
                    plane = np.tile(plane, (tiles, tiles))
                out["png"] = png_url(plane, "field", vmin=0.0, vmax=1.0, size=360)
            if self.result is not None:
                out["result"] = self.result
            if self.error is not None:
                out["error"] = self.error
        return out


# -- simulations ------------------------------------------------------------------------------

#: The fields and colormaps offered by the interface
SIM_FIELDS = (
    "von_mises",
    "energy_density",
    "displacement",
    *(f"{n}_{c}" for n in ("stress", "strain") for c in ("11", "22", "33", "23", "31", "12")),
)
CMAPS = (
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "turbo",
    "coolwarm",
    "RdBu_r",
    "jet",
    "gray",
)
VIDEO_KINDS = ("field", "orthoslices", "turntable", "sweep", "run", "none")
MAX_ELEMENTS_HINT = 200_000


def _number(request: dict, key: str, default, low: float, high: float) -> float:
    value = request.get(key)
    value = default if value in (None, "") else float(value)
    if not low <= value <= high:
        raise ValueError(f"{key}: between {low:g} and {high:g}")
    return value


def _video_format() -> str:
    from morphogen.viz.video import ffmpeg_path

    return "mp4" if ffmpeg_path() else "gif"


def _section(request: dict):
    axis = request.get("section_axis")
    if axis in (None, "", "none"):
        return None
    axis = "xyz".index(axis) if isinstance(axis, str) else int(axis)
    return {"axis": axis, "fraction": _number(request, "section_fraction", 0.5, 0.0, 1.0)}


def _materials(request: dict) -> dict:
    materials = {}
    for key, spec in (request.get("materials") or {}).items():
        if spec in ("void", None) or (isinstance(spec, dict) and spec.get("void")):
            materials[int(key)] = "void"
        else:
            materials[int(key)] = {"young": float(spec["young"]), "poisson": float(spec["poisson"])}
    return materials


def simulation_config(request: dict, folder: Path) -> dict:
    """The configuration of a simulation from the form of the interface: the model, its
    conforming mesh from its level set (``method: fe``) or its voxels (``fft``), the
    apparent ``property`` with its fields; the figures and the video chosen (``video``,
    "none" by default: the export draws them)."""
    model = {"type": request["model"], **_clean(request.get("params", {}))}
    dom = request.get("domain", {})
    size = dom.get("size", 10.0)
    periodic = bool(dom.get("periodic", True))
    method = request.get("method", "fe")
    if method not in ("fe", "fft"):
        raise ValueError("method: fe (conforming mesh) or fft (voxels)")
    prop = request.get("property", "shear_modulus")
    if prop not in ("bulk_modulus", "shear_modulus"):
        raise ValueError("property: bulk_modulus or shear_modulus")
    materials = _materials(request)
    field = request.get("field") or "von_mises"
    if field not in SIM_FIELDS:
        raise ValueError(f"field: one of {', '.join(SIM_FIELDS)}")
    cmap = request.get("cmap") or "viridis"
    if cmap not in CMAPS:
        raise ValueError(f"colormap: one of {', '.join(CMAPS)}")
    bc = "periodic" if periodic or method == "fft" else "kubc"
    result = f"{prop}.{bc}"
    magnification = request.get("magnification")
    magnification = None if magnification in (None, "", "auto") else float(magnification)
    fmt = _video_format()
    view = {"of": result, "color": field}
    if magnification is not None:
        view["magnification"] = magnification
    if method == "fe":
        if _section(request):
            view["section"] = _section(request)
        if request.get("part") == "fluctuation":
            view["part"] = "fluctuation"
    if field == "displacement" and method == "fft":
        raise ValueError("field: the displacement is on the nodes of finite element meshes")
    # finite elements on the conforming mesh of the level set need no voxels
    image = {"of": "image", "formats": ["png", "svg", "tex"]}
    figures = [{"of": "mesh"} if method == "fe" else image, dict(view)]
    kind = request.get("video") or "none"
    if kind not in VIDEO_KINDS:
        raise ValueError(f"video: one of {', '.join(VIDEO_KINDS)}")
    frames = int(_number(request, "frames", 36, 2, MAX_FRAMES * 4))
    options = {
        "formats": [fmt],
        "mode": request.get("mode") or "bounce",
        "frames": frames,
        "fps": _number(request, "fps", 18, 1, 60),
    }
    axis = "xyz".index(request.get("axis") or "z")
    videos = {
        "field": [{**view, "wave": request.get("wave") or "ramp", **options}],
        "orthoslices": [{"of": "image", "kind": "orthoslices", **options}],
        "turntable": [{"of": "mesh", "kind": "turntable", **options}]
        if method == "fe"
        else [{"of": "image", "kind": "surface", **options}],
        "sweep": [{"of": "mesh", "kind": "sweep", "axis": axis, **options}]
        if method == "fe"
        else [{"of": "image", "kind": "sections", "axis": axis, **options}],
        "none": [],
    }[kind]
    if kind == "sweep" and method == "fft":
        del videos[0]["frames"]  # one frame per section
    seed = request.get("seed")
    shape = _shape(request.get("shape"), MAX_FFT_SHAPE if method == "fft" else MAX_SHAPE) or 48
    cfg = {
        "model": model,
        "domain": {"size": size, "periodic": periodic},
        "seed": 1 if seed in (None, "") else int(seed),
        "voxels": {"shape": shape},
        "homogenization": {
            "materials": materials,
            "method": method,
            "boundary_conditions": [bc],
            "properties": [prop],
            "fields": True,
        },
        "outputs": {
            "directory": str(folder),
            "name": "sample",
            "geometry": ["json"],
            "image": [] if method == "fe" else ["vti"],
            "tables": ["txt", "xlsx"],
        },
        "analysis": [] if method == "fe" else ["volume_fraction"],
        "visualization": {"size": [640, 480], "cmap": cmap, "figures": figures},
    }
    if videos:
        cfg["visualization"]["videos"] = videos
    device = request.get("device") or "cpu"
    if device not in ("cpu", "gpu", "auto"):
        raise ValueError("device: cpu, gpu or auto")
    if method == "fft" and device != "cpu":  # the FFT solver on an NVIDIA GPU (CuPy)
        cfg["homogenization"]["device"] = device
    if method == "fe":
        edge = float(np.max(size)) if np.ndim(size) else float(size)
        cfg["mesh"] = {"size": _number(request, "element_size", edge / 12, edge / 200, edge)}
        order = request.get("order")
        if order not in (None, ""):  # 10-node elements by default (the configurations)
            if str(order) not in ("1", "2"):
                raise ValueError("order: 1 (4-node tetrahedra) or 2 (10-node tetrahedra)")
            cfg["mesh"]["order"] = int(order)
        cfg["outputs"]["conforming_mesh"] = ["vtu"]
    return cfg


def _files(key: str, folder: Path) -> list[dict]:
    """The files of a folder of the session for the page, by kind."""
    kinds = {
        ".json": "data",
        ".vti": "data",
        ".vtu": "data",
        ".stl": "data",
        ".yaml": "configuration",
        ".csv": "table",
        ".txt": "table",
        ".xlsx": "table",
        ".png": "figure",
        ".svg": "figure",
        ".tex": "figure",
        ".pdf": "figure",
        ".mp4": "video",
        ".gif": "video",
        ".zip": "archive",
        ".md": "report",
        # the inputs and files of other solvers and formats
        ".inp": "data",
        ".xml": "data",
        ".mfront": "data",
        ".mat": "data",  # Z-mat
        ".msh": "data",
        ".mesh": "data",
        ".xdmf": "data",
        ".h5": "data",
        ".vtk": "data",
        ".std": "data",
        ".bin": "data",
        ".tif": "data",
        ".tiff": "data",
        ".mhd": "data",
        ".raw": "data",
        ".npy": "data",
        ".npz": "data",
        ".tess": "data",
        ".sh": "configuration",
        ".bat": "configuration",
        ".rad": "data",  # OpenRadioss
        ".pvd": "data",  # a ParaView collection: the steps of a loading path
        ".obj": "data",  # the particles of a phase (DEM)
        ".data": "data",  # LAMMPS, LIGGGHTS
    }
    out = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix in kinds:
            out.append(
                {"name": path.name, "kind": kinds[path.suffix], "url": f"/files/{key}/{path.name}"}
            )
    return out


def _numbers_of(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f), {})
    out = {}
    for k, v in row.items():
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and k not in ("realization", "seed"):
            out[k] = x
    return out


class SimulationJob:
    """A simulation in a thread, with its state: of a morphology of the workspace
    (``source``: its key), or of a model and seed of the form (the whole chain of a
    configuration); its result is kept in the workspace, unless it is stopped
    (:meth:`stop`, at the next iteration of its solver)."""

    def __init__(self, request: dict, session: Session):
        from morphogen.config import load_config

        self.session = session
        self.source = None
        if request.get("source"):
            self.source = session.item(request["source"], ("morphology",))
            # a design is not the realization of a model: a stand-in for the configuration,
            # whose steps run on the morphology itself (and no configuration to rerun)
            model = self.source.get("model") or {"type": "tpms", "volume_fraction": 0.5}
            request = {
                **request,
                "model": model["type"],
                "params": {k: v for k, v in model.items() if k != "type"},
                "domain": self.source.get("domain")
                or {"size": list(self.source["micro"].domain.lengths), "periodic": True},
                "seed": self.source.get("seed") or 1,
            }
        self.request = request
        self.key, self.folder = session.folder()
        self.data = simulation_config(request, self.folder)
        self.config = load_config(self.data)  # errors of the form, before the thread
        self.started = time.perf_counter()
        self.state = "running"
        self.result: dict | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> SimulationJob:
        self.thread.start()
        return self

    def stop(self) -> None:
        """Stop at the next checkpoint of the solve (:mod:`morphogen.stopping`)."""
        self._stop.set()

    def _solve(self):
        """The steps of the configuration on the morphology of the workspace, else the
        whole run."""
        from morphogen import pipeline

        if self.source is None:
            pipeline.run(self.config)
            return
        cfg = self.config
        cfg.outputs.directory.mkdir(parents=True, exist_ok=True)
        micro = self.source["micro"]
        values, fe_rows = pipeline._process(micro, cfg, "sample_0000", self.folder, cfg.analysis)
        row = {"realization": 0, "seed": self.source.get("seed") or "", **micro.summary()}
        row.update(values)
        pipeline._write_results(self.folder, [row], fe_rows, 0, 1, cfg.outputs.tables)

    def _run(self):
        from morphogen.stopping import Stopped, stoppable

        try:
            with warnings.catch_warnings(record=True) as caught, stoppable(self._stop):
                warnings.simplefilter("always")
                self._solve()
            reproducible = self.source is None or self.source.get("model") is not None
            if reproducible:
                data = dict(self.data)
                data["outputs"] = {**data["outputs"], "directory": "simulation"}
                (self.folder / "configuration.yaml").write_text(
                    yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
                )
            stem = "sample_0000"
            figures = sorted(self.folder.glob(f"{stem}_*.png"))
            videos = sorted(
                [*self.folder.glob(f"{stem}_*.mp4"), *self.folder.glob(f"{stem}_*.gif")]
            )
            values = _numbers_of(self.folder / "results.csv")
            result = {
                "values": values,
                "figures": [
                    {
                        "title": p.stem.removeprefix(stem + "_").replace("_", " "),
                        "url": f"/files/{self.key}/{p.name}",
                    }
                    for p in figures
                    if "-" not in p.stem
                ],
                "videos": [
                    {
                        "title": p.stem.removeprefix(stem + "_").replace("_", " "),
                        "url": f"/files/{self.key}/{p.name}",
                        "format": p.suffix[1:],
                    }
                    for p in videos
                ],
                "files": _files(self.key, self.folder),
                "key": self.key,
                "time_s": round(time.perf_counter() - self.started, 2),
                "warnings": shown_warnings(caught),
            }
            self._keep(result, values)
            with self._lock:
                self.result, self.state = result, "done"
        except Stopped:  # nothing kept
            with self._lock:
                self.state = "stopped"
        except Exception as exc:  # reported to the page
            with self._lock:
                self.error, self.state = f"{type(exc).__name__}: {exc}", "error"

    def _keep(self, result: dict, values: dict) -> None:
        """The simulation in the workspace, with its apparent property."""
        homog = self.data["homogenization"]
        prop, method = homog["properties"][0], homog["method"]
        name = f"{prop}.{homog['boundary_conditions'][0]}"
        source = self.source
        of = "" if source is None else f" of #{source['number']}"
        by = "conforming mesh (FE)" if method == "fe" else "voxels (FFT)"
        info = {prop: values.get(name)}
        if "mesh_elements" in values:
            info["elements"] = int(values["mesh_elements"])
        materials = {str(k): v for k, v in homog["materials"].items()}
        item = self.session.add(
            self.key,
            "simulation",
            f"{prop.replace('_', ' ')}{of}, {'FE' if method == 'fe' else 'FFT'}",
            origin="simulate",
            source=None if source is None else source["key"],
            property=prop,
            method=method,
            values=values,
            materials=materials,
            config=self.data,
            phases=source.get("phases") if source is not None else None,
            thumbnail=source.get("thumbnail") if source is not None else None,
            info={"method": by, **_finite(info)},
            files={"folder": f"/files/{self.key}/"},
        )
        result["item"] = self.session.public(item)

    def status(self) -> dict:
        with self._lock:
            out = {"state": self.state, "elapsed_s": round(time.perf_counter() - self.started, 1)}
            if self.result is not None:
                out["result"] = self.result
            if self.error is not None:
                out["error"] = self.error
        return out


# -- views and exports -------------------------------------------------------------------------


def _clim(request: dict):
    if request.get("clim_low") not in (None, "") and request.get("clim_high") not in (None, ""):
        return (float(request["clim_low"]), float(request["clim_high"]))
    return None


def _field_files(sim: Path) -> tuple[list[Path], list[Path]]:
    """The files of the fields of a simulation: on its mesh (.vtu), on its voxels (.vti);
    for the run of a tool, which writes none of these, the last of its steps, else the
    last of the meshes of its fields (:func:`~morphogen.gui.tools.meshes_of_result`)."""
    sim = Path(sim)
    meshes = [p for p in sorted(sim.glob("sample_0000_*_*.vtu")) if not p.stem.endswith("_tet")]
    images = sorted(sim.glob("sample_0000_*_*.vti"))
    if meshes or images or not sim.is_dir():
        return meshes, images
    steps = path_steps(sim)
    if steps:
        last = sim / steps[-1]["file"]
        return ([last], []) if last.suffix == ".vtu" else ([], [last])
    from morphogen.gui.tools import meshes_of_result

    return meshes_of_result(sim)[-1:], []


_STEP = re.compile(r"_step(\d{3,})\.vt[iu]$")


def path_steps(folder: Path) -> list[dict]:
    """The fields of the steps of a loading path in a folder, in their order: ``{"file",
    "time", "label"}`` from the manifest that the tool writes (``<path>_steps.json``), or
    the files ``<path>_step<k>.vtu`` / ``.vti`` numbered in order for the runs written
    without one; empty when only the end was written."""
    import json

    folder = Path(folder)
    for manifest in sorted(folder.glob("*_steps.json")):
        try:
            steps = json.loads(manifest.read_text(encoding="utf-8"))["steps"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        found = [s for s in steps if (folder / str(s.get("file", ""))).is_file()]
        if found:
            return found
    if not folder.is_dir():
        return []
    numbered = sorted((int(m.group(1)), p.name) for p in folder.iterdir()
                      if (m := _STEP.search(p.name)))  # fmt: skip
    return [{"file": name, "time": None, "label": f"step {k}"} for k, name in numbered]


def write_pvd(path: Path, steps: list[dict]) -> Path:
    """A ParaView collection (``.pvd``) of the files of the steps of a loading path, at
    their times (their numbers for the steps without one): the series opens in one go."""
    from xml.sax.saxutils import quoteattr

    rows = []
    for k, s in enumerate(steps):
        t = s.get("time")
        rows.append(f'    <DataSet timestep="{k if t is None else float(t):.10g}" part="0" '
                    f"file={quoteattr(str(s['file']))}/>")  # fmt: skip
    path = Path(path)
    path.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n  <Collection>\n'
        + "\n".join(rows)
        + "\n  </Collection>\n</VTKFile>\n",
        encoding="utf-8",
    )
    return path


def _step_file(folder: Path, steps: list[dict], step) -> Path | None:
    """The file of the step ``step`` (an index of ``steps``, negative from the end; None:
    none chosen, the end of the path)."""
    if step in (None, "") or not steps:
        return None
    k = int(step)
    if not -len(steps) <= k < len(steps):
        raise ValueError(f"step: from 0 to {len(steps) - 1}")
    return Path(folder) / steps[k]["file"]


def field_names(sim: Path) -> list[str]:
    """The fields of a simulation that can be drawn (those of its field file: stresses,
    strains, von Mises, energy density, and the internal variables of nonlinear runs, e.g.
    the cumulated plastic strain or the porosity), the norm of the displacement for a mesh."""
    meshes, images = _field_files(sim)
    if meshes:
        from morphogen.io.vtk import read_vtu

        data = read_vtu(meshes[0])
        names = [*data["point_data"], *data["cell_data"]]
        return [n for n in dict.fromkeys(names) if n not in ("phase", "grain", "fluctuation")]
    if images:
        from morphogen.io.vtk import read_vti

        return [img.name for img in read_vti(images[0]) if img.name not in ("phase", "grain")]
    return []


def simulation_fields(request: dict, session: Session) -> dict:
    """The fields of a simulation or of the run of a tool of the workspace (``key``), the
    usual ones first."""
    item = session.item(request.get("key"), ("simulation", "result"))
    names = field_names(session.root / item["key"])
    usual = [n for n in SIM_FIELDS if n in names]
    return {"fields": usual + [n for n in names if n not in usual]}


def _step_frames(sim: Path, steps: list[dict], field: str, request: dict, **view) -> list:
    """The frames of a field along the steps of a loading path (:func:`path_steps`), one
    per step on one colour range: the mesh deformed by the displacement of each step, or
    the middle section of the voxels along ``axis``."""
    from morphogen.viz import video as vv

    files = [sim / s["file"] for s in steps]
    captions = [s.get("label") or f"step {k + 1}" for k, s in enumerate(steps)]
    if files[0].suffix == ".vtu":
        from morphogen.io.vtk import read_vtu

        meshes, us, values = [], [], []
        for path in files:
            data = read_vtu(path)
            arrays = {**data["point_data"], **data["cell_data"]}
            u = np.asarray(arrays["displacement"], dtype=float)
            meshes.append(io.read_tet_mesh(path))
            us.append(u)
            values.append(np.linalg.norm(u, axis=1) if field == "displacement" else arrays[field])
        same = all(m.nodes.shape == meshes[0].nodes.shape
                   and m.elements.shape == meshes[0].elements.shape for m in meshes)  # fmt: skip
        if same:
            return vv.history_frames(meshes[0], us, values, captions=captions, title=field, **view)
        # meshes that change from step to step (elements deleted, cracks): each frame its
        # own mesh, on the colour range of all
        if view.get("clim") is None:
            finite = [np.asarray(v, float)[np.isfinite(v)] for v in values]
            lows = [float(v.min()) for v in finite if v.size]
            view = {**view, "clim": (min(lows), max(float(v.max()) for v in finite if v.size))
                    if lows else None}  # fmt: skip
        return [vv.deformed_view(m, u, v, title=f"{field}: {c}", **view)
                for m, u, v, c in zip(meshes, us, values, captions, strict=True)]  # fmt: skip
    images = [io.load_image(path, field) for path in files]
    lo, hi = view.get("clim") or (None, None)
    axis = "xyz".index(request.get("axis") or "z")
    return vv.time_section_frames(images, axis, size=480, cmap=view.get("cmap", "viridis"),
                                  vmin=lo, vmax=hi, captions=captions, label=field)  # fmt: skip


def _field_view(sim: Path, out: Path, key: str, name: str, request: dict) -> dict:
    """A view of the fields of the simulation of folder ``sim`` (field, colormap, range,
    magnification, cut) written to ``out`` as ``name``.png, and its video (``video``:
    "field" on the deformed mesh under a growing or oscillating load — along the steps of a
    loading path whose steps were written, one frame per step —, "turntable", "sweep" of a
    cross-section; the sections of an FFT image)."""
    from PIL import Image

    from morphogen.viz import video as vv
    from morphogen.viz.export import save_figure
    from morphogen.viz.figures import field_figure

    field = request.get("field") or "von_mises"
    available = field_names(sim)
    if field == "displacement" and not _field_files(sim)[0]:
        raise ValueError("field: the displacement is on the nodes of finite element meshes")
    if field not in available:
        raise ValueError(f"field: one of {', '.join(available) or 'none here'}")
    cmap = request.get("cmap") or "viridis"
    if cmap not in CMAPS:
        raise ValueError(f"colormap: one of {', '.join(CMAPS)}")
    clim = _clim(request)
    kind = request.get("video") or "none"
    if kind not in VIDEO_KINDS:
        raise ValueError(f"video: one of {', '.join(VIDEO_KINDS)}")
    frames_n = int(_number(request, "frames", 36, 2, MAX_FRAMES * 4))
    axis = "xyz".index(request.get("axis") or "z")
    result: dict = {}
    meshes, images = _field_files(sim)
    if meshes:
        from morphogen.io.vtk import read_vtu

        mesh = io.read_tet_mesh(meshes[0])
        data = read_vtu(meshes[0])
        arrays = {**data["point_data"], **data["cell_data"]}
        u = arrays.get("displacement")
        chosen = np.asarray(arrays[field]) if field in arrays else None
        mode = u is None and chosen is not None and chosen.ndim == 2 and chosen.shape[1] == 3
        if mode:  # a mode (the pattern of a bifurcation): it moves the mesh
            u = chosen.astype(float)
        values = np.linalg.norm(u, axis=1) if field == "displacement" or mode else chosen
        magnification = request.get("magnification")
        magnification = None if magnification in (None, "", "auto") else float(magnification)
        section = _section(request)
        cut = None if section is None else (section["axis"], section["fraction"])
        rgb = vv.deformed_view(
            mesh, u, values, magnification=magnification, section=cut, cmap=cmap, clim=clim,
            title=field,
        )  # fmt: skip
        Image.fromarray(rgb).save(out / f"{name}.png")
        frames = None
        steps = path_steps(sim)
        if kind == "field" and len(steps) > 1:  # the path itself, step after step
            frames = _step_frames(sim, steps, field, request, magnification=magnification,
                                  section=cut, cmap=cmap, clim=clim)  # fmt: skip
        elif kind == "field":
            frames = vv.field_frames(
                mesh, u, values, magnification=magnification, section=cut, cmap=cmap,
                clim=clim, title=field, wave=request.get("wave") or "ramp", frames=frames_n,
                scaling=vv.field_scaling(field),
            )  # fmt: skip
        elif kind == "turntable":
            frames = vv.turntable_frames(mesh, field=values, frames=frames_n)
        elif kind == "sweep":
            frames = vv.section_sweep_frames(
                mesh, values, axis=axis, frames=frames_n, cmap=cmap, clim=clim, title=field
            )
        elif kind == "orthoslices":
            raise ValueError("video: orthoslices are views of voxel images (FFT simulations)")
        if frames is not None:
            result["video"] = _video(frames, out, key, name, request, f"{field} ({kind})")
    else:
        if not images:
            raise ValueError("no fields in this simulation")
        if field == "displacement":
            raise ValueError("field: the displacement is on the nodes of finite element meshes")
        values = io.load_image(images[0], field)
        formats = [f for f in request.get("formats") or ["png"] if f in ("png", "svg", "tex")]
        for fmt in formats or ["png"]:
            save_figure(
                field_figure(values, cmap=cmap, clim=clim, label=field), out / f"{name}.{fmt}",
                dpi=120,
            )  # fmt: skip
        if kind != "none":
            lo, hi = clim or (None, None)
            steps = path_steps(sim)
            if kind == "field" and len(steps) > 1:  # the path itself, step after step
                frames = _step_frames(sim, steps, field, request, cmap=cmap, clim=clim)
            elif kind in ("field", "sweep"):
                frames = vv.section_frames(
                    values, axis, size=480, cmap=cmap, vmin=lo, vmax=hi, labels=False,
                    bar=True, label=field,
                )  # fmt: skip
            elif kind == "orthoslices":
                frames = vv.orthoslice_frames(values, frames=frames_n, size=320, cmap=cmap)
            else:
                frames = vv.turntable_frames(values, frames=frames_n)
            result["video"] = _video(frames, out, key, name, request, f"{field} ({kind})")
    result["figure"] = {"title": field, "url": f"/files/{key}/{name}.png"}
    return result


def render(request: dict, session: Session) -> dict:
    """Another view, and optionally a video, of the fields of a simulation (``key``): a
    field, colormap, range, magnification, cut, without solving again."""
    key = str(request["key"])
    folder = session.file(key, "results.csv").parent
    n = len(list(folder.glob("render_*.png"))) + 1
    out = _field_view(folder, folder, key, f"render_{n}", request)
    out["files"] = _files(key, folder)
    return out


#: The 3D view: cells along the longest edge of the grids of the surfaces of morphologies
SURFACE_RESOLUTION = 96
MAX_SURFACE_RESOLUTION = 256


def _surface_resolution(value) -> int:
    n = SURFACE_RESOLUTION if value in (None, "", 0, "0") else int(value)
    if not 8 <= n <= MAX_SURFACE_RESOLUTION:
        raise ValueError(f"resolution: from 8 to {MAX_SURFACE_RESOLUTION} cells per edge")
    return n


def _budget(value) -> int:
    from morphogen.viz.surfaces import MAX_TRIANGLES

    n = MAX_TRIANGLES if value in (None, "", 0, "0") else int(value)
    if not 1000 <= n <= MAX_TRIANGLES:
        raise ValueError(f"budget: from 1000 to {MAX_TRIANGLES} triangles")
    return n


def surface(request: dict, session: Session) -> dict:
    """The surface of an item of the workspace (``key``) for the 3D view of the page
    (:mod:`morphogen.viz.surfaces`, as :func:`~morphogen.viz.surfaces.encode` writes it):
    of a morphology, the interfaces of its phases (or cells) on a grid of ``resolution``
    cells along its longest edge (:data:`SURFACE_RESOLUTION` by default, at most
    :data:`MAX_SURFACE_RESOLUTION`); of a simulation, the faces of the mesh or of the
    voxels of its fields, with ``field`` (by default the von Mises stress) and the
    displacement of the mesh, its sections on a grid of ``resolution`` cells too; of a
    loading path whose steps were written, those of ``step`` (an index of ``steps``, the
    list it returns with their times and labels; by default the end of the path). At most
    ``budget`` triangles (300 000 and below): the grids are made coarser, the meshes
    simplified, to fit."""
    from morphogen.viz import surfaces

    item = session.item(request.get("key"), ("morphology", "simulation", "result"))
    resolution = _surface_resolution(request.get("resolution"))
    budget = _budget(request.get("budget"))
    started = time.perf_counter()
    steps, step = [], None
    with warnings.catch_warnings():  # those of the libraries, e.g. while Numba compiles
        warnings.simplefilter("ignore")
        if item["kind"] == "morphology":
            micro = item.get("micro")
            if micro is None:
                raise ValueError(f"{item['title']}: no microstructure to draw in this session")
            _voxels(micro.domain, resolution + 2, 24.0, "3D view")
            shown = surfaces.morphology_surface(micro, resolution, max_triangles=budget)
            phases = item.get("phases") or {str(k): v for k, v in micro.phase_names.items()}
        else:
            folder = session.root / item["key"]
            if item["kind"] == "result":  # the run of a tool: the mesh of its last step
                from morphogen.gui.tools import meshes_of_result

                meshes, images = meshes_of_result(folder)[::-1], []
                if not meshes and folder.is_dir():  # the fields of its steps (voxels)
                    meshes, images = _field_files(folder)
            else:
                meshes, images = _field_files(folder) if folder.is_dir() else ([], [])
            steps = path_steps(folder) if folder.is_dir() else []
            chosen = _step_file(folder, steps, request.get("step"))
            if chosen is not None:  # a step of the path, in place of its end
                step = int(request["step"]) % len(steps)
                meshes, images = ([chosen], []) if chosen.suffix == ".vtu" else ([], [chosen])
            elif steps:
                step = len(steps) - 1
            field = request.get("field") or None
            if meshes:
                shown = surfaces.vtu_surface(
                    meshes[0], field, resolution=resolution, max_triangles=budget
                )
            elif images:
                source = session.items.get(str(item.get("source")))
                shown = surfaces.vti_surface(
                    images[0],
                    field,
                    phases=_image_phases(folder),
                    morphology=None if source is None else source.get("micro"),
                    max_triangles=budget,
                )
            else:
                raise ValueError(
                    f"{item['title']}: nothing to draw in 3D (no mesh or voxels of fields)"
                )
            phases = item.get("phases") or {}
    out = surfaces.encode(shown)
    if out["field"] is not None:  # the usual fields first, as in the views of the export
        names = out["field"]["names"]
        usual = [n for n in SIM_FIELDS if n in names]
        out["field"]["names"] = usual + [n for n in names if n not in usual]
    out.update(
        key=item["key"],
        kind=item["kind"],
        title=item["title"],
        phases={str(k): v for k, v in phases.items()},
        requested=resolution,
        budget=budget,
        time_s=round(time.perf_counter() - started, 3),
    )
    if len(steps) > 1:  # the steps of a loading path, which the view goes through
        out["steps"] = [{"label": s.get("label"), "time": s.get("time")} for s in steps]
        out["step"] = step
    return out


def _image_phases(folder: Path):
    """The phases of the voxel image of a simulation on voxels (``sample_0000.vti``), if
    it wrote one."""
    from morphogen.io.vtk import read_vti

    path = folder / "sample_0000.vti"
    if not path.is_file():
        return None
    images = read_vti(path, names=("phase",))
    return images[0].data if images else None


def _video(frames, folder: Path, key: str, name: str, request: dict, title: str) -> dict:
    from morphogen.viz.video import write_video

    fmt = _video_format()
    path = folder / f"{name}.{fmt}"
    write_video(
        frames, path, fps=_number(request, "fps", 18, 1, 60), mode=request.get("mode") or "bounce"
    )
    return {"title": title, "url": f"/files/{key}/{path.name}", "format": fmt}


#: What the export writes: of a morphology, of a simulation
MORPHOLOGY_EXPORTS = (
    "geometry", "sections", "level_set", "image", "surface", "mesh", "particles", "blender",
)  # fmt: skip
#: The fluids of the scene for Blender (Mantaflow)
BLENDER_FLUIDS = ("gas", "liquid", "none")
#: The files of the particles of a phase (DEM) by the choice of the form
PARTICLE_FILES = {
    "stl": "particles.stl",
    "stl_each": "particles.stl",
    "obj": "particles.obj",
    "vtu": "particles.vtu",
    "msh": "particles.msh",
    "msh_tetrahedra": "particles_tetrahedra.msh",
    "csv": "particles.csv",
    "data": "particles.data",
    "yade": "particles_yade.txt",
    "blender": "blender_scene.zip",
}
SIMULATION_EXPORTS = ("figure", "tables", "fields", "mesh", "configuration")
MORPHOLOGY_VIDEOS = ("none", "sweep", "orthoslices", "turntable", "formation")


def export(request: dict, session: Session) -> dict:
    """The files of an item of the workspace (``source``), in a folder of their own with
    an archive of them all: of a morphology, its geometry, sections of its level set,
    voxel images of its level set and phases, its surface (STL) and conforming mesh (for
    a morphology of several phases: the implicit function and the surface of each phase,
    ``level_set_phase<k>.vti`` and ``surface_phase<k>.stl``, and the mesh conforming to
    all their interfaces), the particles of one phase without its matrix for the
    discrete element method (:func:`_particles_export`), and a video of its sections
    sweeping it, orthoslices or a turntable; of a simulation or of the run of a tool, a
    view of its fields (figure and video, or the videos of the frames its tool kept), its
    tables, fields (and those of its steps), mesh and configuration."""
    item = session.item(request.get("source"), ("morphology", "simulation", "result"))
    key, folder = session.folder()
    if item["kind"] == "morphology":
        out = _export_morphology(item, request, key, folder)
    else:
        out = _export_simulation(item, request, key, folder, session)
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "export.zip"]
    if files:  # the steps of a run in their subfolders too (the animations of a dynamic run)
        with zipfile.ZipFile(folder / "export.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(folder).as_posix())
        out["archive"] = f"/files/{key}/export.zip"
    out["files"] = _files(key, folder)
    out["source"] = session.public(item)
    return out


def _chosen(request: dict, allowed) -> list[str]:
    chosen = list(request.get("outputs") or [])
    unknown = sorted(set(chosen) - set(allowed))
    if unknown:
        raise ValueError(f"outputs: among {', '.join(allowed)} (not {', '.join(unknown)})")
    return chosen


def _export_morphology(item: dict, request: dict, key: str, folder: Path) -> dict:
    from morphogen.viz import sections as vs

    micro = item["micro"]
    chosen = _chosen(request, MORPHOLOGY_EXPORTS)
    resolution = _resolution(request.get("resolution"))
    shape = _shape(request.get("shape")) or 128
    out: dict = {"figures": [], "videos": []}
    if "geometry" in chosen:
        io.save_microstructure(micro, folder / "microstructure.json")
    if "sections" in chosen:
        for s in vs.sections(micro, resolution):
            name = f"section_{''.join(s.names)}.png"
            (folder / name).write_bytes(encode_png(vs.render(s, "phase")))
            title = f"{s.names[0]}-{s.names[1]} section"
            out["figures"].append({"title": title, "url": f"/files/{key}/{name}"})
    if "level_set" in chosen or "surface" in chosen:
        from morphogen.mesh.surface import surface_mesh
        from morphogen.models.implicit import implicit_functions

        _voxels(micro.domain, shape, 12.0, "level set")
        surfaces = "surface" in chosen
        # one function for two phases; else that of each phase (its surface: but the last,
        # whose surface is that of the others together)
        functions = implicit_functions(micro, shape=shape)
        for k, (suffix, level) in enumerate(functions):
            if "level_set" in chosen:
                io.save_image(level, folder / f"level_set{suffix}.vti")
            if surfaces and (k < len(functions) - 1 or len(functions) == 1):
                io.save_mesh(surface_mesh(level, closed=True), folder / f"surface{suffix}.stl")
    if "image" in chosen:
        _voxels(micro.domain, shape, 6.0, "voxel image")
        io.save_image(micro.voxelize(shape=shape), folder / "image.vti")
    if "mesh" in chosen:
        from morphogen.mesh import conforming_mesh

        edge = float(np.max(micro.domain.lengths))
        size = _number(request, "element_size", edge / 12, edge / 200, edge)
        io.save_mesh(conforming_mesh(micro, size=size), folder / "mesh.vtu")
    if "particles" in chosen:  # one phase without its matrix, for the discrete elements
        _particles_export(micro, request, shape, folder)
    if "blender" in chosen:  # the phases as the objects of Blender, in a fluid domain
        _blender_export(micro, request, shape, folder)
    kind = request.get("video") or "none"
    if kind not in MORPHOLOGY_VIDEOS:
        raise ValueError(f"video: one of {', '.join(MORPHOLOGY_VIDEOS)}")
    if kind != "none":
        from morphogen.viz import video as vv

        frames_n = int(_number(request, "frames", 36, 2, MAX_FRAMES * 4))
        axis = "xyz".index(request.get("axis") or "z")
        if kind == "sweep":  # sections of the level set sweeping the domain
            n = min(resolution, 720)
            fractions = (np.arange(frames_n) + 0.5) / frames_n
            frames = [vs.render(vs.section(micro, axis, float(f), n), "phase") for f in fractions]
        elif kind == "formation":  # the middle section along the formation of the structure
            model = model_from_config(item["model"]) if item.get("model") else None
            if model is None or not hasattr(model, "evolution"):
                raise ValueError(
                    "video: the formation of models grown by a process (automata, growth)"
                )
            seed = item.get("seed")
            states = model.evolution(
                micro.domain,
                None if seed is None else derive_seed(int(seed), 0),
                min(frames_n, MAX_FRAMES),
            )
            kind_of = (
                "labels"
                if states[-1].dtype == np.int32
                else ("phase" if states[-1].dtype == np.uint8 else "field")
            )
            lo = min(float(s.min()) for s in states)
            hi = max(float(s.max()) for s in states)
            frames = []
            for s in states:
                rgb = to_rgb(slice_of(s, 2), kind_of, lo, hi)
                frames.append(upscale(rgb, max(1, 480 // max(rgb.shape[:2]))))
        else:
            image = micro.voxelize(shape=min(shape, 160))
            if kind == "orthoslices":
                frames = vv.orthoslice_frames(image, frames=frames_n, size=320)
            else:
                frames = vv.turntable_frames(image, frames=frames_n)
        out["videos"].append(_video(frames, folder, key, f"video_{kind}", request, kind))
    return out


def _blender_export(micro, request: dict, shape: int, folder: Path) -> Path:
    """The scene of a morphology for Blender (``blender_fluid``: gas, liquid or none, along
    ``blender_axis``), archived: the surfaces of its phases from its conforming mesh (of
    ``element_size``), the fluid of Mantaflow around them and the script that builds it."""
    from morphogen.interop.blender import write_blender_scene

    fluid = request.get("blender_fluid") or "gas"
    if fluid not in BLENDER_FLUIDS:
        raise ValueError(f"fluid of the scene for Blender: one of {', '.join(BLENDER_FLUIDS)}")
    axis = str(request.get("blender_axis") or "x")
    if axis not in ("x", "y", "z"):
        raise ValueError("axis of the flow of the scene for Blender: x, y or z")
    edge = float(np.max(micro.domain.lengths))
    size = _number(request, "element_size", edge / 12, edge / 200, edge)
    scene = write_blender_scene(folder / "blender_scene", micro, size=size, shape=shape,
                                fluid=None if fluid == "none" else fluid,
                                axis="xyz".index(axis))  # fmt: skip
    archive = folder / "blender_scene.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for path in sorted(scene.folder.iterdir()):
            out.write(path, f"blender_scene/{path.name}")
    shutil.rmtree(scene.folder)
    return archive


def _particles_export(micro, request: dict, shape: int, folder: Path) -> Path:
    """The particles of a phase of a morphology (``particles_phase``) in the file of the
    format chosen (``particles_format``, :data:`PARTICLE_FILES`), their surfaces remeshed
    at ``particles_size`` if given; one STL per particle goes in an archive of its own."""
    from morphogen.io.dem import write_particles

    phases = [int(p) for p in micro.implicit_phases()]
    phase = int(request.get("particles_phase") or (1 if 1 in phases else phases[0]))
    if phase not in phases:
        raise ValueError(f"phase of the particles: one of {', '.join(map(str, phases))}")
    fmt = request.get("particles_format") or "stl"
    if fmt not in PARTICLE_FILES:
        raise ValueError(f"format of the particles: one of {', '.join(PARTICLE_FILES)}")
    _voxels(micro.domain, shape, 12.0, "particles")
    given = str(request.get("particles_size") or "").strip()
    try:
        size = float(given) if given else None
    except ValueError:
        raise ValueError(f"triangle size of the particles: a length, got {given!r}") from None
    if size is not None and not size > 0:
        raise ValueError("triangle size of the particles: a positive length")
    if fmt == "blender":  # the rigid bodies of Blender: its scene and its script, archived
        from morphogen.interop.blender import write_blender

        scene = write_blender(folder / "blender", micro, phase, shape=shape)
        archive = folder / PARTICLE_FILES[fmt]
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
            for path in sorted(scene.folder.iterdir()):
                out.write(path, f"blender/{path.name}")
        shutil.rmtree(scene.folder)
        return archive
    if fmt != "stl_each":
        elements = "tetrahedra" if fmt == "msh_tetrahedra" else "triangles"
        return write_particles(folder / PARTICLE_FILES[fmt], micro, phase, shape=shape,
                               format="yade" if fmt == "yade" else None, size=size,
                               elements=elements)  # fmt: skip
    each = folder / "particles"
    files = write_particles(folder / "particles.stl", micro, phase, shape=shape, separate=True,
                            size=size)  # fmt: skip
    archive = folder / "particles_stl.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for path in files:
            out.write(path, path.name)
    shutil.rmtree(each)
    return archive


def _run_videos(sim: Path, folder: Path, key: str, request: dict) -> list[dict]:
    """The videos of a run from the frames that its tool kept without writing them
    (``<name>.frames.npz``, :meth:`~morphogen.gui.tools.ToolContext.video`), at their
    speed and mode unless the request gives them; for a run whose tool kept none, the
    video of a field along its steps (kind "field")."""
    kept = sorted(Path(sim).glob("*.frames.npz"))
    if not kept:
        names = field_names(sim) if len(path_steps(sim)) > 1 else []
        if names:
            field = request.get("field") if request.get("field") in names else names[0]
            view = {**request, "video": "field", "field": field}
            return [_field_view(sim, folder, key, "run", view)["video"]]
        raise ValueError("video: this run kept no frames (its videos were written, or none)")
    out = []
    for path in kept:
        with np.load(path) as data:
            frames = list(data["frames"])
            fps, mode, title = float(data["fps"]), str(data["mode"]), str(data["title"])
        name = path.name[: -len(".frames.npz")]
        given = {"fps": request.get("fps") or fps, "mode": request.get("mode") or mode}
        out.append(_video(frames, folder, key, name, given, title))
    return out


def _export_simulation(item: dict, request: dict, key: str, folder: Path, session) -> dict:
    from morphogen.io.tables import write_table

    sim = session.root / item["key"]  # its folder: of a simulation, or of the run of a tool
    chosen = _chosen(request, SIMULATION_EXPORTS)
    out: dict = {"figures": [], "videos": []}
    kind = request.get("video") or "none"
    if kind == "run":  # the videos of the run from the frames its tool kept
        out["videos"] += _run_videos(sim, folder, key, request)
        request = {**request, "video": "none"}
    if "figure" in chosen or kind not in ("none", "run"):
        view = _field_view(sim, folder, key, "view", request)
        if "figure" in chosen:
            out["figures"].append(view["figure"])
        else:
            for p in folder.glob("view.*"):
                if p.suffix in (".png", ".svg", ".tex"):
                    p.unlink()
        if "video" in view:
            out["videos"].append(view["video"])
    if "tables" in chosen:  # those of a simulation, or every table of the run of a tool
        stems = ["results", "homogenization"] if item["kind"] == "simulation" else sorted(
            p.stem for p in sim.glob("*.csv"))  # fmt: skip
        for stem in stems:
            path = sim / f"{stem}.csv"
            if path.exists():
                shutil.copy2(path, folder / path.name)
                with path.open(encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                for fmt in request.get("table_formats") or ["txt", "xlsx"]:
                    if fmt in ("txt", "xlsx"):
                        write_table(folder / f"{stem}.{fmt}", rows)
    if "fields" in chosen:
        meshes, images = _field_files(sim)
        for path in [*meshes, *images]:
            shutil.copy2(path, folder / path.name)
        steps = path_steps(sim)
        if steps:  # those of the steps of a path or of a run, a time series for ParaView
            for s in steps:
                target = folder / s["file"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sim / s["file"], target)
            write_pvd(folder / "steps.pvd", steps)
    if "mesh" in chosen:  # tetrahedra, or the triangles of a structure of the plane
        for path in [*sim.glob("sample_0000_tet.*"), *sim.glob("sample_0000_tri.*")]:
            shutil.copy2(path, folder / path.name)
    if "configuration" in chosen and (sim / "configuration.yaml").exists():
        shutil.copy2(sim / "configuration.yaml", folder / "configuration.yaml")
    return out


# -- saving and opening a workspace -------------------------------------------------------------

#: The manifest of a saved workspace
WORKSPACE_FILE = "workspace.json"


def _default_directory() -> Path:
    return Path.home() / "morphogen" / "workspaces" / time.strftime("%Y%m%d-%H%M%S")


def save_workspace(request: dict, session: Session) -> dict:
    """Write the workspace of the session to a folder (``directory``, by default a new one
    under ``~/morphogen/workspaces``): the folder of each item with its files, the
    microstructure of each morphology (``micro.json``) and the manifest of the items, so
    that :func:`open_workspace` restores them in another session."""
    import json

    from morphogen._utils import to_jsonable

    directory = Path(str(request.get("directory") or "").strip() or _default_directory())
    directory = directory.expanduser()
    items = sorted(session.items.values(), key=lambda i: i["number"])
    if not items:
        raise ValueError("the workspace is empty: nothing to save")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in items:
        source = session.root / item["key"]
        target = directory / item["key"]
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        if item.get("micro") is not None:
            io.save_microstructure(item["micro"], target / "micro.json")
        manifest.append(to_jsonable({k: v for k, v in item.items() if k != "micro"}))
    (directory / WORKSPACE_FILE).write_text(
        json.dumps({"morphogen": _version(), "items": manifest}, indent=1), encoding="utf-8"
    )
    return {"saved": len(manifest), "directory": str(directory)}


def _version() -> str:
    from morphogen import __version__

    return __version__


def open_workspace(request: dict, session: Session) -> dict:
    """Add the items of a saved workspace (``directory``, see :func:`save_workspace``) to
    the workspace of the session, under new keys (their links and sources renamed)."""
    import json
    import re

    directory = Path(str(request.get("directory") or "").strip()).expanduser()
    manifest = directory / WORKSPACE_FILE
    if not manifest.is_file():
        raise ValueError(f"no saved workspace in {directory} (no {WORKSPACE_FILE})")
    items = json.loads(manifest.read_text(encoding="utf-8")).get("items", [])
    keys: dict[str, str] = {}
    opened = []
    for item in items:  # in their order: the sources come first
        old = item["key"]
        new, folder = session.folder()
        keys[old] = new
        saved = directory / old
        if saved.is_dir():
            shutil.copytree(saved, folder, dirs_exist_ok=True)
        text = json.dumps(item)
        for before, after in keys.items():  # its files, sources and the items it took
            text = text.replace(before, after)
        data = json.loads(text)
        for k in ("key", "kind", "number", "title"):
            data.pop(k, None)
        if (folder / "micro.json").is_file():
            data["micro"] = io.load_microstructure(folder / "micro.json")
        title = re.sub(r"^#\d+ ", "", item.get("title", item["kind"]))
        opened.append(session.public(session.add(new, item["kind"], title, **data)))
    return {"opened": len(opened), "items": opened}


def tool_result(key: str, session: Session) -> dict:
    """The run of a tool kept in the workspace (a result, or the simulation or behaviour a
    tool made): its tool and tab, the parameters of its form and its result, to show them
    again."""
    item = session.item(key, ("result", "simulation", "behaviour"))
    if "tool" not in item:
        raise ValueError(f"{item['title']} is not the run of a tool")
    return {k: item[k] for k in ("key", "title", "tool", "tab", "params", "result")}


# -- database of results and Ashby charts ----------------------------------------------------


def _database(request: dict):
    from morphogen.results import ResultsDatabase

    path = str(request.get("database") or "").strip()
    if not path:
        raise ValueError("database: the path of a results database (.db)")
    if request.get("create") is not True and not Path(path).exists():
        raise ValueError(f"no database {path} (add a run to create it)")
    return ResultsDatabase(path)


def _conditions(request: dict) -> list[str]:
    where = request.get("where") or []
    if isinstance(where, str):
        where = [w for part in where.splitlines() for w in part.split(";")]
    return [w.strip() for w in where if w.strip()]


def database_info(request: dict) -> dict:
    """Runs of a database and the keys of its records (those matching ``where``)."""
    with _database(request) as db:
        conditions = _conditions(request)
        keys = db.keys(conditions)
        return {
            "runs": db.runs(),
            "records": max((v["count"] for v in keys.values()), default=0),
            "numeric": [k for k, v in keys.items() if v["numeric"]],
            "text": [k for k, v in keys.items() if not v["numeric"]],
        }


def database_add(request: dict) -> dict:
    """Add a run directory to a database (created if needed)."""
    directory = str(request.get("directory") or "").strip()
    if not directory:
        raise ValueError("directory: the output directory of a run")
    with _database({**request, "create": True}) as db:
        n = db.add_run(directory, name=request.get("name") or None)
    return {"added": n}


def database_add_session(request: dict, session: Session) -> dict:
    """Add the simulations of the workspace (``keys``, by default all) to a database
    (created if needed), each a run named after its item, for the Ashby charts."""
    from morphogen import pipeline

    keys = request.get("keys") or [i["key"] for i in session.workspace()]
    items = [
        session.item(k, ("simulation",))
        for k in keys
        if session.items.get(k, {}).get("kind") == "simulation"
    ]
    if not items:
        raise ValueError("no simulation in the workspace: simulate a morphology first")
    added = 0
    with _database({**request, "create": True}) as db:
        for item in items:
            folder = session.file(item["key"], "results.csv").parent
            rows = pipeline.collect_results(folder)
            config = {k: v for k, v in (item.get("config") or {}).items() if k != "outputs"}
            added += db.add_records(
                rows, name=item["title"], config=config, path=f"session:{item['key']}"
            )
    return {"added": added, "runs": len(items)}


def ashby(request: dict) -> dict:
    """Points of an Ashby chart of the records matching ``where``, with the axis labels
    and the guide lines."""
    from morphogen.results import ashby_points, axis_label, default_axes

    with _database(request) as db:
        records = db.records(_conditions(request))
    if not records:
        raise ValueError("no record matches the conditions")
    x, y = request.get("x") or None, request.get("y") or None
    if x is None or y is None:
        dx, dy = default_axes(records)
        x, y = x or dx, y or dy
    by = request.get("by", "model.type") or None
    log = bool(request.get("log", True))
    data = ashby_points(records, x, y, by, log=log, groups=request.get("groups") or None)
    guides = request.get("guides")
    if guides is None:
        guides = [1, 2] if log and x == "solid_fraction" and y.endswith(".relative") else []
    data.update(
        x_label=axis_label(x),
        y_label=axis_label(y),
        guides=[float(n) for n in guides],
        matched=len(records),
    )
    return data
