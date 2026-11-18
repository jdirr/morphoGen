"""The tools of the tab Stability: the patterns that periodic architectured materials may
adopt after buckling, predicted from the symmetry of their cell by group theory (Azulay &
Combescure 2024; :mod:`morphogen.homogenization.bifurcation`), for lattice frames of beams
(:mod:`morphogen.homogenization.frames`) and for the morphologies of the workspace, with
the Bloch waves of the buckling that tell the cell a pattern needs; the bifurcations of a
cell along a nonlinear loading path in finite strains (the pattern transformation of a
square array of holes); and the band structures of the elastic Bloch waves of a cell and
their band gaps (:mod:`morphogen.homogenization.bloch`).

A morphology is analysed on its primitive cell (one unit cell of its model, or its box)
repeated 1 or 2 times along each axis: on a conforming mesh of its level set, meshed in
the irreducible part of the cell and mirrored when the cell is symmetric about its
mid-planes (:func:`morphogen.mesh.mirrored_mesh`, whose symmetries are exact), or on six
tetrahedra per voxel of its voxel image (the Kuhn triangulation, which has the symmetries
of the image); a plane morphology (one voxel thick) as a slab of two elements, its level
set the same through the slab."""

from __future__ import annotations

import threading
import time
import warnings
from fractions import Fraction
from itertools import pairwise, product
from math import lcm

import numpy as np

from morphogen.gui.tools import Field, tab, tool

tab(
    "stability",
    "Stability",
    "Stability of periodic cells by group theory (Azulay & Combescure 2024): the "
    "irreducible representations of the symmetry of a cell and of its loading, their "
    "critical loads and multiplicities, the first bifurcation, its post-bifurcated patterns "
    "and the cell they need (Bloch waves), for lattice frames and for the morphologies of "
    "the workspace; bifurcations along nonlinear loading paths in finite strains (the "
    "pattern transformation of a square array of holes); and the band structures of the "
    "elastic Bloch waves of a cell, with their band gaps.",
    order=60,
)

PLANE_LATTICES = ("honeycomb", "triangular", "square", "kagome")
SPACE_LATTICES = ("cubic", "bcc", "octet", "kelvin", "diamond", "pyrochlore")
LATTICES = (
    ("honeycomb", "honeycomb (plane)"),
    ("triangular", "triangular (plane)"),
    ("square", "square (plane)"),
    ("kagome", "kagome (plane)"),
    ("cubic", "cubic lattice of struts (space)"),
    ("bcc", "body-centred cubic lattice of struts (space)"),
    ("octet", "octet truss (space)"),
    ("kelvin", "Kelvin cell (space)"),
    ("diamond", "diamond lattice of struts (space: the honeycomb of space)"),
    ("pyrochlore", "pyrochlore lattice of tetrahedra (space: the kagome of space)"),
)
LOADINGS = (
    ("hydrostatic", "hydrostatic compression (equibiaxial in the plane)"),
    ("uniaxial", "uniaxial compression along an axis"),
    ("biaxial", "biaxial compression in the x-y plane, E22 = ratio E11"),
    ("shear", "shear in the x-y plane (ε12 = γ12 / 2)"),
    ("custom", "the components of a strain"),
)
AXES = (("x", "x"), ("y", "y"), ("z", "z"))
PATTERNS = (
    ("critical", "of the irreps of the first bifurcation"),
    ("all", "of every irrep"),
)
MESHES = (
    ("auto", "auto: a mirrored conforming mesh if the cell is symmetric, else voxels"),
    ("mirrored", "a conforming mesh of its level set, mirrored (exact symmetries; MMG)"),
    ("voxels", "six tetrahedra per voxel of its voxel image"),
)
CELLS = (
    ("unit", "one unit cell of its model (TPMS, strut lattices), else its box"),
    ("box", "the box of the morphology"),
)
ORDERS = (
    ("1", "4-node tetrahedra (stiff in the bending of slender struts and walls)"),
    ("2", "10-node tetrahedra (accurate in bending; slower)"),
)
#: The models whose realizations repeat a unit cell of edges ``cell_size``
UNIT_CELL_MODELS = ("tpms", "strut_lattice")
#: The relative asymmetry of a sampled level set below which a mid-plane is a mirror
SYMMETRY_TOLERANCE = 1e-5
#: Patterns shown as views and videos, at most
SHOWN = 3
#: Patterns drawn in the figure of every irrep, at most (those of the lowest loads)
ALL_DRAWN = 30
#: Frames of the videos of the patterns oscillating
FRAMES = 36


class _Busy:
    """Reports the progress of a long call from a thread: the fraction creeping from
    ``start`` towards ``end`` (half way after ``expected`` seconds) with the time spent."""

    def __init__(self, ctx, start: float, end: float, message: str, expected: float = 20.0):
        self.ctx, self.start, self.end = ctx, start, end
        self.message, self.expected = message, max(float(expected), 1.0)
        self._stop = threading.Event()

    def __enter__(self) -> _Busy:
        self.ctx.progress(self.start, self.message)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.ctx.progress(self.end, self.message)

    def _run(self):
        t0 = time.perf_counter()
        while not self._stop.wait(0.5):
            t = time.perf_counter() - t0
            fraction = self.start + (self.end - self.start) * t / (t + self.expected)
            self.ctx.progress(fraction, f"{self.message} ({t:.0f} s)")


# -- loadings ---------------------------------------------------------------------------


def _loading_fields(default: str = "hydrostatic") -> list[Field]:
    return [
        Field(
            "loading",
            "choice",
            default,
            choices=LOADINGS,
            doc="the direction of the macroscopic strain; the critical loads λ are the "
            "magnitudes of its largest component at the bifurcations",
        ),
        Field("axis", "choice", "x", choices=AXES, when={"loading": "uniaxial"}),
        Field("ratio", "float", 1.0, when={"loading": "biaxial"}, doc="E22 / E11"),
        Field(
            "strain",
            "floats",
            label="strain components",
            when={"loading": "custom"},
            doc="ε11, ε22, ε33, ε23, ε31, ε12 (a plane frame: ε11, ε22, ε12); negative in "
            "compression",
        ),
    ]


def _strain(p: dict, dim: int) -> tuple[np.ndarray, str]:
    """The macroscopic strain of a loading of the form (dim x dim, its largest component
    of magnitude 1, compressions negative) and its description."""
    kind = p["loading"]
    e = np.zeros((dim, dim))
    if kind == "hydrostatic":
        e[:] = -np.eye(dim)
        text = "equibiaxial compression" if dim == 2 else "hydrostatic compression"
    elif kind == "uniaxial":
        axis = "xyz".index(p.get("axis") or "x")
        if axis >= dim:
            raise ValueError("axis: x or y for a plane frame")
        e[axis, axis] = -1.0
        text = f"uniaxial compression along {'xyz'[axis]}"
    elif kind == "biaxial":
        ratio = float(p.get("ratio") if p.get("ratio") is not None else 1.0)
        e[0, 0], e[1, 1] = -1.0, -ratio
        text = f"biaxial compression, E22 = {ratio:g} E11"
    elif kind == "shear":
        e[0, 1] = e[1, 0] = 1.0
        text = "shear ε12"
    else:
        values = [float(v) for v in p.get("strain") or []]
        order = (
            ((0, 0), (1, 1), (0, 1))
            if dim == 2
            else ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))
        )
        if len(values) != len(order):
            names = "ε11, ε22, ε12" if dim == 2 else "ε11, ε22, ε33, ε23, ε31, ε12"
            raise ValueError(f"strain components: {len(order)} values, {names}")
        for (a, b), v in zip(order, values, strict=True):
            e[a, b] = e[b, a] = v
        text = "the strain " + ", ".join(f"{v:g}" for v in values)
    scale = float(np.abs(e).max())
    if not scale > 0:
        raise ValueError("strain components: not all zero")
    return e / scale, text


# -- the results of the analyses ----------------------------------------------------------


def _critical(b) -> list[int]:
    """The irreps of the first bifurcation (several when their loads coincide)."""
    if b.first is None:
        return []
    load = b.first[0]
    return np.flatnonzero(np.abs(b.loads - load) <= 1e-6 * abs(load)).tolist()


def _wave_vectors(b, cells) -> list:
    """The wave vectors of each irrep (:meth:`~morphogen.homogenization.bifurcation.
    Bifurcation.wave_vectors`) when the pure translations of the group are primitive ones,
    by an edge of a primitive cell along the axes of two of them; else (centring
    translations, those of the body- or face-centred lattices, or cells of more primitive
    cells) None for every irrep."""
    g = b.group
    dim = g.rotations.shape[1]
    pure = np.abs(g.rotations - np.eye(dim)).max(axis=(1, 2)) < 1e-9
    halves = 2.0 * g.translations[pure] @ np.linalg.inv(g.lattice).T
    m = np.round(halves)
    single = np.asarray(cells[:dim]) == 1
    if np.abs(halves - m).max() > 1e-6 or np.any(m[:, single] % 2):
        return [None] * len(b.irreps)
    return [b.wave_vectors(i) for i in range(len(b.irreps))]


def _waves(waves) -> str:
    """Wave vectors in units of π over the primitive edges (Γ for zero)."""
    if waves is None:
        return ""
    return ", ".join("Γ" if not any(k) else "(" + ", ".join(map(str, k)) + ")" for k in waves)


def _doubled(waves) -> str:
    """The axes along which the patterns of these wave vectors double the period."""
    if not waves:
        return ""
    axes = sorted({i for k in waves for i, v in enumerate(k) if v})
    return ", ".join("xyz"[i] for i in axes) if axes else "none"


def _irrep_rows(b, critical, waves, *, load="critical load λ", scale=1.0) -> list[dict]:
    """The rows of the table of the irreps: their loads (times ``scale``, in the column
    ``load``), rank, wave vectors (``waves``: of each irrep) and patterns."""
    loads = np.where(np.isfinite(b.loads), b.loads, np.inf)
    rank = np.empty(len(loads), dtype=int)
    rank[np.argsort(loads, kind="stable")] = np.arange(1, len(loads) + 1)
    rows = []
    for i, irrep in enumerate(b.irreps):
        finite = bool(np.isfinite(b.loads[i]))
        rows.append(
            {
                "irrep": i + 1,
                "dimension": irrep.dim,
                "multiplicity": int(b.multiplicities[i]),
                load: float(b.loads[i]) * scale if finite else None,
                "rank": int(rank[i]) if finite else None,
                "wave vectors (π/a)": _waves(waves[i]),
                "period doubled along": _doubled(waves[i]),
                "patterns": sum(p.irrep == i for p in b.patterns),
                "first bifurcation": "yes" if i in critical else "",
            }
        )
    return rows


def _pattern_rows(b, *, load="critical load λ", scale=1.0) -> list[dict]:
    rows = []
    for p in b.patterns:
        d = p.description
        rows.append(
            {
                "irrep": p.irrep + 1,
                "point group": d["point_group"],
                "order |H|": d["order"],
                "translations": d["translations"],
                "conjugates": len(p.subgroups),
                "dim Fix(H)": p.dimension,
                "energy": p.energy,
                load: p.load * scale if np.isfinite(p.load) else None,
            }
        )
    return rows


def _first(b, waves) -> tuple[dict, str]:
    """The values of the first bifurcation and a sentence on it (``waves``: the wave
    vectors of each irrep)."""
    if b.first is None:
        return {"bifurcation": "none"}, "No bifurcation under this loading."
    load, i, count = b.first
    waves = waves[i]
    doubled = _doubled(waves)
    values = {
        "first critical load λ": load,
        "irrep of the first bifurcation": i + 1,
        "its dimension": b.irreps[i].dim,
        "multiplicity": count,
    }
    if doubled:
        values["period doubled along"] = doubled
    kinds = {1: "simple", 2: "double", 3: "triple", 4: "quadruple", 6: "sextuple"}
    classes = [p.description for p in b.patterns if p.irrep in _critical(b)]
    text = (
        f"First bifurcation at λ = {load:.5g}, in irrep {i + 1} of dimension "
        f"{b.irreps[i].dim}: a {kinds.get(count, f'{count}-fold')} point"
    )
    if classes:
        text += "; its patterns: " + ", ".join(
            f"{d['point_group']} (|H| = {d['order']}, {d['translations']} "
            f"translation{'s' if d['translations'] > 1 else ''})"
            for d in classes
        )
    if doubled and doubled != "none":
        text += f". Its wave vectors {_waves(waves)} double the period along {doubled}"
    elif doubled == "none":
        text += ". Its patterns repeat in every primitive cell"
    return values, text + "."


def _patterns_of(b, irreps) -> list:
    """The patterns of these irreps, the softest first."""
    return sorted((p for p in b.patterns if p.irrep in irreps), key=lambda p: p.energy)


def _first_irreps(b, most: int) -> list[int]:
    """The irreps with patterns, by increasing critical load (in their order without
    loads), while their patterns number at most ``most`` (one irrep at least)."""
    present = sorted(
        {p.irrep for p in b.patterns}, key=lambda i: (np.nan_to_num(b.loads[i], nan=np.inf), i)
    )
    out, count = [], 0
    for i in present:
        count += sum(p.irrep == i for p in b.patterns)
        if out and count > most:
            break
        out.append(i)
    return out


def _title(p) -> str:
    d = p.description
    return f"irrep {p.irrep + 1}: {d['point_group']}, |H| = {d['order']}"


# -- lattice frames -------------------------------------------------------------------------


@tool(
    "frame_bifurcation",
    "stability",
    "Post-bifurcation of a lattice frame",
    "The post-bifurcated patterns of a periodic frame of Euler–Bernoulli beams by group "
    "theory (Azulay & Combescure 2024): a plane lattice (honeycomb, triangular, square, "
    "kagome) or a lattice of struts in space, in a cell of n × n (× n) primitive cells, "
    "under a macroscopic strain. The symmetry group of the cell restricted to the loading, "
    "its irreducible representations, the multiplicity of each among the fluctuations and "
    "its lowest critical load, the first bifurcation (its irrep, multiplicity and wave "
    "vectors), and the pattern of each axial isotropy subgroup (equivariant branching "
    "lemma); the Bloch-wave buckling of the primitive cell over its Brillouin zone tells "
    "the critical wavelength, hence the cell the patterns need (Geymonat, Müller & "
    "Triantafyllidis 1993).",
    [
        Field("lattice", "choice", "honeycomb", choices=LATTICES),
        Field(
            "cells",
            "int",
            label="primitive cells per side",
            doc="n × n in the plane (1 to 6), n × n × n in space (1 or 2); blank: 2 in the "
            "plane, 1 in space",
        ),
        Field(
            "elements",
            "int",
            label="beams per wall",
            doc="each wall or strut cut in (blank: 4 in the plane, 2 in space)",
        ),
        Field(
            "thickness",
            "float",
            0.05,
            doc="of the walls, relative to their length",
            when={"lattice": list(PLANE_LATTICES)},
        ),
        Field(
            "radius",
            "float",
            0.04,
            doc="of the struts, relative to the edge of the cubic cell",
            when={"lattice": list(SPACE_LATTICES)},
        ),
        *_loading_fields(),
        Field("patterns", "choice", "all", choices=PATTERNS, doc="the patterns computed"),
        Field(
            "bloch_grid",
            "int",
            label="Bloch-wave grid",
            doc="wave vectors per reciprocal vector over the Brillouin zone of the primitive "
            "cell (blank: 12 in the plane, 4 in space; 0: none)",
        ),
        Field("video", "bool", True, doc="a video of each pattern of the first bifurcation"),
    ],
    order=1,
    menu="Bifurcations",
)
def frame_bifurcation(p: dict, ctx) -> dict:
    from morphogen.homogenization.bifurcation import post_bifurcation
    from morphogen.homogenization.frames import SPACE_LATTICES as STRUTS
    from morphogen.homogenization.frames import Frame, SpaceFrame
    from morphogen.symmetry import Group

    kind = p["lattice"]
    space = kind in SPACE_LATTICES
    dim = 3 if space else 2
    n = int(p["cells"] or (1 if space else 2))
    elements = int(p["elements"] or (2 if space else 4))
    limit = 2 if space else 6
    if not 1 <= n <= limit:
        raise ValueError(f"cells: from 1 to {limit} for a {'space' if space else 'plane'} frame")
    if not 1 <= elements <= 16:
        raise ValueError("beams per wall: from 1 to 16")
    if space:
        radius = float(p["radius"])
        if not 0 < radius < 0.25:
            raise ValueError("radius: between 0 and 0.25 of the cubic cell")
        frame = SpaceFrame.lattice_of(kind, (n,) * 3, elements=elements, radius=radius)
        cell = SpaceFrame.lattice_of(kind, (1,) * 3, elements=elements, radius=radius)
        length = float(STRUTS[kind]["strut"])
    else:
        thickness = float(p["thickness"])
        if not 0 < thickness < 0.5:
            raise ValueError("thickness: between 0 and 0.5 of the walls")
        frame = Frame.lattice_of(kind, (n, n), elements=elements, thickness=thickness)
        cell = Frame.lattice_of(kind, (1, 1), elements=elements, thickness=thickness)
        length = 1.0
    strain, loading = _strain(p, dim)
    grid = p["bloch_grid"]
    grid = (4 if space else 12) if grid is None else int(grid)
    if not 0 <= grid <= (8 if space else 48):
        raise ValueError(f"Bloch-wave grid: from 0 to {8 if space else 48}")
    ctx.progress(0.05, "axial forces of the beams")
    forces, _ = frame.axial_forces(strain)
    compressed = float(forces.min()) < -1e-9 * float(frame.axial.max())
    expected = 1.0 + (frame.n_dofs / 400.0) ** 2
    with _Busy(ctx, 0.1, 0.55, "group-theoretic analysis", expected):
        if compressed:
            b = post_bifurcation(frame, strain, patterns=p["patterns"])
        else:  # no beam in compression: no buckling, the patterns of the symmetry alone
            group = Group.generate(frame.generators, frame.lattice, frame.centre)
            group = group.subgroup(group.stabilizer_of(strain))
            b = post_bifurcation(frame, None, group=group, patterns="all")
    critical = _critical(b)
    waves = _wave_vectors(b, (n,) * dim)
    cells = " × ".join([str(n)] * dim)
    values: dict = {
        "lattice": kind,
        "cell": f"{cells} primitive cells",
        "beams": len(frame.beams),
        "dofs": frame.n_dofs,
        "group order": b.group.order,
        "irreps": len(b.irreps),
    }
    first, sentence = _first(b, waves)
    values.update(first)
    if not compressed:
        sentence = "No beam is in compression: no buckling; the patterns of the symmetry alone."
    text = [
        f"{kind.capitalize()}, a cell of {cells} primitive cells ({frame.n_dofs} dofs), "
        f"{loading}: a group of {b.group.order} operations restricted to the loading, "
        f"{len(b.irreps)} irreps.",
        sentence,
    ]
    if b.first is not None:
        ratio = -b.first[0] * forces * length**2 / frame.bending
        values["axial force N L²/EI"] = float(ratio.max())
        text.append(
            f"The most compressed walls then carry N L²/EI = {ratio.max():.4g} (Euler: π² "
            "between pinned ends, 4π² clamped)."
        )
    ctx.table(_irrep_rows(b, critical, waves), "irreps", "irreducible representations")
    if b.patterns:
        ctx.table(_pattern_rows(b), "patterns", "patterns: the axial isotropy subgroups")
    ctx.progress(0.6, "figures of the patterns")
    if critical and any(q.irrep in critical for q in b.patterns):
        ctx.figure(
            b.figure(frame, irreps=critical), "patterns_first", "patterns of the first bifurcation"
        )
    if p["patterns"] == "all" and b.patterns and (len(critical) < len(b.irreps)):
        drawn = _first_irreps(b, ALL_DRAWN)
        title = (
            "patterns of every irrep"
            if len(drawn) == len({q.irrep for q in b.patterns})
            else f"patterns of the {len(drawn)} first irreps"
        )
        ctx.figure(b.figure(frame, irreps=drawn), "patterns_all", title)
    if p["video"] and critical:
        from morphogen.viz.video import frame_pattern_frames

        shown = _patterns_of(b, critical)[:SHOWN]
        for k, pattern in enumerate(shown):
            ctx.progress(0.7 + 0.1 * k / len(shown), "videos of the patterns")
            frames = frame_pattern_frames(frame, pattern.mode, frames=FRAMES, size=420)
            ctx.video(frames, f"pattern_{k + 1}", _title(pattern), mode="loop")
    if grid and compressed:
        ctx.progress(0.82, "Bloch waves of the primitive cell")
        bloch = _frame_bloch(cell, strain, grid, space, b, n)
        if bloch is not None:
            values.update(bloch["values"])
            text.append(bloch["text"])
            ctx.figure(bloch["figure"], "bloch", "Bloch-wave buckling of the primitive cell")
            ctx.table(bloch["rows"], "bloch", "Bloch-wave buckling (lowest loads first)")
    return {"values": values, "text": " ".join(text)}


def _frame_bloch(cell, strain, n: int, space: bool, b, cells: int) -> dict | None:
    """The lowest load of the Bloch modes of the primitive cell over an ``n`` per side
    grid of its Brillouin zone, the wave vector where it is reached and the cell that
    holds it (the smallest of those of equal loads)."""
    from matplotlib.figure import Figure

    dim = 3 if space else 2
    fractions = np.array(list(product(range(n), repeat=dim)), dtype=float) / n
    recip = cell.reciprocal()
    loads = cell.bloch_buckling(strain, fractions @ recip.T)[:, 0]
    if not np.isfinite(loads).any():
        return None
    low = float(np.nanmin(loads))

    def needed(f) -> int:
        return lcm(*(Fraction(round(v * n), n).denominator for v in f))

    near = np.flatnonzero(np.abs(loads - low) <= 1e-6 * low)
    best = min(near, key=lambda i: (needed(fractions[i]), i))
    f, m = fractions[best], needed(fractions[best])
    frac = "(" + ", ".join(str(Fraction(round(v * n), n)) for v in f) + ")"
    side = " × ".join([str(m)] * dim)
    held = cells % m == 0
    values = {
        "Bloch: lowest load": low,
        "Bloch: wave vector (b)": frac,
        "Bloch: cell needed": side,
        "Bloch: held by the cell": bool(held),
        "Bloch: load at Γ": float(loads[0]),
    }
    text = (
        f"Bloch waves of the primitive cell: the lowest load over a {n}^{dim} grid of the "
        f"Brillouin zone, {low:.5g}, at k = {frac} in the reciprocal basis: a cell of {side} "
        "primitive cells holds it"
    )
    if held:
        text += ", as the analysed one does."
    elif b.first is not None:
        text += f"; the analysed cell does not (its first load {b.first[0]:.5g})."
    else:
        text += "; the analysed cell does not."
    closest = np.minimum(f, 1.0 - f)
    if any(closest) and closest.max() <= 1.0 / n + 1e-12:
        text += (
            " The lowest loads lie next to Γ: a long-wave (macroscopic) instability, which "
            "no finite cell holds; refine the grid."
        )
    fig = Figure(figsize=(5.4, 4.2), layout="constrained")
    ax = fig.subplots()
    if space:  # along the edges of the cube: Γ X M Γ R
        corners = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]])
        steps = 4
        path = [corners[0]]
        for a, c in pairwise(corners):
            path += [a + t * (c - a) for t in np.linspace(0, 1, steps + 1)[1:]]
        path = np.array(path)
        along = cell.bloch_buckling(strain, path @ recip.T)[:, 0]
        ax.plot(np.arange(len(path)), along, color="C0", lw=1.4, label="primitive cell")
        ax.set_xticks(np.arange(0, len(path), steps), ["Γ", "X", "M", "Γ", "R"])
        ax.axhline(low, color="k", ls=":", lw=1.0, label=f"lowest of the grid, at {frac}")
        if b.first is not None:
            ax.axhline(b.first[0], color="C3", ls="--", lw=1.0, label="the analysed cell")
        ax.set_ylabel("critical load λ")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
    else:
        image = (loads / low).reshape(n, n).T
        shown = ax.imshow(
            image,
            origin="lower",
            extent=(-0.5 / n, 1 - 0.5 / n, -0.5 / n, 1 - 0.5 / n),
            vmin=1.0,
            vmax=max(min(4.0, float(np.nanmax(image))), 1.01),
            cmap="viridis_r",
        )
        fig.colorbar(shown, ax=ax, label="λ / lowest λ")
        ax.plot(*f, "r+", ms=12, mew=2)
        ax.set_xlabel("k · a1 / 2π")
        ax.set_ylabel("k · a2 / 2π")
    ax.set_title("Bloch-wave buckling of the primitive cell", fontsize=10)
    order = np.argsort(np.where(np.isfinite(loads), loads, np.inf), kind="stable")
    rows = [
        {
            **{f"k{i + 1} (b)": float(fractions[j, i]) for i in range(dim)},
            "load λ": float(loads[j]) if np.isfinite(loads[j]) else None,
            "cell needed": needed(fractions[j]),
        }
        for j in order
    ]
    return {"values": values, "text": text, "figure": fig, "rows": rows}


# -- the cells of morphologies --------------------------------------------------------------


class _CellBody:
    """The primitive cell of a morphology as an implicit body of its level set (as
    :class:`~morphogen.mesh.ImplicitBody`): the box ``[lower, lower + size]``, periodic;
    a plane morphology as a slab ``thickness`` thick whose level set is that of its
    mid-plane through the slab."""

    def __init__(self, micro, lower, size, thickness: float | None = None):
        from morphogen.domain import Domain

        self.micro, self.thickness = micro, thickness
        lower, size = np.array(lower, dtype=float), np.array(size, dtype=float)
        if thickness is not None:
            lower[2], size[2] = 0.0, float(thickness)
        self.domain = Domain(size=tuple(size), origin=tuple(lower), periodic=True)

    def level_set(self, shape=None, *, voxel_size=None, grid=None):
        from morphogen.domain import Domain, Grid
        from morphogen.image import VoxelImage

        if grid is None:
            grid = Grid.for_domain(self.domain, shape=shape, voxel_size=voxel_size)
        if self.thickness is None:
            return self.micro.level_set(grid=grid)
        box, own = grid.domain, self.micro.domain
        plane = Grid(
            Domain(
                size=(box.size[0], box.size[1], own.size[2]),
                origin=(box.origin[0], box.origin[1], own.origin[2]),
                periodic=False,
            ),
            (grid.shape[0], grid.shape[1], 1),
        )
        values = np.asarray(self.micro.level_set(grid=plane).data)
        data = np.ascontiguousarray(np.broadcast_to(values, grid.shape))
        return VoxelImage(data, spacing=tuple(grid.spacing), origin=tuple(grid.origin))


def _planar(micro) -> bool:
    from morphogen.gui.api import _planar

    return _planar(micro)


def _periodic(item: dict) -> None:
    """Warn that a morphology generated in a window of the medium is taken as periodic."""
    if not item["micro"].domain.periodic:
        warnings.warn(
            f"{item['title']} is not periodic: its cell is analysed as a periodic one, whose "
            "opposite faces may not match",
            stacklevel=2,
        )


def _primitive(item: dict, choice: str) -> tuple[np.ndarray, np.ndarray, str]:
    """The primitive cell of a morphology: the lower corner and edges of one unit cell of
    its model (``choice`` "unit", for the models of :data:`UNIT_CELL_MODELS` whose box
    holds a whole number of them) or of its box, and its description."""
    micro = item["micro"]
    lower = np.asarray(micro.domain.lower, dtype=float)
    lengths = np.asarray(micro.domain.lengths, dtype=float)
    model = item.get("model") or {}
    if choice == "unit" and model.get("type") in UNIT_CELL_MODELS:
        cell = np.array(np.broadcast_to(np.asarray(model.get("cell_size", 1.0), float), (3,)))
        if _planar(micro):
            cell[2] = lengths[2]
        ratio = lengths / cell
        whole = np.abs(ratio - np.rint(ratio)) <= 1e-6 * np.maximum(ratio, 1.0)
        if np.all(whole) and np.all(np.rint(ratio) >= 1):
            counts = " × ".join(f"{r:g}" for r in np.rint(ratio))
            return lower, cell, f"a unit cell of its {model['type']} ({counts} in its box)"
    return lower, lengths, "its box"


def _mirrors(body, n: int = 24) -> tuple[list[int], list[tuple[int, int]]]:
    """The mid-planes of a cell about which its level set (sampled on ``n`` points per
    edge) is symmetric, and the diagonal planes of pairs of them of equal edges about
    which it is symmetric too (the chain of pairs of :func:`~morphogen.mesh.
    mirrored_mesh`)."""
    from morphogen.domain import Domain, Grid

    box = body.domain
    shape = (n, n, 2 if body.thickness is not None else n)
    try:
        sample = body.level_set(
            grid=Grid(Domain(size=box.size, origin=box.origin, periodic=False), shape)
        )
    except NotImplementedError:
        return [], []
    v = np.asarray(sample.data, dtype=float)
    scale = float(np.abs(v).max()) or 1.0
    tol = SYMMETRY_TOLERANCE * scale
    mirrors = [k for k in range(3) if np.abs(v - np.flip(v, axis=k)).max() <= tol]
    lengths = np.asarray(box.size, dtype=float)
    pairs = [
        (a, b)
        for a in mirrors
        for b in mirrors
        if a < b
        and v.shape[a] == v.shape[b]
        and abs(lengths[a] - lengths[b]) <= 1e-9 * lengths.max()
        and np.abs(v - np.swapaxes(v, a, b)).max() <= tol
    ]
    if len(pairs) == 3:  # the cube: two of them generate the third
        pairs = [(0, 1), (1, 2)]
    return mirrors, pairs[:2]


def _materials_of(item: dict, given) -> dict:
    """The materials of the phases of a morphology (``given``: those of the form, phase
    -> "void" or isotropic constants; by default, as the forms of the page, void where the
    phase is named so, else E = 1 for phase 0 and 10 for the others, ν = 0.3)."""
    from morphogen.homogenization.materials import VOID, Material

    names = {int(k): v for k, v in (item.get("phases") or {"0": "0", "1": "1"}).items()}
    if given:
        out = {}
        for k, spec in given.items():
            out[int(k)] = (
                VOID
                if spec == "void"
                else Material(young=float(spec["young"]), poisson=float(spec["poisson"]))
            )
    else:
        out = {
            k: VOID if name == "void" else Material(young=1.0 if k == 0 else 10.0, poisson=0.3)
            for k, name in names.items()
        }
    if all(m.void for m in out.values()):
        raise ValueError("materials: every phase is void")
    return out


def _solid_phases(materials: dict) -> list[int]:
    return sorted(k for k, m in materials.items() if not m.void)


def _cell_sample(item: dict, p: dict, materials: dict, ctx) -> dict:
    """The periodic cell of a morphology for the analyses by group theory: a conforming
    mesh of its level set, mirrored (the symmetries of the cell about its mid-planes
    exact; ``p["mesh"]`` "mirrored", or "auto" when the cell is symmetric about one of
    them at least), or the voxel image of its cell (``p["resolution"]`` voxels along its
    shortest edge, even: the Kuhn triangulation is symmetric under translations by even
    numbers of voxels); ``p["cells"]`` copies of the cell along each axis (1 or 2; the
    plane ones along x and y only). Returns the sample, its cells, the phases kept and a
    description."""
    from morphogen.domain import Domain, Grid
    from morphogen.image import VoxelImage
    from morphogen.mesh import mirrored_mesh

    micro = item["micro"]
    planar = _planar(micro)
    lower, size, origin = _primitive(item, p.get("cell") or "unit")
    cells = p.get("cells") or ([2, 2, 1] if planar else [1, 1, 1])
    cells = list(cells) * 3 if len(cells) == 1 else list(cells)  # one count for all axes
    if len(cells) != 3 or any(c not in (1, 2) for c in cells):
        raise ValueError("cells: three counts of 1 or 2 primitive cells (x, y, z)")
    if planar:
        cells = [cells[0], cells[1], 1]
    cells = tuple(int(c) for c in cells)
    solid = _solid_phases(materials)
    present = sorted(int(k) for k in (item.get("phases") or {"0": 0, "1": 1}))
    missing = [k for k in present if k not in materials]
    if missing:
        raise ValueError(f"materials: none for phase(s) {missing}")
    edge = float(size[:2].min() if planar else size.min())
    kind = p.get("mesh") or "auto"
    out = {"cells": cells, "planar": planar, "origin": origin}
    two_phases = set(present) <= {0, 1}
    if kind in ("auto", "mirrored"):
        h = p.get("element_size") or edge / (8 if planar else 6)
        if not 0 < h <= edge:
            raise ValueError("element size: positive, at most the edge of the cell")
        body = _CellBody(micro, lower, size, 2 * h if planar else None)
        ctx.progress(0.03, "symmetries of the level set")
        mirrors, pairs = _mirrors(body) if two_phases else ([], [])
        if kind == "mirrored" and not two_phases:
            raise ValueError(
                "mesh: a conforming mesh of the level set has two phases; this "
                "morphology has more (use voxels)"
            )
        if kind == "mirrored" and not mirrors:
            raise ValueError(
                "mesh: the level set of this cell is symmetric about none of its "
                "mid-planes (or it has none): use voxels"
            )
        if mirrors:
            meshed = "solid" if solid == [1] else "all"
            ctx.progress(0.06, "conforming mesh (MMG), mirrored")
            mesh = mirrored_mesh(
                body,
                size=h,
                mirrors=mirrors,
                diagonals=pairs,
                cells=cells,
                phases=meshed,
                check=False,
            )
            planes = ", ".join("xyz"[k] for k in mirrors)
            if pairs:
                planes += "; diagonals " + ", ".join(f"{'xyz'[a]}{'xyz'[b]}" for a, b in pairs)
            out.update(
                sample=mesh,
                phases=None,
                kind="mirrored",
                description=f"a conforming mesh of the level set of {origin} (elements "
                f"of {h:.3g}), meshed in its irreducible part and mirrored about the "
                f"planes {planes}",
            )
            return out
    n = int(p.get("resolution") or 8)
    n += n % 2
    if not 2 <= n <= 64:
        raise ValueError("voxels per edge: from 2 to 64")
    h = edge / n
    shape = np.maximum(1, np.rint(size / h)).astype(int)
    if planar:
        shape[2] = 1
    grid = Grid(Domain(size=tuple(size), origin=tuple(lower), periodic=True), tuple(shape))
    ctx.progress(0.05, "voxel image of the cell")
    data = np.tile(np.asarray(micro.voxelize(grid=grid).data), cells)
    spacing = (h, h, h) if planar else tuple(grid.spacing)
    image = VoxelImage(data, spacing=spacing, origin=tuple(lower), periodic=True)
    out.update(
        sample=image,
        phases=solid,
        kind="voxels",
        description=f"the voxel image of {origin}, {' × '.join(map(str, shape))} voxels "
        "of six tetrahedra",
    )
    return out


def _cells_text(cells) -> str:
    return " × ".join(map(str, cells))


def _unresolved(exc: Exception, kind: str) -> Exception:
    """The error of a cell left without material (its solid in pieces that touch no face of
    the box), with what to change; other errors as they are."""
    if "material" not in str(exc):
        return exc
    hint = (
        "more voxels per edge: these do not resolve the solid of the cell into a body that "
        "reaches its faces (or a conforming mesh)"
        if kind == "voxels"
        else "a finer mesh (element size)"
    )
    return ValueError(f"{exc}: give {hint}")


def _check_size(sample, cells, order: int, phases, *, split: bool) -> None:
    """Refuse a cell whose dense bases of the isotypic subspaces would not fit in memory:
    about a quarter of the fluctuations each (three arrays of them), those of a primitive
    cell when the analysis splits the cell by its translations (``split``)."""
    from morphogen.gui.api import check_memory

    if hasattr(sample, "n_nodes"):  # a mesh
        nodes = float(sample.n_nodes) * (8 if order == 2 and sample.order == 1 else 1)
    else:  # about a node per voxel kept
        kept = np.isin(sample.data, phases).mean() if phases else 1.0
        nodes = float(np.prod(np.asarray(sample.shape) + 1)) * kept * (8 if order == 2 else 1)
    dofs = 3.0 * nodes
    block = dofs / int(np.prod(cells)) if split and dofs > 1500 else dofs
    check_memory(6.0 * block * block, f"the bases of the analysis (about {dofs:.0f} dofs)")


# -- views of the patterns on meshes ----------------------------------------------------------


def _mesh_frames(mesh, u, *, planar: bool, frames: int = 1, phase: bool = False):
    """Views of a tetrahedral mesh deformed by a mode ``u`` (nodes x 3): one view, or the
    mode oscillating over ``frames``; coloured by the norm of ``u`` (or by the phases,
    undeformed: ``phase``), seen from above when ``planar``. VTK renders them when it can
    here, else matplotlib."""
    from morphogen.viz.video import deformed_view, field_frames, vtk_can_render

    view = {"elevation": 0.0, "azimuth": 0.0} if planar else {}  # VTK looks down z
    if vtk_can_render():
        if phase:
            return [deformed_view(mesh, None, "phase", edges=True, **view)]
        if frames == 1:
            return [deformed_view(mesh, u, u, title="|u|", **view)]
        return field_frames(mesh, u, frames=frames, title="|u|", **view)
    return _matplotlib_frames(mesh, None if phase else u, planar=planar, frames=frames)


def _matplotlib_frames(mesh, u, *, planar: bool, frames: int = 1) -> list[np.ndarray]:
    """The views of :func:`_mesh_frames` drawn by matplotlib: the outer faces of the mesh
    (its top face when planar), coloured by the norm of ``u`` (or by the phases)."""
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = mesh.free_faces()[:, :3]
    x0 = mesh.nodes
    if planar:
        faces = faces[np.all(np.abs(x0[faces, 2] - mesh.upper[2]) <= mesh.tolerance(), axis=1)]
    if u is None:  # the phases, undeformed
        u = np.zeros_like(x0)
        phase_of = np.zeros(mesh.n_nodes, dtype=int)
        phase_of[mesh.elements[:, :4].ravel()] = np.repeat(mesh.phase, 4)
        colours = colormaps["tab10"](phase_of[faces].max(axis=1) % 10)
    else:
        norm = np.linalg.norm(u, axis=1)
        colours = colormaps["viridis"](norm[faces].mean(axis=1) / max(norm.max(), 1e-300))
    peak = float(np.linalg.norm(u, axis=1).max())
    m = 0.1 * float(np.ptp(x0, axis=0).max()) / max(peak, 1e-300)
    fig = Figure(figsize=(5.6, 4.8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    lo, hi = x0.min(axis=0) - m * peak, x0.max(axis=0) + m * peak
    if planar:
        ax = fig.add_axes((0, 0, 1, 1))
        coll = PolyCollection(x0[faces][:, :, :2], facecolors=colours, edgecolors="face")
        ax.add_collection(coll)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
    else:
        ax = fig.add_subplot(projection="3d")
        coll = Poly3DCollection(x0[faces], facecolors=colours, edgecolors="none")
        ax.add_collection3d(coll)
        for setter, a, b in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi, strict=True):
            setter(a, b)
        ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)))
    ax.set_axis_off()
    scales = [1.0] if frames == 1 else np.sin(2 * np.pi * np.arange(frames) / frames)
    out = []
    for s in scales:
        x = x0 + s * m * u
        coll.set_verts(x[faces][:, :, :2] if planar else x[faces])
        canvas.draw()
        out.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
    return out


def _show_patterns(b, ctx, critical, *, planar: bool, video: bool, start: float) -> None:
    """Views of the analysed mesh and of the patterns of the critical irreps (the softest
    first, :data:`SHOWN` at most) and videos of them oscillating."""
    ctx.progress(start, "views of the mesh and of the patterns")
    ctx.figure(
        _mesh_frames(b.mesh, None, planar=planar, phase=True)[0],
        "mesh",
        "the analysed mesh (phases)",
    )
    shown = _patterns_of(b, critical)[:SHOWN]
    for k, pattern in enumerate(shown):
        u = b.displacement(pattern.mode)
        ctx.figure(_mesh_frames(b.mesh, u, planar=planar)[0], f"pattern_{k + 1}", _title(pattern))
        if video:
            ctx.progress(start + (0.98 - start) * (k + 1) / len(shown), "videos of the patterns")
            ctx.video(
                _mesh_frames(b.mesh, u, planar=planar, frames=FRAMES),
                f"pattern_{k + 1}",
                _title(pattern),
                mode="loop",
            )


# -- morphologies ---------------------------------------------------------------------------


def _mesh_fields(default_mesh: str = "auto", resolution: int = 8) -> list:
    return [
        Field("cell", "choice", "unit", choices=CELLS, label="primitive cell"),
        Field(
            "cells",
            "ints",
            doc="primitive cells along x, y and z, 1 or 2 each, or one count for all (blank: "
            "2 × 2 × 1 for a plane morphology, 1 × 1 × 1 otherwise)",
        ),
        Field("mesh", "choice", default_mesh, choices=MESHES),
        Field(
            "element_size",
            "float",
            unit="length",
            when={"mesh": ["auto", "mirrored"]},
            doc="of the conforming mesh (blank: 1/8 of the shortest edge of a plane cell, 1/6 "
            "in space)",
        ),
        Field(
            "resolution",
            "int",
            resolution,
            label="voxels per edge",
            when={"mesh": ["auto", "voxels"]},
            doc="along the shortest edge of the primitive cell (even; a multiple of 4 keeps "
            "the translations by half a cell)",
        ),
    ]


@tool(
    "morphology_bifurcation",
    "stability",
    "Post-bifurcation of a morphology",
    "The post-bifurcated patterns of a morphology of the workspace by group theory (Azulay "
    "& Combescure 2024): its primitive cell (a unit cell of its model, or its box) "
    "repeated 1 or 2 times along each axis, meshed conforming to its level set and "
    "mirrored when the cell is symmetric about its mid-planes (exact symmetries), or cut "
    "into six tetrahedra per voxel; its symmetry group (detected on the mesh) restricted "
    "to a macroscopic strain, the linear buckling of each irreducible representation "
    "(about the initial stress of the periodic linear solution), the first bifurcation, "
    "its patterns on the mesh and its wave vectors: the patterns of a wave vector of "
    "components π/a double the period along those axes. A plane morphology is a slab of "
    "two elements, periodic through its thickness.",
    [
        Field("source", "item", label="morphology", required=True),
        *_mesh_fields(),
        Field("order", "choice", "1", choices=ORDERS, label="elements"),
        Field("materials", "materials", of="source"),
        *_loading_fields(),
        Field("patterns", "choice", "critical", choices=PATTERNS, doc="the patterns computed"),
        Field("video", "bool", True, doc="a video of each pattern of the first bifurcation"),
    ],
    order=2,
    menu="Bifurcations",
)
def morphology_bifurcation(p: dict, ctx) -> dict:
    from morphogen.homogenization.bifurcation import post_bifurcation

    item = p["source"]
    _periodic(item)
    materials = _materials_of(item, p["materials"])
    strain, loading = _strain(p, 3)
    cell = _cell_sample(item, p, materials, ctx)
    order = int(p["order"] or 1)
    sample = cell["sample"]
    size = sample.n_nodes if hasattr(sample, "n_nodes") else int(np.prod(sample.shape))
    _check_size(sample, cell["cells"], order, cell["phases"], split=True)
    with _Busy(ctx, 0.1, 0.7, "group-theoretic analysis", 2.0 + size / 300.0 * order**2):
        try:
            b = post_bifurcation(
                sample,
                strain,
                materials=materials,
                cells=cell["cells"],
                order=order,
                phases=cell["phases"],
                patterns=p["patterns"],
            )
        except ValueError as exc:
            raise _unresolved(exc, cell["kind"]) from None
    critical = _critical(b)
    waves = _wave_vectors(b, cell["cells"])
    dofs = int(b.lift.shape[1])
    values: dict = {
        "mesh": cell["kind"],
        "cells": _cells_text(cell["cells"]),
        "elements": b.mesh.n_elements,
        "dofs": dofs,
        "group order": b.group.order,
        "irreps": len(b.irreps),
    }
    first, sentence = _first(b, waves)
    values.update(first)
    text = [
        f"{item['title']}: {cell['description']}; {_cells_text(cell['cells'])} primitive "
        f"cells, {b.mesh.n_elements} elements, {dofs} dofs. {loading.capitalize()}: a group "
        f"of {b.group.order} operations restricted to the loading, {len(b.irreps)} irreps.",
        sentence,
    ]
    if b.group.order == 1:
        text.append("The mesh has no symmetry: every mode is in the trivial irrep.")
    ctx.table(_irrep_rows(b, critical, waves), "irreps", "irreducible representations")
    if b.patterns:
        ctx.table(_pattern_rows(b), "patterns", "patterns: the axial isotropy subgroups")
        b.write(ctx.path("patterns.vtu"), irreps=critical or None)
    _show_patterns(b, ctx, critical, planar=cell["planar"], video=p["video"], start=0.72)
    return {"values": values, "text": " ".join(text)}


# -- along nonlinear paths --------------------------------------------------------------------

CELL_KINDS = (
    ("holes", "a square array of circular holes (Mullin et al. 2007; Bertoldi et al. 2008)"),
    ("morphology", "a morphology of the workspace"),
)
PATH_LOADINGS = (
    ("uniaxial_stress", "uniaxial compression, the lateral stresses free"),
    ("uniaxial_strain", "uniaxial compression, the lateral strains held"),
    ("equibiaxial", "equibiaxial compression in the x-y plane"),
    ("hydrostatic", "hydrostatic compression (equibiaxial in the plane)"),
)
BEHAVIOURS = (
    ("elastic", "elastic: Hencky in finite strains"),
    ("plastic", "von Mises elastoplastic, linear hardening"),
)
KINEMATICS = (("finite", "finite strains"), ("small", "small strains"))
NAMES = ("11", "22", "33", "23", "31", "12")


def _path(p: dict, planar: bool) -> tuple[dict, int, str]:
    """The loading path of the form (a nominal compressive strain reached at the time 1,
    shears held at zero, the out-of-plane strain of a plane cell too), the loaded axis and
    a description."""
    eps, n = float(p["strain"]), int(p["increments"])
    if not 0 < eps < 0.9:
        raise ValueError("strain: a compression between 0 and 0.9")
    if not 2 <= n <= 400:
        raise ValueError("increments: from 2 to 400")
    kind = p["loading"]
    axis = "xyz".index(p.get("axis") or "y")
    if planar and axis == 2 and kind.startswith("uniaxial"):
        raise ValueError("axis: x or y for a plane cell")
    spec = {"23": 0.0, "31": 0.0, "12": 0.0}
    if planar:
        spec["33"] = 0.0
    if kind.startswith("uniaxial"):
        spec[NAMES[axis]] = -eps
        if kind == "uniaxial_strain":
            for k in range(3):
                spec.setdefault(NAMES[k], 0.0)
        text = f"uniaxial compression along {'xyz'[axis]}, the lateral " + (
            "strains held" if kind == "uniaxial_strain" else "stresses free"
        )
    elif kind == "equibiaxial" or planar:
        spec["11"] = spec["22"] = -eps
        axis, text = 0, "equibiaxial compression in the x-y plane"
    else:
        spec["11"] = spec["22"] = spec["33"] = -eps
        axis, text = 0, "hydrostatic compression"
    if planar:
        text += ", in plane strain"
    return {"times": [0, 1], "increments": n, "strain": spec}, axis, text


def _behaviour(materials: dict, p: dict) -> dict:
    """The materials with the behaviour of the form: von Mises plasticity of yield strain
    and hardening relative to the Young's modulus of each phase."""
    import dataclasses

    from morphogen.homogenization.behaviours import Plasticity

    if p["behaviour"] != "plastic":
        return materials
    strain, hardening = float(p["yield_strain"]), float(p["hardening"])
    if not strain > 0 or hardening < 0:
        raise ValueError("yield strain: positive; hardening: non-negative")
    return {
        k: m
        if m.void
        else dataclasses.replace(
            m, plasticity=Plasticity(strain * m.young, hardening=hardening * m.young)
        )
        for k, m in materials.items()
    }


def _holes(p: dict, ctx) -> dict:
    """The 2 x 2 cell of a square array of circular holes (cells of edge 1, one layer):
    a conforming mesh mirrored with the diagonals of the square, or voxels."""
    from morphogen.domain import Domain
    from morphogen.image import VoxelImage
    from morphogen.mesh import ImplicitBody, mirrored_mesh

    porosity = float(p["porosity"])
    if not 0 < porosity < np.pi / 4:
        raise ValueError("porosity: between 0 and π/4 (holes apart)")
    radius = float(np.sqrt(porosity / np.pi))
    out = {"cells": (2, 2, 1), "planar": True, "radius": radius}
    if p["mesh"] in ("auto", "mirrored"):
        h = float(p["element_size"] or 0.12)
        if not 0 < h <= 0.5:
            raise ValueError("element size: between 0 and 0.5 of the cell")

        def solid(points):
            d = points[:, :2] % 1.0 - 0.5
            return radius - np.hypot(d[:, 0], d[:, 1])

        ctx.progress(0.05, "conforming mesh (MMG), mirrored")
        body = ImplicitBody(Domain(size=(1.0, 1.0, 2 * h), periodic=True), solid)
        mesh = mirrored_mesh(body, size=h, diagonals=((0, 1),), cells=(2, 2, 1), phases="solid")
        out.update(
            sample=mesh,
            phases=None,
            kind="mirrored",
            band=0.5 * h,
            description=f"a conforming mesh of the holes (elements of {h:.3g} of the "
            "cell), mirrored with the diagonals of the square",
        )
        return out
    n = int(p["resolution"] or 12)
    n += n % 2
    if not 4 <= n <= 64:
        raise ValueError("voxels per edge: from 4 to 64")
    c = (np.arange(n) + 0.5) / n - 0.5
    solid = (np.hypot(c[:, None], c[None, :]) > radius).astype(np.uint8)
    data = np.tile(solid[:, :, None], (2, 2, 1))
    out.update(
        sample=VoxelImage(data, spacing=(1.0 / n,) * 3, periodic=True),
        phases=[1],
        kind="voxels",
        band=1.0 / n,
        description=f"voxels of the holes, {n} per edge of the cell",
    )
    return out


def _ellipticities(nodes, u, radius: float, band: float) -> np.ndarray:
    """The ellipticity of each hole of the 2 x 2 cell, (0, 0), (0, 1), (1, 0), (1, 1), under
    a displacement: its radial displacement along x less that along y (the nodes within
    ``band`` of its edge)."""
    out = []
    for hx in (0, 1):
        for hy in (0, 1):
            d = nodes[:, :2] - np.array([hx + 0.5, hy + 0.5])
            rad = np.hypot(d[:, 0], d[:, 1])
            ring = np.abs(rad - radius) < band
            radial = (u[:, 0] * d[:, 0] + u[:, 1] * d[:, 1]) / np.maximum(rad, 1e-12)
            along_x = ring & (np.abs(d[:, 0]) > 0.9 * rad)
            along_y = ring & (np.abs(d[:, 1]) > 0.9 * rad)
            if not (along_x.any() and along_y.any()):
                return np.full(4, np.nan)
            out.append(radial[along_x].mean() - radial[along_y].mean())
    return np.array(out)


@tool(
    "path_bifurcation",
    "stability",
    "Bifurcation along a nonlinear path",
    "The first bifurcation of a periodic cell along a nonlinear loading path (the stability "
    "of the principal path itself, beyond linear buckling): the cell is loaded in finite "
    "strains (Hencky elasticity, or von Mises elastoplasticity), and at each converged step "
    "its consistent tangent, the components of prescribed stress condensed, is "
    "block-diagonalized in the isotypic subspaces of its symmetry group restricted to the "
    "path; the first block whose lowest eigenvalue changes sign marks the bifurcation: its "
    "critical strain, irrep, multiplicity and patterns (the softest modes of the axial "
    "classes there). A square array of circular holes in compression turns its holes into "
    "alternating, mutually orthogonal ellipses (Mullin et al. 2007; Bertoldi et al. 2008), "
    "an irrep that doubles the period; on the bifurcated branch (an imperfection along the "
    "pattern), its stress levels off and its Poisson's ratio turns negative.",
    [
        Field("cell", "choice", "holes", choices=CELL_KINDS),
        Field("source", "item", label="morphology", required=True, when={"cell": "morphology"}),
        Field("porosity", "float", 0.5, when={"cell": "holes"}, doc="area fraction of the holes"),
        Field(
            "primitive",
            "choice",
            "unit",
            choices=CELLS,
            label="primitive cell",
            when={"cell": "morphology"},
        ),
        Field(
            "cells",
            "ints",
            when={"cell": "morphology"},
            doc="primitive cells along x, y and z, 1 or 2 each, or one count for all (blank: "
            "2 × 2 × 1 for a plane morphology, 1 × 1 × 1 otherwise)",
        ),
        Field("mesh", "choice", "voxels", choices=MESHES),
        Field(
            "element_size",
            "float",
            unit="length",
            when={"mesh": ["auto", "mirrored"]},
            doc="of the conforming mesh (blank: 0.12 of the cell of the holes; 1/8 of the "
            "shortest edge of a plane morphology, 1/6 in space)",
        ),
        Field(
            "resolution",
            "int",
            12,
            label="voxels per edge",
            when={"mesh": ["auto", "voxels"]},
            doc="along the shortest edge of the primitive cell (even)",
        ),
        Field("young", "float", 1.0, label="Young's modulus", when={"cell": "holes"}),
        Field("poisson", "float", 0.3, label="Poisson's ratio", when={"cell": "holes"}),
        Field("materials", "materials", of="source", when={"cell": "morphology"}),
        Field("behaviour", "choice", "elastic", choices=BEHAVIOURS),
        Field(
            "yield_strain",
            "float",
            0.01,
            when={"behaviour": "plastic"},
            doc="the yield stress over the Young's modulus of each phase",
        ),
        Field(
            "hardening",
            "float",
            0.1,
            when={"behaviour": "plastic"},
            doc="the linear hardening modulus over the Young's modulus",
        ),
        Field("kinematics", "choice", "finite", choices=KINEMATICS),
        Field(
            "order",
            "choice",
            "1",
            choices=ORDERS,
            label="elements",
            doc="10-node tetrahedra for plasticity (4-node ones lock under plastic flow)",
        ),
        Field("loading", "choice", "uniaxial_stress", choices=PATH_LOADINGS),
        Field(
            "axis",
            "choice",
            "y",
            choices=AXES,
            when={"loading": ["uniaxial_stress", "uniaxial_strain"]},
        ),
        Field(
            "strain",
            "float",
            0.12,
            label="largest strain",
            doc="the nominal compressive strain at the end of the path",
        ),
        Field("increments", "int", 12, doc="steps of the path (divided when Newton fails)"),
        Field(
            "branch",
            "bool",
            False,
            label="follow the bifurcated branch",
            doc="the path followed again with an imperfection of 2 % of the cell along "
            "the critical pattern",
        ),
        Field("video", "bool", True, doc="a video of the critical pattern oscillating"),
    ],
    order=3,
    groups=(
        ("Cell", "cell"),
        ("Mesh", "mesh"),
        ("Material", "young"),
        ("Solver", "kinematics"),
        ("Loading", "loading"),
        ("Output", "video", True),
    ),
    menu="Bifurcations",
)
def path_bifurcation_tool(p: dict, ctx) -> dict:
    from morphogen.homogenization.bifurcation import path_bifurcation
    from morphogen.homogenization.materials import Material

    holes = p["cell"] == "holes"
    if holes:
        materials = {1: Material(young=float(p["young"]), poisson=float(p["poisson"]))}
        cell = _holes(p, ctx)
        title = f"A square array of circular holes (porosity {float(p['porosity']):g})"
    else:
        item = p["source"]
        _periodic(item)
        materials = _materials_of(item, p["materials"])
        cell = _cell_sample(item, {**p, "cell": p["primitive"]}, materials, ctx)
        title = item["title"]
    materials = _behaviour(materials, p)
    path, axis, loading = _path(p, cell["planar"])
    eps = float(p["strain"])
    order = int(p["order"] or 1)
    sample = cell["sample"]
    size = sample.n_nodes if hasattr(sample, "n_nodes") else int(np.prod(sample.shape))
    expected = 3.0 + size * int(p["increments"]) / 2000.0 * (2 if p["branch"] else 1) * order**3
    _check_size(sample, cell["cells"], order, cell["phases"], split=False)
    with _Busy(ctx, 0.1, 0.7, "following the loading path", expected):
        try:
            b = path_bifurcation(
                sample,
                path,
                materials=materials,
                kinematics=p["kinematics"],
                cells=cell["cells"],
                order=order,
                phases=cell["phases"],
                branch=0.02 if p["branch"] else None,
            )
        except ValueError as exc:
            raise _unresolved(exc, cell["kind"]) from None
    dofs = int(b.lift.shape[1])
    values: dict = {
        "mesh": cell["kind"],
        "cells": _cells_text(cell["cells"]),
        "elements": b.mesh.n_elements,
        "dofs": dofs,
        "group order": b.group.order,
        "irreps": len(b.irreps),
        "steps": len(b.times) - 1,
    }
    text = [
        f"{title}: {cell['description']}; {_cells_text(cell['cells'])} primitive cells, "
        f"{b.mesh.n_elements} elements, {dofs} dofs. {loading.capitalize()} to {eps:g} in "
        f"{p['kinematics']} strains: a group of {b.group.order} operations restricted to the "
        f"path, {len(b.irreps)} irreps."
    ]
    critical = []
    waves = _wave_vectors(b, cell["cells"])
    if b.first is None:
        reached = -float(b.strains[-1, axis])
        values["bifurcation"] = "none"
        text.append(f"The principal path stays stable up to a nominal strain of {reached:.4g}.")
    else:
        t, i, count = b.first
        critical = np.flatnonzero(np.abs(b.loads - t) <= 1e-3 * abs(t)).tolist()
        values.update(
            {
                "critical nominal strain": -float(b.critical_strain[axis]),
                **{f"critical strain {NAMES[k]}": float(b.critical_strain[k]) for k in range(3)},
                "irrep of the first bifurcation": i + 1,
                "its dimension": b.irreps[i].dim,
                "multiplicity": count,
            }
        )
        doubled = _doubled(waves[i])
        if doubled:
            values["period doubled along"] = doubled
        sentence = (
            f"First bifurcation at a nominal strain of {-b.critical_strain[axis]:.4g} (the "
            f"strains {', '.join(f'{v:.4g}' for v in b.critical_strain[:3])}), in irrep "
            f"{i + 1} of dimension {b.irreps[i].dim}, multiplicity {count}"
        )
        if doubled and doubled != "none":
            sentence += f": its wave vectors {_waves(waves[i])} double the period along {doubled}"
        text.append(sentence + ".")
    # the loads of a path are the times of the crossings: in nominal strains
    load = "critical nominal strain"
    rows = _irrep_rows(b, critical, waves, load=load, scale=eps)
    ctx.table(rows, "irreps", "irreducible representations")
    if b.patterns:
        ctx.table(
            _pattern_rows(b, load=load, scale=eps),
            "patterns",
            "patterns: the axial isotropy subgroups",
        )
    ctx.progress(0.72, "figures")
    ctx.figure(
        _spectra_figure(b, axis, critical),
        "stability",
        "lowest eigenvalue of the tangent in each isotypic subspace",
    )
    ctx.table(_spectra_rows(b, axis), "spectra", "lowest eigenvalues along the path")
    ctx.figure(_response_figure(b, axis), "response", "response along the path")
    if holes and critical:
        pattern = _patterns_of(b, critical)[0]
        e = _ellipticities(b.mesh.nodes, b.displacement(pattern.mode), cell["radius"], cell["band"])
        if np.all(np.isfinite(e)) and np.abs(e).max() > 0:
            e = e / np.abs(e).max()
            alternating = bool(e[0] * e[1] < 0 and e[0] * e[2] < 0 and e[0] * e[3] > 0)
            values["holes"] = "alternating ellipses" if alternating else "other pattern"
            names = ("(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)")
            ctx.table(
                [{"hole": h, "ellipticity": float(v)} for h, v in zip(names, e, strict=True)],
                "ellipticities",
                "ellipticity of the holes in the pattern (relative)",
            )
            if alternating:
                text.append("The holes turn into alternating, mutually orthogonal ellipses.")
    if b.branch is not None and b.first is not None:
        ctx.figure(_branch_figure(b, axis, eps), "branch", "the bifurcated branch")
        growth = abs(b.amplitudes[-1]) / max(abs(b.amplitudes[0]), 1e-300)
        stress = -b.branch.stress[:, axis]  # compressive
        late = stress[b.branch.times > 1.6 * b.first[0]]
        values["branch: growth of the pattern"] = float(growth)
        sentence = f"On the bifurcated branch the amplitude of the pattern grows {growth:.3g} times"
        if late.size > 2 and abs(late.mean()) > 0:
            spread = float(np.ptp(late) / abs(late.mean()))
            values["branch: stress spread"] = spread
            sentence += (
                f", the stress varying by {100 * spread:.2g} % past 1.6 times the critical strain"
            )
        for k in (k for k in range(3) if NAMES[k] not in path["strain"]):  # the free strains
            values[f"branch: final strain {NAMES[k]}"] = float(b.branch.strain[-1, k])
            sentence += f"; the strain {NAMES[k]} ends at {b.branch.strain[-1, k]:.4g}"
            if b.branch.strain[-1, k] < 0:
                sentence += " (a negative Poisson's ratio)"
        text.append(sentence + ".")
    if p["behaviour"] == "plastic" and order == 1:
        text.append("4-node tetrahedra lock under plastic flow: prefer 10-node ones.")
    if critical:
        _show_patterns(b, ctx, critical, planar=cell["planar"], video=p["video"], start=0.78)
    return {"values": values, "text": " ".join(text)}


def _spectra_figure(b, axis: int, critical):
    """The lowest eigenvalue of the tangent in each isotypic subspace along the path, over
    its value in the natural state: a block that loses its positive definiteness crosses 0."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.6, 4.0), layout="constrained")
    ax = fig.subplots()
    strain = -b.strains[:, axis]
    for i in range(len(b.irreps)):
        lam = b.spectra[:, i, 0]
        if not np.isfinite(lam).any() or not lam[0] > 0:
            continue
        bold = i in critical
        ax.plot(
            strain,
            lam / lam[0],
            lw=2.0 if bold else 0.9,
            color=None if bold else "0.6",
            label=f"irrep {i + 1}" if bold else None,
            zorder=3 if bold else 2,
        )
    ax.axhline(0.0, color="k", lw=0.8)
    if b.critical_strain is not None:
        ax.axvline(-b.critical_strain[axis], color="C3", ls="--", lw=1.0, label="bifurcation")
    ax.set_xlabel(f"nominal compressive strain {NAMES[axis]}")
    ax.set_ylabel("lowest eigenvalue / its natural value")
    ax.set_title("stability of the principal path (irreps in grey)", fontsize=10)
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, frameon=False)
    return fig


def _spectra_rows(b, axis: int) -> list[dict]:
    present = [i for i in range(len(b.irreps)) if np.isfinite(b.spectra[:, i, 0]).any()]
    return [
        {
            "time": float(t),
            "nominal strain": -float(e[axis]),
            **{f"irrep {i + 1}": float(row[i, 0]) for i in present},
        }
        for t, e, row in zip(b.times, b.strains, b.spectra, strict=True)
    ]


def _response_figure(b, axis: int):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    paths = [("principal path", b.principal)]
    if b.branch is not None:
        paths.append(("bifurcated branch", b.branch))
    for label, result in paths:
        ax.plot(-result.strain[:, axis], -result.stress[:, axis], lw=1.6, label=label)
    if b.critical_strain is not None:
        ax.axvline(-b.critical_strain[axis], color="C3", ls="--", lw=1.0, label="bifurcation")
    ax.set_xlabel(f"nominal compressive strain {NAMES[axis]}")
    ax.set_ylabel(f"nominal compressive stress {NAMES[axis]}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    return fig


def _branch_figure(b, axis: int, eps: float):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.4, 3.8), layout="constrained")
    ax = fig.subplots()
    ax.semilogy(b.branch_times * eps, np.abs(b.amplitudes), lw=1.6, color="C1")
    if b.critical_strain is not None:
        ax.axvline(-b.critical_strain[axis], color="C3", ls="--", lw=1.0)
    ax.set_xlabel(f"nominal compressive strain {NAMES[axis]}")
    ax.set_ylabel("amplitude of the pattern")
    ax.set_title("the pattern along the bifurcated branch", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    return fig


# -- Bloch waves ------------------------------------------------------------------------------

BAND_MESHES = (
    ("voxels", "six tetrahedra per voxel of its voxel image"),
    ("conforming", "a conforming mesh of its level set (MMG, matching faces)"),
)


def _corners(text: str | None, planar: bool) -> list[str]:
    """The corners of the Brillouin zone of a path written ``G X M G R``."""
    from morphogen.homogenization.bloch import POINTS

    text = text or ("G X M G" if planar else "G X M G R")
    names = [
        ("G" if c in ("Γ", "g") else c.upper())
        for c in text.replace(",", " ").replace("-", " ").replace("–", " ").split()
    ]
    unknown = [c for c in names if c not in POINTS]
    if unknown or len(names) < 2:
        raise ValueError(f"path: two corners at least among {', '.join(POINTS)}")
    return names


@tool(
    "band_structure",
    "stability",
    "Bloch-wave band structure",
    "The dispersion of the elastic waves of a periodic morphology: the angular frequencies "
    "of its Bloch waves (finite elements of its primitive cell, its consistent mass) along "
    "the edges of the irreducible Brillouin zone (Γ–X–M–Γ–R for a cubic cell), in units of "
    "2π c_T / L (c_T the shear wave speed of the first solid phase, L the edge of the cell "
    "along x), and its band gaps: the ranges between two bands where no wave propagates "
    "along the path (complete band gaps when the path holds the extrema of the bands).",
    [
        Field("source", "item", label="morphology", required=True),
        Field("cell", "choice", "unit", choices=CELLS, label="primitive cell"),
        Field("mesh", "choice", "voxels", choices=BAND_MESHES),
        Field(
            "resolution",
            "int",
            6,
            label="voxels per edge",
            when={"mesh": "voxels"},
            doc="along the shortest edge of the primitive cell",
        ),
        Field(
            "element_size",
            "float",
            unit="length",
            when={"mesh": "conforming"},
            doc="of the conforming mesh (blank: 1/4 of the shortest edge of the cell)",
        ),
        Field("order", "choice", "2", choices=ORDERS, label="elements"),
        Field("materials", "materials", of="source"),
        Field("densities", "floats", doc="of the phases 0, 1, ... (blank: 1 for every phase)"),
        Field(
            "path",
            "str",
            doc="corners of the Brillouin zone among G (Γ), X, Y, Z, M, R (blank: G X M G R; "
            "G X M G in the plane)",
        ),
        Field("points", "int", 3, doc="wave vectors per segment of the path"),
        Field("bands", "int", 8),
    ],
    order=4,
    menu="Waves",
)
def band_structure_tool(p: dict, ctx) -> dict:
    import dataclasses

    from morphogen.domain import Domain, Grid
    from morphogen.homogenization.bloch import (
        BandStructure,
        band_structure,
        brillouin_path,
        voxel_tet_mesh,
    )
    from morphogen.image import VoxelImage
    from morphogen.mesh import conforming_mesh

    item = p["source"]
    _periodic(item)
    micro = item["micro"]
    planar = _planar(micro)
    materials = _materials_of(item, p["materials"])
    phases = sorted(materials)
    densities = p["densities"] or [1.0] * (max(phases) + 1)
    if len(densities) < max(phases) + 1 or any(not r > 0 for r in densities):
        raise ValueError(f"densities: {max(phases) + 1} positive values, one per phase")
    materials = {
        k: m if m.void else dataclasses.replace(m, density=float(densities[k]))
        for k, m in materials.items()
    }
    solid = _solid_phases(materials)
    corners = _corners(p["path"], planar)
    points, bands, order = int(p["points"]), int(p["bands"]), int(p["order"] or 2)
    if not 1 <= points <= 24 or not 1 <= bands <= 40:
        raise ValueError("points: from 1 to 24 per segment; bands: from 1 to 40")
    lower, size, origin = _primitive(item, p["cell"] or "unit")
    edge = float(size[:2].min() if planar else size.min())
    if p["mesh"] == "conforming":
        h = float(p["element_size"] or edge / 4)
        if not 0 < h <= edge:
            raise ValueError("element size: positive, at most the edge of the cell")
        ctx.progress(0.03, "conforming mesh (MMG)")
        body = _CellBody(micro, lower, size, 2 * h if planar else None)
        meshed = "solid" if solid == [1] else "all"
        mesh = conforming_mesh(body, size=h, phases=meshed, periodic=True)
        description = f"a conforming mesh of the level set of {origin} (elements of {h:.3g})"
    else:
        n = int(p["resolution"] or 6)
        if not 2 <= n <= 32:
            raise ValueError("voxels per edge: from 2 to 32")
        h = edge / n
        shape = np.maximum(1, np.rint(size / h)).astype(int)
        if planar:
            shape[2] = 1
        grid = Grid(Domain(size=tuple(size), origin=tuple(lower), periodic=True), tuple(shape))
        data = np.asarray(micro.voxelize(grid=grid).data)
        spacing = (h, h, h) if planar else tuple(grid.spacing)
        mesh = voxel_tet_mesh(
            VoxelImage(data, spacing=spacing, origin=tuple(lower), periodic=True), solid
        )
        description = f"the voxel image of {origin}, {' × '.join(map(str, shape))} voxels"
    ks, distance, labels = brillouin_path(mesh.size, corners, points)
    frequencies = []
    for s in range(len(corners) - 1):  # one segment at a time, for the progress
        ctx.progress(
            0.05 + 0.85 * s / (len(corners) - 1), f"Bloch waves {corners[s]}–{corners[s + 1]}"
        )
        chunk = ks[(0 if s == 0 else 1 + s * points) : 1 + (s + 1) * points]
        try:
            bands_of = band_structure(mesh, materials, chunk, bands=bands, order=order)
        except ValueError as exc:
            raise _unresolved(exc, p["mesh"]) from None
        frequencies.append(bands_of.frequencies)
    result = BandStructure(ks, distance, np.vstack(frequencies), labels)
    reference = materials[solid[0]]
    speed = float(np.sqrt(reference.young / (2 * (1 + reference.poisson)) / reference.density))
    length = float(size[0])
    scale = length / (2 * np.pi * speed)
    ctx.progress(0.92, "figures")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.6, 4.4), layout="constrained")
    ax = fig.subplots()
    result.plot(ax, scale=scale, ylabel="ω L / 2π c_T")
    ax.set_title(f"band structure (c_T of phase {solid[0]})", fontsize=10)
    ctx.figure(fig, "bands", "band structure")
    gaps = result.gaps()
    rows = [
        {
            "bands": f"{n_} – {n_ + 1}",
            "low": low * scale,
            "high": high * scale,
            "width": (high - low) * scale,
            "relative width": 2 * (high - low) / (high + low),
            "low ω": low,
            "high ω": high,
        }
        for n_, low, high in gaps
    ]
    ctx.table(rows, "gaps", "band gaps (ω L / 2π c_T)")
    names = {round(float(d), 12): name for d, name in labels}
    ctx.table(
        [
            {
                "corner": names.get(round(float(d), 12), ""),
                "distance": float(d),
                **{f"k{'xyz'[a]}": float(k[a]) for a in range(3)},
                **{f"band {j + 1}": float(w) * scale for j, w in enumerate(f)},
            }
            for d, k, f in zip(distance, ks, result.frequencies, strict=True)
        ],
        "frequencies",
        "frequencies along the path (ω L / 2π c_T)",
    )
    values: dict = {
        "mesh": "conforming" if p["mesh"] == "conforming" else "voxels",
        "elements": mesh.n_elements,
        "wave vectors": len(ks),
        "bands": bands,
        "band gaps": len(gaps),
        "c_T": speed,
        "L": length,
    }
    text = [
        f"{item['title']}: {description}, {mesh.n_elements} elements of order {order}; "
        f"{len(ks)} wave vectors along {'–'.join('Γ' if c == 'G' else c for c in corners)}, "
        f"{bands} bands."
    ]
    if gaps:
        widest = max(rows, key=lambda r: r["relative width"])
        values["widest gap: bands"] = widest["bands"]
        values["widest gap: relative width"] = widest["relative width"]
        text.append(
            f"{len(gaps)} band gap{'s' if len(gaps) > 1 else ''} along the path; the widest "
            f"between the bands {widest['bands']}, from {widest['low']:.4g} to "
            f"{widest['high']:.4g} (relative width {widest['relative width']:.3g})."
        )
    else:
        text.append("No band gap along the path among these bands.")
    return {"values": values, "text": " ".join(text)}
