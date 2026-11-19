"""Cellular design of laser-architectured sheets: the phases of a grid of cells, and
their optimization by cellular automata and by densities (the loop of the SCOLASTIC
project: a generative module, the finite elements of the load case, a local rule).

**Cells.** The parallel part of a specimen is a grid of square cells
(:class:`CellGrid`, 1 mm by default: the width of a laser track, the increment of the
path, and a volume large enough for each cell to carry a homogenized behaviour). Each
cell holds a :class:`Phase`: a flow law, a critical plastic strain (the fracture of a
brittle phase: the specimen fails when a point of it reaches it) and a cost (its share
of the treatment budget: 0 for the untreated sheet, 1 for a full treatment, between for
the mild and strong levels of a multi-level treatment). The tracks are drawn on the
blank before the specimen is cut: the end columns of the grid continue along x to the
grips (``extend``).

**Load case.** The tensile test of the specimen (:func:`~morphogen.laser.mechanics.tensile_test`):
the engineering curve up to the failure, the first of the maximum force (diffuse
necking) and of the fracture of a brittle point. Measures (:func:`failure_measures`):
the tensile strength reached before the failure, the failure strain (``ductility``),
the work absorbed up to it per unit volume (``energy``, MJ/m³: the plastically
dissipated energy of Wang et al. 2023), and their product with the strength. The
objective is one of them, the constraints bound the others (exponential penalties).

**Sensitivities by replay.** A test records the deformation gradients of the Gauss
points at every step; replaying them with the law of another phase
(:func:`~morphogen.laser.mechanics.replay`) gives the work and the plastic strains each
element would have seen with it, the kinematics held: the first-order change of the
force at every step if one cell alone changed its phase, and whether a brittle phase
there would have broken first. :meth:`CellProblem.estimates` turns them into the value
of the objective each cell would give the design in each phase — the failure included,
which a derivative would miss (the fracture of the reinforcement that limits the
ductility, Wang et al. 2023, is a threshold). The replays of all the cells cost a few
tenths of a second; the next finite element test checks them.

**Cellular automata.**

* :func:`beso` — the bidirectional evolutionary rule (Huang & Xie 2007, 2010; Querin,
  Steven & Xie 1998) on cells: the benefit of each phase over the untreated one,
  filtered over a radius (the neighbourhood of the automaton) and averaged with the
  previous iteration; the budget moves to its target by ``er`` per iteration, each cell
  takes the phase that maximizes its benefit minus ``λ`` times its cost (``λ`` meets the
  budget: with two phases, the cells of largest benefit are treated), at most one level
  per iteration and a share ``max_change`` of the cells changing. Several levels of
  treatment (mild, strong, very strong: SCOLASTIC) are phases of increasing cost.
  ``multigrid``: the cells are first coarse (4 mm, then 2 mm), each solution starting
  the finer grid (the multigrid automata of Abdalla, Kim & Gürdal 2004 and Zakhama et
  al. 2009). ``exchange``: the phases of neighbouring cells exchanged, at a constant
  budget (a Kawasaki rule), the best estimates tested by finite elements.
* :func:`hca` — the hybrid cellular automaton of Tovar et al. (2006): continuous
  treatment densities (the flow stress mixes the phases by the weight ``x^q``, the
  critical strain of the brittle phase grows as ``x^-2`` when the treatment is
  partial), each cell driven by the estimated derivative of the objective (its
  stimulus, averaged over the neighbourhood) towards the set point that meets the
  budget, a proportional controller of gain ``kp``; the densities then rounded to the
  levels.
* :func:`topology` — density-based topology optimization (Bendsøe 1989; SIMP): the
  same densities filtered (Bruns & Tortorelli 2001) and projected (Wang, Lazarov &
  Sigmund 2011), the derivatives from the replays, the method of moving asymptotes
  (:class:`~morphogen.optimization.mma.MMA`) with the budget as constraint.

**Bands and tracks.** :func:`band_fractions` gives the share of a band of width ``t``
along polylines (straight or corrugated tracks of constant width: the ideal designs of
Wang et al. 2023) in the region of each Gauss point, for :meth:`CellProblem.evaluate`.
:meth:`CellGrid.stamp` draws polylines with a square spot of some cells,
:func:`treatable` tells whether such a spot draws a design (the minimum width of the
automata), :func:`design_tracks` gives the centre lines of the tracks that draw one (a
path for :class:`~morphogen.laser.design.SheetCase`, the process model).

What the replays miss: they hold the kinematics, so that the strain a change moves onto
its neighbours (a gap that necks, a corner that concentrates) escapes them — the
finite element tests of the automata keep only the true gains, and the derivatives of
:func:`hca` and :func:`topology` see the strength a cell adds, not the fracture it
causes next to it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import sparse

from morphogen.laser.materials import FlowLaw
from morphogen.laser.mechanics import Membrane, TensileResult, replay
from morphogen.laser.patterns import distance_to_polyline
from morphogen.laser.specimen import QuadMesh, Specimen
from morphogen.optimization.filters import heaviside, heaviside_derivative

#: Measures of a design (:func:`failure_measures`), the objectives among them.
OBJECTIVES = ("energy", "strength_ductility", "ductility", "tensile_strength")
MEASURES = (*OBJECTIVES, "yield_strength", "uniform_elongation", "failure_strain", "budget")
#: Symmetries of the designs: mirrors about the middle of the length (``x``), of the
#: width (``y``), and the half turn (``point``).
SYMMETRIES = ("x", "y", "point")
#: The critical plastic strain of the laser martensite of Wang et al. (2023), fitted to
#: the failure of their straight reinforcement (sample 1, at an engineering strain of
#: 0.083) with the laws of :func:`wang_phases`, averaged over :data:`FRACTURE_RADIUS`.
MARTENSITE_CRITICAL = 0.072
#: The radius (mm) over which the plastic strain of a brittle phase is averaged
#: (:class:`~morphogen.laser.mechanics.Fracture`): about the width of a track, the
#: crack crossing it whole; with it the failure strains of the corrugated samples of
#: Wang et al. (2023) follow their tests.
FRACTURE_RADIUS = 1.5


@dataclass(frozen=True)
class Phase:
    """A material state of the cells (see the module): ``law``, ``critical`` plastic
    strain (infinite: ductile) and ``cost`` in the treatment budget."""

    name: str
    law: FlowLaw
    critical: float = math.inf
    cost: float = 1.0

    @classmethod
    def from_config(cls, data) -> Phase:
        if isinstance(data, Phase):
            return data
        d = dict(data)
        law = d.pop("law")
        if not isinstance(law, FlowLaw):
            law = dict(law)
            if "k" in law:  # Hollomon from the yield stress
                law = FlowLaw.hollomon(law["yield_strength"], law["k"], law["n"])
            else:
                law = FlowLaw(**law)
        critical = d.pop("critical", math.inf)
        return cls(law=law, critical=math.inf if critical is None else float(critical), **d)


def wang_phases(critical: float = MARTENSITE_CRITICAL) -> tuple[Phase, Phase]:
    """The phases of Wang et al. (2023, Table 3): the annealed 0.23 C steel (ferrite and
    pearlite, Hollomon σ0 280 MPa, K 986 MPa, n 0.24) and its laser martensite (1300,
    1966, 0.06), brittle beyond ``critical``."""
    return (
        Phase("ferrite", FlowLaw.hollomon(280.0, 986.0, 0.24), cost=0.0),
        Phase("martensite", FlowLaw.hollomon(1300.0, 1966.0, 0.06), critical=critical),
    )


#: Rules of the critical plastic strain of a reinforcement softer than the laser
#: martensite (:func:`contrast_phases`).
FRACTURE_RULES = ("graded", "brittle", "ductile")


def contrast_phases(
    contrast: float, fracture: str = "graded", critical: float = MARTENSITE_CRITICAL
) -> tuple[Phase, Phase]:
    """The matrix of :func:`wang_phases` and a reinforcement whose yield stress is
    ``contrast`` times the matrix's, from 1 to the contrast ``c_M = 1300/280`` of the
    as-quenched laser martensite (the hardest state of the steel): the Hollomon
    parameters ``σ0``, ``K`` and ``n`` interpolated linearly between those of the matrix
    and of the martensite at the share ``a = (contrast − 1)/(c_M − 1)``, the states
    between the annealed steel and its martensite (partly transformed by an intercritical
    peak, or tempered), a harder state hardening less. The critical plastic strain of
    the reinforcement (:data:`FRACTURE_RULES`): ``graded``, ``critical / a²`` (the rule
    of a partial treatment, as in :func:`hca`: the martensite breaks at ``critical``, a
    softer state later); ``brittle``, ``critical`` whatever the contrast; ``ductile``,
    none (the necking alone). The last two bound the first."""
    ferrite, martensite = wang_phases(critical)
    lo, hi = ferrite.law, martensite.law
    top = hi.sigma0 / lo.sigma0
    c = float(contrast)
    if not 1.0 <= c <= top:
        raise ValueError(f"contrast: from 1 to {top:.4g} (the laser martensite)")
    if fracture not in FRACTURE_RULES:
        raise ValueError(f"fracture: one of {FRACTURE_RULES}")
    a = (c - 1.0) / (top - 1.0)
    law = FlowLaw.hollomon(
        *((1.0 - a) * x + a * y for x, y in ((lo.sigma0, hi.sigma0), (lo.a, hi.a), (lo.b, hi.b)))
    )
    if fracture == "ductile":
        crit = math.inf
    elif fracture == "brittle":
        crit = float(critical)
    else:
        crit = float(critical) / a**2 if a > 0 else math.inf
    return ferrite, Phase("reinforcement", law, critical=crit)


# -- the cells -------------------------------------------------------------------------


@dataclass
class CellGrid:
    """Square cells of about ``size`` mm over ``[x0, x0 + length] x [y0, y0 + width]``
    (the parallel part of a specimen, the region of a structure): ``ny`` rows (along y)
    and ``nx`` columns; a design is an array ``(ny, nx)``."""

    length: float
    width: float
    size: float = 1.0
    x0: float = 0.0
    y0: float = 0.0

    def __post_init__(self):
        if not (self.length > 0 and self.width > 0 and self.size > 0):
            raise ValueError("cell grid: positive length, width and size")
        self.nx = max(1, round(self.length / self.size))
        self.ny = max(1, round(self.width / self.size))
        self.dx = self.length / self.nx
        self.dy = self.width / self.ny

    @classmethod
    def of(cls, specimen: Specimen, size: float = 1.0) -> CellGrid:
        return cls(specimen.length, specimen.width, size)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def count(self) -> int:
        return self.nx * self.ny

    def centres(self) -> np.ndarray:
        """The centres of the cells ``(ny, nx, 2)``."""
        x = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        y = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        X, Y = np.meshgrid(x, y)
        return np.stack([X, Y], axis=-1)

    def locate(self, points, extend: bool = True) -> np.ndarray:
        """The cell (flat index) of each point, -1 outside the grid; ``extend``: the end
        columns continue along x beyond the parallel part."""
        p = np.asarray(points, dtype=float)
        x, y = p[..., 0] - self.x0, p[..., 1] - self.y0
        xc = np.clip(x, 0.0, self.length * (1 - 1e-12)) if extend else x
        j = np.floor(xc / self.dx).astype(int)
        i = np.floor(y / self.dy).astype(int)
        ok = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)
        return np.where(ok, i * self.nx + np.clip(j, 0, self.nx - 1), -1)

    def filter_matrix(self, radius: float) -> sparse.csr_matrix:
        """The neighbourhood of the automaton: weights ``max(0, radius - d)`` between
        cell centres (mm), rows normalized; the identity for ``radius <= size``."""
        if radius <= min(self.dx, self.dy):
            return sparse.identity(self.count, format="csr")
        reach_x = math.ceil(radius / self.dx)
        reach_y = math.ceil(radius / self.dy)
        rows, cols, vals = [], [], []
        idx = np.arange(self.count).reshape(self.shape)
        for di in range(-reach_y, reach_y + 1):
            for dj in range(-reach_x, reach_x + 1):
                d = math.hypot(di * self.dy, dj * self.dx)
                if d >= radius:
                    continue
                a = idx[max(0, -di) : self.ny - max(0, di), max(0, -dj) : self.nx - max(0, dj)]
                b = idx[max(0, di) : self.ny + min(0, di), max(0, dj) : self.nx + min(0, dj)]
                rows.append(a.ravel())
                cols.append(b.ravel())
                vals.append(np.full(a.size, radius - d))
        m = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.count, self.count),
        )
        s = np.asarray(m.sum(axis=1)).ravel()
        return sparse.diags(1.0 / s) @ m

    def orbits(self, symmetry=(), period: float | None = None) -> np.ndarray:
        """The orbit of each cell under the ``symmetry`` (:data:`SYMMETRIES`) and the
        translation by ``period`` (mm, a divisor of the length in cells): a design
        constant on the orbits is symmetric and periodic."""
        sym = set(symmetry or ())
        bad = sym - set(SYMMETRIES)
        if bad:
            raise ValueError(f"symmetry: among {SYMMETRIES}")
        if len(sym) >= 2:
            sym = set(SYMMETRIES)
        idx = np.arange(self.count).reshape(self.shape)
        images = []
        if "x" in sym:
            images.append(idx[:, ::-1])
        if "y" in sym:
            images.append(idx[::-1, :])
        if "point" in sym:
            images.append(idx[::-1, ::-1])
        if period:
            m = round(period / self.dx)
            if m < 1 or self.nx % m:
                raise ValueError("period: a divisor of the length, in cells")
            images.append(np.roll(idx, -m, axis=1))
        rep = idx.ravel().copy()
        while True:  # the least index of each orbit
            new = rep.copy()
            for img in images:
                new = np.minimum(new, new[img.ravel()])
            if np.array_equal(new, rep):
                break
            rep = new
        return np.unique(rep, return_inverse=True)[1]

    def transfer(self, design, other: CellGrid) -> np.ndarray:
        """A design of this grid sampled at the cell centres of ``other`` (multigrid)."""
        d = np.asarray(design)
        cells = self.locate(other.centres())
        return d.ravel()[cells].reshape(other.shape)

    def coverage(self, polylines, width: float, n: int = 6) -> np.ndarray:
        """The share of each cell ``(ny, nx)`` within ``width / 2`` of the polylines."""
        t = (np.arange(n) + 0.5) / n
        x = self.x0 + (np.arange(self.nx)[:, None] + t[None, :]).ravel() * self.dx
        y = self.y0 + (np.arange(self.ny)[:, None] + t[None, :]).ravel() * self.dy
        X, Y = np.meshgrid(x, y)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        inside = np.zeros(len(pts), dtype=bool)
        for poly in polylines:
            inside |= distance_to_polyline(pts, np.asarray(poly, dtype=float)) <= 0.5 * width
        return inside.reshape(self.ny, n, self.nx, n).mean(axis=(1, 3))

    def rasterize(self, polylines, width: float, budget: float | None = None) -> np.ndarray:
        """The cells of a band of ``width`` along the polylines (phase 1): those most
        covered, as many as the ``budget`` asks (default: those covered at least by
        half)."""
        cov = self.coverage(polylines, width).ravel()
        n = int((cov >= 0.5).sum()) if budget is None else round(budget * self.count)
        out = np.zeros(self.count, dtype=int)
        out[np.argsort(-cov, kind="stable")[:n]] = 1
        return out.reshape(self.shape)

    def stamp(self, polylines, width: int = 2) -> np.ndarray:
        """The cells a square spot of ``width`` cells covers along the polylines (phase
        1): a design :func:`treatable` by construction, the tracks of a laser spot."""
        out = np.zeros(self.shape, dtype=int)
        step = 0.25 * min(self.dx, self.dy)
        for poly in polylines:
            p = np.asarray(poly, dtype=float)
            seg = np.diff(p, axis=0)
            length = np.hypot(seg[:, 0], seg[:, 1])
            s = np.concatenate([[0.0], np.cumsum(length)])
            t = np.arange(0.0, s[-1] + step, step)
            x = np.interp(t, s, p[:, 0])
            y = np.interp(t, s, p[:, 1])
            i0 = np.rint((y - self.y0) / self.dy - 0.5 * width).astype(int)
            j0 = np.rint((x - self.x0) / self.dx - 0.5 * width).astype(int)
            for a, b in set(zip(i0.tolist(), j0.tolist(), strict=True)):
                out[max(a, 0) : max(a + width, 0), max(b, 0) : max(b + width, 0)] = 1
        return out

    def band(self, budget: float, rows: int | None = None) -> np.ndarray:
        """The straight design: the central rows treated (phase 1), as many as the
        ``budget`` (fraction of the cells) asks, rounded."""
        n = round(budget * self.ny) if rows is None else int(rows)
        d = np.zeros(self.shape, dtype=int)
        lo = (self.ny - n) // 2
        d[lo : lo + n] = 1
        return d


def quarter_points(mesh: QuadMesh, n: int = 4) -> np.ndarray:
    """Points of the quarter of each element around each of its Gauss points: ``(e, 4,
    n², 2)``, regular sub-grids of the quarters of the reference square."""
    xe = mesh.nodes[mesh.quads]
    t = (np.arange(n) + 0.5) / n
    out = np.empty((len(mesh.quads), 4, n * n, 2))
    for g, (sr, ss) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        r, s = np.meshgrid(sr * t, ss * t)
        r, s = r.ravel(), s.ravel()
        shape = 0.25 * np.stack(
            [(1 - r) * (1 - s), (1 + r) * (1 - s), (1 + r) * (1 + s), (1 - r) * (1 + s)], axis=1
        )
        out[:, g] = np.einsum("ka,eai->eki", shape, xe)
    return out


def band_fractions(mesh: QuadMesh, polylines, width: float, n: int = 4) -> np.ndarray:
    """The share ``(e, 4)`` of the region of each Gauss point within ``width / 2`` of
    the polylines (tracks of constant width)."""
    pts = quarter_points(mesh, n)
    flat = pts.reshape(-1, 2)
    inside = np.zeros(len(flat), dtype=bool)
    for poly in polylines:
        inside |= distance_to_polyline(flat, np.asarray(poly, dtype=float)) <= 0.5 * width
    return inside.reshape(pts.shape[:3]).mean(axis=2)


def corrugation(
    specimen: Specimen,
    amplitude: float,
    period: float | None,
    *,
    count: int = 1,
    margin: float = 10.0,
    resolution: int = 64,
):
    """The centre lines of ``count`` corrugated tracks ``y = yc + H sin(2π x / P)``
    (``period`` None: straight), evenly spread over the width, through the grips."""
    x0, x1 = specimen.extent()
    x = np.linspace(x0 - margin, x1 + margin, resolution * 8 + 1)
    w = specimen.width
    lines = []
    for k in range(count):
        yc = (k + 0.5) * w / count
        y = np.full_like(x, yc) if not period else yc + amplitude * np.sin(2 * math.pi * x / period)
        lines.append(np.column_stack([x, y]))
    return lines


def constant_area_width(width: float, amplitude: float, period: float | None) -> float:
    """The width of a sinusoidal track of the same area as a straight one of ``width``
    (Wang 2022: the spot shrunk by defocusing, ``t = t0 P / L``, ``L`` the arc length of
    a period)."""
    if not period or amplitude == 0:
        return width
    u = np.linspace(0.0, 2 * math.pi, 4001)
    k = 2 * math.pi * amplitude / period
    arc = np.trapezoid(np.sqrt(1.0 + (k * np.cos(u)) ** 2), u) / (2 * math.pi) * period
    return width * period / arc


def design_tracks(
    grid: CellGrid, design, width: int = 2, margin: float = 5.0, cross: bool = False
) -> list:
    """The centre lines (polylines, mm) of the tracks that draw a design: the runs of
    treated cells of each column, linked to the runs they overlap in the next column (the
    straightest continuation first where a track splits), crossed at their midpoint —
    or, with ``cross``, from end to end when longer than a spot of ``width`` cells (a
    square spot then covers every cell; a real track, wider than the spot, would treat
    too much) —; the tracks that reach the ends of the grid continue along x by
    ``margin`` (the blank is treated before the specimen is cut)."""
    d = np.asarray(design) > 0
    runs = []  # (column, first row, end row)
    by_col = []
    for j in range(grid.nx):
        col = np.concatenate([[False], d[:, j], [False]])
        starts = np.flatnonzero(~col[:-1] & col[1:])
        ends = np.flatnonzero(col[:-1] & ~col[1:])
        ids = []
        for a, b in zip(starts, ends, strict=True):
            ids.append(len(runs))
            runs.append((j, int(a), int(b)))
        by_col.append(ids)
    adj = {i: set() for i in range(len(runs))}
    for j in range(grid.nx - 1):
        for u in by_col[j]:
            for v in by_col[j + 1]:
                _, a, b = runs[u]
                _, c, e = runs[v]
                if min(b, e) > max(a, c):  # the runs share a row
                    adj[u].add(v)
                    adj[v].add(u)
    xy = np.array(
        [(grid.x0 + (j + 0.5) * grid.dx, grid.y0 + 0.5 * (a + b) * grid.dy) for j, a, b in runs]
    )
    used = set()

    def edge(u, v):
        return (min(u, v), max(u, v))

    def walk(start):
        path = [start]
        cur, prev = start, None
        while True:
            options = [v for v in adj[cur] if edge(cur, v) not in used]
            if not options:
                return path
            if prev is None:
                nxt = min(
                    options, key=lambda v: (xy[v, 0] < xy[cur, 0], abs(xy[v, 1] - xy[cur, 1]))
                )
            else:
                ahead = xy[cur] - xy[prev]

                def turn(v, ahead=ahead, cur=cur):
                    dv = xy[v] - xy[cur]
                    return -float(ahead @ dv) / (np.linalg.norm(ahead) * np.linalg.norm(dv) + 1e-12)

                nxt = min(options, key=turn)
            used.add(edge(cur, nxt))
            path.append(nxt)
            prev, cur = cur, nxt

    tracks = []
    order = sorted(range(len(runs)), key=lambda i: (len(adj[i]) % 2 == 0, xy[i, 0], xy[i, 1]))
    for i in order:
        while any(edge(i, v) not in used for v in adj[i]):
            tracks.append(walk(i))
        if not adj[i]:
            tracks.append([i])
    out = []
    half = 0.5 * width * grid.dy
    for t in tracks:
        pts = []
        for n, i in enumerate(t):
            j, a, b = runs[i]
            x = grid.x0 + (j + 0.5) * grid.dx
            lo, hi = grid.y0 + a * grid.dy + half, grid.y0 + b * grid.dy - half
            if not cross or hi - lo <= 1e-9:  # the midpoint of the run
                pts.append((x, grid.y0 + 0.5 * (a + b) * grid.dy))
                continue
            # a longer run: the track crosses it, from the side of the previous run
            ref = xy[t[n - 1], 1] if n > 0 else (xy[t[n + 1], 1] if len(t) > 1 else lo)
            ends = [(x, lo), (x, hi)]
            if abs(ref - hi) < abs(ref - lo):
                ends.reverse()
            if n == 0 and len(t) > 1:
                ends.reverse()
            pts += ends
        p = np.array(pts, dtype=float)
        if len(p) == 1:
            p = np.vstack([p[0] - [0.25 * grid.dx, 0.0], p[0] + [0.25 * grid.dx, 0.0]])
        if p[0, 0] > p[-1, 0]:
            p = p[::-1]
        x0, x1 = grid.x0, grid.x0 + grid.length
        if margin and p[0, 0] <= x0 + 0.5 * grid.dx + 1e-9:  # the ends continue to the grips
            p = np.vstack([[x0 - margin, p[0, 1]], p])
        if margin and p[-1, 0] >= x1 - 0.5 * grid.dx - 1e-9:
            p = np.vstack([p, [x1 + margin, p[-1, 1]]])
        out.append(p)
    return out


# -- the measures ----------------------------------------------------------------------


def failure_measures(strain, stress, failure, *, young: float = 210000.0, cap=None) -> dict:
    """Measures of engineering curves ``stress`` (steps x m, or steps) on the common
    ``strain`` with the failure strains ``failure`` (m, infinite: none): the maximum
    (diffuse necking: the uniform elongation, a parabola through the points around the
    largest), the failure at the first of the necking and the fracture
    (``failure_strain``, alias ``ductility``), the strength reached before it, the work
    up to it (MJ/m³) and the 0.2 % proof stress. ``cap``: the strength of the weakest
    cross-section (m): a curve that reaches it necks there. Arrays of length m (or
    floats)."""
    e = np.asarray(strain, dtype=float)
    s = np.asarray(stress, dtype=float)
    single = s.ndim == 1
    if single:
        s = s[:, None]
    m = s.shape[1]
    fail = np.broadcast_to(np.asarray(failure, dtype=float), (m,))
    if cap is not None:
        capv = np.broadcast_to(np.asarray(cap, dtype=float), (m,))
        over = s >= capv[None, :]
        over[0] = False
        hit = over.any(axis=0)
        if hit.any():
            kc = np.argmax(over, axis=0)
            cols_ = np.arange(m)
            s0, s1 = s[kc - 1, cols_], s[kc, cols_]
            f = np.clip((capv - s0) / np.where(s1 > s0, s1 - s0, 1.0), 0.0, 1.0)
            ecap = np.where(hit, e[kc - 1] + f * (e[kc] - e[kc - 1]), np.inf)
            fail = np.minimum(fail, ecap)
            s = np.minimum(s, capv[None, :])
    ns = len(e)
    cols = np.arange(m)
    k = np.argmax(s, axis=0)
    eu, su = e[k].copy(), s[k, cols].copy()
    inner = (k > 0) & (k < ns - 1)
    if inner.any():
        kk = k[inner]
        x0, x1, x2 = e[kk - 1], e[kk], e[kk + 1]
        y0, y1, y2 = s[kk - 1, cols[inner]], s[kk, cols[inner]], s[kk + 1, cols[inner]]
        d0, d2 = x0 - x1, x2 - x1
        den = d0 * d2 * (d0 - d2)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = ((y0 - y1) * d2 - (y2 - y1) * d0) / den
            b = ((y2 - y1) * d0 * d0 - (y0 - y1) * d2 * d2) / den
            xm = -b / (2 * a)
        ok = (a < 0) & (xm >= d0) & (xm <= d2) & np.isfinite(xm)
        sel = np.flatnonzero(inner)[ok]
        eu[sel] = x1[ok] + xm[ok]
        su[sel] = y1[ok] + a[ok] * xm[ok] ** 2 + b[ok] * xm[ok]
    ef = np.minimum(eu, fail)
    ef = np.minimum(ef, e[-1])
    # the curve at the failure strain, the work and the strength up to it
    cum = np.vstack([np.zeros((1, m)), np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(e)[:, None], 0)])
    i = np.clip(np.searchsorted(e, ef, side="right") - 1, 0, ns - 2)
    span = e[i + 1] - e[i]
    t = np.where(span > 0, (ef - e[i]) / np.where(span > 0, span, 1.0), 0.0)
    sf = s[i, cols] + t * (s[i + 1, cols] - s[i, cols])
    energy = cum[i, cols] + 0.5 * (s[i, cols] + sf) * (ef - e[i])
    before = np.where(e[:, None] <= ef[None, :], s, -np.inf)
    strength = np.maximum(before.max(axis=0), sf)
    strength = np.where(ef >= eu - 1e-15, np.maximum(strength, su), strength)
    # the proof stress
    line = s - young * (e[:, None] - 0.002)
    below = line <= 0
    below[0] = False
    has = below.any(axis=0)
    j = np.where(has, np.argmax(below, axis=0), 1)
    l0, l1 = line[j - 1, cols], line[j, cols]
    f = np.where(l0 != l1, l0 / np.where(l0 != l1, l0 - l1, 1.0), 0.0)
    ys = np.where(has, s[j - 1, cols] + f * (s[j, cols] - s[j - 1, cols]), np.nan)
    out = {
        "energy": energy,
        "ductility": ef,
        "failure_strain": ef,
        "tensile_strength": strength,
        "strength_ductility": strength * ef,
        "uniform_elongation": eu,
        "yield_strength": ys,
    }
    if single:
        return {k_: float(v[0]) for k_, v in out.items()}
    return out


def _crossing(strain, ratio) -> np.ndarray:
    """The strain at which each column of ``ratio`` (steps x n, non-decreasing) first
    reaches 1 (infinite: never)."""
    hit = ratio >= 1.0
    has = hit.any(axis=0)
    k = np.where(has, np.argmax(hit, axis=0), 1)
    k = np.maximum(k, 1)
    cols = np.arange(ratio.shape[1])
    r0, r1 = ratio[k - 1, cols], ratio[k, cols]
    f = np.clip((1.0 - r0) / np.maximum(r1 - r0, 1e-12), 0.0, 1.0)
    e = strain[k - 1] + f * (strain[k] - strain[k - 1])
    return np.where(has, e, np.inf)


def phase_strength(phase: Phase, young: float = 210000.0) -> float:
    """The engineering tensile strength of a phase alone (MPa): the largest
    ``σ(p) exp(−ε)`` (``ε = p + σ/E``) up to its necking or its critical strain."""
    p = np.linspace(0.0, min(phase.critical, 1.5), 3001)
    sig = phase.law.stress(p)
    return float(np.max(sig * np.exp(-(p + sig / young))))


# -- the problem -----------------------------------------------------------------------


@dataclass
class CellEvaluation:
    """A design and its tensile test: ``measures`` (:func:`failure_measures`, the
    ``budget`` used), ``value`` (the objective with the penalties of the constraints)."""

    design: np.ndarray
    result: TensileResult
    measures: dict
    value: float
    membrane: Membrane | None = field(default=None, repr=False)
    fractions: np.ndarray | None = field(default=None, repr=False)


@dataclass
class CellProblem:
    """The load case of a cellular design (see the module): a ``specimen`` (its parallel
    part is the grid of cells of ``size`` mm), the ``phases`` (phase 0: the untreated
    sheet), the mesh (``element`` mm, a divisor of the cell size), the ``objective``
    (:data:`OBJECTIVES`, maximized) and the ``constraints`` ``{measure: [low, high]}``
    (penalty ``exp(-penalty * relative violation)`` on the objective)."""

    specimen: Specimen = field(default_factory=Specimen)
    phases: tuple = field(default_factory=wang_phases)
    size: float = 1.0
    element: float = 0.5
    objective: str = "energy"
    constraints: dict = field(default_factory=dict)
    penalty: float = 10.0
    extend: bool = True
    young: float = 210000.0
    poisson: float = 0.3
    max_strain: float = 0.35
    failure_margin: float = 0.03
    fracture_radius: float | None = FRACTURE_RADIUS
    layers: int = 1
    case: object = None

    def __post_init__(self):
        self.specimen = Specimen.from_config(self.specimen)
        self.phases = tuple(Phase.from_config(p) for p in self.phases)
        if len(self.phases) < 2:
            raise ValueError("cell problem: at least two phases")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective: one of {OBJECTIVES}")
        for name in self.constraints:
            if name not in MEASURES:
                raise ValueError(f"constraint on {name!r}: one of {MEASURES}")
        from morphogen.laser.structures import TensionCase, case_from_config

        self.case = case_from_config(self.case) or TensionCase(self.specimen, self.element)
        x0, x1, y0, y1 = self.case.region
        self.grid = CellGrid(x1 - x0, y1 - y0, self.size, x0=x0, y0=y0)
        self.mesh = self.case.mesh
        centroid = self.mesh.nodes[self.mesh.quads].mean(axis=1)
        self.element_cell = self.grid.locate(centroid, extend=self.extend and self.case.extend)
        inside = self.element_cell >= 0
        self.cell_matrix = sparse.csr_matrix(
            (np.ones(int(inside.sum())), (self.element_cell[inside], np.flatnonzero(inside))),
            shape=(self.grid.count, len(self.mesh.quads)),
        )
        #: the cells that hold elements: the others (a hole, the outside) never change
        self.active = np.asarray(self.cell_matrix.sum(axis=1)).ravel() > 0
        self.costs = np.array([p.cost for p in self.phases], dtype=float)
        self._laws = np.array(
            [[p.law.code, p.law.sigma0, p.law.a, p.law.b] for p in self.phases], dtype=float
        )
        self._critical = np.array([p.critical for p in self.phases], dtype=float)
        self.strengths = np.array([phase_strength(p, self.young) for p in self.phases])
        self._weights = self._same = None
        if self.fracture_radius:
            from scipy.spatial import cKDTree

            pairs = cKDTree(centroid).query_pairs(self.fracture_radius, output_type="ndarray")
            d = np.linalg.norm(centroid[pairs[:, 0]] - centroid[pairs[:, 1]], axis=1)
            n = len(centroid)
            rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
            cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
            r = self.fracture_radius
            vals = np.concatenate([r - d, r - d, np.full(n, r)])
            self._weights = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
            ec = self.element_cell
            keep = (ec[rows] == ec[cols]) & (ec[rows] >= 0)
            self._same = sparse.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n))

    # -- designs to membranes ----------------------------------------------------------

    def fractions(self, design) -> np.ndarray:
        """The shares ``(e, 4, phases)`` of the phases at the Gauss points of a design
        (phase indices per cell, or for two phases the densities of the second)."""
        d = np.asarray(design)
        ne = len(self.mesh.quads)
        frac = np.zeros((ne, 4, len(self.phases)))
        cell = self.element_cell
        inside = cell >= 0
        if np.issubdtype(d.dtype, np.integer):
            ph = np.zeros(ne, dtype=int)
            ph[inside] = d.ravel()[cell[inside]]
            frac[np.arange(ne), :, ph] = 1.0
        else:
            if len(self.phases) != 2:
                raise ValueError("densities: two phases")
            x = np.zeros(ne)
            x[inside] = d.ravel()[cell[inside]]
            frac[:, :, 1] = x[:, None]
            frac[:, :, 0] = 1.0 - x[:, None]
        return frac

    def _arrays(self, frac, q: float = 1.0, graded: bool = False):
        """The flow laws ``(e, 4, layers, 9)`` and the critical strains of the shares
        ``frac`` of the phases at the Gauss points: the two largest shares mix their flow
        stresses (by the weight ``w^q`` of the second); the critical strain is that of
        the largest share, or (``graded``: the densities of a partial treatment, two
        phases) ``critical / w²`` of the brittle phase at the share ``w``."""
        order = np.argsort(-frac, axis=2, kind="stable")
        i = order[..., 0]
        j = order[..., 1]
        fi = np.take_along_axis(frac, i[..., None], 2)[..., 0]
        fj = np.take_along_axis(frac, j[..., None], 2)[..., 0]
        tot = fi + fj
        w = np.where(tot > 0, fj / np.where(tot > 0, tot, 1.0), 0.0) ** q
        law = np.concatenate([self._laws[i], self._laws[j], w[..., None]], axis=-1)
        c = self._critical
        if graded and len(c) == 2 and np.isfinite(c).sum() == 1:
            b = int(np.argmin(c))
            sb = frac[..., b]
            crit = np.where(sb > 0, c[b] / np.maximum(sb, 1e-12) ** 2, np.inf)
        else:
            crit = c[i]
        nl = self.layers
        law = np.repeat(law[:, :, None, :], nl, axis=2)
        crit = np.repeat(crit[:, :, None], nl, axis=2)
        return np.ascontiguousarray(law), np.ascontiguousarray(crit)

    def membrane(self, frac, q: float = 1.0, graded: bool = False) -> Membrane:
        law, crit = self._arrays(frac, q, graded)
        flow = [law[..., k] for k in range(9)]
        return Membrane(
            self.mesh,
            flow,
            young=self.young,
            poisson=self.poisson,
            thickness=self.case.thickness,
            critical=crit,
            fracture_radius=self.fracture_radius,
            springs=self.case.springs,
        )

    def budget(self, design) -> float:
        """The treatment budget a design uses: the mean cost of its active cells (those
        that hold elements)."""
        d = np.asarray(design).ravel()[self.active]
        if np.issubdtype(d.dtype, np.integer):
            return float(self.costs[d].mean())
        return float(self.costs[0] + (self.costs[1] - self.costs[0]) * d.mean())

    # -- evaluation --------------------------------------------------------------------

    def _slope(self, res) -> float:
        """The elastic slope of the nominal curve: Young's modulus in tension, the first
        steps' secant for a structure."""
        if self.case.kind == "tension":
            return self.young
        e, s = np.asarray(res.strain), np.asarray(res.stress)
        j = max(1, min(2, len(e) - 1))
        return float(s[j] / e[j]) if len(e) > 1 and e[j] > 0 else self.young

    def value_of(self, measures) -> np.ndarray | float:
        """The objective with the penalties of the constraints."""
        v = np.asarray(measures[self.objective], dtype=float)
        excess = np.zeros_like(v)
        for name, (lo, hi) in self.constraints.items():
            m = np.asarray(measures[name], dtype=float)
            if lo is not None:
                excess = excess + np.maximum(0.0, (lo - m) / abs(lo))
            if hi is not None:
                excess = excess + np.maximum(0.0, (m - hi) / abs(hi))
        out = v * np.exp(-self.penalty * excess)
        return float(out) if out.ndim == 0 else out

    def evaluate(
        self, design=None, *, fractions=None, record: bool = False, q: float = 1.0
    ) -> CellEvaluation:
        """The tensile test of a design (or of the ``fractions`` of the phases at the
        Gauss points, :func:`band_fractions`); ``record``: keep the history for the
        replays."""
        frac = self.fractions(design) if fractions is None else np.asarray(fractions, float)
        graded = design is not None and not np.issubdtype(np.asarray(design).dtype, np.integer)
        mem = self.membrane(frac, q, graded)
        res = self.case.run(
            mem,
            max_strain=self.max_strain if self.case.kind == "tension" else None,
            failure_margin=self.failure_margin,
            record_history=record,
            keep_fields=True,
        )
        fail = res.failure_strain if math.isfinite(res.failure_strain) else math.inf
        meas = failure_measures(res.strain, res.stress, fail, young=self._slope(res))
        if design is not None:
            meas["budget"] = self.budget(design)
        else:
            vol = self.mesh.gauss()[2]
            gauge = self.element_cell >= 0
            share = (frac[gauge] * vol[gauge][..., None]).sum(axis=(0, 1)) / vol[gauge].sum()
            meas["budget"] = float(share @ self.costs)
        value = self.value_of(meas)
        return CellEvaluation(
            None if design is None else np.asarray(design).copy(),
            res,
            meas,
            float(value),
            mem if record else None,
            frac,
        )

    # -- sensitivities -----------------------------------------------------------------

    def _columns(self, ev: CellEvaluation):
        """The strength of each cell of a design (MPa, the mixture of its phases) and of
        each cross-section of cells (N), for the weakest-section cap of the estimates."""
        frac = ev.fractions
        inside = self.element_cell >= 0
        vol = self.mesh.gauss()[2]
        cell_frac = np.zeros((self.grid.count, len(self.phases)))
        np.add.at(
            cell_frac, self.element_cell[inside], (frac[inside] * vol[inside][..., None]).sum(1)
        )
        tot = cell_frac.sum(axis=1, keepdims=True)
        cell_frac = cell_frac / np.where(tot > 0, tot, 1.0)
        cell_strength = cell_frac @ self.strengths
        area = self.grid.dy * self.case.thickness
        cols = (cell_strength.reshape(self.grid.shape) * area).sum(axis=0)
        return cell_strength, cols

    def _cap(self, columns, strength_alt) -> np.ndarray:
        """The strength (MPa) of the weakest cross-section if each cell alone took the
        strength ``strength_alt`` (cells)."""
        cell_strength, cols = columns
        area = self.grid.dy * self.case.thickness
        col = np.arange(self.grid.count) % self.grid.nx
        new_col = cols[col] + (strength_alt - cell_strength) * area
        order = np.argsort(cols)
        m1 = cols[order[0]]
        m2 = cols[order[1]] if len(cols) > 1 else np.inf
        others = np.where(col == order[0], m2, m1)
        return np.minimum(new_col, others) / self.case.section

    def _effects(self, ev: CellEvaluation, base, law_alt, crit_alt):
        """The change of the work of each cell (steps x cells, mJ) and the strain at
        which it would break (cells) had it alone taken the laws ``law_alt`` and the
        critical strains ``crit_alt`` of its elements (the replays, see the module)."""
        mem = ev.membrane
        alt = replay(mem, ev.result.history, law=law_alt, critical=crit_alt)
        dW = (self.cell_matrix @ (alt.work - base["work"]).T).T
        if self.fracture_radius:
            # the average over the neighbours: those of the same cell switched too
            from morphogen.laser.mechanics import Fracture

            fx = Fracture(mem.centroids, mem.w0, crit_alt, None)
            bvol = fx.vol.sum(axis=(1, 2))
            pe = fx.element_strain(alt.plastic)  # steps x e
            same = self._same
            num = base["num"] - base["num_same"] + (same @ (pe * bvol).T).T
            den = base["den"] - base["den_same"] + same @ bvol
            avg = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
            brittle = bvol > 0
            ratio = np.where(brittle, avg / np.where(brittle, fx.element_critical, 1.0), 0.0)
        else:
            ratio = alt.ratio
        cross_alt = _crossing(base["strain"], ratio)
        cell_alt = np.full(self.grid.count, np.inf)
        inside = self.element_cell >= 0
        np.minimum.at(cell_alt, self.element_cell[inside], cross_alt[inside])
        return dW, cell_alt

    def _combine(self, ev: CellEvaluation, base, dW, fail, cap) -> np.ndarray:
        """The values of designs whose work differs from the current one by ``dW``
        (steps x m), breaking at ``fail`` (m) and whose weakest section holds ``cap``
        (MPa, m): the force changed by the derivative of the work with respect to the
        grip displacement."""
        strain, disp = base["strain"], base["displacement"]
        dd = np.diff(disp)
        mid = np.diff(dW, axis=0) / np.where(dd > 0, dd, np.inf)[:, None]
        dF = np.zeros_like(dW)
        dF[1:-1] = 0.5 * (mid[:-1] + mid[1:])
        dF[-1] = mid[-1]
        s_new = base["stress"][:, None] + dF / self.case.section
        meas = failure_measures(strain, s_new, fail, young=base["young"], cap=cap)
        meas["budget"] = np.full(dW.shape[1], ev.measures.get("budget", 0.0))
        return np.asarray(self.value_of(meas), dtype=float)

    def _switch(self, ev: CellEvaluation, base, law_alt, crit_alt, strength_alt) -> np.ndarray:
        """The value of the design if each cell alone took the laws ``law_alt`` (and
        critical strains ``crit_alt``) of its elements, estimated from the replays (see
        the module): the force changed by the work of its elements, its elements broken
        at the strain where their own (averaged) plastic strain reaches the critical
        one, the others as in the design, the weakest cross-section of cells capping
        the strength."""
        dW, cell_alt = self._effects(ev, base, law_alt, crit_alt)
        fail = np.minimum(base["others"], cell_alt)
        cap = None if base["columns"] is None else self._cap(base["columns"], strength_alt)
        return self._combine(ev, base, dW, fail, cap)

    def exchanges(self, ev: CellEvaluation, reach: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
        """The estimated gain of exchanging the phases of two neighbouring cells of
        different phases (centres within ``reach`` cells; two phases): the pairs
        ``(c, c')`` (c treated, c' not) and their gains — the moves of a track that
        bends or shifts without breaking, at a constant budget (a Kawasaki rule)."""
        if len(self.phases) != 2:
            raise ValueError("exchanges: two phases")
        d = np.asarray(ev.design).ravel()
        base = self._base(ev)
        ne = len(self.mesh.quads)
        effects = []
        for k in range(2):
            frac = np.zeros((ne, 4, 2))
            frac[:, :, k] = 1.0
            law, crit = self._arrays(frac)
            effects.append(self._effects(ev, base, law, crit))
        g = self.grid
        idx = np.arange(g.count).reshape(g.shape)
        pairs = []
        r = math.floor(reach)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if (di, dj) == (0, 0) or math.hypot(di, dj) > reach:
                    continue
                a = idx[max(0, -di) : g.ny - max(0, di), max(0, -dj) : g.nx - max(0, dj)]
                b = idx[max(0, di) : g.ny + min(0, di), max(0, dj) : g.nx + min(0, dj)]
                a, b = a.ravel(), b.ravel()
                keep = (d[a] == 1) & (d[b] == 0)
                pairs.append(np.column_stack([a[keep], b[keep]]))
        pairs = np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=int)
        if not len(pairs):
            return pairs, np.zeros(0)
        c, cp = pairs[:, 0], pairs[:, 1]
        dW = effects[0][0][:, c] + effects[1][0][:, cp]
        # the failure: the cells but c as in the design, c' as the treated phase
        cell_cur = base["cells"]
        order = np.argsort(cell_cur)
        m1, m2 = cell_cur[order[0]], cell_cur[order[1]]
        others = np.where(c == order[0], m2, m1)
        others = np.minimum(others, base["fixed"])
        fail = np.minimum(others, effects[1][1][cp])
        cap = None
        if base["columns"] is not None:  # the weakest section: the columns of c and c'
            cols = base["columns"][1]
            area = g.dy * self.case.thickness
            col_c, col_p = c % g.nx, cp % g.nx
            delta = (self.strengths[1] - self.strengths[0]) * area
            new = np.repeat(cols[None, :], len(pairs), axis=0)
            rows = np.arange(len(pairs))
            new[rows, col_c] -= delta
            new[rows, col_p] += delta
            cap = new.min(axis=1) / self.case.section
        values = self._combine(ev, base, dW, fail, cap)
        return pairs, values - base["value"]

    def _base(self, ev: CellEvaluation) -> dict:
        res = ev.result
        if ev.membrane is None or not res.history:
            raise ValueError("estimates: evaluate the design with record=True")
        cur = replay(ev.membrane, res.history)
        strain = cur.strain
        cross = _crossing(strain, cur.ratio)
        inside = self.element_cell >= 0
        cell_cur = np.full(self.grid.count, np.inf)
        np.minimum.at(cell_cur, self.element_cell[inside], cross[inside])
        fixed = float(cross[~inside].min(initial=np.inf))
        order = np.argsort(cell_cur)
        m1 = cell_cur[order[0]]
        m2 = cell_cur[order[1]] if len(order) > 1 else np.inf
        others = np.full(self.grid.count, m1)
        others[order[0]] = m2
        others = np.minimum(others, fixed)
        stress = np.interp(strain, res.strain, res.stress)
        base = {
            "strain": strain,
            "displacement": cur.displacement,
            "work": cur.work,
            "stress": stress,
            "others": others,
            "cells": cell_cur,
            "fixed": fixed,
            "columns": self._columns(ev) if self.case.columns else None,
            "young": self._slope(res),
        }
        if self.fracture_radius:
            fx = ev.membrane.fracture
            bvol = fx.vol.sum(axis=(1, 2))
            bp = (fx.element_strain(cur.plastic) * bvol).T  # e x steps
            W, same = self._weights, self._same
            base["num"] = (W @ bp).T
            base["num_same"] = (same @ bp).T
            base["den"] = W @ bvol
            base["den_same"] = same @ bvol
        meas = failure_measures(
            strain, stress[:, None], np.array([min(m1, fixed)]), young=base["young"]
        )
        meas["budget"] = np.array([ev.measures.get("budget", 0.0)])
        base["value"] = float(np.asarray(self.value_of(meas))[0])
        return base

    def estimates(self, ev: CellEvaluation) -> np.ndarray:
        """The value of the design if each cell alone took each phase ``(cells,
        phases)`` (its current phase: the design's value), from the replays."""
        base = self._base(ev)
        ne = len(self.mesh.quads)
        out = np.empty((self.grid.count, len(self.phases)))
        for k, _phase in enumerate(self.phases):
            frac = np.zeros((ne, 4, len(self.phases)))
            frac[:, :, k] = 1.0
            law, crit = self._arrays(frac)
            out[:, k] = self._switch(
                ev, base, law, crit, np.full(self.grid.count, self.strengths[k])
            )
        if ev.design is not None and np.issubdtype(np.asarray(ev.design).dtype, np.integer):
            d = np.asarray(ev.design).ravel()
            out[np.arange(self.grid.count), d] = base["value"]
        return out - base["value"]

    def phase_effects(self, ev: CellEvaluation, base=None) -> list:
        """For each phase, the change of the work of each cell and the strain at which
        it would break had it alone taken that phase (:meth:`_effects`)."""
        base = self._base(ev) if base is None else base
        ne = len(self.mesh.quads)
        out = []
        for k in range(len(self.phases)):
            frac = np.zeros((ne, 4, len(self.phases)))
            frac[:, :, k] = 1.0
            law, crit = self._arrays(frac)
            out.append(self._effects(ev, base, law, crit))
        return out

    def move_values(self, ev: CellEvaluation, base, effects, moves) -> np.ndarray:
        """The estimated gain of each move, a set of cells and their new phases
        ``(cells, phases)``, all changed together (first order: their works add up,
        the first of their fractures and of the others' counts, the cross-sections
        change)."""
        if not moves:
            return np.zeros(0)
        g = self.grid
        columns = base["columns"] is not None
        if columns:
            cell_strength, cols = base["columns"]
        area = g.dy * self.case.thickness
        order = np.argsort(base["cells"])
        ns = len(base["strain"])
        dW = np.zeros((ns, len(moves)))
        fail = np.empty(len(moves))
        cap = np.empty(len(moves))
        for m, (cells, phases) in enumerate(moves):
            cells = np.asarray(cells, dtype=int)
            phases = np.asarray(phases, dtype=int)
            inside = set(cells.tolist())
            other = next((base["cells"][c] for c in order if c not in inside), np.inf)
            f = min(other, base["fixed"])
            new_cols = cols.copy() if columns else None
            for c, k in zip(cells, phases, strict=True):
                dW[:, m] += effects[k][0][:, c]
                f = min(f, effects[k][1][c])
                if columns:
                    new_cols[c % g.nx] += (self.strengths[k] - cell_strength[c]) * area
            fail[m] = f
            cap[m] = new_cols.min() / self.case.section if columns else np.inf
        return self._combine(ev, base, dW, fail, cap if columns else None) - base["value"]

    def derivatives(self, ev: CellEvaluation, density, q: float = 1.0, step: float = 0.05):
        """The derivative of the value with respect to the density of each cell (two
        phases), by replays of the densities ``x ± step`` (one-sided at the bounds)."""
        base = self._base(ev)
        x = np.asarray(density, dtype=float).ravel()
        up = np.minimum(x + step, 1.0)
        lo = np.maximum(x - step, 0.0)
        vals = []
        for xx in (up, lo):
            law, crit = self._arrays(self.fractions(xx.reshape(self.grid.shape)), q, True)
            strength = self.strengths[0] + (self.strengths[1] - self.strengths[0]) * xx**q
            vals.append(self._switch(ev, base, law, crit, strength))
        span = np.maximum(up - lo, 1e-12)
        return (vals[0] - vals[1]) / span


# -- cellular automata -----------------------------------------------------------------


@dataclass
class CellOptimization:
    """The result of an automaton: the best ``design`` and its ``evaluation``, the
    ``history`` of the iterations (value, measures, budget, time) and their designs."""

    design: np.ndarray
    evaluation: CellEvaluation
    history: list = field(default_factory=list)
    designs: list = field(default_factory=list, repr=False)
    time_s: float = 0.0
    method: str = ""
    evaluations: int = 0
    density: np.ndarray | None = field(default=None, repr=False)

    def summary(self) -> dict:
        return {"method": self.method, "value": self.evaluation.value, **self.evaluation.measures}


def initial_design(grid: CellGrid, budget: float, kind: str = "band", seed: int = 0):
    """A starting design of two phases: the central ``band`` of rows, ``random`` cells,
    or (``uniform``) the densities equal to the budget."""
    if kind == "band":
        return grid.band(budget)
    if kind == "random":
        rng = np.random.default_rng(seed)
        d = np.zeros(grid.count, dtype=int)
        d[rng.permutation(grid.count)[: round(budget * grid.count)]] = 1
        return d.reshape(grid.shape)
    if kind == "uniform":
        return np.full(grid.shape, float(budget))
    raise ValueError("initial design: band, random or uniform")


def _assign(values, costs, current, target, orbit_size, step, max_change):
    """The phases of the orbits that maximize ``Σ values − λ cost`` at the ``target``
    cost, within ``step`` levels of the current phases and a share ``max_change`` of
    changed cells (a hysteresis bonus to the current phases)."""
    n_orb, n_ph = values.shape
    levels = np.arange(n_ph)
    allowed = np.abs(levels[None, :] - current[:, None]) <= step
    cost = costs[None, :] * orbit_size[:, None]
    span = np.abs(values).max() + 1e-30
    keep_bonus = np.zeros_like(values)
    keep_bonus[np.arange(n_orb), current] = 1.0

    def choose(lam, bonus):
        score = values + bonus * span * keep_bonus - lam * cost
        score = np.where(allowed, score, -np.inf)
        return np.argmax(score, axis=1)

    def solve(bonus):
        steps = np.diff(np.unique(costs))
        reach = 100.0 * span / max(steps.min() if steps.size else 1.0, 1e-12)
        lo, hi = -reach - 1.0, reach + 1.0
        for _ in range(80):
            lam = 0.5 * (lo + hi)
            used = (costs[choose(lam, bonus)] * orbit_size).sum()
            if used > target:
                lo = lam
            else:
                hi = lam
        c_lo, c_hi = choose(lo, bonus), choose(hi, bonus)
        u_lo = (costs[c_lo] * orbit_size).sum()
        u_hi = (costs[c_hi] * orbit_size).sum()
        return c_lo if abs(u_lo - target) < abs(u_hi - target) else c_hi

    total = orbit_size.sum()
    pick = solve(0.0)
    if max_change is None or (orbit_size * (pick != current)).sum() <= max_change * total:
        return pick
    lo, hi = 0.0, 2.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        trial = solve(mid)
        if (orbit_size * (trial != current)).sum() > max_change * total:
            lo = mid
        else:
            hi = mid
            pick = trial
    return pick


def _neighbour_pairs(grid: CellGrid, orbit, design, reach: float, active=None):
    """The pairs of orbits ``(a, b)`` of the same size, ``a`` treated and ``b`` not,
    with (active) cells within ``reach`` cells of each other."""
    d = np.asarray(design).ravel()
    act = np.ones(grid.count, dtype=bool) if active is None else np.asarray(active)
    idx = np.arange(grid.count).reshape(grid.shape)
    size = np.bincount(orbit)
    found = set()
    r = math.floor(reach)
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if (di, dj) == (0, 0) or math.hypot(di, dj) > reach:
                continue
            a = idx[max(0, -di) : grid.ny - max(0, di), max(0, -dj) : grid.nx - max(0, dj)]
            b = idx[max(0, di) : grid.ny + min(0, di), max(0, dj) : grid.nx + min(0, dj)]
            a, b = a.ravel(), b.ravel()
            keep = (d[a] == 1) & (d[b] == 0) & act[a] & act[b]
            for oa, ob in zip(orbit[a[keep]], orbit[b[keep]], strict=True):
                if oa != ob and size[oa] == size[ob]:
                    found.add((int(oa), int(ob)))
    return sorted(found)


def symmetrize(design, orbit) -> np.ndarray:
    """A design made constant on the orbits (the phase most of their cells hold)."""
    d = np.asarray(design)
    flat = d.ravel()
    n = orbit.max() + 1
    out = np.empty_like(flat)
    for o in range(n):
        vals = flat[orbit == o]
        out[orbit == o] = np.bincount(vals).argmax()
    return out.reshape(d.shape)


def treatable(design, width: int = 2) -> bool:
    """Whether a design (two phases) can be drawn by a spot of ``width`` cells: every
    treated cell lies in a treated square of ``width`` cells (the design equals its
    morphological opening), and the squares join by sides or corners wherever the
    treated cells do (no neck thinner than the spot)."""
    from scipy import ndimage

    d = np.asarray(design) > 0
    if width <= 1 or not d.any():
        return True
    # a spot may overhang the edges: the specimen is cut after the treatment
    d = np.pad(d, width, mode="edge")
    st = np.ones((width, width), dtype=bool)
    if not np.array_equal(ndimage.binary_opening(d, structure=st), d):
        return False
    eroded = ndimage.binary_erosion(d, structure=st, origin=(-(width // 2),) * 2)
    n_d = ndimage.label(d)[1]
    n_e = ndimage.label(eroded, structure=np.ones((3, 3), dtype=bool))[1]
    return n_e == n_d


def make_treatable(design, width: int = 2) -> np.ndarray:
    """The largest part of a design a spot of ``width`` cells draws (its opening)."""
    from scipy import ndimage

    d = np.asarray(design) > 0
    if width <= 1:
        return d.astype(int)
    st = np.ones((width, width), dtype=bool)
    pad = np.pad(d, width, mode="edge")
    return ndimage.binary_opening(pad, structure=st)[width:-width, width:-width].astype(int)


def beso(
    problem: CellProblem,
    design=None,
    *,
    budget: float | None = 0.2,
    er: float = 0.02,
    radius: float = 1.5,
    iterations: int = 40,
    step: int = 1,
    max_change: float | None = 0.1,
    averaging: bool = True,
    symmetry=(),
    period: float | None = None,
    tol: float = 1e-3,
    patience: int = 6,
    verify: bool = True,
    tries: int = 3,
    tabu_length: int = 3,
    moves: str = "swap",
    reach: float = 2.0,
    candidates: int = 4,
    min_width: int | None = None,
    multigrid=None,
    callback=None,
) -> CellOptimization:
    """The bidirectional evolutionary automaton on the cells (see the module), from a
    ``design`` (default: the central band of the ``budget``) towards the ``budget``
    (mean cost of the cells); the cells change by orbits of the ``symmetry`` and of
    the translation by ``period`` (the failure of a periodic pattern moves to the next
    period if one alone changes). ``moves``:

    * ``swap`` — the BESO rule: the estimated benefit of each phase of each orbit over
      the untreated sheet, filtered over ``radius`` (mm) and averaged with the
      previous iteration; each orbit takes the phase of largest benefit minus ``λ``
      times its cost (``λ`` meets the budget, which moves by ``er`` per iteration), at
      most ``step`` levels and a share ``max_change`` of the cells changing.
      ``verify``: at the budget, the new design replaces the current one only if its
      finite element test does not lower the value (the replays hold the kinematics:
      a gap in a track that necks, a corner where the strain concentrates escape
      them); refused, the changes are halved, ``tries`` times, then their orbits keep
      their phase for ``tabu_length`` iterations.
    * ``exchange`` (two phases, at the budget) — the exchanges of the phases of two
      orbits with neighbouring cells (within ``reach`` cells), a track that bends or
      shifts without breaking (a Kawasaki rule): the ``candidates`` of largest
      estimated gains are tested by finite elements, the best taken if it raises the
      value, the others tabu.

    ``min_width`` (cells, two phases): only designs a spot of this width draws are
    tried (:func:`treatable`): without it the automaton breaks the tracks into islands
    and corner contacts that carry no load and cannot be made.

    Stops when the value stalls for ``patience`` iterations or nothing changes.
    ``multigrid``: the coarser cell sizes (mm) to solve first, e.g. ``(2, 1)``."""
    t0 = time.perf_counter()
    if multigrid:
        hist, designs, start, grid_prev, evals = [], [], design, None, 0
        for size in [*sorted((float(v) for v in multigrid), reverse=True), problem.size]:
            sub = problem if size == problem.size else replace(problem, size=size)
            if start is not None and grid_prev is not None:
                start = grid_prev.transfer(start, sub.grid)
            r = beso(
                sub,
                start,
                budget=budget,
                er=er,
                radius=max(radius, 1.5 * size),
                iterations=iterations,
                step=step,
                max_change=max_change,
                averaging=averaging,
                symmetry=symmetry,
                period=period,
                tol=tol,
                patience=patience,
                verify=verify,
                tries=tries,
                tabu_length=tabu_length,
                moves=moves,
                reach=reach,
                candidates=candidates,
                min_width=min_width,
                callback=callback,
            )
            hist += [{**h, "size": size} for h in r.history]
            designs += r.designs
            evals += r.evaluations
            start, grid_prev = r.design, sub.grid
        r.history, r.designs, r.evaluations = hist, designs, evals
        r.time_s = time.perf_counter() - t0
        r.method = f"beso-{moves}-multigrid"
        return r
    if moves not in ("swap", "exchange"):
        raise ValueError("moves: swap or exchange")
    grid = problem.grid
    orbit = grid.orbits(symmetry, period)
    n_orb = orbit.max() + 1
    # the sizes of the orbits in active cells: the budget counts them only
    size = np.bincount(orbit, weights=problem.active.astype(float), minlength=n_orb)
    n_cells = float(problem.active.sum())
    x = initial_design(grid, budget) if design is None else np.asarray(design).astype(int)
    if x.shape != grid.shape:
        raise ValueError(f"design of shape {grid.shape}")
    x = symmetrize(x, orbit)
    if budget is None:
        budget = problem.budget(x)
    H = grid.filter_matrix(radius)
    costs = problem.costs
    history, designs = [], []
    evaluations = 0

    def at_budget(b):
        return abs(b - budget) <= 0.5 * size.max() / n_cells + 1e-12

    def log(it, e, accepted):
        history.append(
            {
                "iteration": it,
                "evaluation": evaluations,
                "accepted": accepted,
                "value": e.value,
                **{k: e.measures[k] for k in MEASURES if k in e.measures},
                "time_s": time.perf_counter() - t0,
            }
        )
        designs.append(np.asarray(e.design).copy())
        if callback is not None:
            callback(it, e)

    ev = problem.evaluate(x, record=True)
    evaluations += 1
    log(0, ev, True)
    best = ev if at_budget(problem.budget(x)) else None
    prev = None
    tabu = {}
    stall = 0
    for it in range(1, iterations + 1):
        cur_budget = problem.budget(x)
        base = problem._base(ev)
        effects = problem.phase_effects(ev, base)
        cur = np.zeros(n_orb, dtype=int)
        cur[orbit] = x.ravel()
        members = [np.flatnonzero(orbit == o) for o in range(n_orb)]
        tabu = {k: v - 1 for k, v in tabu.items() if v > 1}
        if moves == "exchange" and at_budget(cur_budget):
            if len(problem.phases) != 2:
                raise ValueError("exchange moves: two phases")
            pairs = [
                (a, b)
                for a, b in _neighbour_pairs(grid, orbit, x, reach, problem.active)
                if ("x", a, b) not in tabu
            ]
            mv = [
                (
                    np.concatenate([members[a], members[b]]),
                    np.concatenate([np.zeros(len(members[a]), int), np.ones(len(members[b]), int)]),
                )
                for a, b in pairs
            ]
            if min_width:
                keep = []
                for i, (cells, phases) in enumerate(mv):
                    xn = x.ravel().copy()
                    xn[cells] = phases
                    if treatable(xn.reshape(grid.shape), min_width):
                        keep.append(i)
                pairs = [pairs[i] for i in keep]
                mv = [mv[i] for i in keep]
            gains = problem.move_values(ev, base, effects, mv)
            order = np.argsort(-gains)[:candidates]
            trials = []
            for i in order:
                xn = x.ravel().copy()
                cells, phases = mv[i]
                xn[cells] = phases
                xn = xn.reshape(grid.shape)
                evn = problem.evaluate(xn, record=True)
                evaluations += 1
                log(it, evn, False)
                trials.append((evn.value, i, xn, evn))
            if not trials:
                break
            value, i, xn, evn = max(trials, key=lambda t: t[0])
            if value > ev.value * (1.0 + 1e-9):
                history[-len(trials) + [t[1] for t in trials].index(i)]["accepted"] = True
                gain = value - ev.value
                x, ev = xn, evn
                if best is None or ev.value > best.value:
                    best = ev
                stall = stall + 1 if gain <= tol * abs(ev.value) else 0
            else:
                for _v, j, _x, _e in trials:
                    tabu[("x", *pairs[j])] = tabu_length
                stall += 1
            if stall >= patience:
                break
            continue
        # the BESO rule on the orbits: the benefit of each phase over the untreated sheet
        mv = [
            (members[o], np.full(len(members[o]), k))
            for o in range(n_orb)
            for k in range(len(costs))
        ]
        gains = problem.move_values(ev, base, effects, mv).reshape(n_orb, len(costs))
        gains[np.arange(n_orb), cur] = 0.0
        rel = gains - gains[:, [0]]
        field_ = (rel / np.where(size > 0, size, 1.0)[:, None])[orbit]  # per cell
        field_ = H @ (field_ * problem.active[:, None])
        vals = np.zeros((n_orb, len(costs)))
        np.add.at(vals, orbit, field_)
        if averaging and prev is not None:
            vals = 0.5 * (vals + prev)
        prev = vals
        frozen = np.array([("s", o) in tabu for o in range(n_orb)])
        frozen |= ~np.bincount(orbit, weights=problem.active, minlength=n_orb).astype(bool)
        vals_t = vals.copy()
        if frozen.any():  # recently refused changes: the orbits keep their phase
            vals_t[frozen] -= 1e6 * (np.abs(vals).max() + 1.0)
            vals_t[frozen, cur[frozen]] = vals[frozen, cur[frozen]]
        target = cur_budget + np.clip(budget - cur_budget, -er, er)
        limit = max_change
        accepted, proposed, changed = False, False, None
        for _attempt in range(tries if verify else 1):
            pick = _assign(vals_t, costs, cur, target * n_cells, size, step, limit)
            changed = pick != cur
            if not changed.any():
                break
            proposed = True
            xn = pick[orbit].reshape(grid.shape)
            if min_width and len(costs) == 2 and not treatable(xn, min_width):
                limit = 0.5 * (limit if limit is not None else (size * changed).sum() / n_cells)
                if limit * n_cells < size[size > 0].min():
                    break
                continue
            evn = problem.evaluate(xn, record=True)
            evaluations += 1
            moving = not (at_budget(cur_budget) and at_budget(problem.budget(xn)))
            ok = not verify or moving or evn.value >= ev.value * (1.0 - 1e-9)
            log(it, evn, ok)
            if ok:
                accepted = True
                break
            limit = 0.5 * (limit if limit is not None else (size * changed).sum() / n_cells)
            if limit * n_cells < size[size > 0].min():
                break
        if accepted:
            gain = evn.value - ev.value
            x, ev = xn, evn
            if at_budget(problem.budget(x)) and (best is None or ev.value > best.value):
                best = ev
            stall = stall + 1 if abs(gain) <= tol * abs(ev.value) else 0
        else:
            if changed is not None:
                for o in np.flatnonzero(changed):
                    tabu[("s", int(o))] = tabu_length
            stall += 1
        if not proposed and not frozen.any():
            break  # no orbit wants to change
        if stall >= patience:
            break
    best = best or ev
    best.membrane = None
    out = CellOptimization(
        best.design, best, history, designs, time.perf_counter() - t0, f"beso-{moves}"
    )
    out.evaluations = evaluations
    return out


def hca(
    problem: CellProblem,
    density=None,
    *,
    budget: float = 0.24,
    kp: float = 0.25,
    radius: float = 1.5,
    q: float = 2.0,
    iterations: int = 30,
    move: float = 0.2,
    levels: int | None = None,
    tol: float = 2e-3,
    callback=None,
) -> CellOptimization:
    """The hybrid cellular automaton (see the module) on the treatment densities of two
    phases: each cell's stimulus, the derivative of the value (from the replays)
    averaged over its neighbourhood, drives its density by ``kp`` (relative to the
    largest stimulus, at most ``move``) towards the set point that meets the
    ``budget``; finally rounded to ``levels`` (default: the two phases, at the
    budget)."""
    t0 = time.perf_counter()
    grid = problem.grid
    x = initial_design(grid, budget, "uniform") if density is None else np.asarray(density, float)
    H = grid.filter_matrix(radius)
    history, designs = [], []
    best = None
    for it in range(iterations):
        ev = problem.evaluate(x, record=True, q=q)
        history.append(
            {
                "iteration": it,
                "value": ev.value,
                "budget": problem.budget(x),
                **{k: ev.measures[k] for k in MEASURES if k in ev.measures},
                "time_s": time.perf_counter() - t0,
            }
        )
        designs.append(x.copy())
        if callback is not None:
            callback(it, ev)
        stim = H @ problem.derivatives(ev, x, q=q)
        scale = np.abs(stim).max() + 1e-30
        xr = x.ravel()

        def update(setpoint, xr=xr, stim=stim, scale=scale):
            dx = np.clip(kp * (stim - setpoint) / scale, -move, move)
            return np.clip(xr + dx, 0.0, 1.0)

        lo, hi = stim.min() - scale / kp, stim.max() + scale / kp
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if update(mid).mean() > budget:
                lo = mid
            else:
                hi = mid
        xn = update(0.5 * (lo + hi))
        change = np.abs(xn - xr).max()
        x = xn.reshape(grid.shape)
        if change < tol:
            break
    final = round_design(x, budget, levels)
    ev = problem.evaluate(final)
    history.append(
        {
            "iteration": len(history),
            "value": ev.value,
            "budget": problem.budget(final),
            **{k: ev.measures[k] for k in MEASURES if k in ev.measures},
            "time_s": time.perf_counter() - t0,
        }
    )
    best = ev
    out = CellOptimization(final, best, history, designs, time.perf_counter() - t0, "hca")
    out.density = x
    return out


def round_design(density, budget: float | None = None, levels: int | None = None):
    """Densities rounded to two phases (the largest at the ``budget``: the share of
    cells treated) or to ``levels`` + 1 equally spaced levels (phase indices)."""
    x = np.asarray(density, dtype=float)
    if levels:
        return np.rint(x * levels).astype(int)
    n = round((x.mean() if budget is None else budget) * x.size)
    out = np.zeros(x.size, dtype=int)
    out[np.argsort(-x.ravel(), kind="stable")[:n]] = 1
    return out.reshape(x.shape)


def topology(
    problem: CellProblem,
    density=None,
    *,
    budget: float = 0.24,
    radius: float = 1.5,
    q: float = 2.0,
    beta: float = 1.0,
    beta_max: float = 16.0,
    iterations: int = 40,
    move: float = 0.15,
    tol: float = 2e-3,
    callback=None,
) -> CellOptimization:
    """Density-based topology optimization (see the module): the densities filtered
    over ``radius`` and projected (``beta`` doubled every 10 iterations up to
    ``beta_max``), the mixture ``w^q``, the derivatives of the replays, MMA under the
    budget constraint; the result rounded to two phases at the budget."""
    from morphogen.optimization.mma import MMA

    t0 = time.perf_counter()
    grid = problem.grid
    n = grid.count
    H = grid.filter_matrix(radius)
    x = np.full(n, float(budget)) if density is None else np.asarray(density, float).ravel()
    opt = MMA(n, 2, 0.0, 1.0, move=move)
    history, designs = [], []
    scale = None
    for it in range(iterations):
        xt = H @ x
        xp = heaviside(xt, beta) if beta > 0 else xt
        ev = problem.evaluate(xp.reshape(grid.shape), record=True, q=q)
        history.append(
            {
                "iteration": it,
                "value": ev.value,
                "budget": float(xp.mean()),
                "beta": beta,
                **{k: ev.measures[k] for k in MEASURES if k in ev.measures},
                "time_s": time.perf_counter() - t0,
            }
        )
        designs.append(xp.reshape(grid.shape).copy())
        if callback is not None:
            callback(it, ev)
        g = problem.derivatives(ev, xp, q=q)
        dxp = heaviside_derivative(xt, beta) if beta > 0 else np.ones_like(xt)
        dJ = H.T @ (g * dxp)
        if scale is None:
            scale = abs(ev.value) + 1e-30
        f0 = -ev.value / scale
        df0 = -dJ / scale
        vol = xp.mean() - budget  # the budget as an equality: two inequalities
        dvol = H.T @ (dxp / n)
        xn = opt.update(x, f0, df0, np.array([vol, -vol]), np.vstack([dvol, -dvol]))
        change = np.abs(xn - x).max()
        x = xn
        if (it + 1) % 10 == 0 and beta < beta_max:
            beta = min(2 * beta, beta_max)
            opt.reset()
        elif change < tol and beta >= beta_max:
            break
    xp = heaviside(H @ x, beta) if beta > 0 else H @ x
    final = round_design(xp.reshape(grid.shape), budget)
    ev = problem.evaluate(final)
    history.append(
        {
            "iteration": len(history),
            "value": ev.value,
            "budget": problem.budget(final),
            **{k: ev.measures[k] for k in MEASURES if k in ev.measures},
            "time_s": time.perf_counter() - t0,
        }
    )
    out = CellOptimization(final, ev, history, designs, time.perf_counter() - t0, "topology")
    out.density = xp.reshape(grid.shape)
    return out
