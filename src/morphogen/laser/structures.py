"""Load cases of laser-architectured sheets beyond the tensile test of a dog-bone: the
compact tension specimen (the blunting of a crack, the third application of SCOLASTIC)
and the L-shaped plate (the strength at a re-entrant corner), and the tensile test
itself as one of them.

**Meshes.** Structured blocks of quadrilaterals graded towards the singular points
(:func:`graded`: elements of ``fine`` mm there, growing geometrically to ``element``
mm): the CT specimen of ASTM E399 proportions (width ``W`` from the load line, height
1.2 W, pin holes of diameter 0.25 W at 0.275 W from the crack plane, a crack of length
``a`` from the load line, its faces unconnected nodes; each pin a node joined by stiff
bars to the half of its hole it bears on, so that the specimen turns about it: the
elastic load-line compliance within 2.5 % of Saxena & Hudak 1978), the L-shaped plate
(a square of side ``size`` without the quadrant beyond the ``arm``).

**Test** (:func:`load_test`): displacement control of a set of degrees of freedom along
a direction (the load-line displacement ``δ``), others held, the force the reaction
along that direction; the same Newton iterations and adaptive steps as
:func:`~morphogen.laser.mechanics.tensile_test`, until a displacement, a drop of the
force after its maximum, or a margin beyond the first fracture of a point (the critical
plastic strains of the membrane, averaged over its fracture radius: the initiation of a
crack, ductile in the matrix, brittle in a hard phase). The curve is nominal: the force
over a ``section`` and the displacement over a ``length`` (MPa and strain), so that
:func:`~morphogen.laser.cells.failure_measures` reads its maximum (the limit load), the
first fracture (the initiation), the work up to it.

**Cases** (:class:`TensionCase`, :class:`CompactTension`, :class:`LPlate`): a mesh, the
conditions of the test, the nominal section and length, the region of the cells of a
design (:class:`~morphogen.laser.cells.CellProblem` takes a case).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import linalg as splinalg

from morphogen.laser.mechanics import Membrane, TensileResult, curve_measures, tensile_test
from morphogen.laser.specimen import QuadMesh, Specimen

# -- meshes ----------------------------------------------------------------------------


def graded(
    a: float,
    b: float,
    element: float,
    *,
    focus=(),
    fine: float | None = None,
    growth: float = 1.2,
    box: tuple | None = None,
) -> np.ndarray:
    """Coordinates from ``a`` to ``b``: elements of ``fine`` at the ``focus`` points,
    growing by ``growth`` to ``element`` away from them (uniform without focus); or
    uniform elements of ``fine`` over ``box = (lo, hi)`` (on the grid of the first
    focus inside it), growing away from its ends."""
    if not b > a:
        raise ValueError("graded: b > a")
    if box is not None and fine is not None:
        lo, hi = max(float(box[0]), a), min(float(box[1]), b)
        anchor = next((float(f) for f in focus if lo <= f <= hi), lo)
        k0 = math.ceil((lo - anchor) / fine - 1e-9)
        k1 = math.floor((hi - anchor) / fine + 1e-9)
        inner = anchor + fine * np.arange(k0, k1 + 1)
        left = (
            graded(a, inner[0], element, focus=(inner[0],), fine=fine, growth=growth)
            if inner[0] > a + 1e-9
            else np.array([a])
        )
        right = (
            graded(inner[-1], b, element, focus=(inner[-1],), fine=fine, growth=growth)
            if inner[-1] < b - 1e-9
            else np.array([b])
        )
        return np.unique(np.concatenate([left, inner, right]).round(12))
    focus = [float(f) for f in focus if a - 1e-12 <= f <= b + 1e-12]
    if not focus or fine is None:
        n = max(1, math.ceil((b - a) / element))
        return np.linspace(a, b, n + 1)
    pts = {a, b, *focus}
    for f in focus:
        for sign in (-1.0, 1.0):
            x, h = f, fine
            while True:
                x += sign * h
                end = b if sign > 0 else a
                if abs(end - x) < 0.5 * h or not a < x < b:  # no sliver at the ends
                    break
                pts.add(x)
                h = min(h * growth, element)
    xs = np.array(sorted(pts))
    # merge points closer than a third of the local size, keeping the ends and foci
    keep = [xs[0]]
    fixed = {a, b, *focus}
    for x in xs[1:]:
        if x - keep[-1] < 0.3 * min(fine, element):
            if x in fixed:
                keep[-1] = x
            continue
        keep.append(x)
    return np.array(keep)


def _tensor(xs, ys):
    """Nodes and counter-clockwise quadrilaterals of the grid ``xs x ys``."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    ny = len(ys)
    idx = np.arange(len(xs) * ny).reshape(len(xs), ny)
    quads = np.column_stack(
        [idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()]
    )
    return nodes, quads


def _compact(nodes, quads):
    """Drop the nodes no quadrilateral uses; the new numbering of the old nodes."""
    used = np.unique(quads)
    new = -np.ones(len(nodes), dtype=np.int64)
    new[used] = np.arange(len(used))
    return nodes[used], new[quads], new


def _quadmesh(nodes, quads) -> QuadMesh:
    empty = np.zeros(0, dtype=np.int64)
    return QuadMesh(
        np.asarray(nodes, float), np.asarray(quads, np.int64), empty, empty, empty, empty, (0, 0)
    )


@dataclass
class Domain:
    """A mesh and its named sets of nodes."""

    mesh: QuadMesh
    sets: dict = field(default_factory=dict)
    points: dict = field(default_factory=dict)


def ct_mesh(
    width: float = 20.0,
    crack: float = 0.5,
    *,
    element: float = 1.0,
    fine: float = 0.125,
    growth: float = 1.2,
    refine: tuple | None = None,
) -> Domain:
    """The compact tension specimen (ASTM E399 proportions; the module): ``x`` from the
    load line, the crack along ``y = 0`` from the front face ``x = −0.25 W`` to ``x =
    a = crack W``; elements of ``fine`` at the tip, or over the box ``refine = (x0, x1,
    y1)`` (``x0 ≤ x ≤ x1``, ``|y| ≤ y1``: the paths of a growing crack). Sets:
    ``upper_pin`` and ``lower_pin`` (the halves of the pin holes the pins bear on),
    ``mouth_upper`` and ``mouth_lower`` (the crack mouth); points: ``tip``."""
    W = float(width)
    a = float(crack) * W
    if not 0.2 * W <= a <= 0.8 * W:
        raise ValueError("ct: 0.2 <= a/W <= 0.8")
    h = 0.6 * W
    r, yc = 0.125 * W, 0.275 * W
    bx = by = None
    if refine is not None:  # a box of uniform fine elements (on the grid of the tip)
        x0, x1, y1 = (float(v) for v in refine)
        bx, by = (x0, x1), (0.0, y1)
    xs = graded(-0.25 * W, W, element, focus=(a,), fine=fine, growth=growth, box=bx)
    ys = graded(0.0, h, element, focus=(0.0,), fine=fine, growth=growth, box=by)
    up_n, up_q = _tensor(xs, ys)
    lo_n, lo_q = _tensor(xs, -ys[::-1])
    nx, ny = len(xs), len(ys)
    # the lower block: its row y = 0 is its last; merge the ligament nodes with the upper
    offset = len(up_n)
    lo_ids = np.arange(len(lo_n)) + offset
    lo_grid = lo_ids.reshape(nx, ny)
    up_grid = np.arange(len(up_n)).reshape(nx, ny)
    lig = xs >= a - 1e-9
    lo_grid[lig, -1] = up_grid[lig, 0]
    remap = lo_grid.ravel()
    nodes = np.vstack([up_n, lo_n])
    quads = np.vstack([up_q, remap[lo_q]])
    # the pin holes
    cen = nodes[quads].mean(axis=1)
    inside = np.zeros(len(quads), dtype=bool)
    for s in (1.0, -1.0):
        inside |= np.hypot(cen[:, 0], cen[:, 1] - s * yc) < r
    removed, kept = quads[inside], quads[~inside]
    nodes, kept, new = _compact(nodes, kept)
    rim = np.intersect1d(np.unique(removed), np.unique(quads[~inside]))
    rim = new[rim]
    rim = rim[rim >= 0]
    p = nodes[rim]
    upper = rim[(p[:, 1] > 0) & (p[:, 1] >= yc)]
    lower = rim[(p[:, 1] < 0) & (p[:, 1] <= -yc)]
    front = np.isclose(nodes[:, 0], -0.25 * W) & np.isclose(nodes[:, 1], 0.0)
    # the pins: a node at the centre of each hole, bars to the half of the rim it bears on
    first = len(nodes)
    nodes = np.vstack([nodes, [[0.0, yc], [0.0, -yc]]])
    mouth = np.flatnonzero(front)
    # the two nodes at the mouth: the one of the upper block first
    mu = new[up_grid[0, 0]]
    ml = new[lo_grid[0, -1]]
    if not (np.isin(mu, mouth) and np.isin(ml, mouth)):  # pragma: no cover - geometry
        raise RuntimeError("ct: crack mouth not found")
    tip = new[up_grid[np.flatnonzero(lig)[0], 0]]
    return Domain(
        _quadmesh(nodes, kept),
        {
            "upper_pin": upper,
            "lower_pin": lower,
            "mouth_upper": np.array([mu]),
            "mouth_lower": np.array([ml]),
            "pins": np.array([first, first + 1]),
        },
        {"tip": nodes[tip].copy(), "a": a, "W": W},
    )


def pin_springs(domain: Domain, stiffness: float) -> tuple:
    """Bars from each pin node to the rim nodes it bears on (see :func:`ct_mesh`): the
    specimen turns freely about its pins."""
    nodes = domain.mesh.nodes
    i, j = [], []
    rims = (domain.sets["upper_pin"], domain.sets["lower_pin"])
    for pin, rim in zip(domain.sets["pins"], rims, strict=True):
        i += [pin] * len(rim)
        j += list(rim)
    i, j = np.array(i), np.array(j)
    d = nodes[j] - nodes[i]
    e = d / np.linalg.norm(d, axis=1, keepdims=True)
    return i, j, np.full(len(i), float(stiffness)), e


def l_mesh(
    size: float = 40.0,
    arm: float = 16.0,
    *,
    element: float = 1.0,
    fine: float = 0.25,
    growth: float = 1.2,
) -> Domain:
    """The L-shaped plate: the square ``[0, size]²`` without ``[arm, size]²``, the
    re-entrant corner at ``(arm, arm)``. Sets: ``top`` (``y = size``, held), ``tip``
    (``x = size``, loaded)."""
    L, b = float(size), float(arm)
    if not 0 < b < L:
        raise ValueError("l-plate: 0 < arm < size")
    xs = graded(0.0, L, element, focus=(b,), fine=fine, growth=growth)
    nodes, quads = _tensor(xs, xs)
    cen = nodes[quads].mean(axis=1)
    keep = ~((cen[:, 0] > b) & (cen[:, 1] > b))
    nodes, quads, _ = _compact(nodes, quads[keep])
    top = np.flatnonzero(np.isclose(nodes[:, 1], L))
    tip = np.flatnonzero(np.isclose(nodes[:, 0], L))
    return Domain(_quadmesh(nodes, quads), {"top": top, "tip": tip}, {"corner": np.array([b, b])})


# -- the test --------------------------------------------------------------------------


def load_test(
    membrane: Membrane,
    fixed,
    moved,
    direction,
    *,
    length: float,
    section: float,
    max_disp: float,
    step: float = 0.004,
    max_step: float = 0.02,
    drop: float = 0.05,
    tol: float = 1e-5,
    max_iterations: int = 20,
    max_steps: int = 400,
    failure_margin: float | None = None,
    record_history: bool = False,
    monitors: dict | None = None,
    keep_fields: bool = True,
    erosion: bool = False,
    max_breaks: int = 2000,
    until=None,
) -> TensileResult:
    """Displacement control (see the module): the dofs ``moved`` follow ``δ
    direction`` (or a path: ``direction`` a function of ``δ`` giving their values and
    their derivatives, the force then the work-conjugate of ``δ``), the dofs ``fixed``
    are held; steps of ``step`` (then up to
    ``max_step``) times ``max_disp``; stops at ``max_disp``, when the force falls by
    ``drop`` after its maximum, or ``failure_margin`` (nominal strain) beyond the
    first fracture. With ``erosion``, the crack grows: after each step the elements
    whose ratio to fracture reaches one break (:class:`~morphogen.laser.mechanics.Membrane`:
    a residual stiffness; the averages over the living elements), the specimen is
    brought back to equilibrium at the same displacement, again until none breaks
    (a crack that runs unstably at fixed displacement); ``fields["broken"]``: the
    broken elements in their order and the strain at which each broke (at most
    ``max_breaks``). The result's strain is ``δ / length``, its stress the force over
    ``section``; ``monitors``: functions of the displacements recorded at each step
    (``fields["monitors"]``, kept with or without the fields). The membrane may be seen
    through a linear map of reduced unknowns (its ``full`` gives the nodal displacements:
    the periodic cell of :mod:`~morphogen.laser.periodic`). ``until``: a function of the
    displacements checked after each step (after the monitors): the test stops when it
    returns true."""
    t0 = time.perf_counter()
    fixed = np.unique(np.asarray(fixed, dtype=np.int64))
    moved = np.asarray(moved, dtype=np.int64)
    if callable(direction):  # a path: the values of the moved dofs and their rates
        path = direction
    else:
        dirn = np.asarray(direction, dtype=float)

        def path(d):
            return d * dirn, dirn

    if np.shape(path(0.0)[0]) != moved.shape:
        raise ValueError("load test: one direction per moved dof")
    held = np.unique(np.concatenate([fixed, moved]))
    free = np.setdiff1d(np.arange(membrane.ndof), held)
    membrane.reset()
    u = membrane.u.copy()
    f, K, _, _ = membrane.evaluate(u)
    disp, inc = 0.0, step * max_disp
    min_inc = 1e-4 * step * max_disp
    rec = {"strain": [0.0], "stress": [0.0], "force": [0.0], "displacement": [0.0]}
    mons = {k: [float(fn(u))] for k, fn in (monitors or {}).items()}
    history = [(0.0, 0.0, membrane.Fn.copy())] if record_history else []
    has_critical = membrane.fracture.any
    failure, ratio_prev = math.nan, 0.0
    fmax, steps, iterations = 0.0, 0, 0
    scale_f = 1e-9 * membrane.young * section
    broken, broken_at = [], []

    breakable = membrane.fracture.finite.any(axis=(1, 2))

    def newton(trial):
        """Newton's method at fixed displacements; a correction that turns an element
        inside out is halved (the iterates near a crack tip, after a break)."""
        nonlocal iterations
        fi, Ki, state, bad = membrane.evaluate(trial)
        iterations += 1
        if bad or not np.all(np.isfinite(fi)):
            return False, trial, fi, Ki, state, 0
        for it in range(max_iterations):
            r = fi[free]
            scale = max(np.linalg.norm(fi[held]), scale_f)
            if np.linalg.norm(r) <= tol * scale:
                return True, trial, fi, Ki, state, it
            try:
                du = splinalg.spsolve(Ki[free][:, free].tocsc(), -r)
            except RuntimeError:
                break
            if not np.all(np.isfinite(du)):
                break
            step, done = 1.0, False
            for _ in range(8):
                cand = trial.copy()
                cand[free] += step * du
                f2, K2, s2, bad2 = membrane.evaluate(cand)
                iterations += 1
                if not bad2 and np.all(np.isfinite(f2)):
                    done = True
                    break
                step *= 0.5
            if not done:
                break
            trial, fi, Ki, state = cand, f2, K2, s2
        return False, trial, fi, Ki, state, max_iterations

    while True:
        target = disp + inc
        trial = u.copy()
        dpres = np.zeros(membrane.ndof)
        dpres[moved] = path(target)[0] - path(disp)[0]
        rhs = -(f[free] + K[free][:, moved] @ dpres[moved])
        try:
            trial[free] += splinalg.spsolve(K[free][:, free].tocsc(), rhs)
        except RuntimeError:
            pass
        trial[fixed] = 0.0
        trial[moved] = path(target)[0]
        converged, trial, fi, Ki, state, it = newton(trial)
        if not converged and erosion and has_critical and broken:
            # no equilibrium: the elements of the front of the crack the trial turned
            # inside out have failed, broken before the step is cut
            ratios = membrane.fracture.ratios(membrane.p_n, membrane.alive)
            quads = membrane.mesh.quads
            front = np.isin(quads, quads[~membrane.alive]).any(axis=1)
            inv = np.flatnonzero(
                getattr(membrane, "inverted", np.zeros(len(ratios), bool))
                & membrane.alive
                & breakable
                & front
            )
            if inv.size and len(broken) < max_breaks:
                inv = inv[np.argsort(-ratios[inv])][:2]
                membrane.alive[inv] = False
                broken.extend(inv.tolist())
                broken_at.extend([disp / length] * len(inv))
                ok, trial2, fi2, Ki2, state2, _ = newton(u.copy())
                if ok:
                    u, f, K = trial2, fi2, Ki2
                    membrane.commit(u, state2)
                continue
        if not converged:
            inc *= 0.5
            if inc < min_inc:
                break
            continue
        u, f, K = trial, fi, Ki
        membrane.commit(u, state)
        disp = target
        steps += 1
        strain = disp / length
        if erosion and has_critical:  # the crack grows at this displacement
            while len(broken) < max_breaks:
                ratios = membrane.fracture.ratios(membrane.p_n, membrane.alive)
                now = np.flatnonzero(ratios >= 1.0)
                if not now.size:
                    break
                # the most critical first, one at a time: the stresses redistribute
                now = now[np.argsort(-ratios[now])][:1]
                membrane.alive[now] = False
                broken.extend(now.tolist())
                broken_at.extend([strain] * len(now))
                ok, trial2, fi2, Ki2, state2, _ = newton(u.copy())
                if not ok:  # the element broken, the step kept as it is
                    break
                u, f, K = trial2, fi2, Ki2
                membrane.commit(u, state2)
        force = float(f[moved] @ path(disp)[1])
        if record_history:
            history.append((strain, disp, membrane.Fn.copy()))
        ratio = (
            float(membrane.fracture.ratios(membrane.p_n).max())
            if has_critical and not broken
            else (1.0 if broken else 0.0)
        )
        if ratio >= 1.0 and not math.isfinite(failure):
            e_prev = rec["strain"][-1]
            frac = (1.0 - ratio_prev) / max(ratio - ratio_prev, 1e-12)
            failure = e_prev + min(max(frac, 0.0), 1.0) * (strain - e_prev)
        ratio_prev = ratio
        rec["strain"].append(strain)
        rec["stress"].append(force / section)
        rec["force"].append(force)
        rec["displacement"].append(disp)
        for k, fn in (monitors or {}).items():
            mons[k].append(float(fn(u)))
        fmax = max(fmax, force)
        if (
            disp >= max_disp * (1 - 1e-12)
            or (force < (1.0 - drop) * fmax)
            or steps >= max_steps
            or (failure_margin is not None and strain >= failure + failure_margin)
            or (until is not None and until(u))
        ):
            break
        if it <= 4:
            inc *= 1.5
        elif it >= 8:
            inc *= 0.7
        inc = max(min(inc, max_step * max_disp, max_disp - disp), min_inc)
    arr = {k: np.asarray(v) for k, v in rec.items()}
    ys, uts, eu, energy, necked = curve_measures(arr["strain"], arr["stress"])
    fields = {"monitors": {k: np.asarray(v) for k, v in mons.items()}} if monitors else {}
    if erosion:
        fields["broken"] = (np.asarray(broken, dtype=np.int64), np.asarray(broken_at))
    if keep_fields:
        full = getattr(membrane, "full", None)  # the nodal displacements of reduced unknowns
        fields.update(
            {
                "mesh": (membrane.mesh.nodes, membrane.mesh.quads),
                "displacement": (u if full is None else full(u)).reshape(-1, 2).copy(),
                "plastic_strain": membrane.p_n.mean(axis=2),
                "stress": membrane.sig_n.mean(axis=2),
                "monitors": {k: np.asarray(v) for k, v in mons.items()},
            }
        )
    return TensileResult(
        strain=arr["strain"],
        stress=arr["stress"],
        force=arr["force"],
        displacement=arr["displacement"],
        yield_strength=ys,
        tensile_strength=uts,
        uniform_elongation=eu,
        energy=energy,
        necked=necked,
        steps=steps,
        iterations=iterations,
        time_s=time.perf_counter() - t0,
        failure_strain=failure,
        history=history,
        fields=fields,
    )


# -- the cases -------------------------------------------------------------------------


@dataclass
class TensionCase:
    """The tensile test of a specimen (:func:`~morphogen.laser.mechanics.tensile_test`):
    the cells over its parallel part, the tracks continuing to the grips."""

    specimen: Specimen = field(default_factory=Specimen)
    element: float = 0.5

    kind = "tension"
    extend = True
    columns = True

    def __post_init__(self):
        self.specimen = Specimen.from_config(self.specimen)
        self.mesh = self.specimen.mesh(self.element)
        self.thickness = self.specimen.thickness
        self.region = self.specimen.band()
        self.section = self.specimen.width * self.specimen.thickness
        self.springs = None

    def run(
        self,
        membrane,
        *,
        max_strain=0.35,
        failure_margin=0.03,
        record_history=False,
        keep_fields=True,
    ):
        return tensile_test(
            self.specimen,
            membrane,
            max_strain=max_strain,
            failure_margin=failure_margin,
            record_history=record_history,
            keep_fields=keep_fields,
        )


@dataclass
class CompactTension:
    """A compact tension specimen (see the module) pulled by its pins: the nominal
    stress ``F / (B W)``, the nominal strain the load-line displacement over ``W``; the
    cells over ``region`` (default: a box around the crack tip and the ligament); the
    crack mouth opening displacement among the monitors; ``refine``: the box of fine
    elements (:func:`ct_mesh`); ``propagate``: the crack grows by the breaking of the
    elements (:func:`load_test` with ``erosion``) until the force has fallen to 5 % of
    its maximum (the specimen separated) or ``max_disp``."""

    width: float = 20.0
    crack: float = 0.5
    thickness: float = 1.0
    element: float = 1.0
    fine: float = 0.125
    growth: float = 1.2
    region: tuple | None = None
    max_disp: float | None = None
    refine: tuple | None = None
    propagate: bool = False

    kind = "ct"
    extend = False
    columns = False

    def __post_init__(self):
        self.domain = ct_mesh(
            self.width,
            self.crack,
            element=self.element,
            fine=self.fine,
            growth=self.growth,
            refine=self.refine,
        )
        self.mesh = self.domain.mesh
        W, a = self.width, self.crack * self.width
        if self.region is None:
            self.region = (a - 0.2 * W, a + 0.4 * W, -0.25 * W, 0.25 * W)
        self.section = self.thickness * W
        self.length = W
        s = self.domain.sets
        pu, pl = s["pins"]
        self.springs = pin_springs(self.domain, 1e3 * 210000.0 * self.thickness)
        self.fixed = np.array([2 * pu, 2 * pl])  # the pins hold x and turn freely
        self.moved = np.array([2 * pu + 1, 2 * pl + 1])
        self.direction = np.array([0.5, -0.5])
        mu, ml = s["mouth_upper"][0], s["mouth_lower"][0]
        self.monitors = {"cmod": lambda u, mu=mu, ml=ml: u[2 * mu + 1] - u[2 * ml + 1]}

    def run(
        self,
        membrane,
        *,
        max_strain=None,
        failure_margin=0.01,
        record_history=False,
        keep_fields=True,
    ):
        max_disp = self.max_disp or 0.1 * self.width
        if max_strain is not None:
            max_disp = min(max_disp, max_strain * self.width)
        if self.propagate:  # the crack grows through the ligament, to separation
            return load_test(
                membrane,
                self.fixed,
                self.moved,
                self.direction,
                length=self.length,
                section=self.section,
                max_disp=max_disp,
                drop=0.95,
                record_history=record_history,
                monitors=self.monitors,
                keep_fields=keep_fields,
                erosion=True,
                max_steps=2000,
            )
        return load_test(
            membrane,
            self.fixed,
            self.moved,
            self.direction,
            length=self.length,
            section=self.section,
            max_disp=max_disp,
            failure_margin=failure_margin,
            record_history=record_history,
            monitors=self.monitors,
            keep_fields=keep_fields,
        )


@dataclass
class LPlate:
    """An L-shaped plate (see the module) held along its top edge and pulled down at the
    end of its arm: the nominal stress ``F / (t arm)``, the nominal strain the
    displacement over ``size``; the cells over ``region`` (default: around the
    re-entrant corner)."""

    size: float = 40.0
    arm: float = 16.0
    thickness: float = 1.0
    element: float = 1.0
    fine: float = 0.25
    growth: float = 1.2
    region: tuple | None = None
    max_disp: float | None = None

    kind = "l_plate"
    extend = False
    columns = False

    def __post_init__(self):
        self.domain = l_mesh(
            self.size, self.arm, element=self.element, fine=self.fine, growth=self.growth
        )
        self.mesh = self.domain.mesh
        b = self.arm
        if self.region is None:
            self.region = (0.0, 2.0 * b, 0.0, 2.0 * b)
        self.section = self.thickness * b
        self.length = self.size
        top, tip = self.domain.sets["top"], self.domain.sets["tip"]
        self.springs = None
        self.fixed = np.concatenate([2 * top, 2 * top + 1])
        self.moved = 2 * tip + 1
        self.direction = -np.ones(len(tip))
        self.monitors = {}

    def run(
        self,
        membrane,
        *,
        max_strain=None,
        failure_margin=0.005,
        record_history=False,
        keep_fields=True,
    ):
        max_disp = self.max_disp or 0.1 * self.size
        if max_strain is not None:
            max_disp = min(max_disp, max_strain * self.size)
        return load_test(
            membrane,
            self.fixed,
            self.moved,
            self.direction,
            length=self.length,
            section=self.section,
            max_disp=max_disp,
            failure_margin=failure_margin,
            record_history=record_history,
            keep_fields=keep_fields,
        )


CASES = {"tension": TensionCase, "ct": CompactTension, "l_plate": LPlate}


def case_from_config(data):
    """A case of :data:`CASES` from a mapping (``kind`` and its parameters)."""
    if data is None:
        return None
    if not isinstance(data, dict):
        return data
    d = dict(data)
    kind = d.pop("kind", "tension")
    if kind not in CASES:
        raise ValueError(f"case: one of {sorted(CASES)}")
    return CASES[kind](**d)
