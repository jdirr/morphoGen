"""Isogeometric shape optimization: the control points of the geometry as the design.

In an isogeometric analysis the geometry is its control points, and moving them moves
the boundary exactly and smoothly: they are the natural variables of a shape
optimization (Wall, Frenzel & Cyron 2008; Qian 2010), without the remeshing and the
jagged boundaries of the methods that move the nodes of a mesh. Three levels:

* the *variables* ``x``: each moves some control points of the design along given
  directions (:meth:`ShapeDesign.variable`; :meth:`ShapeDesign.radial` scales the control
  points of a hole about its centre, with the symmetries of the cell). With the
  ``symmetry`` of the pattern — mirrors, rotations (:func:`mirror`, :func:`rotation`),
  and the periods of the cell — a variable moves the images of its control points as
  well: the design keeps its symmetry, and the control points of the boundary of a
  cell that are a period apart move together;
* the *design*, a coarse assembly whose control points are ``P(x) = P_0 + M x``;
* the *analysis*, a refinement of the design (elements cut, degree raised): for given
  weights its control points are linear in those of the design, ``Q = T P``
  (:meth:`~morphogen.iga.geometry.Multipatch.refinement`). The geometry is the same at
  both levels; the design keeps few variables, the analysis its accuracy.

Sensitivities. Let ``A(u, v) = int (E_u + eps(u)) : D : (E_v + eps(v)) dOmega`` for two
fields of given unknowns and uniform strains. Moving the control point ``a`` by ``dQ``
moves the points by ``V = R_a dQ``; then ``d(dOmega) = div V dOmega`` and ``d(grad u) =
-grad u grad V``, so that::

    dA / dQ_a = int Sigma grad R_a dOmega,
    Sigma = (eps_u : D : eps_v) 1 - grad u^T sigma_v - grad v^T sigma_u,

the energy-momentum tensor of Eshelby for the pair: the gradients are *analytical*, one
integral per pair of fields and no further solution. For a structure under given loads
(``K u = f``) the compliance ``c = f . u`` has ``dc/dQ = -dA(u, u)/dQ``; for a periodic
cell the effective tensor is ``C_IJ = A(chi_I, chi_J) / |Y|`` with the fluctuations of
:mod:`morphogen.iga.homogenization`, and ``dC_IJ/dQ = (dA/dQ) / |Y|`` since each
fluctuation makes its energy stationary. The area of the material has ``dA/dQ_a = int
grad R_a dOmega``. The chain ``d/dx = M^T T^T d/dQ`` brings them to the variables.

Optimization (:func:`optimize`, :func:`optimize_cell`, :func:`optimize_structure`):
sequential quadratic programming (SLSQP of SciPy) or the method of moving asymptotes
(:mod:`morphogen.optimization.mma`), with bounds on the variables that keep the patches
from folding. The objectives and constraints of a cell are functions of its effective
tensor and of its fraction of material — a modulus, Poisson's ratio, the distance to a
target tensor —, differentiated numerically with respect to these few numbers; linear
constraints on the variables (``linear``) keep sizes of the geometry — the width of a
ligament, the thickness of a member — since the control points are linear in them.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize
from scipy.spatial import cKDTree

from morphogen.iga.analysis import Analysis
from morphogen.iga.geometry import Multipatch, Patch
from morphogen.iga.homogenization import Cell, CellSolution

METHODS = ("slsqp", "mma")
SYMMETRIES = (None, "d2", "d4")


#: The largest group of symmetries of a design (those of a cubic cell are 48)
_GROUP = 192


class FoldedDesign(ValueError):
    """The variables of a design fold one of its patches (a Jacobian that is not
    positive): no analysis there."""


def mirror(axis: int, at: float = 0.0, dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """The mirror in the line (plane) where the coordinate ``axis`` is ``at``, as an
    operation ``x -> matrix x + shift`` (for the ``symmetry`` of a variable)."""
    matrix, shift = np.eye(dim), np.zeros(dim)
    matrix[axis, axis] = -1.0
    shift[axis] = 2.0 * float(at)
    return matrix, shift


def rotation(angle: float, centre=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """The rotation of the plane by ``angle`` (degrees, anticlockwise) about ``centre``,
    as an operation ``x -> matrix x + shift`` (for the ``symmetry`` of a variable)."""
    turn = round(float(angle) / 90.0)
    if abs(float(angle) - 90.0 * turn) < 1e-12:  # exact quarter-turns
        c, s = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][turn % 4]
    else:
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    matrix = np.array([[c, -s], [s, c]])
    centre = np.asarray(centre, dtype=float)
    return matrix, centre - matrix @ centre


def _group(operations, lattice, dim: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The operations ``x -> matrix x + shift`` that given ones generate, the shifts
    counted up to the periods of ``lattice`` (when there is one)."""
    inverse = None if lattice is None else np.linalg.inv(lattice)

    def reduced(matrix, shift):
        if inverse is not None:
            share = (shift @ inverse) % 1.0
            share[share > 1.0 - 1e-7] = 0.0
            shift = share @ lattice
        key = (tuple(np.round(matrix, 6).ravel() + 0.0), tuple(np.round(shift, 6) + 0.0))
        return key, (matrix, shift)

    generators = [(np.eye(dim), np.zeros(dim))]
    for matrix, shift in operations:
        matrix, shift = np.asarray(matrix, dtype=float), np.asarray(shift, dtype=float)
        if matrix.shape != (dim, dim) or shift.shape != (dim,):
            raise ValueError(f"symmetry: operations (matrix, shift) of {dim} dimensions")
        if np.abs(matrix @ matrix.T - np.eye(dim)).max() > 1e-9:
            raise ValueError("symmetry: mirrors and rotations (orthogonal matrices)")
        generators.append((matrix, shift))
    group = dict(reduced(m, t) for m, t in generators)
    frontier = list(group.values())
    while frontier:
        found = []
        for m1, t1 in frontier:
            for m2, t2 in generators[1:]:
                key, operation = reduced(m1 @ m2, m1 @ t2 + t1)
                if key not in group:
                    group[key] = operation
                    found.append(operation)
        if len(group) > _GROUP:
            raise ValueError(
                "symmetry: these operations do not make a finite group (up to the periods "
                "of the lattice of the design)"
            )
        frontier = found
    return list(group.values())


def stress_tensors(flux: np.ndarray, dim: int, nd: int) -> np.ndarray:
    """The fluxes of the points as tensors (... x unknowns x dim): the stress tensors of
    Voigt's components (elasticity), or the flux vectors as one row (conduction)."""
    if nd == 1:
        return flux[..., None, :]
    out = np.empty((*flux.shape[:-1], dim, dim))
    if dim == 2:
        out[..., 0, 0], out[..., 1, 1] = flux[..., 0], flux[..., 1]
        out[..., 0, 1] = out[..., 1, 0] = flux[..., 2]
    else:
        for k in range(3):
            out[..., k, k] = flux[..., k]
        out[..., 1, 2] = out[..., 2, 1] = flux[..., 3]
        out[..., 0, 2] = out[..., 2, 0] = flux[..., 4]
        out[..., 0, 1] = out[..., 1, 0] = flux[..., 5]
    return out


def energy_gradients(analysis: Analysis, fields, uniform=None) -> np.ndarray:
    """The derivatives of ``A(u_i, u_j)`` (see the module) with respect to the control
    points of the analysis, for the fields ``fields`` (unknowns x m) and their uniform
    strains ``uniform`` (components x m; none by default): m x m x control points x
    dim."""
    fields = np.asarray(fields, dtype=float)
    fields = fields[:, None] if fields.ndim == 1 else fields
    m = fields.shape[1]
    model, dim, nd = analysis.model, analysis.dim, analysis.nd
    out = np.zeros((m, m, model.n_nodes, dim))
    eye = np.eye(dim)
    for pts, b, d, dofs in analysis.points():
        ue = fields[dofs]  # elements x unknowns of the element x m
        measure = np.einsum("eqsi,eim->eqsm", b, ue)
        if uniform is not None:
            measure = measure + np.asarray(uniform, dtype=float)[None, None]
        flux = np.einsum("st,eqtm->eqsm", d, measure)
        ua = ue.reshape(len(dofs), -1, nd, m)
        grad = np.einsum("eqak,eacm->eqmck", pts.gradients, ua)  # of the unknowns
        sigma = stress_tensors(np.moveaxis(flux, -1, 2), dim, nd)  # elements x points x m x ...
        for i in range(m):
            for j in range(i, m):
                energy = np.einsum("eqs,eqs->eq", measure[..., i], flux[..., j])
                tensor = energy[..., None, None] * eye
                tensor = tensor - np.einsum("eqck,eqcj->eqkj", grad[:, :, i], sigma[:, :, j])
                tensor = tensor - np.einsum("eqck,eqcj->eqkj", grad[:, :, j], sigma[:, :, i])
                g = np.einsum("eq,eqkj,eqaj->eak", pts.weights, tensor, pts.gradients)
                np.add.at(out[i, j], pts.nodes, g)
    for i in range(m):
        for j in range(i):
            out[i, j] = out[j, i]
    return out


def measure_gradient(analysis: Analysis) -> np.ndarray:
    """The derivative of the area (volume) of the assembly with respect to its control
    points: ``int grad R_a`` (control points x dim)."""
    out = np.zeros((analysis.model.n_nodes, analysis.dim))
    for pts, _, _, _ in analysis.points():
        np.add.at(out, pts.nodes, np.einsum("eq,eqak->eak", pts.weights, pts.gradients))
    return out


class ShapeDesign:
    """A design (see the module): the coarse assembly ``model`` and the refinement of
    its analysis (``subdivisions`` of every element and degrees raised by ``elevate``,
    one value or one per direction). Variables are added by :meth:`variable` and
    :meth:`radial`."""

    def __init__(self, model, *, subdivisions=4, elevate=1):
        if isinstance(model, Patch):
            model = Multipatch(model)
        self.model = model
        self.fine, self.matrix = model.refinement(subdivisions, elevate)
        self._moves: list[tuple[np.ndarray, np.ndarray]] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.names: list[str] = []

    @property
    def n(self) -> int:
        """The number of variables."""
        return len(self._moves)

    @property
    def moves(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """For each variable, the control points of the design that it moves and their
        directions (the moves of a unit of the variable)."""
        return [(nodes.copy(), directions.copy()) for nodes, directions in self._moves]

    def variable(self, nodes, directions, lower: float = -np.inf, upper: float = np.inf,
                 name: str | None = None, symmetry=None) -> int:  # fmt: skip
        """A variable that moves the control points ``nodes`` of the design by its value
        times their ``directions`` (one vector each, or one for all); returns its index.

        ``symmetry``: operations ``(matrix, shift)`` that map the design onto itself,
        ``x -> matrix x + shift`` (:func:`mirror`, :func:`rotation`), up to the periods
        of its ``lattice``. The variable then moves the images of its control points by
        all the operations that these generate, along the images of their directions —
        ``nodes`` are those of one copy; a control point that an operation leaves in
        place keeps the part of its direction that the operation keeps. With an empty
        list, only the control points a period apart move together."""
        nodes = np.atleast_1d(np.asarray(nodes, dtype=np.int64))
        directions = np.broadcast_to(
            np.asarray(directions, dtype=float), (len(nodes), self.model.dim)
        ).copy()
        if nodes.size == 0 or nodes.min() < 0 or nodes.max() >= self.model.n_nodes:
            raise ValueError("variable: control points of the design")
        if not lower < upper:
            raise ValueError("variable: lower < upper")
        if symmetry is not None:
            nodes, directions = self._images(nodes, directions, symmetry)
        self._moves.append((nodes, directions))
        self.lower.append(float(lower))
        self.upper.append(float(upper))
        self.names.append(name or f"x{len(self._moves)}")
        return len(self._moves) - 1

    def _images(self, nodes, directions, symmetry) -> tuple[np.ndarray, np.ndarray]:
        """The control points that the symmetry maps those of a variable onto, and their
        directions."""
        model = self.model
        group = _group(symmetry, model.lattice, model.dim)
        periods = np.zeros((1, model.dim))
        if model.lattice is not None:
            combos = itertools.product(range(-3, 4), repeat=model.dim)
            periods = np.array(list(combos), dtype=float) @ model.lattice
        tree = cKDTree(model.nodes)
        tolerance = 1e-6 * float(np.ptp(model.nodes, axis=0).max())
        moves: dict[int, np.ndarray] = {}
        for node, direction in zip(nodes, directions, strict=True):
            images: dict[int, list] = {}
            for matrix, shift in group:
                distance, found = tree.query(matrix @ model.nodes[node] + shift + periods)
                found = found[distance <= tolerance]
                if not len(found):
                    raise ValueError(
                        f"symmetry: no control point at an image of the control point {node}; "
                        "the operations must map the design onto itself"
                    )
                for other in found:
                    images.setdefault(int(other), []).append(matrix @ direction)
            for other, parts in images.items():  # as many operations for each image
                moves[other] = moves.get(other, 0.0) + np.sum(parts, axis=0) / len(images[node])
        ids = np.array(sorted(moves), dtype=np.int64)
        out = np.array([moves[k] for k in ids])
        if np.abs(out).max() < 1e-12 * max(float(np.abs(directions).max()), 1e-300):
            raise ValueError("symmetry: it leaves nothing of the move of this variable")
        return ids, out

    def radial(self, nodes, centre, lower: float = -0.3, upper: float = 0.3,
               symmetry: str | None = None) -> list[int]:  # fmt: skip
        """Variables that scale the positions of control points about ``centre``: ``P =
        centre + (1 + x) (P_0 - centre)``, ``x`` between ``lower`` and ``upper``. One
        variable per control point, or per set of control points that the ``symmetry``
        of the cell maps onto each other: ``"d2"`` (the two mirrors of a rectangle
        through the centre) or ``"d4"`` (those of a square, diagonals included)."""
        if symmetry not in SYMMETRIES:
            raise ValueError(f"symmetry: one of {SYMMETRIES}")
        nodes = np.unique(np.asarray(nodes, dtype=np.int64))
        offsets = self.model.nodes[nodes] - np.asarray(centre, dtype=float)
        if symmetry is None:
            groups = [[k] for k in range(len(nodes))]
        else:
            folded = np.abs(offsets[:, :2])
            if symmetry == "d4":
                folded = np.sort(folded, axis=1)
            scale = float(np.abs(offsets).max()) or 1.0
            keys = np.round(folded / (1e-7 * scale)).astype(np.int64)
            _, inverse = np.unique(keys, axis=0, return_inverse=True)
            inverse = inverse.ravel()
            groups = [np.flatnonzero(inverse == g).tolist() for g in range(inverse.max() + 1)]
        return [self.variable(nodes[g], offsets[g], lower, upper) for g in groups]

    # -- geometries ------------------------------------------------------------------------

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.lower), np.array(self.upper)

    def control(self, x) -> np.ndarray:
        """The control points of the design for the variables ``x``."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n,):
            raise ValueError(f"x: {self.n} variables")
        nodes = self.model.nodes.copy()
        for value, (ids, directions) in zip(x, self._moves, strict=True):
            np.add.at(nodes, ids, value * directions)
        return nodes

    def design_model(self, x) -> Multipatch:
        """The design moved by the variables ``x``."""
        return self.model.with_nodes(self.control(x))

    def analysis_model(self, x) -> Multipatch:
        """The refined assembly of the analysis for the variables ``x``; a
        :class:`FoldedDesign` if a patch of the design folds there (a negative Jacobian,
        its boundary included: :meth:`~morphogen.iga.geometry.Multipatch.jacobians`)."""
        control = self.control(x)
        low, high = self.model.with_nodes(control).jacobians()
        if low < -1e-9 * high:
            raise FoldedDesign(
                "a patch of the design folds for these variables (a negative Jacobian); "
                "tighten their bounds"
            )
        return self.fine.with_nodes(self.matrix @ control)

    def pull(self, gradient) -> np.ndarray:
        """The derivative with respect to the variables of a quantity whose derivative
        with respect to the control points of the analysis is ``gradient`` (control
        points x dim)."""
        coarse = self.matrix.T @ np.asarray(gradient, dtype=float)
        return np.array(
            [float(np.sum(coarse[ids] * directions)) for ids, directions in self._moves]
        )


@dataclass
class CellResponse:
    """The effective ``tensor`` of a cell and its ``fraction`` of material for some
    variables, with their derivatives (``d_tensor``: n x n x variables; ``d_fraction``:
    variables) and the solution."""

    tensor: np.ndarray
    fraction: float
    d_tensor: np.ndarray
    d_fraction: np.ndarray
    solution: CellSolution = field(repr=False)


def cell_response(design: ShapeDesign, x, materials, *, physics: str = "elastic",
                  plane: str = "strain") -> CellResponse:  # fmt: skip
    """The effective tensor and the fraction of material of the periodic cell of a
    design for the variables ``x``, and their derivatives with respect to ``x``."""
    cell = Cell(design.analysis_model(x), materials, physics=physics, plane=plane)
    solution = cell.solve()
    n = len(solution.tensor)
    gradients = energy_gradients(cell.analysis, solution.fluctuations, np.eye(n))
    d_tensor = np.empty((n, n, design.n))
    for i in range(n):
        for j in range(n):
            d_tensor[i, j] = design.pull(gradients[i, j]) / cell.measure
    d_fraction = design.pull(measure_gradient(cell.analysis)) / cell.measure
    return CellResponse(solution.tensor, solution.fraction, d_tensor, d_fraction, solution)


@dataclass
class StructureResponse:
    """The ``compliance`` ``f . u`` of a structure and the ``measure`` (area, volume) of
    its material, with their derivatives with respect to the variables, the unknowns
    and the analysis."""

    compliance: float
    measure: float
    d_compliance: np.ndarray
    d_measure: np.ndarray
    u: np.ndarray = field(repr=False)
    analysis: Analysis = field(repr=False)


def structure_response(design: ShapeDesign, x, materials, loads: Callable, supports: Callable,
                       *, plane: str = "strain") -> StructureResponse:  # fmt: skip
    """The compliance of the structure of a design under the nodal forces
    ``loads(analysis)`` with the unknowns ``supports(analysis)`` fixed (functions of the
    :class:`~morphogen.iga.analysis.Analysis`: the loads must not depend on the
    variables — tractions on sides that the design does not move)."""
    analysis = Analysis(design.analysis_model(x), materials, plane=plane)
    forces = np.asarray(loads(analysis), dtype=float)
    u = analysis.solve(forces, supports(analysis))
    gradient = -energy_gradients(analysis, u)[0, 0]
    return StructureResponse(
        float(forces @ u),
        analysis.measure(),
        design.pull(gradient),
        design.pull(measure_gradient(analysis)),
        u,
        analysis,
    )


@dataclass
class ShapeResult:
    """An optimized shape: the variables ``x``, the ``design`` and the ``analysis``
    assemblies, the ``objective`` and the ``constraints`` there, the ``history`` of the
    iterations (objective, largest constraint, variables) and, for a cell, its effective
    ``tensor`` and ``fraction`` of material."""

    x: np.ndarray
    design: Multipatch
    analysis: Multipatch
    objective: float
    constraints: np.ndarray
    iterations: int
    converged: bool
    history: list = field(default_factory=list, repr=False)
    tensor: np.ndarray | None = None
    fraction: float | None = None
    time_s: float = 0.0
    message: str = ""


def optimize(
    evaluate: Callable,
    x0,
    lower,
    upper,
    *,
    method: str = "slsqp",
    maxiter: int = 100,
    tolerance: float = 1e-7,
    move: float = 0.2,
    callback: Callable | None = None,
):
    """Minimize ``f(x)`` subject to ``g_i(x) <= 0`` and ``lower <= x <= upper``, with
    ``evaluate(x) -> (f, df, g, dg)`` (``g``: m values, ``dg``: m x n). ``method``:
    ``"slsqp"`` (sequential quadratic programming, SciPy) or ``"mma"`` (moving asymptotes).
    ``move`` bounds the steps as a share of the bounds: each step of MMA, and the first
    of SLSQP (whose objective is scaled for it; the later ones follow its estimate of the
    curvature). Returns ``(x, history, converged, message)``: the history holds the
    points that the iterations kept, and ``callback(iteration, x, f, g)`` is called for
    each. A trial point where ``evaluate`` raises :class:`FoldedDesign` is turned down:
    the line search of SLSQP shortens its step, the step of MMA is halved."""
    if method not in METHODS:
        raise ValueError(f"method: one of {METHODS}")
    x0 = np.clip(np.asarray(x0, dtype=float), lower, upper)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    trail: list[dict] = []  # the points evaluated, in order
    history: list = []

    def at(x) -> dict:
        x = np.asarray(x, dtype=float)
        if trail and np.array_equal(trail[-1]["x"], x):
            return trail[-1]
        try:
            f, df, g, dg = evaluate(x)
        except FoldedDesign:
            valid = next((entry for entry in reversed(trail) if entry["valid"]), None)
            if valid is None:
                raise
            # a trial step too far: a value that the line search turns down
            entry = dict(valid, x=x.copy(), f=valid["f"] + 10.0 * (abs(valid["f"]) + 1.0))
            entry["valid"] = False
        else:
            g = np.atleast_1d(np.asarray(g, dtype=float))
            entry = {"x": x.copy(), "f": float(f), "df": np.asarray(df, dtype=float), "g": g,
                     "dg": np.asarray(dg, dtype=float).reshape(len(g), len(x)),
                     "valid": True}  # fmt: skip
        trail.append(entry)
        return entry

    def record(entry) -> None:
        history.append({"objective": entry["f"], "x": entry["x"].copy(),
                        "constraint": float(entry["g"].max(initial=-np.inf))})  # fmt: skip
        if callback is not None:
            callback(len(history) - 1, entry["x"].copy(), entry["f"], entry["g"])

    first = at(x0)
    record(first)
    if method == "slsqp":
        # a first step of the size of ``move``: SLSQP starts from the steepest descent
        spans = (upper - lower)[np.isfinite(upper - lower)]
        typical = float(np.median(spans)) if spans.size else 1.0
        scale = max(1.0, float(np.abs(first["df"]).max(initial=0.0)) / (move * typical))

        def kept(_) -> None:
            # SciPy calls this at the first trial point of an iteration: the point
            # evaluated before it is the one that the previous iteration kept
            if len(trail) > 1 and not np.array_equal(trail[-2]["x"], history[-1]["x"]):
                record(trail[-2])

        constraints = [
            {
                "type": "ineq",
                "fun": lambda x, i=i: -at(x)["g"][i],
                "jac": lambda x, i=i: -at(x)["dg"][i],
            }
            for i in range(len(first["g"]))
        ]
        result = scipy.optimize.minimize(
            lambda x: (at(x)["f"] / scale, at(x)["df"] / scale),
            x0,
            jac=True,
            method="SLSQP",
            bounds=list(zip(lower, upper, strict=True)),
            constraints=constraints,
            callback=kept,
            options={"maxiter": int(maxiter), "ftol": float(tolerance) / scale},
        )
        last = at(np.asarray(result.x, dtype=float))
        if not np.array_equal(last["x"], history[-1]["x"]):
            record(last)
        return last["x"], history, bool(result.success), str(result.message)
    from morphogen.optimization.mma import MMA

    x = x0.copy()
    m = max(len(first["g"]), 1)
    if m >= len(x):
        raise ValueError("mma: fewer constraints than variables")
    scale = abs(first["f"]) or 1.0
    optimizer = MMA(len(x), m, lower, upper, move=move)
    for _ in range(int(maxiter)):
        entry = at(x)
        g, dg = entry["g"], entry["dg"]
        if len(g) == 0:  # MMA needs a constraint: one that never binds
            g, dg = np.array([-1.0]), np.zeros((1, len(x)))
        new = optimizer.update(x, entry["f"] / scale, entry["df"] / scale, g, dg)
        for _ in range(30):  # a step that folds the design is halved
            if at(new)["valid"]:
                break
            new = x + 0.5 * (new - x)
        else:
            raise FoldedDesign("mma: every step from these variables folds the design")
        change = float(np.abs(new - x).max() / max(float((upper - lower).max()), 1e-30))
        x = new
        record(at(x))
        if change < tolerance:
            return x, history, True, "the variables no longer change"
    return x, history, False, f"{maxiter} iterations"


def _partials(function: Callable, tensor: np.ndarray, fraction: float):
    """The value of ``function(tensor, fraction)`` and its derivatives with respect to
    the entries of the tensor and to the fraction (central differences)."""
    value = float(function(tensor, fraction))
    step = 1e-6 * float(np.abs(tensor).max())
    d_tensor = np.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            plus, minus = tensor.copy(), tensor.copy()
            plus[i, j] += step
            minus[i, j] -= step
            d_tensor[i, j] = (function(plus, fraction) - function(minus, fraction)) / (2 * step)
    d_fraction = (function(tensor, fraction + 1e-6) - function(tensor, fraction - 1e-6)) / 2e-6
    return value, d_tensor, float(d_fraction)


def optimize_cell(design: ShapeDesign, materials, objective: Callable, constraints=(), *,
                  linear=None, x0=None, physics: str = "elastic", plane: str = "strain",
                  method: str = "slsqp", maxiter: int = 100, tolerance: float = 1e-7,
                  move: float = 0.2, callback: Callable | None = None) -> ShapeResult:  # fmt: skip
    """The shape of a periodic cell that minimizes ``objective(tensor, fraction)`` under
    ``constraint(tensor, fraction) <= 0`` for each of the ``constraints``: functions of
    the effective tensor of the cell (Voigt's notation) and of its fraction of
    material — for instance ``lambda c, f: -plane_constants(c)["bulk_modulus"]`` with
    ``lambda c, f: f - 0.6``. ``linear = (A, b)`` adds the constraints ``A x <= b`` on
    the variables (sizes of the geometry: the control points are linear in them); they
    follow the others in the ``constraints`` of the result, as ``A x - b``. The
    variables start from ``x0`` (zeros by default: the design as it is)."""
    t0 = time.perf_counter()
    functions = [objective, *constraints]
    rows = np.zeros((0, design.n)), np.zeros(0)
    if linear is not None:
        rows = np.atleast_2d(np.asarray(linear[0], dtype=float)), np.atleast_1d(linear[1])
        if rows[0].shape != (len(rows[1]), design.n):
            raise ValueError(f"linear: (A, b) with one row of {design.n} coefficients per bound")

    def evaluate(x):
        try:
            response = cell_response(design, x, materials, physics=physics, plane=plane)
        except FoldedDesign:
            raise
        except ValueError as exc:
            raise FoldedDesign(
                f"{exc}: the design folds for these variables; tighten their bounds"
            ) from None
        values, gradients = [], []
        for function in functions:
            value, d_tensor, d_fraction = _partials(function, response.tensor, response.fraction)
            values.append(value)
            gradients.append(
                np.einsum("ij,ijx->x", d_tensor, response.d_tensor)
                + d_fraction * response.d_fraction
            )
        return (
            values[0],
            gradients[0],
            np.concatenate([values[1:], rows[0] @ x - rows[1]]),
            np.concatenate([np.array(gradients[1:]).reshape(-1, design.n), rows[0]]),
        )

    lower, upper = design.bounds()
    x0 = np.zeros(design.n) if x0 is None else np.asarray(x0, dtype=float)
    x, history, converged, message = optimize(
        evaluate, x0, lower, upper, method=method, maxiter=maxiter, tolerance=tolerance,
        move=move, callback=callback,
    )  # fmt: skip
    response = cell_response(design, x, materials, physics=physics, plane=plane)
    values = [float(f(response.tensor, response.fraction)) for f in functions]
    return ShapeResult(
        x, design.design_model(x), design.analysis_model(x), values[0],
        np.concatenate([values[1:], rows[0] @ x - rows[1]]),
        len(history) - 1, converged, history, response.tensor, response.fraction,
        time.perf_counter() - t0, message,
    )  # fmt: skip


def optimize_structure(
    design: ShapeDesign,
    materials,
    loads: Callable,
    supports: Callable,
    *,
    measure: float | None = None,
    x0=None,
    plane: str = "strain",
    method: str = "slsqp",
    maxiter: int = 100,
    tolerance: float = 1e-7,
    move: float = 0.2,
    callback: Callable | None = None,
) -> ShapeResult:
    """The shape of a structure of smallest compliance under given loads
    (:func:`structure_response`) whose material has at most the area (volume)
    ``measure`` (by default that of the design as it is)."""
    t0 = time.perf_counter()
    x0 = np.zeros(design.n) if x0 is None else np.asarray(x0, dtype=float)
    if measure is None:
        measure = Analysis(design.analysis_model(np.zeros(design.n)), materials).measure()
    start = structure_response(design, x0, materials, loads, supports, plane=plane)

    def evaluate(x):
        try:
            r = structure_response(design, x, materials, loads, supports, plane=plane)
        except FoldedDesign:
            raise
        except ValueError as exc:
            raise FoldedDesign(
                f"{exc}: the design folds for these variables; tighten their bounds"
            ) from None
        return (
            r.compliance / start.compliance,
            r.d_compliance / start.compliance,
            np.array([r.measure / measure - 1.0]),
            r.d_measure[None, :] / measure,
        )

    lower, upper = design.bounds()
    x, history, converged, message = optimize(
        evaluate, x0, lower, upper, method=method, maxiter=maxiter, tolerance=tolerance,
        move=move, callback=callback,
    )  # fmt: skip
    final = structure_response(design, x, materials, loads, supports, plane=plane)
    return ShapeResult(
        x, design.design_model(x), design.analysis_model(x), final.compliance,
        np.array([final.measure / measure - 1.0]), len(history) - 1, converged, history,
        None, None, time.perf_counter() - t0, message,
    )  # fmt: skip


__all__ = [
    "METHODS",
    "CellResponse",
    "FoldedDesign",
    "ShapeDesign",
    "ShapeResult",
    "StructureResponse",
    "cell_response",
    "energy_gradients",
    "measure_gradient",
    "mirror",
    "optimize",
    "optimize_cell",
    "optimize_structure",
    "rotation",
    "structure_response",
]
