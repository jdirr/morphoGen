"""The mixed finite elements of micromorphic media and their Newton solver.

The displacement ``u`` is interpolated on the ten nodes of quadratic tetrahedra and the
``n_g`` generalized degrees of freedom (the components of the microdeformation, a scalar
micromorphic variable...) on their four corners: one order lower, so that the relative
deformation ``∇u - χ`` can vanish (a penalty on it with equal orders locks). The degrees
of freedom are numbered node by node (three components of ``u``, then ``n_g`` on a
corner); the sparse pattern of the matrix follows from the node graph of the mesh, and
the element matrices, computed by chunks of elements in parallel, are added into it in
place (one pass, no list of triplets: an element of the full micromorphic medium has 66
degrees of freedom, 4356 entries).

Elements are integrated at the four points of the degree-two rule of the tetrahedron
(exact for the products of the gradients of quadratic functions and of linear
functions), where the behaviours keep their internal variables.

The solver (:meth:`MixedProblem.run`) grows prescribed degrees of freedom with a load
factor; Newton's method with a line search, the steps divided when it fails.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit

from morphogen.homogenization.fe import GAUSS, node_pattern
from morphogen.mesh.tetra import TET_EDGES, TetMesh

#: Gauss points (barycentric coordinates) and weights of the elements
POINTS, WEIGHTS = GAUSS[2]
#: Elements whose matrices are computed together (memory: 8 x 66² bytes each)
CHUNK = 2048


@njit(cache=True)
def _dof_pattern(ptr, idx, nd, offset):
    """The CSR pattern of the degrees of freedom from the node graph (``ptr``, ``idx``):
    the position of each neighbour's block in the rows of a node, the start and length
    of the rows of each node, the row pointers and the column indices."""
    n = len(nd)
    cum = np.empty(len(idx), dtype=np.int64)
    length = np.empty(n, dtype=np.int64)
    for v in range(n):
        c = 0
        for q in range(ptr[v], ptr[v + 1]):
            cum[q] = c
            c += nd[idx[q]]
        length[v] = c
    base = np.empty(n + 1, dtype=np.int64)
    base[0] = 0
    for v in range(n):
        base[v + 1] = base[v] + nd[v] * length[v]
    n_dofs = offset[n]
    indptr = np.empty(n_dofs + 1, dtype=np.int64)
    indices = np.empty(base[n], dtype=np.int64)
    for v in range(n):
        for i in range(nd[v]):
            start = base[v] + i * length[v]
            indptr[offset[v] + i] = start
            pos = start
            for q in range(ptr[v], ptr[v + 1]):
                w = idx[q]
                for j in range(nd[w]):
                    indices[pos] = offset[w] + j
                    pos += 1
    indptr[n_dofs] = base[n]
    return cum, base, length, indptr, indices


@njit(cache=True)
def _scatter(elements, first, ke, fe, lnode, comp, offset, base, length, ptr, idx, cum, data, rhs):
    """Add element vectors (and matrices, if ``data`` is not empty) of the elements
    ``first, first + 1...`` into the global vector (matrix data)."""
    n_loc = len(lnode)
    qpos = np.empty((10, 10), dtype=np.int64)
    with_matrix = len(data) > 0
    for e in range(fe.shape[0]):
        el = elements[first + e]
        if with_matrix:
            for a in range(10):
                v = el[a]
                for c in range(10):
                    w = el[c]
                    lo = ptr[v]
                    hi = ptr[v + 1] - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if idx[mid] < w:
                            lo = mid + 1
                        else:
                            hi = mid
                    qpos[a, c] = lo
        for al in range(n_loc):
            a = lnode[al]
            v = el[a]
            rhs[offset[v] + comp[al]] += fe[e, al]
            if with_matrix:
                row = base[v] + comp[al] * length[v]
                for be in range(n_loc):
                    data[row + cum[qpos[a, lnode[be]]] + comp[be]] += ke[e, al, be]


class MixedSpace:
    """The degrees of freedom of a mesh (made quadratic if it is linear, with straight
    edges): three displacements per node, ``n_g`` generalized ones per corner node (see
    the module)."""

    def __init__(self, mesh: TetMesh, n_g: int):
        mesh = mesh.quadratic().straight()
        self.mesh = mesh
        self.n_g = int(n_g)
        n = mesh.n_nodes
        self.corner = np.zeros(n, dtype=bool)
        self.corner[np.unique(mesh.elements[:, :4])] = True
        self.nd = 3 + self.n_g * self.corner.astype(np.int64)
        self.offset = np.concatenate([[0], np.cumsum(self.nd)]).astype(np.int64)
        self.n_dofs = int(self.offset[-1])
        self._ptr, self._idx = node_pattern(mesh.elements, n)
        self._cum, self._base, self._length, indptr, indices = _dof_pattern(
            self._ptr, self._idx, self.nd, self.offset
        )
        small = indptr[-1] < 2**31 - 1
        self._indptr = indptr.astype(np.int32) if small else indptr
        self._indices = indices.astype(np.int32) if small else indices
        n_loc = 30 + 4 * self.n_g
        self.lnode = np.array(
            [a // 3 if a < 30 else (a - 30) // max(self.n_g, 1) for a in range(n_loc)]
        )
        self.comp = np.array(
            [a % 3 if a < 30 else 3 + (a - 30) % max(self.n_g, 1) for a in range(n_loc)]
        )
        self.elements = np.ascontiguousarray(mesh.elements, dtype=np.int64)

    @property
    def n_local(self) -> int:
        return 30 + 4 * self.n_g

    def dofs(self, nodes, components) -> np.ndarray:
        """The degrees of freedom (nodes x components) of components (0-2: ``u``, 3...:
        the generalized ones, on corner nodes only) of nodes."""
        nodes = np.atleast_1d(np.asarray(nodes, dtype=np.int64))
        comps = np.atleast_1d(np.asarray(components, dtype=np.int64))
        if np.any(comps >= self.nd[nodes][:, None]):
            raise ValueError("a generalized degree of freedom on a node that is not a corner")
        return self.offset[nodes][:, None] + comps[None, :]

    def new_matrix_data(self) -> np.ndarray:
        return np.zeros(len(self._indices))

    def matrix(self, data) -> sp.csr_matrix:
        return sp.csr_matrix((data, self._indices, self._indptr), shape=(self.n_dofs, self.n_dofs))

    def add(self, first, ke, fe, data, rhs) -> None:
        """Add the matrices ``ke`` and vectors ``fe`` of the elements ``first...``."""
        _scatter(self.elements, first, ke, fe, self.lnode, self.comp, self.offset, self._base,
                 self._length, self._ptr, self._idx, self._cum, data, rhs)  # fmt: skip

    def displacement(self, values) -> np.ndarray:
        """The displacement (nodes x 3) of a vector of degrees of freedom."""
        return np.asarray(values)[self.offset[:-1, None] + np.arange(3)]

    def generalized(self, values) -> np.ndarray:
        """The generalized degrees of freedom at every node (nodes x n_g): those of the
        corners, interpolated linearly at the mid-edge nodes."""
        values = np.asarray(values)
        out = np.full((self.mesh.n_nodes, self.n_g), np.nan)
        corners = np.flatnonzero(self.corner)
        out[corners] = values[self.offset[corners][:, None] + 3 + np.arange(self.n_g)]
        el = self.elements
        for k, (i, j) in enumerate(TET_EDGES):
            out[el[:, 4 + k]] = 0.5 * (out[el[:, i]] + out[el[:, j]])
        return out


@dataclass
class PathResult:
    """The steps of a problem: load factors, reactions (steps x prescribed sets x
    components of each set, summed over its nodes), Newton iterations, the final degrees
    of freedom and internal variables and, with ``fields``, the degrees of freedom and the
    internal variables of the steps."""

    loads: np.ndarray
    reactions: list
    iterations: np.ndarray
    values: np.ndarray
    state: np.ndarray
    fields: list = field(default_factory=list, repr=False)
    states: list = field(default_factory=list, repr=False)
    time_s: float = 0.0


class MixedProblem:
    """A boundary value problem on a :class:`MixedSpace`; subclasses give
    ``_assemble(values, state, tangent) -> (residual, matrix data or None, new state)``
    (the residual: the internal forces, no external ones) and ``_initial_state()``."""

    space: MixedSpace

    def _assemble(self, values, state, tangent):  # pragma: no cover - abstract
        raise NotImplementedError

    def _initial_state(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def run(
        self,
        prescribed,
        load: float = 1.0,
        *,
        steps: int = 10,
        tol: float = 1e-6,
        max_newton: int = 25,
        max_cuts: int = 12,
        fields: bool = False,
        refactor: float = 0.0,
        ordering: str = "COLAMD",
        duration: float = 1.0,
        predictor: bool = False,
    ) -> PathResult:
        """Grow the prescribed degrees of freedom to ``load`` times their values in
        ``steps`` steps (divided when Newton's method fails): entries ``(nodes,
        components, value)``, components 0-2 for the displacement and 3... for the
        generalized ones (corner nodes only), the value a number or an array of the
        nodes x components (0: held). The factors of the tangent (SuperLU, column
        ``ordering``) are kept from an iteration to the next while each divides the
        residual by more than ``1 / refactor`` (a modified Newton method for large problems,
        whose cost the factorization dominates; by default 0, Newton's method: stale
        tangents may miss the bifurcation of a softening body). ``duration``: the time of
        the whole load, for viscous behaviours (a step lasts its fraction of it: the
        attribute ``_dt`` of the problem during its iterations). ``predictor``: the
        iterations of a step start from the free degrees of freedom that the last
        converged tangent predicts for the prescribed increment (``K_ff Δu_f = -K_fd
        Δu_d``), instead of the converged ones, which strain the elements along the
        loaded boundary alone: there they may yield, and the iterations of a plastic body
        stall on that first guess; when they fail, the same step from the converged ones
        (near a bifurcation, the tangent may predict another branch)."""
        t0 = time.perf_counter()
        space = self.space
        dofs, values, sets = [], [], []
        for nodes, comps, value in prescribed:
            d = space.dofs(nodes, comps)
            dofs.append(d.ravel())
            values.append(np.broadcast_to(np.asarray(value, dtype=float), d.shape).ravel())
            sets.append(d)
        dofs = np.concatenate(dofs)
        values = np.concatenate(values)
        if len(np.unique(dofs)) != len(dofs):
            raise ValueError("prescribed: a degree of freedom is given twice")
        free = np.setdiff1d(np.arange(space.n_dofs), dofs)
        u = np.zeros(space.n_dofs)
        state = self._initial_state()
        loads, reactions, its, snaps = [0.0], [[np.zeros(s.shape[1]) for s in sets]], [0], []
        kept = []
        if fields:
            snaps.append(u.copy())
            kept.append(state.copy())
        lam, step, cuts = 0.0, load / steps, 0
        last = None  # with the predictor: the factors of K_ff and K_fd of the last tangent
        plain = False  # the predictor failed: plain guesses until the next convergence
        while lam < load * (1 - 1e-12):
            step = min(step, load - lam)
            target = lam + step
            trial = u.copy()
            trial[dofs] = target * values
            self._dt = step / load * float(duration)
            if predictor and last is None:  # the tangent of the initial state
                rows = space.matrix(self._assemble(u, state, True)[1])[free]
                try:
                    last = (spla.splu(rows[:, free].tocsc(), permc_spec=ordering), rows[:, dofs])
                except RuntimeError:  # singular: no predictor
                    last = ()
            # a tangent nearly singular (a bifurcation) may predict a wrong branch: after a
            # failure from the predictor, the same step from the converged values
            guessed = bool(predictor and last) and not plain
            if guessed:
                guess = last[0].solve(-(last[1] @ ((target - lam) * values)))
                if np.all(np.isfinite(guess)):
                    trial[free] += guess
            ok, scale, lu, previous = False, None, None, None
            for iterations in range(1, max_newton + 1):
                fresh = lu is None or previous is None
                r, data, new_state = self._assemble(trial, state, True)
                norm = float(np.linalg.norm(r[free]))
                if not np.isfinite(norm):
                    break
                if scale is None:
                    scale = max(float(np.linalg.norm(r[dofs])), norm, 1e-30)
                if norm <= tol * scale and iterations > 1:
                    ok = True
                    break
                if fresh or norm > refactor * previous:
                    try:
                        rows = space.matrix(data)[free]
                        lu = spla.splu(rows[:, free].tocsc(), permc_spec=ordering)
                    except RuntimeError:  # singular
                        break
                    coupling = rows[:, dofs] if predictor else None
                du = lu.solve(-r[free])
                if not np.all(np.isfinite(du)):
                    break
                alpha = 1.0
                for _ in range(4):  # a backtracking line search on the residual
                    probe = trial.copy()
                    probe[free] += alpha * du
                    r_probe, _, _ = self._assemble(probe, state, False)
                    probe_norm = float(np.linalg.norm(r_probe[free]))
                    if probe_norm < norm or alpha < 0.2:
                        break
                    alpha *= 0.5
                if not np.isfinite(probe_norm):
                    break
                trial[free] += alpha * du
                scale = max(scale, float(np.linalg.norm(r_probe[dofs])))
                previous = norm
            if not ok:
                if guessed:
                    plain = True
                    continue
                cuts += 1
                if cuts > max_cuts:
                    raise RuntimeError(f"micromorphic problem: no convergence at load {target:g}")
                step *= 0.5
                continue
            u, state, lam, cuts = trial, new_state, target, 0
            if predictor:
                last, plain = (lu, coupling), False
            loads.append(lam)
            reactions.append([r[s].sum(axis=0) for s in sets])
            its.append(iterations)
            if fields:
                snaps.append(u.copy())
                kept.append(state.copy())
            if iterations <= 4:
                step = min(step * 1.5, load / steps)
        return PathResult(np.array(loads), reactions, np.array(its), u, state, snaps, kept,
                          time.perf_counter() - t0)  # fmt: skip
