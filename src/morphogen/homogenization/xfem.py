"""Extended finite elements for homogenization: the interfaces of a morphology cut the
elements of a regular grid instead of being meshed or drawn as stairs of voxels.

A two-phase morphology is given by its level set ``phi`` at the nodes of a regular grid
of the periodic box (:class:`LevelSetSample`; negative in the phase 1). The grid is cut
into six linear tetrahedra per cell, in which the level set is interpolated linearly: the
interface is a plane within each element. Following Moës, Cloirec, Cartraud & Remacle
(2003), the displacement (or the potential of a transport problem) is enriched where an
interface between two materials cuts the elements::

    u(x) = sum_i N_i(x) u_i + sum_(j enriched) N_j(x) psi(x) a_j,
    psi(x) = sum_i N_i(x) |phi_i| - |sum_i N_i(x) phi_i|,

the ridge function ``psi`` continuous, with a kink on the interface (the jump of the
normal strain that the continuity of the tractions asks for) and zero in the elements
that the interface does not cut: no blending elements. The elements cut are integrated on
the sub-tetrahedra of their two sides, each with its material (four Gauss points each:
the stiffness is integrated exactly). Where one side is a void or an inert phase, the
elements are not enriched and only their solid part is integrated (Sukumar et al. 2001);
the nodes without any solid around them are left out.

The discretization keeps the convergence of conforming meshes without a mesher: a plane
interface of any orientation is represented exactly (laminates oblique to the grid are
solved to round-off, where voxels draw stairs), and curved interfaces to second order in
the size of the elements. The level set is moved away from the nodes it passes too close
to (``snap``, a fraction of the edge of the cells), which bounds the conditioning.

The problems are those of the periodic element (:class:`XFEMElasticProblem`,
:class:`XFEMThermalProblem`; the thermal one for every transport phenomenon): the
unknowns are the periodic fluctuations, standard and enriched, on the torus of the grid;
:func:`~morphogen.homogenization.apparent.homogenize` takes a :class:`LevelSetSample`
as it takes a mesh or an image. Loading paths with the nonlinear behaviours of the
phases are followed on the same discretization, in small or finite strains
(:class:`NonlinearXFEMProblem`): the integration points of the cut elements are those of
their sub-tetrahedra, each with the behaviour of its side; in finite strains the
formulation is the total Lagrangian one of the finite elements, the level set cutting the
reference configuration. Linear tetrahedra lock under
isochoric plastic flow: for plasticity prefer conforming quadratic meshes or the voxels
of the FFT solver (B-bar); damage, viscoelasticity and compressible behaviours do not
suffer from it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from morphogen.homogenization.core import (
    Loading,
    MacroscopicLoadings,
    PhaseStatistics,
    Solution,
    boundary_condition,
)
from morphogen.homogenization.elements import tensor_to_voigt, voigt_to_tensor
from morphogen.homogenization.materials import Material, transport_physics
from morphogen.homogenization.nonlinear import _PathSolver
from morphogen.homogenization.solvers import NEGLIGIBLE, LinearSolver, SolveInfo
from morphogen.mesh.tetra import TetMesh

log = logging.getLogger("morphogen")

#: Barycentric coordinates of the four Gauss points of a tetrahedron (degree 2)
_A, _B = 0.5854101966249685, 0.1381966011250105
_Q4 = np.array([[_A, _B, _B, _B], [_B, _A, _B, _B], [_B, _B, _A, _B], [_B, _B, _B, _A]])
#: Elements whose matrices are built at once (the memory of their strain operators)
_CHUNK = 2048
#: Voigt rows of the strain of a displacement component with a gradient: (row, component
#: of the displacement, component of the gradient)
_STRAIN = ((0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 1, 2), (3, 2, 1), (4, 0, 2), (4, 2, 0),
           (5, 0, 1), (5, 1, 0))  # fmt: skip


@dataclass
class LevelSetSample:
    """A two-phase morphology on the nodes of a regular grid of a periodic box: ``values``
    (nx x ny x nz), its level set at the nodes ``lower + (i, j, k) h`` (negative in the
    phase ``inside``, positive in the phase ``outside``; the node of index ``n`` along an
    axis is that of index 0, one period further)."""

    values: np.ndarray
    lower: tuple = (0.0, 0.0, 0.0)
    upper: tuple = (1.0, 1.0, 1.0)
    inside: int = 1
    outside: int = 0

    def __post_init__(self) -> None:
        self.values = np.ascontiguousarray(self.values, dtype=float)
        if self.values.ndim != 3 or min(self.values.shape) < 1:
            raise ValueError("LevelSetSample: the level set at the nodes of a grid (nx, ny, nz)")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("LevelSetSample: finite values of the level set")
        self.lower = tuple(float(v) for v in self.lower)
        self.upper = tuple(float(v) for v in self.upper)
        if not all(b > a for a, b in zip(self.lower, self.upper, strict=True)):
            raise ValueError("LevelSetSample: a box with positive edges")
        if self.inside == self.outside:
            raise ValueError("LevelSetSample: two different phases")

    @classmethod
    def of(cls, micro, shape=None, *, spacing: float | None = None) -> LevelSetSample:
        """The sample of a periodic microstructure with a level set (a model realization,
        an :class:`~morphogen.mesh.conforming.ImplicitBody`): ``shape`` cells along the
        longest edge of its box (or a shape per axis), or cells of edge ``spacing``."""
        from morphogen.domain import Domain, Grid

        domain: Domain = micro.domain
        if not domain.periodic:
            raise ValueError(
                "XFEM homogenization is periodic: a periodic microstructure (its level set "
                "on the torus of the grid)"
            )
        lengths = np.asarray(domain.lengths, dtype=float)
        if (shape is None) == (spacing is None):
            raise ValueError("LevelSetSample.of: give the shape of the grid or its spacing")
        if spacing is not None:
            n = np.maximum(1, np.rint(lengths / float(spacing))).astype(int)
        elif np.ndim(shape) == 0:
            n = np.maximum(1, np.rint(lengths / lengths.max() * int(shape))).astype(int)
        else:
            n = np.asarray(shape, dtype=int)
        h = lengths / n
        lower = np.asarray(domain.lower, dtype=float)
        sampling = Domain(size=tuple(lengths), origin=tuple(lower - h / 2), periodic=True)
        try:
            field = micro.level_set(grid=Grid(sampling, tuple(int(k) for k in n)))
        except NotImplementedError:
            raise ValueError(
                f"{type(micro).__name__}: no level-set function (tessellations are meshed "
                "grain by grain or solved on voxels)"
            ) from None
        return cls(np.asarray(field.data, dtype=float), tuple(lower), tuple(lower + lengths))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values.shape

    def phase_fractions(self, snap: float = 1e-3) -> dict[int, float]:
        """The volume fractions of the two phases: those of the elements and of the parts
        of the elements that the interface cuts (the geometry that the problems solve)."""
        from morphogen.mesh.conforming import background_mesh

        n = np.asarray(self.shape)
        nodes, tets, _, _ = background_mesh(n, self.lower, self.upper)
        ijk = np.stack(np.unravel_index(np.arange(len(nodes)), n + 1), axis=1)
        wrap = ((ijk[:, 0] % n[0]) * n[1] + ijk[:, 1] % n[1]) * n[2] + ijk[:, 2] % n[2]
        limit = snap * float(self.spacing.min())
        phi = self.values.ravel()
        phi = np.where(np.abs(phi) < limit, np.where(phi < 0, -limit, limit), phi)
        x, pe = nodes[tets], phi[wrap[tets]]
        neg = (pe < 0).sum(axis=1)
        inside = float(_gradients(x[neg == 4])[1].sum())
        cut = neg % 4 != 0
        if cut.any():
            verts, side = subtetrahedra(x[cut], pe[cut])
            vol = np.abs(np.linalg.det(verts[:, :, 1:] - verts[:, :, :1])) / 6.0
            inside += float(vol[side < 0].sum())
        fraction = inside / self.volume
        return dict(sorted({self.inside: fraction, self.outside: 1.0 - fraction}.items()))

    @property
    def spacing(self) -> np.ndarray:
        return (np.asarray(self.upper) - np.asarray(self.lower)) / np.asarray(self.shape)

    @property
    def volume(self) -> float:
        return float(np.prod(np.asarray(self.upper) - np.asarray(self.lower)))


# -- geometry of the cut elements ------------------------------------------------------------


def _gradients(x: np.ndarray):
    """Gradients of the shape functions (m x 4 x 3) and volumes of linear tetrahedra of
    corners ``x`` (m x 4 x 3)."""
    dm = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)
    inv = np.linalg.inv(dm)
    grads = np.concatenate([-inv.sum(axis=1, keepdims=True), inv], axis=1)
    return grads, np.abs(np.linalg.det(dm)) / 6.0


def subtetrahedra(x: np.ndarray, phi: np.ndarray):
    """The tetrahedra of corners ``x`` (m x 4 x 3) cut by the zero of the linear
    interpolant of their nodal values ``phi`` (m x 4, none zero, of both signs): the
    corners of their sub-tetrahedra (m x 6 x 4 x 3) and the sign of ``phi`` in each (m x
    6; 0 for the two unused ones of an element cut with one node alone on its side, which
    has a tetrahedron and a prism of three)."""
    m = len(x)
    neg = phi < 0
    count = neg.sum(axis=1)
    if np.any((count == 0) | (count == 4)):
        raise ValueError("subtetrahedra: elements whose nodal values have both signs")
    verts = np.zeros((m, 6, 4, 3))
    side = np.zeros((m, 6))

    def ordered(sel, first):
        order = np.argsort(~first, axis=1, kind="stable")
        return (np.take_along_axis(x[sel], order[:, :, None], axis=1),
                np.take_along_axis(phi[sel], order, axis=1))  # fmt: skip

    for alone, lone_side in ((1, -1.0), (3, 1.0)):  # one node alone on the side lone_side
        sel = np.flatnonzero(count == alone)
        if not sel.size:
            continue
        xs, ps = ordered(sel, neg[sel] if lone_side < 0 else ~neg[sel])
        t = ps[:, :1] / (ps[:, :1] - ps[:, 1:])
        p = xs[:, :1] + t[:, :, None] * (xs[:, 1:] - xs[:, :1])
        o = xs[:, 1:]
        verts[sel, 0] = np.stack([xs[:, 0], p[:, 0], p[:, 1], p[:, 2]], axis=1)
        verts[sel, 1] = np.stack([o[:, 0], o[:, 1], o[:, 2], p[:, 0]], axis=1)
        verts[sel, 2] = np.stack([o[:, 1], o[:, 2], p[:, 0], p[:, 1]], axis=1)
        verts[sel, 3] = np.stack([o[:, 2], p[:, 0], p[:, 1], p[:, 2]], axis=1)
        side[sel, 0] = lone_side
        side[sel, 1:4] = -lone_side
    sel = np.flatnonzero(count == 2)
    if sel.size:
        xs, ps = ordered(sel, ~neg[sel])  # a, b positive, then c, d negative

        def cut(i, j):
            t = ps[:, i] / (ps[:, i] - ps[:, j])
            return xs[:, i] + t[:, None] * (xs[:, j] - xs[:, i])

        a, b, c, d = (xs[:, k] for k in range(4))
        pac, pad, pbc, pbd = cut(0, 2), cut(0, 3), cut(1, 2), cut(1, 3)
        pieces = ((a, pac, pad, b), (pac, pad, b, pbc), (pad, b, pbc, pbd),
                  (c, pac, pbc, d), (pac, pbc, d, pad), (pbc, d, pad, pbd))  # fmt: skip
        for k, piece in enumerate(pieces):
            verts[sel, k] = np.stack(piece, axis=1)
        side[sel, :3] = 1.0
        side[sel, 3:] = -1.0
    return verts, side


class _Cut:
    """The elements cut by the interface: their corners ``x`` (m x 4 x 3), nodal level
    sets ``phi``, gradients ``grads`` and, at their 24 integration points (four per
    sub-tetrahedron; zero weights for the unused ones), the weights ``w``, the sides
    ``side`` (the sign of the level set), the shape functions ``n`` (m x 24 x 4) and the
    ridge function ``psi``; ``gpsi`` (m x 2 x 3): the gradient of the ridge on the sides
    +1 and -1."""

    def __init__(self, x: np.ndarray, phi: np.ndarray):
        m = len(x)
        self.x, self.phi = x, phi
        self.grads, self.volume = _gradients(x)
        verts, side = subtetrahedra(x, phi)
        edges = verts[:, :, 1:] - verts[:, :, :1]
        vol = np.abs(np.linalg.det(edges)) / 6.0
        pts = np.einsum("qv,msvd->msqd", _Q4, verts).reshape(m, 24, 3)
        self.w = np.repeat(vol / 4.0, 4, axis=1)
        self.side = np.repeat(side, 4, axis=1)
        lam = np.einsum("mad,mqd->mqa", self.grads[:, 1:], pts - x[:, None, 0])
        self.n = np.concatenate([1.0 - lam.sum(axis=2, keepdims=True), lam], axis=2)
        # |phi_a| - s phi_a on the side s: the nodal values of the ridge on that side
        coef = np.stack([np.abs(phi) - phi, np.abs(phi) + phi], axis=1)  # sides +1, -1
        self.gpsi = np.einsum("msa,mad->msd", coef, self.grads)
        which = (self.side < 0).astype(int)  # 0 on the side +1 (and unused), 1 on the side -1
        self.psi = np.einsum("mqa,mqa->mq", self.n, np.take_along_axis(
            coef, which[:, :, None].repeat(4, axis=2), axis=1))  # fmt: skip
        self.which = which

    def __len__(self) -> int:
        return len(self.x)

    def functions(self, sl: slice, enriched: bool) -> np.ndarray:
        """The gradients (m x 24 x functions x 3) of the functions of the elements ``sl``
        at their points: the four shape functions, then (``enriched``) their products by
        the ridge."""
        g = self.grads[sl]
        std = np.broadcast_to(g[:, None], (g.shape[0], 24, 4, 3))
        if not enriched:
            return std
        gpsi = np.take_along_axis(self.gpsi[sl], self.which[sl][:, :, None].repeat(3, axis=2),
                                  axis=1)  # fmt: skip
        enr = (
            self.psi[sl][:, :, None, None] * g[:, None]
            + self.n[sl][:, :, :, None] * gpsi[:, :, None, :]
        )
        return np.concatenate([std, enr], axis=2)


def _operator(g: np.ndarray, nd: int) -> np.ndarray:
    """The strain operators (..., 6, 3 functions) of displacements, or the gradient
    operators (..., 3, functions) of a potential, from the gradients ``g`` (..., functions,
    3) of their functions; the unknowns of a function are consecutive."""
    if nd == 1:
        return np.swapaxes(g, -1, -2)
    nf = g.shape[-2]
    b = np.zeros((*g.shape[:-2], 6, 3 * nf))
    for row, comp, axis in _STRAIN:
        b[..., row, comp::3] = g[..., axis]
    return b


def _gradient_operator(g: np.ndarray) -> np.ndarray:
    """The operators (..., 9, 3 functions) of the displacement gradients ``H_iJ`` (stored
    row by row, as :mod:`morphogen.homogenization.finite`) of displacements, from the
    gradients ``g`` (..., functions, 3) of their functions."""
    nf = g.shape[-2]
    b = np.zeros((*g.shape[:-2], 9, 3 * nf))
    for i in range(3):
        for j in range(3):
            b[..., 3 * i + j, i::3] = g[..., j]
    return b


# -- problems ----------------------------------------------------------------------------------


class _XFEMProblem(MacroscopicLoadings):
    """The periodic problem of a :class:`LevelSetSample` (see the module); ``snap``: the
    level set is moved to this fraction of the edge of the cells from the nodes it passes
    closer to; ``solver``: the linear solver (:mod:`morphogen.homogenization.solvers`)."""

    physics = ""
    nd = 0
    bc = "periodic"

    def __init__(self, sample: LevelSetSample, materials: dict[int, Material],
                 bc: str = "periodic", *, solver: str = "auto", tol: float = 1e-8,
                 maxiter: int | None = None, snap: float = 1e-3,
                 physics: str | None = None):  # fmt: skip
        if not isinstance(sample, LevelSetSample):
            raise TypeError("the XFEM solver needs a LevelSetSample (LevelSetSample.of)")
        if boundary_condition(bc) != "periodic":
            raise ValueError(
                "the XFEM solver of level sets uses periodic boundary conditions; for kubc, "
                "mixed or subc conditions, mesh the microstructure (morphogen.mesh.conforming)"
            )
        if physics is not None:
            if transport_physics(physics) != (self.nd == 1):
                kind = "a transport phenomenon" if self.nd == 1 else "'elastic'"
                raise ValueError(f"{type(self).__name__}: physics {kind}, got {physics!r}")
            self.physics = physics
        if not 0.0 < snap < 0.5:
            raise ValueError("snap: a fraction of the edge of the cells in (0, 0.5)")
        missing = [p for p in (sample.inside, sample.outside) if p not in materials]
        if missing:
            raise ValueError(f"no material for phase(s) {missing}")
        self.sample = sample
        self.materials = materials
        self.box_volume = sample.volume
        self.tol, self.maxiter, self.solver_method = float(tol), maxiter, solver
        self.nc = 6 if self.nd == 3 else 3
        self._units: dict = {}
        self._solver = None
        self._build(float(snap))

    # -- set-up ------------------------------------------------------------------------

    def _law(self, material: Material):
        """The matrix (stiffness or conductivity) and the thermal strain of a phase; None
        for a phase that carries nothing."""
        if material.inert(self.physics):
            return None
        if not material.has(self.physics):
            raise ValueError(f"a phase has no parameters for the physics {self.physics!r}")
        d = np.asarray(material.matrix(self.physics), dtype=float)
        eigen = np.zeros(self.nc)
        if self.nd == 3:
            eigen = tensor_to_voigt(material.expansion_tensor(), strain=True)
        return d, eigen

    def _build(self, snap: float) -> None:
        from morphogen.mesh.conforming import background_mesh

        s = self.sample
        n = np.asarray(s.shape)
        h = s.spacing
        nodes, tets, _, _ = background_mesh(n, s.lower, s.upper)
        g = n + 1
        ijk = np.stack(np.unravel_index(np.arange(len(nodes)), g), axis=1)
        wrap = ((ijk[:, 0] % n[0]) * n[1] + ijk[:, 1] % n[1]) * n[2] + ijk[:, 2] % n[2]
        self.n_nodes = int(np.prod(n))
        phi = s.values.ravel().copy()
        limit = snap * float(h.min())
        phi = np.where(np.abs(phi) < limit, np.where(phi < 0, -limit, limit), phi)
        self.phi = phi
        x = nodes[tets]
        torus = wrap[tets]
        pe = phi[torus]
        neg = (pe < 0).sum(axis=1)
        laws = {1.0: self._law(self.materials[s.outside]),
                -1.0: self._law(self.materials[s.inside])}  # fmt: skip
        self.laws = laws
        self.empty = all(law is None for law in laws.values())
        whole = neg % 4 == 0
        cut = np.flatnonzero(~whole)
        both = laws[1.0] is not None and laws[-1.0] is not None
        #: the uncut elements that carry a material: corners, nodes on the torus, side
        side = np.where(neg[whole] == 4, -1.0, 1.0)
        keep = np.where(side < 0, laws[-1.0] is not None, laws[1.0] is not None)
        self.el_x, self.el_nodes, self.el_side = x[whole][keep], torus[whole][keep], side[keep]
        self.el_grads, self.el_volume = _gradients(self.el_x)
        #: the cut elements: enriched between two materials, their solid part otherwise
        self.cut = _Cut(x[cut], pe[cut]) if cut.size and not self.empty else None
        self.cut_nodes = torus[cut]
        self.enriched = bool(both and cut.size)
        self.enr_index = np.full(self.n_nodes, -1, dtype=np.int64)
        if self.enriched:
            ids = np.unique(self.cut_nodes)
            self.enr_index[ids] = np.arange(ids.size)
        self.n_enriched = int((self.enr_index >= 0).sum())
        self.n_dofs = self.nd * (self.n_nodes + self.n_enriched)
        # the background mesh of the fields: each element in the phase of most of its volume
        phase = np.where(neg >= 2, s.inside, s.outside)
        self.mesh = TetMesh(nodes, tets, phase, np.asarray(s.lower), np.asarray(s.upper),
                            periodic=True)  # fmt: skip
        self._wrap, self._whole, self._keep, self._cut_ids = wrap, whole, keep, cut
        if not self.empty:
            self._assemble()

    def _element_dofs(self, nodes: np.ndarray, enriched: bool) -> np.ndarray:
        """The unknowns of elements of nodes ``nodes`` (m x 4): those of their four
        standard functions, then of the enriched ones."""
        nd = self.nd
        comp = np.arange(nd)
        std = (nd * nodes[:, :, None] + comp).reshape(len(nodes), -1)
        if not enriched:
            return std
        enr = (nd * (self.n_nodes + self.enr_index[nodes])[:, :, None] + comp).reshape(
            len(nodes), -1
        )
        return np.concatenate([std, enr], axis=1)

    def _cut_weights(self, sl: slice) -> dict:
        """The weights of the points of the cut elements ``sl`` on each side that carries a
        material."""
        c = self.cut
        return {v: np.where(c.side[sl] == v, c.w[sl], 0.0) for v, law in self.laws.items()
                if law is not None}  # fmt: skip

    def _assemble(self) -> None:
        nd = self.nd
        n = self.n_dofs
        rows, cols, vals = [], [], []
        # the uncut elements: constant strains
        b = _operator(self.el_grads, nd)
        for v, law in self.laws.items():
            sel = np.flatnonzero(self.el_side == v)
            if law is None or not sel.size:
                continue
            for lo in range(0, sel.size, 16 * _CHUNK):
                part = sel[lo : lo + 16 * _CHUNK]
                bt = np.swapaxes(b[part], 1, 2)
                ke = self.el_volume[part, None, None] * (bt @ (law[0] @ b[part]))
                dofs = self._element_dofs(self.el_nodes[part], False)
                rows.append(np.repeat(dofs, dofs.shape[1], axis=1).ravel())
                cols.append(np.tile(dofs, (1, dofs.shape[1])).ravel())
                vals.append(ke.ravel())
        if self.cut is not None:
            for lo in range(0, len(self.cut), _CHUNK):
                sl = slice(lo, lo + _CHUNK)
                bq = _operator(self.cut.functions(sl, self.enriched), nd)
                ke = 0.0
                m, nf = bq.shape[0], bq.shape[-1]
                for v, w in self._cut_weights(sl).items():
                    # sum_q w_q B_q^T D B_q, the points of each element stacked
                    db = (self.laws[v][0] @ bq) * w[:, :, None, None]
                    ke = ke + np.swapaxes(bq.reshape(m, -1, nf), 1, 2) @ db.reshape(m, -1, nf)
                dofs = self._element_dofs(self.cut_nodes[sl], self.enriched)
                rows.append(np.repeat(dofs, dofs.shape[1], axis=1).ravel())
                cols.append(np.tile(dofs, (1, dofs.shape[1])).ravel())
                vals.append(ke.ravel())
        k = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        ).tocsr()
        self.matrix = (0.5 * (k + k.T)).tocsr()
        diag = self.matrix.diagonal()
        # the unknowns that carry something: not those of the nodes in the voids
        active = diag > 1e-14 * diag.max()
        self._regularization = 0.0
        if any(law is None for law in self.laws.values()):
            # parts of the solid that nothing holds (cut off by the voids) move freely: a
            # small stiffness keeps the system definite without loading the others
            self._regularization = 1e-10 * float(diag.mean())
            shift = sp.diags(np.where(active, self._regularization, 0.0))
            self.matrix = (self.matrix + shift).tocsr()
        #: one node held against the translations of the periodic fluctuation
        held = int(np.flatnonzero(active[: nd * self.n_nodes : nd])[0])
        active[nd * held : nd * held + nd] = False
        self.free = np.flatnonzero(active)
        volume = float(self.el_volume.sum())
        if self.cut is not None:
            volume += float(sum(w.sum() for w in self._cut_weights(slice(None)).values()))
        self.material_fraction = volume / self.box_volume

    def _linear_solver(self) -> LinearSolver:
        if self._solver is None:
            reduced = self.matrix[self.free][:, self.free].tocsr()
            near_null = None
            if self.nd == 3 and reduced.shape[0] > 0:
                near_null = self._rigid_modes()[self.free]
            self._solver = LinearSolver(reduced, method=self.solver_method, tol=self.tol,
                                        maxiter=self.maxiter, near_null=near_null)  # fmt: skip
        return self._solver

    def _rigid_modes(self) -> np.ndarray:
        """Translations and rotations of the standard unknowns (zero on the enriched): the
        near-null space of the multigrid."""
        nd, nn = self.nd, self.n_nodes
        x = np.zeros((nn, 3))
        x[self._wrap] = self.mesh.nodes
        x = x - x.mean(axis=0)
        modes = np.zeros((self.n_dofs, 6))
        for c in range(3):
            modes[c : nd * nn : nd, c] = 1.0
        modes[1 : nd * nn : nd, 3], modes[2 : nd * nn : nd, 3] = -x[:, 2], x[:, 1]
        modes[2 : nd * nn : nd, 4], modes[0 : nd * nn : nd, 4] = -x[:, 0], x[:, 2]
        modes[0 : nd * nn : nd, 5], modes[1 : nd * nn : nd, 5] = -x[:, 1], x[:, 0]
        return modes

    # -- loads and averages --------------------------------------------------------------

    def _affine(self, x: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """The values (m x 4 nd) of the affine field of the macroscopic strain (gradient)
        ``x`` at the corners of elements."""
        if self.nd == 1:
            return corners @ x
        e = voigt_to_tensor(x, strain=True)
        return (corners @ e.T).reshape(len(corners), -1)

    def _generalized(self, q: np.ndarray, x: np.ndarray):
        """The strains (gradients) of the uncut elements (m x nc) and of the points of the
        cut ones (m x 24 x nc), for the fluctuation ``q`` and the macroscopic ``x``."""
        nd = self.nd
        b = _operator(self.el_grads, nd)
        ue = q[self._element_dofs(self.el_nodes, False)]
        whole = np.einsum("mra,ma->mr", b, ue) + x
        cut = None
        if self.cut is not None:
            cut = np.empty((len(self.cut), 24, self.nc))
            for lo in range(0, len(self.cut), _CHUNK):
                sl = slice(lo, lo + _CHUNK)
                bq = _operator(self.cut.functions(sl, self.enriched), nd)
                ue = q[self._element_dofs(self.cut_nodes[sl], self.enriched)]
                cut[sl] = np.einsum("mqra,ma->mqr", bq, ue) + x
        return whole, cut

    def _load(self, stress_of) -> tuple[np.ndarray, float]:
        """The right-hand side ``-int B^T s`` of the stresses ``s = stress_of(law)`` of
        the phases (uniform in each: those of a macroscopic strain or of a change of
        temperature), and its scale without the cancellations between elements."""
        nd = self.nd
        f = np.zeros(self.n_dofs)
        scale = 0.0
        b = _operator(self.el_grads, nd)
        for v, law in self.laws.items():
            sel = np.flatnonzero(self.el_side == v)
            if law is None:
                continue
            s = stress_of(law)
            if sel.size:
                fe = self.el_volume[sel, None] * np.einsum("mra,r->ma", b[sel], s)
                np.add.at(f, self._element_dofs(self.el_nodes[sel], False), -fe)
                scale += float(np.sum(fe * fe))
            if self.cut is not None:
                for lo in range(0, len(self.cut), _CHUNK):
                    sl = slice(lo, lo + _CHUNK)
                    bq = _operator(self.cut.functions(sl, self.enriched), nd)
                    w = self._cut_weights(sl)[v]
                    fe = np.einsum("mq,mqra,r->ma", w, bq, s)
                    np.add.at(f, self._element_dofs(self.cut_nodes[sl], self.enriched), -fe)
                    scale += float(np.sum(fe * fe))
        return f, float(np.sqrt(scale))

    def _solve(self, rhs: np.ndarray, scale: float):
        q = np.zeros(self.n_dofs)
        if float(np.linalg.norm(rhs[self.free])) <= NEGLIGIBLE * scale:
            return q, SolveInfo("xfem", self.free.size, 0, 0.0, 0.0, 0.0)
        solver = self._linear_solver()
        q[self.free], info = solver.solve(rhs[self.free])
        info.method = f"xfem-{info.method}"
        return q, info

    def _sign(self) -> float:
        """The flux of a transport is ``-K grad``; a stress ``D eps``."""
        return 1.0 if self.nd == 3 else -1.0

    def _field(self, x: np.ndarray, dt: float = 0.0):
        """The periodic fluctuation (standard and enriched unknowns) of the macroscopic
        strain (gradient) ``x`` and the change of temperature ``dt``."""
        x = np.asarray(x, dtype=float)
        rhs, scale = self._load(lambda law: law[0] @ (x - dt * law[1]))
        return self._solve(rhs, scale)

    def _integrals(self, q: np.ndarray, x: np.ndarray, dt: float):
        """Per side: the volume, the integrals of the strains, stresses and energy
        density, and of their squares."""
        whole, cut = self._generalized(q, x)
        out = {}
        for v, law in self.laws.items():
            if law is None:
                continue
            d, eigen = law
            sel = self.el_side == v
            eps = [whole[sel]]
            wts = [self.el_volume[sel]]
            if cut is not None:
                w = self._cut_weights(slice(None))[v]
                on = w > 0
                eps.append(cut[on])
                wts.append(w[on])
            eps, wts = np.concatenate(eps), np.concatenate(wts)
            sig = self._sign() * (eps - dt * eigen) @ d
            energy = np.einsum("pi,pi->p", sig, eps - dt * eigen) * self._sign()
            out[v] = (wts, eps, sig, energy)
        return out

    def _dual(self, q, x, dt=0.0) -> np.ndarray:
        total = np.zeros(self.nc)
        for wts, _, sig, _ in self._integrals(q, np.asarray(x, dtype=float), dt).values():
            total += wts @ sig
        return total / self.box_volume

    def _evaluate(self, loading, q, x, dt, info, fields, t0) -> Solution:
        elastic = self.nd == 3
        x = np.asarray(x, dtype=float)
        parts = self._integrals(q, x, dt)
        total, energy, phases = np.zeros(self.nc), 0.0, []
        names = ("strain", "stress", "energy") if elastic else ("gradient", "flux", "dissipation")
        half = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) if elastic else 1.0
        for v, (wts, eps, sig, dens) in parts.items():
            vol = wts.sum()
            total += wts @ sig
            energy += wts @ dens
            values = (eps * half, sig, dens[:, None])
            mean = {n: wts @ a / vol for n, a in zip(names, values, strict=True)}
            std = {n: np.sqrt(np.maximum(wts @ (a * a) / vol - mean[n] ** 2, 0.0))
                   for n, a in zip(names, values, strict=True)}  # fmt: skip
            phase = self.sample.inside if v < 0 else self.sample.outside
            phases.append(PhaseStatistics(phase, float(vol / self.box_volume), mean, std))
        fld = self._fields(q, x, dt) if fields else None
        info.solve_s = time.perf_counter() - t0
        mean_stress = total / self.box_volume
        return Solution(
            physics=self.physics,
            bc=self.bc,
            loading=loading,
            strain=voigt_to_tensor(x, strain=True) if elastic else x.copy(),
            stress=voigt_to_tensor(mean_stress, strain=False) if elastic else mean_stress,
            energy=float(energy / self.box_volume),
            material_fraction=self.material_fraction,
            phases=sorted(phases, key=lambda p: p.phase),
            info=info,
            fields=fld,
            support=self.mesh if fields else None,
        )

    def _fields(self, q, x, dt) -> dict:
        """The means over the elements of the background mesh of the strains, stresses
        and energy (zero in the voids), and the nodal displacement (the affine field and
        the standard fluctuation: the enrichment vanishes at the nodes)."""
        elastic = self.nd == 3
        whole, cut = self._generalized(q, x)
        m = self.mesh.n_elements
        eps, sig, dens = np.zeros((m, self.nc)), np.zeros((m, self.nc)), np.zeros(m)
        ids = np.flatnonzero(self._whole)[self._keep]
        for v, law in self.laws.items():
            if law is None:
                continue
            d, eigen = law
            sel = self.el_side == v
            eps[ids[sel]] = whole[sel]
            sig[ids[sel]] = self._sign() * (whole[sel] - dt * eigen) @ d
            if cut is not None:
                w = self._cut_weights(slice(None))[v]
                s = self._sign() * (cut - dt * eigen) @ d
                eps[self._cut_ids] += np.einsum("mq,mqr->mr", w, cut) / self.cut.volume[:, None]
                sig[self._cut_ids] += np.einsum("mq,mqr->mr", w, s) / self.cut.volume[:, None]
        dens = np.einsum("mi,mi->m", sig, eps) * self._sign()
        nd, nn = self.nd, self.n_nodes
        nodal = q[: nd * nn].reshape(nn, nd)[self._wrap]
        if elastic:
            nodal = nodal + self.mesh.nodes @ voigt_to_tensor(x, strain=True).T
            return {"strain": eps, "stress": sig, "energy": dens, "displacement": nodal}
        nodal = nodal[:, 0] + self.mesh.nodes @ x
        return {"gradient": eps, "flux": sig, "dissipation": dens, "temperature": nodal}

    def _empty_solution(self, loading: Loading, t0: float) -> Solution:
        elastic = self.nd == 3
        values = np.where(loading.strain_controlled, loading.array, 0.0)
        return Solution(
            physics=self.physics,
            bc=self.bc,
            loading=loading,
            strain=voigt_to_tensor(values, strain=True) if elastic else values,
            stress=np.zeros((3, 3)) if elastic else np.zeros(3),
            energy=0.0,
            material_fraction=0.0,
            phases=[],
            info=SolveInfo("none", 0, 0, 0.0, 0.0, time.perf_counter() - t0),
        )

    def summary(self) -> dict:
        """The sizes of the discretization: elements, those cut, nodes, enriched nodes and
        unknowns."""
        return {
            "elements": int(self.mesh.n_elements),
            "cut": 0 if self.cut is None else len(self.cut),
            "nodes": self.n_nodes,
            "enriched": self.n_enriched,
            "unknowns": int(self.free.size) if not self.empty else 0,
        }


class XFEMElasticProblem(_XFEMProblem):
    """Periodic linear (thermo-)elasticity of a :class:`LevelSetSample` by extended finite
    elements (see the module)."""

    physics = "elastic"
    nd = 3


class XFEMThermalProblem(_XFEMProblem):
    """Periodic steady conduction of a :class:`LevelSetSample` by extended finite
    elements, or with ``physics`` any other transport phenomenon of a scalar potential
    (:data:`~morphogen.homogenization.materials.TRANSPORT`)."""

    physics = "thermal"
    nd = 1


# -- loading paths -------------------------------------------------------------------------------


class _XFEMPaths:
    """Discretization of the loading paths by extended finite elements (see
    :class:`~morphogen.homogenization.nonlinear._PathSolver`): the integration points are
    those of the uncut elements (one each), then those of the sub-tetrahedra of the cut
    elements on the sides that carry a material; the tangent is assembled at each
    iteration and solved as that of the finite elements. In finite strains (total
    Lagrangian: the level set cuts the elements of the reference configuration, and the
    enrichment functions are those of its points) the measures are the displacement
    gradients, of nine components, their conjugates the first Piola-Kirchhoff stresses,
    and the macroscopic gradient is upper triangular
    (:mod:`morphogen.homogenization.finite`)."""

    def __init__(self, problem):
        from morphogen.homogenization.solvers import choose_method

        self.problem = problem
        p = problem
        self.finite = bool(p.finite)
        self.n_q = p.free.size
        self.n_whole = len(p.el_volume)
        self.b_whole = self._operator(p.el_grads)
        self.dofs_whole = p._element_dofs(p.el_nodes, False)
        self.solver = None
        self.iterative = choose_method(p.solver_method, self.n_q) != "direct"

    # -- the arrays of the points and their behaviours: those of the CPU -----------------

    def arrays(self, n_pts: int, nc: int, width: int) -> tuple:
        from morphogen.homogenization.nonlinear import _PointArrays

        return _PointArrays.arrays(self, n_pts, nc, width)

    def integrate(self, *args) -> int:
        from morphogen.homogenization.nonlinear import _PointArrays

        return _PointArrays.integrate(self, *args)

    def stress_scale(self, sigma, wts) -> float:
        return float(np.sum(np.abs(wts) * np.sqrt(np.einsum("ij,ij->i", sigma, sigma))))

    def asymmetry(self, tangent, wts) -> float:
        from morphogen.homogenization.nonlinear import asymmetry

        return asymmetry(tangent, wts, self.problem.point_group)

    def symmetrize(self, tangent) -> None:
        tangent[...] = 0.5 * (tangent + np.swapaxes(tangent, 1, 2))

    def host(self, *arrays) -> tuple:
        return arrays

    # -- kinematics, forces and tangents --------------------------------------------------

    def _full(self, q) -> np.ndarray:
        out = np.zeros(self.problem.n_dofs)
        out[self.problem.free] = q
        return out

    def _operator(self, g: np.ndarray) -> np.ndarray:
        """The operators of the measures of strain of displacements (the strains, or the
        displacement gradients in finite strains) from the gradients of their functions."""
        return _gradient_operator(g) if self.finite else _operator(g, 3)

    def _macro(self, e) -> np.ndarray:
        """The measure of strain of the macroscopic components ``e`` (6)."""
        from morphogen.homogenization import finite as fin

        e = np.asarray(e, dtype=float)
        return fin.gradient_of(e) if self.finite else e

    def _measures(self, q_full, e):
        """The measures of strain of the uncut elements and of the points of the cut
        ones (m x 24 x components) for the fluctuation ``q_full`` and the macroscopic
        components ``e``."""
        p = self.problem
        x = self._macro(e)
        ue = q_full[self.dofs_whole]
        whole = np.einsum("mra,ma->mr", self.b_whole, ue) + x
        cut = None
        if p.cut is not None:
            cut = np.empty((len(p.cut), 24, len(x)))
            for lo in range(0, len(p.cut), _CHUNK):
                sl = slice(lo, lo + _CHUNK)
                bq = self._operator(p.cut.functions(sl, p.enriched))
                ue = q_full[p._element_dofs(p.cut_nodes[sl], p.enriched)]
                cut[sl] = np.einsum("mqra,ma->mqr", bq, ue) + x
        return whole, cut

    def displacement(self, q, e) -> np.ndarray:
        p = self.problem
        nodal = self._full(q)[: 3 * p.n_nodes].reshape(-1, 3)[p._wrap]
        if self.finite:
            gradient = self._macro(e).reshape(3, 3)
        else:
            gradient = voigt_to_tensor(e, strain=True)
        return (nodal + p.mesh.nodes @ gradient.T).ravel()

    def measures(self, q, e, grads, wts) -> None:
        p = self.problem
        whole, cut = self._measures(self._full(q), e)
        grads[: self.n_whole] = whole
        wts[: self.n_whole] = p.el_volume
        if cut is not None:
            grads[self.n_whole :] = cut[p.cut_points]
            wts[self.n_whole :] = p.cut.w[p.cut_points]

    def _cut_arrays(self, values, sl: slice):
        """The values of the points of the cut elements ``sl`` (m x 24 x ...), zero at
        the points that carry nothing."""
        p = self.problem
        lo, hi, _ = sl.indices(len(p.cut))
        mask = p.cut_points[lo:hi]
        start = self.n_whole + int(p.cut_points[:lo].sum())
        out = np.zeros((hi - lo, 24, *values.shape[1:]))
        out[mask] = values[start : start + int(mask.sum())]
        return out, np.where(mask, p.cut.w[lo:hi], 0.0)

    def forces(self, sigma, wts):
        p = self.problem
        f = np.zeros(p.n_dofs)
        fe = p.el_volume[:, None] * np.einsum("mra,mr->ma", self.b_whole, sigma[: self.n_whole])
        np.add.at(f, self.dofs_whole, fe)
        scale = float(np.sum(fe * fe))
        if p.cut is not None:
            for lo in range(0, len(p.cut), _CHUNK):
                sl = slice(lo, lo + _CHUNK)
                bq = self._operator(p.cut.functions(sl, p.enriched))
                s, w = self._cut_arrays(sigma, sl)
                fe = np.einsum("mq,mqra,mqr->ma", w, bq, s)
                np.add.at(f, p._element_dofs(p.cut_nodes[sl], p.enriched), fe)
                scale += float(np.sum(fe * fe))
        total = np.einsum("p,pi->i", wts, sigma)
        return f[p.free], total, float(np.sqrt(scale))

    def _tangent(self, tangent, wts) -> sp.csr_matrix:
        """The tangent stiffness of all the unknowns, ``sum int B^T C B``."""
        p = self.problem
        rows, cols, vals = [], [], []
        b = self.b_whole
        ke = p.el_volume[:, None, None] * (np.swapaxes(b, 1, 2) @ (tangent[: self.n_whole] @ b))
        dofs = self.dofs_whole
        rows.append(np.repeat(dofs, 12, axis=1).ravel())
        cols.append(np.tile(dofs, (1, 12)).ravel())
        vals.append(ke.ravel())
        if p.cut is not None:
            for lo in range(0, len(p.cut), _CHUNK):
                sl = slice(lo, lo + _CHUNK)
                bq = self._operator(p.cut.functions(sl, p.enriched))
                c, w = self._cut_arrays(tangent, sl)
                m, nf = bq.shape[0], bq.shape[-1]
                cb = (c @ bq) * w[:, :, None, None]
                ke = np.swapaxes(bq.reshape(m, -1, nf), 1, 2) @ cb.reshape(m, -1, nf)
                dofs = p._element_dofs(p.cut_nodes[sl], p.enriched)
                rows.append(np.repeat(dofs, nf, axis=1).ravel())
                cols.append(np.tile(dofs, (1, nf)).ravel())
                vals.append(ke.ravel())
        n = p.n_dofs
        return sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        ).tocsr()

    def full_tangent(self, tangent, wts) -> sp.csr_matrix:
        return self._tangent(tangent, wts)

    def linearize(self, tangent, wts, first: bool, symmetric: bool | None = None):
        p = self.problem
        k = self._tangent(tangent, wts)
        kqq = k[p.free][:, p.free].tocsr()
        if p._regularization:
            kqq = (kqq + p._regularization * sp.identity(kqq.shape[0])).tocsr()
        symmetric = p.symmetrize if symmetric is None else symmetric
        if first or self.solver is None:
            near_null = p._rigid_modes()[p.free]
            self.solver = LinearSolver(kqq, method=p.solver_method, tol=p.tol, maxiter=p.maxiter,
                                       near_null=near_null, symmetric=symmetric)  # fmt: skip
        else:
            self.solver = self.solver.updated(kqq, symmetric)
        return self.solver

    def macro_columns(self, tangent, wts) -> tuple[np.ndarray, np.ndarray]:
        """``int B^T C`` on all the unknowns (n x 6) and ``int C`` (6 x 6), for the six
        macroscopic components (in finite strains: those of the upper triangular
        gradient among the nine)."""
        p = self.problem
        if self.finite:
            from morphogen.homogenization.finite import MACRO

            macro = list(MACRO)
            tangent = tangent[:, :, macro]
        columns = np.zeros((p.n_dofs, 6))
        bt = np.swapaxes(self.b_whole, 1, 2)
        fe = p.el_volume[:, None, None] * (bt @ tangent[: self.n_whole])
        np.add.at(columns, self.dofs_whole, fe)
        if p.cut is not None:
            for lo in range(0, len(p.cut), _CHUNK):
                sl = slice(lo, lo + _CHUNK)
                bq = self._operator(p.cut.functions(sl, p.enriched))
                c, w = self._cut_arrays(tangent, sl)
                fe = np.einsum("mq,mqra,mqrs->mas", w, bq, c)
                np.add.at(columns, p._element_dofs(p.cut_nodes[sl], p.enriched), fe)
        kee = np.einsum("p,pij->ij", wts, tangent)
        return columns, kee[macro] if self.finite else kee

    def coupling(self, tangent, wts, unknown):
        columns, kee = self.macro_columns(tangent, wts)
        return columns[self.problem.free][:, unknown], kee

    def macro_response(self, tangent, wts, v) -> np.ndarray:
        """``K_Ew v``: the integral of the stress increments of the fluctuation ``v``."""
        p = self.problem
        whole, cut = self._measures(self._full(v), np.zeros(6))
        eps = whole if cut is None else np.concatenate([whole, cut[p.cut_points]])
        out = np.einsum("p,pij,pj->i", wts, tangent, eps)
        if self.finite:
            from morphogen.homogenization.finite import macro_part

            out = macro_part(out)
        return out


class NonlinearXFEMProblem(_PathSolver, _XFEMProblem):
    """Loading paths of a :class:`LevelSetSample` with the nonlinear behaviours of its
    phases, by extended finite elements in small or finite strains (``kinematics``; see
    the module and :class:`~morphogen.homogenization.nonlinear.NonlinearFEProblem` for
    the options of the iterations): :meth:`run` follows a
    :class:`~morphogen.homogenization.paths.LoadingPath`. The linear problems of the
    sample are those of :class:`XFEMElasticProblem`."""

    physics = "elastic"
    nd = 3

    def __init__(self, sample: LevelSetSample, materials: dict[int, Material],
                 bc: str = "periodic", *, solver: str = "auto", tol: float = 1e-8,
                 maxiter: int | None = None, snap: float = 1e-3, newton_tol: float = 1e-6,
                 max_newton: int = 25, max_cuts: int = 10, kinematics: str = "small",
                 tangent: str = "auto"):  # fmt: skip
        from morphogen.homogenization.materials import material_table
        from morphogen.homogenization.nonlinear import behaviour_table

        super().__init__(sample, materials, bc, solver=solver, tol=tol, maxiter=maxiter, snap=snap)
        self._path_options(newton_tol, max_newton, max_cuts, kinematics, tangent)
        if self.empty:
            return
        s = self.sample
        phase = np.where(self.el_side < 0, s.inside, s.outside)
        self.cut_points = np.zeros((0, 24), dtype=bool)
        if self.cut is not None:
            carried = np.zeros(self.cut.w.shape, dtype=bool)
            for v, law in self.laws.items():
                if law is not None:
                    carried |= (self.cut.side == v) & (self.cut.w > 0)
            self.cut_points = carried
            phase = np.concatenate([phase, np.where(self.cut.side[carried] < 0, s.inside,
                                                    s.outside)])  # fmt: skip
        self.table = material_table(phase, materials, "elastic")
        self.groups = self.table.phases
        beh = behaviour_table(phase, materials, self.table, finite=self.finite)
        self._point_behaviours(beh, self.table.mat, self.table.stat, 1)
        if beh.nonlinear:
            log.info(
                "XFEM loading paths: linear tetrahedra lock under isochoric plastic flow "
                "(prefer conforming quadratic meshes or voxels for plasticity)"
            )

    def _discretization(self) -> _XFEMPaths:
        return _XFEMPaths(self)

    def _support(self):
        return self.mesh

    def _element_mean(self, values) -> np.ndarray:
        """The means of the values of the points over the elements of the background
        mesh (zero in the voids): those of the points of the sub-tetrahedra of the cut
        elements, weighted."""
        n_whole = len(self.el_volume)
        out = np.zeros((self.mesh.n_elements, *values.shape[1:]))
        out[np.flatnonzero(self._whole)[self._keep]] = values[:n_whole]
        if self.cut is not None:
            rows = np.nonzero(self.cut_points)[0]
            w = self.cut.w[self.cut_points]
            total = np.bincount(rows, w, minlength=len(self.cut))
            share = w / total[rows]
            part = np.zeros((len(self.cut), *values.shape[1:]))
            np.add.at(part, rows, share.reshape(-1, *[1] * (values.ndim - 1)) * values[n_whole:])
            out[self._cut_ids] = part
        return out
