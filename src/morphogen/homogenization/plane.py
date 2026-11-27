"""Finite elements of the plane on triangular meshes (3-node and 6-node triangles).

The structure of a mesh of the plane (:class:`~morphogen.mesh.triangles.TriMesh`) is
invariant across its plane: the columnar grains of a tessellation of coplanar germs, the
section of a long fibre composite, a layer whose fields do not vary through its thickness.
The fields of such a structure under a uniform macroscopic loading do not vary along ``z``
either: the problems are those of the plane, and their solutions those of the structure in
space.

**Elasticity** (:class:`PlaneElasticProblem`). The displacement is ``u = E x + w(x, y)``,
``E`` the macroscopic strain (its six components), ``w`` the fluctuation, a function of the
plane: the strain is ``E + sym grad w`` with ``d/dz = 0``, so that ``eps_33 = E_33`` is
uniform and the anti-plane component ``w_3`` gives the shears ``eps_13``, ``eps_23``.

``plane="generalized"`` (default)
    the three components of ``w`` are free: the apparent stiffness is the 6 x 6 tensor of the
    structure in space (generalized plane strain, the anti-plane shear coupled to the plane
    by the crystals whose axes leave it);
``plane="strain"``
    ``w_3 = 0``: plane strain, the fluctuation in the plane only; the block 11, 22, 12 of
    the tensor is the stiffness of plane strain (the shears 13, 23 then uniform).

**Conduction** (:class:`PlaneThermalProblem`), and every transport of a scalar potential:
``T = G . x + w(x, y)``, the gradient ``G + grad w`` with ``d/dz = 0``; the apparent tensor
is the 3 x 3 tensor of the structure in space.

**Boundary conditions**, on the sides of the rectangle: ``periodic`` (``w`` the same at the
facing nodes of opposite sides, the meshes of periodic structures having matching sides),
``kubc`` (``w = 0`` on the sides), ``subc`` (the tractions ``Sigma n``, the flux ``Q . n``,
on the sides; the unknowns the displacement or the temperature, rigid motions removed, and
the uniform components that no field of the plane gives, ``E_33`` — and ``E_13``, ``E_23``
in plane strain — or ``G_3``, unknowns of their own, dual to ``Sigma_33`` — ``Sigma_13``,
``Sigma_23`` — or ``Q_3``). Porous meshes (void phases, or parts of the rectangle not
meshed) take periodic and kinematic conditions, averages over the rectangle.

The elements are integrated exactly on straight triangles: one point for 3-node elements,
three for 6-node elements (degree 2). :meth:`~morphogen.mesh.triangles.TriMesh.extruded`
gives the same problems in space on a layer of tetrahedra, periodic across it: the 4-node
elements of the layer reproduce those of the plane.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.sparse as sp

from morphogen.homogenization.core import (
    Loading,
    MacroscopicLoadings,
    PhaseStatistics,
    Solution,
    boundary_condition,
)
from morphogen.homogenization.elements import voigt_to_tensor
from morphogen.homogenization.materials import Material, material_table, transport_physics
from morphogen.homogenization.solvers import (
    NEGLIGIBLE,
    LinearSolver,
    SolveInfo,
    choose_method,
    norm,
)
from morphogen.mesh.triangles import TriMesh

log = logging.getLogger("morphogen")

#: The hypotheses of the elastic problems of the plane
PLANES = ("generalized", "strain")
#: The boundary conditions of the problems of the plane
PLANE_CONDITIONS = ("kubc", "subc", "periodic")
#: Gauss points (barycentric coordinates) and weights (of the area) of the triangles
GAUSS = {
    1: (np.array([[1.0, 1.0, 1.0]]) / 3.0, np.array([1.0])),
    2: (np.array([[4.0, 1.0, 1.0], [1.0, 4.0, 1.0], [1.0, 1.0, 4.0]]) / 6.0, np.full(3, 1 / 3)),
}
#: Engineering shears of the Voigt components (11, 22, 33, 23, 31, 12)
_ENGINEERING = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def _gradients(mesh: TriMesh) -> tuple[np.ndarray, np.ndarray]:
    """The gradients of the shape functions at the Gauss points of the elements,
    ``(m, q, n, 2)``, and the weights of the points (their share of the areas), ``(m, q)``.
    The 6-node elements are those of their straight corners (mid-edge nodes at the middles
    of their edges)."""
    p = mesh.nodes[mesh.elements[:, :3]]
    j = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=2)  # columns: the edges
    det = j[:, 0, 0] * j[:, 1, 1] - j[:, 0, 1] * j[:, 1, 0]
    rows = [
        np.stack([j[:, 1, 1], -j[:, 0, 1]], axis=1),
        np.stack([-j[:, 1, 0], j[:, 0, 0]], axis=1),
    ]
    inv = np.stack(rows, axis=1) / det[:, None, None]  # the rows: the gradients of l1, l2
    dl = np.stack([-inv[:, 0] - inv[:, 1], inv[:, 0], inv[:, 1]], axis=1)  # (m, 3, 2)
    area = 0.5 * det
    points, weights = GAUSS[mesh.order]
    if mesh.order == 1:
        return dl[:, None], area[:, None] * weights[None]
    grads = []
    for lam in points:
        g = [(4.0 * lam[k] - 1.0) * dl[:, k] for k in range(3)]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            g.append(4.0 * (lam[a] * dl[:, b] + lam[b] * dl[:, a]))
        grads.append(np.stack(g, axis=1))
    return np.stack(grads, axis=1), area[:, None] * weights[None]


def _strain_operator(grads: np.ndarray, nd: int) -> np.ndarray:
    """The strains (Voigt, engineering shears) of the nodal displacements of the elements,
    ``(m, q, 6, n nd)``: ``nd`` 3 (the anti-plane component too) or 2."""
    m, q, n, _ = grads.shape
    b = np.zeros((m, q, 6, n, nd))
    gx, gy = grads[..., 0], grads[..., 1]
    b[:, :, 0, :, 0] = gx
    b[:, :, 1, :, 1] = gy
    b[:, :, 5, :, 0] = gy
    b[:, :, 5, :, 1] = gx
    if nd == 3:
        b[:, :, 3, :, 2] = gy
        b[:, :, 4, :, 2] = gx
    return b.reshape(m, q, 6, n * nd)


def _gradient_operator(grads: np.ndarray) -> np.ndarray:
    """The gradients (three components, the third zero) of the nodal values, ``(m, q, 3,
    n)``."""
    m, q, n, _ = grads.shape
    b = np.zeros((m, q, 3, n))
    b[:, :, :2, :] = np.swapaxes(grads, 2, 3)
    return b


class _PlaneProblem(MacroscopicLoadings):
    physics = ""
    #: the Voigt (vector) components that the fields of the plane do not give, unknowns of
    #: their own under static conditions
    globals_: tuple[int, ...] = ()

    def __init__(self, mesh: TriMesh, materials: dict[int, Material], bc: str = "periodic", *,
                 solver: str = "auto", tol: float = 1e-8, maxiter: int | None = None,
                 regularization: float = 1e-12, physics: str | None = None) -> None:  # fmt: skip
        if not isinstance(mesh, TriMesh):
            raise TypeError("the solver of the plane needs a TriMesh (morphogen.mesh.triangles)")
        if physics is not None:
            if transport_physics(physics) != (self.n_components == 3):
                kind = "a transport phenomenon" if self.n_components == 3 else "'elastic'"
                raise ValueError(f"{type(self).__name__}: physics {kind}, got {physics!r}")
            self.physics = physics
        self.bc = boundary_condition(bc)
        if self.bc not in PLANE_CONDITIONS:
            raise ValueError(f"the problems of the plane take the conditions {PLANE_CONDITIONS}")
        self.box_volume = mesh.box_area()  # the averages: over the rectangle
        phases = [int(p) for p in np.unique(mesh.phase)]
        missing = [p for p in phases if p not in materials]
        if missing:
            raise ValueError(f"no material for phase(s) {missing}; phases present: {phases}")
        voids = [p for p in phases if materials[p].inert(self.physics)]
        solid = [p for p in phases if p not in voids]
        covered = float(np.abs(mesh.areas()).sum()) / mesh.box_area()
        self.has_voids = bool(voids) or covered < 1.0 - 1e-9
        if self.bc == "subc" and self.has_voids:
            raise ValueError("static uniform (Neumann) conditions cannot be applied to porous "
                             "media (Dirrenberger et al. 2014, sec. 3.4.2)")  # fmt: skip
        if voids:
            mesh = mesh.subset(solid) if solid else None
        self.empty = mesh is None or mesh.n_elements == 0
        if self.empty:
            log.warning("no load-carrying material in the sample: its apparent properties are zero")
        if (self.bc == "periodic" and not self.empty and not mesh.matching_sides()
                and not self.has_voids):  # fmt: skip
            raise ValueError("periodic conditions need a mesh whose opposite sides match "
                             "(a periodic structure meshed with periodic=True)")  # fmt: skip
        self.mesh = mesh
        self.solver_method = solver
        self.tol = tol
        self.maxiter = maxiter
        self.regularization = float(regularization) if self.has_voids else 0.0
        self.groups = tuple(solid)
        self._units: dict = {}
        self._systems: dict = {}
        self._matrix = None
        if not self.empty:
            self.table = material_table(mesh.phase, materials, self.physics, grains=mesh.cell,
                                        orientations=mesh.orientations)  # fmt: skip
            self.groups = self.table.phases
            self.group = self.table.stat
            self.mat = self.table.mat
            self.grads, self.weights = _gradients(mesh)
            self.operator = self._operator()

    # -- the system ----------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        return 6 if self.physics == "elastic" else 3

    @property
    def n_global(self) -> int:
        """The unknowns of the uniform components (static conditions)."""
        return len(self.globals_) if self.bc == "subc" else 0

    def _dofs(self) -> np.ndarray:
        """The degrees of freedom of the nodes of each element, ``(m, n nd)``."""
        e = self.mesh.elements
        return (e[:, :, None] * self.nd + np.arange(self.nd)).reshape(len(e), -1)

    def _full_operator(self) -> tuple[np.ndarray, np.ndarray]:
        """The operator of the nodal unknowns and of the global ones, ``(m, q, c, n nd +
        g)``, and the degrees of freedom of its columns ``(m, n nd + g)``."""
        b, dofs = self.operator, self._dofs()
        g = self.n_global
        if not g:
            return b, dofs
        m, q, c, _ = b.shape
        extra = np.zeros((m, q, c, g))
        for k, comp in enumerate(self.globals_):
            extra[:, :, comp, k] = 1.0
        n_field = self.mesh.n_nodes * self.nd
        cols = np.broadcast_to(n_field + np.arange(g), (m, g))
        return np.concatenate([b, extra], axis=3), np.concatenate([dofs, cols], axis=1)

    def _stiffness(self) -> sp.csr_matrix:
        if self._matrix is None:
            b, dofs = self._full_operator()
            active = self.mat >= 0
            d = self.table.matrices[np.maximum(self.mat, 0)]
            k = np.einsum("mq,mqai,mab,mqbj->mij", self.weights, b, d, b, optimize=True)
            k[~active] = 0.0
            size = self.mesh.n_nodes * self.nd + self.n_global
            rows = np.repeat(dofs, dofs.shape[1], axis=1).ravel()
            cols = np.tile(dofs, (1, dofs.shape[1])).ravel()
            self._matrix = sp.coo_matrix((k.ravel(), (rows, cols)), shape=(size, size)).tocsr()
        return self._matrix

    def _uniform_load(self, x: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
        """The nodal loads of a uniform strain (gradient) ``x`` and of the change of
        temperature, ``-int B^T D (x - a dT)``, and the size of the loads of the elements
        before they sum (a load below ``NEGLIGIBLE`` times it is round-off: the uniform
        field solves the problem)."""
        b, dofs = self._full_operator()
        d = self.table.matrices[np.maximum(self.mat, 0)]
        e = np.broadcast_to(np.asarray(x, dtype=float), (len(d), len(x))).copy()
        if dt and self.physics == "elastic":
            e = e - dt * self.table.eigen[np.maximum(self.mat, 0)]
        s = np.einsum("mab,mb->ma", d, e)
        s[self.mat < 0] = 0.0
        f = -np.einsum("mq,mqai,ma->mi", self.weights, b, s, optimize=True)
        load = np.zeros(self.mesh.n_nodes * self.nd + self.n_global)
        np.add.at(load, dofs.ravel(), f.ravel())
        return load, float(np.sqrt(np.sum(f * f)))

    def _side_load(self, t: np.ndarray) -> np.ndarray:
        """The consistent nodal loads of the uniform tractions (fluxes) ``t(n)`` on the
        sides: ``t`` maps an outward normal to the load per unit length (``nd``)."""
        m = self.mesh
        load = np.zeros((m.n_nodes, self.nd))
        for axis in (0, 1):
            for side in (0, 1):
                edges = m.side_edges(axis, side)
                if not len(edges):
                    continue
                normal = np.zeros(2)
                normal[axis] = 1.0 if side else -1.0
                per = np.atleast_1d(t(normal))
                length = np.linalg.norm(m.nodes[edges[:, 1]] - m.nodes[edges[:, 0]], axis=1)
                if m.order == 1:
                    shares = [(edges[:, 0], 0.5), (edges[:, 1], 0.5)]
                else:  # Simpson: 1/6, 1/6 at the ends, 2/3 at the middle
                    shares = [(edges[:, 0], 1 / 6), (edges[:, 1], 1 / 6), (edges[:, 2], 2 / 3)]
                for nodes, share in shares:
                    np.add.at(load, nodes, (share * length)[:, None] * per[None, :])
        out = np.zeros(m.n_nodes * self.nd + self.n_global)
        out[: m.n_nodes * self.nd] = load.ravel()
        return out

    def _supports(self) -> list[int]:
        """The nodes nearest to the corners (x0, y0) and (x1, y0) of the rectangle."""
        m = self.mesh
        a = int(np.argmin(np.sum((m.nodes - m.lower) ** 2, axis=1)))
        target = np.array([m.upper[0], m.lower[1]])
        b = int(np.argmin(np.sum((m.nodes - target) ** 2, axis=1)))
        return [a, b]

    def _reduction(self):
        """The matrix ``T`` of the unknowns ``w = T x`` of the boundary condition."""
        m, nd = self.mesh, self.nd
        n = m.n_nodes
        size = n * nd + self.n_global
        if self.bc == "periodic":
            master = np.arange(n)
            for axis in (0, 1):
                pairs = m.periodic_pairs(axis)
                master[pairs[:, 1]] = pairs[:, 0]
            for _ in range(4):  # the corners: one node
                master = master[master]
            heads, index = np.unique(master, return_inverse=True)
            t_nodes = sp.csr_matrix((np.ones(n), (np.arange(n), index)), shape=(n, len(heads)))
            t = sp.kron(t_nodes, sp.identity(nd), format="csr") if nd > 1 else t_nodes
            keep = np.arange(t.shape[1])
            if not self.has_voids:  # the translations: zero at one node
                keep = keep[nd:]
            return t[:, keep].tocsr()
        fixed = np.zeros(size, dtype=bool)
        if self.bc == "kubc":
            sides = np.unique(np.concatenate([m.side_nodes(a, s) for a in (0, 1) for s in (0, 1)]))
            fixed[(sides[:, None] * nd + np.arange(nd)).ravel()] = True
        else:  # subc: the rigid motions
            a, b = self._supports()
            fixed[a * nd : a * nd + nd] = True
            if self.physics == "elastic":
                fixed[b * nd + 1] = True  # the rotation in the plane
        keep = np.flatnonzero(~fixed)
        return sp.csr_matrix((np.ones(keep.size), (keep, np.arange(keep.size))),
                             shape=(size, keep.size))  # fmt: skip

    def _system(self):
        if "system" not in self._systems:
            t = self._reduction()
            reduced = (t.T @ self._stiffness() @ t).tocsr()
            if self.regularization > 0 and reduced.shape[0]:
                diag = reduced.diagonal()
                reduced = (reduced + sp.diags(np.full(diag.size, self.regularization
                                                      * float(np.mean(diag))))).tocsr()  # fmt: skip
            method = choose_method(self.solver_method, reduced.shape[0])
            near = None
            if method != "direct":
                near = self._near_null(t)
            solver = LinearSolver(reduced, method=method, tol=self.tol, maxiter=self.maxiter,
                                  near_null=near)  # fmt: skip
            self._systems["system"] = (t, solver)
        return self._systems["system"]

    def _near_null(self, t) -> np.ndarray:
        """The rigid motions (translations; the rotation in the plane but for periodic
        conditions), in the reduced unknowns."""
        m, nd = self.mesh, self.nd
        n = m.n_nodes
        modes = []
        for c in range(nd):
            v = np.zeros((n, nd))
            v[:, c] = 1.0
            modes.append(v)
        if self.physics == "elastic" and self.bc != "periodic":
            x = m.nodes - m.nodes.mean(axis=0)
            v = np.zeros((n, nd))
            v[:, 0], v[:, 1] = -x[:, 1], x[:, 0]
            modes.append(v)
        full = np.stack([np.concatenate([v.ravel(), np.zeros(self.n_global)]) for v in modes],
                        axis=1)  # fmt: skip
        tt = (t.T @ t).diagonal()
        return (t.T @ full) / np.where(tt > 0, tt, 1.0)[:, None]

    def _solve(self, rhs: np.ndarray, scale: float | None):
        t, solver = self._system()
        b = t.T @ rhs
        if scale is not None and norm(b) <= NEGLIGIBLE * scale:
            return np.zeros(t.shape[0]), SolveInfo(solver.method, b.size, 0, 0.0,
                                                   solver.setup_s, 0.0)  # fmt: skip
        x, info = solver.solve(b)
        return t @ x, info

    # -- the fields ----------------------------------------------------------------------

    def _field(self, x: np.ndarray, dt: float = 0.0):
        """The unknowns of the native loading ``x`` (the macroscopic strain or gradient;
        the stress or flux under ``subc``) and of the change of temperature: the
        fluctuation (``subc``: the displacement or temperature, and the global unknowns)."""
        if self.native in ("strain", "gradient"):
            rhs, scale = self._uniform_load(x, dt)
        else:
            rhs, scale = self._static_load(x), None
            if dt:
                rhs = rhs + self._uniform_load(np.zeros(len(x)), dt)[0]
        return self._solve(rhs, scale)

    def _point_values(self, state: np.ndarray, x: np.ndarray, dt: float):
        """At the Gauss points, ``(m, q, c)`` each: the strains (tensor components) or the
        gradients, the stresses or the fluxes, and the energy densities, ``sigma : (eps -
        a dT)`` or ``K grad T . grad T``."""
        b, dofs = self._full_operator()
        e = np.einsum("mqai,mi->mqa", b, state[dofs], optimize=True)
        if self.native in ("strain", "gradient"):
            e = e + np.asarray(x, dtype=float)
        d = self.table.matrices[np.maximum(self.mat, 0)]
        free = e
        if dt and self.physics == "elastic":
            free = e - dt * self.table.eigen[np.maximum(self.mat, 0)][:, None, :]
        s = np.einsum("mab,mqb->mqa", d, free)
        energy = np.einsum("mqa,mqa->mq", s, free)
        if self.physics == "elastic":
            e = e * np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])  # the tensor components
        else:
            s = -s  # the flux q = -K grad T
        s[self.mat < 0] = 0.0
        energy[self.mat < 0] = 0.0
        return e, s, energy

    def _sums(self, state: np.ndarray, x: np.ndarray, dt: float) -> np.ndarray:
        """Per group: the integrals of the strain (gradient), the stress (flux), the energy
        density, their squares and the area (the columns of the solvers in space)."""
        e, s, energy = self._point_values(state, x, dt)
        w = self.weights * (self.mat >= 0)[:, None]

        def integral(v):
            return np.einsum("mq,mqa->ma", w, v)

        dens = energy[..., None]
        cols = [integral(e), integral(s), integral(dens), integral(e * e), integral(s * s),
                integral(dens * dens), w.sum(axis=1)[:, None]]  # fmt: skip
        per = np.concatenate(cols, axis=1)
        out = np.zeros((len(self.groups), per.shape[1]))
        np.add.at(out, self.group, per)
        return out

    def _phase_statistics(self, sums, names) -> list[PhaseStatistics]:
        c = self.n_components
        out = []
        for g, phase in enumerate(self.groups):
            vol = sums[g, -1]
            if vol <= 0:
                continue
            mean, std = {}, {}
            for k, (name, size) in enumerate(zip(names, (c, c, 1), strict=True)):
                pos = (0, c, 2 * c)[k]
                m = sums[g, pos : pos + size] / vol
                sq = sums[g, 2 * c + 1 + pos : 2 * c + 1 + pos + size] / vol
                mean[name] = m
                std[name] = np.sqrt(np.maximum(sq - m * m, 0.0))
            out.append(PhaseStatistics(phase, float(vol / self.box_volume), mean, std))
        return out

    def _empty_solution(self, loading: Loading, t0: float) -> Solution:
        elastic = self.physics == "elastic"
        values = np.where(loading.strain_controlled, loading.array, 0.0)
        strain = voigt_to_tensor(values, strain=True) if elastic else values
        return Solution(physics=self.physics, bc=self.bc, loading=loading, strain=strain,
                        stress=np.zeros((3, 3)) if elastic else np.zeros(3), energy=0.0,
                        material_fraction=0.0, phases=[],
                        info=SolveInfo("none", 0, 0, 0.0, 0.0, time.perf_counter() - t0),
                        )  # fmt: skip

    def _macroscopic(self, state, x) -> np.ndarray:
        """The macroscopic strain (3 x 3) or gradient (3) of a solution: the loading, or
        under static conditions the uniform components solved for."""
        n_field = self.mesh.n_nodes * self.nd
        if self.native in ("strain", "gradient"):
            values = np.asarray(x, dtype=float)
        else:
            values = np.zeros(self.n_components)
            for k, comp in enumerate(self.globals_):
                values[comp] = state[n_field + k]
        return voigt_to_tensor(values, strain=True) if self.physics == "elastic" else values

    def _element_fields(self, state, x, dt) -> dict:
        """The element means of the strain (gradient), the stress (flux) and the energy
        density, and the nodal displacement (temperature) of the points of the plane at its
        height: the macroscopic part ``E x`` (``G . x``) and the fluctuation (under static
        conditions, the field and the uniform components)."""
        e, s, energy = self._point_values(state, x, dt)
        w = self.weights / self.weights.sum(axis=1, keepdims=True)
        m = self.mesh
        n = m.n_nodes
        points = np.column_stack([m.nodes, np.full(n, m.z)])
        macro = self._macroscopic(state, x)
        if self.nd > 1:
            nodal = state[: n * self.nd].reshape(n, self.nd) + (points @ macro.T)[:, : self.nd]
        else:
            nodal = state[:n] + points @ macro
        return {"strain": np.einsum("mq,mqa->ma", w, e), "stress": np.einsum("mq,mqa->ma", w, s),
                "energy": np.einsum("mq,mq->m", w, energy), "nodal": nodal}  # fmt: skip


class PlaneElasticProblem(_PlaneProblem):
    """Linear (thermo-)elasticity of the plane (see the module): ``plane`` ``"generalized"``
    (the anti-plane fluctuation free) or ``"strain"`` (plane strain); crystals are turned by
    the orientations of the grains of the mesh."""

    physics = "elastic"

    def __init__(self, mesh: TriMesh, materials: dict[int, Material], bc: str = "periodic", *,
                 plane: str = "generalized", **options) -> None:  # fmt: skip
        if plane not in PLANES:
            raise ValueError(f"plane: one of {PLANES}")
        self.plane = plane
        self.nd = 3 if plane == "generalized" else 2
        self.globals_ = (2,) if plane == "generalized" else (2, 3, 4)
        super().__init__(mesh, materials, bc, **options)

    def _operator(self) -> np.ndarray:
        return _strain_operator(self.grads, self.nd)

    def _static_load(self, x: np.ndarray) -> np.ndarray:
        sigma = voigt_to_tensor(x, strain=False)
        nd = self.nd
        load = self._side_load(lambda n: (sigma[:, :2] @ n)[:nd])
        for k, comp in enumerate(self.globals_):  # the work of Sigma on the uniform strains
            load[self.mesh.n_nodes * nd + k] = self.box_volume * float(x[comp])
        return load

    def _dual(self, state: np.ndarray, x: np.ndarray, dt: float = 0.0) -> np.ndarray:
        """The average stress (strain loadings) or strain (stress loadings, engineering
        shears), in the Voigt order."""
        total = self._sums(state, x, dt).sum(axis=0) / self.box_volume
        if self.native == "strain":
            return total[6:12].copy()
        return total[0:6] * _ENGINEERING

    def _evaluate(self, loading, state, x, dt, info, fields, t0) -> Solution:
        sums = self._sums(state, x, dt)
        total = sums.sum(axis=0)
        stress = voigt_to_tensor(total[6:12] / self.box_volume, strain=False)
        if self.native == "strain":
            strain = voigt_to_tensor(x, strain=True)
        else:
            strain = voigt_to_tensor(total[0:6] / self.box_volume, strain=False)
        fld = None
        if fields:
            f = self._element_fields(state, x, dt)
            fld = {"strain": f["strain"], "stress": f["stress"], "energy": f["energy"],
                   "displacement": f["nodal"]}  # fmt: skip
        info.solve_s = time.perf_counter() - t0
        return Solution(physics="elastic", bc=self.bc, loading=loading, strain=strain,
                        stress=stress, energy=float(total[12] / self.box_volume),
                        material_fraction=float(total[-1] / self.box_volume),
                        phases=self._phase_statistics(sums, ("strain", "stress", "energy")),
                        info=info, fields=fld, support=self.mesh if fields else None)  # fmt: skip


class PlaneThermalProblem(_PlaneProblem):
    """Steady linear conduction of the plane (see the module), or with ``physics`` any other
    transport phenomenon of a scalar potential, with the coefficient of the materials for it;
    the phases whose coefficient is zero (insulators) are left out as pores."""

    physics = "thermal"
    nd = 1
    globals_ = (2,)

    def _operator(self) -> np.ndarray:
        return _gradient_operator(self.grads)

    def _static_load(self, x: np.ndarray) -> np.ndarray:
        q = np.asarray(x, dtype=float)
        load = self._side_load(lambda n: -(q[:2] @ n))
        load[self.mesh.n_nodes] = -self.box_volume * float(q[2])
        return load

    def _dual(self, state: np.ndarray, x: np.ndarray, dt: float = 0.0) -> np.ndarray:
        """The average flux (gradient loadings) or gradient (flux loadings)."""
        total = self._sums(state, x, dt).sum(axis=0) / self.box_volume
        return (total[3:6] if self.native == "gradient" else total[0:3]).copy()

    def _evaluate(self, loading, state, x, dt, info, fields, t0) -> Solution:
        sums = self._sums(state, x, dt)
        total = sums.sum(axis=0)
        grad = (
            np.array(x, dtype=float) if self.native == "gradient" else total[0:3] / self.box_volume
        )
        fld = None
        if fields:
            f = self._element_fields(state, x, dt)
            fld = {"gradient": f["strain"], "flux": f["stress"], "dissipation": f["energy"],
                   "temperature": f["nodal"]}  # fmt: skip
        info.solve_s = time.perf_counter() - t0
        return Solution(physics=self.physics, bc=self.bc, loading=loading, strain=grad,
                        stress=total[3:6] / self.box_volume,
                        energy=float(total[6] / self.box_volume),
                        material_fraction=float(total[-1] / self.box_volume),
                        phases=self._phase_statistics(sums, ("gradient", "flux", "dissipation")),
                        info=info, fields=fld, support=self.mesh if fields else None)  # fmt: skip


__all__ = ["GAUSS", "PLANES", "PLANE_CONDITIONS", "PlaneElasticProblem", "PlaneThermalProblem"]
