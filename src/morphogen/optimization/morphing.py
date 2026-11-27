"""Inverse design of active structures: the layout of two materials that expand
differently with a stimulus (heat, a diffusing species) — or of one of them and voids —
found so that a structure takes target shapes when it is stimulated: at equilibrium, or at
chosen times while the stimulus diffuses into it, the fourth dimension of 4D-printed
active composites (Mao et al. 2015; Maute et al. 2015; Hamel et al. 2019).

The design domain is a box of voxels, trilinear hexahedra as in
:mod:`morphogen.coupling.thermochemoelastic`, held by supports on some of its faces. The
density ``rho`` of a voxel is its share of the material 1 (the material 0 elsewhere),
interpolated as SIMP between two solids (Bendsøe & Sigmund 1999; Sigmund & Torquato
1997): the elasticity ``C(rho) = C_0 + rho^p (C_1 - C_0)``, the eigenstrain per unit of
the stimulus ``a(rho) = a_0 + rho^q (a_1 - a_0)``, and, when the stimulus diffuses, the
capacity and the conductivity (permeability) linearly. With voids (``void=True``) a second
density, that of the solid, scales the elasticity of the voxel by ``eps + (1 - eps)
rho_s^p`` (the eigenstrain is that of its solid). The designs are filtered (the linear
hat of radius ``filter_radius`` voxels, not periodic; Bruns & Tortorelli 2001), projected
(Heaviside, continuation of its sharpness up to ``projection``; Wang, Lazarov & Sigmund
2011), uniform along the axis ``extrude`` when given, and updated by the method of moving
asymptotes (:class:`~morphogen.optimization.mma.MMA`) under the volume constraints.

**Objective.** The targets are displacements of points (of the nodes nearest to them) at
output times: ``J = sum_k w_k sum_i (u_k,i - v_k,i)^2 / sum_k w_k sum_i v_k,i^2`` (1 for a
structure that does not move).

**States.** At equilibrium the stimulus rises by ``change`` everywhere: ``K(rho) u =
f(rho)``, ``f = sum_e F_e(rho) a_e(rho) change``. In time (``times``) the stimulus
diffuses from the faces ``faces`` held at values (finite volumes at the voxel centres,
implicit Euler steps: ``A_n theta^n = B_n theta^(n-1) + b_n``, ``A_n = V C / dt_n + K_D``,
``B_n = V C / dt_n``), and the displacement follows it quasi-statically at each output
time, ``K u_k = f(theta^(n_k))``.

**Sensitivities** (exact adjoints): ``K lambda_k = dJ/du_k``; at equilibrium ``dJ/drho_e =
lambda_e . (df_e/drho_e - dK_e/drho_e u_e)``; in time, the adjoint of the stimulus runs
backwards, ``A_n mu_n = r_n + B_(n+1) mu_(n+1)`` from ``mu_(N+1) = 0``, its sources the
loads of the displacements, ``r_(n_k) = sum_e lambda_k,e . F_e a_e``, and ``dJ/drho_e``
gains ``-sum_n mu_n^T (dA_n/drho_e theta^n - dB_n/drho_e theta^(n-1) - db_n/drho_e)``: the
capacity, the harmonic conductances of the faces of the voxel and those of the faces of the
box it touches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from morphogen.coupling.thermochemoelastic import (
    FACES,
    ThermoChemoElastic,
    ThermoChemoMaterial,
    VoxelSpace,
    constrained_dofs,
    rigid_modes,
    support_candidates,
    supports_of,
    time_steps,
)
from morphogen.homogenization.elements import hex_element
from morphogen.homogenization.solvers import LinearSolver
from morphogen.image import VoxelImage
from morphogen.mesh.voxel import HEX_CORNERS
from morphogen.optimization.filters import heaviside, heaviside_derivative
from morphogen.optimization.mma import MMA

STIMULI = ("temperature", "activity")


@dataclass(frozen=True)
class MorphingTarget:
    """The displacements to reach: the component ``component`` (0, 1, 2) of the
    displacements of the nodes nearest to ``points`` (rows of coordinates) should be
    ``values`` (one per point, or one for all) at the output time of index ``time``, with
    the weight ``weight``."""

    points: tuple
    component: int
    values: tuple
    time: int = -1
    weight: float = 1.0

    def __post_init__(self) -> None:
        pts = np.atleast_2d(np.asarray(self.points, dtype=float))
        if pts.shape[1] != 3:
            raise ValueError("points: rows of three coordinates")
        vals = np.broadcast_to(np.asarray(self.values, dtype=float), (len(pts),))
        if self.component not in (0, 1, 2):
            raise ValueError("component: 0, 1 or 2")
        if not self.weight > 0:
            raise ValueError("weight: positive")
        object.__setattr__(self, "points", tuple(map(tuple, pts)))
        object.__setattr__(self, "values", tuple(map(float, vals)))


def _along_axes(value) -> np.ndarray:
    """The components along the axes of a coefficient (one value, three principal values
    or a 3 x 3 tensor)."""
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        return np.full(3, float(a))
    return a.copy() if a.shape == (3,) else np.diag(a).copy()


def hat_filter(shape, radius: float) -> sp.csr_matrix:
    """The density filter of a box of voxels (not periodic): the mean over the voxels
    closer than ``radius`` voxels, weighted by ``radius - distance`` (rows normalized; the
    identity below one voxel)."""
    shape = tuple(int(s) for s in shape)
    n = int(np.prod(shape))
    if radius < 1.0:
        return sp.identity(n, format="csr")
    reach = int(np.ceil(radius)) - 1 if float(radius).is_integer() else int(np.floor(radius))
    ranges = [np.arange(-reach, reach + 1) if s > 1 else np.zeros(1, int) for s in shape]
    offsets = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(3, -1).T
    dist = np.linalg.norm(offsets, axis=1)
    keep = dist < radius
    offsets, weights = offsets[keep], radius - dist[keep]
    idx = np.indices(shape).reshape(3, -1).T
    rows, cols, vals = [], [], []
    for o, w in zip(offsets, weights, strict=True):
        other = idx + o
        ok = np.all((other >= 0) & (other < np.array(shape)), axis=1)
        rows.append(np.flatnonzero(ok))
        cols.append(np.ravel_multi_index(other[ok].T, shape))
        vals.append(np.full(int(ok.sum()), w))
    m = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    ).tocsr()
    return sp.diags(1.0 / np.asarray(m.sum(axis=1)).ravel()) @ m


@dataclass
class MorphingDesign:
    """The result of an optimization: the densities of the material 1 (``density``) and of
    the solid (``solid``, with voids; else None) of the voxels, after the filter and the
    projection; the design variables; the objective and its history; the displacements
    of the nodes at the output times (nodes ``n + 1`` per axis, times first); the
    displacements reached at the points of the targets and those wanted."""

    density: np.ndarray
    solid: np.ndarray | None
    design: np.ndarray
    objective: float
    displacement: np.ndarray
    reached: list
    wanted: list
    iterations: int
    converged: bool
    history: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    time_s: float = 0.0

    def phases(self, threshold: float = 0.5) -> np.ndarray:
        """The layout of the phases of the voxels: 0 and 1, the materials, 2 the voids."""
        out = (self.density > threshold).astype(np.uint8)
        if self.solid is not None:
            out[self.solid <= threshold] = 2
        return out

    @property
    def grey(self) -> float:
        """The share of the voxels whose densities stay between 0.1 and 0.9."""
        d = self.density if self.solid is None else np.minimum(self.density, self.solid)
        return float(np.mean((d > 0.1) & (d < 0.9)))


class MorphingOptimization:
    """The layout of the material 1 in the material 0 (and voids, ``void=True``) in a box
    of ``shape`` voxels of edges ``spacing``, for the ``targets``
    (:class:`MorphingTarget`).

    The stimulus is the ``temperature`` (the eigenstrain per kelvin is the thermal
    expansion of the materials, :class:`~morphogen.coupling.ThermoChemoMaterial`; their
    heat capacity and conductivity if it diffuses) or the ``activity`` of a species (its
    swelling per unit of activity ``beta S``; capacity ``S`` and permeability ``D S``). At
    equilibrium it rises by ``change`` everywhere; with output ``times`` it diffuses from
    the ``faces`` held at values (numbers or functions of time) from ``initial`` (the
    strains count from it), in ``steps`` implicit Euler steps per interval (those of the
    first growing geometrically), and the targets name the output times by index.
    ``supports``: the faces ``"clamped"`` or in ``"symmetry"`` (by default the face
    ``x-`` clamped); the rigid motions left are removed by statically determinate
    supports. ``volume_fraction``: the largest share of the box of the material 1 (with
    voids, of the box); ``solid_fraction``: that of the solid (with voids). ``penalty``
    and ``eigen_penalty``: the exponents ``p`` and ``q``; ``void_stiffness``: ``eps``;
    ``filter_radius`` in voxels; ``extrude``: the axis along which the design is uniform;
    ``projection``: the largest sharpness of the projection (0: none), doubled from 1 every
    ``continuation`` iterations (or when the design settles); ``move``, ``iterations``,
    ``tolerance``: those of the method of moving asymptotes (the largest change of a design
    variable at which a sharp design has converged)."""

    def __init__(
        self,
        shape,
        spacing,
        materials,
        targets,
        *,
        stimulus: str = "temperature",
        change: float = 1.0,
        times=None,
        faces: dict | None = None,
        initial: float = 0.0,
        steps: int = 10,
        supports: dict | None = None,
        void: bool = False,
        volume_fraction: float = 0.5,
        solid_fraction: float | None = None,
        penalty: float = 3.0,
        eigen_penalty: float = 1.0,
        void_stiffness: float = 1e-6,
        filter_radius: float = 2.0,
        extrude: int | None = None,
        projection: float = 32.0,
        continuation: int = 30,
        move: float = 0.2,
        iterations: int = 200,
        tolerance: float = 2e-3,
    ):
        self.shape = tuple(int(s) for s in shape)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError("shape: three positive numbers of voxels")
        self.spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,)).copy()
        if stimulus not in STIMULI:
            raise ValueError(f"stimulus: {', '.join(STIMULI)}")
        materials = tuple(
            m if isinstance(m, ThermoChemoMaterial) else ThermoChemoMaterial(**m) for m in materials
        )
        if len(materials) != 2 or any(m.void for m in materials):
            raise ValueError("materials: the two solids of densities 0 and 1")
        self.materials = materials
        self.stimulus = stimulus
        self.change = float(change)
        self.transient = times is not None
        self.times = None if times is None else np.atleast_1d(np.asarray(times, dtype=float))
        self.faces = dict(faces or {})
        if self.transient and not self.faces:
            raise ValueError("faces: the faces held at a value, in time")
        for name in self.faces:
            if name not in FACES:
                raise ValueError(f"faces: among {', '.join(FACES)}")
        self.initial = float(initial)
        self.steps = int(steps)
        self.supports = {"x-": "clamped"} if supports is None else dict(supports)
        self.void = bool(void)
        if self.void and self.transient:
            raise ValueError("void: at equilibrium only (the stimulus would enter the voids)")
        self.volume_fraction = float(volume_fraction)
        self.solid_fraction = None if solid_fraction is None else float(solid_fraction)
        if not 0.0 < self.volume_fraction <= 1.0:
            raise ValueError("volume_fraction: in (0, 1]")
        if self.void and not (
            self.solid_fraction is not None and self.volume_fraction <= self.solid_fraction <= 1
        ):
            raise ValueError("solid_fraction: with voids, between volume_fraction and 1")
        self.penalty, self.eigen_penalty = float(penalty), float(eigen_penalty)
        if self.penalty < 1.0 or self.eigen_penalty < 1.0:
            raise ValueError("penalty, eigen_penalty: at least 1")
        self.void_stiffness = float(void_stiffness)
        if not 0.0 < self.void_stiffness < 0.1:
            raise ValueError("void_stiffness: in (0, 0.1)")
        self.filter_radius = float(filter_radius)
        self.extrude = None if extrude is None else int(extrude)
        if self.extrude not in (None, 0, 1, 2):
            raise ValueError("extrude: an axis (0, 1, 2) or None")
        self.projection = float(projection)
        if self.projection < 0:
            raise ValueError("projection: a non-negative sharpness (0: none)")
        self.continuation = int(continuation)
        self.move, self.iterations = float(move), int(iterations)
        self.tolerance = float(tolerance)
        self._mechanics()
        self._targets(targets)
        if self.transient:
            self.space = VoxelSpace(self.shape, self.spacing)
            self._steps = time_steps(self.times, self.steps)
            self._outputs = [n for n, (_, out) in enumerate(self._steps) if out]
        reduced = list(self.shape)
        if self.extrude is not None:
            reduced[self.extrude] = 1
        self.design_shape = tuple(reduced)
        self.filter = hat_filter(self.design_shape, self.filter_radius)

    # -- set-up ------------------------------------------------------------------------------

    def _coefficients(self, m: ThermoChemoMaterial):
        """The eigenstrain per unit of the stimulus, the capacity and the conductivities
        along the axes of a material."""
        if self.stimulus == "temperature":
            return m.thermal_strain(), float(m.heat_capacity), _along_axes(m.conductivity)
        return (
            m.chemical_strain() * m.solubility,
            float(m.solubility),
            _along_axes(m.diffusivity) * m.solubility,
        )

    def _mechanics(self) -> None:
        shape, h = self.shape, self.spacing
        nodes_shape = tuple(s + 1 for s in shape)
        self.nodes_shape = nodes_shape
        idx = np.indices(nodes_shape).reshape(3, -1).T
        self.coordinates = idx * h
        n_nodes = int(np.prod(nodes_shape))
        ijk = np.indices(shape).reshape(3, -1).T
        corners = ijk[:, None, :] + HEX_CORNERS[None, :, :]
        ny, nz = nodes_shape[1], nodes_shape[2]
        self.element_nodes = (corners[..., 0] * ny + corners[..., 1]) * nz + corners[..., 2]
        self.element_dofs = (self.element_nodes[:, :, None] * 3 + np.arange(3)).reshape(-1, 24)
        element = hex_element(tuple(float(v) for v in h))
        volume = float(np.prod(h))
        self.cell_volume = volume
        b_mean = np.asarray(element.b_mean)
        k, f, a, cap, kappa = [], [], [], [], []
        for m in self.materials:
            d = m.elastic_matrix()
            k.append(element.weight * np.einsum("gai,ab,gbj->ij", element.b_gauss, d,
                                                element.b_gauss))  # fmt: skip
            f.append(volume * b_mean.T @ d)
            eig, c, cond = self._coefficients(m)
            a.append(eig)
            cap.append(c)
            kappa.append(cond)
        self.k0, self.dk = k[0], k[1] - k[0]
        self.f0, self.df = f[0], f[1] - f[0]
        self.a0, self.da = a[0], a[1] - a[0]
        self.c0, self.dc = cap[0], cap[1] - cap[0]
        self.p0, self.dp = np.asarray(kappa[0]), np.asarray(kappa[1]) - np.asarray(kappa[0])
        n_el = len(self.element_dofs)
        self._rows = np.broadcast_to(self.element_dofs[:, :, None], (n_el, 24, 24)).ravel()
        self._cols = np.broadcast_to(self.element_dofs[:, None, :], (n_el, 24, 24)).ravel()
        node_idx = idx.T

        def face_nodes(name):
            axis, side = FACES.index(name) // 2, FACES.index(name) % 2
            return np.flatnonzero(node_idx[axis] == (0 if side == 0 else nodes_shape[axis] - 1))

        used = np.ones(n_nodes, dtype=bool)
        fixed = constrained_dofs(self.supports, face_nodes, used)
        modes = rigid_modes(self.coordinates)
        supported = supports_of(modes, np.flatnonzero(fixed),
                                support_candidates(self.coordinates, ~fixed))  # fmt: skip
        fixed[supported] = True
        self.free = np.flatnonzero(~fixed)
        self.n_dofs = 3 * n_nodes

    def _targets(self, targets) -> None:
        targets = [targets] if isinstance(targets, MorphingTarget) else list(targets)
        if not targets:
            raise ValueError("targets: at least one")
        n_out = 1 if not self.transient else len(self.times)
        tree = cKDTree(self.coordinates)
        self.targets = targets
        self._target_dofs = []
        norm = 0.0
        for t in targets:
            k = t.time if t.time >= 0 else n_out + t.time
            if not 0 <= k < n_out:
                raise ValueError(f"targets: time {t.time} out of the {n_out} output times")
            nodes = tree.query(np.asarray(t.points))[1]
            self._target_dofs.append((k, 3 * nodes + t.component, np.asarray(t.values),
                                      float(t.weight)))  # fmt: skip
            norm += t.weight * float(np.sum(np.square(t.values)))
        if norm <= 0:
            raise ValueError("targets: some displacement to reach (not all zero)")
        self._norm = norm

    def bending_target(self, curvature: float, *, axis: int = 0, normal: int = 2, time: int = -1,
                       weight: float = 1.0) -> MorphingTarget:  # fmt: skip
        """The target of a strip along ``axis``, clamped at its lower face along that axis,
        that bends to ``curvature`` in the plane of ``normal`` (positive: towards its lower
        face): the deflections ``-k s^2 / 2`` of the line of nodes of the middle of its
        section, ``s`` the distance from the clamped face."""
        other = 3 - axis - normal
        x = self.coordinates
        mid_n = 0.5 * x[:, normal].max()
        mid_o = 0.5 * x[:, other].max()
        levels_n = np.unique(x[:, normal])
        levels_o = np.unique(x[:, other])
        ln = levels_n[np.argmin(np.abs(levels_n - mid_n))]
        lo = levels_o[np.argmin(np.abs(levels_o - mid_o))]
        line = (x[:, normal] == ln) & (x[:, other] == lo) & (x[:, axis] > 0)
        pts = x[line]
        s = pts[:, axis]
        return MorphingTarget(pts, normal, -0.5 * curvature * s**2, time=time, weight=weight)

    # -- densities -----------------------------------------------------------------------------

    def _fields(self) -> int:
        return 2 if self.void else 1

    def densities(self, x, beta: float):
        """The densities of the voxels (the material 1, and the solid with voids) of the
        design variables ``x``, after the filter, the projection and the extrusion, and the
        derivatives of the projection."""
        n = int(np.prod(self.design_shape))
        x = np.asarray(x, dtype=float).reshape(self._fields(), n)
        out, dproj = [], []
        for field_x in x:
            filtered = self.filter @ field_x
            rho = heaviside(filtered, beta) if beta > 0 else filtered
            d = heaviside_derivative(filtered, beta) if beta > 0 else np.ones(n)
            out.append(self._extruded(rho))
            dproj.append(d)
        return out, dproj

    def _extruded(self, values: np.ndarray) -> np.ndarray:
        v = values.reshape(self.design_shape)
        if self.extrude is not None:
            v = np.repeat(v, self.shape[self.extrude], axis=self.extrude)
        return v.ravel()

    def _reduced(self, g: np.ndarray) -> np.ndarray:
        """The transpose of the extrusion: the sums along its axis."""
        g = g.reshape(self.shape)
        if self.extrude is not None:
            g = g.sum(axis=self.extrude, keepdims=True)
        return g.ravel()

    # -- the state and its adjoint ------------------------------------------------------------

    def _interpolations(self, rho, solid):
        p, q, eps = self.penalty, self.eigen_penalty, self.void_stiffness
        r = np.clip(rho, 0.0, 1.0)
        s = r**p
        ds = p * np.maximum(r, 1e-12) ** (p - 1.0)
        e = r**q
        de = q * np.maximum(r, 1e-12) ** (q - 1.0)
        if solid is None:
            t, dt = np.ones_like(r), np.zeros_like(r)
        else:
            rs = np.clip(solid, 0.0, 1.0)
            t = eps + (1.0 - eps) * rs**p
            dt = (1.0 - eps) * p * np.maximum(rs, 1e-12) ** (p - 1.0)
        return s, ds, e, de, t, dt

    def _stiffness(self, s, t) -> LinearSolver:
        data = t[:, None, None] * (self.k0[None] + s[:, None, None] * self.dk[None])
        k = sp.coo_matrix((data.ravel(), (self._rows, self._cols)),
                          shape=(self.n_dofs, self.n_dofs)).tocsr()  # fmt: skip
        k.sum_duplicates()
        return LinearSolver(k[self.free][:, self.free].tocsr(), method="auto", tol=1e-12)

    def _unit_loads(self, s, e, t) -> np.ndarray:
        """The nodal forces of each voxel per unit of its stimulus (elements x 24)."""
        f = t[:, None, None] * (self.f0[None] + s[:, None, None] * self.df[None])
        a = self.a0[None] + e[:, None] * self.da[None]
        return np.einsum("eij,ej->ei", f, a)

    def _scatter(self, element_values: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_dofs)
        np.add.at(out, self.element_dofs.ravel(), element_values.ravel())
        return out

    def _solve(self, solver: LinearSolver, f: np.ndarray) -> np.ndarray:
        u = np.zeros(self.n_dofs)
        u[self.free] = solver.solve(f[self.free])[0]
        return u

    def _transport(self, rho):
        """The capacities (lumped), the conductances of the faces and the loads of the faces
        held, of the densities ``rho``."""
        space = self.space
        cap = space.lump(self.c0 + rho * self.dc)
        kappa = self.p0[None, :] + rho[:, None] * self.dp[None, :]
        g = space.conductances(kappa)
        loads = space.face_loads(kappa, self.faces)
        return cap, kappa, g, loads

    def objective(self, rho, solid=None, *, gradient: bool = True):
        """The objective of the densities of the voxels (the material 1; the solid with
        voids), its derivatives with respect to them (or None) and the displacements of the
        nodes at the output times (times x nodes x 3)."""
        s, ds, e, de, t, dt = self._interpolations(rho, solid)
        solver = self._stiffness(s, t)
        unit = self._unit_loads(s, e, t)
        n_out = len(self.times) if self.transient else 1
        if self.transient:
            thetas, solvers, cap, kappa, g, loads = self._diffuse(rho)
            stimuli = [thetas[n] - self.initial for n in self._outputs]
        else:
            stimuli = [np.full(len(rho), self.change)] * n_out
        disp = [self._solve(solver, self._scatter(unit * st[:, None])) for st in stimuli]
        value = 0.0
        dj = [np.zeros(self.n_dofs) for _ in range(n_out)]
        for k, dofs, values, w in self._target_dofs:
            r = disp[k][dofs] - values
            value += w * float(r @ r)
            np.add.at(dj[k], dofs, 2.0 * w * r / self._norm)
        value /= self._norm
        u_out = np.array([u.reshape(-1, 3) for u in disp])
        if not gradient:
            return value, None, u_out
        g_rho = np.zeros(len(rho))
        g_solid = None if solid is None else np.zeros(len(rho))
        sources = []
        for k in range(n_out):
            lam = self._solve(solver, dj[k])
            lam_e = lam[self.element_dofs]
            u_e = disp[k][self.element_dofs]
            st = stimuli[k]
            ku = np.einsum("ij,ej->ei", self.dk, u_e)
            f_sol = self.f0[None] + s[:, None, None] * self.df[None]
            a = self.a0[None] + e[:, None] * self.da[None]
            dload = (ds[:, None] * (self.df @ a.T).T + np.einsum("eij,j->ei", f_sol, self.da)
                     * de[:, None])  # fmt: skip
            g_rho += np.einsum("ei,ei->e", lam_e, t[:, None] * (dload * st[:, None]
                                                                - ds[:, None] * ku))  # fmt: skip
            if solid is not None:
                k_sol = self.k0[None] + s[:, None, None] * self.dk[None]
                ksu = np.einsum("eij,ej->ei", k_sol, u_e)
                fa = np.einsum("eij,ej->ei", f_sol, a)
                g_solid += dt * np.einsum("ei,ei->e", lam_e, fa * st[:, None] - ksu)
            sources.append(np.einsum("ei,ei->e", lam_e, unit))
        if self.transient:
            g_rho += self._transport_gradient(rho, thetas, solvers, sources, cap, kappa, g, loads)
        return value, (g_rho, g_solid), u_out

    def _diffuse(self, rho):
        """The stimulus at every step (implicit Euler), and the solvers of each step length."""
        space = self.space
        cap, kappa, g, loads = self._transport(rho)
        lap = space.laplacian(g)
        if loads:
            lap = lap + sp.diags(sum(loads.values()))
        theta = np.full(space.n, self.initial)
        thetas = [theta]
        solvers: dict = {}
        t = 0.0
        for dt, _ in self._steps:
            if dt > 0:
                if dt not in solvers:
                    solvers[dt] = LinearSolver((sp.diags(cap / dt) + lap).tocsr(),
                                               method="auto", tol=1e-13)  # fmt: skip
                t += dt
                rhs = cap / dt * theta
                for name, vec in loads.items():
                    value = self.faces[name]
                    rhs = rhs + vec * (float(value(t)) if callable(value) else float(value))
                theta = solvers[dt].solve(rhs)[0]
            thetas.append(theta)
        return thetas[1:], solvers, cap, kappa, g, loads

    def _transport_gradient(self, rho, thetas, solvers, sources, cap, kappa, g, loads):
        """The part of the derivatives that goes through the stimulus: the adjoint of the
        steps run backwards."""
        space = self.space
        n_steps = len(self._steps)
        r = [None] * n_steps
        for k, n in enumerate(self._outputs):
            r[n] = sources[k] if r[n] is None else r[n] + sources[k]
        grad = np.zeros(len(rho))
        mu_next = np.zeros(space.n)
        dt_next = None
        i, j, axis = space.face_i, space.face_j, space.face_axis
        pi, pj = kappa[i, axis], kappa[j, axis]
        dpa = self.dp[axis]
        den = (pi + pj) ** 2
        dgi = np.divide(2.0 * pj * pj, den, out=np.zeros_like(den), where=den > 0)
        dgj = np.divide(2.0 * pi * pi, den, out=np.zeros_like(den), where=den > 0)
        dgi *= space.face_ratio * dpa
        dgj *= space.face_ratio * dpa
        times = np.cumsum([dt for dt, _ in self._steps])
        for n in range(n_steps - 1, -1, -1):
            dt, _ = self._steps[n]
            theta = thetas[n]
            previous = thetas[n - 1] if n > 0 else np.full(space.n, self.initial)
            if dt == 0:  # an output at the start: the initial state, which no design moves
                continue
            rhs = np.zeros(space.n) if r[n] is None else r[n].copy()
            if dt_next is not None:
                rhs += cap / dt_next * mu_next
            mu = solvers[dt].solve(rhs)[0]
            dcap = space.cell_volume * self.dc
            grad -= mu * dcap * (theta - previous) / dt
            flux = (mu[i] - mu[j]) * (theta[i] - theta[j])
            np.add.at(grad, i, -dgi * flux)
            np.add.at(grad, j, -dgj * flux)
            for name in loads:
                cells, a, ratio = space.box[name]
                value = self.faces[name]
                v = float(value(times[n])) if callable(value) else float(value)
                contrib = np.zeros(space.n)
                np.add.at(contrib, cells, self.dp[a] * ratio * mu[cells] * (theta[cells] - v))
                grad -= contrib
            mu_next, dt_next = mu, dt
        return grad

    # -- the optimization ----------------------------------------------------------------------

    def evaluate(self, x, beta: float = 0.0, *, gradient: bool = True):
        """The objective of the design variables ``x`` (the densities of the design grid,
        those of the solid after them with voids) and its derivatives with respect to them,
        for the sharpness ``beta`` of the projection (0: none)."""
        (rho, *rest), dproj = self.densities(x, beta)
        solid = rest[0] if rest else None
        value, grads, _ = self.objective(rho, solid, gradient=gradient)
        if not gradient:
            return value, None
        out = []
        for gd, dp in zip([grads[0]] + ([grads[1]] if solid is not None else []), dproj,
                          strict=True):  # fmt: skip
            out.append(self.filter.T @ (dp * self._reduced(gd)))
        return value, np.concatenate(out)

    def _volumes(self, x, beta):
        """The constraints (shares over their bounds, minus 1) and their derivatives."""
        (rho, *rest), dproj = self.densities(x, beta)
        n = len(rho)
        if not self.void:
            value = float(rho.mean()) / self.volume_fraction - 1.0
            grad = self.filter.T @ (dproj[0] * self._reduced(np.full(n, 1.0 / n)))
            return np.array([value]), grad[None, :] / self.volume_fraction
        solid = rest[0]
        nd = int(np.prod(self.design_shape))
        g1 = np.zeros(2 * nd)
        g1[:nd] = self.filter.T @ (dproj[0] * self._reduced(solid / n))
        g1[nd:] = self.filter.T @ (dproj[1] * self._reduced(rho / n))
        g2 = np.zeros(2 * nd)
        g2[nd:] = self.filter.T @ (dproj[1] * self._reduced(np.full(n, 1.0 / n)))
        values = np.array([float((rho * solid).mean()) / self.volume_fraction - 1.0,
                           float(solid.mean()) / self.solid_fraction - 1.0])  # fmt: skip
        return values, np.vstack([g1 / self.volume_fraction, g2 / self.solid_fraction])

    def run(self, x0=None, callback=None) -> MorphingDesign:
        """Optimize from ``x0`` (by default uniform densities at their bounds);
        ``callback(iteration, densities, objective)`` is called at every iteration and
        stops the optimization when it returns True."""
        t0 = time.perf_counter()
        nd = int(np.prod(self.design_shape))
        if x0 is None:
            parts = [np.full(nd, self.volume_fraction)]
            if self.void:
                parts.append(np.full(nd, self.solid_fraction))
                parts[0] = np.full(nd, min(1.0, self.volume_fraction / self.solid_fraction))
            x = np.concatenate(parts)
        else:
            x = np.asarray(x0, dtype=float).ravel().copy()
        n_cons = 2 if self.void else 1
        mma = MMA(x.size, n_cons, 0.0, 1.0, move=self.move)
        beta = 1.0 if self.projection > 0 else 0.0
        history = {"objective": [], "change": [], "beta": [], "volume": []}
        converged = False
        value = np.nan
        it = 0
        for it in range(1, self.iterations + 1):
            value, grad = self.evaluate(x, beta)
            cons, dcons = self._volumes(x, beta)
            x_new = np.clip(mma.update(x, 10.0 * value, 10.0 * grad, cons, dcons), 0.0, 1.0)
            change = float(np.max(np.abs(x_new - x)))
            x = x_new
            history["objective"].append(float(value))
            history["change"].append(change)
            history["beta"].append(beta)
            history["volume"].append(float(cons[0] + 1.0) * self.volume_fraction)
            if callback is not None and callback(it, self.densities(x, beta)[0], value) is True:
                break
            sharp = beta == 0.0 or beta >= self.projection
            if change < self.tolerance and sharp:
                converged = True
                break
            if beta > 0 and not sharp and (change < self.tolerance or it % self.continuation == 0):
                beta = min(2.0 * beta, self.projection)
                mma.reset()
        (rho, *rest), _ = self.densities(x, beta)
        solid = rest[0] if rest else None
        value, _, disp = self.objective(rho, solid, gradient=False)
        reached, wanted = [], []
        for k, dofs, values, _ in self._target_dofs:
            reached.append(disp[k].ravel()[dofs])
            wanted.append(values)
        shape = (len(disp), *self.nodes_shape, 3)
        return MorphingDesign(
            density=rho.reshape(self.shape),
            solid=None if solid is None else solid.reshape(self.shape),
            design=x, objective=float(value), displacement=disp.reshape(shape),
            reached=reached, wanted=wanted, iterations=it, converged=converged,
            history=history, settings=self.settings(), time_s=time.perf_counter() - t0,
        )  # fmt: skip

    def settings(self) -> dict:
        """The parameters of the optimization."""
        return {
            "shape": self.shape, "spacing": tuple(map(float, self.spacing)),
            "stimulus": self.stimulus, "change": self.change,
            "times": None if self.times is None else self.times.tolist(),
            "faces": {k: v for k, v in self.faces.items() if not callable(v)},
            "initial": self.initial, "steps": self.steps, "supports": self.supports,
            "void": self.void, "volume_fraction": self.volume_fraction,
            "solid_fraction": self.solid_fraction, "penalty": self.penalty,
            "eigen_penalty": self.eigen_penalty, "filter_radius": self.filter_radius,
            "extrude": self.extrude, "projection": self.projection,
            "continuation": self.continuation, "move": self.move,
            "iterations": self.iterations, "tolerance": self.tolerance,
        }  # fmt: skip

    def verify(self, design: MorphingDesign, threshold: float = 0.5):
        """The layout of the phases of a design (densities above ``threshold``: the
        material 1; with voids, solids below it removed) solved by
        :class:`~morphogen.coupling.ThermoChemoElastic`, voids true pores: its result and
        the displacements it reaches at the points of the targets (one array per
        target)."""
        phases = design.phases(threshold)
        image = VoxelImage(phases, spacing=tuple(self.spacing))
        mats = {0: self.materials[0], 1: self.materials[1], 2: ThermoChemoMaterial(void=True)}
        mats = {p: m for p, m in mats.items() if np.any(phases == p)}
        supports = {k: v for k, v in self.supports.items()}
        offset = 300.0
        if self.stimulus == "temperature":
            held = {k: (lambda t, v=v: offset + (v(t) if callable(v) else v))
                    for k, v in self.faces.items()}  # fmt: skip
            options = {"reference_temperature": offset + (self.initial if self.transient else 0.0),
                       "initial_temperature": offset + (self.initial if self.transient
                                                        else self.change),
                       "temperature": held}  # fmt: skip
        else:
            options = {"initial_activity": self.initial if self.transient else self.change,
                       "reference_activity": self.initial if self.transient else 0.0,
                       "activity": self.faces}  # fmt: skip
        problem = ThermoChemoElastic(image, mats, supports=supports, periodic=False, **options)
        times = self.times if self.transient else [0.0]
        result = problem.run(times, steps=self.steps)
        tree = cKDTree(result.nodes)
        reached = []
        for target in self.targets:
            k = target.time if target.time >= 0 else len(times) + target.time
            nodes = tree.query(np.asarray(target.points))[1]
            reached.append(result.displacement[k].reshape(-1, 3)[nodes, target.component])
        return result, reached
