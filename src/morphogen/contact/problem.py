"""Bodies in contact: the quasi-static solver of :mod:`morphogen.fracture.cohesive` with
the forces of the surfaces that touch (see :mod:`morphogen.contact`)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from morphogen.contact import kernels
from morphogen.contact.kernels import (
    P_BEHIND,
    P_DEPTH,
    P_MU,
    P_RIM,
    P_SHARE,
    P_SIZE,
    P_STICK,
    P_STIFF,
    P_TOUCH,
    RIM,
    RIM_NODES,
    S_FACE,
    S_GAP,
    S_LAMBDA,
    S_PRESSURE,
    S_SHARE,
    S_TX,
    S_WORK,
    STATE,
)
from morphogen.contact.surfaces import Obstacle, Pair, Surface
from morphogen.fracture.cohesive import (
    CohesiveProblem,
    CohesiveResult,
    Contact,
    Interfaces,
    plus,
)
from morphogen.mesh.tetra import TetMesh


class _Pass:
    """The slave points of a surface against a master surface or a rigid obstacle: their
    faces, coordinates and weights, the parameters of the contact, their state at the end
    of the last step and that of the last evaluation."""

    def __init__(self, slave: Surface, master: Surface | None, own: bool, obstacle, nodes, par,
                 shares: int = 1, rule: str = "gauss"):  # fmt: skip
        self.slave, self.master, self.own, self.obstacle = slave.faces, None, own, obstacle
        #: the passes of the contact: each carries its share of the pressure
        self.shares = int(shares)
        self.rims = np.zeros((0, 3), dtype=np.bool_)
        if master is not None:
            self.master = master.faces
            self.rims = np.ascontiguousarray(master.rims())
        self.surface = slave
        self.reference = np.ascontiguousarray(nodes, dtype=float)  # the nodes at rest
        #: the node that each is (the images of a periodic cell: see its problem)
        self.ident = np.arange(len(self.reference), dtype=np.int64)
        self.face, self.xi, self.eta, self.weight = slave.points(nodes, rule)
        self.par = par
        self.state = self.fresh()
        self.trial = self.state.copy()
        self.multiplier = np.zeros(len(self.face))
        self.pressed = np.zeros(len(self.face), dtype=bool)
        self.resultant = np.zeros(3)
        self.last = np.full(len(self.face), -1, dtype=np.int64)  # the faces last seen
        # the reach of the search: the deepest penetration, and the size of the faces
        self.reach = float(par[P_DEPTH]) + 2.0 * slave.size(nodes)

    def fresh(self) -> np.ndarray:
        """The state of points that face nothing."""
        state = np.zeros((len(self.face), STATE))
        state[:, S_SHARE] = 1.0
        state[:, S_FACE] = -1.0
        state[:, S_GAP] = np.inf
        return state

    def near(self, x: np.ndarray, neighbours: int):
        """The slave points that have master faces within reach, and these faces (rows:
        the nearest by their centroids; -1: none)."""
        points = self.surface.positions(x, self.face, self.xi, self.eta)
        centroids = x[self.master[:, :3]].mean(axis=1)
        count = min(int(neighbours), len(centroids))
        # a face holds points up to its size away from its centroid
        _, found = cKDTree(centroids).query(points, k=count, distance_upper_bound=self.reach)
        found = np.asarray(found).reshape(len(points), count)
        found = np.where(found >= len(centroids), -1, found)
        index = np.flatnonzero(found[:, 0] >= 0)
        return index, np.ascontiguousarray(found[index], dtype=np.int64)


def contact_forces(passes, x, x_old, neighbours: int, target: float, tangent: bool, predict: bool):
    """The forces of the passes of contacts on the dofs of the nodes at the positions
    ``x`` (``x_old``: at the end of the last step) and, with ``tangent``, their tangent
    (None if no point presses); ``target``: the load factor, which places the obstacles;
    ``predict``: the tangent of the predictor of a step. The state of the slave points of
    this evaluation is kept in their passes (``trial``, ``pressed``, ``resultant``,
    ``last``)."""
    n = x.size
    f = np.zeros(n)
    rows, cols, values = [], [], []
    for p in passes:
        ns = p.slave.shape[1]
        if p.obstacle is None:
            index, candidates = p.near(x, neighbours)
            nd = 3 * (ns + p.master.shape[1])
        else:
            index, candidates = np.arange(len(p.face)), None
            nd = 3 * ns
        trial = p.fresh()
        p.pressed = np.zeros(len(p.face), dtype=bool)
        p.resultant = np.zeros(3)
        if not len(index):
            p.trial = trial
            continue

        def evaluate(index, candidates, tangent, p=p, nd=nd):
            count = len(index)
            fe = np.zeros((count, nd))
            ke = np.zeros((count if tangent else 1, nd, nd))
            new = np.zeros((count, STATE))
            active = np.zeros(count, dtype=np.bool_)
            old = np.ascontiguousarray(p.state[index])
            shared = (p.face[index], p.xi[index], p.eta[index], p.weight[index])
            if p.obstacle is None:
                nodes = np.zeros((count, nd // 3), dtype=np.int64)
                last = np.ascontiguousarray(p.last[index])
                kernels.pair_forces(
                    x, x_old, p.reference, p.ident, p.slave, p.master, p.rims, *shared,
                    candidates, last, p.par, old, new, p.own, tangent, predict, fe, ke, nodes,
                    active,
                )  # fmt: skip
            else:
                geometry = p.obstacle.geometry(target)
                kernels.obstacle_forces(x, p.slave, *shared, p.obstacle.kind, geometry, p.par,
                                        old, new, tangent, predict, fe, ke, active)  # fmt: skip
                nodes = p.slave[p.face[index]]
            return fe, ke, new, active, nodes

        fe, ke, new, active, nodes = evaluate(index, candidates, False)
        trial[index] = new
        p.trial = trial
        if not active.any():
            continue
        on = np.flatnonzero(active)
        if tangent:  # of the points that press (or that the predictor closes) alone
            chosen = None if candidates is None else np.ascontiguousarray(candidates[on])
            fe_on, ke, _, still, nodes_on = evaluate(index[on], chosen, True)
            fe_on, ke, nodes_on = fe_on[still], ke[still], nodes_on[still]
        else:
            fe_on, nodes_on = fe[on], nodes[on]
        dofs = (3 * nodes_on[:, :, None] + np.arange(3)).reshape(len(nodes_on), nd)
        np.add.at(f, dofs.ravel(), fe_on.ravel())
        if tangent:
            rows.append(np.repeat(dofs, nd, axis=1).ravel())
            cols.append(np.tile(dofs, (1, nd)).ravel())
            values.append(ke.ravel())
        p.pressed[index] = new[:, S_PRESSURE] > 0.0
        # the force of the slave surface on its master, or on its obstacle
        p.resultant = fe[on][:, : 3 * ns].reshape(len(on), ns, 3).sum(axis=(0, 1))
    for p in passes:  # the faces that the points keep at the next evaluation
        p.last = p.trial[:, S_FACE].astype(np.int64)
    k = None
    if tangent and rows:
        k = sp.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )
    return f, k


def contact_pass(nodes, slave, master, own, obstacle, spec, stiffest: float, shares: int = 1):
    """A pass of a contact: the slave points of ``slave`` against ``master`` (or against
    the ``obstacle``), with the parameters of ``spec`` (a pair or an obstacle); the two
    passes of a symmetric pair, or the two sides of a surface that touches itself, share
    the penalty and the stick, so that both together are those asked for."""
    size = slave.size(nodes)
    if master is not None:  # the penalty: that of the coarser of the two surfaces
        size = max(size, master.size(nodes))
    stiffness = 10.0 * stiffest / size if spec.stiffness is None else float(spec.stiffness)
    stick = stiffness if spec.tangential is None else float(spec.tangential)
    depth = getattr(spec, "depth", None)
    rule = getattr(spec, "points", "gauss")
    par = np.zeros(9)
    par[P_RIM] = RIM_NODES if rule == "nodes" else RIM
    par[P_SHARE] = 1.0 if rule == "nodes" else 0.0
    par[P_STIFF], par[P_STICK] = stiffness / shares, stick / shares
    par[P_MU] = float(spec.friction)
    par[P_DEPTH] = 0.5 * size if depth is None else float(depth)
    par[P_TOUCH] = 1e-9 * size
    par[P_SIZE] = size
    # a face behind a point at rest, within two faces of it (or twice the depth looked
    # for), is the other side of a wall of the body
    par[P_BEHIND] = 2.0 * max(size, par[P_DEPTH])
    return _Pass(slave, master, own, obstacle, nodes, par, shares, rule)


def stiffest_modulus(materials, phases) -> float:
    """The largest Young's modulus (or normal stiffness) of the materials of the phases
    present."""
    return max(_stiffness(m) for p, m in materials.items() if not m.void and np.any(phases == p))


@dataclass
class ContactResult:
    """The steps of a contact problem: the load factors, the reactions of the prescribed
    sets (steps x sets x 3), the Newton iterations, the final displacement (nodes x 3),
    with ``fields`` those of the steps, and for each contact (the pairs, then the
    obstacles: ``names``) at each step the ``area`` in contact (reference area of the
    slave points that press; of one of the two sides for a symmetric pair or a surface
    that touches itself), the largest ``pressure`` (the points of the two sides then each
    carry half of it: theirs are doubled) and the ``forces`` of the slave
    surface on its master or on its obstacle (steps x contacts x 3); the deepest
    ``penetration`` of each step. ``work``: at each step, ``"external"`` (of the prescribed
    displacements and forces), ``"bulk"``, ``"interfaces"`` and ``"contact"`` (of their
    internal forces: they add up to the external work; that of the contact is the energy
    of the penalty and of friction, less the work of the obstacles that move),
    ``"friction"`` (dissipated by friction) and ``"obstacles"`` (done by the obstacles on
    the body). ``points``: for each contact, the slave points at the end, a dict of their
    ``position``, ``pressure``, tangential ``traction``, ``gap``, ``weight`` (reference
    area; of the points at the nodes of the faces, the share that the other surface sees)
    and energy ``dissipated`` by friction per unit area. ``cohesive``: the steps of
    the cohesive interfaces of the problem, if any."""

    loads: np.ndarray
    reactions: np.ndarray
    iterations: np.ndarray
    displacement: np.ndarray
    area: np.ndarray
    pressure: np.ndarray
    forces: np.ndarray
    penetration: np.ndarray
    names: list
    points: list = field(default_factory=list, repr=False)
    fields: list = field(default_factory=list, repr=False)
    time_s: float = 0.0
    work: dict = field(default_factory=dict, repr=False)
    cohesive: CohesiveResult | None = field(default=None, repr=False)

    def rows(self) -> list[dict]:
        """One row per step: the load factor, the reactions of each prescribed set, the
        area, the largest pressure and the force of each contact, the deepest
        penetration and the works."""
        out = []
        for k, lam in enumerate(self.loads):
            row = {"load": float(lam)}
            for s in range(self.reactions.shape[1]):
                for c, name in enumerate("xyz"):
                    row[f"R{s + 1}{name}"] = float(self.reactions[k, s, c])
            for j, name in enumerate(self.names):
                row[f"area_{name}"] = float(self.area[k, j])
                row[f"pressure_{name}"] = float(self.pressure[k, j])
                for c, axis in enumerate("xyz"):
                    row[f"F{axis}_{name}"] = float(self.forces[k, j, c])
            row["penetration"] = float(self.penetration[k])
            for name, values in self.work.items():
                row[f"work_{name}"] = float(values[k])
            out.append(row)
        return out


class ContactProblem(CohesiveProblem):
    """A body, or several in one mesh (:func:`~morphogen.contact.surfaces.assemble`), whose
    surfaces may touch each other, themselves and rigid obstacles (see
    :mod:`morphogen.contact`): ``mesh`` (tetrahedra of 4 or 10 nodes), the ``materials`` of
    its phases, the ``pairs`` of surfaces (:class:`~morphogen.contact.surfaces.Pair`) and
    the ``obstacles`` (:class:`~morphogen.contact.surfaces.Obstacle`).

    ``kinematics`` and ``bulk``: those of :class:`~morphogen.fracture.cohesive.
    CohesiveProblem` (the gaps are always those of the deformed surfaces; in small strains
    the bulk alone is linear). ``interfaces`` and ``law``: cohesive interfaces of the mesh
    as well. ``augmentations``: the largest number of updates of the multipliers of the
    augmented Lagrangian per step (0: the penalty alone), made while a point penetrates
    deeper than ``tolerance`` (a length; by default a hundredth of the size of the
    smallest faces). ``neighbours``: the master faces tried for each slave point, the
    nearest by their centroids."""

    def __init__(
        self,
        mesh: TetMesh,
        materials,
        pairs=(),
        obstacles=(),
        *,
        kinematics: str = "finite",
        bulk: str = "auto",
        interfaces: Interfaces | None = None,
        law=None,
        augmentations: int = 0,
        tolerance: float | None = None,
        neighbours: int = 12,
    ):
        if (interfaces is None) != (law is None):
            raise ValueError("ContactProblem: cohesive interfaces and their law together")
        if interfaces is None:
            empty = np.zeros((0, 3 if mesh.order == 1 else 6), dtype=np.int64)
            interfaces = Interfaces(empty, empty, np.zeros((0, 2), dtype=np.int64))
            law = Contact(1.0)
        super().__init__(mesh, interfaces, materials, law, kinematics=kinematics, bulk=bulk)
        self.pairs, self.obstacles = list(pairs), list(obstacles)
        if not self.pairs and not self.obstacles:
            raise ValueError("ContactProblem: at least one pair of surfaces or one obstacle")
        if augmentations < 0 or neighbours < 1:
            raise ValueError("ContactProblem: augmentations >= 0, neighbours >= 1")
        self.augmentations, self.neighbours = int(augmentations), int(neighbours)
        stiffest = stiffest_modulus(materials, mesh.phase)
        self._passes: list[_Pass] = []
        self._owners: list[list[tuple[int, float]]] = []  # the passes of each contact, signed
        self.names: list[str] = []
        sizes = []
        for k, pair in enumerate(self.pairs):
            if not isinstance(pair, Pair):
                raise TypeError("ContactProblem: pairs of surfaces (Pair)")
            self._check(pair.slave)
            owners = [(len(self._passes), 1.0)]
            master = pair.slave if pair.own else pair.master
            self._check(master)
            both = pair.symmetric and not pair.own
            # the two passes of a symmetric pair, or the points of the two sides of a
            # surface that touches itself, each carry half of the pressure
            shares = 2 if both or pair.own else 1
            self._passes.append(
                self._pass(pair.slave, master, pair.own, None, pair, stiffest, shares)
            )
            if both:
                owners.append((len(self._passes), -1.0))
                self._passes.append(
                    self._pass(master, pair.slave, False, None, pair, stiffest, shares)
                )
            self._owners.append(owners)
            self.names.append(f"pair{k + 1}")
            sizes.append(pair.slave.size(mesh.nodes))
        for obstacle in self.obstacles:
            if not isinstance(obstacle, Obstacle):
                raise TypeError("ContactProblem: obstacles (Obstacle)")
            self._check(obstacle.surface)
            self._owners.append([(len(self._passes), 1.0)])
            self._passes.append(
                self._pass(obstacle.surface, None, False, obstacle, obstacle, stiffest)
            )
            taken = sum(name.startswith(obstacle.name) for name in self.names)
            self.names.append(obstacle.name + (str(taken + 1) if taken else ""))
            sizes.append(obstacle.surface.size(mesh.nodes))
        self.tolerance = 0.01 * min(sizes) if tolerance is None else float(tolerance)
        if not self.tolerance > 0:
            raise ValueError("ContactProblem: a positive tolerance")
        self._target = 0.0
        self._x_old = np.ascontiguousarray(mesh.nodes)
        self._records: dict = {}

    # -- set-up ----------------------------------------------------------------------------

    def _check(self, surface: Surface) -> None:
        if not isinstance(surface, Surface):
            raise TypeError("ContactProblem: surfaces (Surface.of(mesh, ...))")
        if surface.order != self.mesh.order:
            raise ValueError(
                "ContactProblem: faces of three nodes for 4-node tetrahedra, of six for "
                "10-node ones (Surface.of on the mesh itself)"
            )
        if not surface.n_faces:
            raise ValueError("ContactProblem: a surface without faces")

    def _pass(self, slave, master, own, obstacle, spec, stiffest: float, shares: int = 1):
        return contact_pass(self.mesh.nodes, slave, master, own, obstacle, spec, stiffest, shares)

    # -- forces ----------------------------------------------------------------------------

    def _predictor(self) -> bool:
        return True

    def _contact(self, u, tangent: bool, predict: bool):
        """The forces of the contacts (all the dofs) for the displacement ``u`` and, with
        ``tangent``, their tangent; the state of the slave points of this evaluation is
        kept in their passes."""
        x = np.ascontiguousarray(self.mesh.nodes + u.reshape(-1, 3))
        return contact_forces(self._passes, x, self._x_old, self.neighbours, self._target,
                              tangent, predict)  # fmt: skip

    def _forces(self, u, hist, tangent: bool, rate: float = 0.0, predict: bool = False):
        f, k, hist_new = super()._forces(u, hist, tangent, rate, predict)
        if not np.all(np.isfinite(f)):  # a point of the bulk failed: the step is divided
            return f, k, hist_new
        fc, kc = self._contact(u, tangent, predict)
        self._parts = (*self._parts, fc)
        if kc is not None:
            k = kc if k is None else plus(k, kc)
        return f + fc, k, hist_new

    # -- the steps -------------------------------------------------------------------------

    def _part_names(self) -> tuple:
        return (*super()._part_names(), "contact")

    def _begin(self, u, load: float) -> None:
        """The state of the slave points before the loading: the faces and the gaps that
        they see, none pressing; the records of the steps."""
        self._target = float(load)
        self._x_old = np.ascontiguousarray(self.mesh.nodes + u.reshape(-1, 3))
        for p in self._passes:
            p.state = p.fresh()
            p.multiplier = np.zeros(len(p.face))
        self._contact(u, False, False)
        for p in self._passes:
            p.state = p.trial.copy()
            p.state[:, S_PRESSURE] = 0.0
            p.state[:, S_TX : S_TX + 3] = 0.0
            p.state[:, S_WORK] = 0.0
        count = len(self._owners)
        self._records = {
            "area": [np.zeros(count)],
            "pressure": [np.zeros(count)],
            "forces": [np.zeros((count, 3))],
            "penetration": [0.0],
            "friction": [0.0],
            "obstacles": [0.0],
            "resultants": [np.zeros(3) for _ in self.obstacles],
            "load": float(load),
        }

    def _attempt(self, target: float) -> None:
        """A step is tried towards the load factor ``target``: the multipliers of the end
        of the last one."""
        self._target = float(target)
        self._augmented = 0
        for p in self._passes:
            p.state[:, S_LAMBDA] = p.multiplier
            p.last = p.state[:, S_FACE].astype(np.int64)

    def _augment(self) -> bool:
        """After Newton's method: whether the multipliers are updated and the step solved
        again (points that penetrate deeper than the tolerance)."""
        if self._augmented >= self.augmentations:
            return False
        deepest = max(
            (float(-p.trial[p.pressed, S_GAP].min()) for p in self._passes if p.pressed.any()),
            default=0.0,
        )
        if deepest <= self.tolerance:
            return False
        for p in self._passes:
            p.state[:, S_LAMBDA] = p.trial[:, S_PRESSURE]
        self._augmented += 1
        return True

    def _accept(self, u) -> None:
        """A step converged: the state of the slave points is kept, its records made."""
        rec = self._records
        self._x_old = np.ascontiguousarray(self.mesh.nodes + u.reshape(-1, 3))
        count = len(self._owners)
        area, pressure, forces = np.zeros(count), np.zeros(count), np.zeros((count, 3))
        deepest, rubbed = 0.0, 0.0
        for j, owners in enumerate(self._owners):
            for index, sign in owners:
                p = self._passes[index]
                area[j] += (p.weight * p.trial[:, S_SHARE])[p.pressed].sum() / p.shares
                if p.pressed.any():
                    most = p.shares * float(p.trial[p.pressed, S_PRESSURE].max())
                    pressure[j] = max(pressure[j], most)
                    deepest = max(deepest, float(-p.trial[p.pressed, S_GAP].min()))
                forces[j] += sign * p.resultant
        for p in self._passes:
            p.state = p.trial.copy()
            p.multiplier = p.state[:, S_LAMBDA].copy()
            rubbed += float(p.weight @ p.state[:, S_WORK])
        work = rec["obstacles"][-1]
        first = len(self.pairs)
        for k, obstacle in enumerate(self.obstacles):
            moved = obstacle.translation(self._target) - obstacle.translation(rec["load"])
            # the obstacle pushes the body by the opposite of the force that it receives
            work -= 0.5 * (rec["resultants"][k] + forces[first + k]) @ moved
            rec["resultants"][k] = forces[first + k].copy()
        rec["load"] = self._target
        rec["area"].append(area)
        rec["pressure"].append(pressure)
        rec["forces"].append(forces)
        rec["penetration"].append(max(deepest, 0.0))
        rec["friction"].append(rubbed)
        rec["obstacles"].append(work)

    def run(self, prescribed, load: float = 1.0, **options) -> ContactResult:
        """Grow the prescribed displacements, the forces and the motions of the obstacles
        to the load factor ``load``: the entries and options of
        :meth:`~morphogen.fracture.cohesive.CohesiveProblem.run` (``steps``, ``tol``,
        ``max_newton``, ``max_cuts``, ``fields``, ``forces``, ``duration``, ``viscosity``
        and ``history`` of the cohesive interfaces), but its ``initial`` state."""
        if options.get("initial") is not None:
            raise ValueError("ContactProblem.run: no initial state (the contacts have none)")
        res = super().run(prescribed, load, **options)
        rec = self._records
        work = dict(res.work)
        work.pop("friction", None)
        if self.interfaces.n_elements == 0:
            work.pop("interfaces", None)
        work["friction"] = np.array(rec["friction"])
        if self.interfaces.n_elements and "friction" in res.work:
            work["friction"] = work["friction"] + res.work["friction"]
        work["obstacles"] = np.array(rec["obstacles"])
        return ContactResult(
            res.loads,
            res.reactions,
            res.iterations,
            res.displacement,
            np.array(rec["area"]),
            np.array(rec["pressure"]),
            np.array(rec["forces"]),
            np.array(rec["penetration"]),
            list(self.names),
            self._slave_points(res.displacement),
            res.fields,
            res.time_s,
            work,
            res if self.interfaces.n_elements else None,
        )

    def _slave_points(self, displacement) -> list:
        """The slave points of each contact at the end (see :class:`ContactResult`); those
        of the two passes of a symmetric pair one after the other."""
        x = self.mesh.nodes + displacement
        out = []
        for owners in self._owners:
            parts = [self._passes[index] for index, _ in owners]
            out.append(
                {
                    "position": np.concatenate(
                        [p.surface.positions(x, p.face, p.xi, p.eta) for p in parts]
                    ),
                    "pressure": np.concatenate([p.shares * p.state[:, S_PRESSURE] for p in parts]),
                    "traction": np.concatenate(
                        [p.shares * p.state[:, S_TX : S_TX + 3] for p in parts]
                    ),
                    "gap": np.concatenate([p.state[:, S_GAP] for p in parts]),
                    "weight": np.concatenate([p.weight * p.state[:, S_SHARE] for p in parts]),
                    "dissipated": np.concatenate([p.state[:, S_WORK] for p in parts]),
                }
            )
        return out

    def nodal_pressure(self) -> np.ndarray:
        """The pressure of the contacts at the nodes at the end of the last step: at each
        node of a slave surface, the mean over the faces around it, weighted by their
        areas, of the mean pressure of their points; 0 elsewhere."""
        total = np.zeros(self.mesh.n_nodes)
        area = np.zeros(self.mesh.n_nodes)
        for p in self._passes:
            count = len(p.face) // len(p.slave)
            weight = p.weight.reshape(-1, count)
            carried = (p.state[:, S_SHARE] * p.state[:, S_PRESSURE]).reshape(-1, count)
            mean = p.shares * (weight * carried).sum(axis=1)
            for column in p.slave.T:
                np.add.at(total, column, mean)
                np.add.at(area, column, weight.sum(axis=1))
        return np.divide(total, area, out=np.zeros_like(total), where=area > 0)

    def write(self, path, result, step: int = -1):
        """The body (``<path>.vtu``: the displacement of ``step`` with ``fields``, else the
        final one, and at the end the pressure of the contacts at the nodes); returns the
        path."""
        from pathlib import Path

        from morphogen.io.meshes import write_vtu_tet

        u = result.fields[step] if result.fields else result.displacement
        data = {"displacement": u}
        if not result.fields or step in (-1, len(result.fields) - 1):
            data["contact_pressure"] = self.nodal_pressure()
        return write_vtu_tet(Path(path).with_suffix(".vtu"), self.mesh, point_data=data)


def _stiffness(material) -> float:
    """Young's modulus of a material, or the largest normal stiffness of its matrix."""
    if material.young:
        return float(material.young)
    return float(np.max(np.diag(np.asarray(material.stiffness, dtype=float))[:3]))
