"""Remeshed crack propagation in inelastic solids: the transfer of the state of a body
between the meshes of successive cracks, and the ductile growth of kinked cracks.

*Transfer* (:func:`transfer_state`). A solved body (a
:class:`~morphogen.fracture.mixed.CrackSolution` whose bulk is the behaviours of
morphoGen's points, :class:`~morphogen.fracture.bulk.PointBulk`: von Mises
elasto-viscoplasticity, porous or crystal plasticity, MFront laws) hands its state to the
mesh of the next crack: the internal variables of its points (one per linear tetrahedron),
their work density and committed stress, and the displacement of its nodes, the initial
guess of the next solution. Each point of the new mesh (an element centroid, or a node for
the displacement) is located in an element of the old mesh (barycentric coordinates, among
the elements whose centroids are nearest), the two lips of the old crack kept apart:
behind the old front, a point only takes the values of the elements of its own side (the
side of its element in the new mesh, or of the elements around a node). Then:

* ``"nearest"``: the values of that element (one point per element: the identity onto the
  same mesh, exactly; a transfer that keeps the extreme values but carries the old elements'
  facets into the new mesh);
* ``"projection"``: the values projected on the nodes of the old mesh (the means of the
  elements around each node weighted by their volumes; the lips, whose nodes are
  duplicated, apart) interpolated at the point: smoother, at the price of some diffusion of
  the peaks of the fields.

In a polycrystal (meshes whose elements carry the grains of a tessellation, ``mesh.cell``,
and their orientations: :func:`~morphogen.fracture.lefm.crack_mesh` with ``grains``), a point
takes the values of its own grain only, located among its elements and projected on the
nodes from them (the nodes of a grain boundary have a value per grain): the slips and the
``Fp⁻¹`` of the slip systems of a grain, in the axes of its lattice, are never mixed with
those of a neighbour. The displacement is interpolated linearly (exact for the old field),
across the grain boundaries. On the crack meshes of a morphology (whose phases are those of
its materials, the sides of the crack apart: :func:`~morphogen.fracture.lefm.crack_mesh`
with ``morphology``), the state moves material by material in the same way (the phases of
one material alike; grains and materials both, when the meshes have both). The ``Fp⁻¹`` of
crystals, interpolated, is made isochoric again (``det = 1``); the plastic strains of the
logarithmic space of finite strains, deviatoric, remain so. The strain measure committed
with the transferred stress is that of the transferred displacement on the new mesh, and
the work density is the dissipated part moved plus the free energy of the new state; the
transferred state, not in equilibrium on the new mesh, is brought to it at the load of the
transfer before the loading resumes (``CohesiveProblem.run(initial=..., start=...)``:
Newton's method over a vanishing time, the plastic correction of the points included, in
one step from the transferred state), which dissipates nothing: the dissipated energy moved
is kept and the free energy is that of the equilibrium.

*Ductile growth* (:func:`propagate_ductile`): at each load step (a fraction of the load
factor) the body is loaded further from its committed state; once the crack driving force
at the tip, ``J_tip`` (:meth:`~morphogen.fracture.mixed.CrackSolution.release_rate` with
``energy="tip"``: the free energy and the configurational forces of the internal variables)
reaches the ``toughness``, or the porosity of the porous points near the tip reaches a
critical value (the local approach of ductile fracture: Rousselier 1987; Pineau 2006), the
front advances by an increment in the direction of kinking (the interaction integrals of
the elastic estimates), the body is meshed again for the new crack (a front grown into a
pore going through it), its state transferred and brought to equilibrium, and the
criterion checked again before the load grows. With a toughness of the phases and of their
interfaces (:mod:`morphogen.fracture.toughness`), ``J_tip`` is held against the toughness of
the phase at the front and the advance is the candidate of the largest ``G(θ) / Γ(θ)``, the
kink of the bulk or a deflection along an interface ahead, once that ratio reaches 1. The far
driving force at each advance against the advance draws the resistance curve: ``J_ψ`` on a
domain around the front (``energy="free"``: the configurational force of the free energy
on the domain, ``J_tip`` plus the shielding of the internal variables inside it, the
``J_far`` of Simha, Fischer, Shan, Chen & Kolednik 2008), which rises as the plastic wake
of the growing crack shields its tip. The J of the deformation theory (``energy="work"``),
the measure of the loading of a stationary crack, is kept too; once the wake unloads it
depends on the domain.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from morphogen.fracture.cohesive import CohesiveProblem, Contact, Interfaces
from morphogen.fracture.lefm import crack_faces
from morphogen.fracture.mixed import (
    CrackSolution,
    _as_materials,
    element_sides,
    kink_angle,
    material_groups,
    through_voids,
)
from morphogen.mesh.tetra import TetMesh

#: Methods of transfer of the state of the points
METHODS = ("nearest", "projection")


def _labels_of_elements(mesh: TetMesh, crack, old_crack) -> np.ndarray:
    """The side of each element behind the front of ``old_crack`` (``crack`` gives the sides
    of the elements of ``mesh``), 0 ahead of it."""
    centroids = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    return element_sides(mesh, crack) * (old_crack.psi(centroids) < 0)


def _labels_of_nodes(mesh: TetMesh, element_labels: np.ndarray) -> np.ndarray:
    """The side of each node: that of the elements around it when they all have it, 0
    otherwise."""
    n = mesh.n_nodes
    corners = mesh.elements[:, :4]
    plus = np.bincount(corners.ravel(), np.repeat(element_labels > 0, 4), minlength=n)
    minus = np.bincount(corners.ravel(), np.repeat(element_labels < 0, 4), minlength=n)
    return np.where((plus > 0) & (minus == 0), 1.0, np.where((minus > 0) & (plus == 0), -1.0, 0.0))


def locate(points, mesh: TetMesh, point_labels=None, element_labels=None, k: int = 16,
           point_groups=None, element_groups=None):  # fmt: skip
    """The element of ``mesh`` (linear tetrahedra) containing each point (rows) and its
    barycentric coordinates there (points x 4), among the ``k`` elements whose centroids are
    nearest and whose labels agree with the point's (a label 0 agrees with every one) and,
    with ``point_groups`` and ``element_groups`` (grains), of its group (always, unless no
    element of ``mesh`` is); points outside every candidate take the nearest one, their
    coordinates clipped."""
    from scipy.spatial import cKDTree

    points = np.atleast_2d(np.asarray(points, dtype=float))
    x = mesh.nodes[mesh.elements[:, :4]]
    origin = x[:, 0]
    inverse = np.linalg.inv(
        np.stack([x[:, 1] - origin, x[:, 2] - origin, x[:, 3] - origin], axis=2)
    )
    tree = cKDTree(x.mean(axis=1))
    n = len(points)
    if point_labels is None:
        point_labels = np.zeros(n)
    if element_labels is None:
        element_labels = np.zeros(mesh.n_elements)
    groups = point_groups is not None and element_groups is not None
    if groups:
        point_groups = np.asarray(point_groups)
        element_groups = np.asarray(element_groups)
    chosen = np.zeros(n, dtype=np.int64)
    bary = np.zeros((n, 4))
    todo = np.arange(n)
    for count in (k, 4 * k, None):
        if count is None:  # no candidate on the right side: the labels dropped
            point_labels = np.zeros(n)
            count = 4 * k
        count = min(count, mesh.n_elements)
        _, candidates = tree.query(points[todo], k=count)
        candidates = candidates.reshape(len(todo), count)
        local = np.einsum("nkij,nkj->nki", inverse[candidates],
                          points[todo, None, :] - origin[candidates])  # fmt: skip
        coordinates = np.concatenate([1.0 - local.sum(axis=2, keepdims=True), local], axis=2)
        mine = element_labels[candidates]
        agree = (mine == 0) | (point_labels[todo, None] == 0) | (mine == point_labels[todo, None])
        if groups:
            agree &= element_groups[candidates] == point_groups[todo, None]
        score = np.where(agree, coordinates.min(axis=2), -np.inf)
        best = np.argmax(score, axis=1)
        rows = np.arange(len(todo))
        found = np.isfinite(score[rows, best])
        chosen[todo[found]] = candidates[rows, best][found]
        bary[todo[found]] = coordinates[rows, best][found]
        todo = todo[~found]
        if not len(todo):
            break
    if groups and len(todo):  # far from the elements of their group: its nearest one
        centroids = x.mean(axis=1)
        for g in np.unique(point_groups[todo]):
            members = np.flatnonzero(element_groups == g)
            if not len(members):  # a group that the mesh does not resolve: any element
                members = np.arange(mesh.n_elements)
            rows = todo[point_groups[todo] == g]
            _, near = cKDTree(centroids[members]).query(points[rows])
            chosen[rows] = members[near]
            local = np.einsum(
                "nij,nj->ni", inverse[chosen[rows]], points[rows] - origin[chosen[rows]]
            )
            bary[rows] = np.concatenate([1.0 - local.sum(axis=1, keepdims=True), local], axis=1)
    bary = np.clip(bary, 0.0, None)
    return chosen, bary / bary.sum(axis=1, keepdims=True)


def _nodal(mesh: TetMesh, values: np.ndarray, mask=None) -> np.ndarray:
    """Element values (elements x m) projected on the nodes: the means of the elements
    around each node weighted by their volumes (the elements of ``mask`` only)."""
    corners = mesh.elements[:, :4]
    if mask is not None:
        corners, values = corners[mask], values[mask]
    x = mesh.nodes[corners]
    volumes = np.abs(np.einsum("ij,ij->i", x[:, 1] - x[:, 0],
                               np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))) / 6.0  # fmt: skip
    weight = np.bincount(corners.ravel(), np.repeat(volumes, 4), minlength=mesh.n_nodes)
    out = np.zeros((mesh.n_nodes, values.shape[1]))
    for a in range(4):
        np.add.at(out, corners[:, a], volumes[:, None] * values)
    return out / np.where(weight > 0, weight, 1.0)[:, None]


def transfer_displacement(source: CrackSolution, mesh: TetMesh, crack) -> np.ndarray:
    """The displacement (nodes x 3) of a solved body interpolated at the nodes of the mesh
    of another crack (see the module)."""
    old = source.mesh
    old_labels = _labels_of_elements(old, source.crack, source.crack)
    new_labels = _labels_of_elements(mesh, crack, source.crack)
    element, bary = locate(mesh.nodes, old, _labels_of_nodes(mesh, new_labels), old_labels)
    corners = old.elements[element, :4]
    return np.einsum("na,nac->nc", bary, source.displacement[corners])


def transfer_state(source: CrackSolution, mesh: TetMesh, bulk, crack, method: str = "projection"):
    """The state of a solved inelastic body (its :class:`~morphogen.fracture.bulk.PointBulk`)
    transferred to the ``bulk`` of the opened ``mesh`` of ``crack`` (see the module): the
    internal variables, the work densities and the committed stresses of its points by
    ``method`` (``"nearest"`` or ``"projection"``), the strain measures of the transferred
    displacement, which it returns (nodes x 3)."""
    if method not in METHODS:
        raise ValueError(f"transfer: {METHODS}")
    if source.bulk is None:
        raise ValueError("transfer_state: the source has no bulk of points (an elastic body)")
    old, before = source.mesh, source.bulk
    if before.old.shape[1] != bulk.old.shape[1] or before.finite != bulk.finite:
        raise ValueError("transfer_state: other behaviours or kinematics")
    old_labels = _labels_of_elements(old, source.crack, source.crack)
    new_labels = _labels_of_elements(mesh, crack, source.crack)
    centroids = mesh.nodes[mesh.elements[:, :4]].mean(axis=1)
    grains = old.cell is not None and mesh.cell is not None  # a polycrystal: grain by grain
    if grains and not _same_orientations(old, mesh):
        raise ValueError("transfer_state: the grains of the two meshes differ")
    old_groups, new_groups = _groups(old, mesh, bulk.materials)  # by grain, by material
    element, bary = locate(centroids, old, new_labels, old_labels, point_groups=new_groups,
                           element_groups=old_groups)  # fmt: skip
    dissipated = before.work - before.free_energy()  # NaN for MFront laws: their work moves
    fields = np.hstack([before.old, before.work[:, None], np.nan_to_num(dissipated)[:, None],
                        before.committed_stress])  # fmt: skip
    if method == "nearest":
        moved = fields[element]
    elif old_groups is not None:  # projected on the nodes from the elements of each group
        moved = np.empty((len(element), fields.shape[1]))
        for g in np.unique(old_groups[element]):
            nodal = _nodal(old, fields, old_groups == g)
            rows = np.flatnonzero(old_groups[element] == g)
            corners = old.elements[element[rows], :4]
            moved[rows] = np.einsum("na,nam->nm", bary[rows], nodal[corners])
    else:
        nodal = _nodal(old, fields)
        moved = np.einsum("na,nam->nm", bary, nodal[old.elements[element, :4]])
    width = before.old.shape[1]
    state = moved[:, :width].copy()
    if method == "projection" and bulk.finite:  # Fp⁻¹ of crystals isochoric again
        from morphogen.homogenization.behaviours import CRYSTAL
        from morphogen.homogenization.crystal import CP_ALPHA

        crystal = np.flatnonzero(bulk.kind == CRYSTAL)
        if len(crystal):
            slips = bulk.behaviours.data[7][bulk.pmat[crystal]]
            for row, ns in zip(crystal, slips, strict=True):
                off = CP_ALPHA + 2 * int(ns)
                g = state[row, off : off + 9].reshape(3, 3) + np.eye(3)
                g /= np.cbrt(np.linalg.det(g))
                state[row, off : off + 9] = (g - np.eye(3)).ravel()
    bulk.old[...] = state
    bulk.new[...] = state
    bulk.committed_stress[...] = moved[:, width + 2 :]
    displacement = transfer_displacement(source, mesh, crack)
    from morphogen.homogenization import finite as fin
    from morphogen.homogenization.nonlinear import _point_strains

    kernel = fin.point_gradients if bulk.finite else _point_strains
    kernel(mesh.nodes, mesh.elements, 1, np.ascontiguousarray(displacement.ravel()), bulk.gp,
           bulk.gw, bulk.grads, bulk.wts)  # fmt: skip
    bulk.committed_grads[...] = bulk.grads
    # the work density: the dissipated part moves, the free energy is that of the new state
    free = bulk.free_energy()
    bulk.work[...] = np.where(np.isfinite(free), moved[:, width + 1] + free, moved[:, width])
    return displacement


def _groups(old: TetMesh, new: TetMesh, materials: dict):
    """The groups of the elements of two meshes between which the state of the points never
    moves (None, None: no groups): the grains of polycrystals (``cell``), the materials of
    the phases of the crack meshes of a morphology (whose sides are ``side``; phases of the
    same material alike), or both."""
    grains = old.cell is not None and new.cell is not None
    phases = old.side is not None and new.side is not None
    if not (grains or phases):
        return None, None
    count = 1 + int(max(old.phase.max(), new.phase.max())) if phases else 1
    out = []
    for m in (old, new):
        g = np.zeros(m.n_elements, dtype=np.int64)
        if phases:
            g += material_groups(m.phase, materials)
        if grains:
            g += count * m.cell
        out.append(g)
    return out[0], out[1]


def _same_orientations(a: TetMesh, b: TetMesh) -> bool:
    """Whether two meshes have the same crystal orientations of their cells."""
    if a.orientations is None or b.orientations is None:
        return a.orientations is None and b.orientations is None
    return a.orientations.shape == b.orientations.shape and np.allclose(a.orientations,
                                                                        b.orientations)  # fmt: skip


def dissipation(solution: CrackSolution) -> float:
    """The energy dissipated in the bulk of a solved body: its work less its free energy
    (stored elastically and by hardening)."""
    return solution.energy("work") - solution.energy("free")


# -- ductile growth ------------------------------------------------------------------------


@dataclass
class DuctileGrowth:
    """A ductile growth (see the module): the cracks at each advance, the load factors of
    the advances, the driving forces at the tip (``tip``) and far from it (``far``, ``J_ψ``
    on the domain; ``work``, the J of the deformation theory), the stress intensity factors
    of the elastic estimates (advances x 3), the angles of the advances, every load step
    (``history``: rows of the advance, the load, the driving forces and the porosity near
    the tip), with ``keep`` the solutions of the advances, and with a toughness of the
    phases or of their interfaces the advances chosen (``advances``: the
    :class:`~morphogen.fracture.toughness.Advance` of each crack; that of the last one is
    not taken)."""

    cracks: list
    loads: np.ndarray
    tip: np.ndarray
    far: np.ndarray
    work: np.ndarray
    factors: np.ndarray
    angles: np.ndarray
    history: list
    solutions: list = field(default_factory=list, repr=False)
    advances: list = field(default_factory=list, repr=False)

    def rows(self) -> list[dict]:
        """One row per advance: the length, the advance, the load factor, the driving
        forces, the factors and the angle (and with a toughness of the phases or of their
        interfaces, the bulk or an interface, what it meets, its toughness and its ratio
        ``G(θ) / Γ(θ)``)."""
        from morphogen.fracture.mixed import _advance_columns

        base = self.cracks[0].size if self.cracks else 0.0
        rows = [
            {"advance": k, "length": c.size, "delta_a": c.size - base, "load": float(lam),
             "J_tip": float(jt), "J_far": float(jf), "J_work": float(jw), "K_I": float(f[0]),
             "K_II": float(f[1]), "K_III": float(f[2]), "angle_deg": math.degrees(a)}
            for k, (c, lam, jt, jf, jw, f, a) in enumerate(zip(
                self.cracks, self.loads, self.tip, self.far, self.work, self.factors,
                self.angles, strict=True))
        ]  # fmt: skip
        if self.advances:
            for row, advance in zip(rows, self.advances, strict=True):
                row.update(_advance_columns(advance))
        return rows

    def resistance(self):
        """The resistance curve: the advances of the crack and the far driving forces."""
        base = self.cracks[0].size if self.cracks else 0.0
        return np.array([c.size - base for c in self.cracks]), np.asarray(self.far)


def _problem(mesh, crack, materials, kinematics, contact, bulk):
    """The opened mesh of a crack, its lips and its cohesive problem (as
    :func:`~morphogen.fracture.mixed.solve_crack`)."""
    opened, faces = crack_faces(mesh, crack)
    materials = _as_materials(materials, opened)
    if contact:
        interfaces, law = faces, Contact(float(contact))
    else:
        empty = np.zeros((0, 3), dtype=np.int64)
        interfaces, law = Interfaces(empty, empty, np.zeros((0, 2), dtype=np.int64)), Contact(1.0)
    problem = CohesiveProblem(opened, interfaces, materials, law, kinematics=kinematics, bulk=bulk)
    return opened, faces, materials, problem


def _snapshot(solution: CrackSolution) -> CrackSolution:
    """A solution whose bulk keeps its state when the problem goes on."""
    out = copy.copy(solution)
    if solution.bulk is not None:
        bulk = copy.copy(solution.bulk)
        for name in ("old", "new", "work", "committed_stress", "committed_grads"):
            setattr(bulk, name, getattr(solution.bulk, name).copy())
        out.bulk = bulk
    out._cache = {}
    return out


def _tip_porosity(solution: CrackSolution, radius: float) -> float:
    """The largest porosity of the porous points within ``radius`` of the front (0 without
    porous points)."""
    bulk = solution.bulk
    if bulk is None:
        return 0.0
    from morphogen.homogenization.behaviours import POROUS
    from morphogen.homogenization.porous import porosity

    mesh = solution.mesh
    r = solution.crack.frame(mesh.nodes[mesh.elements].mean(axis=1))[0]
    near = (bulk.kind == POROUS) & (r < radius)
    if not near.any():
        return 0.0
    return float(porosity(bulk.old[near], bulk.behaviours.data[2], bulk.pmat[near]).max())


def propagate_ductile(
    mesh_of,
    crack,
    materials,
    prescribed,
    *,
    load_step: float,
    increment: float,
    steps: int,
    toughness=None,
    porosity: float | None = None,
    max_load: float = math.inf,
    kinematics: str = "small",
    transfer: str = "projection",
    solver_steps: int = 2,
    radius: float | None = None,
    domain: tuple | None = None,
    forces=None,
    contact: float | None = None,
    bulk: str = "auto",
    kink: bool = True,
    keep: bool = False,
    **options,
) -> DuctileGrowth:
    """Grow a crack in an inelastic body (see the module): ``mesh_of(crack)`` meshes the body
    conforming to it (:func:`~morphogen.fracture.lefm.crack_mesh`), ``prescribed`` (entries,
    or a function of the opened mesh) and ``forces`` (nodal forces, or a function of the
    opened mesh) are grown by ``load_step`` of the load factor in ``solver_steps`` steps
    between the checks of the criterion: ``J_tip >= toughness``, or the porosity of the porous
    points within ``radius`` of the front (by default 1.5 element) ``>= porosity``; each of
    the ``steps`` advances the front by ``increment`` in the direction of kinking (straight
    with ``kink=False``), the state transferred by ``transfer`` (``"nearest"`` or
    ``"projection"``); the far driving forces on the ``domain`` ``(r_in, r_out)`` (by
    default that of ``release_rate``: 1 to 4 elements); the loading stops at ``max_load``;
    ``options`` those of
    :meth:`~morphogen.fracture.cohesive.CohesiveProblem.run` (``tol``, ``max_newton``...),
    but ``duration`` is the time of a unit of the load factor (by default 1: the load factor
    is the time of the viscous behaviours), the same over all the loadings.

    ``toughness``: one value, or a :class:`~morphogen.fracture.toughness.Toughness` of the
    phases and of their interfaces (or a dict of the phases): the criterion is then, at each
    load step, the largest ratio ``G(θ) / Γ(θ)`` of the candidate advances reaching 1
    (:func:`~morphogen.fracture.toughness.advance_candidates`, the bulk taking ``J_tip``:
    ``J_tip`` against the toughness of the phase at the front, unless the advance enters a
    tougher phase within the increment or an interface ahead is more favourable), and the
    advance is that candidate (``kink=False``: the bulk straight ahead)."""
    from morphogen.fracture.toughness import (
        Toughness,
        advance_candidates,
        choose_advance,
        interface_paths,
    )

    if toughness is None and porosity is None:
        raise ValueError("propagate_ductile: a toughness or a critical porosity")
    if not (load_step > 0 and increment > 0):
        raise ValueError("propagate_ductile: a positive load step and increment")
    if transfer not in METHODS:
        raise ValueError(f"transfer: {METHODS}")
    rule = None if toughness is None or np.isscalar(toughness) else Toughness.of(toughness)
    staged = rule is not None and not rule.uniform  # the candidates at each load step
    if rule is not None and rule.uniform:
        toughness = rule.value
    unit = options.pop("duration", None)

    def timed(target):  # run() spreads its duration over the loading from 0 to its target
        return {} if unit is None else {"duration": float(unit) * target}

    cracks, loads, tips, fars, works, factors, angles = [], [], [], [], [], [], []
    history, solutions, advances = [], [], []
    r_in, r_out = (None, None) if domain is None else domain
    lam, previous = 0.0, None
    finite = kinematics == "finite"
    for advance in range(int(steps) + 1):
        opened, faces, mats, problem = _problem(mesh_of(crack), crack, materials, kinematics,
                                                contact, bulk)  # fmt: skip
        entries = prescribed(opened) if callable(prescribed) else prescribed
        dead = forces(opened) if callable(forces) else forces
        common = {**options, "forces": dead}
        solution = paths = None
        if previous is not None:  # the state of the last crack, brought to equilibrium
            if problem.bulk is not None:
                u = transfer_state(previous, opened, problem.bulk, crack, transfer)
            else:
                u = transfer_displacement(previous, opened, crack)
            result = problem.run(entries, lam, steps=1, start=lam, initial=u, **common,
                                 **timed(lam))  # fmt: skip
        while True:
            target = min(lam + load_step, max_load)
            if previous is None and solution is None:  # the first loading, from rest
                result = problem.run(entries, target, steps=solver_steps, **common,
                                     **timed(target))  # fmt: skip
            elif solution is not None:  # a further load step
                result = problem.run(entries, target, steps=solver_steps, start=lam,
                                     initial=solution.displacement, equilibrate=False,
                                     **common, **timed(target))  # fmt: skip
            lam = float(result.loads[-1])
            final = None if dead is None else lam * np.asarray(dead, dtype=float).ravel()
            solution = CrackSolution(opened, faces, crack, mats, finite, result.displacement,
                                     result, final, contact, bulk=problem.bulk)  # fmt: skip
            size = solution.element_size()
            j_tip = solution.release_rate(energy="tip")
            j_far = solution.release_rate(r_in, r_out, energy="free")
            j_work = solution.release_rate(r_in, r_out, energy="work")
            f_tip = _tip_porosity(solution, 1.5 * size if radius is None else float(radius))
            row = {"advance": advance, "load": lam, "J_tip": j_tip, "J_far": j_far,
                   "J_work": j_work, "porosity_tip": f_tip}  # fmt: skip
            if staged:  # the candidate advances, the bulk taking J_tip
                modes = _modes(solution)
                if paths is None:  # the interfaces ahead: one mesh, one crack
                    paths = interface_paths(solution, rule, increment)
                choice = choose_advance(advance_candidates(solution, rule, increment,
                                                           modes=modes, driving=j_tip,
                                                           kink=kink, paths=paths))  # fmt: skip
                row["ratio"] = choice.ratio
                broken = choice.ratio >= 1.0
            else:
                broken = toughness is not None and j_tip >= toughness
            history.append(row)
            met = broken or (porosity is not None and f_tip >= porosity)
            if met or lam >= max_load:
                break
        if staged:
            k = (modes["K_I"], modes["K_II"], modes["K_III"])
            angle, grown = choice.angle, choice.crack
            advances.append(choice)
        else:
            k = tuple(_modes(solution)[key] for key in ("K_I", "K_II", "K_III"))
            angle = kink_angle(k[0], k[1]) if kink and np.all(np.isfinite(k[:2])) else 0.0
            grown = crack.grown(increment, angle)
        cracks.append(crack)
        loads.append(lam)
        tips.append(j_tip)
        fars.append(j_far)
        works.append(j_work)
        factors.append(k)
        angles.append(angle)
        if keep:
            solutions.append(_snapshot(solution))
        if advance == steps or not met:
            break
        previous = solution
        crack = through_voids(grown, solution.mesh, increment)
    return DuctileGrowth(cracks, np.array(loads), np.array(tips), np.array(fars),
                         np.array(works), np.array(factors), np.array(angles), history,
                         solutions, advances)  # fmt: skip


def _modes(solution: CrackSolution) -> dict:
    """The modes of a solved body (the elastic estimates of its interaction integrals), NaN
    without isotropic elasticity at the front."""
    try:
        return solution.modes()
    except (TypeError, ValueError):  # no isotropic elasticity at the front
        keys = ("K_I", "K_II", "K_III", "G", "G_I", "G_II", "G_III", "angle")
        return dict.fromkeys(keys, math.nan)
