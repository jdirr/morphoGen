"""Debonding in the cells of the nonlinear homogenization: the interfaces of the phases of
a periodic cell (or the boundaries of its grains) that open, break, close again and rub
along a loading path.

:class:`CohesiveCell` is the periodic element of :class:`~morphogen.homogenization.nonlinear.
NonlinearFEProblem` (the fluctuation and the macroscopic strain components as unknowns,
loading paths in strains, stresses or both, the behaviours of morphoGen's points) with the
interface elements of :mod:`morphogen.fracture.cohesive` between its phases: the bilinear
mixed-mode law, the exponential one, or the contact of lips without adhesion, each with
Coulomb's friction of the broken share of a point.

*Insertion on the torus* (:func:`open_cell`). A periodic cell is a torus: a node of a face
of the cell and its translate on the opposite face are one point of the medium, and so
are a face of an element there and its translate. The faces to open are chosen among the
faces that two elements share on the torus — two elements of different phases on either
side of a face of the cell included — and the nodes are duplicated there: the corners of
the elements around each point of the torus are grouped by the faces (through that point)
by which they remain connected, and each group gets its own node, or its own nodes where
the point has translates in the cell; the nodes of a group are linked by the periodic
conditions, those of different groups are not. An interface that crosses a face of the
cell thus goes on in the next cell, its two lips each with their images.

*Images.* An interface element that lies on a face of the cell joins a face of an element
there and the image of a face of the opposite side: the image of a node of displacement
``u`` moved by a period ``L`` is displaced by ``u + H L``, ``H`` the macroscopic
displacement gradient, so that its separation is the jump of the fluctuation, and the
forces that the image receives go to the node itself.

*Macroscopic stress.* The tractions of an interface are opposite on its two lips, at one
place of the reference cell: they carry no part of the macroscopic stress, the integral of
the stress of the points over the volume of the cell — but for the interfaces on its
faces, whose forces work in a change of ``H`` by the force that crosses the face times
the period (``Σ f ⊗ X`` over the nodes, the images at their own places). The macroscopic
strain is that of the points and of the separations: ``E = <ε> + 1/V ∫ sym([u] ⊗ n) dS``.

*Softening.* An interface that breaks at once has no equilibrium beyond a limit point
under prescribed strains (a snap-back): the steps are divided, and the viscous
regularization of Gao & Bower (2004), with the time of the path, lets them through
(``viscosity``; :mod:`morphogen.fracture.cohesive`).

*Iterations.* Lips open, close and rub along Newton's corrections. The first tangent of
a step keeps closed the lips of broken points that touched at its start; the line search
is nonmonotone (the residual may rise, below the largest of the last :data:`WINDOW`
iterates: an active-set iteration that does not cycle); and Coulomb's bound is reached
over an elastic slip (``slip``) — lips that come to touch, whose pressure and slip both
start from nothing, would otherwise slide along a direction that means nothing.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from morphogen.fracture.cohesive import (
    _GAUSS,
    _GAUSS6,
    BILINEAR,
    BROKEN,
    CONTACT,
    STATE,
    Bilinear,
    Contact,
    Exponential,
    Interfaces,
    _interfaces,
    _interfaces6,
    _shape6,
    _with_mid_nodes,
)
from morphogen.homogenization.nonlinear import Extended, NonlinearFEProblem, _SurfacePaths
from morphogen.mesh.tetra import TET_EDGES, TET_FACES, TetMesh

#: The curves that a cell records at each step (``PathResult.series``): the area of the
#: interfaces that is broken (reference area), their mean damage, their largest normal
#: opening and the energy dissipated by friction
SERIES = ("debonded_area", "interface_damage", "interface_opening", "interface_dissipation")

#: The number of iterates whose largest residual bounds the line search of Newton's
#: corrections (a nonmonotone search)
WINDOW = 4

#: The parameters of the laws of the interfaces given as mappings (:func:`interface_law`)
LAW_OPTIONS = {
    "bilinear": (
        "stiffness",
        "normal_strength",
        "shear_strength",
        "mode_I",
        "mode_II",
        "eta",
        "friction",
    ),
    "exponential": ("strength", "energy", "beta", "penalty", "friction"),
    "contact": ("stiffness", "tangential", "friction"),
}


def interface_law(spec, young: float, size: float):
    """The traction-separation law of a mapping: its ``law`` (``"bilinear"``, by default,
    ``"exponential"`` or ``"contact"``: lips without adhesion) and the parameters of
    :class:`~morphogen.fracture.cohesive.Bilinear` (``normal_strength`` and ``mode_I`` at
    least), :class:`~morphogen.fracture.cohesive.Exponential` (``strength`` and
    ``energy``) or :class:`~morphogen.fracture.cohesive.Contact`. Left out: the shear
    strength is the normal one, the energy of mode II that of mode I times the square of
    the ratio of the strengths (the separations at failure in the ratio of the
    strengths), and the penalty ``stiffness`` is ``100 young / size``: a Young's modulus
    (the stiffest of the cell) over a length (the size of its interface elements)."""
    spec = dict(spec)
    kind = spec.pop("law", "bilinear")
    if kind not in LAW_OPTIONS:
        raise ValueError(f"interfaces: the law is one of {list(LAW_OPTIONS)}, got {kind!r}")
    unknown = sorted(set(spec) - set(LAW_OPTIONS[kind]))
    if unknown:
        raise ValueError(
            f"interfaces: unknown parameters {unknown} of the law {kind!r}; "
            f"use {list(LAW_OPTIONS[kind])}"
        )
    try:
        values = {key: None if value is None else float(value) for key, value in spec.items()}
    except (TypeError, ValueError):
        raise ValueError(f"interfaces: the parameters of the law {kind!r} are numbers") from None
    friction = values.get("friction") or 0.0
    penalty = values.get("stiffness") or 100.0 * float(young) / float(size)
    try:
        if kind == "exponential":
            return Exponential(
                strength=values["strength"],
                energy=values["energy"],
                beta=values.get("beta") or 1.0,
                penalty=values.get("penalty"),
                friction=friction,
            )
        if kind == "contact":
            return Contact(penalty, friction=friction, tangential=values.get("tangential"))
        normal = values["normal_strength"]
        shear = values.get("shear_strength") or normal
        mode_i = values["mode_I"]
        mode_ii = values.get("mode_II") or mode_i * (shear / normal) ** 2
        return Bilinear(penalty, normal, shear, mode_i, mode_ii, values.get("eta") or 2.0, friction)
    except KeyError as exc:
        raise ValueError(f"interfaces: the law {kind!r} needs {exc.args[0]!r}") from None


def _classes(mesh: TetMesh, periodic: bool) -> tuple[np.ndarray, np.ndarray]:
    """The point of the torus of each node (the first node of its class of translates)
    and its translation from it, in periods; without periodicity, each node its own."""
    cls = np.arange(mesh.n_nodes)
    shift = np.zeros((mesh.n_nodes, 3), dtype=np.int64)
    if periodic:
        slave, master, moved = mesh.periodic_links()
        cls[slave], shift[slave] = master, moved
    return cls, shift


def _links(ids, copy, orig, shift) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The periodic links of nodes ``ids`` that are copies ``copy`` of points of the torus
    (the nodes of one copy are translates), from the translations ``shift`` of their
    original nodes ``orig``: each node but the first of its copy, the first, and the
    periods between them."""
    by = np.lexsort((ids, copy))
    ids, copy, orig = ids[by], copy[by], orig[by]
    lead = np.ones(len(ids), dtype=bool)
    lead[1:] = copy[1:] != copy[:-1]
    start = np.maximum.accumulate(np.where(lead, np.arange(len(ids)), 0))
    rest = ~lead
    return ids[rest], ids[start[rest]], shift[orig[rest]] - shift[orig[start[rest]]]


def _selection(mesh: TetMesh, faces, e1, e2, centroids):
    """The faces to open among those that the elements ``e1`` and ``e2`` share, and
    whether their two sides are to be swapped (side -: the lower label)."""
    if callable(faces):
        selected = np.asarray(faces(centroids), dtype=bool)
        if selected.shape != e1.shape:
            raise ValueError(f"open_cell: one value per shared face ({len(e1)})")
        return selected, np.zeros(len(e1), dtype=bool)
    pairs = None
    if isinstance(faces, str):
        if faces not in ("phases", "grains"):
            raise ValueError(
                "open_cell: 'phases', 'grains', pairs of phases or a function of centroids"
            )
        if faces == "grains" and mesh.cell is None:
            raise ValueError("open_cell: 'grains' needs the grain of each element (mesh.cell)")
        label = mesh.phase if faces == "phases" else mesh.cell
    else:
        try:
            pairs = {tuple(sorted(int(p) for p in pair)) for pair in faces}
        except (TypeError, ValueError):
            pairs = None
        if not pairs or any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
            raise ValueError(
                "open_cell: 'phases', 'grains', pairs of two different phases or a "
                "function of centroids"
            )
        label = mesh.phase
    a, b = label[e1], label[e2]
    selected = a != b
    if pairs is not None:
        low, high = np.minimum(a, b), np.maximum(a, b)
        wanted = np.zeros(len(a), dtype=bool)
        for p, q in pairs:
            wanted |= (low == p) & (high == q)
        selected &= wanted
    return selected, a > b


def _open_corners(mesh: TetMesh, faces, periodic: bool):
    """:func:`open_cell` for a mesh of 4-node elements; also the copy of the torus of
    each node of the opened mesh (-1 for a node of no element)."""
    tets = np.asarray(mesh.elements[:, :4], dtype=np.int64)
    m, n = len(tets), mesh.n_nodes
    size = np.asarray(mesh.size, dtype=float)
    cls, shift = _classes(mesh, periodic)
    local = np.asarray(TET_FACES)
    corners = tets[:, local]  # elements x 4 faces x 3
    # the corners of each face in an order that its translates share: by their point of
    # the torus, then by their translation
    base = int(np.abs(shift).max(initial=0)) + 1
    code = cls[corners]
    for c in range(3):
        code = code * (2 * base + 1) + shift[corners][..., c] + base
    order = np.argsort(code, axis=-1, kind="stable")
    canon = np.take_along_axis(corners, order, axis=-1)
    moved = shift[canon]
    key = np.concatenate(
        [cls[canon], (moved[:, :, 1:] - moved[:, :, :1]).reshape(m, 4, 6)], axis=-1
    ).reshape(4 * m, 9)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    if counts.max(initial=0) > 2:
        raise ValueError("open_cell: a face shared by more than two elements")
    paired = counts[inverse] == 2
    if periodic:  # every face of the boundary of the cell has its translate
        if mesh.whole:
            outer = mesh.outer.ravel()
        else:
            tol = mesh.tolerance()
            low = np.all((np.abs(mesh.nodes - mesh.lower) <= tol)[corners], axis=2)
            high = np.all((np.abs(mesh.nodes - mesh.upper) <= tol)[corners], axis=2)
            outer = np.any(low | high, axis=-1).ravel()
        if np.any(outer & ~paired):
            raise ValueError(
                "open_cell: the opposite faces of the cell do not carry the same faces of "
                "elements (a periodic mesh: conforming_mesh(..., periodic=True), or the "
                "mesh of a periodic voxel image)"
            )
    by = np.argsort(inverse, kind="stable")
    same = inverse[by][1:] == inverse[by][:-1]
    first, second = by[:-1][same], by[1:][same]
    e1, f1, e2, f2 = first // 4, first % 4, second // 4, second % 4
    centroids = mesh.nodes[corners[e1, f1]].mean(axis=1)
    selected, swap = _selection(mesh, faces, e1, e2, centroids)
    e1, e2 = np.where(swap, e2, e1), np.where(swap, e1, e2)
    f1, f2 = np.where(swap, f2, f1), np.where(swap, f1, f2)
    # the local nodes of the two sides of each shared face, in the shared order
    l1 = np.take_along_axis(local[f1], order[e1, f1], axis=1)
    l2 = np.take_along_axis(local[f2], order[e2, f2], axis=1)
    kept = ~selected
    graph = sp.coo_matrix(
        (
            np.ones(3 * int(kept.sum())),
            ((4 * e1[kept, None] + l1[kept]).ravel(), (4 * e2[kept, None] + l2[kept]).ravel()),
        ),
        shape=(4 * m, 4 * m),
    )
    n_groups, group = connected_components(graph, directed=False)
    # the points away from the opened faces keep their nodes, whatever holds their elements
    flat = tets.ravel()
    touched = np.zeros(n, dtype=bool)
    touched[cls[np.take_along_axis(tets[e1[selected]], l1[selected], axis=1)]] = True
    group = np.where(touched[cls[flat]], group, n_groups + cls[flat])
    _, group = np.unique(group, return_inverse=True)
    n_groups = int(group.max(initial=-1)) + 1
    pair, inverse = np.unique(flat * n_groups + group, return_inverse=True)
    orig, copy = pair // n_groups, pair % n_groups
    head = np.ones(len(pair), dtype=bool)
    head[1:] = orig[1:] != orig[:-1]
    ids = np.where(head, orig, 0)
    ids[~head] = n + np.arange(int((~head).sum()))
    new_tets = ids[inverse.ravel()].reshape(m, 4)
    opened = mesh._derived(np.concatenate([mesh.nodes, mesh.nodes[orig[~head]]]), new_tets)
    node_copy = np.full(opened.n_nodes, -1, dtype=np.int64)
    node_copy[ids] = copy
    links = _links(ids, copy, orig, shift)
    # the interface elements: the normal from side - to side +
    a, b = e1[selected], e2[selected]
    la, lb = l1[selected], l2[selected]
    minus = np.take_along_axis(new_tets[a], la, axis=1)
    plus = np.take_along_axis(new_tets[b], lb, axis=1)
    oa, ob = np.take_along_axis(tets[a], la, axis=1), np.take_along_axis(tets[b], lb, axis=1)
    periods = shift[ob[:, 0]] - shift[oa[:, 0]]
    x = mesh.nodes[oa]
    normal = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    towards = mesh.nodes[tets[b]].mean(axis=1) - periods * size - mesh.nodes[tets[a]].mean(axis=1)
    flip = np.einsum("ij,ij->i", normal, towards) < 0
    minus[flip] = minus[flip][:, [0, 2, 1]]
    plus[flip] = plus[flip][:, [0, 2, 1]]
    phases = np.stack([mesh.phase[a], mesh.phase[b]], axis=1)
    return opened, Interfaces(minus, plus, phases), periods, links, node_copy


def open_cell(mesh: TetMesh, faces="phases", *, periodic: bool = True):
    """Open faces of the mesh of a periodic cell on its torus (see the module): ``faces``
    is ``"phases"`` (the faces between different phases; side - the lower phase),
    ``"grains"`` (between different grains of a polycrystal, ``mesh.cell``), pairs of
    phases (the faces between the two phases of a pair) or a function of the centroids
    (rows) of the faces that two elements share, returning a mask.

    Returns the mesh with the duplicated nodes, whose :meth:`~morphogen.mesh.tetra.
    TetMesh.periodic_links` are those of the copies (the two lips of an interface that
    crosses a face of the cell each linked to their own translates); the interface
    elements of the opened faces; and for each of them the periods (three integers) from
    its side - to its side +: zeros inside the cell, the period that an interface on a
    face of the cell crosses. ``periodic`` false: the faces of the cell are not joined
    (:func:`~morphogen.fracture.cohesive.insert_interfaces`), as for kinematic
    conditions. A mesh of 10-node elements is opened by its corners, its mid-edge nodes
    made again where they were."""
    if mesh.order == 1:
        opened, interfaces, periods, links, _ = _open_corners(mesh, faces, periodic)
        opened._cache["links"] = links
        return opened, interfaces, periods
    opened, interfaces, periods, links, node_copy = _open_corners(mesh.linear(), faces, periodic)
    quad, interfaces = _with_mid_nodes(mesh, opened, interfaces)
    # the mid-edge nodes: translates on the torus when their original nodes are, between
    # the same copies of their two corners
    cls, shift = _classes(mesh, periodic)
    mids, index = np.unique(quad.elements[:, 4:].ravel(), return_index=True)
    orig = mesh.elements[:, 4:].ravel()[index]
    ends = node_copy[quad.elements[:, TET_EDGES].reshape(-1, 2)[index]]
    key = np.column_stack([cls[orig], ends.min(axis=1), ends.max(axis=1)])
    _, copy = np.unique(key, axis=0, return_inverse=True)
    more = _links(mids, copy.ravel(), orig, shift)
    quad._cache["links"] = tuple(np.concatenate([a, b]) for a, b in zip(links, more, strict=True))
    return quad, interfaces, periods


def _laws_by_pair(law) -> dict | None:
    """The laws of a mapping of pairs of phases (keys: two phases; the pairs sorted), or
    None for one law: an object, or a mapping of its parameters."""
    if not isinstance(law, dict):
        if not (hasattr(law, "kind") and hasattr(law, "params")):
            raise TypeError(
                "CohesiveCell: a traction-separation law (Bilinear, Exponential or Contact "
                "of morphogen.fracture), a mapping of its parameters, or a mapping of pairs "
                "of phases to their laws"
            )
        return None
    if not law or not all(isinstance(key, (tuple, list, frozenset)) for key in law):
        return None
    out = {}
    for key, value in law.items():
        try:
            pair = tuple(sorted(int(p) for p in key))
        except (TypeError, ValueError):
            pair = ()
        if len(pair) != 2:
            raise ValueError(f"CohesiveCell: laws by pairs of two phases, got the key {key!r}")
        if not isinstance(value, dict) and not (
            hasattr(value, "kind") and hasattr(value, "params")
        ):
            raise TypeError(f"CohesiveCell: the law of the pair {pair}: a law, or its parameters")
        out[pair] = value
    return out


def _owners(mesh: TetMesh, interfaces: Interfaces) -> np.ndarray:
    """The elements on the two sides of each interface element (elements x 2): those of
    which the corners of its side - and of its side + are a face."""
    corners = np.sort(mesh.elements[:, np.asarray(TET_FACES)].reshape(-1, 3), axis=1)
    sides = [np.sort(side[:, :3], axis=1) for side in (interfaces.minus, interfaces.plus)]
    _, inverse = np.unique(np.concatenate([corners, *sides]), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    owner = np.full(int(inverse.max()) + 1, -1, dtype=np.int64)
    owner[inverse[: len(corners)]] = np.repeat(np.arange(mesh.n_elements), 4)
    n = interfaces.n_elements
    found = owner[inverse[len(corners) :]].reshape(2, n).T
    if np.any(found < 0):
        raise ValueError("CohesiveCell: interface elements that are not faces of the mesh")
    return found


def _point_positions(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The positions (elements x points x 3) of the integration points of interface
    elements, on their side ``faces`` (3 or 6 nodes)."""
    x = nodes[faces]
    if faces.shape[1] == 3:
        shapes = np.array([[1.0 - xi - eta, xi, eta] for xi, eta in _GAUSS])
    else:
        shapes = np.empty((len(_GAUSS6), 6))
        dxi, deta = np.empty(6), np.empty(6)
        for g, (xi, eta, _) in enumerate(_GAUSS6):
            _shape6(xi, eta, shapes[g], dxi, deta)
    return np.einsum("gi,eic->egc", shapes, x)


class CohesiveCell(NonlinearFEProblem):
    """A cell whose interfaces may debond along a loading path (see the module): the
    ``mesh`` of its phases (4- or 10-node tetrahedra; periodic conditions need the same
    faces of elements on the opposite faces of the cell), their ``materials``, the
    traction-separation ``law`` of the interfaces (:class:`~morphogen.fracture.cohesive.
    Bilinear`, :class:`~morphogen.fracture.cohesive.Exponential`, or :class:`~morphogen.
    fracture.cohesive.Contact` for lips without adhesion, each with its coefficient of
    ``friction``; or a mapping of its parameters: :func:`interface_law`, with the
    stiffest Young's modulus of the cell and the size of its interface elements) and the
    boundary conditions (``"periodic"`` or ``"kubc"``). A law for each pair of phases: a
    mapping of the pairs to their laws, ``{(0, 1): law, (0, 2): other}`` — the interfaces
    are then those of these pairs, unless ``faces`` says otherwise (``"grains"``: every
    boundary of a grain, with the law of the two phases that it parts, ``(1, 1)`` for two
    grains of the phase 1).

    ``faces``: the faces that are interfaces (:func:`open_cell`: ``"phases"``,
    ``"grains"``, pairs of phases or a function of centroids). ``viscosity``: the time of
    the viscous regularization of the damage of the interfaces (Gao & Bower 2004), in the
    time of the paths; 0: none. ``slip``: the elastic slip of the friction, over which
    lips that stick reach Coulomb's bound (a stiffness of the stick of ``μ p / slip``,
    whatever that of the law; by default 0.005 times the size of the interface
    elements): the friction of lips that come to touch, whose pressure and slip both
    start from nothing, would otherwise have no direction. The other options are those of
    :class:`~morphogen.homogenization.nonlinear.NonlinearFEProblem` (``kinematics``:
    ``"small"`` by default, or ``"finite"``, the tractions then turning with the
    interfaces; ``newton_tol``, ``max_cuts``...); the tangent is factorized
    (``solver="direct"``) and kept whole.

    :meth:`run` returns the response to a path with, in its ``series`` (columns of its
    curves, :data:`SERIES`), at each step the area of the interfaces whose points are
    broken ``debonded_area`` (reference area), their mean damage ``interface_damage``,
    their largest normal opening ``interface_opening`` and the energy that friction
    dissipated ``interface_dissipation``. Its fields have, for each element, the damage
    and the opening of the interface elements on its faces (``interface_damage``,
    ``interface_opening``: the largest of their means; zero away from the interfaces).

    ``laws``: the laws of the interfaces, by pair of phases; ``law``: the first of them."""

    def __init__(
        self,
        mesh: TetMesh,
        materials,
        law,
        bc: str = "periodic",
        *,
        faces="phases",
        viscosity: float = 0.0,
        slip: float | None = None,
        **options,
    ):
        if options.pop("solver", "direct") not in ("direct", "auto"):
            raise ValueError(
                "CohesiveCell: the tangent of the interfaces is factorized (solver 'direct')"
            )
        by_pair = _laws_by_pair(law)
        if by_pair is not None and isinstance(faces, str) and faces == "phases":
            faces = [pair for pair in by_pair if pair[0] != pair[1]]
        if not (math.isfinite(viscosity) and viscosity >= 0):
            raise ValueError("CohesiveCell: a viscosity of 0 or more")
        options.setdefault("tangent", "full")
        super().__init__(mesh, materials, bc, solver="direct", **options)
        if self.empty:
            raise ValueError("CohesiveCell: no load-carrying material in the cell")
        self.viscosity = float(viscosity)
        # the mesh that carries the load (voids and floating parts left out), opened
        opened, interfaces, periods = open_cell(self.mesh, faces, periodic=self.bc == "periodic")
        if not interfaces.n_elements:
            raise ValueError("CohesiveCell: no interface to open in the cell (faces)")
        self.mesh, self.interfaces, self.periods = opened, interfaces, periods
        self._areas = interfaces.areas(opened)
        self._set_laws(law if by_pair is None else by_pair, materials)
        #: the opening below which the lips of a point touch, for the predictor of a step
        self._touch = 1e-6 * math.sqrt(float(self._areas.mean()))
        self.slip = 0.005 * math.sqrt(float(self._areas.mean())) if slip is None else float(slip)
        if not (math.isfinite(self.slip) and self.slip > 0):
            raise ValueError("CohesiveCell: a positive elastic slip")
        plus = interfaces.plus.copy()
        crossing = np.any(periods != 0, axis=1)
        node, offset = None, None
        if crossing.any():  # the images of the nodes of the sides + across the faces
            mine = interfaces.plus[crossing]
            moved = np.repeat(periods[crossing][:, None, :], mine.shape[1], axis=1)
            key = np.column_stack([mine.ravel(), moved.reshape(-1, 3)])
            images, inverse = np.unique(key, axis=0, return_inverse=True)
            plus[crossing] = opened.n_nodes + inverse.reshape(mine.shape)
            node, offset = images[:, 0], -images[:, 1:] * np.asarray(opened.size, dtype=float)
        self._extended = Extended(opened.nodes, node, offset)
        self._plus = np.ascontiguousarray(plus)
        side = np.hstack([interfaces.minus, self._plus])
        nd = 3 * side.shape[1]
        self._idofs = (3 * side[:, :, None] + np.arange(3)).reshape(len(side), nd)
        self._rows = np.repeat(self._idofs, nd, axis=1).ravel()
        self._cols = np.tile(self._idofs, (1, nd)).ravel()
        self._points = 3 if interfaces.order == 1 else 6
        self._owners = _owners(opened, interfaces)
        # parts that the interfaces alone hold, once they are broken: a shift of the
        # diagonal of the tangent
        self.regularization = max(self.regularization, 1e-12)
        # the scales of the residuals have floors: a cell whose interfaces are broken
        # carries nothing, and no residual is small against forces that are zero. The
        # force of the strength of the interfaces on one of their elements, and that
        # strength as a stress of the cell
        strength = self._strength(materials)
        self._floor = strength * float(self._areas.mean())
        self._floor_stress = strength * self.box_volume
        #: the line search of Newton's corrections is not monotone: the residual may
        #: rise, below the largest of this number of the last iterates, where lips open
        #: and close along a correction (an active-set iteration)
        self._window = WINDOW
        self._virial = 0.0
        self._dt = 0.0
        self._begin()

    def _set_laws(self, law, materials) -> None:
        """The laws of the interface elements: one for all (a law, or a mapping of its
        parameters), or one for each pair of phases (a mapping of the pairs to theirs)."""
        from morphogen.contact.problem import stiffest_modulus

        stiffest = stiffest_modulus(materials, self.mesh.phase)
        size = math.sqrt(float(self._areas.mean()))

        def made(spec):
            return interface_law(spec, stiffest, size) if isinstance(spec, dict) else spec

        n = self.interfaces.n_elements
        if not isinstance(law, dict) or not all(isinstance(key, tuple) for key in law):
            self.laws = {None: made(law)}
            index = [None]
        else:
            pairs = np.sort(self.interfaces.phases, axis=1)
            found = sorted({(int(p), int(q)) for p, q in pairs})
            missing = [pair for pair in found if pair not in law]
            if missing:
                raise ValueError(
                    f"CohesiveCell: no law for the interfaces between the phases {missing} "
                    f"(laws: {sorted(law)})"
                )
            self.laws = {pair: made(law[pair]) for pair in found}
            index = [np.flatnonzero((pairs[:, 0] == p) & (pairs[:, 1] == q)) for p, q in found]
        self.law = next(iter(self.laws.values()))
        self.par = self.law.params()
        #: for each law: its parameters, its interface elements (None: all of them)
        self._groups = [
            (one, one.params(), None if where is None or len(where) == n else where)
            for one, where in zip(self.laws.values(), index, strict=True)
        ]

    def _strength(self, materials) -> float:
        """The strength of the interfaces: the lowest of those of their laws, or for lips
        without adhesion the stress of a strain of 1e-6 in the stiffest phase."""
        values = [
            getattr(law, name)
            for law in self.laws.values()
            for name in ("normal_strength", "shear_strength", "strength")
            if hasattr(law, name)
        ]
        if values:
            return float(min(values))
        from morphogen.contact.problem import stiffest_modulus

        return 1e-6 * stiffest_modulus(materials, self.mesh.phase)

    # -- the periodic conditions of the opened mesh ------------------------------------------

    def _reduction(self, key):
        if key[0] != "periodic":
            return super()._reduction(key)
        n = self.mesh.n_nodes
        slave, master, _ = self.mesh.periodic_links()
        target = np.arange(n)
        target[slave] = master
        masters = np.flatnonzero(target == np.arange(n))
        position = np.full(n, -1)
        position[masters] = np.arange(masters.size)
        t_nodes = sp.csr_matrix(
            (np.ones(n), (np.arange(n), position[target])), shape=(n, masters.size)
        )
        t = sp.kron(t_nodes, sp.identity(3), format="csr")
        rep = (masters[:, None] * 3 + np.arange(3)).ravel()
        keep = np.arange(0 if self.has_voids else 3, t.shape[1])  # translations: a node held
        return t[:, keep].tocsr(), rep[keep]

    def _stiffness(self):
        raise NotImplementedError(
            "CohesiveCell: the response to loading paths (run); the linear properties are "
            "those of the mesh without interfaces (FEElasticProblem)"
        )

    # -- the forces of the interfaces ----------------------------------------------------------

    def _discretization(self) -> _SurfacePaths:
        return _SurfacePaths(self)

    def _surface_forces(self, x, tangent: bool, first: bool):
        """The forces of the interface elements at the positions ``x`` of the extended
        nodes, and their tangent; the state of their points is tried from that of the
        start of the step. ``first``: the tangent is that of the predictor of a step, the
        lips that touched at its start closed, and sticking where they are pressed —
        those of broken points whose opening was below a millionth of the size of the
        interface elements: a cell brought back to its shape at rest has all its lips
        there, open or closed by a round-off."""
        ext = self._extended
        u = np.ascontiguousarray(x - ext.reference)
        ni, nd = self._idofs.shape
        fe = np.zeros((ni, nd))
        ke = np.zeros((ni if tangent else 1, nd, nd))
        trial = np.zeros_like(self._state)
        kernel, gauss = (_interfaces, _GAUSS) if nd == 18 else (_interfaces6, _GAUSS6)
        predict = tangent and first
        start = None
        if predict:  # the lips that touched at the start of the step
            start = self._state.copy()
            start[..., 4] = np.where(start[..., 4] <= self._touch, 0.0, start[..., 4])
        for law, base, where in self._groups:
            par = base.copy()
            par[6] = self._dt / self.viscosity if self.viscosity > 0 else 0.0
            par[9] = -self.slip  # the elastic slip of the friction
            every = where is None
            minus = self.interfaces.minus if every else self.interfaces.minus[where]
            plus = self._plus if every else self._plus[where]
            state = self._state if every else self._state[where]
            mine = (trial, fe, ke) if every else (
                np.zeros_like(state),
                np.zeros((len(where), nd)),
                np.zeros((len(where) if tangent else 1, nd, nd)),
            )  # fmt: skip
            shared = (ext.reference, minus, plus, u, self.finite, law.kind)
            kernel(*shared, par, state, mine[0], gauss, tangent and not predict, mine[1], mine[2])
            if predict:
                par = par.copy()
                par[10] = 1.0
                begun = start if every else start[where]
                kernel(*shared, par, begun, np.zeros_like(state), gauss, True,
                       np.zeros_like(mine[1]), mine[2])  # fmt: skip
            if not every:
                trial[where], fe[where] = mine[0], mine[1]
                if tangent:
                    ke[where] = mine[2]
        self._trial = trial
        n = 3 * len(ext.node)
        f = np.bincount(self._idofs.ravel(), fe.ravel(), minlength=n)
        if not tangent:
            return f, None
        return f, sp.csr_matrix((ke.ravel(), (self._rows, self._cols)), shape=(n, n))

    def _extra_stress(self):
        """The part of the macroscopic stress that the interfaces on the faces of the
        cell carry (see the module), at the last evaluation."""
        return self._virial / self.box_volume

    # -- the state of the interfaces along a path ---------------------------------------------

    def _begin(self) -> None:
        """Sound interfaces, before a path."""
        self._state = np.zeros((self.interfaces.n_elements, self._points, STATE))
        self._trial = self._state.copy()
        self._virial = 0.0
        self._history = [(0.0, 0.0, 0.0, 0.0)]

    def damage(self) -> np.ndarray:
        """The damage of the points of the interfaces (elements x points) at the end of
        the last step."""
        hist = self._state[..., 0]
        out = hist.copy()  # the bilinear law: its history is its damage
        for law, par, where in self._groups:
            mine = slice(None) if where is None else where
            if law.kind == CONTACT:  # the lips of a crack: broken; bonded points: sound
                out[mine] = hist[mine] >= 0.0
            elif law.kind != BILINEAR:
                out[mine] = 1.0 - np.exp(-hist[mine] * math.e * par[0] / par[1])
        return out

    def _settle(self, state) -> bool:
        """On a converged state: the state of the points of the interfaces is kept, and
        the step recorded."""
        self._state = self._trial
        damage = self.damage()
        weights = self.interfaces.weights
        broken = float(self._areas @ ((damage >= BROKEN) @ weights))
        mean = float(self._areas @ (damage @ weights)) / float(self._areas.sum())
        opening = max(float(self._state[..., 4].max()), 0.0)
        rubbed = float(self._areas @ (self._state[..., 3] @ weights))
        self._history.append((broken, mean, opening, rubbed))
        return False

    def run(self, path, **options):
        """The response to a loading path (:meth:`~morphogen.homogenization.nonlinear.
        NonlinearFEProblem.run`) from sound interfaces, with the records of the
        interfaces in its ``series`` (see the class)."""
        self._begin()
        try:
            result = super().run(path, **options)
        except RuntimeError as exc:
            if self.viscosity > 0 or all(law.kind == CONTACT for law in self.laws.values()):
                raise
            raise RuntimeError(
                f"{exc}; interfaces that break at once have no equilibrium under prescribed "
                "strains (a snap-back): give the cell a viscosity, the time of the "
                "regularization of their damage"
            ) from exc
        history = np.array(self._history[: len(result.times)])
        result.series.update({name: history[:, j] for j, name in enumerate(SERIES)})
        return result

    def _fields(self, sigma, grads, state, u) -> dict:
        """With the damage and the opening of the interfaces as fields of the elements
        that they part (the largest of the means of the interface elements on the faces
        of each element)."""
        out = super()._fields(sigma, grads, state, u)
        weights = self.interfaces.weights
        damage = self.damage() @ weights
        opening = np.maximum(self._state[..., 4], 0.0) @ weights
        for name, values in (("interface_damage", damage), ("interface_opening", opening)):
            field = np.zeros(self.mesh.n_elements)
            for side in range(2):
                np.maximum.at(field, self._owners[:, side], values)
            out[name] = field
        return out

    def interface_points(self) -> dict:
        """The integration points of the interfaces at the end of the last path (rows:
        elements x points): their ``position`` (reference), ``damage``, normal
        ``opening`` (negative: lips pressed on each other), frictional ``slip`` (two
        components in the frame of each point), energy ``dissipated`` by friction per
        unit area and ``weight`` (reference area)."""
        weights = self.interfaces.weights
        return {
            "position": _point_positions(self.mesh.nodes, self.interfaces.minus).reshape(-1, 3),
            "damage": self.damage().ravel(),
            "opening": self._state[..., 4].ravel(),
            "slip": self._state[..., 1:3].reshape(-1, 2),
            "dissipated": self._state[..., 3].ravel(),
            "weight": (self._areas[:, None] * weights[None, :]).ravel(),
        }
