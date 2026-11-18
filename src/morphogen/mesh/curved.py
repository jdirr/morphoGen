"""Quadratic tetrahedra whose nodes lie on the interfaces of the phases.

A conforming mesh (:func:`morphogen.mesh.conforming.conforming_mesh`) follows the
interfaces with flat triangles, whose corners lie on the level set as the background
mesh interpolates it: the phases of the mesh are polyhedra, which lose the volume
between the chords and the interface (a few percent of an inclusion meshed with a few
elements per radius). Ten-node tetrahedra made of them by adding the middles of their
edges (:meth:`~morphogen.mesh.tetra.TetMesh.quadratic`) keep that geometry: their
quadratic displacements are then more accurate than the domain they are computed on.

:func:`curved_mesh` puts the nodes of the interfaces on the implicit surfaces of the
microstructure (:meth:`~morphogen.models.base.Microstructure.phase_level_set`), the
corner nodes and the mid-edge nodes: each face of an interface becomes the quadratic
patch through six points of the surface, and the elements on both sides isoparametric
elements with curved edges (the other edges stay straight). The distance of the mesh to
the interface then falls as the cube of the size of the elements instead of its square,
and the volumes of the phases as its fourth power.

The surface between the phases ``a`` and ``b`` is the zero of ``φ_a - φ_b``, as for the
mesher; the wall of a pore left out of the mesh, that of the function of its solid. The
structures that give the functions of their phases at any point
(:meth:`~morphogen.models.base.Microstructure.phase_level_sets_at`: the unions of spheres
and of segments and their coatings, the fields, the leaves, the cells, the combinations of
these...) have their nodes put on their surfaces exactly; the functions of the others are
sampled on a grid finer than the elements and interpolated by cubic splines (which round
the folds of these functions over a cell or two: a layer thinner than a few cells loses or
gains some volume). This is the one curvature of the 10-node elements, of the meshes made
on a grid and of the exact ones (:mod:`morphogen.mesh.exact`) alike. A node is moved to
the zero along the gradient (Newton's method), within the
faces of the box for the nodes that lie on them (so that the faces stay plane, and
opposite faces keep matching nodes). Nodes where more than two phases meet stay where
they are, as do the nodes of whole periodic cells of polycrystals, whose faces are
plane. A move is bounded by a share of the edges around the node, and reduced where it
would flatten an element (the straight tetrahedra after the moves of the corners) or
where the volume element of a curved one would fall below a share of that of its
straight tetrahedron: the node of the element that moves most gives half of its move
away, until the element holds; the corner nodes held back start again from where they
are, in three steps at most.

Sharp edges. A mesh that follows the sharp edges of its phases (the grooves where the
spheres or segments of a union overlap; ``sharp_edges`` of
:func:`~morphogen.mesh.conforming.conforming_mesh`) has edges along them
(:attr:`~morphogen.mesh.tetra.TetMesh.sharp`), and knows the two smooth pieces of the
structure that meet along each (:mod:`morphogen.models.pieces`). The nodes of these
edges, corners and middles, are put where the surfaces of the two pieces meet — the
smallest move that brings them on both, by Newton's method — and on the three surfaces
where three sharp edges join: the curved edges of the elements follow the sharp edge.

What the curved elements do not mend is what the mesher did not resolve: the sharp edges
of the structures that do not give their smooth pieces are rounded over about an
element, as in the mesh of 4-node elements, and the gaps and layers thinner than the
grid of the mesher (``background``) are bridged or pinched. The nodes there stay off
their surfaces (``mesh.metadata["curved"]`` counts those that lie on them).
"""

from __future__ import annotations

import logging
import time
from functools import partial

import numpy as np
from scipy import ndimage

from morphogen.mesh.tetra import (
    TET10_NODES,
    TET_EDGES,
    TET_FACE_MIDS,
    TET_FACES,
    TetMesh,
    corner_gradients,
    jacobian_ratios,
    tet_quality,
    tet_volumes,
)

log = logging.getLogger("morphogen")

#: Cells of the grid of the implicit functions per size of the elements, the cells added
#: around the box for the splines, and the largest number of nodes of the grid along an
#: axis (a coarser sampling beyond it)
SAMPLES_PER_ELEMENT = 8
_PAD = 6
MAX_SAMPLES = 256
#: The steps of the corner nodes towards their surfaces, and the rounds of each way of
#: reducing the moves that an element does not bear (see :func:`_give_way`)
_STEPS = 3
_ROUNDS = 12
#: The four integration points of the 10-node tetrahedron (barycentric coordinates)
_GAUSS = np.full((4, 4), 0.1381966011250105) + (0.5854101966249685 - 0.1381966011250105) * np.eye(4)


class _Implicit:
    """The implicit functions of the phases of a microstructure at any point: those of
    the structure itself where it gives them there
    (:meth:`~morphogen.models.base.Microstructure.phase_level_sets_at`; an object with a
    ``level_set_at`` and no functions on grids: its two phases), exact; otherwise the
    functions sampled at the nodes of a regular grid of the box and interpolated by cubic
    splines (periodic along the ``axes`` where the mesh has matching faces)."""

    def __init__(self, micro, spacing: float, axes, phases, reach: float):
        from morphogen.mesh.conforming import node_level_set
        from morphogen.models.implicit import functions_at, implicit_phases_of, point_functions

        present = implicit_phases_of(micro)
        lengths = np.asarray(micro.domain.lengths, dtype=float)
        self.lower = np.asarray(micro.domain.lower, dtype=float)
        self.coefficients: dict[int, np.ndarray] = {}
        self.exact = None
        two = present == [0, 1]
        for k in phases:
            if k not in present:
                raise ValueError(f"phase {k}: the phases of this structure are {present}")
        at = getattr(micro, "implicit_at", None)
        centre = (self.lower + 0.5 * lengths)[None, :]
        if at is not None:
            exact = all(at(k, centre, 2.0 * reach) is not None for k in phases)
        else:  # an object with point-wise functions only
            exact = not hasattr(micro, "level_set") and point_functions(micro)
        if exact:
            # the functions of the structure at the points themselves, within the size of
            # the elements (the moves are a share of their edges; the functions of many
            # objects, leaves or cells, cost as the cube of that reach)
            self.exact = partial(functions_at, micro, reach=reach)
            self.band = reach
            self.spacing = np.full(3, 1e-3 * reach)  # the step of the differences
            self.sampling = 0.0
            return
        for k in sorted(phases):
            if two and (1 - k) in self.coefficients:  # the other phase: the opposite
                self.coefficients[k] = -self.coefficients[1 - k]
                continue
            values, shape = node_level_set(micro, spacing, None if two else k)
            if two and k == 0:
                values = -values
            self.spacing = lengths / np.array(shape)
            for d in range(3):
                width = [(0, 0)] * 3
                width[d] = (_PAD, _PAD)
                if axes[d]:  # the last nodes are the first ones, one period further
                    values = np.pad(np.take(values, range(shape[d]), axis=d), width, mode="wrap")
                else:
                    values = np.pad(values, width, mode="reflect", reflect_type="odd")
            self.coefficients[k] = ndimage.spline_filter(values, order=3, mode="nearest")
        self.sampling = float(self.spacing.max())

    def _values(self, phases, points: np.ndarray) -> dict[int, np.ndarray]:
        """The functions of ``phases`` at the points."""
        if self.exact is not None:
            out = self.exact(points)
            return {k: np.clip(out[k], -self.band, self.band) for k in phases}
        index = ((points - self.lower) / self.spacing + _PAD).T
        return {
            k: ndimage.map_coordinates(
                self.coefficients[k], index, order=3, mode="nearest", prefilter=False
            )
            for k in phases
        }

    def gap(self, a: int, b: int, points: np.ndarray):
        """The function whose zero is the interface of the phases ``a`` and ``b`` (``b``
        negative: the boundary of ``a``) at ``points``, and its gradient (central
        differences over half a cell of the grid)."""

        def at(where):
            if b < 0:
                return self._values((a,), where)[a]
            out = self._values((a, b), where)
            return out[a] - out[b]

        value = at(points)
        gradient = np.empty_like(points)
        for d in range(3):
            shift = np.zeros(3)
            shift[d] = 0.5 * self.spacing[d]
            gradient[:, d] = (at(points + shift) - at(points - shift)) / self.spacing[d]
        return value, gradient


def _interfaces(mesh: TetMesh) -> tuple[np.ndarray, np.ndarray]:
    """The faces of a 10-node mesh between two phases, or between a phase and a pore left
    out of the mesh (the free faces inside the box): their six nodes (corners, then
    mid-edge nodes) and their two phases, the second -1 for a pore."""
    elements = mesh.elements
    corners = np.sort(elements[:, TET_FACES].reshape(-1, 3), axis=1)
    _, inverse, counts = np.unique(corners, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse.ravel(), kind="stable")
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    first = order[start]  # one (element, face) of each face of the mesh
    e1, f1 = first // 4, first % 4
    shared = counts == 2
    p1 = mesh.phase[e1]
    p2 = np.full(len(counts), -1, dtype=np.int64)
    p2[shared] = mesh.phase[order[start[shared] + 1] // 4]
    x = mesh.nodes[elements[e1[:, None], TET_FACES[f1]]]  # (faces, 3 corners, 3)
    tol = mesh.tolerance()
    on_box = np.zeros(len(counts), dtype=bool)
    for d in range(3):
        for target in (mesh.lower[d], mesh.upper[d]):
            on_box |= np.all(np.abs(x[:, :, d] - target) <= tol, axis=1)
    keep = (shared & (p1 != p2)) | (~shared & ~on_box)
    e1, f1, p1, p2 = e1[keep], f1[keep], p1[keep], p2[keep]
    faces = np.concatenate(
        [elements[e1[:, None], TET_FACES[f1]], elements[e1[:, None], TET_FACE_MIDS[f1]]], axis=1
    )
    low, high = np.minimum(p1, p2), np.maximum(p1, p2)
    pairs = np.where((p2 < 0)[:, None], np.stack([p1, p2], 1), np.stack([low, high], 1))
    return faces, pairs


def _matched_pairs(mesh: TetMesh, axes) -> list[np.ndarray]:
    """The pairs (node of a lower face, its image on the upper face) along the axes
    where the faces of the mesh match."""
    return [
        mesh.periodic_pairs(d) if axes[d] else np.zeros((0, 2), dtype=np.int64) for d in range(3)
    ]


def _representatives(n: int, pairs) -> np.ndarray:
    """The node that stands for each node and its images across the box (itself, for a
    node without images)."""
    rep = np.arange(n)
    for _ in range(3):
        for pair in pairs:
            if len(pair):
                rep[pair[:, 1]] = rep[pair[:, 0]]
    while True:
        further = rep[rep]
        if np.array_equal(further, rep):
            return rep
        rep = further


def _give_way(scale: np.ndarray, at: np.ndarray, far: np.ndarray, round_: int) -> None:
    """Reduce the moves of the nodes ``at`` of the elements that do not bear them (a row
    per element): in the first rounds the node of each element that moves most (``far``)
    gives half of its move away; in the next ones all of its nodes do; in the last ones
    they stay where they were."""
    if round_ < _ROUNDS:
        worst = at[np.arange(len(at)), np.argmax(scale[at] * far[at], axis=1)]
        scale[worst] *= 0.5
    elif round_ < 2 * _ROUNDS:
        scale[at.ravel()] *= 0.5
    else:
        scale[at.ravel()] = 0.0


def _share(scale: np.ndarray, pairs) -> None:
    """The smallest factor of a node and of its images across the box, for both."""
    for _ in range(3):  # the nodes of the edges and corners of the box have several images
        for pair in pairs:
            if len(pair):
                least = np.minimum(scale[pair[:, 0]], scale[pair[:, 1]])
                scale[pair[:, 0]] = scale[pair[:, 1]] = least


def _copy_images(nodes: np.ndarray, pairs) -> None:
    """The nodes of the upper faces at the places of their images of the lower faces."""
    for _ in range(3):
        for axis, pair in enumerate(pairs):
            if len(pair):
                others = [d for d in range(3) if d != axis]
                nodes[np.ix_(pair[:, 1], others)] = nodes[np.ix_(pair[:, 0], others)]


def _project(implicit, points, a, b, free, cap) -> tuple[np.ndarray, np.ndarray]:
    """``points`` moved to the zero of the function of their pair of phases along its
    gradient (Newton's method), within the directions ``free`` and by ``cap`` at most;
    the points that find no zero closer than where they started stay there. Returns the
    points and those that reached the surface."""
    out = points.copy()
    start, _ = implicit.gap(a, b, points)
    scale = float(implicit.spacing.min())
    value = start
    for _ in range(12):
        value, gradient = implicit.gap(a, b, out)
        gradient = gradient * free
        norm2 = (gradient**2).sum(axis=1)
        slope = np.sqrt(np.maximum(norm2, 1e-300))
        if np.all(np.abs(value) <= 1e-7 * scale * slope):
            break
        step = np.where(
            (norm2 > 1e-12)[:, None],
            -value[:, None] * gradient / np.maximum(norm2, 1e-300)[:, None],
            0.0,
        )
        move = out + step - points
        length = np.linalg.norm(move, axis=1)
        over = length > cap
        move[over] *= (cap[over] / length[over])[:, None]
        out = points + move
    value, gradient = implicit.gap(a, b, out)
    slope = np.sqrt(np.maximum(((gradient * free) ** 2).sum(axis=1), 1e-300))
    reached = np.abs(value) <= 1e-4 * scale * slope
    closer = np.abs(value) < np.abs(start)
    out[~closer] = points[~closer]
    return out, reached & closer


def _project_sharp(pieces, points, free, cap, step: float, band: float):
    """``points`` moved to where the surfaces of ``pieces`` meet — two pieces: the sharp
    edge along which they meet; three: the point where sharp edges join — by Newton's
    method on the functions of the pieces, the smallest move at each step, within the
    directions ``free`` and by ``cap`` at most. Returns the points and those that reached
    all the surfaces (the others are left where they were)."""
    out = points.copy()
    count = len(pieces)

    def functions(where):
        return np.stack([piece.at(where, band) for piece in pieces], axis=1)

    for _ in range(12):
        value = functions(out)
        if np.abs(value).max(initial=0.0) <= 1e-7 * step:
            break
        jacobian = np.empty((len(out), count, 3))
        for d in range(3):
            shift = np.zeros(3)
            shift[d] = step
            jacobian[:, :, d] = (functions(out + shift) - functions(out - shift)) / (2.0 * step)
        jacobian *= free[:, None, :]
        move = out - np.einsum("ndk,nk->nd", np.linalg.pinv(jacobian), value) - points
        length = np.linalg.norm(move, axis=1)
        over = length > cap
        move[over] *= (cap[over] / length[over])[:, None]
        out = points + move
    reached = np.abs(functions(out)).max(axis=1, initial=0.0) <= 1e-3 * step
    out[~reached] = points[~reached]
    return out, reached


def _sharp_nodes(quad: TetMesh, rep: np.ndarray) -> dict[int, tuple[int, ...]]:
    """The nodes of the sharp edges of a 10-node mesh (:attr:`~morphogen.mesh.tetra.
    TetMesh.sharp`) and the pieces that meet at each: the two of its edges for a corner
    node or the node of the middle of a sharp edge, more where sharp edges join; for
    the nodes that stand for them and their images across the box."""
    found: dict[int, set[int]] = {}
    if quad.sharp is None or quad.sharp_pieces is None or not len(quad.sharp):
        return {}
    ends = np.sort(quad.elements[:, TET_EDGES].reshape(-1, 2), axis=1)
    keys = ends[:, 0] * quad.n_nodes + ends[:, 1]
    keys, first = np.unique(keys, return_index=True)
    middles = quad.elements[:, 4:].reshape(-1)[first]
    edges = np.sort(quad.sharp, axis=1)
    wanted = edges[:, 0] * quad.n_nodes + edges[:, 1]
    place = np.clip(np.searchsorted(keys, wanted), 0, len(keys) - 1)
    there = keys[place] == wanted
    rows = zip(edges.tolist(), quad.sharp_pieces.tolist(), middles[place].tolist(),
               there.tolist(), strict=True)  # fmt: skip
    for (a, b), pair, middle, known in rows:
        for node in (a, b, middle) if known else (a, b):
            found.setdefault(int(rep[node]), set()).update(pair)
    return {node: tuple(sorted(pieces)) for node, pieces in found.items()}


def curved_mesh(
    mesh: TetMesh,
    micro,
    *,
    sampling: float | None = None,
    corners: bool = True,
    max_move: float = 0.3,
    min_jacobian: float = 0.3,
) -> TetMesh:
    """The 10-node mesh of ``mesh`` (4 or 10 nodes) whose nodes on the interfaces of the
    phases lie on the implicit surfaces of the microstructure ``micro``: isoparametric
    elements with curved edges along the interfaces (see the module).

    The mesh of a microstructure without curved interfaces (polycrystals meshed by their
    cells) comes back with straight edges. ``mesh.metadata["curved"]`` counts the nodes
    of the interfaces and those that lie on their surfaces in the end (``nodes``), and
    gives the largest departure of a mid-edge node from the middle of its edge. The
    nodes that stay off their surfaces are those of the lines where three phases meet,
    and those that an element holds back: at the sharp edges of a phase (the grooves
    where two overlapping spheres meet), which the mesher rounds, and in the gaps and
    layers that the grid of the mesher does not resolve (``background``).

    Parameters
    ----------
    sampling
        Spacing of the grid on which the implicit functions are sampled, for the
        structures that do not give them at any point (by default an eighth of the size
        of the elements, coarser for very fine meshes of large boxes).
    corners
        Move the corner nodes of the interfaces to the surfaces too (default); false
        moves only the mid-edge nodes.
    max_move
        The largest move of a node, as a share of the shortest edge around it (corner
        nodes, in each of their three steps at most) or of its edge (mid-edge nodes).
    min_jacobian
        The smallest ratio allowed between the volume element of a curved element (at
        its nodes and integration points) and that of its straight tetrahedron; the
        most bent edge of an element below it is straightened by half, until it holds.
    """
    quad = mesh.straight() if mesh.order == 2 else mesh.quadratic()
    if quad.whole or quad.n_elements == 0:
        return quad
    t0 = time.perf_counter()
    faces, pairs = _interfaces(quad)
    if not len(faces):
        return quad
    n = quad.n_nodes
    matching = quad.metadata.get("matching_faces", [0, 1, 2] if quad.periodic else [])
    axes = tuple(d in matching for d in range(3))
    images = _matched_pairs(quad, axes)
    rep = _representatives(n, images)
    # the pair of phases of each node of the interfaces, the same for its images across
    # the box: the nodes of one pair move
    width = int(pairs.max()) + 2
    code = pairs[:, 0] * width + (pairs[:, 1] + 1)
    lowest = np.full(n, np.iinfo(np.int64).max)
    highest = np.full(n, -1)
    np.minimum.at(lowest, rep[faces.ravel()], np.repeat(code, 6))
    np.maximum.at(highest, rep[faces.ravel()], np.repeat(code, 6))
    lowest, highest = lowest[rep], highest[rep]
    single = (highest >= 0) & (lowest == highest)
    tol = quad.tolerance()
    free = np.ones((n, 3))
    for d in range(3):
        on_face = (np.abs(quad.nodes[:, d] - quad.lower[d]) <= tol) | (
            np.abs(quad.nodes[:, d] - quad.upper[d]) <= tol
        )
        free[on_face, d] = 0.0
    movable = single & free.any(axis=1)
    is_corner = np.zeros(n, dtype=bool)
    is_corner[quad.elements[:, :4]] = True

    size = float(quad.metadata.get("size", 0.0)) or float(
        np.median(np.linalg.norm(np.diff(quad.nodes[quad.elements[:, TET_EDGES]], axis=2), axis=-1))
    )
    lengths = np.asarray(micro.domain.lengths, dtype=float)
    spacing = float(sampling) if sampling else size / SAMPLES_PER_ELEMENT
    spacing = max(spacing, float(lengths.max()) / MAX_SAMPLES)
    used = np.unique(pairs[np.isin(faces, np.flatnonzero(movable)).any(axis=1)])
    implicit = _Implicit(micro, spacing, axes, [int(k) for k in used if k >= 0], size)

    nodes = quad.nodes.copy()
    edges = quad.elements[:, TET_EDGES].reshape(-1, 2)
    # the nodes of the sharp edges that the mesh follows, and the pieces that meet there
    sharp = _sharp_nodes(quad, rep)
    pieces = None
    if sharp:
        depth = float(quad.metadata.get("sharp_edges", {}).get("depth", 0.0))
        found = getattr(micro, "implicit_pieces", lambda depth: None)(depth)
        pieces = None if found is None else found[0]
        if pieces is None or max(max(group) for group in sharp.values()) >= len(pieces):
            sharp, pieces = {}, None  # another structure than that of the mesh

    def groups(chosen: np.ndarray):
        """The nodes ``chosen`` by pair of phases: their places in ``chosen``, the pair."""
        for value in np.unique(lowest[chosen]):
            yield lowest[chosen] == value, int(value // width), int(value % width) - 1

    def project(chosen: np.ndarray, cap: np.ndarray, edges_too: bool = True) -> np.ndarray:
        """The places on their surfaces of the nodes ``chosen`` (those found): on the
        surface of their two phases, or (``edges_too``) where the pieces of a sharp edge
        meet."""
        target = nodes[chosen].copy()
        for sel, a, b in groups(chosen):
            target[sel], _ = _project(
                implicit, nodes[chosen[sel]], a, b, free[chosen[sel]], cap[sel]
            )
        if sharp and edges_too:
            along: dict[tuple, list[int]] = {}
            for place, node in enumerate(chosen.tolist()):
                if node in sharp:
                    along.setdefault(sharp[node], []).append(place)
            for group, places in along.items():
                at = np.array(places)
                moved, reached = _project_sharp(
                    [pieces[k] for k in group], nodes[chosen[at]], free[chosen[at]], cap[at],
                    1e-3 * size, 2.0 * size,
                )  # fmt: skip
                target[at[reached]] = moved[reached]
        return target

    # 1. the corner nodes, as far as the straight tetrahedra keep their shape: in a few
    #    steps, the nodes that an element held back starting again from where they are
    own = rep == np.arange(n)  # one node for a node and its images
    chosen = np.flatnonzero(movable & is_corner & own)
    if corners and chosen.size:
        volume0 = tet_volumes(nodes, quad.elements)
        quality0 = tet_quality(nodes, quad.elements)
        # a last step without the sharp edges: a node that the elements hold back from
        # its sharp edge goes to the surface of its two phases, a shorter way
        for step in range(_STEPS + bool(sharp)):
            edge_length = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
            shortest = np.full(n, np.inf)
            np.minimum.at(shortest, edges[:, 0], edge_length)
            np.minimum.at(shortest, edges[:, 1], edge_length)
            move = np.zeros((n, 3))
            cap = max_move * shortest[chosen]
            move[chosen] = project(chosen, cap, step < _STEPS) - nodes[chosen]
            move = move[rep]  # the images of a node across the box move as it does
            far = np.linalg.norm(move, axis=1)
            if far.max() <= 1e-9 * size:
                if step < _STEPS and sharp:
                    continue
                break
            scale = np.ones(n)
            bad = np.zeros(quad.n_elements, dtype=bool)
            check = np.flatnonzero((far[quad.elements[:, :4]] > 0.0).any(axis=1))
            for round_ in range(3 * _ROUNDS):
                trial = nodes + scale[:, None] * move
                part = quad.elements[check]
                bad[check] = (tet_volumes(trial, part) < 0.5 * volume0[check]) | (
                    tet_quality(trial, part) < 0.5 * quality0[check]
                )
                if not bad.any():
                    break
                before = scale.copy()
                _give_way(scale, quad.elements[bad, :4], far, round_)
                _share(scale, images)
                # the elements of the nodes whose moves changed are checked again
                check = np.flatnonzero((scale != before)[quad.elements[:, :4]].any(axis=1))
            else:  # an element still does not hold: no node moves in this step
                if step < _STEPS and sharp:
                    continue
                break
            nodes += scale[:, None] * move
            _copy_images(nodes, images)
    # 2. the mid-edge nodes: the middles of the straight edges, then those of the
    #    interfaces on their surfaces
    ends = quad.elements[:, TET_EDGES]
    nodes[quad.elements[:, 4:]] = 0.5 * (nodes[ends[..., 0]] + nodes[ends[..., 1]])
    middles = nodes.copy()
    chosen = np.flatnonzero(movable & ~is_corner & own)
    if chosen.size:
        length = np.ones(n)
        length[quad.elements[:, 4:].ravel()] = np.linalg.norm(
            nodes[ends[..., 0]] - nodes[ends[..., 1]], axis=-1
        ).ravel()
        move = np.zeros((n, 3))
        cap = max_move * length[chosen]
        move[chosen] = project(chosen, cap) - nodes[chosen]
        move = move[rep]
        scale = np.ones(n)
        mids = quad.elements[:, 4:]
        points = np.concatenate([TET10_NODES, _GAUSS])
        gradients = np.zeros((quad.n_elements, 4, 3))
        bent = np.flatnonzero((np.abs(move[mids]).max(axis=2) > 0.0).any(axis=1))
        gradients[bent] = corner_gradients(nodes, quad.elements[bent])

        def hold() -> None:
            """Reduce the moves of the mid-edge nodes until every element bears them."""
            far = np.linalg.norm(move, axis=1) / length
            bad = np.zeros(quad.n_elements, dtype=bool)
            check = np.flatnonzero((far[mids] > 0.0).any(axis=1))  # the elements with a bent edge
            for round_ in range(3 * _ROUNDS):
                # the departures of the mid-edge nodes from the middles of their edges
                deviation = scale[mids[check], None] * move[mids[check]]
                ratio = jacobian_ratios(gradients[check], deviation, points).min(axis=1)
                bad[check] = ratio < min_jacobian
                if not bad.any():
                    return
                before = scale.copy()
                _give_way(scale, mids[bad], far, round_)
                _share(scale, images)
                check = np.flatnonzero((scale != before)[mids].any(axis=1))
            scale[:] = 0.0  # an element still does not hold: straight edges everywhere

        hold()
        # the middles of sharp edges that the elements hold back: to the surface of their
        # two phases instead, a shorter way
        late = np.array([k for k in chosen.tolist() if k in sharp and scale[k] < 1.0], dtype=int)
        if late.size:
            cap = max_move * length[late]
            again = np.zeros((n, 3))
            again[late] = project(late, cap, False) - nodes[late]
            again = again[rep]
            moved = np.flatnonzero(np.isin(rep, late))
            move[moved] = again[moved]
            scale[moved] = 1.0
            _share(scale, images)
            hold()
        nodes = middles + scale[:, None] * move
        _copy_images(nodes, images)
    out = quad._derived(nodes, quad.elements)
    # the nodes that lie on their surfaces in the end
    chosen = np.flatnonzero(movable)
    reached = 0
    for sel, a, b in groups(chosen):
        value, gradient = implicit.gap(a, b, nodes[chosen[sel]])
        slope = np.sqrt(np.maximum(((gradient * free[chosen[sel]]) ** 2).sum(axis=1), 1e-300))
        reached += int(np.count_nonzero(np.abs(value) <= 1e-6 * size * slope))
    out.metadata["curved"] = {
        "nodes": reached,
        "sharp_nodes": len(sharp),
        "interface_nodes": int(np.count_nonzero(highest >= 0)),
        "max_deviation": float(np.linalg.norm(nodes - middles, axis=1).max()),
        "sampling": implicit.sampling,  # 0: the functions of the model itself
        "time_s": round(time.perf_counter() - t0, 3),
    }
    log.info(
        "curved mesh: %d of %d nodes of the interfaces on their surfaces (%.1f s)",
        reached,
        int(np.count_nonzero(highest >= 0)),
        time.perf_counter() - t0,
    )
    return out
