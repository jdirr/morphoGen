"""Balanced octrees of a box, refined where a level set needs it, and their conforming
tetrahedralization.

The box is split into root cells (about ``size`` wide); a cell whose lines do not resolve
the level set (the test of :func:`morphogen.mesh.exact.adaptive_mesh` on its 27 lines of
three points: edges, middle lines of its faces and axes) is split into eight, down to the
smallest size. The octree is then balanced: two leaves that touch (by a face, an edge or a
corner, across the periodic faces too) differ by one level at most. Its tetrahedra are cones
from the centres of the leaves over their faces, each face (or quarter of a face, where the
leaf across it is finer) a fan from its centre over its corners and the nodes in the middle
of its sides: the same triangles on both sides of every face, so the mesh is conforming, and
on opposite faces of the box along the periodic axes, which match. Every tetrahedron joins
the centre of a leaf to a quarter of one of its faces at least: their shapes are bounded.
"""

from __future__ import annotations

import itertools

import numpy as np

from morphogen.mesh.tetra import tet_volumes

#: The 27 points of a cell, in half cells from its lower corner (index ``9 a + 3 b + c``)
LATTICE = np.array(list(itertools.product(range(3), repeat=3)), dtype=np.int64)
#: The eight children of a cell
CHILDREN = np.array(list(itertools.product(range(2), repeat=3)), dtype=np.int64)
#: The 26 neighbours of a cell
NEIGHBOURS = np.array([d for d in itertools.product((-1, 0, 1), repeat=3) if any(d)])


def _others(axis: int) -> tuple[int, int]:
    return tuple(d for d in range(3) if d != axis)  # type: ignore[return-value]


def _lines() -> list[tuple[int, list[int]]]:
    """The 27 lines of three points of :data:`LATTICE` (9 along each axis): the axis and
    the indices of their points in order."""
    out = []
    for d in range(3):
        b, c = _others(d)
        for u, v in itertools.product(range(3), repeat=2):
            idx = []
            for t in range(3):
                p = [0, 0, 0]
                p[d], p[b], p[c] = t, u, v
                idx.append(9 * p[0] + 3 * p[1] + p[2])
            out.append((d, idx))
    return out


LINES = _lines()


def _segments() -> np.ndarray:
    """The segments between points of :data:`LATTICE` that the tetrahedra of a cell, or of
    its children, may take as edges: those between neighbours (the unit squares and cubes
    of the lattice, their diagonals included) and those between corners (the edges of the
    cell, the diagonals of its faces and its own)."""
    corner = np.all(LATTICE % 2 == 0, axis=1)
    out = []
    for a, b in itertools.combinations(range(len(LATTICE)), 2):
        d = np.abs(LATTICE[a] - LATTICE[b]).max()
        if d == 1 or (d == 2 and corner[a] and corner[b]):
            out.append((a, b))
    return np.asarray(out, dtype=np.int64)


#: The 186 segments of :data:`LATTICE` that may be edges of tetrahedra (:func:`_segments`)
SEGMENTS = _segments()


class Octree:
    """Leaves ``(level, i, j, k)`` of an octree over ``n`` root cells along each axis of the
    box ``[lower, upper]`` (a leaf of level ``l`` is ``2^l`` times smaller than a root cell),
    periodic along ``axes``, at most ``maxlevel`` deep."""

    def __init__(self, lower, upper, n, axes, maxlevel: int):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.n = np.asarray(n, dtype=np.int64)
        self.h = (self.upper - self.lower) / self.n
        self.axes = np.asarray(axes, dtype=bool)
        self.maxlevel = int(maxlevel)
        self._base = int(self.n.max() << (self.maxlevel + 2)) + 1
        grid = np.meshgrid(*(np.arange(k) for k in self.n), indexing="ij")
        self.ijk = np.stack(grid, axis=-1).reshape(-1, 3).astype(np.int64)
        self.level = np.zeros(len(self.ijk), dtype=np.int64)
        self._keys = None

    # -- leaves -----------------------------------------------------------------------------

    def _key(self, level, ijk) -> np.ndarray:
        m = self._base
        return ((np.asarray(level) * m + ijk[:, 0]) * m + ijk[:, 1]) * m + ijk[:, 2]

    def _sorted_keys(self) -> np.ndarray:
        if self._keys is None:
            self._keys = np.sort(self._key(self.level, self.ijk))
        return self._keys

    def is_leaf(self, level, ijk) -> np.ndarray:
        """Whether the cells ``(level, ijk)`` (in the box) are leaves."""
        keys = self._sorted_keys()
        k = self._key(level, ijk)
        pos = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
        return keys[pos] == k

    def wrap(self, level, ijk) -> tuple[np.ndarray, np.ndarray]:
        """Cell indices wrapped along the periodic axes, and whether they lie in the box."""
        count = self.n[None, :] << np.asarray(level)[..., None]
        out = np.where(self.axes, np.mod(ijk, count), ijk)
        return out, np.all((out >= 0) & (out < count), axis=1)

    def sizes(self, level) -> np.ndarray:
        """The sides of cells of the levels (rows)."""
        return self.h[None, :] / (1 << np.asarray(level))[:, None]

    def points(self, level, ijk) -> np.ndarray:
        """The 27 points of the cells, ``(k, 27, 3)``."""
        cell = self.sizes(level)
        origin = self.lower + ijk * cell
        return origin[:, None, :] + LATTICE[None, :, :] * (0.5 * cell[:, None, :])

    def split(self, mask) -> np.ndarray:
        """Leaves replaced by their eight children; the indices of the children."""
        level, ijk = self.level[mask], self.ijk[mask]
        kids = (2 * ijk[:, None, :] + CHILDREN[None, :, :]).reshape(-1, 3)
        keep = ~mask
        self.level = np.concatenate([self.level[keep], np.repeat(level + 1, 8)])
        self.ijk = np.concatenate([self.ijk[keep], kids])
        self._keys = None
        return np.arange(len(self.level) - len(kids), len(self.level))

    # -- refinement -------------------------------------------------------------------------

    def refine(self, test) -> list[int]:
        """Split the leaves for which ``test(points)`` (of their 27 points ``(k, 27, 3)``,
        and their sides ``(k, 3)``) is true, then their children, down to ``maxlevel``;
        returns the number of leaves split at each level."""
        todo = np.arange(len(self.level))
        counts = []
        while todo.size:
            level = self.level[todo]
            split = np.asarray(test(self.points(level, self.ijk[todo]), self.sizes(level)))
            split &= level < self.maxlevel
            counts.append(int(split.sum()))
            if not split.any():
                break
            mask = np.zeros(len(self.level), dtype=bool)
            mask[todo[split]] = True
            todo = self.split(mask)
        return counts

    def balance(self) -> int:
        """Split leaves until any two that touch (across the periodic faces too) differ by
        one level at most; returns the number of splits."""
        total = 0
        while True:
            mask = np.zeros(len(self.level), dtype=bool)
            for lv in range(2, self.maxlevel + 1):
                sel = np.flatnonzero(self.level == lv)
                if not sel.size:
                    continue
                for d in NEIGHBOURS:
                    nb, inside = self.wrap(lv, self.ijk[sel] + d)
                    nb = nb[inside]
                    for up in range(2, lv + 1):  # the coarse leaves that hold the neighbours
                        anc = nb >> up
                        hit = self.is_leaf(lv - up, anc)
                        if hit.any():
                            mask |= np.isin(self._key(self.level, self.ijk),
                                            self._key(lv - up, anc[hit]))  # fmt: skip
            if not mask.any():
                return total
            total += int(mask.sum())
            self.split(mask)

    # -- tetrahedra -------------------------------------------------------------------------

    def tetrahedra(self):
        """The conforming tetrahedralization of the leaves (see the module): nodes,
        positive tetrahedra, the triangles of the faces of the box with their references
        (1-6: x0, x1, y0, y1, z0, z1), and the integer coordinates of the nodes (in quarters
        of the smallest cells)."""
        r = 1 << (self.maxlevel + 1)  # node units per root cell
        unit = (r >> self.level)[:, None]  # sides of the leaves, in node units
        total = self.n * r
        # the corners of the leaves (folded along the periodic axes): the nodes in the
        # middle of the sides of coarser faces
        corners = (self.ijk[:, None, :] + CHILDREN[None, :, :]) * unit[:, None, :]
        corner_keys = np.unique(self._node_key(self._fold(corners.reshape(-1, 3), total)))
        centres = self.ijk * unit + unit // 2
        faces = []
        plain = np.ones(len(self.level), dtype=bool)  # leaves of six plain faces: Kuhn
        for d in range(3):
            b, c = _others(d)
            for s in (0, 1):
                step = np.zeros(3, dtype=np.int64)
                step[d] = 2 * s - 1
                nb, inside = self.wrap(self.level, self.ijk + step)
                child = 2 * nb
                child[:, d] += 1 - s  # the children of the neighbour along the face
                finer = inside & (self.level < self.maxlevel)
                finer[finer] = self.is_leaf(self.level[finer] + 1, child[finer])
                origin = self.ijk * unit
                origin[:, d] += s * unit[:, 0]
                middles = self._middles(origin, unit, d, corner_keys, total)
                plain &= ~finer & ~middles.any(axis=1)
                faces.append((d, s, finer, origin, middles))
        tri_nodes, tet_apex, box_ref = [], [], []
        for d, s, finer, origin, middles in faces:
            b, c = _others(d)
            # the squares: whole faces (fanned when they have middles), or their quarters
            whole = np.flatnonzero(~finer)
            squares = [(whole, origin[whole], unit[whole, 0], middles[whole],
                        self.ijk[whole][:, [b, c]])]  # fmt: skip
            quarter = np.flatnonzero(finer)
            half = unit[quarter, 0] // 2
            for qb, qc in itertools.product(range(2), repeat=2):
                shift = np.zeros((len(quarter), 3), dtype=np.int64)
                shift[:, b], shift[:, c] = qb * half, qc * half
                index = 2 * self.ijk[quarter][:, [b, c]] + (qb, qc)
                squares.append((quarter, origin[quarter] + shift, half,
                                np.zeros((len(quarter), 4), dtype=bool), index))  # fmt: skip
            for leaves, o, u, mids, index in squares:
                if not len(leaves):
                    continue
                tri = _square_triangles(o, u, b, c, mids, index)
                k = tri.shape[1]
                plane = o[:, d]
                on_box = np.where((plane == 0) | (plane == total[d]), 1 + 2 * d + s, 0)
                keep = ~plain[leaves] | (on_box > 0)  # Kuhn leaves: their box faces only
                tri_nodes.append(tri[keep].reshape(-1, 3, 3))
                apex = np.where(plain[leaves][:, None], -1, centres[leaves])  # no cone
                tet_apex.append(np.repeat(apex[keep], k, axis=0))
                box_ref.append(np.repeat(on_box[keep], k))
        tri_nodes = np.concatenate(tri_nodes)
        tet_apex = np.concatenate(tet_apex)
        box_ref = np.concatenate(box_ref)
        valid = np.all(tri_nodes[:, :, 0] >= 0, axis=1)  # the empty slots of the fans
        tri_nodes, tet_apex, box_ref = tri_nodes[valid], tet_apex[valid], box_ref[valid]
        kuhn = self._kuhn(np.flatnonzero(plain), unit)
        cone = tet_apex[:, 0] >= 0
        coords = np.concatenate([tri_nodes.reshape(-1, 3), tet_apex[cone], kuhn.reshape(-1, 3)])
        keys = self._node_key(coords)
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        ints = coords[first]
        nodes = self.lower + ints * (self.h / r)
        for d in range(3):  # the faces of the box exactly
            nodes[ints[:, d] == total[d], d] = self.upper[d]
        m = len(tri_nodes)
        tri = inverse[: 3 * m].reshape(m, 3)
        n_cone = int(cone.sum())
        apex = inverse[3 * m : 3 * m + n_cone]
        tets = np.concatenate([np.column_stack([tri[cone], apex]),
                               inverse[3 * m + n_cone :].reshape(-1, 4)])  # fmt: skip
        flip = tet_volumes(nodes, tets) < 0
        tets[flip] = tets[flip][:, [0, 2, 1, 3]]
        boundary = box_ref > 0
        tri, refs = tri[boundary], box_ref[boundary]
        # outwards
        axis, side = (refs - 1) // 2, (refs - 1) % 2
        a, b, c = nodes[tri[:, 0]], nodes[tri[:, 1]], nodes[tri[:, 2]]
        normal = np.cross(b - a, c - a)[np.arange(len(tri)), axis]
        flip = (normal > 0) != (side == 1)
        tri[flip] = tri[flip][:, [0, 2, 1]]
        return nodes, tets, tri, refs, ints

    def _middles(self, origin, unit, d, corner_keys, total) -> np.ndarray:
        """Whether the middles of the four sides of the faces (lower corners ``origin``,
        sides ``unit``, normal to ``d``) are nodes (corners of finer leaves)."""
        b, c = _others(d)
        h = unit[:, 0] // 2
        out = np.zeros((len(origin), 4), dtype=bool)
        offsets = ((h, 0), (unit[:, 0], h), (h, unit[:, 0]), (0, h))
        for q, (ob, oc) in enumerate(offsets):
            m = origin.copy()
            m[:, b] += ob
            m[:, c] += oc
            key = self._node_key(self._fold(m, total))
            pos = np.minimum(np.searchsorted(corner_keys, key), len(corner_keys) - 1)
            out[:, q] = (corner_keys[pos] == key) & (unit[:, 0] >= 4)
        return out

    def _kuhn(self, leaves, unit) -> np.ndarray:
        """The six tetrahedra (node coordinates, ``(6 k, 4, 3)``) of the Kuhn triangulation
        of leaves: the paths along the axes from a corner to the opposite one, reflected
        where the index of the leaf is odd (its faces split along the diagonals of
        :func:`_square_triangles`)."""
        if not len(leaves):
            return np.zeros((0, 4, 3), dtype=np.int64)
        odd = self.ijk[leaves] % 2 == 1
        origin = self.ijk[leaves] * unit[leaves]
        out = []
        for perm in itertools.permutations(range(3)):
            corner = np.zeros(3, dtype=np.int64)
            path = [corner.copy()]
            for axis in perm:
                corner[axis] += 1
                path.append(corner.copy())
            out.append(np.stack([origin + np.where(odd, 1 - p, p) * unit[leaves] for p in path],
                                axis=1))  # fmt: skip
        return np.concatenate(out)

    def _node_key(self, coords) -> np.ndarray:
        m = int((self.n.max() << (self.maxlevel + 1)) + 2)
        c = np.asarray(coords, dtype=np.int64)
        return (c[:, 0] * m + c[:, 1]) * m + c[:, 2]

    def _fold(self, coords, total) -> np.ndarray:
        return np.where(self.axes, np.mod(coords, total), coords)


def _square_triangles(origin, u, b, c, middles, index) -> np.ndarray:
    """Triangles ``(k, 8, 3, 3)`` (node coordinates; empty slots -1) of squares of lower
    corners ``origin`` and sides ``u`` in the plane of the axes ``b`` and ``c``: fans from
    their centres over their corners and the ``middles`` of their sides that are nodes, or
    when none is, two triangles split by the diagonal of the Kuhn triangulation of the cells
    on either side (through the corner of parities ``index % 2``, the indices of the square
    in its plane, among the cells of its size): the same seen from both sides."""
    k = len(origin)
    u = np.asarray(u, dtype=np.int64)
    eb = np.zeros(3, dtype=np.int64)
    ec = np.zeros(3, dtype=np.int64)
    eb[b], ec[c] = 1, 1
    uu = u[:, None]
    corners = [origin, origin + uu * eb, origin + uu * (eb + ec), origin + uu * ec]
    h = uu // 2
    mids = [origin + h * eb, origin + uu * eb + h * ec, origin + h * eb + uu * ec, origin + h * ec]
    centre = origin + h * (eb + ec)
    out = np.full((k, 8, 3, 3), -1, dtype=np.int64)
    fan = middles.any(axis=1)
    for q in range(4):
        has = middles[:, q]
        p0, p1, m = corners[q], corners[(q + 1) % 4], mids[q]
        out[fan, 2 * q, 0] = centre[fan]
        out[fan, 2 * q, 1] = p0[fan]
        out[fan, 2 * q, 2] = np.where(has[fan, None], m[fan], p1[fan])
        both = fan & has
        out[both, 2 * q + 1, 0] = centre[both]
        out[both, 2 * q + 1, 1] = m[both]
        out[both, 2 * q + 1, 2] = p1[both]
    flat = ~fan
    same = (index[:, 0] % 2) == (index[:, 1] % 2)  # the diagonal (0, 0)-(1, 1)
    c0, c1, c2, c3 = (x[flat] for x in corners)
    sm = same[flat][:, None]
    out[flat, 0] = np.stack([c0, c1, np.where(sm, c2, c3)], axis=1)
    out[flat, 1] = np.stack([np.where(sm, c0, c1), c2, c3], axis=1)
    return out
