"""Regularization of cell complexes: removal of the features smaller than a length.

A tessellation has edges, facets and pieces of every size; a mesh must resolve them with
elements of their size, or with flat elements. Following the idea of Quey, Dawson and
Barbe (2011), the features smaller than ``delta`` are removed before meshing:

* *caps* — pieces that rise less than ``delta`` above a face of the box (cells that
  barely cross it) — are flattened onto that face: their other facets become faces of
  the box for the neighbouring pieces. A cap that overhangs its facet on the face cannot
  be flattened this way; it joins the cell of the neighbour with which it shares its
  largest facet (its volume changes cell, the geometry is kept);
* short edges are contracted to a point, the shortest first; when contracting an edge
  alone would leave an invalid piece (a tiny facet with several short edges), its nearest
  neighbours are contracted with it.

Each change is kept only if every piece it touches remains a valid star polyhedron (the
cones from its centroid over its facets have positive volumes) touching the faces of the
box by facets, not along lines. The vertices on the faces of the box stay on them. The
faces of a periodic complex remain exact translates of each other: a vertex of a face
moves and merges together with its periodic images, and the facet of a flattened cap's
cell on the opposite face is split as the cap's facet was. The facets become slightly
non-planar.

The whole cells of a periodic tessellation (``CellComplex.whole``) have no caps: only their
short edges are contracted, every vertex moving and merging together with its
translates, so that the facets bounding their union remain pairs of translates.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from morphogen.mesh.polyhedra import CellComplex


class _Complex:
    """Mutable copy of a cell complex."""

    def __init__(self, cx: CellComplex):
        self.cx = cx
        self.V = cx.vertices.copy()
        self.facets = [list(map(int, f)) for f in cx.facets]
        self.fp = [list(map(int, p)) for p in cx.facet_pieces]
        self.fbox = [int(b) for b in cx.facet_box]
        self.pfac = [set(map(int, pf)) for pf in cx.piece_facets]
        self.alive_f = [True] * len(self.facets)
        self.alive_p = [True] * len(self.pfac)
        self.cell = np.array(cx.piece_cell, dtype=np.int64)
        self.lower, self.upper = cx.lower, cx.upper
        self.size = cx.upper - cx.lower
        self.whole = bool(cx.whole)
        self.tol = 1e-9 * float(self.size.max())
        self.vfac: list[set] = [set() for _ in range(len(self.V))]
        for f, loop in enumerate(self.facets):
            for v in loop:
                self.vfac[v].add(f)

    # -- geometry ------------------------------------------------------------------------

    def planes(self, v: int) -> set:
        """Faces of the box (axis, side) containing vertex ``v`` (none for whole cells)."""
        out = set()
        if self.whole:
            return out
        for d in range(3):
            if abs(self.V[v, d] - self.lower[d]) <= self.tol:
                out.add((d, 0))
            elif abs(self.V[v, d] - self.upper[d]) <= self.tol:
                out.add((d, 1))
        return out

    def oriented(self, f: int, p: int) -> list:
        return self.facets[f] if self.fp[f][0] == p else self.facets[f][::-1]

    def piece_vertices(self, p: int) -> list:
        return sorted({v for f in self.pfac[p] for v in self.facets[f]})

    def piece_ok(self, p: int, min_volume: float) -> bool:
        """Closed star polyhedron: every edge of its facets is shared by two facets in
        opposite directions, and the cones from its centroid over the facet fans have
        positive volumes."""
        if len(self.pfac[p]) < 4:
            return False
        edges: dict = {}
        for f in self.pfac[p]:
            loop = self.oriented(f, p)
            for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
                edges[(a, b)] = edges.get((a, b), 0) + 1
        for (a, b), count in edges.items():
            if count != 1 or edges.get((b, a), 0) != 1:
                return False
        # an edge on a face of the box belongs to a facet of the piece on that face (a
        # piece touching the box along a line only is not a valid subdomain)
        box_edges = set()
        for f in self.pfac[p]:
            if self.fbox[f] > 0:
                loop = self.facets[f]
                box_edges.update(zip(loop, loop[1:] + loop[:1], strict=True))
                box_edges.update(zip(loop[1:] + loop[:1], loop, strict=True))
        for f in self.pfac[p]:
            if self.fbox[f] > 0:
                continue
            loop = self.facets[f]
            for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
                if (a, b) not in box_edges and self.planes(a) & self.planes(b):
                    return False
        verts = self.piece_vertices(p)
        g = self.V[verts].mean(axis=0)
        total = 0.0
        for f in self.pfac[p]:
            loop = self.oriented(f, p)
            if len(set(loop)) < 3:
                return False
            pts = self.V[loop]
            c = pts.mean(axis=0)
            nxt = np.roll(pts, -1, axis=0)
            vol = np.cross(pts - c, nxt - c) @ (g - c) / 6.0
            # outward triangles: the centroid lies behind them
            if np.any(vol >= -min_volume):
                return False
            total -= vol.sum()
        return total > 0

    def facet_area(self, f: int) -> float:
        p = self.V[self.facets[f]]
        c = p.mean(axis=0)
        return 0.5 * float(np.linalg.norm(np.cross(p - c, np.roll(p, -1, axis=0) - c).sum(axis=0)))

    # -- edits ---------------------------------------------------------------------------

    def snapshot(self, facets, pieces, verts):
        verts = set(verts)
        for f in facets:
            verts.update(self.facets[f])
        return (
            {
                f: (list(self.facets[f]), self.alive_f[f], list(self.fp[f]), self.fbox[f])
                for f in facets
            },
            {p: (set(self.pfac[p]), self.alive_p[p]) for p in pieces},
            {v: (self.V[v].copy(), set(self.vfac[v])) for v in verts},
        )

    def restore(self, snap):
        fs, ps, vs = snap
        for f, (loop, alive, fp, fb) in fs.items():
            self.facets[f], self.alive_f[f], self.fp[f], self.fbox[f] = loop, alive, fp, fb
        for p, (pf, alive) in ps.items():
            self.pfac[p], self.alive_p[p] = pf, alive
        for v, (x, vf) in vs.items():
            self.V[v], self.vfac[v] = x, vf

    def clean_facet(self, f: int) -> None:
        """Remove repeated vertices; kill the facet if fewer than three remain."""
        loop = self.facets[f]
        out = [v for i, v in enumerate(loop) if v != loop[i - 1]] if len(loop) > 1 else loop
        if len(out) > 1 and out[0] == out[-1]:
            out = out[:-1]
        self.facets[f] = out
        if len(set(out)) < 3 or len(set(out)) != len(out):
            self.kill_facet(f)

    def kill_facet(self, f: int) -> None:
        if not self.alive_f[f]:
            return
        self.alive_f[f] = False
        for p in self.fp[f]:
            if p >= 0:
                self.pfac[p].discard(f)
        for v in set(self.facets[f]):
            self.vfac[v].discard(f)

    def add_vertex(self, x) -> int:
        self.V = np.vstack([self.V, np.asarray(x, dtype=float)[None, :]])
        self.vfac.append(set())
        return len(self.V) - 1

    def add_facet(self, loop: list, pieces: list, box: int) -> int:
        f = len(self.facets)
        self.facets.append(list(loop))
        self.fp.append(list(pieces))
        self.fbox.append(int(box))
        self.alive_f.append(True)
        for p in pieces:
            if p >= 0:
                self.pfac[p].add(f)
        for v in loop:
            self.vfac[v].add(f)
        return f

    def discard_new(self, n_vertices: int, n_facets: int) -> None:
        """Forget the vertices and facets added since the counts given (after restore)."""
        for f in range(n_facets, len(self.facets)):
            self.alive_f[f] = False
        for v in range(n_vertices, len(self.V)):
            self.vfac[v] = set()

    def merge(self, a: int, b: int, x) -> None:
        """Vertex ``b`` disappears into ``a``, moved to ``x``."""
        self.V[a] = x
        for f in list(self.vfac[b]):
            self.facets[f] = [a if v == b else v for v in self.facets[f]]
            self.vfac[a].add(f)
        self.vfac[b] = set()
        for f in list(self.vfac[a]):
            if self.alive_f[f]:
                self.clean_facet(f)


def _translates(cx: _Complex) -> dict:
    """Translates of the vertices of whole cells: ``v -> {offset: copy}``. They are exact
    but for rounding, while distinct vertices are farther apart than the tolerance of
    their assembly (an edge may be shorter than a micrometre of a unit box)."""
    from morphogen.mesh.polyhedra import torus_classes

    labels, shift = torus_classes(cx.V, cx.lower, cx.size, 0.1 * cx.tol)
    members: dict = {}
    for v, lab in enumerate(labels.tolist()):
        members.setdefault(lab, []).append(v)
    out: dict = {}
    for group in members.values():
        if len(group) < 2:
            continue
        for v in group:
            out[v] = {tuple(int(k) for k in shift[u] - shift[v]): u for u in group if u != v}
    return out


def _images(cx: _Complex) -> dict:
    """Periodic images of the vertices of the faces: ``v -> {offset: image}``."""
    if cx.whole:
        return _translates(cx)
    on = [v for v in range(len(cx.V)) if cx.planes(v)]
    if not on:
        return {}
    tree = cKDTree(cx.V[on])
    out: dict = {}
    for v in on:
        for d, side in cx.planes(v):
            shift = np.zeros(3)
            shift[d] = cx.size[d] if side == 0 else -cx.size[d]
            dist, j = tree.query(cx.V[v] + shift)
            if dist <= 1e3 * cx.tol:
                key = tuple(int(s) for s in np.sign(shift))
                out.setdefault(v, {})[key] = on[j]
    # images across several faces (box edges and corners): compose the offsets
    changed = True
    while changed:
        changed = False
        for v, imgs in list(out.items()):
            for k1, w in list(imgs.items()):
                for k2, u in out.get(w, {}).items():
                    k = tuple(np.add(k1, k2))
                    if any(k) and k not in imgs and u != v and max(map(abs, k)) <= 1:
                        imgs[k] = u
                        changed = True
    return out


def _touching(cx: _Complex, verts) -> set:
    pieces = set()
    for v in verts:
        for f in cx.vfac[v]:
            pieces.update(p for p in cx.fp[f] if p >= 0)
    return pieces


def _opposite_facet(cx: _Complex, f: int, off: tuple, images: dict) -> int | None:
    """The facet of the opposite face whose vertices are the images of those of ``f``."""
    imgs = [images.get(v, {}).get(off) for v in cx.facets[f]]
    if any(i is None for i in imgs):
        return None
    d = int(np.flatnonzero(off)[0])
    ref = 2 * d + 1 + (1 if off[d] > 0 else 0)
    cand = [
        g
        for g in set.intersection(*(cx.vfac[i] for i in imgs))
        if cx.alive_f[g] and cx.fbox[g] == ref and set(cx.facets[g]) == set(imgs)
    ]
    return cand[0] if len(cand) == 1 else None


def _relabel(cx: _Complex, p: int) -> bool:
    """Give a piece to the cell of the neighbour with which it shares its largest facet."""
    best, cell = 0.0, None
    for f in cx.pfac[p]:
        q = cx.fp[f][1] if cx.fp[f][0] == p else cx.fp[f][0]
        if q >= 0 and cx.cell[q] != cx.cell[p]:
            area = cx.facet_area(f)
            if area > best:
                best, cell = area, cx.cell[q]
    if cell is None:
        return False
    cx.cell[p] = cell
    return True


def _flatten_caps(cx: _Complex, delta: float, min_volume: float, images: dict) -> tuple:
    """Flatten the pieces that rise less than ``delta`` above a face of the box. In a
    periodic complex, the facet of the same cell on the opposite face is split alike (new
    vertices, images of the flattened ones), so that opposite faces remain translates.
    The caps that cannot be flattened (overhanging the face) join a neighbouring cell.
    Returns the numbers of flattened and relabelled caps."""
    done = relabelled = 0
    for p in range(len(cx.pfac)):
        if not cx.alive_p[p]:
            continue
        box_facets = [f for f in cx.pfac[p] if cx.fbox[f] > 0]
        verts = cx.piece_vertices(p)
        best = None
        for ref in {cx.fbox[f] for f in box_facets}:
            d, side = (ref - 1) // 2, (ref - 1) % 2
            plane = cx.lower[d] if side == 0 else cx.upper[d]
            height = float(np.max(np.abs(cx.V[verts, d] - plane)))
            if height < delta and (best is None or height < best[0]):
                best = (height, d, side, plane, ref)
        if best is None:
            continue
        _, d, side, plane, ref = best
        moved = [v for v in verts if abs(cx.V[v, d] - plane) > cx.tol]
        opposite, off = None, None
        if images:
            # a piece on one face only, whose other vertices lie inside the box
            if len(box_facets) == 1 and not any(cx.planes(v) for v in moved):
                off = tuple(int(k == d) * (1 if side == 0 else -1) for k in range(3))
                opposite = _opposite_facet(cx, box_facets[0], off, images)
            if opposite is None:
                relabelled += _relabel(cx, p)
                continue
        facets = set(cx.pfac[p])
        pieces = _touching(cx, moved) | {p}
        if opposite is not None:
            pieces.add(cx.fp[opposite][0])
        for q in pieces:
            facets |= cx.pfac[q]
        snap = cx.snapshot(facets, pieces, moved)
        n_vertices, n_facets = len(cx.V), len(cx.facets)
        for v in moved:
            cx.V[v, d] = plane
        # the piece disappears: its facets on the face go, the others join the face
        joined = []
        for f in list(cx.pfac[p]):
            if cx.fbox[f] > 0:
                cx.kill_facet(f)
                continue
            other = cx.fp[f][1] if cx.fp[f][0] == p else cx.fp[f][0]
            if cx.fp[f][0] == p:  # the loop pointed away from p, into the other piece
                cx.facets[f] = cx.facets[f][::-1]
            cx.fp[f] = [other, -1]
            cx.fbox[f] = ref
            if cx.facet_area(f) <= 1e-12 * float(cx.size.max()) ** 2:
                cx.kill_facet(f)
            else:
                joined.append(f)
        cx.pfac[p] = set()
        cx.alive_p[p] = False
        new_images = []
        if opposite is not None:
            # the facet of the cell on the opposite face, split as the joined facets
            owner, ref_b = cx.fp[opposite][0], cx.fbox[opposite]
            cx.kill_facet(opposite)
            shift = np.zeros(3)
            shift[d] = off[d] * cx.size[d]
            back = tuple(-k for k in off)
            for v in sorted({v for f in joined for v in cx.facets[f]} & set(moved)):
                u = cx.add_vertex(cx.V[v] + shift)
                images[v], images[u] = {off: u}, {back: v}
                new_images += [v, u]
            for f in joined:  # the outward normal of the opposite face is reversed
                cx.add_facet([images[v][off] for v in cx.facets[f]][::-1], [owner, -1], ref_b)
        if all(cx.piece_ok(q, min_volume) for q in pieces if q != p and cx.alive_p[q]):
            done += 1
        else:
            cx.restore(snap)
            cx.discard_new(n_vertices, n_facets)
            for v in new_images:
                images.pop(v, None)
            relabelled += _relabel(cx, p)
    return done, relabelled


def _contract(cx: _Complex, cluster: set, min_volume: float, images: dict) -> list | None:
    """Contract a set of vertices to one point, with their periodic images; the change is
    kept if every piece it touches remains valid. Returns the vertices kept (the point
    and its images), or None."""
    planes = {v: cx.planes(v) for v in cluster}
    union = set().union(*planes.values())
    if len({d for d, _ in union}) < len(union):
        return None  # opposite faces of the box
    most = max(len(s) for s in planes.values())
    # the most constrained vertex, with the most periodic images
    keep = min(cluster, key=lambda v: (-len(planes[v]), -len(images.get(v, {})), v))
    if not all(s <= planes[keep] for s in planes.values()):
        return None  # on different faces of the box
    # the point: the mean of the most constrained vertices, on their faces
    target = cx.V[[v for v in cluster if len(planes[v]) == most]].mean(axis=0)
    for d, side in union:
        target[d] = cx.lower[d] if side == 0 else cx.upper[d]
    groups = [(keep, [v for v in cluster if v != keep])]
    if images and (union or cx.whole):
        # a vertex of a face (of whole cells: any vertex) moves with its images, and the
        # images merge alike
        ik = images.get(keep, {})
        on_faces = [images.get(v, {}) for v in cluster if v != keep and (planes[v] or cx.whole)]
        if not all(set(iv) <= set(ik) for iv in on_faces):
            return None  # the images differ: the faces differ here
        for off, image in ik.items():
            groups.append((image, [iv[off] for iv in on_faces if off in iv]))
    verts = {v for keep_v, gone in groups for v in [keep_v, *gone]}
    pieces = _touching(cx, verts)
    facets = set()
    for q in pieces:
        facets |= cx.pfac[q]
    for v in verts:
        facets |= cx.vfac[v]
    snap = cx.snapshot(facets, pieces, verts)
    origin = cx.V[keep].copy()
    for keep_v, gone in groups:
        x = target + (cx.V[keep_v] - origin)  # the same move for the images
        cx.V[keep_v] = x
        for v in gone:
            cx.merge(keep_v, v, x)
    if all(cx.piece_ok(q, min_volume) for q in pieces if cx.alive_p[q] and cx.pfac[q]):
        return [keep_v for keep_v, _ in groups]
    cx.restore(snap)
    return None


def _neighbours(cx: _Complex, cluster: set) -> set:
    out = set()
    for v in cluster:
        for f in cx.vfac[v]:
            loop = cx.facets[f]
            i = loop.index(v)
            out.update((loop[i - 1], loop[(i + 1) % len(loop)]))
    return out - cluster


def _collapse_edges(
    cx: _Complex, delta: float, min_volume: float, images: dict, max_cluster: int = 6
) -> int:
    """Contract the edges shorter than ``delta`` (with their periodic images), the
    shortest first. When contracting an edge alone would leave an invalid piece (several
    tiny edges around a small facet), its nearest neighbours are contracted with it, up to
    ``max_cluster`` vertices within ``delta`` of their centre."""
    heap = []

    def push(a, b):
        length = float(np.linalg.norm(cx.V[a] - cx.V[b]))
        if length < delta:
            heapq.heappush(heap, (length, min(a, b), max(a, b)))

    for f, loop in enumerate(cx.facets):
        if cx.alive_f[f]:
            for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
                push(a, b)
    done = 0
    while heap:
        _, a, b = heapq.heappop(heap)
        if a == b or not cx.vfac[a] or not cx.vfac[b]:
            continue
        if not any(b in cx.facets[f] for f in cx.vfac[a]):
            continue  # no longer an edge
        cluster = {a, b}
        kept = _contract(cx, cluster, min_volume, images)
        while kept is None and len(cluster) < max_cluster:
            centre = cx.V[list(cluster)].mean(axis=0)
            near = _neighbours(cx, cluster)
            if not near:
                break
            w = min(near, key=lambda v: float(np.linalg.norm(cx.V[v] - centre)))
            if np.linalg.norm(cx.V[w] - centre) >= delta:
                break
            cluster.add(w)
            kept = _contract(cx, cluster, min_volume, images)
        if kept is None:
            continue
        done += len(cluster) - 1
        for keep_v in kept:
            for f in list(cx.vfac[keep_v]):
                loop = cx.facets[f]
                i = loop.index(keep_v)
                push(keep_v, loop[i - 1])
                push(keep_v, loop[(i + 1) % len(loop)])
    return done


def _fix_t_junctions(cx: _Complex) -> int:
    """Insert into every facet edge the vertices lying on it, so that neighbouring facets
    are subdivided alike."""
    used = sorted({v for f, loop in enumerate(cx.facets) if cx.alive_f[f] for v in loop})
    if not used:
        return 0
    tree = cKDTree(cx.V[used])
    added = 0
    changed = True
    while changed:
        changed = False
        for f, loop in enumerate(cx.facets):
            if not cx.alive_f[f]:
                continue
            out = []
            for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
                out.append(a)
                pa, pb = cx.V[a], cx.V[b]
                ab = pb - pa
                length2 = float(ab @ ab)
                near = tree.query_ball_point(0.5 * (pa + pb), 0.5 * np.sqrt(length2) + cx.tol)
                inside = []
                for k in near:
                    w = used[k]
                    if w in (a, b):
                        continue
                    t = float((cx.V[w] - pa) @ ab) / length2
                    if 0.0 < t < 1.0 and np.linalg.norm(pa + t * ab - cx.V[w]) <= 1e3 * cx.tol:
                        inside.append((t, w))
                for _, w in sorted(inside):
                    if w not in out and w not in loop:
                        out.append(w)
                        cx.vfac[w].add(f)
                        added += 1
                        changed = True
            cx.facets[f] = out
    return added


def regularize(cx: CellComplex, delta: float) -> CellComplex:
    """Complex without the caps and edges smaller than ``delta`` (see the module)."""
    if not delta > 0:
        return cx
    m = _Complex(cx)
    min_volume = 1e-15 * float(np.prod(m.size))
    images = _images(m) if cx.periodic else {}
    caps, relabelled = (0, 0) if m.whole else _flatten_caps(m, delta, min_volume, images)
    edges = _collapse_edges(m, delta, min_volume, images)
    tj = _fix_t_junctions(m)
    # compact
    alive_f = [f for f in range(len(m.facets)) if m.alive_f[f]]
    alive_p = [p for p in range(len(m.pfac)) if m.alive_p[p] and m.pfac[p]]
    fmap = {f: k for k, f in enumerate(alive_f)}
    pmap = {p: k for k, p in enumerate(alive_p)}
    used = sorted({v for f in alive_f for v in m.facets[f]})
    vmap = {v: k for k, v in enumerate(used)}
    return CellComplex(
        vertices=m.V[used],
        facets=[np.array([vmap[v] for v in m.facets[f]], dtype=np.int64) for f in alive_f],
        facet_pieces=np.array(
            [[pmap[p] if p >= 0 else -1 for p in m.fp[f]] for f in alive_f], dtype=np.int64
        ),
        facet_box=np.array([m.fbox[f] for f in alive_f], dtype=np.int64),
        piece_cell=m.cell[alive_p],
        piece_facets=[
            np.array(sorted(fmap[f] for f in m.pfac[p]), dtype=np.int64) for p in alive_p
        ],
        lower=cx.lower,
        upper=cx.upper,
        periodic=cx.periodic,
        whole=cx.whole,
        metadata={
            **cx.metadata,
            "regularization": float(delta),
            "flattened_caps": caps,
            "relabelled_caps": relabelled,
            "contracted_edges": edges,
            "t_junctions": tj,
        },
    )
