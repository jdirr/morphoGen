"""The smooth pieces of the phases of structures made of spheres and segments.

The implicit function of a union of primitives is the smallest of their distances,
``min_i d_i``: smooth where one primitive is the nearest, folded where two are as near.
Its zero level set, the boundary of the union, has a *sharp edge* along the line where
the surfaces of two overlapping primitives meet (the groove between two spheres, the
line where two fibres cross). A function sampled on a grid cannot hold such an edge: a
mesher that cuts the box by the zero of ``min_i d_i`` rounds it over a cell.

Each primitive alone is smooth. The primitives of a union are therefore grouped into
*pieces* such that two primitives that overlap are never in the same piece — a colouring
of the graph of their overlaps (:func:`colours`), periodic images included. The distance
to a piece is smooth near its surface, the phase is the union of its pieces, and a mesher
that cuts the box by the surface of each piece in turn gets every sharp edge as the line
where two of its cuts meet (:func:`morphogen.mesh.conforming.conforming_mesh`). A coating
of thickness ``t`` is the union of the same primitives with their radii increased by
``t``, the core of a shell that of the primitives with their radii decreased: other
pieces, coloured again since thicker primitives overlap more.

A structure gives its pieces by :meth:`~morphogen.models.base.Microstructure.
implicit_pieces`, in the order of the cuts: a point belongs to the phase of the first
piece that holds it, or to the phase of what is left.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from morphogen.kernels.rsa import segment_distance2
from morphogen.primitives import Segments, Spheres


@dataclass
class Piece:
    """A smooth piece of a phase: primitives that do not overlap (a union), and the
    ``phase`` of the points that it holds."""

    phase: int
    union: object  # a PrimitiveUnion

    def at(self, points, band: float) -> np.ndarray:
        """The signed distance to the piece (negative inside) at ``points``, clipped to
        ``[-band, band]``."""
        return self.union.signed_distance(points, float(band))


@njit(cache=True)
def _penetrations(start, ray, radii, first, second, out):
    """The depth by which the capsules ``first[k]`` and ``second[k]`` enter each other:
    the sum of their radii less the distance of their axes."""
    for k in range(first.shape[0]):
        i, j = first[k], second[k]
        distance = np.sqrt(segment_distance2(start[i], ray[i], start[j], ray[j]))
        out[k] = radii[i] + radii[j] - distance


def _capsules(spheres: Spheres, segments: Segments, offset: float):
    """The primitives as capsules (spheres first): starts, rays to their ends, radii
    increased by ``offset``."""
    start = np.concatenate([spheres.centers, segments.start])
    ray = np.concatenate([np.zeros_like(spheres.centers), segments.end - segments.start])
    radii = np.concatenate([spheres.radii, segments.radii]) + float(offset)
    return np.ascontiguousarray(start), np.ascontiguousarray(ray), np.ascontiguousarray(radii)


def overlaps(union, offset: float = 0.0, depth: float = 0.0) -> np.ndarray:
    """The pairs of primitives of a union (spheres first, then segments) that enter each
    other by more than ``depth``, their radii increased by ``offset``; through the faces
    of a periodic box as well. Rows ``(i, j)`` with ``i < j``."""
    start, ray, radii = _capsules(union.spheres, union.segments, offset)
    n = len(radii)
    alive = radii > 0.0
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    middle = start + 0.5 * ray
    reach = 0.5 * np.linalg.norm(ray, axis=1) + np.maximum(radii, 0.0)
    # the primitives and their images around the box, with the primitive of each
    index = [np.arange(n)]
    shifts = [np.zeros((n, 3))]
    if union.domain.periodic:
        lengths = np.asarray(union.domain.lengths, dtype=float)
        lower = np.asarray(union.domain.lower, dtype=float) - 2.0 * reach.max()
        upper = np.asarray(union.domain.upper, dtype=float) + 2.0 * reach.max()
        most = int(np.ceil(4.0 * reach.max() / lengths.min())) + 1
        for shift in np.ndindex(*(2 * most + 1,) * 3):
            move = (np.array(shift) - most) * lengths
            if not move.any():
                continue
            moved = middle + move
            near = np.all((moved + reach[:, None] >= lower) & (moved - reach[:, None] <= upper), 1)
            index.append(np.flatnonzero(near))
            shifts.append(np.broadcast_to(move, (int(near.sum()), 3)))
    index = np.concatenate(index)
    shifts = np.concatenate(shifts)
    tree = cKDTree(middle[index] + shifts)
    found = tree.query_ball_point(middle, reach + reach.max())
    first = np.repeat(np.arange(n), [len(f) for f in found])
    second = np.concatenate([np.asarray(f, dtype=np.int64) for f in found])
    keep = (index[second] != first) & alive[first] & alive[index[second]]
    keep &= np.linalg.norm(middle[first] - tree.data[second], axis=1) < (
        reach[first] + reach[index[second]]
    )
    first, second = first[keep], second[keep]
    if not len(first):
        return np.zeros((0, 2), dtype=np.int64)
    # the capsules and their images in one array, for the kernel
    all_start = np.ascontiguousarray(start[index] + shifts)
    all_ray = np.ascontiguousarray(ray[index])
    all_radii = np.ascontiguousarray(radii[index])
    deep = np.empty(len(first))
    _penetrations(all_start, all_ray, all_radii, first, second, deep)
    pairs = np.stack([first, index[second]], axis=1)[deep > float(depth)]
    return np.unique(np.sort(pairs, axis=1), axis=0)


def colours(n: int, pairs: np.ndarray) -> np.ndarray:
    """A colour for each of ``n`` vertices of a graph such that the two vertices of each
    of its edges ``pairs`` differ: the vertices of largest degree first, each takes the
    smallest colour that its neighbours do not have (Welsh & Powell 1967)."""
    colour = np.full(n, -1, dtype=np.int64)
    neighbours: list[list[int]] = [[] for _ in range(n)]
    for i, j in np.asarray(pairs, dtype=np.int64):
        neighbours[i].append(int(j))
        neighbours[j].append(int(i))
    order = sorted(range(n), key=lambda k: (-len(neighbours[k]), k))
    for k in order:
        taken = {colour[q] for q in neighbours[k]}
        c = 0
        while c in taken:
            c += 1
        colour[k] = c
    return colour


def union_pieces(union, offset: float = 0.0, depth: float = 0.0) -> list:
    """The pieces of a union of spheres and segments whose radii are increased by
    ``offset`` (decreased, if negative: the primitives that vanish are left out): unions
    of primitives that do not enter each other by more than ``depth``, as few as the
    colouring finds."""
    from morphogen.models.union import PrimitiveUnion

    spheres, segments = union.spheres, union.segments
    n_s = len(spheres)
    radii = np.concatenate([spheres.radii, segments.radii]) + float(offset)
    colour = colours(len(radii), overlaps(union, offset, depth))
    out = []
    for c in range(int(colour.max(initial=-1)) + 1):
        mask = (colour == c) & (radii > 0.0)
        if not mask.any():
            continue
        s, g = mask[:n_s], mask[n_s:]
        out.append(
            PrimitiveUnion(
                union.domain,
                Spheres(spheres.centers[s], spheres.radii[s] + offset),
                Segments(segments.start[g], segments.end[g], segments.radii[g] + offset,
                         segments.round_caps),
            )
        )  # fmt: skip
    return out


__all__ = ["Piece", "colours", "overlaps", "union_pieces"]
