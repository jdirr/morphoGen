"""Connected components and percolation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from morphogen.analysis.basic import as_array
from morphogen.image import VoxelImage


@dataclass
class Components:
    """Connected components of one phase.

    ``percolating[axis]`` tells whether a component joins the two faces normal to
    ``axis`` (inside the box, i.e. without using periodicity); ``spanning_fraction`` is
    the fraction of the phase volume in components joining at least one pair of faces.
    For periodic images, components touching across faces are merged before counting.
    """

    n_components: int
    sizes: np.ndarray
    percolating: dict[str, bool]
    spanning_fraction: float
    labels: np.ndarray | None = field(default=None, repr=False)


def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndimage.generate_binary_structure(3, 1)
    if connectivity == 18:
        return ndimage.generate_binary_structure(3, 2)
    if connectivity == 26:
        return ndimage.generate_binary_structure(3, 3)
    raise ValueError("connectivity: 6, 18 or 26")


def _merge_periodic(labels: np.ndarray, n: int) -> np.ndarray:
    """Map label -> representative after joining components across opposite faces."""
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for axis in range(3):
        a = np.take(labels, 0, axis=axis).ravel()
        b = np.take(labels, -1, axis=axis).ravel()
        mask = (a > 0) & (b > 0)
        for u, v in np.unique(np.stack([a[mask], b[mask]], axis=1), axis=0):
            ru, rv = find(int(u)), find(int(v))
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
    return np.array([find(i) for i in range(n + 1)])


def connected_components(
    image, phase: int = 1, *, connectivity: int = 6, periodic=None, return_labels=False
) -> Components:
    """The connected components of a phase of an image (``connectivity``: 6, 18 or 26 neighbours)
    and whether they percolate along each axis, through the faces of a periodic image
    (``periodic``, by default that of the image); their labels with ``return_labels``."""
    data = as_array(image)
    if periodic is None:
        periodic = image.periodic if isinstance(image, VoxelImage) else False
    mask = data == phase
    labels, n = ndimage.label(mask, structure=_structure(connectivity))
    percolating, spanning = {}, np.zeros(n + 1, dtype=bool)
    for axis, name in enumerate("xyz"):
        first = np.unique(np.take(labels, 0, axis=axis))
        last = np.unique(np.take(labels, -1, axis=axis))
        both = np.intersect1d(first[first > 0], last[last > 0])
        percolating[name] = bool(len(both))
        spanning[both] = True
    sizes_raw = np.bincount(labels.ravel(), minlength=n + 1)
    phase_volume = int(sizes_raw[1:].sum())
    spanning_fraction = float(sizes_raw[spanning].sum() / phase_volume) if phase_volume else 0.0
    if periodic and n:
        rep = _merge_periodic(labels, n)
        roots, inverse = np.unique(rep[1:], return_inverse=True)
        sizes = np.bincount(inverse, weights=sizes_raw[1:]).astype(np.int64)
        if return_labels:
            relabel = np.concatenate([[0], inverse + 1])
            labels = relabel[labels]
        n = len(roots)
    else:
        sizes = sizes_raw[1:]
    return Components(
        n_components=int(n),
        sizes=sizes,
        percolating=percolating,
        spanning_fraction=spanning_fraction,
        labels=labels if return_labels else None,
    )


def _wrapping(labels: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Whether a component of ``labels`` (labelled inside the box) wraps around the periodic
    box along each axis: a cycle of neighbours through the faces winds around the axis, so
    that the component tiles into a cluster unbounded along it."""
    shape = labels.shape
    edges = [np.empty((0, 5), dtype=np.int64)]
    for d in np.argwhere(structure) - 1:
        if tuple(d) <= (0, 0, 0):  # one offset of each opposite pair, not the centre
            continue
        for a in np.flatnonzero(d):  # voxels whose neighbour x + d is across the face normal to a
            index = [np.arange(n) for n in shape]
            index[a] = np.array([shape[a] - 1 if d[a] > 0 else 0])
            moved = [i + k for i, k in zip(index, d, strict=True)]
            u = labels[np.ix_(*index)]
            v = labels[np.ix_(*[m % n for m, n in zip(moved, shape, strict=True)])]
            shift = np.ix_(*[m // n for m, n in zip(moved, shape, strict=True)])
            shift = np.stack(np.broadcast_arrays(*shift), axis=-1)
            keep = (u > 0) & (v > 0)
            edges.append(np.column_stack([u[keep], v[keep], shift[keep]]))
    parent = list(range(int(labels.max()) + 1))
    frame = np.zeros((len(parent), 3), dtype=np.int64)  # box of a label's copy in its parent's

    def find(a):
        path = []
        while parent[a] != a:
            path.append(a)
            a = parent[a]
        for b in reversed(path[:-1]):  # path compression, frames made relative to the root
            frame[b] += frame[parent[b]]
            parent[b] = a
        return a

    wraps = np.zeros(3, dtype=bool)
    for u, v, *s in np.unique(np.concatenate(edges), axis=0).tolist():
        ru, rv = find(u), find(v)
        winding = frame[u] + s - frame[v]  # copy of v met from u's copy, minus v's own copy
        if ru == rv:
            wraps |= winding != 0
        else:
            parent[rv] = ru
            frame[rv] = winding
    return wraps


def wrapping(image, phase: int = 1, *, connectivity: int = 6) -> dict[str, bool]:
    """Whether a component of a phase of an image wraps around the periodic box along each
    axis (``connectivity``: 6, 18 or 26 neighbours): a cycle of neighbours through the faces
    winds around the axis, so that the tiled component is a cluster unbounded along it, the
    percolation of a periodic medium. Joining the two faces normal to the axis inside the
    box (the flags ``percolating`` of :func:`connected_components`) neither implies it (a
    staircase from face to face closes into no cycle) nor follows from it (a helix through
    the faces normal to y wraps along x and y and joins no pair of faces)."""
    structure = _structure(connectivity)
    labels, _ = ndimage.label(as_array(image) == phase, structure=structure)
    return dict(zip("xyz", _wrapping(labels, structure).tolist(), strict=True))
