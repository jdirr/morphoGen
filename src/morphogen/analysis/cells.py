"""Statistics of label images (cells of tessellations, grains).

Negative labels mark voxels outside every cell (e.g. the untransformed matrix of a
partially crystallized structure): they are ignored.
"""

from __future__ import annotations

import numpy as np

from morphogen.analysis.basic import as_array
from morphogen.image import VoxelImage


def cell_volumes(labels, n_cells: int | None = None) -> np.ndarray:
    """Volume of every label 0..n_cells-1 (voxel counts times the voxel volume)."""
    data = as_array(labels)
    if data.dtype.kind not in "iu":
        raise ValueError("cell_volumes needs an integer label image")
    data = data.ravel()
    counts = np.bincount(data[data >= 0], minlength=n_cells or 0)
    vv = labels.voxel_volume if isinstance(labels, VoxelImage) else 1.0
    return counts * vv


def cell_adjacency(labels, periodic=None) -> np.ndarray:
    """Pairs ``(a, b)``, ``a < b``, of labels sharing at least one voxel face."""
    data = as_array(labels).astype(np.int64)
    if periodic is None:
        periodic = labels.periodic if isinstance(labels, VoxelImage) else False
    n = int(data.max()) + 1 if data.size else 0
    keys = []
    for axis in range(3):
        if periodic:
            other = np.roll(data, -1, axis=axis)
            a, b = data.ravel(), other.ravel()
        else:
            a = np.take(data, np.arange(data.shape[axis] - 1), axis=axis).ravel()
            b = np.take(data, np.arange(1, data.shape[axis]), axis=axis).ravel()
        diff = (a != b) & (a >= 0) & (b >= 0)
        lo = np.minimum(a[diff], b[diff])
        hi = np.maximum(a[diff], b[diff])
        keys.append(np.unique(lo * n + hi))
    if not keys:
        return np.empty((0, 2), dtype=np.int64)
    k = np.unique(np.concatenate(keys))
    return np.stack([k // n, k % n], axis=1)


def cell_neighbors(labels, periodic=None, n_cells: int | None = None) -> np.ndarray:
    """Number of face-sharing neighbours of every label (voxel estimate)."""
    pairs = cell_adjacency(labels, periodic)
    n = n_cells or max(0, int(as_array(labels).max(initial=-1)) + 1)
    return np.bincount(pairs.ravel(), minlength=n)
