"""Fractal dimension of a phase by box counting (Minkowski-Bouligand dimension)."""

from __future__ import annotations

import math

import numpy as np

from morphogen.analysis.basic import as_array


def box_counting(image, phase: int = 1, sizes=None) -> dict:
    """Number ``N(s)`` of boxes of edge ``s`` voxels that contain the phase, and the
    dimension ``D = -d log N / d log s`` (least-squares slope over the sizes).

    The default sizes are the powers of two from 1 to an eighth of the smallest edge of
    the image; axes of one voxel (2D images) are not divided. For a fractal aggregate,
    use sizes between the particle size and the size of the clusters.
    """
    data = as_array(image) == phase
    if data.ndim != 3:
        raise ValueError("box_counting needs a 3D image ((nx, ny, 1) for 2D images)")
    axes = [n for n in data.shape if n > 1]
    if not axes:
        raise ValueError("box_counting: the image has a single voxel")
    if sizes is None:
        sizes = [2**k for k in range(int(math.log2(max(1, min(axes) // 8))) + 1)]
    sizes = sorted({int(s) for s in sizes})
    if len(sizes) < 2 or sizes[0] < 1:
        raise ValueError("box_counting: at least two box sizes of one voxel or more")
    counts = []
    for s in sizes:
        steps = [s if n > 1 else 1 for n in data.shape]
        if any(n < st for n, st in zip(data.shape, steps, strict=True)):
            raise ValueError(f"box_counting: boxes of {s} voxels exceed the image")
        trimmed = data[
            tuple(slice(0, (n // st) * st) for n, st in zip(data.shape, steps, strict=True))
        ]
        shape = []
        for n, st in zip(trimmed.shape, steps, strict=True):
            shape += [n // st, st]
        counts.append(int(trimmed.reshape(shape).any(axis=(1, 3, 5)).sum()))
    counts = np.asarray(counts)
    valid = counts > 0
    dimension = float("nan")
    if valid.sum() >= 2:
        slope = np.polyfit(np.log(np.asarray(sizes)[valid]), np.log(counts[valid]), 1)[0]
        dimension = float(-slope)
    return {"sizes": sizes, "counts": counts.tolist(), "dimension": dimension}


def box_dimension(image, phase: int = 1, sizes=None) -> float:
    """Box-counting dimension of a phase (see :func:`box_counting`)."""
    return box_counting(image, phase, sizes)["dimension"]
