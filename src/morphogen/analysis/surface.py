"""Specific surface area by the Cauchy-Crofton formula.

The number of boundary crossings per unit length of test lines, ``P_L``, is counted
along discrete directions of the voxel lattice; for any structure
``S_V = 2 <P_L>``, the mean being taken over isotropically distributed directions. It
is approximated by a weighted sum over 3 (axes) or 13 (axes, face and body diagonals)
directions, each weighted by the solid angle of its Voronoi cell on the unit sphere.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import SphericalVoronoi

from morphogen.analysis.basic import as_array, indicator
from morphogen.image import VoxelImage

DIRECTIONS_3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
DIRECTIONS_13 = (
    *DIRECTIONS_3,
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (-1, 1, 1),
)


def crofton_weights(directions, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Solid-angle weights (summing to 1) of the physical directions ``d * spacing``."""
    d = np.asarray(directions, dtype=float) * np.asarray(spacing, dtype=float)
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    pts = np.vstack([u, -u])
    sv = SphericalVoronoi(pts, radius=1.0, threshold=1e-8)
    areas = sv.calculate_areas()
    n = len(u)
    w = areas[:n] + areas[n:]
    return w / w.sum()


def _pair_slices(d, shape):
    src, dst = [], []
    for c, n in zip(d, shape, strict=True):
        if c > 0:
            src.append(slice(0, n - c))
            dst.append(slice(c, n))
        elif c < 0:
            src.append(slice(-c, n))
            dst.append(slice(0, n + c))
        else:
            src.append(slice(None))
            dst.append(slice(None))
    return tuple(src), tuple(dst)


def transitions(data: np.ndarray, d, periodic: bool) -> tuple[int, int]:
    """Number of pairs (x, x + d) with different values, and number of pairs."""
    if periodic:
        shifted = np.roll(data, shift=tuple(-c for c in d), axis=(0, 1, 2))
        return int(np.count_nonzero(data != shifted)), int(data.size)
    src, dst = _pair_slices(d, data.shape)
    a, b = data[src], data[dst]
    return int(np.count_nonzero(a != b)), int(a.size)


def interface_density(data: np.ndarray, spacing, *, periodic: bool, directions=13) -> float:
    """Area per unit volume of the boundaries between different values of ``data``."""
    dirs = DIRECTIONS_13 if directions == 13 else DIRECTIONS_3 if directions == 3 else None
    if dirs is None:
        raise ValueError("directions: 3 or 13")
    spacing = np.asarray(spacing, dtype=float)
    weights = crofton_weights(dirs, spacing)
    total = 0.0
    for w, d in zip(weights, dirs, strict=True):
        if not periodic and any(abs(c) >= n for c, n in zip(d, data.shape, strict=True)):
            continue
        n_cross, n_pairs = transitions(data, d, periodic)
        step = float(np.linalg.norm(np.asarray(d) * spacing))
        if n_pairs:
            total += w * n_cross / (n_pairs * step)
    return 2.0 * total


def specific_surface(image, phase: int = 1, *, directions: int = 13, periodic=None) -> float:
    """Interface area of ``phase`` per unit volume of the image (Crofton estimate)."""
    data = as_array(image)
    if data.dtype.kind == "f":
        raise ValueError("specific_surface needs a phase image (not partial volumes)")
    spacing = image.spacing if isinstance(image, VoxelImage) else (1.0, 1.0, 1.0)
    if periodic is None:
        periodic = image.periodic if isinstance(image, VoxelImage) else False
    ind = indicator(data, phase)
    return interface_density(ind, spacing, periodic=periodic, directions=directions)


def boundary_density(labels, *, directions: int = 13, periodic=None) -> float:
    """Area of the boundaries between cells per unit volume, for a label image."""
    data = as_array(labels)
    spacing = labels.spacing if isinstance(labels, VoxelImage) else (1.0, 1.0, 1.0)
    if periodic is None:
        periodic = labels.periodic if isinstance(labels, VoxelImage) else False
    return interface_density(data, spacing, periodic=periodic, directions=directions)
