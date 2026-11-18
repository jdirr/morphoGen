"""Chords, granulometry, local thickness, Euler characteristic and geometric tortuosity.

* *Chord lengths*: lengths of the runs of a phase along the lines of the grid (the
  linear sections of the phase). On periodic images, runs wrap around the faces; on
  other images, runs cut by the faces are left out.
* *Granulometry by openings*: fraction of the phase left by the opening with a ball of
  radius r (erosion then dilation, both exact Euclidean balls through distance
  transforms; Matheron 1975, Serra 1982). Its decrease with r is the size distribution of
  the phase; the *local thickness* of a voxel is twice the largest radius of an opening
  that keeps it (Hildebrand & Rüegsegger 1997): the pore sizes are the local thickness of
  the pores.
* *Euler characteristic* of a phase (6-connected voxels): number of components - tunnels +
  cavities, from the counts of voxels, pairs, squares and cubes of voxels (Ohser &
  Mücklich 2000); its density, with the volume fraction and the specific surface, gives
  three of the Minkowski functionals.
* *Geometric tortuosity*: shortest paths inside the phase (26-connected voxels, Euclidean
  steps) from one face of the box to the opposite face, divided by the distance between
  the faces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from morphogen.analysis.basic import indicator


def _periodic(image, periodic) -> bool:
    if periodic is not None:
        return bool(periodic)
    return bool(getattr(image, "periodic", False))


def _spacing(image) -> np.ndarray:
    return np.asarray(getattr(image, "spacing", (1.0, 1.0, 1.0)), dtype=float)


# -- chords ----------------------------------------------------------------------------------


def chord_lengths(image, phase: int = 1, axis: int = 0, *, periodic=None) -> np.ndarray:
    """Lengths of the chords of ``phase`` along ``axis`` (physical units)."""
    data = indicator(image, phase)
    lines = np.moveaxis(data, axis, -1).reshape(-1, data.shape[axis])
    n = lines.shape[1]
    h = _spacing(image)[axis]
    if _periodic(image, periodic):
        full = lines.all(axis=1)
        lines = lines[~full]  # lines inside the phase: chords longer than the box
        if not len(lines):
            return np.zeros(0)
        # start every line on a voxel outside the phase: no run wraps around
        shift = np.argmin(lines, axis=1)
        idx = (np.arange(n)[None, :] + shift[:, None]) % n
        lines = np.take_along_axis(lines, idx, axis=1)
        padded = np.pad(lines, ((0, 0), (1, 1)))
        d = np.diff(padded.astype(np.int8), axis=1)
        starts = np.nonzero(d == 1)
        ends = np.nonzero(d == -1)
        return (ends[1] - starts[1]).astype(float) * h
    padded = np.pad(lines, ((0, 0), (1, 1)))
    d = np.diff(padded.astype(np.int8), axis=1)
    _, s_col = np.nonzero(d == 1)
    _, e_col = np.nonzero(d == -1)
    keep = (s_col > 0) & (e_col < n)  # runs cut by the faces are left out
    return (e_col[keep] - s_col[keep]).astype(float) * h


def mean_chord_length(image, phase: int = 1, *, periodic=None) -> float:
    """Mean chord length of ``phase`` over the three axes (all chords pooled)."""
    chords = np.concatenate([chord_lengths(image, phase, a, periodic=periodic) for a in range(3)])
    return float(chords.mean()) if chords.size else float("nan")


# -- openings, granulometry, local thickness -------------------------------------------------------


def _distance(mask: np.ndarray, spacing, periodic: bool, pad: int) -> np.ndarray:
    """Euclidean distance from every voxel of ``mask`` to the nearest voxel outside it
    (periodic images: through the faces, up to ``pad`` voxels)."""
    if periodic and pad > 0:
        big = np.pad(mask, pad, mode="wrap")
        dist = ndimage.distance_transform_edt(big, sampling=spacing)
        return dist[pad:-pad, pad:-pad, pad:-pad]
    return ndimage.distance_transform_edt(mask, sampling=spacing)


@dataclass
class Granulometry:
    """Openings of a phase by balls of increasing radius."""

    radii: np.ndarray  # physical radii
    fractions: np.ndarray  # volume fraction of the phase kept by the opening of each radius
    thickness: np.ndarray  # local thickness (2 x largest opening radius) of each voxel

    @property
    def size_distribution(self) -> np.ndarray:
        """Volume fraction of the phase whose largest opening radius is each radius."""
        f = np.append(self.fractions, 0.0)
        return f[:-1] - f[1:]

    @property
    def mean_radius(self) -> float:
        """Mean of the radius of the largest ball (of the radii probed) containing a
        point of the phase."""
        dist = self.size_distribution
        return float(np.sum(dist * self.radii) / dist.sum()) if dist.sum() > 0 else float("nan")


def granulometry(image, phase: int = 1, radii=None, *, periodic=None) -> Granulometry:
    """Openings of ``phase`` by balls of the radii given (physical units; default: steps
    of half a voxel up to the largest inscribed ball). The opening of radius r is the
    union of the balls of radius r (the voxels closer than r to their centre) whose centre
    is at least r from the voxels outside the phase, as in the local thickness of
    Hildebrand & Rüegsegger (1997): a layer of n voxels has thickness n voxels. Each voxel
    gets the largest radius whose opening keeps it, so that the fractions kept decrease
    with the radius (discrete Euclidean balls do not make exact granulometries)."""
    mask = indicator(image, phase)
    h = _spacing(image)
    per = _periodic(image, periodic)
    vol = max(int(mask.sum()), 1)
    hmin = float(h.min())
    pad = 0
    if per:  # wrap the image by more than the largest ball (bounded by the plain transform)
        rough = ndimage.distance_transform_edt(mask, sampling=h)
        pad = min(int(np.ceil(rough.max() / hmin)) + 2, max(mask.shape))
    dist = _distance(mask, h, per, pad)
    if radii is None:
        radii = hmin * (1.0 + 0.5 * np.arange(max(0, int(2.0 * (dist.max() / hmin - 1.0))) + 1))
    radii = np.sort(np.asarray(radii, dtype=float))
    radius = np.where(mask, 0.5 * hmin, 0.0)  # largest opening radius of every voxel
    for r in radii:
        # balls of radius r centred on voxels at least r from the voxels outside the
        # phase; each covers the voxels closer than r to its centre
        centres = dist >= r
        if not centres.any():
            break
        reach = _distance(~centres, h, per, pad)  # distance to the nearest centre
        radius[mask & (reach < r)] = r
    kept = radius[mask]
    fractions = np.array([np.count_nonzero(kept >= r) / vol for r in radii])
    return Granulometry(radii=radii, fractions=fractions, thickness=2.0 * radius)


def local_thickness(image, phase: int = 1, *, periodic=None) -> np.ndarray:
    """Local thickness of every voxel of ``phase`` (0 elsewhere)."""
    return granulometry(image, phase, periodic=periodic).thickness


# -- Euler characteristic ---------------------------------------------------------------------


def euler_characteristic(image, phase: int = 1, *, periodic=None) -> int:
    """Euler characteristic (number of components - tunnels + cavities) of ``phase``,
    6-connected (its complement 26-connected): vertices - edges + squares - cubes of the
    complex whose vertices are the voxels of the phase, edges their pairs of face
    neighbours, squares and cubes their 2 x 2 and 2 x 2 x 2 blocks (Ohser & Mücklich
    2000). Periodic images are counted on the torus. Objects closer than a voxel touch on
    the image: the estimate is biased by their contacts (for random objects, about half
    the number of pairs closer than a voxel)."""
    x = indicator(image, phase)
    if not _periodic(image, periodic):
        x = np.pad(x, 1)

    def nxt(a, axis):
        return np.roll(a, -1, axis=axis)

    vertices = int(x.sum())
    edges = sum(int((x & nxt(x, a)).sum()) for a in range(3))
    squares = 0
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pair = x & nxt(x, a)
        squares += int((pair & nxt(pair, b)).sum())
    cubes = x & nxt(x, 0)
    cubes = cubes & nxt(cubes, 1)
    cubes = cubes & nxt(cubes, 2)
    return vertices - edges + squares - int(cubes.sum())


@dataclass
class Minkowski:
    """Densities of the four Minkowski functionals of a phase: ``V_V``, ``S_V``, the
    integral of mean curvature ``M_V`` (``(1/2) int (1/R1 + 1/R2) dS`` per unit volume)
    and the Euler characteristic ``chi_V``."""

    volume_fraction: float
    specific_surface: float
    euler_density: float  # Euler characteristic per unit volume
    mean_curvature: float = float("nan")


def minkowski(image, phase: int = 1, *, periodic=None) -> Minkowski:
    """The Minkowski densities of a phase of an image: its volume fraction, specific surface,
    density of Euler characteristic and density of mean curvature."""
    from morphogen.analysis.surface import specific_surface
    from morphogen.analysis.topology import mean_curvature_density

    volume = float(np.prod(np.asarray(indicator(image, phase).shape) * _spacing(image)))
    return Minkowski(
        volume_fraction=float(indicator(image, phase).mean()),
        specific_surface=specific_surface(image, phase, periodic=periodic),
        euler_density=euler_characteristic(image, phase, periodic=periodic) / volume,
        mean_curvature=mean_curvature_density(image, phase, periodic=periodic),
    )


# -- geometric tortuosity -----------------------------------------------------------------------


def geodesic_distance(image, phase: int = 1, axis: int = 0) -> np.ndarray:
    """Length of the shortest path inside ``phase`` (26-connected voxels, Euclidean steps
    between voxel centres) from the face ``axis = 0`` of the box to every voxel (inf: not
    reached; NaN outside the phase)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    mask = indicator(image, phase)
    h = _spacing(image)
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[mask] = np.arange(int(mask.sum()))
    rows, cols, weights = [], [], []

    def pair(d, n):  # slices of the voxels and of their neighbours at offset d
        return (slice(0, n - d), slice(d, n)) if d >= 0 else (slice(-d, n), slice(0, n + d))

    for offset in np.ndindex(3, 3, 3):
        d = np.array(offset) - 1
        if tuple(d) <= (0, 0, 0):
            continue  # each pair of neighbours once
        sa, sb = zip(*(pair(int(di), n) for di, n in zip(d, mask.shape, strict=True)), strict=True)
        a, b = index[sa], index[sb]
        ok = (a >= 0) & (b >= 0)
        rows.append(a[ok])
        cols.append(b[ok])
        weights.append(np.full(int(ok.sum()), float(np.linalg.norm(d * h))))
    n = int(mask.sum())
    graph = coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    ).tocsr()
    sources = np.take(index, 0, axis=axis)
    sources = sources[sources >= 0]
    out = np.full(mask.shape, np.nan)
    if sources.size == 0:
        out[mask] = np.inf
        return out
    dist = dijkstra(graph, directed=False, indices=sources, min_only=True)
    out[mask] = dist + 0.5 * h[axis]  # from the face, not from the first voxel centres
    return out


def geometric_tortuosity(image, phase: int = 1, axis: int = 0) -> float:
    """Mean length of the shortest paths through ``phase`` between the faces normal to
    ``axis``, divided by the edge of the box (inf when the phase does not percolate)."""
    dist = geodesic_distance(image, phase, axis)
    outlet = np.take(dist, -1, axis=axis)
    outlet = outlet[np.isfinite(outlet) | np.isinf(outlet)]
    reached = outlet[np.isfinite(outlet)]
    if reached.size == 0:
        return float("inf")
    h = _spacing(image)[axis]
    length = dist.shape[axis] * h
    return float(np.mean(reached + 0.5 * h) / length)
