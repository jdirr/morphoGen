"""Statistical descriptors of random heterogeneous materials (Torquato 2002).

* *Lineal-path function* ``L(z)`` (Lu & Torquato 1992): the probability that a segment
  of length ``z`` lies entirely in the phase, along the axes of the grid (segments of
  ``k + 1`` voxel centres at the lag ``k``). ``L(0)`` is the volume fraction; ``L`` keeps
  the connectedness along lines that the covariance ``S_2`` ignores.
* *Two-point cluster function* ``C_2(r)`` (Torquato, Beasley & Chiew 1988): the
  probability that ``x`` and ``x + r`` lie in the same cluster of the phase (connected
  components, merged across the faces of periodic images); ``C_2 <= S_2``, and ``C_2``
  keeps a long-range part only beyond the percolation threshold.
* *Pore-size function* (Prager 1963; Torquato & Avellaneda 1991): the law of the distance
  ``δ`` from a point of the phase to the nearest point of the other phase: its
  exceedance ``F(δ)`` (``F(0) = 1``), density ``P(δ) = -F'(δ)`` and moments; the mean
  pore size is ``∫ F(δ) dδ``. On voxels, ``δ`` is the distance between voxel centres less
  half a voxel (the interface lies between the centres), exact for interfaces along the
  axes and of the first order in the voxel size otherwise (the mean pore size of a
  Boolean model of spheres of five voxels about 4 % low).
* *Radial distribution function* ``g(r)`` and *structure factor* ``S(k)`` of points in a
  periodic box (the centres of packings): ``g`` is the density of the pairs at the
  distance ``r`` over that of independent points; ``S(k) = |Σ_j e^{-i k·x_j}|² / N`` on
  the wave vectors of the box, whose limit ``S(k → 0) = 0`` defines hyperuniform point
  patterns (Torquato & Stillinger 2003).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from morphogen.analysis.basic import indicator
from morphogen.analysis.connectivity import connected_components
from morphogen.analysis.covariance import AxisFunction
from morphogen.analysis.morphology import _distance, _periodic, _spacing


def _runs(lines: np.ndarray, periodic: bool) -> tuple[np.ndarray, int]:
    """The lengths of the runs of ones of the rows of a boolean array, and the number of
    rows full of ones (periodic rows: the runs wrap around)."""
    n = lines.shape[1]
    full = 0
    if periodic:
        is_full = lines.all(axis=1)
        full = int(is_full.sum())
        lines = lines[~is_full]
        if not len(lines):
            return np.zeros(0, dtype=np.int64), full
        # start every row on a zero: no run wraps around
        shift = np.argmin(lines, axis=1)
        lines = np.take_along_axis(lines, (np.arange(n)[None, :] + shift[:, None]) % n, axis=1)
    d = np.diff(np.pad(lines, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    return (np.nonzero(d == -1)[1] - np.nonzero(d == 1)[1]).astype(np.int64), full


def lineal_path(image, phase: int = 1, *, max_lag: int | None = None, periodic=None):
    """Lineal-path function of ``phase`` along the three axes, for the lags 0..max_lag
    voxels (half the image by default): an :class:`AxisFunction`."""
    field = indicator(image, phase)
    spacing = _spacing(image)
    periodic = _periodic(image, periodic)
    lags, values = {}, {}
    for axis, name in enumerate("xyz"):
        n = field.shape[axis]
        m = n // 2 if max_lag is None else min(int(max_lag), n - 1)
        lines = np.moveaxis(field, axis, -1).reshape(-1, n)
        runs, full = _runs(lines, periodic)
        counts = np.bincount(runs, minlength=n + 1).astype(np.float64)
        ell = np.arange(n + 1, dtype=np.float64)
        # the segments of k + 1 voxels in a run of length l: l - k, for l > k
        above = np.cumsum((counts * ell)[::-1])[::-1]  # sum over l >= k of l n_l
        number = np.cumsum(counts[::-1])[::-1]
        k = np.arange(m + 1)
        inside = above[k + 1] - k * number[k + 1] + full * n
        positions = lines.shape[0] * (n if periodic else n - k)
        values[name] = inside / positions
        lags[name] = k * spacing[axis]
    return AxisFunction(lags, values, float(np.mean(field, dtype=np.float64)))


def two_point_cluster(image, phase: int = 1, *, max_lag: int | None = None, periodic=None):
    """Two-point cluster function of ``phase`` along the three axes (face-connected
    clusters), for the lags 0..max_lag voxels: an :class:`AxisFunction`."""
    periodic = _periodic(image, periodic)
    labels = connected_components(image, phase, periodic=periodic, return_labels=True).labels
    spacing = _spacing(image)
    lags, values = {}, {}
    for axis, name in enumerate("xyz"):
        n = labels.shape[axis]
        m = n // 2 if max_lag is None else min(int(max_lag), n - 1)
        out = np.empty(m + 1)
        for k in range(m + 1):
            if periodic:
                other = np.roll(labels, -k, axis=axis)
                out[k] = np.mean((labels == other) & (labels > 0))
            else:
                a = np.take(labels, np.arange(n - k), axis=axis)
                b = np.take(labels, np.arange(k, n), axis=axis)
                out[k] = np.mean((a == b) & (a > 0))
        values[name] = out
        lags[name] = np.arange(m + 1) * spacing[axis]
    return AxisFunction(lags, values, float(np.mean(labels > 0)))


@dataclass
class PoreSizes:
    """The pore-size function of a phase: the bin edges ``delta``, the exceedance ``F``
    at the edges, the density ``P`` in the bins, and the first two moments of ``δ``."""

    delta: np.ndarray
    exceedance: np.ndarray
    density: np.ndarray
    mean: float
    second_moment: float


def pore_size_function(image, phase: int = 0, *, bins: int = 64, periodic=None) -> PoreSizes:
    """The pore-size function of ``phase`` (by default the pores, phase 0): the law of the
    distance from its points to the other phase (see the module)."""
    mask = indicator(image, phase).astype(bool)
    if not mask.any():
        raise ValueError(f"pore sizes: no voxel of phase {phase}")
    if mask.all():
        raise ValueError(f"pore sizes: the image is all of phase {phase}")
    spacing = _spacing(image)
    periodic = _periodic(image, periodic)
    dist = _distance(mask, spacing, False, 0)
    if periodic:  # through the faces: a margin of the largest distance inside the box
        pad = min(int(np.ceil(dist.max() / spacing.min())) + 1, max(mask.shape) // 2)
        dist = _distance(mask, spacing, True, pad)
    delta = np.maximum(dist[mask] - 0.5 * float(spacing.min()), 0.0)
    edges = np.linspace(0.0, float(delta.max()) * (1.0 + 1e-12), bins + 1)
    ordered = np.sort(delta)
    exceedance = 1.0 - np.searchsorted(ordered, edges, side="left") / len(ordered)
    density = -np.diff(exceedance) / np.diff(edges)
    return PoreSizes(edges, exceedance, density, float(delta.mean()), float(np.mean(delta**2)))


def _wrapped(points, box) -> np.ndarray:
    p = np.mod(np.asarray(points, dtype=float), box)
    p[p >= box] = 0.0
    return p


def radial_distribution(points, box, *, r_max: float | None = None, bins: int = 100):
    """The radial distribution function ``g(r)`` of points in a periodic box of edges
    ``box``: the centres of the bins of ``r`` (up to ``r_max``, a quarter of the
    smallest edge by default) and ``g`` in them."""
    box = np.asarray(box, dtype=float)
    pts = _wrapped(points, box)
    n = len(pts)
    if n < 2:
        raise ValueError("radial distribution: two points at least")
    r_max = float(box.min()) / 4.0 if r_max is None else float(r_max)
    if r_max > box.min() / 2.0:
        raise ValueError("radial distribution: r_max up to half the box")
    pairs = cKDTree(pts, boxsize=box).query_pairs(r_max, output_type="ndarray")
    d = pts[pairs[:, 0]] - pts[pairs[:, 1]]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    edges = np.linspace(0.0, r_max, bins + 1)
    counts = np.histogram(r, bins=edges)[0].astype(np.float64)
    shells = 4.0 / 3.0 * math.pi * np.diff(edges**3)
    density = n / float(np.prod(box))
    return 0.5 * (edges[1:] + edges[:-1]), 2.0 * counts / (n * density * shells)


def structure_factor(points, box, *, k_max: float | None = None, bins: int = 40):
    """The structure factor ``S(k)`` of points in a periodic box, averaged over the wave
    vectors ``2π n / L`` (``n`` integer, not all zero) in bins of ``|k|`` up to ``k_max``
    (twenty times the smallest wave number by default): the centres of the bins with
    wave vectors and ``S`` in them."""
    box = np.asarray(box, dtype=float)
    pts = _wrapped(points, box)
    n_pts = len(pts)
    k0 = 2.0 * math.pi / box
    k_max = 20.0 * float(k0.min()) if k_max is None else float(k_max)
    n_max = np.floor(k_max / k0).astype(int)
    # the density modes, separable along the axes: rho(k) = sum_j prod_a e^{-i k_a x_ja}
    waves = [np.exp(-1j * np.outer(pts[:, a], k0[a] * np.arange(-n_max[a], n_max[a] + 1)))
             for a in range(3)]  # fmt: skip
    rho = np.einsum("ja,jb,jc->abc", waves[0], waves[1], waves[2], optimize=True)
    grids = np.meshgrid(*[k0[a] * np.arange(-n_max[a], n_max[a] + 1) for a in range(3)],
                        indexing="ij")  # fmt: skip
    k = np.sqrt(sum(g**2 for g in grids))
    s = np.abs(rho) ** 2 / n_pts
    keep = (k > 0) & (k <= k_max)
    edges = np.linspace(0.0, k_max, bins + 1)
    which = np.digitize(k[keep], edges) - 1
    total = np.bincount(which, weights=s[keep], minlength=bins)[:bins]
    count = np.bincount(which, minlength=bins)[:bins]
    centres = 0.5 * (edges[1:] + edges[:-1])
    filled = count > 0
    return centres[filled], total[filled] / count[filled]
