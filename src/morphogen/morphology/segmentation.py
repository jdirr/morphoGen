"""Images turned into phases and particles: thresholds of grey levels, the particles of a
phase and their shapes, and the scores of a segmentation against a reference.

* *Thresholds* (Otsu 1979): the level ``t`` that maximizes the variance between the classes
  of the histogram, ``ω0 ω1 (μ0 - μ1)^2``; for ``k`` classes the ``k - 1`` levels that
  maximize ``sum_c ω_c μ_c^2`` (the same criterion, Liao, Chen & Chung 2001), by dynamic
  programming over the bins of the histogram. A Gaussian smoothing first removes the noise
  of the grey levels; the histogram may be equalized for display.
* *Particles*: the connected parts of a phase, those that touch split by the watershed of
  their distance map (:func:`morphogen.analysis.particle_sizes`), and the second moments
  of the voxels of each: a solid ellipsoid of semi-axes ``a >= b >= c`` has the moments
  ``a^2 / 5, b^2 / 5, c^2 / 5`` along its axes (an ellipse ``a^2 / 4, b^2 / 4`` in a
  plane), so that each particle has the semi-axes and the axes of the ellipsoid of its
  moments, and the laws of the sizes that models take (a lognormal law of the moments of
  the equivalent radii, or their empirical law).
* *Scores*: the agreement of a segmentation with a reference, voxel by voxel: accuracy,
  precision and recall of a phase, its intersection over union (Jaccard) and its Dice
  coefficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from morphogen.image import VoxelImage


def _array(image) -> np.ndarray:
    data = image.data if isinstance(image, VoxelImage) else np.asarray(image)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError("an image of two or three dimensions")
    return data


def as_image(data, spacing=(1.0, 1.0, 1.0), *, periodic: bool = False, name="grey") -> VoxelImage:
    """An array of two dimensions (a section, ``(nx, ny)``) or three (a volume, or a stack
    of sections along its last axis) as a voxel image."""
    if isinstance(data, VoxelImage):
        return data
    arr = _array(data)
    return VoxelImage(arr, spacing=spacing, periodic=periodic, name=name)


def stack(sections, spacing=(1.0, 1.0, 1.0), *, name: str = "grey") -> VoxelImage:
    """A volume from sections of the same shape (arrays ``(nx, ny)``, read by any library),
    stacked along z in their order."""
    arrays = [np.asarray(s) for s in sections]
    if not arrays or any(a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("stack: sections of two dimensions and of the same shape")
    return VoxelImage(np.stack(arrays, axis=-1), spacing=spacing, name=name)


def _histogram(values, bins: int):
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite values")
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        raise ValueError("a constant image: no threshold")
    counts, edges = np.histogram(v, bins=int(bins), range=(lo, hi))
    centres = 0.5 * (edges[1:] + edges[:-1])
    return counts.astype(float) / v.size, centres, edges


def otsu(values, bins: int = 256) -> float:
    """Otsu's threshold of grey levels: the level between the classes of the histogram that
    maximizes their variance between classes."""
    return multi_otsu(values, 2, bins)[0]


def multi_otsu(values, classes: int = 3, bins: int = 256) -> np.ndarray:
    """The ``classes - 1`` thresholds of grey levels that maximize the variance between the
    classes of the histogram (``sum_c ω_c μ_c^2``), by dynamic programming over its bins; each
    threshold is the upper edge of the last bin of a class."""
    k = int(classes)
    if k < 2:
        raise ValueError("classes: at least 2")
    p, x, edges = _histogram(values, bins)
    n = len(p)
    if n < k:
        raise ValueError("fewer bins than classes")
    cum_w = np.concatenate([[0.0], np.cumsum(p)])
    cum_m = np.concatenate([[0.0], np.cumsum(p * x)])

    def score(i, j):  # the bins i .. j - 1 as one class
        w = cum_w[j] - cum_w[i]
        m = cum_m[j] - cum_m[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(w > 0.0, m * m / w, 0.0)

    best = np.full((k + 1, n + 1), -np.inf)
    arg = np.zeros((k + 1, n + 1), dtype=np.int64)
    best[0, 0] = 0.0
    ends = np.arange(n + 1)
    for c in range(1, k + 1):
        for j in range(c, n + 1):
            i = ends[c - 1 : j]
            cand = best[c - 1, i] + score(i, j)
            m = int(np.argmax(cand))
            best[c, j] = cand[m]
            arg[c, j] = i[m]
    cuts = []
    j = n
    for c in range(k, 1, -1):
        j = int(arg[c, j])
        cuts.append(j)
    return np.array([edges[c] for c in reversed(cuts)])


def equalize(image, bins: int = 256) -> np.ndarray:
    """The grey levels mapped by their cumulative distribution onto ``[0, 1]`` (histogram
    equalization)."""
    data = _array(image).astype(float)
    p, _, edges = _histogram(data, bins)
    cdf = np.cumsum(p)
    index = np.clip(np.searchsorted(edges, data, side="right") - 1, 0, len(cdf) - 1)
    return cdf[index]


def smooth(image, sigma: float = 1.0) -> np.ndarray:
    """A Gaussian smoothing of the grey levels (``sigma`` in voxels; wrapped across the faces
    of a periodic image, the plane of a section kept)."""
    data = _array(image).astype(float)
    mode = "wrap" if isinstance(image, VoxelImage) and image.periodic else "nearest"
    sig = [float(sigma) if n > 1 else 0.0 for n in data.shape]
    return ndimage.gaussian_filter(data, sig, mode=mode)


def segment(image, classes: int = 2, *, thresholds=None, sigma: float = 0.0, bins: int = 256):
    """The phases of a grey-level image: ``classes`` phases by increasing grey level, split by
    the ``thresholds`` given or by Otsu's (multi-level) thresholds, after a Gaussian
    smoothing of ``sigma`` voxels. Returns the image of the phases (``uint8``), whose
    metadata keep the thresholds."""
    src = as_image(image)
    data = smooth(src, sigma) if sigma > 0.0 else _array(src).astype(float)
    if thresholds is None:
        thresholds = multi_otsu(data, classes, bins)
    thresholds = np.sort(np.atleast_1d(np.asarray(thresholds, dtype=float)))
    phases = np.searchsorted(thresholds, data, side="right").astype(np.uint8)
    return VoxelImage(
        phases,
        spacing=src.spacing,
        origin=src.origin,
        periodic=src.periodic,
        name="phase",
        metadata={**src.metadata, "thresholds": thresholds.tolist(), "sigma": float(sigma)},
    )


def scores(predicted, reference, phase: int = 1) -> dict[str, float]:
    """The agreement of a segmentation with a reference for one phase: the share of the voxels
    of the same phase in both (``accuracy``), ``precision`` and ``recall`` of the phase, its
    intersection over union ``iou`` and its ``dice`` coefficient."""
    a = _array(predicted) == phase
    b = _array(reference) == phase
    if a.shape != b.shape:
        raise ValueError("scores: images of the same shape")
    tp = float(np.sum(a & b))
    fp = float(np.sum(a & ~b))
    fn = float(np.sum(~a & b))
    tn = float(a.size) - tp - fp - fn

    def ratio(x, y):
        return x / y if y > 0 else float("nan")

    return {
        "accuracy": (tp + tn) / a.size,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "dice": ratio(2.0 * tp, 2.0 * tp + fp + fn),
    }


# ----------------------------------------------------------------------------------
# Particles and their shapes


@dataclass
class ParticleShapes:
    """The particles of a phase with the ellipsoids of their moments: ``semi_axes`` (n x 3,
    decreasing; the third 0 for the particles of a section), ``axes`` (n x 3 x 3, the
    columns along the semi-axes), the equivalent ``radii`` (of the balls of their volumes,
    or of the discs of their areas in a section) and the ``weights`` of the particles (0 for
    those cut by the frame of an image that is not periodic)."""

    semi_axes: np.ndarray
    axes: np.ndarray
    radii: np.ndarray
    weights: np.ndarray
    centroids: np.ndarray
    density: float
    planar: bool

    @property
    def count(self) -> int:
        return len(self.radii)

    def _w(self) -> np.ndarray:
        return self.weights / self.weights.sum()

    def moments(self, k: int) -> float:
        """The number moment of order ``k`` of the equivalent radii."""
        return float(np.dot(self._w(), self.radii**k))

    @property
    def aspect_ratios(self) -> np.ndarray:
        """``b / a`` and ``c / a`` (``b / a`` alone in a section) of each particle."""
        a = self.semi_axes[:, :1]
        if self.planar:
            return self.semi_axes[:, 1:2] / a
        return self.semi_axes[:, 1:] / a

    def radius_law(self, kind: str = "lognormal"):
        """The law of the equivalent radii for a model: ``"lognormal"`` (of their first two
        number moments), ``"constant"`` (their mean) or ``"empirical"`` (their values with
        their weights)."""
        from morphogen.distributions import Constant, Discrete, LogNormal

        m1, m2 = self.moments(1), self.moments(2)
        if kind == "constant":
            return Constant(m1)
        if kind == "empirical":
            keep = self.weights > 0
            return Discrete(tuple(self.radii[keep]), tuple(self.weights[keep]))
        if kind == "lognormal":
            cv2 = max(m2 / (m1 * m1) - 1.0, 0.0)
            if cv2 < 1e-6:
                return Constant(m1)
            return LogNormal(m1, math.sqrt(cv2) * m1)
        raise ValueError('kind: "lognormal", "constant" or "empirical"')


def particle_shapes(image, phase: int = 1, *, separate: bool = True, prominence=None,
                    min_radius: float = 0.0) -> ParticleShapes:  # fmt: skip
    """The particles of ``phase`` (:func:`morphogen.analysis.particle_sizes`: connected parts
    split by the watershed of their distance map) and the ellipsoids of their moments. A
    section (an image one voxel thick along z) gives ellipses in its plane."""
    from morphogen.analysis.particles import particle_sizes

    src = as_image(image)
    if src.data.dtype.kind == "f":
        raise ValueError("particle_shapes: an image of phases (segment it first)")
    parts = particle_sizes(
        src, phase, separate=separate, prominence=prominence, min_radius=min_radius
    )
    labels = parts.labels
    n = parts.count
    planar = labels.shape[2] == 1
    h = np.asarray(src.spacing, dtype=float)
    idx = np.nonzero(labels)
    which = labels[idx] - 1
    pos = np.stack(idx, axis=1).astype(float) * h
    if src.periodic:  # the voxels of a particle nearest its centroid across the faces
        lengths = np.asarray(labels.shape, dtype=float) * h
        centre = (parts.centroids - np.asarray(src.origin) - 0.5 * h)[which]
        pos = centre + (pos - centre + 0.5 * lengths) % lengths - 0.5 * lengths
    count = np.bincount(which, minlength=n).astype(float)
    mean = np.stack([np.bincount(which, pos[:, a], n) for a in range(3)], axis=1) / count[:, None]
    d = pos - mean[which]
    cov = np.zeros((n, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            cov[:, a, b] = cov[:, b, a] = np.bincount(which, d[:, a] * d[:, b], n) / count
    # each voxel is a cell of the grid, not a point: its own moments h^2 / 12
    cov += np.diag(h**2 / 12.0)[None]
    vals, vecs = np.linalg.eigh(cov)
    vals, vecs = vals[:, ::-1], vecs[:, :, ::-1]
    factor = 4.0 if planar else 5.0
    semi = np.sqrt(np.clip(vals, 0.0, None) * factor)
    if planar:
        semi[:, 2] = 0.0
        area = count * h[0] * h[1]
        radii = np.sqrt(area / math.pi)
        box = float(labels.shape[0] * h[0] * labels.shape[1] * h[1])
    else:
        radii = parts.radii
        box = parts.volume
    density = float(parts.weights.sum() / box)
    return ParticleShapes(semi, vecs, radii, parts.weights, parts.centroids, density, planar)
