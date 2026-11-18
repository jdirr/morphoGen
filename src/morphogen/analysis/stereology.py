"""Stereology: the sections of a phase by the planes of the grid, and the recovery of the
size distribution of its particles from them (Wicksell 1925; Saltykov 1967), for spheres
and for aligned ellipsoids of a common shape and orientation, as Degdeg & Dirrenberger
for the pores of a copper foam.

A plane normal to the unit vector ``n`` at the distance ``d`` from the centre of the
ellipsoid ``x . M x = 1``, ``M^-1 = R diag(a^2, b^2, c^2) R^T``, cuts it in an ellipse of
area ``pi abc (s - d^2) / s^(3/2)``, ``s = n . M^-1 n``. Uniformly placed particles have
``u = d / sqrt(s)`` uniform over ``[-1, 1]``: the equivalent radius of their sections is
``r = r_max sqrt(1 - u^2)``, the law of a sphere, ``F(v) = 1 - sqrt(1 - v^2)`` for
``v = r / r_max``, with ``r_max = kappa a``, ``kappa = sqrt(l1 l2) / s_hat^(1/4)`` for the
shape factors ``l1 = b/a``, ``l2 = c/a`` and ``s_hat = n . R diag(1, l1^2, l2^2) R^T n``;
the planes of a family cut the particles in proportion to their height ``H = 2 a
sqrt(s_hat)``. On classes of radii of width ``delta``, a particle of the 3D class ``I``
(``r_max = I delta``) gives a section of the 2D class ``J <= I`` with the probability
``P_IJ = (sqrt(I^2 - (J - 1)^2) - sqrt(I^2 - J^2)) / I`` (the Scheil–Schwartz–Saltykov
matrix), so that the areal densities of the sections are ``N_A,J = sum_I P_IJ H_I N_V,I``,
inverted class by class from the largest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from morphogen.analysis.basic import indicator
from morphogen.distributions import Discrete
from morphogen.image import VoxelImage

#: In-plane 8-connectivity of the slices of a stack whose first axis is normal to them.
_PLANE = np.zeros((3, 3, 3), dtype=bool)
_PLANE[1] = True


@dataclass
class Sections:
    """The sections of a phase by the planes normal to ``axis``: their equivalent radii
    ``sqrt(area / pi)``, their weights (1, or the Miles–Lantuéjoul factor of a section
    inside the frame of a non-periodic image) and the total area of the planes."""

    axis: int
    radii: np.ndarray
    weights: np.ndarray
    area: float

    def areal_density(self, edges) -> np.ndarray:
        """Sections per unit area in the classes of radii between ``edges``."""
        counts, _ = np.histogram(self.radii, bins=edges, weights=self.weights)
        return counts / self.area

    def mean_radius(self) -> float:
        """Number mean of the section radii."""
        return float(np.average(self.radii, weights=self.weights))


def _separate(work: np.ndarray, h, prominence) -> np.ndarray:
    """Watershed of the distance map of each slice from its maxima of the given
    prominence (and the highest point of each section)."""
    from morphogen.kernels.watershed import flood, prominent_maxima

    prominence = 0.5 * min(h) if prominence is None else float(prominence)
    out = np.zeros(work.shape, dtype=np.int32)
    count = 0
    for k in range(work.shape[0]):
        mask = work[k]
        if not mask.any():
            continue
        dist = ndimage.distance_transform_edt(mask, sampling=h)
        peaks = prominent_maxima(dist, mask, prominence)
        markers, m = ndimage.label(peaks, structure=np.ones((3, 3), dtype=bool))
        markers = markers.astype(np.int32)
        markers[~mask] = -1
        flood(-dist, markers)
        markers[~mask] = 0
        out[k] = np.where(markers > 0, markers + count, 0)
        count += m
    return out


def section_sizes(
    image,
    phase: int = 1,
    axis: int = 2,
    *,
    separate: bool = False,
    prominence: float | None = None,
    min_radius: float = 0.0,
    periodic=None,
) -> Sections:
    """The sections of ``phase`` by the planes of voxels normal to ``axis`` (8-connected
    pixels). ``separate``: touching sections are split by a watershed of their distance
    map from its maxima of prominence at least ``prominence`` (physical units, default
    half a pixel), in the manner of the binary watershed of Fiji. Sections of equivalent
    radius below ``min_radius`` are dropped. On periodic images each section counts once,
    whole across the faces; otherwise the sections cut by the frame are dropped and the
    others weighted by ``n1 n2 / ((n1 - 1 - w1)(n2 - 1 - w2))`` for their extents ``w``
    in pixels (Miles–Lantuéjoul)."""
    x = indicator(image, phase).astype(bool)
    spacing = np.asarray(image.spacing if isinstance(image, VoxelImage) else (1.0, 1.0, 1.0), float)
    if periodic is None:
        periodic = bool(image.periodic) if isinstance(image, VoxelImage) else False
    axis = int(axis)
    mask = np.ascontiguousarray(np.moveaxis(x, axis, 0))
    h = spacing[[a for a in range(3) if a != axis]]
    n0, n1, n2 = mask.shape
    work = np.pad(mask, ((0, 0), (n1, n1), (n2, n2)), mode="wrap") if periodic else mask
    labels, n = ndimage.label(work, structure=_PLANE)
    if separate and n:
        labels = _separate(work, h, prominence)
    pixels = np.bincount(labels.ravel())
    radii, weights = [], []
    for k, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        lo1, hi1, lo2, hi2 = box[1].start, box[1].stop, box[2].start, box[2].stop
        if lo1 == 0 or lo2 == 0 or hi1 == work.shape[1] or hi2 == work.shape[2]:
            continue  # cut by the frame (or wrapping onto itself)
        if periodic:
            if not (n1 <= lo1 < 2 * n1 and n2 <= lo2 < 2 * n2):
                continue  # another copy of a section of the window
            weight = 1.0
        else:
            free = (n1 - 1 - (hi1 - lo1)) * (n2 - 1 - (hi2 - lo2))
            if free <= 0:
                continue
            weight = n1 * n2 / free
        radii.append(np.sqrt(pixels[k] * h[0] * h[1] / np.pi))
        weights.append(weight)
    radii, weights = np.asarray(radii, float), np.asarray(weights, float)
    keep = radii >= float(min_radius)
    return Sections(axis, radii[keep], weights[keep], float(n0 * n1 * h[0] * n2 * h[1]))


def saltykov_matrix(n: int) -> np.ndarray:
    """``P[I - 1, J - 1]``: the probability that a particle of the 3D class ``I`` (largest
    section radius ``I delta``) is cut in a section of the 2D class ``J`` (radii between
    ``(J - 1) delta`` and ``J delta``), classes counted upward; its rows sum to 1."""
    i = np.arange(1, n + 1, dtype=float)[:, None]
    j = np.arange(1, n + 1, dtype=float)[None, :]
    upper = np.sqrt(np.clip(i**2 - (j - 1.0) ** 2, 0.0, None))
    lower = np.sqrt(np.clip(i**2 - j**2, 0.0, None))
    return np.where(j <= i, (upper - lower) / i, 0.0)


def section_scales(shape=(1.0, 1.0), rotation=None, axis: int = 2) -> tuple[float, float]:
    """``(kappa, h)``: an ellipsoid of short semi-axis ``a``, shape factors ``shape =
    (b/a, c/a)`` and axes ``rotation`` (columns: the directions of its axes a, b, c in the
    frame of the image; default x, y, z) is cut by the planes normal to ``axis`` in sections
    of equivalent radius at most ``kappa a``, and spans ``h a`` across them. From a fabric
    tensor ``f``: ``shape = (f.ratios[1], f.ratios[0])``, ``rotation = f.axes[:, ::-1]``."""
    l1, l2 = (float(v) for v in shape)
    r = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    s_hat = float(r[int(axis)] @ np.diag([1.0, l1 * l1, l2 * l2]) @ r[int(axis)])
    return float(np.sqrt(l1 * l2) / s_hat**0.25), float(2.0 * np.sqrt(s_hat))


@dataclass
class Unfolding:
    """A 3D size distribution recovered from sections: the short semi-axes of its classes
    (``radii``, on the mean scale of the directions) and their numbers per unit volume
    (``density``, the mean of the directions); per direction (``axes``), the densities
    (``directional``), the largest short semi-axis ``a_max`` (``scales``), the measured and
    refolded areal densities of the 2D classes and the relative L1 error of the refolding."""

    radii: np.ndarray
    density: np.ndarray
    directional: np.ndarray
    scales: np.ndarray
    measured: np.ndarray
    refolded: np.ndarray
    errors: np.ndarray
    axes: tuple[int, ...]

    def fractions(self) -> np.ndarray:
        """Number fractions of the classes."""
        return self.density / self.density.sum()

    def distribution(self) -> Discrete:
        """The number distribution of the short semi-axes (classes of negative density
        dropped), e.g. the ``seed_radius`` of a :class:`~morphogen.models.LevelSetFoam`."""
        keep = self.density > 0
        return Discrete(tuple(map(float, self.radii[keep])), tuple(map(float, self.density[keep])))


def _quantile(sections: Sections, q: float) -> float:
    """Weighted ``q``-quantile of the section radii."""
    order = np.argsort(sections.radii)
    cumulative = np.cumsum(sections.weights[order])
    k = min(int(np.searchsorted(cumulative, q * cumulative[-1])), len(order) - 1)
    return float(sections.radii[order][k])


def _unfold(measured, heights, p, classes: int):
    """Densities of the ``classes`` largest 3D classes from the areal densities of each
    direction (Saltykov's recursion), their mean and its refolding."""
    bins = p.shape[0]
    top = np.arange(bins - classes, bins)
    directional = np.zeros((len(measured), classes))
    for d, (n_a, h) in enumerate(zip(measured, heights, strict=True)):
        n_v = np.zeros(bins)
        for i in top[::-1]:
            n_v[i] = (n_a[i] - p[i + 1 :, i] @ (h[i + 1 :] * n_v[i + 1 :])) / (p[i, i] * h[i])
        directional[d] = n_v[top]
    mean = directional.mean(axis=0)
    refolded = np.array([(mean * h[top]) @ p[top] for h in heights])
    errors = np.abs(refolded - measured).sum(axis=1) / np.maximum(measured.sum(axis=1), 1e-300)
    return directional, mean, refolded, errors


def unfold_sections(
    sections,
    *,
    shape=(1.0, 1.0),
    rotation=None,
    bins: int = 20,
    classes: int | None = None,
    percentile: float = 99.0,
) -> Unfolding:
    """The size distribution of particles of a common shape (``shape``, ``rotation``: see
    :func:`section_scales`) from their sections normal to one or several axes (a
    :class:`Sections` or a list), as Degdeg & Dirrenberger: in each direction the sections
    up to the ``percentile`` of their radii, ``r_max``, fall in ``bins`` classes of width
    ``r_max / bins``, and the ``classes`` largest 3D classes (short semi-axes ``I / bins``
    of ``a_max = r_max / kappa``) come from Saltykov's recursion; their densities are
    averaged over the directions class by class. ``classes=None``: the number that
    minimizes the summed L1 error of refolding the mean through each direction (several
    directions: with one, the refolding of all the classes is exact)."""
    sections = [sections] if isinstance(sections, Sections) else list(sections)
    if classes is None and len(sections) < 2:
        raise ValueError("unfold_sections: give the number of classes for a single direction")
    p = saltykov_matrix(int(bins))
    measured, heights, scales = [], [], []
    for sec in sections:
        if not len(sec.radii):
            raise ValueError(f"unfold_sections: no sections normal to axis {sec.axis}")
        kappa, h = section_scales(shape, rotation, sec.axis)
        r_max = _quantile(sec, float(percentile) / 100.0)
        measured.append(sec.areal_density(np.linspace(0.0, r_max, int(bins) + 1)))
        a_max = r_max / kappa
        heights.append(h * a_max * np.arange(1, int(bins) + 1) / int(bins))
        scales.append(a_max)
    measured = np.asarray(measured)
    if classes is None:
        totals = [_unfold(measured, heights, p, c)[3].sum() for c in range(1, int(bins) + 1)]
        classes = int(np.argmin(totals)) + 1
    directional, mean, refolded, errors = _unfold(measured, heights, p, int(classes))
    top = np.arange(int(bins) - int(classes), int(bins)) + 1
    radii = float(np.mean(scales)) * top / int(bins)
    return Unfolding(
        radii,
        mean,
        directional,
        np.asarray(scales),
        measured,
        refolded,
        errors,
        tuple(s.axis for s in sections),
    )


def _mean_radii(sections) -> dict[int, float]:
    if isinstance(sections, dict):
        return {int(a): float(r) for a, r in sections.items()}
    return {s.axis: s.mean_radius() for s in sections}


def shape_consistency(sections, shape=(1.0, 1.0), rotation=None) -> np.ndarray:
    """The mean section radius over ``kappa`` in each direction (the axes of ``sections``,
    a list of :class:`Sections` or a mapping axis -> mean radius): ``(pi / 4) m2 / m1``
    of the short semi-axes in every direction when the particles have the given shape."""
    means = _mean_radii(sections)
    return np.array([r / section_scales(shape, rotation, a)[0] for a, r in means.items()])


def consistent_shape(sections, rotation=None) -> tuple[float, float]:
    """The shape factors ``(b/a, c/a)`` for which :func:`shape_consistency` is the same in
    the three directions: ``r_D^4 s_hat_D`` common to them, a linear system in the squared
    shape factors (``b/a = (r_x / r_y)^2``, ``c/a = (r_x / r_z)^2`` for the default axes).
    It is ill-conditioned for axes far from the planes of the grid."""
    means = _mean_radii(sections)
    if sorted(means) != [0, 1, 2]:
        raise ValueError("consistent_shape: the mean section radii normal to x, y and z")
    r = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    w = np.linalg.solve(r**2, 1.0 / np.array([means[a] for a in range(3)]) ** 4)
    if np.any(w <= 0):
        raise ValueError("consistent_shape: no ellipsoid of these axes fits the section radii")
    return float(np.sqrt(w[1] / w[0])), float(np.sqrt(w[2] / w[0]))
