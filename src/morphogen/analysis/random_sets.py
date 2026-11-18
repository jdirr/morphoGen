"""Random sets and random functions probed by structuring elements (Matheron 1975; Serra
1982; Jeulin 2021).

A structuring element ``K`` placed at every point ``x`` probes a set ``A``: the dilation
``A (+) K' = {x : K_x hits A}`` and the erosion ``A (-) K' = {x : K_x inside A}`` (``K'``
the transposed element). A random closed set is characterized by its Choquet capacity
``T(K) = P(K hits A) = 1 - Q(K)``, ``Q(K) = P(K inside A^c)``, which for an ergodic set is
the volume fraction of ``A (+) K'`` in one realization. Pairs of points give the
covariance (``T({0, h}) = 2 p - C(h)``), segments the lineal path of the complement,
balls the spherical contact distribution. A grey-level image ``Z`` (a random function) is
probed by its erosion by pairs of points: the morphological covariance
``E[min(Z(x), Z(x + h))]``; the Minkowski functionals of its excursion sets ``{Z > u}``
are curves of the level ``u``.

Periodic images are probed on the torus; other images by minus sampling: only the
positions where the element lies inside the box are counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from morphogen.analysis.basic import as_array, indicator
from morphogen.analysis.covariance import AxisFunction
from morphogen.image import VoxelImage

ELEMENTS = ("ball", "segment", "pair", "cube")


@dataclass
class Capacity:
    """Choquet capacity ``T(K)`` of a phase for structuring elements of growing size: the
    radius of balls, the length of segments, the distance of pairs of points, the edge of
    cubes (``sizes``, in length units)."""

    element: str
    sizes: np.ndarray
    values: np.ndarray
    volume_fraction: float

    @property
    def complement(self) -> np.ndarray:
        """``Q(K) = P(K inside A^c) = 1 - T(K)``."""
        return 1.0 - self.values

    def contact_distribution(self) -> np.ndarray:
        """``H(r) = 1 - Q(K_r) / q``: the distribution function of the distance from a
        point of the complement to the phase, measured with the element (the spherical
        contact distribution for balls, the linear one for segments)."""
        q = 1.0 - self.volume_fraction
        if q <= 0.0:
            return np.full_like(self.values, np.nan)
        return 1.0 - self.complement / q


class MorphologicalCovariance(AxisFunction):
    """Mean of the erosion (or dilation) of a grey-level image by pairs of points along
    the x, y and z axes; ``volume_fraction`` holds the mean of the image."""


def _periodic(image, periodic) -> bool:
    if periodic is None:
        return bool(image.periodic) if isinstance(image, VoxelImage) else False
    return bool(periodic)


def _spacing(image) -> np.ndarray:
    spacing = image.spacing if isinstance(image, VoxelImage) else (1.0, 1.0, 1.0)
    return np.asarray(spacing, dtype=float)


def _direction(direction) -> np.ndarray:
    d = np.asarray(direction)
    if d.shape != (3,) or not np.all(d == np.round(d)) or not np.any(d):
        raise ValueError("direction: three integers, not all zero (a direction of the grid)")
    return d.astype(np.int64)


def _shifted(x: np.ndarray, offset, periodic: bool) -> np.ndarray:
    """``y[i] = x[i + offset]``; outside the box ``False`` (or zero) if not periodic."""
    offset = [int(o) for o in offset]
    if periodic:
        return np.roll(x, [-o for o in offset], axis=(0, 1, 2))
    out = np.zeros_like(x)
    src, dst = [], []
    for o, n in zip(offset, x.shape, strict=True):
        if abs(o) >= n:
            return out
        src.append(slice(o, n) if o >= 0 else slice(0, n + o))
        dst.append(slice(0, n - o) if o >= 0 else slice(-o, n))
    out[tuple(dst)] = x[tuple(src)]
    return out


def _fits(shape, offset) -> int:
    """Number of positions ``x`` of the box with ``x + offset`` in the box."""
    return int(np.prod([max(n - abs(int(o)), 0) for n, o in zip(shape, offset, strict=True)]))


def _ball(mask, spacing, periodic, radii) -> np.ndarray:
    comp = ~mask
    if not mask.any():
        return np.zeros(len(radii))
    if not comp.any():
        return np.ones(len(radii))
    rmax = float(np.max(radii))
    if periodic:
        pad = min(int(np.ceil(rmax / spacing.min())) + 1, max(mask.shape))
        big = np.pad(comp, pad, mode="wrap")
        dist = ndimage.distance_transform_edt(big, sampling=spacing)[pad:-pad, pad:-pad, pad:-pad]
        d = np.sort(dist[comp])
        p = float(mask.mean())
        hits = np.searchsorted(d, np.asarray(radii) * (1.0 + 1e-12), side="right")
        return p + hits / mask.size
    # minus sampling: the balls that lie in the box (their centre at r from the faces)
    dist = ndimage.distance_transform_edt(comp, sampling=spacing)
    margins = [
        np.minimum(np.arange(n), np.arange(n)[::-1]) * h
        for n, h in zip(mask.shape, spacing, strict=True)
    ]
    margin = np.minimum(
        np.minimum(margins[0][:, None, None], margins[1][None, :, None]), margins[2][None, None, :]
    )
    out = np.empty(len(radii))
    for k, r in enumerate(radii):
        inside = margin >= r * (1.0 - 1e-12)
        out[k] = float(np.mean(dist[inside] <= r * (1.0 + 1e-12))) if inside.any() else np.nan
    return out


def _segment(mask, d, periodic, counts) -> np.ndarray:
    """T of digital segments of ``counts`` points along the grid direction ``d``."""
    comp = ~mask
    run = comp.copy()
    wanted = {int(n): k for k, n in enumerate(counts)}
    out = np.empty(len(counts))
    for n in range(1, int(max(counts)) + 1):
        if n > 1:
            run &= _shifted(comp, (n - 1) * d, periodic)
        if n in wanted:
            total = mask.size if periodic else _fits(mask.shape, (n - 1) * d)
            out[wanted[n]] = 1.0 - run.sum() / total if total else np.nan
    return out


def _pair(mask, d, periodic, steps) -> np.ndarray:
    comp = ~mask
    out = np.empty(len(steps))
    for k, s in enumerate(steps):
        both = comp & _shifted(comp, int(s) * d, periodic)
        total = mask.size if periodic else _fits(mask.shape, int(s) * d)
        out[k] = 1.0 - both.sum() / total if total else np.nan
    return out


def _cube(mask, periodic, counts) -> np.ndarray:
    comp = (~mask).view(np.uint8)
    out = np.empty(len(counts))
    for k, n in enumerate(counts):
        n = int(n)
        if periodic:
            eroded = ndimage.minimum_filter(comp, size=n, mode="wrap")
            out[k] = 1.0 - eroded.mean()
        else:
            eroded = ndimage.minimum_filter(comp, size=n, mode="constant", cval=0)
            total = _fits(mask.shape, (n - 1,) * 3)
            out[k] = 1.0 - eroded.sum() / total if total else np.nan
    return out


def choquet_capacity(
    image, element: str = "ball", sizes=None, *, phase: int = 1, direction=(1, 0, 0), periodic=None
) -> Capacity:
    """Choquet capacity ``T(K) = P(K hits A)`` of ``phase`` (the set ``A``) for structuring
    elements ``K`` of the given ``sizes`` (length units), estimated as the volume
    fraction of the dilation of ``A`` by ``K``.

    ``element``: ``"ball"`` (radius; digital balls of the voxel centres closer than the
    radius, through the Euclidean distance map), ``"segment"`` (length, along the grid
    ``direction``, e.g. ``(1, 1, 0)``), ``"pair"`` (two points distant of the size along
    ``direction``: ``T = 2 p - C(h)``) or ``"cube"`` (edge). Segments, pairs and cubes have
    points on the grid: their sizes are rounded to whole steps and the measured sizes are
    returned. ``sizes`` defaults to one voxel steps up to a quarter of the box (half of
    it for segments and pairs).
    """
    if element not in ELEMENTS:
        raise ValueError(f"element: one of {', '.join(ELEMENTS)}")
    mask = indicator(image, phase)
    spacing = _spacing(image)
    per = _periodic(image, periodic)
    extent = float(np.min(np.asarray(mask.shape) * spacing))
    p = float(mask.mean())
    if element == "ball":
        radii = (
            np.arange(0.0, 0.25 * extent + 1e-9, spacing.min())
            if sizes is None
            else np.atleast_1d(np.asarray(sizes, dtype=float))
        )
        if np.any(radii < 0):
            raise ValueError("sizes: non-negative radii")
        return Capacity(element, radii, _ball(mask, spacing, per, radii), p)
    if element == "cube":
        step = float(spacing.min())
        limit = 0.25 * extent
    else:
        d = _direction(direction)
        step = float(np.linalg.norm(d * spacing))
        limit = 0.5 * extent
    if sizes is None:
        steps = np.arange(0, int(limit / step) + 1)
    else:
        steps = np.unique(np.rint(np.atleast_1d(np.asarray(sizes, dtype=float)) / step)).astype(int)
        if np.any(steps < 0):
            raise ValueError("sizes: non-negative sizes")
    if element == "segment":
        values = _segment(mask, d, per, steps + 1)
    elif element == "pair":
        values = _pair(mask, d, per, steps)
    else:
        values = _cube(mask, per, steps + 1)
    return Capacity(element, steps * step, values, p)


def dilation(
    image, element: str, size: float, *, phase: int = 1, direction=(1, 0, 0), periodic=None
):
    """Indicator of ``A (+) K'``: the voxels ``x`` where ``K_x`` hits the phase (``K`` as in
    :func:`choquet_capacity`, the element anchored at its first point or its centre)."""
    return _transform(image, element, size, phase, direction, periodic, dilate=True)


def erosion(
    image, element: str, size: float, *, phase: int = 1, direction=(1, 0, 0), periodic=None
):
    """Indicator of ``A (-) K'``: the voxels ``x`` where ``K_x`` lies inside the phase
    (outside a non-periodic box counts as outside the phase)."""
    return _transform(image, element, size, phase, direction, periodic, dilate=False)


def opening(
    image, element: str, size: float, *, phase: int = 1, direction=(1, 0, 0), periodic=None
):
    """``(A (-) K') (+) K``: the union of the translates of ``K`` inside the phase."""
    per = _periodic(image, periodic)
    eroded = erosion(image, element, size, phase=phase, direction=direction, periodic=per)
    return _transform(eroded, element, size, 1, direction, per, dilate=True, transpose=True)


def closing(
    image, element: str, size: float, *, phase: int = 1, direction=(1, 0, 0), periodic=None
):
    """``(A (+) K') (-) K``: the complement of the opening of the complement."""
    per = _periodic(image, periodic)
    dilated = dilation(image, element, size, phase=phase, direction=direction, periodic=per)
    return ~_transform(~dilated, element, size, 1, direction, per, dilate=True, transpose=True)


def _footprint(element: str, size: float, spacing, direction) -> np.ndarray:
    """Offsets (k x 3 integers) of the points of the element."""
    if element == "ball":
        reach = np.floor(size / spacing + 1e-9).astype(int)
        grid = np.stack(
            np.meshgrid(*[np.arange(-m, m + 1) for m in reach], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        return grid[np.linalg.norm(grid * spacing, axis=1) <= size * (1.0 + 1e-12)]
    if element == "cube":
        n = int(np.rint(size / spacing.min())) + 1
        return np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    d = _direction(direction)
    k = int(np.rint(size / np.linalg.norm(d * spacing)))
    if element == "pair":
        return np.array([[0, 0, 0], k * d])
    return np.arange(k + 1)[:, None] * d[None, :]


def _transform(image, element, size, phase, direction, periodic, *, dilate, transpose=False):
    if element not in ELEMENTS:
        raise ValueError(f"element: one of {', '.join(ELEMENTS)}")
    x = (
        indicator(image, phase)
        if not isinstance(image, np.ndarray) or image.dtype != bool
        else image
    )
    per = _periodic(image, periodic)
    offsets = _footprint(element, float(size), _spacing(image), direction)
    if transpose:
        offsets = -offsets
    out = np.zeros_like(x) if dilate else np.ones_like(x)
    for o in offsets:
        s = _shifted(x, o, per)
        if dilate:
            out |= s
        else:
            if not per:  # outside the box: outside the phase
                s = s & _shifted(np.ones_like(x), o, per)
            out &= s
    return out


def morphological_covariance(
    image, *, max_lag: int | None = None, periodic=None, operation: str = "erosion"
) -> MorphologicalCovariance:
    """``E[min(Z(x), Z(x + h))]`` (erosion by pairs of points; ``operation="dilation"``:
    the maximum) of a grey-level image along the three axes, lags 0..max_lag voxels
    (default half the box). For a binary image the erosion gives the covariance."""
    if operation not in ("erosion", "dilation"):
        raise ValueError('operation: "erosion" or "dilation"')
    z = np.asarray(as_array(image), dtype=np.float64)
    op = np.minimum if operation == "erosion" else np.maximum
    per = _periodic(image, periodic)
    spacing = _spacing(image)
    lags, values = {}, {}
    for axis, name in enumerate("xyz"):
        n = z.shape[axis]
        m = n // 2 if max_lag is None else min(int(max_lag), n - 1)
        v = np.empty(m + 1)
        for k in range(m + 1):
            if per:
                v[k] = op(z, np.roll(z, -k, axis=axis)).mean()
            else:
                a = np.take(z, np.arange(0, n - k), axis=axis)
                b = np.take(z, np.arange(k, n), axis=axis)
                v[k] = op(a, b).mean()
        values[name] = v
        lags[name] = np.arange(m + 1) * spacing[axis]
    return MorphologicalCovariance(lags, values, float(z.mean()))


def minkowski_curves(image, thresholds, *, periodic=None) -> dict[str, np.ndarray]:
    """Densities of the four Minkowski functionals of the excursion sets ``{Z > u}`` of a
    grey-level image for the levels ``thresholds``: the volume fraction, the specific
    surface, the integral of mean curvature and the Euler characteristic per unit
    volume."""
    from morphogen.analysis.morphology import minkowski

    z = np.asarray(as_array(image))
    per = _periodic(image, periodic)
    spacing = tuple(_spacing(image))
    levels = np.atleast_1d(np.asarray(thresholds, dtype=float))
    out = {
        k: np.empty(len(levels))
        for k in ("volume_fraction", "specific_surface", "mean_curvature", "euler_density")
    }
    for i, u in enumerate(levels):
        m = minkowski(VoxelImage((z > u).astype(np.uint8), spacing=spacing, periodic=per), 1)
        for k in out:
            out[k][i] = getattr(m, k)
    return {"thresholds": levels, **out}
