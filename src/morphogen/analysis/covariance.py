"""Covariance (two-point probability) functions of images.

``C(h) = P(x in A, x + h in A)`` is estimated with fast Fourier transforms: circularly
for periodic images, with zero padding and normalization by the number of pairs
otherwise. ``C(0)`` is the volume fraction and ``C(h) -> p^2`` when correlations vanish.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft

from morphogen import settings
from morphogen.analysis.basic import indicator
from morphogen.image import VoxelImage

_ROWS = 1 << 15


@dataclass
class AxisFunction:
    """A function of the lags along the x, y and z axes (the covariance, the lineal-path
    and two-point cluster functions of :mod:`morphogen.analysis.correlation`)."""

    lags: dict[str, np.ndarray]
    values: dict[str, np.ndarray]
    volume_fraction: float

    def mean(self) -> tuple[np.ndarray, np.ndarray]:
        """Average of the three axes (requires cubic voxels and equal lengths)."""
        n = min(len(v) for v in self.values.values())
        lags = self.lags["x"][:n]
        return lags, np.mean([v[:n] for v in self.values.values()], axis=0)


class AxisCovariance(AxisFunction):
    """Covariance along the x, y and z axes."""


def _axis_correlation(field: np.ndarray, axis: int, max_lag: int, periodic: bool) -> np.ndarray:
    x = np.moveaxis(field, axis, -1)
    n = x.shape[-1]
    rows = x.reshape(-1, n)
    size = n if periodic else int(sfft.next_fast_len(2 * n))
    acc = np.zeros(max_lag + 1)
    for start in range(0, rows.shape[0], _ROWS):
        block = rows[start : start + _ROWS].astype(np.float64)
        f = sfft.rfft(block, n=size, axis=-1, workers=settings.workers())
        corr = sfft.irfft(f * np.conj(f), n=size, axis=-1, workers=settings.workers())
        acc += corr[:, : max_lag + 1].sum(axis=0)
    pairs = rows.shape[0] * (n if periodic else (n - np.arange(max_lag + 1)))
    return acc / pairs


def covariance(image, phase: int = 1, *, max_lag: int | None = None, periodic=None):
    """Covariance of ``phase`` along the three axes, for lags 0..max_lag voxels.

    ``max_lag`` defaults to half the image size along each axis.
    """
    field = indicator(image, phase)
    spacing = image.spacing if isinstance(image, VoxelImage) else (1.0, 1.0, 1.0)
    if periodic is None:
        periodic = image.periodic if isinstance(image, VoxelImage) else False
    lags, values = {}, {}
    for axis, name in enumerate("xyz"):
        n = field.shape[axis]
        m = n // 2 if max_lag is None else min(int(max_lag), n - 1)
        values[name] = _axis_correlation(field, axis, m, periodic)
        lags[name] = np.arange(m + 1) * spacing[axis]
    return AxisCovariance(lags, values, float(np.mean(field, dtype=np.float64)))


def covariance_map(image, phase: int = 1) -> np.ndarray:
    """Full 3D covariance of a periodic image, zero lag at the centre (fftshift order)."""
    field = indicator(image, phase).astype(np.float64)
    f = sfft.rfftn(field, workers=settings.workers())
    corr = sfft.irfftn(f * np.conj(f), s=field.shape, workers=settings.workers()) / field.size
    return sfft.fftshift(corr)
