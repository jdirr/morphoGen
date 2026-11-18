"""Phase fractions."""

from __future__ import annotations

import numpy as np

from morphogen.image import VoxelImage


def as_array(image) -> np.ndarray:
    """The array of an image: the data of a :class:`~morphogen.image.VoxelImage`, or any array-like
    as a NumPy array."""
    return image.data if isinstance(image, VoxelImage) else np.asarray(image)


def indicator(image, phase: int = 1) -> np.ndarray:
    """Indicator of ``phase`` (boolean), or the partial-volume field of a float image."""
    data = as_array(image)
    if data.dtype.kind == "f":
        if phase not in (0, 1):
            raise ValueError("fraction images only describe phases 0 and 1")
        return data if phase == 1 else 1.0 - data
    return data == phase


def volume_fraction(image, phase: int = 1) -> float:
    """Volume fraction of ``phase`` (mean of the partial volumes for float images)."""
    ind = indicator(image, phase)
    return float(np.mean(ind, dtype=np.float64))


def phase_fractions(image) -> dict[int, float]:
    """Volume fraction of every phase index present in an integer image."""
    data = as_array(image)
    if data.dtype.kind == "f":
        raise ValueError("phase_fractions needs an integer phase image")
    values, counts = np.unique(data, return_counts=True)
    return {int(v): float(c) / data.size for v, c in zip(values, counts, strict=True)}
