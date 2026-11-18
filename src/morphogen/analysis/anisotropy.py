"""Anisotropy of a phase: the mean intercept length (MIL) along directions and its fabric
tensor (Whitehouse 1974; Harrigan & Mann 1984), as BoneJ computes it for trabecular bone
and Degdeg & Dirrenberger for an open-cell copper foam.

Along a direction ``d``, the test lines cross the boundary of the phase ``P_L(d)`` times
per unit length; the mean intercept length of the phase is ``MIL(d) = 2 V_V / P_L(d)``,
the mean length of its chords along ``d``. The MIL of an ellipsoidal structure follows an
ellipsoid: ``1 / MIL(d)^2 = d . M . d``, the fabric tensor ``M`` fitted by least squares to
the 13 directions of the grid (the three axes, the six face and the four body diagonals,
in physical units for anisotropic voxels); its principal MILs ``1 / sqrt(m_i)`` give the
anisotropy ratios and the degree of anisotropy ``1 - MIL_min / MIL_max``.

The covariance of a cellular phase passes through a first minimum below its plateau
``p^2`` before reaching it (the hole effect), at lags of the order of the cell radii:
Degdeg & Dirrenberger take these lags along the three axes as a second measure of the
aspect of the cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morphogen.analysis.basic import indicator
from morphogen.analysis.covariance import covariance
from morphogen.analysis.surface import DIRECTIONS_13, transitions
from morphogen.image import VoxelImage


@dataclass
class Fabric:
    """The MIL along the ``directions`` (unit vectors), the fabric tensor ``M``, its
    principal MILs (decreasing) and their ``axes`` (columns), the degree of anisotropy."""

    directions: np.ndarray
    mil: np.ndarray
    tensor: np.ndarray
    principal: np.ndarray
    axes: np.ndarray

    @property
    def ratios(self) -> np.ndarray:
        """Principal MILs over the smallest one (the anisotropy ratios)."""
        return self.principal / self.principal[-1]

    @property
    def degree(self) -> float:
        """Degree of anisotropy ``1 - MIL_min / MIL_max`` (0: isotropic)."""
        return float(1.0 - self.principal[-1] / self.principal[0])


def mean_intercept_length(image, phase: int = 1, *, periodic=None) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions (13 x 3) and the mean intercept length of ``phase`` along each."""
    x = indicator(image, phase)
    spacing = np.asarray(image.spacing if isinstance(image, VoxelImage) else (1.0, 1.0, 1.0), float)
    if periodic is None:
        periodic = bool(image.periodic) if isinstance(image, VoxelImage) else False
    fraction = float(x.mean())
    dirs, mils = [], []
    for d in DIRECTIONS_13:
        if not periodic and any(abs(c) >= n for c, n in zip(d, x.shape, strict=True)):
            continue
        n_cross, n_pairs = transitions(x, d, periodic)
        vec = np.asarray(d, dtype=float) * spacing
        step = float(np.linalg.norm(vec))
        p_l = n_cross / (n_pairs * step) if n_pairs else 0.0
        dirs.append(vec / step)
        mils.append(2.0 * fraction / p_l if p_l > 0 else np.inf)
    return np.asarray(dirs), np.asarray(mils)


def fabric_tensor(image, phase: int = 1, *, periodic=None) -> Fabric:
    """The MIL fabric tensor of ``phase`` (see the module)."""
    dirs, mil = mean_intercept_length(image, phase, periodic=periodic)
    finite = np.isfinite(mil)
    if finite.sum() < 6:
        raise ValueError("fabric_tensor: the phase crosses too few lines")
    u = dirs[finite]
    rows = np.column_stack(
        [
            u[:, 0] ** 2,
            u[:, 1] ** 2,
            u[:, 2] ** 2,
            2 * u[:, 1] * u[:, 2],
            2 * u[:, 2] * u[:, 0],
            2 * u[:, 0] * u[:, 1],
        ]
    )
    m, *_ = np.linalg.lstsq(rows, 1.0 / mil[finite] ** 2, rcond=None)
    tensor = np.array([[m[0], m[5], m[4]], [m[5], m[1], m[3]], [m[4], m[3], m[2]]])
    values, vectors = np.linalg.eigh(tensor)
    principal = 1.0 / np.sqrt(np.clip(values, 1e-300, None))  # increasing values: decreasing MILs
    return Fabric(dirs, mil, tensor, principal, vectors)


@dataclass
class HoleEffect:
    """The first minimum of the covariance below its ``plateau`` ``p^2`` along the x, y
    and z axes: its ``lags`` (physical units; NaN where the covariance has none) and
    ``values``."""

    lags: np.ndarray
    values: np.ndarray
    plateau: float

    @property
    def ratios(self) -> np.ndarray:
        """The lags in decreasing order over the smallest (``c/a``, ``b/a``, 1)."""
        lags = np.sort(self.lags)[::-1]
        return lags / lags[-1]


def covariance_minima(
    image, phase: int = 1, *, max_lag: int | None = None, periodic=None
) -> HoleEffect:
    """The first local minimum of the covariance of ``phase`` along each axis after it
    falls below its plateau, refined by the parabola through the three nearest lags."""
    cov = covariance(image, phase, max_lag=max_lag, periodic=periodic)
    plateau = cov.volume_fraction**2
    lags, values = np.full(3, np.nan), np.full(3, np.nan)
    for k, name in enumerate("xyz"):
        c, h = cov.values[name], cov.lags[name]
        below = np.flatnonzero(c < plateau)
        if below.size == 0:
            continue
        for i in range(max(int(below[0]), 1), len(c) - 1):
            if c[i] <= c[i - 1] and c[i] <= c[i + 1]:
                left, mid, right = c[i - 1], c[i], c[i + 1]
                curvature = left - 2.0 * mid + right
                shift = 0.5 * (left - right) / curvature if curvature > 0 else 0.0
                lags[k] = h[i] + shift * (h[1] - h[0])
                values[k] = mid - 0.25 * (left - right) * shift
                break
    return HoleEffect(lags, values, plateau)
