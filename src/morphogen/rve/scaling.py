"""Statistical size of the representative volume element (RVE).

Following Matheron (1971), Kanit et al. (2003) and Lantuéjoul (1991), the ensemble
variance of the average of a property Z over a volume V scales as

    D_Z^2(V) = D_Z^2 (A3 / V)          (finite integral range A3)
    D_Z^2(V) = K V^(-gamma)            (general power law, K = D_Z^2 A3*^gamma)

With n realizations of volume V, the relative error on the mean is
eps_rel = 2 D_Z(V) / (Zbar sqrt(n)), hence the RVE volume

    V_RVE = (4 K / (eps_rel^2 Zbar^2 n))^(1 / gamma).

"Volumes" may equally be lengths or areas (1D/2D samples, e.g. Bironeau et al. 2016), the
edge of a sample then ``V^(1/d)`` in dimension ``d`` (the side of a square of area ``V``).

Shifted tilings (the window method on a periodic image, ``shifts > 1``). The ``T`` tilings
of the windows of edge ``w`` shifted by the offsets ``k w // shifts`` along each axis are
translates of the first on the torus: by stationarity the windows of each have the joint
law of the unshifted ones, and its variance the law of theirs. The mean over all the
windows of all the tilings is the mean of the tiling means, each the image mean when the
edges of the image are multiples of ``w``. ``D^2(V)`` is the variance pooled within the
tilings, ``sum_t sum_(k in t) (Z_k - Zbar_t)^2 / sum_t (n_t - 1)``, with ``n_t = n``
windows in each the mean of their unbiased variances: of the expectation of the variance
of one tiling (unbiased under the hypothesis that makes that one unbiased, its windows
uncorrelated as for ``V >> A3``), of a smaller scatter. Not the variance of the ``T n``
windows about their mean with ``T n - 1`` degrees of freedom, lower by the factor ``T (n -
1) / (T n - 1)``: the windows of different tilings overlap, so that their mean (the image
mean) is known no better than from one tiling, the deviations of each tiling summing to
zero. The count behind the uncertainties stays ``n``, the windows of one tiling (the
independent ones): the shifts leave the mean unchanged, whose standard error stays ``D(V) /
sqrt(n)``; the relative errors, the Student tests of the components of the tensors and the
weights ``(n - 1) / 2`` of the fits count ``n`` too, conservatively for the variance, to
which the overlapping windows add correlated, not independent, samples. The fits and the
RVE sizes follow from these variances, unchanged.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as _stats

from morphogen.analysis.basic import as_array, indicator


def point_variance(fraction: float, z1: float = 1.0, z0: float = 0.0) -> float:
    """D_Z^2 = p (1 - p) (Z1 - Z0)^2 of a two-phase medium (p: fraction of phase 1)."""
    return fraction * (1.0 - fraction) * (z1 - z0) ** 2


def point_variance_multiphase(fractions, values) -> float:
    """D_Z^2 = sum_i P_i (Z_i - Zbar)^2 (Yang et al. 2019, eq. 19)."""
    p = np.asarray(fractions, dtype=float)
    z = np.asarray(values, dtype=float)
    zbar = np.sum(p * z) / np.sum(p)
    return float(np.sum(p * (z - zbar) ** 2) / np.sum(p))


@dataclass(frozen=True)
class SizeStatistics:
    """Ensemble statistics of a property for several sample sizes.

    ``dimension``: that of the samples, whose measures the volumes are (3, cubes; 2, the
    squares of a planar image, their areas), their edges ``V^(1/dimension)``
    (:meth:`edges`). ``tilings``: the number of shifted tilings of each size by the window
    method (:func:`window_statistics`), ``n`` then the windows of one tiling and
    ``variance`` their variance pooled within the tilings (see the module); None: one
    tiling, or realizations.
    """

    volume: np.ndarray
    n: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    dimension: int = 3
    tilings: np.ndarray | None = None

    def edges(self) -> np.ndarray:
        """The edges of the samples, ``V^(1/dimension)``: of cubes, of squares."""
        return np.array([float(v) ** (1.0 / self.dimension) for v in self.volume])

    def relative_error(self) -> np.ndarray:
        """``eps_rel = 2 sqrt(D^2(V) / n) / |Zbar|`` for each size."""
        return 2.0 * np.sqrt(self.variance / self.n) / np.abs(self.mean)

    def rows(self) -> list[dict]:
        """One row per size: its volume, its count ``n`` (and its tilings when shifted), the
        mean, the variance and the relative error."""
        err = self.relative_error()
        columns = zip(self.volume, self.n, self.mean, self.variance, err, strict=True)
        rows = []
        for i, (v, k, m, s, e) in enumerate(columns):
            row = {"volume": float(v), "n": int(k)}
            if self.tilings is not None:
                row["tilings"] = int(self.tilings[i])
            row.update(mean=float(m), variance=float(s), relative_error=float(e))
            rows.append(row)
        return rows


def ensemble_statistics(volumes, values) -> SizeStatistics:
    """Group values by volume: mean and unbiased variance per size."""
    v = np.asarray(volumes, dtype=float)
    z = np.asarray(values, dtype=float)
    if v.shape != z.shape:
        raise ValueError("volumes and values must have the same length")
    ok = np.isfinite(z)
    v, z = v[ok], z[ok]
    sizes = np.unique(v)
    n = np.array([np.count_nonzero(v == s) for s in sizes])
    mean = np.array([z[v == s].mean() for s in sizes])
    var = np.array(
        [z[v == s].var(ddof=1) if c > 1 else np.nan for s, c in zip(sizes, n, strict=True)]
    )
    return SizeStatistics(sizes, n, mean, var)


@dataclass(frozen=True)
class ScalingLaw:
    """D_Z^2(V) = K V^(-gamma), fitted in log-log coordinates."""

    K: float
    gamma: float
    gamma_fixed: bool = False
    r2: float = math.nan
    n_points: int = 0
    volume_min: float = math.nan
    volume_max: float = math.nan
    #: standard errors of gamma (free fits with at least 3 sizes) and of ln K
    gamma_se: float = math.nan
    log_k_se: float = math.nan

    def _t(self, level: float, n_params: int) -> float:
        dof = self.n_points - n_params
        return float(_stats.t.ppf(0.5 + level / 2.0, dof)) if dof > 0 else math.nan

    def gamma_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Confidence interval of gamma (Student t, from the scatter about the fit)."""
        if self.gamma_fixed:
            return self.gamma, self.gamma
        half = self._t(level, 2) * self.gamma_se
        return self.gamma - half, self.gamma + half

    def k_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Confidence interval of K."""
        half = self._t(level, 1 if self.gamma_fixed else 2) * self.log_k_se
        return self.K * math.exp(-half), self.K * math.exp(half)

    def variance(self, volume) -> np.ndarray:
        return self.K * np.asarray(volume, dtype=float) ** (-self.gamma)

    def relative_error(self, volume, n, mean) -> np.ndarray:
        return 2.0 * np.sqrt(self.variance(volume) / np.asarray(n, dtype=float)) / abs(mean)

    def rve_volume(self, eps_rel, mean: float, n=1) -> np.ndarray:
        """Volume for which n realizations give the mean within ``eps_rel`` (relative)."""
        eps = np.asarray(eps_rel, dtype=float)
        n = np.asarray(n, dtype=float)
        return (4.0 * self.K / (eps**2 * mean**2 * n)) ** (1.0 / self.gamma)

    def realizations(self, volume, eps_rel, mean: float) -> np.ndarray:
        """Number of realizations of the given volume needed for ``eps_rel``."""
        eps = np.asarray(eps_rel, dtype=float)
        return 4.0 * self.variance(volume) / (eps**2 * mean**2)

    def integral_range(self, point_var: float) -> float:
        """A3* = (K / D_Z^2)^(1/gamma) (the integral range A3 when gamma = 1)."""
        return (self.K / point_var) ** (1.0 / self.gamma)

    def to_dict(self) -> dict:
        return {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in asdict(self).items()
        }


def fit_scaling_law(
    volumes, variances, *, gamma: float | None = None, volume_range=None, weights=None
) -> ScalingLaw:
    """Least-squares fit of log D^2 = log K - gamma log V (gamma free or imposed)."""
    v = np.asarray(volumes, dtype=float)
    d2 = np.asarray(variances, dtype=float)
    ok = np.isfinite(v) & np.isfinite(d2) & (v > 0) & (d2 > 0)
    if volume_range is not None:
        lo, hi = volume_range
        ok &= (v >= (lo if lo is not None else -np.inf)) & (v <= (hi if hi is not None else np.inf))
    v, d2 = v[ok], d2[ok]
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float)[ok]
    needed = 1 if gamma is not None else 2
    if len(v) < needed:
        raise ValueError(f"fit_scaling_law: need at least {needed} sizes with a positive variance")
    x, y = np.log(v), np.log(d2)
    m = len(v)
    xm = np.average(x, weights=w)
    sxx = np.sum(w * (x - xm) ** 2)
    if gamma is None:
        ym = np.average(y, weights=w)
        if sxx == 0:
            raise ValueError("fit_scaling_law: need at least two different volumes")
        slope = np.sum(w * (x - xm) * (y - ym)) / sxx
        g, log_k = -slope, ym - slope * xm
    else:
        g = float(gamma)
        log_k = np.average(y + g * x, weights=w)
    pred = log_k - g * x
    ss_res = np.sum(w * (y - pred) ** 2)
    ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    n_params = 1 if gamma is not None else 2
    gamma_se = log_k_se = math.nan
    if m > n_params:
        s2 = ss_res / (m - n_params)
        if gamma is None:
            gamma_se = math.sqrt(s2 / sxx)
            log_k_se = math.sqrt(s2 * (1.0 / np.sum(w) + xm**2 / sxx))
        else:
            log_k_se = math.sqrt(s2 / np.sum(w))
    return ScalingLaw(
        K=float(math.exp(log_k)),
        gamma=float(g),
        gamma_fixed=gamma is not None,
        r2=float(r2),
        n_points=m,
        volume_min=float(v.min()),
        volume_max=float(v.max()),
        gamma_se=float(gamma_se),
        log_k_se=float(log_k_se),
    )


def fit_statistics(stats: SizeStatistics, *, gamma=None, volume_range=None) -> ScalingLaw:
    """The scaling law of the variance with the volume (:func:`fit_scaling_law`) fitted to the
    statistics of a study by size, ``gamma`` given or fitted, within ``volume_range``."""
    return fit_scaling_law(stats.volume, stats.variance, gamma=gamma, volume_range=volume_range)


def rve_volume_classic(point_var, integral_range, eps_rel, mean, n=1) -> float:
    """V_RVE = 4 D_Z^2 A3 / (eps^2 Zbar^2 n) (Kanit et al. 2003)."""
    return 4.0 * point_var * integral_range / (eps_rel**2 * mean**2 * n)


def _shift_count(shifts, periodic: bool) -> int:
    """The number of tilings per axis of the window method: a positive integer, more than
    one only on a periodic image (the shifted tilings wrap around it). The refusals of
    :func:`window_statistics`, :func:`~morphogen.rve.windows.window_tensors` and the RVE
    tool of the interface."""
    if isinstance(shifts, bool) or not isinstance(shifts, (int, np.integer)) or shifts < 1:
        raise ValueError(f"shifts: a positive integer (tilings per axis), got {shifts!r}")
    if shifts > 1 and not periodic:
        raise ValueError("shifts: the shifted tilings wrap around the image, a periodic one")
    return int(shifts)


def _offsets(edge: int, shifts: int, planar: bool) -> np.ndarray:
    """The offsets (voxels, ``(t, 3)``) of the tilings of the windows of ``edge`` voxels
    shifted ``shifts`` times along each axis that they tile (x, y, and z but across a planar
    image): the distinct ``k edge // shifts``, ``k < shifts``, all their combinations, the
    unshifted tiling first."""
    steps = sorted({k * edge // shifts for k in range(shifts)})
    product = itertools.product(steps, steps, [0] if planar else steps)
    return np.array(list(product), dtype=np.int64)


def _pooled_variance(values: np.ndarray, tilings: np.ndarray) -> np.ndarray:
    """The variance of ``values`` (along their first axis) pooled within their ``tilings``
    (the tiling of each value), ``sum_t sum_(k in t) (z_k - zbar_t)^2 / sum_t (n_t - 1)``
    (NaN without a tiling of two values; see the module)."""
    squares = np.zeros(values.shape[1:])
    dof = 0
    for t in np.unique(tilings):
        x = values[tilings == t]
        if len(x) > 1:
            squares = squares + ((x - x.mean(axis=0)) ** 2).sum(axis=0)
            dof += len(x) - 1
    return squares / dof if dof else np.full(values.shape[1:], np.nan)


def _moments(values: np.ndarray, tilings: np.ndarray | None = None) -> tuple:
    """The count, the mean and the unbiased variance of values: of realizations, or of the
    windows of one tiling (``tilings`` None or a single one); of the windows of shifted
    tilings (``tilings`` the tiling of each), the windows of one (their mean count), the
    mean of all and their variance pooled within the tilings (see the module)."""
    groups = 1 if tilings is None else np.unique(tilings).size
    if groups == 1:
        return values.size, values.mean(), values.var(ddof=1) if values.size > 1 else np.nan
    return values.size // groups, values.mean(), float(_pooled_variance(values, tilings))


def window_statistics(
    image, window_sizes, *, phase: int | None = 1, spacing=None, shifts: int = 1, periodic=None
):
    """Window method: statistics of the mean over non-overlapping cubic sub-volumes.

    ``window_sizes`` are window edges in voxels. With ``phase`` the averaged quantity
    is the indicator of that phase (volume fraction); with ``phase=None`` the image
    values themselves (e.g. a local field computed by an FFT or FE solver). Across a
    planar image (``nz = 1``) the windows are squares of ``w`` pixels, whose measure is
    their area (``dimension = 2``, their edges the roots of their areas). ``shifts``: the
    tilings per axis, 1 (by default) the tiling from the origin of the image, the
    remainder left out; ``shifts > 1`` adds the tilings shifted by the offsets ``k w //
    shifts`` voxels (``k < shifts``) along each axis that the windows tile, all their
    combinations (``shifts^3`` tilings of cubes, ``shifts^2`` of squares, fewer for
    windows of fewer voxels), wrapping around the periodic image (``periodic``: None, that
    of a :class:`~morphogen.image.VoxelImage`; an array is not): the mean of all their
    windows, their variance pooled within the tilings, ``n`` the windows of one tiling and
    ``tilings`` their number (see the module). The tensors of the windows:
    :func:`morphogen.rve.windows.window_tensors`.
    """
    data = as_array(image)
    field = data.astype(np.float64) if phase is None else indicator(data, phase)
    if spacing is None:
        spacing = getattr(image, "spacing", (1.0, 1.0, 1.0))
    periodic = getattr(image, "periodic", False) if periodic is None else periodic
    shifts = _shift_count(shifts, periodic)
    planar = field.shape[2] == 1
    pixel = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))[:2]
    voxel_volume = float(np.prod(pixel if planar else spacing))
    volumes, n, mean, var, tilings = [], [], [], [], []
    for w in window_sizes:
        w = int(w)
        across = 1 if planar else w
        counts = [field.shape[0] // w, field.shape[1] // w, field.shape[2] // across]
        if w < 1 or min(counts) < 1:
            raise ValueError(f"window size {w} does not fit in the image {field.shape}")
        nx, ny, nz = counts
        means = []
        for offset in _offsets(w, shifts, planar):  # (0, 0, 0) first: the field itself
            shifted = np.roll(field, tuple(-offset), axis=(0, 1, 2)) if offset.any() else field
            block = shifted[: nx * w, : ny * w, : nz * across].reshape(nx, w, ny, w, nz, across)
            means.append(block.mean(axis=(1, 3, 5), dtype=np.float64).ravel())
        labels = np.repeat(np.arange(len(means)), means[0].size)
        count, m, v = _moments(np.concatenate(means), labels)
        volumes.append(w * w * across * voxel_volume)
        n.append(count)
        mean.append(m)
        var.append(v)
        tilings.append(len(means))
    return SizeStatistics(
        np.array(volumes),
        np.array(n),
        np.array(mean),
        np.array(var),
        dimension=2 if planar else 3,
        tilings=np.array(tilings) if shifts > 1 else None,
    )


def rve_table(
    law: ScalingLaw, mean: float, eps_rel=(0.01, 0.05), n=(1, 10), *, dimension: int = 3
) -> list[dict]:
    """RVE volume and edge for each pair (eps_rel, n): the edge of a cube, ``V^(1/3)``, or
    of a square of area ``V`` (``dimension = 2``), ``V^(1/2)``."""
    rows = []
    for e in np.atleast_1d(eps_rel):
        for k in np.atleast_1d(n):
            vol = float(law.rve_volume(e, mean, k))
            edge = vol ** (1.0 / dimension)
            rows.append({"eps_rel": float(e), "n": int(k), "volume": vol, "edge": edge})
    return rows
