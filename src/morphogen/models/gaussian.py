"""Excursion sets of stationary Gaussian random fields."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy import fft as sfft
from scipy import stats

from morphogen import settings, theory
from morphogen._utils import check_fraction, check_positive, exactly_one
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.models.affine import (
    ANISOTROPY,
    Anisotropy,
    anisotropy_from_config,
    invariant_theory,
)
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.seeding import SeedLike, make_rng, seed_from_description

#: Spectral cut-off: modes with ``|k| * correlation_length`` above this are dropped
#: (the Gaussian spectrum is then below exp(-18) of its maximum).
K_CUTOFF = 6.0
#: Margin (in correlation lengths) added around non-periodic domains.
MARGIN = 3.0


def _half_space_modes(n_max: np.ndarray, box: np.ndarray, kmax: float):
    """Integer wave vectors m (one of each pair ±m, m != 0) with |2 pi m / box| <= kmax."""
    axes = [np.arange(-n, n + 1) for n in n_max]
    mx, my, mz = np.meshgrid(*axes, indexing="ij")
    m = np.stack([mx.ravel(), my.ravel(), mz.ravel()], axis=1)
    half = (
        (m[:, 0] > 0)
        | ((m[:, 0] == 0) & (m[:, 1] > 0))
        | ((m[:, 0] == 0) & (m[:, 1] == 0) & (m[:, 2] > 0))
    )
    m = m[half]
    k = 2.0 * np.pi * m / box
    keep = np.einsum("ij,ij->i", k, k) <= kmax * kmax
    return m[keep], k[keep]


@register_kind
class GaussianField(Microstructure):
    """Phase 1 = {Y > threshold} for a unit-variance Gaussian field Y.

    Y is a finite sum of Fourier modes on the generation box (the domain if periodic,
    the domain enlarged by 3 correlation lengths otherwise), so the same realization
    can be evaluated exactly at any resolution.
    """

    kind = "gaussian_field"

    def __init__(
        self,
        domain,
        correlation_length,
        threshold,
        box_lower,
        box_lengths,
        mode_seed,
        anisotropy=None,
        **kwargs,
    ):
        super().__init__(domain, **kwargs)
        #: The affine anisotropy of the covariance ``rho(|A^-1 h|)`` (None: isotropic)
        self.anisotropy = anisotropy
        self.correlation_length = float(correlation_length)
        self.threshold = float(threshold)
        self.box_lower = np.asarray(box_lower, dtype=float)
        self.box_lengths = np.asarray(box_lengths, dtype=float)
        self.mode_seed = mode_seed
        self._modes = None

    def modes(self):
        """Integer wave vectors and complex amplitudes C_m: Y(x) = Re sum C_m exp(i k_m.x)."""
        if self._modes is None:
            ell = self.correlation_length
            kmax = K_CUTOFF / ell
            if self.anisotropy is None:
                n_max = np.floor(kmax * self.box_lengths / (2.0 * np.pi)).astype(int)
                m, k = _half_space_modes(n_max, self.box_lengths, kmax)
            else:  # the spectrum of Y(A^-1 x) at k is that of Y at A^T k, |A^T k| >= s_min |k|
                reach = kmax / float(self.anisotropy.principal.min())
                n_max = np.floor(reach * self.box_lengths / (2.0 * np.pi)).astype(int)
                m, k = _half_space_modes(n_max, self.box_lengths, reach)
                k = k @ self.anisotropy.matrix
                keep = np.einsum("ij,ij->i", k, k) <= kmax * kmax
                m, k = m[keep], k[keep]
            spectrum = np.exp(-0.5 * ell * ell * np.einsum("ij,ij->i", k, k))
            var = spectrum / spectrum.sum()
            rng = np.random.default_rng(seed_from_description(self.mode_seed))
            a = rng.standard_normal(len(m))
            b = rng.standard_normal(len(m))
            # A cos + B sin = Re[(A - iB) e^{i k.x}]
            amplitudes = np.sqrt(var) * (a - 1j * b)
            self._modes = (m, amplitudes)
        return self._modes

    def field(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """Values of the unit-variance Gaussian field at the voxel centres (float32)."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        m, amp = self.modes()
        if len(m) == 0:
            raise ValueError("gaussian_excursion: correlation length too large for the box")
        x0 = grid.origin + 0.5 * grid.spacing - self.box_lower
        aligned = self.domain.periodic and np.all(np.array(grid.shape) > 2 * np.abs(m).max(axis=0))
        if aligned:
            data = self._field_fft(grid, m, amp, x0)
        else:
            data = self._field_separable(grid, m, amp, x0)
        return VoxelImage.from_grid(data, grid, name="field", metadata=self.image_metadata())

    def _field_fft(self, grid, m, amp, x0):
        # Y_j = Re sum_m C_m e^{i k_m.x0} e^{2 i pi m.j / n}: one inverse FFT on the grid
        k = 2.0 * np.pi * m / self.box_lengths
        coef = amp * np.exp(1j * (k @ x0))
        n = np.array(grid.shape)
        spec = np.zeros(tuple(n), dtype=np.complex128)
        idx = np.mod(m, n)
        np.add.at(spec, (idx[:, 0], idx[:, 1], idx[:, 2]), coef)
        y = sfft.ifftn(spec, workers=settings.workers()) * spec.size
        return np.ascontiguousarray(y.real, dtype=np.float32)

    def _field_separable(self, grid, m, amp, x0):
        # dense coefficient block, contracted axis by axis with exp(i k_a x_a), the axes of
        # fewest voxels first: a plane contracts its normal first, which keeps the blocks
        # to the size of the plane (x, y, z in turn for a cube)
        n_max = np.abs(m).max(axis=0)
        dense = np.zeros(tuple(2 * n_max + 1), dtype=np.complex128)
        dense[m[:, 0] + n_max[0], m[:, 1] + n_max[1], m[:, 2] + n_max[2]] = amp
        order = [int(a) for a in np.argsort(grid.shape, kind="stable")]
        out = np.transpose(dense, order)
        for axis in order:
            ma = np.arange(-n_max[axis], n_max[axis] + 1)
            xa = x0[axis] + np.arange(grid.shape[axis]) * grid.spacing[axis]
            e = np.exp(2j * np.pi * np.outer(ma, xa) / self.box_lengths[axis])
            out = np.tensordot(out, e, axes=([0], [0]))  # contracted axis moves to the end
        return np.ascontiguousarray(np.transpose(out.real, np.argsort(order)), dtype=np.float32)

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        if ss == 1:
            data = (self.field(grid=grid).data > self.threshold).astype(np.uint8)
            return VoxelImage.from_grid(data, grid, name="phase", metadata=self.image_metadata())
        fine = Grid(grid.domain, tuple(n * ss for n in grid.shape))
        inside = (self.field(grid=fine).data > self.threshold).astype(np.float32)
        nx, ny, nz = grid.shape
        data = inside.reshape(nx, ss, ny, ss, nz, ss).mean(axis=(1, 3, 5), dtype=np.float32)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def values_at(self, points) -> np.ndarray:
        """The Gaussian field ``Y`` at points of space (rows): the sum of its modes."""
        from morphogen.kernels.implicit import fourier_points

        m, amp = self.modes()
        if len(m) == 0:
            raise ValueError("gaussian_excursion: correlation length too large for the box")
        pts = np.ascontiguousarray(points, dtype=float).reshape(-1, 3)
        out = np.empty(len(pts))
        fourier_points(np.ascontiguousarray(m, dtype=np.int64), np.ascontiguousarray(amp.real),
                       np.ascontiguousarray(amp.imag), self.box_lengths, self.box_lower, pts,
                       out)  # fmt: skip
        return out

    def contains(self, points) -> np.ndarray:
        return self.values_at(points) > self.threshold

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """``(threshold - Y) * correlation_length`` at the points, as :meth:`level_set`."""
        return (self.threshold - self.values_at(points)) * self.correlation_length

    @property
    def lipschitz(self) -> float:
        """A bound of the slope of :meth:`level_set_at` (three times the root mean square
        of the gradient of ``Y``, times the correlation length)."""
        m, amp = self.modes()
        k = 2.0 * np.pi * m / self.box_lengths
        rms = np.sqrt(np.sum(np.abs(amp) ** 2 * np.einsum("ij,ij->i", k, k)))
        return float(3.0 * rms * self.correlation_length)

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """``(threshold - Y) * correlation_length``: negative in phase 1, roughly a distance
        near the interface."""
        f = self.field(shape, voxel_size=voxel_size, grid=grid)
        data = (self.threshold - f.data) * np.float32(self.correlation_length)
        return f.with_data(data.astype(np.float32), name="level_set")

    def summary(self) -> dict:
        return {"n_modes": len(self.modes()[0])}

    def _geometry(self) -> dict:
        out = {
            "correlation_length": self.correlation_length,
            "threshold": self.threshold,
            "box_lower": self.box_lower.tolist(),
            "box_lengths": self.box_lengths.tolist(),
            "mode_seed": self.mode_seed,
        }
        if self.anisotropy is not None:
            out["anisotropy"] = self.anisotropy.to_config()
        return out

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            geometry["correlation_length"],
            geometry["threshold"],
            geometry["box_lower"],
            geometry["box_lengths"],
            geometry["mode_seed"],
            anisotropy_from_config(geometry.get("anisotropy")),
            **kwargs,
        )


@register
@dataclass(frozen=True)
class GaussianExcursion(Model):
    """Excursion set {Y > threshold} of a Gaussian field with covariance exp(-h^2 / 2 l^2).

    Give the volume fraction ``p`` of the excursion (threshold ``Phi^-1(1 - p)``) or the
    threshold itself. Periodic boxes should be at least ~10 correlation lengths wide. An
    ``anisotropy`` puts the metric of :mod:`morphogen.models.affine` in the covariance,
    ``rho(|A^-1 h|)``: correlation lengths ``s_i l`` along its principal axes, in periodic
    boxes too, whatever the axes (the modes of the box weighted by the spectrum at
    ``A^T k``).
    """

    name = "gaussian_excursion"
    phases: ClassVar[dict[int, str]] = {0: "phase 0", 1: "excursion"}

    correlation_length: float = param(
        1.0, doc="Correlation length l of the Gaussian covariance", unit="length"
    )
    volume_fraction: float | None = param(
        None, doc="Volume fraction of the excursion", kind="fraction"
    )
    threshold: float | None = param(None, doc="Level of the excursion {Y > threshold}")
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        check_positive(self.correlation_length, "correlation_length")
        exactly_one("volume_fraction", self.volume_fraction, "threshold", self.threshold)
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")

    @property
    def level(self) -> float:
        if self.threshold is not None:
            return self.threshold
        return float(stats.norm.isf(self.volume_fraction))

    def generate(self, domain: Domain, seed: SeedLike = None) -> GaussianField:
        rng, seed_info = make_rng(seed)
        ell = self.correlation_length
        a = None if self.anisotropy is None or self.anisotropy.is_identity else self.anisotropy
        reach = ell if a is None else ell * float(a.principal.max())
        if domain.periodic:
            lower, lengths = domain.lower, domain.lengths
        else:
            lower = domain.lower - MARGIN * reach
            lengths = domain.lengths + 2.0 * MARGIN * reach
        # the modes are drawn from a child seed so that they can be regenerated lazily
        mode_seed = {"entropy": int(rng.integers(0, 2**63)), "spawn_key": []}
        return GaussianField(
            domain, ell, self.level, lower, lengths, mode_seed, a, model=self, seed=seed_info
        )

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        lam = self.level
        p = float(stats.norm.sf(lam))
        # second spectral moment -rho''(0) of the covariance: 1 / l^2
        lam2 = 1.0 / self.correlation_length**2
        g = math.exp(-0.5 * lam**2)
        out = {
            "threshold": lam,
            "volume_fraction": p,
            "specific_surface": 2.0 / math.pi * math.sqrt(lam2) * g,
            # Adler's Gaussian kinematic formulae: 2 pi times the Euler density of the
            # plane sections, and the Euler density of the excursion
            "mean_curvature_density": lam2 / math.sqrt(2.0 * math.pi) * lam * g,
            "euler_density": lam2**1.5 / (4.0 * math.pi**2) * (lam**2 - 1.0) * g,
            "integral_range": self.integral_range(),
            "variance_exponent": 1.0,
        }
        if self.anisotropy is not None and not self.anisotropy.is_identity:
            out = invariant_theory(out)
        return out

    def covariance(self, h) -> np.ndarray:
        """C(h) = P(Y(x) > l, Y(x + h) > l) with rho(h) = exp(-h^2 / 2 l^2)."""
        h = np.asarray(h, dtype=float)
        rho = np.exp(-0.5 * (h / self.correlation_length) ** 2)
        return theory.bivariate_normal_upper(self.level, rho)

    def integral_range(self) -> float:
        lam = self.level
        p = float(stats.norm.sf(lam))

        def f(h):
            return float(self.covariance(h)) - p * p

        return theory.radial_integral(f, 8.0 * self.correlation_length) / (p * (1.0 - p))
