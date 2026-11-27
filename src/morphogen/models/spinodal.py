"""Spinodal decomposition: the two phases that a mixture quenched inside its spinodal forms
by the Cahn–Hilliard equation, and the coarsening of their pattern.

The composition ``c`` (``-1`` and ``1`` the two phases at equilibrium) of a mixture of the
free energy ``∫ f(c) + (κ/2) |∇c|^2``, ``f(c) = (c^2 - 1)^2 / 4``, evolves by the
Cahn–Hilliard equation (Cahn & Hilliard 1958; Cahn 1961)

    ∂c/∂t = ∇²(f'(c) - κ ∇²c),        f'(c) = c^3 - c.

Inside the spinodal, ``f''(c̄) = 3 c̄^2 - 1 < 0`` (``|c̄| < 1/√3``), the small fluctuations
about the mean composition ``c̄`` grow at the rate ``ω(k) = -k^2 (f''(c̄) + κ k^2)``, the
fastest at ``k_m = sqrt(-f''(c̄) / 2κ)``, of the wavelength ``λ_m = 2π / k_m`` and the rate
``ω_m = f''(c̄)^2 / 4κ``; the phases then separate, and their pattern coarsens as ``t^(1/3)``
(Lifshitz & Slyozov 1961; Huse 1986). The fraction of the phase 1 tends to that of the
lever rule, ``p = (1 + c̄) / 2``.

The equation is integrated on a periodic grid by the semi-implicit Fourier spectral
method with a linear stabilization (Chen & Shen 1998; Zhu, Chen, Shen & Tikare 1999):
``ĉ^{n+1} = (ĉ^n (1 + Δt S k^2) - Δt k^2 F[f'(c^n)]) / (1 + Δt S k^2 + Δt κ k^4)``, of the
stabilization ``S = 2``, stable for any step where ``|f''| <= 2S``; the mean composition is
the mode ``k = 0``, kept exactly. The unit of length is set by ``wavelength`` (``λ_m``,
``resolution`` points of the grid), the age of the pattern by ``coarsening`` (the time in
units of ``1 / ω_m``), the initial fluctuations by ``noise``. A domain that is not periodic
is a window of a periodic box larger by a wavelength on each side.

The realization is the composition on the grid, interpolated trilinearly between its
points: the phase 1 is ``c > 0``, its implicit function ``-c`` divided by the gradient of
the composition across the interfaces (a distance near them), as the images of
:mod:`morphogen.models.image`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy import fft as sfft

from morphogen import settings
from morphogen._utils import check_fraction, check_positive
from morphogen.domain import Domain
from morphogen.image import VoxelImage
from morphogen.models.base import Model, param, register, register_kind
from morphogen.models.image import ImageStructure
from morphogen.seeding import SeedLike, make_rng, seed_from_description

#: The linear stabilization of the scheme
STABILIZATION = 2.0
#: The bounds of the fraction (the spinodal: ``|c̄| < 1/√3``)
SPINODAL = (0.5 * (1.0 - 1.0 / math.sqrt(3.0)), 0.5 * (1.0 + 1.0 / math.sqrt(3.0)))


def wave_numbers(shape) -> list[np.ndarray]:
    """The wave numbers of a periodic grid of unit spacing, along each axis (to broadcast)."""
    out = []
    for axis, n in enumerate(shape):
        k = 2.0 * np.pi * sfft.fftfreq(n)
        view = [1] * len(shape)
        view[axis] = n
        out.append(k.reshape(view))
    return out


def cahn_hilliard(c, kappa: float, time: float, *, steps: int | None = None, energy=False):
    """The composition ``c`` (an array on a periodic grid of unit spacing) after ``time`` of
    the Cahn–Hilliard equation of the gradient energy ``kappa`` (see the module), by the
    stabilized semi-implicit spectral scheme in ``steps`` equal steps (default: of a
    hundredth of ``1/ω_m`` at first, which resolves the linear stage, growing by 2 % a step
    up to ``1/ω_m``). Returns the composition and, with ``energy``, the free energy per point
    after each step."""
    c = np.array(c, dtype=np.float64)
    kk = sum(k * k for k in np.broadcast_arrays(*wave_numbers(c.shape)))
    mean = float(c.mean())
    fpp = abs(3.0 * mean * mean - 1.0)
    unit = 4.0 * kappa / (fpp * fpp)  # 1 / ω_m
    if steps is None:
        dts, t, dt = [], 0.0, 0.01 * unit
        while t < time:
            dt = min(dt, time - t)
            dts.append(dt)
            t += dt
            dt = min(dt * 1.02, unit)
    else:
        dts = [time / int(steps)] * int(steps)
    history = []
    hat = sfft.fftn(c, workers=settings.workers())
    s = STABILIZATION
    for dt in dts:
        nonlinear = sfft.fftn(c**3 - c, workers=settings.workers())
        hat = (hat * (1.0 + dt * s * kk) - dt * kk * nonlinear) / (
            1.0 + dt * s * kk + dt * kappa * kk * kk
        )
        c = sfft.ifftn(hat, workers=settings.workers()).real
        if energy:
            history.append(free_energy(c, kappa))
    return (c, np.asarray(history)) if energy else c


def free_energy(c, kappa: float) -> float:
    """``∫ f(c) + (κ/2) |∇c|^2`` per point of a periodic grid of unit spacing (the gradient
    spectral)."""
    hat = sfft.fftn(c)
    ks = np.broadcast_arrays(*wave_numbers(c.shape))
    grad2 = sum(np.abs(sfft.ifftn(1j * k * hat)) ** 2 for k in ks)
    return float(np.mean(0.25 * (c * c - 1.0) ** 2 + 0.5 * kappa * grad2))


def structure_peak(c) -> float:
    """The wave number (grid units) of the maximum of the radially averaged structure factor
    ``|ĉ(k)|^2`` of a field on a periodic grid (its mean removed): ``2π / k`` is the
    characteristic length of the pattern."""
    hat = sfft.fftn(c - c.mean())
    power = np.abs(hat) ** 2
    ks = np.broadcast_arrays(*wave_numbers(c.shape))
    k = np.sqrt(sum(q * q for q in ks))
    dk = 2.0 * np.pi / max(c.shape)
    bins = np.rint(k / dk).astype(int).ravel()
    total = np.bincount(bins, power.ravel())
    count = np.bincount(bins)
    mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    mean[0] = 0.0
    j = int(np.argmax(mean))
    if 0 < j < len(mean) - 1:  # the parabola through the three bins of the peak
        a, b, d = mean[j - 1], mean[j], mean[j + 1]
        den = a - 2.0 * b + d
        shift = 0.5 * (a - d) / den if den != 0.0 else 0.0
        return (j + float(np.clip(shift, -0.5, 0.5))) * dk
    return j * dk


@register_kind
class SpinodalStructure(ImageStructure):
    """The phases of a spinodal decomposition: the composition on its grid (see the module)."""

    kind = "spinodal"

    def summary(self) -> dict:
        return dict(self.metadata.get("summary", {}))

    @property
    def phase_names(self) -> dict[int, str]:
        return dict(Spinodal.phases)

    def _geometry(self) -> dict:
        return {}

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        model = kwargs.get("model")
        if model is None:
            raise ValueError("a spinodal structure is read back with its model")
        return model.generate(domain, seed=seed_from_description(kwargs.get("seed")))


@register
@dataclass(frozen=True)
class Spinodal(Model):
    """The two phases of a spinodal decomposition by the Cahn–Hilliard equation, at an age
    of their coarsening (see the module)."""

    name = "spinodal"
    phases: ClassVar[dict[int, str]] = {0: "phase 0", 1: "phase 1"}

    volume_fraction: float = param(
        0.5,
        doc="Fraction of the phase 1 at equilibrium (the lever rule: the mean composition "
        "2p - 1), within the spinodal: 0.212 to 0.788",
        kind="fraction",
    )
    wavelength: float = param(
        1.0,
        doc="Wavelength of the fastest fluctuations of the early decomposition, the unit of "
        "length of the pattern",
        unit="length",
    )
    coarsening: float = param(
        10.0,
        doc="Age of the pattern: the time in units of the growth time of the fastest "
        "fluctuations (the pattern coarsens as its cube root)",
    )
    resolution: int = param(12, doc="Points of the grid per wavelength", kind="int")
    noise: float = param(0.05, doc="Amplitude of the initial fluctuations of the composition")

    def _validate(self) -> None:
        check_fraction(self.volume_fraction, "volume_fraction")
        low, high = SPINODAL
        if not low < self.volume_fraction < high:
            raise ValueError(
                f"volume_fraction: within the spinodal, between {low:.3f} and {high:.3f} "
                "(outside it the mixture decomposes by nucleation)"
            )
        check_positive(self.wavelength, "wavelength")
        check_positive(self.coarsening, "coarsening")
        check_positive(self.noise, "noise")
        if self.resolution < 6:
            raise ValueError("resolution: at least 6 points per wavelength")

    @property
    def mean_composition(self) -> float:
        return 2.0 * self.volume_fraction - 1.0

    @property
    def kappa(self) -> float:
        """The gradient energy in grid units: ``λ_m`` of ``resolution`` points."""
        fpp = abs(3.0 * self.mean_composition**2 - 1.0)
        return 0.5 * fpp * (self.resolution / (2.0 * math.pi)) ** 2

    def generate(self, domain: Domain, seed: SeedLike = None) -> SpinodalStructure:
        rng, seed_info = make_rng(seed)
        h = self.wavelength / self.resolution
        margin = 0.0 if domain.periodic else self.wavelength
        box = domain.lengths + 2.0 * margin
        shape = np.maximum(4, np.rint(box / h)).astype(int)
        spacing = box / shape
        c0 = self.mean_composition + self.noise * rng.uniform(-1.0, 1.0, tuple(shape))
        c0 += self.mean_composition - c0.mean()  # the mean composition exactly
        fpp = abs(3.0 * self.mean_composition**2 - 1.0)
        unit = 4.0 * self.kappa / (fpp * fpp)
        c = cahn_hilliard(c0, self.kappa, self.coarsening * unit)
        k = structure_peak(c)
        length = 2.0 * math.pi / k * float(spacing.min()) if k > 0 else float("nan")
        image = VoxelImage(
            c.astype(np.float32), spacing=tuple(spacing),
            origin=tuple(domain.lower - margin), periodic=domain.periodic, name="composition",
        )  # fmt: skip
        summary = {
            "grid": [int(n) for n in shape],
            "pattern_length": length,
            "fraction": float(np.mean(c > 0.0)),
            "mean_composition": float(c.mean()),
        }
        return SpinodalStructure(
            domain, image, ("grey", c.astype(np.float32), (0.0,)), domain.lower,
            model=self, seed=seed_info, metadata={"summary": summary},
        )  # fmt: skip

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """The fraction of the lever rule, the wavelength and the rate of the fastest
        fluctuations of the early decomposition (the rate in units of time per unit of
        mobility and length: ``ω_m = f''(c̄)^2 / 4κ``)."""
        fpp = 3.0 * self.mean_composition**2 - 1.0
        h = self.wavelength / self.resolution
        kappa = self.kappa * h * h  # in units of length
        return {
            "volume_fraction": self.volume_fraction,
            "mean_composition": self.mean_composition,
            "fastest_wavelength": self.wavelength,
            "fastest_rate": fpp * fpp / (4.0 * kappa),
        }


__all__ = ["SPINODAL", "Spinodal", "SpinodalStructure", "cahn_hilliard", "free_energy",
           "structure_peak", "wave_numbers"]  # fmt: skip
