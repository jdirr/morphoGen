"""Stochastic reconstruction of two-phase microstructures from their statistical
descriptors (Yeong & Torquato 1998).

A realization is a periodic lattice of cells of edge ``cell_size`` covering the domain
(a :class:`~morphogen.models.lattice_field.LatticeField`), annealed from random cells of
the volume fraction (:mod:`morphogen.kernels.annealing`): the two-point probability
``S_2`` of phase 1 along the axes of the lattice, and its lineal-path function ``L``
with the weight ``lineal_path``, approach their targets, those of

* a Debye random medium (``correlation_length`` ``a``, Debye, Anderson & Brumberger
  1957): ``S_2(r) = p^2 + p (1 - p) e^{-r/a}``, specific surface ``4 p (1 - p) / a``;
* a realization of another model (``reference``) on the same lattice, periodic;
* a phase image (``image``: a file), its descriptors measured at its own voxels and
  interpolated at the lags of the lattice (``p^2`` and 0 beyond its half size).

``S_2`` alone does not determine a microstructure: two microstructures with the same
``S_2`` can differ in their connectivity, which ``L`` (and the chords) retains in part.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen._utils import check_fraction, check_positive
from morphogen.domain import Domain, Grid
from morphogen.kernels.annealing import anneal
from morphogen.models.base import Model, param, register
from morphogen.models.lattice_field import LatticeField, lattice_shape
from morphogen.seeding import SeedLike, make_rng


def _measured(image, rmax: int, lags: np.ndarray, n_axes: int):
    """``S_2`` and ``L`` of phase 1 of an image at the physical ``lags`` of each axis (rows),
    interpolated from its own lags, and its volume fraction."""
    from morphogen.analysis.correlation import lineal_path
    from morphogen.analysis.covariance import covariance

    cov = covariance(image, 1)
    lp = lineal_path(image, 1)
    p = cov.volume_fraction
    s = np.full((3, rmax + 1), p * p)
    line = np.zeros((3, rmax + 1))
    for a, name in enumerate("xyz"[:n_axes]):
        s[a] = np.interp(lags[a], cov.lags[name], cov.values[name], right=p * p)
        line[a] = np.interp(lags[a], lp.lags[name], lp.values[name], right=0.0)
    return s, line, p


@register
@dataclass(frozen=True)
class Reconstruction(Model):
    """Two-phase microstructure annealed to the two-point probability (and the lineal-path
    function) of a Debye medium, of a model or of an image (Yeong & Torquato 1998)."""

    name = "reconstruction"
    phases: ClassVar[dict[int, str]] = {0: "phase 0", 1: "phase 1"}

    reference: Model | None = param(
        None, doc="A model whose realization on the lattice gives the targets", kind="model"
    )
    image: str | None = param(
        None, doc="A phase image (file) whose phase 1 gives the targets", kind="str"
    )
    correlation_length: float | None = param(
        None,
        doc="Debye random medium: S2(r) = p^2 + p (1 - p) exp(-r / a)",
        unit="length",
    )
    volume_fraction: float | None = param(
        None, doc="Fraction of phase 1 of the Debye medium", kind="fraction"
    )
    lineal_path: float = param(
        0.0, doc="Weight of the lineal-path function in the energy (0: S2 alone)"
    )
    max_lag: int | None = param(
        None,
        doc="Largest lag of the descriptors, in cells (default: a quarter of the lattice, "
        "at most 32)",
        kind="int",
    )
    sweeps: int = param(100, doc="Trial swaps per cell", kind="int")
    cooling: float = param(
        0.98, doc="Factor of the temperature after every tenth of a sweep", kind="fraction"
    )
    tolerance: float = param(1e-10, doc="Energy at which the annealing stops")
    cell_size: float = param(1.0, doc="Edge of the cells of the lattice", unit="length")
    dimension: int = param(
        3, doc="3, or 2 for planar structures (one cell through the thickness)", kind="int"
    )
    smoothing: float = param(
        0.0,
        doc="Gaussian smoothing (in cells) of the interface for other resolutions and "
        "conforming meshes (0: trilinear interpolation of the cells)",
        unit="cells",
    )

    def _validate(self) -> None:
        given = [k for k in ("reference", "image", "correlation_length")
                 if getattr(self, k) is not None]  # fmt: skip
        if len(given) != 1:
            raise ValueError(
                "give one target: reference (a model), image (a file) or correlation_length "
                "(a Debye medium)"
            )
        if self.correlation_length is not None:
            check_positive(self.correlation_length, "correlation_length")
            if self.volume_fraction is None:
                raise ValueError("volume_fraction: the fraction of phase 1 of the Debye medium")
            check_fraction(self.volume_fraction, "volume_fraction")
            if self.lineal_path:
                raise ValueError("lineal_path: unknown for a Debye medium (give a reference)")
        elif self.volume_fraction is not None:
            raise ValueError("volume_fraction: that of the reference or of the image")
        check_positive(self.lineal_path, "lineal_path", allow_zero=True)
        if self.max_lag is not None and self.max_lag < 1:
            raise ValueError("max_lag: a positive number of cells")
        if self.sweeps < 1:
            raise ValueError("sweeps: a positive number of swaps per cell")
        if not 0.0 < self.cooling <= 1.0:
            raise ValueError("cooling: a factor in (0, 1]")
        check_positive(self.tolerance, "tolerance", allow_zero=True)
        check_positive(self.cell_size, "cell_size")
        if self.dimension not in (2, 3):
            raise ValueError("dimension: 2 or 3")
        check_positive(self.smoothing, "smoothing", allow_zero=True)

    def _lattice(self, domain: Domain) -> tuple[Grid, int, int]:
        """The periodic lattice of the domain, its axes and the largest lag."""
        shape = lattice_shape(domain, self.cell_size, self.dimension)
        n_axes = 3 if self.dimension == 3 else 2
        if self.max_lag is not None:
            rmax = int(self.max_lag)
        else:
            rmax = min(32, max(1, min(shape[:n_axes]) // 4))
        if rmax > min(shape[:n_axes]) // 2:
            raise ValueError(f"max_lag: at most half the lattice ({min(shape[:n_axes]) // 2})")
        return Grid(dataclasses.replace(domain, periodic=True), shape), n_axes, rmax

    def targets(self, domain: Domain, seed: SeedLike = None):
        """The target ``S_2`` and ``L`` along the axes (rows, lags 0..max_lag) and the
        volume fraction, on the lattice of the domain."""
        rng, _ = make_rng(seed)
        return self._targets(domain, rng)

    def _targets(self, domain: Domain, rng):
        lattice, n_axes, rmax = self._lattice(domain)
        lags = np.arange(rmax + 1)[None, :] * np.asarray(lattice.spacing, dtype=float)[:, None]
        if self.correlation_length is not None:
            p = float(self.volume_fraction)
            s = p * p + p * (1.0 - p) * np.exp(-lags / self.correlation_length)
            return s, np.zeros_like(s), p
        if self.reference is not None:
            ref = self.reference.generate(lattice.domain, seed=rng)
            image = ref.voxelize(grid=lattice)
        else:
            from morphogen import io

            image = io.load_image(self.image)
        s, line, p = _measured(image, rmax, lags, n_axes)
        return s, line, p

    def generate(self, domain: Domain, seed: SeedLike = None) -> LatticeField:
        rng, seed_info = make_rng(seed)
        lattice, n_axes, rmax = self._lattice(domain)
        target_s, target_l, p = self._targets(domain, rng)
        cells = int(np.prod(lattice.shape))
        img = np.zeros(cells, dtype=np.uint8)
        img[rng.permutation(cells)[: round(p * cells)]] = 1
        img = img.reshape(lattice.shape)
        stage = max(1, cells // 10)
        initial, energy, trials, accepted = anneal(
            img,
            np.ascontiguousarray(target_s),
            np.ascontiguousarray(target_l),
            float(self.lineal_path),
            n_axes,
            rmax,
            int(self.sweeps) * cells,
            float(self.cooling),
            stage,
            float(self.tolerance),
            int(rng.integers(2**31 - 1)),
        )
        attributes = {"initial_energy": float(initial), "energy": float(energy),
                      "trials": int(trials), "accepted": int(accepted),
                      "max_lag": rmax}  # fmt: skip
        return LatticeField(domain, img, smoothing=self.smoothing, attributes=attributes,
                            model=self, seed=seed_info)  # fmt: skip

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        if self.correlation_length is not None:
            p, a = float(self.volume_fraction), float(self.correlation_length)
            return {"volume_fraction": p, "specific_surface": 4.0 * p * (1.0 - p) / a,
                    "integral_range": 8.0 * math.pi * a**3}  # fmt: skip
        if self.reference is not None:
            ref = self.reference.theory(domain)
            return {k: ref[k] for k in ("volume_fraction",) if k in ref}
        return {}
