"""Properties that evolve with chemical ageing, and their maps in microstructures.

An ageing tracer (the amide concentration of oxidized epoxies, Ernault et al. 2019; the
degree of polymerization of cellulose in paper, Vibert et al. 2024; an absorbed amount
of oxygen or water) sets the local parameters through measured laws
(:class:`PropertyLaw`: piecewise linear tables, constant beyond them). A field of the
tracer in a microstructure (an oxidation or diffusion profile) is binned into graded
phases (:func:`graded_materials`) that the homogenization solvers and the exports take
like any other phases; a chemical shrinkage or a swelling becomes the expansion of the
graded phases under a unit change of temperature (``Loading(..., temperature=1)``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from morphogen.analysis.basic import as_array
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage

GAS_CONSTANT = 8.314462618


@dataclass(frozen=True)
class PropertyLaw:
    """Material parameters (``young``, ``poisson``, ``conductivity``, ``expansion``...)
    as piecewise linear functions of a tracer: ``values[name][i]`` at ``tracer[i]``
    (increasing), constant beyond the table."""

    tracer: tuple
    values: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = np.asarray(self.tracer, dtype=float)
        if x.ndim != 1 or len(x) < 1 or np.any(np.diff(x) <= 0):
            raise ValueError("tracer: increasing values")
        for name, v in self.values.items():
            if len(v) != len(x):
                raise ValueError(f"values[{name!r}]: one value per tracer value")
        object.__setattr__(self, "tracer", tuple(float(t) for t in x))
        object.__setattr__(
            self, "values", {k: tuple(float(a) for a in v) for k, v in self.values.items()}
        )

    def __call__(self, x) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=float)
        return {k: np.interp(x, self.tracer, v) for k, v in self.values.items()}

    def material(self, x: float, **fixed) -> Material:
        """The material at the tracer value ``x`` (``fixed``: the other parameters)."""
        params = {k: float(v) for k, v in self(float(x)).items()}
        params.update(fixed)
        return Material(**params)


def graded_materials(
    phases,
    tracer,
    laws: dict[int, PropertyLaw],
    materials: dict[int, Material] | None = None,
    *,
    levels: int = 16,
    first_label: int | None = None,
    **fixed,
) -> tuple[VoxelImage, dict[int, Material]]:
    """Graded phases of an ageing microstructure: the voxels of each phase with a law in
    ``laws`` are binned by their ``tracer`` value into ``levels`` classes of equal width,
    each a new phase whose material is the law at the middle of its class (``fixed``:
    parameters common to all, e.g. ``poisson=0.37``); the other phases keep their labels
    and ``materials``. Returns the label image and the materials of all its labels."""
    labels = np.asarray(as_array(phases)).astype(np.int64)
    x = np.asarray(as_array(tracer), dtype=float)
    if x.shape != labels.shape:
        raise ValueError("tracer: one value per voxel of the phase image")
    materials = dict(materials or {})
    out = labels.copy()
    table: dict[int, Material] = {
        int(p): materials[int(p)] for p in np.unique(labels) if int(p) not in laws
    }
    missing = [int(p) for p in np.unique(labels) if int(p) not in laws and int(p) not in materials]
    if missing:
        raise ValueError(f"materials: none for the phases {missing}")
    nxt = int(labels.max()) + 1 if first_label is None else int(first_label)
    for phase, law in laws.items():
        mask = labels == int(phase)
        if not mask.any():
            continue
        lo, hi = float(x[mask].min()), float(x[mask].max())
        width = (hi - lo) / levels if hi > lo else 1.0
        if hi > lo:
            k = np.clip(((x[mask] - lo) / width).astype(np.int64), 0, levels - 1)
        else:
            k = np.zeros(int(mask.sum()), dtype=np.int64)
        used = np.unique(k)
        lut = np.full(levels, -1, dtype=np.int64)
        lut[used] = nxt + np.arange(len(used))
        out[mask] = lut[k]
        for b in used:
            middle = lo + (b + 0.5) * width if hi > lo else lo
            table[int(lut[b])] = law.material(middle, **fixed)
        nxt += len(used)
    if isinstance(phases, VoxelImage):
        return phases.with_data(out.astype(np.int32), name="material"), table
    return VoxelImage(out.astype(np.int32), name="material"), table


@dataclass(frozen=True)
class Hydrolysis:
    """Chain scissions of cellulose by acid hydrolysis, Ekenstam's law:
    ``1/DP(t) - 1/DP(0) = k t`` (``k`` in 1/s at ``temperature``, Arrhenius with
    ``activation_energy`` J/mol). Vibert et al. (2024) found that the mechanical
    properties of aged papers fall faster below a critical degree of polymerization
    ``critical_dp`` (about 750), whatever the ageing (hydrolysis or oxidation): the
    fibres embrittle, more than their bonds."""

    dp0: float = 1500.0
    rate: float = 1e-10
    temperature: float = 353.15
    activation_energy: float = 110e3
    critical_dp: float = 750.0

    def rate_at(self, temperature: float | None = None) -> float:
        if temperature is None:
            return self.rate
        return self.rate * math.exp(
            -self.activation_energy / GAS_CONSTANT * (1.0 / temperature - 1.0 / self.temperature)
        )

    def dp(self, t, temperature: float | None = None) -> np.ndarray:
        """Degree of polymerization after the time ``t`` (s)."""
        return 1.0 / (1.0 / self.dp0 + self.rate_at(temperature) * np.asarray(t, dtype=float))

    def scissions(self, t, temperature: float | None = None) -> np.ndarray:
        """Scissions per initial chain, ``DP(0)/DP(t) - 1``."""
        return self.dp0 / self.dp(t, temperature) - 1.0

    def time_to(self, dp: float | None = None, temperature: float | None = None) -> float:
        """Time to reach the degree of polymerization ``dp`` (default: the critical one)."""
        dp = self.critical_dp if dp is None else dp
        return (1.0 / dp - 1.0 / self.dp0) / self.rate_at(temperature)
