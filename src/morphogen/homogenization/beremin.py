"""Cleavage by the local approach: the Weibull stress of Beremin (1983) on the fields of a
loading path (:func:`path_beremin`).

Cleavage starts at microcracks that the plastic strain nucleates, the weakest of which
gives way: the probability that a volume ``V`` under a stress ``sigma`` breaks is that of
its weakest link, and for a field of stresses::

    P_f = 1 - exp(-(sigma_w / sigma_u)^m),      sigma_w^m = sum_i <sigma_i>^m V_i / V_0,

with ``sigma_w`` the Weibull stress, the sum over the points of the plastic zone (their
volumes ``V_i``), ``m`` the modulus of Weibull, ``sigma_u`` the scale of the cleavage
stresses of the reference volume ``V_0``. The stress ``sigma_i`` that drives the cleavage
of a point is its largest principal stress (the original criterion, for polycrystals
whose grains are smaller than the points), or, where the points are those of the grains
of a polycrystal, the largest normal stress of the cleavage planes of their crystal
(:data:`~morphogen.homogenization.crystal.CLEAVAGE`): {100} in ferrite, the basal plane
in zinc — the planes of the lattice as it is at each time, where the fields of the path
have its orientation (a lattice that its twins turned, or that turned with the finite
strains of its crystal). A point counts from the time its cumulated plastic strain
passes ``plastic`` (it then belongs to the plastic zone, whatever happens later), and a
cell that has survived a Weibull stress has survived the smaller ones: the probability
at a time is that of the largest Weibull stress reached. A cell stands for a point of a
part: for ``N`` cells under the same loading (the gauge of a specimen), ``V_0 / N`` in
place of ``V_0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from morphogen.homogenization.crystal import CLEAVAGE, OPENING, cleavage_systems
from morphogen.orientations import euler_to_matrix

__all__ = [
    "Beremin",
    "BereminResult",
    "failure_probability",
    "normal_stress",
    "path_beremin",
    "principal_stress",
    "weibull_stress",
]

#: The curves that the criterion adds to a loading path, and its field
SERIES = ("beremin_weibull", "beremin_probability")
FIELD = "beremin_stress"


def _tensors(stress) -> np.ndarray:
    """Stress tensors (..., 3, 3) of Voigt components (..., 6): 11, 22, 33, 23, 31, 12."""
    s = np.asarray(stress, dtype=float)
    if s.shape[-1] != 6:
        raise ValueError(f"beremin: stresses of 6 components, got {s.shape}")
    out = np.empty((*s.shape[:-1], 3, 3))
    out[..., 0, 0], out[..., 1, 1], out[..., 2, 2] = s[..., 0], s[..., 1], s[..., 2]
    out[..., 1, 2] = out[..., 2, 1] = s[..., 3]
    out[..., 2, 0] = out[..., 0, 2] = s[..., 4]
    out[..., 0, 1] = out[..., 1, 0] = s[..., 5]
    return out


def principal_stress(stress) -> np.ndarray:
    """The largest principal stress of stresses (..., 6)."""
    return np.linalg.eigvalsh(_tensors(stress))[..., -1]


def normal_stress(stress, planes: str = "cube", orientations=None, c_over_a=None) -> np.ndarray:
    """The largest normal stress ``n . sigma . n`` of the cleavage planes of a family
    (:data:`~morphogen.homogenization.crystal.CLEAVAGE`) at points of stresses (..., 6) in
    the axes of the sample; ``orientations``: the Bunge angles (degrees) of the crystal
    of each point (..., 3), or of all of them (3); none: the axes of the sample."""
    if planes not in CLEAVAGE:
        raise ValueError(f"cleavage planes: one of {sorted(CLEAVAGE)}, got {planes!r}")
    normals, _, kinds, _ = cleavage_systems(planes, c_over_a)
    normals = normals[kinds == OPENING]  # in the axes of the crystal
    sigma = _tensors(stress)
    if orientations is not None:
        euler = np.asarray(orientations, dtype=float)
        if euler.shape[-1] != 3 or (euler.ndim > 1 and euler.shape[:-1] != sigma.shape[:-2]):
            raise ValueError(
                f"beremin: the Bunge angles of each point {sigma.shape[:-2]}, or three "
                f"angles, got {euler.shape}"
            )
        g = euler_to_matrix(euler.reshape(-1, 3)).reshape(*euler.shape[:-1], 3, 3)
        # the stress in the axes of the crystal: g sigma g^T
        sigma = g @ sigma @ np.swapaxes(g, -1, -2)
    return np.einsum("pi,...ij,pj->...p", normals, sigma, normals).max(axis=-1)


def weibull_stress(driving, volumes, modulus: float, volume: float) -> float:
    """The Weibull stress ``(sum_i <sigma_i>^m V_i / V_0)^(1/m)`` of the points of
    stresses ``driving`` and volumes ``volumes`` (NaN: points that do not count)."""
    sigma = np.asarray(driving, dtype=float).ravel()
    v = np.broadcast_to(np.asarray(volumes, dtype=float), np.shape(driving)).ravel()
    keep = np.isfinite(sigma) & (sigma > 0.0)
    if not keep.any():
        return 0.0
    # scaled by the largest stress: powers of 20 and more of stresses of hundreds of MPa
    top = float(sigma[keep].max())
    total = float(np.sum((sigma[keep] / top) ** modulus * v[keep]) / volume)
    return top * total ** (1.0 / modulus)


def failure_probability(weibull, modulus: float, scale: float):
    """``1 - exp(-(sigma_w / sigma_u)^m)``."""
    return -np.expm1(-((np.asarray(weibull, dtype=float) / scale) ** modulus))


@dataclass(frozen=True)
class Beremin:
    """The criterion of Beremin (1983): the ``modulus`` ``m`` of Weibull, the ``scale``
    ``sigma_u`` of the cleavage stresses of the reference ``volume`` ``V_0``. ``planes``:
    the cleavage planes of the crystals of the points, whose largest normal stress drives
    the cleavage (none: the largest principal stress), with ``c_over_a`` for hexagonal
    crystals; ``plastic``: the cumulated plastic strain beyond which a point belongs to
    the plastic zone (none: every point counts)."""

    modulus: float
    scale: float
    volume: float
    planes: str | None = None
    c_over_a: float | None = None
    plastic: float | None = None

    def __post_init__(self) -> None:
        for name in ("modulus", "scale", "volume"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ValueError(f"beremin: {name}, a positive number, got {value!r}")
            object.__setattr__(self, name, float(value))
        if self.modulus < 1.0:
            raise ValueError("beremin: a modulus of Weibull of 1 or more")
        if self.planes is not None and self.planes not in CLEAVAGE:
            raise ValueError(f"beremin: planes, one of {sorted(CLEAVAGE)}, got {self.planes!r}")
        if self.plastic is not None and not self.plastic >= 0:
            raise ValueError("beremin: plastic, a cumulated plastic strain of 0 or more")

    @classmethod
    def from_config(cls, spec) -> Beremin:
        """``{modulus, scale, volume, planes, c_over_a, plastic}``."""
        if isinstance(spec, Beremin):
            return spec
        if not isinstance(spec, dict):
            raise ValueError("beremin: a mapping (modulus, scale, volume, planes, plastic)")
        allowed = {"modulus", "scale", "volume", "planes", "c_over_a", "plastic"}
        unknown = sorted(set(spec) - allowed)
        if unknown:
            raise ValueError(f"beremin: unknown parameters {unknown}; use {sorted(allowed)}")
        missing = [k for k in ("modulus", "scale", "volume") if k not in spec]
        if missing:
            raise ValueError(f"beremin: give {missing}")
        return cls(**spec)

    def driving(self, stress, orientations=None) -> np.ndarray:
        """The stress that drives the cleavage of points of stresses (..., 6)."""
        if self.planes is None:
            return principal_stress(stress)
        return normal_stress(stress, self.planes, orientations, self.c_over_a)

    def weibull(self, driving, volumes) -> float:
        return weibull_stress(driving, volumes, self.modulus, self.volume)

    def probability(self, weibull):
        return failure_probability(weibull, self.modulus, self.scale)


@dataclass
class BereminResult:
    """The cleavage of a cell along a loading path: at each of its ``times``, the
    ``weibull`` stress of its fields and the ``probability`` of failure up to then; the
    ``driving`` stress of its points at the time of the largest Weibull stress (NaN
    outside the plastic zone), their ``weights`` and their ``support``."""

    times: np.ndarray
    weibull: np.ndarray
    probability: np.ndarray
    driving: np.ndarray
    weights: np.ndarray | None = field(default=None, repr=False)
    support: object | None = field(default=None, repr=False)

    @property
    def largest(self) -> float:
        """The largest Weibull stress of the path."""
        return float(self.weibull.max())

    def fields(self) -> dict:
        """The fields of the points, by name (for the writers of fields)."""
        return {FIELD: self.driving}

    def components(self) -> dict:
        """The scalar results: the largest Weibull stress and the probability of failure
        at the end of the path."""
        return {"weibull": self.largest, "probability": float(self.probability[-1])}

    def series(self, times) -> dict | None:
        """The curves of the criterion at the ``times`` of a path (:data:`SERIES`: the
        Weibull stress and the probability of failure, zero before the first fields), or
        None when the fields were not kept at each of them."""
        times = np.asarray(times, dtype=float)
        if times.size != self.times.size + 1 or not np.allclose(times[1:], self.times):
            return None
        return {
            SERIES[0]: np.concatenate([[0.0], self.weibull]),
            SERIES[1]: np.concatenate([[0.0], self.probability]),
        }


def path_beremin(result, criterion, *, orientations=None) -> BereminResult:
    """The criterion of Beremin along a loading path: ``result``, a :class:`~morphogen.
    homogenization.paths.PathResult` that kept its fields (``fields="steps"``, or the
    times wanted); ``criterion``: a :class:`Beremin` or its mapping; ``orientations``: the
    Bunge angles of the crystals of the points of the fields (those of the grains of the
    elements of a mesh, of the voxels of an image), for the normal stress of their
    cleavage planes (none: crystals in the axes of the sample). Where the fields of the
    path have the ``orientation`` of the lattices at each time (those that their twins
    turn; finite strains), the planes are those of the lattices as they are then. The
    stresses are those of the fields (Cauchy's in finite strains), the volumes those of
    the elements of a mesh (the points of an image weigh the same)."""
    law = Beremin.from_config(criterion)
    times = np.array([t for t in sorted(result.fields) if "stress" in result.fields[t]])
    if times.size == 0:
        raise ValueError("beremin: the stress fields of the path (run it with fields='steps')")
    support = result.support
    first = np.asarray(result.fields[times[0]]["stress"], dtype=float)
    shape = first.shape[:-1]
    if support is not None and hasattr(support, "volumes"):  # a mesh
        weights = np.abs(support.volumes())
    elif support is not None and hasattr(support, "spacing"):  # an image
        weights = np.full(shape, float(np.prod(support.spacing)))
    else:
        weights = np.ones(shape)
    if weights.shape != shape:
        raise ValueError(f"beremin: one volume per point {shape}, got {weights.shape}")
    zone = np.zeros(shape, dtype=bool) if law.plastic is not None else np.ones(shape, dtype=bool)
    weibull = np.zeros(times.size)
    best = np.full(shape, np.nan)
    for k, t in enumerate(times):
        arrays = result.fields[t]
        stress = np.asarray(arrays["stress"], dtype=float)
        solid = np.all(np.isfinite(stress), axis=-1)
        if law.plastic is not None:
            zone |= solid & (np.nan_to_num(np.asarray(arrays["p"], dtype=float)) > law.plastic)
        driving = np.full(shape, np.nan)
        counted = zone & solid
        if counted.any():
            euler = orientations
            if euler is not None and np.ndim(euler) > 1:
                euler = np.asarray(euler, dtype=float).reshape(*shape, 3)[counted]
            if law.planes is not None and "orientation" in arrays:
                # the lattices as they are at that time (NaN: not those of crystals)
                now = np.asarray(arrays["orientation"], dtype=float).reshape(*shape, 3)[counted]
                known = np.all(np.isfinite(now), axis=-1)
                first = np.zeros(3) if euler is None else np.asarray(euler, dtype=float)
                euler = np.where(known[:, None], now, np.broadcast_to(first, now.shape))
            driving[counted] = law.driving(stress[counted], euler)
        weibull[k] = law.weibull(driving, weights)
        if weibull[k] >= weibull[:k].max(initial=0.0):
            best = driving
    probability = law.probability(np.maximum.accumulate(weibull))
    return BereminResult(times=times, weibull=weibull, probability=np.asarray(probability),
                          driving=best, weights=weights, support=support)  # fmt: skip
