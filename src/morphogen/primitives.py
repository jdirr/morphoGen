"""Explicit geometric primitives: spheres and segments (cylinders or spherocylinders).

These are the "vector" description of grain-based microstructures. Rasterization,
signed distances and exports all work from these arrays.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from morphogen.domain import Domain


@dataclass
class Spheres:
    centers: np.ndarray  # (n, 3)
    radii: np.ndarray  # (n,)

    def __post_init__(self) -> None:
        self.centers = np.ascontiguousarray(self.centers, dtype=float).reshape(-1, 3)
        self.radii = np.ascontiguousarray(self.radii, dtype=float).reshape(-1)
        if len(self.radii) != len(self.centers):
            raise ValueError("Spheres: centers and radii must have the same length")

    def __len__(self) -> int:
        return len(self.radii)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.radii[:, None]
        return self.centers - r, self.centers + r

    def extent(self) -> float:
        """Largest distance from a primitive's reference point to its surface."""
        return float(self.radii.max()) if len(self) else 0.0

    def subset(self, mask) -> Spheres:
        return Spheres(self.centers[mask], self.radii[mask])

    def translated(self, shift) -> Spheres:
        return Spheres(self.centers + np.asarray(shift, dtype=float), self.radii.copy())

    @staticmethod
    def concatenate(items: list[Spheres]) -> Spheres:
        if not items:
            return Spheres(np.empty((0, 3)), np.empty(0))
        return Spheres(
            np.concatenate([s.centers for s in items]), np.concatenate([s.radii for s in items])
        )

    def volumes(self) -> np.ndarray:
        return 4.0 / 3.0 * np.pi * self.radii**3

    def to_dict(self) -> dict:
        return {"centers": self.centers.tolist(), "radii": self.radii.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> Spheres:
        return cls(np.array(data["centers"], dtype=float), np.array(data["radii"], dtype=float))


@dataclass
class Segments:
    """Cylinders of axis ``[start, end]`` with flat caps, or spherocylinders (round caps)."""

    start: np.ndarray  # (n, 3)
    end: np.ndarray  # (n, 3)
    radii: np.ndarray  # (n,)
    round_caps: bool = True

    def __post_init__(self) -> None:
        self.start = np.ascontiguousarray(self.start, dtype=float).reshape(-1, 3)
        self.end = np.ascontiguousarray(self.end, dtype=float).reshape(-1, 3)
        self.radii = np.ascontiguousarray(self.radii, dtype=float).reshape(-1)
        if not (len(self.start) == len(self.end) == len(self.radii)):
            raise ValueError("Segments: start, end and radii must have the same length")
        self.round_caps = bool(self.round_caps)

    def __len__(self) -> int:
        return len(self.radii)

    @property
    def lengths(self) -> np.ndarray:
        return np.linalg.norm(self.end - self.start, axis=1)

    @property
    def midpoints(self) -> np.ndarray:
        return 0.5 * (self.start + self.end)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.radii[:, None]
        return np.minimum(self.start, self.end) - r, np.maximum(self.start, self.end) + r

    def extent(self) -> float:
        if not len(self):
            return 0.0
        half = 0.5 * self.lengths
        if self.round_caps:
            return float((half + self.radii).max())
        return float(np.sqrt(half**2 + self.radii**2).max())

    def subset(self, mask) -> Segments:
        return Segments(self.start[mask], self.end[mask], self.radii[mask], self.round_caps)

    def translated(self, shift) -> Segments:
        s = np.asarray(shift, dtype=float)
        return Segments(self.start + s, self.end + s, self.radii.copy(), self.round_caps)

    @staticmethod
    def concatenate(items: list[Segments], round_caps: bool = True) -> Segments:
        if not items:
            return Segments(np.empty((0, 3)), np.empty((0, 3)), np.empty(0), round_caps)
        caps = {s.round_caps for s in items}
        if len(caps) > 1:
            raise ValueError("cannot concatenate segments with different cap types")
        return Segments(
            np.concatenate([s.start for s in items]),
            np.concatenate([s.end for s in items]),
            np.concatenate([s.radii for s in items]),
            items[0].round_caps,
        )

    def volumes(self) -> np.ndarray:
        v = np.pi * self.radii**2 * self.lengths
        if self.round_caps:
            v = v + 4.0 / 3.0 * np.pi * self.radii**3
        return v

    def to_dict(self) -> dict:
        return {
            "start": self.start.tolist(),
            "end": self.end.tolist(),
            "radii": self.radii.tolist(),
            "round_caps": self.round_caps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Segments:
        return cls(
            np.array(data["start"], dtype=float),
            np.array(data["end"], dtype=float),
            np.array(data["radii"], dtype=float),
            bool(data.get("round_caps", True)),
        )


Primitive = Spheres | Segments


def overlapping_box(prims: Primitive, lower, upper, margin: float = 0.0) -> np.ndarray:
    """Mask of the primitives whose bounding box meets ``[lower - margin, upper + margin]``."""
    lo, hi = prims.bounds()
    lower = np.asarray(lower, dtype=float) - margin
    upper = np.asarray(upper, dtype=float) + margin
    return np.all((hi >= lower) & (lo <= upper), axis=1)


def with_periodic_images(prims: Primitive, domain: Domain, margin: float = 0.0) -> Primitive:
    """Primitives plus the periodic copies needed to cover the box (and ``margin`` around).

    Every primitive whose bounding box, shifted by a combination of box periods, meets
    the box is added with that shift, so that a non-periodic rasterization of the
    result over the box equals the periodic microstructure.
    """
    if len(prims) == 0:
        return prims
    lengths = domain.lengths
    lo, hi = prims.bounds()
    # number of periods spanned by the largest primitive (plus the margin)
    span = np.max((hi - lo + 2 * margin) / lengths)
    m = int(np.ceil(span)) + 1
    lower, upper = domain.lower - margin, domain.upper + margin
    pieces = [prims.subset(np.all((hi >= lower) & (lo <= upper), axis=1))]
    for shift in itertools.product(range(-m, m + 1), repeat=3):
        if shift == (0, 0, 0):
            continue
        s = np.array(shift) * lengths
        mask = np.all((hi + s >= lower) & (lo + s <= upper), axis=1)
        if np.any(mask):
            pieces.append(prims.subset(mask).translated(s))
    return type(prims).concatenate(pieces)


def restricted_to_box(prims: Primitive, domain: Domain, margin: float = 0.0) -> Primitive:
    """Primitives whose bounding box meets the (non-periodic) box enlarged by ``margin``."""
    return prims.subset(overlapping_box(prims, domain.lower, domain.upper, margin))
