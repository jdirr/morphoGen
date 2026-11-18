"""Simulation domains (boxes) and the voxel grids that discretize them.

Conventions used throughout morphoGen:

* a domain is the axis-aligned box ``[origin, origin + size)``;
* a voxel grid of shape ``(nx, ny, nz)`` splits the domain into equal voxels,
  voxel ``(i, j, k)`` being centred at ``origin + (i + 1/2, j + 1/2, k + 1/2) * spacing``;
* arrays are indexed ``[i, j, k]`` along ``(x, y, z)``; file writers convert to the
  x-fastest order used by VTK, MetaImage and FFT codes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morphogen._utils import as_triplet


@dataclass(frozen=True)
class Domain:
    """Axis-aligned box in which microstructures are generated.

    Parameters
    ----------
    size
        Edge lengths (a single value for a cube), in the length unit of the study.
    origin
        Lower corner of the box.
    periodic
        If true the box is a torus: grains, fibres and cells wrap around the faces so
        that the sample tiles space (periodic boundary conditions, FFT solvers).
        If false the sample is an unbiased window cut from the infinite random medium.
    unit
        Name of the length unit (metadata only, e.g. ``"um"``).
    """

    size: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    periodic: bool = True
    unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", as_triplet(self.size, "domain.size", positive=True))
        object.__setattr__(self, "origin", as_triplet(self.origin, "domain.origin"))
        object.__setattr__(self, "periodic", bool(self.periodic))
        object.__setattr__(self, "unit", str(self.unit))

    @classmethod
    def cube(cls, length: float, *, periodic: bool = True, origin=(0.0, 0.0, 0.0), unit: str = ""):
        """Cubic domain of edge ``length``."""
        return cls(size=(length, length, length), origin=origin, periodic=periodic, unit=unit)

    @property
    def lower(self) -> np.ndarray:
        return np.array(self.origin, dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return self.lower + np.array(self.size, dtype=float)

    @property
    def center(self) -> np.ndarray:
        return self.lower + 0.5 * np.array(self.size, dtype=float)

    @property
    def lengths(self) -> np.ndarray:
        return np.array(self.size, dtype=float)

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    @property
    def surface_area(self) -> float:
        lx, ly, lz = self.size
        return 2.0 * (lx * ly + ly * lz + lz * lx)

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.size))

    def dilated(self, margin: float) -> Domain:
        """Non-periodic box enlarged by ``margin`` on every side."""
        m = float(margin)
        return Domain(
            size=tuple(s + 2 * m for s in self.size),
            origin=tuple(o - m for o in self.origin),
            periodic=False,
            unit=self.unit,
        )

    def contains(self, points) -> np.ndarray:
        """Boolean mask of the points lying in ``[lower, upper)``."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        return np.all((p >= self.lower) & (p < self.upper), axis=1)

    def wrap(self, points) -> np.ndarray:
        """Bring points back into the box by periodicity."""
        p = np.asarray(points, dtype=float)
        lo, ln = self.lower, self.lengths
        w = lo + np.mod(p - lo, ln)
        # np.mod can return the period itself for tiny negative inputs
        return np.where(w >= lo + ln, lo, w)

    def distance_to_box(self, points) -> np.ndarray:
        """Euclidean distance from points to the box (0 inside)."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        d = np.maximum(np.maximum(self.lower - p, p - self.upper), 0.0)
        return np.sqrt(np.sum(d * d, axis=1))

    def grid(self, shape=None, *, voxel_size=None) -> Grid:
        """Voxel grid covering the domain.

        Give either ``shape`` (an integer is the number of voxels along the longest edge,
        voxels being as cubic as possible; or three integers) or ``voxel_size``.
        """
        return Grid.for_domain(self, shape=shape, voxel_size=voxel_size)

    def to_dict(self) -> dict:
        return {
            "size": list(self.size),
            "origin": list(self.origin),
            "periodic": self.periodic,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Domain:
        unknown = set(data) - {"size", "origin", "periodic", "unit"}
        if unknown:
            raise ValueError(f"domain: unknown keys {sorted(unknown)}")
        if "size" not in data:
            raise ValueError("domain: 'size' is required")
        return cls(
            size=data["size"],
            origin=data.get("origin", (0.0, 0.0, 0.0)),
            periodic=data.get("periodic", True),
            unit=data.get("unit", ""),
        )


@dataclass(frozen=True)
class Grid:
    """Regular voxel grid over a domain (see the module docstring for conventions)."""

    domain: Domain
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        shape = as_triplet(self.shape, "grid.shape", positive=True, integer=True)
        object.__setattr__(self, "shape", shape)

    @classmethod
    def for_domain(cls, domain: Domain, shape=None, voxel_size=None) -> Grid:
        if (shape is None) == (voxel_size is None):
            raise ValueError("give exactly one of 'shape' and 'voxel_size'")
        lengths = domain.lengths
        if voxel_size is not None:
            h = as_triplet(voxel_size, "voxel_size", positive=True)
            n = np.maximum(1, np.rint(lengths / np.array(h))).astype(int)
        elif np.ndim(shape) == 0:
            n_long = int(shape)
            if n_long < 1 or n_long != shape:
                raise ValueError(f"shape: expected a positive integer, got {shape!r}")
            h = lengths.max() / n_long
            n = np.maximum(1, np.rint(lengths / h)).astype(int)
        else:
            n = np.array(as_triplet(shape, "shape", positive=True, integer=True))
        return cls(domain=domain, shape=tuple(int(v) for v in n))

    @property
    def spacing(self) -> np.ndarray:
        return self.domain.lengths / np.array(self.shape, dtype=float)

    @property
    def origin(self) -> np.ndarray:
        return self.domain.lower

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    @property
    def voxel_volume(self) -> float:
        return float(np.prod(self.spacing))

    def centers(self, axis: int) -> np.ndarray:
        """Coordinates of the voxel centres along ``axis``."""
        h = self.spacing[axis]
        return self.origin[axis] + (np.arange(self.shape[axis]) + 0.5) * h

    def center_points(self, k_start: int = 0, k_stop: int | None = None) -> np.ndarray:
        """Voxel centres of the slab ``k_start <= k < k_stop`` as an (n, 3) array.

        Points are ordered like ``array[:, :, k_start:k_stop].ravel()`` (C order).
        """
        k_stop = self.shape[2] if k_stop is None else k_stop
        x, y = self.centers(0), self.centers(1)
        z = self.centers(2)[k_start:k_stop]
        gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
        return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
