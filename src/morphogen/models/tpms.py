"""Triply periodic minimal surface (TPMS) architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen._utils import as_triplet, check_fraction
from morphogen.domain import Domain, Grid
from morphogen.image import VoxelImage
from morphogen.kernels import implicit
from morphogen.models.affine import ANISOTROPY, Anisotropy, anisotropic
from morphogen.models.base import Microstructure, Model, param, register, register_kind
from morphogen.seeding import SeedLike, make_rng

#: Samples per unit-cell edge used to relate the level to the volume fraction.
_CALIBRATION_SAMPLES = 96


def _unit_cell_values(code: int, n: int = _CALIBRATION_SAMPLES) -> np.ndarray:
    """phi at the centres of an n^3 grid of one unit cell."""
    t = 2.0 * np.pi * (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(t, t, t, indexing="ij")
    sx, sy, sz, cx, cy, cz = np.sin(X), np.sin(Y), np.sin(Z), np.cos(X), np.cos(Y), np.cos(Z)
    if code == implicit.GYROID:
        return (sx * cy + sy * cz + sz * cx).ravel()
    if code == implicit.SCHWARZ_P:
        return (cx + cy + cz).ravel()
    if code == implicit.SCHWARZ_D:
        return (sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz).ravel()
    return (3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz).ravel()


@register_kind
class TPMSStructure(Microstructure):
    """Solid ``{phi <= level}`` (network) or ``{|phi| <= level}`` (sheet)."""

    kind = "tpms"

    def __init__(self, domain, surface, variant, level, cell_size, **kwargs):
        super().__init__(domain, **kwargs)
        self.surface = surface
        self.variant = variant
        self.level = float(level)
        self.cell_size = np.asarray(as_triplet(cell_size, "cell_size", positive=True))
        self._code = implicit.SURFACES[surface]
        self._k = 2.0 * np.pi / self.cell_size

    def _args(self):
        return self._code, self.variant == "sheet", self.level, self._k, self.domain.lower

    def voxelize(self, shape=None, *, voxel_size=None, grid: Grid | None = None, supersampling=1):
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        ss = int(supersampling)
        if not 1 <= ss <= 6:
            raise ValueError("supersampling must be an integer between 1 and 6")
        out = np.zeros(grid.shape, dtype=np.uint8)
        implicit.tpms_count(
            *self._args(), grid.origin, grid.spacing, np.array(grid.shape, dtype=np.int64), ss, out
        )
        if ss == 1:
            return VoxelImage.from_grid(out, grid, name="phase", metadata=self.image_metadata())
        data = out.astype(np.float32) / np.float32(ss**3)
        return VoxelImage.from_grid(data, grid, name="fraction", metadata=self.image_metadata())

    def level_set(self, shape=None, *, voxel_size=None, grid: Grid | None = None) -> VoxelImage:
        """First-order signed distance ``(phi - t) / |grad phi|`` (negative in the solid)."""
        grid = self.grid(shape, voxel_size=voxel_size, grid=grid)
        out = np.empty(grid.shape, dtype=np.float32)
        implicit.tpms_level_set(
            *self._args(), grid.origin, grid.spacing, np.array(grid.shape, dtype=np.int64), out
        )
        return VoxelImage.from_grid(out, grid, name="level_set", metadata=self.image_metadata())

    def contains(self, points) -> np.ndarray:
        pts = np.ascontiguousarray(points, dtype=float).reshape(-1, 3)
        out = np.zeros(len(pts), dtype=np.bool_)
        implicit.tpms_points(*self._args(), pts, out)
        return out

    #: The first-order distance is a distance near the surface; margin for its curvature
    lipschitz = 1.5

    def level_set_at(self, points, reach: float = np.inf) -> np.ndarray:
        """``(phi - t) / |grad phi|`` at the points, as :meth:`level_set`."""
        pts = np.ascontiguousarray(points, dtype=float).reshape(-1, 3)
        out = np.empty(len(pts))
        implicit.tpms_level_points(*self._args(), pts, out)
        return out

    def summary(self) -> dict:
        cells = self.domain.lengths / self.cell_size
        return {"surface": self.surface, "cells": "x".join(f"{c:g}" for c in cells)}

    def _geometry(self) -> dict:
        return {
            "surface": self.surface,
            "variant": self.variant,
            "level": self.level,
            "cell_size": self.cell_size.tolist(),
        }

    @classmethod
    def _from_geometry(cls, domain, geometry, **kwargs):
        return cls(
            domain,
            geometry["surface"],
            geometry["variant"],
            geometry["level"],
            geometry["cell_size"],
            **kwargs,
        )


@register
@anisotropic
@dataclass(frozen=True)
class TPMS(Model):
    """TPMS lattice: gyroid, Schwarz P (primitive), Schwarz D (diamond) or Neovius.

    ``network``: solid ``{phi <= level}``; ``sheet``: solid ``{|phi| <= level}`` (walls around
    the minimal surface). Give the level or the target volume fraction (solid fraction),
    from which the level is computed on a fine sampling of the unit cell.
    """

    name = "tpms"
    phases: ClassVar[dict[int, str]] = {0: "void", 1: "solid"}
    count_label = "unit cells"

    surface: str = param("gyroid", doc="gyroid, schwarz_p, schwarz_d or neovius", kind="str")
    cell_size: float | tuple = param(
        1.0, doc="Edge of the unit cell (one value or three)", kind="vector", unit="length"
    )
    variant: str = param("network", doc="'network' (solid skeleton) or 'sheet' (walls)", kind="str")
    level: float | None = param(None, doc="Level t of the iso-surface")
    volume_fraction: float | None = param(None, doc="Target solid volume fraction", kind="fraction")
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)

    def _validate(self) -> None:
        if self.surface not in implicit.SURFACES:
            raise ValueError(
                f"surface: expected one of {sorted(implicit.SURFACES)}, got {self.surface!r}"
            )
        if self.variant not in ("network", "sheet"):
            raise ValueError(f"variant: expected 'network' or 'sheet', got {self.variant!r}")
        as_triplet(self.cell_size, "cell_size", positive=True)
        if (self.level is None) == (self.volume_fraction is None):
            raise ValueError("give exactly one of 'level' and 'volume_fraction'")
        if self.volume_fraction is not None:
            check_fraction(self.volume_fraction, "volume_fraction")
        if self.variant == "sheet" and self.level is not None and self.level < 0:
            raise ValueError("level: must be non-negative for sheet TPMS")

    @property
    def iso_level(self) -> float:
        if self.level is not None:
            return float(self.level)
        values = _unit_cell_values(implicit.SURFACES[self.surface])
        if self.variant == "sheet":
            values = np.abs(values)
        return float(np.quantile(values, self.volume_fraction))

    def solid_fraction(self) -> float:
        """Volume fraction of the solid for the current level (numerical)."""
        values = _unit_cell_values(implicit.SURFACES[self.surface], 128)
        if self.variant == "sheet":
            values = np.abs(values)
        return float(np.mean(values <= self.iso_level))

    def generate(self, domain: Domain, seed: SeedLike = None) -> TPMSStructure:
        _, seed_info = make_rng(seed)
        cell = np.array(as_triplet(self.cell_size, "cell_size", positive=True))
        if domain.periodic:
            ratio = domain.lengths / cell
            if np.any(np.abs(ratio - np.rint(ratio)) > 1e-6 * np.maximum(ratio, 1.0)):
                raise ValueError(
                    "tpms: a periodic domain must contain a whole number of unit cells "
                    f"(domain {domain.size}, cell {tuple(cell)})"
                )
        return TPMSStructure(
            domain,
            self.surface,
            self.variant,
            self.iso_level,
            tuple(cell),
            model=self,
            seed=seed_info,
        )

    def expected_count(self, domain: Domain) -> float:
        """Number of unit cells in the domain."""
        return domain.volume / float(np.prod(np.broadcast_to(self.cell_size, 3)))

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        return {"level": self.iso_level, "volume_fraction": self.solid_fraction()}
