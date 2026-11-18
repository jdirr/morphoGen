"""Polycrystals of measured orientation maps (EBSD).

The model ``ebsd`` reads an orientation map (:mod:`morphogen.io.ebsd`: an ``.ang`` or a
``.ctf`` file), finds its grains, and gives a window of it as a microstructure: a lattice
of one cell through the thickness of the domain, each cell with the phase, the label and
the mean orientation of its grain — a columnar polycrystal whose section is the map, for
the solvers of the homogenization (crystal elasticity and plasticity turned by the
orientations of the grains).

The window has the size of the domain in the plane of the map: its first rows and
columns (the top left corner of its image), or a window drawn at random in it for each
realization (``window: random``) — the realizations of an RVE study are then windows of
the measured microstructure.

The axes of the domain are those of the image of the map: the first to the right, the
second upwards, the third towards the viewer; the orientations of the file are brought
into them (``axes: map``; ``setting``: that of the system of an ``.ang`` file; see
:mod:`morphogen.io.ebsd`).
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen._utils import check_positive
from morphogen.domain import Domain
from morphogen.models.base import Model, param, register
from morphogen.models.lattice_field import LatticeField
from morphogen.orientations import euler_to_matrix
from morphogen.seeding import SeedLike, make_rng

#: Where the window of a realization is taken in the map
WINDOWS = ("origin", "random")


@functools.lru_cache(maxsize=4)
def _grains_of(path: str, modified: float, threshold: float, min_size: int, axes: str,
               setting: int, turn: tuple | None):  # fmt: skip
    """The grains of the map of a file (kept for the realizations that follow)."""
    from morphogen.io.ebsd import read_ebsd

    ebsd = read_ebsd(path, axes=axes, setting=setting)
    if turn is not None:
        ebsd = ebsd.turned(euler_to_matrix(np.array(turn))[0])
    return ebsd.grains(threshold, min_size=min_size)


@register
@dataclass(frozen=True)
class EBSD(Model):
    """Polycrystal of a measured orientation map (EBSD): the grains of the map, with
    their orientations, in a window of the size of the domain."""

    name = "ebsd"
    phases: ClassVar[dict[int, str]] = {1: "crystals"}
    supports_periodic: ClassVar[bool] = False
    has_level_set: ClassVar[bool] = False
    count_label = "grains"
    #: The grains carry orientations (those of the map)
    crystal_orientations: ClassVar[bool] = True
    #: The parameters that are files: a configuration file gives them from its folder
    path_parameters: ClassVar[tuple[str, ...]] = ("file",)

    file: str = param(
        doc="Orientation map: an .ang (EDAX/TSL) or a .ctf (Oxford Instruments/HKL) file "
        "(from the folder of the configuration file), or the name of a map that comes "
        "with morphoGen (ebsd_map.ctf)",
        kind="str",
    )
    threshold: float = param(
        5.0,
        doc="Disorientation below which two neighbouring points belong to one grain",
        unit="degrees",
    )
    min_size: int = param(
        5,
        doc="Grains of fewer points are dropped: their points, and those that were not "
        "indexed, take the grain of the nearest point",
        kind="int",
    )
    window: str = param(
        "origin",
        doc="origin (the window of the size of the domain at the first rows and columns "
        "of the map: the top left corner of its image) or random (a window drawn in the "
        "map for each realization)",
        kind="str",
    )
    axes: str = param(
        "map",
        doc="map (the orientations of the file are brought into the axes of the image of "
        "the map, those of the domain: the first to the right, the second upwards, the "
        "third towards the viewer) or file (they are taken as the file gives them)",
        kind="str",
    )
    setting: int = param(
        1,
        doc="Setting of the axes of the orientations of an .ang file: their first axis "
        "points down the image (1, the usual one), up (2), to the right (3) or to the "
        "left (4)",
        kind="int",
    )
    turn: tuple[float, ...] | None = param(
        None,
        doc="Bunge angles (degrees) of other axes of the sample in those of the "
        "orientations: the orientations are brought into these axes (another setting "
        "of the system)",
        kind="floats",
    )

    def _validate(self) -> None:
        if not self.file:
            raise ValueError("file: the orientation map (.ang or .ctf)")
        if not 0.0 < self.threshold < 180.0:
            raise ValueError("threshold: an angle between 0 and 180 degrees")
        if self.min_size < 1:
            raise ValueError("min_size: a positive number of points")
        if self.window not in WINDOWS:
            raise ValueError(f"window: one of {WINDOWS}")
        if self.axes not in ("map", "file"):
            raise ValueError("axes: map or file")
        if self.setting not in (1, 2, 3, 4):
            raise ValueError("setting: 1, 2, 3 or 4")
        if self.turn is not None and len(self.turn) != 3:
            raise ValueError("turn: three Bunge angles (degrees)")

    def path(self) -> str:
        """The file of the map: ``file``, or the map of its name that comes with
        morphoGen where there is no such file."""
        from morphogen.io.ebsd import example_map

        path = os.path.abspath(os.path.expanduser(self.file))
        if not os.path.isfile(path):
            try:
                path = str(example_map(os.path.basename(path)))
            except ValueError:
                raise ValueError(f"file: no orientation map at {self.file!r}") from None
        return path

    def grains(self):
        """The grains of the whole map (:class:`morphogen.io.ebsd.Grains`)."""
        path = self.path()
        turn = None if self.turn is None else tuple(float(v) for v in self.turn)
        return _grains_of(path, os.path.getmtime(path), float(self.threshold),
                          int(self.min_size), self.axes, int(self.setting), turn)  # fmt: skip

    def active_phases(self) -> dict[int, str]:
        grains = self.grains()
        return {k: (p.name or f"phase {k}") for k, p in enumerate(grains.phases, start=1)
                if np.any(grains.phase == k)}  # fmt: skip

    def _window(self, domain: Domain) -> tuple[int, int]:
        """The numbers of columns and rows of the window of a domain in the map."""
        grains = self.grains()
        ny, nx = grains.labels.shape
        check_positive(domain.size[0], "domain.size")
        columns = max(1, round(domain.size[0] / grains.step[0]))
        rows = max(1, round(domain.size[1] / grains.step[1]))
        if columns > nx or rows > ny:
            width, height = nx * grains.step[0], ny * grains.step[1]
            raise ValueError(
                f"domain: {domain.size[0]:g} x {domain.size[1]:g} is larger than the map "
                f"({width:g} x {height:g} {grains.unit})"
            )
        return columns, rows

    def generate(self, domain: Domain, seed: SeedLike = None) -> LatticeField:
        if domain.periodic:
            raise ValueError("ebsd: a window of a map is not periodic (domain.periodic: false)")
        rng, seed_info = make_rng(seed)
        grains = self.grains()
        ny, nx = grains.labels.shape
        columns, rows = self._window(domain)
        x0 = int(rng.integers(nx - columns + 1)) if self.window == "random" else 0
        y0 = int(rng.integers(ny - rows + 1)) if self.window == "random" else 0
        part = grains.labels[y0 : y0 + rows, x0 : x0 + columns]
        present, inverse = np.unique(part[part >= 0], return_inverse=True)
        labels = np.full(part.shape, -1, dtype=np.int32)
        labels[part >= 0] = inverse
        # (the rows of the map from the top; the second axis of the domain upwards)
        cells = np.ascontiguousarray(labels[::-1].T[:, :, None])
        phase = grains.phase[present]
        phases = np.where(cells >= 0, phase[np.maximum(cells, 0)], 0).astype(np.uint8)
        attributes = {
            "file": os.path.basename(self.file),
            "window": [x0, y0, columns, rows],
            "step": list(grains.step),
            "grain_phase": phase.tolist(),
            "grain_spread": grains.spread[present].tolist(),
        }
        return LatticeField(domain, phases, labels=cells,
                            orientations=grains.orientations[present], attributes=attributes,
                            model=self, seed=seed_info)  # fmt: skip

    def expected_count(self, domain: Domain) -> float | None:
        grains = self.grains()
        ny, nx = grains.labels.shape
        columns, rows = self._window(domain)
        return grains.count * (columns * rows) / (nx * ny)


__all__ = ["EBSD", "WINDOWS"]
