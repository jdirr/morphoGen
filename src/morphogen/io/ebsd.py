"""Orientation maps of electron backscatter diffraction (EBSD): the text files of the
acquisition systems, and the grains of a map.

Two formats are read and written: ``.ang`` (EDAX/TSL OIM: Euler angles in radians, a
header of comments) and ``.ctf`` (Oxford Instruments/HKL Channel 5: Euler angles in
degrees, a header of keys). A map (:class:`EBSDMap`) holds, at each point of a
rectangular grid of steps ``(dx, dy)``, the phase (0: a point that was not indexed) and
the orientation of the crystal (Bunge Euler angles, degrees), with the measures of
quality of the file; its rows are those of the file, from the top of the image. A
hexagonal grid (``.ang``) is read row by row onto a rectangular one of the same steps:
every other row is then half a step away from where it was measured.

**Axes.** The axes of a map are those of its image as the acquisition systems show it:
the first to the right (along the columns), the second upwards (against the rows, which
the files number from the top), the third towards the viewer, out of the sample. A
``.ctf`` file gives the orientations in these axes. An ``.ang`` file gives them in axes
turned about the third one, by an angle that depends on a setting of the software
(:data:`ANG_SETTINGS`): with the usual one its first axis (the rolling direction) points
down the image and its second one to the right. The maps are read with their
orientations brought into the axes of the image (``axes="map"``), so that the grains of
a polycrystal and their lattices are in the same axes; ``axes="file"`` keeps the angles
of the file, and :meth:`EBSDMap.turned` brings the orientations into any other axes.

The axes of the crystals are those of morphoGen (:mod:`morphogen.texture`): for the
hexagonal and trigonal lattices the first axis along ``a = [2 -1 -1 0]``, as in an
``.ang`` file; a ``.ctf`` file has it along ``a* = [1 0 -1 0]``, and its crystals are
turned by 30 degrees about ``c`` when they are read (and back when they are written).

**Grains** (:meth:`EBSDMap.grains`). Two neighbouring points of the same phase belong to
one grain when their disorientation (:func:`morphogen.texture.disorientation`, with the
symmetries of the phase) is below a threshold; the grains are the connected sets of such
points. Each has its mean orientation (:func:`morphogen.texture.mean_orientation`) and
the spread of its points about it. The grains of fewer points than a minimum are
dropped, and the points without a grain (those that were not indexed, the dropped ones)
take the grain of the nearest point that has one. The result (:class:`Grains`) gives a
polycrystal (:meth:`Grains.microstructure`: the map extruded through a thickness, with
the labels and the orientations of its grains) for the solvers of the homogenization.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy import ndimage

from morphogen.domain import Domain
from morphogen.orientations import euler_to_matrix, matrix_to_euler, uniform_unit_vectors
from morphogen.seeding import make_rng
from morphogen.texture import CLASSES, _nearest_rotation, _turns, ipf_colours, symmetries

#: The symmetry of a phase in an ``.ang`` file (the point group of its proper rotations)
#: as a Laue class of :data:`morphogen.texture.CLASSES`
ANG_SYMMETRY = {43: "cubic", 23: "m-3", 62: "hexagonal", 6: "6/m", 42: "tetragonal",
                4: "4/m", 32: "trigonal", 3: "-3", 22: "orthorhombic", 2: "monoclinic",
                1: "triclinic"}  # fmt: skip
#: The Laue group of a phase in a ``.ctf`` file
CTF_LAUE = {11: "cubic", 10: "m-3", 9: "hexagonal", 8: "6/m", 7: "trigonal", 6: "-3",
            5: "tetragonal", 4: "4/m", 3: "orthorhombic", 2: "monoclinic",
            1: "triclinic"}  # fmt: skip
#: The settings of the axes of the orientations of an ``.ang`` file (numbered as by the
#: reader of MTEX): the angle (degrees) about the third axis from them to the axes of
#: the image of the map. The first axis of the orientations points down the image with
#: the setting 1 (the usual one), up with the setting 2, to the right with the setting 3
#: and to the left with the setting 4
ANG_SETTINGS = {1: 90.0, 2: -90.0, 3: 0.0, 4: 180.0}
#: The axes in which the orientations of a map are read and written
AXES = ("map", "file")
#: The classes whose first axis is along a* in a ``.ctf`` file and along a in morphoGen
_ABOUT_C = ("hexagonal", "6/m", "trigonal", "-3")


def _check_axes(axes: str, setting: int = 1) -> None:
    if axes not in AXES:
        raise ValueError(f"axes: one of {AXES}, got {axes!r}")
    if setting not in ANG_SETTINGS:
        raise ValueError(f"setting: one of {tuple(ANG_SETTINGS)}, got {setting!r}")


@dataclass(frozen=True)
class Phase:
    """A phase of a map: its name, its Laue class (one of
    :data:`morphogen.texture.CLASSES`; none when the file gives another one: its
    orientations are then compared without symmetry) and the constants of its cell (a,
    b, c and the three angles)."""

    name: str = ""
    lattice: str | None = "cubic"
    cell: tuple[float, ...] = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)

    def __post_init__(self) -> None:
        if self.lattice is not None and self.lattice not in CLASSES:
            raise ValueError(f"phase: a lattice among {CLASSES} (or none), got {self.lattice!r}")
        object.__setattr__(self, "cell", tuple(float(v) for v in self.cell))

    @property
    def c_over_a(self) -> float:
        """The ratio of the third constant of the cell to the first."""
        return self.cell[2] / self.cell[0] if len(self.cell) >= 3 and self.cell[0] > 0 else 1.0


@njit(cache=True)
def _root(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit(cache=True)
def _close(a, b, ops, count, cosine) -> bool:
    """Whether the disorientation of two crystals is below the angle of ``cosine``."""
    best = -4.0
    for s in range(count):
        trace = 0.0
        for i in range(3):
            for j in range(3):
                d = b[j, 0] * a[i, 0] + b[j, 1] * a[i, 1] + b[j, 2] * a[i, 2]
                trace += ops[s, i, j] * d
        if trace > best:
            best = trace
    return 0.5 * (best - 1.0) >= cosine


@njit(cache=True)
def _segment(g, phase, ops, counts, cosine):
    """The connected sets of neighbouring points of one phase whose disorientation is
    below the angle of ``cosine``: a label per point (-1: not indexed), from 0."""
    ny, nx = phase.shape
    parent = np.arange(ny * nx)
    for iy in range(ny):
        for ix in range(nx):
            p = phase[iy, ix]
            if p <= 0:
                continue
            i = iy * nx + ix
            if ix + 1 < nx and phase[iy, ix + 1] == p:
                if _close(g[iy, ix], g[iy, ix + 1], ops[p], counts[p], cosine):
                    a, b = _root(parent, i), _root(parent, i + 1)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
            if iy + 1 < ny and phase[iy + 1, ix] == p:
                if _close(g[iy, ix], g[iy + 1, ix], ops[p], counts[p], cosine):
                    a, b = _root(parent, i), _root(parent, i + nx)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
    labels = np.full(ny * nx, -1, dtype=np.int32)
    count = 0
    for i in range(ny * nx):
        if phase[i // nx, i % nx] <= 0:
            continue
        r = _root(parent, i)
        if labels[r] < 0:
            labels[r] = count
            count += 1
        labels[i] = labels[r]
    return labels.reshape(ny, nx), count


@njit(cache=True)
def _grain_sums(g, labels, phase, reference, ops, counts, total, angle):
    """For each grain, the sum of the matrices of its points (``total``), each taken as
    the symmetric equivalent nearest to the reference of the grain, and the sum of their
    disorientations from it (``angle``, radians)."""
    total[:, :, :] = 0.0
    angle[:] = 0.0
    ny, nx = labels.shape
    for iy in range(ny):
        for ix in range(nx):
            k = labels[iy, ix]
            if k < 0:
                continue
            p = phase[iy, ix]
            best = -4.0
            which = 0
            for s in range(counts[p]):
                trace = 0.0
                for i in range(3):
                    for j in range(3):
                        d = 0.0
                        for m in range(3):
                            d += g[iy, ix, j, m] * reference[k, i, m]
                        trace += ops[p, s, i, j] * d
                if trace > best:
                    best = trace
                    which = s
            for i in range(3):
                for m in range(3):
                    e = 0.0
                    for j in range(3):
                        e += ops[p, which, i, j] * g[iy, ix, j, m]
                    total[k, i, m] += e
            angle[k] += math.acos(min(1.0, max(-1.0, 0.5 * (best - 1.0))))


@dataclass(frozen=True)
class Grains:
    """The grains of an orientation map (:meth:`EBSDMap.grains`)."""

    #: The grain of each point, ``(ny, nx)``; -1: none
    labels: np.ndarray
    #: The mean orientation of each grain (Bunge Euler angles, degrees), ``(n, 3)``
    orientations: np.ndarray
    #: The phase of each grain (from 1: ``phases[phase - 1]``)
    phase: np.ndarray
    #: The number of points of each grain
    size: np.ndarray
    #: The mean disorientation (degrees) of the indexed points of each grain from its
    #: mean orientation
    spread: np.ndarray
    #: The steps of the grid along x and y, and its unit of length
    step: tuple[float, float]
    #: The phases of the map
    phases: tuple[Phase, ...] = ()
    unit: str = "um"

    @property
    def count(self) -> int:
        return len(self.size)

    @property
    def areas(self) -> np.ndarray:
        """The area of each grain."""
        return self.size * (self.step[0] * self.step[1])

    @property
    def diameters(self) -> np.ndarray:
        """The diameter of the disc of the area of each grain."""
        return 2.0 * np.sqrt(self.areas / math.pi)

    def lattice(self, phase: int = 1) -> str:
        """The Laue class of a phase (``"triclinic"`` when the file does not give it)."""
        known = self.phases[phase - 1].lattice if 0 < phase <= len(self.phases) else None
        return known or "triclinic"

    def table(self) -> dict:
        """The grains as the columns of a table (:func:`morphogen.io.write_table`): the
        number of each grain, its phase, its Bunge angles (degrees), its number of
        points, its area, its diameter and the spread of its orientations (degrees)."""
        return {
            "grain": np.arange(self.count),
            "phase": self.phase,
            "phi1": self.orientations[:, 0],
            "Phi": self.orientations[:, 1],
            "phi2": self.orientations[:, 2],
            "points": self.size,
            "area": self.areas,
            "diameter": self.diameters,
            "spread": self.spread,
        }

    def boundaries(self):
        """The boundaries between the grains of one phase: the pairs of neighbouring
        grains ``(b, 2)``, the length of the boundary of each pair (the sides of the
        points that they share, in the unit of the map) and its disorientation
        (degrees, with the symmetries of the phase)."""
        from morphogen.texture import disorientation

        lab = self.labels
        found: dict[tuple[int, int], float] = {}
        # (neighbours along the columns share a side of the step of the rows, and conversely)
        sides = ((lab[:, :-1], lab[:, 1:], self.step[1]), (lab[:-1, :], lab[1:, :], self.step[0]))
        for first, second, length in sides:
            keep = (first != second) & (first >= 0) & (second >= 0)
            a, b = first[keep], second[keep]
            keep = self.phase[a] == self.phase[b]
            low, high = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
            pairs, counts = np.unique(np.stack([low, high], axis=1), axis=0, return_counts=True)
            for (i, j), n in zip(pairs.tolist(), counts.tolist(), strict=True):
                found[(i, j)] = found.get((i, j), 0.0) + n * length
        if not found:
            return np.empty((0, 2), dtype=np.int64), np.empty(0), np.empty(0)
        pairs = np.array(sorted(found), dtype=np.int64)
        lengths = np.array([found[tuple(p)] for p in pairs.tolist()])
        angles = np.empty(len(pairs))
        g = euler_to_matrix(self.orientations)
        for k in np.unique(self.phase[pairs[:, 0]]):
            of = self.phase[pairs[:, 0]] == k
            angles[of] = disorientation(g[pairs[of, 0]], g[pairs[of, 1]], self.lattice(int(k)))
        return pairs, lengths, angles

    def summary(self, *, poles=None, halfwidth: float = 10.0) -> dict:
        """The numbers of the grains of a map, phase by phase: the grains, their share
        of the area, their mean diameter (by number and by area), the mean spread inside
        them, the number of their boundaries with the mean of their disorientations by
        length (40.74 degrees for cubic crystals at random) and, for the cubic and
        hexagonal phases, the largest density of the pole figures of the families
        ``poles`` (lists of indices; {100}, {110}, {111} or (0001), {10-10}, {10-11} by
        default), the grains weighted by their areas, in multiples of a random texture."""
        from morphogen.texture import pole_figure

        pairs, lengths, angles = self.boundaries()
        out = {"grains": self.count, "unit": self.unit, "phases": []}
        defaults = {"cubic": ((1, 0, 0), (1, 1, 0), (1, 1, 1)),
                    "hexagonal": ((0, 0, 0, 1), (1, 0, -1, 0), (1, 0, -1, 1))}  # fmt: skip
        for k, phase in enumerate(self.phases, start=1):
            of = self.phase == k
            if not of.any():
                continue
            areas = self.areas[of]
            between = of[pairs[:, 0]] if len(pairs) else np.zeros(0, dtype=bool)
            entry = {
                "phase": k,
                "name": phase.name or f"phase {k}",
                "lattice": self.lattice(k),
                "grains": int(of.sum()),
                "area_fraction": float(areas.sum() / self.areas.sum()),
                "mean_diameter": float(self.diameters[of].mean()),
                "mean_diameter_by_area": float(np.sum(areas * self.diameters[of]) / areas.sum()),
                "mean_spread": float(self.spread[of].mean()),
                "boundaries": int(between.sum()),
            }
            if between.any():
                entry["mean_disorientation"] = float(
                    np.sum(lengths[between] * angles[between]) / lengths[between].sum()
                )
            family = {"m-3": "cubic", "6/m": "hexagonal"}.get(entry["lattice"], entry["lattice"])
            if family in defaults:
                ratio = phase.c_over_a if family == "hexagonal" else None
                entry["pole_figures"] = {}
                for indices in poles or defaults[family]:
                    figure = pole_figure(self.orientations[of], indices, family, weights=areas,
                                         halfwidth=halfwidth, c_over_a=ratio)  # fmt: skip
                    name = "".join(str(int(v)) for v in indices)
                    entry["pole_figures"][name] = figure.maximum
            out["phases"].append(entry)
        return out

    def microstructure(self, thickness: float | None = None, *, periodic: bool = False):
        """The polycrystal of the map: a lattice of one cell through a thickness (the
        smaller step by default), each cell with the phase, the label and the
        orientation of its grain (:class:`~morphogen.models.lattice_field.LatticeField`;
        the points without a grain are of phase 0), in the axes of the image of the map:
        the last row of the map along the first axis, at the origin."""
        from morphogen.models.lattice_field import LatticeField

        ny, nx = self.labels.shape
        depth = float(min(self.step) if thickness is None else thickness)
        domain = Domain((nx * self.step[0], ny * self.step[1], depth), periodic=periodic,
                        unit=self.unit)  # fmt: skip
        labels = np.ascontiguousarray(self.labels[::-1].T[:, :, None])
        phases = np.where(labels >= 0, self.phase[np.maximum(labels, 0)], 0).astype(np.uint8)
        return LatticeField(domain, phases, labels=labels, orientations=self.orientations)


class EBSDMap:
    """An orientation map on a rectangular grid.

    Parameters
    ----------
    euler
        Bunge Euler angles (degrees) of the crystal of each point, ``(ny, nx, 3)``: the
        rows of the file, from the top of the image (the second axis points against
        them), and the columns along the first axis.
    phase
        The phase of each point, ``(ny, nx)``: 0 for a point that was not indexed, ``k``
        for the phase ``phases[k - 1]``.
    step
        The steps of the grid along x and y.
    phases
        The phases of the map (one cubic phase by default).
    quality
        Other arrays ``(ny, nx)`` of the file by name (``"iq"`` and ``"ci"`` of an
        ``.ang`` file; ``"bands"``, ``"error"``, ``"mad"``, ``"bc"`` and ``"bs"`` of a
        ``.ctf`` file).
    """

    def __init__(self, euler, phase, step=(1.0, 1.0), phases=None, quality=None,
                 unit: str = "um") -> None:  # fmt: skip
        self.euler = np.ascontiguousarray(euler, dtype=float)
        self.phase = np.ascontiguousarray(phase, dtype=np.int32)
        if self.euler.ndim != 3 or self.euler.shape[2] != 3:
            raise ValueError("euler: an array (ny, nx, 3)")
        if self.phase.shape != self.euler.shape[:2]:
            raise ValueError("phase: an array (ny, nx), as the Euler angles")
        self.step = (float(step[0]), float(step[1]))
        if not (self.step[0] > 0 and self.step[1] > 0):
            raise ValueError("step: two positive lengths")
        self.phases = tuple(phases) if phases else (Phase(),)
        if self.phase.max(initial=0) > len(self.phases) or self.phase.min(initial=0) < 0:
            raise ValueError("phase: from 0 (not indexed) to the number of phases")
        self.quality = {k: np.asarray(v) for k, v in (quality or {}).items()}
        for name, values in self.quality.items():
            if values.shape != self.phase.shape:
                raise ValueError(f"quality {name!r}: an array (ny, nx)")
        self.unit = str(unit)

    # -- description ---------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """The numbers of rows and of columns."""
        return self.phase.shape

    @property
    def size(self) -> tuple[float, float]:
        """The lengths of the map along x and y."""
        return (self.shape[1] * self.step[0], self.shape[0] * self.step[1])

    @property
    def indexed(self) -> np.ndarray:
        """The points that were indexed."""
        return self.phase > 0

    def lattice(self, phase: int = 1) -> str:
        """The Laue class of a phase (``"triclinic"`` when the file does not give it)."""
        return self.phases[phase - 1].lattice or "triclinic"

    def orientations(self, phase: int | None = None) -> np.ndarray:
        """The Euler angles ``(n, 3)`` of the indexed points (of one phase)."""
        mask = self.indexed if phase is None else self.phase == phase
        return self.euler[mask]

    def colours(self, direction=(0.0, 0.0, 1.0)) -> np.ndarray:
        """The map as an image ``(ny, nx, 3)`` of colours (from 0 to 1): each point by the
        direction of its crystal along a direction of the sample
        (:func:`morphogen.texture.ipf_colours`); black where it was not indexed, grey for
        a phase whose lattice has no colour key."""
        out = np.zeros((*self.shape, 3))
        for k, phase in enumerate(self.phases, start=1):
            mask = self.phase == k
            if not mask.any():
                continue
            key = {"cubic": "cubic", "m-3": "cubic", "hexagonal": "hexagonal",
                   "6/m": "hexagonal"}.get(phase.lattice)  # fmt: skip
            if key:
                out[mask] = ipf_colours(self.euler[mask], direction, key)
            else:
                out[mask] = 0.5
        return out

    # -- other maps ----------------------------------------------------------------------

    def _with(self, **changes) -> EBSDMap:
        fields = {"euler": self.euler, "phase": self.phase, "step": self.step,
                  "phases": self.phases, "quality": self.quality, "unit": self.unit}  # fmt: skip
        fields.update(changes)
        return EBSDMap(**fields)

    def turned(self, rotation) -> EBSDMap:
        """The map with its orientations in other axes of the sample: ``rotation`` (3 x
        3) gives the new axes in the old ones (its rows), so that ``g`` becomes ``g
        rotation^T``. The positions of the points are not changed."""
        r = np.asarray(rotation, dtype=float).reshape(3, 3)
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-9) or np.linalg.det(r) < 0:
            raise ValueError("rotation: a rotation matrix")
        g = euler_to_matrix(self.euler.reshape(-1, 3)) @ r.T
        euler = matrix_to_euler(g).reshape(self.euler.shape)
        return self._with(euler=np.where(self.indexed[..., None], euler, 0.0))

    def _about_c(self, angle: float) -> EBSDMap:
        """The map with the crystals of its hexagonal and trigonal phases described in
        axes turned by ``angle`` degrees about ``c`` (the third Euler angle shifted)."""
        euler = self.euler.copy()
        for k, phase in enumerate(self.phases, start=1):
            if phase.lattice in _ABOUT_C:
                mask = self.phase == k
                euler[mask, 2] = np.mod(euler[mask, 2] - angle, 360.0)
        return self._with(euler=euler)

    def _about_normal(self, angle: float) -> EBSDMap:
        """The map with its orientations in axes of the sample turned by ``angle``
        degrees about the third one (the first Euler angle shifted)."""
        if angle == 0.0:
            return self
        euler = self.euler.copy()
        mask = self.indexed
        euler[mask, 0] = np.mod(euler[mask, 0] - angle, 360.0)
        return self._with(euler=euler)

    def cropped(self, columns: slice, rows: slice) -> EBSDMap:
        """The part of the map of some columns and rows."""
        quality = {k: v[rows, columns] for k, v in self.quality.items()}
        return self._with(euler=self.euler[rows, columns], phase=self.phase[rows, columns],
                          quality=quality)  # fmt: skip

    @classmethod
    def from_grains(cls, labels, orientations, step=(1.0, 1.0), *, phase=None, phases=None,
                    unit: str = "um") -> EBSDMap:  # fmt: skip
        """The map of an image of labels ``(ny, nx)``, its rows from the top, whose
        grains have the given orientations (and phases, 1 by default); negative labels
        are points that were not indexed. The section of a computed polycrystal
        (:func:`synthetic_map`), to be written for the tools that read the maps."""
        lab = np.asarray(labels)
        e = np.asarray(orientations, dtype=float).reshape(-1, 3)
        of = np.ones(len(e), dtype=np.int32) if phase is None else np.asarray(phase, np.int32)
        inside = lab >= 0
        index = np.maximum(lab, 0)
        return cls(np.where(inside[..., None], e[index], 0.0), np.where(inside, of[index], 0),
                   step, phases, unit=unit)  # fmt: skip

    # -- grains --------------------------------------------------------------------------

    def grains(self, threshold: float = 5.0, *, min_size: int = 1, fill: bool = True) -> Grains:
        """The grains of the map: connected sets of neighbouring points of one phase
        whose disorientation is below ``threshold`` degrees, those of fewer than
        ``min_size`` points dropped; with ``fill`` the points without a grain take that
        of the nearest point that has one."""
        if not 0.0 < threshold < 180.0:
            raise ValueError("threshold: an angle between 0 and 180 degrees")
        ny, nx = self.shape
        groups = [np.eye(3)[None]] + [symmetries(p.lattice or "triclinic") for p in self.phases]
        counts = np.array([len(s) for s in groups], dtype=np.int64)
        ops = np.zeros((len(groups), counts.max(), 3, 3))
        for k, s in enumerate(groups):
            ops[k, : len(s)] = s
        g = euler_to_matrix(self.euler.reshape(-1, 3)).reshape(ny, nx, 3, 3)
        labels, count = _segment(g, self.phase, ops, counts, math.cos(math.radians(threshold)))
        size = np.bincount(labels[labels >= 0], minlength=count)
        if min_size > 1:  # the small grains: dropped, the others numbered again
            keep = size >= min_size
            new = np.where(keep, np.cumsum(keep) - 1, -1).astype(np.int32)
            labels = np.where(labels >= 0, new[np.maximum(labels, 0)], -1).astype(np.int32)
            count = int(keep.sum())
        if count == 0:
            raise ValueError(
                "grains: no grain in the map (no point indexed, or min_size too large)"
            )
        measured = labels >= 0
        # the phase and a reference of each grain: its first point
        first = np.full(count, -1, dtype=np.int64)
        flat = labels.ravel()
        order = np.flatnonzero(flat >= 0)
        first[flat[order][::-1]] = order[::-1]
        phase = self.phase.ravel()[first].astype(np.int32)
        reference = g.reshape(-1, 3, 3)[first].copy()
        total, angle = np.empty((count, 3, 3)), np.empty(count)
        for _ in range(2):
            _grain_sums(g, labels, self.phase, reference, ops, counts, total, angle)
            reference = np.stack([_nearest_rotation(t) for t in total])
        _grain_sums(g, labels, self.phase, reference, ops, counts, total, angle)
        points = np.bincount(labels[measured], minlength=count)
        spread = np.degrees(angle / np.maximum(points, 1))
        if fill and not measured.all():
            nearest = ndimage.distance_transform_edt(
                ~measured, sampling=(self.step[1], self.step[0]), return_distances=False,
                return_indices=True,
            )  # fmt: skip
            labels = labels[nearest[0], nearest[1]]
        size = np.bincount(labels[labels >= 0], minlength=count)
        return Grains(np.ascontiguousarray(labels, dtype=np.int32), matrix_to_euler(reference),
                      phase, size, spread, self.step, self.phases, self.unit)  # fmt: skip


# -- the .ang files of EDAX/TSL ------------------------------------------------------------


def _ang_phases(header: list[str]) -> tuple[Phase, ...]:
    phases, current = [], None
    for line in header:
        text = line.lstrip("#").strip()
        if re.match(r"Phase\s+\d+", text):
            current = {"name": "", "lattice": None, "cell": (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)}
            phases.append(current)
        elif current is not None:
            match = re.match(r"(\w+)\s*(.*)", text)
            key, value = (match.group(1), match.group(2).strip()) if match else ("", "")
            if key == "MaterialName":
                current["name"] = value
            elif key == "Symmetry":
                current["lattice"] = ANG_SYMMETRY.get(int(float(value.split()[0])))
            elif key == "LatticeConstants":
                current["cell"] = tuple(float(v) for v in value.split()[:6])
    return tuple(Phase(**p) for p in phases) or (Phase(),)


def read_ang(path, *, axes: str = "map", setting: int = 1) -> EBSDMap:
    """The map of an ``.ang`` file (EDAX/TSL OIM): the columns phi1, PHI, phi2 (radians),
    x, y, image quality, confidence index and phase of its rows, on the square or
    hexagonal grid of its header. A point is not indexed where its confidence index is
    negative or its angles are out of range. The orientations are brought into the axes
    of the map (``axes="map"``; ``setting``: that of the system, one of
    :data:`ANG_SETTINGS`) or left in those of the file (``axes="file"``)."""
    _check_axes(axes, setting)
    with open(path, encoding="utf-8", errors="replace") as handle:
        header = [line for line in handle if line.startswith("#")]
    keys = {}
    for line in header:
        match = re.match(r"#\s*(GRID|XSTEP|YSTEP|NCOLS_ODD|NCOLS_EVEN|NROWS)\s*:\s*(\S+)", line)
        if match:
            keys[match.group(1)] = match.group(2)
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 5:
        raise ValueError(f"{path}: an .ang file has five columns at least (angles, x, y)")
    x, y = data[:, 3], data[:, 4]
    dx = float(keys["XSTEP"]) if "XSTEP" in keys else _step(x)
    dy = float(keys["YSTEP"]) if "YSTEP" in keys else _step(y)
    hexagonal = keys.get("GRID", "SqrGrid").lower().startswith("hex")
    rows = np.rint((y - y.min()) / dy).astype(np.int64)
    shift = np.where(rows % 2 == 1, 0.5, 0.0) if hexagonal else 0.0
    columns = np.rint((x - x.min()) / dx - shift).astype(np.int64)
    columns = np.maximum(columns, 0)
    ny, nx = int(rows.max()) + 1, int(columns.max()) + 1
    phases = _ang_phases(header)
    euler = np.degrees(data[:, :3])
    ci = data[:, 6] if data.shape[1] > 6 else np.zeros(len(data))
    indexed = (ci >= 0.0) & np.all(
        (data[:, :3] >= -1e-6) & (data[:, :3] <= 2 * math.pi + 1e-3), axis=1
    )
    listed = data[:, 7].astype(np.int64) if data.shape[1] > 7 else np.zeros(len(data), np.int64)
    if len(phases) == 1:
        number = np.ones(len(data), dtype=np.int64)
    else:
        number = np.clip(listed, 0, len(phases))
        indexed &= number > 0
    out_euler = np.zeros((ny, nx, 3))
    out_phase = np.zeros((ny, nx), dtype=np.int32)
    out_euler[rows, columns] = np.where(indexed[:, None], euler, 0.0)
    out_phase[rows, columns] = np.where(indexed, number, 0)
    quality = {}
    for name, column in (("iq", 5), ("ci", 6)):
        if data.shape[1] > column:
            values = np.zeros((ny, nx))
            values[rows, columns] = data[:, column]
            quality[name] = values
    ebsd = EBSDMap(out_euler, out_phase, (dx, dy), phases, quality)
    return ebsd._about_normal(ANG_SETTINGS[setting]) if axes == "map" else ebsd


def _step(values: np.ndarray) -> float:
    """The step of a grid from the coordinates of its points."""
    unique = np.unique(np.round(values, 9))
    return float(np.min(np.diff(unique))) if unique.size > 1 else 1.0


def write_ang(path, ebsd: EBSDMap, *, axes: str = "map", setting: int = 1) -> Path:
    """Writes a map as an ``.ang`` file on a square grid (angles in radians; a point that
    was not indexed has a confidence index of -1). ``axes="map"``: the orientations of
    the map are in the axes of its image, and are written in those of the system
    (``setting``); ``axes="file"``: they are written as they are."""
    _check_axes(axes, setting)
    if axes == "map":
        ebsd = ebsd._about_normal(-ANG_SETTINGS[setting])
    path = Path(path)
    ny, nx = ebsd.shape
    codes = {v: k for k, v in ANG_SYMMETRY.items()}
    lines = ["# TEM_PIXperUM          1.000000", "# x-star                0.500000",
             "# y-star                0.500000", "# z-star                0.500000",
             "# WorkingDistance       15.000000", "#"]  # fmt: skip
    for k, phase in enumerate(ebsd.phases, start=1):
        cell = "  ".join(f"{v:.3f}" for v in phase.cell[:6])
        lines += [f"# Phase {k}", f"# MaterialName  \t{phase.name or f'phase {k}'}",
                  "# Formula     \t", "# Info \t\t",
                  f"# Symmetry              {codes.get(phase.lattice, 1)}",
                  f"# LatticeConstants      {cell}", "# NumberFamilies        0", "#"]  # fmt: skip
    lines += ["# GRID: SqrGrid", f"# XSTEP: {ebsd.step[0]:.6f}", f"# YSTEP: {ebsd.step[1]:.6f}",
              f"# NCOLS_ODD: {nx}", f"# NCOLS_EVEN: {nx}", f"# NROWS: {ny}", "#",
              "# OPERATOR: \tmorphoGen", "# SAMPLEID: \t", "# SCANID: \t", "#"]  # fmt: skip
    yy, xx = np.meshgrid(np.arange(ny) * ebsd.step[1], np.arange(nx) * ebsd.step[0], indexing="ij")
    indexed = ebsd.indexed
    table = np.column_stack([
        np.radians(ebsd.euler.reshape(-1, 3)), xx.ravel(), yy.ravel(),
        ebsd.quality.get("iq", np.where(indexed, 100.0, 0.0)).ravel(),
        np.where(indexed, ebsd.quality.get("ci", np.ones((ny, nx))), -1.0).ravel(),
        (ebsd.phase if len(ebsd.phases) > 1 else np.zeros((ny, nx))).ravel(),
        np.ones(ny * nx), np.zeros(ny * nx),
    ])  # fmt: skip
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        np.savetxt(handle, table, fmt="%9.5f %9.5f %9.5f %12.5f %12.5f %8.1f %6.3f %2d %6d %6.3f")
    return path


# -- the .ctf files of Oxford Instruments/HKL ----------------------------------------------


def read_ctf(path, *, axes: str = "map") -> EBSDMap:
    """The map of a ``.ctf`` file (Channel Text File): the phases of its header, and the
    columns Phase, X, Y, Euler1, Euler2, Euler3 (degrees) of its rows, with the bands,
    the error, the mean angular deviation and the contrasts; the phase 0 is a point that
    was not indexed. Decimal commas are read. The orientations of the file are in the
    axes of the image of the map (``axes``: both values give them as they are); the
    hexagonal and trigonal crystals are turned to the axes of morphoGen (the first along
    ``a``)."""
    _check_axes(axes)
    with open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()
    keys: dict[str, str] = {}
    phases: list[Phase] = []
    start = None
    k = 0
    while k < len(lines):
        parts = lines[k].split("\t") if "\t" in lines[k] else lines[k].split()
        if not parts:
            k += 1
            continue
        if parts[0] == "Phases":
            for line in lines[k + 1 : k + 1 + int(parts[1])]:
                item = line.replace(",", ".").split("\t")
                cell = [float(v) for v in item[0].split(";")] + [
                    float(v) for v in item[1].split(";")
                ]
                laue = int(item[3]) if len(item) > 3 and item[3].strip() else 0
                phases.append(Phase(item[2] if len(item) > 2 else "", CTF_LAUE.get(laue), cell))
            k += int(parts[1])
        elif parts[0] == "Phase" and "Euler1" in parts:
            start = k
            break
        elif len(parts) >= 2:
            keys[parts[0]] = parts[1]
        k += 1
    if start is None:
        raise ValueError(
            f"{path}: no table of points (a line 'Phase X Y ... Euler1 Euler2 Euler3')"
        )
    names = [name.strip() for name in lines[start].split("\t")]
    body = lines[start + 1 :]
    if body and "," in body[0]:
        body = [line.replace(",", ".") for line in body]
    data = np.loadtxt(body, ndmin=2)
    column = {name: data[:, i] for i, name in enumerate(names) if i < data.shape[1]}
    x, y = column["X"], column["Y"]
    dx = float(keys["XStep"].replace(",", ".")) if "XStep" in keys else _step(x)
    dy = float(keys["YStep"].replace(",", ".")) if "YStep" in keys else _step(y)
    columns = np.rint((x - x.min()) / dx).astype(np.int64)
    rows = np.rint((y - y.min()) / dy).astype(np.int64)
    nx = int(keys["XCells"]) if "XCells" in keys else int(columns.max()) + 1
    ny = int(keys["YCells"]) if "YCells" in keys else int(rows.max()) + 1
    number = np.clip(column["Phase"].astype(np.int64), 0, max(len(phases), 1))
    euler = np.zeros((ny, nx, 3))
    phase = np.zeros((ny, nx), dtype=np.int32)
    angles = np.column_stack([column["Euler1"], column["Euler2"], column["Euler3"]])
    euler[rows, columns] = np.where(number[:, None] > 0, angles, 0.0)
    phase[rows, columns] = number
    quality = {}
    for name in ("Bands", "Error", "MAD", "BC", "BS"):
        if name in column:
            values = np.zeros((ny, nx))
            values[rows, columns] = column[name]
            quality[name.lower()] = values
    return EBSDMap(euler, phase, (dx, dy), tuple(phases) or None, quality)._about_c(30.0)


def write_ctf(path, ebsd: EBSDMap, *, axes: str = "map") -> Path:
    """Writes a map as a ``.ctf`` file (angles in degrees; the phase 0 for a point that
    was not indexed). The orientations are written as they are: the axes of the image of
    the map are those of the file (``axes``: both values). The hexagonal and trigonal
    crystals are written with their first axis along ``a*``."""
    _check_axes(axes)
    ebsd = ebsd._about_c(-30.0)
    path = Path(path)
    ny, nx = ebsd.shape
    codes = {v: k for k, v in CTF_LAUE.items()}
    settings = ("Mag\t0.0000\tCoverage\t0\tDevice\t0\tKV\t0.0000\tTiltAngle\t0.0000\tTiltAxis\t0"
                "\tDetectorOrientationE1\t0.0000\tDetectorOrientationE2\t0.0000"
                "\tDetectorOrientationE3\t0.0000\tWorkingDistance\t0.0000"
                "\tInsertionDistance\t0.0000\t")  # fmt: skip
    lines = ["Channel Text File", "Prj\tmorphoGen", "Author\tmorphoGen", "JobMode\tGrid",
             f"XCells\t{nx}", f"YCells\t{ny}", f"XStep\t{ebsd.step[0]:.6f}",
             f"YStep\t{ebsd.step[1]:.6f}", "AcqE1\t0.0000", "AcqE2\t0.0000", "AcqE3\t0.0000",
             f"Euler angles refer to Sample Coordinate system (CS0)!\t{settings}",
             f"Phases\t{len(ebsd.phases)}"]  # fmt: skip
    for k, phase in enumerate(ebsd.phases, start=1):
        cell = list(phase.cell[:6]) + [90.0] * (6 - len(phase.cell[:6]))
        lengths = ";".join(f"{v:.3f}" for v in cell[:3])
        angles = ";".join(f"{v:.3f}" for v in cell[3:6])
        lines.append(f"{lengths}\t{angles}\t{phase.name or f'phase {k}'}"
                     f"\t{codes.get(phase.lattice, 1)}\t0\t\t\t")  # fmt: skip
    lines.append("Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS")
    yy, xx = np.meshgrid(np.arange(ny) * ebsd.step[1], np.arange(nx) * ebsd.step[0], indexing="ij")
    indexed = ebsd.indexed
    zero = np.zeros((ny, nx))
    table = np.column_stack([
        ebsd.phase.ravel(), xx.ravel(), yy.ravel(),
        ebsd.quality.get("bands", np.where(indexed, 8, 0)).ravel(),
        ebsd.quality.get("error", zero).ravel(),
        ebsd.euler.reshape(-1, 3),
        ebsd.quality.get("mad", np.where(indexed, 0.5, 0.0)).ravel(),
        ebsd.quality.get("bc", np.where(indexed, 128, 0)).ravel(),
        ebsd.quality.get("bs", np.where(indexed, 128, 0)).ravel(),
    ])  # fmt: skip
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        np.savetxt(handle, table, delimiter="\t",
                   fmt=["%d", "%.4f", "%.4f", "%d", "%d", "%.4f", "%.4f", "%.4f", "%.4f", "%d",
                        "%d"])  # fmt: skip
    return path


def synthetic_map(micro, shape, *, layer: int = 0, orientations=None, phase=None, phases=None,
                  scatter: float = 0.0, missing: float = 0.0, boundaries: float = 0.0,
                  seed=None) -> EBSDMap:  # fmt: skip
    """The map that an acquisition would give of a layer of a polycrystal: the cells of
    ``micro`` (a microstructure with labels and ``orientations``: a tessellation, grains
    grown or measured) on the grid ``shape`` (nx, ny, nz) of its domain, the layer
    ``layer`` along its third axis, seen from the end of that axis (the axes of the map
    are those of the domain). Each point has the orientation of its cell
    (``orientations``: other ones, per cell; ``phase``: the phase of each cell, 1 by
    default, among ``phases``), turned about a random axis by an angle of the half-normal
    law of standard deviation ``scatter`` degrees; a share ``missing`` of the points, and
    a share ``boundaries`` of those next to another cell, are not indexed."""
    if not (scatter >= 0 and 0 <= missing <= 1 and 0 <= boundaries <= 1):
        raise ValueError("synthetic_map: scatter >= 0, missing and boundaries between 0 and 1")
    image = micro.labels(shape)
    if not 0 <= layer < image.data.shape[2]:
        raise ValueError(f"layer: from 0 to {image.data.shape[2] - 1}")
    labels = np.ascontiguousarray(image.data[:, ::-1, layer].T)  # (the rows from the top)
    given = micro.orientations if orientations is None else orientations
    if given is None:
        raise ValueError("synthetic_map: the cells have no orientations (crystal_orientations)")
    step = (float(image.spacing[0]), float(image.spacing[1]))
    ebsd = EBSDMap.from_grains(labels, given, step, phase=phase, phases=phases,
                               unit=micro.domain.unit)  # fmt: skip
    rng, _ = make_rng(seed)
    ny, nx = labels.shape
    euler, number = ebsd.euler, ebsd.phase.copy()
    if scatter > 0:
        angles = np.radians(np.abs(rng.normal(0.0, scatter, ny * nx)))
        turns = _turns(uniform_unit_vectors(rng, ny * nx), angles)
        euler = matrix_to_euler(euler_to_matrix(euler.reshape(-1, 3)) @ turns).reshape(ny, nx, 3)
    lost = rng.random((ny, nx)) < missing
    if boundaries > 0:
        edge = np.zeros((ny, nx), dtype=bool)
        edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        edge[:, 1:] |= labels[:, :-1] != labels[:, 1:]
        edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
        edge[1:, :] |= labels[:-1, :] != labels[1:, :]
        lost |= edge & (rng.random((ny, nx)) < boundaries)
    number[lost] = 0
    return EBSDMap(np.where(number[..., None] > 0, euler, 0.0), number, step, ebsd.phases,
                   unit=ebsd.unit)  # fmt: skip


def example_map(name: str = "ebsd_map.ctf") -> Path:
    """The path of a map that comes with morphoGen (``ebsd_map.ctf``: the map of the
    template ``ebsd``, a section of a polycrystal of nickel as an acquisition would give
    it, written by examples/ebsd_map.py)."""
    from importlib import resources

    path = Path(str(resources.files("morphogen") / "templates" / name))
    if Path(name).suffix.lower() not in (".ang", ".ctf") or not path.is_file():
        raise ValueError(f"example_map: no map {name!r} comes with morphoGen")
    return path


def read_ebsd(path, *, axes: str = "map", setting: int = 1) -> EBSDMap:
    """The map of an ``.ang`` or a ``.ctf`` file (by its extension), its orientations in
    the axes of the map (``axes="map"``; ``setting``: that of the system of an ``.ang``
    file) or in those of the file (``axes="file"``)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".ang":
        return read_ang(path, axes=axes, setting=setting)
    if suffix == ".ctf":
        _check_axes(axes, setting)
        return read_ctf(path, axes=axes)
    raise ValueError(f"{path}: an orientation map is an .ang or a .ctf file")


def write_ebsd(path, ebsd: EBSDMap, *, axes: str = "map", setting: int = 1) -> Path:
    """Writes a map as an ``.ang`` or a ``.ctf`` file (by its extension); ``axes`` and
    ``setting`` as in :func:`read_ebsd`."""
    suffix = Path(path).suffix.lower()
    if suffix == ".ang":
        return write_ang(path, ebsd, axes=axes, setting=setting)
    if suffix == ".ctf":
        _check_axes(axes, setting)
        return write_ctf(path, ebsd, axes=axes)
    raise ValueError(f"{path}: an orientation map is an .ang or a .ctf file")


__all__ = [
    "ANG_SETTINGS",
    "ANG_SYMMETRY",
    "AXES",
    "CTF_LAUE",
    "EBSDMap",
    "Grains",
    "Phase",
    "example_map",
    "read_ang",
    "read_ctf",
    "read_ebsd",
    "synthetic_map",
    "write_ang",
    "write_ctf",
    "write_ebsd",
]
