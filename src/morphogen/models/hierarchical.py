"""Hierarchical lattices of tubes: frames of generation G (Rayneau-Kirkhope, Mao & Farr,
"Ultralight fractal structures from hollow tubes", Phys. Rev. Lett. 109, 204301, 2012).

A frame of generation 1 between two nodes is a column of ``n`` regular octahedra
(triangular antiprisms: consecutive triangles turned by 60 degrees) closed by two
regular tetrahedra whose apexes are the nodes; its length is ``sqrt(2/3) (n + 2) a``
for an edge ``a``. Under compression along the column, the edges of its triangles carry
tension and its lateral and end edges compression. A frame of generation G replaces
each compressed edge by a frame of generation G - 1 of the same length; a frame of
generation 0 is a tube. The lattice joins the nodes of a face-centred cubic lattice to
their 12 nearest and 6 next-nearest neighbours with such frames, made of solid or
hollow tubes. For hollow tubes, the volume of material that carries a load index ``f``
without buckling scales as ``f^((G+2)/(G+3))``, against ``f^(1/2)`` for solid struts:
the hierarchy makes very light structures stable.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

import numpy as np

from morphogen._utils import check_positive
from morphogen.domain import Domain
from morphogen.models.base import Model, model_from_config, param, register
from morphogen.models.combination import Combined
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments
from morphogen.seeding import SeedLike, make_rng

_H = math.sqrt(2.0 / 3.0)  # height of a regular octahedron or tetrahedron, per edge

#: Nodes of the face-centred cubic cell, and half of the bonds of each (the others are
#: those of the neighbouring nodes): nearest neighbours (12) and next-nearest (6).
FCC_NODES = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], dtype=float)
_NN = [
    v
    for v in itertools.product((-0.5, 0.0, 0.5), repeat=3)
    if sum(x != 0 for x in v) == 2 and next(x for x in v if x != 0) > 0
]
NEAREST = np.array(_NN, dtype=float)
NEXT_NEAREST = np.eye(3)
CONNECTIONS = ("nn+nnn", "nn", "nnn")


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to ``axis`` (unit) and to each other."""
    ref = np.eye(3)[int(np.argmin(np.abs(axis)))]
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def frame(p, q, generation: int, cells) -> tuple[list, list]:
    """Tubes of a frame of generation ``generation`` from ``p`` to ``q`` (``cells``:
    octahedra of the frames of each generation, outermost first): their ends, and the
    generation of the frame that holds each (for radii by generation)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if generation == 0:
        return [(p, q)], [0]
    n = int(cells[0])
    axis = q - p
    length = float(np.linalg.norm(axis))
    e3 = axis / length
    e1, e2 = _basis(e3)
    edge = length / (_H * (n + 2))
    radius = edge / math.sqrt(3.0)  # of the circle through the corners of a triangle
    layers = []
    for k in range(n + 1):
        centre = p + _H * edge * (k + 1) * e3
        angles = (k % 2) * math.pi / 3.0 + 2.0 * math.pi * np.arange(3) / 3.0
        ring = np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2
        layers.append(centre + radius * ring)
    tubes, owners = [], []
    for layer in layers:  # the triangles: tension, tubes of this frame
        for j in range(3):
            tubes.append((layer[j], layer[(j + 1) % 3]))
            owners.append(generation)
    compressed = [(p, v) for v in layers[0]] + [(v, q) for v in layers[-1]]
    for below, above in itertools.pairwise(layers):
        for v in below:  # to the two nearest corners of the next triangle
            near = np.argsort(np.linalg.norm(above - v, axis=1))[:2]
            compressed += [(v, above[i]) for i in near]
    for a, b in compressed:
        t, o = frame(a, b, generation - 1, cells[1:])
        tubes += t
        owners += o
    return tubes, owners


@lru_cache(maxsize=32)
def _cell_fraction(config: str, n: int) -> float:
    """Solid fraction of a unit cell of a hierarchical lattice (voxel estimate)."""
    model = model_from_config(json.loads(config))
    unit = Domain.cube(model.cell_size, periodic=True)
    micro = model.generate(unit, 0)
    return float(micro.voxelize((n, n, n)).data.astype(bool).mean())


@register
@dataclass(frozen=True)
class HierarchicalLattice(Model):
    """Hierarchical lattice: frames of generation G of (hollow) tubes on an fcc lattice.

    The nodes of the face-centred cubic lattice of edge ``cell_size`` are joined to their
    nearest (``nn``: the octet truss) and next-nearest (``nnn``: the cube edges)
    neighbours by frames of generation ``generations`` (see the module): columns of
    ``cells`` regular octahedra whose compressed edges are frames of the generation
    below, recursively (``cells``: one number, or one per generation from the outermost).
    Generation 0 is a plain strut lattice. Tubes of ``tube_radius`` (one value, or one
    per generation of the frames that hold them, from G down to 0), hollow with
    ``wall_thickness``.
    """

    name = "hierarchical_lattice"
    phases: ClassVar[dict[int, str]] = {0: "void", 1: "tubes"}
    count_label = "unit cells"

    cell_size: float = param(1.0, doc="Edge of the cubic cell of the fcc lattice", unit="length")
    generations: int = param(1, doc="Generation G of the frames (0: tubes)", kind="int")
    cells: tuple | int = param(
        4, doc="Octahedra per frame: one number, or one per generation (outermost first)",
        kind="json",
    )  # fmt: skip
    tube_radius: tuple | float = param(
        0.01, doc="Radius of the tubes, or one per generation (G down to 0)", kind="json"
    )
    wall_thickness: float | None = param(
        None, doc="Wall thickness of hollow tubes (default: solid)", unit="length"
    )
    connections: str = param(
        "nn+nnn", doc="Bonds: nn+nnn (nearest and next-nearest), nn or nnn", kind="str"
    )

    def _validate(self) -> None:
        check_positive(self.cell_size, "cell_size")
        if not 0 <= self.generations <= 3:
            raise ValueError("generations: from 0 to 3")
        if self.connections not in CONNECTIONS:
            raise ValueError(f"connections: one of {list(CONNECTIONS)}")
        counts = self.cells_per_generation
        if len(counts) != self.generations or any(n < 1 for n in counts):
            raise ValueError("cells: a positive number, or one per generation")
        radii = self.radii
        if len(radii) != self.generations + 1 or any(not r > 0 for r in radii):
            raise ValueError("tube_radius: a positive radius, or one per generation (G to 0)")
        if self.wall_thickness is not None:
            check_positive(self.wall_thickness, "wall_thickness")
            if self.wall_thickness >= min(radii):
                raise ValueError("wall_thickness: smaller than the tube radius")
        shortest = self.edges[-1]
        if 4.0 * radii[-1] > shortest:
            raise ValueError(
                f"tube_radius: the finest tubes ({shortest:.4g} long) would merge into "
                "blobs: thinner tubes, larger cells, fewer octahedra or generations"
            )

    @property
    def cells_per_generation(self) -> tuple[int, ...]:
        c = self.cells
        if np.ndim(c) == 0:
            return (int(c),) * self.generations
        return tuple(int(v) for v in c)

    @property
    def radii(self) -> tuple[float, ...]:
        """Tube radius of each generation, from G down to 0."""
        r = self.tube_radius
        if np.ndim(r) == 0:
            return (float(r),) * (self.generations + 1)
        return tuple(float(v) for v in r)

    @property
    def edges(self) -> list[float]:
        """Length of the edges of the frames of each generation, from the shortest bond
        (G) down to the finest tubes (0)."""
        bond = self.cell_size * (1.0 if self.connections == "nnn" else math.sqrt(0.5))
        out = [bond]
        for n in self.cells_per_generation:
            out.append(out[-1] / (_H * (n + 2)))
        return out

    def bonds(self) -> tuple[np.ndarray, np.ndarray]:
        """Ends of the bonds of one cell, in units of the cell size."""
        vectors = []
        if self.connections in ("nn+nnn", "nn"):
            vectors.append(NEAREST)
        if self.connections in ("nn+nnn", "nnn"):
            vectors.append(NEXT_NEAREST)
        v = np.concatenate(vectors)
        start = np.repeat(FCC_NODES, len(v), axis=0)
        return start, start + np.tile(v, (len(FCC_NODES), 1))

    def unit_tubes(self) -> Segments:
        """Tubes of one cell (anchored at the origin), with their radii."""
        start, end = self.bonds()
        tubes, owners = [], []
        for p, q in zip(start * self.cell_size, end * self.cell_size, strict=True):
            t, o = frame(p, q, self.generations, self.cells_per_generation)
            tubes += t
            owners += o
        ends = np.array(tubes)
        radii = np.array(self.radii[::-1])[np.array(owners)]  # owners: generation, 0 finest
        return Segments(ends[:, 0], ends[:, 1], radii, round_caps=True)

    def generate(self, domain: Domain, seed: SeedLike = None):
        _, seed_info = make_rng(seed)
        a = self.cell_size
        if domain.periodic:
            ratio = domain.lengths / a
            if np.any(np.abs(ratio - np.rint(ratio)) > 1e-6 * np.maximum(ratio, 1.0)):
                raise ValueError(
                    "hierarchical_lattice: a periodic domain must contain a whole number of cells"
                )
        unit = self.unit_tubes()
        lower = domain.lower - (0.0 if domain.periodic else a)
        upper = domain.upper + (0.0 if domain.periodic else a)
        lo = np.floor(lower / a).astype(int)
        hi = np.ceil(upper / a).astype(int) - 1
        shifts = np.array(
            list(itertools.product(*[range(lo[d], max(hi[d], lo[d]) + 1) for d in range(3)])),
            dtype=float,
        )
        shifts *= a
        segments = Segments(
            (shifts[:, None, :] + unit.start[None]).reshape(-1, 3),
            (shifts[:, None, :] + unit.end[None]).reshape(-1, 3),
            np.tile(unit.radii, len(shifts)),
            round_caps=True,
        )
        if not domain.periodic:
            seg_lo, seg_hi = segments.bounds()
            segments = segments.subset(
                np.all((seg_hi >= domain.lower) & (seg_lo <= domain.upper), axis=1)
            )
        outer = PrimitiveUnion(domain, segments=segments, model=self, seed=seed_info)
        if self.wall_thickness is None:
            return outer
        from morphogen.models.nested import bores_of

        bores = bores_of(outer, self.wall_thickness)
        return Combined(domain, "difference", [outer, bores], model=self, seed=seed_info)

    def expected_count(self, domain: Domain) -> float:
        return domain.volume / self.cell_size**3

    def solid_fraction(self, n: int = 96) -> float:
        """Solid fraction (voxel estimate on ``n^3`` voxels of a cell)."""
        return _cell_fraction(json.dumps(self.to_config(), sort_keys=True), int(n))

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        """Number and length of the tubes; the volume fraction (estimated on voxels)."""
        unit = self.unit_tubes()
        length = float(unit.lengths.sum())
        edges = self.edges
        out = {
            "tubes_per_cell": float(len(unit)),
            "tube_length_per_cell": length,
            "finest_edge": edges[-1],
        }
        h = edges[-1] / 6.0  # a few voxels across the finest tubes
        n = int(min(256, max(48, math.ceil(self.cell_size / max(h, 1e-12)))))
        out["volume_fraction"] = self.solid_fraction(n)
        return out
