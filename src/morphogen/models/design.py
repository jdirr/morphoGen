"""Microstructures from combinatorial design: tile assemblies and random strut lattices.

``tile_assembly``
    Compatible assemblies of building blocks (polarized blocks of combinatorial
    metamaterials, pipe networks) drawn by wave function collapse or uniformly by
    transfer matrices; their voxels form a lattice microstructure.
``random_lattice``
    Strut lattices with defects: struts removed at random down to a mean coordination,
    nodes displaced at random; the rigidity of the pin-jointed framework (mechanisms,
    self-stresses, floppy strains) is analysed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from morphogen._utils import check_fraction, check_positive
from morphogen.design.csp import sample_uniform, wave_function_collapse
from morphogen.design.frames import Framework
from morphogen.design.tiles import TILE_SETS, pipe_tiles, polarized_blocks
from morphogen.distributions import Constant, Distribution
from morphogen.domain import Domain
from morphogen.models import _sizes as sizes
from morphogen.models.affine import ANISOTROPY, Anisotropy, anisotropic
from morphogen.models.base import Model, param, register
from morphogen.models.foam import FRAME_STIFFNESS, frame_summary
from morphogen.models.lattice import CELLS
from morphogen.models.lattice_field import LatticeField, _encode
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Segments
from morphogen.seeding import SeedLike, make_rng

ASSEMBLY_METHODS = ("wfc", "uniform", "random")

#: Mean coordination of the intact lattices.
PARENT_COORDINATION = {"cubic": 6.0, "bcc": 8.0, "fcc": 6.0, "octet": 12.0}

#: Largest framework (nodes) whose rigidity is analysed.
MAX_RIGIDITY_NODES = 1000


@register
@dataclass(frozen=True)
class TileAssembly(Model):
    """Compatible assemblies of building blocks with matching faces.

    ``polarized`` blocks (combinatorial metamaterials after Coulais et al. 2016) or
    ``pipes`` (networks without dead ends at the tile boundaries).
    """

    name = "tile_assembly"
    phases: ClassVar[dict[int, str]] = {0: "void", 1: "solid"}
    count_label = "tiles"

    tiles: str = param("polarized", doc="Tile set: polarized or pipes", kind="str")
    tile_length: float = param(1.0, doc="Edge of a tile", unit="length")
    tile_cells: int = param(8, doc="Lattice cells per tile edge", kind="int")
    dimension: int = param(3, doc="3, or 2 for planar assemblies", kind="int")
    method: str = param(
        "wfc",
        doc="wfc (wave function collapse), uniform (exact uniform sampling, narrow "
        "boxes only) or random (independent tiles, ignoring compatibility)",
        kind="str",
    )
    channel_radius: float = param(
        0.25, doc="polarized: radius of the channel along the block axis (tile edges)"
    )
    pipe_radius: float = param(0.2, doc="pipes: radius of the pipes (tile edges)")
    open_probability: float = param(
        0.5, doc="pipes: probability that a face is open", kind="fraction"
    )
    degrees: tuple[float, ...] | None = param(
        None, doc="pipes: allowed numbers of open faces per tile (default: all)", kind="floats"
    )
    boundary: int | None = param(
        None,
        doc="Label the outer faces of non-periodic boxes must match (pipes: 0 closes the network)",
        kind="int",
    )
    smoothing: float = param(
        0.0, doc="Gaussian smoothing (in cells) of the interface off the lattice", unit="cells"
    )

    def _validate(self) -> None:
        if self.tiles not in TILE_SETS:
            raise ValueError(f"tiles: one of {list(TILE_SETS)}")
        check_positive(self.tile_length, "tile_length")
        if self.tile_cells < 2:
            raise ValueError("tile_cells: at least 2")
        if self.dimension not in (2, 3):
            raise ValueError("dimension: 2 or 3")
        if self.method not in ASSEMBLY_METHODS:
            raise ValueError(f"method: one of {ASSEMBLY_METHODS}")
        if not 0.0 <= self.channel_radius < 0.5:
            raise ValueError("channel_radius: in [0, 0.5)")
        if not 0.0 < self.pipe_radius < 0.5:
            raise ValueError("pipe_radius: in (0, 0.5)")
        check_fraction(self.open_probability, "open_probability")
        if self.degrees is not None and any(
            d != int(d) or not 0 <= d <= 2 * self.dimension for d in self.degrees
        ):
            raise ValueError(f"degrees: integers from 0 to {2 * self.dimension}")
        check_positive(self.smoothing, "smoothing", allow_zero=True)

    def tile_set(self):
        if self.tiles == "polarized":
            return polarized_blocks(self.tile_cells, self.channel_radius, self.dimension)
        degrees = None if self.degrees is None else [int(d) for d in self.degrees]
        return pipe_tiles(
            self.tile_cells,
            self.pipe_radius,
            self.dimension,
            open_probability=self.open_probability,
            degrees=degrees,
        )

    def tile_grid(self, domain: Domain) -> tuple[int, int, int]:
        n = np.maximum(1, np.rint(domain.lengths / self.tile_length)).astype(int)
        if self.dimension == 2:
            n[2] = 1
        return tuple(int(v) for v in n)

    def expected_count(self, domain: Domain) -> float:
        return float(np.prod(self.tile_grid(domain)))

    def generate(self, domain: Domain, seed: SeedLike = None) -> LatticeField:
        rng, seed_info = make_rng(seed)
        tiles = self.tile_set()
        shape = self.tile_grid(domain)
        periodic = domain.periodic
        if self.tiles == "polarized" and periodic and any(n % 2 for n in shape[: self.dimension]):
            raise ValueError(
                "polarized blocks alternate along every line: a periodic domain needs an "
                f"even number of tiles along each axis (got {shape[: self.dimension]})"
            )
        if self.method == "wfc":
            assembly = wave_function_collapse(
                tiles, shape, rng, periodic=periodic, boundary=self.boundary
            )
        elif self.method == "uniform":
            if self.boundary is not None:
                raise ValueError("boundary labels are only used by the wfc method")
            assembly = sample_uniform(tiles, shape, rng, periodic=periodic)
        else:
            p = tiles.weights / tiles.weights.sum()
            assembly = rng.choice(tiles.n_tiles, size=shape, p=p)
        voxels = tiles.assemble(assembly)
        return LatticeField(
            domain,
            voxels,
            smoothing=self.smoothing,
            model=self,
            seed=seed_info,
            attributes={
                "tile_grid": list(shape),
                "assembly": _encode(np.asarray(assembly, dtype=np.int16)),
                "tile_names": list(tiles.names),
                "mismatches": tiles.mismatches(assembly, periodic),
            },
        )


@register
@anisotropic
@dataclass(frozen=True)
class RandomLattice(Model):
    """Strut lattices with defects: missing struts and displaced nodes.

    Struts of a cubic, bcc, fcc or octet lattice are removed at random, down to a mean
    coordination or by a fraction, and nodes are displaced at random. The rigidity of
    the pin-jointed framework tells whether it is stretching-dominated (no floppy
    macroscopic strain; Maxwell's count needs a coordination of at least 6), and with
    ``frame_stiffness`` the stiffness of its frame of beams (rigid joints) is added to the
    summary. A distribution of the strut radius gives each strut its own (the scatter of
    the struts of printed lattices), kept in the metadata for the frame of beams.
    """

    name = "random_lattice"
    phases: ClassVar[dict[int, str]] = {0: "void", 1: "struts"}
    count_label = "unit cells"

    cell: str = param("octet", doc="Parent unit cell: cubic, bcc, fcc or octet", kind="str")
    cell_size: float = param(1.0, doc="Edge of the cubic unit cell", unit="length")
    strut_radius: Distribution = param(
        Constant(0.08),
        doc="Strut radius: a number (monodisperse) or a distribution (polydisperse, one "
        "radius per strut)",
        kind="distribution",
        unit="length",
    )
    coordination: float | None = param(
        None,
        doc="Target mean number of struts per node: struts are removed at random, "
        "keeping min_degree struts at every node",
    )
    removal_fraction: float | None = param(
        None,
        doc="Fraction of the struts removed at random (then dangling struts)",
        kind="fraction",
    )
    min_degree: int = param(
        3, doc="Fewest struts per node when diluting to a coordination", kind="int"
    )
    perturbation: float = param(
        0.0, doc="Random displacement of the nodes, in cell sizes (uniform in a ball)"
    )
    rigidity: bool = param(
        True,
        doc=f"Analyse the rigidity of the pin-jointed framework (up to {MAX_RIGIDITY_NODES} nodes)",
        kind="bool",
    )
    anisotropy: Anisotropy | None = param(None, doc=ANISOTROPY, kind="anisotropy", kw_only=True)
    frame_stiffness: bool = param(False, doc=FRAME_STIFFNESS, kind="bool", kw_only=True)

    def _validate(self) -> None:
        if self.cell not in CELLS:
            raise ValueError(f"cell: one of {list(CELLS)}")
        check_positive(self.cell_size, "cell_size")
        if self.strut_radius.upper() >= 0.5 * self.cell_size:
            raise ValueError("strut_radius: smaller than half the cell size")
        if self.coordination is not None and self.removal_fraction is not None:
            raise ValueError("give at most one of 'coordination' and 'removal_fraction'")
        if self.coordination is not None:
            check_positive(self.coordination, "coordination")
        if self.removal_fraction is not None:
            check_fraction(self.removal_fraction, "removal_fraction")
        if self.min_degree < 1:
            raise ValueError("min_degree: at least 1")
        if not 0.0 <= self.perturbation < 0.5:
            raise ValueError("perturbation: in [0, 0.5) cell sizes")

    def theory(self, domain: Domain | None = None) -> dict[str, float]:
        out = {"parent_coordination": PARENT_COORDINATION[self.cell], "isostatic_coordination": 6.0}
        if self.coordination is not None:
            out["coordination"] = float(self.coordination)
        return out

    def expected_count(self, domain: Domain) -> float:
        return domain.volume / self.cell_size**3

    def framework(self, n_cells, rng: np.random.Generator) -> Framework:
        """The random framework of ``n_cells`` unit cells."""
        frame = Framework.lattice(self.cell, n_cells, self.cell_size)
        if self.coordination is not None:
            frame = frame.diluted(rng, coordination=self.coordination, min_degree=self.min_degree)
        elif self.removal_fraction is not None:
            frame = frame.diluted(rng, fraction=self.removal_fraction)
        if self.perturbation > 0:
            frame = frame.perturbed(rng, self.perturbation * self.cell_size)
        return frame

    def generate(self, domain: Domain, seed: SeedLike = None) -> PrimitiveUnion:
        rng, seed_info = make_rng(seed)
        a = self.cell_size
        ratio = domain.lengths / a
        if domain.periodic:
            if np.any(np.abs(ratio - np.rint(ratio)) > 1e-6 * np.maximum(ratio, 1.0)):
                raise ValueError(
                    "random_lattice: a periodic domain must contain a whole number of cells"
                )
            n_cells = np.rint(ratio).astype(int)
            origin = domain.lower
        else:  # a window of a larger periodic lattice
            n_cells = np.ceil(ratio).astype(int) + 2
            origin = domain.lower - a
        frame = self.framework(n_cells, rng)
        law = self.strut_radius
        radii = sizes.draw(law, rng, frame.n_bars) if sizes.polydisperse(law) else law.moment(1)
        segments = frame.segments(radii, origin)
        if not domain.periodic:
            lo, hi = segments.bounds()
            keep = np.all((hi >= domain.lower) & (lo <= domain.upper), axis=1)
            segments = Segments(
                segments.start[keep], segments.end[keep], segments.radii[keep], round_caps=True
            )
        summary = {
            "n_nodes": frame.n_nodes,
            "n_struts": frame.n_bars,
            "coordination": round(frame.coordination(), 6),
            **sizes.size_summary("strut_radius", radii, law),
        }
        if self.rigidity and domain.periodic and frame.n_nodes <= MAX_RIGIDITY_NODES:
            r = frame.rigidity()
            summary.update(
                mechanisms=r["mechanisms"],
                self_stresses=r["self_stresses"],
                floppy_strains=r["floppy_strains"],
            )
        if self.frame_stiffness:  # the periodic lattice (larger than a window)
            summary.update(frame_summary(frame, radii))
        metadata = {"summary": summary, "framework": frame.to_dict()}
        if np.ndim(radii):  # the radius of each strut, in the order of the bars
            metadata["radii"] = np.asarray(radii).tolist()
        return PrimitiveUnion(domain, segments=segments, model=self, seed=seed_info,
                              metadata=metadata)  # fmt: skip
