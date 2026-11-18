"""Cellular automata and lattice growth processes.

* :mod:`morphogen.automata.lattice`: neighbourhoods (von Neumann, Moore, edges,
  Euclidean; 2D or 3D), neighbour counts and FFT neighbourhood sums;
* :mod:`morphogen.automata.rules`: transition rules (Life-like ``B/S``, majority,
  annealing, majority at conserved volume fraction, probabilistic, Young's
  activator-inhibitor);
* :mod:`morphogen.automata.engine`: synchronous, asynchronous and sequential evolution,
  with the history of the volume fraction and of the interface density;
* :mod:`morphogen.automata.reaction`: Gray-Scott reaction-diffusion on the grid and, with
  finite-volume Laplacians, on tilings;
* :mod:`morphogen.automata.tilings`: other lattices of cells (triangles, hexagons,
  Penrose rhombi, Ammann-Beenker squares and rhombi, prisms, BCC and FCC Voronoi cells,
  golden rhombohedra; the periodic approximants of the quasi-periodic ones) with CSR
  neighbourhoods.
"""

from morphogen.automata.engine import (
    UPDATES,
    CellularAutomaton,
    Evolution,
    interface_density,
    random_state,
)
from morphogen.automata.lattice import (
    NEIGHBOURHOODS,
    kernel_sum,
    neighbour_counts,
    offsets,
    sublattices,
)
from morphogen.automata.rules import (
    NAMED_RULES,
    ActivatorInhibitorRule,
    ConservedMajorityRule,
    LatticeContext,
    LifeRule,
    MajorityRule,
    ProbabilisticRule,
    Rule,
    make_rule,
)
from morphogen.automata.tilings import (
    LATTICES,
    Tiling,
    TilingContext,
    make_tiling,
)

__all__ = [
    "LATTICES",
    "NAMED_RULES",
    "NEIGHBOURHOODS",
    "UPDATES",
    "ActivatorInhibitorRule",
    "CellularAutomaton",
    "ConservedMajorityRule",
    "Evolution",
    "LatticeContext",
    "LifeRule",
    "MajorityRule",
    "ProbabilisticRule",
    "Rule",
    "Tiling",
    "TilingContext",
    "interface_density",
    "kernel_sum",
    "make_rule",
    "make_tiling",
    "neighbour_counts",
    "offsets",
    "random_state",
    "sublattices",
]
