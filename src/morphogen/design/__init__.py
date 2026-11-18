"""Combinatorial design: tile assemblies, strut frameworks and search algorithms.

* :mod:`morphogen.design.tiles`: tile sets with labelled faces (polarized blocks of
  combinatorial metamaterials, pipe networks), rotations, geometry of assemblies;
* :mod:`morphogen.design.csp`: compatible assemblies by wave function collapse with
  backtracking, exact counting and uniform sampling by transfer matrices, exact count
  of polarized-block assemblies;
* :mod:`morphogen.design.frames`: periodic pin-jointed frameworks, random dilution and
  perturbation, Maxwell-Calladine rigidity analysis;
* :mod:`morphogen.design.search`: simulated annealing, genetic algorithm, inverse
  design of cellular-automaton rules.
"""

from morphogen.design.csp import (
    TransferMatrix,
    count_assemblies,
    count_polarized_blocks,
    sample_uniform,
    wave_function_collapse,
)
from morphogen.design.frames import Framework
from morphogen.design.search import (
    RuleFit,
    SearchResult,
    descriptor_cost,
    fit_rule,
    genetic_algorithm,
    simulated_annealing,
    structure_descriptors,
)
from morphogen.design.tiles import TILE_SETS, TileSet, pipe_tiles, polarized_blocks

__all__ = [
    "TILE_SETS",
    "Framework",
    "RuleFit",
    "SearchResult",
    "TileSet",
    "TransferMatrix",
    "count_assemblies",
    "count_polarized_blocks",
    "descriptor_cost",
    "fit_rule",
    "genetic_algorithm",
    "pipe_tiles",
    "polarized_blocks",
    "sample_uniform",
    "simulated_annealing",
    "structure_descriptors",
    "wave_function_collapse",
]
