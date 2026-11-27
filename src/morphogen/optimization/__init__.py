"""Topology optimization of periodic cells by inverse homogenization.

* :mod:`morphogen.optimization.cell`: homogenized tensors of voxel cells with graded
  moduli and their exact sensitivities (FFT-preconditioned conjugate gradients);
* :mod:`morphogen.optimization.filters`: density filter, Heaviside projection, symmetry;
* :mod:`morphogen.optimization.mma`: method of moving asymptotes (Svanberg);
* :mod:`morphogen.optimization.topology`: objectives (bulk and shear moduli, Poisson's
  ratio, conductivity, target tensor) and methods (OC, MMA, BESO, hybrid cellular
  automaton);
* :mod:`morphogen.optimization.run`: configuration files and outputs
  (``morphogen optimize``);
* :mod:`morphogen.optimization.morphing`: the inverse design of active structures, the
  layout of materials that expand with a stimulus for target shapes at equilibrium or in
  time.
"""

from morphogen.optimization.cell import PeriodicCell
from morphogen.optimization.filters import DensityFilter, Symmetry, heaviside
from morphogen.optimization.mma import MMA
from morphogen.optimization.morphing import (
    MorphingDesign,
    MorphingOptimization,
    MorphingTarget,
    hat_filter,
)
from morphogen.optimization.run import load_optimization_config, run_optimization
from morphogen.optimization.topology import (
    METHODS,
    OBJECTIVES,
    CellDesign,
    CellOptimization,
    count_hinges,
    initial_density,
    tensor_properties,
)

__all__ = [
    "METHODS",
    "MMA",
    "OBJECTIVES",
    "CellDesign",
    "CellOptimization",
    "DensityFilter",
    "MorphingDesign",
    "MorphingOptimization",
    "MorphingTarget",
    "PeriodicCell",
    "Symmetry",
    "count_hinges",
    "hat_filter",
    "heaviside",
    "initial_density",
    "load_optimization_config",
    "run_optimization",
    "tensor_properties",
]
