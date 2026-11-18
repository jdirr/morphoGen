"""Isogeometric analysis: NURBS geometries, their analysis and their shape optimization.

* :mod:`morphogen.iga.basis`: B-spline functions, knot vectors, refinement operators;
* :mod:`morphogen.iga.geometry`: NURBS patches and their assemblies (:class:`Patch`,
  :class:`Multipatch`), their refinements;
* :mod:`morphogen.iga.shapes`: rectangles, rings, plates and periodic cells with holes
  (boxes, parallelograms, the sheet with alternating holes);
* :mod:`morphogen.iga.analysis`: linear elasticity and conduction (:class:`Analysis`);
* :mod:`morphogen.iga.homogenization`: the effective tensors of periodic cells, boxes
  or cells of any lattice (:func:`homogenize`, :class:`Cell`);
* :mod:`morphogen.iga.optimization`: shape optimization with the control points as
  variables, linked by the symmetries of a pattern, and analytical sensitivities
  (:class:`ShapeDesign`, :func:`optimize_cell`, :func:`optimize_structure`).

The tutorial ``docs/tutorial_iga.md`` follows the package from a patch to the shape of
an auxetic sheet.
"""

from morphogen.iga import basis, homogenization, optimization, shapes
from morphogen.iga.analysis import Analysis, conduction_matrix, elastic_matrix
from morphogen.iga.geometry import Multipatch, Patch
from morphogen.iga.homogenization import Cell, CellSolution, homogenize, plane_constants
from morphogen.iga.optimization import (
    ShapeDesign,
    ShapeResult,
    optimize_cell,
    optimize_structure,
)

__all__ = [
    "Analysis",
    "Cell",
    "CellSolution",
    "Multipatch",
    "Patch",
    "ShapeDesign",
    "ShapeResult",
    "basis",
    "conduction_matrix",
    "elastic_matrix",
    "homogenization",
    "homogenize",
    "optimization",
    "optimize_cell",
    "optimize_structure",
    "plane_constants",
    "shapes",
]
