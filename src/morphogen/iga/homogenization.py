"""Periodic homogenization of NURBS cells: effective stiffness and conductivity.

A cell is an assembly whose outer boundary is a rectangle (a box) with matching control
points on opposite sides (:func:`morphogen.iga.shapes.cell_with_hole`): the trace of a
field on a side of a patch depends on the control points of that side only, so that
equal unknowns on matching control points make the field periodic exactly. Its holes
are free of tractions; several materials are the phases of its patches.

The cell need not be a box. With the periods of the medium (``lattice``: one vector per
row; the :attr:`~morphogen.iga.geometry.Multipatch.lattice` of the assembly by default)
it is any set of patches that tiles the plane (the space) by these translations — a
parallelogram (the rhombus of a hexagonal array), or whole grains of the pattern cut
nowhere (:func:`morphogen.iga.shapes.alternating_holes`): the control points of its
boundary that are a period apart are then one, and ``|Y|`` is the area (volume) of the
lattice, ``|det(lattice)|``.

For each unit macroscopic strain ``E_I`` (gradient, for conduction) the fluctuation
``chi_I`` — periodic, fixed at one control point — solves ``K chi_I = -L_I``, with ``L_I =
int B^T D E_I`` the forces of the uniform strain, and the effective tensor is the mean
energy of the pair of total strains::

    C_IJ = (1 / |Y|) int (E_I + eps(chi_I)) : D : (E_J + eps(chi_J)) dY
         = (1 / |Y|) (int D_IJ dY + L_I . chi_J),

``|Y|`` the area (volume) of the box, holes included: the stiffness in Voigt's notation
with engineering shears (``(11, 22, 12)`` in the plane), or the conductivity tensor.

In the plane, ``plane="strain"`` gives the plane block of the stiffness of the cell
extruded along the third axis (what a solver of the solid gives with ``eps_33 = 0``);
``plane="stress"`` that of a thin sheet.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from morphogen.iga.analysis import Analysis
from morphogen.iga.geometry import Multipatch, Patch


def _periods(lattice: np.ndarray) -> np.ndarray:
    """The translations that can bring a side of a cell onto another: the combinations
    of its periods with the coefficients -1, 0 and 1, one of each pair of opposites."""
    dim = len(lattice)
    combos = [c for c in itertools.product((-1, 0, 1), repeat=dim) if any(c)]
    kept = [c for c in combos if next(v for v in c if v) > 0]
    return np.array(kept, dtype=float) @ lattice


def _lattice_classes(model: Multipatch, lattice: np.ndarray, tolerance: float) -> np.ndarray:
    """The classes of the control points of a cell of given periods: those of its
    boundary that are a period apart share one."""
    boundary = model.boundary_nodes()
    points = model.nodes[boundary]
    tol = tolerance * float(np.abs(lattice).max())
    tree = cKDTree(points)
    rows, cols = [], []
    for shift in _periods(lattice):
        distance, match = tree.query(points + shift)
        found = distance <= tol
        rows.append(boundary[found])
        cols.append(boundary[match[found]])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    if not len(rows):
        raise ValueError(
            "not a periodic cell: no control points of its boundary are a period apart"
        )
    if np.abs(model.weights[rows] - model.weights[cols]).max() > 1e-9:
        raise ValueError("not a periodic cell: different weights a period apart")
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(model.n_nodes,) * 2)
    _, labels = connected_components(graph, directed=False)
    return labels


def periodic_classes(model: Multipatch, tolerance: float = 1e-8, lattice=None) -> np.ndarray:
    """The class of each control point of a cell: the control points of opposite sides
    of its box that match (the same position along the side, the same weight) share
    one; an error if a side has control points without a match. With the periods of the
    cell (``lattice``, or those of the assembly), the control points of its boundary
    that are a period apart share one."""
    lattice = model.lattice if lattice is None else np.asarray(lattice, dtype=float)
    if lattice is not None:
        return _lattice_classes(model, lattice, tolerance)
    lower, upper = model.lower, model.upper
    size = float((upper - lower).max())
    tol = tolerance * size
    rows, cols = [], []
    for axis in range(model.dim):
        low = np.flatnonzero(np.abs(model.nodes[:, axis] - lower[axis]) <= tol)
        high = np.flatnonzero(np.abs(model.nodes[:, axis] - upper[axis]) <= tol)
        others = [d for d in range(model.dim) if d != axis]
        if len(low) != len(high) or not len(low):
            raise ValueError(
                f"not a periodic cell: {len(low)} and {len(high)} control points on the "
                f"sides across axis {axis}"
            )
        distance, match = cKDTree(model.nodes[high][:, others]).query(model.nodes[low][:, others])
        if distance.max() > tol or len(np.unique(match)) != len(low):
            raise ValueError(
                f"not a periodic cell: the control points of the sides across axis {axis} "
                "do not match"
            )
        if np.abs(model.weights[low] - model.weights[high[match]]).max() > 1e-9:
            raise ValueError(f"not a periodic cell: different weights across axis {axis}")
        rows.append(low)
        cols.append(high[match])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(model.n_nodes,) * 2)
    _, labels = connected_components(graph, directed=False)
    return labels


@dataclass
class CellSolution:
    """The effective ``tensor`` of a cell, the area (volume) ``measure`` of its box, the
    share ``fraction`` of it that the patches fill, the fluctuations of the unit
    macroscopic strains (``fluctuations``: unknowns x components) and the analysis."""

    tensor: np.ndarray
    measure: float
    fraction: float
    fluctuations: np.ndarray
    analysis: Analysis = field(repr=False)

    def constants(self) -> dict[str, float]:
        """The engineering constants of the tensor (:func:`plane_constants` in the
        plane; the mean conductivity for conduction)."""
        if self.analysis.physics == "conduction":
            return {"conductivity": float(np.trace(self.tensor) / len(self.tensor))}
        if self.tensor.shape == (3, 3):
            return plane_constants(self.tensor)
        from morphogen.optimization.topology import tensor_properties

        return tensor_properties(self.tensor)


def plane_constants(tensor) -> dict[str, float]:
    """The engineering constants of a plane stiffness (3 x 3; Voigt 11, 22, 12 with
    engineering shears): Young's moduli and Poisson's ratios of its plane law, ``young_1 =
    1 / S_11``, ``poisson_12 = -S_12 / S_11`` (the contraction along 2 under a tension
    along 1), the shear modulus ``C_66`` and the bulk modulus of the plane, ``(C_11 +
    C_22 + 2 C_12) / 4``."""
    c = np.asarray(tensor, dtype=float)
    s = np.linalg.inv(c)
    return {
        "young_1": float(1.0 / s[0, 0]),
        "young_2": float(1.0 / s[1, 1]),
        "poisson_12": float(-s[0, 1] / s[0, 0]),
        "poisson_21": float(-s[0, 1] / s[1, 1]),
        "shear_modulus": float(c[2, 2]),
        "bulk_modulus": float((c[0, 0] + c[1, 1] + 2.0 * c[0, 1]) / 4.0),
    }


class Cell:
    """A periodic cell (see the module): the assembly ``model``, the ``materials`` of
    its phases, the physics (``"elastic"``, with ``plane`` for surfaces, or
    ``"conduction"``) and, for a cell that is not a box, the periods ``lattice`` (those
    of the assembly by default). :meth:`solve` gives its effective tensor."""

    def __init__(self, model, materials, *, physics: str = "elastic", plane: str = "strain",
                 extra: int = 0, tolerance: float = 1e-8, lattice=None):  # fmt: skip
        if isinstance(model, Patch):
            model = Multipatch(model)
        self.analysis = Analysis(model, materials, physics=physics, plane=plane, extra=extra)
        self.model = model
        lattice = model.lattice if lattice is None else np.asarray(lattice, dtype=float)
        self.lattice = lattice
        self.classes = periodic_classes(model, tolerance, lattice)
        if lattice is None:
            self.measure = float(np.prod(model.upper - model.lower))
        else:
            self.measure = float(abs(np.linalg.det(lattice)))
            if self.analysis.measure() > (1.0 + 1e-6) * self.measure:
                raise ValueError(
                    "not a periodic cell: its patches fill more than a period of the lattice"
                )
        nd = self.analysis.nd
        # the unknowns of a class of control points are one; the first class is fixed
        kept = np.flatnonzero(self.classes != self.classes[0])
        _, reduced = np.unique(self.classes[kept], return_inverse=True)
        n_free = (reduced.max() + 1) * nd if kept.size else 0
        rows = (nd * kept[:, None] + np.arange(nd)).ravel()
        cols = (nd * reduced[:, None] + np.arange(nd)).ravel()
        self.reduction = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(self.analysis.n_dofs, n_free)
        ).tocsr()

    def solve(self) -> CellSolution:
        """The effective tensor of the cell and the fluctuations of its unit loadings."""
        analysis = self.analysis
        k = analysis.matrix()
        loads, total = analysis.uniform_loads()
        t = self.reduction
        kqq = (t.T @ k @ t).tocsc()
        solve = spla.factorized(kqq)
        chi = np.zeros_like(loads)
        for j in range(loads.shape[1]):
            chi[:, j] = t @ solve(-(t.T @ loads[:, j]))
        tensor = (total + loads.T @ chi) / self.measure
        tensor = 0.5 * (tensor + tensor.T)
        return CellSolution(tensor, self.measure, analysis.measure() / self.measure, chi, analysis)


def homogenize(model, materials, *, physics: str = "elastic", plane: str = "strain",
               extra: int = 0, lattice=None) -> CellSolution:  # fmt: skip
    """The effective tensor of a periodic NURBS cell (:class:`Cell`)."""
    cell = Cell(model, materials, physics=physics, plane=plane, extra=extra, lattice=lattice)
    return cell.solve()


__all__ = ["Cell", "CellSolution", "homogenize", "periodic_classes", "plane_constants"]
