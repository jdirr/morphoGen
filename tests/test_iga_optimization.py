"""Isogeometric shape optimization (:mod:`morphogen.iga.optimization`): the analytical
shape derivatives against central differences (the effective tensor and the fraction of
a periodic cell, in elasticity and conduction; the compliance and the area of a
structure), the hole of largest bulk modulus of a periodic cell against the bound of
Hashin and Shtrikman (the structure of Vigdergauz), the hole of a plate under biaxial
tension against the ellipse of the infinite plate, the two optimizers, the variables
that the symmetries of a pattern link, and the design of an auxetic sheet (a hole
found again from its effective constants; the lowest Poisson's ratio at a given
stiffness)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from morphogen import iga
from morphogen.homogenization import Material
from morphogen.iga import optimization as opt
from morphogen.iga import shapes

STEEL = Material(young=1000.0, poisson=0.3)


def _hole(model) -> np.ndarray:
    """The control points of the holes of an assembly of patches from sides to arcs."""
    return np.unique(np.concatenate([model.side_nodes(k, 1, 1) for k in range(len(model.patches))]))


def _free_design():
    """A cell with a hole whose control points move freely (two variables each)."""
    cell = shapes.cell_with_hole((1.0, 1.0), 0.3, elements=2)
    design = opt.ShapeDesign(cell, subdivisions=3, elevate=(0, 1))
    for node in _hole(cell):
        for direction in ((1.0, 0.0), (0.0, 1.0)):
            design.variable([node], [direction], -0.1, 0.1)
    return design


def test_design_and_its_geometries():
    """The control points of the analysis are linear in the variables; the refined
    geometry is that of the design, moved."""
    design = _free_design()
    assert design.model.n_nodes == 24 and design.n == 24 and design.fine.n_nodes == 140
    lower, upper = design.bounds()
    assert np.all(lower == -0.1) and np.all(upper == 0.1)
    rng = np.random.default_rng(0)
    x, y = 0.03 * rng.standard_normal(design.n), 0.03 * rng.standard_normal(design.n)
    a, b = design.analysis_model(x).nodes, design.analysis_model(y).nodes
    np.testing.assert_allclose(
        design.analysis_model(0.5 * (x + y)).nodes, 0.5 * (a + b), atol=1e-14
    )
    params = rng.random((30, 2))
    coarse, fine = design.design_model(x), design.analysis_model(x)
    for p, q in zip(coarse.patches, fine.patches, strict=True):
        np.testing.assert_allclose(q.at(params), p.at(params), atol=1e-13)
    # radial variables with the symmetries of the square: the hole stays symmetric
    cell = shapes.cell_with_hole((1.0, 1.0), 0.3, elements=3)
    radial = opt.ShapeDesign(cell, subdivisions=2, elevate=(0, 1))
    variables = radial.radial(_hole(cell), (0.5, 0.5), -0.3, 0.3, symmetry="d4")
    assert len(variables) == 3 and len(radial.radial(_hole(cell), (0.5, 0.5), symmetry="d2")) == 5
    moved = radial.control(np.array([0.1, -0.05, 0.2, 0, 0, 0, 0, 0.0]))[_hole(cell)] - 0.5
    for image in (moved * [-1, 1], moved * [1, -1], moved[:, ::-1]):
        assert all(np.abs(moved - point).sum(axis=1).min() < 1e-12 for point in image)
    with pytest.raises(ValueError, match="variables"):
        radial.control(np.zeros(3))
    with pytest.raises(ValueError, match="symmetry"):
        radial.radial(_hole(cell), (0.5, 0.5), symmetry="c3")


@pytest.mark.parametrize("physics", ["elastic", "conduction"])
def test_derivatives_of_a_cell_against_differences(physics):
    """The derivatives of the effective tensor and of the fraction of material of a cell
    with respect to the positions of the control points of its hole: the analytical ones
    (the energy-momentum tensor of the fluctuations) against central differences."""
    design = _free_design()
    materials = {0: Material(young=1000.0, poisson=0.3, conductivity=2.0)}
    rng = np.random.default_rng(1)
    x = 0.02 * rng.standard_normal(design.n)
    response = opt.cell_response(design, x, materials, physics=physics)
    n = 3 if physics == "elastic" else 2
    assert response.tensor.shape == (n, n) and response.d_tensor.shape == (n, n, design.n)
    assert 0.6 < response.fraction < 0.8
    h = 1e-6
    for _ in range(3):
        v = rng.standard_normal(design.n)
        plus = opt.cell_response(design, x + h * v, materials, physics=physics)
        minus = opt.cell_response(design, x - h * v, materials, physics=physics)
        expected = (plus.tensor - minus.tensor) / (2 * h)
        np.testing.assert_allclose(response.d_tensor @ v, expected, rtol=1e-6,
                                   atol=1e-7 * np.abs(expected).max())  # fmt: skip
        assert response.d_fraction @ v == pytest.approx(
            (plus.fraction - minus.fraction) / (2 * h), rel=1e-6
        )


def _plate(half: float, elements: int, subdivisions, elevate=(1, 2)):
    """A quarter plate with a hole pulled twice as much along x as along y, its hole
    scaled about its centre control point by control point."""
    plate = shapes.plate_with_hole(1.0, half)
    if elements > 1:
        plate = plate.refined((elements, 1))
    design = opt.ShapeDesign(plate, subdivisions=subdivisions, elevate=elevate)
    design.radial(_hole(plate), (0.0, 0.0), -0.5, 0.6)

    def loads(analysis):
        return analysis.side_load(0, 1, 0, [2.0, 0.0]) + analysis.side_load(1, 1, 0, [0.0, 1.0])

    def supports(analysis):
        x = analysis.model.nodes
        return np.concatenate(
            [
                analysis.dofs(np.flatnonzero(np.abs(x[:, 1]) < 1e-12), [1]),
                analysis.dofs(np.flatnonzero(np.abs(x[:, 0]) < 1e-12), [0]),
            ]
        )

    return design, loads, supports


def test_derivatives_of_a_structure_against_differences():
    """The compliance of a plate with a hole under tractions on its outer sides and the
    area of its material: their derivatives against central differences."""
    design, loads, supports = _plate(4.0, 1, 4, (0, 1))
    assert design.n == 5
    rng = np.random.default_rng(2)
    x = 0.05 * rng.standard_normal(design.n)
    response = opt.structure_response(design, x, {0: STEEL}, loads, supports)
    assert response.compliance == pytest.approx(response.u @ loads(response.analysis))
    h = 1e-6
    v = rng.standard_normal(design.n)
    plus = opt.structure_response(design, x + h * v, {0: STEEL}, loads, supports)
    minus = opt.structure_response(design, x - h * v, {0: STEEL}, loads, supports)
    assert response.d_compliance @ v == pytest.approx(
        (plus.compliance - minus.compliance) / (2 * h), rel=1e-6
    )
    assert response.d_measure @ v == pytest.approx(
        (plus.measure - minus.measure) / (2 * h), rel=1e-6
    )


@pytest.mark.parametrize("method", ["slsqp", "mma"])
def test_hole_of_largest_bulk_modulus_reaches_the_bound(method):
    """A square cell with a hole of 40 % of its area: the hole of largest bulk modulus
    is that of Vigdergauz (1994), whose bulk modulus is the upper bound of Hashin and
    Shtrikman, ``K G (1 - f) / (G + f K)`` in the plane. From a circle (1.4 % below the
    bound), three radial variables with the symmetries of the square reach it within
    0.05 %, the fraction of material kept."""
    d = iga.elastic_matrix(STEEL, 2, "strain")
    bulk, shear, porosity = 0.5 * (d[0, 0] + d[0, 1]), d[2, 2], 0.4
    bound = bulk * shear * (1 - porosity) / (shear + porosity * bulk)
    cell = shapes.cell_with_hole((1.0, 1.0), math.sqrt(porosity / math.pi), elements=4)
    design = opt.ShapeDesign(cell, subdivisions=4, elevate=(1, 2))
    design.radial(_hole(cell), (0.5, 0.5), -0.3, 0.3, symmetry="d4")
    start = opt.cell_response(design, np.zeros(design.n), {0: STEEL})
    circle = iga.plane_constants(start.tensor)["bulk_modulus"]
    assert start.fraction == pytest.approx(0.6, rel=1e-6) and 0.984 < circle / bound < 0.988
    seen = []
    result = opt.optimize_cell(
        design,
        {0: STEEL},
        lambda c, f: -iga.plane_constants(c)["bulk_modulus"] / bulk,
        [lambda c, f: f / 0.6 - 1.0],
        method=method,
        maxiter=60,
        tolerance=1e-10 if method == "slsqp" else 1e-5,
        callback=lambda k, x, f, g: seen.append(f),
    )
    best = iga.plane_constants(result.tensor)["bulk_modulus"]
    assert result.fraction == pytest.approx(0.6, abs=2e-5) and abs(result.constraints[0]) < 1e-4
    assert best / bound == pytest.approx(1.0, abs=5e-4) and best > 1.01 * circle
    assert len(seen) == result.iterations + 1 and seen[-1] < seen[0]
    assert result.tensor[0, 0] == pytest.approx(result.tensor[1, 1], rel=1e-9)
    assert result.design.n_nodes == cell.n_nodes and result.analysis.n_nodes == design.fine.n_nodes
    if method == "slsqp":
        assert result.converged and result.iterations < 20


@pytest.mark.slow
def test_hole_of_a_plate_under_biaxial_tension_is_an_ellipse():
    """A hole of given area in a large plate pulled twice as much along x as along y:
    the hole of smallest compliance is the ellipse whose half-axes are in the ratio of
    the tensions (Cherepanov 1974; Banichuk 1983). In a quarter plate twelve radii wide,
    seven radial variables: half-axes within 1.5 % of sqrt(2) and 1 / sqrt(2), the hole
    within 1.5 % of an ellipse, its area kept."""
    design, loads, supports = _plate(12.0, 2, (6, 12))
    assert design.n == 7
    result = opt.optimize_structure(design, {0: STEEL}, loads, supports, maxiter=200,
                                    tolerance=1e-12)  # fmt: skip
    assert result.converged and abs(result.constraints[0]) < 1e-8
    assert result.history[0]["objective"] == 1.0 and result.history[-1]["objective"] < 0.999
    curve = np.concatenate([p.side(1, 1).sample(40) for p in result.design.patches])
    a, b = curve[:, 0].max(), curve[:, 1].max()
    assert a == pytest.approx(math.sqrt(2.0), rel=0.015) and b == pytest.approx(
        math.sqrt(0.5), rel=0.015
    )
    assert a / b == pytest.approx(2.0, rel=0.02)
    assert np.abs(np.hypot(curve[:, 0] / a, curve[:, 1] / b) - 1.0).max() < 0.015


def _sheet(a: float = 0.6, b: float = 0.1, subdivisions=3, elevate=(0, 1)):
    """The sheet with alternating holes and its two variables: the half-axes of its
    holes, all the holes together (the middles of the ligaments follow them)."""
    cell = shapes.alternating_holes(1.0, (a, b))
    design = opt.ShapeDesign(cell, subdivisions=subdivisions, elevate=elevate)
    flank, tip = cell.ids[0], cell.ids[1]  # the patches of the hole at the origin
    hole = np.unique(np.concatenate([flank[:, 0], tip[:, 0]]))
    symmetry = [opt.mirror(0), opt.mirror(1), opt.rotation(90.0, (0.5, 0.5))]
    x = cell.nodes
    design.variable([*hole, *tip[1:, 1]], [*(x[hole] * [1 / a, 0.0]), [0.5, 0.0], [0.5, 0.0]],
                    -0.2, 0.25, "a", symmetry=symmetry)  # fmt: skip
    design.variable([*hole, *flank[:2, 1]], [*(x[hole] * [0.0, 1 / b]), [0.0, 0.5], [0.0, 0.5]],
                    0.03 - b, 0.1, "b", symmetry=symmetry)  # fmt: skip
    return cell, design


def test_variables_linked_by_the_symmetries_of_a_pattern():
    """The mirrors and the quarter-turn of the sheet with alternating holes link the
    half-axes of all its holes to two variables: the sheet they make is that built with
    these half-axes, its stiffness keeps the symmetries of the square, and the
    derivatives are those of central differences."""
    matrix, shift = opt.rotation(90.0, (0.5, 0.5))
    np.testing.assert_array_equal(matrix, [[0.0, -1.0], [1.0, 0.0]])
    np.testing.assert_allclose(matrix @ [1.0, 0.0] + shift, [0.5 + 0.5, 0.5 + 0.5])
    np.testing.assert_allclose(opt.rotation(30.0)[0] @ [1.0, 0.0], [math.sqrt(0.75), 0.5])
    matrix, shift = opt.mirror(1, at=0.25)
    np.testing.assert_allclose(matrix @ [0.3, 0.1] + shift, [0.3, 0.4])
    cell, design = _sheet()
    assert design.n == 2 and design.names == ["a", "b"]
    assert [len(nodes) for nodes, _ in design.moves] == [53, 53]
    # every tip moves outwards by the first variable, along the axis of its hole
    tip = cell.ids[1][-1, 0]
    nodes, directions = design.moves[0]
    moves = dict(zip(nodes.tolist(), directions, strict=True))
    for point, direction in (((0.6, 0.0), (1.0, 0.0)), ((1.0, 0.6), (0.0, 1.0)),
                             ((0.4, 1.0), (-1.0, 0.0)), ((2.0, 0.4), (0.0, -1.0))):  # fmt: skip
        node = int(np.argmin(np.abs(cell.nodes - point).sum(axis=1)))
        np.testing.assert_allclose(cell.nodes[node], point, atol=1e-12)
        np.testing.assert_allclose(moves[node], direction, atol=1e-12)
    assert np.array_equal(moves[int(tip)], [1.0, 0.0])
    x = np.array([0.1, -0.03])
    response = opt.cell_response(design, x, {0: STEEL}, plane="stress")
    assert response.fraction == pytest.approx(1.0 - math.pi * 0.7 * 0.07, rel=1e-7)
    built = iga.homogenize(shapes.alternating_holes(1.0, (0.7, 0.07)).refined(3, (0, 1)),
                           {0: STEEL}, plane="stress")  # fmt: skip
    np.testing.assert_allclose(response.tensor, built.tensor, rtol=2e-3, atol=0.1)
    c = response.tensor
    assert c[0, 0] == pytest.approx(c[1, 1], rel=1e-9) and abs(c[0, 2]) < 1e-9 * c[0, 0]
    h = 1e-6
    for v in (np.array([1.0, 0.0]), np.array([0.3, -1.0])):
        plus = opt.cell_response(design, x + h * v, {0: STEEL}, plane="stress")
        minus = opt.cell_response(design, x - h * v, {0: STEEL}, plane="stress")
        expected = (plus.tensor - minus.tensor) / (2 * h)
        np.testing.assert_allclose(response.d_tensor @ v, expected, rtol=1e-6,
                                   atol=1e-7 * np.abs(expected).max())  # fmt: skip
    # a hole that reaches its neighbour folds the patches of the ligament
    with pytest.raises(opt.FoldedDesign, match="folds"):
        design.analysis_model(np.array([0.25, 0.1]))
    # the periods alone: a control point of the boundary of the cell and its image
    other = opt.ShapeDesign(cell, subdivisions=1, elevate=0)
    middle = cell.ids[1][-1, 1]  # the middle of the ligament of the lower side
    other.variable([middle], [1.0, 0.0], symmetry=[])
    nodes, directions = other.moves[0]
    assert len(nodes) == 2 and middle in nodes
    np.testing.assert_allclose(np.abs(np.diff(cell.nodes[nodes], axis=0)), [[1.0, 1.0]])
    np.testing.assert_allclose(directions, [[1.0, 0.0], [1.0, 0.0]])
    # operations that the pattern does not have, or that keep nothing of a move
    with pytest.raises(ValueError, match="no control point at an image"):
        other.variable([tip], [1.0, 0.0], symmetry=[opt.rotation(90.0, (0.3, 0.3))])
    with pytest.raises(ValueError, match="leaves nothing"):
        other.variable([tip], [0.0, 1.0], symmetry=[opt.mirror(1)])
    with pytest.raises(ValueError, match="finite group"):
        other.variable([tip], [1.0, 0.0], symmetry=[opt.rotation(1.0, (0.3, 0.3))])
    with pytest.raises(ValueError, match="orthogonal"):
        other.variable([tip], [1.0, 0.0], symmetry=[(2.0 * np.eye(2), np.zeros(2))])


def test_hole_found_again_from_its_effective_constants():
    """Inverse design with a known answer: Young's modulus and Poisson's ratio of the
    sheet whose holes have the half-axes 0.7 and 0.07 are the goal; from the holes 0.6
    and 0.1 the optimizer finds these half-axes again (to 1e-6), and the goal exactly."""
    _, design = _sheet()
    answer = np.array([0.1, -0.03])
    goal = iga.plane_constants(opt.cell_response(design, answer, {0: STEEL}, plane="stress").tensor)
    assert goal["poisson_12"] == pytest.approx(-0.751, abs=2e-3)

    def distance(c, f):
        k = iga.plane_constants(c)
        return (k["young_1"] / goal["young_1"] - 1.0) ** 2 + (
            k["poisson_12"] - goal["poisson_12"]
        ) ** 2

    result = opt.optimize_cell(design, {0: STEEL}, distance, plane="stress", maxiter=60,
                               tolerance=1e-14)  # fmt: skip
    assert result.converged and result.iterations < 40
    assert result.history[0]["objective"] > 1.0 and result.objective < 1e-12
    np.testing.assert_allclose(result.x, answer, atol=1e-6)
    found = iga.plane_constants(result.tensor)
    assert found["young_1"] == pytest.approx(goal["young_1"], rel=1e-6)
    assert found["poisson_12"] == pytest.approx(goal["poisson_12"], abs=1e-6)
    assert len(result.constraints) == 0


def test_lowest_poisson_ratio_at_a_given_stiffness():
    """The sheet with alternating holes of lowest Poisson's ratio that keeps a twentieth
    of Young's modulus of the plain sheet, with slits at least 0.06 wide: the slits take
    their least width and the stiffness its bound; the ratio falls from -0.53 to -0.81,
    which a finer analysis of the design confirms. A linear constraint on the variables
    keeps the ligaments instead: the holes then lengthen until it holds as an equality."""

    def ratio(c, f):
        return iga.plane_constants(c)["poisson_12"]

    cell, design = _sheet(subdivisions=4, elevate=(1, 2))
    start = opt.cell_response(design, np.zeros(2), {0: STEEL}, plane="stress")
    assert ratio(start.tensor, 0.0) == pytest.approx(-0.530, abs=2e-3)
    result = opt.optimize_cell(
        design,
        {0: STEEL},
        ratio,
        [lambda c, f: 1.0 - iga.plane_constants(c)["young_1"] / 50.0],
        linear=([[1.0, 1.0]], [0.25]),  # ligaments of 0.05 at least: a + b <= 0.95
        plane="stress",
        maxiter=60,
        tolerance=1e-10,
    )
    assert result.converged and result.iterations < 30
    assert result.objective == pytest.approx(-0.814, abs=2e-3)
    assert result.x[1] == pytest.approx(-0.07, abs=1e-9)  # the least width of the slits
    np.testing.assert_allclose(result.constraints, [0.0, result.x.sum() - 0.25], atol=1e-7)
    assert result.constraints[1] < -0.1 and len(result.history) == result.iterations + 1
    assert iga.plane_constants(result.tensor)["young_1"] == pytest.approx(50.0, rel=1e-6)
    low, high = result.design.jacobians()
    assert 0.0 < low < high and np.allclose(result.design.lattice, cell.lattice)
    fine = iga.homogenize(result.design.refined(8, (1, 2)), {0: STEEL}, plane="stress").constants()
    assert fine["poisson_12"] == pytest.approx(result.objective, abs=3e-3)
    assert fine["young_1"] == pytest.approx(50.0, rel=0.02)
    # the ligaments kept at 0.2 of the spacing, and nothing else
    kept = opt.optimize_cell(design, {0: STEEL}, ratio, linear=([[1.0, 1.0]], [0.1]),
                             plane="stress", maxiter=60, tolerance=1e-10)  # fmt: skip
    assert kept.converged and kept.x.sum() == pytest.approx(0.1, abs=1e-8)
    assert abs(kept.constraints[0]) < 1e-8 and kept.objective == pytest.approx(-0.843, abs=3e-3)
    np.testing.assert_allclose(kept.x, [0.17, -0.07], atol=1e-7)
    with pytest.raises(ValueError, match="linear"):
        opt.optimize_cell(design, {0: STEEL}, ratio, linear=([[1.0, 1.0, 0.0]], [0.1]))


def test_folded_designs_and_methods():
    """A step that folds a patch is turned down by the line search; a design that folds
    at its start is an error, as an unknown method."""
    cell = shapes.cell_with_hole((1.0, 1.0), 0.3)
    design = opt.ShapeDesign(cell, subdivisions=2, elevate=(0, 1))
    design.radial(_hole(cell), (0.5, 0.5), -2.5, 2.5, symmetry="d4")
    with pytest.raises(opt.FoldedDesign, match="folds"):
        opt.optimize_cell(design, {0: STEEL}, lambda c, f: c[0, 0], x0=np.full(design.n, -2.2))
    result = opt.optimize_cell(design, {0: STEEL}, lambda c, f: -c[0, 0] / 1000.0, maxiter=30)
    assert result.fraction > 0.95 and result.tensor[0, 0] > 1200.0  # the hole closes
    # the history holds the points that the iterations kept, none of the trials turned down
    values = [entry["objective"] for entry in result.history]
    assert max(values) < 0.0 and values[-1] == min(values) == pytest.approx(result.objective)
    with pytest.raises(ValueError, match="method"):
        opt.optimize(lambda x: (0.0, x, [], []), [0.0], [-1.0], [1.0], method="newton")
