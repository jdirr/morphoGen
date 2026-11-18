"""Isogeometric analysis (:mod:`morphogen.iga`): B-spline bases and their refinement
operators, exact arcs and NURBS assemblies, the patch test, Lamé's thick-walled cylinder
and Kirsch's plate with a hole against their exact solutions, the effective tensors of
periodic cells (a homogeneous cell, a laminate, a cell with a hole against the finite
elements of the solid), in the plane and in space, for elasticity and conduction; cells
of any lattice (other cuts of the same medium, the hexagonal and square arrays of holes
against the series of Perrins et al., the sheet with alternating holes)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import morphogen as mg
from morphogen import iga
from morphogen.homogenization import Material, homogenize
from morphogen.iga import basis as bs
from morphogen.iga import shapes
from morphogen.mesh.conforming import has_mmg

needs_mmg = pytest.mark.skipif(not has_mmg(), reason="MMG (pymmg) is not installed")
STEEL = Material(young=1000.0, poisson=0.3)


# -- bases ----------------------------------------------------------------------------------


def test_basis_functions_and_their_derivatives():
    """The functions of a knot vector with repeated knots: non-negative, summing to one,
    their derivatives summing to zero and equal to central differences; the functions
    of a span against the recurrence of Cox and de Boor written in full."""
    knots = np.array([0, 0, 0, 0, 0.2, 0.5, 0.5, 0.9, 1, 1, 1, 1.0])
    degree = 3
    assert bs.n_functions(knots, degree) == 8
    u = np.linspace(0.0, 1.0, 41)
    span, values = bs.basis(knots, degree, u, 2)
    assert values.shape == (3, 41, 4) and values[0].min() >= 0.0
    np.testing.assert_allclose(values[0].sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(values[1].sum(axis=1), 0.0, atol=1e-12)
    assert span[0] == 3 and span[-1] == 7 and np.all(knots[span] <= u)

    # the definition, by recursion on the degree
    def cox(i, p, x):
        if p == 0:
            last = knots[i + 1] == knots[-1] and x == knots[-1]
            return float(knots[i] <= x < knots[i + 1] or (last and knots[i] < knots[i + 1]))
        out = 0.0
        if knots[i + p] > knots[i]:
            out += (x - knots[i]) / (knots[i + p] - knots[i]) * cox(i, p - 1, x)
        if knots[i + p + 1] > knots[i + 1]:
            out += (knots[i + p + 1] - x) / (knots[i + p + 1] - knots[i + 1]) * cox(i + 1, p - 1, x)
        return out

    dense = bs.collocation(knots, degree, u)
    reference = np.array([[cox(i, degree, x) for i in range(8)] for x in u])
    np.testing.assert_allclose(dense, reference, atol=1e-14)
    inside = np.array([0.07, 0.31, 0.44, 0.63, 0.77, 0.95])
    h = 1e-6
    for order in (1, 2):
        plus = bs.collocation(knots, degree, inside + h, order - 1)
        minus = bs.collocation(knots, degree, inside - h, order - 1)
        np.testing.assert_allclose(
            bs.collocation(knots, degree, inside, order), (plus - minus) / (2 * h), atol=1e-6
        )
    # the Greville abscissae: the coefficients of the identity
    np.testing.assert_allclose(dense @ bs.greville(knots, degree), u, atol=1e-14)
    with pytest.raises(ValueError, match="open"):
        bs.check_knots([0, 0, 0.3, 0.6, 1, 1], 2)
    with pytest.raises(ValueError, match="outside"):
        bs.basis(knots, degree, [1.2])


def test_refinement_operators():
    """Knots inserted and the degree raised: the refined space contains the functions of
    the coarse one, with the coefficients of the embedding (rows that sum to one)."""
    knots = np.array([0, 0, 0, 0.3, 0.3, 0.8, 1, 1, 1.0])
    u = np.linspace(0.0, 1.0, 57)
    for subdivisions, elevate in ((2, 0), (1, 1), (3, 2)):
        new = bs.refined_knots(knots, 2, subdivisions, elevate)
        matrix = bs.embedding(knots, 2, new, 2 + elevate)
        assert matrix.shape == (bs.n_functions(new, 2 + elevate), 6)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(
            bs.collocation(new, 2 + elevate, u) @ matrix, bs.collocation(knots, 2, u), atol=1e-12
        )
    # the continuity of the knots is kept: a double knot of degree 2 stays a kink
    assert np.count_nonzero(bs.refined_knots(knots, 2, 1, 1) == 0.3) == 3
    with pytest.raises(ValueError, match="does not contain"):
        bs.embedding(knots, 2, bs.open_knots(5, 2), 2)


# -- geometry -------------------------------------------------------------------------------


def test_exact_arcs_and_refined_patches():
    """A ring is a ring: its radii to round-off, at any refinement; the area of its
    quarter; the control points of the refined patch from those of the coarse one."""
    ring = shapes.annulus(1.0, 2.0)
    assert ring.degrees == (2, 1) and ring.rational and ring.n_elements == 1
    radii = np.linalg.norm(ring.sample(9), axis=-1)
    np.testing.assert_allclose(radii[:, 0], 1.0, atol=1e-14)
    np.testing.assert_allclose(radii[:, -1], 2.0, atol=1e-14)
    fine = ring.refined(4, 1)
    assert fine.degrees == (3, 2) and fine.shape == (7, 6) and fine.n_elements == 16
    params = np.random.default_rng(0).random((40, 2))
    np.testing.assert_allclose(fine.at(params), ring.at(params), atol=1e-13)
    _, jac = ring.jacobian(params)
    assert np.linalg.det(jac).min() > 0.0
    model = iga.Multipatch(ring)
    refined, matrix = model.refinement((3, 2), (1, 2))
    np.testing.assert_allclose(matrix @ model.nodes, refined.nodes, atol=1e-13)
    np.testing.assert_allclose(np.asarray(matrix.sum(axis=1)).ravel(), 1.0, atol=1e-13)
    moved = model.nodes * [1.3, 0.8]  # an ellipse: a linear map of the control points
    again = refined.with_nodes(matrix @ moved)
    np.testing.assert_allclose(
        again.patches[0].at(params), ring.at(params) * [1.3, 0.8], atol=1e-13
    )
    area = iga.Analysis(refined, {0: STEEL}).measure()
    assert area == pytest.approx(0.75 * math.pi, rel=1e-7)
    with pytest.raises(ValueError, match="control points"):
        iga.Patch((1, 1), (bs.open_knots(2, 1),) * 2, np.zeros((3, 2, 2)))


def test_assemblies_share_their_sides():
    """The two patches of a quarter plate share the control points of their common
    side; a cell with a hole has matching control points on its opposite sides."""
    plate = shapes.plate_with_hole(1.0, 4.0)
    assert len(plate.patches) == 2 and plate.n_nodes == 2 * 6 - 2
    shared = np.intersect1d(plate.ids[0], plate.ids[1])
    np.testing.assert_allclose(plate.nodes[shared], [[4.0, 4.0], [math.sqrt(0.5)] * 2], atol=1e-12)
    fine = plate.refined(3, 1)
    assert fine.n_nodes == 2 * (5 * 6) - 5 and len(fine.free_sides()) == 6
    cell = shapes.cell_with_hole((2.0, 1.0), (0.5, 0.2, 0.3), elements=2)
    assert len(cell.patches) == 4 and len(cell.free_sides()) == 8
    classes = iga.homogenization.periodic_classes(cell.refined(2, (0, 1)))
    assert len(np.unique(classes)) < cell.refined(2, (0, 1)).n_nodes
    hole = np.concatenate([p.side(1, 1).sample(20) for p in cell.patches]) - [1.0, 0.5]
    c, s = math.cos(0.3), math.sin(0.3)
    local = hole @ np.array([[c, -s], [s, c]])
    np.testing.assert_allclose((local[:, 0] / 0.5) ** 2 + (local[:, 1] / 0.2) ** 2, 1.0, atol=1e-12)
    with pytest.raises(ValueError, match="periodic"):
        iga.homogenization.periodic_classes(plate)


# -- analysis -------------------------------------------------------------------------------


@pytest.mark.parametrize("plane", ["strain", "stress"])
def test_patch_test(plane):
    """A uniform strain prescribed on the boundary of curved rational patches is found
    inside, to the error of the Gauss rule on rational functions (which more points
    reduce)."""
    gradient = np.array([[0.01, 0.004], [-0.002, 0.007]])
    worst = []
    for extra in (0, 3):
        model = shapes.plate_with_hole(1.0, 4.0).refined(2, (0, 1))
        analysis = iga.Analysis(model, {0: STEEL}, plane=plane, extra=extra)
        exact = model.nodes @ gradient.T
        boundary = model.boundary_nodes()
        u = analysis.solve(
            np.zeros(analysis.n_dofs), analysis.dofs(boundary), exact[boundary].ravel()
        )
        worst.append(np.abs(u.reshape(-1, 2) - exact).max())
        strain = np.array([0.01, 0.007, 0.002])
        for values in analysis.fields(u, 3):
            assert np.abs(values["strain"] - strain).max() < 1e-6
    assert worst[0] < 1e-7 and worst[1] < 1e-11


def test_patch_test_of_a_solid():
    """The same for a solid: a quarter tube (a ring swept along its axis)."""
    model = shapes.extruded(shapes.annulus(1.0, 2.0), 0.5).refined(2, (0, 1, 1))
    assert model.ndim == 3 and model.patches[0].degrees == (2, 2, 2)
    analysis = iga.Analysis(model, {0: STEEL}, extra=2)
    assert analysis.measure() == pytest.approx(0.75 * math.pi * 0.5, rel=1e-9)
    gradient = np.array([[0.01, 0.004, 0.0], [-0.002, 0.007, 0.003], [0.001, 0.0, -0.005]])
    exact = model.nodes @ gradient.T
    boundary = model.boundary_nodes()
    u = analysis.solve(np.zeros(analysis.n_dofs), analysis.dofs(boundary), exact[boundary].ravel())
    np.testing.assert_allclose(u.reshape(-1, 3), exact, atol=1e-9)


def _lame(inner, outer, pressure, young, poisson):
    def displacement(points):
        r = np.linalg.norm(points, axis=1)
        radial = (1 + poisson) * pressure * inner**2 / (young * (outer**2 - inner**2))
        radial = radial * ((1 - 2 * poisson) * r + outer**2 / r)
        return radial[:, None] * points / r[:, None]

    def stress(points):
        r = np.linalg.norm(points, axis=1)
        c, s = points[:, 0] / r, points[:, 1] / r
        factor = pressure * inner**2 / (outer**2 - inner**2)
        srr, stt = factor * (1 - outer**2 / r**2), factor * (1 + outer**2 / r**2)
        return np.stack(
            [srr * c**2 + stt * s**2, srr * s**2 + stt * c**2, (srr - stt) * s * c], axis=1
        )

    return displacement, stress


@pytest.mark.parametrize(("elevate", "rates"), [((0, 1), (3.0, 2.0)), ((1, 2), (4.0, 3.0))])
def test_thick_walled_cylinder_against_lame(elevate, rates):
    """A quarter of a thick-walled cylinder under internal pressure in plane strain, its
    geometry exact: the displacement converges at the order ``p + 1`` and the stresses at
    the order ``p`` (degrees 2 and 3)."""
    displacement, stress = _lame(1.0, 2.0, 1.0, 1000.0, 0.3)
    ring = shapes.annulus(1.0, 2.0)
    errors = []
    for subdivisions in (2, 4, 8):
        model = iga.Multipatch(ring.refined(subdivisions, elevate))
        analysis = iga.Analysis(model, {0: STEEL})
        x = model.nodes
        fixed = np.concatenate(
            [
                analysis.dofs(np.flatnonzero(np.abs(x[:, 1]) < 1e-12), [1]),
                analysis.dofs(np.flatnonzero(np.abs(x[:, 0]) < 1e-12), [0]),
            ]
        )
        forces = analysis.side_load(0, 1, 0, lambda points, normals: -1.0 * normals)
        u = analysis.solve(forces, fixed)
        errors.append(analysis.error(u, displacement, stress))
    for name, rate in zip(("l2", "energy"), rates, strict=True):
        observed = math.log2(errors[1][name] / errors[2][name])
        assert observed == pytest.approx(rate, abs=0.35), (name, observed)
    assert errors[2]["l2"] < (2e-4 if elevate == (0, 1) else 2e-6)


def test_cylinder_of_a_solid():
    """The same cylinder as a solid between two planes (plane strain): the radial
    displacement of the bore within 0.1 % with 4 elements across the wall."""
    displacement, _ = _lame(1.0, 2.0, 1.0, 1000.0, 0.3)
    model = shapes.extruded(shapes.annulus(1.0, 2.0), 0.4).refined((4, 4, 1), (0, 1, 0))
    analysis = iga.Analysis(model, {0: STEEL})
    x = model.nodes
    fixed = np.concatenate(
        [
            analysis.dofs(np.flatnonzero(np.abs(x[:, 1]) < 1e-12), [1]),
            analysis.dofs(np.flatnonzero(np.abs(x[:, 0]) < 1e-12), [0]),
            analysis.dofs(np.arange(model.n_nodes), [2]),
        ]
    )
    forces = analysis.side_load(0, 1, 0, lambda points, normals: -1.0 * normals)
    u = analysis.solve(forces, fixed).reshape(-1, 3)
    bore = analysis.at(u, 0, [[0.3, 0.0, 0.5], [0.8, 0.0, 0.2]])
    np.testing.assert_allclose(bore["u"][:, :2], displacement(bore["x"][:, :2]), rtol=1e-3)


def _kirsch(points, radius=1.0, tension=1.0):
    """The stresses around a hole in an infinite plate pulled along x (Kirsch 1898)."""
    r = np.linalg.norm(points, axis=1)
    t = np.arctan2(points[:, 1], points[:, 0])
    q2, q4 = (radius / r) ** 2, (radius / r) ** 4
    sxx = 1 - q2 * (1.5 * np.cos(2 * t) + np.cos(4 * t)) + 1.5 * q4 * np.cos(4 * t)
    syy = -q2 * (0.5 * np.cos(2 * t) - np.cos(4 * t)) - 1.5 * q4 * np.cos(4 * t)
    sxy = -q2 * (0.5 * np.sin(2 * t) + np.sin(4 * t)) + 1.5 * q4 * np.sin(4 * t)
    return tension * np.stack([sxx, syy, sxy], axis=1)


def test_plate_with_a_hole_against_kirsch():
    """A quarter of an infinite plate with a hole, the exact tractions on its outer
    sides: the stresses converge to Kirsch's solution in the energy norm (order 3 for
    cubic functions), and the stress at the top of the hole to three times the tension."""

    def traction(points, normals):
        s = _kirsch(points)
        return np.stack(
            [s[:, 0] * normals[:, 0] + s[:, 2] * normals[:, 1],
             s[:, 2] * normals[:, 0] + s[:, 1] * normals[:, 1]], axis=1,
        )  # fmt: skip

    base = shapes.plate_with_hole(1.0, 4.0)
    errors, peaks = [], []
    for subdivisions in (4, 8, 16):
        model = base.refined(subdivisions, (1, 2))
        analysis = iga.Analysis(model, {0: STEEL})
        x = model.nodes
        fixed = np.concatenate(
            [
                analysis.dofs(np.flatnonzero(np.abs(x[:, 1]) < 1e-12), [1]),
                analysis.dofs(np.flatnonzero(np.abs(x[:, 0]) < 1e-12), [0]),
            ]
        )
        forces = analysis.side_load(0, 1, 0, traction) + analysis.side_load(1, 1, 0, traction)
        u = analysis.solve(forces, fixed)
        errors.append(analysis.error(u, None, _kirsch)["energy"])
        top = analysis.at(u, 1, [[1.0, 1.0]])  # the end of the arc of the second patch
        np.testing.assert_allclose(top["x"][0], [0.0, 1.0], atol=1e-12)
        peaks.append(top["stress"][0, 0])
    assert errors[2] < 1e-3 < errors[0]
    assert math.log2(errors[1] / errors[2]) > 2.3
    assert peaks[2] == pytest.approx(3.0, abs=0.01) and abs(peaks[1] - 3.0) < 0.05


# -- periodic cells -------------------------------------------------------------------------


def test_homogeneous_cell_and_laminate():
    """A homogeneous cell has the stiffness of its material; a laminate of two layers
    its exact effective stiffness (plane strain): the harmonic mean across the layers,
    the arithmetic one along them with the correction of the lateral contraction."""
    d0 = iga.elastic_matrix(STEEL, 2, "strain")
    out = iga.homogenize(shapes.rectangle((0, 0), (2, 1), degree=2), {0: STEEL})
    np.testing.assert_allclose(out.tensor, d0, rtol=1e-12, atol=1e-9)
    assert out.fraction == pytest.approx(1.0) and out.measure == pytest.approx(2.0)
    soft = Material(young=10.0, poisson=0.4)
    d1 = iga.elastic_matrix(soft, 2, "strain")
    layers = iga.Multipatch(
        [
            shapes.rectangle((0, 0), (0.3, 1), degree=2, phase=0),
            shapes.rectangle((0.3, 0), (1, 1), degree=2, phase=1),
        ]
    )
    out = iga.homogenize(layers, {0: STEEL, 1: soft})

    def mean(a, b):
        return 0.3 * a + 0.7 * b

    c11 = 1.0 / mean(1 / d0[0, 0], 1 / d1[0, 0])
    c12 = mean(d0[0, 1] / d0[0, 0], d1[0, 1] / d1[0, 0]) * c11
    c22 = mean(d0[1, 1] - d0[0, 1] ** 2 / d0[0, 0], d1[1, 1] - d1[0, 1] ** 2 / d1[0, 0])
    c22 += c12**2 / c11
    c66 = 1.0 / mean(1 / d0[2, 2], 1 / d1[2, 2])
    exact = np.array([[c11, c12, 0.0], [c12, c22, 0.0], [0.0, 0.0, c66]])
    np.testing.assert_allclose(out.tensor, exact, rtol=1e-10, atol=1e-9)
    # conduction: the harmonic and arithmetic means
    conductors = {0: Material(young=1.0, poisson=0.3, conductivity=5.0),
                  1: Material(young=1.0, poisson=0.3, conductivity=1.0)}  # fmt: skip
    k = iga.homogenize(layers, conductors, physics="conduction").tensor
    np.testing.assert_allclose(k, np.diag([1.0 / mean(1 / 5.0, 1.0), mean(5.0, 1.0)]), atol=1e-12)
    # plane stress of an isotropic sheet: E / (1 - nu^2) (1, nu; nu, 1)
    sheet = iga.homogenize(shapes.rectangle(degree=1), {0: STEEL}, plane="stress").tensor
    assert sheet[0, 0] == pytest.approx(1000.0 / 0.91) and sheet[0, 1] == pytest.approx(
        300.0 / 0.91
    )
    constants = iga.plane_constants(sheet)
    assert constants["young_1"] == pytest.approx(1000.0) and constants[
        "poisson_12"
    ] == pytest.approx(0.3)


def test_cell_with_a_hole_converges():
    """A square cell with a circular hole of a fifth of its area: the fraction of
    material exact, a tensor with the symmetries of the square, which the refinements
    bring down to the same values (cubic functions: within 0.02 % with 880 unknowns)."""
    exact = 1.0 - math.pi * 0.25**2
    tensors = {}
    for key, (subdivisions, elevate) in {"coarse": (2, (0, 1)), "fine": (8, (0, 1)),
                                         "cubic": (8, (1, 2))}.items():  # fmt: skip
        out = iga.homogenize(shapes.cell_with_hole((1.0, 1.0), 0.25).refined(subdivisions, elevate),
                             {0: STEEL})  # fmt: skip
        c = out.tensor
        assert out.fraction == pytest.approx(exact, rel=2e-7)
        assert c[0, 0] == pytest.approx(c[1, 1], rel=1e-10) and abs(c[0, 2]) < 1e-9 * c[0, 0]
        tensors[key] = c
    assert tensors["coarse"][0, 0] > tensors["fine"][0, 0] > tensors["cubic"][0, 0]
    np.testing.assert_allclose(tensors["fine"], tensors["cubic"], rtol=5e-4, atol=1e-6)
    np.testing.assert_allclose(
        [tensors["cubic"][0, 0], tensors["cubic"][0, 1], tensors["cubic"][2, 2]],
        [769.89, 266.23, 194.68], rtol=2e-4,
    )  # fmt: skip


def _array_of_holes(fraction: float, array: str) -> float:
    """The conductivity of a sheet with a regular array of circular holes, over that of
    the sheet: the series of Perrins, McKenzie & McPhedran (1979) for the square and the
    hexagonal arrays (``fraction``: the share of the holes)."""
    f = fraction
    if array == "square":
        return 1 - 2 * f / (1 + f - 0.305827 * f**4 / (1 - 1.402958 * f**8) - 0.013362 * f**8)
    return 1 - 2 * f / (1 + f - 0.075422 * f**6 / (1 - 1.060283 * f**12) - 0.000076 * f**12)


def test_cells_of_any_lattice():
    """A cell is any set of patches that tiles the plane by its periods. The square cell
    with an inclined elliptic hole, as a box and with its periods given; with one of its
    four patches moved by a period (no longer a box); with two of them moved and another
    pair of periods of the same lattice: the same effective stiffness to round-off."""
    cell = shapes.cell_with_hole((1.0, 1.0), (0.3, 0.2, 0.4), elements=2)
    assert cell.lattice is None
    box = iga.homogenize(cell.refined(3, (0, 1)), {0: STEEL})
    same = iga.homogenize(cell.refined(3, (0, 1)), {0: STEEL}, lattice=np.eye(2))
    np.testing.assert_allclose(same.tensor, box.tensor, rtol=1e-12)
    assert abs(box.tensor[0, 2]) > 1.0  # the inclined hole: no symmetry to hide a mistake
    one, two, three, four = cell.patches
    moved = iga.Multipatch([one.with_points(one.points - [1.0, 0.0]), two, three, four],
                           lattice=np.eye(2))  # fmt: skip
    assert moved.lower[0] < -0.3 and np.array_equal(moved.refined(2).lattice, np.eye(2))
    out = iga.homogenize(moved.refined(3, (0, 1)), {0: STEEL})
    np.testing.assert_allclose(out.tensor, box.tensor, rtol=1e-11, atol=1e-9)
    assert out.fraction == pytest.approx(box.fraction, rel=1e-12) and out.measure == 1.0
    oblique = iga.Multipatch(
        [one.with_points(one.points - [1.0, 0.0]), two.with_points(two.points - [1.0, 1.0]),
         three, four], lattice=[[1.0, 0.0], [1.0, 1.0]],
    )  # fmt: skip
    out = iga.homogenize(oblique.refined(3, (0, 1)), {0: STEEL})
    np.testing.assert_allclose(out.tensor, box.tensor, rtol=1e-11, atol=1e-9)
    # a longer period along x: strips with gaps between them, which carry nothing across
    strips = iga.homogenize(cell.refined(3, (0, 1)), {0: STEEL}, lattice=[[1.3, 0.0], [0.0, 1.0]])
    assert strips.fraction == pytest.approx(box.fraction / 1.3)
    assert strips.tensor[1, 1] > 300.0 and abs(strips.tensor[0, 0]) < 1e-9 * strips.tensor[1, 1]
    # periods that are not those of the patches
    with pytest.raises(ValueError, match="not a periodic cell"):
        iga.homogenize(cell, {0: STEEL}, lattice=[[1.3, 0.0], [0.0, 1.2]])
    with pytest.raises(ValueError, match="more than a period"):
        iga.homogenize(shapes.repeated(cell, (2, 1)), {0: STEEL}, lattice=np.eye(2))
    with pytest.raises(ValueError, match="lattice"):
        iga.Multipatch(cell.patches, lattice=[[1.0, 0.0], [2.0, 0.0]])
    # a cell and its neighbour: twice the period, the same medium
    double = shapes.repeated(cell, (2, 1))
    assert len(double.patches) == 8 and np.allclose(double.lattice, [[2.0, 0.0], [0.0, 1.0]])
    out = iga.homogenize(double.refined(3, (0, 1)), {0: STEEL})
    np.testing.assert_allclose(out.tensor, box.tensor, rtol=1e-10, atol=1e-8)


def test_arrays_of_holes_against_the_series_of_perrins():
    """The conductivity of a sheet with circular holes on a square lattice (a box) and on
    a hexagonal one (a rhombus of 60 degrees: an oblique lattice) against the series of
    Perrins, McKenzie & McPhedran (1979), within 0.002 % — the estimate of Maxwell is 0.1
    to 0.5 % away. The hexagonal array is isotropic: equal conductivities along the two
    axes, and an elastic stiffness with ``C_66 = (C_11 - C_12) / 2``."""
    sheet = Material(young=1000.0, poisson=0.3, conductivity=2.0)
    rhombus = [[1.0, 0.0], [0.5, math.sqrt(0.75)]]
    for radius in (0.3, 0.4):
        cell = shapes.cell_with_hole(rhombus, radius)
        assert np.allclose(cell.lattice, rhombus)
        holes = math.pi * radius**2 / math.sqrt(0.75)
        out = iga.homogenize(cell.refined(8, (1, 2)), {0: sheet}, physics="conduction")
        assert out.measure == pytest.approx(math.sqrt(0.75))
        assert out.fraction == pytest.approx(1.0 - holes, rel=1e-8)
        series = 2.0 * _array_of_holes(holes, "hexagonal")
        np.testing.assert_allclose(np.diag(out.tensor), series, rtol=2e-5)
        assert abs(out.tensor[0, 1]) < 1e-5 * series
        assert abs(2.0 * (1 - 2 * holes / (1 + holes)) / series - 1.0) > 6e-5  # Maxwell
    out = iga.homogenize(shapes.cell_with_hole((1.0, 1.0), 0.3).refined(8, (1, 2)), {0: sheet},
                         physics="conduction")  # fmt: skip
    series = 2.0 * _array_of_holes(math.pi * 0.09, "square")
    np.testing.assert_allclose(np.diag(out.tensor), series, rtol=2e-6)
    # elasticity: the isotropy of the hexagonal array comes with the refinement
    gaps = []
    for subdivisions in (2, 4, 8):
        c = iga.homogenize(shapes.cell_with_hole(rhombus, 0.35).refined(subdivisions, (1, 2)),
                           {0: STEEL}).tensor  # fmt: skip
        gaps.append(abs(c[2, 2] - 0.5 * (c[0, 0] - c[0, 1])) / c[2, 2])
    assert gaps[0] > 5 * gaps[1] > 25 * gaps[2] and gaps[2] < 1e-4
    assert c[0, 0] == pytest.approx(c[1, 1], rel=2e-5) and abs(c[0, 2]) < 2e-5 * c[0, 0]
    with pytest.raises(ValueError, match="anticlockwise"):
        shapes.cell_with_hole([[0.5, math.sqrt(0.75)], [1.0, 0.0]], 0.3)


def test_sheet_with_alternating_holes():
    """Elongated holes on a square lattice, turned in turn along its two axes
    (:func:`~morphogen.iga.shapes.alternating_holes`): two whole grains as the cell of
    an oblique lattice. The fraction of material is exact and the stiffness has the
    symmetries of the square; with circular holes the sheet is the square array, and
    its stiffness that of the box with one hole within 0.0001 %; thin ligaments make it
    auxetic (the rotating squares)."""
    cell = shapes.alternating_holes(1.0, (0.6, 0.1))
    assert len(cell.patches) == 16 and np.allclose(cell.lattice, [[1.0, 1.0], [1.0, -1.0]])
    low, high = cell.jacobians()
    assert 0.0 < low < high
    flank, tip = cell.ids[0][:, 0], cell.ids[1][:, 0]  # the arcs of the hole at the origin
    np.testing.assert_allclose(cell.nodes[flank[0]], [0.0, 0.1], atol=1e-15)
    np.testing.assert_allclose(cell.nodes[tip[-1]], [0.6, 0.0], atol=1e-15)
    assert flank[-1] == tip[0]
    out = iga.homogenize(cell.refined(4, (1, 2)), {0: STEEL}, plane="stress")
    assert out.measure == 2.0 and out.fraction == pytest.approx(1 - math.pi * 0.06, rel=1e-8)
    c = out.tensor
    assert c[0, 0] == pytest.approx(c[1, 1], rel=1e-10)
    assert abs(c[0, 2]) < 1e-9 * c[0, 0] and abs(c[1, 2]) < 1e-9 * c[0, 0]
    constants = out.constants()
    assert constants["poisson_12"] == pytest.approx(-0.530, abs=1e-3)
    assert constants["young_1"] == pytest.approx(157.4, rel=2e-3)
    # the ratio falls as the ligaments thin: 0.5, 0.45, 0.22 and 0.1 of the spacing
    ratios = []
    for hole in ((0.25, 0.25), (0.45, 0.1), (0.7, 0.08), (0.85, 0.05)):
        sheet = iga.homogenize(shapes.alternating_holes(1.0, hole).refined(4, (1, 2)),
                               {0: STEEL}, plane="stress")  # fmt: skip
        ratios.append(sheet.constants()["poisson_12"])
    np.testing.assert_allclose(ratios, [0.266, -0.074, -0.764, -0.950], atol=2e-3)
    # circles: the square array of the box with one hole
    for plane in ("strain", "stress"):
        box = iga.homogenize(shapes.cell_with_hole((1.0, 1.0), 0.25).refined(8, (1, 2)),
                             {0: STEEL}, plane=plane).tensor  # fmt: skip
        grains = iga.homogenize(shapes.alternating_holes(1.0, 0.25).refined(6, (1, 2)),
                                {0: STEEL}, plane=plane).tensor  # fmt: skip
        np.testing.assert_allclose(grains, box, rtol=2e-6, atol=2e-6 * box.max())
    # the polygons of the patches hold their area; their image, the fractions of the phases
    area = sum(
        0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
        for q in cell.polygons(64)
    )
    assert area == pytest.approx(2.0 * (1 - math.pi * 0.06), rel=1e-4)
    image = cell.voxelize(200, lower=(0.0, 0.0), upper=(2.0, 2.0), depth=2)
    assert image.data.shape == (200, 200, 2) and image.metadata["hole"] == 1
    assert (image.data == 1).mean() == pytest.approx(math.pi * 0.06, abs=2e-3)
    layers = iga.Multipatch(
        [shapes.rectangle((0, 0), (0.3, 1), phase=0), shapes.rectangle((0.3, 0), (1, 1), phase=2)]
    )
    stripes = layers.voxelize((10, 4)).data[:, :, 0]
    assert np.all(stripes[:3] == 0) and np.all(stripes[3:] == 2)
    with pytest.raises(ValueError, match="voxelize"):
        cell.voxelize((8, 8), lower=(0.0, 0.0), upper=(0.0, 1.0))
    with pytest.raises(ValueError, match="holes apart"):
        shapes.alternating_holes(1.0, (0.8, 0.25))
    with pytest.raises(ValueError, match="half-axes"):
        shapes.alternating_holes(1.0, (0.6, 0.1, 0.3))


@pytest.mark.slow
def test_sheet_with_alternating_holes_against_the_fft_solver():
    """The auxetic sheet (holes of half-axes 0.6 and 0.1) by another method altogether:
    the square of four holes as an image of 256 x 256 voxels with voids
    (:meth:`~morphogen.iga.geometry.Multipatch.voxelize`), solved by the FFT solver. The
    plane components of its stiffness are those of the isogeometric analysis in plane
    strain within 1.5 % — ``C_12`` negative, -119 for a sheet of modulus 1000."""
    cell = shapes.alternating_holes(1.0, (0.6, 0.1))
    plane = iga.homogenize(cell.refined(8, (1, 2)), {0: STEEL}).tensor
    assert plane[0, 1] == pytest.approx(-119.1, abs=0.2)
    n = 256
    voxels = cell.voxelize((n, n), lower=(0.0, 0.0), upper=(2.0, 2.0))  # a period: four holes
    assert voxels.periodic and voxels.data.shape == (n, n, 1) and voxels.spacing[0] == 2.0 / n
    # the image is that of the ellipses, but for a few pixels on their outlines
    centres = (np.arange(n) + 0.5) * 2.0 / n
    x, y = np.meshgrid(centres, centres, indexing="ij")
    image = np.zeros((n, n), dtype=np.uint8)
    for i in range(3):
        for j in range(3):
            along, across = (0.6, 0.1) if (i + j) % 2 == 0 else (0.1, 0.6)
            image[((x - i) / along) ** 2 + ((y - j) / across) ** 2 < 1.0] = 1
    assert (image != voxels.data[:, :, 0]).mean() < 1e-3
    solid = homogenize(voxels, {0: STEEL, 1: Material(void=True)}, properties=["stiffness"])[
        "stiffness.periodic"
    ]["tensor"]
    np.testing.assert_allclose(solid[np.ix_([0, 1, 5], [0, 1, 5])], plane, rtol=0.015, atol=0.5)


@pytest.mark.slow
@needs_mmg
def test_cell_with_a_hole_against_the_finite_elements_of_the_solid():
    """The same cell as a solid with a cylindrical hole, meshed with curved 10-node
    tetrahedra (53,000 nodes): the plane components of its stiffness are those of the
    isogeometric analysis in plane strain (880 unknowns) within 0.1 %."""
    from morphogen.models.union import PrimitiveUnion
    from morphogen.primitives import Segments

    plane = iga.homogenize(shapes.cell_with_hole((1.0, 1.0), 0.25).refined(8, (1, 2)),
                           {0: STEEL}).tensor  # fmt: skip
    domain = mg.Domain.cube(1.0, periodic=True)
    hole = Segments(np.array([[0.5, 0.5, -1.0]]), np.array([[0.5, 0.5, 2.0]]), np.array([0.25]))
    mesh = mg.mesh.conforming_mesh(PrimitiveUnion(domain, None, hole), size=0.08, order=2,
                                   phases="matrix")  # fmt: skip
    solid = homogenize(mesh, {0: STEEL, 1: Material(void=True)}, properties=["stiffness"])[
        "stiffness.periodic"
    ]["tensor"]
    np.testing.assert_allclose(solid[np.ix_([0, 1, 5], [0, 1, 5])], plane, rtol=1e-3, atol=0.2)
