"""The assembly of the finite element paths on several threads: the colours of the
elements (two elements of a colour have no node in common), and the tangents, the nodal
forces and the couplings assembled colour by colour against their sums element by
element, with the same results on one thread and on several."""

import numba
import numpy as np
import pytest

from helpers import box_mesh
from morphogen.homogenization import Material
from morphogen.homogenization.fe import CHUNKS, element_colours
from morphogen.homogenization.nonlinear import NonlinearFEProblem


def two_phases(centres):
    return centres[:, 0] > 0.5


@pytest.mark.parametrize("order", [1, 2])
def test_colours_of_the_elements(order):
    """The elements sorted by colour are a permutation; two elements of a colour have no
    node in common; within a colour the elements keep their order; the colours are at
    most those of the elements around the nodes of an element."""
    mesh = box_mesh((4, 3, 5), upper=(1.0, 1.0, 1.0), order=order, perturb=0.2)
    elements, n_nodes = mesh.elements, mesh.n_nodes
    n_el, nn = elements.shape
    by_colour, colours = element_colours(elements, n_nodes)
    assert np.array_equal(np.sort(by_colour), np.arange(n_el))
    assert colours[0] == 0 and colours[-1] == n_el and np.all(np.diff(colours) > 0)
    around = np.bincount(elements.ravel(), minlength=n_nodes)  # the elements around a node
    assert colours.size - 1 >= around.max()  # (those around a node all differ)
    assert colours.size - 1 <= nn * (around.max() - 1) + 1
    for c in range(colours.size - 1):
        mine = by_colour[colours[c] : colours[c + 1]]
        assert np.all(np.diff(mine) > 0)
        assert np.unique(elements[mine]).size == mine.size * nn
    # the same colours for the same mesh: a greedy colouring in the order of the elements
    again = element_colours(elements, n_nodes)
    assert np.array_equal(again[0], by_colour) and np.array_equal(again[1], colours)


def state(kinematics, integration, cells=3, order=2):
    """A discretization of two phases of rubber on tetrahedra moved at random (10-node
    ones by default), with the stresses and the tangents of its points at a strained
    state."""
    law = {"model": "neo_hookean"}
    materials = {
        0: Material(young=3.0, poisson=0.4, hyperelasticity=law),
        1: Material(young=30.0, poisson=0.3, hyperelasticity=law),
    }
    mesh = box_mesh((cells,) * 3, upper=(1.0, 1.0, 1.0), phase=two_phases, order=order,
                    perturb=0.15)  # fmt: skip
    problem = NonlinearFEProblem(mesh, materials, "periodic", kinematics=kinematics,
                                 integration=integration, tangent="full")  # fmt: skip
    disc = problem._discretization()
    assert disc.bbar == (integration == "bbar")
    nc = 9 if problem.finite else 6
    grads, wts, sigma, tangent, old, new = disc.arrays(
        problem.kind.size, nc, problem.behaviours.width
    )
    rng = np.random.default_rng(1)
    q = 0.02 * rng.standard_normal(disc.n_q)
    e = np.array([0.05, -0.02, 0.01, 0.03, -0.01, 0.02])
    disc.measures(q, e, grads, wts)
    assert disc.integrate(grads, 0.0, 1.0, old, new, sigma, tangent, 0, 1.0) == 0
    return disc, sigma, tangent, wts


def assembled(disc, sigma, tangent, wts):
    """The nodal forces with their sums, the tangent of all the degrees of freedom and
    the couplings with the macroscopic strains (in this order, that of a Newton
    iteration: the tangent of F-bar takes the stresses that the forces have kept)."""
    r_q, total, fscale = disc.forces(sigma.copy(), wts)
    k = disc.full_tangent(tangent, wts)
    columns, kee = disc.macro_columns(tangent, wts)
    return k.data.copy(), r_q.copy(), total.copy(), np.array([fscale]), columns.copy(), kee.copy()


#: Kinematics, integration, cells along each side and order of the elements: curved
#: 10-node tetrahedra (a few elements per colour: one per chunk), and 4-node ones in
#: numbers (colours of more elements than chunks: several per chunk)
CASES = [
    ("small", "full", 3, 2),
    ("finite", "full", 3, 2),
    ("small", "bbar", 3, 2),
    ("finite", "bbar", 3, 2),
    ("small", "full", 10, 1),
    ("finite", "full", 10, 1),
]


@pytest.mark.parametrize("kinematics, integration, cells, order", CASES)
def test_assembly_by_colours(kinematics, integration, cells, order):
    """The tangent, the nodal forces, their sums and the couplings assembled colour by
    colour, the elements of a colour in parallel, are those of the elements taken one by
    one in their order (to round-off: the contributions at a node are added colour by
    colour), and are the same to the bit on one thread and on several."""
    disc, sigma, tangent, wts = state(kinematics, integration, cells, order)
    n_el = disc.mesh.elements.shape[0]
    assert disc.colours.size - 1 < n_el
    assert (np.diff(disc.colours).max() > CHUNKS) == (order == 1)
    by_colours = assembled(disc, sigma, tangent, wts)
    several = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        on_one_thread = assembled(disc, sigma, tangent, wts)
    finally:
        numba.set_num_threads(several)
    for more, one in zip(by_colours, on_one_thread, strict=True):
        assert np.array_equal(more, one)
    # one element per colour: the elements one by one, in their order
    colours = disc.by_colour, disc.colours
    disc.by_colour, disc.colours = np.arange(n_el), np.arange(n_el + 1)
    try:
        one_by_one = assembled(disc, sigma, tangent, wts)
    finally:
        disc.by_colour, disc.colours = colours
    for found, reference in zip(by_colours, one_by_one, strict=True):
        scale = np.abs(reference).max()
        assert scale > 0.0
        assert np.abs(found - reference).max() <= 1e-13 * scale
