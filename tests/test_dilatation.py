"""Selective reduced integration of the 10-node tetrahedra (``integration="bbar"``: the
volume change of each element as its mean, :mod:`morphogen.homogenization.dilatation`):
homogeneous fields, the tangents of B-bar and F-bar against finite differences of the
forces, a thick-walled cylinder under pressure against Lamé's solution near
incompressibility, a nearly incompressible rubber along a path against its material
point, stiff spheres in a nearly incompressible matrix, and the options."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg as spla

import morphogen as mg
from morphogen.config import ConfigError, load_config
from morphogen.homogenization import Material, dilatation, homogenize, simulate_point
from morphogen.homogenization import finite as fin
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.fe import assemble
from morphogen.homogenization.nonlinear import NonlinearFEProblem
from morphogen.image import VoxelImage
from morphogen.mesh.conforming import background_mesh, has_mmg
from morphogen.mesh.tetra import TET_FACE_MIDS, TET_FACES, TetMesh

needs_mmg = pytest.mark.skipif(not has_mmg(), reason="MMG (pymmg) is not installed")
#: The rule of the triangle exact to degree 4: xi, eta, weight
RULE = np.array(
    [
        [0.445948490915965, 0.445948490915965, 0.223381589678011],
        [0.108103018168070, 0.445948490915965, 0.223381589678011],
        [0.445948490915965, 0.108103018168070, 0.223381589678011],
        [0.091576213509771, 0.091576213509771, 0.109951743655322],
        [0.816847572980459, 0.091576213509771, 0.109951743655322],
        [0.091576213509771, 0.816847572980459, 0.109951743655322],
    ]
)


def cube(n: int = 2, amplitude: float = 0.03, seed: int = 0) -> TetMesh:
    """A periodic cube of 10-node tetrahedra in two layers of phases across x, its
    mid-edge nodes moved at random (curved elements), alike on opposite faces."""
    lower, upper = np.zeros(3), np.ones(3)
    nodes, tets, _, _ = background_mesh((n, n, n), lower, upper)
    phase = (nodes[tets[:, :4]].mean(axis=1)[:, 0] > 0.5).astype(int)
    mesh = TetMesh(nodes, tets, phase, lower, upper, periodic=True).quadratic()
    rng = np.random.default_rng(seed)
    x = mesh.nodes.copy()
    mids = np.unique(mesh.elements[:, 4:])
    move = amplitude / n * rng.uniform(-1.0, 1.0, (len(mids), 3))
    move[np.isclose(x[mids], 0.0) | np.isclose(x[mids], 1.0)] = 0.0
    x[mids] += move
    for _ in range(2):
        for axis in range(3):
            pairs = mesh.periodic_pairs(axis)
            others = [d for d in range(3) if d != axis]
            x[np.ix_(pairs[:, 1], others)] = x[np.ix_(pairs[:, 0], others)]
    return mesh._derived(x, mesh.elements)


def isotropic(young: float, poisson: float) -> np.ndarray:
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = young / (2 * (1 + poisson))
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[np.arange(3), np.arange(3)] += 2 * mu
    d[np.arange(3, 6), np.arange(3, 6)] = mu
    return d


# -- names and options ------------------------------------------------------------------------


def test_names_and_options():
    assert dilatation.INTEGRATIONS == ("full", "bbar")
    for name in ("bbar", "B-bar", "reduced", "selective", "f_bar"):
        assert dilatation.integration_of(name) == "bbar"
    assert dilatation.integration_of(None) == dilatation.integration_of("full") == "full"
    with pytest.raises(ValueError, match="integration"):
        dilatation.integration_of("one point")
    rubber, steel = Material(young=3.0, poisson=0.4995), Material(young=200.0, poisson=0.3)
    assert dilatation.nearly_incompressible({0: rubber, 1: steel}, [0, 1]) == [0]
    # a 4-node tetrahedron has one point: its strain is its mean, and nothing changes
    lower, upper = np.zeros(3), np.ones(3)
    nodes, tets, _, _ = background_mesh((2, 2, 2), lower, upper)
    linear = TetMesh(nodes, tets, np.zeros(len(tets), np.int64), lower, upper, periodic=True)
    mat, d = np.zeros(linear.n_elements, np.int64), isotropic(3.0, 0.45)[None]
    assert abs(assemble(linear, 3, mat, d, "bbar") - assemble(linear, 3, mat, d)).max() == 0.0
    # the conduction has no volume change
    quad, k = linear.quadratic(), np.eye(3)[None]
    assert abs(assemble(quad, 1, mat, k, "bbar") - assemble(quad, 1, mat, k)).max() == 0.0
    spec = {
        "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.2},
        "domain": {"size": 4.0, "periodic": True},
        "mesh": {"size": 1.0},
        "homogenization": {
            "method": "fe",
            "materials": {0: {"young": 3.0, "poisson": 0.499}, 1: {"young": 70.0, "poisson": 0.2}},
            "integration": "reduced",
            "nonlinear": {"integration": "full"},
        },
    }
    options = load_config(spec).homogenization.options()
    assert options["integration"] == "bbar" and options["nonlinear"]["integration"] == "full"
    spec["homogenization"]["integration"] = "none"
    with pytest.raises(ConfigError, match="integration"):
        load_config(spec)
    spec["homogenization"].update(method="xfem", integration="bbar", nonlinear={})
    spec["voxels"] = {"shape": 8}
    with pytest.raises(ConfigError, match="integration"):
        load_config(spec)


# -- homogeneous fields -----------------------------------------------------------------------


@pytest.mark.parametrize("bc", ["periodic", "kubc"])
def test_homogeneous_solid_has_its_moduli(bc):
    """Uniform strains are in the space of the elements, and the mean of a uniform
    volume change is that volume change: a homogeneous solid, nearly incompressible, has
    its own stiffness with either integration — to round-off with straight edges, to the
    error of the four-point rule on curved ones (which its bulk modulus amplifies)."""
    solid = Material(young=3.0, poisson=0.499)
    exact = isotropic(3.0, 0.499)
    for amplitude, tolerance in ((0.0, 1e-9), (0.03, 2e-5)):
        mesh = cube(amplitude=amplitude)
        for integration in ("full", "bbar"):
            out = homogenize(mesh, {0: solid, 1: solid}, properties=["stiffness"], bcs=[bc],
                             integration=integration)  # fmt: skip
            tensor = out[f"stiffness.{bc}"]["tensor"]
            np.testing.assert_allclose(tensor, exact, rtol=tolerance, atol=500 * tolerance)


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_rubber_along_a_path_against_its_material_point(kinematics):
    """A nearly incompressible neo-Hookean solid stretched by 20 %: the periodic element
    follows its material point with the mean dilatation as with every point."""
    rubber = Material(young=3.0, poisson=0.499, hyperelasticity={"model": "neo_hookean"})
    path = {"strain": {"11": [0.0, 0.2]}, "times": [0.0, 1.0], "increments": 4}
    point = simulate_point(rubber, path, kinematics=kinematics)
    for integration in ("full", "bbar"):
        out = homogenize(
            cube(amplitude=0.02),
            {0: rubber, 1: rubber},
            properties=[],
            paths={"pull": path},
            nonlinear={"kinematics": kinematics, "integration": integration},
        )["pull.periodic"]
        np.testing.assert_allclose(out.stress[-1], point.stress[-1], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out.strain[-1], point.strain[-1], rtol=1e-5, atol=1e-7)


# -- the tangents -----------------------------------------------------------------------------


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_tangent_is_the_derivative_of_the_forces(kinematics):
    """The tangent of B-bar (F-bar) on curved elements of two phases: symmetric, and the
    derivative of the nodal forces and of the macroscopic stress with respect to the
    fluctuation and to the macroscopic strain (central differences)."""
    law = {"model": "neo_hookean"}
    materials = {
        0: Material(young=3.0, poisson=0.45, hyperelasticity=law),
        1: Material(young=30.0, poisson=0.3, hyperelasticity=law),
    }
    problem = NonlinearFEProblem(cube(), materials, "periodic", kinematics=kinematics,
                                 integration="bbar", tangent="full")  # fmt: skip
    disc = problem._discretization()
    assert disc.bbar and problem.integration == "bbar"
    nc = 9 if problem.finite else 6
    grads, wts, sigma, tangent, old, new = disc.arrays(
        problem.kind.size, nc, problem.behaviours.width
    )
    rng = np.random.default_rng(1)
    q0 = 0.02 * rng.standard_normal(disc.n_q)
    e0 = np.array([0.05, -0.02, 0.01, 0.03, -0.01, 0.02])

    def forces(q, e):
        disc.measures(q, e, grads, wts)
        assert disc.integrate(grads, 0.0, 1.0, old, new, sigma, tangent, 0, 1.0) == 0
        r_q, total, _ = disc.forces(sigma, wts)
        return r_q.copy(), (fin.macro_part(total) if problem.finite else total.copy())

    forces(q0, e0)
    k = disc.full_tangent(tangent, wts)
    columns, kee = disc.macro_columns(tangent, wts)
    kqq = (disc.t_tr @ k @ disc.t_mat).toarray()
    assert abs(kqq - kqq.T).max() < 1e-12 * abs(kqq).max()
    v, de, h = rng.standard_normal(disc.n_q), rng.standard_normal(6), 1e-6
    plus, minus = forces(q0 + h * v, e0)[0], forces(q0 - h * v, e0)[0]
    np.testing.assert_allclose(kqq @ v, (plus - minus) / (2 * h), rtol=1e-6, atol=1e-8)
    (rq_p, s_p), (rq_m, s_m) = forces(q0, e0 + h * de), forces(q0, e0 - h * de)
    np.testing.assert_allclose(
        disc.t_tr @ columns @ de, (rq_p - rq_m) / (2 * h), rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(kee @ de, (s_p - s_m) / (2 * h), rtol=1e-6, atol=1e-8)


def test_linear_matrix_is_the_tangent_of_the_paths():
    """The stiffness of the elastic problems with ``bbar`` is the tangent of the loading
    paths at the natural state (the same element, written twice)."""
    mesh = cube()
    materials = {0: Material(young=3.0, poisson=0.45), 1: Material(young=30.0, poisson=0.3)}
    problem = NonlinearFEProblem(mesh, materials, "periodic", integration="bbar")
    disc = problem._discretization()
    grads, wts, _, tangent, _, _ = disc.arrays(problem.kind.size, 6, 1)
    disc.measures(np.zeros(disc.n_q), np.zeros(6), grads, wts)
    tangent[:] = problem.table.matrices[np.repeat(problem.mat, 4)]
    k_paths = disc.full_tangent(tangent, wts)
    k_linear = assemble(mesh, 3, problem.mat, problem.table.matrices, "bbar")
    assert abs(k_paths - k_linear).max() < 1e-10 * abs(k_linear).max()
    k_full = assemble(mesh, 3, problem.mat, problem.table.matrices)
    assert abs(k_full - k_linear).max() > 1e-3 * abs(k_linear).max()


# -- a thick-walled cylinder under pressure ---------------------------------------------------


def _shape6(xi: float, eta: float):
    zeta = 1.0 - xi - eta
    shape = np.array([zeta * (2 * zeta - 1), xi * (2 * xi - 1), eta * (2 * eta - 1),
                      4 * zeta * xi, 4 * xi * eta, 4 * eta * zeta])  # fmt: skip
    dxi = np.array([1 - 4 * zeta, 4 * xi - 1, 0.0, 4 * (zeta - xi), 4 * eta, -4 * eta])
    deta = np.array([1 - 4 * zeta, 0.0, 4 * eta - 1, -4 * xi, 4 * xi, 4 * (zeta - eta)])
    return shape, dxi, deta


def _cylinder(n_r: int, n_t: int, inner: float, outer: float, height: float = 0.25):
    """A quarter of a thick-walled cylinder: the 10-node tetrahedra of a box of
    ``n_r x n_t x 1`` voxels mapped to polar coordinates (curved edges), and the
    coordinates of its nodes in the box."""
    spacing = (1.0 / n_r, 1.0 / n_t, height)
    box = voxel_tet_mesh(
        VoxelImage(np.zeros((n_r, n_t, 1), np.uint8), spacing=spacing, periodic=False)
    ).quadratic()
    xi, eta, z = box.nodes.T
    r, theta = inner + xi * (outer - inner), eta * np.pi / 2
    nodes = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    upper = np.array([outer, outer, height])
    return TetMesh(nodes, box.elements, box.phase, np.zeros(3), upper), box.nodes


def _pressure(mesh: TetMesh, box: np.ndarray, pressure: float) -> np.ndarray:
    """Nodal forces of a pressure on the inner face of the cylinder (``xi = 0``)."""
    f = np.zeros((mesh.n_nodes, 3))
    corners = mesh.elements[:, TET_FACES]
    for e, k in zip(*np.nonzero(np.all(box[corners][..., 0] < 1e-9, axis=2)), strict=True):
        ids = np.concatenate([mesh.elements[e, TET_FACES[k]], mesh.elements[e, TET_FACE_MIDS[k]]])
        ordered = list(ids[:3])
        for i, j in ((0, 1), (1, 2), (2, 0)):  # the mid-edge node of each edge of the face
            middle = 0.5 * (box[ids[i]] + box[ids[j]])
            ordered.append(next(m for m in ids[3:] if np.allclose(box[m], middle)))
        x = mesh.nodes[ordered]
        for xi, eta, weight in RULE:
            shape, dxi, deta = _shape6(xi, eta)
            area = 0.5 * weight * np.linalg.norm(np.cross(dxi @ x, deta @ x))
            point = shape @ x
            radial = np.array([point[0], point[1], 0.0]) / np.hypot(point[0], point[1])
            f[ordered] += area * pressure * shape[:, None] * radial[None, :]
    return f.ravel()


def test_thick_cylinder_under_pressure_near_incompressibility():
    """Lamé's cylinder in plane strain, Poisson's ratio 0.4999: ``u_r = (1 + nu) p a^2 ((1
    - 2 nu) r + b^2 / r) / (E (b^2 - a^2))``. With two elements across the wall the mean
    dilatation gives the displacement of the bore within 0.2 % and every node within
    0.7 %, twice as close as the integration at every point; both converge."""
    inner, outer, young, poisson, pressure = 1.0, 2.0, 1.0, 0.4999, 1.0
    worst = {}
    for n_r, n_t in ((2, 4), (4, 8)):
        mesh, box = _cylinder(n_r, n_t, inner, outer)
        x = mesh.nodes
        fixed = np.concatenate(
            [
                3 * np.flatnonzero(np.abs(x[:, 0]) < 1e-9),  # symmetry: u_x = 0 on x = 0
                3 * np.flatnonzero(np.abs(x[:, 1]) < 1e-9) + 1,
                3 * np.arange(mesh.n_nodes) + 2,  # plane strain
            ]
        )
        free = np.setdiff1d(np.arange(3 * mesh.n_nodes), fixed)
        f = _pressure(mesh, box, pressure)
        assert f.reshape(-1, 3)[:, 0].sum() == pytest.approx(pressure * inner * 0.25, rel=1e-3)
        r = np.hypot(x[:, 0], x[:, 1])
        exact = (1 + poisson) * pressure * inner**2 / (young * (outer**2 - inner**2))
        exact = exact * ((1 - 2 * poisson) * r + outer**2 / r)
        mat, d = np.zeros(mesh.n_elements, np.int64), isotropic(young, poisson)[None]
        for integration in ("full", "bbar"):
            k = assemble(mesh, 3, mat, d, integration).tocsr()
            u = np.zeros(3 * mesh.n_nodes)
            u[free] = spla.spsolve(k[free][:, free].tocsc(), f[free])
            u = u.reshape(-1, 3)
            radial = (u[:, 0] * x[:, 0] + u[:, 1] * x[:, 1]) / r
            worst[integration, n_r] = np.abs(radial / exact - 1.0).max()
            if integration == "bbar" and n_r == 2:
                bore = np.abs(r - inner) < 1e-9
                assert radial[bore].mean() == pytest.approx(exact[bore].mean(), rel=2e-3)
    assert worst["bbar", 2] < 0.007 and worst["bbar", 4] < 0.0015
    assert worst["bbar", 2] < 0.6 * worst["full", 2] and worst["bbar", 4] < 0.6 * worst["full", 4]
    assert worst["full", 4] < 0.3 * worst["full", 2]


# -- stiff spheres in a nearly incompressible matrix --------------------------------------------


@pytest.mark.slow
@needs_mmg
def test_stiff_spheres_in_a_nearly_incompressible_matrix():
    """A cubic array of stiff spheres in a matrix of Poisson's ratio 0.4999: with the
    mean dilatation the two shear moduli of the cell are those of a finer mesh within
    0.2 % with elements of the size of the radius (4,800 nodes on the background grid;
    and of a mesh of 40,000 nodes within 0.1 %); integrated at every point the elements
    lock — 2.4 and 6 % too stiff there, and still 0.5 and 1 % with 40,000 nodes."""
    from morphogen.models.union import PrimitiveUnion
    from morphogen.primitives import Spheres

    domain = mg.Domain.cube(4.0, periodic=True)
    micro = PrimitiveUnion(domain, Spheres(np.array([[1.9, 2.15, 2.05]]), np.array([1.2])), None)
    materials = {0: Material(young=3.0, poisson=0.4999), 1: Material(young=70.0, poisson=0.22)}
    shear = {}
    for size, integrations in ((1.2, ("full", "bbar")), (0.8, ("bbar",))):
        mesh = mg.mesh.conforming_mesh(micro, size=size, order=2, method="grid")
        for integration in integrations:
            c = homogenize(mesh, materials, properties=["stiffness"], integration=integration)[
                "stiffness.periodic"
            ]["tensor"]
            shear[integration, size] = np.array([c[3, 3], 0.5 * (c[0, 0] - c[0, 1])])
    fine = shear["bbar", 0.8]
    np.testing.assert_allclose(shear["bbar", 1.2], fine, rtol=2e-3)
    np.testing.assert_allclose(fine, [1.2502, 1.3738], rtol=2e-3)  # those of 40,000 nodes
    assert shear["full", 1.2][0] > 1.02 * fine[0] and shear["full", 1.2][1] > 1.05 * fine[1]
