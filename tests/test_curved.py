"""Quadratic tetrahedra with curved edges: the isoparametric 10-node elements of the
finite elements (their volumes, their matrices against the formulas of the reference
element, the linear fields they hold), and the meshes whose nodes lie on the interfaces
of the phases (:mod:`morphogen.mesh.curved`): the volume of a sphere, the conductivity of
a simple cubic array of spheres against the series of Sangani & Acrivos (1983), coated
spheres, the options of the mesher and of the configurations."""

from __future__ import annotations

import numpy as np
import pytest

import morphogen as mg
from morphogen.config import ConfigError, load_config
from morphogen.homogenization import Material, homogenize, simulate_point
from morphogen.homogenization.fe import GAUSS, assemble
from morphogen.mesh.conforming import background_mesh, has_mmg
from morphogen.mesh.curved import _interfaces, curved_mesh
from morphogen.mesh.tetra import TET_EDGES, TetMesh, tet_volumes
from morphogen.models import model_from_config
from morphogen.models.nested import CoatedStructure
from morphogen.models.union import PrimitiveUnion
from morphogen.primitives import Spheres

needs_mmg = pytest.mark.skipif(not has_mmg(), reason="MMG (pymmg) is not installed")


def one_element() -> TetMesh:
    """A tetrahedron of general shape with ten nodes."""
    nodes = np.array([[0.1, 0.0, 0.2], [1.3, 0.2, 0.0], [0.3, 1.1, 0.1], [0.2, 0.4, 0.9]])
    mesh = TetMesh(nodes, np.array([[0, 1, 2, 3]]), np.array([0]), np.zeros(3), np.full(3, 1.5))
    return mesh.quadratic()


def bent(mesh: TetMesh, moves) -> TetMesh:
    """The mesh with its nodes moved by ``moves`` (nodes x 3)."""
    return mesh._derived(mesh.nodes + moves, mesh.elements)


def structured(n: int = 4, amplitude: float = 0.03, seed: int = 0) -> TetMesh:
    """A cube of six 10-node tetrahedra per cell, in two layers of phases across x, whose
    mid-edge nodes are moved at random by ``amplitude`` times the cell, within the faces
    of the box for those that lie on them, alike on opposite faces."""
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
    for _ in range(2):  # the nodes of the edges of the box have several images
        for axis in range(3):
            pairs = mesh.periodic_pairs(axis)
            others = [d for d in range(3) if d != axis]
            x[np.ix_(pairs[:, 1], others)] = x[np.ix_(pairs[:, 0], others)]
    return mesh._derived(x, mesh.elements)


def sphere(radius: float, size: float = 4.0):
    """One sphere in a periodic cube: a simple cubic array."""
    centre = np.array([1.9, 2.15, 2.05])
    domain = mg.Domain.cube(size, periodic=True)
    return PrimitiveUnion(domain, Spheres(centre[None, :], np.array([radius])), None), centre


class Sampled:
    """A structure that gives its implicit functions at the points of grids only."""

    def __init__(self, micro):
        self._micro = micro

    def __getattr__(self, name):
        if name == "implicit_at":
            raise AttributeError(name)
        return getattr(self._micro, name)


def coated_sphere(size: float = 3.4):
    """A sphere of radius 1 with a coating of 0.3 in a periodic cube: three phases whose
    interfaces are two spheres, apart from each other and from their images."""
    domain = mg.Domain.cube(size, periodic=True)
    inner = PrimitiveUnion(domain, Spheres(np.array([[1.6, 1.75, 1.7]]), np.array([1.0])), None)
    micro = CoatedStructure(domain, inner, 0.3, "outside", 2, 0)
    fractions = {1: 4.0 / 3.0 * np.pi / size**3, 2: 4.0 / 3.0 * np.pi * (1.3**3 - 1.0) / size**3}
    return micro, fractions


# -- the elements ---------------------------------------------------------------------------


def test_straight_elements():
    mesh = one_element()
    assert not mesh.curved and mesh.min_jacobian() == 1.0 and mesh.straight() is mesh
    assert np.all(mesh.deviations() == 0.0) and mesh.linear().deviations() is None
    assert mesh.volumes() == pytest.approx(tet_volumes(mesh.nodes, mesh.elements))


@pytest.mark.parametrize("edge", range(6))
def test_volume_of_an_element_with_a_curved_edge(edge):
    """A mid-edge node moved by ``d`` adds ``d . (A1 n1 + A2 n2) / 3`` to the volume, with
    ``A n`` the area vectors of the two faces of its edge (the integral of the gradient of
    its function, a third of the area on each of them): exactly, for one node."""
    mesh = one_element()
    rng = np.random.default_rng(edge)
    d = 0.2 * rng.uniform(-1.0, 1.0, 3)
    moves = np.zeros_like(mesh.nodes)
    moves[mesh.elements[0, 4 + edge]] = d
    curved = bent(mesh, moves)
    i, j = mesh.elements[0, TET_EDGES[edge]]
    corners = mesh.nodes[:4]
    centre = corners.mean(axis=0)
    expected = mesh.volumes()[0]
    for k in range(4):  # the faces opposite to the two other corners hold the edge
        if k in (i, j):
            continue
        a, b, c = corners[[m for m in range(4) if m != k]]
        area = 0.5 * np.cross(b - a, c - a)
        area *= np.sign(area @ (a - centre))  # outward
        expected += d @ area / 3.0
    assert curved.curved and not mesh.curved
    assert curved.volumes()[0] == pytest.approx(expected, rel=1e-13)
    assert curved.straight().volumes()[0] == pytest.approx(mesh.volumes()[0], rel=1e-13)
    assert np.allclose(curved.straight().nodes, mesh.nodes, atol=1e-15)
    assert np.abs(curved.deviations()[0, edge] - d).max() < 1e-15


def _reference_matrices(x: np.ndarray, d: np.ndarray, k: np.ndarray):
    """Stiffness (30 x 30) and conductivity (10 x 10) matrices of a 10-node tetrahedron of
    nodes ``x`` by the formulas of the isoparametric element: the Jacobian of the map
    from the reference tetrahedron at each of the four integration points."""
    points, weights = GAUSS[2]
    ke, kc = np.zeros((30, 30)), np.zeros((10, 10))
    for bary, w in zip(points, weights, strict=True):
        dn = np.zeros((10, 4))  # derivatives by the barycentric coordinates
        for a in range(4):
            dn[a, a] = 4.0 * bary[a] - 1.0
        for e, (i, j) in enumerate(TET_EDGES):
            dn[4 + e, i], dn[4 + e, j] = 4.0 * bary[j], 4.0 * bary[i]
        dxi = dn[:, 1:] - dn[:, :1]  # by the three coordinates of the reference element
        jac = x.T @ dxi
        grad = dxi @ np.linalg.inv(jac)
        dv = w * abs(np.linalg.det(jac)) / 6.0
        b = np.zeros((6, 30))
        for a in range(10):
            gx, gy, gz = grad[a]
            b[:, 3 * a : 3 * a + 3] = [
                [gx, 0, 0], [0, gy, 0], [0, 0, gz], [0, gz, gy], [gz, 0, gx], [gy, gx, 0],
            ]  # fmt: skip
        ke += dv * b.T @ d @ b
        kc += dv * grad @ k @ grad.T
    return ke, kc


@pytest.mark.parametrize("amplitude", [0.0, 0.05, 0.15])
def test_matrices_of_an_element_against_the_isoparametric_formulas(amplitude):
    mesh = one_element()
    rng = np.random.default_rng(3)
    moves = np.zeros_like(mesh.nodes)
    moves[4:] = amplitude * rng.uniform(-1.0, 1.0, (6, 3))
    mesh = bent(mesh, moves)
    a = rng.uniform(-1.0, 1.0, (6, 6))
    d = a @ a.T + 6.0 * np.eye(6)  # any symmetric positive definite law
    c = rng.uniform(-1.0, 1.0, (3, 3))
    k = c @ c.T + 3.0 * np.eye(3)
    ke, kc = _reference_matrices(mesh.nodes[mesh.elements[0]], d, k)
    mat = np.zeros(1, dtype=np.int64)
    stiffness = assemble(mesh, 3, mat, d[None]).toarray()
    conduction = assemble(mesh, 1, mat, k[None]).toarray()
    order = mesh.elements[0]
    dofs = (3 * order[:, None] + np.arange(3)).ravel()
    assert stiffness[np.ix_(dofs, dofs)] == pytest.approx(ke, rel=1e-11, abs=1e-11 * abs(ke).max())
    assert conduction[np.ix_(order, order)] == pytest.approx(
        kc, rel=1e-11, abs=1e-11 * abs(kc).max()
    )


def test_volumes_of_a_curved_mesh_fill_the_box():
    mesh = structured(amplitude=0.05)
    assert mesh.curved and mesh.matching_faces()
    assert 0.0 < mesh.min_jacobian() < 1.0
    assert mesh.volumes().sum() == pytest.approx(1.0, rel=1e-13)
    # the plane x = 1/2 between the phases: curved, the phases exchange volume
    fractions = mesh.phase_fractions()
    assert sum(fractions.values()) == pytest.approx(1.0, rel=1e-13)
    assert fractions[1] != pytest.approx(0.5, abs=1e-6)
    assert mesh.straight().phase_fractions()[1] == pytest.approx(0.5, rel=1e-12)


@pytest.mark.parametrize("bc", ["kubc", "periodic", "mixed", "subc"])
def test_linear_fields_on_curved_elements(bc):
    """A homogeneous solid meshed with curved elements has its own moduli and
    conductivity: the linear fields are in the space of the isoparametric elements. Under
    prescribed stresses (fluxes) the loads of the faces are integrated on their curved
    triangles and the result is exact; under prescribed strains it is exact up to the
    four-point rule, which the Jacobian of a curved element exceeds."""
    mesh = structured(amplitude=0.03)
    solid = Material(young=3.0, poisson=0.35, conductivity=0.7)
    names = ["bulk_modulus", "shear_modulus", "conductivity"]
    out = homogenize(mesh, {0: solid, 1: solid}, bcs=[bc], properties=names, tol=1e-12)
    bulk, shear = solid.moduli()
    tolerance = 1e-12 if bc == "subc" else 2e-6  # mid-edge nodes moved by 3 % of the cells
    assert out[f"bulk_modulus.{bc}"].value == pytest.approx(bulk, rel=tolerance)
    assert out[f"shear_modulus.{bc}"].value == pytest.approx(shear, rel=tolerance)
    assert out[f"conductivity.{bc}"].value == pytest.approx(0.7, rel=tolerance)
    phases = out[f"bulk_modulus.{bc}"].solution.phases
    assert sum(p.fraction for p in phases) == pytest.approx(1.0, rel=1e-6)


def test_laminate_with_curved_elements_inside_the_layers():
    """Two layers whose interface is plane, meshed with elements curved elsewhere: the
    moduli of the laminate (the mean of the stiffnesses along the layers, of the
    compliances across them)."""
    mesh = structured(amplitude=0.0)
    rng = np.random.default_rng(1)
    interface = np.unique(_interfaces(mesh)[0])
    moves = 0.01 * rng.uniform(-1.0, 1.0, mesh.nodes.shape)
    mids = np.zeros(mesh.n_nodes, dtype=bool)
    mids[mesh.elements[:, 4:]] = True
    inside = np.all((mesh.nodes > 1e-9) & (mesh.nodes < 1.0 - 1e-9), axis=1)
    moves[~(mids & inside)] = 0.0
    moves[interface] = 0.0  # the interface stays plane
    mesh = bent(mesh, moves)
    assert mesh.curved
    k0, k1 = 1.0, 10.0
    materials = {
        0: Material(young=1.0, poisson=0.3, conductivity=k0),
        1: Material(young=1.0, poisson=0.3, conductivity=k1),
    }
    out = homogenize(
        mesh, materials, bcs=["periodic"], properties=["conductivity_tensor"], tol=1e-12
    )
    tensor = out["conductivity_tensor.periodic"]["tensor"]
    assert tensor[0, 0] == pytest.approx(2.0 / (1.0 / k0 + 1.0 / k1), rel=2e-6)  # across
    assert tensor[1, 1] == pytest.approx(0.5 * (k0 + k1), rel=2e-6)  # along


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_plastic_path_of_a_homogeneous_solid_on_curved_elements(kinematics):
    """Along a loading path a homogeneous elastoplastic solid meshed with curved elements
    follows its material point, in small and finite strains."""
    mesh = structured(n=2, amplitude=0.03)
    alloy = Material(
        young=70e3, poisson=0.33, plasticity={"yield": 100.0, "isotropic": [{"Q": 80.0, "b": 20.0}]}
    )
    path = {"times": [0, 1], "increments": 3, "strain": {"11": [0, 0.01]}}
    point = simulate_point(alloy, path, kinematics=kinematics)
    out = homogenize(
        mesh, {0: alloy, 1: alloy}, bcs=["periodic"], properties=[], paths={"pull": path},
        nonlinear={"kinematics": kinematics},
    )["pull.periodic"]  # fmt: skip
    assert out.stress[-1, 0] == pytest.approx(point.stress[-1, 0], rel=1e-5)
    assert out.strain[-1, 1] == pytest.approx(point.strain[-1, 1], rel=1e-5)
    assert out.plastic[-1] == pytest.approx(point.plastic[-1], rel=1e-5)


# -- meshes on the interfaces -----------------------------------------------------------------


@needs_mmg
def test_volume_of_a_meshed_sphere():
    """The polyhedron that MMG makes of a sphere on the background grid loses a tenth of
    its volume with elements of the size of its radius; the curved elements keep it
    within a thousandth, and their nodes lie on the sphere. The exact mesh, the default:
    the same 4-node mesh at both orders."""
    radius = 1.2
    micro, centre = sphere(radius)
    exact = 4.0 / 3.0 * np.pi * radius**3 / 64.0
    errors = {}
    for size in (1.2, 0.4):
        mesh = mg.mesh.conforming_mesh(micro, size=size, method="grid")
        curved = curved_mesh(mesh, micro)
        assert curved.order == 2 and curved.curved and not mesh.quadratic().curved
        assert curved.n_elements == mesh.n_elements and curved.matching_faces()
        assert sum(curved.phase_fractions().values()) == pytest.approx(1.0, rel=1e-12)
        assert curved.min_jacobian() >= 0.3
        errors[size] = (
            mesh.phase_fractions()[1] / exact - 1.0,
            curved.phase_fractions()[1] / exact - 1.0,
        )
        faces, pairs = _interfaces(curved)
        assert np.all(pairs == [0, 1])
        on = np.unique(faces)
        offset = curved.nodes[on] - centre
        offset -= 4.0 * np.round(offset / 4.0)
        distance = np.abs(np.linalg.norm(offset, axis=1) - radius)
        info = curved.metadata["curved"]
        assert info["sampling"] == 0.0 and info["interface_nodes"] == on.size
        # most nodes are on the sphere to round-off; the others moved towards it as far
        # as their elements allow
        assert np.mean(distance < 1e-9 * radius) > 0.9 and info["nodes"] > 0.9 * on.size
    # the mesher makes them by default; without them, the middles of the straight edges
    default = mg.mesh.conforming_mesh(micro, size=0.4, order=2, method="grid")
    assert default.curved and np.allclose(default.nodes, curved.nodes, atol=1e-12)
    flat = mg.mesh.conforming_mesh(micro, size=0.4, order=2, curved=False, method="grid")
    assert not flat.curved and flat.phase_fractions()[1] == pytest.approx(
        mesh.phase_fractions()[1], rel=1e-12
    )
    assert errors[1.2][0] < -0.08 and abs(errors[1.2][1]) < 2e-3
    assert errors[0.4][0] < -0.01 and abs(errors[0.4][1]) < 2e-4
    assert abs(errors[0.4][1]) < 0.3 * abs(errors[1.2][1])
    # the exact route: its 4-node mesh, curved in the same way at order 2
    linear = mg.mesh.conforming_mesh(micro, size=0.4)
    assert linear.metadata["mesher"].startswith("adaptive level set")
    quadratic = mg.mesh.conforming_mesh(micro, size=0.4, order=2)
    assert np.allclose(quadratic.nodes, curved_mesh(linear, micro).nodes, atol=1e-12)


def sangani_acrivos(fraction: float, ratio: float) -> float:
    """The conductivity of a simple cubic array of spheres of conductivity ``ratio`` in a
    matrix of unit conductivity (Sangani & Acrivos 1983, Proc. R. Soc. A 386, 263)."""
    c1, c2, c3, c4, c5, c6 = 1.3047, 0.4054, 0.0723, 0.2305, 0.1526, 0.0105
    l1, l2, l3, l4, l5 = ((ratio - 1.0) / (ratio + s) for s in (2.0, 4 / 3, 6 / 5, 8 / 7, 10 / 9))
    c = fraction
    series = (
        c1 * l2 * c ** (10 / 3) * (1.0 + c4 * l3 * c ** (11 / 3)) / (1.0 - c2 * l2 * c ** (7 / 3))
        + c3 * l3 * c ** (14 / 3)
        + c5 * l4 * c**6
        + c6 * l5 * c ** (22 / 3)
    )
    return 1.0 - 3.0 * c / (-1.0 / l1 + c + series)


@needs_mmg
@pytest.mark.parametrize("ratio", [10.0, 0.0])
def test_conductivity_of_a_cubic_array_of_spheres(ratio):
    """A simple cubic array of spheres filling 27 % of the volume, ten times as conducting
    as their matrix or insulating (pores left out of the mesh, with curved walls), meshed
    with elements three quarters of the radius on the background grid: the curved
    elements give the series of Sangani & Acrivos within a few ten-thousandths (two
    thousandths for the pores), where the same elements with straight edges are 3 % off."""
    radius = 1.6
    micro, _ = sphere(radius)
    fraction = 4.0 / 3.0 * np.pi * radius**3 / 64.0
    reference = sangani_acrivos(fraction, ratio)
    assert reference == pytest.approx({10.0: 1.7645, 0.0: 0.6435}[ratio], abs=1e-4)
    materials = {
        0: Material(young=1.0, poisson=0.3, conductivity=1.0),
        1: Material(young=1.0, poisson=0.3, conductivity=max(ratio, 1.0)),
    }
    mesh = mg.mesh.conforming_mesh(
        micro, size=1.2, phases="all" if ratio else "matrix", method="grid"
    )
    values = {}
    for name, sample in (("straight", mesh.quadratic()), ("curved", curved_mesh(mesh, micro))):
        out = homogenize(sample, materials, bcs=["periodic"], properties=["conductivity"])
        values[name] = out["conductivity.periodic"].value
    assert values["curved"] == pytest.approx(reference, rel=6e-4 if ratio else 3e-3)
    assert abs(values["straight"] / reference - 1.0) > 0.025


def test_functions_of_the_phases_at_any_point():
    """The unions of spheres and their coatings give the implicit functions of their
    phases at any point: those of their grids. The other structures give none."""
    union, _ = sphere(1.2)
    coated, _ = coated_sphere()
    for micro, phases in ((union, [0, 1]), (coated, [0, 1, 2])):
        grid = micro.grid(shape=24)
        axes = [grid.origin[d] + (np.arange(24) + 0.5) * grid.spacing[d] for d in range(3)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        assert micro.implicit_phases() == phases
        for k in phases:
            field = np.asarray(micro.phase_level_set(k, shape=24).data, dtype=float)
            band = float(np.abs(field).max())
            at = micro.implicit_at(k, points, band)
            assert at.shape == (24**3,) and np.abs(at).max() <= band
            assert np.allclose(at.reshape(field.shape), field, atol=1e-6)
    # the interface of the coating and of the matrix: the sphere of radius 1.3
    centre = coated.inner.spheres.centers[0]
    on = centre + 1.3 * np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    assert np.allclose(coated.implicit_at(0, on, 1.0), 0.0, atol=1e-12)
    assert np.allclose(coated.implicit_at(2, on, 1.0), 0.0, atol=1e-12)
    assert np.allclose(coated.implicit_at(1, on, 1.0), 0.3, atol=1e-12)
    cells = model_from_config({"type": "voronoi", "n_cells": 8}).generate(
        mg.Domain.cube(4.0, periodic=True), seed=1
    )
    assert cells.implicit_at(1, on, 1.0) is None and Sampled(union).level_set is not None


@needs_mmg
def test_coated_sphere_on_its_two_interfaces():
    """Three phases whose interfaces are two spheres: the flat faces of the mesher lose
    more than 2 % of the sphere and of its coating, the curved elements less than
    0.15 %, with the functions of the structure at the nodes themselves and with their
    cubic splines; nine nodes in ten lie on their surfaces, and no volume element falls
    below 0.3 of that of its straight tetrahedron."""
    micro, exact = coated_sphere()
    flat = mg.mesh.conforming_mesh(micro, size=0.5, method="grid")
    assert set(np.unique(flat.phase)) == {0, 1, 2}
    for k in (1, 2):
        assert flat.phase_fractions()[k] < 0.98 * exact[k]
    for structure, sampled in ((micro, False), (Sampled(micro), True)):
        curved = curved_mesh(flat, structure)
        info = curved.metadata["curved"]
        assert (info["sampling"] > 0.0) == sampled
        assert curved.curved and curved.matching_faces() and curved.min_jacobian() >= 0.3
        assert info["nodes"] > 0.9 * info["interface_nodes"]
        for k in (1, 2):
            assert curved.phase_fractions()[k] == pytest.approx(exact[k], rel=1.5e-3)
        assert sum(curved.phase_fractions().values()) == pytest.approx(1.0, rel=1e-12)
    pairs = {tuple(int(k) for k in p) for p in _interfaces(curved)[1]}
    assert pairs == {(0, 2), (1, 2)}


@pytest.mark.slow
@needs_mmg
def test_coated_spheres_on_their_interfaces():
    """Random coated spheres: with curved elements the spheres keep their volume within
    0.1 % (3.7 % less with flat faces). The coatings of spheres closer than twice their
    thickness overlap: their volume is the sum of the shells less the lenses that two
    coated spheres share, and they meet along sharp edges, grooves that the mesh follows
    (tests/test_sharp.py): the coatings keep their volume within 0.2 % — 0.9 % too much
    when the grooves are rounded (``sharp_edges=False``), which no node can mend."""
    model = model_from_config(
        {
            "type": "coating",
            "model": {"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.2},
            "thickness": 0.3,
        }
    )
    micro = model.generate(mg.Domain.cube(5.0, periodic=True), seed=2)
    centres = micro.inner.spheres.centers
    gaps = centres[:, None, :] - centres[None, :, :]
    gaps -= 5.0 * np.round(gaps / 5.0)
    apart = np.linalg.norm(gaps, axis=2)[np.triu_indices(len(centres), 1)]
    apart = apart[apart < 2.6]
    lenses = np.sum(np.pi * (2.6 - apart) ** 2 * (apart + 5.2) / 12.0)
    assert len(apart) == 9 and apart.min() > 2.0  # the spheres themselves do not touch
    exact = {
        1: len(centres) * 4.0 / 3.0 * np.pi / 125.0,
        2: (len(centres) * 4.0 / 3.0 * np.pi * (1.3**3 - 1.0) - lenses) / 125.0,
    }
    assert exact[2] == pytest.approx(0.2300, abs=1e-4)  # 0.2407 for shells that would not overlap
    flat = mg.mesh.conforming_mesh(micro, size=0.5, method="grid")
    curved, straight = curved_mesh(flat, micro), flat.quadratic()
    assert curved.curved and not straight.curved and curved.matching_faces()
    info = curved.metadata["curved"]
    assert info["sampling"] == 0.0  # the functions of the structure at the nodes
    assert 0.8 * info["interface_nodes"] < info["nodes"] < info["interface_nodes"]
    assert info["sharp_nodes"] > 400 and len(flat.sharp) > 200  # the grooves of the coatings
    assert curved.min_jacobian() >= 0.3
    assert straight.phase_fractions() == pytest.approx(flat.phase_fractions(), rel=1e-12)
    pairs = {tuple(int(k) for k in p) for p in _interfaces(curved)[1]}
    assert pairs == {(0, 2), (1, 2)}
    assert flat.phase_fractions()[1] < 0.99 * exact[1]
    assert curved.phase_fractions()[1] == pytest.approx(exact[1], rel=1.5e-3)
    assert curved.phase_fractions()[2] == pytest.approx(exact[2], rel=2.5e-3)
    assert sum(curved.phase_fractions().values()) == pytest.approx(1.0, rel=1e-12)
    # the grooves rounded: the coatings fill them, by 0.9 % of their volume
    rounded = mg.mesh.conforming_mesh(micro, size=0.5, order=2, sharp_edges=False, method="grid")
    assert rounded.sharp is None
    assert 1.005 * exact[2] < rounded.phase_fractions()[2] < 1.015 * exact[2]


@pytest.mark.slow
@needs_mmg
def test_faces_of_a_mesh_of_three_phases_are_split_alike():
    """The opposite faces of a periodic mesh of three phases have the same nodes and the
    same edges: its 10-node mesh has a mid-edge node for each of the opposite face, and
    the periodic conditions tie nodes to nodes. (The mesher numbered the vertices of its
    second cut as the first one made them, and split up to six faces along the other
    diagonal on one side: the corners matched, the edges did not.)"""
    from morphogen.homogenization.fe import periodic_map

    model = model_from_config(
        {
            "type": "coating",
            "model": {"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.2},
            "thickness": 0.3,
        }
    )
    micro = model.generate(mg.Domain.cube(5.0, periodic=True), seed=2)
    flat = mg.mesh.conforming_mesh(micro, size=0.9, method="grid")
    assert set(np.unique(flat.phase)) == {0, 1, 2} and flat.matching_faces()
    for mesh in (flat.quadratic(), curved_mesh(flat, micro)):
        assert mesh.matching_faces()
        for axis in range(3):
            assert len(mesh.periodic_pairs(axis)) == mesh.face_nodes(axis, 1).size
        _, _, counts = periodic_map(mesh)
        assert counts["interpolated"] == 0 and counts["free"] == 0


@needs_mmg
def test_options_of_the_mesher_and_of_the_configurations():
    micro, _ = sphere(1.2)
    with pytest.raises(ValueError, match="curved"):
        mg.mesh.conforming_mesh(micro, size=1.2, order=1, curved=True)
    spec = {
        "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.2},
        "domain": {"size": 4.0, "periodic": True},
        "mesh": {"size": 1.0, "order": 2, "curved": False},
    }
    cfg = load_config(spec)
    assert cfg.mesh.curved is False and cfg.mesh.kwargs()["curved"] is False
    spec["mesh"] = {"size": 1.0, "order": 2}
    assert "curved" not in load_config(spec).mesh.kwargs()  # the default of the mesher
    spec["mesh"] = {"size": 1.0, "curved": True}
    with pytest.raises(ConfigError, match=r"mesh\.curved"):
        load_config(spec)
    spec["mesh"] = {"size": 1.0, "order": 2, "curved": "yes"}
    with pytest.raises(ConfigError, match=r"mesh\.curved"):
        load_config(spec)
