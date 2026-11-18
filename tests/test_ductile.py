"""Remeshed crack propagation in inelastic solids: the transfer of the state of the points
between meshes (the identity onto the same mesh, the dissipation kept in small and finite
strains, onto a refined mesh and for crystals, the equilibrium regained), and ductile
growth: the elastic limit, today's propagation, the resistance curve of a viscoplastic
plate and the growth by the porosity at the tip; polycrystals: the grains on the crack
meshes, the transfer grain by grain and the growth through the grains."""

import dataclasses
import math

import numpy as np
import pytest

from morphogen.fracture import lefm
from morphogen.fracture.ductile import (
    _nodal,
    _problem,
    dissipation,
    locate,
    propagate_ductile,
    transfer_displacement,
    transfer_state,
)
from morphogen.fracture.mixed import KinkedCrack, propagate_kinked, solve_crack
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.crystal import CP_ALPHA, CrystalPlasticity
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage
from morphogen.mesh.conforming import has_mmg

W, H = 16.0, 32.0
HARDENING = Material(young=1000.0, poisson=0.3, plasticity={"yield": 1.0, "hardening": 50.0})
VISCOUS = Material(young=1000.0, poisson=0.3, plasticity={
    "yield": 1.0, "hardening": 50.0, "norton": {"K": 1.0, "n": 5.0}})  # fmt: skip
CRACK = lefm.Crack("straight", (0.0, 16.0, 0.0), (0, 1, 0), 4.5, direction=(1, 0, 0))


def _plate(h, thickness=1.0):
    shape = (round(W / h), round(H / h), round(thickness / h))
    return voxel_tet_mesh(VoxelImage(np.zeros(shape, np.uint8), spacing=(h, h, h), periodic=False))


def _grips(m, u=0.03):
    """The faces y = 0 and H clamped and pulled apart by 2u."""
    top = np.flatnonzero(m.nodes[:, 1] > H - 1e-9)
    bottom = np.flatnonzero(m.nodes[:, 1] < 1e-9)
    return [(top, [0, 1, 2], [0.0, u, 0.0]), (bottom, [0, 1, 2], [0.0, -u, 0.0])]


def _mean_p(solution):
    x = solution.mesh.nodes[solution.mesh.elements]
    v = np.abs(
        np.einsum("ij,ij->i", x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))
    )
    return float(v @ solution.bulk.old[:, 6] / v.sum())


def test_location_in_a_mesh():
    """The points of a mesh lie in elements of it (their own vertices: one barycentric
    coordinate 1), the lips kept apart by the labels of the elements."""
    mesh = _plate(2.0, thickness=2.0)
    element, bary = locate(mesh.nodes, mesh)
    assert np.allclose(bary.max(axis=1), 1.0)
    rebuilt = np.einsum("na,nac->nc", bary, mesh.nodes[mesh.elements[element]])
    np.testing.assert_allclose(rebuilt, mesh.nodes, atol=1e-12)
    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    labels = np.where(centroids[:, 1] > 16.0, 1.0, -1.0)
    probe = np.array([[3.0, 16.0 + 1e-9, 0.5], [3.0, 16.0 - 1e-9, 0.5]])
    element, _ = locate(probe, mesh, np.array([1.0, -1.0]), labels)
    assert np.all(labels[element] == [1.0, -1.0])


def test_location_within_groups():
    """With groups (the grains of a polycrystal), a point is located among the elements of
    its own group, the nearest of them when it lies outside them all; a group that has no
    element in the mesh takes any."""
    mesh = _plate(2.0, thickness=2.0)
    groups = (mesh.nodes[mesh.elements].mean(axis=1)[:, 0] > 8.0).astype(int)
    probe = np.array([[8.5, 10.0, 1.0], [7.5, 10.0, 1.0], [7.5, 10.0, 1.0]])
    element, bary = locate(probe, mesh, point_groups=[0, 1, 2], element_groups=groups)
    assert groups[element[0]] == 0 and groups[element[1]] == 1  # across the boundary x = 8
    assert np.all(np.isfinite(bary)) and np.allclose(bary.sum(axis=1), 1.0)


def test_transfer_onto_the_same_mesh_is_the_identity():
    """The state of a hardening plate pulled by its grips (a crack of 4.5, the yield strain
    1/1000, pulled to three times it): carried onto the same mesh by ``"nearest"``, the
    identity (its equilibrium regained at once, within the tolerance of Newton's method),
    the loading going on as in one go; by ``"projection"``, smoothed (the plastic strains of
    the tip spread over the nodes), the dissipation kept (the dissipated density moves, the
    free energy follows the new state)."""
    mesh = _plate(2.0, thickness=2.0)
    first = solve_crack(mesh, CRACK, HARDENING, _grips, steps=4)
    same = solve_crack(mesh, CRACK, HARDENING, _grips, initial=first, transfer="nearest")
    scale = np.abs(first.displacement).max()
    assert np.abs(same.displacement - first.displacement).max() < 1e-7 * scale
    assert np.abs(same.bulk.old - first.bulk.old).max() < 1e-9
    assert dissipation(same) == pytest.approx(dissipation(first), rel=1e-9)
    assert same.release_rate(energy="tip") == pytest.approx(
        first.release_rate(energy="tip"), rel=1e-6
    )
    on = solve_crack(
        mesh, CRACK, HARDENING, _grips, 1.5, steps=2, initial=first, transfer="nearest"
    )
    once = solve_crack(mesh, CRACK, HARDENING, _grips, 1.5, steps=6)
    assert on.release_rate() == pytest.approx(once.release_rate(), rel=0.01)
    assert dissipation(on) == pytest.approx(dissipation(once), rel=1e-3)
    smooth = solve_crack(mesh, CRACK, HARDENING, _grips, initial=first, transfer="projection")
    assert dissipation(smooth) == pytest.approx(dissipation(first), rel=1e-9)
    assert np.abs(smooth.bulk.old - first.bulk.old).max() > 1e-4  # smoothed


def test_transfer_in_finite_strains():
    """The plate in finite strains, its state projected onto the same mesh: the dissipation
    kept exactly (the means of the projection, interpolated at the centroids, conserve it;
    the equilibrium regained dissipates nothing, although the free energy now follows the
    interpolated displacement), the driving forces those of small strains (strains of a
    few 10⁻³)."""
    mesh = _plate(2.0, thickness=2.0)
    drives = {}
    for kinematics in ("small", "finite"):
        first = solve_crack(mesh, CRACK, HARDENING, _grips, steps=4, kinematics=kinematics)
        smooth = solve_crack(mesh, CRACK, HARDENING, _grips, initial=first,
                             transfer="projection", kinematics=kinematics)  # fmt: skip
        assert dissipation(smooth) == pytest.approx(dissipation(first), rel=1e-9)
        drives[kinematics] = smooth.release_rate(energy="tip"), smooth.release_rate()
    np.testing.assert_allclose(drives["finite"], drives["small"], rtol=0.01)


def test_transfer_of_a_crystal():
    """A cubic crystal with octahedral slip (finite strains): the ``Fp⁻¹`` of its points,
    projected, made isochoric again (the means of isochoric tensors are not), the
    dissipation kept."""
    slip = CrystalPlasticity.from_config({"tau0": 60.0, "Q": 40.0, "b": 5.0, "K": 5.0, "n": 20.0})
    slipping = dataclasses.replace(Material.cubic(168e3, 121e3, 75e3), crystal_plasticity=slip)
    mesh = _plate(2.0, thickness=2.0)
    first = solve_crack(mesh, CRACK, slipping, _grips, steps=4, kinematics="finite", duration=1.0)

    def determinants(bulk, state):
        offsets = CP_ALPHA + 2 * bulk.behaviours.data[7][bulk.pmat]
        g = np.stack([row[o : o + 9] for row, o in zip(state, offsets, strict=True)])
        return np.linalg.det(g.reshape(-1, 3, 3) + np.eye(3))

    means = _nodal(first.mesh, first.bulk.old)[first.mesh.elements].mean(axis=1)
    assert np.abs(determinants(first.bulk, means) - 1.0).max() > 1e-9
    opened, _, _, problem = _problem(mesh, CRACK, slipping, "finite", None, "auto")
    transfer_state(first, opened, problem.bulk, CRACK, "projection")
    assert np.abs(determinants(problem.bulk, problem.bulk.old) - 1.0).max() < 1e-12
    moved = solve_crack(mesh, CRACK, slipping, _grips, initial=first, kinematics="finite",
                        duration=1.0)  # fmt: skip
    assert dissipation(moved) == pytest.approx(dissipation(first), rel=1e-9)


def test_transfer_onto_a_refined_mesh():
    """The state of the plate of elements of 2 carried onto elements of 1, which nest in
    them (Kuhn's triangulation): the dissipation exactly by both methods, the mean plastic
    strain within 10 % (the fine mesh corrects the equilibrium of the coarse one), the
    equilibrium regained, the displacement of the old mesh reproduced at its nodes."""
    from scipy.spatial import cKDTree

    coarse, fine = _plate(2.0, thickness=2.0), _plate(1.0, thickness=2.0)
    low = solve_crack(coarse, CRACK, HARDENING, _grips, steps=4)
    for method in ("nearest", "projection"):
        moved = solve_crack(fine, CRACK, HARDENING, _grips, initial=low, transfer=method)
        assert 0 < moved.steps.equilibration < 25
        assert dissipation(moved) == pytest.approx(dissipation(low), rel=1e-6)
        assert _mean_p(moved) == pytest.approx(_mean_p(low), rel=0.12)
    away = np.abs(low.mesh.nodes[:, 1] - 16.0) > 1e-9  # the lips: two nodes at a place
    distance, at = cKDTree(moved.mesh.nodes).query(low.mesh.nodes[away])
    assert distance.max() < 1e-12
    u = transfer_displacement(low, moved.mesh, CRACK)
    np.testing.assert_allclose(u[at], low.displacement[away], rtol=0, atol=1e-14)


def _mesh_of(crack):
    return lefm.crack_mesh((0, 0, 0), (W, H, 1.0), crack, size=1.0)


def _mixed(m):
    """Tension and shear (0.7) of the faces of the plate."""
    f = lefm.face_forces(m, lambda c: c[:, 1] > H - 1e-9, [0.7, 1.0, 0.0])
    f += lefm.face_forces(m, lambda c: c[:, 1] < 1e-9, [-0.7, -1.0, 0.0])
    f += lefm.face_forces(m, lambda c: c[:, 0] > W - 1e-9, [0.0, 0.7, 0.0])
    return f + lefm.face_forces(m, lambda c: c[:, 0] < 1e-9, [0.0, -0.7, 0.0])


def _rigid(m):
    def nearest(p):
        return int(np.argmin(np.linalg.norm(m.nodes - np.asarray(p), axis=1)))

    a, b = nearest((W, H / 2, 0.0)), nearest((W, H, 0.0))
    return [(a, [0, 1], 0.0), (b, 0, 0.0), (np.flatnonzero(m.nodes[:, 2] < 1e-9), 2, 0.0)]


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_elastic_limit_of_ductile_growth():
    """An elastic plate in tension and shear, its crack growing once its G reaches 1.5 times
    that of the unit load: the angles of today's propagation exactly (its fronts rounded,
    the same meshes), the first advance at the first load step beyond sqrt(1.5) (G as the
    square of the load), and the further advances at once (dead loads: unstable growth)."""
    elastic = Material(young=1000.0, poisson=0.0)
    crack = KinkedCrack.of(CRACK)
    today = propagate_kinked(
        _mesh_of, crack, elastic, _rigid, forces=_mixed, increment=1.5, steps=2
    )
    toughness = 1.5 * today.rates[0]
    growth = propagate_ductile(_mesh_of, crack, elastic, _rigid, forces=_mixed, load_step=0.25,
                               increment=1.5, steps=2, toughness=toughness)  # fmt: skip
    np.testing.assert_allclose(growth.angles, today.angles, atol=1e-9)
    assert np.allclose(growth.loads, 1.25) and math.sqrt(1.5) < growth.loads[0]
    assert growth.tip[0] / toughness == pytest.approx(1.25**2 / 1.5, rel=1e-6)
    assert np.all(np.diff(growth.tip) > 0) and len(growth.rows()) == 3


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_resistance_curve_of_a_viscoplastic_plate():
    """The plate of a hardening viscoplastic metal (Norton; the load factor the time) pulled
    by its grips, its crack growing straight by 1.5 once ``J_tip`` reaches 0.03, the state
    carried by the nearest element: the load of each advance higher than the last (the
    plastic wake shields the tip), the far J rising from the first advance (the resistance
    curve)."""
    growth = propagate_ductile(_mesh_of, KinkedCrack.of(CRACK), VISCOUS, _grips, load_step=0.5,
                               increment=1.5, steps=2, toughness=0.03, kink=False,
                               transfer="nearest")  # fmt: skip
    advance, far = growth.resistance()
    np.testing.assert_allclose(advance, [0.0, 1.5, 3.0], atol=1e-9)
    assert np.all(growth.tip >= 0.03) and np.all(np.diff(growth.loads) > 0)
    assert far[1] > 1.1 * far[0]


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_growth_by_the_porosity_at_the_tip():
    """A plate of GTN points (``f0 = 0.01``) pulled by its grips, its crack advancing once the
    porosity within 1.5 element of the front reaches 0.0105 (the local approach): the load
    steps before each advance below it, the voids of the new tip grown again under a higher
    load."""
    from morphogen.homogenization.porous import PorousPlasticity

    gtn = PorousPlasticity("gurson", yield_stress=1.0, porosity=0.01, hardening=50.0)
    metal = Material(young=1000.0, poisson=0.3, porous_plasticity=gtn)
    growth = propagate_ductile(_mesh_of, KinkedCrack.of(CRACK), metal, _grips, load_step=0.5,
                               increment=1.5, steps=1, porosity=0.0105, kink=False,
                               transfer="nearest")  # fmt: skip
    porosity = np.array([row["porosity_tip"] for row in growth.history])
    advance = np.array([row["advance"] for row in growth.history])
    last = np.r_[advance[1:] != advance[:-1], True]  # the last load step of each crack
    assert np.all(porosity[last] >= 0.0105) and np.all(porosity[~last] < 0.0105)
    assert len(growth.cracks) == 2 and growth.loads[1] > growth.loads[0]


# -- polycrystals -------------------------------------------------------------------------------


def _polycrystal():
    """Twelve columnar grains of a cubic crystal (a Voronoi tessellation of germs in a plane,
    random orientations) in a plate of 8 x 8 x 0.5, a crack of 2.5 from the middle of its
    left edge, the plate pulled by its faces y = 0 and 8 in plane strain."""
    from morphogen.domain import Domain
    from morphogen.models.tessellation import Tessellation

    rng = np.random.default_rng(3)
    germs = np.column_stack([rng.uniform(0, 8, 12), rng.uniform(0, 8, 12), np.full(12, 0.25)])
    euler = rng.uniform(0, 1, (12, 3)) * [360.0, 180.0, 360.0]
    tessellation = Tessellation(Domain(size=(8.0, 8.0, 0.5), periodic=False), germs,
                                orientations=euler)  # fmt: skip

    def grips(m, u=1.0):
        x = m.nodes
        top, bottom = x[:, 1] > 8.0 - 1e-9, x[:, 1] < 1e-9
        faces = (x[:, 2] < 1e-9) | (x[:, 2] > 0.5 - 1e-9)
        corner = int(np.argmin(np.linalg.norm(x - [8.0, 4.0, 0.0], axis=1)))
        return [(np.flatnonzero(top & ~faces), 1, u), (np.flatnonzero(bottom & ~faces), 1, -u),
                (np.flatnonzero(top & faces), [1, 2], [u, 0.0]),
                (np.flatnonzero(bottom & faces), [1, 2], [-u, 0.0]),
                (np.flatnonzero(faces & ~top & ~bottom), 2, 0.0), (corner, 0, 0.0)]  # fmt: skip

    def mesh_of(crack):
        return lefm.crack_mesh((0, 0, 0), (8.0, 8.0, 0.5), crack, size=0.5, grains=tessellation)

    cubic = Material.cubic(168e3, 121e3, 75e3)
    crystal = dataclasses.replace(cubic, crystal_plasticity=CrystalPlasticity.from_config(
        {"tau0": 60.0, "Q": 40.0, "b": 5.0, "K": 5.0, "n": 20.0}))  # fmt: skip
    crack = KinkedCrack(((0.0, 4.0, 0.0), (2.5, 4.0, 0.0)))
    return tessellation, grips, mesh_of, cubic, crystal, crack


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_crack_meshes_of_a_polycrystal():
    """The crack meshes of a polycrystal carry its grains (each element in the cell of its
    centroid, the orientations of the cells, kept by the opened mesh); the elastic bulk and
    the points of crystal elasticity are the same body (the grains turned in both); the
    transfer onto the same mesh is the identity, and onto the mesh of another crack it
    moves the state of each grain within the grain, never across a boundary."""
    tessellation, grips, mesh_of, cubic, crystal, crack = _polycrystal()
    mesh = mesh_of(crack)
    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    assert np.array_equal(mesh.cell, tessellation.cell_of(centroids))
    assert np.array_equal(mesh.orientations, tessellation.orientations)
    elastic = solve_crack(mesh, crack, cubic, lambda m: grips(m, 0.004), bulk="elastic")
    points = solve_crack(mesh, crack, cubic, lambda m: grips(m, 0.004), bulk="points", steps=1)
    assert np.array_equal(points.mesh.orientations, tessellation.orientations)
    scale = np.abs(elastic.displacement).max()
    assert np.abs(points.displacement - elastic.displacement).max() < 1e-9 * scale
    g = elastic.release_rate()
    assert points.release_rate(energy="tip") == pytest.approx(g, rel=1e-9)
    assert np.isfinite(elastic.modes()["angle"])  # the Voigt-Reuss-Hill estimate of a crystal
    plastic = solve_crack(mesh, crack, crystal, lambda m: grips(m, 0.012), steps=4)
    assert plastic.bulk.old[:, 6].max() > 1e-3  # the grains slip
    opened, _, _, problem = _problem(mesh, crack, crystal, "small", None, "auto")
    transfer_state(plastic, opened, problem.bulk, crack, "nearest")
    assert np.array_equal(problem.bulk.old, plastic.bulk.old)
    grown = crack.grown(1.0, math.radians(20.0))
    other, _, _, target = _problem(mesh_of(grown), grown, crystal, "small", None, "auto")
    plastic.bulk.old[...] = plastic.mesh.cell[:, None]  # the state: the grain of the point
    for method in ("nearest", "projection"):
        transfer_state(plastic, other, target.bulk, grown, method)
        assert np.abs(target.bulk.old - other.cell[:, None]).max() < 1e-12


@pytest.mark.slow
@pytest.mark.skipif(not has_mmg(), reason="MMG")
def test_ductile_growth_through_a_polycrystal():
    """The crack grows by the driving force at its tip through the grains of crystal
    plasticity, remeshed at each advance (the state moved grain by grain): it crosses grain
    boundaries and deflects there (the elastic estimates of the interaction integrals with
    the isotropic moduli of the crystal)."""
    tessellation, grips, mesh_of, _, crystal, crack = _polycrystal()
    growth = propagate_ductile(mesh_of, crack, crystal, grips, load_step=0.001, increment=1.0,
                               steps=2, toughness=0.6, max_load=0.03, transfer="nearest",
                               duration=1.0)  # fmt: skip
    assert len(growth.cracks) == 3 and np.all(growth.tip >= 0.6)
    assert np.all(np.diff(growth.loads) >= 0) and np.all(np.isfinite(growth.angles))
    grains = [int(tessellation.cell_of(np.atleast_2d(c.tip))[0]) for c in growth.cracks]
    assert len(set(grains)) >= 2  # across a grain boundary
    assert np.abs(np.degrees(growth.angles)).max() > 1.0  # not straight: deflected
