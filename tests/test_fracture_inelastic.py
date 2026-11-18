"""Cracks in inelastic solids, in small and finite strains: the behaviours of morphoGen's
points as the bulk of the crack solvers (their elastic limit against the elastic bulks),
the driving forces of a crack in a hardening plate (the J of the deformation theory against
the release of the work of the loads, the free energy and the configurational forces of
the internal variables at the tip), a cohesive crack in an elasto-viscoplastic plate in
finite strains (the dissipation of the bulk, the rate) and the intergranular cracking of a
polycrystal in crystal plasticity (cohesive grain boundaries)."""

import dataclasses

import numpy as np
import pytest

from morphogen.fracture import lefm
from morphogen.fracture.cohesive import Bilinear, CohesiveProblem, Contact, insert_interfaces
from morphogen.fracture.mixed import energy_release_rate, solve_crack
from morphogen.homogenization.bloch import voxel_tet_mesh
from morphogen.homogenization.crystal import CrystalPlasticity
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage
from morphogen.mesh.tetra import TetMesh

W, H = 16.0, 32.0
ELASTIC = Material(young=1000.0, poisson=0.3)
HARDENING = Material(young=1000.0, poisson=0.3, plasticity={"yield": 1.0, "hardening": 50.0})
CRACK = lefm.Crack("straight", (0.0, 16.0, 0.0), (0, 1, 0), 4.5, direction=(1, 0, 0))


def _plate():
    image = VoxelImage(np.zeros((16, 32, 2), np.uint8), spacing=(1.0, 1.0, 1.0), periodic=False)
    return voxel_tet_mesh(image)


def _grips(u):
    """The faces y = 0 and H clamped and pulled apart by 2u."""

    def entries(m):
        top = np.flatnonzero(m.nodes[:, 1] > H - 1e-9)
        bottom = np.flatnonzero(m.nodes[:, 1] < 1e-9)
        return [(top, [0, 1, 2], [0.0, u, 0.0]), (bottom, [0, 1, 2], [0.0, -u, 0.0])]

    return entries


@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_elastic_limit_of_the_points(kinematics):
    """Points that never yield are the elastic bulk (Hencky's elasticity against the
    neo-Hookean solid in finite strains, at small strains), their work, free energy and
    tip driving force the G-θ of the elastic body; the works add up."""
    mesh = _plate()
    never = dataclasses.replace(HARDENING, plasticity=dataclasses.replace(
        HARDENING.plasticity, yield_stress=1e9))  # fmt: skip
    steps = 2 if kinematics == "finite" else 1
    elastic = solve_crack(mesh, CRACK, ELASTIC, _grips(0.02), kinematics=kinematics, steps=steps)
    points = solve_crack(mesh, CRACK, never, _grips(0.02), kinematics=kinematics, steps=steps)
    assert elastic.bulk is None and points.bulk is not None
    tol = 1e-9 if kinematics == "small" else 2e-3
    scale = np.abs(elastic.displacement).max()
    assert np.abs(points.displacement - elastic.displacement).max() < tol * scale
    g = elastic.release_rate()
    for energy in ("work", "free", "tip"):
        assert points.release_rate(energy=energy) == pytest.approx(g, rel=tol)
    work = points.steps.work
    assert work["external"][-1] == pytest.approx(work["bulk"][-1], rel=1e-9)
    assert points.energy("work") == pytest.approx(work["bulk"][-1], rel=1e-9)
    assert points.energy("free") == pytest.approx(points.energy("work"), rel=tol)


@pytest.mark.slow
def test_driving_forces_in_a_hardening_plate():
    """An edge crack in a plate of von Mises hardening yielding at 1/1000 of its modulus,
    pulled to three times that strain: the J of the work density approaches the release
    of the work of the loads by the larger domains (Begley & Landes); with the free energy
    alone it changes with the domain, and the configurational forces of the gradients of
    the plastic strain and hardening restore a tip value independent of it, below the far
    J (the shielding of the plastic zone); the same in finite strains."""
    mesh = _plate()

    def solve(crack, kinematics="small"):
        return solve_crack(mesh, crack, HARDENING, _grips(0.05), kinematics=kinematics, steps=8)

    plastic = solve(CRACK)
    assert np.count_nonzero(plastic.bulk.old[:, 6] > 0) > 0.3 * mesh.n_elements
    radii = (3.0, 4.0, 5.0)
    work = [plastic.release_rate(r_in=1.0, r_out=r, energy="work") for r in radii]
    free = [plastic.release_rate(r_in=1.0, r_out=r, energy="free") for r in radii]
    tip = [plastic.release_rate(r_in=1.0, r_out=r, energy="tip") for r in radii]
    begley_landes = energy_release_rate(solve, CRACK, 0.5)
    for j in work:
        assert j == pytest.approx(begley_landes, rel=0.05)
    assert max(tip) - min(tip) < 0.02 * tip[0]
    assert max(free) - min(free) > 0.04 * free[0]
    assert tip[-1] < 0.9 * work[-1]
    assert plastic.energy("free") < 0.5 * plastic.energy("work")  # most of the work dissipated
    finite = solve(CRACK, "finite")
    tip_finite = [finite.release_rate(r_in=1.0, r_out=r, energy="tip") for r in radii]
    assert max(tip_finite) - min(tip_finite) < 0.02 * tip_finite[0]
    assert tip_finite[-1] == pytest.approx(tip[-1], rel=0.02)  # small strains on the whole


@pytest.mark.slow
def test_ductile_crack_in_a_porous_plate():
    """The same plate of Gurson-Tvergaard-Needleman (the porous points of
    :mod:`morphogen.homogenization.porous`, a matrix yielding at 1/1000 of the modulus):
    the voids grow most at the tip, the driving forces of the work, of the free energy and
    at the tip in decreasing order (the hardening of the matrix counted as dissipated), the
    same in small and finite strains."""
    from morphogen.homogenization.porous import PorousPlasticity, porosity

    gtn = PorousPlasticity("gurson", yield_stress=1.0, porosity=0.01, hardening=50.0)
    metal = Material(young=1000.0, poisson=0.3, porous_plasticity=gtn)
    mesh = _plate()
    forces = {}
    for kinematics in ("small", "finite"):
        solution = solve_crack(mesh, CRACK, metal, _grips(0.02), kinematics=kinematics, steps=8)
        bulk = solution.bulk
        assert bulk is not None
        f = porosity(bulk.old, bulk.behaviours.data[2], bulk.pmat)
        r = CRACK.frame(solution.mesh.nodes[solution.mesh.elements].mean(axis=1))[0]
        assert f[r < 1.0].max() > 1.01 * 0.01 and f[r < 1.0].mean() > f[r > 8.0].mean()
        work, free, tip = (solution.release_rate(energy=e) for e in ("work", "free", "tip"))
        assert work > free > tip > 0.5 * work
        forces[kinematics] = (work, tip)
    np.testing.assert_allclose(forces["finite"], forces["small"], rtol=0.01)


def test_cohesive_crack_in_a_viscoplastic_plate():
    """A cohesive zone ahead of an edge crack in finite strains: in an elastic plate the
    stored energy is released when it breaks; in an elasto-viscoplastic one, yielding
    lowers the peak load, the bulk dissipates more than the interfaces (the dissipation per
    unit of cracked area grows: the R-curve of bulk plasticity), and a faster loading raises
    the peak load (viscous overstress); the works of the bulk and the interfaces make the
    work of the loads."""
    mesh = voxel_tet_mesh(VoxelImage(np.zeros((8, 8, 1), np.uint8), periodic=False))
    opened, faces = insert_interfaces(mesh, lambda c: np.abs(c[:, 1] - 4.0) < 1e-9)
    ahead = faces.subset(faces.centroids(opened)[:, 0] > 2.0)
    law = Bilinear(stiffness=1e5, normal_strength=10.0, shear_strength=10.0, mode_I=0.2,
                   mode_II=0.2)  # fmt: skip
    x = opened.nodes
    top, bottom = np.flatnonzero(x[:, 1] > 8.0 - 1e-9), np.flatnonzero(x[:, 1] < 1e-9)
    grips = [(top, [0, 1, 2], [0.0, 1.0, 0.0]), (bottom, [0, 1, 2], [0.0, -1.0, 0.0])]
    viscoplastic = Material(young=1000.0, poisson=0.3, plasticity={
        "yield": 4.0, "hardening": 50.0, "norton": {"K": 5.0, "n": 5.0}})  # fmt: skip
    out = {}
    for name, material, duration in (("elastic", ELASTIC, 10.0), ("slow", viscoplastic, 10.0),
                                     ("fast", viscoplastic, 0.1)):  # fmt: skip
        problem = CohesiveProblem(opened, ahead, {0: material}, law, kinematics="finite")
        result = problem.run(grips, 0.12, steps=60, viscosity=1e-4, duration=duration)
        work = result.work
        assert work["external"][-1] == pytest.approx(  # within the residual of Newton
            work["bulk"][-1] + work["interfaces"][-1], rel=1e-6
        )
        free = 0.0 if problem.bulk is None else float(problem.bulk.free_energy() @ problem.bulk.wts)
        out[name] = result, work["bulk"][-1] - free, work["interfaces"][-1]
    elastic, slow, fast = out["elastic"], out["slow"], out["fast"]
    assert elastic[0].cracked[-1] == pytest.approx(6.0) and abs(elastic[1]) < 1e-3
    assert slow[0].cracked[-1] == pytest.approx(6.0)
    assert slow[0].reactions[:, 0, 1].max() < 0.9 * elastic[0].reactions[:, 0, 1].max()
    assert slow[1] > slow[2] > 0.2 * 6.0  # the bulk dissipates; the interfaces at least G_c
    assert (slow[1] + slow[2]) / 6.0 > 1.2 * (elastic[1] + elastic[2]) / 6.0
    assert fast[0].reactions[:, 0, 1].max() > 1.05 * slow[0].reactions[:, 0, 1].max()


def test_intergranular_cracking_of_a_polycrystal():
    """Four grains of a cubic crystal with octahedral slip, their boundaries cohesive,
    pulled across one of them in finite strains: only that boundary breaks; the grains slip
    before (and dissipate), which lowers the peak load against crystal elasticity; the
    orientations of the grains follow the opened mesh."""
    grains = np.zeros((4, 4, 2), np.uint8)
    grains[2:, :2], grains[:2, 2:], grains[2:, 2:] = 1, 2, 3
    base = voxel_tet_mesh(VoxelImage(grains, periodic=False))
    euler = np.array(
        [[0.0, 0.0, 0.0], [45.0, 30.0, 10.0], [120.0, 60.0, 80.0], [200.0, 15.0, 45.0]]
    )
    mesh = TetMesh(base.nodes, base.elements, np.zeros_like(base.phase), base.lower, base.upper,
                   cell=base.phase, orientations=euler)  # fmt: skip
    opened, faces = insert_interfaces(mesh, lambda c: np.ones(len(c), dtype=bool), sides=mesh.cell)
    assert faces.n_elements == 32 and np.array_equal(opened.orientations, euler)
    cubic = Material.cubic(168e3, 121e3, 75e3)
    slipping = dataclasses.replace(cubic, crystal_plasticity=CrystalPlasticity.from_config(
        {"tau0": 60.0, "Q": 40.0, "b": 5.0, "K": 5.0, "n": 20.0}))  # fmt: skip
    law = Bilinear(stiffness=1e7, normal_strength=150.0, shear_strength=150.0, mode_I=0.5,
                   mode_II=0.5)  # fmt: skip
    x = opened.nodes
    left, right = np.flatnonzero(x[:, 0] < 1e-9), np.flatnonzero(x[:, 0] > 4.0 - 1e-9)
    a = int(np.argmin(np.linalg.norm(x - [0.0, 0.0, 0.0], axis=1)))
    b = int(np.argmin(np.linalg.norm(x - [0.0, 4.0, 0.0], axis=1)))
    prescribed = [(left, 0, 0.0), (right, 0, 1.0), (a, [1, 2], 0.0), (b, 2, 0.0)]
    corners = opened.nodes[faces.minus]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    across = np.abs(normals[:, 0]) > 0.9 * np.linalg.norm(normals, axis=1)
    peaks = {}
    for name, material in (("elastic", cubic), ("plastic", slipping)):
        problem = CohesiveProblem(opened, faces, {0: material}, law, kinematics="finite")
        assert problem.bulk is not None  # anisotropic in finite strains: Hencky's points
        result = problem.run(prescribed, 0.02, steps=40, viscosity=1e-4, duration=1.0)
        damage = result.damage.mean(axis=1)
        assert damage[across].min() > 0.99 and damage[~across].max() < 0.05
        assert result.cracked[-1] == pytest.approx(8.0)
        peaks[name] = result.reactions[:, 1, 0].max()
        slip = problem.bulk.old[:, 6].max()
        dissipated = result.work["bulk"][-1] - problem.bulk.free_energy() @ problem.bulk.wts
        if name == "plastic":
            assert slip > 1e-3 and dissipated > 0.2 * result.work["interfaces"][-1]
        else:
            assert slip == 0.0 and abs(dissipated) < 1e-3 * result.work["external"][-1]
    assert peaks["plastic"] < 0.9 * peaks["elastic"]


def test_contact_and_bulk_options():
    """The contact of the lips with a plastic bulk, and the choices of the bulk."""
    mesh = _plate()
    closing = solve_crack(mesh, CRACK, HARDENING, _grips(-0.02), contact=True, steps=4)
    jump = closing.displacement[closing.faces.plus] - closing.displacement[closing.faces.minus]
    assert jump[..., 1].min() > -1e-4 and closing.bulk is not None
    opened, faces = lefm.crack_faces(mesh, CRACK)
    with pytest.raises(ValueError, match="inelastic"):
        CohesiveProblem(opened, faces, {0: HARDENING}, Contact(1.0), bulk="elastic")
    with pytest.raises(ValueError, match="bulk"):
        CohesiveProblem(opened, faces, {0: ELASTIC}, Contact(1.0), bulk="plastic")
    forced = CohesiveProblem(opened, faces, {0: ELASTIC}, Contact(1.0), bulk="points")
    assert forced.bulk is not None and forced.bulk.free_energy().shape == (opened.n_elements,)
