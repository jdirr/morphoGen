"""Thermo-chemo-elastic fields in time (morphogen.coupling.thermochemoelastic) against
closed forms: Crank's slabs, free and clamped expansions, a periodic laminate, Timoshenko's
bilayer, the curvature of a beam that swells from a face, the energy of an adiabatic
reaction, Arrhenius' scaling of the times and the Boltzmann equilibrium of a species drawn
by the stress."""

from __future__ import annotations

import re

import numpy as np
import pytest

from helpers import box_mesh
from morphogen.coupling import (
    ThermoChemoElastic,
    ThermoChemoMaterial,
    arrhenius,
    beam_curvature,
    thermochemoelastic,
    timoshenko_curvature,
)
from morphogen.coupling.diffusion import slab_profile, slab_uptake, transient_diffusion
from morphogen.coupling.thermochemoelastic import R_GAS, bernoulli
from morphogen.homogenization.materials import Material
from morphogen.image import VoxelImage

ALL = ("x-", "x+", "y-", "y+", "z-", "z+")


def test_species_as_transient_diffusion():
    """The finite volumes of the species are those of coupling.diffusion: the same
    concentrations, steps and faces, to round-off."""
    rng = np.random.default_rng(0)
    image = VoxelImage((rng.random((6, 5, 7)) < 0.4).astype(np.uint8), spacing=(0.1,) * 3)
    times = [0.01, 0.05, 0.2]
    ref = transient_diffusion(image, {0: 1.0, 1: 0.1}, times, solubility={0: 1.0, 1: 3.0},
                              exposed=("x-", "z+"), steps=30)  # fmt: skip
    mats = {
        0: ThermoChemoMaterial(young=1.0, poisson=0.3, diffusivity=1.0, solubility=1.0),
        1: ThermoChemoMaterial(young=2.0, poisson=0.3, diffusivity=0.1, solubility=3.0),
    }
    out = thermochemoelastic(image, mats, times, activity={"x-": 1.0, "z+": 1.0}, steps=30)
    for a, b in zip(ref.fields, out.concentration, strict=True):
        np.testing.assert_allclose(b, a, atol=1e-12)
    np.testing.assert_allclose(out.history["uptake"], ref.uptake, atol=1e-12)


def test_slab_conduction_and_the_order_of_trbdf2():
    """A slab heated on both faces: its mean temperature against Crank's series (eq. 4.18);
    in time, implicit Euler of the first order and TR-BDF2 of the second (their errors
    against 400 steps of TR-BDF2 halved and quartered from 10 to 20 steps per interval)."""
    n = 24
    image = VoxelImage(np.zeros((1, 1, n), np.uint8), spacing=(1.0 / n,) * 3)
    m = ThermoChemoMaterial(young=1.0, poisson=0.25, conductivity=2.0, heat_capacity=4.0)
    times = np.array([0.005, 0.02, 0.05, 0.1, 0.3])
    common = {"reference_temperature": 300.0, "temperature": {"z-": 310.0, "z+": 310.0}}

    def rise(scheme, steps):
        out = thermochemoelastic(image, {0: m}, times, scheme=scheme, steps=steps, **common)
        return (out.history["mean_temperature"] - 300.0) / 10.0

    exact = slab_uptake(times, 1.0, 0.5)
    assert np.abs(rise("euler", 40) - exact).max() < 8e-3
    assert np.abs(rise("trbdf2", 40) - exact).max() < 6e-3
    fine = rise("trbdf2", 400)
    e10, e20 = (np.abs(rise("euler", k) - fine).max() for k in (10, 20))
    t10, t20 = (np.abs(rise("trbdf2", k) - fine).max() for k in (10, 20))
    assert 1.8 < e10 / e20 < 2.2 and 3.5 < t10 / t20 < 4.5
    assert t10 < 0.02 * e10
    # the profile at the end, against eq. 4.17
    out = thermochemoelastic(image, {0: m}, times, scheme="trbdf2", steps=40, **common)
    z = (np.arange(n) + 0.5) / n - 0.5
    profile = (out.temperature[-1][0, 0] - 300.0) / 10.0
    np.testing.assert_allclose(profile, slab_profile(z, 0.3, 1.0, 0.5), atol=2e-3)


def _steel(**extra):
    return ThermoChemoMaterial(young=200e9, poisson=0.3, expansion=1e-5, conductivity=50.0,
                               **extra)  # fmt: skip


def test_free_and_clamped_expansion_on_voxels_and_meshes():
    """A body heated uniformly by 50 K: free, no stress and the strain alpha dT; clamped on
    every face, sigma = -3 K alpha dT (exact for both discretizations)."""
    image = VoxelImage(np.zeros((4, 3, 5), np.uint8), spacing=(1e-3,) * 3)
    mesh = box_mesh(n=(3, 2, 4), upper=(3e-3, 2e-3, 4e-3), periodic=False)
    k = 200e9 / (3.0 * (1.0 - 0.6))
    for sample in (image, mesh):
        common = {"reference_temperature": 300.0, "initial_temperature": 350.0}
        free = thermochemoelastic(sample, {0: _steel()}, [0.0], **common)
        assert np.abs(free.stress).max() < 1e-12 * 3 * k * 5e-4
        d = free.displacement[0].reshape(-1, 3)
        x = free.nodes
        strain = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), d, rcond=None)[0][:3]
        np.testing.assert_allclose(strain, 5e-4 * np.eye(3), atol=1e-14)
        clamped = thermochemoelastic(sample, {0: _steel()}, [0.0],
                                     supports=dict.fromkeys(ALL, "clamped"), **common)  # fmt: skip
        s = clamped.stress[0].reshape(-1, 6)
        np.testing.assert_allclose(s[:, :3], -3.0 * k * 5e-4, rtol=1e-10)
        assert np.abs(s[:, 3:]).max() < 1e-8 * 3.0 * k * 5e-4


def test_periodic_laminate_expands_as_its_layers():
    """A periodic laminate of two phases, free of mean stress: the in-plane strain is the
    mean of the expansions weighted by the biaxial moduli E / (1 - nu), the strain across
    is that of free layers in their common in-plane strain; with the mean strain imposed
    to zero, the mean stress of the clamped layers."""
    labels = np.zeros((2, 2, 8), np.uint8)
    labels[:, :, 5:] = 1
    image = VoxelImage(labels, spacing=(1.0, 1.0, 1.0), periodic=True)
    mats = {0: ThermoChemoMaterial(young=3.0, poisson=0.35, expansion=6e-5),
            1: ThermoChemoMaterial(young=70.0, poisson=0.22, expansion=8e-6)}  # fmt: skip
    dt = 20.0
    out = thermochemoelastic(image, mats, [0.0], reference_temperature=300.0,
                             initial_temperature=300.0 + dt)  # fmt: skip
    f = np.array([5 / 8, 3 / 8])
    e = np.array([3.0, 70.0])
    nu = np.array([0.35, 0.22])
    a = np.array([6e-5, 8e-6]) * dt
    m = e / (1.0 - nu)
    plane = np.sum(f * m * a) / np.sum(f * m)
    across = np.sum(f * (a - 2.0 * nu / e * m * (plane - a)))
    np.testing.assert_allclose(out.mean_strain[0][:3], [plane, plane, across], rtol=1e-10)
    assert np.abs(out.mean_strain[0][3:]).max() < 1e-15
    np.testing.assert_allclose(out.history["mean_stress"][0], 0.0, atol=1e-12)
    # the layers held in their plane (the in-plane mean strains imposed to zero), free
    # across: sigma_33 = 0 in each layer, sigma_11 = sigma_22 = -E alpha dT / (1 - nu)
    fixed = thermochemoelastic(image, mats, [0.0], reference_temperature=300.0,
                               initial_temperature=300.0 + dt,
                               mean_strain={0: 0.0, 1: 0.0, 5: 0.0})  # fmt: skip
    s = fixed.history["mean_stress"][0]
    plane_stress = np.sum(f * (-m * a))
    np.testing.assert_allclose(s[:2], plane_stress, rtol=1e-9)
    np.testing.assert_allclose(s[2:], 0.0, atol=1e-9 * abs(plane_stress))
    column = fixed.stress[0][0, 0]  # the voxels of a column through the layers
    expected = np.where(np.arange(8) < 5, -m[0] * a[0], -m[1] * a[1])
    np.testing.assert_allclose(column[:, 0], expected, rtol=1e-9)
    np.testing.assert_allclose(column[:, 1], expected, rtol=1e-9)
    np.testing.assert_allclose(column[:, 2], 0.0, atol=1e-9 * abs(plane_stress))


def test_bilayer_bends_as_timoshenko():
    """A free bilayer strip whose upper layer expands (nu = 0): its curvature within 1.5 %
    of Timoshenko's (four voxels per layer), towards its lower face."""
    labels = np.zeros((60, 2, 8), np.uint8)
    labels[:, :, 4:] = 1
    image = VoxelImage(labels, spacing=(1.0, 1.0, 1.0))
    mats = {0: ThermoChemoMaterial(young=1.0, poisson=0.0),
            1: ThermoChemoMaterial(young=3.0, poisson=0.0, expansion=1e-3)}  # fmt: skip
    out = thermochemoelastic(image, mats, [0.0], reference_temperature=300.0,
                             initial_temperature=301.0)  # fmt: skip
    kappa = out.curvature(0, span=(10, 50))
    assert kappa > 0
    assert kappa == pytest.approx(timoshenko_curvature(1e-3, 4.0, 4.0, 1.0, 3.0), rel=0.015)


def test_strip_swelling_from_its_top_face():
    """A strip that a species enters by its upper face (the lower one impermeable): its
    curvature in time is that of the beam whose eigenstrain is Crank's profile of the half
    slab, within 3 % (twelve voxels through the thickness)."""
    nx, nz = 48, 12
    image = VoxelImage(np.zeros((nx, 1, nz), np.uint8), spacing=(1.0, 1.0, 1.0))
    m = ThermoChemoMaterial(young=1.0, poisson=0.0, diffusivity=2.0, swelling=1e-3)
    times = np.array([2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    out = thermochemoelastic(image, {0: m}, times, activity={"z+": 1.0}, steps=30,
                             scheme="trbdf2")  # fmt: skip
    z = np.linspace(0.0, nz, 801)
    for k, t in enumerate(times):
        exact = beam_curvature(z - 0.5 * nz, 1.0, 1e-3 * slab_profile(z, t, 2.0 * nz, 2.0))
        assert out.curvature(k, span=(8, 40)) == pytest.approx(exact, rel=0.03)


def test_adiabatic_reaction_conserves_its_energy():
    """An adiabatic sample where a species is consumed and heats it (with Arrhenius' law of
    the rate): ``C (T - T0) = H (c0 - c)`` at every time (both schemes); without the
    activation, ``c = c0 exp(-k t)``."""
    image = VoxelImage(np.zeros((2, 2, 2), np.uint8), spacing=(1.0, 1.0, 1.0))
    common = {"young": 1.0, "poisson": 0.2, "conductivity": 1.0, "heat_capacity": 2.0,
              "diffusivity": 1.0, "solubility": 3.0, "reaction_rate": 0.5,
              "reaction_heat": 10.0}  # fmt: skip
    times = np.array([0.5, 1.0, 2.0, 4.0])
    hot = ThermoChemoMaterial(reaction_activation=40e3, **common)
    for scheme in ("euler", "trbdf2"):
        out = thermochemoelastic(image, {0: hot}, times, reference_temperature=300.0,
                                 initial_activity=1.0, steps=100, scheme=scheme)  # fmt: skip
        heat = 2.0 * (out.history["mean_temperature"] - 300.0)
        np.testing.assert_allclose(heat, 10.0 * (3.0 - out.history["uptake"]), atol=1e-8)
        assert out.info["max_iterations"] > 1
    out = thermochemoelastic(image, {0: ThermoChemoMaterial(**common)}, times,
                             initial_activity=1.0, steps=100, scheme="trbdf2")  # fmt: skip
    np.testing.assert_allclose(out.history["uptake"] / 3.0, np.exp(-0.5 * times), atol=2e-5)


def test_arrhenius_scales_the_times():
    """A diffusion at 330 K with an activation energy is that at the reference temperature
    with the times scaled by the Arrhenius factor, exactly."""
    image = VoxelImage(np.zeros((1, 1, 20), np.uint8), spacing=(0.05,) * 3)
    m = ThermoChemoMaterial(young=1.0, poisson=0.2, diffusivity=1.0, diffusion_activation=50e3)
    f = float(arrhenius(50e3, 330.0, 300.0))
    assert f == pytest.approx(np.exp(-50e3 / R_GAS * (1 / 330.0 - 1 / 300.0)))
    times = np.array([0.01, 0.05, 0.1])
    faces = {"z-": 1.0, "z+": 1.0}
    hot = thermochemoelastic(image, {0: m}, times, reference_temperature=300.0,
                             initial_temperature=330.0, activity=faces, steps=20)  # fmt: skip
    ref = thermochemoelastic(image, {0: m}, times * f, reference_temperature=300.0,
                             activity=faces, steps=20)  # fmt: skip
    np.testing.assert_allclose(hot.history["uptake"], ref.history["uptake"], atol=1e-13)
    np.testing.assert_allclose(hot.history["uptake"], slab_uptake(f * times, 1.0, 1.0), atol=0.02)


def test_species_drawn_by_the_stress_reach_boltzmann():
    """A species of partial molar volume Omega in a closed bilayer bent by its thermal
    expansion: it gathers where the hydrostatic stress is tensile, ``u ~ exp(Omega
    sigma_h / R T)`` exactly at equilibrium (Scharfetter–Gummel fluxes), its content kept;
    with its own swelling the stress depends on it, and the fixed point still reaches
    Boltzmann's distribution of its own stress."""
    labels = np.zeros((24, 1, 8), np.uint8)
    labels[:, :, 4:] = 1
    image = VoxelImage(labels, spacing=(1e-3,) * 3)
    omega = 2e-6

    def mats(swelling):
        return {
            0: ThermoChemoMaterial(young=200e9, poisson=0.3, diffusivity=1e-6,
                                   molar_volume=omega, swelling=swelling, conductivity=1.0),
            1: ThermoChemoMaterial(young=100e9, poisson=0.3, expansion=2e-5, diffusivity=1e-6,
                                   molar_volume=omega, swelling=swelling, conductivity=1.0),
        }  # fmt: skip

    for swelling, tol in ((0.0, 1e-12), (0.02, 1e-7)):
        out = thermochemoelastic(image, mats(swelling), [50.0, 2000.0, 2e5],
                                 reference_temperature=300.0, initial_temperature=400.0,
                                 initial_activity=1.0, steps=20)  # fmt: skip
        for k in (1, 2):
            log_u = np.log(out.activity[k])
            excess = log_u - omega * out.hydrostatic[k] / (R_GAS * 400.0)
            assert np.ptp(log_u) > 0.03  # (the swelling of the species evens it out)
            assert np.ptp(excess) < tol * np.ptp(log_u)
        np.testing.assert_allclose(out.history["uptake"], 1.0, rtol=1e-10)
        # the strong coupling (the swelling of the species stresses the strip by
        # E beta c ~ 2 GPa, Omega sigma / R T ~ 1) takes several relaxed iterations
        assert (out.info["max_iterations"] > 5) == bool(swelling)
    assert bernoulli(np.array([-0.3]))[0] == pytest.approx(np.exp(0.3) * bernoulli(0.3))


def test_meshes_against_slabs_and_conservation():
    """On a mesh of linear tetrahedra: the uptake of a slab against Crank's series within
    1 % (sixteen elements through it); a closed bilayer keeps the content of its species
    while the stress draws it."""
    mesh = box_mesh(n=(2, 2, 16), upper=(0.2, 0.2, 1.0), periodic=False)
    m = ThermoChemoMaterial(young=1.0, poisson=0.3, diffusivity=1.0)
    times = np.array([0.01, 0.05, 0.1])
    out = thermochemoelastic(mesh, {0: m}, times, activity={"z-": 1.0, "z+": 1.0}, steps=30)
    np.testing.assert_allclose(out.history["uptake"], slab_uptake(times, 1.0, 1.0), rtol=0.01)
    mesh = box_mesh(n=(8, 1, 4), upper=(8.0, 1.0, 4.0), periodic=False,
                    phase=lambda c: (c[:, 2] > 2.0).astype(int))  # fmt: skip
    mats = {
        0: ThermoChemoMaterial(young=200e9, poisson=0.3, diffusivity=1e-6, molar_volume=2e-6),
        1: ThermoChemoMaterial(young=100e9, poisson=0.3, expansion=2e-5, diffusivity=1e-6,
                               molar_volume=2e-6),
    }  # fmt: skip
    out = thermochemoelastic(mesh, mats, [1e5, 1e6], reference_temperature=300.0,
                             initial_temperature=400.0, initial_activity=1.0, steps=10)  # fmt: skip
    np.testing.assert_allclose(out.history["uptake"], 1.0, rtol=1e-10)
    tension = out.hydrostatic[-1] > 0
    concentration = out.concentration[-1]
    assert concentration[tension].mean() > concentration[~tension].mean()


def test_supports_leave_no_reaction():
    """The supports that remove the rigid motions of a free body are statically
    determinate: the forces they would exert under a self-equilibrated eigenstrain
    vanish."""
    labels = np.zeros((10, 3, 4), np.uint8)
    labels[:, :, 2:] = 1
    image = VoxelImage(labels, spacing=(1.0, 1.0, 1.0))
    mats = {0: ThermoChemoMaterial(young=1.0, poisson=0.3),
            1: ThermoChemoMaterial(young=2.0, poisson=0.2, expansion=1e-3)}  # fmt: skip
    problem = ThermoChemoElastic(image, mats, reference_temperature=300.0,
                                 initial_temperature=310.0)  # fmt: skip
    mech = problem.space.mechanics
    eigen = problem.eigenstrain(np.full(problem.space.n, 310.0), np.zeros(problem.space.n))
    u, _, _ = mech.solve(eigen)
    residual = mech.matrix @ u.ravel() - mech.load(eigen[mech.solid])
    assert mech.supported.size == 6
    assert np.abs(residual[mech.supported]).max() < 1e-10 * np.abs(mech.load(eigen)).max()


def test_symmetry_planes_and_materials_of_the_package():
    """A quarter of a heated plate on two symmetry planes is the free plate; a Material of
    the package gives its coefficients."""
    m = ThermoChemoMaterial.from_material(
        Material(young=200e9, poisson=0.3, expansion=1e-5, conductivity=50.0),
        heat_capacity=3.6e6,
    )
    assert m.heat_capacity == 3.6e6 and m.thermal_strain()[0] == 1e-5
    image = VoxelImage(np.zeros((4, 4, 2), np.uint8), spacing=(1e-3,) * 3)
    out = thermochemoelastic(image, {0: m}, [0.0], reference_temperature=300.0,
                             initial_temperature=320.0,
                             supports={"x-": "symmetry", "y-": "symmetry"})  # fmt: skip
    d = out.displacement[0]
    np.testing.assert_allclose(d[0, :, :, 0], 0.0, atol=1e-18)
    np.testing.assert_allclose(d[:, 0, :, 1], 0.0, atol=1e-18)
    np.testing.assert_allclose(d[-1, 0, 0, 0], 4e-3 * 2e-4, rtol=1e-10)
    assert np.abs(out.stress).max() < 1e-3


def test_time_series_in_xdmf(tmp_path):
    """The fields at the output times in an XDMF time series: one grid per time, the raw
    arrays of their sizes (voxels and meshes)."""
    image = VoxelImage(np.zeros((3, 2, 4), np.uint8), spacing=(0.5,) * 3)
    m = ThermoChemoMaterial(young=1.0, poisson=0.3, diffusivity=1.0, swelling=0.01)
    out = thermochemoelastic(image, {0: m}, [0.1, 0.4], activity={"z+": 1.0}, steps=5)
    path = out.write_xdmf(tmp_path / "fields.xdmf")
    text = path.read_text()
    assert text.count('<Grid Name="t') == 2 and 'CollectionType="Temporal"' in text
    assert (tmp_path / "fields_displacement_1.raw").stat().st_size == 4 * 3 * 5 * 3 * 8
    assert (tmp_path / "fields_concentration_0.raw").stat().st_size == 3 * 2 * 4 * 8
    raw = np.fromfile(tmp_path / "fields_concentration_1.raw", "<f8").reshape(4, 2, 3)
    np.testing.assert_allclose(raw.transpose(2, 1, 0), out.concentration[1])
    mesh = box_mesh(n=(2, 2, 2), periodic=False)
    out = thermochemoelastic(mesh, {0: m}, [0.1], activity={"z+": 1.0}, steps=5)
    text = out.write_xdmf(tmp_path / "mesh.xdmf").read_text()
    assert re.search(r'TopologyType="Tetrahedron" NumberOfElements="48"', text)


def test_errors():
    image = VoxelImage(np.zeros((2, 2, 2), np.uint8), periodic=True)
    m = ThermoChemoMaterial(young=1.0, poisson=0.3)
    with pytest.raises(ValueError, match="faces"):
        ThermoChemoElastic(image, {0: m}, temperature={"w+": 1.0})
    with pytest.raises(ValueError, match="no material"):
        ThermoChemoElastic(image, {1: m})
    with pytest.raises(ValueError, match="periodic"):
        ThermoChemoElastic(image, {0: m}, periodic=(0,), temperature={"x-": 300.0})
    with pytest.raises(ValueError, match="young"):
        ThermoChemoMaterial(conductivity=1.0)
    with pytest.raises(ValueError, match="times"):
        thermochemoelastic(image, {0: m}, [1.0, 0.5])
    with pytest.raises(ValueError, match="scheme"):
        thermochemoelastic(image, {0: m}, [1.0], scheme="rk4")
    with pytest.raises(ValueError, match="supports"):
        ThermoChemoElastic(VoxelImage(np.zeros((2, 2, 2), np.uint8)), {0: m},
                           supports={"x-": "glued"})  # fmt: skip
    with pytest.raises(ValueError, match="periodic"):
        ThermoChemoElastic(box_mesh(n=(1, 1, 1)), {0: m}, periodic=True)
