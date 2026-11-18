"""Anisotropic crystals: rotations, orientation averages and bounds, and the
homogenization of polycrystals by finite elements and by the FFT solver."""

import numpy as np
import pytest

import morphogen as mg
from morphogen.homogenization import Material, bounds, homogenize
from morphogen.homogenization.bounds import cubic_shear_bounds
from morphogen.homogenization.materials import (
    material_table,
    rotate_stiffness,
    tensor4_to_voigt,
    voigt_average,
    voigt_to_tensor4,
)
from morphogen.mesh.conforming import conforming_mesh, has_mmg
from morphogen.orientations import random_orientations

needs_mmg = pytest.mark.skipif(not has_mmg(), reason="MMG (pymmg) is not installed")

# copper (GPa), titanium (GPa; conductivities in the basal plane and along c)
CU = Material.cubic(168.4, 121.4, 75.4, conductivity=400.0)
TI = Material.hexagonal(162.4, 92.0, 69.0, 180.7, 46.7, conductivity=(20.0, 30.0))
MANDEL = np.diag([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])


def test_rotations_keep_the_invariants():
    rng = np.random.default_rng(3)
    euler = random_orientations(rng, 20)
    for mat in (CU, TI):
        d = np.array(mat.stiffness)
        rotated = mat.matrix("elastic", euler)
        assert rotated.shape == (20, 6, 6)
        assert np.allclose(rotated, np.swapaxes(rotated, 1, 2))
        eig = np.linalg.eigvalsh(MANDEL @ d @ MANDEL)
        for r in rotated:
            assert np.allclose(np.linalg.eigvalsh(MANDEL @ r @ MANDEL), eig)
            assert np.allclose(voigt_average(r), voigt_average(d))
        k = mat.matrix("thermal", euler)
        assert np.allclose(np.linalg.eigvalsh(k), np.linalg.eigvalsh(mat.matrix("thermal")))
    # Voigt matrices and fourth-order tensors
    d = np.array(TI.stiffness)
    assert np.array_equal(tensor4_to_voigt(voigt_to_tensor4(d)), d)
    # a cubic crystal turned by 45 degrees about [001]
    r = rotate_stiffness(np.array(CU.stiffness), np.array([45.0, 0.0, 0.0]))
    assert r[0, 0] == pytest.approx((168.4 + 121.4) / 2 + 75.4)
    assert r[0, 1] == pytest.approx((168.4 + 121.4) / 2 - 75.4)
    assert r[5, 5] == pytest.approx((168.4 - 121.4) / 2)
    assert r[2, 2] == pytest.approx(168.4) and r[3, 3] == pytest.approx(75.4)
    # the c axis of a hexagonal crystal (third row of g) along the sample axis x
    k = TI.matrix("thermal", np.array([90.0, 90.0, 0.0]))
    assert np.allclose(np.diag(k), [30.0, 20.0, 20.0])


def test_averages_and_bounds_of_copper():
    """Voigt, Reuss and Hashin-Shtrikman shear moduli of copper polycrystals (GPa)."""
    k, gv = CU.moduli("voigt")
    kr, gr = CU.moduli("reuss")
    assert k == pytest.approx(137.0667, abs=1e-4) and kr == pytest.approx(k)
    assert gv == pytest.approx(54.64) and gr == pytest.approx(40.034, abs=1e-3)
    lo, hi = cubic_shear_bounds(168.4, 121.4, 75.4)
    assert lo == pytest.approx(45.96, abs=0.01) and hi == pytest.approx(49.45, abs=0.01)
    b = bounds({1: 1.0}, {1: CU})
    assert b["shear_modulus"]["hs_lower"] == pytest.approx(lo)
    assert b["bulk_modulus"]["hs_lower"] == b["bulk_modulus"]["hs_upper"] == pytest.approx(k)
    # isotropic materials: the averages are the moduli
    steel = Material(young=210.0, poisson=0.3)
    iso = Material.anisotropic(steel.matrix("elastic"))
    assert np.allclose(iso.moduli("voigt"), steel.moduli()) and np.allclose(
        iso.moduli("reuss"), steel.moduli()
    )
    hs = bounds({1: 1.0}, {1: TI})["conductivity"]
    assert 20.0 < hs["reuss"] < hs["hs_lower"] <= hs["hs_upper"] < hs["voigt"] < 30.0


def test_configuration_of_crystals():
    for mat in (CU, TI, Material.anisotropic(np.array(TI.stiffness), conductivity=[1, 2, 3])):
        assert Material.from_config(mat.to_config()) == mat
    cu = Material.from_config({"cubic": [168.4, 121.4, 75.4], "conductivity": 400})
    assert cu == CU and cu.symmetry == "cubic" and not cu.isotropic
    for bad in (
        {"cubic": [1.0, 2.0]},
        {"cubic": [168.4, 121.4, 75.4], "young": 1.0},
        {"stiffness": np.eye(5).tolist()},
        {"stiffness": (-np.eye(6)).tolist()},
        {"young": 1.0, "poisson": 0.3, "conductivity": [1, 2, 3]},
    ):
        with pytest.raises(ValueError):
            Material.from_config(bad)
    with pytest.raises(ValueError, match="isotropic"):
        _ = CU.lame


def test_material_table_per_grain():
    euler = random_orientations(np.random.default_rng(0), 4)
    iso = Material(young=1.0, poisson=0.25, conductivity=2.0)
    table = material_table(
        np.array([1, 1, 0, 1, 1]),
        {0: iso, 1: CU},
        "elastic",
        grains=np.array([2, 3, 0, 2, 1]),
        orientations=euler,
    )
    assert table.phases == (0, 1) and not table.isotropic
    assert table.mat[0] == table.mat[3] and len(set(table.mat.tolist())) == 4
    assert np.allclose(table.matrices[table.mat[1]], CU.matrix("elastic", euler[3]))
    # thermal: copper conducts isotropically, one matrix per phase
    thermal = material_table(np.array([1, 1, 0]), {0: iso, 1: CU}, "thermal")
    assert thermal.isotropic and len(thermal.matrices) == 2


def single_crystal(model_seed=1, n=6):
    """Periodic Voronoi tessellation whose grains all have the same orientation."""
    tess = mg.models.Voronoi(n_cells=n).generate(
        mg.Domain.cube(1.0, periodic=True), seed=model_seed
    )
    tess.orientations = np.tile([30.0, 50.0, 70.0], (tess.n_cells, 1))
    return tess


def test_fft_single_crystal_and_polycrystal():
    tess = single_crystal()
    shape = (12, 12, 12)
    img, lab = tess.voxelize(shape), tess.labels(shape)
    res = homogenize(
        img,
        {1: TI},
        properties=["stiffness", "conductivity_tensor"],
        grains=lab,
        orientations=tess.orientations,
    )
    assert np.allclose(res["stiffness.periodic"]["tensor"], TI.matrix("elastic", [30, 50, 70]))
    assert np.allclose(
        res["conductivity_tensor.periodic"]["tensor"], TI.matrix("thermal", [30, 50, 70])
    )
    # a copper polycrystal: exact bulk modulus, shear modulus between the bounds
    poly = mg.models.Voronoi(n_cells=30, crystal_orientations=True).generate(
        mg.Domain.cube(1.0, periodic=True), seed=5
    )
    shape = (32, 32, 32)
    res = homogenize(
        poly.voxelize(shape), {1: CU}, grains=poly.labels(shape), orientations=poly.orientations
    )
    k = res["bulk_modulus.periodic"]
    assert k.value == pytest.approx((168.4 + 2 * 121.4) / 3, rel=1e-9)
    g = res["shear_modulus.periodic"].value
    assert CU.moduli("reuss")[1] < g < CU.moduli("voigt")[1]
    assert 0.95 * 45.96 < g < 1.05 * 49.45


@needs_mmg
def test_fe_polycrystal_of_copper():
    tess = mg.models.Voronoi(n_cells=30, crystal_orientations=True).generate(
        mg.Domain.cube(1.0, periodic=True), seed=5
    )
    mesh = conforming_mesh(tess, size=0.1)
    res = homogenize(
        mesh,
        {1: CU},
        bcs=["kubc", "periodic", "subc"],
        properties=["bulk_modulus", "shear_modulus"],
    )
    k = (168.4 + 2 * 121.4) / 3
    for bc in ("kubc", "periodic", "subc"):
        assert res[f"bulk_modulus.{bc}"].value == pytest.approx(k, rel=1e-7)
    g = {bc: res[f"shear_modulus.{bc}"].value for bc in ("kubc", "periodic", "subc")}
    assert CU.moduli("reuss")[1] < g["subc"] < g["periodic"] < g["kubc"] < CU.moduli("voigt")[1]
    # the same polycrystal on voxels
    shape = (32, 32, 32)
    fft = homogenize(
        tess.voxelize(shape),
        {1: CU},
        properties=["shear_modulus"],
        grains=tess.labels(shape),
        orientations=tess.orientations,
    )
    assert fft["shear_modulus.periodic"].value == pytest.approx(g["periodic"], rel=0.03)
    # a single crystal: the tensor of the crystal, by finite elements too
    one = single_crystal()
    mesh = conforming_mesh(one, size=0.25)
    res = homogenize(mesh, {1: TI}, properties=["stiffness", "conductivity_tensor"])
    assert np.allclose(res["stiffness.periodic"]["tensor"], TI.matrix("elastic", [30, 50, 70]))
    with pytest.raises(ValueError, match="mesh"):
        homogenize(mesh, {1: TI}, grains=np.zeros(3), orientations=np.zeros((1, 3)))
