"""Strongly coupled linear homogenization on voxel images: piezoelectric, piezomagnetic,
magneto-electro-elastic, magnetoelectric and thermoelectric laws.

The references are exact: laminates (the layers of any symmetric law), the uncoupled
limits (the elastic and dielectric problems of the FFT solver), the dense system of a
small grid, uniform fields (the theorems of uniform thermal strains and of a uniform
Seebeck coefficient) and Levin's relation.
"""

import dataclasses

import numpy as np
import pytest

import morphogen as mg
from helpers import backus
from morphogen.homogenization import Material, bounds, homogenize, make_problem
from morphogen.homogenization.apparent import COUPLED, conductivity_tensor, stiffness_tensor
from morphogen.homogenization.coupled import (
    COUPLINGS,
    CoupledProblem,
    Coupling,
    _apply,
    coupled_entries,
    coupled_law,
    coupled_problem,
    coupled_tensors,
    laminate,
    layout_sizes,
    poled_ceramic,
    rotate_piezo,
    split_law,
)

# Barium titanate and cobalt ferrite, poled along x3 (Li & Dunn 1998; SI units)
BATIO3 = dataclasses.replace(
    Material.hexagonal(166e9, 77e9, 78e9, 162e9, 43e9),
    permittivity=(11.2e-9, 11.2e-9, 12.6e-9),
    permeability=(5e-6, 5e-6, 10e-6),
    coupling={"piezoelectric": {"e31": -4.4, "e33": 18.6, "e15": 11.6}},
)
COFE2O4 = dataclasses.replace(
    Material.hexagonal(286e9, 173e9, 170.5e9, 269.5e9, 45.3e9),
    permittivity=(0.08e-9, 0.08e-9, 0.093e-9),
    permeability=(590e-6, 590e-6, 157e-6),
    coupling={"piezomagnetic": {"q31": 580.3, "q33": 699.7, "q15": 550.0}},
)
EPOXY = Material(young=4e9, poisson=0.35, permittivity=3.7e-11, permeability=1.2566e-6)


def layers(n=8, thickness=3, plane=(3, 2)):
    data = np.zeros((*plane, n), dtype=np.uint8)
    data[:, :, :thickness] = 1
    return mg.VoxelImage(data, spacing=(1.0, 0.7, 0.5), periodic=True)


def random_image(shape=(6, 5, 4), seed=0, fraction=0.5):
    rng = np.random.default_rng(seed)
    return mg.VoxelImage((rng.random(shape) < fraction).astype(np.uint8), (1.0, 0.8, 1.1), True)


def relative(a, b):
    """The largest gap of two symmetric laws relative to their diagonal entries (the
    blocks of a coupled law differ by twenty orders of magnitude)."""
    d = np.sqrt(np.abs(np.diag(b)))
    return float(np.max(np.abs(a - b) / np.outer(d, d)))


# -- the laws ------------------------------------------------------------------------------


def test_the_laminate_of_uncoupled_laws_is_backus_and_maxwell():
    c0, c1 = BATIO3.matrix("elastic"), COFE2O4.matrix("elastic")
    f = [0.3, 0.7]
    assert np.allclose(laminate([c0, c1], f, ("displacement",)), backus([c0, c1], f), rtol=1e-12)
    k0, k1 = np.diag([1.0, 2.0, 3.0]), np.diag([4.0, 5.0, 6.0])
    k = laminate([k0, k1], f, ("potential",))
    assert np.allclose(np.diag(k), [0.3 + 0.7 * 4, 0.3 * 2 + 0.7 * 5, 1 / (0.3 / 3 + 0.7 / 6)])
    with pytest.raises(ValueError, match="fractions"):
        laminate([k0, k1], [0.5, 0.6], ("potential",))


def test_the_laminate_of_a_piezoelectric_stack_in_its_thickness_mode():
    """Independent of the general formula: with no strain in the plane and no field along
    it, each layer is ``(sigma, D) = M (eps, E)`` with ``M = [[c, -e], [e, kappa]]`` (33
    components); ``sigma`` and ``D`` are uniform across the layers and the means of
    ``eps`` and ``E`` are those of the compliances ``M^-1``."""
    f = np.array([0.35, 0.65])
    mats = (BATIO3, EPOXY)
    laws = [coupled_law(m, "piezoelectric")[0] for m in mats]
    star = laminate(laws, f, COUPLINGS["piezoelectric"])
    m_inv = np.zeros((2, 2))
    for fi, law in zip(f, laws, strict=True):
        c, e, kappa = law[2, 2], law[8, 2], -law[8, 8]
        m_inv += fi * np.linalg.inv(np.array([[c, -e], [e, kappa]]))
    m = np.linalg.inv(m_inv)
    assert star[2, 2] == pytest.approx(m[0, 0], rel=1e-12)
    assert star[8, 2] == pytest.approx(m[1, 0], rel=1e-12)
    assert -star[8, 8] == pytest.approx(m[1, 1], rel=1e-12)


def test_laws_of_the_couplings():
    for coupling, layout in COUPLINGS.items():
        nz = layout_sizes(layout)[1]
        thermo = dataclasses.replace(BATIO3, electrical_conductivity=1e3, conductivity=2.0,
                                     coupling={"seebeck": 2e-4})  # fmt: skip
        m = thermo if coupling == "thermoelectric" else BATIO3
        law, lam = coupled_law(m, coupling)
        assert law.shape == (nz, nz) and lam.shape == (nz,) and np.allclose(law, law.T)
    law, _ = coupled_law(BATIO3, "piezoelectric")
    assert np.allclose(law[:6, :6], BATIO3.matrix("elastic"))
    assert np.allclose(law[6:, 6:], -np.diag([11.2e-9, 11.2e-9, 12.6e-9]))
    assert law[8, 2] == 18.6 and law[8, 0] == -4.4 and law[6, 4] == law[7, 3] == 11.6
    # the saddle of the electric enthalpy: six positive and three negative eigenvalues
    # (of the law scaled by its diagonal: its blocks differ by twenty orders of magnitude)
    d = 1.0 / np.sqrt(np.abs(np.diag(law)))
    assert np.sum(np.linalg.eigvalsh(law * np.outer(d, d)) > 0) == 6
    with pytest.raises(ValueError, match="dielectric"):
        coupled_law(Material(young=1.0, poisson=0.3), "piezoelectric")
    with pytest.raises(ValueError, match="coupling"):
        coupled_law(BATIO3, "flexoelectric")
    # a pore filled by air: no stiffness, a permittivity
    air, _ = coupled_law(Material(permittivity=8.854e-12), "piezoelectric")
    assert not air[:6].any() and air[8, 8] == -8.854e-12


def test_a_piezoelectric_matrix_turns_with_its_crystal():
    e = poled_ceramic(-4.4, 18.6, 11.6)
    # 6mm: invariant by any rotation about its axis
    assert np.allclose(rotate_piezo(e, [37.0, 0.0, 0.0]), e, atol=1e-12)
    # poled along x2 after a turn of the crystal by 90 degrees about x1
    turned = rotate_piezo(e, [0.0, 90.0, 0.0])
    g = mg.orientations.euler_to_matrix([0.0, 90.0, 0.0])[0]
    axis = g.T @ np.array([0.0, 0.0, 1.0])  # the axis of the crystal in the sample
    k = int(np.argmax(np.abs(axis)))
    assert k == 1 and turned[k, k] == pytest.approx(18.6 * np.sign(axis[k]))
    assert np.allclose(rotate_piezo(e, np.array([[0.0, 0.0, 0.0], [0.0, 90.0, 0.0]]))[1], turned)
    c = Coupling(piezoelectric=e, pyroelectric=[0.0, 0.0, 2.0], seebeck=3.0)
    assert np.allclose(c.matrix("pyroelectric", [0.0, 90.0, 0.0]), 2.0 * axis)
    assert np.allclose(c.matrix("seebeck"), 3.0 * np.eye(3))
    assert np.allclose(c.matrix("piezomagnetic"), 0.0)
    assert Coupling.from_config(c.to_config()) == c
    with pytest.raises(ValueError, match="3 x 6"):
        Coupling(piezoelectric=np.zeros((6, 3)))
    with pytest.raises(ValueError, match="unknown"):
        Coupling.from_config({"flexoelectric": 1.0})
    with pytest.raises(ValueError, match="e15"):
        Coupling.from_config({"piezoelectric": {"e31": 1.0, "e33": 2.0}})


# -- the solver against exact solutions ----------------------------------------------------


@pytest.mark.parametrize("coupling", sorted(COUPLINGS))
def test_layers_have_the_exact_law_of_a_laminate(coupling):
    if coupling == "thermoelectric":
        mats = {0: Material(electrical_conductivity=2e4, conductivity=1.5,
                            coupling={"seebeck": -1.8e-4}),
                1: Material(electrical_conductivity=(9e4, 9e4, 3e4), conductivity=4.0,
                            coupling={"seebeck": (2.1e-4, 2.1e-4, 1.2e-4)})}  # fmt: skip
    else:
        mats = {0: COFE2O4, 1: BATIO3}
    img = layers()
    problem = coupled_problem(img, mats, coupling, tol=1e-12, temperature=350.0)
    law = problem.tensor()
    laws = [coupled_law(mats[p], coupling, temperature=350.0)[0] for p in (0, 1)]
    exact = laminate(laws, [5 / 8, 3 / 8], COUPLINGS[coupling])
    assert relative(law, exact) < 1e-9
    assert problem.definite == (coupling in ("thermoelectric", "magnetoelectric"))


def test_a_product_property_of_a_laminate():
    """Neither phase is magnetoelectric; the stack is, through the strains they share."""
    out = coupled_tensors(layers(), {0: COFE2O4, 1: BATIO3}, "magnetoelectroelastic", tol=1e-12)
    for m in (COFE2O4, BATIO3):
        own = split_law(coupled_law(m, "magnetoelectroelastic")[0], "magnetoelectroelastic")
        assert not own["magnetoelectric"].any()
    alpha = out["magnetoelectric"]
    assert abs(out["alpha11"]) > 1e-12 and abs(out["alpha33"]) > 1e-12
    assert np.allclose(alpha, np.diag(np.diag(alpha)), atol=1e-6 * np.abs(alpha).max())
    # Cauchy-Schwarz of the coupled law: alpha^2 < kappa mu
    assert out["alpha33"] ** 2 < out["permittivity"][2, 2] * out["permeability"][2, 2]
    assert out["e33"] > 0 and out["q33"] > 0 and out["dm33"] > 0


def test_uncoupled_phases_give_the_elastic_and_dielectric_problems():
    img = random_image()
    mats = {0: dataclasses.replace(EPOXY, coupling=None),
            1: dataclasses.replace(BATIO3, coupling=None)}  # fmt: skip
    out = coupled_tensors(img, mats, "piezoelectric", tol=1e-11)
    c, _ = stiffness_tensor(make_problem(img, mats, "elastic", tol=1e-11))
    k, _ = conductivity_tensor(make_problem(img, mats, "dielectric", tol=1e-11))
    assert np.allclose(out["stiffness"], c, rtol=1e-8, atol=1e-8 * c.max())
    assert np.allclose(out["permittivity"], k, rtol=1e-8, atol=1e-8 * k.max())
    assert np.abs(out["piezoelectric"]).max() < 1e-9 * np.sqrt(c.max() * k.max())


def test_solution_matches_the_dense_system():
    img = random_image(shape=(3, 4, 3), seed=4)
    problem = coupled_problem(img, {0: EPOXY, 1: BATIO3}, "piezoelectric", tol=1e-13)
    nx, ny, nz = problem.shape
    nd, ncomp = problem.nd, problem.nz
    n = nd * nx * ny * nz
    k = np.zeros((n, n))
    unit, col = np.zeros(n), np.empty(n)
    for j in range(n):
        unit[j] = 1.0
        _apply(nx, ny, nz, nd, problem.mat, problem.element_matrices, unit, col)
        k[:, j] = col
        unit[j] = 0.0
    assert np.allclose(k, k.T, atol=1e-13 * np.abs(k).max())
    eig = np.linalg.eigvalsh(k)
    assert eig.min() < -1e-3 * eig.max()  # a saddle: MINRES, not conjugate gradients
    from morphogen.homogenization import kernels
    from morphogen.homogenization.coupled import _sums

    z = np.array([0.3, -0.2, 0.1, 0.05, -0.04, 0.07, 2e8, -1e8, 3e8]) * 1e-3
    zs = z / problem.scale_z
    aff = problem._affine(zs)
    loads = np.ascontiguousarray(problem.element_matrices @ aff)
    dof = np.arange(n, dtype=np.int64)
    rhs = kernels.element_load(nx, ny, nz, True, problem.mat, dof, nd, loads, n)
    w = np.linalg.lstsq(k, rhs, rcond=None)[0]
    sums = _sums(nx, ny, nz, nd, problem.mat, problem.stat, 2, problem._laws, problem._eigen,
                 0.0, w, aff, problem.b_gauss, problem.element.weight)  # fmt: skip
    flux = sums.sum(axis=0)[ncomp : 2 * ncomp] / problem.box_volume / problem.scale_z
    sol = problem.solve(z, fields=True)
    assert np.allclose(sol.flux, flux, rtol=1e-8, atol=1e-10 * np.abs(flux).max())
    # the means of the fields are those of the solution, phase by phase
    assert np.allclose(sol.fields["gradient"].mean(axis=(0, 1, 2)), z, rtol=1e-9, atol=1e-12)
    assert np.allclose(sol.fields["flux"].mean(axis=(0, 1, 2)), sol.flux, rtol=1e-9)
    total = sum(ph["fraction"] * ph["flux"] for ph in sol.phases.values())
    assert np.allclose(total, sol.flux, rtol=1e-12)
    assert sol.energy == pytest.approx(float(sol.flux @ z), rel=1e-7)  # Hill's lemma
    assert sol.fields["fluctuation"].shape == (3, 4, 3, 4)


def test_a_one_three_composite_of_a_ceramic_in_a_polymer():
    """Fibres of a poled ceramic along x3 in a polymer: the stack of the laminate along the
    fibres gives the exact e33, kappa33 and C33 of any section when the phases have the
    same Poisson contraction; here the known trends: d33 close to the ceramic's, the
    hydrostatic and thickness coefficients above it (Smith & Auld 1991)."""
    n = 12
    data = np.zeros((n, n, 1), dtype=np.uint8)
    data[3:9, 3:9] = 1  # 25 % of square fibres
    img = mg.VoxelImage(data, 1.0, periodic=True)
    out = coupled_tensors(img, {0: EPOXY, 1: BATIO3}, "piezoelectric", tol=1e-10)
    ceramic = split_law(coupled_law(BATIO3, "piezoelectric")[0], "piezoelectric")
    assert 0.6 * ceramic["d33"] < out["d33"] < ceramic["d33"]
    assert out["g33"] > 3.0 * ceramic["g33"] and out["g_h"] > 3.0 * ceramic["g_h"]
    assert ceramic["k_t"] < out["k_t"] < ceramic["k33"]
    assert out["e33"] < 0.25 * ceramic["e33"] * 1.2
    # the square section: tetragonal tensors
    e = out["piezoelectric"]
    assert e[2, 0] == pytest.approx(e[2, 1], rel=1e-8) and e[0, 4] == pytest.approx(e[1, 3])
    assert out["iterations"] > 0 and out["residual"] < 1e-10


def test_uniform_thermal_strains_and_levins_relation():
    img = random_image(shape=(6, 6, 6), seed=2)
    # phases that expand alike and are not pyroelectric at zero stress (p = -e alpha at
    # constant strain) expand freely: no stress, no displacement, the same expansion
    alpha = 3e-5
    e = poled_ceramic(-4.4, 18.6, 11.6)
    free = -e @ np.array([alpha, alpha, alpha, 0.0, 0.0, 0.0])
    poled = {"piezoelectric": e, "pyroelectric": free}
    same = {0: dataclasses.replace(EPOXY, expansion=alpha),
            1: dataclasses.replace(BATIO3, expansion=alpha, coupling=poled)}  # fmt: skip
    out = coupled_tensors(img, same, "piezoelectric", tol=1e-11)
    assert np.allclose(out["thermal_expansion"], alpha * np.eye(3), rtol=1e-7, atol=1e-12)
    strain = np.array([alpha, alpha, alpha, 0.0, 0.0, 0.0])
    assert np.allclose(out["pyroelectric"], -out["piezoelectric"] @ strain, rtol=1e-6,
                       atol=1e-9 * 18.6 * alpha)  # fmt: skip
    # without that primary coefficient, the thermal strains of the ceramic charge it (the
    # secondary pyroelectricity) and the fields stress the composite: not the free expansion
    charged = {0: same[0], 1: dataclasses.replace(BATIO3, expansion=alpha)}
    other = coupled_tensors(img, charged, "piezoelectric", tol=1e-11)
    assert np.abs(other["thermal_expansion"] - alpha * np.eye(3)).max() > 1e-3 * alpha
    # layers: the exact thermal flux of the laminate, with a pyroelectric phase
    hot = {0: dataclasses.replace(EPOXY, expansion=6e-5),
           1: dataclasses.replace(BATIO3, expansion=(8e-6, 8e-6, 5e-6),
                                  coupling={"piezoelectric": {"e31": -4.4, "e33": 18.6,
                                                              "e15": 11.6},
                                            "pyroelectric": [0.0, 0.0, -2e-4]})}  # fmt: skip
    problem = coupled_problem(layers(), hot, "piezoelectric", tol=1e-12)
    pairs = [coupled_law(hot[p], "piezoelectric") for p in (0, 1)]
    _, star = laminate([p[0] for p in pairs], [5 / 8, 3 / 8], COUPLINGS["piezoelectric"],
                       eigen=[p[1] for p in pairs])  # fmt: skip
    lam = problem.eigen_flux()
    scale = np.where(np.arange(9) < 6, 1e11 * 1e-5, 1e-4)
    assert np.allclose(lam / scale, star / scale, rtol=1e-8, atol=1e-10)
    # Levin: the expansion of two isotropic phases in a cubic cell from its bulk modulus
    cube = np.zeros((8, 8, 8), dtype=np.uint8)
    cube[2:6, 2:6, 2:6] = 1
    two = {0: Material(young=3.0, poisson=0.35, permittivity=1.0, expansion=7e-5),
           1: Material(young=70.0, poisson=0.22, permittivity=5.0, expansion=5e-6)}  # fmt: skip
    res = coupled_tensors(mg.VoxelImage(cube, 1.0, periodic=True), two, "piezoelectric", tol=1e-11)
    c = res["stiffness"]
    bulk = (c[:3, :3].sum()) / 9.0
    from morphogen.homogenization.bounds import levin_expansion

    levin = levin_expansion([1 - 0.125, 0.125], [two[0], two[1]], bulk)
    assert np.allclose(res["thermal_expansion"], levin * np.eye(3), rtol=1e-6, atol=1e-11)


def test_thermoelectric_composites():
    img = random_image(shape=(6, 6, 6), seed=3)
    t0 = 400.0
    a = Material(electrical_conductivity=5e4, conductivity=1.2, coupling={"seebeck": 2e-4})
    b = Material(electrical_conductivity=2e5, conductivity=6.0, coupling={"seebeck": 2e-4})
    out = coupled_tensors(img, {0: a, 1: b}, "thermoelectric", temperature=t0, tol=1e-11)
    # a uniform Seebeck coefficient is that of the composite, whatever its conductivities
    assert np.allclose(out["seebeck"], 2e-4 * np.eye(3), rtol=1e-8, atol=1e-13)
    # and its conductivities are those of the uncoupled problems
    sigma, _ = conductivity_tensor(make_problem(img, {0: a, 1: b}, "electrical", tol=1e-11))
    k, _ = conductivity_tensor(make_problem(img, {0: a, 1: b}, "thermal", tol=1e-11))
    assert np.allclose(out["electrical_conductivity"], sigma, rtol=1e-7)
    assert np.allclose(out["conductivity"], k, rtol=1e-7)
    # Bergman & Levy (1991): the figure of merit of a composite never exceeds the best phase
    c = Material(electrical_conductivity=8e4, conductivity=2.0, coupling={"seebeck": -1.5e-4})
    mixed = coupled_tensors(img, {0: a, 1: c}, "thermoelectric", temperature=t0, tol=1e-10)
    zt = [m.electrical_conductivity * 4e-8 * 0 for m in (a, c)]
    zt = [5e4 * (2e-4) ** 2 * t0 / 1.2, 8e4 * (1.5e-4) ** 2 * t0 / 2.0]
    for i in (1, 2, 3):
        assert 0.0 <= mixed[f"zt{i}"] <= max(zt) * (1 + 1e-9)
        assert mixed[f"power_factor{i}"] >= 0.0
    own = split_law(coupled_law(a, "thermoelectric", temperature=t0)[0], "thermoelectric",
                    temperature=t0)  # fmt: skip
    assert own["zt1"] == pytest.approx(zt[0], rel=1e-12)
    assert np.allclose(own["conductivity"], 1.2 * np.eye(3), rtol=1e-12)
    with pytest.raises(ValueError, match="temperature"):
        coupled_law(a, "thermoelectric", temperature=0.0)


def test_crystals_turned_by_their_grains():
    """Two grains of the ceramic in layers, the second poled along -x3 (turned by 180
    degrees about x1): the piezoelectric couplings cancel, the stiffness of the crystal,
    which is even, is unchanged in its 33 component up to the stiffening of the fields."""
    data = np.ones((2, 2, 8), dtype=np.uint8)
    grains = np.zeros((2, 2, 8), dtype=np.int64)
    grains[:, :, 4:] = 1
    img = mg.VoxelImage(data, 1.0, periodic=True)
    euler = np.array([[0.0, 0.0, 0.0], [0.0, 180.0, 0.0]])
    out = coupled_tensors(img, {1: BATIO3}, "piezoelectric", grains=grains, orientations=euler,
                          tol=1e-12)  # fmt: skip
    assert np.abs(out["piezoelectric"]).max() < 1e-8 * 18.6
    up = coupled_law(BATIO3, "piezoelectric")[0]
    down = coupled_law(BATIO3, "piezoelectric", euler[1])[0]
    assert np.allclose(down[6:, :6], -up[6:, :6], atol=1e-12)
    exact = laminate([up, down], [0.5, 0.5], COUPLINGS["piezoelectric"])
    assert relative(out["tensor"], exact) < 1e-9
    # open circuit between the opposed layers: the stiffness of the short circuit, stiffened
    assert out["stiffness"][2, 2] > BATIO3.matrix("elastic")[2, 2]


def test_problems_check_their_laws():
    img = random_image()
    with pytest.raises(ValueError, match="symmetric"):
        CoupledProblem(img, np.array([[[1.0, 2.0, 0], [0, 1, 0], [0, 0, 1]]]), ("potential",),
                       np.zeros(img.data.size, dtype=int))  # fmt: skip
    indefinite = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="definite"):
        CoupledProblem(img, indefinite[None], ("potential",), np.zeros(img.data.size, dtype=int))
    with pytest.raises(ValueError, match="layout"):
        CoupledProblem(img, np.eye(3)[None], ("temperature",), np.zeros(img.data.size, dtype=int))
    with pytest.raises(ValueError, match="phase 0"):
        coupled_problem(img, {0: Material(young=1.0, poisson=0.3), 1: BATIO3}, "piezoelectric")
    with pytest.raises(ValueError, match="no material"):
        coupled_problem(img, {1: BATIO3}, "piezoelectric")
    # a negative block of a single potential is solved as well (minus a conduction)
    problem = CoupledProblem(img, np.array([-np.eye(3), -3.0 * np.eye(3)]), ("potential",),
                             img.data.ravel().astype(int), tol=1e-10)  # fmt: skip
    k, _ = conductivity_tensor(
        make_problem(
            img,
            {0: Material(conductivity=1.0), 1: Material(conductivity=3.0)},
            "thermal",
            tol=1e-10,
        )
    )
    assert np.allclose(problem.tensor(), -k, rtol=1e-7)
    # voids carry nothing
    porous = coupled_tensors(img, {0: Material(void=True), 1: BATIO3}, "piezoelectric")
    assert porous["stiffness"][2, 2] < BATIO3.matrix("elastic")[2, 2] * img.data.mean()
    empty = coupled_problem(img, {0: Material(void=True), 1: Material(void=True)}, "piezoelectric")
    assert empty.empty and not empty.tensor().any()


# -- homogenize, configurations and the command line ---------------------------------------


def test_homogenize_gives_the_tensors_of_the_couplings():
    img = layers()
    out = homogenize(img, {0: EPOXY, 1: BATIO3}, properties=["piezoelectric", "stiffness"],
                     tol=1e-11)  # fmt: skip
    piezo = out["piezoelectric.periodic"]
    exact = laminate([coupled_law(m, "piezoelectric")[0] for m in (EPOXY, BATIO3)],
                     [5 / 8, 3 / 8], COUPLINGS["piezoelectric"])  # fmt: skip
    assert relative(piezo["tensor"], exact) < 1e-8
    assert set(COUPLED) == set(COUPLINGS)
    d = piezo["d"]
    assert np.allclose(d @ piezo["stiffness"], piezo["piezoelectric"], rtol=1e-10, atol=1e-12)
    entries = coupled_entries(piezo)
    assert entries["e33"] == piezo["e33"] == piezo["piezoelectric"][2, 2]
    assert entries["C12"] == piezo["stiffness"][0, 1] and "C21" not in entries
    assert entries["e15"] == piezo["piezoelectric"][0, 4] and entries["eps33"] > 0
    from helpers import box_mesh

    with pytest.raises(ValueError, match="voxel images"):
        homogenize(box_mesh(), {0: BATIO3}, properties=["piezoelectric"])


def test_coupled_properties_in_a_configuration(tmp_path, capsys):
    import json

    import yaml

    from morphogen import cli, pipeline
    from morphogen.config import ConfigError, load_config

    materials = {0: EPOXY.to_config(), 1: BATIO3.to_config()}
    data = {
        "model": {"type": "random_laminate", "thickness": 0.9, "spacing": 1.1},
        "domain": {"size": 6.0, "periodic": True},
        "seed": 3,
        "realizations": 2,
        "voxels": {"shape": [2, 2, 24]},
        "homogenization": {"method": "fft", "materials": materials,
                           "properties": ["piezoelectric"], "tolerance": 1e-10},
        "outputs": {"directory": str(tmp_path / "out"), "name": "stack"},
    }  # fmt: skip
    outdir = pipeline.run(load_config(data))
    rows = pipeline.read_rows(outdir / "results.csv")
    assert len(rows) == 2
    for row in rows:
        f1 = row["fraction_1"]
        laws = [coupled_law(m, "piezoelectric")[0] for m in (EPOXY, BATIO3)]
        exact = split_law(laminate(laws, [1 - f1, f1], COUPLINGS["piezoelectric"]),
                          "piezoelectric")  # fmt: skip
        for name in ("e33", "d33", "k_t", "g_h"):
            assert row[f"piezoelectric.periodic.{name}"] == pytest.approx(exact[name], rel=1e-6)
        assert row["piezoelectric.periodic.C33"] == pytest.approx(exact["stiffness"][2, 2])
        assert row["piezoelectric.periodic.eps11"] == pytest.approx(exact["permittivity"][0, 0])
    data["mesh"] = {"size": 0.5}
    data["homogenization"]["method"] = "fe"
    with pytest.raises(ConfigError, match="voxel"):
        load_config(data)
    # the command line, the materials in a file
    image = layers()
    path = tmp_path / "img.vti"
    mg.io.save_image(image, path)
    (tmp_path / "materials.yaml").write_text(yaml.safe_dump(materials), encoding="utf-8")
    args = ["homogenize", str(path), "--materials", str(tmp_path / "materials.yaml"),
            "--property", "piezoelectric", "--tol", "1e-10"]  # fmt: skip
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "piezoelectric.periodic" in text and "k_t" in text and "permittivity:" in text
    assert cli.main([*args, "--json"]) == 0
    res = json.loads(capsys.readouterr().out)["piezoelectric.periodic"]
    assert "Voigt" in res["notation"] and np.array(res["piezoelectric"]).shape == (3, 6)


def test_rve_statistics_of_the_constants_of_a_coupling(tmp_path):
    """The scalar results of a coupling are properties of RVE studies, by their names:
    ``piezoelectric.e33`` (``.periodic``); their statistics are those of the columns of the
    results."""
    import json

    from morphogen import pipeline
    from morphogen.config import ConfigError, load_config
    from morphogen.homogenization.coupled import entry_names

    names = entry_names("piezoelectric")
    assert {"C11", "C66", "e33", "e15", "d33", "eps11", "eps_s33", "k_t", "g_h", "A11",
            "beta3", "p3"} <= set(names)  # fmt: skip
    assert {"alpha11", "alpha33", "q33", "mu11"} <= set(entry_names("magnetoelectroelastic"))
    assert {"zt3", "s11", "sigma33", "k11"} <= set(entry_names("thermoelectric"))
    with pytest.raises(ValueError, match="coupling"):
        entry_names("piezo")
    data = {
        "model": {"type": "random_laminate", "thickness": 0.9, "spacing": 1.1},
        "domain": {"periodic": True},
        "seed": 3,
        "voxels": {"voxel_size": 1.0},
        "homogenization": {"method": "fft", "tolerance": 1e-10, "properties": ["piezoelectric"],
                           "materials": {0: EPOXY.to_config(), 1: BATIO3.to_config()}},
        "rve": {"sizes": [4.0, 6.0], "realizations": [3, 2], "bootstrap": 0,
                "properties": ["piezoelectric.e33", "piezoelectric.k_t.periodic",
                               "volume_fraction"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "stack"},
    }  # fmt: skip
    cfg = load_config(data)
    assert cfg.rve.properties == ["piezoelectric.e33", "piezoelectric.k_t.periodic",
                                  "volume_fraction"]  # fmt: skip
    outdir = pipeline.run(cfg)
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    rows = pipeline.read_rows(outdir / "results.csv")
    for name in ("e33", "k_t"):
        stats = report["properties"][f"piezoelectric.{name}.periodic"]["statistics"]
        assert [s["n"] for s in stats] == [3, 2]
        for s, size in zip(stats, (4.0, 6.0), strict=True):
            column = [r[f"piezoelectric.periodic.{name}"] for r in rows
                      if r["volume"] == pytest.approx(size**3)]  # fmt: skip
            assert s["mean"] == pytest.approx(np.mean(column), rel=1e-12)
    assert "volume_fraction" in report["properties"]
    for bad, match in (
        ("piezoelectric.e99", "unknown result 'e99'"),
        ("piezoelectric.e33.kubc", "periodic conditions"),
        ("piezomagnetic.q33", "unknown value"),
    ):
        with pytest.raises(ConfigError, match=match):
            load_config({**data, "rve": {**data["rve"], "properties": [bad]}})


def test_bounds_are_unchanged_by_the_couplings():
    b = bounds({0: 0.5, 1: 0.5}, {0: EPOXY, 1: dataclasses.replace(EPOXY, young=8e9)})
    assert {"bulk_modulus", "shear_modulus", "permittivity", "permeability"} <= set(b)
