"""Viscoelasticity, hyperelasticity and damage (:mod:`morphogen.homogenization.laws`).

Material points against closed forms (the Prony series of a relaxation, the harmonic
regime, the uniaxial tensions of incompressible rubbers, the softening curve and the
fracture energy of a damaged bar), the tangents against finite differences of the
stresses, the stresses against those of the energies; then the same laws in the solvers:
laminates whose responses are exact, the material point recovered by finite elements and
by the FFT solver, the dissipation of a crack band independent of the voxels, a crack in
a hyperelastic plate and the identification of a Prony series.
"""

import numpy as np
import pytest
from scipy.special import erf

import morphogen as mg
from helpers import box_mesh
from morphogen.homogenization import (
    Material,
    NonlinearFEProblem,
    NonlinearFFTProblem,
    homogenize,
    simulate_point,
)
from morphogen.homogenization.laws import (
    EQUIVALENT_STRAINS,
    FITTED,
    HYPERELASTIC_MODELS,
    SOFTENINGS,
    SPLITS,
    TESTS,
    VOLUMETRIC,
    Damage,
    Hyperelasticity,
    Viscoelasticity,
    analytic_model,
    fit_hyperelasticity,
)
from morphogen.homogenization.materials import bulk_shear, isotropic_matrix, lame

E0, NU0 = 3.0, 0.3
K0, G0 = bulk_shear(E0, NU0)
PRONY = {"shear": [{"g": 0.5, "tau": 1.0}, {"g": 0.3, "tau": 10.0}],
         "bulk": [{"k": 0.2, "tau": 2.0}]}  # fmt: skip
VISCO = Material(young=E0, poisson=NU0, viscoelasticity=PRONY)
#: Every component of the strain prescribed, one of them varying
FULL = {c: [0.0, 0.0, 0.0] for c in ("11", "22", "33", "23", "31", "12")}

#: Parameters of the hyperelastic models of the tests (their shear modulus about 1): those
#: of an isochoric energy, whose tests of incompressible rubbers have closed forms
RUBBERS = {
    "neo_hookean": {"mu": 1.2},
    "mooney_rivlin": {"c10": 0.4, "c01": 0.15},
    "polynomial": {"c10": 0.45, "c01": 0.1, "c20": 0.02, "c11": -0.01},
    "yeoh": {"c10": 0.5, "c20": -0.04, "c30": 0.01},
    "arruda_boyce": {"mu": 1.0, "lock": 2.5},
    "eight_chain": {"mu": 1.0, "lock": 2.5},
    "gent": {"mu": 1.0, "lock": 20.0},
    "van_der_waals": {"mu": 1.0, "lock": 6.0, "a": 0.2, "beta": 0.1},
    "fung": {"mu": 1.0, "b": 0.8},
    "ogden": {"terms": [[0.63, 1.3], [0.0012, 5.0], [-0.01, -2.0]]},
}
#: and the others: fibres, compressible foams
OTHERS = {
    "holzapfel": {"mu": 1.0, "k1": 2.0, "k2": 1.5, "kappa": 0.1,
                  "fibres": [[1, 0, 0], [0.6, 0.8, 0.0]]},
    "blatz_ko": {"mu": 1.0, "f": 0.3},
    "hyperfoam": {"terms": [[0.8, 2.0], [0.2, -1.5]]},
}  # fmt: skip


def rubber(model, bulk=20.0, **options):
    parameters = {**RUBBERS, **OTHERS}[model]
    return Material(hyperelasticity={"model": model, "bulk": bulk, **parameters, **options})


# -- viscoelasticity -----------------------------------------------------------------------


def test_relaxation_follows_the_prony_series():
    """A strain applied at once and held: the stresses are the relaxation moduli."""
    visco = VISCO.viscoelasticity
    hold = {"times": [0, 1e-9, 20.0], "increments": [1, 40]}
    shear = simulate_point(VISCO, {**hold, "strain": {**FULL, "12": [0, 0.01, 0.01]}})
    t = shear.times[1:] - 1e-9
    k, g = visco.relaxation(t, K0, G0)
    assert np.allclose(shear.stress[1:, 5], g * 0.01, rtol=1e-9)
    assert shear.stress[1, 5] == pytest.approx(G0 * 0.01, rel=1e-8)  # the instantaneous modulus
    # a uniaxial strain: the longitudinal modulus K + 4 G / 3, the lateral one K - 2 G / 3
    oed = simulate_point(VISCO, {**hold, "strain": {**FULL, "11": [0, 0.01, 0.01]}})
    assert np.allclose(oed.stress[1:, 0], (k + 4.0 * g / 3.0) * 0.01, rtol=1e-9)
    assert np.allclose(oed.stress[1:, 1], (k - 2.0 * g / 3.0) * 0.01, rtol=1e-9)
    k_inf, g_inf = visco.long_term(K0, G0)
    assert k_inf == pytest.approx(0.8 * K0) and g_inf == pytest.approx(0.2 * G0)
    assert g[-1] == pytest.approx(g_inf + 0.3 * G0 * np.exp(-2.0), rel=1e-6)


def test_the_integration_is_exact_for_a_constant_strain_rate():
    """Whatever the steps: one step and fifty give the stress of the closed form, ``s = 2
    rate (G_inf t + sum_i g_i G0 tau_i (1 - exp(-t / tau_i)))``."""
    rate, t = 0.02, 3.0
    exact = rate * t * 0.2 * G0 + rate * G0 * (
        0.5 * 1.0 * (1 - np.exp(-t / 1.0)) + 0.3 * 10.0 * (1 - np.exp(-t / 10.0))
    )
    for n in (1, 7, 50):
        path = {"times": [0, t], "increments": n, "strain": {**FULL, "12": [0, rate * t]}}
        r = simulate_point(VISCO, {**path, "strain": {k: v[:2] for k, v in path["strain"].items()}})
        assert r.stress[-1, 5] == pytest.approx(exact, rel=1e-10)


def test_creep_reaches_the_relaxed_compliance():
    m = Material(
        young=E0, poisson=NU0, viscoelasticity={"shear": [[0.6, 0.5]], "bulk": [[0.3, 0.2]]}
    )
    r = simulate_point(m, {"times": [0, 1e-9, 40.0], "increments": [1, 160],
                           "stress": {"11": [0, 1.0, 1.0]}})  # fmt: skip
    assert r.strain[1, 0] == pytest.approx(1.0 / E0, rel=1e-6)
    k_inf, g_inf = m.viscoelasticity.long_term(K0, G0)
    young_inf = 9.0 * k_inf * g_inf / (3.0 * k_inf + g_inf)
    assert r.strain[-1, 0] == pytest.approx(1.0 / young_inf, rel=1e-6)
    assert np.all(np.diff(r.strain[1:, 0]) > -1e-9)
    assert np.abs(r.stress[1:, 1:]).max() < 1e-9  # a uniaxial stress throughout


def test_harmonic_regime_gives_the_complex_modulus():
    """A sinusoidal shear followed in time: the amplitude and the phase of the stress are
    those of the complex modulus of the Prony series."""
    visco = VISCO.viscoelasticity
    omega, cycles, per = 1.3, 12, 120
    t = np.linspace(0.0, cycles * 2 * np.pi / omega, cycles * per + 1)
    strain = 0.01 * np.sin(omega * t)
    strains = {c: (strain.tolist() if c == "12" else [0.0] * t.size) for c in FULL}
    path = {"times": t.tolist(), "increments": 1, "strain": strains}
    r = simulate_point(VISCO, path)
    last = slice(-per - 1, None)  # the last cycle: the transient has decayed
    a = np.column_stack([np.sin(omega * t[last]), np.cos(omega * t[last])])
    storage, loss = np.linalg.lstsq(a, r.stress[last, 5], rcond=None)[0] / 0.01
    _, g = visco.complex_moduli(omega, K0, G0)
    assert storage == pytest.approx(g.real, rel=2e-3)
    assert loss == pytest.approx(g.imag, rel=2e-3)
    # the limits: relaxed at rest, instantaneous at high frequencies; one Maxwell branch
    k, g = visco.complex_moduli(np.array([1e-9, 1e9]), K0, G0)
    assert np.allclose(g.real, [0.2 * G0, G0], rtol=1e-6) and np.allclose(g.imag, 0.0, atol=1e-6)
    assert np.allclose(k.real, [0.8 * K0, K0], rtol=1e-6)
    one = Viscoelasticity(shear=[(0.4, 2.0)])
    wt = 0.7 * 2.0
    _, g = one.complex_moduli(0.7, 1.0, 5.0)
    assert g.real == pytest.approx(5.0 * (0.6 + 0.4 * wt**2 / (1 + wt**2)))
    assert g.imag == pytest.approx(5.0 * 0.4 * wt / (1 + wt**2))


def test_times_shift_with_the_temperature():
    wlf = Material(young=E0, poisson=NU0, viscoelasticity={"shear": [[0.5, 1.0]],
                                                           "wlf": [8.86, 101.6]})  # fmt: skip
    visco = wlf.viscoelasticity
    a_t = visco.shift(20.0)
    assert a_t == pytest.approx(10.0 ** (-8.86 * 20.0 / 121.6)) and visco.shift(0.0) == 1.0
    hot = {"times": [0, 1e-9, 0.2], "increments": [1, 10], "temperature": [0, 20.0, 20.0],
           "strain": {**FULL, "12": [0, 0.01, 0.01]}}  # fmt: skip
    r = simulate_point(wlf, hot)
    t = r.times[1:] - 1e-9
    exact = G0 * 0.01 * (0.5 + 0.5 * np.exp(-t / a_t))  # the times of the reference, shifted
    assert np.allclose(r.stress[1:, 5], exact, rtol=1e-7)
    arrhenius = Viscoelasticity(shear=[(0.5, 1.0)], arrhenius=5000.0, reference_temperature=300.0)
    assert arrhenius.shift(10.0) == pytest.approx(np.exp(5000.0 * (1 / 310.0 - 1 / 300.0)))
    assert Viscoelasticity.from_config(arrhenius.to_config()) == arrhenius
    # a thermal expansion: the free expansion of the heated point, without stress
    free = Material(young=E0, poisson=NU0, expansion=1e-4, viscoelasticity=PRONY)
    r = simulate_point(free, {"times": [0, 1.0], "increments": 2, "temperature": [0, 50.0],
                              "stress": {"11": [0, 0]}})  # fmt: skip
    assert np.allclose(r.strain[-1, :3], 5e-3, rtol=1e-9) and np.abs(r.stress).max() < 1e-12


def test_viscoelasticity_is_checked():
    assert Viscoelasticity.from_config(VISCO.viscoelasticity.to_config()) == VISCO.viscoelasticity
    assert Material.from_config(VISCO.to_config()) == VISCO
    assert VISCO.nonlinear and VISCO.analytic
    for bad, match in (
        ({"shear": [[0.7, 1.0], [0.4, 2.0]]}, "less than 1"),
        ({"shear": [[0.5, -1.0]]}, "positive"),
        ({}, "at least one"),
        ({"shear": [[0.5, 1.0]], "wlf": [1.0, 2.0], "arrhenius": 3.0}, "not both"),
        ({"shear": [[0.5, 1.0]], "creep": 1}, "unknown"),
        ({"shear": [{"g": 0.5}]}, "g and tau"),
    ):
        with pytest.raises(ValueError, match=match):
            Viscoelasticity.from_config(bad)
    with pytest.raises(ValueError, match="isotropic"):
        Material(stiffness=np.eye(6), viscoelasticity=PRONY)
    with pytest.raises(ValueError, match="one behaviour"):
        Material(young=1.0, poisson=0.3, viscoelasticity=PRONY, plasticity={"yield": 1.0})
    with pytest.raises(ValueError, match="void"):
        Material(void=True, viscoelasticity=PRONY)


# -- hyperelasticity -----------------------------------------------------------------------


@pytest.mark.parametrize("volumetric", VOLUMETRIC)
@pytest.mark.parametrize("model", sorted(HYPERELASTIC_MODELS))
def test_hyperelastic_stresses_and_tangents_derive_from_the_energy(model, volumetric):
    if model == "saint_venant_kirchhoff":
        m = Material(young=2.6, poisson=0.3, hyperelasticity=model)
    elif model in ("blatz_ko", "hyperfoam"):  # compressible laws: no volumetric energy
        if volumetric != "quadratic":
            with pytest.raises(ValueError, match="compressible"):
                rubber(model, bulk=5.0, volumetric=volumetric)
            return
        m = rubber(model, bulk=5.0)
    else:
        m = rubber(model, bulk=5.0, volumetric=volumetric)
    law = analytic_model(m, finite=True)
    rng = np.random.default_rng(7)
    e = 0.12 * rng.standard_normal((6, 6))
    e[0] = 0.0  # the natural state
    e[1, 3:] = 0.0  # two equal stretches along the axes
    e[1, 1] = e[1, 0]
    e[2] = [0.2, 0.2, 0.2, 0.0, 0.0, 0.0]  # three equal stretches
    none = np.zeros((6, 0))
    s, c, _ = law.step(e, e, none, 1.0)
    h = 1e-6
    # (the fibres of Holzapfel begin to carry where the stretches are equal: no second
    # derivative at the natural state nor under three equal stretches)
    smooth = [1, 3, 4, 5] if model == "holzapfel" else slice(None)
    for k in range(6):
        d = np.zeros(6)
        d[k] = h
        ds = (law.energy(e + d) - law.energy(e - d)) / (2 * h)
        assert np.allclose(s[smooth, k], ds[smooth], rtol=1e-6, atol=1e-7)
        dc = (law.step(e + d, e, none, 1.0)[0] - law.step(e - d, e, none, 1.0)[0]) / (2 * h)
        assert np.allclose(c[smooth, :, k], dc[smooth], rtol=1e-5, atol=2e-6)
    assert np.allclose(c, np.swapaxes(c, 1, 2), rtol=1e-12, atol=1e-12)
    # at the natural state: no stress, the elasticity of the material
    assert not s[0].any()
    assert np.allclose(c[0], isotropic_matrix(*lame(m.young, m.poisson)), rtol=1e-12, atol=1e-13)


def _slopes(model, i1, i2):
    """The derivatives of the isochoric energies of RUBBERS with respect to their two
    invariants, written out."""
    x, y = i1 - 3.0, i2 - 3.0
    lock = RUBBERS["arruda_boyce"]["lock"]
    series = (0.5, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750)
    zero = 0.0 * x
    if model == "neo_hookean":
        return 0.6 + zero, zero
    if model == "mooney_rivlin":
        return 0.4 + zero, 0.15 + zero
    if model == "polynomial":
        return 0.45 + 2 * 0.02 * x - 0.01 * y, 0.1 - 0.01 * x
    if model == "yeoh":
        return 0.5 + 2 * -0.04 * x + 3 * 0.01 * x**2, zero
    if model == "arruda_boyce":
        return sum(i * a * i1 ** (i - 1) / lock ** (2 * i - 2)
                   for i, a in enumerate(series, start=1)), zero  # fmt: skip
    if model == "eight_chain":  # mu / (6 x) of Cohen's inverse Langevin function of x
        u = i1 / (3 * lock**2)
        return (3 - u) / (1 - u) / 6, zero
    if model == "gent":
        return 0.5 * 20.0 / (20.0 - x), zero
    if model == "van_der_waals":
        mix = 0.9 * x + 0.1 * y
        eta = np.sqrt(mix / (6.0**2 - 3.0))
        slope = 0.5 / (1 - eta) - 0.5 * 0.2 * np.sqrt(mix / 2)
        return 0.9 * slope, 0.1 * slope
    assert model == "fung"
    return 0.5 * np.exp(0.8 * x), zero


def _nominal(model, mode, lam):
    """The nominal stresses of the tests of an incompressible rubber of RUBBERS."""
    if model == "ogden":
        terms = RUBBERS[model]["terms"]
        if mode == "shear":  # the principal stretch t: t - 1/t = the amount of shear
            t = lam / 2 + np.sqrt(1 + lam**2 / 4)
            return sum(m * (t**a - t**-a) / (t + 1 / t) for m, a in terms)
        power = {"uniaxial": -0.5, "equibiaxial": -2.0, "planar": -1.0}[mode]
        return sum(m * (lam ** (a - 1) - lam ** (power * a - 1)) for m, a in terms)
    if mode == "uniaxial":
        first, second = _slopes(model, lam**2 + 2 / lam, 2 * lam + lam**-2)
        return 2 * (lam - lam**-2) * (first + second / lam)
    if mode == "equibiaxial":
        first, second = _slopes(model, 2 * lam**2 + lam**-4, lam**4 + 2 * lam**-2)
        return 2 * (lam - lam**-5) * (first + lam**2 * second)
    if mode == "planar":
        first, second = _slopes(model, lam**2 + lam**-2 + 1, lam**2 + lam**-2 + 1)
        return 2 * (lam - lam**-3) * (first + second)
    first, second = _slopes(model, 3 + lam**2, 3 + lam**2)
    return 2 * lam * (first + second)


@pytest.mark.parametrize("model", sorted(RUBBERS))
def test_uniaxial_tension_of_incompressible_rubbers(model):
    """Stretched to twice their length with a bulk modulus of 1e6 shear moduli: the
    nominal stresses of the incompressible solids."""
    r = simulate_point(rubber(model, bulk=1e6), {"times": [0, 1.0], "increments": 20,
                                                 "strain": {"11": [0, 1.0]}},
                       kinematics="finite")  # fmt: skip
    lam = 1.0 + r.strain[1:, 0]
    assert np.allclose(r.stress[1:, 0], _nominal(model, "uniaxial", lam), rtol=5e-5)
    assert np.abs(r.stress[:, 1:]).max() < 1e-6  # the lateral faces are free
    volume = lam * (1.0 + r.strain[1:, 1]) * (1.0 + r.strain[1:, 2])
    assert np.allclose(volume, 1.0, atol=1e-4)
    # the Cauchy stress and the logarithmic strain of the outputs
    assert np.allclose(r.cauchy[1:, 0], r.stress[1:, 0] * lam / volume, rtol=1e-9)
    assert np.allclose(r.log_strain[1:, 0], np.log(lam), rtol=1e-9)


@pytest.mark.parametrize("model", sorted(RUBBERS))
def test_the_tests_of_rubbers(model):
    """Uniaxial, equibiaxial and planar tension and simple shear: the closed forms of the
    incompressible solid, and the law with its bulk modulus, whose lateral stretches
    cancel the lateral stresses, close to them for 1e7 shear moduli."""
    assert model in FITTED and TESTS == ("uniaxial", "equibiaxial", "planar", "shear")
    law = rubber(model, bulk=1e7).hyperelasticity.law()
    stretches = np.array([0.7, 0.9, 1.0, 1.3, 2.0])
    for mode in TESTS:
        x = np.array([-0.5, 0.0, 0.4, 1.2]) if mode == "shear" else stretches
        exact = _nominal(model, mode, x)
        assert law.nominal_stress(mode, x, incompressible=True) == pytest.approx(exact, rel=1e-12)
        assert law.nominal_stress(mode, x) == pytest.approx(exact, rel=2e-4, abs=2e-4)
    # equibiaxial tension is uniaxial compression of the square of the inverse stretch
    two = law.nominal_stress("equibiaxial", stretches, incompressible=True)
    one = law.nominal_stress("uniaxial", stretches**-2.0, incompressible=True)
    assert two == pytest.approx(-one / stretches**3, rel=1e-12, abs=1e-12)
    with pytest.raises(ValueError, match="mode"):
        law.nominal_stress("torsion", stretches)


def test_compressible_foams():
    """The foam of Blatz and Ko (f = 0, Poisson's ratio 1/4): its lateral stretch is the
    stretch to the power -1/4, its nominal stress mu (l^-1/2 - l^-3); a hyperfoam without
    lateral contraction: sum_p 2 mu_p / alpha_p (l^(alpha_p - 1) - 1 / l)."""
    foam = Material(hyperelasticity={"model": "blatz_ko", "mu": 1.5, "bulk": 2.5})
    assert foam.poisson == pytest.approx(0.25) and foam.young == pytest.approx(3.75)
    law = foam.hyperelasticity.law()
    x = np.array([0.6, 0.9, 1.0, 1.5, 3.0])
    assert law.nominal_stress("uniaxial", x) == pytest.approx(1.5 * (x**-0.5 - x**-3), abs=1e-12)
    r = simulate_point(foam, {"times": [0, 1.0], "increments": 10, "strain": {"11": [0, 1.0]}},
                       kinematics="finite")  # fmt: skip
    lam = 1.0 + r.strain[1:, 0]
    assert r.stress[1:, 0] == pytest.approx(1.5 * (lam**-0.5 - lam**-3), rel=1e-8)
    assert 1.0 + r.strain[1:, 1] == pytest.approx(lam**-0.25, rel=1e-8)
    terms = [[0.8, 2.0], [0.2, -1.5]]
    crushable = Material(hyperelasticity={"model": "hyperfoam", "terms": terms, "bulk": 2 / 3})
    assert crushable.poisson == pytest.approx(0.0, abs=1e-12)
    assert crushable.young == pytest.approx(2.0)  # the shear modulus: the sum of the mu_p
    law = crushable.hyperelasticity.law()
    exact = sum(2 * m / a * (x ** (a - 1) - 1 / x) for m, a in terms)
    assert law.nominal_stress("uniaxial", x) == pytest.approx(exact, abs=1e-12)
    r = simulate_point(crushable, {"times": [0, 1.0], "increments": 10,
                                   "strain": {"11": [0, -0.5]}}, kinematics="finite")  # fmt: skip
    lam = 1.0 + r.strain[1:, 0]
    assert r.stress[1:, 0] == pytest.approx(
        sum(2 * m / a * (lam ** (a - 1) - 1 / lam) for m, a in terms), rel=1e-9
    )
    assert np.abs(r.strain[:, 1:3]).max() < 1e-12  # no lateral strain
    with pytest.raises(ValueError, match="compressible"):
        law.nominal_stress("uniaxial", x, incompressible=True)


def test_fibres_of_holzapfel():
    """A family of fibres in a neo-Hookean matrix, incompressible: along the fibres the
    nominal stress mu (l - l^-2) + 2 k1 l (l^2 - 1) exp(k2 (l^2 - 1)^2); the fibres carry
    no compression, nor a tension across them."""
    k1, k2 = 2.0, 1.5
    tissue = Material(hyperelasticity={"model": "holzapfel", "mu": 1.0, "k1": k1, "k2": k2,
                                       "fibres": [[2.0, 0, 0]], "bulk": 1e6})  # fmt: skip
    assert dict(tissue.hyperelasticity.parameters)["fibres"] == ((1.0, 0.0, 0.0),)

    def pull(component, strain):
        path = {"times": [0, 1.0], "increments": 20, "strain": {component: [0, strain]}}
        r = simulate_point(tissue, path, kinematics="finite")
        k = int(component[0]) - 1
        return 1.0 + r.strain[1:, k], r.stress[1:, k], r

    lam, stress, _ = pull("11", 0.5)
    matrix = lam - lam**-2
    fibres = 2 * k1 * lam * (lam**2 - 1) * np.exp(k2 * (lam**2 - 1) ** 2)
    assert stress == pytest.approx(matrix + fibres, rel=1e-3)
    assert stress[-1] > 50 * matrix[-1]
    lam, stress, _ = pull("11", -0.3)
    assert stress == pytest.approx(lam - lam**-2, rel=1e-5)
    lam, stress, r = pull("22", 0.5)
    assert stress == pytest.approx(lam - lam**-2, rel=1e-5)
    assert r.strain[-1, 0] == pytest.approx(r.strain[-1, 2], rel=1e-6)  # the fibres shorten
    # dispersed fibres (kappa = 1/3: no direction): an isotropic stiffening
    law = {"model": "holzapfel", "mu": 1.0, "k1": k1, "k2": k2, "kappa": 1 / 3,
           "fibres": [[1, 0, 0]], "bulk": 1e6}  # fmt: skip
    spread = Material(hyperelasticity=law)
    a = simulate_point(spread, {"times": [0, 1.0], "increments": 8, "strain": {"11": [0, 0.4]}},
                       kinematics="finite")  # fmt: skip
    b = simulate_point(spread, {"times": [0, 1.0], "increments": 8, "strain": {"33": [0, 0.4]}},
                       kinematics="finite")  # fmt: skip
    assert a.stress[:, 0] == pytest.approx(b.stress[:, 2], rel=1e-9)
    with pytest.raises(ValueError, match="isotropic"):
        tissue.hyperelasticity.law().nominal_stress("uniaxial", [1.2])


def test_softening_of_mullins():
    """Ogden and Roxburgh: on the way back from a stretch the stress is that of the
    virgin solid times 1 - erf((Wm - W) / (m + beta Wm)) / r, Wm the largest energy it
    has had; beyond it the virgin curve again; the tangent is that of this stress."""
    softening = (2.0, 0.3, 0.1)
    m = Material(hyperelasticity={"model": "neo_hookean", "mu": 1.0, "bulk": 1e6,
                                  "mullins": softening})  # fmt: skip
    assert Hyperelasticity.from_config(m.hyperelasticity.to_config()) == m.hyperelasticity
    path = {"times": [0, 1.0, 2.0, 3.0], "increments": [10, 10, 15],
            "strain": {"11": [0, 1.0, 0.2, 1.5]}}  # fmt: skip
    r = simulate_point(m, path, kinematics="finite")
    lam = 1.0 + r.strain[:, 0]
    virgin = lam - lam**-2
    energy = 0.5 * (lam**2 + 2 / lam - 3)
    reached = np.maximum.accumulate(energy)
    eta = 1.0 - erf((reached - energy) / (0.3 + 0.1 * reached)) / 2.0
    assert r.stress[1:, 0] == pytest.approx((eta * virgin)[1:], rel=1e-5)
    assert eta[:11] == pytest.approx(1.0) and eta[11:21].max() < 0.99 and eta[-1] == 1.0
    assert 0.5 < eta.min() < 0.51  # softened by half at most: 1 - 1 / r
    law = analytic_model(m, finite=True)
    rng = np.random.default_rng(3)
    e = 0.1 * rng.standard_normal((5, 6))
    state = np.full((5, 1), 0.5)  # an energy that these strains have not: unloading
    _, c, new = law.step(e, e, state, 1.0)
    assert np.array_equal(new, state)
    h = 1e-6
    for k in range(6):
        d = np.zeros(6)
        d[k] = h
        dc = (law.step(e + d, e, state, 1.0)[0] - law.step(e - d, e, state, 1.0)[0]) / (2 * h)
        assert np.allclose(c[:, :, k], dc, rtol=1e-5, atol=2e-6)
    assert np.allclose(c, np.swapaxes(c, 1, 2), rtol=1e-12, atol=1e-12)
    for bad in ([1.0, 0.3, 0.1], [2.0, 0.0, 0.0], [2.0, 0.3]):
        with pytest.raises(ValueError, match="mullins"):
            Hyperelasticity("neo_hookean", mullins=bad)
    with pytest.raises(ValueError, match="viscoelasticity"):
        analytic_model(
            Material(hyperelasticity={"model": "neo_hookean", "mu": 1.0, "bulk": 50.0,
                                      "mullins": softening},
                     viscoelasticity={"shear": [{"g": 0.5, "tau": 1.0}]}),
            finite=True,
        )  # fmt: skip


@pytest.mark.parametrize("model", sorted(RUBBERS))
def test_laws_fitted_to_the_tests_of_a_rubber(model):
    """The parameters of each law recovered from its own stresses in uniaxial and
    equibiaxial tension and in simple shear."""
    law = rubber(model).hyperelasticity.law()
    x, g = np.linspace(1.05, 2.2, 12), np.linspace(0.1, 1.0, 6)
    tests = {mode: (v, law.nominal_stress(mode, v, incompressible=True))
             for mode, v in (("uniaxial", x), ("equibiaxial", x), ("shear", g))}  # fmt: skip
    start = None
    if model == "polynomial":
        start = dict.fromkeys(RUBBERS[model], 0.0)
    elif model == "van_der_waals":
        start = {"beta": 0.0}  # its second invariant too
    fitted = fit_hyperelasticity(model, tests, start=start, terms=3)
    assert isinstance(fitted, Hyperelasticity) and fitted.model == model
    found = fitted.law(bulk=1.0, shear=1.0)
    for mode, (v, stress) in tests.items():
        assert found.nominal_stress(mode, v, incompressible=True) == pytest.approx(stress, abs=1e-8)
    if model != "ogden":  # (its pairs in any order)
        for name, value in RUBBERS[model].items():
            assert dict(fitted.parameters)[name] == pytest.approx(value, rel=1e-5, abs=1e-8)


def test_fits_of_the_tests_of_rubbers():
    """One test fitted by a law of fewer parameters: least squares; a parameter held;
    relative errors; what the fit refuses."""
    truth = Hyperelasticity("yeoh", (("c10", 0.5), ("c20", -0.04), ("c30", 0.01)))
    law = truth.law(bulk=1.0, shear=1.0)
    x = np.linspace(1.1, 3.0, 20)
    tests = {"uniaxial": (x, law.nominal_stress("uniaxial", x, incompressible=True))}
    one = fit_hyperelasticity("neo_hookean", tests)
    line = 2 * (x - x**-2)  # the stress of a neo-Hookean solid over its mu / 2... times 2
    best = float(line @ tests["uniaxial"][1] / (line @ line))  # least squares of c10
    assert dict(one.parameters)["mu"] == pytest.approx(2 * best, rel=1e-6)
    held = fit_hyperelasticity("yeoh", tests, fixed={"c30": 0.01})
    assert dict(held.parameters) == pytest.approx({"c10": 0.5, "c20": -0.04, "c30": 0.01})
    weighted = fit_hyperelasticity("neo_hookean", tests, relative=True)
    assert dict(weighted.parameters)["mu"] != pytest.approx(dict(one.parameters)["mu"], rel=1e-3)
    for model in ("holzapfel", "blatz_ko", "hyperfoam", "saint_venant_kirchhoff", "rubber"):
        with pytest.raises(ValueError, match="model among"):
            fit_hyperelasticity(model, tests)
    with pytest.raises(ValueError, match="modes"):
        fit_hyperelasticity("neo_hookean", {"torsion": (x, x)})
    with pytest.raises(ValueError, match="as many"):
        fit_hyperelasticity("neo_hookean", {"uniaxial": (x, x[:-1])})
    with pytest.raises(ValueError, match="coefficients"):
        fit_hyperelasticity("polynomial", tests)


def test_simple_shear_and_small_strains_of_a_neo_hookean_solid():
    m = rubber("neo_hookean")
    shear = {"times": [0, 1.0], "increments": 8,
             "strain": {k: v[:2] for k, v in {**FULL, "12": [0, 0.8]}.items()}}  # fmt: skip
    r = simulate_point(m, shear, kinematics="finite")
    assert np.allclose(r.stress[:, 5], 1.2 * r.strain[:, 5], rtol=1e-10)  # P12 = mu gamma
    # in small strains: the elastic material of its moduli at the natural state
    small = simulate_point(m, shear)
    assert np.allclose(small.stress[:, 5], 1.2 * small.strain[:, 5], rtol=1e-12)
    assert m.young == pytest.approx(9 * 20.0 * 1.2 / (3 * 20.0 + 1.2))
    assert Material.from_config(m.to_config()) == m


def test_hyperelasticity_is_checked():
    # the moduli of the material where the law gives none
    m = Material(young=2.6, poisson=0.3, hyperelasticity="neo_hookean")
    law = analytic_model(m, finite=True)
    assert law.mu == pytest.approx(1.0) and law.k == pytest.approx(2.6 / 1.2)
    ab = Material(young=2.6, poisson=0.3, hyperelasticity={"model": "arruda_boyce", "lock": 3.0})
    assert Hyperelasticity.from_config(ab.hyperelasticity.to_config()) == ab.hyperelasticity
    with pytest.raises(ValueError, match="shear modulus at the natural state"):
        Material(young=2.6, poisson=0.3, hyperelasticity={"model": "neo_hookean", "mu": 2.0})
    with pytest.raises(ValueError, match="bulk modulus"):
        Material(hyperelasticity={"model": "neo_hookean", "mu": 2.0})
    for bad, match in (
        ({"model": "rubber_band"}, "one of"),
        ({"model": "mooney_rivlin", "c10": 1.0, "c01": 0.1, "c40": 0.1}, "unknown"),
        ({"model": "polynomial"}, "coefficients"),
        ({"model": "polynomial", "c10": 1.0, "c70": 0.1}, "unknown"),
        ({"model": "yeoh", "c10": 1.0, "c01": 0.1}, "unknown"),
        ({"model": "eight_chain", "lock": 0.9}, "above 1"),
        ({"model": "van_der_waals", "lock": 1.5}, "square root"),
        ({"model": "van_der_waals", "lock": 6.0, "beta": 1.5}, "beta"),
        ({"model": "fung", "b": 0.0}, "positive"),
        ({"model": "holzapfel", "k1": 1.0, "k2": 1.0}, "fibres"),
        ({"model": "holzapfel", "k1": 1.0, "k2": 1.0, "fibres": [[0, 0, 0]]}, "directions"),
        (
            {"model": "holzapfel", "k1": 1.0, "k2": 1.0, "fibres": [[1, 0, 0]], "kappa": 0.5},
            "kappa",
        ),
        ({"model": "blatz_ko", "f": 2.0}, "from 0 to 1"),
        (
            {"model": "hyperfoam", "terms": [[1.0, 2.0]], "volumetric": "logarithmic"},
            "compressible",
        ),
        ({"model": "mooney_rivlin", "c10": 1.0}, "c01"),
        ({"model": "neo_hookean", "c10": 1.0}, "unknown"),
        ({"model": "gent", "lock": -1.0}, "positive"),
        ({"model": "ogden", "terms": [[1.0, 0.0]]}, "alpha"),
        ({"model": "neo_hookean", "volumetric": "cubic"}, "volumetric"),
    ):
        with pytest.raises(ValueError, match=match):
            Hyperelasticity.from_config(bad)
    with pytest.raises(ValueError, match="one behaviour"):
        Material(young=2.6, poisson=0.3, hyperelasticity="neo_hookean",
                 damage={"threshold": 1e-3, "ultimate": 1e-2})  # fmt: skip
    # a deformation that turns the matter inside out fails: no stress, the step is refused
    s, _, _ = law.step(np.array([[-0.6, 0, 0, 0, 0, 0.0]]), np.zeros((1, 6)), np.zeros((1, 0)), 1.0)
    assert np.isnan(s).all()


def test_finite_viscoelasticity():
    """The branches of a Prony series on a hyperelastic law (Simo 1987): a simple shear
    held relaxes its nominal shear stress as the series; the same series in the
    logarithmic strain space joins the law of small strains at small strains."""
    m = Material(hyperelasticity={"model": "neo_hookean", "mu": 1.2, "bulk": 50.0},
                 viscoelasticity={"shear": [[0.6, 1.0]]})  # fmt: skip
    path = {
        "times": [0, 1e-9, 12.0],
        "increments": [1, 30],
        "strain": {**FULL, "12": [0, 0.8, 0.8]},
    }
    r = simulate_point(m, path, kinematics="finite")
    t = r.times[1:] - 1e-9
    assert np.allclose(r.stress[1:, 5], 1.2 * 0.8 * (0.4 + 0.6 * np.exp(-t)), rtol=1e-8)
    tiny = {"times": [0, 1.0, 3.0], "increments": [5, 5], "strain": {"11": [0, 1e-6, 1e-6]}}
    small = simulate_point(VISCO, tiny)
    finite = simulate_point(VISCO, tiny, kinematics="finite")
    assert np.allclose(finite.stress, small.stress, rtol=1e-5, atol=1e-14)
    assert np.allclose(finite.strain, small.strain, rtol=1e-5, atol=1e-14)


# -- damage --------------------------------------------------------------------------------

YOUNG, STRENGTH, ENERGY = 30e3, 3.0, 0.1


def concrete(**options):
    values = {"strength": STRENGTH, "fracture_energy": ENERGY, "length": 5.0, **options}
    return Material(young=YOUNG, poisson=0.2, damage=values)


@pytest.mark.parametrize("softening", SOFTENINGS)
@pytest.mark.parametrize("equivalent", EQUIVALENT_STRAINS)
def test_tension_follows_the_softening_law(equivalent, softening):
    """A uniaxial tension: the stress of the softening law in closed form, its area the
    fracture energy over the length of the band, whatever the equivalent strain."""
    m = concrete(equivalent=equivalent, softening=softening)
    k0, kf = m.damage.strains(YOUNG)
    assert k0 == pytest.approx(STRENGTH / YOUNG)
    end = 9.0 * kf if softening == "exponential" else 1.1 * kf
    r = simulate_point(m, {"times": [0, 1.0, 2.0], "increments": [2, 600],
                           "strain": {"11": [0, k0, end]}}, tangent="full")  # fmt: skip
    e, s = r.strain[:, 0], r.stress[:, 0]
    if softening == "exponential":
        exact = np.where(e <= k0, YOUNG * e, STRENGTH * np.exp(-(e - k0) / (kf - k0)))
    else:
        exact = np.where(e <= k0, YOUNG * e, STRENGTH * np.maximum(kf - e, 0.0) / (kf - k0))
    exact = np.maximum(exact, m.damage.residual * YOUNG * e)  # the stiffness the damage leaves
    assert np.allclose(s, exact, rtol=1e-6, atol=1e-7 * STRENGTH)
    assert s.max() == pytest.approx(STRENGTH, rel=1e-9)
    assert np.trapezoid(s, e) * 5.0 == pytest.approx(ENERGY, rel=2e-3)
    assert np.abs(r.stress[:, 1:]).max() < 1e-8 * STRENGTH
    # the variable p of the outputs is the damage: it grows from 0 to (almost) 1
    assert r.plastic[2] < 1e-12 and np.all(np.diff(r.plastic) >= 0.0)
    assert r.plastic[-1] == pytest.approx(1.0 - s[-1] / (YOUNG * e[-1]), rel=1e-9)


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("equivalent", EQUIVALENT_STRAINS)
def test_tangent_of_a_growing_damage(equivalent, split):
    m = Material(young=YOUNG, poisson=0.2, damage={
        "threshold": 1e-4, "ultimate": 8e-4, "equivalent": equivalent, "split": split,
        "viscosity": 0.3})  # fmt: skip
    law = analytic_model(m, finite=False)
    rng = np.random.default_rng(1)
    e = 4e-4 * rng.standard_normal((8, 6))
    e[:4, :3] += 3e-4  # opened
    e[4:, :3] -= 4e-4  # compressed
    old = np.column_stack([np.full(8, 1.5e-4), np.full(8, 0.1)])
    _, c, new = law.step(e, e, old, 0.5)
    h = 1e-9
    for k in range(6):
        d = np.zeros(6)
        d[k] = h
        fd = (law.step(e + d, e, old, 0.5)[0] - law.step(e - d, e, old, 0.5)[0]) / (2 * h)
        assert np.allclose(c[:, :, k], fd, rtol=1e-5, atol=1e-6 * YOUNG)
    assert np.all(new[:, 1] >= 0.1) and np.all(new[:, 0] >= 1.5e-4)
    # symmetric for the energy norm only: the solvers keep the others whole (GMRES)
    asymmetry = np.abs(c - np.swapaxes(c, 1, 2)).max() / YOUNG
    assert (asymmetry < 1e-12) == (equivalent == "energy")
    assert law.symmetric == (equivalent == "energy")


def test_damage_is_irreversible_and_delayed():
    m = concrete(softening="linear")
    k0, kf = m.damage.strains(YOUNG)
    mid = 0.5 * (k0 + kf)
    cycle = {"times": [0, 1, 2, 3], "increments": [20, 10, 20],
             "strain": {"11": [0, mid, 0.2 * mid, 1.2 * mid]}}  # fmt: skip
    r = simulate_point(m, cycle, tangent="full")
    d = r.plastic
    assert d[20] == pytest.approx(kf * (mid - k0) / (mid * (kf - k0)), rel=1e-9)
    # unloading and reloading along the damaged secant, then the softening again
    assert np.allclose(d[20:31], d[20]) and np.allclose(d[31:36], d[20])
    assert np.allclose(r.stress[20:31, 0], (1 - d[20]) * YOUNG * r.strain[20:31, 0], rtol=1e-9)
    assert d[-1] > d[20]
    # a viscosity delays the damage: less of it at a faster rate, the same at rest
    slow = concrete(softening="linear", viscosity=1e-3)
    fast = concrete(softening="linear", viscosity=10.0)
    path = {"times": [0, 1.0, 400.0], "increments": [20, 40], "strain": {"11": [0, mid, mid]}}
    a, b = (simulate_point(x, path, tangent="full") for x in (slow, fast))
    assert a.plastic[20] == pytest.approx(d[20], rel=2e-2)
    assert b.plastic[20] < 0.2 * d[20] and b.stress[20, 0] > a.stress[20, 0]
    assert b.plastic[-1] == pytest.approx(d[20], rel=1e-6)


def test_strengths_in_compression():
    """Mazars' strain sees the lateral extension of a compressed point only: its strength
    in compression is 1 / (sqrt(2) nu) times that in tension; the modified von Mises
    strain has the ratio given; the energy norm none."""
    comp = {"times": [0, 1.0], "increments": 400, "strain": {"11": [0, -30 * STRENGTH / YOUNG]}}
    peaks = {}
    for equivalent in EQUIVALENT_STRAINS:
        m = concrete(equivalent=equivalent, ratio=8.0, fracture_energy=20.0)
        peaks[equivalent] = -simulate_point(m, comp, tangent="full").stress[:, 0].min()
    assert peaks["energy"] == pytest.approx(STRENGTH, rel=2e-3)
    assert peaks["mazars"] == pytest.approx(STRENGTH / (np.sqrt(2.0) * 0.2), rel=2e-3)
    assert peaks["mises"] == pytest.approx(8.0 * STRENGTH, rel=2e-3)
    # the volumetric split: a hydrostatic compression never damages, a tension does
    m = concrete(equivalent="energy", split="volumetric")
    press = {
        "times": [0, 1.0],
        "increments": 10,
        "strain": {c: [0, -1e-2 if c in ("11", "22", "33") else 0.0] for c in FULL},
    }
    r = simulate_point(m, press)
    k = YOUNG / (3 * (1 - 0.4))
    assert r.plastic[-1] == 0.0 and r.stress[-1, 0] == pytest.approx(-3e-2 * k, rel=1e-12)
    pull = {**press, "strain": {c: [-x for x in v] for c, v in press["strain"].items()}}
    assert simulate_point(m, pull, tangent="full").plastic[-1] > 0.99


def test_damage_is_checked():
    d = Damage(strength=3.0, fracture_energy=0.1, length=2.0)
    assert Damage.from_config(d.to_config()) == d
    m = concrete()
    assert Material.from_config(m.to_config()) == m
    for bad, match in (
        ({"ultimate": 1e-3}, "threshold"),
        ({"threshold": 1e-4, "strength": 3.0, "ultimate": 1e-3}, "threshold"),
        ({"threshold": 1e-4}, "ultimate strain"),
        ({"threshold": 1e-4, "fracture_energy": 0.1}, "length"),
        ({"threshold": 1e-4, "ultimate": 1e-3, "softening": "cubic"}, "softening"),
        ({"threshold": 1e-4, "ultimate": 1e-3, "equivalent": "rankine"}, "equivalent"),
        ({"threshold": 1e-4, "ultimate": 1e-3, "ratio": 0.5}, "ratio"),
        ({"threshold": 1e-4, "ultimate": 1e-3, "residual": 0.0}, "residual"),
        ({"threshold": 1e-4, "ultimate": 1e-3, "healing": 1.0}, "unknown"),
    ):
        with pytest.raises(ValueError, match=match):
            Damage.from_config(bad)
    # elements too large for the fracture energy: the stress would snap back
    with pytest.raises(ValueError, match="snap back"):
        concrete(length=2.0 * YOUNG * ENERGY / STRENGTH**2 * 1.01)


# -- the laws in the solvers ---------------------------------------------------------------


def layers(n=8, thickness=3):
    data = np.zeros((2, 2, n), dtype=np.uint8)
    data[:, :, :thickness] = 1
    return mg.VoxelImage(data, 1.0, periodic=True)


def test_viscoelastic_layers_by_both_solvers():
    """Layers sheared in their plane are in parallel: the mean of their relaxations."""
    elastic = Material(young=30.0, poisson=0.2)
    path = {
        "times": [0, 1e-9, 5.0],
        "increments": [1, 12],
        "strain": {**FULL, "12": [0, 0.01, 0.01]},
    }
    out = homogenize(layers(), {0: VISCO, 1: elastic}, properties=[], paths={"shear": path})
    r = out["shear.periodic"]
    _, g = VISCO.viscoelasticity.relaxation(r.times[1:] - 1e-9, K0, G0)
    assert np.allclose(r.stress[1:, 5], (5 / 8 * g + 3 / 8 * 12.5) * 0.01, rtol=1e-8)
    assert r.phases[0]["stress"].shape[0] == r.times.size
    mesh = box_mesh((2, 2, 4), upper=(1, 1, 1), phase=lambda c: (c[:, 2] < 0.5).astype(int))
    fe = homogenize(mesh, {0: VISCO, 1: elastic}, properties=[], paths={"shear": path},
                    bcs=["periodic"])["shear.periodic"]  # fmt: skip
    assert np.allclose(fe.stress[1:, 5], (0.5 * g + 0.5 * 12.5) * 0.01, rtol=1e-8)


@pytest.mark.parametrize("solver", ["fft", "fe"])
def test_a_homogeneous_rubber_is_its_material_point(solver):
    m = rubber("mooney_rivlin")
    tension = {"times": [0, 1.0], "increments": 6, "strain": {"11": [0, 0.5]}}
    ref = simulate_point(m, tension, kinematics="finite")
    if solver == "fft":
        image = mg.VoxelImage(np.ones((3, 3, 3), dtype=np.uint8), 1.0, periodic=True)
        r = NonlinearFFTProblem(image, {1: m}, kinematics="finite").run(tension)
    else:
        mesh = box_mesh((2, 2, 2), upper=(1, 1, 1))
        r = NonlinearFEProblem(mesh, {0: m}, "periodic", kinematics="finite").run(tension)
    assert np.allclose(r.stress, ref.stress, rtol=1e-5, atol=1e-6)
    assert np.allclose(r.strain, ref.strain, rtol=1e-5, atol=1e-7)
    assert np.allclose(r.cauchy, ref.cauchy, rtol=1e-5, atol=1e-6)


def test_the_dissipation_of_a_crack_band_does_not_depend_on_the_voxels():
    """A bar of 5, 10 or 20 voxels pulled until its weakest voxel breaks: with the length
    of the damage law that of the voxels, the work done is the fracture energy per unit
    area, whatever their size (without Poisson's contraction, which the neighbours of the
    band would constrain)."""
    length = 10.0
    works = []
    for n in (5, 10, 20):
        h = length / n
        weak = Material(young=YOUNG, poisson=0.0, damage={
            "strength": STRENGTH, "fracture_energy": ENERGY, "length": h, "softening": "linear",
            "residual": 1e-9})  # fmt: skip
        data = np.zeros((1, 1, n), dtype=np.uint8)
        data[:, :, n // 2] = 1
        image = mg.VoxelImage(data, h, periodic=True)
        _, kf = weak.damage.strains(YOUNG)
        path = {"times": [0, 1.0], "increments": 60, "strain": {"33": [0, 1.05 * kf * h / length]}}
        problem = NonlinearFFTProblem(image, {0: Material(young=YOUNG, poisson=0.0), 1: weak})
        r = problem.run(path)
        assert r.stress[:, 2].max() == pytest.approx(STRENGTH, rel=2e-2)
        assert abs(r.stress[-1, 2]) < 1e-3 * STRENGTH
        assert r.phases[1]["plastic"][-1] > 0.999  # the damage of the band
        works.append(np.trapezoid(r.stress[:, 2], r.strain[:, 2]) * length)
    assert np.allclose(works, ENERGY, rtol=1e-2)
    assert max(works) - min(works) < 2e-3 * ENERGY


def test_a_prony_series_identified_on_a_relaxation():
    """The macroscopic law of viscoelastic layers in parallel with elastic ones: one
    Maxwell branch of the same time, its weight that of the mixture."""
    from morphogen.homogenization.identification import identify

    one = Material(young=E0, poisson=NU0, viscoelasticity={"shear": [[0.5, 2.0]]})
    stiff = 12.5
    path = {
        "times": [0, 1e-9, 8.0],
        "increments": [1, 16],
        "strain": {**FULL, "12": [0, 0.01, 0.01]},
    }
    ref = homogenize(layers(), {0: one, 1: Material(young=30.0, poisson=0.2)}, properties=[],
                     paths={"shear": path})["shear.periodic"]  # fmt: skip
    g_mix = 5 / 8 * G0 + 3 / 8 * stiff
    start = {"young": 2.6 * g_mix, "poisson": 0.3, "viscoelasticity": {"shear": [[0.1, 0.7]]}}
    fit = identify(start, ref.path, ref.times, ref.stress, ["g1", "tau1"], components=[5])
    assert fit.values["tau1"] == pytest.approx(2.0, rel=1e-4)
    assert fit.values["g1"] == pytest.approx(5 / 8 * 0.5 * G0 / g_mix, rel=1e-4)
    assert fit.error < 1e-6 < fit.initial_error
    with pytest.raises(ValueError, match="shear terms"):
        identify(start, ref.path, ref.times, ref.stress, ["g2"], components=[5])
    with pytest.raises(ValueError, match="parameter"):
        identify(start, ref.path, ref.times, ref.stress, ["yield"], components=[5])


def test_a_crack_in_a_hyperelastic_plate():
    """The laws of this module as the bulk of the crack solvers: a neo-Hookean plate at
    small strains has the energy release rate of the elastic one."""
    from morphogen.fracture import lefm
    from morphogen.fracture.mixed import solve_crack
    from morphogen.homogenization.bloch import voxel_tet_mesh

    image = mg.VoxelImage(np.zeros((8, 16, 1), np.uint8), spacing=(1.0, 1.0, 1.0), periodic=False)
    mesh = voxel_tet_mesh(image)
    crack = lefm.Crack("straight", (0.0, 8.0, 0.0), (0, 1, 0), 2.5, direction=(1, 0, 0))

    def grips(m):
        top = np.flatnonzero(m.nodes[:, 1] > 16.0 - 1e-9)
        bottom = np.flatnonzero(m.nodes[:, 1] < 1e-9)
        return [(top, [0, 1, 2], [0.0, 0.01, 0.0]), (bottom, [0, 1, 2], [0.0, -0.01, 0.0])]

    elastic = Material(young=1000.0, poisson=0.3)
    hyper = Material(young=1000.0, poisson=0.3, hyperelasticity="neo_hookean")
    ref = solve_crack(mesh, crack, elastic, grips, kinematics="finite", steps=2)
    sol = solve_crack(mesh, crack, hyper, grips, kinematics="finite", steps=2)
    assert sol.bulk is not None
    scale = np.abs(ref.displacement).max()
    assert np.abs(sol.displacement - ref.displacement).max() < 2e-3 * scale
    assert sol.release_rate(energy="work") == pytest.approx(ref.release_rate(), rel=5e-3)


def test_exports_refuse_the_laws_without_a_source(tmp_path):
    from morphogen.io.mfront import write_mfront

    for m in (VISCO, rubber("gent"), concrete()):
        with pytest.raises(ValueError, match="no source"):
            write_mfront(m, tmp_path / "law.mfront")
