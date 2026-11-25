"""Macroscopic loadings: mixed strain/stress control (the macroscopic strain as unknowns
dual to the stress, like the periodic element of Z-set), thermal expansion, and the
post-processing of apparent properties."""

import math

import numpy as np
import pytest

import morphogen as mg
from helpers import box_mesh
from morphogen.homogenization import (
    VOID,
    Loading,
    Material,
    conductivity_tensor,
    homogenize,
    make_problem,
    stiffness_tensor,
)
from morphogen.homogenization.apparent import thermal_expansion
from morphogen.homogenization.elements import tensor_to_voigt
from morphogen.homogenization.materials import isotropic_matrix, lame
from morphogen.homogenization.postprocess import (
    NOTATION,
    PLANE_NOTATION,
    anisotropy,
    compliance,
    directional_modulus,
    effective_density,
    engineering_constants,
    matrix_entries,
    normalized_values,
    notation,
    spatial_modulus,
    sphere_directions,
)

MATRIX = Material(young=2.8, poisson=0.39, conductivity=0.3, expansion=95e-6, density=1.14)
FIBRE = Material(young=89.0, poisson=0.3, conductivity=3.0, expansion=5.5e-6, density=2.7)
MATS = {0: MATRIX, 1: FIBRE}


def random_phases(seed, fraction=0.35):
    rng = np.random.default_rng(seed)
    return lambda c: (rng.random(len(c)) < fraction).astype(int)


def voigt_strain(sol):
    return tensor_to_voigt(sol.strain, strain=True)


def voigt_stress(sol):
    return tensor_to_voigt(sol.stress, strain=False)


def rosen_hashin(compliance, f1, m0=MATRIX, m1=FIBRE):
    """Exact thermal expansion of a two-phase composite of isotropic phases from its
    compliance (Levin 1967; Rosen & Hashin 1970)."""
    s0 = np.linalg.inv(isotropic_matrix(*lame(m0.young, m0.poisson)))
    s1 = np.linalg.inv(isotropic_matrix(*lame(m1.young, m1.poisson)))
    unit = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    mean = (1 - f1) * m0.expansion + f1 * m1.expansion
    jump = (m1.expansion - m0.expansion) / (1 / (3 * m1.bulk) - 1 / (3 * m0.bulk))
    return mean * unit + jump * (compliance - ((1 - f1) * s0 + f1 * s1)) @ unit


def test_loadings():
    mixed = Loading.mixed(strain={"11": 0.01, "32": 0.0}, stress=[None, 1, 2, None, 3, 4])
    assert mixed.kind == "mixed" and mixed.value == (0.01, 1, 2, 0, 3, 4)
    assert mixed.strain_controlled.tolist() == [True, False, False, True, False, False]
    assert mixed.physics == "elastic" and mixed.to_dict()["controls"][0] is True
    uni = Loading.uniaxial(1, strain=0.02, temperature=5.0)
    assert uni.strain_controlled.tolist() == [False, True, False, False, False, False]
    assert uni.value[1] == 0.02 and uni.temperature == 5.0
    assert Loading.uniaxial(2, stress=3.0).kind == "stress"
    heat = Loading.mixed(gradient={"1": 1.0}, flux=[None, 0.0, 0.0])
    assert heat.physics == "thermal" and heat.strain_controlled.tolist() == [True, False, False]
    assert Loading.strain(np.eye(3), temperature=2.0).to_dict()["temperature"] == 2.0
    for bad, match in (
        (lambda: Loading.mixed(strain={"11": 1}, stress={"11": 1}), "twice"),
        (lambda: Loading.mixed(strain={"11": 1}), "not prescribed"),
        (lambda: Loading.mixed(strain={"14": 1}), "unknown component"),
        (lambda: Loading.gradient([1, 0, 0]).__class__("gradient", [1, 0, 0], None, 1.0),
         "temperature"),
        (lambda: Loading("mixed", [0] * 6), "controls"),
        (lambda: Loading.uniaxial(0), "strain or the stress"),
    ):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            bad()


@pytest.mark.parametrize("bc", ["periodic", "kubc", "subc"])
@pytest.mark.parametrize("order", [1, 2])
def test_mixed_control(bc, order):
    """Stress, strain and mixed loadings all follow the apparent stiffness: the
    macroscopic unknowns are exact (up to the solver tolerance)."""
    mesh = box_mesh((4, 4, 4), phase=random_phases(1), order=order, perturb=0.2)
    problem = make_problem(mesh, MATS, "elastic", bc, tol=1e-12)
    c, _ = stiffness_tensor(problem)
    s = np.linalg.inv(c)
    sol = problem.solve(Loading.uniaxial(0, strain=0.01))
    e, sig = voigt_strain(sol), voigt_stress(sol)
    assert e[0] == pytest.approx(0.01) and np.abs(sig[1:]).max() < 1e-9 * sig[0]
    assert sig[0] / e[0] == pytest.approx(1 / s[0, 0], rel=1e-8)
    assert -e[1] / e[0] == pytest.approx(-s[0, 1] / s[0, 0], rel=1e-6)
    stress = np.array([1.0, -0.5, 0.2, 0.3, -0.1, 0.4])
    sol = problem.solve(Loading.stress(stress))
    np.testing.assert_allclose(voigt_strain(sol), s @ stress, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(voigt_stress(sol), stress, atol=1e-8)
    # any mix: the prescribed components of both kinds, and Sigma = C E
    rng = np.random.default_rng(3)
    controls = rng.random(6) < 0.5
    values = rng.uniform(-1, 1, 6)
    names = ("11", "22", "33", "23", "31", "12")
    loading = Loading.mixed(
        strain={n: v for n, v, k in zip(names, values, controls, strict=True) if k},
        stress={n: v for n, v, k in zip(names, values, controls, strict=True) if not k},
    )
    sol = problem.solve(loading)
    e, sig = voigt_strain(sol), voigt_stress(sol)
    np.testing.assert_allclose(e[controls], values[controls], atol=1e-12)
    np.testing.assert_allclose(sig[~controls], values[~controls], atol=1e-7)
    np.testing.assert_allclose(sig, c @ e, atol=1e-7 * np.abs(c).max())


def test_mixed_conditions_need_strains():
    mesh = box_mesh((3, 3, 3), phase=random_phases(0))
    problem = make_problem(mesh, MATS, "elastic", "mixed")
    with pytest.raises(ValueError, match="kubc, periodic or subc"):
        problem.solve(Loading.stress(np.eye(3)))
    with pytest.raises(ValueError, match="loading is thermal"):
        problem.solve(Loading.gradient([1, 0, 0]))


@pytest.mark.parametrize("bc", ["periodic", "kubc", "subc"])
def test_thermal_expansion(bc):
    """The apparent thermal expansion of two isotropic phases follows exactly from the
    apparent compliance (Levin, Rosen & Hashin), on any mesh and boundary condition."""
    mesh = box_mesh((4, 5, 4), phase=random_phases(2), perturb=0.2)
    problem = make_problem(mesh, MATS, "elastic", bc, tol=1e-12)
    c, _ = stiffness_tensor(problem)
    a, sol = thermal_expansion(problem)
    f1 = mesh.phase_fractions()[1]
    expected = rosen_hashin(np.linalg.inv(c), f1)
    np.testing.assert_allclose(tensor_to_voigt(a, strain=True), expected, rtol=1e-6, atol=1e-12)
    assert np.abs(voigt_stress(sol)).max() < 1e-12
    # E = S : Sigma + A dT under a mixed loading with a change of temperature
    loading = Loading.uniaxial(2, stress=0.5, temperature=10.0)
    sol = problem.solve(loading)
    s = np.linalg.inv(c)
    sigma = np.array([0, 0, 0.5, 0, 0, 0])
    np.testing.assert_allclose(
        voigt_strain(sol), s @ sigma + 10.0 * tensor_to_voigt(a, strain=True), atol=1e-9
    )


def test_uniform_expansion_and_fft():
    """One expanding material: free expansion without stress; FFT and FE agree with the
    exact relation for two phases."""
    uniform = {0: MATRIX, 1: MATRIX}
    mesh = box_mesh((3, 3, 3), phase=random_phases(4), perturb=0.2)
    for bc in ("periodic", "kubc", "subc"):
        a, _ = thermal_expansion(make_problem(mesh, uniform, "elastic", bc))
        np.testing.assert_allclose(a, 95e-6 * np.eye(3), atol=1e-15)
    heated = Loading.strain(np.zeros(6), temperature=1.0)
    sol = make_problem(mesh, uniform, "elastic", "kubc").solve(heated)
    k = MATRIX.bulk
    np.testing.assert_allclose(sol.stress, -3 * k * 95e-6 * np.eye(3), rtol=1e-10)
    img = mg.VoxelImage(
        (np.random.default_rng(5).random((6, 6, 6)) < 0.3).astype(np.uint8), periodic=True
    )
    problem = make_problem(img, MATS, "elastic", tol=1e-12)
    c, _ = stiffness_tensor(problem)
    a, _ = thermal_expansion(problem)
    expected = rosen_hashin(np.linalg.inv(c), img.data.mean())
    np.testing.assert_allclose(tensor_to_voigt(a, strain=True), expected, rtol=1e-6, atol=1e-12)
    out = homogenize(img, MATS, properties=["thermal_expansion"], tol=1e-12)
    entry = out["thermal_expansion.periodic"]
    assert entry["A11"] == pytest.approx(a[0, 0]) and entry["mean"] == pytest.approx(
        np.trace(a) / 3
    )


def test_mixed_conduction():
    mesh = box_mesh((4, 3, 4), phase=random_phases(6), perturb=0.2)
    for bc in ("periodic", "kubc", "subc"):
        problem = make_problem(mesh, MATS, "thermal", bc, tol=1e-12)
        k, _ = conductivity_tensor(problem)
        sol = problem.solve(Loading.mixed(gradient={"1": 1.0}, flux={"2": 0.0, "3": 0.0}))
        assert sol.strain[0] == pytest.approx(1.0) and np.abs(sol.stress[1:]).max() < 1e-9
        # the conductivity along 1 with free transverse gradients: 1 / (K^-1)_11
        assert -sol.stress[0] == pytest.approx(1.0 / np.linalg.inv(k)[0, 0], rel=1e-7)


def test_porous_stress_control():
    """A porous medium under a prescribed stress: the macroscopic strain is solved for (not
    the average over the solid), the macroscopic stress is the solid fraction times the
    average stress over the solid."""
    mesh = box_mesh((5, 5, 5), phase=random_phases(7, 0.75), perturb=0.1)
    mats = {0: VOID, 1: MATRIX}
    problem = make_problem(mesh, mats, "elastic", "periodic", tol=1e-12)
    c, _ = stiffness_tensor(problem)
    sigma = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sol = problem.solve(Loading.stress(sigma))
    np.testing.assert_allclose(voigt_strain(sol), np.linalg.solve(c, sigma), rtol=1e-7, atol=1e-12)
    (solid,) = sol.phases
    np.testing.assert_allclose(solid.fraction * solid.mean["stress"], sigma, atol=1e-9)
    assert not np.allclose(solid.mean["strain"], tensor_to_voigt(sol.strain, strain=True))


def test_engineering_constants_and_anisotropy():
    lam, mu = lame(70.0, 0.25)
    iso = engineering_constants(isotropic_matrix(lam, mu))
    assert iso["E1"] == pytest.approx(70.0) and iso["nu23"] == pytest.approx(0.25)
    assert iso["G12"] == pytest.approx(70.0 / 2.5)
    a = anisotropy(isotropic_matrix(lam, mu))
    assert a["universal_anisotropy"] == pytest.approx(0.0, abs=1e-12)
    assert a["zener_ratio"] == pytest.approx(1.0) and a["isotropy_distance"] < 1e-14
    copper = np.array(Material.cubic(168.4, 121.4, 75.4).stiffness)
    a = anisotropy(copper)
    zener = 2 * 75.4 / (168.4 - 121.4)
    assert a["zener_ratio"] == pytest.approx(zener)
    # cubic crystals: A_U = 6 (sqrt(A) - 1 / sqrt(A))^2 / 5 (Ranganathan 2008)
    expected = 6.0 / 5.0 * (math.sqrt(zener) - 1.0 / math.sqrt(zener)) ** 2
    assert a["universal_anisotropy"] == pytest.approx(expected, rel=1e-10)


#: An orthotropic material by its engineering constants (morphoGen's Voigt order 11, 22,
#: 33, 23, 31, 12, engineering shears): three distinct shear moduli, so that an exchange
#: of the shear components (TFEL and Abaqus order theirs 12, 13, 23) shows
ORTHOTROPIC = {"E1": 120.0, "E2": 60.0, "E3": 30.0, "nu12": 0.3, "nu13": 0.25, "nu23": 0.35,
               "G23": 7.0, "G31": 11.0, "G12": 17.0}  # fmt: skip


def _orthotropic_compliance(e: dict) -> np.ndarray:
    """The compliance of orthotropic engineering constants, written out: ``S_ii = 1/E_i``,
    ``S_ij = -nu_ij/E_i`` and the inverse shear moduli on the shear diagonal."""
    young = (e["E1"], e["E2"], e["E3"])
    s = np.diag([*(1.0 / y for y in young), 1.0 / e["G23"], 1.0 / e["G31"], 1.0 / e["G12"]])
    for (i, j), nu in (((0, 1), e["nu12"]), ((0, 2), e["nu13"]), ((1, 2), e["nu23"])):
        s[i, j] = s[j, i] = -nu / young[i]
    return s


def _check_compliance(c, s) -> None:
    """S C = I and the engineering constants of S, those of ORTHOTROPIC in its slots."""
    c, s = np.asarray(c, dtype=float), np.asarray(s, dtype=float)
    np.testing.assert_allclose(s @ c, np.eye(6), atol=1e-9)
    e = engineering_constants(c)
    assert s[0, 0] == pytest.approx(1.0 / e["E1"]) and e["E1"] == pytest.approx(120.0)
    assert s[0, 1] == pytest.approx(-e["nu12"] / e["E1"]) and e["nu12"] == pytest.approx(0.3)
    assert s[0, 1] == pytest.approx(-e["nu21"] / e["E2"]) and s[1, 0] == pytest.approx(s[0, 1])
    for k, name in zip((3, 4, 5), ("G23", "G31", "G12"), strict=True):
        assert s[k, k] == pytest.approx(1.0 / e[name])
        assert s[k, k] == pytest.approx(1.0 / ORTHOTROPIC[name])  # each in its own slot


def _turned(c, theta):
    """A Voigt stiffness in the axes turned by ``theta`` about z (the new first axis along
    ``theta``), through its fourth-order tensor."""
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1)]
    full = np.zeros((3, 3, 3, 3))
    for i, (a, b) in enumerate(pairs):
        for j, (k, m) in enumerate(pairs):
            for p, q in ((a, b), (b, a)):
                for r, s in ((k, m), (m, k)):
                    full[p, q, r, s] = c[i, j]
    co, si = math.cos(theta), math.sin(theta)
    rot = np.array([[co, si, 0.0], [-si, co, 0.0], [0.0, 0.0, 1.0]])
    full = np.einsum("ia,jb,kc,ld,abcd->ijkl", rot, rot, rot, rot, full)
    return np.array([[full[a, b, k, m] for k, m in pairs] for a, b in pairs])


def test_directional_modulus():
    lam, mu = lame(70.0, 0.25)
    iso = isotropic_matrix(lam, mu)
    angles = np.linspace(0.0, math.pi, 37)
    # isotropic: E in plane stress, E / (1 - nu^2) in plane strain, in every direction
    np.testing.assert_allclose(directional_modulus(iso, angles, plane_strain=False), 70.0)
    np.testing.assert_allclose(directional_modulus(iso, angles), 70.0 / (1.0 - 0.25**2))
    # orthotropic: E1 and E2 along the axes; along any angle, the modulus 1/S'11 of the
    # stiffness turned by it (plane stress) or of its in-plane block (plane strain)
    c = np.linalg.inv(_orthotropic_compliance(ORTHOTROPIC))
    e = directional_modulus(c, [0.0, 0.5 * math.pi], plane_strain=False)
    np.testing.assert_allclose(e, [120.0, 60.0])
    k = [0, 1, 5]
    for theta in angles:
        turned = _turned(c, theta)
        stress = 1.0 / np.linalg.inv(turned)[0, 0]
        strain = 1.0 / np.linalg.inv(turned[np.ix_(k, k)])[0, 0]
        assert directional_modulus(c, theta, plane_strain=False) == pytest.approx(stress)
        assert directional_modulus(c, theta) == pytest.approx(strain)
    # a plane stiffness (11, 22, 12) is a plane strain one; the other planes; the angles
    # from the second axis given first
    np.testing.assert_allclose(directional_modulus(c[np.ix_(k, k)], angles),
                               directional_modulus(c, angles))  # fmt: skip
    e = directional_modulus(c, [0.0, 0.5 * math.pi], plane=(1, 2), plane_strain=False)
    np.testing.assert_allclose(e, [60.0, 30.0])
    np.testing.assert_allclose(directional_modulus(c, angles, plane=(1, 0)),
                               directional_modulus(c, 0.5 * math.pi - angles))  # fmt: skip
    for bad in ({"stiffness": np.eye(4)}, {"plane": (1, 1)}):
        with pytest.raises(ValueError):
            directional_modulus(bad.get("stiffness", c), angles, plane=bad.get("plane", (0, 1)))


def test_spatial_modulus():
    lam, mu = lame(70.0, 0.25)
    directions = sphere_directions(500)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.all(directions[:, 2] > 0) and abs(directions[:, :2].mean(axis=0)).max() < 0.01
    np.testing.assert_allclose(spatial_modulus(isotropic_matrix(lam, mu), directions), 70.0)
    # in a plane of the axes, the directional modulus in plane stress
    c = np.linalg.inv(_orthotropic_compliance(ORTHOTROPIC))
    angles = np.linspace(0.0, math.pi, 37)
    turned = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    np.testing.assert_allclose(spatial_modulus(c, 3.0 * turned),
                               directional_modulus(c, angles, plane_strain=False))  # fmt: skip
    # a cubic crystal: 1 / S11 along <100>, 1 / (S11 - 2 (S11 - S12 - S44 / 2) / 3) along <111>
    cubic = np.zeros((6, 6))
    cubic[:3, :3] = 120.0
    cubic[[0, 1, 2], [0, 1, 2]] = 170.0
    cubic[[3, 4, 5], [3, 4, 5]] = 75.0
    s = np.linalg.inv(cubic)
    e = spatial_modulus(cubic, [[1, 0, 0], [1, 1, 1]])
    expected = 1.0 / (s[0, 0] - 2.0 * (s[0, 0] - s[0, 1] - 0.5 * s[3, 3]) / 3.0)
    np.testing.assert_allclose(e, [1.0 / s[0, 0], expected])
    with pytest.raises(ValueError):
        spatial_modulus(np.eye(3), directions)


def test_compliance_of_a_singular_stiffness():
    """A stiffness singular to round-off, that of layers of void across which nothing is
    stiff (C33, C44, C55 and the couplings with them at round-off), has no compliance:
    NaN rather than the inverse of its round-off (S33 -3e15 printed before); an exactly
    singular one raises as before; a regular one is its inverse, to the bit, and so is one
    only badly scaled (a design stiff along its fibres whose plane is as stiff as its void:
    a condition number of 1e13, its eigenvalues 1e-14 of each other)."""
    from morphogen.homogenization.postprocess import compliance

    lam, mu = 0.3 / (1.3 * 0.4), 1.0 / 2.6  # E = 1, nu = 0.3
    c = np.zeros((6, 6))
    c[:3, :3] = lam
    c[[0, 1, 2], [0, 1, 2]] += 2.0 * mu
    c[[3, 4, 5], [3, 4, 5]] = mu
    assert np.array_equal(compliance(c), np.linalg.inv(c))
    voids = c.copy()  # nothing stiff across the layers normal to z
    voids[2, :] = voids[:, 2] = 0.0
    voids[3, 3] = voids[4, 4] = 0.0
    round_off = voids.copy()
    round_off[2, 2], round_off[3, 3], round_off[4, 4] = 3e-17, 2e-17, 1e-17
    round_off[0, 2] = round_off[2, 0] = 4e-18
    w = np.linalg.eigvalsh(round_off)
    assert 0.0 < w[0] < 1e-16 * w[-1]
    assert np.isnan(compliance(round_off)).all()
    with pytest.raises(np.linalg.LinAlgError):
        compliance(voids)
    assert np.isnan(compliance(np.diag([1.0, 1e-17, 1.0]))).all()  # in the plane too
    scaled = np.diag([8e-12, 9e-12, 3.28, 4e-13, 9.6e-14, 1.2e-12])
    scaled[0, 1] = scaled[1, 0] = 9.7e-13
    assert np.linalg.cond(scaled) > 1e13
    np.testing.assert_allclose(compliance(scaled) @ scaled, np.eye(6), atol=1e-12)


def test_stiffness_and_compliance(tmp_path, capsys):
    """Every output of a stiffness C gives its compliance S = C^-1 in the same notation
    (Voigt 11 22 33 23 31 12, engineering shears): S C = I, S11 = 1/E1, S12 = -nu12/E1 =
    -nu21/E2, S44 = 1/G23, S55 = 1/G31, S66 = 1/G12, on an orthotropic material whose three
    shear moduli differ: the apparent stiffness of a homogeneous image (FFT), the columns of
    the results of a run and its notation, the text and JSON of morphogen homogenize."""
    import json

    import yaml

    from morphogen import cli, pipeline
    from morphogen.config import load_config
    from morphogen.results import axis_label

    s_exact = _orthotropic_compliance(ORTHOTROPIC)
    c_exact = np.linalg.inv(s_exact)
    _check_compliance(c_exact, compliance(c_exact))
    assert notation(c_exact) == NOTATION and notation(c_exact[:3, :3]) == PLANE_NOTATION
    assert "11 22 33 23 31 12" in NOTATION and "12 13 23" in NOTATION
    with pytest.raises(ValueError, match="6 x 6"):
        compliance(np.eye(4))
    entries = matrix_entries(s_exact, "S")
    assert len(entries) == 21 and entries["S44"] == pytest.approx(1.0 / 7.0)
    assert entries["S12"] == pytest.approx(-0.3 / 120.0) and "S21" not in entries
    # the apparent stiffness of a homogeneous image: the material itself
    image = mg.VoxelImage(np.zeros((4, 4, 4), dtype=np.uint8), periodic=True)
    res = homogenize(image, {0: Material(stiffness=c_exact)}, properties=["stiffness"])
    res = res["stiffness.periodic"]
    np.testing.assert_allclose(res["tensor"], c_exact, rtol=1e-7, atol=1e-7 * c_exact.max())
    _check_compliance(res["tensor"], res["compliance"])
    np.testing.assert_allclose(res["compliance"], s_exact, rtol=1e-7, atol=1e-9)
    # a run: the columns C.. and S.. of results.csv, their notation in run.json
    stiffness = {"stiffness": c_exact.tolist()}
    cfg = {
        "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
        "domain": {"size": 4.0, "periodic": True},
        "seed": 1,
        "voxels": {"shape": 6},
        "homogenization": {
            "materials": {0: stiffness, 1: stiffness},
            "method": "fft",
            "properties": ["stiffness"],
        },
        "outputs": {"directory": str(tmp_path / "run"), "name": "s"},
    }
    outdir = pipeline.run(load_config(cfg))
    (row,) = pipeline.read_rows(outdir / "results.csv")
    c = np.zeros((6, 6))
    s = np.zeros((6, 6))
    for i in range(6):
        for j in range(i, 6):
            c[i, j] = c[j, i] = row[f"stiffness.periodic.C{i + 1}{j + 1}"]
            s[i, j] = s[j, i] = row[f"stiffness.periodic.S{i + 1}{j + 1}"]
    _check_compliance(c, s)
    assert row["stiffness.periodic.S55"] == pytest.approx(1.0 / row["stiffness.periodic.G31"])
    manifest = json.loads((outdir / "run.json").read_text(encoding="utf-8"))
    assert manifest["stiffness_notation"] == NOTATION
    label = axis_label("stiffness.periodic.S44")
    assert label.startswith("Compliance S44 (periodic; Voigt 11 22 33 23 31 12")
    assert axis_label("stiffness.kubc.C12").startswith("Stiffness C12 (kubc; Voigt")
    assert axis_label("tension.periodic.S11") == "S11 (periodic)"  # a stress, not S
    # the command line: its text and JSON
    mg.io.save_image(image, tmp_path / "ortho.vti")
    (tmp_path / "ortho.yaml").write_text(yaml.safe_dump({0: stiffness}), encoding="utf-8")
    args = ["homogenize", str(tmp_path / "ortho.vti"), "--materials", str(tmp_path / "ortho.yaml")]
    args += ["--property", "stiffness"]
    assert cli.main([*args, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)["stiffness.periodic"]
    _check_compliance(out["tensor"], out["compliance"])
    assert out["notation"] == NOTATION and out["G23"] == pytest.approx(7.0)
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert (
        "stiffness C (Voigt, order 11 22 33 23 31 12 (TFEL and Abaqus: 11 22 33 12 13 23)" in text
    )
    assert "compliance S = C^-1" in text and "G23 7, G31 11, G12 17" in text


def test_normalized_values():
    fractions = {0: 0.25, 1: 0.75}
    mats = {0: VOID, 1: MATRIX}
    values = {"bulk_modulus.periodic": 1.0, "stiffness.kubc.E1": 0.9, "conductivity.kubc": 0.1}
    out = normalized_values(values, fractions, mats)
    assert out["solid_fraction"] == 0.75 and out["density"] == pytest.approx(0.75 * 1.14)
    assert out["bulk_modulus.periodic.relative"] == pytest.approx(1.0 / MATRIX.bulk)
    assert out["bulk_modulus.periodic.normalized"] == pytest.approx(1.0 / (0.75 * MATRIX.bulk))
    assert out["stiffness.kubc.E1.relative"] == pytest.approx(0.9 / 2.8)
    assert out["conductivity.kubc.specific"] == pytest.approx(0.1 / (0.75 * 1.14))
    undefined = {0: Material(young=1, poisson=0.2), 1: MATRIX}
    assert effective_density({0: 0.5, 1: 0.5}, undefined) is None


def test_materials_with_expansion_and_density():
    m = Material.from_config({"young": 2.8, "poisson": 0.39, "expansion": 9.5e-5, "density": 1.1})
    assert m.expansion == 9.5e-5 and m.density == 1.1
    assert Material.from_config(m.to_config()) == m
    hexa = Material.hexagonal(160.0, 90.0, 66.0, 181.0, 46.5, expansion=(1e-5, 2e-5))
    assert hexa.oriented("elastic") and isinstance(hexa.expansion, tuple)
    turned = hexa.expansion_tensor(np.array([0.0, 90.0, 0.0]))
    assert turned[1, 1] == pytest.approx(2e-5) and turned[2, 2] == pytest.approx(1e-5)
    for bad, match in (
        ({"conductivity": 1.0, "expansion": 1e-5}, "elastic parameters"),
        ({"young": 1.0, "poisson": 0.2, "expansion": [1, 2]}, "expansion"),
        ({"young": 1.0, "poisson": 0.2, "density": -1.0}, "density"),
    ):
        with pytest.raises(ValueError, match=match):
            Material.from_config(bad)
    with pytest.raises(ValueError, match="void"):
        Material(void=True, expansion=1e-5)


def test_loadings_in_runs_and_on_the_command_line(tmp_path, capsys):
    import json

    from morphogen import cli, pipeline
    from morphogen.config import ConfigError, load_config

    cfg = {
        "model": {"type": "boolean_spheres", "radius": 1.0, "volume_fraction": 0.3},
        "domain": {"size": 4.0, "periodic": True},
        "seed": 1,
        "voxels": {"shape": 10},
        "homogenization": {
            "method": "fft",
            "materials": {0: MATRIX.to_config(), 1: FIBRE.to_config()},
            "properties": ["bulk_modulus", "stiffness", "thermal_expansion"],
            "loadings": {"tension": {"strain": {11: 0.01}}, "heating": {"temperature": 100}},
            "fields": True,
        },
        "outputs": {"directory": str(tmp_path / "run"), "name": "s"},
    }
    outdir = pipeline.run(load_config(cfg))
    (row,) = pipeline.read_rows(outdir / "results.csv")
    assert row["tension.periodic.E11"] == 0.01 and abs(row["tension.periodic.S22"]) < 1e-9
    assert row["tension.periodic.S11"] / 0.01 == pytest.approx(row["stiffness.periodic.E1"])
    assert row["stiffness.periodic.S11"] == pytest.approx(1 / row["stiffness.periodic.E1"])
    assert row["heating.periodic.E11"] == pytest.approx(100 * row["thermal_expansion.periodic.A11"])
    assert row["solid_fraction"] == 1.0 and row["density"] == pytest.approx(
        row["fraction_0"] * 1.14 + row["fraction_1"] * 2.7
    )
    assert 0 < row["bulk_modulus.periodic.normalized"] <= 1.0
    assert row["bulk_modulus.periodic.relative"] == row["bulk_modulus.periodic.normalized"]
    long = pipeline.read_rows(outdir / "homogenization.csv")
    assert {r["property"] for r in long} == {"bulk_modulus", "tension", "heating"}
    assert (outdir / "s_0000_tension_periodic.vti").exists()
    bad = {**cfg["homogenization"], "loadings": {"stiffness": {"strain": {11: 1}}}}
    with pytest.raises(ConfigError, match="not a valid name"):
        load_config({**cfg, "homogenization": bad})
    bad = {**cfg["homogenization"], "loadings": {"x": {"strain": {11: 1}, "flux": {1: 0}}}}
    with pytest.raises(ConfigError, match=r"loadings\.x"):
        load_config({**cfg, "homogenization": bad})
    # the command line
    img = mg.VoxelImage(
        (np.random.default_rng(0).random((6, 6, 6)) < 0.4).astype(np.uint8), periodic=True
    )
    mg.io.save_image(img, tmp_path / "img.vti")
    args = ["homogenize", str(tmp_path / "img.vti"), "--material"]
    args += ["0=young:2.8,poisson:0.39,expansion:9.5e-5", "1=young:89,poisson:0.3,expansion:5.5e-6"]
    args += ["--property", "thermal_expansion", "--loading", "pull: S33=1", "--loading", "dT=1"]
    assert cli.main([*args, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["loadings"]["pull.periodic"]["S33"] == pytest.approx(1.0)
    heated = out["loadings"]["loading2.periodic"]
    assert heated["E11"] == pytest.approx(out["thermal_expansion.periodic"]["A11"])
    assert "thermal_expansion" in out["bounds"]
    assert cli.main([*args[:-2], "--loading", "X1=2"]) == 2
    assert "unknown component" in capsys.readouterr().err
