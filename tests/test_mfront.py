"""MFront behaviours through MGIS: the materials and configurations, the conventions of
MGIS (the factors sqrt(2) and the order (12, 13, 23) of the shear components of symmetric
tensors, the order of the components of gradients, the tangents, the rotations to the
axes of crystals), and morphoGen's behaviours against
their MFront twins (``tests/mfront``) at a material point and along paths, by finite
elements and on voxels (CPU and GPU), in small and finite strains. The twins need MGIS
and MFront (conda-forge); without them those tests are skipped."""

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from helpers import box_mesh
from morphogen import cli, gpu, io
from morphogen.config import load_config
from morphogen.homogenization import (
    CrystalPlasticity,
    Material,
    NonlinearFEProblem,
    NonlinearFFTProblem,
    Plasticity,
    simulate_point,
)
from morphogen.homogenization import mfront as mf
from morphogen.homogenization.materials import isotropic_matrix, rotate_stiffness
from morphogen.homogenization.mfront import MFrontBehaviour
from morphogen.image import VoxelImage
from morphogen.orientations import euler_to_matrix, random_orientations

TWINS = Path(__file__).parent / "mfront"
needs_mfront = pytest.mark.skipif(
    not (mf.available() and shutil.which("mfront")), reason="MGIS and MFront (conda-forge)"
)
PLASTICITY = Plasticity(yield_stress=100, hardening=500, isotropic=((50, 20),),
                        kinematic=((20000, 200), (5000, 0)), viscosity=50, exponent=5)  # fmt: skip
NATIVE = Material(young=70e3, poisson=0.33, plasticity=PLASTICITY)
HARD = Material(young=400e3, poisson=0.2)
CRYSTAL = Material.cubic(250000, 160000, 120000)
CRYSTAL_PLASTICITY = CrystalPlasticity.from_config(
    {"families": ["octahedral"], "tau0": 80, "Q": 20, "b": 10, "C": 20000, "D": 300, "K": 25,
     "n": 10}
)  # fmt: skip


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    """The libraries of the twins, compiled once."""
    cache = tmp_path_factory.mktemp("mfront")
    names = ("VonMisesTwin", "VonMisesHenckyTwin", "MericCailletaudTwin", "Elasticity",
             "OrthotropicElasticity")  # fmt: skip
    return {n: mf.compile_mfront(TWINS / f"{n}.mfront", directory=cache) for n in names}


def twin(twins, name, base=None, **kwargs):
    base = base or Material(young=70e3, poisson=0.33)
    behaviour = MFrontBehaviour(library=str(twins[name]), behaviour=name, **kwargs)
    return dataclasses.replace(base, mfront=behaviour)


def close(a, b, tol):
    """The curves of two paths: the same steps, stresses and cumulated plastic strains."""
    assert a.n_steps == b.n_steps
    assert np.allclose(b.stress, a.stress, rtol=0, atol=tol * np.abs(a.stress).max())
    assert np.allclose(b.plastic, a.plastic, rtol=0, atol=tol * max(np.abs(a.plastic).max(), 1e-12))


def test_mfront_materials(tmp_path):
    """MFront behaviours of materials: configurations, round trips, errors, the paths of
    the sources relative to the configuration file."""
    spec = {"source": "laws/VonMises.mfront", "material_properties": {"E": 2e5, "nu": 0.3},
            "parameters": {"K": 50}, "temperature": 300.0}  # fmt: skip
    b = MFrontBehaviour.from_config(spec)
    assert b.behaviour == "VonMises" and b.material_properties == (("E", 2e5), ("nu", 0.3))
    assert MFrontBehaviour.from_config(b.to_config()) == b
    m = Material.from_config({"young": 70e3, "poisson": 0.33, "mfront": spec})
    assert m.nonlinear and Material.from_config(m.to_config()) == m
    assert MFrontBehaviour.from_config("law.mfront").source == "law.mfront"
    for bad, match in [
        ({"library": "lib.so"}, "name of the behaviour"),
        ({"library": "lib.so", "source": "a.mfront"}, "library or the source"),
        ({"source": "a.mfront", "temperature": -1}, "absolute"),
        ({"source": "a.mfront", "colour": 1}, "unknown keys"),
    ]:
        with pytest.raises(ValueError, match=match):
            MFrontBehaviour.from_config(bad)
    with pytest.raises(ValueError, match="one of them"):
        Material(young=70e3, poisson=0.33, plasticity=PLASTICITY, mfront=spec)
    with pytest.raises(ValueError, match="elastic parameters"):
        Material(conductivity=1.0, mfront=spec)
    with pytest.raises(ValueError, match="void"):
        Material(void=True, mfront=spec)
    cfg = {
        "model": {"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.2},
        "domain": {"size": 2.0, "periodic": True},
        "voxels": {"shape": 8},
        "homogenization": {"method": "fft", "materials": {
            0: {"young": 7e4, "poisson": 0.33, "mfront": {"source": "laws/VonMises.mfront"}},
            1: {"young": 4e5, "poisson": 0.2}},
            "paths": {"tension": {"times": [0, 1], "strain": {"11": 0.01}}}},
    }  # fmt: skip
    homog = load_config(cfg, base_dir=tmp_path).homogenization
    assert homog.materials[0].mfront.source == str(tmp_path / "laws" / "VonMises.mfront")


def test_mfront_info(capsys):
    """morphogen info: MGIS and the MFront generator (TFEL), or why they are missing."""
    found = mf.info()
    assert found["mgis"] == mf.available()
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "mfront:" in out
    if not found["mgis"]:
        assert mf.INSTALL in out
    elif found["mfront"]:
        assert found["tfel"] and f"TFEL {found['tfel']} and MGIS" in out


def _random_symmetric(rng, n):
    a = rng.normal(size=(n, 3, 3))
    return 0.5 * (a + np.swapaxes(a, 1, 2))


def test_mgis_conventions():
    """The conversions to and from MGIS: TFEL's order of the shear components (12, 13, 23,
    with their factors sqrt(2)) against morphoGen's (23, 31, 12), the energy preserved,
    the tangents against morphoGen's isotropic, orthotropic and turned cubic stiffnesses,
    the order of the components of gradients and stresses of finite strains, and their
    rotations."""
    # distinct shears: morphoGen's engineering strains (23, 31, 12) to TFEL's slots
    root2 = np.sqrt(2)
    e = np.array([[0.0, 0.0, 0.0, 6.0, 4.0, 2.0]])  # gamma_23, gamma_31, gamma_12
    tfel = [[0, 0, 0, root2, 2 * root2, 3 * root2]]  # sqrt(2) eps_12, eps_13, eps_23
    assert np.allclose(mf._strains_in(e, None), tfel)
    voigt = [[0, 0, 0, 3, 2, 1]]  # sigma_23, sigma_31, sigma_12
    assert np.allclose(mf._stress_out(np.array([[0, 0, 0, 1, 2, 3]]) * root2, None), voigt)
    g12, g13, g23 = 1e4, 3e4, 2e4
    tangent = np.diag([1.0, 1.0, 1.0, 2 * g12, 2 * g13, 2 * g23])[None]  # MGIS's slots
    assert np.allclose(np.diag(mf._tangent_out(tangent, None)[0])[3:], [g23, g13, g12])
    rng = np.random.default_rng(0)
    n = 5
    sym = _random_symmetric(rng, n)
    e = mf._voigt(sym) * np.array([1, 1, 1, 2, 2, 2])  # engineering strains
    stress = _random_symmetric(rng, n)
    energy = np.einsum("nij,nij->n", stress, sym)
    assert np.allclose(np.einsum("ni,ni->n", mf._to_mgis(stress), mf._strains_in(e, None)), energy)
    assert np.allclose(mf._stress_out(mf._to_mgis(stress), None), mf._voigt(stress))
    g = euler_to_matrix(random_orientations(rng, n))
    turned = mf._stress_out(mf._to_mgis(mf._turn(stress, g, back=False)), g)
    assert np.allclose(turned, mf._voigt(stress))  # to the lattice and back
    # tangents in MGIS's components: isotropic, and a cubic crystal turned by its grain
    lam, mu = 1.2e5, 0.8e5
    iso = lam * np.outer([1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]) + 2 * mu * np.eye(6)
    for rot in (None, g):
        out = mf._tangent_out(np.repeat(iso[None], n, axis=0), rot)
        assert np.allclose(out, isotropic_matrix(lam, mu))
    c11, c12, c44 = 250e3, 160e3, 120e3
    cubic = np.zeros((6, 6))
    cubic[:3, :3] = c12
    cubic[[0, 1, 2], [0, 1, 2]] = c11
    euler = random_orientations(rng, n)
    in_mgis = cubic.copy()
    in_mgis[3:, 3:] = 2 * c44 * np.eye(3)
    out = mf._tangent_out(np.repeat(in_mgis[None], n, axis=0), euler_to_matrix(euler))
    assert np.allclose(out, rotate_stiffness(CRYSTAL.stiffness, euler))
    # finite strains: the order (11, 22, 33, 12, 21, 13, 31, 23, 32) of MGIS
    hg = rng.normal(size=(n, 9)) * 0.1
    f = np.eye(3) + hg.reshape(n, 3, 3)
    order = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    assert np.allclose(mf._gradients_in(hg, None), np.stack([f[:, i, j] for i, j in order], 1))
    p = rng.normal(size=(n, 9))
    rows = mf._pk1_out(p, None).reshape(n, 3, 3)
    assert np.allclose(np.stack([rows[:, i, j] for i, j in order], 1), p)
    # a linear map P = A : F turned to the lattice and back is the turned tangent
    a = rng.normal(size=(9, 9))
    a = a + a.T
    k = np.empty((9, 9))
    for r, (i, j) in enumerate(order):
        for c, (kk, ll) in enumerate(order):
            k[r, c] = a[3 * i + j, 3 * kk + ll]
    assert np.allclose(mf._dpdf_out(k[None], None)[0], a)
    turned = mf._dpdf_out(np.repeat(k[None], n, axis=0), g)
    fs = rng.normal(size=(n, 3, 3))
    fl = np.einsum("nip,npq,njq->nij", g, fs, g)  # to the axes of the lattice
    pl = (a @ fl.reshape(n, 9, 1)).reshape(n, 3, 3)
    ps = np.einsum("npi,npq,nqj->nij", g, pl, g)
    assert np.allclose(np.einsum("nij,nj->ni", turned, fs.reshape(n, 9)), ps.reshape(n, 9))
    # nonsymmetric tangents (for GMRES), every entry distinct: those of MGIS (TFEL's slots
    # 12, 13, 23, sqrt(2) on each shear, K_ab = w_a w_b D_PaPb) come back whole as
    # morphoGen's (23, 31, 12, engineering strains), or as their symmetric parts
    voigt = np.array([0, 1, 2, 5, 4, 3])  # morphoGen's slot of each slot of TFEL
    w = np.array([1.0, 1.0, 1.0, root2, root2, root2])
    d = rng.permutation(np.arange(1.0, 37.0)).reshape(6, 6) * 1e3
    k = np.outer(w, w) * d[np.ix_(voigt, voigt)]
    assert np.allclose(mf._tangent_out(k[None], None, symmetric=False)[0], d, rtol=1e-14)
    assert np.allclose(mf._tangent_out(k[None], None)[0], 0.5 * (d + d.T), rtol=1e-14)
    assert d[3, 5] != d[5, 3] and d[3, 4] != d[4, 3]  # the shears 23-12 and 23-31 apart
    out = mf._tangent_out(np.repeat(k[None], n, axis=0), g, symmetric=False)
    eps = rng.normal(size=(n, 6))  # engineering strains in the axes of the sample
    shears = np.array([1, 1, 1, 2, 2, 2])
    lattice = mf._turn(mf._tensors(eps), g, back=False)  # g eps g^T
    sig = np.einsum("ij,nj->ni", d, mf._voigt(lattice) * shears)  # of the lattice
    back = mf._turn(mf._tensors(sig * shears), g, back=True)  # g^T sigma g
    assert np.allclose(np.einsum("nij,nj->ni", out, eps), mf._voigt(back), rtol=1e-12)
    a = rng.normal(size=(9, 9))  # dP/dF without major symmetry
    k = np.empty((9, 9))
    for r, (i, j) in enumerate(order):
        for c, (kk, ll) in enumerate(order):
            k[r, c] = a[3 * i + j, 3 * kk + ll]
    assert np.array_equal(mf._dpdf_out(k[None], None, symmetric=False)[0], a)
    assert np.allclose(mf._dpdf_out(k[None], None)[0], 0.5 * (a + a.T))


@needs_mfront
def test_point_twins(twins):
    """morphoGen's von Mises plasticity (small and finite strains) and crystal plasticity
    (a turned crystal) against their MFront twins at a material point; material
    properties of a behaviour."""
    cyclic = {"times": [0, 10, 20, 30], "increments": [10, 5, 10],
              "strain": {"11": [0, 0.01, -0.01, -0.01]},
              "stress": {"12": [0, 0, 20, 20]}}  # fmt: skip
    mises = twin(twins, "VonMisesTwin")
    close(simulate_point(NATIVE, cyclic), simulate_point(mises, cyclic), 1e-8)
    finite = {"times": [0, 10, 20], "increments": [12, 8], "strain": {"11": [0, 0.3, 0.3],
              "12": [0, 0, 0.2]}}  # fmt: skip
    hencky = twin(twins, "VonMisesHenckyTwin")
    close(simulate_point(NATIVE, finite, kinematics="finite"),
          simulate_point(hencky, finite, kinematics="finite"), 1e-8)  # fmt: skip
    path = {"times": [0, 10, 20], "increments": [10, 10], "strain": {"11": [0, 0.01, -0.005]}}
    native = dataclasses.replace(CRYSTAL, crystal_plasticity=CRYSTAL_PLASTICITY)
    mfront = twin(twins, "MericCailletaudTwin", CRYSTAL)
    for euler in ((0.0, 0.0, 0.0), (30.0, 40.0, 50.0)):
        close(simulate_point(native, path, orientation=euler),
              simulate_point(mfront, path, orientation=euler), 1e-8)  # fmt: skip
    # the orientation reaches an orthotropic behaviour whatever the elastic parameters of
    # its material (isotropic here: those of the reference media only)
    loose = twin(twins, "MericCailletaudTwin", Material(young=2e5, poisson=0.3))
    close(simulate_point(native, path, orientation=(30.0, 40.0, 50.0)),
          simulate_point(loose, path, orientation=(30.0, 40.0, 50.0)), 1e-8)  # fmt: skip
    elastic = twin(twins, "Elasticity", material_properties={"E": 2e5, "nu": 0.25})
    res = simulate_point(elastic, {"times": [0, 1], "strain": {"11": 0.001}})
    assert res.stress[-1, 0] == pytest.approx(2e5 * 0.001)  # uniaxial stress
    assert res.strain[-1, 1] == pytest.approx(-0.25 * 0.001)


@needs_mfront
def test_shear_order(twins):
    """The order of the shear components: MGIS stores them as (12, 13, 23), the order of
    TFEL (that of Cast3M, Code_Aster and Abaqus too), morphoGen as (23, 31, 12), the usual
    order of Voigt. An orthotropic law whose shear moduli differ answers each slot
    of MGIS with its own modulus, and through morphoGen equals morphoGen's elasticity of
    the same stiffness, in the axes of the crystal and turned."""
    import mgis.behaviour as bv

    g12, g23, g13 = 1e4, 2e4, 3e4  # those of OrthotropicElasticity.mfront
    b = bv.load(str(twins["OrthotropicElasticity"]), "OrthotropicElasticity",
                bv.Hypothesis.Tridimensional)  # fmt: skip
    m = bv.MaterialDataManager(b, 3)
    for state in (m.s0, m.s1):
        bv.setExternalStateVariable(state, "Temperature", 293.15)
    m.s1.gradients[:] = 1e-3 * np.eye(6)[3:]  # sqrt(2) eps_ij in one slot of each point
    integration = bv.IntegrationType.IntegrationWithConsistentTangentOperator
    assert bv.integrate(m, integration, 1.0, 0, 3) >= 0
    expected = np.zeros((3, 6))
    expected[[0, 1, 2], [3, 4, 5]] = 2e-3 * np.array([g12, g13, g23])
    assert np.allclose(np.array(m.s1.thermodynamic_forces), expected, rtol=0, atol=1e-9)
    e1, e2, e3, n12, n23, n13 = 2e5, 1.5e5, 1e5, 0.3, 0.25, 0.2
    compliance = np.diag([1 / e1, 1 / e2, 1 / e3, 1 / g23, 1 / g13, 1 / g12])  # 23, 31, 12
    compliance[0, 1] = compliance[1, 0] = -n12 / e1
    compliance[0, 2] = compliance[2, 0] = -n13 / e1
    compliance[1, 2] = compliance[2, 1] = -n23 / e2
    native = Material(stiffness=np.linalg.inv(compliance))
    law = twin(twins, "OrthotropicElasticity", native)
    for slot, name, modulus in ((3, "23", g23), (4, "31", g13), (5, "12", g12)):
        res = simulate_point(law, {"times": [0, 1], "strain": {name: 1e-3}})
        assert np.allclose(res.stress[-1], modulus * 1e-3 * np.eye(6)[slot], rtol=0, atol=1e-9)
    path = {"times": [0, 10], "increments": 5, "strain": {"11": 1e-3, "22": -4e-4,
            "33": 2e-4, "23": 3e-3, "31": -2e-3, "12": 1e-3}}  # fmt: skip
    for euler in ((0.0, 0.0, 0.0), (30.0, 40.0, 50.0)):
        close(simulate_point(native, path, orientation=euler),
              simulate_point(law, path, orientation=euler), 1e-12)  # fmt: skip


@needs_mfront
def test_solver_twins(twins):
    """The twins along paths: a composite with a viscoplastic matrix on voxels (B-bar,
    F-bar) and by finite elements (periodic and kinematic conditions), small and finite
    strains, with the fields; a voxel polycrystal with crystal plasticity."""
    rng = np.random.default_rng(2)
    image = VoxelImage((rng.random((6, 5, 7)) < 0.3).astype(np.uint8), spacing=(0.5, 0.5, 0.5))
    small = {"times": [0, 10, 20], "increments": [6, 4], "strain": {"11": [0, 0.01, 0.002]}}
    large = {"times": [0, 10], "increments": 8, "strain": {"11": [0, 0.1]}}
    cases = (("small", "VonMisesTwin", small), ("finite", "VonMisesHenckyTwin", large))
    for kin, name, path in cases:
        a, b = (NonlinearFFTProblem(image, {0: m, 1: HARD}, kinematics=kin).run(path, fields=True)
                for m in (NATIVE, twin(twins, name)))  # fmt: skip
        close(a, b, 1e-9)
        fa, fb = a.fields[10.0], b.fields[10.0]
        assert np.allclose(np.nan_to_num(fb["p"]), np.nan_to_num(fa["p"]), atol=1e-9)
    mesh = box_mesh((2, 2, 2), phase=lambda c: (c[:, 0] > 0.6).astype(int), perturb=0.2)
    fe_large = {"times": [0, 10], "increments": 6, "strain": {"11": [0, 0.05]}}
    cases = (("small", "VonMisesTwin", small), ("finite", "VonMisesHenckyTwin", fe_large))
    for kin, name, path in cases:
        for bc in ("periodic", "kubc"):
            a, b = (NonlinearFEProblem(mesh, {0: m, 1: HARD}, bc, kinematics=kin).run(path)
                    for m in (NATIVE, twin(twins, name)))  # fmt: skip
            close(a, b, 1e-9)
    n = 6
    centres = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) / n
    seeds = rng.random((4, 3))
    d = centres[:, None, :] - seeds[None, :, :]
    d -= np.round(d)
    grains = np.argmin((d**2).sum(-1), axis=1).reshape(n, n, n)
    poly = VoxelImage(np.zeros((n, n, n), dtype=np.uint8), spacing=(1 / n,) * 3)
    orientations = random_orientations(rng, 4)
    path = {"times": [0, 10], "increments": 6, "strain": {"33": 0.01}}
    a, b = (NonlinearFFTProblem(poly, {0: m}, grains=grains, orientations=orientations).run(path)
            for m in (dataclasses.replace(CRYSTAL, crystal_plasticity=CRYSTAL_PLASTICITY),
                      twin(twins, "MericCailletaudTwin", CRYSTAL)))  # fmt: skip
    close(a, b, 1e-9)


@needs_mfront
def test_tangent_symmetry_of_mfront_behaviours(twins, tmp_path):
    """Whether an MFront behaviour has a symmetric tangent: the trait of its
    tridimensional hypothesis in the header MFront writes beside the library
    (``@ProvidesSymmetricTangentOperator``: the elastic laws; not the implicit twins);
    without that header, not symmetric. The solvers keep the conjugate gradients for the
    first, and keep the full tangents of the others (as those of their native twins, whose
    Armstrong-Frederick hardening and slip hardening make them nonsymmetric), which
    ``tangent="auto"`` measures on voxels (symmetrized below ``ASYMMETRY``); forced onto
    the symmetrized tangents, the twin of the crystal gives the same curve within the
    tolerance of Newton's method."""
    expected = {"Elasticity": True, "OrthotropicElasticity": True, "VonMisesTwin": False,
                "VonMisesHenckyTwin": False, "MericCailletaudTwin": False}  # fmt: skip
    for name, symmetric in expected.items():
        assert mf.declared_symmetric(twins[name], name) is symmetric, name
    alone = tmp_path / twins["Elasticity"].name
    shutil.copy(twins["Elasticity"], alone)
    assert mf.declared_symmetric(alone, "Elasticity") is None
    rng = np.random.default_rng(2)
    image = VoxelImage((rng.random((4, 3, 5)) < 0.3).astype(np.uint8))
    elastic = twin(twins, "Elasticity", material_properties={"E": 7e4, "nu": 0.33})
    assert NonlinearFFTProblem(image, {0: elastic, 1: HARD}).symmetrize
    for law in (NATIVE, twin(twins, "VonMisesTwin")):
        assert not NonlinearFFTProblem(image, {0: law, 1: HARD}).symmetrize
    n = 4
    grains = np.arange(n**3).reshape(n, n, n) % 3
    poly = VoxelImage(np.zeros((n, n, n), dtype=np.uint8), spacing=(1 / n,) * 3)
    orientations = random_orientations(rng, 3)
    path = {"times": [0, 10], "increments": 4, "strain": {"33": 0.01}}
    mfront = twin(twins, "MericCailletaudTwin", CRYSTAL)
    full, sym = (NonlinearFFTProblem(poly, {0: mfront}, grains=grains, orientations=orientations,
                                     tangent=tangent).run(path)
                 for tangent in ("auto", "symmetric"))  # fmt: skip
    close(full, sym, 1e-5)


@needs_mfront
@pytest.mark.skipif(not gpu.available(), reason="no usable GPU (CuPy, CUDA)")
def test_gpu_twins(twins):
    """The points of MFront behaviours on the GPU path of the voxels (MGIS on the CPU)."""
    rng = np.random.default_rng(4)
    image = VoxelImage((rng.random((6, 5, 7)) < 0.3).astype(np.uint8))
    path = {"times": [0, 10], "increments": 6, "strain": {"11": 0.01}}
    m = twin(twins, "VonMisesTwin")
    a, b = (NonlinearFFTProblem(image, {0: m, 1: HARD}, device=device).run(path)
            for device in ("cpu", "gpu"))  # fmt: skip
    close(a, b, 1e-12)


@needs_mfront
def test_mfront_errors(twins, tmp_path):
    """Behaviours of the other kinematics, missing material properties, unknown
    parameters, sources that do not compile."""
    with pytest.raises(ValueError, match="STANDARDFINITESTRAINBEHAVIOUR"):
        simulate_point(twin(twins, "VonMisesTwin"), {"times": [0, 1], "strain": {"11": 0.01}},
                       kinematics="finite")  # fmt: skip
    with pytest.raises(ValueError, match="material properties"):
        simulate_point(twin(twins, "Elasticity"), {"times": [0, 1], "strain": {"11": 1e-3}})
    with pytest.raises(ValueError, match="no parameters"):
        simulate_point(twin(twins, "VonMisesTwin", parameters={"nothing": 1.0}),
                       {"times": [0, 1], "strain": {"11": 1e-3}})  # fmt: skip
    broken = tmp_path / "Broken.mfront"
    broken.write_text("@DSL Implicit;\n@Behaviour Broken;\n@Nonsense;\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not compile"):
        mf.compile_mfront(broken, directory=tmp_path / "cache")


@needs_mfront
def test_mfront_command_line(twins, tmp_path):
    """morphogen homogenize with an MFront material: its source relative to the file of
    the materials, compiled on first use."""
    rng = np.random.default_rng(5)
    image = VoxelImage((rng.random((5, 5, 5)) < 0.3).astype(np.uint8))
    io.save_image(image, tmp_path / "c.vti")
    (tmp_path / "laws").mkdir()
    shutil.copy(TWINS / "VonMisesTwin.mfront", tmp_path / "laws")
    mats = {0: {"young": 70e3, "poisson": 0.33, "mfront": {"source": "laws/VonMisesTwin.mfront"}},
            1: HARD.to_config()}  # fmt: skip
    (tmp_path / "mats.json").write_text(json.dumps(mats), encoding="utf-8")
    report = tmp_path / "r.json"
    code = cli.main(["homogenize", str(tmp_path / "c.vti"), "--materials",
                     str(tmp_path / "mats.json"), "--path", "tension: t=0,10 n=5 E11=0,0.01",
                     "--curves", str(tmp_path / "curves"), "-o", str(report)])  # fmt: skip
    assert code == 0 and (tmp_path / "curves").exists()
