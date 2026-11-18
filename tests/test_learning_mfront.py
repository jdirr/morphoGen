"""MFront sources of the learned laws (morphogen.learning.mfront): the sources of the four
models (their trained parameters as constants, to the bit; the strain measures, the
symmetry, the state variables, the thermal expansion in morphoGen's components; the
errors; ``morphogen mfront``); compiled by MFront, the behaviours integrated by MGIS
against morphoGen's own integration of the same laws along paths of the six components,
in small strains and in finite strains (laws of the logarithmic and of the Green-Lagrange
strain), heated and turned, and in tensions whose free components need the tangent;
their tangents against finite differences in TFEL's components; the shear order of TFEL
(12, 13, 23, with sqrt(2)) against morphoGen's (23, 31, 12) on a law whose three shear
moduli differ; the laws of an AMITEX export (umat) and of an Abaqus input (abaqus), small
and finite strains, called as the codes call them. The compiled behaviours need MGIS and
MFront (conda-forge); without them those tests are skipped."""

import ctypes
import dataclasses
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from morphogen import cli, io
from morphogen.homogenization import mfront as mf
from morphogen.homogenization.mfront import MFrontBehaviour
from morphogen.homogenization.nonlinear import simulate_point
from morphogen.image import VoxelImage
from morphogen.io.mfront import (
    ABAQUS_AXES,
    mfront_source,
    mfront_state_size,
    mfront_strain_measure,
    mfront_symmetry,
    write_mfront,
)
from morphogen.learning import (
    FeedForward,
    Potential,
    Recurrent,
    Standard,
    generate,
    learned_material,
    random_paths,
)
from morphogen.learning import standard as std
from morphogen.learning.mfront import _port
from morphogen.orientations import random_orientations

needs_mfront = pytest.mark.skipif(
    not (mf.available() and shutil.which("mfront")), reason="MGIS and MFront (conda-forge)"
)
#: An orthotropic stiffness in morphoGen's Voigt order (11, 22, 33, 23, 31, 12): its three
#: shear moduli differ (G23 = 60, G31 = 40, G12 = 20 GPa), so that an exchange of shear
#: components shows
ORTHOTROPIC = np.array([[250.0, 100.0, 80.0, 0, 0, 0], [100.0, 200.0, 90.0, 0, 0, 0],
                        [80.0, 90.0, 150.0, 0, 0, 0], [0, 0, 0, 60.0, 0, 0],
                        [0, 0, 0, 0, 40.0, 0], [0, 0, 0, 0, 0, 20.0]]) * 1000.0  # fmt: skip
#: Three principal expansions in the axes of the material
EXPANSION = (1e-5, 2e-5, 3e-5)
EULER = (30.0, 50.0, 70.0)
#: the six strain components, small and finite strains (the upper triangle of the gradient)
SIX = {"times": [0, 10, 20], "increments": [10, 10],
       "strain": {"11": [0, 0.003, -0.0015], "22": [0, -0.001, 0.0], "33": [0, -0.0012, 0.0006],
                  "23": [0, 0.0012, 0.0012], "31": [0, 0.0006, 0.0],
                  "12": [0, -0.0003, 0.0009]}}  # fmt: skip
SIX_FINITE = {"times": [0, 10, 20], "increments": [10, 10],
              "strain": {"11": [0, 0.04, 0.04], "22": [0, -0.01, -0.01],
                         "33": [0, -0.015, -0.005], "23": [0, 0.005, 0.015],
                         "31": [0, 0.0, 0.01], "12": [0, 0.01, -0.005]}}  # fmt: skip
#: the changes of temperature of the heated paths
HEAT = {"small": [0, 60, 20], "finite": [0, 300, 600]}
#: tensions whose other components are free (zero stress)
TENSION = {"small": {"times": [0, 10], "increments": 10, "strain": {"11": [0, 0.003]}},
           "finite": {"times": [0, 10], "increments": 10, "strain": {"11": [0, 0.04]}}}  # fmt: skip
#: the models (small networks), of the strain of finite strains, with or without the time
#: step (recurrent, standard: rate-independent and viscous dissipations)
MODELS = {
    "feedforward": lambda strain, rate: FeedForward((8, 5), strain=strain),
    "potential": lambda strain, rate: Potential((7, 5), strain=strain),
    "recurrent": lambda strain, rate: Recurrent(6, rate=rate, strain=strain),
    "standard": lambda strain, rate: Standard(5, internal=3, rate=rate, strain=strain),
}
#: the laws compiled: the kinematics, the strain of the model, whether it expands (the law
#: of the Green-Lagrange strain does not: refused) and the time step as an input
CASES = [("small", "logarithmic", True, True), ("finite", "logarithmic", True, False),
         ("finite", "green-lagrange", False, False)]  # fmt: skip


@pytest.fixture(scope="module")
def data():
    """The responses of the orthotropic crystal in its axes along random walks, in small
    and finite strains (Hencky's elasticity)."""
    crystal = {"stiffness": ORTHOTROPIC.tolist()}
    return {
        "small": generate(crystal, random_paths(4, segments=2, increments=3, amplitude=0.004,
                                                seed=1)),
        "finite": generate(crystal, random_paths(4, segments=2, increments=3, amplitude=0.05,
                                                 seed=1), kinematics="finite"),
    }  # fmt: skip


def _model(kind, data, strain="logarithmic", rate=False, seed=2):
    """A model set up on ``data``, its parameters moved off their start (its network on)."""
    model = MODELS[kind](strain, rate)
    model.setup(data)
    rng = np.random.default_rng(seed)
    for k, v in model.params.items():
        model.params[k] = v + 0.1 * rng.normal(size=v.shape)
    return model


def _array(text, name):
    """The values of the constant array ``name`` of a generated source."""
    body = re.search(rf"constexpr (?:double|int) {name}\[[^=]*= \{{(.*?)\}};", text, re.S)
    return np.array([float(v) for v in
                     re.findall(r"-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", body.group(1))])  # fmt: skip


def _scalar(text, name):
    return float(re.search(rf"constexpr \w+ {name} = ([^;]+);", text).group(1))


def test_learned_sources(data, tmp_path):
    """The sources of the four models: their scales and trained parameters as constants,
    to the bit; the strain measure (Hencky's for the laws of the logarithmic strain,
    Green-Lagrange's for the others), the symmetry (orthotropic in the axes of the material,
    the axes of Abaqus in finite strains), the tangents declared symmetric for the
    potential and the standard model, the state of the recurrent and standard models, the
    thermal expansion in morphoGen's components (engineering shears); morphoGen's C++
    port, a law per section; the errors; ``morphogen mfront``."""
    for kind in MODELS:
        port = _port(kind)
        assert port.count("inline int law(") == 1 and "// ---- common" in port
        for kinematics, strain, hot, rate in CASES:
            model = _model(kind, data[kinematics], strain, rate)
            material = learned_material(model, expansion=EXPANSION if hot else None)
            text = mfront_source(material, "Law", kinematics=kinematics)
            measure = {"small": None, "logarithmic": "Hencky",
                       "green-lagrange": "GreenLagrange"}[strain if kinematics == "finite"
                                                          else "small"]  # fmt: skip
            assert mfront_strain_measure(material, kinematics) == measure
            assert (f"@StrainMeasure {measure};" in text) == (measure is not None)
            assert ("@StrainMeasure" in text) == (measure is not None)
            assert mfront_symmetry(material, kinematics) == "orthotropic"
            assert "@OrthotropicBehaviour;" in text and "@DSL Default;" in text
            assert (ABAQUS_AXES in text) == (kinematics == "finite")
            symmetric = kind in ("potential", "standard")
            assert ("@ProvidesSymmetricTangentOperator;" in text) == symmetric
            assert ("@ProvidesTangentOperator;" in text) != symmetric
            assert mfront_state_size(material, kinematics) == model.n_state
            assert (f"@AuxiliaryStateVariable real lv[{model.n_state}];" in text) == (
                model.n_state > 0
            )
            alpha = np.array([*EXPANSION, 0, 0, 0]) if hot else np.zeros(6)
            assert np.array_equal(_array(text, "EXPANSION"), alpha)
            assert ("(T + dT - 293.15)" in text) == hot
            s = model.scales
            for name, value in (
                ("SCALE_STRAIN", s.strain),
                ("SCALE_STRESS", s.stress),
                ("SCALE_INCREMENT", s.increment),
                ("SCALE_TIME", s.time),
            ):
                assert _scalar(text, name) == value  # to the bit
            p = model.params
            if kind == "feedforward":
                expected = {"W": np.concatenate([p["W0"].ravel(), p["W1"].ravel()]),
                            "B": np.concatenate([p["b0"], p["b1"]]), "WOUT": p["Wout"],
                            "BOUT": p["bout"], "LIN": p["L"]}  # fmt: skip
            elif kind == "potential":
                w, wout, amat = model._weights()
                expected = {"U": np.concatenate([p["U0"].ravel(), p["U1"].ravel()]),
                            "BIAS": np.concatenate([p["b0"], p["b1"]]), "WPOS": w[1],
                            "WOUT": wout, "AMAT": amat,
                            "G0": model._gradient(np.zeros((1, 6)))[0]}  # fmt: skip
                assert np.array_equal(_array(text, "OFFSETS"), [0, 7])
            elif kind == "recurrent":
                expected = {"WI": p["Wi"], "WH": p["Wh"], "BG": p["b"], "BN": p["bn"],
                            "V": p["V"], "Q": p["Q"], "C": p["c"]}  # fmt: skip
                assert _scalar(text, "N_INPUTS") == (13 if rate else 12)
                assert f"constexpr bool LOGARITHMIC = {str(model.logarithmic).lower()};" in text
            else:
                u, b, w, amat, g0, _, bm, gm, pm, cm, vm, delta, *_ = model._values()
                expected = {"U": u, "BIAS": b, "WR": w, "A": amat, "G0": g0, "BM": bm,
                            "GM": gm, "P": pm, "CM": cm, "VM": vm}  # fmt: skip
                assert _scalar(text, "DELTA") == delta
                assert _scalar(text, "NEWTON_TOL") == std.NEWTON_TOL
                assert f"constexpr bool VISCOUS = {str(rate).lower()};" in text
            for name, value in expected.items():
                assert np.array_equal(_array(text, name), np.ravel(value)), (kind, name)
    # an expansion tensor with shear components: alpha_12 in morphoGen's slot 12, doubled
    model = _model("potential", data["small"])
    sheared = ((1e-5, 1e-6, 0.0), (1e-6, 2e-5, 0.0), (0.0, 0.0, 3e-5))
    text = mfront_source(learned_material(model, expansion=sheared), "Sheared")
    assert np.array_equal(_array(text, "EXPANSION"), [1e-5, 2e-5, 3e-5, 0.0, 0.0, 2e-6])
    # a law in the axes of the sample: isotropic, its isotropic expansion
    still = learned_material({"model": model, "oriented": False}, expansion=1.2e-5)
    text = mfront_source(still, "Still")
    assert "@OrthotropicBehaviour;" not in text and mfront_symmetry(still) == "isotropic"
    assert np.array_equal(_array(text, "EXPANSION"), [1.2e-5] * 3 + [0.0] * 3)
    # morphogen mfront: the learned phase of a materials file (its model relative to it)
    model.save(tmp_path / "law.npz")
    materials = {0: {"young": 1e5, "poisson": 0.3, "learned": "law.npz"},
                 1: {"young": 1e5, "poisson": 0.3}}  # fmt: skip
    (tmp_path / "mats.json").write_text(json.dumps(materials), encoding="utf-8")
    assert cli.main(["mfront", "--materials", str(tmp_path / "mats.json"), "-o",
                     str(tmp_path / "cli")]) == 0  # fmt: skip
    text = (tmp_path / "cli" / "matsPhase0.mfront").read_text(encoding="utf-8")
    assert "@Behaviour matsPhase0;" in text and "of law.npz" in text
    assert not (tmp_path / "cli" / "matsPhase1.mfront").exists()
    gl = _model("feedforward", data["finite"], "green-lagrange")
    for call, match in [
        (lambda: mfront_source(learned_material(model), "Small", kinematics="finite"),
         "green-lagrange data"),
        (lambda: mfront_source(learned_material(gl), "Finite"), "small data"),
        (lambda: mfront_source(learned_material(gl, expansion=1e-5), "Hot", kinematics="finite"),
         "Green-Lagrange strain that expands"),
        (lambda: mfront_source(learned_material({"model": model, "oriented": False},
                                                expansion=EXPANSION), "Turned"),
         "axes of the sample"),
    ]:  # fmt: skip
        with pytest.raises(ValueError, match=match):
            call()


@pytest.fixture(scope="module")
def compiled(data, tmp_path_factory):
    """The learned materials of each model and case (:data:`CASES`, in the axes of the
    crystal) and their MFront laws, compiled by MFront for MGIS (``generic``; in the cache
    of morphoGen's MFront laws, by the hash of their sources)."""
    directory = tmp_path_factory.mktemp("learned")
    out = {}
    for kind in MODELS:
        for kinematics, strain, hot, rate in CASES:
            model = _model(kind, data[kinematics], strain, rate)
            material = learned_material(model, expansion=EXPANSION if hot else None)
            name = f"{kind.capitalize()}{kinematics.capitalize()}{'GL' if not hot else ''}"
            source = write_mfront(material, directory / f"{name}.mfront", kinematics=kinematics)
            library = mf.compile_mfront(source)
            law = dataclasses.replace(material, learned=None, expansion=None,
                                      mfront=MFrontBehaviour(library=str(library),
                                                             behaviour=name))  # fmt: skip
            out[kind, kinematics, strain] = (material, law, hot)
    return out


def _gap(a, b):
    """The largest gap of the stresses of two curves, relative to the largest stress."""
    assert np.array_equal(a.times, b.times)
    return float(np.abs(a.stress - b.stress).max() / np.abs(a.stress).max())


@needs_mfront
@pytest.mark.parametrize("kind", list(MODELS))
def test_learned_behaviours(compiled, kind):
    """The compiled laws against morphoGen's integration of the same learned laws at a
    material point, along paths of the six components: small strains, finite strains (the
    laws of the logarithmic strain through MFront's Hencky measure, those of the
    Green-Lagrange strain through its own), heated (the thermal split of the Hencky points
    in finite strains), turned (the orthotropic law in the axes of the crystal); to the
    round-off (measured: below 6e-16 in small strains, 2e-14 in finite strains). Then
    tensions whose free components Newton's method solves for with the tangent, to its
    tolerance and with the same iterations: MGIS through morphoGen gives the exact
    tangents at a material point, also those of the perceptron and of the recurrent
    network, not symmetric (their symmetric parts did not converge)."""
    for kinematics, strain, hot, _ in CASES:
        material, law, _ = compiled[kind, kinematics, strain]
        six = SIX if kinematics == "small" else SIX_FINITE
        tol = 1e-13 if kinematics == "small" else 1e-12
        paths = [six, {**six, "temperature": HEAT[kinematics]}] if hot else [six]
        for path in paths:
            for euler in (None, EULER):
                a = simulate_point(material, path, kinematics=kinematics, orientation=euler)
                b = simulate_point(law, path, kinematics=kinematics, orientation=euler)
                assert _gap(a, b) < tol, (kinematics, strain, path.get("temperature"), euler)
        a = simulate_point(material, TENSION[kinematics], kinematics=kinematics, orientation=EULER)
        b = simulate_point(law, TENSION[kinematics], kinematics=kinematics, orientation=EULER)
        assert _gap(a, b) < 1e-9 and a.iterations.sum() == b.iterations.sum()
        assert np.allclose(a.strain, b.strain, rtol=0, atol=1e-9 * np.abs(a.strain).max())


@needs_mfront
def test_learned_laws_on_voxels(compiled):
    """The compiled laws beside a stiff phase on voxels, a tension whose lateral stresses
    are free: MGIS through morphoGen reads whether a law declares its tangent symmetric
    (:func:`~morphogen.homogenization.mfront.declared_symmetric`), so that the potential
    and the standard model keep the conjugate gradients, and the perceptron and the
    recurrent network their full tangents, solved by GMRES: they follow morphoGen's own
    learned points with the same Newton iterations, where their symmetric parts (forced)
    take more iterations, or fail."""
    from morphogen.homogenization import Material, NonlinearFFTProblem

    stiff = Material(young=400e3, poisson=0.2)
    rng = np.random.default_rng(3)
    image = VoxelImage((rng.random((3, 3, 3)) < 0.3).astype(np.uint8))
    path = TENSION["small"]
    for kind in MODELS:
        material, law, _ = compiled[kind, "small", "logarithmic"]
        symmetric = kind in ("potential", "standard")
        assert mf.declared_symmetric(law.mfront.library, law.mfront.behaviour) is symmetric
        problems = [NonlinearFFTProblem(image, {0: m, 1: stiff}) for m in (material, law)]
        assert all(p.symmetrize is symmetric for p in problems)
        if symmetric:
            continue
        a, b = (p.run(path) for p in problems)
        assert _gap(a, b) < 1e-9 and a.iterations.sum() == b.iterations.sum(), kind
        try:
            forced = NonlinearFFTProblem(image, {0: law, 1: stiff}, tangent="symmetric",
                                         max_cuts=4).run(path)  # fmt: skip
        except RuntimeError:  # no convergence
            continue
        assert forced.iterations.sum() > 1.5 * b.iterations.sum(), kind


def _mgis_step(law, g0, g1, isv0, dt, heat):
    """A step of a compiled law through MGIS: its thermodynamic forces at the end, its
    tangent and its internal variables, in MGIS's components (TFEL's order 11, 22, 33, 12,
    13, 23 with sqrt(2) on the shears; F and P: 11, 22, 33, 12, 21, 13, 31, 23, 32); ``heat``:
    morphoGen's changes of temperature at the start and at the end."""
    mgis = mf._mgis()
    bv = mgis.behaviour
    loaded = mf.load_behaviour(law.mfront, finite=len(g0) == 9)
    m = bv.MaterialDataManager(loaded.behaviour, 1)
    m.s0.gradients[:] = np.asarray(g0)[None]
    m.s1.gradients[:] = np.asarray(g1)[None]
    if loaded.n_isv:
        m.s0.internal_state_variables[:] = np.asarray(isv0)[None]
    bv.setExternalStateVariable(m.s0, "Temperature", 293.15 + heat[0])
    bv.setExternalStateVariable(m.s1, "Temperature", 293.15 + heat[1])
    integration = bv.IntegrationType.IntegrationWithConsistentTangentOperator
    assert bv.integrate(mgis.ThreadPool(1), m, integration, float(dt)) >= 0
    n = len(g0)
    isv = np.array(m.s1.internal_state_variables)[0] if loaded.n_isv else np.zeros(0)
    return np.array(m.s1.thermodynamic_forces)[0], np.array(m.K).reshape(n, n), isv


@needs_mfront
def test_learned_tangents(compiled, data, tmp_path):
    """The tangents of the compiled laws (``Dt``, turned by MFront to the operator of MGIS:
    dsigma/deps in small strains, dP/dF in finite strains) against central differences of
    their stresses in MGIS's components, from a state of the model off zero and heated;
    and the shear order: a linear law (the perceptron's network silenced: its linear term
    the orthotropic stiffness, fitted) answers TFEL's slots 12, 13, 23 with 2 G12, 2 G13
    and 2 G23 (20, 40 and 60 GPa: TFEL's components carry sqrt(2) on the shears)."""
    rng = np.random.default_rng(5)
    worst = 0.0
    for (kind, kinematics, strain), (material, law, hot) in compiled.items():
        n = 6 if kinematics == "small" else 9
        model = material.learned.load()
        if n == 6:  # TFEL's components of two small strains
            g0, g1 = 1e-3 * rng.normal(size=6), 1e-3 * rng.normal(size=6)
            h = 1e-7
        else:  # MGIS's components of two gradients, the lower triangle too
            g0 = np.r_[1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0] + 0.02 * rng.normal(size=9)
            g1 = g0 + 0.01 * rng.normal(size=9)
            h = 1e-6
        isv0 = 0.3 * rng.normal(size=model.n_state)
        heat = (20.0, 50.0) if hot else (0.0, 0.0)
        _, k, _ = _mgis_step(law, g0, g1, isv0, 0.5, heat)
        fd = np.empty((n, n))
        for j in range(n):
            d = np.zeros(n)
            d[j] = h
            up = _mgis_step(law, g0, g1 + d, isv0, 0.5, heat)[0]
            down = _mgis_step(law, g0, g1 - d, isv0, 0.5, heat)[0]
            fd[:, j] = (up - down) / (2 * h)
        gap = np.abs(fd - k).max() / np.abs(k).max()
        assert gap < 1e-6, (kind, kinematics, strain, gap)
        worst = max(worst, gap)
    # the shear order: a linear law of the orthotropic stiffness (its network silenced)
    model = FeedForward((8, 5))
    model.setup(data["small"])
    model.params["Wout"][:] = 0.0
    linear = learned_material(model)
    source = write_mfront(linear, tmp_path / "Linear.mfront")
    library = mf.compile_mfront(source, directory=tmp_path / "c")
    law = dataclasses.replace(
        linear, learned=None, mfront=MFrontBehaviour(library=str(library), behaviour="Linear")
    )
    _, k, _ = _mgis_step(law, np.zeros(6), 1e-3 * rng.normal(size=6), [], 1.0, (0.0, 0.0))
    g23, g31, g12 = np.diag(ORTHOTROPIC)[3:]
    assert np.allclose(np.diag(k)[3:], [2 * g12, 2 * g31, 2 * g23], rtol=1e-10)
    assert np.allclose(k[:3, :3], ORTHOTROPIC[:3, :3], rtol=1e-10)
    off = k - np.diag(np.diag(k))
    off[:3, :3] = 0.0
    assert np.abs(off).max() < 1e-10 * np.abs(k).max()
    # turned, a shear of the sample: morphoGen's turned crystal
    for component in ("23", "31", "12"):
        shear = {"times": [0, 1], "increments": 1,
                 "strain": {c: [0, 0.002 if c == component else 0.0]
                            for c in ("11", "22", "33", "23", "31", "12")}}  # fmt: skip
        a = simulate_point(linear, shear, orientation=EULER)
        b = simulate_point(law, shear, orientation=EULER)
        assert _gap(a, b) < 1e-13
    assert worst < 1e-6


def test_unoriented_learned_laws_in_abaqus(data, tmp_path):
    """A learned law in the axes of the sample (``oriented: false``) runs unturned in an
    Abaqus input of small strains, as morphoGen runs it: the sections of its grains take no
    *ORIENTATION, those of a law in the axes of its material do."""
    from helpers import box_mesh

    model = _model("feedforward", data["small"])
    mesh = box_mesh((1, 1, 1), phase=lambda c: 1 + (c[:, 0] > 0.5))
    euler = np.array([[30.0, 40.0, 50.0], [112.0, 63.0, 201.0]])
    mesh = dataclasses.replace(mesh, cell=mesh.phase.copy() - 1, orientations=euler)
    materials = {1: learned_material({"model": model, "oriented": False}),
                 2: learned_material({"model": model})}  # fmt: skip
    text = io.save_mesh(mesh, tmp_path / "ab.inp", materials=materials).read_text(encoding="utf-8")
    sections = re.findall(r"\*SOLID SECTION, ELSET=CELL_(\d+), MATERIAL=[^,\s]+(.*)", text)
    assert sorted(sections) == [("0", ""), ("1", ", ORIENTATION=ORI_1")]


@needs_mfront
def test_learned_laws_in_amitex_and_abaqus(data, tmp_path):
    """Learned phases of an AMITEX export (the umat interface) and of an Abaqus input (the
    abaqus interface), in small and finite strains: their laws compiled as the export says
    and called as the codes call them (the coefficients and axes of each zone, AMITEX's
    components; the *ORIENTATION of the sections in Abaqus, and in finite strains the axes
    of its orthotropic laws in their first six state variables) against morphoGen's
    integration of the learned laws, each grain turned."""
    from helpers import box_mesh
    from test_mfront_export import (
        _abaqus_finite,
        _amitex_finite,
        _dgibi_sizes,
        _umat,
        _umat_curve,
        _zone_coefficients,
    )

    small = {1: learned_material(_model("feedforward", data["small"])),
             2: learned_material(_model("recurrent", data["small"], rate=True)),
             3: learned_material(_model("standard", data["small"]))}  # fmt: skip
    rng = np.random.default_rng(4)
    phase = rng.integers(1, 4, size=(4, 4, 4)).astype(np.uint8)
    grains = rng.integers(0, 3, size=phase.shape)
    orientations = random_orientations(rng, 3)
    io.write_amitex(tmp_path / "amitex", VoxelImage(phase), small, grains=grains,
                    orientations=orientations, paths={"t": SIX}, name="ln")  # fmt: skip
    directory = tmp_path / "amitex"
    script = (directory / "ln_run.sh").read_text(encoding="utf-8")
    build = re.search(r"\(cd mfront && (mfront [^)]*)\)", script).group(1).split()
    subprocess.run(build, cwd=directory / "mfront", check=True, capture_output=True)
    library = ctypes.CDLL(str(directory / "mfront" / "src" / "libUmatBehaviour.so"))
    root = ET.parse(directory / "ln_material.xml").getroot()
    manifest = json.loads((directory / "ln_amitex.json").read_text(encoding="utf-8"))
    for el, entry in zip(root.findall("Material"), manifest["materials"], strict=True):
        behaviour = entry["mfront"].split("/")[1].removesuffix(".mfront")
        coeffs = {int(c.get("Index")): c for c in el.findall("Coeff")}
        n_props, n_state = _dgibi_sizes(directory / "mfront" / "castem" / f"{behaviour}.dgibi")
        assert sorted(coeffs) == list(range(1, n_props + 1))
        assert len(el.findall("IntVar")) == max(n_state, 1) == entry["internal_variables"]
        for zone in range(entry["zones"]):
            props = _zone_coefficients(directory, coeffs, n_props, zone)
            native = simulate_point(
                small[entry["phase"]], SIX, orientation=orientations[entry["grains"][zone]]
            )
            step = _umat(library, el.get("Law"), props, max(n_state, 1))
            stress = _umat_curve(step, native.strain, native.times)
            assert np.allclose(stress, native.stress, rtol=0,
                               atol=1e-12 * np.abs(native.stress).max())  # fmt: skip
    # Abaqus in small strains: the sections of the grains in their *ORIENTATION
    mesh = box_mesh((1, 1, 1), phase=lambda c: 1 + (c[:, 0] > 0.5) + (c[:, 1] > 0.5))
    euler = np.array([[0.0, 0.0, 0.0], [30.0, 40.0, 50.0], [112.0, 63.0, 201.0],
                      [10.0, 20.0, 30.0]])  # fmt: skip
    mesh = dataclasses.replace(mesh, cell=mesh.phase.copy(), orientations=euler)
    text = io.save_mesh(mesh, tmp_path / "ab.inp", materials=small).read_text(encoding="utf-8")
    folder, build = re.search(r"cd (\S+) && (mfront [^\n]*)", text).groups()
    subprocess.run(build.split(), cwd=tmp_path / folder, check=True, capture_output=True)
    library = ctypes.CDLL(str(tmp_path / folder / "src" / "libABAQUSBEHAVIOUR.so"))
    axes = dict(re.findall(r"\*ORIENTATION, NAME=(\S+), SYSTEM=RECTANGULAR\n([^\n]*)", text))
    sections = re.findall(r"ELSET=CELL_(\d+), MATERIAL=([^,\s]+), ORIENTATION=(\S+)", text)
    assert len(sections) == 3
    for cell, name, orientation in sections:
        a, b = np.split(np.array(axes[orientation].split(","), dtype=float), 2)
        e1 = a / np.linalg.norm(a)
        e2 = b - (b @ e1) * e1
        e2 /= np.linalg.norm(e2)
        rotation = np.array([e1, e2, np.cross(e1, e2)])  # Abaqus's local axes (rows)
        block = text.split(f"*MATERIAL, NAME={name}\n")[1]
        depvar = re.match(r"\*DEPVAR\n(\d+),\n", block)
        native = simulate_point(small[int(cell)], SIX, orientation=euler[int(cell)])
        function = name.removeprefix("ABAQUSBEHAVIOUR_")
        step = _umat(library, function, [0.0], int(depvar.group(1)) if depvar else 0, ndi=3)
        stress = _umat_curve(step, native.strain, native.times, rotation, abaqus=True)
        assert np.allclose(stress, native.stress, rtol=0,
                           atol=1e-12 * np.abs(native.stress).max())  # fmt: skip
    # finite strains: the laws of the logarithmic and of the Green-Lagrange strain
    finite = {
        0: learned_material(_model("potential", data["finite"])),
        1: learned_material(_model("recurrent", data["finite"])),
        2: learned_material(_model("standard", data["finite"], "green-lagrange")),
        3: learned_material(_model("feedforward", data["finite"], "green-lagrange")),
    }

    def native(material, e):
        return simulate_point(material, SIX_FINITE, kinematics="finite", orientation=e)

    phase = rng.integers(0, 4, size=(4, 4, 4)).astype(np.uint8)
    io.write_amitex(tmp_path / "finite", VoxelImage(phase), finite, grains=grains,
                    orientations=orientations, paths={"t": SIX_FINITE}, name="lf",
                    kinematics="finite")  # fmt: skip
    _amitex_finite(tmp_path / "finite", finite, orientations, native)
    mesh = box_mesh((1, 1, 1), phase=lambda c: (c[:, 0] > 0.5).astype(int) + 2 * (c[:, 1] > 0.5))
    mesh = dataclasses.replace(mesh, cell=mesh.phase.copy(), orientations=euler)
    path = io.save_mesh(mesh, tmp_path / "abf.inp", materials=finite, kinematics="finite")
    _abaqus_finite(path, finite, euler, native)
