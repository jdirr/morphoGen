"""MFront sources of morphoGen's behaviours (morphogen.io.mfront): the sources (the brick
of von Mises plasticity; the scheme of crystal plasticity with morphoGen's slip systems
in TFEL's components), their state variables and errors; compiled by MFront, the
behaviours integrated by MGIS against morphoGen's kernels at a material point (von Mises:
viscous and rate-independent, small and finite strains; porous plasticity, GTN:
rate-independent and viscous, coalescence, nucleation, along paths that grow the porosity;
the other porous plasticities, morphoGen's kernel in their sources: Rousselier's
criterion, the elliptic one, the Gaussian and exponential nucleations from thresholds, a
table of f*, the coalescence of q3 > q1², kinematic hardening;
crystals: Franciosi's interactions, two families, isotropic elasticity, turned), and
through the umat interface
called as AMITEX calls it (the coefficients and internal variables of an export, the axes
of its zones, AMITEX's components). The compiled behaviours need MGIS and MFront
(conda-forge); without them those tests are skipped."""

import ctypes
import dataclasses
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from morphogen import cli, io
from morphogen.homogenization import CrystalPlasticity, Material, Plasticity, simulate_point
from morphogen.homogenization import finite as fin
from morphogen.homogenization import mfront as mf
from morphogen.homogenization.materials import isotropic_matrix
from morphogen.homogenization.mfront import MFrontBehaviour
from morphogen.homogenization.porous import PorousPlasticity
from morphogen.image import VoxelImage
from morphogen.io.mfront import (
    ABAQUS_AXES,
    mfront_source,
    mfront_state_size,
    mfront_strain_measure,
    mfront_symmetry,
    orthotropic_constants,
    umat_function,
    write_mfront,
)
from morphogen.orientations import random_orientations

TWINS = Path(__file__).parent / "mfront"
needs_mfront = pytest.mark.skipif(
    not (mf.available() and shutil.which("mfront")), reason="MGIS and MFront (conda-forge)"
)
VISCOUS = Plasticity(yield_stress=100, hardening=500, isotropic=((50, 20), (30, 200)),
                     kinematic=((20000, 200), (5000, 0)), viscosity=50, exponent=5)  # fmt: skip
RATE_INDEPENDENT = Plasticity(yield_stress=150, hardening=1000, isotropic=((80, 15),),
                              kinematic=((30000, 300),))  # fmt: skip
#: Hill's criterion of distinct coefficients, with the hardenings and the viscosity of VISCOUS
HILL = dataclasses.replace(VISCOUS, hill=(1.3, 0.8, 1.1, 0.7, 1.25, 0.9))
CUBIC = Material.cubic(168e3, 121e3, 75e3)
FRANCIOSI = CrystalPlasticity.from_config(
    {"families": ["octahedral"], "tau0": 40, "Q": 15, "b": 8, "C": 15000, "D": 250, "K": 20,
     "n": 12, "interaction": [1, 1, 0.6, 12.3, 1.6, 1.3]}
)  # fmt: skip
TWO_FAMILIES = CrystalPlasticity.from_config(
    {"families": {"octahedral": {"tau0": 80}, "cube": {"tau0": 120, "Q": 40, "C": 0}},
     "Q": 20, "b": 10, "C": 20000, "D": 300, "K": 25, "n": 10, "interaction": [1.0, 1.4]}
)  # fmt: skip
BCC = CrystalPlasticity.from_config(
    {"families": ["bcc110", "bcc112"], "tau0": 60, "Q": 10, "b": 5, "C": 5000, "D": 100,
     "K": 30, "n": 8}
)  # fmt: skip
CYCLIC = {"times": [0, 10, 20, 30, 40], "increments": [20, 10, 20, 10],
          "strain": {"11": [0, 0.01, 0.01, -0.01, -0.01],
                     "12": [0, 0.004, 0.004, 0, 0]}}  # fmt: skip
FINITE = {"times": [0, 10, 20], "increments": [30, 20],
          "strain": {"11": [0, 0.3, 0.3], "12": [0, 0.0, 0.2]}}  # fmt: skip
CRYSTAL_PATH = {"times": [0, 10, 20], "increments": [20, 20],
                "strain": {"11": [0, 0.01, -0.005], "23": [0, 0.004, 0.004]}}  # fmt: skip
#: all six components of the gradient (upper triangle), in finite strains
FULL_PATH_FINITE = {"times": [0, 10, 20], "increments": [20, 20],
                    "strain": {"11": [0, 0.08, 0.08], "22": [0, -0.02, -0.02],
                               "33": [0, -0.03, -0.01], "23": [0, 0.01, 0.03],
                               "31": [0, 0.0, 0.02], "12": [0, 0.02, -0.01]}}  # fmt: skip
#: all six components of the strain, for the umat interface
FULL_PATH = {"times": [0, 10, 20], "increments": [20, 20],
             "strain": {"11": [0, 0.01, -0.005], "22": [0, -0.003, 0.0],
                        "33": [0, -0.004, 0.002], "23": [0, 0.004, 0.004],
                        "31": [0, 0.002, 0.0], "12": [0, -0.001, 0.003]}}  # fmt: skip
#: triaxial paths of the six components that grow the porosity of the porous materials, in
#: small and finite strains
POROUS_PATH = {"times": [0, 10, 20], "increments": [20, 20],
               "strain": {"11": [0, 0.012, 0.02], "22": [0, 0.01, 0.016],
                          "33": [0, 0.008, 0.014], "23": [0, 0.0, 0.002],
                          "31": [0, 0.001, 0.0], "12": [0, 0.003, -0.002]}}  # fmt: skip
POROUS_FINITE = {"times": [0, 10, 20], "increments": [20, 20],
                 "strain": {"11": [0, 0.04, 0.06], "22": [0, 0.03, 0.04],
                            "33": [0, 0.02, 0.035], "23": [0, 0.01, 0.02],
                            "31": [0, 0.0, 0.01], "12": [0, 0.01, -0.01]}}  # fmt: skip


def _gtn(**kw):
    """A porous material of Gurson-Tvergaard-Needleman: its matrix with Voce terms, the
    porosity f0 = 0.01 (``kw``: the other parameters of the porous plasticity)."""
    spec = {"criterion": "gurson", "yield_stress": 200.0, "porosity": 0.01,
            "hardening": 500.0, "isotropic": ((80.0, 15.0), (30.0, 200.0)), "q1": 1.5,
            "q2": 1.0, **kw}  # fmt: skip
    return Material(young=200e3, poisson=0.3, porous_plasticity=PorousPlasticity(**spec))


#: Chu & Needleman's Gaussian nucleation and the exponential one, from thresholds of p
GAUSSIAN = {"type": "gaussian", "fN": 0.04, "eN": 0.02, "sN": 0.01, "yield": 0.005}
EXPONENTIAL = {"type": "exponential", "B": 1.0, "e0": 0.02, "yield": 0.003}
#: the coefficients of the elliptic criterion, tables of the porosity
ELLIPTIC = {"criterion": "elliptic", "C": ((0.0, 1.0), (0.2, 2.0)), "F": ((0.0, 0.0), (0.2, 0.4))}


def _porous(**kw):
    """A porous material outside MFront's brick (morphoGen's kernel in its source): its
    matrix with a Voce term (``kw``: the other parameters of the porous plasticity)."""
    spec = {"yield_stress": 200.0, "hardening": 500.0, "isotropic": ((80.0, 15.0),), **kw}
    return Material(young=200e3, poisson=0.3, porous_plasticity=PorousPlasticity(**spec))


def _compiled(directory, material, name, kinematics="small"):
    """The MFront law of the source of a material (compiled in ``directory``)."""
    source = write_mfront(material, directory / f"{name}.mfront", kinematics=kinematics)
    law = MFrontBehaviour(library=str(mf.compile_mfront(source, directory=directory / "c")),
                          behaviour=name)  # fmt: skip
    return dataclasses.replace(material, plasticity=None, crystal_plasticity=None,
                               porous_plasticity=None, mfront=law, expansion=None)  # fmt: skip


def _array(text, name):
    """The values of the constant array ``name`` of a generated source."""
    body = re.search(rf"constexpr double {name}\[[^=]*= \{{(.*?)\}};", text, re.S).group(1)
    return np.array([float(v) for v in re.findall(r"-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", body)])


def close(a, b, tol=1e-8):
    """The curves of two paths at their common times (one may divide a step)."""
    common, ia, ib = np.intersect1d(np.round(a.times, 12), np.round(b.times, 12),
                                    return_indices=True)  # fmt: skip
    assert common.size == min(a.n_steps, b.n_steps) + 1
    scale = np.abs(a.stress).max()
    assert np.allclose(b.stress[ib], a.stress[ia], rtol=0, atol=tol * scale)
    assert np.allclose(b.plastic[ib], a.plastic[ia], rtol=0,
                       atol=tol * max(np.abs(a.plastic).max(), 1e-12))  # fmt: skip


def test_mfront_sources(tmp_path):
    """The sources: the brick of von Mises plasticity and its terms, the logarithmic strain
    of finite strains; the slip systems of crystals in TFEL's components (11, 22, 33, then
    sqrt(2) times 12, 13, 23), their parameters and interaction matrix, the orthotropic
    constants in MFront's order; the state variables; the errors."""
    mises = Material(young=70e3, poisson=0.33, plasticity=VISCOUS)
    text = mfront_source(mises, "Mises")
    for line in ('@Behaviour Mises;',
                 'stress_potential : "Hooke" {young_modulus : 70000.0, poisson_ratio : 0.33}',
                 'inelastic_flow : "Norton" {',
                 'isotropic_hardening : "Linear" {R0 : 100.0, H : 500.0}',
                 'isotropic_hardening : "Voce" {R0 : 0, Rinf : 50.0, b : 20.0}',
                 'isotropic_hardening : "Voce" {R0 : 0, Rinf : 30.0, b : 200.0}',
                 'kinematic_hardening : "Armstrong-Frederick" {C : 20000.0, D : 200.0}',
                 'kinematic_hardening : "Prager" {C : 5000.0}', "K : 50.0", "n : 5.0"):  # fmt: skip
        assert line in text
    assert "@StrainMeasure" not in text
    assert "@StrainMeasure Hencky;" in mfront_source(mises, "MisesLog", kinematics="finite")
    plastic = mfront_source(Material(young=70e3, poisson=0.33, plasticity=RATE_INDEPENDENT), "P")
    assert 'inelastic_flow : "Plastic"' in plastic and "K :" not in plastic
    assert mfront_state_size(mises) == 7 + 6 * 2 and mfront_symmetry(mises) == "isotropic"
    crystal = dataclasses.replace(CUBIC, crystal_plasticity=FRANCIOSI)
    text = mfront_source(crystal, "Grain")
    m = _array(text, "m").reshape(12, 6)
    normals, directions = FRANCIOSI.systems()
    schmid = 0.5 * (np.einsum("si,sj->sij", directions, normals)
                    + np.einsum("si,sj->sij", normals, directions))  # fmt: skip
    assert np.allclose(m[:, :3], schmid[:, [0, 1, 2], [0, 1, 2]])
    assert np.allclose(m[:, 3:], np.sqrt(2) * schmid[:, [0, 0, 1], [1, 2, 2]])  # 12, 13, 23
    rng = np.random.default_rng(3)
    sigma = rng.normal(size=(3, 3))
    sigma = sigma + sigma.T
    tfel = np.r_[np.diag(sigma), np.sqrt(2) * sigma[[0, 0, 1], [1, 2, 2]]]
    assert np.allclose(m @ tfel, np.einsum("sij,ij->s", schmid, sigma))  # tau = sigma : m
    assert np.allclose(_array(text, "h").reshape(12, 12), FRANCIOSI.interaction_matrix())
    for k, key in enumerate(("tau0", "Q", "b", "C", "D", "K", "n")):
        assert np.allclose(_array(text, key), FRANCIOSI.params_array()[:, k])
    line = re.search(r"@ComputeStiffnessTensor<UnAltered>\{(.*)\};", text).group(1)
    assert np.allclose([float(v) for v in line.split(",")], orthotropic_constants(CUBIC.stiffness))
    assert "@OrthotropicBehaviour;" in text and "@IntegerConstant Nss = 12;" in text
    assert mfront_state_size(crystal) == 43 and mfront_symmetry(crystal) == "orthotropic"
    two = dataclasses.replace(CUBIC, crystal_plasticity=TWO_FAMILIES)
    assert mfront_state_size(two) == 7 + 3 * 18
    # crystals in finite strains: morphoGen's kernel of the host, its constants, the systems
    text = mfront_source(crystal, "CrystalLog", kinematics="finite")
    assert "@DSL DefaultFiniteStrain;" in text and "@TangentOperator<DPK1_DF>{" in text
    assert "inline int cp_finite(" in text and "__device__" not in text
    assert "__global__" not in text and "#define NS" not in text  # no macro reaches TFEL
    assert "constexpr int NS = 12;" in text and "@AuxiliaryStateVariable real mg[49];" in text
    sys = _array(text, "systems").reshape(12, 6)
    assert np.allclose(sys, np.concatenate([directions, normals], axis=1))  # l, n
    assert np.allclose(_array(text, "params").reshape(12, 7), FRANCIOSI.params_array())
    assert np.allclose(_array(text, "stiffness").reshape(6, 6), CUBIC.stiffness)
    assert mfront_state_size(crystal, "finite") == 49 + 1
    assert ABAQUS_AXES in text and mfront_strain_measure(crystal, "finite") is None
    assert umat_function("CrystalLog") == "umatcrystallog"  # the deformation gradient
    # elasticity in finite strains: Hencky's, isotropic or in the axes of the crystal
    elastic = Material(young=2e5, poisson=0.25)
    text = mfront_source(elastic, "Iso", kinematics="finite")
    assert "@StrainMeasure Hencky;" in text and "@OrthotropicBehaviour;" not in text
    lam, mu = elastic.lame
    assert f"lambda = {lam!r};" in text and f"mu = {mu!r};" in text
    assert mfront_state_size(elastic, "finite") == 0
    assert mfront_symmetry(elastic, "finite") == "isotropic"
    assert umat_function("Iso", mfront_strain_measure(elastic, "finite")) == "umatiso_malls"
    text = mfront_source(CUBIC, "Cubic", kinematics="finite")
    assert "@OrthotropicBehaviour;" in text and ABAQUS_AXES in text
    line = re.search(r"@ComputeStiffnessTensor<UnAltered>\{(.*)\};", text).group(1)
    assert np.allclose([float(v) for v in line.split(",")], orthotropic_constants(CUBIC.stiffness))
    assert mfront_symmetry(CUBIC, "finite") == "orthotropic"
    assert mfront_strain_measure(mises, "finite") == "Hencky"
    assert mfront_strain_measure(mises) is None and umat_function("Mises") == "umatmises"
    # the engineering constants in MFront's order, from three distinct shear moduli
    e1, e2, e3, n12, n23, n13, g12, g23, g13 = 2e5, 1.5e5, 1e5, 0.3, 0.25, 0.2, 1e4, 2e4, 3e4
    s = np.diag([1 / e1, 1 / e2, 1 / e3, 1 / g23, 1 / g13, 1 / g12])  # morphoGen: 23, 31, 12
    s[0, 1] = s[1, 0] = -n12 / e1
    s[0, 2] = s[2, 0] = -n13 / e1
    s[1, 2] = s[2, 1] = -n23 / e2
    expected = [e1, e2, e3, n12, n23, n13, g12, g23, g13]
    assert np.allclose(orthotropic_constants(np.linalg.inv(s)), expected)
    iso = orthotropic_constants(isotropic_matrix(*Material(young=2e5, poisson=0.25).lame))
    assert np.allclose(iso, [2e5] * 3 + [0.25] * 3 + [8e4] * 3)
    path = write_mfront(mises, tmp_path / "laws" / "Alloy.mfront")
    assert "@Behaviour Alloy;" in path.read_text(encoding="utf-8")
    # morphogen mfront: a source per nonlinear phase; in finite strains, per phase
    mats = {0: mises.to_config(), 1: Material(young=4e5, poisson=0.2).to_config()}
    (tmp_path / "alloy-2.json").write_text(json.dumps(mats), encoding="utf-8")
    assert cli.main(["mfront", "--materials", str(tmp_path / "alloy-2.json"), "-o",
                     str(tmp_path / "cli")]) == 0  # fmt: skip
    assert (tmp_path / "cli" / "alloy2Phase0.mfront").exists()
    assert not (tmp_path / "cli" / "alloy2Phase1.mfront").exists()
    assert cli.main(["mfront", "--materials", str(tmp_path / "alloy-2.json"), "-o",
                     str(tmp_path / "log"), "--kinematics", "finite"]) == 0  # fmt: skip
    text = (tmp_path / "log" / "alloy2Phase0.mfront").read_text(encoding="utf-8")
    assert "@Behaviour alloy2Phase0;" in text and "@StrainMeasure Hencky;" in text
    text = (tmp_path / "log" / "alloy2Phase1.mfront").read_text(encoding="utf-8")
    assert "@StrainMeasure Hencky;" in text and "constexpr auto mu =" in text
    assert cli.main(["mfront", "--material", "0=young:1e5,poisson:0.3"]) != 0  # nothing
    coupled = np.array(CUBIC.stiffness, dtype=float)
    coupled[0, 3] = coupled[3, 0] = 1e3
    hot = Material(young=1e5, poisson=0.3, plasticity=RATE_INDEPENDENT, expansion=1e-5)
    # thermal expansion in small strains: the Hooke potential of the brick, the principal
    # values of Hill's criterion and of a crystal
    assert "poisson_ratio : 0.3, thermal_expansion : 1e-05}" in mfront_source(hot, "Hot")
    hill_hot = Material(young=1e5, poisson=0.3, plasticity=HILL, expansion=(1e-5, 2e-5, 3e-5))
    assert ("thermal_expansion1 : 1e-05, thermal_expansion2 : 2e-05, thermal_expansion3 : 3e-05"
            in mfront_source(hill_hot, "HillHot"))  # fmt: skip
    warm = dataclasses.replace(CUBIC, crystal_plasticity=FRANCIOSI, expansion=(1e-5, 1.5e-5, 2e-5))
    assert "@ComputeThermalExpansion {1e-05, 1.5e-05, 2e-05};" in mfront_source(warm, "Warm")
    sheared = dataclasses.replace(warm, expansion=((1e-5, 1e-6, 0), (1e-6, 1e-5, 0), (0, 0, 1e-5)))
    # finite strains: MFront's mean coefficient whose stretch is morphoGen's exp(alpha dT)
    assert '"1e-05*(1+(1e-05*(T-293.15))*(0.5+' in mfront_source(hot, "HotLog", kinematics="finite")
    # porous plasticity (GTN): the porous criterion of the brick, the porosity in its
    # implicit system, f0 until the matrix flows; Tvergaard & Needleman's coalescence (f_r
    # of morphoGen's slope of f* when q3 is not q1²), the constant nucleation
    gtn = mfront_source(_gtn(), "Gtn")
    assert ('criterion : "GursonTvergaardNeedleman1982" {f_c : 1.0, f_r : 2.0, q_1 : 1.5, '
            "q_2 : 1.0, q_3 : 2.25}," in gtn)  # fmt: skip
    assert 'algorithm : "standard implicit scheme"' in gtn and '"Plastic" {' in gtn
    assert "@InitLocalVariables<AtBeginning>{\n  if (this->p <= 0) {\n    this->f = 0.01;" in gtn
    assert mfront_state_size(_gtn()) == 9 and mfront_symmetry(_gtn()) == "isotropic"
    assert mfront_strain_measure(_gtn(), "finite") == "Hencky"
    assert '"Norton" {' in mfront_source(_gtn(viscosity=100.0, exponent=4.0), "GtnNorton")
    assert "f_c : 0.02, f_r : 0.15," in mfront_source(_gtn(fc=0.02, fF=0.15), "Coalescence")
    slope = mfront_source(_gtn(fc=0.02, fF=0.15, q3=2.0), "Slope")
    fu = (1.5 - np.sqrt(1.5**2 - 2.0)) / 2.0  # MFront's ultimate porosity
    fr = float(re.search(r"f_r : ([^,]+),", slope).group(1))
    assert abs((fu - 0.02) / (fr - 0.02) - (1 / 1.5 - 0.02) / (0.15 - 0.02)) < 1e-14
    nucleated = _gtn(nucleation={"type": "constant", "A": 0.05})
    assert ('nucleation_model : "PowerLaw (strain)" {fn : 0.05, en : 1e-200, m : 0, fmax : 1}'
            in mfront_source(nucleated, "Nucleated"))  # fmt: skip
    assert mfront_state_size(nucleated) == 10
    porous_hot = dataclasses.replace(_gtn(), expansion=1.2e-5)
    assert "poisson_ratio : 0.3, thermal_expansion : 1.2e-05}" in mfront_source(porous_hot, "H")
    # the others: morphoGen's kernel in the source (porous_update), its row of parameters,
    # its internal variables then p and f (the elastic strain first, in small strains)
    for material in [_porous(criterion="rousselier", porosity=0.01, sigma1=300.0, D=2.0),
                     _porous(criterion="elliptic", porosity=0.3, C=2.0, F=0.3),
                     _gtn(nucleation=GAUSSIAN), _gtn(nucleation=EXPONENTIAL),
                     _gtn(nucleation={"type": "constant", "A": 0.05, "yield": 0.1}),
                     _gtn(effective=((0.0, 0.0), (0.05, 0.08))), _gtn(fc=0.02, fF=0.15, q3=3.0),
                     _gtn(kinematic=((3e4, 200.0),)),
                     _porous(**ELLIPTIC, kinematic=((3e4, 0),))]:  # fmt: skip
        nk = len(material.porous_plasticity.kinematic)
        row = material.porous_plasticity.params(*material.lame)
        for kinematics, n_state in (("small", 18 + 6 * nk), ("finite", 12 + 6 * nk)):
            text = mfront_source(material, "Own", kinematics=kinematics)
            assert "mk::porous_update(mk::params, e6, eth, dt, old, nw, s6, k6)" in text
            assert "@Brick" not in text and np.array_equal(_array(text, "params"), row)
            assert ("@StrainMeasure Hencky;" in text) == (kinematics == "finite")
            assert mfront_state_size(material, kinematics) == n_state
            assert mfront_symmetry(material, kinematics) == "isotropic"
        assert mfront_strain_measure(material, "finite") == "Hencky"
    theta = "1.2e-05 * (T + dT - 293.15)"  # morphoGen's thermal strain
    own_hot = mfront_source(dataclasses.replace(_gtn(kinematic=((3e4, 200.0),)), expansion=1.2e-5),
                            "OwnHot")  # fmt: skip
    assert f"const double eth[6] = {{{theta}, {theta}, {theta}, 0, 0, 0}};" in own_hot
    for call, match in [
        (lambda: mfront_source(sheared, "S"), "shear components"),
        (lambda: mfront_source(dataclasses.replace(hot, expansion=(1e-5, 2e-5, 3e-5)), "A"),
         "anisotropic expansion of a von Mises"),
        (lambda: mfront_source(Material(young=1e5, poisson=0.3), "E"), "nonlinear"),
        (lambda: mfront_source(Material(void=True), "V", kinematics="finite"), "void"),
        (lambda: mfront_source(crystal, "C", kinematics="large"), "kinematics"),
        (lambda: umat_function("C", "Linearised"), "strain measure"),
        (lambda: mfront_source(Material(young=1e5, poisson=0.3, mfront="law.mfront"), "M"),
         "already"),
        (lambda: mfront_source(mises, "2bad"), "identifier"),
        (lambda: write_mfront(mises, tmp_path / "law.txt"), ".mfront"),
        (lambda: orthotropic_constants(coupled), "couples"),
        (lambda: mfront_source(dataclasses.replace(porous_hot, expansion=(1e-5, 2e-5, 3e-5)),
                               "P"), "anisotropic expansion of a porous"),
    ]:  # fmt: skip
        with pytest.raises(ValueError, match=match):
            call()


@needs_mfront
def test_exported_behaviours(tmp_path):
    """The behaviours written by morphoGen, compiled by MFront and integrated by MGIS,
    against morphoGen's kernels at a material point."""

    def compiled(material, name, kinematics="small"):
        return _compiled(tmp_path, material, name, kinematics)

    viscous = Material(young=70e3, poisson=0.33, plasticity=VISCOUS)
    close(simulate_point(viscous, CYCLIC), simulate_point(compiled(viscous, "Viscous"), CYCLIC))
    logarithmic = compiled(viscous, "ViscousLog", "finite")
    close(simulate_point(viscous, FINITE, kinematics="finite"),
          simulate_point(logarithmic, FINITE, kinematics="finite"))  # fmt: skip
    plastic = Material(young=70e3, poisson=0.33, plasticity=RATE_INDEPENDENT)
    close(simulate_point(plastic, CYCLIC), simulate_point(compiled(plastic, "Yield"), CYCLIC))
    # Hill's criterion, an orthotropic behaviour in the axes of the sample: the six
    # components of a path, in small and finite strains
    hill = Material(young=70e3, poisson=0.33, plasticity=HILL)
    close(simulate_point(hill, FULL_PATH), simulate_point(compiled(hill, "HillLaw"), FULL_PATH))
    close(simulate_point(hill, FULL_PATH_FINITE, kinematics="finite"),
          simulate_point(compiled(hill, "HillLawLog", "finite"), FULL_PATH_FINITE,
                         kinematics="finite"))  # fmt: skip
    for name, elastic, cp, euler in [
        ("Franciosi", CUBIC, FRANCIOSI, (30.0, 40.0, 50.0)),
        ("TwoFamilies", Material.cubic(250e3, 160e3, 120e3), TWO_FAMILIES, (112.0, 63.0, 201.0)),
        ("Bcc", Material(young=210e3, poisson=0.29), BCC, (30.0, 40.0, 50.0)),  # isotropic
    ]:  # fmt: skip
        native = dataclasses.replace(elastic, crystal_plasticity=cp)
        law = compiled(native, name)
        close(simulate_point(native, CRYSTAL_PATH, orientation=euler),
              simulate_point(law, CRYSTAL_PATH, orientation=euler))  # fmt: skip
    # thermal expansion along paths whose temperature changes: a von Mises cycle, Hill's
    # criterion with an orthotropic expansion in the axes of the sample, a turned crystal with
    # one in the axes of the crystal
    hot = Material(young=70e3, poisson=0.33, plasticity=VISCOUS, expansion=1.2e-5)
    cycle = {**CYCLIC, "temperature": [0, 40, 40, 80, 80]}
    close(simulate_point(hot, cycle), simulate_point(compiled(hot, "ViscousHot"), cycle))
    hill_hot = Material(young=70e3, poisson=0.33, plasticity=HILL, expansion=(1e-5, 2e-5, 3e-5))
    full = {**FULL_PATH, "temperature": [0, 60, 20]}
    close(simulate_point(hill_hot, full), simulate_point(compiled(hill_hot, "HillHot"), full))
    warm = dataclasses.replace(CUBIC, crystal_plasticity=FRANCIOSI, expansion=(1e-5, 1.5e-5, 2e-5))
    crystal_hot = {**CRYSTAL_PATH, "temperature": [0, 50, 100]}
    finite_hot = {"times": [0, 10, 20], "increments": [20, 20],
                  "strain": {"11": [0, 0.08, 0.08], "12": [0, 0.02, -0.01]},
                  "temperature": [0, 300, 600]}  # fmt: skip
    for material, name in ((hot, "ViscousHotLog"), (hill_hot, "HillHotLog")):
        law = compiled(material, name, "finite")
        close(simulate_point(material, finite_hot, kinematics="finite"),
              simulate_point(law, finite_hot, kinematics="finite"))  # fmt: skip
    law = compiled(warm, "FranciosiHot")
    for euler in ((30.0, 40.0, 50.0), (112.0, 63.0, 201.0)):
        close(simulate_point(warm, crystal_hot, orientation=euler),
              simulate_point(law, crystal_hot, orientation=euler))  # fmt: skip
    # the sources of finite strains that hold morphoGen's own kernels take its logarithmic
    # thermal strain alpha dT: a turned crystal, Hencky's elasticity isotropic and cubic
    hot_path = {**FULL_PATH_FINITE, "temperature": [0, 300, 600]}
    for name, native, euler in (
        ("FranciosiHotLog", warm, (30.0, 40.0, 50.0)),
        ("IsoHotLog", Material(young=200e3, poisson=0.3, expansion=1.2e-5), None),
        ("CubicHotLog", dataclasses.replace(CUBIC, expansion=(1e-5, 1.5e-5, 2e-5)),
         (30.0, 40.0, 50.0)),
    ):  # fmt: skip
        law = compiled(native, name, "finite")
        close(simulate_point(native, hot_path, kinematics="finite", orientation=euler),
              simulate_point(law, hot_path, kinematics="finite", orientation=euler))  # fmt: skip
    # crystals in finite strains (morphoGen's kernel in the source): a path of the six
    # components, and a tension whose free components need the consistent tangent
    tension = {"times": [0, 10], "increments": 20, "strain": {"11": [0, 0.08]}}
    for name, elastic, cp in [("FranciosiLog", CUBIC, FRANCIOSI),
                              ("TwoFamiliesLog", Material.cubic(250e3, 160e3, 120e3),
                               TWO_FAMILIES)]:  # fmt: skip
        native = dataclasses.replace(elastic, crystal_plasticity=cp)
        law = compiled(native, name, "finite")
        for path in (FULL_PATH_FINITE, tension):
            a = simulate_point(native, path, kinematics="finite", orientation=(30.0, 40.0, 50.0))
            b = simulate_point(law, path, kinematics="finite", orientation=(30.0, 40.0, 50.0))
            close(a, b)
            assert a.iterations.sum() == b.iterations.sum()  # the same tangent
    # Hencky's elasticity in finite strains: isotropic, and a turned cubic crystal
    for name, elastic, euler in [("IsoLog", Material(young=200e3, poisson=0.3), None),
                                 ("CubicLog", CUBIC, (30.0, 40.0, 50.0))]:  # fmt: skip
        law = compiled(elastic, name, "finite")
        close(
            simulate_point(elastic, FULL_PATH_FINITE, kinematics="finite", orientation=euler),
            simulate_point(law, FULL_PATH_FINITE, kinematics="finite", orientation=euler),
        )
    # porous plasticity (GTN), the porosity in the implicit system of the brick, along
    # triaxial paths that grow it: rate-independent with Voce terms, Norton's flow with the
    # coalescence of Tvergaard & Needleman, the constant nucleation with a coalescence of q3
    # other than q1² (f_r of morphoGen's slope), thermal expansion, in small strains and in
    # the space of the logarithmic strain
    viscous_gtn = _gtn(viscosity=100.0, exponent=4.0)
    nucleation = {"type": "constant", "A": 0.05}
    for name, material, path, kinematics in [
        ("Gtn", _gtn(), POROUS_PATH, "small"),
        ("GtnNorton", _gtn(viscosity=100.0, exponent=4.0, fc=0.02, fF=0.15), POROUS_PATH,
         "small"),
        ("GtnNucleation", _gtn(porosity=0.002, fc=0.02, fF=0.15, q3=2.0, nucleation=nucleation),
         POROUS_PATH, "small"),
        ("GtnHot", dataclasses.replace(viscous_gtn, expansion=1.2e-5),
         {**POROUS_PATH, "temperature": [0, 60, 20]}, "small"),
        ("GtnLog", _gtn(), POROUS_FINITE, "finite"),
        ("GtnHotLog", dataclasses.replace(viscous_gtn, expansion=1.2e-5),
         {**POROUS_FINITE, "temperature": [0, 300, 600]}, "finite"),
    ]:  # fmt: skip
        native = simulate_point(material, path, kinematics=kinematics)
        assert native.porosity[-1] > native.porosity[0] + 0.03  # the porosity grows
        close(native, simulate_point(compiled(material, name, kinematics), path,
                                     kinematics=kinematics))  # fmt: skip


#: the porous plasticities outside MFront's brick, morphoGen's kernel in their sources:
#: name, material, path and kinematics
OWN_POROUS = [
    ("Rousselier", _porous(criterion="rousselier", porosity=0.01, sigma1=300.0, D=2.0,
                           nucleation=GAUSSIAN), POROUS_PATH, "small"),
    ("RousselierLog", _porous(criterion="rousselier", porosity=0.01, sigma1=300.0, D=2.0,
                              viscosity=100.0, exponent=4.0, nucleation=EXPONENTIAL),
     POROUS_FINITE, "finite"),
    ("Elliptic", _porous(**ELLIPTIC, porosity=0.05, viscosity=100.0, exponent=4.0,
                         kinematic=((3e4, 200.0), (5e3, 0.0))), POROUS_PATH, "small"),
    ("EllipticHotLog", dataclasses.replace(_porous(**ELLIPTIC, porosity=0.05), expansion=1.2e-5),
     {**POROUS_FINITE, "temperature": [0, 300, 600]}, "finite"),
    ("GtnTableHot", dataclasses.replace(
        _gtn(effective=((0.0, 0.0), (0.02, 0.02), (0.05, 0.12), (0.2, 0.6)),
             nucleation={"type": "constant", "A": 0.05, "yield": 0.004},
             kinematic=((3e4, 200.0),)), expansion=1.2e-5),
     {**POROUS_PATH, "temperature": [0, 60, 20]}, "small"),
    ("GtnQ3Log", _gtn(fc=0.02, fF=0.15, q3=3.0, kinematic=((2e4, 100.0), (4e3, 0.0))),
     POROUS_FINITE, "finite"),
]  # fmt: skip


@needs_mfront
def test_exported_porous_kernels(tmp_path):
    """The porous plasticities outside MFront's brick, morphoGen's kernel in their sources
    (porous_update), compiled by MFront and integrated by MGIS against morphoGen's kernels
    along triaxial paths that grow the porosity (:data:`OWN_POROUS`): Rousselier's
    criterion with Chu & Needleman's Gaussian nucleation and, viscous, with the exponential
    one, both from thresholds; the elliptic criterion (tables of the porosity) with
    Armstrong-Frederick and Prager backstresses; GTN with a table of f*, a threshold of the
    constant nucleation and a backstress, and with the coalescence of q3 > q1²; thermal
    expansion; in small strains and in the space of the logarithmic strain. Then a tension
    whose free components need the consistent tangent of the kernel. The kernel is
    morphoGen's: the curves agree to the roundoff (1e-12 of the stress scale; measured, 9e-16
    in small strains, 6e-14 through MFront's logarithmic strain)."""
    tension = {"times": [0, 10], "increments": 20, "strain": {"11": [0, 0.06]}}
    for name, material, path, kinematics in OWN_POROUS:
        native = simulate_point(material, path, kinematics=kinematics)
        assert native.porosity[-1] > native.porosity[0] + 0.03  # the porosity grows
        law = _compiled(tmp_path, material, name, kinematics)
        close(native, simulate_point(law, path, kinematics=kinematics), tol=1e-12)
        a = simulate_point(material, tension, kinematics=kinematics)
        b = simulate_point(law, tension, kinematics=kinematics)
        close(a, b, tol=1e-12)
        assert a.iterations.sum() == b.iterations.sum()  # the same tangent


def _umat(library, law, props, nstatv, ndi=2, statev=None):
    """A step of a law of a UMAT interface: the strains and stresses in the order (11, 22,
    33, 12, 13, 23) of AMITEX and Abaqus, engineering shear strains; the coefficients of
    the material, DROT the identity, three dimensions (NDI: 2 for the umat interface of
    Cast3M that AMITEX calls, 3 for Abaqus), ``nstatv`` state variables (``statev``: their
    initial values). In finite strains, ``f0`` and ``f1``: the deformation gradients at the
    start and the end of the step (DFGRD0, DFGRD1, in Fortran's order; JSTEP(3) = 1, the
    NLGEOM of Abaqus), the stress Cauchy's; ``heat``: morphoGen's change of temperature
    at the start of the step and its change over it (TEMP, from 293.15 K, and DTEMP).
    The step returns KINC and PNEWDT."""
    d, i, p = ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double)
    function = getattr(library, law)
    statev = np.zeros(max(nstatv, 1)) if statev is None else np.array(statev, dtype=float)
    nprops = i(len(props))
    props = np.ascontiguousarray(props if len(props) else [0.0], dtype=float)

    def ptr(a):
        return a.ctypes.data_as(p)

    def step(stran, dstran, dt, stress, f0=None, f1=None, heat=(0.0, 0.0)):
        ddsdde = np.zeros(36)  # 0 on input: the integration only (umat)
        scalars = [d(0.0) for _ in range(7)]
        time, predef, dpred, coords = np.zeros(2), np.zeros(1), np.zeros(1), np.zeros(3)
        finite = f0 is not None
        g0, g1 = (f0, f1) if finite else (np.eye(3), np.eye(3))
        dfgrd0, dfgrd1 = (np.ascontiguousarray(g.T.ravel()) for g in (g0, g1))  # columns
        drot = np.eye(3).ravel()
        dtime, pnewdt, celent = d(dt), d(1.0), d(1.0)
        temp, dtemp = d(293.15 + heat[0]), d(heat[1])
        n_d, nshr, ntens, nst = i(ndi), i(3), i(6), i(nstatv)
        ints, kstep, kinc = [i(1) for _ in range(4)], (i * 4)(1, 1, int(finite), 0), i(1)
        function(ptr(stress), ptr(statev), ptr(ddsdde), *[ctypes.byref(x) for x in scalars],
                 ptr(stran), ptr(dstran), ptr(time), ctypes.byref(dtime), ctypes.byref(temp),
                 ctypes.byref(dtemp), ptr(predef), ptr(dpred), ctypes.c_char_p(b"MAT"),
                 ctypes.byref(n_d), ctypes.byref(nshr), ctypes.byref(ntens),
                 ctypes.byref(nst), ptr(props), ctypes.byref(nprops), ptr(coords), ptr(drot),
                 ctypes.byref(pnewdt), ctypes.byref(celent), ptr(dfgrd0), ptr(dfgrd1),
                 *[ctypes.byref(x) for x in ints], kstep, ctypes.byref(kinc),
                 ctypes.c_int(3))  # fmt: skip
        return kinc.value, pnewdt.value

    return step


def _finite_curve(step, native, abaqus=False, temperature=None):
    """The Cauchy stresses (morphoGen's Voigt order) of a UMAT law of finite strains along
    the displacement gradients of morphoGen's curve of a path: the deformation gradients
    in the axes of the sample, the stresses in the order (11, 22, 33, 12, 13, 23);
    ``temperature``: morphoGen's change of temperature at the times of the curve."""
    grads, times = fin.gradient_of(native.strain), native.times
    heat = np.zeros(len(times)) if temperature is None else np.asarray(temperature)
    stress, zero = np.zeros(6), np.zeros(6)
    out = [np.zeros(6)]
    for k in range(1, len(grads)):
        f0, f1 = (np.eye(3) + grads[j].reshape(3, 3) for j in (k - 1, k))
        kinc, pnewdt = step(zero, zero, times[k] - times[k - 1], stress, f0, f1,
                            heat=(heat[k - 1], heat[k] - heat[k - 1]))  # fmt: skip
        assert pnewdt >= 1 if abaqus else kinc >= 1
        out.append(stress[[0, 1, 2, 5, 4, 3]])
    return np.array(out)


def _same_cauchy(stress, native, tol=1e-8):
    assert np.allclose(stress, native.cauchy, rtol=0, atol=tol * np.abs(native.cauchy).max())


def _umat_curve(step, strains, times, rotation=None, abaqus=False, temperature=None):
    """The stresses (morphoGen's order) of a UMAT law along the strains (morphoGen's Voigt
    order, engineering shears) of a path, called with the components (11, 22, 33, 12, 13,
    23) in the axes ``rotation`` (rows: the local axes of an orientation, which Abaqus
    gives its laws) or of the sample; ``temperature``: morphoGen's change of temperature at
    the times."""
    g = np.eye(3) if rotation is None else rotation
    heat = np.zeros(len(times)) if temperature is None else np.asarray(temperature)

    def to_call(e):
        t = np.array([[e[0], e[5] / 2, e[4] / 2], [e[5] / 2, e[1], e[3] / 2],
                      [e[4] / 2, e[3] / 2, e[2]]])  # fmt: skip
        t = g @ t @ g.T
        return np.array([t[0, 0], t[1, 1], t[2, 2], 2 * t[0, 1], 2 * t[0, 2], 2 * t[1, 2]])

    def from_call(s):
        t = g.T @ np.array([[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]]) @ g
        return np.array([t[0, 0], t[1, 1], t[2, 2], t[1, 2], t[2, 0], t[0, 1]])

    stress = np.zeros(6)
    out = [np.zeros(6)]
    for k in range(1, len(strains)):
        stran = to_call(strains[k - 1])
        dstran = to_call(strains[k]) - stran
        kinc, pnewdt = step(stran, dstran, times[k] - times[k - 1], stress,
                            heat=(heat[k - 1], heat[k] - heat[k - 1]))  # fmt: skip
        assert pnewdt >= 1 if abaqus else kinc >= 1  # Abaqus: PNEWDT < 1 on failure
        out.append(from_call(stress))
    return np.array(out)


def _dgibi_sizes(path):
    """The numbers of material properties and of state variables of the umat interface
    in three dimensions (the example of Cast3M written by MFront)."""
    text = path.read_text(encoding="utf-8").split("** Tridimensional example")[1]
    sizes = []
    for key in ("coel", "statev"):
        found = re.search(rf"{key} = 'MOTS'(.*?);", text, re.S)
        sizes.append(len(re.findall(r"'[^']*'", found.group(1))) if found else 0)
    return sizes


@needs_mfront
def test_amitex_umat_laws(tmp_path):
    """An AMITEX export of nonlinear materials: its MFront laws compiled as its script
    does (umat interface), the coefficients and internal variables of its material file
    against those of the interface (MFront's example of Cast3M), and the laws called as
    AMITEX calls them, with the coefficients of the file and the axes of the zones,
    against morphoGen's kernels along a path of the six strain components (a triaxial one
    for the porous plasticities: GTN of the brick, Rousselier's criterion of morphoGen's
    kernel)."""
    rng = np.random.default_rng(4)
    phase = (rng.random((4, 4, 4)) < 0.5).astype(np.uint8) + 1
    grains = rng.integers(0, 3, size=phase.shape)
    orientations = random_orientations(rng, 3)
    phase[0] = 3  # Hill's criterion: orthotropic, its axes those of the sample
    phase[1] = 4  # porous plasticity (GTN), along a path that grows its porosity
    phase[2] = 5  # porous plasticity outside the brick (morphoGen's kernel in its source)
    mats = {1: Material(young=70e3, poisson=0.33, plasticity=VISCOUS),
            2: dataclasses.replace(CUBIC, crystal_plasticity=FRANCIOSI),
            3: Material(young=70e3, poisson=0.33, plasticity=HILL),
            4: _gtn(viscosity=100.0, exponent=4.0, fc=0.02, fF=0.15,
                    nucleation={"type": "constant", "A": 0.05}),
            5: OWN_POROUS[0][1]}  # fmt: skip
    export = io.write_amitex(tmp_path, VoxelImage(phase), mats, grains=grains,
                             orientations=orientations, paths={"t": FULL_PATH},
                             name="nl")  # fmt: skip
    script = (tmp_path / "nl_run.sh").read_text(encoding="utf-8")
    build = re.search(r"\(cd mfront && (mfront [^)]*)\)", script).group(1).split()
    subprocess.run(build, cwd=tmp_path / "mfront", check=True, capture_output=True)
    library = ctypes.CDLL(str(tmp_path / "mfront" / "src" / "libUmatBehaviour.so"))
    root = ET.parse(tmp_path / "nl_material.xml").getroot()
    manifest = json.loads((tmp_path / "nl_amitex.json").read_text(encoding="utf-8"))
    for el, entry in zip(root.findall("Material"), manifest["materials"], strict=True):
        behaviour = entry["mfront"].split("/")[1].removesuffix(".mfront")
        coeffs = {int(c.get("Index")): c for c in el.findall("Coeff")}
        n_props, n_state = _dgibi_sizes(tmp_path / "mfront" / "castem" / f"{behaviour}.dgibi")
        assert sorted(coeffs) == list(range(1, n_props + 1))
        assert len(el.findall("IntVar")) == n_state == entry["internal_variables"]
        native_material = mats[entry["phase"]]
        for zone in range(entry["zones"]):
            props = []
            for k in range(1, n_props + 1):
                c = coeffs[k]
                if c.get("Type") == "Constant":
                    props.append(float(c.get("Value")))
                else:  # the axes of the zone
                    data = (tmp_path / c.get("File")).read_bytes().split(b"\n", 2)[2]
                    props.append(np.frombuffer(data, dtype=">f8")[zone])
            euler = orientations[entry["grains"][zone]] if "grains" in entry else None
            path = POROUS_PATH if native_material.porous else FULL_PATH
            native = simulate_point(native_material, path, orientation=euler)
            step = _umat(library, el.get("Law"), props, n_state)
            stress = _umat_curve(step, native.strain, native.times)
            assert np.allclose(stress, native.stress, rtol=0,
                               atol=1e-8 * np.abs(native.stress).max())  # fmt: skip
    assert all(p.exists() for p in export.files)


@needs_mfront
@pytest.mark.parametrize("hot", [False, True])
def test_abaqus_umat_laws(tmp_path, hot):
    """An Abaqus input of nonlinear materials: its MFront laws compiled with the abaqus
    interface as its header says, its materials (ABAQUSBEHAVIOUR_<behaviour>_3D: the
    function <BEHAVIOUR>_3D of libABAQUSBEHAVIOUR.so, as the dispatcher of MFront reads
    the name) and their *DEPVAR, and the laws called as Abaqus calls them, in the axes of
    the *ORIENTATION of each grain, against morphoGen's kernels along a path of the six
    strain components (a triaxial one for the porous plasticities: GTN of the brick, and
    with a table of f* and a backstress, morphoGen's kernel); ``hot``: the materials
    expand (isotropic, in the axes of the crystal, of the sample) while the temperature
    changes, the nodes at 293.15 K first (*INITIAL CONDITIONS) and the steps at absolute
    temperatures."""
    from helpers import box_mesh

    def phase(c):  # Hill's criterion (in the axes of the sample) and the porous above
        top = np.where(c[:, 0] > 0.5, np.where(c[:, 1] > 0.5, 4, 3), 2)
        return np.where(c[:, 2] > 0.4, top, (c[:, 0] + c[:, 1] > 1.15).astype(int))

    mesh = box_mesh((2, 2, 2), phase=phase)
    euler = np.array([[0.0, 0.0, 0.0], [30.0, 40.0, 50.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])  # fmt: skip
    mesh = dataclasses.replace(mesh, cell=mesh.phase.copy(), orientations=euler)
    mats = {0: Material(young=70e3, poisson=0.33, plasticity=VISCOUS),
            1: dataclasses.replace(CUBIC, crystal_plasticity=FRANCIOSI),
            2: Material(young=70e3, poisson=0.33, plasticity=HILL),
            3: _gtn(viscosity=100.0, exponent=4.0, fc=0.02, fF=0.15,
                    nucleation={"type": "constant", "A": 0.05}),
            4: dataclasses.replace(OWN_POROUS[4][1], expansion=None)}  # fmt: skip
    if hot:
        expansions = {0: 1.2e-5, 1: (1e-5, 1.5e-5, 2e-5), 2: (1e-5, 2e-5, 3e-5), 3: 1.2e-5,
                      4: 1.2e-5}  # fmt: skip
        mats = {p: dataclasses.replace(m, expansion=expansions[p]) for p, m in mats.items()}
    text = io.save_mesh(mesh, tmp_path / "ab.inp", materials=mats).read_text(encoding="utf-8")
    assert text.endswith("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nALL_NODES, 293.15\n") == hot
    folder, build = re.search(r"cd (\S+) && (mfront [^\n]*)", text).groups()
    subprocess.run(build.split(), cwd=tmp_path / folder, check=True, capture_output=True)
    library = ctypes.CDLL(str(tmp_path / folder / "src" / "libABAQUSBEHAVIOUR.so"))
    found = re.findall(r"\*MATERIAL, NAME=(\S+)\n\*DEPVAR\n(\d+),\n\*USER MATERIAL, "
                       r"CONSTANTS=(\d+), UNSYMM\n([^\n]*)\n", text)  # fmt: skip
    depvar = {name: int(n) for name, n, _, _ in found}
    constants = {name: [float(v) for v in row.split(",")] for name, _, n, row in found}
    assert all(len(constants[name]) == int(n) for name, _, n, _ in found)
    axes = dict(re.findall(r"\*ORIENTATION, NAME=(\S+), SYSTEM=RECTANGULAR\n([^\n]*)", text))
    sections = re.findall(r"ELSET=CELL_(\d+), MATERIAL=([^,\s]+), ORIENTATION=(\S+)", text)
    assert len(sections) == 5 and len(depvar) == 5
    for cell, name, orientation in sections:
        a, b = np.split(np.array(axes[orientation].split(","), dtype=float), 2)
        e1 = a / np.linalg.norm(a)
        e2 = b - (b @ e1) * e1
        e2 /= np.linalg.norm(e2)
        rotation = np.array([e1, e2, np.cross(e1, e2)])  # Abaqus's local axes (rows)
        phase = int(cell)  # one grain of each phase
        path = POROUS_PATH if mats[phase].porous else FULL_PATH  # one that grows the porosity
        path = {**path, "temperature": [0, 60, 20]} if hot else path
        native = simulate_point(mats[phase], path,
                                orientation=euler[phase] if phase == 1 else None)  # fmt: skip
        function = name.removeprefix("ABAQUSBEHAVIOUR_")  # as the dispatcher reads it
        step = _umat(library, function, constants[name], depvar[name], ndi=3)
        heat = np.interp(native.times, path["times"], path.get("temperature", [0, 0, 0]))
        stress = _umat_curve(step, native.strain, native.times, rotation, abaqus=True,
                             temperature=heat)  # fmt: skip
        assert np.allclose(stress, native.stress, rtol=0,
                           atol=1e-8 * np.abs(native.stress).max())  # fmt: skip


def _abaqus_example(path):
    """The constants and internal variables of a law in three dimensions, in MFront's
    example of Abaqus (``abaqus/<behaviour>.inp``)."""
    text = path.read_text(encoding="utf-8").split("'Tridimensional'")[1]
    constants = int(re.search(r"\*User Material, constants=(\d+)", text).group(1))
    depvar = re.search(r"\*Depvar\n(\d+),", text)
    return constants, int(depvar.group(1)) if depvar else 0


@needs_mfront
def test_user_laws_in_amitex_and_abaqus(tmp_path, monkeypatch):
    """The user's MFront laws in AMITEX and Abaqus inputs: their sources compiled for the
    umat and abaqus interfaces, their coefficients (those that umat imposes, merged with
    the material properties of their glossary names, then the others; for Abaqus, their
    material properties) and internal variables against MFront's examples, their
    parameters read from the directory of the runs, and the laws called as AMITEX and
    Abaqus call them against morphoGen's integration of the same laws through MGIS."""
    from helpers import box_mesh

    laws = {
        "VonMisesTwin": Material(young=70e3, poisson=0.33,
                                 mfront={"source": str(TWINS / "VonMisesTwin.mfront")}),
        "MericCailletaudTwin": dataclasses.replace(CUBIC, mfront=MFrontBehaviour(
            source=str(TWINS / "MericCailletaudTwin.mfront"), parameters={"tau0": 70.0})),
        "GlossaryElasticity": Material(young=1e5, poisson=0.2, mfront=MFrontBehaviour(
            source=str(TWINS / "GlossaryElasticity.mfront"),
            material_properties={"YoungModulus": 2e5, "PoissonRatio": 0.25, "Rate": 0.5})),
    }  # fmt: skip
    # AMITEX: three phases, the crystal in three grains
    rng = np.random.default_rng(5)
    phase = rng.integers(1, 4, size=(4, 4, 4)).astype(np.uint8)
    grains = rng.integers(0, 3, size=phase.shape)
    orientations = random_orientations(rng, 3)
    mats = dict(zip((1, 2, 3), laws.values(), strict=True))
    amitex = tmp_path / "amitex"
    io.write_amitex(amitex, VoxelImage(phase), mats, grains=grains, orientations=orientations,
                    paths={"t": FULL_PATH}, name="user")  # fmt: skip
    assert (amitex / "MericCailletaudTwin-parameters.txt").read_text() == "tau0 70.0\n"
    build = re.search(r"\(cd mfront && (mfront [^)]*)\)", (amitex / "user_run.sh").read_text())
    subprocess.run(build.group(1).split(), cwd=amitex / "mfront", check=True, capture_output=True)
    monkeypatch.chdir(amitex)  # where the laws read their parameters
    library = ctypes.CDLL(str(amitex / "mfront" / "src" / "libUmatBehaviour.so"))
    root = ET.parse(amitex / "user_material.xml").getroot()
    manifest = json.loads((amitex / "user_amitex.json").read_text(encoding="utf-8"))
    for el, entry in zip(root.findall("Material"), manifest["materials"], strict=True):
        behaviour = entry["mfront"].split("/")[1].removesuffix(".mfront")
        coeffs = {int(c.get("Index")): c for c in el.findall("Coeff")}
        n_props, n_state = _dgibi_sizes(amitex / "mfront" / "castem" / f"{behaviour}.dgibi")
        assert sorted(coeffs) == list(range(1, n_props + 1))
        assert len(el.findall("IntVar")) == max(n_state, 1)
        for zone in range(entry["zones"]):
            props = []
            for k in range(1, n_props + 1):
                c = coeffs[k]
                if c.get("Type") == "Constant":
                    props.append(float(c.get("Value")))
                else:
                    data = (amitex / c.get("File")).read_bytes().split(b"\n", 2)[2]
                    props.append(np.frombuffer(data, dtype=">f8")[zone])
            euler = orientations[entry["grains"][zone]] if "grains" in entry else None
            native = simulate_point(mats[entry["phase"]], FULL_PATH, orientation=euler)
            step = _umat(library, el.get("Law"), props, max(n_state, 1))
            stress = _umat_curve(step, native.strain, native.times)
            assert np.allclose(stress, native.stress, rtol=0,
                               atol=1e-8 * np.abs(native.stress).max())  # fmt: skip
    # Abaqus: two grains, the crystal turned
    mesh = box_mesh((1, 1, 1), phase=lambda c: (c[:, 0] + c[:, 1] > 1.15).astype(int))
    euler = np.array([[0.0, 0.0, 0.0], [30.0, 40.0, 50.0]])
    mesh = dataclasses.replace(mesh, cell=mesh.phase.copy(), orientations=euler)
    mats = {0: laws["GlossaryElasticity"], 1: laws["MericCailletaudTwin"]}
    abaqus = tmp_path / "abaqus"
    abaqus.mkdir()
    text = io.save_mesh(mesh, abaqus / "user.inp", materials=mats).read_text(encoding="utf-8")
    assert (abaqus / "MericCailletaudTwin-parameters.txt").exists()
    folder, build = re.search(r"cd (\S+) && (mfront [^\n]*)", text).groups()
    subprocess.run(build.split(), cwd=abaqus / folder, check=True, capture_output=True)
    monkeypatch.chdir(abaqus)
    library = ctypes.CDLL(str(abaqus / folder / "src" / "libABAQUSBEHAVIOUR.so"))
    axes = dict(re.findall(r"\*ORIENTATION, NAME=(\S+), SYSTEM=RECTANGULAR\n([^\n]*)", text))
    for phase in (0, 1):
        behaviour = mats[phase].mfront.behaviour
        name = f"ABAQUSBEHAVIOUR_{behaviour.upper()}_3D_PHASE{phase}"
        block = text.split(f"*MATERIAL, NAME={name}\n")[1].split("*SOLID SECTION")[0]
        depvar = re.match(r"\*DEPVAR\n(\d+),\n", block)
        n_state = int(depvar.group(1)) if depvar else 0
        found = re.search(r"\*USER MATERIAL, CONSTANTS=(\d+), UNSYMM\n([^*]*)", block)
        constants = [float(v) for v in found.group(2).replace("\n", ",").split(",") if v.strip()]
        n_props, n_example = _abaqus_example(abaqus / folder / "abaqus" / f"{behaviour}.inp")
        assert len(constants) == int(found.group(1)) == max(n_props, 1) and n_state == n_example
        a, b = np.split(np.array(axes[f"ORI_{phase}"].split(","), dtype=float), 2)
        e1 = a / np.linalg.norm(a)
        e2 = b - (b @ e1) * e1
        e2 /= np.linalg.norm(e2)
        rotation = np.array([e1, e2, np.cross(e1, e2)])
        native = simulate_point(mats[phase], FULL_PATH, orientation=euler[phase])
        step = _umat(library, name.removeprefix("ABAQUSBEHAVIOUR_").removesuffix(f"_PHASE{phase}"),
                     constants, n_state, ndi=3)  # fmt: skip
        stress = _umat_curve(step, native.strain, native.times, rotation, abaqus=True)
        assert np.allclose(stress, native.stress, rtol=0,
                           atol=1e-8 * np.abs(native.stress).max())  # fmt: skip


def _zone_coefficients(directory, coeffs, n_props, zone):
    """The coefficients of a zone of an AMITEX material: constants, and the values of the
    zone in the binary files of the axes."""
    props = []
    for k in range(1, n_props + 1):
        c = coeffs[k]
        if c.get("Type") == "Constant":
            props.append(float(c.get("Value")))
        else:
            data = (directory / c.get("File")).read_bytes().split(b"\n", 2)[2]
            props.append(np.frombuffer(data, dtype=">f8")[zone])
    return props


def _amitex_finite(directory, mats, orientations, reference):
    """The MFront laws of an AMITEX export in finite strains, compiled as its script does,
    their coefficients and internal variables against MFront's examples of Cast3M, and
    each zone called as AMITEX calls it in finite strains against ``reference(material,
    euler)`` (morphoGen's curve of the path)."""
    script = next(directory.glob("*_run.sh")).read_text(encoding="utf-8")
    build = re.search(r"\(cd mfront && (mfront [^)]*)\)", script).group(1).split()
    subprocess.run(build, cwd=directory / "mfront", check=True, capture_output=True)
    library = ctypes.CDLL(str(directory / "mfront" / "src" / "libUmatBehaviour.so"))
    root = ET.parse(next(directory.glob("*_material.xml"))).getroot()
    manifest = json.loads(next(directory.glob("*_amitex.json")).read_text(encoding="utf-8"))
    assert manifest["kinematics"] == "finite"
    for el, entry in zip(root.findall("Material"), manifest["materials"], strict=True):
        if el.get("Law") == "elasiso":
            continue
        behaviour = entry["mfront"].split("/")[1].removesuffix(".mfront")
        coeffs = {int(c.get("Index")): c for c in el.findall("Coeff")}
        n_props, n_state = _dgibi_sizes(directory / "mfront" / "castem" / f"{behaviour}.dgibi")
        assert sorted(coeffs) == list(range(1, n_props + 1))
        assert len(el.findall("IntVar")) == max(n_state, 1) == entry["internal_variables"]
        for zone in range(entry["zones"]):
            props = _zone_coefficients(directory, coeffs, n_props, zone)
            euler = orientations[entry["grains"][zone]] if "grains" in entry else None
            native = reference(mats[entry["phase"]], euler)
            step = _umat(library, el.get("Law"), props, max(n_state, 1))
            _same_cauchy(_finite_curve(step, native), native)


def _abaqus_finite(path, mats, euler, reference, constants=None, temperature=None):
    """The MFront laws of an Abaqus input in finite strains, compiled as its header says,
    their *DEPVAR against MFront's examples, and each grain (a phase) called as Abaqus
    calls it with NLGEOM (the axes of the orthotropic laws in their first six state
    variables, from *INITIAL CONDITIONS) against ``reference(material, euler)``;
    ``temperature``: the times and changes of temperature of its path."""
    from helpers import abaqus_initial_conditions

    text = path.read_text(encoding="utf-8")
    folder, build = re.search(r"cd (\S+) && (mfront [^\n]*)", text).groups()
    subprocess.run(build.split(), cwd=path.parent / folder, check=True, capture_output=True)
    library = ctypes.CDLL(str(path.parent / folder / "src" / "libABAQUSBEHAVIOUR.so"))
    initial = abaqus_initial_conditions(text) if "*INITIAL CONDITIONS" in text else {}
    sections = re.findall(r"\*SOLID SECTION, ELSET=CELL_(\d+), MATERIAL=(\S+)\n", text)
    assert len(sections) == len(mats)
    for cell, name in sections:
        phase = int(cell)  # one grain of each phase
        block = text.split(f"*MATERIAL, NAME={name}\n")[1]
        depvar = re.match(r"\*DEPVAR\n(\d+),\n", block)
        n_state = int(depvar.group(1)) if depvar else 0
        function = name.removeprefix("ABAQUSBEHAVIOUR_").split("_3D")[0] + "_3D"
        behaviour = next(p.stem for p in (path.parent / folder).glob("*.mfront")
                         if p.stem.upper() == function.removesuffix("_3D"))  # fmt: skip
        _, n_example = _abaqus_example(path.parent / folder / "abaqus" / f"{behaviour}.inp")
        assert n_state == n_example
        statev = initial[f"CELL_{cell}"][0][:n_state] if f"CELL_{cell}" in initial else None
        values = [0.0] if constants is None else constants[phase]
        native = reference(mats[phase], euler[phase] if mats[phase].oriented("elastic") else None)
        step = _umat(library, function, values, n_state, ndi=3, statev=statev)
        heat = None if temperature is None else np.interp(native.times, *temperature)
        _same_cauchy(_finite_curve(step, native, abaqus=True, temperature=heat), native)


def _native_finite(material, euler):
    return simulate_point(material, FULL_PATH_FINITE, kinematics="finite", orientation=euler)


#: the materials of the exports in finite strains: Hencky's elasticity (isotropic, cubic),
#: von Mises in the logarithmic strain space, crystal plasticity (morphoGen's kernel)
FINITE_MATS = {0: Material(young=200e3, poisson=0.3), 1: CUBIC,
               2: Material(young=70e3, poisson=0.33, plasticity=VISCOUS),
               3: dataclasses.replace(CUBIC, crystal_plasticity=FRANCIOSI)}  # fmt: skip


@needs_mfront
def test_amitex_finite_laws(tmp_path):
    """An AMITEX export in finite strains: its MFront laws compiled as its script does
    (umat), their coefficients and internal variables against MFront's examples of
    Cast3M, and the laws of each zone called as AMITEX calls them in finite strains (the
    deformation gradients DFGRD0 and DFGRD1, the Cauchy stress; the axes of the zones)
    against morphoGen's finite strains, along a path of the six components (the principal
    axes turn)."""
    rng = np.random.default_rng(6)
    phase = rng.integers(0, 4, size=(4, 4, 4)).astype(np.uint8)
    grains = rng.integers(0, 3, size=phase.shape)
    orientations = random_orientations(rng, 3)
    io.write_amitex(tmp_path, VoxelImage(phase), FINITE_MATS, grains=grains,
                    orientations=orientations, paths={"t": FULL_PATH_FINITE}, name="gd",
                    kinematics="finite")  # fmt: skip
    _amitex_finite(tmp_path, FINITE_MATS, orientations, _native_finite)


@needs_mfront
@pytest.mark.parametrize("hot", [False, True])
def test_abaqus_finite_laws(tmp_path, hot):
    """An Abaqus input in finite strains: its MFront laws compiled with the abaqus
    interface as its header says, their *DEPVAR against MFront's examples, and the laws
    called as Abaqus calls them with NLGEOM (the deformation gradients, the Cauchy stress;
    the axes of the orthotropic laws in their first six state variables, from *INITIAL
    CONDITIONS) against morphoGen's finite strains, each grain turned; ``hot``: the
    materials expand while the temperature changes (from 293.15 K, the initial
    temperature of the nodes)."""
    from helpers import box_mesh

    mesh = box_mesh((1, 1, 1), phase=lambda c: (c[:, 0] > 0.5).astype(int) + 2 * (c[:, 1] > 0.5))
    euler = np.array([[0.0, 0.0, 0.0], [30.0, 40.0, 50.0], [112.0, 63.0, 201.0],
                      [10.0, 20.0, 30.0]])  # fmt: skip
    mesh = dataclasses.replace(mesh, cell=mesh.phase.copy(), orientations=euler)
    assert sorted(set(mesh.phase.tolist())) == [0, 1, 2, 3]
    if not hot:
        path = io.save_mesh(mesh, tmp_path / "gd.inp", materials=FINITE_MATS,
                            kinematics="finite")  # fmt: skip
        _abaqus_finite(path, FINITE_MATS, euler, _native_finite)
        return
    # Hencky's elasticity and the crystals, morphoGen's own sources (the brick's expanding
    # laws of finite strains are refused: the abaqus interface has no stress-free
    # expansion in finite strains); von Mises does not expand
    expansions = {0: 1.2e-5, 1: (1e-5, 1.5e-5, 2e-5), 3: (1e-5, 1.5e-5, 2e-5)}
    mats = {p: dataclasses.replace(m, expansion=expansions.get(p))
            for p, m in FINITE_MATS.items()}  # fmt: skip
    hot_path = {**FULL_PATH_FINITE, "temperature": [0, 300, 600]}
    path = io.save_mesh(mesh, tmp_path / "hot.inp", materials=mats, kinematics="finite")
    assert "*INITIAL CONDITIONS, TYPE=TEMPERATURE\nALL_NODES, 293.15\n" in path.read_text()
    _abaqus_finite(path, mats, euler,
                   lambda m, e: simulate_point(m, hot_path, kinematics="finite", orientation=e),
                   temperature=(hot_path["times"], hot_path["temperature"]))  # fmt: skip


@needs_mfront
def test_user_finite_laws(tmp_path):
    """The user's MFront laws of finite strains in AMITEX and Abaqus inputs: Hencky's strain
    measure (the function _malls of umat), the deformation gradient (Saint Venant-Kirchhoff,
    its moduli material properties without glossary names), an orthotropic Hencky law with
    three distinct shear moduli whose axes Abaqus gives through its state variables;
    called as the codes call them against morphoGen's integration of the same laws through
    MGIS; an orthotropic law of finite strains without that policy, refused for Abaqus."""
    from helpers import box_mesh
    from morphogen.io.mfront import user_behaviour

    laws = {
        0: Material(young=70e3, poisson=0.33,
                    mfront={"source": str(TWINS / "VonMisesHenckyTwin.mfront")}),
        1: Material(young=1e5, poisson=0.3, mfront=MFrontBehaviour(
            source=str(TWINS / "SaintVenantKirchhoff.mfront"),
            material_properties={"E": 2e5, "nu": 0.3})),
        2: dataclasses.replace(CUBIC, mfront=MFrontBehaviour(
            source=str(TWINS / "OrthotropicHencky.mfront"))),
    }  # fmt: skip
    measures = [user_behaviour(m, "finite").strain_measure for m in laws.values()]
    assert measures == ["Hencky", None, "Hencky"] and user_behaviour(laws[2], "finite").abaqus_axes
    rng = np.random.default_rng(7)
    phase = rng.integers(0, 3, size=(4, 4, 4)).astype(np.uint8)
    grains = rng.integers(0, 3, size=phase.shape)
    orientations = random_orientations(rng, 3)
    amitex = tmp_path / "amitex"
    io.write_amitex(amitex, VoxelImage(phase), laws, grains=grains, orientations=orientations,
                    paths={"t": FULL_PATH_FINITE}, name="user", kinematics="finite")  # fmt: skip
    root = ET.parse(amitex / "user_material.xml").getroot()
    assert [el.get("Law") for el in root.findall("Material")] == [
        "umatvonmiseshenckytwin_malls",
        "umatsaintvenantkirchhoff",
        "umatorthotropichencky_malls",
    ]
    _amitex_finite(amitex, laws, orientations, _native_finite)
    mesh = box_mesh((1, 1, 1), phase=lambda c: (c[:, 0] > 0.5).astype(int) + (c[:, 1] > 0.5))
    euler = np.array([[0.0, 0.0, 0.0], [30.0, 40.0, 50.0], [112.0, 63.0, 201.0]])
    mesh = dataclasses.replace(mesh, cell=mesh.phase.copy(), orientations=euler)
    assert sorted(set(mesh.phase.tolist())) == [0, 1, 2]
    abaqus = tmp_path / "abaqus"
    abaqus.mkdir()
    path = io.save_mesh(mesh, abaqus / "user.inp", materials=laws, kinematics="finite")
    text = path.read_text(encoding="utf-8")
    assert "*USER MATERIAL, CONSTANTS=2, UNSYMM\n200000, 0.3\n" in text  # E, nu
    _abaqus_finite(path, laws, euler, _native_finite, constants={0: [0.0], 1: [2e5, 0.3],
                                                                  2: [0.0]})  # fmt: skip
    source = (TWINS / "OrthotropicHencky.mfront").read_text(encoding="utf-8")
    bare = tmp_path / "Bare.mfront"
    bare.write_text(source.replace("@AbaqusOrthotropyManagementPolicy[abaqus] MFront;\n", "")
                    .replace("OrthotropicHencky", "Bare"), encoding="utf-8")  # fmt: skip
    plain = {2: dataclasses.replace(CUBIC, mfront=MFrontBehaviour(source=str(bare)))}
    with pytest.raises(ValueError, match="AbaqusOrthotropyManagementPolicy"):
        io.save_mesh(mesh, tmp_path / "bare.inp", materials=plain, kinematics="finite")
