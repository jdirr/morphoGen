"""Material files of Z-mat (Z-set): reading the bricks of gen_evp behaviours, their
translation to morphoGen's kernels or to MFront's brick StandardElastoViscoPlasticity
(Hill's criterion in TFEL's convention), configurations and the command line, the coverage
of Z-mat's library; with MGIS and MFront, the translations compiled, the native kernels
against MFront for the same file, and Hill's yield stresses in tension and in each shear."""

import json
import shutil

import numpy as np
import pytest
import yaml

from morphogen import cli
from morphogen.config import load_config
from morphogen.homogenization import mfront as mf
from morphogen.homogenization import zmat
from morphogen.homogenization.materials import Material

CHABOCHE = """
% the first example of gen_evp in the manual of Z-mat
***behavior gen_evp
 **elasticity isotropic
      young 200000.
      poisson 0.30
 **potential gen_evp ev
   *criterion mises
   *flow norton
           n 7.0
           K 400.
   *kinematic nonlinear
         C 15000.0
         D 300.0
   *kinematic nonlinear
         C 6000.0
         D 100.0
   *isotropic constant
           R0 130.0
***return
"""

PLASTIC = """
***material
   *integration theta_method_a 1.0 1.e-10 1500
***behavior gen_evp
**elasticity isotropic
   young 2.1e5
   poisson 0.3
 **thermal_strain isotropic
   alpha 1.2e-5
 **potential gen_evp ep
   *flow plasticity
   *criterion mises
   *isotropic nonlinear
      R0 200.
      Q 2000.0
      b 0.26
   *kinematic nonlinear X
      C 25500.0
      D 81.
***return
"""

HILL = """
***behavior gen_evp
 **elasticity isotropic
   young 70000. poisson 0.33
 **potential gen_evp ep
   *flow plasticity
   *criterion hill
    hill1 {h1} hill2 {h2} hill3 {h3} hill4 {h4} hill5 {h5} hill6 {h6}
   *isotropic linear
    R0 100. H 10.
***return
"""

TWO = """
***behavior gen_evp
 **elasticity isotropic
     young 200000.
     poisson 0.3
 **potential gen_evp ev
   *criterion mises
   *flow hyperbolic
     K 22.3
     m 1.44
     eps0 .202e-8
   *kinematic nonlinear X1
     C 2000.0
     D 50.0
   *isotropic constant
     R0 20.0
 **potential gen_evp ep
   *kinematic nonlinear_evrad X2
     C 25000.0
     D 10.0
     eta 0.4
   *isotropic nonlinear_sum
     R0 850.0
     Q1 30. b1 2000.
     Q2 -10. b2 20.
***return
"""

ANISO = {"h1": 1, "h2": 2, "h3": 3, "h4": 1.2, "h5": 1.7, "h6": 0.8}

GURSON = """
***material
   *integration theta_method_a 1.0 1.e-10 1500
   *initialize_variable
      f 0.01
***behavior porous_plastic lagrange_rotate_no_J
 **elasticity isotropic
      young 210000. poisson 0.3
 **porous_potential
    *porous_criterion gurson
           fs f
           0. 0.
           0.05 0.05
           0.6667 0.25
           q1 1.5
           q2 1.
    *shear_anisotropy mises
    *isotropic_hardening nonlinear
           R0 300.
           Q 200.
           b 10.
    *flow norton K 10. n 5.
    *strain_nucleation gaussian
           en 0.3 fn 0.04 sn 0.1
 **save_D
***return
"""

COMPACTION = """
***behavior porous_plastic
 **elasticity isotropic
    young 100000. poisson 0.3
 **porous_potential
   *porous_criterion elliptic
    F f
    0. 0.
    0.6 0.5
    C f
    1. 0.
    4. 0.5
   *flow plasticity
   *isotropic_hardening constant
    R0 100.
***return
"""


def test_reading_and_translating_behaviours(tmp_path):
    native = zmat.describe(CHABOCHE)
    assert native["translation"] == "native"
    plasticity = native["material"]["plasticity"]
    assert plasticity == {
        "yield": 130.0,
        "kinematic": [{"C": 15000.0, "D": 300.0}, {"C": 6000.0, "D": 100.0}],
        "norton": {"K": 400.0, "n": 7.0},
    }
    material = zmat.zmat_material(PLASTIC)
    assert material.young == 210000.0 and material.expansion == pytest.approx(1.2e-5)
    assert material.plasticity.isotropic == ((2000.0, 0.26),)
    assert material.plasticity.kinematic == ((25500.0, 81.0),) and not material.plasticity.viscous
    cubic = zmat.zmat_material(
        "***behavior linear_elastic\n**elasticity cubic\n y1111 162321.0\n y1122 78075.0\n"
        " y1212 110615.0\n***return\n"
    )
    np.testing.assert_allclose(
        np.asarray(cubic.stiffness)[[0, 0, 3], [0, 1, 3]], [162321.0, 78075.0, 110615.0]
    )
    # Hill's criterion: native, its coefficients by component (Z-set's order: 11, 22, 33,
    # 12, 23, 31); for MFront, an orthotropic behaviour with TFEL's coefficients
    hill = zmat.describe(HILL.format(**ANISO))
    assert hill["translation"] == "native" and hill["kind"] == "plastic"
    aniso = zmat.zmat_material(HILL.format(**ANISO)).plasticity
    assert aniso.hill == (1.0, 2.0, 3.0, 1.2, 1.7, 0.8)
    assert aniso.to_config()["hill"] == {"11": 1.0, "22": 2.0, "33": 3.0, "12": 1.2, "23": 1.7,
                                         "31": 0.8}  # fmt: skip
    source = zmat.zmat_mfront_source(zmat.read_zmat(HILL.format(**ANISO))[0])
    assert "@OrthotropicBehaviour;" in source and '"Hill" {F : ' in source
    # MFront: several potentials, user-defined flows
    two = zmat.describe(TWO)["source"]
    assert two.count("inelastic_flow") == 2 and '"UserDefinedViscoplasticity"' in two
    assert 'vp : "zA * sinh((f / zK) ** zN) ** zM"' in two and "zA : 2.02e-09" in two
    assert '"Burlet-Cailletaud" {C : 25000.0, D : 10.0, eta : 0.4}' in two
    assert "Rinf : -10.0" in two  # softening: MFront, not the native kernels
    forced = zmat.zmat_material(CHABOCHE, native=False, directory=tmp_path, name="cha-boche")
    assert forced.plasticity is None and forced.mfront.behaviour == "ZmatChaboche"
    assert forced.mfront.plastic_strain == "EquivalentViscoplasticStrain"
    assert forced.mfront.source.endswith("ZmatChaboche.mfront")
    several = zmat.zmat_material(TWO, directory=tmp_path)
    assert several.mfront.plastic_strain == "EquivalentViscoplasticStrain0"
    errors = [
        ("***behavior umat\n***return\n", "behaviour 'umat'"),
        (CHABOCHE.replace("***return", "**interaction ev::X ev::X 1.\n***return"), "interaction"),
        (CHABOCHE.replace("R0 130.0", "R0 temperature"), "not a number"),
        (
            CHABOCHE.replace("*criterion mises", "*criterion tresca").replace(
                "*flow norton", "*flow hyperbolic"
            ),
            "criterion 'tresca'",
        ),
        ("***behavior linear_elastic\n**elasticity orthotropic\n c44 1.\n***return\n", "y1111"),
        ("% nothing\n", "behavior in the file"),
    ]
    for text, match in errors:
        with pytest.raises(ValueError, match=match):
            zmat.zmat_material(text, directory=tmp_path)


def test_porous_plastic_files(tmp_path, capsys):
    """porous_plastic: GTN with a table of the effective porosity (Tvergaard and Needleman's
    coalescence), the coefficients of a flow on its line, Chu and Needleman's nucleation and
    the initial porosity of *initialize_variable; the elliptic criterion of compaction with
    C and F tables of the porosity; Rousselier; the kinematic terms of gen_evp (linear and
    nonlinear, named) of GTN; what is not translated, and the command line."""
    from morphogen.homogenization.nonlinear import simulate_point

    out = zmat.describe(GURSON)
    assert out["translation"] == "native" and out["kind"] == "porous"
    pp = out["material"]["porous_plasticity"]
    assert pp["criterion"] == "gurson" and pp["porosity"] == 0.01 and pp["q3"] == 2.25
    assert pp["effective"] == [[0.0, 0.0], [0.05, 0.05], [0.25, 0.6667]]
    assert pp["norton"] == {"K": 10.0, "n": 5.0} and pp["isotropic"] == [{"Q": 200.0, "b": 10.0}]
    assert pp["nucleation"] == {"type": "gaussian", "yield": 0.0, "fN": 0.04, "eN": 0.3,
                                "sN": 0.1}  # fmt: skip
    assert any("lagrange_rotate_no_j" in note for note in out["notes"])
    material = zmat.zmat_material(GURSON)
    assert material.porous and material.plasticity is None and material.young == 210000.0
    path = {"times": [0, 1], "increments": 10, "strain": {"11": 0.05, "22": 0.02, "33": 0.02}}
    res = simulate_point(material, path, kinematics="finite")
    assert res.porosity[0] == 0.01 and res.porosity[-1] > 0.012
    foam = zmat.zmat_material(COMPACTION, porosity=0.4).porous_plasticity
    assert foam.criterion == "elliptic" and foam.porosity == 0.4
    assert foam.C == ((0.0, 1.0), (0.5, 4.0)) and foam.F == ((0.0, 0.0), (0.5, 0.6))
    assert "initial porosity 0" in zmat.describe(COMPACTION)["notes"][0]
    table = "F f\n    0. 0.\n    0.6 0.5\n    C f\n    1. 0.\n    4. 0.5"
    assert table in COMPACTION
    rousselier = COMPACTION.replace("elliptic", "modified_rousselier").replace(
        table, "sigma1 200. D 2. q2 1.5"
    )
    r = zmat.zmat_material(rousselier).porous_plasticity
    assert (r.criterion, r.sigma1, r.D, r.q2) == ("rousselier", 200.0, 2.0, 1.5)
    # the kinematic hardening of the matrix: the bricks of gen_evp
    kinematic = "*flow norton K 10. n 5.\n    *kinematic nonlinear X1\n           C 30000.\n"
    kinematic += "           D 200.\n    *kinematic linear X2\n           C 5000."
    hardened = GURSON.replace("*flow norton K 10. n 5.", kinematic)
    assert kinematic in hardened
    kin = zmat.zmat_material(hardened).porous_plasticity
    assert kin.kinematic == ((30000.0, 200.0), (5000.0, 0.0)) and kin.criterion == "gurson"
    assert zmat.describe(hardened)["material"]["porous_plasticity"]["kinematic"] == [
        {"C": 30000.0, "D": 200.0}, {"C": 5000.0, "D": 0.0}]  # fmt: skip
    foam = COMPACTION.replace("*flow plasticity", "*flow plasticity\n *kinematic linear\n  C 1.")
    assert zmat.zmat_material(foam).porous_plasticity.kinematic == ((1.0, 0.0),)
    errors = [
        (GURSON.replace("R0 300.", "R0 T\n 300. 20.\n 250. 400."), "depending on variables"),
        (COMPACTION.replace("elliptic", "cam_clay"), "cam_clay"),
        (COMPACTION.replace("***return", "**adiabatic_heating .9\n***return"), "adiabatic"),
        (rousselier.replace("*flow plasticity", "*flow plasticity\n *kinematic linear\n  C 1."),
         "criteria gurson and elliptic"),
        (COMPACTION.replace("*flow plasticity", "*flow plasticity\n *kinematic ziegler\n  C 1."),
         "kinematic hardening 'ziegler'"),
        (COMPACTION.replace("*flow plasticity", "*flow plasticity\n *kinematic linear\n  D 1."),
         "unknown coefficients"),
        (GURSON.replace("*shear_anisotropy mises", "*shear_anisotropy hill"), "mises only"),
        (GURSON.replace("gaussian", "crack_like gaussian"), "crack_like"),
    ]  # fmt: skip
    for text, match in errors:
        with pytest.raises(ValueError, match=match):
            zmat.zmat_material(text)
    with pytest.raises(ValueError, match="kernels only"):
        zmat.zmat_material(COMPACTION, native=False)
    (tmp_path / "gtn.mat").write_text(GURSON, encoding="utf-8")
    assert cli.main(["zmat", str(tmp_path / "gtn.mat")]) == 0
    text = capsys.readouterr().out
    assert "(finite elements, FFT, GPU)" in text and "porous_plasticity" in text and "note:" in text


def test_hill_coefficients_in_tfel_convention():
    """The quadratic form of Z-mat's Hill criterion (3/2 s : M : s, components 11, 22, 33,
    12, 23, 31) equals TFEL's F (s11 - s22)^2 + G (s22 - s33)^2 + H (s33 - s11)^2 + 2 L
    s12^2 + 2 M s13^2 + 2 N s23^2 with the translated coefficients, on random stresses."""
    one = zmat._hill({f"hill{i}": 1.0 for i in range(1, 7)})
    assert one == pytest.approx({"F": 0.5, "G": 0.5, "H": 0.5, "L": 1.5, "M": 1.5, "N": 1.5})
    h = [ANISO[f"h{i}"] for i in range(1, 7)]
    c = zmat._hill({f"hill{i}": v for i, v in enumerate(h, start=1)})
    rng = np.random.default_rng(3)
    for _ in range(20):
        s11, s22, s33, s12, s13, s23 = rng.normal(size=6)
        mean = (s11 + s22 + s33) / 3
        d = np.array([s11, s22, s33]) - mean
        zset = 1.5 * (
            h[0] * d[0] ** 2
            + h[1] * d[1] ** 2
            + h[2] * d[2] ** 2
            + 2 * h[3] * s12**2
            + 2 * h[4] * s23**2
            + 2 * h[5] * s13**2
        )
        tfel = (
            c["F"] * (s11 - s22) ** 2
            + c["G"] * (s22 - s33) ** 2
            + c["H"] * (s33 - s11) ** 2
            + 2 * c["L"] * s12**2
            + 2 * c["M"] * s13**2
            + 2 * c["N"] * s23**2
        )
        assert tfel == pytest.approx(zset, rel=1e-12)


def test_configurations_and_command_line(tmp_path, capsys):
    (tmp_path / "alloy.mat").write_text(CHABOCHE, encoding="utf-8")
    (tmp_path / "hill.mat").write_text(HILL.format(**ANISO), encoding="utf-8")
    (tmp_path / "two.mat").write_text(TWO, encoding="utf-8")
    cfg = load_config(
        {
            "model": {"type": "boolean_spheres", "radius": 0.2, "volume_fraction": 0.2},
            "domain": {"size": 1.0},
            "voxels": {"shape": 8},
            "homogenization": {
                "materials": {
                    0: {"zmat": "alloy.mat", "density": 2.7},
                    1: {"young": 4e5, "poisson": 0.2},
                },
                "method": "fft",
                "properties": ["bulk_modulus"],
            },
        },
        base_dir=tmp_path,
    )
    alloy = cfg.homogenization.materials[0]
    assert alloy.plasticity.viscosity == 400.0 and alloy.density == 2.7
    with pytest.raises(Exception, match="zmat"):
        load_config(
            {
                "model": {"type": "boolean_spheres", "radius": 0.2, "volume_fraction": 0.2},
                "domain": {"size": 1.0},
                "voxels": {"shape": 8},
                "homogenization": {
                    "materials": {
                        0: {"zmat": "alloy.mat", "young": 1.0},
                        1: {"young": 4e5, "poisson": 0.2},
                    }
                },
            },
            base_dir=tmp_path,
        )
    assert cli.main(["zmat", str(tmp_path / "alloy.mat")]) == 0
    out = capsys.readouterr().out
    assert "morphoGen's kernels" in out and yaml.safe_load(out.split("\n", 1)[1])["young"] == 2e5
    assert (
        cli.main(["zmat", str(tmp_path / "two.mat"), "-o", str(tmp_path / "laws"), "--json"]) == 0
    )
    done = json.loads(capsys.readouterr().out)
    assert done["translation"] == "mfront" and (tmp_path / "laws" / "ZmatTwo.mfront").exists()
    assert cli.main(["zmat", str(tmp_path / "hill.mat"), "--json"]) == 0
    done = json.loads(capsys.readouterr().out)
    assert done["translation"] == "native" and done["material"]["plasticity"]["hill"]["33"] == 3
    assert cli.main(["zmat", "--coverage"]) == 0
    table = capsys.readouterr().out
    assert table.startswith("| category | Z-mat |") and "gen_evp" in table and "hill" in table


def test_materials_written(tmp_path):
    """The materials of morphoGen written as material files of Z-mat and read back as the
    same: each isotropic hardening brick, the kinematic terms (linear and nonlinear,
    named when several), Norton's flow, the thermal strain, cubic and anisotropic
    elasticities (distinct shear moduli: the components named, whatever their order); porous
    plasticity (Gurson with the table of its coalescence and a nucleation, the modified
    Rousselier criterion, the elliptic one with tables, the initial porosity, the kinematic
    terms of GTN and of the elliptic criterion); the materials a file of Z-mat does not hold
    refused."""
    plastic = {"young": 1.0, "poisson": 0.3}
    cases = [
        {"young": 200000.0, "poisson": 0.3},
        {**plastic, "expansion": 1.2e-5, "plasticity": {"yield": 130.0}},
        {**plastic, "plasticity": {"yield": 0.0019999999999924245, "hardening": 0.05}},
        {**plastic, "plasticity": {"yield": 0.002, "isotropic": [{"Q": 0.001, "b": 100.0}]}},
        {**plastic, "plasticity": {"yield": 0.002, "hardening": 0.01,
                                   "isotropic": [{"Q": 0.001, "b": 100.0}]}},
        {**plastic, "plasticity": {"yield": 0.002, "isotropic": [{"Q": 0.001, "b": 100.0},
                                                                 {"Q": 0.002, "b": 10.0}],
                                   "kinematic": [{"C": 0.2, "D": 200.0}, {"C": 0.05, "D": 0.0}],
                                   "norton": {"K": 0.01, "n": 5.0}}},
        {**plastic, "plasticity": {"yield": 0.002, "hardening": 0.01,
                                   "hill": {"11": 1.3, "22": 0.8, "12": 0.7, "31": 1.2}}},
        {"cubic": [168.4, 121.4, 75.4]},
    ]  # fmt: skip
    for config in cases:
        text = zmat.write_zmat(config)
        assert zmat.zmat_material(text).to_config() == Material.from_config(config).to_config()
    text = zmat.write_zmat(cases[5], tmp_path / "chaboche.mat", title="identified\non two runs")
    assert text.splitlines()[:2] == ["% identified", "% on two runs"]
    assert (tmp_path / "chaboche.mat").read_text(encoding="utf-8") == text
    assert "*kinematic nonlinear X1" in text and "*kinematic linear X2" in text
    rounded = zmat.write_zmat(cases[2], digits=3)
    assert "R0 0.002\n" in rounded and "H 0.05\n" in rounded
    with pytest.raises(ValueError, match="digits"):
        zmat.write_zmat(cases[2], digits=0)
    # a general stiffness: its shear moduli distinct, back in their places
    c = np.diag([10.0, 11.0, 12.0, 5.0, 6.0, 7.0])
    c[0, 1] = c[1, 0] = 2.0
    c[1, 2] = c[2, 1] = 4.0
    c[3, 4] = c[4, 3] = 0.5  # a coupling of the shears 23 and 31
    text = zmat.write_zmat({"stiffness": c.tolist()})
    assert "y2323 5.0" in text and "y3131 6.0" in text and "y1212 7.0" in text
    assert "y2331 0.5" in text
    assert np.array_equal(np.array(zmat.zmat_material(text).stiffness), c)
    hexagonal = Material.from_config({"hexagonal": [162.4, 92.0, 69.0, 180.7, 46.7]})
    back = zmat.zmat_material(zmat.write_zmat(hexagonal))
    assert np.array_equal(np.array(back.stiffness), np.array(hexagonal.stiffness))
    # porous plasticity, read back with its initial porosity
    matrix = {
        "yield": 300.0,
        "isotropic": [{"Q": 200.0, "b": 10.0}],
        "norton": {"K": 10.0, "n": 5.0},
    }
    nucleation = {"type": "gaussian", "fN": 0.04, "eN": 0.3, "sN": 0.1, "yield": 0.01}
    porous = [
        {"criterion": "gurson", "porosity": 0.01, "q1": 1.5, "q2": 1.0, "fc": 0.05, "fF": 0.25,
         "nucleation": nucleation, **matrix},
        {"criterion": "rousselier", "porosity": 0.002, "sigma1": 500.0, "D": 2.0, "q2": 1.2,
         "yield": 300.0, "hardening": 50.0},
        {"criterion": "gurson", "porosity": 0.02, "q1": 1.5, "q2": 1.0,
         "kinematic": [{"C": 30000.0, "D": 200.0}, {"C": 5000.0, "D": 0.0}], **matrix},
        {"criterion": "elliptic", "porosity": 0.4, "C": [[0.0, 1.0], [0.5, 4.0]],
         "F": [[0.0, 0.0], [0.6, 0.5]], "yield": 100.0, "kinematic": [{"C": 2000.0, "D": 50.0}]},
    ]  # fmt: skip
    for pp in porous:
        config = {"young": 210000.0, "poisson": 0.3, "porous_plasticity": pp}
        text = zmat.write_zmat(config)
        back = zmat.zmat_material(text).to_config()
        assert back == Material.from_config(config).to_config(), pp["criterion"]
    assert "*porous_criterion elliptic" in text and "      f 0.4\n" in text
    assert "*kinematic nonlinear\n      C 2000.0\n      D 50.0\n" in text
    hardened = zmat.write_zmat({"young": 210000.0, "poisson": 0.3, "porous_plasticity": porous[2]})
    assert "*kinematic nonlinear X1" in hardened and "*kinematic linear X2" in hardened
    gurson = zmat.write_zmat({"young": 210000.0, "poisson": 0.3, "porous_plasticity": porous[0]})
    assert "      fs f\n" in gurson and "      fn 0.04\n" in gurson
    assert zmat.read_zmat(gurson)[0].kind == "porous_plastic"
    terms = [{"Q": 0.001, "b": 100.0}, {"Q": 0.002, "b": 10.0}]
    linear_and_terms = {**plastic, "plasticity": {"yield": 0.002, "hardening": 0.01,
                                                  "isotropic": terms}}  # fmt: skip
    for config, message in (
        ("void", "a void"),
        (linear_and_terms, "no isotropic brick"),
        ({**plastic, "expansion": [1e-5, 2e-5, 3e-5]}, "isotropic thermal expansion"),
        (
            {**plastic, "porous_plasticity": {"criterion": "gurson", "yield": 1.0, "q3": 2.0}},
            "q3 is q1²",
        ),
        (
            {
                **plastic,
                "expansion": 1e-5,
                "porous_plasticity": {"criterion": "gurson", "yield": 1.0},
            },
            "thermal strain",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            zmat.write_zmat(config)


def test_coverage_table():
    statuses = {row[2] for row in zmat.COVERAGE}
    assert statuses <= {"native", "mfront", "elsewhere", "user", "no"}
    categories = {row[0] for row in zmat.COVERAGE}
    assert {"model", "criterion", "flow", "isotropic", "kinematic", "elasticity"} <= categories
    assert zmat.coverage_markdown().count("\n") == len(zmat.COVERAGE) + 2


needs_mfront = pytest.mark.skipif(
    not (mf.available() and shutil.which("mfront")), reason="MGIS and MFront (conda-forge)"
)


@needs_mfront
def test_translations_against_mfront(tmp_path):
    """The native kernels and MFront's brick agree on the same files; Hill's criterion
    yields at R0 / sqrt(3/2 (4 h1 + h2 + h3) / 9) in tension along x (and alike along y),
    and at R0 / sqrt(3 h) in each shear: the order of the shear components checked."""
    from morphogen.homogenization.nonlinear import simulate_point

    path = {"times": [0, 1, 2], "increments": [25, 25], "strain": {"11": [0, 0.01, -0.004]}}
    for k, text in enumerate((CHABOCHE, PLASTIC, HILL.format(**ANISO))):
        native = zmat.zmat_material(text)
        other = zmat.zmat_material(text, native=False, directory=tmp_path, name=f"Cmp{k}")
        a, b = simulate_point(native, path), simulate_point(other, path)
        gap = np.abs(a.stress[:, 0] - b.stress[:, 0]).max() / np.abs(a.stress[:, 0]).max()
        assert gap < 1e-8 and b.plastic[-1] == pytest.approx(a.plastic[-1], rel=1e-8)
    aniso = zmat.zmat_material(HILL.format(**ANISO), native=False, directory=tmp_path,
                               name="Hill")  # fmt: skip
    h = ANISO
    expected = {
        "11": 100.0 / np.sqrt(1.5 * (4 * h["h1"] + h["h2"] + h["h3"]) / 9),
        "22": 100.0 / np.sqrt(1.5 * (h["h1"] + 4 * h["h2"] + h["h3"]) / 9),
        "12": 100.0 / np.sqrt(3 * h["h4"]),
        "23": 100.0 / np.sqrt(3 * h["h5"]),
        "31": 100.0 / np.sqrt(3 * h["h6"]),
    }
    column = {"11": 0, "22": 1, "23": 3, "31": 4, "12": 5}
    for axis, value in expected.items():
        r = simulate_point(
            aniso, {"times": [0, 1], "increments": [200], "strain": {axis: [0, 0.008]}}
        )
        first = int(np.argmax(r.plastic > 1e-12))
        assert r.stress[first, column[axis]] == pytest.approx(value, rel=1e-4), axis  # H p
    for k, text in enumerate((TWO,)):  # several potentials and user-defined flows run
        r = simulate_point(
            zmat.zmat_material(text, directory=tmp_path, name=f"Two{k}"),
            {"times": [0, 1], "increments": [20], "strain": {"11": [0, 0.01]}},
        )
        assert r.plastic[-1] > 0 and r.stress[-1, 0] > 200.0
