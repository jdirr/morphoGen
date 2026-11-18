"""Weak couplings of elasticity: swelling and poroelasticity.

Exact relations: Levin's for the swelling of two-phase composites (and its uniform-field
case), those of Biot and Gassmann for a porous medium whose solid is homogeneous
(isotropic, then a crystal), and the reciprocity that gives the tensor of Biot of any
solid from its drained solutions; with the FFT solver of voxel images, the finite elements
of meshes and the extended finite elements of level sets.
"""

import numpy as np
import pytest

import morphogen as mg
from helpers import box_mesh
from morphogen.homogenization import (
    VOID,
    LevelSetSample,
    Material,
    homogenize,
    make_problem,
    poroelastic_tensors,
    stiffness_tensor,
    swelling_tensor,
)
from morphogen.homogenization.core import Loading
from morphogen.homogenization.poroelastic import (
    POROELASTIC_ENTRIES,
    UNDRAINED_ENTRIES,
    pressure_materials,
    undrained,
)

K_S, G_S = 10.0, 6.0
SOLID = Material.from_moduli(K_S, G_S)
ONE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def voigt(tensor, strain=False):
    """A symmetric tensor in Voigt notation (engineering shears of a strain)."""
    t = np.asarray(tensor)
    f = 2.0 if strain else 1.0
    return np.array([t[0, 0], t[1, 1], t[2, 2], f * t[1, 2], f * t[2, 0], f * t[0, 1]])


def pore_image(n=12, radii=(0.3, 0.3, 0.3)):
    """A periodic cell with an ellipsoidal pore (phase 1) at its centre."""
    x = (np.arange(n) + 0.5) / n - 0.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    inside = (X / radii[0]) ** 2 + (Y / radii[1]) ** 2 + (Z / radii[2]) ** 2 < 1.0
    return mg.VoxelImage(inside.astype(np.uint8), 1.0 / n, periodic=True)


def two_phases(n=10, seed=0, fraction=0.4):
    rng = np.random.default_rng(seed)
    blocks = (rng.random((n // 2, n // 2, n // 2)) < fraction).astype(np.uint8)
    return mg.VoxelImage(np.kron(blocks, np.ones((2, 2, 2), np.uint8)), 1.0 / n, periodic=True)


def homogeneous_solid(out, hydrostatic):
    """The relations of a homogeneous solid, of strain ``hydrostatic`` (Voigt) under a unit
    hydrostatic tension: ``b = 1 - C_d : S_s : 1``, ``1 / N = (b - phi 1) : S_s : 1``."""
    b = ONE - out["tensor"] @ hydrostatic
    inverse_n = (b - out["porosity"] * ONE) @ hydrostatic
    return b, 1.0 / inverse_n


# --------------------------------------------------------------------------- swelling


def test_swelling_follows_levins_relation():
    """Two isotropic phases, whatever their arrangement: ``B* = beta_1 1 + (beta_2 -
    beta_1) (S* : 1 - 1 / (3 K_1)) / (1 / (3 K_2) - 1 / (3 K_1))``."""
    image = two_phases()
    k1, k2, b1, b2 = 4.0, 30.0, 0.02, 0.003
    mats = {0: Material.from_moduli(k1, 2.0).__class__.from_config(
                {"bulk": k1, "shear": 2.0, "swelling": b1}),
            1: Material.from_config({"bulk": k2, "shear": 12.0, "swelling": b2})}  # fmt: skip
    tensor, sol = swelling_tensor(image, mats, tol=1e-10)
    assert np.abs(sol.stress).max() < 1e-8
    stiffness, _ = stiffness_tensor(make_problem(image, mats, "elastic", tol=1e-10))
    p = np.linalg.inv(stiffness) @ ONE  # the strain of a unit hydrostatic tension
    exact = b1 * ONE + (b2 - b1) * (p - ONE / (3 * k1)) / (1 / (3 * k2) - 1 / (3 * k1))
    assert np.allclose(voigt(tensor, strain=True), exact, rtol=1e-6, atol=1e-9)
    assert b2 < tensor[0, 0] < b1
    # the thermal analogy: the same tensor as an expansion of the same coefficients
    thermal = {p: Material.from_config({**m.to_config(), "expansion": m.swelling})
               for p, m in mats.items()}  # fmt: skip
    out = homogenize(image, thermal, properties=["thermal_expansion", "swelling"], tol=1e-10)
    assert np.allclose(out["swelling.periodic"]["tensor"], tensor, atol=1e-10)
    assert np.allclose(out["thermal_expansion.periodic"]["tensor"], tensor, atol=1e-10)
    assert out["swelling.periodic"]["B11"] == pytest.approx(tensor[0, 0])
    assert out["swelling.periodic"]["mean"] == pytest.approx(np.trace(tensor) / 3)


def test_swelling_of_phases_that_swell_alike_and_of_a_dry_phase():
    image = two_phases(seed=3)
    soft, stiff = {"bulk": 4.0, "shear": 2.0}, {"bulk": 30.0, "shear": 12.0}
    # the same strain everywhere: no stress, whatever the stiffnesses
    same = {0: Material.from_config({**soft, "swelling": 0.01}),
            1: Material.from_config({**stiff, "swelling": 0.01})}  # fmt: skip
    tensor, _ = swelling_tensor(image, same, tol=1e-10)
    assert np.allclose(tensor, 0.01 * np.eye(3), atol=1e-10)
    # stiff fibres that stay dry restrain the matrix; a crystal swells along its axes
    dry = {0: Material.from_config({**soft, "swelling": 0.01}), 1: Material.from_config(stiff)}
    tensor, _ = swelling_tensor(image, dry, tol=1e-10)
    assert 0.0 < tensor[0, 0] < 0.01
    wood = Material.from_config({"hexagonal": [10.0, 4.0, 3.0, 40.0, 5.0],
                                 "swelling": [0.03, 0.03, 0.001]})  # fmt: skip
    assert wood.oriented("elastic") and wood.to_config()["swelling"][2][2] == 0.001
    tensor, _ = swelling_tensor(two_phases(n=4), {0: wood, 1: wood})
    assert np.allclose(tensor, np.diag([0.03, 0.03, 0.001]), atol=1e-10)
    for bad, match in (
        ({"swelling": 0.1}, "elastic parameters"),
        ({**soft, "swelling": [1.0, 2.0]}, "swelling: one coefficient"),
        ({**soft, "swelling": float("nan")}, "swelling: finite"),
    ):
        with pytest.raises(ValueError, match=match):
            Material.from_config(bad)
    with pytest.raises(ValueError, match="does not swell"):
        Material(void=True, swelling=0.1)
    with pytest.raises(ValueError, match="no material swells"):
        swelling_tensor(image, {0: SOLID, 1: SOLID})


def test_swelling_by_finite_elements_and_boundary_conditions():
    """A layered box mesh under periodic conditions: the layers share their in-plane
    strain and are free along their normal, which gives their swelling in closed form."""
    k1, g1, k2, g2, b1, b2 = 4.0, 2.0, 30.0, 12.0, 0.02, 0.0
    mats = {0: Material.from_config({"bulk": k1, "shear": g1, "swelling": b1}),
            1: Material.from_config({"bulk": k2, "shear": g2})}  # fmt: skip
    mesh = box_mesh((2, 2, 4), upper=(1.0, 1.0, 1.0),
                    phase=lambda c: (c[:, 2] > 0.5).astype(int))  # fmt: skip
    tensor, _ = swelling_tensor(mesh, mats, "periodic", tol=1e-12)
    # layers of equal thickness: the in-plane strain e shared, their normal stresses zero
    m1, m2 = k1 + 4 * g1 / 3, k2 + 4 * g2 / 3  # oedometric moduli
    l1, l2 = k1 - 2 * g1 / 3, k2 - 2 * g2 / 3
    # biaxial moduli of the layers free to strain along their normal
    q1, q2 = 2 * (l1 + g1) - 2 * l1**2 / m1, 2 * (l2 + g2) - 2 * l2**2 / m2
    plane = (q1 * b1 + q2 * b2) / (q1 + q2)
    normal = 0.5 * sum(b + 2 * lam / m * (b - plane)
                       for b, lam, m in ((b1, l1, m1), (b2, l2, m2)))  # fmt: skip
    assert np.allclose(np.diag(tensor), [plane, plane, normal], rtol=1e-8)
    # the other conditions (not the mixed ones: no stress-free state of the whole strain)
    out = homogenize(mesh, mats, properties=["swelling"], bcs=["kubc", "subc", "mixed"])
    assert set(out) == {"swelling.kubc", "swelling.subc"}
    for res in out.values():
        assert 0.0 < res["B11"] < b1 and 0.0 < res["B33"] < 1.5 * b1


# ----------------------------------------------------------------------- poroelasticity


def test_pores_in_a_homogeneous_isotropic_solid():
    """``b = 1 - K_d / K_s``, ``1 / N = (b - phi) / K_s`` and Gassmann's undrained
    modulus, for a spherical pore."""
    out = poroelastic_tensors(pore_image(), {0: SOLID, 1: VOID}, tol=1e-10, fluid_bulk=2.0)
    phi, kd, b = out["porosity"], out["bulk_modulus"], out["biot_coefficient"]
    assert phi == pytest.approx(pore_image().data.mean())
    assert b == pytest.approx(1 - kd / K_S, rel=1e-7)
    assert np.allclose(out["biot"], b * np.eye(3), atol=1e-8)
    assert phi < b < 1.0
    assert out["biot_modulus"] == pytest.approx(K_S / (b - phi), rel=1e-6)
    gassmann = kd + b**2 / (phi / 2.0 + (b - phi) / K_S)
    assert out["undrained_bulk_modulus"] == pytest.approx(gassmann, rel=1e-7)
    assert out["storage_modulus"] == pytest.approx((gassmann - kd) / b**2, rel=1e-6)
    assert out["skempton_coefficient"] == pytest.approx((1 - kd / gassmann) / b, rel=1e-6)
    assert out["b11"] == pytest.approx(b) and out["B11"] == pytest.approx(
        out["skempton_coefficient"], rel=1e-6
    )
    assert set(POROELASTIC_ENTRIES + UNDRAINED_ENTRIES) <= set(out)
    assert np.allclose(out["tensor"] @ out["compliance"], np.eye(6), atol=1e-10)


def test_an_oblate_pore_has_a_tensor_of_biot():
    """A flat pore opens along its axis: ``b = 1 - C_d : S_s : 1`` is not spherical."""
    out = poroelastic_tensors(pore_image(radii=(0.4, 0.4, 0.15)), {0: SOLID, 1: VOID}, tol=1e-10)
    b, n = homogeneous_solid(out, ONE / (3 * K_S))
    assert np.allclose(voigt(out["biot"]), b, atol=1e-8)
    assert out["biot_modulus"] == pytest.approx(n, rel=1e-6)
    assert out["b33"] > 1.5 * out["b11"] and out["b11"] == pytest.approx(out["b22"], rel=1e-6)


def test_pores_in_a_crystal():
    """A homogeneous anisotropic solid: its strain under a hydrostatic stress is not
    spherical, and ``b = 1 - C_d : S_s : 1`` holds with it."""
    zinc = Material.hexagonal(165.0, 31.0, 50.0, 62.0, 40.0)
    out = poroelastic_tensors(pore_image(), {0: zinc, 1: VOID}, tol=1e-10)
    hydrostatic = np.linalg.inv(np.array(zinc.stiffness)) @ ONE
    assert hydrostatic[2] > 3 * hydrostatic[0]
    b, n = homogeneous_solid(out, hydrostatic)
    assert np.allclose(voigt(out["biot"]), b, atol=1e-7)
    assert out["biot_modulus"] == pytest.approx(n, rel=1e-6)
    mats = pressure_materials({0: zinc, 1: VOID})
    assert mats[1] is VOID and np.allclose(voigt(mats[0].expansion_tensor(), True), -hydrostatic)


def test_the_tensor_of_biot_of_a_heterogeneous_solid_by_reciprocity():
    """Two solid phases around the pores: the formulae of a homogeneous solid no longer
    hold, but the pores still strain by ``b : E`` in the drained problems (``b = 1 - (1 -
    phi) 1 : <A>_s``), which the solution of the pressure must recover."""
    image = pore_image()
    data = image.data.copy()
    data[:, :, :6][data[:, :, :6] == 0] = 2  # a stiff half of the solid
    image = mg.VoxelImage(data, image.spacing, periodic=True)
    mats = {0: SOLID, 1: VOID, 2: Material.from_moduli(40.0, 25.0)}
    out = poroelastic_tensors(image, mats, tol=1e-11)
    problem = make_problem(image, mats, "elastic", tol=1e-11)
    b = np.zeros(6)
    for k in range(6):
        unit = np.zeros(6)
        unit[k] = 1.0
        sol = problem.solve(Loading.strain(unit))
        solid = sum(p.fraction * p.mean["strain"][:3].sum() for p in sol.phases)
        b[k] = (1.0 if k < 3 else 0.0) - solid
    assert np.allclose(voigt(out["biot"]), b, atol=1e-7)
    # neither solid alone gives it
    for k_s in (K_S, 40.0):
        assert abs(out["biot_coefficient"] - (1 - out["bulk_modulus"] / k_s)) > 0.02
    assert out["b33"] != pytest.approx(out["b11"], rel=1e-3)
    assert out["biot_modulus"] > 0


def test_poroelasticity_by_finite_elements_and_level_sets():
    """The relations of a homogeneous solid hold in every discretization: a box mesh with
    a cubic pore (periodic and kinematic uniform conditions), a level set cut by
    extended finite elements."""

    def phase(c):
        return (np.abs(c - 0.5).max(axis=1) < 0.25).astype(int)

    mesh = box_mesh((4, 4, 4), upper=(1.0, 1.0, 1.0), phase=phase)
    for bc in ("periodic", "kubc"):
        out = poroelastic_tensors(mesh, {0: SOLID, 1: VOID}, bc, tol=1e-12)
        assert out["porosity"] == pytest.approx(0.125)
        b, n = homogeneous_solid(out, ONE / (3 * K_S))
        assert np.allclose(voigt(out["biot"]), b, atol=1e-8)
        assert out["biot_modulus"] == pytest.approx(n, rel=1e-6)
    # the kinematic conditions stiffen the drained sample: a smaller coefficient of Biot
    periodic = poroelastic_tensors(mesh, {0: SOLID, 1: VOID}, "periodic", tol=1e-12)
    assert out["bulk_modulus"] > periodic["bulk_modulus"]
    assert out["biot_coefficient"] < periodic["biot_coefficient"]
    with pytest.raises(ValueError, match="periodic or kinematic uniform"):
        poroelastic_tensors(mesh, {0: SOLID, 1: VOID}, "subc")
    with pytest.raises(ValueError, match="no pore"):
        poroelastic_tensors(mesh, {0: SOLID, 1: SOLID})


def test_poroelasticity_of_a_level_set():
    n = 8
    x = (np.arange(n) + 0.0) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    values = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) - 0.3  # < 0 inside
    sample = LevelSetSample(values, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), inside=1, outside=0)
    out = poroelastic_tensors(sample, {0: SOLID, 1: VOID}, tol=1e-12)
    # the sphere, as the polyhedron of its level set on this coarse grid
    assert out["porosity"] == pytest.approx(4 / 3 * np.pi * 0.3**3, rel=0.1)
    b, n_biot = homogeneous_solid(out, ONE / (3 * K_S))
    assert np.allclose(voigt(out["biot"]), b, atol=1e-7)
    assert out["biot_modulus"] == pytest.approx(n_biot, rel=1e-5)
    # close to the estimate of Mori and Tanaka for spherical pores, b = phi / (1 - (1 -
    # phi) alpha) with alpha = (1 + nu) / (3 (1 - nu))
    alpha = (1 + SOLID.poisson) / (3 * (1 - SOLID.poisson))
    estimate = out["porosity"] / (1 - (1 - out["porosity"]) * alpha)
    assert out["biot_coefficient"] == pytest.approx(estimate, rel=0.1)


def test_undrained_limits():
    c = np.array(SOLID.matrix("elastic")) * 0.7
    b = 0.4 * np.eye(3)
    stiff = undrained(c, b, 50.0, 0.2, 1e12)  # an incompressible fluid: M -> N
    assert stiff["storage_modulus"] == pytest.approx(50.0, rel=1e-6)
    soft = undrained(c, b, 50.0, 0.2, 1e-9)  # a gas: the drained medium
    assert np.allclose(soft["undrained_stiffness"], c, atol=1e-6)
    assert soft["skempton_coefficient"] < 1e-6
    rigid = undrained(c, np.zeros((3, 3)), float("inf"), 0.0, 2.0)
    assert np.isinf(rigid["storage_modulus"]) and np.allclose(rigid["undrained_stiffness"], c)
    with pytest.raises(ValueError, match="fluid_bulk"):
        undrained(c, b, 50.0, 0.2, 0.0)


def test_weak_couplings_in_configurations_and_statistics(tmp_path, capsys):
    import json

    import yaml

    from morphogen import cli, pipeline
    from morphogen.config import ConfigError, load_config

    data = {
        "model": {"type": "boolean_spheres", "radius": 0.3, "volume_fraction": 0.15},
        "domain": {"periodic": True},
        "seed": 5,
        "voxels": {"voxel_size": 0.125},
        "homogenization": {
            "method": "fft",
            "materials": {0: {"bulk": K_S, "shear": G_S, "swelling": 0.01}, 1: "void"},
            "properties": ["poroelastic", "swelling"],
            "fluid_bulk_modulus": 2.0,
        },
        "rve": {"sizes": [1.0, 1.5], "realizations": [3, 2], "bootstrap": 0,
                "properties": ["poroelastic.biot_coefficient", "poroelastic.biot_modulus.periodic",
                               "poroelastic.skempton_coefficient", "swelling.mean"]},
        "outputs": {"directory": str(tmp_path / "out"), "name": "foam"},
    }  # fmt: skip
    cfg = load_config(data)
    assert cfg.homogenization.fluid_bulk_modulus == 2.0
    outdir = pipeline.run(cfg)
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    rows = pipeline.read_rows(outdir / "results.csv")
    for row in rows:  # a homogeneous solid: the exact relations, the pores do not swell it
        phi = row["poroelastic.periodic.porosity"]
        c = {ij: row[f"poroelastic.periodic.C{ij}"] for ij in ("11", "22", "33", "12", "13", "23")}
        kd = (
            c["11"] + c["22"] + c["33"] + 2 * (c["12"] + c["13"] + c["23"])
        ) / 9.0  # 1 : C : 1 / 9
        assert row["poroelastic.periodic.biot_coefficient"] == pytest.approx(1 - kd / K_S, rel=1e-5)
        assert row["swelling.periodic.mean"] == pytest.approx(0.01, rel=1e-6)
        assert phi == pytest.approx(row["fraction_1"])
        assert row["poroelastic.periodic.C11"] > 0 and row["poroelastic.periodic.b11"] > phi
    stats = report["properties"]["poroelastic.biot_coefficient.periodic"]["statistics"]
    assert [s["n"] for s in stats] == [3, 2]
    first = [r["poroelastic.periodic.biot_coefficient"] for r in rows
             if r["volume"] == pytest.approx(1.0)]  # fmt: skip
    assert stats[0]["mean"] == pytest.approx(np.mean(first), rel=1e-12)
    assert {"poroelastic.biot_modulus.periodic", "poroelastic.skempton_coefficient.periodic",
            "swelling.mean.periodic"} <= set(report["properties"])  # fmt: skip
    homog = data["homogenization"]
    for change, match in (
        ({"materials": {0: {"bulk": K_S, "shear": G_S}, 1: "void"}}, "swelling of a material"),
        ({"materials": {0: {"bulk": K_S, "shear": G_S, "swelling": 0.01},
                        1: {"bulk": 1.0, "shear": 1.0}}}, "needs pores"),
        ({"fluid_bulk_modulus": -1.0}, "positive bulk modulus"),
        ({"properties": ["swelling"]}, "that of the property 'poroelastic'"),
    ):  # fmt: skip
        with pytest.raises(ConfigError, match=match):
            load_config({**data, "homogenization": {**homog, **change}, "rve": None})
    for bad, match in (
        ("poroelastic.b99", "unknown result 'b99'"),
        ("poroelastic.biot_coefficient.subc", "computed under"),
        ("swelling.A11", "unknown result 'A11'"),
    ):
        with pytest.raises(ConfigError, match=match):
            load_config({**data, "rve": {**data["rve"], "properties": [bad]}})
    no_fluid = {k: v for k, v in homog.items() if k != "fluid_bulk_modulus"}
    with pytest.raises(ConfigError, match="unknown result 'skempton_coefficient'"):
        load_config({**data, "homogenization": no_fluid})
    # the command line
    path = tmp_path / "pore.vti"
    mg.io.save_image(pore_image(n=8), path)
    materials = {0: {"bulk": K_S, "shear": G_S, "swelling": 0.01}, 1: "void"}
    (tmp_path / "materials.yaml").write_text(yaml.safe_dump(materials), encoding="utf-8")
    args = ["homogenize", str(path), "--materials", str(tmp_path / "materials.yaml"),
            "--property", "poroelastic", "swelling", "--fluid-bulk-modulus", "2.0"]  # fmt: skip
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "tensor of Biot" in text and "skempton_coefficient" in text and "swelling" in text
    assert cli.main([*args, "--json"]) == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["poroelastic.periodic"]["biot_coefficient"] > 0
    assert loaded["swelling.periodic"]["mean"] == pytest.approx(0.01, rel=1e-6)
