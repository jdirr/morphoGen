"""Transport phenomena of a scalar potential beside heat conduction: electrical
conduction, dielectrics, magnetostatics and diffusion share its solvers, each with the
coefficient of the materials for it.

The checks are exact: the laminates (arithmetic and harmonic means, also of anisotropic
layers), the identity of the problems of two transports with the same coefficients, the
insulating phases that count as pores, the tensors turned by the orientations of the
grains, and the Hashin-Shtrikman bounds.
"""

import numpy as np
import pytest

import morphogen as mg
from helpers import box_mesh
from morphogen.homogenization import (
    VOID,
    FEThermalProblem,
    FFTElasticProblem,
    FFTThermalProblem,
    Loading,
    Material,
    apparent_conductivity,
    bounds,
    conductivity_tensor,
    homogenize,
    make_problem,
)
from morphogen.homogenization.apparent import PHYSICS, PROPERTIES, TENSORS, TRANSPORT_PROPERTIES
from morphogen.homogenization.materials import TRANSPORT, material_table, transport_physics

#: The coefficient of each transport other than the thermal one
OTHERS = {p: n for p, n in TRANSPORT.items() if p != "thermal"}


def random_image(shape=(10, 12, 8), fraction=0.4, seed=3):
    rng = np.random.default_rng(seed)
    data = (rng.random(shape) < fraction).astype(np.uint8)
    return mg.VoxelImage(data, spacing=(1.0, 0.8, 1.25), periodic=True)


def layers(n=8, thickness=3):
    """Layers normal to z: phase 1 in the first ``thickness`` planes of ``n``."""
    data = np.zeros((4, 4, n), dtype=np.uint8)
    data[:, :, :thickness] = 1
    return mg.VoxelImage(data, spacing=1.0, periodic=True)


def test_the_properties_of_every_transport_are_registered():
    assert set(TRANSPORT_PROPERTIES) == set(TRANSPORT)
    for physics, (scalar, tensor) in TRANSPORT_PROPERTIES.items():
        assert scalar == TRANSPORT[physics] and tensor == f"{scalar}_tensor"
        assert scalar in PROPERTIES and tensor in TENSORS
        assert PHYSICS[scalar] == PHYSICS[tensor] == physics
        assert transport_physics(physics)
    assert not transport_physics("elastic")
    with pytest.raises(ValueError, match="physics"):
        transport_physics("acoustic")


@pytest.mark.parametrize("physics", sorted(OTHERS))
def test_a_transport_with_the_coefficients_of_the_conduction_is_the_conduction(physics):
    """The same problem to the bit: the solvers only read another coefficient."""
    name = OTHERS[physics]
    img = random_image()
    heat = {0: Material(conductivity=1.0), 1: Material(conductivity=17.0)}
    other = {0: Material(**{name: 1.0}), 1: Material(**{name: 17.0})}
    ref = homogenize(img, heat, properties=["conductivity", "conductivity_tensor"])
    out = homogenize(img, other, properties=[name, f"{name}_tensor"])
    assert out[f"{name}.periodic"].property == name
    assert out[f"{name}.periodic"].value == ref["conductivity.periodic"].value
    assert out[f"{name}.periodic"].solution.physics == physics
    assert np.array_equal(
        out[f"{name}_tensor.periodic"]["tensor"], ref["conductivity_tensor.periodic"]["tensor"]
    )


def test_all_the_transports_of_a_composite_are_solved_together():
    """Within the Hashin-Shtrikman bounds: the blocks of a white noise are cubes, whose
    second-order tensors are isotropic on average (blocks of 3 voxels: the trilinear
    elements of single voxels are too stiff by more than the width of the bounds)."""
    rng = np.random.default_rng(5)
    noise = (rng.random((7, 7, 7)) < 0.4).astype(np.uint8)
    img = mg.VoxelImage(np.kron(noise, np.ones((3, 3, 3), dtype=np.uint8)), 1.0, periodic=True)
    mats = {
        0: Material(young=1.0, poisson=0.3, conductivity=1.0, electrical_conductivity=1e-3,
                    permittivity=2.5, permeability=1.0, diffusivity=4.0),
        1: Material(young=9.0, poisson=0.2, conductivity=30.0, electrical_conductivity=5.0,
                    permittivity=80.0, permeability=200.0, diffusivity=0.5),
    }  # fmt: skip
    names = ["bulk_modulus", *(f"{name}_tensor" for name in TRANSPORT.values())]
    out = homogenize(img, mats, properties=names)
    fractions = mg.analysis.phase_fractions(img)
    limits = bounds(fractions, mats)
    for name in TRANSPORT.values():
        value = out[f"{name}_tensor.periodic"]["mean"]
        b = limits[name]
        assert b["reuss"] < b["hs_lower"] < value < b["hs_upper"] < b["voigt"]


def test_laminate_of_anisotropic_dielectrics_is_exact():
    """Layers normal to z: the in-plane block is the arithmetic mean of ``K_pp - K_pn
    K_nn^-1 K_np`` corrected by the continuity of the normal flux, the normal component the
    harmonic mean (the exact laminate formula of any second-order tensor)."""
    k1 = np.array([[5.0, 0.6, 0.4], [0.6, 3.0, -0.3], [0.4, -0.3, 2.0]])
    k0 = np.array([[1.0, -0.1, 0.2], [-0.1, 1.5, 0.1], [0.2, 0.1, 0.7]])
    f1 = 3.0 / 8.0
    mats = {0: Material(permittivity=k0), 1: Material(permittivity=k1)}
    out = homogenize(layers(), mats, properties=["permittivity_tensor"], tol=1e-12)
    k = out["permittivity_tensor.periodic"]["tensor"]

    # exact: with n = z, D_n and the in-plane E are uniform
    def parts(m):
        return m[:2, :2], m[:2, 2], m[2, 2]

    fr = ((1.0 - f1, k0), (f1, k1))
    knn = 1.0 / sum(f / parts(m)[2] for f, m in fr)
    a = sum(f * parts(m)[1] / parts(m)[2] for f, m in fr)
    kpp = sum(f * (parts(m)[0] - np.outer(parts(m)[1], parts(m)[1]) / parts(m)[2]) for f, m in fr)
    exact = np.zeros((3, 3))
    exact[2, 2] = knn
    exact[:2, 2] = exact[2, :2] = knn * a
    exact[:2, :2] = kpp + knn * np.outer(a, a)
    assert np.allclose(k, exact, rtol=1e-9, atol=1e-11)


def test_an_insulating_phase_is_a_pore_of_its_physics_only():
    img = random_image(seed=8)
    insulated = {
        0: Material(young=1.0, poisson=0.3, electrical_conductivity=0.0, conductivity=1.0),
        1: Material(young=5.0, poisson=0.3, electrical_conductivity=7.0, conductivity=2.0),
    }
    porous = {0: VOID, 1: Material(conductivity=7.0)}
    out = homogenize(img, insulated, properties=["electrical_conductivity", "conductivity"])
    ref = homogenize(img, porous, properties=["conductivity"])
    assert out["electrical_conductivity.periodic"].value == ref["conductivity.periodic"].value
    sol = out["electrical_conductivity.periodic"].solution
    assert sol.material_fraction == pytest.approx(img.data.mean())
    # the heat still flows through both phases, and the stresses too
    assert out["conductivity.periodic"].solution.material_fraction == pytest.approx(1.0)
    assert insulated[0].inert("electrical") and not insulated[0].inert("thermal")
    assert not insulated[0].inert("elastic")
    table = material_table(img.data, insulated, "electrical")
    assert np.array_equal(table.mat >= 0, img.data.ravel() == 1)
    # every phase insulating: no current at all
    dead = {p: Material(electrical_conductivity=0.0) for p in (0, 1)}
    none = homogenize(img, dead, properties=["electrical_conductivity"])
    assert none["electrical_conductivity.periodic"].value == 0.0


def test_a_crystal_turns_its_tensors_with_its_grains():
    """One crystal turned by 90 degrees about z in half of the box: its principal
    diffusivities along x and y are swapped there, and the layers normal to x are in
    series along x and in parallel along y and z."""
    d = (3.0, 1.0, 2.0)
    crystal = Material(diffusivity=d)
    assert crystal.symmetry == "anisotropic" and crystal.oriented("diffusion")
    assert not crystal.oriented("thermal")
    data = np.ones((8, 2, 2), dtype=np.uint8)
    grains = np.zeros((8, 2, 2), dtype=np.int64)
    grains[4:] = 1
    img = mg.VoxelImage(data, spacing=1.0, periodic=True)
    euler = np.array([[0.0, 0.0, 0.0], [90.0, 0.0, 0.0]])
    out = homogenize(img, {1: crystal}, properties=["diffusivity_tensor"], tol=1e-12,
                     grains=grains, orientations=euler)  # fmt: skip
    k = out["diffusivity_tensor.periodic"]["tensor"]
    exact = np.diag([2.0 / (1.0 / 3.0 + 1.0), 0.5 * (1.0 + 3.0), 2.0])
    assert np.allclose(k, exact, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("bc", ["kubc", "periodic", "subc", "mixed"])
def test_finite_elements_solve_every_transport(bc):
    def phase(c):
        return (c[:, 2] > 0.4).astype(np.int64)

    mesh = box_mesh((3, 3, 4), upper=(1.0, 1.0, 0.8), phase=phase)
    heat = {0: Material(conductivity=1.0), 1: Material(conductivity=6.0)}
    magnetic = {0: Material(permeability=1.0), 1: Material(permeability=6.0)}
    ref = FEThermalProblem(mesh, heat, bc)
    problem = make_problem(mesh, magnetic, "magnetic", bc)
    assert isinstance(problem, FEThermalProblem) and problem.physics == "magnetic"
    for axis in range(3):
        a = apparent_conductivity(problem, axis)
        b = apparent_conductivity(ref, axis)
        assert a.property == "permeability" and a.solution.physics == "magnetic"
        assert a.value == pytest.approx(b.value, rel=1e-12)
    # layers normal to z (two elements thick each), periodic: in series along z, in
    # parallel in their plane
    if bc == "periodic":
        k, _ = conductivity_tensor(problem)
        assert k[0, 0] == pytest.approx(0.5 * (1.0 + 6.0), rel=1e-9)
        assert k[2, 2] == pytest.approx(1.0 / (0.5 / 1.0 + 0.5 / 6.0), rel=1e-9)


def test_finite_elements_leave_the_insulators_out():
    def phase(c):  # an insulating layer in the middle of the box, normal to x
        return (np.abs(c[:, 0] - 0.5) > 0.25).astype(np.int64)

    mesh = box_mesh((4, 3, 3), upper=(1.0, 1.0, 1.0), phase=phase)
    mats = {0: Material(diffusivity=0.0, conductivity=1.0), 1: Material(diffusivity=2.0)}
    problem = make_problem(mesh, mats, "diffusion", "periodic")
    assert problem.has_voids and set(np.unique(problem.mesh.phase)) == {1}
    k, _ = conductivity_tensor(problem)
    # half the box diffuses, in parallel with nothing along y and z; nothing crosses x
    assert np.allclose(np.diag(k), [0.0, 1.0, 1.0], atol=1e-9)
    # the same problem as the heat through a mesh of the conducting half
    ref = make_problem(mesh, {0: VOID, 1: Material(conductivity=2.0)}, "thermal", "kubc")
    kubc = make_problem(mesh, mats, "diffusion", "kubc")
    assert np.allclose(conductivity_tensor(kubc)[0], conductivity_tensor(ref)[0], rtol=1e-12)
    with pytest.raises(ValueError, match="porous"):
        make_problem(mesh, mats, "diffusion", "subc")


def test_problems_take_the_physics_of_their_kind_only():
    img = random_image()
    mats = {0: Material(young=1.0, poisson=0.3, permittivity=1.0),
            1: Material(young=2.0, poisson=0.3, permittivity=3.0)}  # fmt: skip
    with pytest.raises(ValueError, match="physics"):
        FFTThermalProblem(img, mats, physics="elastic")
    with pytest.raises(ValueError, match="physics"):
        FFTElasticProblem(img, mats, physics="dielectric")
    with pytest.raises(ValueError, match="physics"):
        make_problem(img, mats, "acoustic")
    with pytest.raises(ValueError, match="no electrical_conductivity"):
        make_problem(img, mats, "electrical")
    problem = make_problem(img, mats, "dielectric")
    # a gradient of any transport loads the problem; a strain does not
    a = problem.solve(Loading.gradient([1.0, 0.0, 0.0]))
    b = problem.solve(Loading.gradient([1.0, 0.0, 0.0], physics="dielectric"))
    assert np.array_equal(a.stress, b.stress) and a.physics == "dielectric"
    with pytest.raises(ValueError, match="elastic"):
        problem.solve(Loading.strain(np.eye(3)))


def test_loadings_name_their_physics():
    lo = Loading.from_spec({"physics": "electrical", "gradient": {"1": 2.0}})
    assert lo.physics == "electrical" and lo.kind == "mixed"
    assert lo.to_dict()["physics"] == "electrical"
    assert Loading.parse("physics=diffusion G1=1 Q2=0 Q3=0").physics == "diffusion"
    assert Loading.gradient([1, 0, 0]).physics == "thermal"
    assert "physics" not in Loading.gradient([1, 0, 0]).to_dict()
    assert Loading.strain(np.eye(3)).physics == "elastic"
    assert Loading.mixed(gradient={"1": 1}, flux={"2": 0, "3": 0}, physics="magnetic").physics == (
        "magnetic"
    )
    with pytest.raises(ValueError, match="physics"):
        Loading.gradient([1, 0, 0], physics="elastic")
    with pytest.raises(ValueError, match="physics"):
        Loading("strain", np.zeros(6), physics="thermal")
    with pytest.raises(ValueError, match="physics"):
        Loading.from_spec({"physics": "sound", "gradient": {"1": 1.0}})


def test_named_loadings_are_solved_by_the_problem_of_their_physics():
    img = layers()
    mats = {0: Material(conductivity=1.0, electrical_conductivity=10.0),
            1: Material(conductivity=3.0, electrical_conductivity=40.0)}  # fmt: skip
    out = homogenize(
        img,
        mats,
        properties=[],
        loadings={
            "heat": {"gradient": {"3": 1.0}},
            "current": {"physics": "electrical", "gradient": {"3": 1.0}},
            "driven": {"physics": "electrical", "flux": {"1": -2.0}},
        },
        tol=1e-12,
    )
    f1 = 3.0 / 8.0
    series = lambda a, b: 1.0 / ((1.0 - f1) / a + f1 / b)  # noqa: E731
    assert out["heat.periodic"].components()["Q3"] == pytest.approx(-series(1.0, 3.0), rel=1e-9)
    assert out["current.periodic"].components()["Q3"] == pytest.approx(
        -series(10.0, 40.0), rel=1e-9
    )
    assert out["current.periodic"].solution.physics == "electrical"
    # a current imposed along the layers: the gradient that drives it through them in parallel
    parallel = (1.0 - f1) * 10.0 + f1 * 40.0
    assert out["driven.periodic"].components()["G1"] == pytest.approx(2.0 / parallel, rel=1e-9)


def test_materials_of_transport_coefficients():
    m = Material(permittivity=(2.0, 2.0, 2.0))
    assert m.permittivity == 2.0 and m.isotropic and m.has("dielectric")
    assert not m.has("thermal") and not m.has("elastic") and not m.elastic
    t = Material(young=1.0, poisson=0.3, permeability=[[2.0, 0.5, 0.0], [0.5, 3.0, 0.0],
                                                       [0.0, 0.0, 1.0]])  # fmt: skip
    assert not t.isotropic and t.oriented("magnetic") and not t.oriented("elastic")
    assert np.allclose(t.matrix("magnetic"), np.array(t.permeability))
    turned = t.matrix("magnetic", np.array([90.0, 0.0, 0.0]))
    assert np.allclose(turned, [[3.0, -0.5, 0.0], [-0.5, 2.0, 0.0], [0.0, 0.0, 1.0]])
    assert t.mean_conductivity("voigt", "magnetic") == pytest.approx(2.0)
    for spec in (m, t, Material(conductivity=2.0, diffusivity=0.0, density=3.0)):
        assert Material.from_config(spec.to_config()) == spec
    assert Material.from_config({"electrical_conductivity": 5}).electrical_conductivity == 5.0
    with pytest.raises(ValueError, match="non-negative"):
        Material(diffusivity=-1.0)
    with pytest.raises(ValueError, match="positive definite"):
        Material(permittivity=[[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="void"):
        Material(void=True, permittivity=1.0)
    with pytest.raises(ValueError, match="no permeability"):
        m.matrix("magnetic")
    with pytest.raises(ValueError, match="needs"):
        Material()


def test_bounds_of_every_transport_and_of_insulators():
    mats = {0: Material(electrical_conductivity=0.0, permittivity=1.0),
            1: Material(electrical_conductivity=4.0, permittivity=10.0)}  # fmt: skip
    b = bounds({0: 0.6, 1: 0.4}, mats)
    assert set(b) == {"electrical_conductivity", "permittivity"}
    assert b["electrical_conductivity"]["reuss"] == b["electrical_conductivity"]["hs_lower"] == 0.0
    assert b["electrical_conductivity"]["voigt"] == pytest.approx(1.6)
    # Hashin-Shtrikman's upper bound: the coating of the insulator by the conductor
    assert b["electrical_conductivity"]["hs_upper"] == pytest.approx(
        4.0 * (1.0 - 3.0 * 0.6 / (3.0 - 0.4)), rel=1e-12
    )
    eps = b["permittivity"]
    assert eps["reuss"] < eps["hs_lower"] < eps["hs_upper"] < eps["voigt"]
    # a polycrystal of one uniaxial crystal: the bounds of its principal values
    crystal = Material(permittivity=(4.0, 4.0, 9.0))
    poly = bounds({1: 1.0}, {1: crystal})["permittivity"]
    assert poly["reuss"] < poly["hs_lower"] < poly["hs_upper"] < poly["voigt"]
    assert poly["voigt"] == pytest.approx(17.0 / 3.0)


# -- configurations, the command line and RVE studies ------------------------------------


def mosaic_config(tmp_path, **homogenization):
    return {
        "model": {"type": "voronoi", "n_cells": 20, "phase_fractions": [0.4, 0.6]},
        "domain": {"size": 4.0, "periodic": True},
        "seed": 11,
        "realizations": 2,
        "voxels": {"shape": 12},
        "homogenization": {
            "method": "fft",
            "materials": {
                0: {"electrical_conductivity": 1.0, "permittivity": [2.0, 2.0, 3.0],
                    "diffusivity": 0.0, "density": 1.0},
                1: {"electrical_conductivity": 50.0, "permittivity": 9.0, "diffusivity": 1.5,
                    "density": 3.0},
            },
            "boundary_conditions": ["periodic"],
            "properties": ["electrical_conductivity", "permittivity_tensor", "diffusivity"],
            **homogenization,
        },
        "outputs": {"directory": str(tmp_path / "out"), "name": "mosaic"},
    }  # fmt: skip


def test_batch_run_of_transport_properties(tmp_path):
    from morphogen import pipeline
    from morphogen.config import ConfigError, load_config

    current = {"physics": "electrical", "gradient": {"3": 1}}
    data = mosaic_config(tmp_path, fields=True, loadings={"current": current})
    cfg = load_config(data)
    assert cfg.homogenization.series() == ["electrical_conductivity.periodic",
                                           "diffusivity.periodic"]  # fmt: skip
    assert cfg.homogenization.tensors() == ["permittivity_tensor.periodic"]
    outdir = pipeline.run(cfg)
    rows = pipeline.read_rows(outdir / "results.csv")
    assert len(rows) == 2
    for row in rows:
        sigma = row["electrical_conductivity.periodic"]
        assert 1.0 < sigma < 50.0
        # relative to the mean of the phases (Voigt), and per unit density
        mean = row["fraction_0"] * 1.0 + row["fraction_1"] * 50.0
        assert row["electrical_conductivity.periodic.normalized"] == pytest.approx(sigma / mean)
        assert row["electrical_conductivity.periodic.specific"] == pytest.approx(
            sigma / row["density"]
        )
        assert row["permittivity_tensor.periodic.C33"] > row["permittivity_tensor.periodic.C11"]
        assert row["permittivity_tensor.periodic.mean"] == pytest.approx(
            sum(row[f"permittivity_tensor.periodic.C{i}{i}"] for i in (1, 2, 3)) / 3.0
        )
        assert 0.0 < row["diffusivity.periodic"] < 1.5 * row["fraction_1"]
        assert row["current.periodic.Q3"] < 0.0 and row["current.periodic.G3"] == 1.0
    long = pipeline.read_rows(outdir / "homogenization.csv")
    assert {r["property"] for r in long} == {"electrical_conductivity", "diffusivity", "current"}
    images = mg.io.read_vti(outdir / "mosaic_0001_electrical_conductivity_periodic.vti")
    assert {"flux_1", "gradient_1", "dissipation"} <= {im.name for im in images}
    # a phase without the coefficient of a property asked for
    del data["homogenization"]["materials"][0]["diffusivity"]
    with pytest.raises((ConfigError, ValueError), match="diffusivity"):
        pipeline.run(load_config(data))


def test_rve_study_of_a_transport_property(tmp_path):
    """The statistics of the apparent electrical conductivity and of the permittivity
    tensor over the realizations, as those of the thermal conductivity."""
    import json

    from morphogen import pipeline
    from morphogen.config import load_config

    data = mosaic_config(tmp_path)
    del data["realizations"]
    data["domain"] = {"periodic": True}
    data["model"] = {"type": "voronoi", "intensity": 0.5, "phase_fractions": [0.4, 0.6]}
    data["voxels"] = {"voxel_size": 0.5}
    data["rve"] = {"sizes": [3.0, 5.0], "realizations": [6, 4], "bootstrap": 20,
                   "properties": ["electrical_conductivity", "permittivity_tensor",
                                  "volume_fraction"]}  # fmt: skip
    outdir = pipeline.run(load_config(data))
    report = json.loads((outdir / "rve_report.json").read_text(encoding="utf-8"))
    series = report["properties"]["electrical_conductivity.periodic"]
    stats = series["statistics"]
    assert [s["n"] for s in stats] == [6, 4]
    assert stats[0]["std"] > stats[1]["std"] > 0.0  # the scatter falls with the volume
    assert series["fit"]["gamma"] > 0 and series["rve"][0]["volume"] > 0
    assert np.isfinite(stats[0]["isotropy"])
    b = report["bounds"]["electrical_conductivity"]  # Wiener's: a phase is a crystal
    assert b["reuss"] < stats[1]["mean"] < b["voigt"]
    tensor = report["tensors"]["permittivity_tensor.periodic"]
    assert tensor["property"] == "permittivity_tensor" and tensor["bc"] == "periodic"
    assert "electric displacement" in tensor["notation"]
    assert "dielectric impermeability" in tensor["notation"]
    mean = np.array(tensor["volumes"][-1]["stiffness"])
    assert mean.shape == (3, 3) and mean[2, 2] > mean[0, 0]
    md = (outdir / "rve_report.md").read_text(encoding="utf-8")
    assert "electrical_conductivity" in md and "permittivity" in md
    assert (outdir / "rve_permittivity_tensor.csv").exists()


def test_window_study_of_a_transport_tensor(tmp_path):
    """The window method on the tensor of another transport: the mean of the windows that
    tile the realization is its apparent tensor (Hill's lemma)."""
    from morphogen.rve import window_tensors

    rng = np.random.default_rng(2)
    noise = (rng.random((6, 6, 6)) < 0.5).astype(np.uint8)
    img = mg.VoxelImage(np.kron(noise, np.ones((2, 2, 2), dtype=np.uint8)), 1.0, periodic=True)
    mats = {0: Material(permeability=1.0), 1: Material(permeability=8.0)}
    windows = window_tensors(img, mats, [3, 6], property="permeability_tensor", tol=1e-10)
    assert windows.tensors[0].shape == (64, 3, 3)
    for tensors in windows.tensors:
        assert np.allclose(tensors.mean(axis=0), windows.apparent, rtol=1e-7, atol=1e-9)
    ref = window_tensors(
        img,
        {0: Material(conductivity=1.0), 1: Material(conductivity=8.0)},
        [3, 6],
        property="conductivity_tensor",
        tol=1e-10,
    )
    assert np.array_equal(windows.tensors[0], ref.tensors[0])
    stats = windows.statistics()
    assert stats.property == "permeability_tensor" and "magnetic induction" in stats.notation


def test_cli_homogenize_transport(tmp_path, capsys):
    import json

    from morphogen import cli

    rng = np.random.default_rng(0)
    image = mg.VoxelImage((rng.random((6, 6, 6)) < 0.5).astype(np.uint8), periodic=True)
    path = tmp_path / "img.vti"
    mg.io.save_image(image, path)
    args = ["homogenize", str(path), "--material", "0=electrical_conductivity:0,permittivity:1",
            "1=electrical_conductivity:10,permittivity:2/2/5", "--property",
            "electrical_conductivity", "permittivity_tensor"]  # fmt: skip
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "electrical_conductivity" in out and "bounds of permittivity" in out
    load = "push: physics=dielectric G1=0 G2=0 G3=1"
    assert cli.main([*args, "--json", "--loading", load]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["results"][0]["property"] == "electrical_conductivity"
    tensor = np.array(res["permittivity_tensor.periodic"]["tensor"])
    assert tensor[2, 2] > tensor[0, 0]
    assert res["loadings"]["push.periodic"]["Q3"] == pytest.approx(-tensor[2, 2], rel=1e-6)
    assert cli.main(["homogenize", str(path), "--material", "0=void", "1=young:1,poisson:0.3",
                     "--property", "permeability"]) == 2  # fmt: skip
    assert "no permeability" in capsys.readouterr().err
