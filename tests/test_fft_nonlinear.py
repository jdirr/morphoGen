"""The nonlinear FFT solver of voxel images: the periodic element on trilinear voxels
(B-bar and F-bar, or full integration), each Newton correction solved by conjugate
gradients preconditioned in Fourier space, in small and finite strains, with von Mises
and crystal plasticity."""

import dataclasses
import json

import numpy as np
import pytest
import yaml

from helpers import laminate_path
from morphogen import cli, io, pipeline
from morphogen.config import ConfigError, load_config
from morphogen.homogenization import (
    CrystalPlasticity,
    Loading,
    LoadingPath,
    Material,
    NonlinearFFTProblem,
    Plasticity,
    homogenize,
    make_problem,
    simulate_point,
)
from morphogen.homogenization import nonlinear as nl
from morphogen.homogenization.elements import tensor_to_voigt
from morphogen.homogenization.finite import gradient_of, macro_part
from morphogen.homogenization.nonlinear import _update
from morphogen.image import VoxelImage
from morphogen.orientations import euler_to_matrix, random_orientations

E, NU = 200e3, 0.3
#: Copper-like cubic elasticity (MPa)
CUBIC = (168e3, 121e3, 75e3)


def crystal(expansion=None, **params) -> Material:
    """A cubic crystal with octahedral slip."""
    spec = {"tau0": 100.0, "K": 5.0, "n": 20.0, **params}
    return dataclasses.replace(
        Material.cubic(*CUBIC, expansion=expansion),
        crystal_plasticity=CrystalPlasticity.from_config(spec),
    )


def _state(problem, q, e):
    """Discretization, residual of the equilibrium, integral of the stress and tangents of
    the points of an elastic state (fluctuation ``q``, macroscopic components ``e``)."""
    disc = problem._discretization()
    n_pts = problem.kind.size
    nc = 9 if problem.finite else 6
    grads, wts = np.zeros((n_pts, nc)), np.zeros(n_pts)
    sigma, tangent = np.zeros((n_pts, nc)), np.zeros((n_pts, nc, nc))
    old = np.zeros((n_pts, problem.behaviours.width))
    new = np.zeros_like(old)
    disc.measures(q, e, grads, wts)
    failed = _update(problem.finite, problem.kind, problem.point_mat, problem.pmat,
                     problem.behaviours.data, grads, 0.0, 1.0, old, new, sigma, tangent, 0,
                     1.0)  # fmt: skip
    assert failed == 0
    r_q, total, _ = disc.forces(sigma, wts)
    if problem.finite:
        total = macro_part(total)
    return disc, r_q, total, tangent


@pytest.mark.parametrize("kinematics", ["small", "finite"])
@pytest.mark.parametrize("integration", ["bbar", "full"])
def test_tangent_of_the_voxels(kinematics, integration):
    """The tangent operator applied matrix-free, its coupling with the macroscopic strain
    and the macroscopic tangent are the derivatives of the nodal forces and of the
    integral of the stress (finite differences, elastic phases, a strained state), and
    the operator is symmetric."""
    rng = np.random.default_rng(5)
    data = (rng.random((4, 3, 5)) < 0.5).astype(np.uint8)
    mats = {0: Material(young=70e3, poisson=0.33), 1: Material.cubic(*CUBIC)}
    grains = np.arange(data.size).reshape(data.shape)
    orientations = random_orientations(rng, data.size)
    image = VoxelImage(data, spacing=(0.5, 0.7, 0.4))
    problem = NonlinearFFTProblem(image, mats, grains=grains, orientations=orientations,
                                  kinematics=kinematics, integration=integration)  # fmt: skip
    n_q = 3 * data.size
    q = 0.01 * rng.normal(size=n_q)
    e = np.array([0.01, -0.004, 0.006, 0.008, -0.005, 0.003])
    disc, _, _, tangent = _state(problem, q, e)
    solver = disc.linearize(tangent, None, True)
    u, v = rng.normal(size=n_q), rng.normal(size=n_q)
    kv = solver.apply(v)
    h = 1e-6
    fd = (_state(problem, q + h * v, e)[1] - _state(problem, q - h * v, e)[1]) / (2 * h)
    assert np.allclose(kv, fd, rtol=0, atol=1e-6 * np.abs(fd).max())
    assert u @ kv == pytest.approx(v @ solver.apply(u), rel=1e-10)
    unknown = np.array([0, 2, 3, 5])
    g, kee = disc.coupling(tangent, None, unknown)
    for k, j in enumerate(unknown):
        d = np.zeros(6)
        d[j] = h
        up, down = _state(problem, q, e + d), _state(problem, q, e - d)
        assert np.allclose(g[:, k], (up[1] - down[1]) / (2 * h), rtol=0,
                           atol=1e-6 * np.abs(g).max())  # fmt: skip
        assert np.allclose(kee[:, j], (up[2] - down[2]) / (2 * h), rtol=0,
                           atol=1e-6 * np.abs(kee).max())  # fmt: skip
    # K_Ew = K_wE^T: the macroscopic stress varies with the fluctuation as the forces with E
    dtotal = (_state(problem, q + h * v, e)[2] - _state(problem, q - h * v, e)[2]) / (2 * h)
    assert np.allclose(dtotal[unknown], g.T @ v, rtol=0, atol=1e-6 * np.abs(g.T @ v).max())


def test_homogeneous_image():
    """A homogeneous image follows its material point: von Mises elasto-viscoplasticity
    along a cyclic path mixing a strain, a shear stress and a change of temperature, and a
    single crystal in small and finite strains."""
    m = Material(young=E, poisson=NU, expansion=1e-5, plasticity=Plasticity(
        300.0, isotropic=[(200.0, 30.0)], kinematic=[(5e4, 300.0)], viscosity=150.0,
        exponent=6.0))  # fmt: skip
    path = {"times": [0, 10, 20, 30], "increments": [8, 8, 8],
            "strain": {"11": [0.01, -0.01, 0.0]}, "stress": {"12": [40, -40, 0]},
            "temperature": [50, 50, 0]}  # fmt: skip
    ref = simulate_point(m, path)
    image = VoxelImage(np.zeros((3, 2, 4), dtype=np.uint8), spacing=(0.5, 0.7, 0.3))
    res = NonlinearFFTProblem(image, {0: m}).run(path)
    assert np.allclose(res.stress, ref.stress, rtol=0, atol=2e-6 * np.abs(ref.stress).max())
    assert np.allclose(res.strain, ref.strain, rtol=0, atol=5e-8)
    assert res.plastic[-1] == pytest.approx(ref.plastic[-1], rel=1e-6)
    assert res.phases[0]["plastic"][-1] == pytest.approx(ref.plastic[-1], rel=1e-6)
    m = crystal(expansion=1e-5, tau0=80.0, Q=40.0, b=8.0, C=3e4, D=300.0, K=30.0, n=8.0)
    euler = np.array([20.0, 35.0, 50.0])
    path = {"times": [0, 10, 20], "increments": [6, 6], "strain": {"11": [0.01, -0.005]},
            "stress": {"12": [40, -40]}, "temperature": [50, 0]}  # fmt: skip
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    grains = np.zeros(image.shape, int)
    for kinematics in ("small", "finite"):
        ref = simulate_point(m, path, orientation=euler, kinematics=kinematics)
        problem = NonlinearFFTProblem(image, {0: m}, grains=grains, orientations=euler[None],
                                      kinematics=kinematics)  # fmt: skip
        res = problem.run(path)
        scale = np.abs(ref.stress).max()
        assert np.allclose(res.stress, ref.stress, rtol=0, atol=2e-6 * scale)
        assert np.allclose(res.strain, ref.strain, rtol=0, atol=5e-8)
        assert res.plastic[-1] == pytest.approx(ref.plastic[-1], rel=1e-6)
        if kinematics == "finite":
            assert np.allclose(res.cauchy, ref.cauchy, rtol=0, atol=2e-6 * scale)


def test_elastic_paths_match_the_linear_solver():
    """With linear behaviours and full integration, a path reaches the solution of the
    condensed linear FFT problem, pores included; B-bar changes the discretization."""
    m0 = Material(young=2.8, poisson=0.39, expansion=95e-6)
    m1 = Material(young=89.0, poisson=0.3, expansion=5.5e-6)
    rng = np.random.default_rng(3)
    data = (rng.random((6, 5, 7)) < 0.4).astype(np.uint8)
    data[rng.random(data.shape) < 0.15] = 2
    mats = {0: m0, 1: m1, 2: Material(void=True)}
    image = VoxelImage(data)
    loading = Loading.mixed(strain={"11": 0.01}, stress={"22": 0, "33": 0, "23": 0, "31": 0,
                            "12": 0.002}, temperature=50.0)  # fmt: skip
    sol = make_problem(image, mats, "elastic").solve(loading)
    path = {"times": [0, 1], "increments": 1, "strain": {"11": 0.01}, "stress": {"12": 0.002},
            "temperature": 50.0}  # fmt: skip
    full = NonlinearFFTProblem(image, mats, integration="full", newton_tol=1e-9).run(path)
    strain = tensor_to_voigt(sol.strain, strain=True)
    stress = tensor_to_voigt(sol.stress, strain=False)
    assert np.allclose(full.strain[-1], strain, rtol=0, atol=1e-9)
    assert np.allclose(full.stress[-1], stress, rtol=0, atol=1e-8)
    assert sorted(full.phases) == [0, 1]  # the pores have no integration points
    bbar = NonlinearFFTProblem(image, mats).run(path)
    assert not np.allclose(bbar.strain[-1], strain, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="integration"):
        NonlinearFFTProblem(image, mats, integration="reduced")


def test_plastic_laminate():
    """A periodic laminate of an elasto-viscoplastic and an elastic layer: the voxels
    follow the exact laminate response along a cyclic path with a shear stress and a
    change of temperature; the fields are those of the voxels."""
    soft = Material(young=70e3, poisson=0.33, plasticity=Plasticity(
        100.0, isotropic=[(80.0, 20.0)], kinematic=[(2e4, 200.0)], viscosity=50.0,
        exponent=5.0))  # fmt: skip
    hard = Material(young=E, poisson=NU, expansion=12e-6)
    path = {"times": [0, 1, 2, 3], "increments": [6, 6, 6],
            "strain": {"11": [0.006, -0.004, 0.003]}, "stress": {"12": [20, -20, 0]},
            "temperature": [30, 30, 0]}  # fmt: skip
    strain, stress = laminate_path([soft, hard], [0.5, 0.5], path)
    data = np.zeros((2, 3, 4), dtype=np.uint8)
    data[:, :, 2:] = 1
    image = VoxelImage(data, spacing=(0.5, 1 / 3, 0.25))
    res = NonlinearFFTProblem(image, {0: soft, 1: hard}).run(path, fields=True)
    assert np.allclose(res.stress, stress, rtol=0, atol=2e-6 * np.abs(stress).max())
    assert np.allclose(res.strain, strain, rtol=0, atol=1e-8)
    assert sorted(res.fields) == [1.0, 2.0, 3.0] and res.support is image
    fields = res.fields[3.0]
    assert fields["stress"].shape == (*data.shape, 6)
    assert fields["displacement"].shape == (*data.shape, 3)
    soft_p = fields["p"][data == 0]
    assert np.ptp(soft_p) < 1e-6 * soft_p.max() and np.all(fields["p"][data == 1] == 0)
    assert res.phases[0]["plastic"][-1] == pytest.approx(soft_p.mean(), rel=1e-9)


@pytest.mark.slow
def test_bbar_does_not_lock():
    """Elastic cubes (22 %) in a plastic matrix: fully integrated voxels lock under the
    isochoric plastic flow (the flow stress is twice too high on 10^3 voxels), while with
    B-bar the flow stress changes little when the voxels are refined, in small and finite
    strains."""
    matrix = Material(young=70e3, poisson=0.33, plasticity=Plasticity(100.0, hardening=200.0))
    mats = {0: matrix, 1: Material(young=400e3, poisson=0.2)}
    path = {"times": [0, 1], "increments": 10, "strain": {"11": 0.02}}
    flow = {}
    for n in (10, 20):
        centres = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) / n
        data = np.all(np.abs(centres - 0.5) < 0.3, axis=1).reshape(n, n, n).astype(np.uint8)
        image = VoxelImage(data, spacing=(1 / n,) * 3)
        for integration in ("bbar", "full") if n == 10 else ("bbar",):
            for kinematics in ("small", "finite") if integration == "bbar" else ("small",):
                res = NonlinearFFTProblem(image, mats, integration=integration,
                                          kinematics=kinematics).run(path)  # fmt: skip
                flow[n, integration, kinematics] = res.stress[-1, 0]
    assert flow[10, "full", "small"] > 1.8 * flow[10, "bbar", "small"]
    for kinematics in ("small", "finite"):
        coarse, fine = flow[10, "bbar", kinematics], flow[20, "bbar", kinematics]
        assert fine < coarse < 1.08 * fine
    # the nominal stress of finite strains is below the small strain stress at 2 %
    nominal = flow[20, "bbar", "finite"]
    assert nominal < flow[20, "bbar", "small"] < 1.05 * nominal


def voxel_polycrystal(n=10, n_cells=6, seed=1):
    """Periodic Voronoi grains on a grid of voxels, random orientations."""
    rng = np.random.default_rng(seed)
    centres = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) / n
    seeds = rng.random((n_cells, 3))
    d = centres[:, None, :] - seeds[None, :, :]
    d -= np.round(d)
    grains = np.argmin((d**2).sum(-1), axis=1).reshape(n, n, n)
    image = VoxelImage(np.zeros((n, n, n), dtype=np.uint8), spacing=(1 / n,) * 3)
    return image, grains, random_orientations(rng, n_cells)


@pytest.mark.slow
def test_polycrystal():
    """Tension of a voxel polycrystal: the flow stress lies between the Sachs (largest
    Schmid factor, 1/2) and Taylor (3.06) estimates; in finite strains the lattices rotate
    and a tension with free faces is a uniaxial Cauchy stress."""
    m = crystal(tau0=60.0, K=10.0, n=15.0)
    image, grains, orientations = voxel_polycrystal()
    path = {"times": [0, 10], "increments": 10, "strain": {"33": 0.01}}
    small = NonlinearFFTProblem(image, {0: m}, grains=grains, orientations=orientations).run(path)
    assert small.info["cuts"] == 0
    assert 2.0 * 60.0 < small.stress[-1, 2] < 3.06 * 61.0 * 1.2  # viscous stress: a few MPa
    finite = NonlinearFFTProblem(image, {0: m}, grains=grains, orientations=orientations,
                                 kinematics="finite").run(path, fields=[10.0])  # fmt: skip
    assert np.allclose(finite.stress[:, 2], small.stress[:, 2], rtol=0.03, atol=1.0)
    f = np.eye(3) + gradient_of(finite.strain[-1]).reshape(3, 3)
    assert f[1, 0] == f[2, 0] == f[2, 1] == 0.0  # upper triangular
    cauchy = finite.cauchy[-1]
    assert np.all(np.abs(cauchy[[0, 1, 3, 4, 5]]) < 1e-6 * cauchy[2])
    turned = finite.fields[10.0]["orientation"].reshape(-1, 3)
    g0 = euler_to_matrix(orientations[grains.ravel()])
    g1 = euler_to_matrix(turned)
    trace = np.trace(g1 @ np.swapaxes(g0, 1, 2), axis1=1, axis2=2)
    angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
    assert 0.0 < angle.max() < 5.0  # lattice rotations of a few tenths of a degree


def test_fewer_iterations_for_the_same_path(monkeypatch):
    """A stiff inclusion in a viscoplastic matrix, pulled with free faces in 20 equal
    steps (the macroscopic unknowns are condensed, their block kept from a correction to
    the next): the steps predicted at the second order where the response does not turn,
    and the condensed block updated by the secants of the corrections, take fewer Newton
    and linear iterations than the first-order prediction and the block as it was
    computed — each of the two alone as well —, without a divided step, for the same
    stresses to the tolerance of the equilibria."""
    soft = Material(young=70e3, poisson=0.33, plasticity=Plasticity(
        100.0, isotropic=[(80.0, 20.0)], kinematic=[(2e4, 200.0)], viscosity=50.0,
        exponent=5.0))  # fmt: skip
    hard = Material(young=400e3, poisson=0.2)
    n = 6
    centres = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) / n - 0.5
    data = np.zeros((n, n, n), dtype=np.uint8)
    data.ravel()[(centres**2).sum(axis=1) < 0.35**2] = 1
    image = VoxelImage(data, spacing=(1 / n,) * 3)
    path = {"times": [0, 20], "increments": 20, "strain": {"11": 0.03}}
    secant = nl._secant

    def run(order: int, update: bool):
        monkeypatch.setattr(nl, "_secant", secant if update else lambda schur, *pair: schur)
        problem = NonlinearFFTProblem(image, {0: soft, 1: hard})
        problem.prediction = order
        res = problem.run(path)
        assert res.info["cuts"] == 0
        return res, int(res.iterations.sum()), res.info["linear_iterations"]

    plain, newton, linear = run(1, False)
    assert np.all(plain.iterations[-5:] == 2)  # (a steady flow: two corrections a step)
    for order, update in ((2, False), (1, True), (2, True)):
        res, its, lin = run(order, update)
        assert its < newton and lin < linear
        assert np.allclose(res.stress, plain.stress, rtol=0, atol=2e-6 * plain.stress.max())
    assert NonlinearFFTProblem.prediction == 2  # the default: both
    assert its <= 0.9 * newton and lin <= 0.9 * linear


def test_steps_are_divided():
    """A step that does not converge is divided; Voce's saturation is integrated exactly
    whatever the steps."""
    m = Material(young=E, poisson=NU, plasticity=Plasticity(300.0, isotropic=[(200.0, 30.0)]))
    image = VoxelImage(np.zeros((2, 2, 2), dtype=np.uint8))
    tension = LoadingPath.uniaxial(0, times=[0, 1], strain=0.05, increments=1)
    res = NonlinearFFTProblem(image, {0: m}, max_newton=1).run(tension)
    assert res.info["cuts"] > 0 and res.n_steps > 1
    s, p = res.stress[-1, 0], res.plastic[-1]
    assert s == pytest.approx(300 + 200 * (1 - np.exp(-30 * p)), rel=1e-6)
    with pytest.raises(RuntimeError, match="no convergence"):
        NonlinearFFTProblem(image, {0: m}, max_newton=0, max_cuts=2).run(tension)


def test_homogenize_configurations_and_command_line(tmp_path):
    """Loading paths on images through homogenize, configurations (method fft), the
    pipeline (curves, fields as .vti) and the command line."""
    soft = Material(young=70e3, poisson=0.33, plasticity=Plasticity(100.0, hardening=1000.0))
    mats = {0: soft, 1: Material(young=E, poisson=NU)}
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[2:] = 1
    image = VoxelImage(data, spacing=(0.25,) * 3)
    paths = {"tension": {"times": [0, 1], "increments": 4, "strain": {"11": 0.004}}}
    out = homogenize(image, mats, properties=[], paths=paths, nonlinear={"integration": "full"})
    res = out["tension.periodic"]
    assert sorted(out) == ["tension.periodic"] and res.components()["E11"] == pytest.approx(0.004)
    with pytest.raises(ValueError, match="conforming mesh"):
        homogenize(image, mats, bcs=["kubc"], properties=[], paths=paths)
    from helpers import box_mesh

    # a mesh takes the option too (the mean dilatation of its 10-node tetrahedra)
    mesh = box_mesh((1, 1, 1))
    for integration in ("full", "bbar"):
        on_mesh = homogenize(mesh, mats, properties=[], paths=paths,
                             nonlinear={"integration": integration})  # fmt: skip
        assert on_mesh["tension.periodic"].components()["E11"] == pytest.approx(0.004)
    with pytest.raises(ValueError, match="integration"):
        homogenize(mesh, mats, properties=[], paths=paths, nonlinear={"integration": "one"})
    # configurations and the pipeline: a voxel polycrystal with crystal plasticity
    cfg = yaml.safe_load(cli._templates()["homogenization_crystal_plasticity_fft"])
    assert cfg["homogenization"]["method"] == "fft"
    cfg["model"]["n_cells"] = 3
    cfg["voxels"] = {"shape": 6}
    cfg["homogenization"]["paths"]["tension"] = {
        "times": [0, 6],
        "increments": 3,
        "strain": {"11": 0.006},
    }
    cfg["outputs"]["directory"] = str(tmp_path / "out")
    spec = load_config(cfg).homogenization
    assert spec.options()["nonlinear"]["kinematics"] == "finite"
    outdir = pipeline.run(load_config(cfg))
    rows = pipeline.read_rows(outdir / "polycrystal_0000_tension_periodic.csv")
    assert len(rows) == 4 and {"LE11", "TS11", "TS11_1", "p"} <= set(rows[-1])
    assert float(rows[-1]["TS11"]) > float(rows[-1]["S11"]) > 0
    fields = io.vtk.read_vti(outdir / "polycrystal_0000_tension_periodic_1.vti")
    names = {f.name for f in fields}
    assert {"stress_11", "p", "orientation_1", "displacement_1"} <= names
    bad = json.loads(json.dumps(cfg))
    bad["homogenization"]["nonlinear"]["integration"] = "simpson"
    with pytest.raises(ConfigError, match="integration"):
        load_config(bad)
    # "reduced" is B-bar; the finite elements of a mesh take the option as the voxels do
    other = json.loads(json.dumps(cfg))
    other["homogenization"]["nonlinear"]["integration"] = "reduced"
    assert load_config(other).homogenization.options()["nonlinear"]["integration"] == "bbar"
    other["homogenization"]["method"] = "fe"
    other["mesh"] = {"size": 0.5}
    other["homogenization"]["nonlinear"]["integration"] = "full"
    assert load_config(other).homogenization.options()["nonlinear"]["integration"] == "full"
    # the command line, on an image file
    image_file = tmp_path / "laminate.vti"
    io.save_image(image, image_file)
    (tmp_path / "mats.yaml").write_text(
        yaml.safe_dump({0: soft.to_config(), 1: {"young": E, "poisson": NU}}), encoding="utf-8"
    )
    report = tmp_path / "report.json"
    code = cli.main([
        "homogenize", str(image_file), "--materials", str(tmp_path / "mats.yaml"),
        "--path", "tension: t=0,1 n=4 E11=0.004", "--curves", str(tmp_path / "curves"),
        "--fields", str(tmp_path / "fields"), "-o", str(report),
    ])  # fmt: skip
    assert code == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    bbar = homogenize(image, mats, properties=[], paths=paths)["tension.periodic"]
    assert data["paths"]["tension.periodic"]["S11"] == pytest.approx(bbar.stress[-1, 0], rel=1e-6)
    assert (tmp_path / "curves" / "laminate_tension_periodic.csv").exists()
    assert (tmp_path / "fields" / "laminate_tension_periodic_1.vti").exists()
