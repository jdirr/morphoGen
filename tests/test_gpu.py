"""The GPU backend of the FFT solvers (CuPy): the choice of the device, and the same
solutions as on the CPU (linear problems, the behaviours of the points, the tangent
operators of the voxels, loading paths). The comparisons are skipped without a usable
GPU."""

import dataclasses
import json

import numpy as np
import pytest

from morphogen import cli, gpu, io
from morphogen.config import ConfigError, load_config
from morphogen.homogenization import (
    CrystalPlasticity,
    Loading,
    Material,
    NonlinearFFTProblem,
    Plasticity,
    apparent_bulk_modulus,
    apparent_conductivity,
    homogenize,
    make_problem,
)
from morphogen.image import VoxelImage
from morphogen.orientations import random_orientations

needs_gpu = pytest.mark.skipif(not gpu.available(), reason="no usable GPU (CuPy, CUDA)")


def porous_image(shape=(10, 9, 11), seed=1):
    rng = np.random.default_rng(seed)
    data = (rng.random(shape) < 0.4).astype(np.uint8)
    data[rng.random(shape) < 0.12] = 2
    return VoxelImage(data, spacing=(0.5, 0.6, 0.4))


MATS = {
    0: Material(young=70e3, poisson=0.33, conductivity=1.0, expansion=2e-5),
    1: Material(young=400e3, poisson=0.2, conductivity=30.0, expansion=5e-6),
    2: Material(void=True),
}


@needs_gpu
def test_kernels_compile_after_gmsh_opened_a_file(tmp_path):
    """Gmsh narrows the search path of the DLLs of the process when it opens a file
    (Windows): NVRTC, which opens its library of builtins by name at each compilation, still
    finds it, kept loaded since the probe of the GPU (the tests of the GPU failed after
    those of Gmsh in the same process)."""
    import time

    import cupy

    gmsh = pytest.importorskip("gmsh")
    from morphogen.mesh import voxel_mesh

    assert gpu.available()
    path = tmp_path / "m.msh"
    io.save_mesh(voxel_mesh(VoxelImage(data=np.ones((2, 2, 2), dtype=np.uint8))), path)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
    finally:
        gmsh.finalize()
    nonce = time.time_ns() % 1_000_003  # a new source: CuPy's cache of kernels left out
    twice = cupy.ElementwiseKernel("float64 x", "float64 y", f"y = 2 * x + {nonce}", "twice")
    assert twice(cupy.arange(3.0)).get().tolist() == [nonce, nonce + 2.0, nonce + 4.0]


def test_choice_of_the_device(monkeypatch):
    assert gpu.device_of("cpu") == "cpu"
    assert gpu.device_of("auto") == ("gpu" if gpu.available() else "cpu")
    with pytest.raises(ValueError, match="device"):
        gpu.device_of("tpu")
    monkeypatch.setattr(gpu, "_probe", lambda: (False, "CuPy is not installed"))
    assert gpu.device_of("auto") == "cpu" and gpu.info() == {
        "gpu": None,
        "reason": "CuPy is not installed",
    }
    with pytest.raises(RuntimeError, match="CuPy is not installed"):
        make_problem(porous_image(), MATS, "elastic", device="gpu")
    from helpers import box_mesh

    with pytest.raises(ValueError, match="finite elements run on the CPU"):
        homogenize(box_mesh((1, 1, 1)), {0: MATS[0]}, device="gpu")
    base = {
        "model": {"type": "boolean_spheres", "radius": 0.5, "volume_fraction": 0.2},
        "domain": {"size": 2.0, "periodic": True},
        "voxels": {"shape": 8},
        "homogenization": {"materials": {0: {"young": 7e4, "poisson": 0.33},
                                         1: {"young": 4e5, "poisson": 0.2}}},
    }  # fmt: skip
    for change, match in [
        ({"device": "tpu"}, "device"),
        ({"device": "gpu", "method": "fe"}, "CPU"),
    ]:
        data = json.loads(json.dumps(base))
        data["homogenization"].update(change)
        if change.get("method") == "fe":
            data["mesh"] = {"size": 0.5}
        with pytest.raises(ConfigError, match=match):
            load_config(data)
    data = json.loads(json.dumps(base))
    data["homogenization"]["device"] = "auto"
    assert load_config(data).homogenization.options()["device"] == "auto"


@needs_gpu
def test_linear_fft_on_the_gpu(tmp_path):
    """The apparent moduli, a mixed thermo-elastic loading and the conductivity of a
    porous image: the same solutions and iterations on the GPU."""
    image = porous_image()
    loading = Loading.mixed(strain={"11": 0.01}, stress={"22": 0, "33": 0, "23": 0, "31": 0,
                            "12": 0.5}, temperature=30.0)  # fmt: skip
    out = {}
    for device in ("cpu", "gpu"):
        elastic = make_problem(image, MATS, "elastic", device=device)
        thermal = make_problem(image, MATS, "thermal", device=device)
        k = apparent_bulk_modulus(elastic)
        out[device] = (k.value, elastic.solve(loading).stress, apparent_conductivity(thermal).value,
                       k.solution.info)  # fmt: skip
    (k0, s0, c0, i0), (k1, s1, c1, i1) = out["cpu"], out["gpu"]
    assert k1 == pytest.approx(k0, rel=1e-10) and c1 == pytest.approx(c0, rel=1e-10)
    assert np.allclose(s1, s0, rtol=0, atol=1e-9 * np.abs(s0).max())
    assert (i0.method, i1.method) == ("fft", "fft-gpu") and i1.iterations == i0.iterations
    # the command line
    io.save_image(image, tmp_path / "porous.vti")
    (tmp_path / "mats.yaml").write_text(
        json.dumps({str(p): m.to_config() for p, m in MATS.items()}), encoding="utf-8"
    )
    report = tmp_path / "report.json"
    args = ["homogenize", str(tmp_path / "porous.vti"), "--materials", str(tmp_path / "mats.yaml")]
    code = cli.main([*args, "--device", "gpu", "-o", str(report)])
    assert code == 0
    rows = json.loads(report.read_text(encoding="utf-8"))["results"]
    assert rows[0]["solver"] == "fft-gpu" and rows[0]["value"] == pytest.approx(k0, rel=1e-10)


def _state(problem):
    """Discretization and tangents of a strained elastic state of the voxels (on the
    device of the problem)."""
    disc = problem._discretization()
    n_pts = problem.kind.size
    nc = 9 if problem.finite else 6
    grads, wts, sigma, tangent, old, new = disc.arrays(n_pts, nc, problem.behaviours.width)
    q = 0.01 * np.random.default_rng(1).normal(size=disc.n_q)
    disc.measures(q, np.array([0.01, -0.004, 0.006, 0.008, -0.005, 0.003]), grads, wts)
    assert disc.integrate(grads, 0.0, 1.0, old, new, sigma, tangent, 0, 1.0) == 0
    disc.forces(sigma, wts)
    return disc, tangent, wts


@needs_gpu
@pytest.mark.parametrize("kinematics", ["small", "finite"])
@pytest.mark.parametrize("integration", ["bbar", "full"])
def test_tangent_operators_on_the_gpu(kinematics, integration):
    """The CUDA kernels of the measures, the conjugate stresses and the tangent operators
    of the voxels (B-bar, F-bar and full integration) give those of the CPU to round-off,
    and so do their solutions."""
    import cupy as cp

    rng = np.random.default_rng(5)
    image = porous_image((5, 4, 6), seed=5)
    mats = {0: Material(young=70e3, poisson=0.33), 1: Material.cubic(168e3, 121e3, 75e3),
            2: Material(void=True)}  # fmt: skip
    grains = np.arange(image.data.size).reshape(image.shape)
    orientations = random_orientations(rng, image.data.size)
    out = {}
    v = rng.normal(size=3 * image.data.size)
    for device in ("cpu", "gpu"):
        problem = NonlinearFFTProblem(image, mats, grains=grains, orientations=orientations,
                                      kinematics=kinematics, integration=integration,
                                      device=device)  # fmt: skip
        disc, tangent, wts = _state(problem)
        solver = disc.linearize(tangent, wts, True)
        kv = solver.apply(v) if device == "cpu" else cp.asnumpy(disc(cp.asarray(v)))
        g, kee = disc.coupling(tangent, wts, np.array([0, 2, 3, 5]))
        x, info = solver.solve(kv, tol=1e-10)
        out[device] = (kv, g, kee, x, info.iterations)
    for a, b in zip(out["cpu"][:4], out["gpu"][:4], strict=True):
        assert np.allclose(b, a, rtol=0, atol=1e-12 * np.abs(a).max())
    assert out["cpu"][4] == out["gpu"][4]


def _behaviour_materials():
    """Every behaviour of the points: elasticity with thermal expansion, von Mises
    viscoplasticity (Voce and nonlinear kinematic hardening) and rate-independent
    plasticity, crystals of 12 and 18 slip systems (Franciosi's interactions, kinematic
    hardening), a void, and Hill's criterion, viscous and rate-independent."""
    hill = (1.3, 0.8, 1.1, 0.7, 1.25, 0.9)
    franciosi = [1.0, 1.0, 0.6, 12.3, 1.6, 1.3]
    oct12 = CrystalPlasticity.from_config({"tau0": 60, "Q": 20, "b": 10, "C": 20000, "D": 300,
                                           "K": 10, "n": 15, "interaction": franciosi})  # fmt: skip
    oc18 = CrystalPlasticity.from_config({"families": {"octahedral": {}, "cube": {"tau0": 90}},
                                          "tau0": 70, "K": 20, "n": 8, "Q": 10, "b": 5, "C": 5000,
                                          "D": 100, "interaction": [1.0, 1.4]})  # fmt: skip
    return {
        0: Material(young=70e3, poisson=0.3, expansion=1e-5),
        1: Material(young=70e3, poisson=0.33, plasticity=Plasticity(
            yield_stress=100, hardening=500, isotropic=((50, 20),),
            kinematic=((20000, 200), (5000, 0)), viscosity=50, exponent=5)),
        2: Material(young=200e3, poisson=0.3, expansion=2e-5, plasticity=Plasticity(
            yield_stress=150, hardening=1000, kinematic=((10000, 0),))),
        3: dataclasses.replace(Material.cubic(168e3, 121e3, 75e3), crystal_plasticity=oct12),
        4: dataclasses.replace(Material.cubic(250e3, 160e3, 120e3), crystal_plasticity=oc18),
        5: Material(void=True),
        6: Material(young=70e3, poisson=0.3, plasticity=Plasticity(
            yield_stress=100, hardening=500, isotropic=((50, 20),),
            kinematic=((20000, 200), (5000, 0)), viscosity=50, exponent=5, hill=hill)),
        7: Material(young=200e3, poisson=0.3, expansion=2e-5, plasticity=Plasticity(
            yield_stress=150, hardening=1000, kinematic=((10000, 0),), hill=hill[::-1])),
    }  # fmt: skip


@needs_gpu
@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_behaviours_on_the_gpu(kinematics):
    """The CUDA kernels of the behaviours give the stresses, tangents and internal
    variables of the Numba kernels to round-off, point by point: every behaviour, along
    steps that start from zero, from the current iterate and from the previous step, with
    changes of temperature, up to strains that need substeps."""
    import cupy as cp

    rng = np.random.default_rng(3)
    shape = (6, 5, 7)
    image = VoxelImage(rng.integers(0, 8, size=shape).astype(np.uint8))
    grains = np.arange(image.data.size).reshape(shape)
    orientations = random_orientations(rng, image.data.size)
    scale = 0.004 if kinematics == "small" else 0.02
    factors = [(1.0, 0, 20.0), (1.05, 1, 20.0), (2.0, 2, 40.0)]
    if kinematics == "small":
        factors.append((8.0, 0, 40.0))
    out = {}
    for device in ("cpu", "gpu"):
        problem = NonlinearFFTProblem(image, _behaviour_materials(), grains=grains,
                                      orientations=orientations, kinematics=kinematics,
                                      device=device)  # fmt: skip
        disc = problem._discretization()
        n_pts, nc = problem.kind.size, 9 if problem.finite else 6
        g, _wts, sig, tan, old, new = disc.arrays(n_pts, nc, problem.behaviours.width)
        xp = cp if device == "gpu" else np
        grads = scale * np.random.default_rng(4).normal(size=(n_pts, nc))
        steps = []
        for k, (factor, guess, dtemp) in enumerate(factors):
            g[...] = xp.asarray(factor * grads)
            failed = disc.integrate(g, dtemp, 1.0, old, new, sig, tan, guess, 1.0)
            steps.append((failed, *disc.host(sig.copy(), tan.copy(), new.copy())))
            if k in (1, 2):
                old, new = new, old  # an accepted step
        out[device] = steps
    for cpu, gpu_step in zip(out["cpu"], out["gpu"], strict=True):
        assert cpu[0] == gpu_step[0] == 0
        for a, b, tol in zip(cpu[1:], gpu_step[1:], (1e-12, 1e-10, 1e-12), strict=True):
            assert np.allclose(b, a, rtol=0, atol=tol * np.abs(a).max())


def _porous_materials():
    """Porous plasticity: GTN with Chu and Needleman's nucleation and Tvergaard and
    Needleman's coalescence (the porosity starting near fc), viscous GTN, Rousselier, the
    elliptic criterion of compaction with its tables, and with kinematic hardening GTN
    (Armstrong-Frederick and Prager terms, a nucleation) and the viscous elliptic one."""
    return {
        0: Material(young=200e3, poisson=0.3, expansion=1e-5, porous_plasticity={
            "criterion": "gurson", "yield": 300.0, "hardening": 1000.0, "porosity": 0.04,
            "isotropic": [{"Q": 100.0, "b": 10.0}], "fc": 0.05, "fF": 0.25, "nucleation": {
                "type": "gaussian", "fN": 0.04, "eN": 0.02, "sN": 0.01}}),
        1: Material(young=200e3, poisson=0.3, porous_plasticity={
            "criterion": "gurson", "yield": 300.0, "porosity": 0.02,
            "norton": {"K": 300.0, "n": 5.0}}),
        2: Material(young=200e3, poisson=0.3, porous_plasticity={
            "criterion": "rousselier", "yield": 300.0, "hardening": 800.0, "porosity": 0.01,
            "sigma1": 400.0, "D": 2.0}),
        3: Material(young=10e3, poisson=0.3, porous_plasticity={
            "criterion": "elliptic", "yield": 20.0, "hardening": 50.0, "porosity": 0.4,
            "C": [[0, 1.0], [0.5, 3.0]], "F": [[0, 0.0], [0.5, 0.5]]}),
        4: Material(young=200e3, poisson=0.3, expansion=1e-5, porous_plasticity={
            "criterion": "gurson", "yield": 300.0, "hardening": 500.0, "porosity": 0.02,
            "kinematic": [{"C": 40000.0, "D": 300.0}, {"C": 8000.0, "D": 0.0}],
            "nucleation": {"type": "constant", "A": 0.3}}),
        5: Material(young=10e3, poisson=0.3, porous_plasticity={
            "criterion": "elliptic", "yield": 20.0, "porosity": 0.4,
            "norton": {"K": 10.0, "n": 3.0}, "C": [[0, 1.0], [0.5, 3.0]],
            "F": [[0, 0.0], [0.5, 0.5]], "kinematic": [{"C": 2000.0, "D": 50.0}]}),
    }  # fmt: skip


@needs_gpu
@pytest.mark.parametrize("kinematics", ["small", "finite"])
def test_porous_behaviours_on_the_gpu(kinematics):
    """The CUDA port of porous plasticity gives the stresses, tangents and internal
    variables of the Numba kernel to round-off, point by point: GTN (its nucleation, its
    coalescence), viscous GTN, Rousselier and the elliptic criterion, with kinematic
    hardening GTN and the viscous elliptic criterion (their backstresses), along steps that
    start from zero, from the current iterate and from the previous step, with changes of
    temperature."""
    import cupy as cp

    rng = np.random.default_rng(7)
    shape = (6, 5, 7)
    image = VoxelImage(rng.integers(0, 6, size=shape).astype(np.uint8))
    scale = 0.002 if kinematics == "small" else 0.003
    factors = [(1.0, 0, 20.0), (1.05, 1, 20.0), (2.0, 2, 40.0)]
    out = {}
    for device in ("cpu", "gpu"):
        problem = NonlinearFFTProblem(image, _porous_materials(), kinematics=kinematics,
                                      device=device)  # fmt: skip
        disc = problem._discretization()
        n_pts, nc = problem.kind.size, 9 if problem.finite else 6
        g, _wts, sig, tan, old, new = disc.arrays(n_pts, nc, problem.behaviours.width)
        xp = cp if device == "gpu" else np
        grads = scale * np.random.default_rng(8).normal(size=(n_pts, nc))
        steps = []
        for k, (factor, guess, dtemp) in enumerate(factors):
            g[...] = xp.asarray(factor * grads)
            failed = disc.integrate(g, dtemp, 1.0, old, new, sig, tan, guess, 1.0)
            steps.append((failed, *disc.host(sig.copy(), tan.copy(), new.copy())))
            if k in (1, 2):
                old, new = new, old  # an accepted step
        out[device] = steps
    plastic = np.abs(out["cpu"][-1][3][:, 6]) > 0  # p of the last step
    assert plastic.mean() > 0.5
    back = np.abs(out["cpu"][-1][3][:, 10:]).max(axis=1) > 0  # the backstresses
    assert back.sum() > 20
    for cpu, gpu_step in zip(out["cpu"], out["gpu"], strict=True):
        assert cpu[0] == gpu_step[0] == 0
        for a, b, tol in zip(cpu[1:], gpu_step[1:], (1e-12, 1e-10, 1e-12), strict=True):
            assert np.allclose(b, a, rtol=0, atol=tol * np.abs(a).max())


@needs_gpu
def test_plastic_paths_and_fields_on_the_gpu():
    """A composite with a viscoplastic matrix along a mixed path (tension, lateral
    stresses free, a change of temperature), small strains with B-bar and finite strains
    with F-bar: the same curves, phase averages and fields on the GPU."""
    rng = np.random.default_rng(2)
    data = (rng.random((7, 6, 8)) < 0.3).astype(np.uint8)
    image = VoxelImage(data, spacing=(0.5, 0.5, 0.5))
    mats = _behaviour_materials()
    mats = {0: mats[1], 1: mats[0]}
    path = {"times": [0, 10, 20], "increments": [4, 4], "strain": {"11": [0, 0.01, 0.004]},
            "stress": {"22": 0, "33": 0}, "temperature": [0, 30, 30]}  # fmt: skip
    for kinematics in ("small", "finite"):
        res = {}
        for device in ("cpu", "gpu"):
            problem = NonlinearFFTProblem(image, mats, kinematics=kinematics, device=device)
            res[device] = problem.run(path, fields=True)
        cpu, gpu_res = res["cpu"], res["gpu"]
        scale = np.abs(cpu.stress).max()
        assert np.allclose(gpu_res.stress, cpu.stress, rtol=0, atol=1e-6 * scale)
        assert np.allclose(gpu_res.strain, cpu.strain, rtol=0, atol=1e-6 * np.abs(cpu.strain).max())
        assert np.allclose(gpu_res.plastic, cpu.plastic, rtol=1e-5, atol=1e-12)
        for phase, values in cpu.phases.items():
            assert np.allclose(gpu_res.phases[phase]["stress"], values["stress"], rtol=0,
                               atol=1e-6 * scale)  # fmt: skip
        assert gpu_res.n_steps == cpu.n_steps and cpu.plastic[-1] > 0
        for t, fields in cpu.fields.items():
            for name, values in fields.items():
                ok = np.isfinite(values)
                assert np.allclose(gpu_res.fields[t][name][ok], values[ok], rtol=1e-5,
                                   atol=1e-6 * np.abs(values[ok]).max())  # fmt: skip


@needs_gpu
def test_paths_on_the_gpu():
    """A voxel polycrystal with crystal plasticity along a tension, in small and finite
    strains: the same curves on the GPU."""
    rng = np.random.default_rng(1)
    n, n_cells = 8, 4
    centres = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) / n
    seeds = rng.random((n_cells, 3))
    d = centres[:, None, :] - seeds[None, :, :]
    d -= np.round(d)
    grains = np.argmin((d**2).sum(-1), axis=1).reshape(n, n, n)
    image = VoxelImage(np.zeros((n, n, n), dtype=np.uint8), spacing=(1 / n,) * 3)
    import dataclasses

    m = dataclasses.replace(Material.cubic(168e3, 121e3, 75e3),
                            crystal_plasticity=CrystalPlasticity.from_config(
                                {"tau0": 60.0, "K": 10.0, "n": 15.0}))  # fmt: skip
    path = {"times": [0, 10], "increments": 5, "strain": {"33": 0.01}}
    orientations = random_orientations(rng, n_cells)
    for kinematics in ("small", "finite"):
        curves = {}
        for device in ("cpu", "gpu"):
            problem = NonlinearFFTProblem(image, {0: m}, grains=grains, orientations=orientations,
                                          kinematics=kinematics, device=device)  # fmt: skip
            curves[device] = problem.run(path)
        cpu, gpu_res = curves["cpu"], curves["gpu"]
        scale = np.abs(cpu.stress).max()
        assert np.allclose(gpu_res.stress, cpu.stress, rtol=0, atol=1e-6 * scale)
        assert gpu_res.n_steps == cpu.n_steps
