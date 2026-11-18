"""Speed of the FFT solvers on the CPU and on the GPU (morphogen.gpu).

* Linear: the apparent bulk modulus of elastic spheres (30 %, contrast 10) on n^3 voxels.
* Nonlinear: a Voronoi polycrystal with crystal plasticity pulled to 1 % (10 steps) on
  n^3 voxels, small strains.
* Porous: stiff spheres (20 %) in a GTN matrix (1 % of voids) pulled to 1 % (10 steps)
  on n^3 voxels.

Usage: python examples/gpu_benchmark.py [linear sizes...] [--paths sizes...]
[--porous sizes...] (default: 64 96 128 160 192 --paths 24 32 --porous 24 32; ``--paths``
or ``--porous`` alone runs those paths only). The GPU needs CuPy (see morphogen.gpu); the
kernels are compiled before the timings.
"""

import sys
import time

import numpy as np

import morphogen as mg
from morphogen import gpu
from morphogen.homogenization import (
    Material,
    NonlinearFFTProblem,
    apparent_bulk_modulus,
    materials_from_config,
)
from morphogen.homogenization.fft import FFTElasticProblem
from morphogen.models.base import model_from_config


def linear(n: int, device: str) -> tuple[float, int, float]:
    micro = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3).generate(
        mg.Domain.cube(10.0), seed=1
    )
    image = micro.voxelize(shape=n)
    mats = {0: Material(young=70e3, poisson=0.33), 1: Material(young=700e3, poisson=0.2)}
    t0 = time.perf_counter()
    result = apparent_bulk_modulus(FFTElasticProblem(image, mats, device=device))
    return result.value, result.solution.info.iterations, time.perf_counter() - t0


def nonlinear(n: int, device: str) -> tuple[float, int, float]:
    micro = model_from_config({"type": "voronoi", "n_cells": 12, "crystal_orientations": True})
    micro = micro.generate(mg.Domain.cube(1.0), seed=11)
    image = micro.voxelize(shape=n)
    labels = micro.labels(grid=image.grid())
    mats = materials_from_config({1: {"cubic": [250000, 160000, 120000], "crystal_plasticity": {
        "families": ["octahedral"], "tau0": 80, "Q": 20, "b": 10, "C": 20000, "D": 300, "K": 25,
        "n": 10, "interaction": [1.0, 1.0, 0.6, 12.3, 1.6, 1.3]}}})  # fmt: skip
    path = {"times": [0, 10], "increments": 10, "strain": {"11": 0.01}}
    t0 = time.perf_counter()
    problem = NonlinearFFTProblem(image, mats, grains=labels, orientations=micro.orientations,
                                  device=device)  # fmt: skip
    res = problem.run(path)
    return res.stress[-1, 0], res.info["linear_iterations"], time.perf_counter() - t0


def porous(n: int, device: str) -> tuple[float, float, int, float]:
    micro = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.2).generate(
        mg.Domain.cube(10.0), seed=2
    )
    image = micro.voxelize(shape=n)
    gtn = {"criterion": "gurson", "yield": 300.0, "hardening": 5000.0, "porosity": 0.01}
    mats = {0: Material(young=200e3, poisson=0.3, porous_plasticity=gtn),
            1: Material(young=400e3, poisson=0.2)}  # fmt: skip
    path = {"times": [0, 10], "increments": 10, "strain": {"11": 0.01}}
    t0 = time.perf_counter()
    res = NonlinearFFTProblem(image, mats, device=device).run(path)
    porosity = res.phases[0]["porosity"][-1]
    return res.stress[-1, 0], porosity, res.info["linear_iterations"], time.perf_counter() - t0


def main(argv) -> None:
    sizes, paths, pores = [], [], []
    target = sizes
    for arg in argv:
        if arg == "--paths":
            target = paths
        elif arg == "--porous":
            target = pores
        else:
            target.append(int(arg))
    if not sizes and "--paths" not in argv and "--porous" not in argv:
        sizes, paths, pores = [64, 96, 128, 160, 192], [24, 32], [24, 32]
    print(gpu.info())
    devices = ["cpu", "gpu"] if gpu.available() else ["cpu"]
    for device in devices:  # compile the kernels
        if sizes:
            linear(16, device)
        if paths:
            nonlinear(8, device)
        if pores:
            porous(8, device)
    print(f"{'linear':>10s} {'device':>6s} {'k (MPa)':>12s} {'iter':>5s} {'time_s':>8s}")
    for n in sizes:
        for device in devices:
            k, it, t = linear(n, device)
            print(f"{n:>9d}^3 {device:>6s} {k:12.6g} {it:5d} {t:8.2f}")
    print(f"{'paths':>10s} {'device':>6s} {'S11 (MPa)':>12s} {'CG':>5s} {'time_s':>8s}")
    for n in paths:
        for device in devices:
            s, it, t = nonlinear(n, device)
            print(f"{n:>9d}^3 {device:>6s} {s:12.6g} {it:5d} {t:8.2f}")
    if pores:
        print(f"{'porous':>10s} {'device':>6s} {'S11 (MPa)':>12s} {'f':>7s} {'CG':>5s} "
              f"{'time_s':>8s}")  # fmt: skip
    for n in pores:
        for device in devices:
            s, f, it, t = porous(n, device)
            print(f"{n:>9d}^3 {device:>6s} {s:12.6g} {f:7.4f} {it:5d} {t:8.2f}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main(sys.argv[1:])
