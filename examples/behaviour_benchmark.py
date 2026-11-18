"""Cost of the behaviours of the integration points on the CPU threads (the Numba kernels
of morphogen.homogenization.behaviours and .finite): an increment in the middle of a
loading, the tenth of ten towards random strains of a few 10⁻³ (each point its own), for
elasticity, von Mises plasticity (Voce and Armstrong-Frederick hardenings), Hill's
criterion, Gurson's porous plasticity (with a nucleation) and an fcc crystal (12 slip
systems), in small and finite strains.

Usage: python examples/behaviour_benchmark.py [points] (default 200 000). The kernels are
compiled before the timings; the best of three runs is shown.
"""

import dataclasses
import sys
import time

import numba
import numpy as np

from morphogen.homogenization import Material
from morphogen.homogenization.behaviours import Plasticity, update_points
from morphogen.homogenization.crystal import CrystalPlasticity
from morphogen.homogenization.finite import update_points_finite
from morphogen.homogenization.materials import material_table
from morphogen.homogenization.nonlinear import behaviour_table
from morphogen.orientations import random_orientations

HARDENING = {"yield_stress": 100.0, "hardening": 500.0, "isotropic": ((50.0, 20.0),),
             "kinematic": ((20000.0, 200.0),)}  # fmt: skip
GURSON = {"criterion": "gurson", "yield": 100.0, "hardening": 500.0, "porosity": 0.02,
          "q1": 1.5, "q2": 1.0, "nucleation": {"type": "gaussian", "fN": 0.04, "eN": 0.05,
                                               "sN": 0.02}}  # fmt: skip
CRYSTAL = {"tau0": 60, "Q": 20, "b": 10, "C": 20000, "D": 300, "K": 10, "n": 15,
           "interaction": [1.0, 1.0, 0.6, 12.3, 1.6, 1.3]}  # fmt: skip
HILL = Plasticity(**HARDENING, hill=(1.3, 0.8, 1.1, 0.7, 1.25, 0.9))
FCC = dataclasses.replace(Material.cubic(168e3, 121e3, 75e3),
                          crystal_plasticity=CrystalPlasticity.from_config(CRYSTAL))  # fmt: skip
MATERIALS = {
    "elastic": Material(young=70e3, poisson=0.3),
    "von Mises": Material(young=70e3, poisson=0.3, plasticity=Plasticity(**HARDENING)),
    "Hill": Material(young=70e3, poisson=0.3, plasticity=HILL),
    "Gurson": Material(young=70e3, poisson=0.3, porous_plasticity=GURSON),
    "crystal (fcc)": FCC,
}


def increment(material: Material, n: int, finite: bool) -> tuple[float, int]:
    """Seconds per point of the last of ten increments of ``n`` points, failed points."""
    rng = np.random.default_rng(0)
    phase = np.zeros(n, dtype=np.int64)
    grains = np.arange(n)
    table = material_table(phase, {0: material}, "elastic", grains=grains,
                           orientations=random_orientations(rng, n))  # fmt: skip
    beh = behaviour_table(phase, {0: material}, table, finite=finite)
    nc = 9 if finite else 6
    final = rng.normal(0.0, 4e-3, (n, nc))
    if not finite:
        final[:, :3] += rng.normal(0.0, 3e-3, (n, 1))  # some volume change
    old = np.zeros((n, beh.width))
    new = np.zeros_like(old)
    stress = np.zeros((n, nc))
    tangent = np.zeros((n, nc, nc))
    kernel = update_points_finite if finite else update_points

    def run(strain):
        return kernel(beh.kind, table.mat, beh.pmat, beh.data, strain, 0.0, 1.0, old, new,
                      stress, tangent, 0, 1.0)  # fmt: skip

    failed = 0
    for k in range(1, 10):
        failed += run(final * k / 10)
        old[:] = new
    best = np.inf
    for _ in range(3):
        t0 = time.perf_counter()
        last = run(final)
        best = min(best, time.perf_counter() - t0)
    return best / n, failed + last


def main(argv) -> None:
    n = int(argv[0]) if argv else 200_000
    print(f"{n} points, {numba.get_num_threads()} threads")
    for material in MATERIALS.values():  # compile the kernels
        for finite in (False, True):
            increment(material, 10, finite)
    print(f"{'behaviour':>14s} {'small (ns)':>11s} {'finite (ns)':>12s} {'failed':>7s}")
    for name, material in MATERIALS.items():
        small, f_small = increment(material, n, False)
        large, f_large = increment(material, n, True)
        print(f"{name:>14s} {1e9 * small:11.0f} {1e9 * large:12.0f} {f_small + f_large:7d}")


if __name__ == "__main__":
    main(sys.argv[1:])
