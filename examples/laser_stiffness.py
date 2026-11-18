"""Architectured sheets by localized laser treatment: what the designs of §14 do to the
stiffness of a sheet (docs/laser.md, §14).

The phases of a laser-treated steel share Young's modulus within a few per cent: the
designs change the plastic response of the sheet, not its stiffness. Pairs of states
whose moduli differ make elastic architectures: an aged metastable beta titanium whose
tracks the laser solution-treats (109 and 82 GPa), a superelastic NiTi whose tracks it
turns martensitic (68 and about 30 GPa). For each design
of ``examples/laser_morphologies.py`` (its cells of 0.5 mm) and each pair, the
effective in-plane stiffness of the sheet (morphoGen's FFT homogenization of the
periodic cell, a plane-stress stiffness condensed from it): the mean Young's modulus
over the directions, its spread (max − min over the mean), the range of Poisson's
ratio, against the bounds of Voigt and Reuss.

Writes a CSV file; reads the designs written by ``laser_morphologies.py``. About two
minutes on 16 processes.
"""

import argparse
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CELL = 0.5  # mm
#: the pairs of states (MPa, Poisson's ratio): untreated, treated; see docs/laser.md §14
PAIRS = {
    # martensite about 3 % below the ferrite-pearlite
    "steel: ferrite-pearlite / laser martensite": ((210000.0, 0.29), (204000.0, 0.29)),
    # an aged metastable beta titanium (Ti-15-3) whose tracks the laser solution-treats
    "beta titanium: aged / laser solution-treated": ((109000.0, 0.32), (82000.0, 0.33)),
    # a superelastic NiTi (austenite) whose tracks the laser turns martensitic
    "NiTi: austenite / laser-processed martensite": ((68000.0, 0.33), (30000.0, 0.33)),
}
ANGLES = np.linspace(0.0, math.pi, 37)


def plane_stress(design, pair) -> dict:
    """The effective plane-stress stiffness of a periodic design (cells of ``CELL``):
    ``E(θ)`` over the directions, its mean and spread, Poisson's ratio ``ν(θ)``."""
    from morphogen.homogenization import Material, homogenize
    from morphogen.image import VoxelImage

    (e0, n0), (e1, n1) = pair
    img = VoxelImage(
        data=np.asarray(design).T[:, :, None].astype(np.uint8),
        spacing=(CELL, CELL, CELL),
        origin=(0.0, 0.0, 0.0),
        periodic=True,
    )
    mats = {0: Material(young=e0, poisson=n0), 1: Material(young=e1, poisson=n1)}
    r = homogenize(img, mats, properties=["stiffness"])["stiffness.periodic"]
    # plane stress: condense the out-of-plane stress to zero from the full stiffness
    C = np.asarray(r["tensor"])
    keep, out = [0, 1, 5], [2]
    Cps = C[np.ix_(keep, keep)] - C[np.ix_(keep, out)] @ np.linalg.solve(
        C[np.ix_(out, out)], C[np.ix_(out, keep)]
    )
    S = np.linalg.inv(Cps)
    c, s = np.cos(ANGLES), np.sin(ANGLES)
    inv_e = (
        S[0, 0] * c**4
        + S[1, 1] * s**4
        + (2 * S[0, 1] + S[2, 2]) * c**2 * s**2
        + 2 * S[0, 2] * c**3 * s
        + 2 * S[1, 2] * c * s**3
    )
    E = 1.0 / inv_e
    # Poisson's ratio for a uniaxial stress along n: −ε_tt / ε_nn
    nu = []
    for ci, si in zip(c, s, strict=True):
        sig = np.array([ci * ci, si * si, ci * si])
        eps = S @ sig
        ett = eps[0] * si * si + eps[1] * ci * ci - eps[2] * ci * si
        enn = eps[0] * ci * ci + eps[1] * si * si + eps[2] * ci * si
        nu.append(-ett / enn)
    f = float(np.mean(np.asarray(design) > 0))
    return {
        "E_mean": float(E.mean()),
        "E_min": float(E.min()),
        "E_max": float(E.max()),
        "E_spread": float((E.max() - E.min()) / E.mean()),
        "nu_min": float(min(nu)),
        "nu_max": float(max(nu)),
        "voigt": (1 - f) * e0 + f * e1,
        "reuss": 1.0 / ((1 - f) / e0 + f / e1),
        "treated": f,
    }


def _init() -> None:
    import numba

    from morphogen._platform import full_speed

    full_speed()
    numba.set_num_threads(1)  # one thread per process: the processes share the cores


def evaluate(job) -> dict:
    name, design, pair_name = job
    out = plane_stress(design, PAIRS[pair_name])
    out.update({"design": name, "pair": pair_name})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--designs", type=Path, default=Path("results/laser_morphologies.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/laser_stiffness.csv"))
    args = parser.parse_args()
    designs = dict(np.load(args.designs))
    todo = [(n, d, p) for p in PAIRS for n, d in designs.items()]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        rows = list(pool.map(evaluate, todo))
    print(f"{len(rows)} cells in {time.perf_counter() - t0:.0f} s -> {args.output}\n")
    fields = ["pair", "design", "treated", "E_mean", "E_min", "E_max", "E_spread"]
    fields += ["nu_min", "nu_max", "voigt", "reuss"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(rows)
    for p in PAIRS:
        print(f"{p}")
        print(f"  {'design':28s} treated  E mean (GPa)  [Reuss, Voigt]   E spread  nu range")
        for r in (r for r in rows if r["pair"] == p):
            print(
                f"  {r['design']:28s} {r['treated']:5.2f}   {r['E_mean'] / 1000:6.1f}"
                f"       [{r['reuss'] / 1000:5.1f}, {r['voigt'] / 1000:5.1f}]   "
                f"{r['E_spread']:5.2f}    {r['nu_min']:.2f} to {r['nu_max']:.2f}"
            )


if __name__ == "__main__":
    main()
