"""Architectured sheets by localized laser treatment: the morphological models of
morphoGen as laser treatments of a sheet (docs/laser.md, §14).

The spot of the laser (1 mm) sets the smallest feature, 1 mm × 1 mm, not a grid: the
designs are drawn on cells of 0.5 mm and opened by a square spot of two cells
(``spot_opening``), so that every treated cell lies in a treated square of 1 mm². Each
2D model of morphoGen (the automata, the growth of grains, the tiles, the networks, the
honeycombs, nacre, the random fields, the Boolean discs...) and the laser patterns of the
module draw a periodic cell of 24 × 24 mm, with about a third of the sheet treated, the
treated cells hard (the laser martensite of the annealed 0.23 C steel, or a tempered
one, ``contrast_phases(3)``). For each design: its treatment (``treatment_measures``: the
connectivity of the phases, the share of the untreated cells next to the treatment, the
time to treat a sheet of 300 × 300 mm) and the sheet it makes (``sheet_tests``: the
periodic cell pulled at 0°, 45° and 90°: strengths, failure strain, energy, Lankford's
coefficient and their spreads over the directions).

Writes the results (CSV) and, with ``--figure``, the mosaic of the designs and the map
of their properties. The tests run in parallel: about ten minutes on 16 processes.
"""

import argparse
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SIZE = 24.0  # mm: the side of the periodic cell
CELL = 0.5  # mm: the cells of the designs (and the elements)
N = round(SIZE / CELL)
SPOT = 2  # cells: the spot of 1 mm
SHEET = 300.0  # mm: the side of the sheet whose treatment is timed


def _micro(config: dict, seed: int, thickness: float):
    from morphogen.domain import Domain
    from morphogen.models import model_from_config

    domain = Domain(size=(SIZE, SIZE, thickness), periodic=True)
    return model_from_config(config).generate(domain, seed=seed)


def model(config: dict, seed: int = 1, thickness: float = 1.0) -> np.ndarray:
    """The design ``(ny, nx)`` of a morphoGen model on the periodic cell (phase > 0
    treated)."""
    img = _micro(config, seed, thickness).voxelize(shape=(N, N, 1)).data[:, :, 0]  # [x, y]
    return (np.asarray(img).T > 0).astype(np.int64)


def labels(config: dict, seed: int = 1) -> np.ndarray:
    return np.asarray(_micro(config, seed, 1.0).labels(shape=(N, N, 1)).data[:, :, 0]).T


def tracks(lines, width: int = SPOT) -> np.ndarray:
    """Polylines (mm) drawn by a spot of ``width`` cells, repeated over the cell."""
    from morphogen.laser.cells import CellGrid

    grid = CellGrid(3 * SIZE, 3 * SIZE, CELL, x0=-SIZE, y0=-SIZE)
    big = grid.stamp(lines, width=width)
    n = N
    return np.clip(
        sum(big[i * n : (i + 1) * n, j * n : (j + 1) * n] for i in range(3) for j in range(3)), 0, 1
    )


def straight(spacing: float = 3.0) -> np.ndarray:
    x = np.array([-SIZE, 2 * SIZE])
    ys = np.arange(-SIZE, 2 * SIZE, spacing) + 0.5 * spacing
    return tracks([np.column_stack([x, [y, y]]) for y in ys])


def crossed(spacing: float = 6.0) -> np.ndarray:
    a = np.array([-SIZE, 2 * SIZE])
    s = np.arange(-SIZE, 2 * SIZE, spacing) + 0.5 * spacing
    lines = [np.column_stack([a, [v, v]]) for v in s] + [np.column_stack([[v, v], a]) for v in s]
    return tracks(lines)


def corrugated(period: float = 12.0, height: float = 1.5, spacing: float = 3.0) -> np.ndarray:
    """Sine tracks along x (the corrugations of Wang et al.), ``spacing`` mm apart."""
    x = np.linspace(-SIZE, 2 * SIZE, 1200)
    ys = np.arange(-SIZE, 2 * SIZE, spacing) + 0.5 * spacing
    return tracks([np.column_stack([x, y + height * np.sin(2 * math.pi * x / period)]) for y in ys])


def polycrystal(grain: float = 8.0, spacing: float = 3.0, seed: int = 2) -> np.ndarray:
    """A meta-polycrystal: the grains of a 2D Voronoi tessellation (nucleation and
    growth by site saturation), each filled with tracks 1 mm wide of its own random
    direction, ``spacing`` mm apart."""
    lab = labels(
        {
            "type": "nucleation_growth",
            "nucleation": "site_saturation",
            "nucleus_density": 1.0 / grain**2,
            "dimension": 2,
            "cell_size": CELL,
        },
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, math.pi, int(lab.max()) + 2)
    y, x = (np.mgrid[0:N, 0:N] + 0.5) * CELL
    th = angles[np.maximum(lab, 0)]
    s = -x * np.sin(th) + y * np.cos(th)
    return ((np.mod(s, spacing) < 1.0) & (lab >= 0)).astype(np.int64)


def catalogue() -> dict:
    """The designs, opened by the spot: ``name -> (family, design)``."""
    c = {
        "straight tracks": ("laser pattern", straight(3.0)),
        "corrugated tracks": ("laser pattern", corrugated(spacing=4.0)),
        "crossed tracks": ("laser pattern", crossed(6.0)),
        "hexagonal honeycomb": (
            "honeycomb",
            model({"type": "honeycomb", "cell_size": 6.0, "wall_thickness": 1.0}),
        ),
        "Voronoi walls": (
            "honeycomb",
            model(
                {
                    "type": "honeycomb",
                    "arrangement": "voronoi",
                    "cell_size": 6.0,
                    "wall_thickness": 1.0,
                    "regularity": 0.5,
                }
            ),
        ),
    }
    c["Voronoi cells, soft walls"] = (
        "honeycomb (complement)",
        1
        - model(
            {
                "type": "honeycomb",
                "arrangement": "voronoi",
                "cell_size": 8.0,
                "wall_thickness": 1.0,
                "regularity": 0.5,
            }
        ),
    )
    c["nacre (bricks)"] = (
        "nacre",
        model(
            {
                "type": "nacre",
                "arrangement": "brick",
                "tablet_size": 4.0,
                "tablet_thickness": 2.0,
                "mortar_thickness": 1.0,
                "gap": 1.0,
                "aspect": 2.0,
                "stagger": "u",
                "axis": 1,
            },
            thickness=4.0,
        ),
    )
    c["Gaussian excursion"] = (
        "random field",
        model(
            {"type": "gaussian_excursion", "correlation_length": 1.0, "volume_fraction": 0.33},
            thickness=0.25,
        ),
    )
    c["Boolean discs"] = (
        "germ-grain",
        model({"type": "boolean_spheres", "radius": 2.0, "volume_fraction": 0.33}, thickness=5.0),
    )
    c["Turing pattern"] = (
        "automaton",
        model(
            {
                "type": "activator_inhibitor",
                "activation_radius": 1.0,
                "inhibition_radius": 2.5,
                "volume_fraction": 0.35,
                "cell_size": CELL,
                "dimension": 2,
            }
        ),
    )
    c["coarsened automaton"] = (
        "automaton",
        model(
            {
                "type": "cellular_automaton",
                "rule": "conserved",
                "steps": 20,
                "volume_fraction": 0.35,
                "cell_size": CELL,
                "dimension": 2,
            }
        ),
    )
    c["Wang tiles (pipes)"] = (
        "tile assembly",
        model(
            {
                "type": "tile_assembly",
                "tiles": "pipes",
                "tile_length": 6.0,
                "tile_cells": 12,
                "dimension": 2,
                "pipe_radius": 0.12,
                "open_probability": 0.6,
            }
        ),
    )
    c["Delaunay fibre network"] = (
        "foam",
        model({"type": "delaunay_network", "intensity": 0.015, "radius": 0.5, "dimension": 2}),
    )
    c["random laminate"] = (
        "laminate",
        model(
            {
                "type": "random_laminate",
                "thickness": 1.0,
                "spacing": 2.0,
                "axis": 1,
                "hole_fraction": 0.2,
                "hole_radius": 1.5,
            }
        ),
    )
    c["Johnson-Mehl grains"] = (
        "growth",
        model(
            {
                "type": "nucleation_growth",
                "nucleation": "continuous",
                "nucleation_rate": 0.03,
                "growth_rate": 1.0,
                "time": 2.3,
                "dimension": 2,
                "cell_size": CELL,
            }
        ),
    )
    c["meta-polycrystal"] = ("growth + tracks", polycrystal())
    from morphogen.laser.architectures import spot_opening

    return {k: (fam, spot_opening(d, SPOT)) for k, (fam, d) in c.items()}


#: the hard phases: the as-quenched laser martensite and a tempered one (graded rule)
HARD = {"martensite": 1300.0 / 280.0, "tempered": 3.0}
ANGLES = (0.0, 45.0, 90.0)
FIELDS = (
    "design",
    "family",
    "hard",
    "treated",
    "hard_spans",
    "soft_spans",
    "heat_affected",
    "sheet_time_h",
    *(f"{k}_{a:.0f}" for a in ANGLES for k in ("rp", "rm", "failure", "energy", "r", "broke")),
    "rp_mean",
    "rp_spread",
    "rm_mean",
    "failure_mean",
    "failure_min",
    "energy_mean",
    "energy_min",
    "r_spread",
)


def _init() -> None:
    import numba

    from morphogen._platform import full_speed

    full_speed()
    numba.set_num_threads(1)  # one thread per process: the processes share the cores


def evaluate(job) -> dict:
    """The treatment and the sheet of one design: ``(name, family, design, hard)``."""
    from morphogen.laser.architectures import sheet_tests, treatment_measures
    from morphogen.laser.cells import contrast_phases

    name, family, design, hard = job
    p = treatment_measures(design, pixel=CELL, spot=2 * CELL)
    s = sheet_tests(design, pixel=CELL, phases=contrast_phases(HARD[hard]), angles=ANGLES)
    row = {
        "design": name,
        "family": family,
        "hard": hard,
        "treated": p["treated"],
        "hard_spans": "".join("xy"[i] for i in range(2) if p["hard_spans"][i]) or "-",
        "soft_spans": "".join("xy"[i] for i in range(2) if p["soft_spans"][i]) or "-",
        "heat_affected": p["heat_affected"],
        "sheet_time_h": p["treated"] * SHEET**2 / (2 * CELL) / 10.0 / 3600.0,
    }
    failures, energies, rs = [], [], []
    for a in ANGLES:
        m = s["directions"][a]
        row.update(
            {
                f"rp_{a:.0f}": m["yield_strength"],
                f"rm_{a:.0f}": m["tensile_strength"],
                f"failure_{a:.0f}": m["failure_strain"],
                f"energy_{a:.0f}": m["energy"],
                f"r_{a:.0f}": m["r_value"],
                f"broke_{a:.0f}": m["broke"],
            }
        )
        failures.append(m["failure_strain"])
        energies.append(m["energy"])
        rs.append(m["r_value"])
    row.update(
        {
            "rp_mean": s["yield_strength_mean"],
            "rp_spread": s["yield_strength_spread"],
            "rm_mean": s["tensile_strength_mean"],
            "failure_mean": float(np.mean(failures)),
            "failure_min": float(np.min(failures)),
            "energy_mean": float(np.mean(energies)),
            "energy_min": float(np.min(energies)),
            "r_spread": float((max(rs) - min(rs)) / np.mean(rs)) if np.mean(rs) else math.nan,
        }
    )
    return row


def jobs(cat: dict) -> list:
    out = [("untreated", "reference", np.zeros((N, N), np.int64), "martensite")]
    for hard in HARD:
        out.append(("fully treated", "reference", np.ones((N, N), np.int64), hard))
        out += [(name, fam, d, hard) for name, (fam, d) in cat.items()]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--output", type=Path, default=Path("results/laser_morphologies.csv"))
    parser.add_argument("--figure", type=Path, help="draw the figure to this file")
    args = parser.parse_args()
    cat = catalogue()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **{k: d for k, (f, d) in cat.items()})
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        rows = list(pool.map(evaluate, jobs(cat)))
    print(f"{len(rows)} sheets in {time.perf_counter() - t0:.0f} s -> {args.output}\n")
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(
        f"  {'design':28s} {'hard':10s} treated spans(h/s) HAZ  time(h)  R_p 0/45/90     "
        f"R_m   failure 0/45/90      energy  r spread"
    )
    for r in rows:
        print(
            f"  {r['design']:28s} {r['hard']:10s} {r['treated']:5.2f}  {r['hard_spans']:2s}/"
            f"{r['soft_spans']:2s}  {r['heat_affected']:4.2f} {r['sheet_time_h']:5.1f}  "
            + "/".join(f"{r[f'rp_{a:.0f}']:4.0f}" for a in ANGLES)
            + f"  {r['rm_mean']:5.0f}  "
            + "/".join(f"{100 * r[f'failure_{a:.0f}']:4.1f}" for a in ANGLES)
            + f"  {r['energy_mean']:5.0f}  {r['r_spread']:5.2f}"
        )


if __name__ == "__main__":
    main()
