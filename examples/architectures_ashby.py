"""Normalized Ashby charts of architected and bio-inspired cellular materials.

Every run is recorded in a database of results (SQLite); the charts place the relative
Young's modulus E1 / Es of each microstructure against its relative density, with the
scaling laws of cellular solids (Gibson & Ashby): stretching-dominated lattices (octet)
follow E / Es ~ phi, bending-dominated foams (Kelvin cells, trabecular bone) phi^2, and
the walls of honeycombs bend in their plane (phi^3). One chart for the lattices, one for
the bio-inspired and fractal structures, three families each.

Usage: python examples/architectures_ashby.py [output directory]   (a few minutes)
"""

import math
import sys
from pathlib import Path

from morphogen import pipeline
from morphogen.config import load_config
from morphogen.results import ResultsDatabase, ashby_chart

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/architectures")
database = out / "architectures.db"
densities = (0.08, 0.15, 0.3)


def family(name, model, shape=48, size=1.0, density=None):
    """One run of a family: the periodic cell homogenized by FFT, recorded in the
    database under the name of the family."""
    cfg = {
        "model": model,
        "domain": {"size": size, "periodic": True},
        "seed": 1,
        "voxels": {"shape": shape},
        "homogenization": {
            "method": "fft",
            "materials": {0: "void", 1: {"young": 1.0, "poisson": 0.3, "density": 1.0}},
            "properties": ["stiffness"],
            "tolerance": 1e-6,
        },
        "outputs": {
            "directory": str(out / f"{name}_{density}"),
            "name": name,
            "database": str(database),
        },
    }
    pipeline.run(load_config(cfg))


for phi in densities:
    family("octet", {"type": "strut_lattice", "cell": "octet", "volume_fraction": phi}, density=phi)
    family(
        "kelvin", {"type": "strut_lattice", "cell": "kelvin", "volume_fraction": phi}, density=phi
    )
    family(
        "gyroid",
        {"type": "tpms", "surface": "gyroid", "variant": "sheet", "volume_fraction": phi},
        density=phi,
    )
    family(
        "trabecular",
        {"type": "open_foam", "n_cells": 16, "volume_fraction": phi, "relaxation": 10,
         "anisotropy": 1.4, "axis": 0},
        density=phi,
    )  # fmt: skip
    t = 1.0 - math.sqrt(1.0 - phi)  # walls of hexagons 1 wide: phi = 1 - (1 - t)^2
    family(
        "honeycomb",
        {"type": "honeycomb", "cell_size": 1.0, "wall_thickness": t, "axis": 2},
        shape=[96, 166, 2],
        size=[2.0, 2.0 * math.sqrt(3.0), 0.04],
        density=phi,
    )
for level in (1, 2, 3):
    family("menger", {"type": "fractal_sponge", "level": level}, shape=81, density=level)

charts = {
    "lattices": (["octet", "kelvin", "gyroid"], (1, 2)),
    "bio-inspired": (["trabecular", "honeycomb", "menger"], (1, 2, 3)),
}
with ResultsDatabase(database) as db:
    print(f"{len(db.records())} records in {database}")
    for chart, (names, guides) in charts.items():
        records = db.records([f"run_name in {','.join(names)}"])
        for theme in ("light", "dark"):
            path = ashby_chart(
                records,
                out / f"ashby_{chart.replace('-', '_')}_{theme}.svg",
                y="stiffness.periodic.E1.relative",
                by="run_name",
                groups=names,
                guides=guides,
                theme=theme,
                title=f"Cellular materials: {chart}",
            )
            print("chart:", path)
