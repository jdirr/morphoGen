"""Polycrystals: conforming meshes grain by grain, anisotropic crystals, exports for other
codes, and a parallel batch of realizations.

Usage: python examples/polycrystal.py [output directory]   (needs MMG: pip install pymmg)
"""

import sys
import time
from pathlib import Path

import morphogen as mg
from morphogen import pipeline
from morphogen.config import load_config
from morphogen.homogenization import Material, bounds, homogenize

COPPER = Material.cubic(168.4, 121.4, 75.4, conductivity=400.0)  # GPa, W/m/K


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # 1. A copper polycrystal of 50 grains, meshed grain by grain
    grains = mg.models.Voronoi(n_cells=50, crystal_orientations=True).generate(
        mg.Domain.cube(1.0, periodic=True), seed=1
    )
    t0 = time.perf_counter()
    mesh = mg.mesh.conforming_mesh(grains, size=0.08, min_feature=0.02)
    q = mesh.quality()
    print(
        f"mesh: {mesh.n_elements} elements, {mesh.summary()['n_grains']} grains, quality "
        f"min {q.min():.3f} mean {q.mean():.3f}, matching faces {mesh.matching_faces()} "
        f"({time.perf_counter() - t0:.1f} s)"
    )

    # 2. Apparent moduli: the bulk modulus of a cubic polycrystal is (C11 + 2 C12) / 3
    results = homogenize(
        mesh,
        {1: COPPER},
        bcs=["kubc", "periodic", "subc"],
        properties=["bulk_modulus", "shear_modulus"],
    )
    for key, res in results.items():
        print(f"  {key:24s} {res.value:9.3f} GPa")
    hs = bounds({1: 1.0}, {1: COPPER})["shear_modulus"]
    print(f"  shear modulus bounds: Hashin-Shtrikman {hs['hs_lower']:.2f}-{hs['hs_upper']:.2f}")

    # 3. Exports: Abaqus/CalculiX with orientations, periodic equations, FEniCSx, DAMASK
    mg.io.save_mesh(mesh, out / "grains.inp")
    mg.io.write_periodic_equations(out / "grains_equations.inp", mesh)
    mg.io.save_mesh(mesh, out / "grains.xdmf")
    mg.io.write_damask(
        out / "grains_damask", grains.labels(shape=64), grains.orientations, materials={1: COPPER}
    )
    mg.io.write_neper(out / "grains", grains)

    # 4. A batch of realizations in four processes, inclusions and grain boundaries
    config = {
        "model": {
            "type": "combination",
            "operation": "inclusions",
            "models": [
                {
                    "type": "voronoi",
                    "n_cells": 50,
                    "crystal_orientations": True,
                    "boundary_thickness": 0.01,
                },
                {"type": "hardcore_spheres", "radius": 0.05, "volume_fraction": 0.05},
            ],
        },
        "domain": {"size": 1.0, "periodic": True},
        "seed": 11,
        "realizations": 8,
        "voxels": {"shape": 48},
        "homogenization": {
            "materials": {
                0: {"young": 60.0, "poisson": 0.35},  # grain boundaries
                1: COPPER.to_config(),
                2: {"young": 380.0, "poisson": 0.22},  # alumina
            },
            "properties": ["bulk_modulus", "shear_modulus"],
        },
        "outputs": {"directory": str(out / "batch"), "name": "composite"},
    }
    t0 = time.perf_counter()
    outdir = pipeline.run(load_config(config), jobs=4)
    rows = pipeline.read_rows(outdir / "results.csv")
    mean = sum(r["bulk_modulus.periodic"] for r in rows) / len(rows)
    print(
        f"batch: {len(rows)} realizations in {time.perf_counter() - t0:.1f} s, "
        f"mean bulk modulus {mean:.2f} GPa"
    )


if __name__ == "__main__":  # needed: the processes of the batch import this module
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/polycrystal"))
