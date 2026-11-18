"""Time the generation and voxelization of every model at a given resolution.

Usage: python examples/benchmark.py [n_voxels_per_edge] [threads]

Lattice models (automata, growth, assemblies) are simulated on a lattice of
``n_voxels_per_edge / 2`` cells per edge, then voxelized at the full resolution.
"""

import sys
import time

import morphogen as mg

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
if len(sys.argv) > 2:
    mg.set_num_threads(int(sys.argv[2]))

CASES = [
    ("boolean_spheres", mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3), 40.0, True),
    (
        "boolean_cylinders",
        mg.models.BooleanCylinders(radius=0.5, length=8.0, volume_fraction=0.15),
        40.0,
        True,
    ),
    ("poisson_fibers", mg.models.PoissonFibers(radius=1.0, volume_fraction=0.16), 60.0, False),
    ("hardcore_spheres", mg.models.HardcoreSpheres(radius=1.0, volume_fraction=0.3), 30.0, True),
    ("voronoi", mg.models.Voronoi(n_cells=5000), 1.0, True),
    (
        "laguerre",
        mg.models.Laguerre(intensity=5000.0, radius={"type": "uniform", "low": 0.01, "high": 0.04}),
        1.0,
        True,
    ),
    (
        "gaussian_excursion",
        mg.models.GaussianExcursion(correlation_length=1.0, volume_fraction=0.3),
        32.0,
        True,
    ),
    ("tpms", mg.models.TPMS(cell_size=2.0, volume_fraction=0.2, variant="sheet"), 10.0, True),
    (
        "strut_lattice",
        mg.models.StrutLattice(cell="octet", cell_size=2.0, strut_radius=0.15),
        10.0,
        True,
    ),
]

M = N // 2  # cells per edge of the lattice models
LATTICE_CASES = [
    ("cellular_automaton", mg.models.CellularAutomatonModel(rule="anneal", steps=30)),
    ("activator_inhibitor", mg.models.ActivatorInhibitorModel(steps=15)),
    ("reaction_diffusion", mg.models.ReactionDiffusionModel(pattern="labyrinth")),
    (
        "nucleation_growth",
        mg.models.NucleationGrowthModel(nucleus_density=5000.0 / M**3, coarsening_steps=20),
    ),
    (
        "nucleation_growth",
        mg.models.NucleationGrowthModel(nucleation="continuous", nucleation_rate=5e-5),
    ),
    ("aggregation", mg.models.AggregationModel(process="dla", n_seeds=20, volume_fraction=0.05)),
    ("tile_assembly", mg.models.TileAssembly(tiles="polarized", tile_cells=8)),
]


def lattice_models():
    domain = mg.Domain.cube(float(M))
    for _, model in LATTICE_CASES:
        size = 2.0 if isinstance(model, mg.models.TileAssembly) else 32.0  # compile
        model.generate(mg.Domain.cube(size), seed=0).voxelize(shape=8)
    print(f"\n{'lattice model':20s} {'generate':>9s} {'voxelize':>9s}  ({M}^3 cells, {N}^3 voxels)")
    for name, model in LATTICE_CASES:
        dom = mg.Domain.cube(M / 8.0) if isinstance(model, mg.models.TileAssembly) else domain
        t0 = time.perf_counter()
        micro = model.generate(dom, seed=1)
        t1 = time.perf_counter()
        micro.voxelize(shape=N)
        t2 = time.perf_counter()
        print(f"{name:20s} {t1 - t0:8.2f}s {t2 - t1:8.2f}s  {micro.summary()}")


def main():
    # compile the kernels once on a tiny case
    for _, model, size, periodic in CASES:
        model.generate(mg.Domain.cube(size, periodic=periodic), seed=0).voxelize(shape=8)
    print(f"{'model':20s} {'generate':>9s} {'voxelize':>9s}  (shape {N}^3)")
    for name, model, size, periodic in CASES:
        t0 = time.perf_counter()
        micro = model.generate(mg.Domain.cube(size, periodic=periodic), seed=1)
        t1 = time.perf_counter()
        micro.voxelize(shape=N)
        t2 = time.perf_counter()
        print(f"{name:20s} {t1 - t0:8.2f}s {t2 - t1:8.2f}s  {micro.summary()}")
    lattice_models()


if __name__ == "__main__":
    main()
