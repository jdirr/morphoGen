# Performance

Generation and voxelization at 256³ voxels (16.8 M voxels), 4-core container,
`python examples/benchmark.py 256`:

| Model | Setting | Voxelization |
|---|---|---|
| boolean_spheres | 5 452 spheres | 0.18 s |
| boolean_cylinders | 1 657 fibres | 0.18 s |
| poisson_fibers | 322 fibres | 0.13 s |
| hardcore_spheres | 1 934 spheres, p = 0.3 | 0.14 s |
| strut_lattice | octet, 3 300 struts | 0.18 s |
| tpms | sheet gyroid, 125 cells | 0.39 s |
| gaussian_excursion | 59 756 Fourier modes | 1.9 s |
| voronoi / laguerre | 5 000 cells | 5.0 s / 6.3 s |

Memory goes mostly into the outputs: 1 byte per voxel for phase images (512³ = 134 MB),
4 bytes for labels and fields.

Lattice models, simulated on 128³ cells and voxelized at 256³ (same command):

| Model | Setting | Generation | Voxelization |
|---|---|---|---|
| cellular_automaton | annealing rule, 30 steps | 4.2 s | 0.8 s |
| activator_inhibitor | 15 steps (FFT neighbourhood sums) | 1.1 s | 0.4 s |
| reaction_diffusion | labyrinth, 4 000 steps | 14 s | 0.5 s |
| nucleation_growth | site saturation, 4 850 grains, 20 Potts sweeps | 3.4 s | 0.5 s |
| nucleation_growth | continuous nucleation, 1 157 grains | 1.8 s | 0.6 s |
| aggregation | DLA, 20 clusters, 105 k particles | 16 s | 0.5 s |
| tile_assembly | 16³ polarized blocks (wave function collapse) | 0.01 s | 0.4 s |

Polycrystal meshes (Voronoi, periodic; regularization at a quarter of the element size;
MMG):

| Grains | Element size | Elements | Time | Quality (mean / 0.1 % quantile) |
|---|---|---|---|---|
| 30 | 0.08 | 27 500 | 2.6 s | 0.72 / 0.10 |
| 100 | 0.05 | 108 000 | 9.5 s | 0.74 / 0.10 |
| 300 | 0.04 | 212 000 | 20 s | 0.74 / 0.11 |

Topology optimization (`python examples/topology_optimization.py` and the templates):
48² planar cells in 10–20 s (150–300 iterations), 64² in about a minute, 24³ cells in
about a minute (130 iterations).

Homogenization (`python examples/homogenization.py`, same container):

| Sample | Discretization | Meshing | Solve (per loading) |
|---|---|---|---|
| Boolean spheres, 6³, both phases | 96 k tetrahedra (P1), 43–56 k unknowns | 19 s | 0.7–3.3 s, 31–66 iterations |
| sheet gyroid cell, solid only | 62 k tetrahedra (P2), 305 k unknowns | 72 s | 15–25 s, 50–80 iterations |
| Voronoi mosaic, contrast 20 | 64³ voxels (FFT) | none | 3–3.5 s, 37–38 iterations |

The gyroid's stiffness comes out cubic to 0.01 % (normal–shear couplings below
10⁻⁴ C11), with its solid fraction within 0.6 % of the target.

FFT solvers on the GPU (`python examples/gpu_benchmark.py`; a laptop with an Intel
i9-13900H, 20 threads, and an NVIDIA RTX 3000 Ada laptop GPU, 8 GB):

| Problem | Voxels | CPU | GPU |
|---|---|---|---|
| bulk modulus, Boolean spheres (30 %, contrast 10) | 64³ / 128³ / 192³ / 256³ | 1.4 / 11.1 / 38.6 / 103 s | 0.23 / 2.0 / 5.5 / 29 s |
| crystal plasticity, 12 grains, 1 % in 10 steps | 24³ / 32³ / 48³ | 8.9 / 22 / 125 s | 5.6 / 10 / 44 s |
| stiff spheres in a GTN matrix, tension to 1 % in 10 steps (`--porous`) | 32³ / 48³ / 64³ | 11.5 / 51 / 161 s | 4.8 / 15 / 46 s |

The linear solves are 5 to 7 times faster up to 192³ voxels (3.5 times at 256³); the
loading paths of the polycrystal 1.6 to 2.8 times and those of the porous composite 2.4
to 3.5 times, the integration of the behaviours on the GPU being bound by the double
precision of this laptop GPU ([theory notes](theory.md), §35).

The behaviours of the integration points on the CPU
(`python examples/behaviour_benchmark.py`, the same laptop, 20 threads): an increment
in the middle of a loading, per point:

| Behaviour | Small strains | Finite strains |
|---|---|---|
| elasticity | 10 ns | 254 ns |
| von Mises (Voce and Armstrong-Frederick hardenings) | 88 ns | 302 ns |
| Hill's criterion (the same hardenings) | 345 ns | 542 ns |
| Gurson-Tvergaard-Needleman, with a nucleation | 522 ns | 701 ns |
| crystal plasticity, fcc (12 slip systems) | 1.4 µs | 2.6 µs |

An update of the points of 128³ voxels (2.1 M points) thus takes 0.2 s in a von Mises
matrix and 1.1 s in a porous one; finite strains add the logarithmic strain and its
tangent, about a quarter of a microsecond per point.

The study of the template `rve_poisson_fibers_fe` (25 realizations, KUBC and MBC, three
properties) takes about 7 minutes in one process. Realizations are independent: run
them in processes (`-j`) or in cluster job arrays (`morphogen submit`).
