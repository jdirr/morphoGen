# User guide

A study with morphoGen follows one chain: a model of the microstructure, its realizations,
their images and meshes, their measures against theory, their apparent properties, their
responses along loading paths up to fracture, and the size of the volume that represents
the medium. The pages of this guide follow that chain. Each says what morphoGen does at
its step, then shows how, in Python and from configuration files; the
[tutorials](tutorials.md) follow whole studies, and the [recipes](howto.md) answer short
questions.

| Step | Page | What it covers |
|---|---|---|
| Generate | [Models of microstructures](models.md) | the 46 models by family and the shapes of Microgen, their anisotropy, their sizes (monodisperse or polydisperse), automata and growth, nested and bio-inspired models |
| Describe | [Geometry, voxels and meshes](geometry.md) | exact descriptions, implicit functions, voxel images, conforming meshes, symmetric cells |
| Measure | [Measures and theory](measures.md) | the closed forms of the models, the measures of images, percolation, crystallographic textures |
| Compute | [Apparent properties](properties.md) | the solvers, elasticity and conduction, polycrystals, mixed loadings, the other physics, frames of beams, waves |
| | [Guide to the physics](physics.md) | every physical problem, its solver and the name of its property |
| Load | [Behaviours along loading paths](behaviours.md) | plasticity, crystals, finite strains, MFront and Z-set laws, identification, reduced and learned models, micromorphic continua |
| Break | [Fracture, contact and fatigue](fracture.md) | the phase field, cohesive zones and cracks, interfaces that debond, contact and friction, high-cycle fatigue |
| Size | [Size of the representative volume](rve.md) | the statistics of the RVE, a study, nonlinear responses, results of other codes |
| Design | [Design and optimization](design.md) | assemblies and rigidity, topology optimization, isogeometric shape optimization, patterns after buckling |

Two application modules build on this chain: [laser-architectured sheets](laser.md)
(`morphogen.laser`) and [explicit dynamics with OpenRadioss](dynamics.md)
(`morphogen.dynamics`). Every step runs from Python, from
[configuration files and the command line](running.md) and from the
[graphical interface](interface.md), and its results go to
[other codes](interop.md).

## The package

The modules of `morphogen` follow the same steps: `models` and `automata` generate,
`mesh` and `io` discretize and write, `analysis` measures, `homogenization`, `fracture`,
`contact` and `micromorphic` compute, `rve` and `results` gather the studies. The
[map of the package](api.md#the-package-at-a-glance) lists them all, with the page of the
API reference of each.

## Examples

The folder [`examples/`](https://github.com/jdirr/morphoGen/tree/main/examples) of the
repository holds complete scripts, to run and to change:

| Topic | Scripts |
|---|---|
| A first script | `minimal.py`: generate, measure, compare with theory, export |
| Studies | `rve_study.py` (a Voronoi mosaic, its integral range recovered), `homogenization.py` (finite elements and FFT), `multiphysics.py` (one microstructure homogenized for its elastic, thermal, dielectric and diffusive properties, its expansions, its poroelastic and piezoelectric laws, its dynamic moduli, and cut by extended finite elements), `acoustic_foam.py` (from an open-cell foam to the parameters of its equivalent fluid and the sound absorption of layers of it), `polycrystal.py` (meshes of grains, crystals, exports, a parallel batch), `ebsd_map.py` (an orientation map from end to end: its grains, its texture, the stiffness of its polycrystal, windows of it as the realizations of a study), `alloy_design.py` (an alloy from a tool of alloy design: its scales as microstructures and their elasticity) |
| Generation and design | `automata_and_design.py` (automata, growth, aggregation, tile assemblies, rigidity, inverse design of rules), `topology_optimization.py` (bulk modulus and auxetic cells), `architectures_ashby.py` (lattices, foams, trabecular bone, honeycombs and fractal sponges homogenized, recorded in a database and placed on a normalized Ashby chart), `scale_rich_networks.py` and `scale_rich_networks_3d.py` (the laws of the networks of Both et al. 2026, in the plane and in space), `exact_meshes.py` (a morphology of each route meshed from its exact geometry) |
| Behaviours and fracture | `fracture.py` (cohesive zones of a double cantilever beam against the corrected beam theory, an edge crack by XFEM against the handbook), `microfoam.py` (the hole size effect of nickel foams with the microfoam model, a crack as a band of micromorphic damage on two meshes, the densification of the foam compressed to half its height in finite strains, a ductile crack by micromorphic ductile damage on two meshes), `ductile_triaxiality.py` (micromorphic ductile damage with and without the stress triaxiality), `hill_of_a_laminate.py` (Hill's law of a laminate identified on its responses, written for Z-set), `gurson_of_pores.py` (Gurson's q1 and q2 of spherical pores in a von Mises matrix, from a tension and a confined one), `pattern_transformation.py` (the pattern transformation of a square array of holes by the post-bifurcation analysis) |
| Performance | `benchmark.py` (generation and voxelization of every model), `gpu_benchmark.py` (the FFT solvers on the CPU and on the GPU), `behaviour_benchmark.py` (the cost of each behaviour of the integration points); their figures are on the page [Performance](performance.md) |
