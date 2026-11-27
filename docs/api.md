# API reference

The Python interface of morphoGen, written from its docstrings. Each package lists its
public names, those of its `__all__` (`from morphogen.homogenization import Material`),
linked to the module that defines them, then its modules; each module documents what it
defines. The commands and the keys of the configuration files are described in the
[configuration reference](configuration.md). The graphical interface (`morphogen.gui`),
the compiled kernels (`morphogen.kernels`) and the functions behind the commands
(`morphogen.cli`) are not part of it. Which of these names stay stable from one version
to the next is the subject of [](stability.md).

## The package at a glance

`import morphogen as mg` gives the domains and images (`mg.Domain`, `mg.VoxelImage`),
the models (`mg.models`, `mg.model_from_config`, `mg.list_models`), `mg.export` and the
packages used most (`mg.analysis`, `mg.mesh`, `mg.io`, `mg.homogenization`, `mg.rve`...);
the others are imported by name (`from morphogen import fracture`). The packages follow
the steps of a study:

| Step | Package | What it holds |
|---|---|---|
| Generate | [`morphogen.models`](api/morphogen.models.rst) | the 43 models of microstructures and the shapes of Microgen (3MAH), their realizations and their theory |
| | [`morphogen.automata`](api/morphogen.automata.rst) | cellular automata and lattice growth processes, on grids and tilings |
| | [`morphogen.design`](api/morphogen.design.rst) | tile assemblies, strut frameworks and their rigidity, searches of design spaces |
| | [`morphogen.optimization`](api/morphogen.optimization.rst) | topology optimization of periodic cells by inverse homogenization, and the inverse design of active structures |
| | [`morphogen.iga`](api/morphogen.iga.rst) | isogeometric analysis of NURBS geometries and their shape optimization |
| Describe | [`morphogen.mesh`](api/morphogen.mesh.rst) | conforming, exact and voxel meshes |
| | [`morphogen.io`](api/morphogen.io.rst) | images, meshes, geometries and the inputs of other codes, read and written |
| | [`morphogen.interop`](api/morphogen.interop.rst) | other libraries and programs: the arrays of JAX, PyTorch and CuPy, Microgen (3MAH), Blender, the tools of alloy design |
| | [`morphogen.viz`](api/morphogen.viz.rst) | sections, figures, 3D views and videos |
| Measure | [`morphogen.analysis`](api/morphogen.analysis.rst) | the morphological measures of images, percolation, stereology |
| | [`morphogen.theory`](api/morphogen.theory.rst) | the closed forms of stochastic geometry used by the models |
| | [`morphogen.texture`](api/morphogen.texture.rst) | crystallographic textures and orientation maps |
| Compute | [`morphogen.homogenization`](api/morphogen.homogenization.rst) | apparent properties, materials and behaviours, loading paths, the solvers |
| | [`morphogen.fracture`](api/morphogen.fracture.rst) | cohesive zones, cracks and their growth, the phase field |
| | [`morphogen.contact`](api/morphogen.contact.rst) | contact, self-contact and friction of the surfaces of a mesh |
| | [`morphogen.micromorphic`](api/morphogen.micromorphic.rst) | micromorphic continua: the microfoam model, micromorphic damage |
| | [`morphogen.coupling`](api/morphogen.coupling.rst) | transient diffusion, oxidation, ageing, hygromorphs, and the thermo-chemo-elastic fields in time |
| | [`morphogen.learning`](api/morphogen.learning.rst) | behaviours learned from data |
| | [`morphogen.symmetry`](api/morphogen.symmetry.rst) | symmetry groups of periodic cells, for the patterns after buckling |
| | [`morphogen.gpu`](api/morphogen.gpu.rst) | the choice of the device of the FFT solvers |
| Study | [`morphogen.rve`](api/morphogen.rve.rst) | the statistical size of the representative volume element |
| | [`morphogen.results`](api/morphogen.results.rst) | the database of the results and the Ashby charts |
| | [`morphogen.config`](api/morphogen.config.rst), [`morphogen.pipeline`](api/morphogen.pipeline.rst), [`morphogen.parallel`](api/morphogen.parallel.rst) | configuration files, their runs, in parallel and on clusters |
| Applications | [`morphogen.laser`](api/morphogen.laser.rst) | metal sheets architectured by localized laser treatment |
| | [`morphogen.dynamics`](api/morphogen.dynamics.rst) | explicit dynamics with OpenRadioss |
| Foundations | [`morphogen.domain`](api/morphogen.domain.rst), [`morphogen.image`](api/morphogen.image.rst), [`morphogen.primitives`](api/morphogen.primitives.rst) | boxes and grids, voxel images, spheres and segments |
| | [`morphogen.distributions`](api/morphogen.distributions.rst), [`morphogen.orientations`](api/morphogen.orientations.rst), [`morphogen.seeding`](api/morphogen.seeding.rst) | laws of sizes and of orientations, reproducible random numbers |
| | [`morphogen.settings`](api/morphogen.settings.rst), [`morphogen.stopping`](api/morphogen.stopping.rst) | threads, and the stop of long computations |

## Modules

```{toctree}
:maxdepth: 2

api/morphogen
```
