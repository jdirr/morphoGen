# Recipes: how do I...

Short answers to frequent questions, each a snippet to copy. They follow each other in
one Python session, on small sizes so that all of them run in half a minute (the tests
of morphoGen run them); the [tutorials](tutorials.md) walk through whole studies and the
[guide to the physics](physics.md) explains what the solvers compute.

```python
import numpy as np
import morphogen as mg
from morphogen.homogenization import Material, homogenize
```

## Geometry

### ... get the same realization again?

A realization is a function of its model, its domain and its seed. The realizations of
a run take the seeds derived from the seed of its configuration, and the geometry itself
(centres, radii, cells...) is saved and read back exactly:

```python
from morphogen.seeding import derive_seed

model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.2)
domain = mg.Domain.cube(8.0, periodic=True)
micro = model.generate(domain, seed=derive_seed(7, 0))  # the realization 0 of the seed 7
again = model.generate(domain, seed=derive_seed(7, 0))
assert np.array_equal(micro.voxelize(shape=24).data, again.voxelize(shape=24).data)
mg.io.save_microstructure(micro, "spheres.json")
same = mg.io.load_microstructure("spheres.json")  # voxelized or meshed at any resolution
```

### ... build a model from the keys of a configuration?

Every model has a name and parameters, those of the `model` section of a configuration
file and of the forms of the interface:

```python
from morphogen.models import get_model, model_from_config

print(len(mg.list_models()))  # 39
print([p["name"] for p in get_model("hardcore_spheres").parameters()])
packing = model_from_config({"type": "hardcore_spheres", "radius": 1.0, "volume_fraction": 0.2})
```

### ... put inclusions in a matrix that has its own structure?

Models combine (union, intersection, difference, inclusions in a matrix) and nest
(coatings, particles clustered in aggregates):

```python
both = model_from_config(
    {
        "type": "combination",
        "operation": "inclusions",  # each model after the first is a phase in the first
        "models": [
            {"type": "voronoi", "n_cells": 8, "boundary_thickness": 0.3},
            {"type": "hardcore_spheres", "radius": 0.8, "volume_fraction": 0.1},
        ],
    }
)
print(both.active_phases())  # 0: grain boundaries, 1: cells, 2: the inclusions
mixed = both.generate(domain, seed=1)
print(mixed.implicit_phases())  # each phase has its implicit function: [0, 1, 2]
```

### ... study a planar section instead of the volume?

One layer of voxels through the middle of the volume: a plane strain for the solvers of
voxels, on far fewer voxels. In a configuration: `voxels: {shape: 64, section: z}`.

```python
image = micro.voxelize(shape=32)
section = image.layer(axis=2)  # the middle layer normal to z, one voxel thick
print(image.shape, section.shape)  # (32, 32, 32) (32, 32, 1)
```

### ... start from a measured orientation map (EBSD)?

An `.ang` (EDAX/TSL) or a `.ctf` (Oxford Instruments/HKL) file is read with its
orientations in the axes of its image; its grains are the sets of neighbouring points
less than a threshold apart, each with its mean orientation, and their polycrystal goes
to every solver. Here the map that comes with morphoGen:

```python
from morphogen import texture
from morphogen.io import ebsd

measured = ebsd.read_ebsd(ebsd.example_map())  # or the path of a file: "my_map.ang"
grains = measured.grains(threshold=5.0, min_size=5)
figure = texture.pole_figure(grains.orientations, (1, 0, 0), weights=grains.areas)
print(grains.count, round(figure.maximum, 1))  # 56 grains; {100} poles 5.7 times random
crystals = grains.microstructure()  # the grains as columns through one layer of cells
shape = (64, 48, 1)  # the points of the map
by_grain = {"grains": crystals.labels(shape).data, "orientations": crystals.orientations}
nickel = {1: Material.cubic(246.5, 147.3, 124.7)}
moduli = homogenize(crystals.voxelize(shape), nickel, properties=["bulk_modulus"], **by_grain)
print(moduli["bulk_modulus.periodic"].value)  # (C11 + 2 C12) / 3 = 180.37, exactly
```

In a configuration, `model: {type: ebsd, file: my_map.ctf, window: random}` takes a
window of the map for each realization (the template `ebsd`). An `.ang` file has its
orientations in axes that depend on a setting of the acquisition (`setting: 1` to `4`):
[theory](theory.md), §92. `morphogen texture my_map.ctf -o texture/` prints the numbers
of the map and writes its tables and figures (the map with its grain boundaries, pole
figures, the disorientations of the boundaries).

A map is a section: where the third dimension matters, a polycrystal of space takes the
texture of the map, each cell the orientation of a grain drawn with its area:

```python
textured = mg.models.Voronoi(
    n_cells=200,
    crystal_orientations=True,
    texture={"type": "measured", "file": "ebsd_map.ctf", "spread": 2.0},
)
polycrystal = textured.generate(mg.Domain.cube(10.0), seed=1)
poles = texture.pole_figure(polycrystal.orientations, (1, 0, 0))
print(round(poles.at([0, 0, 1]), 1))  # the cube component of the map: 5.7 times random
```

### ... get a surface for a printer or a CAD tool?

The zero level of the implicit function of a phase, closed on the faces of the box:

```python
from morphogen.mesh.surface import surface_mesh

surface = surface_mesh(micro.level_set(shape=48), closed=True)
mg.io.save_mesh(surface, "spheres.stl")
print(len(surface.faces), surface.volume() / domain.volume)  # near the volume fraction
```

### ... mesh only the solid of a porous medium?

```python
solid = mg.mesh.conforming_mesh(micro, size=1.0, phases="matrix")  # the spheres are pores
print(solid.n_elements, solid.phase_fractions())  # the matrix alone: phase 0
```

`phases="solid"` keeps phase 1 when phase 0 is the void; a configuration does it by
itself when a material is `void` (`mesh: {phases: auto}`).

## Properties

### ... get the whole stiffness tensor and its engineering constants?

```python
materials = {0: Material(young=3.0, poisson=0.35), 1: Material(young=70.0, poisson=0.22)}
stiffness = homogenize(image, materials, properties=["stiffness"])["stiffness.periodic"]
print(stiffness["tensor"].shape, stiffness["E1"], stiffness["nu12"], stiffness["G23"])
print(stiffness["universal_anisotropy"])  # 0 for an isotropic tensor
```

The tensor is in Voigt notation 11, 22, 33, 23, 31, 12 with engineering shears;
`stiffness["compliance"]` is its inverse in the same notation.

### ... prescribe stresses instead of strains?

A loading names the components it prescribes, as strains or as stresses; the others are
free of stress. A uniaxial stress along x:

```python
out = homogenize(image, materials, properties=[], loadings={"pull": {"stress": {"11": 1.0}}})
response = out["pull.periodic"].components()  # the macroscopic strains E.. and stresses S..
print(response["E11"], response["S22"])  # 1 / E1, and a zero lateral stress
```

### ... compute a conductivity, a permittivity, a diffusivity?

The same problem with the coefficient of each phase for the physics:

```python
conducting = {
    0: Material(young=3.0, poisson=0.35, conductivity=0.2, permittivity=3.0),
    1: Material(young=70.0, poisson=0.22, conductivity=30.0, permittivity=10.0),
}
out = homogenize(image, conducting, properties=["conductivity", "permittivity_tensor"])
print(out["conductivity.periodic"].value, out["permittivity_tensor.periodic"]["tensor"][0, 0])
```

### ... know between which bounds a property must lie?

```python
from morphogen import analysis
from morphogen.homogenization import bounds

limits = bounds(analysis.phase_fractions(image), materials)["bulk_modulus"]
print(limits["reuss"], limits["hs_lower"], limits["hs_upper"], limits["voigt"])
```

### ... choose between voxels, a conforming mesh and extended finite elements?

The FFT solver on voxels for speed and large volumes (periodic conditions; on an NVIDIA
GPU with `device="gpu"`); finite elements on a conforming mesh for accurate interfaces,
the kinematic, static and mixed boundary conditions, and cracks; extended finite elements
(`LevelSetSample`) for two-phase structures without a mesher. The three take the same
materials and give the same results as they refine:

```python
from morphogen.homogenization import LevelSetSample

mesh = mg.mesh.conforming_mesh(micro, size=1.0)
names = ("voxels", "mesh", "cut grid")
samples = (micro.voxelize(shape=24), mesh, LevelSetSample.of(micro, 10))
for name, sample in zip(names, samples, strict=True):
    out = homogenize(sample, materials, bcs=["periodic"], properties=["bulk_modulus"])
    print(name, out["bulk_modulus.periodic"].value)
```

## Behaviours

### ... write a loading path?

Times, increments and components in a line, or as a mapping; strains `E`, stresses `S`,
the components not named free of stress:

```python
from morphogen.homogenization import LoadingPath

tension = LoadingPath.parse("t=0,1 n=10 E11=0,0.01")
cycle = LoadingPath.parse("t=0,1,2,3 n=5 E11=0,0.01,-0.01,0.01")
creep = LoadingPath.from_spec(
    {"times": [0, 1, 100], "increments": [5, 10], "stress": {"11": [0, 50, 50]}}
)
print(len(cycle.schedule()), creep.strain_controlled)
```

### ... try a behaviour without a microstructure?

At a material point: homogeneous strain and stress, the free components solved for.

```python
from morphogen.homogenization import simulate_point

alloy = Material(
    young=70e3, poisson=0.33, plasticity={"yield": 100.0, "isotropic": [{"Q": 80.0, "b": 20.0}]}
)
point = simulate_point(alloy, cycle)
print(point.stress[5, 0], point.stress[10, 0], point.plastic[-1])  # it yields, both ways
```

### ... give a phase a hyperelastic law, and follow it in finite strains?

```python
rubber = Material(hyperelasticity={"model": "neo_hookean", "mu": 0.5, "bulk": 5.0})
stretch = LoadingPath.parse("t=0,1 n=5 E11=0,0.5")
large = simulate_point(rubber, stretch, kinematics="finite")
print(large.stress[-1, 0], large.cauchy[-1, 0], large.log_strain[-1, 0])  # nominal, true
filled = {0: rubber, 1: Material(young=50.0, poisson=0.3)}
out = homogenize(
    micro.voxelize(shape=8), filled, properties=[], paths={"pull": stretch},
    nonlinear={"kinematics": "finite"},
)  # fmt: skip
print(out["pull.periodic"].stress[-1, 0] / large.stress[-1, 0])  # stiffer than the rubber
```

The laws: `neo_hookean`, `mooney_rivlin`, `polynomial`, `yeoh`, `arruda_boyce`,
`eight_chain`, `gent`, `van_der_waals`, `fung`, `ogden`, `holzapfel`, `blatz_ko`,
`hyperfoam`; `fit_hyperelasticity` fits them to tests
([a rubber, from its tests to its tearing](tutorial_rubber.md)).

### ... break a cell?

The phase field: a toughness for the phases that break and a regularization length of
two voxels at least. Here a section pulled until it breaks:

```python
from morphogen.fracture import phase_field_fracture

fracture = {"toughness": {0: 1e-4}, "length": 2 * 8.0 / 16, "model": "AT1"}
plane = micro.voxelize(shape=16).layer()
pull = LoadingPath.parse("t=0,1 n=10 E11=0,0.02")
broken = phase_field_fracture(plane, materials, fracture, pull)
print(broken.strength(), broken.stress[-1, 0], broken.crack_surface[-1])
```

`kinematics="finite"` runs it in finite strains (hyperelastic or elastic phases).

## Studies

### ... find the size of the representative volume from results I already have?

The variance of a property over realizations of several volumes, fitted as
`D²(V) = K V^-γ`, gives the volume for a relative error with a number of realizations:

```python
from morphogen import rve

rng = np.random.default_rng(0)
volumes = np.repeat([64.0, 216.0, 512.0], 200)  # 200 realizations of three volumes
values = 5.0 + rng.standard_normal(volumes.size) * np.sqrt(4.0 / volumes)  # K = 4, gamma = 1
law = rve.fit_statistics(rve.ensemble_statistics(volumes, values), gamma=1.0)
print(law.K, law.gamma)  # near 4, with gamma imposed (free without it)
print(rve.rve_table(law, values.mean())[0])  # the volume for 1 % with one realization
```

`morphogen fit results.csv` does the same from a table of any origin; the tutorial
[how large is a representative volume?](tutorial_rve.md) goes through the method.

### ... run many realizations, on my machine or on a cluster?

```bash
morphogen run study.yaml -j 8                 # 8 processes on this machine
morphogen run study.yaml -j 8 --resume        # after an interruption: only what is missing
morphogen status out/study                    # the realizations done and missing
morphogen submit study.yaml --scheduler slurm --tasks 20 --time 02:00:00
morphogen merge out/study                     # gather the results of the tasks
```

### ... keep every result, and compare architectures on a chart?

```bash
morphogen run study.yaml --db results.db      # the run joins a database (SQLite)
morphogen db list --db results.db
morphogen ashby --db results.db --x solid_fraction --y stiffness.periodic.E1.relative -o chart.png
```

## Other codes

### ... send a mesh and its materials to Abaqus or CalculiX?

```python
mg.io.save_mesh(mesh, "composite.inp", materials=materials)  # element sets, sections, materials
mg.io.write_periodic_equations("composite_equations.inp", mesh)  # the periodic constraints
mg.io.save_mesh(mg.mesh.voxel_mesh(image), "voxels.inp")  # hexahedra, a set per phase
```

### ... run AMITEX_FFTP on an image?

```python
files = mg.io.write_amitex("amitex", image, materials, paths={"tension": tension}, name="spheres")
print(sorted(p.name for p in files.files)[:3])  # the fields, the XML files and the script
```

`morphogen amitex read amitex/ --name spheres` reads the results of the run back.

### ... get the MFront source of a behaviour?

```python
mg.io.write_mfront(alloy, "laws/Alloy.mfront")  # for Abaqus, Cast3M, code_aster, AMITEX_FFTP
```
