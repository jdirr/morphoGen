# A guide to the physics

This page is for researchers in the mechanics and physics of materials who want to know
*what morphoGen computes, with which solver, and how to ask for it*, before reading the
[theory notes](theory.md) or the [API reference](api.md). It goes through every physical
problem of the homogenization module: linear properties (elastic, thermal, electrical,
dielectric, magnetic, diffusive), their weak and strong couplings, harmonic regimes and
waves (viscoelastic, electromagnetic, optical, acoustic), nonlinear behaviours along
loading paths (plasticity, viscoelasticity, hyperelasticity, damage), fracture, the flow
of a fluid through the pores and the acoustics of porous media, and the statistics of the
size of the representative volume element, which apply to all of them.

The scripts `examples/multiphysics.py` and `examples/acoustic_foam.py` are two complete
studies to run and change. The Python snippets below follow each other in one session and
run in seconds to a minute on a laptop (the first run of each solver compiles its kernels, once); they are run by the
tests of the project, so that they stay correct. The [tutorial](tutorial.md) follows one
study from end to end; the [user guide](guide.md) goes through each step, and the
[configuration reference](configuration.md) lists every option.

## 1. The chain of a computation

Every study follows the same four steps.

1. **A model** generates a microstructure: an explicit description (centres, radii, axes,
   germs) that can be saved and regenerated exactly (the [gallery](gallery.md) shows the
   models).
2. **A sample** discretizes it. There are four kinds, each with its solver:

   | Sample | Made by | Solver | Boundary conditions |
   |---|---|---|---|
   | voxel image (`VoxelImage`) | `micro.voxelize(shape=...)` | FFT-preconditioned conjugate gradients on trilinear voxels (CPU or GPU) | periodic |
   | conforming mesh (`TetMesh`) | `mg.mesh.conforming_mesh(micro, ...)` | finite elements: 4-node tetrahedra, or 10-node ones (`order=2`) with curved edges on the interfaces | kinematic uniform (`kubc`), static uniform (`subc`), `mixed`, `periodic` |
   | level set on a grid (`LevelSetSample`) | `LevelSetSample.of(micro, shape)` | extended finite elements: a regular grid cut by the interfaces | periodic |
   | mesh of the plane (`TriMesh`) | `mg.mesh.plane_mesh(micro, ...)`: the columnar grains of seeds of the plane, the section of any structure | finite elements of the plane: 3- or 6-node triangles, generalized plane strain or plane strain; loading paths on its layer in space | `kubc`, `subc`, `periodic` (paths: `periodic`) |

3. **A problem** applies macroscopic loadings to the sample for one physics and gives an
   *apparent property*: `homogenize(sample, materials, properties=[...])`.
4. **Statistics** over realizations and sizes tell how far the apparent property of a
   finite volume is from the effective one, and the size of the representative volume
   element (RVE) for a wanted precision (§9).

Voxels are the fastest and need no mesh, but they draw interfaces as stairs; conforming
meshes follow the interfaces and take every boundary condition; the extended finite
elements follow the interfaces without a mesher (§8 compares them).

## 2. Conventions

* **Units**: none are imposed; use one consistent system. The examples of this page are in
  SI units (Pa, W/(m K), S/m, F/m, m²/s).
* **Tensors**: stresses and strains use the Voigt order `11 22 33 23 31 12`, the strains
  with engineering shears (`gamma_23 = 2 eps_23`); a stiffness is a 6 x 6 matrix in that
  order. Second-order tensors (conductivities, expansions) are 3 x 3.
* **Fluxes** follow the constitutive law of their physics: `q = -k grad T`,
  `j = -sigma grad V` (i.e. `j = sigma E`), `D = eps E`, `B = mu H`, `j = -D grad c`.
* **Phases** are the integers of the image; `materials` maps each phase to a `Material`.
  `VOID` (or `"void"` in a configuration) is a pore for every physics; a zero coefficient
  makes a phase inert for one physics only (an electrical insulator that still conducts
  heat).
* **Results** are named `"<property>.<boundary condition>"`, e.g.
  `"bulk_modulus.periodic"`, in Python and in the tables of results.

## 3. A sample

A Boolean model of spheres in a periodic cube, voxelized; the same realization serves
every section below.

```python
import numpy as np

import morphogen as mg
from morphogen.homogenization import VOID, Material, homogenize

model = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.2)
micro = model.generate(mg.Domain.cube(6.0, periodic=True), seed=1)
image = micro.voxelize(shape=16)  # 16³ voxels: phase 0 (matrix), phase 1 (spheres)
```

## 4. Linear properties

### Elasticity

The scalar properties are objects with a `value`; the tensors are mappings.

```python
matrix = Material(young=70e9, poisson=0.33)  # aluminium
sphere = Material(young=410e9, poisson=0.19)  # silicon carbide
out = homogenize(image, {0: matrix, 1: sphere}, properties=["bulk_modulus", "stiffness"])
bulk = out["bulk_modulus.periodic"].value  # Pa
stiffness = out["stiffness.periodic"]
print(stiffness["tensor"].shape, stiffness["E1"], stiffness["nu12"])  # (6, 6) and constants
```

Crystals take a stiffness matrix (`Material.cubic`, `Material.hexagonal`,
`Material.anisotropic`) that each grain of a polycrystal turns by its orientation.

### Transport of a scalar potential

Heat conduction, electrical conduction, dielectrics, magnetostatics and diffusion are the
same mathematical problem, `div(a grad u) = 0`; each has its coefficient in `Material`
and its properties:

| Physics | Potential, flux | Coefficient of `Material` | Properties |
|---|---|---|---|
| thermal | temperature, heat flux | `conductivity` | `conductivity`, `conductivity_tensor` |
| electrical | electric potential, current density | `electrical_conductivity` | `electrical_conductivity`, `electrical_conductivity_tensor` |
| dielectric | electric potential, electric displacement | `permittivity` | `permittivity`, `permittivity_tensor` |
| magnetic | magnetic potential, magnetic induction | `permeability` | `permeability`, `permeability_tensor` |
| diffusion | concentration, flux of matter | `diffusivity` | `diffusivity`, `diffusivity_tensor` |

A coefficient is one value or, in the axes of a crystal, three principal values or a
3 x 3 tensor.

```python
phases = {
    0: Material(
        conductivity=0.2, electrical_conductivity=1e-12, permittivity=3.0, diffusivity=1e-12
    ),
    1: Material(
        conductivity=30.0, electrical_conductivity=1e4, permittivity=10.0, diffusivity=0.0
    ),  # the spheres are impermeable
}
out = homogenize(
    image,
    phases,
    properties=["conductivity", "electrical_conductivity", "permittivity_tensor", "diffusivity"],
)
k = out["conductivity.periodic"].value
eps = out["permittivity_tensor.periodic"]["tensor"]  # 3 x 3
tortuosity = 1e-12 * (1 - image.data.mean()) / out["diffusivity.periodic"].value
```

The Wiener and Hashin–Shtrikman bounds of each property come with the results
(`morphogen.homogenization.bounds`).

### Weak couplings: a field that strains the phases

A uniform field that strains each phase differently is an eigenstrain problem of the
elastic solvers: a change of temperature (`expansion`), the content of an absorbed
species such as moisture (`swelling`), the pressure of a fluid in the pores (the
poroelasticity of Biot).

```python
filled = {
    0: Material(young=3e9, poisson=0.35, expansion=60e-6, swelling=0.4),
    1: Material(young=70e9, poisson=0.22, expansion=5e-6),
}  # the glass stays dry
out = homogenize(image, filled, properties=["thermal_expansion", "swelling"])
alpha = out["thermal_expansion.periodic"]["tensor"]  # 3 x 3, per kelvin
beta = out["swelling.periodic"]["mean"]  # strain per unit moisture content

rock = {0: Material(young=20e9, poisson=0.2), 1: VOID}
porous = homogenize(image, rock, properties=["poroelastic"], fluid_bulk_modulus=2.2e9)[
    "poroelastic.periodic"
]
print(porous["biot_coefficient"], porous["biot_modulus"], porous["skempton_coefficient"])
```

`poroelastic` gives the drained stiffness `C_d`, the tensor `b` and the modulus `N` of
Biot (`Sigma = C_d : E - b p`, change of porosity `b : E + p / N`) for any solid, even
heterogeneous or crystalline, and with the bulk modulus of the fluid the undrained
stiffness and the tensor of Skempton. The transient diffusion of a species, oxidation and
ageing are in `morphogen.coupling`: `transient_diffusion` and `oxidation_field` run in the
voxels of an image (moisture or oxygen entering by its exposed faces, around the phases
that bar their way; the oxidation products of a polymer matrix between its fibres), and
`graded_materials` turns the field of an ageing tracer into graded phases for the
solvers above.

### Strong couplings: fields solved together

Where the constitutive law itself couples two physics, their fields are solved together
(voxel images): piezoelectric, piezomagnetic, magneto-electro-elastic, rigid
magnetoelectric and thermoelectric composites. The coupling coefficients of a phase are
its `coupling`; the property named after the coupling gives the tensors of the effective
law and the constants derived from them.

```python
import dataclasses

ceramic = dataclasses.replace(  # barium titanate poled along x3
    Material.hexagonal(166e9, 77e9, 78e9, 162e9, 43e9),
    permittivity=(11.2e-9, 11.2e-9, 12.6e-9),
    coupling={"piezoelectric": {"e31": -4.4, "e33": 18.6, "e15": 11.6}},
)
polymer = Material(young=4e9, poisson=0.35, permittivity=3.7e-11)
law = homogenize(image, {0: polymer, 1: ceramic}, properties=["piezoelectric"])[
    "piezoelectric.periodic"
]
print(law["piezoelectric"].shape, law["d33"], law["k_t"])  # e (3 x 6), C/N, coupling factor
```

A composite of a piezoelectric and a piezomagnetic phase is magnetoelectric although
neither phase is (`magnetoelectroelastic`: a product property); `thermoelectric` gives
the effective Seebeck coefficient, conductivities and figure of merit.

### Couplings in time: heat, a species and the strains they cause

Where the stimulus takes time to reach the phases, the fields are followed in the volume
of the sample, three-dimensional fields at each output time:
`morphogen.coupling.thermochemoelastic` solves together the temperature (Fourier's law, a
volumetric heat capacity, the heat that a reaction releases), the activity of a species
(its diffusion, its solubility in each phase, a first-order consumption, its drift by
the gradient of the hydrostatic stress when it has a partial molar volume) and the
displacement, in quasi-static equilibrium under the thermal and chemical expansions.
The diffusivity and the rate of the reaction follow Arrhenius' law; fields coupled both
ways (a reaction that heats, a species that the stress draws and that swells) are solved
to a fixed point at each step. It runs on voxel images (finite volumes and trilinear
hexahedra; the periodic axes free to expand, or held) and on meshes of linear
tetrahedra, by implicit Euler or TR-BDF2 steps.

```python
from morphogen.coupling import ThermoChemoMaterial, thermochemoelastic

# moisture enters the cell by its faces x- and x+ (lengths in mm, times in s, stresses in Pa)
humid = {
    0: ThermoChemoMaterial(young=3e9, poisson=0.35, diffusivity=1e-6, swelling=3e-3),
    1: ThermoChemoMaterial(young=70e9, poisson=0.22),  # the glass stays dry
}
wet = thermochemoelastic(image, humid, [1e5, 1e6, 1e7], activity={"x-": 1.0, "x+": 1.0})
print(wet.history["uptake"], wet.mean_strain[:, 1])  # the uptake; the cell swells along y
print(wet.von_mises[-1].max())  # the stresses where the resin swells around the glass
wet.write_xdmf("wet.xdmf")  # the fields at each time, for ParaView
```

The faces are held at a temperature or an activity (`temperature=`, `activity=`: a value
or a function of time) or left insulated; `supports` clamp faces or make them planes of
symmetry (`{"x-": "clamped"}`), the rigid motions that remain being removed without
reaction; `points` follows displacements in the history, and `curvature` gives that of a
strip. The coefficients of a phase are those of its `ThermoChemoMaterial` (or of a
`Material`, with `ThermoChemoMaterial.from_material`). Where the expansion of the phases
should give a structure a shape, at equilibrium or along time, their layout is designed
by the inverse problem of [](design.md) (`morphogen.optimization.MorphingOptimization`).

## 5. Dynamics, flow and sound

### Harmonic regimes

Under a harmonic excitation, dissipative phases have complex laws, and the composite a
complex effective tensor at each frequency, as long as the wavelengths are much longer
than the microstructure: storage and loss moduli of viscoelastic phases
(`dynamic_stiffness`), conductivity and permittivity of leaky dielectrics
(`complex_conductivity`: the interfacial polarization of Maxwell and Wagner), and the
effective refractive index of optical media (`refractive_index`).

```python
resin = Material(
    young=3e9,
    poisson=0.35,
    viscoelasticity={
        "shear": [[0.5, 1e-2], [0.3, 1.0]],  # Prony series: (relative modulus, time)
        "bulk": [[0.2, 0.1]],
    },
)
glass = Material(young=70e9, poisson=0.22)
dma = homogenize(
    image, {0: resin, 1: glass}, properties=["dynamic_stiffness"], frequencies=[0.1, 10.0, 1000.0]
)["dynamic_stiffness.periodic"]
print(dma["shear_storage_2"], dma["shear_loss_2"], dma["shear_tan_delta_2"])  # at 10 rad/s

optics = {0: Material(refractive_index=1.5), 1: Material(refractive_index=(2.4, 0.05))}
index = homogenize(image, optics, properties=["refractive_index"], wavelengths=[0.6])[
    "refractive_index.periodic"
]
print(index["index_1"], index["extinction_1"])  # n and k of the composite
```

### Waves in periodic media

When the wavelength is comparable to the period, the waves are those of Bloch:
`band_structure` gives the dispersion and the band gaps of the elastic waves of a
periodic cell, and of the waves named by `waves`: `"acoustic"` (the pressure of a
fluid), `"electromagnetic"` (light in a three-dimensional crystal: the vector waves of
Maxwell's equations, by plane waves on the voxels of an image) and, in the plane of a
medium invariant along an axis (a cell one voxel thick), `"shear"` (antiplane shear) and
the two polarizations `"tm"` and `"te"` of light (photonic crystals of rods or holes).

```python
from morphogen.homogenization import band_structure

n = 12
x = (np.arange(n) + 0.5) / n - 0.5
rods = (x[:, None] ** 2 + x[None, :] ** 2 < 0.2**2).astype(np.uint8)[:, :, None]
crystal = mg.VoxelImage(rods, 1.0 / n, periodic=True)  # one voxel thick: a planar cell
air, alumina = Material(permittivity=1.0), Material(permittivity=8.9)
bands = band_structure(
    crystal, {0: air, 1: alumina}, ("G", "X", "M", "G"), bands=3, points=4, waves="tm"
)
gaps = [(low / (2 * np.pi), high / (2 * np.pi)) for _, low, high in bands.gaps()]
print(gaps)  # about (0.32, 0.44) in units of c / a: the gap of Joannopoulos et al.
```

`long_wavelength_speeds` gives the speeds of the homogenized medium (Wood's law for the
sound of mixtures); `scalar_band_structure` takes the coefficients of
`div(a grad u) + omega² b u = 0` directly, for fluids and rigid scatterers.

In three dimensions light is a vector wave. Dielectric spheres on the diamond lattice
open a complete band gap, the first photonic crystal with one (Ho, Chan and Soukoulis):

```python
grid = (np.arange(48) + 0.5) / 48
position = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1)
sites = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]  # face-centred cubic
spheres = np.zeros((48, 48, 48), bool)
for site in sites:
    for shift in (0.125, -0.125):  # the two atoms of the diamond lattice
        d = position - (np.array(site) + shift)
        d -= np.rint(d)
        spheres |= (d**2).sum(axis=-1) < 0.25**2
diamond = mg.VoxelImage(spheres.astype(np.uint8), 1.0 / 48, periodic=True)
silicon = Material(permittivity=11.56)
light = band_structure(
    diamond, {0: air, 1: silicon}, bands=10, points=2, waves="electromagnetic", coarsen=4
)
print([(n, low / (2 * np.pi), high / (2 * np.pi)) for n, low, high in light.gaps()])
# one gap, above the band 8 of the cubic cell: about 0.40 to 0.44 c / a
```

The plane waves are those of a grid `coarsen` times coarser than the voxels (here 12³
for 48³ voxels), whose cells cut by an interface take an averaged permittivity tensor:
voxelize three or four times finer than the plane waves you can afford. The phases may
be anisotropic (a permittivity tensor) or magnetic (a permeability), and the same waves
are found in a crystal of rods for wave vectors that leave its plane.

### Flow through the pores, and the sound absorbed by a porous layer

A fluid that creeps through the pores obeys Darcy's law, `<v> = -(k / eta) grad p`:
`hydraulic_permeability` solves the Stokes flow in the pores (the void phases) by the
lattice Boltzmann method and gives the permeability tensor `k`, a squared length that
does not depend on the fluid. `acoustic` adds the other parameters of the equivalent
fluid of a rigid porous frame (tortuosity, viscous and thermal characteristic lengths,
thermal permeability), from which the model of Johnson, Champoux, Allard and Lafarge
gives the sound absorption of a layer. Here a bed of fibres, sound across them:

```python
from morphogen.homogenization import absorption
from morphogen.homogenization.acoustics import along_axis

n = 24
x = (np.arange(n) + 0.5) / n - 0.5
fibres = (x[:, None] ** 2 + x[None, :] ** 2 < 0.05 / np.pi).astype(np.uint8)[:, :, None]
mat = mg.VoxelImage(fibres, 4e-6, periodic=True)  # voxels of 4 µm: fibres of 24 µm, 5 %
sound = homogenize(mat, {0: VOID, 1: glass}, properties=["acoustic"])["acoustic.periodic"]
print(sound["permeability1"], sound["tortuosity1"], sound["viscous_length1"])  # m², -, m
layer = absorption([500.0, 1000.0, 2000.0, 4000.0], 0.04, **along_axis(sound, 0))
print(layer["absorption"])  # of 4 cm of the mat on a rigid wall, at normal incidence
room = absorption(
    [500.0, 1000.0, 2000.0, 4000.0],
    0.04,
    incidence="diffuse",  # every angle (or an angle in radians); the wave is refracted
    transverse=along_axis(sound, 2),  # the frame along the layer: that of the fibres
    **along_axis(sound, 0),
)
print(room["absorption"])  # in a diffuse field
```

The lengths are in the unit of the voxels: an image in metres gives SI values, as here.

### In time: loading paths

A loading path prescribes macroscopic strains or stresses in time; the components that
are not given are free (zero stress). The result holds the curves.

```python
path = {
    "times": [0.0, 0.01, 1.0],
    "increments": [4, 8],
    "strain": {"11": [0.0, 0.01, 0.01]},
}  # stretch along x, then hold
relax = homogenize(image, {0: resin, 1: glass}, properties=[], paths={"relax": path})[
    "relax.periodic"
]
print(relax.times, relax.stress[:, 0])  # the stress relaxes while the strain is held
```

## 6. Nonlinear behaviours

A behaviour is a field of `Material`; a material that has one follows it along loading
paths, with the elastic constants of the material as its (initial) elasticity.

| Field of `Material` | Behaviour |
|---|---|
| `plasticity` | von Mises or Hill elasto-(visco)plasticity, isotropic and kinematic hardening |
| `crystal_plasticity` | slip on the systems of a cubic or hexagonal lattice (polycrystals); `twins` that shear in one sense until the crystal has twinned, and back when the stress is reversed (at a share of their resistance); lattices that become those of their twins (`twin_reorientation`: point by point, or grain by grain); `cleavage` planes that open and soften (a crack band of one element, or a damage band of a given width: `regularization`, their microdamage), close under pressure and rub, and that the slip weakens (`coupling`) |
| `porous_plasticity` | Gurson–Tvergaard–Needleman, Rousselier: ductile damage by void growth |
| `viscoelasticity` | generalized Maxwell model (Prony series), WLF or Arrhenius shifts |
| `hyperelasticity` | neo-Hookean, Mooney–Rivlin and the polynomial of Rivlin, Yeoh, Arruda–Boyce and the eight chains, Gent, van der Waals, Fung, Ogden, the fibres of Holzapfel–Gasser–Ogden, the compressible foams of Blatz–Ko and Ogden–Hill, the softening of Mullins (finite strains); fitted to tests by `fit_hyperelasticity` |
| `damage` | isotropic damage of quasi-brittle solids, with a fracture energy and the stiffness of cracks that close (`split: spectral`); Mazars' damage of concrete (`law: mazars`); beside a `plasticity`, the damage of its effective stress: Lemaitre's ductile damage (`law: lemaitre`) or the creep damage of Kachanov and Rabotnov (`law: kachanov`) |
| `mfront`, `learned` | an MFront behaviour, a law learned from data |

All of them run at a material point (`simulate_point`), by finite elements, on voxels
(CPU and GPU), on level sets and in the bulk of the crack solvers; the
small- or finite-strain kinematics is an option of the path
(`nonlinear={"kinematics": "finite"}`).

```python
brittle = Material(
    young=30e9,
    poisson=0.2,
    damage={
        "strength": 3e6,  # tensile strength (Pa)
        "fracture_energy": 100.0,  # J/m², dissipated by a band one voxel wide
        "length": 6.0 / 16,  # the size of the voxels
        "viscosity": 0.2,  # delays the damage: keeps the softening problem well posed
    },
)
pull = {"times": [0.0, 1.0], "increments": 8, "strain": {"11": [0.0, 2.4e-4]}}
curve = homogenize(image, {0: brittle, 1: glass}, properties=[], paths={"pull": pull})[
    "pull.periodic"
]
print(curve.strength(), curve.work(), curve.plastic[-1])  # peak stress, work, mean damage
```

`strength()` is the largest stress of the loading and `work()` the work done per unit
volume: the two quantities of a softening response that the RVE statistics take (§9).

### Finite strains

With `nonlinear={"kinematics": "finite"}` (`kinematics="finite"` at a material point) a
path prescribes components of the displacement gradient and gives the nominal stress
(the first Piola–Kirchhoff stress); the result also holds the true stress (`cauchy`) and
the logarithmic strain (`log_strain`). The hyperelastic energies are written for these
kinematics; the elastic phases then follow the energy of Hencky, and plasticity runs in
the space of the logarithmic strain ([theory notes](theory.md), §33). A rubber alone,
then filled with the spheres of the sample, stretched by half its length:

```python
from morphogen.homogenization import simulate_point

rubber = Material(
    hyperelasticity={"model": "yeoh", "c10": 0.25e6, "c20": -0.02e6, "c30": 0.01e6, "bulk": 5e6}
)  # Pa: a shear modulus of 0.5 MPa at small strains
filler = Material(young=100e6, poisson=0.3)
stretch = {"times": [0.0, 1.0], "increments": 5, "strain": {"11": [0.0, 0.5]}}
alone = simulate_point(rubber, stretch, kinematics="finite")
print(alone.stress[-1, 0], alone.cauchy[-1, 0], alone.log_strain[-1, 0])  # nominal, true, ln 1.5
reinforced = homogenize(
    micro.voxelize(shape=12),
    {0: rubber, 1: filler},
    properties=[],
    paths={"stretch": stretch},
    nonlinear={"kinematics": "finite"},
)["stretch.periodic"]
print(reinforced.stress[-1, 0] / alone.stress[-1, 0])  # about 2: the spheres stiffen the rubber
```

On a mesh, a nearly incompressible rubber (a Poisson's ratio of 0.49 or more) takes
`integration="bbar"`: the volume change of each 10-node tetrahedron as its mean (`F`-bar),
without which the elements lock ([theory notes](theory.md), §77); the voxels do so by
default.

`fit_hyperelasticity` fits the energies to tension, compression and shear tests; the
tutorial [a rubber, from its tests to its tearing](tutorial_rubber.md) goes from such
tests to the softening of Mullins and to a filled rubber.

## 7. Fracture

| Method | Module | Use |
|---|---|---|
| stress intensity factors, crack growth (conforming meshes and XFEM cracks) | `morphogen.fracture` (`lefm`, `mixed`, `bulk`) | a crack in a structure or a cell, linear or inelastic bulk |
| cracks of any surface and front | `morphogen.fracture.surface` | elliptical, inclined, surface cracks: `K_I`, `K_II`, `K_III` along a curved front, each point of which advances and kinks on its own |
| cohesive zones | `morphogen.fracture.cohesive` | interfaces and prescribed crack paths |
| high-cycle fatigue | `morphogen.homogenization.fatigue` (`path_fatigue`; `homogenization.fatigue`) | where an elastic cell would start to crack under a cyclic loading, and its factor of safety: Crossland, Sines, Dang Van, Findley, Matake on the stresses of a cycle |
| interfaces that debond along loading paths | `morphogen.fracture.periodic` (`CohesiveCell`; `nonlinear: {interfaces: ...}`) | particles, fibres or grains that come apart in a periodic cell: the composite softens in tension and its lips carry compression again, with friction |
| ductile fracture | `Material(porous_plasticity=...)` | void growth and coalescence along loading paths |
| damage with a fracture energy | `Material(damage=...)` | diffuse cracking of quasi-brittle phases; concrete in tension and compression (`law: mazars`) |
| damage coupled to plasticity | `Material(plasticity=..., damage={"law": "lemaitre", ...})` | ductile damage and low-cycle fatigue of metals by the plastic strain; creep to rupture (`law: kachanov`) |
| contact, self-contact and friction | `morphogen.contact` | lips, pores and struts that close and rub; a sample between platens or under an indenter (`compression`, `indentation`); the walls of the pores of a periodic cell (`nonlinear={"contact": ...}`) |
| phase field | `homogenize(..., fracture=...)` | cracks that nucleate, branch and merge in a microstructure; with `nonlinear={"kinematics": "finite"}`, in rubbers and soft solids that stretch much before they break |

The cohesive zones and the cracks of conforming meshes take 10-node tetrahedra too:
`insert_interfaces` opens such a mesh with interface elements of six nodes a side (on
its curved faces, if its nodes lie on the interfaces of the phases), and
`conforming_work(..., order=2)` puts the mid-edge nodes of the crack front at the quarter
of their edges, where the fields vary as the square root of the distance to it (three
times fewer elements across a bending arm, a quarter of the error on the energy release
rate: [theory notes](theory.md), §46 and §47).

The phase-field method needs only the toughness `G_c` of the phases that break and a
length `l`, resolved by at least two voxels. It runs on the volume of an image as on a
section; here a planar section of the sample (thirty times fewer voxels than its
volume) is pulled until it breaks:

```python
section = micro.voxelize(shape=24).layer(axis=2)  # the middle layer normal to z
fracture = {"toughness": {0: 100.0}, "length": 0.5, "model": "AT1"}  # the spheres do not break
pull = {"times": [0.0, 1.0], "increments": 12, "strain": {"11": [0.0, 1.2e-4]}}
broken = homogenize(
    section,
    {0: Material(young=30e9, poisson=0.2), 1: glass},
    properties=[],
    paths={"pull": pull},
    fracture=fracture,
)["pull.periodic"]
print(broken.strength(), broken.fracture_energy[-1])  # Pa, J/m³ spent in the cracks
```

In a configuration, `voxels: {shape: 24, section: z}` studies the same section. In
finite strains the energy that breaks is that of the behaviour at the stretches reached
(`split: volumetric` spares the hydrostatic compression). `phase_field_fracture`, the
function behind `fracture=`, also takes a crack to start from. A notched strip of the
rubber of §6, stretched across its crack with its length held (the pure shear test),
tears when the energy of the strip far from the crack, times its height, reaches the
toughness (Rivlin and Thomas 1953): at a stretch of 1.5 here.

```python
from morphogen.fracture import phase_field_fracture

h = 1.0 / 8  # a strip of height 1 and length 2, in voxels of size h
sheet = mg.VoxelImage(np.zeros((16, 8, 1), dtype=np.uint8), h, periodic=True)
notch = np.zeros(sheet.shape, dtype=bool)
notch[:8, 4, 0] = True  # broken from the start, over half the length
tear = {"toughness": 1.3e5, "length": 2 * h, "model": "AT1", "split": "volumetric"}  # J/m²
held = {"11": [0.0, 0.0], "23": [0.0, 0.0], "13": [0.0, 0.0], "12": [0.0, 0.0]}
across = {
    "times": [0.0, 1.0],
    "increments": 8,
    "strain": {"22": [0.0, 0.64], **held},
    "stress": {"33": [0.0, 0.0]},
}
torn = phase_field_fracture(sheet, {0: rubber}, tear, across, initial=notch, kinematics="finite")
print(1.0 + torn.strain[:, 1])  # the stretch of each step
print(torn.stress[:, 1])  # the nominal stress: it falls to nothing between 1.48 and 1.56
```

The [tutorial on a rubber](tutorial_rubber.md) refines this test, and the templates
`fracture_composite` and `fracture_rubber` of the [gallery](gallery.md#fracture) break
cells of spheres and of holes from a configuration.

## 8. Dependence on the mesh, and how to control it

Two different things are called mesh dependence.

**Discretization error of smooth problems.** Voxels draw interfaces as stairs, which
costs a first-order error on the apparent properties; refine the image until the property
settles, or use a discretization that follows the interfaces: a conforming mesh, or the
extended finite elements, which cut a regular grid by the level set of the morphology
(plane interfaces are then exact, curved ones second order):

```python
from morphogen.homogenization import LevelSetSample

sample = LevelSetSample.of(micro, 12)  # a grid of 12³ cells cut by the spheres
cut = homogenize(sample, {0: matrix, 1: sphere}, properties=["bulk_modulus"])
print(cut["bulk_modulus.periodic"].value, bulk)  # against the 16³ voxels of §4
```

**Pathological dependence of softening problems.** A local softening law dissipates less
and less as the elements shrink. morphoGen offers four remedies: the *crack band*
(`damage.fracture_energy` with `damage.length`: a band of one element dissipates the
fracture energy, whatever its size), a *viscous regularization* (`damage.viscosity`), a
*length of the model itself* (the phase field of §7; the micromorphic damage of
`morphogen.micromorphic`) and *discrete cracks* (cohesive zones and XFEM cracks of
`morphogen.fracture`). The dissipated energy is then independent of the discretization;
check it by running two resolutions.

## 9. The size of the representative volume element

Every result above is the apparent property of one finite realization. The statistical
method of Kanit et al. (2003) runs realizations at several sizes, fits the decay of the
variance of each property with the volume (its integral range) and gives the number of
realizations, or the size, for a wanted relative error. It applies to *every* quantity:
name them in `rve.properties`. The tutorial [how large is a representative
volume?](tutorial_rve.md) goes through the method on spheres and on fibres, whose
correlations of long range change the answer.

```yaml
model: {type: boolean_spheres, radius: 1.0, volume_fraction: 0.2}
domain: {periodic: true}
seed: 1
voxels: {voxel_size: 0.5}
homogenization:
  method: fft
  materials:
    0: {young: 3.0e+9, poisson: 0.35, conductivity: 0.2, swelling: 0.4,
        viscoelasticity: {shear: [[0.5, 0.01], [0.3, 1.0]]}}
    1: {young: 70.0e+9, poisson: 0.22, conductivity: 30.0}
  properties: [bulk_modulus, conductivity, swelling, dynamic_stiffness]
  frequencies: [10.0]
  paths:
    pull: {times: [0, 1], increments: 4, strain: {"11": [0, 0.01]}}
rve:
  sizes: [4.0, 6.0, 8.0]
  realizations: [16, 8, 4]
  properties:
    - volume_fraction
    - bulk_modulus                      # a scalar property
    - conductivity
    - swelling.mean                     # a named result of a tensor
    - dynamic_stiffness.shear_loss_1    # ... of a harmonic property
    - pull.S11                          # the stress at the end of a path
    - pull.strength                     # its largest stress, pull.work: its work
outputs: {directory: out/rve, name: spheres}
```

```bash
morphogen run rve.yaml          # out/rve: results.csv, rve_report.md/.json/.tex, plots
```

The names follow one rule: a measure of the morphology; a scalar property (all its
boundary conditions, or `<property>.<bc>`); `stiffness` and the tensors of the transport
phenomena (their mean tensor per size); `<property>.<result>` for the named results of
the couplings, of the harmonic properties, of the poroelastic law, of the flow and of
the tensors of expansion (`piezoelectric.e33`, `poroelastic.biot_coefficient`,
`hydraulic_permeability.mean`, `acoustic.tortuosity`, `thermal_expansion.A11`); `<path>.<quantity>[@<time>]` for a column of the curves of a
loading path, its `strength` or its `work` (and `fracture_energy`, `crack_surface` with a
phase field).

## 10. From the command line

```bash
morphogen homogenize image.vti --materials materials.yaml --property stiffness conductivity
morphogen homogenize image.vti --materials materials.yaml --property piezoelectric
morphogen homogenize image.vti --materials materials.yaml --property dynamic_stiffness --frequencies 0.1 10 1000
morphogen homogenize image.vti --materials materials.yaml --property poroelastic --fluid-bulk-modulus 2.2e9
morphogen homogenize image.vti --materials materials.yaml --property hydraulic_permeability acoustic
morphogen bands cell.vti --material 0=permittivity:1 1=permittivity:8.9 --waves tm --path G X M G --figure bands.png
morphogen bands crystal.vti --material 0=permittivity:1 1=permittivity:11.56 --waves electromagnetic --coarsen 4 --figure light.png
```

`materials.yaml` maps the phases to the same keys as `Material`
(`{0: {young: 3.0e+9, poisson: 0.35}, 1: void}`). In a configuration file,
`homogenization.method` is `fft`, `fe` or `xfem`. Seven templates are complete
configurations to start from (`morphogen template NAME -o NAME.yaml`, then
`morphogen run NAME.yaml`; the [gallery](gallery.md) shows them):
`homogenization_transport`, `homogenization_piezoelectric`,
`homogenization_viscoelastic`, `homogenization_poroelastic`, `homogenization_xfem`,
`homogenization_acoustic_foam` and `rve_multiphysics`.

In the graphical interface (`morphogen gui`), the tab **Physics** of the menu *Simulate*
runs four of these on a morphology of the workspace: its transport properties, the flow
through its pores with the sound absorption of a layer (at normal or oblique incidence,
or in a diffuse field), the dynamic moduli of a viscoelastic composite, and the band
structure of light or of sound in a periodic morphology.

## 11. What runs where

| Problem | Voxels (FFT) | Meshes (FE) | Level sets (XFEM) |
|---|:-:|:-:|:-:|
| elasticity, transport, thermal expansion, swelling, poroelasticity | yes | yes | yes |
| strong couplings, harmonic regimes | yes | – | – |
| flow through the pores, acoustic parameters | yes | – | – |
| Bloch waves: elastic, acoustic, planar light | cut into tetrahedra | yes | – |
| Bloch waves of light in three dimensions | yes (plane waves) | – | – |
| loading paths with the behaviours of §6, in small or finite strains | yes (CPU, GPU) | yes | yes |
| phase field of fracture, in small or finite strains | yes | – | – |
| cracks (planar, kinked, of any surface and front), cohesive zones | – | yes | – |
| RVE statistics | yes | yes | yes |

Limits to keep in mind: the homogenized properties assume a separation of scales (the
harmonic properties are quasi-static: wavelengths much longer than the microstructure;
use the Bloch waves otherwise); the electromagnetic waves of three-dimensional crystals
are solved on voxel images, with lossless phases;
the strong couplings, the harmonic regimes, the flow through the pores and the phase
field are periodic problems of voxel images (cubic voxels for the flow, whose curved
walls are stairs: refine until the permeability settles); the extended finite elements use linear tetrahedra, which lock in
isochoric plastic flow (prefer voxels or quadratic meshes for plasticity).

## 12. Adding a physics or a behaviour

* **A transport phenomenon** is one entry of the table `TRANSPORT` of
  `morphogen/homogenization/materials.py` (the name of the physics and of its coefficient)
  and one field of `Material`: the solvers, the properties, the bounds and the RVE
  studies follow.
* **A strong coupling** is one entry of `COUPLINGS` of
  `morphogen/homogenization/coupled.py` (the fields of its layout) with the assembly of
  its law in `coupled_law` and its constants in `split_law`; the solver takes any
  symmetric law whose diagonal blocks are definite.
* **A behaviour in closed form** is a class of `morphogen/homogenization/laws.py` with one
  method `step(strain, strain_before, state_before, dt) -> (stress, tangent, state)` for
  all the points of a phase at once, returned by `analytic_model`; material points,
  finite elements, voxels and cracks then integrate it without another change. A
  behaviour written in MFront needs no code at all (`Material(mfront=...)`).

Each contribution comes with tests against exact solutions (laminates, uniform fields,
closed forms) and a section of the [theory notes](theory.md): see
[Development](development.md).
