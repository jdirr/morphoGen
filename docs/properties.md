# Apparent properties

The apparent properties of a volume element are its responses to macroscopic loadings
under boundary conditions: kinematic (KUBC), mixed (MBC), periodic (PBC) and static
(SUBC). morphoGen computes them by finite elements on conforming meshes, by an FFT solver
on voxel images, or by extended finite elements on a grid cut by the interfaces, for
elasticity and for the other physics of heterogeneous materials. The
[guide to the physics](physics.md) goes through each problem, its solver and the name of
its property; this page shows the main uses.

## Solvers

* Finite elements (4-node tetrahedra, or 10-node isoparametric ones whose curved edges
  follow the interfaces, and the sharp edges where spheres, fibres or their coatings
  overlap: the elements of the analyses of a configuration, by default):
  KUBC, mixed (MBC), periodic (PBC) and
  static (SUBC) conditions; bulk and shear moduli, conductivity, stiffness and
  conductivity tensors; energy estimates, per-phase field statistics, isotropy
  criteria, local fields for ParaView. Algebraic multigrid solvers (no compiled
  dependency), about 10⁵–10⁶ unknowns on a laptop.
* FFT solver of voxel images (periodic), for tessellations and any image, linear or
  nonlinear: loading paths are solved by Newton–Krylov iterations (conjugate gradients
  preconditioned in Fourier space, the tangent applied without assembly) on voxels
  that do not lock under plastic flow (B-bar, F-bar). With CuPy, the FFT solvers run on
  an NVIDIA GPU (`device: gpu`, `--device gpu`), the loading paths with their
  integration points: on a laptop, 5 to 7 times faster for linear problems, 1.6 to 2.8
  times along crystal plasticity paths and 2.4 to 3.5 along porous ones.
* Extended finite elements (`method: xfem`, `LevelSetSample`): a regular grid cut by
  the interfaces of the morphology, without a mesher and without the stairs of the
  voxels; elasticity, transport and loading paths.
* Finite elements of the plane (3- and 6-node triangles, `TriMesh`): the structures
  invariant across their plane — the columnar grains of seeds of the plane, the section
  of long fibres — in generalized plane strain (the stiffness of the structure in space)
  or plane strain, and in conduction; KUBC, periodic and static (SUBC) conditions on the
  sides of the rectangle.

Which one to choose is the subject of a [recipe](howto.md) and of the
[guide to the physics](physics.md#8-dependence-on-the-mesh-and-how-to-control-it).

## Elasticity and conduction

```python
import morphogen as mg
from morphogen.homogenization import VOID, Material, homogenize

micro = mg.models.BooleanSpheres(radius=1.0, volume_fraction=0.3).generate(
    mg.Domain.cube(20.0, periodic=True), seed=42
)
image = micro.voxelize(shape=128)

mesh = mg.mesh.conforming_mesh(micro, size=0.5)  # tetrahedra following the spheres
materials = {
    0: Material(young=1.0, poisson=0.3, conductivity=1.0),
    1: Material(young=20.0, poisson=0.3, conductivity=10.0),
}
results = homogenize(
    mesh,
    materials,
    bcs=["kubc", "mixed", "periodic"],
    properties=["bulk_modulus", "shear_modulus", "conductivity"],
)
print(results["bulk_modulus.kubc"].value)

# porous media: mesh only the solid (phases="solid"), and give the pores a VOID
# voxel images (e.g. tessellations) are solved by FFT, under periodic conditions:
results = homogenize(image, materials, properties=["stiffness"])
stiffness = results["stiffness.periodic"]
print(stiffness["tensor"])  # C: Voigt 11 22 33 23 31 12, engineering shears
print(stiffness["compliance"])  # S = C^-1 in the same notation: S44 = 1/G23, S66 = 1/G12
print(stiffness["E1"], stiffness["nu12"], stiffness["G23"])  # its engineering constants
```

Every output of a stiffness gives its compliance beside it, in the same notation:
`results.csv` has the columns `stiffness.<bc>.C11` ... `C66` and `stiffness.<bc>.S11`
... `S66` (upper triangles) and `run.json` states their notation; `morphogen homogenize`,
`frame-stiffness`, `amitex read` and `optimize` print both (their JSON has `compliance`
and `notation` beside the stiffness), and the tools of the interface show both as tables
with their .csv. TFEL (MFront) and Abaqus order their shears 12, 13, 23: morphoGen's
`C44` (plane 23) is their `C66`.

## Polycrystals

Crystals have a cubic, hexagonal or any stiffness and conductivity, turned by the
orientation of each grain, in both solvers; the bulk modulus of cubic polycrystals comes
out exact, and the shear modulus within the Hashin–Shtrikman bounds.

```python
from morphogen.homogenization import Material, homogenize

# a copper polycrystal: 50 grains with random orientations, meshed grain by grain
grains = mg.models.Voronoi(n_cells=50, crystal_orientations=True).generate(
    mg.Domain.cube(1.0, periodic=True), seed=1
)
mesh = mg.mesh.conforming_mesh(grains, size=0.08, min_feature=0.02)
copper = Material.cubic(168.4, 121.4, 75.4)
results = homogenize(
    mesh,
    {1: copper},
    bcs=["kubc", "periodic", "subc"],
    properties=["bulk_modulus", "shear_modulus"],
)
print(results["bulk_modulus.subc"].value)  # (C11 + 2 C12) / 3 = 137.07, exactly

mg.io.save_mesh(mesh, "grains.inp")  # element sets and orientations per grain
mg.io.write_periodic_equations("grains_equations.inp", mesh)
mg.io.save_mesh(mesh, "grains.xdmf")  # FEniCSx, with phase, grain and face tags
mg.io.write_damask(
    "grains", grains.labels(shape=64), grains.orientations, materials={1: copper}
)  # DAMASK grid and material.yaml

# combinations: alumina inclusions and an amorphous boundary phase in the polycrystal
model = mg.models.Combination(
    operation="inclusions",
    models=(
        mg.models.Voronoi(n_cells=50, crystal_orientations=True, boundary_thickness=0.01),
        mg.models.HardcoreSpheres(radius=0.05, volume_fraction=0.05),
    ),
)
print(model.active_phases())  # {1: 'cells', 0: 'grain boundaries', 2: 'inclusions ...'}
```

Features smaller than `min_feature` (short cell edges, thin caps of cells on the faces
of the box) are removed before meshing; `min_feature=0` meshes the exact cells. Without
MMG, or for voxel workflows, the same polycrystals are homogenized by FFT on their
images with the grain labels (`homogenize(image, materials, grains=labels,
orientations=grains.orientations)`).

## Structures of the plane

A structure that does not vary along z — the columnar grains of a tessellation whose germs
lie in one plane, the section of long fibres along z, a layer whose fields do not vary
through its thickness — is meshed in its plane and solved there ([theory](theory.md) §119):

```python
from morphogen.homogenization import Material, homogenize
from morphogen.models.seeded import SeededLaguerre

# grains grown from elliptical seeds, columnar through the box: their exact polygons
sheet = SeededLaguerre(dimension=2, aspect_ratio=2.0, crystal_orientations=True).generate(
    mg.Domain(size=(3.0, 3.0, 0.1), periodic=True), seed=3
)
mesh = mg.mesh.plane_mesh(sheet, size=0.1, order=2)  # triangles, matching sides
results = homogenize(
    mesh,
    {1: Material.cubic(168.4, 121.4, 75.4)},
    bcs=["kubc", "periodic", "subc"],
    properties=["stiffness"],
    plane="generalized",  # the anti-plane fluctuation free; "strain": plane strain
)
print(results["stiffness.periodic"]["tensor"])  # the 6 x 6 stiffness of the sheet in space

mg.io.save_mesh(mesh, "sheet.inp")  # CPE6 elements, the sets of the grains and sides
section = mg.mesh.plane_mesh(micro, size=0.5, z=10.0)  # the section of the spheres above
```

The mesh in space of the same grains (`conforming_mesh`) is their exact polygons extruded
through the box (`TriMesh.extruded`, three tetrahedra per prism): its periodic problems are
those of the plane. The loading paths of a mesh of the plane (plasticity, crystal plasticity)
are followed on its layer in space (`TriMesh.layer`), periodic across it, under periodic
conditions. In a configuration, `mesh: {plane: auto}` (the default) meshes the
columnar grains of seeds of the plane in their plane, `plane: true` the section of any
structure at `mesh.z`, and `homogenization.plane` chooses the hypothesis;
`morphogen homogenize` takes `--plane` and `--z`.

## Macroscopic loadings and thermal expansion

Loadings prescribe any mix of macroscopic strain and stress components (uniaxial stress,
plane strain...): the periodic element solves them exactly by condensation of the
macroscopic unknowns, for every boundary condition and both solvers. The phases expand
with the temperature, which gives the thermal stresses and the apparent thermal
expansion tensor (Levin's relation and bounds).

```python
from morphogen.homogenization import Loading, homogenize

materials = {
    0: Material(young=2800.0, poisson=0.39, expansion=95e-6),  # epoxy
    1: Material(young=89000.0, poisson=0.3, expansion=5.5e-6),  # glass
}
results = homogenize(
    image,
    materials,
    properties=["stiffness", "thermal_expansion"],
    loadings={
        "tension": {"stress": {"33": 100.0}},  # the rest free
        "plane": Loading.parse("E11=0.001 E33=0"),
        "heating": {"temperature": 100.0},
    },
)
print(results["thermal_expansion.periodic"]["tensor"])
print(results["tension.periodic"].components())  # E11 ... E12, S11 ... S12, energy
```

## Other physics

One homogenization module covers the mechanical, thermal, electrical, electromagnetic,
chemical, diffusive, optical and acoustic problems of heterogeneous materials, static and
dynamic, with their couplings:

* Transport of a scalar potential: thermal and electrical conduction, dielectrics,
  magnetostatics and diffusion share the solvers, the bounds and the RVE studies
  (`conductivity`, `electrical_conductivity`, `permittivity`, `permeability`,
  `diffusivity`, scalars and tensors; isotropic or crystalline phases, inert phases).
* Weak couplings: thermal expansion, hygroscopic and chemical swelling (`swelling`),
  the poroelasticity of Biot (`poroelastic`: drained stiffness, tensor and modulus of
  Biot of any solid, undrained response of Gassmann and Skempton); transient diffusion,
  oxidation and ageing (`morphogen.coupling`).
* Strong couplings, their fields solved together on voxel images: piezoelectric,
  piezomagnetic, magneto-electro-elastic (product properties), magnetoelectric and
  thermoelectric composites, with the constants of their effective laws (`d33`, `k_t`,
  `ZT`...).
* Harmonic regimes: storage and loss moduli of viscoelastic composites
  (`dynamic_stiffness`), complex conductivity and permittivity of leaky dielectrics
  (`complex_conductivity`: Maxwell–Wagner polarization), effective complex refractive
  index (`refractive_index`).
* Flow and sound in porous media: the permeability of Darcy's law by a Stokes flow
  through the pores (`hydraulic_permeability`, a lattice Boltzmann method on the
  voxels, or the elements of Taylor and Hood on the triangles of the meshes of the plane),
  the parameters of the equivalent fluid of a rigid frame (`acoustic`:
  tortuosity, viscous and thermal characteristic lengths, thermal permeability) and the
  sound absorption of a layer (`homogenization.absorption`, the model of Johnson,
  Champoux, Allard and Lafarge).

Every transport phenomenon of a scalar potential has its coefficient and its properties;
couplings and harmonic regimes are properties too:

```python
import dataclasses

phases = {
    0: Material(conductivity=0.2, electrical_conductivity=1e-12, permittivity=3.0),
    1: Material(conductivity=30.0, electrical_conductivity=1e4, permittivity=10.0),
}
results = homogenize(
    image, phases, properties=["conductivity", "electrical_conductivity", "permittivity_tensor"]
)

# a piezoelectric composite: the fields solved together, the constants of the law
ceramic = dataclasses.replace(
    Material.hexagonal(166e9, 77e9, 78e9, 162e9, 43e9),
    permittivity=(11.2e-9, 11.2e-9, 12.6e-9),
    coupling={"piezoelectric": {"e31": -4.4, "e33": 18.6, "e15": 11.6}},
)
polymer = Material(young=4e9, poisson=0.35, permittivity=3.7e-11)
law = homogenize(image, {0: polymer, 1: ceramic}, properties=["piezoelectric"])
print(law["piezoelectric.periodic"]["d33"], law["piezoelectric.periodic"]["k_t"])

# storage and loss moduli of a viscoelastic matrix at three angular frequencies
resin = Material(
    young=3e9,
    poisson=0.35,
    viscoelasticity={"shear": [[0.5, 1e-2], [0.3, 1.0]], "bulk": [[0.2, 0.1]]},
)
dma = homogenize(
    image,
    {0: resin, 1: Material(young=70e9, poisson=0.22)},
    properties=["dynamic_stiffness"],
    frequencies=[0.1, 10.0, 1000.0],
)
print(dma["dynamic_stiffness.periodic"]["shear_tan_delta_2"])

# a porous rock: Biot's tensor and modulus, Skempton's coefficient with water in the pores
rock = homogenize(
    image,
    {0: Material(young=20e9, poisson=0.2), 1: VOID},
    properties=["poroelastic"],
    fluid_bulk_modulus=2.2e9,
)
print(rock["poroelastic.periodic"]["biot_coefficient"])
```

Transport also runs in time (`morphogen.coupling`): transient diffusion in
microstructures (moisture, oxygen, solvents), thermal oxidation of polymers (closed-loop
scheme, diffusion-limited profiles of a plate, and in the volume of a microstructure
around its fibres or fillers), properties that evolve with an ageing tracer mapped to
graded phases, hydrolysis of cellulose, hygromorph multilayers bending as moisture
diffuses (orthotropic layers, their properties homogenized from microstructures);
diffusion from configurations (`diffusion` section); with FEniCSx, the diffusion and free
swelling of meshed samples (hygromorph strips bending in time) and the stresses of
oxidized ones.

## Lattices and fibre networks as frames of beams

Random periodic frameworks (Delaunay fibre networks, Voronoi open foams, random
lattices) have the effective stiffness of frames of Euler-Bernoulli beams of circular
sections with rigid joints, or of exact Timoshenko beams (`shear_coefficient`: a
coefficient per beam, or Cowper's), by a sparse periodic problem: the Voigt stiffness,
its split into stretching, bending, torsion and shear, the affine estimate and the
non-affinity of the deformation (slender Delaunay networks at 0.82 of the affine
estimate, open foams at E = 1.27 E_s phi^2).

The softening of networks of non-identical beams of Ban et al. (2016) is exact to second
order in the coefficient of variation and against replicas; with the axial stiffness
lowered, Timoshenko beams whose shear stiffness follows the area of their general
sections reproduce their Voronoi case (s = 0.46) and come within a factor 1.4 of their
Delaunay case. Against the FFT solver on the voxels of the same realizations, the frames
are 2 to 30 % softer than the continuum, the struts merging in its joints; Timoshenko
beams rigid over a third of their radius at the joints (`rigid_joints`) follow it within
4 % for open foams from 6 to 15 % (7 % at 2 %) and a Delaunay network at 12 %.

The frames are computed by `homogenization.networks`, in the summaries of the network
models with `frame_stiffness: true`, and for any realization of them, with every option,
by `morphogen frame-stiffness` and the tool Stiffness of the frame of beams of the
interface. Both run Ban's replicas of the stiffness of the beams (normal, lognormal or
bimodal, several coefficients of variation) against the exact `1 - s CV^2` and
`v CV^2`, with their table and figure; the axial force and the energy of each beam
under a uniaxial stress (`beam_forces`) draw the force chains and the beams that bend
(`viz.figures.frame_figure`, in the tool).

## Waves in periodic media

Band structures of the elastic waves of periodic cells (meshes or voxel images) along the
edges of the Brillouin zone, their band gaps and figures
(`homogenization.band_structure`); the scalar waves of sonic and photonic crystals
(`waves="acoustic"`, `"shear"`, `"tm"`, `"te"`: the pressure of fluids, antiplane
shear, the two polarizations of light in planar crystals), rigid scatterers, and the
speeds of the homogenized medium at long wavelengths (Wood's law); light in
three-dimensional photonic crystals (`waves="electromagnetic"`: the vector waves of
Maxwell's equations by plane waves on the voxels, dielectric, anisotropic or magnetic
phases, interfaces averaged within the cells; the band gap of the diamond lattice).

## Post-processing and bounds

* Post-processing: the compliance S = C⁻¹ beside every stiffness C, in tables, JSON
  and the text of the commands (morphoGen's Voigt order 11, 22, 33, 23, 31, 12 with
  engineering shears, stated with each, TFEL and Abaqus ordering their shears 12, 13,
  23), engineering constants, anisotropy indices (universal index, Zener
  ratio, distance to isotropy), macroscopic values of porous media
  (Σ = φ⟨σ⟩_solid), relative, normalized and specific properties.
* Voigt, Reuss and Hashin–Shtrikman bounds, also for polycrystals of cubic crystals.

## In a configuration

A `mesh` section describes the conforming meshes, and a `homogenization` section the
materials, boundary conditions and properties:

```yaml
mesh:
  size: 0.5                 # largest element edge; the interfaces are refined
  order: 2                  # 10-node tetrahedra (the default of an analysis); 1: 4-node
homogenization:
  materials:
    0: void                 # pores (not meshed)
    1: {young: 210000.0, poisson: 0.3, conductivity: 100.0}
  boundary_conditions: [kubc, mixed]
  properties: [bulk_modulus, shear_modulus, conductivity]
```

`method: auto`, the default, takes these finite elements on conforming meshes, meshed
from the exact geometry wherever the model knows it (when the configuration has no `mesh`
section, a light one: 10 elements along the shortest edge of the domain, the interfaces
within an eighth of an element, its level set sampled at the voxels of a `voxels` section
if any); the voxels are the last resort, for what only the FFT solver does (the GPU, the
coupled, harmonic and flow properties, the phase-field fracture, the reduced models of the
NTFA, the window method of RVE studies, planar sections of voxels) or without the mesher
MMG. `method: fft` (with a `voxels` section and `boundary_conditions: [periodic]`) asks
for the FFT solver, on an NVIDIA GPU with `device: gpu`. Crystals are given as
`{cubic: [C11, C12, C44]}`, `{hexagonal: [C11, C12, C13, C33, C44]}` or
`{stiffness: <6 x 6>}`; tessellations with `crystal_orientations: true` are meshed grain
by grain (`mesh: {size: 0.08, min_feature: 0.02}`) and their crystals turned by the
orientations of the grains. Aligned fibres in a matrix — the tows of woven textiles, the
plies of laminates — are given by their constituents, `{unidirectional: {fibre: {...},
matrix: {...}, fibre_fraction: 0.6}}`: the transversely isotropic material of the
composite cylinder assemblage (Hashin & Rosen 1964), its axis 3 along the fibres, with its
conductivities and expansions (`morphogen.homogenization.unidirectional`, theory notes
§110).

Templates to start from: `homogenization_spheres` (the four conditions on a composite),
`homogenization_gyroid` (stiffness tensor of a lattice, local fields),
`homogenization_voronoi` (FFT), `homogenization_polycrystal` (copper, finite elements on
grains), `homogenization_polycrystal_inclusions` (crystals, grain boundaries and
inclusions, FFT), `homogenization_mixed_loading` (thermo-elastic loadings mixing strains
and stresses). The other physics: `homogenization_transport` (thermal and electrical
conductivities, permittivity and diffusivity on the same voxels),
`homogenization_piezoelectric` (a strong coupling), `homogenization_viscoelastic`
(storage and loss moduli at `frequencies`, relaxation in time),
`homogenization_poroelastic` (Biot's law, `fluid_bulk_modulus`), `homogenization_xfem`
(`method: xfem`: the level set of the model cuts the grid of the `voxels` section) and
`homogenization_acoustic_foam` (Darcy's permeability and the acoustic parameters of an
open-cell foam). The templates of the loading paths are in
[Behaviours along loading paths](behaviours.md), those of the RVE studies in
[Size of the representative volume](rve.md), and the [gallery](gallery.md) shows every
template with its image.

From the command line, for one sample (a geometry `.json`, a mesh `.vtu`/`.mesh`, or an
image for the FFT solver):

```bash
morphogen homogenize spheres.json --size 0.5 --material 0=void 1=young:210e3,poisson:0.3 \
    --bc kubc mixed --property bulk_modulus shear_modulus --fields fields/
```

## Database of results and Ashby charts

Every run can be recorded in a database (one SQLite file): its configuration and the
results of each realization, filtered by conditions on any key (`model.type == gyroid`,
`solid_fraction < 0.3`). Normalized Ashby charts (relative property against relative
density, scaling laws of cellular solids) come from Python, `morphogen ashby` and the
graphical interface.

```bash
morphogen run gyroid.yaml --db study.db          # or outputs.database in the file
morphogen db list --db study.db
morphogen db export --db study.db --where "model.type == tpms" -o gyroids.csv
morphogen ashby --db study.db -o ashby.svg       # relative modulus vs relative density
```
