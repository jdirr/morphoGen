# Glossary

The words of this documentation that have a precise meaning here, each in a few lines
with where to read more. The [theory notes](theory.md) give the formulas, the
[guide to the physics](physics.md) the problems and their solvers.

```{glossary}
:sorted:

apparent property
  The property that a finite volume shows under given {term}`boundary conditions`: what
  `homogenize` computes. It depends on the volume, on the realization and on the
  conditions, and tends to the {term}`effective property` as the volume grows
  ([theory notes](theory.md), §10).

effective property
  The property of the homogeneous medium that stands for the heterogeneous one at a
  larger scale: the limit of the apparent properties of growing volumes, whatever their
  boundary conditions.

homogenization
  Computing the properties of that equivalent homogeneous medium from the properties of
  the phases and their arrangement, by solving a problem on a volume of the
  microstructure.

representative volume element
  A volume large enough for the mean of an apparent property over `n` realizations to
  lie within a relative error `ε` of the effective one (RVE; Kanit et al. 2003). It is
  not one size: it depends on the property, on the contrast of the phases, on `ε` and
  on `n` — many small volumes can do the work of a large one, as long as they are large
  enough for the boundary conditions not to bias them
  ([guide to the physics](physics.md), §9).

integral range
  The volume `A₃` over which a random medium is correlated: the variance of the mean of
  a property over a much larger volume `V` is the {term}`point variance` times
  `A₃ / V`, as if the volume held `V / A₃` independent pieces
  ([theory notes](theory.md), §9).

point variance
  The variance of a property from point to point: `p (1 − p) (Z₁ − Z₂)²` for two phases
  of values `Z₁` and `Z₂`, the first of volume fraction `p`.

scaling law
  `D²(V) = K V^-γ`, the variance of an apparent property against the volume, fitted to
  realizations of several volumes. `γ = 1` when the integral range is finite; it is
  smaller for correlations of long range (2/3 for fibres on Poisson lines), and the
  representative volume then grows much faster with the precision asked.

boundary conditions
  Of the problem solved on a volume: kinematic uniform (`kubc`: the displacements of
  the macroscopic strain on the boundary), static uniform (`subc`: the tractions of the
  macroscopic stress), `mixed`, and `periodic`. On the same volume the apparent
  stiffness under `subc` is below the periodic one, itself below that under `kubc`;
  the three meet as the volume grows ([theory notes](theory.md), §10).

contrast
  The ratio of a property between two phases. The larger it is, the more iterations the
  {term}`FFT solver` takes and the larger the representative volume.

realization
  One draw of a random model in a {term}`domain`, fixed by its {term}`seed`: an explicit
  description of the geometry (centres and radii, axes, cells), not an image.

seed
  The integer that fixes the random draws of a realization. The realization `i` of a
  run takes `derive_seed(seed, i)` from the seed of its configuration.

domain
  The box of a realization: its sizes, its origin, and whether it is {term}`periodic`.

periodic
  A periodic realization wraps its grains across the faces of its box, which then tiles
  space: the cell of an infinite medium, as the periodic boundary conditions assume.

phase
  A constituent of a microstructure, an integer of its images and meshes: 0 for what
  surrounds (the matrix, the pores of a lattice), 1 for what the model places, further
  integers for coatings, inclusions and the like.

voxel image
  A regular grid of cells, each holding the integer of its phase (`VoxelImage`, from
  `micro.voxelize`): the sample of the FFT solver, with interfaces drawn as stairs.

implicit function
  A function negative inside a phase, zero on its boundary and positive outside — a
  level set. Every phase of every model has one, exact where the model puts its
  interfaces; sections, surfaces and conforming meshes are cut from it
  ([theory notes](theory.md), §75).

conforming mesh
  A mesh of tetrahedra whose faces follow the interfaces between the phases, with
  matching nodes on the opposite faces of a periodic box (`mg.mesh.conforming_mesh`,
  with the mesher MMG): the sample of the finite elements, of 4-node elements or of
  {term}`quadratic elements <quadratic element>`.

quadratic element
  A tetrahedron of ten nodes, its corners and the middles of its edges, whose
  displacements are quadratic: the element of the finite element analyses (the 4-node
  one is too stiff, and locks under plastic flow). Its edges may be curved — an
  isoparametric element —, and the nodes of the interfaces are put on the implicit
  surfaces of the phases, so that the mesh keeps their volumes
  ([theory notes](theory.md), §76).

FFT solver
  The solver of voxel images under periodic conditions: conjugate gradients on the
  finite elements of the voxels, preconditioned in Fourier space by a homogeneous
  reference medium. It needs no mesh, runs on the CPU or a GPU, and its iterations grow
  with the contrast, not with the number of voxels ([theory notes](theory.md), §12).

extended finite elements
  Finite elements on a regular grid that the interfaces cut (XFEM): the cut elements
  are enriched so that the strain can jump across the interface, which a level set
  places. They follow the interfaces without a mesher (`LevelSetSample`;
  [theory notes](theory.md), §63); the same idea carries cracks through a mesh.

Voigt notation
  Stresses and strains as six components in the order 11, 22, 33, 23, 31, 12, the
  strains with engineering shears (`γ₂₃ = 2 ε₂₃`); a stiffness is then a 6 × 6 matrix.

loading path
  Components of the macroscopic strain or stress prescribed in time, in increments; the
  components that are not prescribed are free of stress (`LoadingPath`).

material point
  A point of a homogeneous material, without a microstructure: `simulate_point` follows
  a loading path there, to try a behaviour or to compare a composite with its matrix.

nominal stress
  The force per unit of initial area (the first Piola–Kirchhoff stress): what a path in
  finite strains gives, with the true stress (Cauchy: per unit of current area) and the
  logarithmic strain beside it.

hyperelasticity
  A behaviour that derives from a strain energy, for the large reversible strains of
  rubbers, soft tissues and foams (neo-Hookean, Mooney–Rivlin, Yeoh, Arruda–Boyce,
  Ogden...; [theory notes](theory.md), §61).

toughness
  `G_c`, the energy that a unit area of crack costs (J/m²): the critical energy release
  rate, the only fracture property that the {term}`phase field` asks for.

phase field
  Of fracture: a field `d` from 0 (sound) to 1 (broken) that spreads the cracks over a
  length `l`. The cracks start, turn, branch and merge by the minimization of an energy,
  without being meshed or tracked ([theory notes](theory.md), §64). Its two usual crack
  densities are AT2, whose damage grows from the first load, and AT1, which has an
  elastic domain and no damage away from the cracks.

cohesive zone
  A surface that opens under a traction which falls as it opens, and dissipates a
  fracture energy: an interface that debonds, or a crack along a path known in advance.

covariance
  `C(h)`, the probability that two points a vector `h` apart both lie in a phase. `C(0)`
  is the volume fraction, and `C` falls to its square at distances beyond which the
  medium is no longer correlated.

specific surface
  The area of the interface between two phases per unit volume.

percolation threshold
  The volume fraction above which a phase forms a cluster that spans the medium: from
  there a conducting phase in an insulator conducts across it
  ([theory notes](theory.md), §55).

tortuosity
  How much longer the shortest paths through a phase are than straight lines (the
  geometric tortuosity), or how much the phase slows a diffusion through it (from the
  effective diffusivity).

Boolean model
  Grains placed at the points of a Poisson process, free to overlap; their union is the
  phase 1. Its volume fraction, covariance and integral range are closed forms
  ([theory notes](theory.md), §1).

tessellation
  A partition of space into cells — those of Voronoi, nearest to their germs, or of
  Laguerre, with weights: the grains of a polycrystal, the cells of a foam.

configuration
  A YAML file that describes a run: the model, the domain, the voxels or the mesh, the
  outputs, the measures, the homogenization, an RVE study (`morphogen run`; the keys are
  in the [configuration reference](configuration.md)).

template
  An example configuration that comes with morphoGen (`morphogen template`): the
  [gallery](gallery.md) shows them all.

workspace
  In the graphical interface, the list of the morphologies, simulations, behaviours and
  runs of the session, from which every tab takes its inputs.
```
